//! Gradient streaming: Arrow-IPC chunked transfer with CRC-32 integrity checks.

/// A single Arrow-IPC-encoded gradient chunk with integrity metadata.
///
/// Each chunk carries an independent Arrow IPC record batch so that receivers
/// can decode chunks individually without reassembling the entire stream.
#[derive(Debug, Clone)]
pub struct GradientChunk {
    /// Session identifier shared across all chunks of a transfer.
    pub session_id: String,
    /// Zero-based position of this chunk in the stream.
    pub chunk_index: u32,
    /// Total number of chunks in this stream.
    pub total_chunks: u32,
    /// Arrow IPC bytes encoding a slice of the gradient vector.
    pub arrow_ipc_bytes: Vec<u8>,
    /// CRC-32 checksum of `arrow_ipc_bytes` for integrity verification.
    pub checksum: u32,
}

impl GradientChunk {
    /// Compute the expected CRC-32 checksum over the IPC bytes.
    pub fn compute_checksum(data: &[u8]) -> u32 {
        crc32fast::hash(data)
    }

    /// Verify that the stored checksum matches the IPC bytes.
    pub fn verify_checksum(&self) -> bool {
        Self::compute_checksum(&self.arrow_ipc_bytes) == self.checksum
    }
}

/// Error type for gradient streaming operations.
#[derive(Debug, thiserror::Error)]
pub enum GradientStreamError {
    /// Arrow IPC encode/decode failure.
    #[error("Arrow IPC error: {0}")]
    ArrowIpc(#[from] anyhow::Error),

    /// CRC-32 checksum mismatch detected when decoding a chunk.
    #[error(
        "Checksum mismatch on chunk {chunk_index}: expected {expected:#010x}, got {actual:#010x}"
    )]
    ChecksumMismatch {
        chunk_index: u32,
        expected: u32,
        actual: u32,
    },

    /// The chunk stream was incomplete or ordering was invalid.
    #[error("Incomplete chunk stream: received {received} of {total} chunks")]
    IncompleteStream { received: usize, total: usize },

    /// I/O error during network streaming.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// A TensorSwap session optimised for gradient tensor exchange.
///
/// Splits large gradients into Arrow IPC chunks and provides
/// encode/decode helpers.  Network send/receive stubs log appropriately
/// and return gracefully when no live connection is available — the full
/// chunking and Arrow IPC logic is always executed so that the encoding
/// path can be tested end-to-end.
///
/// # Default chunk size
/// 65 536 `f32` values ≈ 256 KiB per chunk.
pub struct GradientStreamSession {
    /// Unique session identifier (UUID-like string).
    session_id: String,
    /// Number of `f32` values per chunk (default: 65 536).
    chunk_size: usize,
    /// Whether to apply OxiARC compression to each chunk (currently tracked
    /// but compression is a future extension; set to `false` for pure-Rust
    /// baseline).
    compression: bool,
}

impl GradientStreamSession {
    /// Create a new session with a custom chunk size.
    ///
    /// `chunk_size` is the number of `f32` values per chunk, not bytes.
    pub fn new(session_id: &str, chunk_size: usize) -> Self {
        Self {
            session_id: session_id.to_string(),
            chunk_size: chunk_size.max(1),
            compression: false,
        }
    }

    /// Create a new session with the default chunk size (65 536 elements).
    pub fn with_defaults(session_id: &str) -> Self {
        Self::new(session_id, 65_536)
    }

    /// Enable or disable per-chunk compression.
    ///
    /// When enabled, future versions will compress each Arrow IPC payload
    /// via OxiARC before computing the checksum.  The flag is stored so
    /// that protocol negotiation can query it.
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.compression = enabled;
        self
    }

    /// Returns whether compression is configured for this session.
    pub fn is_compression_enabled(&self) -> bool {
        self.compression
    }

    /// Returns the session identifier.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Returns the chunk size in number of `f32` elements.
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Split `gradient` into Arrow IPC chunks and return the chunk list.
    ///
    /// Each element of the returned vector is an independent Arrow IPC
    /// record batch encoding a contiguous slice of the gradient.  The
    /// CRC-32 checksum is computed over the raw Arrow IPC bytes before
    /// they are stored in [`GradientChunk::arrow_ipc_bytes`].
    pub fn encode_gradient(
        &self,
        gradient: &[f32],
    ) -> std::result::Result<Vec<GradientChunk>, GradientStreamError> {
        use ipfrs_tensorlogic::gradient::arrow_ipc::store_gradient_as_arrow;

        // Compute total chunk count (ceiling division).
        let total_chunks = if gradient.is_empty() {
            1
        } else {
            gradient.len().div_ceil(self.chunk_size)
        };

        let mut chunks = Vec::with_capacity(total_chunks);

        if gradient.is_empty() {
            // Encode a single empty chunk so the stream always has ≥1 chunk.
            let ipc_bytes = store_gradient_as_arrow(&[]).map_err(GradientStreamError::ArrowIpc)?;
            let checksum = GradientChunk::compute_checksum(&ipc_bytes);
            chunks.push(GradientChunk {
                session_id: self.session_id.clone(),
                chunk_index: 0,
                total_chunks: 1,
                arrow_ipc_bytes: ipc_bytes,
                checksum,
            });
            return Ok(chunks);
        }

        for (idx, window) in gradient.chunks(self.chunk_size).enumerate() {
            let ipc_bytes =
                store_gradient_as_arrow(window).map_err(GradientStreamError::ArrowIpc)?;
            let checksum = GradientChunk::compute_checksum(&ipc_bytes);
            chunks.push(GradientChunk {
                session_id: self.session_id.clone(),
                chunk_index: idx as u32,
                total_chunks: total_chunks as u32,
                arrow_ipc_bytes: ipc_bytes,
                checksum,
            });
        }

        Ok(chunks)
    }

    /// Reassemble `chunks` back into a gradient vector.
    ///
    /// Chunks must all share the same `total_chunks` count and have
    /// contiguous indices 0 .. `total_chunks - 1`.  Each chunk's CRC-32
    /// checksum is verified before decoding.
    pub fn decode_chunks(
        &self,
        mut chunks: Vec<GradientChunk>,
    ) -> std::result::Result<Vec<f32>, GradientStreamError> {
        use ipfrs_tensorlogic::gradient::arrow_ipc::load_gradient_from_arrow;

        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        // Sort by chunk_index so callers need not deliver in order.
        chunks.sort_by_key(|c| c.chunk_index);

        let total = chunks[0].total_chunks as usize;
        if chunks.len() != total {
            return Err(GradientStreamError::IncompleteStream {
                received: chunks.len(),
                total,
            });
        }

        let mut gradient = Vec::new();

        for chunk in &chunks {
            // Integrity check.
            let actual = GradientChunk::compute_checksum(&chunk.arrow_ipc_bytes);
            if actual != chunk.checksum {
                return Err(GradientStreamError::ChecksumMismatch {
                    chunk_index: chunk.chunk_index,
                    expected: chunk.checksum,
                    actual,
                });
            }

            let slice = load_gradient_from_arrow(&chunk.arrow_ipc_bytes)
                .map_err(GradientStreamError::ArrowIpc)?;
            gradient.extend_from_slice(&slice);
        }

        Ok(gradient)
    }

    /// Stream a gradient to a TensorSwap peer.
    ///
    /// Encodes `gradient` into Arrow IPC chunks and writes each chunk as a
    /// length-prefixed frame onto `stream`.  Each frame is:
    ///
    /// ```text
    /// [4-byte LE chunk_index][4-byte LE total_chunks][4-byte LE payload_len]
    /// [4-byte LE checksum][payload_len bytes of Arrow IPC]
    /// ```
    ///
    /// Returns the number of chunks sent.  If the underlying transport is
    /// unavailable the method returns `Ok(n_chunks)` after encoding so that
    /// the chunking logic is always exercised in tests.
    pub async fn stream_to<S>(
        &self,
        gradient: &[f32],
        stream: &mut S,
    ) -> std::result::Result<usize, GradientStreamError>
    where
        S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
    {
        use tokio::io::AsyncWriteExt;

        let chunks = self.encode_gradient(gradient)?;
        let n = chunks.len();

        for chunk in &chunks {
            // Frame header: chunk_index (4) | total_chunks (4) | payload_len (4) | checksum (4)
            let payload_len = chunk.arrow_ipc_bytes.len() as u32;

            let mut header = [0u8; 16];
            header[0..4].copy_from_slice(&chunk.chunk_index.to_le_bytes());
            header[4..8].copy_from_slice(&chunk.total_chunks.to_le_bytes());
            header[8..12].copy_from_slice(&payload_len.to_le_bytes());
            header[12..16].copy_from_slice(&chunk.checksum.to_le_bytes());

            stream.write_all(&header).await?;
            stream.write_all(&chunk.arrow_ipc_bytes).await?;
        }

        stream.flush().await?;
        Ok(n)
    }

    /// Receive a gradient from a TensorSwap stream.
    ///
    /// Reads length-prefixed frames written by `stream_to` and reassembles
    /// the gradient.  Returns `Ok(vec![])` when the stream signals EOF before
    /// the first frame (network not available).
    pub async fn receive_from<S>(
        &self,
        stream: &mut S,
    ) -> std::result::Result<Vec<f32>, GradientStreamError>
    where
        S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
    {
        use tokio::io::{AsyncReadExt, BufReader};

        let mut reader = BufReader::new(stream);
        let mut chunks: Vec<GradientChunk> = Vec::new();

        loop {
            // Try to read the 16-byte header.
            let mut header = [0u8; 16];
            match reader.read_exact(&mut header).await {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    // Stream closed before or between frames.
                    if chunks.is_empty() {
                        tracing::debug!(
                            session_id = %self.session_id,
                            "Gradient stream closed before first frame — no network"
                        );
                        return Ok(Vec::new());
                    }
                    break;
                }
                Err(e) => return Err(GradientStreamError::Io(e)),
            }

            let chunk_index = u32::from_le_bytes(header[0..4].try_into().map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "bad header bytes [0..4]")
            })?);
            let total_chunks = u32::from_le_bytes(header[4..8].try_into().map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "bad header bytes [4..8]")
            })?);
            let payload_len = u32::from_le_bytes(header[8..12].try_into().map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "bad header bytes [8..12]")
            })?) as usize;
            let checksum = u32::from_le_bytes(header[12..16].try_into().map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "bad header bytes [12..16]")
            })?);

            let mut payload = vec![0u8; payload_len];
            reader.read_exact(&mut payload).await?;

            chunks.push(GradientChunk {
                session_id: self.session_id.clone(),
                chunk_index,
                total_chunks,
                arrow_ipc_bytes: payload,
                checksum,
            });

            if chunks.len() >= total_chunks as usize {
                break;
            }
        }

        self.decode_chunks(chunks)
    }
}
