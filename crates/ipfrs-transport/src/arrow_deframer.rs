//! Arrow IPC stream deframer for TensorSwap.
//!
//! When receiving Arrow IPC data over TensorSwap, chunks arrive in arbitrary
//! sizes. This module reassembles them into complete Arrow IPC messages using
//! the standard frame format:
//!
//! ```text
//! [ 4 bytes continuation marker 0xFF_FF_FF_FF ]
//! [ 4 bytes metadata length (LE u32) ]
//! [ metadata_length bytes flatbuffer ]
//! [ 8 bytes body length (LE u64) ]  ← only present when metadata_length > 0
//! [ body_length bytes body data ]
//! ```
//!
//! An EOS marker is signalled when metadata_length == 0x00_00_00_00.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The type of an Arrow IPC frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrowFrameType {
    /// Schema message (metadata only, body is empty).
    Schema,
    /// Record batch message.
    RecordBatch,
    /// Dictionary batch message.
    DictionaryBatch,
    /// End-of-stream marker (metadata_length == 0).
    EosMarker,
}

/// A fully reassembled Arrow IPC frame.
#[derive(Debug, Clone)]
pub struct ArrowFrame {
    /// Serialized Arrow flatbuffer metadata.
    pub metadata: Vec<u8>,
    /// Record batch body (may be empty for schema / EOS messages).
    pub body: Vec<u8>,
    /// Logical frame type.
    pub frame_type: ArrowFrameType,
    /// Monotonically increasing frame counter within the stream (starts at 0).
    pub sequence: u64,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors returned by [`ArrowStreamDeframer::push`].
#[derive(Debug, Error)]
pub enum DeframerError {
    /// The 4-byte continuation marker was not `0xFF_FF_FF_FF`.
    #[error("invalid continuation marker: expected FF FF FF FF, got {got:02X?}")]
    InvalidContinuationMarker { got: [u8; 4] },

    /// The metadata length field exceeds the configured maximum.
    #[error("metadata too large: {size} bytes exceeds maximum {max} bytes")]
    MetadataTooLarge { size: u32, max: u32 },

    /// The body length field exceeds the configured maximum.
    #[error("body too large: {size} bytes exceeds maximum {max} bytes")]
    BodyTooLarge { size: u64, max: u64 },

    /// The input stream ended before a frame was complete.
    #[error("unexpected end of input while assembling frame")]
    UnexpectedEof,
}

// ---------------------------------------------------------------------------
// Internal state machine
// ---------------------------------------------------------------------------

const CONTINUATION_MARKER: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xFF];

/// Internal state of the deframer.
///
/// Each variant represents the portion of the frame format that still needs
/// to be consumed from the input stream.
enum DeframerState {
    /// Waiting for the 4-byte continuation marker `0xFF_FF_FF_FF`.
    WaitingForContinuation,

    /// Reading the 4-byte metadata length field (little-endian u32).
    ReadingMetadataLen { buf: [u8; 4], read: usize },

    /// Reading `len` bytes of flatbuffer metadata.
    ReadingMetadata { len: u32, buf: Vec<u8>, read: usize },

    /// Reading the 8-byte body-length header followed by the body.
    ///
    /// Phase 1 (`body_len == BODY_LEN_PENDING`): reading the 8-byte header.
    ///   In this phase `buf` holds the header bytes read so far and `read`
    ///   counts how many of those 8 bytes have been consumed.
    ///
    /// Phase 2 (`body_len` is the actual value): reading the body bytes.
    ///   `buf` holds the body bytes read so far and `read` counts how many.
    ReadingBody {
        metadata: Vec<u8>,
        body_len: u64,
        buf: Vec<u8>,
        read: usize,
    },
}

/// Sentinel value used in `ReadingBody.body_len` to signal that we are still
/// reading the 8-byte body-length header.
const BODY_LEN_PENDING: u64 = u64::MAX;

impl std::fmt::Debug for DeframerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeframerState::WaitingForContinuation => write!(f, "WaitingForContinuation"),
            DeframerState::ReadingMetadataLen { read, .. } => {
                write!(f, "ReadingMetadataLen(read={read})")
            }
            DeframerState::ReadingMetadata { len, read, .. } => {
                write!(f, "ReadingMetadata(len={len}, read={read})")
            }
            DeframerState::ReadingBody { body_len, read, .. } => {
                write!(f, "ReadingBody(body_len={body_len}, read={read})")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Atomically-updated statistics for an [`ArrowStreamDeframer`].
#[derive(Debug, Default)]
pub struct DeframerStats {
    /// Total bytes fed via [`ArrowStreamDeframer::push`].
    pub total_bytes_pushed: AtomicU64,
    /// Total fully-assembled frames returned.
    pub total_frames_complete: AtomicU64,
    /// Total errors encountered.
    pub total_errors: AtomicU64,
    /// Total times [`ArrowStreamDeframer::reset`] was called explicitly.
    pub total_resets: AtomicU64,
}

/// A snapshot of [`DeframerStats`] with plain `u64` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeframerStatsSnapshot {
    pub total_bytes_pushed: u64,
    pub total_frames_complete: u64,
    pub total_errors: u64,
    pub total_resets: u64,
}

impl DeframerStats {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Take a consistent (relaxed) snapshot of all counters.
    pub fn snapshot(&self) -> DeframerStatsSnapshot {
        DeframerStatsSnapshot {
            total_bytes_pushed: self.total_bytes_pushed.load(Ordering::Relaxed),
            total_frames_complete: self.total_frames_complete.load(Ordering::Relaxed),
            total_errors: self.total_errors.load(Ordering::Relaxed),
            total_resets: self.total_resets.load(Ordering::Relaxed),
        }
    }
}

// ---------------------------------------------------------------------------
// Deframer
// ---------------------------------------------------------------------------

/// Default maximum metadata size: 64 MiB.
const DEFAULT_MAX_METADATA_BYTES: u32 = 64 * 1024 * 1024;

/// Default maximum body size: 2 GiB.
const DEFAULT_MAX_BODY_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Reassembles Arrow IPC stream chunks into complete [`ArrowFrame`]s.
///
/// # Frame format (simplified)
///
/// ```text
/// 4 bytes   continuation marker  (must be 0xFF_FF_FF_FF)
/// 4 bytes   metadata length LE u32  (0 → EOS)
/// N bytes   metadata flatbuffer
/// 8 bytes   body length LE u64      (only when metadata_length > 0)
/// M bytes   body
/// ```
#[derive(Debug)]
pub struct ArrowStreamDeframer {
    /// Current parser state.
    state: DeframerState,
    /// Continuation marker bytes accumulated so far (for partial reads).
    cont_buf: [u8; 4],
    /// How many continuation marker bytes have been read so far.
    cont_read: usize,
    /// Frame sequence counter.
    sequence: u64,
    /// Maximum metadata size in bytes.
    max_metadata_bytes: u32,
    /// Maximum body size in bytes.
    max_body_bytes: u64,
    /// Shared stats handle.
    stats: Arc<DeframerStats>,
}

impl Default for ArrowStreamDeframer {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrowStreamDeframer {
    /// Create a new deframer with default limits (64 MiB metadata, 2 GiB body).
    pub fn new() -> Self {
        Self {
            state: DeframerState::WaitingForContinuation,
            cont_buf: [0u8; 4],
            cont_read: 0,
            sequence: 0,
            max_metadata_bytes: DEFAULT_MAX_METADATA_BYTES,
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            stats: DeframerStats::new(),
        }
    }

    /// Create a deframer with explicit limits.
    pub fn with_limits(max_metadata_bytes: u32, max_body_bytes: u64) -> Self {
        Self {
            state: DeframerState::WaitingForContinuation,
            cont_buf: [0u8; 4],
            cont_read: 0,
            sequence: 0,
            max_metadata_bytes,
            max_body_bytes,
            stats: DeframerStats::new(),
        }
    }

    /// Returns `true` when the deframer is idle (waiting for the next frame's
    /// continuation marker), i.e., no partial frame is buffered.
    pub fn is_idle(&self) -> bool {
        matches!(self.state, DeframerState::WaitingForContinuation) && self.cont_read == 0
    }

    /// Reset the deframer to the idle state.  Increments the reset counter.
    pub fn reset(&mut self) {
        self.state = DeframerState::WaitingForContinuation;
        self.cont_buf = [0u8; 4];
        self.cont_read = 0;
        self.stats.total_resets.fetch_add(1, Ordering::Relaxed);
    }

    /// Return a clone of the shared statistics handle.
    pub fn stats(&self) -> Arc<DeframerStats> {
        Arc::clone(&self.stats)
    }

    /// Feed `data` into the deframer.  Returns all complete frames assembled
    /// during this call, or a [`DeframerError`] if the stream is malformed.
    ///
    /// On error the deframer is automatically reset so the caller may attempt
    /// to continue with later data (e.g., after re-synchronisation).
    pub fn push(&mut self, data: &[u8]) -> Result<Vec<ArrowFrame>, DeframerError> {
        self.stats
            .total_bytes_pushed
            .fetch_add(data.len() as u64, Ordering::Relaxed);

        let mut cursor = 0usize;
        let mut frames = Vec::new();

        while cursor < data.len() {
            if let Some(frame) = self.step(data, &mut cursor)? {
                self.stats
                    .total_frames_complete
                    .fetch_add(1, Ordering::Relaxed);
                frames.push(frame);
            }
        }

        Ok(frames)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Advance the state machine by consuming bytes from `data[*cursor..]`.
    /// Returns `Some(frame)` when a frame is complete, `None` when more data
    /// is needed or when a state transition happened without producing output.
    fn step(
        &mut self,
        data: &[u8],
        cursor: &mut usize,
    ) -> Result<Option<ArrowFrame>, DeframerError> {
        match &self.state {
            // ------------------------------------------------------------------
            // 1. Waiting for the 4-byte continuation marker
            // ------------------------------------------------------------------
            DeframerState::WaitingForContinuation => {
                // Accumulate bytes into cont_buf until we have 4.
                while self.cont_read < 4 && *cursor < data.len() {
                    self.cont_buf[self.cont_read] = data[*cursor];
                    *cursor += 1;
                    self.cont_read += 1;
                }

                if self.cont_read < 4 {
                    // Still waiting for more bytes.
                    return Ok(None);
                }

                // Full marker received — validate.
                let got = self.cont_buf;
                self.cont_read = 0;
                self.cont_buf = [0u8; 4];

                if got != CONTINUATION_MARKER {
                    self.stats.total_errors.fetch_add(1, Ordering::Relaxed);
                    self.state = DeframerState::WaitingForContinuation;
                    return Err(DeframerError::InvalidContinuationMarker { got });
                }

                // Transition: start reading the 4-byte metadata length.
                self.state = DeframerState::ReadingMetadataLen {
                    buf: [0u8; 4],
                    read: 0,
                };
                Ok(None)
            }

            // ------------------------------------------------------------------
            // 2. Reading the 4-byte metadata length (LE u32)
            // ------------------------------------------------------------------
            DeframerState::ReadingMetadataLen { .. } => {
                // Extract fields by replacing state with sentinel then restoring.
                let (mut buf, mut read) = if let DeframerState::ReadingMetadataLen { buf, read } =
                    std::mem::replace(&mut self.state, DeframerState::WaitingForContinuation)
                {
                    (buf, read)
                } else {
                    unreachable!()
                };

                while read < 4 && *cursor < data.len() {
                    buf[read] = data[*cursor];
                    *cursor += 1;
                    read += 1;
                }

                if read < 4 {
                    self.state = DeframerState::ReadingMetadataLen { buf, read };
                    return Ok(None);
                }

                let meta_len = u32::from_le_bytes(buf);

                // EOS: meta_len == 0.
                if meta_len == 0 {
                    let seq = self.sequence;
                    self.sequence += 1;
                    self.state = DeframerState::WaitingForContinuation;
                    return Ok(Some(ArrowFrame {
                        metadata: Vec::new(),
                        body: Vec::new(),
                        frame_type: ArrowFrameType::EosMarker,
                        sequence: seq,
                    }));
                }

                // Enforce metadata size limit.
                if meta_len > self.max_metadata_bytes {
                    self.stats.total_errors.fetch_add(1, Ordering::Relaxed);
                    self.state = DeframerState::WaitingForContinuation;
                    return Err(DeframerError::MetadataTooLarge {
                        size: meta_len,
                        max: self.max_metadata_bytes,
                    });
                }

                self.state = DeframerState::ReadingMetadata {
                    len: meta_len,
                    buf: vec![0u8; meta_len as usize],
                    read: 0,
                };
                Ok(None)
            }

            // ------------------------------------------------------------------
            // 3. Reading `len` bytes of flatbuffer metadata
            // ------------------------------------------------------------------
            DeframerState::ReadingMetadata { .. } => {
                let (len, mut buf, mut read) =
                    if let DeframerState::ReadingMetadata { len, buf, read } =
                        std::mem::replace(&mut self.state, DeframerState::WaitingForContinuation)
                    {
                        (len, buf, read)
                    } else {
                        unreachable!()
                    };

                let needed = len as usize - read;
                let available = data.len() - *cursor;
                let take = needed.min(available);

                buf[read..read + take].copy_from_slice(&data[*cursor..*cursor + take]);
                *cursor += take;
                read += take;

                if read < len as usize {
                    self.state = DeframerState::ReadingMetadata { len, buf, read };
                    return Ok(None);
                }

                // Metadata complete — begin reading the 8-byte body-length header.
                self.state = DeframerState::ReadingBody {
                    metadata: buf,
                    body_len: BODY_LEN_PENDING,
                    buf: Vec::with_capacity(8),
                    read: 0,
                };
                Ok(None)
            }

            // ------------------------------------------------------------------
            // 4. Reading 8-byte body-length header then body bytes
            // ------------------------------------------------------------------
            DeframerState::ReadingBody { .. } => {
                let (metadata, mut body_len, mut buf, mut read) =
                    if let DeframerState::ReadingBody {
                        metadata,
                        body_len,
                        buf,
                        read,
                    } = std::mem::replace(&mut self.state, DeframerState::WaitingForContinuation)
                    {
                        (metadata, body_len, buf, read)
                    } else {
                        unreachable!()
                    };

                // Sub-phase 1: read the 8-byte body-length header.
                if body_len == BODY_LEN_PENDING {
                    while read < 8 && *cursor < data.len() {
                        buf.push(data[*cursor]);
                        *cursor += 1;
                        read += 1;
                    }

                    if read < 8 {
                        self.state = DeframerState::ReadingBody {
                            metadata,
                            body_len: BODY_LEN_PENDING,
                            buf,
                            read,
                        };
                        return Ok(None);
                    }

                    // Parse the 8 header bytes.
                    let len_arr: [u8; 8] =
                        buf[..8].try_into().expect("slice is known to be 8 bytes");
                    body_len = u64::from_le_bytes(len_arr);

                    // Enforce body size limit.
                    if body_len > self.max_body_bytes {
                        self.stats.total_errors.fetch_add(1, Ordering::Relaxed);
                        self.state = DeframerState::WaitingForContinuation;
                        return Err(DeframerError::BodyTooLarge {
                            size: body_len,
                            max: self.max_body_bytes,
                        });
                    }

                    // Prepare body buffer and reset read counter for body bytes.
                    buf = vec![0u8; body_len as usize];
                    read = 0;
                }

                // Sub-phase 2: read body_len bytes of body data.
                if body_len == 0 {
                    let seq = self.sequence;
                    self.sequence += 1;
                    self.state = DeframerState::WaitingForContinuation;
                    return Ok(Some(ArrowFrame {
                        metadata,
                        body: Vec::new(),
                        frame_type: ArrowFrameType::RecordBatch,
                        sequence: seq,
                    }));
                }

                let needed = body_len as usize - read;
                let available = data.len() - *cursor;
                let take = needed.min(available);

                buf[read..read + take].copy_from_slice(&data[*cursor..*cursor + take]);
                *cursor += take;
                read += take;

                if (read as u64) < body_len {
                    self.state = DeframerState::ReadingBody {
                        metadata,
                        body_len,
                        buf,
                        read,
                    };
                    return Ok(None);
                }

                // Body complete — emit frame.
                let seq = self.sequence;
                self.sequence += 1;
                self.state = DeframerState::WaitingForContinuation;
                Ok(Some(ArrowFrame {
                    metadata,
                    body: buf,
                    frame_type: ArrowFrameType::RecordBatch,
                    sequence: seq,
                }))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test frame builders
// ---------------------------------------------------------------------------

/// Build a raw Arrow IPC frame for testing.
///
/// Format: `[FF FF FF FF][meta_len_le4][metadata][body_len_le8][body]`
pub fn build_test_frame(metadata: &[u8], body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&CONTINUATION_MARKER);
    out.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
    out.extend_from_slice(metadata);
    out.extend_from_slice(&(body.len() as u64).to_le_bytes());
    out.extend_from_slice(body);
    out
}

/// Build a raw Arrow IPC EOS frame for testing.
pub fn build_test_eos() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&CONTINUATION_MARKER);
    out.extend_from_slice(&0u32.to_le_bytes());
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // T-01: Single complete frame parsed correctly
    // -----------------------------------------------------------------------
    #[test]
    fn test_single_complete_frame() {
        let meta = b"flatbuf-meta";
        let body = b"body-data-here";
        let raw = build_test_frame(meta, body);

        let mut df = ArrowStreamDeframer::new();
        let frames = df.push(&raw).expect("should succeed");

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].metadata, meta);
        assert_eq!(frames[0].body, body);
        assert_eq!(frames[0].sequence, 0);
        assert!(df.is_idle());
    }

    // -----------------------------------------------------------------------
    // T-02: Frame split across two pushes
    // -----------------------------------------------------------------------
    #[test]
    fn test_frame_split_across_two_pushes() {
        let meta = b"schema-meta-bytes";
        let body = b"record-body";
        let raw = build_test_frame(meta, body);

        let split = raw.len() / 2;
        let mut df = ArrowStreamDeframer::new();

        let frames_first = df.push(&raw[..split]).expect("first push ok");
        assert!(frames_first.is_empty(), "no complete frames yet");
        assert!(!df.is_idle());

        let frames_second = df.push(&raw[split..]).expect("second push ok");
        assert_eq!(frames_second.len(), 1);
        assert_eq!(frames_second[0].metadata, meta);
        assert_eq!(frames_second[0].body, body);
    }

    // -----------------------------------------------------------------------
    // T-03: Multiple frames in one push
    // -----------------------------------------------------------------------
    #[test]
    fn test_multiple_frames_in_one_push() {
        let mut raw = Vec::new();
        for i in 0u8..3 {
            let meta = vec![i; 8];
            let body = vec![i + 10; 16];
            raw.extend_from_slice(&build_test_frame(&meta, &body));
        }

        let mut df = ArrowStreamDeframer::new();
        let frames = df.push(&raw).expect("ok");
        assert_eq!(frames.len(), 3);
        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(frame.sequence, i as u64);
            assert_eq!(frame.metadata, vec![i as u8; 8]);
            assert_eq!(frame.body, vec![i as u8 + 10; 16]);
        }
    }

    // -----------------------------------------------------------------------
    // T-04: EOS marker produces EosMarker frame
    // -----------------------------------------------------------------------
    #[test]
    fn test_eos_marker_produces_eos_frame() {
        let eos = build_test_eos();
        let mut df = ArrowStreamDeframer::new();
        let frames = df.push(&eos).expect("ok");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].frame_type, ArrowFrameType::EosMarker);
        assert!(frames[0].metadata.is_empty());
        assert!(frames[0].body.is_empty());
    }

    // -----------------------------------------------------------------------
    // T-05: Invalid continuation marker returns error
    // -----------------------------------------------------------------------
    #[test]
    fn test_invalid_continuation_marker() {
        let bad: Vec<u8> = vec![0x00, 0x01, 0x02, 0x03, 0xFF, 0xFF, 0xFF, 0xFF];
        let mut df = ArrowStreamDeframer::new();
        let result = df.push(&bad);
        assert!(result.is_err());
        match result.unwrap_err() {
            DeframerError::InvalidContinuationMarker { got } => {
                assert_eq!(got, [0x00, 0x01, 0x02, 0x03]);
            }
            other => panic!("unexpected error: {other}"),
        }
        // After error the deframer should be idle again.
        assert!(df.is_idle());
    }

    // -----------------------------------------------------------------------
    // T-06: Frame with empty body (body_length == 0)
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_body_frame() {
        let meta = b"schema-only";
        let raw = build_test_frame(meta, b"");

        let mut df = ArrowStreamDeframer::new();
        let frames = df.push(&raw).expect("ok");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].metadata, meta);
        assert!(frames[0].body.is_empty());
    }

    // -----------------------------------------------------------------------
    // T-07: is_idle state tracking
    // -----------------------------------------------------------------------
    #[test]
    fn test_is_idle_state_tracking() {
        let meta = b"m";
        let body = b"b";
        let raw = build_test_frame(meta, body);

        let mut df = ArrowStreamDeframer::new();
        assert!(df.is_idle(), "idle before any data");

        // Feed partial: just the continuation marker
        df.push(&raw[..4]).expect("ok");
        assert!(!df.is_idle(), "not idle mid-frame");

        // Feed the rest
        df.push(&raw[4..]).expect("ok");
        assert!(df.is_idle(), "idle after frame completed");
    }

    // -----------------------------------------------------------------------
    // T-08: reset clears state
    // -----------------------------------------------------------------------
    #[test]
    fn test_reset_clears_state() {
        let meta = b"partial";
        let body = b"data";
        let raw = build_test_frame(meta, body);

        let mut df = ArrowStreamDeframer::new();
        df.push(&raw[..4]).expect("ok");
        assert!(!df.is_idle());

        df.reset();
        assert!(df.is_idle());

        // After reset, a fresh frame should be parseable.
        let frames = df.push(&raw).expect("ok after reset");
        assert_eq!(frames.len(), 1);
    }

    // -----------------------------------------------------------------------
    // T-09: Stats accumulation
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_accumulation() {
        let raw1 = build_test_frame(b"meta1", b"body1");
        let raw2 = build_test_frame(b"meta2", b"body2");
        let combined = [raw1.clone(), raw2.clone()].concat();

        let mut df = ArrowStreamDeframer::new();
        df.push(&combined).expect("ok");

        let snap = df.stats().snapshot();
        assert_eq!(snap.total_bytes_pushed, combined.len() as u64);
        assert_eq!(snap.total_frames_complete, 2);
        assert_eq!(snap.total_errors, 0);
        assert_eq!(snap.total_resets, 0);
    }

    // -----------------------------------------------------------------------
    // T-10: Large frame within limits
    // -----------------------------------------------------------------------
    #[test]
    fn test_large_frame_within_limits() {
        let meta = vec![0xABu8; 1024];
        let body = vec![0xCDu8; 65536];
        let raw = build_test_frame(&meta, &body);

        let mut df = ArrowStreamDeframer::new();
        let frames = df.push(&raw).expect("ok");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].metadata.len(), 1024);
        assert_eq!(frames[0].body.len(), 65536);
    }

    // -----------------------------------------------------------------------
    // T-11: Sequence counter increments
    // -----------------------------------------------------------------------
    #[test]
    fn test_sequence_counter_increments() {
        let mut raw = Vec::new();
        for _ in 0..5 {
            raw.extend_from_slice(&build_test_frame(b"meta", b"body"));
        }

        let mut df = ArrowStreamDeframer::new();
        let frames = df.push(&raw).expect("ok");
        assert_eq!(frames.len(), 5);
        for (expected_seq, frame) in frames.iter().enumerate() {
            assert_eq!(frame.sequence, expected_seq as u64);
        }
    }

    // -----------------------------------------------------------------------
    // T-12: MetadataTooLarge error
    // -----------------------------------------------------------------------
    #[test]
    fn test_metadata_too_large_error() {
        // Craft a frame that claims a huge metadata length.
        let mut raw = Vec::new();
        raw.extend_from_slice(&CONTINUATION_MARKER);
        // Claim metadata_length = 65 MB (above default 64 MB limit).
        let huge: u32 = 65 * 1024 * 1024;
        raw.extend_from_slice(&huge.to_le_bytes());

        let mut df = ArrowStreamDeframer::new();
        let result = df.push(&raw);
        assert!(result.is_err());
        match result.unwrap_err() {
            DeframerError::MetadataTooLarge { size, max } => {
                assert_eq!(size, huge);
                assert_eq!(max, DEFAULT_MAX_METADATA_BYTES);
            }
            other => panic!("unexpected: {other}"),
        }

        let snap = df.stats().snapshot();
        assert_eq!(snap.total_errors, 1);
    }

    // -----------------------------------------------------------------------
    // T-13: BodyTooLarge error
    // -----------------------------------------------------------------------
    #[test]
    fn test_body_too_large_error() {
        let meta = b"valid-meta";
        let mut raw = Vec::new();
        raw.extend_from_slice(&CONTINUATION_MARKER);
        raw.extend_from_slice(&(meta.len() as u32).to_le_bytes());
        raw.extend_from_slice(meta);
        // Claim body_length = 3 GB (above default 2 GB limit).
        let huge_body: u64 = 3 * 1024 * 1024 * 1024;
        raw.extend_from_slice(&huge_body.to_le_bytes());

        let mut df = ArrowStreamDeframer::new();
        let result = df.push(&raw);
        assert!(result.is_err());
        match result.unwrap_err() {
            DeframerError::BodyTooLarge { size, max } => {
                assert_eq!(size, huge_body);
                assert_eq!(max, DEFAULT_MAX_BODY_BYTES);
            }
            other => panic!("unexpected: {other}"),
        }

        let snap = df.stats().snapshot();
        assert_eq!(snap.total_errors, 1);
    }

    // -----------------------------------------------------------------------
    // T-14: Frame followed immediately by EOS in one push
    // -----------------------------------------------------------------------
    #[test]
    fn test_frame_followed_by_eos() {
        let mut raw = Vec::new();
        raw.extend_from_slice(&build_test_frame(b"schema", b""));
        raw.extend_from_slice(&build_test_eos());

        let mut df = ArrowStreamDeframer::new();
        let frames = df.push(&raw).expect("ok");
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].frame_type, ArrowFrameType::RecordBatch);
        assert_eq!(frames[1].frame_type, ArrowFrameType::EosMarker);
        assert_eq!(frames[0].sequence, 0);
        assert_eq!(frames[1].sequence, 1);
        assert!(df.is_idle());
    }

    // -----------------------------------------------------------------------
    // T-15: Byte-at-a-time feeding (extreme fragmentation)
    // -----------------------------------------------------------------------
    #[test]
    fn test_byte_at_a_time_feeding() {
        let meta = b"granular-meta";
        let body = b"granular-body";
        let raw = build_test_frame(meta, body);

        let mut df = ArrowStreamDeframer::new();
        let mut all_frames = Vec::new();
        for byte in &raw {
            let mut frames = df.push(std::slice::from_ref(byte)).expect("ok");
            all_frames.append(&mut frames);
        }
        assert_eq!(all_frames.len(), 1);
        assert_eq!(all_frames[0].metadata, meta);
        assert_eq!(all_frames[0].body, body);
        assert!(df.is_idle());
    }

    // -----------------------------------------------------------------------
    // T-16: Reset increments reset counter
    // -----------------------------------------------------------------------
    #[test]
    fn test_reset_increments_counter() {
        let mut df = ArrowStreamDeframer::new();
        df.reset();
        df.reset();
        df.reset();
        let snap = df.stats().snapshot();
        assert_eq!(snap.total_resets, 3);
    }

    // -----------------------------------------------------------------------
    // T-17: Stats error counter increments on bad marker
    // -----------------------------------------------------------------------
    #[test]
    fn test_error_counter_on_bad_marker() {
        let bad = [0xDE, 0xAD, 0xBE, 0xEF];
        let mut df = ArrowStreamDeframer::new();
        let _ = df.push(&bad);
        let snap = df.stats().snapshot();
        assert_eq!(snap.total_errors, 1);
    }
}
