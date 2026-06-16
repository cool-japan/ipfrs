//! Length-delimited message encoding/decoding for peer communication.
//!
//! Provides `PeerMessageCodec` which encodes messages with a 4-byte big-endian
//! length prefix and an optional CRC32 checksum for integrity verification.
//!
//! # Wire Format
//!
//! ```text
//! +-------------------+-------------------+--------------------+
//! | Length (4 bytes)   | Payload (N bytes) | CRC32 (4 bytes)    |
//! | big-endian u32     |                   | (optional)         |
//! +-------------------+-------------------+--------------------+
//! ```
//!
//! The length prefix encodes the payload size only (not including itself or the
//! checksum). When checksums are enabled, the CRC32 is computed over the payload
//! bytes and appended after the payload.

use std::fmt;

// ---------------------------------------------------------------------------
// CRC32 lookup table (IEEE polynomial 0xEDB88320, reflected)
// ---------------------------------------------------------------------------

const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut crc = i;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i as usize] = crc;
        i += 1;
    }
    table
};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by [`PeerMessageCodec`] encode/decode operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecError {
    /// The supplied buffer is too small to contain a valid message.
    BufferTooSmall,
    /// The payload exceeds the configured maximum message size.
    MessageTooLarge,
    /// The length prefix contains an invalid or inconsistent value.
    InvalidLength,
    /// The CRC32 checksum of the payload does not match the stored checksum.
    ChecksumMismatch,
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferTooSmall => write!(f, "buffer too small"),
            Self::MessageTooLarge => write!(f, "message too large"),
            Self::InvalidLength => write!(f, "invalid length prefix"),
            Self::ChecksumMismatch => write!(f, "checksum mismatch"),
        }
    }
}

impl std::error::Error for CodecError {}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`PeerMessageCodec`].
#[derive(Debug, Clone)]
pub struct CodecConfig {
    /// Maximum allowed payload size in bytes (default: 16 MiB).
    pub max_message_size: usize,
    /// Whether to append/verify a CRC32 checksum (default: `true`).
    pub use_checksum: bool,
    /// Number of bytes used for the length prefix (always 4).
    pub length_prefix_bytes: usize,
}

impl Default for CodecConfig {
    fn default() -> Self {
        Self {
            max_message_size: 16_777_216, // 16 MiB
            use_checksum: true,
            length_prefix_bytes: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Encoded message metadata
// ---------------------------------------------------------------------------

/// Metadata about an encoded message.
#[derive(Debug, Clone)]
pub struct EncodedMessage {
    /// The full wire-format bytes (length prefix + payload + optional checksum).
    pub data: Vec<u8>,
    /// The size of the original payload before encoding.
    pub original_size: usize,
    /// The CRC32 checksum of the payload, if checksums are enabled.
    pub checksum: Option<u32>,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Cumulative codec statistics.
#[derive(Debug, Clone)]
pub struct CodecStats {
    /// Total number of messages successfully encoded.
    pub messages_encoded: u64,
    /// Total number of messages successfully decoded.
    pub messages_decoded: u64,
    /// Total payload bytes encoded (before framing overhead).
    pub bytes_encoded: u64,
    /// Total payload bytes decoded.
    pub bytes_decoded: u64,
}

// ---------------------------------------------------------------------------
// PeerMessageCodec
// ---------------------------------------------------------------------------

/// Length-delimited message codec with optional CRC32 integrity checking.
///
/// # Example
///
/// ```rust
/// use ipfrs_network::message_codec::{PeerMessageCodec, CodecConfig};
///
/// let codec = PeerMessageCodec::new(CodecConfig::default());
/// let wire = codec.encode(b"hello").expect("encode");
/// let payload = codec.decode(&wire).expect("decode");
/// assert_eq!(payload, b"hello");
/// ```
pub struct PeerMessageCodec {
    config: CodecConfig,
    messages_encoded: std::cell::Cell<u64>,
    messages_decoded: std::cell::Cell<u64>,
    bytes_encoded: std::cell::Cell<u64>,
    bytes_decoded: std::cell::Cell<u64>,
}

impl fmt::Debug for PeerMessageCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PeerMessageCodec")
            .field("config", &self.config)
            .field("messages_encoded", &self.messages_encoded.get())
            .field("messages_decoded", &self.messages_decoded.get())
            .finish()
    }
}

impl PeerMessageCodec {
    /// Create a new codec with the given configuration.
    pub fn new(config: CodecConfig) -> Self {
        Self {
            config,
            messages_encoded: std::cell::Cell::new(0),
            messages_decoded: std::cell::Cell::new(0),
            bytes_encoded: std::cell::Cell::new(0),
            bytes_decoded: std::cell::Cell::new(0),
        }
    }

    /// Encode a payload into wire format.
    ///
    /// Returns the complete frame: `[4-byte length BE][payload][optional 4-byte CRC32]`.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::MessageTooLarge`] if `payload.len()` exceeds
    /// `config.max_message_size`.
    pub fn encode(&self, payload: &[u8]) -> Result<Vec<u8>, CodecError> {
        if payload.len() > self.config.max_message_size {
            return Err(CodecError::MessageTooLarge);
        }

        let total = self.estimate_encoded_size(payload.len());
        let mut buf = Vec::with_capacity(total);

        // Length prefix (payload size only, big-endian u32).
        let len_u32 = payload.len() as u32;
        buf.extend_from_slice(&len_u32.to_be_bytes());

        // Payload.
        buf.extend_from_slice(payload);

        // Optional CRC32 checksum.
        if self.config.use_checksum {
            let crc = Self::crc32(payload);
            buf.extend_from_slice(&crc.to_be_bytes());
        }

        // Update stats.
        self.messages_encoded.set(self.messages_encoded.get() + 1);
        self.bytes_encoded
            .set(self.bytes_encoded.get() + payload.len() as u64);

        Ok(buf)
    }

    /// Encode a payload and return an [`EncodedMessage`] with metadata.
    pub fn encode_with_metadata(&self, payload: &[u8]) -> Result<EncodedMessage, CodecError> {
        if payload.len() > self.config.max_message_size {
            return Err(CodecError::MessageTooLarge);
        }

        let checksum = if self.config.use_checksum {
            Some(Self::crc32(payload))
        } else {
            None
        };

        let data = self.encode(payload)?;
        // `encode` already updated stats, but we called it internally so
        // we need to undo the double-count. Alternatively, we could inline
        // the logic, but reusing `encode` is cleaner. Undo stats bump:
        self.messages_encoded.set(self.messages_encoded.get() - 1);
        self.bytes_encoded
            .set(self.bytes_encoded.get() - payload.len() as u64);

        // Now bump once for this call.
        self.messages_encoded.set(self.messages_encoded.get() + 1);
        self.bytes_encoded
            .set(self.bytes_encoded.get() + payload.len() as u64);

        Ok(EncodedMessage {
            data,
            original_size: payload.len(),
            checksum,
        })
    }

    /// Decode a wire-format frame and return the payload bytes.
    ///
    /// # Errors
    ///
    /// - [`CodecError::BufferTooSmall`] if `data` is shorter than the length prefix.
    /// - [`CodecError::InvalidLength`] if the frame is truncated or the length
    ///   prefix is inconsistent with the buffer size.
    /// - [`CodecError::MessageTooLarge`] if the decoded length exceeds the
    ///   configured maximum.
    /// - [`CodecError::ChecksumMismatch`] if checksums are enabled and the
    ///   stored checksum doesn't match the computed one.
    pub fn decode(&self, data: &[u8]) -> Result<Vec<u8>, CodecError> {
        if data.len() < 4 {
            return Err(CodecError::BufferTooSmall);
        }

        let payload_len = self.decode_length(data)?;

        if payload_len > self.config.max_message_size {
            return Err(CodecError::MessageTooLarge);
        }

        let checksum_size = if self.config.use_checksum { 4 } else { 0 };
        let expected_total = 4 + payload_len + checksum_size;

        if data.len() < expected_total {
            return Err(CodecError::InvalidLength);
        }

        let payload = &data[4..4 + payload_len];

        if self.config.use_checksum {
            let stored_bytes: [u8; 4] = data[4 + payload_len..4 + payload_len + 4]
                .try_into()
                .map_err(|_| CodecError::InvalidLength)?;
            let stored_crc = u32::from_be_bytes(stored_bytes);
            let computed_crc = Self::crc32(payload);
            if stored_crc != computed_crc {
                return Err(CodecError::ChecksumMismatch);
            }
        }

        self.messages_decoded.set(self.messages_decoded.get() + 1);
        self.bytes_decoded
            .set(self.bytes_decoded.get() + payload_len as u64);

        Ok(payload.to_vec())
    }

    /// Compute a CRC32 checksum (IEEE / ISO 3309) of the given data.
    pub fn crc32(data: &[u8]) -> u32 {
        let mut crc: u32 = 0xFFFF_FFFF;
        for &byte in data {
            let index = ((crc ^ u32::from(byte)) & 0xFF) as usize;
            crc = (crc >> 8) ^ CRC32_TABLE[index];
        }
        crc ^ 0xFFFF_FFFF
    }

    /// Estimate the total encoded size for a payload of the given length.
    ///
    /// Returns `4 + payload_size + (4 if checksum enabled)`.
    pub fn estimate_encoded_size(&self, payload_size: usize) -> usize {
        let checksum_overhead = if self.config.use_checksum { 4 } else { 0 };
        4 + payload_size + checksum_overhead
    }

    /// Peek at the length prefix without performing a full decode.
    ///
    /// Returns the payload length encoded in the first 4 bytes.
    ///
    /// # Errors
    ///
    /// - [`CodecError::BufferTooSmall`] if `data` has fewer than 4 bytes.
    /// - [`CodecError::InvalidLength`] if the decoded length is zero but
    ///   additional bytes suggest corruption (currently not enforced —
    ///   zero-length payloads are valid).
    pub fn decode_length(&self, data: &[u8]) -> Result<usize, CodecError> {
        if data.len() < 4 {
            return Err(CodecError::BufferTooSmall);
        }
        let bytes: [u8; 4] = data[..4]
            .try_into()
            .map_err(|_| CodecError::BufferTooSmall)?;
        Ok(u32::from_be_bytes(bytes) as usize)
    }

    /// Return cumulative codec statistics.
    pub fn stats(&self) -> CodecStats {
        CodecStats {
            messages_encoded: self.messages_encoded.get(),
            messages_decoded: self.messages_decoded.get(),
            bytes_encoded: self.bytes_encoded.get(),
            bytes_decoded: self.bytes_decoded.get(),
        }
    }

    /// Return a reference to the current configuration.
    pub fn config(&self) -> &CodecConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_codec() -> PeerMessageCodec {
        PeerMessageCodec::new(CodecConfig::default())
    }

    fn no_checksum_codec() -> PeerMessageCodec {
        PeerMessageCodec::new(CodecConfig {
            use_checksum: false,
            ..Default::default()
        })
    }

    // ---- roundtrip tests ----

    #[test]
    fn roundtrip_basic() {
        let codec = default_codec();
        let payload = b"hello, ipfrs";
        let wire = codec.encode(payload).expect("encode failed");
        let decoded = codec.decode(&wire).expect("decode failed");
        assert_eq!(decoded, payload);
    }

    #[test]
    fn roundtrip_empty_payload() {
        let codec = default_codec();
        let wire = codec.encode(b"").expect("encode empty");
        let decoded = codec.decode(&wire).expect("decode empty");
        assert!(decoded.is_empty());
    }

    #[test]
    fn roundtrip_no_checksum() {
        let codec = no_checksum_codec();
        let payload = b"no checksum";
        let wire = codec.encode(payload).expect("encode");
        let decoded = codec.decode(&wire).expect("decode");
        assert_eq!(decoded, payload);
    }

    #[test]
    fn roundtrip_large_payload() {
        let codec = default_codec();
        let payload = vec![0xAB_u8; 65_536];
        let wire = codec.encode(&payload).expect("encode large");
        let decoded = codec.decode(&wire).expect("decode large");
        assert_eq!(decoded, payload);
    }

    #[test]
    fn roundtrip_binary_data() {
        let codec = default_codec();
        let payload: Vec<u8> = (0..=255).collect();
        let wire = codec.encode(&payload).expect("encode binary");
        let decoded = codec.decode(&wire).expect("decode binary");
        assert_eq!(decoded, payload);
    }

    #[test]
    fn roundtrip_single_byte() {
        let codec = default_codec();
        let wire = codec.encode(&[0x42]).expect("encode");
        let decoded = codec.decode(&wire).expect("decode");
        assert_eq!(decoded, vec![0x42]);
    }

    // ---- checksum tests ----

    #[test]
    fn crc32_known_values() {
        // "123456789" should produce 0xCBF43926 (IEEE CRC32).
        let crc = PeerMessageCodec::crc32(b"123456789");
        assert_eq!(crc, 0xCBF4_3926);
    }

    #[test]
    fn crc32_empty() {
        let crc = PeerMessageCodec::crc32(b"");
        assert_eq!(crc, 0x0000_0000);
    }

    #[test]
    fn corrupted_payload_detected() {
        let codec = default_codec();
        let mut wire = codec.encode(b"integrity").expect("encode");
        // Corrupt one byte in the payload region.
        wire[5] ^= 0xFF;
        let result = codec.decode(&wire);
        assert_eq!(result, Err(CodecError::ChecksumMismatch));
    }

    #[test]
    fn corrupted_checksum_detected() {
        let codec = default_codec();
        let mut wire = codec.encode(b"test").expect("encode");
        // Corrupt the last byte (part of the checksum).
        let last = wire.len() - 1;
        wire[last] ^= 0x01;
        let result = codec.decode(&wire);
        assert_eq!(result, Err(CodecError::ChecksumMismatch));
    }

    #[test]
    fn no_checksum_no_verification() {
        let codec = no_checksum_codec();
        let mut wire = codec.encode(b"test").expect("encode");
        // Corrupt the payload — should still decode since checksum is off.
        wire[5] ^= 0xFF;
        let result = codec.decode(&wire);
        assert!(result.is_ok());
    }

    // ---- size enforcement ----

    #[test]
    fn message_too_large() {
        let codec = PeerMessageCodec::new(CodecConfig {
            max_message_size: 100,
            ..Default::default()
        });
        let payload = vec![0u8; 101];
        assert_eq!(codec.encode(&payload), Err(CodecError::MessageTooLarge));
    }

    #[test]
    fn message_exactly_max_size() {
        let codec = PeerMessageCodec::new(CodecConfig {
            max_message_size: 100,
            ..Default::default()
        });
        let payload = vec![0u8; 100];
        let wire = codec.encode(&payload).expect("encode at limit");
        let decoded = codec.decode(&wire).expect("decode at limit");
        assert_eq!(decoded, payload);
    }

    #[test]
    fn decode_rejects_oversized_length() {
        let codec = PeerMessageCodec::new(CodecConfig {
            max_message_size: 10,
            ..Default::default()
        });
        // Craft a frame claiming payload length of 100.
        let mut wire = vec![0, 0, 0, 100];
        wire.extend_from_slice(&[0u8; 100]);
        wire.extend_from_slice(&[0u8; 4]); // dummy checksum
        assert_eq!(codec.decode(&wire), Err(CodecError::MessageTooLarge));
    }

    // ---- buffer too small ----

    #[test]
    fn decode_empty_buffer() {
        let codec = default_codec();
        assert_eq!(codec.decode(&[]), Err(CodecError::BufferTooSmall));
    }

    #[test]
    fn decode_short_buffer() {
        let codec = default_codec();
        assert_eq!(codec.decode(&[0, 0, 1]), Err(CodecError::BufferTooSmall));
    }

    #[test]
    fn decode_truncated_payload() {
        let codec = default_codec();
        // Length says 10 bytes, but only 2 bytes of payload present.
        let wire = vec![0, 0, 0, 10, 0xAA, 0xBB];
        assert_eq!(codec.decode(&wire), Err(CodecError::InvalidLength));
    }

    #[test]
    fn decode_truncated_checksum() {
        let codec = default_codec();
        // Correct length for 4-byte payload, but missing checksum bytes.
        let mut wire = vec![0, 0, 0, 4];
        wire.extend_from_slice(&[1, 2, 3, 4]);
        // Only 2 of 4 checksum bytes.
        wire.extend_from_slice(&[0, 0]);
        assert_eq!(codec.decode(&wire), Err(CodecError::InvalidLength));
    }

    // ---- length prefix ----

    #[test]
    fn length_prefix_correctness() {
        let codec = default_codec();
        let payload = vec![0u8; 300];
        let wire = codec.encode(&payload).expect("encode");
        // First 4 bytes should be 300 in big-endian.
        assert_eq!(&wire[..4], &[0, 0, 1, 44]); // 300 = 0x012C
    }

    #[test]
    fn decode_length_peek() {
        let codec = default_codec();
        let payload = b"peek test";
        let wire = codec.encode(payload).expect("encode");
        let peeked = codec.decode_length(&wire).expect("peek");
        assert_eq!(peeked, payload.len());
    }

    #[test]
    fn decode_length_too_short() {
        let codec = default_codec();
        assert_eq!(
            codec.decode_length(&[0, 1]),
            Err(CodecError::BufferTooSmall)
        );
    }

    // ---- estimate_encoded_size ----

    #[test]
    fn estimate_with_checksum() {
        let codec = default_codec();
        assert_eq!(codec.estimate_encoded_size(100), 4 + 100 + 4);
    }

    #[test]
    fn estimate_without_checksum() {
        let codec = no_checksum_codec();
        assert_eq!(codec.estimate_encoded_size(100), 4 + 100);
    }

    #[test]
    fn estimate_matches_actual() {
        let codec = default_codec();
        let payload = b"size check";
        let wire = codec.encode(payload).expect("encode");
        assert_eq!(wire.len(), codec.estimate_encoded_size(payload.len()));
    }

    #[test]
    fn estimate_zero_payload() {
        let codec = default_codec();
        let wire = codec.encode(b"").expect("encode empty");
        assert_eq!(wire.len(), codec.estimate_encoded_size(0));
    }

    // ---- stats tracking ----

    #[test]
    fn stats_initial_zero() {
        let codec = default_codec();
        let s = codec.stats();
        assert_eq!(s.messages_encoded, 0);
        assert_eq!(s.messages_decoded, 0);
        assert_eq!(s.bytes_encoded, 0);
        assert_eq!(s.bytes_decoded, 0);
    }

    #[test]
    fn stats_encode_tracked() {
        let codec = default_codec();
        let _ = codec.encode(b"abc");
        let _ = codec.encode(b"defgh");
        let s = codec.stats();
        assert_eq!(s.messages_encoded, 2);
        assert_eq!(s.bytes_encoded, 8); // 3 + 5
    }

    #[test]
    fn stats_decode_tracked() {
        let codec = default_codec();
        let w1 = codec.encode(b"one").expect("e1");
        let w2 = codec.encode(b"two").expect("e2");
        let _ = codec.decode(&w1);
        let _ = codec.decode(&w2);
        let s = codec.stats();
        assert_eq!(s.messages_decoded, 2);
        assert_eq!(s.bytes_decoded, 6);
    }

    #[test]
    fn stats_failed_encode_not_counted() {
        let codec = PeerMessageCodec::new(CodecConfig {
            max_message_size: 5,
            ..Default::default()
        });
        let _ = codec.encode(&[0u8; 10]); // should fail
        assert_eq!(codec.stats().messages_encoded, 0);
    }

    #[test]
    fn stats_failed_decode_not_counted() {
        let codec = default_codec();
        let _ = codec.decode(&[]); // should fail
        assert_eq!(codec.stats().messages_decoded, 0);
    }

    // ---- encode_with_metadata ----

    #[test]
    fn encode_with_metadata_checksum_present() {
        let codec = default_codec();
        let em = codec.encode_with_metadata(b"meta").expect("meta encode");
        assert_eq!(em.original_size, 4);
        assert!(em.checksum.is_some());
        // Should round-trip.
        let decoded = codec.decode(&em.data).expect("meta decode");
        assert_eq!(decoded, b"meta");
    }

    #[test]
    fn encode_with_metadata_no_checksum() {
        let codec = no_checksum_codec();
        let em = codec.encode_with_metadata(b"no_crc").expect("meta encode");
        assert_eq!(em.original_size, 6);
        assert!(em.checksum.is_none());
    }

    // ---- misc ----

    #[test]
    fn codec_error_display() {
        assert_eq!(
            format!("{}", CodecError::BufferTooSmall),
            "buffer too small"
        );
        assert_eq!(
            format!("{}", CodecError::MessageTooLarge),
            "message too large"
        );
        assert_eq!(
            format!("{}", CodecError::InvalidLength),
            "invalid length prefix"
        );
        assert_eq!(
            format!("{}", CodecError::ChecksumMismatch),
            "checksum mismatch"
        );
    }

    #[test]
    fn codec_debug_impl() {
        let codec = default_codec();
        let dbg = format!("{:?}", codec);
        assert!(dbg.contains("PeerMessageCodec"));
    }

    #[test]
    fn config_default_values() {
        let cfg = CodecConfig::default();
        assert_eq!(cfg.max_message_size, 16_777_216);
        assert!(cfg.use_checksum);
        assert_eq!(cfg.length_prefix_bytes, 4);
    }
}
