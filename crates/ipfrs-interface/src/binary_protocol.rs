//! Binary Protocol for High-Speed IPFRS Communication
//!
//! Provides a compact binary message format for efficient operations:
//! - Lower overhead than JSON/HTTP
//! - Protocol versioning for backward compatibility
//! - Zero-copy deserialization where possible
//! - Support for all major IPFRS operations

use bytes::{Buf, BufMut, Bytes, BytesMut};
use ipfrs_core::Cid;
use std::io::{self, Cursor};
use thiserror::Error;

// ============================================================================
// Protocol Constants
// ============================================================================

/// Current protocol version
pub const PROTOCOL_VERSION: u8 = 1;

/// Magic bytes to identify IPFRS binary protocol
pub const MAGIC: [u8; 4] = *b"IPFS";

/// Maximum message size (16MB)
pub const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024;

// ============================================================================
// Message Types
// ============================================================================

/// Binary protocol message type identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageType {
    /// Get a block by CID
    GetBlock = 0x01,
    /// Put a block
    PutBlock = 0x02,
    /// Check if block exists
    HasBlock = 0x03,
    /// Delete a block
    DeleteBlock = 0x04,
    /// Batch get blocks
    BatchGet = 0x05,
    /// Batch put blocks
    BatchPut = 0x06,
    /// Batch has blocks
    BatchHas = 0x07,
    /// Success response
    Success = 0x80,
    /// Error response
    Error = 0x81,
}

impl MessageType {
    /// Convert from u8
    pub fn from_u8(value: u8) -> Result<Self, ProtocolError> {
        match value {
            0x01 => Ok(MessageType::GetBlock),
            0x02 => Ok(MessageType::PutBlock),
            0x03 => Ok(MessageType::HasBlock),
            0x04 => Ok(MessageType::DeleteBlock),
            0x05 => Ok(MessageType::BatchGet),
            0x06 => Ok(MessageType::BatchPut),
            0x07 => Ok(MessageType::BatchHas),
            0x80 => Ok(MessageType::Success),
            0x81 => Ok(MessageType::Error),
            _ => Err(ProtocolError::InvalidMessageType(value)),
        }
    }

    /// Convert to u8
    pub fn to_u8(self) -> u8 {
        self as u8
    }
}

// ============================================================================
// Message Structure
// ============================================================================

/// Binary protocol message
///
/// Wire format:
/// ```text
/// +------+------+------+------+----------+---------+
/// | MAGIC (4) | VER | TYPE | MSG_ID (4) | PAYLOAD |
/// +------+------+------+------+----------+---------+
/// ```
#[derive(Debug, Clone)]
pub struct BinaryMessage {
    /// Protocol version
    pub version: u8,
    /// Message type
    pub msg_type: MessageType,
    /// Message ID for request/response matching
    pub message_id: u32,
    /// Message payload
    pub payload: Bytes,
}

impl BinaryMessage {
    /// Create a new binary message
    pub fn new(msg_type: MessageType, message_id: u32, payload: Bytes) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            msg_type,
            message_id,
            payload,
        }
    }

    /// Serialize message to bytes
    pub fn encode(&self) -> Result<Bytes, ProtocolError> {
        let total_size = 4 + 1 + 1 + 4 + self.payload.len();
        if total_size > MAX_MESSAGE_SIZE {
            return Err(ProtocolError::MessageTooLarge(total_size));
        }

        let mut buf = BytesMut::with_capacity(total_size);

        // Magic bytes
        buf.put_slice(&MAGIC);
        // Version
        buf.put_u8(self.version);
        // Message type
        buf.put_u8(self.msg_type.to_u8());
        // Message ID
        buf.put_u32(self.message_id);
        // Payload
        buf.put_slice(&self.payload);

        Ok(buf.freeze())
    }

    /// Deserialize message from bytes
    pub fn decode(data: &[u8]) -> Result<Self, ProtocolError> {
        if data.len() < 10 {
            return Err(ProtocolError::InvalidMessageSize(data.len()));
        }

        if data.len() > MAX_MESSAGE_SIZE {
            return Err(ProtocolError::MessageTooLarge(data.len()));
        }

        let mut cursor = Cursor::new(data);

        // Check magic bytes
        let mut magic = [0u8; 4];
        cursor.copy_to_slice(&mut magic);
        if magic != MAGIC {
            return Err(ProtocolError::InvalidMagic(magic));
        }

        // Read version
        let version = cursor.get_u8();
        if version > PROTOCOL_VERSION {
            return Err(ProtocolError::UnsupportedVersion(version));
        }

        // Read message type
        let msg_type = MessageType::from_u8(cursor.get_u8())?;

        // Read message ID
        let message_id = cursor.get_u32();

        // Read payload
        let position = cursor.position() as usize;
        let payload = Bytes::copy_from_slice(&data[position..]);

        Ok(Self {
            version,
            msg_type,
            message_id,
            payload,
        })
    }
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// Get block request
#[derive(Debug, Clone)]
pub struct GetBlockRequest {
    pub cid: Cid,
}

impl GetBlockRequest {
    /// Encode to bytes
    pub fn encode(&self) -> Result<Bytes, ProtocolError> {
        let cid_bytes = self.cid.to_bytes();
        let mut buf = BytesMut::with_capacity(cid_bytes.len());
        buf.put_slice(&cid_bytes);
        Ok(buf.freeze())
    }

    /// Decode from bytes
    pub fn decode(data: &[u8]) -> Result<Self, ProtocolError> {
        let cid = Cid::try_from(data).map_err(|e| ProtocolError::InvalidCid(e.to_string()))?;
        Ok(Self { cid })
    }
}

/// Put block request
#[derive(Debug, Clone)]
pub struct PutBlockRequest {
    pub data: Bytes,
}

impl PutBlockRequest {
    /// Encode to bytes
    pub fn encode(&self) -> Result<Bytes, ProtocolError> {
        Ok(self.data.clone())
    }

    /// Decode from bytes
    pub fn decode(data: &[u8]) -> Result<Self, ProtocolError> {
        Ok(Self {
            data: Bytes::copy_from_slice(data),
        })
    }
}

/// Has block request
#[derive(Debug, Clone)]
pub struct HasBlockRequest {
    pub cid: Cid,
}

impl HasBlockRequest {
    /// Encode to bytes
    pub fn encode(&self) -> Result<Bytes, ProtocolError> {
        let cid_bytes = self.cid.to_bytes();
        let mut buf = BytesMut::with_capacity(cid_bytes.len());
        buf.put_slice(&cid_bytes);
        Ok(buf.freeze())
    }

    /// Decode from bytes
    pub fn decode(data: &[u8]) -> Result<Self, ProtocolError> {
        let cid = Cid::try_from(data).map_err(|e| ProtocolError::InvalidCid(e.to_string()))?;
        Ok(Self { cid })
    }
}

/// Batch get request
#[derive(Debug, Clone)]
pub struct BatchGetRequest {
    pub cids: Vec<Cid>,
}

impl BatchGetRequest {
    /// Encode to bytes
    pub fn encode(&self) -> Result<Bytes, ProtocolError> {
        let mut buf = BytesMut::new();

        // Write count
        buf.put_u32(self.cids.len() as u32);

        // Write each CID
        for cid in &self.cids {
            let cid_bytes = cid.to_bytes();
            buf.put_u16(cid_bytes.len() as u16);
            buf.put_slice(&cid_bytes);
        }

        Ok(buf.freeze())
    }

    /// Decode from bytes
    pub fn decode(data: &[u8]) -> Result<Self, ProtocolError> {
        let mut cursor = Cursor::new(data);

        // Read count
        if cursor.remaining() < 4 {
            return Err(ProtocolError::InvalidMessageSize(cursor.remaining()));
        }
        let count = cursor.get_u32() as usize;

        let mut cids = Vec::with_capacity(count);

        // Read each CID
        for _ in 0..count {
            if cursor.remaining() < 2 {
                return Err(ProtocolError::InvalidMessageSize(cursor.remaining()));
            }
            let len = cursor.get_u16() as usize;

            if cursor.remaining() < len {
                return Err(ProtocolError::InvalidMessageSize(cursor.remaining()));
            }

            let position = cursor.position() as usize;
            let cid_data = &data[position..position + len];
            let cid =
                Cid::try_from(cid_data).map_err(|e| ProtocolError::InvalidCid(e.to_string()))?;
            cids.push(cid);
            cursor.set_position((position + len) as u64);
        }

        Ok(Self { cids })
    }
}

/// Success response
#[derive(Debug, Clone)]
pub struct SuccessResponse {
    pub data: Bytes,
}

impl SuccessResponse {
    /// Encode to bytes
    pub fn encode(&self) -> Result<Bytes, ProtocolError> {
        Ok(self.data.clone())
    }

    /// Decode from bytes
    pub fn decode(data: &[u8]) -> Result<Self, ProtocolError> {
        Ok(Self {
            data: Bytes::copy_from_slice(data),
        })
    }
}

/// Error response
#[derive(Debug, Clone)]
pub struct ErrorResponse {
    pub error_code: u16,
    pub message: String,
}

impl ErrorResponse {
    /// Encode to bytes
    pub fn encode(&self) -> Result<Bytes, ProtocolError> {
        let message_bytes = self.message.as_bytes();
        let mut buf = BytesMut::with_capacity(2 + 2 + message_bytes.len());

        // Error code
        buf.put_u16(self.error_code);
        // Message length
        buf.put_u16(message_bytes.len() as u16);
        // Message
        buf.put_slice(message_bytes);

        Ok(buf.freeze())
    }

    /// Decode from bytes
    pub fn decode(data: &[u8]) -> Result<Self, ProtocolError> {
        let mut cursor = Cursor::new(data);

        if cursor.remaining() < 4 {
            return Err(ProtocolError::InvalidMessageSize(cursor.remaining()));
        }

        let error_code = cursor.get_u16();
        let message_len = cursor.get_u16() as usize;

        if cursor.remaining() < message_len {
            return Err(ProtocolError::InvalidMessageSize(cursor.remaining()));
        }

        let position = cursor.position() as usize;
        let message_bytes = &data[position..position + message_len];
        let message = String::from_utf8(message_bytes.to_vec())
            .map_err(|e| ProtocolError::InvalidUtf8(e.to_string()))?;

        Ok(Self {
            error_code,
            message,
        })
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Binary protocol errors
#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("Invalid magic bytes: {0:?}")]
    InvalidMagic([u8; 4]),

    #[error("Unsupported protocol version: {0}")]
    UnsupportedVersion(u8),

    #[error("Invalid message type: {0}")]
    InvalidMessageType(u8),

    #[error("Invalid message size: {0}")]
    InvalidMessageSize(usize),

    #[error("Message too large: {0} bytes")]
    MessageTooLarge(usize),

    #[error("Invalid CID: {0}")]
    InvalidCid(String),

    #[error("Invalid UTF-8: {0}")]
    InvalidUtf8(String),

    #[error("IO error: {0}")]
    Io(#[from] io::Error),
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_type_conversion() {
        assert_eq!(
            MessageType::from_u8(0x01).expect("test: 0x01 should map to GetBlock"),
            MessageType::GetBlock
        );
        assert_eq!(MessageType::GetBlock.to_u8(), 0x01);
        assert!(MessageType::from_u8(0xFF).is_err());
    }

    #[test]
    fn test_binary_message_encode_decode() {
        let payload = Bytes::from("test payload");
        let msg = BinaryMessage::new(MessageType::GetBlock, 42, payload.clone());

        let encoded = msg
            .encode()
            .expect("test: BinaryMessage encode should succeed");
        let decoded =
            BinaryMessage::decode(&encoded).expect("test: BinaryMessage decode should succeed");

        assert_eq!(decoded.version, PROTOCOL_VERSION);
        assert_eq!(decoded.msg_type, MessageType::GetBlock);
        assert_eq!(decoded.message_id, 42);
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn test_message_too_large() {
        let large_payload = Bytes::from(vec![0u8; MAX_MESSAGE_SIZE]);
        let msg = BinaryMessage::new(MessageType::GetBlock, 1, large_payload);
        assert!(msg.encode().is_err());
    }

    #[test]
    fn test_invalid_magic() {
        let data = vec![0xFF, 0xFF, 0xFF, 0xFF, 1, 1, 0, 0, 0, 42];
        let result = BinaryMessage::decode(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_get_request_encode_decode() {
        // Create test CIDs from actual blocks
        use ipfrs_core::Block;
        let block1 = Block::new(Bytes::from("test data 1"))
            .expect("test: Block creation should succeed for test data 1");
        let block2 = Block::new(Bytes::from("test data 2"))
            .expect("test: Block creation should succeed for test data 2");
        let cid1 = *block1.cid();
        let cid2 = *block2.cid();

        let request = BatchGetRequest {
            cids: vec![cid1, cid2],
        };

        let encoded = request
            .encode()
            .expect("test: BatchGetRequest encode should succeed");
        let decoded =
            BatchGetRequest::decode(&encoded).expect("test: BatchGetRequest decode should succeed");

        assert_eq!(decoded.cids.len(), 2);
        assert_eq!(decoded.cids[0], cid1);
        assert_eq!(decoded.cids[1], cid2);
    }

    #[test]
    fn test_error_response_encode_decode() {
        let response = ErrorResponse {
            error_code: 404,
            message: "Block not found".to_string(),
        };

        let encoded = response
            .encode()
            .expect("test: ErrorResponse encode should succeed");
        let decoded =
            ErrorResponse::decode(&encoded).expect("test: ErrorResponse decode should succeed");

        assert_eq!(decoded.error_code, 404);
        assert_eq!(decoded.message, "Block not found");
    }

    #[test]
    fn test_protocol_versioning() {
        let payload = Bytes::from("test");
        let mut msg = BinaryMessage::new(MessageType::GetBlock, 1, payload);

        // Test current version
        msg.version = PROTOCOL_VERSION;
        let encoded = msg
            .encode()
            .expect("test: encode should succeed with current protocol version");
        assert!(BinaryMessage::decode(&encoded).is_ok());

        // Test future version (should fail)
        msg.version = PROTOCOL_VERSION + 1;
        let encoded = msg
            .encode()
            .expect("test: encode should succeed with future protocol version");
        assert!(BinaryMessage::decode(&encoded).is_err());
    }
}
