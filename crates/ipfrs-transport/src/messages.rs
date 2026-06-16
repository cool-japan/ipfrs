//! Protocol message definitions for TensorSwap/Bitswap
//!
//! Defines the wire format for block exchange messages compatible
//! with IPFS Bitswap while adding TensorSwap extensions.
//!
//! # Example
//!
//! ```
//! use ipfrs_transport::messages::{Message, WantEntry};
//! use multihash::Multihash;
//! use cid::Cid;
//!
//! // Create a test CID
//! let hash = Multihash::wrap(0x12, &[0u8; 32]).unwrap();
//! let cid = Cid::new_v1(0x55, hash);
//!
//! // Create a want list message
//! let want_entry = WantEntry::with_priority(cid, 10);
//! let message = Message::want_list(vec![want_entry], false);
//!
//! // Serialize to bytes
//! let bytes = message.to_bytes().unwrap();
//!
//! // Deserialize back
//! let decoded = Message::from_bytes(&bytes).unwrap();
//!
//! // Verify roundtrip
//! match decoded {
//!     Message::WantList(wl) => {
//!         assert_eq!(wl.entries.len(), 1);
//!         assert_eq!(wl.entries[0].priority, 10);
//!     }
//!     _ => panic!("Expected WantList message"),
//! }
//! ```

use ipfrs_core::Cid;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Serialize CID as string
fn serialize_cid<S>(cid: &Cid, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(&cid.to_string())
}

/// Deserialize CID from string
fn deserialize_cid<'de, D>(deserializer: D) -> Result<Cid, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.parse().map_err(serde::de::Error::custom)
}

/// Message type for block exchange protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    /// Want list - request blocks from peers
    WantList(WantList),
    /// Block data response
    Block(BlockMessage),
    /// Notification that peer has a block
    Have(HaveMessage),
    /// Notification that peer doesn't have a block
    DontHave(DontHaveMessage),
    /// Cancel a previous want
    Cancel(CancelMessage),
}

/// Want list containing block requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WantList {
    /// List of wanted blocks
    pub entries: Vec<WantEntry>,
    /// Whether this is a full want list or incremental update
    pub full: bool,
}

/// Entry in a want list
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WantEntry {
    /// CID of wanted block
    #[serde(serialize_with = "serialize_cid", deserialize_with = "deserialize_cid")]
    pub cid: Cid,
    /// Priority (higher = more important)
    pub priority: i32,
    /// Whether to send the block or just confirmation
    pub send_dont_have: bool,
    /// Cancel this want
    pub cancel: bool,
}

impl WantEntry {
    /// Create a new want entry with default priority
    pub fn new(cid: Cid) -> Self {
        Self {
            cid,
            priority: 0,
            send_dont_have: false,
            cancel: false,
        }
    }

    /// Create a want entry with specific priority
    pub fn with_priority(cid: Cid, priority: i32) -> Self {
        Self {
            cid,
            priority,
            send_dont_have: false,
            cancel: false,
        }
    }

    /// Create a cancel entry
    pub fn cancel(cid: Cid) -> Self {
        Self {
            cid,
            priority: 0,
            send_dont_have: false,
            cancel: true,
        }
    }
}

/// Block message containing block data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockMessage {
    /// CID of the block
    #[serde(serialize_with = "serialize_cid", deserialize_with = "deserialize_cid")]
    pub cid: Cid,
    /// Block data
    pub data: Vec<u8>,
}

/// Have message - notify peer we have a block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HaveMessage {
    /// CID of the block we have
    #[serde(serialize_with = "serialize_cid", deserialize_with = "deserialize_cid")]
    pub cid: Cid,
}

/// Don't have message - notify peer we don't have a block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DontHaveMessage {
    /// CID of the block we don't have
    #[serde(serialize_with = "serialize_cid", deserialize_with = "deserialize_cid")]
    pub cid: Cid,
}

/// Cancel message - cancel a previous want
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelMessage {
    /// CID to cancel
    #[serde(serialize_with = "serialize_cid", deserialize_with = "deserialize_cid")]
    pub cid: Cid,
}

impl Message {
    /// Create a want list message
    pub fn want_list(entries: Vec<WantEntry>, full: bool) -> Self {
        Message::WantList(WantList { entries, full })
    }

    /// Create a block message
    pub fn block(cid: Cid, data: Vec<u8>) -> Self {
        Message::Block(BlockMessage { cid, data })
    }

    /// Create a have message
    pub fn have(cid: Cid) -> Self {
        Message::Have(HaveMessage { cid })
    }

    /// Create a don't have message
    pub fn dont_have(cid: Cid) -> Self {
        Message::DontHave(DontHaveMessage { cid })
    }

    /// Create a cancel message
    pub fn cancel(cid: Cid) -> Self {
        Message::Cancel(CancelMessage { cid })
    }

    /// Serialize message to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, oxicode::Error> {
        oxicode::serde::encode_to_vec(self, oxicode::config::standard())
    }

    /// Deserialize message from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, oxicode::Error> {
        oxicode::serde::decode_owned_from_slice(data, oxicode::config::standard()).map(|(v, _)| v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cid() -> Cid {
        "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: valid CID string")
    }

    fn test_cid2() -> Cid {
        "bafybeihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku"
            .parse::<Cid>()
            .expect("test: valid CID string")
    }

    // Basic WantEntry Tests
    #[test]
    fn test_want_entry_creation() {
        let cid = test_cid();

        let entry = WantEntry::new(cid);
        assert_eq!(entry.priority, 0);
        assert!(!entry.cancel);
        assert!(!entry.send_dont_have);

        let priority_entry = WantEntry::with_priority(cid, 10);
        assert_eq!(priority_entry.priority, 10);
        assert!(!priority_entry.cancel);

        let cancel_entry = WantEntry::cancel(cid);
        assert!(cancel_entry.cancel);
        assert_eq!(cancel_entry.priority, 0);
    }

    #[test]
    fn test_want_entry_edge_cases() {
        let cid = test_cid();

        // Max priority
        let max_entry = WantEntry::with_priority(cid, i32::MAX);
        assert_eq!(max_entry.priority, i32::MAX);

        // Min priority
        let min_entry = WantEntry::with_priority(cid, i32::MIN);
        assert_eq!(min_entry.priority, i32::MIN);

        // Zero priority
        let zero_entry = WantEntry::with_priority(cid, 0);
        assert_eq!(zero_entry.priority, 0);

        // Negative priority
        let neg_entry = WantEntry::with_priority(cid, -100);
        assert_eq!(neg_entry.priority, -100);
    }

    // Message Serialization Tests
    #[test]
    fn test_want_list_serialization_roundtrip() {
        let cid1 = test_cid();
        let cid2 = test_cid2();

        let entries = vec![
            WantEntry::with_priority(cid1, 10),
            WantEntry::with_priority(cid2, 5),
        ];

        let msg = Message::want_list(entries.clone(), true);
        let bytes = msg.to_bytes().expect("test: message serialization");
        let decoded = Message::from_bytes(&bytes).expect("test: message deserialization");

        match decoded {
            Message::WantList(want_list) => {
                assert!(want_list.full);
                assert_eq!(want_list.entries.len(), 2);
                assert_eq!(want_list.entries[0].cid, cid1);
                assert_eq!(want_list.entries[0].priority, 10);
                assert_eq!(want_list.entries[1].cid, cid2);
                assert_eq!(want_list.entries[1].priority, 5);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_block_message_serialization_roundtrip() {
        let cid = test_cid();
        let data = vec![1, 2, 3, 4, 5];

        let msg = Message::block(cid, data.clone());
        let bytes = msg.to_bytes().expect("test: message serialization");
        let decoded = Message::from_bytes(&bytes).expect("test: message deserialization");

        match decoded {
            Message::Block(block) => {
                assert_eq!(block.cid, cid);
                assert_eq!(block.data, data);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_have_message_serialization_roundtrip() {
        let cid = test_cid();

        let msg = Message::have(cid);
        let bytes = msg.to_bytes().expect("test: message serialization");
        let decoded = Message::from_bytes(&bytes).expect("test: message deserialization");

        match decoded {
            Message::Have(have) => assert_eq!(have.cid, cid),
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_dont_have_message_serialization_roundtrip() {
        let cid = test_cid();

        let msg = Message::dont_have(cid);
        let bytes = msg.to_bytes().expect("test: message serialization");
        let decoded = Message::from_bytes(&bytes).expect("test: message deserialization");

        match decoded {
            Message::DontHave(dont_have) => assert_eq!(dont_have.cid, cid),
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_cancel_message_serialization_roundtrip() {
        let cid = test_cid();

        let msg = Message::cancel(cid);
        let bytes = msg.to_bytes().expect("test: message serialization");
        let decoded = Message::from_bytes(&bytes).expect("test: message deserialization");

        match decoded {
            Message::Cancel(cancel) => assert_eq!(cancel.cid, cid),
            _ => panic!("Wrong message type"),
        }
    }

    // Edge Case Tests
    #[test]
    fn test_empty_want_list() {
        let msg = Message::want_list(vec![], false);
        let bytes = msg.to_bytes().expect("test: message serialization");
        let decoded = Message::from_bytes(&bytes).expect("test: message deserialization");

        match decoded {
            Message::WantList(want_list) => {
                assert!(!want_list.full);
                assert_eq!(want_list.entries.len(), 0);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_block_with_empty_data() {
        let cid = test_cid();
        let msg = Message::block(cid, vec![]);
        let bytes = msg.to_bytes().expect("test: message serialization");
        let decoded = Message::from_bytes(&bytes).expect("test: message deserialization");

        match decoded {
            Message::Block(block) => {
                assert_eq!(block.cid, cid);
                assert_eq!(block.data.len(), 0);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_block_with_large_data() {
        let cid = test_cid();
        let large_data = vec![42u8; 1_000_000]; // 1 MB
        let msg = Message::block(cid, large_data.clone());
        let bytes = msg.to_bytes().expect("test: message serialization");
        let decoded = Message::from_bytes(&bytes).expect("test: message deserialization");

        match decoded {
            Message::Block(block) => {
                assert_eq!(block.cid, cid);
                assert_eq!(block.data.len(), 1_000_000);
                assert_eq!(block.data, large_data);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_want_list_with_many_entries() {
        let cid = test_cid();
        let entries: Vec<WantEntry> = (0..1000)
            .map(|i| WantEntry::with_priority(cid, i))
            .collect();

        let msg = Message::want_list(entries, true);
        let bytes = msg.to_bytes().expect("test: message serialization");
        let decoded = Message::from_bytes(&bytes).expect("test: message deserialization");

        match decoded {
            Message::WantList(want_list) => {
                assert_eq!(want_list.entries.len(), 1000);
                assert_eq!(want_list.entries[500].priority, 500);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_want_entry_with_all_flags() {
        let cid = test_cid();
        let mut entry = WantEntry::with_priority(cid, 100);
        entry.send_dont_have = true;
        entry.cancel = true;

        let msg = Message::want_list(vec![entry], false);
        let bytes = msg.to_bytes().expect("test: message serialization");
        let decoded = Message::from_bytes(&bytes).expect("test: message deserialization");

        match decoded {
            Message::WantList(want_list) => {
                assert_eq!(want_list.entries[0].priority, 100);
                assert!(want_list.entries[0].send_dont_have);
                assert!(want_list.entries[0].cancel);
            }
            _ => panic!("Wrong message type"),
        }
    }

    // Malformed Input Tests
    #[test]
    fn test_invalid_message_bytes() {
        let invalid_bytes = vec![0xFF, 0xFF, 0xFF, 0xFF];
        let result = Message::from_bytes(&invalid_bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_bytes() {
        let empty_bytes: Vec<u8> = vec![];
        let result = Message::from_bytes(&empty_bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_truncated_message() {
        let cid = test_cid();
        let msg = Message::have(cid);
        let bytes = msg.to_bytes().expect("test: message serialization");

        // Take only first half of bytes
        let truncated = &bytes[..bytes.len() / 2];
        let result = Message::from_bytes(truncated);
        assert!(result.is_err());
    }

    #[test]
    fn test_corrupted_message() {
        let cid = test_cid();
        let msg = Message::have(cid);
        let mut bytes = msg.to_bytes().expect("test: message serialization");

        // Corrupt some bytes
        if bytes.len() > 10 {
            bytes[5] = !bytes[5];
            bytes[10] = !bytes[10];
        }

        // May or may not deserialize, but shouldn't panic
        let _ = Message::from_bytes(&bytes);
    }

    // JSON Serialization Tests
    #[test]
    fn test_json_serialization_want_list() {
        let cid = test_cid();
        let entries = vec![WantEntry::with_priority(cid, 10)];
        let msg = Message::want_list(entries, true);

        let json = serde_json::to_string(&msg).expect("test: JSON serialization");
        let decoded: Message = serde_json::from_str(&json).expect("test: JSON deserialization");

        match decoded {
            Message::WantList(want_list) => {
                assert!(want_list.full);
                assert_eq!(want_list.entries.len(), 1);
                assert_eq!(want_list.entries[0].priority, 10);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_json_serialization_block() {
        let cid = test_cid();
        let data = vec![1, 2, 3];
        let msg = Message::block(cid, data.clone());

        let json = serde_json::to_string(&msg).expect("test: JSON serialization");
        let decoded: Message = serde_json::from_str(&json).expect("test: JSON deserialization");

        match decoded {
            Message::Block(block) => {
                assert_eq!(block.cid, cid);
                assert_eq!(block.data, data);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_json_serialization_have() {
        let cid = test_cid();
        let msg = Message::have(cid);

        let json = serde_json::to_string(&msg).expect("test: JSON serialization");
        let decoded: Message = serde_json::from_str(&json).expect("test: JSON deserialization");

        match decoded {
            Message::Have(have) => assert_eq!(have.cid, cid),
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_json_serialization_dont_have() {
        let cid = test_cid();
        let msg = Message::dont_have(cid);

        let json = serde_json::to_string(&msg).expect("test: JSON serialization");
        let decoded: Message = serde_json::from_str(&json).expect("test: JSON deserialization");

        match decoded {
            Message::DontHave(dont_have) => assert_eq!(dont_have.cid, cid),
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_json_serialization_cancel() {
        let cid = test_cid();
        let msg = Message::cancel(cid);

        let json = serde_json::to_string(&msg).expect("test: JSON serialization");
        let decoded: Message = serde_json::from_str(&json).expect("test: JSON deserialization");

        match decoded {
            Message::Cancel(cancel) => assert_eq!(cancel.cid, cid),
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_invalid_json() {
        let invalid_json = r#"{"invalid": "structure"}"#;
        let result: Result<Message, _> = serde_json::from_str(invalid_json);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_cid_in_json() {
        let invalid_json = r#"{"Have":{"cid":"not-a-valid-cid"}}"#;
        let result: Result<Message, _> = serde_json::from_str(invalid_json);
        assert!(result.is_err());
    }

    // WantEntry Equality Tests
    #[test]
    fn test_want_entry_equality() {
        let cid = test_cid();
        let entry1 = WantEntry::with_priority(cid, 10);
        let entry2 = WantEntry::with_priority(cid, 10);
        assert_eq!(entry1, entry2);

        let entry3 = WantEntry::with_priority(cid, 20);
        assert_ne!(entry1, entry3);

        let cid2 = test_cid2();
        let entry4 = WantEntry::with_priority(cid2, 10);
        assert_ne!(entry1, entry4);
    }
}
