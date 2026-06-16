//! Pin Management System
//!
//! Manages pinned blocks to prevent them from being garbage collected.
//! Supports recursive pinning (pins all referenced blocks) and direct pinning.

use ipfrs_core::{Cid, Error, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

/// Pin type - determines how the block is pinned
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PinType {
    /// Direct pin - only the specified block is pinned
    Direct,
    /// Recursive pin - the block and all referenced blocks are pinned
    Recursive,
}

/// Information about a pinned block
#[derive(Debug, Clone)]
pub struct PinInfo {
    /// Content identifier
    pub cid: Cid,
    /// Pin type
    pub pin_type: PinType,
    /// Optional name/label for the pin
    pub name: Option<String>,
    /// Timestamp when the pin was created
    pub created: chrono::DateTime<chrono::Utc>,
}

/// Serializable version of PinInfo for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PinInfoSerde {
    cid_str: String,
    pin_type: PinType,
    name: Option<String>,
    created: chrono::DateTime<chrono::Utc>,
}

impl From<&PinInfo> for PinInfoSerde {
    fn from(pin: &PinInfo) -> Self {
        Self {
            cid_str: pin.cid.to_string(),
            pin_type: pin.pin_type,
            name: pin.name.clone(),
            created: pin.created,
        }
    }
}

impl PinInfoSerde {
    fn into_pin_info(self) -> Result<PinInfo> {
        use std::str::FromStr;
        let cid = Cid::from_str(&self.cid_str)
            .map_err(|e| Error::Serialization(format!("Invalid CID in stored pin: {}", e)))?;
        Ok(PinInfo {
            cid,
            pin_type: self.pin_type,
            name: self.name,
            created: self.created,
        })
    }
}

/// Pin Manager - manages pinned blocks
pub struct PinManager {
    /// Pinned blocks with their metadata
    pins: Arc<RwLock<HashMap<Cid, PinInfo>>>,
    /// Indirect pins (blocks pinned because they're referenced by recursive pins)
    indirect_pins: Arc<RwLock<HashSet<Cid>>>,
    /// Maps recursive pins to their indirect dependencies
    recursive_deps: Arc<RwLock<HashMap<Cid, HashSet<Cid>>>>,
}

impl PinManager {
    /// Create a new pin manager
    pub fn new() -> Self {
        Self {
            pins: Arc::new(RwLock::new(HashMap::new())),
            indirect_pins: Arc::new(RwLock::new(HashSet::new())),
            recursive_deps: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Pin a block
    ///
    /// # Arguments
    /// * `cid` - Content identifier to pin
    /// * `pin_type` - Whether to pin recursively or directly
    /// * `name` - Optional name/label for the pin
    pub fn pin(&self, cid: Cid, pin_type: PinType, name: Option<String>) -> Result<()> {
        let mut pins = self.pins.write();

        let pin_info = PinInfo {
            cid,
            pin_type,
            name,
            created: chrono::Utc::now(),
        };

        pins.insert(cid, pin_info);

        Ok(())
    }

    /// Add an indirect pin for a recursive pin
    ///
    /// # Arguments
    /// * `root_cid` - The CID of the recursive pin
    /// * `indirect_cid` - The CID of the indirectly pinned block
    pub fn add_indirect(&self, root_cid: Cid, indirect_cid: Cid) {
        let mut indirect = self.indirect_pins.write();
        indirect.insert(indirect_cid);

        let mut deps = self.recursive_deps.write();
        deps.entry(root_cid).or_default().insert(indirect_cid);
    }

    /// Add multiple indirect pins for a recursive pin
    pub fn add_indirect_many(&self, root_cid: Cid, indirect_cids: &[Cid]) {
        let mut indirect = self.indirect_pins.write();
        let mut deps = self.recursive_deps.write();

        let dep_set = deps.entry(root_cid).or_default();

        for &cid in indirect_cids {
            indirect.insert(cid);
            dep_set.insert(cid);
        }
    }

    /// Unpin a block
    ///
    /// # Arguments
    /// * `cid` - Content identifier to unpin
    /// * `_recursive` - Whether to also unpin referenced blocks (currently unused, kept for API compatibility)
    pub fn unpin(&self, cid: &Cid, _recursive: bool) -> Result<()> {
        let mut pins = self.pins.write();

        if let Some(pin_info) = pins.remove(cid) {
            // If it was a recursive pin, remove indirect pins as well
            if pin_info.pin_type == PinType::Recursive {
                let mut deps = self.recursive_deps.write();
                if let Some(indirect_cids) = deps.remove(cid) {
                    // Remove indirect pins that are only referenced by this recursive pin
                    let mut indirect = self.indirect_pins.write();

                    for indirect_cid in indirect_cids {
                        // Check if this indirect pin is still needed by another recursive pin
                        let still_needed = deps.values().any(|set| set.contains(&indirect_cid));

                        if !still_needed {
                            indirect.remove(&indirect_cid);
                        }
                    }
                }
            }
            Ok(())
        } else {
            Err(Error::NotFound(format!("Block not pinned: {}", cid)))
        }
    }

    /// Check if a block is pinned (directly or indirectly)
    pub fn is_pinned(&self, cid: &Cid) -> bool {
        let pins = self.pins.read();
        if pins.contains_key(cid) {
            return true;
        }

        let indirect = self.indirect_pins.read();
        indirect.contains(cid)
    }

    /// Check if a block is directly pinned
    pub fn is_directly_pinned(&self, cid: &Cid) -> bool {
        let pins = self.pins.read();
        pins.contains_key(cid)
    }

    /// List all pinned blocks
    pub fn list(&self) -> Vec<PinInfo> {
        let pins = self.pins.read();
        pins.values().cloned().collect()
    }

    /// Get pin information for a specific CID
    pub fn get(&self, cid: &Cid) -> Option<PinInfo> {
        let pins = self.pins.read();
        pins.get(cid).cloned()
    }

    /// Count total pins
    pub fn count(&self) -> usize {
        let pins = self.pins.read();
        pins.len()
    }

    /// Count indirect pins
    pub fn indirect_count(&self) -> usize {
        let indirect = self.indirect_pins.read();
        indirect.len()
    }

    /// Save pin index to disk
    pub async fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let pin_list: Vec<PinInfoSerde> = {
            let pins = self.pins.read();
            pins.values().map(PinInfoSerde::from).collect()
        };

        let data = oxicode::serde::encode_to_vec(&pin_list, oxicode::config::standard())
            .map_err(|e| Error::Serialization(format!("Failed to serialize pins: {}", e)))?;

        tokio::fs::write(path.as_ref(), data).await?;

        Ok(())
    }

    /// Load pin index from disk
    pub async fn load(&self, path: impl AsRef<Path>) -> Result<()> {
        let data = tokio::fs::read(path.as_ref()).await?;

        let pin_list_serde: Vec<PinInfoSerde> =
            oxicode::serde::decode_owned_from_slice(&data, oxicode::config::standard())
                .map(|(v, _)| v)
                .map_err(|e| Error::Serialization(format!("Failed to deserialize pins: {}", e)))?;

        let mut pins = self.pins.write();
        pins.clear();

        for pin_serde in pin_list_serde {
            let pin_info = pin_serde.into_pin_info()?;
            pins.insert(pin_info.cid, pin_info);
        }

        Ok(())
    }

    /// Clear all pins (use with caution!)
    pub fn clear(&self) {
        let mut pins = self.pins.write();
        pins.clear();

        let mut indirect = self.indirect_pins.write();
        indirect.clear();

        let mut deps = self.recursive_deps.write();
        deps.clear();
    }
}

impl Default for PinManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pin_manager_basic() {
        let manager = PinManager::new();
        let cid = Cid::default();

        // Pin a block
        manager
            .pin(cid, PinType::Direct, Some("test".to_string()))
            .expect("test: pin should succeed");

        // Check if pinned
        assert!(manager.is_pinned(&cid));
        assert!(manager.is_directly_pinned(&cid));

        // Get pin info
        let info = manager.get(&cid).expect("test: get pin should return info");
        assert_eq!(info.cid, cid);
        assert_eq!(info.pin_type, PinType::Direct);
        assert_eq!(info.name, Some("test".to_string()));

        // List pins
        let pins = manager.list();
        assert_eq!(pins.len(), 1);

        // Unpin
        manager
            .unpin(&cid, false)
            .expect("test: unpin should succeed");
        assert!(!manager.is_pinned(&cid));
    }

    #[test]
    fn test_pin_manager_indirect() {
        use bytes::Bytes;
        use ipfrs_core::Block;

        let manager = PinManager::new();

        // Create two different CIDs
        let block1 = Block::new(Bytes::from("data1")).expect("test: block creation should succeed");
        let block2 = Block::new(Bytes::from("data2")).expect("test: block creation should succeed");
        let cid1 = *block1.cid();
        let cid2 = *block2.cid();

        // Pin recursively
        manager
            .pin(cid1, PinType::Recursive, None)
            .expect("test: pin should succeed");

        // Add indirect pin
        manager.add_indirect(cid1, cid2);

        // Check both are pinned
        assert!(manager.is_directly_pinned(&cid1));
        assert!(!manager.is_directly_pinned(&cid2));
        assert!(manager.is_pinned(&cid2)); // But indirectly pinned

        // Count pins
        assert_eq!(manager.count(), 1); // Only direct pin
        assert_eq!(manager.indirect_count(), 1); // Indirect pin
    }

    #[test]
    fn test_pin_manager_recursive_unpin() {
        use bytes::Bytes;
        use ipfrs_core::Block;

        let manager = PinManager::new();

        // Create three different CIDs
        let block1 = Block::new(Bytes::from("data1")).expect("test: block creation should succeed");
        let block2 = Block::new(Bytes::from("data2")).expect("test: block creation should succeed");
        let block3 = Block::new(Bytes::from("data3")).expect("test: block creation should succeed");
        let cid1 = *block1.cid();
        let cid2 = *block2.cid();
        let cid3 = *block3.cid();

        // Pin cid1 recursively with cid2 and cid3 as indirect pins
        manager
            .pin(cid1, PinType::Recursive, None)
            .expect("test: pin should succeed");
        manager.add_indirect(cid1, cid2);
        manager.add_indirect(cid1, cid3);

        // Verify all are pinned
        assert!(manager.is_pinned(&cid1));
        assert!(manager.is_pinned(&cid2));
        assert!(manager.is_pinned(&cid3));

        // Unpin the recursive pin
        manager
            .unpin(&cid1, true)
            .expect("test: unpin should succeed");

        // Verify all indirect pins are removed
        assert!(!manager.is_pinned(&cid1));
        assert!(!manager.is_pinned(&cid2));
        assert!(!manager.is_pinned(&cid3));
        assert_eq!(manager.indirect_count(), 0);
    }

    #[test]
    fn test_pin_manager_shared_indirect() {
        use bytes::Bytes;
        use ipfrs_core::Block;

        let manager = PinManager::new();

        // Create three different CIDs
        let block1 = Block::new(Bytes::from("data1")).expect("test: block creation should succeed");
        let block2 = Block::new(Bytes::from("data2")).expect("test: block creation should succeed");
        let block3 = Block::new(Bytes::from("data3")).expect("test: block creation should succeed");
        let cid1 = *block1.cid();
        let cid2 = *block2.cid();
        let cid3 = *block3.cid();

        // Pin cid1 and cid2 recursively, both reference cid3
        manager
            .pin(cid1, PinType::Recursive, None)
            .expect("test: pin should succeed");
        manager
            .pin(cid2, PinType::Recursive, None)
            .expect("test: pin should succeed");
        manager.add_indirect(cid1, cid3);
        manager.add_indirect(cid2, cid3);

        // Verify all are pinned
        assert!(manager.is_pinned(&cid1));
        assert!(manager.is_pinned(&cid2));
        assert!(manager.is_pinned(&cid3));

        // Unpin cid1
        manager
            .unpin(&cid1, true)
            .expect("test: unpin should succeed");

        // cid3 should still be pinned because cid2 references it
        assert!(!manager.is_pinned(&cid1));
        assert!(manager.is_pinned(&cid2));
        assert!(manager.is_pinned(&cid3));

        // Unpin cid2
        manager
            .unpin(&cid2, true)
            .expect("test: unpin should succeed");

        // Now cid3 should be unpinned
        assert!(!manager.is_pinned(&cid3));
        assert_eq!(manager.indirect_count(), 0);
    }

    #[test]
    fn test_pin_manager_clear() {
        let manager = PinManager::new();
        let cid = Cid::default();

        manager
            .pin(cid, PinType::Direct, None)
            .expect("test: pin should succeed");
        assert_eq!(manager.count(), 1);

        manager.clear();
        assert_eq!(manager.count(), 0);
        assert!(!manager.is_pinned(&cid));
    }

    #[tokio::test]
    async fn test_pin_manager_persistence() {
        let manager = PinManager::new();
        let cid = Cid::default();
        let temp_file =
            std::env::temp_dir().join(format!("ipfrs_pin_test_{}.bin", std::process::id()));

        // Pin and save
        manager
            .pin(cid, PinType::Direct, Some("test".to_string()))
            .expect("test: pin should succeed");
        manager
            .save(&temp_file)
            .await
            .expect("test: save should succeed");

        // Create new manager and load
        let manager2 = PinManager::new();
        manager2
            .load(&temp_file)
            .await
            .expect("test: load should succeed");

        // Verify pin was loaded
        assert!(manager2.is_pinned(&cid));
        let info = manager2
            .get(&cid)
            .expect("test: get pin should return info");
        assert_eq!(info.name, Some("test".to_string()));

        // Cleanup
        let _ = tokio::fs::remove_file(&temp_file).await;
    }
}
