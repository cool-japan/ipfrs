//! Pin management for preventing garbage collection of important blocks.
//!
//! Pinning allows users to mark blocks as important, preventing them from
//! being garbage collected. Supports both direct pins (single block) and
//! recursive pins (entire DAG).
//!
//! # Pin Types
//!
//! - **Direct**: Only pins the specific block
//! - **Recursive**: Pins the block and all blocks it references (entire DAG)
//! - **Indirect**: Block is pinned because it's referenced by a recursively pinned block

use dashmap::DashMap;
use ipfrs_core::{Cid, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Type of pin
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PinType {
    /// Direct pin - only this specific block
    Direct,
    /// Recursive pin - this block and all referenced blocks
    Recursive,
    /// Indirect pin - pinned via a recursive pin (internal use)
    Indirect,
}

/// Metadata about a pinned block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinInfo {
    /// Type of pin
    pub pin_type: PinType,
    /// Reference count (how many times this CID is pinned)
    pub ref_count: u32,
    /// When the pin was created (Unix timestamp)
    pub created_at: u64,
    /// Optional name/label for the pin
    pub name: Option<String>,
    /// For indirect pins, the parent that caused this pin
    pub pinned_by: Option<Vec<u8>>, // Serialized CID
}

impl PinInfo {
    fn new(pin_type: PinType) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            pin_type,
            ref_count: 1,
            created_at: now,
            name: None,
            pinned_by: None,
        }
    }

    fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    fn with_parent(mut self, parent: &Cid) -> Self {
        self.pinned_by = Some(parent.to_bytes());
        self
    }
}

/// Pin manager for tracking pinned blocks
pub struct PinManager {
    /// Map of CID -> PinInfo
    pins: DashMap<Vec<u8>, PinInfo>,
    /// Statistics
    stats: PinStats,
}

/// Statistics about pins
#[derive(Debug, Default)]
pub struct PinStats {
    /// Total number of pins
    total_pins: AtomicU64,
    /// Number of direct pins
    direct_pins: AtomicU64,
    /// Number of recursive pins
    recursive_pins: AtomicU64,
    /// Number of indirect pins
    indirect_pins: AtomicU64,
}

impl PinStats {
    fn increment(&self, pin_type: PinType) {
        self.total_pins.fetch_add(1, Ordering::Relaxed);
        match pin_type {
            PinType::Direct => self.direct_pins.fetch_add(1, Ordering::Relaxed),
            PinType::Recursive => self.recursive_pins.fetch_add(1, Ordering::Relaxed),
            PinType::Indirect => self.indirect_pins.fetch_add(1, Ordering::Relaxed),
        };
    }

    fn decrement(&self, pin_type: PinType) {
        self.total_pins.fetch_sub(1, Ordering::Relaxed);
        match pin_type {
            PinType::Direct => self.direct_pins.fetch_sub(1, Ordering::Relaxed),
            PinType::Recursive => self.recursive_pins.fetch_sub(1, Ordering::Relaxed),
            PinType::Indirect => self.indirect_pins.fetch_sub(1, Ordering::Relaxed),
        };
    }

    /// Get a snapshot of the statistics
    pub fn snapshot(&self) -> PinStatsSnapshot {
        PinStatsSnapshot {
            total_pins: self.total_pins.load(Ordering::Relaxed),
            direct_pins: self.direct_pins.load(Ordering::Relaxed),
            recursive_pins: self.recursive_pins.load(Ordering::Relaxed),
            indirect_pins: self.indirect_pins.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of pin statistics
#[derive(Debug, Clone)]
pub struct PinStatsSnapshot {
    pub total_pins: u64,
    pub direct_pins: u64,
    pub recursive_pins: u64,
    pub indirect_pins: u64,
}

impl PinManager {
    /// Create a new pin manager
    pub fn new() -> Self {
        Self {
            pins: DashMap::new(),
            stats: PinStats::default(),
        }
    }

    /// Pin a block directly (single block only)
    pub fn pin(&self, cid: &Cid) -> Result<()> {
        self.pin_with_type(cid, PinType::Direct, None)
    }

    /// Pin a block with a name
    pub fn pin_named(&self, cid: &Cid, name: String) -> Result<()> {
        self.pin_with_type(cid, PinType::Direct, Some(name))
    }

    /// Pin a block with specific type
    fn pin_with_type(&self, cid: &Cid, pin_type: PinType, name: Option<String>) -> Result<()> {
        let key = cid.to_bytes();

        self.pins
            .entry(key)
            .and_modify(|info| {
                info.ref_count += 1;
                // Upgrade pin type if needed (direct -> recursive)
                if pin_type == PinType::Recursive && info.pin_type == PinType::Direct {
                    self.stats.decrement(PinType::Direct);
                    self.stats.increment(PinType::Recursive);
                    info.pin_type = PinType::Recursive;
                }
            })
            .or_insert_with(|| {
                self.stats.increment(pin_type);
                let mut info = PinInfo::new(pin_type);
                if let Some(n) = name {
                    info = info.with_name(n);
                }
                info
            });

        Ok(())
    }

    /// Pin a block recursively (pins all referenced blocks)
    ///
    /// The `link_resolver` function should return all CIDs that this block links to.
    pub fn pin_recursive<F>(&self, cid: &Cid, link_resolver: F) -> Result<usize>
    where
        F: Fn(&Cid) -> Result<Vec<Cid>>,
    {
        let mut pinned_count = 0;
        let mut to_process = vec![*cid];
        let mut seen = HashSet::new();

        // Pin the root as recursive
        self.pin_with_type(cid, PinType::Recursive, None)?;
        pinned_count += 1;
        seen.insert(*cid);

        // Process all linked blocks
        while let Some(current_cid) = to_process.pop() {
            let links = link_resolver(&current_cid)?;

            for link_cid in links {
                if seen.insert(link_cid) {
                    // Pin as indirect (pinned by the root)
                    self.pin_indirect(&link_cid, cid)?;
                    pinned_count += 1;
                    to_process.push(link_cid);
                }
            }
        }

        Ok(pinned_count)
    }

    /// Pin a block indirectly (used internally for recursive pins)
    fn pin_indirect(&self, cid: &Cid, parent: &Cid) -> Result<()> {
        let key = cid.to_bytes();

        self.pins
            .entry(key)
            .and_modify(|info| {
                info.ref_count += 1;
            })
            .or_insert_with(|| {
                self.stats.increment(PinType::Indirect);
                PinInfo::new(PinType::Indirect).with_parent(parent)
            });

        Ok(())
    }

    /// Unpin a block
    pub fn unpin(&self, cid: &Cid) -> Result<bool> {
        let key = cid.to_bytes();

        let mut removed = false;
        self.pins.remove_if(&key, |_, info| {
            if info.ref_count <= 1 {
                self.stats.decrement(info.pin_type);
                removed = true;
                true // Remove the entry
            } else {
                false // Keep the entry, just decrement
            }
        });

        if !removed {
            // Decrement ref count if entry wasn't removed
            if let Some(mut entry) = self.pins.get_mut(&key) {
                entry.ref_count -= 1;
            }
        }

        Ok(removed)
    }

    /// Unpin a block recursively
    pub fn unpin_recursive<F>(&self, cid: &Cid, link_resolver: F) -> Result<usize>
    where
        F: Fn(&Cid) -> Result<Vec<Cid>>,
    {
        let mut unpinned_count = 0;
        let mut to_process = vec![*cid];
        let mut seen = HashSet::new();

        while let Some(current_cid) = to_process.pop() {
            if !seen.insert(current_cid) {
                continue;
            }

            if self.unpin(&current_cid)? {
                unpinned_count += 1;
            }

            // Get links before unpinning might remove info
            if let Ok(links) = link_resolver(&current_cid) {
                to_process.extend(links);
            }
        }

        Ok(unpinned_count)
    }

    /// Check if a block is pinned
    pub fn is_pinned(&self, cid: &Cid) -> bool {
        self.pins.contains_key(&cid.to_bytes())
    }

    /// Get pin info for a block
    pub fn get_pin_info(&self, cid: &Cid) -> Option<PinInfo> {
        self.pins.get(&cid.to_bytes()).map(|r| r.clone())
    }

    /// List all pinned CIDs
    pub fn list_pins(&self) -> Result<Vec<(Cid, PinInfo)>> {
        let mut result = Vec::new();
        for entry in self.pins.iter() {
            let cid = Cid::try_from(entry.key().clone())
                .map_err(|e| Error::Cid(format!("Invalid CID: {e}")))?;
            result.push((cid, entry.value().clone()));
        }
        Ok(result)
    }

    /// List pins of a specific type
    pub fn list_pins_by_type(&self, pin_type: PinType) -> Result<Vec<Cid>> {
        let mut result = Vec::new();
        for entry in self.pins.iter() {
            if entry.value().pin_type == pin_type {
                let cid = Cid::try_from(entry.key().clone())
                    .map_err(|e| Error::Cid(format!("Invalid CID: {e}")))?;
                result.push(cid);
            }
        }
        Ok(result)
    }

    /// Get statistics
    pub fn stats(&self) -> PinStatsSnapshot {
        self.stats.snapshot()
    }

    /// Save pins to a file
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let pins: HashMap<Vec<u8>, PinInfo> = self
            .pins
            .iter()
            .map(|r| (r.key().clone(), r.value().clone()))
            .collect();

        let data = oxicode::serde::encode_to_vec(&pins, oxicode::config::standard())
            .map_err(|e| Error::Serialization(format!("Failed to serialize pins: {e}")))?;

        std::fs::write(path, data)
            .map_err(|e| Error::Storage(format!("Failed to write pins: {e}")))?;

        Ok(())
    }

    /// Load pins from a file
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let data =
            std::fs::read(path).map_err(|e| Error::Storage(format!("Failed to read pins: {e}")))?;

        let pins: HashMap<Vec<u8>, PinInfo> =
            oxicode::serde::decode_owned_from_slice(&data, oxicode::config::standard())
                .map(|(v, _)| v)
                .map_err(|e| Error::Deserialization(format!("Failed to deserialize pins: {e}")))?;

        let manager = Self::new();

        for (key, info) in pins {
            manager.stats.increment(info.pin_type);
            manager.pins.insert(key, info);
        }

        Ok(manager)
    }

    /// Clear all pins
    pub fn clear(&self) {
        self.pins.clear();
        self.stats.total_pins.store(0, Ordering::Relaxed);
        self.stats.direct_pins.store(0, Ordering::Relaxed);
        self.stats.recursive_pins.store(0, Ordering::Relaxed);
        self.stats.indirect_pins.store(0, Ordering::Relaxed);
    }
}

impl Default for PinManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Pin set - a named collection of pins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinSet {
    /// Name of the pin set
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// CIDs in this set
    pub cids: Vec<Vec<u8>>,
    /// Created timestamp
    pub created_at: u64,
}

impl PinSet {
    /// Create a new pin set
    pub fn new(name: String) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            name,
            description: None,
            cids: Vec::new(),
            created_at: now,
        }
    }

    /// Add a CID to the set
    pub fn add(&mut self, cid: &Cid) {
        let bytes = cid.to_bytes();
        if !self.cids.contains(&bytes) {
            self.cids.push(bytes);
        }
    }

    /// Remove a CID from the set
    pub fn remove(&mut self, cid: &Cid) {
        let bytes = cid.to_bytes();
        self.cids.retain(|c| c != &bytes);
    }

    /// Check if set contains a CID
    pub fn contains(&self, cid: &Cid) -> bool {
        let bytes = cid.to_bytes();
        self.cids.contains(&bytes)
    }

    /// Get all CIDs in the set
    pub fn list_cids(&self) -> Result<Vec<Cid>> {
        self.cids
            .iter()
            .map(|bytes| {
                Cid::try_from(bytes.clone()).map_err(|e| Error::Cid(format!("Invalid CID: {e}")))
            })
            .collect()
    }

    /// Number of items in the set
    pub fn len(&self) -> usize {
        self.cids.len()
    }

    /// Check if set is empty
    pub fn is_empty(&self) -> bool {
        self.cids.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use ipfrs_core::Block;

    fn make_test_cid(data: &[u8]) -> Cid {
        let block = Block::new(Bytes::copy_from_slice(data)).unwrap();
        *block.cid()
    }

    #[test]
    fn test_pin_unpin() {
        let manager = PinManager::new();
        let cid = make_test_cid(b"test block");

        // Pin
        manager.pin(&cid).unwrap();
        assert!(manager.is_pinned(&cid));

        // Check stats
        let stats = manager.stats();
        assert_eq!(stats.total_pins, 1);
        assert_eq!(stats.direct_pins, 1);

        // Unpin
        manager.unpin(&cid).unwrap();
        assert!(!manager.is_pinned(&cid));

        let stats = manager.stats();
        assert_eq!(stats.total_pins, 0);
    }

    #[test]
    fn test_pin_refcount() {
        let manager = PinManager::new();
        let cid = make_test_cid(b"test block");

        // Pin twice
        manager.pin(&cid).unwrap();
        manager.pin(&cid).unwrap();

        let info = manager.get_pin_info(&cid).unwrap();
        assert_eq!(info.ref_count, 2);

        // Unpin once - should still be pinned
        manager.unpin(&cid).unwrap();
        assert!(manager.is_pinned(&cid));

        // Unpin again - should be removed
        manager.unpin(&cid).unwrap();
        assert!(!manager.is_pinned(&cid));
    }

    #[test]
    fn test_list_pins_by_type() {
        let manager = PinManager::new();
        let cid1 = make_test_cid(b"block1");
        let cid2 = make_test_cid(b"block2");

        manager.pin(&cid1).unwrap();
        manager
            .pin_with_type(&cid2, PinType::Recursive, None)
            .unwrap();

        let direct = manager.list_pins_by_type(PinType::Direct).unwrap();
        assert_eq!(direct.len(), 1);
        assert_eq!(direct[0], cid1);

        let recursive = manager.list_pins_by_type(PinType::Recursive).unwrap();
        assert_eq!(recursive.len(), 1);
        assert_eq!(recursive[0], cid2);
    }

    #[test]
    fn test_pin_set() {
        let mut set = PinSet::new("test".to_string());
        let cid1 = make_test_cid(b"block1");
        let cid2 = make_test_cid(b"block2");

        set.add(&cid1);
        set.add(&cid2);
        assert_eq!(set.len(), 2);
        assert!(set.contains(&cid1));

        set.remove(&cid1);
        assert!(!set.contains(&cid1));
        assert_eq!(set.len(), 1);
    }
}
