//! Eventual consistency support for distributed storage.
//!
//! Provides configurable consistency levels, version vectors, and conflict resolution
//! for eventually consistent reads and writes.
//!
//! # Example
//!
//! ```rust,ignore
//! use ipfrs_storage::eventual_consistency::{ConsistencyLevel, EventualStore};
//!
//! let store = EventualStore::new(base_store, ConsistencyLevel::Quorum { read: 2, write: 2 });
//! ```

use crate::traits::BlockStore;
use async_trait::async_trait;
use bytes::Bytes;
use dashmap::DashMap;
use ipfrs_core::{Block, Cid, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Consistency level for read and write operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConsistencyLevel {
    /// Strong consistency - all replicas must agree
    Strong,
    /// Eventual consistency - accept stale reads, async replication
    #[default]
    Eventual,
    /// Quorum-based consistency - R + W > N for linearizability
    Quorum {
        read_quorum: usize,
        write_quorum: usize,
    },
    /// One - only one replica needs to respond (fastest, least consistent)
    One,
}

/// Version vector for tracking causality
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionVector {
    /// Mapping of node ID to version number
    versions: HashMap<u64, u64>,
}

impl VersionVector {
    /// Create a new empty version vector
    pub fn new() -> Self {
        Self {
            versions: HashMap::new(),
        }
    }

    /// Increment the version for a node
    pub fn increment(&mut self, node_id: u64) {
        let version = self.versions.entry(node_id).or_insert(0);
        *version += 1;
    }

    /// Get the version for a node
    pub fn get(&self, node_id: u64) -> u64 {
        *self.versions.get(&node_id).unwrap_or(&0)
    }

    /// Check if this version vector happens before another
    pub fn happens_before(&self, other: &VersionVector) -> bool {
        let mut strictly_less = false;

        for (node_id, version) in &self.versions {
            let other_version = other.get(*node_id);
            if *version > other_version {
                return false; // Not happens-before if any version is greater
            }
            if *version < other_version {
                strictly_less = true;
            }
        }

        // Check for nodes in other but not in self
        for (node_id, version) in &other.versions {
            if !self.versions.contains_key(node_id) && *version > 0 {
                strictly_less = true;
            }
        }

        strictly_less
    }

    /// Check if two version vectors are concurrent (conflicting)
    pub fn is_concurrent(&self, other: &VersionVector) -> bool {
        !self.happens_before(other) && !other.happens_before(self) && self != other
    }

    /// Merge two version vectors (take the maximum of each component)
    pub fn merge(&mut self, other: &VersionVector) {
        for (node_id, version) in &other.versions {
            let current = self.versions.entry(*node_id).or_insert(0);
            *current = (*current).max(*version);
        }
    }
}

impl Default for VersionVector {
    fn default() -> Self {
        Self::new()
    }
}

/// Versioned value with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedValue {
    /// The actual data
    pub data: Vec<u8>,
    /// Version vector for causality tracking
    pub version: VersionVector,
    /// Timestamp of last write
    pub timestamp: u64,
    /// Node ID that performed the write
    pub writer_node_id: u64,
}

impl VersionedValue {
    /// Create a new versioned value
    pub fn new(data: Vec<u8>, node_id: u64) -> Self {
        let mut version = VersionVector::new();
        version.increment(node_id);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time is after UNIX epoch")
            .as_millis() as u64;

        Self {
            data,
            version,
            timestamp,
            writer_node_id: node_id,
        }
    }

    /// Check if this value is newer than another (for last-write-wins)
    pub fn is_newer_than(&self, other: &VersionedValue) -> bool {
        if self.timestamp != other.timestamp {
            self.timestamp > other.timestamp
        } else {
            // Break ties using writer node ID
            self.writer_node_id > other.writer_node_id
        }
    }
}

/// Conflict resolution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConflictResolution {
    /// Last-write-wins based on timestamp
    #[default]
    LastWriteWins,
    /// Keep all conflicting versions (application must resolve)
    KeepAll,
    /// Use version vectors to detect causality
    VectorClock,
}

/// Eventually consistent block store wrapper
pub struct EventualStore<S: BlockStore> {
    /// Underlying block store
    inner: Arc<S>,
    /// Consistency level for operations
    consistency_level: ConsistencyLevel,
    /// Conflict resolution strategy
    conflict_resolution: ConflictResolution,
    /// Local node ID
    node_id: u64,
    /// Versioned values cache
    versions: Arc<DashMap<Cid, VersionedValue>>,
}

impl<S: BlockStore> EventualStore<S> {
    /// Create a new eventually consistent store
    pub fn new(
        inner: S,
        consistency_level: ConsistencyLevel,
        conflict_resolution: ConflictResolution,
        node_id: u64,
    ) -> Self {
        Self {
            inner: Arc::new(inner),
            consistency_level,
            conflict_resolution,
            node_id,
            versions: Arc::new(DashMap::new()),
        }
    }

    /// Get the consistency level
    pub fn consistency_level(&self) -> ConsistencyLevel {
        self.consistency_level
    }

    /// Set the consistency level
    pub fn set_consistency_level(&mut self, level: ConsistencyLevel) {
        self.consistency_level = level;
    }

    /// Resolve conflicts between two versioned values
    fn resolve_conflict(&self, v1: &VersionedValue, v2: &VersionedValue) -> VersionedValue {
        match self.conflict_resolution {
            ConflictResolution::LastWriteWins => {
                if v1.is_newer_than(v2) {
                    v1.clone()
                } else {
                    v2.clone()
                }
            }
            ConflictResolution::VectorClock => {
                // Use version vector to determine causality
                if v1.version.happens_before(&v2.version) {
                    v2.clone()
                } else if v2.version.happens_before(&v1.version) {
                    v1.clone()
                } else {
                    // Concurrent - fall back to last-write-wins
                    if v1.is_newer_than(v2) {
                        v1.clone()
                    } else {
                        v2.clone()
                    }
                }
            }
            ConflictResolution::KeepAll => {
                // For now, just keep the newer one
                // In a real implementation, we'd return both
                if v1.is_newer_than(v2) {
                    v1.clone()
                } else {
                    v2.clone()
                }
            }
        }
    }

    /// Store a versioned value
    pub async fn put_versioned(&self, cid: Cid, value: VersionedValue) -> Result<()> {
        // Check for conflicts
        if let Some(existing) = self.versions.get(&cid) {
            let resolved = self.resolve_conflict(&existing, &value);
            self.versions.insert(cid, resolved.clone());
            let block = Block::new(Bytes::from(resolved.data))?;
            self.inner.put(&block).await?;
        } else {
            self.versions.insert(cid, value.clone());
            let block = Block::new(Bytes::from(value.data))?;
            self.inner.put(&block).await?;
        }

        Ok(())
    }

    /// Get a versioned value
    pub async fn get_versioned(&self, cid: &Cid) -> Result<Option<VersionedValue>> {
        match self.consistency_level {
            ConsistencyLevel::Eventual | ConsistencyLevel::One => {
                // Return local value if available
                if let Some(value) = self.versions.get(cid) {
                    Ok(Some(value.clone()))
                } else if let Some(block) = self.inner.get(cid).await? {
                    // Reconstruct versioned value from block
                    let value = VersionedValue::new(block.data().to_vec(), self.node_id);
                    self.versions.insert(*cid, value.clone());
                    Ok(Some(value))
                } else {
                    Ok(None)
                }
            }
            ConsistencyLevel::Strong | ConsistencyLevel::Quorum { .. } => {
                // For strong consistency, we'd need to query multiple replicas
                // For now, just return local value
                if let Some(value) = self.versions.get(cid) {
                    Ok(Some(value.clone()))
                } else if let Some(block) = self.inner.get(cid).await? {
                    let value = VersionedValue::new(block.data().to_vec(), self.node_id);
                    self.versions.insert(*cid, value.clone());
                    Ok(Some(value))
                } else {
                    Ok(None)
                }
            }
        }
    }

    /// Get the version vector for a CID
    pub fn get_version(&self, cid: &Cid) -> Option<VersionVector> {
        self.versions.get(cid).map(|v| v.version.clone())
    }

    /// Get statistics about the store
    pub fn stats(&self) -> EventualStoreStats {
        EventualStoreStats {
            total_versioned_entries: self.versions.len(),
            consistency_level: self.consistency_level,
            conflict_resolution: self.conflict_resolution,
        }
    }
}

/// Statistics for eventually consistent store
#[derive(Debug, Clone)]
pub struct EventualStoreStats {
    /// Total number of versioned entries
    pub total_versioned_entries: usize,
    /// Current consistency level
    pub consistency_level: ConsistencyLevel,
    /// Conflict resolution strategy
    pub conflict_resolution: ConflictResolution,
}

#[async_trait]
impl<S: BlockStore> BlockStore for EventualStore<S> {
    async fn get(&self, cid: &Cid) -> Result<Option<Block>> {
        if let Some(versioned) = self.get_versioned(cid).await? {
            let block = Block::new(Bytes::from(versioned.data))?;
            Ok(Some(block))
        } else {
            Ok(None)
        }
    }

    async fn put(&self, block: &Block) -> Result<()> {
        let value = VersionedValue::new(block.data().to_vec(), self.node_id);
        self.put_versioned(*block.cid(), value).await
    }

    async fn has(&self, cid: &Cid) -> Result<bool> {
        if self.versions.contains_key(cid) {
            Ok(true)
        } else {
            self.inner.has(cid).await
        }
    }

    async fn delete(&self, cid: &Cid) -> Result<()> {
        self.versions.remove(cid);
        self.inner.delete(cid).await
    }

    fn list_cids(&self) -> Result<Vec<Cid>> {
        self.inner.list_cids()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    async fn flush(&self) -> Result<()> {
        self.inner.flush().await
    }

    async fn get_many(&self, cids: &[Cid]) -> Result<Vec<Option<Block>>> {
        let mut results = Vec::with_capacity(cids.len());
        for cid in cids {
            results.push(self.get(cid).await?);
        }
        Ok(results)
    }

    async fn put_many(&self, blocks: &[Block]) -> Result<()> {
        for block in blocks {
            self.put(block).await?;
        }
        Ok(())
    }

    async fn has_many(&self, cids: &[Cid]) -> Result<Vec<bool>> {
        let mut results = Vec::with_capacity(cids.len());
        for cid in cids {
            results.push(self.has(cid).await?);
        }
        Ok(results)
    }

    async fn delete_many(&self, cids: &[Cid]) -> Result<()> {
        for cid in cids {
            self.delete(cid).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_vector_happens_before() {
        let mut v1 = VersionVector::new();
        v1.increment(1);
        v1.increment(2);

        let mut v2 = VersionVector::new();
        v2.increment(1);
        v2.increment(2);
        v2.increment(2);

        assert!(v1.happens_before(&v2));
        assert!(!v2.happens_before(&v1));
    }

    #[test]
    fn test_version_vector_concurrent() {
        let mut v1 = VersionVector::new();
        v1.increment(1);
        v1.increment(1);

        let mut v2 = VersionVector::new();
        v2.increment(2);
        v2.increment(2);

        assert!(v1.is_concurrent(&v2));
        assert!(v2.is_concurrent(&v1));
    }

    #[test]
    fn test_version_vector_merge() {
        let mut v1 = VersionVector::new();
        v1.increment(1);
        v1.increment(1);

        let mut v2 = VersionVector::new();
        v2.increment(2);
        v2.increment(2);

        v1.merge(&v2);

        assert_eq!(v1.get(1), 2);
        assert_eq!(v1.get(2), 2);
    }

    #[test]
    fn test_versioned_value_newer() {
        let v1 = VersionedValue::new(vec![1, 2, 3], 1);
        std::thread::sleep(std::time::Duration::from_millis(10));
        let v2 = VersionedValue::new(vec![4, 5, 6], 2);

        assert!(v2.is_newer_than(&v1));
        assert!(!v1.is_newer_than(&v2));
    }

    #[test]
    fn test_consistency_levels() {
        assert_eq!(ConsistencyLevel::default(), ConsistencyLevel::Eventual);
        assert_eq!(
            ConsistencyLevel::Quorum {
                read_quorum: 2,
                write_quorum: 2
            },
            ConsistencyLevel::Quorum {
                read_quorum: 2,
                write_quorum: 2
            }
        );
    }
}
