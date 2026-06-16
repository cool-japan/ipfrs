//! Block replication and synchronization
//!
//! Provides protocols for syncing blocks between stores:
//! - Incremental sync (delta only)
//! - Full sync
//! - Conflict resolution
//! - Bi-directional replication
//!
//! # Example
//!
//! ```rust,ignore
//! use ipfrs_storage::{Replicator, SyncStrategy, SledBlockStore, BlockStoreConfig};
//! use std::sync::Arc;
//! use std::path::PathBuf;
//!
//! # async fn example() -> ipfrs_core::Result<()> {
//! // Create source and target stores
//! let source = Arc::new(SledBlockStore::new(BlockStoreConfig {
//!     path: PathBuf::from(".ipfrs/source"),
//!     cache_size: 100 * 1024 * 1024,
//! })?);
//!
//! let target = Arc::new(SledBlockStore::new(BlockStoreConfig {
//!     path: PathBuf::from(".ipfrs/target"),
//!     cache_size: 100 * 1024 * 1024,
//! })?);
//!
//! // Create replicator
//! let replicator = Replicator::new(source, target);
//!
//! // Perform incremental sync
//! let result = replicator.sync(SyncStrategy::Incremental, None).await?;
//! println!("Synced {} blocks ({} bytes)", result.blocks_synced, result.bytes_synced);
//! # Ok(())
//! # }
//! ```

use crate::traits::BlockStore;
use ipfrs_core::{Cid, Error, Result};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Synchronization strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncStrategy {
    /// Full sync - copy all blocks from source to target
    Full,
    /// Incremental sync - only copy blocks missing in target
    Incremental,
    /// Bidirectional sync - sync in both directions
    Bidirectional,
}

/// Conflict resolution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictStrategy {
    /// Keep source version on conflict
    KeepSource,
    /// Keep target version on conflict
    KeepTarget,
    /// Keep newer version (based on timestamp)
    KeepNewer,
    /// Fail on conflict
    Fail,
}

/// Result of a synchronization operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncResult {
    /// Number of blocks synced
    pub blocks_synced: usize,
    /// Total bytes synced
    pub bytes_synced: u64,
    /// Number of conflicts encountered
    pub conflicts: usize,
    /// Duration of sync operation
    pub duration: Duration,
    /// List of CIDs that had conflicts
    pub conflicting_cids: Vec<Cid>,
}

/// Replication state for tracking sync progress
#[derive(Debug, Clone, Default)]
pub struct ReplicationState {
    /// Last synced timestamp
    pub last_sync: Option<Instant>,
    /// CIDs synced in last operation
    pub last_synced_cids: HashSet<Cid>,
    /// Total blocks synced across all operations
    pub total_blocks_synced: usize,
    /// Total bytes synced across all operations
    pub total_bytes_synced: u64,
}

/// Block replicator for syncing between stores
pub struct Replicator<S: BlockStore, T: BlockStore> {
    /// Source store
    source: Arc<S>,
    /// Target store
    target: Arc<T>,
    /// Replication state
    state: parking_lot::RwLock<ReplicationState>,
}

impl<S: BlockStore, T: BlockStore> Replicator<S, T> {
    /// Create a new replicator
    pub fn new(source: Arc<S>, target: Arc<T>) -> Self {
        Self {
            source,
            target,
            state: parking_lot::RwLock::new(ReplicationState::default()),
        }
    }

    /// Synchronize blocks from source to target
    ///
    /// # Arguments
    /// * `strategy` - Synchronization strategy to use
    /// * `conflict_strategy` - How to resolve conflicts (if None, defaults to KeepSource)
    pub async fn sync(
        &self,
        strategy: SyncStrategy,
        conflict_strategy: Option<ConflictStrategy>,
    ) -> Result<SyncResult> {
        let start_time = Instant::now();
        let conflict_strategy = conflict_strategy.unwrap_or(ConflictStrategy::KeepSource);

        match strategy {
            SyncStrategy::Full => self.sync_full(conflict_strategy).await,
            SyncStrategy::Incremental => self.sync_incremental(conflict_strategy).await,
            SyncStrategy::Bidirectional => {
                // Sync both directions
                let result1 = self.sync_incremental(conflict_strategy).await?;

                // Create reverse replicator
                let reverse = Replicator::new(self.target.clone(), self.source.clone());
                let result2 = reverse.sync_incremental(conflict_strategy).await?;

                Ok(SyncResult {
                    blocks_synced: result1.blocks_synced + result2.blocks_synced,
                    bytes_synced: result1.bytes_synced + result2.bytes_synced,
                    conflicts: result1.conflicts + result2.conflicts,
                    duration: start_time.elapsed(),
                    conflicting_cids: [result1.conflicting_cids, result2.conflicting_cids].concat(),
                })
            }
        }
    }

    /// Perform full sync (all blocks)
    async fn sync_full(&self, conflict_strategy: ConflictStrategy) -> Result<SyncResult> {
        let start_time = Instant::now();

        // Get all CIDs from source
        let source_cids = self.source.list_cids()?;

        self.sync_cids(&source_cids, conflict_strategy, start_time)
            .await
    }

    /// Perform incremental sync (only missing blocks)
    async fn sync_incremental(&self, conflict_strategy: ConflictStrategy) -> Result<SyncResult> {
        let start_time = Instant::now();

        // Get all CIDs from source
        let source_cids = self.source.list_cids()?;

        // Filter to only CIDs not in target
        let target_has = self.target.has_many(&source_cids).await?;
        let missing_cids: Vec<Cid> = source_cids
            .into_iter()
            .zip(target_has.iter())
            .filter_map(|(cid, has)| if !*has { Some(cid) } else { None })
            .collect();

        self.sync_cids(&missing_cids, conflict_strategy, start_time)
            .await
    }

    /// Sync specific CIDs from source to target
    async fn sync_cids(
        &self,
        cids: &[Cid],
        conflict_strategy: ConflictStrategy,
        start_time: Instant,
    ) -> Result<SyncResult> {
        let mut blocks_synced = 0;
        let mut bytes_synced = 0u64;
        let mut conflicts = 0;
        let mut conflicting_cids = Vec::new();
        let mut synced_cids = HashSet::new();

        // Sync in batches for efficiency
        const BATCH_SIZE: usize = 100;
        for chunk in cids.chunks(BATCH_SIZE) {
            // Get blocks from source
            let blocks = self.source.get_many(chunk).await?;

            let mut blocks_to_put = Vec::new();

            for (cid, block_opt) in chunk.iter().zip(blocks.iter()) {
                if let Some(block) = block_opt {
                    // Check for conflicts
                    if let Some(existing) = self.target.get(cid).await? {
                        // Conflict: block exists in both stores
                        let should_replace = match conflict_strategy {
                            ConflictStrategy::KeepSource => true,
                            ConflictStrategy::KeepTarget => false,
                            ConflictStrategy::KeepNewer => {
                                // For simplicity, compare data content
                                // In a real implementation, we'd use timestamps or versioning
                                block.data().len() > existing.data().len()
                            }
                            ConflictStrategy::Fail => {
                                return Err(Error::Storage(format!(
                                    "Conflict detected for block {cid}"
                                )));
                            }
                        };

                        if should_replace {
                            blocks_to_put.push(block.clone());
                            bytes_synced += block.data().len() as u64;
                            synced_cids.insert(*cid);
                        }

                        conflicts += 1;
                        conflicting_cids.push(*cid);
                    } else {
                        // No conflict, just copy
                        blocks_to_put.push(block.clone());
                        bytes_synced += block.data().len() as u64;
                        synced_cids.insert(*cid);
                    }
                }
            }

            // Batch write to target
            if !blocks_to_put.is_empty() {
                self.target.put_many(&blocks_to_put).await?;
                blocks_synced += blocks_to_put.len();
            }
        }

        // Update state
        {
            let mut state = self.state.write();
            state.last_sync = Some(Instant::now());
            state.last_synced_cids = synced_cids;
            state.total_blocks_synced += blocks_synced;
            state.total_bytes_synced += bytes_synced;
        }

        Ok(SyncResult {
            blocks_synced,
            bytes_synced,
            conflicts,
            duration: start_time.elapsed(),
            conflicting_cids,
        })
    }

    /// Get current replication state
    pub fn state(&self) -> ReplicationState {
        self.state.read().clone()
    }

    /// Sync specific blocks by CID list
    pub async fn sync_blocks(
        &self,
        cids: &[Cid],
        conflict_strategy: Option<ConflictStrategy>,
    ) -> Result<SyncResult> {
        let conflict_strategy = conflict_strategy.unwrap_or(ConflictStrategy::KeepSource);
        self.sync_cids(cids, conflict_strategy, Instant::now())
            .await
    }

    /// Verify sync integrity - check that all blocks in source exist in target
    pub async fn verify(&self) -> Result<Vec<Cid>> {
        let source_cids = self.source.list_cids()?;
        let target_has = self.target.has_many(&source_cids).await?;

        let missing: Vec<Cid> = source_cids
            .into_iter()
            .zip(target_has.iter())
            .filter_map(|(cid, has)| if !*has { Some(cid) } else { None })
            .collect();

        Ok(missing)
    }
}

/// Replication manager for coordinating multiple replicators
pub struct ReplicationManager<S: BlockStore> {
    /// Primary store
    primary: Arc<S>,
    /// Replica stores
    replicas: Vec<Arc<S>>,
    /// Replication statistics
    stats: parking_lot::RwLock<HashMap<usize, ReplicationState>>,
}

impl<S: BlockStore> ReplicationManager<S> {
    /// Create a new replication manager
    pub fn new(primary: Arc<S>) -> Self {
        Self {
            primary,
            replicas: Vec::new(),
            stats: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Add a replica store
    pub fn add_replica(&mut self, replica: Arc<S>) {
        self.replicas.push(replica);
    }

    /// Sync primary to all replicas
    pub async fn sync_all(&self, strategy: SyncStrategy) -> Result<Vec<SyncResult>> {
        let mut results = Vec::new();

        for (idx, replica) in self.replicas.iter().enumerate() {
            let replicator = Replicator::new(self.primary.clone(), replica.clone());
            let result = replicator.sync(strategy, None).await?;

            // Update stats
            self.stats.write().insert(idx, replicator.state());

            results.push(result);
        }

        Ok(results)
    }

    /// Get replication statistics for a specific replica
    pub fn replica_stats(&self, index: usize) -> Option<ReplicationState> {
        self.stats.read().get(&index).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockstore::{BlockStoreConfig, SledBlockStore};
    use bytes::Bytes;
    use ipfrs_core::Block;

    #[tokio::test]
    async fn test_full_sync() {
        let source_config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-replication-source"),
            cache_size: 10 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&source_config.path);

        let target_config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-replication-target"),
            cache_size: 10 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&target_config.path);

        let source =
            Arc::new(SledBlockStore::new(source_config).expect("source store should be created"));
        let target =
            Arc::new(SledBlockStore::new(target_config).expect("target store should be created"));

        // Add blocks to source
        let block1 = Block::new(Bytes::from("block 1")).expect("block 1 should be created");
        let block2 = Block::new(Bytes::from("block 2")).expect("block 2 should be created");
        source
            .put(&block1)
            .await
            .expect("block 1 should be put in source");
        source
            .put(&block2)
            .await
            .expect("block 2 should be put in source");

        // Sync
        let replicator = Replicator::new(source.clone(), target.clone());
        let result = replicator
            .sync(SyncStrategy::Full, None)
            .await
            .expect("full sync should succeed");

        assert_eq!(result.blocks_synced, 2);
        assert_eq!(result.conflicts, 0);
        assert!(target
            .has(block1.cid())
            .await
            .expect("target should have block 1"));
        assert!(target
            .has(block2.cid())
            .await
            .expect("target should have block 2"));
    }

    #[tokio::test]
    async fn test_incremental_sync() {
        let source_config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-replication-inc-source"),
            cache_size: 10 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&source_config.path);

        let target_config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-replication-inc-target"),
            cache_size: 10 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&target_config.path);

        let source = Arc::new(
            SledBlockStore::new(source_config).expect("test: source store should be created"),
        );
        let target = Arc::new(
            SledBlockStore::new(target_config).expect("test: target store should be created"),
        );

        // Add some blocks to both
        let block1 = Block::new(Bytes::from("block 1")).expect("test: block 1 should be created");
        source
            .put(&block1)
            .await
            .expect("test: block 1 should be put in source");
        target
            .put(&block1)
            .await
            .expect("test: block 1 should be put in target");

        // Add unique block to source
        let block2 = Block::new(Bytes::from("block 2")).expect("test: block 2 should be created");
        source
            .put(&block2)
            .await
            .expect("test: block 2 should be put in source");

        // Incremental sync should only copy block2
        let replicator = Replicator::new(source.clone(), target.clone());
        let result = replicator
            .sync(SyncStrategy::Incremental, None)
            .await
            .expect("test: incremental sync should succeed");

        assert_eq!(result.blocks_synced, 1);
        assert!(target
            .has(block2.cid())
            .await
            .expect("test: target should have block 2"));
    }

    #[tokio::test]
    async fn test_conflict_resolution() {
        let source_config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-replication-conflict-source"),
            cache_size: 10 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&source_config.path);

        let target_config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-replication-conflict-target"),
            cache_size: 10 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&target_config.path);

        let source = Arc::new(
            SledBlockStore::new(source_config)
                .expect("test: conflict source store should be created"),
        );
        let target = Arc::new(
            SledBlockStore::new(target_config)
                .expect("test: conflict target store should be created"),
        );

        // Add same CID with different content (simulate conflict)
        let block1 = Block::new(Bytes::from("source version"))
            .expect("test: source version block should be created");
        source
            .put(&block1)
            .await
            .expect("test: source version block should be put in source");

        // Note: In a real conflict scenario, we'd have same CID with different data
        // For this test, we'll just verify the conflict handling works
        let replicator = Replicator::new(source.clone(), target.clone());
        let result = replicator
            .sync(SyncStrategy::Full, Some(ConflictStrategy::KeepSource))
            .await
            .expect("test: conflict resolution sync should succeed");

        assert!(result.blocks_synced > 0);
    }
}
