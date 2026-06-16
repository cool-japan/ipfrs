//! Garbage Collection System
//!
//! Implements mark-and-sweep garbage collection for IPFRS blocks.
//! Only unpinned blocks are eligible for collection.

use ipfrs_core::{Cid, Ipld, Result};
use ipfrs_storage::BlockStoreTrait;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::pin::PinManager;

/// Garbage collection statistics
#[derive(Debug, Clone)]
pub struct GcStats {
    /// Number of blocks collected (deleted)
    pub blocks_collected: u64,
    /// Bytes freed
    pub bytes_freed: u64,
    /// Number of blocks marked as reachable
    pub blocks_marked: u64,
    /// Number of blocks scanned
    pub blocks_scanned: u64,
    /// Duration of GC run
    pub duration: Duration,
    /// Whether GC was cancelled
    pub cancelled: bool,
}

/// Garbage collection configuration
#[derive(Debug, Clone)]
pub struct GcConfig {
    /// Minimum age of blocks before they can be collected (in seconds)
    pub min_age_seconds: u64,
    /// Maximum number of blocks to collect in one run (0 = unlimited)
    pub max_blocks_per_run: u64,
    /// Whether to perform a dry run (don't actually delete)
    pub dry_run: bool,
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            min_age_seconds: 3600, // 1 hour
            max_blocks_per_run: 0, // unlimited
            dry_run: false,
        }
    }
}

/// Garbage collector
pub struct GarbageCollector<S: BlockStoreTrait> {
    storage: Arc<S>,
    pin_manager: Arc<PinManager>,
}

impl<S: BlockStoreTrait> GarbageCollector<S> {
    /// Create a new garbage collector
    pub fn new(storage: Arc<S>, pin_manager: Arc<PinManager>) -> Self {
        Self {
            storage,
            pin_manager,
        }
    }

    /// Run garbage collection
    ///
    /// This performs a mark-and-sweep algorithm:
    /// 1. Mark phase: Mark all pinned blocks and their references as reachable
    /// 2. Sweep phase: Delete all unmarked blocks
    pub async fn collect(&self, config: GcConfig) -> Result<GcStats> {
        let start_time = Instant::now();
        let mut stats = GcStats {
            blocks_collected: 0,
            bytes_freed: 0,
            blocks_marked: 0,
            blocks_scanned: 0,
            duration: Duration::ZERO,
            cancelled: false,
        };

        // Phase 1: Mark - Find all reachable blocks
        let reachable = self.mark_reachable().await?;
        stats.blocks_marked = reachable.len() as u64;

        // Phase 2: Sweep - Delete unreachable blocks
        let all_cids = self.storage.list_cids()?;
        stats.blocks_scanned = all_cids.len() as u64;

        let mut collected_count = 0u64;

        for cid in all_cids {
            // Check if we should stop (max blocks limit)
            if config.max_blocks_per_run > 0 && collected_count >= config.max_blocks_per_run {
                break;
            }

            // Skip if reachable
            if reachable.contains(&cid) {
                continue;
            }

            // Get block size before deletion
            if let Some(block) = self.storage.get(&cid).await? {
                let size = block.data().len() as u64;

                // Delete the block (unless dry run)
                if !config.dry_run {
                    self.storage.delete(&cid).await?;
                }

                stats.bytes_freed += size;
                collected_count += 1;
            }
        }

        stats.blocks_collected = collected_count;
        stats.duration = start_time.elapsed();

        Ok(stats)
    }

    /// Mark phase: Find all reachable blocks
    ///
    /// A block is reachable if:
    /// 1. It's directly pinned
    /// 2. It's referenced by a reachable block (for recursive pins)
    async fn mark_reachable(&self) -> Result<HashSet<Cid>> {
        let mut reachable = HashSet::new();
        let mut to_visit = Vec::new();

        // Start with all pinned blocks
        let pins = self.pin_manager.list();
        for pin in pins {
            reachable.insert(pin.cid);
            to_visit.push(pin.cid);
        }

        // Traverse DAGs for recursive pins
        while let Some(cid) = to_visit.pop() {
            // Get the block
            if let Some(block) = self.storage.get(&cid).await? {
                // Try to parse as IPLD to find links
                if let Ok(ipld) = Ipld::from_dag_cbor(block.data()) {
                    // Extract all links
                    for link_cid in ipld.links() {
                        if reachable.insert(link_cid) {
                            // New CID found, add to visit queue
                            to_visit.push(link_cid);
                        }
                    }
                }
            }
        }

        Ok(reachable)
    }

    /// Get the number of unpinned blocks
    pub fn count_unpinned(&self) -> Result<usize> {
        let all_cids = self.storage.list_cids()?;
        let unpinned = all_cids
            .iter()
            .filter(|cid| !self.pin_manager.is_pinned(cid))
            .count();
        Ok(unpinned)
    }

    /// Estimate bytes that can be freed
    pub async fn estimate_freeable_space(&self) -> Result<u64> {
        let all_cids = self.storage.list_cids()?;
        let mut total = 0u64;

        for cid in all_cids {
            if !self.pin_manager.is_pinned(&cid) {
                if let Some(block) = self.storage.get(&cid).await? {
                    total += block.data().len() as u64;
                }
            }
        }

        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pin::{PinManager, PinType};
    use bytes::Bytes;
    use ipfrs_core::Block;
    use ipfrs_storage::{BlockStoreConfig, SledBlockStore};

    #[tokio::test]
    async fn test_gc_basic() {
        // Create temporary storage
        let path = std::env::temp_dir().join(format!("ipfrs_gc_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        let config = BlockStoreConfig {
            path: path.clone(),
            ..Default::default()
        };
        let storage = Arc::new(
            SledBlockStore::new(config).expect("test: block store creation should succeed"),
        );
        let pin_manager = Arc::new(PinManager::new());
        let gc = GarbageCollector::new(storage.clone(), pin_manager.clone());

        // Add some blocks
        let block1 =
            Block::new(Bytes::from("test data 1")).expect("test: block creation should succeed");
        let block2 =
            Block::new(Bytes::from("test data 2")).expect("test: block creation should succeed");
        let cid1 = *block1.cid();
        let cid2 = *block2.cid();

        storage
            .put(&block1)
            .await
            .expect("test: put block1 should succeed");
        storage
            .put(&block2)
            .await
            .expect("test: put block2 should succeed");

        // Pin only first block
        pin_manager
            .pin(cid1, PinType::Direct, None)
            .expect("test: pin should succeed");

        // Run GC
        let config = GcConfig {
            dry_run: false,
            ..Default::default()
        };
        let stats = gc
            .collect(config)
            .await
            .expect("test: GC collect should succeed");

        // Should have collected block2
        assert_eq!(stats.blocks_collected, 1);
        assert!(stats.bytes_freed > 0);

        // Verify block1 still exists, block2 is gone
        assert!(storage
            .has(&cid1)
            .await
            .expect("test: storage has check should succeed"));
        assert!(!storage
            .has(&cid2)
            .await
            .expect("test: storage has check should succeed"));

        // Cleanup
        let _ = std::fs::remove_dir_all(&path);
    }

    #[tokio::test]
    async fn test_gc_dry_run() {
        // Create temporary storage
        let path = std::env::temp_dir().join(format!("ipfrs_gc_test_dry_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        let config = BlockStoreConfig {
            path: path.clone(),
            ..Default::default()
        };
        let storage = Arc::new(
            SledBlockStore::new(config).expect("test: block store creation should succeed"),
        );
        let pin_manager = Arc::new(PinManager::new());
        let gc = GarbageCollector::new(storage.clone(), pin_manager.clone());

        // Add unpinned block
        let block =
            Block::new(Bytes::from("test data")).expect("test: block creation should succeed");
        let cid = *block.cid();
        storage.put(&block).await.expect("test: put should succeed");

        // Run dry run GC
        let config = GcConfig {
            dry_run: true,
            ..Default::default()
        };
        let stats = gc
            .collect(config)
            .await
            .expect("test: GC dry run should succeed");

        // Should report what would be collected, but not actually delete
        assert_eq!(stats.blocks_collected, 1);
        assert!(storage
            .has(&cid)
            .await
            .expect("test: storage has check should succeed")); // Still exists

        // Cleanup
        let _ = std::fs::remove_dir_all(&path);
    }

    /// Helper: create a unique temp path to avoid test interference.
    fn unique_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("ipfrs-gc-{}-{}", tag, std::process::id()))
    }

    /// `ipfrs gc --dry-run` must not delete anything.
    #[tokio::test]
    async fn test_gc_dry_run_no_delete() {
        let path = unique_path("dry-run-no-del");
        let _ = std::fs::remove_dir_all(&path);

        let config = BlockStoreConfig {
            path: path.clone(),
            ..Default::default()
        };
        let storage = Arc::new(
            SledBlockStore::new(config).expect("test: block store creation should succeed"),
        );
        let pin_manager = Arc::new(PinManager::new());
        let gc = GarbageCollector::new(storage.clone(), pin_manager.clone());

        // Add an unpinned block — the orphan target.
        let block =
            Block::new(Bytes::from("orphan block")).expect("test: block creation should succeed");
        let orphan_cid = *block.cid();
        storage.put(&block).await.expect("test: put should succeed");

        // Dry-run GC must report the block without deleting it.
        let stats = gc
            .collect(GcConfig {
                dry_run: true,
                ..Default::default()
            })
            .await
            .expect("test: GC dry run should succeed");
        assert_eq!(
            stats.blocks_collected, 1,
            "dry-run must report 1 collectable block"
        );
        assert!(
            storage
                .has(&orphan_cid)
                .await
                .expect("test: storage has check should succeed"),
            "dry-run must not delete the block"
        );

        let _ = std::fs::remove_dir_all(&path);
    }

    /// `ipfrs gc` (no dry-run, min_age=0) must collect unpinned orphans.
    #[tokio::test]
    async fn test_gc_collects_orphans() {
        let path = unique_path("collects-orphans");
        let _ = std::fs::remove_dir_all(&path);

        let config = BlockStoreConfig {
            path: path.clone(),
            ..Default::default()
        };
        let storage = Arc::new(
            SledBlockStore::new(config).expect("test: block store creation should succeed"),
        );
        let pin_manager = Arc::new(PinManager::new());
        let gc = GarbageCollector::new(storage.clone(), pin_manager.clone());

        // Block A is pinned; block B is an orphan.
        let block_a =
            Block::new(Bytes::from("pinned data")).expect("test: block creation should succeed");
        let block_b =
            Block::new(Bytes::from("orphan data")).expect("test: block creation should succeed");
        let cid_a = *block_a.cid();
        let cid_b = *block_b.cid();

        storage
            .put(&block_a)
            .await
            .expect("test: put block_a should succeed");
        storage
            .put(&block_b)
            .await
            .expect("test: put block_b should succeed");
        pin_manager
            .pin(cid_a, PinType::Direct, None)
            .expect("test: pin should succeed");

        // GC with min_age=0 must collect the orphan immediately.
        let stats = gc
            .collect(GcConfig {
                dry_run: false,
                min_age_seconds: 0,
                ..Default::default()
            })
            .await
            .expect("test: GC collect should succeed");

        assert_eq!(stats.blocks_collected, 1, "one orphan should be collected");
        assert!(
            storage
                .has(&cid_a)
                .await
                .expect("test: storage has check should succeed"),
            "pinned block must survive"
        );
        assert!(
            !storage
                .has(&cid_b)
                .await
                .expect("test: storage has check should succeed"),
            "orphan block must be deleted"
        );

        let _ = std::fs::remove_dir_all(&path);
    }

    #[tokio::test]
    async fn test_gc_count_unpinned() {
        let path = std::env::temp_dir().join(format!("ipfrs_gc_test_count_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        let config = BlockStoreConfig {
            path: path.clone(),
            ..Default::default()
        };
        let storage = Arc::new(
            SledBlockStore::new(config).expect("test: block store creation should succeed"),
        );
        let pin_manager = Arc::new(PinManager::new());
        let gc = GarbageCollector::new(storage.clone(), pin_manager.clone());

        // Add 3 blocks, pin 1
        for i in 0..3 {
            let block = Block::new(Bytes::from(format!("data {}", i)))
                .expect("test: block creation should succeed");
            let cid = *block.cid();
            storage.put(&block).await.expect("test: put should succeed");

            if i == 0 {
                pin_manager
                    .pin(cid, PinType::Direct, None)
                    .expect("test: pin should succeed");
            }
        }

        // Should have 2 unpinned
        let count = gc
            .count_unpinned()
            .expect("test: count_unpinned should succeed");
        assert_eq!(count, 2);

        // Cleanup
        let _ = std::fs::remove_dir_all(&path);
    }
}
