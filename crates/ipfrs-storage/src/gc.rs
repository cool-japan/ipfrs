//! Garbage Collection for block storage.
//!
//! Implements mark-and-sweep GC to reclaim space from unreferenced blocks.
//! Works with the pin management system to ensure pinned blocks are retained.
//!
//! # Algorithm
//!
//! 1. **Mark Phase**: Starting from pinned root blocks, traverse all links
//!    to mark reachable blocks.
//! 2. **Sweep Phase**: Delete all blocks that weren't marked as reachable.
//!
//! # Example
//!
//! ```rust,ignore
//! use ipfrs_storage::gc::{GarbageCollector, GcConfig};
//!
//! let gc = GarbageCollector::new(store, pin_manager, GcConfig::default());
//! let result = gc.collect().await?;
//! println!("Collected {} blocks, freed {} bytes", result.blocks_collected, result.bytes_freed);
//! ```

use crate::pinning::{PinManager, PinType};
use crate::traits::BlockStore;
use ipfrs_core::{Cid, Result};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// GC configuration
#[derive(Debug, Clone)]
pub struct GcConfig {
    /// Maximum blocks to collect in a single run (0 = unlimited)
    pub max_blocks_per_run: usize,
    /// Time limit for a single GC run (None = unlimited)
    pub time_limit: Option<Duration>,
    /// Whether to run incrementally (pause between batches)
    pub incremental: bool,
    /// Batch size for incremental GC
    pub batch_size: usize,
    /// Delay between incremental batches
    pub batch_delay: Duration,
    /// Whether to perform a dry run (don't actually delete)
    pub dry_run: bool,
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            max_blocks_per_run: 0,                  // Unlimited
            time_limit: None,                       // No limit
            incremental: false,                     // Full GC by default
            batch_size: 1000,                       // Process 1000 blocks per batch
            batch_delay: Duration::from_millis(10), // 10ms between batches
            dry_run: false,
        }
    }
}

impl GcConfig {
    /// Create a configuration for incremental GC
    pub fn incremental() -> Self {
        Self {
            incremental: true,
            ..Default::default()
        }
    }

    /// Create a configuration for dry run (no actual deletion)
    pub fn dry_run() -> Self {
        Self {
            dry_run: true,
            ..Default::default()
        }
    }

    /// Set max blocks per run
    pub fn with_max_blocks(mut self, max: usize) -> Self {
        self.max_blocks_per_run = max;
        self
    }

    /// Set time limit
    pub fn with_time_limit(mut self, duration: Duration) -> Self {
        self.time_limit = Some(duration);
        self
    }
}

/// Result of a GC run
#[derive(Debug, Clone, Default)]
pub struct GcResult {
    /// Number of blocks collected (deleted)
    pub blocks_collected: u64,
    /// Bytes freed
    pub bytes_freed: u64,
    /// Number of blocks marked as reachable
    pub blocks_marked: u64,
    /// Number of blocks scanned
    pub blocks_scanned: u64,
    /// Duration of the GC run
    pub duration: Duration,
    /// Whether GC was interrupted (time limit, etc.)
    pub interrupted: bool,
    /// Errors encountered during GC (non-fatal)
    pub errors: Vec<String>,
}

/// GC statistics tracking
#[derive(Debug, Default)]
pub struct GcStats {
    /// Total GC runs
    pub total_runs: AtomicU64,
    /// Total blocks collected across all runs
    pub total_blocks_collected: AtomicU64,
    /// Total bytes freed across all runs
    pub total_bytes_freed: AtomicU64,
    /// Last GC run time (as unix timestamp)
    pub last_run_timestamp: AtomicU64,
}

impl GcStats {
    /// Record a GC run
    pub fn record_run(&self, result: &GcResult) {
        self.total_runs.fetch_add(1, Ordering::Relaxed);
        self.total_blocks_collected
            .fetch_add(result.blocks_collected, Ordering::Relaxed);
        self.total_bytes_freed
            .fetch_add(result.bytes_freed, Ordering::Relaxed);
        self.last_run_timestamp.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            Ordering::Relaxed,
        );
    }

    /// Get a snapshot of statistics
    pub fn snapshot(&self) -> GcStatsSnapshot {
        GcStatsSnapshot {
            total_runs: self.total_runs.load(Ordering::Relaxed),
            total_blocks_collected: self.total_blocks_collected.load(Ordering::Relaxed),
            total_bytes_freed: self.total_bytes_freed.load(Ordering::Relaxed),
            last_run_timestamp: self.last_run_timestamp.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of GC statistics
#[derive(Debug, Clone)]
pub struct GcStatsSnapshot {
    pub total_runs: u64,
    pub total_blocks_collected: u64,
    pub total_bytes_freed: u64,
    pub last_run_timestamp: u64,
}

/// Link resolver function type
pub type LinkResolver = Arc<dyn Fn(&Cid) -> Result<Vec<Cid>> + Send + Sync>;

/// Garbage collector for block storage
pub struct GarbageCollector<S: BlockStore> {
    /// The block store to collect from
    store: Arc<S>,
    /// Pin manager for determining roots
    pin_manager: Arc<PinManager>,
    /// Link resolver for traversing DAG structure
    link_resolver: LinkResolver,
    /// Configuration
    config: GcConfig,
    /// Statistics
    stats: GcStats,
    /// Cancel flag for stopping GC
    cancel: AtomicBool,
}

impl<S: BlockStore> GarbageCollector<S> {
    /// Create a new garbage collector
    ///
    /// # Arguments
    /// * `store` - The block store to collect from
    /// * `pin_manager` - Pin manager for determining root blocks
    /// * `link_resolver` - Function to get links from a block
    /// * `config` - GC configuration
    pub fn new(
        store: Arc<S>,
        pin_manager: Arc<PinManager>,
        link_resolver: LinkResolver,
        config: GcConfig,
    ) -> Self {
        Self {
            store,
            pin_manager,
            link_resolver,
            config,
            stats: GcStats::default(),
            cancel: AtomicBool::new(false),
        }
    }

    /// Create with a no-op link resolver (for flat storage without DAG)
    pub fn new_flat(store: Arc<S>, pin_manager: Arc<PinManager>, config: GcConfig) -> Self {
        let link_resolver: LinkResolver = Arc::new(|_| Ok(Vec::new()));
        Self::new(store, pin_manager, link_resolver, config)
    }

    /// Request cancellation of the current GC run
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }

    /// Reset cancel flag
    pub fn reset_cancel(&self) {
        self.cancel.store(false, Ordering::SeqCst);
    }

    /// Check if GC has been cancelled
    fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }

    /// Get GC statistics
    pub fn stats(&self) -> GcStatsSnapshot {
        self.stats.snapshot()
    }

    /// Run garbage collection
    pub async fn collect(&self) -> Result<GcResult> {
        self.reset_cancel();
        let start_time = Instant::now();
        let mut result = GcResult::default();

        // Phase 1: Mark - find all reachable blocks
        let marked = self.mark_phase(&mut result).await?;

        // Check for cancellation or time limit
        if self.should_stop(start_time, &result) {
            result.interrupted = true;
            result.duration = start_time.elapsed();
            self.stats.record_run(&result);
            return Ok(result);
        }

        // Phase 2: Sweep - delete unreachable blocks
        self.sweep_phase(&marked, &mut result).await?;

        result.duration = start_time.elapsed();
        self.stats.record_run(&result);
        Ok(result)
    }

    /// Mark phase: traverse from roots to find all reachable blocks
    #[allow(clippy::unused_async)]
    async fn mark_phase(&self, result: &mut GcResult) -> Result<HashSet<Vec<u8>>> {
        let mut marked: HashSet<Vec<u8>> = HashSet::new();
        let mut to_process: Vec<Cid> = Vec::new();

        // Get all pinned CIDs as roots
        let pins = self.pin_manager.list_pins()?;
        for (cid, info) in pins {
            // Direct and recursive pins are roots
            if info.pin_type == PinType::Direct || info.pin_type == PinType::Recursive {
                to_process.push(cid);
            }
            // All pinned blocks (including indirect) should be marked
            marked.insert(cid.to_bytes());
        }

        // Traverse from roots
        while let Some(cid) = to_process.pop() {
            if self.is_cancelled() {
                break;
            }

            // Get links from this block
            match (self.link_resolver)(&cid) {
                Ok(links) => {
                    for link in links {
                        let link_bytes = link.to_bytes();
                        if marked.insert(link_bytes) {
                            // Newly marked, add to process queue
                            to_process.push(link);
                        }
                    }
                }
                Err(e) => {
                    result
                        .errors
                        .push(format!("Error resolving links for {cid}: {e}"));
                }
            }
        }

        result.blocks_marked = marked.len() as u64;
        Ok(marked)
    }

    /// Sweep phase: delete unreachable blocks
    async fn sweep_phase(&self, marked: &HashSet<Vec<u8>>, result: &mut GcResult) -> Result<()> {
        let start_time = Instant::now();
        let all_cids = self.store.list_cids()?;
        result.blocks_scanned = all_cids.len() as u64;

        let mut to_delete = Vec::new();
        let mut batch_count = 0;

        for cid in all_cids {
            if self.is_cancelled() || self.should_stop(start_time, result) {
                result.interrupted = true;
                break;
            }

            // Check max blocks limit
            if self.config.max_blocks_per_run > 0
                && result.blocks_collected >= self.config.max_blocks_per_run as u64
            {
                result.interrupted = true;
                break;
            }

            let cid_bytes = cid.to_bytes();
            if !marked.contains(&cid_bytes) {
                // Block is not marked - collect it
                if self.config.dry_run {
                    // In dry run mode, just count
                    if let Ok(Some(block)) = self.store.get(&cid).await {
                        result.bytes_freed += block.size();
                    }
                    result.blocks_collected += 1;
                } else {
                    to_delete.push(cid);
                }

                batch_count += 1;

                // Incremental GC: process in batches
                if self.config.incremental && batch_count >= self.config.batch_size {
                    if !self.config.dry_run && !to_delete.is_empty() {
                        self.delete_batch(&to_delete, result).await?;
                        to_delete.clear();
                    }
                    batch_count = 0;
                    tokio::time::sleep(self.config.batch_delay).await;
                }
            }
        }

        // Delete remaining blocks
        if !self.config.dry_run && !to_delete.is_empty() {
            self.delete_batch(&to_delete, result).await?;
        }

        Ok(())
    }

    /// Delete a batch of blocks
    async fn delete_batch(&self, cids: &[Cid], result: &mut GcResult) -> Result<()> {
        for cid in cids {
            // Get size before deletion
            if let Ok(Some(block)) = self.store.get(cid).await {
                result.bytes_freed += block.size();
            }

            // Delete the block
            match self.store.delete(cid).await {
                Ok(()) => {
                    result.blocks_collected += 1;
                }
                Err(e) => {
                    result
                        .errors
                        .push(format!("Error deleting block {cid}: {e}"));
                }
            }
        }
        Ok(())
    }

    /// Check if GC should stop
    fn should_stop(&self, start_time: Instant, _result: &GcResult) -> bool {
        if self.is_cancelled() {
            return true;
        }

        if let Some(limit) = self.config.time_limit {
            if start_time.elapsed() > limit {
                return true;
            }
        }

        false
    }
}

/// GC policy for automatic garbage collection
#[derive(Debug, Clone, Default)]
pub enum GcPolicy {
    /// Manual GC only
    #[default]
    Manual,
    /// Time-based: run every N seconds
    TimeBased { interval_secs: u64 },
    /// Space-based: run when disk usage exceeds threshold
    SpaceBased { threshold_percent: f64 },
    /// Combined: run when either condition is met
    Combined {
        interval_secs: u64,
        threshold_percent: f64,
    },
}

/// Automatic GC scheduler
pub struct GcScheduler<S: BlockStore + 'static> {
    gc: Arc<GarbageCollector<S>>,
    policy: GcPolicy,
    running: AtomicBool,
}

impl<S: BlockStore + 'static> GcScheduler<S> {
    /// Create a new GC scheduler
    pub fn new(gc: Arc<GarbageCollector<S>>, policy: GcPolicy) -> Self {
        Self {
            gc,
            policy,
            running: AtomicBool::new(false),
        }
    }

    /// Check if GC should run based on policy
    pub fn should_run(&self) -> bool {
        match &self.policy {
            GcPolicy::Manual => false,
            GcPolicy::TimeBased { interval_secs } => {
                let stats = self.gc.stats();
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                now.saturating_sub(stats.last_run_timestamp) >= *interval_secs
            }
            GcPolicy::SpaceBased { .. } => {
                // Would need disk usage info - not implemented yet
                false
            }
            GcPolicy::Combined { interval_secs, .. } => {
                let stats = self.gc.stats();
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                now.saturating_sub(stats.last_run_timestamp) >= *interval_secs
            }
        }
    }

    /// Run GC if policy conditions are met
    pub async fn maybe_run(&self) -> Option<GcResult> {
        if !self.should_run() {
            return None;
        }

        // Prevent concurrent runs
        if self
            .running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return None;
        }

        let result = self.gc.collect().await.ok();
        self.running.store(false, Ordering::SeqCst);
        result
    }

    /// Get reference to the garbage collector
    pub fn gc(&self) -> &GarbageCollector<S> {
        &self.gc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockstore::{BlockStoreConfig, SledBlockStore};
    use bytes::Bytes;
    use ipfrs_core::Block;
    use std::path::PathBuf;

    fn make_test_block(data: &[u8]) -> Block {
        Block::new(Bytes::copy_from_slice(data)).unwrap()
    }

    #[tokio::test]
    async fn test_gc_collect_unreachable() {
        let config = BlockStoreConfig {
            path: PathBuf::from("/tmp/ipfrs-test-gc"),
            cache_size: 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);

        let store = Arc::new(SledBlockStore::new(config).unwrap());
        let pin_manager = Arc::new(PinManager::new());

        // Add some blocks
        let block1 = make_test_block(b"block1");
        let block2 = make_test_block(b"block2");
        let block3 = make_test_block(b"block3");

        store.put(&block1).await.unwrap();
        store.put(&block2).await.unwrap();
        store.put(&block3).await.unwrap();

        // Pin only block1
        pin_manager.pin(block1.cid()).unwrap();

        // Create GC
        let gc = GarbageCollector::new_flat(store.clone(), pin_manager, GcConfig::default());

        // Run GC
        let result = gc.collect().await.unwrap();

        // Should have collected 2 blocks (block2 and block3)
        assert_eq!(result.blocks_collected, 2);
        assert_eq!(result.blocks_marked, 1);

        // Verify block1 still exists
        assert!(store.has(block1.cid()).await.unwrap());
        // Verify block2 and block3 are gone
        assert!(!store.has(block2.cid()).await.unwrap());
        assert!(!store.has(block3.cid()).await.unwrap());
    }

    #[tokio::test]
    async fn test_gc_dry_run() {
        let config = BlockStoreConfig {
            path: PathBuf::from("/tmp/ipfrs-test-gc-dry"),
            cache_size: 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);

        let store = Arc::new(SledBlockStore::new(config).unwrap());
        let pin_manager = Arc::new(PinManager::new());

        // Add some blocks
        let block1 = make_test_block(b"block1");
        let block2 = make_test_block(b"block2");

        store.put(&block1).await.unwrap();
        store.put(&block2).await.unwrap();

        // Pin only block1
        pin_manager.pin(block1.cid()).unwrap();

        // Create GC with dry run
        let gc = GarbageCollector::new_flat(store.clone(), pin_manager, GcConfig::dry_run());

        // Run GC
        let result = gc.collect().await.unwrap();

        // Should report 1 block to collect
        assert_eq!(result.blocks_collected, 1);

        // But block2 should still exist (dry run)
        assert!(store.has(block2.cid()).await.unwrap());
    }

    #[test]
    fn test_gc_config() {
        let config = GcConfig::default();
        assert!(!config.dry_run);
        assert!(!config.incremental);

        let config = GcConfig::incremental();
        assert!(config.incremental);

        let config = GcConfig::dry_run().with_max_blocks(100);
        assert!(config.dry_run);
        assert_eq!(config.max_blocks_per_run, 100);
    }

    #[test]
    fn test_gc_stats() {
        let stats = GcStats::default();
        let result = GcResult {
            blocks_collected: 10,
            bytes_freed: 1024,
            ..Default::default()
        };

        stats.record_run(&result);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_runs, 1);
        assert_eq!(snapshot.total_blocks_collected, 10);
        assert_eq!(snapshot.total_bytes_freed, 1024);
    }
}
