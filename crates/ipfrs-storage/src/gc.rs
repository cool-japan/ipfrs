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
use ipfrs_core::{Cid, Error, Result};
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

// ═══════════════════════════════════════════════════════════════════════════
// Orphan-focused GC (v0.3.0 storage hardening)
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for the simplified orphan-focused GC pass.
#[derive(Debug, Clone)]
pub struct OrphanGcConfig {
    /// When `true`, report orphaned blocks without actually deleting them.
    pub dry_run: bool,
    /// Only consider blocks older than this many seconds as orphans.
    /// Because Sled does not store per-block timestamps, this field is
    /// honoured when the caller supplies `block_ages` metadata; otherwise it
    /// is ignored and all unpinned blocks are treated as eligible.
    pub min_age_secs: u64,
    /// Maximum number of blocks to examine per call.
    pub batch_size: usize,
}

impl Default for OrphanGcConfig {
    fn default() -> Self {
        Self {
            dry_run: false,
            min_age_secs: 3600, // 1 hour
            batch_size: 100,
        }
    }
}

impl OrphanGcConfig {
    /// Convenience: dry-run mode with default age filter.
    pub fn dry_run() -> Self {
        Self {
            dry_run: true,
            ..Default::default()
        }
    }
}

/// Result of a single orphan GC run.
#[derive(Debug, Clone, Default)]
pub struct OrphanGcResult {
    /// Blocks examined in this pass.
    pub examined: usize,
    /// Blocks collected (deleted, or would be deleted in dry_run mode).
    pub collected: usize,
    /// Estimated bytes freed.
    pub freed_bytes: usize,
    /// Non-fatal errors encountered.
    pub errors: Vec<String>,
    /// Wall-clock duration of the run in milliseconds.
    pub duration_ms: u64,
}

/// Lightweight garbage collector that targets orphaned blocks — blocks that
/// are present in the store but are not referenced by any pinned CID.
///
/// Unlike the full mark-and-sweep [`GarbageCollector`], this collector does
/// **not** traverse DAG links.  It simply checks every stored CID against the
/// caller-supplied `pinned_cids` set and deletes the rest.
pub struct OrphanGarbageCollector {
    config: OrphanGcConfig,
}

impl OrphanGarbageCollector {
    /// Create a new orphan garbage collector with the given configuration.
    pub fn new(config: OrphanGcConfig) -> Self {
        Self { config }
    }

    /// Collect orphaned blocks from `store`.
    ///
    /// `pinned_cids` – the set of CID *string* representations that must **not**
    /// be deleted.  Every block whose string CID is absent from this set and
    /// is not otherwise protected is treated as an orphan.
    ///
    /// When `snapshot_registry` is supplied, CIDs registered there are also
    /// excluded from collection (they represent HNSW / knowledge-base snapshot
    /// blocks that must outlive their originating sessions).
    pub async fn collect<S: BlockStore>(
        &self,
        store: &S,
        pinned_cids: &std::collections::HashSet<String>,
    ) -> ipfrs_core::Result<OrphanGcResult> {
        self.collect_inner(store, pinned_cids, &HashSet::new())
            .await
    }

    /// Variant of `collect` that additionally consults a
    /// [`SledSnapshotPinRegistry`] so that HNSW snapshot CIDs are never GC'd.
    ///
    /// Only available when the `sled-backend` feature is enabled.
    #[cfg(feature = "sled-backend")]
    pub async fn collect_with_snapshot_registry<S: BlockStore>(
        &self,
        store: &S,
        pinned_cids: &std::collections::HashSet<String>,
        snapshot_registry: Option<&SledSnapshotPinRegistry>,
    ) -> ipfrs_core::Result<OrphanGcResult> {
        // Materialise snapshot-pinned CIDs once so each per-block check is O(1).
        let snapshot_pinned: HashSet<String> = match snapshot_registry {
            Some(reg) => reg.pinned_cid_strings()?,
            None => HashSet::new(),
        };
        self.collect_inner(store, pinned_cids, &snapshot_pinned)
            .await
    }

    /// Internal implementation shared by [`collect`] and [`collect_with_snapshot_registry`].
    async fn collect_inner<S: BlockStore>(
        &self,
        store: &S,
        pinned_cids: &std::collections::HashSet<String>,
        extra_pinned: &HashSet<String>,
    ) -> ipfrs_core::Result<OrphanGcResult> {
        let start = std::time::Instant::now();
        let mut result = OrphanGcResult::default();

        let all_cids = store.list_cids()?;

        for cid in all_cids.iter().take(self.config.batch_size) {
            result.examined += 1;
            let cid_str = cid.to_string();

            if pinned_cids.contains(&cid_str) || extra_pinned.contains(&cid_str) {
                // Pinned by caller or by snapshot registry – must not delete
                continue;
            }

            // Orphan candidate
            if self.config.dry_run {
                // In dry-run mode, count but do not delete
                if let Ok(Some(block)) = store.get(cid).await {
                    result.freed_bytes += block.data().len();
                }
                result.collected += 1;
            } else {
                // Get size before deletion for accounting
                match store.get(cid).await {
                    Ok(Some(block)) => {
                        result.freed_bytes += block.data().len();
                    }
                    Ok(None) => {} // Already gone
                    Err(e) => {
                        result.errors.push(format!("get error for {cid_str}: {e}"));
                    }
                }

                match store.delete(cid).await {
                    Ok(()) => {
                        result.collected += 1;
                    }
                    Err(e) => {
                        result
                            .errors
                            .push(format!("delete error for {cid_str}: {e}"));
                    }
                }
            }
        }

        result.duration_ms = start.elapsed().as_millis() as u64;
        Ok(result)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Snapshot pin registry (v0.3.0)
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// Sled-backed CID snapshot pin registry (v0.3.0)
// Only compiled when the `sled-backend` feature is enabled.
// ═══════════════════════════════════════════════════════════════════════════

/// Persistent registry of HNSW snapshot block CIDs that must survive GC.
///
/// Backed by a dedicated `"snapshot_pins"` Sled tree inside the blockstore
/// database.  Entries survive process restarts, which is critical for ensuring
/// snapshot blocks are never collected by the [`OrphanGarbageCollector`] even
/// across node restarts.
///
/// Key layout: `cid_bytes` → `label_utf8_bytes`
#[cfg(feature = "sled-backend")]
pub struct SledSnapshotPinRegistry {
    tree: sled::Tree,
}

#[cfg(feature = "sled-backend")]
impl SledSnapshotPinRegistry {
    /// Open (or create) the `"snapshot_pins"` tree inside the given Sled `Db`.
    pub fn open(db: &sled::Db) -> Result<Self> {
        let tree = db
            .open_tree("snapshot_pins")
            .map_err(|e| Error::Storage(format!("Failed to open snapshot_pins tree: {e}")))?;
        Ok(Self { tree })
    }

    /// Pin `cid` with a human-readable `label`.  Idempotent — pinning the
    /// same CID again simply overwrites the label.
    pub fn pin(&self, cid: &Cid, label: &str) -> Result<()> {
        let key = cid.to_bytes();
        let value = label.as_bytes().to_vec();
        self.tree
            .insert(key, value)
            .map_err(|e| Error::Storage(format!("Failed to pin CID {cid}: {e}")))?;
        self.tree
            .flush()
            .map_err(|e| Error::Storage(format!("Failed to flush snapshot_pins: {e}")))?;
        Ok(())
    }

    /// Remove the pin for `cid`.  No-ops silently if not present.
    pub fn unpin(&self, cid: &Cid) -> Result<()> {
        let key = cid.to_bytes();
        self.tree
            .remove(key)
            .map_err(|e| Error::Storage(format!("Failed to unpin CID {cid}: {e}")))?;
        self.tree
            .flush()
            .map_err(|e| Error::Storage(format!("Failed to flush snapshot_pins: {e}")))?;
        Ok(())
    }

    /// Return `true` when `cid` is registered in this registry.
    pub fn is_pinned(&self, cid: &Cid) -> Result<bool> {
        let key = cid.to_bytes();
        self.tree
            .contains_key(&key)
            .map_err(|e| Error::Storage(format!("Failed to check pin for CID {cid}: {e}")))
    }

    /// Return all pinned CIDs together with their labels.
    pub fn list_pinned(&self) -> Result<Vec<(Cid, String)>> {
        let mut result = Vec::new();
        for item in self.tree.iter() {
            let (key, value) =
                item.map_err(|e| Error::Storage(format!("Iteration error in snapshot_pins: {e}")))?;
            let cid = Cid::try_from(key.to_vec()).map_err(|e| {
                ipfrs_core::Error::Cid(format!("Invalid CID in snapshot_pins: {e}"))
            })?;
            let label = String::from_utf8_lossy(&value).into_owned();
            result.push((cid, label));
        }
        Ok(result)
    }

    /// Number of CIDs currently pinned in this registry.
    pub fn pin_count(&self) -> Result<usize> {
        Ok(self.tree.len())
    }

    /// Build a `HashSet<String>` of pinned CID strings for use with
    /// [`OrphanGarbageCollector`].
    pub fn pinned_cid_strings(&self) -> Result<HashSet<String>> {
        let mut set = HashSet::new();
        for item in self.tree.iter() {
            let (key, _) =
                item.map_err(|e| Error::Storage(format!("Iteration error in snapshot_pins: {e}")))?;
            let cid = Cid::try_from(key.to_vec()).map_err(|e| {
                ipfrs_core::Error::Cid(format!("Invalid CID in snapshot_pins: {e}"))
            })?;
            set.insert(cid.to_string());
        }
        Ok(set)
    }
}

/// Compute a deterministic opaque identifier for a snapshot file path.
///
/// The returned string is of the form `snapshot:<hex>` and is suitable for use
/// as a key in a [`SnapshotPinRegistry`] or as a GC-exclusion token.  It does
/// **not** represent a content hash — it is purely a path-based identifier that
/// remains stable as long as the path string is unchanged.
pub fn snapshot_pin_id(snapshot_path: &std::path::Path) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    snapshot_path.hash(&mut hasher);
    format!("snapshot:{:x}", hasher.finish())
}

/// Registry of filesystem snapshot paths that must be excluded from GC.
///
/// When the HNSW index or the knowledge-base is flushed to disk the resulting
/// file path should be registered here so the [`OrphanGarbageCollector`] (and
/// any future full GC pass) can honour the exclusion without needing to
/// understand the internal format of those files.
///
/// The registry holds [`std::path::PathBuf`]s rather than CIDs because
/// snapshot files are not (yet) tracked as content-addressed blocks in the
/// primary block store.
pub struct SnapshotPinRegistry {
    pinned_paths: std::collections::HashSet<std::path::PathBuf>,
}

impl SnapshotPinRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            pinned_paths: std::collections::HashSet::new(),
        }
    }

    /// Mark `path` as pinned so it is not eligible for GC removal.
    pub fn pin_snapshot(&mut self, path: std::path::PathBuf) {
        self.pinned_paths.insert(path);
    }

    /// Remove `path` from the pin registry, making it eligible for GC.
    ///
    /// No-ops silently if `path` is not currently pinned.
    pub fn unpin_snapshot(&mut self, path: &std::path::Path) {
        self.pinned_paths.remove(path);
    }

    /// Return `true` if `path` is currently registered as pinned.
    pub fn is_pinned(&self, path: &std::path::Path) -> bool {
        self.pinned_paths.contains(path)
    }

    /// Number of paths currently pinned in this registry.
    pub fn pinned_count(&self) -> usize {
        self.pinned_paths.len()
    }

    /// Iterate over all pinned paths in arbitrary order.
    pub fn pinned_paths(&self) -> impl Iterator<Item = &std::path::PathBuf> {
        self.pinned_paths.iter()
    }

    /// Return `true` when the registry holds no pinned paths.
    pub fn is_empty(&self) -> bool {
        self.pinned_paths.is_empty()
    }
}

impl Default for SnapshotPinRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, feature = "sled-backend"))]
mod tests {
    use super::*;
    use crate::blockstore::{BlockStoreConfig, SledBlockStore};
    use bytes::Bytes;
    use ipfrs_core::Block;
    use std::path::PathBuf;

    fn make_test_block(data: &[u8]) -> Block {
        Block::new(Bytes::copy_from_slice(data)).expect("test data is valid for block construction")
    }

    #[tokio::test]
    async fn test_gc_collect_unreachable() {
        let gc_path = std::env::temp_dir().join(format!("ipfrs-test-gc-{}", std::process::id()));
        let config = BlockStoreConfig {
            path: gc_path.clone(),
            cache_size: 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);

        let store = Arc::new(
            SledBlockStore::new(config).expect("failed to create SledBlockStore for GC test"),
        );
        let pin_manager = Arc::new(PinManager::new());

        // Add some blocks
        let block1 = make_test_block(b"block1");
        let block2 = make_test_block(b"block2");
        let block3 = make_test_block(b"block3");

        store
            .put(&block1)
            .await
            .expect("failed to put block1 into store");
        store
            .put(&block2)
            .await
            .expect("failed to put block2 into store");
        store
            .put(&block3)
            .await
            .expect("failed to put block3 into store");

        // Pin only block1
        pin_manager.pin(block1.cid()).expect("failed to pin block1");

        // Create GC
        let gc = GarbageCollector::new_flat(store.clone(), pin_manager, GcConfig::default());

        // Run GC
        let result = gc.collect().await.expect("GC collect failed");

        // Should have collected 2 blocks (block2 and block3)
        assert_eq!(result.blocks_collected, 2);
        assert_eq!(result.blocks_marked, 1);

        // Verify block1 still exists
        assert!(store
            .has(block1.cid())
            .await
            .expect("failed to check block1 existence"));
        // Verify block2 and block3 are gone
        assert!(!store
            .has(block2.cid())
            .await
            .expect("failed to check block2 existence"));
        assert!(!store
            .has(block3.cid())
            .await
            .expect("failed to check block3 existence"));

        let _ = std::fs::remove_dir_all(&gc_path);
    }

    #[tokio::test]
    async fn test_gc_dry_run() {
        let path = std::env::temp_dir().join("ipfrs-test-gc-dry");
        let config = BlockStoreConfig {
            path: path.clone(),
            cache_size: 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);

        let store =
            Arc::new(SledBlockStore::new(config).expect("test: open sled block store for dry run"));
        let pin_manager = Arc::new(PinManager::new());

        // Add some blocks
        let block1 = make_test_block(b"block1");
        let block2 = make_test_block(b"block2");

        store.put(&block1).await.expect("test: put block1");
        store.put(&block2).await.expect("test: put block2");

        // Pin only block1
        pin_manager.pin(block1.cid()).expect("test: pin block1");

        // Create GC with dry run
        let gc = GarbageCollector::new_flat(store.clone(), pin_manager, GcConfig::dry_run());

        // Run GC
        let result = gc.collect().await.expect("test: gc dry run collect");

        // Should report 1 block to collect
        assert_eq!(result.blocks_collected, 1);

        // But block2 should still exist (dry run)
        assert!(store
            .has(block2.cid())
            .await
            .expect("test: check block2 still exists in dry run"));

        let _ = std::fs::remove_dir_all(&path);
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

    // ── Orphan GC tests (v0.3.0) ──────────────────────────────────────────

    fn unique_gc_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("ipfrs-gc-{}-{}", tag, std::process::id()))
    }

    #[tokio::test]
    async fn test_gc_dry_run_reports_orphans() {
        let path = unique_gc_dir("orphan-dry");
        let _ = std::fs::remove_dir_all(&path);

        let store = Arc::new(
            SledBlockStore::new(BlockStoreConfig {
                path: path.clone(),
                cache_size: 1024 * 1024,
            })
            .expect("test: open sled block store for orphan dry run"),
        );

        let block_pinned = make_test_block(b"pinned block");
        let block_orphan = make_test_block(b"orphan block");
        store
            .put(&block_pinned)
            .await
            .expect("test: put pinned block");
        store
            .put(&block_orphan)
            .await
            .expect("test: put orphan block");

        let pinned: std::collections::HashSet<String> =
            std::iter::once(block_pinned.cid().to_string()).collect();

        let gc = OrphanGarbageCollector::new(OrphanGcConfig {
            dry_run: true,
            batch_size: 100,
            ..Default::default()
        });
        let result = gc
            .collect(store.as_ref(), &pinned)
            .await
            .expect("test: orphan gc dry run collect");

        // Should report 1 orphan without deleting it
        assert_eq!(result.collected, 1);
        assert!(store
            .has(block_orphan.cid())
            .await
            .expect("test: orphan block still exists in dry run"));
        assert!(result.freed_bytes > 0);
        assert!(result.errors.is_empty());

        let _ = std::fs::remove_dir_all(&path);
    }

    #[tokio::test]
    async fn test_gc_collects_unpinned_blocks() {
        let path = unique_gc_dir("orphan-collect");
        let _ = std::fs::remove_dir_all(&path);

        let store = Arc::new(
            SledBlockStore::new(BlockStoreConfig {
                path: path.clone(),
                cache_size: 1024 * 1024,
            })
            .expect("test: open sled block store for unpinned collection"),
        );

        let block_a = make_test_block(b"keep me");
        let block_b = make_test_block(b"delete me");
        store.put(&block_a).await.expect("test: put block_a");
        store.put(&block_b).await.expect("test: put block_b");

        let pinned: std::collections::HashSet<String> =
            std::iter::once(block_a.cid().to_string()).collect();

        let gc = OrphanGarbageCollector::new(OrphanGcConfig {
            dry_run: false,
            batch_size: 100,
            ..Default::default()
        });
        let result = gc
            .collect(store.as_ref(), &pinned)
            .await
            .expect("test: orphan gc collect unpinned");

        assert_eq!(result.collected, 1);
        assert!(
            store
                .has(block_a.cid())
                .await
                .expect("test: check pinned block survives"),
            "pinned block must survive"
        );
        assert!(
            !store
                .has(block_b.cid())
                .await
                .expect("test: check orphan block deleted"),
            "orphan must be deleted"
        );
        assert!(result.errors.is_empty());

        let _ = std::fs::remove_dir_all(&path);
    }

    #[tokio::test]
    async fn test_gc_preserves_pinned_blocks() {
        let path = unique_gc_dir("orphan-pin");
        let _ = std::fs::remove_dir_all(&path);

        let store = Arc::new(
            SledBlockStore::new(BlockStoreConfig {
                path: path.clone(),
                cache_size: 1024 * 1024,
            })
            .expect("test: open sled block store for preserve-pinned test"),
        );

        let b1 = make_test_block(b"pin1");
        let b2 = make_test_block(b"pin2");
        let b3 = make_test_block(b"pin3");
        store.put(&b1).await.expect("test: put b1");
        store.put(&b2).await.expect("test: put b2");
        store.put(&b3).await.expect("test: put b3");

        // Pin ALL blocks
        let pinned: std::collections::HashSet<String> = [
            b1.cid().to_string(),
            b2.cid().to_string(),
            b3.cid().to_string(),
        ]
        .into_iter()
        .collect();

        let gc = OrphanGarbageCollector::new(OrphanGcConfig {
            dry_run: false,
            batch_size: 100,
            ..Default::default()
        });
        let result = gc
            .collect(store.as_ref(), &pinned)
            .await
            .expect("test: orphan gc collect with all blocks pinned");

        assert_eq!(
            result.collected, 0,
            "no blocks should be collected when all are pinned"
        );
        assert!(store.has(b1.cid()).await.expect("test: b1 still exists"));
        assert!(store.has(b2.cid()).await.expect("test: b2 still exists"));
        assert!(store.has(b3.cid()).await.expect("test: b3 still exists"));

        let _ = std::fs::remove_dir_all(&path);
    }

    // ── SnapshotPinRegistry tests ─────────────────────────────────────────

    #[test]
    fn test_snapshot_pin_registry() {
        let mut registry = SnapshotPinRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.pinned_count(), 0);

        let path_a = std::path::PathBuf::from("/var/ipfrs/hnsw_index.snap");
        let path_b = std::path::PathBuf::from("/var/ipfrs/kb.snap");

        registry.pin_snapshot(path_a.clone());
        assert_eq!(registry.pinned_count(), 1);
        assert!(registry.is_pinned(&path_a));
        assert!(!registry.is_pinned(&path_b));
        assert!(!registry.is_empty());

        registry.pin_snapshot(path_b.clone());
        assert_eq!(registry.pinned_count(), 2);
        assert!(registry.is_pinned(&path_b));

        // Idempotent: pinning the same path again does not increase the count
        registry.pin_snapshot(path_a.clone());
        assert_eq!(registry.pinned_count(), 2);

        // Unpin one path
        registry.unpin_snapshot(&path_a);
        assert_eq!(registry.pinned_count(), 1);
        assert!(!registry.is_pinned(&path_a));
        assert!(registry.is_pinned(&path_b));

        // Unpin a path that was never registered — must not panic
        let unknown = std::path::PathBuf::from("/nonexistent.snap");
        registry.unpin_snapshot(&unknown);
        assert_eq!(registry.pinned_count(), 1);

        // Iterate pinned paths
        let paths: Vec<_> = registry.pinned_paths().collect();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], &path_b);
    }

    #[test]
    fn test_snapshot_pin_id_deterministic() {
        let path = std::path::Path::new("/var/ipfrs/snapshots/hnsw_index.snap");
        let id1 = snapshot_pin_id(path);
        let id2 = snapshot_pin_id(path);

        // Same path → same ID
        assert_eq!(id1, id2);

        // IDs must start with the expected prefix
        assert!(id1.starts_with("snapshot:"), "id was: {}", id1);

        // Different paths produce different IDs
        let path2 = std::path::Path::new("/var/ipfrs/snapshots/kb.snap");
        let id3 = snapshot_pin_id(path2);
        assert_ne!(id1, id3);
    }

    // ── SledSnapshotPinRegistry tests (v0.3.0) ───────────────────────────

    fn unique_sled_dir(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("ipfrs-sled-pin-{}-{}", tag, std::process::id()))
    }

    #[test]
    fn test_sled_snapshot_pin_registry_basic() {
        let path = unique_sled_dir("basic");
        let _ = std::fs::remove_dir_all(&path);

        let db = sled::open(&path).expect("test: open sled db");
        let registry =
            SledSnapshotPinRegistry::open(&db).expect("test: open snapshot pin registry");

        let block = make_test_block(b"snapshot block");
        let cid = *block.cid();

        // Initially not pinned
        assert!(!registry
            .is_pinned(&cid)
            .expect("test: check not pinned initially"));
        assert_eq!(registry.pin_count().expect("test: count before pin"), 0);

        // Pin with label
        registry
            .pin(&cid, "hnsw-v1")
            .expect("test: pin cid with label");
        assert!(registry
            .is_pinned(&cid)
            .expect("test: check pinned after pin"));
        assert_eq!(registry.pin_count().expect("test: count after pin"), 1);

        // List returns the entry
        let list = registry.list_pinned().expect("test: list pinned entries");
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].0, cid);
        assert_eq!(list[0].1, "hnsw-v1");

        // Unpin
        registry.unpin(&cid).expect("test: unpin cid");
        assert!(!registry
            .is_pinned(&cid)
            .expect("test: check not pinned after unpin"));
        assert_eq!(registry.pin_count().expect("test: count after unpin"), 0);

        let _ = std::fs::remove_dir_all(&path);
    }

    /// `test_snapshot_pin_survives_gc`:
    /// Pin a CID in the Sled registry, run orphan GC with min_age=0,
    /// verify the pinned CID is NOT deleted.
    #[tokio::test]
    async fn test_snapshot_pin_survives_gc() {
        let store_path = unique_gc_dir("snap-pin-gc");
        let sled_pin_path = unique_sled_dir("snap-pin-gc-reg");
        let _ = std::fs::remove_dir_all(&store_path);
        let _ = std::fs::remove_dir_all(&sled_pin_path);

        let store = Arc::new(
            SledBlockStore::new(BlockStoreConfig {
                path: store_path.clone(),
                cache_size: 1024 * 1024,
            })
            .expect("test: open sled block store for snapshot-pin-gc test"),
        );

        let pinned_block = make_test_block(b"hnsw snapshot block");
        let orphan_block = make_test_block(b"truly orphaned block");

        store
            .put(&pinned_block)
            .await
            .expect("test: put pinned block");
        store
            .put(&orphan_block)
            .await
            .expect("test: put orphan block");

        // Register the pinned block in the Sled snapshot registry
        let db = sled::open(&sled_pin_path).expect("test: open sled db for pin registry");
        let snap_reg =
            SledSnapshotPinRegistry::open(&db).expect("test: open snapshot pin registry");
        snap_reg
            .pin(pinned_block.cid(), "hnsw-snapshot")
            .expect("test: pin block in snapshot registry");

        // Run orphan GC with empty caller-supplied pins (min_age=0 via batch_size=100)
        let gc = OrphanGarbageCollector::new(OrphanGcConfig {
            dry_run: false,
            min_age_secs: 0,
            batch_size: 100,
        });
        let pinned_set = std::collections::HashSet::new();
        let result = gc
            .collect_with_snapshot_registry(store.as_ref(), &pinned_set, Some(&snap_reg))
            .await
            .expect("test: collect with snapshot registry");

        // Orphan block must be deleted
        assert_eq!(result.collected, 1, "orphan block must be collected");
        assert!(
            !store
                .has(orphan_block.cid())
                .await
                .expect("test: check orphan block gone after gc"),
            "orphan must be gone"
        );

        // Pinned block (from Sled registry) must survive
        assert!(
            store
                .has(pinned_block.cid())
                .await
                .expect("test: check snapshot-pinned block survives gc"),
            "snapshot-pinned block must survive GC"
        );

        let _ = std::fs::remove_dir_all(&store_path);
        let _ = std::fs::remove_dir_all(&sled_pin_path);
    }

    // ── CompactionScheduler trigger test (v0.3.0) ─────────────────────────

    /// `test_compaction_scheduler_triggers`:
    /// Advance the scheduler past the threshold and verify `should_compact()`.
    #[test]
    fn test_compaction_scheduler_triggers() {
        use crate::compaction::{CompactionConfig, CompactionScheduler};
        use std::time::Duration;

        // Very low bytes threshold so the first write triggers compaction.
        let sched = CompactionScheduler::new(CompactionConfig {
            idle_threshold: Duration::from_secs(0),
            min_interval: Duration::from_secs(0),
            max_bytes_since_compact: 1,
        });
        // Record a write above the threshold.
        sched.record_write(2);
        assert!(
            sched.should_compact(),
            "scheduler must trigger when bytes threshold is exceeded"
        );
    }
}
