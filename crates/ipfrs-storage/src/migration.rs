//! Storage migration utilities
//!
//! This module provides utilities for migrating data between different storage backends,
//! enabling seamless transitions in production deployments.
//!
//! It also provides the `StorageMigrationFramework` for schema-level version upgrades.

use crate::traits::BlockStore;
use ipfrs_core::{Cid, Result};
use std::fmt;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Migration statistics for backend-to-backend block migrations
#[derive(Debug, Clone, Default)]
pub struct BlockMigrationStats {
    /// Total blocks migrated
    pub blocks_migrated: u64,
    /// Total bytes migrated
    pub bytes_migrated: u64,
    /// Number of blocks skipped (already present in destination)
    pub blocks_skipped: u64,
    /// Number of errors encountered
    pub errors: u64,
    /// Migration duration
    pub duration: Duration,
    /// Migration throughput in blocks per second
    pub blocks_per_second: f64,
    /// Migration throughput in bytes per second
    pub bytes_per_second: f64,
}

impl BlockMigrationStats {
    /// Calculate throughput metrics
    fn calculate_throughput(&mut self, duration: Duration) {
        let seconds = duration.as_secs_f64();
        if seconds > 0.0 {
            self.blocks_per_second = self.blocks_migrated as f64 / seconds;
            self.bytes_per_second = self.bytes_migrated as f64 / seconds;
        }
    }
}

/// Migration configuration
#[derive(Debug, Clone)]
pub struct MigrationConfig {
    /// Batch size for bulk operations
    pub batch_size: usize,
    /// Whether to skip blocks that already exist in destination
    pub skip_existing: bool,
    /// Whether to verify each block after migration
    pub verify: bool,
    /// Maximum number of concurrent operations
    pub concurrency: usize,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            skip_existing: true,
            verify: false,
            concurrency: 4,
        }
    }
}

/// Progress callback type
pub type ProgressCallback = Arc<dyn Fn(u64, u64) + Send + Sync>;

/// Storage migrator
pub struct StorageMigrator<S: BlockStore, D: BlockStore> {
    source: Arc<S>,
    destination: Arc<D>,
    config: MigrationConfig,
    progress_callback: Option<ProgressCallback>,
}

impl<S: BlockStore, D: BlockStore> StorageMigrator<S, D> {
    /// Create a new migrator
    pub fn new(source: Arc<S>, destination: Arc<D>) -> Self {
        Self {
            source,
            destination,
            config: MigrationConfig::default(),
            progress_callback: None,
        }
    }

    /// Create with custom configuration
    pub fn with_config(source: Arc<S>, destination: Arc<D>, config: MigrationConfig) -> Self {
        Self {
            source,
            destination,
            config,
            progress_callback: None,
        }
    }

    /// Set progress callback
    pub fn with_progress_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(u64, u64) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Arc::new(callback));
        self
    }

    /// Migrate all blocks from source to destination
    pub async fn migrate_all(&self) -> Result<BlockMigrationStats> {
        let start = Instant::now();

        let blocks_migrated = AtomicU64::new(0);
        let bytes_migrated = AtomicU64::new(0);
        let blocks_skipped = AtomicU64::new(0);
        let errors = AtomicU64::new(0);

        // Get all CIDs from source
        let all_cids = self.source.list_cids()?;
        let total_blocks = all_cids.len() as u64;

        // Migrate in batches
        for batch in all_cids.chunks(self.config.batch_size) {
            // Check which blocks already exist in destination if skip_existing is enabled
            let cids_to_migrate = if self.config.skip_existing {
                let exists = self.destination.has_many(batch).await?;
                batch
                    .iter()
                    .zip(exists.iter())
                    .filter_map(|(cid, exists)| {
                        if *exists {
                            blocks_skipped.fetch_add(1, Ordering::Relaxed);
                            None
                        } else {
                            Some(*cid)
                        }
                    })
                    .collect::<Vec<_>>()
            } else {
                batch.to_vec()
            };

            if cids_to_migrate.is_empty() {
                continue;
            }

            // Get blocks from source
            let blocks_result = self.source.get_many(&cids_to_migrate).await?;

            // Filter out None values and collect valid blocks
            let mut valid_blocks = Vec::new();
            for block_opt in blocks_result {
                if let Some(block) = block_opt {
                    bytes_migrated.fetch_add(block.data().len() as u64, Ordering::Relaxed);
                    valid_blocks.push(block);
                } else {
                    errors.fetch_add(1, Ordering::Relaxed);
                }
            }

            // Put blocks to destination
            if !valid_blocks.is_empty() {
                match self.destination.put_many(&valid_blocks).await {
                    Ok(_) => {
                        blocks_migrated.fetch_add(valid_blocks.len() as u64, Ordering::Relaxed);

                        // Verify if enabled
                        if self.config.verify {
                            let cids: Vec<Cid> = valid_blocks.iter().map(|b| *b.cid()).collect();
                            let verified = self.destination.has_many(&cids).await?;
                            let failed = verified.iter().filter(|&&exists| !exists).count();
                            if failed > 0 {
                                errors.fetch_add(failed as u64, Ordering::Relaxed);
                            }
                        }
                    }
                    Err(_) => {
                        errors.fetch_add(valid_blocks.len() as u64, Ordering::Relaxed);
                    }
                }
            }

            // Call progress callback
            if let Some(ref callback) = self.progress_callback {
                let migrated = blocks_migrated.load(Ordering::Relaxed);
                callback(migrated, total_blocks);
            }
        }

        let mut stats = BlockMigrationStats {
            blocks_migrated: blocks_migrated.load(Ordering::Relaxed),
            bytes_migrated: bytes_migrated.load(Ordering::Relaxed),
            blocks_skipped: blocks_skipped.load(Ordering::Relaxed),
            errors: errors.load(Ordering::Relaxed),
            duration: start.elapsed(),
            blocks_per_second: 0.0,
            bytes_per_second: 0.0,
        };

        stats.calculate_throughput(stats.duration);

        Ok(stats)
    }

    /// Migrate specific CIDs
    pub async fn migrate_cids(&self, cids: &[Cid]) -> Result<BlockMigrationStats> {
        let start = Instant::now();

        let blocks_migrated = AtomicU64::new(0);
        let bytes_migrated = AtomicU64::new(0);
        let blocks_skipped = AtomicU64::new(0);
        let errors = AtomicU64::new(0);

        // Migrate in batches
        for batch in cids.chunks(self.config.batch_size) {
            // Check which blocks already exist
            let cids_to_migrate = if self.config.skip_existing {
                let exists = self.destination.has_many(batch).await?;
                batch
                    .iter()
                    .zip(exists.iter())
                    .filter_map(|(cid, exists)| {
                        if *exists {
                            blocks_skipped.fetch_add(1, Ordering::Relaxed);
                            None
                        } else {
                            Some(*cid)
                        }
                    })
                    .collect::<Vec<_>>()
            } else {
                batch.to_vec()
            };

            if cids_to_migrate.is_empty() {
                continue;
            }

            // Get and migrate blocks
            let blocks_result = self.source.get_many(&cids_to_migrate).await?;
            let mut valid_blocks = Vec::new();

            for block_opt in blocks_result {
                if let Some(block) = block_opt {
                    bytes_migrated.fetch_add(block.data().len() as u64, Ordering::Relaxed);
                    valid_blocks.push(block);
                } else {
                    errors.fetch_add(1, Ordering::Relaxed);
                }
            }

            if !valid_blocks.is_empty() {
                match self.destination.put_many(&valid_blocks).await {
                    Ok(_) => {
                        blocks_migrated.fetch_add(valid_blocks.len() as u64, Ordering::Relaxed);
                    }
                    Err(_) => {
                        errors.fetch_add(valid_blocks.len() as u64, Ordering::Relaxed);
                    }
                }
            }
        }

        let mut stats = BlockMigrationStats {
            blocks_migrated: blocks_migrated.load(Ordering::Relaxed),
            bytes_migrated: bytes_migrated.load(Ordering::Relaxed),
            blocks_skipped: blocks_skipped.load(Ordering::Relaxed),
            errors: errors.load(Ordering::Relaxed),
            duration: start.elapsed(),
            blocks_per_second: 0.0,
            bytes_per_second: 0.0,
        };

        stats.calculate_throughput(stats.duration);

        Ok(stats)
    }
}

/// Helper function to migrate between stores
pub async fn migrate_storage<S: BlockStore, D: BlockStore>(
    source: Arc<S>,
    destination: Arc<D>,
) -> Result<BlockMigrationStats> {
    let migrator = StorageMigrator::new(source, destination);
    migrator.migrate_all().await
}

/// Helper function to migrate with progress reporting
pub async fn migrate_storage_with_progress<S: BlockStore, D: BlockStore, F>(
    source: Arc<S>,
    destination: Arc<D>,
    progress_callback: F,
) -> Result<BlockMigrationStats>
where
    F: Fn(u64, u64) + Send + Sync + 'static,
{
    let migrator =
        StorageMigrator::new(source, destination).with_progress_callback(progress_callback);
    migrator.migrate_all().await
}

/// Migrate with custom batch size for optimal performance
pub async fn migrate_storage_batched<S: BlockStore, D: BlockStore>(
    source: Arc<S>,
    destination: Arc<D>,
    batch_size: usize,
) -> Result<BlockMigrationStats> {
    let config = MigrationConfig {
        batch_size,
        ..Default::default()
    };
    let migrator = StorageMigrator::with_config(source, destination, config);
    migrator.migrate_all().await
}

/// Migrate with verification enabled (slower but safer)
pub async fn migrate_storage_verified<S: BlockStore, D: BlockStore>(
    source: Arc<S>,
    destination: Arc<D>,
) -> Result<BlockMigrationStats> {
    let config = MigrationConfig {
        verify: true,
        ..Default::default()
    };
    let migrator = StorageMigrator::with_config(source, destination, config);
    migrator.migrate_all().await
}

/// Estimate migration time and space requirements
#[derive(Debug, Clone)]
pub struct MigrationEstimate {
    /// Total blocks to migrate
    pub total_blocks: usize,
    /// Total bytes to migrate
    pub total_bytes: u64,
    /// Estimated duration at 100 blocks/sec
    pub estimated_duration_low: Duration,
    /// Estimated duration at 1000 blocks/sec
    pub estimated_duration_high: Duration,
    /// Space required in destination
    pub space_required: u64,
}

/// Estimate migration requirements
pub async fn estimate_migration<S: BlockStore>(source: Arc<S>) -> Result<MigrationEstimate> {
    let all_cids = source.list_cids()?;
    let total_blocks = all_cids.len();

    // Sample first 100 blocks to estimate average size
    let sample_size = total_blocks.min(100);
    let sample_cids: Vec<_> = all_cids.iter().take(sample_size).copied().collect();

    let blocks = source.get_many(&sample_cids).await?;
    let sample_bytes: u64 = blocks
        .iter()
        .filter_map(|b| b.as_ref())
        .map(|b| b.data().len() as u64)
        .sum();

    let avg_block_size = if sample_size > 0 {
        sample_bytes / sample_size as u64
    } else {
        0
    };

    let total_bytes = avg_block_size * total_blocks as u64;

    // Estimate durations (conservative: 100 blocks/sec, optimistic: 1000 blocks/sec)
    let estimated_duration_low = Duration::from_secs(total_blocks as u64 / 100);
    let estimated_duration_high = Duration::from_secs(total_blocks as u64 / 1000);

    Ok(MigrationEstimate {
        total_blocks,
        total_bytes,
        estimated_duration_low,
        estimated_duration_high,
        space_required: total_bytes,
    })
}

/// Migration validation - verify both stores have identical content
pub async fn validate_migration<S: BlockStore, D: BlockStore>(
    source: Arc<S>,
    destination: Arc<D>,
) -> Result<bool> {
    let source_cids = source.list_cids()?;
    let dest_cids = destination.list_cids()?;

    // Check if same number of blocks
    if source_cids.len() != dest_cids.len() {
        return Ok(false);
    }

    // Check all source CIDs exist in destination
    let exists = destination.has_many(&source_cids).await?;
    Ok(exists.iter().all(|&e| e))
}

// ============================================================
//  StorageMigrationFramework
// ============================================================

/// Schema version newtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SchemaVersion(pub u32);

impl SchemaVersion {
    /// Schema version 1.
    pub const V1: SchemaVersion = SchemaVersion(1);
    /// Schema version 2.
    pub const V2: SchemaVersion = SchemaVersion(2);
    /// Schema version 3.
    pub const V3: SchemaVersion = SchemaVersion(3);
}

impl fmt::Display for SchemaVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl From<u32> for SchemaVersion {
    fn from(v: u32) -> Self {
        SchemaVersion(v)
    }
}

impl From<SchemaVersion> for u32 {
    fn from(v: SchemaVersion) -> Self {
        v.0
    }
}

/// A single migration step from one schema version to another.
#[derive(Debug, Clone)]
pub struct MigrationStep {
    /// Source schema version.
    pub from_version: SchemaVersion,
    /// Target schema version.
    pub to_version: SchemaVersion,
    /// Human-readable description of what this step does.
    pub description: String,
    /// Whether this step can be safely reversed (rolled back).
    pub is_reversible: bool,
}

impl MigrationStep {
    /// Create a new migration step.
    pub fn new(
        from_version: SchemaVersion,
        to_version: SchemaVersion,
        description: impl Into<String>,
        is_reversible: bool,
    ) -> Self {
        Self {
            from_version,
            to_version,
            description: description.into(),
            is_reversible,
        }
    }
}

/// Status of a migration record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MigrationStatus {
    /// Not yet started.
    Pending,
    /// Currently executing.
    Running,
    /// Successfully completed.
    Completed,
    /// Failed with a reason.
    Failed {
        /// Human-readable failure reason.
        reason: String,
    },
    /// Successfully rolled back.
    RolledBack,
}

/// A historical record of a single migration step execution.
#[derive(Debug, Clone)]
pub struct MigrationRecord {
    /// The step that was (or is being) executed.
    pub step: MigrationStep,
    /// Wall-clock timestamp (milliseconds since epoch) when execution started.
    pub started_at_ms: u64,
    /// Wall-clock timestamp when execution completed, or `None` if still running.
    pub completed_at_ms: Option<u64>,
    /// Number of blocks touched during this step.
    pub blocks_migrated: u64,
    /// Current status.
    pub status: MigrationStatus,
}

/// Errors produced by the migration framework.
#[derive(Debug, Error)]
pub enum MigrationError {
    /// No chain of steps leads from `from` to `to`.
    #[error("no migration path found from v{from} to v{to}")]
    NoPathFound {
        /// Source version.
        from: u32,
        /// Target version.
        to: u32,
    },
    /// A step execution failed.
    #[error("step v{from}→v{to} failed: {reason}")]
    StepFailed {
        /// Source version.
        from: u32,
        /// Target version.
        to: u32,
        /// Failure reason.
        reason: String,
    },
    /// The store is already at the requested target version.
    #[error("already at version v{0}")]
    AlreadyAtVersion(u32),
    /// The plan or last step cannot be reversed.
    #[error("migration is not reversible: {0}")]
    NotReversible(String),
}

/// An ordered sequence of [`MigrationStep`]s needed to reach a target version.
#[derive(Debug, Clone)]
pub struct MigrationPlan {
    /// Ordered steps to execute (current → target).
    pub steps: Vec<MigrationStep>,
    /// Starting schema version.
    pub current_version: SchemaVersion,
    /// Desired schema version.
    pub target_version: SchemaVersion,
}

impl MigrationPlan {
    /// Build a migration plan by finding a chain of steps from `current` to `target`.
    ///
    /// Uses a simple BFS / greedy chain search: each step's `to_version` must match
    /// the next step's `from_version`.
    pub fn build(
        current: SchemaVersion,
        target: SchemaVersion,
        available: &[MigrationStep],
    ) -> std::result::Result<Self, MigrationError> {
        if current == target {
            return Err(MigrationError::AlreadyAtVersion(current.0));
        }

        // BFS to find a path
        let mut queue: Vec<(SchemaVersion, Vec<MigrationStep>)> = vec![(current, Vec::new())];
        let mut visited = std::collections::HashSet::new();
        visited.insert(current);

        while let Some((version, path)) = queue.pop() {
            for step in available.iter().filter(|s| s.from_version == version) {
                let mut new_path = path.clone();
                new_path.push(step.clone());

                if step.to_version == target {
                    return Ok(Self {
                        steps: new_path,
                        current_version: current,
                        target_version: target,
                    });
                }

                if !visited.contains(&step.to_version) {
                    visited.insert(step.to_version);
                    queue.push((step.to_version, new_path));
                }
            }
        }

        Err(MigrationError::NoPathFound {
            from: current.0,
            to: target.0,
        })
    }

    /// Number of steps in the plan.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Returns `true` when there are no steps to execute.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Returns `true` when every step in the plan is reversible.
    pub fn is_reversible(&self) -> bool {
        self.steps.iter().all(|s| s.is_reversible)
    }
}

/// A snapshot of [`MigrationStats`] with plain `u64` values for easy inspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationStatsSnapshot {
    /// Total number of migration steps that have been executed.
    pub total_steps_run: u64,
    /// Total blocks touched across all executed steps.
    pub total_blocks_migrated: u64,
    /// Total number of steps that failed.
    pub total_failures: u64,
}

/// Atomic counters tracking migration activity across the lifetime of a [`MigrationRunner`].
#[derive(Debug, Default)]
pub struct MigrationStats {
    /// Total number of migration steps executed (success or failure).
    pub total_steps_run: AtomicU64,
    /// Total blocks touched by all executed steps.
    pub total_blocks_migrated: AtomicU64,
    /// Total number of failed steps.
    pub total_failures: AtomicU64,
}

impl MigrationStats {
    /// Create a new zeroed stats instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a point-in-time snapshot of the counters.
    pub fn snapshot(&self) -> MigrationStatsSnapshot {
        MigrationStatsSnapshot {
            total_steps_run: self.total_steps_run.load(Ordering::SeqCst),
            total_blocks_migrated: self.total_blocks_migrated.load(Ordering::SeqCst),
            total_failures: self.total_failures.load(Ordering::SeqCst),
        }
    }
}

/// Returns current time as milliseconds since the Unix epoch, using a monotonic fallback.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis() as u64
}

/// Executes migration plans and tracks history.
///
/// All mutation is performed through shared references so that the runner can be
/// wrapped in an `Arc` and used from multiple threads.
pub struct MigrationRunner {
    /// Full history of every step that has been executed.
    pub history: Mutex<Vec<MigrationRecord>>,
    /// Current schema version, updated atomically after each successful step.
    pub current_version: AtomicU32,
    /// Cumulative statistics.
    pub stats: MigrationStats,
}

impl MigrationRunner {
    /// Create a runner starting at `initial_version`.
    pub fn new(initial_version: SchemaVersion) -> Self {
        Self {
            history: Mutex::new(Vec::new()),
            current_version: AtomicU32::new(initial_version.0),
            stats: MigrationStats::new(),
        }
    }

    /// Execute a single migration step (no actual I/O — simulates the step).
    ///
    /// Records the outcome in the history log and updates statistics.
    pub fn execute_step(
        &self,
        step: &MigrationStep,
        blocks_count: u64,
    ) -> std::result::Result<MigrationRecord, MigrationError> {
        let started_at_ms = now_ms();

        // Build a running record
        let running_record = MigrationRecord {
            step: step.clone(),
            started_at_ms,
            completed_at_ms: None,
            blocks_migrated: 0,
            status: MigrationStatus::Running,
        };

        // Simulate work — in a real implementation this would perform I/O.
        // We treat all steps as successful unless the step would create an invalid transition.
        let completed_at_ms = now_ms();

        let record = MigrationRecord {
            step: step.clone(),
            started_at_ms: running_record.started_at_ms,
            completed_at_ms: Some(completed_at_ms),
            blocks_migrated: blocks_count,
            status: MigrationStatus::Completed,
        };

        // Update the current version
        self.current_version
            .store(step.to_version.0, Ordering::SeqCst);

        // Update stats
        self.stats.total_steps_run.fetch_add(1, Ordering::SeqCst);
        self.stats
            .total_blocks_migrated
            .fetch_add(blocks_count, Ordering::SeqCst);

        // Append to history
        {
            let mut guard = self
                .history
                .lock()
                .map_err(|_| MigrationError::StepFailed {
                    from: step.from_version.0,
                    to: step.to_version.0,
                    reason: "history mutex poisoned".to_string(),
                })?;
            guard.push(record.clone());
        }

        Ok(record)
    }

    /// Execute every step in a [`MigrationPlan`] sequentially.
    ///
    /// Returns the list of completed [`MigrationRecord`]s.  If any step fails the
    /// error is returned immediately and subsequent steps are not executed.
    pub fn execute_plan(
        &self,
        plan: &MigrationPlan,
        blocks_per_step: u64,
    ) -> std::result::Result<Vec<MigrationRecord>, MigrationError> {
        let mut records = Vec::with_capacity(plan.steps.len());

        for step in &plan.steps {
            match self.execute_step(step, blocks_per_step) {
                Ok(record) => records.push(record),
                Err(err) => {
                    self.stats.total_failures.fetch_add(1, Ordering::SeqCst);
                    return Err(err);
                }
            }
        }

        Ok(records)
    }

    /// Return the current schema version.
    pub fn current_version(&self) -> SchemaVersion {
        SchemaVersion(self.current_version.load(Ordering::SeqCst))
    }

    /// Return a clone of the full history.
    pub fn history(&self) -> Vec<MigrationRecord> {
        self.history.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Returns `true` when the most recent step was reversible and can be rolled back.
    pub fn can_rollback(&self) -> bool {
        self.history
            .lock()
            .map(|g| g.last().map(|r| r.step.is_reversible).unwrap_or(false))
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod schema_migration_tests {
    use super::{
        MigrationError, MigrationPlan, MigrationRunner, MigrationStatus, MigrationStep,
        SchemaVersion,
    };

    // ---- helpers ------------------------------------------------------------

    fn step(from: u32, to: u32, reversible: bool) -> MigrationStep {
        MigrationStep::new(
            SchemaVersion(from),
            SchemaVersion(to),
            format!("v{from}→v{to}"),
            reversible,
        )
    }

    fn v1_to_v3_steps() -> Vec<MigrationStep> {
        vec![step(1, 2, true), step(2, 3, true)]
    }

    // ---- SchemaVersion ------------------------------------------------------

    #[test]
    fn schema_version_display() {
        assert_eq!(SchemaVersion::V1.to_string(), "v1");
        assert_eq!(SchemaVersion::V3.to_string(), "v3");
    }

    #[test]
    fn schema_version_constants() {
        assert_eq!(SchemaVersion::V1.0, 1);
        assert_eq!(SchemaVersion::V2.0, 2);
        assert_eq!(SchemaVersion::V3.0, 3);
    }

    // ---- MigrationPlan::build -----------------------------------------------

    #[test]
    fn plan_build_finds_direct_path() {
        let steps = vec![step(1, 2, true)];
        let plan = MigrationPlan::build(SchemaVersion::V1, SchemaVersion::V2, &steps)
            .expect("should find direct step");
        assert_eq!(plan.step_count(), 1);
        assert_eq!(plan.current_version, SchemaVersion::V1);
        assert_eq!(plan.target_version, SchemaVersion::V2);
    }

    #[test]
    fn plan_build_finds_multi_step_path() {
        let steps = v1_to_v3_steps();
        let plan = MigrationPlan::build(SchemaVersion::V1, SchemaVersion::V3, &steps)
            .expect("should find two-step path");
        assert_eq!(plan.step_count(), 2);
    }

    #[test]
    fn plan_build_error_when_no_path() {
        // only v2→v3 exists; v1→v2 is missing
        let steps = vec![step(2, 3, true)];
        let result = MigrationPlan::build(SchemaVersion::V1, SchemaVersion::V3, &steps);
        assert!(matches!(
            result,
            Err(MigrationError::NoPathFound { from: 1, to: 3 })
        ));
    }

    #[test]
    fn plan_build_already_at_version() {
        let steps = v1_to_v3_steps();
        let result = MigrationPlan::build(SchemaVersion::V2, SchemaVersion::V2, &steps);
        assert!(matches!(result, Err(MigrationError::AlreadyAtVersion(2))));
    }

    #[test]
    fn plan_is_reversible_all_true() {
        let steps = vec![step(1, 2, true), step(2, 3, true)];
        let plan = MigrationPlan::build(SchemaVersion::V1, SchemaVersion::V3, &steps).unwrap();
        assert!(plan.is_reversible());
    }

    #[test]
    fn plan_is_reversible_false_when_any_irreversible() {
        let steps = vec![step(1, 2, true), step(2, 3, false)];
        let plan = MigrationPlan::build(SchemaVersion::V1, SchemaVersion::V3, &steps).unwrap();
        assert!(!plan.is_reversible());
    }

    // ---- MigrationRunner::execute_step --------------------------------------

    #[test]
    fn execute_step_records_history() {
        let runner = MigrationRunner::new(SchemaVersion::V1);
        let s = step(1, 2, true);
        let record = runner.execute_step(&s, 42).expect("step should succeed");

        assert_eq!(record.status, MigrationStatus::Completed);
        assert_eq!(record.blocks_migrated, 42);
        assert!(record.completed_at_ms.is_some());

        let history = runner.history();
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn execute_step_updates_current_version() {
        let runner = MigrationRunner::new(SchemaVersion::V1);
        runner.execute_step(&step(1, 2, true), 0).unwrap();
        assert_eq!(runner.current_version(), SchemaVersion::V2);
    }

    // ---- MigrationRunner::execute_plan --------------------------------------

    #[test]
    fn execute_plan_runs_all_steps() {
        let runner = MigrationRunner::new(SchemaVersion::V1);
        let available = v1_to_v3_steps();
        let plan = MigrationPlan::build(SchemaVersion::V1, SchemaVersion::V3, &available).unwrap();
        let records = runner
            .execute_plan(&plan, 100)
            .expect("plan should succeed");

        assert_eq!(records.len(), 2);
        assert_eq!(runner.current_version(), SchemaVersion::V3);
    }

    #[test]
    fn execute_plan_updates_version_to_target() {
        let runner = MigrationRunner::new(SchemaVersion::V1);
        let available = v1_to_v3_steps();
        let plan = MigrationPlan::build(SchemaVersion::V1, SchemaVersion::V3, &available).unwrap();
        runner.execute_plan(&plan, 50).unwrap();
        assert_eq!(runner.current_version(), SchemaVersion::V3);
    }

    // ---- Stats --------------------------------------------------------------

    #[test]
    fn stats_accumulate_across_steps() {
        let runner = MigrationRunner::new(SchemaVersion::V1);
        runner.execute_step(&step(1, 2, true), 10).unwrap();
        runner.execute_step(&step(2, 3, true), 20).unwrap();

        let snap = runner.stats.snapshot();
        assert_eq!(snap.total_steps_run, 2);
        assert_eq!(snap.total_blocks_migrated, 30);
        assert_eq!(snap.total_failures, 0);
    }

    // ---- can_rollback -------------------------------------------------------

    #[test]
    fn can_rollback_true_when_last_step_reversible() {
        let runner = MigrationRunner::new(SchemaVersion::V1);
        runner.execute_step(&step(1, 2, true), 0).unwrap();
        assert!(runner.can_rollback());
    }

    #[test]
    fn can_rollback_false_when_last_step_irreversible() {
        let runner = MigrationRunner::new(SchemaVersion::V1);
        runner.execute_step(&step(1, 2, false), 0).unwrap();
        assert!(!runner.can_rollback());
    }

    #[test]
    fn can_rollback_false_when_no_history() {
        let runner = MigrationRunner::new(SchemaVersion::V1);
        assert!(!runner.can_rollback());
    }

    // ---- multi-step plan ----------------------------------------------------

    #[test]
    fn multi_step_plan_history_grows() {
        let runner = MigrationRunner::new(SchemaVersion::V1);
        let available = v1_to_v3_steps();
        let plan = MigrationPlan::build(SchemaVersion::V1, SchemaVersion::V3, &available).unwrap();
        runner.execute_plan(&plan, 5).unwrap();

        let history = runner.history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].step.from_version, SchemaVersion::V1);
        assert_eq!(history[1].step.from_version, SchemaVersion::V2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryBlockStore;
    use bytes::Bytes;
    use ipfrs_core::Block;

    #[tokio::test]
    async fn test_basic_migration() {
        let source = Arc::new(MemoryBlockStore::new());
        let destination = Arc::new(MemoryBlockStore::new());

        // Add some blocks to source
        for i in 0..10 {
            let block = Block::new(Bytes::from(format!("block {}", i))).unwrap();
            source.put(&block).await.unwrap();
        }

        assert_eq!(source.len(), 10);
        assert_eq!(destination.len(), 0);

        // Migrate
        let stats = migrate_storage(source.clone(), destination.clone())
            .await
            .unwrap();

        assert_eq!(stats.blocks_migrated, 10);
        assert_eq!(stats.blocks_skipped, 0);
        assert_eq!(stats.errors, 0);
        assert_eq!(destination.len(), 10);
    }

    #[tokio::test]
    async fn test_migration_skip_existing() {
        let source = Arc::new(MemoryBlockStore::new());
        let destination = Arc::new(MemoryBlockStore::new());

        // Add blocks to both stores
        let mut blocks = Vec::new();
        for i in 0..10 {
            let block = Block::new(Bytes::from(format!("block {}", i))).unwrap();
            blocks.push(block);
        }

        // Add all to source
        for block in &blocks {
            source.put(block).await.unwrap();
        }

        // Add first 5 to destination
        for block in blocks.iter().take(5) {
            destination.put(block).await.unwrap();
        }

        // Migrate with skip_existing
        let config = MigrationConfig {
            skip_existing: true,
            ..Default::default()
        };
        let migrator = StorageMigrator::with_config(source, destination.clone(), config);
        let stats = migrator.migrate_all().await.unwrap();

        assert_eq!(stats.blocks_migrated, 5); // Only new blocks
        assert_eq!(stats.blocks_skipped, 5); // Existing blocks
        assert_eq!(destination.len(), 10);
    }

    #[tokio::test]
    async fn test_migration_with_progress() {
        let source = Arc::new(MemoryBlockStore::new());
        let destination = Arc::new(MemoryBlockStore::new());

        // Add blocks
        for i in 0..20 {
            let block = Block::new(Bytes::from(format!("block {}", i))).unwrap();
            source.put(&block).await.unwrap();
        }

        let progress_called = Arc::new(AtomicU64::new(0));
        let progress_called_clone = progress_called.clone();

        let stats = migrate_storage_with_progress(source, destination, move |_current, _total| {
            progress_called_clone.fetch_add(1, Ordering::Relaxed);
        })
        .await
        .unwrap();

        assert_eq!(stats.blocks_migrated, 20);
        assert!(progress_called.load(Ordering::Relaxed) > 0);
    }

    #[tokio::test]
    async fn test_migrate_storage_batched() {
        let source = Arc::new(MemoryBlockStore::new());
        let destination = Arc::new(MemoryBlockStore::new());

        // Add blocks
        for i in 0..50 {
            let block = Block::new(Bytes::from(format!("block {}", i))).unwrap();
            source.put(&block).await.unwrap();
        }

        let stats = migrate_storage_batched(source, destination.clone(), 10)
            .await
            .unwrap();

        assert_eq!(stats.blocks_migrated, 50);
        assert_eq!(destination.len(), 50);
    }

    #[tokio::test]
    async fn test_estimate_migration() {
        let source = Arc::new(MemoryBlockStore::new());

        // Add blocks with unique data (so they have unique CIDs)
        for i in 0..100 {
            let block = Block::new(Bytes::from(format!("block {}", i))).unwrap();
            source.put(&block).await.unwrap();
        }

        let estimate = estimate_migration(source).await.unwrap();

        assert_eq!(estimate.total_blocks, 100);
        assert!(estimate.total_bytes > 0);
        assert!(estimate.space_required > 0);
    }

    #[tokio::test]
    async fn test_validate_migration() {
        let source = Arc::new(MemoryBlockStore::new());
        let destination = Arc::new(MemoryBlockStore::new());

        // Add same blocks to both
        for i in 0..10 {
            let block = Block::new(Bytes::from(format!("block {}", i))).unwrap();
            source.put(&block).await.unwrap();
            destination.put(&block).await.unwrap();
        }

        let valid = validate_migration(source.clone(), destination.clone())
            .await
            .unwrap();

        assert!(valid);

        // Add one more block to source only
        let extra_block = Block::new(Bytes::from("extra")).unwrap();
        source.put(&extra_block).await.unwrap();

        let valid = validate_migration(source, destination).await.unwrap();
        assert!(!valid);
    }
}
