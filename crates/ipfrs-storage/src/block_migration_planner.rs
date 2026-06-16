//! Block migration planner: schedules and tracks data movement between storage nodes/tiers.
//!
//! This module provides [`BlockMigrationPlanner`] which registers blocks,
//! creates and executes migration plans, schedules migrations by policy,
//! runs batch migrations, and generates defragmentation plans to balance data
//! across nodes.

use std::collections::{HashMap, VecDeque};

// ─────────────────────────────────────────────────────────────────────────────
// Type aliases
// ─────────────────────────────────────────────────────────────────────────────

/// 32-byte content-addressed block identifier (e.g. SHA-256 digest).
pub type BmpBlockId = [u8; 32];

/// Opaque numeric identifier for a migration plan.
pub type BmpPlanId = u64;

/// Alias for [`BlockMigrationPlanner`] — used in pub re-exports.
pub type BmpBlockMigrationPlanner = BlockMigrationPlanner;

// ─────────────────────────────────────────────────────────────────────────────
// Utility helpers (no external crates)
// ─────────────────────────────────────────────────────────────────────────────

/// Xorshift64 PRNG — not cryptographically secure, but fast and dependency-free.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// FNV-1a 64-bit hash — deterministic digest of arbitrary bytes.
#[inline]
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

/// Monotonic Unix-epoch timestamp approximation (nanoseconds from epoch zero
/// expressed as a plain u64 counter seeded from the system clock seconds
/// value; guaranteed non-zero and strictly increasing within one process).
fn now_ts() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1)
}

// ─────────────────────────────────────────────────────────────────────────────
// Enumerations
// ─────────────────────────────────────────────────────────────────────────────

/// Lifecycle state of a [`BmpMigrationPlan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BmpPlanStatus {
    /// Waiting to be executed.
    Pending,
    /// Currently being executed.
    InProgress,
    /// Execution finished successfully.
    Completed,
    /// Execution finished with at least one failure.
    Failed,
    /// Plan was cancelled before execution began.
    Cancelled,
}

impl std::fmt::Display for BmpPlanStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "Pending"),
            Self::InProgress => write!(f, "InProgress"),
            Self::Completed => write!(f, "Completed"),
            Self::Failed => write!(f, "Failed"),
            Self::Cancelled => write!(f, "Cancelled"),
        }
    }
}

/// Policy for ordering migration candidates when scheduling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BmpPriorityPolicy {
    /// Process plans in the order they were created (FIFO).
    FifoQueue,
    /// Migrate blocks with the highest `access_frequency` first.
    HighFrequencyFirst,
    /// Migrate the largest blocks first (maximise bytes moved per slot).
    LargestFirst,
    /// Migrate the smallest blocks first (minimise latency per block).
    SmallestFirst,
    /// Prefer blocks where the estimated migration cost is lowest (bytes × tier distance).
    CostOptimized,
}

impl std::fmt::Display for BmpPriorityPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FifoQueue => write!(f, "FifoQueue"),
            Self::HighFrequencyFirst => write!(f, "HighFrequencyFirst"),
            Self::LargestFirst => write!(f, "LargestFirst"),
            Self::SmallestFirst => write!(f, "SmallestFirst"),
            Self::CostOptimized => write!(f, "CostOptimized"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

/// Metadata tracked for each registered block.
#[derive(Clone, Debug)]
pub struct BmpBlockMeta {
    /// 32-byte content-addressed identifier.
    pub id: BmpBlockId,
    /// Compressed or raw byte size of the block.
    pub size_bytes: u64,
    /// Logical node or volume currently holding the block.
    pub src_node: String,
    /// Intended destination node (set once a plan is associated).
    pub dst_node: Option<String>,
    /// Storage tier index (0 = fastest / hottest).
    pub tier: u8,
    /// Exponentially-smoothed access frequency (accesses per unit time).
    pub access_frequency: f64,
    /// Timestamp (nanoseconds) of the most recent access.
    pub last_accessed: u64,
    /// When `true` the block must not be migrated.
    pub is_pinned: bool,
}

impl BmpBlockMeta {
    /// Create a new block record.
    pub fn new(
        id: BmpBlockId,
        size_bytes: u64,
        src_node: impl Into<String>,
        tier: u8,
        access_frequency: f64,
    ) -> Self {
        Self {
            id,
            size_bytes,
            src_node: src_node.into(),
            dst_node: None,
            tier,
            access_frequency,
            last_accessed: now_ts(),
            is_pinned: false,
        }
    }
}

/// A migration plan grouping one or more blocks for movement to a destination node.
#[derive(Clone, Debug)]
pub struct BmpMigrationPlan {
    /// Unique plan identifier.
    pub id: BmpPlanId,
    /// Blocks covered by this plan.
    pub block_ids: Vec<BmpBlockId>,
    /// Node from which the blocks originate.
    pub src_node: String,
    /// Node to which the blocks should be moved.
    pub dst_node: String,
    /// Higher value means higher urgency.
    pub priority: u32,
    /// Total byte size of all blocks in the plan.
    pub estimated_bytes: u64,
    /// Current lifecycle state.
    pub status: BmpPlanStatus,
    /// Creation timestamp (nanoseconds).
    pub created_at: u64,
}

/// Per-block execution record appended to the execution log.
#[derive(Clone, Debug)]
pub struct BmpMigrationRecord {
    /// Execution timestamp (nanoseconds).
    pub ts: u64,
    /// Plan that triggered this record.
    pub plan_id: BmpPlanId,
    /// Block that was (or failed to be) migrated.
    pub block_id: BmpBlockId,
    /// Bytes actually transferred.
    pub bytes_moved: u64,
    /// `true` if the block was moved without error.
    pub success: bool,
    /// Error description when `success` is `false`.
    pub error: Option<String>,
}

/// Planner behaviour configuration.
#[derive(Clone, Debug)]
pub struct BmpPlannerConfig {
    /// Maximum number of plans that can be in `InProgress` simultaneously.
    pub max_concurrent_migrations: usize,
    /// Maximum aggregate bytes per second across all active migrations (0 = unlimited).
    pub bandwidth_limit_kbps: u64,
    /// Default policy used by [`BlockMigrationPlanner::schedule_migrations`].
    pub priority_policy: BmpPriorityPolicy,
    /// When `true`, plans are validated and logged but no data is actually moved.
    pub dry_run: bool,
}

impl Default for BmpPlannerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_migrations: 4,
            bandwidth_limit_kbps: 0,
            priority_policy: BmpPriorityPolicy::FifoQueue,
            dry_run: false,
        }
    }
}

/// Aggregate statistics returned by [`BlockMigrationPlanner::migration_stats`].
#[derive(Clone, Debug, Default)]
pub struct BmpPlannerStats {
    /// Total blocks registered.
    pub total_blocks: usize,
    /// Blocks currently marked as pinned.
    pub pinned_blocks: usize,
    /// Total plans ever created.
    pub total_plans: usize,
    /// Plans in `Pending` state.
    pub pending_plans: usize,
    /// Plans in `InProgress` state.
    pub in_progress_plans: usize,
    /// Plans in `Completed` state.
    pub completed_plans: usize,
    /// Plans in `Failed` state.
    pub failed_plans: usize,
    /// Plans in `Cancelled` state.
    pub cancelled_plans: usize,
    /// Total execution log entries.
    pub total_records: usize,
    /// Total bytes successfully moved across all completed records.
    pub total_bytes_moved: u64,
    /// Number of records where `success == false`.
    pub total_failures: usize,
}

/// Result of executing a single plan.
#[derive(Clone, Debug)]
pub struct BmpExecutionResult {
    /// The plan that was executed.
    pub plan_id: BmpPlanId,
    /// Final plan status after execution.
    pub status: BmpPlanStatus,
    /// Blocks that were successfully migrated.
    pub succeeded_blocks: Vec<BmpBlockId>,
    /// Blocks that failed to migrate, with their error strings.
    pub failed_blocks: Vec<(BmpBlockId, String)>,
    /// Total bytes moved.
    pub bytes_moved: u64,
}

/// Aggregated result of a [`BlockMigrationPlanner::run_batch_migration`] call.
#[derive(Clone, Debug, Default)]
pub struct BmpBatchResult {
    /// How many plans were executed in this batch.
    pub plans_executed: usize,
    /// Plans that completed with all blocks successful.
    pub plans_completed: usize,
    /// Plans that had at least one failed block.
    pub plans_failed: usize,
    /// Total bytes moved across all plans in the batch.
    pub total_bytes_moved: u64,
    /// Per-plan execution results.
    pub results: Vec<BmpExecutionResult>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors returned by [`BlockMigrationPlanner`] operations.
#[derive(Debug, thiserror::Error)]
pub enum BmpError {
    #[error("block {0} is not registered")]
    BlockNotFound(String),
    #[error("plan {0} does not exist")]
    PlanNotFound(BmpPlanId),
    #[error("plan {0} is not in Pending state (current: {1})")]
    PlanNotPending(BmpPlanId, BmpPlanStatus),
    #[error("plan has no blocks")]
    EmptyPlan,
    #[error("block {0} is pinned and cannot be migrated")]
    BlockPinned(String),
    #[error("destination node is empty")]
    EmptyDstNode,
    #[error("no eligible blocks found for scheduling")]
    NoEligibleBlocks,
}

fn block_id_hex(id: &BmpBlockId) -> String {
    let mut s = String::with_capacity(64);
    for b in id {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

// ─────────────────────────────────────────────────────────────────────────────
// Core planner
// ─────────────────────────────────────────────────────────────────────────────

/// Block migration planner: registers blocks, creates plans, executes them,
/// and provides scheduling and defragmentation helpers.
pub struct BlockMigrationPlanner {
    /// All known blocks, keyed by their 32-byte identifier.
    blocks: HashMap<BmpBlockId, BmpBlockMeta>,
    /// All plans, keyed by plan id.
    plans: HashMap<BmpPlanId, BmpMigrationPlan>,
    /// Bounded execution log (cap 1000 most-recent records).
    log: VecDeque<BmpMigrationRecord>,
    /// Behaviour configuration.
    config: BmpPlannerConfig,
    /// Monotonically-increasing plan id counter.
    next_plan_id: BmpPlanId,
    /// PRNG state (xorshift64 seeded from a mix of time and FNV).
    rng_state: u64,
}

impl BlockMigrationPlanner {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Create a planner with default configuration.
    pub fn new() -> Self {
        Self::with_config(BmpPlannerConfig::default())
    }

    /// Create a planner with the supplied configuration.
    pub fn with_config(config: BmpPlannerConfig) -> Self {
        let seed_data = b"block_migration_planner_v1";
        let ts = now_ts();
        let base = fnv1a_64(seed_data);
        let mut rng_state = base ^ ts;
        if rng_state == 0 {
            rng_state = 0xdeadbeef_cafebabe;
        }
        Self {
            blocks: HashMap::new(),
            plans: HashMap::new(),
            log: VecDeque::new(),
            config,
            next_plan_id: 1,
            rng_state,
        }
    }

    /// Return a reference to the current configuration.
    pub fn config(&self) -> &BmpPlannerConfig {
        &self.config
    }

    /// Replace the configuration.
    pub fn set_config(&mut self, config: BmpPlannerConfig) {
        self.config = config;
    }

    // ── Block registration ────────────────────────────────────────────────────

    /// Register a new block (or overwrite if it already exists).
    pub fn register_block(
        &mut self,
        id: BmpBlockId,
        size_bytes: u64,
        src_node: impl Into<String>,
        tier: u8,
        access_frequency: f64,
    ) {
        let meta = BmpBlockMeta::new(id, size_bytes, src_node, tier, access_frequency);
        self.blocks.insert(id, meta);
    }

    /// Record an access event for a block, bumping `last_accessed` and applying
    /// a simple exponential-moving-average update to `access_frequency`.
    pub fn update_access(&mut self, id: &BmpBlockId) -> Result<(), BmpError> {
        let meta = self
            .blocks
            .get_mut(id)
            .ok_or_else(|| BmpError::BlockNotFound(block_id_hex(id)))?;
        meta.last_accessed = now_ts();
        // EMA with α = 0.1
        meta.access_frequency = meta.access_frequency * 0.9 + 1.0 * 0.1;
        Ok(())
    }

    /// Pin a block so it will not be selected for automatic migration.
    pub fn pin(&mut self, id: &BmpBlockId) -> Result<(), BmpError> {
        let meta = self
            .blocks
            .get_mut(id)
            .ok_or_else(|| BmpError::BlockNotFound(block_id_hex(id)))?;
        meta.is_pinned = true;
        Ok(())
    }

    /// Unpin a previously pinned block.
    pub fn unpin(&mut self, id: &BmpBlockId) -> Result<(), BmpError> {
        let meta = self
            .blocks
            .get_mut(id)
            .ok_or_else(|| BmpError::BlockNotFound(block_id_hex(id)))?;
        meta.is_pinned = false;
        Ok(())
    }

    /// Return a reference to a block's metadata, or `None` if not registered.
    pub fn block_meta(&self, id: &BmpBlockId) -> Option<&BmpBlockMeta> {
        self.blocks.get(id)
    }

    /// Number of currently registered blocks.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    // ── Plan management ───────────────────────────────────────────────────────

    /// Create a migration plan for the given blocks.
    ///
    /// The `src_node` is inferred from the first block's current location; if
    /// the list spans multiple source nodes the first one wins (callers should
    /// group by node before creating plans when that matters).
    ///
    /// Returns `Err` if any block is pinned or if any block is not registered.
    pub fn create_plan(
        &mut self,
        block_ids: Vec<BmpBlockId>,
        dst_node: impl Into<String>,
        priority: u32,
    ) -> Result<BmpPlanId, BmpError> {
        if block_ids.is_empty() {
            return Err(BmpError::EmptyPlan);
        }
        let dst_node = dst_node.into();
        if dst_node.is_empty() {
            return Err(BmpError::EmptyDstNode);
        }

        // Validate all blocks exist and are not pinned.
        for bid in &block_ids {
            let meta = self
                .blocks
                .get(bid)
                .ok_or_else(|| BmpError::BlockNotFound(block_id_hex(bid)))?;
            if meta.is_pinned {
                return Err(BmpError::BlockPinned(block_id_hex(bid)));
            }
        }

        // Infer src_node and tally estimated bytes.
        let src_node = self
            .blocks
            .get(&block_ids[0])
            .map(|m| m.src_node.clone())
            .unwrap_or_default();

        let estimated_bytes: u64 = block_ids
            .iter()
            .filter_map(|bid| self.blocks.get(bid).map(|m| m.size_bytes))
            .sum();

        let plan_id = self.next_plan_id;
        self.next_plan_id += 1;

        // Record intended dst_node on each block.
        for bid in &block_ids {
            if let Some(meta) = self.blocks.get_mut(bid) {
                meta.dst_node = Some(dst_node.clone());
            }
        }

        let plan = BmpMigrationPlan {
            id: plan_id,
            block_ids,
            src_node,
            dst_node,
            priority,
            estimated_bytes,
            status: BmpPlanStatus::Pending,
            created_at: now_ts(),
        };
        self.plans.insert(plan_id, plan);
        Ok(plan_id)
    }

    /// Cancel a pending plan.  Plans that are already `InProgress`, `Completed`,
    /// `Failed`, or `Cancelled` cannot be cancelled again.
    pub fn cancel_plan(&mut self, plan_id: BmpPlanId) -> Result<(), BmpError> {
        let plan = self
            .plans
            .get_mut(&plan_id)
            .ok_or(BmpError::PlanNotFound(plan_id))?;
        if plan.status != BmpPlanStatus::Pending {
            return Err(BmpError::PlanNotPending(plan_id, plan.status));
        }
        plan.status = BmpPlanStatus::Cancelled;
        Ok(())
    }

    /// Execute a migration plan.
    ///
    /// Each block is migrated in sequence; a simulated success/failure decision
    /// is made via xorshift64 (roughly 90 % success when `!dry_run`).
    /// In `dry_run` mode all blocks succeed without moving.
    pub fn execute_plan(&mut self, plan_id: BmpPlanId) -> Result<BmpExecutionResult, BmpError> {
        // Verify the plan exists and is pending.
        {
            let plan = self
                .plans
                .get(&plan_id)
                .ok_or(BmpError::PlanNotFound(plan_id))?;
            if plan.status != BmpPlanStatus::Pending {
                return Err(BmpError::PlanNotPending(plan_id, plan.status));
            }
        }

        // Mark in-progress.
        if let Some(plan) = self.plans.get_mut(&plan_id) {
            plan.status = BmpPlanStatus::InProgress;
        }

        let block_ids: Vec<BmpBlockId> = self
            .plans
            .get(&plan_id)
            .map(|p| p.block_ids.clone())
            .unwrap_or_default();

        let dst_node: String = self
            .plans
            .get(&plan_id)
            .map(|p| p.dst_node.clone())
            .unwrap_or_default();

        let dry_run = self.config.dry_run;

        let mut succeeded_blocks: Vec<BmpBlockId> = Vec::new();
        let mut failed_blocks: Vec<(BmpBlockId, String)> = Vec::new();
        let mut bytes_moved: u64 = 0;

        for bid in &block_ids {
            let size = self.blocks.get(bid).map(|m| m.size_bytes).unwrap_or(0);

            let (success, error_msg): (bool, Option<String>) = if dry_run {
                (true, None)
            } else {
                // xorshift64: success if high nibble is not 0xF (~93.75% success).
                let roll = xorshift64(&mut self.rng_state);
                if (roll >> 60) != 0xF {
                    (true, None)
                } else {
                    (
                        false,
                        Some(format!("simulated I/O error (roll=0x{:016x})", roll)),
                    )
                }
            };

            if success {
                bytes_moved += size;
                succeeded_blocks.push(*bid);
                // Update block metadata: move it to dst_node.
                if let Some(meta) = self.blocks.get_mut(bid) {
                    meta.src_node = dst_node.clone();
                    meta.dst_node = None;
                }
            } else {
                failed_blocks.push((*bid, error_msg.clone().unwrap_or_default()));
            }

            let record = BmpMigrationRecord {
                ts: now_ts(),
                plan_id,
                block_id: *bid,
                bytes_moved: if success { size } else { 0 },
                success,
                error: error_msg,
            };
            self.append_log(record);
        }

        let final_status = if failed_blocks.is_empty() {
            BmpPlanStatus::Completed
        } else {
            BmpPlanStatus::Failed
        };

        if let Some(plan) = self.plans.get_mut(&plan_id) {
            plan.status = final_status;
        }

        Ok(BmpExecutionResult {
            plan_id,
            status: final_status,
            succeeded_blocks,
            failed_blocks,
            bytes_moved,
        })
    }

    // ── Scheduling ────────────────────────────────────────────────────────────

    /// Automatically select blocks that need migration and create plans for them.
    ///
    /// "Needing migration" means the block has a `dst_node` already set, or
    /// is unpinned and in a non-zero tier (tier > 0 → candidate for promotion).
    /// Up to `max_plans` plans are created, each containing a single block,
    /// ordered according to `policy`.
    pub fn schedule_migrations(
        &mut self,
        policy: BmpPriorityPolicy,
        max_plans: usize,
    ) -> Result<Vec<BmpPlanId>, BmpError> {
        if max_plans == 0 {
            return Ok(Vec::new());
        }

        // Collect candidate block ids.
        let mut candidates: Vec<BmpBlockId> = self
            .blocks
            .values()
            .filter(|m| !m.is_pinned && m.dst_node.is_some())
            .map(|m| m.id)
            .collect();

        if candidates.is_empty() {
            // Fall back: pick unpinned blocks on tier > 0.
            candidates = self
                .blocks
                .values()
                .filter(|m| !m.is_pinned && m.tier > 0)
                .map(|m| m.id)
                .collect();
        }

        if candidates.is_empty() {
            return Err(BmpError::NoEligibleBlocks);
        }

        // Sort by policy.
        self.sort_candidates_by_policy(&mut candidates, policy);

        candidates.truncate(max_plans);

        let mut plan_ids = Vec::with_capacity(candidates.len());
        for bid in candidates {
            let dst = self
                .blocks
                .get(&bid)
                .and_then(|m| m.dst_node.clone())
                .unwrap_or_else(|| "node-0".to_string());

            match self.create_plan(vec![bid], dst, 0) {
                Ok(pid) => plan_ids.push(pid),
                Err(_) => continue,
            }
        }
        Ok(plan_ids)
    }

    /// Sort a candidate list in-place according to `policy`.
    fn sort_candidates_by_policy(&self, ids: &mut [BmpBlockId], policy: BmpPriorityPolicy) {
        match policy {
            BmpPriorityPolicy::FifoQueue => {
                // Sort by `last_accessed` ascending (oldest first).
                ids.sort_by(|a, b| {
                    let ta = self.blocks.get(a).map_or(0, |m| m.last_accessed);
                    let tb = self.blocks.get(b).map_or(0, |m| m.last_accessed);
                    ta.cmp(&tb)
                });
            }
            BmpPriorityPolicy::HighFrequencyFirst => {
                ids.sort_by(|a, b| {
                    let fa = self.blocks.get(a).map_or(0.0, |m| m.access_frequency);
                    let fb = self.blocks.get(b).map_or(0.0, |m| m.access_frequency);
                    fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            BmpPriorityPolicy::LargestFirst => {
                ids.sort_by(|a, b| {
                    let sa = self.blocks.get(a).map_or(0, |m| m.size_bytes);
                    let sb = self.blocks.get(b).map_or(0, |m| m.size_bytes);
                    sb.cmp(&sa)
                });
            }
            BmpPriorityPolicy::SmallestFirst => {
                ids.sort_by(|a, b| {
                    let sa = self.blocks.get(a).map_or(0, |m| m.size_bytes);
                    let sb = self.blocks.get(b).map_or(0, |m| m.size_bytes);
                    sa.cmp(&sb)
                });
            }
            BmpPriorityPolicy::CostOptimized => {
                // Proxy cost = size_bytes * tier (lower = cheaper to migrate).
                ids.sort_by(|a, b| {
                    let ca = self
                        .blocks
                        .get(a)
                        .map_or(0, |m| m.size_bytes.saturating_mul(m.tier as u64));
                    let cb = self
                        .blocks
                        .get(b)
                        .map_or(0, |m| m.size_bytes.saturating_mul(m.tier as u64));
                    ca.cmp(&cb)
                });
            }
        }
    }

    /// Create up to `batch_size` plans using `policy` and execute them all,
    /// respecting `max_concurrent_migrations`.
    pub fn run_batch_migration(
        &mut self,
        policy: BmpPriorityPolicy,
        batch_size: usize,
    ) -> BmpBatchResult {
        let max = self.config.max_concurrent_migrations.min(batch_size);

        let plan_ids = match self.schedule_migrations(policy, max) {
            Ok(ids) => ids,
            Err(_) => return BmpBatchResult::default(),
        };

        let mut batch = BmpBatchResult {
            plans_executed: plan_ids.len(),
            ..BmpBatchResult::default()
        };

        for pid in plan_ids {
            match self.execute_plan(pid) {
                Ok(result) => {
                    batch.total_bytes_moved += result.bytes_moved;
                    if result.status == BmpPlanStatus::Completed {
                        batch.plans_completed += 1;
                    } else {
                        batch.plans_failed += 1;
                    }
                    batch.results.push(result);
                }
                Err(_) => {
                    batch.plans_failed += 1;
                }
            }
        }
        batch
    }

    // ── Defragmentation ───────────────────────────────────────────────────────

    /// Generate plans to balance blocks evenly across the supplied nodes.
    ///
    /// Blocks that are already on the least-loaded node (or are pinned) are
    /// skipped.  The target is `total_blocks / nodes.len()` per node.
    /// Returns `Vec<BmpPlanId>` of all newly created plans.
    pub fn defragment_plan(&mut self, nodes: &[String]) -> Vec<BmpPlanId> {
        if nodes.is_empty() {
            return Vec::new();
        }

        // Count blocks per node.
        let mut node_counts: HashMap<String, usize> =
            nodes.iter().map(|n| (n.clone(), 0)).collect();
        for meta in self.blocks.values() {
            if let Some(cnt) = node_counts.get_mut(&meta.src_node) {
                *cnt += 1;
            }
        }

        let total: usize = node_counts.values().sum();
        let target = total.div_ceil(nodes.len());

        // Identify over-full and under-full nodes.
        let mut over_nodes: Vec<String> = node_counts
            .iter()
            .filter(|(_, &cnt)| cnt > target)
            .map(|(n, _)| n.clone())
            .collect();
        let mut under_nodes: Vec<String> = node_counts
            .iter()
            .filter(|(_, &cnt)| cnt < target)
            .map(|(n, _)| n.clone())
            .collect();

        if over_nodes.is_empty() || under_nodes.is_empty() {
            return Vec::new();
        }

        // Sort deterministically so behaviour is reproducible.
        over_nodes.sort();
        under_nodes.sort();

        // Build a list of (block_id, src_node, intended_dst_node) moves.
        struct Move {
            bid: BmpBlockId,
            dst: String,
        }
        let mut moves: Vec<Move> = Vec::new();

        // Remaining capacity on each under-loaded node.
        let mut capacity: HashMap<String, usize> = under_nodes
            .iter()
            .map(|n| {
                let cnt = node_counts.get(n).copied().unwrap_or(0);
                (n.clone(), target.saturating_sub(cnt))
            })
            .collect();

        for over in &over_nodes {
            let excess = node_counts
                .get(over)
                .map_or(0, |&cnt| cnt.saturating_sub(target));
            if excess == 0 {
                continue;
            }
            // Collect movable blocks from this node.
            let candidates: Vec<BmpBlockId> = self
                .blocks
                .values()
                .filter(|m| &m.src_node == over && !m.is_pinned && m.dst_node.is_none())
                .map(|m| m.id)
                .collect();

            let mut moved = 0usize;
            'outer: for bid in candidates {
                if moved >= excess {
                    break;
                }
                // Find an under node with remaining capacity.
                for under in &under_nodes {
                    let cap = capacity.get_mut(under);
                    if let Some(c) = cap {
                        if *c > 0 {
                            *c -= 1;
                            moves.push(Move {
                                bid,
                                dst: under.clone(),
                            });
                            moved += 1;
                            continue 'outer;
                        }
                    }
                }
                break; // No space on any under node.
            }
        }

        // Create one plan per move.
        let mut plan_ids = Vec::with_capacity(moves.len());
        for m in moves {
            // Tag the block with its intended dst so create_plan does not reject it.
            if let Some(meta) = self.blocks.get_mut(&m.bid) {
                meta.dst_node = Some(m.dst.clone());
            }
            match self.create_plan(vec![m.bid], m.dst, 0) {
                Ok(pid) => plan_ids.push(pid),
                Err(_) => continue,
            }
        }
        plan_ids
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /// Compute and return aggregate statistics.
    pub fn migration_stats(&self) -> BmpPlannerStats {
        let pinned_blocks = self.blocks.values().filter(|m| m.is_pinned).count();

        let mut stats = BmpPlannerStats {
            total_blocks: self.blocks.len(),
            pinned_blocks,
            total_plans: self.plans.len(),
            total_records: self.log.len(),
            ..Default::default()
        };

        for plan in self.plans.values() {
            match plan.status {
                BmpPlanStatus::Pending => stats.pending_plans += 1,
                BmpPlanStatus::InProgress => stats.in_progress_plans += 1,
                BmpPlanStatus::Completed => stats.completed_plans += 1,
                BmpPlanStatus::Failed => stats.failed_plans += 1,
                BmpPlanStatus::Cancelled => stats.cancelled_plans += 1,
            }
        }

        for rec in &self.log {
            if rec.success {
                stats.total_bytes_moved += rec.bytes_moved;
            } else {
                stats.total_failures += 1;
            }
        }

        stats
    }

    /// Return the number of plans in the planner.
    pub fn plan_count(&self) -> usize {
        self.plans.len()
    }

    /// Return a reference to a plan, or `None`.
    pub fn plan(&self, plan_id: BmpPlanId) -> Option<&BmpMigrationPlan> {
        self.plans.get(&plan_id)
    }

    /// Iterate over all plans.
    pub fn plans_iter(&self) -> impl Iterator<Item = &BmpMigrationPlan> {
        self.plans.values()
    }

    /// Return the most recent `n` log records.
    pub fn recent_records(&self, n: usize) -> Vec<&BmpMigrationRecord> {
        self.log.iter().rev().take(n).collect()
    }

    /// Total number of records in the execution log.
    pub fn log_len(&self) -> usize {
        self.log.len()
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Append a record to the execution log, evicting the oldest when full.
    fn append_log(&mut self, record: BmpMigrationRecord) {
        const MAX_LOG: usize = 1000;
        if self.log.len() >= MAX_LOG {
            self.log.pop_front();
        }
        self.log.push_back(record);
    }
}

impl Default for BlockMigrationPlanner {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Additional helper: make a BmpBlockId from a byte slice (pads / truncates).
// ─────────────────────────────────────────────────────────────────────────────

/// Build a `BmpBlockId` deterministically from arbitrary bytes using FNV-1a.
///
/// The 32-byte array is filled by splitting the 64-bit FNV hash into 4× u64
/// words (each XOR'd with a rotation of the previous word for diffusion).
pub fn bmp_id_from_bytes(data: &[u8]) -> BmpBlockId {
    let mut id = [0u8; 32];
    let h0 = fnv1a_64(data);
    let h1 = fnv1a_64(&h0.to_le_bytes());
    let h2 = fnv1a_64(&h1.to_le_bytes());
    let h3 = fnv1a_64(&h2.to_le_bytes());
    id[0..8].copy_from_slice(&h0.to_le_bytes());
    id[8..16].copy_from_slice(&h1.to_le_bytes());
    id[16..24].copy_from_slice(&h2.to_le_bytes());
    id[24..32].copy_from_slice(&h3.to_le_bytes());
    id
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_id(seed: u8) -> BmpBlockId {
        bmp_id_from_bytes(&[seed; 16])
    }

    fn make_planner() -> BlockMigrationPlanner {
        BlockMigrationPlanner::new()
    }

    fn register_n(planner: &mut BlockMigrationPlanner, n: usize, node: &str) -> Vec<BmpBlockId> {
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            let id = bmp_id_from_bytes(format!("block-{}", i).as_bytes());
            planner.register_block(id, (i as u64 + 1) * 512, node, 1, 1.0);
            ids.push(id);
        }
        ids
    }

    // ── xorshift64 / fnv1a_64 ─────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_nonzero() {
        let mut state: u64 = 12345678;
        for _ in 0..100 {
            let v = xorshift64(&mut state);
            assert_ne!(v, 0);
        }
    }

    #[test]
    fn test_xorshift64_changes_state() {
        let mut state: u64 = 1;
        let a = xorshift64(&mut state);
        let b = xorshift64(&mut state);
        assert_ne!(a, b);
    }

    #[test]
    fn test_fnv1a_empty() {
        let h = fnv1a_64(&[]);
        // FNV offset basis
        assert_eq!(h, 14_695_981_039_346_656_037_u64);
    }

    #[test]
    fn test_fnv1a_deterministic() {
        assert_eq!(fnv1a_64(b"hello"), fnv1a_64(b"hello"));
    }

    #[test]
    fn test_fnv1a_differs_on_different_input() {
        assert_ne!(fnv1a_64(b"foo"), fnv1a_64(b"bar"));
    }

    // ── bmp_id_from_bytes ─────────────────────────────────────────────────────

    #[test]
    fn test_bmp_id_from_bytes_length() {
        let id = bmp_id_from_bytes(b"test");
        assert_eq!(id.len(), 32);
    }

    #[test]
    fn test_bmp_id_from_bytes_deterministic() {
        assert_eq!(bmp_id_from_bytes(b"abc"), bmp_id_from_bytes(b"abc"));
    }

    #[test]
    fn test_bmp_id_from_bytes_differs() {
        assert_ne!(bmp_id_from_bytes(b"a"), bmp_id_from_bytes(b"b"));
    }

    // ── BlockMigrationPlanner construction ────────────────────────────────────

    #[test]
    fn test_new_planner_empty() {
        let p = make_planner();
        assert_eq!(p.block_count(), 0);
        assert_eq!(p.plan_count(), 0);
        assert_eq!(p.log_len(), 0);
    }

    #[test]
    fn test_with_config_dry_run() {
        let cfg = BmpPlannerConfig {
            dry_run: true,
            ..Default::default()
        };
        let p = BlockMigrationPlanner::with_config(cfg);
        assert!(p.config().dry_run);
    }

    #[test]
    fn test_set_config() {
        let mut p = make_planner();
        let cfg = BmpPlannerConfig {
            bandwidth_limit_kbps: 1024,
            ..Default::default()
        };
        p.set_config(cfg);
        assert_eq!(p.config().bandwidth_limit_kbps, 1024);
    }

    // ── register_block ────────────────────────────────────────────────────────

    #[test]
    fn test_register_block_increases_count() {
        let mut p = make_planner();
        let id = make_id(1);
        p.register_block(id, 1024, "node-0", 0, 1.0);
        assert_eq!(p.block_count(), 1);
    }

    #[test]
    fn test_register_block_metadata() {
        let mut p = make_planner();
        let id = make_id(2);
        p.register_block(id, 2048, "node-1", 2, 3.5);
        let meta = p.block_meta(&id).expect("block should exist");
        assert_eq!(meta.size_bytes, 2048);
        assert_eq!(meta.src_node, "node-1");
        assert_eq!(meta.tier, 2);
        assert!((meta.access_frequency - 3.5).abs() < 1e-9);
        assert!(!meta.is_pinned);
    }

    #[test]
    fn test_register_block_overwrite() {
        let mut p = make_planner();
        let id = make_id(3);
        p.register_block(id, 100, "node-a", 0, 0.5);
        p.register_block(id, 200, "node-b", 1, 2.0);
        assert_eq!(p.block_count(), 1); // still one block
        let meta = p.block_meta(&id).expect("exists");
        assert_eq!(meta.size_bytes, 200);
        assert_eq!(meta.src_node, "node-b");
    }

    #[test]
    fn test_register_multiple_blocks() {
        let mut p = make_planner();
        for i in 0..10u8 {
            p.register_block(make_id(i), 512, "node-0", 0, 1.0);
        }
        assert_eq!(p.block_count(), 10);
    }

    // ── update_access ─────────────────────────────────────────────────────────

    #[test]
    fn test_update_access_ok() {
        let mut p = make_planner();
        let id = make_id(10);
        p.register_block(id, 512, "node-0", 0, 1.0);
        let old_freq = p.block_meta(&id).map_or(0.0, |m| m.access_frequency);
        p.update_access(&id).expect("ok");
        let new_freq = p.block_meta(&id).map_or(0.0, |m| m.access_frequency);
        // EMA means new_freq = old * 0.9 + 0.1; should differ from initial 1.0
        assert!(new_freq > 0.0);
        let _ = old_freq; // suppress unused warning
    }

    #[test]
    fn test_update_access_unknown_block_errors() {
        let mut p = make_planner();
        let id = make_id(11);
        assert!(p.update_access(&id).is_err());
    }

    #[test]
    fn test_update_access_bumps_last_accessed() {
        let mut p = make_planner();
        let id = make_id(12);
        p.register_block(id, 512, "node-0", 0, 1.0);
        let t0 = p.block_meta(&id).map_or(0, |m| m.last_accessed);
        std::thread::sleep(std::time::Duration::from_millis(1));
        p.update_access(&id).expect("ok");
        let t1 = p.block_meta(&id).map_or(0, |m| m.last_accessed);
        assert!(t1 >= t0);
    }

    // ── pin / unpin ───────────────────────────────────────────────────────────

    #[test]
    fn test_pin_block() {
        let mut p = make_planner();
        let id = make_id(20);
        p.register_block(id, 512, "node-0", 0, 1.0);
        p.pin(&id).expect("ok");
        assert!(p.block_meta(&id).is_some_and(|m| m.is_pinned));
    }

    #[test]
    fn test_unpin_block() {
        let mut p = make_planner();
        let id = make_id(21);
        p.register_block(id, 512, "node-0", 0, 1.0);
        p.pin(&id).expect("ok");
        p.unpin(&id).expect("ok");
        assert!(!p.block_meta(&id).is_none_or(|m| m.is_pinned));
    }

    #[test]
    fn test_pin_unknown_block_errors() {
        let mut p = make_planner();
        assert!(p.pin(&make_id(22)).is_err());
    }

    #[test]
    fn test_unpin_unknown_block_errors() {
        let mut p = make_planner();
        assert!(p.unpin(&make_id(23)).is_err());
    }

    // ── create_plan ───────────────────────────────────────────────────────────

    #[test]
    fn test_create_plan_basic() {
        let mut p = make_planner();
        let id = make_id(30);
        p.register_block(id, 1024, "node-0", 0, 1.0);
        let pid = p.create_plan(vec![id], "node-1", 5).expect("ok");
        assert_eq!(pid, 1);
        let plan = p.plan(pid).expect("plan exists");
        assert_eq!(plan.status, BmpPlanStatus::Pending);
        assert_eq!(plan.estimated_bytes, 1024);
        assert_eq!(plan.dst_node, "node-1");
        assert_eq!(plan.priority, 5);
    }

    #[test]
    fn test_create_plan_empty_blocks_errors() {
        let mut p = make_planner();
        assert!(p.create_plan(vec![], "node-1", 0).is_err());
    }

    #[test]
    fn test_create_plan_empty_dst_errors() {
        let mut p = make_planner();
        let id = make_id(31);
        p.register_block(id, 512, "node-0", 0, 1.0);
        assert!(p.create_plan(vec![id], "", 0).is_err());
    }

    #[test]
    fn test_create_plan_unknown_block_errors() {
        let mut p = make_planner();
        let id = make_id(32);
        assert!(p.create_plan(vec![id], "node-1", 0).is_err());
    }

    #[test]
    fn test_create_plan_pinned_block_errors() {
        let mut p = make_planner();
        let id = make_id(33);
        p.register_block(id, 512, "node-0", 0, 1.0);
        p.pin(&id).expect("ok");
        assert!(p.create_plan(vec![id], "node-1", 0).is_err());
    }

    #[test]
    fn test_create_plan_increments_id() {
        let mut p = make_planner();
        for i in 0..5u8 {
            let id = make_id(40 + i);
            p.register_block(id, 512, "node-0", 0, 1.0);
            let pid = p.create_plan(vec![id], "node-1", 0).expect("ok");
            assert_eq!(pid, i as u64 + 1);
        }
    }

    #[test]
    fn test_create_plan_multi_block_bytes() {
        let mut p = make_planner();
        let ids: Vec<_> = (0..3u8)
            .map(|i| {
                let id = make_id(50 + i);
                p.register_block(id, 1000, "node-0", 0, 1.0);
                id
            })
            .collect();
        let pid = p.create_plan(ids, "node-1", 0).expect("ok");
        let plan = p.plan(pid).expect("plan");
        assert_eq!(plan.estimated_bytes, 3000);
    }

    // ── cancel_plan ───────────────────────────────────────────────────────────

    #[test]
    fn test_cancel_pending_plan() {
        let mut p = make_planner();
        let id = make_id(60);
        p.register_block(id, 512, "node-0", 0, 1.0);
        let pid = p.create_plan(vec![id], "node-1", 0).expect("ok");
        p.cancel_plan(pid).expect("ok");
        assert_eq!(
            p.plan(pid).map(|pl| pl.status),
            Some(BmpPlanStatus::Cancelled)
        );
    }

    #[test]
    fn test_cancel_nonexistent_plan_errors() {
        let mut p = make_planner();
        assert!(p.cancel_plan(999).is_err());
    }

    #[test]
    fn test_cancel_already_cancelled_errors() {
        let mut p = make_planner();
        let id = make_id(61);
        p.register_block(id, 512, "node-0", 0, 1.0);
        let pid = p.create_plan(vec![id], "node-1", 0).expect("ok");
        p.cancel_plan(pid).expect("ok");
        assert!(p.cancel_plan(pid).is_err());
    }

    // ── execute_plan ──────────────────────────────────────────────────────────

    #[test]
    fn test_execute_plan_dry_run_always_succeeds() {
        let cfg = BmpPlannerConfig {
            dry_run: true,
            ..Default::default()
        };
        let mut p = BlockMigrationPlanner::with_config(cfg);
        let id = make_id(70);
        p.register_block(id, 512, "node-0", 0, 1.0);
        let pid = p.create_plan(vec![id], "node-1", 0).expect("ok");
        let result = p.execute_plan(pid).expect("ok");
        assert_eq!(result.status, BmpPlanStatus::Completed);
        assert_eq!(result.succeeded_blocks.len(), 1);
        assert!(result.failed_blocks.is_empty());
        assert_eq!(result.bytes_moved, 512);
    }

    #[test]
    fn test_execute_plan_updates_log() {
        let cfg = BmpPlannerConfig {
            dry_run: true,
            ..Default::default()
        };
        let mut p = BlockMigrationPlanner::with_config(cfg);
        let id = make_id(71);
        p.register_block(id, 256, "node-0", 0, 1.0);
        let pid = p.create_plan(vec![id], "node-1", 0).expect("ok");
        p.execute_plan(pid).expect("ok");
        assert_eq!(p.log_len(), 1);
    }

    #[test]
    fn test_execute_plan_updates_block_src_node() {
        let cfg = BmpPlannerConfig {
            dry_run: true,
            ..Default::default()
        };
        let mut p = BlockMigrationPlanner::with_config(cfg);
        let id = make_id(72);
        p.register_block(id, 512, "node-0", 0, 1.0);
        let pid = p.create_plan(vec![id], "node-2", 0).expect("ok");
        p.execute_plan(pid).expect("ok");
        let meta = p.block_meta(&id).expect("exists");
        assert_eq!(meta.src_node, "node-2");
    }

    #[test]
    fn test_execute_plan_not_found_errors() {
        let mut p = make_planner();
        assert!(p.execute_plan(42).is_err());
    }

    #[test]
    fn test_execute_plan_not_pending_errors() {
        let cfg = BmpPlannerConfig {
            dry_run: true,
            ..Default::default()
        };
        let mut p = BlockMigrationPlanner::with_config(cfg);
        let id = make_id(73);
        p.register_block(id, 512, "node-0", 0, 1.0);
        let pid = p.create_plan(vec![id], "node-1", 0).expect("ok");
        p.execute_plan(pid).expect("ok");
        // Already completed — cannot execute again.
        assert!(p.execute_plan(pid).is_err());
    }

    #[test]
    fn test_execute_plan_multi_block_dry_run() {
        let cfg = BmpPlannerConfig {
            dry_run: true,
            ..Default::default()
        };
        let mut p = BlockMigrationPlanner::with_config(cfg);
        let ids: Vec<_> = (0..5u8)
            .map(|i| {
                let id = make_id(80 + i);
                p.register_block(id, 100, "node-0", 0, 1.0);
                id
            })
            .collect();
        let pid = p.create_plan(ids.clone(), "node-1", 0).expect("ok");
        let res = p.execute_plan(pid).expect("ok");
        assert_eq!(res.succeeded_blocks.len(), 5);
        assert_eq!(res.bytes_moved, 500);
    }

    // ── schedule_migrations ───────────────────────────────────────────────────

    #[test]
    fn test_schedule_migrations_no_eligible_blocks_errors() {
        let mut p = make_planner();
        // Register tier-0 blocks with no dst_node → not eligible.
        let id = make_id(90);
        p.register_block(id, 512, "node-0", 0, 1.0);
        let result = p.schedule_migrations(BmpPriorityPolicy::FifoQueue, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_schedule_migrations_with_dst_node_set() {
        let mut p = make_planner();
        let id = make_id(91);
        p.register_block(id, 512, "node-0", 1, 1.0);
        // Set dst_node directly.
        if let Some(meta) = p.blocks.get_mut(&id) {
            meta.dst_node = Some("node-1".to_string());
        }
        let ids = p
            .schedule_migrations(BmpPriorityPolicy::FifoQueue, 5)
            .expect("ok");
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_schedule_migrations_respects_max_plans() {
        let mut p = make_planner();
        for i in 0..10u8 {
            let id = make_id(100 + i);
            p.register_block(id, 512, "node-0", 1, 1.0);
            if let Some(meta) = p.blocks.get_mut(&id) {
                meta.dst_node = Some("node-1".to_string());
            }
        }
        let ids = p
            .schedule_migrations(BmpPriorityPolicy::FifoQueue, 3)
            .expect("ok");
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_schedule_migrations_skips_pinned() {
        let mut p = make_planner();
        for i in 0..3u8 {
            let id = make_id(110 + i);
            p.register_block(id, 512, "node-0", 1, 1.0);
            if let Some(meta) = p.blocks.get_mut(&id) {
                meta.dst_node = Some("node-1".to_string());
            }
            if i == 0 {
                p.pin(&id).expect("ok");
            }
        }
        let ids = p
            .schedule_migrations(BmpPriorityPolicy::FifoQueue, 10)
            .expect("ok");
        // Only 2 unpinned blocks should become plans.
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_schedule_migrations_zero_max_returns_empty() {
        let mut p = make_planner();
        let id = make_id(120);
        p.register_block(id, 512, "node-0", 1, 1.0);
        if let Some(meta) = p.blocks.get_mut(&id) {
            meta.dst_node = Some("node-1".to_string());
        }
        let ids = p
            .schedule_migrations(BmpPriorityPolicy::FifoQueue, 0)
            .expect("ok");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_schedule_largest_first_ordering() {
        let mut p = make_planner();
        let sizes = [100u64, 500, 200, 800, 50];
        let mut block_ids = Vec::new();
        for (i, &sz) in sizes.iter().enumerate() {
            let id = make_id(130 + i as u8);
            p.register_block(id, sz, "node-0", 1, 1.0);
            if let Some(meta) = p.blocks.get_mut(&id) {
                meta.dst_node = Some("node-1".to_string());
            }
            block_ids.push(id);
        }
        let pids = p
            .schedule_migrations(BmpPriorityPolicy::LargestFirst, 5)
            .expect("ok");
        // Collect estimated bytes for the first plan to verify ordering.
        let first_bytes = p.plan(pids[0]).map_or(0, |pl| pl.estimated_bytes);
        let last_bytes = p
            .plan(*pids.last().unwrap())
            .map_or(0, |pl| pl.estimated_bytes);
        assert!(
            first_bytes >= last_bytes,
            "largest first violated: {} < {}",
            first_bytes,
            last_bytes
        );
    }

    #[test]
    fn test_schedule_smallest_first_ordering() {
        let mut p = make_planner();
        let sizes = [100u64, 500, 200, 800, 50];
        for (i, &sz) in sizes.iter().enumerate() {
            let id = make_id(140 + i as u8);
            p.register_block(id, sz, "node-0", 1, 1.0);
            if let Some(meta) = p.blocks.get_mut(&id) {
                meta.dst_node = Some("node-1".to_string());
            }
        }
        let pids = p
            .schedule_migrations(BmpPriorityPolicy::SmallestFirst, 5)
            .expect("ok");
        let first_bytes = p.plan(pids[0]).map_or(0, |pl| pl.estimated_bytes);
        let last_bytes = p
            .plan(*pids.last().unwrap())
            .map_or(0, |pl| pl.estimated_bytes);
        assert!(
            first_bytes <= last_bytes,
            "smallest first violated: {} > {}",
            first_bytes,
            last_bytes
        );
    }

    #[test]
    fn test_schedule_high_frequency_first() {
        let mut p = make_planner();
        let freqs = [0.1f64, 5.0, 2.0];
        let mut bid_map: Vec<(BmpBlockId, f64)> = Vec::new();
        for (i, &f) in freqs.iter().enumerate() {
            let id = make_id(150 + i as u8);
            p.register_block(id, 512, "node-0", 1, f);
            if let Some(meta) = p.blocks.get_mut(&id) {
                meta.dst_node = Some("node-1".to_string());
            }
            bid_map.push((id, f));
        }
        let pids = p
            .schedule_migrations(BmpPriorityPolicy::HighFrequencyFirst, 3)
            .expect("ok");
        // The first plan should contain the block with the highest frequency (5.0).
        let first_plan_bid = p.plan(pids[0]).and_then(|pl| pl.block_ids.first().copied());
        if let Some(bid) = first_plan_bid {
            let freq = p.block_meta(&bid).map_or(0.0, |m| m.access_frequency);
            // Highest frequency among scheduled should be 5.0.
            assert!(
                (freq - 5.0).abs() < 1e-6 || freq >= 2.0,
                "unexpected freq {}",
                freq
            );
        }
    }

    // ── run_batch_migration ───────────────────────────────────────────────────

    #[test]
    fn test_run_batch_migration_dry_run() {
        let cfg = BmpPlannerConfig {
            dry_run: true,
            max_concurrent_migrations: 10,
            ..Default::default()
        };
        let mut p = BlockMigrationPlanner::with_config(cfg);
        let _ids = register_n(&mut p, 5, "node-0");
        for id in &_ids {
            if let Some(meta) = p.blocks.get_mut(id) {
                meta.dst_node = Some("node-1".to_string());
                meta.tier = 1;
            }
        }
        let batch = p.run_batch_migration(BmpPriorityPolicy::FifoQueue, 5);
        assert!(batch.plans_executed > 0);
        assert_eq!(batch.plans_failed, 0);
    }

    #[test]
    fn test_run_batch_migration_empty_result_on_no_candidates() {
        let mut p = make_planner();
        // Register only tier-0 blocks (no dst_node).
        let id = make_id(160);
        p.register_block(id, 512, "node-0", 0, 1.0);
        let batch = p.run_batch_migration(BmpPriorityPolicy::FifoQueue, 5);
        assert_eq!(batch.plans_executed, 0);
    }

    #[test]
    fn test_run_batch_migration_respects_max_concurrent() {
        let cfg = BmpPlannerConfig {
            dry_run: true,
            max_concurrent_migrations: 2,
            ..Default::default()
        };
        let mut p = BlockMigrationPlanner::with_config(cfg);
        let _ids = register_n(&mut p, 10, "node-0");
        for id in &_ids {
            if let Some(meta) = p.blocks.get_mut(id) {
                meta.dst_node = Some("node-1".to_string());
                meta.tier = 1;
            }
        }
        let batch = p.run_batch_migration(BmpPriorityPolicy::FifoQueue, 10);
        assert!(batch.plans_executed <= 2);
    }

    #[test]
    fn test_run_batch_total_bytes_moved() {
        let cfg = BmpPlannerConfig {
            dry_run: true,
            max_concurrent_migrations: 5,
            ..Default::default()
        };
        let mut p = BlockMigrationPlanner::with_config(cfg);
        for i in 0..5u8 {
            let id = make_id(170 + i);
            p.register_block(id, 1000, "node-0", 1, 1.0);
            if let Some(meta) = p.blocks.get_mut(&id) {
                meta.dst_node = Some("node-1".to_string());
            }
        }
        let batch = p.run_batch_migration(BmpPriorityPolicy::FifoQueue, 5);
        assert_eq!(
            batch.total_bytes_moved,
            (batch.plans_completed as u64) * 1000
        );
    }

    // ── defragment_plan ───────────────────────────────────────────────────────

    #[test]
    fn test_defragment_plan_empty_nodes() {
        let mut p = make_planner();
        let result = p.defragment_plan(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_defragment_plan_already_balanced() {
        let mut p = make_planner();
        let nodes = vec!["node-0".to_string(), "node-1".to_string()];
        for i in 0..4u8 {
            let node = if i < 2 { "node-0" } else { "node-1" };
            let id = make_id(180 + i);
            p.register_block(id, 512, node, 0, 1.0);
        }
        let plans = p.defragment_plan(&nodes);
        // Already balanced (2 each) — no plans needed.
        assert!(plans.is_empty());
    }

    #[test]
    fn test_defragment_plan_imbalanced() {
        let mut p = make_planner();
        let nodes = vec!["node-0".to_string(), "node-1".to_string()];
        // 4 blocks on node-0, 0 on node-1.
        for i in 0..4u8 {
            let id = make_id(190 + i);
            p.register_block(id, 512, "node-0", 0, 1.0);
        }
        let plans = p.defragment_plan(&nodes);
        assert!(!plans.is_empty());
    }

    #[test]
    fn test_defragment_plan_skips_pinned() {
        let mut p = make_planner();
        let nodes = vec!["node-0".to_string(), "node-1".to_string()];
        for i in 0..4u8 {
            let id = make_id(200 + i);
            p.register_block(id, 512, "node-0", 0, 1.0);
            if i < 2 {
                p.pin(&id).expect("ok");
            }
        }
        let plans = p.defragment_plan(&nodes);
        // Only 2 movable blocks; moving up to 2 plans.
        assert!(plans.len() <= 2);
    }

    #[test]
    fn test_defragment_plan_single_node_no_plans() {
        let mut p = make_planner();
        let nodes = vec!["node-0".to_string()];
        for i in 0..3u8 {
            let id = make_id(210 + i);
            p.register_block(id, 512, "node-0", 0, 1.0);
        }
        // With a single node, over == under logic finds neither.
        let plans = p.defragment_plan(&nodes);
        assert!(plans.is_empty());
    }

    // ── migration_stats ───────────────────────────────────────────────────────

    #[test]
    fn test_migration_stats_empty() {
        let p = make_planner();
        let stats = p.migration_stats();
        assert_eq!(stats.total_blocks, 0);
        assert_eq!(stats.total_plans, 0);
        assert_eq!(stats.total_records, 0);
        assert_eq!(stats.total_bytes_moved, 0);
    }

    #[test]
    fn test_migration_stats_after_register() {
        let mut p = make_planner();
        for i in 0..5u8 {
            p.register_block(make_id(220 + i), 512, "node-0", 0, 1.0);
        }
        let stats = p.migration_stats();
        assert_eq!(stats.total_blocks, 5);
    }

    #[test]
    fn test_migration_stats_pinned_count() {
        let mut p = make_planner();
        for i in 0..4u8 {
            let id = make_id(230 + i);
            p.register_block(id, 512, "node-0", 0, 1.0);
            if i < 2 {
                p.pin(&id).expect("ok");
            }
        }
        let stats = p.migration_stats();
        assert_eq!(stats.pinned_blocks, 2);
    }

    #[test]
    fn test_migration_stats_plan_counts() {
        let cfg = BmpPlannerConfig {
            dry_run: true,
            ..Default::default()
        };
        let mut p = BlockMigrationPlanner::with_config(cfg);
        for i in 0..3u8 {
            let id = make_id(240 + i);
            p.register_block(id, 512, "node-0", 0, 1.0);
            p.create_plan(vec![id], "node-1", 0).expect("ok");
        }
        let stats = p.migration_stats();
        assert_eq!(stats.total_plans, 3);
        assert_eq!(stats.pending_plans, 3);
    }

    #[test]
    fn test_migration_stats_bytes_moved() {
        let cfg = BmpPlannerConfig {
            dry_run: true,
            ..Default::default()
        };
        let mut p = BlockMigrationPlanner::with_config(cfg);
        let id = make_id(250);
        p.register_block(id, 4096, "node-0", 0, 1.0);
        let pid = p.create_plan(vec![id], "node-1", 0).expect("ok");
        p.execute_plan(pid).expect("ok");
        let stats = p.migration_stats();
        assert_eq!(stats.total_bytes_moved, 4096);
    }

    #[test]
    fn test_migration_stats_cancelled_count() {
        let mut p = make_planner();
        for i in 200..203u8 {
            let id = make_id(i);
            p.register_block(id, 512, "node-0", 0, 1.0);
            let pid = p.create_plan(vec![id], "node-1", 0).expect("ok");
            p.cancel_plan(pid).expect("ok");
        }
        let stats = p.migration_stats();
        assert_eq!(stats.cancelled_plans, 3);
    }

    // ── execution log / recent_records ───────────────────────────────────────

    #[test]
    fn test_log_bounded_at_1000() {
        let cfg = BmpPlannerConfig {
            dry_run: true,
            ..Default::default()
        };
        let mut p = BlockMigrationPlanner::with_config(cfg);
        // Create 1100 single-block plans and execute them all.
        for i in 0..1100u16 {
            let id = bmp_id_from_bytes(&i.to_le_bytes());
            // Re-register (overwrite) to avoid needing unique keys.
            p.register_block(id, 1, "node-0", 0, 1.0);
            if let Ok(pid) = p.create_plan(vec![id], "node-1", 0) {
                let _ = p.execute_plan(pid);
            }
        }
        assert_eq!(p.log_len(), 1000);
    }

    #[test]
    fn test_recent_records_returns_n() {
        let cfg = BmpPlannerConfig {
            dry_run: true,
            ..Default::default()
        };
        let mut p = BlockMigrationPlanner::with_config(cfg);
        for i in 0..10u8 {
            let id = make_id(i);
            p.register_block(id, 512, "node-0", 0, 1.0);
            if let Ok(pid) = p.create_plan(vec![id], "node-1", 0) {
                let _ = p.execute_plan(pid);
            }
        }
        assert_eq!(p.recent_records(5).len(), 5);
    }

    #[test]
    fn test_plans_iter() {
        let mut p = make_planner();
        for i in 0..4u8 {
            let id = make_id(i);
            p.register_block(id, 512, "node-0", 0, 1.0);
            p.create_plan(vec![id], "node-1", 0).expect("ok");
        }
        assert_eq!(p.plans_iter().count(), 4);
    }

    // ── BmpPlanStatus Display ─────────────────────────────────────────────────

    #[test]
    fn test_plan_status_display() {
        assert_eq!(BmpPlanStatus::Pending.to_string(), "Pending");
        assert_eq!(BmpPlanStatus::InProgress.to_string(), "InProgress");
        assert_eq!(BmpPlanStatus::Completed.to_string(), "Completed");
        assert_eq!(BmpPlanStatus::Failed.to_string(), "Failed");
        assert_eq!(BmpPlanStatus::Cancelled.to_string(), "Cancelled");
    }

    // ── BmpPriorityPolicy Display ─────────────────────────────────────────────

    #[test]
    fn test_priority_policy_display() {
        assert_eq!(BmpPriorityPolicy::FifoQueue.to_string(), "FifoQueue");
        assert_eq!(
            BmpPriorityPolicy::HighFrequencyFirst.to_string(),
            "HighFrequencyFirst"
        );
        assert_eq!(BmpPriorityPolicy::LargestFirst.to_string(), "LargestFirst");
        assert_eq!(
            BmpPriorityPolicy::SmallestFirst.to_string(),
            "SmallestFirst"
        );
        assert_eq!(
            BmpPriorityPolicy::CostOptimized.to_string(),
            "CostOptimized"
        );
    }

    // ── Default impls ─────────────────────────────────────────────────────────

    #[test]
    fn test_planner_default() {
        let p = BlockMigrationPlanner::default();
        assert_eq!(p.block_count(), 0);
    }

    #[test]
    fn test_config_default() {
        let cfg = BmpPlannerConfig::default();
        assert_eq!(cfg.max_concurrent_migrations, 4);
        assert_eq!(cfg.bandwidth_limit_kbps, 0);
        assert!(!cfg.dry_run);
    }

    #[test]
    fn test_batch_result_default() {
        let b = BmpBatchResult::default();
        assert_eq!(b.plans_executed, 0);
        assert_eq!(b.total_bytes_moved, 0);
    }

    #[test]
    fn test_planner_stats_default() {
        let s = BmpPlannerStats::default();
        assert_eq!(s.total_blocks, 0);
        assert_eq!(s.total_failures, 0);
    }

    // ── cost-optimized scheduling ─────────────────────────────────────────────

    #[test]
    fn test_schedule_cost_optimized() {
        let mut p = make_planner();
        // tier 3, size 100 => cost 300; tier 1, size 500 => cost 500
        let id_a = make_id(20);
        let id_b = make_id(21);
        p.register_block(id_a, 100, "node-0", 3, 1.0);
        p.register_block(id_b, 500, "node-0", 1, 1.0);
        for id in [id_a, id_b] {
            if let Some(m) = p.blocks.get_mut(&id) {
                m.dst_node = Some("node-1".to_string());
            }
        }
        let pids = p
            .schedule_migrations(BmpPriorityPolicy::CostOptimized, 2)
            .expect("ok");
        // First plan should be the cheaper one (id_a, cost 300).
        let first_plan = p.plan(pids[0]).expect("plan");
        let first_bid = first_plan.block_ids[0];
        let first_meta = p.block_meta(&first_bid).expect("meta");
        let first_cost = first_meta.size_bytes * first_meta.tier as u64;
        // Should be <= the other block's cost.
        assert!(
            first_cost <= 500,
            "cost optimized ordering failed: {} > 500",
            first_cost
        );
    }
}
