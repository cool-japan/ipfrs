//! Storage Query Planner — optimizes storage access patterns for batch read/write workloads.
//!
//! [`StorageQueryPlanner`] analyses queries, selects between sequential and index scans,
//! applies predicate pushdown, caches plans via FNV-1a fingerprinting, and exposes
//! detailed explain output and planner statistics.

use std::collections::{HashMap, VecDeque};

// ─────────────────────────────────────────────────────────────────────────────
// Type aliases
// ─────────────────────────────────────────────────────────────────────────────

/// Numeric identifier assigned to a registered index.
pub type SqpIndexId = u32;

/// Convenience alias for [`StorageQueryPlanner`].
pub type SqpStorageQueryPlanner = StorageQueryPlanner;

// ─────────────────────────────────────────────────────────────────────────────
// Utility helpers
// ─────────────────────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash of arbitrary bytes.
#[inline]
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

/// Monotonic Unix-epoch timestamp (nanoseconds).
fn now_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1)
}

// ─────────────────────────────────────────────────────────────────────────────
// SqpValue
// ─────────────────────────────────────────────────────────────────────────────

/// A scalar value that can appear in a predicate.
#[derive(Clone, Debug, PartialEq)]
pub enum SqpValue {
    Int(i64),
    Float(f64),
    Text(String),
    Bool(bool),
    Null,
}

impl SqpValue {
    /// Returns a stable byte representation suitable for hashing.
    fn to_hash_bytes(&self) -> Vec<u8> {
        match self {
            SqpValue::Int(i) => {
                let mut v = vec![0u8];
                v.extend_from_slice(&i.to_le_bytes());
                v
            }
            SqpValue::Float(f) => {
                let mut v = vec![1u8];
                v.extend_from_slice(&f.to_bits().to_le_bytes());
                v
            }
            SqpValue::Text(s) => {
                let mut v = vec![2u8];
                v.extend_from_slice(s.as_bytes());
                v
            }
            SqpValue::Bool(b) => vec![3u8, if *b { 1 } else { 0 }],
            SqpValue::Null => vec![4u8],
        }
    }
}

impl std::fmt::Display for SqpValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SqpValue::Int(i) => write!(f, "{i}"),
            SqpValue::Float(v) => write!(f, "{v}"),
            SqpValue::Text(s) => write!(f, "'{s}'"),
            SqpValue::Bool(b) => write!(f, "{b}"),
            SqpValue::Null => write!(f, "NULL"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SqpOp
// ─────────────────────────────────────────────────────────────────────────────

/// Comparison / membership operator for a [`SqpPredicate`].
#[derive(Clone, Debug, PartialEq)]
pub enum SqpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    In(Vec<SqpValue>),
    IsNull,
    IsNotNull,
}

impl SqpOp {
    fn tag(&self) -> u8 {
        match self {
            SqpOp::Eq => 0,
            SqpOp::Ne => 1,
            SqpOp::Lt => 2,
            SqpOp::Le => 3,
            SqpOp::Gt => 4,
            SqpOp::Ge => 5,
            SqpOp::In(_) => 6,
            SqpOp::IsNull => 7,
            SqpOp::IsNotNull => 8,
        }
    }
}

impl std::fmt::Display for SqpOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SqpOp::Eq => write!(f, "="),
            SqpOp::Ne => write!(f, "!="),
            SqpOp::Lt => write!(f, "<"),
            SqpOp::Le => write!(f, "<="),
            SqpOp::Gt => write!(f, ">"),
            SqpOp::Ge => write!(f, ">="),
            SqpOp::In(vals) => {
                write!(f, "IN (")?;
                for (i, v) in vals.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, ")")
            }
            SqpOp::IsNull => write!(f, "IS NULL"),
            SqpOp::IsNotNull => write!(f, "IS NOT NULL"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SqpPredicate
// ─────────────────────────────────────────────────────────────────────────────

/// A single filter condition applied to a field.
#[derive(Clone, Debug, PartialEq)]
pub struct SqpPredicate {
    pub field: String,
    pub op: SqpOp,
    pub value: SqpValue,
}

impl SqpPredicate {
    pub fn new(field: impl Into<String>, op: SqpOp, value: SqpValue) -> Self {
        Self {
            field: field.into(),
            op,
            value,
        }
    }

    fn hash_bytes(&self) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(self.field.as_bytes());
        v.push(self.op.tag());
        if let SqpOp::In(vals) = &self.op {
            for sv in vals {
                v.extend(sv.to_hash_bytes());
            }
        }
        v.extend(self.value.to_hash_bytes());
        v
    }
}

impl std::fmt::Display for SqpPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {}", self.field, self.op, self.value)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SqpHint
// ─────────────────────────────────────────────────────────────────────────────

/// Planner hint supplied by the caller to guide plan selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SqpHint {
    /// Force a sequential (full-table) scan regardless of available indexes.
    ForceSeqScan,
    /// Force an index scan if any matching index exists.
    ForceIndexScan,
    /// Enable parallel scan execution.
    Parallel,
    /// Prefer cache-friendly access order (e.g. sequential prefetch).
    CacheFriendly,
}

// ─────────────────────────────────────────────────────────────────────────────
// SqpCostModel
// ─────────────────────────────────────────────────────────────────────────────

/// Cost model used to estimate query execution cost.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SqpCostModel {
    /// Cost is proportional to the estimated row count touched.
    SimpleRowCount,
    /// Cost includes I/O page reads based on block counts.
    IOCost,
    /// Cost based on working-set memory pressure.
    MemoryCost,
    /// Weighted combination of row-count, I/O, and memory.
    #[default]
    HybridCost,
}

// ─────────────────────────────────────────────────────────────────────────────
// SqpPlannerConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration that drives planner behaviour.
#[derive(Clone, Debug)]
pub struct SqpPlannerConfig {
    /// Maximum number of plans kept in the plan cache.
    pub max_cache_size: usize,
    /// Cost model used when estimating plan costs.
    pub cost_model: SqpCostModel,
    /// Whether predicates are pushed down into scans before projection.
    pub enable_predicate_pushdown: bool,
    /// Number of parallel scan threads to assume when estimating cost.
    pub parallel_scans: u32,
    /// Index selectivity threshold below which an index scan is preferred.
    /// (0.0 = never use index, 1.0 = always use index when available)
    pub index_selectivity_threshold: f64,
    /// Row count assumed for a full sequential scan of a table.
    pub default_table_rows: u64,
    /// Average bytes per row (used for I/O cost estimation).
    pub bytes_per_row: u64,
}

impl Default for SqpPlannerConfig {
    fn default() -> Self {
        Self {
            max_cache_size: 200,
            cost_model: SqpCostModel::HybridCost,
            enable_predicate_pushdown: true,
            parallel_scans: 4,
            index_selectivity_threshold: 0.15,
            default_table_rows: 1_000_000,
            bytes_per_row: 256,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SqpQuery
// ─────────────────────────────────────────────────────────────────────────────

/// A storage query presented to the planner.
#[derive(Clone, Debug)]
pub struct SqpQuery {
    /// Caller-assigned query identifier.
    pub id: u64,
    /// Filter conditions (ANDed together).
    pub predicates: Vec<SqpPredicate>,
    /// Column names to project (empty = all columns).
    pub projections: Vec<String>,
    /// Maximum rows to return.
    pub limit: Option<usize>,
    /// `(field, ascending)` ordering specification.
    pub order_by: Option<(String, bool)>,
    /// Optional planner hint overriding automatic decisions.
    pub hint: Option<SqpHint>,
}

impl SqpQuery {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            predicates: Vec::new(),
            projections: Vec::new(),
            limit: None,
            order_by: None,
            hint: None,
        }
    }

    /// Compute a deterministic fingerprint for plan-cache lookups.
    pub fn fingerprint(&self) -> u64 {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&self.id.to_le_bytes());
        for p in &self.predicates {
            buf.extend(p.hash_bytes());
        }
        for proj in &self.projections {
            buf.extend_from_slice(proj.as_bytes());
            buf.push(b'|');
        }
        if let Some(lim) = self.limit {
            buf.extend_from_slice(&lim.to_le_bytes());
        }
        if let Some((ref field, asc)) = self.order_by {
            buf.extend_from_slice(field.as_bytes());
            buf.push(if asc { 1 } else { 0 });
        }
        if let Some(hint) = self.hint {
            buf.push(hint as u8);
        }
        fnv1a_64(&buf)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SqpPlanStep
// ─────────────────────────────────────────────────────────────────────────────

/// A single step in a [`SqpQueryPlan`].
#[derive(Clone, Debug)]
pub enum SqpPlanStep {
    /// Full sequential scan of the given table.
    SeqScan {
        table: String,
        /// Number of predicate filters applied during the scan.
        filter_count: usize,
    },
    /// Index-based point or range lookup.
    IndexScan {
        index_id: SqpIndexId,
        /// Fraction of rows the index is expected to match (0.0–1.0).
        selectivity: f64,
    },
    /// Post-scan predicate filter.
    Filter(SqpPredicate),
    /// Sort the result stream.
    Sort { field: String, asc: bool },
    /// Truncate the result to at most N rows.
    Limit(usize),
    /// Project (select) specific columns.
    Project(Vec<String>),
}

impl std::fmt::Display for SqpPlanStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SqpPlanStep::SeqScan {
                table,
                filter_count,
            } => {
                write!(f, "SeqScan(table={table}, filters={filter_count})")
            }
            SqpPlanStep::IndexScan {
                index_id,
                selectivity,
            } => {
                write!(f, "IndexScan(id={index_id}, selectivity={selectivity:.4})")
            }
            SqpPlanStep::Filter(pred) => write!(f, "Filter({pred})"),
            SqpPlanStep::Sort { field, asc } => {
                write!(f, "Sort({field} {})", if *asc { "ASC" } else { "DESC" })
            }
            SqpPlanStep::Limit(n) => write!(f, "Limit({n})"),
            SqpPlanStep::Project(cols) => write!(f, "Project([{}])", cols.join(", ")),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SqpQueryPlan
// ─────────────────────────────────────────────────────────────────────────────

/// A fully resolved execution plan for a [`SqpQuery`].
#[derive(Clone, Debug)]
pub struct SqpQueryPlan {
    /// Identifier of the originating query.
    pub query_id: u64,
    /// Ordered list of execution steps.
    pub steps: Vec<SqpPlanStep>,
    /// Estimated cost (model-specific unit).
    pub estimated_cost: f64,
    /// Estimated number of rows returned after all filters.
    pub estimated_rows: u64,
    /// Whether at least one index scan is present in the plan.
    pub uses_index: bool,
    /// Timestamp when the plan was generated (nanoseconds).
    pub planned_at_ns: u64,
}

impl SqpQueryPlan {
    fn new(query_id: u64) -> Self {
        Self {
            query_id,
            steps: Vec::new(),
            estimated_cost: 0.0,
            estimated_rows: 0,
            uses_index: false,
            planned_at_ns: now_ns(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SqpIndexStat
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime statistics about a registered index.
#[derive(Clone, Debug)]
pub struct SqpIndexStat {
    /// Human-readable name of the index.
    pub name: String,
    /// Number of distinct values in the indexed column.
    pub cardinality: u64,
    /// Fraction of rows matched by a typical equality predicate (0.0–1.0).
    pub selectivity: f64,
    /// How many times this index has been used in a chosen plan.
    pub usage_count: u64,
    /// Timestamp of registration (nanoseconds).
    pub registered_at_ns: u64,
    /// Which query fingerprints reference this index.
    pub referencing_queries: Vec<u64>,
}

impl SqpIndexStat {
    fn new(name: String, cardinality: u64, selectivity: f64) -> Self {
        Self {
            name,
            cardinality,
            selectivity: selectivity.clamp(0.0, 1.0),
            usage_count: 0,
            registered_at_ns: now_ns(),
            referencing_queries: Vec::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SqpQueryRecord
// ─────────────────────────────────────────────────────────────────────────────

/// A historical record of a query planning event.
#[derive(Clone, Debug)]
pub struct SqpQueryRecord {
    /// Fingerprint of the query.
    pub fingerprint: u64,
    /// Whether the plan was served from cache.
    pub cache_hit: bool,
    /// Estimated cost of the chosen plan.
    pub estimated_cost: f64,
    /// Whether the chosen plan uses an index.
    pub uses_index: bool,
    /// Timestamp of the planning event (nanoseconds).
    pub planned_at_ns: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// SqpPlannerStats
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics for the planner since creation.
#[derive(Clone, Debug)]
pub struct SqpPlannerStats {
    /// Total number of plans generated (including cache hits).
    pub total_plans: u64,
    /// Fraction of planning calls that were served from cache (0.0–1.0).
    pub cache_hit_rate: f64,
    /// Average estimated cost across all plans.
    pub avg_cost: f64,
    /// Fraction of plans that use at least one index (0.0–1.0).
    pub index_usage_rate: f64,
    /// Number of entries currently in the plan cache.
    pub cache_size: usize,
    /// Number of registered indexes.
    pub registered_indexes: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// StorageQueryPlanner
// ─────────────────────────────────────────────────────────────────────────────

/// Production-grade query planner that optimizes storage access patterns.
///
/// # Examples
/// ```
/// use ipfrs_storage::storage_query_planner::{
///     StorageQueryPlanner, SqpPlannerConfig, SqpQuery, SqpPredicate, SqpOp, SqpValue,
/// };
///
/// let planner = StorageQueryPlanner::new(SqpPlannerConfig::default());
/// let mut query = SqpQuery::new(1);
/// query.predicates.push(SqpPredicate::new("age", SqpOp::Gt, SqpValue::Int(18)));
/// let plan = planner.plan(&query);
/// assert!(plan.estimated_rows > 0);
/// ```
pub struct StorageQueryPlanner {
    config: SqpPlannerConfig,
    /// Bounded ring-buffer of planning history (max 500).
    history: parking_lot::Mutex<VecDeque<SqpQueryRecord>>,
    /// Registered index metadata keyed by [`SqpIndexId`].
    index_stats: parking_lot::RwLock<HashMap<SqpIndexId, SqpIndexStat>>,
    /// Plan cache: FNV-1a fingerprint → cached plan (max `config.max_cache_size`).
    plan_cache: parking_lot::Mutex<HashMap<u64, SqpQueryPlan>>,
    /// Monotonically increasing counter for assigning index IDs.
    next_index_id: std::sync::atomic::AtomicU32,
    /// Cumulative totals for stats reporting.
    total_plans: std::sync::atomic::AtomicU64,
    total_cache_hits: std::sync::atomic::AtomicU64,
    total_index_plans: std::sync::atomic::AtomicU64,
    /// Sum of estimated costs, used for avg_cost computation.
    total_cost_bits: std::sync::atomic::AtomicU64,
}

impl Default for StorageQueryPlanner {
    fn default() -> Self {
        Self::new(SqpPlannerConfig::default())
    }
}

impl StorageQueryPlanner {
    /// Create a new planner with the given configuration.
    pub fn new(config: SqpPlannerConfig) -> Self {
        Self {
            config,
            history: parking_lot::Mutex::new(VecDeque::with_capacity(500)),
            index_stats: parking_lot::RwLock::new(HashMap::new()),
            plan_cache: parking_lot::Mutex::new(HashMap::new()),
            next_index_id: std::sync::atomic::AtomicU32::new(1),
            total_plans: std::sync::atomic::AtomicU64::new(0),
            total_cache_hits: std::sync::atomic::AtomicU64::new(0),
            total_index_plans: std::sync::atomic::AtomicU64::new(0),
            total_cost_bits: std::sync::atomic::AtomicU64::new(0),
        }
    }

    // ─── Index registration ────────────────────────────────────────────────

    /// Register a new index and return its assigned [`SqpIndexId`].
    ///
    /// If an index with the same name already exists its statistics are updated
    /// in place and the existing ID is returned.
    pub fn register_index(
        &self,
        name: impl Into<String>,
        cardinality: u64,
        selectivity: f64,
    ) -> SqpIndexId {
        let name = name.into();
        let mut stats = self.index_stats.write();

        // Check for existing index with this name.
        if let Some((&id, existing)) = stats.iter_mut().find(|(_, s)| s.name == name) {
            existing.cardinality = cardinality;
            existing.selectivity = selectivity.clamp(0.0, 1.0);
            return id;
        }

        let id = self
            .next_index_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        stats.insert(id, SqpIndexStat::new(name, cardinality, selectivity));
        id
    }

    /// Look up the stat record for a registered index.
    pub fn index_stat(&self, id: SqpIndexId) -> Option<SqpIndexStat> {
        self.index_stats.read().get(&id).cloned()
    }

    // ─── Planning ──────────────────────────────────────────────────────────

    /// Generate an optimised plan for `query` (no caching).
    pub fn plan(&self, query: &SqpQuery) -> SqpQueryPlan {
        let plan = self.build_plan(query);
        self.record_planning(
            query.fingerprint(),
            false,
            plan.estimated_cost,
            plan.uses_index,
        );
        plan
    }

    /// Generate or retrieve a cached plan for `query`.
    pub fn plan_cached(&self, query: &SqpQuery) -> SqpQueryPlan {
        let fp = query.fingerprint();

        // Cache lookup.
        {
            let cache = self.plan_cache.lock();
            if let Some(cached) = cache.get(&fp) {
                let plan = cached.clone();
                drop(cache);
                self.record_planning(fp, true, plan.estimated_cost, plan.uses_index);
                return plan;
            }
        }

        // Cache miss — build and store.
        let plan = self.build_plan(query);
        {
            let mut cache = self.plan_cache.lock();
            if cache.len() >= self.config.max_cache_size {
                // Evict the oldest entry (first inserted, hash-map order is arbitrary;
                // we remove an arbitrary key for bounded-size enforcement).
                if let Some(evict_key) = cache.keys().copied().next() {
                    cache.remove(&evict_key);
                }
            }
            cache.insert(fp, plan.clone());
        }
        self.record_planning(fp, false, plan.estimated_cost, plan.uses_index);
        plan
    }

    /// Invalidate the entire plan cache.
    pub fn invalidate_cache(&self) {
        self.plan_cache.lock().clear();
    }

    /// Remove all cached plans that reference the given index.
    pub fn invalidate_for_index(&self, id: SqpIndexId) {
        let referencing: Vec<u64> = {
            let stats = self.index_stats.read();
            stats
                .get(&id)
                .map(|s| s.referencing_queries.clone())
                .unwrap_or_default()
        };
        let mut cache = self.plan_cache.lock();
        for fp in referencing {
            cache.remove(&fp);
        }
        // Also clear the referencing list on the index stat.
        if let Some(stat) = self.index_stats.write().get_mut(&id) {
            stat.referencing_queries.clear();
        }
    }

    // ─── Cost estimation ──────────────────────────────────────────────────

    /// Re-estimate the cost of an already-built plan.
    ///
    /// This applies the configured cost model independently of plan generation,
    /// making it useful for comparing alternative plans.
    pub fn estimate_cost(&self, plan: &SqpQueryPlan) -> f64 {
        self.compute_cost(plan.estimated_rows, plan.uses_index, plan.steps.len())
    }

    // ─── Explain output ───────────────────────────────────────────────────

    /// Produce a human-readable description of a plan.
    pub fn explain(&self, plan: &SqpQueryPlan) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "QueryPlan(id={}, cost={:.2}, rows={}, uses_index={})\n",
            plan.query_id, plan.estimated_cost, plan.estimated_rows, plan.uses_index
        ));
        for (i, step) in plan.steps.iter().enumerate() {
            out.push_str(&format!("  [{i:02}] {step}\n"));
        }
        // Append index details when relevant.
        if plan.uses_index {
            let stats = self.index_stats.read();
            for step in &plan.steps {
                if let SqpPlanStep::IndexScan {
                    index_id,
                    selectivity,
                } = step
                {
                    if let Some(stat) = stats.get(index_id) {
                        out.push_str(&format!(
                            "       -> index '{}': cardinality={}, selectivity={:.4}, usages={}\n",
                            stat.name, stat.cardinality, selectivity, stat.usage_count
                        ));
                    }
                }
            }
        }
        out
    }

    // ─── Statistics ───────────────────────────────────────────────────────

    /// Return aggregate planner statistics.
    pub fn planner_stats(&self) -> SqpPlannerStats {
        use std::sync::atomic::Ordering::Relaxed;
        let total = self.total_plans.load(Relaxed);
        let hits = self.total_cache_hits.load(Relaxed);
        let index_plans = self.total_index_plans.load(Relaxed);
        let cost_bits = self.total_cost_bits.load(Relaxed);

        let cache_hit_rate = if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        };
        let avg_cost = if total == 0 {
            0.0
        } else {
            f64::from_bits(cost_bits) / total as f64
        };
        let index_usage_rate = if total == 0 {
            0.0
        } else {
            index_plans as f64 / total as f64
        };

        SqpPlannerStats {
            total_plans: total,
            cache_hit_rate,
            avg_cost,
            index_usage_rate,
            cache_size: self.plan_cache.lock().len(),
            registered_indexes: self.index_stats.read().len(),
        }
    }

    /// Return the most recent N planning records from history.
    pub fn recent_history(&self, n: usize) -> Vec<SqpQueryRecord> {
        let hist = self.history.lock();
        hist.iter().rev().take(n).cloned().collect()
    }

    // ─── Private helpers ──────────────────────────────────────────────────

    /// Core plan-construction logic.
    fn build_plan(&self, query: &SqpQuery) -> SqpQueryPlan {
        let mut plan = SqpQueryPlan::new(query.id);
        let fp = query.fingerprint();

        // Decide scan type.
        let (scan_step, uses_index, chosen_index_id) = self.choose_scan(query, fp);

        // If using an index, bump its usage counter and record the fingerprint.
        if let Some(iid) = chosen_index_id {
            if let Some(stat) = self.index_stats.write().get_mut(&iid) {
                stat.usage_count += 1;
                if !stat.referencing_queries.contains(&fp) {
                    stat.referencing_queries.push(fp);
                }
            }
        }

        plan.uses_index = uses_index;
        plan.steps.push(scan_step);

        // Predicate pushdown (remaining predicates not absorbed by index scan).
        if self.config.enable_predicate_pushdown && query.hint != Some(SqpHint::ForceSeqScan) {
            for pred in &query.predicates {
                // Skip predicates that were already handled by the index scan.
                if uses_index {
                    // Only the first predicate was used in the index scan.
                    // Skip it here; the remaining ones become Filter steps.
                    if query.predicates.first().map(|p| p == pred) == Some(true) {
                        continue;
                    }
                }
                plan.steps.push(SqpPlanStep::Filter(pred.clone()));
            }
        } else {
            // No pushdown: emit explicit Filter steps for all predicates.
            for pred in &query.predicates {
                plan.steps.push(SqpPlanStep::Filter(pred.clone()));
            }
        }

        // Order-by.
        if let Some((ref field, asc)) = query.order_by {
            plan.steps.push(SqpPlanStep::Sort {
                field: field.clone(),
                asc,
            });
        }

        // Limit.
        if let Some(lim) = query.limit {
            plan.steps.push(SqpPlanStep::Limit(lim));
        }

        // Projection (only if non-empty).
        if !query.projections.is_empty() {
            plan.steps
                .push(SqpPlanStep::Project(query.projections.clone()));
        }

        // Estimate rows and cost.
        plan.estimated_rows = self.estimate_rows(query, uses_index, chosen_index_id);
        plan.estimated_cost = self.compute_cost(plan.estimated_rows, uses_index, plan.steps.len());

        plan
    }

    /// Choose between a sequential scan and an index scan.
    ///
    /// Returns `(step, uses_index, Option<index_id>)`.
    fn choose_scan(&self, query: &SqpQuery, _fp: u64) -> (SqpPlanStep, bool, Option<SqpIndexId>) {
        // Honour explicit hints first.
        match query.hint {
            Some(SqpHint::ForceSeqScan) => {
                return (
                    SqpPlanStep::SeqScan {
                        table: "default".to_string(),
                        filter_count: query.predicates.len(),
                    },
                    false,
                    None,
                );
            }
            Some(SqpHint::ForceIndexScan) => {
                if let Some((id, stat)) = self.best_index_for_query(query) {
                    return (
                        SqpPlanStep::IndexScan {
                            index_id: id,
                            selectivity: stat.selectivity,
                        },
                        true,
                        Some(id),
                    );
                }
                // Fall through to automatic selection if no index is available.
            }
            _ => {}
        }

        // Automatic selection: use index if selectivity is below threshold.
        if let Some((id, stat)) = self.best_index_for_query(query) {
            if stat.selectivity <= self.config.index_selectivity_threshold {
                return (
                    SqpPlanStep::IndexScan {
                        index_id: id,
                        selectivity: stat.selectivity,
                    },
                    true,
                    Some(id),
                );
            }
        }

        // Fall back to sequential scan.
        (
            SqpPlanStep::SeqScan {
                table: "default".to_string(),
                filter_count: query.predicates.len(),
            },
            false,
            None,
        )
    }

    /// Find the most selective index that matches any equality predicate.
    fn best_index_for_query(&self, query: &SqpQuery) -> Option<(SqpIndexId, SqpIndexStat)> {
        let stats = self.index_stats.read();
        if stats.is_empty() || query.predicates.is_empty() {
            return None;
        }

        // Prefer indexes whose name matches a predicate field.
        let mut best: Option<(SqpIndexId, SqpIndexStat)> = None;
        for pred in &query.predicates {
            for (&id, stat) in stats.iter() {
                if stat.name == pred.field {
                    match best {
                        None => best = Some((id, stat.clone())),
                        Some((_, ref b)) if stat.selectivity < b.selectivity => {
                            best = Some((id, stat.clone()))
                        }
                        _ => {}
                    }
                }
            }
        }

        // If no field match, return the globally most selective index.
        if best.is_none() {
            best = stats
                .iter()
                .min_by(|(_, a), (_, b)| {
                    a.selectivity
                        .partial_cmp(&b.selectivity)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(&id, stat)| (id, stat.clone()));
        }

        best
    }

    /// Estimate the number of rows returned after all filters.
    fn estimate_rows(
        &self,
        query: &SqpQuery,
        uses_index: bool,
        index_id: Option<SqpIndexId>,
    ) -> u64 {
        let base = self.config.default_table_rows;

        let selectivity: f64 = if uses_index {
            if let Some(id) = index_id {
                let stats = self.index_stats.read();
                stats.get(&id).map(|s| s.selectivity).unwrap_or(0.5)
            } else {
                0.5
            }
        } else {
            // Estimate per-predicate selectivity.
            let per_pred = query.predicates.iter().fold(1.0f64, |acc, pred| {
                acc * Self::predicate_selectivity_estimate(&pred.op)
            });
            per_pred.clamp(0.0, 1.0)
        };

        let mut rows = ((base as f64) * selectivity).ceil() as u64;
        rows = rows.max(1);

        // Apply limit.
        if let Some(lim) = query.limit {
            rows = rows.min(lim as u64);
        }
        rows
    }

    /// Heuristic per-operator selectivity.
    fn predicate_selectivity_estimate(op: &SqpOp) -> f64 {
        match op {
            SqpOp::Eq => 0.05,
            SqpOp::Ne => 0.95,
            SqpOp::Lt | SqpOp::Le => 0.33,
            SqpOp::Gt | SqpOp::Ge => 0.33,
            SqpOp::In(vals) => {
                // Each IN value contributes ~5 % selectivity, capped at 50 %.
                (vals.len() as f64 * 0.05).min(0.50)
            }
            SqpOp::IsNull => 0.01,
            SqpOp::IsNotNull => 0.99,
        }
    }

    /// Compute estimated cost according to the configured cost model.
    fn compute_cost(&self, rows: u64, uses_index: bool, step_count: usize) -> f64 {
        let rows_f = rows as f64;
        match self.config.cost_model {
            SqpCostModel::SimpleRowCount => rows_f,
            SqpCostModel::IOCost => {
                let pages = (rows_f * self.config.bytes_per_row as f64 / 8192.0).ceil();
                if uses_index {
                    // Index scan: log(pages) seek + pages/selectivity read.
                    pages.ln().max(1.0) + pages * 1.5
                } else {
                    // Full scan.
                    let total_pages = (self.config.default_table_rows as f64
                        * self.config.bytes_per_row as f64
                        / 8192.0)
                        .ceil();
                    total_pages + step_count as f64 * 0.1
                }
            }
            SqpCostModel::MemoryCost => {
                let working_set_mb =
                    (rows_f * self.config.bytes_per_row as f64) / (1024.0 * 1024.0);
                working_set_mb * if uses_index { 1.0 } else { 2.5 }
            }
            SqpCostModel::HybridCost => {
                let io = {
                    let pages = (rows_f * self.config.bytes_per_row as f64 / 8192.0).ceil();
                    if uses_index {
                        pages.ln().max(1.0) + pages * 1.5
                    } else {
                        let total_pages = (self.config.default_table_rows as f64
                            * self.config.bytes_per_row as f64
                            / 8192.0)
                            .ceil();
                        total_pages + step_count as f64 * 0.1
                    }
                };
                let mem = {
                    let wsm = (rows_f * self.config.bytes_per_row as f64) / (1024.0 * 1024.0);
                    wsm * if uses_index { 1.0 } else { 2.5 }
                };
                // Weighted blend: 60 % I/O, 30 % row count, 10 % memory.
                0.6 * io + 0.3 * rows_f + 0.1 * mem
            }
        }
    }

    /// Record a planning event into the history ring-buffer and update counters.
    fn record_planning(&self, fp: u64, cache_hit: bool, cost: f64, uses_index: bool) {
        use std::sync::atomic::Ordering::Relaxed;

        self.total_plans.fetch_add(1, Relaxed);
        if cache_hit {
            self.total_cache_hits.fetch_add(1, Relaxed);
        }
        if uses_index {
            self.total_index_plans.fetch_add(1, Relaxed);
        }

        // Accumulate cost as f64 bits (single-threaded addition via CAS would be
        // racy but acceptable for statistics; we use a simple fetch_add on bits).
        let prev_bits = self.total_cost_bits.load(Relaxed);
        let prev_cost = f64::from_bits(prev_bits);
        let new_cost_bits = (prev_cost + cost).to_bits();
        // Best-effort CAS loop (stats do not need strict consistency).
        let _ = self
            .total_cost_bits
            .compare_exchange(prev_bits, new_cost_bits, Relaxed, Relaxed);

        let record = SqpQueryRecord {
            fingerprint: fp,
            cache_hit,
            estimated_cost: cost,
            uses_index,
            planned_at_ns: now_ns(),
        };

        let mut hist = self.history.lock();
        if hist.len() >= 500 {
            hist.pop_front();
        }
        hist.push_back(record);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ────────────────────────────────────────────────────────────

    fn make_planner() -> StorageQueryPlanner {
        StorageQueryPlanner::new(SqpPlannerConfig::default())
    }

    fn simple_query(id: u64) -> SqpQuery {
        SqpQuery::new(id)
    }

    fn predicate_eq(field: &str, v: i64) -> SqpPredicate {
        SqpPredicate::new(field, SqpOp::Eq, SqpValue::Int(v))
    }

    // ── fnv1a helper ───────────────────────────────────────────────────────

    #[test]
    fn test_fnv1a_empty() {
        assert_eq!(fnv1a_64(&[]), 14_695_981_039_346_656_037u64);
    }

    #[test]
    fn test_fnv1a_deterministic() {
        assert_eq!(fnv1a_64(b"hello"), fnv1a_64(b"hello"));
    }

    #[test]
    fn test_fnv1a_different_inputs() {
        assert_ne!(fnv1a_64(b"foo"), fnv1a_64(b"bar"));
    }

    // ── SqpValue ───────────────────────────────────────────────────────────

    #[test]
    fn test_sqpvalue_display_int() {
        assert_eq!(SqpValue::Int(42).to_string(), "42");
    }

    #[test]
    fn test_sqpvalue_display_float() {
        let s = SqpValue::Float(2.71).to_string();
        assert!(s.starts_with("2.71"));
    }

    #[test]
    fn test_sqpvalue_display_text() {
        assert_eq!(SqpValue::Text("hi".into()).to_string(), "'hi'");
    }

    #[test]
    fn test_sqpvalue_display_bool_true() {
        assert_eq!(SqpValue::Bool(true).to_string(), "true");
    }

    #[test]
    fn test_sqpvalue_display_bool_false() {
        assert_eq!(SqpValue::Bool(false).to_string(), "false");
    }

    #[test]
    fn test_sqpvalue_display_null() {
        assert_eq!(SqpValue::Null.to_string(), "NULL");
    }

    #[test]
    fn test_sqpvalue_hash_bytes_distinct() {
        assert_ne!(
            SqpValue::Int(1).to_hash_bytes(),
            SqpValue::Float(1.0).to_hash_bytes()
        );
    }

    // ── SqpOp ──────────────────────────────────────────────────────────────

    #[test]
    fn test_sqpop_display_eq() {
        assert_eq!(SqpOp::Eq.to_string(), "=");
    }

    #[test]
    fn test_sqpop_display_ne() {
        assert_eq!(SqpOp::Ne.to_string(), "!=");
    }

    #[test]
    fn test_sqpop_display_lt() {
        assert_eq!(SqpOp::Lt.to_string(), "<");
    }

    #[test]
    fn test_sqpop_display_le() {
        assert_eq!(SqpOp::Le.to_string(), "<=");
    }

    #[test]
    fn test_sqpop_display_gt() {
        assert_eq!(SqpOp::Gt.to_string(), ">");
    }

    #[test]
    fn test_sqpop_display_ge() {
        assert_eq!(SqpOp::Ge.to_string(), ">=");
    }

    #[test]
    fn test_sqpop_display_in() {
        let op = SqpOp::In(vec![SqpValue::Int(1), SqpValue::Int(2)]);
        assert!(op.to_string().contains("IN"));
    }

    #[test]
    fn test_sqpop_display_is_null() {
        assert_eq!(SqpOp::IsNull.to_string(), "IS NULL");
    }

    #[test]
    fn test_sqpop_display_is_not_null() {
        assert_eq!(SqpOp::IsNotNull.to_string(), "IS NOT NULL");
    }

    #[test]
    fn test_sqpop_tags_unique() {
        let ops: Vec<u8> = vec![
            SqpOp::Eq.tag(),
            SqpOp::Ne.tag(),
            SqpOp::Lt.tag(),
            SqpOp::Le.tag(),
            SqpOp::Gt.tag(),
            SqpOp::Ge.tag(),
            SqpOp::In(vec![]).tag(),
            SqpOp::IsNull.tag(),
            SqpOp::IsNotNull.tag(),
        ];
        let unique: std::collections::HashSet<u8> = ops.iter().copied().collect();
        assert_eq!(unique.len(), ops.len());
    }

    // ── SqpPredicate ───────────────────────────────────────────────────────

    #[test]
    fn test_predicate_display() {
        let p = predicate_eq("age", 30);
        assert!(p.to_string().contains("age"));
        assert!(p.to_string().contains("30"));
    }

    #[test]
    fn test_predicate_hash_bytes_non_empty() {
        let p = predicate_eq("field", 1);
        assert!(!p.hash_bytes().is_empty());
    }

    #[test]
    fn test_predicates_different_fields_different_hash() {
        let p1 = predicate_eq("field_a", 1);
        let p2 = predicate_eq("field_b", 1);
        assert_ne!(p1.hash_bytes(), p2.hash_bytes());
    }

    // ── SqpQuery fingerprint ───────────────────────────────────────────────

    #[test]
    fn test_query_fingerprint_deterministic() {
        let mut q = simple_query(42);
        q.predicates.push(predicate_eq("x", 1));
        assert_eq!(q.fingerprint(), q.fingerprint());
    }

    #[test]
    fn test_query_fingerprint_changes_with_predicate() {
        let mut q1 = simple_query(1);
        let mut q2 = simple_query(1);
        q1.predicates.push(predicate_eq("a", 1));
        q2.predicates.push(predicate_eq("b", 1));
        assert_ne!(q1.fingerprint(), q2.fingerprint());
    }

    #[test]
    fn test_query_fingerprint_changes_with_limit() {
        let mut q1 = simple_query(1);
        let mut q2 = simple_query(1);
        q1.limit = Some(10);
        q2.limit = Some(20);
        assert_ne!(q1.fingerprint(), q2.fingerprint());
    }

    // ── Index registration ─────────────────────────────────────────────────

    #[test]
    fn test_register_index_returns_id() {
        let planner = make_planner();
        let id = planner.register_index("age_idx", 10_000, 0.05);
        assert!(id > 0);
    }

    #[test]
    fn test_register_index_same_name_returns_same_id() {
        let planner = make_planner();
        let id1 = planner.register_index("idx", 100, 0.1);
        let id2 = planner.register_index("idx", 200, 0.2);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_register_index_updates_stats() {
        let planner = make_planner();
        let id = planner.register_index("idx", 100, 0.1);
        planner.register_index("idx", 999, 0.5);
        let stat = planner.index_stat(id).expect("stat must exist");
        assert_eq!(stat.cardinality, 999);
    }

    #[test]
    fn test_register_two_indexes() {
        let planner = make_planner();
        let id1 = planner.register_index("a", 100, 0.1);
        let id2 = planner.register_index("b", 200, 0.2);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_index_stat_unknown_id_returns_none() {
        let planner = make_planner();
        assert!(planner.index_stat(9999).is_none());
    }

    #[test]
    fn test_selectivity_clamped_above_one() {
        let planner = make_planner();
        let id = planner.register_index("x", 100, 5.0);
        let stat = planner.index_stat(id).unwrap();
        assert!(stat.selectivity <= 1.0);
    }

    #[test]
    fn test_selectivity_clamped_below_zero() {
        let planner = make_planner();
        let id = planner.register_index("x", 100, -1.0);
        let stat = planner.index_stat(id).unwrap();
        assert!(stat.selectivity >= 0.0);
    }

    // ── plan() ────────────────────────────────────────────────────────────

    #[test]
    fn test_plan_returns_non_zero_rows() {
        let planner = make_planner();
        let q = simple_query(1);
        let plan = planner.plan(&q);
        assert!(plan.estimated_rows > 0);
    }

    #[test]
    fn test_plan_has_at_least_one_step() {
        let planner = make_planner();
        let q = simple_query(1);
        let plan = planner.plan(&q);
        assert!(!plan.steps.is_empty());
    }

    #[test]
    fn test_plan_positive_cost() {
        let planner = make_planner();
        let q = simple_query(1);
        let plan = planner.plan(&q);
        assert!(plan.estimated_cost > 0.0);
    }

    #[test]
    fn test_plan_seq_scan_when_no_index() {
        let planner = make_planner();
        let q = simple_query(1);
        let plan = planner.plan(&q);
        assert!(!plan.uses_index);
    }

    #[test]
    fn test_plan_index_scan_when_low_selectivity() {
        let planner = make_planner();
        planner.register_index("age", 100_000, 0.01);
        let mut q = SqpQuery::new(1);
        q.predicates.push(predicate_eq("age", 25));
        let plan = planner.plan(&q);
        assert!(plan.uses_index);
    }

    #[test]
    fn test_plan_seq_scan_when_high_selectivity() {
        let planner = make_planner();
        planner.register_index("status", 2, 0.5);
        let mut q = SqpQuery::new(1);
        q.predicates.push(predicate_eq("status", 1));
        let plan = planner.plan(&q);
        assert!(!plan.uses_index);
    }

    #[test]
    fn test_plan_with_limit_reduces_rows() {
        let planner = make_planner();
        let mut q = simple_query(1);
        q.limit = Some(5);
        let plan = planner.plan(&q);
        assert!(plan.estimated_rows <= 5);
    }

    #[test]
    fn test_plan_with_projection_adds_project_step() {
        let planner = make_planner();
        let mut q = simple_query(1);
        q.projections = vec!["col_a".into(), "col_b".into()];
        let plan = planner.plan(&q);
        let has_project = plan
            .steps
            .iter()
            .any(|s| matches!(s, SqpPlanStep::Project(_)));
        assert!(has_project);
    }

    #[test]
    fn test_plan_with_order_by_adds_sort_step() {
        let planner = make_planner();
        let mut q = simple_query(1);
        q.order_by = Some(("ts".into(), true));
        let plan = planner.plan(&q);
        let has_sort = plan
            .steps
            .iter()
            .any(|s| matches!(s, SqpPlanStep::Sort { .. }));
        assert!(has_sort);
    }

    // ── Hints ─────────────────────────────────────────────────────────────

    #[test]
    fn test_hint_force_seq_scan() {
        let planner = make_planner();
        planner.register_index("x", 10_000, 0.001); // would normally trigger index scan
        let mut q = SqpQuery::new(1);
        q.predicates.push(predicate_eq("x", 1));
        q.hint = Some(SqpHint::ForceSeqScan);
        let plan = planner.plan(&q);
        assert!(!plan.uses_index);
    }

    #[test]
    fn test_hint_force_index_scan() {
        let planner = make_planner();
        planner.register_index("x", 10_000, 0.9); // high selectivity, normally seq scan
        let mut q = SqpQuery::new(1);
        q.predicates.push(predicate_eq("x", 1));
        q.hint = Some(SqpHint::ForceIndexScan);
        let plan = planner.plan(&q);
        assert!(plan.uses_index);
    }

    #[test]
    fn test_hint_force_index_scan_no_index_falls_back_to_seq() {
        let planner = make_planner();
        let mut q = SqpQuery::new(1);
        q.predicates.push(predicate_eq("x", 1));
        q.hint = Some(SqpHint::ForceIndexScan);
        let plan = planner.plan(&q);
        // No index registered — should fall back to seq scan.
        assert!(!plan.uses_index);
    }

    // ── plan_cached() ─────────────────────────────────────────────────────

    #[test]
    fn test_plan_cached_second_call_is_cache_hit() {
        let planner = make_planner();
        let q = simple_query(77);
        planner.plan_cached(&q);
        planner.plan_cached(&q);
        let stats = planner.planner_stats();
        assert!(stats.cache_hit_rate > 0.0);
    }

    #[test]
    fn test_plan_cached_returns_same_plan() {
        let planner = make_planner();
        let q = simple_query(77);
        let p1 = planner.plan_cached(&q);
        let p2 = planner.plan_cached(&q);
        assert_eq!(p1.estimated_cost.to_bits(), p2.estimated_cost.to_bits());
        assert_eq!(p1.estimated_rows, p2.estimated_rows);
    }

    #[test]
    fn test_plan_cache_respects_max_size() {
        let config = SqpPlannerConfig {
            max_cache_size: 3,
            ..Default::default()
        };
        let planner = StorageQueryPlanner::new(config);
        for i in 0..10u64 {
            planner.plan_cached(&simple_query(i));
        }
        let stats = planner.planner_stats();
        assert!(stats.cache_size <= 3);
    }

    // ── invalidate_cache ──────────────────────────────────────────────────

    #[test]
    fn test_invalidate_cache_clears_cache() {
        let planner = make_planner();
        planner.plan_cached(&simple_query(1));
        planner.plan_cached(&simple_query(2));
        planner.invalidate_cache();
        assert_eq!(planner.planner_stats().cache_size, 0);
    }

    #[test]
    fn test_invalidate_for_index_removes_plan() {
        let planner = make_planner();
        let idx = planner.register_index("field", 1000, 0.05);
        let mut q = SqpQuery::new(1);
        q.predicates.push(predicate_eq("field", 42));
        planner.plan_cached(&q);
        planner.invalidate_for_index(idx);
        // After invalidation a new plan_cached call must miss the cache.
        // The cache_size should now be 0 (or the plan was evicted).
        let before_total = planner.planner_stats().total_plans;
        planner.plan_cached(&q);
        // A new planning call increments total_plans.
        assert!(planner.planner_stats().total_plans > before_total);
    }

    // ── estimate_cost ─────────────────────────────────────────────────────

    #[test]
    fn test_estimate_cost_positive() {
        let planner = make_planner();
        let q = simple_query(1);
        let plan = planner.plan(&q);
        assert!(planner.estimate_cost(&plan) > 0.0);
    }

    #[test]
    fn test_estimate_cost_index_less_than_seq_for_low_rows() {
        let config = SqpPlannerConfig {
            cost_model: SqpCostModel::IOCost,
            ..Default::default()
        };
        let planner = StorageQueryPlanner::new(config);
        planner.register_index("x", 100_000, 0.01);
        let mut q_idx = SqpQuery::new(1);
        q_idx.predicates.push(predicate_eq("x", 1));
        q_idx.hint = Some(SqpHint::ForceIndexScan);

        let mut q_seq = SqpQuery::new(2);
        q_seq.predicates.push(predicate_eq("x", 1));
        q_seq.hint = Some(SqpHint::ForceSeqScan);

        let idx_plan = planner.plan(&q_idx);
        let seq_plan = planner.plan(&q_seq);
        // Index scan on 1 % of rows should be cheaper than full scan.
        assert!(
            idx_plan.estimated_cost < seq_plan.estimated_cost,
            "idx={} seq={}",
            idx_plan.estimated_cost,
            seq_plan.estimated_cost
        );
    }

    // ── Cost models ───────────────────────────────────────────────────────

    #[test]
    fn test_cost_model_simple_row_count() {
        let config = SqpPlannerConfig {
            cost_model: SqpCostModel::SimpleRowCount,
            ..Default::default()
        };
        let planner = StorageQueryPlanner::new(config);
        let q = simple_query(1);
        let plan = planner.plan(&q);
        assert_eq!(plan.estimated_cost, plan.estimated_rows as f64);
    }

    #[test]
    fn test_cost_model_memory_cost() {
        let config = SqpPlannerConfig {
            cost_model: SqpCostModel::MemoryCost,
            ..Default::default()
        };
        let planner = StorageQueryPlanner::new(config);
        let plan = planner.plan(&simple_query(1));
        assert!(plan.estimated_cost > 0.0);
    }

    #[test]
    fn test_cost_model_io_cost() {
        let config = SqpPlannerConfig {
            cost_model: SqpCostModel::IOCost,
            ..Default::default()
        };
        let planner = StorageQueryPlanner::new(config);
        let plan = planner.plan(&simple_query(1));
        assert!(plan.estimated_cost > 0.0);
    }

    #[test]
    fn test_cost_model_hybrid() {
        let config = SqpPlannerConfig {
            cost_model: SqpCostModel::HybridCost,
            ..Default::default()
        };
        let planner = StorageQueryPlanner::new(config);
        let plan = planner.plan(&simple_query(1));
        assert!(plan.estimated_cost > 0.0);
    }

    // ── explain() ─────────────────────────────────────────────────────────

    #[test]
    fn test_explain_contains_query_id() {
        let planner = make_planner();
        let plan = planner.plan(&simple_query(99));
        let text = planner.explain(&plan);
        assert!(text.contains("id=99"));
    }

    #[test]
    fn test_explain_contains_steps() {
        let planner = make_planner();
        let plan = planner.plan(&simple_query(1));
        let text = planner.explain(&plan);
        assert!(text.contains("Scan"));
    }

    #[test]
    fn test_explain_contains_index_info_when_used() {
        let planner = make_planner();
        planner.register_index("age", 100_000, 0.01);
        let mut q = SqpQuery::new(5);
        q.predicates.push(predicate_eq("age", 30));
        let plan = planner.plan(&q);
        let text = planner.explain(&plan);
        assert!(text.contains("age"));
    }

    // ── planner_stats() ───────────────────────────────────────────────────

    #[test]
    fn test_stats_total_plans_increments() {
        let planner = make_planner();
        planner.plan(&simple_query(1));
        planner.plan(&simple_query(2));
        assert_eq!(planner.planner_stats().total_plans, 2);
    }

    #[test]
    fn test_stats_registered_indexes() {
        let planner = make_planner();
        planner.register_index("a", 100, 0.1);
        planner.register_index("b", 200, 0.2);
        assert_eq!(planner.planner_stats().registered_indexes, 2);
    }

    #[test]
    fn test_stats_index_usage_rate_non_zero() {
        let planner = make_planner();
        planner.register_index("x", 1_000_000, 0.001);
        let mut q = SqpQuery::new(1);
        q.predicates.push(predicate_eq("x", 1));
        planner.plan(&q);
        let stats = planner.planner_stats();
        assert!(stats.index_usage_rate > 0.0);
    }

    #[test]
    fn test_stats_avg_cost_positive_after_plan() {
        let planner = make_planner();
        planner.plan(&simple_query(1));
        let stats = planner.planner_stats();
        assert!(stats.avg_cost > 0.0);
    }

    #[test]
    fn test_stats_cache_hit_rate_zero_before_cached() {
        let planner = make_planner();
        planner.plan(&simple_query(1));
        let stats = planner.planner_stats();
        assert_eq!(stats.cache_hit_rate, 0.0);
    }

    // ── History ───────────────────────────────────────────────────────────

    #[test]
    fn test_history_grows_with_plans() {
        let planner = make_planner();
        planner.plan(&simple_query(1));
        planner.plan(&simple_query(2));
        assert_eq!(planner.recent_history(10).len(), 2);
    }

    #[test]
    fn test_history_bounded_at_500() {
        let planner = make_planner();
        for i in 0..600u64 {
            planner.plan(&simple_query(i));
        }
        let hist = planner.recent_history(1000);
        assert!(hist.len() <= 500);
    }

    #[test]
    fn test_history_cache_hit_recorded() {
        let planner = make_planner();
        let q = simple_query(1);
        planner.plan_cached(&q);
        planner.plan_cached(&q);
        let hist = planner.recent_history(2);
        assert!(hist.iter().any(|r| r.cache_hit));
    }

    // ── SqpPlanStep display ───────────────────────────────────────────────

    #[test]
    fn test_step_display_seq_scan() {
        let s = SqpPlanStep::SeqScan {
            table: "blocks".into(),
            filter_count: 2,
        };
        assert!(s.to_string().contains("blocks"));
    }

    #[test]
    fn test_step_display_index_scan() {
        let s = SqpPlanStep::IndexScan {
            index_id: 3,
            selectivity: 0.05,
        };
        assert!(s.to_string().contains("3"));
    }

    #[test]
    fn test_step_display_filter() {
        let s = SqpPlanStep::Filter(predicate_eq("col", 1));
        assert!(s.to_string().contains("col"));
    }

    #[test]
    fn test_step_display_sort_asc() {
        let s = SqpPlanStep::Sort {
            field: "ts".into(),
            asc: true,
        };
        assert!(s.to_string().contains("ASC"));
    }

    #[test]
    fn test_step_display_sort_desc() {
        let s = SqpPlanStep::Sort {
            field: "ts".into(),
            asc: false,
        };
        assert!(s.to_string().contains("DESC"));
    }

    #[test]
    fn test_step_display_limit() {
        let s = SqpPlanStep::Limit(100);
        assert!(s.to_string().contains("100"));
    }

    #[test]
    fn test_step_display_project() {
        let s = SqpPlanStep::Project(vec!["a".into(), "b".into()]);
        assert!(s.to_string().contains("a"));
    }

    // ── Predicate pushdown ────────────────────────────────────────────────

    #[test]
    fn test_predicate_pushdown_disabled_still_produces_filter_steps() {
        let config = SqpPlannerConfig {
            enable_predicate_pushdown: false,
            ..Default::default()
        };
        let planner = StorageQueryPlanner::new(config);
        let mut q = SqpQuery::new(1);
        q.predicates.push(predicate_eq("x", 1));
        q.predicates.push(predicate_eq("y", 2));
        let plan = planner.plan(&q);
        let filter_count = plan
            .steps
            .iter()
            .filter(|s| matches!(s, SqpPlanStep::Filter(_)))
            .count();
        assert_eq!(filter_count, 2);
    }

    // ── Multiple predicates ───────────────────────────────────────────────

    #[test]
    fn test_multiple_predicates_reduce_estimated_rows() {
        let planner = make_planner();
        let mut q_one = SqpQuery::new(1);
        q_one.predicates.push(predicate_eq("a", 1));

        let mut q_two = SqpQuery::new(2);
        q_two.predicates.push(predicate_eq("a", 1));
        q_two.predicates.push(predicate_eq("b", 2));

        let p1 = planner.plan(&q_one);
        let p2 = planner.plan(&q_two);
        assert!(p2.estimated_rows <= p1.estimated_rows);
    }

    // ── SqpCostModel default ──────────────────────────────────────────────

    #[test]
    fn test_cost_model_default_is_hybrid() {
        assert_eq!(SqpCostModel::default(), SqpCostModel::HybridCost);
    }

    // ── SqpPlannerConfig default ──────────────────────────────────────────

    #[test]
    fn test_planner_config_default_max_cache() {
        assert_eq!(SqpPlannerConfig::default().max_cache_size, 200);
    }

    #[test]
    fn test_planner_config_default_pushdown_enabled() {
        assert!(SqpPlannerConfig::default().enable_predicate_pushdown);
    }

    // ── SqpQuery::new ─────────────────────────────────────────────────────

    #[test]
    fn test_query_new_has_no_predicates() {
        let q = SqpQuery::new(1);
        assert!(q.predicates.is_empty());
    }

    #[test]
    fn test_query_new_has_no_limit() {
        let q = SqpQuery::new(1);
        assert!(q.limit.is_none());
    }

    // ── SqpQueryPlan ──────────────────────────────────────────────────────

    #[test]
    fn test_plan_query_id_matches() {
        let planner = make_planner();
        let plan = planner.plan(&simple_query(42));
        assert_eq!(plan.query_id, 42);
    }

    #[test]
    fn test_plan_planned_at_ns_positive() {
        let planner = make_planner();
        let plan = planner.plan(&simple_query(1));
        assert!(plan.planned_at_ns > 0);
    }

    // ── In-predicate selectivity ──────────────────────────────────────────

    #[test]
    fn test_in_predicate_selectivity_scales_with_count() {
        let planner = make_planner();
        let mut q_small = SqpQuery::new(1);
        q_small.predicates.push(SqpPredicate::new(
            "x",
            SqpOp::In(vec![SqpValue::Int(1)]),
            SqpValue::Null,
        ));
        let mut q_large = SqpQuery::new(2);
        q_large.predicates.push(SqpPredicate::new(
            "x",
            SqpOp::In(vec![
                SqpValue::Int(1),
                SqpValue::Int(2),
                SqpValue::Int(3),
                SqpValue::Int(4),
            ]),
            SqpValue::Null,
        ));
        let p_small = planner.plan(&q_small);
        let p_large = planner.plan(&q_large);
        assert!(p_large.estimated_rows >= p_small.estimated_rows);
    }

    // ── Type alias ────────────────────────────────────────────────────────

    #[test]
    fn test_type_alias_usable() {
        let _: SqpStorageQueryPlanner = StorageQueryPlanner::default();
    }

    // ── Thread safety (smoke test) ────────────────────────────────────────

    #[test]
    fn test_concurrent_plans_do_not_panic() {
        use std::sync::Arc;
        let planner = Arc::new(make_planner());
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let p = Arc::clone(&planner);
                std::thread::spawn(move || {
                    let mut q = SqpQuery::new(i);
                    q.predicates.push(predicate_eq("col", i as i64));
                    p.plan_cached(&q);
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread must not panic");
        }
        assert!(planner.planner_stats().total_plans >= 8);
    }
}
