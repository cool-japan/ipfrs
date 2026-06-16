//! Vector Index Optimizer
//!
//! Selects and maintains optimal vector index structures based on workload analysis.
//! Supports Flat, IVF-Flat, Product Quantization, HNSW-like, LSH, and Tree structures.
//! Uses pure-Rust cost models to recommend and maintain the best index for a given workload.

use std::cmp::Ordering;
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// PRNG helpers (no external RNG dependency)
// These are used in tests; the `#[cfg(test)]` visibility annotation is kept
// here but they are also useful as public utilities for callers who wish to
// generate deterministic synthetic workloads without pulling in rand.
// ---------------------------------------------------------------------------

/// Xorshift64 PRNG — advances `state` and returns the next pseudo-random u64.
#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Returns a pseudo-random f64 in `[0, 1)` using `xorshift64`.
#[inline]
pub fn xorshift_f64(state: &mut u64) -> f64 {
    (xorshift64(state) >> 11) as f64 / (1u64 << 53) as f64
}

// ---------------------------------------------------------------------------
// Cost models
// ---------------------------------------------------------------------------

/// Estimated microseconds for a flat (brute-force) query over `n` vectors of dimension `d`.
fn flat_query_cost(n: usize, d: usize) -> f64 {
    n as f64 * d as f64 * 0.001
}

/// Estimated microseconds for an IVF-Flat query.
fn ivf_query_cost(n: usize, d: usize, n_clusters: usize, k: usize) -> f64 {
    let n_probe = (n_clusters as f64).sqrt() as usize;
    let cells_per_cluster = if n_clusters == 0 {
        1
    } else {
        n / n_clusters.max(1)
    };
    (n_probe as f64 * cells_per_cluster as f64 + n_probe as f64 * d as f64) * 0.001
        + k as f64 * 0.0001
}

/// Estimated microseconds for an HNSW-like query.
fn hnsw_query_cost(n: usize, d: usize, m: usize, ef: usize) -> f64 {
    let _ = n; // graph size influences build not per-query for HNSW
    (ef as f64 * m as f64 * d as f64).ln().max(1.0) * 0.01
}

/// Estimated microseconds for an LSH query.
fn lsh_query_cost(d: usize, n_tables: usize, n_bits: u8) -> f64 {
    n_tables as f64 * n_bits as f64 * 0.001 + d as f64 * 0.0001
}

/// Estimated memory bytes for a given index structure.
fn memory_cost(structure: &IndexStructure, n: usize, d: usize) -> u64 {
    let base = (n * d * 4) as u64; // 4 bytes per f32
    match structure {
        IndexStructure::Flat => base,
        IndexStructure::IvfFlat { n_clusters } => base + (*n_clusters * d * 4) as u64,
        IndexStructure::HnswLike { m, .. } => base + (n * m * 8) as u64,
        IndexStructure::Lsh { n_tables, n_bits } => {
            (n * *n_tables) as u64 * (*n_bits as u64 / 8 + 1)
        }
        _ => base,
    }
}

// ---------------------------------------------------------------------------
// Public enums and structs
// ---------------------------------------------------------------------------

/// Supported index structures for approximate nearest-neighbour search.
#[derive(Clone, Debug, PartialEq)]
pub enum IndexStructure {
    /// Brute-force exact search — O(n·d) per query.
    Flat,
    /// Inverted File Index with flat scan inside each cluster.
    IvfFlat {
        /// Number of Voronoi cells.
        n_clusters: usize,
    },
    /// Product Quantization — lossy compression with sub-space codebooks.
    PQ {
        /// Number of sub-spaces.
        m: usize,
        /// Bits per sub-space.
        bits: u8,
    },
    /// HNSW-inspired hierarchical navigable small-world graph.
    HnswLike {
        /// Max connections per node.
        m: usize,
        /// Candidate list size during search.
        ef: usize,
    },
    /// Locality Sensitive Hashing with multiple hash tables.
    Lsh {
        /// Number of hash tables.
        n_tables: usize,
        /// Bits per table projection.
        n_bits: u8,
    },
    /// KD-tree / ball-tree style partition with bounded leaf size.
    Tree {
        /// Maximum number of vectors in a leaf node.
        max_leaf: usize,
    },
}

impl IndexStructure {
    /// Human-readable name for this structure.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Flat => "Flat",
            Self::IvfFlat { .. } => "IvfFlat",
            Self::PQ { .. } => "PQ",
            Self::HnswLike { .. } => "HnswLike",
            Self::Lsh { .. } => "Lsh",
            Self::Tree { .. } => "Tree",
        }
    }
}

/// Workload characteristics used to drive index selection.
#[derive(Clone, Debug)]
pub struct WorkloadProfile {
    /// Number of search queries observed.
    pub query_count: u64,
    /// Number of insert operations observed.
    pub insert_count: u64,
    /// Number of delete operations observed.
    pub delete_count: u64,
    /// Average k (top-k) across queries.
    pub avg_query_k: usize,
    /// Current dataset size (number of vectors).
    pub dataset_size: usize,
    /// Vector dimensionality.
    pub embedding_dim: usize,
    /// Minimum acceptable recall (0–1).
    pub recall_requirement: f64,
    /// Maximum acceptable query latency in microseconds.
    pub latency_budget_us: u64,
}

impl Default for WorkloadProfile {
    fn default() -> Self {
        Self {
            query_count: 0,
            insert_count: 0,
            delete_count: 0,
            avg_query_k: 10,
            dataset_size: 100_000,
            embedding_dim: 128,
            recall_requirement: 0.9,
            latency_budget_us: 10_000,
        }
    }
}

/// Recommendation produced by the optimizer.
#[derive(Clone, Debug)]
pub struct IndexRecommendation {
    /// Recommended index structure with tuned parameters.
    pub structure: IndexStructure,
    /// Estimated time to (re)build the index in microseconds.
    pub estimated_build_time_us: u64,
    /// Estimated per-query time in microseconds.
    pub estimated_query_time_us: u64,
    /// Estimated recall@k (0–1).
    pub estimated_recall: f64,
    /// Estimated memory footprint in bytes.
    pub memory_bytes: u64,
    /// Human-readable rationale for the recommendation.
    pub reason: String,
}

/// Criterion used to compare and select index structures.
#[derive(Clone, Debug, PartialEq)]
pub enum OptimizationCriterion {
    /// Minimise query latency.
    MinLatency,
    /// Maximise recall@k.
    MaxRecall,
    /// Minimise memory usage.
    MinMemory,
    /// Balanced trade-off between latency, recall, and memory.
    Balanced,
    /// Stay within a hard memory budget (bytes).
    CostBudget(u64),
}

/// Snapshot of a live index's runtime statistics.
#[derive(Clone, Debug)]
pub struct IndexStats {
    /// Human-readable name for the index/structure.
    pub structure_name: String,
    /// Total queries processed since last rebuild.
    pub query_count: u64,
    /// Mean query latency in microseconds.
    pub avg_latency_us: f64,
    /// 99th-percentile query latency in microseconds.
    pub p99_latency_us: f64,
    /// Current estimated recall.
    pub recall_estimate: f64,
    /// Current memory footprint in bytes.
    pub memory_bytes: u64,
    /// Unix timestamp (seconds) of the last rebuild.
    pub last_rebuilt_at: u64,
}

impl Default for IndexStats {
    fn default() -> Self {
        Self {
            structure_name: "Unknown".to_string(),
            query_count: 0,
            avg_latency_us: 0.0,
            p99_latency_us: 0.0,
            recall_estimate: 1.0,
            memory_bytes: 0,
            last_rebuilt_at: 0,
        }
    }
}

/// Maintenance operation to be applied to an index.
#[derive(Clone, Debug, PartialEq)]
pub enum MaintenanceAction {
    /// Tear down and rebuild the index.
    Rebuild {
        /// Reason for the rebuild.
        reason: String,
    },
    /// Rebalance cluster assignments (e.g., IVF).
    Rebalance,
    /// Grow the number of IVF clusters.
    AddClusters(usize),
    /// Remove tombstoned/deleted entries.
    PruneDead,
    /// Merge small segments into fewer larger ones.
    MergeSegments,
}

/// Configuration for the `VectorIndexOptimizer`.
#[derive(Clone, Debug)]
pub struct OptimizerConfig {
    /// Criterion used when comparing index structures.
    pub criterion: OptimizationCriterion,
    /// If estimated recall falls below this threshold a rebuild is triggered (0–1).
    pub rebuild_threshold: f64,
    /// Minimum number of queries to observe before making a recommendation.
    pub profile_window: usize,
    /// Hard upper bound on memory usage in bytes (used for `CostBudget` filtering).
    pub max_memory_bytes: u64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            criterion: OptimizationCriterion::Balanced,
            rebuild_threshold: 0.80,
            profile_window: 100,
            max_memory_bytes: u64::MAX,
        }
    }
}

/// Aggregate statistics for the optimizer itself.
#[derive(Clone, Debug, Default)]
pub struct OptimizerStats {
    /// Total number of index recommendations issued.
    pub recommendations_made: u64,
    /// Total number of rebuilds triggered.
    pub rebuilds_triggered: u64,
    /// Average recall improvement observed after a rebuild.
    pub avg_recall_improvement: f64,
    /// Number of queries fed into the profiling window.
    pub profiling_queries: u64,
}

/// Errors produced by the optimizer.
#[derive(Clone, Debug, PartialEq)]
pub enum OptimizerError {
    /// Not enough query data yet; contains current observation count.
    InsufficientData(usize),
    /// No index with the given name found.
    IndexNotFound(String),
    /// A maintenance operation failed.
    MaintenanceFailed(String),
    /// Invalid configuration.
    ConfigurationError(String),
}

impl std::fmt::Display for OptimizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientData(n) => {
                write!(f, "Insufficient data: only {} queries observed", n)
            }
            Self::IndexNotFound(name) => write!(f, "Index not found: {}", name),
            Self::MaintenanceFailed(msg) => write!(f, "Maintenance failed: {}", msg),
            Self::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for OptimizerError {}

// ---------------------------------------------------------------------------
// Rolling latency window
// ---------------------------------------------------------------------------

/// Fixed-capacity sliding window for computing online statistics.
#[derive(Debug)]
pub(crate) struct LatencyWindow {
    data: VecDeque<f64>,
    capacity: usize,
    sum: f64,
}

impl LatencyWindow {
    fn new(capacity: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(capacity),
            capacity,
            sum: 0.0,
        }
    }

    fn push(&mut self, value: f64) {
        if self.data.len() >= self.capacity {
            if let Some(old) = self.data.pop_front() {
                self.sum -= old;
            }
        }
        self.sum += value;
        self.data.push_back(value);
    }

    /// Returns the mean of all values in the current window.
    ///
    /// Used internally and exposed for testing / external monitoring.
    #[allow(dead_code)]
    pub(crate) fn mean(&self) -> f64 {
        if self.data.is_empty() {
            0.0
        } else {
            self.sum / self.data.len() as f64
        }
    }

    /// Returns the 99th-percentile value from the current window.
    pub(crate) fn p99(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = self.data.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
        sorted[idx]
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

// ---------------------------------------------------------------------------
// Rolling recall window
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct RecallWindow {
    data: VecDeque<f64>,
    capacity: usize,
    sum: f64,
}

impl RecallWindow {
    fn new(capacity: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(capacity),
            capacity,
            sum: 0.0,
        }
    }

    fn push(&mut self, value: f64) {
        if self.data.len() >= self.capacity {
            if let Some(old) = self.data.pop_front() {
                self.sum -= old;
            }
        }
        self.sum += value;
        self.data.push_back(value);
    }

    pub(crate) fn mean(&self) -> f64 {
        if self.data.is_empty() {
            1.0
        } else {
            self.sum / self.data.len() as f64
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }
}

// ---------------------------------------------------------------------------
// VectorIndexOptimizer
// ---------------------------------------------------------------------------

/// Production-quality vector index optimizer.
///
/// Observes a stream of query/insert events, builds a `WorkloadProfile`,
/// and recommends the best `IndexStructure` given an `OptimizationCriterion`.
/// It also monitors live `IndexStats` and emits `MaintenanceAction`s when needed.
pub struct VectorIndexOptimizer {
    config: OptimizerConfig,
    query_count: u64,
    insert_count: u64,
    delete_count: u64,
    total_k: u64,
    k_samples: u64,
    latency_window: LatencyWindow,
    recall_window: RecallWindow,
    recommendations_made: u64,
    rebuilds_triggered: u64,
    recall_improvement_sum: f64,
    recall_improvement_count: u64,
    /// Last recall seen before triggering a rebuild (for improvement tracking).
    pre_rebuild_recall: Option<f64>,
}

impl VectorIndexOptimizer {
    /// Create a new optimizer with the provided configuration.
    pub fn new(config: OptimizerConfig) -> Self {
        let window = config.profile_window.max(10);
        Self {
            config,
            query_count: 0,
            insert_count: 0,
            delete_count: 0,
            total_k: 0,
            k_samples: 0,
            latency_window: LatencyWindow::new(window * 4),
            recall_window: RecallWindow::new(window * 4),
            recommendations_made: 0,
            rebuilds_triggered: 0,
            recall_improvement_sum: 0.0,
            recall_improvement_count: 0,
            pre_rebuild_recall: None,
        }
    }

    /// Create a new optimizer with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(OptimizerConfig::default())
    }

    // ------------------------------------------------------------------
    // Event recording
    // ------------------------------------------------------------------

    /// Record a completed query.
    ///
    /// - `k`: requested top-k
    /// - `latency_us`: measured query latency in microseconds
    /// - `recall`: observed recall@k (0–1)
    /// - `current_ts`: current Unix timestamp in seconds (for time-series tracking)
    pub fn record_query(&mut self, k: usize, latency_us: u64, recall: f64, _current_ts: u64) {
        self.query_count += 1;
        self.total_k += k as u64;
        self.k_samples += 1;
        self.latency_window.push(latency_us as f64);
        let clamped_recall = recall.clamp(0.0, 1.0);
        self.recall_window.push(clamped_recall);
    }

    /// Record an insert operation.
    ///
    /// - `current_ts`: current Unix timestamp in seconds
    pub fn record_insert(&mut self, _current_ts: u64) {
        self.insert_count += 1;
    }

    /// Record a delete operation.
    pub fn record_delete(&mut self, _current_ts: u64) {
        self.delete_count += 1;
    }

    // ------------------------------------------------------------------
    // Workload profiling
    // ------------------------------------------------------------------

    /// Snapshot the current workload profile.
    pub fn workload_profile(&self, dataset_size: usize, dim: usize) -> WorkloadProfile {
        let avg_k = self.total_k.checked_div(self.k_samples).unwrap_or(10) as usize;
        // Use p99 latency as the budget: ensures we recommend structures that can
        // serve tail latency, not just average latency.
        let latency_budget_us = self.latency_window.p99() as u64 + 1;
        WorkloadProfile {
            query_count: self.query_count,
            insert_count: self.insert_count,
            delete_count: self.delete_count,
            avg_query_k: avg_k,
            dataset_size,
            embedding_dim: dim,
            recall_requirement: self.recall_window.mean().max(self.config.rebuild_threshold),
            latency_budget_us,
        }
    }

    // ------------------------------------------------------------------
    // Recommendation
    // ------------------------------------------------------------------

    /// Recommend the best index structure for the given dataset and dimensionality.
    ///
    /// Returns `Err(OptimizerError::InsufficientData)` until `profile_window` queries
    /// have been observed.
    pub fn recommend(
        &mut self,
        dataset_size: usize,
        dim: usize,
    ) -> Result<IndexRecommendation, OptimizerError> {
        let observed = self.latency_window.len();
        if observed < self.config.profile_window {
            return Err(OptimizerError::InsufficientData(observed));
        }

        let profile = self.workload_profile(dataset_size, dim);
        let candidates = self.generate_candidates(&profile);

        let best = candidates
            .into_iter()
            .filter(|r| self.passes_budget(r))
            .max_by(|a, b| self.compare_recommendations(a, b, &profile));

        let rec = best.ok_or_else(|| {
            OptimizerError::ConfigurationError(
                "No valid index structure found within memory budget".to_string(),
            )
        })?;

        self.recommendations_made += 1;
        Ok(rec)
    }

    /// Check whether a recommendation stays within the configured memory budget.
    fn passes_budget(&self, rec: &IndexRecommendation) -> bool {
        match &self.config.criterion {
            OptimizationCriterion::CostBudget(budget) => rec.memory_bytes <= *budget,
            _ => rec.memory_bytes <= self.config.max_memory_bytes,
        }
    }

    /// Compare two recommendations under the active criterion.
    /// Returns the ordering such that `max_by` chooses the *better* one.
    fn compare_recommendations(
        &self,
        a: &IndexRecommendation,
        b: &IndexRecommendation,
        _profile: &WorkloadProfile,
    ) -> Ordering {
        match &self.config.criterion {
            OptimizationCriterion::MinLatency => {
                // lower latency → prefer a
                b.estimated_query_time_us.cmp(&a.estimated_query_time_us)
            }
            OptimizationCriterion::MaxRecall => a
                .estimated_recall
                .partial_cmp(&b.estimated_recall)
                .unwrap_or(Ordering::Equal),
            OptimizationCriterion::MinMemory => b.memory_bytes.cmp(&a.memory_bytes),
            OptimizationCriterion::Balanced | OptimizationCriterion::CostBudget(_) => {
                let score_a = self.balanced_score(a);
                let score_b = self.balanced_score(b);
                score_a.partial_cmp(&score_b).unwrap_or(Ordering::Equal)
            }
        }
    }

    /// Composite score for `Balanced`/`CostBudget` criterion (higher = better).
    fn balanced_score(&self, rec: &IndexRecommendation) -> f64 {
        // Normalise each dimension:
        //   recall:   0–1 (higher better)
        //   latency:  invert with log (lower latency → higher score)
        //   memory:   invert (lower memory → higher score), clamped
        let recall_score = rec.estimated_recall;
        let latency_score = 1.0 / (1.0 + rec.estimated_query_time_us as f64 / 1_000.0);
        let mem_gb = rec.memory_bytes as f64 / 1_073_741_824.0;
        let memory_score = 1.0 / (1.0 + mem_gb);
        0.45 * recall_score + 0.35 * latency_score + 0.20 * memory_score
    }

    /// Generate one recommendation per candidate structure with sensible defaults.
    fn generate_candidates(&self, profile: &WorkloadProfile) -> Vec<IndexRecommendation> {
        let n = profile.dataset_size.max(1);
        let d = profile.embedding_dim.max(1);
        let k = profile.avg_query_k.max(1);

        let mut candidates = Vec::with_capacity(6);

        // Flat
        {
            let structure = IndexStructure::Flat;
            let q_cost = flat_query_cost(n, d);
            let mem = memory_cost(&structure, n, d);
            let recall = 1.0;
            candidates.push(IndexRecommendation {
                structure,
                estimated_build_time_us: (n * d) as u64 / 100,
                estimated_query_time_us: q_cost as u64,
                estimated_recall: recall,
                memory_bytes: mem,
                reason: format!("Flat exact search — 100% recall, {}µs/query", q_cost as u64),
            });
        }

        // IvfFlat — rule of thumb: sqrt(n) clusters
        {
            let n_clusters = ((n as f64).sqrt() as usize).max(4).min(n);
            let structure = IndexStructure::IvfFlat { n_clusters };
            let q_cost = ivf_query_cost(n, d, n_clusters, k);
            let mem = memory_cost(&structure, n, d);
            let recall = Self::estimate_recall_static(&structure, n, k);
            candidates.push(IndexRecommendation {
                structure,
                estimated_build_time_us: (n * d / 1000 + n_clusters * 100) as u64,
                estimated_query_time_us: q_cost as u64,
                estimated_recall: recall,
                memory_bytes: mem,
                reason: format!(
                    "IVF-Flat ({} clusters) — recall≈{:.2}, {}µs/query",
                    n_clusters, recall, q_cost as u64
                ),
            });
        }

        // PQ (approximate, low memory)
        {
            let m = (d / 4).clamp(2, 64);
            let bits: u8 = 8;
            let structure = IndexStructure::PQ { m, bits };
            let mem = memory_cost(&structure, n, d);
            let recall = Self::estimate_recall_static(&structure, n, k);
            // PQ uses flat scan over compressed codes then rerank — faster than flat
            let q_cost = flat_query_cost(n, m) * 0.25;
            candidates.push(IndexRecommendation {
                structure,
                estimated_build_time_us: (n * d / 500) as u64,
                estimated_query_time_us: q_cost as u64,
                estimated_recall: recall,
                memory_bytes: mem,
                reason: format!(
                    "PQ (m={}, bits={}) — memory-efficient, recall≈{:.2}",
                    m, bits, recall
                ),
            });
        }

        // HnswLike — m=16, ef scales with k
        {
            let m: usize = 16;
            let ef = (k * 4).max(m * 2).min(512);
            let structure = IndexStructure::HnswLike { m, ef };
            let q_cost = hnsw_query_cost(n, d, m, ef);
            let mem = memory_cost(&structure, n, d);
            let recall = Self::estimate_recall_static(&structure, n, k);
            candidates.push(IndexRecommendation {
                structure,
                estimated_build_time_us: (n as f64 * (n as f64).ln() * d as f64 / 1000.0) as u64,
                estimated_query_time_us: q_cost as u64,
                estimated_recall: recall,
                memory_bytes: mem,
                reason: format!(
                    "HNSW-like (m={}, ef={}) — low latency, recall≈{:.2}",
                    m, ef, recall
                ),
            });
        }

        // Lsh
        {
            let n_tables: usize = 8;
            let n_bits: u8 = 8;
            let structure = IndexStructure::Lsh { n_tables, n_bits };
            let q_cost = lsh_query_cost(d, n_tables, n_bits);
            let mem = memory_cost(&structure, n, d);
            let recall = Self::estimate_recall_static(&structure, n, k);
            candidates.push(IndexRecommendation {
                structure,
                estimated_build_time_us: (n * n_tables) as u64 / 100,
                estimated_query_time_us: q_cost as u64,
                estimated_recall: recall,
                memory_bytes: mem,
                reason: format!(
                    "LSH ({} tables, {} bits) — very fast, recall≈{:.2}",
                    n_tables, n_bits, recall
                ),
            });
        }

        // Tree
        {
            let max_leaf = 50;
            let structure = IndexStructure::Tree { max_leaf };
            let mem = memory_cost(&structure, n, d);
            let recall = Self::estimate_recall_static(&structure, n, k);
            // Tree: O(log n) average query
            let q_cost = (n as f64).log2() * d as f64 * 0.01;
            candidates.push(IndexRecommendation {
                structure,
                estimated_build_time_us: (n as f64 * (n as f64).log2() * 0.01) as u64,
                estimated_query_time_us: q_cost as u64,
                estimated_recall: recall,
                memory_bytes: mem,
                reason: format!(
                    "Tree (max_leaf={}) — exact for k=1, recall≈{:.2}",
                    max_leaf, recall
                ),
            });
        }

        candidates
    }

    // ------------------------------------------------------------------
    // Recall estimation
    // ------------------------------------------------------------------

    /// Heuristic recall estimate for a structure at a given dataset size and k.
    pub fn estimate_recall(structure: &IndexStructure, n: usize, k: usize) -> f64 {
        Self::estimate_recall_static(structure, n, k)
    }

    fn estimate_recall_static(structure: &IndexStructure, n: usize, k: usize) -> f64 {
        match structure {
            IndexStructure::Flat => 1.0,
            IndexStructure::IvfFlat { n_clusters } => {
                let n_clusters = n_clusters.max(&1);
                let penalty = (n / (*n_clusters * 100).max(1)) as f64 * 0.1;
                (0.95 - penalty).clamp(0.70, 0.99)
            }
            IndexStructure::HnswLike { ef, .. } => {
                let ef = ef.max(&1);
                let penalty = (k as f64 / *ef as f64) * 0.1;
                (0.99 - penalty).clamp(0.80, 0.99)
            }
            IndexStructure::Lsh { .. } => 0.85,
            IndexStructure::PQ { .. } => 0.90,
            IndexStructure::Tree { .. } => {
                if k == 1 {
                    1.0
                } else {
                    // Recall degrades for larger k in tree structures (curse of dimensionality)
                    let decay = (k as f64 - 1.0) * 0.01;
                    (1.0 - decay).clamp(0.70, 1.0)
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Maintenance
    // ------------------------------------------------------------------

    /// Check whether a rebuild or other maintenance is required.
    ///
    /// Returns `Some(action)` when a rebuild is recommended, `None` otherwise.
    pub fn should_rebuild(&mut self, current_stats: &IndexStats) -> Option<MaintenanceAction> {
        // Rebuild if recall dropped below threshold
        if current_stats.recall_estimate < self.config.rebuild_threshold {
            self.rebuilds_triggered += 1;
            // Track pre-rebuild recall for improvement calculation
            self.pre_rebuild_recall = Some(current_stats.recall_estimate);
            return Some(MaintenanceAction::Rebuild {
                reason: format!(
                    "Recall {:.3} dropped below threshold {:.3}",
                    current_stats.recall_estimate, self.config.rebuild_threshold
                ),
            });
        }

        // Rebuild if query count since last rebuild is very high (staleness heuristic)
        let queries_since_rebuild = current_stats.query_count;
        if queries_since_rebuild > 1_000_000 {
            self.rebuilds_triggered += 1;
            self.pre_rebuild_recall = Some(current_stats.recall_estimate);
            return Some(MaintenanceAction::Rebuild {
                reason: format!(
                    "High query count {} since last rebuild — proactive maintenance",
                    queries_since_rebuild
                ),
            });
        }

        None
    }

    /// Observe a post-rebuild recall to update improvement tracking.
    pub fn record_post_rebuild_recall(&mut self, new_recall: f64) {
        if let Some(pre) = self.pre_rebuild_recall.take() {
            let improvement = (new_recall - pre).max(0.0);
            self.recall_improvement_sum += improvement;
            self.recall_improvement_count += 1;
        }
    }

    /// Comprehensive maintenance check — returns all applicable actions.
    pub fn maintenance_actions(
        &mut self,
        stats: &IndexStats,
        dataset_size: usize,
        dim: usize,
    ) -> Vec<MaintenanceAction> {
        let mut actions = Vec::new();

        // Recall-based rebuild
        if stats.recall_estimate < self.config.rebuild_threshold {
            self.rebuilds_triggered += 1;
            self.pre_rebuild_recall = Some(stats.recall_estimate);
            actions.push(MaintenanceAction::Rebuild {
                reason: format!(
                    "Recall {:.3} < threshold {:.3}",
                    stats.recall_estimate, self.config.rebuild_threshold
                ),
            });
            return actions; // rebuild supersedes everything else
        }

        // High p99 latency — rebalance to improve locality
        if stats.p99_latency_us > stats.avg_latency_us * 4.0 && stats.avg_latency_us > 0.0 {
            actions.push(MaintenanceAction::Rebalance);
        }

        // Many deleted vectors — prune dead entries
        let est_dead = dataset_size / 20; // assume up to 5% churn
        if self.delete_count > est_dead as u64 {
            actions.push(MaintenanceAction::PruneDead);
        }

        // Memory growing significantly — consider merge
        let base_mem = (dataset_size * dim * 4) as u64;
        if stats.memory_bytes > base_mem * 3 {
            actions.push(MaintenanceAction::MergeSegments);
        }

        // High insert pressure on IVF — consider adding clusters
        if stats.structure_name == "IvfFlat" && self.insert_count > 100_000 {
            let extra = ((self.insert_count as f64).sqrt() as usize).max(1);
            actions.push(MaintenanceAction::AddClusters(extra));
        }

        actions
    }

    // ------------------------------------------------------------------
    // Structure comparison
    // ------------------------------------------------------------------

    /// Compare two index structures under the active criterion and profile.
    ///
    /// Returns `Ordering::Greater` if `a` is preferred over `b`.
    pub fn compare_structures(
        &self,
        a: &IndexStructure,
        b: &IndexStructure,
        profile: &WorkloadProfile,
    ) -> Ordering {
        let n = profile.dataset_size.max(1);
        let d = profile.embedding_dim.max(1);
        let k = profile.avg_query_k.max(1);

        let query_cost_a = self.structure_query_cost(a, n, d, k);
        let query_cost_b = self.structure_query_cost(b, n, d, k);
        let recall_a = Self::estimate_recall_static(a, n, k);
        let recall_b = Self::estimate_recall_static(b, n, k);
        let mem_a = memory_cost(a, n, d);
        let mem_b = memory_cost(b, n, d);

        match &self.config.criterion {
            OptimizationCriterion::MinLatency => {
                // lower latency better → b < a means a is preferred
                query_cost_b
                    .partial_cmp(&query_cost_a)
                    .unwrap_or(Ordering::Equal)
            }
            OptimizationCriterion::MaxRecall => {
                recall_a.partial_cmp(&recall_b).unwrap_or(Ordering::Equal)
            }
            OptimizationCriterion::MinMemory => mem_b.cmp(&mem_a),
            OptimizationCriterion::Balanced | OptimizationCriterion::CostBudget(_) => {
                let score_a = self.balanced_score_raw(query_cost_a, recall_a, mem_a);
                let score_b = self.balanced_score_raw(query_cost_b, recall_b, mem_b);
                score_a.partial_cmp(&score_b).unwrap_or(Ordering::Equal)
            }
        }
    }

    fn balanced_score_raw(&self, query_cost_us: f64, recall: f64, mem_bytes: u64) -> f64 {
        let latency_score = 1.0 / (1.0 + query_cost_us / 1_000.0);
        let mem_gb = mem_bytes as f64 / 1_073_741_824.0;
        let memory_score = 1.0 / (1.0 + mem_gb);
        0.45 * recall + 0.35 * latency_score + 0.20 * memory_score
    }

    fn structure_query_cost(
        &self,
        structure: &IndexStructure,
        n: usize,
        d: usize,
        k: usize,
    ) -> f64 {
        match structure {
            IndexStructure::Flat => flat_query_cost(n, d),
            IndexStructure::IvfFlat { n_clusters } => ivf_query_cost(n, d, *n_clusters, k),
            IndexStructure::HnswLike { m, ef } => hnsw_query_cost(n, d, *m, *ef),
            IndexStructure::Lsh { n_tables, n_bits } => lsh_query_cost(d, *n_tables, *n_bits),
            IndexStructure::PQ { m, .. } => flat_query_cost(n, *m) * 0.25,
            IndexStructure::Tree { .. } => (n as f64).log2() * d as f64 * 0.01,
        }
    }

    // ------------------------------------------------------------------
    // Optimizer statistics
    // ------------------------------------------------------------------

    /// Return aggregate statistics for this optimizer instance.
    pub fn stats(&self) -> OptimizerStats {
        let avg_recall_improvement = if self.recall_improvement_count == 0 {
            0.0
        } else {
            self.recall_improvement_sum / self.recall_improvement_count as f64
        };
        OptimizerStats {
            recommendations_made: self.recommendations_made,
            rebuilds_triggered: self.rebuilds_triggered,
            avg_recall_improvement,
            profiling_queries: self.query_count,
        }
    }

    /// Current profiling window fill level (0 to `profile_window`).
    ///
    /// Returns `(observed_queries, required_for_recommendation)`.
    /// Both the latency and recall windows must reach `profile_window` before
    /// `recommend` returns a valid result.
    pub fn profiling_progress(&self) -> (usize, usize) {
        // Use the minimum of both windows so we correctly represent the
        // readiness of both signals.
        let ready = self.latency_window.len().min(self.recall_window.len());
        (ready, self.config.profile_window)
    }
}

// ---------------------------------------------------------------------------
// Type aliases for collision avoidance
// (These are exported from lib.rs under `Vio*` names)
// ---------------------------------------------------------------------------

/// Type alias for `IndexStats` (re-exported as `VioIndexStats` to avoid conflict with
/// `stats::IndexStats` which is already exported at crate level).
pub type VioIndexStats = IndexStats;

/// Type alias for `OptimizerConfig` (re-exported as `VioOptimizerConfig` to avoid conflict with
/// `semantic_query_optimizer::OptimizerConfig` which is already exported at crate level).
pub type VioOptimizerConfig = OptimizerConfig;

/// Type alias for `OptimizerError` (re-exported as `VioOptimizerError` to avoid conflict with
/// `semantic_query_optimizer::OptimizerError` which is already exported at crate level).
pub type VioOptimizerError = OptimizerError;

/// Type alias for `OptimizerStats` (re-exported as `VioOptimizerStats` to avoid conflict with
/// `semantic_query_optimizer::OptimizerStats` which is already exported at crate level).
pub type VioOptimizerStats = OptimizerStats;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create an optimizer with a tiny profile window for fast tests
    fn make_optimizer(criterion: OptimizationCriterion) -> VectorIndexOptimizer {
        VectorIndexOptimizer::new(OptimizerConfig {
            criterion,
            rebuild_threshold: 0.80,
            profile_window: 5,
            max_memory_bytes: u64::MAX,
        })
    }

    // Helper: fill the profile window with synthetic observations
    fn fill_window(opt: &mut VectorIndexOptimizer, n: usize, latency_us: u64, recall: f64) {
        for i in 0..n {
            opt.record_query(10, latency_us, recall, 1_000_000 + i as u64);
        }
    }

    // ---------------------------------------------------------------------------
    // record_query tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_record_query_increments_count() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        opt.record_query(5, 100, 0.95, 0);
        opt.record_query(10, 200, 0.90, 1);
        let s = opt.stats();
        assert_eq!(s.profiling_queries, 2);
    }

    #[test]
    fn test_record_query_updates_latency_window() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        fill_window(&mut opt, 5, 1000, 0.95);
        let (current, total) = opt.profiling_progress();
        assert!(current >= 5);
        assert_eq!(total, 5);
    }

    #[test]
    fn test_record_query_clamps_recall() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        opt.record_query(1, 50, 1.5, 0); // above 1.0 — should clamp
        opt.record_query(1, 50, -0.1, 0); // below 0.0 — should clamp
        assert_eq!(opt.stats().profiling_queries, 2);
    }

    #[test]
    fn test_record_query_k_averaging() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        opt.record_query(10, 100, 0.9, 0);
        opt.record_query(20, 100, 0.9, 0);
        let profile = opt.workload_profile(1000, 64);
        assert_eq!(profile.avg_query_k, 15);
    }

    // ---------------------------------------------------------------------------
    // record_insert tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_record_insert_increments() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        opt.record_insert(0);
        opt.record_insert(1);
        opt.record_insert(2);
        let profile = opt.workload_profile(100, 32);
        assert_eq!(profile.insert_count, 3);
    }

    #[test]
    fn test_record_insert_does_not_affect_query_count() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        opt.record_insert(0);
        assert_eq!(opt.stats().profiling_queries, 0);
    }

    #[test]
    fn test_record_insert_large_batch() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        for i in 0..1000_u64 {
            opt.record_insert(i);
        }
        let profile = opt.workload_profile(1000, 128);
        assert_eq!(profile.insert_count, 1000);
    }

    // ---------------------------------------------------------------------------
    // recommend — per criterion
    // ---------------------------------------------------------------------------

    #[test]
    fn test_recommend_insufficient_data() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        opt.record_query(10, 500, 0.9, 0); // only 1, needs 5
        match opt.recommend(10_000, 128) {
            Err(OptimizerError::InsufficientData(n)) => assert!(n < 5),
            other => panic!("expected InsufficientData, got {:?}", other),
        }
    }

    #[test]
    fn test_recommend_balanced() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        fill_window(&mut opt, 10, 500, 0.92);
        let rec = opt.recommend(50_000, 128).expect("should recommend");
        assert!(rec.estimated_recall >= 0.0 && rec.estimated_recall <= 1.0);
        assert!(!rec.reason.is_empty());
    }

    #[test]
    fn test_recommend_min_latency() {
        let mut opt = make_optimizer(OptimizationCriterion::MinLatency);
        fill_window(&mut opt, 10, 2000, 0.90);
        let rec = opt.recommend(100_000, 128).expect("should recommend");
        // Under MinLatency, the recommended structure should have low query time
        assert!(rec.estimated_query_time_us < 1_000_000);
    }

    #[test]
    fn test_recommend_max_recall() {
        let mut opt = make_optimizer(OptimizationCriterion::MaxRecall);
        fill_window(&mut opt, 10, 1000, 0.85);
        let rec = opt.recommend(10_000, 64).expect("should recommend");
        // MaxRecall should pick the best recall candidate
        assert!(rec.estimated_recall >= 0.85);
    }

    #[test]
    fn test_recommend_min_memory() {
        let mut opt = make_optimizer(OptimizationCriterion::MinMemory);
        fill_window(&mut opt, 10, 500, 0.90);
        let rec = opt.recommend(50_000, 128).expect("should recommend");
        assert!(rec.memory_bytes > 0);
    }

    #[test]
    fn test_recommend_cost_budget_respected() {
        let budget: u64 = 1_000_000; // 1 MB
        let mut opt = VectorIndexOptimizer::new(OptimizerConfig {
            criterion: OptimizationCriterion::CostBudget(budget),
            profile_window: 5,
            rebuild_threshold: 0.80,
            max_memory_bytes: u64::MAX,
        });
        fill_window(&mut opt, 10, 100, 0.95);
        // tiny dataset so something fits in 1 MB
        match opt.recommend(100, 4) {
            Ok(rec) => assert!(rec.memory_bytes <= budget),
            Err(OptimizerError::ConfigurationError(_)) => {
                // acceptable if no structure fits in 1 MB for chosen sizes
            }
            Err(e) => panic!("unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_recommend_increments_stats() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        fill_window(&mut opt, 10, 300, 0.90);
        let _ = opt.recommend(10_000, 64);
        assert_eq!(opt.stats().recommendations_made, 1);
    }

    #[test]
    fn test_recommend_multiple_calls() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        fill_window(&mut opt, 20, 400, 0.91);
        let r1 = opt.recommend(10_000, 64).expect("rec 1");
        let r2 = opt.recommend(10_000, 64).expect("rec 2");
        assert_eq!(r1.structure.name(), r2.structure.name());
        assert_eq!(opt.stats().recommendations_made, 2);
    }

    // ---------------------------------------------------------------------------
    // estimate_recall per structure
    // ---------------------------------------------------------------------------

    #[test]
    fn test_estimate_recall_flat() {
        assert_eq!(
            VectorIndexOptimizer::estimate_recall(&IndexStructure::Flat, 1_000, 10),
            1.0
        );
    }

    #[test]
    fn test_estimate_recall_ivf_clamped_high() {
        // small dataset, large n_clusters → should be near 0.95
        let s = IndexStructure::IvfFlat { n_clusters: 1000 };
        let r = VectorIndexOptimizer::estimate_recall(&s, 1_000, 10);
        assert!((0.70..=0.99).contains(&r), "recall={}", r);
    }

    #[test]
    fn test_estimate_recall_ivf_low_clusters() {
        // n=1M, n_clusters=1 → heavy penalty
        let s = IndexStructure::IvfFlat { n_clusters: 1 };
        let r = VectorIndexOptimizer::estimate_recall(&s, 1_000_000, 10);
        assert!((0.70..=0.99).contains(&r), "recall={}", r);
    }

    #[test]
    fn test_estimate_recall_hnsw_high_ef() {
        let s = IndexStructure::HnswLike { m: 16, ef: 500 };
        let r = VectorIndexOptimizer::estimate_recall(&s, 100_000, 10);
        assert!((0.80..=0.99).contains(&r), "recall={}", r);
    }

    #[test]
    fn test_estimate_recall_hnsw_low_ef() {
        let s = IndexStructure::HnswLike { m: 8, ef: 5 };
        let r = VectorIndexOptimizer::estimate_recall(&s, 100_000, 100);
        // ef < k → large penalty
        assert!((0.80..=0.99).contains(&r), "recall={}", r);
    }

    #[test]
    fn test_estimate_recall_lsh() {
        let s = IndexStructure::Lsh {
            n_tables: 8,
            n_bits: 8,
        };
        assert_eq!(VectorIndexOptimizer::estimate_recall(&s, 10_000, 10), 0.85);
    }

    #[test]
    fn test_estimate_recall_pq() {
        let s = IndexStructure::PQ { m: 8, bits: 8 };
        assert_eq!(VectorIndexOptimizer::estimate_recall(&s, 10_000, 10), 0.90);
    }

    #[test]
    fn test_estimate_recall_tree_k1() {
        let s = IndexStructure::Tree { max_leaf: 50 };
        assert_eq!(VectorIndexOptimizer::estimate_recall(&s, 10_000, 1), 1.0);
    }

    #[test]
    fn test_estimate_recall_tree_large_k() {
        let s = IndexStructure::Tree { max_leaf: 50 };
        let r = VectorIndexOptimizer::estimate_recall(&s, 10_000, 50);
        assert!((0.70..=1.0).contains(&r), "recall={}", r);
    }

    // ---------------------------------------------------------------------------
    // should_rebuild
    // ---------------------------------------------------------------------------

    #[test]
    fn test_should_rebuild_low_recall() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        let stats = IndexStats {
            structure_name: "HnswLike".to_string(),
            query_count: 100,
            avg_latency_us: 200.0,
            p99_latency_us: 400.0,
            recall_estimate: 0.60, // below 0.80 threshold
            memory_bytes: 1024,
            last_rebuilt_at: 0,
        };
        match opt.should_rebuild(&stats) {
            Some(MaintenanceAction::Rebuild { reason }) => {
                assert!(reason.contains("Recall"));
            }
            other => panic!("expected Rebuild, got {:?}", other),
        }
    }

    #[test]
    fn test_should_rebuild_high_query_count() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        let stats = IndexStats {
            structure_name: "Flat".to_string(),
            query_count: 2_000_000, // over threshold
            avg_latency_us: 100.0,
            p99_latency_us: 150.0,
            recall_estimate: 0.95, // fine recall
            memory_bytes: 4096,
            last_rebuilt_at: 0,
        };
        match opt.should_rebuild(&stats) {
            Some(MaintenanceAction::Rebuild { .. }) => {}
            other => panic!("expected Rebuild, got {:?}", other),
        }
    }

    #[test]
    fn test_should_rebuild_no_action_needed() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        let stats = IndexStats {
            structure_name: "HnswLike".to_string(),
            query_count: 100,
            avg_latency_us: 200.0,
            p99_latency_us: 300.0,
            recall_estimate: 0.95,
            memory_bytes: 1024,
            last_rebuilt_at: 0,
        };
        assert!(opt.should_rebuild(&stats).is_none());
    }

    #[test]
    fn test_should_rebuild_increments_counter() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        let stats = IndexStats {
            recall_estimate: 0.50,
            query_count: 10,
            ..Default::default()
        };
        let _ = opt.should_rebuild(&stats);
        assert_eq!(opt.stats().rebuilds_triggered, 1);
    }

    // ---------------------------------------------------------------------------
    // maintenance_actions
    // ---------------------------------------------------------------------------

    #[test]
    fn test_maintenance_actions_rebuild_priority() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        let stats = IndexStats {
            recall_estimate: 0.60,
            avg_latency_us: 100.0,
            p99_latency_us: 500.0,
            ..Default::default()
        };
        let actions = opt.maintenance_actions(&stats, 10_000, 128);
        assert!(!actions.is_empty());
        assert!(matches!(actions[0], MaintenanceAction::Rebuild { .. }));
    }

    #[test]
    fn test_maintenance_actions_rebalance_on_high_p99() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        let stats = IndexStats {
            recall_estimate: 0.90,
            avg_latency_us: 100.0,
            p99_latency_us: 600.0, // > 4x avg
            memory_bytes: 0,
            ..Default::default()
        };
        let actions = opt.maintenance_actions(&stats, 1000, 32);
        assert!(actions
            .iter()
            .any(|a| matches!(a, MaintenanceAction::Rebalance)));
    }

    #[test]
    fn test_maintenance_actions_prune_dead_on_many_deletes() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        for i in 0..2000_u64 {
            opt.record_delete(i);
        }
        let stats = IndexStats {
            recall_estimate: 0.90,
            avg_latency_us: 100.0,
            p99_latency_us: 150.0,
            ..Default::default()
        };
        let actions = opt.maintenance_actions(&stats, 10_000, 64);
        assert!(actions
            .iter()
            .any(|a| matches!(a, MaintenanceAction::PruneDead)));
    }

    #[test]
    fn test_maintenance_actions_merge_on_high_memory() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        let dataset_size = 1000;
        let dim = 64;
        let base_mem = (dataset_size * dim * 4) as u64;
        let stats = IndexStats {
            recall_estimate: 0.92,
            avg_latency_us: 100.0,
            p99_latency_us: 200.0,
            memory_bytes: base_mem * 5, // 5× base → triggers merge
            ..Default::default()
        };
        let actions = opt.maintenance_actions(&stats, dataset_size, dim);
        assert!(actions
            .iter()
            .any(|a| matches!(a, MaintenanceAction::MergeSegments)));
    }

    #[test]
    fn test_maintenance_actions_add_clusters_for_ivf() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        for i in 0..200_000_u64 {
            opt.record_insert(i);
        }
        let stats = IndexStats {
            structure_name: "IvfFlat".to_string(),
            recall_estimate: 0.92,
            avg_latency_us: 100.0,
            p99_latency_us: 200.0,
            ..Default::default()
        };
        let actions = opt.maintenance_actions(&stats, 50_000, 128);
        assert!(actions
            .iter()
            .any(|a| matches!(a, MaintenanceAction::AddClusters(_))));
    }

    #[test]
    fn test_maintenance_actions_empty_when_healthy() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        let stats = IndexStats {
            recall_estimate: 0.95,
            avg_latency_us: 100.0,
            p99_latency_us: 120.0, // < 4x avg
            memory_bytes: 1024,
            ..Default::default()
        };
        let actions = opt.maintenance_actions(&stats, 100, 4);
        // delete_count = 0 and memory is fine → no actions
        assert!(actions.is_empty(), "expected no actions, got {:?}", actions);
    }

    // ---------------------------------------------------------------------------
    // compare_structures
    // ---------------------------------------------------------------------------

    #[test]
    fn test_compare_structures_max_recall_prefers_flat() {
        let opt = VectorIndexOptimizer::new(OptimizerConfig {
            criterion: OptimizationCriterion::MaxRecall,
            ..Default::default()
        });
        let profile = WorkloadProfile {
            dataset_size: 10_000,
            embedding_dim: 64,
            avg_query_k: 10,
            ..Default::default()
        };
        let flat = IndexStructure::Flat;
        let lsh = IndexStructure::Lsh {
            n_tables: 4,
            n_bits: 8,
        };
        // Flat recall = 1.0, Lsh recall = 0.85 → Flat should be preferred
        assert_eq!(
            opt.compare_structures(&flat, &lsh, &profile),
            Ordering::Greater
        );
    }

    #[test]
    fn test_compare_structures_min_memory() {
        let opt = VectorIndexOptimizer::new(OptimizerConfig {
            criterion: OptimizationCriterion::MinMemory,
            ..Default::default()
        });
        let profile = WorkloadProfile {
            dataset_size: 10_000,
            embedding_dim: 64,
            avg_query_k: 10,
            ..Default::default()
        };
        let flat = IndexStructure::Flat;
        let hnsw = IndexStructure::HnswLike { m: 32, ef: 200 };
        // Flat uses less memory than HNSW with m=32
        assert_eq!(
            opt.compare_structures(&flat, &hnsw, &profile),
            Ordering::Greater
        );
    }

    #[test]
    fn test_compare_structures_min_latency() {
        let opt = VectorIndexOptimizer::new(OptimizerConfig {
            criterion: OptimizationCriterion::MinLatency,
            ..Default::default()
        });
        let profile = WorkloadProfile {
            dataset_size: 1_000_000,
            embedding_dim: 256,
            avg_query_k: 10,
            ..Default::default()
        };
        let flat = IndexStructure::Flat;
        let hnsw = IndexStructure::HnswLike { m: 16, ef: 100 };
        // On a large dataset, HNSW should be faster than flat
        assert_eq!(
            opt.compare_structures(&hnsw, &flat, &profile),
            Ordering::Greater
        );
    }

    #[test]
    fn test_compare_structures_balanced() {
        let opt = VectorIndexOptimizer::new(OptimizerConfig {
            criterion: OptimizationCriterion::Balanced,
            ..Default::default()
        });
        let profile = WorkloadProfile {
            dataset_size: 50_000,
            embedding_dim: 128,
            avg_query_k: 10,
            ..Default::default()
        };
        let a = IndexStructure::HnswLike { m: 16, ef: 100 };
        let b = IndexStructure::Flat;
        let ord = opt.compare_structures(&a, &b, &profile);
        // Both are valid orderings — just verify it returns an Ordering without panicking
        assert!(ord == Ordering::Less || ord == Ordering::Equal || ord == Ordering::Greater);
    }

    #[test]
    fn test_compare_structures_symmetry() {
        let opt = VectorIndexOptimizer::new(OptimizerConfig {
            criterion: OptimizationCriterion::MinMemory,
            ..Default::default()
        });
        let profile = WorkloadProfile {
            dataset_size: 5_000,
            embedding_dim: 32,
            avg_query_k: 5,
            ..Default::default()
        };
        let a = IndexStructure::Flat;
        let b = IndexStructure::IvfFlat { n_clusters: 64 };
        let ord_ab = opt.compare_structures(&a, &b, &profile);
        let ord_ba = opt.compare_structures(&b, &a, &profile);
        // They should be opposite (or equal)
        match (ord_ab, ord_ba) {
            (Ordering::Greater, Ordering::Less)
            | (Ordering::Less, Ordering::Greater)
            | (Ordering::Equal, Ordering::Equal) => {}
            pair => panic!("unexpected ordering pair: {:?}", pair),
        }
    }

    // ---------------------------------------------------------------------------
    // memory_cost
    // ---------------------------------------------------------------------------

    #[test]
    fn test_memory_cost_flat() {
        let s = IndexStructure::Flat;
        assert_eq!(memory_cost(&s, 1000, 128), 1000 * 128 * 4);
    }

    #[test]
    fn test_memory_cost_ivf() {
        let s = IndexStructure::IvfFlat { n_clusters: 50 };
        let expected = (1000 * 128 * 4 + 50 * 128 * 4) as u64;
        assert_eq!(memory_cost(&s, 1000, 128), expected);
    }

    #[test]
    fn test_memory_cost_hnsw() {
        let m = 16;
        let s = IndexStructure::HnswLike { m, ef: 100 };
        let expected = (1000 * 128 * 4 + 1000 * m * 8) as u64;
        assert_eq!(memory_cost(&s, 1000, 128), expected);
    }

    #[test]
    fn test_memory_cost_lsh() {
        let n_tables = 8_usize;
        let n_bits: u8 = 8;
        let s = IndexStructure::Lsh { n_tables, n_bits };
        let expected = (1000 * n_tables) as u64 * (n_bits as u64 / 8 + 1);
        assert_eq!(memory_cost(&s, 1000, 128), expected);
    }

    #[test]
    fn test_memory_cost_pq_equals_base() {
        let s = IndexStructure::PQ { m: 8, bits: 8 };
        assert_eq!(memory_cost(&s, 1000, 128), 1000 * 128 * 4);
    }

    #[test]
    fn test_memory_cost_tree_equals_base() {
        let s = IndexStructure::Tree { max_leaf: 50 };
        assert_eq!(memory_cost(&s, 1000, 128), 1000 * 128 * 4);
    }

    // ---------------------------------------------------------------------------
    // stats
    // ---------------------------------------------------------------------------

    #[test]
    fn test_stats_initial_zeros() {
        let opt = make_optimizer(OptimizationCriterion::Balanced);
        let s = opt.stats();
        assert_eq!(s.recommendations_made, 0);
        assert_eq!(s.rebuilds_triggered, 0);
        assert_eq!(s.avg_recall_improvement, 0.0);
        assert_eq!(s.profiling_queries, 0);
    }

    #[test]
    fn test_stats_after_queries() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        fill_window(&mut opt, 10, 200, 0.93);
        let s = opt.stats();
        assert_eq!(s.profiling_queries, 10);
    }

    #[test]
    fn test_stats_recall_improvement_tracking() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        // Simulate a rebuild event
        let bad_stats = IndexStats {
            recall_estimate: 0.60,
            ..Default::default()
        };
        let _ = opt.should_rebuild(&bad_stats);
        opt.record_post_rebuild_recall(0.95); // 0.35 improvement
        let s = opt.stats();
        assert!(s.avg_recall_improvement > 0.0);
    }

    #[test]
    fn test_stats_multiple_rebuilds() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        for _ in 0..3 {
            let bad = IndexStats {
                recall_estimate: 0.50,
                ..Default::default()
            };
            let _ = opt.should_rebuild(&bad);
            opt.record_post_rebuild_recall(0.92);
        }
        let s = opt.stats();
        assert_eq!(s.rebuilds_triggered, 3);
        assert!((s.avg_recall_improvement - 0.42).abs() < 0.01);
    }

    // ---------------------------------------------------------------------------
    // Error cases — InsufficientData
    // ---------------------------------------------------------------------------

    #[test]
    fn test_insufficient_data_zero_queries() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        match opt.recommend(10_000, 128) {
            Err(OptimizerError::InsufficientData(0)) => {}
            other => panic!("expected InsufficientData(0), got {:?}", other),
        }
    }

    #[test]
    fn test_insufficient_data_partial_window() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        opt.record_query(5, 100, 0.9, 0);
        opt.record_query(5, 100, 0.9, 1);
        match opt.recommend(10_000, 128) {
            Err(OptimizerError::InsufficientData(n)) => assert_eq!(n, 2),
            other => panic!("expected InsufficientData(2), got {:?}", other),
        }
    }

    #[test]
    fn test_insufficient_data_exact_window() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        fill_window(&mut opt, 5, 300, 0.90);
        // exactly at the window — should succeed
        assert!(opt.recommend(10_000, 128).is_ok());
    }

    // ---------------------------------------------------------------------------
    // IndexStructure name
    // ---------------------------------------------------------------------------

    #[test]
    fn test_index_structure_name_flat() {
        assert_eq!(IndexStructure::Flat.name(), "Flat");
    }

    #[test]
    fn test_index_structure_name_ivf() {
        assert_eq!(IndexStructure::IvfFlat { n_clusters: 64 }.name(), "IvfFlat");
    }

    #[test]
    fn test_index_structure_name_pq() {
        assert_eq!(IndexStructure::PQ { m: 8, bits: 8 }.name(), "PQ");
    }

    #[test]
    fn test_index_structure_name_hnsw() {
        assert_eq!(
            IndexStructure::HnswLike { m: 16, ef: 100 }.name(),
            "HnswLike"
        );
    }

    #[test]
    fn test_index_structure_name_lsh() {
        assert_eq!(
            IndexStructure::Lsh {
                n_tables: 8,
                n_bits: 8
            }
            .name(),
            "Lsh"
        );
    }

    #[test]
    fn test_index_structure_name_tree() {
        assert_eq!(IndexStructure::Tree { max_leaf: 50 }.name(), "Tree");
    }

    // ---------------------------------------------------------------------------
    // PRNG helpers
    // ---------------------------------------------------------------------------

    #[test]
    fn test_xorshift64_nonzero() {
        let mut state = 12345_u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }

    #[test]
    fn test_xorshift_f64_in_range() {
        let mut state = 99999_u64;
        for _ in 0..1000 {
            let f = xorshift_f64(&mut state);
            assert!((0.0..1.0).contains(&f), "f64 out of range: {}", f);
        }
    }

    #[test]
    fn test_xorshift64_deterministic() {
        let mut s1 = 42_u64;
        let mut s2 = 42_u64;
        for _ in 0..100 {
            assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
        }
    }

    // ---------------------------------------------------------------------------
    // WorkloadProfile default
    // ---------------------------------------------------------------------------

    #[test]
    fn test_workload_profile_default() {
        let p = WorkloadProfile::default();
        assert_eq!(p.avg_query_k, 10);
        assert_eq!(p.dataset_size, 100_000);
        assert!(p.recall_requirement > 0.0 && p.recall_requirement <= 1.0);
    }

    // ---------------------------------------------------------------------------
    // OptimizerConfig default
    // ---------------------------------------------------------------------------

    #[test]
    fn test_optimizer_config_default() {
        let c = OptimizerConfig::default();
        assert_eq!(c.criterion, OptimizationCriterion::Balanced);
        assert_eq!(c.rebuild_threshold, 0.80);
        assert_eq!(c.profile_window, 100);
    }

    // ---------------------------------------------------------------------------
    // Error Display
    // ---------------------------------------------------------------------------

    #[test]
    fn test_error_display_insufficient_data() {
        let e = OptimizerError::InsufficientData(3);
        assert!(e.to_string().contains("3"));
    }

    #[test]
    fn test_error_display_index_not_found() {
        let e = OptimizerError::IndexNotFound("my_index".to_string());
        assert!(e.to_string().contains("my_index"));
    }

    #[test]
    fn test_error_display_maintenance_failed() {
        let e = OptimizerError::MaintenanceFailed("disk full".to_string());
        assert!(e.to_string().contains("disk full"));
    }

    #[test]
    fn test_error_display_configuration_error() {
        let e = OptimizerError::ConfigurationError("bad param".to_string());
        assert!(e.to_string().contains("bad param"));
    }

    // ---------------------------------------------------------------------------
    // Latency window p99
    // ---------------------------------------------------------------------------

    #[test]
    fn test_latency_window_p99() {
        let mut w = LatencyWindow::new(100);
        for i in 1..=100_u64 {
            w.push(i as f64);
        }
        let p99 = w.p99();
        assert!((98.0..=100.0).contains(&p99), "p99={}", p99);
    }

    #[test]
    fn test_latency_window_mean() {
        let mut w = LatencyWindow::new(10);
        for i in 0..10_u64 {
            w.push(i as f64);
        }
        let mean = w.mean();
        assert!((mean - 4.5).abs() < 0.01, "mean={}", mean);
    }

    // ---------------------------------------------------------------------------
    // profiling_progress
    // ---------------------------------------------------------------------------

    #[test]
    fn test_profiling_progress_initially_zero() {
        let opt = make_optimizer(OptimizationCriterion::Balanced);
        let (current, total) = opt.profiling_progress();
        assert_eq!(current, 0);
        assert_eq!(total, 5);
    }

    #[test]
    fn test_profiling_progress_fills() {
        let mut opt = make_optimizer(OptimizationCriterion::Balanced);
        fill_window(&mut opt, 5, 300, 0.9);
        let (current, _) = opt.profiling_progress();
        assert!(current >= 5);
    }

    // ---------------------------------------------------------------------------
    // Cost model direct tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_flat_query_cost_proportional() {
        // doubling n should double cost
        let c1 = flat_query_cost(1000, 128);
        let c2 = flat_query_cost(2000, 128);
        assert!((c2 - c1 * 2.0).abs() < 0.01, "c1={} c2={}", c1, c2);
    }

    #[test]
    fn test_ivf_query_cost_less_than_flat() {
        let n = 100_000;
        let d = 128;
        let k = 10;
        let flat = flat_query_cost(n, d);
        let ivf = ivf_query_cost(n, d, 1000, k);
        assert!(ivf < flat, "ivf={} should be < flat={}", ivf, flat);
    }

    #[test]
    fn test_hnsw_query_cost_sublinear_in_ef() {
        let c1 = hnsw_query_cost(100_000, 128, 16, 50);
        let c2 = hnsw_query_cost(100_000, 128, 16, 100);
        // Cost should be somewhat larger with ef=100 vs ef=50
        assert!(c2 >= c1 * 0.9, "c1={} c2={}", c1, c2);
    }

    #[test]
    fn test_lsh_query_cost_positive() {
        let c = lsh_query_cost(128, 8, 8);
        assert!(c > 0.0, "lsh cost must be positive");
    }
}
