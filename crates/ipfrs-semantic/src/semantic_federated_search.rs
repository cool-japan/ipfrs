//! Semantic Federated Search Coordinator
//!
//! Merges result sets from multiple remote nodes, deduplicates by CID, re-ranks by combined
//! score, and handles partial failures with configurable quorum requirements.
//!
//! # Design
//!
//! - Nodes are tracked with performance statistics (latency, success rate) that influence
//!   their effective weight during score aggregation.
//! - Four merge strategies cover a range of precision/recall trade-offs:
//!   `SimpleUnion` (max-score union), `WeightedMerge` (weighted average), `QuorumIntersect`
//!   (only CIDs seen by ≥N nodes), and `RankFusion` (Reciprocal Rank Fusion, k=60).
//! - Stats are updated via EWMA (α = 0.1) after every query.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// RemoteNode
// ---------------------------------------------------------------------------

/// A registered remote search node with performance metadata.
#[derive(Debug, Clone)]
pub struct RemoteNode {
    /// Unique node identifier.
    pub id: String,
    /// Base weight assigned by the operator (higher ⇒ more trusted).
    pub weight: f64,
    /// Round-trip latency of the last query to this node, in milliseconds.
    pub last_latency_ms: u64,
    /// Fraction of recent queries that succeeded (in [0.0, 1.0]).
    pub success_rate: f64,
}

impl RemoteNode {
    /// Construct a new `RemoteNode` with the given identifiers and initial stats.
    pub fn new(
        id: impl Into<String>,
        weight: f64,
        last_latency_ms: u64,
        success_rate: f64,
    ) -> Self {
        Self {
            id: id.into(),
            weight,
            last_latency_ms,
            success_rate,
        }
    }

    /// Compute the effective contribution weight of this node.
    ///
    /// `effective_weight = weight × success_rate / (1 + last_latency_ms / 1000)`
    ///
    /// Higher latency and lower success rate both reduce the node's influence.
    pub fn effective_weight(&self) -> f64 {
        self.weight * self.success_rate / (1.0 + self.last_latency_ms as f64 / 1000.0)
    }
}

// ---------------------------------------------------------------------------
// RemoteResult
// ---------------------------------------------------------------------------

/// A single search result returned by a remote node.
#[derive(Debug, Clone)]
pub struct RemoteResult {
    /// Content identifier of the result.
    pub cid: String,
    /// Similarity score as reported by the remote node.
    pub score: f64,
    /// The node that produced this result.
    pub node_id: String,
    /// Arbitrary key-value metadata from the remote node.
    pub metadata: Vec<(String, String)>,
}

impl RemoteResult {
    /// Construct a `RemoteResult`.
    pub fn new(
        cid: impl Into<String>,
        score: f64,
        node_id: impl Into<String>,
        metadata: Vec<(String, String)>,
    ) -> Self {
        Self {
            cid: cid.into(),
            score,
            node_id: node_id.into(),
            metadata,
        }
    }
}

// ---------------------------------------------------------------------------
// FederatedQuery
// ---------------------------------------------------------------------------

/// Query parameters for a federated search request.
#[derive(Debug, Clone)]
pub struct FederatedQuery {
    /// The query embedding vector.
    pub query_embedding: Vec<f64>,
    /// Maximum number of results to return after merging.
    pub top_k: usize,
    /// Minimum merged score threshold; results below this are discarded.
    pub min_score: f64,
    /// Minimum number of successful node responses required for the query to succeed.
    pub required_quorum: usize,
}

impl Default for FederatedQuery {
    fn default() -> Self {
        Self {
            query_embedding: Vec::new(),
            top_k: 10,
            min_score: 0.0,
            required_quorum: 1,
        }
    }
}

impl FederatedQuery {
    /// Construct a `FederatedQuery` with default thresholds and the given embedding.
    pub fn new(query_embedding: Vec<f64>) -> Self {
        Self {
            query_embedding,
            ..Self::default()
        }
    }

    /// Builder: set `top_k`.
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Builder: set `min_score`.
    pub fn with_min_score(mut self, min_score: f64) -> Self {
        self.min_score = min_score;
        self
    }

    /// Builder: set `required_quorum`.
    pub fn with_quorum(mut self, required_quorum: usize) -> Self {
        self.required_quorum = required_quorum;
        self
    }
}

// ---------------------------------------------------------------------------
// NodeResponse
// ---------------------------------------------------------------------------

/// The full response from a single remote node, including timing and success status.
#[derive(Debug, Clone)]
pub struct NodeResponse {
    /// The node that produced this response.
    pub node_id: String,
    /// Results returned by the node (empty on failure).
    pub results: Vec<RemoteResult>,
    /// Round-trip latency of this response in milliseconds.
    pub latency_ms: u64,
    /// Whether the node responded successfully.
    pub success: bool,
}

impl NodeResponse {
    /// Construct a successful `NodeResponse`.
    pub fn success(
        node_id: impl Into<String>,
        results: Vec<RemoteResult>,
        latency_ms: u64,
    ) -> Self {
        Self {
            node_id: node_id.into(),
            results,
            latency_ms,
            success: true,
        }
    }

    /// Construct a failure `NodeResponse` (no results).
    pub fn failure(node_id: impl Into<String>, latency_ms: u64) -> Self {
        Self {
            node_id: node_id.into(),
            results: Vec::new(),
            latency_ms,
            success: false,
        }
    }
}

// ---------------------------------------------------------------------------
// MergeStrategy
// ---------------------------------------------------------------------------

/// Strategy used to combine per-node result lists into a single ranked list.
#[derive(Debug, Clone)]
pub enum MergeStrategy {
    /// Include all CIDs; score = maximum score from any responding node.
    SimpleUnion,

    /// Include all CIDs; score = weighted average of per-node scores, weighted by each node's
    /// `effective_weight()`. If a node did not return a CID, it does not contribute to the
    /// average.
    WeightedMerge,

    /// Include only CIDs that appeared in results from at least `n` distinct nodes;
    /// score = maximum score from any of those nodes.
    QuorumIntersect(usize),

    /// Reciprocal Rank Fusion: for each node sort results by score descending, assign 1-based
    /// ranks, and compute `score = Σ 1/(k + rank)` across all nodes where the CID appeared.
    /// The smoothing constant `k` is fixed at 60 (standard RRF).
    RankFusion,
}

// ---------------------------------------------------------------------------
// FederatedResult
// ---------------------------------------------------------------------------

/// A single deduplicated, merged result from the federated coordinator.
#[derive(Debug, Clone)]
pub struct FederatedResult {
    /// Content identifier.
    pub cid: String,
    /// Merged score (semantics depend on `MergeStrategy`).
    pub merged_score: f64,
    /// IDs of nodes that contributed this result.
    pub contributing_nodes: Vec<String>,
    /// Number of nodes that returned this CID.
    pub appearance_count: usize,
}

impl FederatedResult {
    /// Construct a `FederatedResult`.
    pub fn new(
        cid: impl Into<String>,
        merged_score: f64,
        contributing_nodes: Vec<String>,
        appearance_count: usize,
    ) -> Self {
        Self {
            cid: cid.into(),
            merged_score,
            contributing_nodes,
            appearance_count,
        }
    }
}

// ---------------------------------------------------------------------------
// FederatedStats
// ---------------------------------------------------------------------------

/// Running statistics for the `SemanticFederatedSearch` coordinator.
#[derive(Debug, Clone, Default)]
pub struct FederatedStats {
    /// Total number of federated queries processed.
    pub queries: u64,
    /// EWMA of query latency in milliseconds.
    pub avg_latency_ms: f64,
    /// EWMA of result count per successful query.
    pub avg_results_returned: f64,
    /// Number of queries that returned no results due to quorum failure.
    pub partial_failures: u64,
    /// Number of queries that produced zero results (after filtering).
    pub zero_result_queries: u64,
}

impl FederatedStats {
    /// Update rolling averages using exponential weighted moving average (α = 0.1).
    fn update_ewma(&mut self, latency_ms: f64, results_returned: usize) {
        const ALPHA: f64 = 0.1;
        if self.queries == 1 {
            // First query — seed the EWMA directly.
            self.avg_latency_ms = latency_ms;
            self.avg_results_returned = results_returned as f64;
        } else {
            self.avg_latency_ms = ALPHA * latency_ms + (1.0 - ALPHA) * self.avg_latency_ms;
            self.avg_results_returned =
                ALPHA * results_returned as f64 + (1.0 - ALPHA) * self.avg_results_returned;
        }
    }
}

// ---------------------------------------------------------------------------
// Intermediate bookkeeping types (private)
// ---------------------------------------------------------------------------

/// Per-CID aggregation scratch space used during merging.
#[derive(Debug)]
struct CidAccumulator {
    /// All (node_id, score) pairs seen for this CID across successful responses.
    entries: Vec<(String, f64)>,
}

impl CidAccumulator {
    fn new(node_id: String, score: f64) -> Self {
        Self {
            entries: vec![(node_id, score)],
        }
    }

    fn push(&mut self, node_id: String, score: f64) {
        self.entries.push((node_id, score));
    }

    fn appearance_count(&self) -> usize {
        self.entries.len()
    }

    fn max_score(&self) -> f64 {
        self.entries
            .iter()
            .map(|(_, s)| *s)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    fn weighted_score(&self, nodes: &HashMap<String, RemoteNode>) -> f64 {
        let mut total_weight = 0.0_f64;
        let mut weighted_sum = 0.0_f64;
        for (node_id, score) in &self.entries {
            let w = nodes
                .get(node_id)
                .map(|n| n.effective_weight())
                .unwrap_or(1.0);
            weighted_sum += w * score;
            total_weight += w;
        }
        if total_weight == 0.0 {
            0.0
        } else {
            weighted_sum / total_weight
        }
    }

    fn contributing_nodes(&self) -> Vec<String> {
        self.entries.iter().map(|(id, _)| id.clone()).collect()
    }
}

// ---------------------------------------------------------------------------
// SemanticFederatedSearch
// ---------------------------------------------------------------------------

/// Federated search coordinator.
///
/// Collects results from multiple remote nodes, merges them according to the configured
/// `MergeStrategy`, deduplicates by CID, re-ranks by combined score, and enforces quorum
/// requirements for fault tolerance.
pub struct SemanticFederatedSearch {
    /// Registered remote nodes keyed by their ID.
    nodes: HashMap<String, RemoteNode>,
    /// Active merge strategy.
    strategy: MergeStrategy,
    /// Running statistics.
    stats: FederatedStats,
}

impl SemanticFederatedSearch {
    /// Create a new `SemanticFederatedSearch` with the given merge strategy and no nodes.
    pub fn new(strategy: MergeStrategy) -> Self {
        Self {
            nodes: HashMap::new(),
            strategy,
            stats: FederatedStats::default(),
        }
    }

    // -----------------------------------------------------------------------
    // Node management
    // -----------------------------------------------------------------------

    /// Register a node.  Returns `true` if the node is new, `false` if its ID was already known
    /// (in which case the existing entry is left unchanged).
    pub fn register_node(&mut self, node: RemoteNode) -> bool {
        if self.nodes.contains_key(&node.id) {
            return false;
        }
        self.nodes.insert(node.id.clone(), node);
        true
    }

    /// Remove a node by ID.  Returns `true` if the node existed and was removed.
    pub fn remove_node(&mut self, node_id: &str) -> bool {
        self.nodes.remove(node_id).is_some()
    }

    /// Update a node's performance statistics after a query attempt.
    ///
    /// - `latency_ms`: observed round-trip time.
    /// - `success`: whether the query to the node succeeded.
    ///
    /// `success_rate` is updated with EWMA α = 0.2 on the success indicator (0 or 1).
    pub fn update_node_stats(&mut self, node_id: &str, latency_ms: u64, success: bool) {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.last_latency_ms = latency_ms;
            const SR_ALPHA: f64 = 0.2;
            let outcome = if success { 1.0 } else { 0.0 };
            node.success_rate = SR_ALPHA * outcome + (1.0 - SR_ALPHA) * node.success_rate;
        }
    }

    // -----------------------------------------------------------------------
    // Merge
    // -----------------------------------------------------------------------

    /// Merge responses from multiple nodes into a single ranked result list.
    ///
    /// Returns an empty `Vec` if the number of successful responses is below
    /// `query.required_quorum` (and increments `stats.partial_failures`).
    pub fn merge_responses(
        &mut self,
        query: &FederatedQuery,
        responses: Vec<NodeResponse>,
    ) -> Vec<FederatedResult> {
        // --- Quorum check ------------------------------------------------
        let successful_responses: Vec<&NodeResponse> =
            responses.iter().filter(|r| r.success).collect();
        let successful_count = successful_responses.len();

        self.stats.queries += 1;

        // Derive a representative latency as the average over successful responses.
        let avg_latency = if successful_count > 0 {
            successful_responses
                .iter()
                .map(|r| r.latency_ms as f64)
                .sum::<f64>()
                / successful_count as f64
        } else {
            0.0
        };

        if successful_count < query.required_quorum {
            self.stats.partial_failures += 1;
            self.stats.update_ewma(avg_latency, 0);
            return Vec::new();
        }

        // --- Build per-CID accumulators ----------------------------------
        let accumulators: HashMap<String, CidAccumulator> =
            self.build_accumulators(&successful_responses);

        // --- Apply merge strategy ----------------------------------------
        let mut merged: Vec<FederatedResult> = match &self.strategy.clone() {
            MergeStrategy::SimpleUnion => self.apply_simple_union(&accumulators),
            MergeStrategy::WeightedMerge => self.apply_weighted_merge(&accumulators),
            MergeStrategy::QuorumIntersect(n) => self.apply_quorum_intersect(&accumulators, *n),
            MergeStrategy::RankFusion => self.apply_rank_fusion(&successful_responses),
        };

        // --- Filter, sort, truncate --------------------------------------
        merged.retain(|r| r.merged_score >= query.min_score);
        merged.sort_by(|a, b| {
            b.merged_score
                .partial_cmp(&a.merged_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(query.top_k);

        // --- Update stats ------------------------------------------------
        if merged.is_empty() {
            self.stats.zero_result_queries += 1;
        }
        self.stats.update_ewma(avg_latency, merged.len());

        merged
    }

    // -----------------------------------------------------------------------
    // Deduplication (public helper)
    // -----------------------------------------------------------------------

    /// Deduplicate a slice of `FederatedResult` by CID, keeping the entry with the
    /// highest `merged_score` for each CID.
    pub fn deduplicate(results: &[FederatedResult]) -> Vec<FederatedResult> {
        let mut best: HashMap<&str, &FederatedResult> = HashMap::new();
        for r in results {
            let entry = best.entry(r.cid.as_str()).or_insert(r);
            if r.merged_score > entry.merged_score {
                *entry = r;
            }
        }
        // Sort by descending merged_score for deterministic output.
        let mut out: Vec<FederatedResult> = best.values().map(|r| (*r).clone()).collect();
        out.sort_by(|a, b| {
            b.merged_score
                .partial_cmp(&a.merged_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Return a reference to the running stats.
    pub fn stats(&self) -> &FederatedStats {
        &self.stats
    }

    /// Return the number of registered nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return all nodes that have a non-zero success rate.
    pub fn active_nodes(&self) -> Vec<&RemoteNode> {
        self.nodes
            .values()
            .filter(|n| n.success_rate > 0.0)
            .collect()
    }

    /// Return the top `n` nodes ordered by `effective_weight()` descending.
    pub fn best_nodes(&self, n: usize) -> Vec<&RemoteNode> {
        let mut nodes: Vec<&RemoteNode> = self.nodes.values().collect();
        nodes.sort_by(|a, b| {
            b.effective_weight()
                .partial_cmp(&a.effective_weight())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        nodes.truncate(n);
        nodes
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn build_accumulators(&self, responses: &[&NodeResponse]) -> HashMap<String, CidAccumulator> {
        let mut map: HashMap<String, CidAccumulator> = HashMap::new();
        for response in responses {
            for result in &response.results {
                map.entry(result.cid.clone())
                    .and_modify(|acc| acc.push(result.node_id.clone(), result.score))
                    .or_insert_with(|| CidAccumulator::new(result.node_id.clone(), result.score));
            }
        }
        map
    }

    fn apply_simple_union(
        &self,
        accumulators: &HashMap<String, CidAccumulator>,
    ) -> Vec<FederatedResult> {
        accumulators
            .iter()
            .map(|(cid, acc)| FederatedResult {
                cid: cid.clone(),
                merged_score: acc.max_score(),
                appearance_count: acc.appearance_count(),
                contributing_nodes: acc.contributing_nodes(),
            })
            .collect()
    }

    fn apply_weighted_merge(
        &self,
        accumulators: &HashMap<String, CidAccumulator>,
    ) -> Vec<FederatedResult> {
        accumulators
            .iter()
            .map(|(cid, acc)| FederatedResult {
                cid: cid.clone(),
                merged_score: acc.weighted_score(&self.nodes),
                appearance_count: acc.appearance_count(),
                contributing_nodes: acc.contributing_nodes(),
            })
            .collect()
    }

    fn apply_quorum_intersect(
        &self,
        accumulators: &HashMap<String, CidAccumulator>,
        min_appearances: usize,
    ) -> Vec<FederatedResult> {
        accumulators
            .iter()
            .filter(|(_, acc)| acc.appearance_count() >= min_appearances)
            .map(|(cid, acc)| FederatedResult {
                cid: cid.clone(),
                merged_score: acc.max_score(),
                appearance_count: acc.appearance_count(),
                contributing_nodes: acc.contributing_nodes(),
            })
            .collect()
    }

    /// Reciprocal Rank Fusion.  For each successful response, sort that node's results by
    /// descending score, then assign 1-based ranks.  The RRF score for a CID is
    /// `Σ 1/(k + rank)` summed across all nodes that returned it.
    fn apply_rank_fusion(&self, responses: &[&NodeResponse]) -> Vec<FederatedResult> {
        const K: f64 = 60.0;

        // For each CID accumulate: rrf_score, contributing node set.
        let mut rrf_scores: HashMap<String, f64> = HashMap::new();
        let mut rrf_nodes: HashMap<String, Vec<String>> = HashMap::new();
        let mut rrf_counts: HashMap<String, usize> = HashMap::new();

        for response in responses {
            // Sort this node's results by score descending.
            let mut sorted: Vec<&RemoteResult> = response.results.iter().collect();
            sorted.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for (rank_zero, result) in sorted.iter().enumerate() {
                let rank = (rank_zero + 1) as f64; // 1-based
                let contribution = 1.0 / (K + rank);
                *rrf_scores.entry(result.cid.clone()).or_insert(0.0) += contribution;
                rrf_nodes
                    .entry(result.cid.clone())
                    .or_default()
                    .push(result.node_id.clone());
                *rrf_counts.entry(result.cid.clone()).or_insert(0) += 1;
            }
        }

        rrf_scores
            .into_iter()
            .map(|(cid, score)| {
                let contributing = rrf_nodes.get(&cid).cloned().unwrap_or_default();
                let count = rrf_counts.get(&cid).copied().unwrap_or(0);
                FederatedResult {
                    cid,
                    merged_score: score,
                    contributing_nodes: contributing,
                    appearance_count: count,
                }
            })
            .collect()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper factories
    // -----------------------------------------------------------------------

    fn make_node(id: &str, weight: f64, latency_ms: u64, success_rate: f64) -> RemoteNode {
        RemoteNode::new(id, weight, latency_ms, success_rate)
    }

    fn make_result(cid: &str, score: f64, node_id: &str) -> RemoteResult {
        RemoteResult::new(cid, score, node_id, vec![])
    }

    fn make_result_meta(
        cid: &str,
        score: f64,
        node_id: &str,
        meta: Vec<(&str, &str)>,
    ) -> RemoteResult {
        RemoteResult::new(
            cid,
            score,
            node_id,
            meta.into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        )
    }

    fn make_query(top_k: usize, min_score: f64, quorum: usize) -> FederatedQuery {
        FederatedQuery::new(vec![0.1, 0.2, 0.3])
            .with_top_k(top_k)
            .with_min_score(min_score)
            .with_quorum(quorum)
    }

    fn coordinator_with_nodes(
        strategy: MergeStrategy,
        nodes: Vec<RemoteNode>,
    ) -> SemanticFederatedSearch {
        let mut sfs = SemanticFederatedSearch::new(strategy);
        for n in nodes {
            sfs.register_node(n);
        }
        sfs
    }

    // -----------------------------------------------------------------------
    // 1. RemoteNode::effective_weight
    // -----------------------------------------------------------------------

    #[test]
    fn test_effective_weight_basic() {
        let node = make_node("n1", 2.0, 1000, 0.5);
        // weight=2.0, success_rate=0.5, latency=1000ms => denominator = 1+1=2
        // effective = 2.0 * 0.5 / 2.0 = 0.5
        let ew = node.effective_weight();
        assert!((ew - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_effective_weight_zero_latency() {
        let node = make_node("n1", 1.0, 0, 1.0);
        // effective = 1.0 * 1.0 / (1 + 0) = 1.0
        assert!((node.effective_weight() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_effective_weight_zero_success_rate() {
        let node = make_node("n1", 10.0, 100, 0.0);
        assert_eq!(node.effective_weight(), 0.0);
    }

    #[test]
    fn test_effective_weight_high_latency_reduces_weight() {
        let low = make_node("a", 1.0, 0, 1.0);
        let high = make_node("b", 1.0, 10_000, 1.0);
        assert!(low.effective_weight() > high.effective_weight());
    }

    // -----------------------------------------------------------------------
    // 2. register_node / remove_node
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_node_new() {
        let mut sfs = SemanticFederatedSearch::new(MergeStrategy::SimpleUnion);
        assert!(sfs.register_node(make_node("n1", 1.0, 0, 1.0)));
        assert_eq!(sfs.node_count(), 1);
    }

    #[test]
    fn test_register_node_duplicate_returns_false() {
        let mut sfs = SemanticFederatedSearch::new(MergeStrategy::SimpleUnion);
        sfs.register_node(make_node("n1", 1.0, 0, 1.0));
        assert!(!sfs.register_node(make_node("n1", 2.0, 0, 1.0)));
        assert_eq!(sfs.node_count(), 1);
    }

    #[test]
    fn test_remove_node_existing() {
        let mut sfs = SemanticFederatedSearch::new(MergeStrategy::SimpleUnion);
        sfs.register_node(make_node("n1", 1.0, 0, 1.0));
        assert!(sfs.remove_node("n1"));
        assert_eq!(sfs.node_count(), 0);
    }

    #[test]
    fn test_remove_node_nonexistent() {
        let mut sfs = SemanticFederatedSearch::new(MergeStrategy::SimpleUnion);
        assert!(!sfs.remove_node("ghost"));
    }

    // -----------------------------------------------------------------------
    // 3. update_node_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_update_node_stats_latency() {
        let mut sfs = SemanticFederatedSearch::new(MergeStrategy::SimpleUnion);
        sfs.register_node(make_node("n1", 1.0, 0, 1.0));
        sfs.update_node_stats("n1", 250, true);
        assert_eq!(sfs.nodes["n1"].last_latency_ms, 250);
    }

    #[test]
    fn test_update_node_stats_success_rate_increases() {
        let mut sfs = SemanticFederatedSearch::new(MergeStrategy::SimpleUnion);
        sfs.register_node(make_node("n1", 1.0, 0, 0.5));
        let before = sfs.nodes["n1"].success_rate;
        sfs.update_node_stats("n1", 10, true);
        assert!(sfs.nodes["n1"].success_rate > before);
    }

    #[test]
    fn test_update_node_stats_success_rate_decreases() {
        let mut sfs = SemanticFederatedSearch::new(MergeStrategy::SimpleUnion);
        sfs.register_node(make_node("n1", 1.0, 0, 1.0));
        sfs.update_node_stats("n1", 10, false);
        assert!(sfs.nodes["n1"].success_rate < 1.0);
    }

    #[test]
    fn test_update_node_stats_unknown_node_noop() {
        let mut sfs = SemanticFederatedSearch::new(MergeStrategy::SimpleUnion);
        // Must not panic
        sfs.update_node_stats("ghost", 100, true);
    }

    // -----------------------------------------------------------------------
    // 4. Quorum failures
    // -----------------------------------------------------------------------

    #[test]
    fn test_quorum_failure_increments_partial_failures() {
        let mut sfs = coordinator_with_nodes(
            MergeStrategy::SimpleUnion,
            vec![make_node("n1", 1.0, 0, 1.0)],
        );
        let query = make_query(10, 0.0, 2); // require 2, but only 1 succeeds
        let responses = vec![NodeResponse::success(
            "n1",
            vec![make_result("cid1", 0.9, "n1")],
            50,
        )];
        let results = sfs.merge_responses(&query, responses);
        assert!(results.is_empty());
        assert_eq!(sfs.stats().partial_failures, 1);
    }

    #[test]
    fn test_all_failures_returns_empty() {
        let mut sfs = coordinator_with_nodes(
            MergeStrategy::SimpleUnion,
            vec![make_node("n1", 1.0, 0, 1.0)],
        );
        let query = make_query(10, 0.0, 1);
        let responses = vec![NodeResponse::failure("n1", 100)];
        let results = sfs.merge_responses(&query, responses);
        assert!(results.is_empty());
        assert_eq!(sfs.stats().partial_failures, 1);
    }

    // -----------------------------------------------------------------------
    // 5. SimpleUnion strategy
    // -----------------------------------------------------------------------

    #[test]
    fn test_simple_union_takes_max_score() {
        let mut sfs = coordinator_with_nodes(
            MergeStrategy::SimpleUnion,
            vec![make_node("n1", 1.0, 0, 1.0), make_node("n2", 1.0, 0, 1.0)],
        );
        let query = make_query(10, 0.0, 1);
        let responses = vec![
            NodeResponse::success("n1", vec![make_result("cid1", 0.7, "n1")], 10),
            NodeResponse::success("n2", vec![make_result("cid1", 0.9, "n2")], 10),
        ];
        let results = sfs.merge_responses(&query, responses);
        assert_eq!(results.len(), 1);
        assert!((results[0].merged_score - 0.9).abs() < 1e-12);
    }

    #[test]
    fn test_simple_union_includes_all_cids() {
        let mut sfs = coordinator_with_nodes(
            MergeStrategy::SimpleUnion,
            vec![make_node("n1", 1.0, 0, 1.0), make_node("n2", 1.0, 0, 1.0)],
        );
        let query = make_query(10, 0.0, 1);
        let responses = vec![
            NodeResponse::success("n1", vec![make_result("cidA", 0.8, "n1")], 10),
            NodeResponse::success("n2", vec![make_result("cidB", 0.6, "n2")], 10),
        ];
        let results = sfs.merge_responses(&query, responses);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_simple_union_sorted_descending() {
        let mut sfs = coordinator_with_nodes(
            MergeStrategy::SimpleUnion,
            vec![make_node("n1", 1.0, 0, 1.0)],
        );
        let query = make_query(10, 0.0, 1);
        let responses = vec![NodeResponse::success(
            "n1",
            vec![
                make_result("cidA", 0.3, "n1"),
                make_result("cidB", 0.9, "n1"),
                make_result("cidC", 0.6, "n1"),
            ],
            10,
        )];
        let results = sfs.merge_responses(&query, responses);
        assert_eq!(results.len(), 3);
        assert!(results[0].merged_score >= results[1].merged_score);
        assert!(results[1].merged_score >= results[2].merged_score);
    }

    // -----------------------------------------------------------------------
    // 6. WeightedMerge strategy
    // -----------------------------------------------------------------------

    #[test]
    fn test_weighted_merge_single_node() {
        let mut sfs = coordinator_with_nodes(
            MergeStrategy::WeightedMerge,
            vec![make_node("n1", 1.0, 0, 1.0)],
        );
        let query = make_query(10, 0.0, 1);
        let responses = vec![NodeResponse::success(
            "n1",
            vec![make_result("cid1", 0.75, "n1")],
            10,
        )];
        let results = sfs.merge_responses(&query, responses);
        assert_eq!(results.len(), 1);
        // Single node: weighted avg == the score itself.
        assert!((results[0].merged_score - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_merge_two_equal_weight_nodes() {
        let mut sfs = coordinator_with_nodes(
            MergeStrategy::WeightedMerge,
            vec![make_node("n1", 1.0, 0, 1.0), make_node("n2", 1.0, 0, 1.0)],
        );
        let query = make_query(10, 0.0, 1);
        let responses = vec![
            NodeResponse::success("n1", vec![make_result("cid1", 0.8, "n1")], 10),
            NodeResponse::success("n2", vec![make_result("cid1", 0.6, "n2")], 10),
        ];
        let results = sfs.merge_responses(&query, responses);
        assert_eq!(results.len(), 1);
        // Equal weights => simple average = 0.7
        assert!((results[0].merged_score - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_merge_higher_weight_dominates() {
        let mut sfs = coordinator_with_nodes(
            MergeStrategy::WeightedMerge,
            vec![
                make_node("n1", 10.0, 0, 1.0), // high weight
                make_node("n2", 1.0, 0, 1.0),  // low weight
            ],
        );
        let query = make_query(10, 0.0, 1);
        let responses = vec![
            NodeResponse::success("n1", vec![make_result("cid1", 1.0, "n1")], 10),
            NodeResponse::success("n2", vec![make_result("cid1", 0.0, "n2")], 10),
        ];
        let results = sfs.merge_responses(&query, responses);
        assert_eq!(results.len(), 1);
        // Should be closer to 1.0 than to 0.0 because n1 dominates
        assert!(results[0].merged_score > 0.5);
    }

    // -----------------------------------------------------------------------
    // 7. QuorumIntersect strategy
    // -----------------------------------------------------------------------

    #[test]
    fn test_quorum_intersect_filters_rare_cids() {
        let mut sfs = coordinator_with_nodes(
            MergeStrategy::QuorumIntersect(2),
            vec![make_node("n1", 1.0, 0, 1.0), make_node("n2", 1.0, 0, 1.0)],
        );
        let query = make_query(10, 0.0, 1);
        let responses = vec![
            NodeResponse::success(
                "n1",
                vec![
                    make_result("cidCommon", 0.8, "n1"),
                    make_result("cidRare", 0.7, "n1"),
                ],
                10,
            ),
            NodeResponse::success("n2", vec![make_result("cidCommon", 0.6, "n2")], 10),
        ];
        let results = sfs.merge_responses(&query, responses);
        // cidRare only appears in n1 (count=1 < 2) so it must be excluded
        assert!(results.iter().all(|r| r.cid != "cidRare"));
        assert!(results.iter().any(|r| r.cid == "cidCommon"));
    }

    #[test]
    fn test_quorum_intersect_n1_equals_simple_union() {
        let nodes = vec![make_node("n1", 1.0, 0, 1.0), make_node("n2", 1.0, 0, 1.0)];
        let mut sfs_union = coordinator_with_nodes(MergeStrategy::SimpleUnion, nodes.clone());
        let mut sfs_qi = coordinator_with_nodes(MergeStrategy::QuorumIntersect(1), nodes);

        let query = make_query(10, 0.0, 1);
        let responses = || {
            vec![
                NodeResponse::success(
                    "n1",
                    vec![
                        make_result("cidA", 0.8, "n1"),
                        make_result("cidB", 0.5, "n1"),
                    ],
                    10,
                ),
                NodeResponse::success("n2", vec![make_result("cidC", 0.6, "n2")], 10),
            ]
        };

        let r_union = sfs_union.merge_responses(&query, responses());
        let r_qi = sfs_qi.merge_responses(&query, responses());
        assert_eq!(r_union.len(), r_qi.len());
    }

    // -----------------------------------------------------------------------
    // 8. RankFusion strategy
    // -----------------------------------------------------------------------

    #[test]
    fn test_rank_fusion_top_ranked_from_all_nodes_gets_highest_score() {
        let mut sfs = coordinator_with_nodes(
            MergeStrategy::RankFusion,
            vec![make_node("n1", 1.0, 0, 1.0), make_node("n2", 1.0, 0, 1.0)],
        );
        let query = make_query(10, 0.0, 1);
        let responses = vec![
            NodeResponse::success(
                "n1",
                vec![
                    make_result("cidTop", 0.95, "n1"),
                    make_result("cidLow", 0.4, "n1"),
                ],
                10,
            ),
            NodeResponse::success(
                "n2",
                vec![
                    make_result("cidTop", 0.90, "n2"),
                    make_result("cidLow", 0.3, "n2"),
                ],
                10,
            ),
        ];
        let results = sfs.merge_responses(&query, responses);
        assert_eq!(results[0].cid, "cidTop");
    }

    #[test]
    fn test_rank_fusion_score_formula() {
        // Single node, rank 1 => score = 1/(60+1) ≈ 0.016393
        let mut sfs = coordinator_with_nodes(
            MergeStrategy::RankFusion,
            vec![make_node("n1", 1.0, 0, 1.0)],
        );
        let query = make_query(10, 0.0, 1);
        let responses = vec![NodeResponse::success(
            "n1",
            vec![make_result("cid1", 0.9, "n1")],
            10,
        )];
        let results = sfs.merge_responses(&query, responses);
        let expected = 1.0 / (60.0 + 1.0);
        assert!((results[0].merged_score - expected).abs() < 1e-12);
    }

    #[test]
    fn test_rank_fusion_two_nodes_contribute() {
        // Two nodes, both rank cid1 first => score = 2/(61) ≈ 0.032786
        let mut sfs = coordinator_with_nodes(
            MergeStrategy::RankFusion,
            vec![make_node("n1", 1.0, 0, 1.0), make_node("n2", 1.0, 0, 1.0)],
        );
        let query = make_query(10, 0.0, 1);
        let responses = vec![
            NodeResponse::success("n1", vec![make_result("cid1", 0.9, "n1")], 10),
            NodeResponse::success("n2", vec![make_result("cid1", 0.8, "n2")], 10),
        ];
        let results = sfs.merge_responses(&query, responses);
        let expected = 2.0 / 61.0;
        assert!((results[0].merged_score - expected).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // 9. min_score filter
    // -----------------------------------------------------------------------

    #[test]
    fn test_min_score_filters_low_results() {
        let mut sfs = coordinator_with_nodes(
            MergeStrategy::SimpleUnion,
            vec![make_node("n1", 1.0, 0, 1.0)],
        );
        let query = make_query(10, 0.5, 1);
        let responses = vec![NodeResponse::success(
            "n1",
            vec![
                make_result("cidHigh", 0.8, "n1"),
                make_result("cidLow", 0.3, "n1"),
            ],
            10,
        )];
        let results = sfs.merge_responses(&query, responses);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].cid, "cidHigh");
    }

    // -----------------------------------------------------------------------
    // 10. top_k truncation
    // -----------------------------------------------------------------------

    #[test]
    fn test_top_k_truncation() {
        let mut sfs = coordinator_with_nodes(
            MergeStrategy::SimpleUnion,
            vec![make_node("n1", 1.0, 0, 1.0)],
        );
        let query = make_query(2, 0.0, 1);
        let responses = vec![NodeResponse::success(
            "n1",
            vec![
                make_result("c1", 0.9, "n1"),
                make_result("c2", 0.8, "n1"),
                make_result("c3", 0.7, "n1"),
                make_result("c4", 0.6, "n1"),
            ],
            10,
        )];
        let results = sfs.merge_responses(&query, responses);
        assert_eq!(results.len(), 2);
        // top 2 must be the highest scoring
        assert_eq!(results[0].cid, "c1");
        assert_eq!(results[1].cid, "c2");
    }

    // -----------------------------------------------------------------------
    // 11. deduplicate
    // -----------------------------------------------------------------------

    #[test]
    fn test_deduplicate_keeps_highest_score() {
        let results = vec![
            FederatedResult::new("cid1", 0.9, vec!["n1".into()], 1),
            FederatedResult::new("cid1", 0.5, vec!["n2".into()], 1),
            FederatedResult::new("cid2", 0.7, vec!["n1".into()], 1),
        ];
        let deduped = SemanticFederatedSearch::deduplicate(&results);
        assert_eq!(deduped.len(), 2);
        let cid1 = deduped
            .iter()
            .find(|r| r.cid == "cid1")
            .expect("cid1 missing");
        assert!((cid1.merged_score - 0.9).abs() < 1e-12);
    }

    #[test]
    fn test_deduplicate_sorted_descending() {
        let results = vec![
            FederatedResult::new("cid2", 0.3, vec![], 1),
            FederatedResult::new("cid1", 0.9, vec![], 1),
            FederatedResult::new("cid3", 0.6, vec![], 1),
        ];
        let deduped = SemanticFederatedSearch::deduplicate(&results);
        assert_eq!(deduped.len(), 3);
        assert!(deduped[0].merged_score >= deduped[1].merged_score);
        assert!(deduped[1].merged_score >= deduped[2].merged_score);
    }

    #[test]
    fn test_deduplicate_empty() {
        let results: Vec<FederatedResult> = vec![];
        let deduped = SemanticFederatedSearch::deduplicate(&results);
        assert!(deduped.is_empty());
    }

    #[test]
    fn test_deduplicate_no_duplicates_unchanged_count() {
        let results = vec![
            FederatedResult::new("cid1", 0.9, vec![], 1),
            FederatedResult::new("cid2", 0.7, vec![], 1),
        ];
        let deduped = SemanticFederatedSearch::deduplicate(&results);
        assert_eq!(deduped.len(), 2);
    }

    // -----------------------------------------------------------------------
    // 12. Stats updates
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_query_count_increments() {
        let mut sfs = coordinator_with_nodes(
            MergeStrategy::SimpleUnion,
            vec![make_node("n1", 1.0, 0, 1.0)],
        );
        let query = make_query(10, 0.0, 1);
        sfs.merge_responses(
            &query,
            vec![NodeResponse::success(
                "n1",
                vec![make_result("cid1", 0.9, "n1")],
                50,
            )],
        );
        sfs.merge_responses(
            &query,
            vec![NodeResponse::success(
                "n1",
                vec![make_result("cid2", 0.8, "n1")],
                60,
            )],
        );
        assert_eq!(sfs.stats().queries, 2);
    }

    #[test]
    fn test_stats_zero_result_query() {
        let mut sfs = coordinator_with_nodes(
            MergeStrategy::SimpleUnion,
            vec![make_node("n1", 1.0, 0, 1.0)],
        );
        // Filter everything out with high min_score
        let query = make_query(10, 0.99, 1);
        sfs.merge_responses(
            &query,
            vec![NodeResponse::success(
                "n1",
                vec![make_result("cid1", 0.5, "n1")],
                10,
            )],
        );
        assert_eq!(sfs.stats().zero_result_queries, 1);
    }

    #[test]
    fn test_stats_avg_latency_ewma_seeded_on_first_query() {
        let mut sfs = coordinator_with_nodes(
            MergeStrategy::SimpleUnion,
            vec![make_node("n1", 1.0, 0, 1.0)],
        );
        let query = make_query(10, 0.0, 1);
        sfs.merge_responses(
            &query,
            vec![NodeResponse::success(
                "n1",
                vec![make_result("cid1", 0.9, "n1")],
                100,
            )],
        );
        // First query seeds avg_latency to the observed latency.
        assert!((sfs.stats().avg_latency_ms - 100.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 13. active_nodes / best_nodes
    // -----------------------------------------------------------------------

    #[test]
    fn test_active_nodes_excludes_zero_success_rate() {
        let sfs = coordinator_with_nodes(
            MergeStrategy::SimpleUnion,
            vec![
                make_node("active", 1.0, 0, 0.9),
                make_node("inactive", 1.0, 0, 0.0),
            ],
        );
        let active = sfs.active_nodes();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "active");
    }

    #[test]
    fn test_best_nodes_ordered_by_effective_weight() {
        let sfs = coordinator_with_nodes(
            MergeStrategy::SimpleUnion,
            vec![
                make_node("low", 1.0, 5000, 0.5),
                make_node("high", 2.0, 0, 1.0),
                make_node("mid", 1.0, 0, 0.8),
            ],
        );
        let best = sfs.best_nodes(3);
        // best[0] must have the highest effective_weight
        for i in 0..best.len() - 1 {
            assert!(best[i].effective_weight() >= best[i + 1].effective_weight());
        }
    }

    #[test]
    fn test_best_nodes_limited_by_n() {
        let nodes: Vec<RemoteNode> = (0..5)
            .map(|i| make_node(&format!("n{i}"), 1.0, 0, 1.0))
            .collect();
        let sfs = coordinator_with_nodes(MergeStrategy::SimpleUnion, nodes);
        assert_eq!(sfs.best_nodes(3).len(), 3);
    }

    #[test]
    fn test_best_nodes_n_larger_than_count() {
        let sfs = coordinator_with_nodes(
            MergeStrategy::SimpleUnion,
            vec![make_node("n1", 1.0, 0, 1.0)],
        );
        assert_eq!(sfs.best_nodes(100).len(), 1);
    }

    // -----------------------------------------------------------------------
    // 14. Appearance count and contributing nodes
    // -----------------------------------------------------------------------

    #[test]
    fn test_appearance_count_two_nodes() {
        let mut sfs = coordinator_with_nodes(
            MergeStrategy::SimpleUnion,
            vec![make_node("n1", 1.0, 0, 1.0), make_node("n2", 1.0, 0, 1.0)],
        );
        let query = make_query(10, 0.0, 1);
        let responses = vec![
            NodeResponse::success("n1", vec![make_result("cid1", 0.8, "n1")], 10),
            NodeResponse::success("n2", vec![make_result("cid1", 0.7, "n2")], 10),
        ];
        let results = sfs.merge_responses(&query, responses);
        assert_eq!(results[0].appearance_count, 2);
        assert_eq!(results[0].contributing_nodes.len(), 2);
    }

    // -----------------------------------------------------------------------
    // 15. Metadata is preserved in RemoteResult
    // -----------------------------------------------------------------------

    #[test]
    fn test_metadata_preserved() {
        let r = make_result_meta("cid1", 0.9, "n1", vec![("type", "doc"), ("lang", "en")]);
        assert_eq!(r.metadata.len(), 2);
        assert_eq!(r.metadata[0], ("type".to_string(), "doc".to_string()));
    }

    // -----------------------------------------------------------------------
    // 16. NodeResponse constructors
    // -----------------------------------------------------------------------

    #[test]
    fn test_node_response_success_constructor() {
        let r = NodeResponse::success("n1", vec![make_result("c1", 0.5, "n1")], 99);
        assert!(r.success);
        assert_eq!(r.latency_ms, 99);
        assert_eq!(r.results.len(), 1);
    }

    #[test]
    fn test_node_response_failure_constructor() {
        let r = NodeResponse::failure("n1", 500);
        assert!(!r.success);
        assert!(r.results.is_empty());
        assert_eq!(r.latency_ms, 500);
    }

    // -----------------------------------------------------------------------
    // 17. FederatedQuery builders
    // -----------------------------------------------------------------------

    #[test]
    fn test_federated_query_default_values() {
        let q = FederatedQuery::new(vec![0.1]);
        assert_eq!(q.top_k, 10);
        assert_eq!(q.min_score, 0.0);
        assert_eq!(q.required_quorum, 1);
    }

    #[test]
    fn test_federated_query_builders() {
        let q = FederatedQuery::new(vec![])
            .with_top_k(5)
            .with_min_score(0.3)
            .with_quorum(3);
        assert_eq!(q.top_k, 5);
        assert!((q.min_score - 0.3).abs() < 1e-12);
        assert_eq!(q.required_quorum, 3);
    }

    // -----------------------------------------------------------------------
    // 18. Empty node set
    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_with_no_registered_nodes_but_synthetic_responses() {
        // Responses can arrive without pre-registered nodes; scoring falls back to weight=1.
        let mut sfs = SemanticFederatedSearch::new(MergeStrategy::WeightedMerge);
        let query = make_query(10, 0.0, 1);
        let responses = vec![NodeResponse::success(
            "unknown_node",
            vec![make_result("cid1", 0.6, "unknown_node")],
            20,
        )];
        let results = sfs.merge_responses(&query, responses);
        assert_eq!(results.len(), 1);
    }

    // -----------------------------------------------------------------------
    // 19. Zero quorum requirement
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_quorum_accepts_all_failure_responses() {
        let mut sfs = coordinator_with_nodes(
            MergeStrategy::SimpleUnion,
            vec![make_node("n1", 1.0, 0, 1.0)],
        );
        // required_quorum=0 means even 0 successful nodes is fine
        let query = make_query(10, 0.0, 0);
        let responses = vec![NodeResponse::failure("n1", 10)];
        // 0 successes >= required_quorum 0 → quorum met, but no results to merge
        let results = sfs.merge_responses(&query, responses);
        assert!(results.is_empty());
        // partial_failures must NOT have been incremented
        assert_eq!(sfs.stats().partial_failures, 0);
    }

    // -----------------------------------------------------------------------
    // 20. Multiple queries update stats correctly
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_partial_failures_multiple() {
        let mut sfs = coordinator_with_nodes(
            MergeStrategy::SimpleUnion,
            vec![make_node("n1", 1.0, 0, 1.0)],
        );
        let query = make_query(10, 0.0, 2); // require 2 successes
        for _ in 0..3 {
            sfs.merge_responses(
                &query,
                vec![NodeResponse::success(
                    "n1",
                    vec![make_result("c1", 0.9, "n1")],
                    10,
                )],
            );
        }
        assert_eq!(sfs.stats().partial_failures, 3);
    }

    // -----------------------------------------------------------------------
    // 21. RankFusion respects ordering within a node
    // -----------------------------------------------------------------------

    #[test]
    fn test_rank_fusion_rank_order_correct() {
        // n1 returns [cidA@0.9, cidB@0.5, cidC@0.1]
        // rank(cidA)=1, rank(cidB)=2, rank(cidC)=3
        // cidA should have the highest RRF score from n1
        let mut sfs = coordinator_with_nodes(
            MergeStrategy::RankFusion,
            vec![make_node("n1", 1.0, 0, 1.0)],
        );
        let query = make_query(10, 0.0, 1);
        let responses = vec![NodeResponse::success(
            "n1",
            vec![
                make_result("cidC", 0.1, "n1"),
                make_result("cidA", 0.9, "n1"),
                make_result("cidB", 0.5, "n1"),
            ],
            10,
        )];
        let results = sfs.merge_responses(&query, responses);
        assert_eq!(results[0].cid, "cidA");
        assert!(results[0].merged_score > results[1].merged_score);
    }

    // -----------------------------------------------------------------------
    // 22. FederatedResult::new
    // -----------------------------------------------------------------------

    #[test]
    fn test_federated_result_new() {
        let r = FederatedResult::new("mycid", 0.42, vec!["n1".into(), "n2".into()], 2);
        assert_eq!(r.cid, "mycid");
        assert!((r.merged_score - 0.42).abs() < 1e-12);
        assert_eq!(r.contributing_nodes.len(), 2);
        assert_eq!(r.appearance_count, 2);
    }

    // -----------------------------------------------------------------------
    // 23. WeightedMerge with missing node registration
    // -----------------------------------------------------------------------

    #[test]
    fn test_weighted_merge_unregistered_node_defaults_to_weight_one() {
        let mut sfs = SemanticFederatedSearch::new(MergeStrategy::WeightedMerge);
        let query = make_query(10, 0.0, 1);
        let responses = vec![NodeResponse::success(
            "unregistered",
            vec![make_result("cid1", 0.8, "unregistered")],
            10,
        )];
        let results = sfs.merge_responses(&query, responses);
        assert_eq!(results.len(), 1);
        // With weight=1 fallback, weighted avg of a single entry = the score itself.
        assert!((results[0].merged_score - 0.8).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 24. Successful mixed with failed responses
    // -----------------------------------------------------------------------

    #[test]
    fn test_failed_responses_excluded_from_merge() {
        let mut sfs = coordinator_with_nodes(
            MergeStrategy::SimpleUnion,
            vec![make_node("n1", 1.0, 0, 1.0), make_node("n2", 1.0, 0, 1.0)],
        );
        let query = make_query(10, 0.0, 1);
        let responses = vec![
            NodeResponse::success("n1", vec![make_result("cid_good", 0.9, "n1")], 10),
            NodeResponse::failure("n2", 5000),
        ];
        let results = sfs.merge_responses(&query, responses);
        // Only n1 results should appear
        assert!(results
            .iter()
            .all(|r| r.contributing_nodes.contains(&"n1".to_string())));
        assert!(!results
            .iter()
            .any(|r| r.contributing_nodes.contains(&"n2".to_string())));
    }

    // -----------------------------------------------------------------------
    // 25. node_count after operations
    // -----------------------------------------------------------------------

    #[test]
    fn test_node_count_after_add_remove() {
        let mut sfs = SemanticFederatedSearch::new(MergeStrategy::SimpleUnion);
        sfs.register_node(make_node("n1", 1.0, 0, 1.0));
        sfs.register_node(make_node("n2", 1.0, 0, 1.0));
        assert_eq!(sfs.node_count(), 2);
        sfs.remove_node("n1");
        assert_eq!(sfs.node_count(), 1);
    }
}
