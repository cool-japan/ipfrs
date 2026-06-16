//! Semantic Content Router
//!
//! Routes content queries to the most semantically relevant nodes/shards
//! based on their registered topic embeddings. Uses cosine similarity
//! with load-aware scoring to balance routing across available nodes.

/// A registered topic embedding for a specific node.
///
/// Each node may register multiple topics (one per `TopicEmbedding`).
/// The router uses these embeddings to determine which node best serves
/// a given query embedding.
#[derive(Debug, Clone)]
pub struct TopicEmbedding {
    /// Identifier for the node that owns this topic.
    pub node_id: String,
    /// Human-readable topic label (e.g., "machine-learning", "genomics").
    pub topic: String,
    /// The embedding vector representing this topic in latent space.
    pub embedding: Vec<f32>,
    /// Maximum number of content items this node can handle for this topic.
    pub capacity: usize,
    /// Current number of items stored/handled by this node for this topic.
    pub current_load: usize,
}

impl TopicEmbedding {
    /// Creates a new `TopicEmbedding`.
    pub fn new(
        node_id: impl Into<String>,
        topic: impl Into<String>,
        embedding: Vec<f32>,
        capacity: usize,
    ) -> Self {
        Self {
            node_id: node_id.into(),
            topic: topic.into(),
            embedding,
            capacity,
            current_load: 0,
        }
    }

    /// Returns the load factor: `current_load / capacity.max(1)`.
    pub fn load_factor(&self) -> f64 {
        self.current_load as f64 / self.capacity.max(1) as f64
    }
}

/// Cosine similarity between two vectors.
///
/// Returns `0.0` if either vector has zero magnitude.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

/// A candidate routing result for a single (node, topic) pair.
#[derive(Debug, Clone)]
pub struct RouteScore {
    /// Node identifier.
    pub node_id: String,
    /// Topic label matched.
    pub topic: String,
    /// Cosine similarity between the query and the topic embedding.
    pub similarity: f32,
    /// Load factor at the time the score was computed: `current_load / capacity.max(1)`.
    pub load_factor: f64,
}

impl RouteScore {
    /// Combined routing score that penalizes heavily loaded nodes.
    ///
    /// `combined_score = similarity * (1.0 - load_factor * 0.3)`
    pub fn combined_score(&self) -> f64 {
        self.similarity as f64 * (1.0 - self.load_factor * 0.3)
    }
}

/// The outcome of a single routing decision.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// The query embedding that triggered this decision.
    pub query_embedding: Vec<f32>,
    /// All candidates that passed the similarity threshold, sorted by
    /// `combined_score` descending. Limited to at most `max_candidates`.
    pub candidates: Vec<RouteScore>,
    /// The `node_id` of the top-scoring candidate, if any.
    pub best_node: Option<String>,
}

impl RoutingDecision {
    /// Returns the top-`k` node identifiers ordered by `combined_score` descending.
    ///
    /// If fewer than `k` candidates exist, all are returned.
    pub fn top_k_nodes(&self, k: usize) -> Vec<&str> {
        self.candidates
            .iter()
            .take(k)
            .map(|rs| rs.node_id.as_str())
            .collect()
    }
}

/// Configuration for [`SemanticContentRouter`].
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Minimum cosine similarity required to include a node as a candidate.
    ///
    /// Defaults to `0.6`.
    pub min_similarity: f32,
    /// Maximum number of candidates to keep in a [`RoutingDecision`].
    ///
    /// Defaults to `10`.
    pub max_candidates: usize,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            min_similarity: 0.6,
            max_candidates: 10,
        }
    }
}

/// Accumulated routing statistics.
#[derive(Debug, Clone, Default)]
pub struct RouterStats {
    /// Total number of routing decisions made (successful or not).
    pub total_routed: u64,
    /// Number of routing decisions that produced no candidates.
    pub no_route_count: u64,
}

impl RouterStats {
    /// Fraction of routing decisions that had at least one candidate.
    ///
    /// Returns `0.0` when no decisions have been made yet.
    pub fn route_success_rate(&self) -> f64 {
        if self.total_routed == 0 {
            return 0.0;
        }
        let successful = self.total_routed.saturating_sub(self.no_route_count);
        successful as f64 / self.total_routed as f64
    }
}

/// Routes content queries to the most semantically relevant nodes.
///
/// Maintains a registry of `TopicEmbedding`s and, for each query, computes
/// cosine similarity against every registered embedding, filters by
/// `min_similarity`, ranks by `combined_score`, and caps results at
/// `max_candidates`.
///
/// # Example
///
/// ```rust
/// use ipfrs_semantic::content_router::{
///     RouterConfig, SemanticContentRouter, TopicEmbedding,
/// };
///
/// let config = RouterConfig { min_similarity: 0.5, max_candidates: 5 };
/// let mut router = SemanticContentRouter::new(config);
///
/// let emb = TopicEmbedding::new("node-1", "science", vec![1.0, 0.0, 0.0], 100);
/// router.register_topic(emb);
///
/// let decision = router.route(&[0.9, 0.1, 0.0]);
/// assert!(decision.best_node.is_some());
/// ```
#[derive(Debug)]
pub struct SemanticContentRouter {
    /// All registered topic embeddings.
    topics: Vec<TopicEmbedding>,
    /// Router configuration.
    config: RouterConfig,
    /// Accumulated statistics.
    stats: RouterStats,
}

impl SemanticContentRouter {
    /// Creates a new router with the given configuration.
    pub fn new(config: RouterConfig) -> Self {
        Self {
            topics: Vec::new(),
            config,
            stats: RouterStats::default(),
        }
    }

    /// Registers a topic embedding.
    ///
    /// Multiple topics from the same node are allowed.
    pub fn register_topic(&mut self, topic: TopicEmbedding) {
        self.topics.push(topic);
    }

    /// Updates the current load for a specific (node, topic) pair.
    ///
    /// If no matching entry is found, this is a no-op.
    pub fn update_load(&mut self, node_id: &str, topic: &str, load: usize) {
        for t in &mut self.topics {
            if t.node_id == node_id && t.topic == topic {
                t.current_load = load;
            }
        }
    }

    /// Routes a query embedding to the most suitable nodes.
    ///
    /// Steps:
    /// 1. Compute cosine similarity to every registered topic embedding.
    /// 2. Filter candidates whose similarity is below `min_similarity`.
    /// 3. Sort remaining candidates by `combined_score` (descending).
    /// 4. Retain at most `max_candidates` results.
    /// 5. Update statistics and return the [`RoutingDecision`].
    pub fn route(&mut self, query_embedding: &[f32]) -> RoutingDecision {
        let mut candidates: Vec<RouteScore> = self
            .topics
            .iter()
            .filter_map(|t| {
                let sim = cosine_similarity(query_embedding, &t.embedding);
                if sim >= self.config.min_similarity {
                    Some(RouteScore {
                        node_id: t.node_id.clone(),
                        topic: t.topic.clone(),
                        similarity: sim,
                        load_factor: t.load_factor(),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by combined_score descending; break ties by similarity desc.
        candidates.sort_by(|a, b| {
            b.combined_score()
                .partial_cmp(&a.combined_score())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.similarity
                        .partial_cmp(&a.similarity)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        candidates.truncate(self.config.max_candidates);

        let best_node = candidates.first().map(|rs| rs.node_id.clone());

        self.stats.total_routed += 1;
        if candidates.is_empty() {
            self.stats.no_route_count += 1;
        }

        RoutingDecision {
            query_embedding: query_embedding.to_vec(),
            candidates,
            best_node,
        }
    }

    /// Removes all registered topic embeddings for the given node.
    pub fn unregister_node(&mut self, node_id: &str) {
        self.topics.retain(|t| t.node_id != node_id);
    }

    /// Returns a reference to the accumulated routing statistics.
    pub fn stats(&self) -> &RouterStats {
        &self.stats
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_router(min_sim: f32, max_cand: usize) -> SemanticContentRouter {
        SemanticContentRouter::new(RouterConfig {
            min_similarity: min_sim,
            max_candidates: max_cand,
        })
    }

    fn unit_vec(index: usize, dim: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        if index < dim {
            v[index] = 1.0;
        }
        v
    }

    // ── 1. register_topic stores the embedding ────────────────────────────────
    #[test]
    fn test_register_topic_stores_entry() {
        let mut router = make_router(0.5, 10);
        let emb = TopicEmbedding::new("node-a", "science", unit_vec(0, 4), 100);
        router.register_topic(emb);
        assert_eq!(router.topics.len(), 1);
        assert_eq!(router.topics[0].node_id, "node-a");
        assert_eq!(router.topics[0].topic, "science");
    }

    // ── 2. register_topic allows multiple topics per node ─────────────────────
    #[test]
    fn test_register_multiple_topics_same_node() {
        let mut router = make_router(0.5, 10);
        router.register_topic(TopicEmbedding::new("node-a", "science", unit_vec(0, 4), 50));
        router.register_topic(TopicEmbedding::new("node-a", "arts", unit_vec(1, 4), 50));
        assert_eq!(router.topics.len(), 2);
    }

    // ── 3. route finds best matching node ────────────────────────────────────
    #[test]
    fn test_route_finds_best_match() {
        let mut router = make_router(0.5, 10);
        router.register_topic(TopicEmbedding::new(
            "node-a",
            "topic-a",
            unit_vec(0, 3),
            100,
        ));
        router.register_topic(TopicEmbedding::new(
            "node-b",
            "topic-b",
            unit_vec(1, 3),
            100,
        ));
        // Query is closest to node-a (same direction as dimension 0)
        let query = vec![0.95f32, 0.05, 0.0];
        let decision = router.route(&query);
        assert_eq!(decision.best_node.as_deref(), Some("node-a"));
    }

    // ── 4. similarity below threshold is filtered out ─────────────────────────
    #[test]
    fn test_similarity_below_threshold_filtered() {
        let mut router = make_router(0.9, 10);
        // node-a has dim-0 embedding; query points mostly in dim-1 → low similarity
        router.register_topic(TopicEmbedding::new("node-a", "t1", unit_vec(0, 4), 100));
        let query = unit_vec(1, 4); // orthogonal → similarity = 0.0
        let decision = router.route(&query);
        assert!(decision.candidates.is_empty());
        assert!(decision.best_node.is_none());
    }

    // ── 5. load_factor penalizes loaded nodes ─────────────────────────────────
    #[test]
    fn test_load_factor_penalizes() {
        let mut router = make_router(0.5, 10);
        // Both nodes have similar (non-identical) embeddings to the query.
        let q: Vec<f32> = vec![1.0, 0.0];
        let mut heavy = TopicEmbedding::new("heavy", "t", vec![1.0, 0.0], 100);
        heavy.current_load = 90; // 90 % load
        let light = TopicEmbedding::new("light", "t", vec![1.0, 0.0], 100);
        // light.current_load = 0 (default)
        router.register_topic(heavy);
        router.register_topic(light);

        let decision = router.route(&q);
        // light node should score higher (no load penalty)
        assert_eq!(decision.best_node.as_deref(), Some("light"));
    }

    // ── 6. top_k_nodes returns correct slice ──────────────────────────────────
    #[test]
    fn test_top_k_nodes() {
        let mut router = make_router(0.0, 10);
        router.register_topic(TopicEmbedding::new("node-a", "t1", unit_vec(0, 3), 100));
        router.register_topic(TopicEmbedding::new("node-b", "t2", unit_vec(1, 3), 100));
        router.register_topic(TopicEmbedding::new("node-c", "t3", unit_vec(2, 3), 100));
        // Query aligned with dim-0 ⇒ node-a highest
        let query: Vec<f32> = vec![1.0, 0.0, 0.0];
        let decision = router.route(&query);
        let top2 = decision.top_k_nodes(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0], "node-a");
    }

    // ── 7. top_k_nodes with k > candidates ────────────────────────────────────
    #[test]
    fn test_top_k_nodes_fewer_than_k() {
        let mut router = make_router(0.5, 10);
        router.register_topic(TopicEmbedding::new("node-a", "t1", unit_vec(0, 3), 100));
        let query: Vec<f32> = vec![1.0, 0.0, 0.0];
        let decision = router.route(&query);
        let top5 = decision.top_k_nodes(5);
        assert_eq!(top5.len(), 1); // only 1 candidate
    }

    // ── 8. update_load changes the score ─────────────────────────────────────
    #[test]
    fn test_update_load_changes_score() {
        let mut router = make_router(0.5, 10);
        router.register_topic(TopicEmbedding::new("node-a", "t1", unit_vec(0, 2), 100));

        let query = unit_vec(0, 2);
        let before = router.route(&query);
        let score_before = before.candidates[0].combined_score();

        // Push load to 80 %
        router.update_load("node-a", "t1", 80);

        let after = router.route(&query);
        let score_after = after.candidates[0].combined_score();

        assert!(
            score_after < score_before,
            "score_after={score_after} should be less than score_before={score_before}"
        );
    }

    // ── 9. update_load only touches matching (node, topic) ────────────────────
    #[test]
    fn test_update_load_targets_correct_entry() {
        let mut router = make_router(0.0, 10);
        router.register_topic(TopicEmbedding::new("node-a", "t1", unit_vec(0, 2), 100));
        router.register_topic(TopicEmbedding::new("node-a", "t2", unit_vec(1, 2), 100));

        router.update_load("node-a", "t1", 50);

        assert_eq!(router.topics[0].current_load, 50);
        assert_eq!(router.topics[1].current_load, 0); // unchanged
    }

    // ── 10. unregister_node removes all topics for that node ──────────────────
    #[test]
    fn test_unregister_node_removes() {
        let mut router = make_router(0.5, 10);
        router.register_topic(TopicEmbedding::new("node-a", "t1", unit_vec(0, 3), 100));
        router.register_topic(TopicEmbedding::new("node-a", "t2", unit_vec(1, 3), 100));
        router.register_topic(TopicEmbedding::new("node-b", "t3", unit_vec(2, 3), 100));

        router.unregister_node("node-a");

        assert_eq!(router.topics.len(), 1);
        assert_eq!(router.topics[0].node_id, "node-b");
    }

    // ── 11. unregister_node does nothing for unknown node ─────────────────────
    #[test]
    fn test_unregister_unknown_node_no_op() {
        let mut router = make_router(0.5, 10);
        router.register_topic(TopicEmbedding::new("node-a", "t1", unit_vec(0, 3), 100));
        router.unregister_node("node-x");
        assert_eq!(router.topics.len(), 1);
    }

    // ── 12. no_route_count increments on empty result ─────────────────────────
    #[test]
    fn test_no_route_count_when_empty() {
        let mut router = make_router(0.99, 10);
        // No topics registered ⇒ always empty
        router.route(&[0.1, 0.2]);
        router.route(&[0.3, 0.4]);
        assert_eq!(router.stats().no_route_count, 2);
        assert_eq!(router.stats().total_routed, 2);
    }

    // ── 13. no_route_count does NOT increment on successful route ─────────────
    #[test]
    fn test_no_route_count_not_incremented_on_success() {
        let mut router = make_router(0.5, 10);
        router.register_topic(TopicEmbedding::new("node-a", "t1", unit_vec(0, 2), 100));
        router.route(&unit_vec(0, 2));
        assert_eq!(router.stats().no_route_count, 0);
        assert_eq!(router.stats().total_routed, 1);
    }

    // ── 14. route_success_rate is 0.0 with no decisions ──────────────────────
    #[test]
    fn test_route_success_rate_no_decisions() {
        let router = make_router(0.5, 10);
        assert_eq!(router.stats().route_success_rate(), 0.0);
    }

    // ── 15. route_success_rate correct after mixed decisions ─────────────────
    #[test]
    fn test_route_success_rate_mixed() {
        let mut router = make_router(0.5, 10);
        router.register_topic(TopicEmbedding::new("node-a", "t1", unit_vec(0, 2), 100));
        // Succeeds
        router.route(&unit_vec(0, 2));
        // Fails (orthogonal, similarity = 0)
        router.route(&unit_vec(1, 2));

        let rate = router.stats().route_success_rate();
        assert!((rate - 0.5).abs() < 1e-9, "expected 0.5, got {rate}");
    }

    // ── 16. combined_score formula ────────────────────────────────────────────
    #[test]
    fn test_combined_score_formula() {
        let rs = RouteScore {
            node_id: "n".to_string(),
            topic: "t".to_string(),
            similarity: 0.8,
            load_factor: 0.5,
        };
        // 0.8 * (1.0 - 0.5 * 0.3) = 0.8 * 0.85 = 0.68
        // The similarity field is f32 (0.8f32), so cast to f64 first to avoid
        // precision mismatch when comparing against a pure-f64 expected value.
        let expected = 0.8_f32 as f64 * (1.0_f64 - 0.5_f64 * 0.3_f64);
        assert!((rs.combined_score() - expected).abs() < 1e-9);
    }

    // ── 17. best_node set correctly ──────────────────────────────────────────
    #[test]
    fn test_best_node_set_correctly() {
        let mut router = make_router(0.0, 10);
        router.register_topic(TopicEmbedding::new("node-a", "t1", unit_vec(0, 3), 100));
        router.register_topic(TopicEmbedding::new("node-b", "t2", unit_vec(1, 3), 100));
        // Query in dim-1 ⇒ node-b best
        let decision = router.route(&[0.0, 1.0, 0.0]);
        assert_eq!(decision.best_node.as_deref(), Some("node-b"));
    }

    // ── 18. max_candidates caps results ──────────────────────────────────────
    #[test]
    fn test_max_candidates_caps_results() {
        let mut router = make_router(0.0, 3);
        for i in 0..6 {
            let mut emb = vec![0.0f32; 6];
            emb[i] = 1.0;
            router.register_topic(TopicEmbedding::new(format!("node-{i}"), "t", emb, 100));
        }
        // All embeddings have similarity 0 to query [1,1,1,1,1,1]/√6 but let's use
        // a query that gives non-zero similarity to all.
        let query = vec![1.0f32; 6];
        let decision = router.route(&query);
        assert!(decision.candidates.len() <= 3);
    }

    // ── 19. cosine similarity of identical vectors equals 1.0 ────────────────
    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0f32, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    // ── 20. cosine similarity of orthogonal vectors equals 0.0 ───────────────
    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    // ── 21. load_factor is 0.0 when no load ──────────────────────────────────
    #[test]
    fn test_load_factor_zero() {
        let t = TopicEmbedding::new("n", "t", vec![1.0], 100);
        assert_eq!(t.load_factor(), 0.0);
    }

    // ── 22. load_factor with zero capacity uses max(1) ───────────────────────
    #[test]
    fn test_load_factor_zero_capacity() {
        let mut t = TopicEmbedding::new("n", "t", vec![1.0], 0);
        t.current_load = 5;
        // capacity.max(1) = 1, so load_factor = 5.0
        assert!((t.load_factor() - 5.0).abs() < 1e-9);
    }

    // ── 23. candidates sorted by combined_score desc ──────────────────────────
    #[test]
    fn test_candidates_sorted_desc() {
        let mut router = make_router(0.0, 10);
        // All nodes are equally similar to the query; vary load to create order.
        let q = vec![1.0f32, 0.0];
        for pct in [80usize, 0, 50, 30] {
            let mut t = TopicEmbedding::new(format!("node-{pct}"), "t", vec![1.0, 0.0], 100);
            t.current_load = pct;
            router.register_topic(t);
        }
        let decision = router.route(&q);
        for window in decision.candidates.windows(2) {
            assert!(
                window[0].combined_score() >= window[1].combined_score(),
                "candidates not sorted: {} < {}",
                window[0].combined_score(),
                window[1].combined_score()
            );
        }
    }
}
