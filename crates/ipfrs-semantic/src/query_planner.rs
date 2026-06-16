//! # NearestNeighborQueryPlanner
//!
//! Plans and optimizes k-NN queries over sharded HNSW indexes, choosing between
//! local-only, remote-fanout, and hybrid execution strategies.

/// Execution strategy for a k-NN query plan.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionStrategy {
    /// Query served entirely from the local shard.
    LocalOnly,
    /// Fan out to the listed remote peers.
    RemoteFanout { peer_ids: Vec<String> },
    /// Query both local shard and remote peers.
    Hybrid { local: bool, peer_ids: Vec<String> },
    /// Result is available in the similarity cache.
    Cached { cache_key: u64 },
}

/// Metadata describing a single shard.
#[derive(Debug, Clone)]
pub struct ShardInfo {
    /// Unique shard identifier.
    pub shard_id: String,
    /// Peer that owns this shard. Use `"local"` for the local peer.
    pub peer_id: String,
    /// Number of vectors stored in this shard.
    pub vector_count: u64,
    /// Embedding dimensionality.
    pub dimension: usize,
    /// Expected round-trip latency in milliseconds.
    pub estimated_latency_ms: f64,
    /// Whether this shard resides on the local node.
    pub is_local: bool,
}

/// A fully-resolved query execution plan.
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// FNV-1a hash of the query vector bytes (used as a stable query identifier).
    pub query_id: u64,
    /// Number of nearest neighbours requested.
    pub k: usize,
    /// Chosen execution strategy.
    pub strategy: ExecutionStrategy,
    /// Shards that will be queried under this plan.
    pub shards: Vec<ShardInfo>,
    /// Maximum latency across all selected shards (0.0 if no shards).
    pub estimated_latency_ms: f64,
    /// Expected number of candidate results before final merge.
    pub estimated_results: usize,
}

impl QueryPlan {
    /// Returns `true` when the plan executes entirely on the local node.
    pub fn is_local_only(&self) -> bool {
        matches!(self.strategy, ExecutionStrategy::LocalOnly)
    }
}

/// Configuration for the [`NearestNeighborQueryPlanner`].
#[derive(Debug, Clone)]
pub struct PlannerConfig {
    /// Maximum number of shards / peers to fan out to. Default: `8`.
    pub max_fanout: usize,
    /// Shards whose `estimated_latency_ms` exceeds this value are excluded.
    /// Default: `100.0`.
    pub latency_budget_ms: f64,
    /// Shards with fewer vectors than this threshold are excluded. Default: `100`.
    pub min_vectors_per_shard: u64,
    /// When `true`, local shards are sorted to the front of the candidate list.
    /// Default: `true`.
    pub prefer_local: bool,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            max_fanout: 8,
            latency_budget_ms: 100.0,
            min_vectors_per_shard: 100,
            prefer_local: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute an FNV-1a 64-bit hash over `f32` values by hashing their
/// little-endian byte representation.
fn fnv1a_hash_f32_slice(values: &[f32]) -> u64 {
    const OFFSET_BASIS: u64 = 2_166_136_261_u64;
    const PRIME: u64 = 16_777_619_u64;

    let mut hash = OFFSET_BASIS;
    for &v in values {
        for byte in v.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(PRIME);
        }
    }
    hash
}

// ---------------------------------------------------------------------------
// NearestNeighborQueryPlanner
// ---------------------------------------------------------------------------

/// Plans k-NN queries over a heterogeneous set of HNSW shards.
pub struct NearestNeighborQueryPlanner {
    /// Planner configuration.
    pub config: PlannerConfig,
}

impl NearestNeighborQueryPlanner {
    /// Create a new planner with the given configuration.
    pub fn new(config: PlannerConfig) -> Self {
        Self { config }
    }

    /// Produce an optimised [`QueryPlan`] for the given query vector and `k`.
    ///
    /// # Algorithm
    ///
    /// 1. Compute a stable `query_id` via FNV-1a over the query bytes.
    /// 2. Filter shards by latency budget and minimum vector count.
    /// 3. Optionally sort local shards to the front.
    /// 4. Limit to `max_fanout` shards.
    /// 5. Choose the execution strategy based on the mix of local/remote shards.
    pub fn plan(&self, query_vec: &[f32], k: usize, shards: &[ShardInfo]) -> QueryPlan {
        let query_id = fnv1a_hash_f32_slice(query_vec);

        // --- Step 1: filter ---
        let mut candidates: Vec<ShardInfo> = shards
            .iter()
            .filter(|s| {
                s.estimated_latency_ms <= self.config.latency_budget_ms
                    && s.vector_count >= self.config.min_vectors_per_shard
            })
            .cloned()
            .collect();

        // --- Step 2: sort ---
        if self.config.prefer_local {
            // local shards first, then ascending latency
            candidates.sort_by(|a, b| match (a.is_local, b.is_local) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a
                    .estimated_latency_ms
                    .partial_cmp(&b.estimated_latency_ms)
                    .unwrap_or(std::cmp::Ordering::Equal),
            });
        } else {
            candidates.sort_by(|a, b| {
                a.estimated_latency_ms
                    .partial_cmp(&b.estimated_latency_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // --- Step 3: limit ---
        candidates.truncate(self.config.max_fanout);

        // --- Step 4: strategy ---
        let has_local = candidates.iter().any(|s| s.is_local);
        let remote_peer_ids: Vec<String> = candidates
            .iter()
            .filter(|s| !s.is_local)
            .map(|s| s.peer_id.clone())
            .collect();

        let strategy = if candidates.is_empty() || (has_local && remote_peer_ids.is_empty()) {
            ExecutionStrategy::LocalOnly
        } else if !has_local {
            ExecutionStrategy::RemoteFanout {
                peer_ids: remote_peer_ids,
            }
        } else {
            ExecutionStrategy::Hybrid {
                local: true,
                peer_ids: remote_peer_ids,
            }
        };

        // --- Step 5: derived metrics ---
        let estimated_latency_ms = candidates
            .iter()
            .map(|s| s.estimated_latency_ms)
            .fold(0.0_f64, f64::max);

        let total_vectors: u64 = candidates.iter().map(|s| s.vector_count).sum();
        let upper = (k * candidates.len().max(1)) as u64;
        let raw = upper.min(total_vectors);
        let estimated_results = (raw as usize).max(k.min(total_vectors as usize));

        QueryPlan {
            query_id,
            k,
            strategy,
            shards: candidates,
            estimated_latency_ms,
            estimated_results,
        }
    }

    /// Return a human-readable description of the plan.
    pub fn explain(&self, plan: &QueryPlan) -> String {
        let strategy_desc = match &plan.strategy {
            ExecutionStrategy::LocalOnly => "LocalOnly".to_string(),
            ExecutionStrategy::RemoteFanout { peer_ids } => {
                format!("RemoteFanout(peers={})", peer_ids.join(", "))
            }
            ExecutionStrategy::Hybrid { local, peer_ids } => {
                format!("Hybrid(local={}, peers={})", local, peer_ids.join(", "))
            }
            ExecutionStrategy::Cached { cache_key } => {
                format!("Cached(key={cache_key:#x})")
            }
        };

        format!(
            "QueryPlan {{ id={:#x}, k={}, strategy={}, shards={}, \
             est_latency={:.2}ms, est_results={} }}",
            plan.query_id,
            plan.k,
            strategy_desc,
            plan.shards.len(),
            plan.estimated_latency_ms,
            plan.estimated_results,
        )
    }

    /// Produce a revised plan after `failed_peer` could not be reached.
    ///
    /// All shards owned by `failed_peer` are removed from the original plan's
    /// shard list, and the strategy is recomputed from the survivors.
    pub fn replan_on_failure(&self, plan: &QueryPlan, failed_peer: &str) -> QueryPlan {
        let surviving: Vec<ShardInfo> = plan
            .shards
            .iter()
            .filter(|s| s.peer_id != failed_peer)
            .cloned()
            .collect();

        let has_local = surviving.iter().any(|s| s.is_local);
        let remote_peer_ids: Vec<String> = surviving
            .iter()
            .filter(|s| !s.is_local)
            .map(|s| s.peer_id.clone())
            .collect();

        let strategy = if surviving.is_empty() || (has_local && remote_peer_ids.is_empty()) {
            ExecutionStrategy::LocalOnly
        } else if !has_local {
            ExecutionStrategy::RemoteFanout {
                peer_ids: remote_peer_ids,
            }
        } else {
            ExecutionStrategy::Hybrid {
                local: true,
                peer_ids: remote_peer_ids,
            }
        };

        let estimated_latency_ms = surviving
            .iter()
            .map(|s| s.estimated_latency_ms)
            .fold(0.0_f64, f64::max);

        let total_vectors: u64 = surviving.iter().map(|s| s.vector_count).sum();
        let upper = (plan.k * surviving.len().max(1)) as u64;
        let raw = upper.min(total_vectors);
        let estimated_results = (raw as usize).max(plan.k.min(total_vectors as usize));

        QueryPlan {
            query_id: plan.query_id,
            k: plan.k,
            strategy,
            shards: surviving,
            estimated_latency_ms,
            estimated_results,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_local_shard(id: &str, vectors: u64, latency: f64) -> ShardInfo {
        ShardInfo {
            shard_id: id.to_string(),
            peer_id: "local".to_string(),
            vector_count: vectors,
            dimension: 128,
            estimated_latency_ms: latency,
            is_local: true,
        }
    }

    fn make_remote_shard(id: &str, peer: &str, vectors: u64, latency: f64) -> ShardInfo {
        ShardInfo {
            shard_id: id.to_string(),
            peer_id: peer.to_string(),
            vector_count: vectors,
            dimension: 128,
            estimated_latency_ms: latency,
            is_local: false,
        }
    }

    fn default_planner() -> NearestNeighborQueryPlanner {
        NearestNeighborQueryPlanner::new(PlannerConfig::default())
    }

    fn query_vec() -> Vec<f32> {
        vec![0.1, 0.2, 0.3, 0.4]
    }

    // 1. Empty shards → LocalOnly with no shards
    #[test]
    fn test_plan_empty_shards_local_only() {
        let planner = default_planner();
        let plan = planner.plan(&query_vec(), 5, &[]);
        assert!(matches!(plan.strategy, ExecutionStrategy::LocalOnly));
        assert!(plan.shards.is_empty());
    }

    // 2. Single local shard → LocalOnly
    #[test]
    fn test_plan_single_local_shard() {
        let planner = default_planner();
        let shards = vec![make_local_shard("s0", 500, 5.0)];
        let plan = planner.plan(&query_vec(), 5, &shards);
        assert!(matches!(plan.strategy, ExecutionStrategy::LocalOnly));
        assert_eq!(plan.shards.len(), 1);
    }

    // 3. Single remote shard → RemoteFanout
    #[test]
    fn test_plan_single_remote_shard() {
        let planner = default_planner();
        let shards = vec![make_remote_shard("s1", "peer-A", 500, 20.0)];
        let plan = planner.plan(&query_vec(), 5, &shards);
        assert!(
            matches!(&plan.strategy, ExecutionStrategy::RemoteFanout { peer_ids } if peer_ids == &["peer-A"])
        );
    }

    // 4. Mixed shards → Hybrid
    #[test]
    fn test_plan_mixed_shards_hybrid() {
        let planner = default_planner();
        let shards = vec![
            make_local_shard("s0", 500, 5.0),
            make_remote_shard("s1", "peer-B", 500, 30.0),
        ];
        let plan = planner.plan(&query_vec(), 5, &shards);
        assert!(matches!(
            &plan.strategy,
            ExecutionStrategy::Hybrid { local: true, .. }
        ));
    }

    // 5. Latency budget excludes slow shards
    #[test]
    fn test_plan_respects_latency_budget() {
        let config = PlannerConfig {
            latency_budget_ms: 50.0,
            ..PlannerConfig::default()
        };
        let planner = NearestNeighborQueryPlanner::new(config);
        let shards = vec![
            make_local_shard("s0", 500, 40.0),
            make_remote_shard("s1", "peer-C", 500, 80.0), // too slow
        ];
        let plan = planner.plan(&query_vec(), 5, &shards);
        assert_eq!(plan.shards.len(), 1);
        assert!(plan.shards[0].is_local);
    }

    // 6. min_vectors_per_shard excludes sparse shards
    #[test]
    fn test_plan_respects_min_vectors() {
        let config = PlannerConfig {
            min_vectors_per_shard: 200,
            ..PlannerConfig::default()
        };
        let planner = NearestNeighborQueryPlanner::new(config);
        let shards = vec![
            make_local_shard("s0", 500, 5.0),
            make_remote_shard("s1", "peer-D", 50, 10.0), // too sparse
        ];
        let plan = planner.plan(&query_vec(), 5, &shards);
        assert_eq!(plan.shards.len(), 1);
        assert!(plan.shards[0].is_local);
    }

    // 7. max_fanout limits selected shards
    #[test]
    fn test_plan_respects_max_fanout() {
        let config = PlannerConfig {
            max_fanout: 2,
            ..PlannerConfig::default()
        };
        let planner = NearestNeighborQueryPlanner::new(config);
        let shards: Vec<ShardInfo> = (0..5)
            .map(|i| {
                make_remote_shard(&format!("s{i}"), &format!("peer-{i}"), 500, 10.0 + i as f64)
            })
            .collect();
        let plan = planner.plan(&query_vec(), 5, &shards);
        assert_eq!(plan.shards.len(), 2);
    }

    // 8. prefer_local puts local shard first
    #[test]
    fn test_plan_prefer_local_first() {
        let planner = default_planner();
        let shards = vec![
            make_remote_shard("s1", "peer-E", 500, 5.0), // lower latency but remote
            make_local_shard("s0", 500, 20.0),
        ];
        let plan = planner.plan(&query_vec(), 5, &shards);
        assert!(plan.shards[0].is_local, "local shard should be first");
    }

    // 9. query_id is deterministic for same vector
    #[test]
    fn test_query_id_deterministic() {
        let planner = default_planner();
        let v = vec![1.0_f32, 2.0, 3.0];
        let p1 = planner.plan(&v, 5, &[]);
        let p2 = planner.plan(&v, 5, &[]);
        assert_eq!(p1.query_id, p2.query_id);
    }

    // 10. query_id differs for different vectors
    #[test]
    fn test_query_id_differs_for_different_vectors() {
        let planner = default_planner();
        let p1 = planner.plan(&[1.0_f32, 0.0], 5, &[]);
        let p2 = planner.plan(&[0.0_f32, 1.0], 5, &[]);
        assert_ne!(p1.query_id, p2.query_id);
    }

    // 11. estimated_latency_ms is the max across selected shards
    #[test]
    fn test_estimated_latency_is_max() {
        let planner = default_planner();
        let shards = vec![
            make_local_shard("s0", 500, 10.0),
            make_remote_shard("s1", "peer-F", 500, 45.0),
            make_remote_shard("s2", "peer-G", 500, 30.0),
        ];
        let plan = planner.plan(&query_vec(), 5, &shards);
        assert!((plan.estimated_latency_ms - 45.0).abs() < 1e-9);
    }

    // 12. is_local_only() true/false
    #[test]
    fn test_is_local_only_flag() {
        let planner = default_planner();

        let local_shards = vec![make_local_shard("s0", 500, 5.0)];
        let local_plan = planner.plan(&query_vec(), 5, &local_shards);
        assert!(local_plan.is_local_only());

        let remote_shards = vec![make_remote_shard("s1", "peer-H", 500, 10.0)];
        let remote_plan = planner.plan(&query_vec(), 5, &remote_shards);
        assert!(!remote_plan.is_local_only());
    }

    // 13. explain() returns a non-empty string
    #[test]
    fn test_explain_non_empty() {
        let planner = default_planner();
        let shards = vec![make_local_shard("s0", 500, 5.0)];
        let plan = planner.plan(&query_vec(), 5, &shards);
        let explanation = planner.explain(&plan);
        assert!(!explanation.is_empty());
        assert!(explanation.contains("QueryPlan"));
    }

    // 14. replan_on_failure removes the failed peer's shards
    #[test]
    fn test_replan_removes_failed_peer() {
        let planner = default_planner();
        let shards = vec![
            make_local_shard("s0", 500, 5.0),
            make_remote_shard("s1", "peer-X", 500, 20.0),
            make_remote_shard("s2", "peer-Y", 500, 25.0),
        ];
        let plan = planner.plan(&query_vec(), 5, &shards);
        let new_plan = planner.replan_on_failure(&plan, "peer-X");
        assert!(new_plan.shards.iter().all(|s| s.peer_id != "peer-X"));
        assert_eq!(new_plan.shards.len(), 2);
    }

    // 15. replan_on_failure updates strategy (remote-only → LocalOnly after removing remote)
    #[test]
    fn test_replan_updates_strategy() {
        let planner = default_planner();
        let shards = vec![
            make_local_shard("s0", 500, 5.0),
            make_remote_shard("s1", "peer-Z", 500, 20.0),
        ];
        let plan = planner.plan(&query_vec(), 5, &shards);
        // Initially Hybrid; after removing peer-Z only local remains → LocalOnly
        let new_plan = planner.replan_on_failure(&plan, "peer-Z");
        assert!(matches!(new_plan.strategy, ExecutionStrategy::LocalOnly));
    }

    // 16. estimated_results clamped to k minimum (when total_vectors < k)
    #[test]
    fn test_estimated_results_clamped_to_k_minimum() {
        let planner = default_planner();
        let shards = vec![make_local_shard("s0", 200, 5.0)];
        // k=10, total_vectors=200 → raw = min(10*1, 200) = 10 ≥ k already
        // Use k=300 to force clamping: min(300*1, 200)=200, max(200, min(300,200))=200
        let plan = planner.plan(&query_vec(), 300, &shards);
        assert!(plan.estimated_results >= plan.k.min(200));
    }

    // Bonus: All-filtered scenario returns LocalOnly with empty shards
    #[test]
    fn test_all_filtered_returns_local_only_empty() {
        let config = PlannerConfig {
            latency_budget_ms: 1.0, // everything is too slow
            ..PlannerConfig::default()
        };
        let planner = NearestNeighborQueryPlanner::new(config);
        let shards = vec![
            make_local_shard("s0", 500, 50.0),
            make_remote_shard("s1", "peer-Q", 500, 80.0),
        ];
        let plan = planner.plan(&query_vec(), 5, &shards);
        assert!(matches!(plan.strategy, ExecutionStrategy::LocalOnly));
        assert!(plan.shards.is_empty());
    }
}
