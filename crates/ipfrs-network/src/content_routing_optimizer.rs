//! Content Routing Optimizer
//!
//! Optimizes content routing decisions by maintaining a cost model for fetching
//! blocks from different peers, caching routing decisions, and detecting when
//! cached routes have become stale.

use std::collections::HashMap;

/// Cost model for fetching content from a specific peer.
#[derive(Debug, Clone)]
pub struct RouteCost {
    /// Peer identifier
    pub peer_id: String,
    /// Round-trip latency in milliseconds
    pub latency_ms: f64,
    /// Available bandwidth in bytes per second
    pub bandwidth_bps: u64,
    /// Peer reliability score in range [0.0, 1.0]
    pub reliability: f64,
}

impl RouteCost {
    /// Estimated fetch cost in milliseconds for the given payload size.
    ///
    /// Formula: `(size_bytes / bandwidth_bps) * 1000 + latency_ms`
    pub fn fetch_cost(&self, size_bytes: u64) -> f64 {
        let transfer_ms = size_bytes as f64 / (self.bandwidth_bps.max(1) as f64) * 1000.0;
        transfer_ms + self.latency_ms
    }

    /// Weighted quality score — higher is better.
    ///
    /// Formula: `reliability / fetch_cost(65536).max(1e-9)`
    pub fn weighted_score(&self) -> f64 {
        self.reliability / self.fetch_cost(65536).max(1e-9)
    }
}

/// A cached routing entry for a single CID.
#[derive(Debug, Clone)]
pub struct RouteEntry {
    /// Content identifier this entry is for
    pub cid: String,
    /// Candidate peers sorted by `weighted_score` descending
    pub candidates: Vec<RouteCost>,
    /// Unix timestamp (seconds) at which this entry was last updated
    pub cached_at_secs: u64,
    /// Number of times this entry has been served from cache
    pub hit_count: u64,
}

impl RouteEntry {
    /// Returns a reference to the best (lowest-cost, highest-reliability) peer,
    /// or `None` if there are no candidates.
    pub fn best_peer(&self) -> Option<&RouteCost> {
        self.candidates.first()
    }

    /// Returns `true` when the entry has exceeded its time-to-live.
    pub fn is_stale(&self, ttl_secs: u64, now_secs: u64) -> bool {
        now_secs.saturating_sub(self.cached_at_secs) >= ttl_secs
    }
}

/// Aggregate statistics for the routing optimizer.
#[derive(Debug, Clone, Default)]
pub struct RoutingStats {
    /// Number of lookups that were served from cache
    pub cache_hits: u64,
    /// Number of lookups that were not in cache (or were stale)
    pub cache_misses: u64,
    /// Cumulative number of stale entries that have been evicted
    pub stale_routes: u64,
    /// Current number of entries held in the cache
    pub total_routes: usize,
}

impl RoutingStats {
    /// Fraction of total lookups that were cache hits.
    ///
    /// Returns `0.0` when no lookups have been performed yet.
    pub fn hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

/// Configuration knobs for the [`ContentRoutingOptimizer`].
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Maximum age of a cached route before it is considered stale (seconds)
    pub route_ttl_secs: u64,
    /// Maximum number of candidate peers stored per CID
    pub max_candidates_per_cid: usize,
    /// Minimum reliability score for a peer to be admitted as a candidate
    pub min_reliability: f64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            route_ttl_secs: 300,
            max_candidates_per_cid: 5,
            min_reliability: 0.3,
        }
    }
}

/// Content routing optimizer that caches and manages peer routing decisions.
pub struct ContentRoutingOptimizer {
    /// Routing table keyed by CID string
    routes: HashMap<String, RouteEntry>,
    /// Optimizer configuration
    config: OptimizerConfig,
    /// Running statistics
    stats: RoutingStats,
}

impl ContentRoutingOptimizer {
    /// Creates a new optimizer with the supplied configuration.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            routes: HashMap::new(),
            config,
            stats: RoutingStats::default(),
        }
    }

    /// Adds (or replaces) a routing entry for `cid`.
    ///
    /// Candidates that do not meet the `min_reliability` threshold are
    /// discarded before insertion. The survivors are sorted by
    /// [`RouteCost::weighted_score`] descending and truncated to
    /// `max_candidates_per_cid`.
    pub fn add_route(&mut self, cid: String, candidates: Vec<RouteCost>, now_secs: u64) {
        let mut filtered: Vec<RouteCost> = candidates
            .into_iter()
            .filter(|c| c.reliability >= self.config.min_reliability)
            .collect();

        filtered.sort_by(|a, b| {
            b.weighted_score()
                .partial_cmp(&a.weighted_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        filtered.truncate(self.config.max_candidates_per_cid);

        let entry = RouteEntry {
            cid: cid.clone(),
            candidates: filtered,
            cached_at_secs: now_secs,
            hit_count: 0,
        };

        self.routes.insert(cid, entry);
        self.stats.total_routes = self.routes.len();
    }

    /// Returns the best peer for `cid`, updating cache statistics.
    ///
    /// Returns `None` when:
    /// - The entry does not exist (cache miss).
    /// - The entry is stale — the entry is removed and counted as a stale
    ///   eviction plus a cache miss.
    pub fn best_route(&mut self, cid: &str, now_secs: u64) -> Option<&RouteCost> {
        let ttl = self.config.route_ttl_secs;

        if let Some(entry) = self.routes.get(cid) {
            if entry.is_stale(ttl, now_secs) {
                self.routes.remove(cid);
                self.stats.stale_routes += 1;
                self.stats.cache_misses += 1;
                self.stats.total_routes = self.routes.len();
                return None;
            }
        } else {
            self.stats.cache_misses += 1;
            return None;
        }

        // Entry is present and fresh — record hit.
        let entry = self.routes.get_mut(cid)?;
        self.stats.cache_hits += 1;
        entry.hit_count += 1;

        // Re-borrow immutably to return a reference with the right lifetime.
        self.routes.get(cid).and_then(|e| e.best_peer())
    }

    /// Evicts all stale entries from the cache.
    ///
    /// Returns the number of entries that were removed.
    pub fn evict_stale(&mut self, now_secs: u64) -> usize {
        let ttl = self.config.route_ttl_secs;
        let before = self.routes.len();
        self.routes
            .retain(|_, entry| !entry.is_stale(ttl, now_secs));
        let removed = before - self.routes.len();
        self.stats.stale_routes += removed as u64;
        self.stats.total_routes = self.routes.len();
        removed
    }

    /// Updates the latency and reliability for `peer_id` across **all**
    /// cached routing entries, then re-sorts each affected entry's candidates.
    pub fn update_peer_cost(&mut self, peer_id: &str, new_latency_ms: f64, new_reliability: f64) {
        for entry in self.routes.values_mut() {
            let mut changed = false;
            for candidate in &mut entry.candidates {
                if candidate.peer_id == peer_id {
                    candidate.latency_ms = new_latency_ms;
                    candidate.reliability = new_reliability;
                    changed = true;
                }
            }
            if changed {
                entry.candidates.sort_by(|a, b| {
                    b.weighted_score()
                        .partial_cmp(&a.weighted_score())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
    }

    /// Returns a reference to the current routing statistics.
    pub fn stats(&self) -> &RoutingStats {
        &self.stats
    }

    /// Returns the number of entries currently held in the route cache.
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_peer(
        peer_id: &str,
        latency_ms: f64,
        bandwidth_bps: u64,
        reliability: f64,
    ) -> RouteCost {
        RouteCost {
            peer_id: peer_id.to_string(),
            latency_ms,
            bandwidth_bps,
            reliability,
        }
    }

    fn default_optimizer() -> ContentRoutingOptimizer {
        ContentRoutingOptimizer::new(OptimizerConfig::default())
    }

    // ── 1. new() starts empty ─────────────────────────────────────────────────
    #[test]
    fn test_new_empty() {
        let opt = default_optimizer();
        assert_eq!(opt.route_count(), 0);
        assert_eq!(opt.stats().cache_hits, 0);
        assert_eq!(opt.stats().cache_misses, 0);
    }

    // ── 2. fetch_cost calculation ─────────────────────────────────────────────
    #[test]
    fn test_fetch_cost_calculation() {
        let peer = make_peer("p1", 10.0, 1_000_000, 1.0); // 1 MB/s
                                                          // 65536 bytes @ 1 MB/s = 65.536 ms + 10 ms = 75.536 ms
        let expected = 65536.0 / 1_000_000.0 * 1000.0 + 10.0;
        assert!((peer.fetch_cost(65536) - expected).abs() < 1e-9);
    }

    // ── 3. weighted_score higher for reliable/fast peer ──────────────────────
    #[test]
    fn test_weighted_score_fast_beats_slow() {
        let fast = make_peer("fast", 5.0, 10_000_000, 0.9);
        let slow = make_peer("slow", 200.0, 100_000, 0.9);
        assert!(fast.weighted_score() > slow.weighted_score());
    }

    // ── 4. add_route: filters by min_reliability ─────────────────────────────
    #[test]
    fn test_add_route_filters_by_min_reliability() {
        let mut opt = ContentRoutingOptimizer::new(OptimizerConfig {
            min_reliability: 0.5,
            ..Default::default()
        });
        let candidates = vec![
            make_peer("good", 10.0, 1_000_000, 0.8),
            make_peer("bad", 10.0, 1_000_000, 0.2),
        ];
        opt.add_route("cid1".to_string(), candidates, 1000);
        let entry = opt.routes.get("cid1").expect("entry must exist");
        assert_eq!(entry.candidates.len(), 1);
        assert_eq!(entry.candidates[0].peer_id, "good");
    }

    // ── 5. add_route: sorts by weighted_score descending ─────────────────────
    #[test]
    fn test_add_route_sorts_by_weighted_score() {
        let mut opt = default_optimizer();
        let candidates = vec![
            make_peer("slow", 500.0, 100_000, 0.9),
            make_peer("fast", 5.0, 10_000_000, 0.9),
        ];
        opt.add_route("cid1".to_string(), candidates, 1000);
        let entry = opt.routes.get("cid1").expect("entry must exist");
        assert_eq!(entry.candidates[0].peer_id, "fast");
    }

    // ── 6. add_route: truncates to max_candidates ─────────────────────────────
    #[test]
    fn test_add_route_truncates_to_max_candidates() {
        let mut opt = ContentRoutingOptimizer::new(OptimizerConfig {
            max_candidates_per_cid: 3,
            ..Default::default()
        });
        let candidates: Vec<RouteCost> = (0..10)
            .map(|i| make_peer(&format!("p{i}"), 10.0, 1_000_000, 0.9))
            .collect();
        opt.add_route("cid1".to_string(), candidates, 1000);
        assert_eq!(opt.routes["cid1"].candidates.len(), 3);
    }

    // ── 7. best_route: cache hit increments stats ─────────────────────────────
    #[test]
    fn test_best_route_cache_hit_increments_stats() {
        let mut opt = default_optimizer();
        opt.add_route(
            "cid1".to_string(),
            vec![make_peer("p1", 10.0, 1_000_000, 0.9)],
            1000,
        );
        let result = opt.best_route("cid1", 1000);
        assert!(result.is_some());
        assert_eq!(opt.stats().cache_hits, 1);
        assert_eq!(opt.stats().cache_misses, 0);
        assert_eq!(opt.routes["cid1"].hit_count, 1);
    }

    // ── 8. best_route: stale entry is removed ────────────────────────────────
    #[test]
    fn test_best_route_stale_entry_removed() {
        let mut opt = ContentRoutingOptimizer::new(OptimizerConfig {
            route_ttl_secs: 60,
            ..Default::default()
        });
        opt.add_route(
            "cid1".to_string(),
            vec![make_peer("p1", 10.0, 1_000_000, 0.9)],
            0,
        );
        // Query 120 seconds later — beyond TTL
        let result = opt.best_route("cid1", 120);
        assert!(result.is_none());
        assert_eq!(opt.stats().stale_routes, 1);
        assert_eq!(opt.stats().cache_misses, 1);
        assert_eq!(opt.route_count(), 0);
    }

    // ── 9. best_route: not found increments misses ───────────────────────────
    #[test]
    fn test_best_route_not_found_increments_misses() {
        let mut opt = default_optimizer();
        let result = opt.best_route("nonexistent", 1000);
        assert!(result.is_none());
        assert_eq!(opt.stats().cache_misses, 1);
        assert_eq!(opt.stats().cache_hits, 0);
    }

    // ── 10. evict_stale removes expired entries ───────────────────────────────
    #[test]
    fn test_evict_stale_removes_expired_entries() {
        let mut opt = ContentRoutingOptimizer::new(OptimizerConfig {
            route_ttl_secs: 60,
            ..Default::default()
        });
        opt.add_route(
            "fresh".to_string(),
            vec![make_peer("p1", 5.0, 1_000_000, 0.9)],
            100,
        );
        opt.add_route(
            "stale".to_string(),
            vec![make_peer("p2", 5.0, 1_000_000, 0.9)],
            0,
        );

        let removed = opt.evict_stale(120);
        assert_eq!(removed, 1);
        assert!(opt.routes.contains_key("fresh"));
        assert!(!opt.routes.contains_key("stale"));
    }

    // ── 11. evict_stale returns correct count ─────────────────────────────────
    #[test]
    fn test_evict_stale_returns_count() {
        let mut opt = ContentRoutingOptimizer::new(OptimizerConfig {
            route_ttl_secs: 60,
            ..Default::default()
        });
        for i in 0..5_u64 {
            opt.add_route(
                format!("cid{i}"),
                vec![make_peer("p", 5.0, 1_000_000, 0.9)],
                0,
            );
        }
        let removed = opt.evict_stale(120);
        assert_eq!(removed, 5);
        assert_eq!(opt.route_count(), 0);
    }

    // ── 12. update_peer_cost updates all affected routes ─────────────────────
    #[test]
    fn test_update_peer_cost_updates_all_routes() {
        let mut opt = default_optimizer();
        opt.add_route(
            "cid1".to_string(),
            vec![make_peer("target", 100.0, 500_000, 0.8)],
            0,
        );
        opt.add_route(
            "cid2".to_string(),
            vec![make_peer("target", 100.0, 500_000, 0.8)],
            0,
        );

        opt.update_peer_cost("target", 10.0, 0.95);

        for entry in opt.routes.values() {
            let c = entry
                .candidates
                .iter()
                .find(|c| c.peer_id == "target")
                .expect("should exist");
            assert!((c.latency_ms - 10.0).abs() < 1e-9);
            assert!((c.reliability - 0.95).abs() < 1e-9);
        }
    }

    // ── 13. update_peer_cost re-sorts candidates ──────────────────────────────
    #[test]
    fn test_update_peer_cost_resorts_candidates() {
        let mut opt = default_optimizer();
        // Same bandwidth so latency/reliability dominate the score.
        // p_poor starts with high latency (above min_reliability floor); p_good is currently first.
        let candidates = vec![
            make_peer("p_poor", 100.0, 10_000_000, 0.4),
            make_peer("p_good", 5.0, 10_000_000, 0.9),
        ];
        opt.add_route("cid1".to_string(), candidates, 0);

        // p_good is first; now dramatically boost p_poor: latency=1ms, reliability=1.0
        opt.update_peer_cost("p_poor", 1.0, 1.0);

        // After update the re-sort should place p_poor first
        let entry = opt.routes.get("cid1").expect("entry must exist");
        assert_eq!(entry.candidates[0].peer_id, "p_poor");
    }

    // ── 14. is_stale: fresh entry ─────────────────────────────────────────────
    #[test]
    fn test_is_stale_fresh() {
        let entry = RouteEntry {
            cid: "c".to_string(),
            candidates: vec![],
            cached_at_secs: 1000,
            hit_count: 0,
        };
        assert!(!entry.is_stale(300, 1200)); // 200s elapsed < 300s TTL
    }

    // ── 15. is_stale: stale entry ─────────────────────────────────────────────
    #[test]
    fn test_is_stale_stale() {
        let entry = RouteEntry {
            cid: "c".to_string(),
            candidates: vec![],
            cached_at_secs: 1000,
            hit_count: 0,
        };
        assert!(entry.is_stale(300, 1400)); // 400s elapsed >= 300s TTL
    }

    // ── 16. hit_rate calculation ──────────────────────────────────────────────
    #[test]
    fn test_hit_rate_calculation() {
        let stats = RoutingStats {
            cache_hits: 3,
            cache_misses: 1,
            ..Default::default()
        };
        assert!((stats.hit_rate() - 0.75).abs() < 1e-9);
    }

    // ── 16b. hit_rate zero when no lookups ───────────────────────────────────
    #[test]
    fn test_hit_rate_zero_no_lookups() {
        let stats = RoutingStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    // ── 17. best_peer returns first candidate ─────────────────────────────────
    #[test]
    fn test_best_peer_first_candidate() {
        let entry = RouteEntry {
            cid: "c".to_string(),
            candidates: vec![
                make_peer("p_best", 5.0, 10_000_000, 0.9),
                make_peer("p_second", 50.0, 1_000_000, 0.8),
            ],
            cached_at_secs: 0,
            hit_count: 0,
        };
        assert_eq!(
            entry.best_peer().map(|p| p.peer_id.as_str()),
            Some("p_best")
        );
    }

    // ── 18. route_count is correct ────────────────────────────────────────────
    #[test]
    fn test_route_count_correct() {
        let mut opt = default_optimizer();
        assert_eq!(opt.route_count(), 0);
        opt.add_route(
            "c1".to_string(),
            vec![make_peer("p", 5.0, 1_000_000, 0.9)],
            0,
        );
        assert_eq!(opt.route_count(), 1);
        opt.add_route(
            "c2".to_string(),
            vec![make_peer("p", 5.0, 1_000_000, 0.9)],
            0,
        );
        assert_eq!(opt.route_count(), 2);
    }
}
