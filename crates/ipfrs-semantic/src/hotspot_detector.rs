//! Semantic Hotspot Detector
//!
//! Detects "hotspot" regions in an embedding space — clusters of frequently
//! queried vectors — enabling pre-warming, caching priority, and index
//! rebalancing signals.

// ── Cosine similarity helper ─────────────────────────────────────────────────

/// Compute cosine similarity between two vectors.
///
/// Returns `0.0` if either vector has zero magnitude.
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine_sim: dimension mismatch");
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let (mut dot, mut mag_a, mut mag_b) = (0.0_f32, 0.0_f32, 0.0_f32);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        mag_a += x * x;
        mag_b += y * y;
    }

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    dot / (mag_a.sqrt() * mag_b.sqrt())
}

// ── QueryHit ─────────────────────────────────────────────────────────────────

/// A single query event recorded in the embedding space.
#[derive(Clone, Debug)]
pub struct QueryHit {
    /// The query embedding vector.
    pub embedding: Vec<f32>,
    /// Unix timestamp (seconds) when the query was made.
    pub timestamp_secs: u64,
    /// Unique identifier for the query.
    pub query_id: u64,
}

// ── HotspotRegion ─────────────────────────────────────────────────────────────

/// A cluster of frequently queried vectors in embedding space.
#[derive(Clone, Debug)]
pub struct HotspotRegion {
    /// Centroid of the queries assigned to this region.
    pub center: Vec<f32>,
    /// Number of query hits accumulated by this region.
    pub hit_count: u64,
    /// Maximum distance (1.0 − cosine_sim) from the center to any member query.
    pub radius: f32,
    /// Unix timestamp (seconds) of the most recent hit.
    pub last_hit_secs: u64,
}

impl HotspotRegion {
    /// Returns `true` when the region has not been hit within `ttl_secs`.
    ///
    /// A region is stale when `last_hit_secs + ttl_secs < now_secs`.
    #[inline]
    pub fn is_stale(&self, now_secs: u64, ttl_secs: u64) -> bool {
        self.last_hit_secs.saturating_add(ttl_secs) < now_secs
    }
}

// ── HotspotConfig ─────────────────────────────────────────────────────────────

/// Configuration for [`SemanticHotspotDetector`].
#[derive(Clone, Debug)]
pub struct HotspotConfig {
    /// Cosine similarity threshold: a hit is merged into an existing region
    /// when `cosine_sim(hit, center) >= similarity_threshold`.
    pub similarity_threshold: f32,
    /// Minimum `hit_count` required for a region to appear in [`SemanticHotspotDetector::hotspots`].
    pub min_hits_to_report: u64,
    /// Maximum number of regions to maintain.  When exceeded the region with
    /// the lowest `hit_count` is evicted.
    pub max_regions: usize,
    /// Regions whose `last_hit_secs` is older than this many seconds are
    /// eligible for removal via [`SemanticHotspotDetector::evict_stale`].
    pub ttl_secs: u64,
}

impl Default for HotspotConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.85,
            min_hits_to_report: 3,
            max_regions: 50,
            ttl_secs: 3600,
        }
    }
}

// ── HotspotStats ──────────────────────────────────────────────────────────────

/// Aggregate statistics for the hotspot detector.
#[derive(Clone, Debug)]
pub struct HotspotStats {
    /// Total number of query hits recorded since creation.
    pub total_hits: u64,
    /// Current number of active (non-evicted) regions.
    pub active_regions: usize,
    /// `hit_count` of the hottest region, or `0` if there are no regions.
    pub hottest_region_hits: u64,
    /// Mean `hit_count` across all regions, or `0.0` if there are no regions.
    pub avg_hits_per_region: f64,
}

// ── SemanticHotspotDetector ───────────────────────────────────────────────────

/// Detects frequently queried regions in an embedding space.
///
/// Incoming [`QueryHit`]s are merged into existing [`HotspotRegion`]s when
/// the cosine similarity between the hit and the region's centroid exceeds
/// [`HotspotConfig::similarity_threshold`].  When no existing region matches a
/// new one is created.  Regions that exceed [`HotspotConfig::max_regions`] or
/// have not been hit recently are evicted.
pub struct SemanticHotspotDetector {
    /// Active hotspot regions.
    pub regions: Vec<HotspotRegion>,
    /// Detector configuration.
    pub config: HotspotConfig,
    /// Cumulative hit counter (never decremented on eviction).
    pub total_hits: u64,
}

impl SemanticHotspotDetector {
    /// Create a new detector with the given configuration.
    pub fn new(config: HotspotConfig) -> Self {
        Self {
            regions: Vec::new(),
            config,
            total_hits: 0,
        }
    }

    /// Record a new query hit.
    ///
    /// The hit is merged into the first region whose center has a cosine
    /// similarity ≥ `config.similarity_threshold`.  If no such region exists a
    /// new one is created.  When the number of regions exceeds `max_regions`
    /// the region with the lowest `hit_count` is evicted.
    pub fn record_hit(&mut self, hit: QueryHit) {
        // Find first matching region.
        let match_idx = self.regions.iter().position(|region| {
            cosine_sim(&hit.embedding, &region.center) >= self.config.similarity_threshold
        });

        if let Some(idx) = match_idx {
            let sim = cosine_sim(&hit.embedding, &self.regions[idx].center);
            let distance = 1.0_f32 - sim;
            let region = &mut self.regions[idx];
            region.hit_count += 1;
            if hit.timestamp_secs > region.last_hit_secs {
                region.last_hit_secs = hit.timestamp_secs;
            }
            if distance > region.radius {
                region.radius = distance;
            }
        } else {
            // Create a new region centred on this hit.
            self.regions.push(HotspotRegion {
                center: hit.embedding,
                hit_count: 1,
                radius: 0.0,
                last_hit_secs: hit.timestamp_secs,
            });

            // Enforce max_regions by evicting the coldest region.
            if self.regions.len() > self.config.max_regions {
                let coldest = self
                    .regions
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, r)| r.hit_count)
                    .map(|(i, _)| i);

                if let Some(evict_idx) = coldest {
                    self.regions.swap_remove(evict_idx);
                }
            }
        }

        self.total_hits += 1;
    }

    /// Return references to regions with `hit_count >= min_hits_to_report`,
    /// sorted by `hit_count` descending.
    pub fn hotspots(&self) -> Vec<&HotspotRegion> {
        let mut result: Vec<&HotspotRegion> = self
            .regions
            .iter()
            .filter(|r| r.hit_count >= self.config.min_hits_to_report)
            .collect();

        result.sort_by_key(|r| std::cmp::Reverse(r.hit_count));
        result
    }

    /// Remove regions that are stale relative to `now_secs`.
    pub fn evict_stale(&mut self, now_secs: u64) {
        self.regions
            .retain(|r| !r.is_stale(now_secs, self.config.ttl_secs));
    }

    /// Return aggregate statistics for the detector.
    pub fn stats(&self) -> HotspotStats {
        let active_regions = self.regions.len();

        let hottest_region_hits = self.regions.iter().map(|r| r.hit_count).max().unwrap_or(0);

        let avg_hits_per_region = if active_regions == 0 {
            0.0
        } else {
            let total: u64 = self.regions.iter().map(|r| r.hit_count).sum();
            total as f64 / active_regions as f64
        };

        HotspotStats {
            total_hits: self.total_hits,
            active_regions,
            hottest_region_hits,
            avg_hits_per_region,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Normalised unit vector in a given dimension (for predictable cosine sims).
    fn unit_vec(dim: usize, active: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; dim];
        if active < dim {
            v[active] = 1.0;
        }
        v
    }

    fn default_config() -> HotspotConfig {
        HotspotConfig::default()
    }

    fn make_hit(embedding: Vec<f32>, ts: u64, id: u64) -> QueryHit {
        QueryHit {
            embedding,
            timestamp_secs: ts,
            query_id: id,
        }
    }

    // 1. new() starts with no regions
    #[test]
    fn test_new_starts_empty() {
        let det = SemanticHotspotDetector::new(default_config());
        assert!(det.regions.is_empty());
        assert_eq!(det.total_hits, 0);
    }

    // 2. record_hit creates a region for a new embedding
    #[test]
    fn test_record_hit_creates_region() {
        let mut det = SemanticHotspotDetector::new(default_config());
        det.record_hit(make_hit(unit_vec(4, 0), 100, 1));
        assert_eq!(det.regions.len(), 1);
        assert_eq!(det.regions[0].hit_count, 1);
    }

    // 3. record_hit merges a similar hit into an existing region
    #[test]
    fn test_record_hit_merges_similar() {
        let mut det = SemanticHotspotDetector::new(default_config());
        // Both vectors identical → cosine_sim == 1.0 ≥ 0.85
        let emb = unit_vec(4, 0);
        det.record_hit(make_hit(emb.clone(), 100, 1));
        det.record_hit(make_hit(emb.clone(), 200, 2));
        assert_eq!(det.regions.len(), 1, "identical hits should merge");
    }

    // 4. record_hit increments hit_count correctly
    #[test]
    fn test_record_hit_increments_count() {
        let mut det = SemanticHotspotDetector::new(default_config());
        let emb = unit_vec(4, 0);
        for i in 0..5 {
            det.record_hit(make_hit(emb.clone(), 100 + i, i));
        }
        assert_eq!(det.regions[0].hit_count, 5);
    }

    // 5. record_hit updates last_hit_secs
    #[test]
    fn test_record_hit_updates_last_hit_secs() {
        let mut det = SemanticHotspotDetector::new(default_config());
        let emb = unit_vec(4, 0);
        det.record_hit(make_hit(emb.clone(), 100, 1));
        det.record_hit(make_hit(emb.clone(), 999, 2));
        assert_eq!(det.regions[0].last_hit_secs, 999);
    }

    // 6. record_hit updates radius (distance = 1.0 - sim)
    #[test]
    fn test_record_hit_updates_radius() {
        let mut det = SemanticHotspotDetector::new(HotspotConfig {
            similarity_threshold: 0.5,
            ..default_config()
        });
        // Create region with e0 = [1,0,0,0]
        let e0 = unit_vec(4, 0);
        det.record_hit(make_hit(e0.clone(), 100, 1));
        assert_eq!(det.regions[0].radius, 0.0);

        // A vector at 45° to e0: [1,1,0,0] / sqrt(2)
        let root2 = 2.0_f32.sqrt();
        let e45 = vec![1.0 / root2, 1.0 / root2, 0.0, 0.0];
        let sim = cosine_sim(&e45, &e0);
        let expected_radius = 1.0 - sim;

        det.record_hit(make_hit(e45, 200, 2));
        let actual_radius = det.regions[0].radius;
        assert!(
            (actual_radius - expected_radius).abs() < 1e-5,
            "radius={actual_radius} expected≈{expected_radius}"
        );
    }

    // 7. Different embedding creates a separate region
    #[test]
    fn test_different_embedding_new_region() {
        let mut det = SemanticHotspotDetector::new(default_config());
        // e0 and e1 are orthogonal → cosine_sim == 0.0 < 0.85
        det.record_hit(make_hit(unit_vec(4, 0), 100, 1));
        det.record_hit(make_hit(unit_vec(4, 1), 200, 2));
        assert_eq!(det.regions.len(), 2);
    }

    // 8. max_regions evicts the region with the lowest hit_count
    #[test]
    fn test_max_regions_evicts_coldest() {
        let mut det = SemanticHotspotDetector::new(HotspotConfig {
            max_regions: 3,
            ..default_config()
        });

        // Create 3 orthogonal regions and bump the first two counts.
        let e0 = unit_vec(8, 0);
        let e1 = unit_vec(8, 1);
        let e2 = unit_vec(8, 2);
        let e3 = unit_vec(8, 3);

        det.record_hit(make_hit(e0.clone(), 100, 1)); // region 0: hits=1
        det.record_hit(make_hit(e0.clone(), 110, 2)); // region 0: hits=2
        det.record_hit(make_hit(e0.clone(), 120, 3)); // region 0: hits=3

        det.record_hit(make_hit(e1.clone(), 200, 4)); // region 1: hits=1
        det.record_hit(make_hit(e1.clone(), 210, 5)); // region 1: hits=2

        det.record_hit(make_hit(e2.clone(), 300, 6)); // region 2: hits=1

        // At this point we have 3 regions (= max_regions).
        assert_eq!(det.regions.len(), 3);

        // Adding a 4th orthogonal vector triggers eviction of region with hit_count=1 (e2 or e3).
        det.record_hit(make_hit(e3.clone(), 400, 7));
        assert_eq!(
            det.regions.len(),
            3,
            "after eviction we should still have max_regions"
        );

        // The region with the highest hit_count (3) must still be present.
        let max_hits = det.regions.iter().map(|r| r.hit_count).max().unwrap_or(0);
        assert_eq!(max_hits, 3, "hottest region must survive eviction");
    }

    // 9. hotspots filters by min_hits_to_report
    #[test]
    fn test_hotspots_filters_by_min_hits() {
        let mut det = SemanticHotspotDetector::new(HotspotConfig {
            min_hits_to_report: 3,
            ..default_config()
        });
        let e0 = unit_vec(4, 0);
        let e1 = unit_vec(4, 1);

        // e0 gets 5 hits, e1 gets 2
        for i in 0..5u64 {
            det.record_hit(make_hit(e0.clone(), 100 + i, i));
        }
        for i in 0..2u64 {
            det.record_hit(make_hit(e1.clone(), 200 + i, 10 + i));
        }

        let hot = det.hotspots();
        assert_eq!(hot.len(), 1, "only regions with hit_count>=3 should appear");
        assert_eq!(hot[0].hit_count, 5);
    }

    // 10. hotspots sorted by hit_count descending
    #[test]
    fn test_hotspots_sorted_descending() {
        let mut det = SemanticHotspotDetector::new(HotspotConfig {
            min_hits_to_report: 1,
            ..default_config()
        });
        let vecs = (0..4).map(|i| unit_vec(8, i)).collect::<Vec<_>>();

        // Give them 4, 1, 3, 2 hits respectively.
        let hit_counts = [4u64, 1, 3, 2];
        for (i, &count) in hit_counts.iter().enumerate() {
            for j in 0..count {
                det.record_hit(make_hit(
                    vecs[i].clone(),
                    100 + j,
                    (i * 10 + j as usize) as u64,
                ));
            }
        }

        let hot = det.hotspots();
        let counts: Vec<u64> = hot.iter().map(|r| r.hit_count).collect();
        for window in counts.windows(2) {
            assert!(window[0] >= window[1], "hotspots must be sorted desc");
        }
    }

    // 11. evict_stale removes old regions
    #[test]
    fn test_evict_stale_removes_old() {
        let mut det = SemanticHotspotDetector::new(HotspotConfig {
            ttl_secs: 100,
            ..default_config()
        });
        det.record_hit(make_hit(unit_vec(4, 0), 500, 1));
        // Region last_hit=500; now=700 → 500+100=600 < 700 → stale
        det.evict_stale(700);
        assert!(det.regions.is_empty(), "stale region should be removed");
    }

    // 12. evict_stale keeps fresh regions
    #[test]
    fn test_evict_stale_keeps_fresh() {
        let mut det = SemanticHotspotDetector::new(HotspotConfig {
            ttl_secs: 1000,
            ..default_config()
        });
        det.record_hit(make_hit(unit_vec(4, 0), 500, 1));
        // last_hit=500; now=700 → 500+1000=1500 > 700 → fresh
        det.evict_stale(700);
        assert_eq!(det.regions.len(), 1, "fresh region should survive eviction");
    }

    // 13. is_stale returns false when within TTL
    #[test]
    fn test_is_stale_false_within_ttl() {
        let region = HotspotRegion {
            center: vec![1.0],
            hit_count: 1,
            radius: 0.0,
            last_hit_secs: 1000,
        };
        // last_hit(1000) + ttl(500) = 1500 >= now(1200) → not stale
        assert!(!region.is_stale(1200, 500));
    }

    // 14. is_stale returns true when past TTL
    #[test]
    fn test_is_stale_true_past_ttl() {
        let region = HotspotRegion {
            center: vec![1.0],
            hit_count: 1,
            radius: 0.0,
            last_hit_secs: 1000,
        };
        // last_hit(1000) + ttl(100) = 1100 < now(1500) → stale
        assert!(region.is_stale(1500, 100));
    }

    // 15. stats: total_hits accumulates correctly
    #[test]
    fn test_stats_total_hits() {
        let mut det = SemanticHotspotDetector::new(default_config());
        for i in 0..7u64 {
            // Alternate between two orthogonal vectors to stress merging path.
            let emb = unit_vec(4, (i % 2) as usize);
            det.record_hit(make_hit(emb, 100 + i, i));
        }
        assert_eq!(det.stats().total_hits, 7);
    }

    // 16. stats: active_regions count
    #[test]
    fn test_stats_active_regions() {
        let mut det = SemanticHotspotDetector::new(default_config());
        det.record_hit(make_hit(unit_vec(4, 0), 100, 1));
        det.record_hit(make_hit(unit_vec(4, 1), 200, 2));
        assert_eq!(det.stats().active_regions, 2);
    }

    // 17. stats: hottest_region_hits is 0 when no regions exist
    #[test]
    fn test_stats_hottest_region_hits_empty() {
        let det = SemanticHotspotDetector::new(default_config());
        assert_eq!(det.stats().hottest_region_hits, 0);
    }

    // 17b. stats: hottest_region_hits reflects the most-hit region
    #[test]
    fn test_stats_hottest_region_hits() {
        let mut det = SemanticHotspotDetector::new(default_config());
        let e0 = unit_vec(4, 0);
        let e1 = unit_vec(4, 1);
        for i in 0..4u64 {
            det.record_hit(make_hit(e0.clone(), 100 + i, i));
        }
        det.record_hit(make_hit(e1.clone(), 200, 99));
        assert_eq!(det.stats().hottest_region_hits, 4);
    }

    // 18. stats: avg_hits_per_region
    #[test]
    fn test_stats_avg_hits_per_region() {
        let mut det = SemanticHotspotDetector::new(default_config());
        let e0 = unit_vec(4, 0);
        let e1 = unit_vec(4, 1);

        // e0 → 3 hits, e1 → 1 hit: avg = (3+1)/2 = 2.0
        for i in 0..3u64 {
            det.record_hit(make_hit(e0.clone(), 100 + i, i));
        }
        det.record_hit(make_hit(e1.clone(), 200, 99));

        let stats = det.stats();
        assert_eq!(stats.active_regions, 2);
        assert!((stats.avg_hits_per_region - 2.0).abs() < 1e-9);
    }

    // 19. stats: avg_hits_per_region is 0.0 when no regions
    #[test]
    fn test_stats_avg_hits_empty() {
        let det = SemanticHotspotDetector::new(default_config());
        assert_eq!(det.stats().avg_hits_per_region, 0.0);
    }

    // 20. total_hits is not decremented after eviction
    #[test]
    fn test_total_hits_not_decremented_on_evict() {
        let mut det = SemanticHotspotDetector::new(HotspotConfig {
            ttl_secs: 10,
            ..default_config()
        });
        det.record_hit(make_hit(unit_vec(4, 0), 100, 1));
        det.record_hit(make_hit(unit_vec(4, 0), 105, 2));
        det.evict_stale(200); // evicts the region
        assert_eq!(det.total_hits, 2, "total_hits must not be decremented");
    }

    // 21. cosine_sim returns 0.0 for zero-magnitude vectors
    #[test]
    fn test_cosine_sim_zero_magnitude() {
        let zero = vec![0.0_f32, 0.0, 0.0];
        let unit = vec![1.0_f32, 0.0, 0.0];
        assert_eq!(cosine_sim(&zero, &unit), 0.0);
        assert_eq!(cosine_sim(&unit, &zero), 0.0);
        assert_eq!(cosine_sim(&zero, &zero), 0.0);
    }

    // 22. cosine_sim returns 1.0 for identical unit vectors
    #[test]
    fn test_cosine_sim_identical() {
        let v = vec![1.0_f32, 0.0, 0.0];
        assert!((cosine_sim(&v, &v) - 1.0).abs() < 1e-6);
    }
}
