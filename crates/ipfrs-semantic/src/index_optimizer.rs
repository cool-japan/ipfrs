//! Embedding Index Optimizer
//!
//! Analyzes HNSW index structure and recommends parameter tuning
//! (ef_construction, M, level distribution) to optimize search
//! quality vs. speed trade-offs.

/// Goal for HNSW index tuning.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IndexTuningGoal {
    /// Optimize for highest recall (larger M, ef_construction).
    MaxRecall,
    /// Optimize for fastest search (smaller parameters).
    MaxSpeed,
    /// Middle ground between recall and speed.
    Balanced,
}

/// HNSW index parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct HnswParams {
    /// Max connections per node.
    pub m: usize,
    /// Search width during index build.
    pub ef_construction: usize,
    /// Search width during query.
    pub ef_search: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
        }
    }
}

impl HnswParams {
    /// Returns `true` when the parameter combination is logically valid.
    pub fn is_valid(&self) -> bool {
        self.m >= 2 && self.ef_construction >= self.m && self.ef_search >= 1
    }
}

/// Distribution of nodes across HNSW levels.
///
/// Index 0 corresponds to the base (densest) layer.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelDistribution {
    /// Node count per level; index 0 = base layer.
    pub levels: Vec<usize>,
}

impl LevelDistribution {
    /// Total number of nodes across all levels.
    pub fn total_nodes(&self) -> usize {
        self.levels.iter().sum()
    }

    /// Highest level index (0-based).
    pub fn max_level(&self) -> usize {
        self.levels.len().saturating_sub(1)
    }

    /// Loose structural sanity check: each upper layer should have no more
    /// than roughly `2 * level[i] / m` nodes.
    pub fn is_well_formed(&self, m: usize) -> bool {
        let divisor = m.max(1);
        for i in 0..self.levels.len().saturating_sub(1) {
            let upper_bound = self.levels[i] / divisor * 2;
            if self.levels[i + 1] > upper_bound {
                return false;
            }
        }
        true
    }
}

/// Report produced by [`EmbeddingIndexOptimizer::recommend_params`].
#[derive(Clone, Debug)]
pub struct OptimizationReport {
    /// Parameters currently in use.
    pub current_params: HnswParams,
    /// Recommended parameters after analysis.
    pub recommended_params: HnswParams,
    /// Tuning goal driving the recommendation.
    pub goal: IndexTuningGoal,
    /// Estimated recall delta (positive = better recall).
    pub expected_recall_change: f64,
    /// Estimated latency delta (negative = faster).
    pub expected_speed_change: f64,
    /// Human-readable observations and warnings.
    pub notes: Vec<String>,
}

/// Analyzes an HNSW index and recommends parameter adjustments.
pub struct EmbeddingIndexOptimizer;

impl EmbeddingIndexOptimizer {
    /// Creates a new optimizer instance.
    pub fn new() -> Self {
        Self
    }

    /// Recommends HNSW parameters given the current configuration, the
    /// desired tuning goal, and the approximate number of indexed nodes.
    pub fn recommend_params(
        &self,
        current: HnswParams,
        goal: IndexTuningGoal,
        node_count: usize,
    ) -> OptimizationReport {
        let mut notes = Vec::new();

        if node_count > 100_000 {
            notes.push(format!(
                "Large index detected ({node_count} nodes): consider monitoring memory usage \
                 and build time when increasing M or ef_construction."
            ));
        }

        let (recommended_params, expected_recall_change, expected_speed_change) = match goal {
            IndexTuningGoal::MaxRecall => {
                let new_m = (current.m * 2).min(64);
                let new_ef_construction = (current.ef_construction * 2).min(800);
                let new_ef_search = (current.ef_search * 2).min(500);
                (
                    HnswParams {
                        m: new_m,
                        ef_construction: new_ef_construction,
                        ef_search: new_ef_search,
                    },
                    0.05_f64,
                    -0.30_f64,
                )
            }
            IndexTuningGoal::MaxSpeed => {
                let new_m = (current.m / 2).max(4);
                let new_ef_construction = (current.ef_construction / 2).max(current.m);
                let new_ef_search = (current.ef_search / 2).max(1);
                (
                    HnswParams {
                        m: new_m,
                        ef_construction: new_ef_construction,
                        ef_search: new_ef_search,
                    },
                    -0.08_f64,
                    0.50_f64,
                )
            }
            IndexTuningGoal::Balanced => {
                let new_m = ((current.m + 16) / 2).clamp(8, 32);
                (
                    HnswParams {
                        m: new_m,
                        ef_construction: 200,
                        ef_search: 50,
                    },
                    0.0_f64,
                    0.0_f64,
                )
            }
        };

        OptimizationReport {
            current_params: current,
            recommended_params,
            goal,
            expected_recall_change,
            expected_speed_change,
            notes,
        }
    }

    /// Returns a list of textual observations about the level distribution.
    pub fn analyze_levels(&self, dist: &LevelDistribution, params: &HnswParams) -> Vec<String> {
        let mut observations = Vec::new();

        observations.push(format!(
            "Total nodes across all levels: {}",
            dist.total_nodes()
        ));
        observations.push(format!("Maximum level index: {}", dist.max_level()));

        if dist.is_well_formed(params.m) {
            observations.push(
                "Level distribution is well-formed (each upper layer is within expected bounds)."
                    .to_string(),
            );
        } else {
            observations.push(
                "Level distribution is NOT well-formed: some upper layers exceed expected node \
                 counts. Consider rebuilding the index."
                    .to_string(),
            );
        }

        observations
    }

    /// Rough estimate of the memory footprint in megabytes.
    ///
    /// Calculation: `node_count × m × 8 bytes` per connection.
    pub fn estimate_memory_mb(&self, node_count: usize, params: &HnswParams) -> f64 {
        let bytes = node_count * params.m * 8;
        bytes as f64 / (1024.0 * 1024.0)
    }
}

impl Default for EmbeddingIndexOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // HnswParams::is_valid
    // ------------------------------------------------------------------

    #[test]
    fn test_is_valid_default_params() {
        let p = HnswParams::default();
        assert!(p.is_valid());
    }

    #[test]
    fn test_is_valid_m_less_than_2() {
        let p = HnswParams {
            m: 1,
            ef_construction: 10,
            ef_search: 1,
        };
        assert!(!p.is_valid());
    }

    #[test]
    fn test_is_valid_ef_construction_less_than_m() {
        let p = HnswParams {
            m: 16,
            ef_construction: 8,
            ef_search: 1,
        };
        assert!(!p.is_valid());
    }

    #[test]
    fn test_is_valid_ef_search_zero() {
        let p = HnswParams {
            m: 16,
            ef_construction: 200,
            ef_search: 0,
        };
        assert!(!p.is_valid());
    }

    #[test]
    fn test_is_valid_minimal_valid() {
        let p = HnswParams {
            m: 2,
            ef_construction: 2,
            ef_search: 1,
        };
        assert!(p.is_valid());
    }

    // ------------------------------------------------------------------
    // LevelDistribution
    // ------------------------------------------------------------------

    #[test]
    fn test_level_distribution_total_nodes() {
        let dist = LevelDistribution {
            levels: vec![1000, 100, 10, 1],
        };
        assert_eq!(dist.total_nodes(), 1111);
    }

    #[test]
    fn test_level_distribution_max_level() {
        let dist = LevelDistribution {
            levels: vec![1000, 100, 10],
        };
        assert_eq!(dist.max_level(), 2);
    }

    #[test]
    fn test_level_distribution_max_level_single() {
        let dist = LevelDistribution { levels: vec![500] };
        assert_eq!(dist.max_level(), 0);
    }

    #[test]
    fn test_level_distribution_max_level_empty() {
        let dist = LevelDistribution { levels: vec![] };
        assert_eq!(dist.max_level(), 0);
    }

    #[test]
    fn test_is_well_formed_true() {
        // With m=16, each upper layer should be ≤ lower / 16 * 2.
        // 1000 / 16 * 2 = 125, so 100 is fine.
        // 100 / 16 * 2 = 12, so 10 is fine.
        let dist = LevelDistribution {
            levels: vec![1000, 100, 10],
        };
        assert!(dist.is_well_formed(16));
    }

    #[test]
    fn test_is_well_formed_false() {
        // level[1] = 900 > 1000 / 16 * 2 = 125
        let dist = LevelDistribution {
            levels: vec![1000, 900, 10],
        };
        assert!(!dist.is_well_formed(16));
    }

    // ------------------------------------------------------------------
    // MaxRecall goal
    // ------------------------------------------------------------------

    #[test]
    fn test_max_recall_doubles_m() {
        let opt = EmbeddingIndexOptimizer::new();
        let current = HnswParams {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
        };
        let report = opt.recommend_params(current, IndexTuningGoal::MaxRecall, 1000);
        assert_eq!(report.recommended_params.m, 32);
    }

    #[test]
    fn test_max_recall_doubles_ef_construction() {
        let opt = EmbeddingIndexOptimizer::new();
        let current = HnswParams {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
        };
        let report = opt.recommend_params(current, IndexTuningGoal::MaxRecall, 1000);
        assert_eq!(report.recommended_params.ef_construction, 400);
    }

    #[test]
    fn test_max_recall_doubles_ef_search() {
        let opt = EmbeddingIndexOptimizer::new();
        let current = HnswParams {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
        };
        let report = opt.recommend_params(current, IndexTuningGoal::MaxRecall, 1000);
        assert_eq!(report.recommended_params.ef_search, 100);
    }

    #[test]
    fn test_max_recall_caps_m_at_64() {
        let opt = EmbeddingIndexOptimizer::new();
        let current = HnswParams {
            m: 48,
            ef_construction: 400,
            ef_search: 50,
        };
        let report = opt.recommend_params(current, IndexTuningGoal::MaxRecall, 1000);
        assert_eq!(report.recommended_params.m, 64);
    }

    #[test]
    fn test_max_recall_positive_recall_change() {
        let opt = EmbeddingIndexOptimizer::new();
        let current = HnswParams::default();
        let report = opt.recommend_params(current, IndexTuningGoal::MaxRecall, 500);
        assert!(report.expected_recall_change > 0.0);
    }

    #[test]
    fn test_max_recall_negative_speed_change() {
        let opt = EmbeddingIndexOptimizer::new();
        let current = HnswParams::default();
        let report = opt.recommend_params(current, IndexTuningGoal::MaxRecall, 500);
        assert!(report.expected_speed_change < 0.0);
    }

    // ------------------------------------------------------------------
    // MaxSpeed goal
    // ------------------------------------------------------------------

    #[test]
    fn test_max_speed_halves_m() {
        let opt = EmbeddingIndexOptimizer::new();
        let current = HnswParams {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
        };
        let report = opt.recommend_params(current, IndexTuningGoal::MaxSpeed, 1000);
        assert_eq!(report.recommended_params.m, 8);
    }

    #[test]
    fn test_max_speed_halves_ef_search() {
        let opt = EmbeddingIndexOptimizer::new();
        let current = HnswParams {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
        };
        let report = opt.recommend_params(current, IndexTuningGoal::MaxSpeed, 1000);
        assert_eq!(report.recommended_params.ef_search, 25);
    }

    #[test]
    fn test_max_speed_negative_recall_change() {
        let opt = EmbeddingIndexOptimizer::new();
        let current = HnswParams::default();
        let report = opt.recommend_params(current, IndexTuningGoal::MaxSpeed, 500);
        assert!(report.expected_recall_change < 0.0);
    }

    #[test]
    fn test_max_speed_positive_speed_change() {
        let opt = EmbeddingIndexOptimizer::new();
        let current = HnswParams::default();
        let report = opt.recommend_params(current, IndexTuningGoal::MaxSpeed, 500);
        assert!(report.expected_speed_change > 0.0);
    }

    // ------------------------------------------------------------------
    // Balanced goal
    // ------------------------------------------------------------------

    #[test]
    fn test_balanced_clamps_m_within_range() {
        let opt = EmbeddingIndexOptimizer::new();
        // current.m = 16 => (16+16)/2 = 16, clamped to [8,32] => 16
        let current = HnswParams {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
        };
        let report = opt.recommend_params(current, IndexTuningGoal::Balanced, 1000);
        assert!(report.recommended_params.m >= 8 && report.recommended_params.m <= 32);
    }

    #[test]
    fn test_balanced_zero_recall_and_speed_change() {
        let opt = EmbeddingIndexOptimizer::new();
        let current = HnswParams::default();
        let report = opt.recommend_params(current, IndexTuningGoal::Balanced, 1000);
        assert_eq!(report.expected_recall_change, 0.0);
        assert_eq!(report.expected_speed_change, 0.0);
    }

    #[test]
    fn test_balanced_fixed_ef_values() {
        let opt = EmbeddingIndexOptimizer::new();
        let current = HnswParams::default();
        let report = opt.recommend_params(current, IndexTuningGoal::Balanced, 1000);
        assert_eq!(report.recommended_params.ef_construction, 200);
        assert_eq!(report.recommended_params.ef_search, 50);
    }

    // ------------------------------------------------------------------
    // Large index note
    // ------------------------------------------------------------------

    #[test]
    fn test_large_index_adds_note() {
        let opt = EmbeddingIndexOptimizer::new();
        let current = HnswParams::default();
        let report = opt.recommend_params(current, IndexTuningGoal::MaxRecall, 200_000);
        assert!(!report.notes.is_empty());
    }

    #[test]
    fn test_small_index_no_note() {
        let opt = EmbeddingIndexOptimizer::new();
        let current = HnswParams::default();
        let report = opt.recommend_params(current, IndexTuningGoal::MaxRecall, 50_000);
        assert!(report.notes.is_empty());
    }

    // ------------------------------------------------------------------
    // analyze_levels
    // ------------------------------------------------------------------

    #[test]
    fn test_analyze_levels_returns_strings() {
        let opt = EmbeddingIndexOptimizer::new();
        let dist = LevelDistribution {
            levels: vec![1000, 100, 10],
        };
        let params = HnswParams::default();
        let obs = opt.analyze_levels(&dist, &params);
        assert!(!obs.is_empty());
    }

    #[test]
    fn test_analyze_levels_contains_total_nodes() {
        let opt = EmbeddingIndexOptimizer::new();
        let dist = LevelDistribution {
            levels: vec![500, 50, 5],
        };
        let params = HnswParams::default();
        let obs = opt.analyze_levels(&dist, &params);
        assert!(obs.iter().any(|s| s.contains("555")));
    }

    // ------------------------------------------------------------------
    // estimate_memory_mb
    // ------------------------------------------------------------------

    #[test]
    fn test_estimate_memory_mb_positive() {
        let opt = EmbeddingIndexOptimizer::new();
        let params = HnswParams::default();
        let mb = opt.estimate_memory_mb(10_000, &params);
        assert!(mb > 0.0);
    }

    #[test]
    fn test_estimate_memory_mb_scales_with_node_count() {
        let opt = EmbeddingIndexOptimizer::new();
        let params = HnswParams::default();
        let mb_small = opt.estimate_memory_mb(1_000, &params);
        let mb_large = opt.estimate_memory_mb(10_000, &params);
        assert!(mb_large > mb_small);
    }

    #[test]
    fn test_estimate_memory_mb_zero_nodes() {
        let opt = EmbeddingIndexOptimizer::new();
        let params = HnswParams::default();
        let mb = opt.estimate_memory_mb(0, &params);
        assert_eq!(mb, 0.0);
    }
}
