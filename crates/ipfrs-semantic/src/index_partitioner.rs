//! Adaptive Index Partitioner
//!
//! Dynamically partitions a vector index across shards based on query load
//! patterns, vector density, and latency targets.  The partitioner observes
//! the current distribution of vectors and query load across shards, then
//! emits [`RebalanceAction`]s describing how the cluster should be adjusted
//! to restore balance.
//!
//! ## Design Goals
//!
//! * **Load-driven splits** – shards whose query load exceeds twice the cluster
//!   average are candidates for a [`RebalanceAction::Split`].
//! * **Size-bounded merges** – adjacent shards that are both below the minimum
//!   size threshold are candidates for a [`RebalanceAction::Merge`].
//! * **Live migration** – overloaded shards may shed load to the least-loaded
//!   shard via [`RebalanceAction::Migrate`].
//! * **Idempotent advice** – the same set of partitions always produces the
//!   same set of actions; callers decide when and how to apply them.

// ---------------------------------------------------------------------------
// PartitionBoundary
// ---------------------------------------------------------------------------

/// Describes a contiguous slice of the global vector index that is served by a
/// single shard.
#[derive(Debug, Clone, PartialEq)]
pub struct PartitionBoundary {
    /// Unique identifier for the shard that owns this partition.
    pub shard_id: String,
    /// First vector index owned by this partition (inclusive).
    pub start_idx: u64,
    /// First vector index *not* owned by this partition (exclusive).
    pub end_idx: u64,
    /// Number of vectors currently stored in this partition.
    pub vector_count: u64,
    /// Queries per second currently routed to this shard.
    pub query_load: f64,
}

impl PartitionBoundary {
    /// Returns the fraction of `total_load` that this shard handles.
    ///
    /// A tiny epsilon (`1e-9`) guards against division by zero when the entire
    /// cluster is idle.
    #[inline]
    pub fn load_ratio(&self, total_load: f64) -> f64 {
        self.query_load / total_load.max(1e-9)
    }
}

// ---------------------------------------------------------------------------
// RebalanceAction
// ---------------------------------------------------------------------------

/// An advisory action emitted by [`AdaptiveIndexPartitioner::suggest_rebalance`].
///
/// Callers are responsible for translating these actions into actual data
/// movements; the partitioner itself only analyses the current state and
/// produces recommendations.
#[derive(Debug, Clone, PartialEq)]
pub enum RebalanceAction {
    /// Split `shard_id` at `split_at` (vector index), creating two shards.
    Split {
        /// The shard to split.
        shard_id: String,
        /// The vector index at which to split (new shard starts here).
        split_at: u64,
    },
    /// Merge two adjacent shards into a single shard.
    Merge {
        /// The lower shard (by index range).
        shard_a: String,
        /// The upper shard (by index range).
        shard_b: String,
    },
    /// Move `count` vectors from `from_shard` to `to_shard`.
    Migrate {
        /// Source shard that is shedding load.
        from_shard: String,
        /// Destination shard that absorbs the load.
        to_shard: String,
        /// Number of vectors to move.
        count: u64,
    },
    /// The current layout requires no changes.
    NoChange,
}

// ---------------------------------------------------------------------------
// PartitionerConfig
// ---------------------------------------------------------------------------

/// Tuning knobs for the adaptive partitioner.
#[derive(Debug, Clone)]
pub struct PartitionerConfig {
    /// A shard with more vectors than this will be split.
    pub max_vectors_per_shard: u64,
    /// Two adjacent shards that are both below this limit may be merged.
    pub min_vectors_per_shard: u64,
    /// A shard handling more than `load_imbalance_threshold × avg_load` QPS
    /// will be flagged for migration.
    pub load_imbalance_threshold: f64,
    /// Preferred number of shards in the cluster.
    pub target_shard_count: usize,
}

impl Default for PartitionerConfig {
    fn default() -> Self {
        Self {
            max_vectors_per_shard: 50_000,
            min_vectors_per_shard: 1_000,
            load_imbalance_threshold: 2.0,
            target_shard_count: 8,
        }
    }
}

// ---------------------------------------------------------------------------
// PartitionStats
// ---------------------------------------------------------------------------

/// A snapshot of cluster-wide partitioning statistics.
#[derive(Debug, Clone)]
pub struct PartitionStats {
    /// Number of shards currently tracked by the partitioner.
    pub shard_count: usize,
    /// Total number of vectors across all shards.
    pub total_vectors: u64,
    /// Mean vector count across shards (`0.0` when there are no shards).
    pub avg_vectors_per_shard: f64,
    /// `shard_id` of the shard with the highest query load.
    pub max_load_shard: String,
    /// `shard_id` of the shard with the lowest query load.
    pub min_load_shard: String,
    /// Ratio of the maximum shard load to the average shard load.
    ///
    /// A value of `1.0` indicates perfect balance; higher values indicate
    /// hot-spots.  A tiny epsilon (`1e-9`) prevents division by zero.
    pub imbalance_ratio: f64,
}

// ---------------------------------------------------------------------------
// AdaptiveIndexPartitioner
// ---------------------------------------------------------------------------

/// Manages partition boundaries for a distributed vector index and suggests
/// rebalance actions based on load and size heuristics.
///
/// # Example
///
/// ```
/// use ipfrs_semantic::index_partitioner::{
///     AdaptiveIndexPartitioner, PartitionBoundary, PartitionerConfig,
/// };
///
/// let mut partitioner = AdaptiveIndexPartitioner::new(PartitionerConfig::default());
/// partitioner.add_partition(PartitionBoundary {
///     shard_id: "s0".to_string(),
///     start_idx: 0,
///     end_idx: 10_000,
///     vector_count: 10_000,
///     query_load: 100.0,
/// });
/// let actions = partitioner.suggest_rebalance();
/// println!("{actions:?}");
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveIndexPartitioner {
    /// All currently known partition boundaries, sorted by `start_idx`.
    pub partitions: Vec<PartitionBoundary>,
    /// Configuration controlling when rebalance actions are emitted.
    pub config: PartitionerConfig,
}

impl AdaptiveIndexPartitioner {
    /// Creates an empty partitioner with the supplied configuration.
    pub fn new(config: PartitionerConfig) -> Self {
        Self {
            partitions: Vec::new(),
            config,
        }
    }

    /// Appends a partition and re-sorts all partitions by `start_idx`.
    ///
    /// Keeping partitions sorted enables the binary-search logic in
    /// [`Self::find_shard`] and the adjacency checks in
    /// [`Self::suggest_rebalance`].
    pub fn add_partition(&mut self, p: PartitionBoundary) {
        self.partitions.push(p);
        self.partitions.sort_by_key(|p| p.start_idx);
    }

    /// Returns the sum of `query_load` across all partitions.
    pub fn total_load(&self) -> f64 {
        self.partitions.iter().map(|p| p.query_load).sum()
    }

    /// Returns the sum of `vector_count` across all partitions.
    pub fn total_vectors(&self) -> u64 {
        self.partitions.iter().map(|p| p.vector_count).sum()
    }

    /// Locates the shard that owns `vector_idx` using binary search.
    ///
    /// Returns `None` when no shard covers `vector_idx` (gap or out-of-range).
    pub fn find_shard(&self, vector_idx: u64) -> Option<&PartitionBoundary> {
        // Binary-search for the last partition whose start_idx <= vector_idx.
        let pos = self
            .partitions
            .partition_point(|p| p.start_idx <= vector_idx);
        if pos == 0 {
            return None;
        }
        let candidate = &self.partitions[pos - 1];
        if vector_idx < candidate.end_idx {
            Some(candidate)
        } else {
            None
        }
    }

    /// Computes cluster-wide partitioning statistics from the current state.
    pub fn stats(&self) -> PartitionStats {
        let shard_count = self.partitions.len();
        let total_vectors = self.total_vectors();

        let avg_vectors_per_shard = if shard_count == 0 {
            0.0
        } else {
            total_vectors as f64 / shard_count as f64
        };

        if shard_count == 0 {
            return PartitionStats {
                shard_count: 0,
                total_vectors: 0,
                avg_vectors_per_shard: 0.0,
                max_load_shard: String::new(),
                min_load_shard: String::new(),
                imbalance_ratio: 0.0,
            };
        }

        // Identify min/max load shards.  We use index-based folds to avoid
        // lifetime issues while comparing floats (which don't implement Ord).
        let (max_idx, min_idx) =
            self.partitions
                .iter()
                .enumerate()
                .fold((0usize, 0usize), |(max_i, min_i), (i, p)| {
                    let new_max = if p.query_load > self.partitions[max_i].query_load {
                        i
                    } else {
                        max_i
                    };
                    let new_min = if p.query_load < self.partitions[min_i].query_load {
                        i
                    } else {
                        min_i
                    };
                    (new_max, new_min)
                });

        let max_load = self.partitions[max_idx].query_load;
        let avg_load = self.total_load() / shard_count as f64;
        let imbalance_ratio = max_load / avg_load.max(1e-9);

        PartitionStats {
            shard_count,
            total_vectors,
            avg_vectors_per_shard,
            max_load_shard: self.partitions[max_idx].shard_id.clone(),
            min_load_shard: self.partitions[min_idx].shard_id.clone(),
            imbalance_ratio,
        }
    }

    /// Returns `true` when [`Self::suggest_rebalance`] would emit at least one
    /// non-[`RebalanceAction::NoChange`] action.
    pub fn rebalance_needed(&self) -> bool {
        self.suggest_rebalance()
            .iter()
            .any(|a| !matches!(a, RebalanceAction::NoChange))
    }

    /// Analyses the current partition layout and returns a list of advisory
    /// [`RebalanceAction`]s.
    ///
    /// The algorithm applies three passes in order of decreasing urgency:
    ///
    /// 1. **Size overflow** – any shard with `vector_count > max_vectors_per_shard`
    ///    is split at the midpoint of its index range.
    /// 2. **Underfull merge** – pairs of *adjacent* shards that are *both* below
    ///    `min_vectors_per_shard` and *neither* already scheduled for a split are
    ///    merged.
    /// 3. **Load imbalance** – any shard whose `query_load` exceeds
    ///    `load_imbalance_threshold × avg_load` triggers a migrate action that
    ///    moves half its vectors to the currently lightest shard (if a different
    ///    shard exists).
    ///
    /// When none of the above conditions apply the returned slice contains a
    /// single [`RebalanceAction::NoChange`].
    pub fn suggest_rebalance(&self) -> Vec<RebalanceAction> {
        if self.partitions.is_empty() {
            return vec![RebalanceAction::NoChange];
        }

        let mut actions: Vec<RebalanceAction> = Vec::new();
        // Track which shard_ids are already involved in a split so that we
        // don't also emit a merge for them.
        let mut split_shards: std::collections::HashSet<String> = std::collections::HashSet::new();

        // ---------------------------------------------------------------
        // Pass 1: size overflow → Split
        // ---------------------------------------------------------------
        for p in &self.partitions {
            if p.vector_count > self.config.max_vectors_per_shard {
                let split_at = p.start_idx + (p.end_idx - p.start_idx) / 2;
                split_shards.insert(p.shard_id.clone());
                actions.push(RebalanceAction::Split {
                    shard_id: p.shard_id.clone(),
                    split_at,
                });
            }
        }

        // ---------------------------------------------------------------
        // Pass 2: underfull adjacent shards → Merge
        // ---------------------------------------------------------------
        // Work through sorted partitions pairwise.
        let n = self.partitions.len();
        let mut i = 0usize;
        while i + 1 < n {
            let a = &self.partitions[i];
            let b = &self.partitions[i + 1];
            let both_underfull = a.vector_count < self.config.min_vectors_per_shard
                && b.vector_count < self.config.min_vectors_per_shard;
            let neither_splitting =
                !split_shards.contains(&a.shard_id) && !split_shards.contains(&b.shard_id);
            if both_underfull && neither_splitting {
                actions.push(RebalanceAction::Merge {
                    shard_a: a.shard_id.clone(),
                    shard_b: b.shard_id.clone(),
                });
                // Skip both shards so we don't pair b with c as well.
                i += 2;
            } else {
                i += 1;
            }
        }

        // ---------------------------------------------------------------
        // Pass 3: load imbalance → Migrate
        // ---------------------------------------------------------------
        if self.partitions.len() > 1 {
            let total_load = self.total_load();
            let avg_load = total_load / self.partitions.len() as f64;
            let threshold = avg_load * self.config.load_imbalance_threshold;

            // Find the lightest shard (lowest query_load).
            let lightest_idx = self
                .partitions
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.query_load
                        .partial_cmp(&b.query_load)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            for (idx, p) in self.partitions.iter().enumerate() {
                if idx == lightest_idx {
                    continue;
                }
                if p.query_load > threshold {
                    // Migrate half the overloaded shard's vectors to the
                    // lightest shard.
                    let migrate_count = (p.vector_count / 2).max(1);
                    actions.push(RebalanceAction::Migrate {
                        from_shard: p.shard_id.clone(),
                        to_shard: self.partitions[lightest_idx].shard_id.clone(),
                        count: migrate_count,
                    });
                }
            }
        }

        if actions.is_empty() {
            vec![RebalanceAction::NoChange]
        } else {
            actions
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Convenience builder for a PartitionBoundary.
    fn make_partition(
        shard_id: &str,
        start_idx: u64,
        end_idx: u64,
        vector_count: u64,
        query_load: f64,
    ) -> PartitionBoundary {
        PartitionBoundary {
            shard_id: shard_id.to_string(),
            start_idx,
            end_idx,
            vector_count,
            query_load,
        }
    }

    // 1. new() produces an empty partitioner.
    #[test]
    fn test_new_empty() {
        let p = AdaptiveIndexPartitioner::new(PartitionerConfig::default());
        assert!(p.partitions.is_empty());
        assert_eq!(p.total_vectors(), 0);
        assert_eq!(p.total_load(), 0.0);
    }

    // 2. add_partition and total_vectors.
    #[test]
    fn test_add_partition_total_vectors() {
        let mut p = AdaptiveIndexPartitioner::new(PartitionerConfig::default());
        p.add_partition(make_partition("s0", 0, 1000, 1000, 10.0));
        p.add_partition(make_partition("s1", 1000, 2000, 1000, 20.0));
        assert_eq!(p.total_vectors(), 2000);
        assert_eq!(p.partitions.len(), 2);
    }

    // 3. load_ratio calculation.
    #[test]
    fn test_load_ratio() {
        let pb = make_partition("s0", 0, 1000, 1000, 50.0);
        // 50 / 100 = 0.5
        assert!((pb.load_ratio(100.0) - 0.5).abs() < 1e-9);
        // total_load = 0 → use epsilon
        assert!((pb.load_ratio(0.0) - 50.0 / 1e-9).abs() < 1e-3);
    }

    // 4. find_shard: found.
    #[test]
    fn test_find_shard_found() {
        let mut p = AdaptiveIndexPartitioner::new(PartitionerConfig::default());
        p.add_partition(make_partition("s0", 0, 500, 500, 5.0));
        p.add_partition(make_partition("s1", 500, 1000, 500, 5.0));
        let found = p.find_shard(750).expect("should find shard for 750");
        assert_eq!(found.shard_id, "s1");
    }

    // 5. find_shard: not found (gap or out of range).
    #[test]
    fn test_find_shard_not_found() {
        let mut p = AdaptiveIndexPartitioner::new(PartitionerConfig::default());
        p.add_partition(make_partition("s0", 100, 200, 100, 1.0));
        assert!(p.find_shard(50).is_none()); // before first shard
        assert!(p.find_shard(200).is_none()); // at exclusive end_idx
        assert!(p.find_shard(300).is_none()); // after all shards
    }

    // 6. find_shard: first shard boundary.
    #[test]
    fn test_find_shard_first() {
        let mut p = AdaptiveIndexPartitioner::new(PartitionerConfig::default());
        p.add_partition(make_partition("s0", 0, 100, 100, 1.0));
        p.add_partition(make_partition("s1", 100, 200, 100, 1.0));
        let found = p.find_shard(0).expect("idx 0 must be in s0");
        assert_eq!(found.shard_id, "s0");
    }

    // 7. find_shard: last shard boundary.
    #[test]
    fn test_find_shard_last() {
        let mut p = AdaptiveIndexPartitioner::new(PartitionerConfig::default());
        p.add_partition(make_partition("s0", 0, 100, 100, 1.0));
        p.add_partition(make_partition("s1", 100, 200, 100, 1.0));
        let found = p.find_shard(199).expect("idx 199 must be in s1");
        assert_eq!(found.shard_id, "s1");
    }

    // 8. stats(): empty partitions.
    #[test]
    fn test_stats_empty() {
        let p = AdaptiveIndexPartitioner::new(PartitionerConfig::default());
        let s = p.stats();
        assert_eq!(s.shard_count, 0);
        assert_eq!(s.total_vectors, 0);
        assert_eq!(s.avg_vectors_per_shard, 0.0);
        assert!(s.max_load_shard.is_empty());
        assert!(s.min_load_shard.is_empty());
        assert_eq!(s.imbalance_ratio, 0.0);
    }

    // 9. stats(): single shard.
    #[test]
    fn test_stats_single_shard() {
        let mut p = AdaptiveIndexPartitioner::new(PartitionerConfig::default());
        p.add_partition(make_partition("only", 0, 1000, 1000, 42.0));
        let s = p.stats();
        assert_eq!(s.shard_count, 1);
        assert_eq!(s.total_vectors, 1000);
        assert!((s.avg_vectors_per_shard - 1000.0).abs() < 1e-9);
        assert_eq!(s.max_load_shard, "only");
        assert_eq!(s.min_load_shard, "only");
        // max == avg for a single shard, so imbalance_ratio == 1.
        assert!((s.imbalance_ratio - 1.0).abs() < 1e-9);
    }

    // 10. stats(): multiple shards.
    #[test]
    fn test_stats_multiple_shards() {
        let mut p = AdaptiveIndexPartitioner::new(PartitionerConfig::default());
        p.add_partition(make_partition("s0", 0, 1000, 1000, 10.0));
        p.add_partition(make_partition("s1", 1000, 2000, 500, 30.0));
        let s = p.stats();
        assert_eq!(s.shard_count, 2);
        assert_eq!(s.total_vectors, 1500);
        assert!((s.avg_vectors_per_shard - 750.0).abs() < 1e-9);
        assert_eq!(s.max_load_shard, "s1");
        assert_eq!(s.min_load_shard, "s0");
    }

    // 11. imbalance_ratio > 1 for unbalanced cluster.
    #[test]
    fn test_imbalance_ratio_unbalanced() {
        let mut p = AdaptiveIndexPartitioner::new(PartitionerConfig::default());
        p.add_partition(make_partition("low", 0, 1000, 1000, 5.0));
        p.add_partition(make_partition("high", 1000, 2000, 1000, 95.0));
        let s = p.stats();
        // avg_load = 50.0, max_load = 95.0 → ratio = 95/50 = 1.9
        assert!(s.imbalance_ratio > 1.0);
    }

    // 12. suggest_rebalance: overfull → Split.
    #[test]
    fn test_suggest_rebalance_overfull_split() {
        let cfg = PartitionerConfig {
            max_vectors_per_shard: 1_000,
            ..Default::default()
        };
        let mut p = AdaptiveIndexPartitioner::new(cfg);
        // vector_count exceeds max_vectors_per_shard
        p.add_partition(make_partition("big", 0, 4000, 2000, 10.0));
        let actions = p.suggest_rebalance();
        let split = actions
            .iter()
            .find(|a| matches!(a, RebalanceAction::Split { .. }));
        assert!(split.is_some(), "expected a Split action");
        if let Some(RebalanceAction::Split { shard_id, .. }) = split {
            assert_eq!(shard_id, "big");
        }
    }

    // 13. suggest_rebalance: underfull adjacent shards → Merge.
    #[test]
    fn test_suggest_rebalance_underfull_merge() {
        let cfg = PartitionerConfig {
            min_vectors_per_shard: 1_000,
            max_vectors_per_shard: 50_000,
            ..Default::default()
        };
        let mut p = AdaptiveIndexPartitioner::new(cfg);
        // Both shards below min_vectors_per_shard.
        p.add_partition(make_partition("a", 0, 500, 500, 1.0));
        p.add_partition(make_partition("b", 500, 1000, 500, 1.0));
        let actions = p.suggest_rebalance();
        let merge = actions
            .iter()
            .find(|a| matches!(a, RebalanceAction::Merge { .. }));
        assert!(merge.is_some(), "expected a Merge action");
        if let Some(RebalanceAction::Merge { shard_a, shard_b }) = merge {
            assert_eq!(shard_a, "a");
            assert_eq!(shard_b, "b");
        }
    }

    // 14. suggest_rebalance: overloaded → Migrate.
    #[test]
    fn test_suggest_rebalance_overloaded_migrate() {
        let cfg = PartitionerConfig {
            load_imbalance_threshold: 2.0,
            max_vectors_per_shard: 50_000,
            min_vectors_per_shard: 0,
            ..Default::default()
        };
        let mut p = AdaptiveIndexPartitioner::new(cfg);
        // Three shards: avg = (1000+1+1)/3 ≈ 334; threshold = 2×334 ≈ 668.
        // s0 load 1000 > 668 → triggers Migrate.  s1 is first minimum-load shard.
        p.add_partition(make_partition("s0", 0, 10_000, 10_000, 1000.0));
        p.add_partition(make_partition("s1", 10_000, 20_000, 10_000, 1.0));
        p.add_partition(make_partition("s2", 20_000, 30_000, 10_000, 1.0));
        let actions = p.suggest_rebalance();
        let migrate = actions
            .iter()
            .find(|a| matches!(a, RebalanceAction::Migrate { .. }));
        assert!(migrate.is_some(), "expected a Migrate action");
        if let Some(RebalanceAction::Migrate {
            from_shard,
            to_shard,
            ..
        }) = migrate
        {
            assert_eq!(from_shard, "s0");
            assert_eq!(to_shard, "s1");
        }
    }

    // 15. suggest_rebalance: balanced → NoChange only.
    #[test]
    fn test_suggest_rebalance_balanced_nochange() {
        let cfg = PartitionerConfig {
            max_vectors_per_shard: 50_000,
            min_vectors_per_shard: 100,
            load_imbalance_threshold: 2.0,
            target_shard_count: 8,
        };
        let mut p = AdaptiveIndexPartitioner::new(cfg);
        // Both shards are well within bounds.
        p.add_partition(make_partition("s0", 0, 5000, 5000, 50.0));
        p.add_partition(make_partition("s1", 5000, 10_000, 5000, 60.0));
        let actions = p.suggest_rebalance();
        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], RebalanceAction::NoChange));
    }

    // 16. rebalance_needed() returns true/false correctly.
    #[test]
    fn test_rebalance_needed() {
        let cfg = PartitionerConfig {
            max_vectors_per_shard: 1_000,
            ..Default::default()
        };
        let mut p = AdaptiveIndexPartitioner::new(cfg);
        // Empty partitioner → no rebalance needed.
        assert!(!p.rebalance_needed());
        // Add an overfull shard.
        p.add_partition(make_partition("big", 0, 4000, 4000, 10.0));
        assert!(p.rebalance_needed());
    }

    // 17. total_load() is the sum of all query loads.
    #[test]
    fn test_total_load_sum() {
        let mut p = AdaptiveIndexPartitioner::new(PartitionerConfig::default());
        p.add_partition(make_partition("s0", 0, 1000, 1000, 33.3));
        p.add_partition(make_partition("s1", 1000, 2000, 1000, 66.7));
        assert!((p.total_load() - 100.0).abs() < 1e-6);
    }

    // 18. Split occurs at midpoint of the index range.
    #[test]
    fn test_split_at_midpoint() {
        let cfg = PartitionerConfig {
            max_vectors_per_shard: 1_000,
            ..Default::default()
        };
        let mut p = AdaptiveIndexPartitioner::new(cfg);
        // start=0, end=4000 → midpoint = 2000
        p.add_partition(make_partition("m", 0, 4000, 2000, 5.0));
        let actions = p.suggest_rebalance();
        let split = actions
            .iter()
            .find_map(|a| {
                if let RebalanceAction::Split { shard_id, split_at } = a {
                    if shard_id == "m" {
                        Some(*split_at)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .expect("Split action for 'm' must be present");
        assert_eq!(split, 2000);
    }

    // 19. Migrate goes to lightest shard.
    #[test]
    fn test_migrate_targets_lightest_shard() {
        let cfg = PartitionerConfig {
            load_imbalance_threshold: 2.0,
            max_vectors_per_shard: 50_000,
            min_vectors_per_shard: 0,
            target_shard_count: 8,
        };
        let mut p = AdaptiveIndexPartitioner::new(cfg);
        p.add_partition(make_partition("heavy", 0, 10_000, 10_000, 900.0));
        p.add_partition(make_partition("medium", 10_000, 20_000, 10_000, 100.0));
        p.add_partition(make_partition("light", 20_000, 30_000, 10_000, 1.0));
        let actions = p.suggest_rebalance();
        // The migrate from "heavy" must target "light" (lowest load = 1.0).
        let targets: Vec<&str> = actions
            .iter()
            .filter_map(|a| {
                if let RebalanceAction::Migrate {
                    from_shard,
                    to_shard,
                    ..
                } = a
                {
                    if from_shard == "heavy" {
                        Some(to_shard.as_str())
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        assert!(!targets.is_empty(), "expected a Migrate from 'heavy'");
        assert!(
            targets.iter().all(|&t| t == "light"),
            "migrate target must be 'light', got {targets:?}"
        );
    }
}
