//! # Index Merger
//!
//! Merges multiple partial HNSW embedding indexes from distributed nodes into a
//! unified index, handling deduplication and conflict resolution by cosine distance.
//!
//! ## Overview
//!
//! - [`MergeConflict`] — describes conflicts detected during merging
//! - [`ShardEntry`] — a single vector entry within an index shard
//! - [`IndexShard`] — a partial index shard from a distributed node
//! - [`MergeStats`] — statistics produced by a merge operation
//! - [`MergeConfig`] — configuration for the merge process
//! - [`EmbeddingIndexMerger`] — merges multiple shards into a unified index

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// MergeConflict
// ---------------------------------------------------------------------------

/// Describes a conflict detected while merging index shards.
#[derive(Debug, Clone, PartialEq)]
pub enum MergeConflict {
    /// Same embedding ID appears in multiple shards with meaningfully different vectors.
    ///
    /// `score_diff` is the cosine distance between the two conflicting vectors.
    DuplicateId {
        /// The embedding ID that appears in more than one shard.
        id: u64,
        /// Cosine distance between the existing and the incoming vector.
        score_diff: f32,
    },

    /// An entry's vector dimension does not match the declared shard dimension.
    DimensionMismatch {
        /// The dimension declared by the shard.
        expected: usize,
        /// The actual dimension of the offending entry's vector.
        actual: usize,
    },

    /// Adding further entries would exceed the configured capacity limit.
    CapacityExceeded {
        /// The maximum number of entries allowed.
        limit: usize,
    },
}

impl std::fmt::Display for MergeConflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateId { id, score_diff } => {
                write!(
                    f,
                    "duplicate embedding id {id} with cosine distance {score_diff:.6}"
                )
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::CapacityExceeded { limit } => {
                write!(f, "merged index would exceed capacity limit of {limit}")
            }
        }
    }
}

impl std::error::Error for MergeConflict {}

// ---------------------------------------------------------------------------
// ShardEntry
// ---------------------------------------------------------------------------

/// A single vector entry stored within an [`IndexShard`].
#[derive(Debug, Clone, PartialEq)]
pub struct ShardEntry {
    /// Globally unique identifier for this embedding.
    pub id: u64,
    /// The raw embedding vector.
    pub vector: Vec<f32>,
    /// Arbitrary string tag attached to this entry (e.g., document reference).
    pub metadata: String,
}

impl ShardEntry {
    /// Creates a new `ShardEntry`.
    pub fn new(id: u64, vector: Vec<f32>, metadata: impl Into<String>) -> Self {
        Self {
            id,
            vector,
            metadata: metadata.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// IndexShard
// ---------------------------------------------------------------------------

/// A partial HNSW embedding index from a single distributed node.
#[derive(Debug, Clone)]
pub struct IndexShard {
    /// Human-readable identifier for this shard (e.g., node address or UUID).
    pub shard_id: String,
    /// All vector entries stored in this shard.
    pub entries: Vec<ShardEntry>,
    /// Expected dimensionality of every vector in `entries`.
    pub dimension: usize,
}

impl IndexShard {
    /// Creates a new `IndexShard`.
    pub fn new(shard_id: impl Into<String>, entries: Vec<ShardEntry>, dimension: usize) -> Self {
        Self {
            shard_id: shard_id.into(),
            entries,
            dimension,
        }
    }

    /// Validates that every entry's vector has the declared dimension.
    ///
    /// Returns `Err(MergeConflict::DimensionMismatch)` on the first offending entry.
    pub fn validate(&self) -> Result<(), MergeConflict> {
        for entry in &self.entries {
            if entry.vector.len() != self.dimension {
                return Err(MergeConflict::DimensionMismatch {
                    expected: self.dimension,
                    actual: entry.vector.len(),
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MergeStats
// ---------------------------------------------------------------------------

/// Statistics produced by a single [`EmbeddingIndexMerger::merge`] call.
#[derive(Debug, Clone, Default)]
pub struct MergeStats {
    /// Total number of entries across all input shards before deduplication.
    pub total_input_entries: usize,
    /// Entries skipped because their ID already existed in the output.
    pub deduplicated: usize,
    /// Conflicts recorded during the merge (duplicate IDs beyond the threshold).
    pub conflicts: Vec<MergeConflict>,
    /// Number of entries in the resulting merged index.
    pub output_entries: usize,
    /// Number of shards that were successfully merged.
    pub shards_merged: usize,
}

// ---------------------------------------------------------------------------
// MergeConfig
// ---------------------------------------------------------------------------

/// Configuration that governs how [`EmbeddingIndexMerger`] resolves duplicates.
#[derive(Debug, Clone)]
pub struct MergeConfig {
    /// Maximum number of entries in the merged output (default: 100 000).
    pub max_entries: usize,
    /// Cosine distance above which a duplicate ID is considered a conflict
    /// and recorded in [`MergeStats::conflicts`] (default: 0.01).
    pub conflict_threshold: f32,
    /// When `true`, the first-seen entry wins on an ID collision.
    /// When `false`, the later entry replaces the earlier one.
    pub keep_first: bool,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            max_entries: 100_000,
            conflict_threshold: 0.01,
            keep_first: true,
        }
    }
}

// ---------------------------------------------------------------------------
// EmbeddingIndexMerger
// ---------------------------------------------------------------------------

/// Merges multiple partial HNSW embedding index shards into a unified index.
///
/// # Example
///
/// ```rust
/// use ipfrs_semantic::index_merger::{
///     EmbeddingIndexMerger, IndexShard, MergeConfig, ShardEntry,
/// };
///
/// let config = MergeConfig::default();
/// let merger = EmbeddingIndexMerger::new(config);
///
/// let shard = IndexShard::new(
///     "node-1",
///     vec![ShardEntry::new(1, vec![1.0_f32, 0.0], "doc-a")],
///     2,
/// );
///
/// let (entries, stats) = merger.merge(&[shard]).expect("merge failed");
/// assert_eq!(entries.len(), 1);
/// assert_eq!(stats.output_entries, 1);
/// ```
pub struct EmbeddingIndexMerger {
    /// Configuration for this merger instance.
    pub config: MergeConfig,
}

impl EmbeddingIndexMerger {
    /// Creates a new `EmbeddingIndexMerger` with the supplied configuration.
    pub fn new(config: MergeConfig) -> Self {
        Self { config }
    }

    /// Merges the provided `shards` into a single flat list of [`ShardEntry`] values.
    ///
    /// ## Algorithm
    ///
    /// 1. Each shard is validated; a dimension mismatch causes an immediate `Err`.
    /// 2. Entries are walked in shard order.  A `HashMap<u64, usize>` maps each seen
    ///    embedding ID to its current position in the output vector.
    /// 3. On a duplicate ID the cosine distance between the stored vector and the
    ///    incoming vector is computed:
    ///    - Distance > `conflict_threshold` → [`MergeConflict::DuplicateId`] is
    ///      appended to `stats.conflicts`; the new entry is *always* skipped.
    ///    - Distance ≤ `conflict_threshold` → near-identical copy; skipped silently.
    ///    - When `keep_first = false` the existing slot is overwritten regardless of
    ///      whether the conflict was recorded.
    /// 4. If adding a new (non-duplicate) entry would push the output beyond
    ///    `max_entries`, `Err(CapacityExceeded)` is returned immediately.
    ///
    /// Returns `(merged_entries, stats)` on success.
    pub fn merge(
        &self,
        shards: &[IndexShard],
    ) -> Result<(Vec<ShardEntry>, MergeStats), MergeConflict> {
        // Validate every shard upfront.
        for shard in shards {
            shard.validate()?;
        }

        let mut output: Vec<ShardEntry> = Vec::new();
        // Maps embedding id → index into `output`.
        let mut id_index: HashMap<u64, usize> = HashMap::new();

        let mut stats = MergeStats {
            total_input_entries: shards.iter().map(|s| s.entries.len()).sum(),
            shards_merged: shards.len(),
            ..Default::default()
        };

        for shard in shards {
            for entry in &shard.entries {
                if let Some(&existing_idx) = id_index.get(&entry.id) {
                    // --- duplicate ID ---
                    stats.deduplicated += 1;

                    let dist = Self::cosine_distance(&output[existing_idx].vector, &entry.vector);

                    if dist > self.config.conflict_threshold {
                        stats.conflicts.push(MergeConflict::DuplicateId {
                            id: entry.id,
                            score_diff: dist,
                        });
                    }

                    // When keep_first=false we overwrite the existing slot.
                    if !self.config.keep_first {
                        output[existing_idx] = entry.clone();
                    }
                } else {
                    // --- new entry ---
                    if output.len() >= self.config.max_entries {
                        return Err(MergeConflict::CapacityExceeded {
                            limit: self.config.max_entries,
                        });
                    }
                    let idx = output.len();
                    id_index.insert(entry.id, idx);
                    output.push(entry.clone());
                }
            }
        }

        stats.output_entries = output.len();

        Ok((output, stats))
    }

    /// Computes the cosine distance between two vectors.
    ///
    /// `cosine_distance = 1.0 - cosine_similarity`
    ///
    /// Returns `1.0` when either vector has zero norm (undefined similarity).
    pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0;
        }

        let similarity = dot / (norm_a * norm_b);
        // Clamp to [-1, 1] to guard against floating-point rounding past the boundary.
        let similarity = similarity.clamp(-1.0_f32, 1.0_f32);
        1.0 - similarity
    }

    /// Returns a list of `(shard_id, entry_count)` pairs sorted by entry count
    /// in **descending** order.
    pub fn shard_coverage(shards: &[IndexShard]) -> Vec<(String, usize)> {
        let mut coverage: Vec<(String, usize)> = shards
            .iter()
            .map(|s| (s.shard_id.clone(), s.entries.len()))
            .collect();
        coverage.sort_by_key(|a| std::cmp::Reverse(a.1));
        coverage
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper builders
    // -----------------------------------------------------------------------

    fn make_entry(id: u64, vector: Vec<f32>) -> ShardEntry {
        ShardEntry::new(id, vector, format!("meta-{id}"))
    }

    fn make_shard(shard_id: &str, entries: Vec<ShardEntry>, dimension: usize) -> IndexShard {
        IndexShard::new(shard_id, entries, dimension)
    }

    fn default_merger() -> EmbeddingIndexMerger {
        EmbeddingIndexMerger::new(MergeConfig::default())
    }

    // -----------------------------------------------------------------------
    // 1. new() with config
    // -----------------------------------------------------------------------
    #[test]
    fn test_new_with_config() {
        let config = MergeConfig {
            max_entries: 500,
            conflict_threshold: 0.05,
            keep_first: false,
        };
        let merger = EmbeddingIndexMerger::new(config.clone());
        assert_eq!(merger.config.max_entries, 500);
        assert!((merger.config.conflict_threshold - 0.05).abs() < 1e-6);
        assert!(!merger.config.keep_first);
    }

    // -----------------------------------------------------------------------
    // 2. merge empty shards returns empty output
    // -----------------------------------------------------------------------
    #[test]
    fn test_merge_empty_shards() {
        let merger = default_merger();
        let (entries, stats) = merger.merge(&[]).expect("merge should succeed");
        assert!(entries.is_empty());
        assert_eq!(stats.output_entries, 0);
        assert_eq!(stats.total_input_entries, 0);
        assert_eq!(stats.shards_merged, 0);
    }

    // -----------------------------------------------------------------------
    // 3. merge single shard passes through
    // -----------------------------------------------------------------------
    #[test]
    fn test_merge_single_shard_passthrough() {
        let merger = default_merger();
        let shard = make_shard(
            "s1",
            vec![make_entry(1, vec![1.0, 0.0]), make_entry(2, vec![0.0, 1.0])],
            2,
        );
        let (entries, stats) = merger.merge(&[shard]).expect("merge should succeed");
        assert_eq!(entries.len(), 2);
        assert_eq!(stats.output_entries, 2);
        assert_eq!(stats.total_input_entries, 2);
        assert_eq!(stats.shards_merged, 1);
        assert_eq!(stats.deduplicated, 0);
    }

    // -----------------------------------------------------------------------
    // 4. merge two non-overlapping shards combines all
    // -----------------------------------------------------------------------
    #[test]
    fn test_merge_two_non_overlapping_shards() {
        let merger = default_merger();
        let s1 = make_shard("s1", vec![make_entry(1, vec![1.0, 0.0])], 2);
        let s2 = make_shard("s2", vec![make_entry(2, vec![0.0, 1.0])], 2);
        let (entries, stats) = merger.merge(&[s1, s2]).expect("merge should succeed");
        assert_eq!(entries.len(), 2);
        assert_eq!(stats.output_entries, 2);
        assert_eq!(stats.total_input_entries, 2);
        assert_eq!(stats.shards_merged, 2);
        assert_eq!(stats.deduplicated, 0);
        assert!(stats.conflicts.is_empty());
    }

    // -----------------------------------------------------------------------
    // 5. duplicate ID keep_first=true keeps original
    // -----------------------------------------------------------------------
    #[test]
    fn test_duplicate_keep_first_true() {
        let config = MergeConfig {
            keep_first: true,
            conflict_threshold: 0.01,
            ..Default::default()
        };
        let merger = EmbeddingIndexMerger::new(config);

        let first_vec = vec![1.0_f32, 0.0];
        let second_vec = vec![0.0_f32, 1.0]; // orthogonal — distance = 1.0 > threshold

        let s1 = make_shard(
            "s1",
            vec![ShardEntry::new(42, first_vec.clone(), "first")],
            2,
        );
        let s2 = make_shard(
            "s2",
            vec![ShardEntry::new(42, second_vec.clone(), "second")],
            2,
        );

        let (entries, stats) = merger.merge(&[s1, s2]).expect("merge should succeed");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].vector, first_vec, "should keep first entry");
        assert_eq!(stats.deduplicated, 1);
    }

    // -----------------------------------------------------------------------
    // 6. duplicate ID keep_first=false replaces with newer
    // -----------------------------------------------------------------------
    #[test]
    fn test_duplicate_keep_first_false() {
        let config = MergeConfig {
            keep_first: false,
            conflict_threshold: 0.01,
            ..Default::default()
        };
        let merger = EmbeddingIndexMerger::new(config);

        let first_vec = vec![1.0_f32, 0.0];
        let second_vec = vec![0.0_f32, 1.0]; // orthogonal — high distance

        let s1 = make_shard(
            "s1",
            vec![ShardEntry::new(42, first_vec.clone(), "first")],
            2,
        );
        let s2 = make_shard(
            "s2",
            vec![ShardEntry::new(42, second_vec.clone(), "second")],
            2,
        );

        let (entries, _stats) = merger.merge(&[s1, s2]).expect("merge should succeed");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].vector, second_vec, "should keep last entry");
    }

    // -----------------------------------------------------------------------
    // 7. near-identical duplicate (distance < threshold) skipped silently
    // -----------------------------------------------------------------------
    #[test]
    fn test_near_identical_duplicate_skipped_silently() {
        let config = MergeConfig {
            conflict_threshold: 0.05, // generous threshold
            keep_first: true,
            ..Default::default()
        };
        let merger = EmbeddingIndexMerger::new(config);

        // Two nearly identical unit vectors (tiny perturbation — well within threshold).
        let v1 = vec![1.0_f32, 0.0001];
        let v2 = vec![1.0_f32, 0.0001]; // identical → distance = 0.0

        let s1 = make_shard("s1", vec![ShardEntry::new(7, v1, "a")], 2);
        let s2 = make_shard("s2", vec![ShardEntry::new(7, v2, "b")], 2);

        let (entries, stats) = merger.merge(&[s1, s2]).expect("merge should succeed");
        assert_eq!(entries.len(), 1);
        assert_eq!(stats.deduplicated, 1);
        assert!(
            stats.conflicts.is_empty(),
            "no conflict should be recorded for near-identical vectors"
        );
    }

    // -----------------------------------------------------------------------
    // 8. conflicting duplicate (distance > threshold) recorded in stats
    // -----------------------------------------------------------------------
    #[test]
    fn test_conflicting_duplicate_recorded() {
        let config = MergeConfig {
            conflict_threshold: 0.01,
            keep_first: true,
            ..Default::default()
        };
        let merger = EmbeddingIndexMerger::new(config);

        let v1 = vec![1.0_f32, 0.0];
        let v2 = vec![0.0_f32, 1.0]; // orthogonal → distance = 1.0

        let s1 = make_shard("s1", vec![ShardEntry::new(99, v1, "a")], 2);
        let s2 = make_shard("s2", vec![ShardEntry::new(99, v2, "b")], 2);

        let (entries, stats) = merger.merge(&[s1, s2]).expect("merge should succeed");
        assert_eq!(entries.len(), 1);
        assert_eq!(stats.deduplicated, 1);
        assert_eq!(stats.conflicts.len(), 1);

        match &stats.conflicts[0] {
            MergeConflict::DuplicateId { id, score_diff } => {
                assert_eq!(*id, 99);
                assert!(
                    (*score_diff - 1.0).abs() < 1e-5,
                    "expected ~1.0, got {score_diff}"
                );
            }
            other => panic!("unexpected conflict variant: {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 9. dimension mismatch returns Err(DimensionMismatch)
    // -----------------------------------------------------------------------
    #[test]
    fn test_dimension_mismatch_returns_err() {
        let merger = default_merger();
        // Shard claims dimension=3 but entry has dimension=2.
        let bad_shard = make_shard("bad", vec![make_entry(1, vec![1.0, 0.0])], 3);
        let result = merger.merge(&[bad_shard]);
        match result {
            Err(MergeConflict::DimensionMismatch { expected, actual }) => {
                assert_eq!(expected, 3);
                assert_eq!(actual, 2);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 10. capacity exceeded returns Err(CapacityExceeded)
    // -----------------------------------------------------------------------
    #[test]
    fn test_capacity_exceeded() {
        let config = MergeConfig {
            max_entries: 2,
            ..Default::default()
        };
        let merger = EmbeddingIndexMerger::new(config);

        let entries: Vec<ShardEntry> = (1..=3)
            .map(|i| make_entry(i, vec![i as f32, 0.0]))
            .collect();
        let shard = make_shard("big", entries, 2);

        match merger.merge(&[shard]) {
            Err(MergeConflict::CapacityExceeded { limit }) => assert_eq!(limit, 2),
            other => panic!("expected CapacityExceeded, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 11. cosine_distance identical vectors = 0.0
    // -----------------------------------------------------------------------
    #[test]
    fn test_cosine_distance_identical() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let dist = EmbeddingIndexMerger::cosine_distance(&v, &v);
        assert!(
            dist.abs() < 1e-6,
            "identical vectors → distance 0.0, got {dist}"
        );
    }

    // -----------------------------------------------------------------------
    // 12. cosine_distance orthogonal vectors = 1.0
    // -----------------------------------------------------------------------
    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let dist = EmbeddingIndexMerger::cosine_distance(&a, &b);
        assert!(
            (dist - 1.0).abs() < 1e-6,
            "orthogonal vectors → distance 1.0, got {dist}"
        );
    }

    // -----------------------------------------------------------------------
    // 13. cosine_distance zero vector = 1.0
    // -----------------------------------------------------------------------
    #[test]
    fn test_cosine_distance_zero_vector() {
        let a = vec![1.0_f32, 0.0];
        let zero = vec![0.0_f32, 0.0];
        let dist = EmbeddingIndexMerger::cosine_distance(&a, &zero);
        assert!(
            (dist - 1.0).abs() < 1e-6,
            "zero vector → distance 1.0, got {dist}"
        );

        let dist2 = EmbeddingIndexMerger::cosine_distance(&zero, &a);
        assert!((dist2 - 1.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // 14. validate() catches wrong-dimension entry
    // -----------------------------------------------------------------------
    #[test]
    fn test_validate_catches_wrong_dimension() {
        let shard = make_shard(
            "s",
            vec![
                make_entry(1, vec![1.0, 0.0, 0.0]),
                make_entry(2, vec![0.5, 0.5]), // wrong: dim=2 instead of 3
            ],
            3,
        );
        match shard.validate() {
            Err(MergeConflict::DimensionMismatch { expected, actual }) => {
                assert_eq!(expected, 3);
                assert_eq!(actual, 2);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 15. validate() passes for correct dimensions
    // -----------------------------------------------------------------------
    #[test]
    fn test_validate_passes_for_correct_dimensions() {
        let shard = make_shard(
            "s",
            vec![make_entry(1, vec![1.0, 0.0]), make_entry(2, vec![0.0, 1.0])],
            2,
        );
        assert!(shard.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // 16. shard_coverage sorted descending
    // -----------------------------------------------------------------------
    #[test]
    fn test_shard_coverage_sorted_descending() {
        let s1 = make_shard(
            "large",
            vec![
                make_entry(1, vec![1.0]),
                make_entry(2, vec![0.5]),
                make_entry(3, vec![0.1]),
            ],
            1,
        );
        let s2 = make_shard("small", vec![make_entry(4, vec![0.9])], 1);
        let s3 = make_shard(
            "medium",
            vec![make_entry(5, vec![0.8]), make_entry(6, vec![0.7])],
            1,
        );

        let coverage = EmbeddingIndexMerger::shard_coverage(&[s1, s2, s3]);
        assert_eq!(coverage.len(), 3);
        // Should be large(3) > medium(2) > small(1)
        assert_eq!(coverage[0].0, "large");
        assert_eq!(coverage[0].1, 3);
        assert_eq!(coverage[1].0, "medium");
        assert_eq!(coverage[1].1, 2);
        assert_eq!(coverage[2].0, "small");
        assert_eq!(coverage[2].1, 1);
    }

    // -----------------------------------------------------------------------
    // 17. stats.deduplicated counted correctly
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_deduplicated_counted() {
        let merger = default_merger();

        // 3 shards, each with id=1 and id=2 → 4 duplicates total.
        let make_pair = |tag: &str| {
            make_shard(
                tag,
                vec![
                    ShardEntry::new(1, vec![1.0, 0.0], "a"),
                    ShardEntry::new(2, vec![0.0, 1.0], "b"),
                ],
                2,
            )
        };

        let shards = vec![make_pair("s1"), make_pair("s2"), make_pair("s3")];
        let (entries, stats) = merger.merge(&shards).expect("merge should succeed");

        // Only 2 unique entries.
        assert_eq!(entries.len(), 2);
        // Shards s2 and s3 each contribute 2 duplicates → 4 total.
        assert_eq!(stats.deduplicated, 4);
        assert_eq!(stats.total_input_entries, 6);
        assert_eq!(stats.output_entries, 2);
    }

    // -----------------------------------------------------------------------
    // 18. stats.shards_merged correct
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_shards_merged_correct() {
        let merger = default_merger();

        let shards: Vec<IndexShard> = (0..5)
            .map(|i| {
                make_shard(
                    &format!("shard-{i}"),
                    vec![make_entry(i, vec![i as f32, 0.0])],
                    2,
                )
            })
            .collect();

        let (entries, stats) = merger.merge(&shards).expect("merge should succeed");
        assert_eq!(stats.shards_merged, 5);
        assert_eq!(entries.len(), 5);
        assert_eq!(stats.output_entries, 5);
    }
}
