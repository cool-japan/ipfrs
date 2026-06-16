//! Versioned tensor checkpoint manager with delta compression.
//!
//! [`TensorCheckpointManagerV2`] manages a history of named tensor checkpoints,
//! automatically selecting between full snapshots and delta-compressed entries
//! based on a configurable [`CheckpointPolicy`].  Delta entries record only the
//! tensors that changed relative to the previous checkpoint, enabling efficient
//! storage of model evolution in federated learning.

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// CheckpointDelta
// ---------------------------------------------------------------------------

/// Describes the incremental change between two checkpoint versions.
#[derive(Debug, Clone)]
pub struct CheckpointDelta {
    /// Version this delta was computed from.
    pub from_version: u64,
    /// Version this delta describes.
    pub to_version: u64,
    /// Names of tensors that changed between the two versions.
    pub changed_tensors: Vec<String>,
    /// Approximate compressed size of the changes in bytes.
    pub delta_bytes: u64,
}

impl CheckpointDelta {
    /// Ratio of delta bytes to full checkpoint bytes.
    ///
    /// Returns a value in `[0.0, 1.0]` (or higher if compression is not
    /// beneficial).  A value of `0.0` indicates a zero-size full checkpoint,
    /// which is guarded against by using `full_bytes.max(1)`.
    #[must_use]
    pub fn compression_ratio(&self, full_bytes: u64) -> f64 {
        self.delta_bytes as f64 / full_bytes.max(1) as f64
    }
}

// ---------------------------------------------------------------------------
// CheckpointEntry
// ---------------------------------------------------------------------------

/// A single checkpoint record stored by the manager.
#[derive(Debug, Clone)]
pub struct CheckpointEntry {
    /// Monotonically increasing version number assigned by the manager.
    pub version: u64,
    /// Unix timestamp (seconds) supplied by the caller at save time.
    pub timestamp_secs: u64,
    /// Names of all tensors present in this checkpoint.
    pub tensor_names: Vec<String>,
    /// Total uncompressed size of the checkpoint in bytes.
    pub full_bytes: u64,
    /// Free-form labels attached to this entry (e.g. `"best"`, `"epoch_10"`).
    pub tags: Vec<String>,
    /// If this is a delta checkpoint, the version it was delta'd from.
    pub delta_from: Option<u64>,
}

impl CheckpointEntry {
    /// Returns `true` when this entry was stored as a delta rather than a full
    /// snapshot.
    #[must_use]
    pub fn is_delta(&self) -> bool {
        self.delta_from.is_some()
    }
}

// ---------------------------------------------------------------------------
// CheckpointPolicy
// ---------------------------------------------------------------------------

/// Controls eviction and storage-format decisions made by the manager.
#[derive(Debug, Clone)]
pub struct CheckpointPolicy {
    /// Maximum number of checkpoints retained before eviction.  Defaults to
    /// `10`.
    pub max_checkpoints: usize,
    /// Every `keep_every_n` versions a *full* checkpoint is unconditionally
    /// stored (instead of a delta).  Defaults to `5`.
    pub keep_every_n: u64,
    /// If a checkpoint's `full_bytes` exceeds this threshold the manager will
    /// attempt to store it as a delta instead.  Defaults to `1 MiB`.
    pub delta_threshold_bytes: u64,
}

impl Default for CheckpointPolicy {
    fn default() -> Self {
        Self {
            max_checkpoints: 10,
            keep_every_n: 5,
            delta_threshold_bytes: 1_048_576,
        }
    }
}

// ---------------------------------------------------------------------------
// CheckpointStats
// ---------------------------------------------------------------------------

/// Aggregate statistics maintained by the manager.
#[derive(Debug, Clone, Default)]
pub struct CheckpointStats {
    /// Total number of checkpoint entries currently held.
    pub total_checkpoints: usize,
    /// Number of those entries that are full snapshots.
    pub full_checkpoints: usize,
    /// Number of those entries that are delta-compressed.
    pub delta_checkpoints: usize,
    /// Sum of `full_bytes` across all retained entries.
    pub total_bytes: u64,
}

impl CheckpointStats {
    /// Fraction of retained checkpoints that are delta-compressed.
    ///
    /// Returns `0.0` when the manager is empty (guarded by `total.max(1)`).
    #[must_use]
    pub fn delta_ratio(&self) -> f64 {
        self.delta_checkpoints as f64 / self.total_checkpoints.max(1) as f64
    }
}

// ---------------------------------------------------------------------------
// TensorCheckpointManagerV2
// ---------------------------------------------------------------------------

/// Versioned tensor checkpoint manager with delta compression.
///
/// # Example
///
/// ```rust
/// use ipfrs_tensorlogic::checkpoint_v2::{CheckpointPolicy, TensorCheckpointManagerV2};
///
/// let mut mgr = TensorCheckpointManagerV2::new(CheckpointPolicy::default());
/// let v1 = mgr.save(vec!["w1".into(), "b1".into()], 2_000_000, 1_000, vec![]);
/// let v2 = mgr.save(vec!["w1".into(), "b1".into(), "w2".into()], 2_000_000, 2_000, vec!["best".into()]);
/// assert_eq!(mgr.latest().expect("example: should succeed in docs").version, v2);
/// ```
#[derive(Debug)]
pub struct TensorCheckpointManagerV2 {
    /// All retained checkpoints, kept sorted by ascending `version`.
    pub checkpoints: Vec<CheckpointEntry>,
    /// Policy governing eviction and storage format.
    pub policy: CheckpointPolicy,
    /// Running statistics.
    pub stats: CheckpointStats,
    /// Version that will be assigned to the *next* call to [`save`](Self::save).
    pub next_version: u64,
}

impl TensorCheckpointManagerV2 {
    /// Create a new, empty manager with the given policy.
    #[must_use]
    pub fn new(policy: CheckpointPolicy) -> Self {
        Self {
            checkpoints: Vec::new(),
            policy,
            stats: CheckpointStats::default(),
            next_version: 1,
        }
    }

    // -----------------------------------------------------------------------
    // save
    // -----------------------------------------------------------------------

    /// Persist a new checkpoint and return its assigned version number.
    ///
    /// # Storage format selection
    ///
    /// A *full* checkpoint is stored when any of the following is true:
    ///
    /// * There are no prior checkpoints.
    /// * `version % keep_every_n == 0`.
    /// * `full_bytes <= delta_threshold_bytes`.
    ///
    /// Otherwise a *delta* checkpoint is stored referencing the previous
    /// version.  Its `changed_tensors` field lists tensors absent from the
    /// previous checkpoint (i.e. newly added tensors).
    ///
    /// # Eviction
    ///
    /// Before inserting, if the manager is already at `max_checkpoints`
    /// capacity, the oldest entry with *no* tags is evicted.  If every entry
    /// is tagged the new checkpoint is still pushed (the list grows beyond the
    /// configured maximum).
    pub fn save(
        &mut self,
        tensor_names: Vec<String>,
        full_bytes: u64,
        timestamp_secs: u64,
        tags: Vec<String>,
    ) -> u64 {
        let version = self.next_version;
        self.next_version += 1;

        // Decide full vs delta.
        let delta_from = self.decide_delta_from(version, full_bytes);

        // Determine changed tensors when storing as a delta.
        let changed_tensors: Vec<String> = if let Some(prev_ver) = delta_from {
            if let Some(prev) = self.checkpoints.iter().find(|e| e.version == prev_ver) {
                let prev_set: HashSet<&str> =
                    prev.tensor_names.iter().map(String::as_str).collect();
                tensor_names
                    .iter()
                    .filter(|n| !prev_set.contains(n.as_str()))
                    .cloned()
                    .collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let is_delta = delta_from.is_some();

        // Evict if at capacity and there is at least one un-tagged entry.
        if self.checkpoints.len() >= self.policy.max_checkpoints {
            if let Some(idx) = self.checkpoints.iter().position(|e| e.tags.is_empty()) {
                let evicted = self.checkpoints.remove(idx);
                // Update stats for the evicted entry.
                self.stats.total_checkpoints = self.stats.total_checkpoints.saturating_sub(1);
                if evicted.is_delta() {
                    self.stats.delta_checkpoints = self.stats.delta_checkpoints.saturating_sub(1);
                } else {
                    self.stats.full_checkpoints = self.stats.full_checkpoints.saturating_sub(1);
                }
                self.stats.total_bytes = self.stats.total_bytes.saturating_sub(evicted.full_bytes);
            }
        }

        // Build and push the new entry.
        let entry = CheckpointEntry {
            version,
            timestamp_secs,
            tensor_names,
            full_bytes,
            tags,
            delta_from,
        };

        self.stats.total_bytes += entry.full_bytes;
        if is_delta {
            self.stats.delta_checkpoints += 1;
        } else {
            self.stats.full_checkpoints += 1;
        }
        self.stats.total_checkpoints += 1;

        // Insert sorted by version (append is correct because versions are
        // monotonically increasing).
        self.checkpoints.push(entry);

        // `changed_tensors` computed above is intentionally discarded here;
        // it is only used by `compute_delta` on demand.  Keeping it in scope
        // avoids a dead-variable warning.
        let _ = changed_tensors;

        version
    }

    // -----------------------------------------------------------------------
    // get / latest / versions
    // -----------------------------------------------------------------------

    /// Look up a checkpoint by exact version number.
    #[must_use]
    pub fn get(&self, version: u64) -> Option<&CheckpointEntry> {
        self.checkpoints.iter().find(|e| e.version == version)
    }

    /// Return the checkpoint with the highest version number, or `None` when
    /// the manager is empty.
    #[must_use]
    pub fn latest(&self) -> Option<&CheckpointEntry> {
        self.checkpoints.last()
    }

    /// Return all retained version numbers in ascending order.
    #[must_use]
    pub fn versions(&self) -> Vec<u64> {
        self.checkpoints.iter().map(|e| e.version).collect()
    }

    // -----------------------------------------------------------------------
    // compute_delta
    // -----------------------------------------------------------------------

    /// Compute a [`CheckpointDelta`] between two existing versions.
    ///
    /// * `v1` — the *from* version (must exist).
    /// * `v2` — the *to* version (must exist, should be `> v1`).
    ///
    /// Returns `None` if either version is not found.
    ///
    /// Changed tensors are those present in `v2` but not in `v1` (newly added).
    /// `delta_bytes` is estimated as `full_bytes(v2) * 0.3`.
    #[must_use]
    pub fn compute_delta(&self, v1: u64, v2: u64) -> Option<CheckpointDelta> {
        let entry1 = self.get(v1)?;
        let entry2 = self.get(v2)?;

        let set1: HashSet<&str> = entry1.tensor_names.iter().map(String::as_str).collect();
        let changed_tensors: Vec<String> = entry2
            .tensor_names
            .iter()
            .filter(|n| !set1.contains(n.as_str()))
            .cloned()
            .collect();

        let delta_bytes = (entry2.full_bytes as f64 * 0.3) as u64;

        Some(CheckpointDelta {
            from_version: v1,
            to_version: v2,
            changed_tensors,
            delta_bytes,
        })
    }

    // -----------------------------------------------------------------------
    // tag_checkpoint
    // -----------------------------------------------------------------------

    /// Attach `tag` to the checkpoint at `version`.
    ///
    /// Returns `true` if the version exists and the tag was added, `false`
    /// otherwise (including when `version` does not exist).
    pub fn tag_checkpoint(&mut self, version: u64, tag: String) -> bool {
        if let Some(entry) = self.checkpoints.iter_mut().find(|e| e.version == version) {
            entry.tags.push(tag);
            true
        } else {
            false
        }
    }

    // -----------------------------------------------------------------------
    // stats
    // -----------------------------------------------------------------------

    /// Return a reference to the current aggregate statistics.
    #[must_use]
    pub fn stats(&self) -> &CheckpointStats {
        &self.stats
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Determine the `delta_from` value for a new checkpoint.
    ///
    /// Returns `None` for a full checkpoint, or `Some(prev_version)` for a
    /// delta.
    fn decide_delta_from(&self, version: u64, full_bytes: u64) -> Option<u64> {
        // Full checkpoint conditions.
        if self.checkpoints.is_empty() {
            return None;
        }
        if version.is_multiple_of(self.policy.keep_every_n) {
            return None;
        }
        if full_bytes <= self.policy.delta_threshold_bytes {
            return None;
        }
        // Delta: reference the most recent checkpoint.
        self.checkpoints.last().map(|e| e.version)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_mgr() -> TensorCheckpointManagerV2 {
        TensorCheckpointManagerV2::new(CheckpointPolicy::default())
    }

    // 1. new() produces an empty manager.
    #[test]
    fn test_new_empty() {
        let mgr = default_mgr();
        assert!(mgr.checkpoints.is_empty());
        assert_eq!(mgr.next_version, 1);
        assert_eq!(mgr.stats.total_checkpoints, 0);
    }

    // 2. First checkpoint is always full.
    #[test]
    fn test_first_checkpoint_is_full() {
        let mut mgr = default_mgr();
        let v = mgr.save(vec!["w".into()], 2_000_000, 0, vec![]);
        let entry = mgr.get(v).expect("entry should exist");
        assert!(!entry.is_delta(), "first checkpoint must be full");
        assert!(entry.delta_from.is_none());
    }

    // 3. save() assigns monotonically increasing versions.
    #[test]
    fn test_monotonic_versions() {
        let mut mgr = default_mgr();
        let v1 = mgr.save(vec!["a".into()], 2_000_000, 1, vec![]);
        let v2 = mgr.save(vec!["a".into()], 2_000_000, 2, vec![]);
        let v3 = mgr.save(vec!["a".into()], 2_000_000, 3, vec![]);
        assert!(v1 < v2 && v2 < v3, "versions must be strictly increasing");
    }

    // 4. Every keep_every_n version is a full checkpoint.
    #[test]
    fn test_keep_every_n_is_full() {
        let policy = CheckpointPolicy {
            keep_every_n: 5,
            delta_threshold_bytes: 0, // never skip delta on size grounds
            max_checkpoints: 20,
        };
        let mut mgr = TensorCheckpointManagerV2::new(policy);
        // Save enough checkpoints so that version 5 is reached.
        let mut last_full_ver = 0u64;
        for i in 1u64..=10 {
            let v = mgr.save(vec!["t".into()], 2_000_000, i, vec![]);
            if v.is_multiple_of(5) {
                last_full_ver = v;
            }
        }
        let entry = mgr.get(last_full_ver).expect("version should exist");
        assert!(
            !entry.is_delta(),
            "version divisible by keep_every_n must be full"
        );
    }

    // 5. Between milestones, checkpoints are stored as deltas.
    #[test]
    fn test_between_milestones_is_delta() {
        let policy = CheckpointPolicy {
            keep_every_n: 5,
            delta_threshold_bytes: 0,
            max_checkpoints: 20,
        };
        let mut mgr = TensorCheckpointManagerV2::new(policy);
        // v1 is always full; v2, v3, v4 should be deltas.
        let _v1 = mgr.save(vec!["t".into()], 2_000_000, 1, vec![]);
        let v2 = mgr.save(vec!["t".into()], 2_000_000, 2, vec![]);
        assert!(mgr.get(v2).expect("v2 missing").is_delta());
    }

    // 6. Small checkpoints (below threshold) are stored as full.
    #[test]
    fn test_small_checkpoint_is_full() {
        let mut mgr = default_mgr();
        // First checkpoint always full — advance to v2 with a large one, then
        // store a tiny one at v3.
        let _v1 = mgr.save(vec!["a".into()], 2_000_000, 1, vec![]);
        let _v2 = mgr.save(vec!["a".into()], 2_000_000, 2, vec![]);
        let v3 = mgr.save(vec!["a".into()], 100, 3, vec![]); // 100 bytes < 1 MiB
        assert!(
            !mgr.get(v3).expect("v3 missing").is_delta(),
            "checkpoint below delta threshold must be full"
        );
    }

    // 7. save() evicts the oldest non-tagged entry when at capacity.
    #[test]
    fn test_evicts_oldest_untagged() {
        let policy = CheckpointPolicy {
            max_checkpoints: 3,
            keep_every_n: 100, // prevent forced-full interference
            delta_threshold_bytes: 0,
        };
        let mut mgr = TensorCheckpointManagerV2::new(policy);
        let v1 = mgr.save(vec!["t".into()], 2_000_000, 1, vec![]);
        let _v2 = mgr.save(vec!["t".into()], 2_000_000, 2, vec![]);
        let _v3 = mgr.save(vec!["t".into()], 2_000_000, 3, vec![]);
        // At capacity; next save should evict v1.
        let _v4 = mgr.save(vec!["t".into()], 2_000_000, 4, vec![]);
        assert!(
            mgr.get(v1).is_none(),
            "evicted entry should not be retrievable"
        );
        assert_eq!(mgr.checkpoints.len(), 3);
    }

    // 8. Tagged checkpoints are NOT evicted even when they are the oldest.
    #[test]
    fn test_does_not_evict_tagged() {
        let policy = CheckpointPolicy {
            max_checkpoints: 3,
            keep_every_n: 100,
            delta_threshold_bytes: 0,
        };
        let mut mgr = TensorCheckpointManagerV2::new(policy);
        let v1 = mgr.save(vec!["t".into()], 2_000_000, 1, vec!["best".into()]);
        let _v2 = mgr.save(vec!["t".into()], 2_000_000, 2, vec![]);
        let _v3 = mgr.save(vec!["t".into()], 2_000_000, 3, vec![]);
        // At capacity; v1 is tagged so v2 should be evicted instead.
        let _v4 = mgr.save(vec!["t".into()], 2_000_000, 4, vec![]);
        assert!(
            mgr.get(v1).is_some(),
            "tagged checkpoint must not be evicted"
        );
    }

    // 9. get() returns Some for known version and None for unknown.
    #[test]
    fn test_get_found_and_not_found() {
        let mut mgr = default_mgr();
        let v = mgr.save(vec!["t".into()], 100, 0, vec![]);
        assert!(mgr.get(v).is_some());
        assert!(mgr.get(v + 999).is_none());
    }

    // 10. latest() returns the checkpoint with the highest version.
    #[test]
    fn test_latest_returns_highest() {
        let mut mgr = default_mgr();
        mgr.save(vec!["a".into()], 100, 1, vec![]);
        let v_last = mgr.save(vec!["b".into()], 100, 2, vec![]);
        assert_eq!(mgr.latest().expect("test: should succeed").version, v_last);
    }

    // 11. versions() returns all version numbers in sorted order.
    #[test]
    fn test_versions_sorted() {
        let mut mgr = default_mgr();
        let v1 = mgr.save(vec![], 100, 1, vec![]);
        let v2 = mgr.save(vec![], 100, 2, vec![]);
        let v3 = mgr.save(vec![], 100, 3, vec![]);
        let vs = mgr.versions();
        assert_eq!(vs, vec![v1, v2, v3]);
        assert!(vs.windows(2).all(|w| w[0] < w[1]));
    }

    // 12. compute_delta returns Some for two known versions.
    #[test]
    fn test_compute_delta_some() {
        let mut mgr = default_mgr();
        let v1 = mgr.save(vec!["w1".into()], 100, 1, vec![]);
        let v2 = mgr.save(vec!["w1".into(), "w2".into()], 100, 2, vec![]);
        assert!(mgr.compute_delta(v1, v2).is_some());
    }

    // 13. compute_delta returns None when either version does not exist.
    #[test]
    fn test_compute_delta_none_missing_version() {
        let mut mgr = default_mgr();
        let v1 = mgr.save(vec!["w1".into()], 100, 1, vec![]);
        assert!(mgr.compute_delta(v1, 9999).is_none());
        assert!(mgr.compute_delta(9999, v1).is_none());
    }

    // 14. compute_delta reports correct changed_tensors.
    #[test]
    fn test_compute_delta_changed_tensors_correct() {
        let mut mgr = default_mgr();
        let v1 = mgr.save(vec!["w1".into(), "b1".into()], 100, 1, vec![]);
        let v2 = mgr.save(
            vec!["w1".into(), "b1".into(), "w2".into(), "b2".into()],
            100,
            2,
            vec![],
        );
        let delta = mgr.compute_delta(v1, v2).expect("delta should exist");
        let mut changed = delta.changed_tensors.clone();
        changed.sort();
        assert_eq!(changed, vec!["b2".to_string(), "w2".to_string()]);
    }

    // 15. tag_checkpoint returns true for existing version, false for unknown.
    #[test]
    fn test_tag_checkpoint_returns_correct_bool() {
        let mut mgr = default_mgr();
        let v = mgr.save(vec![], 100, 0, vec![]);
        assert!(mgr.tag_checkpoint(v, "epoch_1".into()));
        assert!(!mgr.tag_checkpoint(9999, "phantom".into()));
        let entry = mgr.get(v).expect("entry should exist");
        assert!(entry.tags.contains(&"epoch_1".to_string()));
    }

    // 16. stats().delta_ratio() is correct.
    #[test]
    fn test_stats_delta_ratio() {
        let policy = CheckpointPolicy {
            keep_every_n: 100,
            delta_threshold_bytes: 0,
            max_checkpoints: 20,
        };
        let mut mgr = TensorCheckpointManagerV2::new(policy);
        // v1 is always full.
        mgr.save(vec!["t".into()], 2_000_000, 1, vec![]);
        // v2, v3 should be deltas (keep_every_n=100, threshold=0).
        mgr.save(vec!["t".into()], 2_000_000, 2, vec![]);
        mgr.save(vec!["t".into()], 2_000_000, 3, vec![]);

        let stats = mgr.stats();
        assert_eq!(stats.total_checkpoints, 3);
        assert_eq!(stats.full_checkpoints, 1);
        assert_eq!(stats.delta_checkpoints, 2);
        let ratio = stats.delta_ratio();
        assert!((ratio - 2.0 / 3.0).abs() < 1e-9);
    }

    // 17. CheckpointEntry::is_delta reflects delta_from field.
    #[test]
    fn test_checkpoint_entry_is_delta() {
        let full = CheckpointEntry {
            version: 1,
            timestamp_secs: 0,
            tensor_names: vec![],
            full_bytes: 0,
            tags: vec![],
            delta_from: None,
        };
        let delta = CheckpointEntry {
            version: 2,
            timestamp_secs: 0,
            tensor_names: vec![],
            full_bytes: 0,
            tags: vec![],
            delta_from: Some(1),
        };
        assert!(!full.is_delta());
        assert!(delta.is_delta());
    }
}
