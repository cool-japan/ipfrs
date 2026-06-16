//! Snapshot differ — computes the minimal set of block operations needed to
//! transform one storage snapshot into another.
//!
//! Used for incremental sync and delta-compression of backup chains.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// BlockOp
// ---------------------------------------------------------------------------

/// A single operation on a content-addressed block.
#[derive(Clone, Debug, PartialEq)]
pub enum BlockOp {
    /// Block must be written / transferred.
    Put { cid: String, size_bytes: u64 },
    /// Block must be removed.
    Delete { cid: String },
    /// Block is present in both snapshots — no action required.
    Unchanged { cid: String },
}

impl BlockOp {
    /// Returns the CID this operation targets.
    pub fn cid(&self) -> &str {
        match self {
            BlockOp::Put { cid, .. } => cid.as_str(),
            BlockOp::Delete { cid } => cid.as_str(),
            BlockOp::Unchanged { cid } => cid.as_str(),
        }
    }
}

// ---------------------------------------------------------------------------
// SnapshotEntry
// ---------------------------------------------------------------------------

/// A single content-addressed block referenced by a snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotEntry {
    /// Content identifier (e.g. a CIDv1 string).
    pub cid: String,
    /// Size of the block in bytes.
    pub size_bytes: u64,
    /// Unix timestamp (seconds) at which this block was created / ingested.
    pub created_at_secs: u64,
    /// Arbitrary tags associated with this block (e.g. `["pinned", "index"]`).
    pub tags: Vec<String>,
}

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

/// An immutable point-in-time view of a set of content-addressed blocks.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    /// Unique identifier for this snapshot.
    pub id: String,
    /// Unix timestamp (seconds) at which the snapshot was taken.
    pub taken_at_secs: u64,
    /// The entries (blocks) present in this snapshot.
    pub entries: Vec<SnapshotEntry>,
}

impl Snapshot {
    /// Returns the number of entries in this snapshot.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Returns the sum of `size_bytes` across all entries.
    pub fn total_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.size_bytes).sum()
    }
}

// ---------------------------------------------------------------------------
// DiffResult
// ---------------------------------------------------------------------------

/// The result of comparing two snapshots.
#[derive(Clone, Debug, PartialEq)]
pub struct DiffResult {
    /// All operations, sorted by CID for determinism.
    ///
    /// When `SnapshotDiffer::include_unchanged` is `false`, `Unchanged` ops
    /// are **excluded** from this vec (but are still counted in `unchanged`).
    pub ops: Vec<BlockOp>,
    /// Number of `Put` operations.
    pub puts: usize,
    /// Number of `Delete` operations.
    pub deletes: usize,
    /// Number of `Unchanged` operations.
    pub unchanged: usize,
    /// Total bytes that would be added.
    pub bytes_added: u64,
    /// Total bytes that would be removed.
    pub bytes_removed: u64,
}

impl DiffResult {
    /// Net byte delta: `bytes_added` − `bytes_removed`.
    pub fn net_bytes(&self) -> i64 {
        self.bytes_added as i64 - self.bytes_removed as i64
    }

    /// `true` if there is at least one `Put` or `Delete` operation.
    pub fn has_changes(&self) -> bool {
        self.puts > 0 || self.deletes > 0
    }
}

// ---------------------------------------------------------------------------
// SnapshotDiffer
// ---------------------------------------------------------------------------

/// Computes the minimal set of [`BlockOp`]s to transition from one
/// [`Snapshot`] to another.
#[derive(Clone, Debug)]
pub struct SnapshotDiffer {
    /// When `true`, `Unchanged` ops are included in [`DiffResult::ops`].
    /// Counts are always tracked regardless of this flag.
    pub include_unchanged: bool,
}

impl SnapshotDiffer {
    /// Creates a new `SnapshotDiffer`.
    pub fn new(include_unchanged: bool) -> Self {
        Self { include_unchanged }
    }

    /// Computes the diff between `old` and `new` snapshots.
    pub fn diff(&self, old: &Snapshot, new: &Snapshot) -> DiffResult {
        // Build lookup for old entries keyed by CID.
        let old_map: HashMap<&str, &SnapshotEntry> =
            old.entries.iter().map(|e| (e.cid.as_str(), e)).collect();

        // Build lookup for new entries to detect deletions cheaply.
        let new_cids: HashMap<&str, ()> =
            new.entries.iter().map(|e| (e.cid.as_str(), ())).collect();

        let mut ops: Vec<BlockOp> = Vec::new();
        let mut puts: usize = 0;
        let mut deletes: usize = 0;
        let mut unchanged: usize = 0;
        let mut bytes_added: u64 = 0;
        let mut bytes_removed: u64 = 0;

        // Walk new entries → Put or Unchanged.
        for entry in &new.entries {
            if old_map.contains_key(entry.cid.as_str()) {
                unchanged += 1;
                if self.include_unchanged {
                    ops.push(BlockOp::Unchanged {
                        cid: entry.cid.clone(),
                    });
                }
            } else {
                puts += 1;
                bytes_added += entry.size_bytes;
                ops.push(BlockOp::Put {
                    cid: entry.cid.clone(),
                    size_bytes: entry.size_bytes,
                });
            }
        }

        // Walk old entries → Delete for anything not in new.
        for entry in &old.entries {
            if !new_cids.contains_key(entry.cid.as_str()) {
                deletes += 1;
                bytes_removed += entry.size_bytes;
                ops.push(BlockOp::Delete {
                    cid: entry.cid.clone(),
                });
            }
        }

        // Sort by CID for determinism.
        ops.sort_by(|a, b| a.cid().cmp(b.cid()));

        DiffResult {
            ops,
            puts,
            deletes,
            unchanged,
            bytes_added,
            bytes_removed,
        }
    }

    /// Applies a slice of [`BlockOp`]s to `base`, producing a new [`Snapshot`].
    ///
    /// - `Delete` removes the matching entry.
    /// - `Put` adds a new entry (`created_at_secs = 0`, `tags = []`).
    /// - `Unchanged` is a no-op (the entry is already present in `base`).
    ///
    /// The returned snapshot has `id = "applied"` and `taken_at_secs = 0`.
    pub fn apply_ops(base: &Snapshot, ops: &[BlockOp]) -> Snapshot {
        // Start with all base entries.
        let mut entries: Vec<SnapshotEntry> = base.entries.clone();

        for op in ops {
            match op {
                BlockOp::Delete { cid } => {
                    entries.retain(|e| &e.cid != cid);
                }
                BlockOp::Put { cid, size_bytes } => {
                    // Only add if not already present (idempotent).
                    if !entries.iter().any(|e| &e.cid == cid) {
                        entries.push(SnapshotEntry {
                            cid: cid.clone(),
                            size_bytes: *size_bytes,
                            created_at_secs: 0,
                            tags: Vec::new(),
                        });
                    }
                }
                BlockOp::Unchanged { .. } => {
                    // No-op — entry is already in base.
                }
            }
        }

        Snapshot {
            id: "applied".to_string(),
            taken_at_secs: 0,
            entries,
        }
    }

    /// Computes diffs between every consecutive pair in `snapshots`.
    ///
    /// Returns a `Vec` of length `snapshots.len() - 1`, or an empty `Vec`
    /// if fewer than two snapshots are provided.
    pub fn chain_diff(&self, snapshots: &[Snapshot]) -> Vec<DiffResult> {
        if snapshots.len() < 2 {
            return Vec::new();
        }
        snapshots
            .windows(2)
            .map(|pair| self.diff(&pair[0], &pair[1]))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(cid: &str, size: u64) -> SnapshotEntry {
        SnapshotEntry {
            cid: cid.to_string(),
            size_bytes: size,
            created_at_secs: 1_000,
            tags: vec!["pinned".to_string()],
        }
    }

    fn make_snapshot(id: &str, entries: Vec<SnapshotEntry>) -> Snapshot {
        Snapshot {
            id: id.to_string(),
            taken_at_secs: 1_000,
            entries,
        }
    }

    // -----------------------------------------------------------------------
    // constructor
    // -----------------------------------------------------------------------

    #[test]
    fn new_include_unchanged_true() {
        let d = SnapshotDiffer::new(true);
        assert!(d.include_unchanged);
    }

    #[test]
    fn new_include_unchanged_false() {
        let d = SnapshotDiffer::new(false);
        assert!(!d.include_unchanged);
    }

    // -----------------------------------------------------------------------
    // diff: edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn diff_both_empty() {
        let d = SnapshotDiffer::new(true);
        let old = make_snapshot("old", vec![]);
        let new = make_snapshot("new", vec![]);
        let result = d.diff(&old, &new);
        assert_eq!(result.ops.len(), 0);
        assert_eq!(result.puts, 0);
        assert_eq!(result.deletes, 0);
        assert_eq!(result.unchanged, 0);
    }

    #[test]
    fn diff_empty_old_new_entries_all_put() {
        let d = SnapshotDiffer::new(true);
        let old = make_snapshot("old", vec![]);
        let new = make_snapshot(
            "new",
            vec![make_entry("cid-a", 100), make_entry("cid-b", 200)],
        );
        let result = d.diff(&old, &new);
        assert_eq!(result.puts, 2);
        assert_eq!(result.deletes, 0);
        assert_eq!(result.unchanged, 0);
        assert!(result
            .ops
            .iter()
            .all(|op| matches!(op, BlockOp::Put { .. })));
    }

    #[test]
    fn diff_old_entries_empty_new_all_delete() {
        let d = SnapshotDiffer::new(true);
        let old = make_snapshot(
            "old",
            vec![make_entry("cid-a", 100), make_entry("cid-b", 200)],
        );
        let new = make_snapshot("new", vec![]);
        let result = d.diff(&old, &new);
        assert_eq!(result.puts, 0);
        assert_eq!(result.deletes, 2);
        assert_eq!(result.unchanged, 0);
        assert!(result
            .ops
            .iter()
            .all(|op| matches!(op, BlockOp::Delete { .. })));
    }

    #[test]
    fn diff_identical_snapshots_all_unchanged() {
        let d = SnapshotDiffer::new(true);
        let entries = vec![make_entry("cid-a", 100), make_entry("cid-b", 200)];
        let old = make_snapshot("old", entries.clone());
        let new = make_snapshot("new", entries);
        let result = d.diff(&old, &new);
        assert_eq!(result.puts, 0);
        assert_eq!(result.deletes, 0);
        assert_eq!(result.unchanged, 2);
        assert!(result
            .ops
            .iter()
            .all(|op| matches!(op, BlockOp::Unchanged { .. })));
    }

    // -----------------------------------------------------------------------
    // diff: mixed operations
    // -----------------------------------------------------------------------

    #[test]
    fn diff_one_added_one_removed_one_unchanged() {
        let d = SnapshotDiffer::new(true);
        let old = make_snapshot(
            "old",
            vec![make_entry("cid-keep", 50), make_entry("cid-old", 100)],
        );
        let new = make_snapshot(
            "new",
            vec![make_entry("cid-keep", 50), make_entry("cid-new", 200)],
        );
        let result = d.diff(&old, &new);
        assert_eq!(result.puts, 1);
        assert_eq!(result.deletes, 1);
        assert_eq!(result.unchanged, 1);
    }

    #[test]
    fn diff_result_sorted_by_cid() {
        let d = SnapshotDiffer::new(true);
        let old = make_snapshot("old", vec![make_entry("zzz", 10)]);
        let new = make_snapshot("new", vec![make_entry("bbb", 20), make_entry("aaa", 30)]);
        let result = d.diff(&old, &new);
        let cids: Vec<&str> = result.ops.iter().map(|op| op.cid()).collect();
        let mut sorted = cids.clone();
        sorted.sort_unstable();
        assert_eq!(cids, sorted, "ops must be sorted by CID");
    }

    #[test]
    fn diff_bytes_added_removed_correct() {
        let d = SnapshotDiffer::new(true);
        let old = make_snapshot("old", vec![make_entry("del", 300)]);
        let new = make_snapshot("new", vec![make_entry("put", 500)]);
        let result = d.diff(&old, &new);
        assert_eq!(result.bytes_added, 500);
        assert_eq!(result.bytes_removed, 300);
    }

    #[test]
    fn diff_net_bytes_positive_when_more_added() {
        let d = SnapshotDiffer::new(true);
        let old = make_snapshot("old", vec![make_entry("del", 100)]);
        let new = make_snapshot("new", vec![make_entry("put", 900)]);
        let result = d.diff(&old, &new);
        assert!(result.net_bytes() > 0);
        assert_eq!(result.net_bytes(), 800_i64);
    }

    #[test]
    fn diff_has_changes_true() {
        let d = SnapshotDiffer::new(true);
        let old = make_snapshot("old", vec![make_entry("a", 1)]);
        let new = make_snapshot("new", vec![make_entry("b", 1)]);
        assert!(d.diff(&old, &new).has_changes());
    }

    #[test]
    fn diff_has_changes_false_when_identical() {
        let d = SnapshotDiffer::new(true);
        let entries = vec![make_entry("a", 1)];
        let old = make_snapshot("old", entries.clone());
        let new = make_snapshot("new", entries);
        assert!(!d.diff(&old, &new).has_changes());
    }

    // -----------------------------------------------------------------------
    // include_unchanged flag
    // -----------------------------------------------------------------------

    #[test]
    fn diff_include_unchanged_false_excludes_unchanged_from_ops() {
        let d = SnapshotDiffer::new(false);
        let entries = vec![make_entry("cid-keep", 50)];
        let old = make_snapshot("old", entries.clone());
        let new = make_snapshot("new", entries);
        let result = d.diff(&old, &new);
        // ops vec should be empty because the only op is Unchanged and flag is false
        assert_eq!(result.ops.len(), 0);
    }

    #[test]
    fn diff_include_unchanged_false_still_counts_unchanged() {
        let d = SnapshotDiffer::new(false);
        let old = make_snapshot(
            "old",
            vec![make_entry("keep", 10), make_entry("old-only", 20)],
        );
        let new = make_snapshot(
            "new",
            vec![make_entry("keep", 10), make_entry("new-only", 30)],
        );
        let result = d.diff(&old, &new);
        assert_eq!(result.unchanged, 1);
        assert_eq!(result.puts, 1);
        assert_eq!(result.deletes, 1);
        // Unchanged not in ops, but Put and Delete are
        assert!(result
            .ops
            .iter()
            .all(|op| !matches!(op, BlockOp::Unchanged { .. })));
    }

    // -----------------------------------------------------------------------
    // apply_ops
    // -----------------------------------------------------------------------

    #[test]
    fn apply_ops_put_adds_entry() {
        let base = make_snapshot("base", vec![]);
        let ops = vec![BlockOp::Put {
            cid: "new-cid".to_string(),
            size_bytes: 42,
        }];
        let result = SnapshotDiffer::apply_ops(&base, &ops);
        assert_eq!(result.entry_count(), 1);
        assert_eq!(result.entries[0].cid, "new-cid");
        assert_eq!(result.entries[0].size_bytes, 42);
    }

    #[test]
    fn apply_ops_delete_removes_entry() {
        let base = make_snapshot("base", vec![make_entry("to-delete", 100)]);
        let ops = vec![BlockOp::Delete {
            cid: "to-delete".to_string(),
        }];
        let result = SnapshotDiffer::apply_ops(&base, &ops);
        assert_eq!(result.entry_count(), 0);
    }

    #[test]
    fn apply_ops_unchanged_has_no_effect() {
        let base = make_snapshot("base", vec![make_entry("stable", 50)]);
        let ops = vec![BlockOp::Unchanged {
            cid: "stable".to_string(),
        }];
        let result = SnapshotDiffer::apply_ops(&base, &ops);
        assert_eq!(result.entry_count(), 1);
        assert_eq!(result.entries[0].cid, "stable");
    }

    // -----------------------------------------------------------------------
    // chain_diff
    // -----------------------------------------------------------------------

    #[test]
    fn chain_diff_empty_input_returns_empty() {
        let d = SnapshotDiffer::new(true);
        let result = d.chain_diff(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn chain_diff_single_snapshot_returns_empty() {
        let d = SnapshotDiffer::new(true);
        let s = make_snapshot("only", vec![make_entry("cid", 1)]);
        let result = d.chain_diff(&[s]);
        assert!(result.is_empty());
    }

    #[test]
    fn chain_diff_two_snapshots_returns_one_result() {
        let d = SnapshotDiffer::new(true);
        let s0 = make_snapshot("s0", vec![make_entry("a", 10)]);
        let s1 = make_snapshot("s1", vec![make_entry("b", 20)]);
        let results = d.chain_diff(&[s0, s1]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].puts, 1);
        assert_eq!(results[0].deletes, 1);
    }
}
