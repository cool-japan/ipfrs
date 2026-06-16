//! Storage snapshot diff — computes and represents differences between two
//! point-in-time storage snapshots.
//!
//! Enables incremental sync, changelog generation, and rollback planning by
//! identifying which content-addressed entries were added, removed, modified,
//! or left unchanged between an *old* and a *new* snapshot.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// SnapshotEntry
// ---------------------------------------------------------------------------

/// A single content-addressed entry recorded in a storage snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotEntry {
    /// Content identifier (e.g. a CIDv1 string).
    pub cid: String,
    /// Raw byte size of the block.
    pub size_bytes: u64,
    /// Logical clock / sequence number at which this entry was last modified.
    pub tick: u64,
}

impl SnapshotEntry {
    /// Construct a new entry.
    pub fn new(cid: impl Into<String>, size_bytes: u64, tick: u64) -> Self {
        Self {
            cid: cid.into(),
            size_bytes,
            tick,
        }
    }
}

// ---------------------------------------------------------------------------
// DiffKind
// ---------------------------------------------------------------------------

/// Describes how a content-addressed entry changed between two snapshots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiffKind {
    /// Present in the new snapshot but not in the old one.
    Added,
    /// Present in the old snapshot but not in the new one.
    Removed,
    /// Present in both snapshots, but `size_bytes` or `tick` differs.
    Modified,
    /// Present in both snapshots, identical in every field.
    Unchanged,
}

// ---------------------------------------------------------------------------
// DiffEntry
// ---------------------------------------------------------------------------

/// A single diff record comparing one CID across two snapshots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiffEntry {
    /// The content identifier this record refers to.
    pub cid: String,
    /// How this entry changed.
    pub kind: DiffKind,
    /// The entry as it appeared in the *old* snapshot; `None` for `Added`.
    pub old_entry: Option<SnapshotEntry>,
    /// The entry as it appeared in the *new* snapshot; `None` for `Removed`.
    pub new_entry: Option<SnapshotEntry>,
}

impl DiffEntry {
    /// Signed byte-size change: `new_size - old_size`.
    ///
    /// Missing sides contribute `0`, so:
    /// - `Added`   → `+new_size`
    /// - `Removed` → `-old_size`
    /// - `Modified` / `Unchanged` → difference in size (may be 0)
    pub fn size_delta(&self) -> i64 {
        let new_sz = self.new_entry.as_ref().map(|e| e.size_bytes).unwrap_or(0) as i64;
        let old_sz = self.old_entry.as_ref().map(|e| e.size_bytes).unwrap_or(0) as i64;
        new_sz - old_sz
    }
}

// ---------------------------------------------------------------------------
// SnapshotDiffResult
// ---------------------------------------------------------------------------

/// The complete result of comparing two snapshots.
#[derive(Clone, Debug, Default)]
pub struct SnapshotDiffResult {
    /// Entries that exist only in the new snapshot.
    pub added: Vec<DiffEntry>,
    /// Entries that exist only in the old snapshot.
    pub removed: Vec<DiffEntry>,
    /// Entries present in both but whose `size_bytes` or `tick` changed.
    pub modified: Vec<DiffEntry>,
    /// Entries present in both and completely identical.
    pub unchanged: Vec<DiffEntry>,
}

impl SnapshotDiffResult {
    /// Sum of `size_delta()` across all `added`, `modified`, and `removed` entries.
    pub fn total_size_delta(&self) -> i64 {
        let sum_group = |v: &[DiffEntry]| -> i64 { v.iter().map(|e| e.size_delta()).sum() };
        sum_group(&self.added) + sum_group(&self.modified) + sum_group(&self.removed)
    }

    /// Returns `true` when at least one add, remove, or modification exists.
    pub fn has_changes(&self) -> bool {
        !self.added.is_empty() || !self.removed.is_empty() || !self.modified.is_empty()
    }

    /// Total number of adds, removes, and modifications.
    pub fn change_count(&self) -> usize {
        self.added.len() + self.removed.len() + self.modified.len()
    }
}

// ---------------------------------------------------------------------------
// DiffStats
// ---------------------------------------------------------------------------

/// Cumulative statistics gathered across all diff operations performed by a
/// [`StorageSnapshotDiff`] instance.
#[derive(Clone, Debug, Default)]
pub struct DiffStats {
    /// Number of times [`StorageSnapshotDiff::diff`] has been called.
    pub total_diffs_computed: u64,
    /// Total number of unique CIDs examined across all diffs.
    pub total_entries_compared: u64,
    /// Total number of changes (adds + removes + modifications) found.
    pub total_changes_found: u64,
}

// ---------------------------------------------------------------------------
// StorageSnapshotDiff
// ---------------------------------------------------------------------------

/// Computes and represents differences between storage snapshots.
///
/// Maintains cumulative [`DiffStats`] across repeated calls so callers can
/// observe aggregate diff activity without re-scanning results.
#[derive(Debug, Default)]
pub struct StorageSnapshotDiff {
    stats: DiffStats,
}

impl StorageSnapshotDiff {
    /// Create a new differ with zeroed statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute the diff between `old` and `new` snapshots.
    ///
    /// Each of the four result buckets is sorted by CID ascending for
    /// deterministic output.  Statistics are updated atomically after the
    /// comparison is complete.
    pub fn diff(
        &mut self,
        old: &HashMap<String, SnapshotEntry>,
        new: &HashMap<String, SnapshotEntry>,
    ) -> SnapshotDiffResult {
        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut modified = Vec::new();
        let mut unchanged = Vec::new();

        // Walk the old snapshot — each CID is either unchanged, modified, or removed.
        for (cid, old_entry) in old {
            match new.get(cid) {
                Some(new_entry) => {
                    let kind = if old_entry.size_bytes == new_entry.size_bytes
                        && old_entry.tick == new_entry.tick
                    {
                        DiffKind::Unchanged
                    } else {
                        DiffKind::Modified
                    };
                    let entry = DiffEntry {
                        cid: cid.clone(),
                        kind: kind.clone(),
                        old_entry: Some(old_entry.clone()),
                        new_entry: Some(new_entry.clone()),
                    };
                    if kind == DiffKind::Unchanged {
                        unchanged.push(entry);
                    } else {
                        modified.push(entry);
                    }
                }
                None => removed.push(DiffEntry {
                    cid: cid.clone(),
                    kind: DiffKind::Removed,
                    old_entry: Some(old_entry.clone()),
                    new_entry: None,
                }),
            }
        }

        // Walk the new snapshot — only CIDs absent from old are added.
        for (cid, new_entry) in new {
            if !old.contains_key(cid) {
                added.push(DiffEntry {
                    cid: cid.clone(),
                    kind: DiffKind::Added,
                    old_entry: None,
                    new_entry: Some(new_entry.clone()),
                });
            }
        }

        // Sort each bucket by CID for deterministic ordering.
        added.sort_by(|a, b| a.cid.cmp(&b.cid));
        removed.sort_by(|a, b| a.cid.cmp(&b.cid));
        modified.sort_by(|a, b| a.cid.cmp(&b.cid));
        unchanged.sort_by(|a, b| a.cid.cmp(&b.cid));

        // Compute unique CIDs examined: old ∪ new.
        let unique_cids = {
            let mut set: std::collections::HashSet<&str> = old.keys().map(String::as_str).collect();
            set.extend(new.keys().map(String::as_str));
            set.len() as u64
        };

        let changes = (added.len() + removed.len() + modified.len()) as u64;

        self.stats.total_diffs_computed += 1;
        self.stats.total_entries_compared += unique_cids;
        self.stats.total_changes_found += changes;

        SnapshotDiffResult {
            added,
            removed,
            modified,
            unchanged,
        }
    }

    /// Apply a previously-computed diff to a mutable base snapshot map,
    /// bringing it forward to the state described by the diff.
    ///
    /// - `Added`    → insert the new entry into `base`.
    /// - `Removed`  → remove the entry from `base`.
    /// - `Modified` → replace the entry in `base` with the new version.
    /// - `Unchanged` → no-op.
    pub fn apply_patch(
        &self,
        base: &mut HashMap<String, SnapshotEntry>,
        diff: &SnapshotDiffResult,
    ) {
        for entry in &diff.added {
            if let Some(new_e) = &entry.new_entry {
                base.insert(entry.cid.clone(), new_e.clone());
            }
        }
        for entry in &diff.removed {
            base.remove(&entry.cid);
        }
        for entry in &diff.modified {
            if let Some(new_e) = &entry.new_entry {
                base.insert(entry.cid.clone(), new_e.clone());
            }
        }
        // Unchanged entries require no action.
    }

    /// Return a reference to the cumulative diff statistics.
    pub fn stats(&self) -> &DiffStats {
        &self.stats
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_entry(cid: &str, size: u64, tick: u64) -> SnapshotEntry {
        SnapshotEntry::new(cid, size, tick)
    }

    fn single_entry_map(cid: &str, size: u64, tick: u64) -> HashMap<String, SnapshotEntry> {
        let mut m = HashMap::new();
        m.insert(cid.to_owned(), make_entry(cid, size, tick));
        m
    }

    // -----------------------------------------------------------------------
    // 1. Added entry detected
    // -----------------------------------------------------------------------

    #[test]
    fn test_diff_added_detected() {
        let old: HashMap<String, SnapshotEntry> = HashMap::new();
        let new = single_entry_map("bafy001", 100, 1);
        let mut differ = StorageSnapshotDiff::new();
        let result = differ.diff(&old, &new);
        assert_eq!(result.added.len(), 1);
        assert_eq!(result.added[0].cid, "bafy001");
        assert_eq!(result.added[0].kind, DiffKind::Added);
        assert!(result.added[0].old_entry.is_none());
        assert!(result.added[0].new_entry.is_some());
    }

    // -----------------------------------------------------------------------
    // 2. Removed entry detected
    // -----------------------------------------------------------------------

    #[test]
    fn test_diff_removed_detected() {
        let old = single_entry_map("bafy001", 100, 1);
        let new: HashMap<String, SnapshotEntry> = HashMap::new();
        let mut differ = StorageSnapshotDiff::new();
        let result = differ.diff(&old, &new);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].cid, "bafy001");
        assert_eq!(result.removed[0].kind, DiffKind::Removed);
        assert!(result.removed[0].old_entry.is_some());
        assert!(result.removed[0].new_entry.is_none());
    }

    // -----------------------------------------------------------------------
    // 3. Modified — size changed
    // -----------------------------------------------------------------------

    #[test]
    fn test_diff_modified_size_changed() {
        let old = single_entry_map("bafy001", 100, 5);
        let new = single_entry_map("bafy001", 200, 5);
        let mut differ = StorageSnapshotDiff::new();
        let result = differ.diff(&old, &new);
        assert_eq!(result.modified.len(), 1);
        assert_eq!(result.modified[0].kind, DiffKind::Modified);
    }

    // -----------------------------------------------------------------------
    // 4. Modified — tick changed
    // -----------------------------------------------------------------------

    #[test]
    fn test_diff_modified_tick_changed() {
        let old = single_entry_map("bafy001", 100, 5);
        let new = single_entry_map("bafy001", 100, 6);
        let mut differ = StorageSnapshotDiff::new();
        let result = differ.diff(&old, &new);
        assert_eq!(result.modified.len(), 1);
        assert_eq!(result.modified[0].kind, DiffKind::Modified);
        assert!(result.modified[0].old_entry.is_some());
        assert!(result.modified[0].new_entry.is_some());
    }

    // -----------------------------------------------------------------------
    // 5. Unchanged detected
    // -----------------------------------------------------------------------

    #[test]
    fn test_diff_unchanged_detected() {
        let old = single_entry_map("bafy001", 100, 5);
        let new = single_entry_map("bafy001", 100, 5);
        let mut differ = StorageSnapshotDiff::new();
        let result = differ.diff(&old, &new);
        assert_eq!(result.unchanged.len(), 1);
        assert_eq!(result.unchanged[0].kind, DiffKind::Unchanged);
        assert!(result.removed.is_empty());
        assert!(result.added.is_empty());
        assert!(result.modified.is_empty());
    }

    // -----------------------------------------------------------------------
    // 6. Empty old → all Added
    // -----------------------------------------------------------------------

    #[test]
    fn test_diff_empty_old_all_added() {
        let old: HashMap<String, SnapshotEntry> = HashMap::new();
        let mut new = HashMap::new();
        new.insert("a".to_owned(), make_entry("a", 10, 1));
        new.insert("b".to_owned(), make_entry("b", 20, 2));
        new.insert("c".to_owned(), make_entry("c", 30, 3));
        let mut differ = StorageSnapshotDiff::new();
        let result = differ.diff(&old, &new);
        assert_eq!(result.added.len(), 3);
        assert!(result.removed.is_empty());
        assert!(result.modified.is_empty());
        assert!(result.unchanged.is_empty());
    }

    // -----------------------------------------------------------------------
    // 7. Empty new → all Removed
    // -----------------------------------------------------------------------

    #[test]
    fn test_diff_empty_new_all_removed() {
        let mut old = HashMap::new();
        old.insert("a".to_owned(), make_entry("a", 10, 1));
        old.insert("b".to_owned(), make_entry("b", 20, 2));
        let new: HashMap<String, SnapshotEntry> = HashMap::new();
        let mut differ = StorageSnapshotDiff::new();
        let result = differ.diff(&old, &new);
        assert_eq!(result.removed.len(), 2);
        assert!(result.added.is_empty());
        assert!(result.modified.is_empty());
        assert!(result.unchanged.is_empty());
    }

    // -----------------------------------------------------------------------
    // 8. Identical snapshots → all Unchanged
    // -----------------------------------------------------------------------

    #[test]
    fn test_diff_identical_snapshots_all_unchanged() {
        let mut snap = HashMap::new();
        snap.insert("a".to_owned(), make_entry("a", 10, 1));
        snap.insert("b".to_owned(), make_entry("b", 20, 2));
        let mut differ = StorageSnapshotDiff::new();
        let result = differ.diff(&snap, &snap);
        assert_eq!(result.unchanged.len(), 2);
        assert!(!result.has_changes());
    }

    // -----------------------------------------------------------------------
    // 9. total_size_delta positive
    // -----------------------------------------------------------------------

    #[test]
    fn test_total_size_delta_positive() {
        let old: HashMap<String, SnapshotEntry> = HashMap::new();
        let new = single_entry_map("bafy001", 500, 1);
        let mut differ = StorageSnapshotDiff::new();
        let result = differ.diff(&old, &new);
        assert_eq!(result.total_size_delta(), 500);
    }

    // -----------------------------------------------------------------------
    // 10. total_size_delta negative
    // -----------------------------------------------------------------------

    #[test]
    fn test_total_size_delta_negative() {
        let old = single_entry_map("bafy001", 500, 1);
        let new: HashMap<String, SnapshotEntry> = HashMap::new();
        let mut differ = StorageSnapshotDiff::new();
        let result = differ.diff(&old, &new);
        assert_eq!(result.total_size_delta(), -500);
    }

    // -----------------------------------------------------------------------
    // 11. total_size_delta mixed
    // -----------------------------------------------------------------------

    #[test]
    fn test_total_size_delta_mixed() {
        let mut old = HashMap::new();
        old.insert("a".to_owned(), make_entry("a", 100, 1));
        old.insert("b".to_owned(), make_entry("b", 200, 1));
        let mut new = HashMap::new();
        // "a" removed → -100
        new.insert("b".to_owned(), make_entry("b", 300, 2)); // modified → +100
        new.insert("c".to_owned(), make_entry("c", 50, 1)); // added → +50
        let mut differ = StorageSnapshotDiff::new();
        let result = differ.diff(&old, &new);
        // delta = -100 + 100 + 50 = 50
        assert_eq!(result.total_size_delta(), 50);
    }

    // -----------------------------------------------------------------------
    // 12. has_changes true
    // -----------------------------------------------------------------------

    #[test]
    fn test_has_changes_true() {
        let old: HashMap<String, SnapshotEntry> = HashMap::new();
        let new = single_entry_map("bafy001", 100, 1);
        let mut differ = StorageSnapshotDiff::new();
        let result = differ.diff(&old, &new);
        assert!(result.has_changes());
    }

    // -----------------------------------------------------------------------
    // 13. has_changes false
    // -----------------------------------------------------------------------

    #[test]
    fn test_has_changes_false() {
        let snap = single_entry_map("bafy001", 100, 1);
        let mut differ = StorageSnapshotDiff::new();
        let result = differ.diff(&snap, &snap);
        assert!(!result.has_changes());
    }

    // -----------------------------------------------------------------------
    // 14. change_count correct
    // -----------------------------------------------------------------------

    #[test]
    fn test_change_count_correct() {
        let mut old = HashMap::new();
        old.insert("a".to_owned(), make_entry("a", 100, 1)); // removed
        old.insert("b".to_owned(), make_entry("b", 200, 1)); // modified
        let mut new = HashMap::new();
        new.insert("b".to_owned(), make_entry("b", 300, 2));
        new.insert("c".to_owned(), make_entry("c", 50, 1)); // added
        let mut differ = StorageSnapshotDiff::new();
        let result = differ.diff(&old, &new);
        // 1 removed + 1 modified + 1 added = 3
        assert_eq!(result.change_count(), 3);
    }

    // -----------------------------------------------------------------------
    // 15. apply_patch — added entries inserted into base
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_patch_adds_entries() {
        let old: HashMap<String, SnapshotEntry> = HashMap::new();
        let new = single_entry_map("bafy001", 100, 1);
        let mut differ = StorageSnapshotDiff::new();
        let diff = differ.diff(&old, &new);
        let mut base: HashMap<String, SnapshotEntry> = HashMap::new();
        differ.apply_patch(&mut base, &diff);
        assert!(base.contains_key("bafy001"));
        assert_eq!(base["bafy001"].size_bytes, 100);
    }

    // -----------------------------------------------------------------------
    // 16. apply_patch — removed entries deleted from base
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_patch_removes_entries() {
        let old = single_entry_map("bafy001", 100, 1);
        let new: HashMap<String, SnapshotEntry> = HashMap::new();
        let mut differ = StorageSnapshotDiff::new();
        let diff = differ.diff(&old, &new);
        let mut base = old.clone();
        differ.apply_patch(&mut base, &diff);
        assert!(!base.contains_key("bafy001"));
    }

    // -----------------------------------------------------------------------
    // 17. apply_patch — modified entries updated in base
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_patch_modifies_entries() {
        let old = single_entry_map("bafy001", 100, 1);
        let new = single_entry_map("bafy001", 999, 2);
        let mut differ = StorageSnapshotDiff::new();
        let diff = differ.diff(&old, &new);
        let mut base = old.clone();
        differ.apply_patch(&mut base, &diff);
        assert_eq!(base["bafy001"].size_bytes, 999);
        assert_eq!(base["bafy001"].tick, 2);
    }

    // -----------------------------------------------------------------------
    // 18. apply_patch — unchanged entries stay in base
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_patch_unchanged_stays() {
        let snap = single_entry_map("bafy001", 100, 1);
        let mut differ = StorageSnapshotDiff::new();
        let diff = differ.diff(&snap, &snap);
        let mut base = snap.clone();
        differ.apply_patch(&mut base, &diff);
        assert_eq!(base["bafy001"].size_bytes, 100);
    }

    // -----------------------------------------------------------------------
    // 19. apply_patch — combined operations
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_patch_combined() {
        let mut old = HashMap::new();
        old.insert("keep".to_owned(), make_entry("keep", 10, 1));
        old.insert("del".to_owned(), make_entry("del", 20, 1));
        old.insert("upd".to_owned(), make_entry("upd", 30, 1));
        let mut new = HashMap::new();
        new.insert("keep".to_owned(), make_entry("keep", 10, 1));
        new.insert("upd".to_owned(), make_entry("upd", 99, 2));
        new.insert("fresh".to_owned(), make_entry("fresh", 50, 3));
        let mut differ = StorageSnapshotDiff::new();
        let diff = differ.diff(&old, &new);
        let mut base = old;
        differ.apply_patch(&mut base, &diff);
        // "del" should be gone
        assert!(!base.contains_key("del"));
        // "keep" untouched
        assert_eq!(base["keep"].size_bytes, 10);
        // "upd" updated
        assert_eq!(base["upd"].size_bytes, 99);
        // "fresh" inserted
        assert_eq!(base["fresh"].size_bytes, 50);
    }

    // -----------------------------------------------------------------------
    // 20. Sorted by cid ascending — added
    // -----------------------------------------------------------------------

    #[test]
    fn test_added_sorted_by_cid() {
        let old: HashMap<String, SnapshotEntry> = HashMap::new();
        let mut new = HashMap::new();
        for &cid in &["zzz", "aaa", "mmm", "bbb"] {
            new.insert(cid.to_owned(), make_entry(cid, 1, 1));
        }
        let mut differ = StorageSnapshotDiff::new();
        let result = differ.diff(&old, &new);
        let cids: Vec<&str> = result.added.iter().map(|e| e.cid.as_str()).collect();
        let mut sorted = cids.clone();
        sorted.sort_unstable();
        assert_eq!(cids, sorted);
    }

    // -----------------------------------------------------------------------
    // 21. Sorted by cid ascending — removed
    // -----------------------------------------------------------------------

    #[test]
    fn test_removed_sorted_by_cid() {
        let mut old = HashMap::new();
        for &cid in &["zzz", "aaa", "mmm"] {
            old.insert(cid.to_owned(), make_entry(cid, 1, 1));
        }
        let new: HashMap<String, SnapshotEntry> = HashMap::new();
        let mut differ = StorageSnapshotDiff::new();
        let result = differ.diff(&old, &new);
        let cids: Vec<&str> = result.removed.iter().map(|e| e.cid.as_str()).collect();
        let mut sorted = cids.clone();
        sorted.sort_unstable();
        assert_eq!(cids, sorted);
    }

    // -----------------------------------------------------------------------
    // 22. Sorted by cid ascending — modified
    // -----------------------------------------------------------------------

    #[test]
    fn test_modified_sorted_by_cid() {
        let mut old = HashMap::new();
        let mut new = HashMap::new();
        for &cid in &["zzz", "aaa", "mmm"] {
            old.insert(cid.to_owned(), make_entry(cid, 1, 1));
            new.insert(cid.to_owned(), make_entry(cid, 2, 1));
        }
        let mut differ = StorageSnapshotDiff::new();
        let result = differ.diff(&old, &new);
        let cids: Vec<&str> = result.modified.iter().map(|e| e.cid.as_str()).collect();
        let mut sorted = cids.clone();
        sorted.sort_unstable();
        assert_eq!(cids, sorted);
    }

    // -----------------------------------------------------------------------
    // 23. Sorted by cid ascending — unchanged
    // -----------------------------------------------------------------------

    #[test]
    fn test_unchanged_sorted_by_cid() {
        let mut snap = HashMap::new();
        for &cid in &["zzz", "aaa", "mmm"] {
            snap.insert(cid.to_owned(), make_entry(cid, 1, 1));
        }
        let mut differ = StorageSnapshotDiff::new();
        let result = differ.diff(&snap, &snap);
        let cids: Vec<&str> = result.unchanged.iter().map(|e| e.cid.as_str()).collect();
        let mut sorted = cids.clone();
        sorted.sort_unstable();
        assert_eq!(cids, sorted);
    }

    // -----------------------------------------------------------------------
    // 24. Stats accumulate across multiple diffs
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_accumulate() {
        let mut differ = StorageSnapshotDiff::new();

        // First diff: 1 added
        let old1: HashMap<String, SnapshotEntry> = HashMap::new();
        let new1 = single_entry_map("a", 10, 1);
        differ.diff(&old1, &new1);

        // Second diff: 1 removed, 1 unchanged
        let mut old2 = HashMap::new();
        old2.insert("b".to_owned(), make_entry("b", 20, 1));
        old2.insert("c".to_owned(), make_entry("c", 30, 1));
        let mut new2 = HashMap::new();
        new2.insert("c".to_owned(), make_entry("c", 30, 1));
        differ.diff(&old2, &new2);

        let s = differ.stats();
        assert_eq!(s.total_diffs_computed, 2);
        // First diff: 1 unique CID; second diff: 2 unique CIDs
        assert_eq!(s.total_entries_compared, 3);
        // First diff: 1 change (added); second diff: 1 change (removed)
        assert_eq!(s.total_changes_found, 2);
    }

    // -----------------------------------------------------------------------
    // 25. stats() returns reference with correct initial state
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_initial_state() {
        let differ = StorageSnapshotDiff::new();
        let s = differ.stats();
        assert_eq!(s.total_diffs_computed, 0);
        assert_eq!(s.total_entries_compared, 0);
        assert_eq!(s.total_changes_found, 0);
    }

    // -----------------------------------------------------------------------
    // 26. size_delta for Added is +new_size
    // -----------------------------------------------------------------------

    #[test]
    fn test_size_delta_added() {
        let entry = DiffEntry {
            cid: "x".to_owned(),
            kind: DiffKind::Added,
            old_entry: None,
            new_entry: Some(make_entry("x", 300, 1)),
        };
        assert_eq!(entry.size_delta(), 300);
    }

    // -----------------------------------------------------------------------
    // 27. size_delta for Removed is -old_size
    // -----------------------------------------------------------------------

    #[test]
    fn test_size_delta_removed() {
        let entry = DiffEntry {
            cid: "x".to_owned(),
            kind: DiffKind::Removed,
            old_entry: Some(make_entry("x", 300, 1)),
            new_entry: None,
        };
        assert_eq!(entry.size_delta(), -300);
    }

    // -----------------------------------------------------------------------
    // 28. change_count is zero for identical snapshots
    // -----------------------------------------------------------------------

    #[test]
    fn test_change_count_zero_identical() {
        let snap = single_entry_map("a", 1, 1);
        let mut differ = StorageSnapshotDiff::new();
        let result = differ.diff(&snap, &snap);
        assert_eq!(result.change_count(), 0);
    }
}
