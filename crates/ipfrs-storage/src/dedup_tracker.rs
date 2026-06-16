//! Block deduplication tracker for IPFRS storage.
//!
//! Tracks which blocks are referenced by multiple DAG nodes, helping GC avoid
//! collecting shared blocks and enabling storage savings reporting.

use std::collections::HashMap;

/// A reference entry tracking how many DAG nodes reference a given block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefEntry {
    /// Content identifier for the block.
    pub cid: String,
    /// Number of DAG nodes referencing this block.
    pub ref_count: u32,
    /// Unix timestamp (seconds) when this block was first seen.
    pub first_seen_secs: u64,
    /// Unix timestamp (seconds) when this block was last seen.
    pub last_seen_secs: u64,
    /// Size of the block in bytes.
    pub size_bytes: u64,
}

impl RefEntry {
    /// Returns `true` if more than one DAG node references this block.
    pub fn is_shared(&self) -> bool {
        self.ref_count > 1
    }

    /// Returns how many bytes are saved by deduplication for this block.
    ///
    /// Calculated as `(ref_count - 1) * size_bytes`.
    pub fn savings_bytes(&self) -> u64 {
        (self.ref_count.saturating_sub(1) as u64).saturating_mul(self.size_bytes)
    }
}

/// A point-in-time snapshot of deduplication statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct DedupStats {
    /// Total number of unique blocks tracked.
    pub total_blocks: usize,
    /// Number of blocks with `ref_count > 1`.
    pub shared_blocks: usize,
    /// Number of blocks with `ref_count == 1`.
    pub unique_blocks: usize,
    /// Sum of all reference counts across all blocks.
    pub total_ref_count: u64,
    /// Sum of `size_bytes` for all tracked blocks (physical storage used).
    pub total_bytes: u64,
    /// Total bytes saved by deduplication (sum of `savings_bytes()`).
    pub saved_bytes: u64,
}

impl DedupStats {
    /// Returns the fraction of blocks that are shared (`shared_blocks / total_blocks`).
    ///
    /// Returns `0.0` when there are no blocks.
    pub fn dedup_ratio(&self) -> f64 {
        self.shared_blocks as f64 / self.total_blocks.max(1) as f64
    }
}

/// Tracks block deduplication across DAG nodes.
///
/// Maintains a mapping from CID to [`RefEntry`], updated as blocks are
/// referenced or dereferenced.  Useful for GC safety checks and for
/// reporting storage savings due to content-addressed deduplication.
#[derive(Debug, Default)]
pub struct BlockDeduplicationTracker {
    entries: HashMap<String, RefEntry>,
}

impl BlockDeduplicationTracker {
    /// Creates a new, empty tracker.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Records a reference to `cid`.
    ///
    /// - If the block is already tracked, its `ref_count` is incremented and
    ///   `last_seen_secs` is updated.
    /// - If the block is new, it is inserted with `ref_count = 1`.
    pub fn add_ref(&mut self, cid: &str, size_bytes: u64, now_secs: u64) {
        if let Some(entry) = self.entries.get_mut(cid) {
            entry.ref_count = entry.ref_count.saturating_add(1);
            entry.last_seen_secs = now_secs;
        } else {
            self.entries.insert(
                cid.to_owned(),
                RefEntry {
                    cid: cid.to_owned(),
                    ref_count: 1,
                    first_seen_secs: now_secs,
                    last_seen_secs: now_secs,
                    size_bytes,
                },
            );
        }
    }

    /// Removes one reference to `cid`.
    ///
    /// - If `ref_count > 1`, it is decremented.
    /// - If `ref_count == 1`, the entry is removed entirely.
    ///
    /// Returns `true` if the CID was found, `false` otherwise.
    pub fn remove_ref(&mut self, cid: &str) -> bool {
        match self.entries.get_mut(cid) {
            Some(entry) if entry.ref_count > 1 => {
                entry.ref_count -= 1;
                true
            }
            Some(_) => {
                self.entries.remove(cid);
                true
            }
            None => false,
        }
    }

    /// Returns the current reference count for `cid`, or `0` if not tracked.
    pub fn ref_count(&self, cid: &str) -> u32 {
        self.entries.get(cid).map_or(0, |e| e.ref_count)
    }

    /// Returns `true` if it is safe for GC to delete `cid`.
    ///
    /// A block is safe to delete when it is not present in the tracker (no
    /// live references) or its stored `ref_count` has somehow reached `0`.
    pub fn is_safe_to_delete(&self, cid: &str) -> bool {
        match self.entries.get(cid) {
            None => true,
            Some(entry) => entry.ref_count == 0,
        }
    }

    /// Returns all entries with `ref_count > 1`, sorted by `savings_bytes` descending.
    pub fn shared_blocks(&self) -> Vec<&RefEntry> {
        let mut shared: Vec<&RefEntry> = self.entries.values().filter(|e| e.is_shared()).collect();
        shared.sort_by_key(|b| std::cmp::Reverse(b.savings_bytes()));
        shared
    }

    /// Returns all entries with `ref_count == 1`.
    pub fn unique_blocks(&self) -> Vec<&RefEntry> {
        self.entries.values().filter(|e| e.ref_count == 1).collect()
    }

    /// Returns a snapshot of current deduplication statistics.
    pub fn stats(&self) -> DedupStats {
        let total_blocks = self.entries.len();
        let mut shared_blocks = 0usize;
        let mut unique_blocks = 0usize;
        let mut total_ref_count = 0u64;
        let mut total_bytes = 0u64;
        let mut saved_bytes = 0u64;

        for entry in self.entries.values() {
            if entry.is_shared() {
                shared_blocks += 1;
            } else {
                unique_blocks += 1;
            }
            total_ref_count += entry.ref_count as u64;
            total_bytes += entry.size_bytes;
            saved_bytes += entry.savings_bytes();
        }

        DedupStats {
            total_blocks,
            shared_blocks,
            unique_blocks,
            total_ref_count,
            total_bytes,
            saved_bytes,
        }
    }

    /// Returns the top `n` entries by `savings_bytes` descending.
    pub fn top_savings(&self, n: usize) -> Vec<&RefEntry> {
        let mut all: Vec<&RefEntry> = self.entries.values().collect();
        all.sort_by_key(|b| std::cmp::Reverse(b.savings_bytes()));
        all.truncate(n);
        all
    }

    /// Removes all entries whose `ref_count` is `0`.
    ///
    /// Returns the number of entries removed.
    pub fn prune_unreferenced(&mut self) -> usize {
        let before = self.entries.len();
        self.entries.retain(|_, e| e.ref_count > 0);
        before - self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn tracker_with_entries() -> BlockDeduplicationTracker {
        let mut t = BlockDeduplicationTracker::new();
        // "a": ref_count=3, size=100 → savings=200
        t.add_ref("a", 100, 10);
        t.add_ref("a", 100, 20);
        t.add_ref("a", 100, 30);
        // "b": ref_count=2, size=50  → savings=50
        t.add_ref("b", 50, 10);
        t.add_ref("b", 50, 20);
        // "c": ref_count=1, size=200 → savings=0
        t.add_ref("c", 200, 10);
        t
    }

    // ── 1. new() empty ────────────────────────────────────────────────────────

    #[test]
    fn test_new_empty() {
        let t = BlockDeduplicationTracker::new();
        assert_eq!(t.entries.len(), 0);
    }

    // ── 2. add_ref: new block inserted with ref_count=1 ───────────────────────

    #[test]
    fn test_add_ref_new_block() {
        let mut t = BlockDeduplicationTracker::new();
        t.add_ref("cid1", 512, 100);
        let entry = t.entries.get("cid1").expect("entry must exist");
        assert_eq!(entry.ref_count, 1);
        assert_eq!(entry.size_bytes, 512);
        assert_eq!(entry.first_seen_secs, 100);
        assert_eq!(entry.last_seen_secs, 100);
        assert_eq!(entry.cid, "cid1");
    }

    // ── 3. add_ref: existing block increments ref_count ───────────────────────

    #[test]
    fn test_add_ref_increments_ref_count() {
        let mut t = BlockDeduplicationTracker::new();
        t.add_ref("cid1", 512, 100);
        t.add_ref("cid1", 512, 200);
        t.add_ref("cid1", 512, 300);
        assert_eq!(t.ref_count("cid1"), 3);
    }

    // ── 4. add_ref: last_seen_secs updated ────────────────────────────────────

    #[test]
    fn test_add_ref_updates_last_seen() {
        let mut t = BlockDeduplicationTracker::new();
        t.add_ref("cid1", 512, 100);
        t.add_ref("cid1", 512, 999);
        let entry = t.entries.get("cid1").expect("entry must exist");
        assert_eq!(entry.first_seen_secs, 100);
        assert_eq!(entry.last_seen_secs, 999);
    }

    // ── 5. remove_ref: ref_count > 1 decrements ──────────────────────────────

    #[test]
    fn test_remove_ref_decrements() {
        let mut t = BlockDeduplicationTracker::new();
        t.add_ref("x", 64, 1);
        t.add_ref("x", 64, 2);
        assert!(t.remove_ref("x"));
        assert_eq!(t.ref_count("x"), 1);
        assert!(t.entries.contains_key("x"));
    }

    // ── 6. remove_ref: ref_count == 1 removes entirely ────────────────────────

    #[test]
    fn test_remove_ref_removes_entry() {
        let mut t = BlockDeduplicationTracker::new();
        t.add_ref("x", 64, 1);
        assert!(t.remove_ref("x"));
        assert!(!t.entries.contains_key("x"));
    }

    // ── 7. remove_ref: not found returns false ────────────────────────────────

    #[test]
    fn test_remove_ref_not_found() {
        let mut t = BlockDeduplicationTracker::new();
        assert!(!t.remove_ref("nonexistent"));
    }

    // ── 8. ref_count: 0 for unknown ───────────────────────────────────────────

    #[test]
    fn test_ref_count_unknown() {
        let t = BlockDeduplicationTracker::new();
        assert_eq!(t.ref_count("ghost"), 0);
    }

    // ── 9. ref_count: correct after multiple add/remove ──────────────────────

    #[test]
    fn test_ref_count_after_add_remove() {
        let mut t = BlockDeduplicationTracker::new();
        t.add_ref("y", 32, 1);
        t.add_ref("y", 32, 2);
        t.add_ref("y", 32, 3);
        t.remove_ref("y");
        assert_eq!(t.ref_count("y"), 2);
        t.remove_ref("y");
        assert_eq!(t.ref_count("y"), 1);
        t.remove_ref("y");
        assert_eq!(t.ref_count("y"), 0); // removed from map
    }

    // ── 10. is_safe_to_delete: true when not in map ──────────────────────────

    #[test]
    fn test_is_safe_to_delete_not_in_map() {
        let t = BlockDeduplicationTracker::new();
        assert!(t.is_safe_to_delete("absent"));
    }

    // ── 11. is_safe_to_delete: false when ref_count > 0 ─────────────────────

    #[test]
    fn test_is_safe_to_delete_has_refs() {
        let mut t = BlockDeduplicationTracker::new();
        t.add_ref("live", 128, 1);
        assert!(!t.is_safe_to_delete("live"));
    }

    // ── 12. is_shared: true for ref_count > 1 ────────────────────────────────

    #[test]
    fn test_is_shared() {
        let e1 = RefEntry {
            cid: "a".to_owned(),
            ref_count: 1,
            first_seen_secs: 0,
            last_seen_secs: 0,
            size_bytes: 100,
        };
        let e2 = RefEntry {
            ref_count: 2,
            ..e1.clone()
        };
        assert!(!e1.is_shared());
        assert!(e2.is_shared());
    }

    // ── 13. savings_bytes: (n-1) * size ──────────────────────────────────────

    #[test]
    fn test_savings_bytes() {
        let entry = RefEntry {
            cid: "z".to_owned(),
            ref_count: 5,
            first_seen_secs: 0,
            last_seen_secs: 0,
            size_bytes: 1000,
        };
        assert_eq!(entry.savings_bytes(), 4000);

        let single = RefEntry {
            ref_count: 1,
            ..entry.clone()
        };
        assert_eq!(single.savings_bytes(), 0);
    }

    // ── 14. shared_blocks sorted by savings desc ──────────────────────────────

    #[test]
    fn test_shared_blocks_sorted() {
        let t = tracker_with_entries();
        let shared = t.shared_blocks();
        // "a": savings=200, "b": savings=50
        assert_eq!(shared.len(), 2);
        assert_eq!(shared[0].cid, "a");
        assert_eq!(shared[1].cid, "b");
    }

    // ── 15. unique_blocks filtered correctly ─────────────────────────────────

    #[test]
    fn test_unique_blocks() {
        let t = tracker_with_entries();
        let unique = t.unique_blocks();
        assert_eq!(unique.len(), 1);
        assert_eq!(unique[0].cid, "c");
    }

    // ── 16. top_savings top-n ─────────────────────────────────────────────────

    #[test]
    fn test_top_savings() {
        let t = tracker_with_entries();
        let top1 = t.top_savings(1);
        assert_eq!(top1.len(), 1);
        assert_eq!(top1[0].cid, "a");

        let top2 = t.top_savings(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].cid, "a");
        assert_eq!(top2[1].cid, "b");

        // requesting more than available returns all
        let top10 = t.top_savings(10);
        assert_eq!(top10.len(), 3);
    }

    // ── 17. stats: all fields correct ─────────────────────────────────────────

    #[test]
    fn test_stats_all_fields() {
        let t = tracker_with_entries();
        let s = t.stats();
        assert_eq!(s.total_blocks, 3);
        assert_eq!(s.shared_blocks, 2); // "a" and "b"
        assert_eq!(s.unique_blocks, 1); // "c"
        assert_eq!(s.total_ref_count, 6); // 3+2+1
        assert_eq!(s.total_bytes, 350); // 100+50+200
        assert_eq!(s.saved_bytes, 250); // 200+50
    }

    // ── 18. dedup_ratio calculation ───────────────────────────────────────────

    #[test]
    fn test_dedup_ratio() {
        let t = tracker_with_entries();
        let s = t.stats();
        // 2 shared out of 3 total
        let ratio = s.dedup_ratio();
        assert!((ratio - 2.0 / 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dedup_ratio_empty() {
        let t = BlockDeduplicationTracker::new();
        let s = t.stats();
        assert_eq!(s.dedup_ratio(), 0.0);
    }

    // ── 19. prune_unreferenced count ─────────────────────────────────────────

    #[test]
    fn test_prune_unreferenced() {
        let mut t = BlockDeduplicationTracker::new();
        t.add_ref("keep", 10, 1);
        t.add_ref("remove_me", 20, 1);

        // Manually set ref_count to 0 to simulate an edge case.
        t.entries
            .get_mut("remove_me")
            .expect("must exist")
            .ref_count = 0;

        let pruned = t.prune_unreferenced();
        assert_eq!(pruned, 1);
        assert!(t.entries.contains_key("keep"));
        assert!(!t.entries.contains_key("remove_me"));
    }

    #[test]
    fn test_prune_unreferenced_none() {
        let mut t = tracker_with_entries();
        let pruned = t.prune_unreferenced();
        assert_eq!(pruned, 0);
    }
}
