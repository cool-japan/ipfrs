//! Storage Block Manifest — catalogs all blocks in a partition with metadata
//! for fast lookup, consistency checking, and export/import.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// FNV-1a 64-bit hash
// ---------------------------------------------------------------------------

/// Computes the FNV-1a 64-bit hash of `bytes`.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;

    let mut hash = OFFSET_BASIS;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// ManifestEntry
// ---------------------------------------------------------------------------

/// A single block entry in the manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct ManifestEntry {
    /// Content identifier for this block.
    pub cid: String,
    /// Size of the block in bytes.
    pub size_bytes: u64,
    /// Content type, e.g. `"raw"`, `"dag-cbor"`, `"arrow-ipc"`.
    pub content_type: String,
    /// Unix timestamp (seconds) when this entry was added.
    pub added_at_secs: u64,
    /// Whether this block is pinned (protected from GC).
    pub pinned: bool,
    /// FNV-1a 64-bit hash of the CID bytes, for integrity verification.
    pub checksum: u64,
}

impl ManifestEntry {
    /// Construct a new `ManifestEntry`, computing the checksum automatically.
    pub fn new(
        cid: impl Into<String>,
        size_bytes: u64,
        content_type: impl Into<String>,
        added_at_secs: u64,
        pinned: bool,
    ) -> Self {
        let cid = cid.into();
        let checksum = fnv1a(cid.as_bytes());
        Self {
            cid,
            size_bytes,
            content_type: content_type.into(),
            added_at_secs,
            pinned,
            checksum,
        }
    }
}

// ---------------------------------------------------------------------------
// ManifestStats
// ---------------------------------------------------------------------------

/// Aggregate statistics over all entries in a manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct ManifestStats {
    /// Total number of entries.
    pub total_entries: usize,
    /// Total storage consumed by all entries.
    pub total_size_bytes: u64,
    /// Number of pinned entries.
    pub pinned_count: usize,
    /// Per-content-type entry count.
    pub content_type_counts: HashMap<String, usize>,
}

// ---------------------------------------------------------------------------
// ManifestFilter
// ---------------------------------------------------------------------------

/// Predicate used to select a subset of manifest entries.
#[derive(Debug, Clone, PartialEq)]
pub enum ManifestFilter {
    /// Include every entry.
    All,
    /// Include only pinned entries.
    PinnedOnly,
    /// Include only entries whose `content_type` matches the given string.
    ContentType(String),
    /// Include only entries added after the given Unix timestamp (exclusive).
    AddedAfter(u64),
    /// Include only entries whose size exceeds the threshold (exclusive).
    SizeAbove(u64),
}

impl ManifestFilter {
    fn matches(&self, entry: &ManifestEntry) -> bool {
        match self {
            ManifestFilter::All => true,
            ManifestFilter::PinnedOnly => entry.pinned,
            ManifestFilter::ContentType(ct) => &entry.content_type == ct,
            ManifestFilter::AddedAfter(threshold) => entry.added_at_secs > *threshold,
            ManifestFilter::SizeAbove(threshold) => entry.size_bytes > *threshold,
        }
    }
}

// ---------------------------------------------------------------------------
// StorageBlockManifest
// ---------------------------------------------------------------------------

/// Catalogs all blocks in a storage partition.
///
/// Keyed by CID; supports fast lookup, consistency verification, filtering,
/// pinning, statistics, merge, and sorted export.
#[derive(Debug, Clone)]
pub struct StorageBlockManifest {
    /// Block entries indexed by CID.
    pub entries: HashMap<String, ManifestEntry>,
}

impl StorageBlockManifest {
    /// Create an empty manifest.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Add (or overwrite) an entry.  The key used is `entry.cid`.
    pub fn add_entry(&mut self, entry: ManifestEntry) {
        self.entries.insert(entry.cid.clone(), entry);
    }

    /// Remove an entry by CID.
    ///
    /// Returns `true` if the entry existed and was removed, `false` otherwise.
    pub fn remove_entry(&mut self, cid: &str) -> bool {
        self.entries.remove(cid).is_some()
    }

    /// Retrieve an entry by CID, or `None` if not present.
    pub fn get_entry(&self, cid: &str) -> Option<&ManifestEntry> {
        self.entries.get(cid)
    }

    /// Mark the entry with the given CID as pinned.
    ///
    /// Returns `false` if no such entry exists.
    pub fn pin(&mut self, cid: &str) -> bool {
        match self.entries.get_mut(cid) {
            Some(entry) => {
                entry.pinned = true;
                true
            }
            None => false,
        }
    }

    /// Mark the entry with the given CID as unpinned.
    ///
    /// Returns `false` if no such entry exists.
    pub fn unpin(&mut self, cid: &str) -> bool {
        match self.entries.get_mut(cid) {
            Some(entry) => {
                entry.pinned = false;
                true
            }
            None => false,
        }
    }

    /// Return all entries matching `filter`, sorted by `added_at_secs` ascending.
    pub fn filter(&self, filter: &ManifestFilter) -> Vec<&ManifestEntry> {
        let mut matched: Vec<&ManifestEntry> = self
            .entries
            .values()
            .filter(|e| filter.matches(e))
            .collect();
        matched.sort_by_key(|e| e.added_at_secs);
        matched
    }

    /// Compute aggregate statistics for the manifest.
    pub fn stats(&self) -> ManifestStats {
        let mut total_size_bytes: u64 = 0;
        let mut pinned_count: usize = 0;
        let mut content_type_counts: HashMap<String, usize> = HashMap::new();

        for entry in self.entries.values() {
            total_size_bytes = total_size_bytes.saturating_add(entry.size_bytes);
            if entry.pinned {
                pinned_count += 1;
            }
            *content_type_counts
                .entry(entry.content_type.clone())
                .or_insert(0) += 1;
        }

        ManifestStats {
            total_entries: self.entries.len(),
            total_size_bytes,
            pinned_count,
            content_type_counts,
        }
    }

    /// Merge entries from `other` into `self`.
    ///
    /// - If a CID in `other` is absent from `self`, it is inserted.
    /// - If a CID is present in both and `other`'s entry is pinned, `self`'s
    ///   entry is also set to pinned (pin propagation).
    pub fn merge(&mut self, other: &StorageBlockManifest) {
        for (cid, other_entry) in &other.entries {
            match self.entries.get_mut(cid) {
                Some(self_entry) => {
                    if other_entry.pinned {
                        self_entry.pinned = true;
                    }
                }
                None => {
                    self.entries.insert(cid.clone(), other_entry.clone());
                }
            }
        }
    }

    /// Return all CIDs in the manifest, sorted alphabetically.
    pub fn export_cids(&self) -> Vec<String> {
        let mut cids: Vec<String> = self.entries.keys().cloned().collect();
        cids.sort();
        cids
    }

    /// Return the CIDs of entries whose stored checksum does not match
    /// the FNV-1a hash recomputed from the CID bytes.
    pub fn verify_checksums(&self) -> Vec<String> {
        let mut mismatches: Vec<String> = self
            .entries
            .values()
            .filter(|e| fnv1a(e.cid.as_bytes()) != e.checksum)
            .map(|e| e.cid.clone())
            .collect();
        mismatches.sort();
        mismatches
    }
}

impl Default for StorageBlockManifest {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(
        cid: &str,
        size_bytes: u64,
        content_type: &str,
        added_at_secs: u64,
        pinned: bool,
    ) -> ManifestEntry {
        ManifestEntry::new(cid, size_bytes, content_type, added_at_secs, pinned)
    }

    // ------------------------------------------------------------------
    // new()
    // ------------------------------------------------------------------

    #[test]
    fn test_new_starts_empty() {
        let manifest = StorageBlockManifest::new();
        assert!(manifest.entries.is_empty());
    }

    // ------------------------------------------------------------------
    // add_entry / get_entry
    // ------------------------------------------------------------------

    #[test]
    fn test_add_entry_stores_correctly() {
        let mut m = StorageBlockManifest::new();
        let entry = make_entry("cid1", 100, "raw", 1000, false);
        m.add_entry(entry.clone());
        let got = m.get_entry("cid1").expect("entry should exist");
        assert_eq!(got, &entry);
    }

    #[test]
    fn test_add_entry_overwrites_existing() {
        let mut m = StorageBlockManifest::new();
        m.add_entry(make_entry("cid1", 100, "raw", 1000, false));
        m.add_entry(make_entry("cid1", 200, "dag-cbor", 2000, true));
        let got = m.get_entry("cid1").expect("entry should exist");
        assert_eq!(got.size_bytes, 200);
        assert_eq!(got.content_type, "dag-cbor");
    }

    #[test]
    fn test_get_entry_none_for_missing() {
        let m = StorageBlockManifest::new();
        assert!(m.get_entry("nonexistent").is_none());
    }

    // ------------------------------------------------------------------
    // remove_entry
    // ------------------------------------------------------------------

    #[test]
    fn test_remove_entry_returns_true_when_found() {
        let mut m = StorageBlockManifest::new();
        m.add_entry(make_entry("cid1", 100, "raw", 1000, false));
        assert!(m.remove_entry("cid1"));
        assert!(m.get_entry("cid1").is_none());
    }

    #[test]
    fn test_remove_entry_returns_false_when_not_found() {
        let mut m = StorageBlockManifest::new();
        assert!(!m.remove_entry("ghost"));
    }

    // ------------------------------------------------------------------
    // pin / unpin
    // ------------------------------------------------------------------

    #[test]
    fn test_pin_returns_false_when_missing() {
        let mut m = StorageBlockManifest::new();
        assert!(!m.pin("ghost"));
    }

    #[test]
    fn test_unpin_returns_false_when_missing() {
        let mut m = StorageBlockManifest::new();
        assert!(!m.unpin("ghost"));
    }

    #[test]
    fn test_pin_sets_pinned_true() {
        let mut m = StorageBlockManifest::new();
        m.add_entry(make_entry("cid1", 100, "raw", 1000, false));
        assert!(m.pin("cid1"));
        assert!(m.get_entry("cid1").unwrap().pinned);
    }

    #[test]
    fn test_unpin_sets_pinned_false() {
        let mut m = StorageBlockManifest::new();
        m.add_entry(make_entry("cid1", 100, "raw", 1000, true));
        assert!(m.unpin("cid1"));
        assert!(!m.get_entry("cid1").unwrap().pinned);
    }

    #[test]
    fn test_pin_returns_true_when_found() {
        let mut m = StorageBlockManifest::new();
        m.add_entry(make_entry("cid1", 100, "raw", 1000, false));
        assert!(m.pin("cid1"));
    }

    #[test]
    fn test_unpin_returns_true_when_found() {
        let mut m = StorageBlockManifest::new();
        m.add_entry(make_entry("cid1", 100, "raw", 1000, true));
        assert!(m.unpin("cid1"));
    }

    // ------------------------------------------------------------------
    // filter
    // ------------------------------------------------------------------

    #[test]
    fn test_filter_all_returns_all() {
        let mut m = StorageBlockManifest::new();
        m.add_entry(make_entry("cid1", 100, "raw", 1000, false));
        m.add_entry(make_entry("cid2", 200, "dag-cbor", 2000, true));
        assert_eq!(m.filter(&ManifestFilter::All).len(), 2);
    }

    #[test]
    fn test_filter_pinned_only() {
        let mut m = StorageBlockManifest::new();
        m.add_entry(make_entry("cid1", 100, "raw", 1000, false));
        m.add_entry(make_entry("cid2", 200, "dag-cbor", 2000, true));
        let result = m.filter(&ManifestFilter::PinnedOnly);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].cid, "cid2");
    }

    #[test]
    fn test_filter_content_type() {
        let mut m = StorageBlockManifest::new();
        m.add_entry(make_entry("cid1", 100, "raw", 1000, false));
        m.add_entry(make_entry("cid2", 200, "dag-cbor", 2000, false));
        m.add_entry(make_entry("cid3", 300, "raw", 3000, false));
        let result = m.filter(&ManifestFilter::ContentType("raw".to_string()));
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_filter_added_after() {
        let mut m = StorageBlockManifest::new();
        m.add_entry(make_entry("cid1", 100, "raw", 1000, false));
        m.add_entry(make_entry("cid2", 200, "raw", 2000, false));
        m.add_entry(make_entry("cid3", 300, "raw", 3000, false));
        let result = m.filter(&ManifestFilter::AddedAfter(1500));
        assert_eq!(result.len(), 2);
        for e in &result {
            assert!(e.added_at_secs > 1500);
        }
    }

    #[test]
    fn test_filter_size_above() {
        let mut m = StorageBlockManifest::new();
        m.add_entry(make_entry("cid1", 100, "raw", 1000, false));
        m.add_entry(make_entry("cid2", 500, "raw", 2000, false));
        m.add_entry(make_entry("cid3", 1000, "raw", 3000, false));
        let result = m.filter(&ManifestFilter::SizeAbove(400));
        assert_eq!(result.len(), 2);
        for e in &result {
            assert!(e.size_bytes > 400);
        }
    }

    #[test]
    fn test_filter_sorted_by_added_at_secs_asc() {
        let mut m = StorageBlockManifest::new();
        // Add in reverse order to ensure sorting is actually performed.
        m.add_entry(make_entry("cid3", 300, "raw", 3000, false));
        m.add_entry(make_entry("cid1", 100, "raw", 1000, false));
        m.add_entry(make_entry("cid2", 200, "raw", 2000, false));
        let result = m.filter(&ManifestFilter::All);
        let timestamps: Vec<u64> = result.iter().map(|e| e.added_at_secs).collect();
        assert_eq!(timestamps, vec![1000, 2000, 3000]);
    }

    // ------------------------------------------------------------------
    // stats
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_total_entries() {
        let mut m = StorageBlockManifest::new();
        m.add_entry(make_entry("cid1", 100, "raw", 1000, false));
        m.add_entry(make_entry("cid2", 200, "raw", 2000, false));
        assert_eq!(m.stats().total_entries, 2);
    }

    #[test]
    fn test_stats_total_size_bytes() {
        let mut m = StorageBlockManifest::new();
        m.add_entry(make_entry("cid1", 100, "raw", 1000, false));
        m.add_entry(make_entry("cid2", 200, "raw", 2000, false));
        assert_eq!(m.stats().total_size_bytes, 300);
    }

    #[test]
    fn test_stats_pinned_count() {
        let mut m = StorageBlockManifest::new();
        m.add_entry(make_entry("cid1", 100, "raw", 1000, true));
        m.add_entry(make_entry("cid2", 200, "raw", 2000, false));
        m.add_entry(make_entry("cid3", 300, "raw", 3000, true));
        assert_eq!(m.stats().pinned_count, 2);
    }

    #[test]
    fn test_stats_content_type_counts() {
        let mut m = StorageBlockManifest::new();
        m.add_entry(make_entry("cid1", 100, "raw", 1000, false));
        m.add_entry(make_entry("cid2", 200, "raw", 2000, false));
        m.add_entry(make_entry("cid3", 300, "dag-cbor", 3000, false));
        let stats = m.stats();
        assert_eq!(*stats.content_type_counts.get("raw").unwrap_or(&0), 2);
        assert_eq!(*stats.content_type_counts.get("dag-cbor").unwrap_or(&0), 1);
    }

    // ------------------------------------------------------------------
    // merge
    // ------------------------------------------------------------------

    #[test]
    fn test_merge_adds_missing_entries() {
        let mut m1 = StorageBlockManifest::new();
        m1.add_entry(make_entry("cid1", 100, "raw", 1000, false));

        let mut m2 = StorageBlockManifest::new();
        m2.add_entry(make_entry("cid2", 200, "raw", 2000, false));

        m1.merge(&m2);
        assert!(m1.get_entry("cid2").is_some());
        assert_eq!(m1.entries.len(), 2);
    }

    #[test]
    fn test_merge_propagates_pin_from_other() {
        let mut m1 = StorageBlockManifest::new();
        m1.add_entry(make_entry("cid1", 100, "raw", 1000, false));

        let mut m2 = StorageBlockManifest::new();
        m2.add_entry(make_entry("cid1", 100, "raw", 1000, true));

        m1.merge(&m2);
        assert!(m1.get_entry("cid1").unwrap().pinned);
    }

    #[test]
    fn test_merge_does_not_unpin_existing() {
        let mut m1 = StorageBlockManifest::new();
        m1.add_entry(make_entry("cid1", 100, "raw", 1000, true));

        let mut m2 = StorageBlockManifest::new();
        m2.add_entry(make_entry("cid1", 100, "raw", 1000, false));

        m1.merge(&m2);
        // Pin should not be cleared by a non-pinned other entry.
        assert!(m1.get_entry("cid1").unwrap().pinned);
    }

    // ------------------------------------------------------------------
    // export_cids
    // ------------------------------------------------------------------

    #[test]
    fn test_export_cids_sorted() {
        let mut m = StorageBlockManifest::new();
        m.add_entry(make_entry("cidZ", 100, "raw", 1000, false));
        m.add_entry(make_entry("cidA", 200, "raw", 2000, false));
        m.add_entry(make_entry("cidM", 300, "raw", 3000, false));
        let cids = m.export_cids();
        assert_eq!(cids, vec!["cidA", "cidM", "cidZ"]);
    }

    // ------------------------------------------------------------------
    // verify_checksums
    // ------------------------------------------------------------------

    #[test]
    fn test_verify_checksums_correct_entries_pass() {
        let mut m = StorageBlockManifest::new();
        m.add_entry(make_entry("cid1", 100, "raw", 1000, false));
        m.add_entry(make_entry("cid2", 200, "raw", 2000, false));
        assert!(m.verify_checksums().is_empty());
    }

    #[test]
    fn test_verify_checksums_detects_mismatch() {
        let mut m = StorageBlockManifest::new();
        let mut bad = make_entry("cid1", 100, "raw", 1000, false);
        bad.checksum = bad.checksum.wrapping_add(1); // corrupt
        m.add_entry(bad);
        m.add_entry(make_entry("cid2", 200, "raw", 2000, false));
        let mismatches = m.verify_checksums();
        assert_eq!(mismatches, vec!["cid1"]);
    }

    // ------------------------------------------------------------------
    // fnv1a helper
    // ------------------------------------------------------------------

    #[test]
    fn test_fnv1a_empty_input() {
        // FNV-1a of empty input is the offset basis.
        assert_eq!(fnv1a(b""), 14_695_981_039_346_656_037);
    }

    #[test]
    fn test_fnv1a_deterministic() {
        let h1 = fnv1a(b"hello");
        let h2 = fnv1a(b"hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fnv1a_different_inputs() {
        assert_ne!(fnv1a(b"foo"), fnv1a(b"bar"));
    }
}
