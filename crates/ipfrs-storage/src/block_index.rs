//! Secondary index over stored blocks for accelerated queries.
//!
//! Provides [`StorageBlockIndex`] which maintains an in-memory inverted index
//! keyed by content type, size bucket (MB granularity), day bucket, and custom
//! tags. Queries filter across any combination of those dimensions without
//! scanning every stored entry.

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────────

/// Bytes per MB (1 MiB = 1_048_576 bytes).
const BYTES_PER_MB: u64 = 1_048_576;

/// Seconds per day.
const SECS_PER_DAY: u64 = 86_400;

// ──────────────────────────────────────────────────────────────────────────────
// IndexKey
// ──────────────────────────────────────────────────────────────────────────────

/// Discriminated key used to address a single bucket in the inverted index.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum IndexKey {
    /// Exact content-type string (e.g. `"image/png"`).
    ContentType(String),
    /// 1-MiB size bucket: `bucket = size_bytes / 1_048_576`.
    SizeBucket(u64),
    /// Day bucket: `bucket = created_at_secs / 86400`.
    DayBucket(u64),
    /// Arbitrary tag string.
    Tag(String),
}

// ──────────────────────────────────────────────────────────────────────────────
// IndexEntry
// ──────────────────────────────────────────────────────────────────────────────

/// Metadata record stored for each block in the index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexEntry {
    /// Content identifier (CID) string.
    pub cid: String,
    /// Serialised size of the block in bytes.
    pub size_bytes: u64,
    /// MIME-style content type string.
    pub content_type: String,
    /// UNIX timestamp (seconds) at which the block was created.
    pub created_at_secs: u64,
    /// Arbitrary user-supplied tags.
    pub tags: Vec<String>,
}

impl IndexEntry {
    /// Derive the [`IndexKey::SizeBucket`] for this entry.
    #[inline]
    pub fn size_bucket(&self) -> IndexKey {
        IndexKey::SizeBucket(self.size_bytes / BYTES_PER_MB)
    }

    /// Derive the [`IndexKey::DayBucket`] for this entry.
    #[inline]
    pub fn day_bucket(&self) -> IndexKey {
        IndexKey::DayBucket(self.created_at_secs / SECS_PER_DAY)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IndexQuery
// ──────────────────────────────────────────────────────────────────────────────

/// Filter specification for [`StorageBlockIndex::query`].
///
/// All fields are optional; only the `Some` variants are applied. Multiple
/// active filters are ANDed together.
#[derive(Clone, Debug, Default)]
pub struct IndexQuery {
    /// Exact content-type match.
    pub content_type: Option<String>,
    /// Lower bound on `size_bytes` (inclusive).
    pub min_size_bytes: Option<u64>,
    /// Upper bound on `size_bytes` (inclusive).
    pub max_size_bytes: Option<u64>,
    /// Require `created_at_secs` strictly greater than this value.
    pub created_after_secs: Option<u64>,
    /// Require the entry's tag list to contain this exact tag.
    pub tag: Option<String>,
}

// ──────────────────────────────────────────────────────────────────────────────
// IndexStats
// ──────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics about the current index state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexStats {
    /// Total number of indexed entries.
    pub total_entries: usize,
    /// Number of distinct content-type strings observed.
    pub unique_content_types: usize,
    /// Number of distinct tag strings observed.
    pub unique_tags: usize,
    /// Sum of `size_bytes` across all entries.
    pub total_size_bytes: u64,
}

// ──────────────────────────────────────────────────────────────────────────────
// StorageBlockIndex
// ──────────────────────────────────────────────────────────────────────────────

/// Secondary index over stored blocks that accelerates queries by content type,
/// size range, creation time, and custom tags without scanning all blocks.
///
/// # Indexing strategy
///
/// Every [`IndexEntry`] inserted into the index is registered under four
/// categories of [`IndexKey`]:
///
/// - `ContentType(content_type)` — one key per entry.
/// - `SizeBucket(size_bytes / 1_MiB)` — coarse size bucket (1 MiB granularity).
/// - `DayBucket(created_at_secs / 86_400)` — coarse day bucket.
/// - `Tag(tag)` for each tag string in `entry.tags`.
///
/// The inverted index maps each [`IndexKey`] to the list of CID strings that
/// carry that key, enabling fast set-based retrieval without a full scan.
pub struct StorageBlockIndex {
    /// Primary storage: maps CID to its full [`IndexEntry`].
    pub entries: HashMap<String, IndexEntry>,
    /// Inverted index: maps an [`IndexKey`] to the CIDs that belong to it.
    pub index: HashMap<IndexKey, Vec<String>>,
}

impl StorageBlockIndex {
    /// Create a new, empty [`StorageBlockIndex`].
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            index: HashMap::new(),
        }
    }

    // ── insert ────────────────────────────────────────────────────────────────

    /// Insert (or replace) an entry in the index.
    ///
    /// If a previous entry exists for the same CID it is removed first so the
    /// inverted index does not accumulate stale pointers.
    pub fn insert(&mut self, entry: IndexEntry) {
        // Remove any existing entry for this CID to keep the index consistent.
        self.remove(&entry.cid.clone());

        let cid = entry.cid.clone();

        // Build the set of index keys this entry should appear under.
        let keys = Self::keys_for_entry(&entry);

        // Store primary record.
        self.entries.insert(cid.clone(), entry);

        // Update inverted index.
        for key in keys {
            self.index.entry(key).or_default().push(cid.clone());
        }
    }

    // ── remove ────────────────────────────────────────────────────────────────

    /// Remove an entry by CID from both the primary store and all index buckets.
    ///
    /// Returns `true` when the entry existed and was removed, `false` otherwise.
    pub fn remove(&mut self, cid: &str) -> bool {
        let entry = match self.entries.remove(cid) {
            Some(e) => e,
            None => return false,
        };

        let keys = Self::keys_for_entry(&entry);

        for key in keys {
            if let Some(cids) = self.index.get_mut(&key) {
                cids.retain(|c| c != cid);
                // Leave empty vecs in place; they are harmless and avoid
                // repeated re-allocation on future inserts.
            }
        }

        true
    }

    // ── query ─────────────────────────────────────────────────────────────────

    /// Query the index with an [`IndexQuery`] filter.
    ///
    /// The implementation performs a full scan of the primary entry map and
    /// applies each active filter in turn.  Results are sorted by
    /// `created_at_secs` descending (newest first).
    pub fn query<'a>(&'a self, q: &IndexQuery) -> Vec<&'a IndexEntry> {
        let mut results: Vec<&IndexEntry> = self
            .entries
            .values()
            .filter(|e| {
                if let Some(ref ct) = q.content_type {
                    if &e.content_type != ct {
                        return false;
                    }
                }
                if let Some(min) = q.min_size_bytes {
                    if e.size_bytes < min {
                        return false;
                    }
                }
                if let Some(max) = q.max_size_bytes {
                    if e.size_bytes > max {
                        return false;
                    }
                }
                if let Some(after) = q.created_after_secs {
                    if e.created_at_secs <= after {
                        return false;
                    }
                }
                if let Some(ref tag) = q.tag {
                    if !e.tags.contains(tag) {
                        return false;
                    }
                }
                true
            })
            .collect();

        // Sort newest first, then by CID for determinism among ties.
        results.sort_by(|a, b| {
            b.created_at_secs
                .cmp(&a.created_at_secs)
                .then_with(|| a.cid.cmp(&b.cid))
        });

        results
    }

    // ── entries_for_key ───────────────────────────────────────────────────────

    /// Look up all entries that are indexed under the given [`IndexKey`].
    ///
    /// Returns entries sorted by CID string for deterministic ordering.
    pub fn entries_for_key<'a>(&'a self, key: &IndexKey) -> Vec<&'a IndexEntry> {
        let cids = match self.index.get(key) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let mut entries: Vec<&IndexEntry> = cids
            .iter()
            .filter_map(|cid| self.entries.get(cid.as_str()))
            .collect();

        entries.sort_by(|a, b| a.cid.cmp(&b.cid));
        entries
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    /// Compute aggregate statistics over all indexed entries.
    pub fn stats(&self) -> IndexStats {
        let total_entries = self.entries.len();

        let mut content_types = std::collections::HashSet::new();
        let mut tags = std::collections::HashSet::new();
        let mut total_size_bytes: u64 = 0;

        for entry in self.entries.values() {
            content_types.insert(entry.content_type.as_str());
            for tag in &entry.tags {
                tags.insert(tag.as_str());
            }
            total_size_bytes = total_size_bytes.saturating_add(entry.size_bytes);
        }

        IndexStats {
            total_entries,
            unique_content_types: content_types.len(),
            unique_tags: tags.len(),
            total_size_bytes,
        }
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    /// Derive all [`IndexKey`]s that an entry should be registered under.
    fn keys_for_entry(entry: &IndexEntry) -> Vec<IndexKey> {
        let mut keys = Vec::with_capacity(3 + entry.tags.len());
        keys.push(IndexKey::ContentType(entry.content_type.clone()));
        keys.push(entry.size_bucket());
        keys.push(entry.day_bucket());
        for tag in &entry.tags {
            keys.push(IndexKey::Tag(tag.clone()));
        }
        keys
    }
}

impl Default for StorageBlockIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// BlockIndexEntry
// ──────────────────────────────────────────────────────────────────────────────

/// A block entry for the secondary block index, keyed by codec, tags, size, and
/// creation tick.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockIndexEntry {
    /// Content identifier (CID) string.
    pub cid: String,
    /// Size of the block in bytes.
    pub size_bytes: u64,
    /// Codec name, e.g. `"dag-cbor"`, `"raw"`.
    pub codec: String,
    /// Monotonic tick at which the block was created (logical clock).
    pub created_tick: u64,
    /// Arbitrary tags attached to this block.
    pub tags: Vec<String>,
}

// ──────────────────────────────────────────────────────────────────────────────
// BlockIndexStats
// ──────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics for a [`SecondaryBlockIndex`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockIndexStats {
    /// Number of entries currently in the index.
    pub entry_count: usize,
    /// Sum of `size_bytes` across all entries.
    pub total_bytes: u64,
    /// Number of distinct codec strings.
    pub unique_codecs: usize,
    /// Number of distinct tag strings.
    pub unique_tags: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// SecondaryBlockIndex
// ──────────────────────────────────────────────────────────────────────────────

/// Secondary index for fast block lookups by codec, tag, size range, and
/// creation-tick range.
///
/// Maintains inverted indices from codec and tag to CID lists so that
/// lookups by those dimensions are O(bucket-size) instead of O(n).
/// Size-range and tick-range queries use linear filtering over the primary
/// entry map.
pub struct SecondaryBlockIndex {
    /// Primary storage: CID → entry.
    entries: HashMap<String, BlockIndexEntry>,
    /// Inverted index: codec → \[CID\].
    by_codec: HashMap<String, Vec<String>>,
    /// Inverted index: tag → \[CID\].
    by_tag: HashMap<String, Vec<String>>,
    /// Running total of `size_bytes` across all entries.
    total_bytes: u64,
}

impl SecondaryBlockIndex {
    /// Create a new, empty [`SecondaryBlockIndex`].
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            by_codec: HashMap::new(),
            by_tag: HashMap::new(),
            total_bytes: 0,
        }
    }

    // ── insert ────────────────────────────────────────────────────────────────

    /// Insert a [`BlockIndexEntry`] into the index.
    ///
    /// If an entry with the same CID already exists it is removed first so
    /// that all inverted indices stay consistent.
    pub fn insert(&mut self, entry: BlockIndexEntry) {
        // Remove stale entry if present.
        let cid_clone = entry.cid.clone();
        self.remove(&cid_clone);

        // Update running total.
        self.total_bytes = self.total_bytes.saturating_add(entry.size_bytes);

        // Update codec index.
        self.by_codec
            .entry(entry.codec.clone())
            .or_default()
            .push(entry.cid.clone());

        // Update tag index.
        for tag in &entry.tags {
            self.by_tag
                .entry(tag.clone())
                .or_default()
                .push(entry.cid.clone());
        }

        // Store primary record.
        self.entries.insert(entry.cid.clone(), entry);
    }

    // ── remove ────────────────────────────────────────────────────────────────

    /// Remove an entry by CID from all indices.
    ///
    /// Returns `Some(entry)` if found, `None` otherwise.
    pub fn remove(&mut self, cid: &str) -> Option<BlockIndexEntry> {
        let entry = self.entries.remove(cid)?;

        // Subtract from running total.
        self.total_bytes = self.total_bytes.saturating_sub(entry.size_bytes);

        // Clean codec index.
        if let Some(cids) = self.by_codec.get_mut(&entry.codec) {
            cids.retain(|c| c != cid);
        }

        // Clean tag index.
        for tag in &entry.tags {
            if let Some(cids) = self.by_tag.get_mut(tag) {
                cids.retain(|c| c != cid);
            }
        }

        Some(entry)
    }

    // ── get ───────────────────────────────────────────────────────────────────

    /// Look up an entry by CID.
    pub fn get(&self, cid: &str) -> Option<&BlockIndexEntry> {
        self.entries.get(cid)
    }

    // ── find_by_codec ─────────────────────────────────────────────────────────

    /// Return all entries that use the given codec, sorted by CID for
    /// deterministic ordering.
    pub fn find_by_codec(&self, codec: &str) -> Vec<&BlockIndexEntry> {
        let cids = match self.by_codec.get(codec) {
            Some(v) => v,
            None => return Vec::new(),
        };
        let mut results: Vec<&BlockIndexEntry> = cids
            .iter()
            .filter_map(|c| self.entries.get(c.as_str()))
            .collect();
        results.sort_by(|a, b| a.cid.cmp(&b.cid));
        results
    }

    // ── find_by_tag ───────────────────────────────────────────────────────────

    /// Return all entries that carry the given tag, sorted by CID.
    pub fn find_by_tag(&self, tag: &str) -> Vec<&BlockIndexEntry> {
        let cids = match self.by_tag.get(tag) {
            Some(v) => v,
            None => return Vec::new(),
        };
        let mut results: Vec<&BlockIndexEntry> = cids
            .iter()
            .filter_map(|c| self.entries.get(c.as_str()))
            .collect();
        results.sort_by(|a, b| a.cid.cmp(&b.cid));
        results
    }

    // ── find_by_size_range ────────────────────────────────────────────────────

    /// Return all entries whose `size_bytes` is within `[min, max]`
    /// (inclusive), sorted by CID.
    pub fn find_by_size_range(&self, min: u64, max: u64) -> Vec<&BlockIndexEntry> {
        let mut results: Vec<&BlockIndexEntry> = self
            .entries
            .values()
            .filter(|e| e.size_bytes >= min && e.size_bytes <= max)
            .collect();
        results.sort_by(|a, b| a.cid.cmp(&b.cid));
        results
    }

    // ── find_by_created_range ─────────────────────────────────────────────────

    /// Return all entries whose `created_tick` is within `[min_tick, max_tick]`
    /// (inclusive), sorted by CID.
    pub fn find_by_created_range(&self, min_tick: u64, max_tick: u64) -> Vec<&BlockIndexEntry> {
        let mut results: Vec<&BlockIndexEntry> = self
            .entries
            .values()
            .filter(|e| e.created_tick >= min_tick && e.created_tick <= max_tick)
            .collect();
        results.sort_by(|a, b| a.cid.cmp(&b.cid));
        results
    }

    // ── entry_count ───────────────────────────────────────────────────────────

    /// Number of entries currently in the index.
    #[inline]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    // ── total_bytes ───────────────────────────────────────────────────────────

    /// Cumulative size in bytes of all indexed blocks.
    #[inline]
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    // ── unique_codecs ─────────────────────────────────────────────────────────

    /// Return all distinct codec strings, sorted alphabetically.
    pub fn unique_codecs(&self) -> Vec<String> {
        let mut codecs: Vec<String> = self
            .by_codec
            .iter()
            .filter(|(_, cids)| !cids.is_empty())
            .map(|(codec, _)| codec.clone())
            .collect();
        codecs.sort();
        codecs
    }

    // ── unique_tags ───────────────────────────────────────────────────────────

    /// Return all distinct tag strings, sorted alphabetically.
    pub fn unique_tags(&self) -> Vec<String> {
        let mut tags: Vec<String> = self
            .by_tag
            .iter()
            .filter(|(_, cids)| !cids.is_empty())
            .map(|(tag, _)| tag.clone())
            .collect();
        tags.sort();
        tags
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    /// Compute aggregate statistics.
    pub fn stats(&self) -> BlockIndexStats {
        BlockIndexStats {
            entry_count: self.entry_count(),
            total_bytes: self.total_bytes,
            unique_codecs: self.unique_codecs().len(),
            unique_tags: self.unique_tags().len(),
        }
    }
}

impl Default for SecondaryBlockIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_entry(
        cid: &str,
        content_type: &str,
        size_bytes: u64,
        created_at_secs: u64,
        tags: &[&str],
    ) -> IndexEntry {
        IndexEntry {
            cid: cid.to_owned(),
            content_type: content_type.to_owned(),
            size_bytes,
            created_at_secs,
            tags: tags.iter().map(|t| t.to_string()).collect(),
        }
    }

    /// Populate an index with a small, varied set of entries.
    fn populated_index() -> StorageBlockIndex {
        let mut idx = StorageBlockIndex::new();
        // entry A: image, 2 MiB, day 0, tags: ["public"]
        idx.insert(make_entry(
            "cid-a",
            "image/png",
            2 * BYTES_PER_MB,
            0,
            &["public"],
        ));
        // entry B: video, 10 MiB, day 1, tags: ["public", "featured"]
        idx.insert(make_entry(
            "cid-b",
            "video/mp4",
            10 * BYTES_PER_MB,
            SECS_PER_DAY,
            &["public", "featured"],
        ));
        // entry C: image, 500 KiB, day 2, tags: ["private"]
        idx.insert(make_entry(
            "cid-c",
            "image/png",
            512_000,
            2 * SECS_PER_DAY,
            &["private"],
        ));
        // entry D: text, 1 byte, day 3, no tags
        idx.insert(make_entry("cid-d", "text/plain", 1, 3 * SECS_PER_DAY, &[]));
        idx
    }

    // ── 1. new() starts empty ─────────────────────────────────────────────────

    #[test]
    fn test_new_is_empty() {
        let idx = StorageBlockIndex::new();
        assert!(idx.entries.is_empty());
        assert!(idx.index.is_empty());
    }

    // ── 2. insert stores entry ────────────────────────────────────────────────

    #[test]
    fn test_insert_stores_entry() {
        let mut idx = StorageBlockIndex::new();
        let e = make_entry("cid-1", "text/plain", 100, 1000, &[]);
        idx.insert(e.clone());
        assert_eq!(idx.entries.get("cid-1"), Some(&e));
    }

    // ── 3. insert updates all index buckets ──────────────────────────────────

    #[test]
    fn test_insert_updates_content_type_index() {
        let mut idx = StorageBlockIndex::new();
        idx.insert(make_entry("cid-1", "text/plain", 100, 0, &[]));
        let key = IndexKey::ContentType("text/plain".to_owned());
        assert!(idx
            .index
            .get(&key)
            .is_some_and(|v| v.contains(&"cid-1".to_owned())));
    }

    #[test]
    fn test_insert_updates_size_bucket_index() {
        let mut idx = StorageBlockIndex::new();
        // 3 MiB → bucket 3
        idx.insert(make_entry("cid-1", "text/plain", 3 * BYTES_PER_MB, 0, &[]));
        let key = IndexKey::SizeBucket(3);
        assert!(idx
            .index
            .get(&key)
            .is_some_and(|v| v.contains(&"cid-1".to_owned())));
    }

    #[test]
    fn test_insert_updates_day_bucket_index() {
        let mut idx = StorageBlockIndex::new();
        // day 5
        idx.insert(make_entry(
            "cid-1",
            "text/plain",
            1,
            5 * SECS_PER_DAY + 3600,
            &[],
        ));
        let key = IndexKey::DayBucket(5);
        assert!(idx
            .index
            .get(&key)
            .is_some_and(|v| v.contains(&"cid-1".to_owned())));
    }

    #[test]
    fn test_insert_updates_tag_indexes() {
        let mut idx = StorageBlockIndex::new();
        idx.insert(make_entry("cid-1", "text/plain", 1, 0, &["alpha", "beta"]));
        assert!(idx
            .index
            .get(&IndexKey::Tag("alpha".to_owned()))
            .is_some_and(|v| v.contains(&"cid-1".to_owned())));
        assert!(idx
            .index
            .get(&IndexKey::Tag("beta".to_owned()))
            .is_some_and(|v| v.contains(&"cid-1".to_owned())));
    }

    // ── 4. remove removes from entries ───────────────────────────────────────

    #[test]
    fn test_remove_removes_from_entries() {
        let mut idx = populated_index();
        assert!(idx.remove("cid-a"));
        assert!(!idx.entries.contains_key("cid-a"));
    }

    // ── 5. remove cleans all index buckets ───────────────────────────────────

    #[test]
    fn test_remove_cleans_content_type_bucket() {
        let mut idx = populated_index();
        idx.remove("cid-a");
        let key = IndexKey::ContentType("image/png".to_owned());
        // cid-c is still an image/png; cid-a should be gone
        let cids = idx.index.get(&key).cloned().unwrap_or_default();
        assert!(!cids.contains(&"cid-a".to_owned()));
    }

    #[test]
    fn test_remove_cleans_size_bucket() {
        let mut idx = populated_index();
        // cid-a is 2 MiB → bucket 2
        idx.remove("cid-a");
        let key = IndexKey::SizeBucket(2);
        let cids = idx.index.get(&key).cloned().unwrap_or_default();
        assert!(!cids.contains(&"cid-a".to_owned()));
    }

    #[test]
    fn test_remove_cleans_tag_bucket() {
        let mut idx = populated_index();
        idx.remove("cid-a");
        let key = IndexKey::Tag("public".to_owned());
        let cids = idx.index.get(&key).cloned().unwrap_or_default();
        assert!(!cids.contains(&"cid-a".to_owned()));
        // cid-b also has "public"
        assert!(cids.contains(&"cid-b".to_owned()));
    }

    // ── 6. remove returns false for unknown cid ───────────────────────────────

    #[test]
    fn test_remove_unknown_cid_returns_false() {
        let mut idx = StorageBlockIndex::new();
        assert!(!idx.remove("nonexistent"));
    }

    // ── 7. query content_type filter ─────────────────────────────────────────

    #[test]
    fn test_query_content_type_filter() {
        let idx = populated_index();
        let q = IndexQuery {
            content_type: Some("image/png".to_owned()),
            ..Default::default()
        };
        let results = idx.query(&q);
        assert_eq!(results.len(), 2);
        for e in &results {
            assert_eq!(e.content_type, "image/png");
        }
    }

    // ── 8. query min_size filter ──────────────────────────────────────────────

    #[test]
    fn test_query_min_size_filter() {
        let idx = populated_index();
        let q = IndexQuery {
            min_size_bytes: Some(5 * BYTES_PER_MB),
            ..Default::default()
        };
        let results = idx.query(&q);
        // Only cid-b (10 MiB) qualifies
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].cid, "cid-b");
    }

    // ── 9. query max_size filter ──────────────────────────────────────────────

    #[test]
    fn test_query_max_size_filter() {
        let idx = populated_index();
        let q = IndexQuery {
            max_size_bytes: Some(BYTES_PER_MB),
            ..Default::default()
        };
        let results = idx.query(&q);
        // cid-c (512 000) and cid-d (1) qualify
        assert_eq!(results.len(), 2);
        for e in &results {
            assert!(e.size_bytes <= BYTES_PER_MB);
        }
    }

    // ── 10. query size range (min + max) ──────────────────────────────────────

    #[test]
    fn test_query_size_range() {
        let idx = populated_index();
        let q = IndexQuery {
            min_size_bytes: Some(BYTES_PER_MB),
            max_size_bytes: Some(5 * BYTES_PER_MB),
            ..Default::default()
        };
        let results = idx.query(&q);
        // Only cid-a (2 MiB) falls in [1 MiB, 5 MiB]
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].cid, "cid-a");
    }

    // ── 11. query created_after filter ───────────────────────────────────────

    #[test]
    fn test_query_created_after_filter() {
        let idx = populated_index();
        // strictly after day 1 → days 2 and 3
        let q = IndexQuery {
            created_after_secs: Some(SECS_PER_DAY),
            ..Default::default()
        };
        let results = idx.query(&q);
        assert_eq!(results.len(), 2);
        for e in &results {
            assert!(e.created_at_secs > SECS_PER_DAY);
        }
    }

    // ── 12. query tag filter ──────────────────────────────────────────────────

    #[test]
    fn test_query_tag_filter() {
        let idx = populated_index();
        let q = IndexQuery {
            tag: Some("featured".to_owned()),
            ..Default::default()
        };
        let results = idx.query(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].cid, "cid-b");
    }

    // ── 13. query multiple filters combined ──────────────────────────────────

    #[test]
    fn test_query_combined_filters() {
        let idx = populated_index();
        let q = IndexQuery {
            content_type: Some("image/png".to_owned()),
            max_size_bytes: Some(BYTES_PER_MB),
            ..Default::default()
        };
        let results = idx.query(&q);
        // Only cid-c is image/png AND <= 1 MiB
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].cid, "cid-c");
    }

    // ── 14. query returns sorted by created_at_secs desc ─────────────────────

    #[test]
    fn test_query_sorted_by_created_at_desc() {
        let idx = populated_index();
        let results = idx.query(&IndexQuery::default());
        assert_eq!(results.len(), 4);
        let times: Vec<u64> = results.iter().map(|e| e.created_at_secs).collect();
        // Must be non-increasing
        for i in 1..times.len() {
            assert!(
                times[i - 1] >= times[i],
                "Expected descending order at index {i}: {times:?}"
            );
        }
    }

    // ── 15. query returns empty for no match ──────────────────────────────────

    #[test]
    fn test_query_no_match_returns_empty() {
        let idx = populated_index();
        let q = IndexQuery {
            content_type: Some("application/octet-stream".to_owned()),
            ..Default::default()
        };
        assert!(idx.query(&q).is_empty());
    }

    // ── 16. entries_for_key ContentType ──────────────────────────────────────

    #[test]
    fn test_entries_for_key_content_type() {
        let idx = populated_index();
        let key = IndexKey::ContentType("image/png".to_owned());
        let entries = idx.entries_for_key(&key);
        assert_eq!(entries.len(), 2);
        let cids: Vec<&str> = entries.iter().map(|e| e.cid.as_str()).collect();
        assert!(cids.contains(&"cid-a"));
        assert!(cids.contains(&"cid-c"));
    }

    // ── 17. entries_for_key Tag ───────────────────────────────────────────────

    #[test]
    fn test_entries_for_key_tag() {
        let idx = populated_index();
        let key = IndexKey::Tag("public".to_owned());
        let entries = idx.entries_for_key(&key);
        assert_eq!(entries.len(), 2);
        let cids: Vec<&str> = entries.iter().map(|e| e.cid.as_str()).collect();
        assert!(cids.contains(&"cid-a"));
        assert!(cids.contains(&"cid-b"));
    }

    // ── 18. entries_for_key SizeBucket ───────────────────────────────────────

    #[test]
    fn test_entries_for_key_size_bucket() {
        let idx = populated_index();
        // cid-a is exactly 2 MiB → bucket 2
        let key = IndexKey::SizeBucket(2);
        let entries = idx.entries_for_key(&key);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].cid, "cid-a");
    }

    // ── 19. entries_for_key DayBucket ────────────────────────────────────────

    #[test]
    fn test_entries_for_key_day_bucket() {
        let idx = populated_index();
        // cid-b was created at exactly SECS_PER_DAY → day bucket 1
        let key = IndexKey::DayBucket(1);
        let entries = idx.entries_for_key(&key);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].cid, "cid-b");
    }

    // ── 20. stats total_entries ───────────────────────────────────────────────

    #[test]
    fn test_stats_total_entries() {
        let idx = populated_index();
        assert_eq!(idx.stats().total_entries, 4);
    }

    // ── 21. stats unique_content_types ───────────────────────────────────────

    #[test]
    fn test_stats_unique_content_types() {
        let idx = populated_index();
        // image/png, video/mp4, text/plain → 3
        assert_eq!(idx.stats().unique_content_types, 3);
    }

    // ── 22. stats total_size_bytes ────────────────────────────────────────────

    #[test]
    fn test_stats_total_size_bytes() {
        let idx = populated_index();
        let expected = 2 * BYTES_PER_MB   // cid-a
            + 10 * BYTES_PER_MB           // cid-b
            + 512_000                     // cid-c
            + 1; // cid-d
        assert_eq!(idx.stats().total_size_bytes, expected);
    }

    // ── bonus: stats unique_tags ──────────────────────────────────────────────

    #[test]
    fn test_stats_unique_tags() {
        let idx = populated_index();
        // public, featured, private → 3
        assert_eq!(idx.stats().unique_tags, 3);
    }

    // ── bonus: default() is same as new() ────────────────────────────────────

    #[test]
    fn test_default_is_empty() {
        let idx = StorageBlockIndex::default();
        assert!(idx.entries.is_empty());
    }

    // ── bonus: re-insert same cid replaces entry ──────────────────────────────

    #[test]
    fn test_reinsert_replaces_entry() {
        let mut idx = StorageBlockIndex::new();
        idx.insert(make_entry("cid-1", "text/plain", 100, 1000, &["old"]));
        idx.insert(make_entry("cid-1", "image/png", 200, 2000, &["new"]));

        assert_eq!(idx.entries.len(), 1);
        let e = idx.entries.get("cid-1").expect("entry must exist");
        assert_eq!(e.content_type, "image/png");

        // Old content-type bucket must no longer contain cid-1
        let old_key = IndexKey::ContentType("text/plain".to_owned());
        let old_cids = idx.index.get(&old_key).cloned().unwrap_or_default();
        assert!(!old_cids.contains(&"cid-1".to_owned()));

        // Old tag bucket must no longer contain cid-1
        let old_tag = IndexKey::Tag("old".to_owned());
        let old_tag_cids = idx.index.get(&old_tag).cloned().unwrap_or_default();
        assert!(!old_tag_cids.contains(&"cid-1".to_owned()));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SecondaryBlockIndex tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod secondary_tests {
    use super::*;

    fn entry(cid: &str, size: u64, codec: &str, tick: u64, tags: &[&str]) -> BlockIndexEntry {
        BlockIndexEntry {
            cid: cid.to_owned(),
            size_bytes: size,
            codec: codec.to_owned(),
            created_tick: tick,
            tags: tags.iter().map(|t| t.to_string()).collect(),
        }
    }

    fn populated() -> SecondaryBlockIndex {
        let mut idx = SecondaryBlockIndex::new();
        idx.insert(entry("cid-a", 1024, "raw", 10, &["pin"]));
        idx.insert(entry("cid-b", 2048, "dag-cbor", 20, &["pin", "important"]));
        idx.insert(entry("cid-c", 512, "raw", 30, &["temp"]));
        idx.insert(entry("cid-d", 4096, "dag-pb", 40, &[]));
        idx
    }

    // ── 1. new is empty ──────────────────────────────────────────────────────

    #[test]
    fn test_secondary_new_is_empty() {
        let idx = SecondaryBlockIndex::new();
        assert_eq!(idx.entry_count(), 0);
        assert_eq!(idx.total_bytes(), 0);
    }

    // ── 2. default is empty ──────────────────────────────────────────────────

    #[test]
    fn test_secondary_default_is_empty() {
        let idx = SecondaryBlockIndex::default();
        assert_eq!(idx.entry_count(), 0);
    }

    // ── 3. insert and get ────────────────────────────────────────────────────

    #[test]
    fn test_secondary_insert_and_get() {
        let mut idx = SecondaryBlockIndex::new();
        let e = entry("cid-1", 100, "raw", 5, &["x"]);
        idx.insert(e.clone());
        let got = idx.get("cid-1");
        assert!(got.is_some());
        assert_eq!(got.map(|g| &g.cid), Some(&"cid-1".to_owned()));
        assert_eq!(got.map(|g| g.size_bytes), Some(100));
    }

    // ── 4. get returns None for missing ──────────────────────────────────────

    #[test]
    fn test_secondary_get_missing() {
        let idx = SecondaryBlockIndex::new();
        assert!(idx.get("nope").is_none());
    }

    // ── 5. remove returns entry ──────────────────────────────────────────────

    #[test]
    fn test_secondary_remove_returns_entry() {
        let mut idx = populated();
        let removed = idx.remove("cid-a");
        assert!(removed.is_some());
        assert_eq!(removed.map(|r| r.cid), Some("cid-a".to_owned()));
        assert!(idx.get("cid-a").is_none());
    }

    // ── 6. remove returns None for missing ───────────────────────────────────

    #[test]
    fn test_secondary_remove_missing() {
        let mut idx = SecondaryBlockIndex::new();
        assert!(idx.remove("nope").is_none());
    }

    // ── 7. remove updates total_bytes ────────────────────────────────────────

    #[test]
    fn test_secondary_remove_updates_total_bytes() {
        let mut idx = SecondaryBlockIndex::new();
        idx.insert(entry("a", 100, "raw", 1, &[]));
        idx.insert(entry("b", 200, "raw", 2, &[]));
        assert_eq!(idx.total_bytes(), 300);
        idx.remove("a");
        assert_eq!(idx.total_bytes(), 200);
    }

    // ── 8. remove updates codec index ────────────────────────────────────────

    #[test]
    fn test_secondary_remove_updates_codec_index() {
        let mut idx = populated();
        idx.remove("cid-a"); // was "raw"
        let raw_entries = idx.find_by_codec("raw");
        let cids: Vec<&str> = raw_entries.iter().map(|e| e.cid.as_str()).collect();
        assert!(!cids.contains(&"cid-a"));
        assert!(cids.contains(&"cid-c")); // cid-c is also raw
    }

    // ── 9. remove updates tag index ──────────────────────────────────────────

    #[test]
    fn test_secondary_remove_updates_tag_index() {
        let mut idx = populated();
        idx.remove("cid-a"); // had tag "pin"
        let pin_entries = idx.find_by_tag("pin");
        let cids: Vec<&str> = pin_entries.iter().map(|e| e.cid.as_str()).collect();
        assert!(!cids.contains(&"cid-a"));
        assert!(cids.contains(&"cid-b")); // cid-b also has "pin"
    }

    // ── 10. find_by_codec ────────────────────────────────────────────────────

    #[test]
    fn test_secondary_find_by_codec() {
        let idx = populated();
        let raw = idx.find_by_codec("raw");
        assert_eq!(raw.len(), 2);
        let cids: Vec<&str> = raw.iter().map(|e| e.cid.as_str()).collect();
        assert!(cids.contains(&"cid-a"));
        assert!(cids.contains(&"cid-c"));
    }

    // ── 11. find_by_codec empty ──────────────────────────────────────────────

    #[test]
    fn test_secondary_find_by_codec_empty() {
        let idx = populated();
        assert!(idx.find_by_codec("dag-json").is_empty());
    }

    // ── 12. find_by_tag ──────────────────────────────────────────────────────

    #[test]
    fn test_secondary_find_by_tag() {
        let idx = populated();
        let pinned = idx.find_by_tag("pin");
        assert_eq!(pinned.len(), 2);
        let cids: Vec<&str> = pinned.iter().map(|e| e.cid.as_str()).collect();
        assert!(cids.contains(&"cid-a"));
        assert!(cids.contains(&"cid-b"));
    }

    // ── 13. find_by_tag empty ────────────────────────────────────────────────

    #[test]
    fn test_secondary_find_by_tag_empty() {
        let idx = populated();
        assert!(idx.find_by_tag("nonexistent").is_empty());
    }

    // ── 14. find_by_tag single match ─────────────────────────────────────────

    #[test]
    fn test_secondary_find_by_tag_single() {
        let idx = populated();
        let important = idx.find_by_tag("important");
        assert_eq!(important.len(), 1);
        assert_eq!(important[0].cid, "cid-b");
    }

    // ── 15. find_by_size_range ───────────────────────────────────────────────

    #[test]
    fn test_secondary_find_by_size_range() {
        let idx = populated();
        let results = idx.find_by_size_range(512, 2048);
        assert_eq!(results.len(), 3); // cid-a(1024), cid-b(2048), cid-c(512)
        for e in &results {
            assert!(e.size_bytes >= 512 && e.size_bytes <= 2048);
        }
    }

    // ── 16. find_by_size_range no match ──────────────────────────────────────

    #[test]
    fn test_secondary_find_by_size_range_none() {
        let idx = populated();
        assert!(idx.find_by_size_range(10000, 20000).is_empty());
    }

    // ── 17. find_by_size_range exact match ───────────────────────────────────

    #[test]
    fn test_secondary_find_by_size_range_exact() {
        let idx = populated();
        let results = idx.find_by_size_range(4096, 4096);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].cid, "cid-d");
    }

    // ── 18. find_by_created_range ────────────────────────────────────────────

    #[test]
    fn test_secondary_find_by_created_range() {
        let idx = populated();
        let results = idx.find_by_created_range(15, 35);
        assert_eq!(results.len(), 2); // cid-b(20), cid-c(30)
        for e in &results {
            assert!(e.created_tick >= 15 && e.created_tick <= 35);
        }
    }

    // ── 19. find_by_created_range no match ───────────────────────────────────

    #[test]
    fn test_secondary_find_by_created_range_none() {
        let idx = populated();
        assert!(idx.find_by_created_range(100, 200).is_empty());
    }

    // ── 20. entry_count ──────────────────────────────────────────────────────

    #[test]
    fn test_secondary_entry_count() {
        let idx = populated();
        assert_eq!(idx.entry_count(), 4);
    }

    // ── 21. total_bytes ──────────────────────────────────────────────────────

    #[test]
    fn test_secondary_total_bytes() {
        let idx = populated();
        assert_eq!(idx.total_bytes(), 1024 + 2048 + 512 + 4096);
    }

    // ── 22. unique_codecs ────────────────────────────────────────────────────

    #[test]
    fn test_secondary_unique_codecs() {
        let idx = populated();
        let codecs = idx.unique_codecs();
        assert_eq!(codecs.len(), 3);
        assert!(codecs.contains(&"raw".to_owned()));
        assert!(codecs.contains(&"dag-cbor".to_owned()));
        assert!(codecs.contains(&"dag-pb".to_owned()));
    }

    // ── 23. unique_tags ──────────────────────────────────────────────────────

    #[test]
    fn test_secondary_unique_tags() {
        let idx = populated();
        let tags = idx.unique_tags();
        assert_eq!(tags.len(), 3);
        assert!(tags.contains(&"pin".to_owned()));
        assert!(tags.contains(&"important".to_owned()));
        assert!(tags.contains(&"temp".to_owned()));
    }

    // ── 24. stats accuracy ───────────────────────────────────────────────────

    #[test]
    fn test_secondary_stats() {
        let idx = populated();
        let s = idx.stats();
        assert_eq!(s.entry_count, 4);
        assert_eq!(s.total_bytes, 1024 + 2048 + 512 + 4096);
        assert_eq!(s.unique_codecs, 3);
        assert_eq!(s.unique_tags, 3);
    }

    // ── 25. re-insert replaces entry ─────────────────────────────────────────

    #[test]
    fn test_secondary_reinsert_replaces() {
        let mut idx = SecondaryBlockIndex::new();
        idx.insert(entry("cid-1", 100, "raw", 1, &["old"]));
        idx.insert(entry("cid-1", 200, "dag-cbor", 2, &["new"]));

        assert_eq!(idx.entry_count(), 1);
        let e = idx.get("cid-1");
        assert!(e.is_some());
        let e = e.expect("checked above");
        assert_eq!(e.codec, "dag-cbor");
        assert_eq!(e.size_bytes, 200);
        assert_eq!(idx.total_bytes(), 200);

        // Old codec bucket should not contain cid-1
        assert!(idx.find_by_codec("raw").is_empty());
        // Old tag should not contain cid-1
        assert!(idx.find_by_tag("old").is_empty());
        // New codec/tag should contain it
        assert_eq!(idx.find_by_codec("dag-cbor").len(), 1);
        assert_eq!(idx.find_by_tag("new").len(), 1);
    }

    // ── 26. multiple entries same codec ───────────────────────────────────────

    #[test]
    fn test_secondary_multiple_same_codec() {
        let mut idx = SecondaryBlockIndex::new();
        idx.insert(entry("a", 10, "raw", 1, &[]));
        idx.insert(entry("b", 20, "raw", 2, &[]));
        idx.insert(entry("c", 30, "raw", 3, &[]));
        let results = idx.find_by_codec("raw");
        assert_eq!(results.len(), 3);
    }

    // ── 27. empty index operations ───────────────────────────────────────────

    #[test]
    fn test_secondary_empty_index_operations() {
        let idx = SecondaryBlockIndex::new();
        assert!(idx.find_by_codec("raw").is_empty());
        assert!(idx.find_by_tag("any").is_empty());
        assert!(idx.find_by_size_range(0, u64::MAX).is_empty());
        assert!(idx.find_by_created_range(0, u64::MAX).is_empty());
        assert!(idx.unique_codecs().is_empty());
        assert!(idx.unique_tags().is_empty());
        let s = idx.stats();
        assert_eq!(s.entry_count, 0);
        assert_eq!(s.total_bytes, 0);
        assert_eq!(s.unique_codecs, 0);
        assert_eq!(s.unique_tags, 0);
    }

    // ── 28. unique_codecs excludes emptied buckets ───────────────────────────

    #[test]
    fn test_secondary_unique_codecs_after_remove() {
        let mut idx = SecondaryBlockIndex::new();
        idx.insert(entry("a", 10, "raw", 1, &[]));
        assert_eq!(idx.unique_codecs().len(), 1);
        idx.remove("a");
        assert!(idx.unique_codecs().is_empty());
    }

    // ── 29. unique_tags excludes emptied buckets ─────────────────────────────

    #[test]
    fn test_secondary_unique_tags_after_remove() {
        let mut idx = SecondaryBlockIndex::new();
        idx.insert(entry("a", 10, "raw", 1, &["only"]));
        assert_eq!(idx.unique_tags().len(), 1);
        idx.remove("a");
        assert!(idx.unique_tags().is_empty());
    }

    // ── 30. find results are sorted by cid ───────────────────────────────────

    #[test]
    fn test_secondary_find_sorted_by_cid() {
        let mut idx = SecondaryBlockIndex::new();
        idx.insert(entry("z", 10, "raw", 1, &["t"]));
        idx.insert(entry("a", 20, "raw", 2, &["t"]));
        idx.insert(entry("m", 30, "raw", 3, &["t"]));

        let by_codec = idx.find_by_codec("raw");
        let cids: Vec<&str> = by_codec.iter().map(|e| e.cid.as_str()).collect();
        assert_eq!(cids, vec!["a", "m", "z"]);

        let by_tag = idx.find_by_tag("t");
        let cids: Vec<&str> = by_tag.iter().map(|e| e.cid.as_str()).collect();
        assert_eq!(cids, vec!["a", "m", "z"]);
    }
}
