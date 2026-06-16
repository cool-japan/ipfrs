//! Secondary index over block metadata enabling fast filtered queries.
//!
//! [`StorageIndexBuilder`] maintains an in-memory index keyed by CID and
//! supports rich filter expressions (size range, time range, tags, content
//! type, logical AND/OR) without scanning the underlying block store.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// IndexField
// ---------------------------------------------------------------------------

/// Dimensions of the secondary index.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum IndexField {
    /// Block size in bytes.
    Size,
    /// Creation timestamp (Unix seconds).
    CreatedAt,
    /// Arbitrary string tag.
    Tag,
    /// MIME-style content type.
    ContentType,
}

// ---------------------------------------------------------------------------
// IndexEntry
// ---------------------------------------------------------------------------

/// A single record stored in the secondary index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexEntry {
    /// Content identifier (CID) – used as the primary key.
    pub cid: String,
    /// Block size in bytes.
    pub size_bytes: u64,
    /// Creation time as Unix epoch seconds.
    pub created_at_secs: u64,
    /// Arbitrary string tags attached to this block.
    pub tags: Vec<String>,
    /// MIME-style content type (e.g. `"application/octet-stream"`).
    pub content_type: String,
}

// ---------------------------------------------------------------------------
// QueryFilter
// ---------------------------------------------------------------------------

/// Filter expression for querying the index.
#[derive(Clone, Debug, PartialEq)]
pub enum QueryFilter {
    /// Blocks whose size falls within `[min, max]` (inclusive).
    SizeRange { min: u64, max: u64 },
    /// Blocks created strictly after `secs` (i.e. `created_at_secs > secs`).
    CreatedAfter { secs: u64 },
    /// Blocks created strictly before `secs` (i.e. `created_at_secs < secs`).
    CreatedBefore { secs: u64 },
    /// Blocks that carry the given tag.
    HasTag { tag: String },
    /// Blocks whose content type matches `ct` exactly.
    ContentType { ct: String },
    /// Both sub-filters must match.
    And(Box<QueryFilter>, Box<QueryFilter>),
    /// At least one sub-filter must match.
    Or(Box<QueryFilter>, Box<QueryFilter>),
}

// ---------------------------------------------------------------------------
// IndexStats
// ---------------------------------------------------------------------------

/// Summary statistics for the index.
#[derive(Clone, Debug, PartialEq)]
pub struct IndexStats {
    /// Total number of entries currently held in the index.
    pub total_entries: usize,
    /// Dimensions that are actively indexed.
    pub indexed_fields: Vec<IndexField>,
}

impl IndexStats {
    /// Returns the fraction of entries that are indexed.
    ///
    /// Because this implementation always indexes every inserted entry this
    /// value is always `1.0`.
    pub fn coverage(&self) -> f64 {
        1.0
    }
}

// ---------------------------------------------------------------------------
// StorageIndexBuilder
// ---------------------------------------------------------------------------

/// Builds and maintains a secondary index over block metadata.
///
/// # Example
///
/// ```rust
/// use ipfrs_storage::index_builder::{
///     IndexEntry, QueryFilter, StorageIndexBuilder,
/// };
///
/// let mut idx = StorageIndexBuilder::new();
/// idx.insert(IndexEntry {
///     cid: "QmA".into(),
///     size_bytes: 1024,
///     created_at_secs: 1_000,
///     tags: vec!["audio".into()],
///     content_type: "audio/mpeg".into(),
/// });
///
/// let results = idx.query(&QueryFilter::HasTag { tag: "audio".into() });
/// assert_eq!(results.len(), 1);
/// ```
pub struct StorageIndexBuilder {
    /// Primary storage: CID → [`IndexEntry`].
    pub entries: HashMap<String, IndexEntry>,
}

impl StorageIndexBuilder {
    /// Creates an empty index.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Inserts (or replaces) an entry keyed by its CID.
    pub fn insert(&mut self, entry: IndexEntry) {
        self.entries.insert(entry.cid.clone(), entry);
    }

    /// Removes the entry with the given CID.
    ///
    /// Returns `true` if an entry was present and removed, `false` otherwise.
    pub fn remove(&mut self, cid: &str) -> bool {
        self.entries.remove(cid).is_some()
    }

    /// Returns all entries that satisfy `filter`, sorted by
    /// `created_at_secs` ascending.
    pub fn query(&self, filter: &QueryFilter) -> Vec<&IndexEntry> {
        let mut results: Vec<&IndexEntry> = self
            .entries
            .values()
            .filter(|e| self.matches(e, filter))
            .collect();

        results.sort_by_key(|e| e.created_at_secs);
        results
    }

    /// Replaces the tag list for the entry identified by `cid`.
    ///
    /// Returns `true` if the entry was found and updated, `false` otherwise.
    pub fn update_tags(&mut self, cid: &str, tags: Vec<String>) -> bool {
        match self.entries.get_mut(cid) {
            Some(entry) => {
                entry.tags = tags;
                true
            }
            None => false,
        }
    }

    /// Returns summary statistics for the current index state.
    pub fn stats(&self) -> IndexStats {
        IndexStats {
            total_entries: self.entries.len(),
            indexed_fields: vec![
                IndexField::Size,
                IndexField::CreatedAt,
                IndexField::Tag,
                IndexField::ContentType,
            ],
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Recursively evaluates `filter` against a single `entry`.
    fn matches(&self, entry: &IndexEntry, filter: &QueryFilter) -> bool {
        match filter {
            QueryFilter::SizeRange { min, max } => {
                entry.size_bytes >= *min && entry.size_bytes <= *max
            }
            QueryFilter::CreatedAfter { secs } => entry.created_at_secs > *secs,
            QueryFilter::CreatedBefore { secs } => entry.created_at_secs < *secs,
            QueryFilter::HasTag { tag } => entry.tags.iter().any(|t| t == tag),
            QueryFilter::ContentType { ct } => &entry.content_type == ct,
            QueryFilter::And(left, right) => {
                self.matches(entry, left) && self.matches(entry, right)
            }
            QueryFilter::Or(left, right) => self.matches(entry, left) || self.matches(entry, right),
        }
    }
}

impl Default for StorageIndexBuilder {
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
        created_at_secs: u64,
        tags: &[&str],
        content_type: &str,
    ) -> IndexEntry {
        IndexEntry {
            cid: cid.to_owned(),
            size_bytes,
            created_at_secs,
            tags: tags.iter().map(|s| s.to_string()).collect(),
            content_type: content_type.to_owned(),
        }
    }

    fn populated_index() -> StorageIndexBuilder {
        let mut idx = StorageIndexBuilder::new();
        idx.insert(make_entry(
            "QmA",
            100,
            1_000,
            &["image", "png"],
            "image/png",
        ));
        idx.insert(make_entry("QmB", 500, 2_000, &["video"], "video/mp4"));
        idx.insert(make_entry(
            "QmC",
            1_000,
            3_000,
            &["audio", "mp3"],
            "audio/mpeg",
        ));
        idx.insert(make_entry(
            "QmD",
            50,
            4_000,
            &["image", "jpeg"],
            "image/jpeg",
        ));
        idx.insert(make_entry(
            "QmE",
            2_000,
            5_000,
            &[],
            "application/octet-stream",
        ));
        idx
    }

    // -----------------------------------------------------------------------
    // 1. insert + query by SizeRange
    // -----------------------------------------------------------------------
    #[test]
    fn test_insert_and_query_size_range() {
        let idx = populated_index();
        let results = idx.query(&QueryFilter::SizeRange {
            min: 100,
            max: 1_000,
        });
        let cids: Vec<&str> = results.iter().map(|e| e.cid.as_str()).collect();
        assert!(cids.contains(&"QmA"), "QmA (100) should be in [100, 1000]");
        assert!(cids.contains(&"QmB"), "QmB (500) should be in [100, 1000]");
        assert!(cids.contains(&"QmC"), "QmC (1000) should be in [100, 1000]");
        assert!(
            !cids.contains(&"QmD"),
            "QmD (50) should not be in [100, 1000]"
        );
        assert!(
            !cids.contains(&"QmE"),
            "QmE (2000) should not be in [100, 1000]"
        );
    }

    // -----------------------------------------------------------------------
    // 2. SizeRange inclusive on both bounds
    // -----------------------------------------------------------------------
    #[test]
    fn test_size_range_inclusive_bounds() {
        let idx = populated_index();
        // Exact lower bound
        let lower = idx.query(&QueryFilter::SizeRange { min: 50, max: 50 });
        assert_eq!(lower.len(), 1);
        assert_eq!(lower[0].cid, "QmD");

        // Exact upper bound
        let upper = idx.query(&QueryFilter::SizeRange {
            min: 2_000,
            max: 2_000,
        });
        assert_eq!(upper.len(), 1);
        assert_eq!(upper[0].cid, "QmE");
    }

    // -----------------------------------------------------------------------
    // 3. query by CreatedAfter
    // -----------------------------------------------------------------------
    #[test]
    fn test_query_created_after() {
        let idx = populated_index();
        let results = idx.query(&QueryFilter::CreatedAfter { secs: 3_000 });
        let cids: Vec<&str> = results.iter().map(|e| e.cid.as_str()).collect();
        assert!(!cids.contains(&"QmA"));
        assert!(!cids.contains(&"QmB"));
        assert!(!cids.contains(&"QmC"), "3000 is not > 3000");
        assert!(cids.contains(&"QmD"));
        assert!(cids.contains(&"QmE"));
    }

    // -----------------------------------------------------------------------
    // 4. query by CreatedBefore
    // -----------------------------------------------------------------------
    #[test]
    fn test_query_created_before() {
        let idx = populated_index();
        let results = idx.query(&QueryFilter::CreatedBefore { secs: 3_000 });
        let cids: Vec<&str> = results.iter().map(|e| e.cid.as_str()).collect();
        assert!(cids.contains(&"QmA"));
        assert!(cids.contains(&"QmB"));
        assert!(!cids.contains(&"QmC"), "3000 is not < 3000");
        assert!(!cids.contains(&"QmD"));
        assert!(!cids.contains(&"QmE"));
    }

    // -----------------------------------------------------------------------
    // 5. query by HasTag
    // -----------------------------------------------------------------------
    #[test]
    fn test_query_has_tag() {
        let idx = populated_index();
        let results = idx.query(&QueryFilter::HasTag {
            tag: "image".into(),
        });
        let cids: Vec<&str> = results.iter().map(|e| e.cid.as_str()).collect();
        assert!(cids.contains(&"QmA"));
        assert!(cids.contains(&"QmD"));
        assert!(!cids.contains(&"QmB"));
        assert!(!cids.contains(&"QmC"));
        assert!(!cids.contains(&"QmE"));
    }

    // -----------------------------------------------------------------------
    // 6. query by ContentType
    // -----------------------------------------------------------------------
    #[test]
    fn test_query_content_type() {
        let idx = populated_index();
        let results = idx.query(&QueryFilter::ContentType {
            ct: "video/mp4".into(),
        });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].cid, "QmB");
    }

    // -----------------------------------------------------------------------
    // 7. And combines filters
    // -----------------------------------------------------------------------
    #[test]
    fn test_and_filter() {
        let idx = populated_index();
        // image tag AND size in [0, 100]: QmA (image, 100) and QmD (image, 50) both qualify
        let filter = QueryFilter::And(
            Box::new(QueryFilter::HasTag {
                tag: "image".into(),
            }),
            Box::new(QueryFilter::SizeRange { min: 0, max: 100 }),
        );
        let results = idx.query(&filter);
        let cids: Vec<&str> = results.iter().map(|e| e.cid.as_str()).collect();
        assert!(
            cids.contains(&"QmA"),
            "QmA has image tag and size=100 which is within [0,100]"
        );
        assert!(
            cids.contains(&"QmD"),
            "QmD has image tag and size=50 which is within [0,100]"
        );
        assert!(!cids.contains(&"QmB"), "QmB has no image tag");
        assert!(!cids.contains(&"QmC"), "QmC has no image tag");
        assert_eq!(results.len(), 2);
    }

    // -----------------------------------------------------------------------
    // 8. And filter – no result when both conditions cannot be met together
    // -----------------------------------------------------------------------
    #[test]
    fn test_and_filter_no_match() {
        let idx = populated_index();
        // video AND image tag – no entry has both
        let filter = QueryFilter::And(
            Box::new(QueryFilter::HasTag {
                tag: "video".into(),
            }),
            Box::new(QueryFilter::HasTag {
                tag: "image".into(),
            }),
        );
        let results = idx.query(&filter);
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // 9. Or combines filters
    // -----------------------------------------------------------------------
    #[test]
    fn test_or_filter() {
        let idx = populated_index();
        // audio OR video
        let filter = QueryFilter::Or(
            Box::new(QueryFilter::HasTag {
                tag: "audio".into(),
            }),
            Box::new(QueryFilter::HasTag {
                tag: "video".into(),
            }),
        );
        let results = idx.query(&filter);
        let cids: Vec<&str> = results.iter().map(|e| e.cid.as_str()).collect();
        assert!(cids.contains(&"QmB"));
        assert!(cids.contains(&"QmC"));
        assert!(!cids.contains(&"QmA"));
        assert!(!cids.contains(&"QmD"));
        assert!(!cids.contains(&"QmE"));
    }

    // -----------------------------------------------------------------------
    // 10. Or filter – returns entry matching either branch
    // -----------------------------------------------------------------------
    #[test]
    fn test_or_filter_matches_both_branches() {
        let idx = populated_index();
        let filter = QueryFilter::Or(
            Box::new(QueryFilter::ContentType {
                ct: "image/png".into(),
            }),
            Box::new(QueryFilter::ContentType {
                ct: "image/jpeg".into(),
            }),
        );
        let results = idx.query(&filter);
        assert_eq!(results.len(), 2);
    }

    // -----------------------------------------------------------------------
    // 11. remove
    // -----------------------------------------------------------------------
    #[test]
    fn test_remove_existing_entry() {
        let mut idx = populated_index();
        let removed = idx.remove("QmB");
        assert!(removed, "remove should return true for existing CID");
        let results = idx.query(&QueryFilter::ContentType {
            ct: "video/mp4".into(),
        });
        assert!(
            results.is_empty(),
            "QmB should no longer appear after removal"
        );
    }

    #[test]
    fn test_remove_nonexistent_entry() {
        let mut idx = populated_index();
        let removed = idx.remove("QmDoesNotExist");
        assert!(!removed, "remove should return false for unknown CID");
    }

    // -----------------------------------------------------------------------
    // 12. update_tags
    // -----------------------------------------------------------------------
    #[test]
    fn test_update_tags_existing() {
        let mut idx = populated_index();
        let updated = idx.update_tags("QmE", vec!["new-tag".into(), "another".into()]);
        assert!(updated);

        let results = idx.query(&QueryFilter::HasTag {
            tag: "new-tag".into(),
        });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].cid, "QmE");
    }

    #[test]
    fn test_update_tags_nonexistent() {
        let mut idx = populated_index();
        let updated = idx.update_tags("QmNone", vec!["x".into()]);
        assert!(!updated, "update_tags should return false for unknown CID");
    }

    // -----------------------------------------------------------------------
    // 13. query result sorted by created_at ascending
    // -----------------------------------------------------------------------
    #[test]
    fn test_query_sorted_by_created_at() {
        let idx = populated_index();
        let results = idx.query(&QueryFilter::SizeRange {
            min: 0,
            max: u64::MAX,
        });
        let times: Vec<u64> = results.iter().map(|e| e.created_at_secs).collect();
        let mut sorted = times.clone();
        sorted.sort_unstable();
        assert_eq!(
            times, sorted,
            "results must be sorted by created_at_secs asc"
        );
    }

    // -----------------------------------------------------------------------
    // 14. empty result
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_result() {
        let idx = populated_index();
        let results = idx.query(&QueryFilter::ContentType {
            ct: "text/plain".into(),
        });
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // 15. stats.total_entries
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_total_entries() {
        let idx = populated_index();
        let stats = idx.stats();
        assert_eq!(stats.total_entries, 5);
    }

    #[test]
    fn test_stats_total_entries_after_remove() {
        let mut idx = populated_index();
        idx.remove("QmA");
        assert_eq!(idx.stats().total_entries, 4);
    }

    // -----------------------------------------------------------------------
    // 16. query all with always-true filter (Or of size 0..MAX)
    // -----------------------------------------------------------------------
    #[test]
    fn test_query_all_always_true() {
        let idx = populated_index();
        // SizeRange [0, u64::MAX] matches every entry
        let results = idx.query(&QueryFilter::SizeRange {
            min: 0,
            max: u64::MAX,
        });
        assert_eq!(results.len(), 5, "all 5 entries should be returned");
    }

    // -----------------------------------------------------------------------
    // 17. IndexStats.coverage always returns 1.0
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_coverage() {
        let idx = populated_index();
        let stats = idx.stats();
        assert!((stats.coverage() - 1.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // 18. indexed_fields contains all four dimensions
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_indexed_fields() {
        let idx = populated_index();
        let stats = idx.stats();
        assert!(stats.indexed_fields.contains(&IndexField::Size));
        assert!(stats.indexed_fields.contains(&IndexField::CreatedAt));
        assert!(stats.indexed_fields.contains(&IndexField::Tag));
        assert!(stats.indexed_fields.contains(&IndexField::ContentType));
    }

    // -----------------------------------------------------------------------
    // 19. insert overwrites duplicate CID
    // -----------------------------------------------------------------------
    #[test]
    fn test_insert_overwrites_existing_cid() {
        let mut idx = StorageIndexBuilder::new();
        idx.insert(make_entry("QmX", 100, 1_000, &["old"], "text/plain"));
        idx.insert(make_entry("QmX", 200, 2_000, &["new"], "application/json"));

        assert_eq!(
            idx.stats().total_entries,
            1,
            "duplicate CID must not inflate count"
        );

        let results = idx.query(&QueryFilter::HasTag { tag: "new".into() });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].size_bytes, 200);
    }

    // -----------------------------------------------------------------------
    // 20. Nested And + Or (compound filter)
    // -----------------------------------------------------------------------
    #[test]
    fn test_nested_and_or_filter() {
        let idx = populated_index();
        // (image tag OR audio tag) AND size < 500
        let filter = QueryFilter::And(
            Box::new(QueryFilter::Or(
                Box::new(QueryFilter::HasTag {
                    tag: "image".into(),
                }),
                Box::new(QueryFilter::HasTag {
                    tag: "audio".into(),
                }),
            )),
            Box::new(QueryFilter::SizeRange { min: 0, max: 499 }),
        );
        let results = idx.query(&filter);
        // QmA (image, 100) ✓  QmC (audio, 1000) ✗  QmD (image, 50) ✓
        let cids: Vec<&str> = results.iter().map(|e| e.cid.as_str()).collect();
        assert!(cids.contains(&"QmA"));
        assert!(cids.contains(&"QmD"));
        assert!(!cids.contains(&"QmC"));
        assert_eq!(results.len(), 2);
    }
}
