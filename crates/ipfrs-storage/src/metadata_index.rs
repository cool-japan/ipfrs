//! Secondary index over block metadata fields for efficient filtered queries.
//!
//! [`StorageMetadataIndex`] maintains an in-memory inverted index keyed by
//! [`MetadataField`] variants (tags, content-type, owner, size bucket, tick
//! bucket). Queries combine AND / NOT constraints without scanning every stored
//! block entry.

use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────────────────────────────────────
// MetadataField
// ─────────────────────────────────────────────────────────────────────────────

/// Discriminated metadata field used as an index key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MetadataField {
    /// Arbitrary tag string (e.g. `"important"`, `"cold-storage"`).
    Tag(String),
    /// MIME-style content-type string (e.g. `"image/png"`).
    ContentType(String),
    /// Peer-ID of the content owner.
    Owner(String),
    /// Quantised size bucket: `block_size / 65536` (integer division).
    SizeBucket(u64),
    /// Quantised tick bucket: `created_at_tick / 1000` (integer division).
    TickBucket(u64),
}

// ─────────────────────────────────────────────────────────────────────────────
// IndexEntry (aliased as MetadataIndexEntry in pub use)
// ─────────────────────────────────────────────────────────────────────────────

/// Metadata record stored for each block in the metadata index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetadataIndexEntry {
    /// Numeric block identifier (primary key).
    pub block_id: u64,
    /// Content identifier string (CID).
    pub cid: String,
    /// All metadata fields associated with this block.
    pub fields: Vec<MetadataField>,
}

// ─────────────────────────────────────────────────────────────────────────────
// SortField
// ─────────────────────────────────────────────────────────────────────────────

/// Sort dimension for [`MetadataQuery`] results.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetadataSortField {
    /// Sort ascending by `block_id`.
    BlockId,
    /// Sort alphabetically ascending by `cid`.
    Cid,
}

// ─────────────────────────────────────────────────────────────────────────────
// MetadataQuery
// ─────────────────────────────────────────────────────────────────────────────

/// A structured query against the metadata index.
#[derive(Clone, Debug, Default)]
pub struct MetadataQuery {
    /// All of these fields must be present on the block (AND semantics).
    pub must_have: Vec<MetadataField>,
    /// None of these fields may be present on the block (NOT semantics).
    pub must_not_have: Vec<MetadataField>,
    /// Optional sort dimension applied before `limit`.
    pub sort_by: Option<MetadataSortField>,
    /// Optional maximum number of results to return.
    pub limit: Option<usize>,
}

// ─────────────────────────────────────────────────────────────────────────────
// QueryResult (aliased as MetadataQueryResult in pub use)
// ─────────────────────────────────────────────────────────────────────────────

/// A single result entry returned by [`StorageMetadataIndex::query`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetadataQueryResult {
    /// Numeric block identifier.
    pub block_id: u64,
    /// Content identifier string (CID).
    pub cid: String,
    /// The subset of `must_have` fields that matched on this block.
    pub matched_fields: Vec<MetadataField>,
}

// ─────────────────────────────────────────────────────────────────────────────
// MetadataIndexStats
// ─────────────────────────────────────────────────────────────────────────────

/// Cumulative statistics for a [`StorageMetadataIndex`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MetadataIndexStats {
    /// Total number of entries currently held.
    pub total_entries: usize,
    /// Sum of `fields.len()` across all entries.
    pub total_fields_indexed: usize,
    /// Total number of times [`StorageMetadataIndex::query`] has been called.
    pub total_queries: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// StorageMetadataIndex
// ─────────────────────────────────────────────────────────────────────────────

/// Secondary index over block metadata fields enabling efficient filtered queries.
///
/// # Design
///
/// Two data structures are maintained in tandem:
///
/// * `entries` — a `HashMap<block_id, MetadataIndexEntry>` for O(1) point
///   lookups and to support fast replacement when the same block is re-inserted.
/// * `field_index` — an inverted index mapping each `MetadataField` to the set
///   of `block_id`s that carry that field.
///
/// Queries intersect the posting lists of all `must_have` fields (giving the
/// candidate set), then subtract any block whose entry contains a `must_not_have`
/// field.
pub struct StorageMetadataIndex {
    /// Primary store: block_id → entry.
    pub entries: HashMap<u64, MetadataIndexEntry>,
    /// Inverted index: field → block_ids.
    pub field_index: HashMap<MetadataField, Vec<u64>>,
    /// Running statistics.
    pub stats: MetadataIndexStats,
}

impl StorageMetadataIndex {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Create an empty [`StorageMetadataIndex`].
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            field_index: HashMap::new(),
            stats: MetadataIndexStats::default(),
        }
    }

    // ── Mutation ──────────────────────────────────────────────────────────────

    /// Insert (or replace) a block entry.
    ///
    /// If a block with the same `block_id` already exists its old field index
    /// postings are removed before the new entry is written.
    pub fn insert(&mut self, entry: MetadataIndexEntry) {
        // Remove old postings if the block already exists.
        if let Some(old) = self.entries.remove(&entry.block_id) {
            self.stats.total_entries -= 1;
            self.stats.total_fields_indexed -= old.fields.len();
            for field in &old.fields {
                if let Some(ids) = self.field_index.get_mut(field) {
                    ids.retain(|&id| id != old.block_id);
                    if ids.is_empty() {
                        self.field_index.remove(field);
                    }
                }
            }
        }

        // Build new postings.
        let block_id = entry.block_id;
        let field_count = entry.fields.len();
        for field in &entry.fields {
            self.field_index
                .entry(field.clone())
                .or_default()
                .push(block_id);
        }

        self.entries.insert(block_id, entry);
        self.stats.total_entries += 1;
        self.stats.total_fields_indexed += field_count;
    }

    /// Remove the entry for `block_id`.
    ///
    /// Returns `true` if an entry was present and removed, `false` otherwise.
    pub fn remove(&mut self, block_id: u64) -> bool {
        match self.entries.remove(&block_id) {
            None => false,
            Some(old) => {
                self.stats.total_entries -= 1;
                self.stats.total_fields_indexed -= old.fields.len();
                for field in &old.fields {
                    if let Some(ids) = self.field_index.get_mut(field) {
                        ids.retain(|&id| id != block_id);
                        if ids.is_empty() {
                            self.field_index.remove(field);
                        }
                    }
                }
                true
            }
        }
    }

    // ── Query ─────────────────────────────────────────────────────────────────

    /// Execute a structured metadata query.
    ///
    /// # Algorithm
    ///
    /// 1. **Candidate set** – if `must_have` is empty, all block IDs are
    ///    candidates; otherwise the candidate set is the intersection of the
    ///    posting lists for each `must_have` field.
    /// 2. **Exclusion** – remove any candidate whose entry contains *any*
    ///    `must_not_have` field.
    /// 3. **Projection** – build a [`MetadataQueryResult`] per surviving
    ///    candidate recording which `must_have` fields matched.
    /// 4. **Sort** – apply `sort_by` if present.
    /// 5. **Limit** – truncate to `limit` if present.
    /// 6. **Stats** – increment `total_queries`.
    pub fn query(&mut self, q: &MetadataQuery) -> Vec<MetadataQueryResult> {
        // Step 1: compute candidate block_id set.
        let candidates: HashSet<u64> = if q.must_have.is_empty() {
            self.entries.keys().copied().collect()
        } else {
            // Start from the posting list of the first must_have field,
            // then intersect with each subsequent field's posting list.
            let mut iter = q.must_have.iter();
            // Safety: must_have is non-empty, so first() always returns Some.
            let first_field = iter.next().expect("must_have checked non-empty");
            let initial: HashSet<u64> = self
                .field_index
                .get(first_field)
                .map(|v| v.iter().copied().collect())
                .unwrap_or_default();

            iter.fold(initial, |acc, field| {
                let posting: HashSet<u64> = self
                    .field_index
                    .get(field)
                    .map(|v| v.iter().copied().collect())
                    .unwrap_or_default();
                acc.intersection(&posting).copied().collect()
            })
        };

        // Step 2: build must_not_have field set for O(1) membership checks.
        let exclude_set: HashSet<&MetadataField> = q.must_not_have.iter().collect();

        // Step 3: filter and project.
        let mut results: Vec<MetadataQueryResult> = candidates
            .into_iter()
            .filter_map(|block_id| {
                let entry = self.entries.get(&block_id)?;
                // Exclude if any entry field is in the exclusion set.
                let excluded = entry.fields.iter().any(|f| exclude_set.contains(f));
                if excluded {
                    return None;
                }
                // Collect which must_have fields are actually present.
                let matched_fields: Vec<MetadataField> = q
                    .must_have
                    .iter()
                    .filter(|f| entry.fields.contains(f))
                    .cloned()
                    .collect();
                Some(MetadataQueryResult {
                    block_id: entry.block_id,
                    cid: entry.cid.clone(),
                    matched_fields,
                })
            })
            .collect();

        // Step 4: sort.
        match q.sort_by {
            Some(MetadataSortField::BlockId) => {
                results.sort_by_key(|r| r.block_id);
            }
            Some(MetadataSortField::Cid) => {
                results.sort_by(|a, b| a.cid.cmp(&b.cid));
            }
            None => {}
        }

        // Step 5: apply limit.
        if let Some(limit) = q.limit {
            results.truncate(limit);
        }

        // Step 6: update stats.
        self.stats.total_queries += 1;

        results
    }

    // ── Point lookup ──────────────────────────────────────────────────────────

    /// Return a reference to the entry for `block_id`, or `None`.
    pub fn get(&self, block_id: u64) -> Option<&MetadataIndexEntry> {
        self.entries.get(&block_id)
    }

    /// Return a reference to the current statistics snapshot.
    pub fn stats(&self) -> &MetadataIndexStats {
        &self.stats
    }
}

impl Default for StorageMetadataIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_entry(block_id: u64, cid: &str, fields: Vec<MetadataField>) -> MetadataIndexEntry {
        MetadataIndexEntry {
            block_id,
            cid: cid.to_string(),
            fields,
        }
    }

    fn png_entry(block_id: u64) -> MetadataIndexEntry {
        make_entry(
            block_id,
            &format!("cid-{block_id}"),
            vec![
                MetadataField::ContentType("image/png".to_string()),
                MetadataField::Tag("photo".to_string()),
                MetadataField::Owner("alice".to_string()),
                MetadataField::SizeBucket(block_id % 4),
                MetadataField::TickBucket(block_id / 5),
            ],
        )
    }

    // ── insert ────────────────────────────────────────────────────────────────

    #[test]
    fn test_insert_creates_entry() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(png_entry(1));
        assert!(idx.get(1).is_some());
        assert_eq!(idx.stats().total_entries, 1);
    }

    #[test]
    fn test_insert_updates_field_index() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(png_entry(1));
        let key = MetadataField::ContentType("image/png".to_string());
        assert!(idx.field_index.contains_key(&key));
        assert!(idx.field_index[&key].contains(&1));
    }

    #[test]
    fn test_insert_multiple_entries_share_field() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(png_entry(1));
        idx.insert(png_entry(2));
        let key = MetadataField::ContentType("image/png".to_string());
        let ids = &idx.field_index[&key];
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }

    #[test]
    fn test_insert_replaces_existing_entry() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(make_entry(
            10,
            "cid-old",
            vec![MetadataField::Tag("old-tag".to_string())],
        ));
        // Replace with new data.
        idx.insert(make_entry(
            10,
            "cid-new",
            vec![MetadataField::Tag("new-tag".to_string())],
        ));
        assert_eq!(idx.stats().total_entries, 1);
        assert_eq!(idx.get(10).unwrap().cid, "cid-new");
        // Old field must be gone from field_index.
        let old_key = MetadataField::Tag("old-tag".to_string());
        assert!(!idx.field_index.contains_key(&old_key));
        // New field must exist.
        let new_key = MetadataField::Tag("new-tag".to_string());
        assert!(idx.field_index.contains_key(&new_key));
    }

    #[test]
    fn test_insert_replace_updates_stats_correctly() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(make_entry(
            5,
            "cid-5",
            vec![
                MetadataField::Tag("a".to_string()),
                MetadataField::Tag("b".to_string()),
            ],
        ));
        assert_eq!(idx.stats().total_fields_indexed, 2);
        // Replace with 3 fields.
        idx.insert(make_entry(
            5,
            "cid-5",
            vec![
                MetadataField::Tag("x".to_string()),
                MetadataField::Tag("y".to_string()),
                MetadataField::Tag("z".to_string()),
            ],
        ));
        assert_eq!(idx.stats().total_entries, 1);
        assert_eq!(idx.stats().total_fields_indexed, 3);
    }

    // ── remove ────────────────────────────────────────────────────────────────

    #[test]
    fn test_remove_returns_true_when_present() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(png_entry(1));
        assert!(idx.remove(1));
    }

    #[test]
    fn test_remove_returns_false_when_absent() {
        let mut idx = StorageMetadataIndex::new();
        assert!(!idx.remove(999));
    }

    #[test]
    fn test_remove_cleans_field_index() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(png_entry(1));
        idx.remove(1);
        let key = MetadataField::ContentType("image/png".to_string());
        assert!(!idx.field_index.contains_key(&key));
    }

    #[test]
    fn test_remove_partial_field_index_cleanup() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(png_entry(1));
        idx.insert(png_entry(2));
        idx.remove(1);
        let key = MetadataField::ContentType("image/png".to_string());
        // Field still present because block 2 has it.
        assert!(idx.field_index.contains_key(&key));
        assert!(!idx.field_index[&key].contains(&1));
        assert!(idx.field_index[&key].contains(&2));
    }

    #[test]
    fn test_remove_updates_stats() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(png_entry(1));
        let initial_fields = idx.stats().total_fields_indexed;
        idx.remove(1);
        assert_eq!(idx.stats().total_entries, 0);
        assert_eq!(idx.stats().total_fields_indexed, 0);
        assert!(initial_fields > 0);
    }

    // ── get ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_get_some() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(png_entry(42));
        assert!(idx.get(42).is_some());
        assert_eq!(idx.get(42).unwrap().cid, "cid-42");
    }

    #[test]
    fn test_get_none() {
        let idx = StorageMetadataIndex::new();
        assert!(idx.get(0).is_none());
    }

    // ── query: must_have (single field) ──────────────────────────────────────

    #[test]
    fn test_query_must_have_single_field_matches() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(png_entry(1));
        idx.insert(png_entry(2));
        idx.insert(make_entry(
            3,
            "cid-3",
            vec![MetadataField::ContentType("text/plain".to_string())],
        ));

        let q = MetadataQuery {
            must_have: vec![MetadataField::ContentType("image/png".to_string())],
            must_not_have: vec![],
            sort_by: Some(MetadataSortField::BlockId),
            limit: None,
        };
        let results = idx.query(&q);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].block_id, 1);
        assert_eq!(results[1].block_id, 2);
    }

    #[test]
    fn test_query_must_have_single_field_no_match() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(png_entry(1));
        let q = MetadataQuery {
            must_have: vec![MetadataField::Tag("nonexistent".to_string())],
            ..Default::default()
        };
        let results = idx.query(&q);
        assert!(results.is_empty());
    }

    // ── query: must_have (multiple fields / AND) ──────────────────────────────

    #[test]
    fn test_query_must_have_multiple_fields_and_semantics() {
        let mut idx = StorageMetadataIndex::new();
        // block 1: has both png AND photo
        idx.insert(png_entry(1));
        // block 2: has png but NOT photo
        idx.insert(make_entry(
            2,
            "cid-2",
            vec![MetadataField::ContentType("image/png".to_string())],
        ));

        let q = MetadataQuery {
            must_have: vec![
                MetadataField::ContentType("image/png".to_string()),
                MetadataField::Tag("photo".to_string()),
            ],
            ..Default::default()
        };
        let results = idx.query(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].block_id, 1);
    }

    #[test]
    fn test_query_must_have_matched_fields_populated() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(png_entry(1));
        let q = MetadataQuery {
            must_have: vec![
                MetadataField::ContentType("image/png".to_string()),
                MetadataField::Tag("photo".to_string()),
            ],
            sort_by: Some(MetadataSortField::BlockId),
            ..Default::default()
        };
        let results = idx.query(&q);
        assert_eq!(results.len(), 1);
        assert!(results[0]
            .matched_fields
            .contains(&MetadataField::ContentType("image/png".to_string())));
        assert!(results[0]
            .matched_fields
            .contains(&MetadataField::Tag("photo".to_string())));
    }

    // ── query: must_not_have ──────────────────────────────────────────────────

    #[test]
    fn test_query_must_not_have_excludes_entries() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(png_entry(1)); // has Owner("alice")
        idx.insert(make_entry(
            2,
            "cid-2",
            vec![
                MetadataField::ContentType("image/png".to_string()),
                MetadataField::Owner("bob".to_string()),
            ],
        ));

        let q = MetadataQuery {
            must_have: vec![MetadataField::ContentType("image/png".to_string())],
            must_not_have: vec![MetadataField::Owner("alice".to_string())],
            sort_by: Some(MetadataSortField::BlockId),
            limit: None,
        };
        let results = idx.query(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].block_id, 2);
    }

    #[test]
    fn test_query_must_not_have_excludes_all() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(png_entry(1));
        let q = MetadataQuery {
            must_have: vec![MetadataField::ContentType("image/png".to_string())],
            must_not_have: vec![MetadataField::Tag("photo".to_string())],
            ..Default::default()
        };
        let results = idx.query(&q);
        assert!(results.is_empty());
    }

    // ── query: sort ───────────────────────────────────────────────────────────

    #[test]
    fn test_query_sort_by_block_id_ascending() {
        let mut idx = StorageMetadataIndex::new();
        for id in [5u64, 3, 1, 4, 2] {
            idx.insert(make_entry(
                id,
                &format!("cid-{id}"),
                vec![MetadataField::Tag("common".to_string())],
            ));
        }
        let q = MetadataQuery {
            must_have: vec![MetadataField::Tag("common".to_string())],
            sort_by: Some(MetadataSortField::BlockId),
            ..Default::default()
        };
        let results = idx.query(&q);
        let ids: Vec<u64> = results.iter().map(|r| r.block_id).collect();
        assert_eq!(ids, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_query_sort_by_cid_alphabetical_ascending() {
        let mut idx = StorageMetadataIndex::new();
        let entries = vec![
            make_entry(1, "zed", vec![MetadataField::Tag("t".to_string())]),
            make_entry(2, "alpha", vec![MetadataField::Tag("t".to_string())]),
            make_entry(3, "mango", vec![MetadataField::Tag("t".to_string())]),
        ];
        for e in entries {
            idx.insert(e);
        }
        let q = MetadataQuery {
            must_have: vec![MetadataField::Tag("t".to_string())],
            sort_by: Some(MetadataSortField::Cid),
            ..Default::default()
        };
        let results = idx.query(&q);
        let cids: Vec<&str> = results.iter().map(|r| r.cid.as_str()).collect();
        assert_eq!(cids, vec!["alpha", "mango", "zed"]);
    }

    // ── query: limit ──────────────────────────────────────────────────────────

    #[test]
    fn test_query_limit_truncates_results() {
        let mut idx = StorageMetadataIndex::new();
        for id in 1..=10u64 {
            idx.insert(make_entry(
                id,
                &format!("cid-{id}"),
                vec![MetadataField::Tag("bulk".to_string())],
            ));
        }
        let q = MetadataQuery {
            must_have: vec![MetadataField::Tag("bulk".to_string())],
            sort_by: Some(MetadataSortField::BlockId),
            limit: Some(3),
            ..Default::default()
        };
        let results = idx.query(&q);
        assert_eq!(results.len(), 3);
        // First three by block_id.
        assert_eq!(results[0].block_id, 1);
        assert_eq!(results[1].block_id, 2);
        assert_eq!(results[2].block_id, 3);
    }

    #[test]
    fn test_query_limit_zero_returns_empty() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(png_entry(1));
        let q = MetadataQuery {
            must_have: vec![MetadataField::ContentType("image/png".to_string())],
            limit: Some(0),
            ..Default::default()
        };
        let results = idx.query(&q);
        assert!(results.is_empty());
    }

    // ── query: empty must_have returns all ────────────────────────────────────

    #[test]
    fn test_query_empty_must_have_returns_all() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(png_entry(1));
        idx.insert(png_entry(2));
        idx.insert(make_entry(
            3,
            "cid-3",
            vec![MetadataField::Owner("eve".to_string())],
        ));

        let q = MetadataQuery {
            must_have: vec![],
            sort_by: Some(MetadataSortField::BlockId),
            ..Default::default()
        };
        let results = idx.query(&q);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_query_empty_must_have_matched_fields_is_empty() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(png_entry(1));
        let q = MetadataQuery::default();
        let results = idx.query(&q);
        assert_eq!(results.len(), 1);
        assert!(results[0].matched_fields.is_empty());
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_total_fields_indexed() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(make_entry(
            1,
            "a",
            vec![
                MetadataField::Tag("x".to_string()),
                MetadataField::SizeBucket(1),
            ],
        ));
        idx.insert(make_entry(
            2,
            "b",
            vec![MetadataField::Tag("y".to_string())],
        ));
        assert_eq!(idx.stats().total_fields_indexed, 3);
    }

    #[test]
    fn test_stats_total_queries_increments() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(png_entry(1));
        let q = MetadataQuery::default();
        idx.query(&q);
        idx.query(&q);
        idx.query(&q);
        assert_eq!(idx.stats().total_queries, 3);
    }

    #[test]
    fn test_stats_total_entries_after_operations() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(png_entry(1));
        idx.insert(png_entry(2));
        idx.insert(png_entry(3));
        idx.remove(2);
        assert_eq!(idx.stats().total_entries, 2);
    }

    // ── owner / size / tick bucket variants ──────────────────────────────────

    #[test]
    fn test_query_owner_field() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(make_entry(
            1,
            "c1",
            vec![MetadataField::Owner("alice".to_string())],
        ));
        idx.insert(make_entry(
            2,
            "c2",
            vec![MetadataField::Owner("bob".to_string())],
        ));
        let q = MetadataQuery {
            must_have: vec![MetadataField::Owner("alice".to_string())],
            ..Default::default()
        };
        let results = idx.query(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].block_id, 1);
    }

    #[test]
    fn test_query_size_bucket_field() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(make_entry(1, "c1", vec![MetadataField::SizeBucket(0)]));
        idx.insert(make_entry(2, "c2", vec![MetadataField::SizeBucket(1)]));
        idx.insert(make_entry(3, "c3", vec![MetadataField::SizeBucket(0)]));
        let q = MetadataQuery {
            must_have: vec![MetadataField::SizeBucket(0)],
            sort_by: Some(MetadataSortField::BlockId),
            ..Default::default()
        };
        let results = idx.query(&q);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].block_id, 1);
        assert_eq!(results[1].block_id, 3);
    }

    #[test]
    fn test_query_tick_bucket_field() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(make_entry(1, "c1", vec![MetadataField::TickBucket(0)]));
        idx.insert(make_entry(2, "c2", vec![MetadataField::TickBucket(1)]));
        let q = MetadataQuery {
            must_have: vec![MetadataField::TickBucket(1)],
            ..Default::default()
        };
        let results = idx.query(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].block_id, 2);
    }

    // ── edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_query_on_empty_index() {
        let mut idx = StorageMetadataIndex::new();
        let q = MetadataQuery::default();
        let results = idx.query(&q);
        assert!(results.is_empty());
    }

    #[test]
    fn test_insert_entry_with_no_fields() {
        let mut idx = StorageMetadataIndex::new();
        idx.insert(make_entry(1, "c1", vec![]));
        assert_eq!(idx.stats().total_entries, 1);
        assert_eq!(idx.stats().total_fields_indexed, 0);
        // Empty must_have should still return it.
        let q = MetadataQuery {
            sort_by: Some(MetadataSortField::BlockId),
            ..Default::default()
        };
        let results = idx.query(&q);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_default_impl() {
        let idx = StorageMetadataIndex::default();
        assert_eq!(idx.stats().total_entries, 0);
    }
}
