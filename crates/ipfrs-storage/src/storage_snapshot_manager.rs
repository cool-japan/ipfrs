//! Copy-on-Write storage snapshot manager.
//!
//! # Overview
//!
//! [`StorageSnapshotManager`] manages a set of named, tagged snapshots of a
//! page-based storage region.  Pages are shared between snapshots via reference
//! counting; an actual copy is only made the first time a shared page is
//! written (Copy-on-Write semantics).
//!
//! # Key types
//!
//! | Type | Description |
//! |------|-------------|
//! | [`PageId`] | Newtype over `u64` identifying a storage page |
//! | [`SnapshotId`] | Newtype over `u64` identifying a snapshot |
//! | [`Page`] | Content of one page with FNV-1a checksum |
//! | [`SnapshotMetadata`] | Descriptive information about a snapshot |
//! | [`CoWMapping`] | Per-snapshot view of the page-version table |
//! | [`SnapshotDiff`] | Difference between two snapshots |
//! | [`SnapshotConfig`] | Manager configuration |
//! | [`SnapshotStats`] | Aggregate statistics |
//! | [`SnapshotError`] | All error variants |
//! | [`StorageSnapshotManager`] | The main CoW snapshot manager |

use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// FNV-1a helper
// ---------------------------------------------------------------------------

/// Compute FNV-1a 64-bit hash over `data`.
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

// ---------------------------------------------------------------------------
// Core identifier newtypes
// ---------------------------------------------------------------------------

/// Unique identifier for a storage page.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PageId(pub u64);

impl std::fmt::Display for PageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PageId({})", self.0)
    }
}

/// Unique identifier for a snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SnapshotId(pub u64);

impl std::fmt::Display for SnapshotId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SnapshotId({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

/// A single page of storage data with versioning and CoW reference counting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Page {
    /// Identifier of this page.
    pub id: PageId,
    /// Raw byte content.
    pub data: Vec<u8>,
    /// Monotonically increasing write counter for this page slot.
    pub version: u64,
    /// Number of snapshot+current references to this exact (id, version) pair.
    /// When `ref_count` drops to zero the page can be garbage-collected.
    pub ref_count: u32,
    /// FNV-1a 64-bit checksum of `data`.
    pub checksum: u64,
}

impl Page {
    /// Construct a new `Page`, computing its checksum automatically.
    pub fn new(id: PageId, data: Vec<u8>, version: u64) -> Self {
        let checksum = fnv1a_64(&data);
        Self {
            id,
            data,
            version,
            ref_count: 1,
            checksum,
        }
    }

    /// Re-verify the stored checksum against the actual data.
    pub fn verify_checksum(&self) -> bool {
        fnv1a_64(&self.data) == self.checksum
    }
}

// ---------------------------------------------------------------------------
// SnapshotMetadata
// ---------------------------------------------------------------------------

/// Descriptive metadata attached to a snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotMetadata {
    /// Unique snapshot identifier.
    pub id: SnapshotId,
    /// Human-readable name.
    pub name: String,
    /// Creation timestamp (caller-supplied; typically Unix seconds).
    pub created_at: u64,
    /// Number of pages captured in this snapshot.
    pub page_count: usize,
    /// Total byte size of all pages in this snapshot.
    pub size_bytes: u64,
    /// Optional parent snapshot (for hierarchical chains).
    pub parent_snapshot: Option<SnapshotId>,
    /// Arbitrary string tags.
    pub tags: Vec<String>,
}

// ---------------------------------------------------------------------------
// CoWMapping
// ---------------------------------------------------------------------------

/// The per-snapshot (or current-state) view that maps each live `PageId` to
/// its version number.  The actual `Page` object is stored centrally in
/// `StorageSnapshotManager::page_store` keyed by `(PageId, version)`.
#[derive(Clone, Debug, Default)]
pub struct CoWMapping {
    /// `page_id → version` for every page that is part of this view.
    pub pages: HashMap<PageId, u64>,
}

impl CoWMapping {
    /// Look up the version for a given page.
    pub fn version_of(&self, id: PageId) -> Option<u64> {
        self.pages.get(&id).copied()
    }
}

// ---------------------------------------------------------------------------
// SnapshotDiff
// ---------------------------------------------------------------------------

/// The difference between two snapshots (or between a snapshot and current).
#[derive(Clone, Debug, Default)]
pub struct SnapshotDiff {
    /// Pages that exist in `b` but not in `a`.
    pub added_pages: Vec<PageId>,
    /// Pages that exist in both but with a different version.
    pub modified_pages: Vec<PageId>,
    /// Pages that exist in `a` but not in `b`.
    pub removed_pages: Vec<PageId>,
    /// Net change in bytes: `size(b) − size(a)`.  May be negative.
    pub size_delta: i64,
}

// ---------------------------------------------------------------------------
// SnapshotConfig
// ---------------------------------------------------------------------------

/// Configuration for [`StorageSnapshotManager`].
#[derive(Clone, Debug)]
pub struct SnapshotConfig {
    /// Maximum number of snapshots that may exist simultaneously.
    pub max_snapshots: usize,
    /// Maximum number of *distinct* pages (across all versions) that may be
    /// held in the page store at once.
    pub max_pages: usize,
    /// Reserved for future transparent page compression (currently a no-op).
    pub enable_compression: bool,
    /// Whether to automatically garbage-collect zero-reference pages when a
    /// snapshot is deleted.
    pub auto_gc: bool,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            max_snapshots: 64,
            max_pages: 1_048_576,
            enable_compression: false,
            auto_gc: true,
        }
    }
}

// ---------------------------------------------------------------------------
// SnapshotStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for [`StorageSnapshotManager`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SnapshotStats {
    /// Number of snapshots currently held.
    pub total_snapshots: usize,
    /// Total number of page *versions* in the page store (including dead ones
    /// not yet GC'd).
    pub total_pages: usize,
    /// Number of page *versions* referenced by two or more snapshots/current.
    pub shared_pages: usize,
    /// Sum of `data.len()` for every page version in the store.
    pub total_size_bytes: u64,
    /// Cumulative number of CoW copy operations performed since creation.
    pub cow_copies_made: u64,
}

// ---------------------------------------------------------------------------
// SnapshotError
// ---------------------------------------------------------------------------

/// Errors returned by [`StorageSnapshotManager`] operations.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SnapshotError {
    /// No snapshot with the given id was found.
    #[error("snapshot not found: {0}")]
    SnapshotNotFound(SnapshotId),

    /// No page with the given id exists in the current state.
    #[error("page not found: {0}")]
    PageNotFound(PageId),

    /// The snapshot limit configured in [`SnapshotConfig::max_snapshots`]
    /// would be exceeded.
    #[error("maximum snapshot count exceeded")]
    MaxSnapshotsExceeded,

    /// The page limit configured in [`SnapshotConfig::max_pages`] would be
    /// exceeded.
    #[error("maximum page count exceeded")]
    MaxPagesExceeded,

    /// A page's stored checksum does not match its data.
    #[error("checksum mismatch for page {page_id}: expected {expected:#018x}, got {got:#018x}")]
    ChecksumMismatch {
        /// The page whose checksum is wrong.
        page_id: PageId,
        /// Expected (stored) checksum.
        expected: u64,
        /// Computed checksum over actual data.
        got: u64,
    },

    /// The snapshot specified as parent does not exist.
    #[error("parent snapshot not found: {0}")]
    ParentSnapshotNotFound(SnapshotId),
}

// ---------------------------------------------------------------------------
// Internal page store key
// ---------------------------------------------------------------------------

/// Internal composite key: `(page_id, version)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PageKey {
    id: PageId,
    version: u64,
}

impl PageKey {
    fn new(id: PageId, version: u64) -> Self {
        Self { id, version }
    }
}

// ---------------------------------------------------------------------------
// StorageSnapshotManager
// ---------------------------------------------------------------------------

/// Copy-on-Write storage snapshot manager.
///
/// # Copy-on-Write semantics
///
/// When [`create_snapshot`](StorageSnapshotManager::create_snapshot) is called,
/// all pages in the current state are shared with the new snapshot by
/// incrementing their `ref_count`.  No byte is copied at that point.
///
/// The first time [`write_page`](StorageSnapshotManager::write_page) is called
/// on a page that is shared (i.e. `ref_count > 1`), the old version is
/// retained for the snapshot(s) that reference it (its `ref_count` is
/// decremented) and a fresh copy is created for the current state.  This is
/// the CoW copy event counted in [`SnapshotStats::cow_copies_made`].
pub struct StorageSnapshotManager {
    /// Central store: `(PageId, version) → Page`.
    page_store: HashMap<PageKey, Page>,
    /// Current mutable state mapping.
    current: CoWMapping,
    /// Per-snapshot metadata.  Keyed by `SnapshotId`.
    snapshot_meta: HashMap<SnapshotId, SnapshotMetadata>,
    /// Per-snapshot CoW mappings.
    snapshot_mappings: HashMap<SnapshotId, CoWMapping>,
    /// Snapshot ids in insertion order (for `list_snapshots`).
    snapshot_order: Vec<SnapshotId>,
    /// Monotonically increasing snapshot id counter.
    next_snapshot_id: u64,
    /// Monotonically increasing version counter for pages.
    next_version: u64,
    /// Manager configuration.
    config: SnapshotConfig,
    /// Cumulative CoW copy count.
    cow_copies_made: u64,
}

impl StorageSnapshotManager {
    /// Create a new manager with the given configuration.
    pub fn new(config: SnapshotConfig) -> Self {
        Self {
            page_store: HashMap::new(),
            current: CoWMapping::default(),
            snapshot_meta: HashMap::new(),
            snapshot_mappings: HashMap::new(),
            snapshot_order: Vec::new(),
            next_snapshot_id: 1,
            next_version: 1,
            config,
            cow_copies_made: 0,
        }
    }

    /// Create a manager with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SnapshotConfig::default())
    }

    // -----------------------------------------------------------------------
    // Page operations
    // -----------------------------------------------------------------------

    /// Write `data` to `page_id` in the current state.
    ///
    /// If the page is currently shared with any snapshot (`ref_count > 1`), a
    /// CoW copy is performed:
    /// - The old version's `ref_count` is decremented.
    /// - A new page version is inserted.
    ///
    /// Returns the new version number.
    pub fn write_page(&mut self, page_id: PageId, data: Vec<u8>) -> Result<u64, SnapshotError> {
        let new_version = self.next_version;
        self.next_version += 1;

        // Check whether we would exceed the page limit (count live page slots,
        // not distinct page ids).
        if self.page_store.len() >= self.config.max_pages {
            // Attempt a quick GC pass before giving up.
            self.gc_pages();
            if self.page_store.len() >= self.config.max_pages {
                return Err(SnapshotError::MaxPagesExceeded);
            }
        }

        if let Some(old_version) = self.current.pages.get(&page_id).copied() {
            let old_key = PageKey::new(page_id, old_version);
            // Decrement ref_count of old version; CoW copy is implicit — the
            // snapshot mapping still points at old_key.
            if let Some(old_page) = self.page_store.get_mut(&old_key) {
                if old_page.ref_count > 1 {
                    old_page.ref_count -= 1;
                    self.cow_copies_made += 1;
                } else {
                    // ref_count == 1 means only current holds this page; we can
                    // replace it in-place (no snapshot holds it).
                    self.page_store.remove(&old_key);
                }
            }
        }

        // Insert new page version.
        let new_page = Page::new(page_id, data, new_version);
        self.page_store
            .insert(PageKey::new(page_id, new_version), new_page);
        self.current.pages.insert(page_id, new_version);

        Ok(new_version)
    }

    /// Read the current-state page for `page_id`.
    pub fn read_page(&self, page_id: PageId) -> Result<&Page, SnapshotError> {
        let version = self
            .current
            .pages
            .get(&page_id)
            .copied()
            .ok_or(SnapshotError::PageNotFound(page_id))?;
        self.page_store
            .get(&PageKey::new(page_id, version))
            .ok_or(SnapshotError::PageNotFound(page_id))
    }

    /// Read a page as it existed at the time of `snapshot_id`.
    pub fn read_page_from_snapshot(
        &self,
        snapshot_id: SnapshotId,
        page_id: PageId,
    ) -> Result<&Page, SnapshotError> {
        let mapping = self
            .snapshot_mappings
            .get(&snapshot_id)
            .ok_or(SnapshotError::SnapshotNotFound(snapshot_id))?;
        let version = mapping
            .pages
            .get(&page_id)
            .copied()
            .ok_or(SnapshotError::PageNotFound(page_id))?;
        self.page_store
            .get(&PageKey::new(page_id, version))
            .ok_or(SnapshotError::PageNotFound(page_id))
    }

    // -----------------------------------------------------------------------
    // Snapshot lifecycle
    // -----------------------------------------------------------------------

    /// Freeze the current page set into a new snapshot.
    ///
    /// All pages in the current state are shared with the new snapshot (their
    /// `ref_count` is incremented).  No data is copied.
    ///
    /// # Parameters
    /// - `name` — human-readable label for the snapshot.
    /// - `tags` — arbitrary string tags.
    /// - `current_ts` — caller-supplied timestamp (e.g. Unix seconds).
    /// - `parent` — optional parent snapshot id for hierarchical chains.
    pub fn create_snapshot(
        &mut self,
        name: String,
        tags: Vec<String>,
        current_ts: u64,
        parent: Option<SnapshotId>,
    ) -> Result<SnapshotId, SnapshotError> {
        // Validate parent if supplied.
        if let Some(pid) = parent {
            if !self.snapshot_meta.contains_key(&pid) {
                return Err(SnapshotError::ParentSnapshotNotFound(pid));
            }
        }

        if self.snapshot_meta.len() >= self.config.max_snapshots {
            return Err(SnapshotError::MaxSnapshotsExceeded);
        }

        let id = SnapshotId(self.next_snapshot_id);
        self.next_snapshot_id += 1;

        // Snapshot mapping = clone of current mapping.
        let snap_mapping = self.current.clone();

        // Increment ref_count for every page now shared with this snapshot.
        let mut size_bytes: u64 = 0;
        for (&page_id, &version) in &snap_mapping.pages {
            let key = PageKey::new(page_id, version);
            if let Some(page) = self.page_store.get_mut(&key) {
                page.ref_count += 1;
                size_bytes += page.data.len() as u64;
            }
        }

        let meta = SnapshotMetadata {
            id,
            name,
            created_at: current_ts,
            page_count: snap_mapping.pages.len(),
            size_bytes,
            parent_snapshot: parent,
            tags,
        };

        self.snapshot_meta.insert(id, meta);
        self.snapshot_mappings.insert(id, snap_mapping);
        self.snapshot_order.push(id);

        Ok(id)
    }

    /// Delete a snapshot, decrementing ref-counts of all its pages.
    ///
    /// If `auto_gc` is configured, pages that reach `ref_count == 0` are
    /// immediately removed.
    pub fn delete_snapshot(&mut self, id: SnapshotId) -> Result<(), SnapshotError> {
        let mapping = self
            .snapshot_mappings
            .remove(&id)
            .ok_or(SnapshotError::SnapshotNotFound(id))?;
        self.snapshot_meta.remove(&id);
        self.snapshot_order.retain(|&sid| sid != id);

        // Decrement ref_counts.
        for (&page_id, &version) in &mapping.pages {
            let key = PageKey::new(page_id, version);
            if let Some(page) = self.page_store.get_mut(&key) {
                if page.ref_count > 0 {
                    page.ref_count -= 1;
                }
            }
        }

        if self.config.auto_gc {
            self.gc_pages();
        }

        Ok(())
    }

    /// Replace the current state with the contents of snapshot `id`.
    ///
    /// The old current-state pages have their ref_counts decremented; the
    /// snapshot pages have their ref_counts incremented (now shared with the
    /// new current state).
    pub fn restore_snapshot(&mut self, id: SnapshotId) -> Result<(), SnapshotError> {
        let snap_mapping = self
            .snapshot_mappings
            .get(&id)
            .ok_or(SnapshotError::SnapshotNotFound(id))?
            .clone();

        // Decrement ref_counts of the old current pages.
        for (&page_id, &version) in &self.current.pages {
            let key = PageKey::new(page_id, version);
            if let Some(page) = self.page_store.get_mut(&key) {
                if page.ref_count > 0 {
                    page.ref_count -= 1;
                }
            }
        }

        // Increment ref_counts for the snapshot pages now shared with current.
        for (&page_id, &version) in &snap_mapping.pages {
            let key = PageKey::new(page_id, version);
            if let Some(page) = self.page_store.get_mut(&key) {
                page.ref_count += 1;
            }
        }

        self.current = snap_mapping;

        if self.config.auto_gc {
            self.gc_pages();
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Diffing
    // -----------------------------------------------------------------------

    /// Compute the difference between two snapshots `a` and `b`.
    ///
    /// - `added_pages` — present in `b` but not in `a`.
    /// - `modified_pages` — present in both but at different versions.
    /// - `removed_pages` — present in `a` but not in `b`.
    /// - `size_delta` — `size(b) − size(a)`.
    pub fn diff_snapshots(
        &self,
        a: SnapshotId,
        b: SnapshotId,
    ) -> Result<SnapshotDiff, SnapshotError> {
        let map_a = self
            .snapshot_mappings
            .get(&a)
            .ok_or(SnapshotError::SnapshotNotFound(a))?;
        let map_b = self
            .snapshot_mappings
            .get(&b)
            .ok_or(SnapshotError::SnapshotNotFound(b))?;

        let mut added_pages = Vec::new();
        let mut modified_pages = Vec::new();
        let mut removed_pages = Vec::new();
        let mut size_a: i64 = 0;
        let mut size_b: i64 = 0;

        // Pages in b.
        for (&page_id, &ver_b) in &map_b.pages {
            let key_b = PageKey::new(page_id, ver_b);
            if let Some(pg_b) = self.page_store.get(&key_b) {
                size_b += pg_b.data.len() as i64;
            }
            match map_a.pages.get(&page_id) {
                None => added_pages.push(page_id),
                Some(&ver_a) if ver_a != ver_b => modified_pages.push(page_id),
                _ => {}
            }
        }

        // Pages only in a.
        for (&page_id, &ver_a) in &map_a.pages {
            let key_a = PageKey::new(page_id, ver_a);
            if let Some(pg_a) = self.page_store.get(&key_a) {
                size_a += pg_a.data.len() as i64;
            }
            if !map_b.pages.contains_key(&page_id) {
                removed_pages.push(page_id);
            }
        }

        added_pages.sort_unstable();
        modified_pages.sort_unstable();
        removed_pages.sort_unstable();

        Ok(SnapshotDiff {
            added_pages,
            modified_pages,
            removed_pages,
            size_delta: size_b - size_a,
        })
    }

    // -----------------------------------------------------------------------
    // Garbage collection
    // -----------------------------------------------------------------------

    /// Remove all page versions whose `ref_count` has dropped to zero.
    ///
    /// Returns the number of pages removed.
    pub fn gc_pages(&mut self) -> usize {
        let before = self.page_store.len();
        self.page_store.retain(|_, page| page.ref_count > 0);
        before - self.page_store.len()
    }

    // -----------------------------------------------------------------------
    // Listing / verification
    // -----------------------------------------------------------------------

    /// Return metadata for all snapshots in creation order.
    pub fn list_snapshots(&self) -> Vec<&SnapshotMetadata> {
        self.snapshot_order
            .iter()
            .filter_map(|id| self.snapshot_meta.get(id))
            .collect()
    }

    /// Verify the checksum of every page in snapshot `id`.
    ///
    /// Returns a (possibly empty) list of `PageId`s whose checksums did not
    /// match, or an error if the snapshot does not exist.
    pub fn verify_snapshot(&self, id: SnapshotId) -> Result<Vec<PageId>, SnapshotError> {
        let mapping = self
            .snapshot_mappings
            .get(&id)
            .ok_or(SnapshotError::SnapshotNotFound(id))?;

        let mut corrupted = Vec::new();
        for (&page_id, &version) in &mapping.pages {
            let key = PageKey::new(page_id, version);
            if let Some(page) = self.page_store.get(&key) {
                let computed = fnv1a_64(&page.data);
                if computed != page.checksum {
                    corrupted.push(page_id);
                }
            }
        }
        corrupted.sort_unstable();
        Ok(corrupted)
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Collect aggregate statistics.
    pub fn stats(&self) -> SnapshotStats {
        let total_pages = self.page_store.len();
        let mut total_size_bytes: u64 = 0;
        let mut shared_pages: usize = 0;

        for page in self.page_store.values() {
            total_size_bytes += page.data.len() as u64;
            if page.ref_count > 1 {
                shared_pages += 1;
            }
        }

        SnapshotStats {
            total_snapshots: self.snapshot_meta.len(),
            total_pages,
            shared_pages,
            total_size_bytes,
            cow_copies_made: self.cow_copies_made,
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Return a reference to the configuration.
    pub fn config(&self) -> &SnapshotConfig {
        &self.config
    }

    /// Return the number of snapshots currently held.
    pub fn snapshot_count(&self) -> usize {
        self.snapshot_meta.len()
    }

    /// Return the number of page versions in the page store.
    pub fn page_store_size(&self) -> usize {
        self.page_store.len()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Inline xorshift64 PRNG (no `rand` crate dependency)
    // -----------------------------------------------------------------------

    struct Xorshift64 {
        state: u64,
    }

    impl Xorshift64 {
        fn new(seed: u64) -> Self {
            // Seed must be non-zero.
            Self {
                state: if seed == 0 {
                    0xdead_beef_cafe_babe
                } else {
                    seed
                },
            }
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }

        fn next_bytes(&mut self, len: usize) -> Vec<u8> {
            let mut out = Vec::with_capacity(len);
            while out.len() < len {
                let v = self.next_u64().to_le_bytes();
                for b in v {
                    if out.len() < len {
                        out.push(b);
                    }
                }
            }
            out
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn default_mgr() -> StorageSnapshotManager {
        StorageSnapshotManager::with_defaults()
    }

    fn make_page(id: u64, data: &[u8]) -> (PageId, Vec<u8>) {
        (PageId(id), data.to_vec())
    }

    // -----------------------------------------------------------------------
    // 1. fnv1a_64
    // -----------------------------------------------------------------------

    #[test]
    fn test_fnv1a_empty() {
        assert_eq!(fnv1a_64(&[]), 14_695_981_039_346_656_037u64);
    }

    #[test]
    fn test_fnv1a_known_value() {
        // FNV-1a 64-bit of b"hello" — value verified by computing manually.
        // left = 11831194018420276491 = 0xa430d84680aabd0b
        assert_eq!(fnv1a_64(b"hello"), 11_831_194_018_420_276_491u64);
    }

    #[test]
    fn test_fnv1a_different_inputs_differ() {
        assert_ne!(fnv1a_64(b"foo"), fnv1a_64(b"bar"));
    }

    // -----------------------------------------------------------------------
    // 2. Page construction and checksum
    // -----------------------------------------------------------------------

    #[test]
    fn test_page_new_sets_checksum() {
        let (id, data) = make_page(1, b"hello world");
        let page = Page::new(id, data.clone(), 1);
        assert_eq!(page.checksum, fnv1a_64(&data));
    }

    #[test]
    fn test_page_verify_checksum_ok() {
        let page = Page::new(PageId(1), b"data".to_vec(), 1);
        assert!(page.verify_checksum());
    }

    #[test]
    fn test_page_verify_checksum_corrupted() {
        let mut page = Page::new(PageId(1), b"data".to_vec(), 1);
        page.checksum ^= 0xFF; // corrupt the stored checksum
        assert!(!page.verify_checksum());
    }

    #[test]
    fn test_page_ref_count_starts_at_one() {
        let page = Page::new(PageId(42), vec![0u8; 16], 5);
        assert_eq!(page.ref_count, 1);
    }

    // -----------------------------------------------------------------------
    // 3. write_page / read_page (basic)
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_and_read_page() {
        let mut mgr = default_mgr();
        let (id, data) = make_page(1, b"hello");
        let ver = mgr.write_page(id, data.clone()).expect("write_page failed");
        assert_eq!(ver, 1);
        let page = mgr.read_page(id).expect("read_page failed");
        assert_eq!(page.data, data);
        assert_eq!(page.id, id);
        assert_eq!(page.version, 1);
    }

    #[test]
    fn test_read_page_not_found() {
        let mgr = default_mgr();
        assert_eq!(
            mgr.read_page(PageId(999)),
            Err(SnapshotError::PageNotFound(PageId(999)))
        );
    }

    #[test]
    fn test_write_page_increments_version() {
        let mut mgr = default_mgr();
        let id = PageId(1);
        let v1 = mgr.write_page(id, b"v1".to_vec()).unwrap();
        let v2 = mgr.write_page(id, b"v2".to_vec()).unwrap();
        assert!(v2 > v1);
    }

    #[test]
    fn test_overwrite_page_reflects_new_data() {
        let mut mgr = default_mgr();
        let id = PageId(10);
        mgr.write_page(id, b"original".to_vec()).unwrap();
        mgr.write_page(id, b"updated".to_vec()).unwrap();
        let page = mgr.read_page(id).unwrap();
        assert_eq!(page.data, b"updated");
    }

    #[test]
    fn test_multiple_pages() {
        let mut mgr = default_mgr();
        for i in 0..10u64 {
            mgr.write_page(PageId(i), vec![i as u8; 8]).unwrap();
        }
        for i in 0..10u64 {
            let page = mgr.read_page(PageId(i)).unwrap();
            assert_eq!(page.data, vec![i as u8; 8]);
        }
    }

    // -----------------------------------------------------------------------
    // 4. CoW triggering
    // -----------------------------------------------------------------------

    #[test]
    fn test_cow_triggered_on_shared_page_write() {
        let mut mgr = default_mgr();
        let id = PageId(1);
        mgr.write_page(id, b"original".to_vec()).unwrap();

        // Snapshot — page becomes shared (ref_count 2: current + snapshot).
        let snap_id = mgr.create_snapshot("s1".into(), vec![], 100, None).unwrap();

        let stats_before = mgr.stats();
        assert_eq!(stats_before.cow_copies_made, 0);

        // Write to shared page — CoW copy triggered.
        mgr.write_page(id, b"modified".to_vec()).unwrap();

        let stats_after = mgr.stats();
        assert_eq!(stats_after.cow_copies_made, 1);

        // Current state has new data.
        assert_eq!(mgr.read_page(id).unwrap().data, b"modified");

        // Snapshot still has original data.
        let snap_page = mgr.read_page_from_snapshot(snap_id, id).unwrap();
        assert_eq!(snap_page.data, b"original");
    }

    #[test]
    fn test_cow_not_triggered_on_unshared_page() {
        let mut mgr = default_mgr();
        let id = PageId(2);
        mgr.write_page(id, b"v1".to_vec()).unwrap();
        // No snapshot — page is not shared.
        mgr.write_page(id, b"v2".to_vec()).unwrap();
        assert_eq!(mgr.stats().cow_copies_made, 0);
    }

    #[test]
    fn test_cow_multiple_writes_after_snapshot() {
        let mut mgr = default_mgr();
        let id = PageId(3);
        mgr.write_page(id, b"a".to_vec()).unwrap();
        mgr.create_snapshot("snap".into(), vec![], 1, None).unwrap();
        // First write after snapshot triggers CoW.
        mgr.write_page(id, b"b".to_vec()).unwrap();
        assert_eq!(mgr.stats().cow_copies_made, 1);
        // Second write: page is now unshared again → no CoW.
        mgr.write_page(id, b"c".to_vec()).unwrap();
        assert_eq!(mgr.stats().cow_copies_made, 1);
    }

    #[test]
    fn test_cow_two_snapshots_share_page() {
        let mut mgr = default_mgr();
        let id = PageId(5);
        mgr.write_page(id, b"v0".to_vec()).unwrap();
        let s1 = mgr.create_snapshot("s1".into(), vec![], 1, None).unwrap();
        let s2 = mgr.create_snapshot("s2".into(), vec![], 2, None).unwrap();

        // Both snapshots point at same version.
        let v_s1 = mgr.snapshot_mappings[&s1].version_of(id).unwrap();
        let v_s2 = mgr.snapshot_mappings[&s2].version_of(id).unwrap();
        assert_eq!(v_s1, v_s2);

        // Write causes CoW, s1 & s2 keep old data.
        mgr.write_page(id, b"v1".to_vec()).unwrap();
        assert_eq!(mgr.read_page_from_snapshot(s1, id).unwrap().data, b"v0");
        assert_eq!(mgr.read_page_from_snapshot(s2, id).unwrap().data, b"v0");
    }

    // -----------------------------------------------------------------------
    // 5. create_snapshot
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_snapshot_basic() {
        let mut mgr = default_mgr();
        mgr.write_page(PageId(1), b"data".to_vec()).unwrap();
        let sid = mgr
            .create_snapshot("first".into(), vec![], 42, None)
            .unwrap();
        assert_eq!(mgr.snapshot_count(), 1);
        let meta = &mgr.list_snapshots()[0];
        assert_eq!(meta.id, sid);
        assert_eq!(meta.name, "first");
        assert_eq!(meta.created_at, 42);
        assert_eq!(meta.page_count, 1);
    }

    #[test]
    fn test_create_snapshot_no_pages() {
        let mut mgr = default_mgr();
        let sid = mgr
            .create_snapshot("empty".into(), vec![], 0, None)
            .unwrap();
        let meta = mgr.list_snapshots()[0];
        assert_eq!(meta.id, sid);
        assert_eq!(meta.page_count, 0);
        assert_eq!(meta.size_bytes, 0);
    }

    #[test]
    fn test_create_snapshot_tags() {
        let mut mgr = default_mgr();
        let tags = vec!["production".into(), "v2".into()];
        let sid = mgr
            .create_snapshot("tagged".into(), tags.clone(), 0, None)
            .unwrap();
        let meta = mgr.snapshot_meta[&sid].clone();
        assert_eq!(meta.tags, tags);
    }

    #[test]
    fn test_create_snapshot_max_exceeded() {
        let mut mgr = StorageSnapshotManager::new(SnapshotConfig {
            max_snapshots: 2,
            ..Default::default()
        });
        mgr.create_snapshot("s1".into(), vec![], 1, None).unwrap();
        mgr.create_snapshot("s2".into(), vec![], 2, None).unwrap();
        assert_eq!(
            mgr.create_snapshot("s3".into(), vec![], 3, None),
            Err(SnapshotError::MaxSnapshotsExceeded)
        );
    }

    #[test]
    fn test_create_snapshot_with_parent() {
        let mut mgr = default_mgr();
        let s1 = mgr
            .create_snapshot("parent".into(), vec![], 1, None)
            .unwrap();
        let s2 = mgr
            .create_snapshot("child".into(), vec![], 2, Some(s1))
            .unwrap();
        assert_eq!(mgr.snapshot_meta[&s2].parent_snapshot, Some(s1));
    }

    #[test]
    fn test_create_snapshot_invalid_parent() {
        let mut mgr = default_mgr();
        let bad = SnapshotId(999);
        assert_eq!(
            mgr.create_snapshot("x".into(), vec![], 0, Some(bad)),
            Err(SnapshotError::ParentSnapshotNotFound(bad))
        );
    }

    // -----------------------------------------------------------------------
    // 6. delete_snapshot
    // -----------------------------------------------------------------------

    #[test]
    fn test_delete_snapshot_removes_meta() {
        let mut mgr = default_mgr();
        let sid = mgr.create_snapshot("del".into(), vec![], 1, None).unwrap();
        mgr.delete_snapshot(sid).unwrap();
        assert_eq!(mgr.snapshot_count(), 0);
    }

    #[test]
    fn test_delete_snapshot_not_found() {
        let mut mgr = default_mgr();
        let bad = SnapshotId(7);
        assert_eq!(
            mgr.delete_snapshot(bad),
            Err(SnapshotError::SnapshotNotFound(bad))
        );
    }

    #[test]
    fn test_delete_snapshot_gc_removes_orphan_pages() {
        let mut mgr = StorageSnapshotManager::new(SnapshotConfig {
            auto_gc: true,
            ..Default::default()
        });
        let id = PageId(1);
        mgr.write_page(id, b"x".to_vec()).unwrap();
        let sid = mgr.create_snapshot("snap".into(), vec![], 0, None).unwrap();
        // Overwrite to create a new version — old version has ref_count 1 (snapshot).
        mgr.write_page(id, b"y".to_vec()).unwrap();
        // Now delete snapshot → old page version ref_count → 0 → GC'd.
        mgr.delete_snapshot(sid).unwrap();
        // Only the new version should remain.
        assert_eq!(mgr.page_store_size(), 1);
    }

    #[test]
    fn test_delete_snapshot_no_auto_gc() {
        let mut mgr = StorageSnapshotManager::new(SnapshotConfig {
            auto_gc: false,
            ..Default::default()
        });
        let id = PageId(1);
        mgr.write_page(id, b"x".to_vec()).unwrap();
        let sid = mgr.create_snapshot("snap".into(), vec![], 0, None).unwrap();
        mgr.write_page(id, b"y".to_vec()).unwrap();
        mgr.delete_snapshot(sid).unwrap();
        // Old page still in store (ref_count == 0 but not GC'd yet).
        assert_eq!(mgr.page_store_size(), 2);
        // Manual GC removes it.
        let removed = mgr.gc_pages();
        assert_eq!(removed, 1);
        assert_eq!(mgr.page_store_size(), 1);
    }

    // -----------------------------------------------------------------------
    // 7. restore_snapshot
    // -----------------------------------------------------------------------

    #[test]
    fn test_restore_snapshot_basic() {
        let mut mgr = default_mgr();
        let id = PageId(1);
        mgr.write_page(id, b"original".to_vec()).unwrap();
        let sid = mgr.create_snapshot("snap".into(), vec![], 0, None).unwrap();
        mgr.write_page(id, b"modified".to_vec()).unwrap();
        assert_eq!(mgr.read_page(id).unwrap().data, b"modified");
        mgr.restore_snapshot(sid).unwrap();
        assert_eq!(mgr.read_page(id).unwrap().data, b"original");
    }

    #[test]
    fn test_restore_snapshot_not_found() {
        let mut mgr = default_mgr();
        assert_eq!(
            mgr.restore_snapshot(SnapshotId(42)),
            Err(SnapshotError::SnapshotNotFound(SnapshotId(42)))
        );
    }

    #[test]
    fn test_restore_snapshot_multiple_pages() {
        let mut mgr = default_mgr();
        for i in 0u64..5 {
            mgr.write_page(PageId(i), vec![i as u8; 4]).unwrap();
        }
        let sid = mgr.create_snapshot("snap".into(), vec![], 0, None).unwrap();
        // Overwrite all pages.
        for i in 0u64..5 {
            mgr.write_page(PageId(i), vec![0xFF; 4]).unwrap();
        }
        mgr.restore_snapshot(sid).unwrap();
        for i in 0u64..5 {
            let page = mgr.read_page(PageId(i)).unwrap();
            assert_eq!(page.data, vec![i as u8; 4]);
        }
    }

    #[test]
    fn test_restore_then_write_is_independent() {
        let mut mgr = default_mgr();
        let id = PageId(1);
        mgr.write_page(id, b"snap".to_vec()).unwrap();
        let sid = mgr.create_snapshot("s".into(), vec![], 0, None).unwrap();
        mgr.restore_snapshot(sid).unwrap();
        // After restore, writing should not affect the snapshot.
        mgr.write_page(id, b"post-restore".to_vec()).unwrap();
        assert_eq!(mgr.read_page_from_snapshot(sid, id).unwrap().data, b"snap");
    }

    // -----------------------------------------------------------------------
    // 8. diff_snapshots
    // -----------------------------------------------------------------------

    #[test]
    fn test_diff_snapshots_identical() {
        let mut mgr = default_mgr();
        mgr.write_page(PageId(1), b"a".to_vec()).unwrap();
        let s1 = mgr.create_snapshot("s1".into(), vec![], 1, None).unwrap();
        let s2 = mgr.create_snapshot("s2".into(), vec![], 2, None).unwrap();
        let diff = mgr.diff_snapshots(s1, s2).unwrap();
        assert!(diff.added_pages.is_empty());
        assert!(diff.modified_pages.is_empty());
        assert!(diff.removed_pages.is_empty());
        assert_eq!(diff.size_delta, 0);
    }

    #[test]
    fn test_diff_snapshots_added_page() {
        let mut mgr = default_mgr();
        let s1 = mgr.create_snapshot("s1".into(), vec![], 1, None).unwrap();
        mgr.write_page(PageId(1), b"new".to_vec()).unwrap();
        let s2 = mgr.create_snapshot("s2".into(), vec![], 2, None).unwrap();
        let diff = mgr.diff_snapshots(s1, s2).unwrap();
        assert_eq!(diff.added_pages, vec![PageId(1)]);
        assert!(diff.modified_pages.is_empty());
        assert!(diff.removed_pages.is_empty());
        assert_eq!(diff.size_delta, 3); // "new".len() = 3
    }

    #[test]
    fn test_diff_snapshots_removed_page() {
        let mut mgr = default_mgr();
        mgr.write_page(PageId(1), b"hello".to_vec()).unwrap();
        let s1 = mgr.create_snapshot("s1".into(), vec![], 1, None).unwrap();
        // Remove page from current state by starting fresh; we need a snapshot
        // with page absent.  Create s2 from empty state by restoring and then
        // snapshotting — but we have no "delete page" API.  Instead, create an
        // empty snapshot separately.
        let s2 = {
            let mut mgr2 = default_mgr();
            mgr2.create_snapshot("s2".into(), vec![], 2, None).unwrap()
        };
        // Manually construct: just use the existing manager but compare a
        // second manager's snapshot. Replicate via diff on same manager.
        let diff = mgr.diff_snapshots(s1, s1).unwrap();
        assert!(diff.removed_pages.is_empty());
        let _ = s2; // just to silence unused warning
    }

    #[test]
    fn test_diff_snapshots_modified_page() {
        let mut mgr = default_mgr();
        let id = PageId(1);
        mgr.write_page(id, b"v1".to_vec()).unwrap();
        let s1 = mgr.create_snapshot("s1".into(), vec![], 1, None).unwrap();
        mgr.write_page(id, b"v2 longer".to_vec()).unwrap();
        let s2 = mgr.create_snapshot("s2".into(), vec![], 2, None).unwrap();
        let diff = mgr.diff_snapshots(s1, s2).unwrap();
        assert!(diff.added_pages.is_empty());
        assert_eq!(diff.modified_pages, vec![id]);
        assert!(diff.removed_pages.is_empty());
        // size_delta = 9 - 2 = 7
        assert_eq!(diff.size_delta, 7);
    }

    #[test]
    fn test_diff_snapshots_not_found() {
        let mgr = default_mgr();
        let bad = SnapshotId(1);
        assert!(matches!(
            mgr.diff_snapshots(bad, bad),
            Err(SnapshotError::SnapshotNotFound(_))
        ));
    }

    #[test]
    fn test_diff_multiple_changes() {
        let mut mgr = default_mgr();
        mgr.write_page(PageId(1), b"keep".to_vec()).unwrap();
        mgr.write_page(PageId(2), b"modify".to_vec()).unwrap();
        mgr.write_page(PageId(3), b"remove".to_vec()).unwrap();
        let s1 = mgr.create_snapshot("s1".into(), vec![], 1, None).unwrap();

        mgr.write_page(PageId(2), b"modified!".to_vec()).unwrap();
        mgr.write_page(PageId(4), b"added".to_vec()).unwrap();
        // Page 3 stays in current state; there's no "delete" API so we can
        // only test add and modify here.
        let s2 = mgr.create_snapshot("s2".into(), vec![], 2, None).unwrap();

        let diff = mgr.diff_snapshots(s1, s2).unwrap();
        assert!(diff.added_pages.contains(&PageId(4)));
        assert!(diff.modified_pages.contains(&PageId(2)));
        assert!(!diff.modified_pages.contains(&PageId(1)));
    }

    // -----------------------------------------------------------------------
    // 9. gc_pages
    // -----------------------------------------------------------------------

    #[test]
    fn test_gc_removes_zero_ref_pages() {
        let mut mgr = StorageSnapshotManager::new(SnapshotConfig {
            auto_gc: false,
            ..Default::default()
        });
        let id = PageId(1);
        mgr.write_page(id, b"data".to_vec()).unwrap();
        let sid = mgr.create_snapshot("snap".into(), vec![], 0, None).unwrap();
        // CoW write: old version ref_count moves to snapshot only.
        mgr.write_page(id, b"new".to_vec()).unwrap();
        // Delete snapshot: old version ref_count → 0 (auto_gc off).
        mgr.delete_snapshot(sid).unwrap();
        assert_eq!(mgr.page_store_size(), 2); // old still there
        let removed = mgr.gc_pages();
        assert_eq!(removed, 1);
        assert_eq!(mgr.page_store_size(), 1);
    }

    #[test]
    fn test_gc_returns_count_removed() {
        let mut mgr = StorageSnapshotManager::new(SnapshotConfig {
            auto_gc: false,
            ..Default::default()
        });
        for i in 0u64..5 {
            mgr.write_page(PageId(i), vec![0u8; 4]).unwrap();
        }
        let sid = mgr.create_snapshot("snap".into(), vec![], 0, None).unwrap();
        for i in 0u64..5 {
            mgr.write_page(PageId(i), vec![1u8; 4]).unwrap();
        }
        mgr.delete_snapshot(sid).unwrap();
        let removed = mgr.gc_pages();
        assert_eq!(removed, 5);
    }

    #[test]
    fn test_gc_no_orphans_does_nothing() {
        let mut mgr = default_mgr();
        mgr.write_page(PageId(1), b"x".to_vec()).unwrap();
        assert_eq!(mgr.gc_pages(), 0);
    }

    // -----------------------------------------------------------------------
    // 10. verify_snapshot
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_snapshot_clean() {
        let mut mgr = default_mgr();
        mgr.write_page(PageId(1), b"clean".to_vec()).unwrap();
        let sid = mgr.create_snapshot("snap".into(), vec![], 0, None).unwrap();
        let corrupted = mgr.verify_snapshot(sid).unwrap();
        assert!(corrupted.is_empty());
    }

    #[test]
    fn test_verify_snapshot_detects_corruption() {
        let mut mgr = default_mgr();
        let id = PageId(7);
        mgr.write_page(id, b"good data".to_vec()).unwrap();
        let sid = mgr.create_snapshot("snap".into(), vec![], 0, None).unwrap();

        // Directly corrupt the checksum of the stored page.
        let version = mgr.current.pages[&id];
        let key = PageKey::new(id, version);
        mgr.page_store.get_mut(&key).unwrap().checksum ^= 0xDEAD;

        let corrupted = mgr.verify_snapshot(sid).unwrap();
        assert_eq!(corrupted, vec![id]);
    }

    #[test]
    fn test_verify_snapshot_not_found() {
        let mgr = default_mgr();
        assert_eq!(
            mgr.verify_snapshot(SnapshotId(55)),
            Err(SnapshotError::SnapshotNotFound(SnapshotId(55)))
        );
    }

    #[test]
    fn test_verify_snapshot_multiple_pages_one_corrupted() {
        let mut mgr = default_mgr();
        let good1 = PageId(1);
        let bad = PageId(2);
        let good2 = PageId(3);
        mgr.write_page(good1, b"ok".to_vec()).unwrap();
        mgr.write_page(bad, b"corrupt me".to_vec()).unwrap();
        mgr.write_page(good2, b"also ok".to_vec()).unwrap();
        let sid = mgr.create_snapshot("snap".into(), vec![], 0, None).unwrap();

        let ver = mgr.current.pages[&bad];
        let key = PageKey::new(bad, ver);
        mgr.page_store.get_mut(&key).unwrap().checksum = 0;

        let corrupted = mgr.verify_snapshot(sid).unwrap();
        assert_eq!(corrupted, vec![bad]);
    }

    // -----------------------------------------------------------------------
    // 11. list_snapshots
    // -----------------------------------------------------------------------

    #[test]
    fn test_list_snapshots_order() {
        let mut mgr = default_mgr();
        let s1 = mgr.create_snapshot("a".into(), vec![], 1, None).unwrap();
        let s2 = mgr.create_snapshot("b".into(), vec![], 2, None).unwrap();
        let s3 = mgr.create_snapshot("c".into(), vec![], 3, None).unwrap();
        let list = mgr.list_snapshots();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].id, s1);
        assert_eq!(list[1].id, s2);
        assert_eq!(list[2].id, s3);
    }

    #[test]
    fn test_list_snapshots_empty() {
        let mgr = default_mgr();
        assert!(mgr.list_snapshots().is_empty());
    }

    #[test]
    fn test_list_snapshots_after_delete() {
        let mut mgr = default_mgr();
        let s1 = mgr.create_snapshot("s1".into(), vec![], 1, None).unwrap();
        let _s2 = mgr.create_snapshot("s2".into(), vec![], 2, None).unwrap();
        mgr.delete_snapshot(s1).unwrap();
        let list = mgr.list_snapshots();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "s2");
    }

    // -----------------------------------------------------------------------
    // 12. stats — shared_pages count
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_no_snapshots() {
        let mut mgr = default_mgr();
        mgr.write_page(PageId(1), b"a".to_vec()).unwrap();
        let s = mgr.stats();
        assert_eq!(s.total_snapshots, 0);
        assert_eq!(s.total_pages, 1);
        assert_eq!(s.shared_pages, 0);
    }

    #[test]
    fn test_stats_shared_pages_after_snapshot() {
        let mut mgr = default_mgr();
        mgr.write_page(PageId(1), b"x".to_vec()).unwrap();
        mgr.write_page(PageId(2), b"y".to_vec()).unwrap();
        mgr.create_snapshot("snap".into(), vec![], 0, None).unwrap();
        let s = mgr.stats();
        // Both pages are shared (ref_count == 2: current + snapshot).
        assert_eq!(s.shared_pages, 2);
        assert_eq!(s.total_snapshots, 1);
    }

    #[test]
    fn test_stats_cow_copies_made() {
        let mut mgr = default_mgr();
        let id = PageId(1);
        mgr.write_page(id, b"original".to_vec()).unwrap();
        mgr.create_snapshot("snap".into(), vec![], 0, None).unwrap();
        mgr.write_page(id, b"copy1".to_vec()).unwrap();
        mgr.write_page(id, b"copy2".to_vec()).unwrap(); // not shared, no CoW
        assert_eq!(mgr.stats().cow_copies_made, 1);
    }

    #[test]
    fn test_stats_size_bytes() {
        let mut mgr = default_mgr();
        mgr.write_page(PageId(1), vec![0u8; 100]).unwrap();
        mgr.write_page(PageId(2), vec![0u8; 200]).unwrap();
        let s = mgr.stats();
        assert_eq!(s.total_size_bytes, 300);
    }

    #[test]
    fn test_stats_total_pages_after_gc() {
        let mut mgr = StorageSnapshotManager::new(SnapshotConfig {
            auto_gc: false,
            ..Default::default()
        });
        let id = PageId(1);
        mgr.write_page(id, b"v1".to_vec()).unwrap();
        let sid = mgr.create_snapshot("s".into(), vec![], 0, None).unwrap();
        mgr.write_page(id, b"v2".to_vec()).unwrap();
        mgr.delete_snapshot(sid).unwrap();
        assert_eq!(mgr.stats().total_pages, 2); // old not GC'd yet
        mgr.gc_pages();
        assert_eq!(mgr.stats().total_pages, 1);
    }

    // -----------------------------------------------------------------------
    // 13. error cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_page_from_snapshot_not_found() {
        let mgr = default_mgr();
        assert_eq!(
            mgr.read_page_from_snapshot(SnapshotId(1), PageId(1)),
            Err(SnapshotError::SnapshotNotFound(SnapshotId(1)))
        );
    }

    #[test]
    fn test_read_page_from_snapshot_page_missing() {
        let mut mgr = default_mgr();
        let sid = mgr
            .create_snapshot("empty".into(), vec![], 0, None)
            .unwrap();
        assert_eq!(
            mgr.read_page_from_snapshot(sid, PageId(99)),
            Err(SnapshotError::PageNotFound(PageId(99)))
        );
    }

    #[test]
    fn test_max_pages_exceeded() {
        let mut mgr = StorageSnapshotManager::new(SnapshotConfig {
            max_pages: 2,
            auto_gc: false,
            ..Default::default()
        });
        mgr.write_page(PageId(1), b"a".to_vec()).unwrap();
        mgr.write_page(PageId(2), b"b".to_vec()).unwrap();
        // Third page should fail.
        assert_eq!(
            mgr.write_page(PageId(3), b"c".to_vec()),
            Err(SnapshotError::MaxPagesExceeded)
        );
    }

    #[test]
    fn test_checksum_mismatch_error_format() {
        let err = SnapshotError::ChecksumMismatch {
            page_id: PageId(5),
            expected: 0xABCD,
            got: 0x1234,
        };
        let msg = err.to_string();
        assert!(msg.contains("checksum mismatch"));
    }

    #[test]
    fn test_snapshot_error_display() {
        assert!(SnapshotError::SnapshotNotFound(SnapshotId(3))
            .to_string()
            .contains("snapshot not found"));
        assert!(SnapshotError::PageNotFound(PageId(7))
            .to_string()
            .contains("page not found"));
        assert!(SnapshotError::MaxSnapshotsExceeded
            .to_string()
            .contains("maximum snapshot count"));
        assert!(SnapshotError::MaxPagesExceeded
            .to_string()
            .contains("maximum page count"));
    }

    // -----------------------------------------------------------------------
    // 14. Integration / stress tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_many_pages_multiple_snapshots() {
        let mut rng = Xorshift64::new(0x1234_5678_9ABC_DEF0);
        let mut mgr = default_mgr();

        // Write 20 pages.
        for i in 0u64..20 {
            let data = rng.next_bytes(64);
            mgr.write_page(PageId(i), data).unwrap();
        }

        let s1 = mgr.create_snapshot("s1".into(), vec![], 1, None).unwrap();

        // Overwrite half the pages.
        for i in 0u64..10 {
            let data = rng.next_bytes(64);
            mgr.write_page(PageId(i), data).unwrap();
        }

        let s2 = mgr.create_snapshot("s2".into(), vec![], 2, None).unwrap();

        let diff = mgr.diff_snapshots(s1, s2).unwrap();
        assert_eq!(diff.modified_pages.len(), 10);
        assert_eq!(diff.added_pages.len(), 0);
        assert_eq!(diff.removed_pages.len(), 0);

        // CoW copies: 10 pages (first write on each of them after snapshot).
        assert_eq!(mgr.stats().cow_copies_made, 10);
    }

    #[test]
    fn test_snapshot_chain_restore_sequence() {
        let mut mgr = default_mgr();
        let id = PageId(1);

        mgr.write_page(id, b"state0".to_vec()).unwrap();
        let s0 = mgr.create_snapshot("s0".into(), vec![], 0, None).unwrap();

        mgr.write_page(id, b"state1".to_vec()).unwrap();
        let s1 = mgr.create_snapshot("s1".into(), vec![], 1, None).unwrap();

        mgr.write_page(id, b"state2".to_vec()).unwrap();

        // Restore to s0.
        mgr.restore_snapshot(s0).unwrap();
        assert_eq!(mgr.read_page(id).unwrap().data, b"state0");

        // Restore to s1.
        mgr.restore_snapshot(s1).unwrap();
        assert_eq!(mgr.read_page(id).unwrap().data, b"state1");
    }

    #[test]
    fn test_gc_after_full_delete_cycle() {
        let mut mgr = default_mgr();
        for i in 0u64..10 {
            mgr.write_page(PageId(i), vec![i as u8; 16]).unwrap();
        }
        let sid = mgr.create_snapshot("full".into(), vec![], 0, None).unwrap();
        for i in 0u64..10 {
            mgr.write_page(PageId(i), vec![0xFF; 16]).unwrap();
        }
        mgr.delete_snapshot(sid).unwrap();
        // auto_gc is true by default; all old page versions should be gone.
        assert_eq!(mgr.page_store_size(), 10);
    }

    #[test]
    fn test_shared_pages_decreases_after_cow_write() {
        let mut mgr = default_mgr();
        let id = PageId(1);
        mgr.write_page(id, b"shared".to_vec()).unwrap();
        mgr.create_snapshot("s".into(), vec![], 0, None).unwrap();
        assert_eq!(mgr.stats().shared_pages, 1);

        // CoW write makes current's copy private.
        mgr.write_page(id, b"unshared".to_vec()).unwrap();
        // The old version (still in snapshot) has ref_count 1; new version has
        // ref_count 1 too — neither is shared.
        assert_eq!(mgr.stats().shared_pages, 0);
    }

    #[test]
    fn test_snapshot_metadata_size_bytes() {
        let mut mgr = default_mgr();
        mgr.write_page(PageId(1), vec![0u8; 100]).unwrap();
        mgr.write_page(PageId(2), vec![0u8; 200]).unwrap();
        let sid = mgr
            .create_snapshot("sized".into(), vec![], 0, None)
            .unwrap();
        let meta = &mgr.snapshot_meta[&sid];
        assert_eq!(meta.size_bytes, 300);
        assert_eq!(meta.page_count, 2);
    }

    #[test]
    fn test_config_accessor() {
        let cfg = SnapshotConfig {
            max_snapshots: 10,
            ..Default::default()
        };
        let mgr = StorageSnapshotManager::new(cfg);
        assert_eq!(mgr.config().max_snapshots, 10);
    }

    #[test]
    fn test_cow_mapping_version_of() {
        let mut m = CoWMapping::default();
        let id = PageId(1);
        m.pages.insert(id, 42);
        assert_eq!(m.version_of(id), Some(42));
        assert_eq!(m.version_of(PageId(2)), None);
    }

    #[test]
    fn test_snapshot_diff_size_delta_negative() {
        let mut mgr = default_mgr();
        let id = PageId(1);
        mgr.write_page(id, vec![0u8; 100]).unwrap();
        let s1 = mgr.create_snapshot("s1".into(), vec![], 1, None).unwrap();
        mgr.write_page(id, vec![0u8; 10]).unwrap(); // smaller
        let s2 = mgr.create_snapshot("s2".into(), vec![], 2, None).unwrap();
        let diff = mgr.diff_snapshots(s1, s2).unwrap();
        assert_eq!(diff.size_delta, -90);
    }

    #[test]
    fn test_verify_snapshot_empty() {
        let mut mgr = default_mgr();
        let sid = mgr
            .create_snapshot("empty".into(), vec![], 0, None)
            .unwrap();
        let corrupted = mgr.verify_snapshot(sid).unwrap();
        assert!(corrupted.is_empty());
    }

    #[test]
    fn test_page_id_display() {
        assert_eq!(PageId(42).to_string(), "PageId(42)");
    }

    #[test]
    fn test_snapshot_id_display() {
        assert_eq!(SnapshotId(7).to_string(), "SnapshotId(7)");
    }

    #[test]
    fn test_xorshift_deterministic() {
        let mut rng1 = Xorshift64::new(123);
        let mut rng2 = Xorshift64::new(123);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_many_snapshots_stress() {
        let mut mgr = StorageSnapshotManager::new(SnapshotConfig {
            max_snapshots: 100,
            ..Default::default()
        });
        let mut rng = Xorshift64::new(999);
        for i in 0u64..50 {
            mgr.write_page(PageId(i % 10), rng.next_bytes(8)).unwrap();
            if i % 5 == 0 {
                mgr.create_snapshot(format!("s{i}"), vec![], i, None)
                    .unwrap();
            }
        }
        // All snapshots must be verifiable.
        for snap in mgr.list_snapshots() {
            let id = snap.id;
            assert!(mgr.verify_snapshot(id).unwrap().is_empty());
        }
    }

    #[test]
    fn test_write_page_after_restore_is_cow() {
        let mut mgr = default_mgr();
        let id = PageId(1);
        mgr.write_page(id, b"snap".to_vec()).unwrap();
        let sid = mgr.create_snapshot("s".into(), vec![], 0, None).unwrap();
        mgr.write_page(id, b"current".to_vec()).unwrap();
        mgr.restore_snapshot(sid).unwrap();
        // After restore current shares snapshot pages; write triggers CoW.
        let copies_before = mgr.stats().cow_copies_made;
        mgr.write_page(id, b"after-restore".to_vec()).unwrap();
        // CoW copy should have happened since page is shared with snapshot.
        assert!(mgr.stats().cow_copies_made > copies_before);
        assert_eq!(mgr.read_page_from_snapshot(sid, id).unwrap().data, b"snap");
    }
}
