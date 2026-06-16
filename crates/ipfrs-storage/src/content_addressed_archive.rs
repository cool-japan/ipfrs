//! Content-Addressed Archive — an append-only, in-memory store for immutable data blobs.
//!
//! Each blob is keyed by its FNV-1a-64 content identifier (CID).  The flat byte array
//! (`data`) grows monotonically; removed entries are tombstoned rather than reclaimed,
//! preserving the append-only invariant while still allowing logical deletion.

use std::collections::{BTreeMap, HashMap};

/// Compute a 64-bit FNV-1a content identifier, returned as a 16-char hex string.
pub fn compute_cid(data: &[u8]) -> String {
    let mut h: u64 = 14_695_981_039_346_656_037_u64;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211_u64);
    }
    format!("{:016x}", h)
}

// ── Error ─────────────────────────────────────────────────────────────────────

/// Errors that can occur during archive operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ArchiveError {
    /// A blob with this CID already exists and overwrites are disabled.
    #[error("duplicate CID: {0}")]
    DuplicateCid(String),

    /// No live entry with this CID exists.
    #[error("CID not found: {0}")]
    CidNotFound(String),

    /// The archive has reached its configured capacity limit.
    #[error("archive is full")]
    ArchiveFull,

    /// On-disk (or in-memory) data does not match the recorded CID.
    #[error("corrupted entry: {0}")]
    CorruptedEntry(String),

    /// The entry exists but has been tombstoned (logically removed).
    #[error("tombstoned entry: {0}")]
    TombstonedEntry(String),
}

// ── Core data types ───────────────────────────────────────────────────────────

/// Metadata record for a single archived blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveEntry {
    /// FNV-1a-64 content identifier (16 hex chars).
    pub cid: String,
    /// Byte offset into the flat `data` array.
    pub offset: u64,
    /// Byte length of the blob.
    pub length: u64,
    /// Whether the blob is stored compressed (always `false` in the current implementation).
    pub compressed: bool,
    /// User-supplied and internal key/value metadata.
    /// The reserved key `"_tombstone"` is set to `"true"` for removed entries.
    pub metadata: HashMap<String, String>,
    /// Unix timestamp (seconds or millis — caller-supplied) at insertion time.
    pub inserted_at: u64,
}

impl ArchiveEntry {
    /// Returns `true` if this entry has been logically removed.
    pub fn is_tombstoned(&self) -> bool {
        self.metadata
            .get("_tombstone")
            .map(|v| v == "true")
            .unwrap_or(false)
    }
}

/// A blob together with its archive metadata, used when exporting entries.
#[derive(Debug, Clone)]
pub struct ArchiveBlock {
    /// Raw blob bytes.
    pub data: Vec<u8>,
    /// Metadata record.
    pub entry: ArchiveEntry,
}

/// The ordered index of all entries (live + tombstoned).
#[derive(Debug, Clone, Default)]
pub struct ArchiveIndex {
    /// All entries, keyed and sorted by CID for deterministic iteration.
    pub entries: BTreeMap<String, ArchiveEntry>,
    /// Offset at which the next blob will be written into the flat array.
    pub next_offset: u64,
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Capacity and behaviour configuration for [`ContentAddressedArchive`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveConfig {
    /// Maximum total live bytes before the archive is considered full.
    pub max_size_bytes: u64,
    /// Maximum number of live entries before the archive is considered full.
    pub max_entries: usize,
    /// When `true`, inserting a CID that already exists replaces the old entry;
    /// when `false` (the default), it returns [`ArchiveError::DuplicateCid`].
    pub allow_overwrites: bool,
}

impl Default for ArchiveConfig {
    fn default() -> Self {
        Self {
            max_size_bytes: 1_073_741_824, // 1 GiB
            max_entries: 1_000_000,
            allow_overwrites: false,
        }
    }
}

// ── Statistics ────────────────────────────────────────────────────────────────

/// Aggregate statistics about the archive.
#[derive(Debug, Clone)]
pub struct ArchiveStats {
    /// Number of live (non-tombstoned) entries.
    pub live_entries: usize,
    /// Number of tombstoned entries.
    pub tombstoned_entries: usize,
    /// Total live bytes stored.
    pub total_bytes: u64,
    /// Mean size of live entries in bytes; `0.0` if no live entries.
    pub avg_entry_size_bytes: f64,
    /// `total_bytes / max_size_bytes` — fraction of capacity used.
    pub utilization_fraction: f64,
}

// ── Archive ───────────────────────────────────────────────────────────────────

/// An append-only, content-addressed archive for storing immutable data blobs.
///
/// Data is kept in a flat in-memory byte array (`data`).  Each entry records
/// the offset and length of its blob so that slices can be returned without
/// copying.  Removed entries are tombstoned in their metadata; their bytes
/// remain in the flat array but are excluded from all live-entry computations.
#[derive(Debug, Clone)]
pub struct ContentAddressedArchive {
    /// Archive capacity and behaviour settings.
    pub config: ArchiveConfig,
    /// CID → entry index.
    pub index: ArchiveIndex,
    /// Flat byte array holding all blob data (including tombstoned blobs).
    pub data: Vec<u8>,
}

impl ContentAddressedArchive {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Create a new, empty archive with the given configuration.
    pub fn new(config: ArchiveConfig) -> Self {
        Self {
            config,
            index: ArchiveIndex::default(),
            data: Vec::new(),
        }
    }

    // ── Capacity checks ───────────────────────────────────────────────────────

    /// Returns `true` when the archive has reached either of its capacity limits.
    pub fn is_full(&self) -> bool {
        self.total_data_bytes() >= self.config.max_size_bytes
            || self.entry_count() >= self.config.max_entries
    }

    // ── Core mutation ─────────────────────────────────────────────────────────

    /// Insert a blob into the archive.
    ///
    /// Computes the CID from `data`, checks for duplicates, enforces capacity
    /// limits, appends the bytes to the flat array, and records the entry.
    ///
    /// Returns the computed CID on success.
    pub fn put(
        &mut self,
        data: Vec<u8>,
        metadata: HashMap<String, String>,
        now: u64,
    ) -> Result<String, ArchiveError> {
        let cid = compute_cid(&data);

        // Duplicate check
        if let Some(existing) = self.index.entries.get(&cid) {
            if existing.is_tombstoned() {
                // A previously tombstoned entry: treat as a fresh insert
                // (the old bytes remain in the flat array; we append new ones).
            } else if !self.config.allow_overwrites {
                return Err(ArchiveError::DuplicateCid(cid));
            }
            // allow_overwrites == true → fall through and replace the index entry.
        }

        // Capacity guard (only for genuinely new live entries)
        if self.is_full() {
            // Check whether this is just an overwrite of an existing live entry —
            // in that case we don't increase the live count/size, so we can proceed.
            let is_live_overwrite = self
                .index
                .entries
                .get(&cid)
                .map(|e| !e.is_tombstoned())
                .unwrap_or(false);
            if !is_live_overwrite {
                return Err(ArchiveError::ArchiveFull);
            }
        }

        let offset = self.index.next_offset;
        let length = data.len() as u64;

        self.data.extend_from_slice(&data);
        self.index.next_offset += length;

        let entry = ArchiveEntry {
            cid: cid.clone(),
            offset,
            length,
            compressed: false,
            metadata,
            inserted_at: now,
        };
        self.index.entries.insert(cid.clone(), entry);

        Ok(cid)
    }

    /// Retrieve the raw bytes of the blob identified by `cid`.
    pub fn get(&self, cid: &str) -> Result<&[u8], ArchiveError> {
        match self.index.entries.get(cid) {
            None => Err(ArchiveError::CidNotFound(cid.to_owned())),
            Some(entry) if entry.is_tombstoned() => {
                Err(ArchiveError::TombstonedEntry(cid.to_owned()))
            }
            Some(entry) => {
                let start = entry.offset as usize;
                let end = start + entry.length as usize;
                Ok(&self.data[start..end])
            }
        }
    }

    /// Retrieve the metadata entry for the blob identified by `cid`.
    pub fn get_entry(&self, cid: &str) -> Result<&ArchiveEntry, ArchiveError> {
        match self.index.entries.get(cid) {
            None => Err(ArchiveError::CidNotFound(cid.to_owned())),
            Some(entry) if entry.is_tombstoned() => {
                Err(ArchiveError::TombstonedEntry(cid.to_owned()))
            }
            Some(entry) => Ok(entry),
        }
    }

    /// Returns `true` if a **live** entry with `cid` exists.
    pub fn contains(&self, cid: &str) -> bool {
        self.index
            .entries
            .get(cid)
            .map(|e| !e.is_tombstoned())
            .unwrap_or(false)
    }

    /// Logically remove the entry for `cid`, returning the original blob bytes.
    ///
    /// The entry is kept in the index with `"_tombstone" = "true"` in its
    /// metadata; the raw bytes remain in the flat `data` array.
    pub fn remove(&mut self, cid: &str) -> Result<Vec<u8>, ArchiveError> {
        match self.index.entries.get_mut(cid) {
            None => Err(ArchiveError::CidNotFound(cid.to_owned())),
            Some(entry) if entry.is_tombstoned() => {
                Err(ArchiveError::TombstonedEntry(cid.to_owned()))
            }
            Some(entry) => {
                let start = entry.offset as usize;
                let end = start + entry.length as usize;
                let blob = self.data[start..end].to_vec();
                entry
                    .metadata
                    .insert("_tombstone".to_owned(), "true".to_owned());
                Ok(blob)
            }
        }
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Return the sorted list of all live CID strings (alphabetical order).
    pub fn list_cids(&self) -> Vec<&str> {
        self.index
            .entries
            .iter()
            .filter(|(_, e)| !e.is_tombstoned())
            .map(|(k, _)| k.as_str())
            .collect()
    }

    /// Return live entries sorted by CID.
    pub fn list_entries(&self) -> Vec<&ArchiveEntry> {
        self.index
            .entries
            .values()
            .filter(|e| !e.is_tombstoned())
            .collect()
    }

    /// Total bytes occupied by live entries.
    pub fn total_data_bytes(&self) -> u64 {
        self.index
            .entries
            .values()
            .filter(|e| !e.is_tombstoned())
            .map(|e| e.length)
            .sum()
    }

    /// Number of live entries.
    pub fn entry_count(&self) -> usize {
        self.index
            .entries
            .values()
            .filter(|e| !e.is_tombstoned())
            .count()
    }

    // ── Integrity ─────────────────────────────────────────────────────────────

    /// Verify every live entry by recomputing its CID from stored bytes.
    ///
    /// Returns the CIDs of entries where the recomputed CID differs from the
    /// recorded CID (indicating data corruption).
    pub fn verify_integrity(&self) -> Vec<String> {
        let mut corrupt = Vec::new();
        for (cid, entry) in &self.index.entries {
            if entry.is_tombstoned() {
                continue;
            }
            let start = entry.offset as usize;
            let end = start + entry.length as usize;
            let recomputed = compute_cid(&self.data[start..end]);
            if &recomputed != cid {
                corrupt.push(cid.clone());
            }
        }
        corrupt
    }

    // ── Export / Merge ────────────────────────────────────────────────────────

    /// Export all live entries as [`ArchiveBlock`] values.
    pub fn export_entries(&self) -> Vec<ArchiveBlock> {
        self.index
            .entries
            .values()
            .filter(|e| !e.is_tombstoned())
            .map(|entry| {
                let start = entry.offset as usize;
                let end = start + entry.length as usize;
                ArchiveBlock {
                    data: self.data[start..end].to_vec(),
                    entry: entry.clone(),
                }
            })
            .collect()
    }

    /// Import all live entries from `other` that are not already present in
    /// `self`.  Returns the number of entries added.
    ///
    /// Entries that already exist (even as tombstones) in `self` are skipped.
    pub fn merge(
        &mut self,
        other: &ContentAddressedArchive,
        now: u64,
    ) -> Result<usize, ArchiveError> {
        let mut added = 0usize;

        // Collect blobs first to avoid borrow conflicts.
        let blocks: Vec<ArchiveBlock> = other.export_entries();

        for block in blocks {
            // Skip CIDs already present (live or tombstoned) in self.
            if self.index.entries.contains_key(&block.entry.cid) {
                continue;
            }
            // Respect capacity limits.
            if self.is_full() {
                return Err(ArchiveError::ArchiveFull);
            }
            // Strip the source timestamp and use the caller-supplied `now`.
            let mut meta = block.entry.metadata.clone();
            meta.remove("_tombstone");
            self.put(block.data, meta, now)?;
            added += 1;
        }

        Ok(added)
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /// Compute aggregate statistics for the archive.
    pub fn stats(&self) -> ArchiveStats {
        let live_entries = self.entry_count();
        let tombstoned_entries = self
            .index
            .entries
            .values()
            .filter(|e| e.is_tombstoned())
            .count();
        let total_bytes = self.total_data_bytes();
        let avg_entry_size_bytes = if live_entries == 0 {
            0.0
        } else {
            total_bytes as f64 / live_entries as f64
        };
        let utilization_fraction = if self.config.max_size_bytes == 0 {
            0.0
        } else {
            total_bytes as f64 / self.config.max_size_bytes as f64
        };
        ArchiveStats {
            live_entries,
            tombstoned_entries,
            total_bytes,
            avg_entry_size_bytes,
            utilization_fraction,
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::content_addressed_archive::{
        compute_cid, ArchiveConfig, ArchiveError, ContentAddressedArchive,
    };

    fn empty_archive() -> ContentAddressedArchive {
        ContentAddressedArchive::new(ArchiveConfig::default())
    }

    fn meta(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    // ── compute_cid ───────────────────────────────────────────────────────────

    #[test]
    fn test_compute_cid_empty() {
        // FNV-1a of empty slice = FNV offset basis, formatted as hex.
        let cid = compute_cid(&[]);
        assert_eq!(cid.len(), 16);
        assert_eq!(cid, format!("{:016x}", 14_695_981_039_346_656_037_u64));
    }

    #[test]
    fn test_compute_cid_single_byte() {
        let cid = compute_cid(&[0x41]);
        assert_eq!(cid.len(), 16);
        // Different data must produce a different CID.
        assert_ne!(cid, compute_cid(&[0x42]));
    }

    #[test]
    fn test_compute_cid_deterministic() {
        let data = b"hello world";
        assert_eq!(compute_cid(data), compute_cid(data));
    }

    #[test]
    fn test_compute_cid_different_data() {
        assert_ne!(compute_cid(b"foo"), compute_cid(b"bar"));
    }

    #[test]
    fn test_compute_cid_length() {
        for data in [b"".as_slice(), b"a", b"hello", b"hello world!"] {
            assert_eq!(compute_cid(data).len(), 16);
        }
    }

    // ── Basic put / get ───────────────────────────────────────────────────────

    #[test]
    fn test_put_returns_cid() {
        let mut archive = empty_archive();
        let data = b"test blob".to_vec();
        let expected_cid = compute_cid(&data);
        let cid = archive.put(data, HashMap::new(), 1).unwrap();
        assert_eq!(cid, expected_cid);
    }

    #[test]
    fn test_get_returns_correct_data() {
        let mut archive = empty_archive();
        let data = b"hello".to_vec();
        let cid = archive.put(data.clone(), HashMap::new(), 0).unwrap();
        assert_eq!(archive.get(&cid).unwrap(), data.as_slice());
    }

    #[test]
    fn test_get_unknown_cid() {
        let archive = empty_archive();
        assert_eq!(
            archive.get("0000000000000000"),
            Err(ArchiveError::CidNotFound("0000000000000000".to_owned()))
        );
    }

    #[test]
    fn test_multiple_blobs_independent() {
        let mut archive = empty_archive();
        let cid1 = archive.put(b"alpha".to_vec(), HashMap::new(), 0).unwrap();
        let cid2 = archive.put(b"beta".to_vec(), HashMap::new(), 1).unwrap();
        assert_eq!(archive.get(&cid1).unwrap(), b"alpha");
        assert_eq!(archive.get(&cid2).unwrap(), b"beta");
    }

    #[test]
    fn test_put_preserves_metadata() {
        let mut archive = empty_archive();
        let m = meta(&[("key", "value"), ("type", "test")]);
        let cid = archive.put(b"data".to_vec(), m.clone(), 42).unwrap();
        let entry = archive.get_entry(&cid).unwrap();
        assert_eq!(entry.metadata.get("key"), Some(&"value".to_owned()));
        assert_eq!(entry.metadata.get("type"), Some(&"test".to_owned()));
        assert_eq!(entry.inserted_at, 42);
    }

    // ── Duplicate handling ────────────────────────────────────────────────────

    #[test]
    fn test_duplicate_cid_rejected_by_default() {
        let mut archive = empty_archive();
        let data = b"same data".to_vec();
        archive.put(data.clone(), HashMap::new(), 0).unwrap();
        let result = archive.put(data, HashMap::new(), 1);
        assert!(matches!(result, Err(ArchiveError::DuplicateCid(_))));
    }

    #[test]
    fn test_duplicate_allowed_with_overwrites() {
        let mut archive = ContentAddressedArchive::new(ArchiveConfig {
            allow_overwrites: true,
            ..ArchiveConfig::default()
        });
        let data = b"same data".to_vec();
        let cid1 = archive.put(data.clone(), HashMap::new(), 0).unwrap();
        let cid2 = archive.put(data, meta(&[("v", "2")]), 1).unwrap();
        assert_eq!(cid1, cid2);
        // Latest metadata should win.
        assert_eq!(
            archive.get_entry(&cid1).unwrap().metadata.get("v"),
            Some(&"2".to_owned())
        );
    }

    // ── contains ─────────────────────────────────────────────────────────────

    #[test]
    fn test_contains_live_entry() {
        let mut archive = empty_archive();
        let cid = archive.put(b"exist".to_vec(), HashMap::new(), 0).unwrap();
        assert!(archive.contains(&cid));
    }

    #[test]
    fn test_contains_absent_entry() {
        let archive = empty_archive();
        assert!(!archive.contains("deadbeefdeadbeef"));
    }

    // ── remove / tombstone ────────────────────────────────────────────────────

    #[test]
    fn test_remove_returns_data() {
        let mut archive = empty_archive();
        let data = b"to be removed".to_vec();
        let cid = archive.put(data.clone(), HashMap::new(), 0).unwrap();
        let removed = archive.remove(&cid).unwrap();
        assert_eq!(removed, data);
    }

    #[test]
    fn test_remove_tombstones_entry() {
        let mut archive = empty_archive();
        let cid = archive.put(b"bye".to_vec(), HashMap::new(), 0).unwrap();
        archive.remove(&cid).unwrap();
        assert!(!archive.contains(&cid));
        assert_eq!(
            archive.get(&cid),
            Err(ArchiveError::TombstonedEntry(cid.clone()))
        );
    }

    #[test]
    fn test_remove_unknown_cid() {
        let mut archive = empty_archive();
        assert_eq!(
            archive.remove("0000000000000000"),
            Err(ArchiveError::CidNotFound("0000000000000000".to_owned()))
        );
    }

    #[test]
    fn test_remove_already_tombstoned() {
        let mut archive = empty_archive();
        let cid = archive.put(b"once".to_vec(), HashMap::new(), 0).unwrap();
        archive.remove(&cid).unwrap();
        assert_eq!(
            archive.remove(&cid),
            Err(ArchiveError::TombstonedEntry(cid))
        );
    }

    // ── list_cids / list_entries ──────────────────────────────────────────────

    #[test]
    fn test_list_cids_sorted() {
        let mut archive = empty_archive();
        archive.put(b"aaa".to_vec(), HashMap::new(), 0).unwrap();
        archive.put(b"bbb".to_vec(), HashMap::new(), 1).unwrap();
        archive.put(b"ccc".to_vec(), HashMap::new(), 2).unwrap();
        let cids = archive.list_cids();
        let mut sorted = cids.clone();
        sorted.sort_unstable();
        assert_eq!(cids, sorted);
    }

    #[test]
    fn test_list_cids_excludes_tombstones() {
        let mut archive = empty_archive();
        let cid1 = archive.put(b"keep".to_vec(), HashMap::new(), 0).unwrap();
        let cid2 = archive.put(b"remove".to_vec(), HashMap::new(), 1).unwrap();
        archive.remove(&cid2).unwrap();
        let cids = archive.list_cids();
        assert!(cids.contains(&cid1.as_str()));
        assert!(!cids.contains(&cid2.as_str()));
    }

    #[test]
    fn test_list_entries_excludes_tombstones() {
        let mut archive = empty_archive();
        archive.put(b"alive".to_vec(), HashMap::new(), 0).unwrap();
        let cid = archive.put(b"dead".to_vec(), HashMap::new(), 1).unwrap();
        archive.remove(&cid).unwrap();
        assert_eq!(archive.list_entries().len(), 1);
    }

    // ── Counters ──────────────────────────────────────────────────────────────

    #[test]
    fn test_entry_count() {
        let mut archive = empty_archive();
        assert_eq!(archive.entry_count(), 0);
        archive.put(b"one".to_vec(), HashMap::new(), 0).unwrap();
        assert_eq!(archive.entry_count(), 1);
        let cid = archive.put(b"two".to_vec(), HashMap::new(), 1).unwrap();
        assert_eq!(archive.entry_count(), 2);
        archive.remove(&cid).unwrap();
        assert_eq!(archive.entry_count(), 1);
    }

    #[test]
    fn test_total_data_bytes() {
        let mut archive = empty_archive();
        archive.put(b"12345".to_vec(), HashMap::new(), 0).unwrap(); // 5
        archive.put(b"abcde".to_vec(), HashMap::new(), 1).unwrap(); // 5
        assert_eq!(archive.total_data_bytes(), 10);
    }

    #[test]
    fn test_total_data_bytes_excludes_tombstones() {
        let mut archive = empty_archive();
        archive.put(b"12345".to_vec(), HashMap::new(), 0).unwrap();
        let cid = archive.put(b"abcde".to_vec(), HashMap::new(), 1).unwrap();
        archive.remove(&cid).unwrap();
        assert_eq!(archive.total_data_bytes(), 5);
    }

    // ── is_full ───────────────────────────────────────────────────────────────

    #[test]
    fn test_is_full_by_entry_count() {
        let mut archive = ContentAddressedArchive::new(ArchiveConfig {
            max_entries: 2,
            max_size_bytes: 1_073_741_824,
            allow_overwrites: false,
        });
        archive.put(b"one".to_vec(), HashMap::new(), 0).unwrap();
        archive.put(b"two".to_vec(), HashMap::new(), 1).unwrap();
        assert!(archive.is_full());
        let result = archive.put(b"three".to_vec(), HashMap::new(), 2);
        assert_eq!(result, Err(ArchiveError::ArchiveFull));
    }

    #[test]
    fn test_is_full_by_size() {
        let mut archive = ContentAddressedArchive::new(ArchiveConfig {
            max_size_bytes: 10,
            max_entries: 1_000_000,
            allow_overwrites: false,
        });
        archive
            .put(b"12345678901".to_vec(), HashMap::new(), 0)
            .unwrap(); // 11 bytes
        assert!(archive.is_full());
        let result = archive.put(b"extra".to_vec(), HashMap::new(), 1);
        assert_eq!(result, Err(ArchiveError::ArchiveFull));
    }

    // ── verify_integrity ─────────────────────────────────────────────────────

    #[test]
    fn test_verify_integrity_clean() {
        let mut archive = empty_archive();
        archive.put(b"good".to_vec(), HashMap::new(), 0).unwrap();
        archive.put(b"data".to_vec(), HashMap::new(), 1).unwrap();
        assert!(archive.verify_integrity().is_empty());
    }

    #[test]
    fn test_verify_integrity_detects_corruption() {
        let mut archive = empty_archive();
        let cid = archive
            .put(b"original".to_vec(), HashMap::new(), 0)
            .unwrap();

        // Corrupt the flat data array in-place.
        let offset = archive.index.entries[&cid].offset as usize;
        archive.data[offset] ^= 0xFF;

        let corrupt = archive.verify_integrity();
        assert_eq!(corrupt.len(), 1);
        assert_eq!(corrupt[0], cid);
    }

    #[test]
    fn test_verify_integrity_skips_tombstones() {
        let mut archive = empty_archive();
        let cid = archive.put(b"deleted".to_vec(), HashMap::new(), 0).unwrap();
        // Corrupt the bytes first.
        let offset = archive.index.entries[&cid].offset as usize;
        archive.data[offset] ^= 0xFF;
        // Now tombstone — corruption of tombstoned entries is ignored.
        archive
            .index
            .entries
            .get_mut(&cid)
            .unwrap()
            .metadata
            .insert("_tombstone".to_owned(), "true".to_owned());
        assert!(archive.verify_integrity().is_empty());
    }

    // ── export_entries ────────────────────────────────────────────────────────

    #[test]
    fn test_export_entries_count() {
        let mut archive = empty_archive();
        archive.put(b"a".to_vec(), HashMap::new(), 0).unwrap();
        archive.put(b"b".to_vec(), HashMap::new(), 1).unwrap();
        let cid = archive.put(b"c".to_vec(), HashMap::new(), 2).unwrap();
        archive.remove(&cid).unwrap();
        let blocks = archive.export_entries();
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn test_export_entries_data_correct() {
        let mut archive = empty_archive();
        let data = b"export me".to_vec();
        let cid = archive.put(data.clone(), HashMap::new(), 0).unwrap();
        let blocks = archive.export_entries();
        let block = blocks.iter().find(|b| b.entry.cid == cid).unwrap();
        assert_eq!(block.data, data);
    }

    // ── merge ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_merge_adds_missing_entries() {
        let mut src = empty_archive();
        src.put(b"src1".to_vec(), HashMap::new(), 0).unwrap();
        src.put(b"src2".to_vec(), HashMap::new(), 1).unwrap();

        let mut dst = empty_archive();
        let added = dst.merge(&src, 100).unwrap();
        assert_eq!(added, 2);
        assert_eq!(dst.entry_count(), 2);
    }

    #[test]
    fn test_merge_skips_existing_entries() {
        let mut src = empty_archive();
        let cid = src.put(b"shared".to_vec(), HashMap::new(), 0).unwrap();

        let mut dst = empty_archive();
        dst.put(b"shared".to_vec(), HashMap::new(), 1).unwrap();

        let added = dst.merge(&src, 100).unwrap();
        assert_eq!(added, 0);
        // Only one entry: the original.
        assert_eq!(dst.entry_count(), 1);
        assert_eq!(dst.get(&cid).unwrap(), b"shared");
    }

    #[test]
    fn test_merge_skips_tombstoned_in_src() {
        let mut src = empty_archive();
        let cid = src.put(b"bye".to_vec(), HashMap::new(), 0).unwrap();
        src.remove(&cid).unwrap();

        let mut dst = empty_archive();
        let added = dst.merge(&src, 100).unwrap();
        assert_eq!(added, 0);
        assert_eq!(dst.entry_count(), 0);
    }

    #[test]
    fn test_merge_respects_capacity() {
        let mut src = empty_archive();
        src.put(b"item1".to_vec(), HashMap::new(), 0).unwrap();
        src.put(b"item2".to_vec(), HashMap::new(), 1).unwrap();

        let mut dst = ContentAddressedArchive::new(ArchiveConfig {
            max_entries: 1,
            ..ArchiveConfig::default()
        });
        dst.put(b"pre-existing".to_vec(), HashMap::new(), 0)
            .unwrap();

        let result = dst.merge(&src, 100);
        assert_eq!(result, Err(ArchiveError::ArchiveFull));
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty() {
        let archive = empty_archive();
        let s = archive.stats();
        assert_eq!(s.live_entries, 0);
        assert_eq!(s.tombstoned_entries, 0);
        assert_eq!(s.total_bytes, 0);
        assert_eq!(s.avg_entry_size_bytes, 0.0);
        assert_eq!(s.utilization_fraction, 0.0);
    }

    #[test]
    fn test_stats_live_and_tombstoned() {
        let mut archive = empty_archive();
        archive.put(b"hello".to_vec(), HashMap::new(), 0).unwrap(); // 5 bytes
        let cid = archive.put(b"world".to_vec(), HashMap::new(), 1).unwrap(); // 5 bytes
        archive.remove(&cid).unwrap();

        let s = archive.stats();
        assert_eq!(s.live_entries, 1);
        assert_eq!(s.tombstoned_entries, 1);
        assert_eq!(s.total_bytes, 5);
        assert_eq!(s.avg_entry_size_bytes, 5.0);
    }

    #[test]
    fn test_stats_utilization() {
        let mut archive = ContentAddressedArchive::new(ArchiveConfig {
            max_size_bytes: 100,
            ..ArchiveConfig::default()
        });
        archive
            .put(b"1234567890".to_vec(), HashMap::new(), 0)
            .unwrap(); // 10 bytes
        let s = archive.stats();
        assert!((s.utilization_fraction - 0.1_f64).abs() < f64::EPSILON);
    }

    // ── get_entry ─────────────────────────────────────────────────────────────

    #[test]
    fn test_get_entry_fields() {
        let mut archive = empty_archive();
        let m = meta(&[("owner", "alice")]);
        let cid = archive.put(b"payload".to_vec(), m, 999).unwrap();
        let entry = archive.get_entry(&cid).unwrap();
        assert_eq!(entry.cid, cid);
        assert_eq!(entry.length, 7);
        assert!(!entry.compressed);
        assert_eq!(entry.inserted_at, 999);
        assert_eq!(entry.metadata.get("owner"), Some(&"alice".to_owned()));
    }

    #[test]
    fn test_get_entry_not_found() {
        let archive = empty_archive();
        assert!(matches!(
            archive.get_entry("0000000000000000"),
            Err(ArchiveError::CidNotFound(_))
        ));
    }

    #[test]
    fn test_get_entry_tombstoned() {
        let mut archive = empty_archive();
        let cid = archive.put(b"gone".to_vec(), HashMap::new(), 0).unwrap();
        archive.remove(&cid).unwrap();
        assert!(matches!(
            archive.get_entry(&cid),
            Err(ArchiveError::TombstonedEntry(_))
        ));
    }

    // ── Offset correctness after multiple puts ────────────────────────────────

    #[test]
    fn test_offsets_are_contiguous() {
        let mut archive = empty_archive();
        let blobs: &[&[u8]] = &[b"one", b"two", b"three"];
        let mut cids = Vec::new();
        for blob in blobs {
            cids.push(archive.put(blob.to_vec(), HashMap::new(), 0).unwrap());
        }
        let mut expected_offset = 0u64;
        for (blob, cid) in blobs.iter().zip(&cids) {
            let entry = archive.index.entries.get(cid).unwrap();
            assert_eq!(entry.offset, expected_offset);
            assert_eq!(entry.length, blob.len() as u64);
            expected_offset += blob.len() as u64;
        }
    }

    // ── Empty data blob ───────────────────────────────────────────────────────

    #[test]
    fn test_put_empty_blob() {
        let mut archive = empty_archive();
        let cid = archive.put(vec![], HashMap::new(), 0).unwrap();
        assert_eq!(archive.get(&cid).unwrap(), b"");
        assert_eq!(archive.entry_count(), 1);
        assert_eq!(archive.total_data_bytes(), 0);
    }

    // ── Large blob round-trip ─────────────────────────────────────────────────

    #[test]
    fn test_large_blob_round_trip() {
        let mut archive = empty_archive();
        let data: Vec<u8> = (0..65_536).map(|i| (i % 256) as u8).collect();
        let cid = archive.put(data.clone(), HashMap::new(), 0).unwrap();
        assert_eq!(archive.get(&cid).unwrap(), data.as_slice());
    }

    // ── Tombstone re-insert ───────────────────────────────────────────────────

    #[test]
    fn test_reinsert_after_tombstone() {
        let mut archive = empty_archive();
        let data = b"reborn".to_vec();
        let cid = archive.put(data.clone(), HashMap::new(), 0).unwrap();
        archive.remove(&cid).unwrap();
        // Re-inserting the same data should succeed (tombstoned entry treated as new).
        let cid2 = archive.put(data.clone(), meta(&[("v", "2")]), 1).unwrap();
        assert_eq!(cid, cid2);
        assert!(archive.contains(&cid));
        assert_eq!(archive.get(&cid).unwrap(), data.as_slice());
    }
}
