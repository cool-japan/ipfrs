//! HNSW index persistence
//!
//! Serializes HNSW vector indexes to content-addressed blocks and restores
//! them on startup. Uses oxicode for binary serialization.

use ipfrs_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::Path;
use tracing::{debug, info, warn};

/// Snapshot version constant for forward-compatibility checks
const SNAPSHOT_VERSION: u32 = 1;

/// A serializable entry representing a single indexed vector
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexEntry {
    /// Internal node ID within the HNSW graph
    pub id: u32,
    /// CID string representation for the indexed content
    pub cid: String,
    /// The raw (un-normalized) feature vector
    pub vector: Vec<f32>,
    /// Maximum layer this node participates in
    pub max_layer: usize,
}

/// Serializable snapshot of the complete HNSW index state
///
/// Captures every field needed to reconstruct an identical index at load time.
/// The `layer_connections` tensor encodes the sparse adjacency lists of the
/// multi-layer graph: `layer_connections[layer][node_id]` is the sorted list of
/// neighbor IDs at that layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexSnapshot {
    /// Format version — must equal `SNAPSHOT_VERSION`
    pub version: u32,
    /// Vector dimension of every entry
    pub dimension: usize,
    /// efConstruction parameter used when building the index
    pub ef_construction: usize,
    /// Maximum number of bi-directional connections per layer (M parameter)
    pub m: usize,
    /// Indexed vectors in insertion order
    pub entries: Vec<IndexEntry>,
    /// Adjacency lists: `layer_connections[layer][node_id]` → neighbor IDs
    pub layer_connections: Vec<Vec<Vec<u32>>>,
    /// Map from CID string to arbitrary JSON metadata
    pub metadata_map: HashMap<String, String>,
    /// Unix timestamp (seconds) at snapshot creation
    pub created_at: u64,
    /// Node ID of the HNSW entry-point (top-layer entry node), if any
    pub entry_point: Option<u32>,
}

impl IndexSnapshot {
    /// Validate that the snapshot is internally consistent
    ///
    /// Returns an error string describing the first inconsistency found.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.version != SNAPSHOT_VERSION {
            return Err(format!(
                "unsupported snapshot version {} (expected {})",
                self.version, SNAPSHOT_VERSION
            ));
        }
        if self.dimension == 0 {
            return Err("dimension must be > 0".into());
        }
        for (idx, entry) in self.entries.iter().enumerate() {
            if entry.vector.len() != self.dimension {
                return Err(format!(
                    "entry {} has vector length {} but dimension is {}",
                    idx,
                    entry.vector.len(),
                    self.dimension
                ));
            }
        }
        let n = self.entries.len() as u32;
        for (layer_idx, layer) in self.layer_connections.iter().enumerate() {
            for (node_idx, neighbors) in layer.iter().enumerate() {
                for &nb in neighbors {
                    if nb >= n {
                        return Err(format!(
                            "layer {} node {} has neighbor {} which is out of range (n={})",
                            layer_idx, node_idx, nb, n
                        ));
                    }
                }
            }
        }
        if let Some(ep) = self.entry_point {
            if ep >= n {
                return Err(format!("entry_point {} is out of range (n={})", ep, n));
            }
        }
        Ok(())
    }
}

/// Manages saving and loading [`IndexSnapshot`] files
///
/// # Example
///
/// ```rust,no_run
/// use ipfrs_semantic::persistence::{IndexPersistence, IndexSnapshot};
///
/// let p = IndexPersistence::new("/path/to/my_index.snap");
/// // p.save(&snapshot)?;
/// // let snapshot = p.load()?;
/// ```
pub struct IndexPersistence {
    path: std::path::PathBuf,
}

impl IndexPersistence {
    /// Create a new `IndexPersistence` that reads/writes to `path`
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Serialize `snapshot` and write it atomically to the configured path
    ///
    /// Uses a write-to-temp-then-rename strategy so partial writes never
    /// corrupt an existing good snapshot.
    pub fn save(&self, snapshot: &IndexSnapshot) -> Result<()> {
        // Validate before serializing
        if let Err(msg) = snapshot.validate() {
            return Err(Error::InvalidInput(format!(
                "snapshot validation failed: {}",
                msg
            )));
        }

        // Serialize with oxicode serde compat layer
        let bytes = oxicode::serde::encode_to_vec(snapshot, oxicode::config::standard())
            .map_err(|e| Error::Serialization(format!("oxicode encode failed: {}", e)))?;

        // Ensure parent directory exists
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(Error::Io)?;
        }

        // Write atomically via a sibling temp file
        let tmp_path = self.path.with_extension("snap.tmp");
        {
            let mut f = std::fs::File::create(&tmp_path).map_err(Error::Io)?;
            f.write_all(&bytes).map_err(Error::Io)?;
            f.flush().map_err(Error::Io)?;
        }
        std::fs::rename(&tmp_path, &self.path).map_err(Error::Io)?;

        info!(
            path = %self.path.display(),
            entries = snapshot.entries.len(),
            bytes = bytes.len(),
            "HNSW index snapshot saved"
        );
        Ok(())
    }

    /// Deserialize and return the [`IndexSnapshot`] stored at the configured path
    pub fn load(&self) -> Result<IndexSnapshot> {
        if !self.path.exists() {
            return Err(Error::NotFound(format!(
                "snapshot file not found: {}",
                self.path.display()
            )));
        }

        let bytes = std::fs::read(&self.path).map_err(Error::Io)?;

        let (snapshot, _consumed): (IndexSnapshot, _) =
            oxicode::serde::decode_from_slice(&bytes, oxicode::config::standard())
                .map_err(|e| Error::Deserialization(format!("oxicode decode failed: {}", e)))?;

        // Validate the loaded snapshot
        snapshot
            .validate()
            .map_err(|msg| Error::InvalidData(format!("loaded snapshot is corrupt: {}", msg)))?;

        debug!(
            path = %self.path.display(),
            entries = snapshot.entries.len(),
            dimension = snapshot.dimension,
            "HNSW index snapshot loaded"
        );
        Ok(snapshot)
    }

    /// Return `true` if the snapshot file exists on disk
    pub fn exists(&self) -> bool {
        self.path.exists()
    }

    /// Delete the snapshot file.  No-ops silently if the file is absent.
    pub fn delete(&self) -> Result<()> {
        if self.path.exists() {
            std::fs::remove_file(&self.path).map_err(Error::Io)?;
            warn!(path = %self.path.display(), "HNSW snapshot deleted");
        }
        Ok(())
    }

    /// Return the configured snapshot path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Smart save: use an incremental delta when fewer than 10 % of total
    /// entries are dirty, otherwise fall back to a full snapshot.
    ///
    /// This is the preferred entry-point for `Node::stop` because it avoids
    /// re-writing the entire index on every shutdown when only a handful of
    /// embeddings have changed.
    ///
    /// After a successful write the tracker inside `index` is marked clean.
    ///
    /// # Decision rule
    /// * If the index is empty → full snapshot (no-op, but valid state).
    /// * If `dirty_count / total_count < 0.10` AND a full base snapshot
    ///   already exists on disk → incremental delta.
    /// * Otherwise → full snapshot.
    pub fn save_smart(&self, index: &crate::hnsw::VectorIndex) -> Result<()> {
        let total = index.len();
        let dirty = index.dirty_count();

        let use_incremental = total > 0
            && dirty < total / 10  // <10 % dirty
            && self.exists(); // base snapshot must exist

        if use_incremental {
            let base_version = index.tracker_version();
            let delta = index.snapshot_incremental(base_version)?;
            self.save_incremental(&delta)?;
            index.mark_tracker_clean();
            info!(
                dirty,
                total,
                base_version,
                delta_version = delta.delta_version,
                "HNSW index saved as incremental delta"
            );
        } else {
            let snap = index.snapshot()?;
            self.save(&snap)?;
            index.record_full_snapshot();
            debug!(entries = total, "HNSW index saved as full snapshot");
        }

        Ok(())
    }

    /// Path for the incremental (delta) snapshot alongside the full snapshot
    ///
    /// For a base path of `foo/index.snap` this returns `foo/index.snap.delta`.
    pub fn incremental_path(&self) -> std::path::PathBuf {
        let mut p = self.path.clone();
        let ext = p
            .extension()
            .map(|e| {
                let mut s = e.to_os_string();
                s.push(".delta");
                s
            })
            .unwrap_or_else(|| std::ffi::OsString::from("snap.delta"));
        p.set_extension(ext);
        p
    }

    /// Serialize `snapshot` and write it as an incremental (delta) snapshot
    ///
    /// The delta file sits alongside the full snapshot at `<base>.delta`.
    /// Uses the same atomic write-then-rename strategy as `save`.
    pub fn save_incremental(&self, snapshot: &IncrementalSnapshot) -> Result<()> {
        let delta_path = self.incremental_path();

        let bytes =
            oxicode::serde::encode_to_vec(snapshot, oxicode::config::standard()).map_err(|e| {
                Error::Serialization(format!("oxicode encode incremental failed: {}", e))
            })?;

        // Ensure parent directory exists
        if let Some(parent) = delta_path.parent() {
            std::fs::create_dir_all(parent).map_err(Error::Io)?;
        }

        let tmp_path = delta_path.with_extension("delta.tmp");
        {
            let mut f = std::fs::File::create(&tmp_path).map_err(Error::Io)?;
            f.write_all(&bytes).map_err(Error::Io)?;
            f.flush().map_err(Error::Io)?;
        }
        std::fs::rename(&tmp_path, &delta_path).map_err(Error::Io)?;

        info!(
            path = %delta_path.display(),
            changed = snapshot.changed_entries.len(),
            deleted = snapshot.deleted_ids.len(),
            base_version = snapshot.base_version,
            delta_version = snapshot.delta_version,
            "Incremental HNSW snapshot saved"
        );
        Ok(())
    }

    /// Deserialize and return the [`IncrementalSnapshot`] stored at the delta path
    pub fn load_incremental(&self) -> Result<IncrementalSnapshot> {
        let delta_path = self.incremental_path();

        if !delta_path.exists() {
            return Err(Error::NotFound(format!(
                "incremental snapshot file not found: {}",
                delta_path.display()
            )));
        }

        let bytes = std::fs::read(&delta_path).map_err(Error::Io)?;

        let (snapshot, _consumed): (IncrementalSnapshot, _) =
            oxicode::serde::decode_from_slice(&bytes, oxicode::config::standard()).map_err(
                |e| Error::Deserialization(format!("oxicode decode incremental failed: {}", e)),
            )?;

        debug!(
            path = %delta_path.display(),
            changed = snapshot.changed_entries.len(),
            deleted = snapshot.deleted_ids.len(),
            "Incremental HNSW snapshot loaded"
        );
        Ok(snapshot)
    }

    /// Apply an incremental delta to an existing full [`IndexSnapshot`] in place
    ///
    /// Entries whose IDs appear in `delta.deleted_ids` are removed, then all
    /// `delta.changed_entries` are upserted (replacing existing entries with
    /// the same `id` or appending new ones).
    pub fn apply_incremental(base: &mut IndexSnapshot, delta: &IncrementalSnapshot) -> Result<()> {
        // Remove deleted entries
        if !delta.deleted_ids.is_empty() {
            let deleted_set: HashSet<u32> = delta.deleted_ids.iter().copied().collect();
            base.entries.retain(|e| !deleted_set.contains(&e.id));
        }

        // Upsert changed entries
        for changed in &delta.changed_entries {
            if let Some(existing) = base.entries.iter_mut().find(|e| e.id == changed.id) {
                *existing = changed.clone();
            } else {
                base.entries.push(changed.clone());
            }
        }

        debug!(
            base_version = delta.base_version,
            delta_version = delta.delta_version,
            entries_after = base.entries.len(),
            "Applied incremental snapshot delta"
        );
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Incremental snapshot types
// ═══════════════════════════════════════════════════════════════════════════

/// Tracks which HNSW entries have been modified since the last full snapshot
///
/// The caller is responsible for calling `mark_dirty` whenever a vector is
/// inserted, updated, or logically removed from the index.  After writing a
/// full or incremental snapshot the caller should call `mark_clean` so the
/// dirty set is cleared and the version counter is advanced.
pub struct IncrementalTracker {
    /// Entry IDs that have changed since the last snapshot
    dirty_ids: HashSet<u32>,
    /// Snapshot version counter — incremented on every `mark_clean` call
    version: u64,
    /// Wall-clock time of the last full snapshot (not the last incremental one)
    last_full_snapshot: Option<std::time::SystemTime>,
}

impl IncrementalTracker {
    /// Create a fresh tracker with no dirty entries and version 0
    pub fn new() -> Self {
        Self {
            dirty_ids: HashSet::new(),
            version: 0,
            last_full_snapshot: None,
        }
    }

    /// Record that entry `id` has been inserted or modified
    pub fn mark_dirty(&mut self, id: u32) {
        self.dirty_ids.insert(id);
    }

    /// Clear the dirty set and advance the version counter
    ///
    /// Should be called immediately after a snapshot (full or incremental) has
    /// been successfully written to stable storage.
    pub fn mark_clean(&mut self) {
        self.dirty_ids.clear();
        self.version = self.version.saturating_add(1);
    }

    /// Record that a full snapshot was taken at `time`
    pub fn record_full_snapshot(&mut self, time: std::time::SystemTime) {
        self.last_full_snapshot = Some(time);
        self.mark_clean();
    }

    /// Return a reference to the current set of dirty entry IDs
    pub fn dirty_ids(&self) -> &HashSet<u32> {
        &self.dirty_ids
    }

    /// Return `true` if any entries have been modified since the last snapshot
    pub fn is_dirty(&self) -> bool {
        !self.dirty_ids.is_empty()
    }

    /// Number of entries that are currently dirty
    pub fn dirty_count(&self) -> usize {
        self.dirty_ids.len()
    }

    /// Current snapshot version (incremented after each `mark_clean`)
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Timestamp of the last full snapshot, if any
    pub fn last_full_snapshot(&self) -> Option<std::time::SystemTime> {
        self.last_full_snapshot
    }
}

impl Default for IncrementalTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// A lightweight delta snapshot: only the entries that changed since a base
/// full snapshot was taken
///
/// To reconstruct the full state, load the base [`IndexSnapshot`] first and
/// then call [`IndexPersistence::apply_incremental`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalSnapshot {
    /// Version of the base full snapshot this delta was derived from
    pub base_version: u64,
    /// Version assigned to this incremental snapshot
    pub delta_version: u64,
    /// Entries that were inserted or modified since the base snapshot
    pub changed_entries: Vec<IndexEntry>,
    /// IDs of entries that were logically deleted since the base snapshot
    pub deleted_ids: Vec<u32>,
    /// Unix timestamp (seconds) at the time this delta was created
    pub created_at: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot() -> IndexSnapshot {
        IndexSnapshot {
            version: 1,
            dimension: 4,
            ef_construction: 200,
            m: 16,
            entries: vec![
                IndexEntry {
                    id: 0,
                    cid: "test_cid_0".to_string(),
                    vector: vec![1.0, 0.0, 0.0, 0.0],
                    max_layer: 0,
                },
                IndexEntry {
                    id: 1,
                    cid: "test_cid_1".to_string(),
                    vector: vec![0.0, 1.0, 0.0, 0.0],
                    max_layer: 0,
                },
            ],
            layer_connections: vec![vec![vec![1], vec![0]]],
            metadata_map: HashMap::new(),
            created_at: 12345,
            entry_point: Some(0),
        }
    }

    fn temp_snap_path(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("ipfrs_hnsw_test_{}_{}", tag, nanos))
    }

    #[test]
    fn test_snapshot_save_load_roundtrip() {
        let dir = temp_snap_path("roundtrip");
        std::fs::create_dir_all(&dir).expect("create temp dir");

        let persistence = IndexPersistence::new(dir.join("index.snap"));
        let snapshot = make_snapshot();

        persistence.save(&snapshot).expect("save");
        assert!(persistence.exists(), "file must exist after save");

        let loaded = persistence.load().expect("load");
        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.dimension, 4);
        assert_eq!(loaded.ef_construction, 200);
        assert_eq!(loaded.m, 16);
        assert_eq!(loaded.entries.len(), 2);
        assert_eq!(loaded.entries[0].cid, "test_cid_0");
        assert_eq!(loaded.entries[0].vector, vec![1.0f32, 0.0, 0.0, 0.0]);
        assert_eq!(loaded.entries[1].cid, "test_cid_1");
        assert_eq!(loaded.entries[1].vector, vec![0.0f32, 1.0, 0.0, 0.0]);
        assert_eq!(loaded.entry_point, Some(0));
        assert_eq!(loaded.created_at, 12345);

        persistence.delete().expect("delete");
        assert!(!persistence.exists(), "file must be gone after delete");
    }

    #[test]
    fn test_persistence_not_found() {
        let persistence = IndexPersistence::new("/nonexistent/path/index.snap");
        assert!(!persistence.exists());
        assert!(persistence.load().is_err());
    }

    #[test]
    fn test_save_creates_parent_dirs() {
        let base = temp_snap_path("mkdir");
        // path does not exist yet
        let nested = base.join("a").join("b").join("c").join("index.snap");
        let persistence = IndexPersistence::new(&nested);

        let snapshot = make_snapshot();
        persistence
            .save(&snapshot)
            .expect("save should create parent dirs");
        assert!(persistence.exists());

        // cleanup
        let _ = persistence.delete();
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_overwrite_is_atomic() {
        let dir = temp_snap_path("atomic");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let persistence = IndexPersistence::new(dir.join("index.snap"));

        let snap_a = make_snapshot();
        let mut snap_b = make_snapshot();
        snap_b.created_at = 99999;
        snap_b.entries.push(IndexEntry {
            id: 2,
            cid: "test_cid_2".to_string(),
            vector: vec![0.0, 0.0, 1.0, 0.0],
            max_layer: 1,
        });

        persistence.save(&snap_a).expect("first save");
        persistence.save(&snap_b).expect("second save");

        let loaded = persistence.load().expect("load");
        assert_eq!(loaded.created_at, 99999);
        assert_eq!(loaded.entries.len(), 3);

        let _ = persistence.delete();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_snapshot_validate_version_mismatch() {
        let mut snap = make_snapshot();
        snap.version = 99;
        assert!(snap.validate().is_err());
    }

    #[test]
    fn test_snapshot_validate_dimension_mismatch() {
        let mut snap = make_snapshot();
        snap.entries[0].vector = vec![1.0, 2.0]; // only 2 dims, but dimension = 4
        assert!(snap.validate().is_err());
    }

    #[test]
    fn test_snapshot_validate_out_of_range_neighbor() {
        let mut snap = make_snapshot();
        snap.layer_connections[0][0] = vec![999]; // no node 999
        assert!(snap.validate().is_err());
    }
}

#[cfg(test)]
mod incremental_tests {
    use super::*;

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("ipfrs_incr_test_{}_{}", tag, nanos))
    }

    fn make_base_snapshot() -> IndexSnapshot {
        IndexSnapshot {
            version: 1,
            dimension: 3,
            ef_construction: 100,
            m: 8,
            entries: vec![
                IndexEntry {
                    id: 0,
                    cid: "cid0".to_string(),
                    vector: vec![1.0, 0.0, 0.0],
                    max_layer: 0,
                },
                IndexEntry {
                    id: 1,
                    cid: "cid1".to_string(),
                    vector: vec![0.0, 1.0, 0.0],
                    max_layer: 0,
                },
            ],
            layer_connections: vec![vec![vec![1], vec![0]]],
            metadata_map: HashMap::new(),
            created_at: 1000,
            entry_point: Some(0),
        }
    }

    #[test]
    fn test_incremental_tracker_dirty_tracking() {
        let mut tracker = IncrementalTracker::new();
        assert!(!tracker.is_dirty());
        assert_eq!(tracker.dirty_count(), 0);
        assert_eq!(tracker.version(), 0);

        tracker.mark_dirty(5);
        tracker.mark_dirty(10);
        tracker.mark_dirty(5); // duplicate — set semantics
        assert!(tracker.is_dirty());
        assert_eq!(tracker.dirty_count(), 2);
        assert!(tracker.dirty_ids().contains(&5));
        assert!(tracker.dirty_ids().contains(&10));

        tracker.mark_clean();
        assert!(!tracker.is_dirty());
        assert_eq!(tracker.dirty_count(), 0);
        assert_eq!(tracker.version(), 1);

        // Second clean cycle
        tracker.mark_dirty(99);
        assert_eq!(tracker.dirty_count(), 1);
        tracker.mark_clean();
        assert_eq!(tracker.version(), 2);
    }

    #[test]
    fn test_incremental_snapshot_save_load() {
        let dir = temp_dir("save_load");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let persistence = IndexPersistence::new(dir.join("index.snap"));

        let delta = IncrementalSnapshot {
            base_version: 3,
            delta_version: 4,
            changed_entries: vec![IndexEntry {
                id: 7,
                cid: "cid7".to_string(),
                vector: vec![0.5, 0.5, 0.0],
                max_layer: 1,
            }],
            deleted_ids: vec![2, 3],
            created_at: 9999,
        };

        persistence
            .save_incremental(&delta)
            .expect("save incremental");

        let loaded = persistence.load_incremental().expect("load incremental");
        assert_eq!(loaded.base_version, 3);
        assert_eq!(loaded.delta_version, 4);
        assert_eq!(loaded.changed_entries.len(), 1);
        assert_eq!(loaded.changed_entries[0].id, 7);
        assert_eq!(loaded.deleted_ids, vec![2, 3]);
        assert_eq!(loaded.created_at, 9999);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_apply_incremental_to_base() {
        let mut base = make_base_snapshot();

        let delta = IncrementalSnapshot {
            base_version: 0,
            delta_version: 1,
            changed_entries: vec![
                // Modify entry 0
                IndexEntry {
                    id: 0,
                    cid: "cid0_v2".to_string(),
                    vector: vec![0.9, 0.1, 0.0],
                    max_layer: 0,
                },
                // Append new entry 2
                IndexEntry {
                    id: 2,
                    cid: "cid2".to_string(),
                    vector: vec![0.0, 0.0, 1.0],
                    max_layer: 0,
                },
            ],
            deleted_ids: vec![1], // remove entry 1
            created_at: 2000,
        };

        IndexPersistence::apply_incremental(&mut base, &delta).expect("apply incremental");

        // Entry 1 was deleted
        assert!(base.entries.iter().all(|e| e.id != 1));
        // Entry 0 was updated
        let e0 = base.entries.iter().find(|e| e.id == 0).expect("entry 0");
        assert_eq!(e0.cid, "cid0_v2");
        assert_eq!(e0.vector, vec![0.9f32, 0.1, 0.0]);
        // Entry 2 was added
        assert!(base.entries.iter().any(|e| e.id == 2));
        // Overall count: started with 2, deleted 1, added 1 new → still 2
        assert_eq!(base.entries.len(), 2);
    }

    #[test]
    fn test_incremental_path_naming() {
        let p = IndexPersistence::new("/some/dir/index.snap");
        let delta = p.incremental_path();
        // Should end with .snap.delta
        assert!(
            delta.to_string_lossy().ends_with(".snap.delta"),
            "unexpected delta path: {}",
            delta.display()
        );
        // Parent directory should be the same
        assert_eq!(delta.parent(), std::path::Path::new("/some/dir").into());
    }

    #[test]
    fn test_incremental_tracker_record_full_snapshot() {
        let mut tracker = IncrementalTracker::new();
        tracker.mark_dirty(1);
        tracker.mark_dirty(2);

        let now = std::time::SystemTime::now();
        tracker.record_full_snapshot(now);

        // Should be clean and version bumped
        assert!(!tracker.is_dirty());
        assert_eq!(tracker.version(), 1);
        assert!(tracker.last_full_snapshot().is_some());
    }

    #[test]
    fn test_apply_incremental_empty_delta() {
        let mut base = make_base_snapshot();
        let original_len = base.entries.len();

        let delta = IncrementalSnapshot {
            base_version: 0,
            delta_version: 1,
            changed_entries: vec![],
            deleted_ids: vec![],
            created_at: 0,
        };

        IndexPersistence::apply_incremental(&mut base, &delta).expect("apply empty delta");
        assert_eq!(base.entries.len(), original_len);
    }

    /// An incremental snapshot built from a [`VectorIndex`] with 100 entries
    /// and 5 dirty entries must contain exactly 5 changed entries.
    #[test]
    fn test_incremental_snapshot_only_dirty() {
        use crate::hnsw::{DistanceMetric, VectorIndex};

        const DIM: usize = 8;
        const TOTAL: usize = 100;
        const DIRTY_COUNT: usize = 5;

        // Build an index with 100 embeddings
        let mut index =
            VectorIndex::new(DIM, DistanceMetric::L2, 8, 50).expect("create VectorIndex");

        let mut cids = Vec::with_capacity(TOTAL);
        for i in 0..TOTAL {
            // Create a unique valid CID using Block::new
            // We'll construct CIDs from dummy bytes via ipfrs_core
            let data = bytes::Bytes::from(format!("embed-data-{}", i));
            let block = ipfrs_core::Block::new(data).expect("create block");
            let cid = *block.cid();
            cids.push(cid);
            // Build a simple unit vector with one hot at position i % DIM
            let mut v = vec![0.0f32; DIM];
            v[i % DIM] = 1.0 + (i as f32) * 0.001; // small variation to avoid duplicates
            index.add_embedding(&cid, &v).expect("add_embedding");
        }

        // Simulate that the full snapshot was already taken: reset dirty set.
        // (record_full_snapshot clears dirty_ids and advances version)
        index.record_full_snapshot();

        // Now "dirty" exactly 5 entries by inserting 5 new embeddings after the
        // snapshot.  We cannot modify existing entries (VectorIndex does not
        // support updates), so we use 5 new CIDs.
        let dirty_start = TOTAL;
        for i in dirty_start..(dirty_start + DIRTY_COUNT) {
            let data = bytes::Bytes::from(format!("new-embed-{}", i));
            let block = ipfrs_core::Block::new(data).expect("create block");
            let cid = *block.cid();
            let mut v = vec![0.0f32; DIM];
            v[i % DIM] = 2.0 + (i as f32) * 0.001;
            index.add_embedding(&cid, &v).expect("add_embedding dirty");
        }

        assert_eq!(
            index.dirty_count(),
            DIRTY_COUNT,
            "expected exactly {} dirty entries after insertions",
            DIRTY_COUNT
        );

        // Build incremental snapshot from the dirty tracker
        let base_version = index.tracker_version();
        let delta = index
            .snapshot_incremental(base_version)
            .expect("snapshot_incremental");

        assert_eq!(
            delta.changed_entries.len(),
            DIRTY_COUNT,
            "incremental delta must contain exactly {} changed entries",
            DIRTY_COUNT
        );
        assert!(
            delta.deleted_ids.is_empty(),
            "no deletions should be recorded"
        );
    }
}
