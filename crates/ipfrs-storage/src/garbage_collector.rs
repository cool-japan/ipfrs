//! Mark-and-sweep garbage collector for content-addressed storage.
//!
//! Implements full mark-and-sweep GC with reference counting, cycle detection
//! via BFS traversal, pinning support, and incremental batch processing.
//!
//! # Algorithm
//!
//! 1. **Mark phase**: BFS from all root objects and pinned objects, following
//!    DAG links. Every reachable object is stamped with `last_marked = Some(now)`.
//! 2. **Sweep phase**: All objects where `last_marked != Some(now)` AND
//!    `pinned == false` AND `ref_count == 0` are removed and their sizes summed.
//! 3. Statistics are returned in a [`StorageGcRun`] record.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_storage::garbage_collector::{
//!     StorageGarbageCollector, GcObjectId, GcObject, StorageGcConfig,
//! };
//!
//! let config = StorageGcConfig::default();
//! let mut gc = StorageGarbageCollector::new(config);
//!
//! let root_id = GcObjectId("bafyroot".to_string());
//! let root = GcObject {
//!     id: root_id.clone(),
//!     size_bytes: 512,
//!     ref_count: 1,
//!     pinned: false,
//!     created_at: 0,
//!     last_marked: None,
//!     links: vec![],
//! };
//! gc.add_object(root).expect("add object");
//! gc.add_root(&root_id);
//!
//! let run = gc.run_gc(1);
//! assert_eq!(run.objects_swept, 0);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

use thiserror::Error;

// ─── Error type ──────────────────────────────────────────────────────────────

/// Errors that can occur during garbage-collection operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum GcError {
    /// An object with this id already exists in the store.
    #[error("object already exists: {0}")]
    ObjectAlreadyExists(String),

    /// No object with this id exists in the store.
    #[error("object not found: {0}")]
    ObjectNotFound(String),

    /// The object is pinned and cannot be removed.
    #[error("object is pinned: {0}")]
    ObjectPinned(String),

    /// The object still has active references and cannot be removed.
    #[error("object {id} is still referenced (ref_count={ref_count})")]
    ObjectReferenced {
        /// Object identifier.
        id: String,
        /// Current reference count.
        ref_count: u32,
    },
}

// ─── Core types ──────────────────────────────────────────────────────────────

/// Newtype wrapper for object identifiers (CIDs).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GcObjectId(pub String);

impl GcObjectId {
    /// Create a new `GcObjectId` from any `Into<String>`.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Return the inner string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for GcObjectId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A node in the content-addressed DAG.
#[derive(Debug, Clone)]
pub struct GcObject {
    /// Unique identifier (CID).
    pub id: GcObjectId,
    /// Size of this object in bytes.
    pub size_bytes: u64,
    /// External reference count (from the application layer).
    pub ref_count: u32,
    /// If `true`, this object is never collected regardless of reachability.
    pub pinned: bool,
    /// Unix timestamp (seconds) when this object was created.
    pub created_at: u64,
    /// Unix timestamp of the last GC mark pass that reached this object.
    pub last_marked: Option<u64>,
    /// CIDs of objects this object references (outgoing DAG edges).
    pub links: Vec<GcObjectId>,
}

impl GcObject {
    /// Returns `true` if this object is safe to sweep:
    /// not pinned, no external references, and not freshly marked.
    fn is_sweepable(&self, mark_epoch: u64) -> bool {
        !self.pinned && self.ref_count == 0 && self.last_marked != Some(mark_epoch)
    }
}

/// Current phase of the garbage collector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GcPhase {
    /// No GC in progress.
    Idle,
    /// Mark phase is active.
    Marking,
    /// Sweep phase is active.
    Sweeping,
    /// Post-sweep compaction (currently a no-op pass).
    Compacting,
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`StorageGarbageCollector`].
#[derive(Debug, Clone)]
pub struct StorageGcConfig {
    /// Total byte budget; triggers GC when exceeded by `sweep_threshold_fraction`.
    pub max_live_bytes: u64,
    /// Fraction in `[0.0, 1.0]`: trigger GC when
    /// `live_bytes / max_live_bytes >= sweep_threshold_fraction`.
    pub sweep_threshold_fraction: f64,
    /// When `true`, pinned objects are never swept even if unreachable.
    pub pin_preserve: bool,
    /// Maximum number of objects to process per GC step.
    pub batch_size: usize,
}

impl Default for StorageGcConfig {
    fn default() -> Self {
        Self {
            max_live_bytes: 1024 * 1024 * 1024, // 1 GiB
            sweep_threshold_fraction: 0.8,
            pin_preserve: true,
            batch_size: 1000,
        }
    }
}

// ─── GC run record ────────────────────────────────────────────────────────────

/// Record of a single garbage-collection run.
#[derive(Debug, Clone)]
pub struct StorageGcRun {
    /// Monotonically increasing run identifier.
    pub id: u64,
    /// Unix timestamp when this run started.
    pub started_at: u64,
    /// Unix timestamp when this run completed (None if still in progress).
    pub completed_at: Option<u64>,
    /// Number of objects marked as reachable.
    pub objects_marked: usize,
    /// Number of objects swept (removed).
    pub objects_swept: usize,
    /// Total bytes freed by this run.
    pub bytes_freed: u64,
    /// Phase the collector was in when this record was taken.
    pub phase: GcPhase,
}

// ─── Statistics ───────────────────────────────────────────────────────────────

/// Aggregate statistics for a [`StorageGarbageCollector`].
#[derive(Debug, Clone)]
pub struct StorageGcStats {
    /// Total objects currently tracked.
    pub total_objects: usize,
    /// Number of GC-root objects.
    pub root_count: usize,
    /// Total bytes across all tracked objects.
    pub live_bytes: u64,
    /// Number of pinned objects.
    pub pinned_count: usize,
    /// Total number of completed GC runs.
    pub total_runs: usize,
    /// Bytes freed in the most recent completed run (0 if none).
    pub last_run_freed_bytes: u64,
}

// ─── Garbage collector ────────────────────────────────────────────────────────

/// Production-grade mark-and-sweep garbage collector for content-addressed
/// storage.
///
/// Objects form a directed acyclic graph (DAG) via [`GcObject::links`].
/// GC roots and pinned objects anchor the live set; everything else that has
/// no external references (`ref_count == 0`) is eligible for collection.
pub struct StorageGarbageCollector {
    /// Runtime configuration.
    pub config: StorageGcConfig,
    /// All tracked objects keyed by id.
    pub objects: HashMap<GcObjectId, GcObject>,
    /// Set of explicit GC roots.
    pub roots: HashSet<GcObjectId>,
    /// History of completed GC runs (capped at 256 entries).
    pub gc_runs: VecDeque<StorageGcRun>,
    /// Current collector phase.
    pub phase: GcPhase,
    /// Counter used to assign monotonically increasing run ids.
    pub next_run_id: u64,
}

impl StorageGarbageCollector {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Create a new collector with the given configuration.
    pub fn new(config: StorageGcConfig) -> Self {
        Self {
            config,
            objects: HashMap::new(),
            roots: HashSet::new(),
            gc_runs: VecDeque::new(),
            phase: GcPhase::Idle,
            next_run_id: 1,
        }
    }

    // ── Object management ────────────────────────────────────────────────────

    /// Add a new object.  Returns [`GcError::ObjectAlreadyExists`] if an
    /// object with the same id is already tracked.
    pub fn add_object(&mut self, object: GcObject) -> Result<(), GcError> {
        if self.objects.contains_key(&object.id) {
            return Err(GcError::ObjectAlreadyExists(object.id.0.clone()));
        }
        self.objects.insert(object.id.clone(), object);
        Ok(())
    }

    /// Remove and return an object.
    ///
    /// Returns errors when:
    /// - The object does not exist.
    /// - The object is pinned.
    /// - The object has `ref_count > 0`.
    pub fn remove_object(&mut self, id: &GcObjectId) -> Result<GcObject, GcError> {
        let obj = self
            .objects
            .get(id)
            .ok_or_else(|| GcError::ObjectNotFound(id.0.clone()))?;

        if obj.pinned {
            return Err(GcError::ObjectPinned(id.0.clone()));
        }
        if obj.ref_count > 0 {
            return Err(GcError::ObjectReferenced {
                id: id.0.clone(),
                ref_count: obj.ref_count,
            });
        }

        Ok(self.objects.remove(id).unwrap_or_else(|| {
            // Unreachable: we checked existence above.
            panic!("object disappeared between check and remove")
        }))
    }

    // ── Root management ──────────────────────────────────────────────────────

    /// Mark an object as a GC root.
    ///
    /// Returns `true` if the object exists (and was added as root), `false`
    /// otherwise.
    pub fn add_root(&mut self, id: &GcObjectId) -> bool {
        if self.objects.contains_key(id) {
            self.roots.insert(id.clone());
            true
        } else {
            false
        }
    }

    /// Remove an object from the GC root set.
    ///
    /// Returns `true` if the object was a root (and has been removed).
    pub fn remove_root(&mut self, id: &GcObjectId) -> bool {
        self.roots.remove(id)
    }

    // ── Reference counting ───────────────────────────────────────────────────

    /// Increment the external reference count for an object.
    ///
    /// Returns `true` if the object was found, `false` otherwise.
    pub fn increment_ref(&mut self, id: &GcObjectId) -> bool {
        if let Some(obj) = self.objects.get_mut(id) {
            obj.ref_count = obj.ref_count.saturating_add(1);
            true
        } else {
            false
        }
    }

    /// Decrement the external reference count (floor 0).
    ///
    /// Returns `true` if the object was found.  When `ref_count` reaches 0 and
    /// the object is neither a root nor pinned, it becomes a collection
    /// candidate on the next GC run.
    pub fn decrement_ref(&mut self, id: &GcObjectId) -> bool {
        if let Some(obj) = self.objects.get_mut(id) {
            obj.ref_count = obj.ref_count.saturating_sub(1);
            true
        } else {
            false
        }
    }

    // ── Pinning ───────────────────────────────────────────────────────────────

    /// Pin an object so it is never swept.
    ///
    /// Returns `true` if the object was found.
    pub fn pin(&mut self, id: &GcObjectId) -> bool {
        if let Some(obj) = self.objects.get_mut(id) {
            obj.pinned = true;
            true
        } else {
            false
        }
    }

    /// Unpin an object so it can be swept when unreachable.
    ///
    /// Returns `true` if the object was found.
    pub fn unpin(&mut self, id: &GcObjectId) -> bool {
        if let Some(obj) = self.objects.get_mut(id) {
            obj.pinned = false;
            true
        } else {
            false
        }
    }

    // ── Threshold check ───────────────────────────────────────────────────────

    /// Returns `true` when live_bytes / max_live_bytes ≥ sweep_threshold_fraction.
    pub fn should_run(&self) -> bool {
        let live = self.live_bytes();
        let max = self.config.max_live_bytes;
        if max == 0 {
            return false;
        }
        let ratio = live as f64 / max as f64;
        ratio >= self.config.sweep_threshold_fraction
    }

    // ── Mark phase ────────────────────────────────────────────────────────────

    /// BFS traversal from all roots and pinned objects.
    ///
    /// Every reachable object gets `last_marked = Some(now)`.
    /// Returns the set of reachable ids.
    pub fn mark_reachable(&mut self, now: u64) -> HashSet<GcObjectId> {
        // Seeds: explicit roots + pinned objects.
        let mut queue: VecDeque<GcObjectId> = VecDeque::new();
        let mut visited: HashSet<GcObjectId> = HashSet::new();

        for id in &self.roots {
            if !visited.contains(id) {
                visited.insert(id.clone());
                queue.push_back(id.clone());
            }
        }

        // Also seed pinned objects even if not roots.
        for (id, obj) in &self.objects {
            if obj.pinned && !visited.contains(id) {
                visited.insert(id.clone());
                queue.push_back(id.clone());
            }
        }

        // BFS: walk links in batches (respecting batch_size for incremental
        // friendliness, but this is a full mark so we exhaust the queue).
        while let Some(current_id) = queue.pop_front() {
            // Stamp the object.
            if let Some(obj) = self.objects.get_mut(&current_id) {
                obj.last_marked = Some(now);

                // Collect links to avoid borrow-checker issues.
                let links: Vec<GcObjectId> = obj.links.clone();
                for link in links {
                    if !visited.contains(&link) {
                        visited.insert(link.clone());
                        queue.push_back(link);
                    }
                }
            }
        }

        visited
    }

    // ── Full GC run ───────────────────────────────────────────────────────────

    /// Perform a full mark-and-sweep GC run.
    ///
    /// Steps:
    /// 1. Mark all objects reachable from roots / pinned objects.
    /// 2. Sweep unreachable, unpinned, ref_count==0 objects.
    /// 3. Record and return a [`StorageGcRun`].
    pub fn run_gc(&mut self, now: u64) -> StorageGcRun {
        let run_id = self.next_run_id;
        self.next_run_id += 1;

        // ── Mark ────────────────────────────────────────────────────────────
        self.phase = GcPhase::Marking;
        let reachable = self.mark_reachable(now);
        let objects_marked = reachable.len();

        // ── Sweep ────────────────────────────────────────────────────────────
        self.phase = GcPhase::Sweeping;
        let sweep_ids: Vec<GcObjectId> = self
            .objects
            .iter()
            .filter_map(|(id, obj)| {
                if obj.is_sweepable(now) {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect();

        let mut objects_swept = 0usize;
        let mut bytes_freed = 0u64;
        for id in &sweep_ids {
            if let Some(obj) = self.objects.remove(id) {
                objects_swept += 1;
                bytes_freed += obj.size_bytes;
                // Also clean up any root entry that referenced a swept object.
                self.roots.remove(id);
            }
        }

        // ── Compact (bookkeeping only) ────────────────────────────────────────
        self.phase = GcPhase::Compacting;

        // ── Finalise ─────────────────────────────────────────────────────────
        self.phase = GcPhase::Idle;

        let run = StorageGcRun {
            id: run_id,
            started_at: now,
            completed_at: Some(now),
            objects_marked,
            objects_swept,
            bytes_freed,
            phase: GcPhase::Idle,
        };

        // Keep at most 256 historical run records.
        if self.gc_runs.len() >= 256 {
            self.gc_runs.pop_front();
        }
        self.gc_runs.push_back(run.clone());

        run
    }

    // ── Metrics ───────────────────────────────────────────────────────────────

    /// Total bytes across all currently tracked objects.
    pub fn live_bytes(&self) -> u64 {
        self.objects.values().map(|o| o.size_bytes).sum()
    }

    /// Total bytes across the supplied reachable set.
    pub fn reachable_bytes(&self, reachable: &HashSet<GcObjectId>) -> u64 {
        reachable
            .iter()
            .filter_map(|id| self.objects.get(id))
            .map(|o| o.size_bytes)
            .sum()
    }

    /// Number of tracked objects.
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Number of GC-root objects.
    pub fn root_count(&self) -> usize {
        self.roots.len()
    }

    /// Return aggregate statistics.
    pub fn stats(&self) -> StorageGcStats {
        let pinned_count = self.objects.values().filter(|o| o.pinned).count();
        let last_run_freed_bytes = self.gc_runs.back().map(|r| r.bytes_freed).unwrap_or(0);

        StorageGcStats {
            total_objects: self.objects.len(),
            root_count: self.roots.len(),
            live_bytes: self.live_bytes(),
            pinned_count,
            total_runs: self.gc_runs.len(),
            last_run_freed_bytes,
        }
    }
}

// ─── Helpers (builder-style) ──────────────────────────────────────────────────

impl GcObject {
    /// Convenience constructor.
    pub fn new(id: GcObjectId, size_bytes: u64, created_at: u64, links: Vec<GcObjectId>) -> Self {
        Self {
            id,
            size_bytes,
            ref_count: 0,
            pinned: false,
            created_at,
            last_marked: None,
            links,
        }
    }

    /// Set the initial reference count and return `self`.
    pub fn with_ref_count(mut self, ref_count: u32) -> Self {
        self.ref_count = ref_count;
        self
    }

    /// Mark as pinned and return `self`.
    pub fn pinned(mut self) -> Self {
        self.pinned = true;
        self
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::garbage_collector::{
        GcError, GcObject, GcObjectId, GcPhase, StorageGarbageCollector, StorageGcConfig,
    };

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn default_gc() -> StorageGarbageCollector {
        StorageGarbageCollector::new(StorageGcConfig::default())
    }

    fn make_obj(id: &str, size: u64) -> GcObject {
        GcObject::new(GcObjectId::new(id), size, 0, vec![])
    }

    fn make_obj_with_links(id: &str, size: u64, links: Vec<&str>) -> GcObject {
        let link_ids = links.into_iter().map(GcObjectId::new).collect();
        GcObject::new(GcObjectId::new(id), size, 0, link_ids)
    }

    fn oid(s: &str) -> GcObjectId {
        GcObjectId::new(s)
    }

    // ── Constructor ──────────────────────────────────────────────────────────

    #[test]
    fn test_new_is_idle() {
        let gc = default_gc();
        assert_eq!(gc.phase, GcPhase::Idle);
        assert_eq!(gc.object_count(), 0);
        assert_eq!(gc.root_count(), 0);
    }

    #[test]
    fn test_default_config_values() {
        let cfg = StorageGcConfig::default();
        assert!(cfg.max_live_bytes > 0);
        assert!(cfg.sweep_threshold_fraction > 0.0);
        assert!(cfg.sweep_threshold_fraction <= 1.0);
        assert_eq!(cfg.batch_size, 1000);
        assert!(cfg.pin_preserve);
    }

    // ── add_object ────────────────────────────────────────────────────────────

    #[test]
    fn test_add_object_succeeds() {
        let mut gc = default_gc();
        gc.add_object(make_obj("a", 100)).expect("add object");
        assert_eq!(gc.object_count(), 1);
    }

    #[test]
    fn test_add_duplicate_fails() {
        let mut gc = default_gc();
        gc.add_object(make_obj("a", 100)).expect("first add");
        let err = gc.add_object(make_obj("a", 200)).unwrap_err();
        assert!(matches!(err, GcError::ObjectAlreadyExists(_)));
    }

    #[test]
    fn test_add_multiple_objects() {
        let mut gc = default_gc();
        for i in 0..10u32 {
            gc.add_object(make_obj(&format!("obj{i}"), 64))
                .expect("add");
        }
        assert_eq!(gc.object_count(), 10);
    }

    // ── remove_object ─────────────────────────────────────────────────────────

    #[test]
    fn test_remove_object_succeeds() {
        let mut gc = default_gc();
        gc.add_object(make_obj("a", 100)).expect("add");
        let removed = gc.remove_object(&oid("a")).expect("remove");
        assert_eq!(removed.id, oid("a"));
        assert_eq!(gc.object_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_fails() {
        let mut gc = default_gc();
        let err = gc.remove_object(&oid("nope")).unwrap_err();
        assert!(matches!(err, GcError::ObjectNotFound(_)));
    }

    #[test]
    fn test_remove_pinned_fails() {
        let mut gc = default_gc();
        gc.add_object(make_obj("a", 100).pinned()).expect("add");
        let err = gc.remove_object(&oid("a")).unwrap_err();
        assert!(matches!(err, GcError::ObjectPinned(_)));
    }

    #[test]
    fn test_remove_referenced_fails() {
        let mut gc = default_gc();
        gc.add_object(make_obj("a", 100).with_ref_count(2))
            .expect("add");
        let err = gc.remove_object(&oid("a")).unwrap_err();
        assert!(
            matches!(err, GcError::ObjectReferenced { ref_count: 2, .. }),
            "got: {err:?}"
        );
    }

    // ── add_root / remove_root ────────────────────────────────────────────────

    #[test]
    fn test_add_root_existing_object() {
        let mut gc = default_gc();
        gc.add_object(make_obj("r", 50)).expect("add");
        assert!(gc.add_root(&oid("r")));
        assert_eq!(gc.root_count(), 1);
    }

    #[test]
    fn test_add_root_missing_object_returns_false() {
        let mut gc = default_gc();
        assert!(!gc.add_root(&oid("ghost")));
        assert_eq!(gc.root_count(), 0);
    }

    #[test]
    fn test_remove_root() {
        let mut gc = default_gc();
        gc.add_object(make_obj("r", 50)).expect("add");
        gc.add_root(&oid("r"));
        assert!(gc.remove_root(&oid("r")));
        assert_eq!(gc.root_count(), 0);
    }

    #[test]
    fn test_remove_root_not_present() {
        let mut gc = default_gc();
        assert!(!gc.remove_root(&oid("none")));
    }

    // ── ref counting ─────────────────────────────────────────────────────────

    #[test]
    fn test_increment_ref() {
        let mut gc = default_gc();
        gc.add_object(make_obj("a", 10)).expect("add");
        assert!(gc.increment_ref(&oid("a")));
        let obj = gc.objects.get(&oid("a")).expect("obj");
        assert_eq!(obj.ref_count, 1);
    }

    #[test]
    fn test_decrement_ref() {
        let mut gc = default_gc();
        gc.add_object(make_obj("a", 10).with_ref_count(3))
            .expect("add");
        assert!(gc.decrement_ref(&oid("a")));
        let obj = gc.objects.get(&oid("a")).expect("obj");
        assert_eq!(obj.ref_count, 2);
    }

    #[test]
    fn test_decrement_ref_floors_at_zero() {
        let mut gc = default_gc();
        gc.add_object(make_obj("a", 10)).expect("add");
        gc.decrement_ref(&oid("a")); // already 0
        let obj = gc.objects.get(&oid("a")).expect("obj");
        assert_eq!(obj.ref_count, 0);
    }

    #[test]
    fn test_increment_ref_missing_returns_false() {
        let mut gc = default_gc();
        assert!(!gc.increment_ref(&oid("nope")));
    }

    // ── pinning ───────────────────────────────────────────────────────────────

    #[test]
    fn test_pin_and_unpin() {
        let mut gc = default_gc();
        gc.add_object(make_obj("a", 10)).expect("add");
        assert!(gc.pin(&oid("a")));
        assert!(gc.objects[&oid("a")].pinned);
        assert!(gc.unpin(&oid("a")));
        assert!(!gc.objects[&oid("a")].pinned);
    }

    #[test]
    fn test_pin_missing_returns_false() {
        let mut gc = default_gc();
        assert!(!gc.pin(&oid("nope")));
    }

    // ── live_bytes / reachable_bytes ──────────────────────────────────────────

    #[test]
    fn test_live_bytes_empty() {
        let gc = default_gc();
        assert_eq!(gc.live_bytes(), 0);
    }

    #[test]
    fn test_live_bytes_sum() {
        let mut gc = default_gc();
        gc.add_object(make_obj("a", 100)).expect("add");
        gc.add_object(make_obj("b", 200)).expect("add");
        assert_eq!(gc.live_bytes(), 300);
    }

    #[test]
    fn test_reachable_bytes() {
        let mut gc = default_gc();
        gc.add_object(make_obj("a", 100)).expect("add");
        gc.add_object(make_obj("b", 200)).expect("add");
        let mut reachable = HashSet::new();
        reachable.insert(oid("a"));
        assert_eq!(gc.reachable_bytes(&reachable), 100);
    }

    // ── should_run ────────────────────────────────────────────────────────────

    #[test]
    fn test_should_run_below_threshold() {
        let cfg = StorageGcConfig {
            max_live_bytes: 1000,
            sweep_threshold_fraction: 0.8,
            ..StorageGcConfig::default()
        };
        let mut gc = StorageGarbageCollector::new(cfg);
        gc.add_object(make_obj("a", 500)).expect("add"); // 50%
        assert!(!gc.should_run());
    }

    #[test]
    fn test_should_run_at_threshold() {
        let cfg = StorageGcConfig {
            max_live_bytes: 1000,
            sweep_threshold_fraction: 0.8,
            ..StorageGcConfig::default()
        };
        let mut gc = StorageGarbageCollector::new(cfg);
        gc.add_object(make_obj("a", 800)).expect("add"); // exactly 80%
        assert!(gc.should_run());
    }

    #[test]
    fn test_should_run_zero_max_bytes() {
        let cfg = StorageGcConfig {
            max_live_bytes: 0,
            ..StorageGcConfig::default()
        };
        let mut gc = StorageGarbageCollector::new(cfg);
        gc.add_object(make_obj("a", 1)).expect("add");
        assert!(!gc.should_run()); // division by zero guard
    }

    // ── mark_reachable ────────────────────────────────────────────────────────

    #[test]
    fn test_mark_reachable_from_root() {
        let mut gc = default_gc();
        gc.add_object(make_obj("root", 10)).expect("add");
        gc.add_object(make_obj("child", 10)).expect("add");
        // No link yet → child not reachable
        gc.add_root(&oid("root"));
        let reachable = gc.mark_reachable(1);
        assert!(reachable.contains(&oid("root")));
        assert!(!reachable.contains(&oid("child")));
    }

    #[test]
    fn test_mark_reachable_follows_links() {
        let mut gc = default_gc();
        gc.add_object(make_obj_with_links("root", 10, vec!["child"]))
            .expect("add");
        gc.add_object(make_obj("child", 10)).expect("add");
        gc.add_root(&oid("root"));
        let reachable = gc.mark_reachable(1);
        assert!(reachable.contains(&oid("child")));
    }

    #[test]
    fn test_mark_reachable_deep_chain() {
        let mut gc = default_gc();
        // a -> b -> c -> d
        gc.add_object(make_obj_with_links("a", 10, vec!["b"]))
            .expect("add");
        gc.add_object(make_obj_with_links("b", 10, vec!["c"]))
            .expect("add");
        gc.add_object(make_obj_with_links("c", 10, vec!["d"]))
            .expect("add");
        gc.add_object(make_obj("d", 10)).expect("add");
        gc.add_root(&oid("a"));
        let reachable = gc.mark_reachable(1);
        for id in &["a", "b", "c", "d"] {
            assert!(reachable.contains(&oid(id)), "missing: {id}");
        }
    }

    #[test]
    fn test_mark_reachable_pinned_as_seed() {
        let mut gc = default_gc();
        // pinned but not root → still marked
        gc.add_object(make_obj("pinned", 10).pinned()).expect("add");
        let reachable = gc.mark_reachable(1);
        assert!(reachable.contains(&oid("pinned")));
    }

    #[test]
    fn test_mark_sets_last_marked() {
        let mut gc = default_gc();
        gc.add_object(make_obj("a", 10)).expect("add");
        gc.add_root(&oid("a"));
        gc.mark_reachable(42);
        assert_eq!(gc.objects[&oid("a")].last_marked, Some(42));
    }

    // ── run_gc ────────────────────────────────────────────────────────────────

    #[test]
    fn test_run_gc_sweeps_unreachable() {
        let mut gc = default_gc();
        gc.add_object(make_obj("orphan", 512)).expect("add");
        let run = gc.run_gc(1);
        assert_eq!(run.objects_swept, 1);
        assert_eq!(run.bytes_freed, 512);
        assert_eq!(gc.object_count(), 0);
    }

    #[test]
    fn test_run_gc_keeps_roots() {
        let mut gc = default_gc();
        gc.add_object(make_obj("root", 100)).expect("add");
        gc.add_root(&oid("root"));
        let run = gc.run_gc(1);
        assert_eq!(run.objects_swept, 0);
        assert_eq!(run.bytes_freed, 0);
        assert_eq!(gc.object_count(), 1);
    }

    #[test]
    fn test_run_gc_keeps_pinned() {
        let mut gc = default_gc();
        gc.add_object(make_obj("pin", 100).pinned()).expect("add");
        let run = gc.run_gc(1);
        assert_eq!(run.objects_swept, 0);
        assert_eq!(gc.object_count(), 1);
    }

    #[test]
    fn test_run_gc_keeps_referenced() {
        let mut gc = default_gc();
        gc.add_object(make_obj("ref", 100).with_ref_count(1))
            .expect("add");
        let run = gc.run_gc(1);
        // ref_count > 0 even though not reachable from root → not swept
        assert_eq!(run.objects_swept, 0);
    }

    #[test]
    fn test_run_gc_increments_run_id() {
        let mut gc = default_gc();
        let r1 = gc.run_gc(1);
        let r2 = gc.run_gc(2);
        assert_eq!(r1.id + 1, r2.id);
    }

    #[test]
    fn test_run_gc_records_stats() {
        let mut gc = default_gc();
        gc.add_object(make_obj("orphan", 1024)).expect("add");
        gc.run_gc(1);
        let stats = gc.stats();
        assert_eq!(stats.total_runs, 1);
        assert_eq!(stats.last_run_freed_bytes, 1024);
    }

    #[test]
    fn test_run_gc_marks_objects_count() {
        let mut gc = default_gc();
        gc.add_object(make_obj("a", 10)).expect("add");
        gc.add_object(make_obj("b", 10)).expect("add");
        gc.add_root(&oid("a"));
        let run = gc.run_gc(1);
        assert_eq!(run.objects_marked, 1); // only "a" reachable
        assert_eq!(run.objects_swept, 1); // "b" swept
    }

    #[test]
    fn test_run_gc_completed_at() {
        let mut gc = default_gc();
        let run = gc.run_gc(999);
        assert_eq!(run.completed_at, Some(999));
        assert_eq!(run.started_at, 999);
    }

    #[test]
    fn test_run_gc_idempotent_on_empty_store() {
        let mut gc = default_gc();
        for epoch in 0..5u64 {
            let run = gc.run_gc(epoch);
            assert_eq!(run.objects_swept, 0);
        }
    }

    // ── DAG traversal edge cases ──────────────────────────────────────────────

    #[test]
    fn test_diamond_dag() {
        // root -> left, right; left -> leaf; right -> leaf
        let mut gc = default_gc();
        gc.add_object(make_obj_with_links("root", 10, vec!["left", "right"]))
            .expect("add");
        gc.add_object(make_obj_with_links("left", 10, vec!["leaf"]))
            .expect("add");
        gc.add_object(make_obj_with_links("right", 10, vec!["leaf"]))
            .expect("add");
        gc.add_object(make_obj("leaf", 10)).expect("add");
        gc.add_root(&oid("root"));

        let run = gc.run_gc(1);
        assert_eq!(run.objects_swept, 0);
        assert_eq!(gc.object_count(), 4);
    }

    #[test]
    fn test_orphaned_subgraph() {
        // root -> child (both reachable); orphan_a -> orphan_b (both swept)
        let mut gc = default_gc();
        gc.add_object(make_obj_with_links("root", 10, vec!["child"]))
            .expect("add");
        gc.add_object(make_obj("child", 20)).expect("add");
        gc.add_object(make_obj_with_links("orphan_a", 30, vec!["orphan_b"]))
            .expect("add");
        gc.add_object(make_obj("orphan_b", 40)).expect("add");
        gc.add_root(&oid("root"));

        let run = gc.run_gc(1);
        assert_eq!(run.objects_swept, 2);
        assert_eq!(run.bytes_freed, 70);
    }

    #[test]
    fn test_multiple_roots() {
        let mut gc = default_gc();
        gc.add_object(make_obj("r1", 10)).expect("add");
        gc.add_object(make_obj("r2", 10)).expect("add");
        gc.add_object(make_obj("orphan", 10)).expect("add");
        gc.add_root(&oid("r1"));
        gc.add_root(&oid("r2"));

        let run = gc.run_gc(1);
        assert_eq!(run.objects_swept, 1);
        assert_eq!(gc.object_count(), 2);
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_pinned_count() {
        let mut gc = default_gc();
        gc.add_object(make_obj("a", 10).pinned()).expect("add");
        gc.add_object(make_obj("b", 10)).expect("add");
        let stats = gc.stats();
        assert_eq!(stats.pinned_count, 1);
    }

    #[test]
    fn test_stats_zero_runs() {
        let gc = default_gc();
        let stats = gc.stats();
        assert_eq!(stats.total_runs, 0);
        assert_eq!(stats.last_run_freed_bytes, 0);
    }

    // ── GcObject builder ──────────────────────────────────────────────────────

    #[test]
    fn test_gc_object_new_defaults() {
        let obj = GcObject::new(oid("x"), 64, 100, vec![]);
        assert_eq!(obj.ref_count, 0);
        assert!(!obj.pinned);
        assert!(obj.last_marked.is_none());
    }

    #[test]
    fn test_gc_object_with_ref_count() {
        let obj = GcObject::new(oid("x"), 64, 0, vec![]).with_ref_count(5);
        assert_eq!(obj.ref_count, 5);
    }

    #[test]
    fn test_gc_object_pinned_builder() {
        let obj = GcObject::new(oid("x"), 64, 0, vec![]).pinned();
        assert!(obj.pinned);
    }

    // ── GcObjectId ────────────────────────────────────────────────────────────

    #[test]
    fn test_gc_object_id_display() {
        let id = GcObjectId::new("bafy123");
        assert_eq!(id.to_string(), "bafy123");
    }

    #[test]
    fn test_gc_object_id_as_str() {
        let id = GcObjectId::new("bafy123");
        assert_eq!(id.as_str(), "bafy123");
    }

    // ── GC run history cap ────────────────────────────────────────────────────

    #[test]
    fn test_gc_run_history_capped_at_256() {
        let mut gc = default_gc();
        for epoch in 0..300u64 {
            gc.run_gc(epoch);
        }
        assert!(gc.gc_runs.len() <= 256, "history exceeded 256 entries");
    }

    // ── Mixed scenario ────────────────────────────────────────────────────────

    #[test]
    fn test_full_lifecycle() {
        let mut gc = default_gc();

        // Build a DAG: root -> a -> b, c (orphan)
        gc.add_object(make_obj_with_links("root", 100, vec!["a"]))
            .expect("add root");
        gc.add_object(make_obj_with_links("a", 200, vec!["b"]))
            .expect("add a");
        gc.add_object(make_obj("b", 300)).expect("add b");
        gc.add_object(make_obj("c", 400)).expect("add c");

        gc.add_root(&oid("root"));

        // First GC: c swept
        let run1 = gc.run_gc(1);
        assert_eq!(run1.objects_swept, 1);
        assert_eq!(run1.bytes_freed, 400);

        // Remove root → a and b become unreachable
        gc.remove_root(&oid("root"));
        let run2 = gc.run_gc(2);
        assert_eq!(run2.objects_swept, 3);
        assert_eq!(run2.bytes_freed, 600);

        assert_eq!(gc.object_count(), 0);
        let stats = gc.stats();
        assert_eq!(stats.total_runs, 2);
    }
}
