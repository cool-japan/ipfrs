//! `TensorSliceManager` — named tensor slice management with copy-on-write semantics,
//! bounds checking, and overlap detection.
//!
//! # Overview
//!
//! A [`TensorSliceManager`] owns a [`TensorShape`] that describes the overall tensor
//! dimensions and a registry of named [`TensorSlice`]s.  Each slice is a contiguous
//! sub-region described by a [`SliceSpec`] (per-dimension `[start, end)` ranges) and
//! carries its own flattened `f32` data buffer together with a version counter and a
//! dirty flag that supports copy-on-write workflows.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// TensorShape
// ---------------------------------------------------------------------------

/// Describes the shape of a multi-dimensional tensor in C (row-major) order.
///
/// # Example
///
/// ```
/// use ipfrs_tensorlogic::slice_manager::TensorShape;
///
/// let shape = TensorShape { dims: vec![4, 8, 16] };
/// assert_eq!(shape.total_elements(), 512);
/// assert_eq!(shape.strides(), vec![128, 16, 1]);
/// assert_eq!(shape.flat_index(&[1, 2, 3]), Some(1 * 128 + 2 * 16 + 3));
/// assert_eq!(shape.flat_index(&[4, 0, 0]), None); // out of bounds
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    /// Per-dimension sizes; e.g. `[4, 8, 16]` for a 3-D tensor.
    pub dims: Vec<usize>,
}

impl TensorShape {
    /// Returns the total number of elements (product of all dimension sizes).
    ///
    /// An empty shape (0-dimensional tensor) returns `1`.
    #[must_use]
    pub fn total_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// Returns the C-order (row-major) strides for each dimension.
    ///
    /// `strides[i] = dims[i+1] * dims[i+2] * … * dims[rank-1]`.
    /// The last stride is always `1`.  For a scalar tensor (no dims) an empty
    /// `Vec` is returned.
    #[must_use]
    pub fn strides(&self) -> Vec<usize> {
        let rank = self.dims.len();
        let mut strides = vec![1usize; rank];
        // Fill from the second-to-last dimension backwards.
        for i in (0..rank.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }

    /// Converts a per-dimension index slice into a flat (linear) index.
    ///
    /// Returns `None` when:
    /// - `indices.len() != self.dims.len()`, or
    /// - any `indices[i] >= self.dims[i]`.
    #[must_use]
    pub fn flat_index(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.dims.len() {
            return None;
        }
        let strides = self.strides();
        let mut flat = 0usize;
        for (i, (&idx, &dim)) in indices.iter().zip(self.dims.iter()).enumerate() {
            if idx >= dim {
                return None;
            }
            flat += idx * strides[i];
        }
        Some(flat)
    }
}

// ---------------------------------------------------------------------------
// SliceSpec
// ---------------------------------------------------------------------------

/// Describes a per-dimension `[start, end)` range that selects a contiguous
/// sub-region of a tensor.
///
/// Both `start` and `end` must have the same length as the enclosing tensor's
/// rank, and for each dimension `start[i] <= end[i]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceSpec {
    /// Per-dimension start index (inclusive).
    pub start: Vec<usize>,
    /// Per-dimension end index (exclusive).
    pub end: Vec<usize>,
}

impl SliceSpec {
    /// Number of elements covered by this slice.
    ///
    /// Computed as `∏ (end[i] − start[i])`.  Zero-length dimensions contribute
    /// a factor of `0`, resulting in an empty slice.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.start
            .iter()
            .zip(self.end.iter())
            .map(|(&s, &e)| e.saturating_sub(s))
            .product()
    }

    /// Returns `true` when `self` and `other` share at least one element.
    ///
    /// Two slices overlap iff for **every** dimension `d`:
    /// `self.start[d] < other.end[d]  &&  other.start[d] < self.end[d]`.
    ///
    /// If the two specs have different ranks the function returns `false` rather
    /// than panicking.
    #[must_use]
    pub fn overlaps(&self, other: &SliceSpec) -> bool {
        if self.start.len() != other.start.len() {
            return false;
        }
        self.start
            .iter()
            .zip(self.end.iter())
            .zip(other.start.iter().zip(other.end.iter()))
            .all(|((&s, &e), (&os, &oe))| s < oe && os < e)
    }
}

// ---------------------------------------------------------------------------
// TensorSlice
// ---------------------------------------------------------------------------

/// A named, versioned, copy-on-write slice of a tensor.
#[derive(Debug, Clone)]
pub struct TensorSlice {
    /// Human-readable name used as the registry key.
    pub name: String,
    /// The region of the parent tensor this slice covers.
    pub spec: SliceSpec,
    /// Flattened element data; length must equal `spec.element_count()`.
    pub data: Vec<f32>,
    /// Monotonically increasing write counter; starts at `0`.
    pub version: u64,
    /// `true` when the slice has been written since the last [`TensorSliceManager::flush_all`].
    pub dirty: bool,
}

// ---------------------------------------------------------------------------
// SliceManagerStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for a [`TensorSliceManager`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceManagerStats {
    /// Total number of registered slices.
    pub total_slices: usize,
    /// Number of slices whose `dirty` flag is set.
    pub dirty_slices: usize,
    /// Sum of all element counts across all slices.
    pub total_elements: usize,
}

impl SliceManagerStats {
    /// Total memory used by all slice data buffers, in bytes.
    ///
    /// Each `f32` element occupies 4 bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.total_elements * 4
    }
}

// ---------------------------------------------------------------------------
// TensorSliceManager
// ---------------------------------------------------------------------------

/// Registry that manages named slices of a fixed-shape tensor.
///
/// # Semantics
///
/// * **Bounds checking** — every [`SliceSpec`] is validated against the
///   manager's [`TensorShape`] on creation.
/// * **Copy-on-write** — each write increments `version` and marks the slice
///   `dirty`; calling [`flush_all`](TensorSliceManager::flush_all) clears all
///   dirty flags (e.g. after persisting to storage).
/// * **Overlap detection** — [`overlapping_slices`](TensorSliceManager::overlapping_slices)
///   lists every registered slice that overlaps a given query spec.
pub struct TensorSliceManager {
    /// Shape of the underlying tensor.
    pub shape: TensorShape,
    /// Registered slices, keyed by name.
    pub slices: HashMap<String, TensorSlice>,
}

impl TensorSliceManager {
    /// Creates a new, empty manager for a tensor of the given `shape`.
    #[must_use]
    pub fn new(shape: TensorShape) -> Self {
        Self {
            shape,
            slices: HashMap::new(),
        }
    }

    /// Registers a new named slice.
    ///
    /// # Errors
    ///
    /// Returns `Err` with a descriptive message when:
    /// - a slice with the same `name` already exists,
    /// - `spec.end[i] > shape.dims[i]` for any dimension, or
    /// - `data.len() != spec.element_count()`.
    pub fn create_slice(
        &mut self,
        name: String,
        spec: SliceSpec,
        data: Vec<f32>,
    ) -> Result<(), String> {
        if self.slices.contains_key(&name) {
            return Err(format!("slice '{}' already exists", name));
        }

        // Validate rank.
        let rank = self.shape.dims.len();
        if spec.start.len() != rank || spec.end.len() != rank {
            return Err(format!(
                "spec rank {} does not match shape rank {}",
                spec.start.len(),
                rank
            ));
        }

        // Validate per-dimension bounds.
        for d in 0..rank {
            if spec.end[d] > self.shape.dims[d] {
                return Err(format!(
                    "spec.end[{d}] = {} exceeds shape.dims[{d}] = {}",
                    spec.end[d], self.shape.dims[d]
                ));
            }
            if spec.start[d] > spec.end[d] {
                return Err(format!(
                    "spec.start[{d}] = {} > spec.end[{d}] = {}",
                    spec.start[d], spec.end[d]
                ));
            }
        }

        // Validate data length.
        let expected = spec.element_count();
        if data.len() != expected {
            return Err(format!(
                "data.len() = {} but spec.element_count() = {}",
                data.len(),
                expected
            ));
        }

        self.slices.insert(
            name.clone(),
            TensorSlice {
                name,
                spec,
                data,
                version: 0,
                dirty: false,
            },
        );
        Ok(())
    }

    /// Overwrites the data of an existing slice.
    ///
    /// On success the slice's `version` is incremented and `dirty` is set to
    /// `true`.  Returns `false` when the slice does not exist or `data` has the
    /// wrong length.
    pub fn write_slice(&mut self, name: &str, data: Vec<f32>) -> bool {
        match self.slices.get_mut(name) {
            Some(slice) if slice.data.len() == data.len() => {
                slice.data = data;
                slice.version += 1;
                slice.dirty = true;
                true
            }
            _ => false,
        }
    }

    /// Returns an immutable reference to the named slice, or `None`.
    #[must_use]
    pub fn read_slice(&self, name: &str) -> Option<&TensorSlice> {
        self.slices.get(name)
    }

    /// Clears the `dirty` flag on every registered slice.
    ///
    /// Intended to be called after all dirty slices have been flushed to
    /// persistent storage.
    pub fn flush_all(&mut self) {
        for slice in self.slices.values_mut() {
            slice.dirty = false;
        }
    }

    /// Returns the names of all registered slices that overlap `spec`.
    ///
    /// The order of the returned names is unspecified (depends on the
    /// `HashMap` iteration order).
    #[must_use]
    pub fn overlapping_slices<'a>(&'a self, spec: &SliceSpec) -> Vec<&'a str> {
        self.slices
            .values()
            .filter(|s| s.spec.overlaps(spec))
            .map(|s| s.name.as_str())
            .collect()
    }

    /// Removes the named slice from the registry.
    ///
    /// Returns `true` if the slice existed and was removed, `false` otherwise.
    pub fn remove_slice(&mut self, name: &str) -> bool {
        self.slices.remove(name).is_some()
    }

    /// Computes aggregate statistics over all registered slices.
    #[must_use]
    pub fn stats(&self) -> SliceManagerStats {
        let total_slices = self.slices.len();
        let dirty_slices = self.slices.values().filter(|s| s.dirty).count();
        let total_elements = self.slices.values().map(|s| s.data.len()).sum();
        SliceManagerStats {
            total_slices,
            dirty_slices,
            total_elements,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helper constructors ------------------------------------------------

    fn shape_3d() -> TensorShape {
        TensorShape {
            dims: vec![4, 8, 16],
        }
    }

    fn spec(start: Vec<usize>, end: Vec<usize>) -> SliceSpec {
        SliceSpec { start, end }
    }

    fn make_data(n: usize, fill: f32) -> Vec<f32> {
        vec![fill; n]
    }

    // ---- TensorShape --------------------------------------------------------

    #[test]
    fn test_total_elements() {
        let s = TensorShape {
            dims: vec![4, 8, 16],
        };
        assert_eq!(s.total_elements(), 512);
    }

    #[test]
    fn test_total_elements_scalar() {
        let s = TensorShape { dims: vec![] };
        assert_eq!(s.total_elements(), 1);
    }

    #[test]
    fn test_strides_3d() {
        let s = shape_3d();
        assert_eq!(s.strides(), vec![128, 16, 1]);
    }

    #[test]
    fn test_strides_1d() {
        let s = TensorShape { dims: vec![10] };
        assert_eq!(s.strides(), vec![1]);
    }

    #[test]
    fn test_strides_2d() {
        let s = TensorShape { dims: vec![3, 5] };
        assert_eq!(s.strides(), vec![5, 1]);
    }

    #[test]
    fn test_flat_index_in_bounds() {
        let s = shape_3d();
        assert_eq!(s.flat_index(&[1, 2, 3]), Some(128 + 2 * 16 + 3));
        assert_eq!(s.flat_index(&[0, 0, 0]), Some(0));
        assert_eq!(s.flat_index(&[3, 7, 15]), Some(3 * 128 + 7 * 16 + 15));
    }

    #[test]
    fn test_flat_index_out_of_bounds() {
        let s = shape_3d();
        // First dimension out of bounds (4 >= 4)
        assert_eq!(s.flat_index(&[4, 0, 0]), None);
        // Last dimension out of bounds
        assert_eq!(s.flat_index(&[0, 0, 16]), None);
        // Wrong rank
        assert_eq!(s.flat_index(&[0, 0]), None);
    }

    // ---- SliceSpec ----------------------------------------------------------

    #[test]
    fn test_element_count() {
        let sp = spec(vec![1, 2, 4], vec![3, 6, 12]);
        // (3-1) * (6-2) * (12-4) = 2 * 4 * 8 = 64
        assert_eq!(sp.element_count(), 64);
    }

    #[test]
    fn test_element_count_full_slice() {
        let sp = spec(vec![0, 0, 0], vec![4, 8, 16]);
        assert_eq!(sp.element_count(), 512);
    }

    #[test]
    fn test_overlaps_true() {
        let a = spec(vec![0, 0], vec![4, 4]);
        let b = spec(vec![2, 2], vec![6, 6]);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_overlaps_false_adjacent() {
        // Adjacent but not overlapping: a ends at 4, b starts at 4.
        let a = spec(vec![0, 0], vec![4, 4]);
        let b = spec(vec![4, 0], vec![8, 4]);
        assert!(!a.overlaps(&b));
        assert!(!b.overlaps(&a));
    }

    #[test]
    fn test_overlaps_false_separated() {
        let a = spec(vec![0, 0], vec![2, 2]);
        let b = spec(vec![5, 5], vec![8, 8]);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_overlaps_contained() {
        let outer = spec(vec![0, 0], vec![8, 8]);
        let inner = spec(vec![2, 2], vec![4, 4]);
        assert!(outer.overlaps(&inner));
        assert!(inner.overlaps(&outer));
    }

    // ---- TensorSliceManager — create_slice ----------------------------------

    #[test]
    fn test_create_slice_success() {
        let mut mgr = TensorSliceManager::new(shape_3d());
        let sp = spec(vec![0, 0, 0], vec![2, 4, 8]);
        let n = sp.element_count(); // 2*4*8 = 64
        let result = mgr.create_slice("a".to_string(), sp, make_data(n, 1.0));
        assert!(result.is_ok());
        assert!(mgr.read_slice("a").is_some());
    }

    #[test]
    fn test_create_slice_duplicate_name_error() {
        let mut mgr = TensorSliceManager::new(shape_3d());
        let sp = spec(vec![0, 0, 0], vec![1, 1, 1]);
        mgr.create_slice("s".to_string(), sp.clone(), make_data(1, 0.0))
            .expect("test: should succeed");
        let err = mgr.create_slice("s".to_string(), sp, make_data(1, 0.0));
        assert!(err.is_err());
        let msg = err.unwrap_err();
        assert!(msg.contains("already exists"), "unexpected message: {msg}");
    }

    #[test]
    fn test_create_slice_bounds_check_error() {
        let mut mgr = TensorSliceManager::new(shape_3d());
        // end[2] = 17 > shape.dims[2] = 16  → out of bounds
        let sp = spec(vec![0, 0, 0], vec![1, 1, 17]);
        let err = mgr.create_slice("bad".to_string(), sp, make_data(17, 0.0));
        assert!(err.is_err());
    }

    #[test]
    fn test_create_slice_wrong_data_length_error() {
        let mut mgr = TensorSliceManager::new(shape_3d());
        let sp = spec(vec![0, 0, 0], vec![2, 2, 2]); // element_count = 8
        let err = mgr.create_slice("bad".to_string(), sp, make_data(5, 0.0)); // 5 ≠ 8
        assert!(err.is_err());
    }

    // ---- TensorSliceManager — write_slice -----------------------------------

    #[test]
    fn test_write_slice_increments_version() {
        let mut mgr = TensorSliceManager::new(shape_3d());
        let sp = spec(vec![0, 0, 0], vec![1, 1, 4]);
        mgr.create_slice("v".to_string(), sp, make_data(4, 0.0))
            .expect("test: should succeed");

        assert_eq!(
            mgr.read_slice("v").expect("test: should succeed").version,
            0
        );

        mgr.write_slice("v", make_data(4, 1.0));
        assert_eq!(
            mgr.read_slice("v").expect("test: should succeed").version,
            1
        );

        mgr.write_slice("v", make_data(4, 2.0));
        assert_eq!(
            mgr.read_slice("v").expect("test: should succeed").version,
            2
        );
    }

    #[test]
    fn test_write_slice_sets_dirty_flag() {
        let mut mgr = TensorSliceManager::new(shape_3d());
        let sp = spec(vec![0, 0, 0], vec![1, 1, 4]);
        mgr.create_slice("d".to_string(), sp, make_data(4, 0.0))
            .expect("test: should succeed");

        assert!(!mgr.read_slice("d").expect("test: should succeed").dirty);
        mgr.write_slice("d", make_data(4, 9.0));
        assert!(mgr.read_slice("d").expect("test: should succeed").dirty);
    }

    #[test]
    fn test_write_slice_returns_false_missing() {
        let mut mgr = TensorSliceManager::new(shape_3d());
        assert!(!mgr.write_slice("nonexistent", make_data(4, 0.0)));
    }

    #[test]
    fn test_write_slice_returns_false_wrong_length() {
        let mut mgr = TensorSliceManager::new(shape_3d());
        let sp = spec(vec![0, 0, 0], vec![1, 1, 4]); // element_count = 4
        mgr.create_slice("w".to_string(), sp, make_data(4, 0.0))
            .expect("test: should succeed");
        // Supply 5 elements instead of 4.
        assert!(!mgr.write_slice("w", make_data(5, 0.0)));
        // Version must not have changed.
        assert_eq!(
            mgr.read_slice("w").expect("test: should succeed").version,
            0
        );
    }

    // ---- TensorSliceManager — flush_all -------------------------------------

    #[test]
    fn test_flush_all_clears_dirty() {
        let mut mgr = TensorSliceManager::new(shape_3d());
        for name in ["x", "y", "z"] {
            let sp = spec(vec![0, 0, 0], vec![1, 1, 2]);
            mgr.create_slice(name.to_string(), sp, make_data(2, 0.0))
                .expect("test: should succeed");
            mgr.write_slice(name, make_data(2, 1.0));
        }
        // All three should be dirty.
        assert_eq!(mgr.stats().dirty_slices, 3);

        mgr.flush_all();

        assert_eq!(mgr.stats().dirty_slices, 0);
        for name in ["x", "y", "z"] {
            assert!(!mgr.read_slice(name).expect("test: should succeed").dirty);
        }
    }

    // ---- TensorSliceManager — overlapping_slices ----------------------------

    #[test]
    fn test_overlapping_slices_correct() {
        let shape = TensorShape { dims: vec![10, 10] };
        let mut mgr = TensorSliceManager::new(shape);

        // Slice A: rows 0-4, cols 0-4  (element_count = 16)
        let sa = spec(vec![0, 0], vec![4, 4]);
        mgr.create_slice("A".to_string(), sa, make_data(16, 1.0))
            .expect("test: should succeed");

        // Slice B: rows 3-7, cols 3-7  (element_count = 16) — overlaps A
        let sb = spec(vec![3, 3], vec![7, 7]);
        mgr.create_slice("B".to_string(), sb, make_data(16, 2.0))
            .expect("test: should succeed");

        // Slice C: rows 6-10, cols 6-10 (element_count = 16) — does NOT overlap A
        let sc = spec(vec![6, 6], vec![10, 10]);
        mgr.create_slice("C".to_string(), sc, make_data(16, 3.0))
            .expect("test: should succeed");

        // Query against A's region.
        let query = spec(vec![0, 0], vec![4, 4]);
        let mut hits = mgr.overlapping_slices(&query);
        hits.sort_unstable();
        assert_eq!(hits, vec!["A", "B"]);
    }

    #[test]
    fn test_non_overlapping_not_returned() {
        let shape = TensorShape { dims: vec![10, 10] };
        let mut mgr = TensorSliceManager::new(shape);

        let sa = spec(vec![0, 0], vec![4, 4]);
        mgr.create_slice("A".to_string(), sa, make_data(16, 0.0))
            .expect("test: should succeed");

        // Query in a completely separate region.
        let query = spec(vec![6, 6], vec![10, 10]);
        let hits = mgr.overlapping_slices(&query);
        assert!(hits.is_empty());
    }

    // ---- TensorSliceManager — read_slice / remove_slice ---------------------

    #[test]
    fn test_read_slice_returns_correct_data() {
        let mut mgr = TensorSliceManager::new(shape_3d());
        let sp = spec(vec![0, 0, 0], vec![1, 1, 4]);
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        mgr.create_slice("r".to_string(), sp, data.clone())
            .expect("test: should succeed");

        let slice = mgr.read_slice("r").expect("slice must exist");
        assert_eq!(slice.data, data);
        assert_eq!(slice.name, "r");
    }

    #[test]
    fn test_read_slice_missing_returns_none() {
        let mgr = TensorSliceManager::new(shape_3d());
        assert!(mgr.read_slice("nope").is_none());
    }

    #[test]
    fn test_remove_slice_existing() {
        let mut mgr = TensorSliceManager::new(shape_3d());
        let sp = spec(vec![0, 0, 0], vec![1, 1, 1]);
        mgr.create_slice("rm".to_string(), sp, make_data(1, 0.0))
            .expect("test: should succeed");

        assert!(mgr.remove_slice("rm"));
        assert!(mgr.read_slice("rm").is_none());
    }

    #[test]
    fn test_remove_slice_missing_returns_false() {
        let mut mgr = TensorSliceManager::new(shape_3d());
        assert!(!mgr.remove_slice("ghost"));
    }

    // ---- SliceManagerStats --------------------------------------------------

    #[test]
    fn test_stats_and_memory_bytes() {
        let shape = TensorShape { dims: vec![2, 2] };
        let mut mgr = TensorSliceManager::new(shape);

        let sp1 = spec(vec![0, 0], vec![2, 2]); // 4 elements
        mgr.create_slice("p".to_string(), sp1, make_data(4, 0.0))
            .expect("test: should succeed");

        let sp2 = spec(vec![0, 0], vec![1, 2]); // 2 elements
        mgr.create_slice("q".to_string(), sp2, make_data(2, 0.0))
            .expect("test: should succeed");

        // Dirty one of them.
        mgr.write_slice("p", make_data(4, 1.0));

        let st = mgr.stats();
        assert_eq!(st.total_slices, 2);
        assert_eq!(st.dirty_slices, 1);
        assert_eq!(st.total_elements, 6); // 4 + 2
        assert_eq!(st.memory_bytes(), 24); // 6 * 4
    }
}
