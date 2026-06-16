//! `TensorSliceView` — zero-copy logical views into tensor data via
//! offset+stride descriptors, supporting slicing, broadcasting, and element
//! access without data duplication.
//!
//! # Overview
//!
//! A [`TensorSliceView`] holds a flat `Vec<f64>` as the backing store and a
//! [`ViewDescriptor`] that describes the logical shape, strides, and starting
//! offset into that store.  All slice and broadcast operations return new
//! descriptors that share the same flat buffer – no data is ever copied.
//!
//! # Key types
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`SliceRange`] | A half-open `[start, stop)` range with a `step` |
//! | [`ViewDescriptor`] | Offset + shape + strides descriptor for a view |
//! | [`BroadcastShape`] | NumPy-style broadcast compatibility and stride computation |
//! | [`SliceViewStats`] | Cumulative counters for views, slices, and broadcasts |
//! | [`TensorSliceView`] | The main manager that owns the data and exposes the API |

// ---------------------------------------------------------------------------
// SliceRange
// ---------------------------------------------------------------------------

/// A half-open range `[start, stop)` with a positive `step`.
///
/// # Examples
///
/// ```
/// use ipfrs_tensorlogic::slice_view::SliceRange;
///
/// let r = SliceRange { start: 0, stop: 10, step: 2 };
/// assert_eq!(r.len(), 5);
/// assert_eq!(r.indices(), vec![0, 2, 4, 6, 8]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceRange {
    /// First index to include.
    pub start: usize,
    /// Exclusive upper bound.
    pub stop: usize,
    /// Step between successive indices; must be ≥ 1 for a non-empty range.
    pub step: usize,
}

impl Default for SliceRange {
    fn default() -> Self {
        Self {
            start: 0,
            stop: 0,
            step: 1,
        }
    }
}

impl SliceRange {
    /// Create a new `SliceRange` with an explicit step.
    #[must_use]
    pub fn new(start: usize, stop: usize, step: usize) -> Self {
        Self { start, stop, step }
    }

    /// Create a contiguous range `[start, stop)` with `step = 1`.
    #[must_use]
    pub fn contiguous(start: usize, stop: usize) -> Self {
        Self {
            start,
            stop,
            step: 1,
        }
    }

    /// Number of elements produced by this range.
    ///
    /// Returns `0` if `step == 0` or `start >= stop`.
    /// Otherwise uses ceiling division: `⌈(stop − start) / step⌉`.
    #[must_use]
    pub fn len(&self) -> usize {
        if self.step == 0 || self.start >= self.stop {
            return 0;
        }
        let span = self.stop - self.start;
        span.div_ceil(self.step)
    }

    /// Returns `true` when the range contains no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Materialise all indices: `start, start+step, start+2*step, …` while `< stop`.
    #[must_use]
    pub fn indices(&self) -> Vec<usize> {
        if self.step == 0 || self.start >= self.stop {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(self.len());
        let mut idx = self.start;
        while idx < self.stop {
            result.push(idx);
            idx = match idx.checked_add(self.step) {
                Some(v) => v,
                None => break,
            };
        }
        result
    }
}

// ---------------------------------------------------------------------------
// ViewDescriptor
// ---------------------------------------------------------------------------

/// Describes a logical view into a flat buffer via an offset, shape, and strides.
///
/// The flat index of element `(i₀, i₁, …, iₙ₋₁)` is:
///
/// ```text
/// base_offset + Σ iₖ * strides[k]
/// ```
///
/// # Examples
///
/// ```
/// use ipfrs_tensorlogic::slice_view::ViewDescriptor;
///
/// // 2×3 row-major view starting at element 0
/// let desc = ViewDescriptor {
///     base_offset: 0,
///     shape: vec![2, 3],
///     strides: vec![3, 1],
/// };
/// assert_eq!(desc.flat_index(&[1, 2]), Some(5));
/// assert!(desc.is_contiguous());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewDescriptor {
    /// Starting element index in the flat backing buffer.
    pub base_offset: usize,
    /// Logical shape of the view.
    pub shape: Vec<usize>,
    /// Logical strides (one per dimension).
    pub strides: Vec<usize>,
}

impl ViewDescriptor {
    /// Number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of logical elements (product of `shape`; `1` for a scalar).
    #[must_use]
    pub fn total_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Compute the flat index for a multi-dimensional index tuple.
    ///
    /// Returns `None` if:
    /// - `indices.len() != self.ndim()`, or
    /// - any `indices[i] >= self.shape[i]`.
    #[must_use]
    pub fn flat_index(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.ndim() {
            return None;
        }
        let mut offset = self.base_offset;
        for (dim, (&idx, (&dim_size, &stride))) in indices
            .iter()
            .zip(self.shape.iter().zip(self.strides.iter()))
            .enumerate()
        {
            let _ = dim; // suppress unused warning
            if idx >= dim_size {
                return None;
            }
            offset = offset.checked_add(idx * stride)?;
        }
        Some(offset)
    }

    /// Returns `true` if the view is C (row-major) contiguous.
    ///
    /// A view is contiguous when `strides[ndim-1] == 1` and each preceding
    /// stride equals the product of the following shape dimensions.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        let n = self.ndim();
        if n == 0 {
            return true;
        }
        if self.strides[n - 1] != 1 {
            return false;
        }
        let mut expected = 1usize;
        for i in (0..n).rev() {
            if self.strides[i] != expected {
                return false;
            }
            expected *= self.shape[i];
        }
        true
    }
}

// ---------------------------------------------------------------------------
// BroadcastShape
// ---------------------------------------------------------------------------

/// NumPy-style broadcast compatibility checker and stride generator.
///
/// # Examples
///
/// ```
/// use ipfrs_tensorlogic::slice_view::BroadcastShape;
///
/// let bs = BroadcastShape { shape: vec![1, 3] };
/// assert!(bs.compatible_with(&[4, 3]));
///
/// let strides = bs.broadcast_to(&[4, 3]).expect("example: should succeed in docs");
/// assert_eq!(strides, vec![0, 1]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BroadcastShape {
    /// The source shape to be broadcast.
    pub shape: Vec<usize>,
}

impl BroadcastShape {
    /// Check NumPy-style broadcast compatibility with `other`.
    ///
    /// Dimensions are aligned from the right.  Each pair `(a, b)` must satisfy:
    /// `a == b`, `a == 1`, or `b == 1`.
    #[must_use]
    pub fn compatible_with(&self, other: &[usize]) -> bool {
        let self_ndim = self.shape.len();
        let other_ndim = other.len();

        for k in 0..self_ndim.max(other_ndim) {
            let a = if k < self_ndim {
                self.shape[self_ndim - 1 - k]
            } else {
                1
            };
            let b = if k < other_ndim {
                other[other_ndim - 1 - k]
            } else {
                1
            };
            if a != b && a != 1 && b != 1 {
                return false;
            }
        }
        true
    }

    /// Compute the broadcast strides needed to view `self` as `target`.
    ///
    /// A source dimension of size `1` gets stride `0` (it is repeated along
    /// that axis without touching memory).  Equal dimensions get the natural
    /// row-major stride computed from the *source* shape.
    ///
    /// Returns `None` if the shapes are not compatible.
    #[must_use]
    pub fn broadcast_to(&self, target: &[usize]) -> Option<Vec<usize>> {
        if !self.compatible_with(target) {
            return None;
        }

        let target_ndim = target.len();
        let src_ndim = self.shape.len();

        // Compute natural row-major strides for the source shape.
        let src_strides = row_major_strides(&self.shape);

        let mut strides = vec![0usize; target_ndim];
        for k in 0..target_ndim {
            // Align from the right.
            let src_k_rev = k; // k from the right in target
            let src_k = if src_k_rev < src_ndim {
                src_ndim - 1 - src_k_rev
            } else {
                // Source has no dimension here — treat as 1 → stride 0
                let out_k = target_ndim - 1 - k;
                strides[out_k] = 0;
                continue;
            };
            let out_k = target_ndim - 1 - k;
            let src_dim = self.shape[src_k];
            let tgt_dim = target[out_k];

            if src_dim == 1 && tgt_dim != 1 {
                // Broadcast dimension: stride 0 repeats the single element.
                strides[out_k] = 0;
            } else {
                // Equal dimension: use natural stride.
                strides[out_k] = src_strides[src_k];
            }
        }

        Some(strides)
    }
}

// ---------------------------------------------------------------------------
// SliceViewStats
// ---------------------------------------------------------------------------

/// Cumulative statistics for a [`TensorSliceView`].
#[derive(Debug, Clone, Default)]
pub struct SliceViewStats {
    /// Total number of `TensorSliceView` instances created (via `new`).
    pub total_views_created: u64,
    /// Total number of `slice` calls performed.
    pub total_slices: u64,
    /// Total number of `broadcast_strides` calls performed.
    pub total_broadcasts: u64,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute row-major (C-order) strides for `shape`.
///
/// For a shape `[d₀, d₁, …, dₙ₋₁]` the strides are
/// `[d₁·d₂·…·dₙ₋₁, d₂·…·dₙ₋₁, …, 1]`.
#[must_use]
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    if n == 0 {
        return Vec::new();
    }
    let mut strides = vec![1usize; n];
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// ---------------------------------------------------------------------------
// TensorSliceView
// ---------------------------------------------------------------------------

/// Zero-copy logical view manager over a flat `f64` data buffer.
///
/// All slice/broadcast operations produce new [`ViewDescriptor`]s that share
/// the same backing `data` — no elements are ever duplicated.
///
/// # Examples
///
/// ```
/// use ipfrs_tensorlogic::slice_view::{TensorSliceView, SliceRange};
///
/// // 2×3 tensor filled with 0..6
/// let data: Vec<f64> = (0..6).map(|x| x as f64).collect();
/// let mut view = TensorSliceView::new(data, vec![2, 3]);
///
/// assert_eq!(view.get(&[0, 0]), Some(0.0));
/// assert_eq!(view.get(&[1, 2]), Some(5.0));
///
/// // Slice row 1 only
/// let slice_desc = view.slice(0, SliceRange::contiguous(1, 2)).expect("example: should succeed in docs");
/// assert_eq!(slice_desc.base_offset, 3);
/// assert_eq!(slice_desc.shape, vec![1, 3]);
/// ```
#[derive(Debug)]
pub struct TensorSliceView {
    /// Flat backing buffer.
    pub data: Vec<f64>,
    /// Current logical view descriptor.
    pub descriptor: ViewDescriptor,
    /// Cumulative statistics (slices, broadcasts, views created).
    pub stats: SliceViewStats,
}

impl TensorSliceView {
    /// Create a new `TensorSliceView` over `data` with the given logical `shape`.
    ///
    /// Row-major strides are computed automatically and `base_offset` is set to `0`.
    /// `stats.total_views_created` is incremented.
    #[must_use]
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let strides = row_major_strides(&shape);
        let descriptor = ViewDescriptor {
            base_offset: 0,
            shape,
            strides,
        };
        let stats = SliceViewStats {
            total_views_created: 1,
            ..Default::default()
        };
        Self {
            data,
            descriptor,
            stats,
        }
    }

    /// Retrieve a single element by its multi-dimensional index.
    ///
    /// Returns `None` if the index is out of bounds for the logical shape *or*
    /// if the computed flat index exceeds the backing buffer length.
    #[must_use]
    pub fn get(&self, indices: &[usize]) -> Option<f64> {
        let flat = self.descriptor.flat_index(indices)?;
        self.data.get(flat).copied()
    }

    /// Apply a slice along `dim` and return the resulting [`ViewDescriptor`].
    ///
    /// The `data` buffer is **not** modified; the new descriptor adjusts
    /// `base_offset` and `strides[dim]` to reflect the slice.
    ///
    /// Returns `None` if:
    /// - `dim >= self.descriptor.ndim()`, or
    /// - `range.start >= self.descriptor.shape[dim]`.
    ///
    /// `stats.total_slices` is incremented on success.
    pub fn slice(&mut self, dim: usize, range: SliceRange) -> Option<ViewDescriptor> {
        let ndim = self.descriptor.ndim();
        if dim >= ndim {
            return None;
        }
        if range.start >= self.descriptor.shape[dim] {
            return None;
        }

        let mut new_shape = self.descriptor.shape.clone();
        let mut new_strides = self.descriptor.strides.clone();

        // The new base offset is shifted by range.start * strides[dim].
        let new_base_offset = self
            .descriptor
            .base_offset
            .checked_add(range.start * self.descriptor.strides[dim])?;

        // The length of this dimension becomes range.len() (may be 0 for empty slice).
        new_shape[dim] = range.len();

        // The stride along this dimension is scaled by range.step.
        new_strides[dim] = self.descriptor.strides[dim].checked_mul(range.step.max(1))?;

        let desc = ViewDescriptor {
            base_offset: new_base_offset,
            shape: new_shape,
            strides: new_strides,
        };

        self.stats.total_slices += 1;
        Some(desc)
    }

    /// Compute broadcast strides to view the current logical shape as `target_shape`.
    ///
    /// Uses [`BroadcastShape`] internally.  Returns `None` if the shapes are
    /// incompatible.  `stats.total_broadcasts` is incremented on success.
    pub fn broadcast_strides(&mut self, target_shape: &[usize]) -> Option<Vec<usize>> {
        let bs = BroadcastShape {
            shape: self.descriptor.shape.clone(),
        };
        let strides = bs.broadcast_to(target_shape)?;
        self.stats.total_broadcasts += 1;
        Some(strides)
    }

    /// Return a reference to the cumulative statistics.
    #[must_use]
    pub fn stats(&self) -> &SliceViewStats {
        &self.stats
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── SliceRange::len ───────────────────────────────────────────────────────

    #[test]
    fn slice_range_len_step_zero_is_empty() {
        let r = SliceRange {
            start: 0,
            stop: 10,
            step: 0,
        };
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn slice_range_len_start_ge_stop_is_empty() {
        let r = SliceRange {
            start: 5,
            stop: 5,
            step: 1,
        };
        assert_eq!(r.len(), 0);
        let r2 = SliceRange {
            start: 6,
            stop: 5,
            step: 1,
        };
        assert_eq!(r2.len(), 0);
    }

    #[test]
    fn slice_range_len_step_one() {
        let r = SliceRange::contiguous(2, 7);
        assert_eq!(r.len(), 5);
    }

    #[test]
    fn slice_range_len_step_two_even() {
        // [0,2,4,6,8] => 5 elements from [0,10) step 2
        let r = SliceRange::new(0, 10, 2);
        assert_eq!(r.len(), 5);
    }

    #[test]
    fn slice_range_len_step_two_odd_span() {
        // [0,2,4] => 3 elements from [0,5) step 2
        let r = SliceRange::new(0, 5, 2);
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn slice_range_len_step_larger_than_span() {
        // step=10, span=5 → only start is included
        let r = SliceRange::new(0, 5, 10);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn slice_range_len_step_three() {
        // [1,4,7] => 3 elements from [1,9) step 3
        let r = SliceRange::new(1, 9, 3);
        assert_eq!(r.len(), 3);
    }

    // ── SliceRange::indices ───────────────────────────────────────────────────

    #[test]
    fn slice_range_indices_step_one() {
        let r = SliceRange::contiguous(3, 6);
        assert_eq!(r.indices(), vec![3, 4, 5]);
    }

    #[test]
    fn slice_range_indices_step_two() {
        let r = SliceRange::new(0, 10, 2);
        assert_eq!(r.indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn slice_range_indices_empty_step_zero() {
        let r = SliceRange {
            start: 0,
            stop: 10,
            step: 0,
        };
        assert!(r.indices().is_empty());
    }

    #[test]
    fn slice_range_indices_start_ge_stop() {
        let r = SliceRange::new(5, 3, 1);
        assert!(r.indices().is_empty());
    }

    // ── ViewDescriptor::flat_index ────────────────────────────────────────────

    #[test]
    fn view_descriptor_flat_index_2d_correct() {
        // 3×4 row-major: strides [4, 1]
        let desc = ViewDescriptor {
            base_offset: 0,
            shape: vec![3, 4],
            strides: vec![4, 1],
        };
        assert_eq!(desc.flat_index(&[0, 0]), Some(0));
        assert_eq!(desc.flat_index(&[1, 2]), Some(6));
        assert_eq!(desc.flat_index(&[2, 3]), Some(11));
    }

    #[test]
    fn view_descriptor_flat_index_with_base_offset() {
        let desc = ViewDescriptor {
            base_offset: 10,
            shape: vec![2, 2],
            strides: vec![2, 1],
        };
        assert_eq!(desc.flat_index(&[0, 0]), Some(10));
        assert_eq!(desc.flat_index(&[1, 1]), Some(13));
    }

    #[test]
    fn view_descriptor_flat_index_none_wrong_ndim() {
        let desc = ViewDescriptor {
            base_offset: 0,
            shape: vec![3, 4],
            strides: vec![4, 1],
        };
        assert_eq!(desc.flat_index(&[0]), None);
        assert_eq!(desc.flat_index(&[0, 0, 0]), None);
    }

    #[test]
    fn view_descriptor_flat_index_none_out_of_bounds() {
        let desc = ViewDescriptor {
            base_offset: 0,
            shape: vec![3, 4],
            strides: vec![4, 1],
        };
        assert_eq!(desc.flat_index(&[3, 0]), None); // row out of bounds
        assert_eq!(desc.flat_index(&[0, 4]), None); // col out of bounds
    }

    // ── ViewDescriptor::is_contiguous ─────────────────────────────────────────

    #[test]
    fn view_descriptor_is_contiguous_row_major() {
        let desc = ViewDescriptor {
            base_offset: 0,
            shape: vec![2, 3, 4],
            strides: vec![12, 4, 1],
        };
        assert!(desc.is_contiguous());
    }

    #[test]
    fn view_descriptor_is_not_contiguous_non_unit_last_stride() {
        let desc = ViewDescriptor {
            base_offset: 0,
            shape: vec![2, 3],
            strides: vec![6, 2], // last stride should be 1
        };
        assert!(!desc.is_contiguous());
    }

    #[test]
    fn view_descriptor_is_not_contiguous_broadcast_zero_stride() {
        let desc = ViewDescriptor {
            base_offset: 0,
            shape: vec![3, 4],
            strides: vec![0, 1], // broadcast zero stride
        };
        assert!(!desc.is_contiguous());
    }

    // ── TensorSliceView::new ──────────────────────────────────────────────────

    #[test]
    fn tensor_slice_view_new_correct_strides_1d() {
        let data = vec![1.0, 2.0, 3.0];
        let view = TensorSliceView::new(data, vec![3]);
        assert_eq!(view.descriptor.strides, vec![1]);
        assert_eq!(view.descriptor.base_offset, 0);
        assert_eq!(view.stats.total_views_created, 1);
    }

    #[test]
    fn tensor_slice_view_new_correct_strides_3d() {
        let data = vec![0.0; 24];
        let view = TensorSliceView::new(data, vec![2, 3, 4]);
        // strides: [12, 4, 1]
        assert_eq!(view.descriptor.strides, vec![12, 4, 1]);
    }

    // ── TensorSliceView::get ──────────────────────────────────────────────────

    #[test]
    fn tensor_slice_view_get_correct_element() {
        // 2×3 tensor: [[0,1,2],[3,4,5]]
        let data: Vec<f64> = (0..6).map(|x| x as f64).collect();
        let view = TensorSliceView::new(data, vec![2, 3]);
        assert_eq!(view.get(&[0, 0]), Some(0.0));
        assert_eq!(view.get(&[0, 2]), Some(2.0));
        assert_eq!(view.get(&[1, 0]), Some(3.0));
        assert_eq!(view.get(&[1, 2]), Some(5.0));
    }

    #[test]
    fn tensor_slice_view_get_none_out_of_bounds() {
        let data: Vec<f64> = (0..6).map(|x| x as f64).collect();
        let view = TensorSliceView::new(data, vec![2, 3]);
        assert_eq!(view.get(&[2, 0]), None);
        assert_eq!(view.get(&[0, 3]), None);
    }

    // ── TensorSliceView::slice ────────────────────────────────────────────────

    #[test]
    fn slice_on_dim0_updates_base_offset() {
        // 3×4 tensor; slice dim 0: rows 1..3 → base_offset = 1*4 = 4
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let mut view = TensorSliceView::new(data, vec![3, 4]);
        let desc = view
            .slice(0, SliceRange::contiguous(1, 3))
            .expect("test: should succeed");
        assert_eq!(desc.base_offset, 4);
        assert_eq!(desc.shape, vec![2, 4]);
        assert_eq!(desc.strides, vec![4, 1]);
        assert_eq!(view.stats.total_slices, 1);
    }

    #[test]
    fn slice_with_step_multiplies_stride() {
        // 1-D tensor of 10 elements; slice [0,10) step 3 → stride[0] = 3
        let data: Vec<f64> = (0..10).map(|x| x as f64).collect();
        let mut view = TensorSliceView::new(data, vec![10]);
        let desc = view
            .slice(0, SliceRange::new(0, 10, 3))
            .expect("test: should succeed");
        assert_eq!(desc.strides[0], 3);
        assert_eq!(desc.shape[0], 4); // ⌈10/3⌉ = 4
    }

    #[test]
    fn slice_returns_none_for_invalid_dim() {
        let data = vec![1.0, 2.0, 3.0];
        let mut view = TensorSliceView::new(data, vec![3]);
        assert!(view.slice(1, SliceRange::contiguous(0, 1)).is_none());
    }

    #[test]
    fn slice_returns_none_when_start_ge_shape_dim() {
        let data = vec![1.0, 2.0, 3.0];
        let mut view = TensorSliceView::new(data, vec![3]);
        assert!(view.slice(0, SliceRange::contiguous(3, 4)).is_none());
    }

    #[test]
    fn slice_stat_incremented() {
        let data: Vec<f64> = (0..6).map(|x| x as f64).collect();
        let mut view = TensorSliceView::new(data, vec![2, 3]);
        let _ = view.slice(1, SliceRange::contiguous(0, 2));
        let _ = view.slice(0, SliceRange::contiguous(0, 1));
        assert_eq!(view.stats.total_slices, 2);
    }

    // ── BroadcastShape::compatible_with ──────────────────────────────────────

    #[test]
    fn broadcast_compatible_same_shape() {
        let bs = BroadcastShape { shape: vec![3, 4] };
        assert!(bs.compatible_with(&[3, 4]));
    }

    #[test]
    fn broadcast_compatible_scalar_broadcasts_everywhere() {
        let bs = BroadcastShape { shape: vec![1] };
        assert!(bs.compatible_with(&[5, 7]));
    }

    #[test]
    fn broadcast_compatible_leading_ones() {
        // [1, 3] broadcasts to [4, 3]
        let bs = BroadcastShape { shape: vec![1, 3] };
        assert!(bs.compatible_with(&[4, 3]));
    }

    #[test]
    fn broadcast_incompatible_mismatched_dims() {
        let bs = BroadcastShape { shape: vec![2, 3] };
        assert!(!bs.compatible_with(&[2, 4]));
    }

    // ── TensorSliceView::broadcast_strides ────────────────────────────────────

    #[test]
    fn broadcast_strides_compatible_returns_strides() {
        // [1, 3] → [4, 3]: dim 0 has stride 0, dim 1 has stride 1
        let data = vec![1.0, 2.0, 3.0];
        let mut view = TensorSliceView::new(data, vec![1, 3]);
        let strides = view
            .broadcast_strides(&[4, 3])
            .expect("test: should succeed");
        assert_eq!(strides[0], 0); // broadcast dim
        assert_eq!(strides[1], 1); // preserved dim
        assert_eq!(view.stats.total_broadcasts, 1);
    }

    #[test]
    fn broadcast_strides_incompatible_returns_none() {
        let data = vec![1.0, 2.0];
        let mut view = TensorSliceView::new(data, vec![2]);
        assert!(view.broadcast_strides(&[3]).is_none());
        assert_eq!(view.stats.total_broadcasts, 0);
    }

    #[test]
    fn broadcast_strides_stat_incremented() {
        let data = vec![0.0; 4];
        let mut view = TensorSliceView::new(data, vec![1, 4]);
        let _ = view.broadcast_strides(&[3, 4]);
        let _ = view.broadcast_strides(&[5, 4]);
        assert_eq!(view.stats.total_broadcasts, 2);
    }

    // ── Stats reference ───────────────────────────────────────────────────────

    #[test]
    fn stats_getter_returns_reference() {
        let data = vec![0.0; 6];
        let mut view = TensorSliceView::new(data, vec![2, 3]);
        let _ = view.slice(0, SliceRange::contiguous(0, 1));
        assert_eq!(view.stats().total_slices, 1);
        assert_eq!(view.stats().total_views_created, 1);
    }

    // ── Combined workflow ─────────────────────────────────────────────────────

    #[test]
    fn combined_slice_and_get() {
        // 3×4 tensor: row 2 starts at flat index 8
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let mut view = TensorSliceView::new(data, vec![3, 4]);
        // Slice rows 1..=2 (i.e. [1,3))
        let desc = view
            .slice(0, SliceRange::contiguous(1, 3))
            .expect("test: should succeed");
        // Element [0, 2] in the new view corresponds to [1, 2] in original → flat 6
        let flat = desc.flat_index(&[0, 2]).expect("test: should succeed");
        assert_eq!(view.data[flat], 6.0);
    }

    #[test]
    fn row_major_strides_scalar() {
        assert_eq!(row_major_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn row_major_strides_1d() {
        assert_eq!(row_major_strides(&[5]), vec![1]);
    }

    #[test]
    fn row_major_strides_3d() {
        assert_eq!(row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }
}
