//! Tensor memory layout management for multi-dimensional arrays.
//!
//! Provides layout descriptors that capture shape, strides, byte offsets,
//! and layout transformations (row-major / column-major, transposition, slicing).

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// LayoutOrder
// ---------------------------------------------------------------------------

/// Memory ordering for a multi-dimensional tensor.
///
/// - `RowMajor` (C-style): last dimension varies fastest.
/// - `ColMajor` (Fortran-style): first dimension varies fastest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutOrder {
    /// C-style layout — last dimension changes fastest in memory.
    RowMajor,
    /// Fortran-style layout — first dimension changes fastest in memory.
    ColMajor,
}

// ---------------------------------------------------------------------------
// TensorShape
// ---------------------------------------------------------------------------

/// Describes the logical shape of a tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorShape {
    /// Sizes of each dimension, e.g. `[3, 4, 5]` for a 3-D tensor.
    pub dims: Vec<usize>,
}

impl TensorShape {
    /// Creates a new `TensorShape` from the given dimension sizes.
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Total number of elements (product of all dims).  Returns 1 for a
    /// zero-dimensional (scalar) shape.
    pub fn total_elements(&self) -> usize {
        self.dims
            .iter()
            .copied()
            .fold(1usize, usize::saturating_mul)
    }

    /// Returns `true` for scalar tensors (zero dims or `total_elements == 1`).
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty() || self.total_elements() == 1
    }
}

// ---------------------------------------------------------------------------
// LayoutDescriptor
// ---------------------------------------------------------------------------

/// Complete memory layout descriptor for a tensor.
#[derive(Clone, Debug)]
pub struct LayoutDescriptor {
    /// Logical shape of the tensor.
    pub shape: TensorShape,
    /// Stride (in *elements*, not bytes) for each dimension.
    pub strides: Vec<usize>,
    /// Byte offset from the start of the backing buffer to element [0,…,0].
    pub byte_offset: usize,
    /// Size of a single element in bytes (e.g. 4 for `f32`, 8 for `f64`).
    pub element_size_bytes: usize,
    /// Memory ordering of this layout.
    pub order: LayoutOrder,
}

impl LayoutDescriptor {
    // -----------------------------------------------------------------------
    // Stride computation helpers
    // -----------------------------------------------------------------------

    /// Compute row-major (C-style) strides for `dims`.
    ///
    /// The last dimension has stride 1; each preceding stride equals the
    /// product of all following dimensions.  For `[3, 4, 5]` the result is
    /// `[20, 5, 1]`.
    pub fn row_major_strides(dims: &[usize]) -> Vec<usize> {
        let n = dims.len();
        if n == 0 {
            return Vec::new();
        }
        let mut strides = vec![0usize; n];
        strides[n - 1] = 1;
        // Walk backwards: stride[i] = stride[i+1] * dims[i+1]
        for i in (0..n - 1).rev() {
            strides[i] = strides[i + 1].saturating_mul(dims[i + 1]);
        }
        strides
    }

    /// Compute column-major (Fortran-style) strides for `dims`.
    ///
    /// The first dimension has stride 1; each following stride equals the
    /// product of all preceding dimensions.  For `[3, 4, 5]` the result is
    /// `[1, 3, 12]`.
    pub fn col_major_strides(dims: &[usize]) -> Vec<usize> {
        let n = dims.len();
        if n == 0 {
            return Vec::new();
        }
        let mut strides = vec![0usize; n];
        strides[0] = 1;
        for i in 1..n {
            strides[i] = strides[i - 1].saturating_mul(dims[i - 1]);
        }
        strides
    }

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    /// Create a new layout descriptor.
    ///
    /// Strides are computed from `order`; `byte_offset` starts at 0.
    pub fn new(shape: TensorShape, order: LayoutOrder, element_size_bytes: usize) -> Self {
        let strides = match order {
            LayoutOrder::RowMajor => Self::row_major_strides(&shape.dims),
            LayoutOrder::ColMajor => Self::col_major_strides(&shape.dims),
        };
        Self {
            shape,
            strides,
            byte_offset: 0,
            element_size_bytes,
            order,
        }
    }

    // -----------------------------------------------------------------------
    // Index computation
    // -----------------------------------------------------------------------

    /// Compute the flat (linear) element index for the given multi-dimensional
    /// `indices`.
    ///
    /// Returns `None` when:
    /// - `indices.len() != ndim`, or
    /// - any `indices[i] >= shape.dims[i]`.
    pub fn linear_index(&self, indices: &[usize]) -> Option<usize> {
        let ndim = self.shape.ndim();
        if indices.len() != ndim {
            return None;
        }
        let mut idx = 0usize;
        for (i, &coord) in indices.iter().enumerate() {
            if coord >= self.shape.dims[i] {
                return None;
            }
            idx = idx.saturating_add(coord.saturating_mul(self.strides[i]));
        }
        Some(idx)
    }

    /// Compute the byte offset into the backing buffer for `indices`.
    ///
    /// Returns `None` under the same conditions as `linear_index`.
    pub fn byte_offset_for(&self, indices: &[usize]) -> Option<usize> {
        let lin = self.linear_index(indices)?;
        Some(
            lin.saturating_mul(self.element_size_bytes)
                .saturating_add(self.byte_offset),
        )
    }

    // -----------------------------------------------------------------------
    // Layout properties
    // -----------------------------------------------------------------------

    /// Returns `true` if this layout has row-major (C-contiguous) strides.
    pub fn is_contiguous(&self) -> bool {
        self.strides == Self::row_major_strides(&self.shape.dims)
    }

    /// Total bytes occupied by all elements.
    pub fn total_bytes(&self) -> usize {
        self.shape
            .total_elements()
            .saturating_mul(self.element_size_bytes)
    }

    // -----------------------------------------------------------------------
    // Transformations
    // -----------------------------------------------------------------------

    /// Return a new `LayoutDescriptor` representing the transpose of this
    /// layout.
    ///
    /// Dimension and stride order are reversed; memory ordering is flipped
    /// (`RowMajor` ↔ `ColMajor`).
    pub fn transposed(&self) -> Self {
        let mut new_dims = self.shape.dims.clone();
        new_dims.reverse();
        let mut new_strides = self.strides.clone();
        new_strides.reverse();
        let new_order = match self.order {
            LayoutOrder::RowMajor => LayoutOrder::ColMajor,
            LayoutOrder::ColMajor => LayoutOrder::RowMajor,
        };
        Self {
            shape: TensorShape::new(new_dims),
            strides: new_strides,
            byte_offset: self.byte_offset,
            element_size_bytes: self.element_size_bytes,
            order: new_order,
        }
    }
}

// ---------------------------------------------------------------------------
// LayoutStats
// ---------------------------------------------------------------------------

/// Cumulative statistics gathered by [`TensorMemoryLayout`].
#[derive(Clone, Debug, Default)]
pub struct LayoutStats {
    /// Total number of layout descriptors ever created.
    pub total_layouts_created: u64,
    /// Total number of transposition operations performed.
    pub total_transpositions: u64,
    /// Layouts that were contiguous at creation time.
    pub contiguous_count: u64,
    /// Layouts that were *not* contiguous at creation time.
    pub non_contiguous_count: u64,
}

// ---------------------------------------------------------------------------
// TensorMemoryLayout  (manager)
// ---------------------------------------------------------------------------

/// Manager for a collection of [`LayoutDescriptor`]s.
///
/// Each descriptor is assigned a unique `u64` identifier on creation.
/// Provides statistics tracking for auditing and profiling purposes.
#[derive(Debug)]
pub struct TensorMemoryLayout {
    layouts: HashMap<u64, LayoutDescriptor>,
    next_id: u64,
    stats: LayoutStats,
}

impl TensorMemoryLayout {
    /// Create an empty layout manager.
    pub fn new() -> Self {
        Self {
            layouts: HashMap::new(),
            next_id: 0,
            stats: LayoutStats::default(),
        }
    }

    /// Create a new [`LayoutDescriptor`], store it, and return its id.
    pub fn create(
        &mut self,
        shape: TensorShape,
        order: LayoutOrder,
        element_size_bytes: usize,
    ) -> u64 {
        let descriptor = LayoutDescriptor::new(shape, order, element_size_bytes);
        if descriptor.is_contiguous() {
            self.stats.contiguous_count += 1;
        } else {
            self.stats.non_contiguous_count += 1;
        }
        self.stats.total_layouts_created += 1;
        let id = self.next_id;
        self.next_id += 1;
        self.layouts.insert(id, descriptor);
        id
    }

    /// Transpose an existing layout and store the result as a new entry.
    ///
    /// Returns `None` if `layout_id` does not exist; otherwise returns the
    /// id of the newly created transposed layout.
    pub fn transpose(&mut self, layout_id: u64) -> Option<u64> {
        let transposed = self.layouts.get(&layout_id)?.transposed();
        self.stats.total_transpositions += 1;
        if transposed.is_contiguous() {
            self.stats.contiguous_count += 1;
        } else {
            self.stats.non_contiguous_count += 1;
        }
        self.stats.total_layouts_created += 1;
        let new_id = self.next_id;
        self.next_id += 1;
        self.layouts.insert(new_id, transposed);
        Some(new_id)
    }

    /// Retrieve a reference to a layout descriptor by id.
    pub fn get(&self, layout_id: u64) -> Option<&LayoutDescriptor> {
        self.layouts.get(&layout_id)
    }

    /// Cumulative statistics for this manager.
    pub fn stats(&self) -> &LayoutStats {
        &self.stats
    }
}

impl Default for TensorMemoryLayout {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Type alias (for callers that want the MemoryLayoutShape name)
// ---------------------------------------------------------------------------

/// Alias for [`TensorShape`] used when importing alongside other crates that
/// define their own `TensorShape`.
pub type MemoryLayoutShape = TensorShape;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── TensorShape ─────────────────────────────────────────────────────────

    #[test]
    fn tensor_shape_ndim() {
        let s = TensorShape::new(vec![3, 4, 5]);
        assert_eq!(s.ndim(), 3);
    }

    #[test]
    fn tensor_shape_ndim_empty() {
        let s = TensorShape::new(vec![]);
        assert_eq!(s.ndim(), 0);
    }

    #[test]
    fn tensor_shape_total_elements_3d() {
        let s = TensorShape::new(vec![3, 4, 5]);
        assert_eq!(s.total_elements(), 60);
    }

    #[test]
    fn tensor_shape_total_elements_empty() {
        let s = TensorShape::new(vec![]);
        assert_eq!(s.total_elements(), 1);
    }

    #[test]
    fn tensor_shape_is_scalar_empty_dims() {
        let s = TensorShape::new(vec![]);
        assert!(s.is_scalar());
    }

    #[test]
    fn tensor_shape_is_scalar_single_element() {
        let s = TensorShape::new(vec![1, 1, 1]);
        assert!(s.is_scalar());
    }

    #[test]
    fn tensor_shape_is_not_scalar_multi_element() {
        let s = TensorShape::new(vec![3, 4]);
        assert!(!s.is_scalar());
    }

    // ── Stride computation ───────────────────────────────────────────────────

    #[test]
    fn row_major_strides_3d() {
        let strides = LayoutDescriptor::row_major_strides(&[3, 4, 5]);
        assert_eq!(strides, vec![20, 5, 1]);
    }

    #[test]
    fn row_major_strides_2d() {
        let strides = LayoutDescriptor::row_major_strides(&[4, 6]);
        assert_eq!(strides, vec![6, 1]);
    }

    #[test]
    fn row_major_strides_1d() {
        let strides = LayoutDescriptor::row_major_strides(&[7]);
        assert_eq!(strides, vec![1]);
    }

    #[test]
    fn row_major_strides_empty() {
        let strides = LayoutDescriptor::row_major_strides(&[]);
        assert!(strides.is_empty());
    }

    #[test]
    fn col_major_strides_3d() {
        let strides = LayoutDescriptor::col_major_strides(&[3, 4, 5]);
        assert_eq!(strides, vec![1, 3, 12]);
    }

    #[test]
    fn col_major_strides_2d() {
        let strides = LayoutDescriptor::col_major_strides(&[4, 6]);
        assert_eq!(strides, vec![1, 4]);
    }

    #[test]
    fn col_major_strides_1d() {
        let strides = LayoutDescriptor::col_major_strides(&[5]);
        assert_eq!(strides, vec![1]);
    }

    #[test]
    fn col_major_strides_empty() {
        let strides = LayoutDescriptor::col_major_strides(&[]);
        assert!(strides.is_empty());
    }

    // ── linear_index ────────────────────────────────────────────────────────

    #[test]
    fn linear_index_row_major_corner() {
        let desc = LayoutDescriptor::new(TensorShape::new(vec![3, 4, 5]), LayoutOrder::RowMajor, 4);
        // [0,0,0] → 0
        assert_eq!(desc.linear_index(&[0, 0, 0]), Some(0));
    }

    #[test]
    fn linear_index_row_major_middle() {
        let desc = LayoutDescriptor::new(TensorShape::new(vec![3, 4, 5]), LayoutOrder::RowMajor, 4);
        // [1, 2, 3] → 1*20 + 2*5 + 3*1 = 33
        assert_eq!(desc.linear_index(&[1, 2, 3]), Some(33));
    }

    #[test]
    fn linear_index_col_major() {
        let desc = LayoutDescriptor::new(TensorShape::new(vec![3, 4, 5]), LayoutOrder::ColMajor, 8);
        // strides = [1, 3, 12]; [1, 2, 3] → 1 + 6 + 36 = 43
        assert_eq!(desc.linear_index(&[1, 2, 3]), Some(43));
    }

    #[test]
    fn linear_index_none_wrong_rank() {
        let desc = LayoutDescriptor::new(TensorShape::new(vec![3, 4, 5]), LayoutOrder::RowMajor, 4);
        assert_eq!(desc.linear_index(&[0, 0]), None);
    }

    #[test]
    fn linear_index_none_out_of_bounds() {
        let desc = LayoutDescriptor::new(TensorShape::new(vec![3, 4, 5]), LayoutOrder::RowMajor, 4);
        // dim 0 has size 3, index 3 is out of bounds
        assert_eq!(desc.linear_index(&[3, 0, 0]), None);
    }

    #[test]
    fn linear_index_none_inner_dim_oob() {
        let desc = LayoutDescriptor::new(TensorShape::new(vec![3, 4, 5]), LayoutOrder::RowMajor, 4);
        assert_eq!(desc.linear_index(&[0, 4, 0]), None);
    }

    // ── byte_offset_for ──────────────────────────────────────────────────────

    #[test]
    fn byte_offset_f32_element_size() {
        let desc = LayoutDescriptor::new(
            TensorShape::new(vec![3, 4, 5]),
            LayoutOrder::RowMajor,
            4, // f32
        );
        // linear_index([1,2,3]) = 33 → byte offset = 33 * 4 = 132
        assert_eq!(desc.byte_offset_for(&[1, 2, 3]), Some(132));
    }

    #[test]
    fn byte_offset_f64_element_size() {
        let desc = LayoutDescriptor::new(
            TensorShape::new(vec![3, 4, 5]),
            LayoutOrder::RowMajor,
            8, // f64
        );
        // 33 * 8 = 264
        assert_eq!(desc.byte_offset_for(&[1, 2, 3]), Some(264));
    }

    #[test]
    fn byte_offset_none_oob() {
        let desc = LayoutDescriptor::new(TensorShape::new(vec![3, 4, 5]), LayoutOrder::RowMajor, 4);
        assert_eq!(desc.byte_offset_for(&[3, 0, 0]), None);
    }

    // ── is_contiguous ────────────────────────────────────────────────────────

    #[test]
    fn is_contiguous_row_major_fresh() {
        let desc = LayoutDescriptor::new(TensorShape::new(vec![3, 4, 5]), LayoutOrder::RowMajor, 4);
        assert!(desc.is_contiguous());
    }

    #[test]
    fn is_contiguous_false_after_transpose() {
        let desc = LayoutDescriptor::new(TensorShape::new(vec![3, 4, 5]), LayoutOrder::RowMajor, 4);
        let t = desc.transposed();
        // strides are now [1, 5, 20]; row-major for [5,4,3] would be [12,3,1]
        assert!(!t.is_contiguous());
    }

    #[test]
    fn is_contiguous_col_major_fresh() {
        // ColMajor is NOT row-major contiguous
        let desc = LayoutDescriptor::new(TensorShape::new(vec![3, 4, 5]), LayoutOrder::ColMajor, 4);
        // strides = [1,3,12]; row-major for [3,4,5] = [20,5,1] → not equal
        assert!(!desc.is_contiguous());
    }

    // ── transposed ───────────────────────────────────────────────────────────

    #[test]
    fn transposed_reverses_dims() {
        let desc = LayoutDescriptor::new(TensorShape::new(vec![3, 4, 5]), LayoutOrder::RowMajor, 4);
        let t = desc.transposed();
        assert_eq!(t.shape.dims, vec![5, 4, 3]);
    }

    #[test]
    fn transposed_reverses_strides() {
        let desc = LayoutDescriptor::new(TensorShape::new(vec![3, 4, 5]), LayoutOrder::RowMajor, 4);
        // strides before: [20, 5, 1]
        let t = desc.transposed();
        assert_eq!(t.strides, vec![1, 5, 20]);
    }

    #[test]
    fn transposed_flips_order_row_to_col() {
        let desc = LayoutDescriptor::new(TensorShape::new(vec![3, 4, 5]), LayoutOrder::RowMajor, 4);
        assert_eq!(desc.transposed().order, LayoutOrder::ColMajor);
    }

    #[test]
    fn transposed_flips_order_col_to_row() {
        let desc = LayoutDescriptor::new(TensorShape::new(vec![3, 4, 5]), LayoutOrder::ColMajor, 4);
        assert_eq!(desc.transposed().order, LayoutOrder::RowMajor);
    }

    #[test]
    fn transposed_preserves_element_size() {
        let desc = LayoutDescriptor::new(TensorShape::new(vec![3, 4, 5]), LayoutOrder::RowMajor, 8);
        assert_eq!(desc.transposed().element_size_bytes, 8);
    }

    // ── total_bytes ───────────────────────────────────────────────────────────

    #[test]
    fn total_bytes_f32() {
        let desc = LayoutDescriptor::new(TensorShape::new(vec![3, 4, 5]), LayoutOrder::RowMajor, 4);
        assert_eq!(desc.total_bytes(), 60 * 4);
    }

    #[test]
    fn total_bytes_f64() {
        let desc = LayoutDescriptor::new(TensorShape::new(vec![2, 3]), LayoutOrder::RowMajor, 8);
        assert_eq!(desc.total_bytes(), 6 * 8);
    }

    // ── TensorMemoryLayout manager ───────────────────────────────────────────

    #[test]
    fn manager_create_returns_sequential_ids() {
        let mut mgr = TensorMemoryLayout::new();
        let id0 = mgr.create(TensorShape::new(vec![2, 3]), LayoutOrder::RowMajor, 4);
        let id1 = mgr.create(TensorShape::new(vec![4]), LayoutOrder::ColMajor, 8);
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
    }

    #[test]
    fn manager_get_retrieves_descriptor() {
        let mut mgr = TensorMemoryLayout::new();
        let id = mgr.create(TensorShape::new(vec![3, 4, 5]), LayoutOrder::RowMajor, 4);
        let desc = mgr.get(id).expect("descriptor must exist");
        assert_eq!(desc.shape.dims, vec![3, 4, 5]);
    }

    #[test]
    fn manager_get_missing_id_returns_none() {
        let mgr = TensorMemoryLayout::new();
        assert!(mgr.get(99).is_none());
    }

    #[test]
    fn manager_transpose_creates_new_entry() {
        let mut mgr = TensorMemoryLayout::new();
        let id = mgr.create(TensorShape::new(vec![3, 4, 5]), LayoutOrder::RowMajor, 4);
        let tid = mgr.transpose(id).expect("transpose must succeed");
        assert_ne!(id, tid);
        let t = mgr.get(tid).expect("transposed descriptor must exist");
        assert_eq!(t.shape.dims, vec![5, 4, 3]);
    }

    #[test]
    fn manager_transpose_missing_id_returns_none() {
        let mut mgr = TensorMemoryLayout::new();
        assert!(mgr.transpose(42).is_none());
    }

    #[test]
    fn manager_stats_total_layouts_created() {
        let mut mgr = TensorMemoryLayout::new();
        mgr.create(TensorShape::new(vec![2, 2]), LayoutOrder::RowMajor, 4);
        mgr.create(TensorShape::new(vec![3]), LayoutOrder::RowMajor, 4);
        assert_eq!(mgr.stats().total_layouts_created, 2);
    }

    #[test]
    fn manager_stats_transpositions() {
        let mut mgr = TensorMemoryLayout::new();
        let id = mgr.create(TensorShape::new(vec![2, 3]), LayoutOrder::RowMajor, 4);
        mgr.transpose(id);
        mgr.transpose(id);
        assert_eq!(mgr.stats().total_transpositions, 2);
    }

    #[test]
    fn manager_stats_contiguous_count() {
        let mut mgr = TensorMemoryLayout::new();
        // Row-major is contiguous
        mgr.create(TensorShape::new(vec![2, 3]), LayoutOrder::RowMajor, 4);
        assert_eq!(mgr.stats().contiguous_count, 1);
        assert_eq!(mgr.stats().non_contiguous_count, 0);
    }

    #[test]
    fn manager_stats_non_contiguous_count() {
        let mut mgr = TensorMemoryLayout::new();
        // Col-major is not row-major contiguous
        mgr.create(TensorShape::new(vec![2, 3]), LayoutOrder::ColMajor, 4);
        assert_eq!(mgr.stats().non_contiguous_count, 1);
        assert_eq!(mgr.stats().contiguous_count, 0);
    }

    #[test]
    fn manager_default_is_empty() {
        let mgr = TensorMemoryLayout::default();
        assert_eq!(mgr.stats().total_layouts_created, 0);
        assert!(mgr.get(0).is_none());
    }

    // ── MemoryLayoutShape alias ───────────────────────────────────────────────

    #[test]
    fn memory_layout_shape_alias_works() {
        let s: MemoryLayoutShape = MemoryLayoutShape::new(vec![2, 3, 4]);
        assert_eq!(s.total_elements(), 24);
    }
}
