//! Static shape inference for tensor operation graphs.
//!
//! Provides [`TensorShapeInference`] which computes output shapes for common
//! tensor operations (broadcast, matmul, reshape, transpose, concat, slice)
//! following NumPy-compatible broadcasting rules.
//!
//! # Examples
//!
//! ```
//! use ipfrs_tensorlogic::shape_inference::{TensorShapeInference, TensorShape, ShapeOp, InferenceRule};
//! use std::collections::HashMap;
//!
//! let mut engine = TensorShapeInference::new();
//!
//! // Matrix multiply (3,4) x (4,5) -> (3,5)
//! let rule = InferenceRule {
//!     op: ShapeOp::MatMul,
//!     input_shapes: vec![
//!         TensorShape { dims: vec![3, 4] },
//!         TensorShape { dims: vec![4, 5] },
//!     ],
//!     params: HashMap::new(),
//! };
//! let result = engine.infer(&rule).expect("example: should succeed in docs");
//! assert_eq!(result.dims, vec![3, 5]);
//! ```

use std::collections::HashMap;

/// Shape descriptor for a tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    /// Dimension sizes (e.g. `[3, 4, 5]` for a rank-3 tensor).
    pub dims: Vec<usize>,
}

impl TensorShape {
    /// Create a new shape from dimension sizes.
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.dims.len()
    }
}

/// Supported tensor operations for shape inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeOp {
    /// Element-wise addition (or any binary broadcast-compatible op).
    Add,
    /// Matrix multiplication: (M,K) x (K,N) -> (M,N).
    MatMul,
    /// Reshape to new dimensions (total elements must match).
    Reshape,
    /// Reverse all dimensions.
    Transpose,
    /// Concatenate along an axis.
    Concat,
    /// Slice along an axis.
    Slice,
    /// Broadcast (expand) to a target shape.
    Broadcast,
}

/// An inference rule: an operation together with its input shapes and
/// optional parameters.
#[derive(Debug, Clone)]
pub struct InferenceRule {
    /// The tensor operation.
    pub op: ShapeOp,
    /// Input tensor shapes.
    pub input_shapes: Vec<TensorShape>,
    /// Operation-specific parameters.
    ///
    /// * `Reshape`: keys `"dim0"`, `"dim1"`, ... for target dimensions, plus
    ///   `"ndims"` for the number of target dimensions.
    /// * `Concat`: key `"axis"`.
    /// * `Slice`: keys `"axis"`, `"start"`, `"end"`.
    /// * `Broadcast`: keys `"dim0"`, `"dim1"`, ... for the target shape, plus
    ///   `"ndims"`.
    pub params: HashMap<String, usize>,
}

/// Statistics collected by [`TensorShapeInference`].
#[derive(Debug, Clone)]
pub struct ShapeInferenceStats {
    /// Number of rules successfully applied.
    pub rules_applied: u64,
    /// Number of inference errors encountered.
    pub errors: u64,
}

/// Static shape inference engine for tensor operation graphs.
///
/// Tracks how many rules have been applied and how many errors occurred.
pub struct TensorShapeInference {
    rules_applied: u64,
    errors: u64,
}

impl TensorShapeInference {
    /// Create a new inference engine.
    pub fn new() -> Self {
        Self {
            rules_applied: 0,
            errors: 0,
        }
    }

    /// Infer the output shape for a given inference rule.
    pub fn infer(&mut self, rule: &InferenceRule) -> Result<TensorShape, String> {
        let result = match rule.op {
            ShapeOp::Add => {
                if rule.input_shapes.len() < 2 {
                    return Err(self.record_error("Add requires at least 2 inputs".to_string()));
                }
                Self::broadcast_shape(&rule.input_shapes[0], &rule.input_shapes[1])
            }
            ShapeOp::MatMul => {
                if rule.input_shapes.len() < 2 {
                    return Err(self.record_error("MatMul requires 2 inputs".to_string()));
                }
                Self::matmul_shape(&rule.input_shapes[0], &rule.input_shapes[1])
            }
            ShapeOp::Reshape => {
                if rule.input_shapes.is_empty() {
                    return Err(self.record_error("Reshape requires 1 input".to_string()));
                }
                let new_dims = Self::extract_dims(&rule.params)?;
                Self::reshape_shape(&rule.input_shapes[0], &new_dims)
            }
            ShapeOp::Transpose => {
                if rule.input_shapes.is_empty() {
                    return Err(self.record_error("Transpose requires 1 input".to_string()));
                }
                Ok(Self::transpose_shape(&rule.input_shapes[0]))
            }
            ShapeOp::Concat => {
                if rule.input_shapes.is_empty() {
                    return Err(self.record_error("Concat requires at least 1 input".to_string()));
                }
                let axis = *rule
                    .params
                    .get("axis")
                    .ok_or_else(|| "Concat requires 'axis' parameter".to_string())?;
                Self::concat_shape(&rule.input_shapes, axis)
            }
            ShapeOp::Slice => {
                if rule.input_shapes.is_empty() {
                    return Err(self.record_error("Slice requires 1 input".to_string()));
                }
                let axis = *rule
                    .params
                    .get("axis")
                    .ok_or_else(|| "Slice requires 'axis' parameter".to_string())?;
                let start = *rule
                    .params
                    .get("start")
                    .ok_or_else(|| "Slice requires 'start' parameter".to_string())?;
                let end = *rule
                    .params
                    .get("end")
                    .ok_or_else(|| "Slice requires 'end' parameter".to_string())?;
                Self::slice_shape(&rule.input_shapes[0], axis, start, end)
            }
            ShapeOp::Broadcast => {
                if rule.input_shapes.is_empty() {
                    return Err(self.record_error("Broadcast requires 1 input".to_string()));
                }
                let target_dims = Self::extract_dims(&rule.params)?;
                let target = TensorShape::new(target_dims);
                Self::broadcast_shape(&rule.input_shapes[0], &target)
            }
        };

        match result {
            Ok(shape) => {
                self.rules_applied += 1;
                Ok(shape)
            }
            Err(e) => Err(self.record_error(e)),
        }
    }

    /// Compute the broadcast-compatible output shape of two tensors following
    /// NumPy broadcasting rules.
    ///
    /// Rules:
    /// 1. Shapes are right-aligned.
    /// 2. Dimensions are compatible when they are equal, or one of them is 1.
    /// 3. The output dimension is the maximum of the two.
    pub fn broadcast_shape(a: &TensorShape, b: &TensorShape) -> Result<TensorShape, String> {
        let max_rank = a.rank().max(b.rank());
        let mut result_dims = Vec::with_capacity(max_rank);

        for i in 0..max_rank {
            // Right-align: index from the end.
            let da = if i < a.rank() {
                a.dims[a.rank() - 1 - i]
            } else {
                1
            };
            let db = if i < b.rank() {
                b.dims[b.rank() - 1 - i]
            } else {
                1
            };

            if da == db {
                result_dims.push(da);
            } else if da == 1 {
                result_dims.push(db);
            } else if db == 1 {
                result_dims.push(da);
            } else {
                return Err(format!(
                    "Shapes are not broadcast-compatible: {:?} vs {:?} (dimension {} from right: {} vs {})",
                    a.dims, b.dims, i, da, db
                ));
            }
        }

        result_dims.reverse();
        Ok(TensorShape::new(result_dims))
    }

    /// Compute the output shape of a matrix multiplication.
    ///
    /// For 2-D inputs `(M, K)` and `(K, N)`, the output is `(M, N)`.
    /// For higher-rank inputs the batch dimensions must be broadcast-compatible
    /// and the last two dimensions follow the matrix multiplication rule.
    pub fn matmul_shape(a: &TensorShape, b: &TensorShape) -> Result<TensorShape, String> {
        if a.rank() < 2 || b.rank() < 2 {
            return Err(format!(
                "MatMul requires at least 2-D tensors, got ranks {} and {}",
                a.rank(),
                b.rank()
            ));
        }

        let a_rows = a.dims[a.rank() - 2];
        let a_cols = a.dims[a.rank() - 1];
        let b_rows = b.dims[b.rank() - 2];
        let b_cols = b.dims[b.rank() - 1];

        if a_cols != b_rows {
            return Err(format!(
                "MatMul inner dimensions mismatch: {} vs {}",
                a_cols, b_rows
            ));
        }

        // Broadcast batch dimensions.
        let a_batch = TensorShape::new(a.dims[..a.rank() - 2].to_vec());
        let b_batch = TensorShape::new(b.dims[..b.rank() - 2].to_vec());
        let batch = Self::broadcast_shape(&a_batch, &b_batch)?;

        let mut result_dims = batch.dims;
        result_dims.push(a_rows);
        result_dims.push(b_cols);
        Ok(TensorShape::new(result_dims))
    }

    /// Verify that `new_dims` has the same total number of elements as `input`
    /// and return the reshaped tensor shape.
    pub fn reshape_shape(input: &TensorShape, new_dims: &[usize]) -> Result<TensorShape, String> {
        let input_elems = Self::total_elements(input);
        let output_elems: usize = new_dims.iter().product();

        if input_elems != output_elems {
            return Err(format!(
                "Reshape: total elements mismatch ({} vs {})",
                input_elems, output_elems
            ));
        }

        Ok(TensorShape::new(new_dims.to_vec()))
    }

    /// Return the shape with dimensions reversed (general transpose).
    pub fn transpose_shape(input: &TensorShape) -> TensorShape {
        let mut dims = input.dims.clone();
        dims.reverse();
        TensorShape::new(dims)
    }

    /// Concatenate `inputs` along `axis`. All dimensions except `axis` must
    /// match across all inputs.
    pub fn concat_shape(inputs: &[TensorShape], axis: usize) -> Result<TensorShape, String> {
        if inputs.is_empty() {
            return Err("Concat requires at least 1 input".to_string());
        }

        let rank = inputs[0].rank();
        if axis >= rank {
            return Err(format!(
                "Concat axis {} is out of bounds for rank {}",
                axis, rank
            ));
        }

        let mut concat_dim = 0usize;
        for (i, shape) in inputs.iter().enumerate() {
            if shape.rank() != rank {
                return Err(format!(
                    "Concat: all inputs must have the same rank, input 0 has rank {} but input {} has rank {}",
                    rank, i, shape.rank()
                ));
            }
            for d in 0..rank {
                if d != axis && shape.dims[d] != inputs[0].dims[d] {
                    return Err(format!(
                        "Concat: dimension {} mismatch between input 0 ({}) and input {} ({})",
                        d, inputs[0].dims[d], i, shape.dims[d]
                    ));
                }
            }
            concat_dim = concat_dim
                .checked_add(shape.dims[axis])
                .ok_or_else(|| "Concat: dimension overflow".to_string())?;
        }

        let mut result_dims = inputs[0].dims.clone();
        result_dims[axis] = concat_dim;
        Ok(TensorShape::new(result_dims))
    }

    /// Slice `input` along `axis` from `start` (inclusive) to `end`
    /// (exclusive).
    pub fn slice_shape(
        input: &TensorShape,
        axis: usize,
        start: usize,
        end: usize,
    ) -> Result<TensorShape, String> {
        if axis >= input.rank() {
            return Err(format!(
                "Slice axis {} is out of bounds for rank {}",
                axis,
                input.rank()
            ));
        }

        if start > end {
            return Err(format!(
                "Slice: start ({}) must not exceed end ({})",
                start, end
            ));
        }

        if end > input.dims[axis] {
            return Err(format!(
                "Slice: end ({}) exceeds dimension size ({}) on axis {}",
                end, input.dims[axis], axis
            ));
        }

        let mut result_dims = input.dims.clone();
        result_dims[axis] = end - start;
        Ok(TensorShape::new(result_dims))
    }

    /// Total number of elements in a shape (product of dimensions).
    /// Returns 1 for a scalar (empty dims).
    pub fn total_elements(shape: &TensorShape) -> usize {
        if shape.dims.is_empty() {
            return 1;
        }
        shape.dims.iter().product()
    }

    /// Whether the shape represents a scalar: either empty dims or every
    /// dimension is 1.
    pub fn is_scalar(shape: &TensorShape) -> bool {
        shape.dims.is_empty() || shape.dims.iter().all(|&d| d == 1)
    }

    /// Return collected statistics.
    pub fn stats(&self) -> ShapeInferenceStats {
        ShapeInferenceStats {
            rules_applied: self.rules_applied,
            errors: self.errors,
        }
    }

    // ---- helpers ----

    /// Record an error, bump the counter, and return the message.
    fn record_error(&mut self, msg: String) -> String {
        self.errors += 1;
        msg
    }

    /// Extract ordered dimension list from params map.
    /// Expects keys `"ndims"` and `"dim0"`, `"dim1"`, ...
    fn extract_dims(params: &HashMap<String, usize>) -> Result<Vec<usize>, String> {
        let ndims = *params
            .get("ndims")
            .ok_or_else(|| "Missing 'ndims' parameter".to_string())?;
        let mut dims = Vec::with_capacity(ndims);
        for i in 0..ndims {
            let key = format!("dim{}", i);
            let d = *params
                .get(&key)
                .ok_or_else(|| format!("Missing '{}' parameter", key))?;
            dims.push(d);
        }
        Ok(dims)
    }
}

impl Default for TensorShapeInference {
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

    fn shape(dims: &[usize]) -> TensorShape {
        TensorShape::new(dims.to_vec())
    }

    fn make_params(entries: &[(&str, usize)]) -> HashMap<String, usize> {
        entries.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    // ---- broadcast tests ----

    #[test]
    fn broadcast_same_shape() {
        let a = shape(&[3, 4, 5]);
        let b = shape(&[3, 4, 5]);
        let r = TensorShapeInference::broadcast_shape(&a, &b);
        assert_eq!(r.as_ref().map(|s| &s.dims), Ok(&vec![3, 4, 5]));
    }

    #[test]
    fn broadcast_scalar_and_tensor() {
        let scalar = shape(&[]);
        let tensor = shape(&[2, 3]);
        let r = TensorShapeInference::broadcast_shape(&scalar, &tensor);
        assert_eq!(r.as_ref().map(|s| &s.dims), Ok(&vec![2, 3]));
    }

    #[test]
    fn broadcast_scalar_one_and_tensor() {
        let scalar = shape(&[1]);
        let tensor = shape(&[5, 3]);
        let r = TensorShapeInference::broadcast_shape(&scalar, &tensor);
        assert_eq!(r.as_ref().map(|s| &s.dims), Ok(&vec![5, 3]));
    }

    #[test]
    fn broadcast_different_ranks() {
        let a = shape(&[3, 1]);
        let b = shape(&[2, 3, 4]);
        let r = TensorShapeInference::broadcast_shape(&a, &b);
        assert_eq!(r.as_ref().map(|s| &s.dims), Ok(&vec![2, 3, 4]));
    }

    #[test]
    fn broadcast_incompatible() {
        let a = shape(&[3]);
        let b = shape(&[4]);
        let r = TensorShapeInference::broadcast_shape(&a, &b);
        assert!(r.is_err());
    }

    #[test]
    fn broadcast_ones_expansion() {
        let a = shape(&[1, 4]);
        let b = shape(&[3, 1]);
        let r = TensorShapeInference::broadcast_shape(&a, &b);
        assert_eq!(r.as_ref().map(|s| &s.dims), Ok(&vec![3, 4]));
    }

    #[test]
    fn broadcast_high_rank() {
        let a = shape(&[1, 1, 5]);
        let b = shape(&[8, 1, 1]);
        let r = TensorShapeInference::broadcast_shape(&a, &b);
        assert_eq!(r.as_ref().map(|s| &s.dims), Ok(&vec![8, 1, 5]));
    }

    // ---- matmul tests ----

    #[test]
    fn matmul_valid_2d() {
        let a = shape(&[3, 4]);
        let b = shape(&[4, 5]);
        let r = TensorShapeInference::matmul_shape(&a, &b);
        assert_eq!(r.as_ref().map(|s| &s.dims), Ok(&vec![3, 5]));
    }

    #[test]
    fn matmul_inner_dim_mismatch() {
        let a = shape(&[3, 4]);
        let b = shape(&[5, 6]);
        let r = TensorShapeInference::matmul_shape(&a, &b);
        assert!(r.is_err());
    }

    #[test]
    fn matmul_1d_rejected() {
        let a = shape(&[4]);
        let b = shape(&[4, 3]);
        let r = TensorShapeInference::matmul_shape(&a, &b);
        assert!(r.is_err());
    }

    #[test]
    fn matmul_batched() {
        let a = shape(&[2, 3, 4]);
        let b = shape(&[2, 4, 5]);
        let r = TensorShapeInference::matmul_shape(&a, &b);
        assert_eq!(r.as_ref().map(|s| &s.dims), Ok(&vec![2, 3, 5]));
    }

    #[test]
    fn matmul_batch_broadcast() {
        let a = shape(&[1, 3, 4]);
        let b = shape(&[5, 4, 2]);
        let r = TensorShapeInference::matmul_shape(&a, &b);
        assert_eq!(r.as_ref().map(|s| &s.dims), Ok(&vec![5, 3, 2]));
    }

    // ---- reshape tests ----

    #[test]
    fn reshape_valid() {
        let input = shape(&[2, 3, 4]);
        let r = TensorShapeInference::reshape_shape(&input, &[6, 4]);
        assert_eq!(r.as_ref().map(|s| &s.dims), Ok(&vec![6, 4]));
    }

    #[test]
    fn reshape_element_mismatch() {
        let input = shape(&[2, 3]);
        let r = TensorShapeInference::reshape_shape(&input, &[7]);
        assert!(r.is_err());
    }

    #[test]
    fn reshape_to_flat() {
        let input = shape(&[3, 4, 5]);
        let r = TensorShapeInference::reshape_shape(&input, &[60]);
        assert_eq!(r.as_ref().map(|s| &s.dims), Ok(&vec![60]));
    }

    // ---- transpose tests ----

    #[test]
    fn transpose_2d() {
        let input = shape(&[3, 4]);
        let r = TensorShapeInference::transpose_shape(&input);
        assert_eq!(r.dims, vec![4, 3]);
    }

    #[test]
    fn transpose_3d() {
        let input = shape(&[2, 3, 4]);
        let r = TensorShapeInference::transpose_shape(&input);
        assert_eq!(r.dims, vec![4, 3, 2]);
    }

    #[test]
    fn transpose_scalar() {
        let input = shape(&[]);
        let r = TensorShapeInference::transpose_shape(&input);
        assert_eq!(r.dims, Vec::<usize>::new());
    }

    // ---- concat tests ----

    #[test]
    fn concat_axis0() {
        let a = shape(&[2, 3]);
        let b = shape(&[4, 3]);
        let r = TensorShapeInference::concat_shape(&[a, b], 0);
        assert_eq!(r.as_ref().map(|s| &s.dims), Ok(&vec![6, 3]));
    }

    #[test]
    fn concat_axis1() {
        let a = shape(&[2, 3]);
        let b = shape(&[2, 5]);
        let r = TensorShapeInference::concat_shape(&[a, b], 1);
        assert_eq!(r.as_ref().map(|s| &s.dims), Ok(&vec![2, 8]));
    }

    #[test]
    fn concat_three_inputs() {
        let a = shape(&[1, 4]);
        let b = shape(&[2, 4]);
        let c = shape(&[3, 4]);
        let r = TensorShapeInference::concat_shape(&[a, b, c], 0);
        assert_eq!(r.as_ref().map(|s| &s.dims), Ok(&vec![6, 4]));
    }

    #[test]
    fn concat_dim_mismatch() {
        let a = shape(&[2, 3]);
        let b = shape(&[2, 4]);
        let r = TensorShapeInference::concat_shape(&[a, b], 0);
        assert!(r.is_err());
    }

    #[test]
    fn concat_axis_out_of_bounds() {
        let a = shape(&[2, 3]);
        let r = TensorShapeInference::concat_shape(&[a], 5);
        assert!(r.is_err());
    }

    // ---- slice tests ----

    #[test]
    fn slice_basic() {
        let input = shape(&[10, 5]);
        let r = TensorShapeInference::slice_shape(&input, 0, 2, 7);
        assert_eq!(r.as_ref().map(|s| &s.dims), Ok(&vec![5, 5]));
    }

    #[test]
    fn slice_axis1() {
        let input = shape(&[4, 8]);
        let r = TensorShapeInference::slice_shape(&input, 1, 1, 5);
        assert_eq!(r.as_ref().map(|s| &s.dims), Ok(&vec![4, 4]));
    }

    #[test]
    fn slice_out_of_bounds() {
        let input = shape(&[5, 3]);
        let r = TensorShapeInference::slice_shape(&input, 0, 2, 10);
        assert!(r.is_err());
    }

    #[test]
    fn slice_start_exceeds_end() {
        let input = shape(&[5, 3]);
        let r = TensorShapeInference::slice_shape(&input, 0, 4, 2);
        assert!(r.is_err());
    }

    #[test]
    fn slice_empty_result() {
        let input = shape(&[5, 3]);
        let r = TensorShapeInference::slice_shape(&input, 0, 3, 3);
        assert_eq!(r.as_ref().map(|s| &s.dims), Ok(&vec![0, 3]));
    }

    // ---- total_elements tests ----

    #[test]
    fn total_elements_normal() {
        assert_eq!(TensorShapeInference::total_elements(&shape(&[2, 3, 4])), 24);
    }

    #[test]
    fn total_elements_scalar() {
        assert_eq!(TensorShapeInference::total_elements(&shape(&[])), 1);
    }

    #[test]
    fn total_elements_with_one() {
        assert_eq!(TensorShapeInference::total_elements(&shape(&[1, 1, 1])), 1);
    }

    // ---- is_scalar tests ----

    #[test]
    fn is_scalar_empty() {
        assert!(TensorShapeInference::is_scalar(&shape(&[])));
    }

    #[test]
    fn is_scalar_all_ones() {
        assert!(TensorShapeInference::is_scalar(&shape(&[1, 1, 1])));
    }

    #[test]
    fn is_scalar_not() {
        assert!(!TensorShapeInference::is_scalar(&shape(&[2, 1])));
    }

    // ---- infer (dispatch) tests ----

    #[test]
    fn infer_add() {
        let mut engine = TensorShapeInference::new();
        let rule = InferenceRule {
            op: ShapeOp::Add,
            input_shapes: vec![shape(&[3, 1]), shape(&[1, 4])],
            params: HashMap::new(),
        };
        let r = engine.infer(&rule);
        assert_eq!(r.as_ref().map(|s| &s.dims), Ok(&vec![3, 4]));
        assert_eq!(engine.stats().rules_applied, 1);
    }

    #[test]
    fn infer_reshape() {
        let mut engine = TensorShapeInference::new();
        let rule = InferenceRule {
            op: ShapeOp::Reshape,
            input_shapes: vec![shape(&[2, 6])],
            params: make_params(&[("ndims", 3), ("dim0", 3), ("dim1", 2), ("dim2", 2)]),
        };
        let r = engine.infer(&rule);
        assert_eq!(r.as_ref().map(|s| &s.dims), Ok(&vec![3, 2, 2]));
    }

    #[test]
    fn infer_concat_via_rule() {
        let mut engine = TensorShapeInference::new();
        let rule = InferenceRule {
            op: ShapeOp::Concat,
            input_shapes: vec![shape(&[2, 3]), shape(&[2, 4])],
            params: make_params(&[("axis", 1)]),
        };
        let r = engine.infer(&rule);
        assert_eq!(r.as_ref().map(|s| &s.dims), Ok(&vec![2, 7]));
    }

    #[test]
    fn infer_slice_via_rule() {
        let mut engine = TensorShapeInference::new();
        let rule = InferenceRule {
            op: ShapeOp::Slice,
            input_shapes: vec![shape(&[8, 3])],
            params: make_params(&[("axis", 0), ("start", 1), ("end", 5)]),
        };
        let r = engine.infer(&rule);
        assert_eq!(r.as_ref().map(|s| &s.dims), Ok(&vec![4, 3]));
    }

    #[test]
    fn infer_error_tracking() {
        let mut engine = TensorShapeInference::new();
        let rule = InferenceRule {
            op: ShapeOp::Add,
            input_shapes: vec![shape(&[3]), shape(&[4])],
            params: HashMap::new(),
        };
        assert!(engine.infer(&rule).is_err());
        assert_eq!(engine.stats().errors, 1);
        assert_eq!(engine.stats().rules_applied, 0);
    }

    // ---- chain of operations ----

    #[test]
    fn chain_matmul_then_transpose() {
        let mut engine = TensorShapeInference::new();
        let matmul_rule = InferenceRule {
            op: ShapeOp::MatMul,
            input_shapes: vec![shape(&[3, 4]), shape(&[4, 5])],
            params: HashMap::new(),
        };
        let intermediate = engine.infer(&matmul_rule).expect("matmul should succeed");
        assert_eq!(intermediate.dims, vec![3, 5]);

        let transpose_rule = InferenceRule {
            op: ShapeOp::Transpose,
            input_shapes: vec![intermediate],
            params: HashMap::new(),
        };
        let result = engine
            .infer(&transpose_rule)
            .expect("transpose should succeed");
        assert_eq!(result.dims, vec![5, 3]);
        assert_eq!(engine.stats().rules_applied, 2);
    }

    #[test]
    fn chain_concat_then_reshape() {
        let mut engine = TensorShapeInference::new();

        // Concat two (2,3) along axis 0 -> (4,3)
        let concat_rule = InferenceRule {
            op: ShapeOp::Concat,
            input_shapes: vec![shape(&[2, 3]), shape(&[2, 3])],
            params: make_params(&[("axis", 0)]),
        };
        let after_concat = engine.infer(&concat_rule).expect("concat should succeed");
        assert_eq!(after_concat.dims, vec![4, 3]);

        // Reshape (4,3) -> (2,6)
        let reshape_rule = InferenceRule {
            op: ShapeOp::Reshape,
            input_shapes: vec![after_concat],
            params: make_params(&[("ndims", 2), ("dim0", 2), ("dim1", 6)]),
        };
        let result = engine.infer(&reshape_rule).expect("reshape should succeed");
        assert_eq!(result.dims, vec![2, 6]);
        assert_eq!(engine.stats().rules_applied, 2);
    }

    #[test]
    fn infer_broadcast_via_rule() {
        let mut engine = TensorShapeInference::new();
        let rule = InferenceRule {
            op: ShapeOp::Broadcast,
            input_shapes: vec![shape(&[1, 3])],
            params: make_params(&[("ndims", 2), ("dim0", 4), ("dim1", 3)]),
        };
        let r = engine.infer(&rule);
        assert_eq!(r.as_ref().map(|s| &s.dims), Ok(&vec![4, 3]));
    }
}
