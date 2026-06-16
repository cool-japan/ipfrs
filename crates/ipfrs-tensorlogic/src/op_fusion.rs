//! Tensor operation fusion — detects and fuses sequences of tensor operations
//! into optimized compound operations, reducing memory bandwidth and computation
//! overhead.
//!
//! # Overview
//!
//! Many deep-learning and inference pipelines apply long chains of element-wise
//! operations (scale, bias, activation, clamp, …) to the same tensor.  Executing
//! each operation individually forces multiple round-trips through memory.
//! `TensorOpFusion` analyses a sequence of [`TensorOp`] values and collapses
//! fuseable sub-sequences into a single [`FusedOp`], which a backend can
//! implement in one memory pass.
//!
//! ## Fusion rules (greedy, left-to-right)
//!
//! | Pattern | Result |
//! |---|---|
//! | Scale → Relu → Bias | `ScaleReluBias` |
//! | Scale → Bias | `ScaleBias` |
//! | Clamp → Normalize | `ClampNormalize` |
//! | anything else | `Passthrough` |
//! | empty input | `[Identity]` |
//!
//! Longer patterns are tried first so that the three-op rule takes priority over
//! the two-op rule.

/// A primitive tensor operation that can appear in an un-optimised pipeline.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorOp {
    /// Multiply every element by `factor`.
    Scale { factor: f64 },
    /// Add `offset` to every element.
    Bias { offset: f64 },
    /// Apply ReLU: max(0, x).
    Relu,
    /// Clamp each element to `[min, max]`.
    Clamp { min: f64, max: f64 },
    /// Divide every element by the L2-norm of the tensor.
    Normalize,
    /// Matrix-multiply marker.  Carries the output shape so callers can
    /// reason about dimensions; cannot be fused with neighbouring ops.
    MatMul { rows: usize, cols: usize },
}

/// A fused (compound) operation produced by [`TensorOpFusion::fuse`].
#[derive(Clone, Debug, PartialEq)]
pub enum FusedOp {
    /// Scale followed immediately by Bias: `y = x * scale + bias`.
    ScaleBias { scale: f64, bias: f64 },
    /// Scale, then ReLU, then Bias: `y = max(0, x * scale) + bias`.
    ScaleReluBias { scale: f64, bias: f64 },
    /// Clamp followed by L2-normalisation.
    ClampNormalize { min: f64, max: f64 },
    /// No-op — used when the input sequence was empty.
    Identity,
    /// A single operation that could not be fused with its neighbours.
    Passthrough(TensorOp),
}

/// The result of a fusion pass: the optimised op sequence plus accounting info.
#[derive(Clone, Debug)]
pub struct FusionPlan {
    /// Optimised operation sequence.
    pub ops: Vec<FusedOp>,
    /// Number of ops in the original (unoptimised) sequence.
    pub original_op_count: usize,
    /// Number of ops in the optimised sequence (== `ops.len()`).
    pub fused_op_count: usize,
}

impl FusionPlan {
    /// Fraction of operations eliminated: `(original − fused) / original`.
    ///
    /// Returns `0.0` when `original_op_count` is zero.
    pub fn reduction_ratio(&self) -> f64 {
        if self.original_op_count == 0 {
            return 0.0;
        }
        let reduced = self.original_op_count.saturating_sub(self.fused_op_count);
        reduced as f64 / self.original_op_count as f64
    }
}

/// Cumulative statistics across all [`TensorOpFusion::fuse`] calls.
#[derive(Clone, Debug, Default)]
pub struct FusionStats {
    /// How many times `fuse()` has been called.
    pub total_fusion_runs: u64,
    /// Sum of `original_op_count` across all runs.
    pub total_ops_fused: u64,
    /// Sum of `(original − fused)` across all runs.
    pub total_ops_reduced: u64,
}

/// Stateful engine that fuses primitive tensor-op sequences into compound ops.
///
/// # Example
///
/// ```
/// use ipfrs_tensorlogic::op_fusion::{TensorOp as FusionTensorOp, TensorOpFusion};
///
/// let mut engine = TensorOpFusion::new();
/// let ops = vec![
///     FusionTensorOp::Scale { factor: 2.0 },
///     FusionTensorOp::Bias  { offset: 1.0 },
/// ];
/// let plan = engine.fuse(ops);
/// assert_eq!(plan.fused_op_count, 1);
/// assert!((plan.reduction_ratio() - 0.5).abs() < 1e-10);
/// ```
#[derive(Debug, Default)]
pub struct TensorOpFusion {
    stats: FusionStats,
}

impl TensorOpFusion {
    /// Create a new, zero-stats fusion engine.
    pub fn new() -> Self {
        Self {
            stats: FusionStats::default(),
        }
    }

    /// Fuse `ops` left-to-right using greedy matching and return a [`FusionPlan`].
    pub fn fuse(&mut self, ops: Vec<TensorOp>) -> FusionPlan {
        let original_op_count = ops.len();

        let fused_ops = if ops.is_empty() {
            vec![FusedOp::Identity]
        } else {
            Self::fuse_sequence(ops)
        };

        let fused_op_count = fused_ops.len();
        let reduced = original_op_count.saturating_sub(fused_op_count);

        self.stats.total_fusion_runs += 1;
        self.stats.total_ops_fused += original_op_count as u64;
        self.stats.total_ops_reduced += reduced as u64;

        FusionPlan {
            ops: fused_ops,
            original_op_count,
            fused_op_count,
        }
    }

    /// Return a reference to the accumulated statistics.
    pub fn stats(&self) -> &FusionStats {
        &self.stats
    }

    /// Return `true` when `a` and `b` form the start of a fuseable pattern.
    ///
    /// This covers all two-op pairs that appear in any fusion rule:
    ///
    /// * Scale → Bias  (`ScaleBias`)
    /// * Scale → Relu  (start of `ScaleReluBias`)
    /// * Relu  → Bias  (tail of `ScaleReluBias`)
    /// * Clamp → Normalize (`ClampNormalize`)
    pub fn can_fuse(a: &TensorOp, b: &TensorOp) -> bool {
        matches!(
            (a, b),
            (TensorOp::Scale { .. }, TensorOp::Bias { .. })
                | (TensorOp::Scale { .. }, TensorOp::Relu)
                | (TensorOp::Relu, TensorOp::Bias { .. })
                | (TensorOp::Clamp { .. }, TensorOp::Normalize)
        )
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Core greedy fusion loop.  Assumes `ops` is non-empty.
    fn fuse_sequence(ops: Vec<TensorOp>) -> Vec<FusedOp> {
        let mut result: Vec<FusedOp> = Vec::with_capacity(ops.len());
        let mut cursor = 0usize;

        while cursor < ops.len() {
            // Try the longest (3-op) patterns first.
            if cursor + 2 < ops.len() {
                if let Some(fused) =
                    Self::try_fuse_three(&ops[cursor], &ops[cursor + 1], &ops[cursor + 2])
                {
                    result.push(fused);
                    cursor += 3;
                    continue;
                }
            }

            // Try 2-op patterns.
            if cursor + 1 < ops.len() {
                if let Some(fused) = Self::try_fuse_two(&ops[cursor], &ops[cursor + 1]) {
                    result.push(fused);
                    cursor += 2;
                    continue;
                }
            }

            // Fall back to passthrough.
            result.push(FusedOp::Passthrough(ops[cursor].clone()));
            cursor += 1;
        }

        result
    }

    /// Attempt to fuse three consecutive ops.
    fn try_fuse_three(a: &TensorOp, b: &TensorOp, c: &TensorOp) -> Option<FusedOp> {
        match (a, b, c) {
            (
                TensorOp::Scale { factor: scale },
                TensorOp::Relu,
                TensorOp::Bias { offset: bias },
            ) => Some(FusedOp::ScaleReluBias {
                scale: *scale,
                bias: *bias,
            }),
            _ => None,
        }
    }

    /// Attempt to fuse two consecutive ops.
    fn try_fuse_two(a: &TensorOp, b: &TensorOp) -> Option<FusedOp> {
        match (a, b) {
            (TensorOp::Scale { factor: scale }, TensorOp::Bias { offset: bias }) => {
                Some(FusedOp::ScaleBias {
                    scale: *scale,
                    bias: *bias,
                })
            }
            (TensorOp::Clamp { min, max }, TensorOp::Normalize) => Some(FusedOp::ClampNormalize {
                min: *min,
                max: *max,
            }),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Construction -------------------------------------------------------

    #[test]
    fn new_starts_with_zero_stats() {
        let engine = TensorOpFusion::new();
        let s = engine.stats();
        assert_eq!(s.total_fusion_runs, 0);
        assert_eq!(s.total_ops_fused, 0);
        assert_eq!(s.total_ops_reduced, 0);
    }

    // -- Empty input --------------------------------------------------------

    #[test]
    fn fuse_empty_returns_identity() {
        let mut engine = TensorOpFusion::new();
        let plan = engine.fuse(vec![]);
        assert_eq!(plan.ops, vec![FusedOp::Identity]);
    }

    #[test]
    fn fuse_empty_original_count_zero() {
        let mut engine = TensorOpFusion::new();
        let plan = engine.fuse(vec![]);
        assert_eq!(plan.original_op_count, 0);
    }

    #[test]
    fn reduction_ratio_zero_for_empty() {
        let mut engine = TensorOpFusion::new();
        let plan = engine.fuse(vec![]);
        assert!((plan.reduction_ratio() - 0.0).abs() < f64::EPSILON);
    }

    // -- Scale + Bias -------------------------------------------------------

    #[test]
    fn fuse_scale_bias_produces_scale_bias() {
        let mut engine = TensorOpFusion::new();
        let plan = engine.fuse(vec![
            TensorOp::Scale { factor: 3.0 },
            TensorOp::Bias { offset: 1.5 },
        ]);
        assert_eq!(plan.ops.len(), 1);
        assert_eq!(
            plan.ops[0],
            FusedOp::ScaleBias {
                scale: 3.0,
                bias: 1.5
            }
        );
    }

    #[test]
    fn scale_bias_correct_values() {
        let mut engine = TensorOpFusion::new();
        let plan = engine.fuse(vec![
            TensorOp::Scale { factor: 0.5 },
            TensorOp::Bias { offset: -2.0 },
        ]);
        match &plan.ops[0] {
            FusedOp::ScaleBias { scale, bias } => {
                assert!((scale - 0.5).abs() < f64::EPSILON);
                assert!((bias - (-2.0)).abs() < f64::EPSILON);
            }
            other => panic!("expected ScaleBias, got {:?}", other),
        }
    }

    // -- Scale + Relu + Bias ------------------------------------------------

    #[test]
    fn fuse_scale_relu_bias_produces_scale_relu_bias() {
        let mut engine = TensorOpFusion::new();
        let plan = engine.fuse(vec![
            TensorOp::Scale { factor: 2.0 },
            TensorOp::Relu,
            TensorOp::Bias { offset: 0.1 },
        ]);
        assert_eq!(plan.ops.len(), 1);
        assert_eq!(
            plan.ops[0],
            FusedOp::ScaleReluBias {
                scale: 2.0,
                bias: 0.1
            }
        );
    }

    #[test]
    fn scale_relu_bias_correct_values() {
        let mut engine = TensorOpFusion::new();
        let plan = engine.fuse(vec![
            TensorOp::Scale { factor: -1.0 },
            TensorOp::Relu,
            TensorOp::Bias { offset: 4.0 },
        ]);
        match &plan.ops[0] {
            FusedOp::ScaleReluBias { scale, bias } => {
                assert!((scale - (-1.0)).abs() < f64::EPSILON);
                assert!((bias - 4.0).abs() < f64::EPSILON);
            }
            other => panic!("expected ScaleReluBias, got {:?}", other),
        }
    }

    // -- Clamp + Normalize --------------------------------------------------

    #[test]
    fn fuse_clamp_normalize_produces_clamp_normalize() {
        let mut engine = TensorOpFusion::new();
        let plan = engine.fuse(vec![
            TensorOp::Clamp {
                min: -1.0,
                max: 1.0,
            },
            TensorOp::Normalize,
        ]);
        assert_eq!(plan.ops.len(), 1);
        assert_eq!(
            plan.ops[0],
            FusedOp::ClampNormalize {
                min: -1.0,
                max: 1.0
            }
        );
    }

    // -- Single ops ---------------------------------------------------------

    #[test]
    fn fuse_single_scale_is_passthrough() {
        let mut engine = TensorOpFusion::new();
        let plan = engine.fuse(vec![TensorOp::Scale { factor: 5.0 }]);
        assert_eq!(
            plan.ops,
            vec![FusedOp::Passthrough(TensorOp::Scale { factor: 5.0 })]
        );
    }

    #[test]
    fn fuse_single_relu_is_passthrough() {
        let mut engine = TensorOpFusion::new();
        let plan = engine.fuse(vec![TensorOp::Relu]);
        assert_eq!(plan.ops, vec![FusedOp::Passthrough(TensorOp::Relu)]);
    }

    #[test]
    fn fuse_single_matmul_is_passthrough() {
        let mut engine = TensorOpFusion::new();
        let plan = engine.fuse(vec![TensorOp::MatMul { rows: 4, cols: 8 }]);
        assert_eq!(
            plan.ops,
            vec![FusedOp::Passthrough(TensorOp::MatMul { rows: 4, cols: 8 })]
        );
    }

    // -- MatMul breaks chains -----------------------------------------------

    #[test]
    fn matmul_breaks_fusion_chain() {
        let mut engine = TensorOpFusion::new();
        // Scale and Bias are separated by MatMul → no ScaleBias should emerge.
        let plan = engine.fuse(vec![
            TensorOp::Scale { factor: 2.0 },
            TensorOp::MatMul { rows: 2, cols: 2 },
            TensorOp::Bias { offset: 1.0 },
        ]);
        assert_eq!(plan.ops.len(), 3);
        assert_eq!(
            plan.ops[0],
            FusedOp::Passthrough(TensorOp::Scale { factor: 2.0 })
        );
        assert_eq!(
            plan.ops[1],
            FusedOp::Passthrough(TensorOp::MatMul { rows: 2, cols: 2 })
        );
        assert_eq!(
            plan.ops[2],
            FusedOp::Passthrough(TensorOp::Bias { offset: 1.0 })
        );
    }

    // -- Non-fuseable pairs -------------------------------------------------

    #[test]
    fn bias_then_scale_produces_two_passthroughs() {
        let mut engine = TensorOpFusion::new();
        let plan = engine.fuse(vec![
            TensorOp::Bias { offset: 1.0 },
            TensorOp::Scale { factor: 2.0 },
        ]);
        assert_eq!(plan.ops.len(), 2);
        assert_eq!(
            plan.ops[0],
            FusedOp::Passthrough(TensorOp::Bias { offset: 1.0 })
        );
        assert_eq!(
            plan.ops[1],
            FusedOp::Passthrough(TensorOp::Scale { factor: 2.0 })
        );
    }

    // -- reduction_ratio ----------------------------------------------------

    #[test]
    fn reduction_ratio_scale_bias() {
        let mut engine = TensorOpFusion::new();
        let plan = engine.fuse(vec![
            TensorOp::Scale { factor: 1.0 },
            TensorOp::Bias { offset: 0.0 },
        ]);
        // 2 original → 1 fused → 50 % reduction
        let expected = 0.5;
        assert!((plan.reduction_ratio() - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn reduction_ratio_scale_relu_bias() {
        let mut engine = TensorOpFusion::new();
        let plan = engine.fuse(vec![
            TensorOp::Scale { factor: 1.0 },
            TensorOp::Relu,
            TensorOp::Bias { offset: 0.0 },
        ]);
        // 3 original → 1 fused → 66.6...% reduction
        let expected = 2.0 / 3.0;
        assert!((plan.reduction_ratio() - expected).abs() < 1e-10);
    }

    // -- fused_op_count == ops.len() ----------------------------------------

    #[test]
    fn fused_op_count_equals_ops_len() {
        let mut engine = TensorOpFusion::new();
        let plan = engine.fuse(vec![
            TensorOp::Scale { factor: 1.0 },
            TensorOp::Bias { offset: 0.0 },
            TensorOp::Relu,
        ]);
        assert_eq!(plan.fused_op_count, plan.ops.len());
    }

    // -- Stats --------------------------------------------------------------

    #[test]
    fn stats_fusion_runs_increments() {
        let mut engine = TensorOpFusion::new();
        engine.fuse(vec![TensorOp::Relu]);
        engine.fuse(vec![TensorOp::Relu]);
        assert_eq!(engine.stats().total_fusion_runs, 2);
    }

    #[test]
    fn stats_total_ops_fused_accumulates() {
        let mut engine = TensorOpFusion::new();
        engine.fuse(vec![TensorOp::Relu, TensorOp::Relu]);
        engine.fuse(vec![TensorOp::Scale { factor: 1.0 }]);
        // 2 + 1 = 3
        assert_eq!(engine.stats().total_ops_fused, 3);
    }

    #[test]
    fn stats_total_ops_reduced_correct() {
        let mut engine = TensorOpFusion::new();
        // Run 1: 2 → 1 (reduced by 1)
        engine.fuse(vec![
            TensorOp::Scale { factor: 1.0 },
            TensorOp::Bias { offset: 0.0 },
        ]);
        // Run 2: 3 → 1 (reduced by 2)
        engine.fuse(vec![
            TensorOp::Scale { factor: 2.0 },
            TensorOp::Relu,
            TensorOp::Bias { offset: 0.5 },
        ]);
        assert_eq!(engine.stats().total_ops_reduced, 3);
    }

    // -- can_fuse -----------------------------------------------------------

    #[test]
    fn can_fuse_scale_bias_true() {
        assert!(TensorOpFusion::can_fuse(
            &TensorOp::Scale { factor: 1.0 },
            &TensorOp::Bias { offset: 0.0 }
        ));
    }

    #[test]
    fn can_fuse_scale_relu_true() {
        assert!(TensorOpFusion::can_fuse(
            &TensorOp::Scale { factor: 1.0 },
            &TensorOp::Relu
        ));
    }

    #[test]
    fn can_fuse_relu_bias_true() {
        assert!(TensorOpFusion::can_fuse(
            &TensorOp::Relu,
            &TensorOp::Bias { offset: 0.0 }
        ));
    }

    #[test]
    fn can_fuse_matmul_anything_false() {
        assert!(!TensorOpFusion::can_fuse(
            &TensorOp::MatMul { rows: 2, cols: 2 },
            &TensorOp::Scale { factor: 1.0 }
        ));
        assert!(!TensorOpFusion::can_fuse(
            &TensorOp::MatMul { rows: 2, cols: 2 },
            &TensorOp::Bias { offset: 0.0 }
        ));
        assert!(!TensorOpFusion::can_fuse(
            &TensorOp::MatMul { rows: 2, cols: 2 },
            &TensorOp::Relu
        ));
        assert!(!TensorOpFusion::can_fuse(
            &TensorOp::MatMul { rows: 2, cols: 2 },
            &TensorOp::Normalize
        ));
    }

    // -- Mixed sequences ----------------------------------------------------

    #[test]
    fn mixed_sequence_fuses_correctly() {
        let mut engine = TensorOpFusion::new();
        // [Scale, Bias, Clamp, Normalize, Relu]
        // → ScaleBias, ClampNormalize, Passthrough(Relu)
        let plan = engine.fuse(vec![
            TensorOp::Scale { factor: 2.0 },
            TensorOp::Bias { offset: 1.0 },
            TensorOp::Clamp { min: 0.0, max: 5.0 },
            TensorOp::Normalize,
            TensorOp::Relu,
        ]);
        assert_eq!(plan.fused_op_count, 3);
        assert_eq!(
            plan.ops[0],
            FusedOp::ScaleBias {
                scale: 2.0,
                bias: 1.0
            }
        );
        assert_eq!(plan.ops[1], FusedOp::ClampNormalize { min: 0.0, max: 5.0 });
        assert_eq!(plan.ops[2], FusedOp::Passthrough(TensorOp::Relu));
    }
}
