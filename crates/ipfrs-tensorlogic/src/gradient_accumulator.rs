//! Gradient accumulator for federated learning rounds.
//!
//! Accumulates gradients from multiple federated peers across rounds,
//! applying weighted averaging and optional clipping before committing.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during gradient accumulation.
#[derive(Debug, Error, PartialEq)]
pub enum AccumulationError {
    /// Gradient vector has wrong length.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Commit called with no pending gradients.
    #[error("no gradients pending in accumulator")]
    EmptyAccumulator,

    /// All peer weights sum to zero — cannot compute a weighted average.
    #[error("weight sum is zero; cannot compute weighted average")]
    WeightSumZero,

    /// Clip value supplied to a strategy must be strictly positive.
    #[error("clip value must be positive")]
    ClipValueNonPositive,
}

// ---------------------------------------------------------------------------
// PeerGradient
// ---------------------------------------------------------------------------

/// A gradient vector contributed by a single federated peer.
#[derive(Debug, Clone)]
pub struct PeerGradient {
    /// Unique identifier for the contributing peer.
    pub peer_id: String,
    /// The raw gradient values.
    pub gradient: Vec<f32>,
    /// Contribution weight, e.g. `local_samples / total_samples`.
    pub weight: f64,
    /// Federated round in which this gradient was produced.
    pub round: u64,
}

// ---------------------------------------------------------------------------
// ClipStrategy
// ---------------------------------------------------------------------------

/// Strategy used to clip / clamp gradients before committing.
#[derive(Debug, Clone, PartialEq)]
pub enum ClipStrategy {
    /// No clipping is applied.
    None,
    /// Clip the entire gradient vector so that its L2-norm ≤ `max_norm`.
    GlobalNorm {
        /// Maximum allowed L2-norm.
        max_norm: f32,
    },
    /// Clamp every element independently to `[-max_abs, max_abs]`.
    PerElement {
        /// Maximum allowed absolute value per element.
        max_abs: f32,
    },
}

// ---------------------------------------------------------------------------
// AccumulatorStats
// ---------------------------------------------------------------------------

/// Running statistics for a [`GradientAccumulator`].
#[derive(Debug, Clone, Default)]
pub struct AccumulatorStats {
    /// Total number of peer gradients accumulated (across all commits).
    pub peers_accumulated: usize,
    /// Number of times [`GradientAccumulator::commit`] has been called successfully.
    pub total_rounds: u64,
    /// Weight sum used in the most recent commit.
    pub last_weight_sum: f64,
    /// Total number of gradient vectors that were clipped during commit.
    pub clipped_count: u64,
}

// ---------------------------------------------------------------------------
// GradientAccumulator
// ---------------------------------------------------------------------------

/// Accumulates gradients from multiple federated peers and produces a
/// weighted-averaged, optionally-clipped aggregate on each commit.
#[derive(Debug)]
pub struct GradientAccumulator {
    /// Expected length of every gradient vector.
    pub dimension: usize,
    /// Clipping strategy applied after averaging.
    pub clip_strategy: ClipStrategy,
    /// Gradients waiting to be committed.
    pub pending: Vec<PeerGradient>,
    /// Running statistics updated on each successful commit.
    pub stats: AccumulatorStats,
}

impl GradientAccumulator {
    /// Create a new accumulator expecting gradients of length `dimension`.
    pub fn new(dimension: usize, clip_strategy: ClipStrategy) -> Self {
        Self {
            dimension,
            clip_strategy,
            pending: Vec::new(),
            stats: AccumulatorStats::default(),
        }
    }

    /// Add a peer gradient to the pending queue.
    ///
    /// Returns [`AccumulationError::DimensionMismatch`] when
    /// `pg.gradient.len() != self.dimension`.
    pub fn add(&mut self, pg: PeerGradient) -> Result<(), AccumulationError> {
        if pg.gradient.len() != self.dimension {
            return Err(AccumulationError::DimensionMismatch {
                expected: self.dimension,
                actual: pg.gradient.len(),
            });
        }
        self.pending.push(pg);
        Ok(())
    }

    /// Commit all pending gradients, producing a weighted average.
    ///
    /// After a successful commit the pending queue is cleared and
    /// [`AccumulatorStats`] is updated.
    ///
    /// # Errors
    ///
    /// - [`AccumulationError::EmptyAccumulator`] — no pending gradients.
    /// - [`AccumulationError::WeightSumZero`] — all weights are `0.0`.
    pub fn commit(&mut self) -> Result<Vec<f32>, AccumulationError> {
        if self.pending.is_empty() {
            return Err(AccumulationError::EmptyAccumulator);
        }

        let weight_sum: f64 = self.pending.iter().map(|pg| pg.weight).sum();
        if weight_sum == 0.0 {
            return Err(AccumulationError::WeightSumZero);
        }

        // Compute weighted average.
        let mut result = vec![0.0_f32; self.dimension];
        for pg in &self.pending {
            let scale = pg.weight / weight_sum;
            for (r, g) in result.iter_mut().zip(pg.gradient.iter()) {
                *r += ((*g as f64) * scale) as f32;
            }
        }

        // Apply clipping and track whether it happened.
        let was_clipped = self.apply_clip(&mut result);

        // Update stats.
        self.stats.total_rounds += 1;
        self.stats.peers_accumulated += self.pending.len();
        self.stats.last_weight_sum = weight_sum;
        if was_clipped {
            self.stats.clipped_count += 1;
        }

        self.pending.clear();
        Ok(result)
    }

    /// Apply the accumulator's [`ClipStrategy`] to `grad` in-place.
    ///
    /// Returns `true` if any modification was made.
    pub fn apply_clip(&self, grad: &mut [f32]) -> bool {
        match &self.clip_strategy {
            ClipStrategy::None => false,

            ClipStrategy::GlobalNorm { max_norm } => {
                let norm_sq: f32 = grad.iter().map(|x| x * x).sum();
                let norm = norm_sq.sqrt();
                if norm > *max_norm {
                    let scale = max_norm / norm;
                    for x in grad.iter_mut() {
                        *x *= scale;
                    }
                    true
                } else {
                    false
                }
            }

            ClipStrategy::PerElement { max_abs } => {
                let mut clipped = false;
                for x in grad.iter_mut() {
                    let clamped = x.clamp(-max_abs, *max_abs);
                    if clamped != *x {
                        *x = clamped;
                        clipped = true;
                    }
                }
                clipped
            }
        }
    }

    /// Number of gradients currently waiting to be committed.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Borrow the current statistics.
    pub fn stats(&self) -> &AccumulatorStats {
        &self.stats
    }

    /// Clear all pending gradients and reset statistics to defaults.
    pub fn reset(&mut self) {
        self.pending.clear();
        self.stats = AccumulatorStats::default();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_peer(id: &str, gradient: Vec<f32>, weight: f64, round: u64) -> PeerGradient {
        PeerGradient {
            peer_id: id.to_string(),
            gradient,
            weight,
            round,
        }
    }

    // 1. new() sets dimension and strategy
    #[test]
    fn test_new_sets_fields() {
        let acc = GradientAccumulator::new(4, ClipStrategy::None);
        assert_eq!(acc.dimension, 4);
        assert_eq!(acc.clip_strategy, ClipStrategy::None);
        assert_eq!(acc.pending_count(), 0);
    }

    // 2. add() valid gradient succeeds
    #[test]
    fn test_add_valid_gradient() {
        let mut acc = GradientAccumulator::new(3, ClipStrategy::None);
        let pg = make_peer("a", vec![1.0, 2.0, 3.0], 1.0, 1);
        assert!(acc.add(pg).is_ok());
        assert_eq!(acc.pending_count(), 1);
    }

    // 3. add() wrong dimension returns DimensionMismatch
    #[test]
    fn test_add_wrong_dimension() {
        let mut acc = GradientAccumulator::new(3, ClipStrategy::None);
        let pg = make_peer("a", vec![1.0, 2.0], 1.0, 1);
        match acc.add(pg) {
            Err(AccumulationError::DimensionMismatch { expected, actual }) => {
                assert_eq!(expected, 3);
                assert_eq!(actual, 2);
            }
            other => panic!("expected DimensionMismatch, got {:?}", other),
        }
    }

    // 4. commit() empty returns EmptyAccumulator
    #[test]
    fn test_commit_empty() {
        let mut acc = GradientAccumulator::new(3, ClipStrategy::None);
        assert_eq!(acc.commit(), Err(AccumulationError::EmptyAccumulator));
    }

    // 5. commit() single peer returns its gradient
    #[test]
    fn test_commit_single_peer() {
        let mut acc = GradientAccumulator::new(3, ClipStrategy::None);
        acc.add(make_peer("a", vec![1.0, 2.0, 3.0], 1.0, 1))
            .expect("test: should succeed");
        let result = acc.commit().expect("test: should succeed");
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
    }

    // 6. commit() two equal-weight peers averages correctly
    #[test]
    fn test_commit_two_equal_weight_peers() {
        let mut acc = GradientAccumulator::new(2, ClipStrategy::None);
        acc.add(make_peer("a", vec![0.0, 4.0], 1.0, 1))
            .expect("test: should succeed");
        acc.add(make_peer("b", vec![2.0, 0.0], 1.0, 1))
            .expect("test: should succeed");
        let result = acc.commit().expect("test: should succeed");
        assert!(
            (result[0] - 1.0).abs() < 1e-5,
            "expected 1.0 got {}",
            result[0]
        );
        assert!(
            (result[1] - 2.0).abs() < 1e-5,
            "expected 2.0 got {}",
            result[1]
        );
    }

    // 7. commit() two unequal weights: heavier peer dominates
    #[test]
    fn test_commit_unequal_weights() {
        let mut acc = GradientAccumulator::new(1, ClipStrategy::None);
        // peer A: gradient [10.0], weight 0.9
        // peer B: gradient [0.0],  weight 0.1
        // expected: 10.0 * 0.9 + 0.0 * 0.1 = 9.0
        acc.add(make_peer("a", vec![10.0], 0.9, 1))
            .expect("test: should succeed");
        acc.add(make_peer("b", vec![0.0], 0.1, 1))
            .expect("test: should succeed");
        let result = acc.commit().expect("test: should succeed");
        assert!(
            (result[0] - 9.0).abs() < 1e-5,
            "expected 9.0 got {}",
            result[0]
        );
    }

    // 8. commit() zero weight sum returns WeightSumZero
    #[test]
    fn test_commit_zero_weight_sum() {
        let mut acc = GradientAccumulator::new(2, ClipStrategy::None);
        acc.add(make_peer("a", vec![1.0, 2.0], 0.0, 1))
            .expect("test: should succeed");
        assert_eq!(acc.commit(), Err(AccumulationError::WeightSumZero));
    }

    // 9. commit() clears pending after success
    #[test]
    fn test_commit_clears_pending() {
        let mut acc = GradientAccumulator::new(2, ClipStrategy::None);
        acc.add(make_peer("a", vec![1.0, 2.0], 1.0, 1))
            .expect("test: should succeed");
        acc.commit().expect("test: should succeed");
        assert_eq!(acc.pending_count(), 0);
    }

    // 10. apply_clip GlobalNorm: norm > max_norm clips correctly
    #[test]
    fn test_apply_clip_global_norm_clips() {
        let acc = GradientAccumulator::new(2, ClipStrategy::GlobalNorm { max_norm: 1.0 });
        let mut grad = vec![3.0_f32, 4.0_f32]; // L2 norm = 5.0
        let clipped = acc.apply_clip(&mut grad);
        assert!(clipped);
        let norm_after: f32 = grad.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm_after - 1.0).abs() < 1e-6,
            "norm_after = {}",
            norm_after
        );
        // Direction should be preserved: (3/5, 4/5)
        assert!((grad[0] - 0.6).abs() < 1e-5);
        assert!((grad[1] - 0.8).abs() < 1e-5);
    }

    // 11. apply_clip GlobalNorm: norm <= max_norm unchanged, returns false
    #[test]
    fn test_apply_clip_global_norm_no_clip() {
        let acc = GradientAccumulator::new(2, ClipStrategy::GlobalNorm { max_norm: 10.0 });
        let mut grad = vec![3.0_f32, 4.0_f32]; // norm = 5.0 < 10.0
        let clipped = acc.apply_clip(&mut grad);
        assert!(!clipped);
        assert!((grad[0] - 3.0).abs() < 1e-6);
        assert!((grad[1] - 4.0).abs() < 1e-6);
    }

    // 12. apply_clip PerElement: values clamped correctly
    #[test]
    fn test_apply_clip_per_element_clamps() {
        let acc = GradientAccumulator::new(3, ClipStrategy::PerElement { max_abs: 1.0 });
        let mut grad = vec![2.0_f32, -3.0_f32, 0.5_f32];
        let clipped = acc.apply_clip(&mut grad);
        assert!(clipped);
        assert!((grad[0] - 1.0).abs() < 1e-6);
        assert!((grad[1] - (-1.0)).abs() < 1e-6);
        assert!((grad[2] - 0.5).abs() < 1e-6);
    }

    // 13. apply_clip PerElement: within bounds unchanged
    #[test]
    fn test_apply_clip_per_element_no_clamp() {
        let acc = GradientAccumulator::new(3, ClipStrategy::PerElement { max_abs: 5.0 });
        let mut grad = vec![1.0_f32, -2.0_f32, 3.0_f32];
        let clipped = acc.apply_clip(&mut grad);
        assert!(!clipped);
        assert!((grad[0] - 1.0).abs() < 1e-6);
        assert!((grad[1] - (-2.0)).abs() < 1e-6);
        assert!((grad[2] - 3.0).abs() < 1e-6);
    }

    // 14. apply_clip None: unchanged
    #[test]
    fn test_apply_clip_none() {
        let acc = GradientAccumulator::new(2, ClipStrategy::None);
        let mut grad = vec![100.0_f32, -200.0_f32];
        let clipped = acc.apply_clip(&mut grad);
        assert!(!clipped);
        assert!((grad[0] - 100.0).abs() < 1e-6);
        assert!((grad[1] - (-200.0)).abs() < 1e-6);
    }

    // 15. stats() updated after commit
    #[test]
    fn test_stats_updated_after_commit() {
        let mut acc = GradientAccumulator::new(2, ClipStrategy::None);
        acc.add(make_peer("a", vec![1.0, 2.0], 0.6, 1))
            .expect("test: should succeed");
        acc.add(make_peer("b", vec![3.0, 4.0], 0.4, 1))
            .expect("test: should succeed");
        acc.commit().expect("test: should succeed");
        let s = acc.stats();
        assert_eq!(s.total_rounds, 1);
        assert_eq!(s.peers_accumulated, 2);
        assert!((s.last_weight_sum - 1.0).abs() < 1e-10);
    }

    // 16. pending_count() accurate
    #[test]
    fn test_pending_count() {
        let mut acc = GradientAccumulator::new(2, ClipStrategy::None);
        assert_eq!(acc.pending_count(), 0);
        acc.add(make_peer("a", vec![1.0, 2.0], 1.0, 1))
            .expect("test: should succeed");
        assert_eq!(acc.pending_count(), 1);
        acc.add(make_peer("b", vec![3.0, 4.0], 1.0, 2))
            .expect("test: should succeed");
        assert_eq!(acc.pending_count(), 2);
    }

    // 17. reset() clears pending and stats
    #[test]
    fn test_reset() {
        let mut acc = GradientAccumulator::new(2, ClipStrategy::None);
        acc.add(make_peer("a", vec![1.0, 2.0], 1.0, 1))
            .expect("test: should succeed");
        acc.commit().expect("test: should succeed");
        // Add one more but don't commit
        acc.add(make_peer("b", vec![3.0, 4.0], 1.0, 2))
            .expect("test: should succeed");
        acc.reset();
        assert_eq!(acc.pending_count(), 0);
        let s = acc.stats();
        assert_eq!(s.total_rounds, 0);
        assert_eq!(s.peers_accumulated, 0);
        assert_eq!(s.last_weight_sum, 0.0);
        assert_eq!(s.clipped_count, 0);
    }

    // Bonus: clipped_count increments when GlobalNorm clips during commit
    #[test]
    fn test_stats_clipped_count_increments() {
        let mut acc = GradientAccumulator::new(2, ClipStrategy::GlobalNorm { max_norm: 1.0 });
        acc.add(make_peer("a", vec![3.0, 4.0], 1.0, 1))
            .expect("test: should succeed"); // norm = 5 > 1 → clipped
        acc.commit().expect("test: should succeed");
        assert_eq!(acc.stats().clipped_count, 1);

        // Second commit with a small gradient — should NOT increment clipped_count
        acc.add(make_peer("b", vec![0.1, 0.1], 1.0, 2))
            .expect("test: should succeed");
        acc.commit().expect("test: should succeed");
        assert_eq!(acc.stats().clipped_count, 1);
    }
}
