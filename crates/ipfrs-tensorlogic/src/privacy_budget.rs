//! Differential privacy budget accounting for federated learning.
//!
//! This module provides:
//! - [`PrivacyBudget`]: thread-safe epsilon/delta budget with atomic accounting
//! - [`BudgetSnapshot`]: immutable point-in-time view of the budget
//! - [`RenyiAccountant`]: Rényi DP composition tracker with Gaussian mechanism support
//! - [`PerRoundBudget`]: per-round budget enforcement wrapping a [`PrivacyBudget`]
//! - [`BudgetError`]: typed errors for all failure modes
//!
//! # Example
//!
//! ```rust
//! use ipfrs_tensorlogic::privacy_budget::{PrivacyBudget, RenyiAccountant};
//!
//! // Create a total budget of ε=10.0, δ=1e-5
//! let budget = PrivacyBudget::new(10.0, 1e-5).expect("example: should succeed in docs");
//!
//! // Spend some budget
//! budget.consume(1.0, 1e-6).expect("example: should succeed in docs");
//! assert!(!budget.is_exhausted());
//!
//! // Track Rényi DP composition
//! let mut accountant = RenyiAccountant::new(10.0);
//! accountant.record_gaussian_mechanism(1.1, 1.0, 0.01);
//! let (eps, delta) = accountant.to_dp(1e-5);
//! assert!(eps > 0.0);
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

// ── BudgetError ────────────────────────────────────────────────────────────

/// Errors produced by budget operations.
#[derive(Debug, Error)]
pub enum BudgetError {
    /// Budget parameters were invalid at construction time.
    #[error("invalid budget: {reason}")]
    InvalidBudget { reason: String },

    /// An operation would exceed the epsilon budget.
    #[error("epsilon budget exceeded: need {epsilon_needed}, remaining {epsilon_remaining}")]
    BudgetExceeded {
        epsilon_needed: f64,
        epsilon_remaining: f64,
    },

    /// An operation would exceed the delta budget.
    #[error("delta budget exceeded: need {delta_needed}, remaining {delta_remaining}")]
    DeltaExceeded {
        delta_needed: f64,
        delta_remaining: f64,
    },
}

// ── BudgetSnapshot ─────────────────────────────────────────────────────────

/// Immutable point-in-time snapshot of a [`PrivacyBudget`].
#[derive(Debug, Clone)]
pub struct BudgetSnapshot {
    /// Total epsilon budget allocated.
    pub epsilon_total: f64,
    /// Total delta budget allocated.
    pub delta_total: f64,
    /// Epsilon consumed so far.
    pub epsilon_spent: f64,
    /// Delta consumed so far.
    pub delta_spent: f64,
    /// Epsilon remaining (= total − spent).
    pub epsilon_remaining: f64,
    /// Delta remaining (= total − spent).
    pub delta_remaining: f64,
    /// Number of rounds that have been committed.
    pub round_count: u64,
    /// Whether either epsilon or delta remaining is ≤ 0.
    pub is_exhausted: bool,
}

// ── PrivacyBudget ──────────────────────────────────────────────────────────

/// Thread-safe differential-privacy budget tracker.
///
/// Epsilon and delta consumption are tracked with `AtomicU64` using the
/// `f64::to_bits` / `f64::from_bits` trick, so that the struct can be
/// shared across threads without a `Mutex`.
pub struct PrivacyBudget {
    /// Hard upper bound on total epsilon expenditure.
    epsilon_total: f64,
    /// Hard upper bound on total delta expenditure.
    delta_total: f64,
    /// Atomically tracked epsilon spent (stored as `f64::to_bits`).
    epsilon_spent: AtomicU64,
    /// Atomically tracked delta spent (stored as `f64::to_bits`).
    delta_spent: AtomicU64,
    /// Number of committed rounds.
    round_count: AtomicU64,
}

impl PrivacyBudget {
    /// Create a new budget.
    ///
    /// # Errors
    /// Returns [`BudgetError::InvalidBudget`] if:
    /// - `epsilon_total` ≤ 0
    /// - `delta_total` ≤ 0
    /// - `delta_total` ≥ 1
    pub fn new(epsilon_total: f64, delta_total: f64) -> Result<Self, BudgetError> {
        if epsilon_total <= 0.0 {
            return Err(BudgetError::InvalidBudget {
                reason: format!("epsilon_total must be positive, got {epsilon_total}"),
            });
        }
        if delta_total <= 0.0 {
            return Err(BudgetError::InvalidBudget {
                reason: format!("delta_total must be positive, got {delta_total}"),
            });
        }
        if delta_total >= 1.0 {
            return Err(BudgetError::InvalidBudget {
                reason: format!("delta_total must be < 1.0, got {delta_total}"),
            });
        }
        Ok(Self {
            epsilon_total,
            delta_total,
            epsilon_spent: AtomicU64::new(0f64.to_bits()),
            delta_spent: AtomicU64::new(0f64.to_bits()),
            round_count: AtomicU64::new(0),
        })
    }

    /// Total epsilon budget (immutable).
    #[inline]
    pub fn epsilon_total(&self) -> f64 {
        self.epsilon_total
    }

    /// Total delta budget (immutable).
    #[inline]
    pub fn delta_total(&self) -> f64 {
        self.delta_total
    }

    /// Epsilon remaining (may be slightly negative under concurrent load).
    #[inline]
    pub fn epsilon_remaining(&self) -> f64 {
        let spent = f64::from_bits(self.epsilon_spent.load(Ordering::SeqCst));
        self.epsilon_total - spent
    }

    /// Delta remaining.
    #[inline]
    pub fn delta_remaining(&self) -> f64 {
        let spent = f64::from_bits(self.delta_spent.load(Ordering::SeqCst));
        self.delta_total - spent
    }

    /// Returns `true` if either epsilon or delta remaining is ≤ 0.
    #[inline]
    pub fn is_exhausted(&self) -> bool {
        self.epsilon_remaining() <= 0.0 || self.delta_remaining() <= 0.0
    }

    /// Atomically consume `epsilon` and `delta` from the budget.
    ///
    /// Uses a spin-loop via `fetch_update` to ensure the combined check-then-add
    /// is linearisable.  Both values are consumed together or not at all (epsilon
    /// is rolled back on delta failure).
    ///
    /// # Errors
    /// - [`BudgetError::BudgetExceeded`] if `epsilon` would push spent past the total.
    /// - [`BudgetError::DeltaExceeded`] if `delta` would push spent past the total.
    pub fn consume(&self, epsilon: f64, delta: f64) -> Result<(), BudgetError> {
        // Atomic add for epsilon via fetch_update spin-loop.
        let eps_result =
            self.epsilon_spent
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current_bits| {
                    let current = f64::from_bits(current_bits);
                    let new_val = current + epsilon;
                    if new_val > self.epsilon_total {
                        None // signal failure — do not update
                    } else {
                        Some(new_val.to_bits())
                    }
                });

        if eps_result.is_err() {
            let remaining = self.epsilon_remaining();
            return Err(BudgetError::BudgetExceeded {
                epsilon_needed: epsilon,
                epsilon_remaining: remaining,
            });
        }

        // Atomic add for delta via fetch_update spin-loop.
        let delta_result =
            self.delta_spent
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current_bits| {
                    let current = f64::from_bits(current_bits);
                    let new_val = current + delta;
                    if new_val > self.delta_total {
                        None
                    } else {
                        Some(new_val.to_bits())
                    }
                });

        if delta_result.is_err() {
            // Roll back the epsilon we already committed.
            self.epsilon_spent
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current_bits| {
                    let current = f64::from_bits(current_bits);
                    Some((current - epsilon).to_bits())
                })
                .ok(); // rollback is best-effort; error here would be a logic bug

            let remaining = self.delta_remaining();
            return Err(BudgetError::DeltaExceeded {
                delta_needed: delta,
                delta_remaining: remaining,
            });
        }

        Ok(())
    }

    /// Increment the round counter and return the new count.
    pub fn increment_round(&self) -> u64 {
        self.round_count.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Take an immutable snapshot of the current budget state.
    pub fn snapshot(&self) -> BudgetSnapshot {
        let epsilon_spent = f64::from_bits(self.epsilon_spent.load(Ordering::SeqCst));
        let delta_spent = f64::from_bits(self.delta_spent.load(Ordering::SeqCst));
        let epsilon_remaining = self.epsilon_total - epsilon_spent;
        let delta_remaining = self.delta_total - delta_spent;
        let round_count = self.round_count.load(Ordering::SeqCst);
        BudgetSnapshot {
            epsilon_total: self.epsilon_total,
            delta_total: self.delta_total,
            epsilon_spent,
            delta_spent,
            epsilon_remaining,
            delta_remaining,
            round_count,
            is_exhausted: epsilon_remaining <= 0.0 || delta_remaining <= 0.0,
        }
    }
}

impl std::fmt::Debug for PrivacyBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let snap = self.snapshot();
        f.debug_struct("PrivacyBudget")
            .field("epsilon_total", &snap.epsilon_total)
            .field("epsilon_spent", &snap.epsilon_spent)
            .field("delta_total", &snap.delta_total)
            .field("delta_spent", &snap.delta_spent)
            .field("round_count", &snap.round_count)
            .finish()
    }
}

// ── RenyiAccountant ────────────────────────────────────────────────────────

/// Rényi differential-privacy composition accountant.
///
/// Tracks accumulated Rényi DP epsilon across multiple mechanism applications
/// and converts to (ε, δ)-DP via the standard Rényi-to-DP conversion.
#[derive(Debug, Clone)]
pub struct RenyiAccountant {
    /// Rényi order α (must be > 1).
    alpha: f64,
    /// Accumulated Rényi epsilon.
    rdp_epsilon: f64,
    /// Number of mechanism applications recorded.
    rounds: u64,
}

impl RenyiAccountant {
    /// Create a new accountant with Rényi order `alpha` (default 10.0).
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            rdp_epsilon: 0.0,
            rounds: 0,
        }
    }

    /// Return the current Rényi order α.
    #[inline]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Return the accumulated Rényi epsilon.
    #[inline]
    pub fn rdp_epsilon(&self) -> f64 {
        self.rdp_epsilon
    }

    /// Number of mechanism applications recorded so far.
    #[inline]
    pub fn rounds(&self) -> u64 {
        self.rounds
    }

    /// Record one application of the Gaussian mechanism.
    ///
    /// The Rényi epsilon for one step of the Gaussian mechanism with sensitivity
    /// `sensitivity` and noise multiplier `noise_multiplier` at subsampling rate
    /// `sample_rate` (not used in this simplified formula) is:
    ///
    /// ```text
    /// rdp_ε += α / (2 · noise_multiplier² · sensitivity²)
    /// ```
    pub fn record_gaussian_mechanism(
        &mut self,
        noise_multiplier: f64,
        sensitivity: f64,
        _sample_rate: f64,
    ) {
        let increment = self.alpha / (2.0 * noise_multiplier.powi(2) * sensitivity.powi(2));
        self.rdp_epsilon += increment;
        self.rounds += 1;
    }

    /// Convert accumulated Rényi DP to (ε, δ)-DP.
    ///
    /// Uses the simplified conversion:
    /// ```text
    /// ε = rdp_ε + √|ln(rdp_ε) / ln(δ)|
    /// ```
    ///
    /// Returns `(epsilon, delta)`.
    pub fn to_dp(&self, delta: f64) -> (f64, f64) {
        let rdp = self.rdp_epsilon;
        let epsilon = rdp + (rdp.ln() / delta.ln()).abs().sqrt();
        (epsilon, delta)
    }

    /// Reset the accountant to its initial state (keeps α).
    pub fn reset(&mut self) {
        self.rdp_epsilon = 0.0;
        self.rounds = 0;
    }
}

impl Default for RenyiAccountant {
    fn default() -> Self {
        Self::new(10.0)
    }
}

// ── RoundGuard ─────────────────────────────────────────────────────────────

/// Token returned by [`PerRoundBudget::begin_round`].
///
/// Must be passed back to [`PerRoundBudget::commit_round`] to record actual
/// consumption for the round.
#[derive(Debug)]
pub struct RoundGuard {
    /// Monotonically increasing round identifier (1-based).
    pub round_id: u64,
}

// ── PerRoundBudget ─────────────────────────────────────────────────────────

/// Wraps a [`PrivacyBudget`] and enforces per-round epsilon/delta caps.
///
/// This prevents any single training round from consuming a disproportionate
/// share of the total budget.
#[derive(Debug)]
pub struct PerRoundBudget {
    /// Underlying total budget.
    budget: PrivacyBudget,
    /// Maximum epsilon that may be consumed in a single round.
    max_epsilon_per_round: f64,
    /// Maximum delta that may be consumed in a single round.
    max_delta_per_round: f64,
}

impl PerRoundBudget {
    /// Create a new per-round budget.
    ///
    /// # Errors
    /// Returns [`BudgetError::InvalidBudget`] if the underlying [`PrivacyBudget`]
    /// construction fails, or if per-round caps are non-positive.
    pub fn new(
        epsilon_total: f64,
        delta_total: f64,
        max_epsilon_per_round: f64,
        max_delta_per_round: f64,
    ) -> Result<Self, BudgetError> {
        if max_epsilon_per_round <= 0.0 {
            return Err(BudgetError::InvalidBudget {
                reason: format!(
                    "max_epsilon_per_round must be positive, got {max_epsilon_per_round}"
                ),
            });
        }
        if max_delta_per_round <= 0.0 {
            return Err(BudgetError::InvalidBudget {
                reason: format!("max_delta_per_round must be positive, got {max_delta_per_round}"),
            });
        }
        let budget = PrivacyBudget::new(epsilon_total, delta_total)?;
        Ok(Self {
            budget,
            max_epsilon_per_round,
            max_delta_per_round,
        })
    }

    /// Reference to the underlying total budget.
    #[inline]
    pub fn budget(&self) -> &PrivacyBudget {
        &self.budget
    }

    /// Maximum epsilon allowed per round.
    #[inline]
    pub fn max_epsilon_per_round(&self) -> f64 {
        self.max_epsilon_per_round
    }

    /// Maximum delta allowed per round.
    #[inline]
    pub fn max_delta_per_round(&self) -> f64 {
        self.max_delta_per_round
    }

    /// Begin a new round.
    ///
    /// Checks that the total budget has at least `max_epsilon_per_round` and
    /// `max_delta_per_round` remaining before issuing the guard.
    ///
    /// # Errors
    /// - [`BudgetError::BudgetExceeded`] if epsilon remaining < per-round cap.
    /// - [`BudgetError::DeltaExceeded`] if delta remaining < per-round cap.
    pub fn begin_round(&self) -> Result<RoundGuard, BudgetError> {
        let eps_rem = self.budget.epsilon_remaining();
        if eps_rem < self.max_epsilon_per_round {
            return Err(BudgetError::BudgetExceeded {
                epsilon_needed: self.max_epsilon_per_round,
                epsilon_remaining: eps_rem,
            });
        }
        let delta_rem = self.budget.delta_remaining();
        if delta_rem < self.max_delta_per_round {
            return Err(BudgetError::DeltaExceeded {
                delta_needed: self.max_delta_per_round,
                delta_remaining: delta_rem,
            });
        }
        let round_id = self.budget.increment_round();
        Ok(RoundGuard { round_id })
    }

    /// Commit the round, consuming the actual `epsilon_used` / `delta_used`.
    ///
    /// # Errors
    /// - [`BudgetError::BudgetExceeded`] if `epsilon_used` > `max_epsilon_per_round`.
    /// - [`BudgetError::DeltaExceeded`] if `delta_used` > `max_delta_per_round`.
    /// - Propagates errors from [`PrivacyBudget::consume`].
    pub fn commit_round(
        &self,
        _guard: RoundGuard,
        epsilon_used: f64,
        delta_used: f64,
    ) -> Result<(), BudgetError> {
        if epsilon_used > self.max_epsilon_per_round {
            return Err(BudgetError::BudgetExceeded {
                epsilon_needed: epsilon_used,
                epsilon_remaining: self.max_epsilon_per_round,
            });
        }
        if delta_used > self.max_delta_per_round {
            return Err(BudgetError::DeltaExceeded {
                delta_needed: delta_used,
                delta_remaining: self.max_delta_per_round,
            });
        }
        self.budget.consume(epsilon_used, delta_used)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Valid budget construction
    #[test]
    fn test_valid_budget_construction() {
        let b = PrivacyBudget::new(10.0, 1e-5).expect("test: should succeed");
        assert_eq!(b.epsilon_total(), 10.0);
        assert_eq!(b.delta_total(), 1e-5);
        assert_eq!(b.epsilon_remaining(), 10.0);
        assert_eq!(b.delta_remaining(), 1e-5);
        assert!(!b.is_exhausted());
    }

    // 2. Invalid budget — negative epsilon
    #[test]
    fn test_invalid_budget_negative_epsilon() {
        let err = PrivacyBudget::new(-1.0, 1e-5).unwrap_err();
        assert!(matches!(err, BudgetError::InvalidBudget { .. }));
    }

    // 3. Invalid budget — zero epsilon
    #[test]
    fn test_invalid_budget_zero_epsilon() {
        let err = PrivacyBudget::new(0.0, 1e-5).unwrap_err();
        assert!(matches!(err, BudgetError::InvalidBudget { .. }));
    }

    // 4. Invalid budget — delta >= 1
    #[test]
    fn test_invalid_budget_delta_ge_one() {
        let err = PrivacyBudget::new(10.0, 1.0).unwrap_err();
        assert!(matches!(err, BudgetError::InvalidBudget { .. }));
        let err2 = PrivacyBudget::new(10.0, 2.0).unwrap_err();
        assert!(matches!(err2, BudgetError::InvalidBudget { .. }));
    }

    // 5. Invalid budget — negative delta
    #[test]
    fn test_invalid_budget_negative_delta() {
        let err = PrivacyBudget::new(10.0, -1e-5).unwrap_err();
        assert!(matches!(err, BudgetError::InvalidBudget { .. }));
    }

    // 6. Consume within budget succeeds
    #[test]
    fn test_consume_within_budget() {
        let b = PrivacyBudget::new(10.0, 1e-5).expect("test: should succeed");
        b.consume(1.0, 1e-6).expect("test: should succeed");
        let snap = b.snapshot();
        assert!((snap.epsilon_spent - 1.0).abs() < 1e-12);
        assert!((snap.delta_spent - 1e-6).abs() < 1e-18);
        assert!((snap.epsilon_remaining - 9.0).abs() < 1e-12);
    }

    // 7. Consume exceeding epsilon budget fails
    #[test]
    fn test_consume_exceeds_epsilon() {
        let b = PrivacyBudget::new(1.0, 1e-5).expect("test: should succeed");
        let err = b.consume(2.0, 1e-6).unwrap_err();
        assert!(matches!(err, BudgetError::BudgetExceeded { .. }));
        // epsilon_spent should remain 0 (no partial write)
        assert_eq!(b.epsilon_remaining(), 1.0);
    }

    // 8. Consume exceeding delta budget fails
    #[test]
    fn test_consume_exceeds_delta() {
        let b = PrivacyBudget::new(10.0, 1e-5).expect("test: should succeed");
        // epsilon fine, delta too large
        let err = b.consume(1.0, 1.0).unwrap_err();
        assert!(matches!(err, BudgetError::DeltaExceeded { .. }));
        // epsilon rollback: spent should be 0
        assert!((b.epsilon_remaining() - 10.0).abs() < 1e-10);
    }

    // 9. is_exhausted triggers after full epsilon consumption
    #[test]
    fn test_is_exhausted_epsilon() {
        let b = PrivacyBudget::new(1.0, 1e-5).expect("test: should succeed");
        b.consume(1.0, 1e-6).expect("test: should succeed");
        assert!(b.is_exhausted());
    }

    // 10. Snapshot fields are consistent
    #[test]
    fn test_snapshot_consistency() {
        let b = PrivacyBudget::new(10.0, 1e-4).expect("test: should succeed");
        b.consume(3.0, 2e-5).expect("test: should succeed");
        let snap = b.snapshot();
        assert!((snap.epsilon_spent + snap.epsilon_remaining - snap.epsilon_total).abs() < 1e-10);
        assert!((snap.delta_spent + snap.delta_remaining - snap.delta_total).abs() < 1e-20);
        assert_eq!(snap.is_exhausted, b.is_exhausted());
    }

    // 11. RenyiAccountant accumulates rdp_epsilon
    #[test]
    fn test_renyi_accumulates() {
        let mut acc = RenyiAccountant::new(10.0);
        assert_eq!(acc.rdp_epsilon(), 0.0);
        acc.record_gaussian_mechanism(1.1, 1.0, 0.01);
        assert!(acc.rdp_epsilon() > 0.0);
        let before = acc.rdp_epsilon();
        acc.record_gaussian_mechanism(1.1, 1.0, 0.01);
        assert!(acc.rdp_epsilon() > before);
        assert_eq!(acc.rounds(), 2);
    }

    // 12. RenyiAccountant::to_dp returns positive values
    #[test]
    fn test_renyi_to_dp_positive() {
        let mut acc = RenyiAccountant::new(10.0);
        acc.record_gaussian_mechanism(1.1, 1.0, 0.01);
        let (eps, delta) = acc.to_dp(1e-5);
        assert!(eps > 0.0, "epsilon must be positive, got {eps}");
        assert!((delta - 1e-5).abs() < 1e-15);
    }

    // 13. RenyiAccountant::reset clears accumulated state
    #[test]
    fn test_renyi_reset() {
        let mut acc = RenyiAccountant::new(10.0);
        acc.record_gaussian_mechanism(1.1, 1.0, 0.01);
        acc.record_gaussian_mechanism(1.1, 1.0, 0.01);
        assert!(acc.rdp_epsilon() > 0.0);
        acc.reset();
        assert_eq!(acc.rdp_epsilon(), 0.0);
        assert_eq!(acc.rounds(), 0);
    }

    // 14. PerRoundBudget enforces per-round limit
    #[test]
    fn test_per_round_budget_limit() {
        let prb = PerRoundBudget::new(10.0, 1e-4, 1.0, 1e-5).expect("test: should succeed");
        // Trying to commit more than max_epsilon_per_round must fail
        let guard = prb.begin_round().expect("test: should succeed");
        let err = prb.commit_round(guard, 2.0, 1e-6).unwrap_err();
        assert!(matches!(err, BudgetError::BudgetExceeded { .. }));
    }

    // 15. PerRoundBudget: multiple rounds consume cumulative budget
    #[test]
    fn test_per_round_cumulative_consumption() {
        let prb = PerRoundBudget::new(3.0, 1e-4, 1.0, 1e-5).expect("test: should succeed");
        // Round 1
        let g1 = prb.begin_round().expect("test: should succeed");
        prb.commit_round(g1, 1.0, 1e-6)
            .expect("test: should succeed");
        // Round 2
        let g2 = prb.begin_round().expect("test: should succeed");
        prb.commit_round(g2, 1.0, 1e-6)
            .expect("test: should succeed");
        // Round 3
        let g3 = prb.begin_round().expect("test: should succeed");
        prb.commit_round(g3, 1.0, 1e-6)
            .expect("test: should succeed");
        // All epsilon consumed
        let snap = prb.budget().snapshot();
        assert!((snap.epsilon_spent - 3.0).abs() < 1e-10);
        assert!(snap.is_exhausted);
    }

    // 16. begin_round fails when budget is too low for a full round
    #[test]
    fn test_begin_round_insufficient_budget() {
        let prb = PerRoundBudget::new(1.5, 1e-4, 1.0, 1e-5).expect("test: should succeed");
        let g = prb.begin_round().expect("test: should succeed");
        prb.commit_round(g, 1.0, 1e-6)
            .expect("test: should succeed");
        // Remaining epsilon (0.5) < max_epsilon_per_round (1.0)
        let err = prb.begin_round().unwrap_err();
        assert!(matches!(err, BudgetError::BudgetExceeded { .. }));
    }

    // 17. BudgetSnapshot round_count tracks rounds
    #[test]
    fn test_snapshot_round_count() {
        let prb = PerRoundBudget::new(10.0, 1e-4, 1.0, 1e-5).expect("test: should succeed");
        let g1 = prb.begin_round().expect("test: should succeed");
        assert_eq!(g1.round_id, 1);
        prb.commit_round(g1, 0.5, 1e-6)
            .expect("test: should succeed");
        let g2 = prb.begin_round().expect("test: should succeed");
        assert_eq!(g2.round_id, 2);
        prb.commit_round(g2, 0.5, 1e-6)
            .expect("test: should succeed");
        let snap = prb.budget().snapshot();
        assert_eq!(snap.round_count, 2);
    }
}
