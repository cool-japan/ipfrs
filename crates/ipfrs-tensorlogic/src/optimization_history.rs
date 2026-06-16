//! TensorOptimizationHistory — records and analyzes optimization steps
//! (loss/gradient values) to detect convergence, track best results, and
//! guide adaptive learning rate schedules.

// ---------------------------------------------------------------------------
// OptimizationStep
// ---------------------------------------------------------------------------

/// A single recorded step in the optimization process.
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizationStep {
    /// Monotonically increasing step index.
    pub step: u64,
    /// Loss value at this step.
    pub loss: f64,
    /// L2 norm of the gradient at this step.
    pub gradient_norm: f64,
    /// Learning rate used at this step.
    pub learning_rate: f64,
    /// Logical clock tick when this step was recorded.
    pub timestamp_tick: u64,
}

// ---------------------------------------------------------------------------
// ConvergenceStatus
// ---------------------------------------------------------------------------

/// The convergence state inferred from recent optimization history.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvergenceStatus {
    /// Optimization has not yet converged.
    NotConverged,
    /// Recent improvement is below the threshold but not sustained long enough
    /// to be certain.
    PossiblyConverged,
    /// Sustained low improvement for `patience` consecutive steps — the
    /// optimizer is considered converged.
    Converged,
}

// ---------------------------------------------------------------------------
// OptimizationHistoryConfig
// ---------------------------------------------------------------------------

/// Configuration for [`TensorOptimizationHistory`].
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizationHistoryConfig {
    /// Maximum number of history entries.  When the limit is reached the
    /// oldest entry is evicted.
    pub max_steps: usize,
    /// Number of consecutive steps with improvement < `convergence_threshold`
    /// required before declaring [`ConvergenceStatus::Converged`].
    pub convergence_patience: usize,
    /// Minimum loss improvement required to count a step as making progress.
    pub convergence_threshold: f64,
}

impl Default for OptimizationHistoryConfig {
    fn default() -> Self {
        Self {
            max_steps: 1000,
            convergence_patience: 10,
            convergence_threshold: 1e-6,
        }
    }
}

// ---------------------------------------------------------------------------
// HistoryStats
// ---------------------------------------------------------------------------

/// Aggregated statistics computed over the entire recorded history.
#[derive(Clone, Debug, PartialEq)]
pub struct HistoryStats {
    /// Total number of steps currently retained in the history buffer.
    pub total_steps: usize,
    /// The lowest loss value ever seen (across all recorded steps).
    pub best_loss: f64,
    /// The step index at which `best_loss` was achieved.
    pub best_step: u64,
    /// Loss at the most recently recorded step.
    pub current_loss: f64,
    /// Arithmetic mean of all recorded `gradient_norm` values.
    pub avg_gradient_norm: f64,
    /// Current convergence status.
    pub convergence_status: ConvergenceStatus,
}

// ---------------------------------------------------------------------------
// TensorOptimizationHistory
// ---------------------------------------------------------------------------

/// Records and analyzes a history of optimization steps.
///
/// # Convergence detection
///
/// After each [`record`](TensorOptimizationHistory::record) call the tracker
/// compares the new loss against the previous best.  If the improvement
/// (previous_best − new_loss) is below
/// [`OptimizationHistoryConfig::convergence_threshold`] the
/// `consecutive_no_progress` counter is incremented; otherwise it is reset to
/// zero.  Once the counter reaches
/// [`OptimizationHistoryConfig::convergence_patience`] the status transitions
/// to [`ConvergenceStatus::Converged`]; once it reaches `patience / 2`
/// (integer division) it transitions to
/// [`ConvergenceStatus::PossiblyConverged`].
pub struct TensorOptimizationHistory {
    /// Retained optimization steps (oldest first).
    pub steps: Vec<OptimizationStep>,
    /// Configuration.
    pub config: OptimizationHistoryConfig,
    /// Best (lowest) loss seen so far.
    pub best_loss: f64,
    /// Step index that achieved `best_loss`.
    pub best_step: u64,
    /// Number of consecutive steps that did not improve loss by at least
    /// `convergence_threshold`.
    pub consecutive_no_progress: usize,
}

impl TensorOptimizationHistory {
    /// Creates a new history tracker with the given configuration.
    pub fn new(config: OptimizationHistoryConfig) -> Self {
        Self {
            steps: Vec::new(),
            config,
            best_loss: f64::MAX,
            best_step: 0,
            consecutive_no_progress: 0,
        }
    }

    /// Records a new optimization step.
    ///
    /// If the history buffer is full the oldest entry is evicted.  Best loss
    /// tracking and consecutive-no-progress counting are updated accordingly.
    pub fn record(&mut self, step: OptimizationStep) {
        // Evict oldest entry if at capacity.
        if self.steps.len() >= self.config.max_steps {
            self.steps.remove(0);
        }

        let new_loss = step.loss;
        let prev_best = self.best_loss;

        // First record initialises best tracking.
        if prev_best == f64::MAX {
            self.best_loss = new_loss;
            self.best_step = step.step;
            self.consecutive_no_progress = 0;
            self.steps.push(step);
            return;
        }

        // Compute improvement relative to best seen so far.
        let improvement = prev_best - new_loss;

        if new_loss < self.best_loss {
            self.best_loss = new_loss;
            self.best_step = step.step;
        }

        if improvement < self.config.convergence_threshold {
            self.consecutive_no_progress += 1;
        } else {
            self.consecutive_no_progress = 0;
        }

        self.steps.push(step);
    }

    /// Returns the current [`ConvergenceStatus`] based on the consecutive
    /// no-progress counter.
    pub fn convergence_status(&self) -> ConvergenceStatus {
        if self.steps.is_empty() {
            return ConvergenceStatus::NotConverged;
        }
        let patience = self.config.convergence_patience;
        if self.consecutive_no_progress >= patience {
            ConvergenceStatus::Converged
        } else if self.consecutive_no_progress >= patience / 2 {
            ConvergenceStatus::PossiblyConverged
        } else {
            ConvergenceStatus::NotConverged
        }
    }

    /// Computes the total loss improvement over the last `n` steps.
    ///
    /// Returns `first_loss_in_window − last_loss_in_window`; a positive value
    /// means the loss is decreasing (improving).  Returns `0.0` when fewer
    /// than two steps are available.
    pub fn recent_improvement(&self, n: usize) -> f64 {
        if self.steps.len() < 2 {
            return 0.0;
        }
        let window_n = n.min(self.steps.len());
        let window_start = self.steps.len() - window_n;
        let first_loss = self.steps[window_start].loss;
        let last_loss = self.steps[self.steps.len() - 1].loss;
        first_loss - last_loss
    }

    /// Returns the arithmetic mean of all recorded `gradient_norm` values.
    ///
    /// Returns `0.0` when no steps have been recorded.
    pub fn avg_gradient_norm(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.steps.iter().map(|s| s.gradient_norm).sum();
        sum / self.steps.len() as f64
    }

    /// Computes and returns aggregated [`HistoryStats`] for the current
    /// history buffer.
    pub fn stats(&self) -> HistoryStats {
        let current_loss = self.steps.last().map(|s| s.loss).unwrap_or(f64::MAX);

        HistoryStats {
            total_steps: self.steps.len(),
            best_loss: self.best_loss,
            best_step: self.best_step,
            current_loss,
            avg_gradient_norm: self.avg_gradient_norm(),
            convergence_status: self.convergence_status(),
        }
    }

    /// Returns a reference to the most recently recorded step, or `None` if
    /// the history is empty.
    pub fn last_step(&self) -> Option<&OptimizationStep> {
        self.steps.last()
    }

    /// Resets all history and tracking state.
    pub fn reset(&mut self) {
        self.steps.clear();
        self.best_loss = f64::MAX;
        self.best_step = 0;
        self.consecutive_no_progress = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_history() -> TensorOptimizationHistory {
        TensorOptimizationHistory::new(OptimizationHistoryConfig::default())
    }

    fn make_step(step: u64, loss: f64) -> OptimizationStep {
        OptimizationStep {
            step,
            loss,
            gradient_norm: 0.5,
            learning_rate: 0.01,
            timestamp_tick: step,
        }
    }

    fn make_step_full(step: u64, loss: f64, gradient_norm: f64) -> OptimizationStep {
        OptimizationStep {
            step,
            loss,
            gradient_norm,
            learning_rate: 0.01,
            timestamp_tick: step,
        }
    }

    // -----------------------------------------------------------------------
    // 1. record adds step to history
    // -----------------------------------------------------------------------

    #[test]
    fn test_record_adds_step() {
        let mut h = default_history();
        h.record(make_step(0, 1.0));
        assert_eq!(h.steps.len(), 1);
    }

    #[test]
    fn test_record_multiple_steps() {
        let mut h = default_history();
        for i in 0..5u64 {
            h.record(make_step(i, 1.0 - i as f64 * 0.1));
        }
        assert_eq!(h.steps.len(), 5);
    }

    // -----------------------------------------------------------------------
    // 2. max_steps eviction removes oldest
    // -----------------------------------------------------------------------

    #[test]
    fn test_max_steps_eviction() {
        let config = OptimizationHistoryConfig {
            max_steps: 3,
            convergence_patience: 10,
            convergence_threshold: 1e-6,
        };
        let mut h = TensorOptimizationHistory::new(config);
        for i in 0..5u64 {
            h.record(make_step(i, 1.0 - i as f64 * 0.01));
        }
        assert_eq!(h.steps.len(), 3);
        // Oldest (step 0, 1) should have been evicted; first remaining is step 2.
        assert_eq!(h.steps[0].step, 2);
    }

    #[test]
    fn test_max_steps_boundary() {
        let config = OptimizationHistoryConfig {
            max_steps: 1,
            convergence_patience: 5,
            convergence_threshold: 1e-6,
        };
        let mut h = TensorOptimizationHistory::new(config);
        h.record(make_step(0, 2.0));
        h.record(make_step(1, 1.0));
        assert_eq!(h.steps.len(), 1);
        assert_eq!(h.steps[0].step, 1);
    }

    // -----------------------------------------------------------------------
    // 3. best_loss and best_step track minimum
    // -----------------------------------------------------------------------

    #[test]
    fn test_best_loss_tracks_minimum() {
        let mut h = default_history();
        h.record(make_step(0, 3.0));
        h.record(make_step(1, 1.0));
        h.record(make_step(2, 2.0));
        assert!((h.best_loss - 1.0).abs() < 1e-12);
        assert_eq!(h.best_step, 1);
    }

    #[test]
    fn test_best_loss_initialized_correctly() {
        let mut h = default_history();
        h.record(make_step(5, 42.0));
        assert!((h.best_loss - 42.0).abs() < 1e-12);
        assert_eq!(h.best_step, 5);
    }

    #[test]
    fn test_best_step_updates_when_loss_decreases() {
        let mut h = default_history();
        h.record(make_step(0, 10.0));
        h.record(make_step(1, 5.0));
        h.record(make_step(2, 7.0));
        h.record(make_step(3, 2.0));
        assert!((h.best_loss - 2.0).abs() < 1e-12);
        assert_eq!(h.best_step, 3);
    }

    // -----------------------------------------------------------------------
    // 4. convergence_status: NotConverged / PossiblyConverged / Converged
    // -----------------------------------------------------------------------

    #[test]
    fn test_convergence_status_empty() {
        let h = default_history();
        assert_eq!(h.convergence_status(), ConvergenceStatus::NotConverged);
    }

    #[test]
    fn test_convergence_status_not_converged() {
        let mut h = default_history();
        // Big decreasing improvements each step.
        for i in 0..5u64 {
            h.record(make_step(i, 100.0 - i as f64 * 10.0));
        }
        assert_eq!(h.convergence_status(), ConvergenceStatus::NotConverged);
    }

    #[test]
    fn test_convergence_status_converged() {
        let config = OptimizationHistoryConfig {
            max_steps: 1000,
            convergence_patience: 5,
            convergence_threshold: 1e-3,
        };
        let mut h = TensorOptimizationHistory::new(config);
        // First step sets best.
        h.record(make_step(0, 1.0));
        // Record 5 steps with negligible improvement (below threshold).
        for i in 1..=5u64 {
            h.record(make_step(i, 1.0 - i as f64 * 1e-9));
        }
        assert_eq!(h.convergence_status(), ConvergenceStatus::Converged);
    }

    #[test]
    fn test_convergence_status_possibly_converged() {
        let config = OptimizationHistoryConfig {
            max_steps: 1000,
            convergence_patience: 8,
            convergence_threshold: 1e-3,
        };
        let mut h = TensorOptimizationHistory::new(config);
        // First step sets best.
        h.record(make_step(0, 1.0));
        // Record patience/2 = 4 steps with negligible improvement.
        for i in 1..=4u64 {
            h.record(make_step(i, 1.0 - i as f64 * 1e-9));
        }
        assert_eq!(h.convergence_status(), ConvergenceStatus::PossiblyConverged);
    }

    // -----------------------------------------------------------------------
    // 5. patience/2 boundary for PossiblyConverged
    // -----------------------------------------------------------------------

    #[test]
    fn test_possibly_converged_boundary_exact() {
        let config = OptimizationHistoryConfig {
            max_steps: 1000,
            convergence_patience: 10,
            convergence_threshold: 1e-3,
        };
        let mut h = TensorOptimizationHistory::new(config);
        h.record(make_step(0, 1.0));
        // patience/2 = 5 steps without progress → PossiblyConverged.
        for i in 1..=5u64 {
            h.record(make_step(i, 1.0 - i as f64 * 1e-9));
        }
        assert_eq!(h.convergence_status(), ConvergenceStatus::PossiblyConverged);
    }

    #[test]
    fn test_not_converged_below_patience_half() {
        let config = OptimizationHistoryConfig {
            max_steps: 1000,
            convergence_patience: 10,
            convergence_threshold: 1e-3,
        };
        let mut h = TensorOptimizationHistory::new(config);
        h.record(make_step(0, 1.0));
        // Only 4 steps without progress → still NotConverged (4 < patience/2=5).
        for i in 1..=4u64 {
            h.record(make_step(i, 1.0 - i as f64 * 1e-9));
        }
        assert_eq!(h.convergence_status(), ConvergenceStatus::NotConverged);
    }

    #[test]
    fn test_converged_at_full_patience() {
        let config = OptimizationHistoryConfig {
            max_steps: 1000,
            convergence_patience: 6,
            convergence_threshold: 1e-3,
        };
        let mut h = TensorOptimizationHistory::new(config);
        h.record(make_step(0, 1.0));
        for i in 1..=6u64 {
            h.record(make_step(i, 1.0 - i as f64 * 1e-9));
        }
        assert_eq!(h.convergence_status(), ConvergenceStatus::Converged);
    }

    // -----------------------------------------------------------------------
    // 6. consecutive_no_progress resets on real improvement
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_progress_counter_resets() {
        let config = OptimizationHistoryConfig {
            max_steps: 1000,
            convergence_patience: 3,
            convergence_threshold: 1e-3,
        };
        let mut h = TensorOptimizationHistory::new(config);
        h.record(make_step(0, 1.0));
        // 3 tiny steps → would be Converged if not reset.
        h.record(make_step(1, 1.0 - 1e-9));
        h.record(make_step(2, 1.0 - 2e-9));
        // A big improvement resets the counter.
        h.record(make_step(3, 0.0));
        // Now only 0 no-progress steps → NotConverged.
        assert_eq!(h.convergence_status(), ConvergenceStatus::NotConverged);
    }

    // -----------------------------------------------------------------------
    // 7. recent_improvement
    // -----------------------------------------------------------------------

    #[test]
    fn test_recent_improvement_empty() {
        let h = default_history();
        assert!((h.recent_improvement(5) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_recent_improvement_one_step() {
        let mut h = default_history();
        h.record(make_step(0, 1.0));
        assert!((h.recent_improvement(5) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_recent_improvement_full_window() {
        let mut h = default_history();
        for i in 0..5u64 {
            h.record(make_step(i, 5.0 - i as f64));
        }
        // First loss=5.0, last loss=1.0 → improvement=4.0 over last 5 steps.
        assert!((h.recent_improvement(5) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_recent_improvement_partial_window() {
        let mut h = default_history();
        for i in 0..10u64 {
            h.record(make_step(i, 10.0 - i as f64));
        }
        // Last 3 steps: losses 8.0, 9.0... wait, losses are 10-i: step9=1, step8=2, step7=3.
        // Window of 3: steps[7..10] → losses 3.0, 2.0, 1.0 → improvement = 3.0-1.0 = 2.0
        assert!((h.recent_improvement(3) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_recent_improvement_n_larger_than_history() {
        let mut h = default_history();
        h.record(make_step(0, 10.0));
        h.record(make_step(1, 6.0));
        // n=100 > 2 steps, uses all → improvement = 10.0 - 6.0 = 4.0
        assert!((h.recent_improvement(100) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_recent_improvement_negative_when_loss_increases() {
        let mut h = default_history();
        h.record(make_step(0, 1.0));
        h.record(make_step(1, 2.0));
        // first - last = 1.0 - 2.0 = -1.0 (worsening)
        assert!((h.recent_improvement(2) - (-1.0)).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // 8. avg_gradient_norm
    // -----------------------------------------------------------------------

    #[test]
    fn test_avg_gradient_norm_empty() {
        let h = default_history();
        assert!((h.avg_gradient_norm() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_avg_gradient_norm_single() {
        let mut h = default_history();
        h.record(make_step_full(0, 1.0, 0.4));
        assert!((h.avg_gradient_norm() - 0.4).abs() < 1e-12);
    }

    #[test]
    fn test_avg_gradient_norm_multiple() {
        let mut h = default_history();
        h.record(make_step_full(0, 1.0, 1.0));
        h.record(make_step_full(1, 0.9, 2.0));
        h.record(make_step_full(2, 0.8, 3.0));
        // mean = (1+2+3)/3 = 2.0
        assert!((h.avg_gradient_norm() - 2.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // 9. reset
    // -----------------------------------------------------------------------

    #[test]
    fn test_reset_clears_steps() {
        let mut h = default_history();
        for i in 0..5u64 {
            h.record(make_step(i, 1.0));
        }
        h.reset();
        assert!(h.steps.is_empty());
    }

    #[test]
    fn test_reset_restores_best_loss() {
        let mut h = default_history();
        h.record(make_step(0, 0.5));
        h.reset();
        assert_eq!(h.best_loss, f64::MAX);
    }

    #[test]
    fn test_reset_clears_best_step() {
        let mut h = default_history();
        h.record(make_step(7, 0.1));
        h.reset();
        assert_eq!(h.best_step, 0);
    }

    #[test]
    fn test_reset_clears_consecutive_no_progress() {
        let config = OptimizationHistoryConfig {
            max_steps: 1000,
            convergence_patience: 3,
            convergence_threshold: 1e-3,
        };
        let mut h = TensorOptimizationHistory::new(config);
        h.record(make_step(0, 1.0));
        h.record(make_step(1, 1.0 - 1e-9));
        h.record(make_step(2, 1.0 - 2e-9));
        h.reset();
        assert_eq!(h.consecutive_no_progress, 0);
    }

    #[test]
    fn test_reset_allows_fresh_recording() {
        let mut h = default_history();
        h.record(make_step(0, 5.0));
        h.reset();
        h.record(make_step(0, 3.0));
        assert_eq!(h.steps.len(), 1);
        assert!((h.best_loss - 3.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // 10. stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_empty() {
        let h = default_history();
        let s = h.stats();
        assert_eq!(s.total_steps, 0);
        assert_eq!(s.best_loss, f64::MAX);
        assert_eq!(s.best_step, 0);
        assert_eq!(s.current_loss, f64::MAX);
        assert!((s.avg_gradient_norm - 0.0).abs() < 1e-12);
        assert_eq!(s.convergence_status, ConvergenceStatus::NotConverged);
    }

    #[test]
    fn test_stats_correct_values() {
        let mut h = default_history();
        h.record(make_step_full(0, 2.0, 1.0));
        h.record(make_step_full(1, 1.0, 3.0));
        let s = h.stats();
        assert_eq!(s.total_steps, 2);
        assert!((s.best_loss - 1.0).abs() < 1e-12);
        assert_eq!(s.best_step, 1);
        assert!((s.current_loss - 1.0).abs() < 1e-12);
        assert!((s.avg_gradient_norm - 2.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // 11. last_step
    // -----------------------------------------------------------------------

    #[test]
    fn test_last_step_empty() {
        let h = default_history();
        assert!(h.last_step().is_none());
    }

    #[test]
    fn test_last_step_returns_latest() {
        let mut h = default_history();
        h.record(make_step(0, 2.0));
        h.record(make_step(1, 1.0));
        let last = h.last_step().expect("should have last step");
        assert_eq!(last.step, 1);
        assert!((last.loss - 1.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // 12. first record initialises best correctly
    // -----------------------------------------------------------------------

    #[test]
    fn test_first_record_sets_best() {
        let mut h = default_history();
        h.record(make_step(42, 7.5));
        assert!((h.best_loss - 7.5).abs() < 1e-12);
        assert_eq!(h.best_step, 42);
        assert_eq!(h.consecutive_no_progress, 0);
    }

    // -----------------------------------------------------------------------
    // 13. ConvergenceStatus derives
    // -----------------------------------------------------------------------

    #[test]
    fn test_convergence_status_copy() {
        let s = ConvergenceStatus::Converged;
        let t = s; // Copy
        assert_eq!(s, t);
    }

    #[test]
    fn test_convergence_status_debug() {
        let s = format!("{:?}", ConvergenceStatus::PossiblyConverged);
        assert_eq!(s, "PossiblyConverged");
    }
}
