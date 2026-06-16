//! Early stopping monitor for training loops.
//!
//! Provides patience-based early stopping with configurable criteria,
//! minimum delta thresholds, and minimum epoch enforcement.

use std::collections::HashMap;

/// Criterion used to decide when to stop training.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopCriterion {
    /// Stop when loss stops decreasing.
    MinLoss,
    /// Stop when accuracy stops increasing.
    MaxAccuracy,
    /// Custom metric, lower is better.
    MinMetric(String),
    /// Custom metric, higher is better.
    MaxMetric(String),
}

/// Decision returned by the monitor after each epoch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopDecision {
    /// Training should continue.
    Continue,
    /// Training should stop, with the given reason.
    Stop(String),
}

/// Configuration for the early stopping monitor.
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Which criterion to optimise.
    pub criterion: StopCriterion,
    /// Number of epochs without improvement before stopping.
    pub patience: usize,
    /// Minimum change to qualify as improvement.
    pub min_delta: f64,
    /// Do not stop before this many epochs have elapsed.
    pub min_epochs: usize,
    /// Whether the caller intends to restore the best checkpoint.
    pub restore_best: bool,
}

/// Metrics recorded for a single epoch.
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    /// Zero-based epoch index.
    pub epoch: usize,
    /// Training or validation loss.
    pub loss: f64,
    /// Optional accuracy value.
    pub accuracy: Option<f64>,
    /// Arbitrary named metrics.
    pub custom_metrics: HashMap<String, f64>,
}

/// Cumulative statistics tracked by the monitor.
#[derive(Debug, Clone, Default)]
pub struct EarlyStoppingStats {
    /// Total epochs observed.
    pub total_epochs: u64,
    /// Number of epochs that showed improvement.
    pub improvements: u64,
    /// Best metric value seen so far.
    pub best_value: f64,
    /// Epoch at which the best value was observed.
    pub best_epoch: usize,
    /// Epoch at which training was stopped, if any.
    pub stopped_at: Option<usize>,
}

/// Monitors training progress and decides when to stop.
pub struct EarlyStoppingMonitor {
    config: EarlyStoppingConfig,
    history: Vec<EpochMetrics>,
    best_value: f64,
    best_epoch: usize,
    epochs_without_improvement: usize,
    stopped: bool,
    stats: EarlyStoppingStats,
}

impl EarlyStoppingMonitor {
    /// Create a new monitor with the given configuration.
    pub fn new(config: EarlyStoppingConfig) -> Self {
        let initial_best = match &config.criterion {
            StopCriterion::MinLoss | StopCriterion::MinMetric(_) => f64::INFINITY,
            StopCriterion::MaxAccuracy | StopCriterion::MaxMetric(_) => f64::NEG_INFINITY,
        };
        Self {
            config,
            history: Vec::new(),
            best_value: initial_best,
            best_epoch: 0,
            epochs_without_improvement: 0,
            stopped: false,
            stats: EarlyStoppingStats {
                best_value: initial_best,
                ..Default::default()
            },
        }
    }

    /// Check whether training should stop after observing the given metrics.
    ///
    /// Returns [`StopDecision::Continue`] or [`StopDecision::Stop`] with a reason.
    pub fn check(&mut self, metrics: EpochMetrics) -> StopDecision {
        if self.stopped {
            return StopDecision::Stop("Already stopped".to_string());
        }

        let epoch = metrics.epoch;
        let value = self.extract_metric(&metrics);
        self.history.push(metrics);
        self.stats.total_epochs += 1;

        let value = match value {
            Some(v) => v,
            None => {
                // Metric not available — treat as no improvement.
                self.epochs_without_improvement += 1;
                return self.maybe_stop(epoch);
            }
        };

        // NaN is never an improvement.
        if value.is_nan() {
            self.epochs_without_improvement += 1;
            return self.maybe_stop(epoch);
        }

        if self.is_improvement(value) {
            self.best_value = value;
            self.best_epoch = epoch;
            self.epochs_without_improvement = 0;
            self.stats.improvements += 1;
            self.stats.best_value = value;
            self.stats.best_epoch = epoch;
        } else {
            self.epochs_without_improvement += 1;
        }

        self.maybe_stop(epoch)
    }

    /// Return whether `value` is an improvement over the current best,
    /// respecting [`EarlyStoppingConfig::min_delta`] and criterion direction.
    pub fn is_improvement(&self, value: f64) -> bool {
        if value.is_nan() {
            return false;
        }
        match &self.config.criterion {
            StopCriterion::MinLoss | StopCriterion::MinMetric(_) => {
                value < self.best_value - self.config.min_delta
            }
            StopCriterion::MaxAccuracy | StopCriterion::MaxMetric(_) => {
                value > self.best_value + self.config.min_delta
            }
        }
    }

    /// Best metric value observed so far.
    pub fn best_value(&self) -> f64 {
        self.best_value
    }

    /// Epoch at which the best value was observed.
    pub fn best_epoch(&self) -> usize {
        self.best_epoch
    }

    /// Number of consecutive epochs without improvement.
    pub fn epochs_without_improvement(&self) -> usize {
        self.epochs_without_improvement
    }

    /// Whether the monitor has triggered a stop.
    pub fn should_stop(&self) -> bool {
        self.stopped
    }

    /// Reset all state so the monitor can be reused.
    pub fn reset(&mut self) {
        let initial_best = match &self.config.criterion {
            StopCriterion::MinLoss | StopCriterion::MinMetric(_) => f64::INFINITY,
            StopCriterion::MaxAccuracy | StopCriterion::MaxMetric(_) => f64::NEG_INFINITY,
        };
        self.history.clear();
        self.best_value = initial_best;
        self.best_epoch = 0;
        self.epochs_without_improvement = 0;
        self.stopped = false;
        self.stats = EarlyStoppingStats {
            best_value: initial_best,
            ..Default::default()
        };
    }

    /// Full history of epoch metrics.
    pub fn history(&self) -> &[EpochMetrics] {
        &self.history
    }

    /// Cumulative statistics.
    pub fn stats(&self) -> &EarlyStoppingStats {
        &self.stats
    }

    /// How many more epochs of no-improvement are allowed before stopping.
    pub fn remaining_patience(&self) -> usize {
        self.config
            .patience
            .saturating_sub(self.epochs_without_improvement)
    }

    /// Extract the metric value that matches the configured criterion.
    pub fn extract_metric(&self, metrics: &EpochMetrics) -> Option<f64> {
        match &self.config.criterion {
            StopCriterion::MinLoss => Some(metrics.loss),
            StopCriterion::MaxAccuracy => metrics.accuracy,
            StopCriterion::MinMetric(name) => metrics.custom_metrics.get(name).copied(),
            StopCriterion::MaxMetric(name) => metrics.custom_metrics.get(name).copied(),
        }
    }

    // --- private helpers ---

    fn maybe_stop(&mut self, epoch: usize) -> StopDecision {
        // Never stop before min_epochs.
        if self.stats.total_epochs < self.config.min_epochs as u64 {
            return StopDecision::Continue;
        }
        if self.epochs_without_improvement >= self.config.patience {
            self.stopped = true;
            self.stats.stopped_at = Some(epoch);
            let reason = format!(
                "No improvement for {} epochs (best={:.6} at epoch {})",
                self.config.patience, self.best_value, self.best_epoch
            );
            StopDecision::Stop(reason)
        } else {
            StopDecision::Continue
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> EarlyStoppingConfig {
        EarlyStoppingConfig {
            criterion: StopCriterion::MinLoss,
            patience: 3,
            min_delta: 0.0,
            min_epochs: 0,
            restore_best: false,
        }
    }

    fn epoch(epoch: usize, loss: f64) -> EpochMetrics {
        EpochMetrics {
            epoch,
            loss,
            accuracy: None,
            custom_metrics: HashMap::new(),
        }
    }

    fn epoch_with_acc(epoch: usize, loss: f64, acc: f64) -> EpochMetrics {
        EpochMetrics {
            epoch,
            loss,
            accuracy: Some(acc),
            custom_metrics: HashMap::new(),
        }
    }

    fn epoch_with_custom(epoch: usize, loss: f64, key: &str, val: f64) -> EpochMetrics {
        let mut m = HashMap::new();
        m.insert(key.to_string(), val);
        EpochMetrics {
            epoch,
            loss,
            accuracy: None,
            custom_metrics: m,
        }
    }

    // --- MinLoss tests ---

    #[test]
    fn min_loss_stops_after_patience_exhausted() {
        let mut mon = EarlyStoppingMonitor::new(default_config());
        assert_eq!(mon.check(epoch(0, 1.0)), StopDecision::Continue);
        assert_eq!(mon.check(epoch(1, 1.1)), StopDecision::Continue);
        assert_eq!(mon.check(epoch(2, 1.2)), StopDecision::Continue);
        match mon.check(epoch(3, 1.3)) {
            StopDecision::Stop(_) => {}
            StopDecision::Continue => panic!("expected Stop"),
        }
    }

    #[test]
    fn min_loss_continues_on_improvement() {
        let mut mon = EarlyStoppingMonitor::new(default_config());
        assert_eq!(mon.check(epoch(0, 1.0)), StopDecision::Continue);
        assert_eq!(mon.check(epoch(1, 0.9)), StopDecision::Continue);
        assert_eq!(mon.check(epoch(2, 0.8)), StopDecision::Continue);
        assert_eq!(mon.check(epoch(3, 0.7)), StopDecision::Continue);
        assert_eq!(mon.check(epoch(4, 0.6)), StopDecision::Continue);
    }

    #[test]
    fn min_loss_resets_patience_on_improvement() {
        let mut mon = EarlyStoppingMonitor::new(default_config());
        assert_eq!(mon.check(epoch(0, 1.0)), StopDecision::Continue);
        assert_eq!(mon.check(epoch(1, 1.1)), StopDecision::Continue); // no improvement
        assert_eq!(mon.check(epoch(2, 1.2)), StopDecision::Continue); // no improvement
        assert_eq!(mon.check(epoch(3, 0.5)), StopDecision::Continue); // improvement!
        assert_eq!(mon.epochs_without_improvement(), 0);
        assert_eq!(mon.check(epoch(4, 0.6)), StopDecision::Continue); // no improvement (1)
        assert_eq!(mon.check(epoch(5, 0.7)), StopDecision::Continue); // no improvement (2)
        match mon.check(epoch(6, 0.8)) {
            StopDecision::Stop(_) => {}
            StopDecision::Continue => panic!("expected Stop"),
        }
    }

    // --- MaxAccuracy tests ---

    #[test]
    fn max_accuracy_stops_after_patience() {
        let config = EarlyStoppingConfig {
            criterion: StopCriterion::MaxAccuracy,
            patience: 2,
            min_delta: 0.0,
            min_epochs: 0,
            restore_best: false,
        };
        let mut mon = EarlyStoppingMonitor::new(config);
        assert_eq!(
            mon.check(epoch_with_acc(0, 1.0, 0.8)),
            StopDecision::Continue
        );
        assert_eq!(
            mon.check(epoch_with_acc(1, 0.9, 0.79)),
            StopDecision::Continue
        );
        match mon.check(epoch_with_acc(2, 0.8, 0.78)) {
            StopDecision::Stop(_) => {}
            StopDecision::Continue => panic!("expected Stop"),
        }
    }

    #[test]
    fn max_accuracy_continues_on_improvement() {
        let config = EarlyStoppingConfig {
            criterion: StopCriterion::MaxAccuracy,
            patience: 2,
            min_delta: 0.0,
            min_epochs: 0,
            restore_best: false,
        };
        let mut mon = EarlyStoppingMonitor::new(config);
        assert_eq!(
            mon.check(epoch_with_acc(0, 1.0, 0.5)),
            StopDecision::Continue
        );
        assert_eq!(
            mon.check(epoch_with_acc(1, 0.9, 0.6)),
            StopDecision::Continue
        );
        assert_eq!(
            mon.check(epoch_with_acc(2, 0.8, 0.7)),
            StopDecision::Continue
        );
        assert!(!mon.should_stop());
    }

    // --- min_delta tests ---

    #[test]
    fn min_delta_prevents_tiny_improvements() {
        let config = EarlyStoppingConfig {
            criterion: StopCriterion::MinLoss,
            patience: 2,
            min_delta: 0.1,
            min_epochs: 0,
            restore_best: false,
        };
        let mut mon = EarlyStoppingMonitor::new(config);
        assert_eq!(mon.check(epoch(0, 1.0)), StopDecision::Continue);
        // Decrease by 0.05 — below min_delta, not improvement.
        assert_eq!(mon.check(epoch(1, 0.95)), StopDecision::Continue);
        assert_eq!(mon.epochs_without_improvement(), 1);
        match mon.check(epoch(2, 0.96)) {
            StopDecision::Stop(_) => {}
            StopDecision::Continue => panic!("expected Stop"),
        }
    }

    #[test]
    fn min_delta_allows_large_improvements() {
        let config = EarlyStoppingConfig {
            criterion: StopCriterion::MinLoss,
            patience: 2,
            min_delta: 0.1,
            min_epochs: 0,
            restore_best: false,
        };
        let mut mon = EarlyStoppingMonitor::new(config);
        assert_eq!(mon.check(epoch(0, 1.0)), StopDecision::Continue);
        assert_eq!(mon.check(epoch(1, 0.8)), StopDecision::Continue); // 0.2 > 0.1
        assert_eq!(mon.epochs_without_improvement(), 0);
    }

    #[test]
    fn min_delta_for_max_accuracy() {
        let config = EarlyStoppingConfig {
            criterion: StopCriterion::MaxAccuracy,
            patience: 2,
            min_delta: 0.05,
            min_epochs: 0,
            restore_best: false,
        };
        let mut mon = EarlyStoppingMonitor::new(config);
        assert_eq!(
            mon.check(epoch_with_acc(0, 1.0, 0.7)),
            StopDecision::Continue
        );
        // +0.02 — not enough.
        assert_eq!(
            mon.check(epoch_with_acc(1, 0.9, 0.72)),
            StopDecision::Continue
        );
        assert_eq!(mon.epochs_without_improvement(), 1);
        // +0.1 from best — enough.
        assert_eq!(
            mon.check(epoch_with_acc(2, 0.8, 0.81)),
            StopDecision::Continue
        );
        assert_eq!(mon.epochs_without_improvement(), 0);
    }

    // --- min_epochs enforcement ---

    #[test]
    fn min_epochs_prevents_early_stop() {
        let config = EarlyStoppingConfig {
            criterion: StopCriterion::MinLoss,
            patience: 1,
            min_delta: 0.0,
            min_epochs: 5,
            restore_best: false,
        };
        let mut mon = EarlyStoppingMonitor::new(config);
        // Even with patience=1, should not stop before 5 epochs.
        for i in 0..4 {
            assert_eq!(mon.check(epoch(i, 1.0 + i as f64)), StopDecision::Continue);
        }
        // Epoch 5 — patience already exhausted, should stop.
        match mon.check(epoch(4, 5.0)) {
            StopDecision::Stop(_) => {}
            StopDecision::Continue => panic!("expected Stop after min_epochs"),
        }
    }

    #[test]
    fn min_epochs_allows_stop_at_boundary() {
        let config = EarlyStoppingConfig {
            criterion: StopCriterion::MinLoss,
            patience: 2,
            min_delta: 0.0,
            min_epochs: 3,
            restore_best: false,
        };
        let mut mon = EarlyStoppingMonitor::new(config);
        assert_eq!(mon.check(epoch(0, 1.0)), StopDecision::Continue);
        assert_eq!(mon.check(epoch(1, 1.1)), StopDecision::Continue);
        // Third epoch — min_epochs reached, patience exhausted.
        match mon.check(epoch(2, 1.2)) {
            StopDecision::Stop(_) => {}
            StopDecision::Continue => panic!("expected Stop"),
        }
    }

    // --- Custom metrics ---

    #[test]
    fn custom_min_metric() {
        let config = EarlyStoppingConfig {
            criterion: StopCriterion::MinMetric("val_loss".to_string()),
            patience: 2,
            min_delta: 0.0,
            min_epochs: 0,
            restore_best: false,
        };
        let mut mon = EarlyStoppingMonitor::new(config);
        assert_eq!(
            mon.check(epoch_with_custom(0, 999.0, "val_loss", 1.0)),
            StopDecision::Continue
        );
        assert_eq!(
            mon.check(epoch_with_custom(1, 999.0, "val_loss", 0.9)),
            StopDecision::Continue
        );
        assert_eq!(mon.epochs_without_improvement(), 0);
        assert_eq!(
            mon.check(epoch_with_custom(2, 999.0, "val_loss", 1.0)),
            StopDecision::Continue
        );
        match mon.check(epoch_with_custom(3, 999.0, "val_loss", 1.1)) {
            StopDecision::Stop(_) => {}
            StopDecision::Continue => panic!("expected Stop"),
        }
    }

    #[test]
    fn custom_max_metric() {
        let config = EarlyStoppingConfig {
            criterion: StopCriterion::MaxMetric("f1_score".to_string()),
            patience: 2,
            min_delta: 0.0,
            min_epochs: 0,
            restore_best: false,
        };
        let mut mon = EarlyStoppingMonitor::new(config);
        assert_eq!(
            mon.check(epoch_with_custom(0, 999.0, "f1_score", 0.5)),
            StopDecision::Continue
        );
        assert_eq!(
            mon.check(epoch_with_custom(1, 999.0, "f1_score", 0.6)),
            StopDecision::Continue
        );
        assert_eq!(
            mon.check(epoch_with_custom(2, 999.0, "f1_score", 0.55)),
            StopDecision::Continue
        );
        match mon.check(epoch_with_custom(3, 999.0, "f1_score", 0.54)) {
            StopDecision::Stop(_) => {}
            StopDecision::Continue => panic!("expected Stop"),
        }
    }

    #[test]
    fn missing_custom_metric_counts_as_no_improvement() {
        let config = EarlyStoppingConfig {
            criterion: StopCriterion::MinMetric("val_loss".to_string()),
            patience: 2,
            min_delta: 0.0,
            min_epochs: 0,
            restore_best: false,
        };
        let mut mon = EarlyStoppingMonitor::new(config);
        // Epoch 0: metric present.
        assert_eq!(
            mon.check(epoch_with_custom(0, 1.0, "val_loss", 0.5)),
            StopDecision::Continue
        );
        // Epoch 1, 2: metric missing — no improvement.
        assert_eq!(mon.check(epoch(1, 1.0)), StopDecision::Continue);
        match mon.check(epoch(2, 1.0)) {
            StopDecision::Stop(_) => {}
            StopDecision::Continue => panic!("expected Stop"),
        }
    }

    // --- reset ---

    #[test]
    fn reset_clears_all_state() {
        let mut mon = EarlyStoppingMonitor::new(default_config());
        mon.check(epoch(0, 1.0));
        mon.check(epoch(1, 1.1));
        mon.check(epoch(2, 1.2));
        mon.check(epoch(3, 1.3)); // stopped
        assert!(mon.should_stop());
        assert_eq!(mon.history().len(), 4);

        mon.reset();
        assert!(!mon.should_stop());
        assert!(mon.history().is_empty());
        assert_eq!(mon.epochs_without_improvement(), 0);
        assert_eq!(mon.best_value(), f64::INFINITY);
        assert_eq!(mon.stats().total_epochs, 0);
    }

    // --- improvement detection ---

    #[test]
    fn is_improvement_min_loss() {
        let mon = EarlyStoppingMonitor::new(default_config());
        // best is INFINITY, so any finite value is an improvement.
        assert!(mon.is_improvement(1.0));
        assert!(mon.is_improvement(-100.0));
    }

    #[test]
    fn is_improvement_max_accuracy() {
        let config = EarlyStoppingConfig {
            criterion: StopCriterion::MaxAccuracy,
            patience: 3,
            min_delta: 0.0,
            min_epochs: 0,
            restore_best: false,
        };
        let mon = EarlyStoppingMonitor::new(config);
        // best is NEG_INFINITY, so any finite value is an improvement.
        assert!(mon.is_improvement(0.0));
        assert!(mon.is_improvement(0.5));
    }

    #[test]
    fn is_improvement_nan_never_improves() {
        let mon = EarlyStoppingMonitor::new(default_config());
        assert!(!mon.is_improvement(f64::NAN));
    }

    // --- history ---

    #[test]
    fn history_tracks_all_epochs() {
        let mut mon = EarlyStoppingMonitor::new(default_config());
        mon.check(epoch(0, 1.0));
        mon.check(epoch(1, 0.9));
        mon.check(epoch(2, 0.8));
        assert_eq!(mon.history().len(), 3);
        assert_eq!(mon.history()[0].epoch, 0);
        assert_eq!(mon.history()[2].epoch, 2);
    }

    // --- stats ---

    #[test]
    fn stats_track_improvements() {
        let mut mon = EarlyStoppingMonitor::new(default_config());
        mon.check(epoch(0, 1.0)); // improvement (first)
        mon.check(epoch(1, 0.9)); // improvement
        mon.check(epoch(2, 0.8)); // improvement
        mon.check(epoch(3, 0.9)); // no improvement
        let s = mon.stats();
        assert_eq!(s.total_epochs, 4);
        assert_eq!(s.improvements, 3);
        assert!((s.best_value - 0.8).abs() < 1e-10);
        assert_eq!(s.best_epoch, 2);
        assert!(s.stopped_at.is_none());
    }

    #[test]
    fn stats_record_stopped_at() {
        let mut mon = EarlyStoppingMonitor::new(default_config());
        mon.check(epoch(0, 1.0));
        mon.check(epoch(1, 1.1));
        mon.check(epoch(2, 1.2));
        mon.check(epoch(3, 1.3)); // stop
        assert_eq!(mon.stats().stopped_at, Some(3));
    }

    // --- stop reason message ---

    #[test]
    fn stop_reason_includes_details() {
        let mut mon = EarlyStoppingMonitor::new(default_config());
        mon.check(epoch(0, 0.5));
        mon.check(epoch(1, 0.6));
        mon.check(epoch(2, 0.7));
        match mon.check(epoch(3, 0.8)) {
            StopDecision::Stop(reason) => {
                assert!(reason.contains("3 epochs"), "reason: {reason}");
                assert!(reason.contains("0.5"), "reason: {reason}");
                assert!(reason.contains("epoch 0"), "reason: {reason}");
            }
            StopDecision::Continue => panic!("expected Stop"),
        }
    }

    // --- edge cases ---

    #[test]
    fn nan_loss_does_not_improve() {
        let mut mon = EarlyStoppingMonitor::new(default_config());
        mon.check(epoch(0, 1.0));
        mon.check(epoch(1, f64::NAN));
        assert_eq!(mon.epochs_without_improvement(), 1);
        assert!((mon.best_value() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn identical_values_no_improvement() {
        let mut mon = EarlyStoppingMonitor::new(default_config());
        mon.check(epoch(0, 1.0));
        mon.check(epoch(1, 1.0)); // same — not an improvement
        assert_eq!(mon.epochs_without_improvement(), 1);
    }

    #[test]
    fn identical_values_with_zero_delta_still_no_improvement() {
        let config = EarlyStoppingConfig {
            criterion: StopCriterion::MinLoss,
            patience: 3,
            min_delta: 0.0,
            min_epochs: 0,
            restore_best: false,
        };
        let mut mon = EarlyStoppingMonitor::new(config);
        mon.check(epoch(0, 1.0));
        mon.check(epoch(1, 1.0));
        // Strictly less than required — equal is not improvement.
        assert_eq!(mon.epochs_without_improvement(), 1);
    }

    // --- remaining patience ---

    #[test]
    fn remaining_patience_countdown() {
        let mut mon = EarlyStoppingMonitor::new(default_config());
        assert_eq!(mon.remaining_patience(), 3);
        mon.check(epoch(0, 1.0)); // improvement → reset
        assert_eq!(mon.remaining_patience(), 3);
        mon.check(epoch(1, 1.1)); // no improvement
        assert_eq!(mon.remaining_patience(), 2);
        mon.check(epoch(2, 1.2));
        assert_eq!(mon.remaining_patience(), 1);
        mon.check(epoch(3, 1.3)); // stop
        assert_eq!(mon.remaining_patience(), 0);
    }

    #[test]
    fn remaining_patience_resets_on_improvement() {
        let mut mon = EarlyStoppingMonitor::new(default_config());
        mon.check(epoch(0, 1.0));
        mon.check(epoch(1, 1.1)); // no
        mon.check(epoch(2, 1.2)); // no
        assert_eq!(mon.remaining_patience(), 1);
        mon.check(epoch(3, 0.5)); // improvement!
        assert_eq!(mon.remaining_patience(), 3);
    }

    // --- extract_metric ---

    #[test]
    fn extract_metric_min_loss() {
        let mon = EarlyStoppingMonitor::new(default_config());
        let m = epoch(0, 0.42);
        assert_eq!(mon.extract_metric(&m), Some(0.42));
    }

    #[test]
    fn extract_metric_max_accuracy_present() {
        let config = EarlyStoppingConfig {
            criterion: StopCriterion::MaxAccuracy,
            ..default_config()
        };
        let mon = EarlyStoppingMonitor::new(config);
        let m = epoch_with_acc(0, 1.0, 0.95);
        assert_eq!(mon.extract_metric(&m), Some(0.95));
    }

    #[test]
    fn extract_metric_max_accuracy_absent() {
        let config = EarlyStoppingConfig {
            criterion: StopCriterion::MaxAccuracy,
            ..default_config()
        };
        let mon = EarlyStoppingMonitor::new(config);
        let m = epoch(0, 1.0);
        assert_eq!(mon.extract_metric(&m), None);
    }

    #[test]
    fn extract_metric_custom_present() {
        let config = EarlyStoppingConfig {
            criterion: StopCriterion::MinMetric("rmse".to_string()),
            ..default_config()
        };
        let mon = EarlyStoppingMonitor::new(config);
        let m = epoch_with_custom(0, 1.0, "rmse", std::f64::consts::PI);
        assert_eq!(mon.extract_metric(&m), Some(std::f64::consts::PI));
    }

    #[test]
    fn extract_metric_custom_absent() {
        let config = EarlyStoppingConfig {
            criterion: StopCriterion::MinMetric("rmse".to_string()),
            ..default_config()
        };
        let mon = EarlyStoppingMonitor::new(config);
        let m = epoch(0, 1.0);
        assert_eq!(mon.extract_metric(&m), None);
    }

    // --- already stopped ---

    #[test]
    fn returns_stop_once_stopped() {
        let mut mon = EarlyStoppingMonitor::new(default_config());
        mon.check(epoch(0, 1.0));
        mon.check(epoch(1, 1.1));
        mon.check(epoch(2, 1.2));
        mon.check(epoch(3, 1.3)); // stop
        assert!(mon.should_stop());
        match mon.check(epoch(4, 0.1)) {
            StopDecision::Stop(reason) => assert!(reason.contains("Already")),
            StopDecision::Continue => panic!("expected Stop"),
        }
    }

    // --- best_epoch and best_value after improvements ---

    #[test]
    fn best_epoch_tracks_correctly() {
        let mut mon = EarlyStoppingMonitor::new(default_config());
        mon.check(epoch(0, 1.0));
        assert_eq!(mon.best_epoch(), 0);
        mon.check(epoch(1, 0.8));
        assert_eq!(mon.best_epoch(), 1);
        mon.check(epoch(2, 0.9));
        assert_eq!(mon.best_epoch(), 1); // still 1
    }

    #[test]
    fn best_value_tracks_correctly() {
        let mut mon = EarlyStoppingMonitor::new(default_config());
        mon.check(epoch(0, 1.0));
        assert!((mon.best_value() - 1.0).abs() < 1e-10);
        mon.check(epoch(1, 0.5));
        assert!((mon.best_value() - 0.5).abs() < 1e-10);
        mon.check(epoch(2, 0.7));
        assert!((mon.best_value() - 0.5).abs() < 1e-10);
    }
}
