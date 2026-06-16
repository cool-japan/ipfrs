//! TensorCheckpointScheduler — automatic scheduling of tensor checkpoint saves.
//!
//! Manages checkpoint saves triggered by configurable policies:
//! - **StepInterval**: save every N training steps
//! - **LossImprovement**: save when loss decreases by at least `min_delta`
//! - **TickInterval**: save every N wall-clock ticks (monotonic counter)
//!
//! The scheduler maintains a bounded history of checkpoint records, pruning
//! the oldest when the configured maximum is exceeded. All statistics are
//! tracked per-trigger-type for observability.

/// Specifies the condition under which an automatic checkpoint save is triggered.
#[derive(Clone, Debug, PartialEq)]
pub enum CheckpointTrigger {
    /// Trigger every `every_n_steps` training steps.
    StepInterval { every_n_steps: u64 },
    /// Trigger when the loss improves (decreases) by at least `min_delta`.
    LossImprovement { min_delta: f64 },
    /// Trigger every `every_n_ticks` monotonic ticks.
    TickInterval { every_n_ticks: u64 },
}

/// A record of a single checkpoint save event.
#[derive(Clone, Debug)]
pub struct CheckpointRecord {
    /// Unique, monotonically-increasing identifier for this checkpoint.
    pub checkpoint_id: u64,
    /// Training step at which this checkpoint was saved.
    pub step: u64,
    /// Loss value at save time.
    pub loss: f64,
    /// The trigger that caused this checkpoint to be saved.
    pub trigger: CheckpointTrigger,
    /// Tick counter value at save time.
    pub saved_at_tick: u64,
}

/// Configuration for [`TensorCheckpointScheduler`].
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    /// Ordered list of triggers; the first matching trigger fires the checkpoint.
    pub triggers: Vec<CheckpointTrigger>,
    /// Maximum number of checkpoint records to retain; oldest is removed when exceeded.
    pub max_checkpoints: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            triggers: Vec::new(),
            max_checkpoints: 5,
        }
    }
}

/// Cumulative statistics for [`TensorCheckpointScheduler`].
#[derive(Clone, Debug, Default)]
pub struct SchedulerStats {
    /// Total number of checkpoints saved (across all trigger types).
    pub total_checkpoints_saved: u64,
    /// Checkpoints triggered by a [`CheckpointTrigger::StepInterval`] rule.
    pub triggered_by_step: u64,
    /// Checkpoints triggered by a [`CheckpointTrigger::LossImprovement`] rule.
    pub triggered_by_loss: u64,
    /// Checkpoints triggered by a [`CheckpointTrigger::TickInterval`] rule.
    pub triggered_by_tick: u64,
    /// Number of checkpoint records deleted due to exceeding `max_checkpoints`.
    pub checkpoints_pruned: u64,
}

/// Manages automatic scheduling of tensor checkpoint saves.
///
/// Evaluates a configurable list of [`CheckpointTrigger`]s on each call to
/// [`advance`][TensorCheckpointScheduler::advance] and records a checkpoint
/// whenever the first matching trigger fires.
pub struct TensorCheckpointScheduler {
    /// Configuration (triggers + retention policy).
    pub config: SchedulerConfig,
    /// Ordered (oldest-first) list of saved checkpoint records.
    pub checkpoints: Vec<CheckpointRecord>,
    /// Next checkpoint ID to assign (starts at 0, increments after use).
    pub next_checkpoint_id: u64,
    /// Training step at which the last checkpoint was saved.
    pub last_checkpoint_step: u64,
    /// Tick value at which the last checkpoint was saved.
    pub last_checkpoint_tick: u64,
    /// Best (lowest) loss seen so far; initialised to `f64::MAX`.
    pub best_loss: f64,
    /// Running statistics.
    pub stats: SchedulerStats,
}

impl TensorCheckpointScheduler {
    /// Create a new scheduler with the given configuration.
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            checkpoints: Vec::new(),
            next_checkpoint_id: 0,
            last_checkpoint_step: 0,
            last_checkpoint_tick: 0,
            best_loss: f64::MAX,
            stats: SchedulerStats::default(),
        }
    }

    /// Evaluate all configured triggers and return the first one that fires,
    /// or `None` if no trigger is currently satisfied.
    ///
    /// Trigger semantics:
    /// - `StepInterval { every_n_steps: n }`: fires when
    ///   `current_step > 0` AND `(current_step − last_checkpoint_step) >= n`.
    /// - `LossImprovement { min_delta: d }`: fires when
    ///   `best_loss == f64::MAX` (first observation) OR
    ///   `(best_loss − current_loss) >= d`.
    /// - `TickInterval { every_n_ticks: n }`: fires when
    ///   `current_tick > 0` AND `(current_tick − last_checkpoint_tick) >= n`.
    pub fn should_checkpoint(
        &self,
        current_step: u64,
        current_loss: f64,
        current_tick: u64,
    ) -> Option<CheckpointTrigger> {
        for trigger in &self.config.triggers {
            let fires = match trigger {
                CheckpointTrigger::StepInterval { every_n_steps } => {
                    current_step > 0
                        && current_step.saturating_sub(self.last_checkpoint_step) >= *every_n_steps
                }
                CheckpointTrigger::LossImprovement { min_delta } => {
                    self.best_loss == f64::MAX || (self.best_loss - current_loss) >= *min_delta
                }
                CheckpointTrigger::TickInterval { every_n_ticks } => {
                    current_tick > 0
                        && current_tick.saturating_sub(self.last_checkpoint_tick) >= *every_n_ticks
                }
            };
            if fires {
                return Some(trigger.clone());
            }
        }
        None
    }

    /// Record a checkpoint save event.
    ///
    /// Updates internal state (step/tick cursors, best loss) and statistics,
    /// then prunes the oldest record if the history exceeds `max_checkpoints`.
    ///
    /// Returns the newly assigned checkpoint ID.
    pub fn record_checkpoint(
        &mut self,
        step: u64,
        loss: f64,
        trigger: CheckpointTrigger,
        current_tick: u64,
    ) -> u64 {
        let id = self.next_checkpoint_id;
        self.next_checkpoint_id += 1;

        let record = CheckpointRecord {
            checkpoint_id: id,
            step,
            loss,
            trigger: trigger.clone(),
            saved_at_tick: current_tick,
        };
        self.checkpoints.push(record);

        // Update cursors.
        self.last_checkpoint_step = step;
        self.last_checkpoint_tick = current_tick;

        // Update best loss.
        if loss < self.best_loss {
            self.best_loss = loss;
        }

        // Update stats.
        self.stats.total_checkpoints_saved += 1;
        match &trigger {
            CheckpointTrigger::StepInterval { .. } => self.stats.triggered_by_step += 1,
            CheckpointTrigger::LossImprovement { .. } => self.stats.triggered_by_loss += 1,
            CheckpointTrigger::TickInterval { .. } => self.stats.triggered_by_tick += 1,
        }

        // Prune oldest if over limit.
        if self.checkpoints.len() > self.config.max_checkpoints {
            self.checkpoints.remove(0);
            self.stats.checkpoints_pruned += 1;
        }

        id
    }

    /// Advance the scheduler by one observation.
    ///
    /// Evaluates triggers; if one fires, records a checkpoint and returns
    /// `Some(checkpoint_id)`. Returns `None` when no checkpoint is needed.
    pub fn advance(&mut self, step: u64, loss: f64, current_tick: u64) -> Option<u64> {
        let trigger = self.should_checkpoint(step, loss, current_tick)?;
        let id = self.record_checkpoint(step, loss, trigger, current_tick);
        Some(id)
    }

    /// Return a reference to the most recently saved checkpoint record, if any.
    pub fn latest_checkpoint(&self) -> Option<&CheckpointRecord> {
        self.checkpoints.last()
    }

    /// Return a reference to the cumulative statistics.
    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn step_scheduler(every_n_steps: u64) -> TensorCheckpointScheduler {
        let config = SchedulerConfig {
            triggers: vec![CheckpointTrigger::StepInterval { every_n_steps }],
            max_checkpoints: 5,
        };
        TensorCheckpointScheduler::new(config)
    }

    fn loss_scheduler(min_delta: f64) -> TensorCheckpointScheduler {
        let config = SchedulerConfig {
            triggers: vec![CheckpointTrigger::LossImprovement { min_delta }],
            max_checkpoints: 5,
        };
        TensorCheckpointScheduler::new(config)
    }

    fn tick_scheduler(every_n_ticks: u64) -> TensorCheckpointScheduler {
        let config = SchedulerConfig {
            triggers: vec![CheckpointTrigger::TickInterval { every_n_ticks }],
            max_checkpoints: 5,
        };
        TensorCheckpointScheduler::new(config)
    }

    // ── StepInterval ────────────────────────────────────────────────────────

    #[test]
    fn step_interval_fires_at_correct_step() {
        let mut sched = step_scheduler(10);
        let id = sched.advance(10, 1.0, 0);
        assert!(id.is_some(), "should fire at step 10");
    }

    #[test]
    fn step_interval_does_not_fire_before_interval() {
        let mut sched = step_scheduler(10);
        assert!(sched.advance(9, 1.0, 0).is_none());
    }

    #[test]
    fn step_interval_does_not_fire_at_step_zero() {
        let mut sched = step_scheduler(1);
        // step == 0 should never trigger even with n=1
        assert!(sched.advance(0, 1.0, 0).is_none());
    }

    #[test]
    fn step_interval_fires_multiple_times() {
        let mut sched = step_scheduler(5);
        let id1 = sched.advance(5, 1.0, 0);
        assert!(id1.is_some());
        // After recording last_checkpoint_step=5, next fire should be at 10
        assert!(sched.advance(9, 0.9, 0).is_none());
        let id2 = sched.advance(10, 0.9, 0);
        assert!(id2.is_some());
        assert_ne!(
            id1.expect("test: should succeed"),
            id2.expect("test: should succeed")
        );
    }

    #[test]
    fn step_interval_fires_exactly_at_multiple() {
        let mut sched = step_scheduler(3);
        assert!(sched.advance(2, 1.0, 0).is_none());
        assert!(sched.advance(3, 1.0, 0).is_some());
    }

    #[test]
    fn step_interval_stats_incremented() {
        let mut sched = step_scheduler(5);
        sched.advance(5, 1.0, 0);
        assert_eq!(sched.stats().triggered_by_step, 1);
        assert_eq!(sched.stats().total_checkpoints_saved, 1);
        assert_eq!(sched.stats().triggered_by_loss, 0);
        assert_eq!(sched.stats().triggered_by_tick, 0);
    }

    // ── LossImprovement ─────────────────────────────────────────────────────

    #[test]
    fn loss_improvement_fires_on_first_observation() {
        // best_loss starts at f64::MAX so any loss triggers it
        let mut sched = loss_scheduler(0.01);
        assert!(sched.advance(1, 2.0, 1).is_some());
    }

    #[test]
    fn loss_improvement_fires_when_improved_enough() {
        let mut sched = loss_scheduler(0.1);
        // seed best_loss
        sched.advance(1, 2.0, 1);
        // improvement of exactly 0.1 should fire
        let id = sched.advance(2, 1.9, 2);
        assert!(id.is_some());
    }

    #[test]
    fn loss_improvement_does_not_fire_for_insufficient_improvement() {
        let mut sched = loss_scheduler(0.1);
        sched.advance(1, 2.0, 1); // seeds best_loss = 2.0
                                  // improvement is only 0.05 < 0.1
        assert!(sched.advance(2, 1.95, 2).is_none());
    }

    #[test]
    fn loss_improvement_does_not_fire_when_loss_worsens() {
        let mut sched = loss_scheduler(0.01);
        sched.advance(1, 1.0, 1); // best_loss = 1.0
        assert!(sched.advance(2, 1.5, 2).is_none());
    }

    #[test]
    fn loss_improvement_stats_incremented() {
        let mut sched = loss_scheduler(0.0);
        sched.advance(1, 1.0, 1);
        assert_eq!(sched.stats().triggered_by_loss, 1);
    }

    #[test]
    fn loss_improvement_best_loss_updated_correctly() {
        let mut sched = loss_scheduler(0.1);
        sched.advance(1, 5.0, 1); // best_loss becomes 5.0
        assert!((sched.best_loss - 5.0).abs() < f64::EPSILON);
        sched.advance(2, 4.5, 2); // improvement = 0.5 >= 0.1 → fires, best_loss → 4.5
                                  // 4.5 < 5.0 so best_loss is updated
        assert!((sched.best_loss - 4.5).abs() < f64::EPSILON);
    }

    #[test]
    fn loss_improvement_best_loss_not_updated_when_worse() {
        let mut sched = loss_scheduler(0.5);
        sched.advance(1, 1.0, 1); // best_loss = 1.0
        sched.advance(2, 1.5, 2); // no trigger, loss is worse
        assert!((sched.best_loss - 1.0).abs() < f64::EPSILON);
    }

    // ── TickInterval ────────────────────────────────────────────────────────

    #[test]
    fn tick_interval_fires_at_correct_tick() {
        let mut sched = tick_scheduler(100);
        assert!(sched.advance(1, 1.0, 100).is_some());
    }

    #[test]
    fn tick_interval_does_not_fire_before_interval() {
        let mut sched = tick_scheduler(100);
        assert!(sched.advance(1, 1.0, 99).is_none());
    }

    #[test]
    fn tick_interval_does_not_fire_at_tick_zero() {
        let mut sched = tick_scheduler(1);
        assert!(sched.advance(1, 1.0, 0).is_none());
    }

    #[test]
    fn tick_interval_fires_multiple_times() {
        let mut sched = tick_scheduler(50);
        let id1 = sched.advance(1, 1.0, 50);
        assert!(id1.is_some());
        assert!(sched.advance(2, 0.9, 99).is_none());
        let id2 = sched.advance(3, 0.9, 100);
        assert!(id2.is_some());
        assert_ne!(
            id1.expect("test: should succeed"),
            id2.expect("test: should succeed")
        );
    }

    #[test]
    fn tick_interval_stats_incremented() {
        let mut sched = tick_scheduler(10);
        sched.advance(1, 1.0, 10);
        assert_eq!(sched.stats().triggered_by_tick, 1);
        assert_eq!(sched.stats().total_checkpoints_saved, 1);
    }

    // ── Multiple triggers ───────────────────────────────────────────────────

    #[test]
    fn multiple_triggers_first_matching_fires() {
        // Step fires at step 5; Loss fires on first call (best_loss == MAX).
        // If order is [Step, Loss], first evaluation is Step. Step not satisfied
        // (step 0), then Loss fires. But since step=0 doesn't satisfy StepInterval,
        // LossImprovement fires.
        let config = SchedulerConfig {
            triggers: vec![
                CheckpointTrigger::StepInterval { every_n_steps: 5 },
                CheckpointTrigger::LossImprovement { min_delta: 0.0 },
            ],
            max_checkpoints: 10,
        };
        let sched = TensorCheckpointScheduler::new(config);
        // step=0 → StepInterval not satisfied; LossImprovement fires (best_loss==MAX)
        let trigger = sched.should_checkpoint(0, 1.0, 0);
        assert_eq!(
            trigger,
            Some(CheckpointTrigger::LossImprovement { min_delta: 0.0 })
        );
    }

    #[test]
    fn multiple_triggers_step_fires_before_loss_when_first() {
        // Order is [Step, Loss]. At step=5, Step fires first.
        let config = SchedulerConfig {
            triggers: vec![
                CheckpointTrigger::StepInterval { every_n_steps: 5 },
                CheckpointTrigger::LossImprovement { min_delta: 0.0 },
            ],
            max_checkpoints: 10,
        };
        let mut sched = TensorCheckpointScheduler::new(config);
        // Seed best_loss with a large value at step 0 indirectly:
        // manually set best_loss to skip the MAX guard.
        sched.best_loss = 10.0;
        // Now at step=5, StepInterval fires. LossImprovement would not fire
        // (10.0 - 9.5 = 0.5 >= 0.0 so it would too, but Step is checked first).
        let trigger = sched.should_checkpoint(5, 9.5, 0);
        assert_eq!(
            trigger,
            Some(CheckpointTrigger::StepInterval { every_n_steps: 5 })
        );
    }

    #[test]
    fn multiple_triggers_neither_fires_returns_none() {
        let config = SchedulerConfig {
            triggers: vec![
                CheckpointTrigger::StepInterval { every_n_steps: 10 },
                CheckpointTrigger::TickInterval { every_n_ticks: 100 },
            ],
            max_checkpoints: 5,
        };
        let mut sched = TensorCheckpointScheduler::new(config);
        // Seed last positions so intervals are not met.
        sched.last_checkpoint_step = 5;
        sched.last_checkpoint_tick = 50;
        assert!(sched.should_checkpoint(9, 1.0, 99).is_none());
    }

    // ── max_checkpoints pruning ─────────────────────────────────────────────

    #[test]
    fn max_checkpoints_pruning_oldest_removed() {
        let config = SchedulerConfig {
            triggers: vec![CheckpointTrigger::StepInterval { every_n_steps: 1 }],
            max_checkpoints: 3,
        };
        let mut sched = TensorCheckpointScheduler::new(config);
        for step in 1u64..=4 {
            sched.advance(step, 1.0, step);
        }
        assert_eq!(sched.checkpoints.len(), 3);
        // Oldest (step 1) should be gone; first kept is step 2.
        assert_eq!(sched.checkpoints[0].step, 2);
    }

    #[test]
    fn checkpoints_pruned_stat_incremented() {
        let config = SchedulerConfig {
            triggers: vec![CheckpointTrigger::StepInterval { every_n_steps: 1 }],
            max_checkpoints: 2,
        };
        let mut sched = TensorCheckpointScheduler::new(config);
        for step in 1u64..=4 {
            sched.advance(step, 1.0, step);
        }
        // 4 saved, 2 max → 2 pruned
        assert_eq!(sched.stats().checkpoints_pruned, 2);
    }

    #[test]
    fn no_pruning_when_within_limit() {
        let config = SchedulerConfig {
            triggers: vec![CheckpointTrigger::StepInterval { every_n_steps: 1 }],
            max_checkpoints: 10,
        };
        let mut sched = TensorCheckpointScheduler::new(config);
        for step in 1u64..=5 {
            sched.advance(step, 1.0, step);
        }
        assert_eq!(sched.stats().checkpoints_pruned, 0);
        assert_eq!(sched.checkpoints.len(), 5);
    }

    // ── advance / latest_checkpoint / stats ─────────────────────────────────

    #[test]
    fn advance_returns_checkpoint_id_when_triggered() {
        let mut sched = step_scheduler(5);
        let id = sched.advance(5, 1.0, 0);
        assert_eq!(id, Some(0));
    }

    #[test]
    fn advance_returns_none_when_no_trigger() {
        let mut sched = step_scheduler(10);
        assert!(sched.advance(5, 1.0, 0).is_none());
    }

    #[test]
    fn advance_increments_checkpoint_ids_sequentially() {
        let mut sched = step_scheduler(1);
        let id0 = sched.advance(1, 1.0, 0).expect("test: should succeed");
        let id1 = sched.advance(2, 0.9, 0).expect("test: should succeed");
        let id2 = sched.advance(3, 0.8, 0).expect("test: should succeed");
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn latest_checkpoint_returns_last_added() {
        let mut sched = step_scheduler(5);
        assert!(sched.latest_checkpoint().is_none());
        sched.advance(5, 2.0, 10);
        sched.advance(10, 1.5, 20);
        let latest = sched.latest_checkpoint().expect("should exist");
        assert_eq!(latest.step, 10);
        assert!((latest.loss - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn latest_checkpoint_none_when_no_checkpoints() {
        let sched = step_scheduler(10);
        assert!(sched.latest_checkpoint().is_none());
    }

    #[test]
    fn stats_returns_reference_to_stats() {
        let sched = step_scheduler(5);
        let s = sched.stats();
        assert_eq!(s.total_checkpoints_saved, 0);
        assert_eq!(s.checkpoints_pruned, 0);
    }

    #[test]
    fn checkpoint_record_fields_populated_correctly() {
        let mut sched = step_scheduler(5);
        sched.advance(5, 0.42, 17);
        let rec = sched.latest_checkpoint().expect("should exist");
        assert_eq!(rec.checkpoint_id, 0);
        assert_eq!(rec.step, 5);
        assert!((rec.loss - 0.42).abs() < 1e-10);
        assert_eq!(rec.saved_at_tick, 17);
        assert_eq!(
            rec.trigger,
            CheckpointTrigger::StepInterval { every_n_steps: 5 }
        );
    }

    #[test]
    fn scheduler_config_default() {
        let cfg = SchedulerConfig::default();
        assert!(cfg.triggers.is_empty());
        assert_eq!(cfg.max_checkpoints, 5);
    }

    #[test]
    fn no_triggers_never_checkpoints() {
        let config = SchedulerConfig::default();
        let mut sched = TensorCheckpointScheduler::new(config);
        for step in 1u64..=100 {
            assert!(sched.advance(step, 1.0 / step as f64, step).is_none());
        }
        assert_eq!(sched.stats().total_checkpoints_saved, 0);
    }

    #[test]
    fn mixed_trigger_types_all_stats_accurate() {
        // StepInterval(10) checked before TickInterval(20).
        // record_checkpoint always updates BOTH cursors regardless of trigger type.
        let config = SchedulerConfig {
            triggers: vec![
                CheckpointTrigger::StepInterval { every_n_steps: 10 },
                CheckpointTrigger::TickInterval { every_n_ticks: 20 },
            ],
            max_checkpoints: 100,
        };
        let mut sched = TensorCheckpointScheduler::new(config);

        // advance(5, 1.0, 20):
        //   StepInterval: 5-0=5 < 10 → no
        //   TickInterval: 20-0=20 >= 20 → fires
        //   After: last_step=5, last_tick=20, triggered_by_tick=1
        sched.advance(5, 1.0, 20);

        // advance(10, 0.9, 25):
        //   StepInterval: 10-5=5 < 10 → no
        //   TickInterval: 25-20=5 < 20 → no
        //   No checkpoint.
        let r = sched.advance(10, 0.9, 25);
        assert!(r.is_none());

        // advance(15, 0.8, 40):
        //   StepInterval: 15-5=10 >= 10 → fires
        //   After: last_step=15, last_tick=40, triggered_by_step=1
        sched.advance(15, 0.8, 40);

        // advance(20, 0.7, 60):
        //   StepInterval: 20-15=5 < 10 → no
        //   TickInterval: 60-40=20 >= 20 → fires
        //   After: last_tick=60, triggered_by_tick=2
        sched.advance(20, 0.7, 60);

        assert_eq!(sched.stats().triggered_by_step, 1);
        assert_eq!(sched.stats().triggered_by_tick, 2);
        assert_eq!(sched.stats().triggered_by_loss, 0);
        assert_eq!(sched.stats().total_checkpoints_saved, 3);
    }
}
