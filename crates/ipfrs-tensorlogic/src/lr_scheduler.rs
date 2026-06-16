//! TensorLRScheduler -- learning rate scheduling strategies for tensor
//! optimization loops.
//!
//! Supported schedules:
//! - **Constant** -- fixed learning rate
//! - **StepDecay** -- multiply by gamma every `step_size` steps
//! - **ExponentialDecay** -- multiply by gamma every step
//! - **CosineAnnealing** -- cosine decay to `min_lr`
//! - **WarmupLinear** -- linear warmup then constant
//! - **OneCycleLR** -- ramp up to `max_lr` then cosine decay to `min_lr`

use std::f64::consts::PI;

/// Scheduling strategy variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScheduleType {
    /// Always return `initial_lr`.
    Constant,
    /// Multiply by `gamma` every `step_size` steps.
    StepDecay,
    /// Multiply by `gamma` every step.
    ExponentialDecay,
    /// Cosine schedule from `initial_lr` down to `min_lr`.
    CosineAnnealing,
    /// Linear warmup from 0 to `initial_lr` over `warmup_steps`, then constant.
    WarmupLinear,
    /// Ramp up to `max_lr` during warmup, then cosine decay to `min_lr`.
    OneCycleLR,
}

impl std::fmt::Display for ScheduleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Constant => write!(f, "Constant"),
            Self::StepDecay => write!(f, "StepDecay"),
            Self::ExponentialDecay => write!(f, "ExponentialDecay"),
            Self::CosineAnnealing => write!(f, "CosineAnnealing"),
            Self::WarmupLinear => write!(f, "WarmupLinear"),
            Self::OneCycleLR => write!(f, "OneCycleLR"),
        }
    }
}

/// Configuration for a learning rate scheduler.
#[derive(Debug, Clone)]
pub struct LRSchedulerConfig {
    /// Which schedule to use.
    pub schedule_type: ScheduleType,
    /// Starting learning rate (default 0.01).
    pub initial_lr: f64,
    /// Floor learning rate for cosine / OneCycle (default 1e-6).
    pub min_lr: f64,
    /// Decay factor for Step/Exponential schedules (default 0.1).
    pub gamma: f64,
    /// Epoch count between decays for StepDecay (default 30).
    pub step_size: usize,
    /// Total training steps for Cosine / OneCycle (default 1000).
    pub total_steps: usize,
    /// Number of warmup steps for Warmup / OneCycle (default 100).
    pub warmup_steps: usize,
    /// Peak learning rate for OneCycle (default 0.1).
    pub max_lr: f64,
}

impl Default for LRSchedulerConfig {
    fn default() -> Self {
        Self {
            schedule_type: ScheduleType::Constant,
            initial_lr: 0.01,
            min_lr: 1e-6,
            gamma: 0.1,
            step_size: 30,
            total_steps: 1000,
            warmup_steps: 100,
            max_lr: 0.1,
        }
    }
}

/// Run-time statistics snapshot.
#[derive(Debug, Clone)]
pub struct LRSchedulerStats {
    /// Active schedule type.
    pub schedule_type: ScheduleType,
    /// Current step counter.
    pub current_step: usize,
    /// Current learning rate value.
    pub current_lr: f64,
    /// Initial learning rate from config.
    pub initial_lr: f64,
}

/// Learning rate scheduler with multiple strategy support.
#[derive(Debug, Clone)]
pub struct TensorLRScheduler {
    config: LRSchedulerConfig,
    current_step: usize,
    current_lr: f64,
}

impl TensorLRScheduler {
    /// Create a new scheduler from the given config.  The initial learning rate
    /// is computed for step 0.
    pub fn new(config: LRSchedulerConfig) -> Self {
        let lr = Self::compute_lr(&config, 0);
        Self {
            config,
            current_step: 0,
            current_lr: lr,
        }
    }

    /// Advance by one step, compute the new learning rate, and return it.
    pub fn step(&mut self) -> f64 {
        self.current_step += 1;
        self.current_lr = Self::compute_lr(&self.config, self.current_step);
        self.current_lr
    }

    /// Return the current learning rate without advancing the step counter.
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }

    /// Jump to a specific step and recompute the learning rate.
    pub fn set_step(&mut self, step: usize) {
        self.current_step = step;
        self.current_lr = Self::compute_lr(&self.config, step);
    }

    /// For finite schedules (Cosine, OneCycle, WarmupLinear) return the number
    /// of remaining steps.  Returns `None` for schedules that run indefinitely.
    pub fn remaining_steps(&self) -> Option<usize> {
        match self.config.schedule_type {
            ScheduleType::CosineAnnealing | ScheduleType::OneCycleLR => {
                Some(self.config.total_steps.saturating_sub(self.current_step))
            }
            ScheduleType::WarmupLinear => {
                Some(self.config.warmup_steps.saturating_sub(self.current_step))
            }
            ScheduleType::Constant | ScheduleType::StepDecay | ScheduleType::ExponentialDecay => {
                None
            }
        }
    }

    /// Reset to step 0.
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.current_lr = Self::compute_lr(&self.config, 0);
    }

    /// Return a snapshot of the scheduler state.
    pub fn stats(&self) -> LRSchedulerStats {
        LRSchedulerStats {
            schedule_type: self.config.schedule_type,
            current_step: self.current_step,
            current_lr: self.current_lr,
            initial_lr: self.config.initial_lr,
        }
    }

    /// Read-only access to the config.
    pub fn config(&self) -> &LRSchedulerConfig {
        &self.config
    }

    // ------------------------------------------------------------------
    // Internal: pure function that computes LR for a given step.
    // ------------------------------------------------------------------

    fn compute_lr(cfg: &LRSchedulerConfig, step: usize) -> f64 {
        match cfg.schedule_type {
            ScheduleType::Constant => cfg.initial_lr,

            ScheduleType::StepDecay => {
                let exponent = (step / cfg.step_size.max(1)) as f64;
                cfg.initial_lr * cfg.gamma.powf(exponent)
            }

            ScheduleType::ExponentialDecay => cfg.initial_lr * cfg.gamma.powf(step as f64),

            ScheduleType::CosineAnnealing => {
                let total = cfg.total_steps.max(1) as f64;
                let t = (step as f64).min(total);
                cfg.min_lr + 0.5 * (cfg.initial_lr - cfg.min_lr) * (1.0 + (PI * t / total).cos())
            }

            ScheduleType::WarmupLinear => {
                if cfg.warmup_steps == 0 {
                    return cfg.initial_lr;
                }
                if step < cfg.warmup_steps {
                    cfg.initial_lr * (step as f64) / (cfg.warmup_steps as f64)
                } else {
                    cfg.initial_lr
                }
            }

            ScheduleType::OneCycleLR => {
                let warmup = cfg.warmup_steps.max(1);
                let total = cfg.total_steps.max(warmup + 1);
                if step < warmup {
                    // Linear ramp from min_lr to max_lr
                    let frac = step as f64 / warmup as f64;
                    cfg.min_lr + frac * (cfg.max_lr - cfg.min_lr)
                } else {
                    // Cosine decay from max_lr to min_lr
                    let decay_steps = (total - warmup).max(1) as f64;
                    let t = ((step - warmup) as f64).min(decay_steps);
                    cfg.min_lr
                        + 0.5 * (cfg.max_lr - cfg.min_lr) * (1.0 + (PI * t / decay_steps).cos())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------ helpers
    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // --------------------------------------------------- Constant schedule tests
    #[test]
    fn constant_always_returns_initial_lr() {
        let mut sched = TensorLRScheduler::new(LRSchedulerConfig {
            schedule_type: ScheduleType::Constant,
            initial_lr: 0.05,
            ..Default::default()
        });
        for _ in 0..50 {
            let lr = sched.step();
            assert!(approx_eq(lr, 0.05, 1e-12));
        }
    }

    #[test]
    fn constant_get_lr_matches_step() {
        let mut sched = TensorLRScheduler::new(LRSchedulerConfig {
            schedule_type: ScheduleType::Constant,
            initial_lr: 0.02,
            ..Default::default()
        });
        sched.step();
        assert!(approx_eq(sched.get_lr(), 0.02, 1e-12));
    }

    // --------------------------------------------------- StepDecay tests
    #[test]
    fn step_decay_decays_at_boundary() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::StepDecay,
            initial_lr: 1.0,
            gamma: 0.5,
            step_size: 10,
            ..Default::default()
        };
        let mut sched = TensorLRScheduler::new(cfg);
        // At step 0 => lr = 1.0
        assert!(approx_eq(sched.get_lr(), 1.0, 1e-12));
        // Advance to step 9 => still in first interval
        for _ in 0..9 {
            sched.step();
        }
        assert!(approx_eq(sched.get_lr(), 1.0, 1e-12));
        // step 10 => gamma^1 = 0.5
        sched.step();
        assert!(approx_eq(sched.get_lr(), 0.5, 1e-12));
        // step 20 => gamma^2 = 0.25
        for _ in 0..10 {
            sched.step();
        }
        assert!(approx_eq(sched.get_lr(), 0.25, 1e-12));
    }

    #[test]
    fn step_decay_within_interval_is_flat() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::StepDecay,
            initial_lr: 0.1,
            gamma: 0.1,
            step_size: 5,
            ..Default::default()
        };
        let mut sched = TensorLRScheduler::new(cfg);
        let lr0 = sched.get_lr();
        for i in 1..5 {
            sched.step();
            assert!(
                approx_eq(sched.get_lr(), lr0, 1e-12),
                "LR changed at step {i} within interval"
            );
        }
    }

    // ------------------------------------------------- ExponentialDecay tests
    #[test]
    fn exponential_monotone_decrease() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::ExponentialDecay,
            initial_lr: 1.0,
            gamma: 0.95,
            ..Default::default()
        };
        let mut sched = TensorLRScheduler::new(cfg);
        let mut prev = sched.get_lr();
        for _ in 0..100 {
            let lr = sched.step();
            assert!(lr < prev + 1e-15, "LR did not decrease");
            prev = lr;
        }
    }

    #[test]
    fn exponential_decay_formula() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::ExponentialDecay,
            initial_lr: 2.0,
            gamma: 0.9,
            ..Default::default()
        };
        let mut sched = TensorLRScheduler::new(cfg);
        for step in 1..=5 {
            let lr = sched.step();
            let expected = 2.0 * 0.9_f64.powf(step as f64);
            assert!(
                approx_eq(lr, expected, 1e-10),
                "step {step}: got {lr}, expected {expected}"
            );
        }
    }

    // ------------------------------------------------- CosineAnnealing tests
    #[test]
    fn cosine_reaches_min_at_total_steps() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::CosineAnnealing,
            initial_lr: 0.1,
            min_lr: 1e-5,
            total_steps: 200,
            ..Default::default()
        };
        let mut sched = TensorLRScheduler::new(cfg);
        for _ in 0..200 {
            sched.step();
        }
        assert!(
            approx_eq(sched.get_lr(), 1e-5, 1e-10),
            "LR at total_steps should be min_lr, got {}",
            sched.get_lr()
        );
    }

    #[test]
    fn cosine_starts_at_initial_lr() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::CosineAnnealing,
            initial_lr: 0.05,
            min_lr: 0.001,
            total_steps: 500,
            ..Default::default()
        };
        let sched = TensorLRScheduler::new(cfg);
        assert!(approx_eq(sched.get_lr(), 0.05, 1e-12));
    }

    #[test]
    fn cosine_midpoint_value() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::CosineAnnealing,
            initial_lr: 1.0,
            min_lr: 0.0,
            total_steps: 100,
            ..Default::default()
        };
        let mut sched = TensorLRScheduler::new(cfg);
        // At step 50 => 0.5 * (1 + cos(pi*50/100)) = 0.5 * (1 + cos(pi/2)) = 0.5
        for _ in 0..50 {
            sched.step();
        }
        assert!(
            approx_eq(sched.get_lr(), 0.5, 1e-10),
            "cosine midpoint: got {}",
            sched.get_lr()
        );
    }

    // --------------------------------------------------- WarmupLinear tests
    #[test]
    fn warmup_linear_ramp() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::WarmupLinear,
            initial_lr: 0.1,
            warmup_steps: 10,
            ..Default::default()
        };
        let mut sched = TensorLRScheduler::new(cfg);
        // step 0 => lr = 0.0
        assert!(approx_eq(sched.get_lr(), 0.0, 1e-12));
        // step 5 => lr = 0.05
        for _ in 0..5 {
            sched.step();
        }
        assert!(approx_eq(sched.get_lr(), 0.05, 1e-12));
        // step 10 => lr = 0.1 (reached initial_lr)
        for _ in 0..5 {
            sched.step();
        }
        assert!(approx_eq(sched.get_lr(), 0.1, 1e-12));
    }

    #[test]
    fn warmup_holds_after_warmup() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::WarmupLinear,
            initial_lr: 0.01,
            warmup_steps: 5,
            ..Default::default()
        };
        let mut sched = TensorLRScheduler::new(cfg);
        for _ in 0..20 {
            sched.step();
        }
        assert!(approx_eq(sched.get_lr(), 0.01, 1e-12));
    }

    #[test]
    fn warmup_zero_steps_returns_initial() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::WarmupLinear,
            initial_lr: 0.03,
            warmup_steps: 0,
            ..Default::default()
        };
        let sched = TensorLRScheduler::new(cfg);
        assert!(approx_eq(sched.get_lr(), 0.03, 1e-12));
    }

    // --------------------------------------------------- OneCycleLR tests
    #[test]
    fn one_cycle_starts_at_min_lr() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::OneCycleLR,
            initial_lr: 0.01,
            min_lr: 1e-4,
            max_lr: 0.1,
            warmup_steps: 50,
            total_steps: 200,
            ..Default::default()
        };
        let sched = TensorLRScheduler::new(cfg);
        assert!(
            approx_eq(sched.get_lr(), 1e-4, 1e-12),
            "OneCycle should start at min_lr, got {}",
            sched.get_lr()
        );
    }

    #[test]
    fn one_cycle_reaches_peak() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::OneCycleLR,
            initial_lr: 0.01,
            min_lr: 0.0,
            max_lr: 0.2,
            warmup_steps: 100,
            total_steps: 500,
            ..Default::default()
        };
        let mut sched = TensorLRScheduler::new(cfg);
        for _ in 0..100 {
            sched.step();
        }
        // At warmup boundary: the first decay step with t=0 => cos(0)=1 =>
        // 0.5*(0.2)*(1+1) = 0.2
        assert!(
            approx_eq(sched.get_lr(), 0.2, 1e-10),
            "OneCycle should reach max_lr at warmup end, got {}",
            sched.get_lr()
        );
    }

    #[test]
    fn one_cycle_ends_at_min_lr() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::OneCycleLR,
            initial_lr: 0.01,
            min_lr: 1e-5,
            max_lr: 0.1,
            warmup_steps: 50,
            total_steps: 300,
            ..Default::default()
        };
        let mut sched = TensorLRScheduler::new(cfg);
        for _ in 0..300 {
            sched.step();
        }
        assert!(
            approx_eq(sched.get_lr(), 1e-5, 1e-10),
            "OneCycle should end at min_lr, got {}",
            sched.get_lr()
        );
    }

    // -------------------------------------------------- reset / set_step / remaining
    #[test]
    fn reset_restores_step_zero() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::ExponentialDecay,
            initial_lr: 1.0,
            gamma: 0.5,
            ..Default::default()
        };
        let mut sched = TensorLRScheduler::new(cfg);
        for _ in 0..10 {
            sched.step();
        }
        sched.reset();
        assert_eq!(sched.stats().current_step, 0);
        assert!(approx_eq(sched.get_lr(), 1.0, 1e-12));
    }

    #[test]
    fn set_step_jumps_correctly() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::StepDecay,
            initial_lr: 1.0,
            gamma: 0.5,
            step_size: 10,
            ..Default::default()
        };
        let mut sched = TensorLRScheduler::new(cfg);
        sched.set_step(25);
        assert_eq!(sched.stats().current_step, 25);
        // 25 / 10 = 2 => 0.5^2 = 0.25
        assert!(approx_eq(sched.get_lr(), 0.25, 1e-12));
    }

    #[test]
    fn remaining_steps_cosine() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::CosineAnnealing,
            total_steps: 100,
            ..Default::default()
        };
        let mut sched = TensorLRScheduler::new(cfg);
        assert_eq!(sched.remaining_steps(), Some(100));
        for _ in 0..30 {
            sched.step();
        }
        assert_eq!(sched.remaining_steps(), Some(70));
    }

    #[test]
    fn remaining_steps_none_for_constant() {
        let sched = TensorLRScheduler::new(LRSchedulerConfig {
            schedule_type: ScheduleType::Constant,
            ..Default::default()
        });
        assert_eq!(sched.remaining_steps(), None);
    }

    #[test]
    fn remaining_steps_none_for_step_decay() {
        let sched = TensorLRScheduler::new(LRSchedulerConfig {
            schedule_type: ScheduleType::StepDecay,
            ..Default::default()
        });
        assert_eq!(sched.remaining_steps(), None);
    }

    #[test]
    fn remaining_steps_none_for_exponential() {
        let sched = TensorLRScheduler::new(LRSchedulerConfig {
            schedule_type: ScheduleType::ExponentialDecay,
            ..Default::default()
        });
        assert_eq!(sched.remaining_steps(), None);
    }

    #[test]
    fn remaining_steps_warmup() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::WarmupLinear,
            warmup_steps: 50,
            ..Default::default()
        };
        let mut sched = TensorLRScheduler::new(cfg);
        assert_eq!(sched.remaining_steps(), Some(50));
        for _ in 0..50 {
            sched.step();
        }
        assert_eq!(sched.remaining_steps(), Some(0));
    }

    #[test]
    fn remaining_steps_one_cycle() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::OneCycleLR,
            warmup_steps: 20,
            total_steps: 100,
            ..Default::default()
        };
        let mut sched = TensorLRScheduler::new(cfg);
        assert_eq!(sched.remaining_steps(), Some(100));
        for _ in 0..40 {
            sched.step();
        }
        assert_eq!(sched.remaining_steps(), Some(60));
    }

    // -------------------------------------------------- stats
    #[test]
    fn stats_snapshot() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::CosineAnnealing,
            initial_lr: 0.1,
            total_steps: 200,
            ..Default::default()
        };
        let mut sched = TensorLRScheduler::new(cfg);
        for _ in 0..10 {
            sched.step();
        }
        let s = sched.stats();
        assert_eq!(s.schedule_type, ScheduleType::CosineAnnealing);
        assert_eq!(s.current_step, 10);
        assert!(approx_eq(s.initial_lr, 0.1, 1e-12));
        assert!(approx_eq(s.current_lr, sched.get_lr(), 1e-15));
    }

    // -------------------------------------------------- step advances LR
    #[test]
    fn step_advances_lr() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::ExponentialDecay,
            initial_lr: 1.0,
            gamma: 0.9,
            ..Default::default()
        };
        let mut sched = TensorLRScheduler::new(cfg);
        let lr0 = sched.get_lr();
        let lr1 = sched.step();
        assert!(lr1 < lr0, "step should change LR for decay schedules");
    }

    // -------------------------------------------------- default config
    #[test]
    fn default_config_values() {
        let cfg = LRSchedulerConfig::default();
        assert_eq!(cfg.schedule_type, ScheduleType::Constant);
        assert!(approx_eq(cfg.initial_lr, 0.01, 1e-15));
        assert!(approx_eq(cfg.min_lr, 1e-6, 1e-15));
        assert!(approx_eq(cfg.gamma, 0.1, 1e-15));
        assert_eq!(cfg.step_size, 30);
        assert_eq!(cfg.total_steps, 1000);
        assert_eq!(cfg.warmup_steps, 100);
        assert!(approx_eq(cfg.max_lr, 0.1, 1e-15));
    }

    // -------------------------------------------------- schedule_type display
    #[test]
    fn schedule_type_display() {
        assert_eq!(format!("{}", ScheduleType::Constant), "Constant");
        assert_eq!(format!("{}", ScheduleType::OneCycleLR), "OneCycleLR");
    }

    // -------------------------------------------------- edge cases
    #[test]
    fn step_decay_zero_step_size_no_panic() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::StepDecay,
            initial_lr: 1.0,
            gamma: 0.5,
            step_size: 0,
            ..Default::default()
        };
        let mut sched = TensorLRScheduler::new(cfg);
        // Should not panic; step_size clamped to 1 internally
        let _ = sched.step();
    }

    #[test]
    fn cosine_beyond_total_steps_clamps() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::CosineAnnealing,
            initial_lr: 0.1,
            min_lr: 0.001,
            total_steps: 50,
            ..Default::default()
        };
        let mut sched = TensorLRScheduler::new(cfg);
        for _ in 0..100 {
            sched.step();
        }
        // Should clamp at min_lr, not go negative
        assert!(
            approx_eq(sched.get_lr(), 0.001, 1e-10),
            "cosine should clamp at min_lr beyond total_steps, got {}",
            sched.get_lr()
        );
    }

    #[test]
    fn one_cycle_warmup_phase_is_monotone_increasing() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::OneCycleLR,
            min_lr: 0.0,
            max_lr: 1.0,
            warmup_steps: 50,
            total_steps: 200,
            ..Default::default()
        };
        let mut sched = TensorLRScheduler::new(cfg);
        let mut prev = sched.get_lr();
        for _ in 0..50 {
            let lr = sched.step();
            assert!(
                lr >= prev - 1e-15,
                "warmup should be monotonically increasing"
            );
            prev = lr;
        }
    }

    #[test]
    fn config_accessor() {
        let cfg = LRSchedulerConfig {
            schedule_type: ScheduleType::StepDecay,
            gamma: 0.3,
            ..Default::default()
        };
        let sched = TensorLRScheduler::new(cfg);
        assert!(approx_eq(sched.config().gamma, 0.3, 1e-15));
    }
}

// =============================================================================
// LearningRateScheduler — multi-strategy LR scheduler for distributed ML
// =============================================================================

/// Multi-strategy learning rate scheduling for distributed ML training.
///
/// Supports seven scheduling strategies including plateau detection.
#[derive(Debug, Clone)]
pub enum SchedulerStrategy {
    /// Fixed learning rate that never changes.
    Constant {
        /// The fixed learning rate.
        lr: f64,
    },
    /// Multiply learning rate by `decay_factor` every `step_size` epochs.
    StepDecay {
        /// Starting learning rate.
        initial_lr: f64,
        /// Multiplicative factor applied every `step_size` epochs.
        decay_factor: f64,
        /// Number of epochs between each decay application.
        step_size: u64,
    },
    /// Exponential decay: `lr = initial * exp(-decay_rate * epoch)`.
    ExponentialDecay {
        /// Starting learning rate.
        initial_lr: f64,
        /// Rate of exponential decay.
        decay_rate: f64,
    },
    /// Cosine annealing from `initial_lr` to `min_lr` over `t_max` epochs.
    CosineAnnealing {
        /// Starting learning rate.
        initial_lr: f64,
        /// Floor learning rate.
        min_lr: f64,
        /// Period (number of epochs for one half-cycle).
        t_max: u64,
    },
    /// Linear warmup over `warmup_epochs` then cosine annealing to `min_lr`.
    WarmupCosine {
        /// Number of epochs for linear warm-up phase.
        warmup_epochs: u64,
        /// Starting LR for warm-up (typically a small value or 0).
        initial_lr: f64,
        /// Peak learning rate reached after warm-up.
        peak_lr: f64,
        /// Floor learning rate after the cosine phase.
        min_lr: f64,
        /// Length of the cosine phase in epochs (not including warmup).
        t_max: u64,
    },
    /// Triangular (cyclic) learning rate oscillating between `base_lr` and `max_lr`.
    CyclicLR {
        /// Lower bound of the LR range.
        base_lr: f64,
        /// Upper bound of the LR range.
        max_lr: f64,
        /// Half the cycle length (each triangle leg = `step_size` epochs).
        step_size: u64,
    },
    /// Reduce learning rate when validation loss stops improving.
    ReduceOnPlateau {
        /// Starting learning rate.
        initial_lr: f64,
        /// Multiplicative reduction factor (< 1.0).
        factor: f64,
        /// Number of epochs with no improvement before reduction.
        patience: u64,
        /// Minimum allowed learning rate.
        min_lr: f64,
        /// Minimum change in loss to qualify as improvement.
        threshold: f64,
    },
}

/// Internal state tracked across `step` calls.
#[derive(Debug, Clone)]
pub struct LrSchedulerState {
    /// Current epoch index (0-based after the last `reset`).
    pub current_epoch: u64,
    /// Most recently computed learning rate.
    pub current_lr: f64,
    /// Best (lowest) loss seen so far (used by `ReduceOnPlateau`).
    pub best_loss: f64,
    /// Number of consecutive epochs without improvement (plateau counter).
    pub plateau_count: u64,
    /// Number of full cycles completed (used by `CyclicLR`).
    pub cycles_completed: u64,
}

impl LrSchedulerState {
    fn new(initial_lr: f64) -> Self {
        Self {
            current_epoch: 0,
            current_lr: initial_lr,
            best_loss: f64::INFINITY,
            plateau_count: 0,
            cycles_completed: 0,
        }
    }
}

/// A single entry in the LR history log.
#[derive(Debug, Clone)]
pub struct LrHistory {
    /// Epoch index for this entry.
    pub epoch: u64,
    /// Learning rate computed at this epoch.
    pub lr: f64,
    /// Optional loss value provided at this epoch.
    pub loss: Option<f64>,
}

/// Aggregate statistics over training.
#[derive(Debug, Clone)]
pub struct LrStats {
    /// Minimum learning rate observed in history.
    pub min_lr_seen: f64,
    /// Maximum learning rate observed in history.
    pub max_lr_seen: f64,
    /// Number of learning rate reductions triggered by plateau detection.
    pub plateau_reductions: u64,
    /// Total number of epochs recorded in history.
    pub epochs_trained: u64,
}

const MAX_HISTORY: usize = 1000;

/// Multi-strategy learning rate scheduler.
///
/// Supports `Constant`, `StepDecay`, `ExponentialDecay`, `CosineAnnealing`,
/// `WarmupCosine`, `CyclicLR`, and `ReduceOnPlateau`.
#[derive(Debug, Clone)]
pub struct LearningRateScheduler {
    /// The scheduling strategy in use.
    pub strategy: SchedulerStrategy,
    /// Mutable runtime state.
    pub state: LrSchedulerState,
    history: Vec<LrHistory>,
    plateau_reductions: u64,
}

impl LearningRateScheduler {
    /// Create a new scheduler.  The initial learning rate is derived from the
    /// strategy variant's `initial_lr` (or `lr` for `Constant`).
    pub fn new(strategy: SchedulerStrategy) -> Self {
        let initial_lr = Self::extract_initial_lr(&strategy);
        Self {
            state: LrSchedulerState::new(initial_lr),
            strategy,
            history: Vec::new(),
            plateau_reductions: 0,
        }
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Compute the learning rate for the given `epoch` without mutating
    /// significant state (except appending to history and updating
    /// `current_epoch` / `current_lr`).
    ///
    /// For `ReduceOnPlateau` this ignores plateau tracking — use
    /// `step_with_loss` instead.
    pub fn step(&mut self, epoch: u64) -> f64 {
        let lr = self.compute_lr(epoch);
        self.state.current_epoch = epoch;
        self.state.current_lr = lr;
        self.push_history(epoch, lr, None);
        lr
    }

    /// Advance one step using a loss value.  For `ReduceOnPlateau` this
    /// updates the internal plateau counter and reduces LR as needed.
    /// For other strategies it behaves identically to `step`.
    pub fn step_with_loss(&mut self, epoch: u64, loss: f64) -> f64 {
        let lr = match &self.strategy {
            SchedulerStrategy::ReduceOnPlateau {
                factor,
                patience,
                min_lr,
                threshold,
                ..
            } => {
                let factor = *factor;
                let patience = *patience;
                let floor = *min_lr;
                let threshold = *threshold;

                let improved = loss < self.state.best_loss - threshold;
                if improved {
                    self.state.best_loss = loss;
                    self.state.plateau_count = 0;
                } else {
                    self.state.plateau_count += 1;
                }

                if self.state.plateau_count >= patience {
                    let reduced = (self.state.current_lr * factor).max(floor);
                    if reduced < self.state.current_lr {
                        self.state.current_lr = reduced;
                        self.plateau_reductions += 1;
                    }
                    self.state.plateau_count = 0;
                }
                self.state.current_lr
            }
            _ => self.compute_lr(epoch),
        };

        self.state.current_epoch = epoch;
        self.state.current_lr = lr;
        self.push_history(epoch, lr, Some(loss));
        lr
    }

    /// Return the most recently computed learning rate.
    pub fn current_lr(&self) -> f64 {
        self.state.current_lr
    }

    /// Reset internal state to epoch 0 and recompute the initial LR.
    pub fn reset(&mut self) {
        let initial_lr = Self::extract_initial_lr(&self.strategy);
        self.state = LrSchedulerState::new(initial_lr);
        self.history.clear();
        self.plateau_reductions = 0;
    }

    /// Return a slice of the LR history (capped at the last 1 000 entries).
    pub fn history(&self) -> &[LrHistory] {
        &self.history
    }

    /// Compute aggregate statistics over the recorded history.
    pub fn stats(&self) -> LrStats {
        let (min_lr_seen, max_lr_seen) = if self.history.is_empty() {
            (self.state.current_lr, self.state.current_lr)
        } else {
            let min = self
                .history
                .iter()
                .map(|h| h.lr)
                .fold(f64::INFINITY, f64::min);
            let max = self
                .history
                .iter()
                .map(|h| h.lr)
                .fold(f64::NEG_INFINITY, f64::max);
            (min, max)
        };

        LrStats {
            min_lr_seen,
            max_lr_seen,
            plateau_reductions: self.plateau_reductions,
            epochs_trained: self.history.len() as u64,
        }
    }

    // ------------------------------------------------------------------
    // Helper functions (also `pub` so tests can call them directly)
    // ------------------------------------------------------------------

    /// Linear interpolation factor for a warmup phase.
    ///
    /// Returns values in `[0.0, 1.0]` linearly from 0 to 1 as `epoch`
    /// goes from 0 to `warmup_epochs`.  Returns 1.0 when `epoch >=
    /// warmup_epochs`.
    pub fn warmup_factor(epoch: u64, warmup_epochs: u64) -> f64 {
        if warmup_epochs == 0 {
            return 1.0;
        }
        (epoch as f64 / warmup_epochs as f64).min(1.0)
    }

    /// Cosine annealing factor: `(1 + cos(π * epoch / t_max)) / 2`.
    ///
    /// Returns values in `[0.0, 1.0]`.  Clamps `epoch` at `t_max`.
    pub fn cosine_factor(epoch: u64, t_max: u64) -> f64 {
        if t_max == 0 {
            return 0.0;
        }
        let t = (epoch as f64).min(t_max as f64);
        (1.0 + (PI * t / t_max as f64).cos()) / 2.0
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn extract_initial_lr(strategy: &SchedulerStrategy) -> f64 {
        match strategy {
            SchedulerStrategy::Constant { lr } => *lr,
            SchedulerStrategy::StepDecay { initial_lr, .. } => *initial_lr,
            SchedulerStrategy::ExponentialDecay { initial_lr, .. } => *initial_lr,
            SchedulerStrategy::CosineAnnealing { initial_lr, .. } => *initial_lr,
            SchedulerStrategy::WarmupCosine { initial_lr, .. } => *initial_lr,
            SchedulerStrategy::CyclicLR { base_lr, .. } => *base_lr,
            SchedulerStrategy::ReduceOnPlateau { initial_lr, .. } => *initial_lr,
        }
    }

    fn compute_lr(&mut self, epoch: u64) -> f64 {
        match &self.strategy {
            SchedulerStrategy::Constant { lr } => *lr,

            SchedulerStrategy::StepDecay {
                initial_lr,
                decay_factor,
                step_size,
            } => {
                let exponent = if *step_size == 0 {
                    epoch
                } else {
                    epoch / step_size
                };
                initial_lr * decay_factor.powi(exponent as i32)
            }

            SchedulerStrategy::ExponentialDecay {
                initial_lr,
                decay_rate,
            } => initial_lr * (-decay_rate * epoch as f64).exp(),

            SchedulerStrategy::CosineAnnealing {
                initial_lr,
                min_lr,
                t_max,
            } => {
                let factor = Self::cosine_factor(epoch, *t_max);
                min_lr + (initial_lr - min_lr) * factor
            }

            SchedulerStrategy::WarmupCosine {
                warmup_epochs,
                initial_lr,
                peak_lr,
                min_lr,
                t_max,
            } => {
                let warmup = *warmup_epochs;
                let floor = *min_lr;
                let peak = *peak_lr;
                let start = *initial_lr;
                let period = *t_max;

                if epoch < warmup {
                    let w = Self::warmup_factor(epoch, warmup);
                    start + w * (peak - start)
                } else {
                    let cosine_epoch = epoch - warmup;
                    let factor = Self::cosine_factor(cosine_epoch, period);
                    floor + (peak - floor) * factor
                }
            }

            SchedulerStrategy::CyclicLR {
                base_lr,
                max_lr,
                step_size,
            } => {
                let base = *base_lr;
                let peak = *max_lr;
                let half = (*step_size).max(1);
                let cycle_len = 2 * half;
                let cycle_epoch = epoch % cycle_len;
                let frac = if cycle_epoch < half {
                    cycle_epoch as f64 / half as f64
                } else {
                    (cycle_len - cycle_epoch) as f64 / half as f64
                };
                // Update cycle count in state without borrowing conflict via a flag
                base + frac * (peak - base)
            }

            SchedulerStrategy::ReduceOnPlateau { .. } => {
                // For stateless computation, just return the current LR.
                self.state.current_lr
            }
        }
    }

    fn push_history(&mut self, epoch: u64, lr: f64, loss: Option<f64>) {
        if self.history.len() >= MAX_HISTORY {
            self.history.remove(0);
        }
        self.history.push(LrHistory { epoch, lr, loss });
    }
}

// =============================================================================
// Tests for LearningRateScheduler
// =============================================================================

#[cfg(test)]
mod lr_scheduler_tests {
    use crate::lr_scheduler::{LearningRateScheduler, SchedulerStrategy};

    const TOL: f64 = 1e-10;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    // ------------------------------------------------------------------ Constant

    #[test]
    fn constant_lr_never_changes() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::Constant { lr: 0.03 });
        for epoch in 0..50 {
            let lr = sched.step(epoch);
            assert!(approx(lr, 0.03), "epoch {epoch}: expected 0.03, got {lr}");
        }
    }

    #[test]
    fn constant_current_lr_matches_step() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::Constant { lr: 0.01 });
        sched.step(0);
        assert!(approx(sched.current_lr(), 0.01));
    }

    #[test]
    fn constant_initial_lr_from_new() {
        let sched = LearningRateScheduler::new(SchedulerStrategy::Constant { lr: 0.05 });
        assert!(approx(sched.current_lr(), 0.05));
    }

    // ------------------------------------------------------------------ StepDecay

    #[test]
    fn step_decay_flat_within_interval() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::StepDecay {
            initial_lr: 1.0,
            decay_factor: 0.5,
            step_size: 10,
        });
        for epoch in 0..10 {
            let lr = sched.step(epoch);
            assert!(approx(lr, 1.0), "epoch {epoch}: expected 1.0, got {lr}");
        }
    }

    #[test]
    fn step_decay_applies_at_boundary() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::StepDecay {
            initial_lr: 1.0,
            decay_factor: 0.5,
            step_size: 10,
        });
        let lr10 = sched.step(10);
        assert!(approx(lr10, 0.5), "expected 0.5 at epoch 10, got {lr10}");
        let lr20 = sched.step(20);
        assert!(approx(lr20, 0.25), "expected 0.25 at epoch 20, got {lr20}");
    }

    #[test]
    fn step_decay_zero_step_size_no_panic() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::StepDecay {
            initial_lr: 1.0,
            decay_factor: 0.5,
            step_size: 0,
        });
        let _ = sched.step(5); // should not panic
    }

    // ------------------------------------------------------------------ ExponentialDecay

    #[test]
    fn exponential_decay_formula() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::ExponentialDecay {
            initial_lr: 1.0,
            decay_rate: 0.1,
        });
        for epoch in 0u64..=5 {
            let lr = sched.step(epoch);
            let expected = (-0.1_f64 * epoch as f64).exp();
            assert!(
                approx(lr, expected),
                "epoch {epoch}: expected {expected}, got {lr}"
            );
        }
    }

    #[test]
    fn exponential_decay_monotone_decreasing() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::ExponentialDecay {
            initial_lr: 2.0,
            decay_rate: 0.05,
        });
        let mut prev = sched.step(0);
        for epoch in 1..100 {
            let lr = sched.step(epoch);
            assert!(lr < prev + 1e-15, "epoch {epoch}: LR did not decrease");
            prev = lr;
        }
    }

    #[test]
    fn exponential_decay_at_epoch_zero_equals_initial() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::ExponentialDecay {
            initial_lr: 0.5,
            decay_rate: 0.2,
        });
        assert!(approx(sched.step(0), 0.5));
    }

    // ------------------------------------------------------------------ CosineAnnealing

    #[test]
    fn cosine_annealing_starts_at_initial_lr() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::CosineAnnealing {
            initial_lr: 0.1,
            min_lr: 0.001,
            t_max: 100,
        });
        let lr = sched.step(0);
        assert!(approx(lr, 0.1), "expected 0.1, got {lr}");
    }

    #[test]
    fn cosine_annealing_reaches_min_at_t_max() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::CosineAnnealing {
            initial_lr: 0.1,
            min_lr: 1e-4,
            t_max: 100,
        });
        let lr = sched.step(100);
        assert!(approx(lr, 1e-4), "expected min_lr at t_max, got {lr}");
    }

    #[test]
    fn cosine_annealing_midpoint() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::CosineAnnealing {
            initial_lr: 1.0,
            min_lr: 0.0,
            t_max: 100,
        });
        let lr = sched.step(50);
        assert!(approx(lr, 0.5), "cosine midpoint should be 0.5, got {lr}");
    }

    #[test]
    fn cosine_annealing_clamps_beyond_t_max() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::CosineAnnealing {
            initial_lr: 0.1,
            min_lr: 0.001,
            t_max: 50,
        });
        let lr_at_50 = sched.step(50);
        let lr_at_200 = sched.step(200);
        assert!(approx(lr_at_50, lr_at_200), "beyond t_max should clamp");
    }

    // ------------------------------------------------------------------ WarmupCosine

    #[test]
    fn warmup_cosine_starts_at_initial_lr() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::WarmupCosine {
            warmup_epochs: 10,
            initial_lr: 0.0,
            peak_lr: 0.1,
            min_lr: 1e-5,
            t_max: 90,
        });
        let lr = sched.step(0);
        assert!(approx(lr, 0.0), "expected 0.0 at epoch 0, got {lr}");
    }

    #[test]
    fn warmup_cosine_reaches_peak_after_warmup() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::WarmupCosine {
            warmup_epochs: 10,
            initial_lr: 0.0,
            peak_lr: 0.1,
            min_lr: 1e-5,
            t_max: 90,
        });
        // At epoch 10 (first cosine epoch=0) => cosine_factor(0, 90) = 1.0
        // => min_lr + (peak - min_lr) * 1.0 = peak_lr
        let lr = sched.step(10);
        assert!(approx(lr, 0.1), "expected peak_lr at warmup end, got {lr}");
    }

    #[test]
    fn warmup_cosine_linear_during_warmup() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::WarmupCosine {
            warmup_epochs: 10,
            initial_lr: 0.0,
            peak_lr: 0.1,
            min_lr: 1e-5,
            t_max: 90,
        });
        let lr5 = sched.step(5);
        // warmup_factor(5, 10) = 0.5 => 0.0 + 0.5 * 0.1 = 0.05
        assert!(
            approx(lr5, 0.05),
            "expected 0.05 at warmup midpoint, got {lr5}"
        );
    }

    #[test]
    fn warmup_cosine_descends_after_peak() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::WarmupCosine {
            warmup_epochs: 5,
            initial_lr: 0.0,
            peak_lr: 0.1,
            min_lr: 0.0,
            t_max: 100,
        });
        let lr_peak = sched.step(5);
        let lr_later = sched.step(55); // halfway through cosine => ~0.5 amplitude
        assert!(lr_later < lr_peak, "LR should decrease after warmup");
    }

    // ------------------------------------------------------------------ CyclicLR

    #[test]
    fn cyclic_lr_starts_at_base() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::CyclicLR {
            base_lr: 0.001,
            max_lr: 0.01,
            step_size: 10,
        });
        let lr = sched.step(0);
        assert!(approx(lr, 0.001), "expected base_lr at epoch 0, got {lr}");
    }

    #[test]
    fn cyclic_lr_reaches_max_at_step_size() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::CyclicLR {
            base_lr: 0.0,
            max_lr: 1.0,
            step_size: 10,
        });
        let lr = sched.step(10);
        assert!(approx(lr, 1.0), "expected max_lr at step_size, got {lr}");
    }

    #[test]
    fn cyclic_lr_returns_to_base_at_full_cycle() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::CyclicLR {
            base_lr: 0.001,
            max_lr: 0.01,
            step_size: 10,
        });
        let lr = sched.step(20);
        assert!(
            approx(lr, 0.001),
            "expected base_lr after full cycle, got {lr}"
        );
    }

    #[test]
    fn cyclic_lr_is_symmetric() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::CyclicLR {
            base_lr: 0.0,
            max_lr: 1.0,
            step_size: 10,
        });
        let lr5 = sched.step(5);
        let lr15 = sched.step(15);
        assert!(approx(lr5, lr15), "triangular cycle should be symmetric");
    }

    // ------------------------------------------------------------------ ReduceOnPlateau

    #[test]
    fn reduce_on_plateau_decreases_after_patience() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::ReduceOnPlateau {
            initial_lr: 0.1,
            factor: 0.5,
            patience: 3,
            min_lr: 1e-6,
            threshold: 1e-4,
        });
        // Provide improving loss to set best
        sched.step_with_loss(0, 1.0);
        // No improvement for `patience` steps
        sched.step_with_loss(1, 1.0);
        sched.step_with_loss(2, 1.0);
        sched.step_with_loss(3, 1.0);
        let lr = sched.current_lr();
        assert!(lr < 0.1, "LR should decrease after plateau, got {lr}");
    }

    #[test]
    fn reduce_on_plateau_does_not_reduce_when_improving() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::ReduceOnPlateau {
            initial_lr: 0.1,
            factor: 0.5,
            patience: 3,
            min_lr: 1e-6,
            threshold: 1e-4,
        });
        // Steadily improving loss
        for i in 0u64..20 {
            sched.step_with_loss(i, 1.0 / (i as f64 + 1.0));
        }
        assert!(
            approx(sched.current_lr(), 0.1),
            "LR should not change when loss keeps improving, got {}",
            sched.current_lr()
        );
    }

    #[test]
    fn reduce_on_plateau_respects_min_lr() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::ReduceOnPlateau {
            initial_lr: 0.1,
            factor: 0.1,
            patience: 1,
            min_lr: 0.05,
            threshold: 1e-4,
        });
        sched.step_with_loss(0, 1.0);
        sched.step_with_loss(1, 1.0); // triggers plateau
        let lr = sched.current_lr();
        assert!(lr >= 0.05, "LR should not go below min_lr, got {lr}");
    }

    #[test]
    fn reduce_on_plateau_stats_count_reductions() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::ReduceOnPlateau {
            initial_lr: 0.1,
            factor: 0.5,
            patience: 2,
            min_lr: 1e-9,
            threshold: 1e-4,
        });
        sched.step_with_loss(0, 1.0); // sets best
        sched.step_with_loss(1, 1.0); // count=1
        sched.step_with_loss(2, 1.0); // count=2 => reduce #1
        sched.step_with_loss(3, 1.0); // count=1
        sched.step_with_loss(4, 1.0); // count=2 => reduce #2
        assert_eq!(
            sched.stats().plateau_reductions,
            2,
            "expected 2 plateau reductions"
        );
    }

    // ------------------------------------------------------------------ reset

    #[test]
    fn reset_clears_history_and_state() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::Constant { lr: 0.01 });
        for i in 0..10 {
            sched.step(i);
        }
        sched.reset();
        assert_eq!(sched.history().len(), 0);
        assert_eq!(sched.state.current_epoch, 0);
        assert!(approx(sched.current_lr(), 0.01));
    }

    #[test]
    fn reset_clears_plateau_reductions() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::ReduceOnPlateau {
            initial_lr: 0.1,
            factor: 0.5,
            patience: 1,
            min_lr: 1e-9,
            threshold: 1e-4,
        });
        sched.step_with_loss(0, 1.0);
        sched.step_with_loss(1, 1.0); // triggers reduction
        sched.reset();
        assert_eq!(sched.stats().plateau_reductions, 0);
    }

    // ------------------------------------------------------------------ history

    #[test]
    fn history_records_each_step() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::Constant { lr: 0.02 });
        for i in 0..5u64 {
            sched.step(i);
        }
        assert_eq!(sched.history().len(), 5);
    }

    #[test]
    fn history_capped_at_1000() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::Constant { lr: 0.01 });
        for i in 0..1200u64 {
            sched.step(i);
        }
        assert_eq!(sched.history().len(), 1000);
    }

    #[test]
    fn history_loss_recorded_with_step_with_loss() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::Constant { lr: 0.01 });
        sched.step_with_loss(0, 0.42);
        let entry = &sched.history()[0];
        assert_eq!(entry.loss, Some(0.42));
    }

    #[test]
    fn history_no_loss_for_plain_step() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::Constant { lr: 0.01 });
        sched.step(0);
        assert_eq!(sched.history()[0].loss, None);
    }

    // ------------------------------------------------------------------ stats

    #[test]
    fn stats_min_max_lr() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::StepDecay {
            initial_lr: 1.0,
            decay_factor: 0.5,
            step_size: 10,
        });
        sched.step(0); // lr = 1.0
        sched.step(10); // lr = 0.5
        sched.step(20); // lr = 0.25
        let s = sched.stats();
        assert!(approx(s.max_lr_seen, 1.0));
        assert!(approx(s.min_lr_seen, 0.25));
    }

    #[test]
    fn stats_epochs_trained() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::Constant { lr: 0.1 });
        for i in 0..7u64 {
            sched.step(i);
        }
        assert_eq!(sched.stats().epochs_trained, 7);
    }

    // ------------------------------------------------------------------ helper functions

    #[test]
    fn warmup_factor_zero_epochs_returns_one() {
        assert!((LearningRateScheduler::warmup_factor(5, 0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn warmup_factor_linear_interpolation() {
        let f = LearningRateScheduler::warmup_factor(5, 10);
        assert!((f - 0.5).abs() < 1e-15, "expected 0.5, got {f}");
    }

    #[test]
    fn warmup_factor_clamps_at_one() {
        let f = LearningRateScheduler::warmup_factor(20, 10);
        assert!((f - 1.0).abs() < 1e-15, "expected 1.0, got {f}");
    }

    #[test]
    fn cosine_factor_zero_t_max_returns_zero() {
        let f = LearningRateScheduler::cosine_factor(5, 0);
        assert!((f - 0.0).abs() < 1e-15, "expected 0.0, got {f}");
    }

    #[test]
    fn cosine_factor_at_epoch_zero_returns_one() {
        let f = LearningRateScheduler::cosine_factor(0, 100);
        assert!((f - 1.0).abs() < 1e-15, "expected 1.0, got {f}");
    }

    #[test]
    fn cosine_factor_at_t_max_returns_zero() {
        let f = LearningRateScheduler::cosine_factor(100, 100);
        assert!((f - 0.0).abs() < TOL, "expected 0.0, got {f}");
    }

    #[test]
    fn cosine_factor_midpoint_returns_half() {
        let f = LearningRateScheduler::cosine_factor(50, 100);
        assert!((f - 0.5).abs() < TOL, "expected 0.5, got {f}");
    }

    // ------------------------------------------------------------------ state fields

    #[test]
    fn state_plateau_count_tracked() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::ReduceOnPlateau {
            initial_lr: 0.1,
            factor: 0.5,
            patience: 5,
            min_lr: 1e-6,
            threshold: 1e-4,
        });
        sched.step_with_loss(0, 1.0); // best_loss = 1.0
        sched.step_with_loss(1, 1.0); // plateau_count = 1
        sched.step_with_loss(2, 1.0); // plateau_count = 2
        assert_eq!(sched.state.plateau_count, 2);
    }

    #[test]
    fn state_best_loss_updated_on_improvement() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::ReduceOnPlateau {
            initial_lr: 0.1,
            factor: 0.5,
            patience: 5,
            min_lr: 1e-6,
            threshold: 1e-4,
        });
        sched.step_with_loss(0, 2.0);
        sched.step_with_loss(1, 0.5);
        assert!(approx(sched.state.best_loss, 0.5));
    }

    // ------------------------------------------------------------------ LrHistory fields

    #[test]
    fn lr_history_epoch_field_correct() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::Constant { lr: 0.01 });
        sched.step(42);
        assert_eq!(sched.history()[0].epoch, 42);
    }

    #[test]
    fn lr_history_lr_field_correct() {
        let mut sched = LearningRateScheduler::new(SchedulerStrategy::Constant { lr: 0.07 });
        sched.step(0);
        assert!(approx(sched.history()[0].lr, 0.07));
    }
}
