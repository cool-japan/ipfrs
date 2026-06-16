//! Embedding Drift Monitor
//!
//! Monitors concept drift in embedding distributions by comparing incoming vectors
//! against a baseline, alerting when the distribution shifts significantly.

/// Signal indicating the degree of distributional drift detected.
#[derive(Clone, Debug, PartialEq)]
pub enum DriftSignal {
    /// No significant drift detected.
    NoDrift,
    /// Minor drift detected; magnitude in [0.1, 0.3).
    MinorDrift { magnitude: f64 },
    /// Major drift detected; magnitude >= 0.3.
    MajorDrift { magnitude: f64 },
    /// Insufficient data to compute drift (fewer than min_baseline_samples).
    InsufficientData,
}

/// Per-dimension statistics computed from the baseline window.
#[derive(Clone, Debug)]
pub struct BaselineStats {
    /// Per-dimension mean of baseline vectors.
    pub mean: Vec<f32>,
    /// Per-dimension standard deviation of baseline vectors (minimum 1e-6).
    pub std: Vec<f32>,
    /// Number of samples used to compute this baseline.
    pub sample_count: usize,
}

impl BaselineStats {
    /// Compute baseline statistics from a slice of embedding vectors.
    ///
    /// Returns `None` if `samples` is empty or all vectors have zero length.
    pub fn compute(samples: &[Vec<f32>]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let dims = samples[0].len();
        if dims == 0 {
            return None;
        }
        let n = samples.len() as f64;

        // Compute per-dimension mean.
        let mut mean = vec![0.0_f64; dims];
        for vec in samples {
            for (d, &v) in vec.iter().enumerate() {
                mean[d] += v as f64;
            }
        }
        for m in &mut mean {
            *m /= n;
        }

        // Compute per-dimension variance, then std.
        let mut variance = vec![0.0_f64; dims];
        for vec in samples {
            for (d, &v) in vec.iter().enumerate() {
                let diff = v as f64 - mean[d];
                variance[d] += diff * diff;
            }
        }
        let mean_f32: Vec<f32> = mean.iter().map(|&m| m as f32).collect();
        let std_f32: Vec<f32> = variance
            .iter()
            .map(|&v| {
                let s = (v / n).sqrt();
                s.max(1e-6_f64) as f32
            })
            .collect();

        Some(BaselineStats {
            mean: mean_f32,
            std: std_f32,
            sample_count: samples.len(),
        })
    }
}

/// Configuration for the [`EmbeddingDriftMonitor`].
#[derive(Clone, Debug)]
pub struct DriftMonitorConfig {
    /// Minimum number of baseline samples required before drift is computed.
    pub min_baseline_samples: usize,
    /// Normalised magnitude threshold below which drift is classified as [`DriftSignal::NoDrift`].
    pub minor_drift_threshold: f64,
    /// Normalised magnitude threshold at or above which drift is classified as [`DriftSignal::MajorDrift`].
    pub major_drift_threshold: f64,
    /// Maximum number of recent vectors retained in the sliding window.
    pub window_size: usize,
    /// Re-compute the baseline every N new vectors added.
    pub baseline_update_interval: usize,
}

impl Default for DriftMonitorConfig {
    fn default() -> Self {
        Self {
            min_baseline_samples: 50,
            minor_drift_threshold: 0.1,
            major_drift_threshold: 0.3,
            window_size: 100,
            baseline_update_interval: 25,
        }
    }
}

/// Cumulative statistics tracked by the monitor.
#[derive(Clone, Debug, Default)]
pub struct DriftMonitorStats {
    /// Total number of vectors checked for drift.
    pub total_checked: u64,
    /// Number of vectors that triggered a [`DriftSignal::MinorDrift`].
    pub minor_drift_count: u64,
    /// Number of vectors that triggered a [`DriftSignal::MajorDrift`].
    pub major_drift_count: u64,
    /// Number of times the baseline has been recomputed.
    pub baseline_updates: u64,
}

impl DriftMonitorStats {
    /// Fraction of checked vectors that resulted in any drift signal.
    ///
    /// Returns `0.0` when no vectors have been checked yet.
    pub fn drift_rate(&self) -> f64 {
        let total = self.total_checked.max(1);
        (self.minor_drift_count + self.major_drift_count) as f64 / total as f64
    }
}

/// Monitors concept drift in embedding distributions.
///
/// Maintains a sliding window of recent vectors, periodically recomputes
/// a baseline, and classifies each new vector against the baseline using
/// normalised per-dimension deviation.
pub struct EmbeddingDriftMonitor {
    /// Current baseline statistics, if enough samples have been seen.
    pub baseline: Option<BaselineStats>,
    /// Sliding window of recent vectors.
    pub recent: Vec<Vec<f32>>,
    /// Monitor configuration.
    pub config: DriftMonitorConfig,
    /// Cumulative statistics.
    pub stats: DriftMonitorStats,
    /// Number of vectors added since the last baseline update.
    pub update_counter: usize,
}

impl EmbeddingDriftMonitor {
    /// Create a new monitor with the given configuration.
    pub fn new(config: DriftMonitorConfig) -> Self {
        Self {
            baseline: None,
            recent: Vec::new(),
            config,
            stats: DriftMonitorStats::default(),
            update_counter: 0,
        }
    }

    /// Add a vector to the monitor and return a [`DriftSignal`].
    ///
    /// Steps:
    /// 1. Push `vec` into the sliding window, evicting the oldest entry when
    ///    `recent.len() > window_size`.
    /// 2. Increment `update_counter`; if `update_counter >= baseline_update_interval`
    ///    and enough samples exist, recompute the baseline.
    /// 3. If no baseline is available, return [`DriftSignal::InsufficientData`].
    /// 4. Otherwise, compute the mean absolute normalised deviation and classify.
    pub fn add_vector(&mut self, vec: Vec<f32>) -> DriftSignal {
        // 1. Maintain sliding window.
        self.recent.push(vec.clone());
        if self.recent.len() > self.config.window_size {
            self.recent.remove(0);
        }

        // 2. Possibly recompute baseline.
        self.update_counter += 1;
        if self.update_counter >= self.config.baseline_update_interval
            && self.recent.len() >= self.config.min_baseline_samples
        {
            if let Some(new_baseline) = BaselineStats::compute(&self.recent) {
                self.baseline = Some(new_baseline);
                self.update_counter = 0;
                self.stats.baseline_updates += 1;
            }
        }

        // 3. Guard: no baseline yet.
        let baseline = match &self.baseline {
            Some(b) => b,
            None => return DriftSignal::InsufficientData,
        };

        // 4. Compute mean absolute normalised deviation.
        let dims = baseline.mean.len();
        if dims == 0 || vec.len() != dims {
            return DriftSignal::InsufficientData;
        }

        let magnitude: f64 = vec
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let deviation = (v as f64 - baseline.mean[i] as f64) / baseline.std[i] as f64;
                deviation.abs()
            })
            .sum::<f64>()
            / dims as f64;

        self.stats.total_checked += 1;

        if magnitude >= self.config.major_drift_threshold {
            self.stats.major_drift_count += 1;
            DriftSignal::MajorDrift { magnitude }
        } else if magnitude >= self.config.minor_drift_threshold {
            self.stats.minor_drift_count += 1;
            DriftSignal::MinorDrift { magnitude }
        } else {
            DriftSignal::NoDrift
        }
    }

    /// Return a reference to the current baseline statistics, if available.
    pub fn baseline(&self) -> Option<&BaselineStats> {
        self.baseline.as_ref()
    }

    /// Return a reference to the cumulative statistics.
    pub fn stats(&self) -> &DriftMonitorStats {
        &self.stats
    }

    /// Clear the baseline, recent window, and reset counters.
    pub fn reset_baseline(&mut self) {
        self.baseline = None;
        self.recent.clear();
        self.update_counter = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a monitor with a small config for quick tests.
    fn small_config(min_baseline: usize, window: usize, interval: usize) -> DriftMonitorConfig {
        DriftMonitorConfig {
            min_baseline_samples: min_baseline,
            window_size: window,
            baseline_update_interval: interval,
            minor_drift_threshold: 0.1,
            major_drift_threshold: 0.3,
        }
    }

    // Helper: constant embedding vector of given dimension.
    fn const_vec(dim: usize, value: f32) -> Vec<f32> {
        vec![value; dim]
    }

    // Helper: populate monitor with `n` identical vectors so baseline is ready.
    fn seed_monitor(monitor: &mut EmbeddingDriftMonitor, n: usize, dim: usize, value: f32) {
        for _ in 0..n {
            monitor.add_vector(const_vec(dim, value));
        }
    }

    // ---------------------------------------------------------------------------
    // 1. InsufficientData before min_baseline_samples
    // ---------------------------------------------------------------------------
    #[test]
    fn test_insufficient_data_initially() {
        let cfg = small_config(5, 20, 5);
        let mut monitor = EmbeddingDriftMonitor::new(cfg);
        let signal = monitor.add_vector(const_vec(4, 0.5));
        assert_eq!(signal, DriftSignal::InsufficientData);
    }

    #[test]
    fn test_insufficient_data_before_threshold() {
        let cfg = small_config(5, 20, 5);
        let mut monitor = EmbeddingDriftMonitor::new(cfg);
        // Add 4 vectors — one short of min_baseline_samples=5.
        for _ in 0..4 {
            let s = monitor.add_vector(const_vec(4, 0.5));
            assert_eq!(s, DriftSignal::InsufficientData);
        }
    }

    // ---------------------------------------------------------------------------
    // 2. Baseline computed after enough samples
    // ---------------------------------------------------------------------------
    #[test]
    fn test_baseline_computed_after_enough_samples() {
        let cfg = small_config(5, 20, 5);
        let mut monitor = EmbeddingDriftMonitor::new(cfg);
        seed_monitor(&mut monitor, 5, 4, 1.0);
        assert!(monitor.baseline().is_some());
    }

    #[test]
    fn test_baseline_mean_values() {
        let cfg = small_config(4, 20, 4);
        let mut monitor = EmbeddingDriftMonitor::new(cfg);
        // All vectors identical → mean should equal that value.
        seed_monitor(&mut monitor, 4, 3, 2.0);
        let baseline = monitor.baseline().expect("baseline should exist");
        for &m in &baseline.mean {
            assert!((m - 2.0_f32).abs() < 1e-4, "mean={m}");
        }
    }

    // ---------------------------------------------------------------------------
    // 3. NoDrift for mean-similar vector
    // ---------------------------------------------------------------------------
    #[test]
    fn test_no_drift_for_identical_vector() {
        let cfg = small_config(5, 20, 5);
        let mut monitor = EmbeddingDriftMonitor::new(cfg);
        // Build baseline with constant vectors.
        for _ in 0..4 {
            monitor.add_vector(const_vec(4, 0.0));
        }
        // 5th vector triggers baseline computation.
        let signal = monitor.add_vector(const_vec(4, 0.0));
        // Same as mean → NoDrift.
        assert_eq!(signal, DriftSignal::NoDrift);
    }

    #[test]
    fn test_no_drift_small_deviation() {
        let cfg = small_config(5, 20, 5);
        let mut monitor = EmbeddingDriftMonitor::new(cfg);
        seed_monitor(&mut monitor, 5, 4, 1.0);
        // Tiny perturbation that stays below 0.1 normalised threshold.
        // std will be floored at 1e-6, so even a tiny absolute deviation yields large magnitude.
        // Use a vector that is exactly the mean.
        let signal = monitor.add_vector(const_vec(4, 1.0));
        assert_eq!(signal, DriftSignal::NoDrift);
    }

    // ---------------------------------------------------------------------------
    // 4. MinorDrift
    // ---------------------------------------------------------------------------
    #[test]
    fn test_minor_drift_detected() {
        // Use a varied baseline so std is non-trivial.
        let cfg = small_config(4, 20, 4);
        let mut monitor = EmbeddingDriftMonitor::new(cfg);
        // Baseline: alternating 0.0 and 2.0 per sample so mean=1.0 std≈1.0.
        monitor.add_vector(vec![0.0, 0.0, 0.0, 0.0]);
        monitor.add_vector(vec![2.0, 2.0, 2.0, 2.0]);
        monitor.add_vector(vec![0.0, 0.0, 0.0, 0.0]);
        // 4th vector triggers baseline (interval=4).
        monitor.add_vector(vec![2.0, 2.0, 2.0, 2.0]);
        // mean ≈ 1.0, std ≈ 1.0 → deviation of 0.15 → magnitude ≈ 0.15.
        let signal = monitor.add_vector(vec![1.15, 1.15, 1.15, 1.15]);
        match signal {
            DriftSignal::MinorDrift { magnitude } => {
                assert!((0.1..0.3).contains(&magnitude), "magnitude={magnitude}");
            }
            other => panic!("expected MinorDrift, got {other:?}"),
        }
    }

    #[test]
    fn test_minor_drift_stats_increment() {
        let cfg = small_config(4, 20, 4);
        let mut monitor = EmbeddingDriftMonitor::new(cfg);
        monitor.add_vector(vec![0.0; 4]);
        monitor.add_vector(vec![2.0; 4]);
        monitor.add_vector(vec![0.0; 4]);
        monitor.add_vector(vec![2.0; 4]);
        monitor.add_vector(vec![1.15; 4]);
        assert_eq!(monitor.stats().minor_drift_count, 1);
    }

    // ---------------------------------------------------------------------------
    // 5. MajorDrift
    // ---------------------------------------------------------------------------
    #[test]
    fn test_major_drift_detected() {
        let cfg = small_config(4, 20, 4);
        let mut monitor = EmbeddingDriftMonitor::new(cfg);
        monitor.add_vector(vec![0.0; 4]);
        monitor.add_vector(vec![2.0; 4]);
        monitor.add_vector(vec![0.0; 4]);
        monitor.add_vector(vec![2.0; 4]);
        // magnitude = |5.0 - 1.0| / std ≈ 4.0 >> 0.3.
        let signal = monitor.add_vector(vec![5.0; 4]);
        match signal {
            DriftSignal::MajorDrift { magnitude } => {
                assert!(magnitude >= 0.3, "magnitude={magnitude}");
            }
            other => panic!("expected MajorDrift, got {other:?}"),
        }
    }

    #[test]
    fn test_major_drift_stats_increment() {
        let cfg = small_config(4, 20, 4);
        let mut monitor = EmbeddingDriftMonitor::new(cfg);
        // Baseline is built from these 4 vectors (mean≈1.0, std≈1.0).
        // The 4th vector (2.0) is checked after baseline is set → magnitude≈1.0 → MajorDrift.
        monitor.add_vector(vec![0.0; 4]);
        monitor.add_vector(vec![2.0; 4]);
        monitor.add_vector(vec![0.0; 4]);
        monitor.add_vector(vec![2.0; 4]);
        // 5th vector far from mean → additional MajorDrift.
        monitor.add_vector(vec![5.0; 4]);
        // Two major drift events: the 4th and 5th vectors.
        assert_eq!(monitor.stats().major_drift_count, 2);
    }

    // ---------------------------------------------------------------------------
    // 6. Sliding window evicts oldest
    // ---------------------------------------------------------------------------
    #[test]
    fn test_sliding_window_size_capped() {
        let cfg = small_config(3, 5, 3);
        let mut monitor = EmbeddingDriftMonitor::new(cfg);
        for i in 0..10_u8 {
            monitor.add_vector(vec![i as f32; 2]);
        }
        assert!(
            monitor.recent.len() <= 5,
            "window exceeded: {}",
            monitor.recent.len()
        );
    }

    #[test]
    fn test_sliding_window_oldest_evicted() {
        let cfg = small_config(2, 3, 2);
        let mut monitor = EmbeddingDriftMonitor::new(cfg);
        monitor.add_vector(vec![1.0, 1.0]);
        monitor.add_vector(vec![2.0, 2.0]);
        monitor.add_vector(vec![3.0, 3.0]);
        // After 3 inserts into a window of size 3, vec [1.0,1.0] should be gone.
        monitor.add_vector(vec![4.0, 4.0]);
        // Window should now be [2,3,4]; the first vector [1,1] evicted.
        assert!(!monitor.recent.contains(&vec![1.0, 1.0]));
        assert_eq!(monitor.recent.len(), 3);
    }

    // ---------------------------------------------------------------------------
    // 7. Baseline updates at interval
    // ---------------------------------------------------------------------------
    #[test]
    fn test_baseline_update_increments_counter() {
        // interval=3, min_baseline=3, window=20
        let cfg = small_config(3, 20, 3);
        let mut monitor = EmbeddingDriftMonitor::new(cfg);
        // First baseline update at sample 3.
        seed_monitor(&mut monitor, 3, 2, 1.0);
        assert_eq!(monitor.stats().baseline_updates, 1);
        // Next update at sample 6 (3 more).
        seed_monitor(&mut monitor, 3, 2, 1.0);
        assert_eq!(monitor.stats().baseline_updates, 2);
    }

    #[test]
    fn test_no_baseline_update_before_interval() {
        let cfg = small_config(3, 20, 5);
        let mut monitor = EmbeddingDriftMonitor::new(cfg);
        // Only 3 vectors added; interval=5 not reached.
        seed_monitor(&mut monitor, 3, 2, 1.0);
        assert_eq!(monitor.stats().baseline_updates, 0);
        // Baseline is None because update never triggered.
        assert!(monitor.baseline().is_none());
    }

    // ---------------------------------------------------------------------------
    // 8. drift_rate
    // ---------------------------------------------------------------------------
    #[test]
    fn test_drift_rate_zero_initially() {
        let stats = DriftMonitorStats::default();
        assert_eq!(stats.drift_rate(), 0.0);
    }

    #[test]
    fn test_drift_rate_calculation() {
        let stats = DriftMonitorStats {
            total_checked: 10,
            minor_drift_count: 2,
            major_drift_count: 3,
            baseline_updates: 1,
        };
        let rate = stats.drift_rate();
        assert!((rate - 0.5).abs() < 1e-9, "rate={rate}");
    }

    #[test]
    fn test_drift_rate_all_drift() {
        let stats = DriftMonitorStats {
            total_checked: 4,
            minor_drift_count: 4,
            major_drift_count: 0,
            baseline_updates: 0,
        };
        assert!((stats.drift_rate() - 1.0).abs() < 1e-9);
    }

    // ---------------------------------------------------------------------------
    // 9. reset_baseline clears state
    // ---------------------------------------------------------------------------
    #[test]
    fn test_reset_baseline_clears_baseline() {
        let cfg = small_config(3, 20, 3);
        let mut monitor = EmbeddingDriftMonitor::new(cfg);
        seed_monitor(&mut monitor, 3, 2, 1.0);
        assert!(monitor.baseline().is_some());
        monitor.reset_baseline();
        assert!(monitor.baseline().is_none());
    }

    #[test]
    fn test_reset_baseline_clears_recent() {
        let cfg = small_config(3, 20, 3);
        let mut monitor = EmbeddingDriftMonitor::new(cfg);
        seed_monitor(&mut monitor, 3, 2, 1.0);
        monitor.reset_baseline();
        assert!(monitor.recent.is_empty());
    }

    #[test]
    fn test_reset_baseline_resets_update_counter() {
        let cfg = small_config(3, 20, 3);
        let mut monitor = EmbeddingDriftMonitor::new(cfg);
        seed_monitor(&mut monitor, 3, 2, 1.0);
        monitor.reset_baseline();
        assert_eq!(monitor.update_counter, 0);
    }

    // ---------------------------------------------------------------------------
    // 10. stats.baseline_updates increments correctly
    // ---------------------------------------------------------------------------
    #[test]
    fn test_baseline_updates_exact_count() {
        // interval=2, min_baseline=2, window=20 → update at every 2 vectors.
        let cfg = small_config(2, 20, 2);
        let mut monitor = EmbeddingDriftMonitor::new(cfg);
        for _ in 0..8 {
            monitor.add_vector(const_vec(2, 1.0));
        }
        // Updates at samples 2, 4, 6, 8 → 4 updates.
        assert_eq!(monitor.stats().baseline_updates, 4);
    }

    // ---------------------------------------------------------------------------
    // 11. total_checked increments only when baseline exists
    // ---------------------------------------------------------------------------
    #[test]
    fn test_total_checked_only_after_baseline() {
        let cfg = small_config(3, 20, 3);
        let mut monitor = EmbeddingDriftMonitor::new(cfg);
        // First 3 vectors trigger baseline; 3rd one gets drift-checked.
        seed_monitor(&mut monitor, 3, 2, 1.0);
        // total_checked should be 1 (the 3rd vector, which triggered baseline and was then checked).
        assert_eq!(monitor.stats().total_checked, 1);
    }
}
