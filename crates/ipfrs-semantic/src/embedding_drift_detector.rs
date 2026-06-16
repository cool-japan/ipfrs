//! # EmbeddingDriftDetector
//!
//! Production-quality concept drift detection in high-dimensional embedding spaces.
//!
//! Supports multiple statistical methods:
//! - **CentroidDistance** — Euclidean distance between window centroids
//! - **KLDivergence** — Per-dimension Gaussian KL divergence approximation
//! - **PageHinkley** — Sequential change-point detection
//! - **ADWIN** — Adaptive windowing with sub-window comparison
//! - **CUSUMDetector** — Cumulative sum control chart on centroid norms
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_semantic::embedding_drift_detector::{
//!     EmbeddingDriftDetector, DetectorConfig, DetectionMethod,
//! };
//!
//! let config = DetectorConfig {
//!     method: DetectionMethod::CentroidDistance(0.3),
//!     window_size: 50,
//!     reference_window_size: 50,
//!     min_samples_before_detect: 20,
//!     drift_threshold: 0.3,
//! };
//! let mut detector = EmbeddingDriftDetector::new(config);
//!
//! // Feed reference embeddings
//! for i in 0..50_u64 {
//!     let emb = vec![0.1_f64, 0.2, 0.3];
//!     let _ = detector.add_sample(emb, i);
//! }
//!
//! // Trigger a drift
//! for i in 50..100_u64 {
//!     let emb = vec![0.9_f64, 0.8, 0.7];
//!     let _ = detector.add_sample(emb, i);
//! }
//! println!("Stats: {:?}", detector.stats());
//! ```

use std::collections::VecDeque;
use std::fmt;

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Arithmetic mean of a slice; returns 0.0 for empty input.
#[inline]
fn stat_mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Unbiased sample variance of a slice of values; returns 0.0 for fewer than two elements.
/// Used for per-window scalar statistics (e.g., norm distributions).
#[inline]
fn stat_variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = stat_mean(data);
    data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}

/// Cosine distance in [0, 2] (1 – cosine_similarity).
/// Returns 1.0 if lengths differ or either vector is a near-zero vector.
fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 1.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-10 || nb < 1e-10 {
        1.0
    } else {
        1.0 - (dot / (na * nb)).clamp(-1.0, 1.0)
    }
}

/// Euclidean distance between two equal-length vectors.
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ─── PRNG (no rand crate) ───────────────────────────────────────────────────

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[inline]
fn xorshift_f64(state: &mut u64) -> f64 {
    (xorshift64(state) >> 11) as f64 / (1u64 << 53) as f64
}

// ─── Public Error Type ──────────────────────────────────────────────────────

/// Errors produced by [`EmbeddingDriftDetector`].
#[derive(Debug, Clone, PartialEq)]
pub enum DetectorError {
    /// Not enough samples to perform detection.
    InsufficientData(usize),
    /// Embedding dimension does not match the established dimension.
    DimensionMismatch {
        /// Expected embedding dimension.
        expected: usize,
        /// Received embedding dimension.
        got: usize,
    },
    /// The rolling window has no samples.
    WindowEmpty,
    /// Invalid detector configuration.
    ConfigurationError(String),
}

impl fmt::Display for DetectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientData(n) => write!(f, "Insufficient data: {n} samples"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {expected}, got {got}")
            }
            Self::WindowEmpty => write!(f, "Rolling window is empty"),
            Self::ConfigurationError(msg) => write!(f, "Configuration error: {msg}"),
        }
    }
}

impl std::error::Error for DetectorError {}

// ─── Detection Methods ──────────────────────────────────────────────────────

/// Statistical method used to decide whether drift has occurred.
#[derive(Debug, Clone, PartialEq)]
pub enum DetectionMethod {
    /// Euclidean distance between window centroids; field is the drift threshold.
    CentroidDistance(f64),
    /// Approximate KL divergence (per-dim Gaussian); field is the drift threshold.
    KLDivergence(f64),
    /// Page-Hinkley sequential change-point test.
    /// `delta` is the expected magnitude of normal change; `lambda` is the cumulative threshold.
    PageHinkley { delta: f64, lambda: f64 },
    /// Adaptive windowing (ADWIN-style) sub-window comparison.
    /// `delta` is the significance parameter.
    ADWIN { delta: f64 },
    /// Standard CUSUM on centroid norms.
    /// `k` is the allowable slack; `h` is the decision threshold.
    CUSUMDetector { k: f64, h: f64 },
}

// ─── Drift Classification ───────────────────────────────────────────────────

/// Classification of the kind of drift that was detected.
#[derive(Debug, Clone, PartialEq)]
pub enum DriftType {
    /// Centroid shift — the distribution mean has moved.
    ConceptDrift,
    /// Spread change — variance has grown or shrunk.
    VarianceDrift,
    /// Single-dimension shift.
    DimensionDrift {
        /// The dimension index that drifted.
        dim: usize,
    },
    /// Periodic / seasonal pattern.
    SeasonalDrift,
    /// Slow, gradual drift.
    GradualDrift,
    /// Abrupt, sudden change.
    SuddenDrift,
    /// Previously seen distribution has returned.
    RecurringDrift,
}

// ─── Core Data Structures ───────────────────────────────────────────────────

/// A statistical snapshot of the current embedding window.
#[derive(Debug, Clone)]
pub struct DriftSnapshot {
    /// Unique identifier for this snapshot (UUID-style hex string).
    pub snapshot_id: String,
    /// Unix timestamp (ms) when this snapshot was taken.
    pub timestamp: u64,
    /// Per-dimension mean of the window.
    pub centroid: Vec<f64>,
    /// Aggregate variance (mean of per-dimension variances).
    pub variance: f64,
    /// Number of embeddings included.
    pub sample_count: usize,
    /// Per-dimension unbiased sample variance.
    pub covariance_diagonal: Vec<f64>,
}

/// A drift event produced when change is detected.
#[derive(Debug, Clone)]
pub struct DriftSignal {
    /// Identifier of the detector that produced this signal.
    pub detector_id: String,
    /// Classification of the drift kind.
    pub signal_type: DriftType,
    /// Magnitude in [0.0, 1.0].
    pub magnitude: f64,
    /// Statistical confidence in [0.0, 1.0].
    pub confidence: f64,
    /// Unix timestamp (ms) when the drift was detected.
    pub detected_at: u64,
    /// Dimensions that contributed most to the drift.
    pub affected_dimensions: Vec<usize>,
}

// ─── Detector Config ────────────────────────────────────────────────────────

/// Configuration for [`EmbeddingDriftDetector`].
#[derive(Debug, Clone)]
pub struct DetectorConfig {
    /// Statistical detection method.
    pub method: DetectionMethod,
    /// Number of most-recent samples kept in the rolling window.
    pub window_size: usize,
    /// Size of the reference (baseline) window.
    pub reference_window_size: usize,
    /// Minimum samples before drift detection is attempted.
    pub min_samples_before_detect: usize,
    /// Unified drift threshold (also used inside method-specific logic where needed).
    pub drift_threshold: f64,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            method: DetectionMethod::CentroidDistance(0.3),
            window_size: 100,
            reference_window_size: 100,
            min_samples_before_detect: 20,
            drift_threshold: 0.3,
        }
    }
}

// ─── Stats ──────────────────────────────────────────────────────────────────

/// Aggregate statistics for a running [`EmbeddingDriftDetector`].
#[derive(Debug, Clone, Default)]
pub struct DriftStats {
    /// Total snapshots taken.
    pub snapshots_taken: usize,
    /// Total drift events detected.
    pub drifts_detected: usize,
    /// Estimated false-positive rate (heuristic).
    pub false_positive_estimate: f64,
    /// Rolling average magnitude of detected drift signals.
    pub avg_drift_magnitude: f64,
    /// Timestamp (ms) of the most recent drift event.
    pub last_drift_at: Option<u64>,
}

// ─── Internal Page-Hinkley state ────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
struct PageHinkleyState {
    cumsum_pos: f64,
    cumsum_neg: f64,
    running_mean: f64,
    n: usize,
}

// ─── Internal CUSUM state ────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
struct CusumState {
    cumsum_pos: f64,
    cumsum_neg: f64,
    running_mean: f64,
    n: usize,
}

// ─── Main Detector ──────────────────────────────────────────────────────────

/// Detects concept drift in a stream of high-dimensional embeddings.
///
/// Maintains a rolling window and a reference window; when the rolling window
/// fills it is compared to the reference window using the configured
/// [`DetectionMethod`].
pub struct EmbeddingDriftDetector {
    /// Detector identifier (used in emitted [`DriftSignal`] values).
    pub id: String,
    /// Configuration.
    pub config: DetectorConfig,

    // Internal rolling window (most-recent samples).
    rolling: VecDeque<Vec<f64>>,
    // Reference / baseline window.
    reference: VecDeque<Vec<f64>>,
    // Established embedding dimensionality (set on first sample).
    dim: Option<usize>,

    // Page-Hinkley / CUSUM per-detector state.
    ph_state: PageHinkleyState,
    cusum_state: CusumState,

    // PRNG state for jitter / tie-breaking.
    rng_state: u64,

    // History (capped at 100 entries).
    history: VecDeque<DriftSignal>,

    // Running stats.
    stats: DriftStats,

    // Drift magnitude accumulator (for rolling average).
    magnitude_sum: f64,

    // Snapshot counter.
    snapshot_counter: u64,
}

impl EmbeddingDriftDetector {
    /// Creates a new detector with the given configuration.
    pub fn new(config: DetectorConfig) -> Self {
        // Simple entropy seed based on detector construction order.
        static COUNTER: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0x6d2c_a897_fc3b_e14d);
        let seed = COUNTER.fetch_add(0x9e37_79b9_7f4a_7c15, std::sync::atomic::Ordering::Relaxed);

        Self {
            id: format!("edd-{seed:016x}"),
            config,
            rolling: VecDeque::new(),
            reference: VecDeque::new(),
            dim: None,
            ph_state: PageHinkleyState::default(),
            cusum_state: CusumState::default(),
            rng_state: seed | 1, // ensure non-zero
            history: VecDeque::new(),
            stats: DriftStats::default(),
            magnitude_sum: 0.0,
            snapshot_counter: 0,
        }
    }

    /// Creates a new detector with a custom string identifier.
    pub fn with_id(id: impl Into<String>, config: DetectorConfig) -> Self {
        let mut det = Self::new(config);
        det.id = id.into();
        det
    }

    // ── Public API ──────────────────────────────────────────────────────────

    /// Adds one embedding to the rolling window.
    ///
    /// Returns `Ok(Some(signal))` when drift is detected, `Ok(None)` otherwise.
    pub fn add_sample(
        &mut self,
        embedding: Vec<f64>,
        timestamp: u64,
    ) -> Result<Option<DriftSignal>, DetectorError> {
        // Dimension check / initialisation.
        match self.dim {
            None => {
                if embedding.is_empty() {
                    return Err(DetectorError::ConfigurationError(
                        "empty embedding vector".into(),
                    ));
                }
                self.dim = Some(embedding.len());
            }
            Some(d) if d != embedding.len() => {
                return Err(DetectorError::DimensionMismatch {
                    expected: d,
                    got: embedding.len(),
                });
            }
            _ => {}
        }

        // Update sequential-test running stats.
        let norm = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        self.update_ph_state(norm);
        self.update_cusum_state(norm);

        // Maintain rolling window.
        self.rolling.push_back(embedding);
        if self.rolling.len() > self.config.window_size {
            self.rolling.pop_front();
        }

        // Seed the reference window on first fill.
        if self.reference.is_empty() && self.rolling.len() == self.config.window_size {
            for emb in &self.rolling {
                self.reference.push_back(emb.clone());
            }
        }

        // Attempt detection once we have enough samples.
        if self.rolling.len() < self.config.min_samples_before_detect || self.reference.is_empty() {
            return Ok(None);
        }

        // Only perform snapshot comparison when the window is full.
        if self.rolling.len() < self.config.window_size {
            return Ok(None);
        }

        let snap_current = self.take_snapshot(timestamp)?;
        self.stats.snapshots_taken += 1;

        // Build reference snapshot directly from reference window.
        let snap_ref = self.snapshot_from_window(&self.reference.clone(), timestamp)?;

        let maybe_signal = self.compare_snapshots(&snap_ref, &snap_current)?;
        if maybe_signal.magnitude > 0.0 {
            self.record_drift(&maybe_signal);
            return Ok(Some(maybe_signal));
        }

        Ok(None)
    }

    /// Computes a snapshot of the current rolling window.
    pub fn take_snapshot(&mut self, timestamp: u64) -> Result<DriftSnapshot, DetectorError> {
        self.snapshot_from_window(&self.rolling.clone(), timestamp)
    }

    /// Compares two snapshots using the configured [`DetectionMethod`].
    ///
    /// Returns a [`DriftSignal`] whose `magnitude` is `0.0` if no drift was
    /// detected, and positive otherwise.
    pub fn compare_snapshots(
        &self,
        reference: &DriftSnapshot,
        current: &DriftSnapshot,
    ) -> Result<DriftSignal, DetectorError> {
        if reference.centroid.len() != current.centroid.len() {
            return Err(DetectorError::DimensionMismatch {
                expected: reference.centroid.len(),
                got: current.centroid.len(),
            });
        }
        if reference.sample_count == 0 || current.sample_count == 0 {
            return Err(DetectorError::InsufficientData(0));
        }

        match &self.config.method {
            DetectionMethod::CentroidDistance(threshold) => {
                self.detect_centroid_distance(reference, current, *threshold)
            }
            DetectionMethod::KLDivergence(threshold) => {
                self.detect_kl_divergence(reference, current, *threshold)
            }
            DetectionMethod::PageHinkley { delta, lambda } => {
                self.detect_page_hinkley(reference, current, *delta, *lambda)
            }
            DetectionMethod::ADWIN { delta } => self.detect_adwin(reference, current, *delta),
            DetectionMethod::CUSUMDetector { k, h } => {
                self.detect_cusum(reference, current, *k, *h)
            }
        }
    }

    /// Copies the current rolling window into the reference window.
    pub fn reset_reference(&mut self) -> Result<(), DetectorError> {
        if self.rolling.is_empty() {
            return Err(DetectorError::WindowEmpty);
        }
        self.reference.clear();
        for emb in &self.rolling {
            self.reference.push_back(emb.clone());
        }
        // Reset sequential-test accumulators.
        self.ph_state = PageHinkleyState::default();
        self.cusum_state = CusumState::default();
        Ok(())
    }

    /// Returns the last 100 drift signals (oldest first).
    pub fn drift_history(&self) -> Vec<DriftSignal> {
        self.history.iter().cloned().collect()
    }

    /// Returns aggregate statistics for this detector.
    pub fn stats(&self) -> DriftStats {
        self.stats.clone()
    }

    /// Returns the number of samples in the rolling window.
    pub fn window_len(&self) -> usize {
        self.rolling.len()
    }

    /// Returns the number of samples in the reference window.
    pub fn reference_len(&self) -> usize {
        self.reference.len()
    }

    /// Returns the established embedding dimension (if any sample has been added).
    pub fn embedding_dim(&self) -> Option<usize> {
        self.dim
    }

    // ── Internal helpers ────────────────────────────────────────────────────

    fn snapshot_from_window(
        &mut self,
        window: &VecDeque<Vec<f64>>,
        timestamp: u64,
    ) -> Result<DriftSnapshot, DetectorError> {
        if window.is_empty() {
            return Err(DetectorError::WindowEmpty);
        }
        let dim = window[0].len();
        let n = window.len() as f64;

        // Compute per-dimension mean.
        let mut centroid = vec![0.0_f64; dim];
        for emb in window {
            for (c, &v) in centroid.iter_mut().zip(emb.iter()) {
                *c += v;
            }
        }
        for c in &mut centroid {
            *c /= n;
        }

        // Compute per-dimension unbiased sample variance.
        let mut cov_diag = vec![0.0_f64; dim];
        if window.len() >= 2 {
            for emb in window {
                for d in 0..dim {
                    cov_diag[d] += (emb[d] - centroid[d]).powi(2);
                }
            }
            for v in &mut cov_diag {
                *v /= n - 1.0;
            }
        }

        let variance = stat_mean(&cov_diag);

        self.snapshot_counter += 1;
        let snap_id = format!("{}-snap-{:08x}", self.id, self.snapshot_counter);

        Ok(DriftSnapshot {
            snapshot_id: snap_id,
            timestamp,
            centroid,
            variance,
            sample_count: window.len(),
            covariance_diagonal: cov_diag,
        })
    }

    fn record_drift(&mut self, signal: &DriftSignal) {
        self.stats.drifts_detected += 1;
        self.magnitude_sum += signal.magnitude;
        self.stats.avg_drift_magnitude = self.magnitude_sum / self.stats.drifts_detected as f64;
        self.stats.last_drift_at = Some(signal.detected_at);

        // Heuristic false-positive estimate:
        // proportion of detections where magnitude < threshold / 2.
        let low_mag_count = self
            .history
            .iter()
            .filter(|s| s.magnitude < self.config.drift_threshold / 2.0)
            .count() as f64;
        self.stats.false_positive_estimate = low_mag_count / self.stats.drifts_detected as f64;

        self.history.push_back(signal.clone());
        if self.history.len() > 100 {
            self.history.pop_front();
        }
    }

    fn make_signal(
        &self,
        signal_type: DriftType,
        magnitude: f64,
        confidence: f64,
        timestamp: u64,
        affected: Vec<usize>,
    ) -> DriftSignal {
        DriftSignal {
            detector_id: self.id.clone(),
            signal_type,
            magnitude: magnitude.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            detected_at: timestamp,
            affected_dimensions: affected,
        }
    }

    /// Build a "no drift" sentinel signal.
    fn no_drift_signal(&self, timestamp: u64) -> DriftSignal {
        DriftSignal {
            detector_id: self.id.clone(),
            signal_type: DriftType::ConceptDrift,
            magnitude: 0.0,
            confidence: 0.0,
            detected_at: timestamp,
            affected_dimensions: vec![],
        }
    }

    // ─── Detection implementations ──────────────────────────────────────────

    fn detect_centroid_distance(
        &self,
        reference: &DriftSnapshot,
        current: &DriftSnapshot,
        threshold: f64,
    ) -> Result<DriftSignal, DetectorError> {
        let dist = euclidean_distance(&reference.centroid, &current.centroid);
        // Cosine distance provides directional evidence complementing Euclidean distance.
        let cos_dist = cosine_distance(&reference.centroid, &current.centroid);
        let ts = current.timestamp;

        if dist <= threshold {
            return Ok(self.no_drift_signal(ts));
        }

        // Per-dimension contribution.
        let mut dim_deltas: Vec<(usize, f64)> = reference
            .centroid
            .iter()
            .zip(current.centroid.iter())
            .enumerate()
            .map(|(i, (a, b))| (i, (b - a).abs()))
            .collect();
        dim_deltas.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<usize> = dim_deltas.iter().take(5).map(|(i, _)| *i).collect();

        let magnitude = ((dist - threshold) / threshold).clamp(0.0, 1.0);
        // Blend Euclidean-based magnitude with cosine distance evidence for confidence.
        let confidence = (1.0 - (-magnitude * 3.0).exp()) * (0.8 + 0.2 * cos_dist).min(1.0);

        // Classify: large variance change → VarianceDrift; otherwise ConceptDrift.
        let var_ratio = if reference.variance > 1e-12 {
            (current.variance - reference.variance).abs() / reference.variance
        } else {
            0.0
        };
        let drift_type = if var_ratio > 0.5 {
            DriftType::VarianceDrift
        } else {
            DriftType::ConceptDrift
        };

        Ok(self.make_signal(drift_type, magnitude, confidence, ts, top))
    }

    fn detect_kl_divergence(
        &self,
        reference: &DriftSnapshot,
        current: &DriftSnapshot,
        threshold: f64,
    ) -> Result<DriftSignal, DetectorError> {
        let dim = reference.centroid.len();
        let ts = current.timestamp;
        let eps = 1e-8;

        // Per-dimension Gaussian KL: Σ[ln(σ_b/σ_a) + (σ_a² + (μ_a-μ_b)²)/(2σ_b²) - 0.5]
        let mut total_kl = 0.0_f64;
        let mut dim_kl: Vec<(usize, f64)> = Vec::with_capacity(dim);

        for d in 0..dim {
            let mu_a = reference.centroid[d];
            let mu_b = current.centroid[d];
            let sigma_a_sq = reference.covariance_diagonal[d].max(eps);
            let sigma_b_sq = current.covariance_diagonal[d].max(eps);
            let sigma_a = sigma_a_sq.sqrt();
            let sigma_b = sigma_b_sq.sqrt();
            let diff = mu_a - mu_b;

            let kl =
                (sigma_b / sigma_a).ln() + (sigma_a_sq + diff * diff) / (2.0 * sigma_b_sq) - 0.5;
            let kl = kl.max(0.0); // Numerical guard — KL ≥ 0.
            dim_kl.push((d, kl));
            total_kl += kl;
        }

        let avg_kl = total_kl / dim as f64;

        if avg_kl <= threshold {
            return Ok(self.no_drift_signal(ts));
        }

        dim_kl.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let affected: Vec<usize> = dim_kl.iter().take(5).map(|(i, _)| *i).collect();

        let magnitude = ((avg_kl - threshold) / (threshold + 1.0)).clamp(0.0, 1.0);
        let confidence = 1.0 - (-avg_kl).exp();

        Ok(self.make_signal(DriftType::ConceptDrift, magnitude, confidence, ts, affected))
    }

    fn detect_page_hinkley(
        &self,
        reference: &DriftSnapshot,
        current: &DriftSnapshot,
        delta: f64,
        lambda: f64,
    ) -> Result<DriftSignal, DetectorError> {
        let ts = current.timestamp;

        // Compute per-dimension mean change over the windows.
        let dim = reference.centroid.len();
        let mut cumsum_pos = 0.0_f64;
        let mut cumsum_neg = 0.0_f64;
        let mut dim_changes: Vec<(usize, f64)> = Vec::with_capacity(dim);

        for d in 0..dim {
            let change = (current.centroid[d] - reference.centroid[d]).abs();
            dim_changes.push((d, change));
            cumsum_pos += (change - delta).max(0.0);
            cumsum_neg += (-change - delta).max(0.0);
        }

        let test_stat = cumsum_pos.max(cumsum_neg.abs());

        if test_stat <= lambda {
            return Ok(self.no_drift_signal(ts));
        }

        dim_changes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let affected: Vec<usize> = dim_changes.iter().take(5).map(|(i, _)| *i).collect();

        let magnitude = ((test_stat - lambda) / (lambda + 1.0)).clamp(0.0, 1.0);
        let confidence = 1.0 - (-magnitude * 2.0).exp();

        // Classify abruptness: high magnitude → SuddenDrift; moderate → GradualDrift.
        let drift_type = if magnitude > 0.6 {
            DriftType::SuddenDrift
        } else {
            DriftType::GradualDrift
        };

        Ok(self.make_signal(drift_type, magnitude, confidence, ts, affected))
    }

    fn detect_adwin(
        &self,
        reference: &DriftSnapshot,
        current: &DriftSnapshot,
        delta: f64,
    ) -> Result<DriftSignal, DetectorError> {
        let ts = current.timestamp;
        let dim = reference.centroid.len();

        // ADWIN: compare sub-window means. Drift if |mean_A - mean_B| > delta × variance.
        // Use stat_variance of per-dimension variances to get a scalar spread measure.
        let all_variances: Vec<f64> = reference
            .covariance_diagonal
            .iter()
            .chain(current.covariance_diagonal.iter())
            .copied()
            .collect();
        let combined_variance = stat_variance(&all_variances)
            .max((reference.variance + current.variance) / 2.0)
            + 1e-10;

        let mut max_diff = 0.0_f64;
        let mut dim_diffs: Vec<(usize, f64)> = Vec::with_capacity(dim);

        for d in 0..dim {
            let diff = (reference.centroid[d] - current.centroid[d]).abs();
            let combined_var_d =
                (reference.covariance_diagonal[d] + current.covariance_diagonal[d]) / 2.0 + 1e-10;
            let normalised = diff / combined_var_d.sqrt();
            dim_diffs.push((d, normalised));
            if normalised > max_diff {
                max_diff = normalised;
            }
        }

        let threshold = delta * combined_variance.sqrt();
        let centroid_dist = euclidean_distance(&reference.centroid, &current.centroid);

        if centroid_dist <= threshold {
            return Ok(self.no_drift_signal(ts));
        }

        dim_diffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let affected: Vec<usize> = dim_diffs.iter().take(5).map(|(i, _)| *i).collect();

        let magnitude = ((centroid_dist - threshold) / (threshold + 1.0)).clamp(0.0, 1.0);
        let confidence = (max_diff / (delta + 1.0)).clamp(0.0, 1.0);

        Ok(self.make_signal(DriftType::ConceptDrift, magnitude, confidence, ts, affected))
    }

    fn detect_cusum(
        &self,
        reference: &DriftSnapshot,
        current: &DriftSnapshot,
        k: f64,
        h: f64,
    ) -> Result<DriftSignal, DetectorError> {
        let ts = current.timestamp;

        // CUSUM on the L2 norm of (current_centroid - reference_centroid).
        let diff_norm = euclidean_distance(&reference.centroid, &current.centroid);
        let target = reference.variance.sqrt() + 1e-8; // expected norm of small differences.

        let cusum_pos = (diff_norm - target - k).max(0.0);
        let cusum_neg = (-diff_norm + target - k).max(0.0);
        let test_stat = cusum_pos.max(cusum_neg);

        if test_stat <= h {
            return Ok(self.no_drift_signal(ts));
        }

        // Find highest-deviation dimensions.
        let dim = reference.centroid.len();
        let mut dim_deltas: Vec<(usize, f64)> = (0..dim)
            .map(|d| (d, (current.centroid[d] - reference.centroid[d]).abs()))
            .collect();
        dim_deltas.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let affected: Vec<usize> = dim_deltas.iter().take(5).map(|(i, _)| *i).collect();

        let magnitude = ((test_stat - h) / (h + 1.0)).clamp(0.0, 1.0);
        let confidence = 1.0 - (-test_stat / h).exp();

        Ok(self.make_signal(DriftType::SuddenDrift, magnitude, confidence, ts, affected))
    }

    // ─── Sequential-test running state updates ──────────────────────────────

    fn update_ph_state(&mut self, norm: f64) {
        let state = &mut self.ph_state;
        // Welford running mean.
        state.n += 1;
        let delta = norm - state.running_mean;
        state.running_mean += delta / state.n as f64;
        let change = norm - state.running_mean;
        state.cumsum_pos = (state.cumsum_pos + change).max(0.0);
        state.cumsum_neg = (state.cumsum_neg - change).max(0.0);
    }

    fn update_cusum_state(&mut self, norm: f64) {
        let state = &mut self.cusum_state;
        state.n += 1;
        let delta = norm - state.running_mean;
        state.running_mean += delta / state.n as f64;
        let k = 0.5_f64; // default slack
        state.cumsum_pos = (state.cumsum_pos + norm - state.running_mean - k).max(0.0);
        state.cumsum_neg = (state.cumsum_neg - norm + state.running_mean - k).max(0.0);
    }

    /// Draws a pseudo-random f64 from the internal PRNG.
    pub fn random_f64(&mut self) -> f64 {
        xorshift_f64(&mut self.rng_state)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_config(method: DetectionMethod) -> DetectorConfig {
        DetectorConfig {
            method,
            window_size: 20,
            reference_window_size: 20,
            min_samples_before_detect: 10,
            drift_threshold: 0.3,
        }
    }

    fn constant_emb(val: f64, dim: usize) -> Vec<f64> {
        vec![val; dim]
    }

    /// Fill a detector with `n` embeddings whose each component equals `val`.
    fn fill(det: &mut EmbeddingDriftDetector, val: f64, dim: usize, n: usize, ts_start: u64) {
        for i in 0..n {
            let _ = det.add_sample(constant_emb(val, dim), ts_start + i as u64);
        }
    }

    // ── stat_mean / stat_variance ────────────────────────────────────────────

    #[test]
    fn test_stat_mean_empty() {
        assert_eq!(stat_mean(&[]), 0.0);
    }

    #[test]
    fn test_stat_mean_values() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((stat_mean(&data) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_stat_variance_empty() {
        assert_eq!(stat_variance(&[]), 0.0);
    }

    #[test]
    fn test_stat_variance_single() {
        assert_eq!(stat_variance(&[42.0]), 0.0);
    }

    #[test]
    fn test_stat_variance_values() {
        // Unbiased variance of [2, 4, 4, 4, 5, 5, 7, 9] = 4.571...
        let data = [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((stat_variance(&data) - 4.571_428).abs() < 1e-3);
    }

    // ── cosine_distance ───────────────────────────────────────────────────────

    #[test]
    fn test_cosine_distance_identical() {
        let a = [1.0, 0.0, 0.0];
        assert!(cosine_distance(&a, &a).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_distance_zero_vector() {
        let a = [0.0, 0.0];
        let b = [1.0, 1.0];
        assert_eq!(cosine_distance(&a, &b), 1.0);
    }

    #[test]
    fn test_cosine_distance_dim_mismatch() {
        assert_eq!(cosine_distance(&[1.0, 0.0], &[1.0, 0.0, 0.0]), 1.0);
    }

    // ── euclidean_distance ────────────────────────────────────────────────────

    #[test]
    fn test_euclidean_same_point() {
        let a = [1.0, 2.0, 3.0];
        assert!(euclidean_distance(&a, &a).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_known_distance() {
        let a = [0.0, 0.0, 0.0];
        let b = [3.0, 4.0, 0.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_dim_mismatch() {
        assert!(euclidean_distance(&[1.0], &[1.0, 2.0]).is_infinite());
    }

    // ── PRNG ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift_range() {
        let mut state = 0xdeadbeef_cafebabe_u64;
        for _ in 0..1000 {
            let v = xorshift_f64(&mut state);
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_xorshift_non_constant() {
        let mut state = 12345_u64;
        let v1 = xorshift_f64(&mut state);
        let v2 = xorshift_f64(&mut state);
        assert!((v1 - v2).abs() > 1e-15);
    }

    // ── DriftDetector construction ────────────────────────────────────────────

    #[test]
    fn test_new_defaults() {
        let det = EmbeddingDriftDetector::new(DetectorConfig::default());
        assert_eq!(det.window_len(), 0);
        assert_eq!(det.reference_len(), 0);
        assert!(det.embedding_dim().is_none());
    }

    #[test]
    fn test_with_id() {
        let det = EmbeddingDriftDetector::with_id("test-detector", DetectorConfig::default());
        assert_eq!(det.id, "test-detector");
    }

    // ── add_sample: basic ─────────────────────────────────────────────────────

    #[test]
    fn test_add_sample_increments_window() {
        let mut det =
            EmbeddingDriftDetector::new(make_config(DetectionMethod::CentroidDistance(0.3)));
        det.add_sample(vec![1.0, 2.0, 3.0], 0)
            .expect("test: add_sample should succeed for valid embedding");
        assert_eq!(det.window_len(), 1);
        assert_eq!(det.embedding_dim(), Some(3));
    }

    #[test]
    fn test_add_sample_empty_error() {
        let mut det = EmbeddingDriftDetector::new(DetectorConfig::default());
        let err = det
            .add_sample(vec![], 0)
            .expect_err("test: empty embedding should return ConfigurationError");
        assert!(matches!(err, DetectorError::ConfigurationError(_)));
    }

    #[test]
    fn test_add_sample_dim_mismatch() {
        let mut det = EmbeddingDriftDetector::new(DetectorConfig::default());
        det.add_sample(vec![1.0, 2.0], 0)
            .expect("test: first add_sample should succeed");
        let err = det
            .add_sample(vec![1.0, 2.0, 3.0], 1)
            .expect_err("test: mismatched dimension should return DimensionMismatch");
        assert!(matches!(
            err,
            DetectorError::DimensionMismatch {
                expected: 2,
                got: 3
            }
        ));
    }

    #[test]
    fn test_add_sample_window_capped() {
        let cfg = DetectorConfig {
            window_size: 5,
            reference_window_size: 5,
            min_samples_before_detect: 50, // prevent detection
            ..make_config(DetectionMethod::CentroidDistance(0.3))
        };
        let mut det = EmbeddingDriftDetector::new(cfg);
        for i in 0..20_u64 {
            det.add_sample(vec![i as f64], i)
                .expect("test: add_sample should succeed for scalar embedding");
        }
        assert_eq!(det.window_len(), 5);
    }

    // ── Insufficient data ──────────────────────────────────────────────────────

    #[test]
    fn test_no_detection_below_min_samples() {
        let mut det = EmbeddingDriftDetector::new(DetectorConfig {
            min_samples_before_detect: 50,
            window_size: 20,
            reference_window_size: 20,
            ..make_config(DetectionMethod::CentroidDistance(0.3))
        });
        for i in 0..20_u64 {
            let result = det
                .add_sample(vec![0.0, 0.0], i)
                .expect("test: add_sample should succeed for zero embedding");
            assert!(result.is_none(), "should not detect with too few samples");
        }
    }

    // ── take_snapshot ─────────────────────────────────────────────────────────

    #[test]
    fn test_take_snapshot_window_empty() {
        let mut det = EmbeddingDriftDetector::new(DetectorConfig::default());
        let err = det
            .take_snapshot(0)
            .expect_err("test: take_snapshot on empty window should return WindowEmpty");
        assert_eq!(err, DetectorError::WindowEmpty);
    }

    #[test]
    fn test_take_snapshot_centroid() {
        let mut det = EmbeddingDriftDetector::new(DetectorConfig::default());
        det.add_sample(vec![1.0, 2.0], 0)
            .expect("test: add_sample should succeed for 2d embedding");
        det.add_sample(vec![3.0, 4.0], 1)
            .expect("test: add_sample should succeed for 2d embedding");
        let snap = det
            .take_snapshot(10)
            .expect("test: take_snapshot should succeed after adding samples");
        assert!((snap.centroid[0] - 2.0).abs() < 1e-10);
        assert!((snap.centroid[1] - 3.0).abs() < 1e-10);
        assert_eq!(snap.sample_count, 2);
    }

    #[test]
    fn test_take_snapshot_variance() {
        let mut det = EmbeddingDriftDetector::new(DetectorConfig::default());
        det.add_sample(vec![1.0], 0)
            .expect("test: add_sample should succeed for scalar embedding");
        det.add_sample(vec![3.0], 1)
            .expect("test: add_sample should succeed for scalar embedding");
        let snap = det
            .take_snapshot(2)
            .expect("test: take_snapshot should succeed after adding samples");
        // Unbiased variance of [1, 3] = 2.0
        assert!((snap.covariance_diagonal[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_snapshot_id_unique() {
        let mut det = EmbeddingDriftDetector::new(DetectorConfig::default());
        det.add_sample(vec![1.0], 0)
            .expect("test: add_sample should succeed for scalar embedding");
        let s1 = det
            .take_snapshot(1)
            .expect("test: first take_snapshot should succeed");
        let s2 = det
            .take_snapshot(2)
            .expect("test: second take_snapshot should succeed");
        assert_ne!(s1.snapshot_id, s2.snapshot_id);
    }

    // ── compare_snapshots: error cases ────────────────────────────────────────

    #[test]
    fn test_compare_dim_mismatch_error() {
        let det = EmbeddingDriftDetector::new(make_config(DetectionMethod::CentroidDistance(0.3)));
        let snap_a = DriftSnapshot {
            snapshot_id: "a".into(),
            timestamp: 0,
            centroid: vec![0.0, 0.0],
            variance: 0.0,
            sample_count: 10,
            covariance_diagonal: vec![0.0, 0.0],
        };
        let snap_b = DriftSnapshot {
            snapshot_id: "b".into(),
            timestamp: 1,
            centroid: vec![0.0, 0.0, 0.0],
            variance: 0.0,
            sample_count: 10,
            covariance_diagonal: vec![0.0, 0.0, 0.0],
        };
        assert!(matches!(
            det.compare_snapshots(&snap_a, &snap_b),
            Err(DetectorError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_compare_zero_samples_error() {
        let det = EmbeddingDriftDetector::new(make_config(DetectionMethod::CentroidDistance(0.3)));
        let empty_snap = DriftSnapshot {
            snapshot_id: "e".into(),
            timestamp: 0,
            centroid: vec![0.0],
            variance: 0.0,
            sample_count: 0,
            covariance_diagonal: vec![0.0],
        };
        let good_snap = DriftSnapshot {
            snapshot_id: "g".into(),
            timestamp: 0,
            centroid: vec![0.0],
            variance: 0.0,
            sample_count: 5,
            covariance_diagonal: vec![0.0],
        };
        assert!(matches!(
            det.compare_snapshots(&empty_snap, &good_snap),
            Err(DetectorError::InsufficientData(0))
        ));
    }

    // ── CentroidDistance: below threshold (no drift) ───────────────────────────

    #[test]
    fn test_centroid_distance_no_drift() {
        let det = EmbeddingDriftDetector::new(make_config(DetectionMethod::CentroidDistance(1.0)));
        let a = DriftSnapshot {
            snapshot_id: "a".into(),
            timestamp: 0,
            centroid: vec![0.0, 0.0],
            variance: 0.1,
            sample_count: 20,
            covariance_diagonal: vec![0.1, 0.1],
        };
        let b = DriftSnapshot {
            snapshot_id: "b".into(),
            timestamp: 1,
            centroid: vec![0.3, 0.4],
            variance: 0.1,
            sample_count: 20,
            covariance_diagonal: vec![0.1, 0.1],
        };
        // Euclidean(a,b) = 0.5 < threshold 1.0 → no drift
        let sig = det
            .compare_snapshots(&a, &b)
            .expect("test: compare_snapshots should succeed for valid snapshots");
        assert_eq!(sig.magnitude, 0.0);
    }

    // ── CentroidDistance: above threshold (drift) ─────────────────────────────

    #[test]
    fn test_centroid_distance_drift_detected() {
        let det = EmbeddingDriftDetector::new(make_config(DetectionMethod::CentroidDistance(0.3)));
        let a = DriftSnapshot {
            snapshot_id: "a".into(),
            timestamp: 0,
            centroid: vec![0.0, 0.0],
            variance: 0.1,
            sample_count: 20,
            covariance_diagonal: vec![0.1, 0.1],
        };
        let b = DriftSnapshot {
            snapshot_id: "b".into(),
            timestamp: 1,
            centroid: vec![3.0, 4.0],
            variance: 0.1,
            sample_count: 20,
            covariance_diagonal: vec![0.1, 0.1],
        };
        // Euclidean = 5.0 >> threshold 0.3 → drift
        let sig = det
            .compare_snapshots(&a, &b)
            .expect("test: compare_snapshots should succeed for valid snapshots");
        assert!(sig.magnitude > 0.0);
        assert!(!sig.affected_dimensions.is_empty());
    }

    // ── KLDivergence: no drift ─────────────────────────────────────────────────

    #[test]
    fn test_kl_no_drift() {
        let det = EmbeddingDriftDetector::new(make_config(DetectionMethod::KLDivergence(5.0)));
        let snap = DriftSnapshot {
            snapshot_id: "x".into(),
            timestamp: 0,
            centroid: vec![0.5, 0.5],
            variance: 1.0,
            sample_count: 20,
            covariance_diagonal: vec![1.0, 1.0],
        };
        let sig = det
            .compare_snapshots(&snap, &snap)
            .expect("test: compare_snapshots should succeed for identical snapshots");
        assert_eq!(sig.magnitude, 0.0);
    }

    // ── KLDivergence: drift ────────────────────────────────────────────────────

    #[test]
    fn test_kl_drift() {
        let det = EmbeddingDriftDetector::new(make_config(DetectionMethod::KLDivergence(0.01)));
        let a = DriftSnapshot {
            snapshot_id: "a".into(),
            timestamp: 0,
            centroid: vec![0.0, 0.0],
            variance: 0.5,
            sample_count: 20,
            covariance_diagonal: vec![0.5, 0.5],
        };
        let b = DriftSnapshot {
            snapshot_id: "b".into(),
            timestamp: 1,
            centroid: vec![5.0, 5.0],
            variance: 2.0,
            sample_count: 20,
            covariance_diagonal: vec![2.0, 2.0],
        };
        let sig = det
            .compare_snapshots(&a, &b)
            .expect("test: compare_snapshots should succeed for valid snapshots");
        assert!(sig.magnitude > 0.0);
    }

    // ── PageHinkley: no drift ──────────────────────────────────────────────────

    #[test]
    fn test_ph_no_drift() {
        let det = EmbeddingDriftDetector::new(make_config(DetectionMethod::PageHinkley {
            delta: 0.01,
            lambda: 100.0,
        }));
        let snap = DriftSnapshot {
            snapshot_id: "x".into(),
            timestamp: 0,
            centroid: vec![1.0, 1.0],
            variance: 0.1,
            sample_count: 20,
            covariance_diagonal: vec![0.1, 0.1],
        };
        let sig = det
            .compare_snapshots(&snap, &snap)
            .expect("test: compare_snapshots should succeed for identical snapshots");
        assert_eq!(sig.magnitude, 0.0);
    }

    // ── PageHinkley: drift ─────────────────────────────────────────────────────

    #[test]
    fn test_ph_drift() {
        let det = EmbeddingDriftDetector::new(make_config(DetectionMethod::PageHinkley {
            delta: 0.0,
            lambda: 0.5,
        }));
        let a = DriftSnapshot {
            snapshot_id: "a".into(),
            timestamp: 0,
            centroid: vec![0.0, 0.0],
            variance: 0.1,
            sample_count: 20,
            covariance_diagonal: vec![0.1, 0.1],
        };
        let b = DriftSnapshot {
            snapshot_id: "b".into(),
            timestamp: 1,
            centroid: vec![1.0, 1.0],
            variance: 0.1,
            sample_count: 20,
            covariance_diagonal: vec![0.1, 0.1],
        };
        let sig = det
            .compare_snapshots(&a, &b)
            .expect("test: compare_snapshots should succeed for valid snapshots");
        assert!(sig.magnitude > 0.0);
    }

    // ── ADWIN: no drift ────────────────────────────────────────────────────────

    #[test]
    fn test_adwin_no_drift() {
        let det = EmbeddingDriftDetector::new(make_config(DetectionMethod::ADWIN { delta: 0.01 }));
        let snap = DriftSnapshot {
            snapshot_id: "x".into(),
            timestamp: 0,
            centroid: vec![0.5, 0.5],
            variance: 1.0,
            sample_count: 20,
            covariance_diagonal: vec![1.0, 1.0],
        };
        let sig = det
            .compare_snapshots(&snap, &snap)
            .expect("test: compare_snapshots should succeed for identical snapshots");
        assert_eq!(sig.magnitude, 0.0);
    }

    // ── ADWIN: drift ───────────────────────────────────────────────────────────

    #[test]
    fn test_adwin_drift() {
        let det = EmbeddingDriftDetector::new(make_config(DetectionMethod::ADWIN { delta: 0.001 }));
        let a = DriftSnapshot {
            snapshot_id: "a".into(),
            timestamp: 0,
            centroid: vec![0.0, 0.0],
            variance: 0.01,
            sample_count: 20,
            covariance_diagonal: vec![0.01, 0.01],
        };
        let b = DriftSnapshot {
            snapshot_id: "b".into(),
            timestamp: 1,
            centroid: vec![10.0, 10.0],
            variance: 0.01,
            sample_count: 20,
            covariance_diagonal: vec![0.01, 0.01],
        };
        let sig = det
            .compare_snapshots(&a, &b)
            .expect("test: compare_snapshots should succeed for valid snapshots");
        assert!(sig.magnitude > 0.0);
    }

    // ── CUSUMDetector: no drift ───────────────────────────────────────────────

    #[test]
    fn test_cusum_no_drift() {
        let det = EmbeddingDriftDetector::new(make_config(DetectionMethod::CUSUMDetector {
            k: 0.5,
            h: 100.0,
        }));
        let snap = DriftSnapshot {
            snapshot_id: "x".into(),
            timestamp: 0,
            centroid: vec![1.0, 1.0],
            variance: 1.0,
            sample_count: 20,
            covariance_diagonal: vec![1.0, 1.0],
        };
        let sig = det
            .compare_snapshots(&snap, &snap)
            .expect("test: compare_snapshots should succeed for identical snapshots");
        assert_eq!(sig.magnitude, 0.0);
    }

    // ── CUSUMDetector: drift ──────────────────────────────────────────────────

    #[test]
    fn test_cusum_drift() {
        let det = EmbeddingDriftDetector::new(make_config(DetectionMethod::CUSUMDetector {
            k: 0.0,
            h: 0.5,
        }));
        let a = DriftSnapshot {
            snapshot_id: "a".into(),
            timestamp: 0,
            centroid: vec![0.0, 0.0],
            variance: 0.01,
            sample_count: 20,
            covariance_diagonal: vec![0.01, 0.01],
        };
        let b = DriftSnapshot {
            snapshot_id: "b".into(),
            timestamp: 1,
            centroid: vec![5.0, 5.0],
            variance: 0.01,
            sample_count: 20,
            covariance_diagonal: vec![0.01, 0.01],
        };
        let sig = det
            .compare_snapshots(&a, &b)
            .expect("test: compare_snapshots should succeed for valid snapshots");
        assert!(sig.magnitude > 0.0);
    }

    // ── reset_reference ────────────────────────────────────────────────────────

    #[test]
    fn test_reset_reference_empty_error() {
        let mut det = EmbeddingDriftDetector::new(DetectorConfig::default());
        assert_eq!(det.reset_reference(), Err(DetectorError::WindowEmpty));
    }

    #[test]
    fn test_reset_reference_updates() {
        let cfg = DetectorConfig {
            window_size: 10,
            reference_window_size: 10,
            min_samples_before_detect: 100, // no detection yet
            ..make_config(DetectionMethod::CentroidDistance(0.3))
        };
        let mut det = EmbeddingDriftDetector::new(cfg);
        fill(&mut det, 0.5, 3, 5, 0);
        det.reset_reference()
            .expect("test: reset_reference should succeed after adding samples");
        assert_eq!(det.reference_len(), 5);
    }

    // ── drift_history ──────────────────────────────────────────────────────────

    #[test]
    fn test_drift_history_initially_empty() {
        let det = EmbeddingDriftDetector::new(DetectorConfig::default());
        assert!(det.drift_history().is_empty());
    }

    #[test]
    fn test_drift_history_capped_at_100() {
        let cfg = DetectorConfig {
            method: DetectionMethod::CentroidDistance(0.001),
            window_size: 20,
            reference_window_size: 20,
            min_samples_before_detect: 10,
            drift_threshold: 0.001,
        };
        let mut det = EmbeddingDriftDetector::new(cfg);
        // Build a reference window.
        fill(&mut det, 0.1, 2, 20, 0);

        // Now send 110 "drifted" batches (each refills a full window).
        // We cannot easily trigger 110 signals via add_sample because we need
        // a full window each time. Instead, fake it via record_drift.
        for i in 0..110_u64 {
            let sig = DriftSignal {
                detector_id: det.id.clone(),
                signal_type: DriftType::ConceptDrift,
                magnitude: 0.5,
                confidence: 0.9,
                detected_at: i,
                affected_dimensions: vec![],
            };
            det.record_drift(&sig);
        }
        assert_eq!(det.drift_history().len(), 100);
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial() {
        let det = EmbeddingDriftDetector::new(DetectorConfig::default());
        let s = det.stats();
        assert_eq!(s.snapshots_taken, 0);
        assert_eq!(s.drifts_detected, 0);
        assert!(s.last_drift_at.is_none());
    }

    #[test]
    fn test_stats_incremented_on_drift() {
        let mut det = EmbeddingDriftDetector::new(DetectorConfig::default());
        let sig = DriftSignal {
            detector_id: det.id.clone(),
            signal_type: DriftType::SuddenDrift,
            magnitude: 0.8,
            confidence: 0.9,
            detected_at: 42,
            affected_dimensions: vec![0, 1],
        };
        det.record_drift(&sig);
        let s = det.stats();
        assert_eq!(s.drifts_detected, 1);
        assert_eq!(s.last_drift_at, Some(42));
        assert!((s.avg_drift_magnitude - 0.8).abs() < 1e-10);
    }

    // ── End-to-end: add_sample triggers drift signal ──────────────────────────

    #[test]
    fn test_end_to_end_no_drift() {
        let cfg = DetectorConfig {
            method: DetectionMethod::CentroidDistance(2.0),
            window_size: 20,
            reference_window_size: 20,
            min_samples_before_detect: 10,
            drift_threshold: 2.0,
        };
        let mut det = EmbeddingDriftDetector::new(cfg);
        let mut any_signal = false;
        for i in 0..40_u64 {
            let result = det
                .add_sample(constant_emb(0.1, 4), i)
                .expect("test: add_sample should succeed for constant embedding");
            if result.is_some() {
                any_signal = true;
            }
        }
        assert!(!any_signal, "constant embeddings should not trigger drift");
    }

    #[test]
    fn test_end_to_end_drift_detected() {
        let cfg = DetectorConfig {
            method: DetectionMethod::CentroidDistance(0.1),
            window_size: 20,
            reference_window_size: 20,
            min_samples_before_detect: 10,
            drift_threshold: 0.1,
        };
        let mut det = EmbeddingDriftDetector::new(cfg);

        // Reference phase: fill window with near-zero embeddings.
        fill(&mut det, 0.0, 4, 20, 0);
        det.reset_reference()
            .expect("test: reset_reference should succeed after filling the window");

        // Drift phase: add very different embeddings until window refills.
        let mut drift_found = false;
        for i in 20..50_u64 {
            if let Ok(Some(_)) = det.add_sample(constant_emb(100.0, 4), i) {
                drift_found = true;
                break;
            }
        }
        assert!(drift_found, "large centroid shift should trigger drift");
    }

    #[test]
    fn test_end_to_end_kl_drift() {
        let cfg = DetectorConfig {
            method: DetectionMethod::KLDivergence(0.01),
            window_size: 20,
            reference_window_size: 20,
            min_samples_before_detect: 10,
            drift_threshold: 0.01,
        };
        let mut det = EmbeddingDriftDetector::new(cfg);
        fill(&mut det, 0.1, 3, 20, 0);
        det.reset_reference()
            .expect("test: reset_reference should succeed after filling the window");

        let mut drift_found = false;
        for i in 20..60_u64 {
            if let Ok(Some(_)) = det.add_sample(constant_emb(50.0, 3), i) {
                drift_found = true;
                break;
            }
        }
        assert!(drift_found);
    }

    #[test]
    fn test_end_to_end_ph_drift() {
        let cfg = DetectorConfig {
            method: DetectionMethod::PageHinkley {
                delta: 0.0,
                lambda: 0.5,
            },
            window_size: 20,
            reference_window_size: 20,
            min_samples_before_detect: 10,
            drift_threshold: 0.5,
        };
        let mut det = EmbeddingDriftDetector::new(cfg);
        fill(&mut det, 0.0, 2, 20, 0);
        det.reset_reference()
            .expect("test: reset_reference should succeed after filling the window");

        let mut drift_found = false;
        for i in 20..60_u64 {
            if let Ok(Some(_)) = det.add_sample(constant_emb(50.0, 2), i) {
                drift_found = true;
                break;
            }
        }
        assert!(drift_found);
    }

    #[test]
    fn test_end_to_end_adwin_drift() {
        let cfg = DetectorConfig {
            method: DetectionMethod::ADWIN { delta: 0.001 },
            window_size: 20,
            reference_window_size: 20,
            min_samples_before_detect: 10,
            drift_threshold: 0.001,
        };
        let mut det = EmbeddingDriftDetector::new(cfg);
        fill(&mut det, 0.0, 2, 20, 0);
        det.reset_reference()
            .expect("test: reset_reference should succeed after filling the window");

        let mut drift_found = false;
        for i in 20..60_u64 {
            if let Ok(Some(_)) = det.add_sample(constant_emb(50.0, 2), i) {
                drift_found = true;
                break;
            }
        }
        assert!(drift_found);
    }

    #[test]
    fn test_end_to_end_cusum_drift() {
        let cfg = DetectorConfig {
            method: DetectionMethod::CUSUMDetector { k: 0.0, h: 0.5 },
            window_size: 20,
            reference_window_size: 20,
            min_samples_before_detect: 10,
            drift_threshold: 0.5,
        };
        let mut det = EmbeddingDriftDetector::new(cfg);
        fill(&mut det, 0.0, 2, 20, 0);
        det.reset_reference()
            .expect("test: reset_reference should succeed after filling the window");

        let mut drift_found = false;
        for i in 20..60_u64 {
            if let Ok(Some(_)) = det.add_sample(constant_emb(50.0, 2), i) {
                drift_found = true;
                break;
            }
        }
        assert!(drift_found);
    }

    // ── DriftType coverage ────────────────────────────────────────────────────

    #[test]
    fn test_drift_type_variance() {
        // Trigger VarianceDrift branch via CentroidDistance with high variance ratio.
        let det = EmbeddingDriftDetector::new(make_config(DetectionMethod::CentroidDistance(0.01)));
        let a = DriftSnapshot {
            snapshot_id: "a".into(),
            timestamp: 0,
            centroid: vec![0.0],
            variance: 0.0001,
            sample_count: 20,
            covariance_diagonal: vec![0.0001],
        };
        let b = DriftSnapshot {
            snapshot_id: "b".into(),
            timestamp: 1,
            centroid: vec![0.1],
            variance: 10.0,
            sample_count: 20,
            covariance_diagonal: vec![10.0],
        };
        let sig = det
            .compare_snapshots(&a, &b)
            .expect("test: compare_snapshots should succeed for valid snapshots");
        if sig.magnitude > 0.0 {
            assert!(matches!(
                sig.signal_type,
                DriftType::VarianceDrift | DriftType::ConceptDrift
            ));
        }
    }

    #[test]
    fn test_drift_type_sudden() {
        let det = EmbeddingDriftDetector::new(make_config(DetectionMethod::PageHinkley {
            delta: 0.0,
            lambda: 0.01,
        }));
        let a = DriftSnapshot {
            snapshot_id: "a".into(),
            timestamp: 0,
            centroid: vec![0.0],
            variance: 0.01,
            sample_count: 20,
            covariance_diagonal: vec![0.01],
        };
        let b = DriftSnapshot {
            snapshot_id: "b".into(),
            timestamp: 1,
            centroid: vec![100.0],
            variance: 0.01,
            sample_count: 20,
            covariance_diagonal: vec![0.01],
        };
        let sig = det
            .compare_snapshots(&a, &b)
            .expect("test: compare_snapshots should succeed for valid snapshots");
        if sig.magnitude > 0.6 {
            assert!(matches!(sig.signal_type, DriftType::SuddenDrift));
        }
    }

    #[test]
    fn test_drift_type_gradual() {
        let det = EmbeddingDriftDetector::new(make_config(DetectionMethod::PageHinkley {
            delta: 0.0,
            lambda: 0.001,
        }));
        let a = DriftSnapshot {
            snapshot_id: "a".into(),
            timestamp: 0,
            centroid: vec![0.0],
            variance: 0.1,
            sample_count: 20,
            covariance_diagonal: vec![0.1],
        };
        let b = DriftSnapshot {
            snapshot_id: "b".into(),
            timestamp: 1,
            centroid: vec![0.3],
            variance: 0.1,
            sample_count: 20,
            covariance_diagonal: vec![0.1],
        };
        let sig = det
            .compare_snapshots(&a, &b)
            .expect("test: compare_snapshots should succeed for valid snapshots");
        if sig.magnitude > 0.0 && sig.magnitude <= 0.6 {
            assert!(matches!(sig.signal_type, DriftType::GradualDrift));
        }
    }

    // ── Magnitude / confidence bounds ─────────────────────────────────────────

    #[test]
    fn test_magnitude_in_range() {
        let det =
            EmbeddingDriftDetector::new(make_config(DetectionMethod::CentroidDistance(0.001)));
        let a = DriftSnapshot {
            snapshot_id: "a".into(),
            timestamp: 0,
            centroid: vec![0.0, 0.0, 0.0],
            variance: 0.01,
            sample_count: 20,
            covariance_diagonal: vec![0.01, 0.01, 0.01],
        };
        let b = DriftSnapshot {
            snapshot_id: "b".into(),
            timestamp: 1,
            centroid: vec![1e9, 1e9, 1e9],
            variance: 0.01,
            sample_count: 20,
            covariance_diagonal: vec![0.01, 0.01, 0.01],
        };
        let sig = det
            .compare_snapshots(&a, &b)
            .expect("test: compare_snapshots should succeed for valid snapshots");
        assert!((0.0..=1.0).contains(&sig.magnitude));
        assert!((0.0..=1.0).contains(&sig.confidence));
    }

    // ── random_f64 ────────────────────────────────────────────────────────────

    #[test]
    fn test_random_f64_range() {
        let mut det = EmbeddingDriftDetector::new(DetectorConfig::default());
        for _ in 0..100 {
            let v = det.random_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    // ── DetectorError Display ─────────────────────────────────────────────────

    #[test]
    fn test_error_display_insufficient() {
        let e = DetectorError::InsufficientData(5);
        assert!(e.to_string().contains("5"));
    }

    #[test]
    fn test_error_display_dim_mismatch() {
        let e = DetectorError::DimensionMismatch {
            expected: 3,
            got: 5,
        };
        let s = e.to_string();
        assert!(s.contains("3") && s.contains("5"));
    }

    #[test]
    fn test_error_display_window_empty() {
        assert!(DetectorError::WindowEmpty.to_string().contains("empty"));
    }

    #[test]
    fn test_error_display_config() {
        let e = DetectorError::ConfigurationError("bad config".into());
        assert!(e.to_string().contains("bad config"));
    }

    // ── Snapshot from multi-dim window ────────────────────────────────────────

    #[test]
    fn test_snapshot_multi_dim() {
        let mut det = EmbeddingDriftDetector::new(DetectorConfig::default());
        det.add_sample(vec![1.0, 10.0, 100.0], 0)
            .expect("test: add_sample should succeed for 3d embedding");
        det.add_sample(vec![3.0, 30.0, 300.0], 1)
            .expect("test: add_sample should succeed for 3d embedding");
        let snap = det
            .take_snapshot(2)
            .expect("test: take_snapshot should succeed after adding samples");
        assert!((snap.centroid[0] - 2.0).abs() < 1e-10);
        assert!((snap.centroid[1] - 20.0).abs() < 1e-10);
        assert!((snap.centroid[2] - 200.0).abs() < 1e-10);
        assert_eq!(snap.covariance_diagonal.len(), 3);
    }

    // ── KL divergence with identical distributions ────────────────────────────

    #[test]
    fn test_kl_identical_distributions() {
        let det = EmbeddingDriftDetector::new(make_config(DetectionMethod::KLDivergence(0.001)));
        let snap = DriftSnapshot {
            snapshot_id: "s".into(),
            timestamp: 0,
            centroid: vec![0.5, 0.5],
            variance: 1.0,
            sample_count: 20,
            covariance_diagonal: vec![1.0, 1.0],
        };
        // KL(P||P) should be ~0, so no drift.
        let sig = det
            .compare_snapshots(&snap, &snap)
            .expect("test: compare_snapshots should succeed for identical snapshots");
        assert_eq!(sig.magnitude, 0.0);
    }

    // ── Snapshot sample count ─────────────────────────────────────────────────

    #[test]
    fn test_snapshot_sample_count() {
        let mut det = EmbeddingDriftDetector::new(DetectorConfig::default());
        for i in 0..7_u64 {
            det.add_sample(vec![i as f64], i)
                .expect("test: add_sample should succeed for scalar embedding");
        }
        let snap = det
            .take_snapshot(100)
            .expect("test: take_snapshot should succeed after adding samples");
        assert_eq!(snap.sample_count, 7);
    }

    // ── Reference window seed ─────────────────────────────────────────────────

    #[test]
    fn test_reference_seeded_on_fill() {
        let cfg = DetectorConfig {
            window_size: 5,
            reference_window_size: 5,
            min_samples_before_detect: 100,
            ..make_config(DetectionMethod::CentroidDistance(0.3))
        };
        let mut det = EmbeddingDriftDetector::new(cfg);
        fill(&mut det, 0.0, 2, 5, 0);
        assert_eq!(det.reference_len(), 5);
    }

    // ── Multiple detectors have distinct IDs ──────────────────────────────────

    #[test]
    fn test_distinct_ids() {
        let d1 = EmbeddingDriftDetector::new(DetectorConfig::default());
        let d2 = EmbeddingDriftDetector::new(DetectorConfig::default());
        assert_ne!(d1.id, d2.id);
    }

    // ── Signal detector_id matches ────────────────────────────────────────────

    #[test]
    fn test_signal_detector_id_matches() {
        let cfg = DetectorConfig {
            method: DetectionMethod::CentroidDistance(0.001),
            window_size: 10,
            reference_window_size: 10,
            min_samples_before_detect: 5,
            drift_threshold: 0.001,
        };
        let mut det = EmbeddingDriftDetector::with_id("my-detector", cfg);
        fill(&mut det, 0.0, 2, 10, 0);
        det.reset_reference()
            .expect("test: reset_reference should succeed after filling the window");

        for i in 10..30_u64 {
            if let Ok(Some(sig)) = det.add_sample(constant_emb(100.0, 2), i) {
                assert_eq!(sig.detector_id, "my-detector");
                break;
            }
        }
    }

    // ── DriftStats false_positive_estimate ────────────────────────────────────

    #[test]
    fn test_false_positive_estimate_low_mag() {
        let mut det = EmbeddingDriftDetector::new(DetectorConfig {
            drift_threshold: 0.5,
            ..DetectorConfig::default()
        });
        // Add many low-magnitude signals.
        for i in 0..10_u64 {
            let sig = DriftSignal {
                detector_id: det.id.clone(),
                signal_type: DriftType::ConceptDrift,
                magnitude: 0.1, // below threshold/2 = 0.25
                confidence: 0.5,
                detected_at: i,
                affected_dimensions: vec![],
            };
            det.record_drift(&sig);
        }
        // All signals are low-magnitude → FP estimate ≈ 1.0
        assert!(det.stats().false_positive_estimate > 0.5);
    }
}
