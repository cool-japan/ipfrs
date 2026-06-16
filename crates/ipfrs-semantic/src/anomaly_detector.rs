//! Vector Anomaly Detector
//!
//! Detects anomalous vectors relative to a reference distribution using
//! z-score and isolation-score techniques with FNV-1a seeded random projections.

// ── FNV-1a constants ─────────────────────────────────────────────────────────
const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;

/// Perform one step of FNV-1a hashing.
#[inline]
pub(crate) fn fnv1a_step(hash: u64, byte: u8) -> u64 {
    (hash ^ (byte as u64)).wrapping_mul(FNV_PRIME)
}

/// Generate a pseudo-random `f64` in `[0, 1)` from a seed and an index using FNV-1a.
///
/// Used by the [`AnomalyMethod::IsolationScore`] detector to derive per-dimension
/// random splits from `vector_id`.
#[inline]
pub(crate) fn fnv1a_rand_f64(seed: u64, index: u64) -> f64 {
    let mut h = FNV_OFFSET_BASIS;
    for b in seed.to_le_bytes() {
        h = fnv1a_step(h, b);
    }
    for b in index.to_le_bytes() {
        h = fnv1a_step(h, b);
    }
    // Map to [0, 1) via mantissa bits
    let mantissa = h >> 11; // 53 significant bits
    (mantissa as f64) / (1u64 << 53) as f64
}

// ─────────────────────────────────────────────────────────────────────────────

/// Method used for anomaly detection.
#[derive(Clone, Debug, PartialEq)]
pub enum AnomalyMethod {
    /// Flag if any dimension's z-score exceeds the threshold.
    ZScore,
    /// Mahalanobis-like score using per-dimension std-dev normalisation.
    MahalanobisApprox,
    /// Random-projection isolation score (FNV-1a seeded splits).
    IsolationScore,
}

/// Result produced for a single vector by [`VectorAnomalyDetector::detect`].
#[derive(Clone, Debug)]
pub struct AnomalyResult {
    /// Identifier of the checked vector.
    pub vector_id: u64,
    /// Anomaly score — higher values are more anomalous.
    pub score: f64,
    /// Whether the vector is considered anomalous.
    pub is_anomaly: bool,
    /// Detection method used.
    pub method: AnomalyMethod,
    /// Up to 5 dimension indices that contributed most (by |z-score|).
    pub flagged_dims: Vec<usize>,
}

/// Configuration for [`VectorAnomalyDetector`].
#[derive(Clone, Debug)]
pub struct DetectorConfig {
    /// Detection method.
    pub method: AnomalyMethod,
    /// Anomaly threshold.
    ///
    /// Defaults: `3.0` for [`AnomalyMethod::ZScore`] /
    /// [`AnomalyMethod::MahalanobisApprox`], `0.7` for
    /// [`AnomalyMethod::IsolationScore`].
    pub threshold: f64,
    /// Minimum number of reference vectors required before detection starts.
    pub min_samples: usize,
    /// Maximum number of reference vectors to retain (drop oldest on overflow).
    pub max_reference: usize,
}

impl DetectorConfig {
    /// Create a config with sensible defaults for the given method.
    pub fn with_method(method: AnomalyMethod) -> Self {
        let threshold = match method {
            AnomalyMethod::IsolationScore => 0.7,
            _ => 3.0,
        };
        Self {
            method,
            threshold,
            min_samples: 10,
            max_reference: 1000,
        }
    }
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self::with_method(AnomalyMethod::ZScore)
    }
}

/// Running statistics for a [`VectorAnomalyDetector`].
#[derive(Clone, Debug, Default)]
pub struct DetectorStats {
    /// Number of reference vectors currently held.
    pub reference_count: usize,
    /// Total number of vectors that have been checked.
    pub total_checked: u64,
    /// Total number of vectors flagged as anomalous.
    pub total_anomalies: u64,
}

impl DetectorStats {
    /// Fraction of checked vectors that were flagged as anomalous.
    ///
    /// Returns `0.0` if no vectors have been checked yet.
    pub fn anomaly_rate(&self) -> f64 {
        if self.total_checked == 0 {
            0.0
        } else {
            self.total_anomalies as f64 / self.total_checked as f64
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Detects anomalous vectors relative to a sliding-window reference distribution.
///
/// # Example
/// ```
/// use ipfrs_semantic::anomaly_detector::{
///     AnomalyMethod, DetectorConfig, VectorAnomalyDetector,
/// };
///
/// let config = DetectorConfig::with_method(AnomalyMethod::ZScore);
/// let mut detector = VectorAnomalyDetector::new(config);
///
/// // Add reference vectors (all roughly [0.0, 0.0, 0.0])
/// for _ in 0..12 {
///     detector.add_reference(vec![0.0_f32, 0.0, 0.0]);
/// }
///
/// // A normal vector should not be flagged
/// let normal = detector.detect(1, &[0.1, -0.1, 0.05]);
/// assert!(normal.is_some());
///
/// // A large outlier should be flagged
/// let outlier = detector.detect(2, &[100.0, 100.0, 100.0]);
/// assert!(outlier.map_or(false, |r| r.is_anomaly));
/// ```
pub struct VectorAnomalyDetector {
    /// Sliding-window set of reference vectors.
    reference: Vec<Vec<f32>>,
    /// Detector configuration.
    config: DetectorConfig,
    /// Running statistics.
    stats: DetectorStats,
}

impl VectorAnomalyDetector {
    /// Create a new detector with the given configuration.
    pub fn new(config: DetectorConfig) -> Self {
        Self {
            reference: Vec::new(),
            config,
            stats: DetectorStats::default(),
        }
    }

    /// Add a reference vector.
    ///
    /// If the reference set has reached `max_reference`, the oldest vector is
    /// evicted before the new one is appended.
    pub fn add_reference(&mut self, vec: Vec<f32>) {
        if self.reference.len() >= self.config.max_reference {
            self.reference.remove(0);
        }
        self.reference.push(vec);
        self.stats.reference_count = self.reference.len();
    }

    /// Compute per-dimension mean and standard deviation over the reference set.
    ///
    /// Returns `(means, stds)` where each slice has length equal to the
    /// dimensionality of the reference vectors.  The std is clamped to a
    /// minimum of `1e-6` to avoid division by zero.
    ///
    /// Panics if the reference set is empty.
    pub fn compute_mean_std(&self) -> (Vec<f32>, Vec<f32>) {
        let dims = self.reference[0].len();
        let n = self.reference.len() as f32;

        let mut means = vec![0.0_f32; dims];
        for vec in &self.reference {
            for (d, &v) in vec.iter().enumerate() {
                means[d] += v;
            }
        }
        for m in &mut means {
            *m /= n;
        }

        let mut vars = vec![0.0_f32; dims];
        for vec in &self.reference {
            for (d, &v) in vec.iter().enumerate() {
                let diff = v - means[d];
                vars[d] += diff * diff;
            }
        }
        let stds: Vec<f32> = vars.iter().map(|&v| (v / n).sqrt().max(1e-6_f32)).collect();

        (means, stds)
    }

    /// Compute z-scores for each dimension and return top-5 flagged indices.
    fn compute_z_scores(vec: &[f32], means: &[f32], stds: &[f32]) -> Vec<f64> {
        vec.iter()
            .zip(means.iter())
            .zip(stds.iter())
            .map(|((&v, &m), &s)| ((v - m) / s).abs() as f64)
            .collect()
    }

    /// Return the indices of the top-5 dimensions by z-score magnitude.
    fn top5_flagged(z_scores: &[f64]) -> Vec<usize> {
        let mut indexed: Vec<(usize, f64)> =
            z_scores.iter().enumerate().map(|(i, &z)| (i, z)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(5);
        indexed.into_iter().map(|(i, _)| i).collect()
    }

    /// Check a single vector against the reference distribution.
    ///
    /// Returns `None` if the reference set is smaller than `min_samples`.
    pub fn detect(&mut self, vector_id: u64, vec: &[f32]) -> Option<AnomalyResult> {
        if self.reference.len() < self.config.min_samples {
            return None;
        }

        let (means, stds) = self.compute_mean_std();
        let dims = means.len();
        let z_scores = Self::compute_z_scores(vec, &means, &stds);
        let flagged_dims = Self::top5_flagged(&z_scores);
        let threshold = self.config.threshold;

        let (score, is_anomaly) = match self.config.method {
            AnomalyMethod::ZScore => {
                let max_z = z_scores.iter().cloned().fold(0.0_f64, f64::max);
                let anomaly = z_scores.iter().any(|&z| z > threshold);
                (max_z, anomaly)
            }

            AnomalyMethod::MahalanobisApprox => {
                let sum_sq: f64 = z_scores.iter().map(|&z| z * z).sum();
                let score = (sum_sq / dims as f64).sqrt();
                (score, score > threshold)
            }

            AnomalyMethod::IsolationScore => {
                // Use FNV-1a hash of vector_id as seed for per-dimension random splits.
                // A dimension is considered isolated if its normalised deviation exceeds
                // the random split value drawn for that dimension.
                let seed = vector_id;
                let outlier_count = z_scores
                    .iter()
                    .enumerate()
                    .filter(|&(dim_idx, &z)| {
                        // Split threshold drawn from [0, 1) for this (seed, dim) pair
                        let split = fnv1a_rand_f64(seed, dim_idx as u64);
                        // A dimension is "isolated" if |z| > 1.0 AND exceeds the random split
                        z > 1.0 && (z / (z + 1.0)) > split
                    })
                    .count();
                let score = outlier_count as f64 / dims as f64;
                (score, score > threshold)
            }
        };

        self.stats.total_checked += 1;
        if is_anomaly {
            self.stats.total_anomalies += 1;
        }

        Some(AnomalyResult {
            vector_id,
            score,
            is_anomaly,
            method: self.config.method.clone(),
            flagged_dims,
        })
    }

    /// Access running statistics.
    pub fn stats(&self) -> &DetectorStats {
        &self.stats
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn build_detector(method: AnomalyMethod, n_ref: usize, dims: usize) -> VectorAnomalyDetector {
        let config = DetectorConfig::with_method(method);
        let mut det = VectorAnomalyDetector::new(config);
        for _ in 0..n_ref {
            det.add_reference(vec![0.0_f32; dims]);
        }
        det
    }

    fn build_detector_with_refs(
        method: AnomalyMethod,
        refs: Vec<Vec<f32>>,
    ) -> VectorAnomalyDetector {
        let config = DetectorConfig::with_method(method);
        let mut det = VectorAnomalyDetector::new(config);
        for r in refs {
            det.add_reference(r);
        }
        det
    }

    // ── reference management ─────────────────────────────────────────────────

    #[test]
    fn test_add_reference_basic() {
        let config = DetectorConfig::with_method(AnomalyMethod::ZScore);
        let mut det = VectorAnomalyDetector::new(config);
        for i in 0..5 {
            det.add_reference(vec![i as f32]);
        }
        assert_eq!(det.reference.len(), 5);
        assert_eq!(det.stats().reference_count, 5);
    }

    #[test]
    fn test_add_reference_evicts_oldest_at_max() {
        let mut config = DetectorConfig::with_method(AnomalyMethod::ZScore);
        config.max_reference = 3;
        let mut det = VectorAnomalyDetector::new(config);
        // Push sentinel "oldest" vector first
        det.add_reference(vec![999.0_f32]);
        det.add_reference(vec![1.0_f32]);
        det.add_reference(vec![2.0_f32]);
        // Pushing a 4th should evict 999.0
        det.add_reference(vec![3.0_f32]);
        assert_eq!(det.reference.len(), 3);
        assert_eq!(det.stats().reference_count, 3);
        // The oldest (999.0) must no longer be present
        assert!(!det.reference.iter().any(|v| v[0] == 999.0_f32));
        // The newest value must be present
        assert!(det.reference.iter().any(|v| v[0] == 3.0_f32));
    }

    #[test]
    fn test_add_reference_exactly_at_max_no_eviction() {
        let mut config = DetectorConfig::with_method(AnomalyMethod::ZScore);
        config.max_reference = 5;
        let mut det = VectorAnomalyDetector::new(config);
        for i in 0..5 {
            det.add_reference(vec![i as f32]);
        }
        assert_eq!(det.reference.len(), 5);
    }

    // ── detect below min_samples ─────────────────────────────────────────────

    #[test]
    fn test_detect_none_below_min_samples() {
        let config = DetectorConfig::with_method(AnomalyMethod::ZScore);
        let mut det = VectorAnomalyDetector::new(config); // min_samples = 10
        for _ in 0..9 {
            det.add_reference(vec![0.0_f32]);
        }
        let result = det.detect(1, &[0.0]);
        assert!(result.is_none());
    }

    #[test]
    fn test_detect_some_at_min_samples() {
        let config = DetectorConfig::with_method(AnomalyMethod::ZScore);
        let mut det = VectorAnomalyDetector::new(config);
        for _ in 0..10 {
            det.add_reference(vec![0.0_f32]);
        }
        let result = det.detect(1, &[0.0]);
        assert!(result.is_some());
    }

    // ── ZScore method ────────────────────────────────────────────────────────

    #[test]
    fn test_zscore_detects_clear_outlier() {
        let mut det = build_detector(AnomalyMethod::ZScore, 20, 3);
        // All refs are [0,0,0] with near-zero std → any large value will have massive z-score
        let result = det
            .detect(42, &[100.0, 100.0, 100.0])
            .expect("should return Some");
        assert!(result.is_anomaly, "Expected outlier to be flagged");
        assert!(result.score > 3.0, "score={}", result.score);
    }

    #[test]
    fn test_zscore_no_anomaly_for_mean_vector() {
        let mut det = build_detector_with_refs(
            AnomalyMethod::ZScore,
            (0..20).map(|i| vec![i as f32, -(i as f32)]).collect(),
        );
        // Mean of [0..19] = 9.5, mean of [-0..-19] = -9.5
        let result = det
            .detect(1, &[9.5_f32, -9.5_f32])
            .expect("should return Some");
        assert!(
            !result.is_anomaly,
            "Mean vector should not be an anomaly; score={}",
            result.score
        );
    }

    #[test]
    fn test_zscore_score_is_max_z() {
        // Build refs with known variance: all refs are 0; a query of [10,0] gives max_z = 10/std
        let mut det = build_detector(AnomalyMethod::ZScore, 10, 2);
        let result = det.detect(1, &[10.0_f32, 0.0]).expect("Some");
        // score should be much greater than 0
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_zscore_method_field_in_result() {
        let mut det = build_detector(AnomalyMethod::ZScore, 10, 2);
        let result = det.detect(7, &[0.0, 0.0]).expect("Some");
        assert_eq!(result.method, AnomalyMethod::ZScore);
    }

    // ── MahalanobisApprox method ─────────────────────────────────────────────

    #[test]
    fn test_mahalanobis_detects_outlier() {
        let mut det = build_detector(AnomalyMethod::MahalanobisApprox, 20, 4);
        let result = det.detect(1, &[50.0_f32, 50.0, 50.0, 50.0]).expect("Some");
        assert!(result.is_anomaly, "score={}", result.score);
    }

    #[test]
    fn test_mahalanobis_score_formula() {
        // With all refs = 0 and tiny std, a vector far from mean gives large score
        let mut det = build_detector(AnomalyMethod::MahalanobisApprox, 10, 2);
        let result = det.detect(1, &[30.0_f32, 40.0]).expect("Some");
        assert!(result.score > 3.0, "score={}", result.score);
    }

    #[test]
    fn test_mahalanobis_method_field_in_result() {
        let mut det = build_detector(AnomalyMethod::MahalanobisApprox, 10, 2);
        let result = det.detect(3, &[0.0, 0.0]).expect("Some");
        assert_eq!(result.method, AnomalyMethod::MahalanobisApprox);
    }

    // ── IsolationScore method ────────────────────────────────────────────────

    #[test]
    fn test_isolation_detects_outlier() {
        // All dims far from the mean → score close to 1.0 → exceeds 0.7
        let mut det = build_detector(AnomalyMethod::IsolationScore, 20, 10);
        let far: Vec<f32> = vec![50.0_f32; 10];
        let result = det.detect(1, &far).expect("Some");
        assert!(result.is_anomaly, "score={}", result.score);
        assert!(result.score > 0.7, "score={}", result.score);
    }

    #[test]
    fn test_isolation_no_anomaly_for_normal_vector() {
        // Build refs uniformly, query with the mean → score near 0
        let refs: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![i as f32 * 0.01, i as f32 * 0.01])
            .collect();
        let mut det = build_detector_with_refs(AnomalyMethod::IsolationScore, refs);
        // A vector very close to mean with |z|<1 on all dims
        let result = det.detect(99, &[0.095_f32, 0.095]).expect("Some");
        assert!(!result.is_anomaly, "score={}", result.score);
    }

    #[test]
    fn test_isolation_method_field_in_result() {
        let mut det = build_detector(AnomalyMethod::IsolationScore, 10, 3);
        let result = det.detect(5, &[0.0, 0.0, 0.0]).expect("Some");
        assert_eq!(result.method, AnomalyMethod::IsolationScore);
    }

    // ── flagged_dims ─────────────────────────────────────────────────────────

    #[test]
    fn test_flagged_dims_at_most_5() {
        let mut det = build_detector(AnomalyMethod::ZScore, 10, 20);
        let result = det.detect(1, &[100.0_f32; 20]).expect("Some");
        assert!(result.flagged_dims.len() <= 5);
    }

    #[test]
    fn test_flagged_dims_contains_highest_z_dim() {
        // One dimension is far out, rest are zero
        let mut det = build_detector(AnomalyMethod::ZScore, 10, 5);
        let mut query = vec![0.0_f32; 5];
        query[2] = 1000.0; // dim 2 has the extreme z-score
        let result = det.detect(1, &query).expect("Some");
        assert!(
            result.flagged_dims.contains(&2),
            "Expected dim 2 in flagged_dims: {:?}",
            result.flagged_dims
        );
    }

    #[test]
    fn test_flagged_dims_ordering() {
        // Three differing extremes; top-5 should be ordered by descending z-score
        let mut det = build_detector(AnomalyMethod::ZScore, 10, 6);
        let mut query = vec![0.0_f32; 6];
        query[0] = 300.0;
        query[3] = 200.0;
        query[5] = 100.0;
        let result = det.detect(1, &query).expect("Some");
        // dim 0 should appear before dim 3, dim 3 before dim 5 in flagged list
        let pos = |d: usize| result.flagged_dims.iter().position(|&x| x == d);
        assert!(pos(0) < pos(3), "flagged_dims={:?}", result.flagged_dims);
        assert!(pos(3) < pos(5), "flagged_dims={:?}", result.flagged_dims);
    }

    // ── stats ────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_total_checked_increments() {
        let mut det = build_detector(AnomalyMethod::ZScore, 10, 2);
        det.detect(1, &[0.0, 0.0]);
        det.detect(2, &[0.0, 0.0]);
        assert_eq!(det.stats().total_checked, 2);
    }

    #[test]
    fn test_stats_total_anomalies() {
        let mut det = build_detector(AnomalyMethod::ZScore, 10, 2);
        det.detect(1, &[0.0, 0.0]); // normal
        det.detect(2, &[1000.0, 1000.0]); // anomaly
        assert_eq!(det.stats().total_anomalies, 1);
    }

    #[test]
    fn test_anomaly_rate_zero_when_no_checks() {
        let config = DetectorConfig::with_method(AnomalyMethod::ZScore);
        let det = VectorAnomalyDetector::new(config);
        assert_eq!(det.stats().anomaly_rate(), 0.0);
    }

    #[test]
    fn test_anomaly_rate_correct() {
        let mut det = build_detector(AnomalyMethod::ZScore, 10, 1);
        // All refs are 0.0; outlier at 1000 will be flagged
        for _ in 0..4 {
            det.detect(0, &[0.0]);
        }
        det.detect(99, &[1000.0]);
        // 1 anomaly out of 5 checks → rate = 0.2
        let rate = det.stats().anomaly_rate();
        assert!((rate - 0.2).abs() < 1e-9, "rate={rate}");
    }

    // ── compute_mean_std ─────────────────────────────────────────────────────

    #[test]
    fn test_compute_mean_std_correct_mean() {
        let refs = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 4.0]];
        let config = DetectorConfig::with_method(AnomalyMethod::ZScore);
        let mut det = VectorAnomalyDetector::new(config);
        for r in refs {
            det.add_reference(r);
        }
        let (means, _stds) = det.compute_mean_std();
        assert!((means[0] - 2.0).abs() < 1e-5, "mean[0]={}", means[0]);
        assert!((means[1] - 3.0).abs() < 1e-5, "mean[1]={}", means[1]);
    }

    #[test]
    fn test_compute_mean_std_clamped_std() {
        // All reference vectors identical → std should be clamped to 1e-6
        let refs = vec![vec![5.0_f32]; 10];
        let det = build_detector_with_refs(AnomalyMethod::ZScore, refs);
        let (_means, stds) = det.compute_mean_std();
        assert!(
            stds[0] >= 1e-6_f32,
            "std should be clamped to at least 1e-6, got {}",
            stds[0]
        );
    }

    // ── FNV-1a helper ────────────────────────────────────────────────────────

    #[test]
    fn test_fnv1a_rand_f64_in_range() {
        for i in 0..100u64 {
            let v = fnv1a_rand_f64(42, i);
            assert!((0.0..1.0).contains(&v), "v={v}");
        }
    }

    #[test]
    fn test_fnv1a_rand_f64_different_seeds() {
        let a = fnv1a_rand_f64(1, 0);
        let b = fnv1a_rand_f64(2, 0);
        assert_ne!(a, b);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// SemanticAnomalyDetector — distance-from-centroid and IQR-based detection
// ═════════════════════════════════════════════════════════════════════════════

/// Method used by [`SemanticAnomalyDetector`] for anomaly detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticAnomalyMethod {
    /// Z-score of distance from centroid exceeds threshold.
    ZScore,
    /// Distance outside Q3 + multiplier * IQR.
    IQR,
    /// Distance from centroid exceeds mean + multiplier * std_dev.
    DistanceBased,
}

/// Configuration for [`SemanticAnomalyDetector`].
#[derive(Debug, Clone)]
pub struct AnomalyConfig {
    /// Detection method.
    pub method: SemanticAnomalyMethod,
    /// Threshold for z-score method (default 3.0).
    pub z_threshold: f64,
    /// Multiplier for IQR method (default 1.5).
    pub iqr_multiplier: f64,
    /// Multiplier for distance-based method (default 2.0).
    pub distance_multiplier: f64,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            method: SemanticAnomalyMethod::ZScore,
            z_threshold: 3.0,
            iqr_multiplier: 1.5,
            distance_multiplier: 2.0,
        }
    }
}

/// Result of anomaly detection for a single document embedding.
#[derive(Debug, Clone)]
pub struct SemanticAnomalyResult {
    /// Document identifier.
    pub doc_id: String,
    /// Anomaly score (higher = more anomalous).
    pub score: f64,
    /// Whether the embedding is considered anomalous.
    pub is_anomaly: bool,
    /// Detection method used.
    pub method: SemanticAnomalyMethod,
}

/// Statistics for [`SemanticAnomalyDetector`].
#[derive(Debug, Clone)]
pub struct AnomalyDetectorStats {
    /// Number of embeddings currently stored.
    pub embedding_count: usize,
    /// Number of detection runs performed.
    pub detections_run: u64,
    /// Detection method configured.
    pub method: SemanticAnomalyMethod,
}

/// Detects anomalous embeddings using distance-from-centroid analysis.
///
/// Supports three detection methods:
/// - **ZScore**: z-score of distances to centroid
/// - **IQR**: interquartile range on distances
/// - **DistanceBased**: mean + multiplier * std_dev threshold on distances
///
/// # Example
/// ```
/// use ipfrs_semantic::anomaly_detector::{
///     SemanticAnomalyDetector, AnomalyConfig, SemanticAnomalyMethod,
/// };
///
/// let config = AnomalyConfig::default();
/// let mut detector = SemanticAnomalyDetector::new(config);
///
/// // Add some normal embeddings
/// for i in 0..20 {
///     detector.add_embedding(&format!("doc_{i}"), vec![0.1 * i as f64, 0.0, 0.0]);
/// }
///
/// // Detect anomalies
/// let results = detector.detect_all();
/// ```
pub struct SemanticAnomalyDetector {
    config: AnomalyConfig,
    embeddings: Vec<(String, Vec<f64>)>,
    centroid: Vec<f64>,
    detections_run: u64,
}

impl SemanticAnomalyDetector {
    /// Create a new detector with the given configuration.
    pub fn new(config: AnomalyConfig) -> Self {
        Self {
            config,
            embeddings: Vec::new(),
            centroid: Vec::new(),
            detections_run: 0,
        }
    }

    /// Add an embedding and incrementally recompute the centroid.
    pub fn add_embedding(&mut self, doc_id: &str, embedding: Vec<f64>) {
        let n = self.embeddings.len();
        if n == 0 {
            self.centroid = embedding.clone();
        } else {
            // Incremental centroid: new_centroid = old_centroid * n/(n+1) + new_point / (n+1)
            let new_n = (n + 1) as f64;
            if self.centroid.len() == embedding.len() {
                for (c, &e) in self.centroid.iter_mut().zip(embedding.iter()) {
                    *c = *c * (n as f64 / new_n) + e / new_n;
                }
            } else {
                // Dimension mismatch: resize centroid
                self.centroid = Self::compute_centroid_from_iter(
                    self.embeddings
                        .iter()
                        .map(|(_, v)| v.as_slice())
                        .chain(std::iter::once(embedding.as_slice())),
                    embedding.len(),
                );
            }
        }
        self.embeddings.push((doc_id.to_string(), embedding));
    }

    /// Remove an embedding by doc_id. Returns `true` if found and removed.
    ///
    /// Recomputes the centroid from scratch after removal.
    pub fn remove_embedding(&mut self, doc_id: &str) -> bool {
        let before = self.embeddings.len();
        self.embeddings.retain(|(id, _)| id != doc_id);
        let removed = self.embeddings.len() < before;
        if removed {
            if self.embeddings.is_empty() {
                self.centroid.clear();
            } else {
                let dims = self.embeddings[0].1.len();
                self.centroid = Self::compute_centroid_from_iter(
                    self.embeddings.iter().map(|(_, v)| v.as_slice()),
                    dims,
                );
            }
        }
        removed
    }

    /// Run detection on all stored embeddings using the configured method.
    pub fn detect_all(&mut self) -> Vec<SemanticAnomalyResult> {
        self.detections_run += 1;

        if self.embeddings.len() < 2 {
            // With 0 or 1 embeddings, nothing can be anomalous
            return self
                .embeddings
                .iter()
                .map(|(id, _)| SemanticAnomalyResult {
                    doc_id: id.clone(),
                    score: 0.0,
                    is_anomaly: false,
                    method: self.config.method,
                })
                .collect();
        }

        let distances = self.distances_to_centroid();
        let dist_values: Vec<f64> = distances.iter().map(|(_, d)| *d).collect();

        match self.config.method {
            SemanticAnomalyMethod::ZScore => self.detect_zscore(&distances, &dist_values),
            SemanticAnomalyMethod::IQR => self.detect_iqr(&distances, &dist_values),
            SemanticAnomalyMethod::DistanceBased => {
                self.detect_distance_based(&distances, &dist_values)
            }
        }
    }

    /// Check if a single new embedding is anomalous against existing data.
    ///
    /// Does not add the embedding to the detector.
    pub fn detect_single(&self, embedding: &[f64]) -> SemanticAnomalyResult {
        if self.embeddings.len() < 2 || self.centroid.is_empty() {
            return SemanticAnomalyResult {
                doc_id: String::new(),
                score: 0.0,
                is_anomaly: false,
                method: self.config.method,
            };
        }

        let dist = Self::euclidean_distance(embedding, &self.centroid);
        let existing_dists: Vec<f64> = self
            .embeddings
            .iter()
            .map(|(_, v)| Self::euclidean_distance(v, &self.centroid))
            .collect();

        let (score, is_anomaly) = match self.config.method {
            SemanticAnomalyMethod::ZScore => {
                let (mean, std) = Self::mean_std(&existing_dists);
                let z = if std < 1e-12 {
                    0.0
                } else {
                    (dist - mean) / std
                };
                (z.abs(), z.abs() > self.config.z_threshold)
            }
            SemanticAnomalyMethod::IQR => {
                let (_, q3, iqr) = Self::quartiles(&existing_dists);
                let upper = q3 + self.config.iqr_multiplier * iqr;
                (dist, dist > upper)
            }
            SemanticAnomalyMethod::DistanceBased => {
                let (mean, std) = Self::mean_std(&existing_dists);
                let threshold = mean + self.config.distance_multiplier * std;
                (dist, dist > threshold)
            }
        };

        SemanticAnomalyResult {
            doc_id: String::new(),
            score,
            is_anomaly,
            method: self.config.method,
        }
    }

    /// Compute the centroid (mean vector) of the given embeddings.
    pub fn compute_centroid(embeddings: &[(String, Vec<f64>)]) -> Vec<f64> {
        if embeddings.is_empty() {
            return Vec::new();
        }
        let dims = embeddings[0].1.len();
        Self::compute_centroid_from_iter(embeddings.iter().map(|(_, v)| v.as_slice()), dims)
    }

    /// Compute centroid from an iterator of embedding slices.
    fn compute_centroid_from_iter<'a>(
        iter: impl Iterator<Item = &'a [f64]>,
        dims: usize,
    ) -> Vec<f64> {
        let mut sum = vec![0.0_f64; dims];
        let mut count = 0usize;
        for v in iter {
            for (s, &val) in sum.iter_mut().zip(v.iter()) {
                *s += val;
            }
            count += 1;
        }
        if count == 0 {
            return sum;
        }
        let n = count as f64;
        for s in &mut sum {
            *s /= n;
        }
        sum
    }

    /// Euclidean distance between two vectors.
    pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = x - y;
                d * d
            })
            .sum::<f64>()
            .sqrt()
    }

    /// Compute (doc_id, distance) pairs for all stored embeddings to centroid.
    pub fn distances_to_centroid(&self) -> Vec<(String, f64)> {
        self.embeddings
            .iter()
            .map(|(id, v)| {
                let d = Self::euclidean_distance(v, &self.centroid);
                (id.clone(), d)
            })
            .collect()
    }

    /// Number of stored embeddings.
    pub fn embedding_count(&self) -> usize {
        self.embeddings.len()
    }

    /// Get detector statistics.
    pub fn stats(&self) -> AnomalyDetectorStats {
        AnomalyDetectorStats {
            embedding_count: self.embeddings.len(),
            detections_run: self.detections_run,
            method: self.config.method,
        }
    }

    // ── private helpers ─────────────────────────────────────────────────────

    fn detect_zscore(
        &self,
        distances: &[(String, f64)],
        dist_values: &[f64],
    ) -> Vec<SemanticAnomalyResult> {
        let (mean, std) = Self::mean_std(dist_values);
        distances
            .iter()
            .map(|(id, d)| {
                let z = if std < 1e-12 { 0.0 } else { (*d - mean) / std };
                SemanticAnomalyResult {
                    doc_id: id.clone(),
                    score: z.abs(),
                    is_anomaly: z.abs() > self.config.z_threshold,
                    method: SemanticAnomalyMethod::ZScore,
                }
            })
            .collect()
    }

    fn detect_iqr(
        &self,
        distances: &[(String, f64)],
        dist_values: &[f64],
    ) -> Vec<SemanticAnomalyResult> {
        let (_q1, q3, iqr) = Self::quartiles(dist_values);
        let upper = q3 + self.config.iqr_multiplier * iqr;
        distances
            .iter()
            .map(|(id, d)| SemanticAnomalyResult {
                doc_id: id.clone(),
                score: *d,
                is_anomaly: *d > upper,
                method: SemanticAnomalyMethod::IQR,
            })
            .collect()
    }

    fn detect_distance_based(
        &self,
        distances: &[(String, f64)],
        dist_values: &[f64],
    ) -> Vec<SemanticAnomalyResult> {
        let (mean, std) = Self::mean_std(dist_values);
        let threshold = mean + self.config.distance_multiplier * std;
        distances
            .iter()
            .map(|(id, d)| SemanticAnomalyResult {
                doc_id: id.clone(),
                score: *d,
                is_anomaly: *d > threshold,
                method: SemanticAnomalyMethod::DistanceBased,
            })
            .collect()
    }

    fn mean_std(values: &[f64]) -> (f64, f64) {
        if values.is_empty() {
            return (0.0, 0.0);
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        (mean, variance.sqrt())
    }

    /// Compute Q1, Q3, and IQR from a slice of values.
    fn quartiles(values: &[f64]) -> (f64, f64, f64) {
        if values.len() < 2 {
            let v = values.first().copied().unwrap_or(0.0);
            return (v, v, 0.0);
        }
        let mut sorted: Vec<f64> = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1 = Self::percentile_sorted(&sorted, 25.0);
        let q3 = Self::percentile_sorted(&sorted, 75.0);
        (q1, q3, q3 - q1)
    }

    /// Linear interpolation percentile on a pre-sorted slice.
    fn percentile_sorted(sorted: &[f64], pct: f64) -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        if sorted.len() == 1 {
            return sorted[0];
        }
        let rank = (pct / 100.0) * (sorted.len() - 1) as f64;
        let lo = rank.floor() as usize;
        let hi = rank.ceil() as usize;
        let frac = rank - lo as f64;
        if lo == hi {
            sorted[lo]
        } else {
            sorted[lo] * (1.0 - frac) + sorted[hi] * frac
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// SemanticAnomalyDetector Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod semantic_anomaly_tests {
    use super::*;

    fn make_config(method: SemanticAnomalyMethod) -> AnomalyConfig {
        AnomalyConfig {
            method,
            ..AnomalyConfig::default()
        }
    }

    fn cluster_with_outlier() -> SemanticAnomalyDetector {
        let mut det = SemanticAnomalyDetector::new(make_config(SemanticAnomalyMethod::ZScore));
        // Add a tight cluster around origin
        for i in 0..20 {
            det.add_embedding(
                &format!("normal_{i}"),
                vec![0.01 * i as f64, -0.01 * i as f64, 0.0],
            );
        }
        // Add an obvious outlier
        det.add_embedding("outlier", vec![100.0, 100.0, 100.0]);
        det
    }

    // ── basic construction ──────────────────────────────────────────────────

    #[test]
    fn test_new_creates_empty_detector() {
        let det = SemanticAnomalyDetector::new(AnomalyConfig::default());
        assert_eq!(det.embedding_count(), 0);
        assert!(det.centroid.is_empty());
    }

    #[test]
    fn test_default_config_values() {
        let cfg = AnomalyConfig::default();
        assert_eq!(cfg.method, SemanticAnomalyMethod::ZScore);
        assert!((cfg.z_threshold - 3.0).abs() < f64::EPSILON);
        assert!((cfg.iqr_multiplier - 1.5).abs() < f64::EPSILON);
        assert!((cfg.distance_multiplier - 2.0).abs() < f64::EPSILON);
    }

    // ── add / remove embedding ──────────────────────────────────────────────

    #[test]
    fn test_add_embedding_increments_count() {
        let mut det = SemanticAnomalyDetector::new(AnomalyConfig::default());
        det.add_embedding("a", vec![1.0, 2.0]);
        det.add_embedding("b", vec![3.0, 4.0]);
        assert_eq!(det.embedding_count(), 2);
    }

    #[test]
    fn test_add_embedding_updates_centroid() {
        let mut det = SemanticAnomalyDetector::new(AnomalyConfig::default());
        det.add_embedding("a", vec![0.0, 0.0]);
        assert!((det.centroid[0]).abs() < 1e-12);
        det.add_embedding("b", vec![2.0, 4.0]);
        // centroid should be [1.0, 2.0]
        assert!((det.centroid[0] - 1.0).abs() < 1e-9);
        assert!((det.centroid[1] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_remove_embedding_returns_true_if_found() {
        let mut det = SemanticAnomalyDetector::new(AnomalyConfig::default());
        det.add_embedding("x", vec![1.0]);
        assert!(det.remove_embedding("x"));
        assert_eq!(det.embedding_count(), 0);
    }

    #[test]
    fn test_remove_embedding_returns_false_if_not_found() {
        let mut det = SemanticAnomalyDetector::new(AnomalyConfig::default());
        det.add_embedding("x", vec![1.0]);
        assert!(!det.remove_embedding("y"));
        assert_eq!(det.embedding_count(), 1);
    }

    #[test]
    fn test_remove_embedding_updates_centroid() {
        let mut det = SemanticAnomalyDetector::new(AnomalyConfig::default());
        det.add_embedding("a", vec![0.0, 0.0]);
        det.add_embedding("b", vec![4.0, 6.0]);
        det.add_embedding("c", vec![2.0, 3.0]);
        // centroid = [2.0, 3.0]
        det.remove_embedding("c");
        // centroid should be [2.0, 3.0] still (mean of [0,0] and [4,6])
        assert!((det.centroid[0] - 2.0).abs() < 1e-9);
        assert!((det.centroid[1] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_remove_last_embedding_clears_centroid() {
        let mut det = SemanticAnomalyDetector::new(AnomalyConfig::default());
        det.add_embedding("only", vec![5.0, 5.0]);
        det.remove_embedding("only");
        assert!(det.centroid.is_empty());
    }

    // ── euclidean_distance ──────────────────────────────────────────────────

    #[test]
    fn test_euclidean_distance_basic() {
        let d = SemanticAnomalyDetector::euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_euclidean_distance_same_point() {
        let d = SemanticAnomalyDetector::euclidean_distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert!(d.abs() < 1e-12);
    }

    // ── compute_centroid ────────────────────────────────────────────────────

    #[test]
    fn test_compute_centroid_empty() {
        let c = SemanticAnomalyDetector::compute_centroid(&[]);
        assert!(c.is_empty());
    }

    #[test]
    fn test_compute_centroid_single() {
        let embs = vec![("a".to_string(), vec![3.0, 6.0])];
        let c = SemanticAnomalyDetector::compute_centroid(&embs);
        assert!((c[0] - 3.0).abs() < 1e-9);
        assert!((c[1] - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_centroid_multiple() {
        let embs = vec![
            ("a".to_string(), vec![0.0, 0.0]),
            ("b".to_string(), vec![4.0, 8.0]),
        ];
        let c = SemanticAnomalyDetector::compute_centroid(&embs);
        assert!((c[0] - 2.0).abs() < 1e-9);
        assert!((c[1] - 4.0).abs() < 1e-9);
    }

    // ── distances_to_centroid ───────────────────────────────────────────────

    #[test]
    fn test_distances_to_centroid() {
        let mut det = SemanticAnomalyDetector::new(AnomalyConfig::default());
        det.add_embedding("a", vec![0.0, 0.0]);
        det.add_embedding("b", vec![6.0, 8.0]);
        // centroid = [3.0, 4.0]
        let dists = det.distances_to_centroid();
        assert_eq!(dists.len(), 2);
        // distance from [0,0] to [3,4] = 5.0
        assert!((dists[0].1 - 5.0).abs() < 1e-9);
        // distance from [6,8] to [3,4] = 5.0
        assert!((dists[1].1 - 5.0).abs() < 1e-9);
    }

    // ── detect_all: ZScore ──────────────────────────────────────────────────

    #[test]
    fn test_zscore_detects_obvious_outlier() {
        let mut det = cluster_with_outlier();
        let results = det.detect_all();
        let outlier = results
            .iter()
            .find(|r| r.doc_id == "outlier")
            .expect("outlier should be in results");
        assert!(outlier.is_anomaly, "outlier should be flagged");
        assert!(outlier.score > 3.0, "score={}", outlier.score);
    }

    #[test]
    fn test_zscore_normal_not_flagged() {
        let mut det = cluster_with_outlier();
        let results = det.detect_all();
        let normals: Vec<_> = results
            .iter()
            .filter(|r| r.doc_id.starts_with("normal_"))
            .collect();
        let flagged_count = normals.iter().filter(|r| r.is_anomaly).count();
        // At most 1-2 edge cases might be flagged, but definitely not all
        assert!(
            flagged_count <= 2,
            "Too many normals flagged: {flagged_count}/{}",
            normals.len()
        );
    }

    #[test]
    fn test_zscore_method_in_result() {
        let mut det = cluster_with_outlier();
        let results = det.detect_all();
        for r in &results {
            assert_eq!(r.method, SemanticAnomalyMethod::ZScore);
        }
    }

    // ── detect_all: IQR ─────────────────────────────────────────────────────

    #[test]
    fn test_iqr_detects_obvious_outlier() {
        let mut det = SemanticAnomalyDetector::new(make_config(SemanticAnomalyMethod::IQR));
        for i in 0..20 {
            det.add_embedding(
                &format!("normal_{i}"),
                vec![0.01 * i as f64, -0.01 * i as f64, 0.0],
            );
        }
        det.add_embedding("outlier", vec![100.0, 100.0, 100.0]);
        let results = det.detect_all();
        let outlier = results
            .iter()
            .find(|r| r.doc_id == "outlier")
            .expect("outlier in results");
        assert!(outlier.is_anomaly, "outlier should be flagged by IQR");
    }

    #[test]
    fn test_iqr_normal_not_flagged() {
        let mut det = SemanticAnomalyDetector::new(make_config(SemanticAnomalyMethod::IQR));
        for i in 0..20 {
            det.add_embedding(
                &format!("normal_{i}"),
                vec![0.01 * i as f64, -0.01 * i as f64],
            );
        }
        let results = det.detect_all();
        let flagged = results.iter().filter(|r| r.is_anomaly).count();
        // With no outlier, very few should be flagged
        assert!(
            flagged <= 3,
            "Too many flagged: {flagged}/{}",
            results.len()
        );
    }

    #[test]
    fn test_iqr_method_in_result() {
        let mut det = SemanticAnomalyDetector::new(make_config(SemanticAnomalyMethod::IQR));
        det.add_embedding("a", vec![1.0]);
        det.add_embedding("b", vec![2.0]);
        det.add_embedding("c", vec![3.0]);
        let results = det.detect_all();
        for r in &results {
            assert_eq!(r.method, SemanticAnomalyMethod::IQR);
        }
    }

    // ── detect_all: DistanceBased ───────────────────────────────────────────

    #[test]
    fn test_distance_based_detects_outlier() {
        let mut det =
            SemanticAnomalyDetector::new(make_config(SemanticAnomalyMethod::DistanceBased));
        for i in 0..20 {
            det.add_embedding(&format!("normal_{i}"), vec![0.01 * i as f64, 0.0]);
        }
        det.add_embedding("outlier", vec![100.0, 100.0]);
        let results = det.detect_all();
        let outlier = results
            .iter()
            .find(|r| r.doc_id == "outlier")
            .expect("outlier in results");
        assert!(
            outlier.is_anomaly,
            "outlier should be flagged by DistanceBased"
        );
    }

    #[test]
    fn test_distance_based_normal_not_flagged() {
        let mut det =
            SemanticAnomalyDetector::new(make_config(SemanticAnomalyMethod::DistanceBased));
        for i in 0..20 {
            det.add_embedding(&format!("normal_{i}"), vec![0.01 * i as f64, 0.0]);
        }
        let results = det.detect_all();
        let flagged = results.iter().filter(|r| r.is_anomaly).count();
        assert!(
            flagged <= 3,
            "Too many flagged: {flagged}/{}",
            results.len()
        );
    }

    #[test]
    fn test_distance_based_method_in_result() {
        let mut det =
            SemanticAnomalyDetector::new(make_config(SemanticAnomalyMethod::DistanceBased));
        det.add_embedding("a", vec![1.0]);
        det.add_embedding("b", vec![2.0]);
        let results = det.detect_all();
        for r in &results {
            assert_eq!(r.method, SemanticAnomalyMethod::DistanceBased);
        }
    }

    // ── detect_single ───────────────────────────────────────────────────────

    #[test]
    fn test_detect_single_flags_outlier() {
        let mut det = SemanticAnomalyDetector::new(AnomalyConfig::default());
        for i in 0..20 {
            det.add_embedding(
                &format!("n{i}"),
                vec![0.01 * i as f64, -0.01 * i as f64, 0.005 * i as f64],
            );
        }
        let result = det.detect_single(&[100.0, 100.0, 100.0]);
        assert!(result.is_anomaly, "single outlier should be flagged");
    }

    #[test]
    fn test_detect_single_normal_not_flagged() {
        let mut det = SemanticAnomalyDetector::new(AnomalyConfig::default());
        for i in 0..20 {
            det.add_embedding(&format!("n{i}"), vec![0.01 * i as f64, -0.01 * i as f64]);
        }
        let result = det.detect_single(&[0.1, -0.1]);
        assert!(!result.is_anomaly, "normal point should not be flagged");
    }

    #[test]
    fn test_detect_single_empty_detector() {
        let det = SemanticAnomalyDetector::new(AnomalyConfig::default());
        let result = det.detect_single(&[1.0, 2.0]);
        assert!(
            !result.is_anomaly,
            "empty detector should not flag anything"
        );
        assert!((result.score).abs() < 1e-12);
    }

    // ── empty / single embedding edge cases ─────────────────────────────────

    #[test]
    fn test_detect_all_empty_returns_empty() {
        let mut det = SemanticAnomalyDetector::new(AnomalyConfig::default());
        let results = det.detect_all();
        assert!(results.is_empty());
    }

    #[test]
    fn test_detect_all_single_embedding_no_anomaly() {
        let mut det = SemanticAnomalyDetector::new(AnomalyConfig::default());
        det.add_embedding("solo", vec![42.0, 42.0]);
        let results = det.detect_all();
        assert_eq!(results.len(), 1);
        assert!(
            !results[0].is_anomaly,
            "single embedding cannot be anomalous"
        );
    }

    // ── stats ───────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial() {
        let det = SemanticAnomalyDetector::new(AnomalyConfig::default());
        let s = det.stats();
        assert_eq!(s.embedding_count, 0);
        assert_eq!(s.detections_run, 0);
        assert_eq!(s.method, SemanticAnomalyMethod::ZScore);
    }

    #[test]
    fn test_stats_after_operations() {
        let mut det = SemanticAnomalyDetector::new(make_config(SemanticAnomalyMethod::IQR));
        det.add_embedding("a", vec![1.0]);
        det.add_embedding("b", vec![2.0]);
        det.detect_all();
        det.detect_all();
        let s = det.stats();
        assert_eq!(s.embedding_count, 2);
        assert_eq!(s.detections_run, 2);
        assert_eq!(s.method, SemanticAnomalyMethod::IQR);
    }

    // ── score ordering ──────────────────────────────────────────────────────

    #[test]
    fn test_score_ordering_outlier_highest() {
        let mut det = cluster_with_outlier();
        let results = det.detect_all();
        let outlier_score = results
            .iter()
            .find(|r| r.doc_id == "outlier")
            .map(|r| r.score)
            .expect("outlier in results");
        let max_normal_score = results
            .iter()
            .filter(|r| r.doc_id != "outlier")
            .map(|r| r.score)
            .fold(0.0_f64, f64::max);
        assert!(
            outlier_score > max_normal_score,
            "outlier score ({outlier_score}) should exceed max normal ({max_normal_score})"
        );
    }

    #[test]
    fn test_score_ordering_closer_to_centroid_lower_score() {
        let mut det =
            SemanticAnomalyDetector::new(make_config(SemanticAnomalyMethod::DistanceBased));
        for i in 0..20 {
            det.add_embedding(&format!("n{i}"), vec![0.0, 0.0]);
        }
        det.add_embedding("far", vec![10.0, 10.0]);
        det.add_embedding("farther", vec![50.0, 50.0]);
        let results = det.detect_all();
        let far_score = results
            .iter()
            .find(|r| r.doc_id == "far")
            .map(|r| r.score)
            .expect("far");
        let farther_score = results
            .iter()
            .find(|r| r.doc_id == "farther")
            .map(|r| r.score)
            .expect("farther");
        assert!(
            farther_score > far_score,
            "farther ({farther_score}) should score higher than far ({far_score})"
        );
    }

    // ── detect_single with each method ──────────────────────────────────────

    #[test]
    fn test_detect_single_iqr() {
        let mut det = SemanticAnomalyDetector::new(make_config(SemanticAnomalyMethod::IQR));
        for i in 0..20 {
            det.add_embedding(&format!("n{i}"), vec![0.0, 0.0]);
        }
        let result = det.detect_single(&[100.0, 100.0]);
        assert!(result.is_anomaly);
        assert_eq!(result.method, SemanticAnomalyMethod::IQR);
    }

    #[test]
    fn test_detect_single_distance_based() {
        let mut det =
            SemanticAnomalyDetector::new(make_config(SemanticAnomalyMethod::DistanceBased));
        for i in 0..20 {
            det.add_embedding(&format!("n{i}"), vec![0.0, 0.0]);
        }
        let result = det.detect_single(&[100.0, 100.0]);
        assert!(result.is_anomaly);
        assert_eq!(result.method, SemanticAnomalyMethod::DistanceBased);
    }

    // ── custom config thresholds ────────────────────────────────────────────

    #[test]
    fn test_custom_z_threshold() {
        let config = AnomalyConfig {
            method: SemanticAnomalyMethod::ZScore,
            z_threshold: 100.0, // Very high threshold
            ..AnomalyConfig::default()
        };
        let mut det = SemanticAnomalyDetector::new(config);
        for i in 0..20 {
            det.add_embedding(&format!("n{i}"), vec![0.0]);
        }
        det.add_embedding("outlier", vec![10.0]);
        let results = det.detect_all();
        // With threshold=100.0, even moderate outliers should not be flagged
        let flagged = results.iter().filter(|r| r.is_anomaly).count();
        assert_eq!(flagged, 0, "high z_threshold should prevent flagging");
    }

    #[test]
    fn test_custom_iqr_multiplier() {
        let config = AnomalyConfig {
            method: SemanticAnomalyMethod::IQR,
            iqr_multiplier: 0.01, // Very tight
            ..AnomalyConfig::default()
        };
        let mut det = SemanticAnomalyDetector::new(config);
        for i in 0..20 {
            det.add_embedding(&format!("n{i}"), vec![i as f64, 0.0]);
        }
        let results = det.detect_all();
        // With very tight multiplier, more should be flagged
        let flagged = results.iter().filter(|r| r.is_anomaly).count();
        assert!(flagged > 0, "tight iqr_multiplier should flag some");
    }
}
