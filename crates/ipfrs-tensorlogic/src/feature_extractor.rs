//! TensorFeatureExtractor — statistical and structural feature extraction from tensor data.
//!
//! Extracts mean, variance, skewness, min/max, range, L-norms, and histogram features
//! from raw `f64` tensor slices for use in downstream ML pipelines.

/// Which feature to extract from a tensor slice.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FeatureKind {
    /// Arithmetic mean: Σxᵢ / n  (0.0 when empty)
    Mean,
    /// Population variance: E\[x²\] − E\[x\]²  (0.0 when empty)
    Variance,
    /// Standardised third central moment: E[(x−μ)³] / σ³  (0.0 when σ=0 or empty)
    Skewness,
    /// Minimum element  (0.0 when empty)
    Min,
    /// Maximum element  (0.0 when empty)
    Max,
    /// Range: max − min  (0.0 when empty)
    Range,
    /// L¹ norm: Σ|xᵢ|
    L1Norm,
    /// L² norm: √(Σxᵢ²)
    L2Norm,
    /// Histogram: divide [min, max] into `bins` equal-width buckets and count elements.
    /// Produces `bins` values.  When min == max every element falls in bucket 0.
    Histogram {
        /// Number of equal-width histogram bins.
        bins: usize,
    },
}

// ─────────────────────────────────────────────────────────────────────────────

/// A single extracted feature together with its (possibly multi-valued) result.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtractedFeature {
    /// Which feature was extracted.
    pub kind: FeatureKind,
    /// Feature values — length 1 for scalar features, `bins` for Histogram.
    pub values: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`TensorFeatureExtractor`].
#[derive(Clone, Debug)]
pub struct ExtractorConfig {
    /// Ordered list of features to extract.
    pub features: Vec<FeatureKind>,
    /// Default bin count used when a `Histogram { bins }` entry is added via
    /// [`ExtractorConfig::default`].
    pub histogram_bins: usize,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            features: vec![
                FeatureKind::Mean,
                FeatureKind::Variance,
                FeatureKind::Min,
                FeatureKind::Max,
                FeatureKind::L2Norm,
            ],
            histogram_bins: 10,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// The result of extracting features from a single tensor.
#[derive(Clone, Debug)]
pub struct ExtractionResult {
    /// Opaque identifier for the source tensor.
    pub tensor_id: u64,
    /// Extracted features in the order specified by [`ExtractorConfig::features`].
    pub features: Vec<ExtractedFeature>,
}

impl ExtractionResult {
    /// Flatten all feature values into a single `Vec<f64>` in declaration order.
    pub fn feature_vector(&self) -> Vec<f64> {
        self.features
            .iter()
            .flat_map(|f| f.values.iter().copied())
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Cumulative statistics for a [`TensorFeatureExtractor`] session.
#[derive(Clone, Debug, Default)]
pub struct ExtractorStats {
    /// Total number of individual feature extractions performed (features × tensors).
    pub total_extractions: u64,
    /// Total number of tensors processed.
    pub total_tensors_processed: u64,
    /// Running average of the feature-vector length across all processed tensors.
    pub avg_feature_vector_len: f64,
}

// ─────────────────────────────────────────────────────────────────────────────

/// Extracts statistical and structural features from tensor data.
pub struct TensorFeatureExtractor {
    /// Active configuration.
    pub config: ExtractorConfig,
    /// Accumulated statistics.
    pub stats: ExtractorStats,
}

impl TensorFeatureExtractor {
    /// Create a new extractor with the given configuration.
    pub fn new(config: ExtractorConfig) -> Self {
        Self {
            config,
            stats: ExtractorStats::default(),
        }
    }

    /// Extract all configured features from `data` and tag the result with `tensor_id`.
    ///
    /// Updates internal statistics on each call.
    pub fn extract(&mut self, tensor_id: u64, data: &[f64]) -> ExtractionResult {
        let n = data.len();

        // Pre-compute shared statistics once to avoid redundant passes.
        let precomputed = PrecomputedStats::compute(data);

        let features: Vec<ExtractedFeature> = self
            .config
            .features
            .iter()
            .map(|kind| {
                let values = extract_one(kind, data, n, &precomputed);
                ExtractedFeature {
                    kind: kind.clone(),
                    values,
                }
            })
            .collect();

        // ── update stats ──────────────────────────────────────────────────
        let fv_len: usize = features.iter().map(|f| f.values.len()).sum();
        let prev_total = self.stats.total_tensors_processed;
        let prev_avg = self.stats.avg_feature_vector_len;

        self.stats.total_tensors_processed += 1;
        self.stats.total_extractions += features.len() as u64;

        // Incremental mean update: avg_new = (avg_old * prev_total + fv_len) / new_total
        let new_total = prev_total + 1;
        self.stats.avg_feature_vector_len =
            (prev_avg * prev_total as f64 + fv_len as f64) / new_total as f64;

        ExtractionResult {
            tensor_id,
            features,
        }
    }

    /// Extract features from a batch of `(tensor_id, data)` pairs, returning
    /// results in the same order as the input.
    pub fn extract_batch(&mut self, tensors: Vec<(u64, Vec<f64>)>) -> Vec<ExtractionResult> {
        tensors
            .into_iter()
            .map(|(id, data)| self.extract(id, &data))
            .collect()
    }

    /// Return a reference to the accumulated statistics.
    pub fn stats(&self) -> &ExtractorStats {
        &self.stats
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics computed in a single pass over the data, shared across feature kinds.
struct PrecomputedStats {
    mean: f64,
    variance: f64,
    /// σ (std-dev); 0.0 when variance == 0 or n == 0.
    std_dev: f64,
    min: f64,
    max: f64,
}

impl PrecomputedStats {
    fn compute(data: &[f64]) -> Self {
        if data.is_empty() {
            return Self {
                mean: 0.0,
                variance: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
            };
        }

        let n = data.len() as f64;

        // Single-pass: sum, sum-of-squares, min, max.
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut min = data[0];
        let mut max = data[0];

        for &x in data {
            sum += x;
            sum_sq += x * x;
            if x < min {
                min = x;
            }
            if x > max {
                max = x;
            }
        }

        let mean = sum / n;
        // Population variance: E[x²] − E[x]²
        let variance = (sum_sq / n) - (mean * mean);
        // Guard against floating-point negatives near zero.
        let variance = if variance < 0.0 { 0.0 } else { variance };
        let std_dev = variance.sqrt();

        Self {
            mean,
            variance,
            std_dev,
            min,
            max,
        }
    }
}

/// Compute the values for a single `FeatureKind`.
fn extract_one(kind: &FeatureKind, data: &[f64], n: usize, pre: &PrecomputedStats) -> Vec<f64> {
    match kind {
        FeatureKind::Mean => vec![pre.mean],

        FeatureKind::Variance => vec![pre.variance],

        FeatureKind::Skewness => {
            if n == 0 || pre.std_dev == 0.0 {
                return vec![0.0];
            }
            // E[(x − μ)³] / σ³
            let mu = pre.mean;
            let sigma3 = pre.std_dev * pre.std_dev * pre.std_dev;
            let third_moment: f64 = data
                .iter()
                .map(|&x| {
                    let d = x - mu;
                    d * d * d
                })
                .sum::<f64>()
                / n as f64;
            vec![third_moment / sigma3]
        }

        FeatureKind::Min => vec![pre.min],

        FeatureKind::Max => vec![pre.max],

        FeatureKind::Range => vec![pre.max - pre.min],

        FeatureKind::L1Norm => {
            let l1: f64 = data.iter().map(|x| x.abs()).sum();
            vec![l1]
        }

        FeatureKind::L2Norm => {
            let l2: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();
            vec![l2]
        }

        FeatureKind::Histogram { bins } => {
            let bins = *bins;
            if bins == 0 {
                return Vec::new();
            }

            let mut counts = vec![0.0_f64; bins];

            if n == 0 {
                return counts;
            }

            let lo = pre.min;
            let hi = pre.max;

            if (hi - lo).abs() < f64::EPSILON {
                // All values identical — everything in bucket 0.
                counts[0] = n as f64;
            } else {
                let width = hi - lo;
                for &x in data {
                    // Normalise to [0, 1) then map to bucket index.
                    let t = (x - lo) / width;
                    // Clamp to [0, bins-1] — the last element hits exactly 1.0.
                    let idx = ((t * bins as f64) as usize).min(bins - 1);
                    counts[idx] += 1.0;
                }
            }

            counts
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn extractor_for(kinds: Vec<FeatureKind>) -> TensorFeatureExtractor {
        TensorFeatureExtractor::new(ExtractorConfig {
            features: kinds,
            histogram_bins: 10,
        })
    }

    fn single(kind: FeatureKind, data: &[f64]) -> f64 {
        let mut ex = extractor_for(vec![kind]);
        let res = ex.extract(0, data);
        res.features[0].values[0]
    }

    // ── Mean ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_mean_basic() {
        let v = single(FeatureKind::Mean, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((v - 3.0).abs() < 1e-12, "mean={v}");
    }

    #[test]
    fn test_mean_single_element() {
        let v = single(FeatureKind::Mean, &[7.5]);
        assert!((v - 7.5).abs() < 1e-12, "mean={v}");
    }

    #[test]
    fn test_mean_empty() {
        let v = single(FeatureKind::Mean, &[]);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn test_mean_negative() {
        let v = single(FeatureKind::Mean, &[-2.0, -4.0]);
        assert!((v - (-3.0)).abs() < 1e-12, "mean={v}");
    }

    // ── Variance ─────────────────────────────────────────────────────────────

    #[test]
    fn test_variance_basic() {
        // Population variance of [2, 4, 4, 4, 5, 5, 7, 9] == 4.0
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = single(FeatureKind::Variance, &data);
        assert!((v - 4.0).abs() < 1e-10, "var={v}");
    }

    #[test]
    fn test_variance_empty() {
        assert_eq!(single(FeatureKind::Variance, &[]), 0.0);
    }

    #[test]
    fn test_variance_constant_data() {
        // All identical → variance must be zero.
        let data = [3.0_f64; 100];
        let v = single(FeatureKind::Variance, &data);
        assert!(v.abs() < 1e-10, "var={v}");
    }

    #[test]
    fn test_variance_population_formula() {
        // E[x²] − E[x]²  for [1, 3]:  E[x²]=5, E[x]=2, var=5-4=1
        let v = single(FeatureKind::Variance, &[1.0, 3.0]);
        assert!((v - 1.0).abs() < 1e-12, "var={v}");
    }

    // ── Skewness ─────────────────────────────────────────────────────────────

    #[test]
    fn test_skewness_empty() {
        assert_eq!(single(FeatureKind::Skewness, &[]), 0.0);
    }

    #[test]
    fn test_skewness_zero_when_sigma_zero() {
        // All identical → σ=0, skewness must be 0.
        let data = [5.0_f64; 10];
        assert_eq!(single(FeatureKind::Skewness, &data), 0.0);
    }

    #[test]
    fn test_skewness_symmetric_distribution() {
        // A perfectly symmetric distribution has skewness ≈ 0.
        let data: Vec<f64> = (-50..=50).map(|i| i as f64).collect();
        let v = single(FeatureKind::Skewness, &data);
        assert!(v.abs() < 1e-10, "skewness={v}");
    }

    #[test]
    fn test_skewness_right_skewed() {
        // Data with a long right tail should have positive skewness.
        let data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0];
        let v = single(FeatureKind::Skewness, &data);
        assert!(v > 0.0, "expected positive skewness, got {v}");
    }

    // ── Min / Max ─────────────────────────────────────────────────────────────

    #[test]
    fn test_min_basic() {
        assert!((single(FeatureKind::Min, &[3.0, 1.0, 4.0, 1.0, 5.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_max_basic() {
        assert!((single(FeatureKind::Max, &[3.0, 1.0, 4.0, 1.0, 5.0]) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_min_empty() {
        assert_eq!(single(FeatureKind::Min, &[]), 0.0);
    }

    #[test]
    fn test_max_empty() {
        assert_eq!(single(FeatureKind::Max, &[]), 0.0);
    }

    // ── Range ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_range_basic() {
        let v = single(FeatureKind::Range, &[1.0, 5.0, 3.0, -2.0]);
        assert!((v - 7.0).abs() < 1e-12, "range={v}");
    }

    #[test]
    fn test_range_empty() {
        assert_eq!(single(FeatureKind::Range, &[]), 0.0);
    }

    #[test]
    fn test_range_single() {
        assert_eq!(single(FeatureKind::Range, &[42.0]), 0.0);
    }

    // ── L1Norm ───────────────────────────────────────────────────────────────

    #[test]
    fn test_l1norm_basic() {
        // |1| + |-2| + |3| + |-4| = 10
        let v = single(FeatureKind::L1Norm, &[1.0, -2.0, 3.0, -4.0]);
        assert!((v - 10.0).abs() < 1e-12, "l1={v}");
    }

    #[test]
    fn test_l1norm_empty() {
        assert_eq!(single(FeatureKind::L1Norm, &[]), 0.0);
    }

    // ── L2Norm ───────────────────────────────────────────────────────────────

    #[test]
    fn test_l2norm_basic() {
        // √(3² + 4²) = 5
        let v = single(FeatureKind::L2Norm, &[3.0, 4.0]);
        assert!((v - 5.0).abs() < 1e-12, "l2={v}");
    }

    #[test]
    fn test_l2norm_empty() {
        assert_eq!(single(FeatureKind::L2Norm, &[]), 0.0);
    }

    // ── Histogram ─────────────────────────────────────────────────────────────

    #[test]
    fn test_histogram_uniform_distribution() {
        // 100 values uniform in [0, 100), 10 bins → each bin ≈ 10 counts.
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut ex = extractor_for(vec![FeatureKind::Histogram { bins: 10 }]);
        let res = ex.extract(0, &data);
        let counts = &res.features[0].values;
        assert_eq!(counts.len(), 10);
        for &c in counts {
            // each bucket should have exactly 10 elements
            assert!((c - 10.0).abs() < 1.0, "bucket count unexpected: {c}");
        }
    }

    #[test]
    fn test_histogram_min_max_all_in_bucket_zero() {
        // When min == max every element must land in bin 0.
        let data = [7.0_f64; 20];
        let mut ex = extractor_for(vec![FeatureKind::Histogram { bins: 5 }]);
        let res = ex.extract(0, &data);
        let counts = &res.features[0].values;
        assert_eq!(counts.len(), 5);
        assert!((counts[0] - 20.0).abs() < 1e-12, "bin0={}", counts[0]);
        for &c in &counts[1..] {
            assert_eq!(c, 0.0);
        }
    }

    #[test]
    fn test_histogram_empty_data() {
        let mut ex = extractor_for(vec![FeatureKind::Histogram { bins: 4 }]);
        let res = ex.extract(0, &[]);
        let counts = &res.features[0].values;
        assert_eq!(counts.len(), 4);
        for &c in counts {
            assert_eq!(c, 0.0);
        }
    }

    #[test]
    fn test_histogram_known_distribution() {
        // 3 values in [0,3]: 0, 1, 2 — 3 bins of width 1.
        let data = [0.0, 1.0, 2.0];
        let mut ex = extractor_for(vec![FeatureKind::Histogram { bins: 3 }]);
        let res = ex.extract(0, &data);
        let counts = &res.features[0].values;
        assert_eq!(counts.len(), 3);
        // Each of the three values should land in a distinct bin.
        let total: f64 = counts.iter().sum();
        assert!((total - 3.0).abs() < 1e-12, "total={total}");
    }

    // ── feature_vector flattening ─────────────────────────────────────────────

    #[test]
    fn test_feature_vector_flatten() {
        let mut ex = extractor_for(vec![
            FeatureKind::Mean,
            FeatureKind::Histogram { bins: 3 },
            FeatureKind::L2Norm,
        ]);
        let res = ex.extract(42, &[1.0, 2.0, 3.0]);
        let fv = res.feature_vector();
        // Mean(1) + Histogram(3) + L2Norm(1) = 5 values
        assert_eq!(fv.len(), 5, "fv={fv:?}");
    }

    #[test]
    fn test_feature_vector_all_scalar() {
        let mut ex = extractor_for(vec![
            FeatureKind::Mean,
            FeatureKind::Variance,
            FeatureKind::Min,
            FeatureKind::Max,
            FeatureKind::L2Norm,
        ]);
        let res = ex.extract(0, &[1.0, 2.0, 3.0]);
        assert_eq!(res.feature_vector().len(), 5);
    }

    // ── extract_batch ─────────────────────────────────────────────────────────

    #[test]
    fn test_extract_batch_multiple_tensors() {
        let mut ex = extractor_for(vec![FeatureKind::Mean]);
        let results = ex.extract_batch(vec![
            (1, vec![1.0, 2.0, 3.0]),
            (2, vec![4.0, 5.0, 6.0]),
            (3, vec![7.0, 8.0, 9.0]),
        ]);
        assert_eq!(results.len(), 3);
        assert!((results[0].features[0].values[0] - 2.0).abs() < 1e-12);
        assert!((results[1].features[0].values[0] - 5.0).abs() < 1e-12);
        assert!((results[2].features[0].values[0] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_extract_batch_tensor_ids_preserved() {
        let mut ex = extractor_for(vec![FeatureKind::Max]);
        let results = ex.extract_batch(vec![(99, vec![1.0]), (1000, vec![2.0])]);
        assert_eq!(results[0].tensor_id, 99);
        assert_eq!(results[1].tensor_id, 1000);
    }

    // ── stats accumulation ────────────────────────────────────────────────────

    #[test]
    fn test_stats_accumulate_tensors_processed() {
        let mut ex = extractor_for(vec![FeatureKind::Mean, FeatureKind::Variance]);
        ex.extract(0, &[1.0, 2.0]);
        ex.extract(1, &[3.0, 4.0]);
        ex.extract(2, &[5.0, 6.0]);
        assert_eq!(ex.stats().total_tensors_processed, 3);
    }

    #[test]
    fn test_stats_total_extractions() {
        // 3 features × 2 tensors = 6 extractions
        let mut ex = extractor_for(vec![FeatureKind::Mean, FeatureKind::Min, FeatureKind::Max]);
        ex.extract(0, &[1.0]);
        ex.extract(1, &[2.0]);
        assert_eq!(ex.stats().total_extractions, 6);
    }

    #[test]
    fn test_stats_avg_feature_vector_len() {
        // All-scalar config → fv_len == features.len() always.
        let mut ex = extractor_for(vec![FeatureKind::Mean, FeatureKind::L2Norm]);
        ex.extract(0, &[1.0]);
        ex.extract(1, &[2.0]);
        // avg should be 2.0 (both have fv_len=2)
        assert!((ex.stats().avg_feature_vector_len - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_stats_via_accessor() {
        let mut ex = extractor_for(vec![FeatureKind::Mean]);
        ex.extract(0, &[1.0]);
        let s = ex.stats();
        assert_eq!(s.total_tensors_processed, 1);
    }

    // ── default config ────────────────────────────────────────────────────────

    #[test]
    fn test_default_config_features() {
        let cfg = ExtractorConfig::default();
        assert!(cfg.features.contains(&FeatureKind::Mean));
        assert!(cfg.features.contains(&FeatureKind::Variance));
        assert!(cfg.features.contains(&FeatureKind::Min));
        assert!(cfg.features.contains(&FeatureKind::Max));
        assert!(cfg.features.contains(&FeatureKind::L2Norm));
        assert_eq!(cfg.histogram_bins, 10);
    }

    // ── edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_all_features_empty_data() {
        let kinds = vec![
            FeatureKind::Mean,
            FeatureKind::Variance,
            FeatureKind::Skewness,
            FeatureKind::Min,
            FeatureKind::Max,
            FeatureKind::Range,
            FeatureKind::L1Norm,
            FeatureKind::L2Norm,
            FeatureKind::Histogram { bins: 4 },
        ];
        let mut ex = extractor_for(kinds);
        let res = ex.extract(0, &[]);
        for feat in &res.features {
            match &feat.kind {
                FeatureKind::Histogram { bins } => {
                    assert_eq!(feat.values.len(), *bins);
                    for &v in &feat.values {
                        assert_eq!(v, 0.0);
                    }
                }
                _ => {
                    assert_eq!(feat.values.len(), 1);
                    assert_eq!(feat.values[0], 0.0, "kind={:?}", feat.kind);
                }
            }
        }
    }

    #[test]
    fn test_histogram_bin_total_equals_n() {
        // The total count across all histogram bins must equal data.len().
        let data: Vec<f64> = (0..137).map(|i| i as f64 * 0.7 - 10.0).collect();
        let mut ex = extractor_for(vec![FeatureKind::Histogram { bins: 7 }]);
        let res = ex.extract(0, &data);
        let total: f64 = res.features[0].values.iter().sum();
        assert!((total - data.len() as f64).abs() < 1e-10, "total={total}");
    }
}
