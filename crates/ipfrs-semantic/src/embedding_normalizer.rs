//! Vector normalization and transformation for embeddings.
//!
//! Provides various normalization strategies (L1, L2, LInf, MinMax, ZScore, UnitVariance)
//! for embedding vectors, along with clipping, cosine similarity, and statistics tracking.

/// The type of normalization to apply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizationType {
    /// L1 (Manhattan) normalization: sum of absolute values equals target_norm.
    L1,
    /// L2 (Euclidean) normalization: Euclidean length equals target_norm.
    L2,
    /// L-infinity normalization: maximum absolute value equals target_norm.
    LInf,
    /// Min-max normalization: scales values to [0, 1].
    MinMax,
    /// Z-score normalization: transforms to mean=0, std_dev=1.
    ZScore,
    /// Unit variance normalization: scales so variance equals 1, preserving mean.
    UnitVariance,
}

/// Configuration for the normalizer.
#[derive(Debug, Clone)]
pub struct NormalizerConfig {
    /// The normalization strategy to use.
    pub norm_type: NormalizationType,
    /// Small value to avoid division by zero.
    pub epsilon: f64,
    /// For L1/L2/LInf: normalize to this magnitude.
    pub target_norm: f64,
    /// Optional lower bound for clipping.
    pub clip_min: Option<f64>,
    /// Optional upper bound for clipping.
    pub clip_max: Option<f64>,
}

impl Default for NormalizerConfig {
    fn default() -> Self {
        Self {
            norm_type: NormalizationType::L2,
            epsilon: 1e-12,
            target_norm: 1.0,
            clip_min: None,
            clip_max: None,
        }
    }
}

/// Statistics about a single normalization operation.
#[derive(Debug, Clone)]
pub struct NormStats {
    /// Norm of the original vector (using the configured norm type).
    pub original_norm: f64,
    /// Norm of the normalized vector.
    pub normalized_norm: f64,
    /// Minimum value in the normalized vector.
    pub min_value: f64,
    /// Maximum value in the normalized vector.
    pub max_value: f64,
    /// Mean of the normalized vector.
    pub mean: f64,
    /// Standard deviation of the normalized vector.
    pub std_dev: f64,
}

/// Aggregate statistics across multiple normalization operations.
#[derive(Debug, Clone, Default)]
pub struct NormalizerStats {
    /// Total number of vectors normalized.
    pub total_normalized: u64,
    /// Total dimensions processed across all vectors.
    pub total_dimensions: u64,
    /// Running average of original norms.
    pub avg_original_norm: f64,
    /// Running average of normalized norms.
    pub avg_normalized_norm: f64,
}

/// Embedding normalizer with configurable strategies and statistics tracking.
pub struct EmbeddingNormalizer {
    config: NormalizerConfig,
    stats: NormalizerStats,
}

impl EmbeddingNormalizer {
    /// Create a new normalizer with the given configuration.
    pub fn new(config: NormalizerConfig) -> Self {
        Self {
            config,
            stats: NormalizerStats::default(),
        }
    }

    /// Normalize a single embedding vector in-place, returning statistics.
    pub fn normalize(&mut self, embedding: &mut [f64]) -> NormStats {
        let original = embedding.to_vec();

        match self.config.norm_type {
            NormalizationType::L1 => {
                let norm = Self::l1_norm(embedding);
                let divisor = if norm < self.config.epsilon {
                    self.config.epsilon
                } else {
                    norm / self.config.target_norm
                };
                for v in embedding.iter_mut() {
                    *v /= divisor;
                }
            }
            NormalizationType::L2 => {
                let norm = Self::l2_norm(embedding);
                let divisor = if norm < self.config.epsilon {
                    self.config.epsilon
                } else {
                    norm / self.config.target_norm
                };
                for v in embedding.iter_mut() {
                    *v /= divisor;
                }
            }
            NormalizationType::LInf => {
                let norm = Self::linf_norm(embedding);
                let divisor = if norm < self.config.epsilon {
                    self.config.epsilon
                } else {
                    norm / self.config.target_norm
                };
                for v in embedding.iter_mut() {
                    *v /= divisor;
                }
            }
            NormalizationType::MinMax => {
                let min_val = embedding.iter().copied().fold(f64::INFINITY, f64::min);
                let max_val = embedding.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let range = max_val - min_val;
                let divisor = if range < self.config.epsilon {
                    self.config.epsilon
                } else {
                    range
                };
                for v in embedding.iter_mut() {
                    *v = (*v - min_val) / divisor;
                }
            }
            NormalizationType::ZScore => {
                let mean = Self::compute_mean(embedding);
                let std_dev = Self::compute_std_dev(embedding, mean);
                let divisor = if std_dev < self.config.epsilon {
                    self.config.epsilon
                } else {
                    std_dev
                };
                for v in embedding.iter_mut() {
                    *v = (*v - mean) / divisor;
                }
            }
            NormalizationType::UnitVariance => {
                let mean = Self::compute_mean(embedding);
                let std_dev = Self::compute_std_dev(embedding, mean);
                let divisor = if std_dev < self.config.epsilon {
                    self.config.epsilon
                } else {
                    std_dev
                };
                for v in embedding.iter_mut() {
                    *v /= divisor;
                }
            }
        }

        self.clip(embedding);

        let norm_stats = Self::compute_norm_stats(&original, embedding);

        // Update running statistics
        self.stats.total_normalized += 1;
        self.stats.total_dimensions += embedding.len() as u64;
        let n = self.stats.total_normalized as f64;
        self.stats.avg_original_norm +=
            (norm_stats.original_norm - self.stats.avg_original_norm) / n;
        self.stats.avg_normalized_norm +=
            (norm_stats.normalized_norm - self.stats.avg_normalized_norm) / n;

        norm_stats
    }

    /// Normalize a batch of embeddings, returning per-vector statistics.
    pub fn normalize_batch(&mut self, embeddings: &mut [Vec<f64>]) -> Vec<NormStats> {
        embeddings
            .iter_mut()
            .map(|emb| self.normalize(emb))
            .collect()
    }

    /// Compute the L1 (Manhattan) norm of a vector.
    pub fn l1_norm(v: &[f64]) -> f64 {
        v.iter().map(|x| x.abs()).sum()
    }

    /// Compute the L2 (Euclidean) norm of a vector.
    pub fn l2_norm(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Compute the L-infinity norm (maximum absolute value) of a vector.
    pub fn linf_norm(v: &[f64]) -> f64 {
        v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
    }

    /// Compute the arithmetic mean of a vector.
    pub fn compute_mean(v: &[f64]) -> f64 {
        if v.is_empty() {
            return 0.0;
        }
        v.iter().sum::<f64>() / v.len() as f64
    }

    /// Compute the population standard deviation given a precomputed mean.
    pub fn compute_std_dev(v: &[f64], mean: f64) -> f64 {
        if v.is_empty() {
            return 0.0;
        }
        let variance = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64;
        variance.sqrt()
    }

    /// Clip embedding values to the configured bounds (if any).
    pub fn clip(&self, embedding: &mut [f64]) {
        if let Some(lo) = self.config.clip_min {
            for v in embedding.iter_mut() {
                if *v < lo {
                    *v = lo;
                }
            }
        }
        if let Some(hi) = self.config.clip_max {
            for v in embedding.iter_mut() {
                if *v > hi {
                    *v = hi;
                }
            }
        }
    }

    /// Compute cosine similarity between two vectors.
    ///
    /// Returns 0.0 if either vector has zero norm.
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a = Self::l2_norm(a);
        let norm_b = Self::l2_norm(b);
        let denom = norm_a * norm_b;
        if denom < 1e-15 {
            0.0
        } else {
            dot / denom
        }
    }

    /// Compute comprehensive statistics comparing original and normalized vectors.
    pub fn compute_norm_stats(original: &[f64], normalized: &[f64]) -> NormStats {
        let original_norm = Self::l2_norm(original);
        let normalized_norm = Self::l2_norm(normalized);
        let min_value = normalized.iter().copied().fold(f64::INFINITY, f64::min);
        let max_value = normalized.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean = Self::compute_mean(normalized);
        let std_dev = Self::compute_std_dev(normalized, mean);

        // Handle empty vectors gracefully
        let min_value = if min_value == f64::INFINITY {
            0.0
        } else {
            min_value
        };
        let max_value = if max_value == f64::NEG_INFINITY {
            0.0
        } else {
            max_value
        };

        NormStats {
            original_norm,
            normalized_norm,
            min_value,
            max_value,
            mean,
            std_dev,
        }
    }

    /// Get a reference to the aggregate normalizer statistics.
    pub fn stats(&self) -> &NormalizerStats {
        &self.stats
    }

    /// Reset all aggregate statistics to their defaults.
    pub fn reset_stats(&mut self) {
        self.stats = NormalizerStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_normalizer(norm_type: NormalizationType) -> EmbeddingNormalizer {
        EmbeddingNormalizer::new(NormalizerConfig {
            norm_type,
            ..NormalizerConfig::default()
        })
    }

    // ---- L1 normalization ----

    #[test]
    fn test_l1_normalization_basic() {
        let mut n = make_normalizer(NormalizationType::L1);
        let mut v = vec![1.0, -2.0, 3.0];
        n.normalize(&mut v);
        let l1 = EmbeddingNormalizer::l1_norm(&v);
        assert!((l1 - 1.0).abs() < 1e-10, "L1 norm should be 1.0, got {l1}");
    }

    #[test]
    fn test_l1_normalization_signs_preserved() {
        let mut n = make_normalizer(NormalizationType::L1);
        let mut v = vec![2.0, -4.0, 6.0];
        n.normalize(&mut v);
        assert!(v[0] > 0.0);
        assert!(v[1] < 0.0);
        assert!(v[2] > 0.0);
    }

    #[test]
    fn test_l1_normalization_uniform() {
        let mut n = make_normalizer(NormalizationType::L1);
        let mut v = vec![1.0, 1.0, 1.0, 1.0];
        n.normalize(&mut v);
        for val in &v {
            assert!((*val - 0.25).abs() < 1e-10);
        }
    }

    // ---- L2 normalization ----

    #[test]
    fn test_l2_normalization_unit_vector() {
        let mut n = make_normalizer(NormalizationType::L2);
        let mut v = vec![3.0, 4.0];
        n.normalize(&mut v);
        let l2 = EmbeddingNormalizer::l2_norm(&v);
        assert!((l2 - 1.0).abs() < 1e-10, "L2 norm should be 1.0, got {l2}");
    }

    #[test]
    fn test_l2_normalization_already_unit() {
        let mut n = make_normalizer(NormalizationType::L2);
        let orig = vec![0.6, 0.8]; // already unit
        let mut v = orig.clone();
        n.normalize(&mut v);
        for (a, b) in v.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_l2_normalization_negative_values() {
        let mut n = make_normalizer(NormalizationType::L2);
        let mut v = vec![-3.0, -4.0];
        n.normalize(&mut v);
        let l2 = EmbeddingNormalizer::l2_norm(&v);
        assert!((l2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_l2_target_norm() {
        let mut n = EmbeddingNormalizer::new(NormalizerConfig {
            norm_type: NormalizationType::L2,
            target_norm: 5.0,
            ..NormalizerConfig::default()
        });
        let mut v = vec![3.0, 4.0];
        n.normalize(&mut v);
        let l2 = EmbeddingNormalizer::l2_norm(&v);
        assert!((l2 - 5.0).abs() < 1e-10, "L2 norm should be 5.0, got {l2}");
    }

    // ---- LInf normalization ----

    #[test]
    fn test_linf_normalization() {
        let mut n = make_normalizer(NormalizationType::LInf);
        let mut v = vec![1.0, -5.0, 3.0];
        n.normalize(&mut v);
        let linf = EmbeddingNormalizer::linf_norm(&v);
        assert!(
            (linf - 1.0).abs() < 1e-10,
            "LInf norm should be 1.0, got {linf}"
        );
    }

    #[test]
    fn test_linf_normalization_positive() {
        let mut n = make_normalizer(NormalizationType::LInf);
        let mut v = vec![2.0, 4.0, 8.0];
        n.normalize(&mut v);
        assert!((v[2] - 1.0).abs() < 1e-10, "Max element should be 1.0");
        assert!((v[0] - 0.25).abs() < 1e-10);
    }

    // ---- MinMax normalization ----

    #[test]
    fn test_minmax_to_zero_one() {
        let mut n = make_normalizer(NormalizationType::MinMax);
        let mut v = vec![10.0, 20.0, 30.0, 40.0];
        n.normalize(&mut v);
        assert!((v[0] - 0.0).abs() < 1e-10, "Min should map to 0.0");
        assert!((v[3] - 1.0).abs() < 1e-10, "Max should map to 1.0");
        assert!((v[1] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_minmax_negative_range() {
        let mut n = make_normalizer(NormalizationType::MinMax);
        let mut v = vec![-10.0, 0.0, 10.0];
        n.normalize(&mut v);
        assert!((v[0] - 0.0).abs() < 1e-10);
        assert!((v[1] - 0.5).abs() < 1e-10);
        assert!((v[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_minmax_all_same() {
        let mut n = make_normalizer(NormalizationType::MinMax);
        let mut v = vec![5.0, 5.0, 5.0];
        n.normalize(&mut v);
        // All same => range ~ 0, epsilon kicks in, all values should be near 0
        for val in &v {
            assert!(val.is_finite());
        }
    }

    // ---- ZScore normalization ----

    #[test]
    fn test_zscore_mean_zero() {
        let mut n = make_normalizer(NormalizationType::ZScore);
        let mut v = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        n.normalize(&mut v);
        let mean = EmbeddingNormalizer::compute_mean(&v);
        assert!(mean.abs() < 1e-10, "Mean should be ~0, got {mean}");
    }

    #[test]
    fn test_zscore_unit_variance() {
        let mut n = make_normalizer(NormalizationType::ZScore);
        let mut v = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        n.normalize(&mut v);
        let mean = EmbeddingNormalizer::compute_mean(&v);
        let std_dev = EmbeddingNormalizer::compute_std_dev(&v, mean);
        assert!(
            (std_dev - 1.0).abs() < 1e-10,
            "Std dev should be ~1.0, got {std_dev}"
        );
    }

    #[test]
    fn test_zscore_symmetric() {
        let mut n = make_normalizer(NormalizationType::ZScore);
        let mut v = vec![-3.0, -1.0, 1.0, 3.0];
        n.normalize(&mut v);
        let mean = EmbeddingNormalizer::compute_mean(&v);
        assert!(mean.abs() < 1e-10);
    }

    // ---- UnitVariance normalization ----

    #[test]
    fn test_unit_variance() {
        let mut n = make_normalizer(NormalizationType::UnitVariance);
        let mut v = vec![2.0, 4.0, 6.0, 8.0];
        n.normalize(&mut v);
        let mean = EmbeddingNormalizer::compute_mean(&v);
        let std_dev = EmbeddingNormalizer::compute_std_dev(&v, mean);
        assert!(
            (std_dev - 1.0).abs() < 1e-10,
            "Std dev should be ~1.0, got {std_dev}"
        );
    }

    #[test]
    fn test_unit_variance_preserves_relative_ordering() {
        let mut n = make_normalizer(NormalizationType::UnitVariance);
        let mut v = vec![1.0, 3.0, 5.0, 7.0];
        n.normalize(&mut v);
        for i in 0..v.len() - 1 {
            assert!(v[i] < v[i + 1], "Ordering should be preserved");
        }
    }

    // ---- Clipping ----

    #[test]
    fn test_clipping_both_bounds() {
        let mut n = EmbeddingNormalizer::new(NormalizerConfig {
            norm_type: NormalizationType::L2,
            clip_min: Some(-0.5),
            clip_max: Some(0.5),
            ..NormalizerConfig::default()
        });
        let mut v = vec![10.0, -10.0, 0.1];
        n.normalize(&mut v);
        for val in &v {
            assert!(*val >= -0.5 && *val <= 0.5, "Value {val} out of clip range");
        }
    }

    #[test]
    fn test_clipping_min_only() {
        let mut n = EmbeddingNormalizer::new(NormalizerConfig {
            norm_type: NormalizationType::L2,
            clip_min: Some(0.0),
            clip_max: None,
            ..NormalizerConfig::default()
        });
        let mut v = vec![3.0, -4.0];
        n.normalize(&mut v);
        for val in &v {
            assert!(*val >= 0.0, "Value {val} should be >= 0.0");
        }
    }

    #[test]
    fn test_clipping_max_only() {
        let mut n = EmbeddingNormalizer::new(NormalizerConfig {
            norm_type: NormalizationType::L2,
            clip_min: None,
            clip_max: Some(0.3),
            ..NormalizerConfig::default()
        });
        let mut v = vec![3.0, 4.0];
        n.normalize(&mut v);
        for val in &v {
            assert!(*val <= 0.3 + 1e-10, "Value {val} should be <= 0.3");
        }
    }

    // ---- Batch normalization ----

    #[test]
    fn test_batch_normalization() {
        let mut n = make_normalizer(NormalizationType::L2);
        let mut batch = vec![vec![3.0, 4.0], vec![1.0, 0.0], vec![0.0, -5.0]];
        let stats_vec = n.normalize_batch(&mut batch);
        assert_eq!(stats_vec.len(), 3);
        for emb in &batch {
            let l2 = EmbeddingNormalizer::l2_norm(emb);
            assert!((l2 - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_batch_stats_tracking() {
        let mut n = make_normalizer(NormalizationType::L2);
        let mut batch = vec![vec![3.0, 4.0], vec![6.0, 8.0]];
        n.normalize_batch(&mut batch);
        assert_eq!(n.stats().total_normalized, 2);
        assert_eq!(n.stats().total_dimensions, 4);
    }

    // ---- Zero vector handling ----

    #[test]
    fn test_zero_vector_l2() {
        let mut n = make_normalizer(NormalizationType::L2);
        let mut v = vec![0.0, 0.0, 0.0];
        let stats = n.normalize(&mut v);
        // Should not panic; values may be very small but finite
        for val in &v {
            assert!(val.is_finite(), "Expected finite, got {val}");
        }
        assert!(stats.original_norm.abs() < 1e-10);
    }

    #[test]
    fn test_zero_vector_minmax() {
        let mut n = make_normalizer(NormalizationType::MinMax);
        let mut v = vec![0.0, 0.0, 0.0];
        n.normalize(&mut v);
        for val in &v {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_zero_vector_zscore() {
        let mut n = make_normalizer(NormalizationType::ZScore);
        let mut v = vec![0.0, 0.0, 0.0];
        n.normalize(&mut v);
        for val in &v {
            assert!(val.is_finite());
        }
    }

    // ---- Cosine similarity ----

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = EmbeddingNormalizer::cosine_similarity(&a, &a);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "Identical vectors => similarity 1.0"
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = EmbeddingNormalizer::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10, "Orthogonal => similarity 0.0, got {sim}");
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = EmbeddingNormalizer::cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-10, "Opposite => similarity -1.0");
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![1.0, 2.0];
        let b = vec![0.0, 0.0];
        let sim = EmbeddingNormalizer::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10, "Zero vector => similarity 0.0");
    }

    // ---- Stats tracking ----

    #[test]
    fn test_stats_initial() {
        let n = make_normalizer(NormalizationType::L2);
        assert_eq!(n.stats().total_normalized, 0);
        assert_eq!(n.stats().total_dimensions, 0);
    }

    #[test]
    fn test_stats_after_normalize() {
        let mut n = make_normalizer(NormalizationType::L2);
        let mut v = vec![3.0, 4.0];
        n.normalize(&mut v);
        assert_eq!(n.stats().total_normalized, 1);
        assert_eq!(n.stats().total_dimensions, 2);
        assert!((n.stats().avg_original_norm - 5.0).abs() < 1e-10);
        assert!((n.stats().avg_normalized_norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reset_stats() {
        let mut n = make_normalizer(NormalizationType::L2);
        let mut v = vec![3.0, 4.0];
        n.normalize(&mut v);
        n.reset_stats();
        assert_eq!(n.stats().total_normalized, 0);
        assert_eq!(n.stats().total_dimensions, 0);
    }

    // ---- High-dimensional vectors ----

    #[test]
    fn test_high_dimensional_l2() {
        let mut n = make_normalizer(NormalizationType::L2);
        let mut v: Vec<f64> = (0..768).map(|i| (i as f64) * 0.01).collect();
        n.normalize(&mut v);
        let l2 = EmbeddingNormalizer::l2_norm(&v);
        assert!(
            (l2 - 1.0).abs() < 1e-8,
            "High-dim L2 norm should be 1.0, got {l2}"
        );
    }

    #[test]
    fn test_high_dimensional_zscore() {
        let mut n = make_normalizer(NormalizationType::ZScore);
        let mut v: Vec<f64> = (0..512).map(|i| (i as f64) * 0.1 - 25.0).collect();
        n.normalize(&mut v);
        let mean = EmbeddingNormalizer::compute_mean(&v);
        assert!(mean.abs() < 1e-8, "High-dim mean should be ~0, got {mean}");
    }

    // ---- Norm preservation after L2 ----

    #[test]
    fn test_norm_preservation_after_l2() {
        let mut n = make_normalizer(NormalizationType::L2);
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        n.normalize(&mut v);
        let l2 = EmbeddingNormalizer::l2_norm(&v);
        assert!((l2 - 1.0).abs() < 1e-10);

        // Normalizing again should keep it at 1.0
        n.normalize(&mut v);
        let l2_again = EmbeddingNormalizer::l2_norm(&v);
        assert!(
            (l2_again - 1.0).abs() < 1e-10,
            "Idempotent L2 normalization"
        );
    }

    #[test]
    fn test_norm_stats_fields() {
        let mut n = make_normalizer(NormalizationType::L2);
        let mut v = vec![3.0, 4.0];
        let stats = n.normalize(&mut v);
        assert!((stats.original_norm - 5.0).abs() < 1e-10);
        assert!((stats.normalized_norm - 1.0).abs() < 1e-10);
        assert!(stats.min_value <= stats.max_value);
        assert!(stats.std_dev >= 0.0);
    }

    #[test]
    fn test_compute_norm_stats_empty() {
        let stats = EmbeddingNormalizer::compute_norm_stats(&[], &[]);
        assert!((stats.original_norm).abs() < 1e-10);
        assert!((stats.mean).abs() < 1e-10);
    }

    #[test]
    fn test_single_element_vector() {
        let mut n = make_normalizer(NormalizationType::L2);
        let mut v = vec![7.0];
        n.normalize(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_l1_norm_function() {
        let v = vec![1.0, -2.0, 3.0];
        assert!((EmbeddingNormalizer::l1_norm(&v) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_l2_norm_function() {
        let v = vec![3.0, 4.0];
        assert!((EmbeddingNormalizer::l2_norm(&v) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_linf_norm_function() {
        let v = vec![1.0, -7.0, 3.0];
        assert!((EmbeddingNormalizer::linf_norm(&v) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_mean_function() {
        let v = vec![2.0, 4.0, 6.0];
        assert!((EmbeddingNormalizer::compute_mean(&v) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_std_dev_function() {
        // std_dev of [2,4,6] with mean 4 => sqrt((4+0+4)/3) = sqrt(8/3) ≈ 1.6329...
        let v = vec![2.0, 4.0, 6.0];
        let sd = EmbeddingNormalizer::compute_std_dev(&v, 4.0);
        let expected = (8.0_f64 / 3.0).sqrt();
        assert!((sd - expected).abs() < 1e-10);
    }
}
