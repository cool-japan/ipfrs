//! Vector quality analysis and validation utilities
//!
//! This module provides tools for analyzing and validating embedding vectors,
//! detecting anomalies, and ensuring data quality in semantic search systems.

/// Statistics about a vector or collection of vectors
#[derive(Debug, Clone)]
pub struct VectorStats {
    /// Mean of all elements
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// L2 norm (magnitude)
    pub l2_norm: f32,
    /// Number of zero elements
    pub zero_count: usize,
    /// Number of NaN or infinite values
    pub invalid_count: usize,
    /// Dimension of the vector
    pub dimension: usize,
}

/// Quality metrics for a vector
#[derive(Debug, Clone)]
pub struct VectorQuality {
    /// Overall quality score (0.0 - 1.0, higher is better)
    pub quality_score: f32,
    /// Whether the vector is valid (no NaN/Inf)
    pub is_valid: bool,
    /// Whether the vector is normalized
    pub is_normalized: bool,
    /// Sparsity ratio (proportion of near-zero elements)
    pub sparsity: f32,
    /// Whether the vector appears to be degenerate (all same values, etc.)
    pub is_degenerate: bool,
    /// Detailed statistics
    pub stats: VectorStats,
}

/// Anomaly detection result
#[derive(Debug, Clone)]
pub struct AnomalyReport {
    /// Whether an anomaly was detected
    pub is_anomaly: bool,
    /// Confidence score (0.0 - 1.0, higher means more confident it's an anomaly)
    pub confidence: f32,
    /// Type of anomaly detected
    pub anomaly_type: AnomalyType,
    /// Human-readable description
    pub description: String,
}

/// Types of anomalies that can be detected
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyType {
    /// Vector contains invalid values (NaN, Inf)
    InvalidValues,
    /// Vector is degenerate (all zeros, all same value, etc.)
    Degenerate,
    /// Vector has unusual magnitude
    UnusualMagnitude,
    /// Vector is too sparse
    TooSparse,
    /// Vector has unusual distribution
    UnusualDistribution,
    /// No anomaly detected
    None,
}

/// Compute statistics for a vector
pub fn compute_stats(vector: &[f32]) -> VectorStats {
    let n = vector.len();
    if n == 0 {
        return VectorStats {
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            l2_norm: 0.0,
            zero_count: 0,
            invalid_count: 0,
            dimension: 0,
        };
    }

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut zero_count = 0;
    let mut invalid_count = 0;

    for &val in vector {
        if !val.is_finite() {
            invalid_count += 1;
            continue;
        }

        sum += val;
        sum_sq += val * val;
        min = min.min(val);
        max = max.max(val);

        if val.abs() < 1e-8 {
            zero_count += 1;
        }
    }

    let mean = sum / n as f32;
    let variance = (sum_sq / n as f32) - (mean * mean);
    let std_dev = variance.sqrt();
    let l2_norm = sum_sq.sqrt();

    VectorStats {
        mean,
        std_dev,
        min,
        max,
        l2_norm,
        zero_count,
        invalid_count,
        dimension: n,
    }
}

/// Analyze vector quality
pub fn analyze_quality(vector: &[f32]) -> VectorQuality {
    let stats = compute_stats(vector);

    // Check validity
    let is_valid = stats.invalid_count == 0;

    // Check if normalized (L2 norm ≈ 1.0)
    let is_normalized = (stats.l2_norm - 1.0).abs() < 0.01;

    // Compute sparsity
    let sparsity = stats.zero_count as f32 / stats.dimension as f32;

    // Check if degenerate
    let is_degenerate = stats.std_dev < 1e-6 || stats.invalid_count > 0;

    // Compute quality score
    let mut quality_score: f32 = 1.0;

    // Penalize invalid values
    if !is_valid {
        quality_score = 0.0;
    } else {
        // Penalize degenerate vectors
        if is_degenerate {
            quality_score *= 0.3;
        }

        // Penalize high sparsity
        if sparsity > 0.9 {
            quality_score *= 0.5;
        } else if sparsity > 0.7 {
            quality_score *= 0.8;
        }

        // Slight bonus for normalized vectors
        if is_normalized {
            quality_score *= 1.05;
        }

        // Cap at 1.0
        quality_score = quality_score.min(1.0);
    }

    VectorQuality {
        quality_score,
        is_valid,
        is_normalized,
        sparsity,
        is_degenerate,
        stats,
    }
}

/// Detect anomalies in a vector compared to a baseline distribution
///
/// This function compares a vector's statistics against expected values
/// to identify potential anomalies.
#[allow(clippy::too_many_arguments)]
pub fn detect_anomaly(
    vector: &[f32],
    expected_mean: f32,
    expected_std_dev: f32,
    expected_l2_norm: f32,
    mean_tolerance: f32,
    std_dev_tolerance: f32,
    norm_tolerance: f32,
) -> AnomalyReport {
    let quality = analyze_quality(vector);

    // Check for invalid values
    if !quality.is_valid {
        return AnomalyReport {
            is_anomaly: true,
            confidence: 1.0,
            anomaly_type: AnomalyType::InvalidValues,
            description: format!(
                "Vector contains {} invalid values (NaN or Inf)",
                quality.stats.invalid_count
            ),
        };
    }

    // Check for degenerate vectors
    if quality.is_degenerate {
        return AnomalyReport {
            is_anomaly: true,
            confidence: 0.95,
            anomaly_type: AnomalyType::Degenerate,
            description: format!("Vector is degenerate: std_dev={:.6}", quality.stats.std_dev),
        };
    }

    // Check sparsity
    if quality.sparsity > 0.95 {
        return AnomalyReport {
            is_anomaly: true,
            confidence: 0.9,
            anomaly_type: AnomalyType::TooSparse,
            description: format!(
                "Vector is too sparse: {:.1}% zeros",
                quality.sparsity * 100.0
            ),
        };
    }

    // Check magnitude
    let norm_diff = (quality.stats.l2_norm - expected_l2_norm).abs();
    if norm_diff > norm_tolerance {
        let confidence = (norm_diff / expected_l2_norm).min(1.0);
        return AnomalyReport {
            is_anomaly: true,
            confidence,
            anomaly_type: AnomalyType::UnusualMagnitude,
            description: format!(
                "Unusual magnitude: {:.4} (expected {:.4} ± {:.4})",
                quality.stats.l2_norm, expected_l2_norm, norm_tolerance
            ),
        };
    }

    // Check mean
    let mean_diff = (quality.stats.mean - expected_mean).abs();
    if mean_diff > mean_tolerance {
        let confidence = (mean_diff / mean_tolerance).min(1.0) * 0.7;
        return AnomalyReport {
            is_anomaly: true,
            confidence,
            anomaly_type: AnomalyType::UnusualDistribution,
            description: format!(
                "Unusual mean: {:.4} (expected {:.4} ± {:.4})",
                quality.stats.mean, expected_mean, mean_tolerance
            ),
        };
    }

    // Check std dev
    let std_diff = (quality.stats.std_dev - expected_std_dev).abs();
    if std_diff > std_dev_tolerance {
        let confidence = (std_diff / std_dev_tolerance).min(1.0) * 0.6;
        return AnomalyReport {
            is_anomaly: true,
            confidence,
            anomaly_type: AnomalyType::UnusualDistribution,
            description: format!(
                "Unusual std dev: {:.4} (expected {:.4} ± {:.4})",
                quality.stats.std_dev, expected_std_dev, std_dev_tolerance
            ),
        };
    }

    // No anomaly detected
    AnomalyReport {
        is_anomaly: false,
        confidence: 0.0,
        anomaly_type: AnomalyType::None,
        description: "No anomaly detected".to_string(),
    }
}

/// Batch statistics for a collection of vectors
#[derive(Debug, Clone)]
pub struct BatchStats {
    /// Number of vectors
    pub count: usize,
    /// Average quality score
    pub avg_quality: f32,
    /// Number of valid vectors
    pub valid_count: usize,
    /// Number of normalized vectors
    pub normalized_count: usize,
    /// Average sparsity
    pub avg_sparsity: f32,
    /// Statistics across all dimensions
    pub overall_stats: VectorStats,
}

/// Compute batch statistics for multiple vectors
pub fn compute_batch_stats(vectors: &[Vec<f32>]) -> BatchStats {
    if vectors.is_empty() {
        return BatchStats {
            count: 0,
            avg_quality: 0.0,
            valid_count: 0,
            normalized_count: 0,
            avg_sparsity: 0.0,
            overall_stats: VectorStats {
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                l2_norm: 0.0,
                zero_count: 0,
                invalid_count: 0,
                dimension: 0,
            },
        };
    }

    let mut total_quality = 0.0;
    let mut valid_count = 0;
    let mut normalized_count = 0;
    let mut total_sparsity = 0.0;

    // Collect per-dimension statistics
    let dim = vectors[0].len();
    let mut dim_sums = vec![0.0; dim];
    let mut dim_counts = vec![0; dim];

    for vector in vectors {
        let quality = analyze_quality(vector);
        total_quality += quality.quality_score;
        if quality.is_valid {
            valid_count += 1;
        }
        if quality.is_normalized {
            normalized_count += 1;
        }
        total_sparsity += quality.sparsity;

        // Accumulate dimension statistics
        for (i, &val) in vector.iter().enumerate() {
            if i < dim && val.is_finite() {
                dim_sums[i] += val;
                dim_counts[i] += 1;
            }
        }
    }

    // Compute overall statistics across all dimensions
    let all_values: Vec<f32> = vectors.iter().flatten().copied().collect();
    let overall_stats = compute_stats(&all_values);

    BatchStats {
        count: vectors.len(),
        avg_quality: total_quality / vectors.len() as f32,
        valid_count,
        normalized_count,
        avg_sparsity: total_sparsity / vectors.len() as f32,
        overall_stats,
    }
}

/// Find outlier vectors in a batch based on their distance from the mean
pub fn find_outliers(vectors: &[Vec<f32>], threshold: f32) -> Vec<usize> {
    if vectors.is_empty() {
        return Vec::new();
    }

    let dim = vectors[0].len();

    // Compute mean vector
    let mut mean_vec = vec![0.0; dim];
    for vector in vectors {
        for (i, &val) in vector.iter().enumerate() {
            if i < dim && val.is_finite() {
                mean_vec[i] += val;
            }
        }
    }
    for val in &mut mean_vec {
        *val /= vectors.len() as f32;
    }

    // Compute distances from mean
    let distances: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(idx, vector)| {
            let dist = compute_l2_distance(vector, &mean_vec);
            (idx, dist)
        })
        .collect();

    // Compute mean and std dev of distances
    let mean_dist: f32 = distances.iter().map(|(_, d)| d).sum::<f32>() / distances.len() as f32;
    let variance: f32 = distances
        .iter()
        .map(|(_, d)| (d - mean_dist).powi(2))
        .sum::<f32>()
        / distances.len() as f32;
    let std_dist = variance.sqrt();

    // Find outliers (distance > mean + threshold * std)
    let outlier_threshold = mean_dist + threshold * std_dist;
    distances
        .into_iter()
        .filter(|(_, dist)| *dist > outlier_threshold)
        .map(|(idx, _)| idx)
        .collect()
}

/// Compute L2 distance between two vectors
fn compute_l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

/// Diversity score for a set of vectors
///
/// Measures how diverse a collection of vectors is (0.0 = all identical, 1.0 = maximally diverse)
pub fn compute_diversity(vectors: &[Vec<f32>]) -> f32 {
    if vectors.len() < 2 {
        return 0.0;
    }

    let mut total_distance = 0.0;
    let mut count = 0;

    for i in 0..vectors.len() {
        for j in (i + 1)..vectors.len() {
            total_distance += compute_l2_distance(&vectors[i], &vectors[j]);
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    // Normalize by the maximum possible distance (assuming unit vectors)
    let avg_distance = total_distance / count as f32;
    let max_distance = 2.0_f32.sqrt(); // Max L2 distance between unit vectors

    (avg_distance / max_distance).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_stats() {
        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_stats(&vector);

        assert_eq!(stats.dimension, 5);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.invalid_count, 0);
    }

    #[test]
    fn test_analyze_quality_valid() {
        let vector = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let quality = analyze_quality(&vector);

        assert!(quality.is_valid);
        assert!(!quality.is_degenerate);
        assert!(quality.quality_score > 0.5);
    }

    #[test]
    fn test_analyze_quality_invalid() {
        let vector = vec![f32::NAN, 0.2, 0.3, 0.4, 0.5];
        let quality = analyze_quality(&vector);

        assert!(!quality.is_valid);
        assert_eq!(quality.quality_score, 0.0);
    }

    #[test]
    fn test_analyze_quality_degenerate() {
        let vector = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let quality = analyze_quality(&vector);

        assert!(quality.is_degenerate);
        assert!(quality.quality_score < 0.5);
    }

    #[test]
    fn test_detect_anomaly_invalid() {
        let vector = vec![f32::NAN, 0.2, 0.3];
        let report = detect_anomaly(&vector, 0.0, 1.0, 1.0, 0.1, 0.1, 0.1);

        assert!(report.is_anomaly);
        assert_eq!(report.anomaly_type, AnomalyType::InvalidValues);
    }

    #[test]
    fn test_detect_anomaly_normal() {
        let vector = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let stats = compute_stats(&vector);
        let report = detect_anomaly(
            &vector,
            stats.mean,
            stats.std_dev,
            stats.l2_norm,
            0.5,
            0.5,
            0.5,
        );

        assert!(!report.is_anomaly);
        assert_eq!(report.anomaly_type, AnomalyType::None);
    }

    #[test]
    fn test_compute_batch_stats() {
        let vectors = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];

        let stats = compute_batch_stats(&vectors);

        assert_eq!(stats.count, 3);
        assert!(stats.avg_quality > 0.0);
        assert_eq!(stats.valid_count, 3);
    }

    #[test]
    fn test_find_outliers() {
        let vectors = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.1, 0.1, 0.1],
            vec![0.2, 0.2, 0.2],
            vec![10.0, 10.0, 10.0], // Obvious outlier
        ];

        let outliers = find_outliers(&vectors, 1.0);

        assert!(
            outliers.contains(&3),
            "Expected vector at index 3 to be detected as outlier"
        );
        assert_eq!(outliers.len(), 1, "Expected exactly one outlier");
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];

        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        let sim2 = cosine_similarity(&a, &c);
        assert!(sim2.abs() < 1e-6);
    }

    #[test]
    fn test_compute_diversity() {
        // All identical vectors
        let identical = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0]];
        assert_eq!(compute_diversity(&identical), 0.0);

        // Diverse vectors
        let diverse = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
        assert!(compute_diversity(&diverse) > 0.5);
    }
}
