//! Embedding Drift Detector
//!
//! Detects distributional drift in embedding populations over time.
//! Used to trigger index rebuilds when the embedding space has shifted significantly.

/// Signal level for detected drift.
#[derive(Debug, Clone, PartialEq)]
pub enum DriftSignal {
    /// No significant drift detected.
    None,
    /// Mild drift; score in [0.1, 0.3).
    Mild { score: f64 },
    /// Moderate drift; score in [0.3, 0.6).
    Moderate { score: f64 },
    /// Severe drift; score >= 0.6.
    Severe { score: f64 },
}

/// Statistics computed from a population of embeddings.
#[derive(Debug, Clone)]
pub struct PopulationStats {
    /// Mean vector across all embeddings.
    pub centroid: Vec<f32>,
    /// Average cosine distance between sampled pairs.
    pub avg_pairwise_dist: f64,
    /// Number of embeddings used to compute these stats.
    pub sample_count: usize,
    /// Unix timestamp (seconds) when these stats were computed.
    pub computed_at_secs: u64,
}

impl PopulationStats {
    /// Returns `true` if no embeddings were included.
    pub fn is_empty(&self) -> bool {
        self.sample_count == 0
    }
}

/// Report produced by [`EmbeddingDriftDetector::detect`].
#[derive(Debug, Clone)]
pub struct DriftReport {
    /// Classified drift signal.
    pub signal: DriftSignal,
    /// Cosine distance between the baseline centroid and the current centroid.
    pub centroid_shift: f64,
    /// Relative change in average pairwise distance:
    /// `|new - old| / old.max(1e-9)`.
    pub spread_change: f64,
    /// Weighted combined score: `0.7 * centroid_shift + 0.3 * spread_change`.
    pub combined_score: f64,
    /// Human-readable recommendation based on the drift signal.
    pub recommendation: String,
}

/// Configuration for [`EmbeddingDriftDetector`].
#[derive(Debug, Clone)]
pub struct DriftDetectorConfig {
    /// Score threshold below which drift is considered absent. Default: 0.1.
    pub mild_threshold: f64,
    /// Score threshold below which drift is considered mild. Default: 0.3.
    pub moderate_threshold: f64,
    /// Score threshold below which drift is considered moderate. Default: 0.6.
    pub severe_threshold: f64,
    /// Maximum number of pairs to sample when estimating pairwise distance. Default: 100.
    pub sample_size: usize,
}

impl Default for DriftDetectorConfig {
    fn default() -> Self {
        Self {
            mild_threshold: 0.1,
            moderate_threshold: 0.3,
            severe_threshold: 0.6,
            sample_size: 100,
        }
    }
}

/// Detects distributional drift between two snapshots of an embedding population.
pub struct EmbeddingDriftDetector {
    /// Detector configuration.
    pub config: DriftDetectorConfig,
    /// Reference snapshot against which current populations are compared.
    pub baseline: Option<PopulationStats>,
}

impl EmbeddingDriftDetector {
    /// Creates a new detector with the supplied configuration.
    pub fn new(config: DriftDetectorConfig) -> Self {
        Self {
            config,
            baseline: None,
        }
    }

    /// Computes [`PopulationStats`] for a slice of embeddings.
    ///
    /// Centroid is the element-wise mean; pairwise distance is estimated by
    /// sampling consecutive pairs `(i, i+1 mod n)` up to `config.sample_size`.
    pub fn compute_stats(&self, embeddings: &[Vec<f32>], now_secs: u64) -> PopulationStats {
        let n = embeddings.len();

        if n == 0 {
            return PopulationStats {
                centroid: Vec::new(),
                avg_pairwise_dist: 0.0,
                sample_count: 0,
                computed_at_secs: now_secs,
            };
        }

        let dim = embeddings[0].len();

        // Compute element-wise mean (centroid).
        let mut centroid = vec![0.0_f64; dim];
        for emb in embeddings {
            for (c, &v) in centroid.iter_mut().zip(emb.iter()) {
                *c += v as f64;
            }
        }
        let centroid: Vec<f32> = centroid.iter().map(|&s| (s / n as f64) as f32).collect();

        // Estimate average pairwise cosine distance using consecutive-pair sampling.
        let max_pairs = if n < 2 {
            0
        } else {
            // n*(n-1)/2 but guard against overflow
            n.saturating_mul(n.saturating_sub(1)) / 2
        };
        let pairs_to_sample = self.config.sample_size.min(max_pairs);

        let avg_pairwise_dist = if pairs_to_sample == 0 {
            0.0
        } else {
            let mut total_dist = 0.0_f64;
            let mut counted = 0usize;
            let mut i = 0usize;
            while counted < pairs_to_sample {
                let a = i % n;
                let b = (i + 1) % n;
                if a != b {
                    total_dist += Self::cosine_distance(&embeddings[a], &embeddings[b]);
                    counted += 1;
                }
                i += 1;
                if i >= n && counted == 0 {
                    // Only one unique element; bail out.
                    break;
                }
                // Safety: prevent infinite loop when only identical pairs exist.
                if i > n * 2 {
                    break;
                }
            }
            if counted > 0 {
                total_dist / counted as f64
            } else {
                0.0
            }
        };

        PopulationStats {
            centroid,
            avg_pairwise_dist,
            sample_count: n,
            computed_at_secs: now_secs,
        }
    }

    /// Stores `stats` as the new baseline.
    pub fn set_baseline(&mut self, stats: PopulationStats) {
        self.baseline = Some(stats);
    }

    /// Returns `true` if a baseline has been set.
    pub fn has_baseline(&self) -> bool {
        self.baseline.is_some()
    }

    /// Detects drift between the stored baseline and the `current` snapshot.
    ///
    /// If no baseline has been set (or the baseline is empty), returns a
    /// zero-score no-drift report.
    pub fn detect(&self, current: &PopulationStats) -> DriftReport {
        let baseline = match &self.baseline {
            Some(b) if !b.is_empty() => b,
            _ => {
                return DriftReport {
                    signal: DriftSignal::None,
                    centroid_shift: 0.0,
                    spread_change: 0.0,
                    combined_score: 0.0,
                    recommendation: "No action".to_string(),
                };
            }
        };

        let centroid_shift = Self::cosine_distance(&baseline.centroid, &current.centroid);

        let spread_change = (current.avg_pairwise_dist - baseline.avg_pairwise_dist).abs()
            / baseline.avg_pairwise_dist.max(1e-9);

        let combined_score = 0.7 * centroid_shift + 0.3 * spread_change;

        let signal = if combined_score < self.config.mild_threshold {
            DriftSignal::None
        } else if combined_score < self.config.moderate_threshold {
            DriftSignal::Mild {
                score: combined_score,
            }
        } else if combined_score < self.config.severe_threshold {
            DriftSignal::Moderate {
                score: combined_score,
            }
        } else {
            DriftSignal::Severe {
                score: combined_score,
            }
        };

        let recommendation = match &signal {
            DriftSignal::None => "No action".to_string(),
            DriftSignal::Mild { .. } => "Monitor".to_string(),
            DriftSignal::Moderate { .. } => "Consider rebuild".to_string(),
            DriftSignal::Severe { .. } => "Rebuild required".to_string(),
        };

        DriftReport {
            signal,
            centroid_shift,
            spread_change,
            combined_score,
            recommendation,
        }
    }

    /// Computes the cosine distance between two vectors.
    ///
    /// Returns `1.0` if either vector has zero norm.
    pub fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
        let len = a.len().min(b.len());
        if len == 0 {
            return 1.0;
        }

        let mut dot = 0.0_f64;
        let mut norm_a = 0.0_f64;
        let mut norm_b = 0.0_f64;

        for i in 0..len {
            let ai = a[i] as f64;
            let bi = b[i] as f64;
            dot += ai * bi;
            norm_a += ai * ai;
            norm_b += bi * bi;
        }

        let norm_a = norm_a.sqrt();
        let norm_b = norm_b.sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            1.0
        } else {
            let cosine_sim = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
            1.0 - cosine_sim
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_detector() -> EmbeddingDriftDetector {
        EmbeddingDriftDetector::new(DriftDetectorConfig::default())
    }

    // ── 1. new() with config ──────────────────────────────────────────────────
    #[test]
    fn test_new_with_config() {
        let config = DriftDetectorConfig {
            mild_threshold: 0.15,
            moderate_threshold: 0.35,
            severe_threshold: 0.65,
            sample_size: 50,
        };
        let detector = EmbeddingDriftDetector::new(config.clone());
        assert_eq!(detector.config.mild_threshold, 0.15);
        assert_eq!(detector.config.sample_size, 50);
        assert!(detector.baseline.is_none());
    }

    // ── 2. compute_stats: empty embeddings ───────────────────────────────────
    #[test]
    fn test_compute_stats_empty() {
        let detector = default_detector();
        let stats = detector.compute_stats(&[], 0);
        assert!(stats.centroid.is_empty());
        assert_eq!(stats.avg_pairwise_dist, 0.0);
        assert_eq!(stats.sample_count, 0);
        assert!(stats.is_empty());
    }

    // ── 3. compute_stats: single embedding ───────────────────────────────────
    #[test]
    fn test_compute_stats_single() {
        let detector = default_detector();
        let emb = vec![1.0_f32, 2.0, 3.0];
        let stats = detector.compute_stats(std::slice::from_ref(&emb), 42);
        assert_eq!(stats.sample_count, 1);
        assert_eq!(stats.computed_at_secs, 42);
        // Centroid should equal the single embedding.
        for (c, e) in stats.centroid.iter().zip(emb.iter()) {
            assert!((c - e).abs() < 1e-5, "centroid {c} != emb {e}");
        }
        // No pairs possible.
        assert_eq!(stats.avg_pairwise_dist, 0.0);
    }

    // ── 4. compute_stats: two identical embeddings ───────────────────────────
    #[test]
    fn test_compute_stats_two_identical() {
        let detector = default_detector();
        let emb = vec![1.0_f32, 0.0, 0.0];
        let stats = detector.compute_stats(&[emb.clone(), emb.clone()], 0);
        assert_eq!(stats.sample_count, 2);
        for (c, e) in stats.centroid.iter().zip(emb.iter()) {
            assert!((c - e).abs() < 1e-5, "centroid mismatch: {c} vs {e}");
        }
        // Two identical unit vectors → cosine distance = 0.
        assert!(
            stats.avg_pairwise_dist < 1e-9,
            "expected ~0, got {}",
            stats.avg_pairwise_dist
        );
    }

    // ── 5. compute_stats: sample_count set correctly ─────────────────────────
    #[test]
    fn test_compute_stats_sample_count() {
        let detector = default_detector();
        let embeddings: Vec<Vec<f32>> = (0..7).map(|i| vec![i as f32, 0.0]).collect();
        let stats = detector.compute_stats(&embeddings, 0);
        assert_eq!(stats.sample_count, 7);
    }

    // ── 6. set_baseline sets baseline ────────────────────────────────────────
    #[test]
    fn test_set_baseline() {
        let mut detector = default_detector();
        assert!(!detector.has_baseline());
        let stats = detector.compute_stats(&[vec![1.0_f32, 0.0]], 0);
        detector.set_baseline(stats);
        assert!(detector.has_baseline());
    }

    // ── 7. detect: no baseline → no-drift report ─────────────────────────────
    #[test]
    fn test_detect_no_baseline() {
        let detector = default_detector();
        let current = PopulationStats {
            centroid: vec![1.0],
            avg_pairwise_dist: 0.1,
            sample_count: 5,
            computed_at_secs: 0,
        };
        let report = detector.detect(&current);
        assert_eq!(report.signal, DriftSignal::None);
        assert_eq!(report.combined_score, 0.0);
        assert_eq!(report.recommendation, "No action");
    }

    // ── 8. detect: identical current and baseline → combined_score ≈ 0 ───────
    #[test]
    fn test_detect_identical_populations() {
        let mut detector = default_detector();
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let baseline = detector.compute_stats(&embeddings, 0);
        let current = detector.compute_stats(&embeddings, 1);
        detector.set_baseline(baseline);
        let report = detector.detect(&current);
        assert!(
            report.combined_score < 1e-6,
            "expected ≈0, got {}",
            report.combined_score
        );
    }

    // ── 9. detect: shifted centroid → centroid_shift > 0 ─────────────────────
    #[test]
    fn test_detect_shifted_centroid() {
        let mut detector = default_detector();
        let baseline_embs: Vec<Vec<f32>> = vec![vec![1.0_f32, 0.0, 0.0]];
        let current_embs: Vec<Vec<f32>> = vec![vec![0.0_f32, 1.0, 0.0]];
        let baseline = detector.compute_stats(&baseline_embs, 0);
        let current = detector.compute_stats(&current_embs, 1);
        detector.set_baseline(baseline);
        let report = detector.detect(&current);
        assert!(
            report.centroid_shift > 0.0,
            "expected centroid_shift > 0, got {}",
            report.centroid_shift
        );
    }

    // ── 10. DriftSignal::None for score < 0.1 ────────────────────────────────
    #[test]
    fn test_signal_none() {
        let config = DriftDetectorConfig::default();
        let mut detector = EmbeddingDriftDetector::new(config);

        // Baseline: unit vector along X.
        let baseline = PopulationStats {
            centroid: vec![1.0_f32, 0.0],
            avg_pairwise_dist: 0.05,
            sample_count: 10,
            computed_at_secs: 0,
        };
        // Current: almost the same centroid — tiny shift → combined_score < 0.1.
        let current = PopulationStats {
            centroid: vec![0.9999_f32, 0.0141], // ~cos dist ≈ 0.0001
            avg_pairwise_dist: 0.05,
            sample_count: 10,
            computed_at_secs: 1,
        };
        detector.set_baseline(baseline);
        let report = detector.detect(&current);
        assert_eq!(report.signal, DriftSignal::None);
    }

    // ── 11. DriftSignal::Mild for score in [0.1, 0.3) ────────────────────────
    #[test]
    fn test_signal_mild() {
        let mut detector = default_detector();
        // combined_score = 0.7 * centroid_shift + 0.3 * spread_change
        // We drive it to ~0.2 by using a moderate centroid shift.
        // Two orthogonal unit vectors → cosine_dist = 1.0; we'll use partial rotation.
        // cos(θ)≈0.8 → dist≈0.2, combined ≈ 0.7*0.2 + 0 = 0.14  (mild)
        let theta: f64 = std::f64::consts::PI * 0.2; // 36°, cos≈0.809
        let baseline = PopulationStats {
            centroid: vec![1.0_f32, 0.0],
            avg_pairwise_dist: 0.1,
            sample_count: 10,
            computed_at_secs: 0,
        };
        let current = PopulationStats {
            centroid: vec![theta.cos() as f32, theta.sin() as f32],
            avg_pairwise_dist: 0.1,
            sample_count: 10,
            computed_at_secs: 1,
        };
        detector.set_baseline(baseline);
        let report = detector.detect(&current);
        let score = report.combined_score;
        assert!(
            (0.1..0.3).contains(&score),
            "expected mild [0.1,0.3), got {score}"
        );
        assert!(matches!(report.signal, DriftSignal::Mild { .. }));
    }

    // ── 12. DriftSignal::Moderate for score in [0.3, 0.6) ───────────────────
    #[test]
    fn test_signal_moderate() {
        let mut detector = default_detector();
        // cos(θ)≈0.5 → dist≈0.5 → combined ≈ 0.7*0.5 = 0.35 (moderate)
        let theta: f64 = std::f64::consts::PI / 3.0; // 60°, cos=0.5
        let baseline = PopulationStats {
            centroid: vec![1.0_f32, 0.0],
            avg_pairwise_dist: 0.1,
            sample_count: 10,
            computed_at_secs: 0,
        };
        let current = PopulationStats {
            centroid: vec![theta.cos() as f32, theta.sin() as f32],
            avg_pairwise_dist: 0.1,
            sample_count: 10,
            computed_at_secs: 1,
        };
        detector.set_baseline(baseline);
        let report = detector.detect(&current);
        let score = report.combined_score;
        assert!(
            (0.3..0.6).contains(&score),
            "expected moderate [0.3,0.6), got {score}"
        );
        assert!(matches!(report.signal, DriftSignal::Moderate { .. }));
    }

    // ── 13. DriftSignal::Severe for score >= 0.6 ─────────────────────────────
    #[test]
    fn test_signal_severe() {
        let mut detector = default_detector();
        // Orthogonal vectors: dist = 1.0, combined = 0.7 → severe
        let baseline = PopulationStats {
            centroid: vec![1.0_f32, 0.0],
            avg_pairwise_dist: 0.1,
            sample_count: 10,
            computed_at_secs: 0,
        };
        let current = PopulationStats {
            centroid: vec![0.0_f32, 1.0],
            avg_pairwise_dist: 0.1,
            sample_count: 10,
            computed_at_secs: 1,
        };
        detector.set_baseline(baseline);
        let report = detector.detect(&current);
        assert!(
            report.combined_score >= 0.6,
            "expected >=0.6, got {}",
            report.combined_score
        );
        assert!(matches!(report.signal, DriftSignal::Severe { .. }));
    }

    // ── 14. recommendation matches signal ────────────────────────────────────
    #[test]
    fn test_recommendation_matches_signal() {
        let mut detector = default_detector();
        // Severe case.
        let baseline = PopulationStats {
            centroid: vec![1.0_f32, 0.0],
            avg_pairwise_dist: 0.1,
            sample_count: 5,
            computed_at_secs: 0,
        };
        let current = PopulationStats {
            centroid: vec![0.0_f32, 1.0],
            avg_pairwise_dist: 0.1,
            sample_count: 5,
            computed_at_secs: 1,
        };
        detector.set_baseline(baseline.clone());
        let report = detector.detect(&current);
        match &report.signal {
            DriftSignal::None => assert_eq!(report.recommendation, "No action"),
            DriftSignal::Mild { .. } => assert_eq!(report.recommendation, "Monitor"),
            DriftSignal::Moderate { .. } => {
                assert_eq!(report.recommendation, "Consider rebuild")
            }
            DriftSignal::Severe { .. } => {
                assert_eq!(report.recommendation, "Rebuild required")
            }
        }

        // Also check the None case explicitly.
        detector.set_baseline(baseline.clone());
        let same_current = PopulationStats {
            centroid: vec![1.0_f32, 0.0],
            avg_pairwise_dist: 0.1,
            sample_count: 5,
            computed_at_secs: 2,
        };
        let report2 = detector.detect(&same_current);
        assert_eq!(report2.recommendation, "No action");
    }

    // ── 15. cosine_distance identical → 0.0 ──────────────────────────────────
    #[test]
    fn test_cosine_distance_identical() {
        let v = vec![3.0_f32, 4.0, 0.0];
        let dist = EmbeddingDriftDetector::cosine_distance(&v, &v);
        assert!(dist < 1e-9, "expected 0, got {dist}");
    }

    // ── 16. cosine_distance orthogonal → 1.0 ─────────────────────────────────
    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let dist = EmbeddingDriftDetector::cosine_distance(&a, &b);
        assert!((dist - 1.0).abs() < 1e-9, "expected 1.0, got {dist}");
    }

    // ── 17. has_baseline true after set_baseline ──────────────────────────────
    #[test]
    fn test_has_baseline_true_after_set() {
        let mut detector = default_detector();
        assert!(!detector.has_baseline(), "should be false before set");
        let stats = PopulationStats {
            centroid: vec![1.0_f32],
            avg_pairwise_dist: 0.0,
            sample_count: 1,
            computed_at_secs: 0,
        };
        detector.set_baseline(stats);
        assert!(detector.has_baseline(), "should be true after set");
    }
}
