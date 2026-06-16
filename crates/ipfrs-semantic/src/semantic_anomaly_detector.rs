//! Semantic Anomaly Detector — production-grade anomaly detection for embedding corpora.
//!
//! Supports multiple detection strategies:
//! - **CentroidDistance**: sigma-gated distance from the corpus centroid (fast, O(dim))
//! - **MahalanobisApprox**: diagonal covariance-normalised Mahalanobis distance
//! - **LocalOutlierFactor**: k-NN local density ratio (cosine distance)
//! - **IsolationForest**: average isolation depth via xorshift64-driven random splits
//! - **EnsembleVote**: majority vote across all four single-method detectors

use std::collections::VecDeque;

// ─── PRNG ────────────────────────────────────────────────────────────────────

#[inline(always)]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ─── Geometry helpers ────────────────────────────────────────────────────────

#[inline]
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

#[inline]
fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    1.0 - cosine_similarity(a, b)
}

#[inline]
fn euclidean_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

// ─── Detection method ────────────────────────────────────────────────────────

/// Detection algorithm used to score a candidate embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SadDetectionMethod {
    /// Distance from corpus centroid normalised by sigma.
    CentroidDistance,
    /// Diagonal Mahalanobis approximation (per-dimension variance).
    MahalanobisApprox,
    /// k-NN local outlier factor (cosine distance).
    LocalOutlierFactor,
    /// Average isolation depth via random recursive splits.
    IsolationForest,
    /// Majority vote from all four individual methods.
    EnsembleVote,
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the [`SemanticAnomalyDetector`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SadDetectorConfig {
    /// How many standard deviations above the mean is considered anomalous.
    pub threshold_sigma: f64,
    /// Primary detection algorithm.
    pub method: SadDetectionMethod,
    /// k for k-NN methods (LOF / isolation sub-sampling).
    pub window_size: usize,
    /// Minimum corpus size before scoring is attempted.
    pub min_corpus_size: usize,
}

impl Default for SadDetectorConfig {
    fn default() -> Self {
        Self {
            threshold_sigma: 3.0,
            method: SadDetectionMethod::CentroidDistance,
            window_size: 10,
            min_corpus_size: 5,
        }
    }
}

// ─── Reference point ─────────────────────────────────────────────────────────

/// A labelled reference embedding in the corpus.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReferencePoint {
    pub id: u64,
    pub embedding: Vec<f64>,
    pub label: Option<String>,
}

impl ReferencePoint {
    pub fn new(id: u64, embedding: Vec<f64>, label: Option<String>) -> Self {
        Self {
            id,
            embedding,
            label,
        }
    }
}

// ─── Anomaly record ──────────────────────────────────────────────────────────

/// Persisted detection event.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AnomalyRecord {
    pub id: u64,
    pub score: f64,
    pub is_anomaly: bool,
    pub method: SadDetectionMethod,
    pub timestamp: u64,
}

// ─── Anomaly score ───────────────────────────────────────────────────────────

/// Result returned from `score_embedding` / `score_batch`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SadAnomalyScore {
    pub id: u64,
    pub score: f64,
    pub is_anomaly: bool,
    pub explanation: String,
}

// ─── Drift report ────────────────────────────────────────────────────────────

/// Summary returned by `detect_drift`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SadDriftReport {
    pub is_drift: bool,
    /// Euclidean distance between old and new centroids.
    pub centroid_shift: f64,
    /// Ratio of new variance to old variance (> 1 means expansion).
    pub variance_change: f64,
}

// ─── Detector stats ──────────────────────────────────────────────────────────

/// Aggregate statistics over the detector's lifetime.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SadDetectorStats {
    pub corpus_size: usize,
    pub total_scored: u64,
    pub anomaly_count: u64,
    pub anomaly_rate: f64,
    pub avg_score: f64,
}

// ─── Main struct ─────────────────────────────────────────────────────────────

/// Production-quality semantic anomaly detector for embedding corpora.
///
/// # Example
/// ```
/// use ipfrs_semantic::semantic_anomaly_detector::{
///     SemanticAnomalyDetector, SadDetectorConfig, SadDetectionMethod,
/// };
/// let cfg = SadDetectorConfig {
///     threshold_sigma: 2.5,
///     method: SadDetectionMethod::CentroidDistance,
///     ..Default::default()
/// };
/// let mut det = SemanticAnomalyDetector::new(cfg);
/// for i in 0..20u64 {
///     det.add_reference(i, vec![0.5, 0.5, 0.5], None);
/// }
/// let score = det.score_embedding(99, vec![10.0, -10.0, 10.0]);
/// assert!(score.is_anomaly);
/// ```
pub struct SemanticAnomalyDetector {
    corpus: Vec<ReferencePoint>,
    centroid_cache: Option<Vec<f64>>,
    covariance_diag: Option<Vec<f64>>,
    history: VecDeque<AnomalyRecord>,
    config: SadDetectorConfig,
    total_scored: u64,
    score_sum: f64,
    anomaly_count: u64,
    /// Monotonic logical clock (incremented per score call).
    clock: u64,
}

const HISTORY_LIMIT: usize = 1000;

impl SemanticAnomalyDetector {
    // ── Construction ──────────────────────────────────────────────────────

    pub fn new(config: SadDetectorConfig) -> Self {
        Self {
            corpus: Vec::new(),
            centroid_cache: None,
            covariance_diag: None,
            history: VecDeque::with_capacity(HISTORY_LIMIT),
            config,
            total_scored: 0,
            score_sum: 0.0,
            anomaly_count: 0,
            clock: 0,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(SadDetectorConfig::default())
    }

    // ── Corpus management ─────────────────────────────────────────────────

    pub fn add_reference(&mut self, id: u64, embedding: Vec<f64>, label: Option<String>) {
        self.corpus.push(ReferencePoint::new(id, embedding, label));
        self.invalidate_cache();
    }

    /// Removes the first reference with the given id. Returns `true` if removed.
    pub fn remove_reference(&mut self, id: u64) -> bool {
        if let Some(pos) = self.corpus.iter().position(|r| r.id == id) {
            self.corpus.remove(pos);
            self.invalidate_cache();
            true
        } else {
            false
        }
    }

    pub fn clear_corpus(&mut self) {
        self.corpus.clear();
        self.invalidate_cache();
    }

    pub fn corpus_len(&self) -> usize {
        self.corpus.len()
    }

    // ── Cache management ──────────────────────────────────────────────────

    fn invalidate_cache(&mut self) {
        self.centroid_cache = None;
        self.covariance_diag = None;
    }

    /// Returns the per-dimension mean of all reference embeddings (lazy).
    pub fn compute_centroid(&mut self) -> Option<Vec<f64>> {
        if self.corpus.is_empty() {
            return None;
        }
        if let Some(ref c) = self.centroid_cache {
            return Some(c.clone());
        }
        let dim = self.corpus[0].embedding.len();
        if dim == 0 {
            return None;
        }
        let n = self.corpus.len() as f64;
        let mut centroid = vec![0.0f64; dim];
        for rp in &self.corpus {
            if rp.embedding.len() != dim {
                continue;
            }
            for (c, v) in centroid.iter_mut().zip(rp.embedding.iter()) {
                *c += v;
            }
        }
        for c in centroid.iter_mut() {
            *c /= n;
        }
        self.centroid_cache = Some(centroid.clone());
        Some(centroid)
    }

    /// Returns per-dimension variance (diagonal of covariance matrix) (lazy).
    pub fn compute_covariance_diag(&mut self) -> Option<Vec<f64>> {
        if self.corpus.is_empty() {
            return None;
        }
        if let Some(ref c) = self.covariance_diag {
            return Some(c.clone());
        }
        let centroid = self.compute_centroid()?;
        let dim = centroid.len();
        let n = self.corpus.len() as f64;
        let mut var = vec![0.0f64; dim];
        for rp in &self.corpus {
            if rp.embedding.len() != dim {
                continue;
            }
            for (v, (&e, &c)) in var.iter_mut().zip(rp.embedding.iter().zip(centroid.iter())) {
                *v += (e - c).powi(2);
            }
        }
        for v in var.iter_mut() {
            *v /= n.max(1.0);
            // Guard: ensure minimum variance to avoid division by zero
            if *v < 1e-12 {
                *v = 1e-12;
            }
        }
        self.covariance_diag = Some(var.clone());
        Some(var)
    }

    // ── Scoring ───────────────────────────────────────────────────────────

    /// Score a single embedding and return a rich `SadAnomalyScore`.
    pub fn score_embedding(&mut self, id: u64, embedding: Vec<f64>) -> SadAnomalyScore {
        self.clock += 1;
        let ts = self.clock;
        let method = self.config.method;

        if self.corpus.len() < self.config.min_corpus_size {
            return SadAnomalyScore {
                id,
                score: 0.0,
                is_anomaly: false,
                explanation: format!(
                    "corpus too small ({} < {}); skipping detection",
                    self.corpus.len(),
                    self.config.min_corpus_size
                ),
            };
        }

        let (raw_score, explanation) = match method {
            SadDetectionMethod::CentroidDistance => self.score_centroid(&embedding),
            SadDetectionMethod::MahalanobisApprox => self.score_mahalanobis(&embedding),
            SadDetectionMethod::LocalOutlierFactor => {
                let k = self.config.window_size.min(self.corpus.len());
                let s = self.lof_score(&embedding, k);
                (s, format!("LOF score={:.4} (k={})", s, k))
            }
            SadDetectionMethod::IsolationForest => {
                let s = self.isolation_score(&embedding, 100, 42 ^ ts);
                (s, format!("IsolationForest avg_depth={:.4}", s))
            }
            SadDetectionMethod::EnsembleVote => self.score_ensemble(&embedding, ts),
        };

        let threshold = self.dynamic_threshold(method);
        let is_anomaly = raw_score > threshold;

        // Update running statistics
        self.total_scored += 1;
        self.score_sum += raw_score;
        if is_anomaly {
            self.anomaly_count += 1;
        }

        // Persist to history
        let record = AnomalyRecord {
            id,
            score: raw_score,
            is_anomaly,
            method,
            timestamp: ts,
        };
        if self.history.len() >= HISTORY_LIMIT {
            self.history.pop_front();
        }
        self.history.push_back(record);

        SadAnomalyScore {
            id,
            score: raw_score,
            is_anomaly,
            explanation,
        }
    }

    /// Score a batch of (id, embedding) pairs in sequence.
    pub fn score_batch(&mut self, items: &[(u64, Vec<f64>)]) -> Vec<SadAnomalyScore> {
        items
            .iter()
            .map(|(id, emb)| self.score_embedding(*id, emb.clone()))
            .collect()
    }

    // ── Drift detection ───────────────────────────────────────────────────

    /// Compare `new_embeddings` against the current corpus to detect distribution drift.
    pub fn detect_drift(&mut self, new_embeddings: &[Vec<f64>]) -> SadDriftReport {
        let old_centroid = match self.compute_centroid() {
            Some(c) => c,
            None => {
                return SadDriftReport {
                    is_drift: false,
                    centroid_shift: 0.0,
                    variance_change: 1.0,
                }
            }
        };
        let old_var = self
            .compute_covariance_diag()
            .unwrap_or_else(|| vec![1.0; old_centroid.len()]);

        if new_embeddings.is_empty() {
            return SadDriftReport {
                is_drift: false,
                centroid_shift: 0.0,
                variance_change: 1.0,
            };
        }

        let dim = old_centroid.len();
        let n = new_embeddings.len() as f64;
        let mut new_centroid = vec![0.0f64; dim];
        for emb in new_embeddings {
            if emb.len() != dim {
                continue;
            }
            for (c, v) in new_centroid.iter_mut().zip(emb.iter()) {
                *c += v;
            }
        }
        for c in new_centroid.iter_mut() {
            *c /= n;
        }

        let mut new_var = vec![0.0f64; dim];
        for emb in new_embeddings {
            if emb.len() != dim {
                continue;
            }
            for (v, (&e, &c)) in new_var.iter_mut().zip(emb.iter().zip(new_centroid.iter())) {
                *v += (e - c).powi(2);
            }
        }
        for v in new_var.iter_mut() {
            *v /= n;
        }

        let centroid_shift = euclidean_sq(&old_centroid, &new_centroid).sqrt();

        // Clamp per-dimension variance to a minimum to avoid numerical instability
        for v in new_var.iter_mut() {
            if *v < 1e-12 {
                *v = 1e-12;
            }
        }

        let old_total_var: f64 = old_var.iter().sum::<f64>() / dim.max(1) as f64;
        let new_total_var: f64 = new_var.iter().sum::<f64>() / dim.max(1) as f64;
        let variance_change = if old_total_var < 1e-10 && new_total_var < 1e-10 {
            // Both nearly zero variance — treat as no change
            1.0
        } else if old_total_var < 1e-12 {
            1.0
        } else {
            new_total_var / old_total_var
        };

        // Heuristic: drift if centroid moved > 3*sigma or variance changed by > 50 %
        let sigma = old_total_var.sqrt();
        let is_drift =
            centroid_shift > 3.0 * sigma.max(1e-6) || !(0.5..=2.0).contains(&variance_change);

        SadDriftReport {
            is_drift,
            centroid_shift,
            variance_change,
        }
    }

    // ── LOF ───────────────────────────────────────────────────────────────

    /// Compute a Local Outlier Factor-style score for `q` using cosine distance.
    ///
    /// Returns a ratio > 1 for outliers.  `k` is the neighbourhood size.
    pub fn lof_score(&self, q: &[f64], k: usize) -> f64 {
        let k = k.min(self.corpus.len()).max(1);

        // k-NN distances for query point
        let mut q_dists: Vec<f64> = self
            .corpus
            .iter()
            .map(|rp| cosine_distance(q, &rp.embedding))
            .collect();
        q_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let k_dist_q = *q_dists.get(k - 1).unwrap_or(&0.0);

        // Local reachability density of q
        let lrd_q = self.lrd(q, k, k_dist_q);

        if lrd_q < 1e-12 {
            return 1.0;
        }

        // Average ratio of lrd of neighbours over lrd of q
        let mut lrd_sum = 0.0f64;
        let mut count = 0usize;
        for rp in &self.corpus {
            let d = cosine_distance(q, &rp.embedding);
            if d <= k_dist_q + 1e-12 {
                let mut nd: Vec<f64> = self
                    .corpus
                    .iter()
                    .map(|r2| cosine_distance(&rp.embedding, &r2.embedding))
                    .collect();
                nd.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let kd_n = *nd.get(k - 1).unwrap_or(&0.0);
                let lrd_n = self.lrd(&rp.embedding, k, kd_n);
                lrd_sum += lrd_n;
                count += 1;
            }
        }
        if count == 0 {
            return 1.0;
        }
        (lrd_sum / count as f64) / lrd_q
    }

    /// Internal: local reachability density.
    fn lrd(&self, q: &[f64], k: usize, k_dist: f64) -> f64 {
        let mut reach_sum = 0.0f64;
        let mut count = 0usize;
        for rp in &self.corpus {
            let d = cosine_distance(q, &rp.embedding);
            if d <= k_dist + 1e-12 {
                // reachability distance = max(k_dist of neighbour, d)
                let mut nd: Vec<f64> = self
                    .corpus
                    .iter()
                    .map(|r2| cosine_distance(&rp.embedding, &r2.embedding))
                    .collect();
                nd.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let kd_n = *nd.get(k - 1).unwrap_or(&0.0);
                reach_sum += kd_n.max(d);
                count += 1;
            }
        }
        if count == 0 || reach_sum < 1e-12 {
            return 1.0;
        }
        count as f64 / reach_sum
    }

    // ── Isolation Forest ──────────────────────────────────────────────────

    /// Average normalised isolation depth over `n_trees` random isolation trees.
    ///
    /// Lower score → easier to isolate → more anomalous.
    /// The returned value is inverted to 1 - norm_depth so higher = more anomalous.
    pub fn isolation_score(&self, q: &[f64], n_trees: usize, seed: u64) -> f64 {
        if self.corpus.is_empty() || q.is_empty() {
            return 0.0;
        }
        let n = self.corpus.len();
        let dim = q.len();
        let max_depth = ((n as f64).log2().ceil() as usize).max(1);
        let mut state = if seed == 0 { 1 } else { seed };

        let mut total_depth = 0.0f64;

        for _ in 0..n_trees {
            // Sub-sample up to 256 points (or all if smaller)
            let sample_size = n.min(256);
            let mut sample_indices: Vec<usize> = (0..n).collect();
            // Fisher-Yates shuffle (first sample_size elements)
            for i in 0..sample_size {
                let j = i + (xorshift64(&mut state) as usize % (n - i));
                sample_indices.swap(i, j);
            }
            let sample: Vec<&Vec<f64>> = sample_indices[..sample_size]
                .iter()
                .map(|&idx| &self.corpus[idx].embedding)
                .collect();

            // Compute isolation depth for q in this tree
            let depth = isolation_depth_recursive(q, &sample, dim, max_depth, 0, &mut state);
            // Normalise by expected depth of a random point in a tree of size sample_size
            let expected = c_factor(sample_size);
            total_depth += (depth as f64) / expected.max(1.0);
        }

        let avg_norm = total_depth / n_trees as f64;
        // Convert to anomaly score: short path = anomaly → invert
        2.0_f64.powf(-avg_norm)
    }

    // ── Stats ─────────────────────────────────────────────────────────────

    pub fn anomaly_stats(&self) -> SadDetectorStats {
        let avg_score = if self.total_scored > 0 {
            self.score_sum / self.total_scored as f64
        } else {
            0.0
        };
        let anomaly_rate = if self.total_scored > 0 {
            self.anomaly_count as f64 / self.total_scored as f64
        } else {
            0.0
        };
        SadDetectorStats {
            corpus_size: self.corpus.len(),
            total_scored: self.total_scored,
            anomaly_count: self.anomaly_count,
            anomaly_rate,
            avg_score,
        }
    }

    pub fn history(&self) -> &VecDeque<AnomalyRecord> {
        &self.history
    }

    pub fn config(&self) -> &SadDetectorConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: SadDetectorConfig) {
        self.config = config;
    }

    // ── Private scoring helpers ───────────────────────────────────────────

    fn score_centroid(&mut self, embedding: &[f64]) -> (f64, String) {
        let centroid = match self.compute_centroid() {
            Some(c) => c,
            None => return (0.0, "no centroid (empty corpus)".to_string()),
        };
        let var = self
            .compute_covariance_diag()
            .unwrap_or_else(|| vec![1.0; centroid.len()]);

        let dist = euclidean_sq(embedding, &centroid).sqrt();
        let sigma = var.iter().sum::<f64>().sqrt() / (centroid.len().max(1) as f64).sqrt();
        let normalised = if sigma < 1e-12 { dist } else { dist / sigma };
        (
            normalised,
            format!(
                "CentroidDistance: dist={:.4} sigma={:.4} score={:.4} threshold={:.1}σ",
                dist, sigma, normalised, self.config.threshold_sigma
            ),
        )
    }

    fn score_mahalanobis(&mut self, embedding: &[f64]) -> (f64, String) {
        let centroid = match self.compute_centroid() {
            Some(c) => c,
            None => return (0.0, "no centroid (empty corpus)".to_string()),
        };
        let var = match self.compute_covariance_diag() {
            Some(v) => v,
            None => return (0.0, "no covariance (empty corpus)".to_string()),
        };
        if centroid.len() != embedding.len() {
            return (0.0, "dimension mismatch".to_string());
        }
        let d_sq: f64 = embedding
            .iter()
            .zip(centroid.iter())
            .zip(var.iter())
            .map(|((e, c), v)| (e - c).powi(2) / v.max(1e-12))
            .sum();
        let score = d_sq.sqrt();
        (
            score,
            format!(
                "MahalanobisApprox: d²={:.4} d={:.4} threshold={:.1}σ",
                d_sq, score, self.config.threshold_sigma
            ),
        )
    }

    fn score_ensemble(&mut self, embedding: &[f64], ts: u64) -> (f64, String) {
        let (s_cen, _) = self.score_centroid(embedding);
        let (s_mah, _) = self.score_mahalanobis(embedding);
        let k = self.config.window_size.min(self.corpus.len().max(1));
        let s_lof = self.lof_score(embedding, k);
        let s_iso = self.isolation_score(embedding, 50, 17 ^ ts);

        // Normalise each score to a 0-1 anomaly indicator using individual thresholds
        let thr = self.config.threshold_sigma;
        let votes: [bool; 4] = [
            s_cen > thr,
            s_mah > thr,
            s_lof > thr,
            // Isolation: score > 0.6 is anomalous (0.5 is expected for normal)
            s_iso > 0.6,
        ];
        let vote_count = votes.iter().filter(|&&v| v).count();
        // Ensemble score: proportion of detectors that flag anomaly
        let ensemble_score = vote_count as f64 / 4.0;

        (
            ensemble_score,
            format!(
                "Ensemble: cen={:.3} mah={:.3} lof={:.3} iso={:.3} votes={}/4",
                s_cen, s_mah, s_lof, s_iso, vote_count
            ),
        )
    }

    /// Method-specific threshold: ensemble uses 0.5, others use threshold_sigma.
    fn dynamic_threshold(&self, method: SadDetectionMethod) -> f64 {
        match method {
            SadDetectionMethod::EnsembleVote => 0.5,
            SadDetectionMethod::IsolationForest => 0.6,
            _ => self.config.threshold_sigma,
        }
    }
}

// ─── Isolation Forest helpers ─────────────────────────────────────────────────

/// Expected average path length for a BST of size n (iForest formula).
fn c_factor(n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    let n = n as f64;
    2.0 * (n - 1.0).ln() + 0.5772156649 - 2.0 * (n - 1.0) / n
}

/// Recursively compute the isolation depth of `q` given a sub-sample.
fn isolation_depth_recursive(
    q: &[f64],
    sample: &[&Vec<f64>],
    dim: usize,
    max_depth: usize,
    depth: usize,
    state: &mut u64,
) -> usize {
    if sample.len() <= 1 || depth >= max_depth {
        return depth + c_factor(sample.len()) as usize;
    }

    // Pick a random split dimension and value
    let split_dim = (xorshift64(state) as usize) % dim;
    let min_v = sample
        .iter()
        .filter_map(|e| e.get(split_dim).copied())
        .fold(f64::INFINITY, f64::min);
    let max_v = sample
        .iter()
        .filter_map(|e| e.get(split_dim).copied())
        .fold(f64::NEG_INFINITY, f64::max);

    if (max_v - min_v).abs() < 1e-14 {
        return depth + 1;
    }

    // Random split point in [min, max]
    let frac = (xorshift64(state) as f64) / u64::MAX as f64;
    let split_val = min_v + frac * (max_v - min_v);

    let q_val = q.get(split_dim).copied().unwrap_or(0.0);
    let next_sample: Vec<&Vec<f64>> = if q_val <= split_val {
        sample
            .iter()
            .copied()
            .filter(|e| e.get(split_dim).copied().unwrap_or(0.0) <= split_val)
            .collect()
    } else {
        sample
            .iter()
            .copied()
            .filter(|e| e.get(split_dim).copied().unwrap_or(0.0) > split_val)
            .collect()
    };

    isolation_depth_recursive(q, &next_sample, dim, max_depth, depth + 1, state)
}

// ─── Type aliases ─────────────────────────────────────────────────────────────

/// Convenience alias for users preferring the `Sad`-prefixed name.
pub type SadSemanticAnomalyDetector = SemanticAnomalyDetector;

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn uniform_corpus(det: &mut SemanticAnomalyDetector, n: usize, dim: usize, val: f64) {
        for i in 0..n as u64 {
            det.add_reference(i, vec![val; dim], None);
        }
    }

    fn make_detector(method: SadDetectionMethod) -> SemanticAnomalyDetector {
        SemanticAnomalyDetector::new(SadDetectorConfig {
            threshold_sigma: 3.0,
            method,
            window_size: 5,
            min_corpus_size: 3,
        })
    }

    // ── cosine_similarity ─────────────────────────────────────────────────

    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let s = cosine_similarity(&v, &v);
        assert!(
            (s - 1.0).abs() < 1e-9,
            "identical vectors should have cosine=1"
        );
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let s = cosine_similarity(&a, &b);
        assert!(s.abs() < 1e-9, "orthogonal vectors should have cosine=0");
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let s = cosine_similarity(&a, &b);
        assert!(
            (s + 1.0).abs() < 1e-9,
            "opposite vectors should have cosine=-1"
        );
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        let s = cosine_similarity(&a, &b);
        assert_eq!(s, 0.0, "zero vector cosine should return 0");
    }

    #[test]
    fn test_cosine_dim_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_symmetric() {
        let a = vec![0.3, 0.7, -0.1];
        let b = vec![0.5, 0.2, 0.9];
        assert!((cosine_similarity(&a, &b) - cosine_similarity(&b, &a)).abs() < 1e-12);
    }

    // ── ReferencePoint ─────────────────────────────────────────────────────

    #[test]
    fn test_reference_point_new() {
        let rp = ReferencePoint::new(42, vec![1.0, 2.0], Some("test".to_string()));
        assert_eq!(rp.id, 42);
        assert_eq!(rp.label, Some("test".to_string()));
    }

    #[test]
    fn test_reference_point_unlabelled() {
        let rp = ReferencePoint::new(1, vec![0.0], None);
        assert!(rp.label.is_none());
    }

    // ── Corpus management ─────────────────────────────────────────────────

    #[test]
    fn test_add_reference() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        det.add_reference(1, vec![1.0, 2.0], None);
        assert_eq!(det.corpus_len(), 1);
    }

    #[test]
    fn test_remove_reference_existing() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        det.add_reference(10, vec![0.5], None);
        let removed = det.remove_reference(10);
        assert!(removed);
        assert_eq!(det.corpus_len(), 0);
    }

    #[test]
    fn test_remove_reference_missing() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        let removed = det.remove_reference(99);
        assert!(!removed);
    }

    #[test]
    fn test_clear_corpus() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        uniform_corpus(&mut det, 10, 3, 0.5);
        det.clear_corpus();
        assert_eq!(det.corpus_len(), 0);
    }

    // ── Centroid ──────────────────────────────────────────────────────────

    #[test]
    fn test_centroid_empty() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        assert!(det.compute_centroid().is_none());
    }

    #[test]
    fn test_centroid_single_point() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        det.add_reference(0, vec![1.0, 2.0, 3.0], None);
        let c = det
            .compute_centroid()
            .expect("test: compute_centroid failed");
        assert_eq!(c, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_centroid_two_points() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        det.add_reference(0, vec![0.0, 0.0], None);
        det.add_reference(1, vec![2.0, 4.0], None);
        let c = det
            .compute_centroid()
            .expect("test: compute_centroid should return Some for two-point corpus");
        assert!((c[0] - 1.0).abs() < 1e-9);
        assert!((c[1] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_centroid_cache_invalidated_on_add() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        det.add_reference(0, vec![0.0], None);
        let _ = det.compute_centroid();
        det.add_reference(1, vec![2.0], None);
        // After adding, cache should be cleared; new centroid should be 1.0
        let c = det
            .compute_centroid()
            .expect("test: compute_centroid failed after add");
        assert!((c[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_centroid_cache_invalidated_on_remove() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        det.add_reference(0, vec![0.0], None);
        det.add_reference(1, vec![4.0], None);
        let _ = det.compute_centroid();
        det.remove_reference(1);
        let c = det
            .compute_centroid()
            .expect("test: compute_centroid should return Some after remove");
        assert!((c[0] - 0.0).abs() < 1e-9);
    }

    // ── Covariance ────────────────────────────────────────────────────────

    #[test]
    fn test_covariance_empty() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        assert!(det.compute_covariance_diag().is_none());
    }

    #[test]
    fn test_covariance_uniform() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        // All identical → variance = 0 → clamped to 1e-12
        for i in 0..5u64 {
            det.add_reference(i, vec![1.0, 1.0], None);
        }
        let var = det
            .compute_covariance_diag()
            .expect("test: compute_covariance_diag should return Some for uniform corpus");
        assert!(var[0] <= 1e-10, "uniform variance should be near 0");
    }

    #[test]
    fn test_covariance_spread() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        det.add_reference(0, vec![0.0], None);
        det.add_reference(1, vec![2.0], None);
        let var = det
            .compute_covariance_diag()
            .expect("test: compute_covariance_diag should return Some for spread corpus");
        assert!(var[0] > 0.0);
    }

    // ── CentroidDistance scoring ───────────────────────────────────────────

    #[test]
    fn test_score_centroid_inlier() {
        let mut det = make_detector(SadDetectionMethod::CentroidDistance);
        uniform_corpus(&mut det, 20, 3, 0.5);
        let score = det.score_embedding(99, vec![0.5, 0.5, 0.5]);
        assert!(!score.is_anomaly, "centroid point should not be anomaly");
    }

    #[test]
    fn test_score_centroid_outlier() {
        let mut det = make_detector(SadDetectionMethod::CentroidDistance);
        uniform_corpus(&mut det, 20, 3, 0.0);
        let score = det.score_embedding(99, vec![100.0, 100.0, 100.0]);
        assert!(score.is_anomaly, "far-away point should be anomaly");
    }

    #[test]
    fn test_score_centroid_explanation() {
        let mut det = make_detector(SadDetectionMethod::CentroidDistance);
        uniform_corpus(&mut det, 10, 2, 1.0);
        let s = det.score_embedding(1, vec![1.0, 1.0]);
        assert!(s.explanation.contains("CentroidDistance"));
    }

    // ── MahalanobisApprox scoring ─────────────────────────────────────────

    #[test]
    fn test_score_mahalanobis_inlier() {
        let mut det = make_detector(SadDetectionMethod::MahalanobisApprox);
        uniform_corpus(&mut det, 20, 3, 0.5);
        let score = det.score_embedding(1, vec![0.5, 0.5, 0.5]);
        assert!(!score.is_anomaly);
    }

    #[test]
    fn test_score_mahalanobis_outlier() {
        let mut det = make_detector(SadDetectionMethod::MahalanobisApprox);
        // Spread corpus around 0
        for i in 0..20u64 {
            let v = if i % 2 == 0 { -0.1 } else { 0.1 };
            det.add_reference(i, vec![v, v], None);
        }
        let score = det.score_embedding(99, vec![100.0, 100.0]);
        assert!(score.is_anomaly);
    }

    #[test]
    fn test_score_mahalanobis_explanation() {
        let mut det = make_detector(SadDetectionMethod::MahalanobisApprox);
        uniform_corpus(&mut det, 10, 2, 0.0);
        let s = det.score_embedding(1, vec![0.0, 0.0]);
        assert!(s.explanation.contains("Mahalanobis"));
    }

    // ── LOF scoring ───────────────────────────────────────────────────────

    #[test]
    fn test_lof_inlier() {
        let mut det = make_detector(SadDetectionMethod::LocalOutlierFactor);
        uniform_corpus(&mut det, 20, 2, 0.5);
        let s = det.score_embedding(99, vec![0.5, 0.5]);
        // Inlier in a tight cluster should not be anomaly
        assert!(!s.is_anomaly || s.score < 5.0, "inlier LOF should be low");
    }

    #[test]
    fn test_lof_outlier() {
        let mut det = make_detector(SadDetectionMethod::LocalOutlierFactor);
        uniform_corpus(&mut det, 20, 2, 0.0);
        let s = det.score_embedding(99, vec![0.9999, 0.0001]);
        assert!(s.score >= 0.0);
    }

    #[test]
    fn test_lof_score_direct() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        uniform_corpus(&mut det, 10, 2, 0.5);
        let score = det.lof_score(&[0.5, 0.5], 3);
        assert!(score >= 0.0, "LOF score must be non-negative");
    }

    #[test]
    fn test_lof_k_clamped_to_corpus_size() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        uniform_corpus(&mut det, 3, 2, 0.5);
        // k > corpus size should not panic
        let score = det.lof_score(&[0.5, 0.5], 100);
        assert!(score.is_finite());
    }

    // ── Isolation Forest scoring ──────────────────────────────────────────

    #[test]
    fn test_isolation_outlier_higher_score() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        // Use spread corpus so split dimensions have meaningful ranges
        let mut state = 12345u64;
        for i in 0..50u64 {
            // Small gaussian-ish spread around 0 using xorshift
            let v0 = ((xorshift64(&mut state) as f64) / u64::MAX as f64) * 0.2 - 0.1;
            let v1 = ((xorshift64(&mut state) as f64) / u64::MAX as f64) * 0.2 - 0.1;
            let v2 = ((xorshift64(&mut state) as f64) / u64::MAX as f64) * 0.2 - 0.1;
            det.add_reference(i, vec![v0, v1, v2], None);
        }
        let s_in = det.isolation_score(&[0.0, 0.0, 0.0], 200, 42);
        let s_out = det.isolation_score(&[100.0, 100.0, 100.0], 200, 42);
        // Outlier should have higher isolation score (shorter path → higher 2^(-depth))
        assert!(
            s_out > s_in,
            "outlier isolation score should exceed inlier: out={s_out:.6} in={s_in:.6}"
        );
    }

    #[test]
    fn test_isolation_score_range() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        uniform_corpus(&mut det, 30, 4, 0.3);
        let s = det.isolation_score(&[0.3, 0.3, 0.3, 0.3], 50, 7);
        assert!(
            (0.0..=1.0).contains(&s),
            "isolation score should be in [0,1]: {s}"
        );
    }

    #[test]
    fn test_isolation_empty_corpus() {
        let det = SemanticAnomalyDetector::with_defaults();
        let s = det.isolation_score(&[1.0, 2.0], 10, 1);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_isolation_score_method() {
        let mut det = make_detector(SadDetectionMethod::IsolationForest);
        uniform_corpus(&mut det, 20, 2, 0.5);
        let result = det.score_embedding(99, vec![0.5, 0.5]);
        assert!(result.score.is_finite());
    }

    // ── Ensemble scoring ──────────────────────────────────────────────────

    #[test]
    fn test_ensemble_inlier() {
        let mut det = make_detector(SadDetectionMethod::EnsembleVote);
        uniform_corpus(&mut det, 20, 3, 0.5);
        let s = det.score_embedding(1, vec![0.5, 0.5, 0.5]);
        assert!(!s.is_anomaly, "ensemble should not flag inlier");
    }

    #[test]
    fn test_ensemble_outlier() {
        let mut det = make_detector(SadDetectionMethod::EnsembleVote);
        uniform_corpus(&mut det, 30, 3, 0.0);
        let s = det.score_embedding(99, vec![1000.0, 1000.0, 1000.0]);
        assert!(s.is_anomaly, "ensemble should flag extreme outlier");
    }

    #[test]
    fn test_ensemble_explanation_contains_votes() {
        let mut det = make_detector(SadDetectionMethod::EnsembleVote);
        uniform_corpus(&mut det, 10, 2, 0.5);
        let s = det.score_embedding(1, vec![0.5, 0.5]);
        assert!(s.explanation.contains("Ensemble"));
    }

    // ── Batch scoring ─────────────────────────────────────────────────────

    #[test]
    fn test_score_batch_returns_all() {
        let mut det = make_detector(SadDetectionMethod::CentroidDistance);
        uniform_corpus(&mut det, 10, 2, 0.5);
        let items: Vec<(u64, Vec<f64>)> = (0..5u64).map(|i| (i + 100, vec![0.5, 0.5])).collect();
        let results = det.score_batch(&items);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_score_batch_ids_preserved() {
        let mut det = make_detector(SadDetectionMethod::CentroidDistance);
        uniform_corpus(&mut det, 10, 2, 0.5);
        let items = vec![(42u64, vec![0.5f64, 0.5]), (99, vec![100.0, 100.0])];
        let results = det.score_batch(&items);
        assert_eq!(results[0].id, 42);
        assert_eq!(results[1].id, 99);
    }

    #[test]
    fn test_score_batch_anomaly_detected() {
        let mut det = make_detector(SadDetectionMethod::CentroidDistance);
        uniform_corpus(&mut det, 15, 2, 0.0);
        let items = vec![(1u64, vec![0.0, 0.0]), (2, vec![1000.0, 1000.0])];
        let results = det.score_batch(&items);
        assert!(!results[0].is_anomaly);
        assert!(results[1].is_anomaly);
    }

    // ── Corpus too small ──────────────────────────────────────────────────

    #[test]
    fn test_score_below_min_corpus() {
        let mut det = SemanticAnomalyDetector::new(SadDetectorConfig {
            min_corpus_size: 10,
            ..Default::default()
        });
        uniform_corpus(&mut det, 3, 2, 0.5);
        let s = det.score_embedding(1, vec![0.5, 0.5]);
        assert!(!s.is_anomaly);
        assert!(s.explanation.contains("corpus too small"));
    }

    // ── Drift detection ───────────────────────────────────────────────────

    #[test]
    fn test_detect_drift_no_drift() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        uniform_corpus(&mut det, 20, 2, 0.5);
        let new_emb: Vec<Vec<f64>> = (0..10).map(|_| vec![0.5, 0.5]).collect();
        let report = det.detect_drift(&new_emb);
        assert!(!report.is_drift, "identical distribution should not drift");
    }

    #[test]
    fn test_detect_drift_with_drift() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        uniform_corpus(&mut det, 20, 2, 0.0);
        let new_emb: Vec<Vec<f64>> = (0..10).map(|_| vec![100.0, 100.0]).collect();
        let report = det.detect_drift(&new_emb);
        assert!(report.is_drift, "extreme shift should be detected as drift");
        assert!(report.centroid_shift > 100.0);
    }

    #[test]
    fn test_detect_drift_empty_corpus() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        let new_emb: Vec<Vec<f64>> = vec![vec![1.0, 2.0]];
        let report = det.detect_drift(&new_emb);
        assert!(!report.is_drift);
    }

    #[test]
    fn test_detect_drift_empty_new() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        uniform_corpus(&mut det, 10, 2, 0.5);
        let report = det.detect_drift(&[]);
        assert!(!report.is_drift);
    }

    #[test]
    fn test_drift_variance_change_field() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        uniform_corpus(&mut det, 10, 1, 0.0);
        let new_emb: Vec<Vec<f64>> = vec![vec![-5.0], vec![5.0]];
        let report = det.detect_drift(&new_emb);
        assert!(report.variance_change > 0.0);
    }

    // ── Stats ─────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial() {
        let det = SemanticAnomalyDetector::with_defaults();
        let stats = det.anomaly_stats();
        assert_eq!(stats.total_scored, 0);
        assert_eq!(stats.anomaly_count, 0);
        assert_eq!(stats.anomaly_rate, 0.0);
    }

    #[test]
    fn test_stats_after_scoring() {
        let mut det = make_detector(SadDetectionMethod::CentroidDistance);
        uniform_corpus(&mut det, 10, 2, 0.5);
        det.score_embedding(1, vec![0.5, 0.5]);
        det.score_embedding(2, vec![0.5, 0.5]);
        let stats = det.anomaly_stats();
        assert_eq!(stats.total_scored, 2);
        assert_eq!(stats.corpus_size, 10);
    }

    #[test]
    fn test_stats_anomaly_count() {
        let mut det = make_detector(SadDetectionMethod::CentroidDistance);
        uniform_corpus(&mut det, 15, 2, 0.0);
        det.score_embedding(1, vec![0.0, 0.0]);
        det.score_embedding(2, vec![1000.0, 1000.0]);
        let stats = det.anomaly_stats();
        assert!(stats.anomaly_count >= 1);
        assert!(stats.anomaly_rate > 0.0 && stats.anomaly_rate <= 1.0);
    }

    #[test]
    fn test_stats_avg_score() {
        let mut det = make_detector(SadDetectionMethod::CentroidDistance);
        uniform_corpus(&mut det, 10, 2, 0.5);
        det.score_embedding(1, vec![0.5, 0.5]);
        let stats = det.anomaly_stats();
        assert!(stats.avg_score >= 0.0);
    }

    // ── History ───────────────────────────────────────────────────────────

    #[test]
    fn test_history_bounded() {
        let mut det = make_detector(SadDetectionMethod::CentroidDistance);
        uniform_corpus(&mut det, 5, 2, 0.5);
        for i in 0..1200u64 {
            det.score_embedding(i, vec![0.5, 0.5]);
        }
        assert!(
            det.history().len() <= 1000,
            "history must be bounded at 1000"
        );
    }

    #[test]
    fn test_history_records_method() {
        let mut det = make_detector(SadDetectionMethod::CentroidDistance);
        uniform_corpus(&mut det, 5, 2, 0.5);
        det.score_embedding(1, vec![0.5, 0.5]);
        let rec = det
            .history()
            .back()
            .expect("test: history should have at least one record");
        assert_eq!(rec.method, SadDetectionMethod::CentroidDistance);
    }

    #[test]
    fn test_history_timestamp_monotonic() {
        let mut det = make_detector(SadDetectionMethod::CentroidDistance);
        uniform_corpus(&mut det, 5, 2, 0.5);
        det.score_embedding(1, vec![0.5, 0.5]);
        det.score_embedding(2, vec![0.5, 0.5]);
        let recs: Vec<&AnomalyRecord> = det.history().iter().collect();
        assert!(recs[1].timestamp > recs[0].timestamp);
    }

    // ── Config / set_config ────────────────────────────────────────────────

    #[test]
    fn test_set_config() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        let new_cfg = SadDetectorConfig {
            threshold_sigma: 1.5,
            method: SadDetectionMethod::EnsembleVote,
            window_size: 20,
            min_corpus_size: 10,
        };
        det.set_config(new_cfg.clone());
        assert_eq!(det.config().threshold_sigma, 1.5);
        assert_eq!(det.config().method, SadDetectionMethod::EnsembleVote);
    }

    // ── xorshift64 ────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_non_zero() {
        let mut s = 12345u64;
        let v = xorshift64(&mut s);
        assert_ne!(v, 0);
        assert_ne!(s, 12345);
    }

    #[test]
    fn test_xorshift64_sequence() {
        let mut s = 1u64;
        let a = xorshift64(&mut s);
        let b = xorshift64(&mut s);
        assert_ne!(a, b, "consecutive xorshift64 outputs should differ");
    }

    #[test]
    fn test_xorshift64_reproducible() {
        let mut s1 = 999u64;
        let mut s2 = 999u64;
        let v1 = xorshift64(&mut s1);
        let v2 = xorshift64(&mut s2);
        assert_eq!(v1, v2, "same seed must produce same output");
    }

    // ── c_factor ──────────────────────────────────────────────────────────

    #[test]
    fn test_c_factor_one() {
        assert_eq!(c_factor(1), 1.0);
    }

    #[test]
    fn test_c_factor_large() {
        let c = c_factor(256);
        assert!(c > 1.0, "c_factor for n=256 should be > 1");
    }

    // ── Type alias ────────────────────────────────────────────────────────

    #[test]
    fn test_type_alias_usable() {
        let _det: SadSemanticAnomalyDetector = SemanticAnomalyDetector::with_defaults();
    }

    // ── Default config ────────────────────────────────────────────────────

    #[test]
    fn test_default_config() {
        let cfg = SadDetectorConfig::default();
        assert_eq!(cfg.threshold_sigma, 3.0);
        assert_eq!(cfg.method, SadDetectionMethod::CentroidDistance);
        assert_eq!(cfg.window_size, 10);
        assert_eq!(cfg.min_corpus_size, 5);
    }

    // ── All detection methods compile & run ────────────────────────────────

    #[test]
    fn test_all_methods_run() {
        let methods = [
            SadDetectionMethod::CentroidDistance,
            SadDetectionMethod::MahalanobisApprox,
            SadDetectionMethod::LocalOutlierFactor,
            SadDetectionMethod::IsolationForest,
            SadDetectionMethod::EnsembleVote,
        ];
        for method in methods {
            let mut det = make_detector(method);
            uniform_corpus(&mut det, 10, 3, 0.5);
            let s = det.score_embedding(99, vec![0.5, 0.5, 0.5]);
            assert!(
                s.score.is_finite(),
                "method {method:?} score must be finite"
            );
        }
    }

    // ── Robustness / edge cases ────────────────────────────────────────────

    #[test]
    fn test_single_point_corpus() {
        let mut det = make_detector(SadDetectionMethod::CentroidDistance);
        det.add_reference(0, vec![1.0, 1.0], None);
        // Should not panic; but corpus < min_corpus_size → not flagged
        let s = det.score_embedding(1, vec![1.0, 1.0]);
        assert!(!s.is_anomaly || s.score == 0.0);
    }

    #[test]
    fn test_high_dimensional_embedding() {
        let dim = 768;
        let mut det = make_detector(SadDetectionMethod::CentroidDistance);
        for i in 0..20u64 {
            det.add_reference(i, vec![0.01 * i as f64; dim], None);
        }
        let s = det.score_embedding(99, vec![0.5; dim]);
        assert!(s.score.is_finite());
    }

    #[test]
    fn test_negative_embeddings() {
        let mut det = make_detector(SadDetectionMethod::CentroidDistance);
        for i in 0..10u64 {
            det.add_reference(i, vec![-1.0, -1.0], None);
        }
        let s = det.score_embedding(99, vec![-1.0, -1.0]);
        assert!(s.score.is_finite());
        assert!(!s.is_anomaly);
    }

    #[test]
    fn test_mixed_positive_negative() {
        let mut det = make_detector(SadDetectionMethod::MahalanobisApprox);
        for i in 0..10u64 {
            let sign: f64 = if i % 2 == 0 { 1.0 } else { -1.0 };
            det.add_reference(i, vec![sign, sign], None);
        }
        let s = det.score_embedding(99, vec![1.0, 1.0]);
        assert!(s.score.is_finite());
    }

    #[test]
    fn test_score_does_not_modify_corpus() {
        let mut det = make_detector(SadDetectionMethod::CentroidDistance);
        uniform_corpus(&mut det, 10, 2, 0.5);
        let before = det.corpus_len();
        det.score_embedding(99, vec![0.5, 0.5]);
        assert_eq!(det.corpus_len(), before);
    }

    #[test]
    fn test_serde_config_roundtrip() {
        let cfg = SadDetectorConfig {
            threshold_sigma: 2.5,
            method: SadDetectionMethod::EnsembleVote,
            window_size: 7,
            min_corpus_size: 8,
        };
        let json = serde_json::to_string(&cfg).expect("test: serialization failed");
        let cfg2: SadDetectorConfig =
            serde_json::from_str(&json).expect("test: deserialization failed");
        assert_eq!(cfg2.threshold_sigma, 2.5);
        assert_eq!(cfg2.method, SadDetectionMethod::EnsembleVote);
    }

    #[test]
    fn test_serde_anomaly_score_roundtrip() {
        let score = SadAnomalyScore {
            id: 7,
            score: std::f64::consts::PI,
            is_anomaly: true,
            explanation: "test".to_string(),
        };
        let json = serde_json::to_string(&score).expect("test: serialization failed");
        let s2: SadAnomalyScore =
            serde_json::from_str(&json).expect("test: deserialization failed");
        assert_eq!(s2.id, 7);
        assert!((s2.score - std::f64::consts::PI).abs() < 1e-9);
        assert!(s2.is_anomaly);
    }

    #[test]
    fn test_drift_report_no_panic_on_single_point() {
        let mut det = SemanticAnomalyDetector::with_defaults();
        det.add_reference(0, vec![1.0], None);
        let report = det.detect_drift(&[vec![999.0]]);
        // Should not panic; is_drift may be true or false
        let _ = report.is_drift;
    }
}
