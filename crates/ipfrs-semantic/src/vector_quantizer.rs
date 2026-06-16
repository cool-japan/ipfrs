//! # Vector Quantizer
//!
//! Production-grade product quantization (PQ) for compressing high-dimensional embedding
//! vectors into compact codes for efficient approximate nearest-neighbor (ANN) search.
//!
//! ## Overview
//!
//! Product Quantization divides a high-dimensional vector into `M` subspaces, each of
//! dimension `D/M`, and quantizes each subspace independently using a small codebook of
//! `K` centroids learned via k-means. Each sub-vector is then represented by the index
//! of its nearest centroid — a single `u8` value — so the full vector is stored as `M`
//! bytes regardless of the original dimensionality.
//!
//! ## Algorithm
//!
//! 1. **Training**: For each subspace, run k-means on the projected sub-vectors to learn
//!    `codes_per_subspace` centroids. Initialization picks every `N/K`-th training vector
//!    to seed the codebook (deterministic, no external RNG dependency).
//!
//! 2. **Encoding**: Map each sub-vector to the index (`u8`) of its nearest centroid.
//!
//! 3. **Decoding**: Reconstruct the approximate full vector by concatenating the centroid
//!    vectors retrieved from each codebook.
//!
//! 4. **Distance**: Asymmetric distance computes exact sub-vector distances from query
//!    to reconstructed codes; symmetric distance decodes both codes first.

use std::fmt;
use thiserror::Error;

// ---------------------------------------------------------------------------
// VqError
// ---------------------------------------------------------------------------

/// Errors produced by [`VectorQuantizer`] operations.
#[derive(Debug, Clone, Error)]
pub enum VqError {
    /// Operation requires the quantizer to be trained first.
    #[error("quantizer has not been trained yet")]
    NotTrained,
    /// Input vector has wrong dimensionality.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    /// Not enough training vectors to populate all codebook entries.
    #[error("insufficient training data: needed {needed} vectors, got {got}")]
    InsufficientData { needed: usize, got: usize },
    /// A quantizer code is malformed or inconsistent.
    #[error("invalid quantizer code: {0}")]
    InvalidCode(String),
}

// ---------------------------------------------------------------------------
// QuantizerCode
// ---------------------------------------------------------------------------

/// Compact quantized representation of a vector.
///
/// Contains one `u8` index per subspace — the index of the nearest centroid in
/// the corresponding codebook.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizerCode(pub Vec<u8>);

impl QuantizerCode {
    /// Number of codes (= number of subspaces).
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the code vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for QuantizerCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QuantizerCode({} subspaces)", self.0.len())
    }
}

// ---------------------------------------------------------------------------
// Codebook
// ---------------------------------------------------------------------------

/// A learned codebook for one PQ subspace.
///
/// Contains `num_codes` centroids each of dimension `subspace_dim`.
#[derive(Debug, Clone)]
pub struct Codebook {
    /// Centroid vectors; `centroids[i]` has length `subspace_dim`.
    pub centroids: Vec<Vec<f64>>,
    /// Dimensionality of each centroid (= full dim / num_subspaces).
    pub subspace_dim: usize,
    /// Number of centroids in this codebook (≤ 256).
    pub num_codes: u8,
}

impl Codebook {
    /// Return the index of the centroid nearest to `sub_vec` in squared Euclidean distance.
    pub fn nearest_centroid(&self, sub_vec: &[f64]) -> usize {
        let mut best_idx = 0usize;
        let mut best_dist = f64::MAX;

        for (idx, centroid) in self.centroids.iter().enumerate() {
            let dist = squared_euclidean_f64(sub_vec, centroid);
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }
        best_idx
    }

    /// Return a reference to the centroid identified by `code`.
    #[inline]
    pub fn centroid(&self, code: u8) -> &[f64] {
        &self.centroids[code as usize]
    }
}

// ---------------------------------------------------------------------------
// QuantizationConfig
// ---------------------------------------------------------------------------

/// Configuration for [`VectorQuantizer`].
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Number of PQ subspaces `M`. The input dimension must be divisible by this value.
    pub num_subspaces: usize,
    /// Number of centroids per codebook (`K`). Must be ≤ 256 (fits in a `u8`).
    pub codes_per_subspace: u8,
    /// Maximum k-means iterations per subspace.
    pub max_iterations: usize,
    /// K-means convergence threshold: stop when max centroid shift < this value.
    pub convergence_threshold: f64,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            num_subspaces: 8,
            codes_per_subspace: u8::MAX,
            max_iterations: 100,
            convergence_threshold: 1e-6,
        }
    }
}

impl QuantizationConfig {
    /// Create a new configuration with custom parameters.
    pub fn new(
        num_subspaces: usize,
        codes_per_subspace: u8,
        max_iterations: usize,
        convergence_threshold: f64,
    ) -> Self {
        Self {
            num_subspaces,
            codes_per_subspace,
            max_iterations,
            convergence_threshold,
        }
    }
}

// ---------------------------------------------------------------------------
// QuantizationStats
// ---------------------------------------------------------------------------

/// Runtime statistics for a [`VectorQuantizer`].
#[derive(Debug, Clone, Default)]
pub struct QuantizationStats {
    /// Number of codebooks trained (equals `num_subspaces` after training).
    pub codebooks_trained: usize,
    /// Total number of vectors encoded since creation.
    pub total_encoded: u64,
    /// Total number of decode operations performed.
    pub total_decoded: u64,
    /// Running mean squared reconstruction error per dimension across all encode calls.
    pub avg_encode_error: f64,
}

// ---------------------------------------------------------------------------
// VectorQuantizer
// ---------------------------------------------------------------------------

/// Product-quantization based vector compressor.
///
/// # Example
///
/// ```rust
/// use ipfrs_semantic::vector_quantizer::{VectorQuantizer, QuantizationConfig};
///
/// let config = QuantizationConfig::new(4, 16, 50, 1e-6);
/// let mut vq = VectorQuantizer::new(config);
///
/// // Train on representative data (must have >= codes_per_subspace vectors)
/// let training_data: Vec<Vec<f64>> = (0..32)
///     .map(|i| (0..16).map(|d| (i * 16 + d) as f64 * 0.01).collect())
///     .collect();
/// vq.train(&training_data).unwrap();
///
/// let code = vq.encode(&vec![0.5_f64; 16]).unwrap();
/// let reconstructed = vq.decode(&code).unwrap();
/// assert_eq!(reconstructed.len(), 16);
/// ```
pub struct VectorQuantizer {
    /// Quantization parameters.
    pub config: QuantizationConfig,
    /// One codebook per subspace; populated after [`train`](VectorQuantizer::train).
    pub codebooks: Vec<Codebook>,
    /// Whether the quantizer has been trained.
    pub trained: bool,
    /// Runtime statistics.
    pub stats: QuantizationStats,
}

impl fmt::Debug for VectorQuantizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VectorQuantizer")
            .field("num_subspaces", &self.config.num_subspaces)
            .field("codes_per_subspace", &self.config.codes_per_subspace)
            .field("trained", &self.trained)
            .field("total_encoded", &self.stats.total_encoded)
            .finish()
    }
}

impl VectorQuantizer {
    /// Create a new, untrained quantizer with the given configuration.
    pub fn new(config: QuantizationConfig) -> Self {
        Self {
            config,
            codebooks: Vec::new(),
            trained: false,
            stats: QuantizationStats::default(),
        }
    }

    /// Train the quantizer by running k-means over `vectors` for each subspace.
    ///
    /// # Errors
    ///
    /// - [`VqError::InsufficientData`] when `vectors.len() < codes_per_subspace`.
    /// - [`VqError::DimensionMismatch`] when vectors have inconsistent length.
    pub fn train(&mut self, vectors: &[Vec<f64>]) -> Result<(), VqError> {
        let k = self.config.codes_per_subspace as usize;

        if vectors.len() < k {
            return Err(VqError::InsufficientData {
                needed: k,
                got: vectors.len(),
            });
        }

        // Infer dimension from the first vector.
        let dim = vectors[0].len();
        let m = self.config.num_subspaces;

        if m == 0 {
            return Err(VqError::InvalidCode(
                "num_subspaces must be > 0".to_string(),
            ));
        }

        if !dim.is_multiple_of(m) {
            return Err(VqError::DimensionMismatch {
                expected: dim - (dim % m), // nearest divisible value
                got: dim,
            });
        }

        let sub_dim = dim / m;

        // Validate that all vectors have the correct dimension.
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != dim {
                return Err(VqError::DimensionMismatch {
                    expected: dim,
                    got: v.len(),
                });
            }
            let _ = i;
        }

        let mut codebooks = Vec::with_capacity(m);

        for s in 0..m {
            let start = s * sub_dim;
            let end = start + sub_dim;

            // Collect sub-vectors for this subspace.
            let sub_vecs: Vec<&[f64]> = vectors.iter().map(|v| &v[start..end]).collect();

            let centroids = kmeans_f64(
                &sub_vecs,
                k,
                self.config.max_iterations,
                self.config.convergence_threshold,
            );

            codebooks.push(Codebook {
                centroids,
                subspace_dim: sub_dim,
                num_codes: self.config.codes_per_subspace,
            });
        }

        self.codebooks = codebooks;
        self.trained = true;
        self.stats.codebooks_trained = m;

        Ok(())
    }

    /// Encode a vector into a compact [`QuantizerCode`].
    ///
    /// For each subspace the sub-vector is mapped to the index of its nearest centroid.
    ///
    /// # Errors
    ///
    /// - [`VqError::NotTrained`] when the quantizer has not been trained.
    /// - [`VqError::DimensionMismatch`] when `vector.len()` does not match training dimension.
    pub fn encode(&mut self, vector: &[f64]) -> Result<QuantizerCode, VqError> {
        if !self.trained {
            return Err(VqError::NotTrained);
        }

        let expected_dim = self.expected_dim();
        if vector.len() != expected_dim {
            return Err(VqError::DimensionMismatch {
                expected: expected_dim,
                got: vector.len(),
            });
        }

        let m = self.config.num_subspaces;
        let sub_dim = expected_dim / m;
        let mut codes = Vec::with_capacity(m);
        let mut total_sq_err = 0.0f64;

        for (s, cb) in self.codebooks.iter().enumerate() {
            let start = s * sub_dim;
            let end = start + sub_dim;
            let sub_vec = &vector[start..end];

            let idx = cb.nearest_centroid(sub_vec);
            let code = idx as u8;
            codes.push(code);

            // Accumulate per-element squared error for this subspace.
            let centroid = cb.centroid(code);
            let sq_err: f64 = sub_vec
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| {
                    let d = a - b;
                    d * d
                })
                .sum();
            total_sq_err += sq_err;
        }

        // Per-dimension mean squared error across the full vector.
        let call_error = total_sq_err / expected_dim as f64;

        // Welford-style running mean.
        let n = self.stats.total_encoded;
        self.stats.avg_encode_error = if n == 0 {
            call_error
        } else {
            self.stats.avg_encode_error
                + (call_error - self.stats.avg_encode_error) / (n + 1) as f64
        };
        self.stats.total_encoded += 1;

        Ok(QuantizerCode(codes))
    }

    /// Decode a [`QuantizerCode`] back into an approximate full-dimensional vector.
    ///
    /// Reconstructs the vector by concatenating the centroid vectors from each codebook.
    ///
    /// # Errors
    ///
    /// - [`VqError::NotTrained`] when the quantizer has not been trained.
    /// - [`VqError::InvalidCode`] when the code length does not match the number of subspaces.
    pub fn decode(&mut self, code: &QuantizerCode) -> Result<Vec<f64>, VqError> {
        if !self.trained {
            return Err(VqError::NotTrained);
        }

        let m = self.config.num_subspaces;
        if code.len() != m {
            return Err(VqError::InvalidCode(format!(
                "code length {} does not match num_subspaces {}",
                code.len(),
                m
            )));
        }

        let sub_dim = self.codebooks.first().map_or(0, |cb| cb.subspace_dim);
        let mut result = Vec::with_capacity(m * sub_dim);

        for (s, &c) in code.0.iter().enumerate() {
            let cb = &self.codebooks[s];
            if c as usize >= cb.centroids.len() {
                return Err(VqError::InvalidCode(format!(
                    "code {} at subspace {} is out of range (codebook has {} entries)",
                    c,
                    s,
                    cb.centroids.len()
                )));
            }
            result.extend_from_slice(cb.centroid(c));
        }

        self.stats.total_decoded += 1;

        Ok(result)
    }

    /// Encode a batch of vectors.
    ///
    /// All vectors must have the same dimension as the training data.
    ///
    /// # Errors
    ///
    /// Propagates the first error encountered (see [`encode`](VectorQuantizer::encode)).
    pub fn encode_batch(&mut self, vectors: &[Vec<f64>]) -> Result<Vec<QuantizerCode>, VqError> {
        let mut codes = Vec::with_capacity(vectors.len());
        for v in vectors {
            codes.push(self.encode(v)?);
        }
        Ok(codes)
    }

    /// Compute the asymmetric squared L2 distance between a raw query vector and a code.
    ///
    /// This is more accurate than [`symmetric_distance`](VectorQuantizer::symmetric_distance)
    /// because the query is not quantized — only the database vector is approximated.
    ///
    /// # Errors
    ///
    /// - [`VqError::NotTrained`] when not trained.
    /// - [`VqError::DimensionMismatch`] when query length is wrong.
    /// - [`VqError::InvalidCode`] when the code is malformed.
    pub fn asymmetric_distance(&self, query: &[f64], code: &QuantizerCode) -> Result<f64, VqError> {
        if !self.trained {
            return Err(VqError::NotTrained);
        }

        let expected_dim = self.expected_dim();
        if query.len() != expected_dim {
            return Err(VqError::DimensionMismatch {
                expected: expected_dim,
                got: query.len(),
            });
        }

        let m = self.config.num_subspaces;
        if code.len() != m {
            return Err(VqError::InvalidCode(format!(
                "code length {} does not match num_subspaces {}",
                code.len(),
                m
            )));
        }

        let sub_dim = expected_dim / m;
        let mut total_dist = 0.0f64;

        for (s, &c) in code.0.iter().enumerate() {
            let cb = &self.codebooks[s];
            if c as usize >= cb.centroids.len() {
                return Err(VqError::InvalidCode(format!(
                    "code {} at subspace {} is out of range",
                    c, s
                )));
            }
            let start = s * sub_dim;
            let end = start + sub_dim;
            let sub_vec = &query[start..end];
            let centroid = cb.centroid(c);
            total_dist += squared_euclidean_f64(sub_vec, centroid);
        }

        Ok(total_dist)
    }

    /// Compute the symmetric squared L2 distance between two quantizer codes.
    ///
    /// Both codes are decoded to full vectors before computing the distance.
    /// This is less accurate than [`asymmetric_distance`](VectorQuantizer::asymmetric_distance)
    /// but useful when the query is also stored as a code.
    ///
    /// # Errors
    ///
    /// - [`VqError::NotTrained`] when not trained.
    /// - [`VqError::InvalidCode`] when either code is malformed.
    pub fn symmetric_distance(
        &mut self,
        a: &QuantizerCode,
        b: &QuantizerCode,
    ) -> Result<f64, VqError> {
        let decoded_a = self.decode_immutable(a)?;
        let decoded_b = self.decode_immutable(b)?;
        // Don't double-count the stat; only decode() increments the counter.
        // symmetric_distance calls the internal helper that does not mutate stats.
        Ok(squared_euclidean_f64(&decoded_a, &decoded_b))
    }

    /// Compute the per-dimension mean squared reconstruction error for a vector.
    ///
    /// `||vector - decode(encode(vector))||^2 / dim`
    ///
    /// # Errors
    ///
    /// Propagates errors from [`encode`](VectorQuantizer::encode) and
    /// [`decode`](VectorQuantizer::decode).
    pub fn quantization_error(&mut self, vector: &[f64]) -> Result<f64, VqError> {
        let code = self.encode(vector)?;
        let reconstructed = self.decode(&code)?;
        let sq_err: f64 = vector
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum();
        Ok(sq_err / vector.len() as f64)
    }

    /// Compute the mean quantization error over a batch of vectors.
    ///
    /// # Errors
    ///
    /// Propagates the first error encountered.
    pub fn avg_error_on_batch(&mut self, vectors: &[Vec<f64>]) -> Result<f64, VqError> {
        if vectors.is_empty() {
            return Ok(0.0);
        }
        let total: f64 = vectors
            .iter()
            .map(|v| self.quantization_error(v))
            .collect::<Result<Vec<f64>, VqError>>()?
            .into_iter()
            .sum();
        Ok(total / vectors.len() as f64)
    }

    /// Return `(subspace_idx, num_centroids)` pairs for each codebook.
    pub fn codebook_stats(&self) -> Vec<(usize, usize)> {
        self.codebooks
            .iter()
            .enumerate()
            .map(|(i, cb)| (i, cb.centroids.len()))
            .collect()
    }

    // ---------------------------------------------------------------------------
    // Private helpers
    // ---------------------------------------------------------------------------

    /// Expected input dimension based on the first codebook's sub_dim.
    fn expected_dim(&self) -> usize {
        self.codebooks
            .first()
            .map_or(0, |cb| cb.subspace_dim * self.config.num_subspaces)
    }

    /// Decode a code without mutating stats (used internally for distance computation).
    fn decode_immutable(&self, code: &QuantizerCode) -> Result<Vec<f64>, VqError> {
        if !self.trained {
            return Err(VqError::NotTrained);
        }

        let m = self.config.num_subspaces;
        if code.len() != m {
            return Err(VqError::InvalidCode(format!(
                "code length {} does not match num_subspaces {}",
                code.len(),
                m
            )));
        }

        let sub_dim = self.codebooks.first().map_or(0, |cb| cb.subspace_dim);
        let mut result = Vec::with_capacity(m * sub_dim);

        for (s, &c) in code.0.iter().enumerate() {
            let cb = &self.codebooks[s];
            if c as usize >= cb.centroids.len() {
                return Err(VqError::InvalidCode(format!(
                    "code {} at subspace {} is out of range (codebook has {} entries)",
                    c,
                    s,
                    cb.centroids.len()
                )));
            }
            result.extend_from_slice(cb.centroid(c));
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Internal k-means implementation (f64)
// ---------------------------------------------------------------------------

/// Run Lloyd's k-means algorithm on a slice of sub-vectors.
///
/// - `data`: equal-length sub-vectors to cluster.
/// - `k`: desired number of centroids (clamped to `data.len()`).
/// - `max_iters`: maximum Lloyd iterations.
/// - `tol`: convergence threshold; stops when max centroid shift < `tol`.
///
/// Centroid initialisation: pick every `n / k`-th vector (stride-based deterministic
/// seeding — no random number generator required).
fn kmeans_f64(data: &[&[f64]], k: usize, max_iters: usize, tol: f64) -> Vec<Vec<f64>> {
    if data.is_empty() || k == 0 {
        return Vec::new();
    }

    let dim = data[0].len();
    let n = data.len();
    let actual_k = k.min(n);

    // Stride-based deterministic initialisation.
    let stride = if actual_k >= n { 1 } else { n / actual_k };
    let mut centroids: Vec<Vec<f64>> = (0..actual_k)
        .map(|i| data[(i * stride).min(n - 1)].to_vec())
        .collect();

    let mut assignments = vec![0usize; n];

    for _iter in 0..max_iters {
        // ---- Assignment step ------------------------------------------------
        for (i, sv) in data.iter().enumerate() {
            let mut best = 0usize;
            let mut best_dist = f64::MAX;
            for (j, c) in centroids.iter().enumerate() {
                let d = squared_euclidean_f64(sv, c);
                if d < best_dist {
                    best_dist = d;
                    best = j;
                }
            }
            assignments[i] = best;
        }

        // ---- Update step ----------------------------------------------------
        let mut sums = vec![vec![0.0f64; dim]; actual_k];
        let mut counts = vec![0usize; actual_k];

        for (i, sv) in data.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for (d, &x) in sv.iter().enumerate() {
                sums[c][d] += x;
            }
        }

        let mut max_shift = 0.0f64;
        let mut new_centroids = centroids.clone();

        for j in 0..actual_k {
            if counts[j] > 0 {
                let inv = 1.0 / counts[j] as f64;
                let new_c: Vec<f64> = sums[j].iter().map(|&s| s * inv).collect();
                let shift = squared_euclidean_f64(&new_c, &centroids[j]).sqrt();
                if shift > max_shift {
                    max_shift = shift;
                }
                new_centroids[j] = new_c;
            }
            // Centroid with no assigned points keeps its previous position.
        }

        centroids = new_centroids;

        if max_shift < tol {
            break;
        }
    }

    centroids
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Squared Euclidean distance between two equal-length f64 slices.
#[inline]
fn squared_euclidean_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::vector_quantizer::{
        Codebook, QuantizationConfig, QuantizationStats, QuantizerCode, VectorQuantizer, VqError,
    };

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a deterministic `VectorQuantizer` config with small parameters.
    fn small_config(num_subspaces: usize, codes_per_subspace: u8) -> QuantizationConfig {
        QuantizationConfig::new(num_subspaces, codes_per_subspace, 50, 1e-6)
    }

    /// Generate `n` linearly spaced vectors of `dim` dimensions.
    fn make_vectors(n: usize, dim: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| (0..dim).map(|d| (i * dim + d) as f64 * 0.01).collect())
            .collect()
    }

    /// Build a trained `VectorQuantizer` with `dim`-dimensional vectors.
    ///
    /// Uses `n` training vectors (must be ≥ codes_per_subspace).
    fn trained_vq(dim: usize, num_subspaces: usize, codes: u8, n: usize) -> VectorQuantizer {
        let cfg = small_config(num_subspaces, codes);
        let mut vq = VectorQuantizer::new(cfg);
        let data = make_vectors(n, dim);
        vq.train(&data).expect("training should succeed");
        vq
    }

    // -----------------------------------------------------------------------
    // 1. QuantizationConfig defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_default_values() {
        let cfg = QuantizationConfig::default();
        assert_eq!(cfg.num_subspaces, 8);
        assert_eq!(cfg.codes_per_subspace, u8::MAX);
        assert_eq!(cfg.max_iterations, 100);
        assert!((cfg.convergence_threshold - 1e-6).abs() < f64::EPSILON * 100.0);
    }

    // -----------------------------------------------------------------------
    // 2. QuantizationConfig custom constructor
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_custom_values() {
        let cfg = QuantizationConfig::new(4, 32, 50, 1e-4);
        assert_eq!(cfg.num_subspaces, 4);
        assert_eq!(cfg.codes_per_subspace, 32u8);
        assert_eq!(cfg.max_iterations, 50);
        assert!((cfg.convergence_threshold - 1e-4).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // 3. VectorQuantizer::new initial state
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_is_untrained() {
        let vq = VectorQuantizer::new(QuantizationConfig::default());
        assert!(!vq.trained);
    }

    #[test]
    fn test_new_has_empty_codebooks() {
        let vq = VectorQuantizer::new(QuantizationConfig::default());
        assert!(vq.codebooks.is_empty());
    }

    #[test]
    fn test_new_stats_are_zero() {
        let vq = VectorQuantizer::new(QuantizationConfig::default());
        assert_eq!(vq.stats.total_encoded, 0);
        assert_eq!(vq.stats.total_decoded, 0);
        assert_eq!(vq.stats.codebooks_trained, 0);
        assert_eq!(vq.stats.avg_encode_error, 0.0);
    }

    // -----------------------------------------------------------------------
    // 4. train — insufficient data error
    // -----------------------------------------------------------------------

    #[test]
    fn test_train_insufficient_data() {
        let cfg = small_config(4, 16);
        let mut vq = VectorQuantizer::new(cfg);
        let data = make_vectors(4, 16); // only 4 vectors, need 16
        let result = vq.train(&data);
        assert!(matches!(result, Err(VqError::InsufficientData { .. })));
    }

    // -----------------------------------------------------------------------
    // 5. train — dimension mismatch error
    // -----------------------------------------------------------------------

    #[test]
    fn test_train_dimension_not_divisible() {
        let cfg = QuantizationConfig::new(3, 4, 50, 1e-6); // subspaces=3
        let mut vq = VectorQuantizer::new(cfg);
        // dim=10, not divisible by 3
        let data = make_vectors(10, 10);
        let result = vq.train(&data);
        assert!(matches!(result, Err(VqError::DimensionMismatch { .. })));
    }

    // -----------------------------------------------------------------------
    // 6. train — sets trained flag
    // -----------------------------------------------------------------------

    #[test]
    fn test_train_sets_trained_flag() {
        let vq = trained_vq(16, 4, 4, 20);
        assert!(vq.trained);
    }

    // -----------------------------------------------------------------------
    // 7. train — codebook count equals num_subspaces
    // -----------------------------------------------------------------------

    #[test]
    fn test_train_codebook_count() {
        let m = 4;
        let vq = trained_vq(16, m, 4, 20);
        assert_eq!(vq.codebooks.len(), m);
    }

    // -----------------------------------------------------------------------
    // 8. train — codebook stats populated correctly
    // -----------------------------------------------------------------------

    #[test]
    fn test_train_stats_codebooks_trained() {
        let m = 4;
        let vq = trained_vq(16, m, 4, 20);
        assert_eq!(vq.stats.codebooks_trained, m);
    }

    // -----------------------------------------------------------------------
    // 9. encode — fails when not trained
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_not_trained_error() {
        let mut vq = VectorQuantizer::new(small_config(4, 4));
        let result = vq.encode(&[0.0f64; 16]);
        assert!(matches!(result, Err(VqError::NotTrained)));
    }

    // -----------------------------------------------------------------------
    // 10. encode — fails on wrong dimension
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_dimension_mismatch() {
        let mut vq = trained_vq(16, 4, 4, 20);
        let result = vq.encode(&[0.0f64; 8]); // wrong: 8 instead of 16
        assert!(matches!(
            result,
            Err(VqError::DimensionMismatch {
                expected: 16,
                got: 8
            })
        ));
    }

    // -----------------------------------------------------------------------
    // 11. encode — code length equals num_subspaces
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_code_length() {
        let m = 4;
        let mut vq = trained_vq(16, m, 4, 20);
        let code = vq.encode(&[0.5f64; 16]).expect("encode succeeded");
        assert_eq!(code.len(), m);
    }

    // -----------------------------------------------------------------------
    // 12. encode — increments total_encoded
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_increments_stat() {
        let mut vq = trained_vq(16, 4, 4, 20);
        assert_eq!(vq.stats.total_encoded, 0);
        vq.encode(&[0.1f64; 16])
            .expect("test: encode 0.1 vector should succeed");
        vq.encode(&[0.2f64; 16])
            .expect("test: encode 0.2 vector should succeed");
        assert_eq!(vq.stats.total_encoded, 2);
    }

    // -----------------------------------------------------------------------
    // 13. decode — fails when not trained
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_not_trained_error() {
        let mut vq = VectorQuantizer::new(small_config(4, 4));
        let code = QuantizerCode(vec![0u8; 4]);
        let result = vq.decode(&code);
        assert!(matches!(result, Err(VqError::NotTrained)));
    }

    // -----------------------------------------------------------------------
    // 14. decode — fails on wrong code length
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_invalid_code_length() {
        let mut vq = trained_vq(16, 4, 4, 20);
        let code = QuantizerCode(vec![0u8; 3]); // wrong: 3 instead of 4
        let result = vq.decode(&code);
        assert!(matches!(result, Err(VqError::InvalidCode(_))));
    }

    // -----------------------------------------------------------------------
    // 15. decode — reconstructed vector has correct length
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_output_length() {
        let dim = 16;
        let mut vq = trained_vq(dim, 4, 4, 20);
        let code = vq
            .encode(&vec![0.5f64; dim])
            .expect("test: encode 0.5 vector should succeed");
        let decoded = vq
            .decode(&code)
            .expect("test: decode of valid code should succeed");
        assert_eq!(decoded.len(), dim);
    }

    // -----------------------------------------------------------------------
    // 16. decode — increments total_decoded
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_increments_stat() {
        let mut vq = trained_vq(16, 4, 4, 20);
        let code = vq
            .encode(&[0.5f64; 16])
            .expect("test: encode 0.5 vector should succeed");
        let before = vq.stats.total_decoded;
        vq.decode(&code)
            .expect("test: decode of valid code should succeed");
        assert_eq!(vq.stats.total_decoded, before + 1);
    }

    // -----------------------------------------------------------------------
    // 17. encode + decode round-trip: dimension preserved
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_decode_round_trip_dim() {
        let dim = 32;
        let mut vq = trained_vq(dim, 4, 4, 20);
        let vec = make_vectors(1, dim).remove(0);
        let code = vq
            .encode(&vec)
            .expect("test: encode of valid vector should succeed");
        let decoded = vq
            .decode(&code)
            .expect("test: decode of valid code should succeed");
        assert_eq!(decoded.len(), dim);
    }

    // -----------------------------------------------------------------------
    // 18. encode_batch — returns same count as input
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_batch_count() {
        let dim = 16;
        let mut vq = trained_vq(dim, 4, 4, 20);
        let vecs = make_vectors(5, dim);
        let codes = vq
            .encode_batch(&vecs)
            .expect("test: encode_batch of valid vectors should succeed");
        assert_eq!(codes.len(), 5);
    }

    // -----------------------------------------------------------------------
    // 19. encode_batch — fails if any vector has wrong dimension
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_batch_dimension_error() {
        let dim = 16;
        let mut vq = trained_vq(dim, 4, 4, 20);
        let mut vecs = make_vectors(3, dim);
        vecs.push(vec![0.0f64; 8]); // wrong dimension
        let result = vq.encode_batch(&vecs);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // 20. asymmetric_distance — self distance is zero
    // -----------------------------------------------------------------------

    #[test]
    fn test_asymmetric_distance_self_zero() {
        let dim = 16;
        let mut vq = trained_vq(dim, 4, 4, 20);
        let vec = vec![0.5f64; dim];
        let code = vq
            .encode(&vec)
            .expect("test: encode 0.5 vector should succeed");
        // Decode the code to get the reconstructed vector, then measure asymmetric distance.
        let decoded = vq
            .decode(&code)
            .expect("test: decode of valid code should succeed");
        let dist = vq
            .asymmetric_distance(&decoded, &code)
            .expect("test: asymmetric_distance to self should succeed");
        assert!(
            dist < 1e-10,
            "asymmetric distance to self should be ~0, got {dist}"
        );
    }

    // -----------------------------------------------------------------------
    // 21. asymmetric_distance — fails when not trained
    // -----------------------------------------------------------------------

    #[test]
    fn test_asymmetric_distance_not_trained() {
        let vq = VectorQuantizer::new(small_config(4, 4));
        let code = QuantizerCode(vec![0u8; 4]);
        let result = vq.asymmetric_distance(&[0.0f64; 16], &code);
        assert!(matches!(result, Err(VqError::NotTrained)));
    }

    // -----------------------------------------------------------------------
    // 22. asymmetric_distance — dimension mismatch error
    // -----------------------------------------------------------------------

    #[test]
    fn test_asymmetric_distance_dimension_mismatch() {
        let dim = 16;
        let mut vq = trained_vq(dim, 4, 4, 20);
        let code = vq
            .encode(&vec![0.5f64; dim])
            .expect("test: encode 0.5 vector should succeed");
        let result = vq.asymmetric_distance(&[0.0f64; 8], &code);
        assert!(matches!(result, Err(VqError::DimensionMismatch { .. })));
    }

    // -----------------------------------------------------------------------
    // 23. symmetric_distance — self distance is zero
    // -----------------------------------------------------------------------

    #[test]
    fn test_symmetric_distance_self_zero() {
        let dim = 16;
        let mut vq = trained_vq(dim, 4, 4, 20);
        let vec = vec![0.5f64; dim];
        let code = vq
            .encode(&vec)
            .expect("test: encode 0.5 vector should succeed");
        let dist = vq
            .symmetric_distance(&code.clone(), &code)
            .expect("test: symmetric_distance to self should succeed");
        assert!(
            dist < 1e-10,
            "symmetric distance to self should be 0, got {dist}"
        );
    }

    // -----------------------------------------------------------------------
    // 24. symmetric_distance — is symmetric (a,b) == (b,a)
    // -----------------------------------------------------------------------

    #[test]
    fn test_symmetric_distance_is_symmetric() {
        let dim = 16;
        let mut vq = trained_vq(dim, 4, 4, 20);
        let code_a = vq
            .encode(&vec![0.1f64; dim])
            .expect("test: encode 0.1 vector should succeed");
        let code_b = vq
            .encode(&vec![0.9f64; dim])
            .expect("test: encode 0.9 vector should succeed");
        let dist_ab = vq
            .symmetric_distance(&code_a, &code_b)
            .expect("test: symmetric_distance(a,b) should succeed");
        let dist_ba = vq
            .symmetric_distance(&code_b, &code_a)
            .expect("test: symmetric_distance(b,a) should succeed");
        assert!(
            (dist_ab - dist_ba).abs() < 1e-10,
            "distance must be symmetric"
        );
    }

    // -----------------------------------------------------------------------
    // 25. quantization_error — is non-negative
    // -----------------------------------------------------------------------

    #[test]
    fn test_quantization_error_non_negative() {
        let dim = 16;
        let mut vq = trained_vq(dim, 4, 4, 20);
        let err = vq
            .quantization_error(&vec![0.5f64; dim])
            .expect("test: quantization_error should succeed on trained vq");
        assert!(
            err >= 0.0,
            "quantization error must be non-negative, got {err}"
        );
    }

    // -----------------------------------------------------------------------
    // 26. quantization_error — exact match gives zero error
    // -----------------------------------------------------------------------

    #[test]
    fn test_quantization_error_centroid_is_zero() {
        // Build a VQ manually where the single centroid equals our query.
        let cfg = QuantizationConfig::new(1, 1, 10, 1e-10);
        let mut vq = VectorQuantizer::new(cfg);
        // One training vector, one subspace, one centroid → centroid == training vector.
        let query = vec![1.0f64, 2.0, 3.0, 4.0];
        vq.train(std::slice::from_ref(&query))
            .expect("test: training single-vector single-subspace should succeed");
        let err = vq
            .quantization_error(&query)
            .expect("test: quantization_error on exact centroid match should succeed");
        assert!(
            err < 1e-10,
            "error should be ~0 for exact centroid match, got {err}"
        );
    }

    // -----------------------------------------------------------------------
    // 27. avg_error_on_batch — empty batch returns 0
    // -----------------------------------------------------------------------

    #[test]
    fn test_avg_error_empty_batch() {
        let mut vq = trained_vq(16, 4, 4, 20);
        let result = vq
            .avg_error_on_batch(&[])
            .expect("test: avg_error_on_batch of empty slice should return Ok(0.0)");
        assert_eq!(result, 0.0);
    }

    // -----------------------------------------------------------------------
    // 28. avg_error_on_batch — single vector matches quantization_error
    // -----------------------------------------------------------------------

    #[test]
    fn test_avg_error_single_vector() {
        let dim = 16;
        let vec = vec![0.5f64; dim];
        // Use two fresh quantizers trained on the same data for a fair comparison.
        let cfg = small_config(4, 4);
        let mut vq2 = VectorQuantizer::new(cfg);
        let data = make_vectors(20, dim);
        vq2.train(&data).expect("test: training vq2 should succeed");
        let single_err = vq2
            .quantization_error(&vec)
            .expect("test: quantization_error on vq2 should succeed");

        let cfg2 = small_config(4, 4);
        let mut vq3 = VectorQuantizer::new(cfg2);
        vq3.train(&data).expect("test: training vq3 should succeed");
        let batch_err = vq3
            .avg_error_on_batch(&[vec])
            .expect("test: avg_error_on_batch on vq3 should succeed");

        assert!(
            (single_err - batch_err).abs() < 1e-10,
            "avg error of one vector should equal its individual error"
        );
    }

    // -----------------------------------------------------------------------
    // 29. codebook_stats — returns correct count
    // -----------------------------------------------------------------------

    #[test]
    fn test_codebook_stats_length() {
        let m = 4;
        let vq = trained_vq(16, m, 4, 20);
        let stats = vq.codebook_stats();
        assert_eq!(stats.len(), m);
    }

    // -----------------------------------------------------------------------
    // 30. codebook_stats — subspace indices are 0..m-1
    // -----------------------------------------------------------------------

    #[test]
    fn test_codebook_stats_subspace_indices() {
        let m = 4;
        let vq = trained_vq(16, m, 4, 20);
        let stats = vq.codebook_stats();
        for (i, (subspace_idx, _)) in stats.iter().enumerate() {
            assert_eq!(*subspace_idx, i);
        }
    }

    // -----------------------------------------------------------------------
    // 31. codebook_stats — centroid count bounded by codes_per_subspace
    // -----------------------------------------------------------------------

    #[test]
    fn test_codebook_stats_centroid_count() {
        let codes: u8 = 4;
        let vq = trained_vq(16, 4, codes, 20);
        let stats = vq.codebook_stats();
        for (_, num_centroids) in &stats {
            assert!(*num_centroids <= codes as usize);
        }
    }

    // -----------------------------------------------------------------------
    // 32. QuantizerCode::is_empty
    // -----------------------------------------------------------------------

    #[test]
    fn test_quantizer_code_is_empty() {
        let empty = QuantizerCode(vec![]);
        let non_empty = QuantizerCode(vec![0u8]);
        assert!(empty.is_empty());
        assert!(!non_empty.is_empty());
    }

    // -----------------------------------------------------------------------
    // 33. QuantizerCode — clone and PartialEq
    // -----------------------------------------------------------------------

    #[test]
    fn test_quantizer_code_clone_and_eq() {
        let code = QuantizerCode(vec![1u8, 2, 3]);
        let cloned = code.clone();
        assert_eq!(code, cloned);
        let different = QuantizerCode(vec![1u8, 2, 4]);
        assert_ne!(code, different);
    }

    // -----------------------------------------------------------------------
    // 34. VectorQuantizer::debug format
    // -----------------------------------------------------------------------

    #[test]
    fn test_vector_quantizer_debug_format() {
        let vq = VectorQuantizer::new(small_config(4, 4));
        let dbg = format!("{vq:?}");
        assert!(dbg.contains("VectorQuantizer"));
    }

    // -----------------------------------------------------------------------
    // 35. QuantizationStats default values
    // -----------------------------------------------------------------------

    #[test]
    fn test_quantization_stats_default() {
        let stats = QuantizationStats::default();
        assert_eq!(stats.codebooks_trained, 0);
        assert_eq!(stats.total_encoded, 0);
        assert_eq!(stats.total_decoded, 0);
        assert_eq!(stats.avg_encode_error, 0.0);
    }

    // -----------------------------------------------------------------------
    // 36. encode well-separated clusters — nearest gets assigned correctly
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_cluster_assignment() {
        let dim = 8;
        // Two clearly separated clusters: zeros and ones.
        let mut data: Vec<Vec<f64>> = Vec::new();
        for _ in 0..5 {
            data.push(vec![0.0f64; dim]);
        }
        for _ in 0..5 {
            data.push(vec![100.0f64; dim]);
        }
        let cfg = QuantizationConfig::new(2, 2, 50, 1e-8);
        let mut vq = VectorQuantizer::new(cfg);
        vq.train(&data)
            .expect("test: training on two-cluster data should succeed");

        let code_near_zero = vq
            .encode(&vec![0.01f64; dim])
            .expect("test: encode of near-zero vector should succeed");
        let code_near_hundred = vq
            .encode(&vec![99.99f64; dim])
            .expect("test: encode of near-hundred vector should succeed");

        // The two queries should land in different codes.
        assert_ne!(
            code_near_zero, code_near_hundred,
            "well-separated vectors should get different codes"
        );
    }

    // -----------------------------------------------------------------------
    // 37. asymmetric_distance — closer vector gives smaller distance
    // -----------------------------------------------------------------------

    #[test]
    fn test_asymmetric_distance_ordering() {
        let dim = 8;
        let mut data: Vec<Vec<f64>> = Vec::new();
        for i in 0..5 {
            data.push(vec![i as f64; dim]);
        }
        let cfg = QuantizationConfig::new(2, 2, 50, 1e-8);
        let mut vq = VectorQuantizer::new(cfg);
        vq.train(&data)
            .expect("test: training on linear-spaced data should succeed");

        let query = vec![0.0f64; dim];
        let code_near = vq
            .encode(&vec![0.5f64; dim])
            .expect("test: encode near vector should succeed");
        let code_far = vq
            .encode(&vec![4.5f64; dim])
            .expect("test: encode far vector should succeed");

        let dist_near = vq
            .asymmetric_distance(&query, &code_near)
            .expect("test: asymmetric_distance to near code should succeed");
        let dist_far = vq
            .asymmetric_distance(&query, &code_far)
            .expect("test: asymmetric_distance to far code should succeed");

        assert!(
            dist_near <= dist_far,
            "closer code should have smaller asymmetric distance: near={dist_near}, far={dist_far}"
        );
    }

    // -----------------------------------------------------------------------
    // 38. Codebook::nearest_centroid — single centroid always returns 0
    // -----------------------------------------------------------------------

    #[test]
    fn test_codebook_nearest_centroid_single() {
        let cb = Codebook {
            centroids: vec![vec![1.0f64, 2.0, 3.0]],
            subspace_dim: 3,
            num_codes: 1,
        };
        assert_eq!(cb.nearest_centroid(&[0.0, 0.0, 0.0]), 0);
        assert_eq!(cb.nearest_centroid(&[10.0, 10.0, 10.0]), 0);
    }

    // -----------------------------------------------------------------------
    // 39. VqError variants have informative messages
    // -----------------------------------------------------------------------

    #[test]
    fn test_vq_error_messages() {
        let e1 = VqError::NotTrained;
        assert!(!format!("{e1}").is_empty());

        let e2 = VqError::DimensionMismatch {
            expected: 16,
            got: 8,
        };
        let msg = format!("{e2}");
        assert!(msg.contains("16") && msg.contains("8"));

        let e3 = VqError::InsufficientData {
            needed: 256,
            got: 10,
        };
        let msg3 = format!("{e3}");
        assert!(msg3.contains("256") && msg3.contains("10"));

        let e4 = VqError::InvalidCode("bad code".to_string());
        assert!(format!("{e4}").contains("bad code"));
    }

    // -----------------------------------------------------------------------
    // 40. avg_encode_error updates as a running mean (non-negative)
    // -----------------------------------------------------------------------

    #[test]
    fn test_avg_encode_error_running_mean() {
        let dim = 16;
        let mut vq = trained_vq(dim, 4, 4, 20);

        vq.encode(&vec![0.0f64; dim])
            .expect("test: encode 0.0 vector should succeed");
        let e1 = vq.stats.avg_encode_error;
        vq.encode(&vec![0.5f64; dim])
            .expect("test: encode 0.5 vector should succeed");
        let e2 = vq.stats.avg_encode_error;
        vq.encode(&vec![1.0f64; dim])
            .expect("test: encode 1.0 vector should succeed");

        // All errors must be non-negative.
        assert!(e1 >= 0.0);
        assert!(e2 >= 0.0);
    }

    // -----------------------------------------------------------------------
    // 41. encode_batch — all codes have correct length
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_batch_code_lengths() {
        let dim = 16;
        let m = 4;
        let mut vq = trained_vq(dim, m, 4, 20);
        let vecs = make_vectors(8, dim);
        let codes = vq
            .encode_batch(&vecs)
            .expect("test: encode_batch for code lengths check");
        for code in &codes {
            assert_eq!(code.len(), m);
        }
    }

    // -----------------------------------------------------------------------
    // 42. decode with out-of-range code returns InvalidCode
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_out_of_range_code() {
        let dim = 4;
        let cfg = QuantizationConfig::new(1, 2, 10, 1e-6); // only 2 centroids (codes 0,1)
        let mut vq = VectorQuantizer::new(cfg);
        // Need >= 2 training vectors.
        let data = vec![vec![0.0f64; dim], vec![1.0f64; dim]];
        vq.train(&data)
            .expect("test: train for out-of-range code decode test");

        // Code=200 is out of range since codebook only has ≤2 entries.
        let code = QuantizerCode(vec![200u8]);
        let result = vq.decode(&code);
        assert!(matches!(result, Err(VqError::InvalidCode(_))));
    }
}
