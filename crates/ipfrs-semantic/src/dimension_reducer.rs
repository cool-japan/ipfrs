//! # Semantic Dimension Reducer
//!
//! Dimensionality reduction for embeddings using random projection, PCA, or truncation.
//!
//! Provides [`SemanticDimensionReducer`] which can reduce high-dimensional embeddings
//! to lower dimensions while preserving semantic structure (Johnson-Lindenstrauss property
//! for random projection).

/// Reduction method to use for dimensionality reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionMethod {
    /// Gaussian random matrix projection (Johnson-Lindenstrauss)
    RandomProjection,
    /// Simplified PCA via power iteration
    PCA,
    /// Simply take the first N dimensions
    Truncation,
}

/// Configuration for the dimension reducer.
#[derive(Debug, Clone)]
pub struct ReducerConfig {
    /// Dimensionality of input embeddings
    pub input_dim: usize,
    /// Target dimensionality after reduction
    pub output_dim: usize,
    /// Method to use for reduction
    pub method: ReductionMethod,
    /// Seed for reproducible random projection
    pub seed: u64,
}

/// Result metadata from a reduction operation.
#[derive(Debug, Clone)]
pub struct ReductionResult {
    /// Original embedding dimensionality
    pub original_dim: usize,
    /// Reduced embedding dimensionality
    pub reduced_dim: usize,
    /// Reconstruction error (MSE), if computed
    pub reconstruction_error: Option<f64>,
}

/// Statistics about the reducer state.
#[derive(Debug, Clone)]
pub struct ReducerStats {
    /// Input dimensionality
    pub input_dim: usize,
    /// Output dimensionality
    pub output_dim: usize,
    /// Reduction method in use
    pub method: ReductionMethod,
    /// Whether the reducer has been fitted
    pub fitted: bool,
    /// Number of reductions performed
    pub reductions_performed: u64,
}

/// Semantic dimension reducer supporting random projection, PCA, and truncation.
pub struct SemanticDimensionReducer {
    config: ReducerConfig,
    /// Projection matrix: output_dim x input_dim
    projection_matrix: Option<Vec<Vec<f64>>>,
    fitted: bool,
    reductions_performed: u64,
}

/// FNV-1a based PRNG for deterministic random number generation from a seed.
struct FnvPrng {
    state: u64,
}

impl FnvPrng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0xcbf29ce484222325,
        }
    }

    /// Generate next u64 using FNV-1a mixing.
    fn next_u64(&mut self) -> u64 {
        // FNV-1a round
        self.state ^= self.state.wrapping_shr(13);
        self.state = self.state.wrapping_mul(0x100000001b3);
        self.state ^= self.state.wrapping_shr(7);
        self.state = self.state.wrapping_mul(0x100000001b3);
        self.state ^= self.state.wrapping_shr(17);
        self.state
    }

    /// Generate a pseudo-Gaussian value using Box-Muller approximation.
    /// Returns a value approximately distributed as N(0, 1).
    fn next_gaussian(&mut self) -> f64 {
        // Use inverse transform approximation: average of 12 uniform values - 6
        // (central limit theorem approximation)
        let mut sum = 0.0f64;
        for _ in 0..12 {
            let u = (self.next_u64() as f64) / (u64::MAX as f64);
            sum += u;
        }
        sum - 6.0
    }
}

impl SemanticDimensionReducer {
    /// Create a new dimension reducer with the given configuration.
    pub fn new(config: ReducerConfig) -> Self {
        Self {
            config,
            projection_matrix: None,
            fitted: false,
            reductions_performed: 0,
        }
    }

    /// Fit the reducer to the given embeddings.
    ///
    /// - `RandomProjection`: generates a Gaussian random matrix from the seed
    /// - `PCA`: computes top eigenvectors via power iteration
    /// - `Truncation`: no-op (always ready)
    pub fn fit(&mut self, embeddings: &[Vec<f64>]) -> Result<(), String> {
        if self.config.output_dim > self.config.input_dim {
            return Err(format!(
                "output_dim ({}) must be <= input_dim ({})",
                self.config.output_dim, self.config.input_dim
            ));
        }

        // Validate embeddings dimensions
        for (i, emb) in embeddings.iter().enumerate() {
            if emb.len() != self.config.input_dim {
                return Err(format!(
                    "embedding at index {} has dimension {} but expected {}",
                    i,
                    emb.len(),
                    self.config.input_dim
                ));
            }
        }

        match self.config.method {
            ReductionMethod::RandomProjection => {
                self.fit_random_projection()?;
            }
            ReductionMethod::PCA => {
                self.fit_pca(embeddings)?;
            }
            ReductionMethod::Truncation => {
                // Truncation needs no fitting — just take first output_dim dimensions
            }
        }

        self.fitted = true;
        Ok(())
    }

    /// Generate the Gaussian random projection matrix from the configured seed.
    fn fit_random_projection(&mut self) -> Result<(), String> {
        let input_dim = self.config.input_dim;
        let output_dim = self.config.output_dim;
        let mut prng = FnvPrng::new(self.config.seed);

        // Generate output_dim x input_dim matrix with Gaussian entries
        let mut matrix = Vec::with_capacity(output_dim);
        for _ in 0..output_dim {
            let mut row = Vec::with_capacity(input_dim);
            for _ in 0..input_dim {
                row.push(prng.next_gaussian());
            }
            matrix.push(row);
        }

        // Normalize columns for better numerical properties
        self.normalize_columns(&mut matrix);

        self.projection_matrix = Some(matrix);
        Ok(())
    }

    /// Normalize columns of the matrix to unit length.
    fn normalize_columns(&self, matrix: &mut [Vec<f64>]) {
        if matrix.is_empty() {
            return;
        }
        let input_dim = matrix[0].len();
        let output_dim = matrix.len();

        for col in 0..input_dim {
            let mut norm_sq = 0.0f64;
            for row in matrix.iter().take(output_dim) {
                norm_sq += row[col] * row[col];
            }
            let norm = norm_sq.sqrt();
            if norm > 1e-15 {
                for row in matrix.iter_mut().take(output_dim) {
                    row[col] /= norm;
                }
            }
        }
    }

    /// Fit PCA using power iteration to find top eigenvectors of the covariance matrix.
    fn fit_pca(&mut self, embeddings: &[Vec<f64>]) -> Result<(), String> {
        if embeddings.is_empty() {
            return Err("cannot fit PCA with zero embeddings".to_string());
        }

        let n = embeddings.len();
        let d = self.config.input_dim;
        let k = self.config.output_dim;

        // Compute mean
        let mut mean = vec![0.0f64; d];
        for emb in embeddings {
            for (j, val) in emb.iter().enumerate() {
                mean[j] += val;
            }
        }
        let n_f64 = n as f64;
        for m in &mut mean {
            *m /= n_f64;
        }

        // Center the data
        let centered: Vec<Vec<f64>> = embeddings
            .iter()
            .map(|emb| emb.iter().zip(mean.iter()).map(|(v, m)| v - m).collect())
            .collect();

        // Power iteration to find top-k eigenvectors of X^T X / n
        let mut prng = FnvPrng::new(self.config.seed);
        let mut components: Vec<Vec<f64>> = Vec::with_capacity(k);
        let max_iterations = 100;

        for comp_idx in 0..k {
            // Initialize random vector
            let mut v: Vec<f64> = (0..d).map(|_| prng.next_gaussian()).collect();
            let mut v_norm = vec_norm(&v);
            if v_norm > 1e-15 {
                for val in &mut v {
                    *val /= v_norm;
                }
            }

            for _iter in 0..max_iterations {
                // Compute X^T * (X * v) / n  (covariance times v)
                // First: proj_i = centered[i] . v
                let projections: Vec<f64> = centered.iter().map(|row| dot(row, &v)).collect();

                // Then: new_v = sum_i (proj_i * centered[i]) / n
                let mut new_v = vec![0.0f64; d];
                for (i, proj) in projections.iter().enumerate() {
                    for (j, val) in centered[i].iter().enumerate() {
                        new_v[j] += proj * val;
                    }
                }
                for val in &mut new_v {
                    *val /= n_f64;
                }

                // Deflate: remove components from previously found eigenvectors
                for prev in &components {
                    let proj = dot(&new_v, prev);
                    for (j, val) in new_v.iter_mut().enumerate() {
                        *val -= proj * prev[j];
                    }
                }

                // Normalize
                v_norm = vec_norm(&new_v);
                if v_norm < 1e-15 {
                    // Degenerate case — use random direction
                    for val in new_v.iter_mut() {
                        *val = prng.next_gaussian();
                    }
                    v_norm = vec_norm(&new_v);
                    if v_norm > 1e-15 {
                        for val in &mut new_v {
                            *val /= v_norm;
                        }
                    }
                } else {
                    for val in &mut new_v {
                        *val /= v_norm;
                    }
                }

                // Check convergence
                let diff: f64 = v
                    .iter()
                    .zip(new_v.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                v = new_v;
                if diff < 1e-10 {
                    break;
                }
            }

            components.push(v);
            let _ = comp_idx; // suppress unused warning
        }

        // Build projection matrix: k x d (each row is an eigenvector)
        self.projection_matrix = Some(components);
        Ok(())
    }

    /// Transform a single embedding to lower dimensionality.
    pub fn transform(&mut self, embedding: &[f64]) -> Result<Vec<f64>, String> {
        if !self.fitted {
            return Err("reducer has not been fitted yet".to_string());
        }

        if embedding.len() != self.config.input_dim {
            return Err(format!(
                "input dimension mismatch: expected {}, got {}",
                self.config.input_dim,
                embedding.len()
            ));
        }

        let result = match self.config.method {
            ReductionMethod::Truncation => embedding[..self.config.output_dim].to_vec(),
            ReductionMethod::RandomProjection | ReductionMethod::PCA => {
                let matrix = self
                    .projection_matrix
                    .as_ref()
                    .ok_or_else(|| "projection matrix not initialized".to_string())?;
                let mut out = Vec::with_capacity(self.config.output_dim);
                for row in matrix {
                    out.push(dot(row, embedding));
                }
                out
            }
        };

        self.reductions_performed += 1;
        Ok(result)
    }

    /// Fit the reducer and transform all embeddings in one step.
    pub fn fit_transform(&mut self, embeddings: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        self.fit(embeddings)?;
        let mut results = Vec::with_capacity(embeddings.len());
        for emb in embeddings {
            results.push(self.transform(emb)?);
        }
        Ok(results)
    }

    /// Compute the reconstruction error (MSE) between the original and back-projected embedding.
    ///
    /// For `RandomProjection`, uses pseudo-inverse (transpose for column-normalized matrices).
    /// For `PCA`, uses transpose of components.
    /// For `Truncation`, pads with zeros.
    pub fn reconstruction_error(&self, original: &[f64], reduced: &[f64]) -> f64 {
        let reconstructed = match self.config.method {
            ReductionMethod::Truncation => {
                let mut r = reduced.to_vec();
                r.resize(self.config.input_dim, 0.0);
                r
            }
            ReductionMethod::RandomProjection | ReductionMethod::PCA => {
                // Back-project using transpose of projection matrix
                if let Some(matrix) = &self.projection_matrix {
                    let mut r = vec![0.0f64; self.config.input_dim];
                    for (i, row) in matrix.iter().enumerate() {
                        if i < reduced.len() {
                            for (j, &val) in row.iter().enumerate() {
                                r[j] += reduced[i] * val;
                            }
                        }
                    }
                    r
                } else {
                    vec![0.0f64; self.config.input_dim]
                }
            }
        };

        // MSE
        let n = original.len().min(reconstructed.len());
        if n == 0 {
            return 0.0;
        }
        let mse: f64 = original
            .iter()
            .take(n)
            .zip(reconstructed.iter().take(n))
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / n as f64;
        mse
    }

    /// Whether the reducer has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Reset the reducer, clearing the projection matrix and fitted state.
    pub fn reset(&mut self) {
        self.projection_matrix = None;
        self.fitted = false;
        self.reductions_performed = 0;
    }

    /// Get statistics about the reducer.
    pub fn stats(&self) -> ReducerStats {
        ReducerStats {
            input_dim: self.config.input_dim,
            output_dim: self.config.output_dim,
            method: self.config.method,
            fitted: self.fitted,
            reductions_performed: self.reductions_performed,
        }
    }
}

/// Compute dot product of two slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute the L2 norm of a vector.
fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(
        input_dim: usize,
        output_dim: usize,
        method: ReductionMethod,
        seed: u64,
    ) -> ReducerConfig {
        ReducerConfig {
            input_dim,
            output_dim,
            method,
            seed,
        }
    }

    // --- Random Projection Tests ---

    #[test]
    fn test_random_projection_reduces_dim() {
        let config = make_config(100, 10, ReductionMethod::RandomProjection, 42);
        let mut reducer = SemanticDimensionReducer::new(config);
        let embeddings = vec![vec![1.0; 100]; 5];
        reducer.fit(&embeddings).expect("fit should succeed");
        let result = reducer
            .transform(&embeddings[0])
            .expect("transform should succeed");
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_random_projection_deterministic_same_seed() {
        let config1 = make_config(50, 10, ReductionMethod::RandomProjection, 123);
        let config2 = make_config(50, 10, ReductionMethod::RandomProjection, 123);
        let mut r1 = SemanticDimensionReducer::new(config1);
        let mut r2 = SemanticDimensionReducer::new(config2);
        let embeddings = vec![vec![0.5; 50]; 3];
        r1.fit(&embeddings).expect("fit should succeed");
        r2.fit(&embeddings).expect("fit should succeed");
        let t1 = r1.transform(&embeddings[0]).expect("transform");
        let t2 = r2.transform(&embeddings[0]).expect("transform");
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_random_projection_different_seeds_differ() {
        let config1 = make_config(50, 10, ReductionMethod::RandomProjection, 100);
        let config2 = make_config(50, 10, ReductionMethod::RandomProjection, 200);
        let mut r1 = SemanticDimensionReducer::new(config1);
        let mut r2 = SemanticDimensionReducer::new(config2);
        let embeddings = vec![vec![0.5; 50]; 3];
        r1.fit(&embeddings).expect("fit");
        r2.fit(&embeddings).expect("fit");
        let t1 = r1.transform(&embeddings[0]).expect("transform");
        let t2 = r2.transform(&embeddings[0]).expect("transform");
        assert_ne!(t1, t2);
    }

    #[test]
    fn test_random_projection_reconstruction_error() {
        let config = make_config(20, 15, ReductionMethod::RandomProjection, 42);
        let mut reducer = SemanticDimensionReducer::new(config);
        let embedding = (0..20).map(|i| i as f64 * 0.1).collect::<Vec<_>>();
        let embeddings = vec![embedding.clone()];
        reducer.fit(&embeddings).expect("fit");
        let reduced = reducer.transform(&embedding).expect("transform");
        let error = reducer.reconstruction_error(&embedding, &reduced);
        // Error should be finite and non-negative
        assert!(error >= 0.0);
        assert!(error.is_finite());
    }

    // --- Truncation Tests ---

    #[test]
    fn test_truncation_takes_first_n() {
        let config = make_config(10, 5, ReductionMethod::Truncation, 0);
        let mut reducer = SemanticDimensionReducer::new(config);
        let embedding: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let embeddings = vec![embedding.clone()];
        reducer.fit(&embeddings).expect("fit");
        let result = reducer.transform(&embedding).expect("transform");
        assert_eq!(result, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_truncation_reconstruction_error() {
        let config = make_config(10, 5, ReductionMethod::Truncation, 0);
        let mut reducer = SemanticDimensionReducer::new(config);
        let embedding: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let embeddings = vec![embedding.clone()];
        reducer.fit(&embeddings).expect("fit");
        let reduced = reducer.transform(&embedding).expect("transform");
        let error = reducer.reconstruction_error(&embedding, &reduced);
        // Error from truncation: last 5 dims are lost (5,6,7,8,9)
        // MSE = (25+36+49+64+81) / 10 = 255/10 = 25.5
        assert!((error - 25.5).abs() < 1e-10);
    }

    // --- PCA Tests ---

    #[test]
    fn test_pca_reduces_dim() {
        let config = make_config(10, 3, ReductionMethod::PCA, 42);
        let mut reducer = SemanticDimensionReducer::new(config);
        // Create data with clear variance structure
        let mut embeddings = Vec::new();
        for i in 0..20 {
            let mut emb = vec![0.0; 10];
            emb[0] = i as f64 * 10.0; // high variance dimension
            emb[1] = i as f64 * 5.0; // medium variance
            emb[2] = i as f64 * 1.0; // lower variance
            for (j, val) in emb.iter_mut().enumerate().skip(3) {
                *val = 0.01 * (i as f64 + j as f64);
            }
            embeddings.push(emb);
        }
        reducer.fit(&embeddings).expect("fit");
        let result = reducer.transform(&embeddings[0]).expect("transform");
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_pca_preserves_variance_direction() {
        let config = make_config(5, 1, ReductionMethod::PCA, 42);
        let mut reducer = SemanticDimensionReducer::new(config);
        // All variance along first dimension
        let embeddings: Vec<Vec<f64>> = (0..50)
            .map(|i| {
                let mut v = vec![0.0; 5];
                v[0] = i as f64;
                v
            })
            .collect();
        reducer.fit(&embeddings).expect("fit");

        // Two points differing only in the high-variance dim should be well separated
        let t1 = reducer.transform(&embeddings[0]).expect("transform");
        let t2 = reducer.transform(&embeddings[49]).expect("transform");
        let separation = (t1[0] - t2[0]).abs();
        assert!(
            separation > 1.0,
            "PCA should preserve main variance direction, got separation {separation}"
        );
    }

    #[test]
    fn test_pca_empty_embeddings_error() {
        let config = make_config(10, 3, ReductionMethod::PCA, 42);
        let mut reducer = SemanticDimensionReducer::new(config);
        let result = reducer.fit(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_pca_reconstruction_error() {
        let config = make_config(10, 5, ReductionMethod::PCA, 42);
        let mut reducer = SemanticDimensionReducer::new(config);
        let embeddings: Vec<Vec<f64>> = (0..30)
            .map(|i| (0..10).map(|j| (i * 10 + j) as f64 * 0.01).collect())
            .collect();
        reducer.fit(&embeddings).expect("fit");
        let reduced = reducer.transform(&embeddings[0]).expect("transform");
        let error = reducer.reconstruction_error(&embeddings[0], &reduced);
        assert!(error >= 0.0);
        assert!(error.is_finite());
    }

    // --- Error Cases ---

    #[test]
    fn test_transform_error_if_not_fitted() {
        let config = make_config(10, 5, ReductionMethod::RandomProjection, 42);
        let mut reducer = SemanticDimensionReducer::new(config);
        let result = reducer.transform(&[1.0; 10]);
        assert!(result.is_err());
        assert!(result
            .expect_err("should error")
            .contains("not been fitted"));
    }

    #[test]
    fn test_transform_error_wrong_input_dim() {
        let config = make_config(10, 5, ReductionMethod::RandomProjection, 42);
        let mut reducer = SemanticDimensionReducer::new(config);
        let embeddings = vec![vec![1.0; 10]; 3];
        reducer.fit(&embeddings).expect("fit");
        let result = reducer.transform(&[1.0; 7]);
        assert!(result.is_err());
        assert!(result
            .expect_err("should error")
            .contains("dimension mismatch"));
    }

    #[test]
    fn test_fit_error_output_dim_greater_than_input_dim() {
        let config = make_config(5, 10, ReductionMethod::RandomProjection, 42);
        let mut reducer = SemanticDimensionReducer::new(config);
        let result = reducer.fit(&[vec![1.0; 5]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_fit_error_wrong_embedding_dim() {
        let config = make_config(10, 5, ReductionMethod::RandomProjection, 42);
        let mut reducer = SemanticDimensionReducer::new(config);
        let result = reducer.fit(&[vec![1.0; 7]]);
        assert!(result.is_err());
        assert!(result.expect_err("should error").contains("dimension"));
    }

    // --- fit_transform ---

    #[test]
    fn test_fit_transform_random_projection() {
        let config = make_config(20, 5, ReductionMethod::RandomProjection, 42);
        let mut reducer = SemanticDimensionReducer::new(config);
        let embeddings: Vec<Vec<f64>> = (0..10)
            .map(|i| (0..20).map(|j| (i * 20 + j) as f64 * 0.01).collect())
            .collect();
        let results = reducer.fit_transform(&embeddings).expect("fit_transform");
        assert_eq!(results.len(), 10);
        for r in &results {
            assert_eq!(r.len(), 5);
        }
        assert!(reducer.is_fitted());
    }

    #[test]
    fn test_fit_transform_truncation() {
        let config = make_config(8, 3, ReductionMethod::Truncation, 0);
        let mut reducer = SemanticDimensionReducer::new(config);
        let embeddings = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        ];
        let results = reducer.fit_transform(&embeddings).expect("fit_transform");
        assert_eq!(results[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(results[1], vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_fit_transform_pca() {
        let config = make_config(6, 2, ReductionMethod::PCA, 99);
        let mut reducer = SemanticDimensionReducer::new(config);
        let embeddings: Vec<Vec<f64>> = (0..20)
            .map(|i| {
                let mut v = vec![0.0; 6];
                v[0] = i as f64 * 3.0;
                v[1] = i as f64 * 2.0;
                v[2] = i as f64 * 0.1;
                v
            })
            .collect();
        let results = reducer.fit_transform(&embeddings).expect("fit_transform");
        assert_eq!(results.len(), 20);
        for r in &results {
            assert_eq!(r.len(), 2);
        }
    }

    // --- State management ---

    #[test]
    fn test_is_fitted_initially_false() {
        let config = make_config(10, 5, ReductionMethod::RandomProjection, 42);
        let reducer = SemanticDimensionReducer::new(config);
        assert!(!reducer.is_fitted());
    }

    #[test]
    fn test_reset_clears_state() {
        let config = make_config(10, 5, ReductionMethod::RandomProjection, 42);
        let mut reducer = SemanticDimensionReducer::new(config);
        let embeddings = vec![vec![1.0; 10]; 5];
        reducer.fit(&embeddings).expect("fit");
        let _ = reducer.transform(&embeddings[0]);
        assert!(reducer.is_fitted());
        assert!(reducer.stats().reductions_performed > 0);

        reducer.reset();

        assert!(!reducer.is_fitted());
        assert_eq!(reducer.stats().reductions_performed, 0);
        assert!(reducer.transform(&embeddings[0]).is_err());
    }

    #[test]
    fn test_stats_accuracy() {
        let config = make_config(20, 8, ReductionMethod::RandomProjection, 55);
        let mut reducer = SemanticDimensionReducer::new(config);
        let stats = reducer.stats();
        assert_eq!(stats.input_dim, 20);
        assert_eq!(stats.output_dim, 8);
        assert_eq!(stats.method, ReductionMethod::RandomProjection);
        assert!(!stats.fitted);
        assert_eq!(stats.reductions_performed, 0);

        let embeddings = vec![vec![1.0; 20]; 3];
        reducer.fit(&embeddings).expect("fit");
        let _ = reducer.transform(&embeddings[0]);
        let _ = reducer.transform(&embeddings[1]);

        let stats = reducer.stats();
        assert!(stats.fitted);
        assert_eq!(stats.reductions_performed, 2);
    }

    #[test]
    fn test_stats_method_truncation() {
        let config = make_config(10, 5, ReductionMethod::Truncation, 0);
        let reducer = SemanticDimensionReducer::new(config);
        assert_eq!(reducer.stats().method, ReductionMethod::Truncation);
    }

    #[test]
    fn test_stats_method_pca() {
        let config = make_config(10, 5, ReductionMethod::PCA, 0);
        let reducer = SemanticDimensionReducer::new(config);
        assert_eq!(reducer.stats().method, ReductionMethod::PCA);
    }

    // --- Edge cases ---

    #[test]
    fn test_input_dim_equals_output_dim() {
        let config = make_config(5, 5, ReductionMethod::RandomProjection, 42);
        let mut reducer = SemanticDimensionReducer::new(config);
        let embedding = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let embeddings = vec![embedding.clone()];
        reducer.fit(&embeddings).expect("fit");
        let result = reducer.transform(&embedding).expect("transform");
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_input_dim_equals_output_dim_truncation() {
        let config = make_config(5, 5, ReductionMethod::Truncation, 0);
        let mut reducer = SemanticDimensionReducer::new(config);
        let embedding = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let embeddings = vec![embedding.clone()];
        reducer.fit(&embeddings).expect("fit");
        let result = reducer.transform(&embedding).expect("transform");
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_single_embedding() {
        let config = make_config(10, 3, ReductionMethod::RandomProjection, 42);
        let mut reducer = SemanticDimensionReducer::new(config);
        let embeddings = vec![vec![1.0; 10]];
        reducer.fit(&embeddings).expect("fit");
        let result = reducer.transform(&embeddings[0]).expect("transform");
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_single_embedding_pca() {
        // PCA with single embedding should still work (degenerate but no crash)
        let config = make_config(10, 3, ReductionMethod::PCA, 42);
        let mut reducer = SemanticDimensionReducer::new(config);
        let embeddings = vec![vec![1.0; 10]];
        reducer.fit(&embeddings).expect("fit");
        let result = reducer.transform(&embeddings[0]).expect("transform");
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_reduce_to_one_dimension() {
        let config = make_config(50, 1, ReductionMethod::RandomProjection, 42);
        let mut reducer = SemanticDimensionReducer::new(config);
        let embeddings = vec![vec![0.5; 50]; 5];
        reducer.fit(&embeddings).expect("fit");
        let result = reducer.transform(&embeddings[0]).expect("transform");
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_large_reduction_ratio() {
        let config = make_config(1000, 2, ReductionMethod::RandomProjection, 42);
        let mut reducer = SemanticDimensionReducer::new(config);
        let embeddings = vec![vec![0.1; 1000]; 3];
        reducer.fit(&embeddings).expect("fit");
        let result = reducer.transform(&embeddings[0]).expect("transform");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_reductions_counter_increments() {
        let config = make_config(10, 5, ReductionMethod::Truncation, 0);
        let mut reducer = SemanticDimensionReducer::new(config);
        let embeddings = vec![vec![1.0; 10]; 3];
        reducer.fit(&embeddings).expect("fit");
        assert_eq!(reducer.stats().reductions_performed, 0);
        let _ = reducer.transform(&embeddings[0]);
        assert_eq!(reducer.stats().reductions_performed, 1);
        let _ = reducer.transform(&embeddings[1]);
        let _ = reducer.transform(&embeddings[2]);
        assert_eq!(reducer.stats().reductions_performed, 3);
    }

    #[test]
    fn test_reduction_result_struct() {
        let result = ReductionResult {
            original_dim: 100,
            reduced_dim: 10,
            reconstruction_error: Some(0.05),
        };
        assert_eq!(result.original_dim, 100);
        assert_eq!(result.reduced_dim, 10);
        assert!((result.reconstruction_error.expect("should have error") - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_reduction_result_no_error() {
        let result = ReductionResult {
            original_dim: 100,
            reduced_dim: 10,
            reconstruction_error: None,
        };
        assert!(result.reconstruction_error.is_none());
    }

    #[test]
    fn test_reconstruction_error_zero_for_identity_truncation() {
        // When output_dim == input_dim for truncation, reconstruction error should be ~0
        let config = make_config(5, 5, ReductionMethod::Truncation, 0);
        let mut reducer = SemanticDimensionReducer::new(config);
        let embedding = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        reducer.fit(std::slice::from_ref(&embedding)).expect("fit");
        let reduced = reducer.transform(&embedding).expect("transform");
        let error = reducer.reconstruction_error(&embedding, &reduced);
        assert!(
            error < 1e-10,
            "identity truncation should have ~0 error, got {error}"
        );
    }

    #[test]
    fn test_fit_transform_counts_reductions() {
        let config = make_config(10, 3, ReductionMethod::RandomProjection, 42);
        let mut reducer = SemanticDimensionReducer::new(config);
        let embeddings = vec![vec![1.0; 10]; 7];
        let _ = reducer.fit_transform(&embeddings).expect("fit_transform");
        assert_eq!(reducer.stats().reductions_performed, 7);
    }
}
