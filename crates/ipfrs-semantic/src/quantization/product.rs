//! Product Quantization (PQ, OPQ) and benchmarking utilities.

use ipfrs_core::{Error, Result};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use super::scalar::{QuantizedVector, ScalarQuantizer};

/// Product Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizerConfig {
    /// Vector dimension
    pub dimension: usize,
    /// Number of sub-quantizers (sub-vectors)
    pub num_subquantizers: usize,
    /// Bits per sub-quantizer (usually 8)
    pub bits_per_subquantizer: u8,
    /// Codebooks for each sub-quantizer (centroids)
    pub codebooks: Vec<Vec<Vec<f32>>>,
}

/// Product Quantizer for high compression
///
/// Achieves 8-32x compression by dividing vectors into sub-vectors
/// and quantizing each with a codebook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizer {
    /// Configuration
    pub(crate) config: ProductQuantizerConfig,
    /// Sub-vector dimension
    subdimension: usize,
    /// Number of centroids per sub-quantizer
    num_centroids: usize,
    /// Whether trained
    trained: bool,
}

impl ProductQuantizer {
    /// Create a new product quantizer
    ///
    /// # Arguments
    /// * `dimension` - Vector dimension (must be divisible by num_subquantizers)
    /// * `num_subquantizers` - Number of sub-quantizers (typically 8, 16, or 32)
    /// * `bits` - Bits per code (typically 8, giving 256 centroids)
    pub fn new(dimension: usize, num_subquantizers: usize, bits: u8) -> Result<Self> {
        if !dimension.is_multiple_of(num_subquantizers) {
            return Err(Error::InvalidInput(format!(
                "Dimension {} must be divisible by num_subquantizers {}",
                dimension, num_subquantizers
            )));
        }

        if bits > 16 {
            return Err(Error::InvalidInput(
                "Bits per subquantizer must be <= 16".to_string(),
            ));
        }

        let subdimension = dimension / num_subquantizers;
        let num_centroids = 1 << bits;

        Ok(Self {
            config: ProductQuantizerConfig {
                dimension,
                num_subquantizers,
                bits_per_subquantizer: bits,
                codebooks: Vec::new(),
            },
            subdimension,
            num_centroids,
            trained: false,
        })
    }

    /// Create a standard PQ with 8 sub-quantizers and 8 bits
    pub fn standard(dimension: usize) -> Result<Self> {
        Self::new(dimension, 8, 8)
    }

    /// Train the product quantizer using k-means clustering
    ///
    /// # Arguments
    /// * `vectors` - Training vectors
    /// * `max_iterations` - Maximum k-means iterations
    pub fn train(&mut self, vectors: &[Vec<f32>], max_iterations: usize) -> Result<()> {
        if vectors.is_empty() {
            return Err(Error::InvalidInput(
                "Cannot train on empty vector set".to_string(),
            ));
        }

        // Validate dimensions
        for (i, vec) in vectors.iter().enumerate() {
            if vec.len() != self.config.dimension {
                return Err(Error::InvalidInput(format!(
                    "Vector {} has dimension {}, expected {}",
                    i,
                    vec.len(),
                    self.config.dimension
                )));
            }
        }

        // Train each sub-quantizer independently
        self.config.codebooks = Vec::with_capacity(self.config.num_subquantizers);

        for sq in 0..self.config.num_subquantizers {
            let start = sq * self.subdimension;
            let end = start + self.subdimension;

            // Extract sub-vectors for this sub-quantizer
            let subvectors: Vec<Vec<f32>> =
                vectors.iter().map(|v| v[start..end].to_vec()).collect();

            // Run k-means to find centroids
            let centroids = self.kmeans(&subvectors, self.num_centroids, max_iterations)?;
            self.config.codebooks.push(centroids);
        }

        self.trained = true;
        Ok(())
    }

    /// Simple k-means implementation
    fn kmeans(&self, data: &[Vec<f32>], k: usize, max_iterations: usize) -> Result<Vec<Vec<f32>>> {
        if data.is_empty() {
            return Err(Error::InvalidInput("Empty data for k-means".to_string()));
        }

        let dim = data[0].len();
        let n = data.len();
        let actual_k = k.min(n); // Can't have more centroids than data points

        // Initialize centroids using k-means++ style
        let mut centroids = Vec::with_capacity(actual_k);

        // Pick first centroid deterministically using first vector
        centroids.push(data[0].clone());

        // Pick remaining centroids with probability proportional to distance
        for _ in 1..actual_k {
            let mut best_idx = 0;
            let mut best_dist = 0.0f32;

            for (i, vec) in data.iter().enumerate() {
                let min_dist = centroids
                    .iter()
                    .map(|c| self.l2_distance(vec, c))
                    .fold(f32::MAX, |a, b| a.min(b));

                if min_dist > best_dist {
                    best_dist = min_dist;
                    best_idx = i;
                }
            }

            centroids.push(data[best_idx].clone());
        }

        // Run k-means iterations
        let mut assignments = vec![0usize; n];

        for _iter in 0..max_iterations {
            // Assign points to nearest centroid
            let mut changed = false;
            for (i, vec) in data.iter().enumerate() {
                let nearest = centroids
                    .iter()
                    .enumerate()
                    .map(|(j, c)| (j, self.l2_distance(vec, c)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(j, _)| j)
                    .unwrap_or(0);

                if assignments[i] != nearest {
                    assignments[i] = nearest;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update centroids
            let mut new_centroids = vec![vec![0.0f32; dim]; actual_k];
            let mut counts = vec![0usize; actual_k];

            for (i, vec) in data.iter().enumerate() {
                let cluster = assignments[i];
                counts[cluster] += 1;
                for (j, &val) in vec.iter().enumerate() {
                    new_centroids[cluster][j] += val;
                }
            }

            for (i, centroid) in new_centroids.iter_mut().enumerate() {
                if counts[i] > 0 {
                    for val in centroid.iter_mut() {
                        *val /= counts[i] as f32;
                    }
                } else {
                    // Keep old centroid if empty
                    *centroid = centroids[i].clone();
                }
            }

            centroids = new_centroids;
        }

        // Ensure we have exactly k centroids (pad with duplicates if needed)
        while centroids.len() < k {
            centroids.push(centroids[centroids.len() - 1].clone());
        }

        Ok(centroids)
    }

    fn l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    /// Quantize a vector to PQ codes
    pub fn quantize(&self, vector: &[f32]) -> Result<PQCode> {
        if !self.trained {
            return Err(Error::InvalidInput(
                "Product quantizer must be trained before use".to_string(),
            ));
        }

        if vector.len() != self.config.dimension {
            return Err(Error::InvalidInput(format!(
                "Vector has dimension {}, expected {}",
                vector.len(),
                self.config.dimension
            )));
        }

        let mut codes = Vec::with_capacity(self.config.num_subquantizers);

        for sq in 0..self.config.num_subquantizers {
            let start = sq * self.subdimension;
            let end = start + self.subdimension;
            let subvector = &vector[start..end];

            // Find nearest centroid
            let codebook = &self.config.codebooks[sq];
            let (best_idx, _) = codebook
                .iter()
                .enumerate()
                .map(|(i, c)| (i, self.l2_distance(subvector, c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, 0.0));

            codes.push(best_idx as u8);
        }

        Ok(PQCode { codes })
    }

    /// Dequantize PQ codes back to approximate vector
    pub fn dequantize(&self, code: &PQCode) -> Result<Vec<f32>> {
        if !self.trained {
            return Err(Error::InvalidInput(
                "Product quantizer must be trained before use".to_string(),
            ));
        }

        if code.codes.len() != self.config.num_subquantizers {
            return Err(Error::InvalidInput(format!(
                "PQ code has {} elements, expected {}",
                code.codes.len(),
                self.config.num_subquantizers
            )));
        }

        let mut result = Vec::with_capacity(self.config.dimension);

        for (sq, &idx) in code.codes.iter().enumerate() {
            let centroid = &self.config.codebooks[sq][idx as usize];
            result.extend_from_slice(centroid);
        }

        Ok(result)
    }

    /// Compute asymmetric distance (query is not quantized)
    ///
    /// This is the preferred method for search as it's more accurate
    pub fn asymmetric_distance(&self, query: &[f32], code: &PQCode) -> Result<f32> {
        if !self.trained {
            return Err(Error::InvalidInput(
                "Product quantizer must be trained".to_string(),
            ));
        }

        let mut total_dist_sq = 0.0f32;

        for sq in 0..self.config.num_subquantizers {
            let start = sq * self.subdimension;
            let end = start + self.subdimension;
            let subquery = &query[start..end];
            let centroid = &self.config.codebooks[sq][code.codes[sq] as usize];

            for (q, c) in subquery.iter().zip(centroid.iter()) {
                let diff = q - c;
                total_dist_sq += diff * diff;
            }
        }

        Ok(total_dist_sq.sqrt())
    }

    /// Precompute distance tables for fast ADC (Asymmetric Distance Computation)
    ///
    /// Returns a table\[sq\]\[centroid\] = distance from query subvector to centroid
    pub fn compute_distance_table(&self, query: &[f32]) -> Result<Vec<Vec<f32>>> {
        if !self.trained {
            return Err(Error::InvalidInput(
                "Product quantizer must be trained".to_string(),
            ));
        }

        let mut table = Vec::with_capacity(self.config.num_subquantizers);

        for sq in 0..self.config.num_subquantizers {
            let start = sq * self.subdimension;
            let end = start + self.subdimension;
            let subquery = &query[start..end];

            let distances: Vec<f32> = self.config.codebooks[sq]
                .iter()
                .map(|c| {
                    subquery
                        .iter()
                        .zip(c.iter())
                        .map(|(q, c)| (q - c) * (q - c))
                        .sum::<f32>()
                })
                .collect();

            table.push(distances);
        }

        Ok(table)
    }

    /// Fast distance computation using precomputed table
    pub fn distance_from_table(&self, table: &[Vec<f32>], code: &PQCode) -> f32 {
        let mut total = 0.0f32;
        for (sq, &idx) in code.codes.iter().enumerate() {
            total += table[sq][idx as usize];
        }
        total.sqrt()
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        // Original: dimension * 4 bytes (f32)
        // Compressed: num_subquantizers * 1 byte
        (self.config.dimension * 4) as f32 / self.config.num_subquantizers as f32
    }

    /// Check if trained
    pub fn is_trained(&self) -> bool {
        self.trained
    }
}

/// Product Quantization code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQCode {
    /// Centroid indices for each sub-quantizer
    pub codes: Vec<u8>,
}

impl PQCode {
    /// Get memory size in bytes
    pub fn size_bytes(&self) -> usize {
        self.codes.len()
    }
}

/// Optimized Product Quantization (OPQ)
///
/// OPQ extends PQ by learning a rotation matrix that minimizes quantization error.
/// It achieves better accuracy than standard PQ at the same compression ratio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedProductQuantizer {
    /// Base product quantizer
    pq: ProductQuantizer,
    /// Rotation matrix (dimension x dimension)
    #[serde(with = "rotation_matrix_serde")]
    rotation: Option<DMatrix<f32>>,
    /// Whether rotation is trained
    rotation_trained: bool,
}

// Custom serialization for DMatrix
mod rotation_matrix_serde {
    use super::DMatrix;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct MatrixData {
        nrows: usize,
        ncols: usize,
        data: Vec<f32>,
    }

    pub fn serialize<S>(
        matrix: &Option<DMatrix<f32>>,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let opt_data = matrix.as_ref().map(|m| MatrixData {
            nrows: m.nrows(),
            ncols: m.ncols(),
            data: m.as_slice().to_vec(),
        });
        opt_data.serialize(serializer)
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> std::result::Result<Option<DMatrix<f32>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt: Option<MatrixData> = Option::deserialize(deserializer)?;
        Ok(opt.map(|data| DMatrix::from_vec(data.nrows, data.ncols, data.data)))
    }
}

impl OptimizedProductQuantizer {
    /// Create a new OPQ quantizer
    ///
    /// # Arguments
    /// * `dimension` - Vector dimension (must be divisible by num_subquantizers)
    /// * `num_subquantizers` - Number of sub-quantizers
    /// * `bits` - Bits per code (typically 8)
    pub fn new(dimension: usize, num_subquantizers: usize, bits: u8) -> Result<Self> {
        let pq = ProductQuantizer::new(dimension, num_subquantizers, bits)?;
        Ok(Self {
            pq,
            rotation: None,
            rotation_trained: false,
        })
    }

    /// Create standard OPQ with 8 sub-quantizers and 8 bits
    pub fn standard(dimension: usize) -> Result<Self> {
        Self::new(dimension, 8, 8)
    }

    /// Train OPQ with rotation learning
    ///
    /// Uses iterative optimization: alternate between learning rotation and PQ codebooks
    ///
    /// # Arguments
    /// * `vectors` - Training vectors
    /// * `max_iterations` - Max iterations for PQ k-means
    /// * `rotation_iterations` - Iterations for rotation learning (typically 5-10)
    #[allow(clippy::too_many_arguments)]
    pub fn train(
        &mut self,
        vectors: &[Vec<f32>],
        max_iterations: usize,
        rotation_iterations: usize,
    ) -> Result<()> {
        if vectors.is_empty() {
            return Err(Error::InvalidInput(
                "Cannot train on empty vector set".to_string(),
            ));
        }

        let dim = self.pq.config.dimension;

        // Validate dimensions
        for (i, vec) in vectors.iter().enumerate() {
            if vec.len() != dim {
                return Err(Error::InvalidInput(format!(
                    "Vector {} has dimension {}, expected {}",
                    i,
                    vec.len(),
                    dim
                )));
            }
        }

        // Initialize with identity rotation
        let mut rotation = DMatrix::<f32>::identity(dim, dim);

        // Iteratively optimize rotation and PQ
        for iteration in 0..rotation_iterations {
            // Step 1: Rotate vectors
            let rotated = self.apply_rotation_batch(vectors, &rotation);

            // Step 2: Train PQ on rotated vectors
            self.pq.train(&rotated, max_iterations)?;

            // Step 3: Learn better rotation (only if not last iteration)
            if iteration < rotation_iterations - 1 {
                rotation = self.learn_rotation(vectors, &self.pq)?;
            }
        }

        self.rotation = Some(rotation);
        self.rotation_trained = true;

        Ok(())
    }

    /// Learn rotation matrix using SVD
    ///
    /// Finds rotation that aligns data with PQ structure
    #[allow(dead_code)]
    fn learn_rotation(&self, vectors: &[Vec<f32>], pq: &ProductQuantizer) -> Result<DMatrix<f32>> {
        let dim = pq.config.dimension;
        let n = vectors.len();

        // Compute covariance between original and reconstructed vectors
        let mut cov = DMatrix::<f32>::zeros(dim, dim);

        for vec in vectors {
            // Quantize and reconstruct
            let code = pq.quantize(vec)?;
            let reconstructed = pq.dequantize(&code)?;

            // Compute outer product
            let v = DVector::from_vec(vec.clone());
            let r = DVector::from_vec(reconstructed);
            cov += v * r.transpose();
        }

        cov /= n as f32;

        // SVD to find optimal rotation
        let svd = cov.svd(true, true);

        // Rotation = U * V^T
        match (svd.u, svd.v_t) {
            (Some(u), Some(vt)) => Ok(u * vt),
            _ => {
                // If SVD fails, return identity
                Ok(DMatrix::identity(dim, dim))
            }
        }
    }

    /// Apply rotation to a batch of vectors
    fn apply_rotation_batch(&self, vectors: &[Vec<f32>], rotation: &DMatrix<f32>) -> Vec<Vec<f32>> {
        vectors
            .iter()
            .map(|v| self.apply_rotation(v, rotation))
            .collect()
    }

    /// Apply rotation to a single vector
    fn apply_rotation(&self, vector: &[f32], rotation: &DMatrix<f32>) -> Vec<f32> {
        let v = DVector::from_vec(vector.to_vec());
        let rotated = rotation * v;
        rotated.as_slice().to_vec()
    }

    /// Quantize a vector
    pub fn quantize(&self, vector: &[f32]) -> Result<PQCode> {
        if !self.is_trained() {
            return Err(Error::InvalidInput("OPQ must be trained".to_string()));
        }

        // Apply rotation before quantization
        let rotated = match &self.rotation {
            Some(r) => self.apply_rotation(vector, r),
            None => vector.to_vec(),
        };

        self.pq.quantize(&rotated)
    }

    /// Dequantize a code back to vector
    pub fn dequantize(&self, code: &PQCode) -> Result<Vec<f32>> {
        if !self.is_trained() {
            return Err(Error::InvalidInput("OPQ must be trained".to_string()));
        }

        // Dequantize
        let rotated = self.pq.dequantize(code)?;

        // Apply inverse rotation
        match &self.rotation {
            Some(r) => {
                // For orthogonal matrices, inverse = transpose
                let r_inv = r.transpose();
                Ok(self.apply_rotation(&rotated, &r_inv))
            }
            None => Ok(rotated),
        }
    }

    /// Compute asymmetric distance (query is not quantized)
    pub fn asymmetric_distance(&self, query: &[f32], code: &PQCode) -> Result<f32> {
        if !self.is_trained() {
            return Err(Error::InvalidInput("OPQ must be trained".to_string()));
        }

        // Rotate query
        let rotated_query = match &self.rotation {
            Some(r) => self.apply_rotation(query, r),
            None => query.to_vec(),
        };

        self.pq.asymmetric_distance(&rotated_query, code)
    }

    /// Compute distance table for fast batch queries
    pub fn compute_distance_table(&self, query: &[f32]) -> Result<Vec<Vec<f32>>> {
        if !self.is_trained() {
            return Err(Error::InvalidInput("OPQ must be trained".to_string()));
        }

        // Rotate query
        let rotated_query = match &self.rotation {
            Some(r) => self.apply_rotation(query, r),
            None => query.to_vec(),
        };

        self.pq.compute_distance_table(&rotated_query)
    }

    /// Fast distance using precomputed table
    pub fn distance_from_table(&self, table: &[Vec<f32>], code: &PQCode) -> f32 {
        self.pq.distance_from_table(table, code)
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        self.pq.compression_ratio()
    }

    /// Check if trained
    pub fn is_trained(&self) -> bool {
        self.pq.is_trained() && self.rotation_trained
    }

    /// Get the underlying PQ (for testing)
    #[allow(dead_code)]
    pub fn inner_pq(&self) -> &ProductQuantizer {
        &self.pq
    }
}

/// Quantization benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationBenchmark {
    /// Recall at k for various k values
    pub recall_at_k: Vec<(usize, f32)>,
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Average quantization error
    pub avg_quantization_error: f32,
    /// Max quantization error
    pub max_quantization_error: f32,
    /// Memory savings in bytes
    pub memory_savings: usize,
    /// Speed improvement factor (approximate)
    pub speed_factor: f32,
}

impl QuantizationBenchmark {
    /// Generate a summary string
    pub fn summary(&self) -> String {
        let recall_str: Vec<String> = self
            .recall_at_k
            .iter()
            .map(|(k, r)| format!("R@{}: {:.2}%", k, r * 100.0))
            .collect();

        format!(
            "Compression: {:.1}x, Avg Error: {:.4}, {}, Memory Saved: {} bytes",
            self.compression_ratio,
            self.avg_quantization_error,
            recall_str.join(", "),
            self.memory_savings
        )
    }
}

/// Benchmark utilities for quantization evaluation
pub struct QuantizationBenchmarker;

impl QuantizationBenchmarker {
    /// Benchmark scalar quantization on a dataset
    ///
    /// # Arguments
    /// * `quantizer` - Trained scalar quantizer
    /// * `vectors` - Test vectors
    /// * `queries` - Query vectors
    /// * `ground_truth` - Ground truth k-NN for each query (indices into vectors)
    /// * `k_values` - Values of k to measure recall at
    pub fn benchmark_scalar(
        quantizer: &ScalarQuantizer,
        vectors: &[Vec<f32>],
        queries: &[Vec<f32>],
        ground_truth: &[Vec<usize>],
        k_values: &[usize],
    ) -> Result<QuantizationBenchmark> {
        if !quantizer.is_trained() {
            return Err(Error::InvalidInput("Quantizer must be trained".to_string()));
        }

        // Quantize all vectors
        let quantized: Vec<QuantizedVector> = vectors
            .iter()
            .map(|v| quantizer.quantize(v))
            .collect::<Result<Vec<_>>>()?;

        // Compute quantization error
        let mut total_error = 0.0f32;
        let mut max_error = 0.0f32;

        for (i, qv) in quantized.iter().enumerate() {
            let restored = quantizer.dequantize(qv)?;
            let error: f32 = vectors[i]
                .iter()
                .zip(restored.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            total_error += error;
            max_error = max_error.max(error);
        }

        let avg_error = total_error / vectors.len() as f32;

        // Compute recall at k
        let mut recall_at_k = Vec::new();

        for &k in k_values {
            let mut total_recall = 0.0f32;

            for (qi, query) in queries.iter().enumerate() {
                let query_quantized = quantizer.quantize(query)?;

                // Find k nearest using quantized distances
                let mut distances: Vec<(usize, f32)> = quantized
                    .iter()
                    .enumerate()
                    .map(|(i, qv)| {
                        let dist = quantizer
                            .distance_l2_quantized(&query_quantized, qv)
                            .unwrap_or(f32::MAX);
                        (i, dist)
                    })
                    .collect();

                distances
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                let found: std::collections::HashSet<usize> =
                    distances.iter().take(k).map(|(i, _)| *i).collect();

                let gt: std::collections::HashSet<usize> =
                    ground_truth[qi].iter().take(k).copied().collect();

                let intersection = found.intersection(&gt).count();
                total_recall += intersection as f32 / k.min(gt.len()) as f32;
            }

            let recall = total_recall / queries.len() as f32;
            recall_at_k.push((k, recall));
        }

        // Calculate memory savings
        let original_size = vectors.len() * vectors[0].len() * 4; // f32 = 4 bytes
        let quantized_size = vectors.len() * vectors[0].len(); // u8 = 1 byte
        let memory_savings = original_size - quantized_size;

        Ok(QuantizationBenchmark {
            recall_at_k,
            compression_ratio: quantizer.compression_ratio(),
            avg_quantization_error: avg_error,
            max_quantization_error: max_error,
            memory_savings,
            speed_factor: 2.0, // Approximate for int ops vs float ops
        })
    }

    /// Benchmark product quantization on a dataset
    pub fn benchmark_pq(
        pq: &ProductQuantizer,
        vectors: &[Vec<f32>],
        queries: &[Vec<f32>],
        ground_truth: &[Vec<usize>],
        k_values: &[usize],
    ) -> Result<QuantizationBenchmark> {
        if !pq.is_trained() {
            return Err(Error::InvalidInput("PQ must be trained".to_string()));
        }

        // Quantize all vectors
        let codes: Vec<PQCode> = vectors
            .iter()
            .map(|v| pq.quantize(v))
            .collect::<Result<Vec<_>>>()?;

        // Compute quantization error
        let mut total_error = 0.0f32;
        let mut max_error = 0.0f32;

        for (i, code) in codes.iter().enumerate() {
            let restored = pq.dequantize(code)?;
            let error: f32 = vectors[i]
                .iter()
                .zip(restored.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            total_error += error;
            max_error = max_error.max(error);
        }

        let avg_error = total_error / vectors.len() as f32;

        // Compute recall at k using asymmetric distance
        let mut recall_at_k = Vec::new();

        for &k in k_values {
            let mut total_recall = 0.0f32;

            for (qi, query) in queries.iter().enumerate() {
                // Use distance table for fast computation
                let table = pq.compute_distance_table(query)?;

                let mut distances: Vec<(usize, f32)> = codes
                    .iter()
                    .enumerate()
                    .map(|(i, code)| (i, pq.distance_from_table(&table, code)))
                    .collect();

                distances
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                let found: std::collections::HashSet<usize> =
                    distances.iter().take(k).map(|(i, _)| *i).collect();

                let gt: std::collections::HashSet<usize> =
                    ground_truth[qi].iter().take(k).copied().collect();

                let intersection = found.intersection(&gt).count();
                total_recall += intersection as f32 / k.min(gt.len()) as f32;
            }

            let recall = total_recall / queries.len() as f32;
            recall_at_k.push((k, recall));
        }

        // Calculate memory savings
        let original_size = vectors.len() * vectors[0].len() * 4;
        let quantized_size = vectors.len() * codes[0].size_bytes();
        let memory_savings = original_size.saturating_sub(quantized_size);

        Ok(QuantizationBenchmark {
            recall_at_k,
            compression_ratio: pq.compression_ratio(),
            avg_quantization_error: avg_error,
            max_quantization_error: max_error,
            memory_savings,
            speed_factor: 4.0, // Approximate for table lookup vs float ops
        })
    }

    /// Compute ground truth k-NN using brute force L2 distance
    pub fn compute_ground_truth(
        vectors: &[Vec<f32>],
        queries: &[Vec<f32>],
        k: usize,
    ) -> Vec<Vec<usize>> {
        queries
            .iter()
            .map(|query| {
                let mut distances: Vec<(usize, f32)> = vectors
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let dist: f32 = query
                            .iter()
                            .zip(v.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum();
                        (i, dist)
                    })
                    .collect();

                distances
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                distances.iter().take(k).map(|(i, _)| *i).collect()
            })
            .collect()
    }

    /// Compare multiple quantization methods
    pub fn compare_methods(
        vectors: &[Vec<f32>],
        queries: &[Vec<f32>],
        k_values: &[usize],
    ) -> Result<QuantizationComparison> {
        let max_k = *k_values.iter().max().unwrap_or(&10);
        let ground_truth = Self::compute_ground_truth(vectors, queries, max_k);

        // Benchmark scalar quantization
        let mut sq = ScalarQuantizer::uint8(vectors[0].len());
        sq.train(vectors)?;
        let scalar_results =
            Self::benchmark_scalar(&sq, vectors, queries, &ground_truth, k_values)?;

        // Benchmark PQ (if dimension allows)
        let dim = vectors[0].len();
        let pq_results = if dim >= 8 && dim.is_multiple_of(8) {
            let mut pq = ProductQuantizer::new(dim, 8, 8)?;
            pq.train(vectors, 20)?;
            Some(Self::benchmark_pq(
                &pq,
                vectors,
                queries,
                &ground_truth,
                k_values,
            )?)
        } else {
            None
        };

        Ok(QuantizationComparison {
            scalar: scalar_results,
            product: pq_results,
            dataset_size: vectors.len(),
            dimension: dim,
        })
    }
}

/// Comparison of multiple quantization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationComparison {
    /// Scalar quantization results
    pub scalar: QuantizationBenchmark,
    /// Product quantization results (if applicable)
    pub product: Option<QuantizationBenchmark>,
    /// Dataset size
    pub dataset_size: usize,
    /// Vector dimension
    pub dimension: usize,
}

impl QuantizationComparison {
    /// Generate a comparison summary
    pub fn summary(&self) -> String {
        let mut result = format!(
            "Dataset: {} vectors, {} dimensions\n\nScalar Quantization:\n  {}\n",
            self.dataset_size,
            self.dimension,
            self.scalar.summary()
        );

        if let Some(ref pq) = self.product {
            result.push_str(&format!("\nProduct Quantization:\n  {}\n", pq.summary()));
        }

        result
    }

    /// Get the best method for a given k value based on recall
    pub fn best_method_for_k(&self, k: usize) -> (&str, f32) {
        let scalar_recall = self
            .scalar
            .recall_at_k
            .iter()
            .find(|(kv, _)| *kv == k)
            .map(|(_, r)| *r)
            .unwrap_or(0.0);

        if let Some(ref pq) = self.product {
            let pq_recall = pq
                .recall_at_k
                .iter()
                .find(|(kv, _)| *kv == k)
                .map(|(_, r)| *r)
                .unwrap_or(0.0);

            if pq_recall > scalar_recall {
                return ("ProductQuantization", pq_recall);
            }
        }

        ("ScalarQuantization", scalar_recall)
    }
}
