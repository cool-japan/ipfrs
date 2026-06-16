//! Memory-efficient vector stores: INT8 (`QuantizedVectorStore`) and binary (`BinaryVectorStore`).

use super::{dequantize_i8_to_f32, quantize_f32_to_i8};

/// Stores f32 vectors quantized to INT8, reducing memory ~4× vs raw f32.
///
/// Each vector is independently scaled with per-vector `scale` and `zero_point`
/// so that dequantization error is bounded to roughly `scale / 2` per element.
///
/// # Memory layout
/// Internally uses a single flat `Vec<i8>` of length `count * dim`.
pub struct QuantizedVectorStore {
    /// Flattened quantized data: `[v0_d0, v0_d1, …, v1_d0, …]`
    data: Vec<i8>,
    /// Vector dimension
    dim: usize,
    /// Number of stored vectors
    count: usize,
    /// Per-vector scale factors (index → scale)
    scales: Vec<f32>,
    /// Per-vector zero-points (index → zero_point)
    zero_points: Vec<f32>,
}

impl QuantizedVectorStore {
    /// Create an empty store for vectors of `dim` dimensions.
    pub fn new(dim: usize) -> Self {
        Self {
            data: Vec::new(),
            dim,
            count: 0,
            scales: Vec::new(),
            zero_points: Vec::new(),
        }
    }

    /// Quantize `vector` to INT8 and append it to the store.
    ///
    /// Returns the assigned integer ID (0-based).
    pub fn push(&mut self, vector: &[f32]) -> usize {
        debug_assert_eq!(
            vector.len(),
            self.dim,
            "push: vector length {} != dim {}",
            vector.len(),
            self.dim
        );

        let (q, scale, zero_point) = quantize_f32_to_i8(vector);
        self.data.extend_from_slice(&q);
        self.scales.push(scale);
        self.zero_points.push(zero_point);

        let id = self.count;
        self.count += 1;
        id
    }

    /// Dequantize the vector stored at `id` back to f32.
    ///
    /// Returns `None` if `id >= self.len()`.
    pub fn get(&self, id: usize) -> Option<Vec<f32>> {
        if id >= self.count {
            return None;
        }
        let start = id * self.dim;
        let end = start + self.dim;
        let q = &self.data[start..end];
        Some(dequantize_i8_to_f32(
            q,
            self.scales[id],
            self.zero_points[id],
        ))
    }

    /// Compute approximate cosine similarity between two stored vectors.
    ///
    /// Dequantizes both vectors then computes exact cosine similarity in f32.
    /// Returns 0.0 if either id is out of range or a vector has zero norm.
    pub fn cosine_similarity_q(&self, a_id: usize, b_id: usize) -> f32 {
        let a = match self.get(a_id) {
            Some(v) => v,
            None => return 0.0,
        };
        let b = match self.get(b_id) {
            Some(v) => v,
            None => return 0.0,
        };

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a < 1e-9 || norm_b < 1e-9 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    /// Approximate memory usage per stored vector (bytes).
    ///
    /// Counts the `i8` storage only (not scale/zero_point metadata):
    /// `dim * 1 byte` per vector.
    pub fn bytes_per_vector(&self) -> f64 {
        self.dim as f64
    }

    /// Number of stored vectors.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` if no vectors have been stored.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Stores f32 vectors binarized to 1 bit per dimension.
///
/// Each dimension is thresholded at the per-vector mean:
/// `bit = 1 if v[i] >= mean(v) else 0`.
///
/// Bits are packed into `u64` words (64 dims per word), so a 384-dim vector
/// occupies `ceil(384/64) = 6` words = 48 bytes, vs 1536 bytes for f32.
///
/// Similarity is measured with Hamming distance (popcount of XOR).
pub struct BinaryVectorStore {
    /// Packed bit data: `dim_words` u64 words per vector.
    data: Vec<u64>,
    /// Original vector dimension (number of bits per vector).
    dim: usize,
    /// Number of stored vectors.
    count: usize,
    /// Number of u64 words per vector: `(dim + 63) / 64`.
    dim_words: usize,
}

impl BinaryVectorStore {
    /// Create an empty store for vectors of `dim` dimensions.
    pub fn new(dim: usize) -> Self {
        let dim_words = dim.div_ceil(64);
        Self {
            data: Vec::new(),
            dim,
            count: 0,
            dim_words,
        }
    }

    /// Binarize `vector` (threshold at mean) and store it.
    ///
    /// Returns the assigned ID.
    pub fn push(&mut self, vector: &[f32]) -> usize {
        debug_assert_eq!(
            vector.len(),
            self.dim,
            "push: vector length {} != dim {}",
            vector.len(),
            self.dim
        );

        // Compute per-vector mean for threshold
        let mean = if vector.is_empty() {
            0.0_f32
        } else {
            vector.iter().sum::<f32>() / vector.len() as f32
        };

        // Pack bits into u64 words
        let mut words = vec![0u64; self.dim_words];
        for (i, &val) in vector.iter().enumerate() {
            if val >= mean {
                let word_idx = i / 64;
                let bit_idx = i % 64;
                words[word_idx] |= 1u64 << bit_idx;
            }
        }

        self.data.extend_from_slice(&words);
        let id = self.count;
        self.count += 1;
        id
    }

    /// Compute the Hamming distance between two stored vectors.
    ///
    /// Hamming distance = number of bit positions where the two vectors differ.
    /// Returns `u32::MAX` if either id is out of range.
    pub fn hamming_distance(&self, a_id: usize, b_id: usize) -> u32 {
        if a_id >= self.count || b_id >= self.count {
            return u32::MAX;
        }

        let a_start = a_id * self.dim_words;
        let b_start = b_id * self.dim_words;
        let a_words = &self.data[a_start..a_start + self.dim_words];
        let b_words = &self.data[b_start..b_start + self.dim_words];

        a_words
            .iter()
            .zip(b_words.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    /// Approximate memory per stored vector (bytes).
    ///
    /// `dim_words * 8` bytes per vector, e.g. dim=384 → 6 × 8 = 48 bytes.
    pub fn bytes_per_vector(&self) -> f64 {
        (self.dim_words * 8) as f64
    }

    /// Number of stored vectors.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` if no vectors have been stored.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}
