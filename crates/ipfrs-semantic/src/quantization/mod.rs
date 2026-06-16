//! Vector quantization for memory-efficient storage
//!
//! This module provides various quantization methods to compress
//! high-dimensional vectors while preserving similarity search accuracy.

pub mod product;
pub mod scalar;
pub mod stores;

// Re-export public API
pub use product::{
    OptimizedProductQuantizer, PQCode, ProductQuantizer, ProductQuantizerConfig,
    QuantizationBenchmark, QuantizationBenchmarker, QuantizationComparison,
};
pub use scalar::{QuantizedVector, ScalarQuantizer, ScalarQuantizerConfig};
pub use stores::{BinaryVectorStore, QuantizedVectorStore};

// ─────────────────────────────────────────────────────────────────────────────
// INT8 per-vector quantization helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Quantize an f32 slice to INT8 using symmetric min-max scaling.
///
/// The scale factor maps the full dynamic range of `v` to `[-127, 127]`.
/// Zero-point shifts so that the true zero maps to quantized zero.
///
/// # Returns
/// `(quantized, scale, zero_point)` where
/// - `quantized[i] = round((v[i] - zero_point) / scale).clamp(-128, 127)`
/// - dequantization: `v'[i] = quantized[i] * scale + zero_point`
pub fn quantize_f32_to_i8(v: &[f32]) -> (Vec<i8>, f32, f32) {
    if v.is_empty() {
        return (Vec::new(), 1.0, 0.0);
    }

    let min_val = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Symmetric range around mid-point
    let zero_point = (min_val + max_val) * 0.5;
    let half_range = (max_val - min_val) * 0.5;

    // Avoid division by zero for constant vectors
    let scale = if half_range < 1e-9 {
        1.0_f32
    } else {
        half_range / 127.0
    };

    let quantized: Vec<i8> = v
        .iter()
        .map(|&x| {
            let q = ((x - zero_point) / scale).round();
            q.clamp(-128.0, 127.0) as i8
        })
        .collect();

    (quantized, scale, zero_point)
}

/// Dequantize INT8 vector back to f32.
///
/// Inverse of [`quantize_f32_to_i8`]:
/// `v'[i] = quantized[i] * scale + zero_point`
pub fn dequantize_i8_to_f32(q: &[i8], scale: f32, zero_point: f32) -> Vec<f32> {
    q.iter().map(|&qi| qi as f32 * scale + zero_point).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_quantizer_uint8() {
        let mut quantizer = ScalarQuantizer::uint8(4);

        // Train on some vectors
        let vectors = vec![
            vec![0.0, 0.5, 1.0, -0.5],
            vec![1.0, 0.0, 0.5, 0.5],
            vec![0.5, 0.5, 0.0, 0.0],
        ];

        quantizer.train(&vectors).expect("train");
        assert!(quantizer.is_trained());

        // Quantize and dequantize
        let original = vec![0.5, 0.25, 0.75, 0.0];
        let quantized = quantizer.quantize(&original).expect("quantize");
        let restored = quantizer.dequantize(&quantized).expect("dequantize");

        // Check approximate equality
        for (o, r) in original.iter().zip(restored.iter()) {
            assert!((o - r).abs() < 0.05, "Expected {} ~= {}", o, r);
        }

        assert_eq!(quantizer.compression_ratio(), 4.0);
    }

    #[test]
    fn test_scalar_quantizer_int8() {
        let mut quantizer = ScalarQuantizer::int8(4);

        let vectors = vec![vec![-1.0, 0.0, 1.0, -0.5], vec![1.0, -1.0, 0.5, 0.5]];

        quantizer.train(&vectors).expect("train");

        let original = vec![0.0, -0.5, 0.5, 0.25];
        let quantized = quantizer.quantize(&original).expect("quantize");
        let restored = quantizer.dequantize(&quantized).expect("dequantize");

        for (o, r) in original.iter().zip(restored.iter()) {
            assert!((o - r).abs() < 0.1, "Expected {} ~= {}", o, r);
        }
    }

    #[test]
    fn test_scalar_distance() {
        let mut quantizer = ScalarQuantizer::uint8(4);

        let vectors = vec![vec![0.0, 0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0, 1.0]];

        quantizer.train(&vectors).expect("train");

        let a = quantizer.quantize(&[0.0, 0.0, 0.0, 0.0]).expect("qa");
        let b = quantizer.quantize(&[1.0, 1.0, 1.0, 1.0]).expect("qb");

        let dist = quantizer.distance_l2_quantized(&a, &b).expect("dist");
        assert!(dist > 0.0, "Distance should be positive");
    }

    #[test]
    fn test_product_quantizer() {
        // 8-dimensional vectors with 2 sub-quantizers
        let mut pq = ProductQuantizer::new(8, 2, 4).expect("new pq"); // 4 bits = 16 centroids

        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| (0..8).map(|j| i as f32 * 0.01 + j as f32 * 0.1).collect())
            .collect();

        pq.train(&vectors, 10).expect("train");
        assert!(pq.is_trained());

        // Quantize and dequantize
        let original = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let code = pq.quantize(&original).expect("quantize");
        let restored = pq.dequantize(&code).expect("dequantize");

        // Check code size
        assert_eq!(code.codes.len(), 2);
        assert_eq!(restored.len(), 8);

        // Compression ratio should be 8*4 / 2 = 16
        assert_eq!(pq.compression_ratio(), 16.0);
    }

    #[test]
    fn test_pq_distance_table() {
        let mut pq = ProductQuantizer::new(8, 2, 4).expect("new pq");

        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| (0..8).map(|j| i as f32 * 0.02 + j as f32 * 0.1).collect())
            .collect();

        pq.train(&vectors, 5).expect("train");

        let query = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let code = pq.quantize(&vectors[0]).expect("quantize");

        // Compute distance two ways
        let direct_dist = pq.asymmetric_distance(&query, &code).expect("dist");
        let table = pq.compute_distance_table(&query).expect("table");
        let table_dist = pq.distance_from_table(&table, &code);

        // Should be equal (within floating point tolerance)
        assert!((direct_dist - table_dist).abs() < 1e-5);
    }

    #[test]
    fn test_quantization_benchmarker() {
        // Create test dataset
        let dim = 8;
        let n_vectors = 50;
        let n_queries = 5;

        let vectors: Vec<Vec<f32>> = (0..n_vectors)
            .map(|i| (0..dim).map(|j| (i as f32 + j as f32) * 0.1).collect())
            .collect();

        let queries: Vec<Vec<f32>> = (0..n_queries)
            .map(|i| {
                (0..dim)
                    .map(|j| (i as f32 * 2.0 + j as f32) * 0.1)
                    .collect()
            })
            .collect();

        // Test ground truth computation
        let gt = QuantizationBenchmarker::compute_ground_truth(&vectors, &queries, 10);
        assert_eq!(gt.len(), n_queries);
        assert_eq!(gt[0].len(), 10);

        // Test scalar quantization benchmark
        let mut sq = ScalarQuantizer::uint8(dim);
        sq.train(&vectors).expect("train");

        let sq_benchmark =
            QuantizationBenchmarker::benchmark_scalar(&sq, &vectors, &queries, &gt, &[1, 5, 10])
                .expect("benchmark");

        assert_eq!(sq_benchmark.recall_at_k.len(), 3);
        assert!(sq_benchmark.compression_ratio > 1.0);
        assert!(sq_benchmark.memory_savings > 0);
    }

    #[test]
    fn test_quantization_comparison() {
        let dim = 8;
        let n_vectors = 100;
        let n_queries = 10;

        let vectors: Vec<Vec<f32>> = (0..n_vectors)
            .map(|i| (0..dim).map(|j| (i as f32 + j as f32) * 0.05).collect())
            .collect();

        let queries: Vec<Vec<f32>> = (0..n_queries)
            .map(|i| {
                (0..dim)
                    .map(|j| (i as f32 * 3.0 + j as f32) * 0.05)
                    .collect()
            })
            .collect();

        let comparison =
            QuantizationBenchmarker::compare_methods(&vectors, &queries, &[1, 5, 10]).expect("cmp");

        assert!(comparison.dataset_size == n_vectors);
        assert!(comparison.dimension == dim);

        let summary = comparison.summary();
        assert!(!summary.is_empty());

        let (method, _recall) = comparison.best_method_for_k(5);
        assert!(!method.is_empty());
    }

    #[test]
    fn test_opq_basic() {
        let mut opq = OptimizedProductQuantizer::new(8, 2, 4).expect("new opq");

        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| (0..8).map(|j| i as f32 * 0.01 + j as f32 * 0.1).collect())
            .collect();

        // Train with rotation learning (5 iterations)
        opq.train(&vectors, 10, 5).expect("train");
        assert!(opq.is_trained());

        // Quantize and dequantize
        let original = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let code = opq.quantize(&original).expect("quantize");
        let restored = opq.dequantize(&code).expect("dequantize");

        assert_eq!(code.codes.len(), 2);
        assert_eq!(restored.len(), 8);

        // Compression ratio should be same as PQ
        assert_eq!(opq.compression_ratio(), 16.0);

        // Test asymmetric distance
        let query = vec![0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85];
        let dist = opq.asymmetric_distance(&query, &code).expect("dist");
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_opq_distance_table() {
        let mut opq = OptimizedProductQuantizer::new(8, 2, 4).expect("new opq");

        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| (0..8).map(|j| i as f32 * 0.02 + j as f32 * 0.1).collect())
            .collect();

        opq.train(&vectors, 5, 3).expect("train");

        let query = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let code = opq.quantize(&vectors[0]).expect("quantize");

        // Compute distance two ways
        let direct_dist = opq.asymmetric_distance(&query, &code).expect("dist");
        let table = opq.compute_distance_table(&query).expect("table");
        let table_dist = opq.distance_from_table(&table, &code);

        // Should be equal (within floating point tolerance)
        assert!((direct_dist - table_dist).abs() < 1e-4);
    }

    #[test]
    fn test_opq_serialization() {
        let mut opq = OptimizedProductQuantizer::new(8, 2, 4).expect("new opq");

        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| (0..8).map(|j| i as f32 * 0.02 + j as f32 * 0.1).collect())
            .collect();

        opq.train(&vectors, 5, 3).expect("train");

        // Serialize
        let serialized =
            oxicode::serde::encode_to_vec(&opq, oxicode::config::standard()).expect("serialize");

        // Deserialize
        let deserialized: OptimizedProductQuantizer =
            oxicode::serde::decode_owned_from_slice(&serialized, oxicode::config::standard())
                .map(|(v, _)| v)
                .expect("deserialize");

        assert!(deserialized.is_trained());
        assert_eq!(deserialized.compression_ratio(), opq.compression_ratio());

        // Test that deserialized works correctly
        let original = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let code1 = opq.quantize(&original).expect("q1");
        let code2 = deserialized.quantize(&original).expect("q2");

        // Codes should be identical
        assert_eq!(code1.codes, code2.codes);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Tests for INT8 / Binary quantization
    // ─────────────────────────────────────────────────────────────────────────

    /// Helper: cosine similarity of two f32 slices.
    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-9 || nb < 1e-9 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        // For vectors with moderate dynamic range, max reconstruction error < 0.01
        let v: Vec<f32> = (0..16).map(|i| i as f32 * 0.01).collect(); // [0.0, 0.01, ..., 0.15]
        let (q, scale, zero_point) = quantize_f32_to_i8(&v);
        let restored = dequantize_i8_to_f32(&q, scale, zero_point);

        assert_eq!(restored.len(), v.len());
        let max_err = v
            .iter()
            .zip(restored.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_err < 0.01, "max reconstruction error {max_err} >= 0.01");
    }

    #[test]
    fn test_quantized_store_push_get() {
        let dim = 32;
        let mut store = QuantizedVectorStore::new(dim);

        // Push 10 simple unit-like vectors
        let originals: Vec<Vec<f32>> = (0..10usize)
            .map(|i| {
                let v: Vec<f32> = (0..dim)
                    .map(|d| ((i * dim + d) as f32) * 0.001 + 1.0)
                    .collect();
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter().map(|x| x / norm).collect()
            })
            .collect();
        for (i, v_norm) in originals.iter().enumerate() {
            let id = store.push(v_norm);
            assert_eq!(id, i);
        }

        assert_eq!(store.len(), 10);
        assert!(!store.is_empty());
        assert_eq!(store.dim(), dim);

        // Dequantize and verify cosine similarity > 0.99 with original
        for (i, original) in originals.iter().enumerate() {
            let recovered = store.get(i).expect("id must be valid");
            let sim = cosine_sim(original, &recovered);
            assert!(sim > 0.99, "cosine similarity {sim} <= 0.99 for vector {i}");
        }

        // Out-of-range should return None
        assert!(store.get(10).is_none());
    }

    #[test]
    fn test_quantized_store_bytes_per_vector() {
        let dim = 384;
        let store = QuantizedVectorStore::new(dim);
        // dim * 1 byte (i8) per vector = 384 bytes
        assert_eq!(store.bytes_per_vector(), 384.0);
        // Compare to f32 which would be 384 * 4 = 1536 bytes
        let f32_bytes = dim as f64 * 4.0;
        assert!(store.bytes_per_vector() < f32_bytes / 3.0);
    }

    #[test]
    fn test_binary_store_hamming() {
        let dim = 128;
        let mut store = BinaryVectorStore::new(dim);

        // "High" vector: alternating +2 / -2 so mean = 0.
        // Bits set where val >= mean (0): the even indices (val = +2).
        let high: Vec<f32> = (0..dim)
            .map(|i| if i % 2 == 0 { 2.0_f32 } else { -2.0_f32 })
            .collect();
        let id_high = store.push(&high);

        // "Low" vector: negation of "high", also mean = 0.
        // These are the COMPLEMENT of the high bits → max Hamming distance.
        let low: Vec<f32> = high.iter().map(|x| -x).collect();
        let id_low = store.push(&low);

        // Distance to itself must be 0
        assert_eq!(store.hamming_distance(id_high, id_high), 0);
        assert_eq!(store.hamming_distance(id_low, id_low), 0);

        // Negated vector: all bits flipped → max Hamming distance = dim
        assert_eq!(
            store.hamming_distance(id_high, id_low),
            dim as u32,
            "negated vectors should have max hamming distance"
        );

        // Out-of-range returns u32::MAX
        assert_eq!(store.hamming_distance(0, 99), u32::MAX);
    }

    #[test]
    fn test_binary_store_bytes_per_vector() {
        let dim = 384;
        let store = BinaryVectorStore::new(dim);
        // ceil(384/64) = 6 words, 6 * 8 = 48 bytes
        let expected_words = dim.div_ceil(64); // 6
        let expected_bytes = (expected_words * 8) as f64; // 48
        assert_eq!(store.bytes_per_vector(), expected_bytes);
        assert_eq!(store.bytes_per_vector(), 48.0);
    }
}
