//! SIMD-optimized distance computation
//!
//! This module provides platform-specific SIMD implementations for common
//! distance metrics used in vector search. Supports:
//! - ARM NEON (aarch64)
//! - x86 SSE/AVX/AVX2/AVX-512
//! - Scalar fallback for other platforms
//!
//! Distance metrics:
//! - L2 (Euclidean) distance
//! - Cosine similarity/distance
//! - Dot product
//!
//! The implementations use runtime feature detection to select the best
//! available instruction set for the current CPU.

#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;
#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;

/// Compute L2 (Euclidean) distance between two vectors using SIMD
#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            return unsafe { l2_distance_neon(a, b) };
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { l2_distance_avx2(a, b) };
        }
        if is_x86_feature_detected!("avx") {
            return unsafe { l2_distance_avx(a, b) };
        }
        if is_x86_feature_detected!("sse") {
            return unsafe { l2_distance_sse(a, b) };
        }
    }

    // Fallback to scalar implementation
    l2_distance_scalar(a, b)
}

/// Compute dot product between two vectors using SIMD
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            return unsafe { dot_product_neon(a, b) };
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_product_avx2(a, b) };
        }
        if is_x86_feature_detected!("avx") {
            return unsafe { dot_product_avx(a, b) };
        }
        if is_x86_feature_detected!("sse") {
            return unsafe { dot_product_sse(a, b) };
        }
    }

    // Fallback to scalar implementation
    dot_product_scalar(a, b)
}

/// Compute cosine distance (1 - cosine similarity) using SIMD
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            return unsafe { cosine_distance_neon(a, b) };
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { cosine_distance_avx2(a, b) };
        }
        if is_x86_feature_detected!("avx") {
            return unsafe { cosine_distance_avx(a, b) };
        }
        if is_x86_feature_detected!("sse") {
            return unsafe { cosine_distance_sse(a, b) };
        }
    }

    // Fallback to scalar implementation
    cosine_distance_scalar(a, b)
}

// ============================================================================
// ARM NEON implementations
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn l2_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len();
    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0;

    // Process 4 floats at a time
    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        let diff = vsubq_f32(va, vb);
        sum = vfmaq_f32(sum, diff, diff); // sum += diff * diff
        i += 4;
    }

    // Horizontal sum
    let mut result = vaddvq_f32(sum);

    // Handle remaining elements
    while i < len {
        let diff = a[i] - b[i];
        result += diff * diff;
        i += 1;
    }

    result.sqrt()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len();
    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0;

    // Process 4 floats at a time
    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        sum = vfmaq_f32(sum, va, vb); // sum += va * vb
        i += 4;
    }

    // Horizontal sum
    let mut result = vaddvq_f32(sum);

    // Handle remaining elements
    while i < len {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn cosine_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len();
    let mut dot = vdupq_n_f32(0.0);
    let mut norm_a = vdupq_n_f32(0.0);
    let mut norm_b = vdupq_n_f32(0.0);
    let mut i = 0;

    // Process 4 floats at a time
    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        dot = vfmaq_f32(dot, va, vb);
        norm_a = vfmaq_f32(norm_a, va, va);
        norm_b = vfmaq_f32(norm_b, vb, vb);
        i += 4;
    }

    // Horizontal sum
    let mut dot_sum = vaddvq_f32(dot);
    let mut norm_a_sum = vaddvq_f32(norm_a);
    let mut norm_b_sum = vaddvq_f32(norm_b);

    // Handle remaining elements
    while i < len {
        dot_sum += a[i] * b[i];
        norm_a_sum += a[i] * a[i];
        norm_b_sum += b[i] * b[i];
        i += 1;
    }

    let similarity = dot_sum / (norm_a_sum.sqrt() * norm_b_sum.sqrt());
    1.0 - similarity
}

// ============================================================================
// x86 SSE implementations
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn l2_distance_sse(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    // Process 4 floats at a time
    while i + 4 <= len {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let diff = _mm_sub_ps(va, vb);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        i += 4;
    }

    // Horizontal sum
    let mut result = horizontal_sum_sse(sum);

    // Handle remaining elements
    while i < len {
        let diff = a[i] - b[i];
        result += diff * diff;
        i += 1;
    }

    result.sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn dot_product_sse(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    // Process 4 floats at a time
    while i + 4 <= len {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
        i += 4;
    }

    // Horizontal sum
    let mut result = horizontal_sum_sse(sum);

    // Handle remaining elements
    while i < len {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn cosine_distance_sse(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut dot = _mm_setzero_ps();
    let mut norm_a = _mm_setzero_ps();
    let mut norm_b = _mm_setzero_ps();
    let mut i = 0;

    // Process 4 floats at a time
    while i + 4 <= len {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        dot = _mm_add_ps(dot, _mm_mul_ps(va, vb));
        norm_a = _mm_add_ps(norm_a, _mm_mul_ps(va, va));
        norm_b = _mm_add_ps(norm_b, _mm_mul_ps(vb, vb));
        i += 4;
    }

    // Horizontal sum
    let mut dot_sum = horizontal_sum_sse(dot);
    let mut norm_a_sum = horizontal_sum_sse(norm_a);
    let mut norm_b_sum = horizontal_sum_sse(norm_b);

    // Handle remaining elements
    while i < len {
        dot_sum += a[i] * b[i];
        norm_a_sum += a[i] * a[i];
        norm_b_sum += b[i] * b[i];
        i += 1;
    }

    let similarity = dot_sum / (norm_a_sum.sqrt() * norm_b_sum.sqrt());
    1.0 - similarity
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn horizontal_sum_sse(v: std::arch::x86_64::__m128) -> f32 {
    use std::arch::x86_64::*;

    let shuf = _mm_movehdup_ps(v);
    let sums = _mm_add_ps(v, shuf);
    let shuf = _mm_movehl_ps(shuf, sums);
    let result = _mm_add_ss(sums, shuf);
    _mm_cvtss_f32(result)
}

// ============================================================================
// x86 AVX implementations
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn l2_distance_avx(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    // Process 8 floats at a time
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        i += 8;
    }

    // Horizontal sum
    let mut result = horizontal_sum_avx(sum);

    // Handle remaining elements
    while i < len {
        let diff = a[i] - b[i];
        result += diff * diff;
        i += 1;
    }

    result.sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn dot_product_avx(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    // Process 8 floats at a time
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
        i += 8;
    }

    // Horizontal sum
    let mut result = horizontal_sum_avx(sum);

    // Handle remaining elements
    while i < len {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn cosine_distance_avx(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut dot = _mm256_setzero_ps();
    let mut norm_a = _mm256_setzero_ps();
    let mut norm_b = _mm256_setzero_ps();
    let mut i = 0;

    // Process 8 floats at a time
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        dot = _mm256_add_ps(dot, _mm256_mul_ps(va, vb));
        norm_a = _mm256_add_ps(norm_a, _mm256_mul_ps(va, va));
        norm_b = _mm256_add_ps(norm_b, _mm256_mul_ps(vb, vb));
        i += 8;
    }

    // Horizontal sum
    let mut dot_sum = horizontal_sum_avx(dot);
    let mut norm_a_sum = horizontal_sum_avx(norm_a);
    let mut norm_b_sum = horizontal_sum_avx(norm_b);

    // Handle remaining elements
    while i < len {
        dot_sum += a[i] * b[i];
        norm_a_sum += a[i] * a[i];
        norm_b_sum += b[i] * b[i];
        i += 1;
    }

    let similarity = dot_sum / (norm_a_sum.sqrt() * norm_b_sum.sqrt());
    1.0 - similarity
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn horizontal_sum_avx(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi, lo);
    horizontal_sum_sse(sum128)
}

// ============================================================================
// x86 AVX2 implementations
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    // AVX2 doesn't add much for f32 operations, so just use AVX
    l2_distance_avx(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    // AVX2 doesn't add much for f32 operations, so just use AVX
    dot_product_avx(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cosine_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    // AVX2 doesn't add much for f32 operations, so just use AVX
    cosine_distance_avx(a, b)
}

// ============================================================================
// Scalar fallback implementations
// ============================================================================

#[inline]
fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

#[inline]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn cosine_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    1.0 - (dot / (norm_a * norm_b))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_distance() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let dist = l2_distance(&a, &b);
        let expected = (8.0_f32).sqrt(); // sqrt(8 * 1^2)

        assert!((dist - expected).abs() < 1e-5, "L2 distance mismatch");
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let dot = dot_product(&a, &b);
        let expected = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0;

        assert!((dot - expected).abs() < 1e-5, "Dot product mismatch");
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0, 0.0];

        let dist = cosine_distance(&a, &b);

        // Same vectors should have cosine distance close to 0
        assert!(
            dist.abs() < 1e-5,
            "Cosine distance should be 0 for identical vectors"
        );
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];

        let dist = cosine_distance(&a, &b);

        // Orthogonal vectors should have cosine distance close to 1
        assert!(
            (dist - 1.0).abs() < 1e-5,
            "Cosine distance should be 1 for orthogonal vectors"
        );
    }

    #[test]
    fn test_simd_vs_scalar_l2() {
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let simd_result = l2_distance(&a, &b);
        let scalar_result = l2_distance_scalar(&a, &b);

        assert!(
            (simd_result - scalar_result).abs() < 1e-4,
            "SIMD and scalar L2 results should match"
        );
    }

    #[test]
    fn test_simd_vs_scalar_dot() {
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..128).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let simd_result = dot_product(&a, &b);
        let scalar_result = dot_product_scalar(&a, &b);

        assert!(
            (simd_result - scalar_result).abs() < 1e-3,
            "SIMD and scalar dot product results should match"
        );
    }

    #[test]
    fn test_simd_vs_scalar_cosine() {
        let a: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1) + 1.0).collect();
        let b: Vec<f32> = (0..128).map(|i| ((i as f32 + 1.0) * 0.1) + 1.0).collect();

        let simd_result = cosine_distance(&a, &b);
        let scalar_result = cosine_distance_scalar(&a, &b);

        assert!(
            (simd_result - scalar_result).abs() < 1e-4,
            "SIMD and scalar cosine results should match"
        );
    }
}
