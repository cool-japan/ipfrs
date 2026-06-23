//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Cosine similarity between two equal-length vectors.
/// Returns `0.0` if either vector has zero magnitude.
#[inline]
pub(super) fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}
/// Cosine *distance* (1 – similarity), clamped to [0, 1].
#[inline]
pub(super) fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    (1.0 - cosine_similarity(a, b)).clamp(0.0, 1.0)
}
/// Minimal xorshift64 PRNG used for synthetic timestamps when the platform
/// does not expose a wall clock (also useful in tests).
#[inline]
pub(super) fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}
