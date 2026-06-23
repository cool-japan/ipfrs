//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Cosine similarity in [-1, 1].  Returns 0 if either vector is zero-length.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-10 || nb < 1e-10 {
        0.0
    } else {
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }
}
/// Euclidean distance between two vectors.
pub(super) fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}
/// Weighted sum of (slice, weight) pairs.  All slices must share the same length.
pub fn weighted_sum(vecs: &[(&[f64], f64)]) -> Vec<f64> {
    if vecs.is_empty() {
        return vec![];
    }
    let dim = vecs[0].0.len();
    let mut result = vec![0.0f64; dim];
    for (v, w) in vecs {
        for (r, x) in result.iter_mut().zip(v.iter()) {
            *r += x * w;
        }
    }
    result
}
/// Normalize a vector to unit length in-place.  No-op when the vector is near zero.
pub(super) fn normalize_in_place(v: &mut [f64]) {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}
pub(super) fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}
pub(super) fn xorshift_f64(state: &mut u64) -> f64 {
    (xorshift64(state) >> 11) as f64 / (1u64 << 53) as f64
}
/// MMR score for candidate `i` given already-selected items.
pub(super) fn mmr_score(
    candidates: &[(String, f64, &[f64])],
    i: usize,
    selected: &[usize],
    lambda: f64,
) -> f64 {
    let rel = candidates[i].1;
    let max_sim = max_similarity_to_selected(candidates, i, selected);
    lambda * rel - (1.0 - lambda) * max_sim
}
/// Maximum cosine similarity between candidate `i` and any selected item.
pub(super) fn max_similarity_to_selected(
    candidates: &[(String, f64, &[f64])],
    i: usize,
    selected: &[usize],
) -> f64 {
    if selected.is_empty() {
        return 0.0;
    }
    selected
        .iter()
        .map(|&s| cosine_similarity(candidates[i].2, candidates[s].2))
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0)
}
/// DPP kernel value: k(i,j) = rel_i * cos(i,j) * rel_j.
pub(super) fn kernel_val(candidates: &[(String, f64, &[f64])], i: usize, j: usize) -> f64 {
    let cos = cosine_similarity(candidates[i].2, candidates[j].2);
    let cos_shifted = (cos + 1.0) / 2.0;
    candidates[i].1 * cos_shifted * candidates[j].1
}
/// Marginal gain of adding candidate `i` to the current DPP selection.
pub(super) fn dpp_marginal(
    candidates: &[(String, f64, &[f64])],
    i: usize,
    _selected: &[usize],
    l: &[Vec<f64>],
    step: usize,
) -> f64 {
    let k_ii = kernel_val(candidates, i, i);
    let l_sq: f64 = (0..step).map(|t| l[i][t] * l[i][t]).sum();
    (k_ii - l_sq).max(0.0)
}
