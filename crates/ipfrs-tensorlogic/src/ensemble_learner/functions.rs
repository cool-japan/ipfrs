//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ElBaseModel, ElError, ElSample};

#[inline]
pub(super) fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}
/// Draw a uniform f64 in [0, 1) from the PRNG state.
#[inline]
pub(super) fn xorshift_f64(state: &mut u64) -> f64 {
    (xorshift64(state) >> 11) as f64 / (1u64 << 53) as f64
}
/// Draw a usize in [0, n) from the PRNG state.
#[inline]
pub(super) fn xorshift_usize(state: &mut u64, n: usize) -> usize {
    (xorshift64(state) as usize).wrapping_rem(n)
}
/// Bootstrap sample of `n` indices from `[0, pool_size)` with replacement.
pub(super) fn bootstrap_indices(rng: &mut u64, pool_size: usize, n: usize) -> Vec<usize> {
    (0..n).map(|_| xorshift_usize(rng, pool_size)).collect()
}
/// Weighted bootstrap: draw `n` indices according to `weights`.
#[allow(dead_code)]
pub(super) fn weighted_bootstrap(rng: &mut u64, weights: &[f64], n: usize) -> Vec<usize> {
    let total: f64 = weights.iter().sum();
    let cdf: Vec<f64> = weights
        .iter()
        .scan(0.0f64, |acc, w| {
            *acc += w / total;
            Some(*acc)
        })
        .collect();
    (0..n)
        .map(|_| {
            let u = xorshift_f64(rng);
            cdf.partition_point(|&v| v < u).min(weights.len() - 1)
        })
        .collect()
}
/// Find the best decision stump for a set of (weighted) samples, restricted to
/// a candidate feature subset.
///
/// Returns `(feature_index, threshold, direction, weighted_error)`.
pub(super) fn best_stump(
    samples: &[ElSample],
    sample_weights: &[f64],
    feature_subset: &[usize],
) -> Result<(usize, f64, bool, f64), ElError> {
    let n = samples.len();
    if n == 0 {
        return Err(ElError::EmptyTrainingSet);
    }
    let n_feat = samples
        .first()
        .ok_or(ElError::EmptyTrainingSet)?
        .features
        .len();
    if n_feat == 0 {
        return Err(ElError::InvalidConfig(
            "samples must have at least one feature".to_string(),
        ));
    }
    let total_weight: f64 = sample_weights.iter().sum();
    if total_weight <= 0.0 {
        return Err(ElError::Arithmetic(
            "sample weights sum to zero".to_string(),
        ));
    }
    let mut best_err = f64::MAX;
    let mut best_feat = 0usize;
    let mut best_thresh = 0.0f64;
    let mut best_dir = true;
    for &feat_idx in feature_subset {
        let mut vals: Vec<(f64, f64, f64)> = samples
            .iter()
            .zip(sample_weights.iter())
            .map(|(s, &w)| {
                let fv = s.features.get(feat_idx).copied().unwrap_or(0.0);
                (fv, s.label, w)
            })
            .collect();
        vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for i in 0..vals.len().saturating_sub(1) {
            let thresh = (vals[i].0 + vals[i + 1].0) / 2.0;
            for &dir in &[true, false] {
                let err: f64 = vals
                    .iter()
                    .map(|(fv, label, w)| {
                        let pred = if dir { *fv <= thresh } else { *fv > thresh };
                        let pred_val: f64 = if pred { 1.0 } else { -1.0 };
                        let label_sign: f64 = if *label >= 0.0 { 1.0 } else { -1.0 };
                        if (pred_val - label_sign).abs() > 1e-9 {
                            *w
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>()
                    / total_weight;
                if err < best_err {
                    best_err = err;
                    best_feat = feat_idx;
                    best_thresh = thresh;
                    best_dir = dir;
                }
            }
        }
        for &dir in &[true, false] {
            let thresh = vals.first().map(|v| v.0 - 1.0).unwrap_or(-1.0);
            let err: f64 = vals
                .iter()
                .map(|(fv, label, w)| {
                    let pred = if dir { *fv <= thresh } else { *fv > thresh };
                    let pred_val: f64 = if pred { 1.0 } else { -1.0 };
                    let label_sign: f64 = if *label >= 0.0 { 1.0 } else { -1.0 };
                    if (pred_val - label_sign).abs() > 1e-9 {
                        *w
                    } else {
                        0.0
                    }
                })
                .sum::<f64>()
                / total_weight;
            if err < best_err {
                best_err = err;
                best_feat = feat_idx;
                best_thresh = thresh;
                best_dir = dir;
            }
        }
    }
    Ok((best_feat, best_thresh, best_dir, best_err))
}
/// Fit a decision stump to continuous residuals (for gradient boosting).
/// Returns `(feature_index, threshold, direction, leaf_pos, leaf_neg)`.
///
/// `leaf_pos` is the mean residual for samples where the stump predicts +1,
/// `leaf_neg` is the mean residual for samples where the stump predicts -1.
pub(super) fn best_regression_stump(
    samples: &[ElSample],
    residuals: &[f64],
    feature_subset: &[usize],
) -> Result<(usize, f64, bool, f64, f64), ElError> {
    let n = samples.len();
    if n == 0 {
        return Err(ElError::EmptyTrainingSet);
    }
    let n_feat = samples
        .first()
        .ok_or(ElError::EmptyTrainingSet)?
        .features
        .len();
    if n_feat == 0 {
        return Err(ElError::InvalidConfig(
            "samples must have at least one feature".to_string(),
        ));
    }
    let mut best_mse = f64::MAX;
    let mut best_feat = 0usize;
    let mut best_thresh = 0.0f64;
    let mut best_dir = true;
    for &feat_idx in feature_subset {
        let mut vals: Vec<(f64, f64)> = samples
            .iter()
            .zip(residuals.iter())
            .map(|(s, &r)| {
                let fv = s.features.get(feat_idx).copied().unwrap_or(0.0);
                (fv, r)
            })
            .collect();
        vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for i in 0..vals.len().saturating_sub(1) {
            let thresh = (vals[i].0 + vals[i + 1].0) / 2.0;
            for &dir in &[true, false] {
                let (mut sum_pos, mut cnt_pos) = (0.0f64, 0usize);
                let (mut sum_neg, mut cnt_neg) = (0.0f64, 0usize);
                for (fv, r) in &vals {
                    if (dir && *fv <= thresh) || (!dir && *fv > thresh) {
                        sum_pos += r;
                        cnt_pos += 1;
                    } else {
                        sum_neg += r;
                        cnt_neg += 1;
                    }
                }
                let mean_pos = if cnt_pos > 0 {
                    sum_pos / cnt_pos as f64
                } else {
                    0.0
                };
                let mean_neg = if cnt_neg > 0 {
                    sum_neg / cnt_neg as f64
                } else {
                    0.0
                };
                let mse: f64 = vals
                    .iter()
                    .map(|(fv, r)| {
                        let pred = if (dir && *fv <= thresh) || (!dir && *fv > thresh) {
                            mean_pos
                        } else {
                            mean_neg
                        };
                        let d = r - pred;
                        d * d
                    })
                    .sum::<f64>();
                if mse < best_mse {
                    best_mse = mse;
                    best_feat = feat_idx;
                    best_thresh = thresh;
                    best_dir = dir;
                }
            }
        }
    }
    let (mut sum_pos, mut cnt_pos) = (0.0f64, 0usize);
    let (mut sum_neg, mut cnt_neg) = (0.0f64, 0usize);
    for (s, &r) in samples.iter().zip(residuals.iter()) {
        let fv = s.features.get(best_feat).copied().unwrap_or(0.0);
        if (best_dir && fv <= best_thresh) || (!best_dir && fv > best_thresh) {
            sum_pos += r;
            cnt_pos += 1;
        } else {
            sum_neg += r;
            cnt_neg += 1;
        }
    }
    let leaf_pos = if cnt_pos > 0 {
        sum_pos / cnt_pos as f64
    } else {
        0.0
    };
    let leaf_neg = if cnt_neg > 0 {
        sum_neg / cnt_neg as f64
    } else {
        0.0
    };
    Ok((best_feat, best_thresh, best_dir, leaf_pos, leaf_neg))
}
/// Fit a simple perceptron (one gradient descent pass per sample).
pub(super) fn fit_perceptron(
    samples: &[ElSample],
    n_features: usize,
    rng: &mut u64,
    lr: f64,
) -> ElBaseModel {
    let mut weights: Vec<f64> = (0..n_features)
        .map(|_| (xorshift_f64(rng) - 0.5) * 0.01)
        .collect();
    let mut bias = 0.0f64;
    for s in samples {
        let score: f64 = s
            .features
            .iter()
            .zip(weights.iter())
            .map(|(x, w)| x * w)
            .sum::<f64>()
            + bias;
        let label_sign: f64 = if s.label >= 0.0 { 1.0 } else { -1.0 };
        let pred_sign: f64 = if score >= 0.0 { 1.0 } else { -1.0 };
        if (pred_sign - label_sign).abs() > 1e-9 {
            for (w, x) in weights.iter_mut().zip(s.features.iter()) {
                *w += lr * label_sign * x;
            }
            bias += lr * label_sign;
        }
    }
    ElBaseModel::Perceptron {
        weights,
        bias,
        weight: 1.0,
    }
}
