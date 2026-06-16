//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{FuzzyError, FuzzyExpr, FuzzyVariable};

/// Determine the universe bounds [min, max] from the sets of an output variable.
pub(super) fn universe_bounds(var: &FuzzyVariable) -> (f64, f64) {
    if var.sets.is_empty() {
        return (0.0, 1.0);
    }
    let mut u_min = f64::INFINITY;
    let mut u_max = f64::NEG_INFINITY;
    for s in &var.sets {
        if s.universe_min < u_min {
            u_min = s.universe_min;
        }
        if s.universe_max > u_max {
            u_max = s.universe_max;
        }
    }
    if u_min >= u_max {
        (u_min, u_min + 1.0)
    } else {
        (u_min, u_max)
    }
}
/// Check whether an expression references `var_name` in its consequent position.
pub(super) fn expr_targets_var(expr: &FuzzyExpr, var_name: &str) -> bool {
    match expr {
        FuzzyExpr::Is { var, .. } => var == var_name,
        FuzzyExpr::And(l, r) | FuzzyExpr::Or(l, r) => {
            expr_targets_var(l, var_name) || expr_targets_var(r, var_name)
        }
        FuzzyExpr::Not(inner) | FuzzyExpr::Very(inner) | FuzzyExpr::Somewhat(inner) => {
            expr_targets_var(inner, var_name)
        }
    }
}
/// Extract the set name from a consequent expression targeting `var_name`.
/// Returns `None` if the expression is not a direct `Is { var_name, set }`.
pub(super) fn consequent_set_name(expr: &FuzzyExpr, var_name: &str) -> Option<String> {
    match expr {
        FuzzyExpr::Is { var, set } if var == var_name => Some(set.clone()),
        _ => None,
    }
}
/// Name of the output set with highest membership at `x`.
pub(super) fn dominant_set_name(var: &FuzzyVariable, x: f64) -> String {
    var.sets
        .iter()
        .max_by(|a, b| {
            a.mf.evaluate(x)
                .partial_cmp(&b.mf.evaluate(x))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|s| s.name.clone())
        .unwrap_or_default()
}
/// Centroid (centre-of-gravity) defuzzification.
pub(super) fn centroid(agg: &[f64], u_min: f64, step: f64) -> Result<f64, FuzzyError> {
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (i, &mu) in agg.iter().enumerate() {
        let x = u_min + i as f64 * step;
        num += x * mu;
        den += mu;
    }
    if den < f64::EPSILON {
        return Err(FuzzyError::DefuzzFailed(
            "centroid: all membership values are zero".to_string(),
        ));
    }
    Ok(num / den)
}
/// Bisector defuzzification — x where cumulative area = half total area.
pub(super) fn bisector(agg: &[f64], u_min: f64, step: f64) -> Result<f64, FuzzyError> {
    let total: f64 = agg.iter().sum();
    if total < f64::EPSILON {
        return Err(FuzzyError::DefuzzFailed(
            "bisector: total area is zero".to_string(),
        ));
    }
    let half = total / 2.0;
    let mut cum = 0.0_f64;
    for (i, &mu) in agg.iter().enumerate() {
        cum += mu;
        if cum >= half {
            return Ok(u_min + i as f64 * step);
        }
    }
    Ok(u_min + (agg.len() - 1) as f64 * step)
}
/// Mean of maxima defuzzification.
pub(super) fn mean_of_maxima(agg: &[f64], u_min: f64, step: f64) -> Result<f64, FuzzyError> {
    let max_val = agg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val < f64::EPSILON {
        return Err(FuzzyError::DefuzzFailed(
            "mean_of_maxima: maximum membership is zero".to_string(),
        ));
    }
    let mut sum = 0.0_f64;
    let mut count = 0usize;
    for (i, &mu) in agg.iter().enumerate() {
        if (mu - max_val).abs() < 1e-9 {
            sum += u_min + i as f64 * step;
            count += 1;
        }
    }
    if count == 0 {
        return Err(FuzzyError::DefuzzFailed(
            "mean_of_maxima: no maximum found".to_string(),
        ));
    }
    Ok(sum / count as f64)
}
/// Largest of maxima defuzzification.
pub(super) fn largest_of_maxima(agg: &[f64], u_min: f64, step: f64) -> Result<f64, FuzzyError> {
    let max_val = agg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val < f64::EPSILON {
        return Err(FuzzyError::DefuzzFailed(
            "largest_of_maxima: maximum membership is zero".to_string(),
        ));
    }
    for (i, &mu) in agg.iter().enumerate().rev() {
        if (mu - max_val).abs() < 1e-9 {
            return Ok(u_min + i as f64 * step);
        }
    }
    Err(FuzzyError::DefuzzFailed(
        "largest_of_maxima: no maximum found".to_string(),
    ))
}
/// Smallest of maxima defuzzification.
pub(super) fn smallest_of_maxima(agg: &[f64], u_min: f64, step: f64) -> Result<f64, FuzzyError> {
    let max_val = agg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val < f64::EPSILON {
        return Err(FuzzyError::DefuzzFailed(
            "smallest_of_maxima: maximum membership is zero".to_string(),
        ));
    }
    for (i, &mu) in agg.iter().enumerate() {
        if (mu - max_val).abs() < 1e-9 {
            return Ok(u_min + i as f64 * step);
        }
    }
    Err(FuzzyError::DefuzzFailed(
        "smallest_of_maxima: no maximum found".to_string(),
    ))
}
