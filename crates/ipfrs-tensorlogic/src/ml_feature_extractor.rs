//! ML preprocessing `FeatureExtractor` — composable feature engineering pipeline.
//!
//! Provides a production-grade, composable pipeline for feature engineering including
//! scaling, encoding, imputation, and polynomial expansion. Designed to transform a
//! `HashMap<String, FeatureValue>` input into a flat `Vec<f64>` suitable for ML models.
//!
//! # Example
//!
//! ```
//! use ipfrs_tensorlogic::ml_feature_extractor::{
//!     FeatureExtractor, FeatureSpec, FeatureTransform, FeatureValue,
//! };
//! use std::collections::HashMap;
//!
//! let mut extractor = FeatureExtractor::new();
//! extractor.add_spec(FeatureSpec {
//!     name: "age".to_string(),
//!     transforms: vec![
//!         FeatureTransform::StandardScaler { mean: 30.0, std: 10.0 },
//!     ],
//! });
//!
//! let mut input = HashMap::new();
//! input.insert("age".to_string(), FeatureValue::Float(40.0));
//!
//! let result = extractor.extract(&input).expect("example: should succeed in docs");
//! assert_eq!(result.values.len(), 1);
//! // (40 - 30) / 10 = 1.0
//! assert!((result.values[0] - 1.0).abs() < 1e-12);
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the feature extraction pipeline.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum FeatureError {
    /// A required feature was not found and no imputer was configured.
    #[error("missing required feature: {0}")]
    MissingFeature(String),

    /// A transform received a value of the wrong type.
    #[error("type mismatch for feature '{feature}': expected {expected}, got {got}")]
    TypeMismatch {
        feature: String,
        expected: String,
        got: String,
    },

    /// The transform parameters are invalid (e.g., degree = 0).
    #[error("invalid transform: {0}")]
    InvalidTransform(String),

    /// The input batch or data slice is empty.
    #[error("empty input")]
    EmptyInput,
}

// ─────────────────────────────────────────────────────────────────────────────
// Core value type
// ─────────────────────────────────────────────────────────────────────────────

/// A polymorphic feature value that flows through the transform chain.
#[derive(Clone, Debug, PartialEq)]
pub enum FeatureValue {
    /// Continuous floating-point scalar.
    Float(f64),
    /// Integer scalar — coerced to `f64` for numeric transforms.
    Integer(i64),
    /// Boolean flag — coerced to `1.0` / `0.0` for numeric transforms.
    Boolean(bool),
    /// Nominal string category — used by `OneHotEncode` and `ImputeMode`.
    Categorical(String),
    /// Absent / unknown value — consumed by imputer transforms.
    Missing,
}

impl FeatureValue {
    /// Return the type name as a human-readable string (used in error messages).
    pub fn type_name(&self) -> &'static str {
        match self {
            FeatureValue::Float(_) => "Float",
            FeatureValue::Integer(_) => "Integer",
            FeatureValue::Boolean(_) => "Boolean",
            FeatureValue::Categorical(_) => "Categorical",
            FeatureValue::Missing => "Missing",
        }
    }

    /// Coerce to `f64` if the variant is numeric.  Returns `None` for `Categorical`
    /// and `Missing`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            FeatureValue::Float(v) => Some(*v),
            FeatureValue::Integer(v) => Some(*v as f64),
            FeatureValue::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
            FeatureValue::Categorical(_) | FeatureValue::Missing => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Transform enum
// ─────────────────────────────────────────────────────────────────────────────

/// A single reversible or irreversible transform applied to a `FeatureValue`.
#[derive(Clone, Debug, PartialEq)]
pub enum FeatureTransform {
    /// Z-score normalisation: `(x - mean) / std`.  Output is `0.0` when `std ≈ 0`.
    StandardScaler { mean: f64, std: f64 },

    /// Min-max normalisation: `(x - min) / (max - min)`.  Output is `0.0` when range ≈ 0.
    MinMaxScaler { min: f64, max: f64 },

    /// Signed log1p: `ln(1 + |x|) × sign(x)`.
    Log1p,

    /// Signed square-root: `√|x| × sign(x)`.
    Sqrt,

    /// Clamp the value to `[lo, hi]`.
    Clip { lo: f64, hi: f64 },

    /// One-hot encode a `Categorical` value — produces one `f64` per category.
    /// Unknown categories produce an all-zero vector.
    OneHotEncode { categories: Vec<String> },

    /// Threshold binarisation: `1.0` if `x > threshold` else `0.0`.
    Binarize { threshold: f64 },

    /// Polynomial expansion to `degree`: `[x, x², …, xᵈ]`.
    PolynomialFeatures { degree: u32 },

    /// Replace `Missing` with a pre-computed numeric mean.
    ImputeMean { mean: f64 },

    /// Replace a `Missing` categorical with the pre-computed mode string.
    ImputeMode { mode: String },
}

// ─────────────────────────────────────────────────────────────────────────────
// Spec and output types
// ─────────────────────────────────────────────────────────────────────────────

/// An ordered list of transforms to apply to a single named input feature.
#[derive(Clone, Debug)]
pub struct FeatureSpec {
    /// Name of the feature to look up in the input map.
    pub name: String,
    /// Sequential transforms to apply.  Applied in order, left-to-right.
    pub transforms: Vec<FeatureTransform>,
}

/// The fully-expanded output of one extraction call.
#[derive(Clone, Debug)]
pub struct ExtractedFeatures {
    /// Output feature names — one per scalar value.  OneHot expands to
    /// `{name}_cat_{i}` entries.
    pub feature_names: Vec<String>,
    /// Flat output vector aligned with `feature_names`.
    pub values: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal representation used during transform evaluation
// ─────────────────────────────────────────────────────────────────────────────

/// Intermediate value during pipeline evaluation — either a scalar or the
/// expanded binary vector produced by `OneHotEncode`.
#[derive(Clone, Debug)]
enum PipelineValue {
    Scalar(FeatureValue),
    OneHot(Vec<f64>),
}

// ─────────────────────────────────────────────────────────────────────────────
// Stats
// ─────────────────────────────────────────────────────────────────────────────

/// Cumulative statistics for a [`FeatureExtractor`] instance.
#[derive(Clone, Debug, Default)]
pub struct FePipelineStats {
    /// Total number of named output features (sum of per-spec output dims).
    pub total_features: usize,
    /// Total number of `extract` calls made (including batch calls).
    pub total_extractions: u64,
    /// Running average of the output dimension across all extractions.
    pub avg_output_dim: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// FeatureExtractor
// ─────────────────────────────────────────────────────────────────────────────

/// Composable feature engineering pipeline.
///
/// Holds an ordered list of [`FeatureSpec`]s.  Each spec names one input
/// feature and chains transforms that convert it to one or more `f64` outputs.
///
/// # Thread-safety
///
/// `FeatureExtractor` is not `Send`/`Sync` by default because of its internal
/// `AtomicU64` counter — if you need concurrent access, wrap it in a `Mutex`.
pub struct FeatureExtractor {
    specs: Vec<FeatureSpec>,
    total_extractions: AtomicU64,
    extraction_dim_sum: AtomicU64,
}

impl FeatureExtractor {
    /// Create an empty extractor with no specs.
    pub fn new() -> Self {
        Self {
            specs: Vec::new(),
            total_extractions: AtomicU64::new(0),
            extraction_dim_sum: AtomicU64::new(0),
        }
    }

    /// Append a spec to the pipeline (builder-style).
    pub fn add_spec(&mut self, spec: FeatureSpec) -> &mut Self {
        self.specs.push(spec);
        self
    }

    /// Compute the total output dimension from spec metadata alone.
    ///
    /// For `OneHotEncode { categories }` the contribution is
    /// `categories.len()`.  For `PolynomialFeatures { degree }` it is
    /// `degree as usize`.  All other specs contribute `1`.
    pub fn output_dim(&self) -> usize {
        self.specs.iter().map(spec_output_dim).sum()
    }

    /// List all output feature names (expanded for OneHot and Polynomial).
    pub fn feature_names(&self) -> Vec<String> {
        let mut names = Vec::with_capacity(self.output_dim());
        for spec in &self.specs {
            push_feature_names(spec, &mut names);
        }
        names
    }

    /// Extract features from a single `input` map.
    ///
    /// Returns an error if a required feature is missing (no imputer configured)
    /// or a type mismatch prevents a transform from running.
    pub fn extract(
        &self,
        input: &HashMap<String, FeatureValue>,
    ) -> Result<ExtractedFeatures, FeatureError> {
        let mut feature_names: Vec<String> = Vec::with_capacity(self.output_dim());
        let mut values: Vec<f64> = Vec::with_capacity(self.output_dim());

        for spec in &self.specs {
            let raw = input
                .get(&spec.name)
                .cloned()
                .unwrap_or(FeatureValue::Missing);

            let pipeline_val = apply_transforms(spec, raw)?;

            match pipeline_val {
                PipelineValue::Scalar(fv) => {
                    let v = scalar_to_f64(&spec.name, fv)?;
                    feature_names.push(spec.name.clone());
                    values.push(v);
                }
                PipelineValue::OneHot(vec) => {
                    for (i, v) in vec.into_iter().enumerate() {
                        feature_names.push(format!("{}_cat_{}", spec.name, i));
                        values.push(v);
                    }
                }
            }
        }

        // Update statistics atomically.
        let dim = values.len() as u64;
        self.total_extractions.fetch_add(1, Ordering::Relaxed);
        self.extraction_dim_sum.fetch_add(dim, Ordering::Relaxed);

        Ok(ExtractedFeatures {
            feature_names,
            values,
        })
    }

    /// Extract features from a batch of input maps.
    ///
    /// Returns `Err(FeatureError::EmptyInput)` when `inputs` is empty.
    pub fn extract_batch(
        &self,
        inputs: &[HashMap<String, FeatureValue>],
    ) -> Result<Vec<ExtractedFeatures>, FeatureError> {
        if inputs.is_empty() {
            return Err(FeatureError::EmptyInput);
        }
        inputs.iter().map(|inp| self.extract(inp)).collect()
    }

    /// Return a snapshot of accumulated statistics.
    pub fn stats(&self) -> FePipelineStats {
        let total_extractions = self.total_extractions.load(Ordering::Relaxed);
        let dim_sum = self.extraction_dim_sum.load(Ordering::Relaxed);
        let avg_output_dim = if total_extractions == 0 {
            0.0
        } else {
            dim_sum as f64 / total_extractions as f64
        };
        FePipelineStats {
            total_features: self.output_dim(),
            total_extractions,
            avg_output_dim,
        }
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fitting helpers (stateless functions)
// ─────────────────────────────────────────────────────────────────────────────

/// Fit a `StandardScaler` from a slice of observed values.
///
/// Computes population mean and std-dev.  When `values` is empty or has
/// zero variance the returned `std` field is `0.0`.
pub fn fit_standard_scaler(values: &[f64]) -> FeatureTransform {
    if values.is_empty() {
        return FeatureTransform::StandardScaler {
            mean: 0.0,
            std: 0.0,
        };
    }
    let n = values.len() as f64;
    let mean = values.iter().copied().sum::<f64>() / n;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    FeatureTransform::StandardScaler { mean, std }
}

/// Fit a `MinMaxScaler` from a slice of observed values.
///
/// Uses the global min and max.  When `values` is empty both fields are `0.0`.
pub fn fit_minmax_scaler(values: &[f64]) -> FeatureTransform {
    if values.is_empty() {
        return FeatureTransform::MinMaxScaler { min: 0.0, max: 0.0 };
    }
    let mut min = values[0];
    let mut max = values[0];
    for &v in &values[1..] {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    FeatureTransform::MinMaxScaler { min, max }
}

/// Fit a `OneHotEncode` transform from an observed set of category strings.
///
/// Deduplicates and sorts the categories for deterministic ordering.
pub fn fit_onehot(categories: &[String]) -> FeatureTransform {
    let mut cats: Vec<String> = categories.to_vec();
    cats.sort();
    cats.dedup();
    FeatureTransform::OneHotEncode { categories: cats }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Apply all transforms in `spec` to `value`, returning the pipeline result.
fn apply_transforms(
    spec: &FeatureSpec,
    value: FeatureValue,
) -> Result<PipelineValue, FeatureError> {
    let mut current = PipelineValue::Scalar(value);

    for transform in &spec.transforms {
        current = apply_one_transform(spec, transform, current)?;
    }

    Ok(current)
}

/// Apply a single transform to the current pipeline value.
fn apply_one_transform(
    spec: &FeatureSpec,
    transform: &FeatureTransform,
    current: PipelineValue,
) -> Result<PipelineValue, FeatureError> {
    match transform {
        // ── Imputers (only consume Missing) ────────────────────────────────
        FeatureTransform::ImputeMean { mean } => match current {
            PipelineValue::Scalar(FeatureValue::Missing) => {
                Ok(PipelineValue::Scalar(FeatureValue::Float(*mean)))
            }
            other => Ok(other),
        },

        FeatureTransform::ImputeMode { mode } => match current {
            PipelineValue::Scalar(FeatureValue::Missing) => Ok(PipelineValue::Scalar(
                FeatureValue::Categorical(mode.clone()),
            )),
            other => Ok(other),
        },

        // ── Numeric scalar transforms ───────────────────────────────────────
        FeatureTransform::StandardScaler { mean, std } => {
            let x = require_numeric(spec, current)?;
            let result = if std.abs() < f64::EPSILON {
                0.0
            } else {
                (x - mean) / std
            };
            Ok(PipelineValue::Scalar(FeatureValue::Float(result)))
        }

        FeatureTransform::MinMaxScaler { min, max } => {
            let x = require_numeric(spec, current)?;
            let range = max - min;
            let result = if range.abs() < f64::EPSILON {
                0.0
            } else {
                (x - min) / range
            };
            Ok(PipelineValue::Scalar(FeatureValue::Float(result)))
        }

        FeatureTransform::Log1p => {
            let x = require_numeric(spec, current)?;
            let sign = if x >= 0.0 { 1.0 } else { -1.0 };
            Ok(PipelineValue::Scalar(FeatureValue::Float(
                (1.0 + x.abs()).ln() * sign,
            )))
        }

        FeatureTransform::Sqrt => {
            let x = require_numeric(spec, current)?;
            let sign = if x >= 0.0 { 1.0 } else { -1.0 };
            Ok(PipelineValue::Scalar(FeatureValue::Float(
                x.abs().sqrt() * sign,
            )))
        }

        FeatureTransform::Clip { lo, hi } => {
            let x = require_numeric(spec, current)?;
            Ok(PipelineValue::Scalar(FeatureValue::Float(
                x.max(*lo).min(*hi),
            )))
        }

        FeatureTransform::Binarize { threshold } => {
            let x = require_numeric(spec, current)?;
            Ok(PipelineValue::Scalar(FeatureValue::Float(
                if x > *threshold { 1.0 } else { 0.0 },
            )))
        }

        FeatureTransform::PolynomialFeatures { degree } => {
            let x = require_numeric(spec, current)?;
            if *degree == 0 {
                return Err(FeatureError::InvalidTransform(
                    "PolynomialFeatures degree must be >= 1".to_string(),
                ));
            }
            // Polynomial returns a multi-value representation stored as OneHot for consistency.
            // We model it as a FeatureValue::Float after unrolling via a synthetic OneHot vec.
            let mut poly_vals = Vec::with_capacity(*degree as usize);
            let mut pow = x;
            for _ in 1..=*degree {
                poly_vals.push(pow);
                pow *= x;
            }
            Ok(PipelineValue::OneHot(poly_vals))
        }

        // ── OneHotEncode ───────────────────────────────────────────────────
        FeatureTransform::OneHotEncode { categories } => {
            let cat = require_categorical(spec, current)?;
            let one_hot: Vec<f64> = categories
                .iter()
                .map(|c| if c == &cat { 1.0 } else { 0.0 })
                .collect();
            Ok(PipelineValue::OneHot(one_hot))
        }
    }
}

/// Extract a numeric `f64` from a `PipelineValue::Scalar`.
/// Returns `FeatureError::MissingFeature` for `Missing` and `FeatureError::TypeMismatch` for
/// non-numeric types.
fn require_numeric(spec: &FeatureSpec, pv: PipelineValue) -> Result<f64, FeatureError> {
    match pv {
        PipelineValue::Scalar(fv) => match fv.as_f64() {
            Some(v) => Ok(v),
            None => {
                let type_name = fv.type_name();
                if type_name == "Missing" {
                    Err(FeatureError::MissingFeature(spec.name.clone()))
                } else {
                    Err(FeatureError::TypeMismatch {
                        feature: spec.name.clone(),
                        expected: "numeric (Float/Integer/Boolean)".to_string(),
                        got: type_name.to_string(),
                    })
                }
            }
        },
        PipelineValue::OneHot(v) => {
            // Polynomial output can flow through numeric transforms by using the first element.
            // But in practice this is an error — reject it.
            Err(FeatureError::TypeMismatch {
                feature: spec.name.clone(),
                expected: "numeric scalar".to_string(),
                got: format!("multi-value vector of length {}", v.len()),
            })
        }
    }
}

/// Extract a `String` from a `Categorical` pipeline value.
fn require_categorical(spec: &FeatureSpec, pv: PipelineValue) -> Result<String, FeatureError> {
    match pv {
        PipelineValue::Scalar(FeatureValue::Categorical(s)) => Ok(s),
        PipelineValue::Scalar(FeatureValue::Missing) => {
            Err(FeatureError::MissingFeature(spec.name.clone()))
        }
        PipelineValue::Scalar(fv) => Err(FeatureError::TypeMismatch {
            feature: spec.name.clone(),
            expected: "Categorical".to_string(),
            got: fv.type_name().to_string(),
        }),
        PipelineValue::OneHot(v) => Err(FeatureError::TypeMismatch {
            feature: spec.name.clone(),
            expected: "Categorical".to_string(),
            got: format!("multi-value vector of length {}", v.len()),
        }),
    }
}

/// Convert a final scalar `FeatureValue` to `f64` for output.
fn scalar_to_f64(name: &str, fv: FeatureValue) -> Result<f64, FeatureError> {
    match fv.as_f64() {
        Some(v) => Ok(v),
        None => {
            let type_name = fv.type_name();
            if type_name == "Missing" {
                Err(FeatureError::MissingFeature(name.to_string()))
            } else {
                Err(FeatureError::TypeMismatch {
                    feature: name.to_string(),
                    expected: "numeric scalar".to_string(),
                    got: type_name.to_string(),
                })
            }
        }
    }
}

/// Compute the number of scalar outputs produced by a spec.
fn spec_output_dim(spec: &FeatureSpec) -> usize {
    // Walk transforms in reverse to find the last dimensionality-changing transform.
    for transform in spec.transforms.iter().rev() {
        match transform {
            FeatureTransform::OneHotEncode { categories } => return categories.len(),
            FeatureTransform::PolynomialFeatures { degree } => return *degree as usize,
            _ => {}
        }
    }
    1
}

/// Populate `names` with the output feature names for one spec.
fn push_feature_names(spec: &FeatureSpec, names: &mut Vec<String>) {
    let dim = spec_output_dim(spec);
    if dim == 1 {
        names.push(spec.name.clone());
    } else {
        // OneHot or Polynomial expansion.
        for i in 0..dim {
            names.push(format!("{}_cat_{}", spec.name, i));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::ml_feature_extractor::{
        fit_minmax_scaler, fit_onehot, fit_standard_scaler, FeatureError, FeatureExtractor,
        FeatureSpec, FeatureTransform, FeatureValue,
    };

    // ── helpers ──────────────────────────────────────────────────────────────

    fn single_spec(name: &str, transforms: Vec<FeatureTransform>) -> FeatureSpec {
        FeatureSpec {
            name: name.to_string(),
            transforms,
        }
    }

    fn make_input(name: &str, val: FeatureValue) -> HashMap<String, FeatureValue> {
        let mut m = HashMap::new();
        m.insert(name.to_string(), val);
        m
    }

    fn extract_scalar(spec: FeatureSpec, val: FeatureValue) -> f64 {
        let mut ex = FeatureExtractor::new();
        ex.add_spec(spec.clone());
        let input = make_input(&spec.name, val);
        let res = ex.extract(&input).expect("test: should succeed");
        res.values[0]
    }

    // ── StandardScaler ───────────────────────────────────────────────────────

    #[test]
    fn test_standard_scaler_basic() {
        let t = FeatureTransform::StandardScaler {
            mean: 10.0,
            std: 2.0,
        };
        let v = extract_scalar(single_spec("x", vec![t]), FeatureValue::Float(14.0));
        assert!((v - 2.0).abs() < 1e-12, "v={v}");
    }

    #[test]
    fn test_standard_scaler_zero_std() {
        let t = FeatureTransform::StandardScaler {
            mean: 5.0,
            std: 0.0,
        };
        let v = extract_scalar(single_spec("x", vec![t]), FeatureValue::Float(99.0));
        assert_eq!(v, 0.0, "zero std must produce 0.0");
    }

    #[test]
    fn test_standard_scaler_negative_result() {
        let t = FeatureTransform::StandardScaler {
            mean: 10.0,
            std: 2.0,
        };
        let v = extract_scalar(single_spec("x", vec![t]), FeatureValue::Float(6.0));
        assert!((v - (-2.0)).abs() < 1e-12, "v={v}");
    }

    #[test]
    fn test_standard_scaler_integer_input() {
        let t = FeatureTransform::StandardScaler {
            mean: 0.0,
            std: 1.0,
        };
        let v = extract_scalar(single_spec("x", vec![t]), FeatureValue::Integer(3));
        assert!((v - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_standard_scaler_boolean_input() {
        let t = FeatureTransform::StandardScaler {
            mean: 0.0,
            std: 1.0,
        };
        let v = extract_scalar(single_spec("x", vec![t]), FeatureValue::Boolean(true));
        assert!((v - 1.0).abs() < 1e-12);
    }

    // ── MinMaxScaler ──────────────────────────────────────────────────────────

    #[test]
    fn test_minmax_scaler_midpoint() {
        let t = FeatureTransform::MinMaxScaler {
            min: 0.0,
            max: 10.0,
        };
        let v = extract_scalar(single_spec("x", vec![t]), FeatureValue::Float(5.0));
        assert!((v - 0.5).abs() < 1e-12, "v={v}");
    }

    #[test]
    fn test_minmax_scaler_at_min() {
        let t = FeatureTransform::MinMaxScaler {
            min: 0.0,
            max: 10.0,
        };
        let v = extract_scalar(single_spec("x", vec![t]), FeatureValue::Float(0.0));
        assert!((v - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_minmax_scaler_at_max() {
        let t = FeatureTransform::MinMaxScaler {
            min: 0.0,
            max: 10.0,
        };
        let v = extract_scalar(single_spec("x", vec![t]), FeatureValue::Float(10.0));
        assert!((v - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_minmax_scaler_zero_range() {
        let t = FeatureTransform::MinMaxScaler { min: 5.0, max: 5.0 };
        let v = extract_scalar(single_spec("x", vec![t]), FeatureValue::Float(5.0));
        assert_eq!(v, 0.0, "zero range must produce 0.0");
    }

    // ── Log1p ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_log1p_positive() {
        let t = FeatureTransform::Log1p;
        let v = extract_scalar(single_spec("x", vec![t]), FeatureValue::Float(0.0));
        assert!((v - 0.0).abs() < 1e-12, "ln(1+0)=0, v={v}");
    }

    #[test]
    fn test_log1p_positive_value() {
        let t = FeatureTransform::Log1p;
        // ln(1 + 1) = ln(2)
        let v = extract_scalar(single_spec("x", vec![t]), FeatureValue::Float(1.0));
        assert!((v - 2.0_f64.ln()).abs() < 1e-12, "v={v}");
    }

    #[test]
    fn test_log1p_negative_value() {
        // ln(1+|−3|) × sign(−3) = ln(4) × (−1)
        let t = FeatureTransform::Log1p;
        let v = extract_scalar(single_spec("x", vec![t]), FeatureValue::Float(-3.0));
        let expected = -(4.0_f64.ln());
        assert!((v - expected).abs() < 1e-12, "v={v}, expected={expected}");
    }

    // ── Sqrt ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_sqrt_positive() {
        let t = FeatureTransform::Sqrt;
        let v = extract_scalar(single_spec("x", vec![t]), FeatureValue::Float(9.0));
        assert!((v - 3.0).abs() < 1e-12, "v={v}");
    }

    #[test]
    fn test_sqrt_negative() {
        let t = FeatureTransform::Sqrt;
        // √4 × (−1) = −2.0
        let v = extract_scalar(single_spec("x", vec![t]), FeatureValue::Float(-4.0));
        assert!((v - (-2.0)).abs() < 1e-12, "v={v}");
    }

    #[test]
    fn test_sqrt_zero() {
        let t = FeatureTransform::Sqrt;
        let v = extract_scalar(single_spec("x", vec![t]), FeatureValue::Float(0.0));
        assert_eq!(v, 0.0);
    }

    // ── Clip ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_clip_below_lo() {
        let t = FeatureTransform::Clip { lo: 0.0, hi: 1.0 };
        let v = extract_scalar(single_spec("x", vec![t]), FeatureValue::Float(-5.0));
        assert_eq!(v, 0.0);
    }

    #[test]
    fn test_clip_above_hi() {
        let t = FeatureTransform::Clip { lo: 0.0, hi: 1.0 };
        let v = extract_scalar(single_spec("x", vec![t]), FeatureValue::Float(10.0));
        assert_eq!(v, 1.0);
    }

    #[test]
    fn test_clip_within_range() {
        let t = FeatureTransform::Clip { lo: 0.0, hi: 1.0 };
        let v = extract_scalar(single_spec("x", vec![t]), FeatureValue::Float(0.5));
        assert!((v - 0.5).abs() < 1e-12);
    }

    // ── Binarize ──────────────────────────────────────────────────────────────

    #[test]
    fn test_binarize_above_threshold() {
        let t = FeatureTransform::Binarize { threshold: 0.5 };
        let v = extract_scalar(single_spec("x", vec![t]), FeatureValue::Float(0.7));
        assert_eq!(v, 1.0);
    }

    #[test]
    fn test_binarize_below_threshold() {
        let t = FeatureTransform::Binarize { threshold: 0.5 };
        let v = extract_scalar(single_spec("x", vec![t]), FeatureValue::Float(0.3));
        assert_eq!(v, 0.0);
    }

    #[test]
    fn test_binarize_at_threshold() {
        // Exactly at threshold is NOT above → 0.0
        let t = FeatureTransform::Binarize { threshold: 0.5 };
        let v = extract_scalar(single_spec("x", vec![t]), FeatureValue::Float(0.5));
        assert_eq!(v, 0.0);
    }

    // ── PolynomialFeatures ────────────────────────────────────────────────────

    #[test]
    fn test_polynomial_degree2() {
        let spec = single_spec(
            "x",
            vec![FeatureTransform::PolynomialFeatures { degree: 2 }],
        );
        let mut ex = FeatureExtractor::new();
        ex.add_spec(spec);
        let input = make_input("x", FeatureValue::Float(3.0));
        let res = ex.extract(&input).expect("test: should succeed");
        assert_eq!(res.values.len(), 2);
        assert!((res.values[0] - 3.0).abs() < 1e-12, "x={}", res.values[0]);
        assert!((res.values[1] - 9.0).abs() < 1e-12, "x²={}", res.values[1]);
    }

    #[test]
    fn test_polynomial_degree3() {
        let spec = single_spec(
            "x",
            vec![FeatureTransform::PolynomialFeatures { degree: 3 }],
        );
        let mut ex = FeatureExtractor::new();
        ex.add_spec(spec);
        let input = make_input("x", FeatureValue::Float(2.0));
        let res = ex.extract(&input).expect("test: should succeed");
        assert_eq!(res.values.len(), 3);
        assert!((res.values[0] - 2.0).abs() < 1e-12);
        assert!((res.values[1] - 4.0).abs() < 1e-12);
        assert!((res.values[2] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_polynomial_degree0_error() {
        let spec = single_spec(
            "x",
            vec![FeatureTransform::PolynomialFeatures { degree: 0 }],
        );
        let mut ex = FeatureExtractor::new();
        ex.add_spec(spec);
        let input = make_input("x", FeatureValue::Float(1.0));
        let err = ex.extract(&input).unwrap_err();
        assert!(matches!(err, FeatureError::InvalidTransform(_)));
    }

    // ── OneHotEncode ──────────────────────────────────────────────────────────

    #[test]
    fn test_onehot_known_category() {
        let cats = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let spec = single_spec(
            "color",
            vec![FeatureTransform::OneHotEncode { categories: cats }],
        );
        let mut ex = FeatureExtractor::new();
        ex.add_spec(spec);
        let input = make_input("color", FeatureValue::Categorical("b".to_string()));
        let res = ex.extract(&input).expect("test: should succeed");
        assert_eq!(res.values, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_onehot_unknown_category_all_zeros() {
        let cats = vec!["a".to_string(), "b".to_string()];
        let spec = single_spec(
            "x",
            vec![FeatureTransform::OneHotEncode { categories: cats }],
        );
        let mut ex = FeatureExtractor::new();
        ex.add_spec(spec);
        let input = make_input("x", FeatureValue::Categorical("z".to_string()));
        let res = ex.extract(&input).expect("test: should succeed");
        assert_eq!(res.values, vec![0.0, 0.0]);
    }

    #[test]
    fn test_onehot_feature_names_expanded() {
        let cats = vec!["x".to_string(), "y".to_string()];
        let spec = single_spec(
            "col",
            vec![FeatureTransform::OneHotEncode { categories: cats }],
        );
        let mut ex = FeatureExtractor::new();
        ex.add_spec(spec);
        let names = ex.feature_names();
        assert_eq!(names, vec!["col_cat_0", "col_cat_1"]);
    }

    // ── Imputation ────────────────────────────────────────────────────────────

    #[test]
    fn test_impute_mean_missing_value() {
        let t = FeatureTransform::ImputeMean { mean: 42.0 };
        let spec = single_spec(
            "x",
            vec![
                t,
                FeatureTransform::StandardScaler {
                    mean: 42.0,
                    std: 1.0,
                },
            ],
        );
        let mut ex = FeatureExtractor::new();
        ex.add_spec(spec);
        let input = make_input("x", FeatureValue::Missing);
        let res = ex.extract(&input).expect("test: should succeed");
        // (42 - 42) / 1 = 0.0
        assert!((res.values[0] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_impute_mean_not_missing_unchanged() {
        let t = FeatureTransform::ImputeMean { mean: 0.0 };
        let spec = single_spec("x", vec![t]);
        let v = extract_scalar(spec, FeatureValue::Float(7.0));
        assert!((v - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_impute_mode_missing_categorical() {
        let cats = vec!["cat".to_string(), "dog".to_string()];
        let spec = single_spec(
            "pet",
            vec![
                FeatureTransform::ImputeMode {
                    mode: "cat".to_string(),
                },
                FeatureTransform::OneHotEncode { categories: cats },
            ],
        );
        let mut ex = FeatureExtractor::new();
        ex.add_spec(spec);
        let input = make_input("pet", FeatureValue::Missing);
        let res = ex.extract(&input).expect("test: should succeed");
        // "cat" → [1.0, 0.0]
        assert_eq!(res.values, vec![1.0, 0.0]);
    }

    // ── MissingFeature error without imputer ──────────────────────────────────

    #[test]
    fn test_missing_feature_no_imputer_error() {
        let t = FeatureTransform::StandardScaler {
            mean: 0.0,
            std: 1.0,
        };
        let spec = single_spec("x", vec![t]);
        let mut ex = FeatureExtractor::new();
        ex.add_spec(spec);
        let input: HashMap<String, FeatureValue> = HashMap::new();
        let err = ex.extract(&input).unwrap_err();
        assert!(matches!(err, FeatureError::MissingFeature(ref name) if name == "x"));
    }

    #[test]
    fn test_missing_feature_key_not_in_map_error() {
        let t = FeatureTransform::StandardScaler {
            mean: 0.0,
            std: 1.0,
        };
        let spec = single_spec("age", vec![t]);
        let mut ex = FeatureExtractor::new();
        ex.add_spec(spec);
        // Insert a different key
        let input = make_input("height", FeatureValue::Float(1.75));
        let err = ex.extract(&input).unwrap_err();
        assert!(matches!(err, FeatureError::MissingFeature(ref name) if name == "age"));
    }

    // ── Chained transforms ────────────────────────────────────────────────────

    #[test]
    fn test_chain_clip_then_standard_scaler() {
        // Clip to [0, 1] then StandardScaler with mean=0.5, std=0.5
        let spec = single_spec(
            "x",
            vec![
                FeatureTransform::Clip { lo: 0.0, hi: 1.0 },
                FeatureTransform::StandardScaler {
                    mean: 0.5,
                    std: 0.5,
                },
            ],
        );
        let v = extract_scalar(spec, FeatureValue::Float(100.0));
        // Clipped to 1.0, then (1.0 - 0.5) / 0.5 = 1.0
        assert!((v - 1.0).abs() < 1e-12, "v={v}");
    }

    #[test]
    fn test_chain_impute_then_log1p() {
        let spec = single_spec(
            "x",
            vec![
                FeatureTransform::ImputeMean { mean: 0.0 },
                FeatureTransform::Log1p,
            ],
        );
        let v = extract_scalar(spec, FeatureValue::Missing);
        // ln(1 + 0) = 0
        assert_eq!(v, 0.0);
    }

    // ── Multi-spec extractor ──────────────────────────────────────────────────

    #[test]
    fn test_multi_spec_output_order() {
        let mut ex = FeatureExtractor::new();
        ex.add_spec(single_spec(
            "a",
            vec![FeatureTransform::StandardScaler {
                mean: 0.0,
                std: 1.0,
            }],
        ));
        ex.add_spec(single_spec(
            "b",
            vec![FeatureTransform::MinMaxScaler {
                min: 0.0,
                max: 10.0,
            }],
        ));

        let mut input = HashMap::new();
        input.insert("a".to_string(), FeatureValue::Float(1.0));
        input.insert("b".to_string(), FeatureValue::Float(5.0));

        let res = ex.extract(&input).expect("test: should succeed");
        assert_eq!(res.values.len(), 2);
        assert!((res.values[0] - 1.0).abs() < 1e-12); // (1.0-0)/1
        assert!((res.values[1] - 0.5).abs() < 1e-12); // (5-0)/10
    }

    #[test]
    fn test_output_dim() {
        let mut ex = FeatureExtractor::new();
        ex.add_spec(single_spec(
            "a",
            vec![FeatureTransform::StandardScaler {
                mean: 0.0,
                std: 1.0,
            }],
        ));
        ex.add_spec(single_spec(
            "b",
            vec![FeatureTransform::OneHotEncode {
                categories: vec!["x".to_string(), "y".to_string(), "z".to_string()],
            }],
        ));
        ex.add_spec(single_spec(
            "c",
            vec![FeatureTransform::PolynomialFeatures { degree: 2 }],
        ));
        // 1 + 3 + 2 = 6
        assert_eq!(ex.output_dim(), 6);
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_after_single_extract() {
        let mut ex = FeatureExtractor::new();
        ex.add_spec(single_spec(
            "x",
            vec![FeatureTransform::StandardScaler {
                mean: 0.0,
                std: 1.0,
            }],
        ));
        let input = make_input("x", FeatureValue::Float(1.0));
        ex.extract(&input).expect("test: should succeed");
        let s = ex.stats();
        assert_eq!(s.total_extractions, 1);
        assert!((s.avg_output_dim - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_stats_accumulate_over_batch() {
        let mut ex = FeatureExtractor::new();
        ex.add_spec(single_spec("x", vec![FeatureTransform::Log1p]));
        let batch: Vec<HashMap<String, FeatureValue>> = (0..5)
            .map(|i| make_input("x", FeatureValue::Float(i as f64)))
            .collect();
        ex.extract_batch(&batch).expect("test: should succeed");
        assert_eq!(ex.stats().total_extractions, 5);
    }

    // ── extract_batch ─────────────────────────────────────────────────────────

    #[test]
    fn test_extract_batch_empty_returns_error() {
        let ex = FeatureExtractor::new();
        let err = ex.extract_batch(&[]).unwrap_err();
        assert_eq!(err, FeatureError::EmptyInput);
    }

    #[test]
    fn test_extract_batch_multiple_inputs() {
        let mut ex = FeatureExtractor::new();
        ex.add_spec(single_spec(
            "v",
            vec![FeatureTransform::MinMaxScaler {
                min: 0.0,
                max: 10.0,
            }],
        ));
        let batch = vec![
            make_input("v", FeatureValue::Float(0.0)),
            make_input("v", FeatureValue::Float(5.0)),
            make_input("v", FeatureValue::Float(10.0)),
        ];
        let results = ex.extract_batch(&batch).expect("test: should succeed");
        assert_eq!(results.len(), 3);
        assert!((results[0].values[0] - 0.0).abs() < 1e-12);
        assert!((results[1].values[0] - 0.5).abs() < 1e-12);
        assert!((results[2].values[0] - 1.0).abs() < 1e-12);
    }

    // ── Fitting helpers ───────────────────────────────────────────────────────

    #[test]
    fn test_fit_standard_scaler() {
        let t = fit_standard_scaler(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        if let FeatureTransform::StandardScaler { mean, std } = t {
            assert!((mean - 3.0).abs() < 1e-10, "mean={mean}");
            // Population std of [1..5]: variance = 2.0, std = √2
            assert!((std - 2.0_f64.sqrt()).abs() < 1e-10, "std={std}");
        } else {
            panic!("expected StandardScaler");
        }
    }

    #[test]
    fn test_fit_standard_scaler_empty() {
        let t = fit_standard_scaler(&[]);
        assert!(matches!(
            t,
            FeatureTransform::StandardScaler {
                mean: 0.0,
                std: 0.0
            }
        ));
    }

    #[test]
    fn test_fit_minmax_scaler() {
        let t = fit_minmax_scaler(&[3.0, 1.0, 7.0, -2.0]);
        if let FeatureTransform::MinMaxScaler { min, max } = t {
            assert!((min - (-2.0)).abs() < 1e-12);
            assert!((max - 7.0).abs() < 1e-12);
        } else {
            panic!("expected MinMaxScaler");
        }
    }

    #[test]
    fn test_fit_minmax_scaler_empty() {
        let t = fit_minmax_scaler(&[]);
        assert!(matches!(
            t,
            FeatureTransform::MinMaxScaler { min: 0.0, max: 0.0 }
        ));
    }

    #[test]
    fn test_fit_onehot_dedup_sort() {
        let cats = vec![
            "banana".to_string(),
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
        ];
        let t = fit_onehot(&cats);
        if let FeatureTransform::OneHotEncode { categories } = t {
            assert_eq!(categories, vec!["apple", "banana", "cherry"]);
        } else {
            panic!("expected OneHotEncode");
        }
    }

    // ── Type-mismatch errors ──────────────────────────────────────────────────

    #[test]
    fn test_type_mismatch_categorical_to_scaler() {
        let t = FeatureTransform::StandardScaler {
            mean: 0.0,
            std: 1.0,
        };
        let spec = single_spec("x", vec![t]);
        let mut ex = FeatureExtractor::new();
        ex.add_spec(spec);
        let input = make_input("x", FeatureValue::Categorical("hello".to_string()));
        let err = ex.extract(&input).unwrap_err();
        assert!(matches!(err, FeatureError::TypeMismatch { .. }));
    }

    #[test]
    fn test_type_mismatch_float_to_onehot() {
        let t = FeatureTransform::OneHotEncode {
            categories: vec!["a".to_string()],
        };
        let spec = single_spec("x", vec![t]);
        let mut ex = FeatureExtractor::new();
        ex.add_spec(spec);
        let input = make_input("x", FeatureValue::Float(1.0));
        let err = ex.extract(&input).unwrap_err();
        assert!(matches!(err, FeatureError::TypeMismatch { .. }));
    }

    // ── feature_names alignment ───────────────────────────────────────────────

    #[test]
    fn test_feature_names_alignment_with_values() {
        let mut ex = FeatureExtractor::new();
        ex.add_spec(single_spec("score", vec![FeatureTransform::Log1p]));
        ex.add_spec(single_spec(
            "color",
            vec![FeatureTransform::OneHotEncode {
                categories: vec!["red".to_string(), "blue".to_string()],
            }],
        ));

        let mut input = HashMap::new();
        input.insert("score".to_string(), FeatureValue::Float(1.0));
        input.insert(
            "color".to_string(),
            FeatureValue::Categorical("red".to_string()),
        );

        let res = ex.extract(&input).expect("test: should succeed");
        assert_eq!(res.feature_names.len(), res.values.len());
        assert_eq!(res.feature_names[0], "score");
        assert_eq!(res.feature_names[1], "color_cat_0");
        assert_eq!(res.feature_names[2], "color_cat_1");
    }
}
