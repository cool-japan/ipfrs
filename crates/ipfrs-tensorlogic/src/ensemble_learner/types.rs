//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use thiserror::Error;

use super::functions::{
    best_regression_stump, best_stump, bootstrap_indices, fit_perceptron, xorshift_usize,
};

/// Configuration for [`EnsembleLearner`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElLearnerConfig {
    /// Ensemble method to use.
    pub method: ElMethod,
    /// Number of base estimators.
    pub n_estimators: usize,
    /// Learning rate (shrinkage) for boosting methods.
    pub learning_rate: f64,
    /// Maximum depth for decision stump trees (currently 1 = stump).
    pub max_depth: u32,
    /// PRNG seed for reproducibility.
    pub seed: u64,
    /// Fraction of samples to use per bootstrap draw (Bagging/RandomForest).
    pub subsample: f64,
}
impl ElLearnerConfig {
    /// Validate configuration fields, returning an error if invalid.
    pub fn validate(&self) -> Result<(), ElError> {
        if self.n_estimators == 0 {
            return Err(ElError::InvalidConfig(
                "n_estimators must be >= 1".to_string(),
            ));
        }
        if !(0.0 < self.learning_rate && self.learning_rate <= 1.0) {
            return Err(ElError::InvalidConfig(
                "learning_rate must be in (0, 1]".to_string(),
            ));
        }
        if !(0.0 < self.subsample && self.subsample <= 1.0) {
            return Err(ElError::InvalidConfig(
                "subsample must be in (0, 1]".to_string(),
            ));
        }
        Ok(())
    }
}
/// A single base model (decision stump or perceptron).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ElBaseModel {
    /// Axis-aligned threshold classifier / regressor.
    DecisionStump {
        /// Feature index to split on.
        feature_index: usize,
        /// Split threshold.
        threshold: f64,
        /// If `true`, predict `+class_value` when `feature[idx] <= threshold`.
        direction: bool,
        /// Alpha / weight of this model in the ensemble (boosting).
        weight: f64,
    },
    /// Linear threshold classifier.
    Perceptron {
        /// Weight vector.
        weights: Vec<f64>,
        /// Bias term.
        bias: f64,
        /// Alpha / weight of this model in the ensemble.
        weight: f64,
    },
}
impl ElBaseModel {
    /// Raw model weight (alpha) used for weighted voting.
    pub fn weight(&self) -> f64 {
        match self {
            ElBaseModel::DecisionStump { weight, .. } => *weight,
            ElBaseModel::Perceptron { weight, .. } => *weight,
        }
    }
    /// Run the model on one feature vector; returns a real-valued prediction.
    pub fn predict_raw(&self, features: &[f64]) -> Result<f64, ElError> {
        match self {
            ElBaseModel::DecisionStump {
                feature_index,
                threshold,
                direction,
                ..
            } => {
                let x = features
                    .get(*feature_index)
                    .ok_or(ElError::FeatureDimensionMismatch {
                        expected: *feature_index + 1,
                        got: features.len(),
                    })?;
                let positive = if *direction {
                    *x <= *threshold
                } else {
                    *x > *threshold
                };
                Ok(if positive { 1.0 } else { -1.0 })
            }
            ElBaseModel::Perceptron { weights, bias, .. } => {
                if features.len() != weights.len() {
                    return Err(ElError::FeatureDimensionMismatch {
                        expected: weights.len(),
                        got: features.len(),
                    });
                }
                let dot: f64 = features
                    .iter()
                    .zip(weights.iter())
                    .map(|(x, w)| x * w)
                    .sum();
                Ok(dot + bias)
            }
        }
    }
}
/// Errors produced by [`EnsembleLearner`].
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ElError {
    /// Training set is empty.
    #[error("training set is empty")]
    EmptyTrainingSet,
    /// Prediction requested but no models have been trained.
    #[error("no trained models available")]
    NoTrainedModels,
    /// Feature dimensionality mismatch.
    #[error("expected {expected} features, got {got}")]
    FeatureDimensionMismatch { expected: usize, got: usize },
    /// Configuration is invalid.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// Arithmetic error (e.g. NaN, division by zero).
    #[error("arithmetic error: {0}")]
    Arithmetic(String),
    /// The chosen method does not support the requested operation.
    #[error("operation '{op}' is not supported for method '{method}'")]
    UnsupportedOperation { op: String, method: String },
}
/// Prediction returned by [`EnsembleLearner`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElPrediction {
    /// Aggregated prediction value (class vote or regression average).
    pub value: f64,
    /// Confidence in [0, 1]: fraction of models that agree with the majority (classification)
    /// or 1 / (1 + normalised_std) for regression.
    pub confidence: f64,
    /// Number of base models that contributed to this prediction.
    pub n_models: usize,
}
/// A single labelled sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElSample {
    /// Feature vector.
    pub features: Vec<f64>,
    /// Label (continuous for regression; ±1.0 for binary classification).
    pub label: f64,
}
impl ElSample {
    /// Construct a new sample.
    pub fn new(features: Vec<f64>, label: f64) -> Self {
        Self { features, label }
    }
}
/// One entry in the training history ring-buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElTrainingRecord {
    /// Round index (0-based).
    pub round: usize,
    /// Training loss / error at this round.
    pub train_error: f64,
    /// Alpha / weight assigned to this round's model.
    pub alpha: f64,
    /// Method-specific note.
    pub note: String,
}
/// Ensemble learning strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ElMethod {
    /// Independent bootstrap-aggregated base learners; majority/average vote.
    #[default]
    Bagging,
    /// Adaptive Boosting (AdaBoost.M1) with stump base learners.
    AdaBoost,
    /// Gradient Boosting with residual-fitting stumps and shrinkage.
    GradientBoosting,
    /// Bagging + random feature sub-sampling (Random Forest).
    RandomForest,
    /// Train diverse base models then a linear meta-learner (Stacking).
    Stacking,
}
/// Summary statistics returned by [`EnsembleLearner::learner_stats`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElLearnerStats {
    /// Method in use.
    pub method: ElMethod,
    /// Number of trained base models.
    pub n_models: usize,
    /// Total weighted prediction calls since construction.
    pub total_predictions: u64,
    /// Sum of all model weights.
    pub total_weight: f64,
    /// Minimum model weight.
    pub min_weight: f64,
    /// Maximum model weight.
    pub max_weight: f64,
    /// Number of training records in history.
    pub history_len: usize,
    /// Whether the learner has been trained.
    pub is_trained: bool,
}
/// Ensemble learning system supporting Bagging, AdaBoost, Gradient Boosting,
/// Random Forest, and Stacking strategies.
pub struct EnsembleLearner {
    pub(super) config: ElLearnerConfig,
    pub(super) models: Vec<ElBaseModel>,
    pub(super) model_weights: Vec<f64>,
    /// For GradientBoosting: initial constant prediction (mean of labels).
    pub(super) gb_init: f64,
    /// For GradientBoosting: per-stump leaf values (pos, neg) per round.
    pub(super) gb_leaf_values: Vec<(f64, f64)>,
    pub(super) training_history: VecDeque<ElTrainingRecord>,
    pub(super) n_features: usize,
    pub(super) is_trained: bool,
    pub(super) total_predictions: u64,
    /// Meta-learner weights for Stacking [n_estimators + 1] (last = bias).
    pub(super) meta_weights: Vec<f64>,
}
impl EnsembleLearner {
    /// Create a new [`EnsembleLearner`] from the given configuration.
    pub fn new(config: ElLearnerConfig) -> Self {
        Self {
            config,
            models: Vec::new(),
            model_weights: Vec::new(),
            gb_init: 0.0,
            gb_leaf_values: Vec::new(),
            training_history: VecDeque::with_capacity(100),
            n_features: 0,
            is_trained: false,
            total_predictions: 0,
            meta_weights: Vec::new(),
        }
    }
    /// Create with default configuration.
    pub fn default_bagging(n_estimators: usize) -> Self {
        let cfg = ElLearnerConfig {
            n_estimators,
            method: ElMethod::Bagging,
            ..Default::default()
        };
        Self::new(cfg)
    }
    /// Return a reference to the configuration.
    pub fn config(&self) -> &ElLearnerConfig {
        &self.config
    }
    /// Return trained base models.
    pub fn models(&self) -> &[ElBaseModel] {
        &self.models
    }
    /// Train the ensemble on the provided samples.
    pub fn fit(&mut self, samples: &[ElSample]) -> Result<(), ElError> {
        self.config.validate()?;
        if samples.is_empty() {
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
        self.n_features = n_feat;
        self.models.clear();
        self.model_weights.clear();
        self.gb_leaf_values.clear();
        self.training_history.clear();
        self.is_trained = false;
        match self.config.method {
            ElMethod::Bagging => self.fit_bagging(samples)?,
            ElMethod::AdaBoost => self.fit_adaboost(samples)?,
            ElMethod::GradientBoosting => self.fit_gradient_boosting(samples)?,
            ElMethod::RandomForest => self.fit_random_forest(samples)?,
            ElMethod::Stacking => self.fit_stacking(samples)?,
        }
        self.is_trained = true;
        Ok(())
    }
    pub(super) fn fit_bagging(&mut self, samples: &[ElSample]) -> Result<(), ElError> {
        let mut rng = self.config.seed;
        let n = samples.len();
        let boot_size = ((n as f64 * self.config.subsample).round() as usize).max(1);
        let all_features: Vec<usize> = (0..self.n_features).collect();
        for round in 0..self.config.n_estimators {
            let idxs = bootstrap_indices(&mut rng, n, boot_size);
            let sub: Vec<ElSample> = idxs.iter().map(|&i| samples[i].clone()).collect();
            let (feat_idx, thresh, dir, err) = best_stump(
                &sub,
                &vec![1.0 / boot_size as f64; boot_size],
                &all_features,
            )?;
            let model = ElBaseModel::DecisionStump {
                feature_index: feat_idx,
                threshold: thresh,
                direction: dir,
                weight: 1.0,
            };
            self.push_history(ElTrainingRecord {
                round,
                train_error: err,
                alpha: 1.0,
                note: format!("bagging round {round}"),
            });
            self.models.push(model);
            self.model_weights.push(1.0);
        }
        Ok(())
    }
    pub(super) fn fit_adaboost(&mut self, samples: &[ElSample]) -> Result<(), ElError> {
        let n = samples.len();
        let mut weights = vec![1.0 / n as f64; n];
        let all_features: Vec<usize> = (0..self.n_features).collect();
        for round in 0..self.config.n_estimators {
            let (feat_idx, thresh, dir, err) = best_stump(samples, &weights, &all_features)?;
            let err_clipped = err.clamp(1e-10, 1.0 - 1e-10);
            let alpha = 0.5 * ((1.0 - err_clipped) / err_clipped).ln();
            let mut weight_sum = 0.0f64;
            let new_weights: Vec<f64> = samples
                .iter()
                .zip(weights.iter())
                .map(|(s, &w)| {
                    let pred = if (dir
                        && s.features.get(feat_idx).copied().unwrap_or(0.0) <= thresh)
                        || (!dir && s.features.get(feat_idx).copied().unwrap_or(0.0) > thresh)
                    {
                        1.0
                    } else {
                        -1.0
                    };
                    let label_sign = if s.label >= 0.0 { 1.0 } else { -1.0 };
                    let new_w = w * (-alpha * label_sign * pred).exp();
                    weight_sum += new_w;
                    new_w
                })
                .collect();
            if weight_sum <= 0.0 {
                return Err(ElError::Arithmetic(
                    "AdaBoost weight sum became zero".to_string(),
                ));
            }
            weights = new_weights.iter().map(|&w| w / weight_sum).collect();
            self.push_history(ElTrainingRecord {
                round,
                train_error: err,
                alpha,
                note: format!("adaboost alpha={alpha:.4}"),
            });
            let model = ElBaseModel::DecisionStump {
                feature_index: feat_idx,
                threshold: thresh,
                direction: dir,
                weight: alpha,
            };
            self.models.push(model);
            self.model_weights.push(alpha);
            if err < 1e-12 {
                break;
            }
        }
        Ok(())
    }
    pub(super) fn fit_gradient_boosting(&mut self, samples: &[ElSample]) -> Result<(), ElError> {
        let n = samples.len();
        let all_features: Vec<usize> = (0..self.n_features).collect();
        let mean_label = samples.iter().map(|s| s.label).sum::<f64>() / n as f64;
        self.gb_init = mean_label;
        let mut predictions = vec![mean_label; n];
        for round in 0..self.config.n_estimators {
            let residuals: Vec<f64> = samples
                .iter()
                .zip(predictions.iter())
                .map(|(s, &p)| s.label - p)
                .collect();
            let mse = residuals.iter().map(|r| r * r).sum::<f64>() / n as f64;
            let (feat_idx, thresh, dir, leaf_pos, leaf_neg) =
                best_regression_stump(samples, &residuals, &all_features)?;
            for (s, pred) in samples.iter().zip(predictions.iter_mut()) {
                let fv = s.features.get(feat_idx).copied().unwrap_or(0.0);
                let step = if (dir && fv <= thresh) || (!dir && fv > thresh) {
                    leaf_pos
                } else {
                    leaf_neg
                };
                *pred += self.config.learning_rate * step;
            }
            self.push_history(ElTrainingRecord {
                round,
                train_error: mse,
                alpha: self.config.learning_rate,
                note: format!("gb mse={mse:.6}"),
            });
            let model = ElBaseModel::DecisionStump {
                feature_index: feat_idx,
                threshold: thresh,
                direction: dir,
                weight: self.config.learning_rate,
            };
            self.models.push(model);
            self.model_weights.push(self.config.learning_rate);
            self.gb_leaf_values.push((leaf_pos, leaf_neg));
            if mse < 1e-12 {
                break;
            }
        }
        Ok(())
    }
    pub(super) fn fit_random_forest(&mut self, samples: &[ElSample]) -> Result<(), ElError> {
        let mut rng = self.config.seed;
        let n = samples.len();
        let boot_size = ((n as f64 * self.config.subsample).round() as usize).max(1);
        let n_features_per_split = ((self.n_features as f64).sqrt().round() as usize).max(1);
        for round in 0..self.config.n_estimators {
            let idxs = bootstrap_indices(&mut rng, n, boot_size);
            let sub: Vec<ElSample> = idxs.iter().map(|&i| samples[i].clone()).collect();
            let mut feature_indices: Vec<usize> = (0..self.n_features).collect();
            for i in 0..n_features_per_split.min(self.n_features) {
                let j = i + xorshift_usize(&mut rng, self.n_features - i);
                feature_indices.swap(i, j);
            }
            let feature_subset: Vec<usize> =
                feature_indices[..n_features_per_split.min(self.n_features)].to_vec();
            let (feat_idx, thresh, dir, err) = best_stump(
                &sub,
                &vec![1.0 / boot_size as f64; boot_size],
                &feature_subset,
            )?;
            self.push_history(ElTrainingRecord {
                round,
                train_error: err,
                alpha: 1.0,
                note: format!("rf round={round} feat_subset={n_features_per_split}"),
            });
            let model = ElBaseModel::DecisionStump {
                feature_index: feat_idx,
                threshold: thresh,
                direction: dir,
                weight: 1.0,
            };
            self.models.push(model);
            self.model_weights.push(1.0);
        }
        Ok(())
    }
    pub(super) fn fit_stacking(&mut self, samples: &[ElSample]) -> Result<(), ElError> {
        let mut rng = self.config.seed;
        let n = samples.len();
        let split = (n / 2).max(1);
        let base_samples = &samples[..split];
        let meta_samples = &samples[split..];
        let all_features: Vec<usize> = (0..self.n_features).collect();
        let boot_size = base_samples.len().max(1);
        let n_base = self.config.n_estimators.max(2);
        for round in 0..n_base {
            let (model, err) = if round % 3 == 2 {
                let idxs = bootstrap_indices(&mut rng, base_samples.len(), boot_size);
                let sub: Vec<ElSample> = idxs.iter().map(|&i| base_samples[i].clone()).collect();
                let m = fit_perceptron(&sub, self.n_features, &mut rng, self.config.learning_rate);
                (m, 0.5)
            } else {
                let idxs = bootstrap_indices(&mut rng, base_samples.len(), boot_size);
                let sub: Vec<ElSample> = idxs.iter().map(|&i| base_samples[i].clone()).collect();
                let (feat_idx, thresh, dir, err) = best_stump(
                    &sub,
                    &vec![1.0 / boot_size as f64; boot_size],
                    &all_features,
                )?;
                (
                    ElBaseModel::DecisionStump {
                        feature_index: feat_idx,
                        threshold: thresh,
                        direction: dir,
                        weight: 1.0,
                    },
                    err,
                )
            };
            self.push_history(ElTrainingRecord {
                round,
                train_error: err,
                alpha: 1.0,
                note: format!("stacking base round={round}"),
            });
            self.models.push(model);
            self.model_weights.push(1.0);
        }
        let n_meta = meta_samples.len();
        if n_meta == 0 {
            self.meta_weights = vec![1.0 / n_base as f64; n_base + 1];
            return Ok(());
        }
        let mut meta_features: Vec<Vec<f64>> = meta_samples
            .iter()
            .map(|s| {
                self.models
                    .iter()
                    .map(|m| m.predict_raw(&s.features).unwrap_or(0.0))
                    .collect()
            })
            .collect();
        for row in meta_features.iter_mut() {
            row.push(1.0);
        }
        let n_meta_feat = n_base + 1;
        let mut meta_w = vec![0.0f64; n_meta_feat];
        let meta_lr = self.config.learning_rate * 0.1;
        for _epoch in 0..200 {
            for (row, s) in meta_features.iter().zip(meta_samples.iter()) {
                let pred: f64 = row.iter().zip(meta_w.iter()).map(|(x, w)| x * w).sum();
                let err = s.label - pred;
                for (w, x) in meta_w.iter_mut().zip(row.iter()) {
                    *w += meta_lr * err * x;
                }
            }
        }
        self.meta_weights = meta_w;
        Ok(())
    }
    /// Predict the label for one feature vector.
    pub fn predict(&mut self, features: &[f64]) -> Result<ElPrediction, ElError> {
        if !self.is_trained || self.models.is_empty() {
            return Err(ElError::NoTrainedModels);
        }
        if features.len() != self.n_features {
            return Err(ElError::FeatureDimensionMismatch {
                expected: self.n_features,
                got: features.len(),
            });
        }
        self.total_predictions += 1;
        let pred = match self.config.method {
            ElMethod::GradientBoosting => self.predict_gb(features)?,
            ElMethod::Stacking => self.predict_stacking(features)?,
            _ => self.predict_weighted_vote(features)?,
        };
        Ok(pred)
    }
    pub(super) fn predict_weighted_vote(&self, features: &[f64]) -> Result<ElPrediction, ElError> {
        let mut weighted_sum = 0.0f64;
        let mut total_weight = 0.0f64;
        let n = self.models.len();
        for model in &self.models {
            let raw = model.predict_raw(features)?;
            let w = model.weight();
            weighted_sum += w * raw;
            total_weight += w.abs();
        }
        if total_weight <= 0.0 {
            return Err(ElError::Arithmetic(
                "total model weight is zero".to_string(),
            ));
        }
        let value = weighted_sum / total_weight;
        let majority_sign = if value >= 0.0 { 1.0 } else { -1.0 };
        let agree = self
            .models
            .iter()
            .filter(|m| m.predict_raw(features).unwrap_or(0.0) * majority_sign >= 0.0)
            .count();
        let confidence = agree as f64 / n as f64;
        Ok(ElPrediction {
            value,
            confidence,
            n_models: n,
        })
    }
    pub(super) fn predict_gb(&self, features: &[f64]) -> Result<ElPrediction, ElError> {
        let mut pred = self.gb_init;
        let n = self.models.len().min(self.gb_leaf_values.len());
        for i in 0..n {
            let model = &self.models[i];
            let (leaf_pos, leaf_neg) = self.gb_leaf_values[i];
            if let ElBaseModel::DecisionStump {
                feature_index,
                threshold,
                direction,
                weight,
            } = model
            {
                let fv = features.get(*feature_index).copied().ok_or(
                    ElError::FeatureDimensionMismatch {
                        expected: *feature_index + 1,
                        got: features.len(),
                    },
                )?;
                let step = if (*direction && fv <= *threshold) || (!*direction && fv > *threshold) {
                    leaf_pos
                } else {
                    leaf_neg
                };
                pred += weight * step;
            }
        }
        let confidence = 1.0 / (1.0 + (pred - self.gb_init).abs());
        Ok(ElPrediction {
            value: pred,
            confidence: confidence.min(1.0),
            n_models: n,
        })
    }
    pub(super) fn predict_stacking(&self, features: &[f64]) -> Result<ElPrediction, ElError> {
        if self.meta_weights.is_empty() {
            return self.predict_weighted_vote(features);
        }
        let base_preds: Vec<f64> = self
            .models
            .iter()
            .map(|m| m.predict_raw(features).unwrap_or(0.0))
            .collect();
        let n_base = self.models.len();
        let n_meta_feat = self.meta_weights.len();
        let mut row = base_preds.clone();
        row.push(1.0);
        let value: f64 = row[..n_meta_feat.min(row.len())]
            .iter()
            .zip(self.meta_weights.iter())
            .map(|(x, w)| x * w)
            .sum();
        let majority_sign = if value >= 0.0 { 1.0 } else { -1.0 };
        let agree = base_preds
            .iter()
            .filter(|&&p| p * majority_sign >= 0.0)
            .count();
        let confidence = if n_base > 0 {
            agree as f64 / n_base as f64
        } else {
            0.5
        };
        Ok(ElPrediction {
            value,
            confidence,
            n_models: n_base,
        })
    }
    /// Predict for a batch of feature vectors.
    pub fn predict_batch(&mut self, samples: &[Vec<f64>]) -> Vec<Result<ElPrediction, ElError>> {
        samples
            .iter()
            .map(|features| self.predict(features))
            .collect()
    }
    /// Aggregate feature importance across all decision stumps.
    ///
    /// Importance of feature `i` = sum of |alpha| for all stumps that split on `i`,
    /// normalized so that importances sum to 1.
    pub fn feature_importance(&self) -> Vec<f64> {
        if self.n_features == 0 {
            return Vec::new();
        }
        let mut importance = vec![0.0f64; self.n_features];
        for model in &self.models {
            if let ElBaseModel::DecisionStump {
                feature_index,
                weight,
                ..
            } = model
            {
                if *feature_index < importance.len() {
                    importance[*feature_index] += weight.abs();
                }
            }
        }
        let total: f64 = importance.iter().sum();
        if total > 0.0 {
            for imp in importance.iter_mut() {
                *imp /= total;
            }
        }
        importance
    }
    /// Estimate the out-of-bag accuracy for Bagging/RandomForest.
    ///
    /// For each sample `i` we collect predictions from models trained without
    /// sample `i` (approximated by a fresh bootstrap with probability
    /// `(1-1/n)^n ≈ 0.368`). Here we use a simplified estimator: for each
    /// sample, we use the prediction of a model trained on a bootstrap that did
    /// NOT include it. We identify such models by re-running the bootstrap
    /// indices using the deterministic PRNG.
    pub fn oob_score(&self, samples: &[ElSample]) -> f64 {
        if !self.is_trained || samples.is_empty() {
            return 0.0;
        }
        if self.config.method != ElMethod::Bagging && self.config.method != ElMethod::RandomForest {
            return 0.0;
        }
        let n = samples.len();
        let boot_size = ((n as f64 * self.config.subsample).round() as usize).max(1);
        let mut rng = self.config.seed;
        let mut oob_votes: Vec<Vec<f64>> = vec![Vec::new(); n];
        let n_models = self.models.len();
        for m_idx in 0..n_models {
            let idxs = bootstrap_indices(&mut rng, n, boot_size);
            let in_bag: std::collections::HashSet<usize> = idxs.into_iter().collect();
            for sample_idx in 0..n {
                if !in_bag.contains(&sample_idx) {
                    if let Ok(raw) = self.models[m_idx].predict_raw(&samples[sample_idx].features) {
                        oob_votes[sample_idx].push(raw);
                    }
                }
            }
        }
        let mut correct = 0usize;
        let mut counted = 0usize;
        for (votes, s) in oob_votes.iter().zip(samples.iter()) {
            if votes.is_empty() {
                continue;
            }
            let avg: f64 = votes.iter().sum::<f64>() / votes.len() as f64;
            let pred_sign: f64 = if avg >= 0.0 { 1.0 } else { -1.0 };
            let label_sign: f64 = if s.label >= 0.0 { 1.0 } else { -1.0 };
            counted += 1;
            if (pred_sign - label_sign).abs() < 1e-9 {
                correct += 1;
            }
        }
        if counted == 0 {
            return 0.0;
        }
        correct as f64 / counted as f64
    }
    /// Return learner statistics snapshot.
    pub fn learner_stats(&self) -> ElLearnerStats {
        let n = self.models.len();
        let (total_weight, min_w, max_w) = if n == 0 {
            (0.0, 0.0, 0.0)
        } else {
            let total: f64 = self.model_weights.iter().sum();
            let min = self.model_weights.iter().cloned().fold(f64::MAX, f64::min);
            let max = self.model_weights.iter().cloned().fold(f64::MIN, f64::max);
            (total, min, max)
        };
        ElLearnerStats {
            method: self.config.method,
            n_models: n,
            total_predictions: self.total_predictions,
            total_weight,
            min_weight: min_w,
            max_weight: max_w,
            history_len: self.training_history.len(),
            is_trained: self.is_trained,
        }
    }
    pub(super) fn push_history(&mut self, record: ElTrainingRecord) {
        if self.training_history.len() >= 100 {
            self.training_history.pop_front();
        }
        self.training_history.push_back(record);
    }
}
