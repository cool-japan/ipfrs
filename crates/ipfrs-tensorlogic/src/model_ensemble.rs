//! ModelEnsemble — Multi-model ensemble aggregator for distributed inference.
//!
//! Supports voting, averaging, and stacking strategies with per-member tracking,
//! weight normalization, and statistical disagreement measurement.

use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can arise during ensemble aggregation or configuration.
#[derive(Debug, Error, PartialEq)]
pub enum EnsembleError {
    /// Not enough model predictions to satisfy `min_models`.
    #[error("insufficient models: needed {needed}, got {got}")]
    InsufficientModels { needed: usize, got: usize },

    /// A required model was not found (e.g. `require_all` mode).
    #[error("missing model: {0}")]
    MissingModel(String),

    /// A prediction's `outputs` vector is empty.
    #[error("empty outputs in one or more predictions")]
    EmptyOutputs,

    /// The number of weights does not match the number of models.
    #[error("weight count mismatch: expected {expected}, got {got}")]
    WeightCountMismatch { expected: usize, got: usize },
}

// ---------------------------------------------------------------------------
// Strategy
// ---------------------------------------------------------------------------

/// Aggregation strategy used by the ensemble.
#[derive(Debug, Clone, PartialEq)]
pub enum EnsembleStrategy {
    /// Classification: argmax vote; ties broken by lowest class index.
    MajorityVote,

    /// Classification: each model's vote is multiplied by its weight.
    /// Weights are normalised internally; they need not sum to 1.
    WeightedVote { weights: Vec<f64> },

    /// Regression: arithmetic mean of `outputs[0]` across all models.
    MeanAveraging,

    /// Regression: weighted mean of `outputs[0]`.
    WeightedAveraging { weights: Vec<f64> },

    /// Linear combination of per-model outputs using `meta_weights`.
    /// Each model's full `outputs` vector contributes to the final vector.
    Stacking { meta_weights: Vec<f64> },
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single model's prediction package.
#[derive(Debug, Clone)]
pub struct ModelPrediction {
    /// Unique identifier for the model that produced this prediction.
    pub model_id: String,
    /// For classification: `outputs[i]` = probability of class *i*.
    /// For regression: `outputs[0]` = predicted value.
    pub outputs: Vec<f64>,
    /// Confidence score in `[0, 1]`.
    pub confidence: f64,
    /// Wall-clock inference time in milliseconds.
    pub latency_ms: u64,
}

/// The aggregated output of the ensemble.
#[derive(Debug, Clone)]
pub struct EnsembleResult {
    /// Final aggregated outputs (class probabilities or regression value).
    pub final_outputs: Vec<f64>,
    /// Human-readable name of the strategy that was applied.
    pub strategy_used: String,
    /// Number of models whose predictions were included.
    pub participating_models: usize,
    /// Mean confidence across participating models.
    pub avg_confidence: f64,
    /// Mean latency in milliseconds across participating models.
    pub avg_latency_ms: f64,
    /// Disagreement metric:
    /// * Classification → `1 - max_vote_fraction`
    /// * Regression     → standard deviation of `outputs[0]` across models
    pub disagreement: f64,
}

/// Metadata for a single member of the ensemble.
#[derive(Debug, Clone)]
pub struct ModelMember {
    /// Identifier matching `ModelPrediction::model_id`.
    pub model_id: String,
    /// Default weight used when the strategy is weight-based and no
    /// per-call weight list is provided.
    pub weight: f64,
    /// Whether this member participates in aggregation.
    pub enabled: bool,
    /// Total number of times `record_call` was invoked for this member.
    pub call_count: u64,
    /// Number of failed calls recorded via `record_call(_, false)`.
    pub error_count: u64,
}

/// Configuration for `ModelEnsemble`.
#[derive(Debug, Clone)]
pub struct EnsembleConfig {
    /// Strategy used to aggregate predictions.
    pub strategy: EnsembleStrategy,
    /// Minimum number of active predictions required; defaults to 1.
    pub min_models: usize,
    /// Maximum allowed wall-clock time (informational; not enforced here).
    pub timeout_ms: u64,
    /// If `true`, fail when any registered member has no matching prediction.
    pub require_all: bool,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            strategy: EnsembleStrategy::MeanAveraging,
            min_models: 1,
            timeout_ms: 5_000,
            require_all: false,
        }
    }
}

/// Aggregate statistics over all ensemble members.
#[derive(Debug, Clone, PartialEq)]
pub struct EnsembleStats {
    pub total_members: usize,
    pub enabled_members: usize,
    pub total_calls: u64,
    pub total_errors: u64,
    /// Mean error rate across members that have been called at least once.
    pub avg_member_error_rate: f64,
}

// ---------------------------------------------------------------------------
// ModelEnsemble
// ---------------------------------------------------------------------------

/// Multi-model ensemble aggregator supporting voting, averaging, and stacking.
#[derive(Debug)]
pub struct ModelEnsemble {
    pub config: EnsembleConfig,
    pub members: Vec<ModelMember>,
}

impl ModelEnsemble {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new ensemble with the given configuration.
    pub fn new(config: EnsembleConfig) -> Self {
        Self {
            config,
            members: Vec::new(),
        }
    }

    /// Add a model member. Returns `&mut Self` for builder-style chaining.
    pub fn add_member(&mut self, model_id: String, weight: f64) -> &mut Self {
        self.members.push(ModelMember {
            model_id,
            weight,
            enabled: true,
            call_count: 0,
            error_count: 0,
        });
        self
    }

    // -----------------------------------------------------------------------
    // Member management
    // -----------------------------------------------------------------------

    /// Enable the member with the given id. Returns `false` if not found.
    pub fn enable_member(&mut self, model_id: &str) -> bool {
        match self.members.iter_mut().find(|m| m.model_id == model_id) {
            Some(m) => {
                m.enabled = true;
                true
            }
            None => false,
        }
    }

    /// Disable the member with the given id. Returns `false` if not found.
    pub fn disable_member(&mut self, model_id: &str) -> bool {
        match self.members.iter_mut().find(|m| m.model_id == model_id) {
            Some(m) => {
                m.enabled = false;
                true
            }
            None => false,
        }
    }

    /// Record a call result for a member (updates `call_count` / `error_count`).
    pub fn record_call(&mut self, model_id: &str, success: bool) {
        if let Some(m) = self.members.iter_mut().find(|m| m.model_id == model_id) {
            m.call_count += 1;
            if !success {
                m.error_count += 1;
            }
        }
    }

    /// Immutable references to all members.
    pub fn member_stats(&self) -> Vec<&ModelMember> {
        self.members.iter().collect()
    }

    /// Aggregate statistics over all members.
    pub fn stats(&self) -> EnsembleStats {
        let total_members = self.members.len();
        let enabled_members = self.members.iter().filter(|m| m.enabled).count();
        let total_calls: u64 = self.members.iter().map(|m| m.call_count).sum();
        let total_errors: u64 = self.members.iter().map(|m| m.error_count).sum();

        let rates: Vec<f64> = self
            .members
            .iter()
            .filter(|m| m.call_count > 0)
            .map(|m| m.error_count as f64 / m.call_count as f64)
            .collect();

        let avg_member_error_rate = if rates.is_empty() {
            0.0
        } else {
            rates.iter().sum::<f64>() / rates.len() as f64
        };

        EnsembleStats {
            total_members,
            enabled_members,
            total_calls,
            total_errors,
            avg_member_error_rate,
        }
    }

    // -----------------------------------------------------------------------
    // Core aggregation
    // -----------------------------------------------------------------------

    /// Aggregate a slice of model predictions according to the configured strategy.
    ///
    /// Steps:
    /// 1. Filter out predictions whose `model_id` maps to a disabled member.
    /// 2. Optionally check that every enabled member contributed (`require_all`).
    /// 3. Validate the count against `min_models`.
    /// 4. Apply the strategy.
    pub fn aggregate(
        &self,
        predictions: &[ModelPrediction],
    ) -> Result<EnsembleResult, EnsembleError> {
        // Build fast lookup: model_id → enabled status and weight.
        let member_map: HashMap<&str, (bool, f64)> = self
            .members
            .iter()
            .map(|m| (m.model_id.as_str(), (m.enabled, m.weight)))
            .collect();

        // Filter to predictions from enabled members (unknown model ids are
        // treated as enabled with weight 1.0 — they are not registered members).
        let active: Vec<&ModelPrediction> = predictions
            .iter()
            .filter(|p| {
                member_map
                    .get(p.model_id.as_str())
                    .is_none_or(|(enabled, _)| *enabled)
            })
            .collect();

        // require_all: every enabled member must have a matching prediction.
        if self.config.require_all {
            let active_ids: std::collections::HashSet<&str> =
                active.iter().map(|p| p.model_id.as_str()).collect();
            for member in self.members.iter().filter(|m| m.enabled) {
                if !active_ids.contains(member.model_id.as_str()) {
                    return Err(EnsembleError::MissingModel(member.model_id.clone()));
                }
            }
        }

        // Validate minimum model count.
        let n = active.len();
        if n < self.config.min_models {
            return Err(EnsembleError::InsufficientModels {
                needed: self.config.min_models,
                got: n,
            });
        }

        // Validate that no prediction has empty outputs.
        for p in &active {
            if p.outputs.is_empty() {
                return Err(EnsembleError::EmptyOutputs);
            }
        }

        // Compute per-prediction weights using member registry.
        let pred_weights: Vec<f64> = active
            .iter()
            .map(|p| member_map.get(p.model_id.as_str()).map_or(1.0, |(_, w)| *w))
            .collect();

        // Shared statistics.
        let avg_confidence = active.iter().map(|p| p.confidence).sum::<f64>() / n as f64;
        let avg_latency_ms = active.iter().map(|p| p.latency_ms as f64).sum::<f64>() / n as f64;

        // Dispatch to strategy implementation.
        match &self.config.strategy {
            EnsembleStrategy::MajorityVote => {
                self.majority_vote(&active, avg_confidence, avg_latency_ms)
            }
            EnsembleStrategy::WeightedVote { weights } => self.weighted_vote(
                &active,
                weights,
                &pred_weights,
                avg_confidence,
                avg_latency_ms,
            ),
            EnsembleStrategy::MeanAveraging => {
                self.mean_averaging(&active, avg_confidence, avg_latency_ms)
            }
            EnsembleStrategy::WeightedAveraging { weights } => self.weighted_averaging(
                &active,
                weights,
                &pred_weights,
                avg_confidence,
                avg_latency_ms,
            ),
            EnsembleStrategy::Stacking { meta_weights } => self.stacking(
                &active,
                meta_weights,
                &pred_weights,
                avg_confidence,
                avg_latency_ms,
            ),
        }
    }

    // -----------------------------------------------------------------------
    // Strategy implementations
    // -----------------------------------------------------------------------

    fn majority_vote(
        &self,
        active: &[&ModelPrediction],
        avg_confidence: f64,
        avg_latency_ms: f64,
    ) -> Result<EnsembleResult, EnsembleError> {
        let n_classes = active[0].outputs.len();
        let mut vote_counts = vec![0u64; n_classes];

        for pred in active {
            let cls = Self::top_class(&pred.outputs);
            vote_counts[cls] += 1;
        }

        let total_votes = active.len() as f64;
        let final_outputs: Vec<f64> = vote_counts
            .iter()
            .map(|&c| c as f64 / total_votes)
            .collect();

        // Disagreement: 1 - fraction of votes held by the majority class.
        let max_votes = vote_counts.iter().copied().max().unwrap_or(0);
        let disagreement = 1.0 - (max_votes as f64 / total_votes);

        Ok(EnsembleResult {
            final_outputs,
            strategy_used: "MajorityVote".to_string(),
            participating_models: active.len(),
            avg_confidence,
            avg_latency_ms,
            disagreement,
        })
    }

    fn weighted_vote(
        &self,
        active: &[&ModelPrediction],
        strategy_weights: &[f64],
        member_weights: &[f64],
        avg_confidence: f64,
        avg_latency_ms: f64,
    ) -> Result<EnsembleResult, EnsembleError> {
        // Resolve effective weights: strategy_weights override member_weights
        // when the lengths match; otherwise fall back to member_weights.
        let effective: Vec<f64> = if strategy_weights.len() == active.len() {
            strategy_weights.to_vec()
        } else if !strategy_weights.is_empty() {
            return Err(EnsembleError::WeightCountMismatch {
                expected: active.len(),
                got: strategy_weights.len(),
            });
        } else {
            member_weights.to_vec()
        };

        let normed = Self::normalize_weights(&effective);
        let n_classes = active[0].outputs.len();
        let mut final_outputs = vec![0.0f64; n_classes];

        for (pred, &w) in active.iter().zip(normed.iter()) {
            for (i, &v) in pred.outputs.iter().enumerate().take(n_classes) {
                final_outputs[i] += w * v;
            }
        }

        // Disagreement: 1 - max element in final_outputs (max weighted vote share).
        let max_val = final_outputs
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let disagreement = (1.0 - max_val).max(0.0);

        Ok(EnsembleResult {
            final_outputs,
            strategy_used: "WeightedVote".to_string(),
            participating_models: active.len(),
            avg_confidence,
            avg_latency_ms,
            disagreement,
        })
    }

    fn mean_averaging(
        &self,
        active: &[&ModelPrediction],
        avg_confidence: f64,
        avg_latency_ms: f64,
    ) -> Result<EnsembleResult, EnsembleError> {
        let n = active.len() as f64;
        let mean_val: f64 = active.iter().map(|p| p.outputs[0]).sum::<f64>() / n;
        let disagreement = Self::std_dev(
            active
                .iter()
                .map(|p| p.outputs[0])
                .collect::<Vec<_>>()
                .as_slice(),
        );

        Ok(EnsembleResult {
            final_outputs: vec![mean_val],
            strategy_used: "MeanAveraging".to_string(),
            participating_models: active.len(),
            avg_confidence,
            avg_latency_ms,
            disagreement,
        })
    }

    fn weighted_averaging(
        &self,
        active: &[&ModelPrediction],
        strategy_weights: &[f64],
        member_weights: &[f64],
        avg_confidence: f64,
        avg_latency_ms: f64,
    ) -> Result<EnsembleResult, EnsembleError> {
        let effective: Vec<f64> = if strategy_weights.len() == active.len() {
            strategy_weights.to_vec()
        } else if !strategy_weights.is_empty() {
            return Err(EnsembleError::WeightCountMismatch {
                expected: active.len(),
                got: strategy_weights.len(),
            });
        } else {
            member_weights.to_vec()
        };

        let normed = Self::normalize_weights(&effective);
        let weighted_val: f64 = active
            .iter()
            .zip(normed.iter())
            .map(|(p, &w)| p.outputs[0] * w)
            .sum();

        let disagreement = Self::std_dev(
            active
                .iter()
                .map(|p| p.outputs[0])
                .collect::<Vec<_>>()
                .as_slice(),
        );

        Ok(EnsembleResult {
            final_outputs: vec![weighted_val],
            strategy_used: "WeightedAveraging".to_string(),
            participating_models: active.len(),
            avg_confidence,
            avg_latency_ms,
            disagreement,
        })
    }

    fn stacking(
        &self,
        active: &[&ModelPrediction],
        meta_weights: &[f64],
        member_weights: &[f64],
        avg_confidence: f64,
        avg_latency_ms: f64,
    ) -> Result<EnsembleResult, EnsembleError> {
        // Effective per-model stacking weights (meta_weights if lengths match,
        // otherwise fall back to normalised member weights).
        let effective: Vec<f64> = if meta_weights.len() == active.len() {
            Self::normalize_weights(meta_weights)
        } else if !meta_weights.is_empty() {
            return Err(EnsembleError::WeightCountMismatch {
                expected: active.len(),
                got: meta_weights.len(),
            });
        } else {
            Self::normalize_weights(member_weights)
        };

        // Determine output dimensionality from the first prediction.
        let out_dim = active[0].outputs.len();
        let mut final_outputs = vec![0.0f64; out_dim];

        for (pred, &w) in active.iter().zip(effective.iter()) {
            for (i, &v) in pred.outputs.iter().enumerate().take(out_dim) {
                final_outputs[i] += w * v;
            }
        }

        // Disagreement: std dev across participating models of their scalar
        // outputs (use first element for a consistent scalar).
        let scalars: Vec<f64> = active.iter().map(|p| p.outputs[0]).collect();
        let disagreement = Self::std_dev(&scalars);

        Ok(EnsembleResult {
            final_outputs,
            strategy_used: "Stacking".to_string(),
            participating_models: active.len(),
            avg_confidence,
            avg_latency_ms,
            disagreement,
        })
    }

    // -----------------------------------------------------------------------
    // Utility helpers
    // -----------------------------------------------------------------------

    /// Return the index of the maximum element (argmax). Ties: lowest index.
    pub fn top_class(outputs: &[f64]) -> usize {
        outputs.iter().enumerate().fold(
            0usize,
            |best, (i, &v)| {
                if v > outputs[best] {
                    i
                } else {
                    best
                }
            },
        )
    }

    /// Numerically stable softmax (subtract max before exp).
    pub fn softmax(logits: &[f64]) -> Vec<f64> {
        if logits.is_empty() {
            return Vec::new();
        }
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        if sum == 0.0 {
            vec![1.0 / logits.len() as f64; logits.len()]
        } else {
            exps.iter().map(|&e| e / sum).collect()
        }
    }

    /// Divide each weight by the total sum. If the sum is approximately 0,
    /// return a uniform distribution.
    pub fn normalize_weights(weights: &[f64]) -> Vec<f64> {
        if weights.is_empty() {
            return Vec::new();
        }
        let sum: f64 = weights.iter().sum();
        if sum.abs() < f64::EPSILON {
            vec![1.0 / weights.len() as f64; weights.len()]
        } else {
            weights.iter().map(|&w| w / sum).collect()
        }
    }

    /// Population standard deviation of a slice.
    fn std_dev(values: &[f64]) -> f64 {
        let n = values.len();
        if n <= 1 {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        variance.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::model_ensemble::{
        EnsembleConfig, EnsembleError, EnsembleStrategy, ModelEnsemble, ModelPrediction,
    };

    // -------
    // Helpers
    // -------

    fn pred(id: &str, outputs: Vec<f64>, confidence: f64, latency_ms: u64) -> ModelPrediction {
        ModelPrediction {
            model_id: id.to_string(),
            outputs,
            confidence,
            latency_ms,
        }
    }

    fn basic_ensemble(strategy: EnsembleStrategy) -> ModelEnsemble {
        let cfg = EnsembleConfig {
            strategy,
            min_models: 1,
            timeout_ms: 1_000,
            require_all: false,
        };
        ModelEnsemble::new(cfg)
    }

    // -----------------------------------------------------------------------
    // top_class
    // -----------------------------------------------------------------------

    #[test]
    fn test_top_class_simple() {
        assert_eq!(ModelEnsemble::top_class(&[0.1, 0.8, 0.1]), 1);
    }

    #[test]
    fn test_top_class_first_wins_tie() {
        // Tie between index 0 and 2 → lowest wins (index 0).
        assert_eq!(ModelEnsemble::top_class(&[0.5, 0.0, 0.5]), 0);
    }

    #[test]
    fn test_top_class_single_element() {
        assert_eq!(ModelEnsemble::top_class(&[42.0]), 0);
    }

    #[test]
    fn test_top_class_all_equal() {
        assert_eq!(ModelEnsemble::top_class(&[1.0, 1.0, 1.0]), 0);
    }

    // -----------------------------------------------------------------------
    // softmax
    // -----------------------------------------------------------------------

    #[test]
    fn test_softmax_sums_to_one() {
        let out = ModelEnsemble::softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_softmax_numerically_stable_large_inputs() {
        // Would overflow without the max-subtraction trick.
        let out = ModelEnsemble::softmax(&[1000.0, 1001.0, 1002.0]);
        let sum: f64 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_softmax_empty() {
        assert!(ModelEnsemble::softmax(&[]).is_empty());
    }

    #[test]
    fn test_softmax_uniform_on_equal_inputs() {
        let out = ModelEnsemble::softmax(&[0.0, 0.0, 0.0]);
        for v in &out {
            assert!((v - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_softmax_argmax_preserved() {
        let logits = &[0.5, 3.0, 1.0];
        let out = ModelEnsemble::softmax(logits);
        assert_eq!(ModelEnsemble::top_class(&out), 1);
    }

    // -----------------------------------------------------------------------
    // normalize_weights
    // -----------------------------------------------------------------------

    #[test]
    fn test_normalize_weights_basic() {
        let w = ModelEnsemble::normalize_weights(&[1.0, 3.0]);
        assert!((w[0] - 0.25).abs() < 1e-12);
        assert!((w[1] - 0.75).abs() < 1e-12);
    }

    #[test]
    fn test_normalize_weights_already_normed() {
        let w = ModelEnsemble::normalize_weights(&[0.4, 0.6]);
        assert!((w[0] - 0.4).abs() < 1e-12);
        assert!((w[1] - 0.6).abs() < 1e-12);
    }

    #[test]
    fn test_normalize_weights_zero_sum_gives_uniform() {
        let w = ModelEnsemble::normalize_weights(&[0.0, 0.0, 0.0]);
        for v in &w {
            assert!((v - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_normalize_weights_empty() {
        assert!(ModelEnsemble::normalize_weights(&[]).is_empty());
    }

    // -----------------------------------------------------------------------
    // MajorityVote
    // -----------------------------------------------------------------------

    #[test]
    fn test_majority_vote_clear_winner() {
        let mut e = basic_ensemble(EnsembleStrategy::MajorityVote);
        e.add_member("a".into(), 1.0)
            .add_member("b".into(), 1.0)
            .add_member("c".into(), 1.0);

        let preds = vec![
            pred("a", vec![0.9, 0.1], 0.9, 10),
            pred("b", vec![0.8, 0.2], 0.8, 12),
            pred("c", vec![0.1, 0.9], 0.7, 8),
        ];

        let res = e.aggregate(&preds).expect("aggregate");
        // Two votes for class 0, one for class 1.
        assert!((res.final_outputs[0] - 2.0 / 3.0).abs() < 1e-12);
        assert!((res.final_outputs[1] - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(res.strategy_used, "MajorityVote");
        assert_eq!(res.participating_models, 3);
    }

    #[test]
    fn test_majority_vote_tie_lowest_class_wins() {
        let mut e = basic_ensemble(EnsembleStrategy::MajorityVote);
        e.add_member("a".into(), 1.0).add_member("b".into(), 1.0);

        let preds = vec![
            pred("a", vec![0.9, 0.1], 0.9, 10),
            pred("b", vec![0.1, 0.9], 0.9, 10),
        ];

        let res = e.aggregate(&preds).expect("aggregate");
        // Tie → equal vote shares, final_outputs = [0.5, 0.5].
        assert!((res.final_outputs[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_majority_vote_disagreement_unanimous() {
        let mut e = basic_ensemble(EnsembleStrategy::MajorityVote);
        e.add_member("a".into(), 1.0).add_member("b".into(), 1.0);

        let preds = vec![
            pred("a", vec![1.0, 0.0], 1.0, 5),
            pred("b", vec![1.0, 0.0], 1.0, 5),
        ];

        let res = e.aggregate(&preds).expect("aggregate");
        // Unanimous → disagreement = 0.
        assert!(res.disagreement.abs() < 1e-12);
    }

    #[test]
    fn test_majority_vote_avg_stats() {
        let mut e = basic_ensemble(EnsembleStrategy::MajorityVote);
        e.add_member("a".into(), 1.0).add_member("b".into(), 1.0);

        let preds = vec![
            pred("a", vec![1.0, 0.0], 0.6, 10),
            pred("b", vec![1.0, 0.0], 0.8, 20),
        ];

        let res = e.aggregate(&preds).expect("aggregate");
        assert!((res.avg_confidence - 0.7).abs() < 1e-12);
        assert!((res.avg_latency_ms - 15.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // WeightedVote
    // -----------------------------------------------------------------------

    #[test]
    fn test_weighted_vote_basic() {
        let strategy = EnsembleStrategy::WeightedVote {
            weights: vec![3.0, 1.0],
        };
        let mut e = basic_ensemble(strategy);
        e.add_member("a".into(), 1.0).add_member("b".into(), 1.0);

        let preds = vec![
            pred("a", vec![0.8, 0.2], 0.9, 10), // weight 3
            pred("b", vec![0.2, 0.8], 0.7, 10), // weight 1
        ];

        let res = e.aggregate(&preds).expect("aggregate");
        // Normalised weights: [0.75, 0.25]
        // final_outputs[0] = 0.75*0.8 + 0.25*0.2 = 0.65
        // final_outputs[1] = 0.75*0.2 + 0.25*0.8 = 0.35
        assert!((res.final_outputs[0] - 0.65).abs() < 1e-12);
        assert!((res.final_outputs[1] - 0.35).abs() < 1e-12);
        assert_eq!(res.strategy_used, "WeightedVote");
    }

    #[test]
    fn test_weighted_vote_mismatch_error() {
        let strategy = EnsembleStrategy::WeightedVote {
            weights: vec![1.0], // only 1 weight for 2 models
        };
        let mut e = basic_ensemble(strategy);
        e.add_member("a".into(), 1.0).add_member("b".into(), 1.0);

        let preds = vec![
            pred("a", vec![1.0, 0.0], 0.9, 10),
            pred("b", vec![0.0, 1.0], 0.8, 10),
        ];

        let err = e.aggregate(&preds).expect_err("should fail");
        assert!(matches!(err, EnsembleError::WeightCountMismatch { .. }));
    }

    // -----------------------------------------------------------------------
    // MeanAveraging
    // -----------------------------------------------------------------------

    #[test]
    fn test_mean_averaging_basic() {
        let mut e = basic_ensemble(EnsembleStrategy::MeanAveraging);
        e.add_member("a".into(), 1.0).add_member("b".into(), 1.0);

        let preds = vec![pred("a", vec![2.0], 0.8, 5), pred("b", vec![4.0], 0.6, 15)];

        let res = e.aggregate(&preds).expect("aggregate");
        assert!((res.final_outputs[0] - 3.0).abs() < 1e-12);
        assert_eq!(res.strategy_used, "MeanAveraging");
    }

    #[test]
    fn test_mean_averaging_single_model() {
        let mut e = basic_ensemble(EnsembleStrategy::MeanAveraging);
        e.add_member("a".into(), 1.0);

        let preds = vec![pred("a", vec![7.5], 1.0, 1)];

        let res = e.aggregate(&preds).expect("aggregate");
        assert!((res.final_outputs[0] - 7.5).abs() < 1e-12);
        assert!(res.disagreement.abs() < 1e-12);
    }

    #[test]
    fn test_mean_averaging_disagreement_nonzero() {
        let mut e = basic_ensemble(EnsembleStrategy::MeanAveraging);
        e.add_member("a".into(), 1.0).add_member("b".into(), 1.0);

        let preds = vec![pred("a", vec![1.0], 0.9, 5), pred("b", vec![3.0], 0.9, 5)];

        let res = e.aggregate(&preds).expect("aggregate");
        // std_dev([1,3]) = 1.0
        assert!((res.disagreement - 1.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // WeightedAveraging
    // -----------------------------------------------------------------------

    #[test]
    fn test_weighted_averaging_basic() {
        let strategy = EnsembleStrategy::WeightedAveraging {
            weights: vec![1.0, 3.0],
        };
        let mut e = basic_ensemble(strategy);
        e.add_member("a".into(), 1.0).add_member("b".into(), 1.0);

        let preds = vec![pred("a", vec![0.0], 0.9, 10), pred("b", vec![4.0], 0.9, 10)];

        let res = e.aggregate(&preds).expect("aggregate");
        // Normalised weights: [0.25, 0.75]
        // 0.25*0.0 + 0.75*4.0 = 3.0
        assert!((res.final_outputs[0] - 3.0).abs() < 1e-12);
        assert_eq!(res.strategy_used, "WeightedAveraging");
    }

    #[test]
    fn test_weighted_averaging_fallback_to_member_weights() {
        // Empty strategy weights → fall back to member weights.
        let strategy = EnsembleStrategy::WeightedAveraging { weights: vec![] };
        let mut e = basic_ensemble(strategy);
        // member weights: a=1, b=3
        e.add_member("a".into(), 1.0).add_member("b".into(), 3.0);

        let preds = vec![pred("a", vec![0.0], 0.9, 10), pred("b", vec![4.0], 0.9, 10)];

        let res = e.aggregate(&preds).expect("aggregate");
        // normalised [0.25, 0.75] → 3.0
        assert!((res.final_outputs[0] - 3.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Stacking
    // -----------------------------------------------------------------------

    #[test]
    fn test_stacking_basic() {
        let strategy = EnsembleStrategy::Stacking {
            meta_weights: vec![0.5, 0.5],
        };
        let mut e = basic_ensemble(strategy);
        e.add_member("a".into(), 1.0).add_member("b".into(), 1.0);

        let preds = vec![
            pred("a", vec![2.0, 4.0], 0.9, 10),
            pred("b", vec![6.0, 8.0], 0.8, 10),
        ];

        let res = e.aggregate(&preds).expect("aggregate");
        // Normalised weights both 0.5.
        // final[0] = 0.5*2 + 0.5*6 = 4, final[1] = 0.5*4 + 0.5*8 = 6
        assert!((res.final_outputs[0] - 4.0).abs() < 1e-12);
        assert!((res.final_outputs[1] - 6.0).abs() < 1e-12);
        assert_eq!(res.strategy_used, "Stacking");
    }

    #[test]
    fn test_stacking_mismatch_error() {
        let strategy = EnsembleStrategy::Stacking {
            meta_weights: vec![1.0, 2.0, 3.0], // 3 weights but 2 models
        };
        let mut e = basic_ensemble(strategy);
        e.add_member("a".into(), 1.0).add_member("b".into(), 1.0);

        let preds = vec![pred("a", vec![1.0], 0.9, 5), pred("b", vec![2.0], 0.9, 5)];

        let err = e.aggregate(&preds).expect_err("should fail");
        assert!(matches!(err, EnsembleError::WeightCountMismatch { .. }));
    }

    // -----------------------------------------------------------------------
    // Error paths
    // -----------------------------------------------------------------------

    #[test]
    fn test_insufficient_models_error() {
        let cfg = EnsembleConfig {
            strategy: EnsembleStrategy::MeanAveraging,
            min_models: 3,
            timeout_ms: 1_000,
            require_all: false,
        };
        let e = ModelEnsemble::new(cfg);
        let preds = vec![pred("a", vec![1.0], 0.9, 5)];
        let err = e.aggregate(&preds).expect_err("should fail");
        assert_eq!(err, EnsembleError::InsufficientModels { needed: 3, got: 1 });
    }

    #[test]
    fn test_empty_outputs_error() {
        let mut e = basic_ensemble(EnsembleStrategy::MeanAveraging);
        e.add_member("a".into(), 1.0);

        let preds = vec![ModelPrediction {
            model_id: "a".into(),
            outputs: vec![],
            confidence: 0.9,
            latency_ms: 5,
        }];

        let err = e.aggregate(&preds).expect_err("should fail");
        assert_eq!(err, EnsembleError::EmptyOutputs);
    }

    #[test]
    fn test_require_all_missing_member_error() {
        let cfg = EnsembleConfig {
            strategy: EnsembleStrategy::MeanAveraging,
            min_models: 1,
            timeout_ms: 1_000,
            require_all: true,
        };
        let mut e = ModelEnsemble::new(cfg);
        e.add_member("a".into(), 1.0).add_member("b".into(), 1.0);

        // Only model "a" sends a prediction.
        let preds = vec![pred("a", vec![1.0], 0.9, 5)];
        let err = e.aggregate(&preds).expect_err("should fail");
        assert!(matches!(err, EnsembleError::MissingModel(_)));
    }

    // -----------------------------------------------------------------------
    // Member management
    // -----------------------------------------------------------------------

    #[test]
    fn test_enable_disable_member() {
        let mut e = basic_ensemble(EnsembleStrategy::MajorityVote);
        e.add_member("a".into(), 1.0).add_member("b".into(), 1.0);

        assert!(e.disable_member("a"));
        let preds = vec![
            pred("a", vec![0.0, 1.0], 0.9, 5),
            pred("b", vec![1.0, 0.0], 0.9, 5),
        ];
        // "a" is disabled → only "b" participates.
        let res = e.aggregate(&preds).expect("aggregate");
        assert_eq!(res.participating_models, 1);

        assert!(e.enable_member("a"));
        let res2 = e.aggregate(&preds).expect("aggregate after re-enable");
        assert_eq!(res2.participating_models, 2);
    }

    #[test]
    fn test_enable_nonexistent_returns_false() {
        let mut e = basic_ensemble(EnsembleStrategy::MeanAveraging);
        assert!(!e.enable_member("ghost"));
    }

    #[test]
    fn test_disable_nonexistent_returns_false() {
        let mut e = basic_ensemble(EnsembleStrategy::MeanAveraging);
        assert!(!e.disable_member("ghost"));
    }

    #[test]
    fn test_record_call_updates_counts() {
        let mut e = basic_ensemble(EnsembleStrategy::MeanAveraging);
        e.add_member("a".into(), 1.0);

        e.record_call("a", true);
        e.record_call("a", false);
        e.record_call("a", true);

        let m = &e.members[0];
        assert_eq!(m.call_count, 3);
        assert_eq!(m.error_count, 1);
    }

    #[test]
    fn test_record_call_unknown_model_no_panic() {
        let mut e = basic_ensemble(EnsembleStrategy::MeanAveraging);
        // Should silently do nothing.
        e.record_call("ghost", true);
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_no_calls() {
        let mut e = basic_ensemble(EnsembleStrategy::MeanAveraging);
        e.add_member("a".into(), 1.0).add_member("b".into(), 1.0);
        e.disable_member("b");

        let s = e.stats();
        assert_eq!(s.total_members, 2);
        assert_eq!(s.enabled_members, 1);
        assert_eq!(s.total_calls, 0);
        assert_eq!(s.total_errors, 0);
        assert!((s.avg_member_error_rate).abs() < 1e-12);
    }

    #[test]
    fn test_stats_with_calls() {
        let mut e = basic_ensemble(EnsembleStrategy::MeanAveraging);
        e.add_member("a".into(), 1.0).add_member("b".into(), 1.0);

        e.record_call("a", true); // a: 1 call, 0 errors
        e.record_call("b", false); // b: 1 call, 1 error

        let s = e.stats();
        assert_eq!(s.total_calls, 2);
        assert_eq!(s.total_errors, 1);
        // avg of [0.0, 1.0] = 0.5
        assert!((s.avg_member_error_rate - 0.5).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Unknown model ids (not registered members)
    // -----------------------------------------------------------------------

    #[test]
    fn test_unregistered_model_participates_with_default_weight() {
        // No members registered; any prediction gets through.
        let e = basic_ensemble(EnsembleStrategy::MeanAveraging);
        let preds = vec![pred("unknown", vec![5.0], 0.9, 10)];
        let res = e.aggregate(&preds).expect("aggregate");
        assert!((res.final_outputs[0] - 5.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // member_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_member_stats_returns_all() {
        let mut e = basic_ensemble(EnsembleStrategy::MeanAveraging);
        e.add_member("a".into(), 2.0).add_member("b".into(), 3.0);
        let stats = e.member_stats();
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0].model_id, "a");
        assert_eq!(stats[1].model_id, "b");
    }

    // -----------------------------------------------------------------------
    // EnsembleConfig defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_config() {
        let cfg = EnsembleConfig::default();
        assert_eq!(cfg.min_models, 1);
        assert_eq!(cfg.timeout_ms, 5_000);
        assert!(!cfg.require_all);
    }

    // -----------------------------------------------------------------------
    // Softmax edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_softmax_single_element() {
        let out = ModelEnsemble::softmax(&[42.0]);
        assert!((out[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_softmax_negative_inputs() {
        let out = ModelEnsemble::softmax(&[-1.0, -2.0, -3.0]);
        let sum: f64 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        // -1.0 should be the max → largest probability.
        assert!(out[0] > out[1]);
        assert!(out[1] > out[2]);
    }

    // -----------------------------------------------------------------------
    // Disagrement edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_mean_averaging_three_models_disagrement() {
        let mut e = basic_ensemble(EnsembleStrategy::MeanAveraging);
        e.add_member("a".into(), 1.0)
            .add_member("b".into(), 1.0)
            .add_member("c".into(), 1.0);

        let preds = vec![
            pred("a", vec![1.0], 0.9, 5),
            pred("b", vec![2.0], 0.9, 5),
            pred("c", vec![3.0], 0.9, 5),
        ];

        let res = e.aggregate(&preds).expect("aggregate");
        assert!((res.final_outputs[0] - 2.0).abs() < 1e-12);
        // variance = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
        // std_dev = sqrt(2/3) ≈ 0.8165
        let expected_std = (2.0_f64 / 3.0).sqrt();
        assert!((res.disagreement - expected_std).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Weighted vote using member weights (empty strategy weights)
    // -----------------------------------------------------------------------

    #[test]
    fn test_weighted_vote_fallback_to_member_weights() {
        let strategy = EnsembleStrategy::WeightedVote { weights: vec![] };
        let mut e = basic_ensemble(strategy);
        e.add_member("a".into(), 1.0).add_member("b".into(), 3.0);

        let preds = vec![
            pred("a", vec![1.0, 0.0], 0.9, 10),
            pred("b", vec![0.0, 1.0], 0.8, 10),
        ];

        let res = e.aggregate(&preds).expect("aggregate");
        // normed weights: [0.25, 0.75]
        // final[0] = 0.25*1 + 0.75*0 = 0.25
        // final[1] = 0.25*0 + 0.75*1 = 0.75
        assert!((res.final_outputs[0] - 0.25).abs() < 1e-12);
        assert!((res.final_outputs[1] - 0.75).abs() < 1e-12);
    }
}
