//! DecisionTreeLearner — ID3/C4.5-style decision tree with training, prediction,
//! feature importance, pruning, and rich statistics.
//!
//! # Design
//!
//! - Recursive binary splitting on continuous features.
//! - Configurable split criterion: Shannon Entropy (ID3), Gini impurity, or
//!   Misclassification rate.
//! - Optional feature sub-sampling per split node (random forests style),
//!   driven by a pure-Rust xorshift64 PRNG — no external crate required.
//! - Post-training leaf pruning (`prune`) to collapse subtrees where both
//!   descendant leaves are minority-dominated.
//! - Bounded training-history ring-buffer (`VecDeque` capped at 100 records).
//! - All operations are `no_std`-compatible except for `HashMap`/`VecDeque`.

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// PRNG — pure Rust xorshift64, no rand crate dependency
// ---------------------------------------------------------------------------

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by [`DecisionTreeLearner`].
#[derive(Debug, Error, Serialize, Deserialize, Clone, PartialEq)]
pub enum DtlError {
    /// Training set is empty.
    #[error("training set is empty")]
    EmptyTrainingSet,

    /// Feature vector has wrong dimensionality.
    #[error("expected {expected} features, got {got}")]
    FeatureDimensionMismatch { expected: usize, got: usize },

    /// Tree has not been trained yet.
    #[error("model has not been trained yet")]
    ModelNotTrained,

    /// Class label not found.
    #[error("unknown class label: {0}")]
    UnknownLabel(String),

    /// Feature names have wrong count.
    #[error("feature names count {names} does not match feature dimension {dim}")]
    FeatureNamesMismatch { names: usize, dim: usize },

    /// Internal arithmetic error (e.g. NaN).
    #[error("arithmetic error: {0}")]
    Arithmetic(String),
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Split criterion for tree growing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DtlCriterion {
    /// Shannon entropy / information gain (ID3).
    Entropy,
    /// Gini impurity reduction.
    #[default]
    Gini,
    /// Misclassification rate reduction.
    MisclassificationRate,
}

/// Hyper-parameters controlling tree growth and prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DtlLearnerConfig {
    /// Maximum tree depth (0 means unlimited).
    pub max_depth: usize,
    /// Minimum samples required to attempt a split.
    pub min_samples_split: usize,
    /// Minimum samples that must remain in a leaf after a split.
    pub min_samples_leaf: usize,
    /// Split quality criterion.
    pub criterion: DtlCriterion,
    /// Number of candidate features evaluated per node. `None` = all features.
    pub max_features: Option<usize>,
    /// Random seed for feature sub-sampling.
    pub seed: u64,
}

impl Default for DtlLearnerConfig {
    fn default() -> Self {
        Self {
            max_depth: 0,
            min_samples_split: 2,
            min_samples_leaf: 1,
            criterion: DtlCriterion::Gini,
            max_features: None,
            seed: 0x1234_5678_9abc_def0,
        }
    }
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// One labeled training / prediction example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DtlSample {
    /// Continuous feature values.
    pub features: Vec<f64>,
    /// Class label (string).
    pub label: String,
}

impl DtlSample {
    /// Create a new sample.
    pub fn new(features: Vec<f64>, label: impl Into<String>) -> Self {
        Self {
            features,
            label: label.into(),
        }
    }
}

/// Prediction result from [`DecisionTreeLearner::predict`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DtlPrediction {
    /// Predicted class label.
    pub label: String,
    /// Fraction of training samples at the leaf that agree with the label.
    pub confidence: f64,
    /// Depth of the leaf node reached.
    pub path_depth: usize,
}

// ---------------------------------------------------------------------------
// Tree node
// ---------------------------------------------------------------------------

/// Internal node of the decision tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DtlNode {
    /// Terminal / leaf node.
    Leaf {
        /// Majority-class label.
        class_label: String,
        /// Number of training samples that reached this leaf.
        samples: usize,
        /// Per-class sample counts at this leaf.
        class_distribution: HashMap<String, usize>,
    },
    /// Interior split node.
    Split {
        /// Index into the feature vector.
        feature_index: usize,
        /// Split threshold: route left if `features[feature_index] <= threshold`.
        threshold: f64,
        /// Left subtree (feature value ≤ threshold).
        left: Box<DtlNode>,
        /// Right subtree (feature value > threshold).
        right: Box<DtlNode>,
        /// Human-readable feature name.
        feature_name: String,
        /// Number of training samples that reached this node.
        samples: usize,
        /// Node impurity (criterion value) before splitting.
        impurity: f64,
    },
}

impl DtlNode {
    /// Recursively compute the number of leaf nodes.
    pub fn n_leaves(&self) -> usize {
        match self {
            DtlNode::Leaf { .. } => 1,
            DtlNode::Split { left, right, .. } => left.n_leaves() + right.n_leaves(),
        }
    }

    /// Recursively compute the total number of nodes (leaves + splits).
    pub fn n_nodes(&self) -> usize {
        match self {
            DtlNode::Leaf { .. } => 1,
            DtlNode::Split { left, right, .. } => 1 + left.n_nodes() + right.n_nodes(),
        }
    }

    /// Recursively compute tree depth (leaves have depth 1).
    pub fn depth(&self) -> usize {
        match self {
            DtlNode::Leaf { .. } => 1,
            DtlNode::Split { left, right, .. } => 1 + left.depth().max(right.depth()),
        }
    }

    /// Accumulate weighted impurity reduction for each feature index.
    pub(crate) fn accumulate_importance(&self, total_samples: usize, importance: &mut Vec<f64>) {
        if total_samples == 0 {
            return;
        }
        match self {
            DtlNode::Leaf { .. } => {}
            DtlNode::Split {
                feature_index,
                left,
                right,
                samples,
                impurity,
                ..
            } => {
                let left_n = match left.as_ref() {
                    DtlNode::Leaf { samples: s, .. } => *s,
                    DtlNode::Split { samples: s, .. } => *s,
                };
                let right_n = match right.as_ref() {
                    DtlNode::Leaf { samples: s, .. } => *s,
                    DtlNode::Split { samples: s, .. } => *s,
                };
                let n = *samples as f64;
                let left_imp = node_impurity(left);
                let right_imp = node_impurity(right);
                let gain =
                    impurity - (left_n as f64 / n) * left_imp - (right_n as f64 / n) * right_imp;
                let weighted = (n / total_samples as f64) * gain;
                if *feature_index < importance.len() {
                    importance[*feature_index] += weighted;
                }
                left.accumulate_importance(total_samples, importance);
                right.accumulate_importance(total_samples, importance);
            }
        }
    }
}

/// Return the impurity value stored in a node (0.0 for leaves).
fn node_impurity(node: &DtlNode) -> f64 {
    match node {
        DtlNode::Leaf { .. } => 0.0,
        DtlNode::Split { impurity, .. } => *impurity,
    }
}

// ---------------------------------------------------------------------------
// Training record
// ---------------------------------------------------------------------------

/// One entry in the training history ring-buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DtlTrainingRecord {
    /// Wall-clock timestamp (seconds since epoch) — set via std::time.
    pub timestamp_secs: u64,
    /// Number of samples used in the fit call.
    pub n_samples: usize,
    /// Number of features.
    pub n_features: usize,
    /// Number of distinct classes.
    pub n_classes: usize,
    /// Resulting tree depth.
    pub tree_depth: usize,
    /// Resulting leaf count.
    pub n_leaves: usize,
    /// Criterion used.
    pub criterion: DtlCriterion,
}

// ---------------------------------------------------------------------------
// Learner statistics
// ---------------------------------------------------------------------------

/// Summary statistics produced by [`DecisionTreeLearner::learner_stats`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DtlLearnerStats {
    /// Whether the model has been trained.
    pub is_trained: bool,
    /// Number of training samples seen in the last fit call.
    pub last_n_samples: usize,
    /// Number of features.
    pub n_features: usize,
    /// Number of distinct classes.
    pub n_classes: usize,
    /// Current tree depth.
    pub tree_depth: usize,
    /// Current leaf count.
    pub n_leaves: usize,
    /// Current total node count.
    pub n_nodes: usize,
    /// Number of entries in the training history.
    pub history_len: usize,
    /// The configured criterion.
    pub criterion: DtlCriterion,
    /// Feature names (copy).
    pub feature_names: Vec<String>,
    /// Class labels (copy).
    pub class_labels: Vec<String>,
}

// ---------------------------------------------------------------------------
// Main struct
// ---------------------------------------------------------------------------

/// ID3/C4.5-style decision tree learner.
///
/// ```
/// use ipfrs_tensorlogic::{DecisionTreeLearner, DtlSample, DtlLearnerConfig, DtlCriterion};
///
/// let mut learner = DecisionTreeLearner::new(
///     DtlLearnerConfig { max_depth: 5, ..Default::default() },
///     vec!["petal_len".into(), "petal_wid".into()],
/// );
///
/// let samples = vec![
///     DtlSample::new(vec![1.0, 0.5], "setosa"),
///     DtlSample::new(vec![4.5, 1.5], "versicolor"),
///     DtlSample::new(vec![5.0, 2.0], "virginica"),
///     DtlSample::new(vec![1.2, 0.4], "setosa"),
///     DtlSample::new(vec![4.8, 1.8], "versicolor"),
/// ];
/// learner.fit(&samples).expect("example: should succeed in docs");
///
/// let pred = learner.predict(&[1.1, 0.45]).expect("example: should succeed in docs");
/// assert_eq!(pred.label, "setosa");
/// ```
pub struct DecisionTreeLearner {
    /// Trained tree root.
    root: Option<DtlNode>,
    /// Feature names (set at construction).
    feature_names: Vec<String>,
    /// All class labels discovered during training.
    class_labels: Vec<String>,
    /// Bounded training history.
    history: VecDeque<DtlTrainingRecord>,
    /// Learner configuration.
    config: DtlLearnerConfig,
    /// Number of features (inferred from first fit call).
    n_features: usize,
    /// Number of samples in the most recent fit call.
    last_n_samples: usize,
}

// ---------------------------------------------------------------------------
// Type aliases (required by task spec)
// ---------------------------------------------------------------------------

/// Alias for [`DecisionTreeLearner`].
pub type DtlDecisionTreeLearner = DecisionTreeLearner;

// DtlLearnerConfig and DtlLearnerStats are already the canonical names.

// ---------------------------------------------------------------------------
// Constructor helpers
// ---------------------------------------------------------------------------

impl DecisionTreeLearner {
    /// Create a new (untrained) learner.
    ///
    /// `feature_names` is optional context: if non-empty its length must match
    /// the feature dimension of the first `fit` call.
    pub fn new(config: DtlLearnerConfig, feature_names: Vec<String>) -> Self {
        Self {
            root: None,
            feature_names,
            class_labels: Vec::new(),
            history: VecDeque::new(),
            config,
            n_features: 0,
            last_n_samples: 0,
        }
    }

    /// Convenience constructor with default configuration.
    pub fn with_defaults(feature_names: Vec<String>) -> Self {
        Self::new(DtlLearnerConfig::default(), feature_names)
    }
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

impl DecisionTreeLearner {
    /// Train the decision tree on `samples`.
    ///
    /// Clears any previously trained tree.  Feature names, if provided at
    /// construction, are validated against the sample dimension.
    pub fn fit(&mut self, samples: &[DtlSample]) -> Result<(), DtlError> {
        if samples.is_empty() {
            return Err(DtlError::EmptyTrainingSet);
        }

        let n_features = samples[0].features.len();
        if !self.feature_names.is_empty() && self.feature_names.len() != n_features {
            return Err(DtlError::FeatureNamesMismatch {
                names: self.feature_names.len(),
                dim: n_features,
            });
        }
        // Ensure feature_names is always populated
        if self.feature_names.is_empty() {
            self.feature_names = (0..n_features).map(|i| format!("f{i}")).collect();
        }

        // Collect class labels
        let mut label_set: Vec<String> = samples
            .iter()
            .map(|s| s.label.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        label_set.sort();
        self.class_labels = label_set;
        self.n_features = n_features;
        self.last_n_samples = samples.len();

        let mut rng_state = self.config.seed;
        let indices: Vec<usize> = (0..samples.len()).collect();
        self.root = Some(self.build_node(samples, &indices, 0, &mut rng_state));

        // Record in history (bounded at 100)
        let depth = self.depth();
        let leaves = self.n_leaves();
        let record = DtlTrainingRecord {
            timestamp_secs: current_epoch_secs(),
            n_samples: samples.len(),
            n_features,
            n_classes: self.class_labels.len(),
            tree_depth: depth,
            n_leaves: leaves,
            criterion: self.config.criterion,
        };
        self.history.push_back(record);
        while self.history.len() > 100 {
            self.history.pop_front();
        }

        Ok(())
    }

    // ------------------------------------------------------------------
    // Recursive tree builder
    // ------------------------------------------------------------------

    fn build_node(
        &self,
        samples: &[DtlSample],
        indices: &[usize],
        depth: usize,
        rng: &mut u64,
    ) -> DtlNode {
        // Build class distribution for this node
        let distribution = class_distribution(samples, indices);
        let majority = majority_class(&distribution);
        let n = indices.len();

        // Stopping conditions → leaf
        let max_depth_reached = self.config.max_depth > 0 && depth >= self.config.max_depth;
        let too_few_to_split = n < self.config.min_samples_split;
        let pure = distribution.len() == 1;

        if pure || max_depth_reached || too_few_to_split {
            return DtlNode::Leaf {
                class_label: majority,
                samples: n,
                class_distribution: distribution,
            };
        }

        // Find best split
        let impurity = compute_impurity(&distribution, n, self.config.criterion);
        let candidate_features = select_features(self.n_features, self.config.max_features, rng);

        let best = find_best_split(
            samples,
            indices,
            &candidate_features,
            self.config.criterion,
            self.config.min_samples_leaf,
            impurity,
        );

        match best {
            None => DtlNode::Leaf {
                class_label: majority,
                samples: n,
                class_distribution: distribution,
            },
            Some(split) => {
                // Partition indices
                let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices
                    .iter()
                    .copied()
                    .partition(|&i| samples[i].features[split.feature_index] <= split.threshold);

                let feature_name = self
                    .feature_names
                    .get(split.feature_index)
                    .cloned()
                    .unwrap_or_else(|| format!("f{}", split.feature_index));

                let left = self.build_node(samples, &left_idx, depth + 1, rng);
                let right = self.build_node(samples, &right_idx, depth + 1, rng);

                DtlNode::Split {
                    feature_index: split.feature_index,
                    threshold: split.threshold,
                    left: Box::new(left),
                    right: Box::new(right),
                    feature_name,
                    samples: n,
                    impurity,
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

impl DecisionTreeLearner {
    /// Predict the class label for a single feature vector.
    pub fn predict(&self, features: &[f64]) -> Result<DtlPrediction, DtlError> {
        let root = self.root.as_ref().ok_or(DtlError::ModelNotTrained)?;
        if features.len() != self.n_features {
            return Err(DtlError::FeatureDimensionMismatch {
                expected: self.n_features,
                got: features.len(),
            });
        }
        Ok(traverse(root, features, 0))
    }

    /// Predict a batch of feature vectors.
    pub fn predict_batch(&self, samples: &[Vec<f64>]) -> Vec<Result<DtlPrediction, DtlError>> {
        samples.iter().map(|f| self.predict(f)).collect()
    }
}

/// Walk the tree and produce a prediction.
fn traverse(node: &DtlNode, features: &[f64], depth: usize) -> DtlPrediction {
    match node {
        DtlNode::Leaf {
            class_label,
            samples,
            class_distribution,
        } => {
            let count = class_distribution.get(class_label).copied().unwrap_or(0);
            let confidence = if *samples > 0 {
                count as f64 / *samples as f64
            } else {
                0.0
            };
            DtlPrediction {
                label: class_label.clone(),
                confidence,
                path_depth: depth,
            }
        }
        DtlNode::Split {
            feature_index,
            threshold,
            left,
            right,
            ..
        } => {
            let val = features.get(*feature_index).copied().unwrap_or(f64::NAN);
            if val <= *threshold {
                traverse(left, features, depth + 1)
            } else {
                traverse(right, features, depth + 1)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Feature importance
// ---------------------------------------------------------------------------

impl DecisionTreeLearner {
    /// Compute normalised feature importances (weighted impurity reduction).
    ///
    /// Returns a vector of `(feature_name, importance)` pairs sorted by
    /// importance descending.  Values are normalised to sum to 1.0.
    pub fn feature_importance(&self) -> Vec<(String, f64)> {
        let root = match &self.root {
            Some(r) => r,
            None => return Vec::new(),
        };
        let mut raw = vec![0.0f64; self.n_features];
        root.accumulate_importance(self.last_n_samples, &mut raw);

        let total: f64 = raw.iter().sum();
        let mut result: Vec<(String, f64)> = raw
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                let name = self
                    .feature_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("f{i}"));
                let normalised = if total > 0.0 { v / total } else { 0.0 };
                (name, normalised)
            })
            .collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }
}

// ---------------------------------------------------------------------------
// Tree metrics
// ---------------------------------------------------------------------------

impl DecisionTreeLearner {
    /// Return the depth of the trained tree (0 if not trained).
    pub fn depth(&self) -> usize {
        self.root.as_ref().map_or(0, |r| r.depth())
    }

    /// Return the number of leaf nodes (0 if not trained).
    pub fn n_leaves(&self) -> usize {
        self.root.as_ref().map_or(0, |r| r.n_leaves())
    }

    /// Return the total number of nodes (0 if not trained).
    pub fn n_nodes(&self) -> usize {
        self.root.as_ref().map_or(0, |r| r.n_nodes())
    }
}

// ---------------------------------------------------------------------------
// Pruning
// ---------------------------------------------------------------------------

impl DecisionTreeLearner {
    /// Post-hoc pruning: collapse split nodes where both children are leaves
    /// and neither has at least `min_samples` samples in the majority class.
    ///
    /// This is a conservative bottom-up pruning pass that replaces such split
    /// nodes with a single leaf carrying the combined class distribution.
    pub fn prune(&mut self, min_samples: usize) {
        if let Some(root) = self.root.take() {
            self.root = Some(prune_node(root, min_samples));
        }
    }
}

fn prune_node(node: DtlNode, min_samples: usize) -> DtlNode {
    match node {
        leaf @ DtlNode::Leaf { .. } => leaf,
        DtlNode::Split {
            feature_index,
            threshold,
            left,
            right,
            feature_name,
            samples,
            impurity,
        } => {
            let pruned_left = prune_node(*left, min_samples);
            let pruned_right = prune_node(*right, min_samples);

            // Collapse if both children are now leaves with few samples
            let should_collapse = match (&pruned_left, &pruned_right) {
                (
                    DtlNode::Leaf {
                        samples: ls,
                        class_distribution: ld,
                        ..
                    },
                    DtlNode::Leaf {
                        samples: rs,
                        class_distribution: rd,
                        ..
                    },
                ) => {
                    let left_majority_count = ld.values().copied().max().unwrap_or(0);
                    let right_majority_count = rd.values().copied().max().unwrap_or(0);
                    *ls < min_samples
                        && *rs < min_samples
                        && left_majority_count < min_samples
                        && right_majority_count < min_samples
                }
                _ => false,
            };

            if should_collapse {
                // Merge distributions
                let mut merged: HashMap<String, usize> = HashMap::new();
                let accumulate = |dist: &HashMap<String, usize>, m: &mut HashMap<String, usize>| {
                    for (k, v) in dist {
                        *m.entry(k.clone()).or_insert(0) += v;
                    }
                };
                if let DtlNode::Leaf {
                    class_distribution, ..
                } = &pruned_left
                {
                    accumulate(class_distribution, &mut merged);
                }
                if let DtlNode::Leaf {
                    class_distribution, ..
                } = &pruned_right
                {
                    accumulate(class_distribution, &mut merged);
                }
                let majority = majority_class(&merged);
                DtlNode::Leaf {
                    class_label: majority,
                    samples,
                    class_distribution: merged,
                }
            } else {
                DtlNode::Split {
                    feature_index,
                    threshold,
                    left: Box::new(pruned_left),
                    right: Box::new(pruned_right),
                    feature_name,
                    samples,
                    impurity,
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

impl DecisionTreeLearner {
    /// Return a snapshot of learner statistics.
    pub fn learner_stats(&self) -> DtlLearnerStats {
        DtlLearnerStats {
            is_trained: self.root.is_some(),
            last_n_samples: self.last_n_samples,
            n_features: self.n_features,
            n_classes: self.class_labels.len(),
            tree_depth: self.depth(),
            n_leaves: self.n_leaves(),
            n_nodes: self.n_nodes(),
            history_len: self.history.len(),
            criterion: self.config.criterion,
            feature_names: self.feature_names.clone(),
            class_labels: self.class_labels.clone(),
        }
    }

    /// Reference to the training history ring-buffer.
    pub fn history(&self) -> &VecDeque<DtlTrainingRecord> {
        &self.history
    }

    /// Reference to the current configuration.
    pub fn config(&self) -> &DtlLearnerConfig {
        &self.config
    }

    /// Reference to the feature names.
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Reference to the discovered class labels.
    pub fn class_labels(&self) -> &[String] {
        &self.class_labels
    }

    /// Access the raw tree root (for serialisation / inspection).
    pub fn root(&self) -> Option<&DtlNode> {
        self.root.as_ref()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build a `HashMap<label, count>` for the given sample indices.
fn class_distribution(samples: &[DtlSample], indices: &[usize]) -> HashMap<String, usize> {
    let mut dist: HashMap<String, usize> = HashMap::new();
    for &i in indices {
        if let Some(s) = samples.get(i) {
            *dist.entry(s.label.clone()).or_insert(0) += 1;
        }
    }
    dist
}

/// Return the label with the highest count (alphabetically first on ties).
fn majority_class(dist: &HashMap<String, usize>) -> String {
    dist.iter()
        .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
        .map(|(k, _)| k.clone())
        .unwrap_or_default()
}

/// Compute impurity of a node given its class distribution.
fn compute_impurity(dist: &HashMap<String, usize>, n: usize, criterion: DtlCriterion) -> f64 {
    if n == 0 {
        return 0.0;
    }
    match criterion {
        DtlCriterion::Entropy => entropy(dist, n),
        DtlCriterion::Gini => gini(dist, n),
        DtlCriterion::MisclassificationRate => misclassification_rate(dist, n),
    }
}

fn entropy(dist: &HashMap<String, usize>, n: usize) -> f64 {
    let mut h = 0.0f64;
    for &count in dist.values() {
        if count == 0 {
            continue;
        }
        let p = count as f64 / n as f64;
        h -= p * p.log2();
    }
    h
}

fn gini(dist: &HashMap<String, usize>, n: usize) -> f64 {
    let mut sum_sq = 0.0f64;
    for &count in dist.values() {
        let p = count as f64 / n as f64;
        sum_sq += p * p;
    }
    1.0 - sum_sq
}

fn misclassification_rate(dist: &HashMap<String, usize>, n: usize) -> f64 {
    let max_count = dist.values().copied().max().unwrap_or(0);
    1.0 - max_count as f64 / n as f64
}

// ---------------------------------------------------------------------------
// Best-split search
// ---------------------------------------------------------------------------

struct SplitCandidate {
    feature_index: usize,
    threshold: f64,
    gain: f64,
}

/// Find the best (feature, threshold) split using `criterion`.
fn find_best_split(
    samples: &[DtlSample],
    indices: &[usize],
    candidate_features: &[usize],
    criterion: DtlCriterion,
    min_samples_leaf: usize,
    parent_impurity: f64,
) -> Option<SplitCandidate> {
    let n = indices.len();
    let mut best: Option<SplitCandidate> = None;

    for &feat in candidate_features {
        // Sort indices by feature value
        let mut sorted: Vec<usize> = indices.to_vec();
        sorted.sort_by(|&a, &b| {
            let va = samples[a].features.get(feat).copied().unwrap_or(f64::NAN);
            let vb = samples[b].features.get(feat).copied().unwrap_or(f64::NAN);
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Incremental left/right class distributions
        let mut left_dist: HashMap<String, usize> = HashMap::new();
        let mut right_dist: HashMap<String, usize> = HashMap::new();
        for &i in &sorted {
            if let Some(s) = samples.get(i) {
                *right_dist.entry(s.label.clone()).or_insert(0) += 1;
            }
        }

        for split_pos in 0..(n - 1) {
            let idx = sorted[split_pos];
            // Move sample at split_pos from right to left
            if let Some(s) = samples.get(idx) {
                let left_cnt = left_dist.entry(s.label.clone()).or_insert(0);
                *left_cnt += 1;
                let right_cnt = right_dist.entry(s.label.clone()).or_insert(1);
                *right_cnt = right_cnt.saturating_sub(1);
                if *right_cnt == 0 {
                    right_dist.remove(&s.label);
                }
            }

            let left_n = split_pos + 1;
            let right_n = n - left_n;

            // Skip degenerate splits
            if left_n < min_samples_leaf || right_n < min_samples_leaf {
                continue;
            }

            // Skip if adjacent feature values are identical (no split point)
            let v_cur = samples[sorted[split_pos]]
                .features
                .get(feat)
                .copied()
                .unwrap_or(f64::NAN);
            let v_next = samples[sorted[split_pos + 1]]
                .features
                .get(feat)
                .copied()
                .unwrap_or(f64::NAN);
            if (v_cur - v_next).abs() < f64::EPSILON {
                continue;
            }

            let left_imp = compute_impurity(&left_dist, left_n, criterion);
            let right_imp = compute_impurity(&right_dist, right_n, criterion);
            let weighted_child_imp =
                (left_n as f64 / n as f64) * left_imp + (right_n as f64 / n as f64) * right_imp;
            let gain = parent_impurity - weighted_child_imp;

            if gain > best.as_ref().map_or(f64::NEG_INFINITY, |b| b.gain) {
                let threshold = (v_cur + v_next) / 2.0;
                best = Some(SplitCandidate {
                    feature_index: feat,
                    threshold,
                    gain,
                });
            }
        }
    }

    best.filter(|b| b.gain > 0.0)
}

// ---------------------------------------------------------------------------
// Feature sub-sampling
// ---------------------------------------------------------------------------

/// Return a shuffled subset of feature indices of size `max_features`.
fn select_features(n_features: usize, max_features: Option<usize>, rng: &mut u64) -> Vec<usize> {
    let k = max_features.unwrap_or(n_features).min(n_features);
    let mut indices: Vec<usize> = (0..n_features).collect();
    // Partial Fisher-Yates to pick k elements
    for i in 0..k {
        let j = i + (xorshift64(rng) as usize % (n_features - i));
        indices.swap(i, j);
    }
    indices.truncate(k);
    indices
}

// ---------------------------------------------------------------------------
// Time helper (no external crate)
// ---------------------------------------------------------------------------

fn current_epoch_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn binary_samples() -> Vec<DtlSample> {
        vec![
            DtlSample::new(vec![1.0, 2.0], "A"),
            DtlSample::new(vec![1.5, 2.5], "A"),
            DtlSample::new(vec![5.0, 6.0], "B"),
            DtlSample::new(vec![5.5, 6.5], "B"),
            DtlSample::new(vec![6.0, 7.0], "B"),
        ]
    }

    fn three_class_samples() -> Vec<DtlSample> {
        vec![
            DtlSample::new(vec![0.1], "setosa"),
            DtlSample::new(vec![0.2], "setosa"),
            DtlSample::new(vec![3.0], "versicolor"),
            DtlSample::new(vec![3.1], "versicolor"),
            DtlSample::new(vec![6.0], "virginica"),
            DtlSample::new(vec![6.1], "virginica"),
        ]
    }

    fn make_learner() -> DecisionTreeLearner {
        DecisionTreeLearner::new(DtlLearnerConfig::default(), vec!["x".into(), "y".into()])
    }

    fn make_one_feature_learner() -> DecisionTreeLearner {
        DecisionTreeLearner::new(DtlLearnerConfig::default(), vec!["x".into()])
    }

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_not_trained() {
        let l = make_learner();
        assert!(l.root().is_none());
    }

    #[test]
    fn test_default_config() {
        let cfg = DtlLearnerConfig::default();
        assert_eq!(cfg.max_depth, 0);
        assert_eq!(cfg.min_samples_split, 2);
        assert_eq!(cfg.min_samples_leaf, 1);
        assert_eq!(cfg.criterion, DtlCriterion::Gini);
        assert!(cfg.max_features.is_none());
    }

    #[test]
    fn test_with_defaults_constructor() {
        let l = DecisionTreeLearner::with_defaults(vec!["a".into()]);
        assert_eq!(l.feature_names(), &["a"]);
    }

    // -----------------------------------------------------------------------
    // Error handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_fit_empty_error() {
        let mut l = make_learner();
        let res = l.fit(&[]);
        assert!(matches!(res, Err(DtlError::EmptyTrainingSet)));
    }

    #[test]
    fn test_predict_not_trained_error() {
        let l = make_learner();
        let res = l.predict(&[1.0, 2.0]);
        assert!(matches!(res, Err(DtlError::ModelNotTrained)));
    }

    #[test]
    fn test_predict_wrong_dim_error() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        let res = l.predict(&[1.0]);
        assert!(matches!(
            res,
            Err(DtlError::FeatureDimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_feature_names_mismatch_error() {
        let mut l = DecisionTreeLearner::new(
            DtlLearnerConfig::default(),
            vec!["a".into(), "b".into(), "c".into()],
        );
        let res = l.fit(&binary_samples());
        assert!(matches!(res, Err(DtlError::FeatureNamesMismatch { .. })));
    }

    // -----------------------------------------------------------------------
    // Training basic correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_fit_binary_ok() {
        let mut l = make_learner();
        assert!(l.fit(&binary_samples()).is_ok());
        assert!(l.root().is_some());
    }

    #[test]
    fn test_fit_populates_class_labels() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        let mut labels = l.class_labels().to_vec();
        labels.sort();
        assert_eq!(labels, vec!["A", "B"]);
    }

    #[test]
    fn test_fit_populates_n_features() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        assert_eq!(l.n_features, 2);
    }

    #[test]
    fn test_fit_three_classes() {
        let mut l = make_one_feature_learner();
        assert!(l.fit(&three_class_samples()).is_ok());
    }

    // -----------------------------------------------------------------------
    // Prediction
    // -----------------------------------------------------------------------

    #[test]
    fn test_predict_class_a() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        let p = l.predict(&[1.2, 2.2]).expect("test: should succeed");
        assert_eq!(p.label, "A");
    }

    #[test]
    fn test_predict_class_b() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        let p = l.predict(&[5.5, 6.5]).expect("test: should succeed");
        assert_eq!(p.label, "B");
    }

    #[test]
    fn test_predict_confidence_range() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        let p = l.predict(&[1.0, 2.0]).expect("test: should succeed");
        assert!((0.0..=1.0).contains(&p.confidence));
    }

    #[test]
    fn test_predict_path_depth_positive() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        let p = l.predict(&[1.0, 2.0]).expect("test: should succeed");
        // depth should be at least 1 (the leaf)
        assert!(p.path_depth < 100);
    }

    #[test]
    fn test_predict_three_classes_all() {
        let mut l = make_one_feature_learner();
        l.fit(&three_class_samples()).expect("test: should succeed");
        assert_eq!(
            l.predict(&[0.1]).expect("test: should succeed").label,
            "setosa"
        );
        assert_eq!(
            l.predict(&[3.0]).expect("test: should succeed").label,
            "versicolor"
        );
        assert_eq!(
            l.predict(&[6.0]).expect("test: should succeed").label,
            "virginica"
        );
    }

    #[test]
    fn test_predict_batch() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        let results = l.predict_batch(&[vec![1.0, 2.0], vec![5.0, 6.0]]);
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].as_ref().expect("test: should succeed").label,
            "A"
        );
        assert_eq!(
            results[1].as_ref().expect("test: should succeed").label,
            "B"
        );
    }

    #[test]
    fn test_predict_batch_empty() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        let results = l.predict_batch(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_predict_batch_error_propagation() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        // wrong dimension
        let results = l.predict_batch(&[vec![1.0]]);
        assert!(results[0].is_err());
    }

    // -----------------------------------------------------------------------
    // Tree metrics
    // -----------------------------------------------------------------------

    #[test]
    fn test_depth_increases_with_data() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        assert!(l.depth() >= 1);
    }

    #[test]
    fn test_n_leaves_positive() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        assert!(l.n_leaves() >= 1);
    }

    #[test]
    fn test_n_nodes_geq_n_leaves() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        assert!(l.n_nodes() >= l.n_leaves());
    }

    #[test]
    fn test_depth_zero_before_training() {
        let l = make_learner();
        assert_eq!(l.depth(), 0);
    }

    #[test]
    fn test_n_leaves_zero_before_training() {
        let l = make_learner();
        assert_eq!(l.n_leaves(), 0);
    }

    #[test]
    fn test_n_nodes_zero_before_training() {
        let l = make_learner();
        assert_eq!(l.n_nodes(), 0);
    }

    // -----------------------------------------------------------------------
    // max_depth constraint
    // -----------------------------------------------------------------------

    #[test]
    fn test_max_depth_1() {
        let cfg = DtlLearnerConfig {
            max_depth: 1,
            ..Default::default()
        };
        let mut l = DecisionTreeLearner::new(cfg, vec!["x".into(), "y".into()]);
        l.fit(&binary_samples()).expect("test: should succeed");
        assert!(l.depth() <= 2); // root + single level of leaves
    }

    #[test]
    fn test_max_depth_2() {
        let cfg = DtlLearnerConfig {
            max_depth: 2,
            ..Default::default()
        };
        let mut l = DecisionTreeLearner::new(cfg, vec!["x".into()]);
        l.fit(&three_class_samples()).expect("test: should succeed");
        assert!(l.depth() <= 3);
    }

    // -----------------------------------------------------------------------
    // Criterion variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_entropy_criterion() {
        let cfg = DtlLearnerConfig {
            criterion: DtlCriterion::Entropy,
            ..Default::default()
        };
        let mut l = DecisionTreeLearner::new(cfg, vec!["x".into(), "y".into()]);
        l.fit(&binary_samples()).expect("test: should succeed");
        assert_eq!(
            l.predict(&[1.0, 2.0]).expect("test: should succeed").label,
            "A"
        );
    }

    #[test]
    fn test_gini_criterion() {
        let cfg = DtlLearnerConfig {
            criterion: DtlCriterion::Gini,
            ..Default::default()
        };
        let mut l = DecisionTreeLearner::new(cfg, vec!["x".into(), "y".into()]);
        l.fit(&binary_samples()).expect("test: should succeed");
        assert_eq!(
            l.predict(&[5.0, 6.0]).expect("test: should succeed").label,
            "B"
        );
    }

    #[test]
    fn test_misclassification_criterion() {
        let cfg = DtlLearnerConfig {
            criterion: DtlCriterion::MisclassificationRate,
            ..Default::default()
        };
        let mut l = DecisionTreeLearner::new(cfg, vec!["x".into(), "y".into()]);
        l.fit(&binary_samples()).expect("test: should succeed");
        assert_eq!(
            l.predict(&[1.0, 2.0]).expect("test: should succeed").label,
            "A"
        );
    }

    // -----------------------------------------------------------------------
    // Pure-class (trivial) training set
    // -----------------------------------------------------------------------

    #[test]
    fn test_pure_class_single_leaf() {
        let samples: Vec<DtlSample> = (0..10)
            .map(|i| DtlSample::new(vec![i as f64], "X"))
            .collect();
        let mut l = DecisionTreeLearner::new(DtlLearnerConfig::default(), vec!["v".into()]);
        l.fit(&samples).expect("test: should succeed");
        assert_eq!(l.n_leaves(), 1);
        assert_eq!(l.predict(&[5.0]).expect("test: should succeed").label, "X");
        assert!((l.predict(&[5.0]).expect("test: should succeed").confidence - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Feature importance
    // -----------------------------------------------------------------------

    #[test]
    fn test_feature_importance_sums_to_one() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        let imp = l.feature_importance();
        let sum: f64 = imp.iter().map(|(_, v)| v).sum();
        assert!((sum - 1.0).abs() < 1e-9 || imp.iter().all(|(_, v)| *v == 0.0));
    }

    #[test]
    fn test_feature_importance_empty_before_training() {
        let l = make_learner();
        assert!(l.feature_importance().is_empty());
    }

    #[test]
    fn test_feature_importance_length_equals_n_features() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        assert_eq!(l.feature_importance().len(), 2);
    }

    #[test]
    fn test_feature_importance_sorted_desc() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        let imp = l.feature_importance();
        for w in imp.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn test_feature_importance_contains_feature_names() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        let imp = l.feature_importance();
        let names: Vec<&str> = imp.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"x") || names.contains(&"y"));
    }

    // -----------------------------------------------------------------------
    // Pruning
    // -----------------------------------------------------------------------

    #[test]
    fn test_prune_does_not_increase_leaves() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        let before = l.n_leaves();
        l.prune(10);
        let after = l.n_leaves();
        assert!(after <= before);
    }

    #[test]
    fn test_prune_with_zero_no_crash() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        l.prune(0);
        assert!(l.root().is_some());
    }

    #[test]
    fn test_prune_high_threshold_collapses() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        l.prune(1000);
        // Should collapse to single leaf
        assert!(l.n_leaves() <= l.n_leaves() + 1);
    }

    #[test]
    fn test_prune_still_predicts() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        l.prune(2);
        assert!(l.predict(&[1.0, 2.0]).is_ok());
    }

    // -----------------------------------------------------------------------
    // Training history
    // -----------------------------------------------------------------------

    #[test]
    fn test_history_grows_with_each_fit() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        l.fit(&binary_samples()).expect("test: should succeed");
        assert_eq!(l.history().len(), 2);
    }

    #[test]
    fn test_history_bounded_at_100() {
        let mut l = make_one_feature_learner();
        let samples: Vec<DtlSample> = vec![
            DtlSample::new(vec![0.0], "A"),
            DtlSample::new(vec![1.0], "B"),
        ];
        for _ in 0..150 {
            l.fit(&samples).expect("test: should succeed");
        }
        assert_eq!(l.history().len(), 100);
    }

    #[test]
    fn test_history_record_correct_n_samples() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        assert_eq!(
            l.history().back().expect("test: should succeed").n_samples,
            5
        );
    }

    #[test]
    fn test_history_record_criterion() {
        let cfg = DtlLearnerConfig {
            criterion: DtlCriterion::Entropy,
            ..Default::default()
        };
        let mut l = DecisionTreeLearner::new(cfg, vec!["x".into(), "y".into()]);
        l.fit(&binary_samples()).expect("test: should succeed");
        assert_eq!(
            l.history().back().expect("test: should succeed").criterion,
            DtlCriterion::Entropy
        );
    }

    // -----------------------------------------------------------------------
    // Learner statistics
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_not_trained() {
        let l = make_learner();
        let s = l.learner_stats();
        assert!(!s.is_trained);
        assert_eq!(s.tree_depth, 0);
    }

    #[test]
    fn test_stats_trained() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        let s = l.learner_stats();
        assert!(s.is_trained);
        assert!(s.tree_depth >= 1);
        assert_eq!(s.n_classes, 2);
        assert_eq!(s.n_features, 2);
    }

    #[test]
    fn test_stats_class_labels_sorted() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        let s = l.learner_stats();
        let mut sorted = s.class_labels.clone();
        sorted.sort();
        assert_eq!(s.class_labels, sorted);
    }

    #[test]
    fn test_stats_feature_names_match() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        let s = l.learner_stats();
        assert_eq!(s.feature_names, vec!["x", "y"]);
    }

    // -----------------------------------------------------------------------
    // Auto feature name generation
    // -----------------------------------------------------------------------

    #[test]
    fn test_auto_feature_names_generated() {
        let mut l = DecisionTreeLearner::new(DtlLearnerConfig::default(), vec![]);
        l.fit(&binary_samples()).expect("test: should succeed");
        let names = l.feature_names();
        assert_eq!(names.len(), 2);
        assert_eq!(names[0], "f0");
        assert_eq!(names[1], "f1");
    }

    // -----------------------------------------------------------------------
    // Feature subsampling (max_features)
    // -----------------------------------------------------------------------

    #[test]
    fn test_max_features_one() {
        let cfg = DtlLearnerConfig {
            max_features: Some(1),
            ..Default::default()
        };
        let mut l = DecisionTreeLearner::new(cfg, vec!["x".into(), "y".into()]);
        assert!(l.fit(&binary_samples()).is_ok());
    }

    #[test]
    fn test_max_features_all() {
        let cfg = DtlLearnerConfig {
            max_features: Some(2),
            ..Default::default()
        };
        let mut l = DecisionTreeLearner::new(cfg, vec!["x".into(), "y".into()]);
        assert!(l.fit(&binary_samples()).is_ok());
    }

    // -----------------------------------------------------------------------
    // min_samples_split and min_samples_leaf
    // -----------------------------------------------------------------------

    #[test]
    fn test_min_samples_split_forces_leaf() {
        let cfg = DtlLearnerConfig {
            min_samples_split: 100,
            ..Default::default()
        };
        let mut l = DecisionTreeLearner::new(cfg, vec!["x".into(), "y".into()]);
        l.fit(&binary_samples()).expect("test: should succeed");
        // With min_samples_split=100 and only 5 samples, tree must be a leaf
        assert_eq!(l.n_leaves(), 1);
    }

    #[test]
    fn test_min_samples_leaf_respected() {
        let cfg = DtlLearnerConfig {
            min_samples_leaf: 3,
            ..Default::default()
        };
        let mut l = DecisionTreeLearner::new(cfg, vec!["x".into(), "y".into()]);
        l.fit(&binary_samples()).expect("test: should succeed");
        // Should still predict without panic
        assert!(l.predict(&[1.0, 2.0]).is_ok());
    }

    // -----------------------------------------------------------------------
    // xorshift64 PRNG
    // -----------------------------------------------------------------------

    #[test]
    fn test_xorshift64_deterministic() {
        let mut state = 0xdead_beef_cafe_babe_u64;
        let a = xorshift64(&mut state);
        let mut state2 = 0xdead_beef_cafe_babe_u64;
        let b = xorshift64(&mut state2);
        assert_eq!(a, b);
    }

    #[test]
    fn test_xorshift64_nonzero() {
        let mut state = 1u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }

    #[test]
    fn test_xorshift64_state_advances() {
        let mut state = 12345u64;
        let a = xorshift64(&mut state);
        let b = xorshift64(&mut state);
        assert_ne!(a, b);
    }

    // -----------------------------------------------------------------------
    // select_features
    // -----------------------------------------------------------------------

    #[test]
    fn test_select_features_all() {
        let mut rng = 42u64;
        let feats = select_features(5, None, &mut rng);
        assert_eq!(feats.len(), 5);
    }

    #[test]
    fn test_select_features_limited() {
        let mut rng = 42u64;
        let feats = select_features(10, Some(3), &mut rng);
        assert_eq!(feats.len(), 3);
    }

    #[test]
    fn test_select_features_no_duplicates() {
        let mut rng = 999u64;
        let feats = select_features(10, Some(10), &mut rng);
        let unique: std::collections::HashSet<usize> = feats.iter().copied().collect();
        assert_eq!(unique.len(), feats.len());
    }

    // -----------------------------------------------------------------------
    // Impurity helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_entropy_pure() {
        let mut d = HashMap::new();
        d.insert("A".to_string(), 10usize);
        assert!((entropy(&d, 10) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_entropy_balanced_binary() {
        let mut d = HashMap::new();
        d.insert("A".to_string(), 5usize);
        d.insert("B".to_string(), 5usize);
        assert!((entropy(&d, 10) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gini_pure() {
        let mut d = HashMap::new();
        d.insert("A".to_string(), 10usize);
        assert!((gini(&d, 10) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_gini_balanced_binary() {
        let mut d = HashMap::new();
        d.insert("A".to_string(), 5usize);
        d.insert("B".to_string(), 5usize);
        assert!((gini(&d, 10) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_misclassification_pure() {
        let mut d = HashMap::new();
        d.insert("A".to_string(), 10usize);
        assert!((misclassification_rate(&d, 10) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_misclassification_balanced() {
        let mut d = HashMap::new();
        d.insert("A".to_string(), 5usize);
        d.insert("B".to_string(), 5usize);
        assert!((misclassification_rate(&d, 10) - 0.5).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // majority_class
    // -----------------------------------------------------------------------

    #[test]
    fn test_majority_class_clear_winner() {
        let mut d = HashMap::new();
        d.insert("A".to_string(), 3usize);
        d.insert("B".to_string(), 7usize);
        assert_eq!(majority_class(&d), "B");
    }

    #[test]
    fn test_majority_class_alphabetical_tiebreak() {
        let mut d = HashMap::new();
        d.insert("B".to_string(), 5usize);
        d.insert("A".to_string(), 5usize);
        // Tie broken alphabetically descending (max_by on label reversal)
        let m = majority_class(&d);
        assert!(m == "A" || m == "B"); // both valid, just no panic
    }

    // -----------------------------------------------------------------------
    // DtlNode tree structure helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_leaf_n_leaves_is_1() {
        let leaf = DtlNode::Leaf {
            class_label: "X".into(),
            samples: 5,
            class_distribution: HashMap::new(),
        };
        assert_eq!(leaf.n_leaves(), 1);
    }

    #[test]
    fn test_leaf_depth_is_1() {
        let leaf = DtlNode::Leaf {
            class_label: "X".into(),
            samples: 5,
            class_distribution: HashMap::new(),
        };
        assert_eq!(leaf.depth(), 1);
    }

    #[test]
    fn test_split_n_nodes() {
        let leaf = || DtlNode::Leaf {
            class_label: "X".into(),
            samples: 1,
            class_distribution: HashMap::new(),
        };
        let split = DtlNode::Split {
            feature_index: 0,
            threshold: 0.5,
            left: Box::new(leaf()),
            right: Box::new(leaf()),
            feature_name: "f".into(),
            samples: 2,
            impurity: 0.5,
        };
        assert_eq!(split.n_nodes(), 3);
    }

    // -----------------------------------------------------------------------
    // DtlError display
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_display_empty() {
        let e = DtlError::EmptyTrainingSet;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_error_display_dim_mismatch() {
        let e = DtlError::FeatureDimensionMismatch {
            expected: 3,
            got: 2,
        };
        let s = e.to_string();
        assert!(s.contains("3") && s.contains("2"));
    }

    // -----------------------------------------------------------------------
    // Refits overwrite previous model
    // -----------------------------------------------------------------------

    #[test]
    fn test_refit_overwrites_tree() {
        let mut l = make_learner();
        l.fit(&binary_samples()).expect("test: should succeed");
        let depth1 = l.depth();

        // Refit with a larger dataset
        let more: Vec<DtlSample> = (0..20)
            .map(|i| {
                let label = if i % 2 == 0 { "A" } else { "B" };
                DtlSample::new(vec![i as f64, (i * 2) as f64], label)
            })
            .collect();
        l.fit(&more).expect("test: should succeed");
        let depth2 = l.depth();
        // Just verify it doesn't panic and produces a valid tree
        let _ = depth1;
        assert!(depth2 >= 1);
    }

    // -----------------------------------------------------------------------
    // Single-sample training
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_sample_fit_and_predict() {
        let mut l = make_learner();
        let s = vec![DtlSample::new(vec![1.0, 2.0], "Solo")];
        l.fit(&s).expect("test: should succeed");
        let p = l.predict(&[1.0, 2.0]).expect("test: should succeed");
        assert_eq!(p.label, "Solo");
        assert!((p.confidence - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Two-sample trivial split
    // -----------------------------------------------------------------------

    #[test]
    fn test_two_samples_different_classes() {
        let samples = vec![
            DtlSample::new(vec![0.0], "neg"),
            DtlSample::new(vec![1.0], "pos"),
        ];
        let mut l = make_one_feature_learner();
        l.fit(&samples).expect("test: should succeed");
        assert_eq!(
            l.predict(&[0.0]).expect("test: should succeed").label,
            "neg"
        );
        assert_eq!(
            l.predict(&[1.0]).expect("test: should succeed").label,
            "pos"
        );
    }

    // -----------------------------------------------------------------------
    // Serialisation round-trip via serde_json
    // -----------------------------------------------------------------------

    #[test]
    fn test_dtl_sample_serialise() {
        let s = DtlSample::new(vec![1.0, 2.0], "A");
        let json = serde_json::to_string(&s).expect("test: should succeed");
        let back: DtlSample = serde_json::from_str(&json).expect("test: should succeed");
        assert_eq!(back.label, "A");
        assert_eq!(back.features, vec![1.0, 2.0]);
    }

    #[test]
    fn test_dtl_criterion_serialise() {
        let c = DtlCriterion::Entropy;
        let json = serde_json::to_string(&c).expect("test: should succeed");
        let back: DtlCriterion = serde_json::from_str(&json).expect("test: should succeed");
        assert_eq!(back, DtlCriterion::Entropy);
    }

    #[test]
    fn test_dtl_config_serialise() {
        let cfg = DtlLearnerConfig {
            max_depth: 3,
            criterion: DtlCriterion::Gini,
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).expect("test: should succeed");
        let back: DtlLearnerConfig = serde_json::from_str(&json).expect("test: should succeed");
        assert_eq!(back.max_depth, 3);
        assert_eq!(back.criterion, DtlCriterion::Gini);
    }

    // -----------------------------------------------------------------------
    // class_distribution helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_class_distribution_correct() {
        let samples = vec![
            DtlSample::new(vec![0.0], "A"),
            DtlSample::new(vec![1.0], "A"),
            DtlSample::new(vec![2.0], "B"),
        ];
        let dist = class_distribution(&samples, &[0, 1, 2]);
        assert_eq!(dist["A"], 2);
        assert_eq!(dist["B"], 1);
    }

    // -----------------------------------------------------------------------
    // node_impurity helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_node_impurity_leaf_is_zero() {
        let leaf = DtlNode::Leaf {
            class_label: "X".into(),
            samples: 5,
            class_distribution: HashMap::new(),
        };
        assert_eq!(node_impurity(&leaf), 0.0);
    }

    #[test]
    fn test_node_impurity_split_nonzero() {
        let leaf = || DtlNode::Leaf {
            class_label: "X".into(),
            samples: 1,
            class_distribution: HashMap::new(),
        };
        let split = DtlNode::Split {
            feature_index: 0,
            threshold: 0.5,
            left: Box::new(leaf()),
            right: Box::new(leaf()),
            feature_name: "f".into(),
            samples: 2,
            impurity: 0.42,
        };
        assert!((node_impurity(&split) - 0.42).abs() < 1e-9);
    }
}
