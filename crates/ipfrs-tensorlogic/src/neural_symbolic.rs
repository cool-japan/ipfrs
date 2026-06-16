//! Neural-Symbolic Integrator — bridges continuous embedding representations
//! with symbolic logical rule reasoning for hybrid inference.
//!
//! This module implements [`NeuralSymbolicIntegrator`], a production-grade
//! system that combines learned neural patterns with explicit logical rules,
//! enabling inference that benefits from both paradigms.
//!
//! ## Architecture
//!
//! The integrator maintains:
//! - A registry of [`Symbol`]s, each with a name, high-dimensional embedding,
//!   and a base confidence score.
//! - A set of [`LogicalRule`]s that define horn-clause-style implication
//!   relationships between symbols, weighted by confidence and typed by
//!   [`RuleType`].
//!
//! Inference proceeds in three possible modes (controlled by [`InferenceMode`]):
//! - **Pure Symbolic** — forward-chain through rules only.
//! - **Pure Neural** — compute embedding similarity to evidence only.
//! - **Hybrid** — weighted combination of symbolic and neural signals.

use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by [`NeuralSymbolicIntegrator`].
#[derive(Debug, Error, Clone, PartialEq)]
pub enum NsError {
    /// A symbol id referenced in a query or rule does not exist.
    #[error("symbol not found: id {0}")]
    SymbolNotFound(usize),

    /// The provided embedding has a different dimension than the integrator was
    /// configured with.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// The symbol registry is full.
    #[error("maximum symbol count reached")]
    MaxSymbolsReached,

    /// A rule references invalid symbol ids or is otherwise structurally
    /// incorrect.
    #[error("invalid rule: {0}")]
    InvalidRule(String),
}

// ---------------------------------------------------------------------------
// Core identifiers and data types
// ---------------------------------------------------------------------------

/// A lightweight newtype wrapper for a symbol index in the integrator's
/// internal registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymbolId(pub usize);

impl std::fmt::Display for SymbolId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sym:{}", self.0)
    }
}

/// A named concept with a neural embedding and a base confidence score.
///
/// Symbols are the atomic units of knowledge in the integrator.  Each symbol
/// can participate in logical rules as a head or body atom, and its embedding
/// allows neural similarity to be computed against evidence embeddings.
#[derive(Debug, Clone)]
pub struct Symbol {
    /// Stable identifier within this integrator instance.
    pub id: SymbolId,
    /// Human-readable label.
    pub name: String,
    /// Dense embedding vector; length must equal
    /// [`IntegratorConfig::embedding_dim`].
    pub embedding: Vec<f64>,
    /// Prior confidence that this symbol holds in absence of other evidence;
    /// in `[0.0, 1.0]`.
    pub confidence: f64,
}

/// The flavour of a logical rule, controlling how body satisfaction is
/// translated into rule confidence.
#[derive(Debug, Clone, PartialEq)]
pub enum RuleType {
    /// Classical definite clause — body satisfaction is multiplied by the
    /// rule weight unchanged.
    Definite,
    /// Probabilistic rule — weight is further modulated by the body
    /// satisfaction probability.
    Probabilistic,
    /// Soft rule with temperature-controlled sigmoid.  Lower `temperature`
    /// makes the sigmoid steeper (closer to a step function).
    Soft {
        /// Controls sharpness of the sigmoid applied to body satisfaction.
        temperature: f64,
    },
}

/// A single implication rule: `weight * f(body) → head`.
#[derive(Debug, Clone)]
pub struct LogicalRule {
    /// The symbol that this rule can derive confidence for.
    pub head: SymbolId,
    /// Ordered list of body atoms that must be satisfied.
    pub body: Vec<SymbolId>,
    /// Base weight of the rule; in `(0.0, 1.0]` by convention.
    pub weight: f64,
    /// Determines the transfer function from body satisfaction to rule
    /// confidence.
    pub rule_type: RuleType,
}

// ---------------------------------------------------------------------------
// Query / result types
// ---------------------------------------------------------------------------

/// Selects which inference strategy the integrator uses when answering a
/// [`NsQuery`].
#[derive(Debug, Clone, PartialEq)]
pub enum InferenceMode {
    /// Only symbolic forward-chaining through rules.
    PureSymbolic,
    /// Only neural embedding similarity.
    PureNeural,
    /// Weighted blend: `neural_weight * neural + (1 - neural_weight) *
    /// symbolic`.
    Hybrid {
        /// Weight given to the neural signal; must be in `[0.0, 1.0]`.
        neural_weight: f64,
    },
}

/// A query to the integrator asking for the confidence of a target symbol
/// given some observed evidence.
#[derive(Debug, Clone)]
pub struct NsQuery {
    /// The symbol whose confidence we want to estimate.
    pub target: SymbolId,
    /// Observed symbols and their associated confidence values.
    pub evidence: Vec<(SymbolId, f64)>,
    /// Which inference strategy to apply.
    pub mode: InferenceMode,
}

/// The answer returned by [`NeuralSymbolicIntegrator::infer`].
#[derive(Debug, Clone)]
pub struct NsResult {
    /// The queried symbol.
    pub symbol: SymbolId,
    /// Combined confidence estimate produced by the chosen inference mode.
    pub confidence: f64,
    /// Human-readable strings describing the rules or similarities that
    /// contributed to this result.
    pub explanation: Vec<String>,
    /// The raw neural (embedding-similarity) component of the confidence.
    pub neural_contribution: f64,
    /// The raw symbolic (forward-chaining) component of the confidence.
    pub symbolic_contribution: f64,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration parameters for [`NeuralSymbolicIntegrator`].
#[derive(Debug, Clone)]
pub struct IntegratorConfig {
    /// Dimensionality that every symbol embedding must have.
    pub embedding_dim: usize,
    /// Hard cap on the number of symbols that can be registered.
    pub max_symbols: usize,
    /// Maximum recursion depth during symbolic forward-chaining.
    pub inference_depth: usize,
    /// Cosine-similarity threshold below which a symbol is not considered a
    /// *similar* symbol in [`NeuralSymbolicIntegrator::similar_symbols`].
    pub similarity_threshold: f64,
}

impl Default for IntegratorConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            max_symbols: 10_000,
            inference_depth: 5,
            similarity_threshold: 0.7,
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Aggregate statistics for a [`NeuralSymbolicIntegrator`] instance.
#[derive(Debug, Clone, PartialEq)]
pub struct NsStats {
    /// Number of registered symbols.
    pub symbol_count: usize,
    /// Number of registered rules.
    pub rule_count: usize,
    /// Total number of inference calls that have completed successfully.
    pub total_inferences: u64,
    /// Mean L2 norm of all symbol embeddings; useful for sanity-checking that
    /// embeddings are properly normalised.
    pub avg_embedding_norm: f64,
}

// ---------------------------------------------------------------------------
// Main integrator
// ---------------------------------------------------------------------------

/// A production-grade neural-symbolic integration engine.
///
/// Maintains a registry of [`Symbol`]s with continuous embeddings alongside
/// a set of weighted [`LogicalRule`]s.  Inference combines cosine-similarity
/// lookup (neural path) with forward-chaining through rules (symbolic path).
///
/// # Thread safety
///
/// `NeuralSymbolicIntegrator` is **not** `Sync` by default because `infer`
/// takes `&mut self` to update the `total_inferences` counter.  Wrap in a
/// `Mutex` or `RwLock` for shared concurrent access.
pub struct NeuralSymbolicIntegrator {
    /// Configuration that was provided at construction time.
    pub config: IntegratorConfig,
    /// All registered symbols, indexed by their [`SymbolId`].
    pub symbols: Vec<Symbol>,
    /// All registered rules.
    pub rules: Vec<LogicalRule>,
    /// Monotonically increasing counter of successful `infer` calls.
    pub total_inferences: u64,
}

impl NeuralSymbolicIntegrator {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new, empty integrator with the given configuration.
    pub fn new(config: IntegratorConfig) -> Self {
        Self {
            config,
            symbols: Vec::new(),
            rules: Vec::new(),
            total_inferences: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Registry management
    // -----------------------------------------------------------------------

    /// Register a new symbol.
    ///
    /// # Errors
    ///
    /// Returns [`NsError::MaxSymbolsReached`] when the symbol count would
    /// exceed [`IntegratorConfig::max_symbols`].
    ///
    /// Returns [`NsError::DimensionMismatch`] when `embedding.len()` differs
    /// from [`IntegratorConfig::embedding_dim`].
    pub fn add_symbol(
        &mut self,
        name: String,
        embedding: Vec<f64>,
        confidence: f64,
    ) -> Result<SymbolId, NsError> {
        if self.symbols.len() >= self.config.max_symbols {
            return Err(NsError::MaxSymbolsReached);
        }
        if embedding.len() != self.config.embedding_dim {
            return Err(NsError::DimensionMismatch {
                expected: self.config.embedding_dim,
                got: embedding.len(),
            });
        }
        let id = SymbolId(self.symbols.len());
        self.symbols.push(Symbol {
            id,
            name,
            embedding,
            confidence: confidence.clamp(0.0, 1.0),
        });
        Ok(id)
    }

    /// Register a logical rule.
    ///
    /// # Errors
    ///
    /// Returns [`NsError::InvalidRule`] if the head or any body symbol does
    /// not exist in the registry.
    pub fn add_rule(&mut self, rule: LogicalRule) -> Result<(), NsError> {
        if rule.head.0 >= self.symbols.len() {
            return Err(NsError::InvalidRule(format!(
                "head symbol {} does not exist",
                rule.head.0
            )));
        }
        for &body_sym in &rule.body {
            if body_sym.0 >= self.symbols.len() {
                return Err(NsError::InvalidRule(format!(
                    "body symbol {} does not exist",
                    body_sym.0
                )));
            }
        }
        self.rules.push(rule);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Neural helpers
    // -----------------------------------------------------------------------

    /// Compute the cosine similarity between two embedding vectors.
    ///
    /// Returns `0.0` when either vector has zero norm (to avoid division by
    /// zero) and clamps the result to `[-1.0, 1.0]`.
    pub fn neural_similarity(a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len(), "embedding dimension mismatch");
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    // -----------------------------------------------------------------------
    // Symbolic inference
    // -----------------------------------------------------------------------

    /// Forward-chain through rules to estimate confidence in `target`.
    ///
    /// Recursively resolves body atoms up to `depth` levels deep.  Each call
    /// collects evidence from the provided map, multiplies body atoms to obtain
    /// body satisfaction, applies the rule-type transfer function, and returns
    /// the maximum confidence obtained across all matching rules.
    ///
    /// # Arguments
    ///
    /// * `target` — the symbol whose confidence we are computing.
    /// * `evidence` — the set of directly observed (symbol, confidence) pairs.
    /// * `depth` — remaining recursion budget.
    pub fn symbolic_forward_chain(
        &self,
        target: SymbolId,
        evidence: &[(SymbolId, f64)],
        depth: usize,
    ) -> f64 {
        // Build a lookup map from the evidence slice.
        let ev_map: HashMap<usize, f64> = evidence.iter().map(|(s, c)| (s.0, *c)).collect();
        self.symbolic_chain_inner(target, &ev_map, depth)
    }

    /// Internal recursive implementation of symbolic forward chaining.
    fn symbolic_chain_inner(
        &self,
        target: SymbolId,
        ev_map: &HashMap<usize, f64>,
        depth: usize,
    ) -> f64 {
        // Check direct evidence first.
        if let Some(&conf) = ev_map.get(&target.0) {
            return conf;
        }

        if depth == 0 {
            // At the recursion limit, fall back to the symbol's own confidence.
            return self
                .symbols
                .get(target.0)
                .map(|s| s.confidence)
                .unwrap_or(0.0);
        }

        let mut best: f64 = 0.0;

        for rule in &self.rules {
            if rule.head != target {
                continue;
            }
            if rule.body.is_empty() {
                // A fact (empty body) — body satisfaction is 1.0 (vacuously
                // true), so apply the transfer function with full satisfaction.
                let adjusted = self.apply_rule_type(rule, 1.0, rule.weight);
                if adjusted > best {
                    best = adjusted;
                }
                continue;
            }

            // Compute body satisfaction as the product of each body atom's
            // confidence, resolved recursively.
            let body_satisfaction: f64 = rule.body.iter().fold(1.0, |acc, &body_sym| {
                let body_conf = self.symbolic_chain_inner(body_sym, ev_map, depth - 1);
                acc * body_conf
            });

            let rule_confidence = self.apply_rule_type(rule, body_satisfaction, rule.weight);

            if rule_confidence > best {
                best = rule_confidence;
            }
        }

        best
    }

    /// Apply the [`RuleType`] transfer function to obtain the final rule
    /// confidence from the body satisfaction and rule weight.
    fn apply_rule_type(
        &self,
        rule: &LogicalRule,
        body_satisfaction: f64,
        _base_weight: f64,
    ) -> f64 {
        match &rule.rule_type {
            RuleType::Definite => rule.weight * body_satisfaction,
            RuleType::Probabilistic => rule.weight * body_satisfaction * body_satisfaction,
            RuleType::Soft { temperature } => {
                let temp = temperature.max(f64::EPSILON);
                let sigmoid_input = body_satisfaction / temp;
                let sigmoid = 1.0 / (1.0 + (-sigmoid_input).exp());
                rule.weight * sigmoid
            }
        }
    }

    // -----------------------------------------------------------------------
    // Neural inference
    // -----------------------------------------------------------------------

    /// Estimate confidence in `target` purely via embedding similarity.
    ///
    /// For each evidence pair `(sym, conf)`, computes
    /// `neural_similarity(target.embedding, sym.embedding) * conf` and returns
    /// the maximum across all evidence items.  Returns `0.0` when `target`
    /// does not exist or there is no evidence.
    pub fn neural_forward(&self, target: SymbolId, evidence: &[(SymbolId, f64)]) -> f64 {
        let target_sym = match self.symbols.get(target.0) {
            Some(s) => s,
            None => return 0.0,
        };

        let mut best: f64 = 0.0;
        for &(ev_sym_id, conf) in evidence {
            if let Some(ev_sym) = self.symbols.get(ev_sym_id.0) {
                let sim = Self::neural_similarity(&target_sym.embedding, &ev_sym.embedding);
                let score = sim * conf;
                if score > best {
                    best = score;
                }
            }
        }
        best
    }

    // -----------------------------------------------------------------------
    // Main inference entry point
    // -----------------------------------------------------------------------

    /// Answer an [`NsQuery`] using the specified [`InferenceMode`].
    ///
    /// # Errors
    ///
    /// Returns [`NsError::SymbolNotFound`] when `query.target` does not exist.
    pub fn infer(&mut self, query: &NsQuery) -> Result<NsResult, NsError> {
        // Validate target.
        if query.target.0 >= self.symbols.len() {
            return Err(NsError::SymbolNotFound(query.target.0));
        }

        let symbolic =
            self.symbolic_forward_chain(query.target, &query.evidence, self.config.inference_depth);
        let neural = self.neural_forward(query.target, &query.evidence);

        let combined = match &query.mode {
            InferenceMode::PureSymbolic => symbolic,
            InferenceMode::PureNeural => neural,
            InferenceMode::Hybrid { neural_weight } => {
                let nw = neural_weight.clamp(0.0, 1.0);
                nw * neural + (1.0 - nw) * symbolic
            }
        };

        // Collect explanation strings.
        let explanation = self.build_explanation(query.target, &query.evidence);

        self.total_inferences += 1;

        Ok(NsResult {
            symbol: query.target,
            confidence: combined.clamp(0.0, 1.0),
            explanation,
            neural_contribution: neural,
            symbolic_contribution: symbolic,
        })
    }

    /// Build a list of human-readable strings describing which rules
    /// contributed to the inference for `target`.
    fn build_explanation(&self, target: SymbolId, evidence: &[(SymbolId, f64)]) -> Vec<String> {
        let mut explanations: Vec<String> = Vec::new();

        // Identify contributing rules.
        for rule in &self.rules {
            if rule.head != target {
                continue;
            }
            let body_names: Vec<String> = rule
                .body
                .iter()
                .map(|b| {
                    self.symbols
                        .get(b.0)
                        .map(|s| s.name.clone())
                        .unwrap_or_else(|| format!("sym:{}", b.0))
                })
                .collect();
            let head_name = self
                .symbols
                .get(target.0)
                .map(|s| s.name.clone())
                .unwrap_or_else(|| format!("sym:{}", target.0));
            let body_str = if body_names.is_empty() {
                "∅".to_string()
            } else {
                body_names.join(", ")
            };
            explanations.push(format!(
                "rule_{}_from_[{}] (weight={:.3})",
                head_name, body_str, rule.weight
            ));
        }

        // Identify contributing neural evidence.
        if let Some(target_sym) = self.symbols.get(target.0) {
            for &(ev_id, conf) in evidence {
                if let Some(ev_sym) = self.symbols.get(ev_id.0) {
                    let sim = Self::neural_similarity(&target_sym.embedding, &ev_sym.embedding);
                    if sim > 0.0 {
                        explanations.push(format!(
                            "neural_similarity({}, {})={:.3} × conf={:.3}",
                            target_sym.name, ev_sym.name, sim, conf
                        ));
                    }
                }
            }
        }

        explanations
    }

    // -----------------------------------------------------------------------
    // Utility methods
    // -----------------------------------------------------------------------

    /// Return human-readable strings describing every rule whose head or body
    /// involves `id`.
    ///
    /// # Errors
    ///
    /// Returns [`NsError::SymbolNotFound`] if `id` is not registered.
    pub fn explain_symbol(&self, id: SymbolId) -> Result<Vec<String>, NsError> {
        if id.0 >= self.symbols.len() {
            return Err(NsError::SymbolNotFound(id.0));
        }
        let sym_name = &self.symbols[id.0].name;
        let mut lines: Vec<String> = Vec::new();
        for rule in &self.rules {
            if rule.head == id || rule.body.contains(&id) {
                let head_name = self
                    .symbols
                    .get(rule.head.0)
                    .map(|s| s.name.as_str())
                    .unwrap_or("?");
                let body_names: Vec<&str> = rule
                    .body
                    .iter()
                    .map(|b| {
                        self.symbols
                            .get(b.0)
                            .map(|s| s.name.as_str())
                            .unwrap_or("?")
                    })
                    .collect();
                lines.push(format!(
                    "{} ← [{}] (w={:.3}, {:?})",
                    head_name,
                    body_names.join(", "),
                    rule.weight,
                    rule.rule_type
                ));
            }
        }
        if lines.is_empty() {
            lines.push(format!("'{}' has no associated rules", sym_name));
        }
        Ok(lines)
    }

    /// Return the top-`k` symbols most similar (by cosine similarity) to the
    /// embedding of `id`, sorted by descending similarity.
    ///
    /// Only symbols with similarity ≥ [`IntegratorConfig::similarity_threshold`]
    /// are included.  The queried symbol itself is excluded.
    ///
    /// # Errors
    ///
    /// Returns [`NsError::SymbolNotFound`] if `id` is not registered.
    pub fn similar_symbols(&self, id: SymbolId, k: usize) -> Result<Vec<(SymbolId, f64)>, NsError> {
        if id.0 >= self.symbols.len() {
            return Err(NsError::SymbolNotFound(id.0));
        }
        let query_emb = &self.symbols[id.0].embedding;
        let threshold = self.config.similarity_threshold;

        let mut scored: Vec<(SymbolId, f64)> = self
            .symbols
            .iter()
            .filter(|s| s.id != id)
            .map(|s| {
                let sim = Self::neural_similarity(query_emb, &s.embedding);
                (s.id, sim)
            })
            .filter(|(_, sim)| *sim >= threshold)
            .collect();

        // Sort by descending similarity, then by symbol id for determinism.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(k);
        Ok(scored)
    }

    /// Return aggregate statistics for this integrator.
    pub fn stats(&self) -> NsStats {
        let avg_embedding_norm = if self.symbols.is_empty() {
            0.0
        } else {
            let total_norm: f64 = self
                .symbols
                .iter()
                .map(|s| s.embedding.iter().map(|x| x * x).sum::<f64>().sqrt())
                .sum();
            total_norm / self.symbols.len() as f64
        };
        NsStats {
            symbol_count: self.symbols.len(),
            rule_count: self.rules.len(),
            total_inferences: self.total_inferences,
            avg_embedding_norm,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        InferenceMode, IntegratorConfig, LogicalRule, NeuralSymbolicIntegrator, NsError, NsQuery,
        RuleType, SymbolId,
    };

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a unit vector of the given dimension with component `val`.
    fn uniform_emb(dim: usize, val: f64) -> Vec<f64> {
        let norm = (dim as f64 * val * val).sqrt();
        if norm == 0.0 {
            return vec![0.0; dim];
        }
        vec![val / norm; dim]
    }

    /// Build a standard-basis embedding with 1.0 at position `idx`.
    fn basis_emb(dim: usize, idx: usize) -> Vec<f64> {
        let mut v = vec![0.0; dim];
        if idx < dim {
            v[idx] = 1.0;
        }
        v
    }

    fn default_integrator() -> NeuralSymbolicIntegrator {
        NeuralSymbolicIntegrator::new(IntegratorConfig {
            embedding_dim: 4,
            max_symbols: 100,
            inference_depth: 5,
            similarity_threshold: 0.5,
        })
    }

    // -----------------------------------------------------------------------
    // SymbolId tests
    // -----------------------------------------------------------------------

    #[test]
    fn symbol_id_display() {
        let id = SymbolId(42);
        assert_eq!(id.to_string(), "sym:42");
    }

    #[test]
    fn symbol_id_ordering() {
        assert!(SymbolId(0) < SymbolId(1));
        assert!(SymbolId(5) > SymbolId(3));
        assert_eq!(SymbolId(7), SymbolId(7));
    }

    // -----------------------------------------------------------------------
    // add_symbol tests
    // -----------------------------------------------------------------------

    #[test]
    fn add_symbol_returns_sequential_ids() {
        let mut ig = default_integrator();
        let id0 = ig
            .add_symbol("a".into(), basis_emb(4, 0), 0.9)
            .expect("test setup: add symbol 'a'");
        let id1 = ig
            .add_symbol("b".into(), basis_emb(4, 1), 0.8)
            .expect("test setup: add symbol 'b'");
        assert_eq!(id0, SymbolId(0));
        assert_eq!(id1, SymbolId(1));
    }

    #[test]
    fn add_symbol_dimension_mismatch() {
        let mut ig = default_integrator();
        let err = ig.add_symbol("x".into(), vec![0.1, 0.2], 0.5).unwrap_err();
        assert_eq!(
            err,
            NsError::DimensionMismatch {
                expected: 4,
                got: 2
            }
        );
    }

    #[test]
    fn add_symbol_max_reached() {
        let mut ig = NeuralSymbolicIntegrator::new(IntegratorConfig {
            embedding_dim: 2,
            max_symbols: 2,
            inference_depth: 3,
            similarity_threshold: 0.5,
        });
        ig.add_symbol("a".into(), vec![1.0, 0.0], 1.0)
            .expect("test setup: add symbol 'a'");
        ig.add_symbol("b".into(), vec![0.0, 1.0], 1.0)
            .expect("test setup: add symbol 'b'");
        let err = ig.add_symbol("c".into(), vec![0.5, 0.5], 1.0).unwrap_err();
        assert_eq!(err, NsError::MaxSymbolsReached);
    }

    #[test]
    fn add_symbol_clamps_confidence() {
        let mut ig = default_integrator();
        let id = ig
            .add_symbol("over".into(), basis_emb(4, 0), 1.5)
            .expect("test setup: add symbol 'over'");
        assert!((ig.symbols[id.0].confidence - 1.0).abs() < 1e-9);
        let id2 = ig
            .add_symbol("neg".into(), basis_emb(4, 1), -0.3)
            .expect("test setup: add symbol 'neg'");
        assert!((ig.symbols[id2.0].confidence).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // add_rule tests
    // -----------------------------------------------------------------------

    #[test]
    fn add_rule_valid() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 1.0)
            .expect("test setup: add symbol 'a'");
        let b = ig
            .add_symbol("b".into(), basis_emb(4, 1), 1.0)
            .expect("test setup: add symbol 'b'");
        let result = ig.add_rule(LogicalRule {
            head: a,
            body: vec![b],
            weight: 0.8,
            rule_type: RuleType::Definite,
        });
        assert!(result.is_ok());
    }

    #[test]
    fn add_rule_invalid_head() {
        let mut ig = default_integrator();
        let err = ig
            .add_rule(LogicalRule {
                head: SymbolId(99),
                body: vec![],
                weight: 1.0,
                rule_type: RuleType::Definite,
            })
            .unwrap_err();
        assert!(matches!(err, NsError::InvalidRule(_)));
    }

    #[test]
    fn add_rule_invalid_body_sym() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 1.0)
            .expect("test setup: add symbol 'a'");
        let err = ig
            .add_rule(LogicalRule {
                head: a,
                body: vec![SymbolId(50)],
                weight: 0.9,
                rule_type: RuleType::Probabilistic,
            })
            .unwrap_err();
        assert!(matches!(err, NsError::InvalidRule(_)));
    }

    // -----------------------------------------------------------------------
    // neural_similarity tests
    // -----------------------------------------------------------------------

    #[test]
    fn neural_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0, 0.0];
        let sim = NeuralSymbolicIntegrator::neural_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn neural_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = NeuralSymbolicIntegrator::neural_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn neural_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = NeuralSymbolicIntegrator::neural_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-10);
    }

    #[test]
    fn neural_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        let sim = NeuralSymbolicIntegrator::neural_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn neural_similarity_partial() {
        let a = vec![1.0, 1.0];
        let b = vec![1.0, 0.0];
        let sim = NeuralSymbolicIntegrator::neural_similarity(&a, &b);
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((sim - expected).abs() < 1e-10);
    }

    #[test]
    fn neural_similarity_clamp() {
        // Vectors that are exactly parallel should give similarity 1.0;
        // and the result must be in [-1, 1].
        let a = vec![3.0, 4.0];
        let b = vec![6.0, 8.0]; // same direction, double magnitude
        let sim = NeuralSymbolicIntegrator::neural_similarity(&a, &b);
        assert!(sim <= 1.0);
        assert!(sim >= -1.0);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // symbolic_forward_chain tests
    // -----------------------------------------------------------------------

    #[test]
    fn symbolic_chain_direct_evidence() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 0.5)
            .expect("test setup: add symbol 'a'");
        // Direct evidence overrides symbol confidence.
        let conf = ig.symbolic_forward_chain(a, &[(a, 0.9)], 3);
        assert!((conf - 0.9).abs() < 1e-10);
    }

    #[test]
    fn symbolic_chain_simple_rule_definite() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 0.0)
            .expect("test setup: add symbol 'a'");
        let b = ig
            .add_symbol("b".into(), basis_emb(4, 1), 0.0)
            .expect("test setup: add symbol 'b'");
        ig.add_rule(LogicalRule {
            head: a,
            body: vec![b],
            weight: 0.9,
            rule_type: RuleType::Definite,
        })
        .expect("test setup: add rule a <- b");
        let conf = ig.symbolic_forward_chain(a, &[(b, 1.0)], 3);
        // Definite: 0.9 * 1.0 = 0.9
        assert!((conf - 0.9).abs() < 1e-9);
    }

    #[test]
    fn symbolic_chain_probabilistic_rule() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 0.0)
            .expect("test setup: add symbol 'a'");
        let b = ig
            .add_symbol("b".into(), basis_emb(4, 1), 0.0)
            .expect("test setup: add symbol 'b'");
        ig.add_rule(LogicalRule {
            head: a,
            body: vec![b],
            weight: 1.0,
            rule_type: RuleType::Probabilistic,
        })
        .expect("test setup: add probabilistic rule a <- b");
        let conf = ig.symbolic_forward_chain(a, &[(b, 0.8)], 3);
        // Probabilistic: 1.0 * 0.8 * 0.8 = 0.64
        assert!((conf - 0.64).abs() < 1e-9);
    }

    #[test]
    fn symbolic_chain_soft_rule() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 0.0)
            .expect("test setup: add symbol 'a'");
        let b = ig
            .add_symbol("b".into(), basis_emb(4, 1), 0.0)
            .expect("test setup: add symbol 'b'");
        ig.add_rule(LogicalRule {
            head: a,
            body: vec![b],
            weight: 1.0,
            rule_type: RuleType::Soft { temperature: 1.0 },
        })
        .expect("test setup: add soft rule a <- b");
        let conf = ig.symbolic_forward_chain(a, &[(b, 1.0)], 3);
        // Soft: sigmoid(1.0/1.0) = 1/(1+e^-1) ≈ 0.731
        let expected = 1.0 / (1.0 + (-1.0_f64).exp());
        assert!((conf - expected).abs() < 1e-9);
    }

    #[test]
    fn symbolic_chain_empty_body_fact() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 0.0)
            .expect("test setup: add symbol 'a'");
        ig.add_rule(LogicalRule {
            head: a,
            body: vec![],
            weight: 0.7,
            rule_type: RuleType::Definite,
        })
        .expect("test setup: add fact rule for 'a'");
        // An empty-body rule acts as a fact.
        let conf = ig.symbolic_forward_chain(a, &[], 3);
        assert!((conf - 0.7).abs() < 1e-9);
    }

    #[test]
    fn symbolic_chain_depth_limit() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 0.0)
            .expect("test setup: add symbol 'a'");
        let b = ig
            .add_symbol("b".into(), basis_emb(4, 1), 0.0)
            .expect("test setup: add symbol 'b'");
        ig.add_rule(LogicalRule {
            head: a,
            body: vec![b],
            weight: 1.0,
            rule_type: RuleType::Definite,
        })
        .expect("test setup: add rule a <- b");
        // depth=0 and no direct evidence → falls back to symbol confidence (0.0)
        let conf = ig.symbolic_forward_chain(a, &[], 0);
        assert_eq!(conf, 0.0);
    }

    #[test]
    fn symbolic_chain_multi_body() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 0.0)
            .expect("test setup: add symbol 'a'");
        let b = ig
            .add_symbol("b".into(), basis_emb(4, 1), 0.0)
            .expect("test setup: add symbol 'b'");
        let c = ig
            .add_symbol("c".into(), basis_emb(4, 2), 0.0)
            .expect("test setup: add symbol 'c'");
        ig.add_rule(LogicalRule {
            head: a,
            body: vec![b, c],
            weight: 1.0,
            rule_type: RuleType::Definite,
        })
        .expect("test setup: add rule a <- b, c");
        // body_satisfaction = 0.8 * 0.6 = 0.48; result = 1.0 * 0.48
        let conf = ig.symbolic_forward_chain(a, &[(b, 0.8), (c, 0.6)], 3);
        assert!((conf - 0.48).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // neural_forward tests
    // -----------------------------------------------------------------------

    #[test]
    fn neural_forward_no_evidence() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 1.0)
            .expect("test setup: add symbol 'a'");
        assert_eq!(ig.neural_forward(a, &[]), 0.0);
    }

    #[test]
    fn neural_forward_identical_embedding() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 1.0)
            .expect("test setup: add symbol 'a'");
        let b = ig
            .add_symbol("b".into(), basis_emb(4, 0), 1.0)
            .expect("test setup: add symbol 'b'");
        let score = ig.neural_forward(a, &[(b, 0.9)]);
        // similarity = 1.0, result = 1.0 * 0.9 = 0.9
        assert!((score - 0.9).abs() < 1e-9);
    }

    #[test]
    fn neural_forward_orthogonal_embedding() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 1.0)
            .expect("test setup: add symbol 'a'");
        let b = ig
            .add_symbol("b".into(), basis_emb(4, 1), 1.0)
            .expect("test setup: add symbol 'b'");
        let score = ig.neural_forward(a, &[(b, 1.0)]);
        assert!(score.abs() < 1e-10);
    }

    #[test]
    fn neural_forward_invalid_target() {
        let ig = default_integrator();
        // No symbols registered → returns 0.0 gracefully.
        assert_eq!(ig.neural_forward(SymbolId(99), &[]), 0.0);
    }

    // -----------------------------------------------------------------------
    // infer tests
    // -----------------------------------------------------------------------

    #[test]
    fn infer_pure_symbolic() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 0.0)
            .expect("test setup: add symbol 'a'");
        let b = ig
            .add_symbol("b".into(), basis_emb(4, 1), 0.0)
            .expect("test setup: add symbol 'b'");
        ig.add_rule(LogicalRule {
            head: a,
            body: vec![b],
            weight: 1.0,
            rule_type: RuleType::Definite,
        })
        .expect("test setup: add rule a <- b");
        let query = NsQuery {
            target: a,
            evidence: vec![(b, 1.0)],
            mode: InferenceMode::PureSymbolic,
        };
        let result = ig.infer(&query).expect("test setup: infer pure symbolic");
        assert!((result.confidence - 1.0).abs() < 1e-9);
        assert!((result.symbolic_contribution - 1.0).abs() < 1e-9);
    }

    #[test]
    fn infer_pure_neural() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 0.0)
            .expect("test setup: add symbol 'a'");
        let b = ig
            .add_symbol("b".into(), basis_emb(4, 0), 0.0)
            .expect("test setup: add symbol 'b'");
        let query = NsQuery {
            target: a,
            evidence: vec![(b, 0.7)],
            mode: InferenceMode::PureNeural,
        };
        let result = ig.infer(&query).expect("test setup: infer pure neural");
        // similarity=1.0, conf=0.7 → neural=0.7
        assert!((result.confidence - 0.7).abs() < 1e-9);
        assert!((result.neural_contribution - 0.7).abs() < 1e-9);
    }

    #[test]
    fn infer_hybrid() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 0.0)
            .expect("test setup: add symbol 'a'");
        let b = ig
            .add_symbol("b".into(), basis_emb(4, 1), 0.0)
            .expect("test setup: add symbol 'b'");
        // Rule: a ← b, weight=1.0
        ig.add_rule(LogicalRule {
            head: a,
            body: vec![b],
            weight: 1.0,
            rule_type: RuleType::Definite,
        })
        .expect("test setup: add rule a <- b");
        // Evidence: b=1.0, but a and b are orthogonal so neural=0
        let query = NsQuery {
            target: a,
            evidence: vec![(b, 1.0)],
            mode: InferenceMode::Hybrid { neural_weight: 0.5 },
        };
        let result = ig.infer(&query).expect("test setup: infer hybrid");
        // symbolic=1.0, neural≈0.0, hybrid=0.5*0 + 0.5*1.0 = 0.5
        assert!((result.confidence - 0.5).abs() < 1e-9);
    }

    #[test]
    fn infer_target_not_found() {
        let mut ig = default_integrator();
        let query = NsQuery {
            target: SymbolId(99),
            evidence: vec![],
            mode: InferenceMode::PureSymbolic,
        };
        assert_eq!(ig.infer(&query).unwrap_err(), NsError::SymbolNotFound(99));
    }

    #[test]
    fn infer_increments_counter() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 1.0)
            .expect("test setup: add symbol 'a'");
        let query = NsQuery {
            target: a,
            evidence: vec![],
            mode: InferenceMode::PureSymbolic,
        };
        ig.infer(&query).expect("test setup: first infer call");
        ig.infer(&query).expect("test setup: second infer call");
        assert_eq!(ig.total_inferences, 2);
    }

    #[test]
    fn infer_result_confidence_clamped() {
        // Rule weight > 1.0 should still produce clamped result.
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 0.0)
            .expect("test setup: add symbol 'a'");
        ig.add_rule(LogicalRule {
            head: a,
            body: vec![],
            weight: 2.0, // intentionally > 1
            rule_type: RuleType::Definite,
        })
        .expect("test setup: add rule with weight > 1");
        let query = NsQuery {
            target: a,
            evidence: vec![],
            mode: InferenceMode::PureSymbolic,
        };
        let result = ig
            .infer(&query)
            .expect("test setup: infer with oversized weight");
        assert!(result.confidence <= 1.0);
    }

    #[test]
    fn infer_explanation_nonempty_for_matching_rule() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 0.0)
            .expect("test setup: add symbol 'a'");
        let b = ig
            .add_symbol("b".into(), basis_emb(4, 1), 0.0)
            .expect("test setup: add symbol 'b'");
        ig.add_rule(LogicalRule {
            head: a,
            body: vec![b],
            weight: 0.9,
            rule_type: RuleType::Definite,
        })
        .expect("test setup: add rule a <- b");
        let query = NsQuery {
            target: a,
            evidence: vec![(b, 1.0)],
            mode: InferenceMode::PureSymbolic,
        };
        let result = ig.infer(&query).expect("test setup: infer for explanation");
        assert!(!result.explanation.is_empty());
        let rule_exp = result
            .explanation
            .iter()
            .any(|e| e.contains("rule_a_from_"));
        assert!(
            rule_exp,
            "expected rule explanation, got: {:?}",
            result.explanation
        );
    }

    // -----------------------------------------------------------------------
    // explain_symbol tests
    // -----------------------------------------------------------------------

    #[test]
    fn explain_symbol_not_found() {
        let ig = default_integrator();
        assert_eq!(
            ig.explain_symbol(SymbolId(0)),
            Err(NsError::SymbolNotFound(0))
        );
    }

    #[test]
    fn explain_symbol_no_rules() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 1.0)
            .expect("test setup: add symbol 'a'");
        let lines = ig
            .explain_symbol(a)
            .expect("test setup: explain symbol 'a'");
        assert!(!lines.is_empty());
        assert!(lines[0].contains("no associated rules"));
    }

    #[test]
    fn explain_symbol_as_head() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 1.0)
            .expect("test setup: add symbol 'a'");
        let b = ig
            .add_symbol("b".into(), basis_emb(4, 1), 1.0)
            .expect("test setup: add symbol 'b'");
        ig.add_rule(LogicalRule {
            head: a,
            body: vec![b],
            weight: 0.8,
            rule_type: RuleType::Definite,
        })
        .expect("test setup: add rule a <- b");
        let lines = ig
            .explain_symbol(a)
            .expect("test setup: explain symbol 'a'");
        assert!(lines.iter().any(|l| l.contains("a ←")));
    }

    #[test]
    fn explain_symbol_as_body() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 1.0)
            .expect("test setup: add symbol 'a'");
        let b = ig
            .add_symbol("b".into(), basis_emb(4, 1), 1.0)
            .expect("test setup: add symbol 'b'");
        ig.add_rule(LogicalRule {
            head: a,
            body: vec![b],
            weight: 0.8,
            rule_type: RuleType::Definite,
        })
        .expect("test setup: add rule a <- b");
        // b appears in the body; explain_symbol(b) should mention it.
        let lines = ig
            .explain_symbol(b)
            .expect("test setup: explain symbol 'b'");
        assert!(lines.iter().any(|l| l.contains('b')));
    }

    // -----------------------------------------------------------------------
    // similar_symbols tests
    // -----------------------------------------------------------------------

    #[test]
    fn similar_symbols_not_found() {
        let ig = default_integrator();
        assert_eq!(
            ig.similar_symbols(SymbolId(99), 5),
            Err(NsError::SymbolNotFound(99))
        );
    }

    #[test]
    fn similar_symbols_returns_top_k() {
        let mut ig = NeuralSymbolicIntegrator::new(IntegratorConfig {
            embedding_dim: 4,
            max_symbols: 100,
            inference_depth: 3,
            similarity_threshold: 0.0, // accept all
        });
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 1.0)
            .expect("test setup: add symbol 'a'");
        let _b = ig
            .add_symbol("b".into(), basis_emb(4, 0), 1.0)
            .expect("test setup: add symbol 'b'"); // sim=1
        let _c = ig
            .add_symbol("c".into(), basis_emb(4, 0), 1.0)
            .expect("test setup: add symbol 'c'"); // sim=1
        let _d = ig
            .add_symbol("d".into(), basis_emb(4, 1), 1.0)
            .expect("test setup: add symbol 'd'"); // sim=0

        let results = ig
            .similar_symbols(a, 2)
            .expect("test setup: similar symbols for 'a'");
        // Should return at most 2 entries.
        assert!(results.len() <= 2);
    }

    #[test]
    fn similar_symbols_excludes_self() {
        let mut ig = NeuralSymbolicIntegrator::new(IntegratorConfig {
            embedding_dim: 4,
            max_symbols: 100,
            inference_depth: 3,
            similarity_threshold: 0.0,
        });
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 1.0)
            .expect("test setup: add symbol 'a'");
        let results = ig
            .similar_symbols(a, 10)
            .expect("test setup: similar symbols for 'a'");
        // Only `a` is registered — no results possible.
        assert!(results.is_empty());
    }

    #[test]
    fn similar_symbols_threshold_filters() {
        let mut ig = NeuralSymbolicIntegrator::new(IntegratorConfig {
            embedding_dim: 4,
            max_symbols: 100,
            inference_depth: 3,
            similarity_threshold: 0.9, // very high threshold
        });
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 1.0)
            .expect("test setup: add symbol 'a'");
        // b is not identical → similarity < 1.0 with basis vectors...
        // but let us use an orthogonal one to ensure it's filtered.
        let _b = ig
            .add_symbol("b".into(), basis_emb(4, 1), 1.0)
            .expect("test setup: add symbol 'b'");
        let results = ig
            .similar_symbols(a, 10)
            .expect("test setup: similar symbols for 'a'");
        assert!(
            results.is_empty(),
            "orthogonal vector should be below threshold"
        );
    }

    #[test]
    fn similar_symbols_sorted_desc() {
        let mut ig = NeuralSymbolicIntegrator::new(IntegratorConfig {
            embedding_dim: 4,
            max_symbols: 100,
            inference_depth: 3,
            similarity_threshold: 0.0,
        });
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 1.0)
            .expect("test setup: add symbol 'a'");
        // b has high similarity (same direction), c has lower.
        ig.add_symbol("b".into(), basis_emb(4, 0), 1.0)
            .expect("test setup: add symbol 'b'");
        let partial: Vec<f64> = {
            let mut v = vec![0.0; 4];
            v[0] = 0.7;
            v[1] = 0.3;
            // normalise
            let norm = (0.7_f64 * 0.7 + 0.3_f64 * 0.3).sqrt();
            v.iter_mut().for_each(|x| *x /= norm);
            v
        };
        ig.add_symbol("c".into(), partial, 1.0)
            .expect("test setup: add symbol 'c'");
        let results = ig
            .similar_symbols(a, 10)
            .expect("test setup: similar symbols for 'a'");
        // Sorted descending: similarity of b > c.
        if results.len() >= 2 {
            assert!(results[0].1 >= results[1].1);
        }
    }

    // -----------------------------------------------------------------------
    // stats tests
    // -----------------------------------------------------------------------

    #[test]
    fn stats_empty_integrator() {
        let ig = default_integrator();
        let s = ig.stats();
        assert_eq!(s.symbol_count, 0);
        assert_eq!(s.rule_count, 0);
        assert_eq!(s.total_inferences, 0);
        assert_eq!(s.avg_embedding_norm, 0.0);
    }

    #[test]
    fn stats_counts_symbols_and_rules() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 1.0)
            .expect("test setup: add symbol 'a'");
        let b = ig
            .add_symbol("b".into(), basis_emb(4, 1), 1.0)
            .expect("test setup: add symbol 'b'");
        ig.add_rule(LogicalRule {
            head: a,
            body: vec![b],
            weight: 0.9,
            rule_type: RuleType::Definite,
        })
        .expect("test setup: add rule a <- b");
        let s = ig.stats();
        assert_eq!(s.symbol_count, 2);
        assert_eq!(s.rule_count, 1);
    }

    #[test]
    fn stats_avg_norm_unit_vectors() {
        let mut ig = default_integrator();
        ig.add_symbol("a".into(), basis_emb(4, 0), 1.0)
            .expect("test setup: add symbol 'a'");
        ig.add_symbol("b".into(), basis_emb(4, 1), 1.0)
            .expect("test setup: add symbol 'b'");
        let s = ig.stats();
        // Both are unit vectors; average norm should be 1.0.
        assert!((s.avg_embedding_norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn stats_tracks_inferences() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 0.5)
            .expect("test setup: add symbol 'a'");
        let q = NsQuery {
            target: a,
            evidence: vec![],
            mode: InferenceMode::PureSymbolic,
        };
        ig.infer(&q).expect("test setup: first infer call");
        ig.infer(&q).expect("test setup: second infer call");
        ig.infer(&q).expect("test setup: third infer call");
        assert_eq!(ig.stats().total_inferences, 3);
    }

    // -----------------------------------------------------------------------
    // Integration / scenario tests
    // -----------------------------------------------------------------------

    #[test]
    fn scenario_chain_of_rules() {
        // a ← b ← c, with c in evidence.
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), uniform_emb(4, 1.0), 0.0)
            .expect("test setup: add symbol 'a'");
        let b = ig
            .add_symbol("b".into(), uniform_emb(4, 1.0), 0.0)
            .expect("test setup: add symbol 'b'");
        let c = ig
            .add_symbol("c".into(), uniform_emb(4, 1.0), 0.0)
            .expect("test setup: add symbol 'c'");
        ig.add_rule(LogicalRule {
            head: a,
            body: vec![b],
            weight: 0.9,
            rule_type: RuleType::Definite,
        })
        .expect("test setup: add rule a <- b");
        ig.add_rule(LogicalRule {
            head: b,
            body: vec![c],
            weight: 0.8,
            rule_type: RuleType::Definite,
        })
        .expect("test setup: add rule b <- c");
        let q = NsQuery {
            target: a,
            evidence: vec![(c, 1.0)],
            mode: InferenceMode::PureSymbolic,
        };
        let result = ig.infer(&q).expect("test setup: infer chain a <- b <- c");
        // Chain: 0.9 * (0.8 * 1.0) = 0.72
        assert!((result.confidence - 0.72).abs() < 1e-9);
    }

    #[test]
    fn scenario_multiple_rules_max_selected() {
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 0.0)
            .expect("test setup: add symbol 'a'");
        let b = ig
            .add_symbol("b".into(), basis_emb(4, 1), 0.0)
            .expect("test setup: add symbol 'b'");
        let c = ig
            .add_symbol("c".into(), basis_emb(4, 2), 0.0)
            .expect("test setup: add symbol 'c'");
        // Two rules for a; one has higher confidence.
        ig.add_rule(LogicalRule {
            head: a,
            body: vec![b],
            weight: 0.3,
            rule_type: RuleType::Definite,
        })
        .expect("test setup: add low-weight rule a <- b");
        ig.add_rule(LogicalRule {
            head: a,
            body: vec![c],
            weight: 0.9,
            rule_type: RuleType::Definite,
        })
        .expect("test setup: add high-weight rule a <- c");
        let q = NsQuery {
            target: a,
            evidence: vec![(b, 1.0), (c, 1.0)],
            mode: InferenceMode::PureSymbolic,
        };
        let result = ig
            .infer(&q)
            .expect("test setup: infer with competing rules");
        // Max of 0.3 and 0.9 → 0.9
        assert!((result.confidence - 0.9).abs() < 1e-9);
    }

    #[test]
    fn scenario_hybrid_blends_both() {
        let mut ig = default_integrator();
        // Target and evidence share the same embedding → neural=1*conf=0.6
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 0.0)
            .expect("test setup: add symbol 'a'");
        let b = ig
            .add_symbol("b".into(), basis_emb(4, 0), 0.0)
            .expect("test setup: add symbol 'b'"); // identical emb
                                                   // Symbolic: a ← b, weight=0.8
        ig.add_rule(LogicalRule {
            head: a,
            body: vec![b],
            weight: 0.8,
            rule_type: RuleType::Definite,
        })
        .expect("test setup: add rule a <- b");
        let q = NsQuery {
            target: a,
            evidence: vec![(b, 0.6)],
            mode: InferenceMode::Hybrid { neural_weight: 0.4 },
        };
        let result = ig.infer(&q).expect("test setup: infer hybrid blend");
        // neural=0.6, symbolic=0.8*0.6=0.48
        // hybrid = 0.4*0.6 + 0.6*0.48 = 0.24 + 0.288 = 0.528
        assert!((result.confidence - 0.528).abs() < 1e-9);
    }

    #[test]
    fn scenario_soft_rule_low_temperature() {
        // At very low temperature, soft rule approaches a step function.
        let mut ig = default_integrator();
        let a = ig
            .add_symbol("a".into(), basis_emb(4, 0), 0.0)
            .expect("test setup: add symbol 'a'");
        let b = ig
            .add_symbol("b".into(), basis_emb(4, 1), 0.0)
            .expect("test setup: add symbol 'b'");
        ig.add_rule(LogicalRule {
            head: a,
            body: vec![b],
            weight: 1.0,
            rule_type: RuleType::Soft { temperature: 0.01 },
        })
        .expect("test setup: add soft rule a <- b");
        // Body sat = 1.0 → sigmoid(1.0/0.01)=sigmoid(100)≈1.0
        let conf = ig.symbolic_forward_chain(a, &[(b, 1.0)], 3);
        assert!(conf > 0.99, "expected ≈1.0, got {}", conf);
    }
}
