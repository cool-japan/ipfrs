//! Automatic Proof Explanation
//!
//! This module provides automatic generation of natural language explanations
//! for proofs, making them more interpretable and understandable.
//!
//! ## Features
//!
//! - Natural language proof explanations
//! - Multiple explanation styles (concise, detailed, pedagogical)
//! - Template-based explanation generation
//! - Custom explanation templates
//! - Proof step highlighting
//! - Interactive explanation navigation
//!
//! ## Examples
//!
//! ```
//! use ipfrs_tensorlogic::{ProofExplainer, ExplanationStyle, Proof, Predicate, Term, Constant};
//!
//! // Create a simple proof
//! let goal = Predicate {
//!     name: "mortal".to_string(),
//!     args: vec![Term::Const(Constant::String("socrates".to_string()))],
//! };
//! let proof = Proof {
//!     goal,
//!     rule: None,
//!     subproofs: vec![],
//! };
//!
//! let explainer = ProofExplainer::new();
//! let explanation = explainer.explain(&proof, ExplanationStyle::Detailed);
//! println!("{}", explanation);
//! ```

use crate::ir::{Constant, Predicate, Term};
use crate::proof_storage::{ProofFragment, ProofMetadata};
use crate::reasoning::Proof;

#[cfg(test)]
use crate::reasoning::ProofRule;

/// Explanation style for proof explanations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplanationStyle {
    /// Concise one-liner explanations
    Concise,
    /// Detailed step-by-step explanations
    Detailed,
    /// Pedagogical explanations with reasoning context
    Pedagogical,
    /// Formal logical notation
    Formal,
}

/// Proof explanation configuration
#[derive(Debug, Clone)]
pub struct ExplanationConfig {
    /// Explanation style
    pub style: ExplanationStyle,
    /// Include premise details
    pub include_premises: bool,
    /// Include substitution details
    pub include_substitutions: bool,
    /// Maximum depth for nested explanations
    pub max_depth: usize,
    /// Use natural language for predicates
    pub naturalize_predicates: bool,
}

impl Default for ExplanationConfig {
    fn default() -> Self {
        Self {
            style: ExplanationStyle::Detailed,
            include_premises: true,
            include_substitutions: false,
            max_depth: 10,
            naturalize_predicates: true,
        }
    }
}

impl ExplanationConfig {
    /// Create a concise explanation config
    pub fn concise() -> Self {
        Self {
            style: ExplanationStyle::Concise,
            include_premises: false,
            include_substitutions: false,
            max_depth: 3,
            naturalize_predicates: true,
        }
    }

    /// Create a detailed explanation config
    pub fn detailed() -> Self {
        Self {
            style: ExplanationStyle::Detailed,
            include_premises: true,
            include_substitutions: false,
            max_depth: 10,
            naturalize_predicates: true,
        }
    }

    /// Create a pedagogical explanation config
    pub fn pedagogical() -> Self {
        Self {
            style: ExplanationStyle::Pedagogical,
            include_premises: true,
            include_substitutions: true,
            max_depth: 10,
            naturalize_predicates: true,
        }
    }

    /// Create a formal explanation config
    pub fn formal() -> Self {
        Self {
            style: ExplanationStyle::Formal,
            include_premises: true,
            include_substitutions: true,
            max_depth: 10,
            naturalize_predicates: false,
        }
    }
}

/// Natural language proof explainer
pub struct ProofExplainer {
    config: ExplanationConfig,
}

impl ProofExplainer {
    /// Create a new proof explainer with default config
    pub fn new() -> Self {
        Self {
            config: ExplanationConfig::default(),
        }
    }

    /// Create a new proof explainer with custom config
    pub fn with_config(config: ExplanationConfig) -> Self {
        Self { config }
    }

    /// Generate explanation for a proof
    pub fn explain(&self, proof: &Proof, style: ExplanationStyle) -> String {
        let mut config = self.config.clone();
        config.style = style;
        self.explain_with_config(proof, &config, 0)
    }

    /// Generate explanation with custom config
    pub fn explain_with_config(
        &self,
        proof: &Proof,
        config: &ExplanationConfig,
        depth: usize,
    ) -> String {
        if depth > config.max_depth {
            return "[...proof continues...]".to_string();
        }

        match config.style {
            ExplanationStyle::Concise => self.explain_concise(proof, depth),
            ExplanationStyle::Detailed => self.explain_detailed(proof, config, depth),
            ExplanationStyle::Pedagogical => self.explain_pedagogical(proof, config, depth),
            ExplanationStyle::Formal => self.explain_formal(proof, config, depth),
        }
    }

    /// Generate concise explanation
    fn explain_concise(&self, proof: &Proof, depth: usize) -> String {
        let indent = "  ".repeat(depth);
        let goal_str = self.predicate_to_string(&proof.goal);

        if let Some(rule) = &proof.rule {
            if rule.is_fact {
                format!("{}✓ {} (given fact)", indent, goal_str)
            } else {
                format!("{}✓ {} (by rule)", indent, goal_str)
            }
        } else {
            format!("{}✓ {} (assumed)", indent, goal_str)
        }
    }

    /// Generate detailed explanation
    fn explain_detailed(&self, proof: &Proof, config: &ExplanationConfig, depth: usize) -> String {
        let indent = "  ".repeat(depth);
        let mut result = String::new();

        let goal_str = self.predicate_to_string(&proof.goal);

        if let Some(rule) = &proof.rule {
            if rule.is_fact {
                result.push_str(&format!(
                    "{}We know that {} because it is a given fact.\n",
                    indent, goal_str
                ));
            } else {
                result.push_str(&format!(
                    "{}To prove that {}, we apply a rule:\n",
                    indent, goal_str
                ));

                if config.include_premises && !proof.subproofs.is_empty() {
                    result.push_str(&format!("{}  This requires proving:\n", indent));
                    for (i, subproof) in proof.subproofs.iter().enumerate() {
                        result.push_str(&format!(
                            "{}  {}. {}\n",
                            indent,
                            i + 1,
                            self.predicate_to_string(&subproof.goal)
                        ));

                        let sub_explanation = self.explain_with_config(subproof, config, depth + 2);
                        result.push_str(&sub_explanation);
                    }
                }

                result.push_str(&format!("{}  Therefore, {} holds.\n", indent, goal_str));
            }
        } else {
            result.push_str(&format!("{}Assume that {}.\n", indent, goal_str));
        }

        result
    }

    /// Generate pedagogical explanation
    fn explain_pedagogical(
        &self,
        proof: &Proof,
        config: &ExplanationConfig,
        depth: usize,
    ) -> String {
        let indent = "  ".repeat(depth);
        let mut result = String::new();

        let goal_str = self.predicate_to_string(&proof.goal);

        if let Some(rule) = &proof.rule {
            if rule.is_fact {
                result.push_str(&format!(
                    "{}[Base Case] We start with the fact that {}.\n",
                    indent, goal_str
                ));
                result.push_str(&format!(
                    "{}This is an axiom or given information in our knowledge base.\n",
                    indent
                ));
            } else {
                result.push_str(&format!(
                    "{}[Deduction Step] Goal: Prove that {}\n",
                    indent, goal_str
                ));
                result.push_str(&format!("{}Strategy: Apply a logical rule\n", indent));

                if !rule.body.is_empty() {
                    result.push_str(&format!(
                        "{}This rule states: IF {} THEN {}\n",
                        indent,
                        rule.body
                            .iter()
                            .map(|p| self.predicate_to_string(p))
                            .collect::<Vec<_>>()
                            .join(" AND "),
                        self.predicate_to_string(&rule.head)
                    ));
                }

                if config.include_premises && !proof.subproofs.is_empty() {
                    result.push_str(&format!(
                        "{}To apply this rule, we must first establish {} condition(s):\n",
                        indent,
                        proof.subproofs.len()
                    ));
                    for (i, subproof) in proof.subproofs.iter().enumerate() {
                        result.push_str(&format!(
                            "{}Condition {}: {}\n",
                            indent,
                            i + 1,
                            self.predicate_to_string(&subproof.goal)
                        ));
                        let sub_explanation = self.explain_with_config(subproof, config, depth + 1);
                        result.push_str(&sub_explanation);
                    }
                }

                result.push_str(&format!(
                    "{}Since all conditions are satisfied, we conclude that {}.\n",
                    indent, goal_str
                ));
            }
        } else {
            result.push_str(&format!(
                "{}[Assumption] We assume that {}.\n",
                indent, goal_str
            ));
        }

        result
    }

    /// Generate formal explanation
    fn explain_formal(&self, proof: &Proof, config: &ExplanationConfig, depth: usize) -> String {
        let indent = "  ".repeat(depth);
        let mut result = String::new();

        result.push_str(&format!(
            "{}⊢ {}\n",
            indent,
            self.predicate_to_formal(&proof.goal)
        ));

        if let Some(rule) = &proof.rule {
            if rule.is_fact {
                result.push_str(&format!("{}  [Axiom]\n", indent));
            } else {
                if config.include_premises && !proof.subproofs.is_empty() {
                    for subproof in &proof.subproofs {
                        let sub_explanation = self.explain_with_config(subproof, config, depth + 1);
                        result.push_str(&sub_explanation);
                    }
                }
                result.push_str(&format!("{}  [Apply Rule]\n", indent));
            }
        } else {
            result.push_str(&format!("{}  [Assume]\n", indent));
        }

        result
    }

    /// Convert predicate to natural language string
    fn predicate_to_string(&self, pred: &Predicate) -> String {
        if self.config.naturalize_predicates {
            self.naturalize_predicate(pred)
        } else {
            self.predicate_to_formal(pred)
        }
    }

    /// Convert predicate to formal notation
    fn predicate_to_formal(&self, pred: &Predicate) -> String {
        if pred.args.is_empty() {
            pred.name.clone()
        } else {
            format!(
                "{}({})",
                pred.name,
                pred.args
                    .iter()
                    .map(|t| self.term_to_string(t))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }

    /// Convert predicate to natural language
    pub fn naturalize_predicate(&self, pred: &Predicate) -> String {
        // Try to convert common predicates to natural language
        match pred.name.as_str() {
            "mortal" if pred.args.len() == 1 => {
                format!("{} is mortal", self.term_to_string(&pred.args[0]))
            }
            "human" if pred.args.len() == 1 => {
                format!("{} is a human", self.term_to_string(&pred.args[0]))
            }
            "parent" if pred.args.len() == 2 => {
                format!(
                    "{} is a parent of {}",
                    self.term_to_string(&pred.args[0]),
                    self.term_to_string(&pred.args[1])
                )
            }
            "ancestor" if pred.args.len() == 2 => {
                format!(
                    "{} is an ancestor of {}",
                    self.term_to_string(&pred.args[0]),
                    self.term_to_string(&pred.args[1])
                )
            }
            "greater_than" | "gt" if pred.args.len() == 2 => {
                format!(
                    "{} > {}",
                    self.term_to_string(&pred.args[0]),
                    self.term_to_string(&pred.args[1])
                )
            }
            "less_than" | "lt" if pred.args.len() == 2 => {
                format!(
                    "{} < {}",
                    self.term_to_string(&pred.args[0]),
                    self.term_to_string(&pred.args[1])
                )
            }
            "equal" | "eq" if pred.args.len() == 2 => {
                format!(
                    "{} = {}",
                    self.term_to_string(&pred.args[0]),
                    self.term_to_string(&pred.args[1])
                )
            }
            _ => {
                // Default to formal notation
                self.predicate_to_formal(pred)
            }
        }
    }

    /// Convert term to string
    fn term_to_string(&self, term: &Term) -> String {
        match term {
            Term::Var(v) => format!("?{}", v),
            Term::Const(c) => self.constant_to_string(c),
            Term::Fun(f, args) => {
                if args.is_empty() {
                    f.clone()
                } else {
                    format!(
                        "{}({})",
                        f,
                        args.iter()
                            .map(|t| self.term_to_string(t))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            }
            Term::Ref(_) => "[ref]".to_string(),
        }
    }

    /// Convert constant to string
    fn constant_to_string(&self, constant: &Constant) -> String {
        match constant {
            Constant::String(s) => s.clone(),
            Constant::Int(i) => i.to_string(),
            Constant::Bool(b) => b.to_string(),
            Constant::Float(f) => f.clone(),
        }
    }
}

impl Default for ProofExplainer {
    fn default() -> Self {
        Self::new()
    }
}

/// Proof explanation builder for fluent API
pub struct ProofExplanationBuilder {
    proof: Proof,
    config: ExplanationConfig,
}

impl ProofExplanationBuilder {
    /// Create a new builder for a proof
    pub fn new(proof: Proof) -> Self {
        Self {
            proof,
            config: ExplanationConfig::default(),
        }
    }

    /// Set explanation style
    pub fn style(mut self, style: ExplanationStyle) -> Self {
        self.config.style = style;
        self
    }

    /// Include premise details
    pub fn with_premises(mut self) -> Self {
        self.config.include_premises = true;
        self
    }

    /// Include substitution details
    pub fn with_substitutions(mut self) -> Self {
        self.config.include_substitutions = true;
        self
    }

    /// Set maximum depth
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.config.max_depth = depth;
        self
    }

    /// Use natural language
    pub fn natural_language(mut self) -> Self {
        self.config.naturalize_predicates = true;
        self
    }

    /// Use formal notation
    pub fn formal_notation(mut self) -> Self {
        self.config.naturalize_predicates = false;
        self
    }

    /// Build the explanation
    pub fn build(self) -> String {
        let explainer = ProofExplainer::with_config(self.config.clone());
        explainer.explain_with_config(&self.proof, &self.config, 0)
    }
}

/// Fragment-based proof explainer for content-addressed proofs
pub struct FragmentProofExplainer {
    explainer: ProofExplainer,
}

impl FragmentProofExplainer {
    /// Create a new fragment proof explainer
    pub fn new() -> Self {
        Self {
            explainer: ProofExplainer::new(),
        }
    }

    /// Explain a proof fragment
    pub fn explain_fragment(&self, fragment: &ProofFragment, style: ExplanationStyle) -> String {
        let mut result = String::new();

        match style {
            ExplanationStyle::Concise => {
                result.push_str(&format!(
                    "✓ {}",
                    self.explainer.predicate_to_string(&fragment.conclusion)
                ));
                if let Some(rule) = &fragment.rule_applied {
                    result.push_str(&format!(" (by rule '{}')", rule.rule_id));
                }
            }
            ExplanationStyle::Detailed => {
                result.push_str(&format!(
                    "Proof fragment for: {}\n",
                    self.explainer.predicate_to_string(&fragment.conclusion)
                ));
                if let Some(rule) = &fragment.rule_applied {
                    result.push_str(&format!("Applied rule: {}\n", rule.rule_id));
                }
                if !fragment.premise_refs.is_empty() {
                    result.push_str(&format!(
                        "Number of premises: {}\n",
                        fragment.premise_refs.len()
                    ));
                }
                if !fragment.substitution.is_empty() {
                    result.push_str("Substitutions:\n");
                    for (var, term) in &fragment.substitution {
                        result.push_str(&format!(
                            "  {} ← {}\n",
                            var,
                            self.explainer.term_to_string(term)
                        ));
                    }
                }
            }
            ExplanationStyle::Pedagogical => {
                result.push_str(&format!(
                    "[Proof Step] To establish that {}\n",
                    self.explainer.predicate_to_string(&fragment.conclusion)
                ));
                if let Some(rule) = &fragment.rule_applied {
                    result.push_str(&format!("We apply the rule: {}\n", rule.rule_id));
                }
                if !fragment.premise_refs.is_empty() {
                    result.push_str(&format!(
                        "This requires {} supporting facts:\n",
                        fragment.premise_refs.len()
                    ));
                    for (i, _premise) in fragment.premise_refs.iter().enumerate() {
                        result.push_str(&format!("  {}. [Referenced proof fragment]\n", i + 1));
                    }
                }
            }
            ExplanationStyle::Formal => {
                result.push_str(&format!(
                    "⊢ {}\n",
                    self.explainer.predicate_to_formal(&fragment.conclusion)
                ));
                if let Some(rule) = &fragment.rule_applied {
                    result.push_str(&format!("  [Apply {}]\n", rule.rule_id));
                }
            }
        }

        result
    }

    /// Explain proof metadata
    pub fn explain_metadata(&self, metadata: &ProofMetadata) -> String {
        let mut result = String::new();
        result.push_str("Proof Metadata:\n");
        if let Some(created_at) = metadata.created_at {
            result.push_str(&format!("  Created at: {}\n", created_at));
        }
        if let Some(author) = &metadata.created_by {
            result.push_str(&format!("  Author: {}\n", author));
        }
        if let Some(complexity) = metadata.complexity {
            result.push_str(&format!("  Complexity: {} steps\n", complexity));
        }
        result.push_str(&format!("  Depth: {}\n", metadata.depth));
        if !metadata.custom.is_empty() {
            result.push_str("  Custom fields:\n");
            for (key, value) in &metadata.custom {
                result.push_str(&format!("    {}: {}\n", key, value));
            }
        }
        result
    }
}

impl Default for FragmentProofExplainer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_simple_fact() -> Proof {
        Proof {
            goal: Predicate {
                name: "mortal".to_string(),
                args: vec![Term::Const(Constant::String("socrates".to_string()))],
            },
            rule: Some(ProofRule {
                head: Predicate {
                    name: "mortal".to_string(),
                    args: vec![Term::Const(Constant::String("socrates".to_string()))],
                },
                body: vec![],
                is_fact: true,
            }),
            subproofs: vec![],
        }
    }

    fn create_rule_proof() -> Proof {
        let premise = Proof {
            goal: Predicate {
                name: "human".to_string(),
                args: vec![Term::Const(Constant::String("socrates".to_string()))],
            },
            rule: Some(ProofRule {
                head: Predicate {
                    name: "human".to_string(),
                    args: vec![Term::Const(Constant::String("socrates".to_string()))],
                },
                body: vec![],
                is_fact: true,
            }),
            subproofs: vec![],
        };

        Proof {
            goal: Predicate {
                name: "mortal".to_string(),
                args: vec![Term::Const(Constant::String("socrates".to_string()))],
            },
            rule: Some(ProofRule {
                head: Predicate {
                    name: "mortal".to_string(),
                    args: vec![Term::Var("X".to_string())],
                },
                body: vec![Predicate {
                    name: "human".to_string(),
                    args: vec![Term::Var("X".to_string())],
                }],
                is_fact: false,
            }),
            subproofs: vec![premise],
        }
    }

    #[test]
    fn test_concise_explanation() {
        let proof = create_simple_fact();
        let explainer = ProofExplainer::new();
        let explanation = explainer.explain(&proof, ExplanationStyle::Concise);
        assert!(explanation.contains("socrates is mortal"));
        assert!(explanation.contains("given fact"));
    }

    #[test]
    fn test_detailed_explanation() {
        let proof = create_rule_proof();
        let explainer = ProofExplainer::new();
        let explanation = explainer.explain(&proof, ExplanationStyle::Detailed);
        assert!(explanation.contains("socrates"));
    }

    #[test]
    fn test_pedagogical_explanation() {
        let proof = create_rule_proof();
        let explainer = ProofExplainer::new();
        let explanation = explainer.explain(&proof, ExplanationStyle::Pedagogical);
        assert!(explanation.contains("Goal"));
        assert!(explanation.contains("Strategy"));
    }

    #[test]
    fn test_formal_explanation() {
        let proof = create_simple_fact();
        let explainer = ProofExplainer::new();
        let explanation = explainer.explain(&proof, ExplanationStyle::Formal);
        assert!(explanation.contains("⊢"));
        assert!(explanation.contains("Axiom"));
    }

    #[test]
    fn test_builder_pattern() {
        let proof = create_simple_fact();
        let explanation = ProofExplanationBuilder::new(proof)
            .style(ExplanationStyle::Concise)
            .natural_language()
            .build();
        assert!(explanation.contains("socrates"));
    }

    #[test]
    fn test_predicate_naturalization() {
        let explainer = ProofExplainer::new();
        let pred = Predicate {
            name: "parent".to_string(),
            args: vec![
                Term::Const(Constant::String("alice".to_string())),
                Term::Const(Constant::String("bob".to_string())),
            ],
        };
        let natural = explainer.naturalize_predicate(&pred);
        assert!(natural.contains("alice is a parent of bob"));
    }

    #[test]
    fn test_config_presets() {
        let concise = ExplanationConfig::concise();
        assert_eq!(concise.style, ExplanationStyle::Concise);
        assert!(!concise.include_premises);

        let detailed = ExplanationConfig::detailed();
        assert_eq!(detailed.style, ExplanationStyle::Detailed);
        assert!(detailed.include_premises);

        let pedagogical = ExplanationConfig::pedagogical();
        assert_eq!(pedagogical.style, ExplanationStyle::Pedagogical);
        assert!(pedagogical.include_substitutions);
    }
}
