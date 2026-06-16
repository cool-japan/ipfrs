//! TensorLogic rule validator.
//!
//! Validates TensorLogic rules for structural correctness, safety, and
//! resource bounds before they are added to the knowledge base.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can be produced during rule validation.
#[derive(Clone, Debug, PartialEq)]
pub enum ValidationError {
    /// The rule head contains no terms.
    EmptyHead,
    /// A variable that appears in the rule head is not bound in any body term.
    UnboundVariable { var_name: String },
    /// The rule directly depends on itself (circular dependency).
    CircularDependency { rule_id: u64 },
    /// The number of body terms exceeds the configured maximum.
    ExcessiveBodyLength { length: usize, max: usize },
    /// An identical head signature is already registered in the knowledge base.
    DuplicateHead { existing_id: u64 },
    /// The rule weight is outside the allowed range `(0.0, 1.0]`.
    InvalidWeight { weight: f64 },
}

// ---------------------------------------------------------------------------
// Validation result
// ---------------------------------------------------------------------------

/// The outcome of validating a single rule.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// The identifier of the rule that was validated.
    pub rule_id: u64,
    /// All structural/safety errors found during validation.
    pub errors: Vec<ValidationError>,
    /// Non-fatal advisory messages produced during validation.
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Returns `true` when no errors were produced.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Returns the number of errors collected.
    #[inline]
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }
}

// ---------------------------------------------------------------------------
// Validator configuration
// ---------------------------------------------------------------------------

/// Configuration knobs for [`TensorRuleValidator`].
#[derive(Debug, Clone)]
pub struct ValidatorConfig {
    /// Maximum number of body terms allowed per rule.  Default: `64`.
    pub max_body_length: usize,
    /// When `false` (default) a rule whose head signature matches an already-
    /// registered rule is rejected with [`ValidationError::DuplicateHead`].
    pub allow_duplicate_heads: bool,
    /// When `true` (default) the rule weight must be in `(0.0, 1.0]`.
    pub require_positive_weight: bool,
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            max_body_length: 64,
            allow_duplicate_heads: false,
            require_positive_weight: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Rule specification
// ---------------------------------------------------------------------------

/// A rule to be validated.
#[derive(Debug, Clone)]
pub struct RuleSpec {
    /// Unique identifier of the rule.
    pub rule_id: u64,
    /// Terms that form the rule head (conclusion).
    pub head_terms: Vec<String>,
    /// Terms that form the rule body (premises).
    pub body_terms: Vec<String>,
    /// Variables that appear in the head — every one must occur as a substring
    /// in at least one body term.
    pub variables: Vec<String>,
    /// Confidence / probability weight of the rule.
    pub weight: f64,
    /// IDs of other rules this rule depends on (used for circular-dependency
    /// detection).
    pub depends_on: Vec<u64>,
}

// ---------------------------------------------------------------------------
// Validator
// ---------------------------------------------------------------------------

/// Validates [`RuleSpec`] instances before they are committed to the
/// knowledge base.
pub struct TensorRuleValidator {
    /// Validation policy settings.
    pub config: ValidatorConfig,
    /// Maps a head signature (`head_terms` joined with `"|"`) to the
    /// `rule_id` of the rule that first registered it.
    pub registered_heads: HashMap<String, u64>,
}

impl TensorRuleValidator {
    /// Creates a new validator with the supplied configuration.
    pub fn new(config: ValidatorConfig) -> Self {
        Self {
            config,
            registered_heads: HashMap::new(),
        }
    }

    /// Validates `spec` and returns a [`ValidationResult`] describing every
    /// error and warning found.  When the result contains no errors the head
    /// signature is automatically registered so that subsequent duplicate
    /// checks can detect it.
    pub fn validate(&mut self, spec: &RuleSpec) -> ValidationResult {
        let mut errors: Vec<ValidationError> = Vec::new();
        let mut warnings: Vec<String> = Vec::new();

        // ── 1. Empty head ────────────────────────────────────────────────────
        if spec.head_terms.is_empty() {
            errors.push(ValidationError::EmptyHead);
        }

        // ── 2. Unbound variables ─────────────────────────────────────────────
        for var in &spec.variables {
            let bound = spec
                .body_terms
                .iter()
                .any(|body_term| body_term.contains(var.as_str()));
            if !bound {
                errors.push(ValidationError::UnboundVariable {
                    var_name: var.clone(),
                });
            }
        }

        // ── 3. Circular dependency ───────────────────────────────────────────
        if spec.depends_on.contains(&spec.rule_id) {
            errors.push(ValidationError::CircularDependency {
                rule_id: spec.rule_id,
            });
        }

        // ── 4. Excessive body length ─────────────────────────────────────────
        if spec.body_terms.len() > self.config.max_body_length {
            errors.push(ValidationError::ExcessiveBodyLength {
                length: spec.body_terms.len(),
                max: self.config.max_body_length,
            });
        }

        // ── 5. Duplicate head ────────────────────────────────────────────────
        if !self.config.allow_duplicate_heads {
            let signature = spec.head_terms.join("|");
            if let Some(&existing_id) = self.registered_heads.get(&signature) {
                errors.push(ValidationError::DuplicateHead { existing_id });
            }
        }

        // ── 6. Invalid weight ────────────────────────────────────────────────
        if self.config.require_positive_weight && (spec.weight <= 0.0 || spec.weight > 1.0) {
            errors.push(ValidationError::InvalidWeight {
                weight: spec.weight,
            });
        }

        // ── 7. Fact-assertion warning ────────────────────────────────────────
        if spec.body_terms.is_empty() && !spec.head_terms.is_empty() {
            warnings.push("Rule has no body terms (fact assertion)".to_string());
        }

        // ── 8. Register on success ───────────────────────────────────────────
        if errors.is_empty() {
            let signature = spec.head_terms.join("|");
            self.registered_heads.insert(signature, spec.rule_id);
        }

        ValidationResult {
            rule_id: spec.rule_id,
            errors,
            warnings,
        }
    }

    /// Returns how many valid rules have been registered.
    pub fn registered_count(&self) -> usize {
        self.registered_heads.len()
    }

    /// Removes all previously registered head signatures.
    pub fn clear_registry(&mut self) {
        self.registered_heads.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn default_validator() -> TensorRuleValidator {
        TensorRuleValidator::new(ValidatorConfig::default())
    }

    /// Returns a minimal valid rule that passes every check.
    fn valid_spec() -> RuleSpec {
        RuleSpec {
            rule_id: 1,
            head_terms: vec!["parent(X, Y)".to_string()],
            body_terms: vec!["father(X, Y)".to_string()],
            variables: vec!["X".to_string(), "Y".to_string()],
            weight: 1.0,
            depends_on: vec![],
        }
    }

    // ── 1. Valid rule passes ─────────────────────────────────────────────────
    #[test]
    fn test_valid_rule_passes() {
        let mut v = default_validator();
        let result = v.validate(&valid_spec());
        assert!(result.is_valid());
        assert_eq!(result.error_count(), 0);
    }

    // ── 2. EmptyHead detected ────────────────────────────────────────────────
    #[test]
    fn test_empty_head_detected() {
        let mut v = default_validator();
        let spec = RuleSpec {
            head_terms: vec![],
            ..valid_spec()
        };
        let result = v.validate(&spec);
        assert!(!result.is_valid());
        assert!(result.errors.contains(&ValidationError::EmptyHead));
    }

    // ── 3. UnboundVariable detected ──────────────────────────────────────────
    #[test]
    fn test_unbound_variable_detected() {
        let mut v = default_validator();
        let spec = RuleSpec {
            variables: vec!["Z".to_string()],
            body_terms: vec!["father(X, Y)".to_string()],
            ..valid_spec()
        };
        let result = v.validate(&spec);
        assert!(!result.is_valid());
        assert!(result.errors.contains(&ValidationError::UnboundVariable {
            var_name: "Z".to_string()
        }));
    }

    // ── 4. Bound variable passes ─────────────────────────────────────────────
    #[test]
    fn test_bound_variable_passes() {
        let mut v = default_validator();
        let spec = RuleSpec {
            variables: vec!["X".to_string()],
            body_terms: vec!["node(X)".to_string()],
            ..valid_spec()
        };
        let result = v.validate(&spec);
        assert!(result.is_valid());
    }

    // ── 5. CircularDependency detected ───────────────────────────────────────
    #[test]
    fn test_circular_dependency_detected() {
        let mut v = default_validator();
        let spec = RuleSpec {
            depends_on: vec![1],
            ..valid_spec()
        };
        let result = v.validate(&spec);
        assert!(!result.is_valid());
        assert!(result
            .errors
            .contains(&ValidationError::CircularDependency { rule_id: 1 }));
    }

    // ── 6. ExcessiveBodyLength at limit + 1 ──────────────────────────────────
    #[test]
    fn test_excessive_body_length_detected() {
        let mut v = default_validator();
        let body: Vec<String> = (0..65).map(|i| format!("term_{i}(X)")).collect();
        let spec = RuleSpec {
            body_terms: body,
            variables: vec!["X".to_string()],
            ..valid_spec()
        };
        let result = v.validate(&spec);
        assert!(!result.is_valid());
        assert!(result.errors.iter().any(|e| matches!(
            e,
            ValidationError::ExcessiveBodyLength {
                length: 65,
                max: 64
            }
        )));
    }

    // ── 7. ExcessiveBodyLength passes at exactly the limit ───────────────────
    #[test]
    fn test_body_length_at_limit_passes() {
        let mut v = default_validator();
        let body: Vec<String> = (0..64).map(|i| format!("term_{i}(X)")).collect();
        let spec = RuleSpec {
            body_terms: body,
            variables: vec!["X".to_string()],
            ..valid_spec()
        };
        let result = v.validate(&spec);
        assert!(result.is_valid());
    }

    // ── 8. DuplicateHead when allow_duplicate_heads = false ──────────────────
    #[test]
    fn test_duplicate_head_detected() {
        let mut v = default_validator();
        let spec = valid_spec();
        let first = v.validate(&spec);
        assert!(first.is_valid());

        let spec2 = RuleSpec {
            rule_id: 2,
            ..valid_spec()
        };
        let second = v.validate(&spec2);
        assert!(!second.is_valid());
        assert!(second
            .errors
            .contains(&ValidationError::DuplicateHead { existing_id: 1 }));
    }

    // ── 9. DuplicateHead allowed when allow_duplicate_heads = true ───────────
    #[test]
    fn test_duplicate_head_allowed() {
        let config = ValidatorConfig {
            allow_duplicate_heads: true,
            ..Default::default()
        };
        let mut v = TensorRuleValidator::new(config);
        let spec = valid_spec();
        let first = v.validate(&spec);
        assert!(first.is_valid());

        let spec2 = RuleSpec {
            rule_id: 2,
            ..valid_spec()
        };
        let second = v.validate(&spec2);
        assert!(second.is_valid());
    }

    // ── 10. InvalidWeight at 0.0 ─────────────────────────────────────────────
    #[test]
    fn test_invalid_weight_zero() {
        let mut v = default_validator();
        let spec = RuleSpec {
            weight: 0.0,
            ..valid_spec()
        };
        let result = v.validate(&spec);
        assert!(!result.is_valid());
        assert!(result
            .errors
            .contains(&ValidationError::InvalidWeight { weight: 0.0 }));
    }

    // ── 11. InvalidWeight at negative ────────────────────────────────────────
    #[test]
    fn test_invalid_weight_negative() {
        let mut v = default_validator();
        let spec = RuleSpec {
            weight: -0.5,
            ..valid_spec()
        };
        let result = v.validate(&spec);
        assert!(!result.is_valid());
        assert!(result
            .errors
            .contains(&ValidationError::InvalidWeight { weight: -0.5 }));
    }

    // ── 12. InvalidWeight at 1.01 ────────────────────────────────────────────
    #[test]
    fn test_invalid_weight_above_one() {
        let mut v = default_validator();
        let spec = RuleSpec {
            weight: 1.01,
            ..valid_spec()
        };
        let result = v.validate(&spec);
        assert!(!result.is_valid());
        assert!(result
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::InvalidWeight { .. })));
    }

    // ── 13. Weight at exactly 1.0 passes ────────────────────────────────────
    #[test]
    fn test_weight_exactly_one_passes() {
        let mut v = default_validator();
        let spec = RuleSpec {
            weight: 1.0,
            ..valid_spec()
        };
        let result = v.validate(&spec);
        assert!(result.is_valid());
    }

    // ── 14. Weight at 0.5 passes ─────────────────────────────────────────────
    #[test]
    fn test_weight_point_five_passes() {
        let mut v = default_validator();
        let spec = RuleSpec {
            weight: 0.5,
            ..valid_spec()
        };
        let result = v.validate(&spec);
        assert!(result.is_valid());
    }

    // ── 15. is_valid true when no errors ────────────────────────────────────
    #[test]
    fn test_is_valid_true_no_errors() {
        let mut v = default_validator();
        let result = v.validate(&valid_spec());
        assert!(result.is_valid());
        assert_eq!(result.error_count(), 0);
    }

    // ── 16. is_valid false when errors present ───────────────────────────────
    #[test]
    fn test_is_valid_false_with_errors() {
        let mut v = default_validator();
        let spec = RuleSpec {
            head_terms: vec![],
            ..valid_spec()
        };
        let result = v.validate(&spec);
        assert!(!result.is_valid());
        assert!(result.error_count() > 0);
    }

    // ── 17. Warning for fact assertion (empty body) ──────────────────────────
    #[test]
    fn test_warning_for_fact_assertion() {
        let mut v = default_validator();
        let spec = RuleSpec {
            body_terms: vec![],
            variables: vec![],
            ..valid_spec()
        };
        let result = v.validate(&spec);
        assert!(result.is_valid());
        assert!(result.warnings.iter().any(|w| w.contains("fact assertion")));
    }

    // ── 18. registered_count increments on valid rules ───────────────────────
    #[test]
    fn test_registered_count_increments() {
        let mut v = default_validator();
        assert_eq!(v.registered_count(), 0);

        let spec1 = valid_spec();
        v.validate(&spec1);
        assert_eq!(v.registered_count(), 1);

        let spec2 = RuleSpec {
            rule_id: 2,
            head_terms: vec!["sibling(X, Y)".to_string()],
            body_terms: vec!["parent(Z, X)".to_string(), "parent(Z, Y)".to_string()],
            variables: vec!["X".to_string(), "Y".to_string(), "Z".to_string()],
            weight: 0.9,
            depends_on: vec![],
        };
        v.validate(&spec2);
        assert_eq!(v.registered_count(), 2);
    }

    // ── 19. clear_registry resets registered_count ──────────────────────────
    #[test]
    fn test_clear_registry_resets_count() {
        let mut v = default_validator();
        v.validate(&valid_spec());
        assert_eq!(v.registered_count(), 1);
        v.clear_registry();
        assert_eq!(v.registered_count(), 0);
    }

    // ── 20. Multiple errors accumulated ─────────────────────────────────────
    #[test]
    fn test_multiple_errors_accumulated() {
        let mut v = default_validator();
        // Trigger EmptyHead + CircularDependency + InvalidWeight simultaneously.
        let spec = RuleSpec {
            rule_id: 42,
            head_terms: vec![], // → EmptyHead
            body_terms: vec![],
            variables: vec![],
            weight: 0.0,          // → InvalidWeight
            depends_on: vec![42], // → CircularDependency
        };
        let result = v.validate(&spec);
        assert!(!result.is_valid());
        assert!(result.errors.contains(&ValidationError::EmptyHead));
        assert!(result
            .errors
            .contains(&ValidationError::CircularDependency { rule_id: 42 }));
        assert!(result
            .errors
            .contains(&ValidationError::InvalidWeight { weight: 0.0 }));
        assert!(result.error_count() >= 3);
    }

    // ── 21. Invalid rule does not register head ──────────────────────────────
    #[test]
    fn test_invalid_rule_not_registered() {
        let mut v = default_validator();
        let spec = RuleSpec {
            weight: 0.0, // invalid
            ..valid_spec()
        };
        v.validate(&spec);
        assert_eq!(v.registered_count(), 0);
    }

    // ── 22. Non-self circular dependency does NOT trigger error ──────────────
    #[test]
    fn test_non_self_depends_on_ok() {
        let mut v = default_validator();
        let spec = RuleSpec {
            depends_on: vec![99, 100],
            ..valid_spec()
        };
        let result = v.validate(&spec);
        assert!(result.is_valid());
    }

    // ── 23. require_positive_weight = false allows weight 0.0 ───────────────
    #[test]
    fn test_no_weight_requirement_allows_zero() {
        let config = ValidatorConfig {
            require_positive_weight: false,
            ..Default::default()
        };
        let mut v = TensorRuleValidator::new(config);
        let spec = RuleSpec {
            weight: 0.0,
            ..valid_spec()
        };
        let result = v.validate(&spec);
        assert!(result.is_valid());
    }

    // ── 24. ValidationResult rule_id matches spec rule_id ───────────────────
    #[test]
    fn test_result_rule_id_matches() {
        let mut v = default_validator();
        let spec = valid_spec();
        let result = v.validate(&spec);
        assert_eq!(result.rule_id, spec.rule_id);
    }
}
