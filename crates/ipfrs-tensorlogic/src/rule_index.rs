//! Multi-dimensional index over TensorLogic rules.
//!
//! Enables fast lookup by predicate, arity, confidence range, and dependency relationships.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// RuleArity
// ---------------------------------------------------------------------------

/// Categorizes the number of arguments (head arity) of a rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RuleArity {
    /// 0 arguments
    Nullary,
    /// 1 argument
    Unary,
    /// 2 arguments
    Binary,
    /// 3 arguments
    Ternary,
    /// 4 or more arguments
    NAry(usize),
}

impl RuleArity {
    /// Canonical numeric rank used for ordering.
    fn rank(self) -> usize {
        match self {
            Self::Nullary => 0,
            Self::Unary => 1,
            Self::Binary => 2,
            Self::Ternary => 3,
            Self::NAry(n) => n,
        }
    }
}

impl PartialOrd for RuleArity {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RuleArity {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank().cmp(&other.rank())
    }
}

// ---------------------------------------------------------------------------
// IndexedRule
// ---------------------------------------------------------------------------

/// A rule stored inside the [`TensorRuleIndex`].
#[derive(Clone, Debug)]
pub struct IndexedRule {
    /// Unique identifier for the rule.
    pub rule_id: u64,
    /// The predicate symbol at the rule head.
    pub head_predicate: String,
    /// Number of head arguments expressed as an arity category.
    pub arity: RuleArity,
    /// Confidence score in `[0.0, 1.0]`.
    pub confidence: f64,
    /// Predicate symbols that appear in the rule body.
    pub body_predicates: Vec<String>,
    /// Rule IDs that this rule explicitly depends on.
    pub depends_on: Vec<u64>,
    /// Whether the rule is currently active.
    pub active: bool,
}

// ---------------------------------------------------------------------------
// RuleQuery
// ---------------------------------------------------------------------------

/// Filter specification for [`TensorRuleIndex::query`].
#[derive(Clone, Debug, Default)]
pub struct RuleQuery {
    /// Restrict to rules whose head predicate equals this value.
    pub head_predicate: Option<String>,
    /// Restrict to rules with this arity.
    pub arity: Option<RuleArity>,
    /// Restrict to rules with confidence >= this threshold.
    pub min_confidence: Option<f64>,
    /// Restrict to rules whose body contains this predicate.
    pub body_contains: Option<String>,
    /// When `true`, inactive rules are excluded.
    pub active_only: bool,
}

// ---------------------------------------------------------------------------
// RuleIndexStats
// ---------------------------------------------------------------------------

/// Aggregate statistics about a [`TensorRuleIndex`].
#[derive(Clone, Debug, PartialEq)]
pub struct RuleIndexStats {
    /// Total number of rules (active and inactive).
    pub total_rules: usize,
    /// Number of active rules.
    pub active_rules: usize,
    /// Number of distinct head predicates.
    pub unique_predicates: usize,
    /// Mean confidence; `0.0` when there are no rules.
    pub avg_confidence: f64,
}

// ---------------------------------------------------------------------------
// TensorRuleIndex
// ---------------------------------------------------------------------------

/// Multi-dimensional index over [`IndexedRule`] entries.
///
/// Supports O(1) lookup by rule ID and predicate, plus flexible filtered
/// queries across predicate, arity, confidence, body content, and active status.
#[derive(Debug, Default)]
pub struct TensorRuleIndex {
    /// Primary store: rule_id → rule.
    pub rules: HashMap<u64, IndexedRule>,
    /// Secondary index: head_predicate → list of rule_ids.
    pub predicate_index: HashMap<String, Vec<u64>>,
}

impl TensorRuleIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a rule, updating both the primary store and the predicate index.
    ///
    /// If a rule with the same `rule_id` already exists it is replaced, and the
    /// predicate index is kept consistent.
    pub fn insert(&mut self, rule: IndexedRule) {
        // If there is an existing entry with the same id, remove its old
        // predicate-index entry first to avoid stale references.
        if let Some(old) = self.rules.get(&rule.rule_id) {
            let old_pred = old.head_predicate.clone();
            if old_pred != rule.head_predicate {
                if let Some(ids) = self.predicate_index.get_mut(&old_pred) {
                    ids.retain(|&id| id != rule.rule_id);
                    if ids.is_empty() {
                        self.predicate_index.remove(&old_pred);
                    }
                }
            }
        }

        // Update predicate index.
        self.predicate_index
            .entry(rule.head_predicate.clone())
            .or_default()
            .push(rule.rule_id);

        self.rules.insert(rule.rule_id, rule);
    }

    /// Remove the rule with the given `rule_id`.
    ///
    /// Returns `true` on success, `false` when no such rule exists.
    pub fn remove(&mut self, rule_id: u64) -> bool {
        match self.rules.remove(&rule_id) {
            None => false,
            Some(rule) => {
                if let Some(ids) = self.predicate_index.get_mut(&rule.head_predicate) {
                    ids.retain(|&id| id != rule_id);
                    if ids.is_empty() {
                        self.predicate_index.remove(&rule.head_predicate);
                    }
                }
                true
            }
        }
    }

    /// Set a rule's `active` field to `false`.
    ///
    /// Returns `true` on success, `false` when no such rule exists.
    pub fn deactivate(&mut self, rule_id: u64) -> bool {
        match self.rules.get_mut(&rule_id) {
            None => false,
            Some(rule) => {
                rule.active = false;
                true
            }
        }
    }

    /// Set a rule's `active` field to `true`.
    ///
    /// Returns `true` on success, `false` when no such rule exists.
    pub fn activate(&mut self, rule_id: u64) -> bool {
        match self.rules.get_mut(&rule_id) {
            None => false,
            Some(rule) => {
                rule.active = true;
                true
            }
        }
    }

    /// Return rules that match all criteria in `q`.
    ///
    /// Results are sorted by confidence descending; ties are broken by
    /// `rule_id` ascending.
    pub fn query(&self, q: &RuleQuery) -> Vec<&IndexedRule> {
        let mut results: Vec<&IndexedRule> = self
            .rules
            .values()
            .filter(|r| {
                if q.active_only && !r.active {
                    return false;
                }
                if let Some(ref pred) = q.head_predicate {
                    if &r.head_predicate != pred {
                        return false;
                    }
                }
                if let Some(arity) = q.arity {
                    if r.arity != arity {
                        return false;
                    }
                }
                if let Some(min_conf) = q.min_confidence {
                    if r.confidence < min_conf {
                        return false;
                    }
                }
                if let Some(ref body_pred) = q.body_contains {
                    if !r.body_predicates.iter().any(|bp| bp == body_pred) {
                        return false;
                    }
                }
                true
            })
            .collect();

        results.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.rule_id.cmp(&b.rule_id))
        });

        results
    }

    /// Return all rules whose `depends_on` list contains `rule_id`, sorted by
    /// `rule_id` ascending.
    pub fn dependents_of(&self, rule_id: u64) -> Vec<&IndexedRule> {
        let mut results: Vec<&IndexedRule> = self
            .rules
            .values()
            .filter(|r| r.depends_on.contains(&rule_id))
            .collect();
        results.sort_by_key(|r| r.rule_id);
        results
    }

    /// Return all rules with the given head predicate, sorted by confidence
    /// descending.
    ///
    /// Uses the predicate index for efficient lookup.
    pub fn rules_for_predicate(&self, predicate: &str) -> Vec<&IndexedRule> {
        let ids = match self.predicate_index.get(predicate) {
            None => return Vec::new(),
            Some(ids) => ids,
        };

        let mut results: Vec<&IndexedRule> =
            ids.iter().filter_map(|id| self.rules.get(id)).collect();

        results.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.rule_id.cmp(&b.rule_id))
        });

        results
    }

    /// Compute aggregate statistics over the current rule set.
    pub fn stats(&self) -> RuleIndexStats {
        let total_rules = self.rules.len();
        let active_rules = self.rules.values().filter(|r| r.active).count();
        let unique_predicates = self.predicate_index.len();

        let avg_confidence = if total_rules == 0 {
            0.0
        } else {
            let sum: f64 = self.rules.values().map(|r| r.confidence).sum();
            sum / total_rules as f64
        };

        RuleIndexStats {
            total_rules,
            active_rules,
            unique_predicates,
            avg_confidence,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers -----------------------------------------------------------

    fn make_rule(
        rule_id: u64,
        head_predicate: &str,
        arity: RuleArity,
        confidence: f64,
        body_predicates: Vec<&str>,
        depends_on: Vec<u64>,
        active: bool,
    ) -> IndexedRule {
        IndexedRule {
            rule_id,
            head_predicate: head_predicate.to_string(),
            arity,
            confidence,
            body_predicates: body_predicates.into_iter().map(str::to_string).collect(),
            depends_on,
            active,
        }
    }

    // ---- new() -------------------------------------------------------------

    #[test]
    fn new_starts_empty() {
        let idx = TensorRuleIndex::new();
        assert!(idx.rules.is_empty());
        assert!(idx.predicate_index.is_empty());
    }

    // ---- insert ------------------------------------------------------------

    #[test]
    fn insert_stores_rule() {
        let mut idx = TensorRuleIndex::new();
        let rule = make_rule(1, "parent", RuleArity::Binary, 0.9, vec![], vec![], true);
        idx.insert(rule);
        assert_eq!(idx.rules.len(), 1);
        assert!(idx.rules.contains_key(&1));
    }

    #[test]
    fn insert_updates_predicate_index() {
        let mut idx = TensorRuleIndex::new();
        let rule = make_rule(1, "parent", RuleArity::Binary, 0.9, vec![], vec![], true);
        idx.insert(rule);
        assert!(idx.predicate_index.contains_key("parent"));
        assert_eq!(idx.predicate_index["parent"], vec![1]);
    }

    #[test]
    fn insert_multiple_same_predicate() {
        let mut idx = TensorRuleIndex::new();
        idx.insert(make_rule(
            1,
            "parent",
            RuleArity::Binary,
            0.9,
            vec![],
            vec![],
            true,
        ));
        idx.insert(make_rule(
            2,
            "parent",
            RuleArity::Binary,
            0.7,
            vec![],
            vec![],
            true,
        ));
        let ids = &idx.predicate_index["parent"];
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }

    // ---- remove ------------------------------------------------------------

    #[test]
    fn remove_removes_from_both_stores() {
        let mut idx = TensorRuleIndex::new();
        idx.insert(make_rule(
            1,
            "parent",
            RuleArity::Binary,
            0.9,
            vec![],
            vec![],
            true,
        ));
        let removed = idx.remove(1);
        assert!(removed);
        assert!(!idx.rules.contains_key(&1));
        assert!(!idx.predicate_index.contains_key("parent"));
    }

    #[test]
    fn remove_returns_false_for_unknown() {
        let mut idx = TensorRuleIndex::new();
        assert!(!idx.remove(999));
    }

    #[test]
    fn remove_keeps_other_rules_with_same_predicate() {
        let mut idx = TensorRuleIndex::new();
        idx.insert(make_rule(
            1,
            "parent",
            RuleArity::Binary,
            0.9,
            vec![],
            vec![],
            true,
        ));
        idx.insert(make_rule(
            2,
            "parent",
            RuleArity::Binary,
            0.7,
            vec![],
            vec![],
            true,
        ));
        idx.remove(1);
        assert!(idx.predicate_index.contains_key("parent"));
        assert_eq!(idx.predicate_index["parent"], vec![2]);
    }

    // ---- deactivate / activate ---------------------------------------------

    #[test]
    fn deactivate_sets_active_false() {
        let mut idx = TensorRuleIndex::new();
        idx.insert(make_rule(
            1,
            "parent",
            RuleArity::Binary,
            0.9,
            vec![],
            vec![],
            true,
        ));
        assert!(idx.deactivate(1));
        assert!(!idx.rules[&1].active);
    }

    #[test]
    fn activate_sets_active_true() {
        let mut idx = TensorRuleIndex::new();
        idx.insert(make_rule(
            1,
            "parent",
            RuleArity::Binary,
            0.9,
            vec![],
            vec![],
            false,
        ));
        assert!(idx.activate(1));
        assert!(idx.rules[&1].active);
    }

    #[test]
    fn deactivate_returns_false_for_unknown() {
        let mut idx = TensorRuleIndex::new();
        assert!(!idx.deactivate(42));
    }

    #[test]
    fn activate_returns_false_for_unknown() {
        let mut idx = TensorRuleIndex::new();
        assert!(!idx.activate(42));
    }

    // ---- query: head_predicate filter --------------------------------------

    #[test]
    fn query_head_predicate_filter() {
        let mut idx = TensorRuleIndex::new();
        idx.insert(make_rule(
            1,
            "parent",
            RuleArity::Binary,
            0.9,
            vec![],
            vec![],
            true,
        ));
        idx.insert(make_rule(
            2,
            "ancestor",
            RuleArity::Binary,
            0.8,
            vec![],
            vec![],
            true,
        ));

        let q = RuleQuery {
            head_predicate: Some("parent".to_string()),
            ..Default::default()
        };
        let results = idx.query(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].rule_id, 1);
    }

    // ---- query: arity filter -----------------------------------------------

    #[test]
    fn query_arity_filter() {
        let mut idx = TensorRuleIndex::new();
        idx.insert(make_rule(
            1,
            "a",
            RuleArity::Unary,
            0.9,
            vec![],
            vec![],
            true,
        ));
        idx.insert(make_rule(
            2,
            "b",
            RuleArity::Binary,
            0.8,
            vec![],
            vec![],
            true,
        ));

        let q = RuleQuery {
            arity: Some(RuleArity::Unary),
            ..Default::default()
        };
        let results = idx.query(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].rule_id, 1);
    }

    // ---- query: min_confidence filter --------------------------------------

    #[test]
    fn query_min_confidence_filter() {
        let mut idx = TensorRuleIndex::new();
        idx.insert(make_rule(
            1,
            "a",
            RuleArity::Nullary,
            0.9,
            vec![],
            vec![],
            true,
        ));
        idx.insert(make_rule(
            2,
            "b",
            RuleArity::Nullary,
            0.3,
            vec![],
            vec![],
            true,
        ));

        let q = RuleQuery {
            min_confidence: Some(0.5),
            ..Default::default()
        };
        let results = idx.query(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].rule_id, 1);
    }

    // ---- query: body_contains filter ---------------------------------------

    #[test]
    fn query_body_contains_filter() {
        let mut idx = TensorRuleIndex::new();
        idx.insert(make_rule(
            1,
            "a",
            RuleArity::Unary,
            0.9,
            vec!["child", "parent"],
            vec![],
            true,
        ));
        idx.insert(make_rule(
            2,
            "b",
            RuleArity::Unary,
            0.8,
            vec!["sibling"],
            vec![],
            true,
        ));

        let q = RuleQuery {
            body_contains: Some("parent".to_string()),
            ..Default::default()
        };
        let results = idx.query(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].rule_id, 1);
    }

    // ---- query: active_only filter -----------------------------------------

    #[test]
    fn query_active_only_filter() {
        let mut idx = TensorRuleIndex::new();
        idx.insert(make_rule(
            1,
            "a",
            RuleArity::Nullary,
            0.9,
            vec![],
            vec![],
            true,
        ));
        idx.insert(make_rule(
            2,
            "b",
            RuleArity::Nullary,
            0.8,
            vec![],
            vec![],
            false,
        ));

        let q = RuleQuery {
            active_only: true,
            ..Default::default()
        };
        let results = idx.query(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].rule_id, 1);
    }

    // ---- query: multiple filters ANDed ------------------------------------

    #[test]
    fn query_multiple_filters_anded() {
        let mut idx = TensorRuleIndex::new();
        idx.insert(make_rule(
            1,
            "parent",
            RuleArity::Binary,
            0.9,
            vec!["child"],
            vec![],
            true,
        ));
        idx.insert(make_rule(
            2,
            "parent",
            RuleArity::Binary,
            0.4,
            vec!["child"],
            vec![],
            true,
        ));
        idx.insert(make_rule(
            3,
            "parent",
            RuleArity::Unary,
            0.9,
            vec!["child"],
            vec![],
            true,
        ));

        let q = RuleQuery {
            head_predicate: Some("parent".to_string()),
            arity: Some(RuleArity::Binary),
            min_confidence: Some(0.8),
            body_contains: Some("child".to_string()),
            active_only: true,
        };
        let results = idx.query(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].rule_id, 1);
    }

    // ---- query: sorted by confidence desc ----------------------------------

    #[test]
    fn query_sorted_by_confidence_desc() {
        let mut idx = TensorRuleIndex::new();
        idx.insert(make_rule(
            1,
            "a",
            RuleArity::Nullary,
            0.3,
            vec![],
            vec![],
            true,
        ));
        idx.insert(make_rule(
            2,
            "a",
            RuleArity::Nullary,
            0.9,
            vec![],
            vec![],
            true,
        ));
        idx.insert(make_rule(
            3,
            "a",
            RuleArity::Nullary,
            0.6,
            vec![],
            vec![],
            true,
        ));

        let q = RuleQuery::default();
        let results = idx.query(&q);
        assert_eq!(results[0].confidence, 0.9);
        assert_eq!(results[1].confidence, 0.6);
        assert_eq!(results[2].confidence, 0.3);
    }

    // ---- query: confidence tiebreak by rule_id asc ------------------------

    #[test]
    fn query_confidence_tiebreak_by_rule_id_asc() {
        let mut idx = TensorRuleIndex::new();
        idx.insert(make_rule(
            3,
            "a",
            RuleArity::Nullary,
            0.7,
            vec![],
            vec![],
            true,
        ));
        idx.insert(make_rule(
            1,
            "a",
            RuleArity::Nullary,
            0.7,
            vec![],
            vec![],
            true,
        ));
        idx.insert(make_rule(
            2,
            "a",
            RuleArity::Nullary,
            0.7,
            vec![],
            vec![],
            true,
        ));

        let q = RuleQuery::default();
        let results = idx.query(&q);
        let ids: Vec<u64> = results.iter().map(|r| r.rule_id).collect();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    // ---- dependents_of -----------------------------------------------------

    #[test]
    fn dependents_of_returns_correct_rules() {
        let mut idx = TensorRuleIndex::new();
        idx.insert(make_rule(
            1,
            "base",
            RuleArity::Nullary,
            0.9,
            vec![],
            vec![],
            true,
        ));
        idx.insert(make_rule(
            2,
            "derived",
            RuleArity::Unary,
            0.8,
            vec![],
            vec![1],
            true,
        ));
        idx.insert(make_rule(
            3,
            "derived2",
            RuleArity::Unary,
            0.7,
            vec![],
            vec![1, 2],
            true,
        ));
        idx.insert(make_rule(
            4,
            "unrelated",
            RuleArity::Nullary,
            0.6,
            vec![],
            vec![],
            true,
        ));

        let deps = idx.dependents_of(1);
        let ids: Vec<u64> = deps.iter().map(|r| r.rule_id).collect();
        assert_eq!(ids, vec![2, 3]);
    }

    #[test]
    fn dependents_of_empty_when_none_depend() {
        let mut idx = TensorRuleIndex::new();
        idx.insert(make_rule(
            1,
            "a",
            RuleArity::Nullary,
            0.9,
            vec![],
            vec![],
            true,
        ));
        idx.insert(make_rule(
            2,
            "b",
            RuleArity::Nullary,
            0.8,
            vec![],
            vec![],
            true,
        ));

        let deps = idx.dependents_of(1);
        assert!(deps.is_empty());
    }

    // ---- rules_for_predicate -----------------------------------------------

    #[test]
    fn rules_for_predicate_uses_index() {
        let mut idx = TensorRuleIndex::new();
        idx.insert(make_rule(
            1,
            "parent",
            RuleArity::Binary,
            0.9,
            vec![],
            vec![],
            true,
        ));
        idx.insert(make_rule(
            2,
            "child",
            RuleArity::Unary,
            0.8,
            vec![],
            vec![],
            true,
        ));

        let results = idx.rules_for_predicate("parent");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].rule_id, 1);
    }

    #[test]
    fn rules_for_predicate_sorted_by_confidence() {
        let mut idx = TensorRuleIndex::new();
        idx.insert(make_rule(
            1,
            "parent",
            RuleArity::Binary,
            0.4,
            vec![],
            vec![],
            true,
        ));
        idx.insert(make_rule(
            2,
            "parent",
            RuleArity::Binary,
            0.9,
            vec![],
            vec![],
            true,
        ));
        idx.insert(make_rule(
            3,
            "parent",
            RuleArity::Binary,
            0.6,
            vec![],
            vec![],
            true,
        ));

        let results = idx.rules_for_predicate("parent");
        assert_eq!(results[0].confidence, 0.9);
        assert_eq!(results[1].confidence, 0.6);
        assert_eq!(results[2].confidence, 0.4);
    }

    #[test]
    fn rules_for_predicate_returns_empty_for_unknown() {
        let idx = TensorRuleIndex::new();
        assert!(idx.rules_for_predicate("nonexistent").is_empty());
    }

    // ---- stats -------------------------------------------------------------

    #[test]
    fn stats_total_and_active_rules() {
        let mut idx = TensorRuleIndex::new();
        idx.insert(make_rule(
            1,
            "a",
            RuleArity::Nullary,
            0.9,
            vec![],
            vec![],
            true,
        ));
        idx.insert(make_rule(
            2,
            "b",
            RuleArity::Nullary,
            0.8,
            vec![],
            vec![],
            false,
        ));
        idx.insert(make_rule(
            3,
            "c",
            RuleArity::Nullary,
            0.7,
            vec![],
            vec![],
            true,
        ));

        let s = idx.stats();
        assert_eq!(s.total_rules, 3);
        assert_eq!(s.active_rules, 2);
    }

    #[test]
    fn stats_unique_predicates() {
        let mut idx = TensorRuleIndex::new();
        idx.insert(make_rule(
            1,
            "parent",
            RuleArity::Binary,
            0.9,
            vec![],
            vec![],
            true,
        ));
        idx.insert(make_rule(
            2,
            "parent",
            RuleArity::Binary,
            0.7,
            vec![],
            vec![],
            true,
        ));
        idx.insert(make_rule(
            3,
            "child",
            RuleArity::Unary,
            0.8,
            vec![],
            vec![],
            true,
        ));

        let s = idx.stats();
        assert_eq!(s.unique_predicates, 2);
    }

    #[test]
    fn stats_avg_confidence() {
        let mut idx = TensorRuleIndex::new();
        idx.insert(make_rule(
            1,
            "a",
            RuleArity::Nullary,
            0.8,
            vec![],
            vec![],
            true,
        ));
        idx.insert(make_rule(
            2,
            "b",
            RuleArity::Nullary,
            0.6,
            vec![],
            vec![],
            true,
        ));

        let s = idx.stats();
        let expected = (0.8 + 0.6) / 2.0;
        assert!((s.avg_confidence - expected).abs() < 1e-10);
    }

    #[test]
    fn stats_avg_confidence_zero_when_empty() {
        let idx = TensorRuleIndex::new();
        assert_eq!(idx.stats().avg_confidence, 0.0);
    }

    // ---- RuleArity ordering -----------------------------------------------

    #[test]
    fn rule_arity_ordering() {
        assert!(RuleArity::Nullary < RuleArity::Unary);
        assert!(RuleArity::Unary < RuleArity::Binary);
        assert!(RuleArity::Binary < RuleArity::Ternary);
        assert!(RuleArity::Ternary < RuleArity::NAry(4));
        assert!(RuleArity::NAry(4) < RuleArity::NAry(10));
        assert_eq!(RuleArity::Nullary, RuleArity::Nullary);
    }
}
