//! Utility functions for common TensorLogic operations
//!
//! This module provides helper functions that make it easier to work with
//! TensorLogic predicates, terms, and knowledge bases.

use crate::ir::{Constant, KnowledgeBase, Predicate, Rule, Term};
use crate::reasoning::{InferenceEngine, Substitution};
use ipfrs_core::error::Result;
use std::collections::HashMap;

/// Builder for creating predicates more easily
pub struct PredicateBuilder {
    name: String,
    args: Vec<Term>,
}

impl PredicateBuilder {
    /// Create a new predicate builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            args: Vec::new(),
        }
    }

    /// Add a constant string argument
    pub fn arg_str(mut self, value: impl Into<String>) -> Self {
        self.args.push(Term::Const(Constant::String(value.into())));
        self
    }

    /// Add a constant integer argument
    pub fn arg_int(mut self, value: i64) -> Self {
        self.args.push(Term::Const(Constant::Int(value)));
        self
    }

    /// Add a constant boolean argument
    pub fn arg_bool(mut self, value: bool) -> Self {
        self.args.push(Term::Const(Constant::Bool(value)));
        self
    }

    /// Add a variable argument
    pub fn arg_var(mut self, name: impl Into<String>) -> Self {
        self.args.push(Term::Var(name.into()));
        self
    }

    /// Add any term as an argument
    pub fn arg(mut self, term: Term) -> Self {
        self.args.push(term);
        self
    }

    /// Build the predicate
    pub fn build(self) -> Predicate {
        Predicate::new(self.name, self.args)
    }
}

/// Builder for creating rules more easily
pub struct RuleBuilder {
    head: Option<Predicate>,
    body: Vec<Predicate>,
}

impl RuleBuilder {
    /// Create a new rule builder
    pub fn new() -> Self {
        Self {
            head: None,
            body: Vec::new(),
        }
    }

    /// Set the rule head
    pub fn head(mut self, predicate: Predicate) -> Self {
        self.head = Some(predicate);
        self
    }

    /// Add a body predicate
    pub fn body(mut self, predicate: Predicate) -> Self {
        self.body.push(predicate);
        self
    }

    /// Add multiple body predicates
    pub fn bodies(mut self, predicates: Vec<Predicate>) -> Self {
        self.body.extend(predicates);
        self
    }

    /// Build the rule
    pub fn build(self) -> Rule {
        Rule::new(self.head.expect("Rule head must be set"), self.body)
    }
}

impl Default for RuleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for knowledge base operations
pub struct KnowledgeBaseUtils;

impl KnowledgeBaseUtils {
    /// Create a knowledge base from a list of facts
    pub fn from_facts(facts: Vec<Predicate>) -> KnowledgeBase {
        let mut kb = KnowledgeBase::new();
        for fact in facts {
            kb.add_fact(fact);
        }
        kb
    }

    /// Merge two knowledge bases
    pub fn merge(kb1: &KnowledgeBase, kb2: &KnowledgeBase) -> KnowledgeBase {
        let mut merged = kb1.clone();
        for fact in &kb2.facts {
            if !merged.facts.contains(fact) {
                merged.add_fact(fact.clone());
            }
        }
        for rule in &kb2.rules {
            merged.add_rule(rule.clone());
        }
        merged
    }

    /// Filter facts by predicate name
    pub fn filter_facts(kb: &KnowledgeBase, predicate_name: &str) -> Vec<Predicate> {
        kb.facts
            .iter()
            .filter(|p| p.name == predicate_name)
            .cloned()
            .collect()
    }

    /// Count predicates by name
    pub fn count_predicates(kb: &KnowledgeBase) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for fact in &kb.facts {
            *counts.entry(fact.name.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Get all unique predicate names in the knowledge base
    pub fn predicate_names(kb: &KnowledgeBase) -> Vec<String> {
        let mut names: Vec<String> = kb
            .facts
            .iter()
            .map(|p| p.name.clone())
            .chain(kb.rules.iter().map(|r| r.head.name.clone()))
            .collect();
        names.sort_unstable();
        names.dedup();
        names
    }

    /// Check if a fact exists in the knowledge base
    pub fn contains_fact(kb: &KnowledgeBase, fact: &Predicate) -> bool {
        kb.facts.contains(fact)
    }

    /// Remove duplicate facts from a knowledge base
    pub fn deduplicate(kb: &mut KnowledgeBase) {
        kb.facts.sort_by(|a, b| {
            a.name
                .cmp(&b.name)
                .then_with(|| a.args.len().cmp(&b.args.len()))
        });
        kb.facts.dedup();
    }
}

/// Utility functions for query operations
pub struct QueryUtils;

impl QueryUtils {
    /// Execute a simple query and return only the first solution
    pub fn query_one(predicate: &Predicate, kb: &KnowledgeBase) -> Result<Option<Substitution>> {
        let engine = InferenceEngine::new();
        let solutions = engine.query(predicate, kb)?;
        Ok(solutions.into_iter().next())
    }

    /// Execute a query and extract values for a specific variable
    pub fn query_var(
        predicate: &Predicate,
        kb: &KnowledgeBase,
        var_name: &str,
    ) -> Result<Vec<Term>> {
        let engine = InferenceEngine::new();
        let solutions = engine.query(predicate, kb)?;
        Ok(solutions
            .into_iter()
            .filter_map(|subst| subst.get(var_name).cloned())
            .collect())
    }

    /// Check if a goal is provable
    pub fn is_provable(predicate: &Predicate, kb: &KnowledgeBase) -> Result<bool> {
        let engine = InferenceEngine::new();
        let solutions = engine.query(predicate, kb)?;
        Ok(!solutions.is_empty())
    }

    /// Count the number of solutions for a query
    pub fn count_solutions(predicate: &Predicate, kb: &KnowledgeBase) -> Result<usize> {
        let engine = InferenceEngine::new();
        let solutions = engine.query(predicate, kb)?;
        Ok(solutions.len())
    }
}

/// Utility functions for term manipulation
pub struct TermUtils;

impl TermUtils {
    /// Create a constant string term
    pub fn string(value: impl Into<String>) -> Term {
        Term::Const(Constant::String(value.into()))
    }

    /// Create a constant integer term
    pub fn int(value: i64) -> Term {
        Term::Const(Constant::Int(value))
    }

    /// Create a constant boolean term
    pub fn bool(value: bool) -> Term {
        Term::Const(Constant::Bool(value))
    }

    /// Create a variable term
    pub fn var(name: impl Into<String>) -> Term {
        Term::Var(name.into())
    }

    /// Extract string value from a constant term
    pub fn as_string(term: &Term) -> Option<&str> {
        match term {
            Term::Const(Constant::String(s)) => Some(s),
            _ => None,
        }
    }

    /// Extract integer value from a constant term
    pub fn as_int(term: &Term) -> Option<i64> {
        match term {
            Term::Const(Constant::Int(i)) => Some(*i),
            _ => None,
        }
    }

    /// Extract boolean value from a constant term
    pub fn as_bool(term: &Term) -> Option<bool> {
        match term {
            Term::Const(Constant::Bool(b)) => Some(*b),
            _ => None,
        }
    }

    /// Check if term is ground (contains no variables)
    pub fn is_ground(term: &Term) -> bool {
        term.is_ground()
    }

    /// Get all variables in a term
    pub fn variables(term: &Term) -> Vec<String> {
        term.variables()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predicate_builder() {
        let pred = PredicateBuilder::new("parent")
            .arg_str("alice")
            .arg_str("bob")
            .build();

        assert_eq!(pred.name, "parent");
        assert_eq!(pred.args.len(), 2);
        assert!(pred.is_ground());
    }

    #[test]
    fn test_predicate_builder_with_vars() {
        let pred = PredicateBuilder::new("parent")
            .arg_str("alice")
            .arg_var("X")
            .build();

        assert_eq!(pred.name, "parent");
        assert_eq!(pred.args.len(), 2);
        assert!(!pred.is_ground());
    }

    #[test]
    fn test_rule_builder() {
        let head = PredicateBuilder::new("grandparent")
            .arg_var("X")
            .arg_var("Z")
            .build();

        let body1 = PredicateBuilder::new("parent")
            .arg_var("X")
            .arg_var("Y")
            .build();

        let body2 = PredicateBuilder::new("parent")
            .arg_var("Y")
            .arg_var("Z")
            .build();

        let rule = RuleBuilder::new()
            .head(head)
            .body(body1)
            .body(body2)
            .build();

        assert_eq!(rule.head.name, "grandparent");
        assert_eq!(rule.body.len(), 2);
    }

    #[test]
    fn test_kb_from_facts() {
        let facts = vec![
            PredicateBuilder::new("parent")
                .arg_str("alice")
                .arg_str("bob")
                .build(),
            PredicateBuilder::new("parent")
                .arg_str("bob")
                .arg_str("charlie")
                .build(),
        ];

        let kb = KnowledgeBaseUtils::from_facts(facts);
        assert_eq!(kb.facts.len(), 2);
    }

    #[test]
    fn test_kb_merge() {
        let mut kb1 = KnowledgeBase::new();
        kb1.add_fact(PredicateBuilder::new("test").arg_str("a").build());

        let mut kb2 = KnowledgeBase::new();
        kb2.add_fact(PredicateBuilder::new("test").arg_str("b").build());

        let merged = KnowledgeBaseUtils::merge(&kb1, &kb2);
        assert_eq!(merged.facts.len(), 2);
    }

    #[test]
    fn test_filter_facts() {
        let mut kb = KnowledgeBase::new();
        kb.add_fact(
            PredicateBuilder::new("parent")
                .arg_str("a")
                .arg_str("b")
                .build(),
        );
        kb.add_fact(
            PredicateBuilder::new("parent")
                .arg_str("b")
                .arg_str("c")
                .build(),
        );
        kb.add_fact(
            PredicateBuilder::new("knows")
                .arg_str("a")
                .arg_str("c")
                .build(),
        );

        let parents = KnowledgeBaseUtils::filter_facts(&kb, "parent");
        assert_eq!(parents.len(), 2);

        let knows = KnowledgeBaseUtils::filter_facts(&kb, "knows");
        assert_eq!(knows.len(), 1);
    }

    #[test]
    fn test_count_predicates() {
        let mut kb = KnowledgeBase::new();
        kb.add_fact(
            PredicateBuilder::new("parent")
                .arg_str("a")
                .arg_str("b")
                .build(),
        );
        kb.add_fact(
            PredicateBuilder::new("parent")
                .arg_str("b")
                .arg_str("c")
                .build(),
        );
        kb.add_fact(
            PredicateBuilder::new("knows")
                .arg_str("a")
                .arg_str("c")
                .build(),
        );

        let counts = KnowledgeBaseUtils::count_predicates(&kb);
        assert_eq!(counts.get("parent"), Some(&2));
        assert_eq!(counts.get("knows"), Some(&1));
    }

    #[test]
    fn test_term_utils() {
        let str_term = TermUtils::string("alice");
        assert_eq!(TermUtils::as_string(&str_term), Some("alice"));

        let int_term = TermUtils::int(42);
        assert_eq!(TermUtils::as_int(&int_term), Some(42));

        let bool_term = TermUtils::bool(true);
        assert_eq!(TermUtils::as_bool(&bool_term), Some(true));

        let var_term = TermUtils::var("X");
        assert!(!TermUtils::is_ground(&var_term));
    }

    #[test]
    fn test_query_utils() {
        let mut kb = KnowledgeBase::new();
        kb.add_fact(
            PredicateBuilder::new("parent")
                .arg_str("alice")
                .arg_str("bob")
                .build(),
        );

        let query = PredicateBuilder::new("parent")
            .arg_str("alice")
            .arg_var("X")
            .build();

        let is_provable = QueryUtils::is_provable(&query, &kb).expect("test: should succeed");
        assert!(is_provable);

        let count = QueryUtils::count_solutions(&query, &kb).expect("test: should succeed");
        assert_eq!(count, 1);
    }
}
