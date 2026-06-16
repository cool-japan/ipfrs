//! TensorLogic IR types
//!
//! This module defines the Intermediate Representation types for TensorLogic
//! that can be stored and retrieved via IPFRS.

use ipfrs_core::Cid;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// A logical term in TensorLogic
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Term {
    /// Variable (e.g., ?X)
    Var(String),
    /// Constant value
    Const(Constant),
    /// Function application (e.g., f(X, Y))
    Fun(String, Vec<Term>),
    /// Reference to another term via CID
    Ref(TermRef),
}

/// Constant value types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Constant {
    /// String constant
    String(String),
    /// Integer constant
    Int(i64),
    /// Boolean constant
    Bool(bool),
    /// Floating point constant (stored as string for deterministic hashing)
    Float(String),
}

/// Reference to a term via CID
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TermRef {
    /// CID of the referenced term
    #[serde(
        serialize_with = "crate::serialize_cid",
        deserialize_with = "crate::deserialize_cid"
    )]
    pub cid: Cid,
    /// Optional hint about the term (for optimization)
    pub hint: Option<String>,
}

impl TermRef {
    /// Create a new term reference
    pub fn new(cid: Cid) -> Self {
        Self { cid, hint: None }
    }

    /// Create a term reference with a hint
    pub fn with_hint(cid: Cid, hint: String) -> Self {
        Self {
            cid,
            hint: Some(hint),
        }
    }
}

impl Term {
    /// Check if term is a variable
    #[inline]
    pub fn is_var(&self) -> bool {
        matches!(self, Term::Var(_))
    }

    /// Check if term is a constant
    #[inline]
    pub fn is_const(&self) -> bool {
        matches!(self, Term::Const(_))
    }

    /// Check if term is ground (contains no variables)
    #[inline]
    pub fn is_ground(&self) -> bool {
        match self {
            Term::Var(_) => false,
            Term::Const(_) => true,
            Term::Fun(_, args) => args.iter().all(|t| t.is_ground()),
            Term::Ref(_) => true, // References are considered ground
        }
    }

    /// Collect all variables in the term
    pub fn variables(&self) -> Vec<String> {
        let capacity = self.estimate_var_count();
        let mut vars = Vec::with_capacity(capacity);
        self.collect_vars(&mut vars);
        vars.sort_unstable();
        vars.dedup();
        vars
    }

    /// Estimate the number of unique variables (for capacity hint)
    #[inline]
    fn estimate_var_count(&self) -> usize {
        match self {
            Term::Var(_) => 1,
            Term::Const(_) | Term::Ref(_) => 0,
            Term::Fun(_, args) => args.iter().map(|t| t.estimate_var_count()).sum(),
        }
    }

    #[inline]
    fn collect_vars(&self, vars: &mut Vec<String>) {
        match self {
            Term::Var(v) => vars.push(v.clone()),
            Term::Fun(_, args) => {
                for arg in args {
                    arg.collect_vars(vars);
                }
            }
            _ => {}
        }
    }

    /// Get the complexity of the term (number of nodes)
    #[inline]
    pub fn complexity(&self) -> usize {
        match self {
            Term::Var(_) | Term::Const(_) | Term::Ref(_) => 1,
            Term::Fun(_, args) => 1 + args.iter().map(|t| t.complexity()).sum::<usize>(),
        }
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Term::Var(v) => write!(f, "?{}", v),
            Term::Const(c) => write!(f, "{}", c),
            Term::Fun(name, args) => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            Term::Ref(r) => write!(f, "@{}", r.cid),
        }
    }
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Constant::String(s) => write!(f, "\"{}\"", s),
            Constant::Int(i) => write!(f, "{}", i),
            Constant::Bool(b) => write!(f, "{}", b),
            Constant::Float(s) => write!(f, "{}", s),
        }
    }
}

/// A logical predicate
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Predicate {
    /// Predicate name
    pub name: String,
    /// Arguments
    pub args: Vec<Term>,
}

impl Predicate {
    /// Create a new predicate
    pub fn new(name: String, args: Vec<Term>) -> Self {
        Self { name, args }
    }

    /// Get the arity (number of arguments)
    #[inline]
    pub fn arity(&self) -> usize {
        self.args.len()
    }

    /// Check if predicate is ground
    #[inline]
    pub fn is_ground(&self) -> bool {
        self.args.iter().all(|t| t.is_ground())
    }

    /// Collect all variables
    pub fn variables(&self) -> Vec<String> {
        let capacity: usize = self.args.iter().map(|t| t.estimate_var_count()).sum();
        let mut vars = Vec::with_capacity(capacity);
        for arg in &self.args {
            arg.collect_vars(&mut vars);
        }
        vars.sort_unstable();
        vars.dedup();
        vars
    }
}

impl fmt::Display for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}(", self.name)?;
        for (i, arg) in self.args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", arg)?;
        }
        write!(f, ")")
    }
}

/// A logical rule (Horn clause)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    /// Head of the rule
    pub head: Predicate,
    /// Body of the rule (conjunction)
    pub body: Vec<Predicate>,
}

impl Rule {
    /// Create a new rule
    pub fn new(head: Predicate, body: Vec<Predicate>) -> Self {
        Self { head, body }
    }

    /// Create a fact (rule with empty body)
    pub fn fact(head: Predicate) -> Self {
        Self {
            head,
            body: Vec::new(),
        }
    }

    /// Check if this is a fact
    #[inline]
    pub fn is_fact(&self) -> bool {
        self.body.is_empty()
    }

    /// Collect all variables in the rule
    pub fn variables(&self) -> Vec<String> {
        let mut vars = self.head.variables();
        for pred in &self.body {
            for var in pred.variables() {
                if !vars.contains(&var) {
                    vars.push(var);
                }
            }
        }
        vars.sort_unstable();
        vars
    }
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.head)?;
        if !self.body.is_empty() {
            write!(f, " :- ")?;
            for (i, pred) in self.body.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", pred)?;
            }
        }
        write!(f, ".")
    }
}

/// A knowledge base containing facts and rules
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeBase {
    /// Facts (ground predicates)
    pub facts: Vec<Predicate>,
    /// Rules
    pub rules: Vec<Rule>,
}

impl KnowledgeBase {
    /// Create a new empty knowledge base
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a fact
    pub fn add_fact(&mut self, fact: Predicate) {
        self.facts.push(fact);
    }

    /// Add a rule
    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// Get all predicates with a given name
    #[inline]
    pub fn get_predicates(&self, name: &str) -> Vec<&Predicate> {
        self.facts.iter().filter(|p| p.name == name).collect()
    }

    /// Get all rules with a given head predicate name
    #[inline]
    pub fn get_rules(&self, name: &str) -> Vec<&Rule> {
        self.rules.iter().filter(|r| r.head.name == name).collect()
    }

    /// Get statistics
    pub fn stats(&self) -> KnowledgeBaseStats {
        KnowledgeBaseStats {
            num_facts: self.facts.len(),
            num_rules: self.rules.len(),
        }
    }

    /// Build a predicate-name → CID index from a pre-computed rule-CID map.
    ///
    /// `rule_cids` maps each rule's position in `self.rules` to its content-
    /// addressed [`Cid`].  Rules that have no entry in the map are skipped.
    ///
    /// The returned index can be used by distributed reasoners to quickly look
    /// up which peers might hold rules relevant to a given predicate.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let index = kb.index_rules_by_predicate(&cid_map);
    /// let parent_cids = index.get("parent").cloned().unwrap_or_default();
    /// ```
    pub fn index_rules_by_predicate(
        &self,
        rule_cids: &HashMap<usize, Cid>,
    ) -> HashMap<String, Vec<Cid>> {
        let mut index: HashMap<String, Vec<Cid>> = HashMap::new();
        for (idx, rule) in self.rules.iter().enumerate() {
            if let Some(cid) = rule_cids.get(&idx) {
                index.entry(rule.head.name.clone()).or_default().push(*cid);
            }
        }
        index
    }

    /// Build a predicate-name → rule-index index without CID information.
    ///
    /// Useful for local-only query planning when CIDs have not been computed yet.
    pub fn index_rules_by_predicate_local(&self) -> HashMap<String, Vec<usize>> {
        let mut index: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, rule) in self.rules.iter().enumerate() {
            index.entry(rule.head.name.clone()).or_default().push(idx);
        }
        index
    }
}

/// Knowledge base statistics
#[derive(Debug, Clone)]
pub struct KnowledgeBaseStats {
    /// Number of facts
    pub num_facts: usize,
    /// Number of rules
    pub num_rules: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_creation() {
        let var = Term::Var("X".to_string());
        assert!(var.is_var());
        assert!(!var.is_ground());

        let const_term = Term::Const(Constant::String("Alice".to_string()));
        assert!(const_term.is_const());
        assert!(const_term.is_ground());
    }

    #[test]
    fn test_predicate() {
        let pred = Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String("Alice".to_string())),
                Term::Var("X".to_string()),
            ],
        );

        assert_eq!(pred.arity(), 2);
        assert!(!pred.is_ground());
        assert_eq!(pred.variables(), vec!["X".to_string()]);
    }

    #[test]
    fn test_rule() {
        let head = Predicate::new(
            "grandparent".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Z".to_string())],
        );

        let body = vec![
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("Y".to_string()), Term::Var("Z".to_string())],
            ),
        ];

        let rule = Rule::new(head, body);
        assert!(!rule.is_fact());
        assert_eq!(
            rule.variables(),
            vec!["X".to_string(), "Y".to_string(), "Z".to_string()]
        );
    }

    #[test]
    fn test_knowledge_base() {
        let mut kb = KnowledgeBase::new();

        kb.add_fact(Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String("Alice".to_string())),
                Term::Const(Constant::String("Bob".to_string())),
            ],
        ));

        let stats = kb.stats();
        assert_eq!(stats.num_facts, 1);
        assert_eq!(stats.num_rules, 0);
    }
}
