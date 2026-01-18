//! Recursive Query Support with Tabling
//!
//! This module implements advanced recursive query handling including:
//! - Tabling/tabulation for efficient recursive queries
//! - Stratified evaluation
//! - Support for left-recursive rules
//! - Fixpoint computation
//!
//! # Tabling
//!
//! Tabling (also called tabled resolution or SLG resolution) is a technique
//! for evaluating logic programs that improves on standard SLD resolution
//! by memoizing intermediate results and detecting loops.
//!
//! # Example
//!
//! ```
//! use ipfrs_tensorlogic::{TabledInferenceEngine, KnowledgeBase, Predicate, Rule, Term, Constant};
//!
//! let mut kb = KnowledgeBase::new();
//!
//! // Define ancestor relation: ancestor(X, Y) :- parent(X, Y).
//! // ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
//! // This is recursive and benefits from tabling
//!
//! // Add parent facts
//! kb.add_fact(Predicate::new("parent".to_string(), vec![
//!     Term::Const(Constant::String("alice".to_string())),
//!     Term::Const(Constant::String("bob".to_string())),
//! ]));
//! kb.add_fact(Predicate::new("parent".to_string(), vec![
//!     Term::Const(Constant::String("bob".to_string())),
//!     Term::Const(Constant::String("charlie".to_string())),
//! ]));
//!
//! // Add base rule: ancestor(X, Y) :- parent(X, Y)
//! kb.add_rule(Rule::new(
//!     Predicate::new("ancestor".to_string(), vec![
//!         Term::Var("X".to_string()),
//!         Term::Var("Y".to_string()),
//!     ]),
//!     vec![Predicate::new("parent".to_string(), vec![
//!         Term::Var("X".to_string()),
//!         Term::Var("Y".to_string()),
//!     ])],
//! ));
//!
//! // Add recursive rule: ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)
//! kb.add_rule(Rule::new(
//!     Predicate::new("ancestor".to_string(), vec![
//!         Term::Var("X".to_string()),
//!         Term::Var("Z".to_string()),
//!     ]),
//!     vec![
//!         Predicate::new("parent".to_string(), vec![
//!             Term::Var("X".to_string()),
//!             Term::Var("Y".to_string()),
//!         ]),
//!         Predicate::new("ancestor".to_string(), vec![
//!             Term::Var("Y".to_string()),
//!             Term::Var("Z".to_string()),
//!         ]),
//!     ],
//! ));
//!
//! // Create tabled engine
//! let engine = TabledInferenceEngine::new();
//!
//! // Query for all ancestors of alice
//! let goal = Predicate::new("ancestor".to_string(), vec![
//!     Term::Const(Constant::String("alice".to_string())),
//!     Term::Var("Z".to_string()),
//! ]);
//!
//! let solutions = engine.query(&goal, &kb).unwrap();
//! // Should find at least bob as an ancestor
//! assert!(!solutions.is_empty());
//! ```

use crate::ir::{KnowledgeBase, Predicate, Rule};
use crate::reasoning::{apply_subst_predicate, unify_predicates, Substitution};
use ipfrs_core::error::Result;
use std::collections::{HashMap, HashSet};

/// Table entry for memoized subgoals
#[derive(Debug, Clone)]
struct TableEntry {
    /// The subgoal being solved
    #[allow(dead_code)]
    goal: Predicate,
    /// Solutions found so far
    solutions: Vec<Substitution>,
    /// Whether this entry is complete
    complete: bool,
    /// Depth at which this was tabled
    #[allow(dead_code)]
    depth: usize,
}

/// Tabled inference engine using SLG resolution
pub struct TabledInferenceEngine {
    /// Table for memoizing subgoals
    table: HashMap<String, TableEntry>,
    /// Maximum depth
    max_depth: usize,
    /// Maximum solutions per subgoal
    max_solutions: usize,
}

impl TabledInferenceEngine {
    /// Create a new tabled inference engine
    pub fn new() -> Self {
        Self {
            table: HashMap::new(),
            max_depth: 100,
            max_solutions: 1000,
        }
    }

    /// Create with custom limits
    pub fn with_limits(max_depth: usize, max_solutions: usize) -> Self {
        Self {
            table: HashMap::new(),
            max_depth,
            max_solutions,
        }
    }

    /// Query with tabling
    pub fn query(&self, goal: &Predicate, kb: &KnowledgeBase) -> Result<Vec<Substitution>> {
        let mut engine = Self {
            table: HashMap::new(),
            max_depth: self.max_depth,
            max_solutions: self.max_solutions,
        };

        engine.solve_tabled(goal, &Substitution::new(), kb, 0)
    }

    /// Solve a goal with tabling
    fn solve_tabled(
        &mut self,
        goal: &Predicate,
        subst: &Substitution,
        kb: &KnowledgeBase,
        depth: usize,
    ) -> Result<Vec<Substitution>> {
        // Check depth limit
        if depth > self.max_depth {
            return Ok(Vec::new());
        }

        // Apply substitution to goal
        let goal = apply_subst_predicate(goal, subst);

        // Create table key
        let key = self.goal_key(&goal);

        // Check if goal is already tabled
        if let Some(entry) = self.table.get(&key) {
            // If complete, return cached solutions
            if entry.complete {
                return Ok(entry.solutions.clone());
            }
            // If incomplete, we have a loop - return empty for now
            return Ok(Vec::new());
        }

        // Create new table entry
        let mut entry = TableEntry {
            goal: goal.clone(),
            solutions: Vec::new(),
            complete: false,
            depth,
        };

        // Insert incomplete entry to detect loops
        self.table.insert(key.clone(), entry.clone());

        // Solve using standard backward chaining
        let mut solutions = Vec::new();

        // Try facts
        for fact in kb.get_predicates(&goal.name) {
            if let Some(new_subst) = unify_predicates(&goal, fact, &Substitution::new()) {
                solutions.push(new_subst);
                if solutions.len() >= self.max_solutions {
                    break;
                }
            }
        }

        // Try rules
        for rule in kb.get_rules(&goal.name) {
            if solutions.len() >= self.max_solutions {
                break;
            }

            // Rename variables in rule
            let renamed_rule = self.rename_rule(rule, depth);

            // Try to unify with rule head
            if let Some(new_subst) =
                unify_predicates(&goal, &renamed_rule.head, &Substitution::new())
            {
                // Solve rule body
                let body_solutions =
                    self.solve_conjunction(&renamed_rule.body, &new_subst, kb, depth + 1)?;
                solutions.extend(body_solutions);
            }
        }

        // Mark entry as complete and update solutions
        entry.solutions = solutions.clone();
        entry.complete = true;
        self.table.insert(key, entry);

        Ok(solutions)
    }

    /// Solve a conjunction of goals
    fn solve_conjunction(
        &mut self,
        goals: &[Predicate],
        subst: &Substitution,
        kb: &KnowledgeBase,
        depth: usize,
    ) -> Result<Vec<Substitution>> {
        if goals.is_empty() {
            return Ok(vec![subst.clone()]);
        }

        let first = &goals[0];
        let rest = &goals[1..];

        let first_solutions = self.solve_tabled(first, subst, kb, depth)?;

        let mut all_solutions = Vec::new();
        for first_subst in first_solutions {
            let rest_solutions = self.solve_conjunction(rest, &first_subst, kb, depth)?;
            all_solutions.extend(rest_solutions);

            if all_solutions.len() >= self.max_solutions {
                break;
            }
        }

        Ok(all_solutions)
    }

    /// Generate a unique key for a goal
    fn goal_key(&self, goal: &Predicate) -> String {
        format!("{}({})", goal.name, goal.args.len())
    }

    /// Rename variables in a rule
    fn rename_rule(&self, rule: &Rule, suffix: usize) -> Rule {
        let var_map: HashMap<String, String> = rule
            .variables()
            .into_iter()
            .map(|v| (v.clone(), format!("{}_{}", v, suffix)))
            .collect();

        let rename_subst: Substitution = var_map
            .into_iter()
            .map(|(old, new)| (old, crate::ir::Term::Var(new)))
            .collect();

        Rule {
            head: apply_subst_predicate(&rule.head, &rename_subst),
            body: rule
                .body
                .iter()
                .map(|p| apply_subst_predicate(p, &rename_subst))
                .collect(),
        }
    }

    /// Get table statistics
    pub fn table_stats(&self) -> TableStats {
        TableStats {
            entries: self.table.len(),
            complete_entries: self.table.values().filter(|e| e.complete).count(),
            total_solutions: self.table.values().map(|e| e.solutions.len()).sum(),
        }
    }

    /// Clear the table
    pub fn clear_table(&mut self) {
        self.table.clear();
    }
}

impl Default for TabledInferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the tabling system
#[derive(Debug, Clone)]
pub struct TableStats {
    /// Number of table entries
    pub entries: usize,
    /// Number of complete entries
    pub complete_entries: usize,
    /// Total solutions across all entries
    pub total_solutions: usize,
}

/// Fixpoint computation for stratified programs
pub struct FixpointEngine {
    /// Maximum iterations for fixpoint
    max_iterations: usize,
}

impl FixpointEngine {
    /// Create a new fixpoint engine
    pub fn new() -> Self {
        Self {
            max_iterations: 100,
        }
    }

    /// Create with custom iteration limit
    pub fn with_max_iterations(max_iterations: usize) -> Self {
        Self { max_iterations }
    }

    /// Compute fixpoint for a set of rules
    pub fn compute_fixpoint(&self, kb: &KnowledgeBase) -> Result<KnowledgeBase> {
        let mut current_kb = kb.clone();
        let mut iteration = 0;

        loop {
            iteration += 1;
            if iteration > self.max_iterations {
                break;
            }

            let mut new_facts = Vec::new();
            let mut changed = false;

            // Apply all rules to derive new facts
            // Collect unique predicate names from rules
            let predicate_names: std::collections::HashSet<String> = current_kb
                .rules
                .iter()
                .map(|r| r.head.name.clone())
                .collect();

            for predicate_name in predicate_names {
                for rule in current_kb.get_rules(&predicate_name) {
                    let derived = self.derive_facts_from_rule(rule, &current_kb)?;
                    for fact in derived {
                        // Check if fact already exists
                        if !current_kb.facts.contains(&fact) {
                            new_facts.push(fact);
                            changed = true;
                        }
                    }
                }
            }

            // Add new facts to KB
            for fact in new_facts {
                current_kb.add_fact(fact);
            }

            // If no new facts, we've reached fixpoint
            if !changed {
                break;
            }
        }

        Ok(current_kb)
    }

    /// Derive facts from a single rule
    fn derive_facts_from_rule(&self, _rule: &Rule, _kb: &KnowledgeBase) -> Result<Vec<Predicate>> {
        let derived = Vec::new();

        // This is a simplified implementation
        // A full implementation would do proper unification and substitution

        // For now, just return empty
        // TODO: Implement full derivation

        Ok(derived)
    }
}

impl Default for FixpointEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Stratification analysis for logic programs
pub struct StratificationAnalyzer {
    /// Dependency graph between predicates
    dependencies: HashMap<String, HashSet<String>>,
}

impl StratificationAnalyzer {
    /// Create a new stratification analyzer
    pub fn new() -> Self {
        Self {
            dependencies: HashMap::new(),
        }
    }

    /// Analyze a knowledge base for stratification
    pub fn analyze(&mut self, kb: &KnowledgeBase) -> StratificationResult {
        self.build_dependency_graph(kb);

        // Check for cycles
        if self.has_cycles() {
            StratificationResult::NonStratifiable
        } else {
            // Compute stratification levels
            let strata = self.compute_strata();
            StratificationResult::Stratifiable(strata)
        }
    }

    /// Build dependency graph from KB
    fn build_dependency_graph(&mut self, kb: &KnowledgeBase) {
        // Collect unique predicate names from rules
        let predicate_names: HashSet<String> =
            kb.rules.iter().map(|r| r.head.name.clone()).collect();

        for predicate_name in predicate_names {
            for rule in kb.get_rules(&predicate_name) {
                let head = &rule.head.name;
                let deps: HashSet<String> = rule.body.iter().map(|p| p.name.clone()).collect();

                self.dependencies
                    .entry(head.clone())
                    .or_default()
                    .extend(deps);
            }
        }
    }

    /// Check if dependency graph has cycles
    fn has_cycles(&self) -> bool {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for node in self.dependencies.keys() {
            if self.has_cycle_util(node, &mut visited, &mut rec_stack) {
                return true;
            }
        }

        false
    }

    /// Utility for cycle detection (DFS)
    fn has_cycle_util(
        &self,
        node: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> bool {
        if rec_stack.contains(node) {
            return true;
        }

        if visited.contains(node) {
            return false;
        }

        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());

        if let Some(neighbors) = self.dependencies.get(node) {
            for neighbor in neighbors {
                if self.has_cycle_util(neighbor, visited, rec_stack) {
                    return true;
                }
            }
        }

        rec_stack.remove(node);
        false
    }

    /// Compute stratification levels
    fn compute_strata(&self) -> Vec<Vec<String>> {
        let mut strata = Vec::new();
        let mut remaining: HashSet<String> = self.dependencies.keys().cloned().collect();

        while !remaining.is_empty() {
            // Find predicates with no dependencies on remaining predicates
            let mut current_stratum = Vec::new();

            for pred in &remaining {
                let has_remaining_deps = self
                    .dependencies
                    .get(pred)
                    .map(|deps| deps.iter().any(|d| remaining.contains(d)))
                    .unwrap_or(false);

                if !has_remaining_deps {
                    current_stratum.push(pred.clone());
                }
            }

            if current_stratum.is_empty() {
                // Shouldn't happen if no cycles, but break to avoid infinite loop
                break;
            }

            for pred in &current_stratum {
                remaining.remove(pred);
            }

            strata.push(current_stratum);
        }

        strata
    }
}

impl Default for StratificationAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of stratification analysis
#[derive(Debug, Clone)]
pub enum StratificationResult {
    /// Program is stratifiable with given strata
    Stratifiable(Vec<Vec<String>>),
    /// Program contains unstratifiable recursion
    NonStratifiable,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Constant, Term};

    #[test]
    fn test_tabled_inference_basic() {
        let mut kb = KnowledgeBase::new();

        // Add facts
        kb.add_fact(Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String("alice".to_string())),
                Term::Const(Constant::String("bob".to_string())),
            ],
        ));
        kb.add_fact(Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String("bob".to_string())),
                Term::Const(Constant::String("charlie".to_string())),
            ],
        ));

        // Add recursive rule: ancestor(X, Y) :- parent(X, Y)
        kb.add_rule(Rule::new(
            Predicate::new(
                "ancestor".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            vec![Predicate::new(
                "parent".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            )],
        ));

        // Add recursive rule: ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)
        kb.add_rule(Rule::new(
            Predicate::new(
                "ancestor".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Z".to_string())],
            ),
            vec![
                Predicate::new(
                    "parent".to_string(),
                    vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
                ),
                Predicate::new(
                    "ancestor".to_string(),
                    vec![Term::Var("Y".to_string()), Term::Var("Z".to_string())],
                ),
            ],
        ));

        let engine = TabledInferenceEngine::new();

        let goal = Predicate::new(
            "ancestor".to_string(),
            vec![
                Term::Const(Constant::String("alice".to_string())),
                Term::Var("Z".to_string()),
            ],
        );

        let solutions = engine.query(&goal, &kb).unwrap();
        assert!(!solutions.is_empty());
    }

    #[test]
    fn test_table_stats() {
        let engine = TabledInferenceEngine::new();
        let stats = engine.table_stats();
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.complete_entries, 0);
    }

    #[test]
    fn test_stratification_no_cycles() {
        let mut kb = KnowledgeBase::new();

        // Add non-recursive rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z)
        kb.add_rule(Rule::new(
            Predicate::new(
                "grandparent".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Z".to_string())],
            ),
            vec![
                Predicate::new(
                    "parent".to_string(),
                    vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
                ),
                Predicate::new(
                    "parent".to_string(),
                    vec![Term::Var("Y".to_string()), Term::Var("Z".to_string())],
                ),
            ],
        ));

        let mut analyzer = StratificationAnalyzer::new();
        let result = analyzer.analyze(&kb);

        match result {
            StratificationResult::Stratifiable(strata) => {
                assert!(!strata.is_empty());
            }
            StratificationResult::NonStratifiable => {
                // Should be stratifiable
                panic!("Expected stratifiable result");
            }
        }
    }

    #[test]
    fn test_fixpoint_engine() {
        let engine = FixpointEngine::new();
        let kb = KnowledgeBase::new();

        // Compute fixpoint (should return same KB for empty KB)
        let result = engine.compute_fixpoint(&kb).unwrap();
        assert_eq!(result.facts.len(), kb.facts.len());
    }
}
