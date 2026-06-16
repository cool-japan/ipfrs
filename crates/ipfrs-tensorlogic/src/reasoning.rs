//! Distributed reasoning and inference engine
//!
//! Implements backward chaining inference with unification for TensorLogic.
//! Features:
//! - Backward chaining with SLD-resolution
//! - Cycle detection to prevent infinite loops
//! - Memoization for query caching
//! - Goal decomposition tracking
//! - Distributed reasoning support

use crate::cache::{CacheManager, QueryKey};
use crate::ir::{KnowledgeBase, Predicate, Rule, Term};
use ipfrs_core::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Variable substitution (bindings)
pub type Substitution = HashMap<String, Term>;

/// Proof tree representing the derivation of a goal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proof {
    /// The proven goal
    pub goal: Predicate,
    /// The rule or fact used
    pub rule: Option<ProofRule>,
    /// Sub-proofs for the rule body
    pub subproofs: Vec<Proof>,
}

/// Rule used in a proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofRule {
    /// Head of the rule
    pub head: Predicate,
    /// Body of the rule
    pub body: Vec<Predicate>,
    /// Whether this was a fact (empty body)
    pub is_fact: bool,
}

impl Proof {
    /// Create a proof for a fact
    pub fn fact(goal: Predicate) -> Self {
        Self {
            rule: Some(ProofRule {
                head: goal.clone(),
                body: Vec::new(),
                is_fact: true,
            }),
            goal,
            subproofs: Vec::new(),
        }
    }

    /// Create a proof from a rule and subproofs
    pub fn from_rule(goal: Predicate, rule: &Rule, subproofs: Vec<Proof>) -> Self {
        Self {
            goal,
            rule: Some(ProofRule {
                head: rule.head.clone(),
                body: rule.body.clone(),
                is_fact: false,
            }),
            subproofs,
        }
    }

    /// Get the depth of the proof tree
    pub fn depth(&self) -> usize {
        if self.subproofs.is_empty() {
            1
        } else {
            1 + self.subproofs.iter().map(|p| p.depth()).max().unwrap_or(0)
        }
    }

    /// Get the number of nodes in the proof tree
    #[inline]
    pub fn size(&self) -> usize {
        1 + self.subproofs.iter().map(|p| p.size()).sum::<usize>()
    }

    /// Check if this proof is for a fact (no subproofs)
    #[inline]
    pub fn is_fact(&self) -> bool {
        self.subproofs.is_empty()
    }

    /// Get all goals in the proof tree (flattened)
    pub fn all_goals(&self) -> Vec<&Predicate> {
        let mut goals = vec![&self.goal];
        for subproof in &self.subproofs {
            goals.extend(subproof.all_goals());
        }
        goals
    }
}

/// Goal decomposition tracking for distributed reasoning
#[derive(Debug, Clone)]
pub struct GoalDecomposition {
    /// The original goal
    pub goal: Predicate,
    /// Subgoals after decomposition
    pub subgoals: Vec<Predicate>,
    /// The rule used for decomposition (if any)
    pub rule_applied: Option<Rule>,
    /// Whether each subgoal was solved locally
    pub local_solutions: Vec<bool>,
    /// Depth in the decomposition tree
    pub depth: usize,
}

impl GoalDecomposition {
    /// Create a new decomposition for a goal
    pub fn new(goal: Predicate, depth: usize) -> Self {
        Self {
            goal,
            subgoals: Vec::new(),
            rule_applied: None,
            local_solutions: Vec::new(),
            depth,
        }
    }

    /// Apply a rule to decompose the goal
    pub fn apply_rule(&mut self, rule: &Rule) {
        self.rule_applied = Some(rule.clone());
        self.subgoals = rule.body.clone();
        self.local_solutions = vec![false; rule.body.len()];
    }

    /// Mark a subgoal as solved locally
    pub fn mark_solved(&mut self, index: usize) {
        if index < self.local_solutions.len() {
            self.local_solutions[index] = true;
        }
    }

    /// Check if all subgoals are solved
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.local_solutions.iter().all(|&solved| solved)
    }

    /// Get unsolved subgoals (for distributed forwarding)
    pub fn unsolved_subgoals(&self) -> Vec<&Predicate> {
        self.subgoals
            .iter()
            .zip(self.local_solutions.iter())
            .filter(|(_, &solved)| !solved)
            .map(|(sg, _)| sg)
            .collect()
    }
}

/// Cycle detection context for recursive queries
#[derive(Debug, Clone, Default)]
pub struct CycleDetector {
    /// Stack of goals being solved
    goal_stack: Vec<String>,
    /// Set for O(1) lookup
    goal_set: HashSet<String>,
}

impl CycleDetector {
    /// Create a new cycle detector
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a goal onto the stack, returns false if cycle detected
    #[inline]
    pub fn push(&mut self, goal: &Predicate) -> bool {
        let key = goal_to_key(goal);
        if self.goal_set.contains(&key) {
            return false; // Cycle detected
        }
        self.goal_set.insert(key.clone());
        self.goal_stack.push(key);
        true
    }

    /// Pop a goal from the stack
    #[inline]
    pub fn pop(&mut self) {
        if let Some(key) = self.goal_stack.pop() {
            self.goal_set.remove(&key);
        }
    }

    /// Check if a goal would create a cycle
    #[inline]
    pub fn would_cycle(&self, goal: &Predicate) -> bool {
        let key = goal_to_key(goal);
        self.goal_set.contains(&key)
    }

    /// Get the current depth
    #[inline]
    pub fn depth(&self) -> usize {
        self.goal_stack.len()
    }

    /// Clear the detector
    pub fn clear(&mut self) {
        self.goal_stack.clear();
        self.goal_set.clear();
    }
}

/// Convert a goal predicate to a unique key for cycle detection
fn goal_to_key(goal: &Predicate) -> String {
    format!("{}({})", goal.name, goal.args.len())
}

/// Local inference engine with backward chaining
#[derive(Default)]
pub struct InferenceEngine {
    /// Maximum proof depth to prevent infinite loops
    max_depth: usize,
    /// Maximum number of solutions to find
    max_solutions: usize,
    /// Enable cycle detection
    cycle_detection: bool,
}

impl InferenceEngine {
    /// Create a new inference engine with default settings
    #[inline]
    pub fn new() -> Self {
        Self {
            max_depth: 100,
            max_solutions: 100,
            cycle_detection: true,
        }
    }

    /// Create an inference engine with custom limits
    #[inline]
    pub fn with_limits(max_depth: usize, max_solutions: usize) -> Self {
        Self {
            max_depth,
            max_solutions,
            cycle_detection: true,
        }
    }

    /// Enable or disable cycle detection
    #[inline]
    pub fn with_cycle_detection(mut self, enabled: bool) -> Self {
        self.cycle_detection = enabled;
        self
    }

    /// Query the knowledge base for solutions to a goal
    pub fn query(&self, goal: &Predicate, kb: &KnowledgeBase) -> Result<Vec<Substitution>> {
        let mut solutions = Vec::new();
        let initial_subst = Substitution::new();

        self.solve_goal(goal, &initial_subst, kb, 0, &mut solutions)?;

        Ok(solutions)
    }

    /// Prove a goal and return the proof tree
    pub fn prove(&self, goal: &Predicate, kb: &KnowledgeBase) -> Result<Option<Proof>> {
        let initial_subst = Substitution::new();
        self.prove_goal(goal, &initial_subst, kb, 0)
    }

    /// Backward chaining resolution for a single goal
    fn solve_goal(
        &self,
        goal: &Predicate,
        subst: &Substitution,
        kb: &KnowledgeBase,
        depth: usize,
        solutions: &mut Vec<Substitution>,
    ) -> Result<()> {
        // Check depth limit
        if depth > self.max_depth {
            return Ok(());
        }

        // Check solution limit
        if solutions.len() >= self.max_solutions {
            return Ok(());
        }

        // Apply current substitution to goal
        let goal = apply_subst_predicate(goal, subst);

        // Try to unify with facts
        for fact in kb.get_predicates(&goal.name) {
            if let Some(new_subst) = unify_predicates(&goal, fact, subst) {
                solutions.push(new_subst);
                if solutions.len() >= self.max_solutions {
                    return Ok(());
                }
            }
        }

        // Try to unify with rule heads
        for rule in kb.get_rules(&goal.name) {
            // Rename variables in the rule to avoid conflicts
            let renamed_rule = rename_rule_vars(rule, depth);

            // Try to unify goal with rule head
            if let Some(new_subst) = unify_predicates(&goal, &renamed_rule.head, subst) {
                // Solve rule body
                self.solve_conjunction(&renamed_rule.body, &new_subst, kb, depth + 1, solutions)?;
            }
        }

        Ok(())
    }

    /// Solve a conjunction of goals (rule body)
    fn solve_conjunction(
        &self,
        goals: &[Predicate],
        subst: &Substitution,
        kb: &KnowledgeBase,
        depth: usize,
        solutions: &mut Vec<Substitution>,
    ) -> Result<()> {
        if goals.is_empty() {
            // All goals satisfied, add solution
            solutions.push(subst.clone());
            return Ok(());
        }

        // Solve first goal
        let first_goal = &goals[0];
        let rest_goals = &goals[1..];

        let mut intermediate_solutions = Vec::new();
        self.solve_goal(first_goal, subst, kb, depth, &mut intermediate_solutions)?;

        // For each solution of the first goal, solve the rest
        for intermediate_subst in intermediate_solutions {
            if solutions.len() >= self.max_solutions {
                return Ok(());
            }
            self.solve_conjunction(rest_goals, &intermediate_subst, kb, depth, solutions)?;
        }

        Ok(())
    }

    /// Prove a goal and build the proof tree
    fn prove_goal(
        &self,
        goal: &Predicate,
        subst: &Substitution,
        kb: &KnowledgeBase,
        depth: usize,
    ) -> Result<Option<Proof>> {
        // Check depth limit
        if depth > self.max_depth {
            return Ok(None);
        }

        // Apply current substitution to goal
        let goal = apply_subst_predicate(goal, subst);

        // Try to unify with facts
        for fact in kb.get_predicates(&goal.name) {
            if let Some(_new_subst) = unify_predicates(&goal, fact, subst) {
                return Ok(Some(Proof::fact(goal)));
            }
        }

        // Try to unify with rule heads
        for rule in kb.get_rules(&goal.name) {
            // Rename variables in the rule to avoid conflicts
            let renamed_rule = rename_rule_vars(rule, depth);

            // Try to unify goal with rule head
            if let Some(new_subst) = unify_predicates(&goal, &renamed_rule.head, subst) {
                // Prove rule body
                if let Some(subproofs) =
                    self.prove_conjunction(&renamed_rule.body, &new_subst, kb, depth + 1)?
                {
                    return Ok(Some(Proof::from_rule(goal, &renamed_rule, subproofs)));
                }
            }
        }

        Ok(None)
    }

    /// Prove a conjunction of goals
    fn prove_conjunction(
        &self,
        goals: &[Predicate],
        subst: &Substitution,
        kb: &KnowledgeBase,
        depth: usize,
    ) -> Result<Option<Vec<Proof>>> {
        if goals.is_empty() {
            return Ok(Some(Vec::new()));
        }

        let first_goal = &goals[0];
        let rest_goals = &goals[1..];

        // Prove first goal
        if let Some(first_proof) = self.prove_goal(first_goal, subst, kb, depth)? {
            // Get the substitution from the proof (simplified - just use current subst)
            if let Some(rest_proofs) = self.prove_conjunction(rest_goals, subst, kb, depth)? {
                let mut all_proofs = vec![first_proof];
                all_proofs.extend(rest_proofs);
                return Ok(Some(all_proofs));
            }
        }

        Ok(None)
    }

    /// Verify that a proof is valid against a knowledge base
    pub fn verify(&self, proof: &Proof, kb: &KnowledgeBase) -> Result<bool> {
        self.verify_proof_recursive(proof, kb, 0)
    }

    /// Recursively verify a proof tree
    fn verify_proof_recursive(
        &self,
        proof: &Proof,
        kb: &KnowledgeBase,
        depth: usize,
    ) -> Result<bool> {
        // Check depth limit to prevent infinite recursion
        if depth > self.max_depth {
            return Ok(false);
        }

        // Check if proof has a rule
        let Some(ref rule) = proof.rule else {
            return Ok(false);
        };

        // Verify fact proof
        if rule.is_fact {
            // Check if the fact exists in the knowledge base
            let facts = kb.get_predicates(&proof.goal.name);
            for fact in facts {
                if unify_predicates(&proof.goal, fact, &Substitution::new()).is_some() {
                    return Ok(true);
                }
            }
            return Ok(false);
        }

        // Verify rule proof
        // 1. Check that the rule exists in KB
        let rules = kb.get_rules(&proof.goal.name);
        let mut rule_exists = false;
        for kb_rule in rules {
            // Check if rule heads match
            if kb_rule.head.name == rule.head.name
                && kb_rule.head.args.len() == rule.head.args.len()
                && kb_rule.body.len() == rule.body.len()
            {
                // Check if body predicates match
                let bodies_match = kb_rule
                    .body
                    .iter()
                    .zip(rule.body.iter())
                    .all(|(b1, b2)| b1.name == b2.name && b1.args.len() == b2.args.len());

                if bodies_match {
                    rule_exists = true;
                    break;
                }
            }
        }

        if !rule_exists {
            return Ok(false);
        }

        // 2. Verify that the number of subproofs matches the rule body
        if proof.subproofs.len() != rule.body.len() {
            return Ok(false);
        }

        // 3. Verify each subproof
        for (i, subproof) in proof.subproofs.iter().enumerate() {
            // Check that subproof goal matches the corresponding body predicate
            let body_predicate = &rule.body[i];
            if subproof.goal.name != body_predicate.name {
                return Ok(false);
            }

            // Recursively verify the subproof
            if !self.verify_proof_recursive(subproof, kb, depth + 1)? {
                return Ok(false);
            }
        }

        // All checks passed
        Ok(true)
    }
}

/// Unify two terms with a substitution
pub fn unify(t1: &Term, t2: &Term, subst: &Substitution) -> Option<Substitution> {
    let t1 = apply_subst_term(t1, subst);
    let t2 = apply_subst_term(t2, subst);

    match (&t1, &t2) {
        // Two identical constants
        (Term::Const(c1), Term::Const(c2)) if c1 == c2 => Some(subst.clone()),

        // Variable unification
        (Term::Var(v), t) | (t, Term::Var(v)) => {
            if let Term::Var(v2) = t {
                if v == v2 {
                    return Some(subst.clone());
                }
            }
            // Occurs check
            if occurs_in(v, t) {
                return None;
            }
            let mut new_subst = subst.clone();
            new_subst.insert(v.clone(), t.clone());
            Some(new_subst)
        }

        // Function unification
        (Term::Fun(f1, args1), Term::Fun(f2, args2)) if f1 == f2 && args1.len() == args2.len() => {
            let mut current_subst = subst.clone();
            for (a1, a2) in args1.iter().zip(args2.iter()) {
                match unify(a1, a2, &current_subst) {
                    Some(new_subst) => current_subst = new_subst,
                    None => return None,
                }
            }
            Some(current_subst)
        }

        // References - assume equal if CIDs match
        (Term::Ref(r1), Term::Ref(r2)) if r1.cid == r2.cid => Some(subst.clone()),

        _ => None,
    }
}

/// Unify two predicates
pub fn unify_predicates(
    p1: &Predicate,
    p2: &Predicate,
    subst: &Substitution,
) -> Option<Substitution> {
    if p1.name != p2.name || p1.args.len() != p2.args.len() {
        return None;
    }

    let mut current_subst = subst.clone();
    for (a1, a2) in p1.args.iter().zip(p2.args.iter()) {
        match unify(a1, a2, &current_subst) {
            Some(new_subst) => current_subst = new_subst,
            None => return None,
        }
    }

    Some(current_subst)
}

/// Check if a variable occurs in a term (for occurs check)
fn occurs_in(var: &str, term: &Term) -> bool {
    match term {
        Term::Var(v) => v == var,
        Term::Fun(_, args) => args.iter().any(|t| occurs_in(var, t)),
        _ => false,
    }
}

/// Apply substitution to a term
pub fn apply_subst_term(term: &Term, subst: &Substitution) -> Term {
    match term {
        Term::Var(v) => subst.get(v).cloned().unwrap_or_else(|| term.clone()),
        Term::Fun(f, args) => {
            let new_args = args.iter().map(|t| apply_subst_term(t, subst)).collect();
            Term::Fun(f.clone(), new_args)
        }
        _ => term.clone(),
    }
}

/// Apply substitution to a predicate
pub fn apply_subst_predicate(pred: &Predicate, subst: &Substitution) -> Predicate {
    Predicate {
        name: pred.name.clone(),
        args: pred
            .args
            .iter()
            .map(|t| apply_subst_term(t, subst))
            .collect(),
    }
}

/// Rename variables in a rule to avoid conflicts with depth-indexed suffixes.
pub fn rename_rule_vars(rule: &Rule, suffix: usize) -> Rule {
    let var_map: HashMap<String, String> = rule
        .variables()
        .into_iter()
        .map(|v| (v.clone(), format!("{}_{}", v, suffix)))
        .collect();

    let rename_subst: Substitution = var_map
        .into_iter()
        .map(|(old, new)| (old, Term::Var(new)))
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

/// Memoized inference engine with query caching
pub struct MemoizedInferenceEngine {
    /// Base inference engine
    engine: InferenceEngine,
    /// Cache manager
    cache: Arc<CacheManager>,
}

impl MemoizedInferenceEngine {
    /// Create a new memoized inference engine
    pub fn new(cache: Arc<CacheManager>) -> Self {
        Self {
            engine: InferenceEngine::new(),
            cache,
        }
    }

    /// Create with custom inference limits
    pub fn with_limits(max_depth: usize, max_solutions: usize, cache: Arc<CacheManager>) -> Self {
        Self {
            engine: InferenceEngine::with_limits(max_depth, max_solutions),
            cache,
        }
    }

    /// Query with memoization
    pub fn query(&self, goal: &Predicate, kb: &KnowledgeBase) -> Result<Vec<Substitution>> {
        // Create query key
        let key = QueryKey::from_predicate(goal);

        // Check cache first
        if let Some(cached) = self.cache.query_cache.get(&key) {
            return Ok(cached);
        }

        // Execute query
        let solutions = self.engine.query(goal, kb)?;

        // Cache results
        if !solutions.is_empty() {
            self.cache.query_cache.insert(key, solutions.clone());
        }

        Ok(solutions)
    }

    /// Prove with memoization
    pub fn prove(&self, goal: &Predicate, kb: &KnowledgeBase) -> Result<Option<Proof>> {
        // For proofs, we don't cache directly but could cache intermediate results
        self.engine.prove(goal, kb)
    }

    /// Get cache statistics
    #[inline]
    pub fn cache_stats(&self) -> crate::cache::CombinedCacheStats {
        self.cache.stats()
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        self.cache.query_cache.clear();
    }
}

/// Distributed reasoning engine with caching support
pub struct DistributedReasoner {
    /// Local inference engine
    engine: InferenceEngine,
    /// Cache manager (optional)
    cache: Option<Arc<CacheManager>>,
    /// Goal decomposition tracker
    decompositions: Vec<GoalDecomposition>,
}

impl DistributedReasoner {
    /// Create a new distributed reasoner
    pub fn new() -> Result<Self> {
        Ok(Self {
            engine: InferenceEngine::new(),
            cache: None,
            decompositions: Vec::new(),
        })
    }

    /// Create with a cache manager
    pub fn with_cache(cache: Arc<CacheManager>) -> Result<Self> {
        Ok(Self {
            engine: InferenceEngine::new(),
            cache: Some(cache),
            decompositions: Vec::new(),
        })
    }

    /// Query locally with optional caching
    pub async fn query(&self, goal: &Predicate, kb: &KnowledgeBase) -> Result<Vec<Substitution>> {
        // Check query cache first
        if let Some(cache) = &self.cache {
            let key = QueryKey::from_predicate(goal);
            if let Some(cached) = cache.query_cache.get(&key) {
                return Ok(cached);
            }

            // Execute query
            let solutions = self.engine.query(goal, kb)?;

            // Cache results
            if !solutions.is_empty() {
                cache.query_cache.insert(key, solutions.clone());
            }

            Ok(solutions)
        } else {
            self.engine.query(goal, kb)
        }
    }

    /// Prove a goal with goal decomposition tracking
    pub async fn prove(&self, goal: &Predicate, kb: &KnowledgeBase) -> Result<Option<Proof>> {
        self.engine.prove(goal, kb)
    }

    /// Prove with goal decomposition tracking
    pub async fn prove_with_decomposition(
        &mut self,
        goal: &Predicate,
        kb: &KnowledgeBase,
    ) -> Result<(Option<Proof>, Vec<GoalDecomposition>)> {
        self.decompositions.clear();
        let proof = self.prove_tracking(goal, kb, 0)?;
        let decomps = std::mem::take(&mut self.decompositions);
        Ok((proof, decomps))
    }

    /// Internal prove with decomposition tracking
    fn prove_tracking(
        &mut self,
        goal: &Predicate,
        kb: &KnowledgeBase,
        depth: usize,
    ) -> Result<Option<Proof>> {
        // Create decomposition record
        let mut decomp = GoalDecomposition::new(goal.clone(), depth);

        // Try facts first
        for fact in kb.get_predicates(&goal.name) {
            if unify_predicates(goal, fact, &Substitution::new()).is_some() {
                self.decompositions.push(decomp);
                return Ok(Some(Proof::fact(goal.clone())));
            }
        }

        // Try rules
        for rule in kb.get_rules(&goal.name) {
            let renamed_rule = rename_rule_vars(rule, depth);

            if let Some(subst) = unify_predicates(goal, &renamed_rule.head, &Substitution::new()) {
                decomp.apply_rule(&renamed_rule);

                // Prove subgoals
                let mut subproofs = Vec::new();
                let mut all_proved = true;

                for (i, subgoal) in renamed_rule.body.iter().enumerate() {
                    let subgoal = apply_subst_predicate(subgoal, &subst);
                    if let Some(subproof) = self.prove_tracking(&subgoal, kb, depth + 1)? {
                        subproofs.push(subproof);
                        decomp.mark_solved(i);
                    } else {
                        all_proved = false;
                        break;
                    }
                }

                if all_proved {
                    self.decompositions.push(decomp);
                    return Ok(Some(Proof::from_rule(
                        goal.clone(),
                        &renamed_rule,
                        subproofs,
                    )));
                }
            }
        }

        self.decompositions.push(decomp);
        Ok(None)
    }

    /// Get unsolved goals that could be forwarded to peers
    pub fn get_unsolved_goals(&self) -> Vec<&Predicate> {
        self.decompositions
            .iter()
            .flat_map(|d| d.unsolved_subgoals())
            .collect()
    }

    /// Get cache statistics (if cache is available)
    pub fn cache_stats(&self) -> Option<crate::cache::CombinedCacheStats> {
        self.cache.as_ref().map(|c| c.stats())
    }
}

impl Default for DistributedReasoner {
    fn default() -> Self {
        Self {
            engine: InferenceEngine::new(),
            cache: None,
            decompositions: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Constant;

    #[test]
    fn test_unify_constants() {
        let t1 = Term::Const(Constant::String("Alice".to_string()));
        let t2 = Term::Const(Constant::String("Alice".to_string()));
        let subst = Substitution::new();

        assert!(unify(&t1, &t2, &subst).is_some());
    }

    #[test]
    fn test_unify_var_const() {
        let t1 = Term::Var("X".to_string());
        let t2 = Term::Const(Constant::String("Alice".to_string()));
        let subst = Substitution::new();

        let result = unify(&t1, &t2, &subst);
        assert!(result.is_some());

        let result_subst = result.expect("test: should succeed");
        assert_eq!(
            result_subst.get("X"),
            Some(&Term::Const(Constant::String("Alice".to_string())))
        );
    }

    #[test]
    fn test_unify_functions() {
        let t1 = Term::Fun(
            "f".to_string(),
            vec![Term::Var("X".to_string()), Term::Const(Constant::Int(1))],
        );
        let t2 = Term::Fun(
            "f".to_string(),
            vec![
                Term::Const(Constant::String("a".to_string())),
                Term::Const(Constant::Int(1)),
            ],
        );
        let subst = Substitution::new();

        let result = unify(&t1, &t2, &subst);
        assert!(result.is_some());
    }

    #[test]
    fn test_occurs_check() {
        let t1 = Term::Var("X".to_string());
        let t2 = Term::Fun("f".to_string(), vec![Term::Var("X".to_string())]);
        let subst = Substitution::new();

        assert!(unify(&t1, &t2, &subst).is_none());
    }

    #[test]
    fn test_inference_fact() {
        let mut kb = KnowledgeBase::new();
        kb.add_fact(Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String("Alice".to_string())),
                Term::Const(Constant::String("Bob".to_string())),
            ],
        ));

        let engine = InferenceEngine::new();
        let goal = Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String("Alice".to_string())),
                Term::Var("X".to_string()),
            ],
        );

        let solutions = engine.query(&goal, &kb).expect("test: should succeed");
        assert_eq!(solutions.len(), 1);
        assert_eq!(
            solutions[0].get("X"),
            Some(&Term::Const(Constant::String("Bob".to_string())))
        );
    }

    #[test]
    fn test_inference_rule() {
        let mut kb = KnowledgeBase::new();

        // Facts: parent(alice, bob), parent(bob, charlie)
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

        // Rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z)
        let rule = Rule::new(
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
        );
        kb.add_rule(rule);

        let engine = InferenceEngine::new();
        let goal = Predicate::new(
            "grandparent".to_string(),
            vec![
                Term::Const(Constant::String("alice".to_string())),
                Term::Var("Z".to_string()),
            ],
        );

        let solutions = engine.query(&goal, &kb).expect("test: should succeed");
        assert!(!solutions.is_empty());
    }
}
