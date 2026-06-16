//! Rule Dependency Graph
//!
//! Before executing inference, rules must be evaluated in a well-defined order
//! determined by their interdependencies. This module builds a directed dependency
//! graph over rule IDs and computes a topological evaluation schedule so that every
//! rule is processed only after the rules it depends upon have already been applied.
//!
//! # Dependency types
//!
//! | Variant | Meaning |
//! |---------|---------|
//! | [`DependencyType::UsesConclusion`] | The head of one rule appears in the body of another. |
//! | [`DependencyType::SharesBody`] | Two rules share at least one body predicate. |
//! | [`DependencyType::Negation`] | A rule uses the negation of another rule's conclusion. |
//! | [`DependencyType::Subsumption`] | One rule's conclusion is subsumed by another. |
//!
//! # Examples
//!
//! ```
//! use ipfrs_tensorlogic::rule_dependency::{
//!     DependencyType, EvaluationSchedule, RuleDependencyGraph,
//! };
//!
//! let mut g = RuleDependencyGraph::new();
//! g.add_rule("base").expect("example: should succeed in docs");
//! g.add_rule("derived").expect("example: should succeed in docs");
//! g.add_dependency("derived", "base", DependencyType::UsesConclusion).expect("example: should succeed in docs");
//!
//! let order = g.topological_sort().expect("example: should succeed in docs");
//! assert_eq!(order, vec!["base".to_string(), "derived".to_string()]);
//!
//! let sched = EvaluationSchedule::build(&g).expect("example: should succeed in docs");
//! assert_eq!(sched.layer_count(), 2);
//! assert_eq!(sched.total_rules(), 2);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use thiserror::Error;

// ─── RuleId ──────────────────────────────────────────────────────────────────

/// A newtype wrapping a [`String`] that uniquely identifies a rule.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RuleId(pub String);

impl fmt::Display for RuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for RuleId {
    fn from(s: String) -> Self {
        RuleId(s)
    }
}

impl From<&str> for RuleId {
    fn from(s: &str) -> Self {
        RuleId(s.to_string())
    }
}

// ─── DependencyType ──────────────────────────────────────────────────────────

/// Characterises the semantic relationship between two rules in the dependency
/// graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DependencyType {
    /// The conclusion (head) of the `to` rule is used in the body of the
    /// `from` rule.
    UsesConclusion,
    /// Both rules share at least one body predicate, so evaluation order
    /// matters for consistency.
    SharesBody,
    /// The `from` rule uses the negation of a predicate derived by the `to`
    /// rule; must be evaluated after `to` under stratified semantics.
    Negation,
    /// The conclusion of the `from` rule is subsumed by (i.e. is a special
    /// case of) the conclusion of the `to` rule.
    Subsumption,
}

// ─── RuleDependency ──────────────────────────────────────────────────────────

/// A directed edge in the rule dependency graph.
///
/// Semantics: `from` depends on `to`.  `to` must be evaluated before `from`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleDependency {
    /// The rule that depends on another.
    pub from: RuleId,
    /// The rule that must be evaluated first.
    pub to: RuleId,
    /// The nature of the dependency.
    pub dep_type: DependencyType,
}

// ─── DepError ────────────────────────────────────────────────────────────────

/// Errors produced by [`RuleDependencyGraph`] and [`EvaluationSchedule`].
#[derive(Debug, Error)]
pub enum DepError {
    /// A rule with this ID was registered more than once.
    #[error("duplicate rule: {0}")]
    DuplicateRule(String),

    /// A referenced rule ID does not exist in the graph.
    #[error("rule not found: {0}")]
    RuleNotFound(String),

    /// The graph contains at least one cycle, making topological ordering
    /// impossible.  `involved` holds the IDs of the rules participating in the
    /// cycle.
    #[error("cycle detected among rules: {}", involved.join(", "))]
    CycleDetected {
        /// Rule IDs that are part of the cycle.
        involved: Vec<String>,
    },

    /// At least one endpoint of a dependency edge was not registered.
    #[error("dependency endpoint missing: from={from}, to={to}")]
    DependencyEndpointMissing {
        /// The `from` rule ID that was not found.
        from: String,
        /// The `to` rule ID that was not found.
        to: String,
    },
}

// ─── RuleDependencyGraph ─────────────────────────────────────────────────────

/// A directed graph that records dependencies between rules and can derive a
/// safe topological evaluation order.
///
/// Nodes are rule IDs (plain [`String`]s kept in a [`HashSet`] for O(1)
/// membership tests).  Edges are [`RuleDependency`] values stored in a
/// [`Vec`].
///
/// # Invariants
///
/// * Both endpoints of every [`RuleDependency`] must already be registered
///   via [`add_rule`][Self::add_rule] before the edge can be added.
/// * Duplicate rule IDs are rejected with [`DepError::DuplicateRule`].
#[derive(Debug, Default)]
pub struct RuleDependencyGraph {
    /// Set of registered rule IDs.
    pub rules: HashSet<String>,
    /// All dependency edges.
    pub deps: Vec<RuleDependency>,
}

impl RuleDependencyGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new rule.
    ///
    /// # Errors
    ///
    /// Returns [`DepError::DuplicateRule`] if a rule with the same ID is
    /// already present.
    pub fn add_rule(&mut self, id: &str) -> Result<(), DepError> {
        if self.rules.contains(id) {
            return Err(DepError::DuplicateRule(id.to_string()));
        }
        self.rules.insert(id.to_string());
        Ok(())
    }

    /// Add a directed dependency edge `from` → `to` (meaning `from` depends
    /// on `to`).
    ///
    /// # Errors
    ///
    /// Returns [`DepError::DependencyEndpointMissing`] if either endpoint has
    /// not been registered.
    pub fn add_dependency(
        &mut self,
        from: &str,
        to: &str,
        dep_type: DependencyType,
    ) -> Result<(), DepError> {
        let from_exists = self.rules.contains(from);
        let to_exists = self.rules.contains(to);
        if !from_exists || !to_exists {
            return Err(DepError::DependencyEndpointMissing {
                from: from.to_string(),
                to: to.to_string(),
            });
        }
        self.deps.push(RuleDependency {
            from: RuleId::from(from),
            to: RuleId::from(to),
            dep_type,
        });
        Ok(())
    }

    /// Compute a topological ordering of all registered rules using
    /// [Kahn's algorithm].
    ///
    /// The ordering guarantees that for every dependency edge `from` → `to`,
    /// `to` appears *before* `from` in the returned vector.
    ///
    /// [Kahn's algorithm]: https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
    ///
    /// # Errors
    ///
    /// Returns [`DepError::CycleDetected`] if the graph contains a cycle.
    pub fn topological_sort(&self) -> Result<Vec<String>, DepError> {
        // Build adjacency list and in-degree map.
        // Edge direction: (to -> from) for adjacency (from depends on to).
        // In-degree counts how many `to` nodes point *into* each `from` node.
        let mut in_degree: HashMap<String, usize> =
            self.rules.iter().map(|r| (r.clone(), 0)).collect();

        // adjacency: to -> list of `from` rules that depend on it
        let mut adj: HashMap<String, Vec<String>> =
            self.rules.iter().map(|r| (r.clone(), Vec::new())).collect();

        for dep in &self.deps {
            let from_str = dep.from.0.clone();
            let to_str = dep.to.0.clone();
            adj.entry(to_str).or_default().push(from_str.clone());
            *in_degree.entry(from_str).or_insert(0) += 1;
        }

        // Initialise queue with all zero-in-degree nodes (sorted for determinism).
        let mut queue: VecDeque<String> = {
            let mut zeros: Vec<String> = in_degree
                .iter()
                .filter(|(_, &d)| d == 0)
                .map(|(r, _)| r.clone())
                .collect();
            zeros.sort();
            VecDeque::from(zeros)
        };

        let mut result: Vec<String> = Vec::with_capacity(self.rules.len());

        while let Some(node) = queue.pop_front() {
            result.push(node.clone());
            if let Some(dependents) = adj.get(&node) {
                let mut next_batch: Vec<String> = Vec::new();
                for dep_node in dependents {
                    let deg = in_degree.get_mut(dep_node).expect("node always present");
                    *deg -= 1;
                    if *deg == 0 {
                        next_batch.push(dep_node.clone());
                    }
                }
                next_batch.sort();
                for n in next_batch {
                    queue.push_back(n);
                }
            }
        }

        if result.len() != self.rules.len() {
            // Collect all nodes that still have non-zero in-degree — they are
            // part of the cycle.
            let mut involved: Vec<String> = in_degree
                .into_iter()
                .filter(|(_, d)| *d > 0)
                .map(|(r, _)| r)
                .collect();
            involved.sort();
            return Err(DepError::CycleDetected { involved });
        }

        Ok(result)
    }

    /// Return the IDs of rules that `rule_id` directly depends on (i.e. the
    /// `to` endpoints of all edges whose `from` is `rule_id`).
    pub fn dependencies_of(&self, rule_id: &str) -> Vec<String> {
        let mut deps: Vec<String> = self
            .deps
            .iter()
            .filter(|d| d.from.0 == rule_id)
            .map(|d| d.to.0.clone())
            .collect();
        deps.sort();
        deps.dedup();
        deps
    }

    /// Return the IDs of rules that directly depend on `rule_id` (i.e. the
    /// `from` endpoints of all edges whose `to` is `rule_id`).
    pub fn dependents_of(&self, rule_id: &str) -> Vec<String> {
        let mut deps: Vec<String> = self
            .deps
            .iter()
            .filter(|d| d.to.0 == rule_id)
            .map(|d| d.from.0.clone())
            .collect();
        deps.sort();
        deps.dedup();
        deps
    }

    /// Return `true` if the dependency graph contains at least one cycle.
    pub fn has_cycle(&self) -> bool {
        self.topological_sort().is_err()
    }

    /// Return the number of registered rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Return the number of dependency edges.
    pub fn dep_count(&self) -> usize {
        self.deps.len()
    }
}

// ─── EvaluationSchedule ──────────────────────────────────────────────────────

/// A layered evaluation schedule derived from a [`RuleDependencyGraph`].
///
/// Layer 0 contains all rules that have no dependencies.  Layer *k* contains
/// all rules whose dependencies are entirely contained in layers 0 … *k-1*.
///
/// Within each layer, rules can in principle be evaluated in parallel because
/// they are independent of each other.
#[derive(Debug, Clone)]
pub struct EvaluationSchedule {
    /// Rules grouped by evaluation layer.
    ///
    /// `layers[0]` holds rules with no dependencies; `layers[k]` holds rules
    /// whose latest dependency is in layer `k-1`.
    pub layers: Vec<Vec<String>>,
}

impl EvaluationSchedule {
    /// Build an [`EvaluationSchedule`] from a [`RuleDependencyGraph`].
    ///
    /// The implementation performs a BFS / Kahn-style traversal and assigns
    /// each rule to the layer immediately after its deepest dependency.
    ///
    /// # Errors
    ///
    /// Propagates [`DepError::CycleDetected`] if the graph contains a cycle.
    pub fn build(graph: &RuleDependencyGraph) -> Result<Self, DepError> {
        // Build in-degree and adjacency the same way as topological_sort.
        let mut in_degree: HashMap<String, usize> =
            graph.rules.iter().map(|r| (r.clone(), 0)).collect();

        // adjacency: to -> list of `from` rules
        let mut adj: HashMap<String, Vec<String>> = graph
            .rules
            .iter()
            .map(|r| (r.clone(), Vec::new()))
            .collect();

        for dep in &graph.deps {
            let from_str = dep.from.0.clone();
            let to_str = dep.to.0.clone();
            adj.entry(to_str).or_default().push(from_str.clone());
            *in_degree.entry(from_str).or_insert(0) += 1;
        }

        let mut layers: Vec<Vec<String>> = Vec::new();
        let mut processed = 0usize;

        // Seed the first layer with all zero-in-degree nodes.
        let mut current_layer: Vec<String> = {
            let mut v: Vec<String> = in_degree
                .iter()
                .filter(|(_, &d)| d == 0)
                .map(|(r, _)| r.clone())
                .collect();
            v.sort();
            v
        };

        while !current_layer.is_empty() {
            processed += current_layer.len();
            let mut next_layer: Vec<String> = Vec::new();

            for node in &current_layer {
                if let Some(dependents) = adj.get(node) {
                    for dep_node in dependents {
                        let deg = in_degree.get_mut(dep_node).expect("node always present");
                        *deg -= 1;
                        if *deg == 0 {
                            next_layer.push(dep_node.clone());
                        }
                    }
                }
            }

            layers.push(current_layer);
            next_layer.sort();
            next_layer.dedup();
            current_layer = next_layer;
        }

        if processed != graph.rules.len() {
            let mut involved: Vec<String> = in_degree
                .into_iter()
                .filter(|(_, d)| *d > 0)
                .map(|(r, _)| r)
                .collect();
            involved.sort();
            return Err(DepError::CycleDetected { involved });
        }

        Ok(EvaluationSchedule { layers })
    }

    /// Return the number of evaluation layers.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Return the total number of rules across all layers.
    pub fn total_rules(&self) -> usize {
        self.layers.iter().map(|l| l.len()).sum()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_linear_chain(n: usize) -> RuleDependencyGraph {
        let mut g = RuleDependencyGraph::new();
        for i in 0..n {
            g.add_rule(&format!("r{i}")).expect("test: should succeed");
        }
        // r1 depends on r0, r2 depends on r1, …
        for i in 1..n {
            g.add_dependency(
                &format!("r{i}"),
                &format!("r{}", i - 1),
                DependencyType::UsesConclusion,
            )
            .expect("test: should succeed");
        }
        g
    }

    // ── 1: add rules ─────────────────────────────────────────────────────────

    #[test]
    fn test_add_rules_increases_count() {
        let mut g = RuleDependencyGraph::new();
        assert_eq!(g.rule_count(), 0);
        g.add_rule("a").expect("test: should succeed");
        g.add_rule("b").expect("test: should succeed");
        assert_eq!(g.rule_count(), 2);
    }

    // ── 2: duplicate rule error ───────────────────────────────────────────────

    #[test]
    fn test_duplicate_rule_error() {
        let mut g = RuleDependencyGraph::new();
        g.add_rule("x").expect("test: should succeed");
        let err = g.add_rule("x").unwrap_err();
        assert!(matches!(err, DepError::DuplicateRule(ref s) if s == "x"));
    }

    // ── 3: add dependency increases dep_count ────────────────────────────────

    #[test]
    fn test_add_dependency_increases_count() {
        let mut g = RuleDependencyGraph::new();
        g.add_rule("a").expect("test: should succeed");
        g.add_rule("b").expect("test: should succeed");
        g.add_dependency("a", "b", DependencyType::SharesBody)
            .expect("test: should succeed");
        assert_eq!(g.dep_count(), 1);
    }

    // ── 4: dependency with missing endpoint error ─────────────────────────────

    #[test]
    fn test_dependency_missing_endpoint_from() {
        let mut g = RuleDependencyGraph::new();
        g.add_rule("b").expect("test: should succeed");
        let err = g
            .add_dependency("ghost", "b", DependencyType::Negation)
            .unwrap_err();
        assert!(
            matches!(err, DepError::DependencyEndpointMissing { ref from, .. } if from == "ghost")
        );
    }

    #[test]
    fn test_dependency_missing_endpoint_to() {
        let mut g = RuleDependencyGraph::new();
        g.add_rule("a").expect("test: should succeed");
        let err = g
            .add_dependency("a", "ghost", DependencyType::Negation)
            .unwrap_err();
        assert!(matches!(err, DepError::DependencyEndpointMissing { ref to, .. } if to == "ghost"));
    }

    // ── 5: topological_sort — linear chain ───────────────────────────────────

    #[test]
    fn test_topo_sort_linear_chain() {
        let g = make_linear_chain(4);
        let order = g.topological_sort().expect("test: should succeed");
        // r0 must appear before r1, r1 before r2, etc.
        let pos: HashMap<_, _> = order
            .iter()
            .enumerate()
            .map(|(i, r)| (r.as_str(), i))
            .collect();
        for i in 1..4usize {
            assert!(
                pos[&format!("r{}", i - 1).as_str()] < pos[&format!("r{i}").as_str()],
                "r{} must precede r{}",
                i - 1,
                i
            );
        }
    }

    // ── 6: topological_sort — diamond ────────────────────────────────────────

    #[test]
    fn test_topo_sort_diamond() {
        // a <- b <- d
        //      ^
        //      c <- d
        // d depends on both b and c; b and c depend on a.
        let mut g = RuleDependencyGraph::new();
        for r in ["a", "b", "c", "d"] {
            g.add_rule(r).expect("test: should succeed");
        }
        g.add_dependency("b", "a", DependencyType::UsesConclusion)
            .expect("test: should succeed");
        g.add_dependency("c", "a", DependencyType::UsesConclusion)
            .expect("test: should succeed");
        g.add_dependency("d", "b", DependencyType::UsesConclusion)
            .expect("test: should succeed");
        g.add_dependency("d", "c", DependencyType::UsesConclusion)
            .expect("test: should succeed");

        let order = g.topological_sort().expect("test: should succeed");
        let pos: HashMap<&str, usize> = order
            .iter()
            .enumerate()
            .map(|(i, r)| (r.as_str(), i))
            .collect();
        assert!(pos["a"] < pos["b"]);
        assert!(pos["a"] < pos["c"]);
        assert!(pos["b"] < pos["d"]);
        assert!(pos["c"] < pos["d"]);
    }

    // ── 7: topological_sort — cycle → error ──────────────────────────────────

    #[test]
    fn test_topo_sort_cycle_error() {
        let mut g = RuleDependencyGraph::new();
        g.add_rule("x").expect("test: should succeed");
        g.add_rule("y").expect("test: should succeed");
        g.add_dependency("x", "y", DependencyType::UsesConclusion)
            .expect("test: should succeed");
        g.add_dependency("y", "x", DependencyType::UsesConclusion)
            .expect("test: should succeed");

        let err = g.topological_sort().unwrap_err();
        assert!(matches!(err, DepError::CycleDetected { .. }));
    }

    // ── 8: has_cycle — false ─────────────────────────────────────────────────

    #[test]
    fn test_has_cycle_false() {
        let g = make_linear_chain(5);
        assert!(!g.has_cycle());
    }

    // ── 9: has_cycle — true ──────────────────────────────────────────────────

    #[test]
    fn test_has_cycle_true() {
        let mut g = RuleDependencyGraph::new();
        g.add_rule("a").expect("test: should succeed");
        g.add_rule("b").expect("test: should succeed");
        g.add_rule("c").expect("test: should succeed");
        g.add_dependency("a", "b", DependencyType::Subsumption)
            .expect("test: should succeed");
        g.add_dependency("b", "c", DependencyType::Subsumption)
            .expect("test: should succeed");
        g.add_dependency("c", "a", DependencyType::Subsumption)
            .expect("test: should succeed");
        assert!(g.has_cycle());
    }

    // ── 10: dependencies_of ──────────────────────────────────────────────────

    #[test]
    fn test_dependencies_of() {
        let mut g = RuleDependencyGraph::new();
        for r in ["a", "b", "c", "d"] {
            g.add_rule(r).expect("test: should succeed");
        }
        g.add_dependency("d", "b", DependencyType::UsesConclusion)
            .expect("test: should succeed");
        g.add_dependency("d", "c", DependencyType::UsesConclusion)
            .expect("test: should succeed");
        g.add_dependency("b", "a", DependencyType::UsesConclusion)
            .expect("test: should succeed");

        let mut deps = g.dependencies_of("d");
        deps.sort();
        assert_eq!(deps, vec!["b".to_string(), "c".to_string()]);

        let deps_b = g.dependencies_of("b");
        assert_eq!(deps_b, vec!["a".to_string()]);

        let deps_a = g.dependencies_of("a");
        assert!(deps_a.is_empty());
    }

    // ── 11: dependents_of ────────────────────────────────────────────────────

    #[test]
    fn test_dependents_of() {
        let mut g = RuleDependencyGraph::new();
        for r in ["a", "b", "c"] {
            g.add_rule(r).expect("test: should succeed");
        }
        g.add_dependency("b", "a", DependencyType::UsesConclusion)
            .expect("test: should succeed");
        g.add_dependency("c", "a", DependencyType::UsesConclusion)
            .expect("test: should succeed");

        let mut deps = g.dependents_of("a");
        deps.sort();
        assert_eq!(deps, vec!["b".to_string(), "c".to_string()]);

        let empty = g.dependents_of("b");
        assert!(empty.is_empty());
    }

    // ── 12: EvaluationSchedule — basic layers ────────────────────────────────

    #[test]
    fn test_evaluation_schedule_build_basic() {
        let mut g = RuleDependencyGraph::new();
        for r in ["a", "b", "c"] {
            g.add_rule(r).expect("test: should succeed");
        }
        // b depends on a; c depends on b
        g.add_dependency("b", "a", DependencyType::UsesConclusion)
            .expect("test: should succeed");
        g.add_dependency("c", "b", DependencyType::UsesConclusion)
            .expect("test: should succeed");

        let sched = EvaluationSchedule::build(&g).expect("test: should succeed");
        assert_eq!(sched.layer_count(), 3);
        assert_eq!(sched.total_rules(), 3);
        assert_eq!(sched.layers[0], vec!["a".to_string()]);
        assert_eq!(sched.layers[1], vec!["b".to_string()]);
        assert_eq!(sched.layers[2], vec!["c".to_string()]);
    }

    // ── 13: Layer 0 contains independent rules ───────────────────────────────

    #[test]
    fn test_layer_zero_contains_independent_rules() {
        let mut g = RuleDependencyGraph::new();
        // i0, i1, i2 are independent; d depends on i0.
        for r in ["i0", "i1", "i2", "d"] {
            g.add_rule(r).expect("test: should succeed");
        }
        g.add_dependency("d", "i0", DependencyType::UsesConclusion)
            .expect("test: should succeed");

        let sched = EvaluationSchedule::build(&g).expect("test: should succeed");
        // layer 0 must include i1, i2 and i0 (all have in-degree 0).
        let mut layer0 = sched.layers[0].clone();
        layer0.sort();
        assert!(layer0.contains(&"i0".to_string()));
        assert!(layer0.contains(&"i1".to_string()));
        assert!(layer0.contains(&"i2".to_string()));
        assert!(!layer0.contains(&"d".to_string()));
    }

    // ── 14: Rules in later layers have all deps in earlier layers ────────────

    #[test]
    fn test_later_layers_deps_in_earlier_layers() {
        let g = make_linear_chain(6);
        let sched = EvaluationSchedule::build(&g).expect("test: should succeed");

        // Build a map: rule → layer index.
        let mut layer_of: HashMap<String, usize> = HashMap::new();
        for (idx, layer) in sched.layers.iter().enumerate() {
            for r in layer {
                layer_of.insert(r.clone(), idx);
            }
        }

        for dep in &g.deps {
            let from_layer = layer_of[&dep.from.0];
            let to_layer = layer_of[&dep.to.0];
            assert!(
                to_layer < from_layer,
                "dep.to ({}) must be in an earlier layer than dep.from ({})",
                dep.to,
                dep.from
            );
        }
    }

    // ── 15: EvaluationSchedule — diamond layers ───────────────────────────────

    #[test]
    fn test_schedule_diamond_layers() {
        let mut g = RuleDependencyGraph::new();
        for r in ["a", "b", "c", "d"] {
            g.add_rule(r).expect("test: should succeed");
        }
        g.add_dependency("b", "a", DependencyType::UsesConclusion)
            .expect("test: should succeed");
        g.add_dependency("c", "a", DependencyType::UsesConclusion)
            .expect("test: should succeed");
        g.add_dependency("d", "b", DependencyType::UsesConclusion)
            .expect("test: should succeed");
        g.add_dependency("d", "c", DependencyType::UsesConclusion)
            .expect("test: should succeed");

        let sched = EvaluationSchedule::build(&g).expect("test: should succeed");
        assert_eq!(sched.total_rules(), 4);
        // Layer 0: a, Layer 1: b, c, Layer 2: d
        assert_eq!(sched.layers[0], vec!["a".to_string()]);
        let mut l1 = sched.layers[1].clone();
        l1.sort();
        assert_eq!(l1, vec!["b".to_string(), "c".to_string()]);
        assert_eq!(sched.layers[2], vec!["d".to_string()]);
    }

    // ── 16: EvaluationSchedule — cycle → error ───────────────────────────────

    #[test]
    fn test_schedule_cycle_error() {
        let mut g = RuleDependencyGraph::new();
        g.add_rule("p").expect("test: should succeed");
        g.add_rule("q").expect("test: should succeed");
        g.add_dependency("p", "q", DependencyType::Negation)
            .expect("test: should succeed");
        g.add_dependency("q", "p", DependencyType::Negation)
            .expect("test: should succeed");

        let err = EvaluationSchedule::build(&g).unwrap_err();
        assert!(matches!(err, DepError::CycleDetected { .. }));
    }

    // ── 17: RuleId conversions ────────────────────────────────────────────────

    #[test]
    fn test_rule_id_conversions() {
        let from_str: RuleId = RuleId::from("hello");
        let from_string: RuleId = RuleId::from("hello".to_string());
        assert_eq!(from_str, from_string);
        assert_eq!(from_str.to_string(), "hello");
    }

    // ── 18: rule_count / dep_count ───────────────────────────────────────────

    #[test]
    fn test_rule_count_and_dep_count() {
        let g = make_linear_chain(5);
        assert_eq!(g.rule_count(), 5);
        assert_eq!(g.dep_count(), 4); // 4 edges for 5 nodes in a chain
    }
}
