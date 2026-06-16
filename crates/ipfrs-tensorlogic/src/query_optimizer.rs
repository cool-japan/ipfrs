//! TensorQueryOptimizer — Rewrites TensorLogic query plans before execution.
//!
//! Supported rules:
//! - [`OptimizationRule::PushFilterDown`]       — hoist filter below join
//! - [`OptimizationRule::EliminateDeadProject`] — drop empty projection
//! - [`OptimizationRule::ReorderJoin`]          — put cheaper side on left
//! - [`OptimizationRule::FoldConstantLimit`]    — zero-row limit → empty scan
//! - [`OptimizationRule::FlattenNestedFilter`]  — merge double filter

// ---------------------------------------------------------------------------
// QueryNode
// ---------------------------------------------------------------------------

/// A node in a logical query plan.
#[derive(Clone, Debug, PartialEq)]
pub enum QueryNode {
    /// Leaf table scan with an optional predicate string and cost hint.
    Scan {
        predicate: String,
        estimated_rows: u64,
    },
    /// Apply a boolean condition on top of a child plan.
    Filter {
        child: Box<QueryNode>,
        condition: String,
    },
    /// Hash / nested-loop join on a single key.
    Join {
        left: Box<QueryNode>,
        right: Box<QueryNode>,
        join_key: String,
    },
    /// Column projection (field list).
    Project {
        child: Box<QueryNode>,
        fields: Vec<String>,
    },
    /// Row-count cap.
    Limit {
        child: Box<QueryNode>,
        max_rows: u64,
    },
}

// ---------------------------------------------------------------------------
// OptimizationRule
// ---------------------------------------------------------------------------

/// Rewriting rule that can be applied by [`TensorQueryOptimizer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizationRule {
    /// Move a Filter node below a Join node so it is applied earlier.
    PushFilterDown,
    /// Remove a Project node whose field list is empty.
    EliminateDeadProject,
    /// Swap Join children so the cheaper (smaller) side is on the left.
    ReorderJoin,
    /// Replace a `Limit { max_rows: 0, .. }` with an empty scan.
    FoldConstantLimit,
    /// Merge two consecutive Filter nodes into one with `" AND "`.
    FlattenNestedFilter,
}

// ---------------------------------------------------------------------------
// Cost model
// ---------------------------------------------------------------------------

/// Recursively estimate the cost of executing `node`.
pub fn estimated_cost(node: &QueryNode) -> u64 {
    match node {
        QueryNode::Scan { estimated_rows, .. } => *estimated_rows,
        QueryNode::Filter { child, .. } => estimated_cost(child).saturating_div(2),
        QueryNode::Join { left, right, .. } => {
            let lc = estimated_cost(left);
            let rc = estimated_cost(right);
            lc.saturating_mul(rc).saturating_div(100).saturating_add(1)
        }
        QueryNode::Project { child, .. } => estimated_cost(child),
        QueryNode::Limit { child, max_rows } => (*max_rows).min(estimated_cost(child)),
    }
}

// ---------------------------------------------------------------------------
// Rule application (single node, bottom-up)
// ---------------------------------------------------------------------------

/// Apply a single rule to the root node only (children already optimised).
/// Returns `(new_node, changed)`.
fn apply_rule(rule: OptimizationRule, node: QueryNode) -> (QueryNode, bool) {
    match rule {
        OptimizationRule::PushFilterDown => push_filter_down(node),
        OptimizationRule::EliminateDeadProject => eliminate_dead_project(node),
        OptimizationRule::ReorderJoin => reorder_join(node),
        OptimizationRule::FoldConstantLimit => fold_constant_limit(node),
        OptimizationRule::FlattenNestedFilter => flatten_nested_filter(node),
    }
}

fn push_filter_down(node: QueryNode) -> (QueryNode, bool) {
    match node {
        QueryNode::Filter { child, condition } => {
            match *child {
                QueryNode::Join {
                    left,
                    right,
                    join_key,
                } => {
                    // Push the filter onto the left branch of the join.
                    let new_left = Box::new(QueryNode::Filter {
                        child: left,
                        condition: condition.clone(),
                    });
                    let new_node = QueryNode::Join {
                        left: new_left,
                        right,
                        join_key,
                    };
                    (new_node, true)
                }
                other => (
                    QueryNode::Filter {
                        child: Box::new(other),
                        condition,
                    },
                    false,
                ),
            }
        }
        other => (other, false),
    }
}

fn eliminate_dead_project(node: QueryNode) -> (QueryNode, bool) {
    match node {
        QueryNode::Project { child, fields } if fields.is_empty() => (*child, true),
        other => (other, false),
    }
}

fn reorder_join(node: QueryNode) -> (QueryNode, bool) {
    match node {
        QueryNode::Join {
            left,
            right,
            join_key,
        } => {
            let lc = estimated_cost(&left);
            let rc = estimated_cost(&right);
            if rc < lc {
                (
                    QueryNode::Join {
                        left: right,
                        right: left,
                        join_key,
                    },
                    true,
                )
            } else {
                (
                    QueryNode::Join {
                        left,
                        right,
                        join_key,
                    },
                    false,
                )
            }
        }
        other => (other, false),
    }
}

fn fold_constant_limit(node: QueryNode) -> (QueryNode, bool) {
    match node {
        QueryNode::Limit { max_rows: 0, .. } => (
            QueryNode::Scan {
                predicate: "empty".to_string(),
                estimated_rows: 0,
            },
            true,
        ),
        other => (other, false),
    }
}

fn flatten_nested_filter(node: QueryNode) -> (QueryNode, bool) {
    match node {
        QueryNode::Filter {
            child,
            condition: c1,
        } => match *child {
            QueryNode::Filter {
                child: inner,
                condition: c2,
            } => {
                let merged = format!("{c2} AND {c1}");
                (
                    QueryNode::Filter {
                        child: inner,
                        condition: merged,
                    },
                    true,
                )
            }
            other => (
                QueryNode::Filter {
                    child: Box::new(other),
                    condition: c1,
                },
                false,
            ),
        },
        other => (other, false),
    }
}

// ---------------------------------------------------------------------------
// Tree-level rule application (recurse then rewrite root)
// ---------------------------------------------------------------------------

/// Recursively apply `rule` bottom-up across the whole tree.
/// Returns `(new_tree, changed_anywhere)`.
fn apply_rule_tree(rule: OptimizationRule, node: QueryNode) -> (QueryNode, bool) {
    // Recurse into children first, then apply rule at this level.
    let (node, child_changed) = recurse_children(rule, node);
    let (node, self_changed) = apply_rule(rule, node);
    (node, child_changed || self_changed)
}

/// Descend into children, applying `rule` bottom-up, then reassemble.
fn recurse_children(rule: OptimizationRule, node: QueryNode) -> (QueryNode, bool) {
    match node {
        QueryNode::Scan { .. } => (node, false),
        QueryNode::Filter { child, condition } => {
            let (new_child, changed) = apply_rule_tree(rule, *child);
            (
                QueryNode::Filter {
                    child: Box::new(new_child),
                    condition,
                },
                changed,
            )
        }
        QueryNode::Join {
            left,
            right,
            join_key,
        } => {
            let (new_left, cl) = apply_rule_tree(rule, *left);
            let (new_right, cr) = apply_rule_tree(rule, *right);
            (
                QueryNode::Join {
                    left: Box::new(new_left),
                    right: Box::new(new_right),
                    join_key,
                },
                cl || cr,
            )
        }
        QueryNode::Project { child, fields } => {
            let (new_child, changed) = apply_rule_tree(rule, *child);
            (
                QueryNode::Project {
                    child: Box::new(new_child),
                    fields,
                },
                changed,
            )
        }
        QueryNode::Limit { child, max_rows } => {
            let (new_child, changed) = apply_rule_tree(rule, *child);
            (
                QueryNode::Limit {
                    child: Box::new(new_child),
                    max_rows,
                },
                changed,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// OptimizationResult
// ---------------------------------------------------------------------------

/// Summary of a single [`TensorQueryOptimizer::optimize`] call.
#[derive(Clone, Debug)]
pub struct OptimizationResult {
    /// Estimated cost of the original (unoptimised) plan.
    pub original_cost: u64,
    /// Estimated cost of the optimised plan.
    pub optimized_cost: u64,
    /// Ordered list of rules that fired at least once during this run.
    pub rules_applied: Vec<OptimizationRule>,
}

impl OptimizationResult {
    /// Percentage improvement: `(original - optimized) / original * 100`.
    /// Returns `0.0` when `original_cost == 0`.
    pub fn improvement_pct(&self) -> f64 {
        if self.original_cost == 0 {
            return 0.0;
        }
        let saved = self.original_cost.saturating_sub(self.optimized_cost) as f64;
        saved / self.original_cost as f64 * 100.0
    }
}

// ---------------------------------------------------------------------------
// OptimizerStats
// ---------------------------------------------------------------------------

/// Lifetime statistics for a [`TensorQueryOptimizer`] instance.
#[derive(Clone, Debug)]
pub struct OptimizerStats {
    /// Number of times [`TensorQueryOptimizer::optimize`] has been called.
    pub total_optimizations: u64,
    /// Sum of cost reductions across all optimisation calls.
    pub total_cost_saved: u64,
    /// Number of rules currently enabled in the optimizer.
    pub enabled_rules: usize,
}

// ---------------------------------------------------------------------------
// TensorQueryOptimizer
// ---------------------------------------------------------------------------

/// Optimises TensorLogic query plans by repeatedly applying rewriting rules.
pub struct TensorQueryOptimizer {
    /// Ordered list of rules to apply during each pass.
    pub enabled_rules: Vec<OptimizationRule>,
    /// Cumulative count of [`Self::optimize`] invocations.
    pub total_optimizations: u64,
    /// Cumulative cost units saved across all invocations.
    pub total_cost_saved: u64,
}

impl TensorQueryOptimizer {
    /// Create a new optimizer with the given rule set.
    pub fn new(rules: Vec<OptimizationRule>) -> Self {
        Self {
            enabled_rules: rules,
            total_optimizations: 0,
            total_cost_saved: 0,
        }
    }

    /// Optimise `root`, returning the new plan and an [`OptimizationResult`].
    ///
    /// Rules are applied in order; the process repeats until the plan is
    /// stable or 10 passes have been completed.
    pub fn optimize(&mut self, root: QueryNode) -> (QueryNode, OptimizationResult) {
        let original_cost = estimated_cost(&root);
        let mut current = root;
        let mut rules_applied: Vec<OptimizationRule> = Vec::new();

        const MAX_PASSES: usize = 10;

        for _ in 0..MAX_PASSES {
            let mut pass_changed = false;

            for &rule in &self.enabled_rules {
                let (new_node, changed) = apply_rule_tree(rule, current);
                current = new_node;
                if changed {
                    pass_changed = true;
                    if !rules_applied.contains(&rule) {
                        rules_applied.push(rule);
                    }
                }
            }

            if !pass_changed {
                break;
            }
        }

        let optimized_cost = estimated_cost(&current);
        let result = OptimizationResult {
            original_cost,
            optimized_cost,
            rules_applied,
        };

        self.total_optimizations += 1;
        self.total_cost_saved = self
            .total_cost_saved
            .saturating_add(original_cost.saturating_sub(optimized_cost));

        (current, result)
    }

    /// Return a snapshot of lifetime statistics.
    pub fn stats(&self) -> OptimizerStats {
        OptimizerStats {
            total_optimizations: self.total_optimizations,
            total_cost_saved: self.total_cost_saved,
            enabled_rules: self.enabled_rules.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- helpers ------------------------------------------------------------

    fn scan(rows: u64) -> QueryNode {
        QueryNode::Scan {
            predicate: "true".to_string(),
            estimated_rows: rows,
        }
    }

    fn filter(child: QueryNode, cond: &str) -> QueryNode {
        QueryNode::Filter {
            child: Box::new(child),
            condition: cond.to_string(),
        }
    }

    fn join(left: QueryNode, right: QueryNode) -> QueryNode {
        QueryNode::Join {
            left: Box::new(left),
            right: Box::new(right),
            join_key: "id".to_string(),
        }
    }

    fn project(child: QueryNode, fields: Vec<&str>) -> QueryNode {
        QueryNode::Project {
            child: Box::new(child),
            fields: fields.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn limit(child: QueryNode, max_rows: u64) -> QueryNode {
        QueryNode::Limit {
            child: Box::new(child),
            max_rows,
        }
    }

    // --- estimated_cost tests -----------------------------------------------

    #[test]
    fn cost_scan() {
        assert_eq!(estimated_cost(&scan(500)), 500);
    }

    #[test]
    fn cost_filter() {
        // Filter costs half of child (integer division).
        let node = filter(scan(100), "x > 5");
        assert_eq!(estimated_cost(&node), 50);
    }

    #[test]
    fn cost_join() {
        // (100 * 200) / 100 + 1 = 200 + 1 = 201
        let node = join(scan(100), scan(200));
        assert_eq!(estimated_cost(&node), 201);
    }

    #[test]
    fn cost_project() {
        let node = project(scan(300), vec!["a", "b"]);
        assert_eq!(estimated_cost(&node), 300);
    }

    #[test]
    fn cost_limit() {
        let node = limit(scan(1000), 50);
        assert_eq!(estimated_cost(&node), 50);

        let node2 = limit(scan(10), 50);
        assert_eq!(estimated_cost(&node2), 10);
    }

    // --- PushFilterDown tests -----------------------------------------------

    #[test]
    fn push_filter_down_applied() {
        let plan = filter(join(scan(100), scan(200)), "a = 1");
        let mut opt = TensorQueryOptimizer::new(vec![OptimizationRule::PushFilterDown]);
        let (result, info) = opt.optimize(plan);

        // After push-down the outer Filter should be gone; the Join's left
        // child should now be a Filter.
        match result {
            QueryNode::Join { left, .. } => match *left {
                QueryNode::Filter { condition, .. } => {
                    assert_eq!(condition, "a = 1");
                }
                other => panic!("expected Filter on left, got {other:?}"),
            },
            other => panic!("expected Join at root, got {other:?}"),
        }
        assert!(info
            .rules_applied
            .contains(&OptimizationRule::PushFilterDown));
    }

    #[test]
    fn push_filter_down_no_op_on_non_join() {
        // Filter over Scan — rule should not fire.
        let plan = filter(scan(100), "a = 1");
        let mut opt = TensorQueryOptimizer::new(vec![OptimizationRule::PushFilterDown]);
        let (result, info) = opt.optimize(plan.clone());

        assert_eq!(result, plan);
        assert!(info.rules_applied.is_empty());
    }

    // --- EliminateDeadProject tests -----------------------------------------

    #[test]
    fn eliminate_dead_project_removes_empty() {
        let plan = project(scan(100), vec![]);
        let mut opt = TensorQueryOptimizer::new(vec![OptimizationRule::EliminateDeadProject]);
        let (result, info) = opt.optimize(plan);

        assert_eq!(result, scan(100));
        assert!(info
            .rules_applied
            .contains(&OptimizationRule::EliminateDeadProject));
    }

    #[test]
    fn eliminate_dead_project_keeps_non_empty() {
        let plan = project(scan(100), vec!["a", "b"]);
        let mut opt = TensorQueryOptimizer::new(vec![OptimizationRule::EliminateDeadProject]);
        let (result, info) = opt.optimize(plan.clone());

        assert_eq!(result, plan);
        assert!(info.rules_applied.is_empty());
    }

    // --- ReorderJoin tests --------------------------------------------------

    #[test]
    fn reorder_join_swaps_when_right_cheaper() {
        // left cost = 1000, right cost = 10 → should swap
        let plan = join(scan(1000), scan(10));
        let mut opt = TensorQueryOptimizer::new(vec![OptimizationRule::ReorderJoin]);
        let (result, info) = opt.optimize(plan);

        match result {
            QueryNode::Join { left, right, .. } => {
                assert_eq!(estimated_cost(&left), 10);
                assert_eq!(estimated_cost(&right), 1000);
            }
            other => panic!("expected Join, got {other:?}"),
        }
        assert!(info.rules_applied.contains(&OptimizationRule::ReorderJoin));
    }

    #[test]
    fn reorder_join_no_op_when_left_already_smaller() {
        let plan = join(scan(10), scan(1000));
        let mut opt = TensorQueryOptimizer::new(vec![OptimizationRule::ReorderJoin]);
        let (result, info) = opt.optimize(plan.clone());

        assert_eq!(result, plan);
        assert!(info.rules_applied.is_empty());
    }

    // --- FoldConstantLimit tests --------------------------------------------

    #[test]
    fn fold_constant_limit_zero_replaces_subtree() {
        let plan = limit(scan(500), 0);
        let mut opt = TensorQueryOptimizer::new(vec![OptimizationRule::FoldConstantLimit]);
        let (result, info) = opt.optimize(plan);

        assert_eq!(
            result,
            QueryNode::Scan {
                predicate: "empty".to_string(),
                estimated_rows: 0
            }
        );
        assert!(info
            .rules_applied
            .contains(&OptimizationRule::FoldConstantLimit));
    }

    #[test]
    fn fold_constant_limit_nonzero_no_op() {
        let plan = limit(scan(500), 10);
        let mut opt = TensorQueryOptimizer::new(vec![OptimizationRule::FoldConstantLimit]);
        let (result, info) = opt.optimize(plan.clone());

        assert_eq!(result, plan);
        assert!(info.rules_applied.is_empty());
    }

    // --- FlattenNestedFilter tests ------------------------------------------

    #[test]
    fn flatten_nested_filter_merges_conditions() {
        let plan = filter(filter(scan(100), "b = 2"), "a = 1");
        let mut opt = TensorQueryOptimizer::new(vec![OptimizationRule::FlattenNestedFilter]);
        let (result, info) = opt.optimize(plan);

        match result {
            QueryNode::Filter { condition, child } => {
                assert_eq!(condition, "b = 2 AND a = 1");
                assert_eq!(*child, scan(100));
            }
            other => panic!("expected Filter, got {other:?}"),
        }
        assert!(info
            .rules_applied
            .contains(&OptimizationRule::FlattenNestedFilter));
    }

    #[test]
    fn flatten_nested_filter_no_op_single_filter() {
        let plan = filter(scan(100), "a = 1");
        let mut opt = TensorQueryOptimizer::new(vec![OptimizationRule::FlattenNestedFilter]);
        let (result, info) = opt.optimize(plan.clone());

        assert_eq!(result, plan);
        assert!(info.rules_applied.is_empty());
    }

    // --- Multi-rule / convergence tests -------------------------------------

    #[test]
    fn optimize_applies_multiple_rules_in_one_pass() {
        // Plan: Project([], Filter(Join(scan(1000), scan(10)), "x=1"))
        // EliminateDeadProject removes Project
        // PushFilterDown moves Filter into Join's left child
        // ReorderJoin ensures cheaper side is left
        let plan = project(filter(join(scan(1000), scan(10)), "x = 1"), vec![]);
        let mut opt = TensorQueryOptimizer::new(vec![
            OptimizationRule::EliminateDeadProject,
            OptimizationRule::PushFilterDown,
            OptimizationRule::ReorderJoin,
        ]);
        let (_, info) = opt.optimize(plan);
        // At least two rules must have fired.
        assert!(info.rules_applied.len() >= 2);
    }

    #[test]
    fn optimize_repeats_until_stable() {
        // Deeply nested filters — FlattenNestedFilter needs multiple passes.
        let plan = filter(filter(filter(scan(100), "c = 3"), "b = 2"), "a = 1");
        let mut opt = TensorQueryOptimizer::new(vec![OptimizationRule::FlattenNestedFilter]);
        let (result, _) = opt.optimize(plan);

        // After convergence there should be exactly one Filter node.
        match result {
            QueryNode::Filter { child, .. } => {
                assert!(!matches!(*child, QueryNode::Filter { .. }));
            }
            other => panic!("expected Filter at root, got {other:?}"),
        }
    }

    // --- improvement_pct tests ----------------------------------------------

    #[test]
    fn improvement_pct_computed_correctly() {
        let result = OptimizationResult {
            original_cost: 200,
            optimized_cost: 100,
            rules_applied: vec![],
        };
        let pct = result.improvement_pct();
        assert!((pct - 50.0).abs() < 1e-9, "expected 50.0, got {pct}");
    }

    #[test]
    fn improvement_pct_zero_when_original_zero() {
        let result = OptimizationResult {
            original_cost: 0,
            optimized_cost: 0,
            rules_applied: vec![],
        };
        assert_eq!(result.improvement_pct(), 0.0);
    }

    // --- Accumulation / stats tests -----------------------------------------

    #[test]
    fn total_optimizations_increments() {
        let mut opt = TensorQueryOptimizer::new(vec![OptimizationRule::FoldConstantLimit]);
        opt.optimize(limit(scan(100), 0));
        opt.optimize(limit(scan(200), 0));
        assert_eq!(opt.total_optimizations, 2);
    }

    #[test]
    fn total_cost_saved_accumulates() {
        // Verify that total_cost_saved accumulates correctly across calls.
        // We use two calls and check that the optimizer's accumulator equals
        // the sum of (original_cost - optimized_cost) from each call.
        let mut opt = TensorQueryOptimizer::new(vec![
            OptimizationRule::EliminateDeadProject,
            OptimizationRule::FlattenNestedFilter,
        ]);
        let (_, r1) = opt.optimize(project(scan(100), vec![]));
        let (_, r2) = opt.optimize(filter(filter(scan(200), "b=2"), "a=1"));
        let expected = r1
            .original_cost
            .saturating_sub(r1.optimized_cost)
            .saturating_add(r2.original_cost.saturating_sub(r2.optimized_cost));
        assert_eq!(opt.total_cost_saved, expected);
        assert_eq!(opt.total_optimizations, 2);
    }

    #[test]
    fn stats_correct() {
        let mut opt = TensorQueryOptimizer::new(vec![
            OptimizationRule::PushFilterDown,
            OptimizationRule::ReorderJoin,
        ]);
        opt.optimize(scan(0));
        let s = opt.stats();
        assert_eq!(s.total_optimizations, 1);
        assert_eq!(s.enabled_rules, 2);
    }

    #[test]
    fn optimizer_with_empty_rules_no_ops() {
        let plan = filter(join(scan(100), scan(200)), "x = 1");
        let mut opt = TensorQueryOptimizer::new(vec![]);
        let (result, info) = opt.optimize(plan.clone());

        assert_eq!(result, plan);
        assert!(info.rules_applied.is_empty());
    }

    #[test]
    fn rules_applied_tracks_correctly() {
        let plan = limit(scan(500), 0);
        let mut opt = TensorQueryOptimizer::new(vec![
            OptimizationRule::FoldConstantLimit,
            OptimizationRule::PushFilterDown,
        ]);
        let (_, info) = opt.optimize(plan);

        assert!(info
            .rules_applied
            .contains(&OptimizationRule::FoldConstantLimit));
        assert!(!info
            .rules_applied
            .contains(&OptimizationRule::PushFilterDown));
    }

    #[test]
    fn optimize_returns_original_when_no_rules_match() {
        let plan = project(scan(42), vec!["name"]);
        let mut opt = TensorQueryOptimizer::new(vec![
            OptimizationRule::PushFilterDown,
            OptimizationRule::ReorderJoin,
        ]);
        let (result, info) = opt.optimize(plan.clone());

        assert_eq!(result, plan);
        assert!(info.rules_applied.is_empty());
    }

    #[test]
    fn limit_nonzero_not_folded() {
        let plan = limit(scan(500), 1);
        let mut opt = TensorQueryOptimizer::new(vec![OptimizationRule::FoldConstantLimit]);
        let (result, _) = opt.optimize(plan.clone());
        assert_eq!(result, plan);
    }

    // Additional edge-case tests to reach ≥22 total -------------------------

    #[test]
    fn cost_nested_filter() {
        // Filter(Filter(scan(400))) → 400/2/2 = 100
        let node = filter(filter(scan(400), "a"), "b");
        assert_eq!(estimated_cost(&node), 100);
    }

    #[test]
    fn cost_limit_zero() {
        let node = limit(scan(999), 0);
        assert_eq!(estimated_cost(&node), 0);
    }

    #[test]
    fn push_filter_down_does_not_affect_project_child() {
        // Filter(Project(scan)) — Project is not a Join, so no pushdown.
        let plan = filter(project(scan(100), vec!["x"]), "x = 1");
        let mut opt = TensorQueryOptimizer::new(vec![OptimizationRule::PushFilterDown]);
        let (result, info) = opt.optimize(plan.clone());
        assert_eq!(result, plan);
        assert!(info.rules_applied.is_empty());
    }

    #[test]
    fn reorder_join_equal_costs_no_op() {
        let plan = join(scan(100), scan(100));
        let mut opt = TensorQueryOptimizer::new(vec![OptimizationRule::ReorderJoin]);
        let (result, info) = opt.optimize(plan.clone());
        assert_eq!(result, plan);
        assert!(info.rules_applied.is_empty());
    }

    #[test]
    fn flatten_then_push_filter_down_combined() {
        // FlattenNestedFilter then PushFilterDown
        let plan = filter(filter(join(scan(100), scan(200)), "b = 2"), "a = 1");
        let mut opt = TensorQueryOptimizer::new(vec![
            OptimizationRule::FlattenNestedFilter,
            OptimizationRule::PushFilterDown,
        ]);
        let (result, info) = opt.optimize(plan);

        // End result: Join with merged-filter on left side.
        match result {
            QueryNode::Join { left, .. } => match *left {
                QueryNode::Filter { condition, .. } => {
                    assert!(condition.contains("AND"), "condition was: {condition}");
                }
                other => panic!("expected Filter on left, got {other:?}"),
            },
            other => panic!("expected Join at root, got {other:?}"),
        }
        assert!(info
            .rules_applied
            .contains(&OptimizationRule::FlattenNestedFilter));
        assert!(info
            .rules_applied
            .contains(&OptimizationRule::PushFilterDown));
    }

    #[test]
    fn improvement_pct_no_improvement() {
        let result = OptimizationResult {
            original_cost: 100,
            optimized_cost: 100,
            rules_applied: vec![],
        };
        assert_eq!(result.improvement_pct(), 0.0);
    }

    #[test]
    fn improvement_pct_full_elimination() {
        // optimized_cost = 0 → 100% improvement
        let result = OptimizationResult {
            original_cost: 100,
            optimized_cost: 0,
            rules_applied: vec![],
        };
        assert!((result.improvement_pct() - 100.0).abs() < 1e-9);
    }
}
