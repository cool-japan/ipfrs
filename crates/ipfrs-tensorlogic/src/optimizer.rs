//! Query optimization for TensorLogic
//!
//! Optimizes logical queries by:
//! - Reordering predicates in rule bodies for better performance
//! - Selecting optimal join orders
//! - Estimating predicate selectivity
//! - Cost-based query planning
//! - Cardinality estimation
//! - Statistics tracking
//! - Materialized views for common queries

use crate::ir::{KnowledgeBase, Predicate, Rule, Term};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Statistics for a single predicate
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PredicateStats {
    /// Number of facts for this predicate
    pub fact_count: usize,
    /// Number of rules with this predicate as head
    pub rule_count: usize,
    /// Average arity (number of arguments)
    pub avg_arity: f64,
    /// Estimated cardinality after filtering
    pub estimated_cardinality: f64,
    /// Selectivity (0.0 = highly selective, 1.0 = not selective)
    pub selectivity: f64,
}

impl PredicateStats {
    /// Create new stats
    pub fn new(fact_count: usize, rule_count: usize, avg_arity: f64) -> Self {
        Self {
            fact_count,
            rule_count,
            avg_arity,
            estimated_cardinality: fact_count as f64,
            selectivity: 1.0,
        }
    }

    /// Compute selectivity based on total facts
    #[inline]
    pub fn compute_selectivity(&mut self, total_facts: usize) {
        if total_facts == 0 {
            self.selectivity = 1.0;
        } else {
            self.selectivity = self.fact_count as f64 / total_facts as f64;
        }
    }
}

/// Query plan node representing a single operation
#[derive(Debug, Clone)]
pub enum PlanNode {
    /// Scan a predicate (fact lookup)
    Scan {
        predicate: String,
        bound_vars: Vec<String>,
        estimated_rows: f64,
    },
    /// Join two plans
    Join {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
        join_vars: Vec<String>,
        estimated_rows: f64,
    },
    /// Filter results
    Filter {
        input: Box<PlanNode>,
        condition: Predicate,
        estimated_rows: f64,
    },
}

impl PlanNode {
    /// Get estimated row count
    #[inline]
    pub fn estimated_rows(&self) -> f64 {
        match self {
            PlanNode::Scan { estimated_rows, .. } => *estimated_rows,
            PlanNode::Join { estimated_rows, .. } => *estimated_rows,
            PlanNode::Filter { estimated_rows, .. } => *estimated_rows,
        }
    }

    /// Compute cost of this plan
    pub fn cost(&self) -> f64 {
        match self {
            PlanNode::Scan { estimated_rows, .. } => *estimated_rows,
            PlanNode::Join {
                left,
                right,
                estimated_rows,
                ..
            } => left.cost() + right.cost() + *estimated_rows,
            PlanNode::Filter {
                input,
                estimated_rows,
                ..
            } => input.cost() + *estimated_rows * 0.1,
        }
    }
}

/// Query plan for a goal
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Root of the plan tree
    pub root: PlanNode,
    /// Total estimated cost
    pub estimated_cost: f64,
    /// Estimated result cardinality
    pub estimated_rows: f64,
    /// Variables that will be bound
    pub output_vars: Vec<String>,
}

impl QueryPlan {
    /// Create a new query plan
    pub fn new(root: PlanNode) -> Self {
        let estimated_cost = root.cost();
        let estimated_rows = root.estimated_rows();
        Self {
            root,
            estimated_cost,
            estimated_rows,
            output_vars: Vec::new(),
        }
    }

    /// Create with output variables
    pub fn with_vars(root: PlanNode, output_vars: Vec<String>) -> Self {
        let estimated_cost = root.cost();
        let estimated_rows = root.estimated_rows();
        Self {
            root,
            estimated_cost,
            estimated_rows,
            output_vars,
        }
    }
}

/// Query optimizer for TensorLogic
pub struct QueryOptimizer {
    /// Statistics about predicates
    predicate_stats: HashMap<String, PredicateStats>,
    /// Total facts in knowledge base
    total_facts: usize,
    /// Selectivity cache (for backwards compatibility)
    selectivity_cache: HashMap<String, f64>,
}

impl QueryOptimizer {
    /// Create a new query optimizer
    #[inline]
    pub fn new() -> Self {
        Self {
            predicate_stats: HashMap::new(),
            total_facts: 0,
            selectivity_cache: HashMap::new(),
        }
    }

    /// Create a query plan for a conjunction of goals
    pub fn plan_query(&self, goals: &[Predicate], kb: &KnowledgeBase) -> QueryPlan {
        if goals.is_empty() {
            return QueryPlan::new(PlanNode::Scan {
                predicate: "empty".to_string(),
                bound_vars: Vec::new(),
                estimated_rows: 0.0,
            });
        }

        if goals.len() == 1 {
            return self.plan_single_goal(&goals[0], kb);
        }

        // Order goals by selectivity
        let ordered = self.optimize_goal(goals.to_vec(), kb);

        // Build join plan
        let mut current_plan = self.plan_single_goal(&ordered[0], kb);

        for goal in ordered.iter().skip(1) {
            let right_plan = self.plan_single_goal(goal, kb);

            // Find join variables
            let join_vars = self.find_join_vars(&current_plan, &right_plan, goal);

            // Estimate join cardinality
            let estimated_rows = self.estimate_join_cardinality(
                current_plan.estimated_rows,
                right_plan.estimated_rows,
                &join_vars,
            );

            current_plan = QueryPlan::new(PlanNode::Join {
                left: Box::new(current_plan.root),
                right: Box::new(right_plan.root),
                join_vars,
                estimated_rows,
            });
        }

        current_plan
    }

    /// Plan a single goal
    fn plan_single_goal(&self, goal: &Predicate, kb: &KnowledgeBase) -> QueryPlan {
        let fact_count = kb.get_predicates(&goal.name).len();
        let groundness = self.compute_groundness(goal);

        // Estimate rows based on fact count and groundness
        let estimated_rows = if groundness >= 1.0 {
            1.0 // Fully ground query returns at most 1 result
        } else {
            fact_count as f64 * (1.0 - groundness + 0.1)
        };

        let bound_vars: Vec<String> = goal
            .args
            .iter()
            .filter_map(|t| {
                if let Term::Var(v) = t {
                    Some(v.clone())
                } else {
                    None
                }
            })
            .collect();

        QueryPlan::with_vars(
            PlanNode::Scan {
                predicate: goal.name.clone(),
                bound_vars: bound_vars.clone(),
                estimated_rows,
            },
            bound_vars,
        )
    }

    /// Find variables that join two plans
    fn find_join_vars(
        &self,
        left: &QueryPlan,
        _right: &QueryPlan,
        right_goal: &Predicate,
    ) -> Vec<String> {
        let mut join_vars = Vec::new();
        for var in &left.output_vars {
            for arg in &right_goal.args {
                if let Term::Var(v) = arg {
                    if v == var {
                        join_vars.push(var.clone());
                    }
                }
            }
        }
        join_vars
    }

    /// Estimate cardinality of a join
    fn estimate_join_cardinality(
        &self,
        left_rows: f64,
        right_rows: f64,
        join_vars: &[String],
    ) -> f64 {
        if join_vars.is_empty() {
            // Cross product
            left_rows * right_rows
        } else {
            // Estimated selectivity based on join
            let selectivity = 0.1_f64.powi(join_vars.len() as i32);
            (left_rows * right_rows * selectivity).max(1.0)
        }
    }

    /// Get predicate statistics
    #[inline]
    pub fn get_stats(&self, predicate_name: &str) -> Option<&PredicateStats> {
        self.predicate_stats.get(predicate_name)
    }

    /// Get all statistics
    #[inline]
    pub fn all_stats(&self) -> &HashMap<String, PredicateStats> {
        &self.predicate_stats
    }

    /// Estimate cardinality for a predicate
    pub fn estimate_cardinality(&self, predicate: &Predicate, kb: &KnowledgeBase) -> f64 {
        let fact_count = kb.get_predicates(&predicate.name).len() as f64;
        let groundness = self.compute_groundness(predicate);

        // More ground args = lower cardinality
        fact_count * (1.0 - groundness + 0.1)
    }

    /// Optimize a rule by reordering its body predicates
    ///
    /// Reorders predicates to put more selective ones first,
    /// reducing the intermediate result set size
    pub fn optimize_rule(&self, rule: &Rule, kb: &KnowledgeBase) -> Rule {
        if rule.body.is_empty() {
            return rule.clone();
        }

        let body = rule.body.clone();

        // Compute selectivity scores for each predicate
        let mut scores: Vec<(usize, f64)> = body
            .iter()
            .enumerate()
            .map(|(i, pred)| (i, self.estimate_selectivity(pred, kb)))
            .collect();

        // Sort by selectivity (most selective first)
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Reorder body predicates
        let optimized_body: Vec<Predicate> = scores.iter().map(|(i, _)| body[*i].clone()).collect();

        Rule::new(rule.head.clone(), optimized_body)
    }

    /// Estimate selectivity of a predicate
    ///
    /// Returns a value where LOWER is more selective (should execute first)
    /// Higher values indicate less selective predicates
    fn estimate_selectivity(&self, predicate: &Predicate, kb: &KnowledgeBase) -> f64 {
        // Check cache first
        if let Some(&selectivity) = self.selectivity_cache.get(&predicate.name) {
            return selectivity;
        }

        // Count facts with this predicate name
        let fact_count = kb.get_predicates(&predicate.name).len();

        // Estimate based on groundness and fact count
        let groundness = self.compute_groundness(predicate);

        // More facts = less selective (higher score)
        // More ground = more selective (lower score)
        let fact_factor = if fact_count == 0 {
            100.0 // Unknown predicates assumed least selective
        } else {
            fact_count as f64
        };

        // Combine: less ground + more facts = higher (less selective) score
        fact_factor * (1.0 - groundness + 0.1)
    }

    /// Compute how "ground" a predicate is (0.0 = all variables, 1.0 = all constants)
    #[inline]
    fn compute_groundness(&self, predicate: &Predicate) -> f64 {
        if predicate.args.is_empty() {
            return 1.0;
        }

        let ground_count = predicate.args.iter().filter(|t| t.is_ground()).count();
        ground_count as f64 / predicate.args.len() as f64
    }

    /// Update selectivity statistics from a knowledge base
    pub fn update_statistics(&mut self, kb: &KnowledgeBase) {
        // Clear old stats
        self.selectivity_cache.clear();
        self.predicate_stats.clear();
        self.total_facts = kb.facts.len();

        // Count facts by predicate name
        let mut fact_counts: HashMap<String, usize> = HashMap::new();
        let mut arity_sums: HashMap<String, usize> = HashMap::new();

        for fact in &kb.facts {
            *fact_counts.entry(fact.name.clone()).or_insert(0) += 1;
            *arity_sums.entry(fact.name.clone()).or_insert(0) += fact.args.len();
        }

        // Count rules by head predicate
        let mut rule_counts: HashMap<String, usize> = HashMap::new();
        for rule in &kb.rules {
            *rule_counts.entry(rule.head.name.clone()).or_insert(0) += 1;
        }

        let total_facts = kb.facts.len() as f64;
        if total_facts == 0.0 {
            return;
        }

        // Build predicate stats
        let all_predicates: std::collections::HashSet<_> = fact_counts
            .keys()
            .chain(rule_counts.keys())
            .cloned()
            .collect();

        for name in all_predicates {
            let fact_count = *fact_counts.get(&name).unwrap_or(&0);
            let rule_count = *rule_counts.get(&name).unwrap_or(&0);
            let arity_sum = *arity_sums.get(&name).unwrap_or(&0);
            let avg_arity = if fact_count > 0 {
                arity_sum as f64 / fact_count as f64
            } else {
                0.0
            };

            let mut stats = PredicateStats::new(fact_count, rule_count, avg_arity);
            stats.compute_selectivity(self.total_facts);

            // Also update selectivity cache for backwards compatibility
            self.selectivity_cache
                .insert(name.clone(), stats.selectivity);
            self.predicate_stats.insert(name, stats);
        }
    }

    /// Get total fact count
    #[inline]
    pub fn total_facts(&self) -> usize {
        self.total_facts
    }

    /// Optimize a query goal
    ///
    /// For complex goals with multiple predicates, reorder them optimally
    pub fn optimize_goal(&self, goals: Vec<Predicate>, kb: &KnowledgeBase) -> Vec<Predicate> {
        if goals.len() <= 1 {
            return goals;
        }

        let mut scored: Vec<(Predicate, f64)> = goals
            .into_iter()
            .map(|p| {
                let score = self.estimate_selectivity(&p, kb);
                (p, score)
            })
            .collect();

        // Sort by selectivity (most selective first)
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter().map(|(p, _)| p).collect()
    }

    /// Get optimization recommendations for a knowledge base
    pub fn get_recommendations(&self, kb: &KnowledgeBase) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        for rule in &kb.rules {
            if rule.body.len() > 1 {
                let optimized = self.optimize_rule(rule, kb);

                // Check if order changed
                let changed = rule
                    .body
                    .iter()
                    .zip(optimized.body.iter())
                    .any(|(a, b)| a.name != b.name);

                if changed {
                    recommendations.push(OptimizationRecommendation {
                        rule_head: rule.head.name.clone(),
                        original_order: rule.body.iter().map(|p| p.name.clone()).collect(),
                        optimized_order: optimized.body.iter().map(|p| p.name.clone()).collect(),
                        estimated_improvement: 0.5, // Simplified estimate
                    });
                }
            }
        }

        recommendations
    }
}

impl Default for QueryOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Name of the rule head
    pub rule_head: String,
    /// Original predicate order
    pub original_order: Vec<String>,
    /// Optimized predicate order
    pub optimized_order: Vec<String>,
    /// Estimated improvement factor (higher = better)
    pub estimated_improvement: f64,
}

/// Materialized view metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterializedView {
    /// Unique view name
    pub name: String,
    /// Query pattern that defines this view
    pub query: Vec<Predicate>,
    /// Precomputed results
    pub results: Vec<Vec<Term>>,
    /// Time when the view was created/refreshed
    pub last_refresh: SystemTime,
    /// Time-to-live before refresh needed
    pub ttl: Option<Duration>,
    /// Statistics about view usage
    pub access_count: usize,
    /// Total cost saved by using this view
    pub total_cost_saved: f64,
}

impl MaterializedView {
    /// Create a new materialized view
    pub fn new(name: String, query: Vec<Predicate>) -> Self {
        Self {
            name,
            query,
            results: Vec::new(),
            last_refresh: SystemTime::now(),
            ttl: None,
            access_count: 0,
            total_cost_saved: 0.0,
        }
    }

    /// Create with TTL
    pub fn with_ttl(name: String, query: Vec<Predicate>, ttl: Duration) -> Self {
        Self {
            name,
            query,
            results: Vec::new(),
            last_refresh: SystemTime::now(),
            ttl: Some(ttl),
            access_count: 0,
            total_cost_saved: 0.0,
        }
    }

    /// Check if view needs refresh based on TTL
    pub fn needs_refresh(&self) -> bool {
        if let Some(ttl) = self.ttl {
            if let Ok(elapsed) = self.last_refresh.elapsed() {
                return elapsed > ttl;
            }
        }
        false
    }

    /// Refresh the view with new results
    pub fn refresh(&mut self, results: Vec<Vec<Term>>) {
        self.results = results;
        self.last_refresh = SystemTime::now();
    }

    /// Record a view access
    #[inline]
    pub fn record_access(&mut self, cost_saved: f64) {
        self.access_count += 1;
        self.total_cost_saved += cost_saved;
    }

    /// Check if query matches this view
    pub fn matches_query(&self, query: &[Predicate]) -> bool {
        if self.query.len() != query.len() {
            return false;
        }

        self.query
            .iter()
            .zip(query.iter())
            .all(|(a, b)| a.name == b.name && a.args.len() == b.args.len())
    }
}

/// Materialized view manager
pub struct MaterializedViewManager {
    /// All materialized views
    views: HashMap<String, MaterializedView>,
    /// Max number of views to maintain
    max_views: usize,
    /// Minimum access count to keep a view
    min_access_threshold: usize,
}

impl MaterializedViewManager {
    /// Create a new view manager
    pub fn new(max_views: usize) -> Self {
        Self {
            views: HashMap::new(),
            max_views,
            min_access_threshold: 5,
        }
    }

    /// Create a materialized view
    pub fn create_view(
        &mut self,
        name: String,
        query: Vec<Predicate>,
        ttl: Option<Duration>,
    ) -> Result<(), String> {
        if self.views.contains_key(&name) {
            return Err(format!("View '{}' already exists", name));
        }

        // Enforce max views limit
        if self.views.len() >= self.max_views {
            self.evict_least_useful_view();
        }

        let view = if let Some(ttl) = ttl {
            MaterializedView::with_ttl(name.clone(), query, ttl)
        } else {
            MaterializedView::new(name.clone(), query)
        };

        self.views.insert(name, view);
        Ok(())
    }

    /// Drop a materialized view
    pub fn drop_view(&mut self, name: &str) -> Result<(), String> {
        if self.views.remove(name).is_none() {
            return Err(format!("View '{}' does not exist", name));
        }
        Ok(())
    }

    /// Refresh a view with new results
    pub fn refresh_view(&mut self, name: &str, results: Vec<Vec<Term>>) -> Result<(), String> {
        let view = self
            .views
            .get_mut(name)
            .ok_or_else(|| format!("View '{}' does not exist", name))?;
        view.refresh(results);
        Ok(())
    }

    /// Find a view that matches the query
    pub fn find_matching_view(&mut self, query: &[Predicate]) -> Option<&mut MaterializedView> {
        self.views
            .values_mut()
            .find(|view| view.matches_query(query))
    }

    /// Get a view by name
    #[inline]
    pub fn get_view(&self, name: &str) -> Option<&MaterializedView> {
        self.views.get(name)
    }

    /// Get a mutable view by name
    #[inline]
    pub fn get_view_mut(&mut self, name: &str) -> Option<&mut MaterializedView> {
        self.views.get_mut(name)
    }

    /// Get all views
    #[inline]
    pub fn all_views(&self) -> &HashMap<String, MaterializedView> {
        &self.views
    }

    /// Evict the least useful view
    fn evict_least_useful_view(&mut self) {
        if self.views.is_empty() {
            return;
        }

        // Find view with lowest utility score
        let mut min_score = f64::INFINITY;
        let mut evict_name: Option<String> = None;

        for (name, view) in &self.views {
            // Utility score: cost saved per access
            let score = if view.access_count > 0 {
                view.total_cost_saved / view.access_count as f64
            } else {
                0.0
            };

            if score < min_score {
                min_score = score;
                evict_name = Some(name.clone());
            }
        }

        if let Some(name) = evict_name {
            self.views.remove(&name);
        }
    }

    /// Clean up stale views (based on TTL and access count)
    pub fn cleanup_stale_views(&mut self) {
        let to_remove: Vec<String> = self
            .views
            .iter()
            .filter(|(_, view)| {
                view.needs_refresh() || view.access_count < self.min_access_threshold
            })
            .map(|(name, _)| name.clone())
            .collect();

        for name in to_remove {
            self.views.remove(&name);
        }
    }

    /// Get view usage statistics
    pub fn get_statistics(&self) -> ViewStatistics {
        let total_views = self.views.len();
        let total_accesses: usize = self.views.values().map(|v| v.access_count).sum();
        let total_cost_saved: f64 = self.views.values().map(|v| v.total_cost_saved).sum();

        let avg_access_count = if total_views > 0 {
            total_accesses as f64 / total_views as f64
        } else {
            0.0
        };

        ViewStatistics {
            total_views,
            total_accesses,
            total_cost_saved,
            avg_access_count,
        }
    }

    /// Set minimum access threshold for view retention
    #[inline]
    pub fn set_min_access_threshold(&mut self, threshold: usize) {
        self.min_access_threshold = threshold;
    }
}

impl Default for MaterializedViewManager {
    fn default() -> Self {
        Self::new(100)
    }
}

/// View usage statistics
#[derive(Debug, Clone)]
pub struct ViewStatistics {
    /// Total number of views
    pub total_views: usize,
    /// Total number of view accesses
    pub total_accesses: usize,
    /// Total cost saved by using views
    pub total_cost_saved: f64,
    /// Average access count per view
    pub avg_access_count: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Constant, Term};

    #[test]
    fn test_groundness() {
        let optimizer = QueryOptimizer::new();

        // All constants
        let pred1 = Predicate::new(
            "test".to_string(),
            vec![
                Term::Const(Constant::String("a".to_string())),
                Term::Const(Constant::String("b".to_string())),
            ],
        );
        assert_eq!(optimizer.compute_groundness(&pred1), 1.0);

        // All variables
        let pred2 = Predicate::new(
            "test".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        );
        assert_eq!(optimizer.compute_groundness(&pred2), 0.0);

        // Mixed
        let pred3 = Predicate::new(
            "test".to_string(),
            vec![
                Term::Const(Constant::String("a".to_string())),
                Term::Var("Y".to_string()),
            ],
        );
        assert_eq!(optimizer.compute_groundness(&pred3), 0.5);
    }

    #[test]
    fn test_optimize_rule() {
        let optimizer = QueryOptimizer::new();
        let mut kb = KnowledgeBase::new();

        // Add some facts to influence selectivity
        kb.add_fact(Predicate::new(
            "rare".to_string(),
            vec![
                Term::Const(Constant::String("a".to_string())),
                Term::Const(Constant::String("b".to_string())),
            ],
        ));

        for i in 0..100 {
            kb.add_fact(Predicate::new(
                "common".to_string(),
                vec![
                    Term::Const(Constant::Int(i)),
                    Term::Const(Constant::Int(i + 1)),
                ],
            ));
        }

        // Rule with predicates in suboptimal order
        let rule = Rule::new(
            Predicate::new("result".to_string(), vec![Term::Var("X".to_string())]),
            vec![
                Predicate::new(
                    "common".to_string(),
                    vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
                ),
                Predicate::new(
                    "rare".to_string(),
                    vec![Term::Var("Y".to_string()), Term::Var("Z".to_string())],
                ),
            ],
        );

        let optimized = optimizer.optimize_rule(&rule, &kb);

        // The optimizer should put 'rare' before 'common' since it's more selective
        assert_eq!(optimized.body[0].name, "rare");
        assert_eq!(optimized.body[1].name, "common");
    }

    #[test]
    fn test_update_statistics() {
        let mut optimizer = QueryOptimizer::new();
        let mut kb = KnowledgeBase::new();

        // Add facts
        for i in 0..10 {
            kb.add_fact(Predicate::new(
                "parent".to_string(),
                vec![
                    Term::Const(Constant::Int(i)),
                    Term::Const(Constant::Int(i + 1)),
                ],
            ));
        }

        for i in 0..5 {
            kb.add_fact(Predicate::new(
                "child".to_string(),
                vec![
                    Term::Const(Constant::Int(i)),
                    Term::Const(Constant::Int(i + 1)),
                ],
            ));
        }

        optimizer.update_statistics(&kb);

        // Check stats
        assert_eq!(optimizer.total_facts(), 15);

        let parent_stats = optimizer.get_stats("parent").expect("test: should succeed");
        assert_eq!(parent_stats.fact_count, 10);
        assert!((parent_stats.selectivity - (10.0 / 15.0)).abs() < 0.001);

        let child_stats = optimizer.get_stats("child").expect("test: should succeed");
        assert_eq!(child_stats.fact_count, 5);
        assert!((child_stats.selectivity - (5.0 / 15.0)).abs() < 0.001);
    }

    #[test]
    fn test_query_plan_single() {
        let optimizer = QueryOptimizer::new();
        let mut kb = KnowledgeBase::new();

        for i in 0..100 {
            kb.add_fact(Predicate::new(
                "test".to_string(),
                vec![
                    Term::Const(Constant::Int(i)),
                    Term::Const(Constant::Int(i * 2)),
                ],
            ));
        }

        let goal = Predicate::new(
            "test".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        );

        let plan = optimizer.plan_query(&[goal], &kb);

        // Should be a scan node
        matches!(plan.root, PlanNode::Scan { .. });
        assert!(plan.estimated_rows > 0.0);
    }

    #[test]
    fn test_query_plan_join() {
        let optimizer = QueryOptimizer::new();
        let mut kb = KnowledgeBase::new();

        for i in 0..10 {
            kb.add_fact(Predicate::new(
                "parent".to_string(),
                vec![
                    Term::Const(Constant::String(format!("p{}", i))),
                    Term::Const(Constant::String(format!("c{}", i))),
                ],
            ));
            kb.add_fact(Predicate::new(
                "likes".to_string(),
                vec![
                    Term::Const(Constant::String(format!("c{}", i))),
                    Term::Const(Constant::String("pizza".to_string())),
                ],
            ));
        }

        let goals = vec![
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            Predicate::new(
                "likes".to_string(),
                vec![
                    Term::Var("Y".to_string()),
                    Term::Const(Constant::String("pizza".to_string())),
                ],
            ),
        ];

        let plan = optimizer.plan_query(&goals, &kb);

        // Should have a join node
        assert!(plan.estimated_cost > 0.0);
    }

    #[test]
    fn test_predicate_stats() {
        let mut stats = PredicateStats::new(100, 5, 2.5);
        assert_eq!(stats.fact_count, 100);
        assert_eq!(stats.rule_count, 5);
        assert!((stats.avg_arity - 2.5).abs() < 0.001);

        stats.compute_selectivity(1000);
        assert!((stats.selectivity - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_plan_node_cost() {
        let scan = PlanNode::Scan {
            predicate: "test".to_string(),
            bound_vars: vec!["X".to_string()],
            estimated_rows: 100.0,
        };

        assert!((scan.cost() - 100.0).abs() < 0.001);
        assert!((scan.estimated_rows() - 100.0).abs() < 0.001);

        let join = PlanNode::Join {
            left: Box::new(scan.clone()),
            right: Box::new(PlanNode::Scan {
                predicate: "other".to_string(),
                bound_vars: vec!["Y".to_string()],
                estimated_rows: 50.0,
            }),
            join_vars: vec!["X".to_string()],
            estimated_rows: 10.0,
        };

        // Cost should include both scans plus join result
        assert!(join.cost() > 150.0);
    }

    #[test]
    fn test_materialized_view_basic() {
        let query = vec![Predicate::new(
            "parent".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        )];

        let mut view = MaterializedView::new("parent_view".to_string(), query.clone());
        assert_eq!(view.name, "parent_view");
        assert_eq!(view.query.len(), 1);
        assert_eq!(view.results.len(), 0);
        assert_eq!(view.access_count, 0);

        // Add results
        let results = vec![
            vec![
                Term::Const(Constant::String("alice".to_string())),
                Term::Const(Constant::String("bob".to_string())),
            ],
            vec![
                Term::Const(Constant::String("bob".to_string())),
                Term::Const(Constant::String("charlie".to_string())),
            ],
        ];
        view.refresh(results.clone());
        assert_eq!(view.results.len(), 2);

        // Record access
        view.record_access(10.0);
        assert_eq!(view.access_count, 1);
        assert!((view.total_cost_saved - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_materialized_view_ttl() {
        use std::thread;

        let query = vec![Predicate::new(
            "test".to_string(),
            vec![Term::Var("X".to_string())],
        )];

        let ttl = Duration::from_millis(10);
        let view = MaterializedView::with_ttl("test_view".to_string(), query, ttl);

        assert!(!view.needs_refresh());

        // Wait for TTL to expire
        thread::sleep(Duration::from_millis(20));
        assert!(view.needs_refresh());
    }

    #[test]
    fn test_materialized_view_matches_query() {
        let query1 = vec![
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            Predicate::new(
                "likes".to_string(),
                vec![Term::Var("Y".to_string()), Term::Var("Z".to_string())],
            ),
        ];

        let view = MaterializedView::new("view1".to_string(), query1.clone());

        // Same query should match
        assert!(view.matches_query(&query1));

        // Different query should not match
        let query2 = vec![Predicate::new(
            "parent".to_string(),
            vec![Term::Var("A".to_string()), Term::Var("B".to_string())],
        )];
        assert!(!view.matches_query(&query2));
    }

    #[test]
    fn test_view_manager_create_drop() {
        let mut manager = MaterializedViewManager::new(10);

        let query = vec![Predicate::new(
            "test".to_string(),
            vec![Term::Var("X".to_string())],
        )];

        // Create view
        assert!(manager
            .create_view("view1".to_string(), query.clone(), None)
            .is_ok());
        assert_eq!(manager.all_views().len(), 1);

        // Create duplicate should fail
        assert!(manager
            .create_view("view1".to_string(), query, None)
            .is_err());

        // Drop view
        assert!(manager.drop_view("view1").is_ok());
        assert_eq!(manager.all_views().len(), 0);

        // Drop non-existent view should fail
        assert!(manager.drop_view("view1").is_err());
    }

    #[test]
    fn test_view_manager_refresh() {
        let mut manager = MaterializedViewManager::new(10);

        let query = vec![Predicate::new(
            "test".to_string(),
            vec![Term::Var("X".to_string())],
        )];

        manager
            .create_view("view1".to_string(), query, None)
            .expect("test: should succeed");

        let results = vec![vec![Term::Const(Constant::Int(1))]];

        assert!(manager.refresh_view("view1", results.clone()).is_ok());

        let view = manager.get_view("view1").expect("test: should succeed");
        assert_eq!(view.results.len(), 1);
    }

    #[test]
    fn test_view_manager_find_matching() {
        let mut manager = MaterializedViewManager::new(10);

        let query1 = vec![Predicate::new(
            "parent".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        )];

        manager
            .create_view("parent_view".to_string(), query1.clone(), None)
            .expect("test: should succeed");

        // Should find matching view
        let found = manager.find_matching_view(&query1);
        assert!(found.is_some());
        assert_eq!(found.expect("test: should succeed").name, "parent_view");

        // Should not find non-matching view
        let query2 = vec![Predicate::new(
            "likes".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        )];
        let not_found = manager.find_matching_view(&query2);
        assert!(not_found.is_none());
    }

    #[test]
    fn test_view_manager_eviction() {
        let mut manager = MaterializedViewManager::new(3);

        // Create 3 views
        for i in 0..3 {
            let query = vec![Predicate::new(
                format!("pred{}", i),
                vec![Term::Var("X".to_string())],
            )];
            manager
                .create_view(format!("view{}", i), query, None)
                .expect("test: should succeed");
        }

        assert_eq!(manager.all_views().len(), 3);

        // Record different access counts
        if let Some(view) = manager.get_view_mut("view0") {
            view.record_access(100.0);
        }
        if let Some(view) = manager.get_view_mut("view1") {
            view.record_access(50.0);
        }
        // view2 has no accesses

        // Create one more view - should evict view2 (lowest utility)
        let query = vec![Predicate::new(
            "pred3".to_string(),
            vec![Term::Var("X".to_string())],
        )];
        manager
            .create_view("view3".to_string(), query, None)
            .expect("test: should succeed");

        assert_eq!(manager.all_views().len(), 3);
        assert!(manager.get_view("view2").is_none()); // view2 should be evicted
        assert!(manager.get_view("view0").is_some());
        assert!(manager.get_view("view1").is_some());
        assert!(manager.get_view("view3").is_some());
    }

    #[test]
    fn test_view_manager_cleanup_stale() {
        use std::thread;

        let mut manager = MaterializedViewManager::new(10);
        manager.set_min_access_threshold(5);

        // Create view with TTL
        let query1 = vec![Predicate::new(
            "test1".to_string(),
            vec![Term::Var("X".to_string())],
        )];
        manager
            .create_view("view1".to_string(), query1, Some(Duration::from_millis(10)))
            .expect("test: should succeed");

        // Create view with low access count
        let query2 = vec![Predicate::new(
            "test2".to_string(),
            vec![Term::Var("X".to_string())],
        )];
        manager
            .create_view("view2".to_string(), query2, None)
            .expect("test: should succeed");

        // Create view with high access count
        let query3 = vec![Predicate::new(
            "test3".to_string(),
            vec![Term::Var("X".to_string())],
        )];
        manager
            .create_view("view3".to_string(), query3, None)
            .expect("test: should succeed");

        if let Some(view) = manager.get_view_mut("view3") {
            for _ in 0..10 {
                view.record_access(1.0);
            }
        }

        // Wait for TTL to expire
        thread::sleep(Duration::from_millis(20));

        manager.cleanup_stale_views();

        // view1 should be removed (TTL expired)
        // view2 should be removed (low access count)
        // view3 should remain (high access count)
        assert!(manager.get_view("view1").is_none());
        assert!(manager.get_view("view2").is_none());
        assert!(manager.get_view("view3").is_some());
    }

    #[test]
    fn test_view_statistics() {
        let mut manager = MaterializedViewManager::new(10);

        // Create views with different access patterns
        for i in 0..3 {
            let query = vec![Predicate::new(
                format!("pred{}", i),
                vec![Term::Var("X".to_string())],
            )];
            manager
                .create_view(format!("view{}", i), query, None)
                .expect("test: should succeed");

            if let Some(view) = manager.get_view_mut(&format!("view{}", i)) {
                for _ in 0..((i + 1) * 5) {
                    view.record_access(10.0);
                }
            }
        }

        let stats = manager.get_statistics();
        assert_eq!(stats.total_views, 3);
        assert_eq!(stats.total_accesses, 30); // 5 + 10 + 15
        assert!((stats.total_cost_saved - 300.0).abs() < 0.001); // 30 * 10.0
        assert!((stats.avg_access_count - 10.0).abs() < 0.001); // 30 / 3
    }
}
