//! Knowledge Base Query Language
//!
//! This module provides a SPARQL-like query language for semantic knowledge bases:
//! - Triple pattern matching for graph queries
//! - Pattern matching for logic terms with wildcards
//! - Query optimization (join order, filter pushdown)
//! - Complex boolean queries (AND/OR/NOT)

use ipfrs_core::Result;
use ipfrs_tensorlogic::{KnowledgeBase, Predicate, Term};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Query pattern for matching predicates
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QueryPattern {
    /// Exact predicate match
    Exact(Predicate),
    /// Wildcard pattern (name, args with wildcards)
    Pattern {
        name: Option<String>,
        args: Vec<TermPattern>,
    },
    /// Variable binding
    Variable(String),
}

/// Pattern for matching terms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TermPattern {
    /// Exact term match
    Exact(Term),
    /// Wildcard (matches any term)
    Wildcard,
    /// Variable (binds to matched term)
    Variable(String),
    /// Type constraint (e.g., must be constant)
    TypeConstraint(TermType),
}

/// Term type for type constraints
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TermType {
    Var,
    Const,
    Fun,
    Ref,
}

/// Boolean query operators
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BooleanQuery {
    /// Conjunction (AND)
    And(Vec<Query>),
    /// Disjunction (OR)
    Or(Vec<Query>),
    /// Negation (NOT)
    Not(Box<Query>),
    /// Atomic query
    Atom(Query),
}

/// Query filter expressions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FilterExpr {
    /// Equality comparison
    Equals(String, String),
    /// Inequality
    NotEquals(String, String),
    /// Regex match on variable
    Regex(String, String),
    /// Type check
    IsType(String, TermType),
    /// Conjunction of filters
    And(Vec<FilterExpr>),
    /// Disjunction of filters
    Or(Vec<FilterExpr>),
}

/// A query for the knowledge base
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Query {
    /// SELECT clause - variables to return
    pub select: Vec<String>,
    /// WHERE clause - patterns to match
    pub patterns: Vec<QueryPattern>,
    /// FILTER clause - filter expressions
    pub filters: Vec<FilterExpr>,
    /// LIMIT - maximum results
    pub limit: Option<usize>,
    /// OFFSET - skip first N results
    pub offset: Option<usize>,
}

impl Query {
    /// Create a new query
    pub fn new() -> Self {
        Self {
            select: Vec::new(),
            patterns: Vec::new(),
            filters: Vec::new(),
            limit: None,
            offset: None,
        }
    }

    /// Add a SELECT variable
    pub fn select(mut self, var: impl Into<String>) -> Self {
        self.select.push(var.into());
        self
    }

    /// Add a WHERE pattern
    pub fn where_pattern(mut self, pattern: QueryPattern) -> Self {
        self.patterns.push(pattern);
        self
    }

    /// Add a FILTER expression
    pub fn filter(mut self, expr: FilterExpr) -> Self {
        self.filters.push(expr);
        self
    }

    /// Set LIMIT
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Set OFFSET
    pub fn offset(mut self, n: usize) -> Self {
        self.offset = Some(n);
        self
    }
}

impl Default for Query {
    fn default() -> Self {
        Self::new()
    }
}

/// Query execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Variable bindings
    pub bindings: Vec<HashMap<String, Term>>,
    /// Query statistics
    pub stats: QueryStats,
}

/// Query execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStats {
    /// Number of patterns evaluated
    pub patterns_evaluated: usize,
    /// Number of intermediate results
    pub intermediate_results: usize,
    /// Number of final results
    pub final_results: usize,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
}

/// Query executor with optimization
pub struct QueryExecutor {
    /// Knowledge base to query
    kb: KnowledgeBase,
    /// Whether to enable query optimization
    optimize: bool,
}

impl QueryExecutor {
    /// Create a new query executor
    pub fn new(kb: KnowledgeBase) -> Self {
        Self { kb, optimize: true }
    }

    /// Enable or disable query optimization
    pub fn set_optimization(&mut self, enabled: bool) {
        self.optimize = enabled;
    }

    /// Execute a query
    pub fn execute(&self, mut query: Query) -> Result<QueryResult> {
        let start = std::time::Instant::now();

        // Optimize query if enabled
        if self.optimize {
            query = self.optimize_query(query)?;
        }

        // Execute query patterns
        let mut bindings = vec![HashMap::new()];
        let mut patterns_evaluated = 0;
        let mut intermediate_results = 0;

        for pattern in &query.patterns {
            let new_bindings = self.match_pattern(pattern, &bindings)?;
            intermediate_results += new_bindings.len();
            bindings = new_bindings;
            patterns_evaluated += 1;
        }

        // Apply filters
        bindings = self.apply_filters(&query.filters, bindings)?;

        // Apply projection (SELECT clause)
        bindings = self.project_variables(&query.select, bindings);

        // Apply OFFSET and LIMIT
        if let Some(offset) = query.offset {
            bindings = bindings.into_iter().skip(offset).collect();
        }
        if let Some(limit) = query.limit {
            bindings.truncate(limit);
        }

        let execution_time_ms = start.elapsed().as_millis() as u64;
        let final_results = bindings.len();

        Ok(QueryResult {
            bindings,
            stats: QueryStats {
                patterns_evaluated,
                intermediate_results,
                final_results,
                execution_time_ms,
            },
        })
    }

    /// Optimize query (join reordering, filter pushdown)
    fn optimize_query(&self, mut query: Query) -> Result<Query> {
        // Reorder patterns by selectivity (most selective first)
        query.patterns = self.reorder_patterns(query.patterns)?;

        // Push filters down (apply as early as possible)
        // For now, filters are applied after all patterns

        Ok(query)
    }

    /// Reorder patterns by selectivity
    fn reorder_patterns(&self, patterns: Vec<QueryPattern>) -> Result<Vec<QueryPattern>> {
        let mut scored: Vec<(QueryPattern, usize)> = patterns
            .into_iter()
            .map(|p| {
                let selectivity = self.estimate_selectivity(&p);
                (p, selectivity)
            })
            .collect();

        // Sort by selectivity (ascending - most selective first)
        scored.sort_by_key(|(_, s)| *s);

        Ok(scored.into_iter().map(|(p, _)| p).collect())
    }

    /// Estimate selectivity of a pattern (number of matches)
    fn estimate_selectivity(&self, pattern: &QueryPattern) -> usize {
        match pattern {
            QueryPattern::Exact(pred) => {
                // Exact match - check if exists in facts
                if self.kb.facts.contains(pred) {
                    1
                } else {
                    0
                }
            }
            QueryPattern::Pattern { name, args } => {
                // Pattern match - count matching facts
                let mut count = 0;
                for fact in &self.kb.facts {
                    if let Some(n) = name {
                        if &fact.name != n {
                            continue;
                        }
                    }
                    if args.len() != fact.args.len() {
                        continue;
                    }
                    if args
                        .iter()
                        .zip(&fact.args)
                        .all(|(p, t)| self.term_matches(p, t))
                    {
                        count += 1;
                    }
                }
                count
            }
            QueryPattern::Variable(_) => self.kb.facts.len(), // Matches all
        }
    }

    /// Match a pattern against current bindings
    fn match_pattern(
        &self,
        pattern: &QueryPattern,
        current_bindings: &[HashMap<String, Term>],
    ) -> Result<Vec<HashMap<String, Term>>> {
        let mut new_bindings = Vec::new();

        for binding in current_bindings {
            match pattern {
                QueryPattern::Exact(pred) => {
                    // Check if predicate exists in facts
                    if self.kb.facts.contains(pred) {
                        new_bindings.push(binding.clone());
                    }
                }
                QueryPattern::Pattern { name, args } => {
                    // Match against all facts
                    for fact in &self.kb.facts {
                        if let Some(n) = name {
                            if &fact.name != n {
                                continue;
                            }
                        }
                        if args.len() != fact.args.len() {
                            continue;
                        }

                        // Try to match all arguments
                        let mut new_binding = binding.clone();
                        let mut matches = true;

                        for (pattern_arg, fact_arg) in args.iter().zip(&fact.args) {
                            if !self.match_term_pattern(pattern_arg, fact_arg, &mut new_binding) {
                                matches = false;
                                break;
                            }
                        }

                        if matches {
                            new_bindings.push(new_binding);
                        }
                    }
                }
                QueryPattern::Variable(var) => {
                    // Bind variable to all facts
                    for fact in &self.kb.facts {
                        let mut new_binding = binding.clone();
                        // Convert predicate to term representation (simplified)
                        new_binding.insert(var.clone(), Term::Var(fact.name.clone()));
                        new_bindings.push(new_binding);
                    }
                }
            }
        }

        Ok(new_bindings)
    }

    /// Match a term pattern against a term
    fn match_term_pattern(
        &self,
        pattern: &TermPattern,
        term: &Term,
        binding: &mut HashMap<String, Term>,
    ) -> bool {
        match pattern {
            TermPattern::Exact(ref expected) => term == expected,
            TermPattern::Wildcard => true,
            TermPattern::Variable(var) => {
                // Check if variable already bound
                if let Some(bound_term) = binding.get(var) {
                    bound_term == term
                } else {
                    // Bind variable
                    binding.insert(var.clone(), term.clone());
                    true
                }
            }
            TermPattern::TypeConstraint(typ) => self.check_term_type(term, *typ),
        }
    }

    /// Check if term matches type constraint
    fn check_term_type(&self, term: &Term, typ: TermType) -> bool {
        matches!(
            (term, typ),
            (Term::Var(_), TermType::Var)
                | (Term::Const(_), TermType::Const)
                | (Term::Fun(_, _), TermType::Fun)
                | (Term::Ref(_), TermType::Ref)
        )
    }

    /// Check if term matches pattern
    fn term_matches(&self, pattern: &TermPattern, term: &Term) -> bool {
        match pattern {
            TermPattern::Exact(ref expected) => term == expected,
            TermPattern::Wildcard => true,
            TermPattern::Variable(_) => true,
            TermPattern::TypeConstraint(typ) => self.check_term_type(term, *typ),
        }
    }

    /// Apply filter expressions to bindings
    fn apply_filters(
        &self,
        filters: &[FilterExpr],
        bindings: Vec<HashMap<String, Term>>,
    ) -> Result<Vec<HashMap<String, Term>>> {
        let mut result = bindings;

        for filter in filters {
            result.retain(|binding| self.evaluate_filter(filter, binding));
        }

        Ok(result)
    }

    /// Evaluate a filter expression
    fn evaluate_filter(&self, filter: &FilterExpr, binding: &HashMap<String, Term>) -> bool {
        match filter {
            FilterExpr::Equals(var1, var2) => {
                let t1 = binding.get(var1);
                let t2 = binding.get(var2);
                t1.is_some() && t2.is_some() && t1 == t2
            }
            FilterExpr::NotEquals(var1, var2) => {
                let t1 = binding.get(var1);
                let t2 = binding.get(var2);
                t1.is_some() && t2.is_some() && t1 != t2
            }
            FilterExpr::Regex(var, pattern) => {
                if let Some(term) = binding.get(var) {
                    let term_str = format!("{:?}", term);
                    term_str.contains(pattern)
                } else {
                    false
                }
            }
            FilterExpr::IsType(var, typ) => {
                if let Some(term) = binding.get(var) {
                    self.check_term_type(term, *typ)
                } else {
                    false
                }
            }
            FilterExpr::And(exprs) => exprs.iter().all(|e| self.evaluate_filter(e, binding)),
            FilterExpr::Or(exprs) => exprs.iter().any(|e| self.evaluate_filter(e, binding)),
        }
    }

    /// Project variables (SELECT clause)
    fn project_variables(
        &self,
        vars: &[String],
        bindings: Vec<HashMap<String, Term>>,
    ) -> Vec<HashMap<String, Term>> {
        if vars.is_empty() {
            // No projection, return all
            return bindings;
        }

        bindings
            .into_iter()
            .map(|binding| {
                vars.iter()
                    .filter_map(|v| binding.get(v).map(|t| (v.clone(), t.clone())))
                    .collect()
            })
            .collect()
    }

    /// Execute a boolean query
    pub fn execute_boolean(&self, query: &BooleanQuery) -> Result<QueryResult> {
        match query {
            BooleanQuery::And(queries) => {
                // Execute all queries and intersect results
                let mut results: Option<Vec<HashMap<String, Term>>> = None;

                for q in queries {
                    let result = self.execute(q.clone())?;

                    if let Some(existing) = results {
                        // Intersect
                        let new_set: HashSet<_> = result
                            .bindings
                            .into_iter()
                            .map(|b| format!("{:?}", b))
                            .collect();
                        results = Some(
                            existing
                                .into_iter()
                                .filter(|b| new_set.contains(&format!("{:?}", b)))
                                .collect(),
                        );
                    } else {
                        results = Some(result.bindings);
                    }
                }

                let final_results = results.as_ref().map(|r| r.len()).unwrap_or(0);
                Ok(QueryResult {
                    bindings: results.unwrap_or_default(),
                    stats: QueryStats {
                        patterns_evaluated: queries.len(),
                        intermediate_results: 0,
                        final_results,
                        execution_time_ms: 0,
                    },
                })
            }
            BooleanQuery::Or(queries) => {
                // Execute all queries and union results
                let mut all_bindings = Vec::new();
                let mut seen = HashSet::new();

                for q in queries {
                    let result = self.execute(q.clone())?;

                    for binding in result.bindings {
                        let key = format!("{:?}", binding);
                        if seen.insert(key) {
                            all_bindings.push(binding);
                        }
                    }
                }

                Ok(QueryResult {
                    bindings: all_bindings.clone(),
                    stats: QueryStats {
                        patterns_evaluated: queries.len(),
                        intermediate_results: 0,
                        final_results: all_bindings.len(),
                        execution_time_ms: 0,
                    },
                })
            }
            BooleanQuery::Not(query) => {
                // Get all possible bindings, then subtract query results
                let all_result = self.execute(Query::new())?;
                let excluded_result = self.execute(query.as_ref().clone())?;

                let excluded_set: HashSet<_> = excluded_result
                    .bindings
                    .into_iter()
                    .map(|b| format!("{:?}", b))
                    .collect();

                let filtered: Vec<_> = all_result
                    .bindings
                    .into_iter()
                    .filter(|b| !excluded_set.contains(&format!("{:?}", b)))
                    .collect();

                Ok(QueryResult {
                    bindings: filtered.clone(),
                    stats: QueryStats {
                        patterns_evaluated: 1,
                        intermediate_results: 0,
                        final_results: filtered.len(),
                        execution_time_ms: 0,
                    },
                })
            }
            BooleanQuery::Atom(query) => self.execute(query.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ipfrs_tensorlogic::Constant;

    #[test]
    fn test_query_builder() {
        let query = Query::new()
            .select("X")
            .select("Y")
            .where_pattern(QueryPattern::Pattern {
                name: Some("parent".to_string()),
                args: vec![
                    TermPattern::Variable("X".to_string()),
                    TermPattern::Variable("Y".to_string()),
                ],
            })
            .limit(10);

        assert_eq!(query.select.len(), 2);
        assert_eq!(query.patterns.len(), 1);
        assert_eq!(query.limit, Some(10));
    }

    #[test]
    fn test_query_executor() {
        let mut kb = KnowledgeBase::new();

        // Add some facts
        let alice = Term::Const(Constant::String("Alice".to_string()));
        let bob = Term::Const(Constant::String("Bob".to_string()));
        kb.add_fact(Predicate::new(
            "parent".to_string(),
            vec![alice.clone(), bob.clone()],
        ));

        let executor = QueryExecutor::new(kb);

        // Query for all parent relationships
        let query = Query::new().where_pattern(QueryPattern::Pattern {
            name: Some("parent".to_string()),
            args: vec![TermPattern::Wildcard, TermPattern::Wildcard],
        });

        let result = executor
            .execute(query)
            .expect("test: executor execute query executor failed");
        assert!(!result.bindings.is_empty());
    }

    #[test]
    fn test_pattern_matching() {
        let mut kb = KnowledgeBase::new();

        let alice = Term::Const(Constant::String("Alice".to_string()));
        let bob = Term::Const(Constant::String("Bob".to_string()));
        kb.add_fact(Predicate::new("parent".to_string(), vec![alice, bob]));

        let executor = QueryExecutor::new(kb);

        // Query with variable binding
        let query = Query::new()
            .select("X")
            .select("Y")
            .where_pattern(QueryPattern::Pattern {
                name: Some("parent".to_string()),
                args: vec![
                    TermPattern::Variable("X".to_string()),
                    TermPattern::Variable("Y".to_string()),
                ],
            });

        let result = executor
            .execute(query)
            .expect("test: executor execute pattern matching failed");
        assert_eq!(result.bindings.len(), 1);
        assert!(result.bindings[0].contains_key("X"));
        assert!(result.bindings[0].contains_key("Y"));
    }

    #[test]
    fn test_filter_expr() {
        let mut kb = KnowledgeBase::new();

        let alice = Term::Const(Constant::String("Alice".to_string()));
        let bob = Term::Const(Constant::String("Bob".to_string()));
        kb.add_fact(Predicate::new("person".to_string(), vec![alice]));
        kb.add_fact(Predicate::new("person".to_string(), vec![bob]));

        let executor = QueryExecutor::new(kb);

        // Query with type filter
        let query = Query::new()
            .select("X")
            .where_pattern(QueryPattern::Pattern {
                name: Some("person".to_string()),
                args: vec![TermPattern::Variable("X".to_string())],
            })
            .filter(FilterExpr::IsType("X".to_string(), TermType::Const));

        let result = executor
            .execute(query)
            .expect("test: executor execute filter expr failed");
        assert_eq!(result.bindings.len(), 2);
    }
}
