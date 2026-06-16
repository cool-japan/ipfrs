//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::{ExecutionStep, FilterOp, OptimizationRule};

/// Approximate index statistics used for selectivity estimation.
#[derive(Debug, Clone, Default)]
pub struct IndexHints {
    pub total_docs: usize,
    pub term_frequencies: HashMap<String, u64>,
    pub avg_doc_length: f64,
}
/// Configuration for `SemanticQueryOptimizer`.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pub max_terms: usize,
    pub max_embedding_dim: usize,
    pub apply_rules: Vec<OptimizationRule>,
    pub index_stats: IndexHints,
}
/// Cost-annotated execution plan for a semantic query.
#[derive(Debug, Clone)]
pub struct QueryPlan {
    pub original_query: String,
    pub optimized_nodes: Vec<QueryNode>,
    pub estimated_cost: f64,
    pub estimated_results: usize,
    pub execution_steps: Vec<ExecutionStep>,
}
/// A node in the semantic query AST.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryNode {
    Term(String),
    Phrase(Vec<String>),
    Embedding(Vec<f64>),
    And(Vec<QueryNode>),
    Or(Vec<QueryNode>),
    Not(Box<QueryNode>),
    Filter {
        field: String,
        op: FilterOp,
        value: String,
    },
    Boost {
        node: Box<QueryNode>,
        factor: f64,
    },
    Fuzzy {
        term: String,
        max_edits: u8,
    },
}
