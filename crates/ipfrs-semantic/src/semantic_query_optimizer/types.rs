//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::functions::levenshtein;
use super::types_4::{OptimizerConfig, QueryNode, QueryPlan};

/// Errors returned by the optimizer.
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizerError {
    ParseError(String),
    InvalidQuery(String),
    OptimizationFailed(String),
    ConfigurationError(String),
}
/// A single step in the execution plan.
#[derive(Debug, Clone)]
pub struct ExecutionStep {
    pub step_type: StepType,
    pub description: String,
    pub estimated_docs: usize,
    pub estimated_time_us: u64,
}
/// High-level category of an execution step.
#[derive(Debug, Clone, PartialEq)]
pub enum StepType {
    TermLookup,
    VectorScan,
    Filter,
    Join(JoinType),
    Sort,
    Limit,
    Rewrite,
}
/// An optimization rule applied by [`SemanticQueryOptimizer`].
#[derive(Debug, Clone)]
pub enum OptimizationRule {
    ConstantFolding,
    DeduplicateTerms,
    PushDownFilters,
    FlattenNested,
    ReorderBySelectivity,
    ExpandSynonyms(HashMap<String, Vec<String>>),
    EmbeddingCaching,
}
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Token {
    And,
    Or,
    Not,
    Word(String),
    Phrase(Vec<String>),
    LParen,
    RParen,
}
/// Comparison operator for [`QueryNode::Filter`].
#[derive(Debug, Clone, PartialEq)]
pub enum FilterOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    In(Vec<String>),
    Contains,
}
/// Cumulative statistics collected by the optimizer.
#[derive(Debug, Clone, Default)]
pub struct OptimizerStats {
    pub queries_optimized: u64,
    pub rewrites_applied: u64,
    pub avg_cost_reduction: f64,
    pub cache_hits: u64,
}
/// How two document sets are combined in a [`StepType::Join`].
#[derive(Debug, Clone, PartialEq)]
pub enum JoinType {
    Intersect,
    Union,
    Difference,
}
/// Full-featured semantic query optimizer.
pub struct SemanticQueryOptimizer {
    pub(super) config: OptimizerConfig,
    pub(super) embedding_cache: Arc<Mutex<HashMap<String, Vec<f64>>>>,
    pub(super) stats: Arc<Mutex<OptimizerStats>>,
}
impl SemanticQueryOptimizer {
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            embedding_cache: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(OptimizerStats::default())),
        }
    }
    pub fn with_defaults() -> Self {
        Self::new(OptimizerConfig::default())
    }
    /// Parse a query string into a [`QueryNode`] AST.
    pub fn parse(&self, query_str: &str) -> Result<QueryNode, OptimizerError> {
        let tokens = Self::tokenize(query_str)?;
        if tokens.is_empty() {
            return Err(OptimizerError::InvalidQuery(
                "empty query string".to_string(),
            ));
        }
        let mut pos = 0usize;
        let node = Self::parse_expr(&tokens, &mut pos)?;
        if pos < tokens.len() {
            return Err(OptimizerError::ParseError(format!(
                "unexpected token at position {pos}: {:?}",
                tokens[pos]
            )));
        }
        Ok(node)
    }
    pub(super) fn tokenize(input: &str) -> Result<Vec<Token>, OptimizerError> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = input.chars().collect();
        let mut i = 0usize;
        while i < chars.len() {
            if chars[i].is_ascii_whitespace() {
                i += 1;
                continue;
            }
            match chars[i] {
                '(' => {
                    tokens.push(Token::LParen);
                    i += 1;
                }
                ')' => {
                    tokens.push(Token::RParen);
                    i += 1;
                }
                '"' => {
                    i += 1;
                    let mut words = Vec::new();
                    let mut buf = String::new();
                    loop {
                        if i >= chars.len() {
                            return Err(OptimizerError::ParseError(
                                "unterminated quoted phrase".to_string(),
                            ));
                        }
                        if chars[i] == '"' {
                            if !buf.is_empty() {
                                words.push(buf.clone());
                                buf.clear();
                            }
                            i += 1;
                            break;
                        }
                        if chars[i].is_ascii_whitespace() {
                            if !buf.is_empty() {
                                words.push(buf.clone());
                                buf.clear();
                            }
                        } else {
                            buf.push(chars[i]);
                        }
                        i += 1;
                    }
                    if words.is_empty() {
                        return Err(OptimizerError::ParseError(
                            "empty quoted phrase".to_string(),
                        ));
                    }
                    tokens.push(Token::Phrase(words));
                }
                _ => {
                    let mut word = String::new();
                    while i < chars.len()
                        && !chars[i].is_ascii_whitespace()
                        && chars[i] != '('
                        && chars[i] != ')'
                    {
                        word.push(chars[i]);
                        i += 1;
                    }
                    match word.as_str() {
                        "AND" => tokens.push(Token::And),
                        "OR" => tokens.push(Token::Or),
                        "NOT" => tokens.push(Token::Not),
                        _ => tokens.push(Token::Word(word)),
                    }
                }
            }
        }
        Ok(tokens)
    }
    pub(super) fn parse_expr(
        tokens: &[Token],
        pos: &mut usize,
    ) -> Result<QueryNode, OptimizerError> {
        let mut left = Self::parse_not(tokens, pos)?;
        while *pos < tokens.len() {
            match &tokens[*pos] {
                Token::And => {
                    *pos += 1;
                    let right = Self::parse_not(tokens, pos)?;
                    left = match left {
                        QueryNode::And(mut ch) => {
                            ch.push(right);
                            QueryNode::And(ch)
                        }
                        _ => QueryNode::And(vec![left, right]),
                    };
                }
                Token::Or => {
                    *pos += 1;
                    let right = Self::parse_not(tokens, pos)?;
                    left = match left {
                        QueryNode::Or(mut ch) => {
                            ch.push(right);
                            QueryNode::Or(ch)
                        }
                        _ => QueryNode::Or(vec![left, right]),
                    };
                }
                _ => break,
            }
        }
        Ok(left)
    }
    pub(super) fn parse_not(
        tokens: &[Token],
        pos: &mut usize,
    ) -> Result<QueryNode, OptimizerError> {
        if *pos < tokens.len() && tokens[*pos] == Token::Not {
            *pos += 1;
            let child = Self::parse_not(tokens, pos)?;
            return Ok(QueryNode::Not(Box::new(child)));
        }
        Self::parse_atom(tokens, pos)
    }
    pub(super) fn parse_atom(
        tokens: &[Token],
        pos: &mut usize,
    ) -> Result<QueryNode, OptimizerError> {
        if *pos >= tokens.len() {
            return Err(OptimizerError::ParseError(
                "unexpected end of query".to_string(),
            ));
        }
        match &tokens[*pos].clone() {
            Token::LParen => {
                *pos += 1;
                let node = Self::parse_expr(tokens, pos)?;
                if *pos >= tokens.len() || tokens[*pos] != Token::RParen {
                    return Err(OptimizerError::ParseError(
                        "missing closing parenthesis".to_string(),
                    ));
                }
                *pos += 1;
                Ok(node)
            }
            Token::Phrase(words) => {
                *pos += 1;
                Ok(QueryNode::Phrase(words.clone()))
            }
            Token::Word(raw) => {
                let raw = raw.clone();
                *pos += 1;
                Ok(Self::parse_word_atom(&raw))
            }
            other => Err(OptimizerError::ParseError(format!(
                "unexpected token: {other:?}"
            ))),
        }
    }
    pub(super) fn parse_word_atom(raw: &str) -> QueryNode {
        if let Some(cp) = raw.find(':') {
            let field = raw[..cp].to_string();
            let value = raw[cp + 1..].to_string();
            if !field.is_empty() && !value.is_empty() {
                return QueryNode::Filter {
                    field,
                    op: FilterOp::Eq,
                    value,
                };
            }
        }
        if let Some(tp) = raw.rfind('~') {
            let term_part = &raw[..tp];
            if !term_part.is_empty() {
                if let Ok(max_edits) = raw[tp + 1..].parse::<u8>() {
                    return QueryNode::Fuzzy {
                        term: term_part.to_string(),
                        max_edits,
                    };
                }
            }
        }
        if let Some(cp) = raw.rfind('^') {
            let term_part = &raw[..cp];
            if !term_part.is_empty() {
                if let Ok(factor) = raw[cp + 1..].parse::<f64>() {
                    return QueryNode::Boost {
                        node: Box::new(QueryNode::Term(term_part.to_string())),
                        factor,
                    };
                }
            }
        }
        QueryNode::Term(raw.to_string())
    }
    /// Apply all configured rules to `node` and return a [`QueryPlan`].
    pub fn optimize(
        &self,
        node: QueryNode,
        original_query: &str,
    ) -> Result<QueryPlan, OptimizerError> {
        let original_cost = self.estimate_cost(&node);
        let mut current = node;
        let mut rewrites_applied = 0u64;
        for rule in &self.config.apply_rules {
            let (rw, n) = self.apply_rule(current, rule)?;
            current = rw;
            rewrites_applied += n;
        }
        self.validate(&current)?;
        let optimized_cost = self.estimate_cost(&current);
        let estimated_results = self.estimate_results(&current);
        let execution_steps = self.plan_execution(&current);
        if let Ok(mut s) = self.stats.lock() {
            s.queries_optimized += 1;
            s.rewrites_applied += rewrites_applied;
            let reduction = if original_cost > 0.0 {
                1.0 - optimized_cost / original_cost
            } else {
                0.0
            };
            let n = s.queries_optimized as f64;
            s.avg_cost_reduction = s.avg_cost_reduction * (n - 1.0) / n + reduction / n;
        }
        Ok(QueryPlan {
            original_query: original_query.to_string(),
            optimized_nodes: vec![current],
            estimated_cost: optimized_cost,
            estimated_results,
            execution_steps,
        })
    }
    pub(super) fn apply_rule(
        &self,
        node: QueryNode,
        rule: &OptimizationRule,
    ) -> Result<(QueryNode, u64), OptimizerError> {
        Ok(match rule {
            OptimizationRule::ConstantFolding => self.fold_constants(node),
            OptimizationRule::DeduplicateTerms => self.deduplicate_terms(node),
            OptimizationRule::FlattenNested => self.flatten_nested(node),
            OptimizationRule::PushDownFilters => self.push_down_filters(node),
            OptimizationRule::ReorderBySelectivity => self.reorder_by_selectivity(node),
            OptimizationRule::ExpandSynonyms(map) => self.expand_synonyms(node, map),
            OptimizationRule::EmbeddingCaching => self.apply_embedding_caching(node),
        })
    }
    pub(super) fn fold_constants(&self, node: QueryNode) -> (QueryNode, u64) {
        match node {
            QueryNode::And(children) => {
                let mut rewrites = 0u64;
                let mut folded: Vec<QueryNode> = children
                    .into_iter()
                    .map(|c| {
                        let (fc, r) = self.fold_constants(c);
                        rewrites += r;
                        fc
                    })
                    .collect();
                let mut to_remove = Vec::new();
                for i in 0..folded.len() {
                    for j in 0..folded.len() {
                        if i != j {
                            if let QueryNode::Not(inner) = &folded[j] {
                                if &folded[i] == inner.as_ref() {
                                    to_remove.push(i);
                                    to_remove.push(j);
                                }
                            }
                        }
                    }
                }
                to_remove.sort_unstable();
                to_remove.dedup();
                if !to_remove.is_empty() {
                    let mut idx = 0usize;
                    folded.retain(|_| {
                        let keep = !to_remove.contains(&idx);
                        idx += 1;
                        keep
                    });
                    rewrites += 1;
                    return (QueryNode::And(vec![]), rewrites);
                }
                (QueryNode::And(folded), rewrites)
            }
            QueryNode::Or(children) => {
                let mut rewrites = 0u64;
                let folded: Vec<QueryNode> = children
                    .into_iter()
                    .map(|c| {
                        let (fc, r) = self.fold_constants(c);
                        rewrites += r;
                        fc
                    })
                    .collect();
                (QueryNode::Or(folded), rewrites)
            }
            QueryNode::Not(inner) => {
                if let QueryNode::Not(inner2) = *inner {
                    let (n, _) = self.fold_constants(*inner2);
                    return (n, 1);
                }
                let (fc, r) = self.fold_constants(*inner);
                (QueryNode::Not(Box::new(fc)), r)
            }
            QueryNode::Boost {
                node: inner,
                factor,
            } => {
                let (fc, r) = self.fold_constants(*inner);
                (
                    QueryNode::Boost {
                        node: Box::new(fc),
                        factor,
                    },
                    r,
                )
            }
            other => (other, 0),
        }
    }
    pub(super) fn deduplicate_terms(&self, node: QueryNode) -> (QueryNode, u64) {
        match node {
            QueryNode::And(children) => {
                let mut rewrites = 0u64;
                let mut deduped: Vec<QueryNode> = Vec::new();
                for child in children {
                    let (dc, r) = self.deduplicate_terms(child);
                    rewrites += r;
                    if !deduped.contains(&dc) {
                        deduped.push(dc);
                    } else {
                        rewrites += 1;
                    }
                }
                (QueryNode::And(deduped), rewrites)
            }
            QueryNode::Or(children) => {
                let mut rewrites = 0u64;
                let mut deduped: Vec<QueryNode> = Vec::new();
                for child in children {
                    let (dc, r) = self.deduplicate_terms(child);
                    rewrites += r;
                    if !deduped.contains(&dc) {
                        deduped.push(dc);
                    } else {
                        rewrites += 1;
                    }
                }
                (QueryNode::Or(deduped), rewrites)
            }
            QueryNode::Not(inner) => {
                let (dc, r) = self.deduplicate_terms(*inner);
                (QueryNode::Not(Box::new(dc)), r)
            }
            QueryNode::Boost {
                node: inner,
                factor,
            } => {
                let (dc, r) = self.deduplicate_terms(*inner);
                (
                    QueryNode::Boost {
                        node: Box::new(dc),
                        factor,
                    },
                    r,
                )
            }
            other => (other, 0),
        }
    }
    pub(super) fn flatten_nested(&self, node: QueryNode) -> (QueryNode, u64) {
        match node {
            QueryNode::And(children) => {
                let mut rewrites = 0u64;
                let mut flat: Vec<QueryNode> = Vec::new();
                for child in children {
                    let (fc, r) = self.flatten_nested(child);
                    rewrites += r;
                    if let QueryNode::And(ic) = fc {
                        rewrites += 1;
                        flat.extend(ic);
                    } else {
                        flat.push(fc);
                    }
                }
                (QueryNode::And(flat), rewrites)
            }
            QueryNode::Or(children) => {
                let mut rewrites = 0u64;
                let mut flat: Vec<QueryNode> = Vec::new();
                for child in children {
                    let (fc, r) = self.flatten_nested(child);
                    rewrites += r;
                    if let QueryNode::Or(ic) = fc {
                        rewrites += 1;
                        flat.extend(ic);
                    } else {
                        flat.push(fc);
                    }
                }
                (QueryNode::Or(flat), rewrites)
            }
            QueryNode::Not(inner) => {
                let (fc, r) = self.flatten_nested(*inner);
                (QueryNode::Not(Box::new(fc)), r)
            }
            QueryNode::Boost {
                node: inner,
                factor,
            } => {
                let (fc, r) = self.flatten_nested(*inner);
                (
                    QueryNode::Boost {
                        node: Box::new(fc),
                        factor,
                    },
                    r,
                )
            }
            other => (other, 0),
        }
    }
    pub(super) fn push_down_filters(&self, node: QueryNode) -> (QueryNode, u64) {
        match node {
            QueryNode::And(mut children) => {
                let mut rewrites = 0u64;
                let mut processed: Vec<QueryNode> = children
                    .drain(..)
                    .map(|c| {
                        let (pc, r) = self.push_down_filters(c);
                        rewrites += r;
                        pc
                    })
                    .collect();
                let mut filters: Vec<QueryNode> = Vec::new();
                let mut rest: Vec<QueryNode> = Vec::new();
                for child in processed.drain(..) {
                    if matches!(child, QueryNode::Filter { .. }) {
                        filters.push(child);
                    } else {
                        rest.push(child);
                    }
                }
                let had_filters = !filters.is_empty();
                filters.extend(rest);
                if had_filters {
                    rewrites += 1;
                }
                (QueryNode::And(filters), rewrites)
            }
            QueryNode::Or(children) => {
                let mut rewrites = 0u64;
                let processed: Vec<QueryNode> = children
                    .into_iter()
                    .map(|c| {
                        let (pc, r) = self.push_down_filters(c);
                        rewrites += r;
                        pc
                    })
                    .collect();
                (QueryNode::Or(processed), rewrites)
            }
            QueryNode::Not(inner) => {
                let (pc, r) = self.push_down_filters(*inner);
                (QueryNode::Not(Box::new(pc)), r)
            }
            QueryNode::Boost {
                node: inner,
                factor,
            } => {
                let (pc, r) = self.push_down_filters(*inner);
                (
                    QueryNode::Boost {
                        node: Box::new(pc),
                        factor,
                    },
                    r,
                )
            }
            other => (other, 0),
        }
    }
    pub(super) fn reorder_by_selectivity(&self, node: QueryNode) -> (QueryNode, u64) {
        match node {
            QueryNode::And(mut children) => {
                let mut rewrites = 0u64;
                let processed: Vec<QueryNode> = children
                    .drain(..)
                    .map(|c| {
                        let (pc, r) = self.reorder_by_selectivity(c);
                        rewrites += r;
                        pc
                    })
                    .collect();
                let total = self.config.index_stats.total_docs.max(1) as f64;
                let mut indexed: Vec<(f64, QueryNode)> = processed
                    .into_iter()
                    .map(|n| {
                        let s = self.selectivity(&n, total);
                        (s, n)
                    })
                    .collect();
                indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let reordered: Vec<QueryNode> = indexed.into_iter().map(|(_, n)| n).collect();
                if reordered
                    .windows(2)
                    .any(|w| self.selectivity(&w[0], total) > self.selectivity(&w[1], total))
                {
                    rewrites += 1;
                }
                (QueryNode::And(reordered), rewrites)
            }
            QueryNode::Or(children) => {
                let mut rewrites = 0u64;
                let processed: Vec<QueryNode> = children
                    .into_iter()
                    .map(|c| {
                        let (pc, r) = self.reorder_by_selectivity(c);
                        rewrites += r;
                        pc
                    })
                    .collect();
                (QueryNode::Or(processed), rewrites)
            }
            QueryNode::Not(inner) => {
                let (pc, r) = self.reorder_by_selectivity(*inner);
                (QueryNode::Not(Box::new(pc)), r)
            }
            QueryNode::Boost {
                node: inner,
                factor,
            } => {
                let (pc, r) = self.reorder_by_selectivity(*inner);
                (
                    QueryNode::Boost {
                        node: Box::new(pc),
                        factor,
                    },
                    r,
                )
            }
            other => (other, 0),
        }
    }
    pub(super) fn selectivity(&self, node: &QueryNode, total_docs: f64) -> f64 {
        match node {
            QueryNode::Term(t) => {
                let freq = self
                    .config
                    .index_stats
                    .term_frequencies
                    .get(t)
                    .copied()
                    .unwrap_or(1) as f64;
                (freq / total_docs).clamp(0.0, 1.0)
            }
            QueryNode::Filter { .. } => 0.1,
            QueryNode::Phrase(_) => 0.05,
            QueryNode::Embedding(_) => 0.2,
            QueryNode::And(ch) => ch
                .iter()
                .map(|c| self.selectivity(c, total_docs))
                .product::<f64>(),
            QueryNode::Or(ch) => ch
                .iter()
                .map(|c| self.selectivity(c, total_docs))
                .sum::<f64>()
                .min(1.0),
            QueryNode::Not(_) => 0.9,
            QueryNode::Boost { node, .. } => self.selectivity(node, total_docs),
            QueryNode::Fuzzy { term, max_edits } => {
                let base = self
                    .config
                    .index_stats
                    .term_frequencies
                    .get(term)
                    .copied()
                    .unwrap_or(1) as f64;
                (base * (1.0 + *max_edits as f64 * 2.0) / total_docs).clamp(0.0, 1.0)
            }
        }
    }
    pub(super) fn expand_synonyms(
        &self,
        node: QueryNode,
        map: &HashMap<String, Vec<String>>,
    ) -> (QueryNode, u64) {
        match node {
            QueryNode::Term(ref t) => {
                if let Some(syns) = map.get(t) {
                    if !syns.is_empty() {
                        let mut alts = vec![node.clone()];
                        alts.extend(syns.iter().map(|s| QueryNode::Term(s.clone())));
                        return (QueryNode::Or(alts), 1);
                    }
                }
                (node, 0)
            }
            QueryNode::And(children) => {
                let mut rewrites = 0u64;
                let expanded: Vec<QueryNode> = children
                    .into_iter()
                    .map(|c| {
                        let (ec, r) = self.expand_synonyms(c, map);
                        rewrites += r;
                        ec
                    })
                    .collect();
                (QueryNode::And(expanded), rewrites)
            }
            QueryNode::Or(children) => {
                let mut rewrites = 0u64;
                let expanded: Vec<QueryNode> = children
                    .into_iter()
                    .map(|c| {
                        let (ec, r) = self.expand_synonyms(c, map);
                        rewrites += r;
                        ec
                    })
                    .collect();
                (QueryNode::Or(expanded), rewrites)
            }
            QueryNode::Not(inner) => {
                let (ec, r) = self.expand_synonyms(*inner, map);
                (QueryNode::Not(Box::new(ec)), r)
            }
            QueryNode::Boost {
                node: inner,
                factor,
            } => {
                let (ec, r) = self.expand_synonyms(*inner, map);
                (
                    QueryNode::Boost {
                        node: Box::new(ec),
                        factor,
                    },
                    r,
                )
            }
            other => (other, 0),
        }
    }
    pub(super) fn apply_embedding_caching(&self, node: QueryNode) -> (QueryNode, u64) {
        match node {
            QueryNode::Fuzzy { ref term, .. } => {
                if let Ok(cache) = self.embedding_cache.lock() {
                    if cache.contains_key(term) {
                        if let Ok(mut s) = self.stats.lock() {
                            s.cache_hits += 1;
                        }
                    }
                }
                (node, 0)
            }
            QueryNode::And(children) => {
                let mut rewrites = 0u64;
                let processed: Vec<QueryNode> = children
                    .into_iter()
                    .map(|c| {
                        let (pc, r) = self.apply_embedding_caching(c);
                        rewrites += r;
                        pc
                    })
                    .collect();
                (QueryNode::And(processed), rewrites)
            }
            QueryNode::Or(children) => {
                let mut rewrites = 0u64;
                let processed: Vec<QueryNode> = children
                    .into_iter()
                    .map(|c| {
                        let (pc, r) = self.apply_embedding_caching(c);
                        rewrites += r;
                        pc
                    })
                    .collect();
                (QueryNode::Or(processed), rewrites)
            }
            QueryNode::Not(inner) => {
                let (pc, r) = self.apply_embedding_caching(*inner);
                (QueryNode::Not(Box::new(pc)), r)
            }
            QueryNode::Boost {
                node: inner,
                factor,
            } => {
                let (pc, r) = self.apply_embedding_caching(*inner);
                (
                    QueryNode::Boost {
                        node: Box::new(pc),
                        factor,
                    },
                    r,
                )
            }
            other => (other, 0),
        }
    }
    /// Estimate cost: Term=1.0, Phrase=2.0, Embedding=10.0, And=product/total_docs,
    /// Or=sum, Not=1.5×child, Filter=0.5, Boost=child+0.1, Fuzzy=3.0×(max_edits+1).
    pub fn estimate_cost(&self, node: &QueryNode) -> f64 {
        let total = self.config.index_stats.total_docs.max(1) as f64;
        match node {
            QueryNode::Term(_) => 1.0,
            QueryNode::Phrase(_) => 2.0,
            QueryNode::Embedding(_) => 10.0,
            QueryNode::And(ch) => ch.iter().map(|c| self.estimate_cost(c)).product::<f64>() / total,
            QueryNode::Or(ch) => ch.iter().map(|c| self.estimate_cost(c)).sum(),
            QueryNode::Not(inner) => 1.5 * self.estimate_cost(inner),
            QueryNode::Filter { .. } => 0.5,
            QueryNode::Boost { node: inner, .. } => self.estimate_cost(inner) + 0.1,
            QueryNode::Fuzzy { max_edits, .. } => 3.0 * (*max_edits as f64 + 1.0),
        }
    }
    pub(super) fn estimate_results(&self, node: &QueryNode) -> usize {
        let total = self.config.index_stats.total_docs.max(1) as f64;
        (self.selectivity(node, total) * total) as usize
    }
    /// Generate an ordered list of [`ExecutionStep`]s for the given node.
    pub fn plan_execution(&self, node: &QueryNode) -> Vec<ExecutionStep> {
        let mut steps = Vec::new();
        self.plan_node(node, &mut steps);
        steps
    }
    pub(super) fn plan_node(&self, node: &QueryNode, steps: &mut Vec<ExecutionStep>) {
        match node {
            QueryNode::Term(t) => {
                let freq = self
                    .config
                    .index_stats
                    .term_frequencies
                    .get(t)
                    .copied()
                    .unwrap_or(100) as usize;
                steps.push(ExecutionStep {
                    step_type: StepType::TermLookup,
                    description: format!("term lookup: '{t}'"),
                    estimated_docs: freq,
                    estimated_time_us: 10 + freq as u64 / 100,
                });
            }
            QueryNode::Phrase(words) => {
                let docs = (self.config.index_stats.total_docs / 100).max(1);
                steps.push(ExecutionStep {
                    step_type: StepType::TermLookup,
                    description: format!("phrase lookup: \"{}\"", words.join(" ")),
                    estimated_docs: docs,
                    estimated_time_us: 50 + docs as u64 / 100,
                });
            }
            QueryNode::Embedding(v) => {
                let docs = (self.config.index_stats.total_docs as f64 * 0.05).max(1.0) as usize;
                steps.push(ExecutionStep {
                    step_type: StepType::VectorScan,
                    description: format!("vector scan: dim={}", v.len()),
                    estimated_docs: docs,
                    estimated_time_us: 500 + v.len() as u64 * 2,
                });
            }
            QueryNode::And(children) => {
                for c in children {
                    self.plan_node(c, steps);
                }
                let docs = (self.config.index_stats.total_docs / 10).max(1);
                steps.push(ExecutionStep {
                    step_type: StepType::Join(JoinType::Intersect),
                    description: format!("intersect {} sets", children.len()),
                    estimated_docs: docs,
                    estimated_time_us: 20 * children.len() as u64,
                });
            }
            QueryNode::Or(children) => {
                for c in children {
                    self.plan_node(c, steps);
                }
                let docs = self.config.index_stats.total_docs / 3;
                steps.push(ExecutionStep {
                    step_type: StepType::Join(JoinType::Union),
                    description: format!("union {} sets", children.len()),
                    estimated_docs: docs,
                    estimated_time_us: 20 * children.len() as u64,
                });
            }
            QueryNode::Not(inner) => {
                self.plan_node(inner, steps);
                let docs = (self.config.index_stats.total_docs as f64 * 0.9) as usize;
                steps.push(ExecutionStep {
                    step_type: StepType::Join(JoinType::Difference),
                    description: "difference (NOT)".to_string(),
                    estimated_docs: docs,
                    estimated_time_us: 30,
                });
            }
            QueryNode::Filter { field, op, value } => {
                let docs = (self.config.index_stats.total_docs / 20).max(1);
                steps.push(ExecutionStep {
                    step_type: StepType::Filter,
                    description: format!("filter: {field} {op:?} {value}"),
                    estimated_docs: docs,
                    estimated_time_us: 5,
                });
            }
            QueryNode::Boost {
                node: inner,
                factor,
            } => {
                self.plan_node(inner, steps);
                steps.push(ExecutionStep {
                    step_type: StepType::Rewrite,
                    description: format!("apply boost factor {factor:.3}"),
                    estimated_docs: 0,
                    estimated_time_us: 2,
                });
            }
            QueryNode::Fuzzy { term, max_edits } => {
                let docs = (self.config.index_stats.total_docs / 50).max(1);
                steps.push(ExecutionStep {
                    step_type: StepType::TermLookup,
                    description: format!("fuzzy lookup: '{term}' (max_edits={max_edits})"),
                    estimated_docs: docs,
                    estimated_time_us: 100 * (*max_edits as u64 + 1),
                });
            }
        }
    }
    pub(super) fn validate(&self, node: &QueryNode) -> Result<(), OptimizerError> {
        self.validate_node(node, 0)
    }
    pub(super) fn validate_node(
        &self,
        node: &QueryNode,
        depth: usize,
    ) -> Result<(), OptimizerError> {
        if depth > 64 {
            return Err(OptimizerError::InvalidQuery(
                "query nesting exceeds maximum depth of 64".to_string(),
            ));
        }
        match node {
            QueryNode::Embedding(v) if v.len() > self.config.max_embedding_dim => {
                return Err(OptimizerError::InvalidQuery(format!(
                    "embedding dimension {} exceeds maximum {}",
                    v.len(),
                    self.config.max_embedding_dim
                )));
            }
            QueryNode::Embedding(_) => {}
            QueryNode::And(ch) | QueryNode::Or(ch) => {
                for c in ch {
                    self.validate_node(c, depth + 1)?;
                }
            }
            QueryNode::Not(inner) => self.validate_node(inner, depth + 1)?,
            QueryNode::Boost {
                node: inner,
                factor,
            } => {
                if *factor <= 0.0 {
                    return Err(OptimizerError::InvalidQuery(
                        "boost factor must be positive".to_string(),
                    ));
                }
                self.validate_node(inner, depth + 1)?;
            }
            _ => {}
        }
        Ok(())
    }
    /// Store an embedding vector for a term so that `EmbeddingCaching` can reuse it.
    pub fn cache_embedding(&self, term: &str, embedding: Vec<f64>) {
        if let Ok(mut cache) = self.embedding_cache.lock() {
            cache.insert(term.to_string(), embedding);
        }
    }
    /// Retrieve a cached embedding for a term.
    pub fn get_cached_embedding(&self, term: &str) -> Option<Vec<f64>> {
        self.embedding_cache
            .lock()
            .ok()
            .and_then(|c| c.get(term).cloned())
    }
    /// Return a snapshot of optimizer statistics.
    pub fn stats(&self) -> OptimizerStats {
        self.stats.lock().map(|s| s.clone()).unwrap_or_default()
    }
    /// Compute the Levenshtein edit distance between two strings (public, for callers).
    pub fn edit_distance(a: &str, b: &str) -> u8 {
        levenshtein(a, b)
    }
}
