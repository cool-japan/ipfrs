//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::functions::{
    cosine_similarity, dpp_marginal, euclidean_distance, kernel_val, max_similarity_to_selected,
    mmr_score, normalize_in_place, weighted_sum, xorshift_f64,
};

/// A document in the search index.
#[derive(Debug, Clone)]
pub struct SearchDoc {
    /// Unique document identifier.
    pub id: String,
    /// Dense embedding vector.
    pub embedding: Vec<f64>,
    /// Arbitrary key-value metadata.
    pub metadata: Vec<(String, String)>,
}
/// The expanded query produced by context-aware query expansion.
///
/// **Alias**: exported as `CesExpandedQuery` in `lib.rs` to avoid colliding with
/// the already-public `ExpandedQuery` from `query_expander`.
#[derive(Debug, Clone)]
pub struct CesExpandedQuery {
    /// The raw query embedding before expansion.
    pub original: Vec<f64>,
    /// Weighted combination of original + context.
    pub expanded: Vec<f64>,
    /// α — weight given to the context component (0 = no expansion).
    pub expansion_weight: f64,
    /// Weight given to recent history embeddings.
    pub history_weight: f64,
}
/// A single result returned by [`ContextualEmbeddingSearch::search`].
#[derive(Debug, Clone)]
pub struct ContextualResult {
    /// Document identifier.
    pub doc_id: String,
    /// Raw cosine similarity to the expanded query.
    pub relevance_score: f64,
    /// Diversity contribution (higher = more diverse relative to already-selected results).
    pub diversity_score: f64,
    /// Final score = (relevance + diversity) / 2.
    pub final_score: f64,
    /// 1-based rank among returned results.
    pub rank: usize,
    /// Explanation: list of (feature name, contribution value).
    pub explanation: Vec<(String, f64)>,
}
/// Configuration for a single search call.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Maximum number of results to return after diversity re-ranking.
    pub top_k: usize,
    /// Diversity strategy applied during re-ranking.
    pub diversity_strategy: DiversityStrategy,
    /// α ∈ \[0,1\] — how much weight is given to context expansion.  0 = no expansion.
    pub expansion_alpha: f64,
    /// Whether to suppress directions of negative examples.
    pub use_negative_examples: bool,
    /// Retrieve this many candidates before applying diversity re-ranking.
    pub rerank_top_n: usize,
    /// Minimum relevance score; candidates below this threshold are dropped.
    pub min_relevance: f64,
}
/// Errors that can be returned by [`ContextualEmbeddingSearch`].
#[derive(Debug, Clone, PartialEq)]
pub enum SearchError {
    /// The index contains no documents.
    IndexEmpty,
    /// Query or document embedding has wrong dimensionality.
    DimensionMismatch { expected: usize, got: usize },
    /// Fewer results available than requested.
    InsufficientResults(usize),
    /// Bad configuration value.
    ConfigurationError(String),
}
/// Strategy for diversity-aware re-ranking.
#[derive(Debug, Clone)]
pub enum DiversityStrategy {
    /// Maximal Marginal Relevance.  λ ∈ \[0,1\] trades off relevance vs. diversity.
    MaxMarginalRelevance(f64),
    /// Approximate Determinantal Point Process via greedy volume maximisation.
    DeterminantalPointProcess,
    /// Greedy diversity: add next best result only when it is far enough from all selected.
    GreedyDiversify(f64),
    /// No diversity re-ranking; sort purely by relevance score.
    None,
}
/// Session context that shapes query expansion and result personalisation.
#[derive(Debug, Clone)]
pub struct SearchContext {
    /// Unique session identifier (opaque string).
    pub session_id: String,
    /// Human-readable query texts in chronological order.
    pub query_history: Vec<String>,
    /// Positive example embeddings (documents the user liked).
    pub positive_examples: Vec<Vec<f64>>,
    /// Negative example embeddings (documents the user disliked).
    pub negative_examples: Vec<Vec<f64>>,
    /// How many recent queries (from the tail of `query_history`) affect expansion.
    pub context_window: usize,
    /// Embedding counterparts to `query_history` in chronological order.
    pub(crate) query_embeddings: Vec<Vec<f64>>,
}
impl SearchContext {
    /// Create a new, empty context for `session_id`.
    pub fn new(session_id: impl Into<String>, context_window: usize) -> Self {
        Self {
            session_id: session_id.into(),
            query_history: Vec::new(),
            positive_examples: Vec::new(),
            negative_examples: Vec::new(),
            context_window,
            query_embeddings: Vec::new(),
        }
    }
}
/// Running statistics for a [`ContextualEmbeddingSearch`] instance.
#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    /// Total number of search calls completed.
    pub queries_processed: u64,
    /// Cumulative average cosine similarity between original and expanded query.
    pub avg_expansion_similarity: f64,
    /// Number of times diversity re-ranking changed the result order.
    pub diversity_gains: u64,
    /// Number of times a cached result set was returned (future use).
    pub cache_hits: u64,
}
/// Context-aware embedding search engine with query expansion, negative suppression,
/// and diversity-aware re-ranking.
pub struct ContextualEmbeddingSearch {
    /// Document index: id → SearchDoc.
    pub(super) documents: HashMap<String, SearchDoc>,
    /// Insertion-ordered document IDs for deterministic iteration.
    pub(super) doc_order: Vec<String>,
    /// Expected embedding dimensionality (set on first insertion).
    pub(super) dimension: Option<usize>,
    /// Accumulated statistics.
    pub(super) stats: SearchStats,
}
impl ContextualEmbeddingSearch {
    /// Create an empty search engine.
    pub fn new() -> Self {
        Self {
            documents: HashMap::new(),
            doc_order: Vec::new(),
            dimension: None,
            stats: SearchStats::default(),
        }
    }
    /// Add a document to the index.
    ///
    /// The first document sets the expected embedding dimension for all subsequent
    /// insertions and queries.
    pub fn add_document(&mut self, doc: SearchDoc) -> Result<(), SearchError> {
        if doc.embedding.is_empty() {
            return Err(SearchError::ConfigurationError(
                "embedding must not be empty".into(),
            ));
        }
        match self.dimension {
            None => self.dimension = Some(doc.embedding.len()),
            Some(expected) if expected != doc.embedding.len() => {
                return Err(SearchError::DimensionMismatch {
                    expected,
                    got: doc.embedding.len(),
                });
            }
            _ => {}
        }
        let id = doc.id.clone();
        if !self.documents.contains_key(&id) {
            self.doc_order.push(id.clone());
        }
        self.documents.insert(id, doc);
        Ok(())
    }
    /// Remove a document from the index by ID.
    pub fn remove_document(&mut self, id: &str) -> Result<(), SearchError> {
        if self.documents.remove(id).is_none() {
            return Err(SearchError::ConfigurationError(format!(
                "document '{id}' not found"
            )));
        }
        self.doc_order.retain(|x| x != id);
        Ok(())
    }
    /// Compute the expanded query embedding.
    pub(super) fn expand_query(
        &self,
        query: &[f64],
        context: &SearchContext,
        config: &SearchConfig,
    ) -> CesExpandedQuery {
        let alpha = config.expansion_alpha.clamp(0.0, 1.0);
        if alpha <= 1e-10 {
            return CesExpandedQuery {
                original: query.to_vec(),
                expanded: query.to_vec(),
                expansion_weight: 0.0,
                history_weight: 0.0,
            };
        }
        let window = context.context_window.max(1);
        let recent: Vec<&Vec<f64>> = context
            .query_embeddings
            .iter()
            .rev()
            .take(window)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        let dim = query.len();
        let valid_recent: Vec<&[f64]> = recent
            .iter()
            .filter(|v| v.len() == dim)
            .map(|v| v.as_slice())
            .collect();
        let history_weight = if valid_recent.is_empty() { 0.0 } else { 1.0 };
        let mut context_vec = if valid_recent.is_empty() {
            query.to_vec()
        } else {
            let w = 1.0 / valid_recent.len() as f64;
            let pairs: Vec<(&[f64], f64)> = valid_recent.iter().map(|v| (*v, w)).collect();
            weighted_sum(&pairs)
        };
        let valid_pos: Vec<&[f64]> = context
            .positive_examples
            .iter()
            .filter(|v| v.len() == dim)
            .map(|v| v.as_slice())
            .collect();
        if !valid_pos.is_empty() {
            let pos_w = 0.5 / valid_pos.len() as f64;
            for (c, p) in context_vec.iter_mut().zip(
                weighted_sum(&valid_pos.iter().map(|v| (*v, pos_w)).collect::<Vec<_>>()).iter(),
            ) {
                *c += p;
            }
        }
        normalize_in_place(&mut context_vec);
        let pairs: Vec<(&[f64], f64)> = vec![(query, 1.0 - alpha), (&context_vec, alpha)];
        let mut expanded = weighted_sum(&pairs);
        normalize_in_place(&mut expanded);
        CesExpandedQuery {
            original: query.to_vec(),
            expanded,
            expansion_weight: alpha,
            history_weight,
        }
    }
    /// Suppress directions corresponding to negative examples.
    pub(super) fn suppress_negatives(&self, query: &mut [f64], context: &SearchContext) {
        let dim = query.len();
        let valid_neg: Vec<&[f64]> = context
            .negative_examples
            .iter()
            .filter(|v| v.len() == dim)
            .map(|v| v.as_slice())
            .collect();
        if valid_neg.is_empty() {
            return;
        }
        for neg in &valid_neg {
            let neg_norm_sq: f64 = neg.iter().map(|x| x * x).sum();
            if neg_norm_sq < 1e-10 {
                continue;
            }
            let proj: f64 = query
                .iter()
                .zip(neg.iter())
                .map(|(q, n)| q * n)
                .sum::<f64>()
                / neg_norm_sq;
            if proj > 0.0 {
                for (q, n) in query.iter_mut().zip(neg.iter()) {
                    *q -= proj * n;
                }
            }
        }
        normalize_in_place(query);
    }
    /// MMR: Maximal Marginal Relevance.
    pub(super) fn mmr_rerank(
        candidates: &[(String, f64, &[f64])],
        top_k: usize,
        lambda: f64,
    ) -> Vec<(String, f64, f64)> {
        let lambda = lambda.clamp(0.0, 1.0);
        let mut selected: Vec<usize> = Vec::with_capacity(top_k);
        let mut remaining: Vec<usize> = (0..candidates.len()).collect();
        while selected.len() < top_k && !remaining.is_empty() {
            let best_idx = if selected.is_empty() {
                remaining
                    .iter()
                    .copied()
                    .max_by(|&a, &b| {
                        candidates[a]
                            .1
                            .partial_cmp(&candidates[b].1)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(remaining[0])
            } else {
                remaining
                    .iter()
                    .copied()
                    .max_by(|&a, &b| {
                        let mmr_a = mmr_score(candidates, a, &selected, lambda);
                        let mmr_b = mmr_score(candidates, b, &selected, lambda);
                        mmr_a
                            .partial_cmp(&mmr_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(remaining[0])
            };
            let pos = remaining.iter().position(|&x| x == best_idx).unwrap_or(0);
            remaining.remove(pos);
            selected.push(best_idx);
        }
        selected
            .iter()
            .map(|&i| {
                let max_sim = max_similarity_to_selected(candidates, i, &selected);
                let div = 1.0 - max_sim.max(0.0);
                (candidates[i].0.clone(), candidates[i].1, div)
            })
            .collect()
    }
    /// Greedy diversify: skip candidates too close to already-selected ones.
    pub(super) fn greedy_diversify_rerank(
        candidates: &[(String, f64, &[f64])],
        top_k: usize,
        min_dist: f64,
    ) -> Vec<(String, f64, f64)> {
        let mut selected: Vec<usize> = Vec::with_capacity(top_k);
        for (i, _) in candidates.iter().enumerate() {
            if selected.len() >= top_k {
                break;
            }
            let too_close = selected.iter().any(|&s| {
                let dist = euclidean_distance(candidates[i].2, candidates[s].2);
                dist < min_dist
            });
            if !too_close || selected.is_empty() {
                selected.push(i);
            }
        }
        if selected.len() < top_k {
            for i in 0..candidates.len() {
                if selected.len() >= top_k {
                    break;
                }
                if !selected.contains(&i) {
                    selected.push(i);
                }
            }
        }
        selected
            .iter()
            .map(|&i| {
                let max_sim = if selected.len() > 1 {
                    selected
                        .iter()
                        .filter(|&&j| j != i)
                        .map(|&j| cosine_similarity(candidates[i].2, candidates[j].2))
                        .fold(f64::NEG_INFINITY, f64::max)
                } else {
                    0.0
                };
                let div = 1.0 - max_sim.clamp(0.0, 1.0);
                (candidates[i].0.clone(), candidates[i].1, div)
            })
            .collect()
    }
    /// Approximate DPP via greedy volume (kernel matrix determinant) maximisation.
    pub(super) fn dpp_rerank(
        candidates: &[(String, f64, &[f64])],
        top_k: usize,
        rng: &mut u64,
    ) -> Vec<(String, f64, f64)> {
        if candidates.is_empty() {
            return vec![];
        }
        let n = candidates.len();
        let mut selected: Vec<usize> = Vec::with_capacity(top_k);
        let mut remaining: Vec<usize> = (0..n).collect();
        let mut l: Vec<Vec<f64>> = vec![vec![0.0; top_k]; n];
        while selected.len() < top_k && !remaining.is_empty() {
            let step = selected.len();
            let best = if step == 0 {
                remaining
                    .iter()
                    .copied()
                    .max_by(|&a, &b| {
                        let va = candidates[a].1 + xorshift_f64(rng) * 1e-9;
                        let vb = candidates[b].1 + xorshift_f64(rng) * 1e-9;
                        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(remaining[0])
            } else {
                remaining
                    .iter()
                    .copied()
                    .max_by(|&a, &b| {
                        let ga = dpp_marginal(candidates, a, &selected, &l, step);
                        let gb = dpp_marginal(candidates, b, &selected, &l, step);
                        ga.partial_cmp(&gb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(remaining[0])
            };
            let k_best_best = kernel_val(candidates, best, best);
            let l_sq: f64 = (0..step).map(|t| l[best][t] * l[best][t]).sum();
            let diag = (k_best_best - l_sq).max(1e-10).sqrt();
            l[best][step] = diag;
            for &r in &remaining {
                if r == best {
                    continue;
                }
                let k_r_best = kernel_val(candidates, r, best);
                let cross: f64 = (0..step).map(|t| l[r][t] * l[best][t]).sum();
                if diag > 1e-10 {
                    l[r][step] = (k_r_best - cross) / diag;
                }
            }
            let pos = remaining.iter().position(|&x| x == best).unwrap_or(0);
            remaining.remove(pos);
            selected.push(best);
        }
        selected
            .iter()
            .map(|&i| {
                let max_sim = selected
                    .iter()
                    .filter(|&&j| j != i)
                    .map(|&j| cosine_similarity(candidates[i].2, candidates[j].2))
                    .fold(0.0_f64, f64::max);
                let div = 1.0 - max_sim.clamp(0.0, 1.0);
                (candidates[i].0.clone(), candidates[i].1, div)
            })
            .collect()
    }
    /// Search the index.
    ///
    /// # Errors
    ///
    /// * [`SearchError::IndexEmpty`] — no documents indexed.
    /// * [`SearchError::DimensionMismatch`] — query dimension doesn't match index.
    /// * [`SearchError::ConfigurationError`] — `top_k == 0` or `rerank_top_n == 0`.
    pub fn search(
        &mut self,
        query: &[f64],
        context: &SearchContext,
        config: &SearchConfig,
    ) -> Result<Vec<ContextualResult>, SearchError> {
        if config.top_k == 0 {
            return Err(SearchError::ConfigurationError("top_k must be > 0".into()));
        }
        if config.rerank_top_n == 0 {
            return Err(SearchError::ConfigurationError(
                "rerank_top_n must be > 0".into(),
            ));
        }
        if self.documents.is_empty() {
            return Err(SearchError::IndexEmpty);
        }
        let expected_dim = self.dimension.unwrap_or(query.len());
        if query.len() != expected_dim {
            return Err(SearchError::DimensionMismatch {
                expected: expected_dim,
                got: query.len(),
            });
        }
        let expanded_meta = self.expand_query(query, context, config);
        let mut effective_query = expanded_meta.expanded.clone();
        if config.use_negative_examples {
            self.suppress_negatives(&mut effective_query, context);
        }
        let rerank_n = config.rerank_top_n.min(self.documents.len());
        let mut scored: Vec<(String, f64)> = self
            .doc_order
            .iter()
            .filter_map(|id| {
                let doc = self.documents.get(id)?;
                let sim = cosine_similarity(&effective_query, &doc.embedding);
                if sim >= config.min_relevance {
                    Some((id.clone(), sim))
                } else {
                    None
                }
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(rerank_n);
        if scored.is_empty() {
            return Err(SearchError::InsufficientResults(0));
        }
        let candidates_owned: Vec<(String, f64, Vec<f64>)> = scored
            .iter()
            .map(|(id, rel)| {
                let emb = self
                    .documents
                    .get(id)
                    .map(|d| d.embedding.clone())
                    .unwrap_or_default();
                (id.clone(), *rel, emb)
            })
            .collect();
        let candidates: Vec<(String, f64, &[f64])> = candidates_owned
            .iter()
            .map(|(id, rel, emb)| (id.as_str().to_owned(), *rel, emb.as_slice()))
            .collect();
        let top_k = config.top_k.min(candidates.len());
        let relevance_before: Vec<f64> = candidates.iter().take(top_k).map(|c| c.1).collect();
        let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_1337u64;
        let reranked: Vec<(String, f64, f64)> = match &config.diversity_strategy {
            DiversityStrategy::MaxMarginalRelevance(lambda) => {
                Self::mmr_rerank(&candidates, top_k, *lambda)
            }
            DiversityStrategy::GreedyDiversify(min_dist) => {
                Self::greedy_diversify_rerank(&candidates, top_k, *min_dist)
            }
            DiversityStrategy::DeterminantalPointProcess => {
                Self::dpp_rerank(&candidates, top_k, &mut rng_state)
            }
            DiversityStrategy::None => candidates
                .iter()
                .take(top_k)
                .map(|(id, rel, emb)| {
                    let max_sim = candidates
                        .iter()
                        .filter(|(oid, _, _)| oid != id)
                        .take(top_k)
                        .map(|(_, _, oem)| cosine_similarity(emb, oem))
                        .fold(0.0_f64, f64::max);
                    let div = 1.0 - max_sim.clamp(0.0, 1.0);
                    (id.clone(), *rel, div)
                })
                .collect(),
        };
        let reranked_relevances: Vec<f64> = reranked.iter().map(|r| r.1).collect();
        let order_changed = relevance_before
            .iter()
            .zip(reranked_relevances.iter())
            .any(|(a, b)| (a - b).abs() > 1e-9);
        let expansion_sim = cosine_similarity(query, &expanded_meta.expanded);
        let results: Vec<ContextualResult> = reranked
            .into_iter()
            .enumerate()
            .map(|(idx, (doc_id, relevance_score, diversity_score))| {
                let final_score = (relevance_score + diversity_score) / 2.0;
                let explanation = vec![
                    ("relevance".to_string(), relevance_score),
                    ("diversity".to_string(), diversity_score),
                    (
                        "expansion_alpha".to_string(),
                        expanded_meta.expansion_weight,
                    ),
                    ("expansion_sim".to_string(), expansion_sim),
                    ("history_weight".to_string(), expanded_meta.history_weight),
                ];
                ContextualResult {
                    doc_id,
                    relevance_score,
                    diversity_score,
                    final_score,
                    rank: idx + 1,
                    explanation,
                }
            })
            .collect();
        self.stats.queries_processed += 1;
        let n = self.stats.queries_processed as f64;
        self.stats.avg_expansion_similarity =
            ((n - 1.0) * self.stats.avg_expansion_similarity + expansion_sim) / n;
        if order_changed {
            self.stats.diversity_gains += 1;
        }
        Ok(results)
    }
    /// Record a new query into `context`.
    ///
    /// Adds `query_text` to `query_history` and appends the corresponding embedding.
    pub fn update_context(&self, context: &mut SearchContext, query: &[f64], query_text: String) {
        context.query_history.push(query_text);
        context.query_embeddings.push(query.to_vec());
    }
    /// Execute multiple independent searches sharing the same context and config.
    pub fn batch_search(
        &mut self,
        queries: &[Vec<f64>],
        context: &SearchContext,
        config: &SearchConfig,
    ) -> Result<Vec<Vec<ContextualResult>>, SearchError> {
        if queries.is_empty() {
            return Ok(vec![]);
        }
        let mut all_results = Vec::with_capacity(queries.len());
        for query in queries {
            let results = self.search(query, context, config)?;
            all_results.push(results);
        }
        Ok(all_results)
    }
    /// Return accumulated search statistics.
    pub fn stats(&self) -> SearchStats {
        self.stats.clone()
    }
    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.documents.len()
    }
    /// `true` if no documents are indexed.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }
    /// Current embedding dimensionality, or `None` if the index is empty.
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }
}
