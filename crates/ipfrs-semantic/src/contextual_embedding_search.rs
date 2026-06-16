//! Contextual Embedding Search — context-aware vector search with query expansion,
//! negative example suppression, diversity-aware re-ranking, and rich result explanations.
//!
//! # Overview
//!
//! [`ContextualEmbeddingSearch`] maintains an in-memory flat index of [`SearchDoc`]s and
//! exposes a single [`ContextualEmbeddingSearch::search`] method that:
//!
//! 1. Expands the raw query embedding using recent query history and positive examples.
//! 2. Optionally suppresses directions associated with negative examples.
//! 3. Retrieves the top-`rerank_top_n` candidates via brute-force cosine similarity.
//! 4. Re-ranks those candidates with one of four [`DiversityStrategy`] variants.
//! 5. Returns up to `top_k` [`ContextualResult`]s with per-feature score explanations.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Pure-Rust vector math helpers
// ---------------------------------------------------------------------------

/// Cosine similarity in [-1, 1].  Returns 0 if either vector is zero-length.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-10 || nb < 1e-10 {
        0.0
    } else {
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }
}

/// Euclidean distance between two vectors.
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Weighted sum of (slice, weight) pairs.  All slices must share the same length.
pub fn weighted_sum(vecs: &[(&[f64], f64)]) -> Vec<f64> {
    if vecs.is_empty() {
        return vec![];
    }
    let dim = vecs[0].0.len();
    let mut result = vec![0.0f64; dim];
    for (v, w) in vecs {
        for (r, x) in result.iter_mut().zip(v.iter()) {
            *r += x * w;
        }
    }
    result
}

/// Normalize a vector to unit length in-place.  No-op when the vector is near zero.
fn normalize_in_place(v: &mut [f64]) {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ---------------------------------------------------------------------------
// Minimal xorshift PRNG (no rand crate)
// ---------------------------------------------------------------------------

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn xorshift_f64(state: &mut u64) -> f64 {
    (xorshift64(state) >> 11) as f64 / (1u64 << 53) as f64
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

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

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            diversity_strategy: DiversityStrategy::MaxMarginalRelevance(0.5),
            expansion_alpha: 0.3,
            use_negative_examples: true,
            rerank_top_n: 50,
            min_relevance: 0.0,
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

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IndexEmpty => write!(f, "index is empty"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::InsufficientResults(n) => {
                write!(f, "only {n} results available")
            }
            Self::ConfigurationError(msg) => write!(f, "configuration error: {msg}"),
        }
    }
}

impl std::error::Error for SearchError {}

// ---------------------------------------------------------------------------
// Core struct
// ---------------------------------------------------------------------------

/// Context-aware embedding search engine with query expansion, negative suppression,
/// and diversity-aware re-ranking.
pub struct ContextualEmbeddingSearch {
    /// Document index: id → SearchDoc.
    documents: HashMap<String, SearchDoc>,
    /// Insertion-ordered document IDs for deterministic iteration.
    doc_order: Vec<String>,
    /// Expected embedding dimensionality (set on first insertion).
    dimension: Option<usize>,
    /// Accumulated statistics.
    stats: SearchStats,
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

    // ------------------------------------------------------------------
    // Index management
    // ------------------------------------------------------------------

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
                })
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

    // ------------------------------------------------------------------
    // Query helpers
    // ------------------------------------------------------------------

    /// Compute the expanded query embedding.
    fn expand_query(
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

        // Gather recent query embeddings within the context window.
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

        // Compute history mean (if any recent embeddings match dimensionality).
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

        // Incorporate positive examples (with lower weight).
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

        // Blend: expanded = (1-α)*original + α*context.
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
    fn suppress_negatives(&self, query: &mut [f64], context: &SearchContext) {
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

        // Subtract the projection of the query onto each negative direction.
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
            // Only suppress if projection is positive (i.e., moving toward the negative).
            if proj > 0.0 {
                for (q, n) in query.iter_mut().zip(neg.iter()) {
                    *q -= proj * n;
                }
            }
        }
        normalize_in_place(query);
    }

    // ------------------------------------------------------------------
    // Diversity re-ranking strategies
    // ------------------------------------------------------------------

    /// MMR: Maximal Marginal Relevance.
    fn mmr_rerank(
        candidates: &[(String, f64, &[f64])], // (id, relevance, embedding)
        top_k: usize,
        lambda: f64,
    ) -> Vec<(String, f64, f64)> {
        // (id, relevance, diversity_score)
        let lambda = lambda.clamp(0.0, 1.0);
        let mut selected: Vec<usize> = Vec::with_capacity(top_k);
        let mut remaining: Vec<usize> = (0..candidates.len()).collect();

        while selected.len() < top_k && !remaining.is_empty() {
            let best_idx = if selected.is_empty() {
                // First pick: highest relevance.
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
                // Subsequent picks: maximise MMR score.
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
    fn greedy_diversify_rerank(
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

        // If we didn't fill top_k, backfill with closest remaining.
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
    fn dpp_rerank(
        candidates: &[(String, f64, &[f64])],
        top_k: usize,
        rng: &mut u64,
    ) -> Vec<(String, f64, f64)> {
        if candidates.is_empty() {
            return vec![];
        }
        let n = candidates.len();
        // Kernel: k(i,j) = relevance_i * cosine(i,j) * relevance_j
        // Greedy selection: add item maximising marginal log-det contribution.
        let mut selected: Vec<usize> = Vec::with_capacity(top_k);
        let mut remaining: Vec<usize> = (0..n).collect();

        // Cholesky-based incremental greedy DPP (simplified).
        // We maintain L factor implicitly via dot products.
        let mut l: Vec<Vec<f64>> = vec![vec![0.0; top_k]; n]; // l[i][step]

        while selected.len() < top_k && !remaining.is_empty() {
            let step = selected.len();
            let best = if step == 0 {
                // First: pick highest relevance with small random tie-breaking.
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
                // Compute marginal gain for each remaining item.
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

            // Update L factors for the chosen item.
            let k_best_best = kernel_val(candidates, best, best);
            let l_sq: f64 = (0..step).map(|t| l[best][t] * l[best][t]).sum();
            let diag = (k_best_best - l_sq).max(1e-10).sqrt();
            l[best][step] = diag;

            // Update remaining items' L entries.
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

    // ------------------------------------------------------------------
    // Main search
    // ------------------------------------------------------------------

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
        // Validate config.
        if config.top_k == 0 {
            return Err(SearchError::ConfigurationError("top_k must be > 0".into()));
        }
        if config.rerank_top_n == 0 {
            return Err(SearchError::ConfigurationError(
                "rerank_top_n must be > 0".into(),
            ));
        }

        // Validate index.
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

        // 1. Query expansion.
        let expanded_meta = self.expand_query(query, context, config);
        let mut effective_query = expanded_meta.expanded.clone();

        // 2. Negative example suppression.
        if config.use_negative_examples {
            self.suppress_negatives(&mut effective_query, context);
        }

        // 3. Brute-force initial retrieval.
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

        // Build candidate slice referencing live embeddings.
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

        // 4. Diversity re-ranking.
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

        // Check whether diversity changed the order.
        let reranked_relevances: Vec<f64> = reranked.iter().map(|r| r.1).collect();
        let order_changed = relevance_before
            .iter()
            .zip(reranked_relevances.iter())
            .any(|(a, b)| (a - b).abs() > 1e-9);

        // 5. Build ContextualResult list.
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

        // Update stats.
        self.stats.queries_processed += 1;
        let n = self.stats.queries_processed as f64;
        self.stats.avg_expansion_similarity =
            ((n - 1.0) * self.stats.avg_expansion_similarity + expansion_sim) / n;
        if order_changed {
            self.stats.diversity_gains += 1;
        }

        Ok(results)
    }

    // ------------------------------------------------------------------
    // Context update
    // ------------------------------------------------------------------

    /// Record a new query into `context`.
    ///
    /// Adds `query_text` to `query_history` and appends the corresponding embedding.
    pub fn update_context(&self, context: &mut SearchContext, query: &[f64], query_text: String) {
        context.query_history.push(query_text);
        context.query_embeddings.push(query.to_vec());
    }

    // ------------------------------------------------------------------
    // Batch search
    // ------------------------------------------------------------------

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

    // ------------------------------------------------------------------
    // Stats / accessors
    // ------------------------------------------------------------------

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

impl Default for ContextualEmbeddingSearch {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Private helpers used by re-ranking methods
// ---------------------------------------------------------------------------

/// MMR score for candidate `i` given already-selected items.
fn mmr_score(
    candidates: &[(String, f64, &[f64])],
    i: usize,
    selected: &[usize],
    lambda: f64,
) -> f64 {
    let rel = candidates[i].1;
    let max_sim = max_similarity_to_selected(candidates, i, selected);
    lambda * rel - (1.0 - lambda) * max_sim
}

/// Maximum cosine similarity between candidate `i` and any selected item.
fn max_similarity_to_selected(
    candidates: &[(String, f64, &[f64])],
    i: usize,
    selected: &[usize],
) -> f64 {
    if selected.is_empty() {
        return 0.0;
    }
    selected
        .iter()
        .map(|&s| cosine_similarity(candidates[i].2, candidates[s].2))
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0)
}

/// DPP kernel value: k(i,j) = rel_i * cos(i,j) * rel_j.
fn kernel_val(candidates: &[(String, f64, &[f64])], i: usize, j: usize) -> f64 {
    let cos = cosine_similarity(candidates[i].2, candidates[j].2);
    // Shift cosine to [0,1] to keep kernel positive semi-definite.
    let cos_shifted = (cos + 1.0) / 2.0;
    candidates[i].1 * cos_shifted * candidates[j].1
}

/// Marginal gain of adding candidate `i` to the current DPP selection.
fn dpp_marginal(
    candidates: &[(String, f64, &[f64])],
    i: usize,
    _selected: &[usize],
    l: &[Vec<f64>],
    step: usize,
) -> f64 {
    let k_ii = kernel_val(candidates, i, i);
    let l_sq: f64 = (0..step).map(|t| l[i][t] * l[i][t]).sum();
    (k_ii - l_sq).max(0.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------

    fn make_doc(id: &str, embedding: Vec<f64>) -> SearchDoc {
        SearchDoc {
            id: id.to_string(),
            embedding,
            metadata: vec![("key".to_string(), "val".to_string())],
        }
    }

    fn uniform_index(n: usize, dim: usize) -> ContextualEmbeddingSearch {
        let mut engine = ContextualEmbeddingSearch::new();
        for i in 0..n {
            let mut emb = vec![0.0f64; dim];
            // Spread docs across different directions.
            let angle = std::f64::consts::PI * 2.0 * (i as f64) / (n as f64);
            emb[0] = angle.cos();
            if dim > 1 {
                emb[1] = angle.sin();
            }
            engine
                .add_document(make_doc(&format!("doc{i}"), emb))
                .expect("test: add_document should succeed for uniform index doc");
        }
        engine
    }

    fn default_context() -> SearchContext {
        SearchContext::new("test-session", 5)
    }

    fn default_config() -> SearchConfig {
        SearchConfig {
            top_k: 5,
            rerank_top_n: 20,
            ..Default::default()
        }
    }

    // -------------------------------------------------------------------
    // Vector math
    // -------------------------------------------------------------------

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-9, "identical vectors: {sim}");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_weighted_sum_single() {
        let v = vec![1.0, 2.0, 3.0];
        let result = weighted_sum(&[(&v, 2.0)]);
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_weighted_sum_two() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let result = weighted_sum(&[(&a, 0.5), (&b, 0.5)]);
        assert!((result[0] - 0.5).abs() < 1e-9);
        assert!((result[1] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_sum_empty() {
        let result = weighted_sum(&[] as &[(&[f64], f64)]);
        assert!(result.is_empty());
    }

    // -------------------------------------------------------------------
    // Index management
    // -------------------------------------------------------------------

    #[test]
    fn test_add_document_sets_dimension() {
        let mut engine = ContextualEmbeddingSearch::new();
        engine
            .add_document(make_doc("a", vec![1.0, 2.0]))
            .expect("test: add_document should succeed for first doc");
        assert_eq!(engine.dimension(), Some(2));
    }

    #[test]
    fn test_add_document_dimension_mismatch() {
        let mut engine = ContextualEmbeddingSearch::new();
        engine
            .add_document(make_doc("a", vec![1.0, 2.0]))
            .expect("test: add_document should succeed for initial doc");
        let err = engine
            .add_document(make_doc("b", vec![1.0]))
            .expect_err("test: dimension mismatch should produce an error");
        assert_eq!(
            err,
            SearchError::DimensionMismatch {
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    fn test_add_duplicate_overwrites() {
        let mut engine = ContextualEmbeddingSearch::new();
        engine
            .add_document(make_doc("a", vec![1.0, 0.0]))
            .expect("test: add_document should succeed for first insert");
        engine
            .add_document(make_doc("a", vec![0.0, 1.0]))
            .expect("test: add_document should succeed for duplicate overwrite");
        assert_eq!(engine.len(), 1);
    }

    #[test]
    fn test_remove_document() {
        let mut engine = ContextualEmbeddingSearch::new();
        engine
            .add_document(make_doc("a", vec![1.0, 0.0]))
            .expect("test: add_document should succeed before remove");
        engine
            .remove_document("a")
            .expect("test: remove_document should succeed for existing doc");
        assert_eq!(engine.len(), 0);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut engine = ContextualEmbeddingSearch::new();
        let err = engine
            .remove_document("ghost")
            .expect_err("test: removing nonexistent doc should fail");
        matches!(err, SearchError::ConfigurationError(_));
    }

    #[test]
    fn test_add_empty_embedding() {
        let mut engine = ContextualEmbeddingSearch::new();
        let err = engine
            .add_document(make_doc("empty", vec![]))
            .expect_err("test: empty embedding should produce an error");
        matches!(err, SearchError::ConfigurationError(_));
    }

    #[test]
    fn test_is_empty_initially() {
        let engine = ContextualEmbeddingSearch::new();
        assert!(engine.is_empty());
    }

    #[test]
    fn test_len_after_adds() {
        let mut engine = ContextualEmbeddingSearch::new();
        for i in 0..5 {
            engine
                .add_document(make_doc(&format!("d{i}"), vec![i as f64, 0.0]))
                .expect("test: add_document should succeed for each doc in loop");
        }
        assert_eq!(engine.len(), 5);
    }

    // -------------------------------------------------------------------
    // Basic search
    // -------------------------------------------------------------------

    #[test]
    fn test_search_empty_index() {
        let mut engine = ContextualEmbeddingSearch::new();
        let ctx = default_context();
        let cfg = default_config();
        let err = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect_err("test: search on empty index should fail");
        assert_eq!(err, SearchError::IndexEmpty);
    }

    #[test]
    fn test_search_returns_top_k() {
        let mut engine = uniform_index(10, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 3,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: search should succeed and return results");
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_ranks_are_sequential() {
        let mut engine = uniform_index(5, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 5,
            rerank_top_n: 5,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: search should succeed returning ranked results");
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.rank, i + 1);
        }
    }

    #[test]
    fn test_search_query_dimension_mismatch() {
        let mut engine = uniform_index(3, 3);
        let ctx = default_context();
        let cfg = default_config();
        let err = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect_err("test: mismatched query dimension should fail");
        assert_eq!(
            err,
            SearchError::DimensionMismatch {
                expected: 3,
                got: 2
            }
        );
    }

    #[test]
    fn test_search_config_top_k_zero() {
        let mut engine = uniform_index(3, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 0,
            ..Default::default()
        };
        let err = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect_err("test: top_k=0 config should fail");
        matches!(err, SearchError::ConfigurationError(_));
    }

    #[test]
    fn test_search_config_rerank_top_n_zero() {
        let mut engine = uniform_index(3, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            rerank_top_n: 0,
            ..Default::default()
        };
        let err = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect_err("test: rerank_top_n=0 config should fail");
        matches!(err, SearchError::ConfigurationError(_));
    }

    #[test]
    fn test_search_top_k_capped_at_index_size() {
        let mut engine = uniform_index(3, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 100,
            rerank_top_n: 100,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: search should succeed with top_k capped at index size");
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_search_best_result_is_most_similar() {
        let mut engine = ContextualEmbeddingSearch::new();
        engine
            .add_document(make_doc("close", vec![1.0, 0.0]))
            .expect("test: add_document should succeed for close doc");
        engine
            .add_document(make_doc("far", vec![-1.0, 0.0]))
            .expect("test: add_document should succeed for far doc");
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 2,
            rerank_top_n: 2,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            min_relevance: -1.0,
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: search should succeed returning best result");
        assert_eq!(results[0].doc_id, "close");
    }

    #[test]
    fn test_search_min_relevance_filters() {
        let mut engine = uniform_index(8, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 10,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            min_relevance: 0.9,
            ..Default::default()
        };
        let results = engine.search(&[1.0, 0.0], &ctx, &cfg);
        // Either returns some high-sim results or InsufficientResults — both OK.
        match results {
            Ok(r) => {
                for res in &r {
                    assert!(res.relevance_score >= 0.9 - 1e-6);
                }
            }
            Err(SearchError::InsufficientResults(_)) => {}
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    // -------------------------------------------------------------------
    // Query expansion
    // -------------------------------------------------------------------

    #[test]
    fn test_expansion_no_alpha() {
        let engine = ContextualEmbeddingSearch::new();
        let ctx = default_context();
        let cfg = SearchConfig {
            expansion_alpha: 0.0,
            ..Default::default()
        };
        let eq = engine.expand_query(&[1.0, 0.0], &ctx, &cfg);
        assert_eq!(eq.original, vec![1.0, 0.0]);
        assert_eq!(eq.expanded, vec![1.0, 0.0]);
        assert!((eq.expansion_weight).abs() < 1e-9);
    }

    #[test]
    fn test_expansion_shifts_query_toward_history() {
        let mut ctx = SearchContext::new("s", 5);
        let engine = ContextualEmbeddingSearch::new();
        // History: queries pointing in (0,1) direction.
        ctx.query_embeddings.push(vec![0.0, 1.0]);
        ctx.query_embeddings.push(vec![0.0, 1.0]);
        let cfg = SearchConfig {
            expansion_alpha: 0.5,
            ..Default::default()
        };
        let eq = engine.expand_query(&[1.0, 0.0], &ctx, &cfg);
        // Expanded should have a positive Y component now.
        assert!(
            eq.expanded[1] > 0.01,
            "expected y > 0, got {:?}",
            eq.expanded
        );
    }

    #[test]
    fn test_expansion_with_positive_examples() {
        let mut ctx = SearchContext::new("s", 5);
        let engine = ContextualEmbeddingSearch::new();
        ctx.positive_examples.push(vec![0.0, 1.0]);
        let cfg = SearchConfig {
            expansion_alpha: 0.5,
            ..Default::default()
        };
        let eq = engine.expand_query(&[1.0, 0.0], &ctx, &cfg);
        // Positive example in Y direction should increase Y component.
        assert!(eq.expanded[1] > 0.0);
    }

    #[test]
    fn test_expansion_weight_stored() {
        let engine = ContextualEmbeddingSearch::new();
        let ctx = default_context();
        let cfg = SearchConfig {
            expansion_alpha: 0.4,
            ..Default::default()
        };
        let eq = engine.expand_query(&[1.0, 0.0], &ctx, &cfg);
        assert!((eq.expansion_weight - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_expansion_history_weight_zero_when_no_history() {
        let engine = ContextualEmbeddingSearch::new();
        let ctx = default_context();
        let cfg = SearchConfig {
            expansion_alpha: 0.5,
            ..Default::default()
        };
        let eq = engine.expand_query(&[1.0, 0.0], &ctx, &cfg);
        assert!((eq.history_weight).abs() < 1e-9);
    }

    // -------------------------------------------------------------------
    // Negative example suppression
    // -------------------------------------------------------------------

    #[test]
    fn test_negative_suppression_reduces_projection() {
        let engine = ContextualEmbeddingSearch::new();
        let mut ctx = SearchContext::new("s", 5);
        ctx.negative_examples.push(vec![0.0, 1.0]);

        let mut query = vec![0.5, 0.5];
        normalize_in_place(&mut query);
        let original_y = query[1];
        engine.suppress_negatives(&mut query, &ctx);
        assert!(
            query[1] < original_y,
            "Y component should decrease after suppression"
        );
    }

    #[test]
    fn test_negative_suppression_no_effect_orthogonal() {
        let engine = ContextualEmbeddingSearch::new();
        let mut ctx = SearchContext::new("s", 5);
        // Query is (1,0), negative is (0,1) — orthogonal; suppression should be a no-op.
        ctx.negative_examples.push(vec![0.0, 1.0]);

        let mut query = vec![1.0, 0.0];
        engine.suppress_negatives(&mut query, &ctx);
        assert!((query[0] - 1.0).abs() < 1e-6);
        assert!(query[1].abs() < 1e-6);
    }

    #[test]
    fn test_negative_suppression_uses_config() {
        let mut engine = uniform_index(5, 2);
        let mut ctx = SearchContext::new("s", 5);
        ctx.negative_examples.push(vec![-1.0, 0.0]); // push away from -X
        let cfg_with = SearchConfig {
            use_negative_examples: true,
            expansion_alpha: 0.0,
            top_k: 5,
            rerank_top_n: 5,
            diversity_strategy: DiversityStrategy::None,
            min_relevance: -1.0,
        };
        let cfg_without = SearchConfig {
            use_negative_examples: false,
            ..cfg_with.clone()
        };
        // Both should succeed; results may differ.
        engine
            .search(&[1.0, 0.0], &ctx, &cfg_with)
            .expect("test: search with negative examples enabled should succeed");
        engine
            .search(&[1.0, 0.0], &ctx, &cfg_without)
            .expect("test: search with negative examples disabled should succeed");
    }

    // -------------------------------------------------------------------
    // DiversityStrategy::None
    // -------------------------------------------------------------------

    #[test]
    fn test_diversity_none_sorted_by_relevance() {
        let mut engine = uniform_index(6, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 4,
            rerank_top_n: 6,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: search with None strategy should succeed");
        for w in results.windows(2) {
            assert!(
                w[0].relevance_score >= w[1].relevance_score - 1e-9,
                "not sorted by relevance"
            );
        }
    }

    // -------------------------------------------------------------------
    // DiversityStrategy::MaxMarginalRelevance
    // -------------------------------------------------------------------

    #[test]
    fn test_mmr_lambda_1_is_pure_relevance() {
        // λ=1 → MMR = relevance, should produce same order as None.
        let mut engine = uniform_index(8, 2);
        let ctx = default_context();
        let mk = |strategy| SearchConfig {
            top_k: 4,
            rerank_top_n: 8,
            expansion_alpha: 0.0,
            diversity_strategy: strategy,
            ..Default::default()
        };
        let r_none = engine
            .search(&[1.0, 0.0], &ctx, &mk(DiversityStrategy::None))
            .expect("test: search with None diversity should succeed");
        let r_mmr = engine
            .search(
                &[1.0, 0.0],
                &ctx,
                &mk(DiversityStrategy::MaxMarginalRelevance(1.0)),
            )
            .expect("test: search with MMR lambda=1 should succeed");
        // First result should be the same.
        assert_eq!(r_none[0].doc_id, r_mmr[0].doc_id);
    }

    #[test]
    fn test_mmr_lambda_0_maximises_diversity() {
        let mut engine = uniform_index(10, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 5,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::MaxMarginalRelevance(0.0),
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: MMR lambda=0 search should succeed");
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_mmr_diversity_scores_present() {
        let mut engine = uniform_index(10, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 5,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::MaxMarginalRelevance(0.5),
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: MMR search should succeed returning diversity scores");
        for r in &results {
            assert!(r.diversity_score >= 0.0 && r.diversity_score <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn test_mmr_correct_first_pick() {
        // First pick in MMR must be the highest-relevance doc.
        let mut engine = ContextualEmbeddingSearch::new();
        engine
            .add_document(make_doc("best", vec![1.0, 0.0]))
            .expect("test: add_document should succeed for best doc");
        engine
            .add_document(make_doc("second", vec![0.7, 0.7]))
            .expect("test: add_document should succeed for second doc");
        engine
            .add_document(make_doc("third", vec![-1.0, 0.0]))
            .expect("test: add_document should succeed for third doc");
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 3,
            rerank_top_n: 3,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::MaxMarginalRelevance(0.5),
            min_relevance: -1.0,
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: MMR search should succeed identifying best first pick");
        assert_eq!(results[0].doc_id, "best");
    }

    #[test]
    fn test_mmr_single_doc() {
        let mut engine = ContextualEmbeddingSearch::new();
        engine
            .add_document(make_doc("only", vec![1.0, 0.0]))
            .expect("test: add_document should succeed for single doc");
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 1,
            rerank_top_n: 1,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::MaxMarginalRelevance(0.5),
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: MMR search with single doc should succeed");
        assert_eq!(results.len(), 1);
    }

    // -------------------------------------------------------------------
    // DiversityStrategy::GreedyDiversify
    // -------------------------------------------------------------------

    #[test]
    fn test_greedy_diversify_basic() {
        let mut engine = uniform_index(10, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 5,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::GreedyDiversify(0.1),
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: greedy diversify search should succeed");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_greedy_diversify_strict_threshold_backfills() {
        // With a very large min_dist, we still get top_k results (backfill).
        let mut engine = uniform_index(10, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 5,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::GreedyDiversify(999.0),
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: greedy diversify with strict threshold should succeed");
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_greedy_diversify_zero_threshold_like_none() {
        let mut engine = uniform_index(8, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 4,
            rerank_top_n: 8,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::GreedyDiversify(0.0),
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: greedy diversify with zero threshold should succeed");
        assert_eq!(results.len(), 4);
    }

    // -------------------------------------------------------------------
    // DiversityStrategy::DeterminantalPointProcess
    // -------------------------------------------------------------------

    #[test]
    fn test_dpp_basic() {
        let mut engine = uniform_index(10, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 5,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::DeterminantalPointProcess,
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: DPP search should succeed");
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_dpp_scores_in_range() {
        let mut engine = uniform_index(10, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 5,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::DeterminantalPointProcess,
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: DPP search should succeed returning scored results");
        for r in &results {
            assert!((0.0..=1.0 + 1e-6).contains(&r.diversity_score));
        }
    }

    #[test]
    fn test_dpp_single_doc() {
        let mut engine = ContextualEmbeddingSearch::new();
        engine
            .add_document(make_doc("a", vec![1.0, 0.0]))
            .expect("test: add_document should succeed for single DPP doc");
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 1,
            rerank_top_n: 1,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::DeterminantalPointProcess,
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: DPP search with single doc should succeed");
        assert_eq!(results.len(), 1);
    }

    // -------------------------------------------------------------------
    // Final score and explanation
    // -------------------------------------------------------------------

    #[test]
    fn test_final_score_is_average() {
        let mut engine = uniform_index(5, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 3,
            rerank_top_n: 5,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: search should succeed for final score verification");
        for r in &results {
            let expected = (r.relevance_score + r.diversity_score) / 2.0;
            assert!((r.final_score - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn test_explanation_contains_features() {
        let mut engine = uniform_index(5, 2);
        let ctx = default_context();
        let cfg = default_config();
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: search should succeed to check explanation features");
        let keys: Vec<&str> = results[0]
            .explanation
            .iter()
            .map(|(k, _)| k.as_str())
            .collect();
        assert!(keys.contains(&"relevance"));
        assert!(keys.contains(&"diversity"));
        assert!(keys.contains(&"expansion_alpha"));
    }

    // -------------------------------------------------------------------
    // Context update
    // -------------------------------------------------------------------

    #[test]
    fn test_update_context_adds_history() {
        let engine = ContextualEmbeddingSearch::new();
        let mut ctx = default_context();
        engine.update_context(&mut ctx, &[1.0, 0.0], "first query".to_string());
        assert_eq!(ctx.query_history.len(), 1);
        assert_eq!(ctx.query_embeddings.len(), 1);
    }

    #[test]
    fn test_update_context_multiple_queries() {
        let engine = ContextualEmbeddingSearch::new();
        let mut ctx = default_context();
        for i in 0..5 {
            engine.update_context(&mut ctx, &[i as f64, 0.0], format!("query {i}"));
        }
        assert_eq!(ctx.query_history.len(), 5);
        assert_eq!(ctx.query_embeddings.len(), 5);
    }

    #[test]
    fn test_update_context_then_search_uses_history() {
        let mut engine = uniform_index(10, 2);
        let mut ctx = SearchContext::new("s", 5);
        let helper = ContextualEmbeddingSearch::new();
        // Add history pointing to (0,1).
        helper.update_context(&mut ctx, &[0.0, 1.0], "q1".to_string());
        helper.update_context(&mut ctx, &[0.0, 1.0], "q2".to_string());
        let cfg = SearchConfig {
            top_k: 3,
            rerank_top_n: 10,
            expansion_alpha: 0.5,
            diversity_strategy: DiversityStrategy::None,
            ..Default::default()
        };
        // Should succeed without error.
        engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: search with expanded context history should succeed");
    }

    // -------------------------------------------------------------------
    // Batch search
    // -------------------------------------------------------------------

    #[test]
    fn test_batch_search_empty_queries() {
        let mut engine = uniform_index(5, 2);
        let ctx = default_context();
        let cfg = default_config();
        let results = engine
            .batch_search(&[], &ctx, &cfg)
            .expect("test: batch_search with empty queries should succeed");
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_search_multiple_queries() {
        let mut engine = uniform_index(10, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 3,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            ..Default::default()
        };
        let queries = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
        let results = engine
            .batch_search(&queries, &ctx, &cfg)
            .expect("test: batch_search with multiple queries should succeed");
        assert_eq!(results.len(), 3);
        for r in &results {
            assert_eq!(r.len(), 3);
        }
    }

    #[test]
    fn test_batch_search_propagates_error() {
        let mut engine = ContextualEmbeddingSearch::new();
        let ctx = default_context();
        let cfg = default_config();
        let queries = vec![vec![1.0, 0.0]];
        let err = engine
            .batch_search(&queries, &ctx, &cfg)
            .expect_err("test: batch_search on empty index should fail");
        assert_eq!(err, SearchError::IndexEmpty);
    }

    #[test]
    fn test_batch_search_independent_results() {
        let mut engine = uniform_index(10, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 3,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            ..Default::default()
        };
        let q1 = vec![1.0, 0.0];
        let q2 = vec![-1.0, 0.0];
        let batch = engine
            .batch_search(&[q1.clone(), q2.clone()], &ctx, &cfg)
            .expect("test: batch_search should succeed for independent results");
        let single1 = engine
            .search(&q1, &ctx, &cfg)
            .expect("test: single search should succeed for comparison");
        // Batch and single should agree on first result IDs.
        assert_eq!(batch[0][0].doc_id, single1[0].doc_id);
    }

    // -------------------------------------------------------------------
    // Stats
    // -------------------------------------------------------------------

    #[test]
    fn test_stats_initial_zero() {
        let engine = ContextualEmbeddingSearch::new();
        let s = engine.stats();
        assert_eq!(s.queries_processed, 0);
        assert_eq!(s.cache_hits, 0);
    }

    #[test]
    fn test_stats_queries_processed_increments() {
        let mut engine = uniform_index(5, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 3,
            rerank_top_n: 5,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            ..Default::default()
        };
        engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: first search should succeed for stats tracking");
        engine
            .search(&[0.0, 1.0], &ctx, &cfg)
            .expect("test: second search should succeed for stats tracking");
        assert_eq!(engine.stats().queries_processed, 2);
    }

    #[test]
    fn test_stats_avg_expansion_similarity_updates() {
        let mut engine = uniform_index(5, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 3,
            rerank_top_n: 5,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            ..Default::default()
        };
        engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: search should succeed for avg expansion similarity check");
        // With alpha=0, expansion_sim should be 1.0.
        assert!((engine.stats().avg_expansion_similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_stats_batch_updates_correctly() {
        let mut engine = uniform_index(5, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 3,
            rerank_top_n: 5,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            ..Default::default()
        };
        let queries: Vec<Vec<f64>> = vec![vec![1.0, 0.0]; 4];
        engine
            .batch_search(&queries, &ctx, &cfg)
            .expect("test: batch_search should succeed for stats update check");
        assert_eq!(engine.stats().queries_processed, 4);
    }

    // -------------------------------------------------------------------
    // Error cases
    // -------------------------------------------------------------------

    #[test]
    fn test_error_display_index_empty() {
        let e = SearchError::IndexEmpty;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_error_display_dimension_mismatch() {
        let e = SearchError::DimensionMismatch {
            expected: 3,
            got: 2,
        };
        assert!(e.to_string().contains('3'));
        assert!(e.to_string().contains('2'));
    }

    #[test]
    fn test_error_display_insufficient_results() {
        let e = SearchError::InsufficientResults(5);
        assert!(e.to_string().contains('5'));
    }

    #[test]
    fn test_error_display_configuration() {
        let e = SearchError::ConfigurationError("bad value".to_string());
        assert!(e.to_string().contains("bad value"));
    }

    // -------------------------------------------------------------------
    // SearchContext helpers
    // -------------------------------------------------------------------

    #[test]
    fn test_search_context_new() {
        let ctx = SearchContext::new("my-session", 10);
        assert_eq!(ctx.session_id, "my-session");
        assert_eq!(ctx.context_window, 10);
        assert!(ctx.query_history.is_empty());
    }

    #[test]
    fn test_search_context_positive_negative() {
        let mut ctx = SearchContext::new("s", 5);
        ctx.positive_examples.push(vec![1.0, 0.0]);
        ctx.negative_examples.push(vec![-1.0, 0.0]);
        assert_eq!(ctx.positive_examples.len(), 1);
        assert_eq!(ctx.negative_examples.len(), 1);
    }

    // -------------------------------------------------------------------
    // Determinism / reproducibility
    // -------------------------------------------------------------------

    #[test]
    fn test_search_deterministic() {
        let mut engine = uniform_index(10, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 5,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::MaxMarginalRelevance(0.5),
            ..Default::default()
        };
        let r1 = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: first deterministic search should succeed");
        let r2 = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: second deterministic search should succeed");
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.doc_id, b.doc_id);
        }
    }

    #[test]
    fn test_dpp_deterministic() {
        let mut engine = uniform_index(10, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 5,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::DeterminantalPointProcess,
            ..Default::default()
        };
        let r1 = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: first DPP deterministic search should succeed");
        let r2 = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: second DPP deterministic search should succeed");
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.doc_id, b.doc_id);
        }
    }

    // -------------------------------------------------------------------
    // Default impls
    // -------------------------------------------------------------------

    #[test]
    fn test_default_search_config() {
        let cfg = SearchConfig::default();
        assert_eq!(cfg.top_k, 10);
        assert!(cfg.use_negative_examples);
    }

    #[test]
    fn test_default_contextual_embedding_search() {
        let engine = ContextualEmbeddingSearch::default();
        assert!(engine.is_empty());
    }
}
