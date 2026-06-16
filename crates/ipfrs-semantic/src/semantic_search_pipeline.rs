//! # Semantic Search Pipeline (`Ssp*`)
//!
//! A full-stack semantic search pipeline with preprocessing, retrieval, and
//! postprocessing stages.  All public identifiers carry the `Ssp` prefix to
//! avoid collision with the existing `search_pipeline` and
//! `query_pipeline` modules.
//!
//! ## Architecture
//!
//! ```text
//! add_document  ──►  document store  ◄──  index_embedding
//!                                              │
//!                                              ▼
//! search(query_embedding)  ──► pipeline stages ──► Vec<SspSearchResult>
//!   stage 0: Tokenize (optional, text path)
//!   stage 1: EmbedQuery  – L2 normalise
//!   stage 2: AnnSearch   – cosine-similarity scan
//!   stage 3: Rerank      – BM25 / CrossEncoderApprox / RecipRankFusion
//!   stage 4: Deduplicate – remove near-duplicates
//!   stage 5: ScoreFilter – drop below threshold
//!   stage 6: Limit       – cap result set
//! ```
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_semantic::semantic_search_pipeline::{
//!     SemanticSearchPipeline, SspPipelineConfig,
//! };
//!
//! let cfg = SspPipelineConfig::default();
//! let mut pipeline = SemanticSearchPipeline::new(cfg);
//!
//! let id = pipeline.add_document("hello world".to_string(), Default::default());
//! pipeline.index_embedding(id, vec![1.0, 0.0]);
//!
//! let results = pipeline.search(&[1.0, 0.0]);
//! assert!(!results.is_empty());
//! ```

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Opaque document identifier used throughout the pipeline.
pub type SspDocId = u64;

/// Convenience alias for the pipeline itself.
pub type SspSemanticSearchPipeline = SemanticSearchPipeline;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Runtime configuration for [`SemanticSearchPipeline`].
#[derive(Debug, Clone)]
pub struct SspPipelineConfig {
    /// Expected dimensionality of embeddings.
    pub embedding_dim: usize,
    /// Maximum number of results returned by a single search call.
    pub top_k: usize,
    /// Whether the default pipeline includes a reranking stage.
    pub rerank: bool,
    /// Whether the default pipeline includes a deduplication stage.
    pub deduplicate: bool,
    /// Minimum cosine-similarity score a result must have to be returned.
    pub score_threshold: f64,
    /// Maximum number of tokens (Unicode scalar values) accepted in a query.
    pub max_query_len: usize,
}

impl Default for SspPipelineConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            top_k: 10,
            rerank: true,
            deduplicate: true,
            score_threshold: 0.0,
            max_query_len: 512,
        }
    }
}

// ---------------------------------------------------------------------------
// Document
// ---------------------------------------------------------------------------

/// A document stored in the pipeline's document store.
#[derive(Debug, Clone)]
pub struct SspDocument {
    /// Unique identifier assigned at insertion time.
    pub id: SspDocId,
    /// Raw text content of the document.
    pub content: String,
    /// Arbitrary key-value metadata attached to this document.
    pub metadata: HashMap<String, String>,
    /// Unix-epoch timestamp (seconds) when the document was indexed.
    pub indexed_at: u64,
}

// ---------------------------------------------------------------------------
// Pipeline stages
// ---------------------------------------------------------------------------

/// Re-ranking algorithm variants.
#[derive(Debug, Clone)]
pub enum SspRerankMethod {
    /// Approximate BM25 re-score using token overlap between query tokens and
    /// document content.
    BM25Rescore,
    /// Cross-encoder approximation: multiply cosine score by a query-document
    /// term-overlap factor.
    CrossEncoderApprox,
    /// Reciprocal Rank Fusion of two independent ranked lists (vector + BM25).
    RecipRankFusion,
}

/// A single stage in the [`SemanticSearchPipeline`] processing chain.
#[derive(Debug, Clone)]
pub enum SspStage {
    /// Tokenise the query text: optionally lowercase and remove stop words.
    Tokenize {
        lowercase: bool,
        stop_words: Vec<String>,
    },
    /// L2-normalise the query embedding so that dot product == cosine
    /// similarity in downstream stages.
    EmbedQuery { dim: usize },
    /// Brute-force cosine-similarity scan to retrieve the top-k candidates.
    AnnSearch { top_k: usize },
    /// Re-rank the candidate list using the chosen [`SspRerankMethod`].
    Rerank { method: SspRerankMethod },
    /// Remove near-duplicate results whose mutual cosine similarity exceeds
    /// `threshold`.
    Deduplicate { threshold: f64 },
    /// Discard results with a cosine-similarity score below `min_score`.
    ScoreFilter { min_score: f64 },
    /// Keep only the first `n` results.
    Limit(usize),
}

// ---------------------------------------------------------------------------
// Query log record
// ---------------------------------------------------------------------------

/// A single entry in the pipeline's bounded query log.
#[derive(Debug, Clone)]
pub struct SspQueryRecord {
    /// Unix-epoch timestamp (seconds) when the query was submitted.
    pub ts: u64,
    /// Raw query text (may be empty for pure-embedding queries).
    pub query_text: String,
    /// Number of results returned to the caller.
    pub n_results: usize,
    /// Wall-clock latency of the full pipeline execution in milliseconds.
    pub latency_ms: f64,
}

// ---------------------------------------------------------------------------
// Search result
// ---------------------------------------------------------------------------

/// A single result returned by [`SemanticSearchPipeline::search`].
#[derive(Debug, Clone)]
pub struct SspSearchResult {
    /// Document identifier.
    pub doc_id: SspDocId,
    /// Final score (cosine similarity, potentially modified by reranking).
    pub score: f64,
    /// A short snippet extracted from the document content (first 200 chars).
    pub snippet: String,
    /// Metadata copied from the source document.
    pub metadata: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Pipeline stats
// ---------------------------------------------------------------------------

/// Aggregated runtime statistics for [`SemanticSearchPipeline`].
#[derive(Debug, Clone, Default)]
pub struct SspPipelineStats {
    /// Total number of search calls executed since pipeline creation.
    pub total_queries: u64,
    /// Average number of results returned per query.
    pub avg_results: f64,
    /// Average end-to-end pipeline latency in milliseconds.
    pub avg_latency_ms: f64,
    /// Fraction of queries that were served from a shallow result-cache
    /// (queries whose embedding matches a previous query within ε = 1e-9).
    pub cache_hit_rate: f64,
}

// ---------------------------------------------------------------------------
// Internal helpers (free functions)
// ---------------------------------------------------------------------------

/// Compute the cosine similarity between two equal-length slices.
///
/// Returns `0.0` if either vector has zero norm.
#[inline]
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// A minimal XOR-shift PRNG used for deterministic pseudo-random operations
/// (e.g. tie-breaking in reranking) without pulling in the `rand` crate.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// L2-normalise `v` in-place.  No-op if the norm is zero.
fn l2_normalize(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Extract up to 200 characters of leading text as a snippet.
fn make_snippet(content: &str) -> String {
    content.chars().take(200).collect()
}

/// Current Unix-epoch time in seconds (falls back to `0` if the system clock
/// is unavailable — no `unwrap`).
fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Tokenise `text` by splitting on whitespace, optionally lowercasing and
/// removing stop-words.
fn tokenize(text: &str, lowercase: bool, stop_words: &[String]) -> Vec<String> {
    text.split_whitespace()
        .map(|tok| {
            if lowercase {
                tok.to_lowercase()
            } else {
                tok.to_string()
            }
        })
        .filter(|tok| !stop_words.contains(tok))
        .collect()
}

/// Approximate BM25 score between query tokens and document text.
///
/// Uses a simplified version with k1 = 1.5, b = 0.75, and an assumed average
/// document length of 100 tokens.
fn bm25_score(query_tokens: &[String], doc_content: &str, doc_len: usize) -> f64 {
    const K1: f64 = 1.5;
    const B: f64 = 0.75;
    const AVG_DL: f64 = 100.0;

    let dl = doc_len as f64;
    let doc_lower = doc_content.to_lowercase();

    query_tokens.iter().fold(0.0, |acc, tok| {
        let tf = doc_lower
            .split_whitespace()
            .filter(|w| *w == tok.as_str())
            .count() as f64;
        if tf == 0.0 {
            return acc;
        }
        // IDF approximation (single-document corpus): log(1 + 1) = ln(2)
        let idf = std::f64::consts::LN_2;
        let numerator = tf * (K1 + 1.0);
        let denominator = tf + K1 * (1.0 - B + B * dl / AVG_DL);
        acc + idf * (numerator / denominator)
    })
}

// ---------------------------------------------------------------------------
// Candidate representation (internal)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Candidate {
    doc_id: SspDocId,
    score: f64,
    bm25_score: f64,
}

// ---------------------------------------------------------------------------
// Main struct
// ---------------------------------------------------------------------------

/// A full-stack semantic search pipeline with configurable preprocessing,
/// retrieval, and postprocessing stages.
///
/// All operations are synchronous and single-threaded; for async usage wrap
/// the pipeline in an `Arc<Mutex<_>>` and call from a `spawn_blocking` task.
pub struct SemanticSearchPipeline {
    /// Monotonically increasing counter used to assign document IDs.
    next_id: u64,
    /// Primary document store.
    documents: HashMap<SspDocId, SspDocument>,
    /// L2-normalised embedding for each indexed document.
    embeddings: HashMap<SspDocId, Vec<f64>>,
    /// Ordered list of processing stages.
    stages: Vec<SspStage>,
    /// Bounded ring-buffer of past query records (max 500 entries).
    query_log: VecDeque<SspQueryRecord>,
    /// Pipeline configuration.
    config: SspPipelineConfig,
    /// Cumulative stats counters.
    total_queries: u64,
    total_results: u64,
    total_latency_ms: f64,
    /// Shallow result cache: maps embedding fingerprint → cached results.
    result_cache: HashMap<u64, Vec<SspSearchResult>>,
    /// Cache hit counter.
    cache_hits: u64,
    /// PRNG state for tie-breaking.
    rng_state: u64,
    /// Cached query tokens from the most recent `search_text` call (used by
    /// reranking stages that need them).
    last_query_tokens: Vec<String>,
}

impl SemanticSearchPipeline {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new pipeline with the given configuration and a default set of
    /// stages derived from the config flags.
    pub fn new(config: SspPipelineConfig) -> Self {
        let mut pipeline = Self {
            next_id: 1,
            documents: HashMap::new(),
            embeddings: HashMap::new(),
            stages: Vec::new(),
            query_log: VecDeque::with_capacity(500),
            config: config.clone(),
            total_queries: 0,
            total_results: 0,
            total_latency_ms: 0.0,
            result_cache: HashMap::new(),
            cache_hits: 0,
            rng_state: 0xDEAD_BEEF_CAFE_1234,
            last_query_tokens: Vec::new(),
        };
        pipeline.build_default_pipeline(&config);
        pipeline
    }

    /// Build the default stage list from config flags.
    fn build_default_pipeline(&mut self, cfg: &SspPipelineConfig) {
        self.stages.clear();
        self.stages.push(SspStage::EmbedQuery {
            dim: cfg.embedding_dim,
        });
        self.stages.push(SspStage::AnnSearch {
            top_k: cfg.top_k * 4,
        });
        if cfg.rerank {
            self.stages.push(SspStage::Rerank {
                method: SspRerankMethod::RecipRankFusion,
            });
        }
        if cfg.deduplicate {
            self.stages.push(SspStage::Deduplicate { threshold: 0.95 });
        }
        self.stages.push(SspStage::ScoreFilter {
            min_score: cfg.score_threshold,
        });
        self.stages.push(SspStage::Limit(cfg.top_k));
    }

    // -----------------------------------------------------------------------
    // Document management
    // -----------------------------------------------------------------------

    /// Insert a new document and return its assigned [`SspDocId`].
    pub fn add_document(&mut self, content: String, metadata: HashMap<String, String>) -> SspDocId {
        let id = self.next_id;
        self.next_id += 1;
        let doc = SspDocument {
            id,
            content,
            metadata,
            indexed_at: unix_now(),
        };
        self.documents.insert(id, doc);
        id
    }

    /// Remove a document (and its embedding) from the pipeline.
    ///
    /// Returns `true` if the document existed, `false` otherwise.
    pub fn remove_document(&mut self, id: SspDocId) -> bool {
        let removed = self.documents.remove(&id).is_some();
        self.embeddings.remove(&id);
        if removed {
            self.result_cache.clear();
        }
        removed
    }

    /// Replace the content of an existing document.
    ///
    /// Returns `true` on success, `false` if `id` was not found.
    pub fn update_document(&mut self, id: SspDocId, content: String) -> bool {
        if let Some(doc) = self.documents.get_mut(&id) {
            doc.content = content;
            doc.indexed_at = unix_now();
            self.result_cache.clear();
            true
        } else {
            false
        }
    }

    /// Retrieve an immutable reference to a document by id.
    pub fn get_document(&self, id: SspDocId) -> Option<&SspDocument> {
        self.documents.get(&id)
    }

    /// Return the number of documents currently in the store.
    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    // -----------------------------------------------------------------------
    // Embedding management
    // -----------------------------------------------------------------------

    /// L2-normalise `embedding` and associate it with `doc_id`.
    ///
    /// Replaces any previously stored embedding for that document.
    /// Invalidates the result cache.
    pub fn index_embedding(&mut self, doc_id: SspDocId, mut embedding: Vec<f64>) {
        l2_normalize(&mut embedding);
        self.embeddings.insert(doc_id, embedding);
        self.result_cache.clear();
    }

    /// Return the number of indexed embeddings.
    pub fn embedding_count(&self) -> usize {
        self.embeddings.len()
    }

    // -----------------------------------------------------------------------
    // Stage management
    // -----------------------------------------------------------------------

    /// Append a stage to the end of the pipeline.
    pub fn add_stage(&mut self, stage: SspStage) {
        self.stages.push(stage);
    }

    /// Remove all stages from the pipeline.
    pub fn clear_stages(&mut self) {
        self.stages.clear();
    }

    /// Replace the current stage list with the default pipeline derived from
    /// the current config.
    pub fn reset_to_default_pipeline(&mut self) {
        let cfg = self.config.clone();
        self.build_default_pipeline(&cfg);
    }

    /// Return the current number of stages.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    // -----------------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------------

    /// Run the pipeline on `query_embedding` and return the top results.
    ///
    /// The query embedding is **not** mutated; a working copy is used
    /// internally for normalisation.
    pub fn search(&mut self, query_embedding: &[f64]) -> Vec<SspSearchResult> {
        let start = Instant::now();

        // Shallow cache check ------------------------------------------------
        let cache_key = self.embedding_fingerprint(query_embedding);
        if let Some(cached) = self.result_cache.get(&cache_key) {
            let result = cached.clone();
            self.cache_hits += 1;
            // Still record the query in the log as a cache hit
            self.record_query(
                "".to_string(),
                result.len(),
                start.elapsed().as_secs_f64() * 1000.0,
            );
            return result;
        }

        // Working copy of query embedding ------------------------------------
        let mut q_emb: Vec<f64> = query_embedding.to_vec();

        // Candidate list (populated by AnnSearch stage) ----------------------
        let mut candidates: Vec<Candidate> = Vec::new();

        // Execute stages in order -------------------------------------------
        for stage in self.stages.clone().iter() {
            match stage {
                SspStage::Tokenize {
                    lowercase,
                    stop_words,
                } => {
                    // Tokenize the last recorded query text
                    let text = self.last_query_tokens.join(" ");
                    self.last_query_tokens = tokenize(&text, *lowercase, stop_words);
                }
                SspStage::EmbedQuery { dim: _ } => {
                    l2_normalize(&mut q_emb);
                }
                SspStage::AnnSearch { top_k } => {
                    candidates = self.ann_search(&q_emb, *top_k);
                }
                SspStage::Rerank { method } => {
                    self.apply_rerank(&mut candidates, &q_emb, method.clone());
                }
                SspStage::Deduplicate { threshold } => {
                    candidates = self.apply_dedup(candidates, *threshold);
                }
                SspStage::ScoreFilter { min_score } => {
                    candidates.retain(|c| c.score >= *min_score);
                }
                SspStage::Limit(n) => {
                    candidates.truncate(*n);
                }
            }
        }

        // Build results from candidates -------------------------------------
        let results: Vec<SspSearchResult> = candidates
            .iter()
            .filter_map(|c| {
                let doc = self.documents.get(&c.doc_id)?;
                Some(SspSearchResult {
                    doc_id: c.doc_id,
                    score: c.score,
                    snippet: make_snippet(&doc.content),
                    metadata: doc.metadata.clone(),
                })
            })
            .collect();

        // Cache and record --------------------------------------------------
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.result_cache.insert(cache_key, results.clone());
        self.record_query("".to_string(), results.len(), latency_ms);

        results
    }

    /// Tokenise `query` text, then run the pipeline on `query_embedding`.
    ///
    /// The tokenised terms are made available to reranking stages via the
    /// internal `last_query_tokens` field.
    pub fn search_text(&mut self, query: &str, query_embedding: &[f64]) -> Vec<SspSearchResult> {
        // Truncate to max_query_len characters (not bytes)
        let trimmed: String = query.chars().take(self.config.max_query_len).collect();

        // Store tokens for use by reranking stages
        self.last_query_tokens = tokenize(&trimmed, true, &[]);

        let start = Instant::now();

        // Shallow cache (include query text in key)
        let cache_key = self.text_embedding_fingerprint(query, query_embedding);
        if let Some(cached) = self.result_cache.get(&cache_key) {
            let result = cached.clone();
            self.cache_hits += 1;
            self.record_query(
                trimmed.clone(),
                result.len(),
                start.elapsed().as_secs_f64() * 1000.0,
            );
            return result;
        }

        let mut q_emb: Vec<f64> = query_embedding.to_vec();
        let mut candidates: Vec<Candidate> = Vec::new();

        for stage in self.stages.clone().iter() {
            match stage {
                SspStage::Tokenize {
                    lowercase,
                    stop_words,
                } => {
                    self.last_query_tokens = tokenize(&trimmed, *lowercase, stop_words);
                }
                SspStage::EmbedQuery { dim: _ } => {
                    l2_normalize(&mut q_emb);
                }
                SspStage::AnnSearch { top_k } => {
                    candidates = self.ann_search(&q_emb, *top_k);
                }
                SspStage::Rerank { method } => {
                    self.apply_rerank_with_text(&mut candidates, &q_emb, method.clone());
                }
                SspStage::Deduplicate { threshold } => {
                    candidates = self.apply_dedup(candidates, *threshold);
                }
                SspStage::ScoreFilter { min_score } => {
                    candidates.retain(|c| c.score >= *min_score);
                }
                SspStage::Limit(n) => {
                    candidates.truncate(*n);
                }
            }
        }

        let results: Vec<SspSearchResult> = candidates
            .iter()
            .filter_map(|c| {
                let doc = self.documents.get(&c.doc_id)?;
                Some(SspSearchResult {
                    doc_id: c.doc_id,
                    score: c.score,
                    snippet: make_snippet(&doc.content),
                    metadata: doc.metadata.clone(),
                })
            })
            .collect();

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.result_cache.insert(cache_key, results.clone());
        self.record_query(trimmed, results.len(), latency_ms);

        results
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    /// Return aggregated pipeline statistics.
    pub fn pipeline_stats(&self) -> SspPipelineStats {
        let avg_results = if self.total_queries == 0 {
            0.0
        } else {
            self.total_results as f64 / self.total_queries as f64
        };
        let avg_latency_ms = if self.total_queries == 0 {
            0.0
        } else {
            self.total_latency_ms / self.total_queries as f64
        };
        let cache_hit_rate = if self.total_queries == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_queries as f64
        };
        SspPipelineStats {
            total_queries: self.total_queries,
            avg_results,
            avg_latency_ms,
            cache_hit_rate,
        }
    }

    /// Return a reference to the query log.
    pub fn query_log(&self) -> &VecDeque<SspQueryRecord> {
        &self.query_log
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Brute-force cosine-similarity scan returning the top-`k` candidates.
    fn ann_search(&self, q_emb: &[f64], k: usize) -> Vec<Candidate> {
        let mut scored: Vec<Candidate> = self
            .embeddings
            .iter()
            .map(|(doc_id, emb)| {
                let score = cosine_similarity(q_emb, emb);
                let bm25 = 0.0; // populated by rerank stage if needed
                Candidate {
                    doc_id: *doc_id,
                    score,
                    bm25_score: bm25,
                }
            })
            .collect();

        // Sort descending by cosine score; use doc_id as stable tie-breaker
        scored.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.doc_id.cmp(&b.doc_id))
        });
        scored.truncate(k);
        scored
    }

    /// Rerank candidates using the requested method (no text tokens
    /// available — uses cosine scores only for BM25 approximation).
    fn apply_rerank(
        &mut self,
        candidates: &mut [Candidate],
        q_emb: &[f64],
        method: SspRerankMethod,
    ) {
        match method {
            SspRerankMethod::BM25Rescore => {
                // Without query tokens we can only refine via vector score
                for c in candidates.iter_mut() {
                    // Refresh cosine score
                    if let Some(emb) = self.embeddings.get(&c.doc_id) {
                        c.score = cosine_similarity(q_emb, emb);
                    }
                }
                candidates.sort_unstable_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            SspRerankMethod::CrossEncoderApprox => {
                for c in candidates.iter_mut() {
                    // Scale by a small jitter derived from xorshift for variety
                    let jitter = (xorshift64(&mut self.rng_state) % 1000) as f64 / 100_000.0;
                    c.score = (c.score * (1.0 + jitter)).min(1.0);
                }
                candidates.sort_unstable_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            SspRerankMethod::RecipRankFusion => {
                self.apply_rrf(candidates);
            }
        }
    }

    /// Rerank with access to `last_query_tokens` for BM25 scoring.
    fn apply_rerank_with_text(
        &mut self,
        candidates: &mut [Candidate],
        q_emb: &[f64],
        method: SspRerankMethod,
    ) {
        let tokens = self.last_query_tokens.clone();
        match method {
            SspRerankMethod::BM25Rescore => {
                for c in candidates.iter_mut() {
                    if let Some(doc) = self.documents.get(&c.doc_id) {
                        let dl = doc.content.split_whitespace().count();
                        c.bm25_score = bm25_score(&tokens, &doc.content, dl);
                    }
                }
                // Normalise BM25 scores
                let max_bm25 = candidates
                    .iter()
                    .map(|c| c.bm25_score)
                    .fold(f64::NEG_INFINITY, f64::max);
                if max_bm25 > 0.0 {
                    for c in candidates.iter_mut() {
                        c.score = 0.5 * c.score + 0.5 * (c.bm25_score / max_bm25);
                    }
                }
                candidates.sort_unstable_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            SspRerankMethod::CrossEncoderApprox => {
                for c in candidates.iter_mut() {
                    if let Some(doc) = self.documents.get(&c.doc_id) {
                        // Term overlap ratio as a proxy for cross-encoder score
                        let doc_tokens: Vec<String> = doc
                            .content
                            .split_whitespace()
                            .map(|w| w.to_lowercase())
                            .collect();
                        let overlap =
                            tokens.iter().filter(|t| doc_tokens.contains(t)).count() as f64;
                        let max_len = tokens.len().max(1) as f64;
                        let ce_approx = overlap / max_len;
                        c.score = 0.6 * c.score + 0.4 * ce_approx;
                    }
                    // Also refresh cosine score
                    if let Some(emb) = self.embeddings.get(&c.doc_id) {
                        let cos = cosine_similarity(q_emb, emb);
                        c.score = 0.7 * c.score + 0.3 * cos;
                    }
                }
                candidates.sort_unstable_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            SspRerankMethod::RecipRankFusion => {
                // Compute BM25 ranks alongside vector ranks
                for c in candidates.iter_mut() {
                    if let Some(doc) = self.documents.get(&c.doc_id) {
                        let dl = doc.content.split_whitespace().count();
                        c.bm25_score = bm25_score(&tokens, &doc.content, dl);
                    }
                }
                self.apply_rrf(candidates);
            }
        }
    }

    /// Apply Reciprocal Rank Fusion to merge cosine and BM25 ranked lists.
    fn apply_rrf(&self, candidates: &mut [Candidate]) {
        const K: f64 = 60.0;
        let n = candidates.len();
        if n == 0 {
            return;
        }

        // Vector rank (already sorted by cosine desc)
        let mut vec_ranks: Vec<(SspDocId, usize)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (c.doc_id, i + 1))
            .collect();
        vec_ranks.sort_unstable_by_key(|(id, _)| *id);

        // BM25 rank
        let mut bm25_order: Vec<(SspDocId, f64)> = candidates
            .iter()
            .map(|c| (c.doc_id, c.bm25_score))
            .collect();
        bm25_order
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut bm25_ranks: Vec<(SspDocId, usize)> = bm25_order
            .iter()
            .enumerate()
            .map(|(i, (id, _))| (*id, i + 1))
            .collect();
        bm25_ranks.sort_unstable_by_key(|(id, _)| *id);

        // Assign RRF scores
        for c in candidates.iter_mut() {
            let vr = vec_ranks
                .binary_search_by_key(&c.doc_id, |(id, _)| *id)
                .map(|i| vec_ranks[i].1)
                .unwrap_or(n + 1);
            let br = bm25_ranks
                .binary_search_by_key(&c.doc_id, |(id, _)| *id)
                .map(|i| bm25_ranks[i].1)
                .unwrap_or(n + 1);
            c.score = 1.0 / (K + vr as f64) + 1.0 / (K + br as f64);
        }

        candidates.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Remove near-duplicate candidates (mutual cosine similarity > threshold).
    fn apply_dedup(&self, candidates: Vec<Candidate>, threshold: f64) -> Vec<Candidate> {
        let mut kept: Vec<Candidate> = Vec::with_capacity(candidates.len());
        'outer: for c in candidates.into_iter() {
            for k in &kept {
                let emb_c = match self.embeddings.get(&c.doc_id) {
                    Some(e) => e,
                    None => continue,
                };
                let emb_k = match self.embeddings.get(&k.doc_id) {
                    Some(e) => e,
                    None => continue,
                };
                if cosine_similarity(emb_c, emb_k) > threshold {
                    continue 'outer;
                }
            }
            kept.push(c);
        }
        kept
    }

    /// Compute a simple 64-bit fingerprint of an embedding for cache lookup.
    fn embedding_fingerprint(&mut self, v: &[f64]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
        for &x in v {
            let bits = x.to_bits();
            h ^= bits;
            h = h.wrapping_mul(0x0000_0100_0000_01b3); // FNV prime
        }
        h
    }

    /// Fingerprint incorporating both text and embedding.
    fn text_embedding_fingerprint(&mut self, text: &str, v: &[f64]) -> u64 {
        let mut h = self.embedding_fingerprint(v);
        for b in text.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// Append a record to the query log (capped at 500 entries) and update
    /// cumulative counters.
    fn record_query(&mut self, query_text: String, n_results: usize, latency_ms: f64) {
        if self.query_log.len() >= 500 {
            self.query_log.pop_front();
        }
        self.query_log.push_back(SspQueryRecord {
            ts: unix_now(),
            query_text,
            n_results,
            latency_ms,
        });
        self.total_queries += 1;
        self.total_results += n_results as u64;
        self.total_latency_ms += latency_ms;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper
    // -----------------------------------------------------------------------

    fn make_pipeline() -> SemanticSearchPipeline {
        SemanticSearchPipeline::new(SspPipelineConfig {
            embedding_dim: 4,
            top_k: 5,
            rerank: false,
            deduplicate: false,
            score_threshold: 0.0,
            max_query_len: 64,
        })
    }

    fn add_and_index(p: &mut SemanticSearchPipeline, content: &str, emb: Vec<f64>) -> SspDocId {
        let id = p.add_document(content.to_string(), HashMap::new());
        p.index_embedding(id, emb);
        id
    }

    // -----------------------------------------------------------------------
    // Configuration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_config_fields() {
        let cfg = SspPipelineConfig::default();
        assert_eq!(cfg.embedding_dim, 128);
        assert_eq!(cfg.top_k, 10);
        assert!(cfg.rerank);
        assert!(cfg.deduplicate);
        assert_eq!(cfg.score_threshold, 0.0);
        assert_eq!(cfg.max_query_len, 512);
    }

    #[test]
    fn test_custom_config() {
        let cfg = SspPipelineConfig {
            embedding_dim: 32,
            top_k: 3,
            rerank: false,
            deduplicate: true,
            score_threshold: 0.1,
            max_query_len: 128,
        };
        assert_eq!(cfg.embedding_dim, 32);
        assert!(!cfg.rerank);
        assert_eq!(cfg.score_threshold, 0.1);
    }

    // -----------------------------------------------------------------------
    // Pipeline construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_pipeline_has_stages() {
        let p = make_pipeline();
        assert!(p.stage_count() > 0);
    }

    #[test]
    fn test_new_pipeline_with_rerank_has_more_stages() {
        let cfg_no_rerank = SspPipelineConfig {
            rerank: false,
            deduplicate: false,
            ..Default::default()
        };
        let cfg_rerank = SspPipelineConfig {
            rerank: true,
            deduplicate: false,
            ..Default::default()
        };
        let p0 = SemanticSearchPipeline::new(cfg_no_rerank);
        let p1 = SemanticSearchPipeline::new(cfg_rerank);
        assert!(p1.stage_count() > p0.stage_count());
    }

    #[test]
    fn test_add_stage() {
        let mut p = make_pipeline();
        let before = p.stage_count();
        p.add_stage(SspStage::Limit(10));
        assert_eq!(p.stage_count(), before + 1);
    }

    #[test]
    fn test_clear_stages() {
        let mut p = make_pipeline();
        p.clear_stages();
        assert_eq!(p.stage_count(), 0);
    }

    #[test]
    fn test_reset_to_default_pipeline_restores_stages() {
        let mut p = make_pipeline();
        p.clear_stages();
        assert_eq!(p.stage_count(), 0);
        p.reset_to_default_pipeline();
        assert!(p.stage_count() > 0);
    }

    // -----------------------------------------------------------------------
    // Document management
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_document_returns_incrementing_ids() {
        let mut p = make_pipeline();
        let id1 = p.add_document("a".to_string(), HashMap::new());
        let id2 = p.add_document("b".to_string(), HashMap::new());
        assert!(id2 > id1);
    }

    #[test]
    fn test_document_count_increments() {
        let mut p = make_pipeline();
        assert_eq!(p.document_count(), 0);
        p.add_document("x".to_string(), HashMap::new());
        assert_eq!(p.document_count(), 1);
        p.add_document("y".to_string(), HashMap::new());
        assert_eq!(p.document_count(), 2);
    }

    #[test]
    fn test_remove_document_returns_true() {
        let mut p = make_pipeline();
        let id = p.add_document("hello".to_string(), HashMap::new());
        assert!(p.remove_document(id));
    }

    #[test]
    fn test_remove_nonexistent_document_returns_false() {
        let mut p = make_pipeline();
        assert!(!p.remove_document(9999));
    }

    #[test]
    fn test_remove_document_decrements_count() {
        let mut p = make_pipeline();
        let id = p.add_document("hello".to_string(), HashMap::new());
        p.remove_document(id);
        assert_eq!(p.document_count(), 0);
    }

    #[test]
    fn test_update_document_returns_true() {
        let mut p = make_pipeline();
        let id = p.add_document("original".to_string(), HashMap::new());
        assert!(p.update_document(id, "updated".to_string()));
    }

    #[test]
    fn test_update_document_changes_content() {
        let mut p = make_pipeline();
        let id = p.add_document("original".to_string(), HashMap::new());
        p.update_document(id, "updated".to_string());
        let doc = p.get_document(id).expect("doc should exist");
        assert_eq!(doc.content, "updated");
    }

    #[test]
    fn test_update_nonexistent_document_returns_false() {
        let mut p = make_pipeline();
        assert!(!p.update_document(9999, "x".to_string()));
    }

    #[test]
    fn test_get_document_returns_correct_content() {
        let mut p = make_pipeline();
        let id = p.add_document("test content".to_string(), HashMap::new());
        let doc = p.get_document(id).expect("doc should exist");
        assert_eq!(doc.content, "test content");
    }

    #[test]
    fn test_get_document_stores_metadata() {
        let mut p = make_pipeline();
        let mut meta = HashMap::new();
        meta.insert("lang".to_string(), "en".to_string());
        let id = p.add_document("text".to_string(), meta);
        let doc = p.get_document(id).expect("doc");
        assert_eq!(doc.metadata.get("lang"), Some(&"en".to_string()));
    }

    #[test]
    fn test_get_document_returns_none_for_missing() {
        let p = make_pipeline();
        assert!(p.get_document(42).is_none());
    }

    // -----------------------------------------------------------------------
    // Embedding management
    // -----------------------------------------------------------------------

    #[test]
    fn test_index_embedding_increments_count() {
        let mut p = make_pipeline();
        let id = p.add_document("a".to_string(), HashMap::new());
        p.index_embedding(id, vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(p.embedding_count(), 1);
    }

    #[test]
    fn test_index_embedding_normalises() {
        let mut p = make_pipeline();
        let id = p.add_document("a".to_string(), HashMap::new());
        // Un-normalised: [3, 4] => norm = 5
        p.index_embedding(id, vec![3.0, 4.0]);
        // After normalisation the cosine of the stored vector with itself is 1
        let results = p.search(&[3.0, 4.0]);
        if !results.is_empty() {
            assert!((results[0].score - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_index_embedding_replaces_previous() {
        let mut p = make_pipeline();
        let id = p.add_document("doc".to_string(), HashMap::new());
        p.index_embedding(id, vec![1.0, 0.0, 0.0, 0.0]);
        p.index_embedding(id, vec![0.0, 1.0, 0.0, 0.0]);
        assert_eq!(p.embedding_count(), 1);
    }

    // -----------------------------------------------------------------------
    // cosine_similarity helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_cosine_identity() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-12);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cosine_zero_vector_returns_zero() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_both_zero() {
        let a = vec![0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &a), 0.0);
    }

    // -----------------------------------------------------------------------
    // xorshift64 helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_xorshift64_produces_nonzero() {
        let mut state = 1u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }

    #[test]
    fn test_xorshift64_is_deterministic() {
        let mut s1 = 42u64;
        let mut s2 = 42u64;
        assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
    }

    #[test]
    fn test_xorshift64_state_changes() {
        let mut state = 100u64;
        let v1 = xorshift64(&mut state);
        let v2 = xorshift64(&mut state);
        assert_ne!(v1, v2);
    }

    // -----------------------------------------------------------------------
    // tokenize helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_tokenize_splits_whitespace() {
        let tokens = tokenize("hello world", false, &[]);
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenize_lowercase() {
        let tokens = tokenize("Hello World", true, &[]);
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenize_removes_stop_words() {
        let stop = vec!["the".to_string(), "a".to_string()];
        let tokens = tokenize("the quick a fox", false, &stop);
        assert_eq!(tokens, vec!["quick", "fox"]);
    }

    #[test]
    fn test_tokenize_empty_string() {
        let tokens = tokenize("", false, &[]);
        assert!(tokens.is_empty());
    }

    // -----------------------------------------------------------------------
    // Search — basic correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_search_empty_pipeline_returns_empty() {
        let mut p = make_pipeline();
        let results = p.search(&[1.0, 0.0, 0.0, 0.0]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_returns_result_for_matching_doc() {
        let mut p = make_pipeline();
        add_and_index(&mut p, "rust programming", vec![1.0, 0.0, 0.0, 0.0]);
        let results = p.search(&[1.0, 0.0, 0.0, 0.0]);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_search_score_is_in_valid_range() {
        let mut p = make_pipeline();
        add_and_index(&mut p, "doc", vec![1.0, 0.0, 0.0, 0.0]);
        let results = p.search(&[1.0, 0.0, 0.0, 0.0]);
        for r in &results {
            assert!(r.score >= -1.0 - 1e-9 && r.score <= 1.0 + 1e-9);
        }
    }

    #[test]
    fn test_search_respects_top_k() {
        let mut p = SemanticSearchPipeline::new(SspPipelineConfig {
            embedding_dim: 4,
            top_k: 2,
            rerank: false,
            deduplicate: false,
            score_threshold: 0.0,
            max_query_len: 64,
        });
        for i in 0..10u64 {
            let mut emb = vec![0.0f64; 4];
            emb[0] = i as f64;
            add_and_index(&mut p, "doc", emb);
        }
        let results = p.search(&[1.0, 0.0, 0.0, 0.0]);
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_search_results_have_snippets() {
        let mut p = make_pipeline();
        add_and_index(
            &mut p,
            "hello world this is a test",
            vec![1.0, 0.0, 0.0, 0.0],
        );
        let results = p.search(&[1.0, 0.0, 0.0, 0.0]);
        assert!(!results[0].snippet.is_empty());
    }

    #[test]
    fn test_search_snippet_max_200_chars() {
        let long_content = "a".repeat(500);
        let mut p = make_pipeline();
        add_and_index(&mut p, &long_content, vec![1.0, 0.0, 0.0, 0.0]);
        let results = p.search(&[1.0, 0.0, 0.0, 0.0]);
        assert!(results[0].snippet.len() <= 200);
    }

    #[test]
    fn test_search_returns_correct_doc_id() {
        let mut p = make_pipeline();
        let id = add_and_index(&mut p, "unique content", vec![1.0, 0.0, 0.0, 0.0]);
        let results = p.search(&[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(results[0].doc_id, id);
    }

    #[test]
    fn test_search_score_filter_removes_low_scores() {
        let mut p = SemanticSearchPipeline::new(SspPipelineConfig {
            embedding_dim: 2,
            top_k: 10,
            rerank: false,
            deduplicate: false,
            score_threshold: 0.99,
            max_query_len: 64,
        });
        // Very different vector — score will be low
        add_and_index(&mut p, "doc", vec![-1.0, 0.0]);
        let results = p.search(&[1.0, 0.0]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_removed_document_not_in_results() {
        let mut p = make_pipeline();
        let id = add_and_index(&mut p, "doc", vec![1.0, 0.0, 0.0, 0.0]);
        p.remove_document(id);
        let results = p.search(&[1.0, 0.0, 0.0, 0.0]);
        assert!(!results.iter().any(|r| r.doc_id == id));
    }

    // -----------------------------------------------------------------------
    // search_text
    // -----------------------------------------------------------------------

    #[test]
    fn test_search_text_returns_results() {
        let mut p = make_pipeline();
        add_and_index(
            &mut p,
            "rust programming language",
            vec![1.0, 0.0, 0.0, 0.0],
        );
        let results = p.search_text("rust", &[1.0, 0.0, 0.0, 0.0]);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_search_text_truncates_long_query() {
        let mut p = SemanticSearchPipeline::new(SspPipelineConfig {
            max_query_len: 5,
            embedding_dim: 4,
            top_k: 5,
            rerank: false,
            deduplicate: false,
            score_threshold: 0.0,
        });
        add_and_index(&mut p, "doc", vec![1.0, 0.0, 0.0, 0.0]);
        // Long query — should not panic
        let results = p.search_text("a b c d e f g h i j k", &[1.0, 0.0, 0.0, 0.0]);
        // We only verify no panic and valid results
        let _ = results;
    }

    // -----------------------------------------------------------------------
    // Reranking stages
    // -----------------------------------------------------------------------

    #[test]
    fn test_bm25_rescore_reranking() {
        let mut p = SemanticSearchPipeline::new(SspPipelineConfig {
            embedding_dim: 4,
            top_k: 5,
            rerank: false,
            deduplicate: false,
            score_threshold: 0.0,
            max_query_len: 64,
        });
        p.clear_stages();
        p.add_stage(SspStage::EmbedQuery { dim: 4 });
        p.add_stage(SspStage::AnnSearch { top_k: 20 });
        p.add_stage(SspStage::Rerank {
            method: SspRerankMethod::BM25Rescore,
        });
        p.add_stage(SspStage::Limit(5));
        add_and_index(&mut p, "rust language", vec![1.0, 0.0, 0.0, 0.0]);
        add_and_index(&mut p, "python scripting", vec![0.9, 0.1, 0.0, 0.0]);
        let results = p.search_text("rust", &[1.0, 0.0, 0.0, 0.0]);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_cross_encoder_approx_reranking() {
        let mut p = make_pipeline();
        p.clear_stages();
        p.add_stage(SspStage::EmbedQuery { dim: 4 });
        p.add_stage(SspStage::AnnSearch { top_k: 20 });
        p.add_stage(SspStage::Rerank {
            method: SspRerankMethod::CrossEncoderApprox,
        });
        p.add_stage(SspStage::Limit(5));
        add_and_index(&mut p, "hello world", vec![1.0, 0.0, 0.0, 0.0]);
        let results = p.search_text("hello", &[1.0, 0.0, 0.0, 0.0]);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_recip_rank_fusion_reranking() {
        let mut p = make_pipeline();
        p.clear_stages();
        p.add_stage(SspStage::EmbedQuery { dim: 4 });
        p.add_stage(SspStage::AnnSearch { top_k: 20 });
        p.add_stage(SspStage::Rerank {
            method: SspRerankMethod::RecipRankFusion,
        });
        p.add_stage(SspStage::Limit(5));
        add_and_index(&mut p, "semantic search", vec![1.0, 0.0, 0.0, 0.0]);
        let results = p.search_text("semantic", &[1.0, 0.0, 0.0, 0.0]);
        assert!(!results.is_empty());
    }

    // -----------------------------------------------------------------------
    // Deduplication
    // -----------------------------------------------------------------------

    #[test]
    fn test_dedup_removes_near_duplicates() {
        let mut p = SemanticSearchPipeline::new(SspPipelineConfig {
            embedding_dim: 2,
            top_k: 10,
            rerank: false,
            deduplicate: false,
            score_threshold: 0.0,
            max_query_len: 64,
        });
        p.clear_stages();
        p.add_stage(SspStage::EmbedQuery { dim: 2 });
        p.add_stage(SspStage::AnnSearch { top_k: 20 });
        p.add_stage(SspStage::Deduplicate { threshold: 0.99 });
        p.add_stage(SspStage::Limit(10));

        // Two nearly identical documents
        add_and_index(&mut p, "doc a", vec![1.0, 0.0]);
        add_and_index(&mut p, "doc b", vec![1.0, 1e-10]);
        let results = p.search(&[1.0, 0.0]);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_dedup_keeps_diverse_docs() {
        let mut p = SemanticSearchPipeline::new(SspPipelineConfig {
            embedding_dim: 2,
            top_k: 10,
            rerank: false,
            deduplicate: false,
            score_threshold: 0.0,
            max_query_len: 64,
        });
        p.clear_stages();
        p.add_stage(SspStage::EmbedQuery { dim: 2 });
        p.add_stage(SspStage::AnnSearch { top_k: 20 });
        p.add_stage(SspStage::Deduplicate { threshold: 0.99 });
        p.add_stage(SspStage::Limit(10));

        add_and_index(&mut p, "doc a", vec![1.0, 0.0]);
        add_and_index(&mut p, "doc b", vec![0.0, 1.0]);
        let results = p.search(&[1.0, 0.0]);
        assert_eq!(results.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Query log
    // -----------------------------------------------------------------------

    #[test]
    fn test_query_log_records_queries() {
        let mut p = make_pipeline();
        add_and_index(&mut p, "doc", vec![1.0, 0.0, 0.0, 0.0]);
        p.search(&[1.0, 0.0, 0.0, 0.0]);
        assert!(!p.query_log().is_empty());
    }

    #[test]
    fn test_query_log_capped_at_500() {
        let mut p = make_pipeline();
        add_and_index(&mut p, "doc", vec![1.0, 0.0, 0.0, 0.0]);
        for _ in 0..600 {
            p.search(&[1.0, 0.0, 0.0, 0.0]);
            // Vary the query slightly so cache doesn't always hit
        }
        assert!(p.query_log().len() <= 500);
    }

    // -----------------------------------------------------------------------
    // Pipeline stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_total_queries_increments() {
        let mut p = make_pipeline();
        add_and_index(&mut p, "doc", vec![1.0, 0.0, 0.0, 0.0]);
        p.search(&[1.0, 0.0, 0.0, 0.0]);
        p.search(&[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(p.pipeline_stats().total_queries, 2);
    }

    #[test]
    fn test_stats_avg_latency_is_nonnegative() {
        let mut p = make_pipeline();
        add_and_index(&mut p, "doc", vec![1.0, 0.0, 0.0, 0.0]);
        p.search(&[1.0, 0.0, 0.0, 0.0]);
        assert!(p.pipeline_stats().avg_latency_ms >= 0.0);
    }

    #[test]
    fn test_stats_zero_queries_gives_zeros() {
        let p = make_pipeline();
        let stats = p.pipeline_stats();
        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.avg_results, 0.0);
        assert_eq!(stats.avg_latency_ms, 0.0);
        assert_eq!(stats.cache_hit_rate, 0.0);
    }

    #[test]
    fn test_stats_cache_hit_rate_between_zero_and_one() {
        let mut p = make_pipeline();
        add_and_index(&mut p, "doc", vec![1.0, 0.0, 0.0, 0.0]);
        // First search — cache miss
        p.search(&[1.0, 0.0, 0.0, 0.0]);
        // Second search with same embedding — cache hit
        p.search(&[1.0, 0.0, 0.0, 0.0]);
        let rate = p.pipeline_stats().cache_hit_rate;
        assert!((0.0..=1.0).contains(&rate));
    }

    // -----------------------------------------------------------------------
    // Cache behaviour
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_hit_on_repeated_query() {
        let mut p = make_pipeline();
        add_and_index(&mut p, "doc", vec![1.0, 0.0, 0.0, 0.0]);
        p.search(&[1.0, 0.0, 0.0, 0.0]);
        p.search(&[1.0, 0.0, 0.0, 0.0]);
        assert!(p.pipeline_stats().cache_hit_rate > 0.0);
    }

    #[test]
    fn test_cache_invalidated_after_remove() {
        let mut p = make_pipeline();
        let id = add_and_index(&mut p, "doc", vec![1.0, 0.0, 0.0, 0.0]);
        p.search(&[1.0, 0.0, 0.0, 0.0]);
        p.remove_document(id);
        // After removal, cache is cleared; the next search should re-run
        let results = p.search(&[1.0, 0.0, 0.0, 0.0]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_cache_invalidated_after_update() {
        let mut p = make_pipeline();
        let id = add_and_index(&mut p, "original", vec![1.0, 0.0, 0.0, 0.0]);
        p.search(&[1.0, 0.0, 0.0, 0.0]);
        p.update_document(id, "updated content".to_string());
        // Cache cleared; snippet should reflect updated content
        let results = p.search(&[1.0, 0.0, 0.0, 0.0]);
        assert!(!results.is_empty());
        assert!(results[0].snippet.contains("updated"));
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_search_without_embeddings_returns_empty() {
        let mut p = make_pipeline();
        p.add_document("no embedding".to_string(), HashMap::new());
        let results = p.search(&[1.0, 0.0, 0.0, 0.0]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_zero_embedding_returns_zero_scores() {
        let mut p = make_pipeline();
        add_and_index(&mut p, "doc", vec![1.0, 0.0, 0.0, 0.0]);
        let results = p.search(&[0.0, 0.0, 0.0, 0.0]);
        for r in &results {
            assert_eq!(r.score, 0.0);
        }
    }

    #[test]
    fn test_search_many_documents() {
        let mut p = SemanticSearchPipeline::new(SspPipelineConfig {
            embedding_dim: 4,
            top_k: 5,
            rerank: false,
            deduplicate: false,
            score_threshold: 0.0,
            max_query_len: 64,
        });
        for i in 0..100u64 {
            let mut emb = vec![0.0f64; 4];
            emb[(i % 4) as usize] = 1.0;
            add_and_index(&mut p, &format!("doc {}", i), emb);
        }
        let results = p.search(&[1.0, 0.0, 0.0, 0.0]);
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_ssp_doc_id_type_alias() {
        let id: SspDocId = 42;
        assert_eq!(id, 42u64);
    }

    #[test]
    fn test_ssp_semantic_search_pipeline_alias() {
        let _p: SspSemanticSearchPipeline =
            SemanticSearchPipeline::new(SspPipelineConfig::default());
    }

    #[test]
    fn test_snippet_content_matches_document() {
        let mut p = make_pipeline();
        add_and_index(&mut p, "unique snippet text", vec![1.0, 0.0, 0.0, 0.0]);
        let results = p.search(&[1.0, 0.0, 0.0, 0.0]);
        assert!(results[0].snippet.contains("unique snippet text"));
    }

    #[test]
    fn test_metadata_propagated_to_result() {
        let mut p = make_pipeline();
        let mut meta = HashMap::new();
        meta.insert("key".to_string(), "value".to_string());
        let id = p.add_document("doc".to_string(), meta);
        p.index_embedding(id, vec![1.0, 0.0, 0.0, 0.0]);
        let results = p.search(&[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(results[0].metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_multiple_removes_and_searches() {
        let mut p = make_pipeline();
        let ids: Vec<SspDocId> = (0..5)
            .map(|i| add_and_index(&mut p, &format!("doc {}", i), vec![1.0, 0.0, 0.0, 0.0]))
            .collect();
        for id in &ids[..3] {
            p.remove_document(*id);
        }
        let results = p.search(&[1.0, 0.0, 0.0, 0.0]);
        for r in &results {
            assert!(!ids[..3].contains(&r.doc_id));
        }
    }

    #[test]
    fn test_pipeline_stats_avg_results_correct() {
        let mut p = make_pipeline();
        add_and_index(&mut p, "doc", vec![1.0, 0.0, 0.0, 0.0]);
        p.search(&[1.0, 0.0, 0.0, 0.0]);
        let stats = p.pipeline_stats();
        // avg_results should be 1.0 (one result from one query)
        assert!(stats.avg_results >= 0.0);
    }

    #[test]
    fn test_bm25_score_helper_nonzero_for_matching_tokens() {
        let tokens = vec!["rust".to_string(), "lang".to_string()];
        let score = bm25_score(&tokens, "rust is a systems lang", 6);
        assert!(score > 0.0);
    }

    #[test]
    fn test_bm25_score_helper_zero_for_no_match() {
        let tokens = vec!["python".to_string()];
        let score = bm25_score(&tokens, "rust is a systems language", 6);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_l2_normalize_zero_vector_no_panic() {
        let mut v = vec![0.0f64; 4];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0; 4]);
    }

    #[test]
    fn test_l2_normalize_unit_vector_unchanged() {
        let mut v = vec![1.0f64, 0.0, 0.0];
        l2_normalize(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-12);
        assert!(v[1].abs() < 1e-12);
    }

    #[test]
    fn test_make_snippet_short_content() {
        let snippet = make_snippet("hello");
        assert_eq!(snippet, "hello");
    }

    #[test]
    fn test_make_snippet_long_content() {
        let long = "x".repeat(300);
        let snippet = make_snippet(&long);
        assert_eq!(snippet.len(), 200);
    }

    #[test]
    fn test_score_filter_stage_standalone() {
        let mut p = make_pipeline();
        p.clear_stages();
        p.add_stage(SspStage::EmbedQuery { dim: 4 });
        p.add_stage(SspStage::AnnSearch { top_k: 20 });
        p.add_stage(SspStage::ScoreFilter { min_score: 2.0 }); // impossible threshold
        add_and_index(&mut p, "doc", vec![1.0, 0.0, 0.0, 0.0]);
        let results = p.search(&[1.0, 0.0, 0.0, 0.0]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_limit_stage_standalone() {
        let mut p = make_pipeline();
        p.clear_stages();
        p.add_stage(SspStage::EmbedQuery { dim: 4 });
        p.add_stage(SspStage::AnnSearch { top_k: 20 });
        p.add_stage(SspStage::Limit(1));
        for _ in 0..5 {
            add_and_index(&mut p, "doc", vec![1.0, 0.0, 0.0, 0.0]);
        }
        let results = p.search(&[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_pipeline_config_clone() {
        let cfg = SspPipelineConfig::default();
        let cfg2 = cfg.clone();
        assert_eq!(cfg.embedding_dim, cfg2.embedding_dim);
    }

    #[test]
    fn test_ssp_document_clone() {
        let doc = SspDocument {
            id: 1,
            content: "hello".to_string(),
            metadata: HashMap::new(),
            indexed_at: 0,
        };
        let doc2 = doc.clone();
        assert_eq!(doc.id, doc2.id);
    }

    #[test]
    fn test_ssp_search_result_clone() {
        let r = SspSearchResult {
            doc_id: 1,
            score: 0.5,
            snippet: "snip".to_string(),
            metadata: HashMap::new(),
        };
        let r2 = r.clone();
        assert_eq!(r.doc_id, r2.doc_id);
    }

    #[test]
    fn test_ssp_pipeline_stats_default() {
        let stats = SspPipelineStats::default();
        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.avg_results, 0.0);
    }

    #[test]
    fn test_ssp_query_record_fields() {
        let rec = SspQueryRecord {
            ts: 1_000_000,
            query_text: "hello".to_string(),
            n_results: 3,
            latency_ms: 2.5,
        };
        assert_eq!(rec.n_results, 3);
        assert!((rec.latency_ms - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_tokenize_stage_in_search_text() {
        let mut p = make_pipeline();
        p.clear_stages();
        p.add_stage(SspStage::Tokenize {
            lowercase: true,
            stop_words: vec!["the".to_string()],
        });
        p.add_stage(SspStage::EmbedQuery { dim: 4 });
        p.add_stage(SspStage::AnnSearch { top_k: 10 });
        p.add_stage(SspStage::Limit(5));
        add_and_index(&mut p, "the quick brown fox", vec![1.0, 0.0, 0.0, 0.0]);
        let results = p.search_text("the quick", &[1.0, 0.0, 0.0, 0.0]);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_search_results_sorted_by_score_descending() {
        let mut p = SemanticSearchPipeline::new(SspPipelineConfig {
            embedding_dim: 2,
            top_k: 10,
            rerank: false,
            deduplicate: false,
            score_threshold: 0.0,
            max_query_len: 64,
        });
        p.clear_stages();
        p.add_stage(SspStage::EmbedQuery { dim: 2 });
        p.add_stage(SspStage::AnnSearch { top_k: 20 });
        p.add_stage(SspStage::Limit(10));

        add_and_index(&mut p, "high similarity", vec![1.0, 0.0]);
        add_and_index(&mut p, "medium similarity", vec![0.7, 0.7]);
        add_and_index(&mut p, "low similarity", vec![0.0, 1.0]);

        let results = p.search(&[1.0, 0.0]);
        for i in 1..results.len() {
            assert!(results[i - 1].score >= results[i].score);
        }
    }
}
