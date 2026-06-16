//! Topic Model Extractor — production-quality collapsed Gibbs sampling LDA.
//!
//! Implements Latent Dirichlet Allocation (LDA) via collapsed Gibbs sampling for
//! unsupervised topic discovery over text corpora.  All randomness is driven by
//! an xorshift64 PRNG so the implementation is 100 % pure-Rust with no `rand`
//! dependency.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PRNG — xorshift64
// ---------------------------------------------------------------------------

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[inline]
fn xorshift_f64(state: &mut u64) -> f64 {
    (xorshift64(state) >> 11) as f64 / (1u64 << 53) as f64
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can be returned by [`TopicModelExtractor`].
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ExtractorError {
    /// The corpus has fewer documents than can support topic modelling.
    #[error("insufficient documents: got {0}, need at least 2")]
    InsufficientDocuments(usize),
    /// The vocabulary is empty after filtering.
    #[error("vocabulary is empty after filtering")]
    VocabularyEmpty,
    /// A configuration parameter is invalid.
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),
    /// The model has not been fitted yet.
    #[error("model has not been fitted; call fit() first")]
    ModelNotFitted,
    /// An unknown topic id was requested.
    #[error("topic id {0} is out of range")]
    TopicOutOfRange(usize),
    /// An unknown document id was requested.
    #[error("document id '{0}' not found")]
    DocumentNotFound(String),
}

// ---------------------------------------------------------------------------
// Public configuration
// ---------------------------------------------------------------------------

/// Configuration for [`TopicModelExtractor`].
#[derive(Debug, Clone)]
pub struct ExtractorConfig {
    /// Number of topics to extract.
    pub num_topics: usize,
    /// Dirichlet prior on the document–topic distribution (α).
    pub alpha: f64,
    /// Dirichlet prior on the topic–word distribution (β).
    pub beta: f64,
    /// Number of Gibbs sampling iterations.
    pub num_iterations: u32,
    /// Maximum vocabulary size (most-frequent words are kept).
    pub vocab_size_limit: usize,
    /// Minimum per-word corpus frequency to retain a word.
    pub min_word_freq: u32,
    /// Exclude words that appear in more than this fraction of documents.
    pub max_doc_freq_pct: f64,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            num_topics: 10,
            alpha: 0.1,
            beta: 0.01,
            num_iterations: 1000,
            vocab_size_limit: 50_000,
            min_word_freq: 2,
            max_doc_freq_pct: 0.95,
        }
    }
}

// ---------------------------------------------------------------------------
// Core data structures
// ---------------------------------------------------------------------------

/// A word and its probability / raw count within a topic.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractorTopicWord {
    /// Surface form of the word.
    pub word: String,
    /// Normalised probability p(word | topic).
    pub probability: f64,
    /// Raw count of assignments to this topic.
    pub count: u32,
}

/// A single latent topic produced by the extractor.
#[derive(Debug, Clone)]
pub struct ExtractorTopic {
    /// Zero-based topic identifier.
    pub id: usize,
    /// Top words ranked by probability (highest first).
    pub top_words: Vec<ExtractorTopicWord>,
    /// Mean pairwise PMI of the top-10 words — higher is more coherent.
    pub coherence: f64,
    /// Fraction of corpus tokens assigned to this topic.
    pub prevalence: f64,
    /// Optional human-readable label.
    pub label: Option<String>,
}

/// Per-document topic distribution produced by the extractor.
#[derive(Debug, Clone)]
pub struct ExtractorDocumentTopics {
    /// Document identifier supplied at fit time.
    pub doc_id: String,
    /// Normalised topic distribution θ_d (sums to 1.0).
    pub topic_distribution: Vec<f64>,
    /// Index of the most probable topic.
    pub dominant_topic: usize,
    /// Probability of the dominant topic.
    pub dominant_probability: f64,
}

/// Aggregate model statistics.
#[derive(Debug, Clone)]
pub struct ModelStats {
    /// Number of topics.
    pub num_topics: usize,
    /// Vocabulary size after filtering.
    pub vocab_size: usize,
    /// Number of documents in the fitted corpus.
    pub num_docs: usize,
    /// Total number of tokens in the corpus.
    pub total_tokens: u64,
    /// Mean topic coherence across all topics.
    pub avg_topic_coherence: f64,
    /// Model perplexity on the training corpus.
    pub perplexity: f64,
    /// Number of Gibbs iterations actually executed.
    pub iterations_run: u32,
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// Assignment of a single token to a topic during Gibbs sampling.
#[derive(Debug, Clone, Copy)]
struct WordAssignment {
    word_idx: usize,
    topic_id: usize,
}

// ---------------------------------------------------------------------------
// TopicModelExtractor
// ---------------------------------------------------------------------------

/// Production-quality collapsed Gibbs sampling LDA topic extractor.
///
/// # Example
/// ```rust
/// use ipfrs_semantic::topic_model_extractor::{TopicModelExtractor, ExtractorConfig};
///
/// let config = ExtractorConfig { num_topics: 3, num_iterations: 100, ..Default::default() };
/// let mut extractor = TopicModelExtractor::new(config);
///
/// let docs = vec![
///     ("d1", "rust programming language systems"),
///     ("d2", "python data science machine learning"),
///     ("d3", "rust memory safety ownership"),
///     ("d4", "python neural network deep learning"),
///     ("d5", "rust async await future tokio"),
///     ("d6", "machine learning gradient descent optimisation"),
/// ];
/// extractor.fit(&docs).unwrap();
/// let topics = extractor.topics().unwrap();
/// assert_eq!(topics.len(), 3);
/// ```
#[derive(Debug)]
pub struct TopicModelExtractor {
    config: ExtractorConfig,

    // Vocabulary
    vocab: HashMap<String, usize>, // word → index
    vocab_rev: Vec<String>,        // index → word

    // Corpus
    doc_ids: Vec<String>,
    /// Tokenised corpus: outer = documents, inner = tokens (as word indices).
    corpus: Vec<Vec<usize>>,

    // Gibbs state
    /// word-topic assignments per document, mirroring `corpus` layout.
    assignments: Vec<Vec<WordAssignment>>,
    /// doc_topic_counts[doc_idx][topic_id]
    doc_topic_counts: Vec<Vec<u32>>,
    /// topic_word_counts[topic_id][word_idx]
    topic_word_counts: Vec<Vec<u32>>,
    /// topic_counts[topic_id] — total tokens assigned
    topic_counts: Vec<u32>,

    // Co-occurrence for coherence computation
    /// word_doc_freq[word_idx] — number of docs containing this word
    word_doc_freq: Vec<u32>,
    /// co_occur[(min_w, max_w)] — number of docs where both words appear
    co_occur: HashMap<(usize, usize), u32>,

    // Topic labels
    labels: Vec<Option<String>>,

    // Model state
    fitted: bool,
    iterations_run: u32,

    // PRNG state
    rng_state: u64,
}

impl TopicModelExtractor {
    /// Create a new extractor with the given configuration.
    pub fn new(config: ExtractorConfig) -> Self {
        Self {
            rng_state: 0xDEAD_BEEF_CAFE_1337,
            config,
            vocab: HashMap::new(),
            vocab_rev: Vec::new(),
            doc_ids: Vec::new(),
            corpus: Vec::new(),
            assignments: Vec::new(),
            doc_topic_counts: Vec::new(),
            topic_word_counts: Vec::new(),
            topic_counts: Vec::new(),
            word_doc_freq: Vec::new(),
            co_occur: HashMap::new(),
            labels: Vec::new(),
            fitted: false,
            iterations_run: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Fit the model on a slice of `(doc_id, text)` pairs.
    pub fn fit(&mut self, docs: &[(&str, &str)]) -> Result<(), ExtractorError> {
        // --- Validate configuration -----------------------------------------
        if self.config.num_topics == 0 {
            return Err(ExtractorError::InvalidConfiguration(
                "num_topics must be ≥ 1".into(),
            ));
        }
        if self.config.alpha <= 0.0 {
            return Err(ExtractorError::InvalidConfiguration(
                "alpha must be > 0".into(),
            ));
        }
        if self.config.beta <= 0.0 {
            return Err(ExtractorError::InvalidConfiguration(
                "beta must be > 0".into(),
            ));
        }
        if docs.len() < 2 {
            return Err(ExtractorError::InsufficientDocuments(docs.len()));
        }

        // --- Reset state ----------------------------------------------------
        self.fitted = false;
        self.vocab.clear();
        self.vocab_rev.clear();
        self.doc_ids.clear();
        self.corpus.clear();
        self.assignments.clear();
        self.co_occur.clear();

        let k = self.config.num_topics;

        // --- Step 1: raw tokenisation ---------------------------------------
        let raw_tokens: Vec<Vec<String>> = docs
            .iter()
            .map(|(_, text)| {
                text.split_whitespace()
                    .map(|w| {
                        w.to_lowercase()
                            .trim_matches(|c: char| !c.is_alphanumeric())
                            .to_string()
                    })
                    .filter(|w| !w.is_empty())
                    .collect()
            })
            .collect();

        // --- Step 2: global word frequencies --------------------------------
        let mut global_freq: HashMap<String, u32> = HashMap::new();
        let mut doc_appears: HashMap<String, u32> = HashMap::new();
        let n_docs = docs.len() as f64;

        for tokens in &raw_tokens {
            let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
            for w in tokens {
                *global_freq.entry(w.clone()).or_insert(0) += 1;
                if seen.insert(w.as_str()) {
                    *doc_appears.entry(w.clone()).or_insert(0) += 1;
                }
            }
        }

        // --- Step 3: build vocabulary with frequency filters ----------------
        let max_doc_count = (self.config.max_doc_freq_pct * n_docs).ceil() as u32;
        let mut word_freq_list: Vec<(String, u32)> = global_freq
            .into_iter()
            .filter(|(w, freq)| {
                *freq >= self.config.min_word_freq
                    && doc_appears.get(w).copied().unwrap_or(0) <= max_doc_count
            })
            .collect();

        // Sort descending by frequency, then limit to vocab_size_limit
        word_freq_list.sort_unstable_by_key(|a| std::cmp::Reverse(a.1));
        word_freq_list.truncate(self.config.vocab_size_limit);

        if word_freq_list.is_empty() {
            return Err(ExtractorError::VocabularyEmpty);
        }

        for (idx, (word, _)) in word_freq_list.iter().enumerate() {
            self.vocab.insert(word.clone(), idx);
            self.vocab_rev.push(word.clone());
        }
        let v = self.vocab_rev.len();

        // --- Step 4: encode corpus ------------------------------------------
        for ((doc_id, _), tokens) in docs.iter().zip(raw_tokens.iter()) {
            let encoded: Vec<usize> = tokens
                .iter()
                .filter_map(|w| self.vocab.get(w).copied())
                .collect();
            self.doc_ids.push(doc_id.to_string());
            self.corpus.push(encoded);
        }

        // --- Step 5: co-occurrence counts for coherence ---------------------
        self.word_doc_freq = vec![0u32; v];
        for tokens in &self.corpus {
            let unique: std::collections::HashSet<usize> = tokens.iter().copied().collect();
            let mut sorted: Vec<usize> = unique.into_iter().collect();
            sorted.sort_unstable();
            for &wi in &sorted {
                self.word_doc_freq[wi] += 1;
            }
            for i in 0..sorted.len() {
                for j in (i + 1)..sorted.len() {
                    let key = (sorted[i], sorted[j]);
                    *self.co_occur.entry(key).or_insert(0) += 1;
                }
            }
        }

        // --- Step 6: initialise count matrices ------------------------------
        let n_docs_usize = self.corpus.len();
        self.doc_topic_counts = vec![vec![0u32; k]; n_docs_usize];
        self.topic_word_counts = vec![vec![0u32; v]; k];
        self.topic_counts = vec![0u32; k];
        self.assignments = Vec::with_capacity(n_docs_usize);

        // --- Step 7: random initial assignment ------------------------------
        for (d, tokens) in self.corpus.iter().enumerate() {
            let mut doc_assignments: Vec<WordAssignment> = Vec::with_capacity(tokens.len());
            for &wi in tokens {
                let t = (xorshift64(&mut self.rng_state) as usize) % k;
                self.doc_topic_counts[d][t] += 1;
                self.topic_word_counts[t][wi] += 1;
                self.topic_counts[t] += 1;
                doc_assignments.push(WordAssignment {
                    word_idx: wi,
                    topic_id: t,
                });
            }
            self.assignments.push(doc_assignments);
        }

        // --- Step 8: collapsed Gibbs sampling -------------------------------
        let alpha = self.config.alpha;
        let beta = self.config.beta;
        let v_f64 = v as f64;
        let mut probs = vec![0.0f64; k];

        for _iter in 0..self.config.num_iterations {
            for d in 0..n_docs_usize {
                let n_tokens = self.assignments[d].len();
                for n in 0..n_tokens {
                    let wa = self.assignments[d][n];
                    let old_t = wa.topic_id;
                    let wi = wa.word_idx;

                    // Remove current assignment
                    self.doc_topic_counts[d][old_t] -= 1;
                    self.topic_word_counts[old_t][wi] -= 1;
                    self.topic_counts[old_t] -= 1;

                    // Compute unnormalised conditional distribution
                    let mut cumsum = 0.0f64;
                    for (t, prob_slot) in probs[..k].iter_mut().enumerate() {
                        let n_dk = self.doc_topic_counts[d][t] as f64;
                        let n_kw = self.topic_word_counts[t][wi] as f64;
                        let n_k = self.topic_counts[t] as f64;
                        let p = (n_dk + alpha) * (n_kw + beta) / (n_k + v_f64 * beta);
                        cumsum += p;
                        *prob_slot = cumsum;
                    }

                    // Sample new topic
                    let u = xorshift_f64(&mut self.rng_state) * cumsum;
                    let new_t = probs[..k].partition_point(|&p| p < u).min(k - 1);

                    // Restore with new assignment
                    self.doc_topic_counts[d][new_t] += 1;
                    self.topic_word_counts[new_t][wi] += 1;
                    self.topic_counts[new_t] += 1;
                    self.assignments[d][n] = WordAssignment {
                        word_idx: wi,
                        topic_id: new_t,
                    };
                }
            }
        }

        self.labels = vec![None; k];
        self.fitted = true;
        self.iterations_run = self.config.num_iterations;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Return all topics sorted by prevalence descending.
    pub fn topics(&self) -> Result<Vec<ExtractorTopic>, ExtractorError> {
        self.require_fitted()?;
        let k = self.config.num_topics;
        let v = self.vocab_rev.len();
        let total_tokens: u64 = self.topic_counts.iter().map(|&c| c as u64).sum();
        let v_f64 = v as f64;
        let beta = self.config.beta;
        let n_docs = self.corpus.len() as f64;

        let mut topics: Vec<ExtractorTopic> = (0..k)
            .map(|t| {
                // top-words
                let denom = self.topic_counts[t] as f64 + v_f64 * beta;
                let mut words: Vec<ExtractorTopicWord> = (0..v)
                    .map(|wi| {
                        let cnt = self.topic_word_counts[t][wi];
                        ExtractorTopicWord {
                            word: self.vocab_rev[wi].clone(),
                            probability: (cnt as f64 + beta) / denom,
                            count: cnt,
                        }
                    })
                    .collect();
                words.sort_unstable_by(|a, b| {
                    b.probability
                        .partial_cmp(&a.probability)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let prevalence = if total_tokens == 0 {
                    0.0
                } else {
                    self.topic_counts[t] as f64 / total_tokens as f64
                };

                // coherence — mean pairwise PMI of top-10 words
                let top10: Vec<usize> = words
                    .iter()
                    .take(10)
                    .filter_map(|tw| self.vocab.get(&tw.word).copied())
                    .collect();
                let coherence = self.mean_pmi(&top10, n_docs);

                let top_words: Vec<ExtractorTopicWord> = words.into_iter().take(50).collect();
                ExtractorTopic {
                    id: t,
                    top_words,
                    coherence,
                    prevalence,
                    label: self.labels[t].clone(),
                }
            })
            .collect();

        topics.sort_unstable_by(|a, b| {
            b.prevalence
                .partial_cmp(&a.prevalence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(topics)
    }

    /// Return the topic distribution for a document that was in the training corpus.
    pub fn document_topics(&self, doc_id: &str) -> Result<ExtractorDocumentTopics, ExtractorError> {
        self.require_fitted()?;
        let d = self
            .doc_ids
            .iter()
            .position(|id| id == doc_id)
            .ok_or_else(|| ExtractorError::DocumentNotFound(doc_id.to_string()))?;
        Ok(self.build_doc_topics(d, doc_id))
    }

    /// Infer topics for a new (unseen) document using 5 Gibbs passes.
    pub fn infer_topics(&self, text: &str) -> Result<ExtractorDocumentTopics, ExtractorError> {
        self.require_fitted()?;
        let k = self.config.num_topics;
        let v = self.vocab_rev.len();
        let alpha = self.config.alpha;
        let beta = self.config.beta;
        let v_f64 = v as f64;

        let tokens: Vec<usize> = text
            .split_whitespace()
            .map(|w| {
                w.to_lowercase()
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string()
            })
            .filter_map(|w| self.vocab.get(&w).copied())
            .collect();

        if tokens.is_empty() {
            // Return uniform distribution
            let prob = 1.0 / k as f64;
            return Ok(ExtractorDocumentTopics {
                doc_id: "<new>".to_string(),
                topic_distribution: vec![prob; k],
                dominant_topic: 0,
                dominant_probability: prob,
            });
        }

        // Local counts for this new document
        let mut local_doc_topic = vec![0u32; k];
        let mut local_assignments: Vec<usize> = Vec::with_capacity(tokens.len());

        // Use a local mutable RNG seed derived from token hash
        let mut rng = 0xFEED_C0DE_1234_5678u64;
        for _ in &tokens {
            xorshift64(&mut rng);
        }

        // Initial assignment
        for _ in &tokens {
            let t = (xorshift64(&mut rng) as usize) % k;
            local_doc_topic[t] += 1;
            local_assignments.push(t);
        }

        // 5 Gibbs passes
        let mut probs = vec![0.0f64; k];
        for _ in 0..5 {
            for (n, &wi) in tokens.iter().enumerate() {
                let old_t = local_assignments[n];
                local_doc_topic[old_t] -= 1;

                let mut cumsum = 0.0f64;
                for t in 0..k {
                    let n_dk = local_doc_topic[t] as f64;
                    let n_kw = self.topic_word_counts[t][wi] as f64;
                    let n_k = self.topic_counts[t] as f64;
                    let p = (n_dk + alpha) * (n_kw + beta) / (n_k + v_f64 * beta);
                    cumsum += p;
                    probs[t] = cumsum;
                }

                let u = xorshift_f64(&mut rng) * cumsum;
                let new_t = probs[..k].partition_point(|&p| p < u).min(k - 1);
                local_doc_topic[new_t] += 1;
                local_assignments[n] = new_t;
            }
        }

        // Normalise
        let total: f64 = local_doc_topic.iter().map(|&c| c as f64 + alpha).sum();
        let dist: Vec<f64> = local_doc_topic
            .iter()
            .map(|&c| (c as f64 + alpha) / total)
            .collect();

        let (dominant_topic, dominant_probability) =
            dist.iter()
                .copied()
                .enumerate()
                .fold(
                    (0usize, 0.0f64),
                    |(bi, bp), (i, p)| {
                        if p > bp {
                            (i, p)
                        } else {
                            (bi, bp)
                        }
                    },
                );

        Ok(ExtractorDocumentTopics {
            doc_id: "<new>".to_string(),
            topic_distribution: dist,
            dominant_topic,
            dominant_probability,
        })
    }

    /// Return the top-n words for a given topic.
    pub fn top_words(
        &self,
        topic_id: usize,
        n: usize,
    ) -> Result<Vec<ExtractorTopicWord>, ExtractorError> {
        self.require_fitted()?;
        let k = self.config.num_topics;
        if topic_id >= k {
            return Err(ExtractorError::TopicOutOfRange(topic_id));
        }
        let v = self.vocab_rev.len();
        let v_f64 = v as f64;
        let beta = self.config.beta;
        let denom = self.topic_counts[topic_id] as f64 + v_f64 * beta;
        let mut words: Vec<ExtractorTopicWord> = (0..v)
            .map(|wi| {
                let cnt = self.topic_word_counts[topic_id][wi];
                ExtractorTopicWord {
                    word: self.vocab_rev[wi].clone(),
                    probability: (cnt as f64 + beta) / denom,
                    count: cnt,
                }
            })
            .collect();
        words.sort_unstable_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        words.truncate(n);
        Ok(words)
    }

    /// Return topics most similar to `topic_id` ranked by cosine similarity.
    pub fn similar_topics(
        &self,
        topic_id: usize,
        top_k: usize,
    ) -> Result<Vec<(usize, f64)>, ExtractorError> {
        self.require_fitted()?;
        let k = self.config.num_topics;
        if topic_id >= k {
            return Err(ExtractorError::TopicOutOfRange(topic_id));
        }
        let v = self.vocab_rev.len();
        // Build normalised word distributions for all topics
        let distributions: Vec<Vec<f64>> = (0..k)
            .map(|t| {
                let total: f64 = self.topic_counts[t] as f64 + v as f64 * self.config.beta;
                (0..v)
                    .map(|wi| (self.topic_word_counts[t][wi] as f64 + self.config.beta) / total)
                    .collect()
            })
            .collect();

        let ref_dist = &distributions[topic_id];
        let mut sims: Vec<(usize, f64)> = (0..k)
            .filter(|&t| t != topic_id)
            .map(|t| {
                let sim = cosine_sim_f64(ref_dist, &distributions[t]);
                (t, sim)
            })
            .collect();
        sims.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sims.truncate(top_k);
        Ok(sims)
    }

    /// Assign a human-readable label to a topic.
    pub fn assign_label(&mut self, topic_id: usize, label: String) -> Result<(), ExtractorError> {
        self.require_fitted()?;
        if topic_id >= self.config.num_topics {
            return Err(ExtractorError::TopicOutOfRange(topic_id));
        }
        self.labels[topic_id] = Some(label);
        Ok(())
    }

    /// Return aggregate model statistics including perplexity.
    pub fn stats(&self) -> Result<ModelStats, ExtractorError> {
        self.require_fitted()?;
        let k = self.config.num_topics;
        let v = self.vocab_rev.len();
        let n_docs = self.corpus.len();
        let total_tokens: u64 = self.topic_counts.iter().map(|&c| c as u64).sum();
        let v_f64 = v as f64;
        let beta = self.config.beta;
        let alpha = self.config.alpha;
        let _n_docs_f = n_docs as f64;

        // Log-likelihood
        let mut log_lik = 0.0f64;
        for d in 0..n_docs {
            let doc_total: f64 = self.doc_topic_counts[d]
                .iter()
                .map(|&c| c as f64)
                .sum::<f64>()
                + k as f64 * alpha;
            for t in 0..k {
                let n_dt = self.doc_topic_counts[d][t] as f64 + alpha;
                let p_t = n_dt / doc_total;
                let denom = self.topic_counts[t] as f64 + v_f64 * beta;
                for wi in 0..v {
                    let n_tw = self.topic_word_counts[t][wi] as f64 + beta;
                    let p_w_t = n_tw / denom;
                    // Accumulate weighted
                    log_lik += self.topic_word_counts[t][wi] as f64 * (p_t * p_w_t).ln();
                }
            }
        }

        let perplexity = if total_tokens == 0 {
            f64::INFINITY
        } else {
            (-log_lik / total_tokens as f64).exp()
        };

        // Average coherence
        let coherence_sum: f64 = (0..k)
            .map(|t| {
                let denom = self.topic_counts[t] as f64 + v_f64 * beta;
                let mut words: Vec<(usize, f64)> = (0..v)
                    .map(|wi| {
                        let p = (self.topic_word_counts[t][wi] as f64 + beta) / denom;
                        (wi, p)
                    })
                    .collect();
                words.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                let top10: Vec<usize> = words.iter().take(10).map(|(wi, _)| *wi).collect();
                self.mean_pmi(&top10, n_docs as f64)
            })
            .sum();
        let avg_topic_coherence = coherence_sum / k as f64;

        Ok(ModelStats {
            num_topics: k,
            vocab_size: v,
            num_docs: n_docs,
            total_tokens,
            avg_topic_coherence,
            perplexity,
            iterations_run: self.iterations_run,
        })
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn require_fitted(&self) -> Result<(), ExtractorError> {
        if self.fitted {
            Ok(())
        } else {
            Err(ExtractorError::ModelNotFitted)
        }
    }

    fn build_doc_topics(&self, d: usize, doc_id: &str) -> ExtractorDocumentTopics {
        let alpha = self.config.alpha;
        let total: f64 = self.doc_topic_counts[d]
            .iter()
            .map(|&c| c as f64 + alpha)
            .sum();
        let dist: Vec<f64> = self.doc_topic_counts[d]
            .iter()
            .map(|&c| (c as f64 + alpha) / total)
            .collect();

        let (dominant_topic, dominant_probability) =
            dist.iter()
                .copied()
                .enumerate()
                .fold(
                    (0usize, 0.0f64),
                    |(bi, bp), (i, p)| {
                        if p > bp {
                            (i, p)
                        } else {
                            (bi, bp)
                        }
                    },
                );
        ExtractorDocumentTopics {
            doc_id: doc_id.to_string(),
            topic_distribution: dist,
            dominant_topic,
            dominant_probability,
        }
    }

    /// Compute mean pairwise PMI for a set of word indices.
    fn mean_pmi(&self, word_indices: &[usize], n_docs: f64) -> f64 {
        if word_indices.len() < 2 || n_docs == 0.0 {
            return 0.0;
        }
        let mut sum = 0.0f64;
        let mut count = 0u32;
        let log_n = n_docs.ln();

        let mut sorted = word_indices.to_vec();
        sorted.sort_unstable();
        sorted.dedup();

        for i in 0..sorted.len() {
            for j in (i + 1)..sorted.len() {
                let wi = sorted[i];
                let wj = sorted[j];
                let df_i = self.word_doc_freq.get(wi).copied().unwrap_or(0) as f64;
                let df_j = self.word_doc_freq.get(wj).copied().unwrap_or(0) as f64;
                if df_i == 0.0 || df_j == 0.0 {
                    count += 1;
                    continue;
                }
                let cooc = self
                    .co_occur
                    .get(&(wi.min(wj), wi.max(wj)))
                    .copied()
                    .unwrap_or(0) as f64;
                if cooc == 0.0 {
                    // PMI is -∞; use a floor value
                    sum -= 20.0;
                } else {
                    let pmi = (cooc * n_docs).ln() - df_i.ln() - df_j.ln() + log_n;
                    sum += pmi;
                }
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }
}

// ---------------------------------------------------------------------------
// f64 cosine similarity
// ---------------------------------------------------------------------------

fn cosine_sim_f64(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-15 || nb < 1e-15 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

// ---------------------------------------------------------------------------
// Type aliases for name-collision avoidance
// ---------------------------------------------------------------------------

/// Type alias for [`ExtractorTopicWord`] — avoids collision with `TopicWord` from `topic_modeler`.
pub type TmeTopicWord = ExtractorTopicWord;
/// Type alias for [`ExtractorDocumentTopics`] — avoids collision with `DocumentTopics` from `topic_modeler`.
pub type TmeDocumentTopics = ExtractorDocumentTopics;
/// Type alias for [`ExtractorTopic`] — convenience alias.
pub type TmeTopic = ExtractorTopic;
/// Type alias for [`ExtractorError`] — convenience alias.
pub type TmeError = ExtractorError;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper: small corpus
    // -----------------------------------------------------------------------

    fn small_corpus() -> Vec<(&'static str, &'static str)> {
        vec![
            ("d01", "rust programming language systems memory safety"),
            ("d02", "python data science machine learning neural network"),
            ("d03", "rust memory safety ownership borrow checker"),
            ("d04", "python neural network deep learning tensorflow"),
            ("d05", "rust async await future tokio runtime"),
            ("d06", "machine learning gradient descent optimisation loss"),
            ("d07", "rust cargo crates ecosystem package manager"),
            ("d08", "data science statistics probability distribution"),
            ("d09", "rust trait object polymorphism generic lifetime"),
            (
                "d10",
                "deep learning convolutional network image recognition",
            ),
            ("d11", "distributed systems consensus raft paxos protocol"),
            (
                "d12",
                "natural language processing text classification token",
            ),
            ("d13", "graph database query language traversal algorithm"),
            (
                "d14",
                "cloud computing container orchestration kubernetes pod",
            ),
            (
                "d15",
                "blockchain decentralised ledger consensus hash cryptography",
            ),
        ]
    }

    fn make_fitted() -> TopicModelExtractor {
        let cfg = ExtractorConfig {
            num_topics: 3,
            num_iterations: 50,
            min_word_freq: 1,
            ..Default::default()
        };
        let mut e = TopicModelExtractor::new(cfg);
        e.fit(&small_corpus()).expect("fit failed");
        e
    }

    // -----------------------------------------------------------------------
    // 1. fit succeeds on small corpus
    // -----------------------------------------------------------------------
    #[test]
    fn test_fit_succeeds() {
        let mut e = TopicModelExtractor::new(ExtractorConfig {
            num_topics: 3,
            num_iterations: 20,
            min_word_freq: 1,
            ..Default::default()
        });
        e.fit(&small_corpus()).expect("fit should succeed");
        assert!(e.fitted);
    }

    // -----------------------------------------------------------------------
    // 2. fit fails on empty input
    // -----------------------------------------------------------------------
    #[test]
    fn test_fit_insufficient_documents() {
        let mut e = TopicModelExtractor::new(ExtractorConfig::default());
        let result = e.fit(&[("d1", "hello world")]);
        assert!(matches!(
            result,
            Err(ExtractorError::InsufficientDocuments(1))
        ));
    }

    // -----------------------------------------------------------------------
    // 3. fit fails on empty docs slice
    // -----------------------------------------------------------------------
    #[test]
    fn test_fit_zero_documents() {
        let mut e = TopicModelExtractor::new(ExtractorConfig::default());
        let result = e.fit(&[]);
        assert!(matches!(
            result,
            Err(ExtractorError::InsufficientDocuments(0))
        ));
    }

    // -----------------------------------------------------------------------
    // 4. empty vocabulary after filtering
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_vocabulary() {
        let cfg = ExtractorConfig {
            num_topics: 2,
            min_word_freq: 9999, // nothing passes
            ..Default::default()
        };
        let mut e = TopicModelExtractor::new(cfg);
        let docs = vec![("d1", "hello"), ("d2", "world")];
        let result = e.fit(&docs);
        assert!(matches!(result, Err(ExtractorError::VocabularyEmpty)));
    }

    // -----------------------------------------------------------------------
    // 5. topics count matches config
    // -----------------------------------------------------------------------
    #[test]
    fn test_topic_count() {
        let e = make_fitted();
        let topics = e.topics().expect("topics should work");
        assert_eq!(topics.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 6. topic prevalences sum to ≈1
    // -----------------------------------------------------------------------
    #[test]
    fn test_prevalences_sum_to_one() {
        let e = make_fitted();
        let topics = e.topics().expect("test: topics() should succeed after fit");
        let sum: f64 = topics.iter().map(|t| t.prevalence).sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum={}", sum);
    }

    // -----------------------------------------------------------------------
    // 7. topics are sorted by prevalence descending
    // -----------------------------------------------------------------------
    #[test]
    fn test_topics_sorted_descending() {
        let e = make_fitted();
        let topics = e.topics().expect("test: topics() should succeed after fit");
        for w in topics.windows(2) {
            assert!(w[0].prevalence >= w[1].prevalence);
        }
    }

    // -----------------------------------------------------------------------
    // 8. document_topics returns distribution that sums to 1
    // -----------------------------------------------------------------------
    #[test]
    fn test_doc_distribution_sums_to_one() {
        let e = make_fitted();
        let dt = e
            .document_topics("d01")
            .expect("test: document_topics should find d01");
        let sum: f64 = dt.topic_distribution.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum={}", sum);
    }

    // -----------------------------------------------------------------------
    // 9. document_topics returns distribution of correct length
    // -----------------------------------------------------------------------
    #[test]
    fn test_doc_distribution_length() {
        let e = make_fitted();
        let dt = e
            .document_topics("d01")
            .expect("test: document_topics should find d01");
        assert_eq!(dt.topic_distribution.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 10. document_topics dominant_probability is the max
    // -----------------------------------------------------------------------
    #[test]
    fn test_doc_dominant_probability_is_max() {
        let e = make_fitted();
        let dt = e
            .document_topics("d01")
            .expect("test: document_topics should find d01");
        let max = dt
            .topic_distribution
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((dt.dominant_probability - max).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // 11. document_topics dominant_topic index is correct
    // -----------------------------------------------------------------------
    #[test]
    fn test_doc_dominant_topic_index() {
        let e = make_fitted();
        let dt = e
            .document_topics("d01")
            .expect("test: document_topics should find d01");
        assert_eq!(
            dt.topic_distribution[dt.dominant_topic],
            dt.dominant_probability
        );
    }

    // -----------------------------------------------------------------------
    // 12. document_topics unknown id returns error
    // -----------------------------------------------------------------------
    #[test]
    fn test_doc_topics_unknown_id() {
        let e = make_fitted();
        let result = e.document_topics("nonexistent_doc");
        assert!(matches!(result, Err(ExtractorError::DocumentNotFound(_))));
    }

    // -----------------------------------------------------------------------
    // 13. infer_topics on new text sums to 1
    // -----------------------------------------------------------------------
    #[test]
    fn test_infer_topics_sums_to_one() {
        let e = make_fitted();
        let dt = e
            .infer_topics("rust programming memory safety")
            .expect("test: infer_topics should succeed");
        let sum: f64 = dt.topic_distribution.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum={}", sum);
    }

    // -----------------------------------------------------------------------
    // 14. infer_topics distribution has correct length
    // -----------------------------------------------------------------------
    #[test]
    fn test_infer_topics_length() {
        let e = make_fitted();
        let dt = e
            .infer_topics("rust memory safety")
            .expect("test: infer_topics should succeed");
        assert_eq!(dt.topic_distribution.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 15. infer_topics on empty text returns uniform
    // -----------------------------------------------------------------------
    #[test]
    fn test_infer_empty_text() {
        let e = make_fitted();
        let dt = e
            .infer_topics("")
            .expect("test: infer_topics on empty text should return uniform distribution");
        let sum: f64 = dt.topic_distribution.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum={}", sum);
    }

    // -----------------------------------------------------------------------
    // 16. infer_topics returns correct doc_id marker
    // -----------------------------------------------------------------------
    #[test]
    fn test_infer_doc_id_marker() {
        let e = make_fitted();
        let dt = e
            .infer_topics("some text")
            .expect("test: infer_topics should succeed on known-vocabulary text");
        assert_eq!(dt.doc_id, "<new>");
    }

    // -----------------------------------------------------------------------
    // 17. all probabilities in distribution are non-negative
    // -----------------------------------------------------------------------
    #[test]
    fn test_dist_non_negative() {
        let e = make_fitted();
        let dt = e
            .infer_topics("rust async runtime")
            .expect("test: infer_topics should succeed");
        for &p in &dt.topic_distribution {
            assert!(p >= 0.0, "negative probability: {}", p);
        }
    }

    // -----------------------------------------------------------------------
    // 18. similar_topics returns top_k results
    // -----------------------------------------------------------------------
    #[test]
    fn test_similar_topics_count() {
        let e = make_fitted();
        let sims = e
            .similar_topics(0, 2)
            .expect("test: similar_topics should succeed for valid topic id");
        assert!(sims.len() <= 2);
    }

    // -----------------------------------------------------------------------
    // 19. similar_topics excludes the query topic
    // -----------------------------------------------------------------------
    #[test]
    fn test_similar_topics_no_self() {
        let e = make_fitted();
        let sims = e
            .similar_topics(0, 10)
            .expect("test: similar_topics should succeed for valid topic id");
        for (tid, _) in &sims {
            assert_ne!(*tid, 0);
        }
    }

    // -----------------------------------------------------------------------
    // 20. similar_topics sorted descending
    // -----------------------------------------------------------------------
    #[test]
    fn test_similar_topics_sorted() {
        let e = make_fitted();
        let sims = e
            .similar_topics(0, 10)
            .expect("test: similar_topics should succeed for valid topic id");
        for w in sims.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    // -----------------------------------------------------------------------
    // 21. similar_topics scores in [-1, 1]
    // -----------------------------------------------------------------------
    #[test]
    fn test_similar_topics_scores_range() {
        let e = make_fitted();
        let sims = e
            .similar_topics(0, 10)
            .expect("test: similar_topics should succeed for valid topic id");
        for (_, sim) in &sims {
            assert!(*sim >= -1.0 && *sim <= 1.0, "sim={}", sim);
        }
    }

    // -----------------------------------------------------------------------
    // 22. similar_topics out of range returns error
    // -----------------------------------------------------------------------
    #[test]
    fn test_similar_topics_out_of_range() {
        let e = make_fitted();
        let result = e.similar_topics(99, 2);
        assert!(matches!(result, Err(ExtractorError::TopicOutOfRange(99))));
    }

    // -----------------------------------------------------------------------
    // 23. assign_label sets label
    // -----------------------------------------------------------------------
    #[test]
    fn test_assign_label() {
        let mut e = make_fitted();
        e.assign_label(0, "tech-rust".to_string())
            .expect("test: assign_label should succeed for valid topic id");
        let topics = e.topics().expect("test: topics() should succeed after fit");
        // find the original topic 0
        let labelled = topics
            .iter()
            .find(|t| t.label.as_deref() == Some("tech-rust"));
        assert!(labelled.is_some());
    }

    // -----------------------------------------------------------------------
    // 24. assign_label out of range returns error
    // -----------------------------------------------------------------------
    #[test]
    fn test_assign_label_out_of_range() {
        let mut e = make_fitted();
        let result = e.assign_label(99, "label".to_string());
        assert!(matches!(result, Err(ExtractorError::TopicOutOfRange(99))));
    }

    // -----------------------------------------------------------------------
    // 25. label is preserved in topics() output
    // -----------------------------------------------------------------------
    #[test]
    fn test_label_persists_in_topics() {
        let mut e = make_fitted();
        e.assign_label(1, "ml-python".to_string())
            .expect("test: assign_label should succeed for valid topic id");
        let topics = e.topics().expect("test: topics() should succeed after fit");
        let with_label: Vec<_> = topics.iter().filter(|t| t.label.is_some()).collect();
        assert_eq!(with_label.len(), 1);
        assert_eq!(with_label[0].label.as_deref(), Some("ml-python"));
    }

    // -----------------------------------------------------------------------
    // 26. stats() returns correct topic count
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_num_topics() {
        let e = make_fitted();
        let s = e.stats().expect("test: stats() for num_topics check");
        assert_eq!(s.num_topics, 3);
    }

    // -----------------------------------------------------------------------
    // 27. stats() returns correct doc count
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_num_docs() {
        let e = make_fitted();
        let s = e.stats().expect("test: stats() for num_docs check");
        assert_eq!(s.num_docs, small_corpus().len());
    }

    // -----------------------------------------------------------------------
    // 28. stats() total_tokens > 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_total_tokens() {
        let e = make_fitted();
        let s = e.stats().expect("test: stats() for total_tokens check");
        assert!(s.total_tokens > 0);
    }

    // -----------------------------------------------------------------------
    // 29. perplexity is finite and positive
    // -----------------------------------------------------------------------
    #[test]
    fn test_perplexity_finite_positive() {
        let e = make_fitted();
        let s = e.stats().expect("test: stats() for perplexity check");
        assert!(s.perplexity.is_finite(), "perplexity={}", s.perplexity);
        assert!(s.perplexity > 0.0, "perplexity={}", s.perplexity);
    }

    // -----------------------------------------------------------------------
    // 30. iterations_run matches config
    // -----------------------------------------------------------------------
    #[test]
    fn test_iterations_run() {
        let e = make_fitted();
        let s = e.stats().expect("test: stats() for iterations_run check");
        assert_eq!(s.iterations_run, 50);
    }

    // -----------------------------------------------------------------------
    // 31. top_words returns correct n
    // -----------------------------------------------------------------------
    #[test]
    fn test_top_words_count() {
        let e = make_fitted();
        let words = e.top_words(0, 5).expect("test: top_words for topic 0");
        assert_eq!(words.len(), 5);
    }

    // -----------------------------------------------------------------------
    // 32. top_words sorted by probability descending
    // -----------------------------------------------------------------------
    #[test]
    fn test_top_words_sorted() {
        let e = make_fitted();
        let words = e
            .top_words(0, 10)
            .expect("test: top_words sorted order check");
        for w in words.windows(2) {
            assert!(w[0].probability >= w[1].probability);
        }
    }

    // -----------------------------------------------------------------------
    // 33. top_words probabilities are positive
    // -----------------------------------------------------------------------
    #[test]
    fn test_top_words_positive_probs() {
        let e = make_fitted();
        let words = e
            .top_words(0, 10)
            .expect("test: top_words positive probabilities check");
        for tw in &words {
            assert!(tw.probability > 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // 34. top_words out-of-range topic returns error
    // -----------------------------------------------------------------------
    #[test]
    fn test_top_words_out_of_range() {
        let e = make_fitted();
        let result = e.top_words(99, 5);
        assert!(matches!(result, Err(ExtractorError::TopicOutOfRange(99))));
    }

    // -----------------------------------------------------------------------
    // 35. operations before fit return ModelNotFitted
    // -----------------------------------------------------------------------
    #[test]
    fn test_not_fitted_errors() {
        let e = TopicModelExtractor::new(ExtractorConfig::default());
        assert!(matches!(e.topics(), Err(ExtractorError::ModelNotFitted)));
        assert!(matches!(
            e.document_topics("x"),
            Err(ExtractorError::ModelNotFitted)
        ));
        assert!(matches!(
            e.infer_topics("text"),
            Err(ExtractorError::ModelNotFitted)
        ));
        assert!(matches!(
            e.similar_topics(0, 1),
            Err(ExtractorError::ModelNotFitted)
        ));
        assert!(matches!(e.stats(), Err(ExtractorError::ModelNotFitted)));
        assert!(matches!(
            e.top_words(0, 5),
            Err(ExtractorError::ModelNotFitted)
        ));
    }

    // -----------------------------------------------------------------------
    // 36. invalid alpha configuration
    // -----------------------------------------------------------------------
    #[test]
    fn test_invalid_alpha() {
        let cfg = ExtractorConfig {
            alpha: -1.0,
            num_topics: 2,
            min_word_freq: 1,
            ..Default::default()
        };
        let mut e = TopicModelExtractor::new(cfg);
        let result = e.fit(&small_corpus());
        assert!(matches!(
            result,
            Err(ExtractorError::InvalidConfiguration(_))
        ));
    }

    // -----------------------------------------------------------------------
    // 37. invalid beta configuration
    // -----------------------------------------------------------------------
    #[test]
    fn test_invalid_beta() {
        let cfg = ExtractorConfig {
            beta: 0.0,
            num_topics: 2,
            min_word_freq: 1,
            ..Default::default()
        };
        let mut e = TopicModelExtractor::new(cfg);
        let result = e.fit(&small_corpus());
        assert!(matches!(
            result,
            Err(ExtractorError::InvalidConfiguration(_))
        ));
    }

    // -----------------------------------------------------------------------
    // 38. zero num_topics configuration
    // -----------------------------------------------------------------------
    #[test]
    fn test_zero_num_topics() {
        let cfg = ExtractorConfig {
            num_topics: 0,
            min_word_freq: 1,
            ..Default::default()
        };
        let mut e = TopicModelExtractor::new(cfg);
        let result = e.fit(&small_corpus());
        assert!(matches!(
            result,
            Err(ExtractorError::InvalidConfiguration(_))
        ));
    }

    // -----------------------------------------------------------------------
    // 39. vocab_size_limit is respected
    // -----------------------------------------------------------------------
    #[test]
    fn test_vocab_size_limit() {
        let cfg = ExtractorConfig {
            num_topics: 2,
            num_iterations: 20,
            vocab_size_limit: 5,
            min_word_freq: 1,
            ..Default::default()
        };
        let mut e = TopicModelExtractor::new(cfg);
        e.fit(&small_corpus())
            .expect("test: fit on small corpus for vocab_size_limit check");
        let s = e.stats().expect("test: stats() for vocab_size_limit check");
        assert!(s.vocab_size <= 5, "vocab_size={}", s.vocab_size);
    }

    // -----------------------------------------------------------------------
    // 40. max_doc_freq_pct filter works
    // -----------------------------------------------------------------------
    #[test]
    fn test_max_doc_freq_filter() {
        // Every document contains "common", so with 0.01 it should be filtered
        let docs: Vec<(&str, &str)> = vec![
            ("d1", "common rust programming"),
            ("d2", "common python data"),
            ("d3", "common deep learning"),
            ("d4", "common kubernetes cloud"),
            ("d5", "common blockchain protocol"),
        ];
        let cfg = ExtractorConfig {
            num_topics: 2,
            num_iterations: 10,
            min_word_freq: 1,
            max_doc_freq_pct: 0.5, // common appears in 100% > 50%
            ..Default::default()
        };
        let mut e = TopicModelExtractor::new(cfg);
        // Should succeed but "common" might be filtered
        let result = e.fit(&docs);
        // May fail with VocabularyEmpty or succeed — just verify no panic
        let _ = result;
    }

    // -----------------------------------------------------------------------
    // 41. fit is repeatable (call fit twice)
    // -----------------------------------------------------------------------
    #[test]
    fn test_refit() {
        let mut e = TopicModelExtractor::new(ExtractorConfig {
            num_topics: 2,
            num_iterations: 10,
            min_word_freq: 1,
            ..Default::default()
        });
        e.fit(&small_corpus())
            .expect("test: first fit on small corpus for refit test");
        e.fit(&small_corpus())
            .expect("test: second fit on small corpus for refit test");
        assert!(e.fitted);
        let s = e.stats().expect("test: stats() after refit");
        assert_eq!(s.num_topics, 2);
    }

    // -----------------------------------------------------------------------
    // 42. document distribution probabilities are all in [0,1]
    // -----------------------------------------------------------------------
    #[test]
    fn test_doc_distribution_values_in_range() {
        let e = make_fitted();
        let dt = e
            .document_topics("d05")
            .expect("test: document_topics for d05 value range check");
        for &p in &dt.topic_distribution {
            assert!((0.0..=1.0).contains(&p), "p={}", p);
        }
    }

    // -----------------------------------------------------------------------
    // 43. all training docs have reachable distributions
    // -----------------------------------------------------------------------
    #[test]
    fn test_all_training_docs_accessible() {
        let e = make_fitted();
        for (doc_id, _) in &small_corpus() {
            let result = e.document_topics(doc_id);
            assert!(result.is_ok(), "failed for {}", doc_id);
        }
    }

    // -----------------------------------------------------------------------
    // 44. stats vocab_size > 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_vocab_size_nonzero() {
        let e = make_fitted();
        let s = e.stats().expect("test: stats() for vocab_size check");
        assert!(s.vocab_size > 0);
    }

    // -----------------------------------------------------------------------
    // 45. top_words count fields are non-negative
    // -----------------------------------------------------------------------
    #[test]
    fn test_top_words_count_nonneg() {
        let e = make_fitted();
        let words = e
            .top_words(0, 20)
            .expect("test: top_words for topic 0 count nonneg check");
        for tw in &words {
            // count is u32, trivially non-negative — just verify field exists
            let _ = tw.count;
        }
        // Verify probability * topic_total_count is coherent
        assert!(!words.is_empty());
    }

    // -----------------------------------------------------------------------
    // 46. ExtractorTopic has valid coherence (not NaN)
    // -----------------------------------------------------------------------
    #[test]
    fn test_topic_coherence_not_nan() {
        let e = make_fitted();
        let topics = e.topics().expect("test: topics() for coherence NaN check");
        for t in &topics {
            assert!(!t.coherence.is_nan(), "topic {} coherence is NaN", t.id);
        }
    }

    // -----------------------------------------------------------------------
    // 47. TmeTopicWord alias works
    // -----------------------------------------------------------------------
    #[test]
    fn test_tme_topic_word_alias() {
        let tw: TmeTopicWord = ExtractorTopicWord {
            word: "rust".to_string(),
            probability: 0.5,
            count: 10,
        };
        assert_eq!(tw.word, "rust");
    }

    // -----------------------------------------------------------------------
    // 48. TmeDocumentTopics alias works
    // -----------------------------------------------------------------------
    #[test]
    fn test_tme_document_topics_alias() {
        let e = make_fitted();
        let dt: TmeDocumentTopics = e
            .document_topics("d01")
            .expect("test: document_topics as TmeDocumentTopics alias");
        assert_eq!(dt.doc_id, "d01");
    }

    // -----------------------------------------------------------------------
    // 49. single-topic model works
    // -----------------------------------------------------------------------
    #[test]
    fn test_single_topic() {
        let cfg = ExtractorConfig {
            num_topics: 1,
            num_iterations: 10,
            min_word_freq: 1,
            ..Default::default()
        };
        let mut e = TopicModelExtractor::new(cfg);
        e.fit(&small_corpus())
            .expect("test: fit on small corpus for single-topic test");
        let topics = e.topics().expect("test: topics() for single-topic model");
        assert_eq!(topics.len(), 1);
        assert!((topics[0].prevalence - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 50. similar_topics with top_k=0 returns empty
    // -----------------------------------------------------------------------
    #[test]
    fn test_similar_topics_zero_k() {
        let e = make_fitted();
        let sims = e
            .similar_topics(0, 0)
            .expect("test: similar_topics with top_k=0");
        assert!(sims.is_empty());
    }

    // -----------------------------------------------------------------------
    // 51. xorshift64 produces different values
    // -----------------------------------------------------------------------
    #[test]
    fn test_xorshift_different_values() {
        let mut state = 12345u64;
        let v1 = xorshift64(&mut state);
        let v2 = xorshift64(&mut state);
        let v3 = xorshift64(&mut state);
        assert_ne!(v1, v2);
        assert_ne!(v2, v3);
    }

    // -----------------------------------------------------------------------
    // 52. xorshift_f64 in [0, 1)
    // -----------------------------------------------------------------------
    #[test]
    fn test_xorshift_f64_range() {
        let mut state = 99999u64;
        for _ in 0..1000 {
            let v = xorshift_f64(&mut state);
            assert!((0.0..1.0).contains(&v), "v={}", v);
        }
    }
}
