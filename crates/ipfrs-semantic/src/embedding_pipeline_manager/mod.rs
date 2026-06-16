//! Embedding Pipeline Manager — multi-stage text-to-vector transformation engine.
//!
//! Transforms raw text or pre-computed embeddings through an ordered sequence of
//! configurable stages into normalised, searchable embedding vectors.  Each stage
//! is independently timed so bottlenecks can be identified via [`EpmPipelineStats`].
//!
//! ## Naming note
//! Several type names in this module are prefixed with `Epm` to avoid collision with
//! identically-named types already exported by `ipfrs_semantic` from other sub-modules
//! (`embedding_pipeline`, `query_pipeline`, `dimension_reducer`).

mod epm_types;
pub use epm_types::*;

mod epm_processing;
use epm_processing::{
    add_positional_encoding, apply_ngram, apply_stop_word_filter, build_vocab,
    inverse_document_frequencies, quantize_to_byte, reduce_dimensions, term_frequencies,
    tokenize_text, CorpusStats, PipelineState,
};
pub use epm_processing::{l2_normalize, mean_pool, random_projection};

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ---------------------------------------------------------------------------
// EmbeddingPipelineManager
// ---------------------------------------------------------------------------

/// Multi-stage embedding pipeline manager.
///
/// Supports both text-input (`process_text`) and pre-computed embedding input
/// (`process_embeddings`).  Thread-safe — the internal statistics are protected
/// by a `Mutex`.
pub struct EmbeddingPipelineManager {
    config: EpmPipelineConfig,
    state: Arc<Mutex<PipelineState>>,
}

impl EmbeddingPipelineManager {
    /// Create a new manager from the given config.
    pub fn new(config: EpmPipelineConfig) -> Result<Self, EpmPipelineError> {
        let mgr = Self {
            config,
            state: Arc::new(Mutex::new(PipelineState::default())),
        };
        mgr.validate_config()?;
        Ok(mgr)
    }

    /// Process a batch of raw text strings through all pipeline stages.
    ///
    /// `corpus` provides additional documents for IDF estimation; when `None`
    /// the current batch is used as the corpus.
    pub fn process_text(
        &self,
        ids: Vec<String>,
        texts: Vec<String>,
        corpus: Option<&[String]>,
    ) -> Result<EmbeddingBatch, EpmPipelineError> {
        if ids.is_empty() || texts.is_empty() {
            return Err(EpmPipelineError::EmptyInput);
        }
        if ids.len() != texts.len() {
            return Err(EpmPipelineError::InvalidConfig(format!(
                "ids.len() ({}) != texts.len() ({})",
                ids.len(),
                texts.len()
            )));
        }

        let batch_start = Instant::now();
        let n = texts.len();

        // Build combined corpus for IDF if provided.
        let idf_corpus: Vec<String> = match corpus {
            Some(c) => c.to_vec(),
            None => texts.clone(),
        };

        // --- Execute text stages ---
        // Step 1: Tokenize all texts (apply Tokenize, StopWordFilter, NGram sequentially).
        let mut token_lists: Vec<Vec<String>> = texts.iter().map(|t| vec![t.clone()]).collect();

        // Track stage timings.
        let mut stage_state = self
            .state
            .lock()
            .map_err(|e| EpmPipelineError::ProcessingFailed(format!("mutex poisoned: {e}")))?;

        // Run each stage.
        let mut embeddings_opt: Option<Vec<Vec<f64>>> = None;

        for stage in &self.config.stages {
            let stage_start = Instant::now();

            match stage {
                EpmPipelineStage::Tokenize {
                    lowercase,
                    strip_punct,
                } => {
                    token_lists = texts
                        .iter()
                        .map(|t| tokenize_text(t, *lowercase, *strip_punct))
                        .collect();
                }
                EpmPipelineStage::StopWordFilter(stop_words) => {
                    token_lists = token_lists
                        .into_iter()
                        .map(|toks| apply_stop_word_filter(toks, stop_words))
                        .collect();
                }
                EpmPipelineStage::NGram { n } => {
                    token_lists = token_lists
                        .iter()
                        .map(|toks| apply_ngram(toks, *n))
                        .collect();
                }
                EpmPipelineStage::TfIdfWeighting => {
                    // Build IDF from corpus.
                    let corpus_tokens: Vec<Vec<String>> = idf_corpus
                        .iter()
                        .map(|t| tokenize_text(t, true, true))
                        .collect();
                    let idf = inverse_document_frequencies(&corpus_tokens);
                    let tf_maps: Vec<HashMap<String, f64>> = token_lists
                        .iter()
                        .map(|toks| term_frequencies(toks))
                        .collect();
                    let vocab = build_vocab(&tf_maps);
                    if vocab.is_empty() {
                        return Err(EpmPipelineError::StageError {
                            stage: "TfIdfWeighting".to_string(),
                            reason: "empty vocabulary".to_string(),
                        });
                    }
                    embeddings_opt = Some(
                        tf_maps
                            .iter()
                            .map(|tf| epm_processing::tfidf_vector(tf, &idf, &vocab))
                            .collect(),
                    );
                }
                // Numeric stages — applied to embeddings.
                EpmPipelineStage::L2Normalize => {
                    let embs = embeddings_opt.get_or_insert_with(|| {
                        token_lists
                            .iter()
                            .map(|toks| toks.iter().map(|_| 1.0_f64).collect())
                            .collect()
                    });
                    for v in embs.iter_mut() {
                        l2_normalize(v);
                    }
                }
                EpmPipelineStage::DimensionReduce { target_dim, method } => {
                    let embs = embeddings_opt.get_or_insert_with(|| {
                        token_lists
                            .iter()
                            .map(|toks| toks.iter().map(|_| 1.0_f64).collect())
                            .collect()
                    });
                    let stats = CorpusStats::from_embeddings(embs);
                    let reduced: Vec<Vec<f64>> = embs
                        .iter()
                        .map(|v| reduce_dimensions(v, *target_dim, method, stats.as_ref()))
                        .collect();
                    *embs = reduced;
                }
                EpmPipelineStage::QuantizeToByte => {
                    let embs = embeddings_opt.get_or_insert_with(|| {
                        token_lists
                            .iter()
                            .map(|toks| toks.iter().map(|_| 1.0_f64).collect())
                            .collect()
                    });
                    for v in embs.iter_mut() {
                        *v = quantize_to_byte(v);
                    }
                }
                EpmPipelineStage::AddPositionalEncoding { max_len } => {
                    let embs = embeddings_opt.get_or_insert_with(|| {
                        token_lists
                            .iter()
                            .map(|toks| toks.iter().map(|_| 1.0_f64).collect())
                            .collect()
                    });
                    for (pos, v) in embs.iter_mut().enumerate() {
                        add_positional_encoding(v, pos, *max_len);
                    }
                }
            }

            let elapsed_us = stage_start.elapsed().as_micros() as u64;
            stage_state.record_stage(stage.name(), elapsed_us, n);
        }

        // If no TfIdfWeighting stage ran, materialise a simple token-count bag-of-words.
        let output_embeddings = match embeddings_opt {
            Some(e) => e,
            None => {
                // Build bag-of-words from token lists.
                let tf_maps: Vec<HashMap<String, f64>> = token_lists
                    .iter()
                    .map(|toks| term_frequencies(toks))
                    .collect();
                let vocab = build_vocab(&tf_maps);
                if vocab.is_empty() {
                    // Return unit vectors as fallback.
                    vec![vec![1.0_f64]; n]
                } else {
                    tf_maps
                        .iter()
                        .map(|tf| {
                            vocab
                                .iter()
                                .map(|term| tf.get(term).copied().unwrap_or(0.0))
                                .collect()
                        })
                        .collect()
                }
            }
        };

        let batch_us = batch_start.elapsed().as_micros() as u64;
        stage_state.record_batch(n, batch_us);

        Ok(EmbeddingBatch {
            ids,
            texts: Some(texts),
            raw_embeddings: None,
            output_embeddings,
            processing_time_us: batch_us,
        })
    }

    /// Process pre-computed embeddings through the non-text pipeline stages.
    pub fn process_embeddings(
        &self,
        ids: Vec<String>,
        embeddings: Vec<Vec<f64>>,
    ) -> Result<EmbeddingBatch, EpmPipelineError> {
        if ids.is_empty() || embeddings.is_empty() {
            return Err(EpmPipelineError::EmptyInput);
        }
        if ids.len() != embeddings.len() {
            return Err(EpmPipelineError::InvalidConfig(format!(
                "ids.len() ({}) != embeddings.len() ({})",
                ids.len(),
                embeddings.len()
            )));
        }

        let batch_start = Instant::now();
        let n = embeddings.len();
        let raw = embeddings.clone();

        let mut embs = embeddings;

        let mut stage_state = self
            .state
            .lock()
            .map_err(|e| EpmPipelineError::ProcessingFailed(format!("mutex poisoned: {e}")))?;

        for stage in &self.config.stages {
            if stage.requires_tokens() {
                // Skip text-only stages when processing pre-computed embeddings.
                continue;
            }
            let stage_start = Instant::now();

            match stage {
                EpmPipelineStage::L2Normalize => {
                    for v in embs.iter_mut() {
                        l2_normalize(v);
                    }
                }
                EpmPipelineStage::DimensionReduce { target_dim, method } => {
                    let stats = CorpusStats::from_embeddings(&embs);
                    let reduced: Vec<Vec<f64>> = embs
                        .iter()
                        .map(|v| reduce_dimensions(v, *target_dim, method, stats.as_ref()))
                        .collect();
                    embs = reduced;
                }
                EpmPipelineStage::QuantizeToByte => {
                    for v in embs.iter_mut() {
                        *v = quantize_to_byte(v);
                    }
                }
                EpmPipelineStage::AddPositionalEncoding { max_len } => {
                    for (pos, v) in embs.iter_mut().enumerate() {
                        add_positional_encoding(v, pos, *max_len);
                    }
                }
                _ => {} // text-only stages already skipped above
            }

            let elapsed_us = stage_start.elapsed().as_micros() as u64;
            stage_state.record_stage(stage.name(), elapsed_us, n);
        }

        let batch_us = batch_start.elapsed().as_micros() as u64;
        stage_state.record_batch(n, batch_us);

        Ok(EmbeddingBatch {
            ids,
            texts: None,
            raw_embeddings: Some(raw),
            output_embeddings: embs,
            processing_time_us: batch_us,
        })
    }

    /// Append a stage to the pipeline, then re-validate.
    pub fn add_stage(&mut self, stage: EpmPipelineStage) -> Result<(), EpmPipelineError> {
        self.config.stages.push(stage);
        self.validate_config()
    }

    /// Remove the stage at `index`.
    pub fn remove_stage(&mut self, index: usize) -> Result<(), EpmPipelineError> {
        if index >= self.config.stages.len() {
            return Err(EpmPipelineError::InvalidConfig(format!(
                "stage index {index} out of range (pipeline has {} stages)",
                self.config.stages.len()
            )));
        }
        self.config.stages.remove(index);
        Ok(())
    }

    /// Validate the current stage configuration for ordering correctness.
    ///
    /// Rules enforced:
    /// - `TfIdfWeighting` must be preceded by at least one of `Tokenize`,
    ///   `StopWordFilter`, or `NGram`.
    /// - `DimensionReduce` with `target_dim == 0` is rejected.
    /// - `NGram` with `n == 0` is rejected.
    pub fn validate_config(&self) -> Result<(), EpmPipelineError> {
        let mut seen_tokenize = false;
        for stage in &self.config.stages {
            match stage {
                EpmPipelineStage::Tokenize { .. } => {
                    seen_tokenize = true;
                }
                EpmPipelineStage::NGram { n } => {
                    if *n == 0 {
                        return Err(EpmPipelineError::InvalidConfig(
                            "NGram n must be >= 1".to_string(),
                        ));
                    }
                }
                EpmPipelineStage::StopWordFilter(_) => {
                    // Valid at any position.
                }
                EpmPipelineStage::TfIdfWeighting => {
                    if !seen_tokenize {
                        return Err(EpmPipelineError::InvalidConfig(
                            "TfIdfWeighting must be preceded by a Tokenize stage".to_string(),
                        ));
                    }
                }
                EpmPipelineStage::DimensionReduce { target_dim, .. } => {
                    if *target_dim == 0 {
                        return Err(EpmPipelineError::InvalidConfig(
                            "DimensionReduce target_dim must be > 0".to_string(),
                        ));
                    }
                }
                EpmPipelineStage::L2Normalize
                | EpmPipelineStage::QuantizeToByte
                | EpmPipelineStage::AddPositionalEncoding { .. } => {}
            }
        }
        if self.config.output_dim == 0 {
            return Err(EpmPipelineError::InvalidConfig(
                "output_dim must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Run each stage independently on `texts` for `n_runs` repetitions and
    /// return per-stage timing information.
    pub fn benchmark(&self, texts: &[String], n_runs: usize) -> Vec<StageTiming> {
        if texts.is_empty() || n_runs == 0 {
            return vec![];
        }
        let mut timings: Vec<StageTiming> = Vec::new();

        for stage in &self.config.stages {
            let name = stage.name().to_string();
            let mut total_us: u64 = 0;

            // Build a minimal token list for benchmarking text stages.
            let token_lists: Vec<Vec<String>> =
                texts.iter().map(|t| tokenize_text(t, true, true)).collect();
            let tf_maps: Vec<HashMap<String, f64>> = token_lists
                .iter()
                .map(|toks| term_frequencies(toks))
                .collect();
            let vocab = build_vocab(&tf_maps);
            let idf = inverse_document_frequencies(&token_lists);
            let base_embeddings: Vec<Vec<f64>> = tf_maps
                .iter()
                .map(|tf| epm_processing::tfidf_vector(tf, &idf, &vocab))
                .collect();

            for _ in 0..n_runs {
                let start = Instant::now();
                match stage {
                    EpmPipelineStage::Tokenize {
                        lowercase,
                        strip_punct,
                    } => {
                        for t in texts {
                            let _ = tokenize_text(t, *lowercase, *strip_punct);
                        }
                    }
                    EpmPipelineStage::StopWordFilter(sw) => {
                        for toks in &token_lists {
                            let _ = apply_stop_word_filter(toks.clone(), sw);
                        }
                    }
                    EpmPipelineStage::NGram { n } => {
                        for toks in &token_lists {
                            let _ = apply_ngram(toks, *n);
                        }
                    }
                    EpmPipelineStage::TfIdfWeighting => {
                        let corpus_tokens: Vec<Vec<String>> =
                            texts.iter().map(|t| tokenize_text(t, true, true)).collect();
                        let idf_b = inverse_document_frequencies(&corpus_tokens);
                        let tf_b: Vec<HashMap<String, f64>> = token_lists
                            .iter()
                            .map(|toks| term_frequencies(toks))
                            .collect();
                        let vocab_b = build_vocab(&tf_b);
                        for tf in &tf_b {
                            let _ = epm_processing::tfidf_vector(tf, &idf_b, &vocab_b);
                        }
                    }
                    EpmPipelineStage::L2Normalize => {
                        let mut embs = base_embeddings.clone();
                        for v in embs.iter_mut() {
                            l2_normalize(v);
                        }
                    }
                    EpmPipelineStage::DimensionReduce { target_dim, method } => {
                        let stats = CorpusStats::from_embeddings(&base_embeddings);
                        for v in &base_embeddings {
                            let _ = reduce_dimensions(v, *target_dim, method, stats.as_ref());
                        }
                    }
                    EpmPipelineStage::QuantizeToByte => {
                        for v in &base_embeddings {
                            let _ = quantize_to_byte(v);
                        }
                    }
                    EpmPipelineStage::AddPositionalEncoding { max_len } => {
                        let mut embs = base_embeddings.clone();
                        for (pos, v) in embs.iter_mut().enumerate() {
                            add_positional_encoding(v, pos, *max_len);
                        }
                    }
                }
                total_us += start.elapsed().as_micros() as u64;
            }

            let avg_time_us = total_us as f64 / n_runs as f64;
            timings.push(StageTiming {
                stage_name: name,
                avg_time_us,
                total_processed: (texts.len() * n_runs) as u64,
            });
        }
        timings
    }

    /// Return a snapshot of cumulative statistics.
    pub fn stats(&self) -> EpmPipelineStats {
        let state = match self.state.lock() {
            Ok(s) => s,
            Err(e) => e.into_inner(),
        };
        let stage_timings = state
            .stage_time
            .iter()
            .map(|(name, (total_us, total_processed))| StageTiming {
                stage_name: name.clone(),
                avg_time_us: if *total_processed > 0 {
                    *total_us as f64 / *total_processed as f64
                } else {
                    0.0
                },
                total_processed: *total_processed,
            })
            .collect();

        EpmPipelineStats {
            batches_processed: state.batches_processed,
            total_inputs: state.total_inputs,
            avg_batch_time_us: state.avg_batch_time_us(),
            stage_timings,
            output_dim: self.config.output_dim,
        }
    }

    /// Immutable reference to the current config.
    pub fn config(&self) -> &EpmPipelineConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::epm_processing::{
        add_positional_encoding, apply_ngram, apply_stop_word_filter, quantize_to_byte,
        tokenize_text, xorshift64,
    };
    use super::*;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn make_manager(stages: Vec<EpmPipelineStage>) -> EmbeddingPipelineManager {
        let mut config = EpmPipelineConfig::new(32, 4);
        config.stages = stages;
        EmbeddingPipelineManager::new(config).expect("valid config")
    }

    fn text_ids(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("doc{i}")).collect()
    }

    fn sample_texts() -> Vec<String> {
        vec![
            "the quick brown fox jumps over the lazy dog".to_string(),
            "a fast red cat leaps over a sleepy hound".to_string(),
            "rust programming language is fast and safe".to_string(),
        ]
    }

    fn sample_embeddings(n: usize, dim: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| (0..dim).map(|j| (i * dim + j) as f64 / 100.0).collect())
            .collect()
    }

    // ------------------------------------------------------------------
    // xorshift64 PRNG
    // ------------------------------------------------------------------

    #[test]
    fn test_xorshift64_nonzero() {
        let mut state: u64 = 42;
        let v = xorshift64(&mut state);
        assert_ne!(v, 42);
    }

    #[test]
    fn test_xorshift64_sequence_differs() {
        let mut state: u64 = 1;
        let a = xorshift64(&mut state);
        let b = xorshift64(&mut state);
        assert_ne!(a, b);
    }

    #[test]
    fn test_xorshift64_deterministic() {
        let mut s1 = 99u64;
        let mut s2 = 99u64;
        let a: Vec<u64> = (0..10).map(|_| xorshift64(&mut s1)).collect();
        let b: Vec<u64> = (0..10).map(|_| xorshift64(&mut s2)).collect();
        assert_eq!(a, b);
    }

    // ------------------------------------------------------------------
    // l2_normalize
    // ------------------------------------------------------------------

    #[test]
    fn test_l2_normalize_unit_result() {
        let mut v = vec![3.0_f64, 4.0];
        l2_normalize(&mut v);
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let mut v = vec![0.0_f64, 0.0, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_l2_normalize_single_element() {
        let mut v = vec![5.0_f64];
        l2_normalize(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // mean_pool
    // ------------------------------------------------------------------

    #[test]
    fn test_mean_pool_empty() {
        assert_eq!(mean_pool(&[]), Vec::<f64>::new());
    }

    #[test]
    fn test_mean_pool_single() {
        let v = vec![1.0, 2.0, 3.0];
        let result = mean_pool(std::slice::from_ref(&v));
        assert_eq!(result, v);
    }

    #[test]
    fn test_mean_pool_two_vectors() {
        let a = vec![1.0_f64, 2.0];
        let b = vec![3.0_f64, 4.0];
        let result = mean_pool(&[a, b]);
        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[1] - 3.0).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // random_projection
    // ------------------------------------------------------------------

    #[test]
    fn test_random_projection_output_dim() {
        let v: Vec<f64> = (0..128).map(|i| i as f64).collect();
        let out = random_projection(&v, 32, 42);
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn test_random_projection_deterministic() {
        let v: Vec<f64> = (0..64).map(|i| i as f64).collect();
        let a = random_projection(&v, 16, 7);
        let b = random_projection(&v, 16, 7);
        assert_eq!(a, b);
    }

    #[test]
    fn test_random_projection_different_seeds() {
        let v: Vec<f64> = (0..64).map(|i| i as f64 / 64.0).collect();
        let a = random_projection(&v, 16, 1);
        let b = random_projection(&v, 16, 2);
        // Different seeds must produce different results.
        assert_ne!(a, b);
    }

    #[test]
    fn test_random_projection_zero_target() {
        let v = vec![1.0_f64, 2.0];
        let out = random_projection(&v, 0, 1);
        assert_eq!(out.len(), 0);
    }

    // ------------------------------------------------------------------
    // Tokenize stage
    // ------------------------------------------------------------------

    #[test]
    fn test_stage_tokenize_lowercase() {
        let mgr = make_manager(vec![EpmPipelineStage::Tokenize {
            lowercase: true,
            strip_punct: false,
        }]);
        let batch = mgr
            .process_text(text_ids(1), vec!["Hello World".to_string()], None)
            .expect("test: tokenize lowercase stage");
        assert_eq!(batch.output_embeddings.len(), 1);
    }

    #[test]
    fn test_stage_tokenize_strip_punct() {
        let tokens = tokenize_text("hello, world!", false, true);
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(!tokens.iter().any(|t| t.contains(',')));
    }

    #[test]
    fn test_stage_tokenize_no_lowercase() {
        let tokens = tokenize_text("Hello World", false, false);
        assert!(tokens.contains(&"Hello".to_string()));
    }

    // ------------------------------------------------------------------
    // StopWordFilter stage
    // ------------------------------------------------------------------

    #[test]
    fn test_stage_stop_word_filter_removes_words() {
        let stop_words = vec!["the".to_string(), "a".to_string(), "an".to_string()];
        let tokens = vec!["the".to_string(), "quick".to_string(), "fox".to_string()];
        let filtered = apply_stop_word_filter(tokens, &stop_words);
        assert!(!filtered.contains(&"the".to_string()));
        assert!(filtered.contains(&"quick".to_string()));
    }

    #[test]
    fn test_stage_stop_word_filter_pipeline() {
        let stop_words = vec!["the".to_string(), "over".to_string()];
        let mgr = make_manager(vec![
            EpmPipelineStage::Tokenize {
                lowercase: true,
                strip_punct: false,
            },
            EpmPipelineStage::StopWordFilter(stop_words),
        ]);
        let batch = mgr
            .process_text(text_ids(1), vec!["the fox jumps over".to_string()], None)
            .expect("test: stop word filter pipeline");
        assert_eq!(batch.output_embeddings.len(), 1);
    }

    // ------------------------------------------------------------------
    // NGram stage
    // ------------------------------------------------------------------

    #[test]
    fn test_ngram_bigrams() {
        let tokens = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let bigrams = apply_ngram(&tokens, 2);
        assert_eq!(bigrams, vec!["a_b", "b_c"]);
    }

    #[test]
    fn test_ngram_trigrams() {
        let tokens: Vec<String> = vec!["a", "b", "c", "d"]
            .into_iter()
            .map(String::from)
            .collect();
        let trigrams = apply_ngram(&tokens, 3);
        assert_eq!(trigrams, vec!["a_b_c", "b_c_d"]);
    }

    #[test]
    fn test_ngram_unigram_passthrough() {
        let tokens: Vec<String> = vec!["a", "b"].into_iter().map(String::from).collect();
        let result = apply_ngram(&tokens, 1);
        assert_eq!(result, tokens);
    }

    #[test]
    fn test_ngram_too_few_tokens() {
        let tokens = vec!["only".to_string()];
        let result = apply_ngram(&tokens, 3);
        // Less than n tokens → return as-is.
        assert_eq!(result, tokens);
    }

    #[test]
    fn test_ngram_stage_pipeline() {
        let mgr = make_manager(vec![
            EpmPipelineStage::Tokenize {
                lowercase: true,
                strip_punct: false,
            },
            EpmPipelineStage::NGram { n: 2 },
        ]);
        let batch = mgr
            .process_text(
                text_ids(1),
                vec!["alpha beta gamma delta".to_string()],
                None,
            )
            .expect("test: ngram stage pipeline");
        assert_eq!(batch.output_embeddings.len(), 1);
    }

    // ------------------------------------------------------------------
    // TfIdfWeighting stage
    // ------------------------------------------------------------------

    #[test]
    fn test_tfidf_weighting_output_shape() {
        let mgr = make_manager(vec![
            EpmPipelineStage::Tokenize {
                lowercase: true,
                strip_punct: true,
            },
            EpmPipelineStage::TfIdfWeighting,
        ]);
        let texts = sample_texts();
        let n = texts.len();
        let batch = mgr
            .process_text(text_ids(n), texts, None)
            .expect("test: tfidf weighting output shape");
        assert_eq!(batch.output_embeddings.len(), n);
        // All vectors must have the same (vocab) length.
        let dim0 = batch.output_embeddings[0].len();
        for v in &batch.output_embeddings {
            assert_eq!(v.len(), dim0);
        }
    }

    #[test]
    fn test_tfidf_nonnegative_values() {
        let mgr = make_manager(vec![
            EpmPipelineStage::Tokenize {
                lowercase: true,
                strip_punct: true,
            },
            EpmPipelineStage::TfIdfWeighting,
        ]);
        let texts = sample_texts();
        let n = texts.len();
        let batch = mgr
            .process_text(text_ids(n), texts, None)
            .expect("test: tfidf nonnegative values");
        for v in &batch.output_embeddings {
            for &x in v {
                assert!(x >= 0.0, "TF-IDF value should be non-negative");
            }
        }
    }

    #[test]
    fn test_tfidf_with_external_corpus() {
        let corpus = vec!["rust language".to_string(), "python language".to_string()];
        let mgr = make_manager(vec![
            EpmPipelineStage::Tokenize {
                lowercase: true,
                strip_punct: false,
            },
            EpmPipelineStage::TfIdfWeighting,
        ]);
        let batch = mgr
            .process_text(
                text_ids(1),
                vec!["rust is great".to_string()],
                Some(&corpus),
            )
            .expect("test: tfidf with external corpus");
        assert_eq!(batch.output_embeddings.len(), 1);
    }

    // ------------------------------------------------------------------
    // L2Normalize stage
    // ------------------------------------------------------------------

    #[test]
    fn test_pipeline_l2_normalize() {
        let mgr = make_manager(vec![
            EpmPipelineStage::Tokenize {
                lowercase: true,
                strip_punct: true,
            },
            EpmPipelineStage::TfIdfWeighting,
            EpmPipelineStage::L2Normalize,
        ]);
        let texts = sample_texts();
        let n = texts.len();
        let batch = mgr
            .process_text(text_ids(n), texts, None)
            .expect("test: pipeline l2 normalize");
        for v in &batch.output_embeddings {
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-9 || norm < 1e-10, "norm={norm}");
        }
    }

    // ------------------------------------------------------------------
    // DimensionReduce stage (each method)
    // ------------------------------------------------------------------

    #[test]
    fn test_dimension_reduce_truncate() {
        let mgr = make_manager(vec![
            EpmPipelineStage::Tokenize {
                lowercase: true,
                strip_punct: true,
            },
            EpmPipelineStage::TfIdfWeighting,
            EpmPipelineStage::DimensionReduce {
                target_dim: 4,
                method: EpmReductionMethod::TruncateDims,
            },
        ]);
        let texts = sample_texts();
        let n = texts.len();
        let batch = mgr
            .process_text(text_ids(n), texts, None)
            .expect("test: dimension reduce truncate");
        for v in &batch.output_embeddings {
            assert_eq!(v.len(), 4);
        }
    }

    #[test]
    fn test_dimension_reduce_random_projection() {
        let mgr = make_manager(vec![
            EpmPipelineStage::Tokenize {
                lowercase: true,
                strip_punct: true,
            },
            EpmPipelineStage::TfIdfWeighting,
            EpmPipelineStage::DimensionReduce {
                target_dim: 8,
                method: EpmReductionMethod::RandomProjection(42),
            },
        ]);
        let texts = sample_texts();
        let n = texts.len();
        let batch = mgr
            .process_text(text_ids(n), texts, None)
            .expect("test: dimension reduce random projection");
        for v in &batch.output_embeddings {
            assert_eq!(v.len(), 8);
        }
    }

    #[test]
    fn test_dimension_reduce_mean_pooling() {
        let mgr = make_manager(vec![
            EpmPipelineStage::Tokenize {
                lowercase: true,
                strip_punct: true,
            },
            EpmPipelineStage::TfIdfWeighting,
            EpmPipelineStage::DimensionReduce {
                target_dim: 4,
                method: EpmReductionMethod::MeanPooling,
            },
        ]);
        let texts = sample_texts();
        let n = texts.len();
        let batch = mgr
            .process_text(text_ids(n), texts, None)
            .expect("test: dimension reduce mean pooling");
        for v in &batch.output_embeddings {
            assert_eq!(v.len(), 4);
        }
    }

    #[test]
    fn test_dimension_reduce_pca() {
        let mgr = make_manager(vec![
            EpmPipelineStage::Tokenize {
                lowercase: true,
                strip_punct: true,
            },
            EpmPipelineStage::TfIdfWeighting,
            EpmPipelineStage::DimensionReduce {
                target_dim: 4,
                method: EpmReductionMethod::PCA,
            },
        ]);
        let texts = sample_texts();
        let n = texts.len();
        let batch = mgr
            .process_text(text_ids(n), texts, None)
            .expect("test: dimension reduce pca");
        for v in &batch.output_embeddings {
            assert_eq!(v.len(), 4);
        }
    }

    // ------------------------------------------------------------------
    // QuantizeToByte stage
    // ------------------------------------------------------------------

    #[test]
    fn test_quantize_to_byte_range() {
        let v = vec![0.1_f64, 0.5, 1.0, -0.5, 2.0];
        let q = quantize_to_byte(&v);
        assert_eq!(q.len(), v.len());
        for &val in &q {
            assert!(
                (0.0..=255.0).contains(&val),
                "quantized value {val} out of [0,255]"
            );
        }
    }

    #[test]
    fn test_quantize_to_byte_constant_vector() {
        // All-same input → all zeros after quantization.
        let v = vec![3.0_f64; 8];
        let q = quantize_to_byte(&v);
        assert!(q.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_quantize_pipeline_stage() {
        let mgr = make_manager(vec![
            EpmPipelineStage::Tokenize {
                lowercase: true,
                strip_punct: true,
            },
            EpmPipelineStage::TfIdfWeighting,
            EpmPipelineStage::QuantizeToByte,
        ]);
        let texts = sample_texts();
        let n = texts.len();
        let batch = mgr
            .process_text(text_ids(n), texts, None)
            .expect("test: quantize pipeline stage");
        for v in &batch.output_embeddings {
            for &val in v {
                assert!((0.0..=255.0).contains(&val));
            }
        }
    }

    // ------------------------------------------------------------------
    // AddPositionalEncoding stage
    // ------------------------------------------------------------------

    #[test]
    fn test_positional_encoding_changes_vector() {
        let mut v = vec![1.0_f64; 16];
        let original = v.clone();
        add_positional_encoding(&mut v, 0, 512);
        assert_ne!(v, original);
    }

    #[test]
    fn test_positional_encoding_different_positions() {
        let mut a = vec![0.0_f64; 8];
        let mut b = vec![0.0_f64; 8];
        add_positional_encoding(&mut a, 0, 100);
        add_positional_encoding(&mut b, 1, 100);
        assert_ne!(a, b);
    }

    #[test]
    fn test_positional_encoding_pipeline_stage() {
        let mgr = make_manager(vec![
            EpmPipelineStage::Tokenize {
                lowercase: true,
                strip_punct: true,
            },
            EpmPipelineStage::TfIdfWeighting,
            EpmPipelineStage::AddPositionalEncoding { max_len: 128 },
        ]);
        let texts = sample_texts();
        let n = texts.len();
        let batch = mgr
            .process_text(text_ids(n), texts, None)
            .expect("test: process text positional encoding stage");
        assert_eq!(batch.output_embeddings.len(), n);
    }

    // ------------------------------------------------------------------
    // process_text — full roundtrip
    // ------------------------------------------------------------------

    #[test]
    fn test_process_text_roundtrip() {
        let mgr = make_manager(vec![
            EpmPipelineStage::Tokenize {
                lowercase: true,
                strip_punct: true,
            },
            EpmPipelineStage::StopWordFilter(vec!["the".to_string(), "a".to_string()]),
            EpmPipelineStage::NGram { n: 2 },
            EpmPipelineStage::TfIdfWeighting,
            EpmPipelineStage::L2Normalize,
            EpmPipelineStage::DimensionReduce {
                target_dim: 8,
                method: EpmReductionMethod::RandomProjection(1337),
            },
        ]);
        let texts = sample_texts();
        let n = texts.len();
        let batch = mgr
            .process_text(text_ids(n), texts.clone(), None)
            .expect("test: process text roundtrip");
        assert_eq!(batch.ids.len(), n);
        assert_eq!(batch.output_embeddings.len(), n);
        assert!(batch.texts.is_some());
        assert_eq!(batch.processing_time_us, batch.processing_time_us); // always true
    }

    #[test]
    fn test_process_text_preserves_ids() {
        let ids = vec!["foo".to_string(), "bar".to_string(), "baz".to_string()];
        let texts = sample_texts();
        let mgr = make_manager(vec![EpmPipelineStage::Tokenize {
            lowercase: true,
            strip_punct: false,
        }]);
        let batch = mgr
            .process_text(ids.clone(), texts, None)
            .expect("test: process text preserves ids");
        assert_eq!(batch.ids, ids);
    }

    #[test]
    fn test_process_text_empty_error() {
        let mgr = make_manager(vec![]);
        let result = mgr.process_text(vec![], vec![], None);
        assert!(matches!(result, Err(EpmPipelineError::EmptyInput)));
    }

    #[test]
    fn test_process_text_mismatched_ids_error() {
        let mgr = make_manager(vec![EpmPipelineStage::Tokenize {
            lowercase: false,
            strip_punct: false,
        }]);
        let result = mgr.process_text(
            vec!["a".to_string()],
            vec!["hello".to_string(), "world".to_string()],
            None,
        );
        assert!(matches!(result, Err(EpmPipelineError::InvalidConfig(_))));
    }

    // ------------------------------------------------------------------
    // process_embeddings — full roundtrip
    // ------------------------------------------------------------------

    #[test]
    fn test_process_embeddings_roundtrip() {
        let mgr = make_manager(vec![
            EpmPipelineStage::L2Normalize,
            EpmPipelineStage::DimensionReduce {
                target_dim: 4,
                method: EpmReductionMethod::TruncateDims,
            },
        ]);
        let embs = sample_embeddings(3, 16);
        let batch = mgr
            .process_embeddings(text_ids(3), embs.clone())
            .expect("test: process embeddings roundtrip");
        assert_eq!(batch.ids.len(), 3);
        assert_eq!(batch.output_embeddings.len(), 3);
        for v in &batch.output_embeddings {
            assert_eq!(v.len(), 4);
        }
        assert!(batch.raw_embeddings.is_some());
        assert!(batch.texts.is_none());
    }

    #[test]
    fn test_process_embeddings_empty_error() {
        let mgr = make_manager(vec![]);
        let result = mgr.process_embeddings(vec![], vec![]);
        assert!(matches!(result, Err(EpmPipelineError::EmptyInput)));
    }

    #[test]
    fn test_process_embeddings_skips_text_stages() {
        // Text-only stages should be silently skipped.
        let mgr = make_manager(vec![
            EpmPipelineStage::Tokenize {
                lowercase: true,
                strip_punct: true,
            },
            EpmPipelineStage::L2Normalize,
        ]);
        let embs = sample_embeddings(2, 8);
        let batch = mgr
            .process_embeddings(text_ids(2), embs)
            .expect("test: process embeddings skips text stages");
        assert_eq!(batch.output_embeddings.len(), 2);
    }

    #[test]
    fn test_process_embeddings_l2_unit_norm() {
        let mgr = make_manager(vec![EpmPipelineStage::L2Normalize]);
        let embs = sample_embeddings(4, 8);
        let batch = mgr
            .process_embeddings(text_ids(4), embs)
            .expect("test: process embeddings l2 unit norm");
        for v in &batch.output_embeddings {
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-9 || norm < 1e-10);
        }
    }

    // ------------------------------------------------------------------
    // add_stage / remove_stage
    // ------------------------------------------------------------------

    #[test]
    fn test_add_stage_appends() {
        let mut mgr = make_manager(vec![]);
        mgr.add_stage(EpmPipelineStage::L2Normalize)
            .expect("test: add stage");
        assert_eq!(mgr.config().stages.len(), 1);
    }

    #[test]
    fn test_remove_stage_removes() {
        let mut mgr = make_manager(vec![
            EpmPipelineStage::L2Normalize,
            EpmPipelineStage::QuantizeToByte,
        ]);
        mgr.remove_stage(0).expect("test: remove stage");
        assert_eq!(mgr.config().stages.len(), 1);
        assert!(matches!(
            mgr.config().stages[0],
            EpmPipelineStage::QuantizeToByte
        ));
    }

    #[test]
    fn test_remove_stage_out_of_bounds() {
        let mut mgr = make_manager(vec![]);
        let result = mgr.remove_stage(5);
        assert!(matches!(result, Err(EpmPipelineError::InvalidConfig(_))));
    }

    // ------------------------------------------------------------------
    // validate_config
    // ------------------------------------------------------------------

    #[test]
    fn test_validate_tfidf_without_tokenize_fails() {
        let config = EpmPipelineConfig {
            stages: vec![EpmPipelineStage::TfIdfWeighting],
            output_dim: 32,
            batch_size: 4,
        };
        let result = EmbeddingPipelineManager::new(config);
        assert!(matches!(result, Err(EpmPipelineError::InvalidConfig(_))));
    }

    #[test]
    fn test_validate_zero_output_dim_fails() {
        let config = EpmPipelineConfig {
            stages: vec![],
            output_dim: 0,
            batch_size: 4,
        };
        let result = EmbeddingPipelineManager::new(config);
        assert!(matches!(result, Err(EpmPipelineError::InvalidConfig(_))));
    }

    #[test]
    fn test_validate_zero_target_dim_fails() {
        let config = EpmPipelineConfig {
            stages: vec![EpmPipelineStage::DimensionReduce {
                target_dim: 0,
                method: EpmReductionMethod::TruncateDims,
            }],
            output_dim: 32,
            batch_size: 4,
        };
        let result = EmbeddingPipelineManager::new(config);
        assert!(matches!(result, Err(EpmPipelineError::InvalidConfig(_))));
    }

    #[test]
    fn test_validate_zero_ngram_fails() {
        // n==0 is caught during add_stage (validate is re-run).
        let config = EpmPipelineConfig {
            stages: vec![EpmPipelineStage::NGram { n: 0 }],
            output_dim: 32,
            batch_size: 4,
        };
        let result = EmbeddingPipelineManager::new(config);
        assert!(matches!(result, Err(EpmPipelineError::InvalidConfig(_))));
    }

    #[test]
    fn test_validate_valid_config_ok() {
        let config = EpmPipelineConfig {
            stages: vec![
                EpmPipelineStage::Tokenize {
                    lowercase: true,
                    strip_punct: true,
                },
                EpmPipelineStage::TfIdfWeighting,
                EpmPipelineStage::L2Normalize,
            ],
            output_dim: 64,
            batch_size: 8,
        };
        assert!(EmbeddingPipelineManager::new(config).is_ok());
    }

    // ------------------------------------------------------------------
    // benchmark
    // ------------------------------------------------------------------

    #[test]
    fn test_benchmark_returns_timings() {
        let mgr = make_manager(vec![
            EpmPipelineStage::Tokenize {
                lowercase: true,
                strip_punct: true,
            },
            EpmPipelineStage::L2Normalize,
        ]);
        let texts = sample_texts();
        let timings = mgr.benchmark(&texts, 3);
        assert_eq!(timings.len(), 2);
        assert_eq!(timings[0].stage_name, "Tokenize");
        assert_eq!(timings[1].stage_name, "L2Normalize");
    }

    #[test]
    fn test_benchmark_empty_returns_empty() {
        let mgr = make_manager(vec![EpmPipelineStage::L2Normalize]);
        let timings = mgr.benchmark(&[], 5);
        assert!(timings.is_empty());
    }

    #[test]
    fn test_benchmark_zero_runs_returns_empty() {
        let mgr = make_manager(vec![EpmPipelineStage::L2Normalize]);
        let timings = mgr.benchmark(&sample_texts(), 0);
        assert!(timings.is_empty());
    }

    #[test]
    fn test_benchmark_nonnegative_times() {
        let mgr = make_manager(vec![
            EpmPipelineStage::Tokenize {
                lowercase: true,
                strip_punct: false,
            },
            EpmPipelineStage::TfIdfWeighting,
        ]);
        let timings = mgr.benchmark(&sample_texts(), 2);
        for t in &timings {
            assert!(t.avg_time_us >= 0.0);
        }
    }

    // ------------------------------------------------------------------
    // stats
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_initial_zero() {
        let mgr = make_manager(vec![]);
        let stats = mgr.stats();
        assert_eq!(stats.batches_processed, 0);
        assert_eq!(stats.total_inputs, 0);
    }

    #[test]
    fn test_stats_increments_after_process() {
        let mgr = make_manager(vec![EpmPipelineStage::L2Normalize]);
        let embs = sample_embeddings(3, 8);
        mgr.process_embeddings(text_ids(3), embs)
            .expect("test: process embeddings stats increment");
        let stats = mgr.stats();
        assert_eq!(stats.batches_processed, 1);
        assert_eq!(stats.total_inputs, 3);
    }

    #[test]
    fn test_stats_output_dim() {
        let mgr = make_manager(vec![]);
        assert_eq!(mgr.stats().output_dim, 32);
    }

    #[test]
    fn test_stats_multiple_batches() {
        let mgr = make_manager(vec![EpmPipelineStage::L2Normalize]);
        for _ in 0..5 {
            let embs = sample_embeddings(2, 4);
            mgr.process_embeddings(text_ids(2), embs)
                .expect("test: process embeddings multiple batches");
        }
        let stats = mgr.stats();
        assert_eq!(stats.batches_processed, 5);
        assert_eq!(stats.total_inputs, 10);
    }

    // ------------------------------------------------------------------
    // EpmPipelineError variants
    // ------------------------------------------------------------------

    #[test]
    fn test_error_display_empty_input() {
        let e = EpmPipelineError::EmptyInput;
        assert_eq!(e.to_string(), "empty input");
    }

    #[test]
    fn test_error_display_dimension() {
        let e = EpmPipelineError::DimensionError {
            expected: 128,
            got: 64,
        };
        assert!(e.to_string().contains("128"));
        assert!(e.to_string().contains("64"));
    }

    #[test]
    fn test_error_display_stage() {
        let e = EpmPipelineError::StageError {
            stage: "TfIdfWeighting".to_string(),
            reason: "empty vocabulary".to_string(),
        };
        assert!(e.to_string().contains("TfIdfWeighting"));
    }

    #[test]
    fn test_error_invalid_config_clone() {
        let e = EpmPipelineError::InvalidConfig("bad config".to_string());
        let cloned = e.clone();
        assert_eq!(e.to_string(), cloned.to_string());
    }

    // ------------------------------------------------------------------
    // Full pipeline integration (text + embeddings)
    // ------------------------------------------------------------------

    #[test]
    fn test_full_text_pipeline() {
        let config = EpmPipelineConfig {
            stages: vec![
                EpmPipelineStage::Tokenize {
                    lowercase: true,
                    strip_punct: true,
                },
                EpmPipelineStage::StopWordFilter(vec![
                    "the".to_string(),
                    "a".to_string(),
                    "is".to_string(),
                ]),
                EpmPipelineStage::NGram { n: 2 },
                EpmPipelineStage::TfIdfWeighting,
                EpmPipelineStage::L2Normalize,
                EpmPipelineStage::DimensionReduce {
                    target_dim: 16,
                    method: EpmReductionMethod::RandomProjection(99),
                },
                EpmPipelineStage::QuantizeToByte,
                EpmPipelineStage::AddPositionalEncoding { max_len: 256 },
            ],
            output_dim: 16,
            batch_size: 8,
        };
        let mgr = EmbeddingPipelineManager::new(config)
            .expect("test: create pipeline manager for full text pipeline");
        let texts = vec![
            "the quick brown fox".to_string(),
            "rust is a systems language".to_string(),
            "semantic search embeddings".to_string(),
            "machine learning vectors".to_string(),
        ];
        let n = texts.len();
        let batch = mgr
            .process_text(text_ids(n), texts, None)
            .expect("test: process text full pipeline");
        assert_eq!(batch.output_embeddings.len(), n);
        for v in &batch.output_embeddings {
            assert_eq!(v.len(), 16);
        }
    }

    #[test]
    fn test_full_embedding_pipeline() {
        let config = EpmPipelineConfig {
            stages: vec![
                EpmPipelineStage::L2Normalize,
                EpmPipelineStage::DimensionReduce {
                    target_dim: 8,
                    method: EpmReductionMethod::MeanPooling,
                },
                EpmPipelineStage::QuantizeToByte,
                EpmPipelineStage::AddPositionalEncoding { max_len: 64 },
            ],
            output_dim: 8,
            batch_size: 4,
        };
        let mgr = EmbeddingPipelineManager::new(config)
            .expect("test: create pipeline manager for full embedding pipeline");
        let embs = sample_embeddings(5, 32);
        let batch = mgr
            .process_embeddings(text_ids(5), embs)
            .expect("test: process embeddings full pipeline");
        assert_eq!(batch.output_embeddings.len(), 5);
        for v in &batch.output_embeddings {
            assert_eq!(v.len(), 8);
        }
    }
}
