//! # Semantic Context Window
//!
//! Maintains a sliding context window of recent semantic interactions
//! (queries + results), computing context embeddings for session-aware
//! search personalization.

/// A single entry in the semantic context window.
#[derive(Clone, Debug, PartialEq)]
pub enum ContextEntry {
    /// A user query with its embedding and timestamp tick.
    Query {
        /// The raw query text.
        text: String,
        /// The embedding vector for this query.
        embedding: Vec<f32>,
        /// Monotonic tick (session-relative timestamp).
        tick: u64,
    },
    /// A search result that was presented to the user.
    Result {
        /// Document identifier.
        doc_id: u64,
        /// The embedding vector for this result.
        embedding: Vec<f32>,
        /// Relevance score of this result (higher = more relevant).
        relevance: f32,
        /// Monotonic tick (session-relative timestamp).
        tick: u64,
    },
    /// Explicit feedback from the user about a document.
    Feedback {
        /// Document identifier.
        doc_id: u64,
        /// Whether feedback is positive (liked) or negative (disliked).
        positive: bool,
        /// Monotonic tick (session-relative timestamp).
        tick: u64,
    },
}

/// Configuration for a [`SemanticContextWindow`].
#[derive(Clone, Debug, PartialEq)]
pub struct WindowConfig {
    /// Maximum number of entries kept in the sliding window.
    pub max_entries: usize,
    /// Exponential decay weight applied to older entries (0 < decay_factor <= 1).
    pub decay_factor: f32,
    /// Expected dimensionality of embedding vectors.
    pub embedding_dim: usize,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            max_entries: 20,
            decay_factor: 0.9,
            embedding_dim: 128,
        }
    }
}

/// Snapshot statistics about the current context window state.
#[derive(Clone, Debug, PartialEq)]
pub struct ContextStats {
    /// Total number of entries ever added (including evicted ones).
    pub total_entries_added: u64,
    /// Number of [`ContextEntry::Query`] entries currently in the window.
    pub queries_in_window: usize,
    /// Number of [`ContextEntry::Result`] entries currently in the window.
    pub results_in_window: usize,
    /// Number of [`ContextEntry::Feedback`] entries currently in the window.
    pub feedbacks_in_window: usize,
    /// Ratio of positive feedback to total feedback entries in the window.
    /// Returns `0.0` when there are no feedback entries.
    pub positive_feedback_ratio: f32,
}

/// Sliding context window for session-aware semantic search personalization.
///
/// Maintains a bounded list of recent queries, search results, and user feedback.
/// Provides a weighted context embedding that captures recency-biased user intent.
pub struct SemanticContextWindow {
    /// Current window entries, oldest at index 0, newest at end.
    pub entries: Vec<ContextEntry>,
    /// Configuration for this window.
    pub config: WindowConfig,
    /// Monotonically increasing counter of all entries ever added.
    pub total_added: u64,
}

impl SemanticContextWindow {
    /// Creates a new context window with the given configuration.
    pub fn new(config: WindowConfig) -> Self {
        Self {
            entries: Vec::new(),
            config,
            total_added: 0,
        }
    }

    /// Adds an entry to the window.
    ///
    /// If the window is full (`len > max_entries`), the oldest entry is evicted.
    pub fn add(&mut self, entry: ContextEntry) {
        self.entries.push(entry);
        if self.entries.len() > self.config.max_entries {
            self.entries.remove(0);
        }
        self.total_added += 1;
    }

    /// Computes a recency-weighted average context embedding.
    ///
    /// Only [`ContextEntry::Query`] and [`ContextEntry::Result`] entries contribute
    /// embeddings. Entries with the wrong embedding dimension are skipped silently.
    ///
    /// The weight of entry at window index `i` (0 = oldest) is:
    /// `decay_factor ^ (window_len - 1 - i)`
    ///
    /// Returns a zero vector of length `embedding_dim` when no valid embeddings
    /// are present in the window.
    pub fn context_embedding(&self) -> Vec<f32> {
        let dim = self.config.embedding_dim;
        let decay = self.config.decay_factor;
        let window_len = self.entries.len();

        // Collect (index, embedding_ref) for Query and Result entries only.
        let embeddable: Vec<(usize, &Vec<f32>)> = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| match e {
                ContextEntry::Query { embedding, .. } => Some((i, embedding)),
                ContextEntry::Result { embedding, .. } => Some((i, embedding)),
                ContextEntry::Feedback { .. } => None,
            })
            .filter(|(_, emb)| emb.len() == dim)
            .collect();

        if embeddable.is_empty() {
            return vec![0.0_f32; dim];
        }

        // Compute weight for each entry based on its position in the full window.
        let weight_of = |idx: usize| -> f32 {
            let age = (window_len - 1).saturating_sub(idx); // 0 = newest
            decay.powi(age as i32)
        };

        let weight_sum: f32 = embeddable.iter().map(|(i, _)| weight_of(*i)).sum();

        let mut result = vec![0.0_f32; dim];
        for (i, emb) in &embeddable {
            let w = weight_of(*i) / weight_sum;
            for (r, v) in result.iter_mut().zip(emb.iter()) {
                *r += w * v;
            }
        }
        result
    }

    /// Returns the last `n` [`ContextEntry::Query`] entries, newest first.
    pub fn recent_queries(&self, n: usize) -> Vec<&ContextEntry> {
        self.entries
            .iter()
            .rev()
            .filter(|e| matches!(e, ContextEntry::Query { .. }))
            .take(n)
            .collect()
    }

    /// Returns the last `n` [`ContextEntry::Result`] entries, newest first.
    pub fn recent_results(&self, n: usize) -> Vec<&ContextEntry> {
        self.entries
            .iter()
            .rev()
            .filter(|e| matches!(e, ContextEntry::Result { .. }))
            .take(n)
            .collect()
    }

    /// Returns doc_ids with positive feedback, oldest first, deduplicated.
    pub fn positive_docs(&self) -> Vec<u64> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for e in &self.entries {
            if let ContextEntry::Feedback {
                doc_id,
                positive: true,
                ..
            } = e
            {
                if seen.insert(*doc_id) {
                    result.push(*doc_id);
                }
            }
        }
        result
    }

    /// Returns doc_ids with negative feedback, oldest first, deduplicated.
    pub fn negative_docs(&self) -> Vec<u64> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for e in &self.entries {
            if let ContextEntry::Feedback {
                doc_id,
                positive: false,
                ..
            } = e
            {
                if seen.insert(*doc_id) {
                    result.push(*doc_id);
                }
            }
        }
        result
    }

    /// Clears all entries from the window, preserving the configuration.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns a snapshot of current window statistics.
    pub fn stats(&self) -> ContextStats {
        let mut queries_in_window = 0usize;
        let mut results_in_window = 0usize;
        let mut feedbacks_in_window = 0usize;
        let mut positive_count = 0usize;

        for e in &self.entries {
            match e {
                ContextEntry::Query { .. } => queries_in_window += 1,
                ContextEntry::Result { .. } => results_in_window += 1,
                ContextEntry::Feedback { positive, .. } => {
                    feedbacks_in_window += 1;
                    if *positive {
                        positive_count += 1;
                    }
                }
            }
        }

        let positive_feedback_ratio = if feedbacks_in_window == 0 {
            0.0
        } else {
            positive_count as f32 / feedbacks_in_window as f32
        };

        ContextStats {
            total_entries_added: self.total_added,
            queries_in_window,
            results_in_window,
            feedbacks_in_window,
            positive_feedback_ratio,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(val: f32, dim: usize) -> Vec<f32> {
        vec![val; dim]
    }

    fn default_window() -> SemanticContextWindow {
        SemanticContextWindow::new(WindowConfig::default())
    }

    // -----------------------------------------------------------------------
    // add / sliding window eviction
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_single_entry() {
        let mut w = default_window();
        w.add(ContextEntry::Feedback {
            doc_id: 1,
            positive: true,
            tick: 0,
        });
        assert_eq!(w.entries.len(), 1);
        assert_eq!(w.total_added, 1);
    }

    #[test]
    fn test_sliding_window_evicts_oldest() {
        let config = WindowConfig {
            max_entries: 3,
            ..Default::default()
        };
        let mut w = SemanticContextWindow::new(config);
        for tick in 0..4u64 {
            w.add(ContextEntry::Feedback {
                doc_id: tick,
                positive: true,
                tick,
            });
        }
        // Window should contain ticks 1, 2, 3 (tick 0 was evicted).
        assert_eq!(w.entries.len(), 3);
        assert!(matches!(
            &w.entries[0],
            ContextEntry::Feedback { doc_id: 1, .. }
        ));
        assert!(matches!(
            &w.entries[2],
            ContextEntry::Feedback { doc_id: 3, .. }
        ));
    }

    #[test]
    fn test_total_added_counts_all_including_evicted() {
        let config = WindowConfig {
            max_entries: 2,
            ..Default::default()
        };
        let mut w = SemanticContextWindow::new(config);
        for tick in 0..5u64 {
            w.add(ContextEntry::Feedback {
                doc_id: tick,
                positive: false,
                tick,
            });
        }
        assert_eq!(w.total_added, 5);
        assert_eq!(w.entries.len(), 2);
    }

    #[test]
    fn test_add_up_to_capacity_no_eviction() {
        let config = WindowConfig {
            max_entries: 5,
            ..Default::default()
        };
        let mut w = SemanticContextWindow::new(config);
        for tick in 0..5u64 {
            w.add(ContextEntry::Feedback {
                doc_id: tick,
                positive: true,
                tick,
            });
        }
        assert_eq!(w.entries.len(), 5);
        assert_eq!(w.total_added, 5);
    }

    // -----------------------------------------------------------------------
    // context_embedding
    // -----------------------------------------------------------------------

    #[test]
    fn test_context_embedding_empty_returns_zeros() {
        let w = default_window();
        let emb = w.context_embedding();
        assert_eq!(emb.len(), 128);
        assert!(emb.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_context_embedding_only_feedback_returns_zeros() {
        let mut w = default_window();
        w.add(ContextEntry::Feedback {
            doc_id: 1,
            positive: true,
            tick: 0,
        });
        let emb = w.context_embedding();
        assert!(emb.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_context_embedding_single_query_returns_that_embedding() {
        let mut w = default_window();
        let emb_val = 0.5_f32;
        w.add(ContextEntry::Query {
            text: "hello".into(),
            embedding: make_embedding(emb_val, 128),
            tick: 0,
        });
        let result = w.context_embedding();
        assert_eq!(result.len(), 128);
        // Single entry has weight 1.0, so result == the embedding itself.
        for v in &result {
            assert!((v - emb_val).abs() < 1e-6, "expected {emb_val} got {v}");
        }
    }

    #[test]
    fn test_context_embedding_single_result_returns_that_embedding() {
        let mut w = default_window();
        let emb_val = 0.7_f32;
        w.add(ContextEntry::Result {
            doc_id: 42,
            embedding: make_embedding(emb_val, 128),
            relevance: 1.0,
            tick: 0,
        });
        let result = w.context_embedding();
        for v in &result {
            assert!((v - emb_val).abs() < 1e-6);
        }
    }

    #[test]
    fn test_context_embedding_newer_has_higher_weight() {
        // Two queries: old = [0.0; 128], new = [1.0; 128].
        // With decay 0.9 and window_len=2:
        //   weight_old = 0.9^1 = 0.9, weight_new = 0.9^0 = 1.0
        //   sum = 1.9
        //   result = (0.9*0 + 1.0*1) / 1.9 = 1/1.9 ≈ 0.5263
        let mut w = default_window();
        w.add(ContextEntry::Query {
            text: "old".into(),
            embedding: make_embedding(0.0, 128),
            tick: 0,
        });
        w.add(ContextEntry::Query {
            text: "new".into(),
            embedding: make_embedding(1.0, 128),
            tick: 1,
        });
        let result = w.context_embedding();
        let expected = 1.0_f32 / 1.9_f32;
        for v in &result {
            assert!((v - expected).abs() < 1e-5, "expected {expected} got {v}");
        }
    }

    #[test]
    fn test_context_embedding_skips_wrong_dimension() {
        let config = WindowConfig {
            embedding_dim: 4,
            ..Default::default()
        };
        let mut w = SemanticContextWindow::new(config);
        // This entry has dim=8 (wrong), should be skipped.
        w.add(ContextEntry::Query {
            text: "bad".into(),
            embedding: vec![99.0; 8],
            tick: 0,
        });
        // This entry has dim=4 (correct).
        w.add(ContextEntry::Query {
            text: "good".into(),
            embedding: vec![0.5; 4],
            tick: 1,
        });
        let result = w.context_embedding();
        // Only the good entry contributes; result should be [0.5; 4].
        assert_eq!(result.len(), 4);
        for v in &result {
            assert!((v - 0.5).abs() < 1e-6, "got {v}");
        }
    }

    #[test]
    fn test_context_embedding_all_wrong_dimension_returns_zeros() {
        let config = WindowConfig {
            embedding_dim: 4,
            ..Default::default()
        };
        let mut w = SemanticContextWindow::new(config);
        w.add(ContextEntry::Query {
            text: "bad".into(),
            embedding: vec![1.0; 8],
            tick: 0,
        });
        let result = w.context_embedding();
        assert_eq!(result.len(), 4);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_context_embedding_mixed_entries_feedback_ignored() {
        let config = WindowConfig {
            embedding_dim: 2,
            decay_factor: 1.0, // uniform weights for simplicity
            ..Default::default()
        };
        let mut w = SemanticContextWindow::new(config);
        w.add(ContextEntry::Feedback {
            doc_id: 1,
            positive: true,
            tick: 0,
        });
        w.add(ContextEntry::Query {
            text: "q".into(),
            embedding: vec![2.0, 4.0],
            tick: 1,
        });
        w.add(ContextEntry::Feedback {
            doc_id: 2,
            positive: false,
            tick: 2,
        });
        w.add(ContextEntry::Result {
            doc_id: 10,
            embedding: vec![6.0, 8.0],
            relevance: 0.9,
            tick: 3,
        });
        // Two valid embeddings, uniform weight => average of [2,4] and [6,8] = [4,6].
        let result = w.context_embedding();
        assert!((result[0] - 4.0).abs() < 1e-5, "got {}", result[0]);
        assert!((result[1] - 6.0).abs() < 1e-5, "got {}", result[1]);
    }

    // -----------------------------------------------------------------------
    // recent_queries
    // -----------------------------------------------------------------------

    #[test]
    fn test_recent_queries_newest_first() {
        let mut w = default_window();
        for tick in 0..4u64 {
            w.add(ContextEntry::Query {
                text: format!("q{tick}"),
                embedding: make_embedding(0.0, 128),
                tick,
            });
        }
        let recent = w.recent_queries(3);
        assert_eq!(recent.len(), 3);
        // Newest should be tick=3.
        assert!(matches!(recent[0], ContextEntry::Query { tick: 3, .. }));
        assert!(matches!(recent[1], ContextEntry::Query { tick: 2, .. }));
        assert!(matches!(recent[2], ContextEntry::Query { tick: 1, .. }));
    }

    #[test]
    fn test_recent_queries_fewer_than_n() {
        let mut w = default_window();
        w.add(ContextEntry::Query {
            text: "only".into(),
            embedding: make_embedding(0.1, 128),
            tick: 0,
        });
        let recent = w.recent_queries(5);
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_recent_queries_ignores_non_query_entries() {
        let mut w = default_window();
        w.add(ContextEntry::Feedback {
            doc_id: 1,
            positive: true,
            tick: 0,
        });
        w.add(ContextEntry::Query {
            text: "q".into(),
            embedding: make_embedding(0.0, 128),
            tick: 1,
        });
        let recent = w.recent_queries(10);
        assert_eq!(recent.len(), 1);
    }

    // -----------------------------------------------------------------------
    // recent_results
    // -----------------------------------------------------------------------

    #[test]
    fn test_recent_results_newest_first() {
        let mut w = default_window();
        for tick in 0..3u64 {
            w.add(ContextEntry::Result {
                doc_id: tick,
                embedding: make_embedding(0.0, 128),
                relevance: 0.5,
                tick,
            });
        }
        let recent = w.recent_results(2);
        assert_eq!(recent.len(), 2);
        assert!(matches!(recent[0], ContextEntry::Result { tick: 2, .. }));
        assert!(matches!(recent[1], ContextEntry::Result { tick: 1, .. }));
    }

    #[test]
    fn test_recent_results_ignores_non_result_entries() {
        let mut w = default_window();
        w.add(ContextEntry::Query {
            text: "q".into(),
            embedding: make_embedding(0.0, 128),
            tick: 0,
        });
        let recent = w.recent_results(10);
        assert_eq!(recent.len(), 0);
    }

    // -----------------------------------------------------------------------
    // positive_docs / negative_docs
    // -----------------------------------------------------------------------

    #[test]
    fn test_positive_docs_deduplicated_oldest_first() {
        let mut w = default_window();
        w.add(ContextEntry::Feedback {
            doc_id: 5,
            positive: true,
            tick: 0,
        });
        w.add(ContextEntry::Feedback {
            doc_id: 3,
            positive: true,
            tick: 1,
        });
        // Duplicate of doc_id=5
        w.add(ContextEntry::Feedback {
            doc_id: 5,
            positive: true,
            tick: 2,
        });
        let docs = w.positive_docs();
        // Doc 5 appears only once, in insertion order; 3 comes after.
        assert_eq!(docs, vec![5, 3]);
    }

    #[test]
    fn test_positive_docs_excludes_negatives() {
        let mut w = default_window();
        w.add(ContextEntry::Feedback {
            doc_id: 1,
            positive: false,
            tick: 0,
        });
        w.add(ContextEntry::Feedback {
            doc_id: 2,
            positive: true,
            tick: 1,
        });
        let docs = w.positive_docs();
        assert_eq!(docs, vec![2]);
    }

    #[test]
    fn test_negative_docs_deduplicated_oldest_first() {
        let mut w = default_window();
        w.add(ContextEntry::Feedback {
            doc_id: 7,
            positive: false,
            tick: 0,
        });
        w.add(ContextEntry::Feedback {
            doc_id: 8,
            positive: false,
            tick: 1,
        });
        w.add(ContextEntry::Feedback {
            doc_id: 7,
            positive: false,
            tick: 2,
        });
        let docs = w.negative_docs();
        assert_eq!(docs, vec![7, 8]);
    }

    #[test]
    fn test_negative_docs_excludes_positives() {
        let mut w = default_window();
        w.add(ContextEntry::Feedback {
            doc_id: 1,
            positive: true,
            tick: 0,
        });
        let docs = w.negative_docs();
        assert!(docs.is_empty());
    }

    // -----------------------------------------------------------------------
    // stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_counts() {
        let mut w = default_window();
        w.add(ContextEntry::Query {
            text: "a".into(),
            embedding: make_embedding(0.0, 128),
            tick: 0,
        });
        w.add(ContextEntry::Query {
            text: "b".into(),
            embedding: make_embedding(0.0, 128),
            tick: 1,
        });
        w.add(ContextEntry::Result {
            doc_id: 1,
            embedding: make_embedding(0.0, 128),
            relevance: 0.8,
            tick: 2,
        });
        w.add(ContextEntry::Feedback {
            doc_id: 1,
            positive: true,
            tick: 3,
        });
        w.add(ContextEntry::Feedback {
            doc_id: 2,
            positive: false,
            tick: 4,
        });
        let s = w.stats();
        assert_eq!(s.total_entries_added, 5);
        assert_eq!(s.queries_in_window, 2);
        assert_eq!(s.results_in_window, 1);
        assert_eq!(s.feedbacks_in_window, 2);
    }

    #[test]
    fn test_stats_positive_feedback_ratio() {
        let mut w = default_window();
        // 3 positive, 1 negative => ratio = 0.75
        for i in 0..3u64 {
            w.add(ContextEntry::Feedback {
                doc_id: i,
                positive: true,
                tick: i,
            });
        }
        w.add(ContextEntry::Feedback {
            doc_id: 99,
            positive: false,
            tick: 3,
        });
        let s = w.stats();
        assert!((s.positive_feedback_ratio - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_stats_no_feedback_ratio_is_zero() {
        let mut w = default_window();
        w.add(ContextEntry::Query {
            text: "q".into(),
            embedding: make_embedding(0.0, 128),
            tick: 0,
        });
        let s = w.stats();
        assert_eq!(s.positive_feedback_ratio, 0.0);
    }

    #[test]
    fn test_stats_total_added_includes_evicted() {
        let config = WindowConfig {
            max_entries: 2,
            ..Default::default()
        };
        let mut w = SemanticContextWindow::new(config);
        for tick in 0..10u64 {
            w.add(ContextEntry::Feedback {
                doc_id: tick,
                positive: true,
                tick,
            });
        }
        let s = w.stats();
        assert_eq!(s.total_entries_added, 10);
        assert_eq!(s.feedbacks_in_window, 2);
    }

    // -----------------------------------------------------------------------
    // clear
    // -----------------------------------------------------------------------

    #[test]
    fn test_clear_resets_entries_preserves_config() {
        let config = WindowConfig {
            max_entries: 5,
            decay_factor: 0.8,
            embedding_dim: 64,
        };
        let mut w = SemanticContextWindow::new(config.clone());
        w.add(ContextEntry::Feedback {
            doc_id: 1,
            positive: true,
            tick: 0,
        });
        w.add(ContextEntry::Query {
            text: "q".into(),
            embedding: make_embedding(0.0, 64),
            tick: 1,
        });
        w.clear();
        assert!(w.entries.is_empty());
        assert_eq!(w.config, config);
        // total_added is NOT reset (clear only clears entries).
        assert_eq!(w.total_added, 2);
    }

    #[test]
    fn test_clear_then_add_works() {
        let mut w = default_window();
        w.add(ContextEntry::Feedback {
            doc_id: 1,
            positive: true,
            tick: 0,
        });
        w.clear();
        w.add(ContextEntry::Feedback {
            doc_id: 2,
            positive: false,
            tick: 1,
        });
        assert_eq!(w.entries.len(), 1);
        assert_eq!(w.total_added, 2);
    }

    // -----------------------------------------------------------------------
    // edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_recent_queries_zero_n_returns_empty() {
        let mut w = default_window();
        w.add(ContextEntry::Query {
            text: "q".into(),
            embedding: make_embedding(0.0, 128),
            tick: 0,
        });
        assert!(w.recent_queries(0).is_empty());
    }

    #[test]
    fn test_recent_results_zero_n_returns_empty() {
        let mut w = default_window();
        w.add(ContextEntry::Result {
            doc_id: 1,
            embedding: make_embedding(0.0, 128),
            relevance: 1.0,
            tick: 0,
        });
        assert!(w.recent_results(0).is_empty());
    }

    #[test]
    fn test_context_embedding_three_queries_recency_weighted() {
        // decay = 0.5 for easy math; 3 queries all same dim=2.
        // window_len = 3, indices 0,1,2.
        // w0 = 0.5^2 = 0.25, w1 = 0.5^1 = 0.5, w2 = 0.5^0 = 1.0
        // sum = 1.75
        // embeddings: [0,0], [1,1], [2,2]
        // result = (0.25*[0,0] + 0.5*[1,1] + 1.0*[2,2]) / 1.75
        //        = [0 + 0.5 + 2, 0 + 0.5 + 2] / 1.75
        //        = [2.5, 2.5] / 1.75 ≈ [1.4286, 1.4286]
        let config = WindowConfig {
            embedding_dim: 2,
            decay_factor: 0.5,
            ..Default::default()
        };
        let mut w = SemanticContextWindow::new(config);
        for (i, val) in [(0.0_f32), (1.0), (2.0)].iter().enumerate() {
            w.add(ContextEntry::Query {
                text: format!("q{i}"),
                embedding: vec![*val; 2],
                tick: i as u64,
            });
        }
        let result = w.context_embedding();
        let expected = 2.5_f32 / 1.75_f32;
        assert!((result[0] - expected).abs() < 1e-5, "got {}", result[0]);
        assert!((result[1] - expected).abs() < 1e-5, "got {}", result[1]);
    }

    #[test]
    fn test_positive_and_negative_docs_same_doc_id() {
        // Doc 5 has both positive and negative feedback at different ticks.
        let mut w = default_window();
        w.add(ContextEntry::Feedback {
            doc_id: 5,
            positive: true,
            tick: 0,
        });
        w.add(ContextEntry::Feedback {
            doc_id: 5,
            positive: false,
            tick: 1,
        });
        let pos = w.positive_docs();
        let neg = w.negative_docs();
        assert_eq!(pos, vec![5]);
        assert_eq!(neg, vec![5]);
    }

    #[test]
    fn test_window_default_config() {
        let c = WindowConfig::default();
        assert_eq!(c.max_entries, 20);
        assert!((c.decay_factor - 0.9).abs() < 1e-6);
        assert_eq!(c.embedding_dim, 128);
    }
}
