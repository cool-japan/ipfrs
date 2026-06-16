//! # Semantic Attribution Tracker
//!
//! Tracks which source documents contributed to search results and inference outputs,
//! providing attribution chains for explainability and audit.

use std::collections::HashMap;

/// Identifies the type and origin of a contributing source.
#[derive(Clone, Debug, PartialEq)]
pub enum AttributionSource {
    /// A source document contribution with a relevance score.
    Document { doc_id: u64, relevance: f32 },
    /// An embedding contribution with a cosine-similarity score.
    Embedding { embedding_id: u64, similarity: f32 },
    /// An inference result contribution with a confidence score.
    InferenceResult { result_id: u64, confidence: f32 },
}

impl AttributionSource {
    /// Returns the scalar weight for this source (relevance / similarity / confidence).
    pub fn weight(&self) -> f32 {
        match self {
            Self::Document { relevance, .. } => *relevance,
            Self::Embedding { similarity, .. } => *similarity,
            Self::InferenceResult { confidence, .. } => *confidence,
        }
    }
}

/// A single attribution record linking an output to its contributing sources.
#[derive(Clone, Debug)]
pub struct AttributionRecord {
    /// Unique id of this record.
    pub record_id: u64,
    /// The output being attributed (query result id, inference output id, etc.).
    pub output_id: u64,
    /// Ordered list of contributing sources.
    pub sources: Vec<AttributionSource>,
    /// Logical clock tick at which this record was created.
    pub tick: u64,
    /// Optional session that produced this output.
    pub session_id: Option<u64>,
}

impl AttributionRecord {
    /// Returns the source with the highest weight, or `None` when sources is empty.
    pub fn top_source(&self) -> Option<&AttributionSource> {
        self.sources
            .iter()
            .reduce(|best, s| if s.weight() > best.weight() { s } else { best })
    }

    /// Returns the number of contributing sources.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }
}

/// Aggregate statistics over all records held by a [`SemanticAttributionTracker`].
#[derive(Clone, Debug, PartialEq)]
pub struct AttributionStats {
    /// Total number of attribution records.
    pub total_records: usize,
    /// Sum of all source counts across every record.
    pub total_sources: usize,
    /// Number of `AttributionSource::Document` entries across all records.
    pub document_attributions: u64,
    /// Number of `AttributionSource::Embedding` entries across all records.
    pub embedding_attributions: u64,
    /// Number of `AttributionSource::InferenceResult` entries across all records.
    pub inference_attributions: u64,
    /// Average number of sources per record (0.0 when there are no records).
    pub avg_sources_per_record: f64,
}

/// Tracks attribution chains between outputs and their source documents / embeddings /
/// inference results, enabling explainability and audit.
pub struct SemanticAttributionTracker {
    /// Primary store: record_id → record.
    pub records: HashMap<u64, AttributionRecord>,
    /// Secondary index: output_id → list of record_ids.
    pub output_index: HashMap<u64, Vec<u64>>,
    /// Secondary index: session_id → list of record_ids.
    pub session_index: HashMap<u64, Vec<u64>>,
    next_record_id: u64,
}

impl SemanticAttributionTracker {
    /// Creates an empty tracker.
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
            output_index: HashMap::new(),
            session_index: HashMap::new(),
            next_record_id: 0,
        }
    }

    /// Records an attribution event and returns the new `record_id`.
    pub fn record(
        &mut self,
        output_id: u64,
        sources: Vec<AttributionSource>,
        tick: u64,
        session_id: Option<u64>,
    ) -> u64 {
        let record_id = self.next_record_id;
        self.next_record_id += 1;

        let rec = AttributionRecord {
            record_id,
            output_id,
            sources,
            tick,
            session_id,
        };

        self.output_index
            .entry(output_id)
            .or_default()
            .push(record_id);

        if let Some(sid) = session_id {
            self.session_index.entry(sid).or_default().push(record_id);
        }

        self.records.insert(record_id, rec);
        record_id
    }

    /// Returns all records attributed to `output_id`, sorted by `record_id` ascending.
    pub fn records_for_output(&self, output_id: u64) -> Vec<&AttributionRecord> {
        let ids = match self.output_index.get(&output_id) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let mut recs: Vec<&AttributionRecord> =
            ids.iter().filter_map(|id| self.records.get(id)).collect();

        recs.sort_by_key(|r| r.record_id);
        recs
    }

    /// Returns all records belonging to `session_id`, sorted by `tick` ascending then
    /// `record_id` ascending.
    pub fn records_for_session(&self, session_id: u64) -> Vec<&AttributionRecord> {
        let ids = match self.session_index.get(&session_id) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let mut recs: Vec<&AttributionRecord> =
            ids.iter().filter_map(|id| self.records.get(id)).collect();

        recs.sort_by(|a, b| a.tick.cmp(&b.tick).then(a.record_id.cmp(&b.record_id)));
        recs
    }

    /// Returns the `n` document ids most frequently appearing as
    /// [`AttributionSource::Document`] across all records.
    ///
    /// Ties are broken by `doc_id` ascending.  The result is truncated to `n` entries.
    pub fn top_documents(&self, n: usize) -> Vec<u64> {
        let mut freq: HashMap<u64, usize> = HashMap::new();

        for rec in self.records.values() {
            for src in &rec.sources {
                if let AttributionSource::Document { doc_id, .. } = src {
                    *freq.entry(*doc_id).or_insert(0) += 1;
                }
            }
        }

        let mut pairs: Vec<(u64, usize)> = freq.into_iter().collect();
        // Sort descending by frequency, ascending by doc_id for ties.
        pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        pairs.into_iter().take(n).map(|(id, _)| id).collect()
    }

    /// Removes a record and cleans up all secondary indices.
    /// Returns `true` if the record existed.
    pub fn remove_record(&mut self, record_id: u64) -> bool {
        let rec = match self.records.remove(&record_id) {
            Some(r) => r,
            None => return false,
        };

        // Clean output_index.
        if let Some(ids) = self.output_index.get_mut(&rec.output_id) {
            ids.retain(|id| *id != record_id);
            if ids.is_empty() {
                self.output_index.remove(&rec.output_id);
            }
        }

        // Clean session_index.
        if let Some(sid) = rec.session_id {
            if let Some(ids) = self.session_index.get_mut(&sid) {
                ids.retain(|id| *id != record_id);
                if ids.is_empty() {
                    self.session_index.remove(&sid);
                }
            }
        }

        true
    }

    /// Computes aggregate statistics over all held records.
    pub fn stats(&self) -> AttributionStats {
        let total_records = self.records.len();
        let mut total_sources: usize = 0;
        let mut document_attributions: u64 = 0;
        let mut embedding_attributions: u64 = 0;
        let mut inference_attributions: u64 = 0;

        for rec in self.records.values() {
            total_sources += rec.sources.len();
            for src in &rec.sources {
                match src {
                    AttributionSource::Document { .. } => document_attributions += 1,
                    AttributionSource::Embedding { .. } => embedding_attributions += 1,
                    AttributionSource::InferenceResult { .. } => inference_attributions += 1,
                }
            }
        }

        let avg_sources_per_record = if total_records == 0 {
            0.0
        } else {
            total_sources as f64 / total_records as f64
        };

        AttributionStats {
            total_records,
            total_sources,
            document_attributions,
            embedding_attributions,
            inference_attributions,
            avg_sources_per_record,
        }
    }
}

impl Default for SemanticAttributionTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn doc(doc_id: u64, relevance: f32) -> AttributionSource {
        AttributionSource::Document { doc_id, relevance }
    }

    fn emb(embedding_id: u64, similarity: f32) -> AttributionSource {
        AttributionSource::Embedding {
            embedding_id,
            similarity,
        }
    }

    fn inf(result_id: u64, confidence: f32) -> AttributionSource {
        AttributionSource::InferenceResult {
            result_id,
            confidence,
        }
    }

    // ── record creation ───────────────────────────────────────────────────────

    #[test]
    fn record_creates_attribution_record_with_correct_fields() {
        let mut tracker = SemanticAttributionTracker::new();
        let rid = tracker.record(42, vec![doc(1, 0.9)], 10, Some(99));
        assert_eq!(rid, 0);

        let rec = tracker.records.get(&rid).expect("record should exist");
        assert_eq!(rec.record_id, 0);
        assert_eq!(rec.output_id, 42);
        assert_eq!(rec.tick, 10);
        assert_eq!(rec.session_id, Some(99));
        assert_eq!(rec.sources.len(), 1);
    }

    #[test]
    fn record_ids_are_monotonically_increasing() {
        let mut tracker = SemanticAttributionTracker::new();
        let r0 = tracker.record(1, vec![], 0, None);
        let r1 = tracker.record(1, vec![], 1, None);
        let r2 = tracker.record(1, vec![], 2, None);
        assert!(r0 < r1 && r1 < r2);
    }

    #[test]
    fn record_without_session_id_not_in_session_index() {
        let mut tracker = SemanticAttributionTracker::new();
        tracker.record(1, vec![], 0, None);
        assert!(tracker.session_index.is_empty());
    }

    #[test]
    fn record_populates_output_index() {
        let mut tracker = SemanticAttributionTracker::new();
        let rid = tracker.record(5, vec![], 0, None);
        assert_eq!(tracker.output_index.get(&5), Some(&vec![rid]));
    }

    #[test]
    fn record_populates_session_index() {
        let mut tracker = SemanticAttributionTracker::new();
        let rid = tracker.record(5, vec![], 0, Some(7));
        assert_eq!(tracker.session_index.get(&7), Some(&vec![rid]));
    }

    // ── records_for_output ────────────────────────────────────────────────────

    #[test]
    fn records_for_output_sorted_by_record_id_ascending() {
        let mut tracker = SemanticAttributionTracker::new();
        let r0 = tracker.record(10, vec![], 5, None);
        let r1 = tracker.record(10, vec![], 3, None);
        let r2 = tracker.record(10, vec![], 7, None);

        let recs = tracker.records_for_output(10);
        let ids: Vec<u64> = recs.iter().map(|r| r.record_id).collect();
        assert_eq!(ids, vec![r0, r1, r2]);
    }

    #[test]
    fn records_for_output_returns_empty_for_unknown_output() {
        let tracker = SemanticAttributionTracker::new();
        assert!(tracker.records_for_output(999).is_empty());
    }

    #[test]
    fn records_for_output_only_returns_matching_output() {
        let mut tracker = SemanticAttributionTracker::new();
        tracker.record(1, vec![], 0, None);
        tracker.record(2, vec![], 0, None);
        let recs = tracker.records_for_output(1);
        assert!(recs.iter().all(|r| r.output_id == 1));
        assert_eq!(recs.len(), 1);
    }

    // ── records_for_session ───────────────────────────────────────────────────

    #[test]
    fn records_for_session_sorted_by_tick_then_record_id() {
        let mut tracker = SemanticAttributionTracker::new();
        // Deliberately insert in non-sorted tick order.
        let r_tick5 = tracker.record(10, vec![], 5, Some(1));
        let r_tick2 = tracker.record(11, vec![], 2, Some(1));
        let r_tick5b = tracker.record(12, vec![], 5, Some(1)); // same tick as r_tick5

        let recs = tracker.records_for_session(1);
        let ids: Vec<u64> = recs.iter().map(|r| r.record_id).collect();
        // Expected: tick=2 first, then tick=5 (r_tick5 < r_tick5b by record_id).
        assert_eq!(ids, vec![r_tick2, r_tick5, r_tick5b]);
    }

    #[test]
    fn records_for_session_returns_empty_for_unknown_session() {
        let tracker = SemanticAttributionTracker::new();
        assert!(tracker.records_for_session(42).is_empty());
    }

    #[test]
    fn records_for_session_isolates_sessions() {
        let mut tracker = SemanticAttributionTracker::new();
        tracker.record(1, vec![], 0, Some(10));
        tracker.record(2, vec![], 0, Some(20));

        assert_eq!(tracker.records_for_session(10).len(), 1);
        assert_eq!(tracker.records_for_session(20).len(), 1);
    }

    // ── top_source ────────────────────────────────────────────────────────────

    #[test]
    fn top_source_returns_none_for_empty_sources() {
        let rec = AttributionRecord {
            record_id: 0,
            output_id: 0,
            sources: vec![],
            tick: 0,
            session_id: None,
        };
        assert!(rec.top_source().is_none());
    }

    #[test]
    fn top_source_document_highest_relevance() {
        let rec = AttributionRecord {
            record_id: 0,
            output_id: 0,
            sources: vec![doc(1, 0.3), doc(2, 0.95), doc(3, 0.5)],
            tick: 0,
            session_id: None,
        };
        assert_eq!(rec.top_source(), Some(&doc(2, 0.95)));
    }

    #[test]
    fn top_source_embedding_highest_similarity() {
        let rec = AttributionRecord {
            record_id: 0,
            output_id: 0,
            sources: vec![emb(1, 0.4), emb(2, 0.85)],
            tick: 0,
            session_id: None,
        };
        assert_eq!(rec.top_source(), Some(&emb(2, 0.85)));
    }

    #[test]
    fn top_source_inference_highest_confidence() {
        let rec = AttributionRecord {
            record_id: 0,
            output_id: 0,
            sources: vec![inf(1, 0.6), inf(2, 0.99), inf(3, 0.7)],
            tick: 0,
            session_id: None,
        };
        assert_eq!(rec.top_source(), Some(&inf(2, 0.99)));
    }

    #[test]
    fn top_source_mixed_types_picks_global_max() {
        let rec = AttributionRecord {
            record_id: 0,
            output_id: 0,
            sources: vec![doc(1, 0.7), emb(2, 0.8), inf(3, 0.75)],
            tick: 0,
            session_id: None,
        };
        assert_eq!(rec.top_source(), Some(&emb(2, 0.8)));
    }

    // ── top_documents ─────────────────────────────────────────────────────────

    #[test]
    fn top_documents_frequency_ranking() {
        let mut tracker = SemanticAttributionTracker::new();
        tracker.record(1, vec![doc(10, 0.9), doc(20, 0.8)], 0, None);
        tracker.record(2, vec![doc(10, 0.7)], 0, None);

        let top = tracker.top_documents(5);
        // doc 10 appears 2 times, doc 20 appears 1 time.
        assert_eq!(top[0], 10);
        assert_eq!(top[1], 20);
    }

    #[test]
    fn top_documents_tie_breaking_by_doc_id_ascending() {
        let mut tracker = SemanticAttributionTracker::new();
        // doc 30 and doc 5 each appear once.
        tracker.record(1, vec![doc(30, 0.5)], 0, None);
        tracker.record(2, vec![doc(5, 0.5)], 0, None);

        let top = tracker.top_documents(5);
        // Both have freq 1; lower doc_id should come first.
        assert_eq!(top[0], 5);
        assert_eq!(top[1], 30);
    }

    #[test]
    fn top_documents_truncated_to_n() {
        let mut tracker = SemanticAttributionTracker::new();
        tracker.record(1, vec![doc(1, 0.9), doc(2, 0.8), doc(3, 0.7)], 0, None);
        assert_eq!(tracker.top_documents(2).len(), 2);
    }

    #[test]
    fn top_documents_ignores_non_document_sources() {
        let mut tracker = SemanticAttributionTracker::new();
        tracker.record(1, vec![emb(99, 0.9), inf(88, 0.9)], 0, None);
        assert!(tracker.top_documents(10).is_empty());
    }

    // ── remove_record ─────────────────────────────────────────────────────────

    #[test]
    fn remove_record_returns_false_for_unknown_id() {
        let mut tracker = SemanticAttributionTracker::new();
        assert!(!tracker.remove_record(999));
    }

    #[test]
    fn remove_record_removes_from_all_indices() {
        let mut tracker = SemanticAttributionTracker::new();
        let rid = tracker.record(1, vec![doc(10, 0.9)], 0, Some(5));

        assert!(tracker.remove_record(rid));

        assert!(!tracker.records.contains_key(&rid));
        assert!(!tracker.output_index.contains_key(&1));
        assert!(!tracker.session_index.contains_key(&5));
    }

    #[test]
    fn remove_record_partial_output_index_cleanup() {
        let mut tracker = SemanticAttributionTracker::new();
        let r0 = tracker.record(1, vec![], 0, None);
        let r1 = tracker.record(1, vec![], 1, None);

        tracker.remove_record(r0);

        let ids = tracker.output_index.get(&1).expect("index should remain");
        assert_eq!(ids, &vec![r1]);
    }

    #[test]
    fn remove_record_partial_session_index_cleanup() {
        let mut tracker = SemanticAttributionTracker::new();
        let r0 = tracker.record(1, vec![], 0, Some(3));
        let r1 = tracker.record(2, vec![], 1, Some(3));

        tracker.remove_record(r0);

        let ids = tracker.session_index.get(&3).expect("index should remain");
        assert_eq!(ids, &vec![r1]);
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn stats_empty_tracker() {
        let tracker = SemanticAttributionTracker::new();
        let s = tracker.stats();
        assert_eq!(s.total_records, 0);
        assert_eq!(s.total_sources, 0);
        assert_eq!(s.avg_sources_per_record, 0.0);
    }

    #[test]
    fn stats_total_records() {
        let mut tracker = SemanticAttributionTracker::new();
        tracker.record(1, vec![], 0, None);
        tracker.record(2, vec![], 0, None);
        assert_eq!(tracker.stats().total_records, 2);
    }

    #[test]
    fn stats_total_sources() {
        let mut tracker = SemanticAttributionTracker::new();
        tracker.record(1, vec![doc(1, 0.9), emb(2, 0.8)], 0, None);
        tracker.record(2, vec![inf(3, 0.7)], 0, None);
        assert_eq!(tracker.stats().total_sources, 3);
    }

    #[test]
    fn stats_by_type() {
        let mut tracker = SemanticAttributionTracker::new();
        tracker.record(
            1,
            vec![doc(1, 0.9), doc(2, 0.8), emb(1, 0.7), inf(1, 0.6)],
            0,
            None,
        );
        let s = tracker.stats();
        assert_eq!(s.document_attributions, 2);
        assert_eq!(s.embedding_attributions, 1);
        assert_eq!(s.inference_attributions, 1);
    }

    #[test]
    fn stats_avg_sources_per_record() {
        let mut tracker = SemanticAttributionTracker::new();
        tracker.record(1, vec![doc(1, 0.9), doc(2, 0.8)], 0, None); // 2 sources
        tracker.record(2, vec![emb(1, 0.7)], 0, None); // 1 source
                                                       // avg = 3 / 2 = 1.5
        let s = tracker.stats();
        assert!((s.avg_sources_per_record - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_session_index_tracks_correctly() {
        let mut tracker = SemanticAttributionTracker::new();
        tracker.record(1, vec![], 0, Some(42));
        tracker.record(2, vec![], 0, Some(42));
        tracker.record(3, vec![], 0, Some(99));

        assert_eq!(tracker.records_for_session(42).len(), 2);
        assert_eq!(tracker.records_for_session(99).len(), 1);
    }

    #[test]
    fn source_count_correct() {
        let rec = AttributionRecord {
            record_id: 0,
            output_id: 0,
            sources: vec![doc(1, 0.9), emb(2, 0.8), inf(3, 0.7)],
            tick: 0,
            session_id: None,
        };
        assert_eq!(rec.source_count(), 3);
    }

    #[test]
    fn top_documents_n_zero_returns_empty() {
        let mut tracker = SemanticAttributionTracker::new();
        tracker.record(1, vec![doc(1, 0.9)], 0, None);
        assert!(tracker.top_documents(0).is_empty());
    }
}
