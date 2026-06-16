//! Semantic Feedback Loop — relevance feedback collection and query re-ranking.
//!
//! Collects explicit and implicit relevance signals from users and uses them
//! to adjust future search rankings through query expansion and score boosting.

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// FNV-1a helper (no external dep needed for a simple hash)
// ---------------------------------------------------------------------------

/// Compute the FNV-1a 64-bit hash of an arbitrary byte slice.
pub fn fnv1a_64(data: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;

    let mut hash = OFFSET_BASIS;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// Compute the query-id (FNV-1a hash) for an arbitrary query string.
pub fn query_id_for(query_text: &str) -> u64 {
    fnv1a_64(query_text.as_bytes())
}

// ---------------------------------------------------------------------------
// FeedbackType — user-facing relevance judgment
// ---------------------------------------------------------------------------

/// Classification of a search result's relevance as judged by a user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedbackType {
    /// The result was relevant to the query.
    Relevant,
    /// The result was not relevant to the query.
    Irrelevant,
    /// The result was somewhat relevant but not a perfect match.
    PartiallyRelevant,
}

// ---------------------------------------------------------------------------
// FeedbackEntry — a single user-feedback record
// ---------------------------------------------------------------------------

/// A single feedback record submitted by a user for a query–document pair.
#[derive(Debug, Clone)]
pub struct FeedbackEntry {
    /// Identifier of the query this feedback pertains to.
    pub query_id: String,
    /// Identifier of the document this feedback pertains to.
    pub doc_id: String,
    /// The user's relevance judgment.
    pub feedback: FeedbackType,
    /// Monotonic tick at which this entry was recorded.
    pub tick: u64,
    /// User-reported confidence in their judgment (0.0–1.0).
    pub confidence: f64,
}

// ---------------------------------------------------------------------------
// QueryFeedbackSummary — aggregated feedback for a single query
// ---------------------------------------------------------------------------

/// Aggregated feedback statistics for a single query.
#[derive(Debug, Clone)]
pub struct QueryFeedbackSummary {
    /// The query this summary pertains to.
    pub query_id: String,
    /// Number of documents marked `Relevant`.
    pub relevant_count: usize,
    /// Number of documents marked `Irrelevant`.
    pub irrelevant_count: usize,
    /// Number of documents marked `PartiallyRelevant`.
    pub partial_count: usize,
    /// Mean confidence across all feedback entries for this query.
    pub avg_confidence: f64,
    /// Precision: `relevant / (relevant + irrelevant)`.  `0.0` when denominator is zero.
    pub precision: f64,
}

// ---------------------------------------------------------------------------
// FeedbackLoopStats — global loop statistics
// ---------------------------------------------------------------------------

/// Aggregate statistics across all feedback entries in the loop.
#[derive(Debug, Clone)]
pub struct FeedbackLoopStats {
    /// Total number of feedback entries currently stored.
    pub total_entries: usize,
    /// Number of unique query IDs with at least one entry.
    pub unique_queries: usize,
    /// Overall precision across all queries (`None` if no relevant+irrelevant entries).
    pub overall_precision: Option<f64>,
    /// Mean confidence across all entries (`0.0` when empty).
    pub avg_confidence: f64,
}

// ---------------------------------------------------------------------------
// FeedbackSignal
// ---------------------------------------------------------------------------

/// A single relevance signal emitted by a user (explicit or implicit).
#[derive(Clone, Debug, PartialEq)]
pub enum FeedbackSignal {
    /// The user explicitly confirmed that a result was relevant.
    Relevant { result_id: u64, rank: usize },
    /// The user explicitly marked a result as not relevant.
    Irrelevant { result_id: u64, rank: usize },
    /// Implicit positive signal: the user clicked a result and dwelt on it.
    Clicked {
        result_id: u64,
        rank: usize,
        dwell_ms: u64,
    },
}

impl FeedbackSignal {
    /// Return the `result_id` carried by any variant.
    pub fn result_id(&self) -> u64 {
        match self {
            Self::Relevant { result_id, .. } => *result_id,
            Self::Irrelevant { result_id, .. } => *result_id,
            Self::Clicked { result_id, .. } => *result_id,
        }
    }

    /// Return the rank of the result that triggered this signal.
    pub fn rank(&self) -> usize {
        match self {
            Self::Relevant { rank, .. } => *rank,
            Self::Irrelevant { rank, .. } => *rank,
            Self::Clicked { rank, .. } => *rank,
        }
    }

    /// Return true if the signal conveys a positive relevance judgment.
    pub fn is_positive(&self) -> bool {
        matches!(self, Self::Relevant { .. } | Self::Clicked { .. })
    }
}

// ---------------------------------------------------------------------------
// QueryFeedback
// ---------------------------------------------------------------------------

/// All feedback signals collected for a single query.
#[derive(Clone, Debug)]
pub struct QueryFeedback {
    /// FNV-1a hash of the query text.
    pub query_id: u64,
    /// Signals gathered for this query (in insertion order).
    pub signals: Vec<FeedbackSignal>,
    /// Unix timestamp (seconds) when the first signal was collected.
    pub collected_at_secs: u64,
}

impl QueryFeedback {
    /// Create a new `QueryFeedback` with no signals yet.
    pub fn new(query_id: u64, collected_at_secs: u64) -> Self {
        Self {
            query_id,
            signals: Vec::new(),
            collected_at_secs,
        }
    }

    /// Aggregate relevance score across all collected signals.
    ///
    /// Scoring:
    /// * `Relevant`   → +1.0
    /// * `Irrelevant` → -0.5
    /// * `Clicked`    → +0.3 (dwell time is ignored here; see `BoostRecord`)
    ///
    /// The final value is clamped to `[-10.0, 10.0]`.
    pub fn relevance_score(&self) -> f64 {
        let raw: f64 = self
            .signals
            .iter()
            .map(|s| match s {
                FeedbackSignal::Relevant { .. } => 1.0,
                FeedbackSignal::Irrelevant { .. } => -0.5,
                FeedbackSignal::Clicked { .. } => 0.3,
            })
            .sum();
        raw.clamp(-10.0, 10.0)
    }

    /// Return the `result_id`s of all positive signals (`Relevant` + `Clicked`).
    pub fn positive_ids(&self) -> Vec<u64> {
        self.signals
            .iter()
            .filter(|s| s.is_positive())
            .map(|s| s.result_id())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// BoostRecord
// ---------------------------------------------------------------------------

/// Cumulative boost information for a single result document.
#[derive(Clone, Debug)]
pub struct BoostRecord {
    /// The result this record belongs to.
    pub result_id: u64,
    /// Cumulative sum of all boost contributions from feedback signals.
    pub boost_score: f64,
    /// How many signals have contributed to `boost_score`.
    pub feedback_count: u64,
}

impl BoostRecord {
    /// Create a new, empty `BoostRecord`.
    pub fn new(result_id: u64) -> Self {
        Self {
            result_id,
            boost_score: 0.0,
            feedback_count: 0,
        }
    }

    /// The per-signal average boost: `boost_score / max(feedback_count, 1)`.
    pub fn effective_boost(&self) -> f64 {
        self.boost_score / (self.feedback_count.max(1) as f64)
    }
}

// ---------------------------------------------------------------------------
// FeedbackStats
// ---------------------------------------------------------------------------

/// Aggregate statistics across all queries and signals.
#[derive(Clone, Debug, Default)]
pub struct FeedbackStats {
    /// Total number of distinct queries that have received at least one signal.
    pub total_queries: usize,
    /// Grand total of signals recorded (all types).
    pub total_signals: u64,
    /// Number of `Relevant` signals recorded.
    pub relevant_count: u64,
    /// Number of `Irrelevant` signals recorded.
    pub irrelevant_count: u64,
    /// Number of `Clicked` signals recorded.
    pub clicked_count: u64,
}

impl FeedbackStats {
    /// Fraction of positive signals: `(relevant + clicked) / max(total_signals, 1)`.
    pub fn signal_ratio(&self) -> f64 {
        let positive = self.relevant_count + self.clicked_count;
        positive as f64 / (self.total_signals.max(1) as f64)
    }
}

// ---------------------------------------------------------------------------
// SemanticFeedbackLoop
// ---------------------------------------------------------------------------

/// Core feedback-loop engine.
///
/// Ingests relevance signals emitted during search sessions and uses them to
/// boost (or suppress) future rankings of individual results.
#[derive(Debug)]
pub struct SemanticFeedbackLoop {
    /// Per-query feedback records, keyed by `query_id`.
    pub feedback: HashMap<u64, QueryFeedback>,
    /// Per-result cumulative boost records, keyed by `result_id`.
    pub boosts: HashMap<u64, BoostRecord>,
    /// Global statistics.
    pub stats: FeedbackStats,

    // --- user-facing feedback entries ---
    /// Ordered list of user feedback entries.
    entries: Vec<FeedbackEntry>,
    /// Monotonic tick counter for ordering entries.
    current_tick: u64,
    /// Maximum number of entries retained before oldest are evicted.
    max_entries: usize,
}

impl SemanticFeedbackLoop {
    /// Create a new, empty feedback loop with the default capacity of 50 000 entries.
    pub fn new() -> Self {
        Self::with_max_entries(50_000)
    }

    /// Create a new, empty feedback loop with the given maximum entry capacity.
    pub fn with_max_entries(max_entries: usize) -> Self {
        Self {
            feedback: HashMap::new(),
            boosts: HashMap::new(),
            stats: FeedbackStats::default(),
            entries: Vec::new(),
            current_tick: 0,
            max_entries,
        }
    }

    // ------------------------------------------------------------------
    // Signal recording
    // ------------------------------------------------------------------

    /// Record a single relevance signal for the given query.
    ///
    /// * Updates the per-query `QueryFeedback` record.
    /// * Updates the per-result `BoostRecord`.
    /// * Updates global `FeedbackStats`.
    pub fn record_feedback(&mut self, query_id: u64, signal: FeedbackSignal, now_secs: u64) {
        // -------- stats ------------------------------------------------
        self.stats.total_signals += 1;
        match &signal {
            FeedbackSignal::Relevant { .. } => self.stats.relevant_count += 1,
            FeedbackSignal::Irrelevant { .. } => self.stats.irrelevant_count += 1,
            FeedbackSignal::Clicked { .. } => self.stats.clicked_count += 1,
        }

        // -------- boost record -----------------------------------------
        let boost_delta = Self::boost_delta_for(&signal);
        let result_id = signal.result_id();
        let record = self
            .boosts
            .entry(result_id)
            .or_insert_with(|| BoostRecord::new(result_id));
        record.boost_score += boost_delta;
        record.feedback_count += 1;

        // -------- query feedback ---------------------------------------
        let is_new_query = !self.feedback.contains_key(&query_id);
        let qf = self
            .feedback
            .entry(query_id)
            .or_insert_with(|| QueryFeedback::new(query_id, now_secs));
        qf.signals.push(signal);

        if is_new_query {
            self.stats.total_queries += 1;
        }
    }

    /// Compute the boost delta contributed by a single signal to a `BoostRecord`.
    fn boost_delta_for(signal: &FeedbackSignal) -> f64 {
        match signal {
            FeedbackSignal::Relevant { .. } => 1.0,
            FeedbackSignal::Irrelevant { .. } => -0.5,
            FeedbackSignal::Clicked { dwell_ms, .. } => {
                // Base +0.3, scaled by dwell: +0.3 * (1 + dwell_s)
                // No explicit cap is defined in the spec; we keep it unbounded
                // here — `effective_boost` normalises across signal count.
                let dwell_s = *dwell_ms as f64 / 1000.0;
                0.3 * (1.0 + dwell_s)
            }
        }
    }

    // ------------------------------------------------------------------
    // Score application
    // ------------------------------------------------------------------

    /// Re-score and re-rank a list of `(result_id, score)` pairs using boosts.
    ///
    /// For each entry: `new_score = score * (1.0 + effective_boost().max(-0.9))`
    ///
    /// The result list is returned sorted by `new_score` descending.
    pub fn apply_boosts(&self, results: &[(u64, f64)]) -> Vec<(u64, f64)> {
        let mut boosted: Vec<(u64, f64)> = results
            .iter()
            .map(|&(id, score)| {
                let multiplier = 1.0
                    + self
                        .boosts
                        .get(&id)
                        .map(|r| r.effective_boost().max(-0.9))
                        .unwrap_or(0.0);
                (id, score * multiplier)
            })
            .collect();

        // Sort descending by new score; stable sort to preserve tie ordering.
        boosted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        boosted
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Return the top-`k` result IDs sorted by `effective_boost` descending.
    pub fn top_boosted_ids(&self, k: usize) -> Vec<u64> {
        let mut pairs: Vec<(u64, f64)> = self
            .boosts
            .values()
            .map(|r| (r.result_id, r.effective_boost()))
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.into_iter().take(k).map(|(id, _)| id).collect()
    }

    /// Return the positive `result_id`s recorded for a specific query.
    ///
    /// Returns an empty `Vec` if the query has not been seen.
    pub fn positive_ids_for_query(&self, query_id: u64) -> Vec<u64> {
        self.feedback
            .get(&query_id)
            .map(|qf| qf.positive_ids())
            .unwrap_or_default()
    }

    /// Reference to the global statistics.
    pub fn stats(&self) -> &FeedbackStats {
        &self.stats
    }

    // ------------------------------------------------------------------
    // User-facing feedback entry API
    // ------------------------------------------------------------------

    /// Record a user feedback entry.  If the loop is at capacity the oldest
    /// entry is evicted before inserting the new one.
    pub fn record(
        &mut self,
        query_id: &str,
        doc_id: &str,
        feedback: FeedbackType,
        confidence: f64,
    ) {
        let clamped = confidence.clamp(0.0, 1.0);
        let entry = FeedbackEntry {
            query_id: query_id.to_string(),
            doc_id: doc_id.to_string(),
            feedback,
            tick: self.current_tick,
            confidence: clamped,
        };
        self.current_tick += 1;

        if self.entries.len() >= self.max_entries {
            self.entries.remove(0);
        }
        self.entries.push(entry);
    }

    /// Aggregate feedback for a specific query.  Returns `None` if the query
    /// has no recorded entries.
    pub fn get_summary(&self, query_id: &str) -> Option<QueryFeedbackSummary> {
        let mut relevant_count: usize = 0;
        let mut irrelevant_count: usize = 0;
        let mut partial_count: usize = 0;
        let mut conf_sum: f64 = 0.0;
        let mut total: usize = 0;

        for e in &self.entries {
            if e.query_id == query_id {
                total += 1;
                conf_sum += e.confidence;
                match e.feedback {
                    FeedbackType::Relevant => relevant_count += 1,
                    FeedbackType::Irrelevant => irrelevant_count += 1,
                    FeedbackType::PartiallyRelevant => partial_count += 1,
                }
            }
        }

        if total == 0 {
            return None;
        }

        let avg_confidence = conf_sum / total as f64;
        let denom = relevant_count + irrelevant_count;
        let precision = if denom > 0 {
            relevant_count as f64 / denom as f64
        } else {
            0.0
        };

        Some(QueryFeedbackSummary {
            query_id: query_id.to_string(),
            relevant_count,
            irrelevant_count,
            partial_count,
            avg_confidence,
            precision,
        })
    }

    /// Return document IDs marked `Relevant` for the given query.
    pub fn relevant_docs(&self, query_id: &str) -> Vec<String> {
        self.entries
            .iter()
            .filter(|e| e.query_id == query_id && e.feedback == FeedbackType::Relevant)
            .map(|e| e.doc_id.clone())
            .collect()
    }

    /// Return document IDs marked `Irrelevant` for the given query.
    pub fn irrelevant_docs(&self, query_id: &str) -> Vec<String> {
        self.entries
            .iter()
            .filter(|e| e.query_id == query_id && e.feedback == FeedbackType::Irrelevant)
            .map(|e| e.doc_id.clone())
            .collect()
    }

    /// Precision for a single query: `relevant / (relevant + irrelevant)`.
    /// Returns `None` when the query has no relevant or irrelevant entries.
    pub fn precision_at_query(&self, query_id: &str) -> Option<f64> {
        let mut rel: usize = 0;
        let mut irr: usize = 0;
        for e in &self.entries {
            if e.query_id == query_id {
                match e.feedback {
                    FeedbackType::Relevant => rel += 1,
                    FeedbackType::Irrelevant => irr += 1,
                    FeedbackType::PartiallyRelevant => {}
                }
            }
        }
        let denom = rel + irr;
        if denom == 0 {
            None
        } else {
            Some(rel as f64 / denom as f64)
        }
    }

    /// Overall precision across all queries: `total_relevant / (total_relevant + total_irrelevant)`.
    /// Returns `None` when there are no relevant or irrelevant entries at all.
    pub fn overall_precision(&self) -> Option<f64> {
        let mut rel: usize = 0;
        let mut irr: usize = 0;
        for e in &self.entries {
            match e.feedback {
                FeedbackType::Relevant => rel += 1,
                FeedbackType::Irrelevant => irr += 1,
                FeedbackType::PartiallyRelevant => {}
            }
        }
        let denom = rel + irr;
        if denom == 0 {
            None
        } else {
            Some(rel as f64 / denom as f64)
        }
    }

    /// Total number of user feedback entries currently stored.
    pub fn feedback_count(&self) -> usize {
        self.entries.len()
    }

    /// Unique query IDs that have at least one feedback entry.
    pub fn queries_with_feedback(&self) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for e in &self.entries {
            if seen.insert(&e.query_id) {
                result.push(e.query_id.clone());
            }
        }
        result
    }

    /// Advance the internal tick counter by one.
    pub fn tick(&mut self) {
        self.current_tick += 1;
    }

    /// Remove all user feedback entries and reset the tick counter.
    pub fn clear_entries(&mut self) {
        self.entries.clear();
        self.current_tick = 0;
    }

    /// Compute aggregate [`FeedbackLoopStats`] from the current entries.
    pub fn loop_stats(&self) -> FeedbackLoopStats {
        let total_entries = self.entries.len();
        let unique_queries = self.queries_with_feedback().len();
        let overall_precision = self.overall_precision();

        let avg_confidence = if total_entries == 0 {
            0.0
        } else {
            let sum: f64 = self.entries.iter().map(|e| e.confidence).sum();
            sum / total_entries as f64
        };

        FeedbackLoopStats {
            total_entries,
            unique_queries,
            overall_precision,
            avg_confidence,
        }
    }
}

impl Default for SemanticFeedbackLoop {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helper builders
    // ------------------------------------------------------------------

    fn relevant(result_id: u64, rank: usize) -> FeedbackSignal {
        FeedbackSignal::Relevant { result_id, rank }
    }

    fn irrelevant(result_id: u64, rank: usize) -> FeedbackSignal {
        FeedbackSignal::Irrelevant { result_id, rank }
    }

    fn clicked(result_id: u64, rank: usize, dwell_ms: u64) -> FeedbackSignal {
        FeedbackSignal::Clicked {
            result_id,
            rank,
            dwell_ms,
        }
    }

    // ------------------------------------------------------------------
    // Test 1: record Relevant increments relevant_count
    // ------------------------------------------------------------------
    #[test]
    fn test_record_relevant_increments_relevant_count() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record_feedback(1, relevant(10, 0), 1000);
        assert_eq!(fl.stats().relevant_count, 1);
        assert_eq!(fl.stats().irrelevant_count, 0);
        assert_eq!(fl.stats().clicked_count, 0);
        assert_eq!(fl.stats().total_signals, 1);
    }

    // ------------------------------------------------------------------
    // Test 2: record Irrelevant increments irrelevant_count
    // ------------------------------------------------------------------
    #[test]
    fn test_record_irrelevant_increments_irrelevant_count() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record_feedback(1, irrelevant(10, 0), 1000);
        assert_eq!(fl.stats().irrelevant_count, 1);
        assert_eq!(fl.stats().relevant_count, 0);
        assert_eq!(fl.stats().total_signals, 1);
    }

    // ------------------------------------------------------------------
    // Test 3: record Clicked increments clicked_count
    // ------------------------------------------------------------------
    #[test]
    fn test_record_clicked_increments_clicked_count() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record_feedback(1, clicked(10, 0, 500), 1000);
        assert_eq!(fl.stats().clicked_count, 1);
        assert_eq!(fl.stats().relevant_count, 0);
        assert_eq!(fl.stats().total_signals, 1);
    }

    // ------------------------------------------------------------------
    // Test 4: relevance_score — Relevant contributes +1.0
    // ------------------------------------------------------------------
    #[test]
    fn test_relevance_score_relevant_only() {
        let mut qf = QueryFeedback::new(42, 0);
        qf.signals.push(relevant(1, 0));
        qf.signals.push(relevant(2, 1));
        assert!((qf.relevance_score() - 2.0).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // Test 5: relevance_score — Irrelevant contributes -0.5
    // ------------------------------------------------------------------
    #[test]
    fn test_relevance_score_irrelevant_only() {
        let mut qf = QueryFeedback::new(42, 0);
        qf.signals.push(irrelevant(1, 0));
        qf.signals.push(irrelevant(2, 1));
        assert!((qf.relevance_score() - (-1.0)).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // Test 6: relevance_score — Clicked contributes +0.3
    // ------------------------------------------------------------------
    #[test]
    fn test_relevance_score_clicked_only() {
        let mut qf = QueryFeedback::new(42, 0);
        qf.signals.push(clicked(1, 0, 5000));
        assert!((qf.relevance_score() - 0.3).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // Test 7: relevance_score — mixed signals accumulate correctly
    // ------------------------------------------------------------------
    #[test]
    fn test_relevance_score_mixed() {
        let mut qf = QueryFeedback::new(42, 0);
        qf.signals.push(relevant(1, 0)); // +1.0
        qf.signals.push(irrelevant(2, 1)); // -0.5
        qf.signals.push(clicked(3, 2, 0)); // +0.3
        let expected = 1.0 - 0.5 + 0.3;
        assert!((qf.relevance_score() - expected).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // Test 8: relevance_score — clamped to [-10, 10]
    // ------------------------------------------------------------------
    #[test]
    fn test_relevance_score_clamped_positive() {
        let mut qf = QueryFeedback::new(42, 0);
        for i in 0..20 {
            qf.signals.push(relevant(i, i as usize));
        }
        assert!((qf.relevance_score() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_relevance_score_clamped_negative() {
        let mut qf = QueryFeedback::new(42, 0);
        for i in 0..30 {
            qf.signals.push(irrelevant(i, i as usize));
        }
        assert!((qf.relevance_score() - (-10.0)).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // Test 9: positive_ids returns only Relevant + Clicked ids
    // ------------------------------------------------------------------
    #[test]
    fn test_positive_ids() {
        let mut qf = QueryFeedback::new(42, 0);
        qf.signals.push(relevant(10, 0));
        qf.signals.push(irrelevant(20, 1));
        qf.signals.push(clicked(30, 2, 100));
        let ids = qf.positive_ids();
        assert!(ids.contains(&10));
        assert!(!ids.contains(&20));
        assert!(ids.contains(&30));
        assert_eq!(ids.len(), 2);
    }

    // ------------------------------------------------------------------
    // Test 10: apply_boosts re-sorts results by boosted score
    // ------------------------------------------------------------------
    #[test]
    fn test_apply_boosts_resorts() {
        let mut fl = SemanticFeedbackLoop::new();
        // Result 99 gets a positive boost
        fl.record_feedback(1, relevant(99, 1), 0);

        // Initially result 100 has a higher raw score
        let results = vec![(100u64, 0.9), (99u64, 0.5)];
        let boosted = fl.apply_boosts(&results);

        // result 99 should now rank first (score * (1 + 1.0) = 1.0 > 0.9)
        assert_eq!(boosted[0].0, 99);
        assert_eq!(boosted[1].0, 100);
    }

    // ------------------------------------------------------------------
    // Test 11: apply_boosts — result with no boost record is unchanged
    // ------------------------------------------------------------------
    #[test]
    fn test_apply_boosts_no_boost_unchanged() {
        let fl = SemanticFeedbackLoop::new();
        let results = vec![(1u64, 0.8), (2u64, 0.6)];
        let boosted = fl.apply_boosts(&results);
        // Order must be preserved (descending by score, no boost applied)
        assert_eq!(boosted[0].0, 1);
        assert!((boosted[0].1 - 0.8).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // Test 12: effective_boost normalises by feedback_count
    // ------------------------------------------------------------------
    #[test]
    fn test_effective_boost() {
        let mut r = BoostRecord::new(7);
        r.boost_score = 3.0;
        r.feedback_count = 3;
        assert!((r.effective_boost() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_effective_boost_zero_count() {
        let r = BoostRecord::new(7);
        // feedback_count == 0 → max(0,1) = 1
        assert!((r.effective_boost() - 0.0).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // Test 13: top_boosted_ids returns top-k by effective_boost desc
    // ------------------------------------------------------------------
    #[test]
    fn test_top_boosted_ids() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record_feedback(1, relevant(10, 0), 0); // boost = +1.0
        fl.record_feedback(1, relevant(20, 1), 0); // boost = +1.0
        fl.record_feedback(1, relevant(20, 1), 0); // boost += 1.0 → 2.0 total, eff = 1.0
        fl.record_feedback(1, irrelevant(30, 2), 0); // boost = -0.5

        let top = fl.top_boosted_ids(2);
        assert_eq!(top.len(), 2);
        // 10 and 20 should be the top 2 (both effective 1.0, 30 is negative)
        assert!(!top.contains(&30));
    }

    // ------------------------------------------------------------------
    // Test 14: signal_ratio
    // ------------------------------------------------------------------
    #[test]
    fn test_signal_ratio() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record_feedback(1, relevant(1, 0), 0);
        fl.record_feedback(1, clicked(2, 1, 0), 0);
        fl.record_feedback(1, irrelevant(3, 2), 0);
        // ratio = (1 + 1) / 3
        let ratio = fl.stats().signal_ratio();
        assert!((ratio - (2.0 / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_signal_ratio_no_signals() {
        let fl = SemanticFeedbackLoop::new();
        // total_signals = 0 → max(0,1) = 1, ratio = 0/1 = 0
        assert!((fl.stats().signal_ratio() - 0.0).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // Test 15: multiple signals same query accumulate
    // ------------------------------------------------------------------
    #[test]
    fn test_multiple_signals_same_query_accumulate() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record_feedback(42, relevant(1, 0), 1000);
        fl.record_feedback(42, relevant(2, 1), 2000);
        fl.record_feedback(42, irrelevant(3, 2), 3000);

        assert_eq!(fl.feedback[&42].signals.len(), 3);
        assert_eq!(fl.stats().total_queries, 1); // still one unique query
        assert_eq!(fl.stats().total_signals, 3);
    }

    // ------------------------------------------------------------------
    // Test 16: boost from Clicked scales with dwell time
    // ------------------------------------------------------------------
    #[test]
    fn test_boost_clicked_scales_with_dwell() {
        let mut fl = SemanticFeedbackLoop::new();
        // dwell = 1000 ms → delta = 0.3 * (1 + 1.0) = 0.6
        fl.record_feedback(1, clicked(55, 0, 1000), 0);
        let boost = fl.boosts[&55].boost_score;
        assert!((boost - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_boost_clicked_zero_dwell() {
        let mut fl = SemanticFeedbackLoop::new();
        // dwell = 0 ms → delta = 0.3 * (1 + 0.0) = 0.3
        fl.record_feedback(1, clicked(66, 0, 0), 0);
        let boost = fl.boosts[&66].boost_score;
        assert!((boost - 0.3).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // Test 17: stats totals are correct across mixed queries
    // ------------------------------------------------------------------
    #[test]
    fn test_stats_totals_across_queries() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record_feedback(1, relevant(10, 0), 100);
        fl.record_feedback(2, irrelevant(20, 0), 200);
        fl.record_feedback(3, clicked(30, 0, 500), 300);
        fl.record_feedback(1, relevant(40, 1), 400); // 2nd signal for query 1

        let s = fl.stats();
        assert_eq!(s.total_queries, 3);
        assert_eq!(s.total_signals, 4);
        assert_eq!(s.relevant_count, 2);
        assert_eq!(s.irrelevant_count, 1);
        assert_eq!(s.clicked_count, 1);
    }

    // ------------------------------------------------------------------
    // Test 18: positive_ids_for_query — unknown query returns empty
    // ------------------------------------------------------------------
    #[test]
    fn test_positive_ids_for_query_unknown() {
        let fl = SemanticFeedbackLoop::new();
        assert!(fl.positive_ids_for_query(9999).is_empty());
    }

    // ------------------------------------------------------------------
    // Test 19: positive_ids_for_query — returns correct ids
    // ------------------------------------------------------------------
    #[test]
    fn test_positive_ids_for_query_known() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record_feedback(7, relevant(100, 0), 0);
        fl.record_feedback(7, irrelevant(200, 1), 0);
        fl.record_feedback(7, clicked(300, 2, 250), 0);

        let ids = fl.positive_ids_for_query(7);
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&100));
        assert!(ids.contains(&300));
        assert!(!ids.contains(&200));
    }

    // ------------------------------------------------------------------
    // Test 20: query_id_for produces consistent FNV-1a hashes
    // ------------------------------------------------------------------
    #[test]
    fn test_query_id_for_deterministic() {
        let id1 = query_id_for("rust semantic search");
        let id2 = query_id_for("rust semantic search");
        let id3 = query_id_for("different query");
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    // ------------------------------------------------------------------
    // Test 21: apply_boosts with Irrelevant reduces score
    // ------------------------------------------------------------------
    #[test]
    fn test_apply_boosts_irrelevant_reduces_score() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record_feedback(1, irrelevant(77, 0), 0);
        // effective_boost = -0.5, multiplier = 1 + max(-0.5, -0.9) = 0.5
        let results = vec![(77u64, 1.0)];
        let boosted = fl.apply_boosts(&results);
        assert!((boosted[0].1 - 0.5).abs() < 1e-10);
    }

    // ======================================================================
    // User-facing feedback entry API tests (22–50+)
    // ======================================================================

    // Test 22: record adds an entry
    #[test]
    fn test_record_adds_entry() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record("q1", "d1", FeedbackType::Relevant, 0.9);
        assert_eq!(fl.feedback_count(), 1);
    }

    // Test 23: record multiple entries
    #[test]
    fn test_record_multiple_entries() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record("q1", "d1", FeedbackType::Relevant, 0.9);
        fl.record("q1", "d2", FeedbackType::Irrelevant, 0.8);
        fl.record("q2", "d3", FeedbackType::PartiallyRelevant, 0.5);
        assert_eq!(fl.feedback_count(), 3);
    }

    // Test 24: get_summary aggregation
    #[test]
    fn test_get_summary_aggregation() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record("q1", "d1", FeedbackType::Relevant, 1.0);
        fl.record("q1", "d2", FeedbackType::Irrelevant, 0.5);
        fl.record("q1", "d3", FeedbackType::PartiallyRelevant, 0.8);

        let s = fl.get_summary("q1").expect("summary should exist");
        assert_eq!(s.relevant_count, 1);
        assert_eq!(s.irrelevant_count, 1);
        assert_eq!(s.partial_count, 1);
        // precision = 1 / (1+1) = 0.5
        assert!((s.precision - 0.5).abs() < 1e-10);
        // avg confidence = (1.0 + 0.5 + 0.8) / 3
        assert!((s.avg_confidence - (1.0 + 0.5 + 0.8) / 3.0).abs() < 1e-10);
    }

    // Test 25: get_summary returns None for unknown query
    #[test]
    fn test_get_summary_unknown_query() {
        let fl = SemanticFeedbackLoop::new();
        assert!(fl.get_summary("nonexistent").is_none());
    }

    // Test 26: precision_at_query
    #[test]
    fn test_precision_at_query() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record("q1", "d1", FeedbackType::Relevant, 0.9);
        fl.record("q1", "d2", FeedbackType::Relevant, 0.8);
        fl.record("q1", "d3", FeedbackType::Irrelevant, 0.7);
        // precision = 2 / (2+1) = 2/3
        let p = fl.precision_at_query("q1").expect("should have precision");
        assert!((p - 2.0 / 3.0).abs() < 1e-10);
    }

    // Test 27: precision_at_query — only partial entries gives None
    #[test]
    fn test_precision_at_query_partial_only() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record("q1", "d1", FeedbackType::PartiallyRelevant, 0.5);
        assert!(fl.precision_at_query("q1").is_none());
    }

    // Test 28: precision_at_query — unknown query
    #[test]
    fn test_precision_at_query_unknown() {
        let fl = SemanticFeedbackLoop::new();
        assert!(fl.precision_at_query("missing").is_none());
    }

    // Test 29: overall_precision
    #[test]
    fn test_overall_precision() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record("q1", "d1", FeedbackType::Relevant, 0.9);
        fl.record("q1", "d2", FeedbackType::Irrelevant, 0.8);
        fl.record("q2", "d3", FeedbackType::Relevant, 0.7);
        // overall = 2 / (2+1) = 2/3
        let p = fl
            .overall_precision()
            .expect("should have overall precision");
        assert!((p - 2.0 / 3.0).abs() < 1e-10);
    }

    // Test 30: overall_precision None on empty
    #[test]
    fn test_overall_precision_empty() {
        let fl = SemanticFeedbackLoop::new();
        assert!(fl.overall_precision().is_none());
    }

    // Test 31: overall_precision None on partial-only
    #[test]
    fn test_overall_precision_partial_only() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record("q1", "d1", FeedbackType::PartiallyRelevant, 0.5);
        assert!(fl.overall_precision().is_none());
    }

    // Test 32: relevant_docs
    #[test]
    fn test_relevant_docs() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record("q1", "d1", FeedbackType::Relevant, 0.9);
        fl.record("q1", "d2", FeedbackType::Irrelevant, 0.8);
        fl.record("q1", "d3", FeedbackType::Relevant, 0.7);

        let docs = fl.relevant_docs("q1");
        assert_eq!(docs.len(), 2);
        assert!(docs.contains(&"d1".to_string()));
        assert!(docs.contains(&"d3".to_string()));
    }

    // Test 33: irrelevant_docs
    #[test]
    fn test_irrelevant_docs() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record("q1", "d1", FeedbackType::Relevant, 0.9);
        fl.record("q1", "d2", FeedbackType::Irrelevant, 0.8);
        fl.record("q1", "d3", FeedbackType::Irrelevant, 0.7);

        let docs = fl.irrelevant_docs("q1");
        assert_eq!(docs.len(), 2);
        assert!(docs.contains(&"d2".to_string()));
        assert!(docs.contains(&"d3".to_string()));
    }

    // Test 34: relevant_docs empty for unknown query
    #[test]
    fn test_relevant_docs_unknown() {
        let fl = SemanticFeedbackLoop::new();
        assert!(fl.relevant_docs("nope").is_empty());
    }

    // Test 35: max_entries eviction
    #[test]
    fn test_max_entries_eviction() {
        let mut fl = SemanticFeedbackLoop::with_max_entries(3);
        fl.record("q1", "d1", FeedbackType::Relevant, 1.0);
        fl.record("q1", "d2", FeedbackType::Relevant, 1.0);
        fl.record("q1", "d3", FeedbackType::Relevant, 1.0);
        assert_eq!(fl.feedback_count(), 3);

        // This should evict d1
        fl.record("q1", "d4", FeedbackType::Relevant, 1.0);
        assert_eq!(fl.feedback_count(), 3);

        let docs = fl.relevant_docs("q1");
        assert!(!docs.contains(&"d1".to_string()));
        assert!(docs.contains(&"d4".to_string()));
    }

    // Test 36: clear_entries
    #[test]
    fn test_clear_entries() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record("q1", "d1", FeedbackType::Relevant, 0.9);
        fl.record("q2", "d2", FeedbackType::Irrelevant, 0.8);
        assert_eq!(fl.feedback_count(), 2);

        fl.clear_entries();
        assert_eq!(fl.feedback_count(), 0);
        assert!(fl.queries_with_feedback().is_empty());
    }

    // Test 37: queries_with_feedback returns unique query IDs
    #[test]
    fn test_queries_with_feedback() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record("q1", "d1", FeedbackType::Relevant, 0.9);
        fl.record("q2", "d2", FeedbackType::Irrelevant, 0.8);
        fl.record("q1", "d3", FeedbackType::Relevant, 0.7); // duplicate q1

        let queries = fl.queries_with_feedback();
        assert_eq!(queries.len(), 2);
        assert!(queries.contains(&"q1".to_string()));
        assert!(queries.contains(&"q2".to_string()));
    }

    // Test 38: loop_stats accuracy
    #[test]
    fn test_loop_stats_accuracy() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record("q1", "d1", FeedbackType::Relevant, 1.0);
        fl.record("q1", "d2", FeedbackType::Irrelevant, 0.5);
        fl.record("q2", "d3", FeedbackType::Relevant, 0.8);

        let s = fl.loop_stats();
        assert_eq!(s.total_entries, 3);
        assert_eq!(s.unique_queries, 2);
        // overall precision = 2 / (2+1) = 2/3
        let p = s.overall_precision.expect("should have precision");
        assert!((p - 2.0 / 3.0).abs() < 1e-10);
        // avg confidence = (1.0 + 0.5 + 0.8) / 3
        assert!((s.avg_confidence - (1.0 + 0.5 + 0.8) / 3.0).abs() < 1e-10);
    }

    // Test 39: loop_stats on empty loop
    #[test]
    fn test_loop_stats_empty() {
        let fl = SemanticFeedbackLoop::new();
        let s = fl.loop_stats();
        assert_eq!(s.total_entries, 0);
        assert_eq!(s.unique_queries, 0);
        assert!(s.overall_precision.is_none());
        assert!((s.avg_confidence - 0.0).abs() < 1e-10);
    }

    // Test 40: tick advances counter
    #[test]
    fn test_tick_advances_counter() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record("q1", "d1", FeedbackType::Relevant, 1.0);
        let tick_before = fl.entries.last().map(|e| e.tick);

        fl.tick();
        fl.tick();
        fl.record("q1", "d2", FeedbackType::Relevant, 1.0);
        let tick_after = fl.entries.last().map(|e| e.tick);

        // record increments tick internally, but we also called tick() twice
        // first record: tick=0, then current_tick=1
        // tick(): current_tick=2, tick(): current_tick=3
        // second record: tick=3, then current_tick=4
        assert_eq!(tick_before, Some(0));
        assert_eq!(tick_after, Some(3));
    }

    // Test 41: confidence is clamped to [0, 1]
    #[test]
    fn test_confidence_clamped() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record("q1", "d1", FeedbackType::Relevant, 2.0);
        fl.record("q1", "d2", FeedbackType::Relevant, -1.0);

        let s = fl.get_summary("q1").expect("summary should exist");
        // (1.0 + 0.0) / 2 = 0.5
        assert!((s.avg_confidence - 0.5).abs() < 1e-10);
    }

    // Test 42: partial relevance does not affect precision
    #[test]
    fn test_partial_not_in_precision() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record("q1", "d1", FeedbackType::Relevant, 0.9);
        fl.record("q1", "d2", FeedbackType::PartiallyRelevant, 0.8);

        // precision = 1 / (1+0) = 1.0 (partial is excluded)
        let p = fl.precision_at_query("q1").expect("should have precision");
        assert!((p - 1.0).abs() < 1e-10);
    }

    // Test 43: multiple queries precision independence
    #[test]
    fn test_multiple_queries_precision_independence() {
        let mut fl = SemanticFeedbackLoop::new();
        // q1: 1 relevant, 1 irrelevant → 0.5
        fl.record("q1", "d1", FeedbackType::Relevant, 0.9);
        fl.record("q1", "d2", FeedbackType::Irrelevant, 0.8);
        // q2: all relevant → 1.0
        fl.record("q2", "d3", FeedbackType::Relevant, 0.7);
        fl.record("q2", "d4", FeedbackType::Relevant, 0.6);

        let p1 = fl.precision_at_query("q1").expect("q1 precision");
        let p2 = fl.precision_at_query("q2").expect("q2 precision");
        assert!((p1 - 0.5).abs() < 1e-10);
        assert!((p2 - 1.0).abs() < 1e-10);
    }

    // Test 44: eviction preserves newest entries
    #[test]
    fn test_eviction_preserves_newest() {
        let mut fl = SemanticFeedbackLoop::with_max_entries(2);
        fl.record("q1", "old", FeedbackType::Relevant, 1.0);
        fl.record("q1", "mid", FeedbackType::Relevant, 1.0);
        fl.record("q1", "new", FeedbackType::Relevant, 1.0);

        let docs = fl.relevant_docs("q1");
        assert_eq!(docs.len(), 2);
        assert!(!docs.contains(&"old".to_string()));
        assert!(docs.contains(&"mid".to_string()));
        assert!(docs.contains(&"new".to_string()));
    }

    // Test 45: FeedbackType equality
    #[test]
    fn test_feedback_type_equality() {
        assert_eq!(FeedbackType::Relevant, FeedbackType::Relevant);
        assert_ne!(FeedbackType::Relevant, FeedbackType::Irrelevant);
        assert_ne!(FeedbackType::Irrelevant, FeedbackType::PartiallyRelevant);
    }

    // Test 46: get_summary precision with all relevant
    #[test]
    fn test_summary_all_relevant() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record("q1", "d1", FeedbackType::Relevant, 1.0);
        fl.record("q1", "d2", FeedbackType::Relevant, 0.9);

        let s = fl.get_summary("q1").expect("summary");
        assert!((s.precision - 1.0).abs() < 1e-10);
    }

    // Test 47: get_summary precision with all irrelevant
    #[test]
    fn test_summary_all_irrelevant() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record("q1", "d1", FeedbackType::Irrelevant, 0.9);
        fl.record("q1", "d2", FeedbackType::Irrelevant, 0.8);

        let s = fl.get_summary("q1").expect("summary");
        assert!((s.precision - 0.0).abs() < 1e-10);
    }

    // Test 48: with_max_entries constructor
    #[test]
    fn test_with_max_entries_constructor() {
        let fl = SemanticFeedbackLoop::with_max_entries(100);
        assert_eq!(fl.feedback_count(), 0);
        assert_eq!(fl.max_entries, 100);
    }

    // Test 49: irrelevant_docs for unknown query
    #[test]
    fn test_irrelevant_docs_unknown() {
        let fl = SemanticFeedbackLoop::new();
        assert!(fl.irrelevant_docs("nope").is_empty());
    }

    // Test 50: large-scale eviction maintains count
    #[test]
    fn test_large_scale_eviction() {
        let mut fl = SemanticFeedbackLoop::with_max_entries(10);
        for i in 0..100 {
            fl.record("q1", &format!("d{}", i), FeedbackType::Relevant, 0.5);
        }
        assert_eq!(fl.feedback_count(), 10);

        // Only the last 10 docs should remain (d90..d99)
        let docs = fl.relevant_docs("q1");
        assert_eq!(docs.len(), 10);
        assert!(docs.contains(&"d90".to_string()));
        assert!(docs.contains(&"d99".to_string()));
        assert!(!docs.contains(&"d0".to_string()));
    }

    // Test 51: clear_entries does not affect signal-based stats
    #[test]
    fn test_clear_entries_does_not_affect_signals() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record_feedback(
            1,
            FeedbackSignal::Relevant {
                result_id: 10,
                rank: 0,
            },
            0,
        );
        fl.record("q1", "d1", FeedbackType::Relevant, 0.9);

        fl.clear_entries();
        // Signal-based stats remain intact
        assert_eq!(fl.stats().relevant_count, 1);
        // But entries are gone
        assert_eq!(fl.feedback_count(), 0);
    }

    // Test 52: summary with only partial entries has precision 0.0
    #[test]
    fn test_summary_partial_only_precision_zero() {
        let mut fl = SemanticFeedbackLoop::new();
        fl.record("q1", "d1", FeedbackType::PartiallyRelevant, 0.6);
        fl.record("q1", "d2", FeedbackType::PartiallyRelevant, 0.4);

        let s = fl.get_summary("q1").expect("summary");
        assert_eq!(s.partial_count, 2);
        assert_eq!(s.relevant_count, 0);
        assert_eq!(s.irrelevant_count, 0);
        assert!((s.precision - 0.0).abs() < 1e-10);
    }

    // Test 53: default constructor uses 50_000 max_entries
    #[test]
    fn test_default_max_entries() {
        let fl = SemanticFeedbackLoop::new();
        assert_eq!(fl.max_entries, 50_000);
    }

    // Test 54: FeedbackLoopStats clone and debug
    #[test]
    fn test_feedback_loop_stats_clone_debug() {
        let s = FeedbackLoopStats {
            total_entries: 10,
            unique_queries: 3,
            overall_precision: Some(0.75),
            avg_confidence: 0.85,
        };
        let s2 = s.clone();
        assert_eq!(s2.total_entries, 10);
        let _ = format!("{:?}", s2);
    }

    // Test 55: FeedbackEntry clone
    #[test]
    fn test_feedback_entry_clone() {
        let e = FeedbackEntry {
            query_id: "q1".to_string(),
            doc_id: "d1".to_string(),
            feedback: FeedbackType::Relevant,
            tick: 42,
            confidence: 0.95,
        };
        let e2 = e.clone();
        assert_eq!(e2.query_id, "q1");
        assert_eq!(e2.tick, 42);
    }

    // Test 56: QueryFeedbackSummary clone
    #[test]
    fn test_query_feedback_summary_clone() {
        let s = QueryFeedbackSummary {
            query_id: "q1".to_string(),
            relevant_count: 5,
            irrelevant_count: 2,
            partial_count: 1,
            avg_confidence: 0.8,
            precision: 5.0 / 7.0,
        };
        let s2 = s.clone();
        assert_eq!(s2.relevant_count, 5);
        assert!((s2.precision - 5.0 / 7.0).abs() < 1e-10);
    }
}
