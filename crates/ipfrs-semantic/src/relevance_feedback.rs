//! Rocchio-style semantic relevance feedback for iterative query refinement.
//!
//! This module implements the classic Rocchio algorithm adapted for dense vector embeddings.
//! Users mark search results as relevant or non-relevant; the algorithm shifts the query
//! embedding toward relevant documents and away from non-relevant ones.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Classification label assigned to a retrieved document by the user.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeedbackLabel {
    /// Document is relevant to the user's information need.
    Relevant,
    /// Document is not relevant to the user's information need.
    NonRelevant,
    /// Document is ignored in the Rocchio computation.
    Neutral,
}

/// A single feedback item pairing a document with its user-assigned label.
#[derive(Clone, Debug)]
pub struct FeedbackItem {
    /// Opaque document identifier.
    pub doc_id: u64,
    /// Dense embedding vector for this document.
    pub embedding: Vec<f32>,
    /// User-assigned relevance label.
    pub label: FeedbackLabel,
}

/// Rocchio algorithm hyper-parameters.
#[derive(Clone, Debug)]
pub struct RocchioConfig {
    /// Weight applied to the current query vector (α).
    pub alpha: f32,
    /// Weight applied to the centroid of relevant documents (β).
    pub beta: f32,
    /// Weight applied to the centroid of non-relevant documents (γ).
    pub gamma: f32,
    /// When `true` the updated query is L2-normalised before being stored.
    pub normalize_result: bool,
}

impl Default for RocchioConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 0.75,
            gamma: 0.25,
            normalize_result: true,
        }
    }
}

/// State maintained for a single interactive feedback session.
#[derive(Clone, Debug)]
pub struct FeedbackSession {
    /// Unique identifier for this session.
    pub session_id: u64,
    /// The query vector supplied when the session was created.
    pub original_query: Vec<f32>,
    /// The query vector after the most recent Rocchio update (or equal to
    /// `original_query` when no rounds have been applied yet).
    pub current_query: Vec<f32>,
    /// Number of times [`SemanticRelevanceFeedback::apply_rocchio`] has been
    /// called successfully for this session.
    pub rounds: u32,
    /// All feedback items accumulated across every round.
    pub feedback_items: Vec<FeedbackItem>,
}

impl FeedbackSession {
    /// Number of items labelled [`FeedbackLabel::Relevant`].
    pub fn relevant_count(&self) -> usize {
        self.feedback_items
            .iter()
            .filter(|i| i.label == FeedbackLabel::Relevant)
            .count()
    }

    /// Number of items labelled [`FeedbackLabel::NonRelevant`].
    pub fn non_relevant_count(&self) -> usize {
        self.feedback_items
            .iter()
            .filter(|i| i.label == FeedbackLabel::NonRelevant)
            .count()
    }

    /// Cosine distance (1 − cosine_similarity) between the original query and
    /// the current query.  Returns `0.0` when no rounds have been applied.
    pub fn query_shift(&self) -> f32 {
        1.0 - cosine_similarity(&self.original_query, &self.current_query)
    }
}

/// Aggregate statistics across all sessions managed by a
/// [`SemanticRelevanceFeedback`] instance.
#[derive(Clone, Debug)]
pub struct FeedbackStats {
    /// Number of sessions ever created.
    pub total_sessions: usize,
    /// Total Rocchio rounds applied across all sessions.
    pub total_rounds: u64,
    /// Average number of rounds per session (`NaN` when `total_sessions == 0`).
    pub avg_rounds_per_session: f64,
}

// ---------------------------------------------------------------------------
// Core engine
// ---------------------------------------------------------------------------

/// Engine that manages multiple interactive relevance-feedback sessions and
/// applies the Rocchio algorithm to refine query embeddings.
pub struct SemanticRelevanceFeedback {
    /// Active and completed sessions keyed by session id.
    pub sessions: HashMap<u64, FeedbackSession>,
    /// Counter used to assign the next session id (monotonically increasing).
    pub next_session_id: u64,
    /// Rocchio hyper-parameters shared across all sessions.
    pub config: RocchioConfig,
    /// Running total of Rocchio rounds applied across all sessions.
    pub total_rounds: u64,
}

impl SemanticRelevanceFeedback {
    /// Create a new engine with the given configuration.
    pub fn new(config: RocchioConfig) -> Self {
        Self {
            sessions: HashMap::new(),
            next_session_id: 1,
            config,
            total_rounds: 0,
        }
    }

    /// Start a new feedback session for `query`.
    ///
    /// Returns the freshly allocated session id.
    pub fn create_session(&mut self, query: Vec<f32>) -> u64 {
        let id = self.next_session_id;
        self.next_session_id += 1;
        let session = FeedbackSession {
            session_id: id,
            original_query: query.clone(),
            current_query: query,
            rounds: 0,
            feedback_items: Vec::new(),
        };
        self.sessions.insert(id, session);
        id
    }

    /// Append `item` to the feedback list of the session identified by
    /// `session_id`.
    ///
    /// Returns `false` when no such session exists.
    pub fn add_feedback(&mut self, session_id: u64, item: FeedbackItem) -> bool {
        match self.sessions.get_mut(&session_id) {
            Some(session) => {
                session.feedback_items.push(item);
                true
            }
            None => false,
        }
    }

    /// Apply one Rocchio update round to the session identified by
    /// `session_id`.
    ///
    /// The formula is:
    /// ```text
    /// new_query = α·q + β·centroid(R) − γ·centroid(NR)
    /// ```
    /// where the β or γ terms are omitted when the corresponding document set
    /// is empty.
    ///
    /// Returns `Some(new_query)` on success, or `None` when `session_id` is
    /// unknown.
    pub fn apply_rocchio(&mut self, session_id: u64) -> Option<Vec<f32>> {
        let session = self.sessions.get_mut(&session_id)?;

        let dim = session.current_query.len();
        if dim == 0 {
            return Some(Vec::new());
        }

        // --- Collect relevant / non-relevant embeddings ---
        let relevant: Vec<&Vec<f32>> = session
            .feedback_items
            .iter()
            .filter(|i| i.label == FeedbackLabel::Relevant)
            .map(|i| &i.embedding)
            .collect();

        let non_relevant: Vec<&Vec<f32>> = session
            .feedback_items
            .iter()
            .filter(|i| i.label == FeedbackLabel::NonRelevant)
            .map(|i| &i.embedding)
            .collect();

        // --- Compute centroids ---
        let relevant_centroid = compute_centroid(&relevant, dim);
        let non_relevant_centroid = compute_centroid(&non_relevant, dim);

        // --- Rocchio update ---
        let alpha = self.config.alpha;
        let beta = self.config.beta;
        let gamma = self.config.gamma;
        let normalize = self.config.normalize_result;

        let mut new_query: Vec<f32> = session.current_query.iter().map(|&q| alpha * q).collect();

        // β term — skipped when there are no relevant docs (centroid is all zeros)
        if !relevant.is_empty() {
            for (nq, rc) in new_query.iter_mut().zip(relevant_centroid.iter()) {
                *nq += beta * rc;
            }
        }

        // γ term — skipped when there are no non-relevant docs
        if !non_relevant.is_empty() {
            for (nq, nc) in new_query.iter_mut().zip(non_relevant_centroid.iter()) {
                *nq -= gamma * nc;
            }
        }

        // --- Optional L2 normalisation ---
        if normalize {
            l2_normalize(&mut new_query);
        }

        // --- Persist updates ---
        session.current_query = new_query.clone();
        session.rounds += 1;
        self.total_rounds += 1;

        Some(new_query)
    }

    /// Retrieve an immutable reference to a session.
    pub fn session(&self, session_id: u64) -> Option<&FeedbackSession> {
        self.sessions.get(&session_id)
    }

    /// Compute aggregate statistics across all sessions.
    pub fn stats(&self) -> FeedbackStats {
        let total_sessions = self.sessions.len();
        let total_rounds = self.total_rounds;
        let avg_rounds_per_session = if total_sessions == 0 {
            f64::NAN
        } else {
            total_rounds as f64 / total_sessions as f64
        };
        FeedbackStats {
            total_sessions,
            total_rounds,
            avg_rounds_per_session,
        }
    }
}

// ---------------------------------------------------------------------------
// Mathematical helpers
// ---------------------------------------------------------------------------

/// Compute the element-wise mean of `vecs`.  Returns a zero vector of length
/// `dim` when `vecs` is empty.
fn compute_centroid(vecs: &[&Vec<f32>], dim: usize) -> Vec<f32> {
    if vecs.is_empty() {
        return vec![0.0f32; dim];
    }
    let n = vecs.len() as f32;
    let mut centroid = vec![0.0f32; dim];
    for v in vecs {
        for (c, &x) in centroid.iter_mut().zip(v.iter()) {
            *c += x;
        }
    }
    for c in centroid.iter_mut() {
        *c /= n;
    }
    centroid
}

/// L2-normalise `v` in-place.  A zero vector is left unchanged.
fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity between two vectors.  Returns `1.0` when either vector
/// has zero magnitude (treats them as identical to avoid division by zero).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 1.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if mag_a < f32::EPSILON || mag_b < f32::EPSILON {
        return 1.0;
    }
    (dot / (mag_a * mag_b)).clamp(-1.0, 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_engine() -> SemanticRelevanceFeedback {
        SemanticRelevanceFeedback::new(RocchioConfig::default())
    }

    fn unit_vec(dim: usize, idx: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[idx] = 1.0;
        v
    }

    // 1. new() starts empty
    #[test]
    fn test_new_starts_empty() {
        let engine = default_engine();
        assert!(engine.sessions.is_empty());
        assert_eq!(engine.total_rounds, 0);
        assert_eq!(engine.next_session_id, 1);
    }

    // 2. create_session returns incrementing ids
    #[test]
    fn test_create_session_incrementing_ids() {
        let mut engine = default_engine();
        let id1 = engine.create_session(vec![1.0, 0.0]);
        let id2 = engine.create_session(vec![0.0, 1.0]);
        let id3 = engine.create_session(vec![0.5, 0.5]);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
    }

    // 3. create_session stores original and current query
    #[test]
    fn test_create_session_stores_queries() {
        let mut engine = default_engine();
        let q = vec![0.3f32, 0.7, 0.0];
        let id = engine.create_session(q.clone());
        let session = engine.session(id).expect("session must exist");
        assert_eq!(session.original_query, q);
        assert_eq!(session.current_query, q);
    }

    // 4. add_feedback appends item
    #[test]
    fn test_add_feedback_appends() {
        let mut engine = default_engine();
        let id = engine.create_session(vec![1.0, 0.0]);
        let item = FeedbackItem {
            doc_id: 42,
            embedding: vec![0.8, 0.2],
            label: FeedbackLabel::Relevant,
        };
        assert!(engine.add_feedback(id, item));
        let session = engine.session(id).expect("test: session must exist");
        assert_eq!(session.feedback_items.len(), 1);
        assert_eq!(session.feedback_items[0].doc_id, 42);
    }

    // 5. add_feedback returns false for unknown session
    #[test]
    fn test_add_feedback_unknown_session() {
        let mut engine = default_engine();
        let item = FeedbackItem {
            doc_id: 1,
            embedding: vec![1.0],
            label: FeedbackLabel::Relevant,
        };
        assert!(!engine.add_feedback(999, item));
    }

    // 6. apply_rocchio updates current_query
    #[test]
    fn test_apply_rocchio_updates_current_query() {
        let mut engine = SemanticRelevanceFeedback::new(RocchioConfig {
            normalize_result: false,
            ..RocchioConfig::default()
        });
        let id = engine.create_session(vec![1.0, 0.0]);
        engine.add_feedback(
            id,
            FeedbackItem {
                doc_id: 1,
                embedding: vec![0.0, 1.0],
                label: FeedbackLabel::Relevant,
            },
        );
        let new_q = engine.apply_rocchio(id).expect("must return Some");
        let session = engine
            .session(id)
            .expect("test: session must exist after rocchio");
        assert_eq!(session.current_query, new_q);
        // alpha*[1,0] + beta*[0,1] = [1.0, 0.75]
        assert!((new_q[0] - 1.0).abs() < 1e-5);
        assert!((new_q[1] - 0.75).abs() < 1e-5);
    }

    // 7. apply_rocchio with no relevant docs (beta skipped)
    #[test]
    fn test_apply_rocchio_no_relevant_docs() {
        let mut engine = SemanticRelevanceFeedback::new(RocchioConfig {
            normalize_result: false,
            ..RocchioConfig::default()
        });
        let id = engine.create_session(vec![1.0, 0.0]);
        engine.add_feedback(
            id,
            FeedbackItem {
                doc_id: 1,
                embedding: vec![0.0, 1.0],
                label: FeedbackLabel::NonRelevant,
            },
        );
        let new_q = engine.apply_rocchio(id).expect("Some");
        // alpha*[1,0] - gamma*[0,1] = [1.0, -0.25]
        assert!((new_q[0] - 1.0).abs() < 1e-5);
        assert!((new_q[1] - (-0.25)).abs() < 1e-5);
    }

    // 8. apply_rocchio with no non-relevant docs (gamma skipped)
    #[test]
    fn test_apply_rocchio_no_non_relevant_docs() {
        let mut engine = SemanticRelevanceFeedback::new(RocchioConfig {
            normalize_result: false,
            ..RocchioConfig::default()
        });
        let id = engine.create_session(vec![1.0, 0.0]);
        engine.add_feedback(
            id,
            FeedbackItem {
                doc_id: 1,
                embedding: vec![0.0, 1.0],
                label: FeedbackLabel::Relevant,
            },
        );
        let new_q = engine.apply_rocchio(id).expect("Some");
        // alpha*[1,0] + beta*[0,1] = [1.0, 0.75]
        assert!((new_q[0] - 1.0).abs() < 1e-5);
        assert!((new_q[1] - 0.75).abs() < 1e-5);
    }

    // 9. apply_rocchio increments rounds
    #[test]
    fn test_apply_rocchio_increments_rounds() {
        let mut engine = default_engine();
        let id = engine.create_session(vec![1.0, 0.0]);
        assert_eq!(
            engine
                .session(id)
                .expect("test: session must exist before round")
                .rounds,
            0
        );
        engine.apply_rocchio(id);
        assert_eq!(
            engine
                .session(id)
                .expect("test: session must exist after round 1")
                .rounds,
            1
        );
        engine.apply_rocchio(id);
        assert_eq!(
            engine
                .session(id)
                .expect("test: session must exist after round 2")
                .rounds,
            2
        );
    }

    // 10. apply_rocchio returns None for unknown session
    #[test]
    fn test_apply_rocchio_none_for_unknown() {
        let mut engine = default_engine();
        assert!(engine.apply_rocchio(42).is_none());
    }

    // 11. relevant_count correct
    #[test]
    fn test_relevant_count() {
        let mut engine = default_engine();
        let id = engine.create_session(vec![1.0]);
        for _ in 0..3 {
            engine.add_feedback(
                id,
                FeedbackItem {
                    doc_id: 0,
                    embedding: vec![1.0],
                    label: FeedbackLabel::Relevant,
                },
            );
        }
        engine.add_feedback(
            id,
            FeedbackItem {
                doc_id: 1,
                embedding: vec![1.0],
                label: FeedbackLabel::NonRelevant,
            },
        );
        assert_eq!(
            engine
                .session(id)
                .expect("test: session must exist")
                .relevant_count(),
            3
        );
    }

    // 12. non_relevant_count correct
    #[test]
    fn test_non_relevant_count() {
        let mut engine = default_engine();
        let id = engine.create_session(vec![1.0]);
        for _ in 0..2 {
            engine.add_feedback(
                id,
                FeedbackItem {
                    doc_id: 0,
                    embedding: vec![1.0],
                    label: FeedbackLabel::NonRelevant,
                },
            );
        }
        assert_eq!(
            engine
                .session(id)
                .expect("test: session must exist")
                .non_relevant_count(),
            2
        );
    }

    // 13. Neutral items not counted in relevant/non_relevant
    #[test]
    fn test_neutral_not_counted() {
        let mut engine = default_engine();
        let id = engine.create_session(vec![1.0, 0.0]);
        engine.add_feedback(
            id,
            FeedbackItem {
                doc_id: 7,
                embedding: vec![0.5, 0.5],
                label: FeedbackLabel::Neutral,
            },
        );
        let session = engine.session(id).expect("test: session must exist");
        assert_eq!(session.relevant_count(), 0);
        assert_eq!(session.non_relevant_count(), 0);
    }

    // 14. Neutral items not used in Rocchio computation
    #[test]
    fn test_neutral_not_used_in_rocchio() {
        let mut engine = SemanticRelevanceFeedback::new(RocchioConfig {
            normalize_result: false,
            ..RocchioConfig::default()
        });
        let id = engine.create_session(vec![1.0, 0.0]);
        // Add only Neutral — result should be alpha * original_query
        engine.add_feedback(
            id,
            FeedbackItem {
                doc_id: 99,
                embedding: vec![0.0, 1.0],
                label: FeedbackLabel::Neutral,
            },
        );
        let new_q = engine.apply_rocchio(id).expect("Some");
        // alpha * [1.0, 0.0] = [1.0, 0.0]
        assert!((new_q[0] - 1.0).abs() < 1e-5);
        assert!((new_q[1] - 0.0).abs() < 1e-5);
    }

    // 15. normalize_result=true produces unit vector
    #[test]
    fn test_normalize_true_produces_unit_vector() {
        let mut engine = SemanticRelevanceFeedback::new(RocchioConfig {
            normalize_result: true,
            ..RocchioConfig::default()
        });
        let id = engine.create_session(vec![1.0, 0.0]);
        engine.add_feedback(
            id,
            FeedbackItem {
                doc_id: 1,
                embedding: vec![0.0, 1.0],
                label: FeedbackLabel::Relevant,
            },
        );
        let new_q = engine.apply_rocchio(id).expect("Some");
        let norm: f32 = new_q.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm={norm}");
    }

    // 16. normalize_result=false keeps raw result
    #[test]
    fn test_normalize_false_keeps_raw() {
        let mut engine = SemanticRelevanceFeedback::new(RocchioConfig {
            normalize_result: false,
            ..RocchioConfig::default()
        });
        let id = engine.create_session(vec![1.0, 0.0]);
        engine.add_feedback(
            id,
            FeedbackItem {
                doc_id: 1,
                embedding: vec![0.0, 1.0],
                label: FeedbackLabel::Relevant,
            },
        );
        let new_q = engine.apply_rocchio(id).expect("Some");
        // [1.0, 0.75] — norm ≠ 1
        let norm: f32 = new_q.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() > 1e-3,
            "norm should not be 1.0, got {norm}"
        );
    }

    // 17. query_shift is 0.0 when no rounds applied
    #[test]
    fn test_query_shift_zero_no_rounds() {
        let mut engine = default_engine();
        let id = engine.create_session(vec![1.0, 0.0]);
        let shift = engine
            .session(id)
            .expect("test: session should exist for query_shift check")
            .query_shift();
        assert!(shift.abs() < 1e-5, "shift={shift}");
    }

    // 18. query_shift > 0 after Rocchio round
    #[test]
    fn test_query_shift_nonzero_after_rounds() {
        let mut engine = default_engine();
        let id = engine.create_session(unit_vec(4, 0));
        engine.add_feedback(
            id,
            FeedbackItem {
                doc_id: 1,
                embedding: unit_vec(4, 1),
                label: FeedbackLabel::Relevant,
            },
        );
        engine.apply_rocchio(id);
        let shift = engine
            .session(id)
            .expect("test: session should exist after apply_rocchio")
            .query_shift();
        assert!(
            shift > 0.0,
            "shift should be positive after round, got {shift}"
        );
    }

    // 19. multiple rounds accumulate changes
    #[test]
    fn test_multiple_rounds_accumulate() {
        let mut engine = default_engine();
        let id = engine.create_session(unit_vec(3, 0));
        engine.add_feedback(
            id,
            FeedbackItem {
                doc_id: 1,
                embedding: unit_vec(3, 2),
                label: FeedbackLabel::Relevant,
            },
        );
        engine.apply_rocchio(id);
        let shift_after_1 = engine
            .session(id)
            .expect("test: session should exist after first apply_rocchio")
            .query_shift();
        engine.apply_rocchio(id);
        let shift_after_2 = engine
            .session(id)
            .expect("test: session should exist after second apply_rocchio")
            .query_shift();
        // Each round pulls the query further toward the relevant centroid
        assert!(
            shift_after_2 > shift_after_1 || (shift_after_2 - shift_after_1).abs() < 1e-3,
            "shift should not decrease: after_1={shift_after_1}, after_2={shift_after_2}"
        );
        assert_eq!(
            engine
                .session(id)
                .expect("test: session should exist for rounds count check")
                .rounds,
            2
        );
    }

    // 20. stats total_sessions correct
    #[test]
    fn test_stats_total_sessions() {
        let mut engine = default_engine();
        engine.create_session(vec![1.0]);
        engine.create_session(vec![0.5]);
        engine.create_session(vec![0.0]);
        assert_eq!(engine.stats().total_sessions, 3);
    }

    // 21. stats total_rounds correct
    #[test]
    fn test_stats_total_rounds() {
        let mut engine = default_engine();
        let id1 = engine.create_session(vec![1.0, 0.0]);
        let id2 = engine.create_session(vec![0.0, 1.0]);
        engine.apply_rocchio(id1);
        engine.apply_rocchio(id1);
        engine.apply_rocchio(id2);
        assert_eq!(engine.stats().total_rounds, 3);
    }

    // 22. stats avg_rounds_per_session correct
    #[test]
    fn test_stats_avg_rounds_per_session() {
        let mut engine = default_engine();
        let id1 = engine.create_session(vec![1.0, 0.0]);
        let id2 = engine.create_session(vec![0.0, 1.0]);
        engine.apply_rocchio(id1);
        engine.apply_rocchio(id1);
        engine.apply_rocchio(id2);
        // 3 rounds / 2 sessions = 1.5
        let avg = engine.stats().avg_rounds_per_session;
        assert!((avg - 1.5).abs() < 1e-9, "avg={avg}");
    }
}
