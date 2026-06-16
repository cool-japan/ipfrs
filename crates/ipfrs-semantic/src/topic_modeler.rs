//! Semantic Topic Modeller — online clustering approach for latent topic modelling.
//!
//! Models latent topics from a collection of embeddings using a simple online
//! clustering approach, assigning documents to topics and tracking topic drift
//! over time.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Cosine similarity helper
// ---------------------------------------------------------------------------

/// Computes the cosine similarity between two vectors.
///
/// Returns `0.0` when either vector has zero norm.
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-9 || norm_b < 1e-9 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

// ---------------------------------------------------------------------------
// TopicModel
// ---------------------------------------------------------------------------

/// A latent topic represented as a running centroid of assigned embeddings.
#[derive(Debug, Clone)]
pub struct TopicModel {
    /// Unique identifier for this topic.
    pub topic_id: u64,
    /// Running mean of all embeddings assigned to this topic.
    pub centroid: Vec<f32>,
    /// Number of documents assigned to this topic.
    pub member_count: u64,
    /// Sum of assignment weights (cosine similarities at assignment time).
    pub total_weight: f64,
    /// Human-readable label (defaults to `"topic_<id>"`).
    pub label: String,
}

impl TopicModel {
    /// Coherence measure: `member_count / (1.0 + total_weight)`.
    ///
    /// A higher value indicates a more focused (tight) topic.
    pub fn coherence(&self) -> f64 {
        self.member_count as f64 / (1.0 + self.total_weight)
    }
}

// ---------------------------------------------------------------------------
// TopicAssignment
// ---------------------------------------------------------------------------

/// Records the assignment of a document to a topic at a point in time.
#[derive(Debug, Clone)]
pub struct TopicAssignment {
    /// Document identifier.
    pub doc_id: u64,
    /// Topic the document was assigned to.
    pub topic_id: u64,
    /// Cosine similarity to the topic centroid at assignment time.
    /// `1.0` when the document created a brand-new topic.
    pub confidence: f32,
    /// Unix timestamp (seconds) when the assignment was made.
    pub assigned_at_secs: u64,
}

// ---------------------------------------------------------------------------
// ModellerConfig
// ---------------------------------------------------------------------------

/// Configuration for [`SemanticTopicModeller`].
#[derive(Debug, Clone)]
pub struct ModellerConfig {
    /// Maximum number of distinct topics allowed.  Default: `20`.
    pub max_topics: usize,
    /// If the maximum cosine similarity between an embedding and every existing
    /// topic centroid is below this threshold, a new topic is created.  Default: `0.5`.
    pub new_topic_threshold: f32,
    /// Learning rate for the online centroid update rule:
    /// `centroid = (1 - lr) * centroid + lr * embedding`.  Default: `0.1`.
    pub centroid_learning_rate: f32,
}

impl Default for ModellerConfig {
    fn default() -> Self {
        Self {
            max_topics: 20,
            new_topic_threshold: 0.5,
            centroid_learning_rate: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// TopicModellerStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for a [`SemanticTopicModeller`].
#[derive(Debug, Clone)]
pub struct TopicModellerStats {
    /// Total number of documents that have been assigned (including re-assigns).
    pub total_documents: u64,
    /// Current number of distinct topics.
    pub total_topics: usize,
    /// Average member count across all topics.  `0.0` when there are no topics.
    pub avg_topic_size: f64,
    /// Member count of the largest topic.  `0` when there are no topics.
    pub largest_topic_members: u64,
}

// ---------------------------------------------------------------------------
// SemanticTopicModeller
// ---------------------------------------------------------------------------

/// Online topic modeller that clusters embeddings into latent topics.
///
/// New documents are assigned to the closest existing topic (by cosine
/// similarity).  If the best similarity is below [`ModellerConfig::new_topic_threshold`]
/// and the topic limit has not been reached, a fresh topic is created.
pub struct SemanticTopicModeller {
    /// All topics keyed by `topic_id`.
    pub topics: HashMap<u64, TopicModel>,
    /// All assignment records in insertion order.
    pub assignments: Vec<TopicAssignment>,
    /// Monotonically increasing counter used to allocate topic IDs.
    pub next_topic_id: u64,
    /// Modeller configuration.
    pub config: ModellerConfig,
}

impl SemanticTopicModeller {
    /// Creates a new, empty topic modeller with the given configuration.
    pub fn new(config: ModellerConfig) -> Self {
        Self {
            topics: HashMap::new(),
            assignments: Vec::new(),
            next_topic_id: 0,
            config,
        }
    }

    /// Assigns `embedding` (associated with `doc_id`) to a topic.
    ///
    /// # Algorithm
    ///
    /// 1. If there are no topics yet, always create the first topic.
    /// 2. Find the topic whose centroid has the highest cosine similarity to
    ///    `embedding`.
    /// 3. If `max_sim < new_topic_threshold` **and** the number of topics is
    ///    below `max_topics`, create a new topic with `centroid = embedding`.
    /// 4. Otherwise assign to the best topic and update its centroid via the
    ///    online update rule:
    ///    `centroid = (1 - lr) * centroid + lr * embedding`.
    ///
    /// Returns the [`TopicAssignment`] that was recorded.
    pub fn assign(&mut self, doc_id: u64, embedding: Vec<f32>, now_secs: u64) -> TopicAssignment {
        if self.topics.is_empty() {
            // Always create the very first topic.
            return self.create_topic(doc_id, embedding, now_secs);
        }

        // Find best existing topic.
        let (best_id, best_sim) = self
            .topics
            .iter()
            .map(|(id, t)| (*id, cosine_sim(&t.centroid, &embedding)))
            .fold((0u64, f32::NEG_INFINITY), |(bi, bs), (id, s)| {
                if s > bs {
                    (id, s)
                } else {
                    (bi, bs)
                }
            });

        let below_threshold = best_sim < self.config.new_topic_threshold;
        let can_create = self.topics.len() < self.config.max_topics;

        if below_threshold && can_create {
            self.create_topic(doc_id, embedding, now_secs)
        } else {
            // Assign to best existing topic and update centroid.
            let lr = self.config.centroid_learning_rate;
            let topic = self
                .topics
                .get_mut(&best_id)
                .expect("best_id must exist in topics map");

            // Online centroid update: centroid = (1-lr)*centroid + lr*embedding
            for (c, e) in topic.centroid.iter_mut().zip(embedding.iter()) {
                *c = (1.0 - lr) * (*c) + lr * e;
            }
            topic.member_count += 1;
            topic.total_weight += best_sim as f64;

            let assignment = TopicAssignment {
                doc_id,
                topic_id: best_id,
                confidence: best_sim,
                assigned_at_secs: now_secs,
            };
            self.assignments.push(assignment.clone());
            assignment
        }
    }

    /// Returns a reference to a topic by ID, or `None` if not found.
    pub fn topic(&self, topic_id: u64) -> Option<&TopicModel> {
        self.topics.get(&topic_id)
    }

    /// Returns all assignments for the given document, sorted by
    /// `assigned_at_secs` descending (most recent first).
    pub fn assignments_for_doc(&self, doc_id: u64) -> Vec<&TopicAssignment> {
        let mut result: Vec<&TopicAssignment> = self
            .assignments
            .iter()
            .filter(|a| a.doc_id == doc_id)
            .collect();
        result.sort_by_key(|b| std::cmp::Reverse(b.assigned_at_secs));
        result
    }

    /// Returns up to `k` topics sorted by `member_count` descending.
    pub fn top_topics(&self, k: usize) -> Vec<&TopicModel> {
        let mut topics: Vec<&TopicModel> = self.topics.values().collect();
        topics.sort_by_key(|b| std::cmp::Reverse(b.member_count));
        topics.truncate(k);
        topics
    }

    /// Renames a topic.  Returns `false` if the topic ID does not exist.
    pub fn relabel(&mut self, topic_id: u64, label: String) -> bool {
        match self.topics.get_mut(&topic_id) {
            Some(t) => {
                t.label = label;
                true
            }
            None => false,
        }
    }

    /// Returns aggregate statistics for the current state of the modeller.
    pub fn stats(&self) -> TopicModellerStats {
        let total_documents = self.assignments.len() as u64;
        let total_topics = self.topics.len();
        let avg_topic_size = if total_topics == 0 {
            0.0
        } else {
            let sum: u64 = self.topics.values().map(|t| t.member_count).sum();
            sum as f64 / total_topics as f64
        };
        let largest_topic_members = self
            .topics
            .values()
            .map(|t| t.member_count)
            .max()
            .unwrap_or(0);

        TopicModellerStats {
            total_documents,
            total_topics,
            avg_topic_size,
            largest_topic_members,
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Allocates a new topic ID, inserts a new [`TopicModel`], records the
    /// assignment, and returns the [`TopicAssignment`].
    fn create_topic(&mut self, doc_id: u64, embedding: Vec<f32>, now_secs: u64) -> TopicAssignment {
        let topic_id = self.next_topic_id;
        self.next_topic_id += 1;

        let model = TopicModel {
            topic_id,
            label: format!("topic_{}", topic_id),
            centroid: embedding,
            member_count: 1,
            total_weight: 1.0,
        };
        self.topics.insert(topic_id, model);

        let assignment = TopicAssignment {
            doc_id,
            topic_id,
            confidence: 1.0,
            assigned_at_secs: now_secs,
        };
        self.assignments.push(assignment.clone());
        assignment
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_modeller() -> SemanticTopicModeller {
        SemanticTopicModeller::new(ModellerConfig::default())
    }

    fn unit_vec(dim: usize, val: f32) -> Vec<f32> {
        let norm = (val * val * dim as f32).sqrt();
        if norm < 1e-9 {
            vec![0.0; dim]
        } else {
            vec![val / norm; dim]
        }
    }

    // 1. new() starts empty
    #[test]
    fn test_new_starts_empty() {
        let m = default_modeller();
        assert!(m.topics.is_empty());
        assert!(m.assignments.is_empty());
        assert_eq!(m.next_topic_id, 0);
    }

    // 2. assign first document creates first topic
    #[test]
    fn test_assign_first_document_creates_topic() {
        let mut m = default_modeller();
        let emb = unit_vec(4, 1.0);
        let a = m.assign(1, emb, 1000);
        assert_eq!(m.topics.len(), 1);
        assert_eq!(a.topic_id, 0);
        assert_eq!(a.doc_id, 1);
        assert_eq!(a.assigned_at_secs, 1000);
    }

    // 3. assign similar document joins existing topic
    #[test]
    fn test_assign_similar_joins_existing_topic() {
        let mut m = default_modeller();
        // First document.
        let emb1 = vec![1.0_f32, 0.0, 0.0, 0.0];
        m.assign(1, emb1, 100);
        // Very similar second document.
        let emb2 = vec![0.99_f32, 0.141, 0.0, 0.0];
        let a2 = m.assign(2, emb2, 200);
        assert_eq!(m.topics.len(), 1, "should still be one topic");
        assert_eq!(a2.topic_id, 0);
    }

    // 4. assign dissimilar document creates new topic
    #[test]
    fn test_assign_dissimilar_creates_new_topic() {
        let mut m = default_modeller();
        let emb1 = vec![1.0_f32, 0.0, 0.0, 0.0];
        m.assign(1, emb1, 100);
        // Orthogonal vector — similarity is 0.0, well below threshold 0.5.
        let emb2 = vec![0.0_f32, 1.0, 0.0, 0.0];
        let a2 = m.assign(2, emb2, 200);
        assert_eq!(m.topics.len(), 2);
        assert_ne!(a2.topic_id, 0);
    }

    // 5. assign at max_topics assigns to closest existing topic (never creates new)
    #[test]
    fn test_max_topics_no_new_creation() {
        let config = ModellerConfig {
            max_topics: 2,
            new_topic_threshold: 0.5,
            centroid_learning_rate: 0.1,
        };
        let mut m = SemanticTopicModeller::new(config);
        // Create exactly 2 topics with orthogonal vectors.
        m.assign(1, vec![1.0, 0.0, 0.0], 100);
        m.assign(2, vec![0.0, 1.0, 0.0], 200);
        assert_eq!(m.topics.len(), 2);

        // Completely dissimilar (also orthogonal to both), but max_topics reached.
        let a3 = m.assign(3, vec![0.0, 0.0, 1.0], 300);
        assert_eq!(m.topics.len(), 2, "no new topic should be created");
        // Should be assigned to the closest of the two existing topics.
        assert!(m.topics.contains_key(&a3.topic_id));
    }

    // 6. centroid updated via learning rate
    #[test]
    fn test_centroid_updated_via_learning_rate() {
        let config = ModellerConfig {
            max_topics: 5,
            new_topic_threshold: 0.5,
            centroid_learning_rate: 0.5,
        };
        let mut m = SemanticTopicModeller::new(config);
        // Seed with a vector.
        m.assign(1, vec![1.0, 0.0], 100);
        // Assign a similar but different vector.
        let emb2 = vec![0.8_f32, 0.6]; // sim ≈ 0.8
        m.assign(2, emb2.clone(), 200);

        let topic = m.topics.get(&0).expect("topic 0 should exist");
        // After update: centroid = 0.5 * [1,0] + 0.5 * [0.8, 0.6] = [0.9, 0.3]
        assert!((topic.centroid[0] - 0.9).abs() < 1e-5);
        assert!((topic.centroid[1] - 0.3).abs() < 1e-5);
    }

    // 7. member_count increments on join
    #[test]
    fn test_member_count_increments_on_join() {
        let mut m = default_modeller();
        m.assign(1, vec![1.0, 0.0], 100);
        // Similar — joins topic 0.
        m.assign(2, vec![0.99, 0.141], 200);
        let topic = m.topics.get(&0).expect("topic 0");
        assert_eq!(topic.member_count, 2);
    }

    // 8. total_weight accumulates
    #[test]
    fn test_total_weight_accumulates() {
        let mut m = default_modeller();
        m.assign(1, vec![1.0, 0.0], 100);
        // Sim between [1,0] and [0.99, 0.141] ≈ 0.99
        m.assign(2, vec![0.99_f32, 0.141], 200);
        let topic = m.topics.get(&0).expect("topic 0");
        // total_weight starts at 1.0 (from creation) then adds the similarity.
        assert!(topic.total_weight > 1.0);
    }

    // 9. confidence = 1.0 for new topic
    #[test]
    fn test_confidence_one_for_new_topic() {
        let mut m = default_modeller();
        let a = m.assign(1, vec![1.0, 0.0], 100);
        assert!((a.confidence - 1.0).abs() < 1e-6);
    }

    // 10. confidence = similarity for existing topic
    #[test]
    fn test_confidence_equals_similarity_for_existing() {
        let mut m = default_modeller();
        let emb1 = vec![1.0_f32, 0.0];
        m.assign(1, emb1.clone(), 100);
        let emb2 = vec![0.6_f32, 0.8]; // cosine sim to [1,0] = 0.6
        let a2 = m.assign(2, emb2.clone(), 200);
        let expected_sim = cosine_sim(&emb1, &emb2);
        assert!((a2.confidence - expected_sim).abs() < 1e-5);
    }

    // 11. topic() Some for existing id
    #[test]
    fn test_topic_some_for_existing() {
        let mut m = default_modeller();
        m.assign(1, vec![1.0, 0.0], 100);
        assert!(m.topic(0).is_some());
    }

    // 12. topic() None for unknown id
    #[test]
    fn test_topic_none_for_unknown() {
        let m = default_modeller();
        assert!(m.topic(999).is_none());
    }

    // 13. assignments_for_doc returns correct assignments sorted desc
    #[test]
    fn test_assignments_for_doc_sorted_desc() {
        let mut m = default_modeller();
        m.assign(1, vec![1.0, 0.0], 100);
        m.assign(1, vec![0.99, 0.141], 300);
        m.assign(1, vec![0.98, 0.2], 200);
        let doc_assignments = m.assignments_for_doc(1);
        assert_eq!(doc_assignments.len(), 3);
        assert_eq!(doc_assignments[0].assigned_at_secs, 300);
        assert_eq!(doc_assignments[1].assigned_at_secs, 200);
        assert_eq!(doc_assignments[2].assigned_at_secs, 100);
    }

    // 14. assignments_for_doc filters by doc_id
    #[test]
    fn test_assignments_for_doc_filters_correctly() {
        let mut m = default_modeller();
        m.assign(1, vec![1.0, 0.0], 100);
        m.assign(2, vec![0.0, 1.0], 200);
        let doc1_assignments = m.assignments_for_doc(1);
        assert_eq!(doc1_assignments.len(), 1);
        assert_eq!(doc1_assignments[0].doc_id, 1);
    }

    // 15. top_topics sorted by member_count desc
    #[test]
    fn test_top_topics_sorted_by_member_count_desc() {
        let mut m = default_modeller();
        // Topic 0: 3 members.
        m.assign(1, vec![1.0, 0.0], 100);
        m.assign(2, vec![0.99, 0.141], 110);
        m.assign(3, vec![0.98, 0.2], 120);
        // Topic 1: 1 member.
        m.assign(10, vec![0.0, 1.0], 200);

        let top = m.top_topics(10);
        assert_eq!(top.len(), 2);
        assert!(top[0].member_count >= top[1].member_count);
    }

    // 16. top_topics capped at k
    #[test]
    fn test_top_topics_capped_at_k() {
        let mut m = default_modeller();
        m.assign(1, vec![1.0, 0.0], 100);
        m.assign(2, vec![0.0, 1.0], 200);
        let top = m.top_topics(1);
        assert_eq!(top.len(), 1);
    }

    // 17. relabel sets label
    #[test]
    fn test_relabel_sets_label() {
        let mut m = default_modeller();
        m.assign(1, vec![1.0, 0.0], 100);
        let ok = m.relabel(0, "science".to_string());
        assert!(ok);
        assert_eq!(m.topic(0).expect("topic 0").label, "science");
    }

    // 18. relabel returns false for unknown topic
    #[test]
    fn test_relabel_false_for_unknown() {
        let mut m = default_modeller();
        assert!(!m.relabel(99, "ghost".to_string()));
    }

    // 19. stats total_documents
    #[test]
    fn test_stats_total_documents() {
        let mut m = default_modeller();
        m.assign(1, vec![1.0, 0.0], 100);
        m.assign(2, vec![0.99, 0.141], 200);
        let s = m.stats();
        assert_eq!(s.total_documents, 2);
    }

    // 20. stats total_topics
    #[test]
    fn test_stats_total_topics() {
        let mut m = default_modeller();
        m.assign(1, vec![1.0, 0.0], 100);
        m.assign(2, vec![0.0, 1.0], 200);
        let s = m.stats();
        assert_eq!(s.total_topics, 2);
    }

    // 21. stats avg_topic_size
    #[test]
    fn test_stats_avg_topic_size() {
        let mut m = default_modeller();
        // Topic 0: 3 docs.
        m.assign(1, vec![1.0, 0.0], 100);
        m.assign(2, vec![0.99, 0.141], 110);
        m.assign(3, vec![0.98, 0.2], 120);
        // Topic 1: 1 doc.
        m.assign(10, vec![0.0, 1.0], 200);
        let s = m.stats();
        // (3 + 1) / 2 = 2.0
        assert!((s.avg_topic_size - 2.0).abs() < 1e-6);
    }

    // 22. stats avg_topic_size is 0.0 when no topics
    #[test]
    fn test_stats_avg_topic_size_empty() {
        let m = default_modeller();
        let s = m.stats();
        assert_eq!(s.avg_topic_size, 0.0);
    }

    // 23. stats largest_topic_members
    #[test]
    fn test_stats_largest_topic_members() {
        let mut m = default_modeller();
        m.assign(1, vec![1.0, 0.0], 100);
        m.assign(2, vec![0.99, 0.141], 110);
        m.assign(3, vec![0.98, 0.2], 120);
        m.assign(10, vec![0.0, 1.0], 200);
        let s = m.stats();
        assert_eq!(s.largest_topic_members, 3);
    }

    // 24. stats largest_topic_members is 0 when no topics
    #[test]
    fn test_stats_largest_topic_members_empty() {
        let m = default_modeller();
        let s = m.stats();
        assert_eq!(s.largest_topic_members, 0);
    }

    // 25. coherence formula
    #[test]
    fn test_topic_coherence_formula() {
        let t = TopicModel {
            topic_id: 0,
            centroid: vec![1.0],
            member_count: 4,
            total_weight: 3.0,
            label: "t".to_string(),
        };
        // 4 / (1 + 3) = 1.0
        assert!((t.coherence() - 1.0).abs() < 1e-9);
    }

    // 26. cosine_sim identical vectors → 1.0
    #[test]
    fn test_cosine_sim_identical() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert!((cosine_sim(&v, &v) - 1.0).abs() < 1e-6);
    }

    // 27. cosine_sim orthogonal vectors → 0.0
    #[test]
    fn test_cosine_sim_orthogonal() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        assert!(cosine_sim(&a, &b).abs() < 1e-6);
    }

    // 28. cosine_sim zero vector → 0.0
    #[test]
    fn test_cosine_sim_zero_vector() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![1.0_f32, 0.0];
        assert!(cosine_sim(&a, &b).abs() < 1e-6);
    }

    // 29. default label format
    #[test]
    fn test_default_label_format() {
        let mut m = default_modeller();
        m.assign(1, vec![1.0, 0.0], 100);
        let topic = m.topic(0).expect("topic 0");
        assert_eq!(topic.label, "topic_0");
    }

    // 30. next_topic_id increments with each new topic
    #[test]
    fn test_next_topic_id_increments() {
        let mut m = default_modeller();
        m.assign(1, vec![1.0, 0.0, 0.0], 100);
        m.assign(2, vec![0.0, 1.0, 0.0], 200);
        m.assign(3, vec![0.0, 0.0, 1.0], 300);
        assert_eq!(m.next_topic_id, 3);
        assert_eq!(m.topics.len(), 3);
    }
}

// ===========================================================================
// LDA-Based TopicModeler — Collapsed Gibbs Sampling
// ===========================================================================

/// A word and its probability within a topic.
#[derive(Debug, Clone)]
pub struct TopicWord {
    /// The word string.
    pub word: String,
    /// Probability of this word in the topic.
    pub probability: f64,
}

/// A latent topic with its top words and coherence score.
#[derive(Debug, Clone)]
pub struct LdaTopic {
    /// Unique topic identifier (0-indexed).
    pub id: usize,
    /// Top-K words by probability.
    pub top_words: Vec<TopicWord>,
    /// Topic coherence score (PMI-based).
    pub coherence: f64,
}

/// Topic probability distribution for a single document.
#[derive(Debug, Clone)]
pub struct DocumentTopics {
    /// Document identifier.
    pub doc_id: String,
    /// Probability vector over all topics; sums to approximately 1.0.
    pub topic_distribution: Vec<f64>,
    /// Index of the dominant (argmax) topic.
    pub dominant_topic: usize,
}

/// Bag-of-words representation of a document.
#[derive(Debug, Clone)]
pub struct ModelDocument {
    /// Document identifier.
    pub doc_id: String,
    /// Word → count mapping.
    pub word_counts: HashMap<String, u32>,
}

/// Hyperparameters for the LDA topic model.
#[derive(Debug, Clone)]
pub struct TopicModelConfig {
    /// Number of topics to discover.
    pub n_topics: usize,
    /// Number of top words to include per topic.
    pub n_top_words: usize,
    /// Document-topic Dirichlet prior.
    pub alpha: f64,
    /// Topic-word Dirichlet prior.
    pub beta: f64,
    /// Maximum number of Gibbs sampling iterations.
    pub max_iter: u32,
    /// Seed for the xorshift64 PRNG.
    pub seed: u64,
}

impl Default for TopicModelConfig {
    fn default() -> Self {
        Self {
            n_topics: 10,
            n_top_words: 10,
            alpha: 0.1,
            beta: 0.01,
            max_iter: 50,
            seed: 42,
        }
    }
}

/// Full output of a fitted topic model.
#[derive(Debug, Clone)]
pub struct TopicModelResult {
    /// Discovered topics.
    pub topics: Vec<LdaTopic>,
    /// Per-document topic distributions.
    pub doc_topic_distributions: Vec<DocumentTopics>,
    /// Model perplexity on the training corpus.
    pub perplexity: f64,
    /// Number of documents used for fitting.
    pub n_docs: usize,
    /// Vocabulary size.
    pub n_words: usize,
    /// Number of Gibbs iterations actually run.
    pub iterations_run: u32,
}

/// Error type for LDA topic modeling operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TopicModelError {
    /// Fewer documents than required.
    InsufficientDocuments {
        /// Minimum required.
        min: usize,
        /// Actually provided.
        got: usize,
    },
    /// The combined vocabulary of all documents is empty.
    EmptyVocabulary,
    /// A configuration parameter is invalid.
    InvalidConfig(String),
}

impl std::fmt::Display for TopicModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientDocuments { min, got } => {
                write!(f, "insufficient documents: need at least {min}, got {got}")
            }
            Self::EmptyVocabulary => write!(f, "empty vocabulary"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for TopicModelError {}

/// Aggregate statistics derived from a fitted model result.
#[derive(Debug, Clone)]
pub struct TopicModelerStats {
    /// Number of topics in the model.
    pub n_topics: usize,
    /// Vocabulary size used during fitting.
    pub vocabulary_size: usize,
    /// Mean coherence across all topics.
    pub avg_topic_coherence: f64,
    /// For each topic index, the number of documents whose dominant topic is that index.
    pub dominant_topic_distribution: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Xorshift64 PRNG (inline, no external crate)
// ---------------------------------------------------------------------------

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

// ---------------------------------------------------------------------------
// TopicModeler
// ---------------------------------------------------------------------------

/// LDA-based topic modeler using collapsed Gibbs sampling.
///
/// Build the vocabulary from a corpus, run Gibbs sampling, and produce
/// per-topic word distributions and per-document topic distributions.
pub struct TopicModeler {
    /// Model configuration.
    pub config: TopicModelConfig,
    /// Sorted vocabulary list (index → word).
    pub vocabulary: Vec<String>,
    /// Word → vocabulary index lookup.
    pub word_index: HashMap<String, usize>,
}

impl TopicModeler {
    /// Creates a new `TopicModeler` with the given configuration.
    pub fn new(config: TopicModelConfig) -> Self {
        Self {
            config,
            vocabulary: Vec::new(),
            word_index: HashMap::new(),
        }
    }

    /// Validate configuration parameters.
    fn validate_config(&self) -> Result<(), TopicModelError> {
        if self.config.n_topics == 0 {
            return Err(TopicModelError::InvalidConfig(
                "n_topics must be >= 1".to_string(),
            ));
        }
        if self.config.alpha <= 0.0 {
            return Err(TopicModelError::InvalidConfig(
                "alpha must be > 0".to_string(),
            ));
        }
        if self.config.beta <= 0.0 {
            return Err(TopicModelError::InvalidConfig(
                "beta must be > 0".to_string(),
            ));
        }
        if self.config.seed == 0 {
            return Err(TopicModelError::InvalidConfig(
                "seed must be non-zero".to_string(),
            ));
        }
        Ok(())
    }

    /// Build vocabulary from a collection of documents, updating internal state.
    fn build_vocabulary(&mut self, documents: &[ModelDocument]) {
        let mut words: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for doc in documents {
            for word in doc.word_counts.keys() {
                words.insert(word.clone());
            }
        }
        self.vocabulary = words.into_iter().collect();
        self.word_index = self
            .vocabulary
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i))
            .collect();
    }

    /// Fit the topic model to `documents` using collapsed Gibbs sampling.
    ///
    /// Returns a [`TopicModelResult`] with topic word distributions, per-document
    /// topic distributions, perplexity, and metadata.
    pub fn fit(
        &mut self,
        documents: &[ModelDocument],
    ) -> Result<TopicModelResult, TopicModelError> {
        self.validate_config()?;

        let n_docs = documents.len();
        if n_docs < self.config.n_topics {
            return Err(TopicModelError::InsufficientDocuments {
                min: self.config.n_topics,
                got: n_docs,
            });
        }

        self.build_vocabulary(documents);
        let vocab_size = self.vocabulary.len();
        if vocab_size == 0 {
            return Err(TopicModelError::EmptyVocabulary);
        }

        let k = self.config.n_topics;
        let alpha = self.config.alpha;
        let beta = self.config.beta;
        let v_beta = vocab_size as f64 * beta;

        // Expand each document into a flat list of (word_index) tokens, tracking
        // which document each token belongs to.
        // doc_tokens[d] = Vec of word indices (with repetition per count)
        let mut doc_tokens: Vec<Vec<usize>> = Vec::with_capacity(n_docs);
        for doc in documents {
            let mut tokens: Vec<usize> = Vec::new();
            for (word, &count) in &doc.word_counts {
                if let Some(&wi) = self.word_index.get(word) {
                    for _ in 0..count {
                        tokens.push(wi);
                    }
                }
            }
            doc_tokens.push(tokens);
        }

        // Count matrices
        // doc_topic_counts[d][t] — tokens in doc d assigned to topic t
        let mut doc_topic_counts: Vec<Vec<u32>> = vec![vec![0u32; k]; n_docs];
        // topic_word_counts[t][w] — times word w assigned to topic t
        let mut topic_word_counts: Vec<Vec<u32>> = vec![vec![0u32; vocab_size]; k];
        // topic_counts[t] — total tokens assigned to topic t
        let mut topic_counts: Vec<u32> = vec![0u32; k];

        // topic_assignments[d][token_pos] — current topic assignment
        let mut topic_assignments: Vec<Vec<usize>> = Vec::with_capacity(n_docs);

        let mut rng_state: u64 = self.config.seed;

        // Initialise assignments randomly
        for (d, tokens) in doc_tokens.iter().enumerate() {
            let mut assignments: Vec<usize> = Vec::with_capacity(tokens.len());
            for &wi in tokens {
                let t = (xorshift64(&mut rng_state) as usize) % k;
                assignments.push(t);
                doc_topic_counts[d][t] += 1;
                topic_word_counts[t][wi] += 1;
                topic_counts[t] += 1;
            }
            topic_assignments.push(assignments);
        }

        // Gibbs sampling iterations
        let mut probs: Vec<f64> = vec![0.0; k];
        for _iter in 0..self.config.max_iter {
            for d in 0..n_docs {
                let n_tokens = doc_tokens[d].len();
                for pos in 0..n_tokens {
                    let wi = doc_tokens[d][pos];
                    let old_t = topic_assignments[d][pos];

                    // Remove current assignment
                    doc_topic_counts[d][old_t] = doc_topic_counts[d][old_t].saturating_sub(1);
                    topic_word_counts[old_t][wi] = topic_word_counts[old_t][wi].saturating_sub(1);
                    topic_counts[old_t] = topic_counts[old_t].saturating_sub(1);

                    // Compute conditional probabilities
                    let mut cumulative = 0.0_f64;
                    for t in 0..k {
                        let doc_factor = doc_topic_counts[d][t] as f64 + alpha;
                        let word_factor = topic_word_counts[t][wi] as f64 + beta;
                        let norm_factor = topic_counts[t] as f64 + v_beta;
                        let p = doc_factor * word_factor / norm_factor;
                        cumulative += p;
                        probs[t] = cumulative;
                    }

                    // Sample new topic
                    let total = cumulative;
                    let u = ((xorshift64(&mut rng_state) as f64) / (u64::MAX as f64)) * total;
                    let new_t = probs[..k].iter().position(|&cp| u <= cp).unwrap_or(k - 1);

                    // Reassign
                    topic_assignments[d][pos] = new_t;
                    doc_topic_counts[d][new_t] += 1;
                    topic_word_counts[new_t][wi] += 1;
                    topic_counts[new_t] += 1;
                }
            }
        }

        // Build topic word distributions
        let n_top = self.config.n_top_words.min(vocab_size);
        let mut topics: Vec<LdaTopic> = Vec::with_capacity(k);
        for t in 0..k {
            let total_t = topic_counts[t] as f64 + v_beta;
            // Build (word_index, probability) pairs
            let mut word_probs: Vec<(usize, f64)> = (0..vocab_size)
                .map(|w| {
                    let p = (topic_word_counts[t][w] as f64 + beta) / total_t;
                    (w, p)
                })
                .collect();
            word_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top_words: Vec<TopicWord> = word_probs[..n_top]
                .iter()
                .map(|&(wi, prob)| TopicWord {
                    word: self.vocabulary[wi].clone(),
                    probability: prob,
                })
                .collect();
            topics.push(LdaTopic {
                id: t,
                top_words,
                coherence: 0.0, // filled below
            });
        }

        // Build document topic distributions
        let mut doc_topic_distributions: Vec<DocumentTopics> = Vec::with_capacity(n_docs);
        for d in 0..n_docs {
            let n_d: u32 = doc_topic_counts[d].iter().sum();
            let denom = n_d as f64 + k as f64 * alpha;
            let dist: Vec<f64> = (0..k)
                .map(|t| (doc_topic_counts[d][t] as f64 + alpha) / denom)
                .collect();
            let dominant = dist
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            doc_topic_distributions.push(DocumentTopics {
                doc_id: documents[d].doc_id.clone(),
                topic_distribution: dist,
                dominant_topic: dominant,
            });
        }

        // Compute corpus-level word counts for coherence
        let mut corpus_word_counts: HashMap<String, u32> = HashMap::new();
        for doc in documents {
            for (word, &count) in &doc.word_counts {
                *corpus_word_counts.entry(word.clone()).or_insert(0) += count;
            }
        }

        // Fill coherence scores
        for topic in &mut topics {
            topic.coherence = Self::coherence_score_inner(topic, &corpus_word_counts);
        }

        // Compute perplexity
        let perplexity = Self::compute_perplexity(
            documents,
            &doc_topic_distributions,
            &topic_word_counts,
            &topic_counts,
            &self.word_index,
            vocab_size,
            beta,
            v_beta,
        );

        Ok(TopicModelResult {
            topics,
            doc_topic_distributions,
            perplexity,
            n_docs,
            n_words: vocab_size,
            iterations_run: self.config.max_iter,
        })
    }

    /// Infer topic distributions for `documents` given an already fitted model.
    ///
    /// Runs a short round of Gibbs sampling on the new documents, keeping
    /// `topic_word_counts` frozen (using only the fitted model's topic-word
    /// distributions as priors).
    pub fn transform(
        &mut self,
        documents: &[ModelDocument],
        result: &TopicModelResult,
    ) -> Result<Vec<DocumentTopics>, TopicModelError> {
        self.validate_config()?;
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        if result.topics.is_empty() {
            return Err(TopicModelError::InvalidConfig(
                "result contains no topics".to_string(),
            ));
        }

        let k = result.topics.len();
        let alpha = self.config.alpha;

        // Reconstruct topic-word probability lookup from result
        // topic_word_prob[t][word] = probability
        let topic_word_prob: Vec<HashMap<&str, f64>> = result
            .topics
            .iter()
            .map(|topic| {
                topic
                    .top_words
                    .iter()
                    .map(|tw| (tw.word.as_str(), tw.probability))
                    .collect()
            })
            .collect();

        let mut rng_state: u64 = self.config.seed.wrapping_add(1);
        let mut output: Vec<DocumentTopics> = Vec::with_capacity(documents.len());

        for doc in documents {
            // Flatten tokens
            let tokens: Vec<&str> = doc
                .word_counts
                .iter()
                .flat_map(|(w, &c)| std::iter::repeat_n(w.as_str(), c as usize))
                .collect();

            let n_tokens = tokens.len();
            if n_tokens == 0 {
                // Empty document: uniform distribution
                let uniform = 1.0 / k as f64;
                output.push(DocumentTopics {
                    doc_id: doc.doc_id.clone(),
                    topic_distribution: vec![uniform; k],
                    dominant_topic: 0,
                });
                continue;
            }

            // Initialise doc-topic counts
            let mut dt_counts: Vec<u32> = vec![0u32; k];
            let mut assignments: Vec<usize> = Vec::with_capacity(n_tokens);
            for _ in 0..n_tokens {
                let t = (xorshift64(&mut rng_state) as usize) % k;
                assignments.push(t);
                dt_counts[t] += 1;
            }

            // Short inference iterations (use max_iter from config)
            let mut probs: Vec<f64> = vec![0.0; k];
            for _iter in 0..self.config.max_iter {
                for pos in 0..n_tokens {
                    let word = tokens[pos];
                    let old_t = assignments[pos];
                    dt_counts[old_t] = dt_counts[old_t].saturating_sub(1);

                    let mut cumulative = 0.0_f64;
                    for t in 0..k {
                        let doc_factor = dt_counts[t] as f64 + alpha;
                        let word_prob = topic_word_prob[t].get(word).copied().unwrap_or(1e-10_f64);
                        let p = doc_factor * word_prob;
                        cumulative += p;
                        probs[t] = cumulative;
                    }

                    let u = ((xorshift64(&mut rng_state) as f64) / (u64::MAX as f64)) * cumulative;
                    let new_t = probs[..k].iter().position(|&cp| u <= cp).unwrap_or(k - 1);

                    assignments[pos] = new_t;
                    dt_counts[new_t] += 1;
                }
            }

            let n_d: u32 = dt_counts.iter().sum();
            let denom = n_d as f64 + k as f64 * alpha;
            let dist: Vec<f64> = (0..k)
                .map(|t| (dt_counts[t] as f64 + alpha) / denom)
                .collect();
            let dominant = dist
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            output.push(DocumentTopics {
                doc_id: doc.doc_id.clone(),
                topic_distribution: dist,
                dominant_topic: dominant,
            });
        }

        Ok(output)
    }

    /// Compute a simplified PMI-based coherence score for a topic.
    ///
    /// For each pair of top words (w_i, w_j), adds `log((co_count + 1) / (count_j + 1))`
    /// where `co_count` is approximated as `min(count_i, count_j)` (simplified co-occurrence).
    /// Returns the mean over all pairs; returns `0.0` for single-word topics.
    pub fn coherence_score(topic: &LdaTopic, corpus_word_counts: &HashMap<String, u32>) -> f64 {
        Self::coherence_score_inner(topic, corpus_word_counts)
    }

    fn coherence_score_inner(topic: &LdaTopic, corpus_word_counts: &HashMap<String, u32>) -> f64 {
        let words = &topic.top_words;
        if words.len() < 2 {
            return 0.0;
        }
        let mut sum = 0.0_f64;
        let mut count = 0usize;
        for (i, wi) in words.iter().enumerate() {
            let count_i = corpus_word_counts.get(&wi.word).copied().unwrap_or(0) as f64;
            for wj in words.iter().skip(i + 1) {
                let count_j = corpus_word_counts.get(&wj.word).copied().unwrap_or(0) as f64;
                // Simplified co-occurrence: min of individual counts
                let co = count_i.min(count_j);
                sum += ((co + 1.0) / (count_j + 1.0)).ln();
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }

    /// Compute cosine similarity between the word-probability distributions of two topics.
    ///
    /// Both topics must exist in `result.topics`.  Returns `0.0` if either topic id
    /// is out of bounds or the distributions are zero.
    pub fn most_similar_topics(topic_a: usize, topic_b: usize, result: &TopicModelResult) -> f64 {
        if topic_a >= result.topics.len() || topic_b >= result.topics.len() {
            return 0.0;
        }
        // Build dense probability vectors indexed by vocabulary position using top_words
        // We use all top_words probabilities as sparse vectors
        let ta = &result.topics[topic_a];
        let tb = &result.topics[topic_b];

        // Collect all word keys from both topics
        let mut all_words: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for tw in &ta.top_words {
            all_words.insert(tw.word.as_str());
        }
        for tw in &tb.top_words {
            all_words.insert(tw.word.as_str());
        }

        let map_a: HashMap<&str, f64> = ta
            .top_words
            .iter()
            .map(|tw| (tw.word.as_str(), tw.probability))
            .collect();
        let map_b: HashMap<&str, f64> = tb
            .top_words
            .iter()
            .map(|tw| (tw.word.as_str(), tw.probability))
            .collect();

        let mut dot = 0.0_f64;
        let mut norm_a = 0.0_f64;
        let mut norm_b = 0.0_f64;

        for word in &all_words {
            let pa = map_a.get(word).copied().unwrap_or(0.0);
            let pb = map_b.get(word).copied().unwrap_or(0.0);
            dot += pa * pb;
            norm_a += pa * pa;
            norm_b += pb * pb;
        }

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom < 1e-12 {
            0.0
        } else {
            (dot / denom).clamp(-1.0, 1.0)
        }
    }

    /// Return references to the top `n` documents ranked by `topic_distribution[topic_id]`.
    ///
    /// Returns an empty slice if `topic_id` is out of range.
    pub fn top_documents_for_topic(
        topic_id: usize,
        n: usize,
        result: &TopicModelResult,
    ) -> Vec<&DocumentTopics> {
        if result.topics.is_empty() {
            return Vec::new();
        }
        let mut docs: Vec<&DocumentTopics> = result
            .doc_topic_distributions
            .iter()
            .filter(|d| topic_id < d.topic_distribution.len())
            .collect();
        docs.sort_by(|a, b| {
            b.topic_distribution[topic_id]
                .partial_cmp(&a.topic_distribution[topic_id])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        docs.truncate(n);
        docs
    }

    /// Compute aggregate statistics from a fitted model result.
    pub fn stats(result: &TopicModelResult) -> TopicModelerStats {
        let n_topics = result.topics.len();
        let avg_topic_coherence = if n_topics == 0 {
            0.0
        } else {
            let sum: f64 = result.topics.iter().map(|t| t.coherence).sum();
            sum / n_topics as f64
        };

        let mut dominant_topic_distribution: Vec<usize> = vec![0usize; n_topics];
        for doc in &result.doc_topic_distributions {
            if doc.dominant_topic < n_topics {
                dominant_topic_distribution[doc.dominant_topic] += 1;
            }
        }

        TopicModelerStats {
            n_topics,
            vocabulary_size: result.n_words,
            avg_topic_coherence,
            dominant_topic_distribution,
        }
    }

    // -----------------------------------------------------------------------
    // Private: perplexity computation
    // -----------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn compute_perplexity(
        documents: &[ModelDocument],
        doc_dists: &[DocumentTopics],
        topic_word_counts: &[Vec<u32>],
        topic_counts: &[u32],
        word_index: &HashMap<String, usize>,
        vocab_size: usize,
        beta: f64,
        v_beta: f64,
    ) -> f64 {
        let k = topic_word_counts.len();
        let mut log_likelihood = 0.0_f64;
        let mut total_tokens: u64 = 0;

        for (d, doc) in documents.iter().enumerate() {
            let dist = &doc_dists[d].topic_distribution;
            for (word, &count) in &doc.word_counts {
                if count == 0 {
                    continue;
                }
                let wi_opt = word_index.get(word);
                let p_word_doc = match wi_opt {
                    None => 1e-10,
                    Some(&wi) => {
                        let p: f64 = (0..k)
                            .map(|t| {
                                let p_topic = dist[t];
                                let p_word_topic = (topic_word_counts[t][wi] as f64 + beta)
                                    / (topic_counts[t] as f64 + v_beta);
                                p_topic * p_word_topic
                            })
                            .sum();
                        if p <= 0.0 {
                            1e-10
                        } else {
                            p
                        }
                    }
                };
                log_likelihood += count as f64 * p_word_doc.ln();
                total_tokens += count as u64;
            }
            // also account for words in vocabulary not in this doc (zero contribution)
            let _ = vocab_size; // used in v_beta passed in
        }

        if total_tokens == 0 {
            return f64::INFINITY;
        }
        (-log_likelihood / total_tokens as f64).exp()
    }
}

// ===========================================================================
// LDA TopicModeler tests
// ===========================================================================

#[cfg(test)]
mod lda_tests {
    use crate::topic_modeler::{
        xorshift64, LdaTopic, ModelDocument, TopicModelConfig, TopicModelError, TopicModeler,
        TopicWord,
    };
    use std::collections::HashMap;

    fn simple_doc(id: &str, words: &[(&str, u32)]) -> ModelDocument {
        let mut word_counts = HashMap::new();
        for &(w, c) in words {
            word_counts.insert(w.to_string(), c);
        }
        ModelDocument {
            doc_id: id.to_string(),
            word_counts,
        }
    }

    fn make_corpus() -> Vec<ModelDocument> {
        // Two clear clusters: tech and nature
        vec![
            simple_doc(
                "d1",
                &[("rust", 5), ("code", 4), ("compile", 3), ("memory", 3)],
            ),
            simple_doc(
                "d2",
                &[("rust", 4), ("compiler", 5), ("code", 3), ("type", 3)],
            ),
            simple_doc(
                "d3",
                &[("rust", 3), ("memory", 4), ("safe", 5), ("type", 2)],
            ),
            simple_doc(
                "d4",
                &[("forest", 5), ("tree", 4), ("leaf", 3), ("nature", 3)],
            ),
            simple_doc(
                "d5",
                &[("tree", 4), ("forest", 3), ("river", 5), ("nature", 3)],
            ),
            simple_doc(
                "d6",
                &[("leaf", 3), ("nature", 4), ("river", 5), ("forest", 2)],
            ),
        ]
    }

    fn default_config_2topics() -> TopicModelConfig {
        TopicModelConfig {
            n_topics: 2,
            n_top_words: 4,
            alpha: 0.1,
            beta: 0.01,
            max_iter: 30,
            seed: 42,
        }
    }

    // 1. new() initialises empty vocabulary
    #[test]
    fn test_new_empty_vocab() {
        let m = TopicModeler::new(TopicModelConfig::default());
        assert!(m.vocabulary.is_empty());
        assert!(m.word_index.is_empty());
    }

    // 2. fit returns correct n_docs
    #[test]
    fn test_fit_n_docs() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        assert_eq!(result.n_docs, 6);
    }

    // 3. fit returns correct n_words (vocabulary size)
    #[test]
    fn test_fit_n_words() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        // unique words: rust, code, compile, memory, compiler, type, safe, forest, tree, leaf, nature, river
        assert_eq!(result.n_words, 12);
    }

    // 4. fit returns correct number of topics
    #[test]
    fn test_fit_n_topics() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        assert_eq!(result.topics.len(), 2);
    }

    // 5. fit iterations_run equals max_iter
    #[test]
    fn test_fit_iterations_run() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        assert_eq!(result.iterations_run, 30);
    }

    // 6. topic distributions sum to ~1.0
    #[test]
    fn test_doc_topic_distribution_sums_to_one() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        for doc_dist in &result.doc_topic_distributions {
            let sum: f64 = doc_dist.topic_distribution.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "sum={sum}");
        }
    }

    // 7. dominant_topic is within valid range
    #[test]
    fn test_dominant_topic_in_range() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        for doc_dist in &result.doc_topic_distributions {
            assert!(doc_dist.dominant_topic < 2);
        }
    }

    // 8. each topic has n_top_words words
    #[test]
    fn test_topic_top_words_count() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        for topic in &result.topics {
            assert_eq!(topic.top_words.len(), 4);
        }
    }

    // 9. top_words probabilities are positive
    #[test]
    fn test_top_words_probabilities_positive() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        for topic in &result.topics {
            for tw in &topic.top_words {
                assert!(
                    tw.probability > 0.0,
                    "word={} prob={}",
                    tw.word,
                    tw.probability
                );
            }
        }
    }

    // 10. top_words are sorted descending by probability
    #[test]
    fn test_top_words_sorted_descending() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        for topic in &result.topics {
            let probs: Vec<f64> = topic.top_words.iter().map(|tw| tw.probability).collect();
            for i in 1..probs.len() {
                assert!(probs[i - 1] >= probs[i], "not sorted at {i}");
            }
        }
    }

    // 11. perplexity is finite and positive
    #[test]
    fn test_perplexity_finite_positive() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        assert!(result.perplexity.is_finite(), "perplexity not finite");
        assert!(result.perplexity > 0.0);
    }

    // 12. fit error on insufficient documents
    #[test]
    fn test_fit_insufficient_documents() {
        let mut m = TopicModeler::new(TopicModelConfig {
            n_topics: 5,
            ..Default::default()
        });
        let corpus = vec![
            simple_doc("d1", &[("hello", 1)]),
            simple_doc("d2", &[("world", 1)]),
        ];
        let err = m.fit(&corpus).unwrap_err();
        assert_eq!(
            err,
            TopicModelError::InsufficientDocuments { min: 5, got: 2 }
        );
    }

    // 13. fit error on empty vocabulary
    #[test]
    fn test_fit_empty_vocabulary() {
        let mut m = TopicModeler::new(TopicModelConfig {
            n_topics: 1,
            ..Default::default()
        });
        let corpus = vec![simple_doc("d1", &[])];
        let err = m.fit(&corpus).unwrap_err();
        assert_eq!(err, TopicModelError::EmptyVocabulary);
    }

    // 14. InvalidConfig: n_topics = 0
    #[test]
    fn test_invalid_config_n_topics_zero() {
        let mut m = TopicModeler::new(TopicModelConfig {
            n_topics: 0,
            ..Default::default()
        });
        let corpus = make_corpus();
        let err = m.fit(&corpus).unwrap_err();
        matches!(err, TopicModelError::InvalidConfig(_));
    }

    // 15. InvalidConfig: alpha = 0
    #[test]
    fn test_invalid_config_alpha_zero() {
        let mut m = TopicModeler::new(TopicModelConfig {
            alpha: 0.0,
            ..Default::default()
        });
        let corpus = make_corpus();
        let err = m.fit(&corpus).unwrap_err();
        matches!(err, TopicModelError::InvalidConfig(_));
    }

    // 16. InvalidConfig: beta = 0
    #[test]
    fn test_invalid_config_beta_zero() {
        let mut m = TopicModeler::new(TopicModelConfig {
            beta: 0.0,
            ..Default::default()
        });
        let corpus = make_corpus();
        let err = m.fit(&corpus).unwrap_err();
        matches!(err, TopicModelError::InvalidConfig(_));
    }

    // 17. coherence_score returns 0.0 for single-word topic
    #[test]
    fn test_coherence_single_word_zero() {
        let topic = LdaTopic {
            id: 0,
            top_words: vec![TopicWord {
                word: "rust".to_string(),
                probability: 0.5,
            }],
            coherence: 0.0,
        };
        let corpus: HashMap<String, u32> = HashMap::new();
        let score = TopicModeler::coherence_score(&topic, &corpus);
        assert_eq!(score, 0.0);
    }

    // 18. coherence_score returns finite value for multi-word topic
    #[test]
    fn test_coherence_multiword_finite() {
        let topic = LdaTopic {
            id: 0,
            top_words: vec![
                TopicWord {
                    word: "rust".to_string(),
                    probability: 0.4,
                },
                TopicWord {
                    word: "code".to_string(),
                    probability: 0.3,
                },
                TopicWord {
                    word: "memory".to_string(),
                    probability: 0.2,
                },
            ],
            coherence: 0.0,
        };
        let mut corpus = HashMap::new();
        corpus.insert("rust".to_string(), 10u32);
        corpus.insert("code".to_string(), 8u32);
        corpus.insert("memory".to_string(), 5u32);
        let score = TopicModeler::coherence_score(&topic, &corpus);
        assert!(score.is_finite());
    }

    // 19. coherence_score empty corpus → still finite
    #[test]
    fn test_coherence_empty_corpus_finite() {
        let topic = LdaTopic {
            id: 0,
            top_words: vec![
                TopicWord {
                    word: "a".to_string(),
                    probability: 0.5,
                },
                TopicWord {
                    word: "b".to_string(),
                    probability: 0.5,
                },
            ],
            coherence: 0.0,
        };
        let corpus: HashMap<String, u32> = HashMap::new();
        let score = TopicModeler::coherence_score(&topic, &corpus);
        assert!(score.is_finite());
    }

    // 20. most_similar_topics: identical topics → 1.0
    #[test]
    fn test_most_similar_topics_identical() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        let sim = TopicModeler::most_similar_topics(0, 0, &result);
        assert!((sim - 1.0).abs() < 1e-9, "sim={sim}");
    }

    // 21. most_similar_topics: two different topics ∈ [-1, 1]
    #[test]
    fn test_most_similar_topics_in_range() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        let sim = TopicModeler::most_similar_topics(0, 1, &result);
        assert!((-1.0..=1.0).contains(&sim), "sim={sim}");
    }

    // 22. most_similar_topics: out-of-range id → 0.0
    #[test]
    fn test_most_similar_topics_out_of_range() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        let sim = TopicModeler::most_similar_topics(0, 99, &result);
        assert_eq!(sim, 0.0);
    }

    // 23. top_documents_for_topic returns at most n docs
    #[test]
    fn test_top_documents_for_topic_count() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        let top = TopicModeler::top_documents_for_topic(0, 3, &result);
        assert_eq!(top.len(), 3);
    }

    // 24. top_documents_for_topic sorted by topic probability descending
    #[test]
    fn test_top_documents_for_topic_sorted() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        let top = TopicModeler::top_documents_for_topic(0, 6, &result);
        for i in 1..top.len() {
            assert!(
                top[i - 1].topic_distribution[0] >= top[i].topic_distribution[0],
                "not sorted at {i}"
            );
        }
    }

    // 25. stats n_topics matches result
    #[test]
    fn test_stats_n_topics() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        let stats = TopicModeler::stats(&result);
        assert_eq!(stats.n_topics, 2);
    }

    // 26. stats vocabulary_size matches result.n_words
    #[test]
    fn test_stats_vocab_size() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        let stats = TopicModeler::stats(&result);
        assert_eq!(stats.vocabulary_size, result.n_words);
    }

    // 27. stats dominant_topic_distribution sums to n_docs
    #[test]
    fn test_stats_dominant_distribution_sum() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        let stats = TopicModeler::stats(&result);
        let total: usize = stats.dominant_topic_distribution.iter().sum();
        assert_eq!(total, 6);
    }

    // 28. transform returns one result per input document
    #[test]
    fn test_transform_result_count() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        let new_docs = vec![
            simple_doc("new1", &[("rust", 3), ("code", 2)]),
            simple_doc("new2", &[("forest", 4), ("tree", 3)]),
        ];
        let inferred = m.transform(&new_docs, &result).expect("transform failed");
        assert_eq!(inferred.len(), 2);
    }

    // 29. transform distributions sum to ~1.0
    #[test]
    fn test_transform_distributions_sum() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        let new_docs = vec![simple_doc("new1", &[("rust", 3), ("code", 2)])];
        let inferred = m.transform(&new_docs, &result).expect("transform failed");
        let sum: f64 = inferred[0].topic_distribution.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum={sum}");
    }

    // 30. transform empty document → uniform distribution
    #[test]
    fn test_transform_empty_doc_uniform() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        let new_docs = vec![simple_doc("empty", &[])];
        let inferred = m.transform(&new_docs, &result).expect("transform failed");
        let expected = 0.5_f64; // 1.0 / 2 topics
        for &p in &inferred[0].topic_distribution {
            assert!((p - expected).abs() < 1e-9, "p={p}");
        }
    }

    // 31. xorshift64: different seeds produce different values
    #[test]
    fn test_xorshift64_different_seeds() {
        let mut s1 = 42u64;
        let mut s2 = 123u64;
        let v1 = xorshift64(&mut s1);
        let v2 = xorshift64(&mut s2);
        assert_ne!(v1, v2);
    }

    // 32. xorshift64: sequence is deterministic
    #[test]
    fn test_xorshift64_deterministic() {
        let mut s1 = 99u64;
        let mut s2 = 99u64;
        for _ in 0..100 {
            assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
        }
    }

    // 33. doc_id preserved in fit output
    #[test]
    fn test_fit_doc_ids_preserved() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        let ids: Vec<&str> = result
            .doc_topic_distributions
            .iter()
            .map(|d| d.doc_id.as_str())
            .collect();
        assert!(ids.contains(&"d1"));
        assert!(ids.contains(&"d6"));
    }

    // 34. topic ids are 0..n_topics
    #[test]
    fn test_topic_ids_sequential() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        for (i, topic) in result.topics.iter().enumerate() {
            assert_eq!(topic.id, i);
        }
    }

    // 35. vocabulary is sorted alphabetically
    #[test]
    fn test_vocabulary_sorted() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        m.fit(&corpus).expect("fit failed");
        let sorted = {
            let mut v = m.vocabulary.clone();
            v.sort();
            v
        };
        assert_eq!(m.vocabulary, sorted);
    }

    // 36. fit result perplexity is reasonable (< 1000 for small corpus)
    #[test]
    fn test_perplexity_reasonable() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        assert!(
            result.perplexity < 1000.0,
            "perplexity={}",
            result.perplexity
        );
    }

    // 37. TopicModelError Display for InsufficientDocuments
    #[test]
    fn test_error_display_insufficient() {
        let err = TopicModelError::InsufficientDocuments { min: 5, got: 2 };
        let s = err.to_string();
        assert!(s.contains("5"), "msg={s}");
        assert!(s.contains("2"), "msg={s}");
    }

    // 38. TopicModelError Display for EmptyVocabulary
    #[test]
    fn test_error_display_empty_vocab() {
        let err = TopicModelError::EmptyVocabulary;
        let s = err.to_string();
        assert!(s.contains("empty"), "msg={s}");
    }

    // 39. TopicModelError Display for InvalidConfig
    #[test]
    fn test_error_display_invalid_config() {
        let err = TopicModelError::InvalidConfig("test reason".to_string());
        let s = err.to_string();
        assert!(s.contains("test reason"), "msg={s}");
    }

    // 40. stats avg_topic_coherence is finite
    #[test]
    fn test_stats_avg_coherence_finite() {
        let mut m = TopicModeler::new(default_config_2topics());
        let corpus = make_corpus();
        let result = m.fit(&corpus).expect("fit failed");
        let stats = TopicModeler::stats(&result);
        assert!(stats.avg_topic_coherence.is_finite());
    }
}
