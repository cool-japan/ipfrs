//! Semantic Intent Classifier
//!
//! Classifies query intent by comparing query embeddings against registered
//! intent prototype embeddings, enabling intent-aware search routing
//! (informational vs navigational vs transactional vs exploratory vs custom).
//!
//! ## Usage
//!
//! ```rust
//! use ipfrs_semantic::intent_classifier::{
//!     SemanticIntentClassifier, IntentKind, IntentPrototype, ClassifierConfig,
//! };
//!
//! let config = ClassifierConfig::default();
//! let mut classifier = SemanticIntentClassifier::new(config);
//!
//! classifier.register_prototype(IntentPrototype {
//!     intent: IntentKind::Informational,
//!     embedding: vec![1.0, 0.0, 0.0],
//!     weight: 1.0,
//!     example_count: 10,
//! });
//!
//! let result = classifier.classify(&[0.9, 0.1, 0.0]);
//! assert!(result.is_some());
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// IntentKind
// ---------------------------------------------------------------------------

/// The kind of intent a query expresses.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum IntentKind {
    /// User wants information or facts.
    Informational,
    /// User wants to navigate to a specific resource.
    Navigational,
    /// User wants to perform an action.
    Transactional,
    /// User wants to browse or discover content.
    Exploratory,
    /// User-defined custom intent with an arbitrary label.
    Custom { label: String },
}

// ---------------------------------------------------------------------------
// IntentPrototype
// ---------------------------------------------------------------------------

/// A prototype embedding representing a canonical example of a given intent.
#[derive(Clone, Debug)]
pub struct IntentPrototype {
    /// The intent this prototype represents.
    pub intent: IntentKind,
    /// The embedding vector for this prototype.
    pub embedding: Vec<f32>,
    /// Importance weight applied to this prototype's similarity score.
    /// Defaults to `1.0`.
    pub weight: f32,
    /// Number of examples this prototype was derived from.
    pub example_count: u32,
}

// ---------------------------------------------------------------------------
// IntentClassification
// ---------------------------------------------------------------------------

/// The result of classifying a query embedding against registered prototypes.
#[derive(Clone, Debug)]
pub struct IntentClassification {
    /// The best-matching intent.
    pub intent: IntentKind,
    /// Confidence score in `[0.0, 1.0]`; the highest weighted cosine similarity
    /// to any prototype of the winning intent.
    pub confidence: f32,
    /// The second-best intent, provided only when the gap between the best and
    /// second-best weighted similarity is ≥ 0.1.
    pub runner_up: Option<IntentKind>,
}

// ---------------------------------------------------------------------------
// ClassifierConfig
// ---------------------------------------------------------------------------

/// Configuration for [`SemanticIntentClassifier`].
#[derive(Clone, Debug)]
pub struct ClassifierConfig {
    /// Minimum confidence required to return a classification.
    ///
    /// Queries whose best weighted cosine similarity falls below this threshold
    /// are returned as `None`. Defaults to `0.3`.
    pub min_confidence: f32,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// ClassifierStats
// ---------------------------------------------------------------------------

/// Accumulated statistics for a [`SemanticIntentClassifier`].
#[derive(Clone, Debug, Default)]
pub struct ClassifierStats {
    /// Total number of classification calls made.
    pub total_queries: u64,
    /// Number of classifications that returned `Some` (above min_confidence).
    pub classified: u64,
    /// Number of classifications that returned `None` (below min_confidence).
    pub unclassified: u64,
    /// Per-intent classification counts, keyed by the intent label string.
    pub by_intent: HashMap<String, u64>,
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Returns a canonical string label for an [`IntentKind`].
fn intent_label(intent: &IntentKind) -> String {
    match intent {
        IntentKind::Informational => "informational".to_owned(),
        IntentKind::Navigational => "navigational".to_owned(),
        IntentKind::Transactional => "transactional".to_owned(),
        IntentKind::Exploratory => "exploratory".to_owned(),
        IntentKind::Custom { label } => label.clone(),
    }
}

/// Computes cosine similarity between two vectors.
///
/// Returns `0.0` when either vector is empty, the dimensions differ, or either
/// vector has zero norm.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
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
// SemanticIntentClassifier
// ---------------------------------------------------------------------------

/// Classifies query embeddings into intent categories using registered
/// prototype embeddings and weighted cosine similarity.
#[derive(Debug)]
pub struct SemanticIntentClassifier {
    /// Registered intent prototypes.
    prototypes: Vec<IntentPrototype>,
    /// Classifier configuration.
    config: ClassifierConfig,
    /// Accumulated statistics.
    stats: ClassifierStats,
}

impl SemanticIntentClassifier {
    /// Creates a new classifier with the given configuration.
    pub fn new(config: ClassifierConfig) -> Self {
        Self {
            prototypes: Vec::new(),
            config,
            stats: ClassifierStats::default(),
        }
    }

    /// Registers a new intent prototype.
    ///
    /// Multiple prototypes may be registered for the same [`IntentKind`].
    /// When classifying, the maximum weighted similarity across all prototypes
    /// of a given intent is used.
    pub fn register_prototype(&mut self, prototype: IntentPrototype) {
        self.prototypes.push(prototype);
    }

    /// Classifies `query_embedding` against all registered prototypes.
    ///
    /// Returns `None` when:
    /// - No prototypes are registered.
    /// - The best weighted cosine similarity is below `config.min_confidence`.
    ///
    /// Otherwise returns an [`IntentClassification`] with the winning intent,
    /// its confidence score, and an optional runner-up.
    pub fn classify(&mut self, query_embedding: &[f32]) -> Option<IntentClassification> {
        self.stats.total_queries += 1;

        if self.prototypes.is_empty() {
            self.stats.unclassified += 1;
            return None;
        }

        // Compute the maximum weighted similarity for each intent kind.
        // We use a Vec of (IntentKind, f32) rather than a HashMap so that
        // IntentKind need not implement Ord or be hashable by value directly
        // (HashMap<IntentKind, f32> is fine since IntentKind: Eq + Hash).
        let mut intent_scores: HashMap<String, (IntentKind, f32)> = HashMap::new();

        for proto in &self.prototypes {
            let sim = cosine_similarity(query_embedding, &proto.embedding);
            let weighted = sim * proto.weight;
            let label = intent_label(&proto.intent);
            let entry = intent_scores
                .entry(label)
                .or_insert_with(|| (proto.intent.clone(), f32::NEG_INFINITY));
            if weighted > entry.1 {
                entry.1 = weighted;
            }
        }

        // Sort by weighted similarity descending.
        let mut ranked: Vec<(IntentKind, f32)> = intent_scores.into_values().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (best_intent, best_score) = match ranked.first() {
            Some(pair) => pair,
            None => {
                self.stats.unclassified += 1;
                return None;
            }
        };

        if *best_score < self.config.min_confidence {
            self.stats.unclassified += 1;
            return None;
        }

        // Determine runner-up: second intent whose gap from best is >= 0.1.
        let runner_up = ranked.get(1).and_then(|(intent, score)| {
            if best_score - score >= 0.1 {
                Some(intent.clone())
            } else {
                None
            }
        });

        // Update stats.
        self.stats.classified += 1;
        let label = intent_label(best_intent);
        *self.stats.by_intent.entry(label).or_insert(0) += 1;

        Some(IntentClassification {
            intent: best_intent.clone(),
            confidence: *best_score,
            runner_up,
        })
    }

    /// Removes **all** prototypes whose intent matches `intent`.
    pub fn remove_prototype(&mut self, intent: &IntentKind) {
        self.prototypes.retain(|p| &p.intent != intent);
    }

    /// Returns a reference to the accumulated statistics.
    pub fn stats(&self) -> &ClassifierStats {
        &self.stats
    }

    /// Returns the number of registered prototypes.
    pub fn prototype_count(&self) -> usize {
        self.prototypes.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a unit vector along a single axis.
    fn unit(dim: usize, axis: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[axis] = 1.0;
        v
    }

    fn make_classifier() -> SemanticIntentClassifier {
        SemanticIntentClassifier::new(ClassifierConfig::default())
    }

    fn proto(intent: IntentKind, embedding: Vec<f32>) -> IntentPrototype {
        IntentPrototype {
            intent,
            embedding,
            weight: 1.0,
            example_count: 5,
        }
    }

    // -----------------------------------------------------------------------
    // 1. register_prototype adds entry
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_prototype_increments_count() {
        let mut c = make_classifier();
        assert_eq!(c.prototype_count(), 0);
        c.register_prototype(proto(IntentKind::Informational, unit(3, 0)));
        assert_eq!(c.prototype_count(), 1);
        c.register_prototype(proto(IntentKind::Navigational, unit(3, 1)));
        assert_eq!(c.prototype_count(), 2);
    }

    // -----------------------------------------------------------------------
    // 2. classify returns None for empty prototypes
    // -----------------------------------------------------------------------

    #[test]
    fn test_classify_empty_prototypes_returns_none() {
        let mut c = make_classifier();
        assert!(c.classify(&[0.5, 0.5, 0.0]).is_none());
    }

    // -----------------------------------------------------------------------
    // 3. classify returns correct intent
    // -----------------------------------------------------------------------

    #[test]
    fn test_classify_returns_correct_intent() {
        let mut c = make_classifier();
        c.register_prototype(proto(IntentKind::Informational, unit(3, 0)));
        c.register_prototype(proto(IntentKind::Navigational, unit(3, 1)));
        // Query is almost identical to axis-0 (Informational)
        let result = c.classify(&[0.99, 0.01, 0.0]).expect("should classify");
        assert_eq!(result.intent, IntentKind::Informational);
    }

    #[test]
    fn test_classify_navigational_intent() {
        let mut c = make_classifier();
        c.register_prototype(proto(IntentKind::Informational, unit(3, 0)));
        c.register_prototype(proto(IntentKind::Navigational, unit(3, 1)));
        let result = c.classify(&[0.01, 0.99, 0.0]).expect("should classify");
        assert_eq!(result.intent, IntentKind::Navigational);
    }

    #[test]
    fn test_classify_transactional_intent() {
        let mut c = make_classifier();
        c.register_prototype(proto(IntentKind::Transactional, unit(3, 2)));
        let result = c.classify(&[0.0, 0.0, 1.0]).expect("should classify");
        assert_eq!(result.intent, IntentKind::Transactional);
    }

    #[test]
    fn test_classify_exploratory_intent() {
        let mut c = make_classifier();
        c.register_prototype(proto(IntentKind::Exploratory, vec![1.0, 1.0, 0.0]));
        // Normalize manually: [1/sqrt(2), 1/sqrt(2), 0]
        let result = c
            .classify(&[1.0_f32 / 2.0_f32.sqrt(), 1.0_f32 / 2.0_f32.sqrt(), 0.0])
            .expect("should classify");
        assert_eq!(result.intent, IntentKind::Exploratory);
    }

    // -----------------------------------------------------------------------
    // 4. min_confidence threshold
    // -----------------------------------------------------------------------

    #[test]
    fn test_min_confidence_threshold_rejects_low_similarity() {
        let config = ClassifierConfig {
            min_confidence: 0.9,
        };
        let mut c = SemanticIntentClassifier::new(config);
        // Prototype on axis 0, query on axis 1 → similarity = 0.0
        c.register_prototype(proto(IntentKind::Informational, unit(3, 0)));
        assert!(c.classify(&unit(3, 1)).is_none());
    }

    #[test]
    fn test_min_confidence_threshold_accepts_high_similarity() {
        let config = ClassifierConfig {
            min_confidence: 0.5,
        };
        let mut c = SemanticIntentClassifier::new(config);
        c.register_prototype(proto(IntentKind::Informational, unit(3, 0)));
        // Cosine sim between [1,0,0] and [0.9,0.1,0] is ~0.994
        let result = c.classify(&[0.9, 0.1, 0.0]);
        assert!(result.is_some());
        assert!(
            result
                .expect("test: classify should return Some above min_confidence threshold")
                .confidence
                >= 0.5
        );
    }

    #[test]
    fn test_default_min_confidence_is_0_3() {
        let config = ClassifierConfig::default();
        assert!((config.min_confidence - 0.3).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // 5. runner_up when gap >= 0.1
    // -----------------------------------------------------------------------

    #[test]
    fn test_runner_up_present_when_gap_sufficient() {
        let mut c = make_classifier();
        // Informational: exact match to axis 0
        c.register_prototype(proto(IntentKind::Informational, unit(4, 0)));
        // Navigational: points partially toward axis 1 only
        c.register_prototype(proto(IntentKind::Navigational, unit(4, 1)));

        // Query close to axis 0 — large gap between cosine sims
        let result = c.classify(&[1.0, 0.0, 0.0, 0.0]).expect("should classify");
        assert_eq!(result.intent, IntentKind::Informational);
        // Gap = 1.0 - 0.0 = 1.0 >= 0.1 → runner_up should be set
        assert!(result.runner_up.is_some());
        assert_eq!(
            result
                .runner_up
                .expect("test: runner_up should be set when gap >= 0.1"),
            IntentKind::Navigational
        );
    }

    // -----------------------------------------------------------------------
    // 6. runner_up None when gap < 0.1
    // -----------------------------------------------------------------------

    #[test]
    fn test_runner_up_absent_when_gap_small() {
        let mut c = make_classifier();
        // Two prototypes with very similar similarity to the query.
        // Both at 45 degrees from each other in 2D.
        let a = vec![1.0f32 / 2.0f32.sqrt(), 1.0 / 2.0f32.sqrt()];
        let b = vec![1.0f32 / 2.0f32.sqrt(), -1.0 / 2.0f32.sqrt()];
        c.register_prototype(proto(IntentKind::Informational, a));
        c.register_prototype(proto(IntentKind::Navigational, b));
        // Query = [1, 0] → cos_sim to a = 1/sqrt(2) ≈ 0.707, to b = 1/sqrt(2) ≈ 0.707
        // Gap ≈ 0.0 < 0.1 → no runner-up
        let result = c.classify(&[1.0, 0.0]).expect("should classify");
        assert!(result.runner_up.is_none());
    }

    // -----------------------------------------------------------------------
    // 7. Multiple prototypes per intent: max sim used
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_prototypes_per_intent_uses_max() {
        let mut c = make_classifier();
        // Two prototypes for Informational: one bad, one perfect
        c.register_prototype(proto(IntentKind::Informational, unit(3, 1))); // bad
        c.register_prototype(proto(IntentKind::Informational, unit(3, 0))); // perfect
        c.register_prototype(proto(IntentKind::Navigational, unit(3, 2)));

        let result = c.classify(&unit(3, 0)).expect("should classify");
        assert_eq!(result.intent, IntentKind::Informational);
        assert!((result.confidence - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_multiple_prototypes_confidence_reflects_best() {
        let mut c = make_classifier();
        c.register_prototype(proto(IntentKind::Informational, unit(3, 0)));
        c.register_prototype(IntentPrototype {
            intent: IntentKind::Informational,
            embedding: unit(3, 1),
            weight: 1.0,
            example_count: 1,
        });
        // Query = axis 0 → sim with first = 1.0, sim with second = 0.0 → max = 1.0
        let result = c.classify(&unit(3, 0)).expect("should classify");
        assert!((result.confidence - 1.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // 8. Prototype weight affects selection
    // -----------------------------------------------------------------------

    #[test]
    fn test_prototype_weight_boosts_score() {
        let mut c = make_classifier();
        // Informational: low weight, high raw similarity
        c.register_prototype(IntentPrototype {
            intent: IntentKind::Informational,
            embedding: unit(3, 0),
            weight: 0.5,
            example_count: 10,
        });
        // Navigational: high weight, moderate raw similarity (axis 1)
        // Query is axis 0, but navigational weight is very high.
        // Weighted scores: Informational = 1.0 * 0.5 = 0.5
        //                  Navigational  = 0.0 * 100 = 0.0
        // Informational still wins because raw sim is 0 for Navigational.
        let result = c.classify(&unit(3, 0)).expect("should classify");
        assert_eq!(result.intent, IntentKind::Informational);
    }

    #[test]
    fn test_prototype_weight_shifts_winner() {
        let mut c = make_classifier();
        // Query at 45 degrees between axis 0 and axis 1.
        // Without weight: both have sim = 1/sqrt(2) ≈ 0.707 → tie
        // With Navigational weight = 2.0: Navigational weighted = 2 * 0.707 = 1.414
        //                                 Informational weighted = 1 * 0.707 = 0.707
        c.register_prototype(IntentPrototype {
            intent: IntentKind::Informational,
            embedding: unit(2, 0),
            weight: 1.0,
            example_count: 5,
        });
        c.register_prototype(IntentPrototype {
            intent: IntentKind::Navigational,
            embedding: unit(2, 1),
            weight: 2.0,
            example_count: 5,
        });
        let query = vec![1.0_f32 / 2.0_f32.sqrt(), 1.0_f32 / 2.0_f32.sqrt()];
        let result = c.classify(&query).expect("should classify");
        assert_eq!(result.intent, IntentKind::Navigational);
    }

    // -----------------------------------------------------------------------
    // 9. Custom intent classification
    // -----------------------------------------------------------------------

    #[test]
    fn test_custom_intent_classification() {
        let mut c = make_classifier();
        c.register_prototype(proto(
            IntentKind::Custom {
                label: "shopping".to_owned(),
            },
            unit(3, 0),
        ));
        let result = c.classify(&unit(3, 0)).expect("should classify");
        assert_eq!(
            result.intent,
            IntentKind::Custom {
                label: "shopping".to_owned()
            }
        );
    }

    #[test]
    fn test_custom_intent_label_in_stats() {
        let mut c = make_classifier();
        c.register_prototype(proto(
            IntentKind::Custom {
                label: "support".to_owned(),
            },
            unit(3, 2),
        ));
        let _ = c.classify(&unit(3, 2));
        assert_eq!(c.stats().by_intent.get("support").copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_multiple_custom_intents() {
        let mut c = make_classifier();
        c.register_prototype(proto(
            IntentKind::Custom {
                label: "alpha".to_owned(),
            },
            unit(3, 0),
        ));
        c.register_prototype(proto(
            IntentKind::Custom {
                label: "beta".to_owned(),
            },
            unit(3, 1),
        ));
        let result = c.classify(&unit(3, 0)).expect("should classify");
        assert_eq!(
            result.intent,
            IntentKind::Custom {
                label: "alpha".to_owned()
            }
        );
    }

    // -----------------------------------------------------------------------
    // 10. remove_prototype removes all matching
    // -----------------------------------------------------------------------

    #[test]
    fn test_remove_prototype_removes_all_matching() {
        let mut c = make_classifier();
        c.register_prototype(proto(IntentKind::Informational, unit(3, 0)));
        c.register_prototype(proto(IntentKind::Informational, unit(3, 1)));
        c.register_prototype(proto(IntentKind::Navigational, unit(3, 2)));
        assert_eq!(c.prototype_count(), 3);
        c.remove_prototype(&IntentKind::Informational);
        assert_eq!(c.prototype_count(), 1);
        // Ensure the remaining prototype is Navigational
        let result = c.classify(&unit(3, 2)).expect("should classify");
        assert_eq!(result.intent, IntentKind::Navigational);
    }

    #[test]
    fn test_remove_prototype_no_effect_on_other_intents() {
        let mut c = make_classifier();
        c.register_prototype(proto(IntentKind::Navigational, unit(3, 1)));
        c.remove_prototype(&IntentKind::Informational); // nothing to remove
        assert_eq!(c.prototype_count(), 1);
    }

    #[test]
    fn test_remove_custom_prototype() {
        let mut c = make_classifier();
        c.register_prototype(proto(
            IntentKind::Custom {
                label: "buy".to_owned(),
            },
            unit(3, 0),
        ));
        c.register_prototype(proto(IntentKind::Informational, unit(3, 1)));
        c.remove_prototype(&IntentKind::Custom {
            label: "buy".to_owned(),
        });
        assert_eq!(c.prototype_count(), 1);
    }

    // -----------------------------------------------------------------------
    // 11. Stats: classified / unclassified / by_intent
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_empty_prototypes_increments_unclassified() {
        let mut c = make_classifier();
        let _ = c.classify(&[0.1, 0.2]);
        assert_eq!(c.stats().total_queries, 1);
        assert_eq!(c.stats().unclassified, 1);
        assert_eq!(c.stats().classified, 0);
    }

    #[test]
    fn test_stats_successful_classification_increments_classified() {
        let mut c = make_classifier();
        c.register_prototype(proto(IntentKind::Informational, unit(2, 0)));
        let _ = c.classify(&unit(2, 0));
        assert_eq!(c.stats().total_queries, 1);
        assert_eq!(c.stats().classified, 1);
        assert_eq!(c.stats().unclassified, 0);
    }

    #[test]
    fn test_stats_by_intent_counts_correctly() {
        let mut c = make_classifier();
        c.register_prototype(proto(IntentKind::Informational, unit(3, 0)));
        c.register_prototype(proto(IntentKind::Navigational, unit(3, 1)));
        let _ = c.classify(&unit(3, 0)); // Informational
        let _ = c.classify(&unit(3, 0)); // Informational
        let _ = c.classify(&unit(3, 1)); // Navigational
        let by = &c.stats().by_intent;
        assert_eq!(by.get("informational").copied().unwrap_or(0), 2);
        assert_eq!(by.get("navigational").copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_stats_below_threshold_increments_unclassified() {
        let config = ClassifierConfig {
            min_confidence: 0.99,
        };
        let mut c = SemanticIntentClassifier::new(config);
        c.register_prototype(proto(IntentKind::Informational, unit(3, 0)));
        // Query is perpendicular → sim = 0.0 < 0.99
        let _ = c.classify(&unit(3, 1));
        assert_eq!(c.stats().unclassified, 1);
        assert_eq!(c.stats().classified, 0);
    }

    #[test]
    fn test_stats_total_queries_accumulates() {
        let mut c = make_classifier();
        c.register_prototype(proto(IntentKind::Informational, unit(2, 0)));
        for _ in 0..5 {
            let _ = c.classify(&unit(2, 0));
        }
        assert_eq!(c.stats().total_queries, 5);
    }

    // -----------------------------------------------------------------------
    // 12. cosine_similarity edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_cosine_similarity_empty_returns_zero() {
        let sim = cosine_similarity(&[], &[]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_dimension_mismatch_returns_zero() {
        let sim = cosine_similarity(&[1.0, 0.0], &[1.0, 0.0, 0.0]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector_returns_zero() {
        let sim = cosine_similarity(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_identical_vectors_returns_one() {
        let v = vec![0.3, 0.4, 0.5];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_returns_zero() {
        let sim = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite_returns_minus_one() {
        let sim = cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]);
        assert!((sim + 1.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // Additional edge-case / integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_classify_single_prototype_no_runner_up() {
        let mut c = make_classifier();
        c.register_prototype(proto(IntentKind::Informational, unit(3, 0)));
        let result = c.classify(&unit(3, 0)).expect("should classify");
        assert!(result.runner_up.is_none()); // only one intent registered
    }

    #[test]
    fn test_classify_confidence_equals_weighted_cosine() {
        let mut c = make_classifier();
        let embedding = vec![3.0f32, 4.0]; // norm = 5
        c.register_prototype(IntentPrototype {
            intent: IntentKind::Informational,
            embedding: embedding.clone(),
            weight: 1.0,
            example_count: 1,
        });
        // Query same direction, different magnitude
        let query = vec![6.0f32, 8.0]; // norm = 10, same direction
        let result = c.classify(&query).expect("should classify");
        assert!((result.confidence - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_intent_label_helper_all_variants() {
        assert_eq!(intent_label(&IntentKind::Informational), "informational");
        assert_eq!(intent_label(&IntentKind::Navigational), "navigational");
        assert_eq!(intent_label(&IntentKind::Transactional), "transactional");
        assert_eq!(intent_label(&IntentKind::Exploratory), "exploratory");
        assert_eq!(
            intent_label(&IntentKind::Custom {
                label: "foo".to_owned()
            }),
            "foo"
        );
    }

    #[test]
    fn test_prototype_count_after_removal_is_correct() {
        let mut c = make_classifier();
        for _ in 0..5 {
            c.register_prototype(proto(IntentKind::Transactional, unit(3, 2)));
        }
        c.register_prototype(proto(IntentKind::Informational, unit(3, 0)));
        assert_eq!(c.prototype_count(), 6);
        c.remove_prototype(&IntentKind::Transactional);
        assert_eq!(c.prototype_count(), 1);
    }

    #[test]
    fn test_classify_returns_none_when_all_sims_zero() {
        let config = ClassifierConfig {
            min_confidence: 0.01,
        }; // very low threshold
        let mut c = SemanticIntentClassifier::new(config);
        // Prototype on axis 0, query is zero vector → sim = 0.0 < 0.01
        c.register_prototype(proto(IntentKind::Informational, unit(3, 0)));
        let result = c.classify(&[0.0, 0.0, 0.0]);
        assert!(result.is_none());
    }

    #[test]
    fn test_weighted_negative_similarity_treated_as_negative() {
        // Even with weight, negative raw sim stays negative and below threshold.
        let mut c = make_classifier();
        c.register_prototype(IntentPrototype {
            intent: IntentKind::Transactional,
            embedding: unit(2, 0),
            weight: 10.0, // high weight
            example_count: 1,
        });
        // Query is opposite direction → raw sim = -1.0, weighted = -10.0 < 0.3
        let result = c.classify(&[-1.0, 0.0]);
        assert!(result.is_none());
    }
}
