//! Semantic Personalizer — per-user interest profile management
//!
//! Maintains per-user interest profiles built from interaction history, biasing
//! search results toward user preferences via category-level and result-level boosts.

use std::collections::HashMap;

/// Type of interaction a user had with a search result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InteractionType {
    /// User viewed the result (weak positive signal).
    View,
    /// User explicitly liked the result (strong positive signal).
    Like,
    /// User explicitly disliked the result (strong negative signal).
    Dislike,
    /// User saved the result for later (moderate positive signal).
    Save,
    /// User shared the result with others (moderate positive signal).
    Share,
}

impl InteractionType {
    /// Signed weight assigned to this interaction type.
    ///
    /// Positive weights increase interest; negative weights decrease it.
    pub fn weight(self) -> f64 {
        match self {
            Self::View => 0.1,
            Self::Like => 1.0,
            Self::Dislike => -1.0,
            Self::Save => 0.8,
            Self::Share => 0.5,
        }
    }
}

/// A single interaction event between a user and a search result.
#[derive(Clone, Debug)]
pub struct InteractionRecord {
    /// Unique identifier for the search result.
    pub result_id: u64,
    /// The type of interaction.
    pub interaction: InteractionType,
    /// Content category tag associated with the result.
    pub category: String,
    /// Unix timestamp (seconds) at which the interaction occurred.
    pub timestamp_secs: u64,
}

/// Per-user interest profile derived from interaction history.
#[derive(Clone, Debug)]
pub struct UserProfile {
    /// Identifier of the user this profile belongs to.
    pub user_id: u64,
    /// Accumulated weighted scores per content category.
    ///
    /// Positive scores indicate interest; negative scores indicate aversion.
    pub category_scores: HashMap<String, f64>,
    /// Result IDs that have a net positive score across all interactions.
    pub liked_ids: Vec<u64>,
    /// Result IDs that have a net negative score across all interactions.
    pub disliked_ids: Vec<u64>,
    /// Total number of interactions recorded for this user.
    pub interaction_count: u64,
    /// Net score per result ID, used internally to maintain liked/disliked lists.
    pub(crate) result_scores: HashMap<u64, f64>,
}

impl UserProfile {
    /// Create a new, empty profile for the given user.
    pub fn new(user_id: u64) -> Self {
        Self {
            user_id,
            category_scores: HashMap::new(),
            liked_ids: Vec::new(),
            disliked_ids: Vec::new(),
            interaction_count: 0,
            result_scores: HashMap::new(),
        }
    }

    /// Returns categories with a score above `0.5`, sorted by score descending.
    pub fn preferred_categories(&self) -> Vec<String> {
        let mut preferred: Vec<(String, f64)> = self
            .category_scores
            .iter()
            .filter(|(_, &s)| s > 0.5)
            .map(|(cat, &s)| (cat.clone(), s))
            .collect();
        preferred.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        preferred.into_iter().map(|(cat, _)| cat).collect()
    }

    /// Returns categories with a score below `-0.5`, sorted by score ascending (most averse first).
    pub fn aversion_categories(&self) -> Vec<String> {
        let mut aversions: Vec<(String, f64)> = self
            .category_scores
            .iter()
            .filter(|(_, &s)| s < -0.5)
            .map(|(cat, &s)| (cat.clone(), s))
            .collect();
        aversions.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        aversions.into_iter().map(|(cat, _)| cat).collect()
    }
}

/// Score multipliers that bias search results for a specific user.
#[derive(Clone, Debug)]
pub struct PersonalizationBias {
    /// Per-category score multiplier (`1.5` for preferred, `0.5` for aversion categories).
    pub category_boost: HashMap<String, f64>,
    /// Per-result direct score multiplier (`1.2` for liked IDs, `0.8` for disliked IDs).
    pub id_boost: HashMap<u64, f64>,
}

/// Manages per-user interest profiles and applies personalization to search results.
///
/// # Example
///
/// ```rust
/// use ipfrs_semantic::personalizer::{
///     InteractionRecord, InteractionType, SemanticPersonalizer,
/// };
///
/// let mut personalizer = SemanticPersonalizer::new(0.99);
///
/// personalizer.record_interaction(
///     1,
///     InteractionRecord {
///         result_id: 42,
///         interaction: InteractionType::Like,
///         category: "rust".to_string(),
///         timestamp_secs: 1_700_000_000,
///     },
/// );
///
/// let results = vec![(42_u64, 0.8_f64, "rust".to_string())];
/// let biased = personalizer.apply_bias(1, &results);
/// assert!(!biased.is_empty());
/// ```
pub struct SemanticPersonalizer {
    /// Map from user ID to that user's interest profile.
    pub profiles: HashMap<u64, UserProfile>,
    /// Multiplicative decay applied to existing category scores on each new interaction.
    ///
    /// A value of `0.99` means each prior interaction contributes slightly less over time.
    pub decay_rate: f64,
}

impl SemanticPersonalizer {
    /// Create a new `SemanticPersonalizer` with the given decay rate.
    ///
    /// `decay_rate` should be in `(0.0, 1.0]`. A typical value is `0.99`.
    pub fn new(decay_rate: f64) -> Self {
        Self {
            profiles: HashMap::new(),
            decay_rate,
        }
    }

    /// Record a user interaction and update the corresponding interest profile.
    ///
    /// Steps performed:
    /// 1. Upsert the user profile.
    /// 2. Apply decay to all existing `category_scores`.
    /// 3. Add the interaction weight to `category_scores[category]`.
    /// 4. Update `result_scores` and synchronise `liked_ids` / `disliked_ids`.
    pub fn record_interaction(&mut self, user_id: u64, interaction: InteractionRecord) {
        let decay = self.decay_rate;

        let profile = self
            .profiles
            .entry(user_id)
            .or_insert_with(|| UserProfile::new(user_id));

        // Apply temporal decay to all existing category scores.
        for score in profile.category_scores.values_mut() {
            *score *= decay;
        }

        // Add the weighted signal for this interaction's category.
        let category_score = profile
            .category_scores
            .entry(interaction.category.clone())
            .or_insert(0.0);
        *category_score += interaction.interaction.weight();

        // Update net score for this specific result.
        let result_score = profile
            .result_scores
            .entry(interaction.result_id)
            .or_insert(0.0);
        *result_score += interaction.interaction.weight();
        let net = *result_score;
        let result_id = interaction.result_id;

        // Synchronise liked / disliked ID lists based on net score.
        if net > 0.0 {
            if !profile.liked_ids.contains(&result_id) {
                profile.liked_ids.push(result_id);
            }
            profile.disliked_ids.retain(|&id| id != result_id);
        } else if net < 0.0 {
            if !profile.disliked_ids.contains(&result_id) {
                profile.disliked_ids.push(result_id);
            }
            profile.liked_ids.retain(|&id| id != result_id);
        } else {
            // Net score exactly zero: remove from both lists.
            profile.liked_ids.retain(|&id| id != result_id);
            profile.disliked_ids.retain(|&id| id != result_id);
        }

        profile.interaction_count += 1;
    }

    /// Compute personalization bias for a user.
    ///
    /// Returns `None` if the user has no profile.
    pub fn compute_bias(&self, user_id: u64) -> Option<PersonalizationBias> {
        let profile = self.profiles.get(&user_id)?;

        let mut category_boost: HashMap<String, f64> = HashMap::new();
        for cat in profile.preferred_categories() {
            category_boost.insert(cat, 1.5);
        }
        for cat in profile.aversion_categories() {
            category_boost.insert(cat, 0.5);
        }

        let mut id_boost: HashMap<u64, f64> = HashMap::new();
        for &id in &profile.liked_ids {
            id_boost.insert(id, 1.2);
        }
        for &id in &profile.disliked_ids {
            id_boost.insert(id, 0.8);
        }

        Some(PersonalizationBias {
            category_boost,
            id_boost,
        })
    }

    /// Apply personalization bias to a slice of `(result_id, score, category)` tuples.
    ///
    /// Each result's score is multiplied by its category boost and its ID boost.
    /// Results are returned sorted by adjusted score descending.
    ///
    /// If the user has no profile, results are returned sorted by the original scores.
    pub fn apply_bias(&self, user_id: u64, results: &[(u64, f64, String)]) -> Vec<(u64, f64)> {
        let bias = self.compute_bias(user_id);

        let mut adjusted: Vec<(u64, f64)> = results
            .iter()
            .map(|(result_id, score, category)| {
                let new_score = match &bias {
                    Some(b) => {
                        let cat_mult = b.category_boost.get(category).copied().unwrap_or(1.0);
                        let id_mult = b.id_boost.get(result_id).copied().unwrap_or(1.0);
                        score * cat_mult * id_mult
                    }
                    None => *score,
                };
                (*result_id, new_score)
            })
            .collect();

        adjusted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        adjusted
    }

    /// Retrieve an immutable reference to a user's profile.
    pub fn profile(&self, user_id: u64) -> Option<&UserProfile> {
        self.profiles.get(&user_id)
    }

    /// Return aggregate statistics: `(user_count, total_interactions)`.
    pub fn stats(&self) -> (usize, u64) {
        let user_count = self.profiles.len();
        let total_interactions = self.profiles.values().map(|p| p.interaction_count).sum();
        (user_count, total_interactions)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(
        result_id: u64,
        interaction: InteractionType,
        category: &str,
        timestamp_secs: u64,
    ) -> InteractionRecord {
        InteractionRecord {
            result_id,
            interaction,
            category: category.to_string(),
            timestamp_secs,
        }
    }

    // ── Test 1: record_interaction creates a profile ──────────────────────────

    #[test]
    fn test_record_creates_profile() {
        let mut p = SemanticPersonalizer::new(0.99);
        assert!(p.profile(1).is_none());
        p.record_interaction(1, make_record(10, InteractionType::View, "tech", 0));
        assert!(p.profile(1).is_some());
    }

    // ── Test 2: Like increases category score ─────────────────────────────────

    #[test]
    fn test_like_increases_category_score() {
        let mut p = SemanticPersonalizer::new(1.0); // no decay for clarity
        p.record_interaction(1, make_record(10, InteractionType::Like, "music", 0));
        let score = *p
            .profile(1)
            .expect("test: profile for user 1 not found")
            .category_scores
            .get("music")
            .expect("test: music category score not found");
        assert!(
            (score - 1.0).abs() < 1e-9,
            "score should be 1.0, got {score}"
        );
    }

    // ── Test 3: Dislike decreases category score ──────────────────────────────

    #[test]
    fn test_dislike_decreases_category_score() {
        let mut p = SemanticPersonalizer::new(1.0);
        p.record_interaction(1, make_record(20, InteractionType::Dislike, "ads", 0));
        let score = *p
            .profile(1)
            .expect("test: profile for user 1 not found")
            .category_scores
            .get("ads")
            .expect("test: ads category score not found");
        assert!(
            (score - (-1.0)).abs() < 1e-9,
            "score should be -1.0, got {score}"
        );
    }

    // ── Test 4: View adds small positive weight ───────────────────────────────

    #[test]
    fn test_view_weight() {
        let mut p = SemanticPersonalizer::new(1.0);
        p.record_interaction(1, make_record(30, InteractionType::View, "news", 0));
        let score = *p
            .profile(1)
            .expect("test: profile for user 1 not found")
            .category_scores
            .get("news")
            .expect("test: news category score not found");
        assert!((score - 0.1).abs() < 1e-9);
    }

    // ── Test 5: Save weight is 0.8 ────────────────────────────────────────────

    #[test]
    fn test_save_weight() {
        let mut p = SemanticPersonalizer::new(1.0);
        p.record_interaction(1, make_record(40, InteractionType::Save, "cooking", 0));
        let score = *p
            .profile(1)
            .expect("test: profile for user 1 not found")
            .category_scores
            .get("cooking")
            .expect("test: cooking category score not found");
        assert!((score - 0.8).abs() < 1e-9);
    }

    // ── Test 6: Share weight is 0.5 ───────────────────────────────────────────

    #[test]
    fn test_share_weight() {
        let mut p = SemanticPersonalizer::new(1.0);
        p.record_interaction(1, make_record(50, InteractionType::Share, "science", 0));
        let score = *p
            .profile(1)
            .expect("test: profile for user 1 not found")
            .category_scores
            .get("science")
            .expect("test: science category score not found");
        assert!((score - 0.5).abs() < 1e-9);
    }

    // ── Test 7: preferred_categories threshold > 0.5 ─────────────────────────

    #[test]
    fn test_preferred_categories_threshold() {
        let mut p = SemanticPersonalizer::new(1.0);
        // Score = 1.0 (Like) — above threshold
        p.record_interaction(1, make_record(1, InteractionType::Like, "rust", 0));
        // Score = 0.1 (View) — below threshold
        p.record_interaction(1, make_record(2, InteractionType::View, "python", 0));

        let prefs = p
            .profile(1)
            .expect("test: profile for user 1 not found")
            .preferred_categories();
        assert!(prefs.contains(&"rust".to_string()));
        assert!(!prefs.contains(&"python".to_string()));
    }

    // ── Test 8: aversion_categories threshold < -0.5 ─────────────────────────

    #[test]
    fn test_aversion_categories_threshold() {
        let mut p = SemanticPersonalizer::new(1.0);
        // Score = -1.0 (Dislike) — below threshold
        p.record_interaction(1, make_record(1, InteractionType::Dislike, "spam", 0));
        // Score = -0.1 — above threshold (View weight is 0.1, after Dislike we add View)
        // Use View to end up at -0.9 — still below threshold, so let's pick a neutral one
        p.record_interaction(1, make_record(2, InteractionType::View, "neutral", 0));

        let aversions = p
            .profile(1)
            .expect("test: profile for user 1 not found")
            .aversion_categories();
        assert!(aversions.contains(&"spam".to_string()));
        assert!(!aversions.contains(&"neutral".to_string()));
    }

    // ── Test 9: aversion_categories ordering (most negative first) ───────────

    #[test]
    fn test_aversion_categories_ordering() {
        let mut p = SemanticPersonalizer::new(1.0);
        p.record_interaction(1, make_record(1, InteractionType::Dislike, "bad_cat", 0));
        p.record_interaction(
            1,
            make_record(2, InteractionType::Dislike, "terrible_cat", 0),
        );
        p.record_interaction(
            1,
            make_record(3, InteractionType::Dislike, "terrible_cat", 0),
        );

        let aversions = p
            .profile(1)
            .expect("test: profile for user 1 not found")
            .aversion_categories();
        // terrible_cat has score -2.0, bad_cat has score -1.0 → terrible_cat first
        assert_eq!(aversions[0], "terrible_cat");
        assert_eq!(aversions[1], "bad_cat");
    }

    // ── Test 10: liked_ids populated after Like ───────────────────────────────

    #[test]
    fn test_liked_ids_populated() {
        let mut p = SemanticPersonalizer::new(1.0);
        p.record_interaction(1, make_record(99, InteractionType::Like, "art", 0));
        let profile = p.profile(1).expect("test: profile for user 1 not found");
        assert!(profile.liked_ids.contains(&99));
        assert!(!profile.disliked_ids.contains(&99));
    }

    // ── Test 11: disliked_ids populated after Dislike ────────────────────────

    #[test]
    fn test_disliked_ids_populated() {
        let mut p = SemanticPersonalizer::new(1.0);
        p.record_interaction(1, make_record(77, InteractionType::Dislike, "ads", 0));
        let profile = p.profile(1).expect("test: profile for user 1 not found");
        assert!(profile.disliked_ids.contains(&77));
        assert!(!profile.liked_ids.contains(&77));
    }

    // ── Test 12: compute_bias returns correct multipliers ─────────────────────

    #[test]
    fn test_compute_bias() {
        let mut p = SemanticPersonalizer::new(1.0);
        p.record_interaction(1, make_record(10, InteractionType::Like, "rust", 0));
        p.record_interaction(1, make_record(20, InteractionType::Dislike, "java", 0));

        let bias = p.compute_bias(1).expect("bias should exist");
        assert_eq!(bias.category_boost.get("rust").copied(), Some(1.5));
        assert_eq!(bias.category_boost.get("java").copied(), Some(0.5));
        assert_eq!(bias.id_boost.get(&10).copied(), Some(1.2));
        assert_eq!(bias.id_boost.get(&20).copied(), Some(0.8));
    }

    // ── Test 13: compute_bias returns None for unknown user ───────────────────

    #[test]
    fn test_compute_bias_unknown_user() {
        let p = SemanticPersonalizer::new(0.99);
        assert!(p.compute_bias(9999).is_none());
    }

    // ── Test 14: apply_bias re-ranks results ─────────────────────────────────

    #[test]
    fn test_apply_bias_reranks() {
        let mut p = SemanticPersonalizer::new(1.0);
        // User likes result 10 in "rust" category
        p.record_interaction(1, make_record(10, InteractionType::Like, "rust", 0));
        // User dislikes result 20 in "java" category
        p.record_interaction(1, make_record(20, InteractionType::Dislike, "java", 0));

        let results = vec![
            (20_u64, 0.9_f64, "java".to_string()), // originally highest
            (10_u64, 0.7_f64, "rust".to_string()), // boosted: 0.7 * 1.5 * 1.2 = 1.26
        ];

        let biased = p.apply_bias(1, &results);
        // result 10 should now rank first after bias
        assert_eq!(
            biased[0].0, 10,
            "result 10 should be ranked first after bias"
        );
    }

    // ── Test 15: apply_bias with no profile returns sorted-by-score ──────────

    #[test]
    fn test_apply_bias_no_profile() {
        let p = SemanticPersonalizer::new(0.99);
        let results = vec![
            (1_u64, 0.5_f64, "tech".to_string()),
            (2_u64, 0.9_f64, "tech".to_string()),
        ];
        let biased = p.apply_bias(999, &results);
        assert_eq!(biased[0].0, 2); // higher original score first
    }

    // ── Test 16: decay applied on subsequent interaction ──────────────────────

    #[test]
    fn test_decay_applied() {
        let mut p = SemanticPersonalizer::new(0.5); // aggressive decay
        p.record_interaction(1, make_record(1, InteractionType::Like, "topic", 0));
        // After first: score = 1.0
        p.record_interaction(1, make_record(2, InteractionType::Like, "topic", 1));
        // Before adding second Like, existing score decays: 1.0 * 0.5 = 0.5
        // After second: 0.5 + 1.0 = 1.5
        let score = *p
            .profile(1)
            .expect("test: profile for user 1 not found")
            .category_scores
            .get("topic")
            .expect("test: topic category score not found");
        assert!((score - 1.5).abs() < 1e-9, "expected 1.5, got {score}");
    }

    // ── Test 17: stats totals ─────────────────────────────────────────────────

    #[test]
    fn test_stats_totals() {
        let mut p = SemanticPersonalizer::new(0.99);
        p.record_interaction(1, make_record(1, InteractionType::View, "a", 0));
        p.record_interaction(1, make_record(2, InteractionType::Like, "b", 1));
        p.record_interaction(2, make_record(3, InteractionType::Save, "c", 2));

        let (users, interactions) = p.stats();
        assert_eq!(users, 2);
        assert_eq!(interactions, 3);
    }

    // ── Test 18: result migrates from liked to disliked when net goes negative ─

    #[test]
    fn test_result_id_migration_liked_to_disliked() {
        let mut p = SemanticPersonalizer::new(1.0);
        // First Like: net = 1.0 → liked
        p.record_interaction(1, make_record(5, InteractionType::Like, "cat", 0));
        assert!(p
            .profile(1)
            .expect("test: profile for user 1 not found")
            .liked_ids
            .contains(&5));
        // Two Dislikes: net = 1.0 - 1.0 - 1.0 = -1.0 → disliked
        p.record_interaction(1, make_record(5, InteractionType::Dislike, "cat", 1));
        p.record_interaction(1, make_record(5, InteractionType::Dislike, "cat", 2));
        let profile = p.profile(1).expect("test: profile for user 1 not found");
        assert!(!profile.liked_ids.contains(&5), "should no longer be liked");
        assert!(profile.disliked_ids.contains(&5), "should now be disliked");
    }

    // ── Test 19: interaction_count increments correctly ───────────────────────

    #[test]
    fn test_interaction_count() {
        let mut p = SemanticPersonalizer::new(0.99);
        for i in 0..5 {
            p.record_interaction(1, make_record(i, InteractionType::View, "x", i));
        }
        assert_eq!(
            p.profile(1)
                .expect("test: profile for user 1 not found")
                .interaction_count,
            5
        );
    }

    // ── Test 20: preferred_categories sorted by score descending ─────────────

    #[test]
    fn test_preferred_categories_sorted() {
        let mut p = SemanticPersonalizer::new(1.0);
        // "rust": 0.8 (Save)
        p.record_interaction(1, make_record(1, InteractionType::Save, "rust", 0));
        // "science": 1.0 (Like) — higher
        p.record_interaction(1, make_record(2, InteractionType::Like, "science", 0));

        let prefs = p
            .profile(1)
            .expect("test: profile for user 1 not found")
            .preferred_categories();
        // science (1.0) should come before rust (0.8)
        assert_eq!(prefs[0], "science");
        assert_eq!(prefs[1], "rust");
    }
}
