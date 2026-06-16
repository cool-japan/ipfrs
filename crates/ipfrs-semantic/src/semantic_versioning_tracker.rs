//! Semantic drift tracker across model/embedding versions.
//!
//! [`SemanticVersioningTracker`] detects and quantifies how concept embeddings
//! shift between successive model versions, enabling data-driven migration
//! recommendations and compatibility analysis.

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Unique integer identifier for a registered model version.
pub type SvtVersionId = u64;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by [`SemanticVersioningTracker`].
#[derive(Debug, Clone, PartialEq)]
pub enum SvtError {
    /// A version with this ID was not found.
    VersionNotFound(SvtVersionId),
    /// The concept was not registered for the requested version.
    AnchorNotFound {
        concept: String,
        version_id: SvtVersionId,
    },
    /// The two embedding vectors have incompatible lengths.
    DimMismatch { expected: usize, got: usize },
    /// Not enough anchors to compute meaningful statistics.
    InsufficientAnchors { found: usize, required: usize },
    /// An operation was attempted on a deprecated version.
    VersionDeprecated(SvtVersionId),
    /// The concept string is empty or otherwise invalid.
    InvalidConcept,
}

impl std::fmt::Display for SvtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SvtError::VersionNotFound(id) => write!(f, "version {id} not found"),
            SvtError::AnchorNotFound {
                concept,
                version_id,
            } => {
                write!(f, "anchor '{concept}' not found for version {version_id}")
            }
            SvtError::DimMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            SvtError::InsufficientAnchors { found, required } => {
                write!(
                    f,
                    "insufficient anchors: {found} found, {required} required"
                )
            }
            SvtError::VersionDeprecated(id) => write!(f, "version {id} is deprecated"),
            SvtError::InvalidConcept => write!(f, "concept name must not be empty"),
        }
    }
}

impl std::error::Error for SvtError {}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration knobs for [`SemanticVersioningTracker`].
#[derive(Debug, Clone)]
pub struct SvtTrackerConfig {
    /// Cosine-distance threshold above which a concept is considered drifted.
    pub drift_threshold: f64,
    /// Minimum number of shared anchor concepts required to produce a report.
    pub min_anchors: usize,
    /// Maximum number of consecutive version pairs to consider when computing
    /// time-series similarity.
    pub window_size: usize,
    /// When `true`, the tracker automatically marks a version as inactive
    /// when its measured overall drift against the latest active version
    /// exceeds `drift_threshold * 2.0`.
    pub auto_deprecate: bool,
}

impl Default for SvtTrackerConfig {
    fn default() -> Self {
        Self {
            drift_threshold: 0.15,
            min_anchors: 1,
            window_size: 20,
            auto_deprecate: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------------

/// Metadata for a single registered model/embedding version.
#[derive(Debug, Clone)]
pub struct SvtVersion {
    /// Unique numeric identifier (same as the map key).
    pub id: SvtVersionId,
    /// Human-readable label, e.g. `"bert-base-v2"`.
    pub name: String,
    /// Unix-epoch timestamp (seconds) at registration time.
    pub created_at: u64,
    /// Whether this version is currently active (not deprecated).
    pub is_active: bool,
    /// Embedding dimensionality expected for all anchors in this version.
    pub embedding_dim: usize,
    /// Number of anchor concepts registered for this version.
    pub anchor_count: u32,
}

// ---------------------------------------------------------------------------
// Drift event
// ---------------------------------------------------------------------------

/// A single drift observation recorded in the tracker log.
#[derive(Debug, Clone)]
pub struct SvtDriftEvent {
    /// Unix-epoch timestamp when the event was recorded.
    pub ts: u64,
    /// First version in the pair.
    pub version_a: SvtVersionId,
    /// Second version in the pair.
    pub version_b: SvtVersionId,
    /// The anchor concept that was evaluated.
    pub concept: String,
    /// Cosine distance between the two embeddings (0 = identical, 1 = orthogonal).
    pub drift_score: f64,
    /// Whether `drift_score >= config.drift_threshold`.
    pub is_significant: bool,
}

// ---------------------------------------------------------------------------
// Drift report
// ---------------------------------------------------------------------------

/// Aggregated drift analysis between two versions.
#[derive(Debug, Clone)]
pub struct SvtDriftReport {
    /// First version in the comparison.
    pub version_a: SvtVersionId,
    /// Second version in the comparison.
    pub version_b: SvtVersionId,
    /// Mean cosine distance across all shared anchor concepts.
    pub overall_drift: f64,
    /// Concepts whose drift score exceeds `drift_threshold`, sorted by score
    /// (descending).
    pub drifted_concepts: Vec<(String, f64)>,
    /// Concepts whose drift score is at or below `drift_threshold`.
    pub stable_concepts: Vec<String>,
    /// Human-readable migration recommendation.
    pub recommendation: String,
}

// ---------------------------------------------------------------------------
// Tracker stats
// ---------------------------------------------------------------------------

/// Aggregate statistics for the tracker itself.
#[derive(Debug, Clone)]
pub struct SvtTrackerStats {
    /// Total number of registered versions (including deprecated).
    pub total_versions: usize,
    /// Number of currently active versions.
    pub active_versions: usize,
    /// Total number of (concept, version) anchor pairs stored.
    pub total_anchors: usize,
    /// Number of distinct concept names.
    pub distinct_concepts: usize,
    /// Number of drift events stored in the log.
    pub drift_events: usize,
    /// Mean overall drift across all logged events.
    pub mean_logged_drift: f64,
    /// Concept with the highest mean stability score (lowest mean drift).
    pub most_stable_concept: Option<String>,
    /// Concept with the lowest stability score (highest mean drift).
    pub most_drifted_concept: Option<String>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Cosine similarity between two equal-length vectors.
/// Returns `0.0` if either vector has zero magnitude.
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

/// Cosine *distance* (1 – similarity), clamped to [0, 1].
#[inline]
fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    (1.0 - cosine_similarity(a, b)).clamp(0.0, 1.0)
}

/// Minimal xorshift64 PRNG used for synthetic timestamps when the platform
/// does not expose a wall clock (also useful in tests).
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ---------------------------------------------------------------------------
// Maximum drift log capacity
// ---------------------------------------------------------------------------

const DRIFT_LOG_CAP: usize = 500;

// ---------------------------------------------------------------------------
// SemanticVersioningTracker
// ---------------------------------------------------------------------------

/// Tracks semantic drift of concept embeddings across model versions.
///
/// # Overview
///
/// The tracker maintains a registry of named model/embedding *versions* and a
/// set of *anchor concepts* — representative items whose embeddings should be
/// stable between compatible versions.  For each concept that exists in two
/// versions the tracker computes the cosine distance of their embeddings,
/// which it calls the *drift score*.
///
/// # Example
///
/// ```rust
/// use ipfrs_semantic::semantic_versioning_tracker::{
///     SemanticVersioningTracker, SvtTrackerConfig,
/// };
///
/// let config = SvtTrackerConfig { drift_threshold: 0.1, ..Default::default() };
/// let mut tracker = SemanticVersioningTracker::new(config);
///
/// let v1 = tracker.register_version("bert-v1", 3).unwrap();
/// let v2 = tracker.register_version("bert-v2", 3).unwrap();
///
/// tracker.add_anchor("cat", v1, vec![1.0, 0.0, 0.0]).unwrap();
/// tracker.add_anchor("cat", v2, vec![0.98, 0.1, 0.05]).unwrap();
///
/// let report = tracker.compute_drift(v1, v2).unwrap();
/// assert!(report.overall_drift < 0.1);
/// ```
#[derive(Debug)]
pub struct SemanticVersioningTracker {
    /// Registered versions keyed by their numeric ID.
    versions: HashMap<SvtVersionId, SvtVersion>,
    /// Anchor concept → per-version (id, embedding) pairs.
    anchors: HashMap<String, Vec<(SvtVersionId, Vec<f64>)>>,
    /// Bounded log of recorded drift events.
    drift_log: VecDeque<SvtDriftEvent>,
    /// Tracker configuration.
    config: SvtTrackerConfig,
    /// Monotonically increasing ID counter.
    next_id: SvtVersionId,
    /// xorshift64 PRNG state (used for tiebreaking / synthetic timestamps).
    rng_state: u64,
}

/// Convenience alias matching the naming convention requested.
pub type SvtSemanticVersioningTracker = SemanticVersioningTracker;

impl SemanticVersioningTracker {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Creates a new tracker with the supplied configuration.
    pub fn new(config: SvtTrackerConfig) -> Self {
        Self {
            versions: HashMap::new(),
            anchors: HashMap::new(),
            drift_log: VecDeque::with_capacity(DRIFT_LOG_CAP + 1),
            config,
            next_id: 1,
            rng_state: 0x5851_F42D_4C95_7F2D,
        }
    }

    /// Returns the current Unix epoch timestamp in seconds.
    /// Falls back to a value derived from the internal xorshift64 PRNG state
    /// when the system clock is unavailable (e.g. no-std or time went backwards).
    fn now_ts(&mut self) -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or_else(|_| xorshift64(&mut self.rng_state))
    }

    /// Creates a tracker with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SvtTrackerConfig::default())
    }

    // -----------------------------------------------------------------------
    // Version management
    // -----------------------------------------------------------------------

    /// Registers a new version and returns its assigned [`SvtVersionId`].
    ///
    /// # Errors
    ///
    /// Returns [`SvtError::DimMismatch`] (with `expected = 0`) if `dim == 0`.
    pub fn register_version(
        &mut self,
        name: impl Into<String>,
        dim: usize,
    ) -> Result<SvtVersionId, SvtError> {
        if dim == 0 {
            return Err(SvtError::DimMismatch {
                expected: 1,
                got: 0,
            });
        }
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        let version = SvtVersion {
            id,
            name: name.into(),
            created_at: self.now_ts(),
            is_active: true,
            embedding_dim: dim,
            anchor_count: 0,
        };
        self.versions.insert(id, version);
        Ok(id)
    }

    /// Marks a version as deprecated (inactive).
    ///
    /// # Errors
    ///
    /// Returns [`SvtError::VersionNotFound`] if the ID is unknown.
    pub fn deprecate_version(&mut self, id: SvtVersionId) -> Result<(), SvtError> {
        let ver = self
            .versions
            .get_mut(&id)
            .ok_or(SvtError::VersionNotFound(id))?;
        ver.is_active = false;
        Ok(())
    }

    /// Re-activates a previously deprecated version.
    ///
    /// # Errors
    ///
    /// Returns [`SvtError::VersionNotFound`] if the ID is unknown.
    pub fn activate_version(&mut self, id: SvtVersionId) -> Result<(), SvtError> {
        let ver = self
            .versions
            .get_mut(&id)
            .ok_or(SvtError::VersionNotFound(id))?;
        ver.is_active = true;
        Ok(())
    }

    /// Returns an immutable reference to the version metadata.
    ///
    /// # Errors
    ///
    /// Returns [`SvtError::VersionNotFound`] if the ID is unknown.
    pub fn get_version(&self, id: SvtVersionId) -> Result<&SvtVersion, SvtError> {
        self.versions.get(&id).ok_or(SvtError::VersionNotFound(id))
    }

    /// Returns all registered versions sorted by ID (ascending).
    pub fn list_versions(&self) -> Vec<&SvtVersion> {
        let mut v: Vec<&SvtVersion> = self.versions.values().collect();
        v.sort_by_key(|ver| ver.id);
        v
    }

    /// Returns only active versions sorted by ID (ascending).
    pub fn active_versions(&self) -> Vec<&SvtVersion> {
        let mut v: Vec<&SvtVersion> = self.versions.values().filter(|ver| ver.is_active).collect();
        v.sort_by_key(|ver| ver.id);
        v
    }

    // -----------------------------------------------------------------------
    // Anchor management
    // -----------------------------------------------------------------------

    /// Registers a concept embedding for a specific version.
    ///
    /// If an anchor for this `(concept, version_id)` pair already exists it is
    /// **replaced**.
    ///
    /// # Errors
    ///
    /// - [`SvtError::InvalidConcept`] – concept name is empty.
    /// - [`SvtError::VersionNotFound`] – version ID unknown.
    /// - [`SvtError::DimMismatch`] – `embedding.len()` differs from the
    ///   version's declared `embedding_dim`.
    pub fn add_anchor(
        &mut self,
        concept: &str,
        version_id: SvtVersionId,
        embedding: Vec<f64>,
    ) -> Result<(), SvtError> {
        if concept.is_empty() {
            return Err(SvtError::InvalidConcept);
        }
        let ver = self
            .versions
            .get_mut(&version_id)
            .ok_or(SvtError::VersionNotFound(version_id))?;

        if embedding.len() != ver.embedding_dim {
            return Err(SvtError::DimMismatch {
                expected: ver.embedding_dim,
                got: embedding.len(),
            });
        }

        let entry = self.anchors.entry(concept.to_owned()).or_default();

        // Replace existing entry for this version if present.
        if let Some(existing) = entry.iter_mut().find(|(vid, _)| *vid == version_id) {
            existing.1 = embedding;
        } else {
            entry.push((version_id, embedding));
            ver.anchor_count = ver.anchor_count.saturating_add(1);
        }

        Ok(())
    }

    /// Returns the embedding for a specific `(concept, version)` pair.
    ///
    /// # Errors
    ///
    /// - [`SvtError::AnchorNotFound`] if no such pair exists.
    pub fn get_anchor(&self, concept: &str, version_id: SvtVersionId) -> Result<&[f64], SvtError> {
        let entries = self
            .anchors
            .get(concept)
            .ok_or_else(|| SvtError::AnchorNotFound {
                concept: concept.to_owned(),
                version_id,
            })?;
        entries
            .iter()
            .find(|(vid, _)| *vid == version_id)
            .map(|(_, emb)| emb.as_slice())
            .ok_or_else(|| SvtError::AnchorNotFound {
                concept: concept.to_owned(),
                version_id,
            })
    }

    /// Returns all concepts that have anchors registered in the given version.
    pub fn concepts_for_version(&self, version_id: SvtVersionId) -> Vec<&str> {
        self.anchors
            .iter()
            .filter_map(|(concept, entries)| {
                if entries.iter().any(|(vid, _)| *vid == version_id) {
                    Some(concept.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Core drift analysis
    // -----------------------------------------------------------------------

    /// Computes a full drift report between two versions.
    ///
    /// For each concept present in **both** versions the method computes the
    /// cosine distance and records it in the drift log.
    ///
    /// # Errors
    ///
    /// - [`SvtError::VersionNotFound`] if either version ID is unknown.
    /// - [`SvtError::InsufficientAnchors`] if fewer than `config.min_anchors`
    ///   shared concepts exist.
    pub fn compute_drift(
        &mut self,
        ver_a: SvtVersionId,
        ver_b: SvtVersionId,
    ) -> Result<SvtDriftReport, SvtError> {
        // Validate that both versions exist.
        if !self.versions.contains_key(&ver_a) {
            return Err(SvtError::VersionNotFound(ver_a));
        }
        if !self.versions.contains_key(&ver_b) {
            return Err(SvtError::VersionNotFound(ver_b));
        }

        // Collect shared concepts.
        let shared: Vec<String> = self
            .anchors
            .iter()
            .filter_map(|(concept, entries)| {
                let has_a = entries.iter().any(|(vid, _)| *vid == ver_a);
                let has_b = entries.iter().any(|(vid, _)| *vid == ver_b);
                if has_a && has_b {
                    Some(concept.clone())
                } else {
                    None
                }
            })
            .collect();

        if shared.len() < self.config.min_anchors {
            return Err(SvtError::InsufficientAnchors {
                found: shared.len(),
                required: self.config.min_anchors,
            });
        }

        let ts = self.now_ts();
        let threshold = self.config.drift_threshold;
        let mut concept_scores: Vec<(String, f64)> = Vec::with_capacity(shared.len());

        for concept in &shared {
            let entries = match self.anchors.get(concept) {
                Some(e) => e,
                None => continue,
            };
            let emb_a = match entries.iter().find(|(vid, _)| *vid == ver_a) {
                Some((_, e)) => e.as_slice(),
                None => continue,
            };
            let emb_b = match entries.iter().find(|(vid, _)| *vid == ver_b) {
                Some((_, e)) => e.as_slice(),
                None => continue,
            };

            let score = cosine_distance(emb_a, emb_b);
            concept_scores.push((concept.clone(), score));

            // Record to drift log (bounded).
            let event = SvtDriftEvent {
                ts,
                version_a: ver_a,
                version_b: ver_b,
                concept: concept.clone(),
                drift_score: score,
                is_significant: score >= threshold,
            };
            self.push_drift_event(event);
        }

        // Compute overall drift.
        let overall_drift = if concept_scores.is_empty() {
            0.0
        } else {
            concept_scores.iter().map(|(_, s)| s).sum::<f64>() / concept_scores.len() as f64
        };

        // Split into drifted / stable.
        let mut drifted_concepts: Vec<(String, f64)> = concept_scores
            .iter()
            .filter(|(_, s)| *s >= threshold)
            .cloned()
            .collect();
        drifted_concepts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let stable_concepts: Vec<String> = concept_scores
            .iter()
            .filter(|(_, s)| *s < threshold)
            .map(|(c, _)| c.clone())
            .collect();

        // Build recommendation.
        let recommendation =
            self.build_recommendation(ver_a, ver_b, overall_drift, &drifted_concepts);

        // Auto-deprecate if configured.
        if self.config.auto_deprecate && overall_drift >= threshold * 2.0 {
            if let Some(ver) = self.versions.get_mut(&ver_a) {
                ver.is_active = false;
            }
        }

        Ok(SvtDriftReport {
            version_a: ver_a,
            version_b: ver_b,
            overall_drift,
            drifted_concepts,
            stable_concepts,
            recommendation,
        })
    }

    /// Returns concepts whose drift between the two versions exceeds `threshold`,
    /// sorted by score descending.
    ///
    /// # Errors
    ///
    /// Returns [`SvtError::VersionNotFound`] if either ID is unknown.
    pub fn find_drifted_concepts(
        &self,
        ver_a: SvtVersionId,
        ver_b: SvtVersionId,
        threshold: f64,
    ) -> Result<Vec<(String, f64)>, SvtError> {
        if !self.versions.contains_key(&ver_a) {
            return Err(SvtError::VersionNotFound(ver_a));
        }
        if !self.versions.contains_key(&ver_b) {
            return Err(SvtError::VersionNotFound(ver_b));
        }

        let mut result: Vec<(String, f64)> = self
            .anchors
            .iter()
            .filter_map(|(concept, entries)| {
                let emb_a = entries.iter().find(|(vid, _)| *vid == ver_a)?.1.as_slice();
                let emb_b = entries.iter().find(|(vid, _)| *vid == ver_b)?.1.as_slice();
                let score = cosine_distance(emb_a, emb_b);
                if score >= threshold {
                    Some((concept.clone(), score))
                } else {
                    None
                }
            })
            .collect();

        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Time-series analysis
    // -----------------------------------------------------------------------

    /// Computes pairwise cosine **similarity** (not distance) between
    /// consecutive version pairs that share an anchor for `concept`.
    ///
    /// Pairs are ordered by (version_a_id, version_b_id) ascending.  At most
    /// `config.window_size` pairs are returned.
    ///
    /// # Errors
    ///
    /// - [`SvtError::InvalidConcept`] if `concept` is empty.
    /// - [`SvtError::InsufficientAnchors`] if fewer than 2 versions have the
    ///   anchor.
    pub fn semantic_similarity_over_time(
        &self,
        concept: &str,
    ) -> Result<Vec<(SvtVersionId, SvtVersionId, f64)>, SvtError> {
        if concept.is_empty() {
            return Err(SvtError::InvalidConcept);
        }

        let entries = self
            .anchors
            .get(concept)
            .ok_or(SvtError::InsufficientAnchors {
                found: 0,
                required: 2,
            })?;

        // Sort by version ID so we get a canonical ordering.
        let mut sorted: Vec<(SvtVersionId, &[f64])> = entries
            .iter()
            .map(|(vid, emb)| (*vid, emb.as_slice()))
            .collect();
        sorted.sort_by_key(|(vid, _)| *vid);

        if sorted.len() < 2 {
            return Err(SvtError::InsufficientAnchors {
                found: sorted.len(),
                required: 2,
            });
        }

        let window = self.config.window_size.max(1);
        let pairs: Vec<(SvtVersionId, SvtVersionId, f64)> = sorted
            .windows(2)
            .take(window)
            .map(|w| {
                let (va, emb_a) = w[0];
                let (vb, emb_b) = w[1];
                let sim = cosine_similarity(emb_a, emb_b);
                (va, vb, sim)
            })
            .collect();

        Ok(pairs)
    }

    /// Computes a stability score in [0, 1] for a concept across all its
    /// consecutive version pairs.
    ///
    /// `stability = 1 – mean_drift` where `mean_drift` is the mean cosine
    /// distance over all consecutive pairs.  Returns `1.0` if fewer than 2
    /// versions have the anchor (no drift measured).
    ///
    /// # Errors
    ///
    /// - [`SvtError::InvalidConcept`] if `concept` is empty.
    pub fn stability_score(&self, concept: &str) -> Result<f64, SvtError> {
        if concept.is_empty() {
            return Err(SvtError::InvalidConcept);
        }

        let entries = match self.anchors.get(concept) {
            Some(e) => e,
            None => return Ok(1.0),
        };

        let mut sorted: Vec<(SvtVersionId, &[f64])> = entries
            .iter()
            .map(|(vid, emb)| (*vid, emb.as_slice()))
            .collect();
        sorted.sort_by_key(|(vid, _)| *vid);

        if sorted.len() < 2 {
            return Ok(1.0);
        }

        let total_drift: f64 = sorted
            .windows(2)
            .map(|w| cosine_distance(w[0].1, w[1].1))
            .sum();
        let n = (sorted.len() - 1) as f64;
        Ok((1.0 - total_drift / n).clamp(0.0, 1.0))
    }

    // -----------------------------------------------------------------------
    // Migration helpers
    // -----------------------------------------------------------------------

    /// Returns a list of concept names that should be re-verified when
    /// migrating from `from` to `to`.
    ///
    /// A concept is flagged for re-verification when its drift score between
    /// the two versions exceeds `config.drift_threshold`.
    ///
    /// # Errors
    ///
    /// Returns [`SvtError::VersionNotFound`] if either ID is unknown.
    pub fn recommend_migration(
        &self,
        from: SvtVersionId,
        to: SvtVersionId,
    ) -> Result<Vec<String>, SvtError> {
        let drifted = self.find_drifted_concepts(from, to, self.config.drift_threshold)?;
        Ok(drifted.into_iter().map(|(c, _)| c).collect())
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Returns aggregate statistics for this tracker instance.
    pub fn tracker_stats(&self) -> SvtTrackerStats {
        let total_versions = self.versions.len();
        let active_versions = self.versions.values().filter(|v| v.is_active).count();

        let total_anchors: usize = self.anchors.values().map(|v| v.len()).sum();
        let distinct_concepts = self.anchors.len();
        let drift_events = self.drift_log.len();

        let mean_logged_drift = if drift_events == 0 {
            0.0
        } else {
            self.drift_log.iter().map(|e| e.drift_score).sum::<f64>() / drift_events as f64
        };

        // Most stable / most drifted concept (by per-concept mean drift in log).
        let mut concept_drift_sums: HashMap<&str, (f64, usize)> = HashMap::new();
        for event in &self.drift_log {
            let entry = concept_drift_sums
                .entry(event.concept.as_str())
                .or_insert((0.0, 0));
            entry.0 += event.drift_score;
            entry.1 += 1;
        }

        let most_stable_concept = concept_drift_sums
            .iter()
            .min_by(|a, b| {
                let avg_a = a.1 .0 / a.1 .1 as f64;
                let avg_b = b.1 .0 / b.1 .1 as f64;
                avg_a
                    .partial_cmp(&avg_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(c, _)| (*c).to_owned());

        let most_drifted_concept = concept_drift_sums
            .iter()
            .max_by(|a, b| {
                let avg_a = a.1 .0 / a.1 .1 as f64;
                let avg_b = b.1 .0 / b.1 .1 as f64;
                avg_a
                    .partial_cmp(&avg_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(c, _)| (*c).to_owned());

        SvtTrackerStats {
            total_versions,
            active_versions,
            total_anchors,
            distinct_concepts,
            drift_events,
            mean_logged_drift,
            most_stable_concept,
            most_drifted_concept,
        }
    }

    /// Returns a read-only slice of the drift log (oldest first).
    pub fn drift_log(&self) -> &VecDeque<SvtDriftEvent> {
        &self.drift_log
    }

    /// Clears all recorded drift events.
    pub fn clear_drift_log(&mut self) {
        self.drift_log.clear();
    }

    /// Returns the current tracker configuration.
    pub fn config(&self) -> &SvtTrackerConfig {
        &self.config
    }

    /// Returns a mutable reference to the configuration.
    pub fn config_mut(&mut self) -> &mut SvtTrackerConfig {
        &mut self.config
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Pushes an event to the drift log, discarding the oldest entry when
    /// the log is at capacity.
    fn push_drift_event(&mut self, event: SvtDriftEvent) {
        if self.drift_log.len() >= DRIFT_LOG_CAP {
            self.drift_log.pop_front();
        }
        self.drift_log.push_back(event);
    }

    /// Builds a human-readable recommendation string.
    fn build_recommendation(
        &self,
        ver_a: SvtVersionId,
        ver_b: SvtVersionId,
        overall_drift: f64,
        drifted: &[(String, f64)],
    ) -> String {
        let threshold = self.config.drift_threshold;
        if overall_drift < threshold * 0.5 {
            format!(
                "Versions {ver_a} → {ver_b} are semantically compatible \
                 (overall drift {overall_drift:.4} < {half:.4}). \
                 Migration should be transparent.",
                half = threshold * 0.5
            )
        } else if overall_drift < threshold {
            format!(
                "Versions {ver_a} → {ver_b} show minor drift (overall {overall_drift:.4}). \
                 Spot-check {n} concept(s) before production rollout.",
                n = drifted.len()
            )
        } else if overall_drift < threshold * 2.0 {
            let top: Vec<&str> = drifted.iter().take(5).map(|(c, _)| c.as_str()).collect();
            format!(
                "Versions {ver_a} → {ver_b} show significant drift (overall {overall_drift:.4}). \
                 Re-evaluate embeddings for: {top}.",
                top = top.join(", ")
            )
        } else {
            let top: Vec<&str> = drifted.iter().take(10).map(|(c, _)| c.as_str()).collect();
            format!(
                "Versions {ver_a} → {ver_b} are semantically incompatible \
                 (overall drift {overall_drift:.4} >= {dbl:.4}). \
                 Full re-indexing recommended. Affected concepts: {top}.",
                dbl = threshold * 2.0,
                top = top.join(", ")
            )
        }
    }

    // -----------------------------------------------------------------------
    // Batch operations
    // -----------------------------------------------------------------------

    /// Registers multiple anchors at once.
    ///
    /// Returns a `Vec` of errors (one per failed anchor); successful insertions
    /// are committed even if some fail.
    pub fn add_anchors_batch(
        &mut self,
        items: impl IntoIterator<Item = (String, SvtVersionId, Vec<f64>)>,
    ) -> Vec<SvtError> {
        let mut errors = Vec::new();
        let batch: Vec<_> = items.into_iter().collect();
        for (concept, version_id, embedding) in batch {
            if let Err(e) = self.add_anchor(&concept, version_id, embedding) {
                errors.push(e);
            }
        }
        errors
    }

    /// Computes drift reports for all consecutive active-version pairs,
    /// returning `(report_or_error)` for each pair.
    ///
    /// Pairs are ordered by ascending version IDs.
    pub fn compute_all_consecutive_drifts(&mut self) -> Vec<Result<SvtDriftReport, SvtError>> {
        let ids: Vec<SvtVersionId> = {
            let mut v: Vec<SvtVersionId> = self
                .versions
                .values()
                .filter(|ver| ver.is_active)
                .map(|ver| ver.id)
                .collect();
            v.sort_unstable();
            v
        };

        let pairs: Vec<(SvtVersionId, SvtVersionId)> =
            ids.windows(2).map(|w| (w[0], w[1])).collect();

        pairs
            .into_iter()
            .map(|(a, b)| self.compute_drift(a, b))
            .collect()
    }

    /// Returns the mean stability score across all registered concepts.
    ///
    /// Concepts with fewer than two version anchors contribute `1.0`.
    pub fn global_stability(&self) -> f64 {
        if self.anchors.is_empty() {
            return 1.0;
        }
        let total: f64 = self
            .anchors
            .keys()
            .map(|c| self.stability_score(c).unwrap_or(1.0))
            .sum();
        total / self.anchors.len() as f64
    }

    /// Returns all concepts sorted by stability score descending (most
    /// stable first).
    ///
    /// # Errors
    ///
    /// This function only returns `Err` variants from internal calls; in
    /// practice they are suppressed and the concept is scored as `1.0`.
    pub fn concepts_by_stability(&self) -> Vec<(String, f64)> {
        let mut scores: Vec<(String, f64)> = self
            .anchors
            .keys()
            .map(|c| {
                let s = self.stability_score(c).unwrap_or(1.0);
                (c.clone(), s)
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    /// Returns the `n` most drifted concepts between two versions.
    ///
    /// # Errors
    ///
    /// Returns [`SvtError::VersionNotFound`] if either ID is unknown.
    pub fn top_drifted_concepts(
        &self,
        ver_a: SvtVersionId,
        ver_b: SvtVersionId,
        n: usize,
    ) -> Result<Vec<(String, f64)>, SvtError> {
        let mut all = self.find_drifted_concepts(ver_a, ver_b, 0.0)?;
        all.truncate(n);
        Ok(all)
    }

    /// Returns `true` if the two versions are semantically compatible, i.e.
    /// their overall drift is strictly below `drift_threshold`.
    ///
    /// # Errors
    ///
    /// Returns [`SvtError::VersionNotFound`] if either ID is unknown, or
    /// [`SvtError::InsufficientAnchors`] if not enough shared concepts exist.
    pub fn are_compatible(
        &mut self,
        ver_a: SvtVersionId,
        ver_b: SvtVersionId,
    ) -> Result<bool, SvtError> {
        let report = self.compute_drift(ver_a, ver_b)?;
        Ok(report.overall_drift < self.config.drift_threshold)
    }

    /// Removes all anchors for a given version ID and decrements `anchor_count`.
    ///
    /// # Errors
    ///
    /// Returns [`SvtError::VersionNotFound`] if the version is unknown.
    pub fn remove_version_anchors(&mut self, version_id: SvtVersionId) -> Result<usize, SvtError> {
        if !self.versions.contains_key(&version_id) {
            return Err(SvtError::VersionNotFound(version_id));
        }
        let mut removed = 0usize;
        for entries in self.anchors.values_mut() {
            let before = entries.len();
            entries.retain(|(vid, _)| *vid != version_id);
            removed += before - entries.len();
        }
        // Remove any empty concept slots.
        self.anchors.retain(|_, entries| !entries.is_empty());
        // Update anchor count on the version.
        if let Some(ver) = self.versions.get_mut(&version_id) {
            ver.anchor_count = 0;
        }
        Ok(removed)
    }

    /// Computes per-concept drift scores between two versions for all shared
    /// concepts, without recording to the drift log.
    ///
    /// # Errors
    ///
    /// Returns [`SvtError::VersionNotFound`] if either ID is unknown.
    pub fn concept_drift_matrix(
        &self,
        ver_a: SvtVersionId,
        ver_b: SvtVersionId,
    ) -> Result<HashMap<String, f64>, SvtError> {
        if !self.versions.contains_key(&ver_a) {
            return Err(SvtError::VersionNotFound(ver_a));
        }
        if !self.versions.contains_key(&ver_b) {
            return Err(SvtError::VersionNotFound(ver_b));
        }
        let matrix: HashMap<String, f64> = self
            .anchors
            .iter()
            .filter_map(|(concept, entries)| {
                let emb_a = entries.iter().find(|(vid, _)| *vid == ver_a)?.1.as_slice();
                let emb_b = entries.iter().find(|(vid, _)| *vid == ver_b)?.1.as_slice();
                Some((concept.clone(), cosine_distance(emb_a, emb_b)))
            })
            .collect();
        Ok(matrix)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers -----------------------------------------------------------

    fn tracker() -> SemanticVersioningTracker {
        SemanticVersioningTracker::with_defaults()
    }

    fn tracker_with(threshold: f64) -> SemanticVersioningTracker {
        SemanticVersioningTracker::new(SvtTrackerConfig {
            drift_threshold: threshold,
            min_anchors: 1,
            window_size: 20,
            auto_deprecate: false,
        })
    }

    /// Returns a unit vector in dimension `dim` with 1.0 at position `pos`.
    fn unit(dim: usize, pos: usize) -> Vec<f64> {
        let mut v = vec![0.0f64; dim];
        v[pos] = 1.0;
        v
    }

    /// Almost-unit: 1.0 at pos, small epsilon elsewhere.
    fn near_unit(dim: usize, pos: usize, eps: f64) -> Vec<f64> {
        let mut v = vec![eps; dim];
        v[pos] = 1.0;
        v
    }

    // ---- cosine helpers ----------------------------------------------------

    #[test]
    fn cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let s = cosine_similarity(&v, &v);
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let s = cosine_similarity(&a, &b);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let s = cosine_similarity(&a, &b);
        assert!((s + 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let z = vec![0.0, 0.0];
        let a = vec![1.0, 0.0];
        assert_eq!(cosine_similarity(&z, &a), 0.0);
        assert_eq!(cosine_similarity(&a, &z), 0.0);
        assert_eq!(cosine_similarity(&z, &z), 0.0);
    }

    #[test]
    fn cosine_distance_identical_is_zero() {
        let v = vec![1.0, 0.5, 0.5];
        assert!((cosine_distance(&v, &v)).abs() < 1e-10);
    }

    #[test]
    fn cosine_distance_orthogonal_is_one() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-10);
    }

    // ---- xorshift64 --------------------------------------------------------

    #[test]
    fn xorshift64_produces_nonzero() {
        let mut state: u64 = 42;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
        assert_ne!(state, 42);
    }

    #[test]
    fn xorshift64_sequence_different() {
        let mut state: u64 = 0xCAFE_BABE;
        let v1 = xorshift64(&mut state);
        let v2 = xorshift64(&mut state);
        assert_ne!(v1, v2);
    }

    // ---- registration ------------------------------------------------------

    #[test]
    fn register_version_basic() {
        let mut t = tracker();
        let id = t
            .register_version("v1", 128)
            .expect("test: register version v1 with dim 128 should succeed");
        assert!(id > 0);
        let ver = t
            .get_version(id)
            .expect("test: get version by id should succeed for just-registered version");
        assert_eq!(ver.name, "v1");
        assert_eq!(ver.embedding_dim, 128);
        assert!(ver.is_active);
        assert_eq!(ver.anchor_count, 0);
    }

    #[test]
    fn register_version_ids_monotone() {
        let mut t = tracker();
        let a = t
            .register_version("a", 16)
            .expect("test: register version a should succeed");
        let b = t
            .register_version("b", 16)
            .expect("test: register version b should succeed");
        let c = t
            .register_version("c", 16)
            .expect("test: register version c should succeed");
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn register_version_zero_dim_err() {
        let mut t = tracker();
        let res = t.register_version("bad", 0);
        assert!(res.is_err());
    }

    #[test]
    fn deprecate_and_activate() {
        let mut t = tracker();
        let id = t
            .register_version("v1", 8)
            .expect("test: register version v1 should succeed");
        t.deprecate_version(id)
            .expect("test: deprecate version should succeed for known id");
        assert!(
            !t.get_version(id)
                .expect("test: get version should succeed for known id")
                .is_active
        );
        t.activate_version(id)
            .expect("test: activate version should succeed for known id");
        assert!(
            t.get_version(id)
                .expect("test: get version should succeed for known id")
                .is_active
        );
    }

    #[test]
    fn deprecate_nonexistent_err() {
        let mut t = tracker();
        assert!(t.deprecate_version(9999).is_err());
    }

    #[test]
    fn activate_nonexistent_err() {
        let mut t = tracker();
        assert!(t.activate_version(9999).is_err());
    }

    #[test]
    fn list_versions_sorted() {
        let mut t = tracker();
        let c = t
            .register_version("c", 4)
            .expect("test: register version c should succeed");
        let a = t
            .register_version("a", 4)
            .expect("test: register version a should succeed");
        let b = t
            .register_version("b", 4)
            .expect("test: register version b should succeed");
        let listed: Vec<SvtVersionId> = t.list_versions().iter().map(|v| v.id).collect();
        assert!(listed.contains(&a));
        assert!(listed.contains(&b));
        assert!(listed.contains(&c));
        // IDs must be ascending.
        for w in listed.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn active_versions_filters_deprecated() {
        let mut t = tracker();
        let a = t
            .register_version("a", 4)
            .expect("test: register version a should succeed");
        let b = t
            .register_version("b", 4)
            .expect("test: register version b should succeed");
        t.deprecate_version(a)
            .expect("test: deprecate version a should succeed");
        let active_ids: Vec<SvtVersionId> = t.active_versions().iter().map(|v| v.id).collect();
        assert!(!active_ids.contains(&a));
        assert!(active_ids.contains(&b));
    }

    // ---- anchor management -------------------------------------------------

    #[test]
    fn add_anchor_basic() {
        let mut t = tracker();
        let v = t
            .register_version("v", 3)
            .expect("test: register version v should succeed");
        t.add_anchor("cat", v, unit(3, 0))
            .expect("test: add anchor cat to version v should succeed");
        let emb = t
            .get_anchor("cat", v)
            .expect("test: get anchor cat for version v should succeed");
        assert_eq!(emb, &unit(3, 0));
    }

    #[test]
    fn add_anchor_replaces() {
        let mut t = tracker();
        let v = t
            .register_version("v", 3)
            .expect("test: register version v should succeed");
        t.add_anchor("dog", v, unit(3, 0))
            .expect("test: add anchor dog to version v should succeed");
        t.add_anchor("dog", v, unit(3, 1))
            .expect("test: replace anchor dog for version v should succeed"); // replace
        let emb = t
            .get_anchor("dog", v)
            .expect("test: get replaced anchor dog should succeed");
        assert_eq!(emb, &unit(3, 1));
        // anchor_count should not double-count
        assert_eq!(
            t.get_version(v)
                .expect("test: get version v should succeed")
                .anchor_count,
            1
        );
    }

    #[test]
    fn add_anchor_dim_mismatch_err() {
        let mut t = tracker();
        let v = t
            .register_version("v", 4)
            .expect("test: register version v should succeed");
        assert!(t.add_anchor("x", v, vec![1.0, 0.0]).is_err());
    }

    #[test]
    fn add_anchor_empty_concept_err() {
        let mut t = tracker();
        let v = t
            .register_version("v", 2)
            .expect("test: register version v should succeed");
        assert!(t.add_anchor("", v, vec![1.0, 0.0]).is_err());
    }

    #[test]
    fn add_anchor_unknown_version_err() {
        let mut t = tracker();
        assert!(t.add_anchor("x", 999, vec![1.0]).is_err());
    }

    #[test]
    fn get_anchor_missing_concept_err() {
        let mut t = tracker();
        let v = t
            .register_version("v", 2)
            .expect("test: register version v should succeed");
        assert!(t.get_anchor("missing", v).is_err());
    }

    #[test]
    fn get_anchor_missing_version_for_concept_err() {
        let mut t = tracker();
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, vec![1.0, 0.0])
            .expect("test: add anchor x to v1 should succeed");
        assert!(t.get_anchor("x", v2).is_err());
    }

    #[test]
    fn anchor_count_increments() {
        let mut t = tracker();
        let v = t
            .register_version("v", 2)
            .expect("test: register version v should succeed");
        assert_eq!(
            t.get_version(v)
                .expect("test: get version v should succeed")
                .anchor_count,
            0
        );
        t.add_anchor("a", v, vec![1.0, 0.0])
            .expect("test: add anchor a to version v should succeed");
        assert_eq!(
            t.get_version(v)
                .expect("test: get version v should succeed after anchor a")
                .anchor_count,
            1
        );
        t.add_anchor("b", v, vec![0.0, 1.0])
            .expect("test: add anchor b to version v should succeed");
        assert_eq!(
            t.get_version(v)
                .expect("test: get version v should succeed after anchor b")
                .anchor_count,
            2
        );
    }

    #[test]
    fn concepts_for_version_correct() {
        let mut t = tracker();
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("cat", v1, vec![1.0, 0.0])
            .expect("test: add anchor cat to v1 should succeed");
        t.add_anchor("dog", v1, vec![0.0, 1.0])
            .expect("test: add anchor dog to v1 should succeed");
        t.add_anchor("cat", v2, vec![1.0, 0.0])
            .expect("test: add anchor cat to v2 should succeed");
        let v1_concepts = t.concepts_for_version(v1);
        assert!(v1_concepts.contains(&"cat"));
        assert!(v1_concepts.contains(&"dog"));
        let v2_concepts = t.concepts_for_version(v2);
        assert!(v2_concepts.contains(&"cat"));
        assert!(!v2_concepts.contains(&"dog"));
    }

    // ---- drift computation -------------------------------------------------

    #[test]
    fn compute_drift_zero_for_identical_embeddings() {
        let mut t = tracker_with(0.1);
        let v1 = t
            .register_version("v1", 3)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 3)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, vec![1.0, 0.0, 0.0])
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, vec![1.0, 0.0, 0.0])
            .expect("test: add anchor x to v2 should succeed");
        let report = t
            .compute_drift(v1, v2)
            .expect("test: compute drift with identical embeddings should succeed");
        assert!(report.overall_drift < 1e-10);
        assert!(report.drifted_concepts.is_empty());
        assert!(!report.stable_concepts.is_empty());
    }

    #[test]
    fn compute_drift_max_for_orthogonal() {
        let mut t = tracker_with(0.5);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 1))
            .expect("test: add anchor x to v2 should succeed");
        let report = t
            .compute_drift(v1, v2)
            .expect("test: compute drift for orthogonal embeddings should succeed");
        assert!((report.overall_drift - 1.0).abs() < 1e-10);
        assert_eq!(report.drifted_concepts.len(), 1);
    }

    #[test]
    fn compute_drift_unknown_version_err() {
        let mut t = tracker();
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        assert!(t.compute_drift(v1, 9999).is_err());
        assert!(t.compute_drift(9999, v1).is_err());
    }

    #[test]
    fn compute_drift_insufficient_anchors_err() {
        let mut t = SemanticVersioningTracker::new(SvtTrackerConfig {
            drift_threshold: 0.1,
            min_anchors: 3,
            window_size: 10,
            auto_deprecate: false,
        });
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, vec![1.0, 0.0])
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, vec![1.0, 0.0])
            .expect("test: add anchor x to v2 should succeed");
        // Only 1 shared concept, need 3.
        assert!(t.compute_drift(v1, v2).is_err());
    }

    #[test]
    fn compute_drift_records_to_log() {
        let mut t = tracker_with(0.1);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("a", v1, unit(2, 0))
            .expect("test: add anchor a to v1 should succeed");
        t.add_anchor("a", v2, unit(2, 0))
            .expect("test: add anchor a to v2 should succeed");
        t.add_anchor("b", v1, unit(2, 1))
            .expect("test: add anchor b to v1 should succeed");
        t.add_anchor("b", v2, unit(2, 0))
            .expect("test: add anchor b to v2 should succeed");
        let _ = t
            .compute_drift(v1, v2)
            .expect("test: compute drift should succeed");
        assert_eq!(t.drift_log().len(), 2);
    }

    #[test]
    fn compute_drift_report_has_recommendation() {
        let mut t = tracker_with(0.1);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 0))
            .expect("test: add anchor x to v2 should succeed");
        let report = t
            .compute_drift(v1, v2)
            .expect("test: compute drift should succeed");
        assert!(!report.recommendation.is_empty());
    }

    #[test]
    fn compute_drift_drifted_sorted_descending() {
        let mut t = tracker_with(0.01);
        let v1 = t
            .register_version("v1", 4)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 4)
            .expect("test: register version v2 should succeed");
        // large drift
        t.add_anchor("a", v1, unit(4, 0))
            .expect("test: add anchor a to v1 should succeed");
        t.add_anchor("a", v2, unit(4, 1))
            .expect("test: add anchor a to v2 should succeed");
        // small (but still above threshold=0.01) drift:
        // near_unit(4, 0, 0.5) = [1.0, 0.5, 0.5, 0.5]; cosine-distance vs unit(4,0) ≈ 0.24
        t.add_anchor("b", v1, near_unit(4, 0, 0.5))
            .expect("test: add anchor b to v1 should succeed");
        t.add_anchor("b", v2, unit(4, 0))
            .expect("test: add anchor b to v2 should succeed");
        let report = t
            .compute_drift(v1, v2)
            .expect("test: compute drift should succeed");
        // Both should be in drifted (threshold 0.01).
        assert!(report.drifted_concepts.len() >= 2);
        // Must be sorted descending.
        for w in report.drifted_concepts.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    // ---- find_drifted_concepts ---------------------------------------------

    #[test]
    fn find_drifted_concepts_above_threshold() {
        let mut t = tracker_with(0.3);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("same", v1, unit(2, 0))
            .expect("test: add anchor same to v1 should succeed");
        t.add_anchor("same", v2, unit(2, 0))
            .expect("test: add anchor same to v2 should succeed");
        t.add_anchor("diff", v1, unit(2, 0))
            .expect("test: add anchor diff to v1 should succeed");
        t.add_anchor("diff", v2, unit(2, 1))
            .expect("test: add anchor diff to v2 should succeed");
        let drifted = t
            .find_drifted_concepts(v1, v2, 0.3)
            .expect("test: find drifted concepts should succeed");
        assert_eq!(drifted.len(), 1);
        assert_eq!(drifted[0].0, "diff");
    }

    #[test]
    fn find_drifted_concepts_unknown_version_err() {
        let mut t = tracker();
        let v = t
            .register_version("v", 2)
            .expect("test: register version v should succeed");
        assert!(t.find_drifted_concepts(v, 999, 0.1).is_err());
    }

    // ---- semantic_similarity_over_time ------------------------------------

    #[test]
    fn semantic_similarity_over_time_basic() {
        let mut t = tracker_with(0.1);
        let v1 = t
            .register_version("v1", 3)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 3)
            .expect("test: register version v2 should succeed");
        let v3 = t
            .register_version("v3", 3)
            .expect("test: register version v3 should succeed");
        t.add_anchor("cat", v1, unit(3, 0))
            .expect("test: add anchor cat to v1 should succeed");
        t.add_anchor("cat", v2, near_unit(3, 0, 0.01))
            .expect("test: add anchor cat to v2 should succeed");
        t.add_anchor("cat", v3, unit(3, 0))
            .expect("test: add anchor cat to v3 should succeed");
        let pairs = t
            .semantic_similarity_over_time("cat")
            .expect("test: semantic similarity over time for cat should succeed");
        assert_eq!(pairs.len(), 2);
        // All similarities should be high.
        for (_, _, sim) in &pairs {
            assert!(*sim > 0.9);
        }
    }

    #[test]
    fn semantic_similarity_over_time_empty_concept_err() {
        let t = tracker();
        assert!(t.semantic_similarity_over_time("").is_err());
    }

    #[test]
    fn semantic_similarity_over_time_missing_concept_err() {
        let t = tracker();
        assert!(t.semantic_similarity_over_time("ghost").is_err());
    }

    #[test]
    fn semantic_similarity_over_time_single_version_err() {
        let mut t = tracker();
        let v = t
            .register_version("v", 2)
            .expect("test: register version v should succeed");
        t.add_anchor("x", v, unit(2, 0))
            .expect("test: add anchor x to v should succeed");
        assert!(t.semantic_similarity_over_time("x").is_err());
    }

    #[test]
    fn semantic_similarity_over_time_window_limit() {
        let mut t = SemanticVersioningTracker::new(SvtTrackerConfig {
            drift_threshold: 0.1,
            min_anchors: 1,
            window_size: 3,
            auto_deprecate: false,
        });
        for i in 0..10usize {
            let v = t
                .register_version(format!("v{i}"), 2)
                .expect("test: register version should succeed");
            t.add_anchor("x", v, vec![1.0, i as f64])
                .expect("test: add anchor x to version should succeed");
        }
        let pairs = t
            .semantic_similarity_over_time("x")
            .expect("test: semantic similarity over time for x should succeed");
        assert!(pairs.len() <= 3);
    }

    // ---- stability_score ---------------------------------------------------

    #[test]
    fn stability_score_no_anchors_returns_one() {
        let t = tracker();
        // Concept not registered at all.
        let s = t
            .stability_score("ghost")
            .expect("test: stability score for unregistered concept should return Ok(1.0)");
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn stability_score_one_version_returns_one() {
        let mut t = tracker();
        let v = t
            .register_version("v", 2)
            .expect("test: register version v should succeed");
        t.add_anchor("x", v, unit(2, 0))
            .expect("test: add anchor x to v should succeed");
        let s = t
            .stability_score("x")
            .expect("test: stability score should succeed");
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn stability_score_identical_embeddings_is_one() {
        let mut t = tracker();
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 0))
            .expect("test: add anchor x to v2 should succeed");
        let s = t
            .stability_score("x")
            .expect("test: stability score for identical embeddings should succeed");
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn stability_score_orthogonal_embeddings_is_zero() {
        let mut t = tracker();
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 1))
            .expect("test: add anchor x to v2 should succeed");
        let s = t
            .stability_score("x")
            .expect("test: stability score for orthogonal embeddings should succeed");
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn stability_score_empty_concept_err() {
        let t = tracker();
        assert!(t.stability_score("").is_err());
    }

    // ---- recommend_migration -----------------------------------------------

    #[test]
    fn recommend_migration_returns_drifted() {
        let mut t = tracker_with(0.3);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("stable", v1, unit(2, 0))
            .expect("test: add anchor stable to v1 should succeed");
        t.add_anchor("stable", v2, unit(2, 0))
            .expect("test: add anchor stable to v2 should succeed");
        t.add_anchor("drifted", v1, unit(2, 0))
            .expect("test: add anchor drifted to v1 should succeed");
        t.add_anchor("drifted", v2, unit(2, 1))
            .expect("test: add anchor drifted to v2 should succeed");
        let recs = t
            .recommend_migration(v1, v2)
            .expect("test: recommend migration should succeed");
        assert!(recs.contains(&"drifted".to_string()));
        assert!(!recs.contains(&"stable".to_string()));
    }

    #[test]
    fn recommend_migration_unknown_version_err() {
        let mut t = tracker();
        let v = t
            .register_version("v", 2)
            .expect("test: register version v should succeed");
        assert!(t.recommend_migration(v, 9999).is_err());
    }

    // ---- tracker_stats -----------------------------------------------------

    #[test]
    fn tracker_stats_empty() {
        let t = tracker();
        let stats = t.tracker_stats();
        assert_eq!(stats.total_versions, 0);
        assert_eq!(stats.active_versions, 0);
        assert_eq!(stats.total_anchors, 0);
        assert_eq!(stats.distinct_concepts, 0);
        assert_eq!(stats.drift_events, 0);
        assert!((stats.mean_logged_drift).abs() < 1e-10);
    }

    #[test]
    fn tracker_stats_after_operations() {
        let mut t = tracker_with(0.1);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.deprecate_version(v1)
            .expect("test: deprecate version v1 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 0))
            .expect("test: add anchor x to v2 should succeed");
        t.add_anchor("y", v1, unit(2, 1))
            .expect("test: add anchor y to v1 should succeed");
        t.add_anchor("y", v2, unit(2, 1))
            .expect("test: add anchor y to v2 should succeed");
        let _ = t
            .compute_drift(v1, v2)
            .expect("test: compute drift should succeed");
        let stats = t.tracker_stats();
        assert_eq!(stats.total_versions, 2);
        assert_eq!(stats.active_versions, 1);
        assert_eq!(stats.distinct_concepts, 2);
        assert_eq!(stats.total_anchors, 4);
        assert_eq!(stats.drift_events, 2);
    }

    // ---- drift log ---------------------------------------------------------

    #[test]
    fn drift_log_bounded_at_500() {
        let mut t = tracker_with(0.0);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        // Register 600 concepts.
        for i in 0usize..600 {
            let c = format!("c{i}");
            t.add_anchor(&c, v1, unit(2, 0))
                .expect("test: add anchor to v1 should succeed");
            t.add_anchor(&c, v2, unit(2, 1))
                .expect("test: add anchor to v2 should succeed");
        }
        let _ = t
            .compute_drift(v1, v2)
            .expect("test: compute drift with 600 concepts should succeed");
        assert!(t.drift_log().len() <= 500);
    }

    #[test]
    fn clear_drift_log() {
        let mut t = tracker_with(0.1);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 0))
            .expect("test: add anchor x to v2 should succeed");
        let _ = t
            .compute_drift(v1, v2)
            .expect("test: compute drift should succeed");
        assert!(!t.drift_log().is_empty());
        t.clear_drift_log();
        assert!(t.drift_log().is_empty());
    }

    // ---- batch operations --------------------------------------------------

    #[test]
    fn add_anchors_batch_partial_failure() {
        let mut t = tracker();
        let v = t
            .register_version("v", 2)
            .expect("test: register version v should succeed");
        let items = vec![
            ("good".to_string(), v, vec![1.0, 0.0]),
            ("bad".to_string(), v, vec![1.0, 0.0, 0.0]), // dim mismatch
        ];
        let errors = t.add_anchors_batch(items);
        assert_eq!(errors.len(), 1);
        // "good" should still be registered.
        assert!(t.get_anchor("good", v).is_ok());
    }

    #[test]
    fn compute_all_consecutive_drifts() {
        let mut t = tracker_with(0.1);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        let v3 = t
            .register_version("v3", 2)
            .expect("test: register version v3 should succeed");
        for v in [v1, v2, v3] {
            t.add_anchor("x", v, unit(2, 0))
                .expect("test: add anchor x to version should succeed");
        }
        let results = t.compute_all_consecutive_drifts();
        assert_eq!(results.len(), 2);
        for r in results {
            assert!(r.is_ok());
        }
    }

    // ---- global_stability --------------------------------------------------

    #[test]
    fn global_stability_no_concepts_is_one() {
        let t = tracker();
        assert!((t.global_stability() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn global_stability_with_perfect_concepts() {
        let mut t = tracker();
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 0))
            .expect("test: add anchor x to v2 should succeed");
        let gs = t.global_stability();
        assert!((gs - 1.0).abs() < 1e-10);
    }

    // ---- concepts_by_stability ---------------------------------------------

    #[test]
    fn concepts_by_stability_sorted() {
        let mut t = tracker();
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("stable", v1, unit(2, 0))
            .expect("test: add anchor stable to v1 should succeed");
        t.add_anchor("stable", v2, unit(2, 0))
            .expect("test: add anchor stable to v2 should succeed");
        t.add_anchor("unstable", v1, unit(2, 0))
            .expect("test: add anchor unstable to v1 should succeed");
        t.add_anchor("unstable", v2, unit(2, 1))
            .expect("test: add anchor unstable to v2 should succeed");
        let sorted = t.concepts_by_stability();
        // stable should come first (score 1.0 > score 0.0).
        assert_eq!(sorted[0].0, "stable");
        assert_eq!(sorted[1].0, "unstable");
    }

    // ---- top_drifted_concepts ----------------------------------------------

    #[test]
    fn top_drifted_concepts_limits_result() {
        let mut t = tracker_with(0.0);
        let v1 = t
            .register_version("v1", 4)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 4)
            .expect("test: register version v2 should succeed");
        for i in 0..4usize {
            let c = format!("c{i}");
            t.add_anchor(&c, v1, unit(4, 0))
                .expect("test: add anchor to v1 should succeed");
            t.add_anchor(&c, v2, unit(4, (i + 1) % 4))
                .expect("test: add anchor to v2 should succeed");
        }
        let top2 = t
            .top_drifted_concepts(v1, v2, 2)
            .expect("test: top drifted concepts should succeed");
        assert!(top2.len() <= 2);
    }

    // ---- are_compatible ----------------------------------------------------

    #[test]
    fn are_compatible_identical_true() {
        let mut t = tracker_with(0.1);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 0))
            .expect("test: add anchor x to v2 should succeed");
        assert!(t
            .are_compatible(v1, v2)
            .expect("test: are_compatible should succeed"));
    }

    #[test]
    fn are_compatible_orthogonal_false() {
        let mut t = tracker_with(0.1);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 1))
            .expect("test: add anchor x to v2 should succeed");
        assert!(!t
            .are_compatible(v1, v2)
            .expect("test: are_compatible should succeed"));
    }

    // ---- remove_version_anchors --------------------------------------------

    #[test]
    fn remove_version_anchors_clears_entries() {
        let mut t = tracker();
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 0))
            .expect("test: add anchor x to v2 should succeed");
        let removed = t
            .remove_version_anchors(v1)
            .expect("test: remove version anchors should succeed");
        assert_eq!(removed, 1);
        assert!(t.get_anchor("x", v1).is_err());
        assert!(t.get_anchor("x", v2).is_ok());
    }

    #[test]
    fn remove_version_anchors_unknown_version_err() {
        let mut t = tracker();
        assert!(t.remove_version_anchors(9999).is_err());
    }

    // ---- concept_drift_matrix ----------------------------------------------

    #[test]
    fn concept_drift_matrix_correct() {
        let mut t = tracker();
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("same", v1, unit(2, 0))
            .expect("test: add anchor same to v1 should succeed");
        t.add_anchor("same", v2, unit(2, 0))
            .expect("test: add anchor same to v2 should succeed");
        t.add_anchor("diff", v1, unit(2, 0))
            .expect("test: add anchor diff to v1 should succeed");
        t.add_anchor("diff", v2, unit(2, 1))
            .expect("test: add anchor diff to v2 should succeed");
        let matrix = t
            .concept_drift_matrix(v1, v2)
            .expect("test: concept drift matrix should succeed");
        assert!((matrix["same"]).abs() < 1e-10);
        assert!((matrix["diff"] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn concept_drift_matrix_does_not_log() {
        let mut t = tracker();
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 0))
            .expect("test: add anchor x to v2 should succeed");
        let _ = t
            .concept_drift_matrix(v1, v2)
            .expect("test: concept drift matrix should succeed");
        // matrix should NOT push events to the drift log.
        assert!(t.drift_log().is_empty());
    }

    // ---- auto-deprecate ----------------------------------------------------

    #[test]
    fn auto_deprecate_deactivates_old_version() {
        let mut t = SemanticVersioningTracker::new(SvtTrackerConfig {
            drift_threshold: 0.1,
            min_anchors: 1,
            window_size: 10,
            auto_deprecate: true,
        });
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        // Overall drift will be 1.0 (orthogonal) >= 0.1 * 2 = 0.2.
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 1))
            .expect("test: add anchor x to v2 should succeed");
        let _ = t
            .compute_drift(v1, v2)
            .expect("test: compute drift should succeed");
        assert!(
            !t.get_version(v1)
                .expect("test: get version v1 after auto-deprecate should succeed")
                .is_active
        );
    }

    #[test]
    fn auto_deprecate_does_not_deactivate_when_drift_low() {
        let mut t = SemanticVersioningTracker::new(SvtTrackerConfig {
            drift_threshold: 0.5,
            min_anchors: 1,
            window_size: 10,
            auto_deprecate: true,
        });
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 0))
            .expect("test: add anchor x to v2 should succeed");
        let _ = t
            .compute_drift(v1, v2)
            .expect("test: compute drift should succeed");
        // drift = 0.0 < 0.5 * 2 = 1.0 → should NOT deprecate.
        assert!(
            t.get_version(v1)
                .expect("test: get version v1 should succeed")
                .is_active
        );
    }

    // ---- recommendation text -----------------------------------------------

    #[test]
    fn recommendation_transparent_when_very_low_drift() {
        let mut t = tracker_with(0.5);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 0))
            .expect("test: add anchor x to v2 should succeed");
        let r = t
            .compute_drift(v1, v2)
            .expect("test: compute drift should succeed");
        assert!(
            r.recommendation.contains("transparent") || r.recommendation.contains("compatible")
        );
    }

    #[test]
    fn recommendation_full_reindex_when_very_high_drift() {
        let mut t = tracker_with(0.1);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 1))
            .expect("test: add anchor x to v2 should succeed");
        let r = t
            .compute_drift(v1, v2)
            .expect("test: compute drift should succeed");
        assert!(r.recommendation.contains("incompatible") || r.recommendation.contains("re-index"));
    }

    // ---- drift event fields ------------------------------------------------

    #[test]
    fn drift_event_significant_flag() {
        let mut t = tracker_with(0.3);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 1))
            .expect("test: add anchor x to v2 should succeed"); // drift = 1.0 ≥ 0.3
        let _ = t
            .compute_drift(v1, v2)
            .expect("test: compute drift should succeed");
        let event = t
            .drift_log()
            .back()
            .expect("test: drift log should have at least one event after compute_drift");
        assert!(event.is_significant);
        assert!((event.drift_score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn drift_event_not_significant_for_identical() {
        let mut t = tracker_with(0.3);
        let v1 = t
            .register_version("v1", 2)
            .expect("test: register version v1 should succeed");
        let v2 = t
            .register_version("v2", 2)
            .expect("test: register version v2 should succeed");
        t.add_anchor("x", v1, unit(2, 0))
            .expect("test: add anchor x to v1 should succeed");
        t.add_anchor("x", v2, unit(2, 0))
            .expect("test: add anchor x to v2 should succeed");
        let _ = t
            .compute_drift(v1, v2)
            .expect("test: compute drift should succeed");
        let event = t
            .drift_log()
            .back()
            .expect("test: drift log should have at least one event after compute_drift");
        assert!(!event.is_significant);
    }

    // ---- config accessors --------------------------------------------------

    #[test]
    fn config_accessor_returns_correct_threshold() {
        let t = tracker_with(0.42);
        assert!((t.config().drift_threshold - 0.42).abs() < 1e-10);
    }

    #[test]
    fn config_mut_modifies_threshold() {
        let mut t = tracker_with(0.1);
        t.config_mut().drift_threshold = 0.9;
        assert!((t.config().drift_threshold - 0.9).abs() < 1e-10);
    }
}
