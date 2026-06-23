//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};

use super::constants::DRIFT_LOG_CAP;
use super::functions::{cosine_distance, cosine_similarity, xorshift64};
use super::type_aliases::SvtVersionId;

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
    pub(super) versions: HashMap<SvtVersionId, SvtVersion>,
    /// Anchor concept → per-version (id, embedding) pairs.
    pub(super) anchors: HashMap<String, Vec<(SvtVersionId, Vec<f64>)>>,
    /// Bounded log of recorded drift events.
    pub(super) drift_log: VecDeque<SvtDriftEvent>,
    /// Tracker configuration.
    pub(super) config: SvtTrackerConfig,
    /// Monotonically increasing ID counter.
    pub(super) next_id: SvtVersionId,
    /// xorshift64 PRNG state (used for tiebreaking / synthetic timestamps).
    pub(super) rng_state: u64,
}
impl SemanticVersioningTracker {
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
    pub(super) fn now_ts(&mut self) -> u64 {
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
        if !self.versions.contains_key(&ver_a) {
            return Err(SvtError::VersionNotFound(ver_a));
        }
        if !self.versions.contains_key(&ver_b) {
            return Err(SvtError::VersionNotFound(ver_b));
        }
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
        let overall_drift = if concept_scores.is_empty() {
            0.0
        } else {
            concept_scores.iter().map(|(_, s)| s).sum::<f64>() / concept_scores.len() as f64
        };
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
        let recommendation =
            self.build_recommendation(ver_a, ver_b, overall_drift, &drifted_concepts);
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
    /// Pushes an event to the drift log, discarding the oldest entry when
    /// the log is at capacity.
    pub(super) fn push_drift_event(&mut self, event: SvtDriftEvent) {
        if self.drift_log.len() >= DRIFT_LOG_CAP {
            self.drift_log.pop_front();
        }
        self.drift_log.push_back(event);
    }
    /// Builds a human-readable recommendation string.
    pub(super) fn build_recommendation(
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
        self.anchors.retain(|_, entries| !entries.is_empty());
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
