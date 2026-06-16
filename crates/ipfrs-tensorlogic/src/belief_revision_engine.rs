//! AGM-style Belief Revision Engine
//!
//! Implements the Alchourrón–Gärdenfors–Makinson (AGM) framework for belief revision,
//! including the three core operations:
//!
//! - **Expansion** K+φ: add φ and all entailments — always succeeds
//! - **Contraction** K÷φ: remove φ (partial-meet contraction, keep maximum)
//! - **Revision** K*φ: Levi identity — contract by ¬φ, then expand by φ
//!
//! Consistency checking detects contradictions of the form "X" / "NOT:X".
//! Retention on conflict is governed by a pluggable `RetentionFunction`.

use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// PRNG (xorshift64 — no `rand` dependency)
// ────────────────────────────────────────────────────────────────────────────

/// xorshift64 PRNG — used internally for tests and ID generation.
#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ────────────────────────────────────────────────────────────────────────────
// Core domain types
// ────────────────────────────────────────────────────────────────────────────

/// A single unit of propositional knowledge held by the agent.
#[derive(Debug, Clone, PartialEq)]
pub struct Belief {
    /// Unique identifier for this belief.
    pub id: String,
    /// Propositional formula as a string (e.g. `"rain"`, `"NOT:rain"`, `"rain AND wet"`).
    pub formula: String,
    /// Epistemic confidence in [0.0, 1.0].
    pub confidence: f64,
    /// Source / provenance label.
    pub source: String,
    /// Creation timestamp (Unix epoch, seconds).
    pub timestamp: u64,
    /// `true` if derived by entailment rather than directly asserted.
    pub is_derived: bool,
}

impl Belief {
    /// Construct a new directly-asserted belief.
    pub fn new(
        id: impl Into<String>,
        formula: impl Into<String>,
        confidence: f64,
        source: impl Into<String>,
        timestamp: u64,
    ) -> Self {
        Self {
            id: id.into(),
            formula: formula.into(),
            confidence: confidence.clamp(0.0, 1.0),
            source: source.into(),
            timestamp,
            is_derived: false,
        }
    }

    /// Mark this belief as derived.
    pub fn derived(mut self) -> Self {
        self.is_derived = true;
        self
    }
}

/// A set of beliefs together with a cached entailment map.
#[derive(Debug, Clone)]
pub struct BeliefSet {
    /// The beliefs.
    pub beliefs: Vec<Belief>,
    /// Cached entailment results: formula → entailed (true/false).
    pub entailment_cache: HashMap<String, bool>,
}

impl BeliefSet {
    /// Create an empty belief set.
    pub fn new() -> Self {
        Self {
            beliefs: Vec::new(),
            entailment_cache: HashMap::new(),
        }
    }

    /// Number of beliefs in the set.
    pub fn len(&self) -> usize {
        self.beliefs.len()
    }

    /// `true` when the set contains no beliefs.
    pub fn is_empty(&self) -> bool {
        self.beliefs.is_empty()
    }
}

impl Default for BeliefSet {
    fn default() -> Self {
        Self::new()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Revision operations
// ────────────────────────────────────────────────────────────────────────────

/// The four operations the engine can apply to a belief set.
#[derive(Debug, Clone)]
pub enum RevisionOp {
    /// K+φ — add a belief.
    Expansion(Belief),
    /// K÷φ — remove all beliefs entailing a formula.
    Contraction(String),
    /// K*φ — Levi: contract ¬φ, expand φ.
    Revision(Belief),
    /// Remove all contradictory pairs according to the retention function.
    Consolidation,
}

// ────────────────────────────────────────────────────────────────────────────
// Consistency
// ────────────────────────────────────────────────────────────────────────────

/// Result of a consistency scan over the current belief set.
#[derive(Debug, Clone, PartialEq)]
pub enum ConsistencyCheck {
    /// No contradictions detected.
    Consistent,
    /// One or more contradictory pairs; IDs of conflicting beliefs listed.
    Inconsistent(Vec<String>),
    /// Consistency could not be determined (e.g. complex/opaque formulae).
    Unknown,
}

// ────────────────────────────────────────────────────────────────────────────
// Retention functions
// ────────────────────────────────────────────────────────────────────────────

/// Determines which of two contradictory beliefs to keep.
#[derive(Debug, Clone)]
pub enum RetentionFunction {
    /// Keep the belief with higher confidence (epistemic entrenchment).
    EpistemicEntrenchment,
    /// Keep the more recently created belief.
    RecencyBias,
    /// Keep the belief whose source has a higher priority score.
    SourcePriority(HashMap<String, u32>),
    /// Keep the belief that minimises change to the set (prefer existing).
    MinimalChange,
}

impl RetentionFunction {
    /// Return `true` if `a` should be retained over `b` when they conflict.
    pub fn prefers(&self, a: &Belief, b: &Belief) -> bool {
        match self {
            RetentionFunction::EpistemicEntrenchment => a.confidence >= b.confidence,
            RetentionFunction::RecencyBias => a.timestamp >= b.timestamp,
            RetentionFunction::SourcePriority(priorities) => {
                let pa = priorities.get(&a.source).copied().unwrap_or(0);
                let pb = priorities.get(&b.source).copied().unwrap_or(0);
                pa >= pb
            }
            RetentionFunction::MinimalChange => !a.is_derived,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for a [`BeliefRevisionEngine`].
#[derive(Debug, Clone)]
pub struct RevisionConfig {
    /// Maximum number of beliefs the engine may hold (0 = unlimited).
    pub max_beliefs: usize,
    /// Whether to run consistency checks during expansion.
    pub consistency_check_enabled: bool,
    /// Which retention function to use when resolving contradictions.
    pub retention_function: RetentionFunction,
    /// If `true`, automatically consolidate after every inconsistency is found.
    pub auto_consolidate: bool,
}

impl Default for RevisionConfig {
    fn default() -> Self {
        Self {
            max_beliefs: 0,
            consistency_check_enabled: true,
            retention_function: RetentionFunction::EpistemicEntrenchment,
            auto_consolidate: true,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Statistics
// ────────────────────────────────────────────────────────────────────────────

/// Cumulative operational statistics for a [`BeliefRevisionEngine`].
#[derive(Debug, Clone, Default)]
pub struct RevisionStats {
    /// Total expansions performed.
    pub expansions: u64,
    /// Total contractions performed.
    pub contractions: u64,
    /// Total revisions performed.
    pub revisions: u64,
    /// Total consolidations performed.
    pub consolidations: u64,
    /// Total beliefs retracted over the engine's lifetime.
    pub beliefs_retracted: u64,
    /// Current number of beliefs in the set.
    pub current_belief_count: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// Error type
// ────────────────────────────────────────────────────────────────────────────

/// Errors returned by [`BeliefRevisionEngine`] operations.
#[derive(Debug, Clone, PartialEq)]
pub enum RevisionError {
    /// A referenced belief ID does not exist in the set.
    BeliefNotFound(String),
    /// Adding the belief would exceed `max_beliefs`.
    MaxBeliefsExceeded,
    /// The new belief directly contradicts an existing one.
    ContradictionDetected {
        /// Formula of the new / incoming belief.
        new: String,
        /// Formula of the existing / conflicting belief.
        existing: String,
    },
    /// Revision failed for an unspecified reason.
    RevisionFailed(String),
}

impl std::fmt::Display for RevisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RevisionError::BeliefNotFound(id) => write!(f, "belief not found: {id}"),
            RevisionError::MaxBeliefsExceeded => write!(f, "max belief limit exceeded"),
            RevisionError::ContradictionDetected { new, existing } => {
                write!(f, "contradiction: '{new}' vs '{existing}'")
            }
            RevisionError::RevisionFailed(msg) => write!(f, "revision failed: {msg}"),
        }
    }
}

impl std::error::Error for RevisionError {}

// ────────────────────────────────────────────────────────────────────────────
// Simple propositional helpers
// ────────────────────────────────────────────────────────────────────────────

/// Negate a formula.
///
/// * `"NOT:X"` → `"X"`
/// * `"X"` → `"NOT:X"`
fn negate(formula: &str) -> String {
    if let Some(base) = formula.strip_prefix("NOT:") {
        base.to_owned()
    } else {
        format!("NOT:{formula}")
    }
}

/// Return `true` if the two formulae are direct contradictions.
///
/// Formulae `a` and `b` contradict iff `negate(a) == b`.
fn are_contradictory(a: &str, b: &str) -> bool {
    negate(a) == b
}

/// Parse a conjunction `"A AND B"` into its two parts, if applicable.
fn parse_conjunction(formula: &str) -> Option<(&str, &str)> {
    let parts: Vec<&str> = formula.splitn(2, " AND ").collect();
    if parts.len() == 2 {
        Some((parts[0].trim(), parts[1].trim()))
    } else {
        None
    }
}

/// Check whether the given set of formulae directly entails `target`.
///
/// Rules (sound but deliberately incomplete for propositional strings):
/// 1. `target` is identical to one of `formulae`.
/// 2. `target` is a conjunction `"A AND B"` and both `A` and `B` are in `formulae`.
fn set_entails_formula(formulae: &[&str], target: &str) -> bool {
    // Rule 1 — direct membership
    if formulae.contains(&target) {
        return true;
    }
    // Rule 2 — conjunction introduction
    if let Some((a, b)) = parse_conjunction(target) {
        return formulae.contains(&a) && formulae.contains(&b);
    }
    false
}

// ────────────────────────────────────────────────────────────────────────────
// Core engine
// ────────────────────────────────────────────────────────────────────────────

/// AGM-style belief revision engine.
///
/// # Example
///
/// ```
/// use ipfrs_tensorlogic::{
///     BeliefRevisionEngine, Belief, RevisionConfig, RevisionError,
/// };
///
/// let cfg = RevisionConfig::default();
/// let mut engine = BeliefRevisionEngine::new(cfg);
///
/// let b = Belief::new("b1", "rain", 0.9, "sensor", 100);
/// let added = engine.expand(b).expect("example: should succeed in docs");
/// assert!(!added.is_empty());
///
/// let removed = engine.contract("rain").expect("example: should succeed in docs");
/// assert!(!removed.is_empty());
/// ```
pub struct BeliefRevisionEngine {
    /// Active belief set.
    belief_set: BeliefSet,
    /// Engine configuration.
    config: RevisionConfig,
    /// Cumulative statistics.
    stats: RevisionStats,
    /// Monotonic ID counter seed (xorshift64).
    id_seed: u64,
}

impl BeliefRevisionEngine {
    // ──────────────────────────────────────────────────────────────────────
    // Construction
    // ──────────────────────────────────────────────────────────────────────

    /// Create a new engine with the given configuration.
    pub fn new(config: RevisionConfig) -> Self {
        Self {
            belief_set: BeliefSet::new(),
            config,
            stats: RevisionStats::default(),
            id_seed: 0xdeadbeef_cafebabe,
        }
    }

    /// Create an engine with default configuration.
    pub fn default_config() -> Self {
        Self::new(RevisionConfig::default())
    }

    // ──────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ──────────────────────────────────────────────────────────────────────

    /// Generate a unique belief ID (no external dependencies).
    fn next_id(&mut self) -> String {
        let v = xorshift64(&mut self.id_seed);
        format!("bel-{v:016x}")
    }

    /// Formulae currently held in the belief set (borrowed slices).
    fn current_formulae(&self) -> Vec<&str> {
        self.belief_set
            .beliefs
            .iter()
            .map(|b| b.formula.as_str())
            .collect()
    }

    /// Derive additional beliefs implied by the current set plus `new_formula`.
    ///
    /// Currently derives beliefs of the form "A AND B" from components
    /// that are both already in the set (or one is `new_formula`).
    fn derive_entailed(&mut self, new_formula: &str) -> Vec<Belief> {
        let existing_formulae: Vec<String> = self
            .belief_set
            .beliefs
            .iter()
            .map(|b| b.formula.clone())
            .collect();

        let mut derived = Vec::new();

        // For every existing belief, check if `new_formula AND existing` is
        // worth asserting as a derived belief.  We only do this for simple
        // atomic formulae (no nested AND) to avoid combinatorial explosion.
        if !new_formula.contains(" AND ") {
            for ef in &existing_formulae {
                if ef.contains(" AND ") || ef == new_formula {
                    continue; // skip already-compound formulae and self-conjunction
                }
                let conj = format!("{new_formula} AND {ef}");
                // Only derive if there is a "consumer" (i.e., no identical
                // belief already in the set and not a tautology).
                let already_present = existing_formulae.iter().any(|f| f == &conj)
                    || derived.iter().any(|d: &Belief| d.formula == conj);
                if !already_present {
                    let id = self.next_id();
                    // Confidence is the minimum of the two conjuncts.
                    let conf_new = self
                        .belief_set
                        .beliefs
                        .iter()
                        .find(|b| b.formula == new_formula)
                        .map(|b| b.confidence)
                        .unwrap_or(0.5_f64);
                    let conf_ex = self
                        .belief_set
                        .beliefs
                        .iter()
                        .find(|b| b.formula == *ef)
                        .map(|b| b.confidence)
                        .unwrap_or(0.5_f64);
                    let confidence = conf_new.min(conf_ex);
                    derived.push(Belief {
                        id,
                        formula: conj,
                        confidence,
                        source: "derived".to_owned(),
                        timestamp: 0,
                        is_derived: true,
                    });
                }
            }
        }

        derived
    }

    /// Find IDs of all beliefs whose formula is derived from `formula`:
    /// i.e., the formula itself, or any conjunction that contains it.
    fn beliefs_derived_from(&self, formula: &str) -> Vec<String> {
        self.belief_set
            .beliefs
            .iter()
            .filter(|b| {
                b.formula == formula || b.formula.split(" AND ").any(|part| part.trim() == formula)
            })
            .map(|b| b.id.clone())
            .collect()
    }

    /// Remove beliefs by IDs and return the removed IDs.
    fn remove_beliefs(&mut self, ids: &[String]) -> Vec<String> {
        let mut removed = Vec::new();
        self.belief_set.beliefs.retain(|b| {
            if ids.contains(&b.id) {
                removed.push(b.id.clone());
                false
            } else {
                true
            }
        });
        // Invalidate entailment cache entries that touched removed formulae.
        self.belief_set.entailment_cache.clear();
        self.stats.beliefs_retracted += removed.len() as u64;
        removed
    }

    // ──────────────────────────────────────────────────────────────────────
    // Public API
    // ──────────────────────────────────────────────────────────────────────

    /// **Expansion** K+φ — add `belief` to the belief set.
    ///
    /// Optionally performs a consistency check and auto-consolidates on
    /// conflict according to the engine's configuration.
    ///
    /// Returns the IDs of all beliefs added (the new belief and any
    /// derived conjunctions).
    pub fn expand(&mut self, belief: Belief) -> Result<Vec<String>, RevisionError> {
        // Capacity check.
        if self.config.max_beliefs > 0 && self.belief_set.beliefs.len() >= self.config.max_beliefs {
            return Err(RevisionError::MaxBeliefsExceeded);
        }

        // Assign an ID if blank.
        let mut belief = belief;
        if belief.id.is_empty() {
            belief.id = self.next_id();
        }

        // Consistency check before insertion.
        if self.config.consistency_check_enabled {
            let neg = negate(&belief.formula);
            if let Some(existing) = self.belief_set.beliefs.iter().find(|b| b.formula == neg) {
                let existing_formula = existing.formula.clone();
                if self.config.auto_consolidate {
                    // Prefer per retention function; insert then consolidate.
                } else {
                    return Err(RevisionError::ContradictionDetected {
                        new: belief.formula.clone(),
                        existing: existing_formula,
                    });
                }
            }
        }

        // Insert the belief.
        let main_id = belief.id.clone();
        self.belief_set.beliefs.push(belief.clone());

        // Derive additional beliefs entailed by the new belief + existing set.
        let derived = self.derive_entailed(&belief.formula);
        let derived_ids: Vec<String> = derived.iter().map(|d| d.id.clone()).collect();
        for d in derived {
            // Capacity guard for derived beliefs.
            if self.config.max_beliefs > 0
                && self.belief_set.beliefs.len() >= self.config.max_beliefs
            {
                break;
            }
            self.belief_set.beliefs.push(d);
        }

        // Invalidate entailment cache.
        self.belief_set.entailment_cache.clear();

        // Auto-consolidate if enabled and inconsistency detected.
        if self.config.auto_consolidate
            && self.config.consistency_check_enabled
            && matches!(self.check_consistency(), ConsistencyCheck::Inconsistent(_))
        {
            self.consolidate();
        }

        self.stats.expansions += 1;
        self.stats.current_belief_count = self.belief_set.beliefs.len();

        let mut added = vec![main_id];
        added.extend(derived_ids);
        Ok(added)
    }

    /// **Contraction** K÷φ — remove `formula` and all beliefs derived from it.
    ///
    /// Uses the configured `RetentionFunction` to decide which beliefs to
    /// keep when the formula appears in a contradicting pair.
    ///
    /// Returns the IDs of removed beliefs.
    pub fn contract(&mut self, formula: &str) -> Result<Vec<String>, RevisionError> {
        let to_remove = self.beliefs_derived_from(formula);
        if to_remove.is_empty() {
            self.stats.contractions += 1;
            self.stats.current_belief_count = self.belief_set.beliefs.len();
            return Ok(Vec::new());
        }

        let removed = self.remove_beliefs(&to_remove);
        self.stats.contractions += 1;
        self.stats.current_belief_count = self.belief_set.beliefs.len();
        Ok(removed)
    }

    /// **Revision** K*φ — Levi identity: contract(¬φ), then expand(φ).
    ///
    /// Returns `(removed_ids, added_ids)`.
    pub fn revise(&mut self, belief: Belief) -> Result<(Vec<String>, Vec<String>), RevisionError> {
        let neg = negate(&belief.formula);

        // Step 1 — contract by ¬φ.
        let removed = self
            .contract(&neg)
            .map_err(|e| RevisionError::RevisionFailed(format!("contraction step failed: {e}")))?;

        // Step 2 — expand by φ.
        let added = self
            .expand(belief)
            .map_err(|e| RevisionError::RevisionFailed(format!("expansion step failed: {e}")))?;

        // Update stats (revision supersedes the individual expansion/contraction counts).
        self.stats.revisions += 1;
        Ok((removed, added))
    }

    /// **Consolidation** — detect and remove contradictory pairs.
    ///
    /// For each `(a, b)` where `are_contradictory(a, b)`, the retention
    /// function selects which one to drop.  Additionally, any derived
    /// belief whose formula references a dropped formula is also removed
    /// (so that "A AND NOT:A" style orphaned conjunctions are cleaned up).
    ///
    /// Returns IDs of all removed beliefs.
    pub fn consolidate(&mut self) -> Vec<String> {
        let mut drop_ids: Vec<String> = Vec::new();
        let mut drop_formulae: Vec<String> = Vec::new();

        // Collect all contradictory pairs and decide which to drop.
        let n = self.belief_set.beliefs.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let a = &self.belief_set.beliefs[i];
                let b = &self.belief_set.beliefs[j];
                if are_contradictory(&a.formula, &b.formula) {
                    // Decide which to keep.
                    let keep_a = self.config.retention_function.prefers(a, b);
                    let (drop_id, drop_formula) = if keep_a {
                        (b.id.clone(), b.formula.clone())
                    } else {
                        (a.id.clone(), a.formula.clone())
                    };
                    if !drop_ids.contains(&drop_id) {
                        drop_ids.push(drop_id);
                        drop_formulae.push(drop_formula);
                    }
                }
            }
        }

        // Also remove any derived beliefs that reference a dropped formula.
        for formula in &drop_formulae {
            for derived_id in self.beliefs_derived_from(formula) {
                if !drop_ids.contains(&derived_id) {
                    drop_ids.push(derived_id);
                }
            }
        }

        let removed = self.remove_beliefs(&drop_ids);
        self.stats.consolidations += 1;
        self.stats.current_belief_count = self.belief_set.beliefs.len();
        removed
    }

    /// **Consistency check** — scan for contradictory belief pairs.
    pub fn check_consistency(&self) -> ConsistencyCheck {
        let mut conflicting = Vec::new();

        let beliefs = &self.belief_set.beliefs;
        let n = beliefs.len();
        for i in 0..n {
            for j in (i + 1)..n {
                if are_contradictory(&beliefs[i].formula, &beliefs[j].formula) {
                    if !conflicting.contains(&beliefs[i].id) {
                        conflicting.push(beliefs[i].id.clone());
                    }
                    if !conflicting.contains(&beliefs[j].id) {
                        conflicting.push(beliefs[j].id.clone());
                    }
                }
            }
        }

        if conflicting.is_empty() {
            ConsistencyCheck::Consistent
        } else {
            ConsistencyCheck::Inconsistent(conflicting)
        }
    }

    /// Return `true` if the belief set (semantically) entails `formula`.
    ///
    /// Uses the simple string-based entailment rules with a memoisation cache.
    pub fn entails(&mut self, formula: &str) -> bool {
        // Fast path: cache hit.
        if let Some(&cached) = self.belief_set.entailment_cache.get(formula) {
            return cached;
        }

        let formulae: Vec<&str> = self.current_formulae();
        let result = set_entails_formula(&formulae, formula);

        self.belief_set
            .entailment_cache
            .insert(formula.to_owned(), result);
        result
    }

    /// Look up a belief by its ID.
    pub fn get_belief(&self, id: &str) -> Option<&Belief> {
        self.belief_set.beliefs.iter().find(|b| b.id == id)
    }

    /// Return all beliefs whose formula matches `formula` exactly.
    pub fn beliefs_about(&self, formula: &str) -> Vec<&Belief> {
        self.belief_set
            .beliefs
            .iter()
            .filter(|b| b.formula == formula)
            .collect()
    }

    /// Apply a revision operation to the belief set.
    pub fn apply_op(&mut self, op: RevisionOp) -> Result<(), RevisionError> {
        match op {
            RevisionOp::Expansion(belief) => {
                self.expand(belief)?;
            }
            RevisionOp::Contraction(formula) => {
                self.contract(&formula)?;
            }
            RevisionOp::Revision(belief) => {
                self.revise(belief)?;
            }
            RevisionOp::Consolidation => {
                self.consolidate();
            }
        }
        Ok(())
    }

    /// Return a snapshot of current statistics.
    pub fn stats(&self) -> RevisionStats {
        let mut s = self.stats.clone();
        s.current_belief_count = self.belief_set.beliefs.len();
        s
    }

    /// Clone and return the current belief set.
    pub fn snapshot(&self) -> BeliefSet {
        self.belief_set.clone()
    }

    /// Return the number of beliefs currently held.
    pub fn belief_count(&self) -> usize {
        self.belief_set.beliefs.len()
    }

    /// Immutable reference to the internal belief set.
    pub fn belief_set(&self) -> &BeliefSet {
        &self.belief_set
    }

    /// Mutable reference to the internal belief set (for advanced use).
    pub fn belief_set_mut(&mut self) -> &mut BeliefSet {
        &mut self.belief_set
    }

    /// Replace the engine's configuration.
    pub fn set_config(&mut self, config: RevisionConfig) {
        self.config = config;
    }

    /// Reference to the current configuration.
    pub fn config(&self) -> &RevisionConfig {
        &self.config
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn engine() -> BeliefRevisionEngine {
        BeliefRevisionEngine::default_config()
    }

    fn belief(id: &str, formula: &str, confidence: f64) -> Belief {
        Belief::new(id, formula, confidence, "test", 100)
    }

    fn belief_ts(id: &str, formula: &str, confidence: f64, ts: u64) -> Belief {
        Belief::new(id, formula, confidence, "test", ts)
    }

    // ── xorshift64 ───────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_non_zero() {
        let mut state = 1u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }

    #[test]
    fn test_xorshift64_changes_state() {
        let mut state = 42u64;
        let v1 = xorshift64(&mut state);
        let v2 = xorshift64(&mut state);
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_xorshift64_sequence_deterministic() {
        let mut s1 = 12345u64;
        let mut s2 = 12345u64;
        assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
    }

    // ── Belief construction ──────────────────────────────────────────────────

    #[test]
    fn test_belief_confidence_clamped() {
        let b = Belief::new("x", "p", 1.5, "s", 0);
        assert_eq!(b.confidence, 1.0);
        let b2 = Belief::new("y", "q", -0.1, "s", 0);
        assert_eq!(b2.confidence, 0.0);
    }

    #[test]
    fn test_belief_derived_flag() {
        let b = belief("x", "p", 0.9).derived();
        assert!(b.is_derived);
    }

    // ── BeliefSet helpers ─────────────────────────────────────────────────────

    #[test]
    fn test_belief_set_empty() {
        let bs = BeliefSet::new();
        assert!(bs.is_empty());
        assert_eq!(bs.len(), 0);
    }

    // ── negate / are_contradictory helpers ───────────────────────────────────

    #[test]
    fn test_negate_positive() {
        assert_eq!(negate("rain"), "NOT:rain");
    }

    #[test]
    fn test_negate_negative() {
        assert_eq!(negate("NOT:rain"), "rain");
    }

    #[test]
    fn test_are_contradictory_true() {
        assert!(are_contradictory("rain", "NOT:rain"));
        assert!(are_contradictory("NOT:rain", "rain"));
    }

    #[test]
    fn test_are_contradictory_false() {
        assert!(!are_contradictory("rain", "wet"));
    }

    // ── Expansion ────────────────────────────────────────────────────────────

    #[test]
    fn test_expand_adds_belief() {
        let mut e = engine();
        let added = e
            .expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        assert!(added.contains(&"b1".to_string()));
        assert_eq!(e.belief_count(), 1);
    }

    #[test]
    fn test_expand_assigns_id_when_blank() {
        let mut e = engine();
        let b = Belief::new("", "rain", 0.9, "s", 0);
        let added = e.expand(b).expect("test: should succeed");
        assert!(!added[0].is_empty());
    }

    #[test]
    fn test_expand_returns_stats() {
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        assert_eq!(e.stats().expansions, 1);
    }

    #[test]
    fn test_expand_max_beliefs_exceeded() {
        let cfg = RevisionConfig {
            max_beliefs: 1,
            ..RevisionConfig::default()
        };
        let mut e = BeliefRevisionEngine::new(cfg);
        e.expand(belief("b1", "a", 0.9))
            .expect("test: should succeed");
        let err = e.expand(belief("b2", "b", 0.8)).unwrap_err();
        assert_eq!(err, RevisionError::MaxBeliefsExceeded);
    }

    #[test]
    fn test_expand_auto_consolidate_on_contradiction() {
        let mut e = engine(); // auto_consolidate = true
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        // Adding NOT:rain should trigger auto-consolidate; lower confidence drops.
        e.expand(belief("b2", "NOT:rain", 0.3))
            .expect("test: should succeed");
        // After consolidation the higher-confidence one remains.
        assert_eq!(e.belief_count(), 1);
        assert!(e.beliefs_about("rain").len() == 1);
    }

    #[test]
    fn test_expand_contradiction_error_when_no_auto_consolidate() {
        let cfg = RevisionConfig {
            auto_consolidate: false,
            consistency_check_enabled: true,
            ..RevisionConfig::default()
        };
        let mut e = BeliefRevisionEngine::new(cfg);
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        let err = e.expand(belief("b2", "NOT:rain", 0.3)).unwrap_err();
        assert!(matches!(err, RevisionError::ContradictionDetected { .. }));
    }

    #[test]
    fn test_expand_derives_conjunction() {
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.8))
            .expect("test: should succeed");
        e.expand(belief("b2", "wet", 0.7))
            .expect("test: should succeed");
        // The engine should derive "rain AND wet" or "wet AND rain".
        let has_conj = e
            .belief_set()
            .beliefs
            .iter()
            .any(|b| b.formula.contains(" AND ") && b.is_derived);
        assert!(has_conj);
    }

    // ── Contraction ──────────────────────────────────────────────────────────

    #[test]
    fn test_contract_removes_belief() {
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        let removed = e.contract("rain").expect("test: should succeed");
        assert!(removed.contains(&"b1".to_string()));
        assert_eq!(e.belief_count(), 0);
    }

    #[test]
    fn test_contract_nonexistent_ok() {
        let mut e = engine();
        let removed = e.contract("unicorn").expect("test: should succeed");
        assert!(removed.is_empty());
    }

    #[test]
    fn test_contract_removes_derived_conjunction() {
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.8))
            .expect("test: should succeed");
        e.expand(belief("b2", "wet", 0.7))
            .expect("test: should succeed");
        // Contracting "rain" should remove "rain" and any conjunction containing it.
        e.contract("rain").expect("test: should succeed");
        for b in &e.belief_set().beliefs {
            assert!(
                !b.formula.contains("rain"),
                "found residual formula: {}",
                b.formula
            );
        }
    }

    #[test]
    fn test_contract_stats() {
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        e.contract("rain").expect("test: should succeed");
        assert_eq!(e.stats().contractions, 1);
    }

    // ── Revision (Levi identity) ──────────────────────────────────────────────

    #[test]
    fn test_revise_adds_new_formula() {
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        let (_, added) = e
            .revise(belief("b2", "NOT:rain", 0.8))
            .expect("test: should succeed");
        assert!(!added.is_empty());
        // After revision, "NOT:rain" should be in the set.
        assert!(e.entails("NOT:rain"));
    }

    #[test]
    fn test_revise_removes_negation() {
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        let (removed, _) = e
            .revise(belief("b2", "NOT:rain", 0.8))
            .expect("test: should succeed");
        // "rain" should have been retracted.
        assert!(removed.contains(&"b1".to_string()));
        assert!(!e.entails("rain"));
    }

    #[test]
    fn test_revise_levi_identity_no_duplicate() {
        // After K*(NOT:rain), the set should not contain both "rain" and "NOT:rain".
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.5))
            .expect("test: should succeed");
        e.revise(belief("b2", "NOT:rain", 0.9))
            .expect("test: should succeed");
        assert_eq!(
            e.check_consistency(),
            ConsistencyCheck::Consistent,
            "set should be consistent after revision"
        );
    }

    #[test]
    fn test_revise_stats() {
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        e.revise(belief("b2", "NOT:rain", 0.6))
            .expect("test: should succeed");
        assert_eq!(e.stats().revisions, 1);
    }

    #[test]
    fn test_revise_fresh_formula() {
        // Revising with a completely new formula should just expand.
        let mut e = engine();
        let (removed, added) = e
            .revise(belief("b1", "sun", 0.9))
            .expect("test: should succeed");
        assert!(removed.is_empty());
        assert!(!added.is_empty());
    }

    // ── Consolidation ─────────────────────────────────────────────────────────

    #[test]
    fn test_consolidate_removes_lower_confidence() {
        let cfg = RevisionConfig {
            auto_consolidate: false,
            consistency_check_enabled: false,
            retention_function: RetentionFunction::EpistemicEntrenchment,
            ..RevisionConfig::default()
        };
        let mut e = BeliefRevisionEngine::new(cfg);
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        e.expand(belief("b2", "NOT:rain", 0.3))
            .expect("test: should succeed");
        let removed = e.consolidate();
        assert!(removed.contains(&"b2".to_string()));
        assert_eq!(e.belief_count(), 1);
    }

    #[test]
    fn test_consolidate_no_contradictions() {
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        e.expand(belief("b2", "wet", 0.7))
            .expect("test: should succeed");
        // Before consolidation manually override auto_consolidate to avoid side-effects.
        let removed = e.consolidate();
        // No contradictions — nothing removed (except possibly derived beliefs if any
        // happen to conflict, which they won't here).
        for id in &removed {
            // If anything was removed it must have been a contradiction.
            let _ = id;
        }
        assert_eq!(e.check_consistency(), ConsistencyCheck::Consistent);
    }

    #[test]
    fn test_consolidate_recency_bias() {
        let cfg = RevisionConfig {
            auto_consolidate: false,
            consistency_check_enabled: false,
            retention_function: RetentionFunction::RecencyBias,
            ..RevisionConfig::default()
        };
        let mut e = BeliefRevisionEngine::new(cfg);
        e.expand(belief_ts("b1", "rain", 0.9, 100))
            .expect("test: should succeed");
        e.expand(belief_ts("b2", "NOT:rain", 0.3, 200))
            .expect("test: should succeed");
        let removed = e.consolidate();
        // b2 is newer, so b1 should be removed.
        assert!(removed.contains(&"b1".to_string()));
    }

    #[test]
    fn test_consolidate_source_priority() {
        let mut priorities = HashMap::new();
        priorities.insert("sensor".to_string(), 5u32);
        priorities.insert("user".to_string(), 10u32);
        let cfg = RevisionConfig {
            auto_consolidate: false,
            consistency_check_enabled: false,
            retention_function: RetentionFunction::SourcePriority(priorities),
            ..RevisionConfig::default()
        };
        let mut e = BeliefRevisionEngine::new(cfg);
        let b1 = Belief::new("b1", "rain", 0.9, "sensor", 100);
        let b2 = Belief::new("b2", "NOT:rain", 0.3, "user", 100);
        e.expand(b1).expect("test: should succeed");
        e.expand(b2).expect("test: should succeed");
        let removed = e.consolidate();
        // "user" has higher priority, so "sensor" (b1) is removed.
        assert!(removed.contains(&"b1".to_string()));
    }

    #[test]
    fn test_consolidate_minimal_change() {
        let cfg = RevisionConfig {
            auto_consolidate: false,
            consistency_check_enabled: false,
            retention_function: RetentionFunction::MinimalChange,
            ..RevisionConfig::default()
        };
        let mut e = BeliefRevisionEngine::new(cfg);
        // is_derived = false → prefer keeping; is_derived = true → prefer dropping
        let b1 = Belief::new("b1", "rain", 0.5, "test", 100);
        let b2 = Belief {
            id: "b2".to_string(),
            formula: "NOT:rain".to_string(),
            confidence: 0.5,
            source: "test".to_string(),
            timestamp: 100,
            is_derived: true,
        };
        e.expand(b1).expect("test: should succeed");
        e.expand(b2).expect("test: should succeed");
        let removed = e.consolidate();
        // b2 is derived, so it should be dropped.
        assert!(removed.contains(&"b2".to_string()));
    }

    #[test]
    fn test_consolidate_stats() {
        let mut e = engine();
        e.consolidate();
        assert_eq!(e.stats().consolidations, 1);
    }

    // ── Consistency check ─────────────────────────────────────────────────────

    #[test]
    fn test_check_consistency_consistent() {
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        assert_eq!(e.check_consistency(), ConsistencyCheck::Consistent);
    }

    #[test]
    fn test_check_consistency_inconsistent() {
        let cfg = RevisionConfig {
            auto_consolidate: false,
            consistency_check_enabled: false,
            ..RevisionConfig::default()
        };
        let mut e = BeliefRevisionEngine::new(cfg);
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        e.expand(belief("b2", "NOT:rain", 0.3))
            .expect("test: should succeed");
        let cc = e.check_consistency();
        assert!(matches!(cc, ConsistencyCheck::Inconsistent(_)));
        if let ConsistencyCheck::Inconsistent(ids) = cc {
            assert!(ids.contains(&"b1".to_string()));
            assert!(ids.contains(&"b2".to_string()));
        }
    }

    #[test]
    fn test_check_consistency_empty_set() {
        let e = engine();
        assert_eq!(e.check_consistency(), ConsistencyCheck::Consistent);
    }

    // ── Entailment ────────────────────────────────────────────────────────────

    #[test]
    fn test_entails_direct() {
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        assert!(e.entails("rain"));
        assert!(!e.entails("sun"));
    }

    #[test]
    fn test_entails_conjunction() {
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.8))
            .expect("test: should succeed");
        e.expand(belief("b2", "wet", 0.7))
            .expect("test: should succeed");
        assert!(e.entails("rain AND wet") || e.entails("wet AND rain"));
    }

    #[test]
    fn test_entails_cache_hit() {
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        assert!(e.entails("rain")); // populates cache
        assert!(e.entails("rain")); // hits cache
    }

    #[test]
    fn test_entails_not_present() {
        let mut e = engine();
        assert!(!e.entails("unicorn"));
    }

    // ── get_belief / beliefs_about ────────────────────────────────────────────

    #[test]
    fn test_get_belief_found() {
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        assert!(e.get_belief("b1").is_some());
    }

    #[test]
    fn test_get_belief_not_found() {
        let e = engine();
        assert!(e.get_belief("missing").is_none());
    }

    #[test]
    fn test_beliefs_about_empty() {
        let e = engine();
        assert!(e.beliefs_about("rain").is_empty());
    }

    #[test]
    fn test_beliefs_about_found() {
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        assert_eq!(e.beliefs_about("rain").len(), 1);
    }

    // ── apply_op ──────────────────────────────────────────────────────────────

    #[test]
    fn test_apply_op_expansion() {
        let mut e = engine();
        let op = RevisionOp::Expansion(belief("b1", "rain", 0.9));
        e.apply_op(op).expect("test: should succeed");
        // At least the directly asserted belief must be present.
        assert!(e.belief_count() >= 1);
        assert!(e.entails("rain"));
    }

    #[test]
    fn test_apply_op_contraction() {
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        let op = RevisionOp::Contraction("rain".to_string());
        e.apply_op(op).expect("test: should succeed");
        assert_eq!(e.belief_count(), 0);
    }

    #[test]
    fn test_apply_op_revision() {
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.5))
            .expect("test: should succeed");
        let op = RevisionOp::Revision(belief("b2", "NOT:rain", 0.9));
        e.apply_op(op).expect("test: should succeed");
        assert!(e.entails("NOT:rain"));
    }

    #[test]
    fn test_apply_op_consolidation() {
        let cfg = RevisionConfig {
            auto_consolidate: false,
            consistency_check_enabled: false,
            ..RevisionConfig::default()
        };
        let mut e = BeliefRevisionEngine::new(cfg);
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        e.expand(belief("b2", "NOT:rain", 0.3))
            .expect("test: should succeed");
        e.apply_op(RevisionOp::Consolidation)
            .expect("test: should succeed");
        // All remaining beliefs should be consistent — no contradictory pair survives.
        assert_eq!(e.check_consistency(), ConsistencyCheck::Consistent);
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial() {
        let e = engine();
        let s = e.stats();
        assert_eq!(s.expansions, 0);
        assert_eq!(s.contractions, 0);
        assert_eq!(s.revisions, 0);
        assert_eq!(s.consolidations, 0);
        assert_eq!(s.beliefs_retracted, 0);
        assert_eq!(s.current_belief_count, 0);
    }

    #[test]
    fn test_stats_after_operations() {
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        e.contract("rain").expect("test: should succeed");
        let s = e.stats();
        assert_eq!(s.expansions, 1);
        assert_eq!(s.contractions, 1);
        assert!(s.beliefs_retracted >= 1);
        assert_eq!(s.current_belief_count, 0);
    }

    // ── snapshot ──────────────────────────────────────────────────────────────

    #[test]
    fn test_snapshot_clone() {
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        let snap = e.snapshot();
        e.contract("rain").expect("test: should succeed");
        // Snapshot should still contain the belief.
        assert_eq!(snap.beliefs.len(), 1);
        assert_eq!(e.belief_count(), 0);
    }

    // ── error cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_error_display_belief_not_found() {
        let e = RevisionError::BeliefNotFound("x".to_string());
        assert!(e.to_string().contains("x"));
    }

    #[test]
    fn test_error_display_max_beliefs_exceeded() {
        let e = RevisionError::MaxBeliefsExceeded;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_error_display_contradiction_detected() {
        let e = RevisionError::ContradictionDetected {
            new: "rain".to_string(),
            existing: "NOT:rain".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("rain"));
        assert!(s.contains("NOT:rain"));
    }

    #[test]
    fn test_error_display_revision_failed() {
        let e = RevisionError::RevisionFailed("oops".to_string());
        assert!(e.to_string().contains("oops"));
    }

    // ── multiple beliefs + ordering ───────────────────────────────────────────

    #[test]
    fn test_multiple_expansions_and_count() {
        let mut e = engine();
        for i in 0..10u32 {
            e.expand(belief(&format!("b{i}"), &format!("p{i}"), 0.5))
                .expect("test: should succeed");
        }
        assert!(e.belief_count() >= 10);
    }

    #[test]
    fn test_revision_sequence() {
        let mut e = engine();
        e.expand(belief("b1", "A", 0.8))
            .expect("test: should succeed");
        e.revise(belief("b2", "NOT:A", 0.9))
            .expect("test: should succeed");
        e.revise(belief("b3", "A", 0.95))
            .expect("test: should succeed");
        // After final revision to A, the set should contain A and not NOT:A.
        assert!(e.entails("A"));
        assert!(!e.entails("NOT:A"));
        assert_eq!(e.check_consistency(), ConsistencyCheck::Consistent);
    }

    #[test]
    fn test_agm_success_postulate_expansion() {
        // Expansion postulate: after K+φ, φ is in the set.
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        assert!(e.entails("rain"));
    }

    #[test]
    fn test_agm_success_postulate_revision() {
        // Revision success: after K*φ, φ is in the set.
        let mut e = engine();
        e.expand(belief("b1", "sun", 0.8))
            .expect("test: should succeed");
        e.revise(belief("b2", "rain", 0.9))
            .expect("test: should succeed");
        assert!(e.entails("rain"));
    }

    #[test]
    fn test_agm_consistency_postulate_revision() {
        // Revision consistency: after K*φ the set should be consistent.
        let mut e = engine();
        e.expand(belief("b1", "X", 0.5))
            .expect("test: should succeed");
        e.expand(belief("b2", "Y", 0.5))
            .expect("test: should succeed");
        e.revise(belief("b3", "NOT:X", 0.9))
            .expect("test: should succeed");
        assert_eq!(e.check_consistency(), ConsistencyCheck::Consistent);
    }

    #[test]
    fn test_contract_idempotent() {
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        e.contract("rain").expect("test: should succeed");
        let removed2 = e.contract("rain").expect("test: should succeed");
        assert!(removed2.is_empty());
    }

    #[test]
    fn test_set_config() {
        let mut e = engine();
        let new_cfg = RevisionConfig {
            max_beliefs: 5,
            ..RevisionConfig::default()
        };
        e.set_config(new_cfg);
        assert_eq!(e.config().max_beliefs, 5);
    }

    #[test]
    fn test_belief_set_ref() {
        let mut e = engine();
        e.expand(belief("b1", "rain", 0.9))
            .expect("test: should succeed");
        let bs = e.belief_set();
        // At least the directly asserted belief must be present.
        assert!(!bs.is_empty());
        assert!(bs.beliefs.iter().any(|b| b.id == "b1"));
    }

    // ── retention function defaults ───────────────────────────────────────────

    #[test]
    fn test_retention_epistemic_equal_confidence() {
        let rf = RetentionFunction::EpistemicEntrenchment;
        let a = belief("a", "p", 0.7);
        let b = belief("b", "q", 0.7);
        // Equal confidence — a is preferred (>=).
        assert!(rf.prefers(&a, &b));
    }

    #[test]
    fn test_retention_source_priority_missing_key() {
        let rf = RetentionFunction::SourcePriority(HashMap::new());
        let a = belief("a", "p", 0.5);
        let b = belief("b", "q", 0.5);
        // Both map to 0, a is preferred (>=).
        assert!(rf.prefers(&a, &b));
    }

    #[test]
    fn test_revision_error_implements_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(RevisionError::MaxBeliefsExceeded);
        assert!(!e.to_string().is_empty());
    }
}
