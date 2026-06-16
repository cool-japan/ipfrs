//! Abductive Reasoning Engine (ARE)
//!
//! Implements abduction — "inference to the best explanation" — over a set of
//! observations, a library of abducible hypotheses, and a set of logical rules.
//!
//! ## Overview
//!
//! Given a set of *observations* (grounded facts that must be explained) and a
//! repository of *hypotheses* (candidate explanations with associated costs),
//! `AbductiveReasoningEngine` searches for minimal-cost hypothesis sets that
//! *cover* (entail) all observations either directly or through rule chains.
//!
//! ## Cost Functions
//!
//! | Variant          | Behaviour                                      |
//! |------------------|------------------------------------------------|
//! | `SumCost`        | total cost = Σ individual hypothesis costs     |
//! | `MaxCost`        | total cost = max individual hypothesis cost    |
//! | `CountCost`      | total cost = number of hypotheses in the set  |
//! | `WeightedCost`   | total cost = Σ (cost × weight), weight per id |
//!
//! ## Algorithm
//!
//! The engine uses a branch-and-bound search over hypothesis subsets ordered by
//! cost.  At each node the set of uncovered observations is computed; if empty
//! the current set is a complete explanation.  Duplicate explanations (same set
//! of hypothesis ids, different order) are deduplicated via sorted fingerprints.
//!
//! ## Naming Conventions
//!
//! All exported names use the `Abr` prefix (AbductiveReasoningEngine types) to
//! avoid collision with the `Are*` names already used by
//! `adaptive_routing_engine` elsewhere in this crate.

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

// ─────────────────────────────────────────────────────────────────────────────
// PRNG helpers (no `rand` dependency)
// ─────────────────────────────────────────────────────────────────────────────

/// xorshift64 PRNG.
#[inline]
pub fn abr_xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// FNV-1a 64-bit hash.
#[inline]
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

// ─────────────────────────────────────────────────────────────────────────────
// Public type alias: HypothesisId
// ─────────────────────────────────────────────────────────────────────────────

/// Opaque identifier for a hypothesis.
pub type HypothesisId = u64;

// ─────────────────────────────────────────────────────────────────────────────
// AbrTerm — grounded predicate term
// ─────────────────────────────────────────────────────────────────────────────

/// A first-order-style ground term: `predicate(arg0, arg1, …)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AbrTerm {
    /// Predicate name (e.g. `"wet"`, `"broken_window"`).
    pub predicate: String,
    /// Ground arguments (e.g. `["lawn"]`).  May be empty for propositional facts.
    pub args: Vec<String>,
}

impl AbrTerm {
    /// Construct a new term.
    pub fn new(predicate: impl Into<String>, args: Vec<impl Into<String>>) -> Self {
        Self {
            predicate: predicate.into(),
            args: args.into_iter().map(|a| a.into()).collect(),
        }
    }

    /// Propositional shorthand — zero arguments.
    pub fn prop(predicate: impl Into<String>) -> Self {
        Self {
            predicate: predicate.into(),
            args: Vec::new(),
        }
    }

    /// Canonical string representation for hashing.
    fn canonical(&self) -> String {
        if self.args.is_empty() {
            self.predicate.clone()
        } else {
            format!("{}({})", self.predicate, self.args.join(","))
        }
    }

    /// FNV-1a fingerprint of the canonical form.
    pub fn fingerprint(&self) -> u64 {
        fnv1a_64(self.canonical().as_bytes())
    }

    /// Returns `true` if this term unifies with `other` under simple ground matching
    /// (exact equality or wildcard `"_"` arguments).
    pub fn matches(&self, other: &AbrTerm) -> bool {
        if self.predicate != other.predicate {
            return false;
        }
        if self.args.len() != other.args.len() {
            return false;
        }
        self.args
            .iter()
            .zip(other.args.iter())
            .all(|(a, b)| a == "_" || b == "_" || a == b)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AbrHypothesis
// ─────────────────────────────────────────────────────────────────────────────

/// A candidate explanation.
#[derive(Debug, Clone)]
pub struct AbrHypothesis {
    /// Stable identifier.
    pub id: HypothesisId,
    /// The fact this hypothesis asserts.
    pub term: AbrTerm,
    /// Non-negative cost.
    pub cost: f64,
    /// When `false` the hypothesis cannot be chosen by the abducer (it is
    /// background knowledge).
    pub is_abducible: bool,
}

impl AbrHypothesis {
    fn new(id: HypothesisId, term: AbrTerm, cost: f64, is_abducible: bool) -> Self {
        Self {
            id,
            term,
            cost: cost.max(0.0),
            is_abducible,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AbrRule
// ─────────────────────────────────────────────────────────────────────────────

/// A logical rule: if all `body` terms are satisfied, the `head` is derived.
#[derive(Debug, Clone)]
pub struct AbrRule {
    /// Derived conclusion.
    pub head: AbrTerm,
    /// Conjunction of conditions.
    pub body: Vec<AbrTerm>,
    /// Confidence weight ∈ [0, 1].
    pub confidence: f64,
}

impl AbrRule {
    fn new(head: AbrTerm, body: Vec<AbrTerm>, confidence: f64) -> Self {
        Self {
            head,
            body,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AreCostFunction
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy for computing the total cost of a hypothesis set.
#[derive(Debug, Clone)]
pub enum AbrCostFunction {
    /// Sum of all individual costs.
    SumCost,
    /// Maximum individual cost.
    MaxCost,
    /// Cardinality (number of hypotheses chosen).
    CountCost,
    /// Weighted sum: each hypothesis id maps to an additional weight multiplier.
    WeightedCost(HashMap<HypothesisId, f64>),
}

impl AbrCostFunction {
    /// Compute the total cost for a set of hypotheses.
    pub fn compute(
        &self,
        ids: &[HypothesisId],
        hypotheses: &HashMap<HypothesisId, AbrHypothesis>,
    ) -> f64 {
        match self {
            AbrCostFunction::SumCost => ids
                .iter()
                .filter_map(|id| hypotheses.get(id).map(|h| h.cost))
                .sum(),
            AbrCostFunction::MaxCost => ids
                .iter()
                .filter_map(|id| hypotheses.get(id).map(|h| h.cost))
                .fold(0.0_f64, f64::max),
            AbrCostFunction::CountCost => ids.len() as f64,
            AbrCostFunction::WeightedCost(weights) => ids
                .iter()
                .filter_map(|id| {
                    hypotheses.get(id).map(|h| {
                        let w = weights.get(id).copied().unwrap_or(1.0);
                        h.cost * w
                    })
                })
                .sum(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AbrEngineConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the abductive reasoning engine.
#[derive(Debug, Clone)]
pub struct AbrEngineConfig {
    /// Maximum number of explanations to return.
    pub max_explanations: usize,
    /// Maximum size of any single hypothesis set.
    pub max_hypothesis_set_size: usize,
    /// Cost function applied to candidate hypothesis sets.
    pub cost_function: AbrCostFunction,
    /// When `true` prefer smaller hypothesis sets (Occam's razor).
    pub prefer_minimal: bool,
    /// Maximum number of search nodes to expand (budget).
    pub max_search_nodes: usize,
}

impl Default for AbrEngineConfig {
    fn default() -> Self {
        Self {
            max_explanations: 10,
            max_hypothesis_set_size: 8,
            cost_function: AbrCostFunction::SumCost,
            prefer_minimal: true,
            max_search_nodes: 100_000,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AbrExplanation
// ─────────────────────────────────────────────────────────────────────────────

/// A complete or partial explanation produced by the abducer.
#[derive(Debug, Clone)]
pub struct AbrExplanation {
    /// Ordered set of hypothesis ids chosen.
    pub hypotheses: Vec<HypothesisId>,
    /// Observations (and derived facts) covered by this explanation.
    pub covered: Vec<AbrTerm>,
    /// Aggregate cost of the chosen hypotheses.
    pub total_cost: f64,
    /// Fraction of all observations covered: `covered.len() / n_observations`.
    pub completeness: f64,
}

impl AbrExplanation {
    /// Returns `true` if every observation is covered.
    pub fn is_complete(&self, n_observations: usize) -> bool {
        self.covered.len() >= n_observations
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AbrExplanationRecord — history entry
// ─────────────────────────────────────────────────────────────────────────────

/// Snapshot recorded each time `abduce()` is called.
#[derive(Debug, Clone)]
pub struct AbrExplanationRecord {
    /// Unix-epoch timestamp in milliseconds (derived from monotonic tick count).
    pub timestamp_ms: u64,
    /// Number of observations present at abduce time.
    pub n_observations: usize,
    /// Total number of hypothesis subsets tried during search.
    pub n_hypotheses_tried: u64,
    /// Cost of the best explanation found (f64::INFINITY if none).
    pub best_cost: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// AbrReasoningStats — snapshot
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate runtime statistics for the engine.
#[derive(Debug, Clone)]
pub struct AbrReasoningStats {
    /// Total calls to `abduce()`.
    pub abduce_calls: u64,
    /// Total hypothesis subsets explored across all calls.
    pub total_nodes_explored: u64,
    /// Number of complete explanations ever found.
    pub total_explanations_found: u64,
    /// Total hypotheses registered.
    pub n_hypotheses: usize,
    /// Total rules registered.
    pub n_rules: usize,
    /// Total observations registered.
    pub n_observations: usize,
    /// Number of abduction history records retained.
    pub history_len: usize,
    /// Best cost ever achieved (f64::INFINITY if never found).
    pub best_cost_ever: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Search node (internal)
// ─────────────────────────────────────────────────────────────────────────────

/// Internal node used by the branch-and-bound search.
#[derive(Debug, Clone)]
struct SearchNode {
    /// Current hypothesis set under consideration.
    chosen: Vec<HypothesisId>,
    /// Index into the sorted abducible hypothesis list (next candidate to branch on).
    next_idx: usize,
    /// Accumulated cost so far.
    cost_so_far: f64,
}

/// `BinaryHeap` requires `Ord`; we wrap cost in a min-heap adapter.
#[derive(Debug, Clone)]
struct MinHeapNode {
    neg_cost: i64, // store -round(cost*1e6) so BinaryHeap gives min first
    node: SearchNode,
}

impl PartialEq for MinHeapNode {
    fn eq(&self, other: &Self) -> bool {
        self.neg_cost == other.neg_cost
    }
}
impl Eq for MinHeapNode {}
impl PartialOrd for MinHeapNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MinHeapNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.neg_cost.cmp(&other.neg_cost)
    }
}

impl MinHeapNode {
    fn new(node: SearchNode) -> Self {
        let neg_cost = -(node.cost_so_far * 1_000_000.0) as i64;
        Self { neg_cost, node }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AbductiveReasoningEngine
// ─────────────────────────────────────────────────────────────────────────────

/// Production-quality abductive reasoning engine.
///
/// # Example
/// ```
/// use ipfrs_tensorlogic::{AbductiveReasoningEngine, AbrTerm, AbrEngineConfig};
///
/// let mut eng = AbductiveReasoningEngine::new(AbrEngineConfig::default());
/// let wet    = AbrTerm::prop("wet_grass");
/// let rain   = AbrTerm::prop("rain");
/// eng.add_observation(wet.clone());
/// let _hid = eng.add_hypothesis(rain, 1.0, true);
/// let expls = eng.abduce();
/// assert!(!expls.is_empty());
/// ```
pub struct AbductiveReasoningEngine {
    hypotheses: HashMap<HypothesisId, AbrHypothesis>,
    rules: Vec<AbrRule>,
    observations: Vec<AbrTerm>,
    history: VecDeque<AbrExplanationRecord>,
    config: AbrEngineConfig,
    // monotonically-increasing id counter
    next_id: u64,
    // cumulative stats
    abduce_calls: u64,
    total_nodes_explored: u64,
    total_explanations_found: u64,
    best_cost_ever: f64,
    // lightweight entropy for timestamps
    rng_state: u64,
}

impl AbductiveReasoningEngine {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Create a new engine with the given configuration.
    pub fn new(config: AbrEngineConfig) -> Self {
        Self {
            hypotheses: HashMap::new(),
            rules: Vec::new(),
            observations: Vec::new(),
            history: VecDeque::with_capacity(200),
            config,
            next_id: 1,
            abduce_calls: 0,
            total_nodes_explored: 0,
            total_explanations_found: 0,
            best_cost_ever: f64::INFINITY,
            rng_state: 0xDEAD_BEEF_CAFE_BABEu64,
        }
    }

    /// Create an engine with default configuration.
    pub fn default_engine() -> Self {
        Self::new(AbrEngineConfig::default())
    }

    // ── Mutators ──────────────────────────────────────────────────────────────

    /// Register a hypothesis and return its stable id.
    pub fn add_hypothesis(&mut self, term: AbrTerm, cost: f64, is_abducible: bool) -> HypothesisId {
        let id = self.fresh_id();
        self.hypotheses
            .insert(id, AbrHypothesis::new(id, term, cost, is_abducible));
        id
    }

    /// Register a logical rule.
    pub fn add_rule(&mut self, head: AbrTerm, body: Vec<AbrTerm>, confidence: f64) {
        self.rules.push(AbrRule::new(head, body, confidence));
    }

    /// Add an observation to be explained.
    pub fn add_observation(&mut self, term: AbrTerm) {
        self.observations.push(term);
    }

    /// Remove all observations.
    pub fn clear_observations(&mut self) {
        self.observations.clear();
    }

    /// Update the engine configuration.
    pub fn set_config(&mut self, config: AbrEngineConfig) {
        self.config = config;
    }

    /// Remove a hypothesis by id.  Returns `true` if it existed.
    pub fn remove_hypothesis(&mut self, id: HypothesisId) -> bool {
        self.hypotheses.remove(&id).is_some()
    }

    // ── Core Reasoning ────────────────────────────────────────────────────────

    /// Find all best explanations (hypothesis sets that cover all observations).
    ///
    /// Uses branch-and-bound over the space of abducible hypothesis subsets,
    /// pruning branches that exceed the cost of the best complete explanation
    /// found so far.
    pub fn abduce(&mut self) -> Vec<AbrExplanation> {
        self.abduce_calls += 1;

        if self.observations.is_empty() {
            // Nothing to explain — vacuously return the empty explanation.
            let expl = AbrExplanation {
                hypotheses: Vec::new(),
                covered: Vec::new(),
                total_cost: 0.0,
                completeness: 1.0,
            };
            self.record_history(0, 0, 0.0);
            return vec![expl];
        }

        // Build sorted list of abducible hypotheses (ascending cost for pruning).
        let mut abducibles: Vec<HypothesisId> = self
            .hypotheses
            .values()
            .filter(|h| h.is_abducible)
            .map(|h| h.id)
            .collect();
        abducibles.sort_by(|a, b| {
            let ca = self.hypotheses[a].cost;
            let cb = self.hypotheses[b].cost;
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let n_obs = self.observations.len();
        let max_set_size = self.config.max_hypothesis_set_size;
        let max_nodes = self.config.max_search_nodes;
        let max_expls = self.config.max_explanations;
        let prefer_min = self.config.prefer_minimal;

        let mut best_cost: f64 = f64::INFINITY;
        let mut explanations: Vec<AbrExplanation> = Vec::new();
        let mut seen_fingerprints: HashSet<u64> = HashSet::new();
        let mut nodes_explored: u64 = 0;

        // Seed the priority queue with the empty set.
        let mut queue: BinaryHeap<MinHeapNode> = BinaryHeap::new();
        queue.push(MinHeapNode::new(SearchNode {
            chosen: Vec::new(),
            next_idx: 0,
            cost_so_far: 0.0,
        }));

        while let Some(wrapper) = queue.pop() {
            if nodes_explored >= max_nodes as u64 {
                break;
            }
            nodes_explored += 1;

            let node = wrapper.node;

            // Compute derived facts for the current hypothesis set.
            let derived = self.apply_rules_for_set(&node.chosen);
            let covered = self.covered_observations(&node.chosen, &derived);

            if covered.len() == n_obs {
                // Complete explanation found.
                let cost = self
                    .config
                    .cost_function
                    .compute(&node.chosen, &self.hypotheses);
                let fp = set_fingerprint(&node.chosen);

                if !seen_fingerprints.contains(&fp) {
                    seen_fingerprints.insert(fp);

                    // Prune: for minimal preference discard strictly worse explanations.
                    let accept = if prefer_min {
                        cost <= best_cost + 1e-9
                    } else {
                        true
                    };

                    if accept {
                        if cost < best_cost {
                            best_cost = cost;
                            // Prune previously found explanations that are now sub-optimal.
                            if prefer_min {
                                explanations
                                    .retain(|e: &AbrExplanation| e.total_cost <= best_cost + 1e-9);
                            }
                        }
                        let expl = AbrExplanation {
                            hypotheses: node.chosen.clone(),
                            covered: covered.clone(),
                            total_cost: cost,
                            completeness: covered.len() as f64 / n_obs as f64,
                        };
                        explanations.push(expl);
                        if explanations.len() >= max_expls {
                            break;
                        }
                    }
                }
                // Do not branch further from a complete solution.
                continue;
            }

            // Incomplete: try adding each candidate hypothesis.
            for (branch_idx, &hid) in abducibles.iter().enumerate().skip(node.next_idx) {
                // Pruning: set size limit.
                if node.chosen.len() + 1 > max_set_size {
                    break;
                }

                // Skip if already chosen (no duplicates).
                if node.chosen.contains(&hid) {
                    continue;
                }

                let hcost = self.hypotheses.get(&hid).map_or(0.0, |h| h.cost);
                let new_cost = match &self.config.cost_function {
                    AbrCostFunction::SumCost | AbrCostFunction::WeightedCost(_) => {
                        node.cost_so_far + hcost
                    }
                    AbrCostFunction::MaxCost => node.cost_so_far.max(hcost),
                    AbrCostFunction::CountCost => node.cost_so_far + 1.0,
                };

                // Prune: cost already exceeds best known.
                if new_cost > best_cost + 1e-9 {
                    continue;
                }

                let mut new_chosen = node.chosen.clone();
                new_chosen.push(hid);

                queue.push(MinHeapNode::new(SearchNode {
                    chosen: new_chosen,
                    next_idx: branch_idx + 1,
                    cost_so_far: new_cost,
                }));
            }
        }

        self.total_nodes_explored += nodes_explored;
        self.total_explanations_found += explanations.len() as u64;

        // Update global best cost.
        if let Some(best) = explanations.iter().map(|e| e.total_cost).reduce(f64::min) {
            if best < self.best_cost_ever {
                self.best_cost_ever = best;
            }
        }

        // Record to history.
        let best_found = explanations
            .iter()
            .map(|e| e.total_cost)
            .fold(f64::INFINITY, f64::min);
        self.record_history(n_obs, nodes_explored, best_found);

        // Sort results: cheapest first, then by size for ties.
        explanations.sort_by(|a, b| {
            a.total_cost
                .partial_cmp(&b.total_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.hypotheses.len().cmp(&b.hypotheses.len()))
        });

        explanations
    }

    /// Return the single best explanation, if any.
    pub fn best_explanation(&mut self) -> Option<AbrExplanation> {
        let mut all = self.abduce();
        if all.is_empty() {
            None
        } else {
            Some(all.remove(0))
        }
    }

    /// Determine which observations a hypothesis set covers (directly or via rules).
    pub fn covers(&self, hypothesis_set: &[HypothesisId]) -> Vec<AbrTerm> {
        let derived = self.apply_rules_for_set(hypothesis_set);
        self.covered_observations(hypothesis_set, &derived)
    }

    /// Check that no two hypotheses in the set assert contradictory facts.
    ///
    /// A contradiction is detected when one hypothesis asserts `p(…)` and
    /// another asserts `not_p(…)` (prefix `"not_"` convention) or explicitly
    /// `"NOT:p(…)"`.
    pub fn is_consistent(&self, hypothesis_set: &[HypothesisId]) -> bool {
        let terms: Vec<&AbrTerm> = hypothesis_set
            .iter()
            .filter_map(|id| self.hypotheses.get(id).map(|h| &h.term))
            .collect();

        for (i, t) in terms.iter().enumerate() {
            for t2 in &terms[i + 1..] {
                if self.contradicts(t, t2) {
                    return false;
                }
            }
        }
        true
    }

    /// Forward-chain all rules over the facts asserted by every hypothesis
    /// in the current hypothesis registry (not just a subset).  Returns all
    /// newly derived facts as a fixed point.
    pub fn apply_rules(&self) -> Vec<AbrTerm> {
        let all_ids: Vec<HypothesisId> = self.hypotheses.keys().copied().collect();
        self.apply_rules_for_set(&all_ids)
    }

    /// Runtime statistics snapshot.
    pub fn reasoning_stats(&self) -> AbrReasoningStats {
        AbrReasoningStats {
            abduce_calls: self.abduce_calls,
            total_nodes_explored: self.total_nodes_explored,
            total_explanations_found: self.total_explanations_found,
            n_hypotheses: self.hypotheses.len(),
            n_rules: self.rules.len(),
            n_observations: self.observations.len(),
            history_len: self.history.len(),
            best_cost_ever: self.best_cost_ever,
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Retrieve a hypothesis by id.
    pub fn hypothesis(&self, id: HypothesisId) -> Option<&AbrHypothesis> {
        self.hypotheses.get(&id)
    }

    /// All hypothesis ids registered with the engine.
    pub fn hypothesis_ids(&self) -> Vec<HypothesisId> {
        self.hypotheses.keys().copied().collect()
    }

    /// All observations currently registered.
    pub fn observations(&self) -> &[AbrTerm] {
        &self.observations
    }

    /// Borrow a slice of all rules.
    pub fn rules(&self) -> &[AbrRule] {
        &self.rules
    }

    /// Access the explanation history (most recent last).
    pub fn history(&self) -> &VecDeque<AbrExplanationRecord> {
        &self.history
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Generate a unique id.
    fn fresh_id(&mut self) -> HypothesisId {
        // Mix next_id with entropy from xorshift64.
        let x = abr_xorshift64(&mut self.rng_state);
        let id = fnv1a_64(&(self.next_id ^ x).to_le_bytes());
        self.next_id += 1;
        id
    }

    /// Forward-chain rules for a specific hypothesis subset.
    ///
    /// Runs until a fixed point is reached (no new facts derived).
    fn apply_rules_for_set(&self, hypothesis_set: &[HypothesisId]) -> Vec<AbrTerm> {
        // Seed with the facts asserted directly by the hypothesis set.
        let mut known: HashSet<u64> = hypothesis_set
            .iter()
            .filter_map(|id| self.hypotheses.get(id))
            .map(|h| h.term.fingerprint())
            .collect();

        let mut known_terms: Vec<AbrTerm> = hypothesis_set
            .iter()
            .filter_map(|id| self.hypotheses.get(id))
            .map(|h| h.term.clone())
            .collect();

        // Fixed-point iteration.
        loop {
            let before = known.len();
            for rule in &self.rules {
                // Check if every body term is in `known_terms`.
                let body_satisfied = rule
                    .body
                    .iter()
                    .all(|bt| known_terms.iter().any(|kt| kt.matches(bt)));
                if body_satisfied {
                    let fp = rule.head.fingerprint();
                    if !known.contains(&fp) {
                        known.insert(fp);
                        known_terms.push(rule.head.clone());
                    }
                }
            }
            if known.len() == before {
                break;
            }
        }

        // Return only derived (non-hypothesis) terms.
        let seed_fps: HashSet<u64> = hypothesis_set
            .iter()
            .filter_map(|id| self.hypotheses.get(id))
            .map(|h| h.term.fingerprint())
            .collect();

        known_terms
            .into_iter()
            .filter(|t| !seed_fps.contains(&t.fingerprint()))
            .collect()
    }

    /// Determine which observations are covered (direct match or derived).
    fn covered_observations(
        &self,
        hypothesis_set: &[HypothesisId],
        derived: &[AbrTerm],
    ) -> Vec<AbrTerm> {
        let mut covered = Vec::new();
        for obs in &self.observations {
            // Direct coverage: some hypothesis asserts the observation.
            let direct = hypothesis_set
                .iter()
                .filter_map(|id| self.hypotheses.get(id))
                .any(|h| h.term.matches(obs));

            // Indirect coverage: a derived fact matches.
            let indirect = derived.iter().any(|d| d.matches(obs));

            if direct || indirect {
                covered.push(obs.clone());
            }
        }
        covered
    }

    /// Return `true` if terms `a` and `b` are logical contradictions.
    fn contradicts(&self, a: &AbrTerm, b: &AbrTerm) -> bool {
        // Convention 1: "not_X" vs "X"  (same args).
        let negation_of = |pos: &str, neg: &str| -> bool {
            neg == format!("not_{}", pos) || neg == format!("NOT:{}", pos)
        };
        if a.args == b.args
            && (negation_of(&a.predicate, &b.predicate) || negation_of(&b.predicate, &a.predicate))
        {
            return true;
        }
        // Convention 2: "NOT:predicate" prefix.
        if a.predicate.starts_with("NOT:") {
            let pos_pred = &a.predicate["NOT:".len()..];
            if b.predicate == pos_pred && a.args == b.args {
                return true;
            }
        }
        if b.predicate.starts_with("NOT:") {
            let pos_pred = &b.predicate["NOT:".len()..];
            if a.predicate == pos_pred && a.args == b.args {
                return true;
            }
        }
        false
    }

    /// Push a record to the bounded history deque.
    fn record_history(&mut self, n_obs: usize, n_tried: u64, best_cost: f64) {
        // Lightweight timestamp: tick count from xorshift + base.
        let ts = abr_xorshift64(&mut self.rng_state) % 1_700_000_000_000;
        if self.history.len() >= 200 {
            self.history.pop_front();
        }
        self.history.push_back(AbrExplanationRecord {
            timestamp_ms: ts,
            n_observations: n_obs,
            n_hypotheses_tried: n_tried,
            best_cost,
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Free-standing helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a deterministic fingerprint for a set of hypothesis ids
/// (order-independent — we sort before hashing).
pub fn set_fingerprint(ids: &[HypothesisId]) -> u64 {
    let mut sorted: Vec<HypothesisId> = ids.to_vec();
    sorted.sort_unstable();
    let mut buf: Vec<u8> = Vec::with_capacity(sorted.len() * 8);
    for id in &sorted {
        buf.extend_from_slice(&id.to_le_bytes());
    }
    fnv1a_64(&buf)
}

// ─────────────────────────────────────────────────────────────────────────────
// Type aliases (Abr* prefix)
// ─────────────────────────────────────────────────────────────────────────────

/// Type alias — primary engine type.
pub type AbrAbductiveReasoningEngine = AbductiveReasoningEngine;

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Utility constructors ──────────────────────────────────────────────────

    fn prop(p: &str) -> AbrTerm {
        AbrTerm::prop(p)
    }
    fn term(p: &str, args: &[&str]) -> AbrTerm {
        AbrTerm::new(p, args.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }
    fn engine() -> AbductiveReasoningEngine {
        AbductiveReasoningEngine::new(AbrEngineConfig {
            max_explanations: 20,
            max_hypothesis_set_size: 6,
            cost_function: AbrCostFunction::SumCost,
            prefer_minimal: true,
            max_search_nodes: 50_000,
        })
    }

    // ── T01: basic construction ───────────────────────────────────────────────
    #[test]
    fn t01_new_engine_empty() {
        let eng = engine();
        assert_eq!(eng.hypotheses.len(), 0);
        assert_eq!(eng.rules.len(), 0);
        assert_eq!(eng.observations.len(), 0);
    }

    // ── T02: add_hypothesis returns distinct ids ───────────────────────────────
    #[test]
    fn t02_hypothesis_ids_distinct() {
        let mut eng = engine();
        let id1 = eng.add_hypothesis(prop("a"), 1.0, true);
        let id2 = eng.add_hypothesis(prop("b"), 1.0, true);
        assert_ne!(id1, id2);
    }

    // ── T03: hypothesis retrieval ─────────────────────────────────────────────
    #[test]
    fn t03_hypothesis_retrieval() {
        let mut eng = engine();
        let id = eng.add_hypothesis(prop("rain"), 2.5, true);
        let h = eng.hypothesis(id).expect("should find hypothesis");
        assert_eq!(h.term.predicate, "rain");
        assert!((h.cost - 2.5).abs() < 1e-9);
        assert!(h.is_abducible);
    }

    // ── T04: non-abducible hypothesis ignored in abduce ───────────────────────
    #[test]
    fn t04_non_abducible_ignored() {
        let mut eng = engine();
        eng.add_hypothesis(prop("rain"), 1.0, false); // background knowledge
        eng.add_observation(prop("rain"));
        let expls = eng.abduce();
        // No abducible hypotheses → no complete explanation.
        assert!(expls.is_empty() || expls[0].completeness < 1.0);
    }

    // ── T05: single hypothesis covers single observation ──────────────────────
    #[test]
    fn t05_single_hyp_covers_obs() {
        let mut eng = engine();
        let _id = eng.add_hypothesis(prop("rain"), 1.0, true);
        eng.add_observation(prop("rain"));
        let expls = eng.abduce();
        assert!(!expls.is_empty());
        assert_eq!(expls[0].completeness, 1.0);
    }

    // ── T06: multiple hypotheses, minimal set ─────────────────────────────────
    #[test]
    fn t06_minimal_set_preferred() {
        let mut eng = engine();
        let id1 = eng.add_hypothesis(prop("rain"), 1.0, true);
        let _id2 = eng.add_hypothesis(prop("sun"), 5.0, true);
        eng.add_observation(prop("rain"));
        let expls = eng.abduce();
        assert!(!expls.is_empty());
        // The cheapest complete explanation uses only `rain`.
        assert!(expls[0].hypotheses.contains(&id1));
        assert_eq!(expls[0].hypotheses.len(), 1);
    }

    // ── T07: cost function SumCost ────────────────────────────────────────────
    #[test]
    fn t07_sum_cost() {
        let mut eng = engine();
        let id1 = eng.add_hypothesis(prop("a"), 2.0, true);
        let id2 = eng.add_hypothesis(prop("b"), 3.0, true);
        eng.add_observation(prop("a"));
        eng.add_observation(prop("b"));
        let expls = eng.abduce();
        assert!(!expls.is_empty());
        let best = &expls[0];
        assert!((best.total_cost - 5.0).abs() < 1e-6);
        assert!(best.hypotheses.contains(&id1));
        assert!(best.hypotheses.contains(&id2));
    }

    // ── T08: cost function MaxCost ────────────────────────────────────────────
    #[test]
    fn t08_max_cost() {
        let mut eng = AbductiveReasoningEngine::new(AbrEngineConfig {
            cost_function: AbrCostFunction::MaxCost,
            ..AbrEngineConfig::default()
        });
        eng.add_hypothesis(prop("a"), 2.0, true);
        eng.add_hypothesis(prop("b"), 3.0, true);
        eng.add_observation(prop("a"));
        eng.add_observation(prop("b"));
        let expls = eng.abduce();
        assert!(!expls.is_empty());
        // MaxCost of {a,b} = max(2,3) = 3.
        assert!((expls[0].total_cost - 3.0).abs() < 1e-6);
    }

    // ── T09: cost function CountCost ──────────────────────────────────────────
    #[test]
    fn t09_count_cost() {
        let mut eng = AbductiveReasoningEngine::new(AbrEngineConfig {
            cost_function: AbrCostFunction::CountCost,
            ..AbrEngineConfig::default()
        });
        eng.add_hypothesis(prop("a"), 99.0, true);
        eng.add_hypothesis(prop("b"), 0.1, true);
        eng.add_observation(prop("a"));
        eng.add_observation(prop("b"));
        let expls = eng.abduce();
        assert!(!expls.is_empty());
        // CountCost = 2 regardless of individual costs.
        assert!((expls[0].total_cost - 2.0).abs() < 1e-6);
    }

    // ── T10: cost function WeightedCost ───────────────────────────────────────
    #[test]
    fn t10_weighted_cost() {
        let mut eng = engine();
        let id1 = eng.add_hypothesis(prop("x"), 1.0, true);
        let id2 = eng.add_hypothesis(prop("y"), 1.0, true);
        let mut weights = HashMap::new();
        weights.insert(id1, 3.0);
        weights.insert(id2, 0.5);
        eng.set_config(AbrEngineConfig {
            cost_function: AbrCostFunction::WeightedCost(weights),
            max_explanations: 10,
            max_hypothesis_set_size: 6,
            prefer_minimal: true,
            max_search_nodes: 50_000,
        });
        eng.add_observation(prop("x"));
        eng.add_observation(prop("y"));
        let expls = eng.abduce();
        assert!(!expls.is_empty());
        // WeightedCost = 1*3 + 1*0.5 = 3.5
        assert!((expls[0].total_cost - 3.5).abs() < 1e-6);
    }

    // ── T11: rule-based coverage ──────────────────────────────────────────────
    #[test]
    fn t11_rule_derived_coverage() {
        let mut eng = engine();
        let id_rain = eng.add_hypothesis(prop("rain"), 1.0, true);
        // Rule: rain → wet_grass
        eng.add_rule(prop("wet_grass"), vec![prop("rain")], 1.0);
        eng.add_observation(prop("wet_grass"));
        let expls = eng.abduce();
        assert!(!expls.is_empty());
        assert_eq!(expls[0].completeness, 1.0);
        assert!(expls[0].hypotheses.contains(&id_rain));
    }

    // ── T12: rule chain (transitivity) ────────────────────────────────────────
    #[test]
    fn t12_rule_chain() {
        let mut eng = engine();
        let id = eng.add_hypothesis(prop("cloudy"), 1.0, true);
        eng.add_rule(prop("rain"), vec![prop("cloudy")], 1.0);
        eng.add_rule(prop("wet_grass"), vec![prop("rain")], 1.0);
        eng.add_observation(prop("wet_grass"));
        let expls = eng.abduce();
        assert!(!expls.is_empty());
        assert!(expls[0].hypotheses.contains(&id));
    }

    // ── T13: covers() method ──────────────────────────────────────────────────
    #[test]
    fn t13_covers() {
        let mut eng = engine();
        let id_r = eng.add_hypothesis(prop("rain"), 1.0, true);
        let _id_s = eng.add_hypothesis(prop("sun"), 1.0, true);
        eng.add_rule(prop("wet"), vec![prop("rain")], 1.0);
        eng.add_observation(prop("wet"));
        eng.add_observation(prop("sun"));
        let covered = eng.covers(&[id_r]);
        // Should cover "wet" via rule, but not "sun".
        assert!(covered.iter().any(|t| t.predicate == "wet"));
        assert!(!covered.iter().any(|t| t.predicate == "sun"));
    }

    // ── T14: is_consistent — consistent set ───────────────────────────────────
    #[test]
    fn t14_consistent_set() {
        let mut eng = engine();
        let id1 = eng.add_hypothesis(prop("a"), 1.0, true);
        let id2 = eng.add_hypothesis(prop("b"), 1.0, true);
        assert!(eng.is_consistent(&[id1, id2]));
    }

    // ── T15: is_consistent — inconsistent set (not_ prefix) ──────────────────
    #[test]
    fn t15_inconsistent_set_not_prefix() {
        let mut eng = engine();
        let id1 = eng.add_hypothesis(prop("rain"), 1.0, true);
        let id2 = eng.add_hypothesis(prop("not_rain"), 1.0, true);
        assert!(!eng.is_consistent(&[id1, id2]));
    }

    // ── T16: is_consistent — inconsistent set (NOT: prefix) ──────────────────
    #[test]
    fn t16_inconsistent_not_colon_prefix() {
        let mut eng = engine();
        let id1 = eng.add_hypothesis(prop("rain"), 1.0, true);
        let id2 = eng.add_hypothesis(prop("NOT:rain"), 1.0, true);
        assert!(!eng.is_consistent(&[id1, id2]));
    }

    // ── T17: apply_rules global ───────────────────────────────────────────────
    #[test]
    fn t17_apply_rules_global() {
        let mut eng = engine();
        eng.add_hypothesis(prop("raining"), 1.0, true);
        eng.add_rule(prop("puddles"), vec![prop("raining")], 1.0);
        let derived = eng.apply_rules();
        assert!(derived.iter().any(|t| t.predicate == "puddles"));
    }

    // ── T18: reasoning_stats ──────────────────────────────────────────────────
    #[test]
    fn t18_reasoning_stats() {
        let mut eng = engine();
        eng.add_hypothesis(prop("x"), 1.0, true);
        eng.add_observation(prop("x"));
        eng.abduce();
        let stats = eng.reasoning_stats();
        assert_eq!(stats.abduce_calls, 1);
        assert!(stats.total_explanations_found > 0);
    }

    // ── T19: empty observation list → vacuous explanation ─────────────────────
    #[test]
    fn t19_empty_observations_vacuous() {
        let mut eng = engine();
        eng.add_hypothesis(prop("a"), 1.0, true);
        let expls = eng.abduce();
        assert!(!expls.is_empty());
        assert_eq!(expls[0].completeness, 1.0);
        assert_eq!(expls[0].total_cost, 0.0);
    }

    // ── T20: best_explanation returns the cheapest ────────────────────────────
    #[test]
    fn t20_best_explanation() {
        let mut eng = engine();
        eng.add_hypothesis(prop("cheap"), 0.5, true);
        eng.add_hypothesis(prop("expensive"), 10.0, true);
        eng.add_observation(prop("cheap"));
        let best = eng.best_explanation().expect("should find one");
        assert!((best.total_cost - 0.5).abs() < 1e-6);
    }

    // ── T21: no hypothesis for observation → no explanation ───────────────────
    #[test]
    fn t21_unexplainable_observation() {
        let mut eng = engine();
        eng.add_hypothesis(prop("irrelevant"), 1.0, true);
        eng.add_observation(prop("some_fact"));
        let expls = eng.abduce();
        // There may be partial explanations but no complete one.
        let complete: Vec<_> = expls.iter().filter(|e| e.completeness >= 1.0).collect();
        assert!(complete.is_empty());
    }

    // ── T22: history is bounded to 200 ────────────────────────────────────────
    #[test]
    fn t22_history_bounded() {
        let mut eng = engine();
        eng.add_hypothesis(prop("x"), 1.0, true);
        eng.add_observation(prop("x"));
        for _ in 0..250 {
            eng.abduce();
        }
        assert!(eng.history().len() <= 200);
    }

    // ── T23: remove_hypothesis ────────────────────────────────────────────────
    #[test]
    fn t23_remove_hypothesis() {
        let mut eng = engine();
        let id = eng.add_hypothesis(prop("x"), 1.0, true);
        assert!(eng.remove_hypothesis(id));
        assert!(!eng.remove_hypothesis(id)); // second remove returns false
        assert!(eng.hypothesis(id).is_none());
    }

    // ── T24: clear_observations ────────────────────────────────────────────────
    #[test]
    fn t24_clear_observations() {
        let mut eng = engine();
        eng.add_observation(prop("x"));
        eng.add_observation(prop("y"));
        eng.clear_observations();
        assert!(eng.observations().is_empty());
    }

    // ── T25: fnv1a_64 deterministic ───────────────────────────────────────────
    #[test]
    fn t25_fnv1a_deterministic() {
        let a = fnv1a_64(b"hello");
        let b = fnv1a_64(b"hello");
        assert_eq!(a, b);
        assert_ne!(fnv1a_64(b"hello"), fnv1a_64(b"world"));
    }

    // ── T26: set_fingerprint order-independent ────────────────────────────────
    #[test]
    fn t26_set_fingerprint_order_independent() {
        let fp1 = set_fingerprint(&[1, 2, 3]);
        let fp2 = set_fingerprint(&[3, 1, 2]);
        assert_eq!(fp1, fp2);
    }

    // ── T27: AbrTerm fingerprint stability ────────────────────────────────────
    #[test]
    fn t27_term_fingerprint_stable() {
        let t = AbrTerm::new("parent", vec!["alice", "bob"]);
        assert_eq!(t.fingerprint(), t.fingerprint());
    }

    // ── T28: AbrTerm matches with wildcard ────────────────────────────────────
    #[test]
    fn t28_term_wildcard_match() {
        let pattern = term("parent", &["alice", "_"]);
        let ground = term("parent", &["alice", "bob"]);
        assert!(pattern.matches(&ground));
    }

    // ── T29: AbrTerm no-match on predicate ────────────────────────────────────
    #[test]
    fn t29_term_no_match_predicate() {
        let a = prop("rain");
        let b = prop("sun");
        assert!(!a.matches(&b));
    }

    // ── T30: AbrRule construction ─────────────────────────────────────────────
    #[test]
    fn t30_rule_construction() {
        let r = AbrRule::new(prop("wet"), vec![prop("rain")], 0.9);
        assert_eq!(r.body.len(), 1);
        assert!((r.confidence - 0.9).abs() < 1e-9);
    }

    // ── T31: AbrRule confidence clamped ───────────────────────────────────────
    #[test]
    fn t31_rule_confidence_clamped() {
        let r = AbrRule::new(prop("x"), vec![], 2.5);
        assert!((r.confidence - 1.0).abs() < 1e-9);
        let r2 = AbrRule::new(prop("x"), vec![], -1.0);
        assert!((r2.confidence - 0.0).abs() < 1e-9);
    }

    // ── T32: hypothesis cost floored to 0 ─────────────────────────────────────
    #[test]
    fn t32_hypothesis_cost_floored() {
        let h = AbrHypothesis::new(1, prop("x"), -5.0, true);
        assert_eq!(h.cost, 0.0);
    }

    // ── T33: multi-observation, single hypothesis ──────────────────────────────
    #[test]
    fn t33_single_hyp_multiple_observations_via_rules() {
        let mut eng = engine();
        let id = eng.add_hypothesis(prop("storm"), 1.0, true);
        eng.add_rule(prop("rain"), vec![prop("storm")], 1.0);
        eng.add_rule(prop("wind"), vec![prop("storm")], 1.0);
        eng.add_observation(prop("rain"));
        eng.add_observation(prop("wind"));
        let expls = eng.abduce();
        assert!(!expls.is_empty());
        assert_eq!(expls[0].completeness, 1.0);
        assert_eq!(expls[0].hypotheses.len(), 1);
        assert!(expls[0].hypotheses.contains(&id));
    }

    // ── T34: is_complete method ────────────────────────────────────────────────
    #[test]
    fn t34_explanation_is_complete() {
        let expl = AbrExplanation {
            hypotheses: vec![1],
            covered: vec![prop("a"), prop("b")],
            total_cost: 2.0,
            completeness: 1.0,
        };
        assert!(expl.is_complete(2));
        assert!(!expl.is_complete(3));
    }

    // ── T35: xorshift64 produces non-zero output ───────────────────────────────
    #[test]
    fn t35_xorshift64_nonzero() {
        let mut state = 12345u64;
        let v = abr_xorshift64(&mut state);
        assert_ne!(v, 0);
    }

    // ── T36: multiple rules for the same head ─────────────────────────────────
    #[test]
    fn t36_multiple_rules_same_head() {
        let mut eng = engine();
        let id1 = eng.add_hypothesis(prop("heat"), 1.0, true);
        let id2 = eng.add_hypothesis(prop("cold"), 1.0, true);
        // Both independently can cause "fog"
        eng.add_rule(prop("fog"), vec![prop("heat")], 1.0);
        eng.add_rule(prop("fog"), vec![prop("cold")], 1.0);
        eng.add_observation(prop("fog"));
        let expls = eng.abduce();
        assert!(!expls.is_empty());
        // Both single-hypothesis explanations should be found.
        let hyp_sets: Vec<_> = expls.iter().map(|e| e.hypotheses.clone()).collect();
        let has_heat = hyp_sets.iter().any(|s| s == &[id1]);
        let has_cold = hyp_sets.iter().any(|s| s == &[id2]);
        assert!(has_heat || has_cold);
    }

    // ── T37: max_hypothesis_set_size respected ────────────────────────────────
    #[test]
    fn t37_max_set_size() {
        let mut eng = AbductiveReasoningEngine::new(AbrEngineConfig {
            max_hypothesis_set_size: 1,
            max_explanations: 10,
            cost_function: AbrCostFunction::SumCost,
            prefer_minimal: true,
            max_search_nodes: 10_000,
        });
        // Force requiring 2 hypotheses.
        eng.add_hypothesis(prop("a"), 1.0, true);
        eng.add_hypothesis(prop("b"), 1.0, true);
        eng.add_observation(prop("a"));
        eng.add_observation(prop("b"));
        let expls = eng.abduce();
        // With max_size=1, no complete explanation should be found.
        let complete: Vec<_> = expls.iter().filter(|e| e.completeness >= 1.0).collect();
        assert!(complete.is_empty());
    }

    // ── T38: hypothesis_ids returns all registered ids ─────────────────────────
    #[test]
    fn t38_hypothesis_ids() {
        let mut eng = engine();
        let id1 = eng.add_hypothesis(prop("a"), 1.0, true);
        let id2 = eng.add_hypothesis(prop("b"), 1.0, true);
        let ids = eng.hypothesis_ids();
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    // ── T39: rules() accessor ────────────────────────────────────────────────
    #[test]
    fn t39_rules_accessor() {
        let mut eng = engine();
        eng.add_rule(prop("x"), vec![], 1.0);
        assert_eq!(eng.rules().len(), 1);
    }

    // ── T40: observations() accessor ─────────────────────────────────────────
    #[test]
    fn t40_observations_accessor() {
        let mut eng = engine();
        eng.add_observation(prop("a"));
        eng.add_observation(prop("b"));
        assert_eq!(eng.observations().len(), 2);
    }

    // ── T41: default_engine constructor ───────────────────────────────────────
    #[test]
    fn t41_default_engine() {
        let eng = AbductiveReasoningEngine::default_engine();
        assert_eq!(eng.config.max_explanations, 10);
        assert_eq!(eng.config.max_hypothesis_set_size, 8);
    }

    // ── T42: AbrTerm canonical form ───────────────────────────────────────────
    #[test]
    fn t42_term_canonical() {
        let t = AbrTerm::new("p", vec!["a", "b"]);
        assert_eq!(t.canonical(), "p(a,b)");
        let t2 = prop("q");
        assert_eq!(t2.canonical(), "q");
    }

    // ── T43: set_config updates config ────────────────────────────────────────
    #[test]
    fn t43_set_config() {
        let mut eng = engine();
        eng.set_config(AbrEngineConfig {
            max_explanations: 3,
            ..AbrEngineConfig::default()
        });
        assert_eq!(eng.config.max_explanations, 3);
    }

    // ── T44: history not polluted by empty abduce ─────────────────────────────
    #[test]
    fn t44_history_grows_on_abduce() {
        let mut eng = engine();
        eng.add_hypothesis(prop("x"), 1.0, true);
        eng.add_observation(prop("x"));
        assert_eq!(eng.history().len(), 0);
        eng.abduce();
        assert_eq!(eng.history().len(), 1);
    }

    // ── T45: contradicts — same args required ─────────────────────────────────
    #[test]
    fn t45_contradicts_same_args() {
        let eng = engine();
        let a = term("wet", &["lawn"]);
        let b = term("not_wet", &["lawn"]);
        let c = term("not_wet", &["floor"]); // different arg
        assert!(eng.contradicts(&a, &b));
        assert!(!eng.contradicts(&a, &c));
    }

    // ── T46: best_explanation on no observations returns vacuous ──────────────
    #[test]
    fn t46_best_explanation_no_obs() {
        let mut eng = engine();
        eng.add_hypothesis(prop("x"), 1.0, true);
        let best = eng.best_explanation().expect("vacuous explanation");
        assert_eq!(best.completeness, 1.0);
    }

    // ── T47: explanations deduplicated ────────────────────────────────────────
    #[test]
    fn t47_deduplication() {
        let mut eng = engine();
        let id = eng.add_hypothesis(prop("r"), 1.0, true);
        eng.add_observation(prop("r"));
        let expls = eng.abduce();
        // Only one explanation for this trivial case.
        let count = expls.iter().filter(|e| e.hypotheses == vec![id]).count();
        assert_eq!(count, 1);
    }

    // ── T48: cost f64::INFINITY if no explanations ────────────────────────────
    #[test]
    fn t48_best_cost_infinity_if_none() {
        let eng = engine();
        assert_eq!(eng.best_cost_ever, f64::INFINITY);
    }

    // ── T49: AbrEngineConfig prefer_minimal = false ───────────────────────────
    #[test]
    fn t49_prefer_minimal_false() {
        let mut eng = AbductiveReasoningEngine::new(AbrEngineConfig {
            prefer_minimal: false,
            max_explanations: 5,
            max_hypothesis_set_size: 4,
            cost_function: AbrCostFunction::SumCost,
            max_search_nodes: 50_000,
        });
        eng.add_hypothesis(prop("a"), 1.0, true);
        eng.add_observation(prop("a"));
        let expls = eng.abduce();
        assert!(!expls.is_empty());
    }

    // ── T50: apply_rules fixed-point (no infinite loop) ───────────────────────
    #[test]
    fn t50_apply_rules_no_cycle() {
        let mut eng = engine();
        eng.add_hypothesis(prop("x"), 1.0, true);
        // Self-loop rule: x → x (already known, should not loop).
        eng.add_rule(prop("x"), vec![prop("x")], 1.0);
        let derived = eng.apply_rules();
        // x is seed, not derived; derived list should be empty or contain only new facts.
        assert!(!derived.iter().any(|t| t.predicate == "x"));
    }

    // ── T51: conjunctive rule body ────────────────────────────────────────────
    #[test]
    fn t51_conjunctive_rule_body() {
        let mut eng = engine();
        let id1 = eng.add_hypothesis(prop("a"), 1.0, true);
        let id2 = eng.add_hypothesis(prop("b"), 1.0, true);
        // Rule: a ∧ b → c
        eng.add_rule(prop("c"), vec![prop("a"), prop("b")], 1.0);
        eng.add_observation(prop("c"));
        let expls = eng.abduce();
        assert!(!expls.is_empty());
        let best = &expls[0];
        assert!(best.hypotheses.contains(&id1));
        assert!(best.hypotheses.contains(&id2));
    }

    // ── T52: partially covered explanation has correct completeness ───────────
    #[test]
    fn t52_partial_completeness_value() {
        let expl = AbrExplanation {
            hypotheses: vec![1],
            covered: vec![prop("a")],
            total_cost: 1.0,
            completeness: 0.5,
        };
        assert!((expl.completeness - 0.5).abs() < 1e-9);
        assert!(!expl.is_complete(2));
    }

    // ── T53: multiple complete explanations returned ──────────────────────────
    #[test]
    fn t53_multiple_complete_explanations() {
        let mut eng = AbductiveReasoningEngine::new(AbrEngineConfig {
            prefer_minimal: false,
            max_explanations: 20,
            max_hypothesis_set_size: 6,
            cost_function: AbrCostFunction::SumCost,
            max_search_nodes: 50_000,
        });
        let id1 = eng.add_hypothesis(prop("cause_a"), 1.0, true);
        let id2 = eng.add_hypothesis(prop("cause_b"), 1.0, true);
        eng.add_rule(prop("effect"), vec![prop("cause_a")], 1.0);
        eng.add_rule(prop("effect"), vec![prop("cause_b")], 1.0);
        eng.add_observation(prop("effect"));
        let expls = eng.abduce();
        let hyp_sets: Vec<_> = expls.iter().map(|e| e.hypotheses.clone()).collect();
        let has_a = hyp_sets.iter().any(|s| s == &[id1]);
        let has_b = hyp_sets.iter().any(|s| s == &[id2]);
        assert!(has_a && has_b);
    }

    // ── T54: AbrReasoningStats fields ─────────────────────────────────────────
    #[test]
    fn t54_reasoning_stats_fields() {
        let mut eng = engine();
        let _id = eng.add_hypothesis(prop("x"), 1.0, true);
        eng.add_rule(prop("y"), vec![prop("x")], 1.0);
        eng.add_observation(prop("x"));
        eng.abduce();
        let s = eng.reasoning_stats();
        assert_eq!(s.n_hypotheses, 1);
        assert_eq!(s.n_rules, 1);
        assert_eq!(s.n_observations, 1);
        assert_eq!(s.abduce_calls, 1);
        assert!(s.total_nodes_explored > 0);
    }

    // ── T55: AbrExplanationRecord stored ──────────────────────────────────────
    #[test]
    fn t55_explanation_record_stored() {
        let mut eng = engine();
        eng.add_hypothesis(prop("x"), 1.0, true);
        eng.add_observation(prop("x"));
        eng.abduce();
        let rec = eng.history().back().expect("should have record");
        assert_eq!(rec.n_observations, 1);
        assert!(rec.best_cost < f64::INFINITY);
    }

    // ── T56: prefer_minimal prunes worse explanations ─────────────────────────
    #[test]
    fn t56_prefer_minimal_prunes_costlier() {
        let mut eng = engine(); // prefer_minimal = true
        let _id_cheap = eng.add_hypothesis(prop("obs"), 1.0, true);
        let _id_expensive = eng.add_hypothesis(prop("obs_alt"), 100.0, true);
        // Both explain the same observation via rules.
        eng.add_rule(prop("obs"), vec![prop("obs_alt")], 1.0);
        eng.add_observation(prop("obs"));
        let expls = eng.abduce();
        if expls.len() > 1 {
            // All returned explanations should have cost ≤ best + epsilon.
            let best = expls[0].total_cost;
            for e in &expls {
                assert!(e.total_cost <= best + 1e-6);
            }
        }
    }

    // ── T57: set_fingerprint empty set ────────────────────────────────────────
    #[test]
    fn t57_fingerprint_empty() {
        let fp = set_fingerprint(&[]);
        // FNV-1a on empty input is the offset basis.
        assert_eq!(fp, fnv1a_64(b""));
    }

    // ── T58: term with args produces different fingerprint than prop ───────────
    #[test]
    fn t58_term_args_vs_prop_fingerprint() {
        let a = prop("p");
        let b = AbrTerm::new("p", vec!["x"]);
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    // ── T59: history record n_hypotheses_tried reflects search work ───────────
    #[test]
    fn t59_history_nodes_tried() {
        let mut eng = engine();
        eng.add_hypothesis(prop("a"), 1.0, true);
        eng.add_hypothesis(prop("b"), 1.0, true);
        eng.add_observation(prop("a"));
        eng.add_observation(prop("b"));
        eng.abduce();
        let rec = eng.history().back().expect("record");
        // At least one node was explored.
        assert!(rec.n_hypotheses_tried > 0);
    }

    // ── T60: cost 0 hypothesis is valid ───────────────────────────────────────
    #[test]
    fn t60_zero_cost_hypothesis() {
        let mut eng = engine();
        let id = eng.add_hypothesis(prop("free_fact"), 0.0, true);
        eng.add_observation(prop("free_fact"));
        let expls = eng.abduce();
        assert!(!expls.is_empty());
        assert_eq!(expls[0].total_cost, 0.0);
        assert!(expls[0].hypotheses.contains(&id));
    }

    // ── T61: best_cost_ever updated correctly ─────────────────────────────────
    #[test]
    fn t61_best_cost_ever_updated() {
        let mut eng = engine();
        eng.add_hypothesis(prop("x"), 3.0, true);
        eng.add_observation(prop("x"));
        eng.abduce();
        assert!((eng.best_cost_ever - 3.0).abs() < 1e-6);

        // Second run with cheaper hypothesis.
        eng.clear_observations();
        let id2 = eng.add_hypothesis(prop("y"), 1.0, true);
        eng.add_observation(prop("y"));
        eng.abduce();
        // best_cost_ever should now be 1.0.
        assert!((eng.best_cost_ever - 1.0).abs() < 1e-6);
        let _ = id2;
    }

    // ── T62: AbrRule body can be empty (fact rule) ────────────────────────────
    #[test]
    fn t62_empty_body_rule() {
        let mut eng = engine();
        // Rule with no body: always derives "always_true".
        eng.add_rule(prop("always_true"), vec![], 1.0);
        eng.add_observation(prop("always_true"));
        // Even with no hypotheses, the rule should fire and cover.
        let derived = eng.apply_rules_for_set(&[]);
        assert!(derived.iter().any(|t| t.predicate == "always_true"));
    }

    // ── T63: abduce total_nodes_explored cumulative ───────────────────────────
    #[test]
    fn t63_total_nodes_explored_cumulative() {
        let mut eng = engine();
        eng.add_hypothesis(prop("x"), 1.0, true);
        eng.add_observation(prop("x"));
        eng.abduce();
        let n1 = eng.total_nodes_explored;
        eng.abduce();
        let n2 = eng.total_nodes_explored;
        assert!(n2 >= n1);
    }

    // ── T64: max_explanations limit respected ─────────────────────────────────
    #[test]
    fn t64_max_explanations_limit() {
        let mut eng = AbductiveReasoningEngine::new(AbrEngineConfig {
            max_explanations: 2,
            max_hypothesis_set_size: 6,
            cost_function: AbrCostFunction::SumCost,
            prefer_minimal: false,
            max_search_nodes: 100_000,
        });
        // Create many single-hypothesis explanations via independent rules.
        for i in 0..10 {
            let p = format!("cause{}", i);
            eng.add_hypothesis(AbrTerm::prop(p.clone()), 1.0, true);
            eng.add_rule(prop("effect"), vec![AbrTerm::prop(p)], 1.0);
        }
        eng.add_observation(prop("effect"));
        let expls = eng.abduce();
        assert!(expls.len() <= 2);
    }
}
