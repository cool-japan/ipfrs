//! Probabilistic Logic Network (PLN) — uncertain reasoning combining
//! probability theory with logic.
//!
//! PLN represents uncertain beliefs as *indefinite truth values* — (strength,
//! confidence) pairs — and propagates uncertainty through a rich set of
//! inference rules (Deduction, Induction, Abduction, Revision, Conjunction,
//! Disjunction, Negation, Modus Ponens).  The atom/link hypergraph follows the
//! OpenCog AtomSpace model.
//!
//! # Quick Example
//!
//! ```
//! use ipfrs_tensorlogic::{
//!     ProbabilisticLogicNetwork, PlnAtom, PlnLink, TruthValue,
//!     AtomType, LinkType, PlnInferenceRule, PlnConfig,
//! };
//!
//! let mut pln = ProbabilisticLogicNetwork::new(PlnConfig::default());
//!
//! // Add concept nodes
//! let tv_high = TruthValue::new(0.9, 0.8);
//! let cat = PlnAtom::new("cat", "Cat", tv_high, AtomType::Node("ConceptNode".into()));
//! pln.add_atom(cat).expect("example: should succeed in docs");
//!
//! let tv_med = TruthValue::new(0.7, 0.6);
//! let animal = PlnAtom::new("animal", "Animal", tv_med, AtomType::Node("ConceptNode".into()));
//! pln.add_atom(animal).expect("example: should succeed in docs");
//!
//! // Add inheritance link: Cat → Animal
//! let link = PlnLink::new(
//!     "cat_animal",
//!     LinkType::Inheritance,
//!     vec!["cat".into(), "animal".into()],
//!     TruthValue::new(0.95, 0.9),
//! );
//! pln.add_link(link).expect("example: should succeed in docs");
//!
//! let tv = pln.get_tv("cat_animal").expect("example: should succeed in docs");
//! assert!(tv.strength > 0.9);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── Truth Value ──────────────────────────────────────────────────────────────

/// PLN indefinite truth value: (strength, confidence).
///
/// * `strength` — probability estimate ∈ \[0, 1\]
/// * `confidence` — certainty in the strength estimate ∈ \[0, 1\]
///   (0 = no information, 1 = complete certainty)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TruthValue {
    /// Probability estimate.
    pub strength: f64,
    /// Confidence in the strength estimate.
    pub confidence: f64,
}

impl TruthValue {
    /// Create a new truth value, clamping both components to \[0, 1\].
    #[inline]
    pub fn new(strength: f64, confidence: f64) -> Self {
        Self {
            strength: strength.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Return the *unknown* truth value (strength = 0.5, confidence = 0).
    #[inline]
    pub fn unknown() -> Self {
        Self {
            strength: 0.5,
            confidence: 0.0,
        }
    }

    /// Return the *true* truth value (strength = 1, confidence = 1).
    #[inline]
    pub fn certain_true() -> Self {
        Self {
            strength: 1.0,
            confidence: 1.0,
        }
    }

    /// Return the *false* truth value (strength = 0, confidence = 1).
    #[inline]
    pub fn certain_false() -> Self {
        Self {
            strength: 0.0,
            confidence: 1.0,
        }
    }

    /// Effective sample count `n = c / (1 − c)`.
    /// Returns `f64::INFINITY` when `confidence == 1`.
    #[inline]
    pub fn count(self) -> f64 {
        let denom = 1.0 - self.confidence;
        if denom <= 0.0 {
            f64::INFINITY
        } else {
            self.confidence / denom
        }
    }

    /// PLN negation: `NOT A`.
    #[inline]
    pub fn negate(self) -> Self {
        Self::new(1.0 - self.strength, self.confidence)
    }

    /// PLN conjunction: `A AND B`.
    #[inline]
    pub fn conjunction(self, other: Self) -> Self {
        Self::new(
            self.strength * other.strength,
            self.confidence.min(other.confidence),
        )
    }

    /// PLN disjunction: `A OR B`.
    #[inline]
    pub fn disjunction(self, other: Self) -> Self {
        Self::new(
            self.strength + other.strength - self.strength * other.strength,
            self.confidence.min(other.confidence),
        )
    }

    /// PLN revision: merge two independent estimates of the same proposition.
    ///
    /// Uses weighted average by effective sample count:
    /// ```text
    /// s_rev = (s1 * n1 + s2 * n2) / (n1 + n2)
    /// c_rev = (n1 + n2) / (n1 + n2 + 1)
    /// ```
    pub fn revise(self, other: Self) -> Self {
        let n1 = self.count();
        let n2 = other.count();

        // Handle infinite counts (certain beliefs)
        if n1.is_infinite() && n2.is_infinite() {
            // Both certain — average strengths with full confidence
            return Self::new((self.strength + other.strength) / 2.0, 1.0);
        }
        if n1.is_infinite() {
            return self;
        }
        if n2.is_infinite() {
            return other;
        }

        let n_total = n1 + n2;
        if n_total <= 0.0 {
            return Self::unknown();
        }
        let s_rev = (self.strength * n1 + other.strength * n2) / n_total;
        let c_rev = n_total / (n_total + 1.0);
        Self::new(s_rev, c_rev)
    }

    /// PLN deduction: given A→B and B→C, infer A→C.
    ///
    /// ```text
    /// s_AC = s_AB * s_BC + (1 − s_AB) * (s_C − s_B * s_BC) / (1 − s_B)
    /// c_AC = c_AB * c_BC * min(1, c_AB * c_BC)
    /// ```
    pub fn deduction(ab: Self, bc: Self, b: Self, c: Self) -> Self {
        let s_ab = ab.strength;
        let s_bc = bc.strength;
        let s_b = b.strength;
        let s_c = c.strength;

        let s_ac = if (1.0 - s_b).abs() < 1e-12 {
            // s_B ≈ 1 — degenerate; fall back to s_BC
            s_bc
        } else {
            let correction = (s_c - s_b * s_bc) / (1.0 - s_b);
            (s_ab * s_bc + (1.0 - s_ab) * correction).clamp(0.0, 1.0)
        };

        let product = ab.confidence * bc.confidence;
        let c_ac = (product * product.min(1.0)).clamp(0.0, 1.0);

        Self::new(s_ac, c_ac)
    }

    /// PLN induction: given A→B and A→C, infer B→C.
    ///
    /// Simplified formula:
    /// ```text
    /// s_BC = s_AC / max(s_AB, 0.01), clamped [0, 1]
    /// c_BC = c_AB * c_AC * min(1, c_AB * c_AC)
    /// ```
    pub fn induction(ab: Self, ac: Self) -> Self {
        let s_bc = (ac.strength / ab.strength.max(0.01)).clamp(0.0, 1.0);
        let product = ab.confidence * ac.confidence;
        let c_bc = (product * product.min(1.0)).clamp(0.0, 1.0);
        Self::new(s_bc, c_bc)
    }

    /// PLN abduction: given B→A and C→A, infer B→C.
    ///
    /// Dual of induction (swap roles of B and C):
    /// ```text
    /// s_BC = s_BA * s_CA / max(s_A, 0.01), clamped [0, 1]  (simplified)
    /// c_BC = c_BA * c_CA * min(1, c_BA * c_CA)
    /// ```
    pub fn abduction(ba: Self, ca: Self) -> Self {
        let s_bc = (ba.strength * ca.strength).clamp(0.0, 1.0);
        let product = ba.confidence * ca.confidence;
        let c_bc = (product * product.min(1.0)).clamp(0.0, 1.0);
        Self::new(s_bc, c_bc)
    }

    /// PLN Modus Ponens: given A (premise) and A→B (implication), infer B.
    ///
    /// ```text
    /// s_B  = s_A * s_AB + (1 − s_A) * (1 − s_AB) * 0.5   (mixture)
    /// c_B  = c_A * c_AB
    /// ```
    pub fn modus_ponens(a: Self, ab: Self) -> Self {
        let s_b = (a.strength * ab.strength + (1.0 - a.strength) * (1.0 - ab.strength) * 0.5)
            .clamp(0.0, 1.0);
        let c_b = (a.confidence * ab.confidence).clamp(0.0, 1.0);
        Self::new(s_b, c_b)
    }
}

impl fmt::Display for TruthValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TV(s={:.4}, c={:.4})", self.strength, self.confidence)
    }
}

impl Default for TruthValue {
    fn default() -> Self {
        Self::unknown()
    }
}

// ── Atom types ───────────────────────────────────────────────────────────────

/// Type of a PLN link.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LinkType {
    /// Inheritance (subtype/instance-of relationship).
    Inheritance,
    /// Similarity (symmetric relation).
    Similarity,
    /// Logical implication.
    Implication,
    /// Logical equivalence.
    Equivalence,
    /// Logical AND.
    AND,
    /// Logical OR.
    OR,
    /// Logical NOT.
    NOT,
    /// Predicate evaluation (wraps a predicate application).
    Evaluation,
}

/// Type of a PLN atom.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtomType {
    /// A node atom (e.g., `"ConceptNode"`, `"PredicateNode"`).
    Node(String),
    /// A link atom of the given link type.
    Link(LinkType),
}

// ── Core structs ─────────────────────────────────────────────────────────────

/// A PLN atom (node or link wrapper).
#[derive(Debug, Clone)]
pub struct PlnAtom {
    /// Unique identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Associated truth value.
    pub tv: TruthValue,
    /// Atom type.
    pub atom_type: AtomType,
}

impl PlnAtom {
    /// Create a new atom.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        tv: TruthValue,
        atom_type: AtomType,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            tv,
            atom_type,
        }
    }
}

/// A PLN link (hyperedge connecting one or more source atoms).
#[derive(Debug, Clone)]
pub struct PlnLink {
    /// Unique identifier.
    pub id: String,
    /// Link type.
    pub link_type: LinkType,
    /// Ordered list of source atom IDs.
    pub source_ids: Vec<String>,
    /// Associated truth value.
    pub tv: TruthValue,
}

impl PlnLink {
    /// Create a new link.
    pub fn new(
        id: impl Into<String>,
        link_type: LinkType,
        source_ids: Vec<String>,
        tv: TruthValue,
    ) -> Self {
        Self {
            id: id.into(),
            link_type,
            source_ids,
            tv,
        }
    }
}

// ── Inference rules ──────────────────────────────────────────────────────────

/// PLN inference rules.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PlnInferenceRule {
    /// A→B, B→C ⊢ A→C
    Deduction,
    /// A→B, A→C ⊢ B→C
    Induction,
    /// B→A, C→A ⊢ B→C
    Abduction,
    /// Merge two independent estimates of the same proposition.
    Revision,
    /// A AND B
    Conjunction,
    /// A OR B
    Disjunction,
    /// NOT A
    Negation,
    /// A, A→B ⊢ B
    ModusPonens,
}

impl fmt::Display for PlnInferenceRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Deduction => "Deduction",
            Self::Induction => "Induction",
            Self::Abduction => "Abduction",
            Self::Revision => "Revision",
            Self::Conjunction => "Conjunction",
            Self::Disjunction => "Disjunction",
            Self::Negation => "Negation",
            Self::ModusPonens => "ModusPonens",
        };
        f.write_str(s)
    }
}

/// The result of a single PLN inference step.
#[derive(Debug, Clone)]
pub struct PlnInferenceResult {
    /// ID of the concluded atom (may be new or existing).
    pub conclusion_id: String,
    /// The derived truth value for the conclusion.
    pub tv: TruthValue,
    /// Which rule was used.
    pub rule_used: PlnInferenceRule,
    /// IDs of the premises consumed.
    pub premise_ids: Vec<String>,
    /// Change in confidence relative to any prior belief (0 if new).
    pub confidence_delta: f64,
}

// ── Configuration & statistics ────────────────────────────────────────────────

/// Configuration for a `ProbabilisticLogicNetwork`.
#[derive(Debug, Clone)]
pub struct PlnConfig {
    /// Maximum number of atoms (nodes + links) allowed.
    pub max_atoms: usize,
    /// Maximum inference chain depth for `find_chains`.
    pub inference_depth: u8,
    /// Minimum confidence required for an inference to be accepted.
    pub min_confidence_threshold: f64,
    /// If `true`, applying an existing inference revises (merges) the prior TV.
    /// If `false`, the new TV silently replaces the old one.
    pub enable_revision: bool,
}

impl Default for PlnConfig {
    fn default() -> Self {
        Self {
            max_atoms: 100_000,
            inference_depth: 6,
            min_confidence_threshold: 0.01,
            enable_revision: true,
        }
    }
}

/// Runtime statistics for a `ProbabilisticLogicNetwork`.
#[derive(Debug, Clone, Default)]
pub struct PlnStats {
    /// Number of atoms currently stored.
    pub atom_count: usize,
    /// Number of links currently stored.
    pub link_count: usize,
    /// Total inference operations performed.
    pub inferences_performed: u64,
    /// Average confidence across all atoms.
    pub avg_confidence: f64,
}

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors produced by the PLN engine.
#[derive(Debug, Clone, PartialEq)]
pub enum PlnError {
    /// An atom with the given ID was not found.
    AtomNotFound(String),
    /// The derived confidence is below the configured threshold.
    InsufficientConfidence(f64),
    /// Cycle detected while tracing inference chains.
    CyclicInference(Vec<String>),
    /// The rule cannot be applied with the supplied premises.
    InvalidRule(String),
    /// The atom store has reached `max_atoms`.
    MaxAtomsExceeded,
}

impl fmt::Display for PlnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AtomNotFound(id) => write!(f, "atom not found: {id}"),
            Self::InsufficientConfidence(c) => {
                write!(f, "confidence {c:.4} below threshold")
            }
            Self::CyclicInference(path) => write!(f, "cyclic inference: {path:?}"),
            Self::InvalidRule(msg) => write!(f, "invalid rule: {msg}"),
            Self::MaxAtomsExceeded => write!(f, "max atom limit exceeded"),
        }
    }
}

impl std::error::Error for PlnError {}

// ── Internal adjacency helpers ────────────────────────────────────────────────

/// Lightweight adjacency record stored per source atom.
#[derive(Debug, Clone)]
struct OutEdge {
    /// The link atom ID.
    link_id: String,
    /// The other end of the edge (second source for binary links, etc.).
    targets: Vec<String>,
}

// ── Main engine ───────────────────────────────────────────────────────────────

/// Probabilistic Logic Network engine.
///
/// Maintains a hypergraph of atoms and links, executes PLN inference rules,
/// and provides chain-search utilities.
pub struct ProbabilisticLogicNetwork {
    /// All atoms, keyed by ID.
    atoms: HashMap<String, PlnAtom>,
    /// All links, keyed by ID.
    links: HashMap<String, PlnLink>,
    /// Adjacency list: source_id → list of outgoing edges.
    adjacency: HashMap<String, Vec<OutEdge>>,
    /// Engine configuration.
    config: PlnConfig,
    /// Running statistics.
    inferences_performed: u64,
}

impl ProbabilisticLogicNetwork {
    /// Create a new PLN engine with the given configuration.
    pub fn new(config: PlnConfig) -> Self {
        Self {
            atoms: HashMap::new(),
            links: HashMap::new(),
            adjacency: HashMap::new(),
            config,
            inferences_performed: 0,
        }
    }

    // ── Mutation ──────────────────────────────────────────────────────────────

    /// Add an atom to the network.
    ///
    /// Returns `Err(PlnError::MaxAtomsExceeded)` when the atom limit is reached.
    /// Silently replaces any existing atom with the same ID.
    pub fn add_atom(&mut self, atom: PlnAtom) -> Result<(), PlnError> {
        let total = self.atoms.len() + self.links.len();
        if !self.atoms.contains_key(&atom.id) && total >= self.config.max_atoms {
            return Err(PlnError::MaxAtomsExceeded);
        }
        self.atoms.insert(atom.id.clone(), atom);
        Ok(())
    }

    /// Add a link to the network.
    ///
    /// The link is also registered as an atom so its TV can be queried via
    /// `get_tv`.  All `source_ids` must already exist as atoms or links.
    pub fn add_link(&mut self, link: PlnLink) -> Result<(), PlnError> {
        // Validate sources
        for src in &link.source_ids {
            if !self.atoms.contains_key(src) && !self.links.contains_key(src) {
                return Err(PlnError::AtomNotFound(src.clone()));
            }
        }

        let total = self.atoms.len() + self.links.len();
        let is_new = !self.links.contains_key(&link.id);
        if is_new && total >= self.config.max_atoms {
            return Err(PlnError::MaxAtomsExceeded);
        }

        // Build adjacency for binary directed links (first source → second source)
        if link.source_ids.len() >= 2 {
            let src = link.source_ids[0].clone();
            let tgts: Vec<String> = link.source_ids[1..].to_vec();
            let edge = OutEdge {
                link_id: link.id.clone(),
                targets: tgts,
            };
            self.adjacency.entry(src).or_default().push(edge);
        }

        self.links.insert(link.id.clone(), link);
        Ok(())
    }

    // ── Query ─────────────────────────────────────────────────────────────────

    /// Return the truth value of an atom or link by ID.
    pub fn get_tv(&self, id: &str) -> Result<TruthValue, PlnError> {
        if let Some(a) = self.atoms.get(id) {
            return Ok(a.tv);
        }
        if let Some(l) = self.links.get(id) {
            return Ok(l.tv);
        }
        Err(PlnError::AtomNotFound(id.to_string()))
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /// Apply an inference rule to the given premises.
    ///
    /// Validates that the premises exist and that the rule can be applied,
    /// then returns a `PlnInferenceResult` describing the conclusion.  The
    /// caller decides whether to materialise the conclusion via
    /// `apply_inference`.
    pub fn infer(
        &mut self,
        rule: PlnInferenceRule,
        premise_ids: Vec<String>,
    ) -> Result<PlnInferenceResult, PlnError> {
        let tv = self.apply_rule(&rule, &premise_ids)?;

        if tv.confidence < self.config.min_confidence_threshold {
            return Err(PlnError::InsufficientConfidence(tv.confidence));
        }

        // Derive a conclusion ID deterministically from rule + premises
        let conclusion_id = conclusion_id_for(&rule, &premise_ids);

        // Compute confidence_delta relative to any prior belief
        let prior_confidence = self
            .atoms
            .get(&conclusion_id)
            .map(|a| a.tv.confidence)
            .or_else(|| self.links.get(&conclusion_id).map(|l| l.tv.confidence))
            .unwrap_or(0.0);

        let confidence_delta = tv.confidence - prior_confidence;

        self.inferences_performed += 1;

        Ok(PlnInferenceResult {
            conclusion_id,
            tv,
            rule_used: rule,
            premise_ids,
            confidence_delta,
        })
    }

    /// Materialise an inference result.
    ///
    /// * If the conclusion atom does not exist, it is created.
    /// * If it exists and `enable_revision` is `true`, its TV is revised.
    /// * If it exists and `enable_revision` is `false`, its TV is replaced.
    pub fn apply_inference(&mut self, result: PlnInferenceResult) -> Result<(), PlnError> {
        let id = &result.conclusion_id;

        if let Some(existing) = self.atoms.get_mut(id) {
            if self.config.enable_revision {
                existing.tv = existing.tv.revise(result.tv);
            } else {
                existing.tv = result.tv;
            }
        } else if let Some(existing) = self.links.get_mut(id) {
            if self.config.enable_revision {
                existing.tv = existing.tv.revise(result.tv);
            } else {
                existing.tv = result.tv;
            }
        } else {
            // New atom — materialise as a node
            let total = self.atoms.len() + self.links.len();
            if total >= self.config.max_atoms {
                return Err(PlnError::MaxAtomsExceeded);
            }
            let atom = PlnAtom {
                id: result.conclusion_id.clone(),
                name: result.conclusion_id.clone(),
                tv: result.tv,
                atom_type: AtomType::Node(format!("InferredBy({})", result.rule_used)),
            };
            self.atoms.insert(result.conclusion_id, atom);
        }
        Ok(())
    }

    // ── Chain search ──────────────────────────────────────────────────────────

    /// Find all inference chains from `from` to `to` up to `max_depth` hops.
    ///
    /// Returns a list of paths, where each path is a sequence of atom/link IDs
    /// starting at `from` and ending at `to`.
    pub fn find_chains(
        &self,
        from: &str,
        to: &str,
        max_depth: u8,
    ) -> Result<Vec<Vec<String>>, PlnError> {
        // Validate endpoints
        if !self.atoms.contains_key(from) && !self.links.contains_key(from) {
            return Err(PlnError::AtomNotFound(from.to_string()));
        }
        if !self.atoms.contains_key(to) && !self.links.contains_key(to) {
            return Err(PlnError::AtomNotFound(to.to_string()));
        }

        let mut results: Vec<Vec<String>> = Vec::new();
        // BFS state: (current_node, current_path, visited_set)
        let mut queue: VecDeque<(String, Vec<String>, HashSet<String>)> = VecDeque::new();
        {
            let mut init_visited = HashSet::new();
            init_visited.insert(from.to_string());
            queue.push_back((from.to_string(), vec![from.to_string()], init_visited));
        }

        while let Some((node, path, visited)) = queue.pop_front() {
            if path.len() as u8 > max_depth + 1 {
                continue;
            }

            if let Some(edges) = self.adjacency.get(&node) {
                for edge in edges {
                    for tgt in &edge.targets {
                        if visited.contains(tgt) {
                            // Cycle detected — record it but do not follow further
                            continue;
                        }

                        let mut new_path = path.clone();
                        new_path.push(edge.link_id.clone());
                        new_path.push(tgt.clone());

                        if tgt == to {
                            results.push(new_path);
                        } else if (new_path.len() as u8) <= max_depth * 2 + 1 {
                            let mut new_visited = visited.clone();
                            new_visited.insert(tgt.clone());
                            queue.push_back((tgt.clone(), new_path, new_visited));
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    // ── Aggregated queries ────────────────────────────────────────────────────

    /// Return the top `n` atoms sorted descending by confidence.
    pub fn most_confident(&self, n: usize) -> Vec<PlnAtom> {
        let mut atoms: Vec<&PlnAtom> = self.atoms.values().collect();
        atoms.sort_by(|a, b| {
            b.tv.confidence
                .partial_cmp(&a.tv.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        atoms.into_iter().take(n).cloned().collect()
    }

    /// Return all atoms whose confidence is strictly below `threshold`.
    pub fn uncertain_atoms(&self, threshold: f64) -> Vec<PlnAtom> {
        self.atoms
            .values()
            .filter(|a| a.tv.confidence < threshold)
            .cloned()
            .collect()
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /// Return a snapshot of current statistics.
    pub fn stats(&self) -> PlnStats {
        let atom_count = self.atoms.len();
        let link_count = self.links.len();
        let total = atom_count + link_count;

        let sum_conf: f64 = self.atoms.values().map(|a| a.tv.confidence).sum::<f64>()
            + self.links.values().map(|l| l.tv.confidence).sum::<f64>();

        let avg_confidence = if total > 0 {
            sum_conf / total as f64
        } else {
            0.0
        };

        PlnStats {
            atom_count,
            link_count,
            inferences_performed: self.inferences_performed,
            avg_confidence,
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Route the rule to the correct formula and return the derived TV.
    fn apply_rule(
        &self,
        rule: &PlnInferenceRule,
        premise_ids: &[String],
    ) -> Result<TruthValue, PlnError> {
        match rule {
            PlnInferenceRule::Negation => self.rule_negation(premise_ids),
            PlnInferenceRule::Conjunction => self.rule_conjunction(premise_ids),
            PlnInferenceRule::Disjunction => self.rule_disjunction(premise_ids),
            PlnInferenceRule::Revision => self.rule_revision(premise_ids),
            PlnInferenceRule::Deduction => self.rule_deduction(premise_ids),
            PlnInferenceRule::Induction => self.rule_induction(premise_ids),
            PlnInferenceRule::Abduction => self.rule_abduction(premise_ids),
            PlnInferenceRule::ModusPonens => self.rule_modus_ponens(premise_ids),
        }
    }

    fn require_tv(&self, id: &str) -> Result<TruthValue, PlnError> {
        self.get_tv(id)
    }

    fn rule_negation(&self, ids: &[String]) -> Result<TruthValue, PlnError> {
        if ids.len() != 1 {
            return Err(PlnError::InvalidRule(
                "Negation requires exactly 1 premise".into(),
            ));
        }
        Ok(self.require_tv(&ids[0])?.negate())
    }

    fn rule_conjunction(&self, ids: &[String]) -> Result<TruthValue, PlnError> {
        if ids.len() < 2 {
            return Err(PlnError::InvalidRule(
                "Conjunction requires at least 2 premises".into(),
            ));
        }
        let mut tv = self.require_tv(&ids[0])?;
        for id in &ids[1..] {
            tv = tv.conjunction(self.require_tv(id)?);
        }
        Ok(tv)
    }

    fn rule_disjunction(&self, ids: &[String]) -> Result<TruthValue, PlnError> {
        if ids.len() < 2 {
            return Err(PlnError::InvalidRule(
                "Disjunction requires at least 2 premises".into(),
            ));
        }
        let mut tv = self.require_tv(&ids[0])?;
        for id in &ids[1..] {
            tv = tv.disjunction(self.require_tv(id)?);
        }
        Ok(tv)
    }

    fn rule_revision(&self, ids: &[String]) -> Result<TruthValue, PlnError> {
        if ids.len() != 2 {
            return Err(PlnError::InvalidRule(
                "Revision requires exactly 2 premises".into(),
            ));
        }
        let tv1 = self.require_tv(&ids[0])?;
        let tv2 = self.require_tv(&ids[1])?;
        Ok(tv1.revise(tv2))
    }

    /// Deduction: premises must be [AB_link_id, BC_link_id, B_id, C_id].
    fn rule_deduction(&self, ids: &[String]) -> Result<TruthValue, PlnError> {
        if ids.len() != 4 {
            return Err(PlnError::InvalidRule(
                "Deduction requires 4 premises: [AB, BC, B, C]".into(),
            ));
        }
        let ab = self.require_tv(&ids[0])?;
        let bc = self.require_tv(&ids[1])?;
        let b = self.require_tv(&ids[2])?;
        let c = self.require_tv(&ids[3])?;
        Ok(TruthValue::deduction(ab, bc, b, c))
    }

    /// Induction: premises must be [AB_link_id, AC_link_id].
    fn rule_induction(&self, ids: &[String]) -> Result<TruthValue, PlnError> {
        if ids.len() != 2 {
            return Err(PlnError::InvalidRule(
                "Induction requires 2 premises: [AB, AC]".into(),
            ));
        }
        let ab = self.require_tv(&ids[0])?;
        let ac = self.require_tv(&ids[1])?;
        Ok(TruthValue::induction(ab, ac))
    }

    /// Abduction: premises must be [BA_link_id, CA_link_id].
    fn rule_abduction(&self, ids: &[String]) -> Result<TruthValue, PlnError> {
        if ids.len() != 2 {
            return Err(PlnError::InvalidRule(
                "Abduction requires 2 premises: [BA, CA]".into(),
            ));
        }
        let ba = self.require_tv(&ids[0])?;
        let ca = self.require_tv(&ids[1])?;
        Ok(TruthValue::abduction(ba, ca))
    }

    /// Modus Ponens: premises must be [A_id, AB_link_id].
    fn rule_modus_ponens(&self, ids: &[String]) -> Result<TruthValue, PlnError> {
        if ids.len() != 2 {
            return Err(PlnError::InvalidRule(
                "ModusPonens requires 2 premises: [A, A→B]".into(),
            ));
        }
        let a = self.require_tv(&ids[0])?;
        let ab = self.require_tv(&ids[1])?;
        Ok(TruthValue::modus_ponens(a, ab))
    }
}

// ── Utility ───────────────────────────────────────────────────────────────────

/// Build a deterministic conclusion ID from the rule and premise IDs.
fn conclusion_id_for(rule: &PlnInferenceRule, premise_ids: &[String]) -> String {
    let premises_joined = premise_ids.join("_");
    format!("inferred__{rule}__{premises_joined}")
}

// ── Inline xorshift64 for tests ───────────────────────────────────────────────
// (used only inside #[cfg(test)])

#[cfg(test)]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[cfg(test)]
fn rng_f64(state: &mut u64) -> f64 {
    (xorshift64(state) as f64) / (u64::MAX as f64)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn concept(id: &str, s: f64, c: f64) -> PlnAtom {
        PlnAtom::new(
            id,
            id,
            TruthValue::new(s, c),
            AtomType::Node("ConceptNode".into()),
        )
    }

    fn inh_link(id: &str, src: &str, dst: &str, s: f64, c: f64) -> PlnLink {
        PlnLink::new(
            id,
            LinkType::Inheritance,
            vec![src.into(), dst.into()],
            TruthValue::new(s, c),
        )
    }

    fn basic_net() -> ProbabilisticLogicNetwork {
        let mut pln = ProbabilisticLogicNetwork::new(PlnConfig::default());
        pln.add_atom(concept("a", 0.8, 0.9))
            .expect("test: should succeed");
        pln.add_atom(concept("b", 0.7, 0.8))
            .expect("test: should succeed");
        pln.add_atom(concept("c", 0.6, 0.7))
            .expect("test: should succeed");
        pln.add_link(inh_link("ab", "a", "b", 0.9, 0.8))
            .expect("test: should succeed");
        pln.add_link(inh_link("bc", "b", "c", 0.85, 0.75))
            .expect("test: should succeed");
        pln
    }

    // ── TruthValue construction ───────────────────────────────────────────────

    #[test]
    fn tv_clamp() {
        let tv = TruthValue::new(-0.5, 1.5);
        assert_eq!(tv.strength, 0.0);
        assert_eq!(tv.confidence, 1.0);
    }

    #[test]
    fn tv_unknown() {
        let tv = TruthValue::unknown();
        assert_eq!(tv.strength, 0.5);
        assert_eq!(tv.confidence, 0.0);
    }

    #[test]
    fn tv_certain_true() {
        let tv = TruthValue::certain_true();
        assert_eq!(tv.strength, 1.0);
        assert_eq!(tv.confidence, 1.0);
    }

    #[test]
    fn tv_certain_false() {
        let tv = TruthValue::certain_false();
        assert_eq!(tv.strength, 0.0);
        assert_eq!(tv.confidence, 1.0);
    }

    // ── count() ───────────────────────────────────────────────────────────────

    #[test]
    fn tv_count_zero_confidence() {
        let tv = TruthValue::new(0.5, 0.0);
        assert_eq!(tv.count(), 0.0);
    }

    #[test]
    fn tv_count_full_confidence() {
        let tv = TruthValue::certain_true();
        assert!(tv.count().is_infinite());
    }

    #[test]
    fn tv_count_half_confidence() {
        let tv = TruthValue::new(0.8, 0.5);
        // n = 0.5 / 0.5 = 1.0
        assert!((tv.count() - 1.0).abs() < 1e-10);
    }

    // ── Negation ─────────────────────────────────────────────────────────────

    #[test]
    fn tv_negate_basic() {
        let tv = TruthValue::new(0.7, 0.8);
        let neg = tv.negate();
        assert!((neg.strength - 0.3).abs() < 1e-10);
        assert_eq!(neg.confidence, 0.8);
    }

    #[test]
    fn tv_negate_double() {
        let tv = TruthValue::new(0.4, 0.6);
        let double_neg = tv.negate().negate();
        assert!((double_neg.strength - tv.strength).abs() < 1e-10);
        assert_eq!(double_neg.confidence, tv.confidence);
    }

    // ── Conjunction ──────────────────────────────────────────────────────────

    #[test]
    fn tv_conjunction_basic() {
        let a = TruthValue::new(0.8, 0.9);
        let b = TruthValue::new(0.6, 0.7);
        let c = a.conjunction(b);
        assert!((c.strength - 0.48).abs() < 1e-10);
        assert_eq!(c.confidence, 0.7);
    }

    #[test]
    fn tv_conjunction_with_false() {
        let a = TruthValue::new(0.9, 0.9);
        let b = TruthValue::certain_false();
        let c = a.conjunction(b);
        assert!(c.strength < 1e-10);
    }

    #[test]
    fn tv_conjunction_commutativity() {
        let a = TruthValue::new(0.7, 0.8);
        let b = TruthValue::new(0.5, 0.6);
        let ab = a.conjunction(b);
        let ba = b.conjunction(a);
        assert!((ab.strength - ba.strength).abs() < 1e-10);
        assert_eq!(ab.confidence, ba.confidence);
    }

    // ── Disjunction ──────────────────────────────────────────────────────────

    #[test]
    fn tv_disjunction_basic() {
        let a = TruthValue::new(0.5, 0.8);
        let b = TruthValue::new(0.5, 0.7);
        let d = a.disjunction(b);
        // s = 0.5 + 0.5 - 0.25 = 0.75
        assert!((d.strength - 0.75).abs() < 1e-10);
        assert_eq!(d.confidence, 0.7);
    }

    #[test]
    fn tv_disjunction_with_true() {
        let a = TruthValue::new(0.3, 0.8);
        let b = TruthValue::certain_true();
        let d = a.disjunction(b);
        assert!((d.strength - 1.0).abs() < 1e-10);
    }

    #[test]
    fn tv_disjunction_commutativity() {
        let a = TruthValue::new(0.3, 0.9);
        let b = TruthValue::new(0.7, 0.6);
        let ab = a.disjunction(b);
        let ba = b.disjunction(a);
        assert!((ab.strength - ba.strength).abs() < 1e-10);
    }

    // ── Revision ─────────────────────────────────────────────────────────────

    #[test]
    fn tv_revision_basic() {
        let t1 = TruthValue::new(0.8, 0.4); // n1 = 0.4/0.6 ≈ 0.667
        let t2 = TruthValue::new(0.6, 0.6); // n2 = 0.6/0.4 = 1.5
        let rev = t1.revise(t2);
        // n_total ≈ 2.167, s_rev ≈ (0.8*0.667 + 0.6*1.5) / 2.167 ≈ 0.665
        assert!(rev.confidence > t1.confidence);
        assert!(rev.confidence > t2.confidence);
        assert!((0.0..=1.0).contains(&rev.strength));
    }

    #[test]
    fn tv_revision_idempotent_certain() {
        let t1 = TruthValue::certain_true();
        let t2 = TruthValue::certain_true();
        let rev = t1.revise(t2);
        assert_eq!(rev.confidence, 1.0);
        assert_eq!(rev.strength, 1.0);
    }

    #[test]
    fn tv_revision_unknown_improves() {
        // unknown has n=0; evidence has n=1; total n=1 → c_rev = 1/2 = 0.5
        // Adding more evidence (c=0.8, n=4) should yield strictly higher confidence.
        let unknown = TruthValue::unknown();
        let strong_evidence = TruthValue::new(0.9, 0.8); // n=4
        let rev = unknown.revise(strong_evidence);
        // n_total = 4, c_rev = 4/5 = 0.8 >= strong_evidence.confidence
        assert!(rev.confidence >= strong_evidence.confidence);
        // Two independent high-confidence observations should compound
        let second = TruthValue::new(0.85, 0.8);
        let rev2 = rev.revise(second);
        assert!(rev2.confidence > rev.confidence);
    }

    #[test]
    fn tv_revision_symmetry() {
        let a = TruthValue::new(0.7, 0.5);
        let b = TruthValue::new(0.4, 0.3);
        let ab = a.revise(b);
        let ba = b.revise(a);
        assert!((ab.strength - ba.strength).abs() < 1e-10);
        assert!((ab.confidence - ba.confidence).abs() < 1e-10);
    }

    // ── Deduction ────────────────────────────────────────────────────────────

    #[test]
    fn tv_deduction_basic() {
        let ab = TruthValue::new(0.9, 0.8);
        let bc = TruthValue::new(0.85, 0.75);
        let b = TruthValue::new(0.7, 0.8);
        let c = TruthValue::new(0.6, 0.7);
        let ac = TruthValue::deduction(ab, bc, b, c);
        assert!((0.0..=1.0).contains(&ac.strength));
        assert!((0.0..=1.0).contains(&ac.confidence));
    }

    #[test]
    fn tv_deduction_high_confidence_chain() {
        let ab = TruthValue::new(0.95, 0.9);
        let bc = TruthValue::new(0.95, 0.9);
        let b = TruthValue::new(0.8, 0.9);
        let c = TruthValue::new(0.8, 0.9);
        let ac = TruthValue::deduction(ab, bc, b, c);
        // Should be high confidence and high strength
        assert!(ac.strength > 0.8);
    }

    #[test]
    fn tv_deduction_degenerate_sb_one() {
        // s_B == 1.0 should not panic
        let ab = TruthValue::new(0.9, 0.8);
        let bc = TruthValue::new(0.8, 0.7);
        let b = TruthValue::new(1.0, 0.9);
        let c = TruthValue::new(0.9, 0.8);
        let ac = TruthValue::deduction(ab, bc, b, c);
        assert!((0.0..=1.0).contains(&ac.strength));
    }

    // ── Induction ────────────────────────────────────────────────────────────

    #[test]
    fn tv_induction_basic() {
        let ab = TruthValue::new(0.8, 0.7);
        let ac = TruthValue::new(0.6, 0.6);
        let bc = TruthValue::induction(ab, ac);
        assert!((0.0..=1.0).contains(&bc.strength));
        assert!((0.0..=1.0).contains(&bc.confidence));
    }

    #[test]
    fn tv_induction_perfect_ab() {
        // s_AB == 1.0 → s_BC should equal s_AC
        let ab = TruthValue::new(1.0, 0.9);
        let ac = TruthValue::new(0.7, 0.8);
        let bc = TruthValue::induction(ab, ac);
        assert!((bc.strength - 0.7).abs() < 1e-6);
    }

    #[test]
    fn tv_induction_low_ab() {
        // Very low s_AB → s_BC clamped to 1
        let ab = TruthValue::new(0.001, 0.5);
        let ac = TruthValue::new(0.9, 0.8);
        let bc = TruthValue::induction(ab, ac);
        assert!(bc.strength <= 1.0);
    }

    // ── Abduction ────────────────────────────────────────────────────────────

    #[test]
    fn tv_abduction_basic() {
        let ba = TruthValue::new(0.7, 0.8);
        let ca = TruthValue::new(0.8, 0.7);
        let bc = TruthValue::abduction(ba, ca);
        assert!((0.0..=1.0).contains(&bc.strength));
        assert!((0.0..=1.0).contains(&bc.confidence));
    }

    #[test]
    fn tv_abduction_product_strength() {
        let ba = TruthValue::new(0.6, 0.8);
        let ca = TruthValue::new(0.5, 0.7);
        let bc = TruthValue::abduction(ba, ca);
        assert!((bc.strength - 0.3).abs() < 1e-10);
    }

    // ── Modus Ponens ─────────────────────────────────────────────────────────

    #[test]
    fn tv_modus_ponens_basic() {
        let a = TruthValue::new(0.9, 0.8);
        let ab = TruthValue::new(0.95, 0.9);
        let b = TruthValue::modus_ponens(a, ab);
        assert!(b.strength > 0.5);
        assert!((0.0..=1.0).contains(&b.confidence));
    }

    #[test]
    fn tv_modus_ponens_false_premise() {
        let a = TruthValue::certain_false();
        let ab = TruthValue::new(0.9, 0.9);
        let b = TruthValue::modus_ponens(a, ab);
        // s_A=0, s_AB=0.9 → s_B = 0 + 1 * 0.1 * 0.5 = 0.05
        assert!(b.strength < 0.2);
    }

    // ── add_atom ──────────────────────────────────────────────────────────────

    #[test]
    fn add_atom_ok() {
        let mut pln = ProbabilisticLogicNetwork::new(PlnConfig::default());
        pln.add_atom(concept("x", 0.5, 0.5))
            .expect("test: should succeed");
        assert!(pln.get_tv("x").is_ok());
    }

    #[test]
    fn add_atom_max_exceeded() {
        let cfg = PlnConfig {
            max_atoms: 2,
            ..Default::default()
        };
        let mut pln = ProbabilisticLogicNetwork::new(cfg);
        pln.add_atom(concept("a", 0.5, 0.5))
            .expect("test: should succeed");
        pln.add_atom(concept("b", 0.5, 0.5))
            .expect("test: should succeed");
        let r = pln.add_atom(concept("c", 0.5, 0.5));
        assert_eq!(r, Err(PlnError::MaxAtomsExceeded));
    }

    #[test]
    fn add_atom_replace_existing() {
        let mut pln = ProbabilisticLogicNetwork::new(PlnConfig::default());
        pln.add_atom(concept("x", 0.3, 0.4))
            .expect("test: should succeed");
        pln.add_atom(concept("x", 0.9, 0.9))
            .expect("test: should succeed"); // replace
        let tv = pln.get_tv("x").expect("test: should succeed");
        assert!((tv.strength - 0.9).abs() < 1e-10);
    }

    // ── add_link ──────────────────────────────────────────────────────────────

    #[test]
    fn add_link_ok() {
        let pln = basic_net();
        let tv = pln.get_tv("ab").expect("test: should succeed");
        assert!((tv.strength - 0.9).abs() < 1e-10);
    }

    #[test]
    fn add_link_missing_source() {
        let mut pln = ProbabilisticLogicNetwork::new(PlnConfig::default());
        pln.add_atom(concept("a", 0.5, 0.5))
            .expect("test: should succeed");
        let link = inh_link("ab", "a", "missing", 0.9, 0.8);
        let r = pln.add_link(link);
        assert!(matches!(r, Err(PlnError::AtomNotFound(_))));
    }

    // ── get_tv ────────────────────────────────────────────────────────────────

    #[test]
    fn get_tv_missing() {
        let pln = ProbabilisticLogicNetwork::new(PlnConfig::default());
        assert!(matches!(pln.get_tv("nope"), Err(PlnError::AtomNotFound(_))));
    }

    // ── Inference: Negation ──────────────────────────────────────────────────

    #[test]
    fn infer_negation() {
        let mut pln = basic_net();
        let r = pln
            .infer(PlnInferenceRule::Negation, vec!["a".into()])
            .expect("test: should succeed");
        assert!((r.tv.strength - (1.0 - 0.8)).abs() < 1e-10);
        assert!((r.tv.confidence - 0.9).abs() < 1e-10);
    }

    #[test]
    fn infer_negation_wrong_arity() {
        let mut pln = basic_net();
        let r = pln.infer(PlnInferenceRule::Negation, vec!["a".into(), "b".into()]);
        assert!(matches!(r, Err(PlnError::InvalidRule(_))));
    }

    // ── Inference: Conjunction ───────────────────────────────────────────────

    #[test]
    fn infer_conjunction() {
        let mut pln = basic_net();
        let r = pln
            .infer(PlnInferenceRule::Conjunction, vec!["a".into(), "b".into()])
            .expect("test: should succeed");
        assert!((r.tv.strength - (0.8 * 0.7)).abs() < 1e-10);
        assert_eq!(r.tv.confidence, 0.8_f64.min(0.9));
    }

    #[test]
    fn infer_conjunction_three_way() {
        let mut pln = basic_net();
        let r = pln
            .infer(
                PlnInferenceRule::Conjunction,
                vec!["a".into(), "b".into(), "c".into()],
            )
            .expect("test: should succeed");
        assert!((r.tv.strength - (0.8 * 0.7 * 0.6)).abs() < 1e-8);
    }

    // ── Inference: Disjunction ───────────────────────────────────────────────

    #[test]
    fn infer_disjunction() {
        let mut pln = basic_net();
        let r = pln
            .infer(PlnInferenceRule::Disjunction, vec!["a".into(), "b".into()])
            .expect("test: should succeed");
        // 0.8 + 0.7 - 0.56 = 0.94
        assert!((r.tv.strength - 0.94).abs() < 1e-10);
    }

    // ── Inference: Revision ──────────────────────────────────────────────────

    #[test]
    fn infer_revision() {
        let mut pln = basic_net();
        let r = pln
            .infer(PlnInferenceRule::Revision, vec!["a".into(), "b".into()])
            .expect("test: should succeed");
        assert!((0.0..=1.0).contains(&r.tv.strength));
        assert!((0.0..=1.0).contains(&r.tv.confidence));
    }

    #[test]
    fn infer_revision_wrong_arity() {
        let mut pln = basic_net();
        let r = pln.infer(PlnInferenceRule::Revision, vec!["a".into()]);
        assert!(matches!(r, Err(PlnError::InvalidRule(_))));
    }

    // ── Inference: Deduction ─────────────────────────────────────────────────

    #[test]
    fn infer_deduction() {
        let mut pln = basic_net();
        let r = pln
            .infer(
                PlnInferenceRule::Deduction,
                vec!["ab".into(), "bc".into(), "b".into(), "c".into()],
            )
            .expect("test: should succeed");
        assert!((0.0..=1.0).contains(&r.tv.strength));
        assert_eq!(r.rule_used, PlnInferenceRule::Deduction);
    }

    #[test]
    fn infer_deduction_wrong_arity() {
        let mut pln = basic_net();
        let r = pln.infer(PlnInferenceRule::Deduction, vec!["ab".into(), "bc".into()]);
        assert!(matches!(r, Err(PlnError::InvalidRule(_))));
    }

    // ── Inference: Induction ─────────────────────────────────────────────────

    #[test]
    fn infer_induction() {
        let mut pln = basic_net();
        let r = pln
            .infer(PlnInferenceRule::Induction, vec!["ab".into(), "bc".into()])
            .expect("test: should succeed");
        assert!((0.0..=1.0).contains(&r.tv.strength));
    }

    // ── Inference: Abduction ─────────────────────────────────────────────────

    #[test]
    fn infer_abduction() {
        let mut pln = basic_net();
        let r = pln
            .infer(PlnInferenceRule::Abduction, vec!["ab".into(), "bc".into()])
            .expect("test: should succeed");
        assert!((0.0..=1.0).contains(&r.tv.strength));
    }

    // ── Inference: ModusPonens ───────────────────────────────────────────────

    #[test]
    fn infer_modus_ponens() {
        let mut pln = basic_net();
        let r = pln
            .infer(PlnInferenceRule::ModusPonens, vec!["a".into(), "ab".into()])
            .expect("test: should succeed");
        assert!((0.0..=1.0).contains(&r.tv.strength));
    }

    // ── apply_inference ───────────────────────────────────────────────────────

    #[test]
    fn apply_inference_creates_atom() {
        let mut pln = basic_net();
        let r = pln
            .infer(PlnInferenceRule::Negation, vec!["a".into()])
            .expect("test: should succeed");
        let cid = r.conclusion_id.clone();
        pln.apply_inference(r).expect("test: should succeed");
        assert!(pln.get_tv(&cid).is_ok());
    }

    #[test]
    fn apply_inference_revises_existing() {
        let mut pln = basic_net();
        // First inference
        let r1 = pln
            .infer(PlnInferenceRule::Negation, vec!["a".into()])
            .expect("test: should succeed");
        let cid = r1.conclusion_id.clone();
        pln.apply_inference(r1).expect("test: should succeed");
        let tv_before = pln.get_tv(&cid).expect("test: should succeed");

        // Second inference (revision should increase confidence)
        let r2 = pln
            .infer(PlnInferenceRule::Negation, vec!["a".into()])
            .expect("test: should succeed");
        pln.apply_inference(r2).expect("test: should succeed");
        let tv_after = pln.get_tv(&cid).expect("test: should succeed");

        // Revised TV should have higher or equal confidence
        assert!(tv_after.confidence >= tv_before.confidence);
    }

    #[test]
    fn apply_inference_no_revision_replaces() {
        let mut pln = ProbabilisticLogicNetwork::new(PlnConfig {
            enable_revision: false,
            ..Default::default()
        });
        pln.add_atom(concept("a", 0.8, 0.9))
            .expect("test: should succeed");
        let r = pln
            .infer(PlnInferenceRule::Negation, vec!["a".into()])
            .expect("test: should succeed");
        let cid = r.conclusion_id.clone();
        let expected_tv = r.tv;
        pln.apply_inference(r).expect("test: should succeed");
        let tv = pln.get_tv(&cid).expect("test: should succeed");
        assert!((tv.strength - expected_tv.strength).abs() < 1e-10);
    }

    // ── confidence_delta ──────────────────────────────────────────────────────

    #[test]
    fn inference_result_confidence_delta_new_atom() {
        let mut pln = basic_net();
        let r = pln
            .infer(PlnInferenceRule::Negation, vec!["a".into()])
            .expect("test: should succeed");
        // New atom → prior confidence is 0 → delta == tv.confidence
        assert!((r.confidence_delta - r.tv.confidence).abs() < 1e-10);
    }

    // ── find_chains ───────────────────────────────────────────────────────────

    #[test]
    fn find_chains_direct() {
        let pln = basic_net();
        let chains = pln.find_chains("a", "b", 4).expect("test: should succeed");
        // a --ab--> b is one hop
        assert!(!chains.is_empty());
        let first = &chains[0];
        assert_eq!(first[0], "a");
        assert_eq!(*first.last().expect("test: should succeed"), "b");
    }

    #[test]
    fn find_chains_two_hop() {
        let pln = basic_net();
        let chains = pln.find_chains("a", "c", 4).expect("test: should succeed");
        assert!(!chains.is_empty());
        // Should contain a path through b
        let path = &chains[0];
        assert!(path.contains(&"b".to_string()));
    }

    #[test]
    fn find_chains_no_path() {
        let pln = basic_net();
        // c -> a doesn't exist (edges go a→b→c)
        let chains = pln.find_chains("c", "a", 4).expect("test: should succeed");
        assert!(chains.is_empty());
    }

    #[test]
    fn find_chains_missing_from() {
        let pln = basic_net();
        let r = pln.find_chains("missing", "a", 4);
        assert!(matches!(r, Err(PlnError::AtomNotFound(_))));
    }

    #[test]
    fn find_chains_missing_to() {
        let pln = basic_net();
        let r = pln.find_chains("a", "missing", 4);
        assert!(matches!(r, Err(PlnError::AtomNotFound(_))));
    }

    // ── most_confident ────────────────────────────────────────────────────────

    #[test]
    fn most_confident_top1() {
        let pln = basic_net();
        let top = pln.most_confident(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].id, "a"); // confidence 0.9
    }

    #[test]
    fn most_confident_all() {
        let pln = basic_net();
        let top = pln.most_confident(10);
        assert_eq!(top.len(), 3); // only 3 atoms
    }

    #[test]
    fn most_confident_zero() {
        let pln = basic_net();
        let top = pln.most_confident(0);
        assert!(top.is_empty());
    }

    #[test]
    fn most_confident_sorted() {
        let pln = basic_net();
        let top = pln.most_confident(3);
        for i in 0..top.len() - 1 {
            assert!(top[i].tv.confidence >= top[i + 1].tv.confidence);
        }
    }

    // ── uncertain_atoms ───────────────────────────────────────────────────────

    #[test]
    fn uncertain_atoms_basic() {
        let pln = basic_net();
        // All atoms have confidence < 1.0
        let uncertain = pln.uncertain_atoms(1.0);
        assert_eq!(uncertain.len(), 3);
    }

    #[test]
    fn uncertain_atoms_high_threshold() {
        let pln = basic_net();
        // Only atoms with confidence < 0.75 → only "c" (0.7)
        let uncertain = pln.uncertain_atoms(0.75);
        assert_eq!(uncertain.len(), 1);
        assert_eq!(uncertain[0].id, "c");
    }

    #[test]
    fn uncertain_atoms_none() {
        let pln = basic_net();
        let uncertain = pln.uncertain_atoms(0.0);
        assert!(uncertain.is_empty());
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn stats_basic() {
        let pln = basic_net();
        let s = pln.stats();
        assert_eq!(s.atom_count, 3);
        assert_eq!(s.link_count, 2);
        assert_eq!(s.inferences_performed, 0);
        assert!(s.avg_confidence > 0.0);
    }

    #[test]
    fn stats_inferences_count() {
        let mut pln = basic_net();
        pln.infer(PlnInferenceRule::Negation, vec!["a".into()])
            .expect("test: should succeed");
        pln.infer(PlnInferenceRule::Negation, vec!["b".into()])
            .expect("test: should succeed");
        assert_eq!(pln.stats().inferences_performed, 2);
    }

    #[test]
    fn stats_avg_confidence_empty() {
        let pln = ProbabilisticLogicNetwork::new(PlnConfig::default());
        let s = pln.stats();
        assert_eq!(s.avg_confidence, 0.0);
    }

    // ── Insufficient confidence threshold ────────────────────────────────────

    #[test]
    fn infer_below_threshold_rejected() {
        let cfg = PlnConfig {
            min_confidence_threshold: 0.99,
            ..Default::default()
        };
        let mut pln = ProbabilisticLogicNetwork::new(cfg);
        // Atoms with low confidence
        pln.add_atom(concept("a", 0.5, 0.1))
            .expect("test: should succeed");
        let r = pln.infer(PlnInferenceRule::Negation, vec!["a".into()]);
        assert!(matches!(r, Err(PlnError::InsufficientConfidence(_))));
    }

    // ── Randomised stress test ────────────────────────────────────────────────

    #[test]
    fn random_atoms_and_queries() {
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let mut pln = ProbabilisticLogicNetwork::new(PlnConfig::default());

        // Add 20 random atoms
        for i in 0..20u32 {
            let s = rng_f64(&mut state);
            let c = rng_f64(&mut state);
            let id = format!("node_{i}");
            pln.add_atom(concept(&id, s, c))
                .expect("test: should succeed");
        }

        let s = pln.stats();
        assert_eq!(s.atom_count, 20);
        assert!(s.avg_confidence >= 0.0);
        assert!(s.avg_confidence <= 1.0);
    }

    // ── Display / formatting ──────────────────────────────────────────────────

    #[test]
    fn tv_display() {
        let tv = TruthValue::new(0.75, 0.5);
        let s = format!("{tv}");
        assert!(s.contains("0.7500"));
        assert!(s.contains("0.5000"));
    }

    #[test]
    fn rule_display() {
        assert_eq!(format!("{}", PlnInferenceRule::Deduction), "Deduction");
        assert_eq!(format!("{}", PlnInferenceRule::ModusPonens), "ModusPonens");
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn deduction_sb_near_one_no_panic() {
        let ab = TruthValue::new(0.9, 0.8);
        let bc = TruthValue::new(0.8, 0.7);
        let b = TruthValue::new(0.9999999, 0.9);
        let c = TruthValue::new(0.9, 0.8);
        let ac = TruthValue::deduction(ab, bc, b, c);
        assert!((0.0..=1.0).contains(&ac.strength));
    }

    #[test]
    fn revision_with_zero_confidence_atoms() {
        let a = TruthValue::new(0.8, 0.0);
        let b = TruthValue::new(0.3, 0.0);
        let rev = a.revise(b);
        // Both have n=0 → unknown
        assert_eq!(rev.strength, 0.5);
    }

    #[test]
    fn conjunction_chain_three() {
        let a = TruthValue::new(0.9, 0.8);
        let b = TruthValue::new(0.8, 0.7);
        let c = TruthValue::new(0.7, 0.6);
        let abc = a.conjunction(b).conjunction(c);
        assert!((abc.strength - (0.9 * 0.8 * 0.7)).abs() < 1e-8);
        assert_eq!(abc.confidence, 0.6);
    }

    #[test]
    fn find_chains_self_loop_not_followed() {
        let mut pln = ProbabilisticLogicNetwork::new(PlnConfig::default());
        pln.add_atom(concept("x", 0.5, 0.5))
            .expect("test: should succeed");
        // Find chain from x to x with depth 0
        let chains = pln.find_chains("x", "x", 0).expect("test: should succeed");
        // No self-loops in result — adjacency is empty
        assert!(chains.is_empty());
    }
}
