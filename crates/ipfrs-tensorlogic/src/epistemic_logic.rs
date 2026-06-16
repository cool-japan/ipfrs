//! Epistemic Logic Reasoner — multi-agent epistemic logic over Kripke structures.
//!
//! Implements possible-worlds semantics, S5-style knowledge operators (K, M),
//! distributed common knowledge (C), and everyone-knows (E) operators.
//! The common knowledge fixed-point algorithm runs at most `max_depth` iterations
//! before returning [`EpistemicError::MaxDepthExceeded`].
//!
//! # Overview
//!
//! A [`KripkeModel`] is a tuple (W, {R_i}, V) where:
//! - W is a finite set of possible worlds
//! - R_i is an accessibility relation for agent i (i can't distinguish world u from world v if (u,v) ∈ R_i)
//! - V is a valuation mapping each world to its set of true atomic propositions
//!
//! [`EpistemicLogicReasoner`] wraps a Kripke model and provides:
//! - Model-theoretic evaluation of modal formulae
//! - Common knowledge fixed-point computation
//! - Relation closure (reflexive / transitive)
//! - Knowledge sets per agent
//!
//! # Example
//!
//! ```
//! use ipfrs_tensorlogic::epistemic_logic::{
//!     AgentId, EpistemicFormula, EpistemicLogicReasoner,
//! };
//! use std::collections::HashSet;
//!
//! let mut r = EpistemicLogicReasoner::new(50);
//! let w0 = r.add_world(["p".to_string()].into());
//! let w1 = r.add_world(HashSet::new());
//! let alice = AgentId("alice".to_string());
//! r.add_agent(alice.clone());
//! r.add_accessibility(alice.clone(), w0, w1).expect("example: should succeed in docs");
//! r.make_reflexive();
//! r.set_actual_world(w0).expect("example: should succeed in docs");
//!
//! let phi = EpistemicFormula::Atom("p".to_string());
//! // Alice cannot distinguish w0 from w1, so she doesn't know p.
//! let knows_p = EpistemicFormula::Knows {
//!     agent: alice.clone(),
//!     phi: Box::new(phi.clone()),
//! };
//! assert_eq!(r.evaluate_actual(&knows_p).expect("example: should succeed in docs"), false);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

// ─── Newtypes ────────────────────────────────────────────────────────────────

/// Identifies an agent in a multi-agent epistemic system.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AgentId(pub String);

/// Index of a possible world within a [`KripkeModel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorldId(pub usize);

// ─── Formula ─────────────────────────────────────────────────────────────────

/// An epistemic modal formula over atomic propositions.
///
/// The grammar is:
/// ```text
/// φ ::= p | ¬φ | φ∧ψ | φ∨ψ | K_i(φ) | M_i(φ) | E(φ) | C(φ)
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpistemicFormula {
    /// Atomic proposition.
    Atom(String),
    /// Logical negation.
    Not(Box<EpistemicFormula>),
    /// Logical conjunction.
    And(Box<EpistemicFormula>, Box<EpistemicFormula>),
    /// Logical disjunction.
    Or(Box<EpistemicFormula>, Box<EpistemicFormula>),
    /// K_i(φ): agent `agent` knows φ in every accessible world.
    Knows {
        agent: AgentId,
        phi: Box<EpistemicFormula>,
    },
    /// M_i(φ): agent `agent` considers φ possible (there exists an accessible world satisfying φ).
    Possible {
        agent: AgentId,
        phi: Box<EpistemicFormula>,
    },
    /// E(φ): every agent knows φ (but possibly not common knowledge).
    EveryoneKnows(Box<EpistemicFormula>),
    /// C(φ): φ is common knowledge among all agents (fixed-point of E).
    CommonKnowledge(Box<EpistemicFormula>),
}

// ─── Kripke structure ─────────────────────────────────────────────────────────

/// A single possible world with its set of true atomic propositions.
#[derive(Debug, Clone)]
pub struct PossibleWorld {
    /// Unique identifier within the model.
    pub id: WorldId,
    /// Atoms that are true in this world.
    pub true_propositions: HashSet<String>,
}

/// A directed accessibility edge (from, to) for a given agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessibilityRelation {
    /// The agent whose accessibility is described.
    pub agent: AgentId,
    /// Source world.
    pub from: WorldId,
    /// Target world (agent cannot distinguish `from` from `to`).
    pub to: WorldId,
}

/// A finite Kripke model: worlds, per-agent accessibility, and a designated actual world.
#[derive(Debug, Clone)]
pub struct KripkeModel {
    /// All possible worlds in the model.
    pub worlds: Vec<PossibleWorld>,
    /// Accessibility edges.
    pub relations: Vec<AccessibilityRelation>,
    /// The designated actual world.
    pub actual_world: WorldId,
}

impl KripkeModel {
    /// Create an empty model with actual world 0 (not yet valid until a world is added).
    pub fn new() -> Self {
        Self {
            worlds: Vec::new(),
            relations: Vec::new(),
            actual_world: WorldId(0),
        }
    }

    /// All worlds reachable from `world` for `agent` (one-step, including self if reflexive).
    ///
    /// Returns an empty `Vec` if the world has no outgoing relations for this agent.
    pub fn worlds_accessible_from(&self, agent: &AgentId, world: WorldId) -> Vec<WorldId> {
        self.relations
            .iter()
            .filter(|r| &r.agent == agent && r.from == world)
            .map(|r| r.to)
            .collect()
    }

    /// Returns `true` if the given `WorldId` exists in the model.
    fn world_exists(&self, id: WorldId) -> bool {
        self.worlds.iter().any(|w| w.id == id)
    }

    /// Look up a world by id.
    fn get_world(&self, id: WorldId) -> Option<&PossibleWorld> {
        self.worlds.iter().find(|w| w.id == id)
    }
}

impl Default for KripkeModel {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors that can occur during epistemic reasoning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpistemicError {
    /// No world with the given index exists in the model.
    WorldNotFound(usize),
    /// No agent with the given name exists in the reasoner.
    AgentNotFound(String),
    /// The common-knowledge fixed-point iteration exceeded `max_depth`.
    MaxDepthExceeded,
    /// The model has no worlds at all.
    EmptyModel,
}

impl std::fmt::Display for EpistemicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WorldNotFound(id) => write!(f, "world not found: {id}"),
            Self::AgentNotFound(name) => write!(f, "agent not found: {name}"),
            Self::MaxDepthExceeded => write!(f, "common knowledge iteration exceeded max depth"),
            Self::EmptyModel => write!(f, "the Kripke model has no worlds"),
        }
    }
}

impl std::error::Error for EpistemicError {}

// ─── Stats ───────────────────────────────────────────────────────────────────

/// Summary statistics for a [`EpistemicLogicReasoner`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicStats {
    pub world_count: usize,
    pub agent_count: usize,
    pub relation_count: usize,
    pub actual_world: usize,
}

// ─── Reasoner ────────────────────────────────────────────────────────────────

/// Multi-agent epistemic logic reasoner over a finite Kripke model.
///
/// Supports S5-style knowledge (K, M), everyone-knows (E), and common knowledge (C)
/// operators with a configurable fixed-point iteration depth bound.
#[derive(Debug, Clone)]
pub struct EpistemicLogicReasoner {
    /// The underlying Kripke structure.
    pub model: KripkeModel,
    /// Registered agents (order is stable).
    pub agents: Vec<AgentId>,
    /// Maximum fixed-point iterations for common knowledge computation.
    pub max_depth: usize,
    /// Monotonically increasing world id counter.
    next_world_id: usize,
}

impl EpistemicLogicReasoner {
    /// Create a new reasoner with an empty Kripke model.
    pub fn new(max_depth: usize) -> Self {
        Self {
            model: KripkeModel::new(),
            agents: Vec::new(),
            max_depth,
            next_world_id: 0,
        }
    }

    /// Add a world with the given set of true propositions, returning its [`WorldId`].
    pub fn add_world(&mut self, props: HashSet<String>) -> WorldId {
        let id = WorldId(self.next_world_id);
        self.next_world_id += 1;
        self.model.worlds.push(PossibleWorld {
            id,
            true_propositions: props,
        });
        id
    }

    /// Set the designated actual world.  Returns an error if the world does not exist.
    pub fn set_actual_world(&mut self, world: WorldId) -> Result<(), EpistemicError> {
        if !self.model.world_exists(world) {
            return Err(EpistemicError::WorldNotFound(world.0));
        }
        self.model.actual_world = world;
        Ok(())
    }

    /// Register a new agent.  Duplicate registrations are silently ignored.
    pub fn add_agent(&mut self, agent: AgentId) {
        if !self.agents.contains(&agent) {
            self.agents.push(agent);
        }
    }

    /// Add an accessibility relation edge (agent, from → to).
    ///
    /// Returns [`EpistemicError::WorldNotFound`] if either world is unknown.
    /// Duplicate edges are silently ignored.
    pub fn add_accessibility(
        &mut self,
        agent: AgentId,
        from: WorldId,
        to: WorldId,
    ) -> Result<(), EpistemicError> {
        if !self.model.world_exists(from) {
            return Err(EpistemicError::WorldNotFound(from.0));
        }
        if !self.model.world_exists(to) {
            return Err(EpistemicError::WorldNotFound(to.0));
        }
        let rel = AccessibilityRelation { agent, from, to };
        if !self.model.relations.contains(&rel) {
            self.model.relations.push(rel);
        }
        Ok(())
    }

    /// Add reflexive closure: for every (agent, world) pair, add the self-loop.
    pub fn make_reflexive(&mut self) {
        let pairs: Vec<(AgentId, WorldId)> = self
            .agents
            .iter()
            .flat_map(|a| self.model.worlds.iter().map(move |w| (a.clone(), w.id)))
            .collect();
        for (agent, world) in pairs {
            let rel = AccessibilityRelation {
                agent,
                from: world,
                to: world,
            };
            if !self.model.relations.contains(&rel) {
                self.model.relations.push(rel);
            }
        }
    }

    /// Add transitive closure via fixed-point iteration (BFS/forward-chaining).
    pub fn make_transitive(&mut self) {
        // Build an index: (agent, from) -> Vec<to>
        loop {
            let mut new_edges: Vec<AccessibilityRelation> = Vec::new();
            // For each (a, u→v) and (a, v→w) add (a, u→w)
            for r1 in &self.model.relations {
                for r2 in &self.model.relations {
                    if r1.agent == r2.agent && r1.to == r2.from {
                        let candidate = AccessibilityRelation {
                            agent: r1.agent.clone(),
                            from: r1.from,
                            to: r2.to,
                        };
                        if !self.model.relations.contains(&candidate)
                            && !new_edges.contains(&candidate)
                        {
                            new_edges.push(candidate);
                        }
                    }
                }
            }
            if new_edges.is_empty() {
                break;
            }
            self.model.relations.extend(new_edges);
        }
    }

    // ─── Evaluation ──────────────────────────────────────────────────────────

    /// Evaluate `formula` in the given `world`.
    pub fn evaluate(
        &self,
        formula: &EpistemicFormula,
        world: WorldId,
    ) -> Result<bool, EpistemicError> {
        if self.model.worlds.is_empty() {
            return Err(EpistemicError::EmptyModel);
        }
        if !self.model.world_exists(world) {
            return Err(EpistemicError::WorldNotFound(world.0));
        }
        self.eval_rec(formula, world)
    }

    /// Evaluate `formula` at the designated actual world.
    pub fn evaluate_actual(&self, formula: &EpistemicFormula) -> Result<bool, EpistemicError> {
        self.evaluate(formula, self.model.actual_world)
    }

    /// All worlds in which `formula` holds.
    pub fn satisfying_worlds(
        &self,
        formula: &EpistemicFormula,
    ) -> Result<Vec<WorldId>, EpistemicError> {
        if self.model.worlds.is_empty() {
            return Err(EpistemicError::EmptyModel);
        }
        let mut result = Vec::new();
        for w in &self.model.worlds {
            if self.eval_rec(formula, w.id)? {
                result.push(w.id);
            }
        }
        Ok(result)
    }

    /// Returns `true` if the formula holds in every world (is valid / tautology in the model).
    pub fn is_valid(&self, formula: &EpistemicFormula) -> Result<bool, EpistemicError> {
        if self.model.worlds.is_empty() {
            return Err(EpistemicError::EmptyModel);
        }
        for w in &self.model.worlds {
            if !self.eval_rec(formula, w.id)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Returns `true` if the formula holds in at least one world.
    pub fn is_satisfiable(&self, formula: &EpistemicFormula) -> Result<bool, EpistemicError> {
        if self.model.worlds.is_empty() {
            return Err(EpistemicError::EmptyModel);
        }
        for w in &self.model.worlds {
            if self.eval_rec(formula, w.id)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// The set of atomic propositions that agent `agent` *knows* in `world`:
    /// atoms that are true in every world accessible by the agent from `world`.
    pub fn knowledge_set(
        &self,
        agent: &AgentId,
        world: WorldId,
    ) -> Result<Vec<String>, EpistemicError> {
        if !self.agents.contains(agent) {
            return Err(EpistemicError::AgentNotFound(agent.0.clone()));
        }
        if !self.model.world_exists(world) {
            return Err(EpistemicError::WorldNotFound(world.0));
        }
        let accessible = self.model.worlds_accessible_from(agent, world);
        if accessible.is_empty() {
            // If no accessible worlds, agent knows nothing (vacuous case).
            return Ok(Vec::new());
        }
        // Collect propositions true in *all* accessible worlds.
        // Start with the union of all propositions, then intersect.
        let first_world = self
            .model
            .get_world(accessible[0])
            .ok_or(EpistemicError::WorldNotFound(accessible[0].0))?;
        let mut known: HashSet<String> = first_world.true_propositions.clone();
        for &wid in accessible.iter().skip(1) {
            let w = self
                .model
                .get_world(wid)
                .ok_or(EpistemicError::WorldNotFound(wid.0))?;
            known = known.intersection(&w.true_propositions).cloned().collect();
        }
        let mut result: Vec<String> = known.into_iter().collect();
        result.sort();
        Ok(result)
    }

    /// Summary statistics for the current model.
    pub fn stats(&self) -> EpistemicStats {
        EpistemicStats {
            world_count: self.model.worlds.len(),
            agent_count: self.agents.len(),
            relation_count: self.model.relations.len(),
            actual_world: self.model.actual_world.0,
        }
    }

    // ─── Internal recursive evaluator ────────────────────────────────────────

    fn eval_rec(&self, formula: &EpistemicFormula, world: WorldId) -> Result<bool, EpistemicError> {
        match formula {
            EpistemicFormula::Atom(name) => {
                let w = self
                    .model
                    .get_world(world)
                    .ok_or(EpistemicError::WorldNotFound(world.0))?;
                Ok(w.true_propositions.contains(name))
            }

            EpistemicFormula::Not(inner) => Ok(!self.eval_rec(inner, world)?),

            EpistemicFormula::And(lhs, rhs) => {
                Ok(self.eval_rec(lhs, world)? && self.eval_rec(rhs, world)?)
            }

            EpistemicFormula::Or(lhs, rhs) => {
                Ok(self.eval_rec(lhs, world)? || self.eval_rec(rhs, world)?)
            }

            EpistemicFormula::Knows { agent, phi } => self.eval_knows(agent, phi, world),

            EpistemicFormula::Possible { agent, phi } => self.eval_possible(agent, phi, world),

            EpistemicFormula::EveryoneKnows(phi) => self.eval_everyone_knows(phi, world),

            EpistemicFormula::CommonKnowledge(phi) => self.eval_common_knowledge(phi, world),
        }
    }

    /// K_i(φ) at `world`: φ holds in every world accessible by agent i from `world`.
    fn eval_knows(
        &self,
        agent: &AgentId,
        phi: &EpistemicFormula,
        world: WorldId,
    ) -> Result<bool, EpistemicError> {
        let accessible = self.model.worlds_accessible_from(agent, world);
        // If accessible is empty (non-serial relation), knowledge is vacuously true.
        for w in accessible {
            if !self.eval_rec(phi, w)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// M_i(φ) at `world`: φ holds in at least one world accessible by agent i from `world`.
    fn eval_possible(
        &self,
        agent: &AgentId,
        phi: &EpistemicFormula,
        world: WorldId,
    ) -> Result<bool, EpistemicError> {
        let accessible = self.model.worlds_accessible_from(agent, world);
        for w in accessible {
            if self.eval_rec(phi, w)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// E(φ) at `world`: for every registered agent i, K_i(φ) holds.
    fn eval_everyone_knows(
        &self,
        phi: &EpistemicFormula,
        world: WorldId,
    ) -> Result<bool, EpistemicError> {
        for agent in &self.agents {
            if !self.eval_knows(agent, phi, world)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// C(φ): common knowledge fixed-point.
    ///
    /// The semantics: C(φ) holds at w iff φ holds at w and at every world reachable
    /// via any finite sequence of epistemic accessibility steps across all agents.
    ///
    /// Algorithmically: compute the set of worlds reachable from `world` via the
    /// *union* accessibility relation (BFS), then verify φ holds at every such world.
    ///
    /// This is equivalent to: C(φ) = φ ∧ E(C(φ)), whose fixed-point can be computed
    /// iteratively: start with all worlds satisfying φ, then repeatedly remove worlds
    /// where some agent can reach a world that does not satisfy φ in the current set.
    fn eval_common_knowledge(
        &self,
        phi: &EpistemicFormula,
        world: WorldId,
    ) -> Result<bool, EpistemicError> {
        // Step 1: find all worlds reachable from `world` via the union relation (BFS).
        let reachable = self.reachable_via_union(world)?;

        // Step 2: check φ in every reachable world.
        for w in &reachable {
            if !self.eval_rec(phi, *w)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// BFS over the *union* of all agents' accessibility relations from `start`.
    ///
    /// Returns the set of all worlds reachable (including `start` itself if reflexive,
    /// or explicitly if we always include the starting world).
    ///
    /// Bounded by `max_depth` BFS steps to guard against pathological models.
    fn reachable_via_union(&self, start: WorldId) -> Result<Vec<WorldId>, EpistemicError> {
        let mut visited: HashSet<WorldId> = HashSet::new();
        let mut queue: VecDeque<(WorldId, usize)> = VecDeque::new();
        visited.insert(start);
        queue.push_back((start, 0));

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= self.max_depth {
                // If we still have frontier worlds at max_depth, we've exceeded the bound.
                return Err(EpistemicError::MaxDepthExceeded);
            }
            for rel in &self.model.relations {
                if rel.from == current && !visited.contains(&rel.to) {
                    visited.insert(rel.to);
                    queue.push_back((rel.to, depth + 1));
                }
            }
        }

        Ok(visited.into_iter().collect())
    }

    // ─── Extended helpers ─────────────────────────────────────────────────────

    /// Compute the set of worlds satisfying φ that remain stable under the
    /// E-operator restricted to the candidate set (iterated E-operator).
    ///
    /// This gives the maximal set S ⊆ W such that for all w∈S, φ holds and
    /// every agent's accessible worlds from w also lie in S — i.e. the stable
    /// fixed-point characterising common knowledge.
    ///
    /// Returns the stable set or an error if it does not converge within `max_depth`.
    pub fn common_knowledge_worlds(
        &self,
        phi: &EpistemicFormula,
    ) -> Result<HashSet<WorldId>, EpistemicError> {
        if self.model.worlds.is_empty() {
            return Err(EpistemicError::EmptyModel);
        }

        // Initial candidate: worlds satisfying φ.
        let mut candidate: HashSet<WorldId> = HashSet::new();
        for w in &self.model.worlds {
            if self.eval_rec(phi, w.id)? {
                candidate.insert(w.id);
            }
        }

        // Iteratively shrink: remove worlds from which some agent can reach outside.
        for _iter in 0..self.max_depth {
            let mut next = candidate.clone();
            for &w in &candidate {
                for agent in &self.agents {
                    let accessible = self.model.worlds_accessible_from(agent, w);
                    for aw in accessible {
                        if !candidate.contains(&aw) {
                            next.remove(&w);
                            break;
                        }
                    }
                    if !next.contains(&w) {
                        break;
                    }
                }
            }
            if next == candidate {
                return Ok(candidate);
            }
            candidate = next;
        }

        Err(EpistemicError::MaxDepthExceeded)
    }

    /// Build a map from each agent to the set of worlds they *know* to be
    /// indistinguishable from `world` (the equivalence class if relation is an
    /// equivalence; otherwise the reachability set).
    pub fn epistemic_partition(
        &self,
        world: WorldId,
    ) -> Result<HashMap<AgentId, Vec<WorldId>>, EpistemicError> {
        if !self.model.world_exists(world) {
            return Err(EpistemicError::WorldNotFound(world.0));
        }
        let mut map = HashMap::new();
        for agent in &self.agents {
            let reachable = self.model.worlds_accessible_from(agent, world);
            map.insert(agent.clone(), reachable);
        }
        Ok(map)
    }

    /// Check whether the model satisfies the T-axiom for the given agent:
    /// K_i(φ) → φ (knowledge implies truth), i.e. the accessibility relation is reflexive.
    pub fn satisfies_t_axiom(&self, agent: &AgentId) -> bool {
        for world in &self.model.worlds {
            let accessible = self.model.worlds_accessible_from(agent, world.id);
            if !accessible.contains(&world.id) {
                return false;
            }
        }
        true
    }

    /// Check whether the model satisfies the 4-axiom for the given agent:
    /// K_i(φ) → K_i(K_i(φ)), i.e. the accessibility relation is transitive.
    pub fn satisfies_4_axiom(&self, agent: &AgentId) -> bool {
        for r1 in &self.model.relations {
            if &r1.agent != agent {
                continue;
            }
            for r2 in &self.model.relations {
                if &r2.agent != agent || r2.from != r1.to {
                    continue;
                }
                // (r1.from → r1.to → r2.to) must have (r1.from → r2.to)
                let has_direct = self
                    .model
                    .relations
                    .iter()
                    .any(|r| &r.agent == agent && r.from == r1.from && r.to == r2.to);
                if !has_direct {
                    return false;
                }
            }
        }
        true
    }

    /// Check whether the model satisfies the 5-axiom for the given agent:
    /// ¬K_i(φ) → K_i(¬K_i(φ)), i.e. the accessibility relation is Euclidean.
    pub fn satisfies_5_axiom(&self, agent: &AgentId) -> bool {
        // Euclidean: if (u,v) ∈ R and (u,w) ∈ R then (v,w) ∈ R
        let agent_rels: Vec<&AccessibilityRelation> = self
            .model
            .relations
            .iter()
            .filter(|r| &r.agent == agent)
            .collect();
        for r1 in &agent_rels {
            for r2 in &agent_rels {
                if r1.from != r2.from {
                    continue;
                }
                // r1: u→v, r2: u→w  =>  need v→w
                let has_vw = self
                    .model
                    .relations
                    .iter()
                    .any(|r| &r.agent == agent && r.from == r1.to && r.to == r2.to);
                if !has_vw {
                    return false;
                }
            }
        }
        true
    }

    /// Check whether the model satisfies the B-axiom for the given agent:
    /// φ → K_i(M_i(φ)), i.e. the accessibility relation is symmetric.
    pub fn satisfies_b_axiom(&self, agent: &AgentId) -> bool {
        for r in &self.model.relations {
            if &r.agent != agent {
                continue;
            }
            // (u,v) must have (v,u)
            let symmetric = self
                .model
                .relations
                .iter()
                .any(|r2| &r2.agent == agent && r2.from == r.to && r2.to == r.from);
            if !symmetric {
                return false;
            }
        }
        true
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::epistemic_logic::{
        AgentId, EpistemicError, EpistemicFormula, EpistemicLogicReasoner, WorldId,
    };
    use std::collections::HashSet;

    // ── Helpers ─────────────────────────────────────────────────────────────

    fn make_atom(s: &str) -> EpistemicFormula {
        EpistemicFormula::Atom(s.to_string())
    }

    fn alice() -> AgentId {
        AgentId("alice".to_string())
    }
    fn bob() -> AgentId {
        AgentId("bob".to_string())
    }

    fn props(atoms: &[&str]) -> HashSet<String> {
        atoms.iter().map(|s| s.to_string()).collect()
    }

    // ── Test 1: basic atom evaluation ────────────────────────────────────────

    #[test]
    fn test_atom_true() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&["p"]));
        r.set_actual_world(w0).expect("test: should succeed");
        assert!(r
            .evaluate_actual(&make_atom("p"))
            .expect("test: should succeed"));
    }

    #[test]
    fn test_atom_false() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&[]));
        r.set_actual_world(w0).expect("test: should succeed");
        assert!(!r
            .evaluate_actual(&make_atom("q"))
            .expect("test: should succeed"));
    }

    // ── Test 3: negation ────────────────────────────────────────────────────

    #[test]
    fn test_not() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&["p"]));
        r.set_actual_world(w0).expect("test: should succeed");
        let not_p = EpistemicFormula::Not(Box::new(make_atom("p")));
        assert!(!r.evaluate_actual(&not_p).expect("test: should succeed"));
        let not_q = EpistemicFormula::Not(Box::new(make_atom("q")));
        assert!(r.evaluate_actual(&not_q).expect("test: should succeed"));
    }

    // ── Test 4: conjunction ──────────────────────────────────────────────────

    #[test]
    fn test_and() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&["p", "q"]));
        r.set_actual_world(w0).expect("test: should succeed");
        let p_and_q = EpistemicFormula::And(Box::new(make_atom("p")), Box::new(make_atom("q")));
        assert!(r.evaluate_actual(&p_and_q).expect("test: should succeed"));
        let p_and_r = EpistemicFormula::And(Box::new(make_atom("p")), Box::new(make_atom("r")));
        assert!(!r.evaluate_actual(&p_and_r).expect("test: should succeed"));
    }

    // ── Test 5: disjunction ──────────────────────────────────────────────────

    #[test]
    fn test_or() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&["p"]));
        r.set_actual_world(w0).expect("test: should succeed");
        let p_or_q = EpistemicFormula::Or(Box::new(make_atom("p")), Box::new(make_atom("q")));
        assert!(r.evaluate_actual(&p_or_q).expect("test: should succeed"));
        let r_or_s = EpistemicFormula::Or(Box::new(make_atom("r")), Box::new(make_atom("s")));
        assert!(!r.evaluate_actual(&r_or_s).expect("test: should succeed"));
    }

    // ── Test 6: Knows — reflexive model ──────────────────────────────────────

    #[test]
    fn test_knows_reflexive() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&["p"]));
        r.add_agent(alice());
        r.set_actual_world(w0).expect("test: should succeed");
        r.make_reflexive();
        let k_p = EpistemicFormula::Knows {
            agent: alice(),
            phi: Box::new(make_atom("p")),
        };
        // Alice only sees w0 (reflexive), p is true there.
        assert!(r.evaluate_actual(&k_p).expect("test: should succeed"));
    }

    // ── Test 7: Knows — agent doesn't know because of alternative world ──────

    #[test]
    fn test_knows_false_alternative_world() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&["p"]));
        let w1 = r.add_world(props(&[]));
        r.add_agent(alice());
        r.set_actual_world(w0).expect("test: should succeed");
        r.add_accessibility(alice(), w0, w1)
            .expect("test: should succeed");
        r.make_reflexive();
        let k_p = EpistemicFormula::Knows {
            agent: alice(),
            phi: Box::new(make_atom("p")),
        };
        assert!(!r.evaluate_actual(&k_p).expect("test: should succeed"));
    }

    // ── Test 8: Possible operator ────────────────────────────────────────────

    #[test]
    fn test_possible() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&[]));
        let w1 = r.add_world(props(&["p"]));
        r.add_agent(alice());
        r.set_actual_world(w0).expect("test: should succeed");
        r.add_accessibility(alice(), w0, w1)
            .expect("test: should succeed");
        r.make_reflexive();
        let m_p = EpistemicFormula::Possible {
            agent: alice(),
            phi: Box::new(make_atom("p")),
        };
        assert!(r.evaluate_actual(&m_p).expect("test: should succeed"));
    }

    // ── Test 9: EveryoneKnows ────────────────────────────────────────────────

    #[test]
    fn test_everyone_knows() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&["p"]));
        r.add_agent(alice());
        r.add_agent(bob());
        r.set_actual_world(w0).expect("test: should succeed");
        r.make_reflexive();
        let e_p = EpistemicFormula::EveryoneKnows(Box::new(make_atom("p")));
        assert!(r.evaluate_actual(&e_p).expect("test: should succeed"));
    }

    // ── Test 10: EveryoneKnows fails if one agent doesn't know ───────────────

    #[test]
    fn test_everyone_knows_partial_failure() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&["p"]));
        let w1 = r.add_world(props(&[]));
        r.add_agent(alice());
        r.add_agent(bob());
        r.set_actual_world(w0).expect("test: should succeed");
        // Alice can distinguish, Bob cannot
        r.add_accessibility(bob(), w0, w1)
            .expect("test: should succeed");
        r.make_reflexive();
        let e_p = EpistemicFormula::EveryoneKnows(Box::new(make_atom("p")));
        // Bob doesn't know p (sees w1 where p is false)
        assert!(!r.evaluate_actual(&e_p).expect("test: should succeed"));
    }

    // ── Test 11: CommonKnowledge — simple convergence ─────────────────────────

    #[test]
    fn test_common_knowledge_simple() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&["p"]));
        r.add_agent(alice());
        r.add_agent(bob());
        r.set_actual_world(w0).expect("test: should succeed");
        r.make_reflexive();
        let c_p = EpistemicFormula::CommonKnowledge(Box::new(make_atom("p")));
        assert!(r.evaluate_actual(&c_p).expect("test: should succeed"));
    }

    // ── Test 12: CommonKnowledge false — one unreachable world breaks it ──────

    #[test]
    fn test_common_knowledge_false() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&["p"]));
        let w1 = r.add_world(props(&[]));
        r.add_agent(alice());
        r.set_actual_world(w0).expect("test: should succeed");
        r.add_accessibility(alice(), w0, w1)
            .expect("test: should succeed");
        r.make_reflexive();
        // w1 is reachable but doesn't satisfy p
        let c_p = EpistemicFormula::CommonKnowledge(Box::new(make_atom("p")));
        assert!(!r.evaluate_actual(&c_p).expect("test: should succeed"));
    }

    // ── Test 13: knowledge_set ────────────────────────────────────────────────

    #[test]
    fn test_knowledge_set() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&["p", "q"]));
        let w1 = r.add_world(props(&["p"]));
        r.add_agent(alice());
        r.set_actual_world(w0).expect("test: should succeed");
        r.add_accessibility(alice(), w0, w1)
            .expect("test: should succeed");
        r.make_reflexive();
        // Alice sees w0 and w1; p is in both, q only in w0
        let known = r.knowledge_set(&alice(), w0).expect("test: should succeed");
        assert!(known.contains(&"p".to_string()));
        assert!(!known.contains(&"q".to_string()));
    }

    // ── Test 14: add_world returns distinct ids ──────────────────────────────

    #[test]
    fn test_world_ids_distinct() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&[]));
        let w1 = r.add_world(props(&[]));
        let w2 = r.add_world(props(&[]));
        assert_ne!(w0, w1);
        assert_ne!(w1, w2);
    }

    // ── Test 15: set_actual_world error ──────────────────────────────────────

    #[test]
    fn test_set_actual_world_error() {
        let mut r = EpistemicLogicReasoner::new(50);
        let err = r.set_actual_world(WorldId(999)).unwrap_err();
        assert_eq!(err, EpistemicError::WorldNotFound(999));
    }

    // ── Test 16: add_accessibility error — world not found ───────────────────

    #[test]
    fn test_add_accessibility_error_from() {
        let mut r = EpistemicLogicReasoner::new(50);
        r.add_agent(alice());
        let err = r
            .add_accessibility(alice(), WorldId(0), WorldId(1))
            .unwrap_err();
        assert_eq!(err, EpistemicError::WorldNotFound(0));
    }

    #[test]
    fn test_add_accessibility_error_to() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&[]));
        r.add_agent(alice());
        let err = r.add_accessibility(alice(), w0, WorldId(999)).unwrap_err();
        assert_eq!(err, EpistemicError::WorldNotFound(999));
    }

    // ── Test 18: EmptyModel error ────────────────────────────────────────────

    #[test]
    fn test_empty_model_error() {
        let r = EpistemicLogicReasoner::new(50);
        let err = r.evaluate_actual(&make_atom("p")).unwrap_err();
        assert_eq!(err, EpistemicError::EmptyModel);
    }

    // ── Test 19: make_reflexive idempotent ───────────────────────────────────

    #[test]
    fn test_make_reflexive_idempotent() {
        let mut r = EpistemicLogicReasoner::new(50);
        r.add_world(props(&["p"]));
        r.add_agent(alice());
        r.make_reflexive();
        let count1 = r.model.relations.len();
        r.make_reflexive();
        let count2 = r.model.relations.len();
        assert_eq!(count1, count2);
    }

    // ── Test 20: make_transitive ─────────────────────────────────────────────

    #[test]
    fn test_make_transitive() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&[]));
        let w1 = r.add_world(props(&[]));
        let w2 = r.add_world(props(&[]));
        r.add_agent(alice());
        r.add_accessibility(alice(), w0, w1)
            .expect("test: should succeed");
        r.add_accessibility(alice(), w1, w2)
            .expect("test: should succeed");
        r.make_transitive();
        // Should now have w0 → w2
        let accessible = r.model.worlds_accessible_from(&alice(), w0);
        assert!(accessible.contains(&w2));
    }

    // ── Test 21: is_valid ────────────────────────────────────────────────────

    #[test]
    fn test_is_valid_tautology() {
        let mut r = EpistemicLogicReasoner::new(50);
        r.add_world(props(&["p"]));
        r.add_world(props(&["p"]));
        // p∨¬p is always true
        let taut = EpistemicFormula::Or(
            Box::new(make_atom("p")),
            Box::new(EpistemicFormula::Not(Box::new(make_atom("p")))),
        );
        assert!(r.is_valid(&taut).expect("test: should succeed"));
    }

    #[test]
    fn test_is_valid_false() {
        let mut r = EpistemicLogicReasoner::new(50);
        r.add_world(props(&["p"]));
        r.add_world(props(&[])); // p is false here
        assert!(!r.is_valid(&make_atom("p")).expect("test: should succeed"));
    }

    // ── Test 23: is_satisfiable ──────────────────────────────────────────────

    #[test]
    fn test_is_satisfiable_true() {
        let mut r = EpistemicLogicReasoner::new(50);
        r.add_world(props(&["p"]));
        r.add_world(props(&[]));
        assert!(r
            .is_satisfiable(&make_atom("p"))
            .expect("test: should succeed"));
    }

    #[test]
    fn test_is_satisfiable_false() {
        let mut r = EpistemicLogicReasoner::new(50);
        r.add_world(props(&[]));
        assert!(!r
            .is_satisfiable(&make_atom("p"))
            .expect("test: should succeed"));
    }

    // ── Test 25: satisfying_worlds ───────────────────────────────────────────

    #[test]
    fn test_satisfying_worlds() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&["p"]));
        let w1 = r.add_world(props(&[]));
        let worlds = r
            .satisfying_worlds(&make_atom("p"))
            .expect("test: should succeed");
        assert!(worlds.contains(&w0));
        assert!(!worlds.contains(&w1));
    }

    // ── Test 26: stats ───────────────────────────────────────────────────────

    #[test]
    fn test_stats() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&[]));
        let w1 = r.add_world(props(&[]));
        r.add_agent(alice());
        r.add_accessibility(alice(), w0, w1)
            .expect("test: should succeed");
        r.set_actual_world(w0).expect("test: should succeed");
        let s = r.stats();
        assert_eq!(s.world_count, 2);
        assert_eq!(s.agent_count, 1);
        assert_eq!(s.relation_count, 1);
        assert_eq!(s.actual_world, w0.0);
    }

    // ── Test 27: duplicate agent not duplicated ───────────────────────────────

    #[test]
    fn test_duplicate_agent_ignored() {
        let mut r = EpistemicLogicReasoner::new(50);
        r.add_agent(alice());
        r.add_agent(alice());
        assert_eq!(r.agents.len(), 1);
    }

    // ── Test 28: knowledge_set agent not found ────────────────────────────────

    #[test]
    fn test_knowledge_set_agent_not_found() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&["p"]));
        let err = r.knowledge_set(&alice(), w0).unwrap_err();
        assert_eq!(err, EpistemicError::AgentNotFound("alice".to_string()));
    }

    // ── Test 29: knowledge_set world not found ───────────────────────────────

    #[test]
    fn test_knowledge_set_world_not_found() {
        let mut r = EpistemicLogicReasoner::new(50);
        r.add_agent(alice());
        let err = r.knowledge_set(&alice(), WorldId(42)).unwrap_err();
        assert_eq!(err, EpistemicError::WorldNotFound(42));
    }

    // ── Test 30: T-axiom check ────────────────────────────────────────────────

    #[test]
    fn test_t_axiom_reflexive() {
        let mut r = EpistemicLogicReasoner::new(50);
        r.add_world(props(&[]));
        r.add_world(props(&[]));
        r.add_agent(alice());
        r.make_reflexive();
        assert!(r.satisfies_t_axiom(&alice()));
    }

    #[test]
    fn test_t_axiom_not_reflexive() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&[]));
        let w1 = r.add_world(props(&[]));
        r.add_agent(alice());
        r.add_accessibility(alice(), w0, w1)
            .expect("test: should succeed");
        // No reflexive closure
        assert!(!r.satisfies_t_axiom(&alice()));
    }

    // ── Test 32: 4-axiom (transitivity) check ────────────────────────────────

    #[test]
    fn test_4_axiom_transitive() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&[]));
        let w1 = r.add_world(props(&[]));
        let w2 = r.add_world(props(&[]));
        r.add_agent(alice());
        r.add_accessibility(alice(), w0, w1)
            .expect("test: should succeed");
        r.add_accessibility(alice(), w1, w2)
            .expect("test: should succeed");
        r.make_transitive();
        assert!(r.satisfies_4_axiom(&alice()));
    }

    #[test]
    fn test_4_axiom_not_transitive() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&[]));
        let w1 = r.add_world(props(&[]));
        let w2 = r.add_world(props(&[]));
        r.add_agent(alice());
        r.add_accessibility(alice(), w0, w1)
            .expect("test: should succeed");
        r.add_accessibility(alice(), w1, w2)
            .expect("test: should succeed");
        // No transitive closure -> fails 4
        assert!(!r.satisfies_4_axiom(&alice()));
    }

    // ── Test 34: B-axiom (symmetry) check ────────────────────────────────────

    #[test]
    fn test_b_axiom_symmetric() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&[]));
        let w1 = r.add_world(props(&[]));
        r.add_agent(alice());
        r.add_accessibility(alice(), w0, w1)
            .expect("test: should succeed");
        r.add_accessibility(alice(), w1, w0)
            .expect("test: should succeed");
        assert!(r.satisfies_b_axiom(&alice()));
    }

    #[test]
    fn test_b_axiom_not_symmetric() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&[]));
        let w1 = r.add_world(props(&[]));
        r.add_agent(alice());
        r.add_accessibility(alice(), w0, w1)
            .expect("test: should succeed");
        // Missing w1 → w0
        assert!(!r.satisfies_b_axiom(&alice()));
    }

    // ── Test 36: 5-axiom (Euclidean) check ───────────────────────────────────

    #[test]
    fn test_5_axiom_euclidean() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&[]));
        let w1 = r.add_world(props(&[]));
        let w2 = r.add_world(props(&[]));
        r.add_agent(alice());
        // u→v and u→w => v→w
        r.add_accessibility(alice(), w0, w1)
            .expect("test: should succeed");
        r.add_accessibility(alice(), w0, w2)
            .expect("test: should succeed");
        r.add_accessibility(alice(), w1, w1)
            .expect("test: should succeed");
        r.add_accessibility(alice(), w1, w2)
            .expect("test: should succeed");
        r.add_accessibility(alice(), w2, w1)
            .expect("test: should succeed");
        r.add_accessibility(alice(), w2, w2)
            .expect("test: should succeed");
        assert!(r.satisfies_5_axiom(&alice()));
    }

    // ── Test 37: common_knowledge_worlds ─────────────────────────────────────

    #[test]
    fn test_common_knowledge_worlds() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&["p"]));
        let w1 = r.add_world(props(&["p"]));
        let w2 = r.add_world(props(&[]));
        r.add_agent(alice());
        r.add_agent(bob());
        r.set_actual_world(w0).expect("test: should succeed");
        // Alice and Bob both see w0 and w1 but not w2
        r.add_accessibility(alice(), w0, w1)
            .expect("test: should succeed");
        r.add_accessibility(bob(), w0, w1)
            .expect("test: should succeed");
        r.make_reflexive();
        let ck_worlds = r
            .common_knowledge_worlds(&make_atom("p"))
            .expect("test: should succeed");
        assert!(ck_worlds.contains(&w0));
        assert!(ck_worlds.contains(&w1));
        assert!(!ck_worlds.contains(&w2));
    }

    // ── Test 38: nested formula (K(K(p))) ────────────────────────────────────

    #[test]
    fn test_nested_knows() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&["p"]));
        r.add_agent(alice());
        r.set_actual_world(w0).expect("test: should succeed");
        r.make_reflexive();
        // Alice knows that Alice knows p (S4/S5)
        let k_k_p = EpistemicFormula::Knows {
            agent: alice(),
            phi: Box::new(EpistemicFormula::Knows {
                agent: alice(),
                phi: Box::new(make_atom("p")),
            }),
        };
        assert!(r.evaluate_actual(&k_k_p).expect("test: should succeed"));
    }

    // ── Test 39: epistemic_partition ─────────────────────────────────────────

    #[test]
    fn test_epistemic_partition() {
        let mut r = EpistemicLogicReasoner::new(50);
        let w0 = r.add_world(props(&[]));
        let w1 = r.add_world(props(&[]));
        r.add_agent(alice());
        r.add_accessibility(alice(), w0, w1)
            .expect("test: should succeed");
        r.set_actual_world(w0).expect("test: should succeed");
        let partition = r.epistemic_partition(w0).expect("test: should succeed");
        let alice_reachable = partition.get(&alice()).expect("test: should succeed");
        assert!(alice_reachable.contains(&w1));
    }

    // ── Test 40: evaluate — unknown world gives error ─────────────────────────

    #[test]
    fn test_evaluate_unknown_world_error() {
        let mut r = EpistemicLogicReasoner::new(50);
        r.add_world(props(&[]));
        let err = r.evaluate(&make_atom("p"), WorldId(999)).unwrap_err();
        assert_eq!(err, EpistemicError::WorldNotFound(999));
    }

    // ── Test 41: multi-agent scenario — muddy children puzzle sketch ──────────
    //
    // World layout:
    //   w_tt: alice_muddy ∧ bob_muddy   (both muddy)
    //   w_tf: alice_muddy               (only alice muddy)
    //   w_ft: bob_muddy                 (only bob muddy)
    //   w_ff: (neither muddy)
    //
    // Accessibility (standard muddy-children — agent can't see own forehead):
    //   alice: w_tt ↔ w_ft  (she sees bob muddy in both, can't tell if she is muddy)
    //          w_tf ↔ w_ff
    //   bob:   w_tt ↔ w_tf  (he sees alice muddy in both, can't tell if he is muddy)
    //          w_ft ↔ w_ff
    //
    // In w_tt (actual world):
    //   Alice sees {w_tt, w_ft}: alice_muddy is true in w_tt but false in w_ft  => doesn't know
    //   Bob   sees {w_tt, w_tf}: bob_muddy   is true in w_tt but false in w_tf  => doesn't know

    #[test]
    fn test_muddy_children_two_agents() {
        let mut r = EpistemicLogicReasoner::new(100);
        let w_tt = r.add_world(props(&["alice_muddy", "bob_muddy"]));
        let w_tf = r.add_world(props(&["alice_muddy"])); // only alice muddy
        let w_ft = r.add_world(props(&["bob_muddy"])); // only bob muddy
        let _w_ff = r.add_world(props(&[]));
        r.add_agent(alice());
        r.add_agent(bob());
        r.set_actual_world(w_tt).expect("test: should succeed");

        // Alice confuses tt↔ft (she sees bob muddy in both, can't see her own forehead)
        r.add_accessibility(alice(), w_tt, w_ft)
            .expect("test: should succeed");
        r.add_accessibility(alice(), w_ft, w_tt)
            .expect("test: should succeed");
        // Bob confuses tt↔tf (he sees alice muddy in both, can't see his own forehead)
        r.add_accessibility(bob(), w_tt, w_tf)
            .expect("test: should succeed");
        r.add_accessibility(bob(), w_tf, w_tt)
            .expect("test: should succeed");
        r.make_reflexive();

        // Alice sees {w_tt, w_ft}: alice_muddy is true in w_tt but FALSE in w_ft => doesn't know
        let alice_knows_muddy = EpistemicFormula::Knows {
            agent: alice(),
            phi: Box::new(make_atom("alice_muddy")),
        };
        assert!(!r
            .evaluate_actual(&alice_knows_muddy)
            .expect("test: should succeed"));

        // Bob sees {w_tt, w_tf}: bob_muddy is true in w_tt but FALSE in w_tf => doesn't know
        let bob_knows_muddy = EpistemicFormula::Knows {
            agent: bob(),
            phi: Box::new(make_atom("bob_muddy")),
        };
        assert!(!r
            .evaluate_actual(&bob_knows_muddy)
            .expect("test: should succeed"));

        // Alice does KNOW that bob is muddy at w_tt (w_tt has bob_muddy; w_ft has bob_muddy)
        let alice_knows_bob_muddy = EpistemicFormula::Knows {
            agent: alice(),
            phi: Box::new(make_atom("bob_muddy")),
        };
        assert!(r
            .evaluate_actual(&alice_knows_bob_muddy)
            .expect("test: should succeed"));

        // At least one is muddy — true at actual world
        let at_least_one = EpistemicFormula::Or(
            Box::new(make_atom("alice_muddy")),
            Box::new(make_atom("bob_muddy")),
        );
        assert!(r
            .evaluate_actual(&at_least_one)
            .expect("test: should succeed"));
    }
}
