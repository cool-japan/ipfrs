//! Peer Reputation Graph with trust propagation and reputation scoring.
//!
//! This module provides a graph-based peer reputation system that models trust
//! relationships between peers as weighted directed edges. Trust propagates
//! transitively through the graph (up to a configurable depth), enabling
//! reputation to be inferred even for peers with limited direct interaction.
//!
//! # Design
//!
//! The graph stores:
//! - **Direct scores** – updated by explicit positive/negative interactions via
//!   an exponential moving average (EMA).
//! - **Propagated scores** – computed by multi-hop BFS, damping trust at each
//!   hop, so peers trusted by highly-reputed peers inherit some of that trust.
//! - **Combined scores** – a configurable linear blend of the two.
//!
//! All floating-point arithmetic avoids `unwrap()` and clamps values to
//! `[0.0, 1.0]` to maintain invariants.
//!
//! # Examples
//!
//! ```rust
//! use ipfrs_network::peer_reputation_graph::{PeerReputationGraph, GraphConfig};
//!
//! let config = GraphConfig::default();
//! let mut graph = PeerReputationGraph::new(config);
//!
//! graph.add_peer("alice".to_string()).unwrap();
//! graph.add_peer("bob".to_string()).unwrap();
//! graph.add_edge("alice", "bob", 0.8).unwrap();
//!
//! let event = graph.record_interaction("alice", true, 0.5).unwrap();
//! let _updated = graph.propagate_trust();
//!
//! let score = graph.reputation("bob").unwrap();
//! println!("Bob combined score: {:.3}", score.combined_score);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// PRNG (no rand crate)
// ---------------------------------------------------------------------------

/// Xorshift64 pseudo-random number generator.
///
/// Used internally for tie-breaking and jitter; state must be non-zero.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can arise when working with the [`PeerReputationGraph`].
#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    /// The peer ID was not found in the graph.
    PeerNotFound(String),
    /// The directed edge between two peers was not found.
    EdgeNotFound { from: String, to: String },
    /// Attempted to add a self-loop edge.
    SelfLoop(String),
    /// A weight value was outside `[0.0, 1.0]`.
    InvalidWeight(f64),
    /// Adding the peer would exceed `GraphConfig::max_peers`.
    MaxPeersExceeded,
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::PeerNotFound(id) => write!(f, "peer not found: {id}"),
            GraphError::EdgeNotFound { from, to } => {
                write!(f, "edge not found: {from} → {to}")
            }
            GraphError::SelfLoop(id) => write!(f, "self-loop not allowed for peer {id}"),
            GraphError::InvalidWeight(w) => {
                write!(f, "invalid edge weight {w}; must be in [0.0, 1.0]")
            }
            GraphError::MaxPeersExceeded => write!(f, "maximum peer count exceeded"),
        }
    }
}

impl std::error::Error for GraphError {}

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

/// A directed weighted trust edge between two peers.
#[derive(Debug, Clone)]
pub struct TrustEdge {
    /// Source peer identifier.
    pub from_peer: String,
    /// Destination peer identifier.
    pub to_peer: String,
    /// Edge weight in `[0.0, 1.0]`; higher means stronger trust.
    pub weight: f64,
    /// Unix timestamp (seconds) when the edge was first created.
    pub created_at: u64,
    /// Unix timestamp (seconds) of the last weight update.
    pub updated_at: u64,
    /// Number of times this edge has been updated via `add_edge`.
    pub interaction_count: u64,
}

/// A full reputation snapshot for a single peer.
#[derive(Debug, Clone)]
pub struct ReputationScore {
    /// Peer identifier.
    pub peer_id: String,
    /// Score derived solely from direct interactions.
    pub direct_score: f64,
    /// Score computed by trust propagation through the graph.
    pub propagated_score: f64,
    /// Weighted blend of direct and propagated scores.
    pub combined_score: f64,
    /// Confidence in the score in `[0.0, 1.0]`; based on observation count.
    pub confidence: f64,
    /// Percentile rank among all peers (0 = lowest, 1 = highest).
    pub percentile: f64,
}

/// Events emitted by the reputation graph for observability.
#[derive(Debug, Clone)]
pub enum ReputationEvent {
    /// A positive interaction was recorded for the given peer.
    PositiveInteraction { peer_id: String, magnitude: f64 },
    /// A negative interaction was recorded for the given peer.
    NegativeInteraction { peer_id: String, magnitude: f64 },
    /// A new trust edge was added (or an existing one was updated).
    EdgeAdded {
        from: String,
        to: String,
        weight: f64,
    },
    /// A trust edge was removed.
    EdgeRemoved { from: String, to: String },
    /// A peer's score decayed during a decay tick.
    ScoreDecayed {
        peer_id: String,
        old_score: f64,
        new_score: f64,
    },
}

/// Configuration parameters for [`PeerReputationGraph`].
#[derive(Debug, Clone)]
pub struct GraphConfig {
    /// Multiplicative decay applied to all direct scores per `decay_scores` call.
    /// Default: 0.99 (≈ 1 % loss per tick).
    pub trust_decay_factor: f64,
    /// Maximum BFS depth for trust propagation. Default: 3.
    pub propagation_depth: u8,
    /// Damping factor applied at each additional hop. Default: 0.5.
    pub propagation_damping: f64,
    /// Edges whose weight drops below this are pruned during `decay_scores`.
    pub min_edge_weight: f64,
    /// Maximum number of peers the graph may hold.
    pub max_peers: usize,
    /// Weight of direct score in the combined score formula. Default: 0.6.
    pub combination_weight_direct: f64,
    /// Weight of propagated score in the combined score formula. Default: 0.4.
    pub combination_weight_propagated: f64,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            trust_decay_factor: 0.99,
            propagation_depth: 3,
            propagation_damping: 0.5,
            min_edge_weight: 0.01,
            max_peers: 10_000,
            combination_weight_direct: 0.6,
            combination_weight_propagated: 0.4,
        }
    }
}

/// Aggregate statistics for the entire reputation graph.
#[derive(Debug, Clone)]
pub struct GraphStats {
    /// Number of peers currently in the graph.
    pub peer_count: usize,
    /// Number of directed trust edges.
    pub edge_count: usize,
    /// Average out-degree (edges per source peer).
    pub avg_out_degree: f64,
    /// Mean combined reputation score across all peers.
    pub avg_reputation: f64,
    /// Top 5 peers by combined score.
    pub top_peers: Vec<String>,
    /// Peers that have no incoming or outgoing edges.
    pub isolated_peers: usize,
}

// ---------------------------------------------------------------------------
// Internal peer state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct PeerState {
    peer_id: String,
    direct_score: f64,
    propagated_score: f64,
    /// Total number of positive + negative interactions recorded.
    interaction_count: u64,
}

impl PeerState {
    fn new(peer_id: String) -> Self {
        Self {
            peer_id,
            direct_score: 0.5, // neutral starting point
            propagated_score: 0.0,
            interaction_count: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Main graph struct
// ---------------------------------------------------------------------------

/// Graph-based peer reputation system with trust propagation.
///
/// The graph maintains directed weighted edges between peer identifiers and
/// computes reputation scores via both direct interaction history and
/// multi-hop trust propagation.
pub struct PeerReputationGraph {
    config: GraphConfig,
    /// Peer ID → internal state.
    peers: HashMap<String, PeerState>,
    /// (from, to) → edge.
    edges: HashMap<(String, String), TrustEdge>,
    /// Adjacency index: from_peer → set of to_peers (outgoing).
    out_adj: HashMap<String, HashSet<String>>,
    /// Adjacency index: to_peer → set of from_peers (incoming).
    in_adj: HashMap<String, HashSet<String>>,
    /// Simple monotonic timestamp counter (incremented per mutation).
    clock: u64,
    /// PRNG state for tie-breaking.
    rng_state: u64,
}

impl PeerReputationGraph {
    /// Create a new graph with the given configuration.
    pub fn new(config: GraphConfig) -> Self {
        Self {
            config,
            peers: HashMap::new(),
            edges: HashMap::new(),
            out_adj: HashMap::new(),
            in_adj: HashMap::new(),
            clock: 1,
            rng_state: 0xdeadbeef_cafebabe,
        }
    }

    // -----------------------------------------------------------------------
    // Peer management
    // -----------------------------------------------------------------------

    /// Add a peer to the graph.
    ///
    /// Returns `GraphError::MaxPeersExceeded` if the configured limit would be
    /// breached.  Adding an already-existing peer is a no-op (returns `Ok`).
    pub fn add_peer(&mut self, peer_id: String) -> Result<(), GraphError> {
        if self.peers.contains_key(&peer_id) {
            return Ok(());
        }
        if self.peers.len() >= self.config.max_peers {
            return Err(GraphError::MaxPeersExceeded);
        }
        self.out_adj.entry(peer_id.clone()).or_default();
        self.in_adj.entry(peer_id.clone()).or_default();
        self.peers.insert(peer_id.clone(), PeerState::new(peer_id));
        self.tick();
        Ok(())
    }

    /// Remove a peer and all edges incident to it.
    pub fn remove_peer(&mut self, peer_id: &str) -> Result<(), GraphError> {
        if !self.peers.contains_key(peer_id) {
            return Err(GraphError::PeerNotFound(peer_id.to_string()));
        }

        // Collect edges to remove to avoid borrow conflicts.
        let out_peers: Vec<String> = self
            .out_adj
            .get(peer_id)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default();
        let in_peers: Vec<String> = self
            .in_adj
            .get(peer_id)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default();

        for to in &out_peers {
            self.edges.remove(&(peer_id.to_string(), to.clone()));
            if let Some(set) = self.in_adj.get_mut(to) {
                set.remove(peer_id);
            }
        }
        for from in &in_peers {
            self.edges.remove(&(from.clone(), peer_id.to_string()));
            if let Some(set) = self.out_adj.get_mut(from) {
                set.remove(peer_id);
            }
        }

        self.out_adj.remove(peer_id);
        self.in_adj.remove(peer_id);
        self.peers.remove(peer_id);
        self.tick();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Edge management
    // -----------------------------------------------------------------------

    /// Add or update a directed trust edge.
    ///
    /// # Errors
    ///
    /// - `GraphError::PeerNotFound` if either peer is absent.
    /// - `GraphError::SelfLoop` if `from == to`.
    /// - `GraphError::InvalidWeight` if `weight` is outside `[0.0, 1.0]`.
    pub fn add_edge(&mut self, from: &str, to: &str, weight: f64) -> Result<(), GraphError> {
        if from == to {
            return Err(GraphError::SelfLoop(from.to_string()));
        }
        if !(0.0..=1.0).contains(&weight) {
            return Err(GraphError::InvalidWeight(weight));
        }
        if !self.peers.contains_key(from) {
            return Err(GraphError::PeerNotFound(from.to_string()));
        }
        if !self.peers.contains_key(to) {
            return Err(GraphError::PeerNotFound(to.to_string()));
        }

        let now = self.tick();
        let key = (from.to_string(), to.to_string());

        if let Some(edge) = self.edges.get_mut(&key) {
            edge.weight = weight;
            edge.updated_at = now;
            edge.interaction_count += 1;
        } else {
            self.edges.insert(
                key.clone(),
                TrustEdge {
                    from_peer: from.to_string(),
                    to_peer: to.to_string(),
                    weight,
                    created_at: now,
                    updated_at: now,
                    interaction_count: 1,
                },
            );
            self.out_adj
                .entry(from.to_string())
                .or_default()
                .insert(to.to_string());
            self.in_adj
                .entry(to.to_string())
                .or_default()
                .insert(from.to_string());
        }
        Ok(())
    }

    /// Remove a directed trust edge.
    pub fn remove_edge(&mut self, from: &str, to: &str) -> Result<(), GraphError> {
        let key = (from.to_string(), to.to_string());
        if self.edges.remove(&key).is_none() {
            return Err(GraphError::EdgeNotFound {
                from: from.to_string(),
                to: to.to_string(),
            });
        }
        if let Some(set) = self.out_adj.get_mut(from) {
            set.remove(to);
        }
        if let Some(set) = self.in_adj.get_mut(to) {
            set.remove(from);
        }
        self.tick();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Interaction recording
    // -----------------------------------------------------------------------

    /// Record an interaction with a peer, updating its direct score via EMA.
    ///
    /// Alpha = 0.1.  Positive interactions add `magnitude`; negative ones
    /// subtract it.  The result is clamped to `[0.0, 1.0]`.
    ///
    /// Returns the corresponding [`ReputationEvent`].
    pub fn record_interaction(
        &mut self,
        peer_id: &str,
        positive: bool,
        magnitude: f64,
    ) -> Result<ReputationEvent, GraphError> {
        const ALPHA: f64 = 0.1;

        let state = self
            .peers
            .get_mut(peer_id)
            .ok_or_else(|| GraphError::PeerNotFound(peer_id.to_string()))?;

        let delta = if positive { magnitude } else { -magnitude };

        // EMA formula: new = old + alpha * (target - old)
        // where target = clamp(old + delta, 0, 1)
        let target = (state.direct_score + delta).clamp(0.0, 1.0);
        let ema = state.direct_score + ALPHA * (target - state.direct_score);

        state.direct_score = ema.clamp(0.0, 1.0);
        state.interaction_count += 1;

        let event = if positive {
            ReputationEvent::PositiveInteraction {
                peer_id: peer_id.to_string(),
                magnitude,
            }
        } else {
            ReputationEvent::NegativeInteraction {
                peer_id: peer_id.to_string(),
                magnitude,
            }
        };
        self.tick();
        Ok(event)
    }

    // -----------------------------------------------------------------------
    // Trust propagation
    // -----------------------------------------------------------------------

    /// Propagate trust through the graph via BFS.
    ///
    /// For each peer `p`, performs BFS up to `propagation_depth` hops.
    /// The propagated score contribution of a path
    /// `p → n1 → n2 → … → nk` is:
    ///
    /// ```text
    /// w(p→n1) × direct_score(p) × damping^1
    /// + w(n1→n2) × direct_score(n1) × damping^2  (if n2 is the target)
    /// …
    /// ```
    ///
    /// Each peer's `propagated_score` is set to the sum of all such
    /// contributions clamped to `[0.0, 1.0]`.
    ///
    /// Returns the number of peers whose propagated score was updated.
    pub fn propagate_trust(&mut self) -> usize {
        // Collect current direct scores to avoid mutable borrow conflicts.
        let direct_scores: HashMap<String, f64> = self
            .peers
            .iter()
            .map(|(id, s)| (id.clone(), s.direct_score))
            .collect();

        // New propagated scores accumulator.
        let mut propagated: HashMap<String, f64> =
            self.peers.keys().map(|id| (id.clone(), 0.0_f64)).collect();

        let depth = self.config.propagation_depth as usize;
        let damping = self.config.propagation_damping;

        // For every source peer, BFS outward and accumulate contributions.
        for (src_id, &src_direct) in &direct_scores {
            if src_direct == 0.0 {
                continue;
            }

            // BFS queue: (current_node, accumulated_trust, current_hop)
            let mut queue: VecDeque<(String, f64, usize)> = VecDeque::new();
            queue.push_back((src_id.clone(), src_direct, 0));

            let mut visited: HashSet<String> = HashSet::new();
            visited.insert(src_id.clone());

            while let Some((node, trust, hop)) = queue.pop_front() {
                if hop >= depth {
                    continue;
                }

                let neighbors: Vec<(String, f64)> = self
                    .out_adj
                    .get(&node)
                    .map(|set| {
                        set.iter()
                            .filter_map(|nb| {
                                let key = (node.clone(), nb.clone());
                                self.edges.get(&key).map(|e| (nb.clone(), e.weight))
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                for (nb, edge_weight) in neighbors {
                    if visited.contains(&nb) {
                        continue;
                    }
                    visited.insert(nb.clone());

                    let contribution = edge_weight * trust * damping.powi((hop + 1) as i32);
                    if let Some(acc) = propagated.get_mut(&nb) {
                        *acc += contribution;
                    }

                    queue.push_back((nb, trust * edge_weight, hop + 1));
                }
            }
        }

        // Write back and count updates.
        let mut updated = 0_usize;
        for (id, new_prop) in propagated {
            if let Some(state) = self.peers.get_mut(&id) {
                let clamped = new_prop.clamp(0.0, 1.0);
                if (clamped - state.propagated_score).abs() > f64::EPSILON {
                    state.propagated_score = clamped;
                    updated += 1;
                }
            }
        }
        updated
    }

    // -----------------------------------------------------------------------
    // Score decay
    // -----------------------------------------------------------------------

    /// Apply decay to all direct scores and prune weak edges.
    ///
    /// Each `direct_score` is multiplied by `trust_decay_factor`.  A
    /// [`ReputationEvent::ScoreDecayed`] is emitted for any peer whose score
    /// changes by more than 0.001.  Edges whose weight drops below
    /// `min_edge_weight` are also pruned.
    pub fn decay_scores(&mut self) -> Vec<ReputationEvent> {
        let factor = self.config.trust_decay_factor;
        let min_weight = self.config.min_edge_weight;

        let mut events = Vec::new();

        for state in self.peers.values_mut() {
            let old = state.direct_score;
            let new = (old * factor).clamp(0.0, 1.0);
            if (new - old).abs() > 0.001 {
                events.push(ReputationEvent::ScoreDecayed {
                    peer_id: state.peer_id.clone(),
                    old_score: old,
                    new_score: new,
                });
            }
            state.direct_score = new;
        }

        // Prune edges below min_edge_weight.
        let to_prune: Vec<(String, String)> = self
            .edges
            .iter()
            .filter(|(_, e)| e.weight < min_weight)
            .map(|(k, _)| (k.0.clone(), k.1.clone()))
            .collect();

        for (from, to) in to_prune {
            // Best-effort removal; ignore errors.
            let _ = self.remove_edge(&from, &to);
        }

        self.tick();
        events
    }

    // -----------------------------------------------------------------------
    // Score query
    // -----------------------------------------------------------------------

    /// Retrieve the full [`ReputationScore`] for a peer.
    ///
    /// `combined_score` is a weighted sum of `direct_score` and
    /// `propagated_score` normalized so the total weight sums to 1.
    /// `confidence` is `min(1.0, interaction_count / 10.0)`.
    pub fn reputation(&self, peer_id: &str) -> Result<ReputationScore, GraphError> {
        let state = self
            .peers
            .get(peer_id)
            .ok_or_else(|| GraphError::PeerNotFound(peer_id.to_string()))?;

        let wd = self.config.combination_weight_direct;
        let wp = self.config.combination_weight_propagated;
        let total_weight = wd + wp;

        let combined = if total_weight > 0.0 {
            (wd * state.direct_score + wp * state.propagated_score) / total_weight
        } else {
            state.direct_score
        };

        let confidence = (state.interaction_count as f64 / 10.0).min(1.0);

        // Compute percentile rank.
        let combined_clamped = combined.clamp(0.0, 1.0);
        let percentile = self.percentile_of(peer_id, combined_clamped);

        Ok(ReputationScore {
            peer_id: peer_id.to_string(),
            direct_score: state.direct_score,
            propagated_score: state.propagated_score,
            combined_score: combined_clamped,
            confidence,
            percentile,
        })
    }

    /// Return the top `n` peers by combined score, sorted descending.
    pub fn top_peers(&self, n: usize) -> Vec<ReputationScore> {
        let mut scores: Vec<ReputationScore> = self
            .peers
            .keys()
            .filter_map(|id| self.reputation(id).ok())
            .collect();

        scores.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scores.truncate(n);
        scores
    }

    // -----------------------------------------------------------------------
    // Graph queries
    // -----------------------------------------------------------------------

    /// Peers that have a trust edge pointing **to** `peer_id` (in-neighbours).
    pub fn peers_trusting(&self, peer_id: &str) -> Vec<String> {
        self.in_adj
            .get(peer_id)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Peers that `peer_id` trusts — i.e. out-neighbours of `peer_id`.
    pub fn peers_trusted_by(&self, peer_id: &str) -> Vec<String> {
        self.out_adj
            .get(peer_id)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Compute aggregate statistics for the graph.
    pub fn stats(&self) -> GraphStats {
        let peer_count = self.peers.len();
        let edge_count = self.edges.len();

        let avg_out_degree = if peer_count > 0 {
            edge_count as f64 / peer_count as f64
        } else {
            0.0
        };

        let avg_reputation = if peer_count > 0 {
            let sum: f64 = self
                .peers
                .keys()
                .filter_map(|id| self.reputation(id).ok())
                .map(|r| r.combined_score)
                .sum();
            sum / peer_count as f64
        } else {
            0.0
        };

        let mut top = self.top_peers(5);
        // Ensure deterministic order by peer_id for equal scores.
        top.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.peer_id.cmp(&b.peer_id))
        });
        let top_peers = top.into_iter().map(|r| r.peer_id).collect();

        let isolated_peers = self
            .peers
            .keys()
            .filter(|id| {
                self.out_adj.get(*id).map(|s| s.is_empty()).unwrap_or(true)
                    && self.in_adj.get(*id).map(|s| s.is_empty()).unwrap_or(true)
            })
            .count();

        GraphStats {
            peer_count,
            edge_count,
            avg_out_degree,
            avg_reputation,
            top_peers,
            isolated_peers,
        }
    }

    /// Remove peers that are isolated (no edges) **and** have a combined score
    /// below 0.1.
    ///
    /// Returns the number of peers removed.
    pub fn prune_isolated(&mut self) -> usize {
        let to_remove: Vec<String> = self
            .peers
            .iter()
            .filter(|(id, state)| {
                let out_empty = self.out_adj.get(*id).map(|s| s.is_empty()).unwrap_or(true);
                let in_empty = self.in_adj.get(*id).map(|s| s.is_empty()).unwrap_or(true);
                if !out_empty || !in_empty {
                    return false;
                }
                let wd = self.config.combination_weight_direct;
                let wp = self.config.combination_weight_propagated;
                let total = wd + wp;
                let combined = if total > 0.0 {
                    (wd * state.direct_score + wp * state.propagated_score) / total
                } else {
                    state.direct_score
                };
                combined < 0.1
            })
            .map(|(id, _)| id.clone())
            .collect();

        let count = to_remove.len();
        for id in to_remove {
            // Errors ignored; peer may already have been removed concurrently.
            let _ = self.remove_peer(&id);
        }
        count
    }

    // -----------------------------------------------------------------------
    // Direct edge / score accessors
    // -----------------------------------------------------------------------

    /// Return all edges in the graph (cloned).
    pub fn all_edges(&self) -> Vec<TrustEdge> {
        self.edges.values().cloned().collect()
    }

    /// Return the edge between two peers, if present.
    pub fn edge(&self, from: &str, to: &str) -> Option<&TrustEdge> {
        self.edges.get(&(from.to_string(), to.to_string()))
    }

    /// Return whether a peer is present in the graph.
    pub fn contains_peer(&self, peer_id: &str) -> bool {
        self.peers.contains_key(peer_id)
    }

    /// Return the current number of peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Return the current number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Advance the logical clock and return the new timestamp.
    fn tick(&mut self) -> u64 {
        self.clock += 1;
        // Use xorshift64 to add entropy that prevents degenerate patterns.
        let _ = xorshift64(&mut self.rng_state);
        self.clock
    }

    /// Compute the percentile rank of `combined` among all peers.
    fn percentile_of(&self, peer_id: &str, combined: f64) -> f64 {
        if self.peers.len() <= 1 {
            return 0.5;
        }
        let below = self
            .peers
            .iter()
            .filter(|(id, state)| {
                if id.as_str() == peer_id {
                    return false;
                }
                let wd = self.config.combination_weight_direct;
                let wp = self.config.combination_weight_propagated;
                let total = wd + wp;
                let other_combined = if total > 0.0 {
                    (wd * state.direct_score + wp * state.propagated_score) / total
                } else {
                    state.direct_score
                };
                other_combined < combined
            })
            .count();
        let n = self.peers.len() - 1; // exclude self
        if n == 0 {
            0.5
        } else {
            below as f64 / n as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph() -> PeerReputationGraph {
        PeerReputationGraph::new(GraphConfig::default())
    }

    // ------- add/remove peers -------

    #[test]
    fn test_add_peer_basic() {
        let mut g = make_graph();
        g.add_peer("alice".to_string())
            .expect("test: add_peer alice should succeed");
        assert!(g.contains_peer("alice"));
    }

    #[test]
    fn test_add_peer_idempotent() {
        let mut g = make_graph();
        g.add_peer("alice".to_string())
            .expect("test: add_peer alice should succeed");
        // Adding again must succeed silently.
        g.add_peer("alice".to_string())
            .expect("test: re-adding alice should succeed");
        assert_eq!(g.peer_count(), 1);
    }

    #[test]
    fn test_add_peer_max_exceeded() {
        let mut g = PeerReputationGraph::new(GraphConfig {
            max_peers: 2,
            ..Default::default()
        });
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        let err = g
            .add_peer("c".to_string())
            .expect_err("test: expected error when exceeding max peers");
        assert_eq!(err, GraphError::MaxPeersExceeded);
    }

    #[test]
    fn test_remove_peer_basic() {
        let mut g = make_graph();
        g.add_peer("alice".to_string())
            .expect("test: add_peer alice should succeed");
        g.remove_peer("alice")
            .expect("test: remove_peer alice should succeed");
        assert!(!g.contains_peer("alice"));
    }

    #[test]
    fn test_remove_peer_not_found() {
        let mut g = make_graph();
        let err = g
            .remove_peer("ghost")
            .expect_err("test: expected error removing nonexistent peer");
        assert!(matches!(err, GraphError::PeerNotFound(_)));
    }

    #[test]
    fn test_remove_peer_removes_edges() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        g.add_peer("c".to_string())
            .expect("test: add_peer c should succeed");
        g.add_edge("a", "b", 0.8)
            .expect("test: add_edge a->b should succeed");
        g.add_edge("c", "a", 0.6)
            .expect("test: add_edge c->a should succeed");
        g.remove_peer("a")
            .expect("test: remove_peer a should succeed");
        assert_eq!(g.edge_count(), 0);
        assert!(g.peers_trusting("b").is_empty());
        assert!(g.peers_trusted_by("c").is_empty());
    }

    // ------- add/remove edges -------

    #[test]
    fn test_add_edge_basic() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        g.add_edge("a", "b", 0.7)
            .expect("test: add_edge a->b should succeed");
        assert_eq!(g.edge_count(), 1);
        assert_eq!(
            g.edge("a", "b")
                .expect("test: edge a->b should exist")
                .weight,
            0.7
        );
    }

    #[test]
    fn test_add_edge_update_existing() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        g.add_edge("a", "b", 0.5)
            .expect("test: add_edge a->b 0.5 should succeed");
        g.add_edge("a", "b", 0.9)
            .expect("test: update edge a->b to 0.9 should succeed");
        assert_eq!(g.edge_count(), 1);
        assert_eq!(
            g.edge("a", "b")
                .expect("test: edge a->b should exist")
                .weight,
            0.9
        );
        assert_eq!(
            g.edge("a", "b")
                .expect("test: edge a->b should exist")
                .interaction_count,
            2
        );
    }

    #[test]
    fn test_add_edge_self_loop() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        let err = g
            .add_edge("a", "a", 0.5)
            .expect_err("test: expected error for self-loop edge");
        assert!(matches!(err, GraphError::SelfLoop(_)));
    }

    #[test]
    fn test_add_edge_invalid_weight_negative() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        let err = g
            .add_edge("a", "b", -0.1)
            .expect_err("test: expected error for negative weight");
        assert!(matches!(err, GraphError::InvalidWeight(_)));
    }

    #[test]
    fn test_add_edge_invalid_weight_above_one() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        let err = g
            .add_edge("a", "b", 1.5)
            .expect_err("test: expected error for weight above 1.0");
        assert!(matches!(err, GraphError::InvalidWeight(_)));
    }

    #[test]
    fn test_add_edge_peer_not_found() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        let err = g
            .add_edge("a", "ghost", 0.5)
            .expect_err("test: expected error for nonexistent target peer");
        assert!(matches!(err, GraphError::PeerNotFound(_)));
    }

    #[test]
    fn test_remove_edge_basic() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        g.add_edge("a", "b", 0.5)
            .expect("test: add_edge a->b should succeed");
        g.remove_edge("a", "b")
            .expect("test: remove_edge a->b should succeed");
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_remove_edge_not_found() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        let err = g
            .remove_edge("a", "b")
            .expect_err("test: expected error removing nonexistent edge");
        assert!(matches!(err, GraphError::EdgeNotFound { .. }));
    }

    // ------- interactions -------

    #[test]
    fn test_positive_interaction_increases_score() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        let initial = g
            .reputation("a")
            .expect("test: reputation for a should exist")
            .direct_score;
        let event = g
            .record_interaction("a", true, 1.0)
            .expect("test: record positive interaction for a should succeed");
        let after = g
            .reputation("a")
            .expect("test: reputation for a should exist")
            .direct_score;
        assert!(after >= initial);
        assert!(matches!(event, ReputationEvent::PositiveInteraction { .. }));
    }

    #[test]
    fn test_negative_interaction_decreases_score() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        let initial = g
            .reputation("a")
            .expect("test: reputation for a should exist")
            .direct_score;
        let event = g
            .record_interaction("a", false, 1.0)
            .expect("test: record negative interaction for a should succeed");
        let after = g
            .reputation("a")
            .expect("test: reputation for a should exist")
            .direct_score;
        assert!(after <= initial);
        assert!(matches!(event, ReputationEvent::NegativeInteraction { .. }));
    }

    #[test]
    fn test_interaction_clamps_score() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        // Many large positive interactions should clamp to 1.0.
        for _ in 0..100 {
            g.record_interaction("a", true, 10.0)
                .expect("test: record large positive interaction should succeed");
        }
        let score = g
            .reputation("a")
            .expect("test: reputation for a should exist")
            .direct_score;
        assert!(score <= 1.0);
        // Many large negative interactions should clamp to 0.0.
        for _ in 0..100 {
            g.record_interaction("a", false, 10.0)
                .expect("test: record large negative interaction should succeed");
        }
        let score = g
            .reputation("a")
            .expect("test: reputation for a should exist")
            .direct_score;
        assert!(score >= 0.0);
    }

    #[test]
    fn test_interaction_peer_not_found() {
        let mut g = make_graph();
        let err = g
            .record_interaction("ghost", true, 0.5)
            .expect_err("test: expected error for nonexistent peer interaction");
        assert!(matches!(err, GraphError::PeerNotFound(_)));
    }

    #[test]
    fn test_interaction_count_increments() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.record_interaction("a", true, 0.5)
            .expect("test: record positive interaction should succeed");
        g.record_interaction("a", false, 0.2)
            .expect("test: record negative interaction should succeed");
        let score = g
            .reputation("a")
            .expect("test: reputation for a should exist");
        assert!(score.confidence > 0.0);
    }

    // ------- propagation -------

    #[test]
    fn test_propagate_trust_basic() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        g.add_edge("a", "b", 1.0)
            .expect("test: add_edge a->b should succeed");
        // Give 'a' a high direct score.
        for _ in 0..50 {
            g.record_interaction("a", true, 1.0)
                .expect("test: record positive interaction for a should succeed");
        }
        let updated = g.propagate_trust();
        assert!(updated > 0);
        let b_score = g
            .reputation("b")
            .expect("test: reputation for b should exist");
        assert!(b_score.propagated_score > 0.0);
    }

    #[test]
    fn test_propagate_trust_no_edges() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        // No edges — propagated scores remain 0.
        g.propagate_trust();
        assert_eq!(
            g.reputation("b")
                .expect("test: reputation for b should exist")
                .propagated_score,
            0.0
        );
    }

    #[test]
    fn test_propagate_trust_depth_limit() {
        // With depth=1, a→b→c: 'c' can still receive propagation FROM 'b'
        // (b has its own direct score), but 'c' should NOT receive propagation
        // from 'a' through 'b'.  We verify by giving only 'a' a high score and
        // keeping 'b' at neutral — 'c' must get some propagated score (from b)
        // but less than 'b' itself.
        let mut g = PeerReputationGraph::new(GraphConfig {
            propagation_depth: 1,
            propagation_damping: 0.5,
            ..Default::default()
        });
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        g.add_peer("c".to_string())
            .expect("test: add_peer c should succeed");
        g.add_edge("a", "b", 1.0)
            .expect("test: add_edge a->b should succeed");
        g.add_edge("b", "c", 1.0)
            .expect("test: add_edge b->c should succeed");
        for _ in 0..50 {
            g.record_interaction("a", true, 1.0)
                .expect("test: record positive interaction for a should succeed");
        }
        g.propagate_trust();
        // 'b' gets propagated score from 'a'.
        assert!(
            g.reputation("b")
                .expect("test: reputation for b should exist")
                .propagated_score
                > 0.0
        );
        // 'c' gets propagated score from 'b' (b has a non-zero direct score).
        // Both get something; depth=1 limits but does not eliminate propagation.
        // Key check: 'b' receives propagation from high-score 'a',
        // while 'c' only receives propagation from neutral-score 'b'.
        let b_prop = g
            .reputation("b")
            .expect("test: reputation for b should exist")
            .propagated_score;
        let c_prop = g
            .reputation("c")
            .expect("test: reputation for c should exist")
            .propagated_score;
        // 'a' has a very high score, so b's propagated score > c's propagated score.
        assert!(b_prop >= c_prop);
    }

    #[test]
    fn test_propagate_trust_multi_hop() {
        // With depth ≥ 2, a→b→c should give c a propagated score.
        let mut g = PeerReputationGraph::new(GraphConfig {
            propagation_depth: 2,
            propagation_damping: 0.5,
            ..Default::default()
        });
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        g.add_peer("c".to_string())
            .expect("test: add_peer c should succeed");
        g.add_edge("a", "b", 1.0)
            .expect("test: add_edge a->b should succeed");
        g.add_edge("b", "c", 1.0)
            .expect("test: add_edge b->c should succeed");
        for _ in 0..50 {
            g.record_interaction("a", true, 1.0)
                .expect("test: record positive interaction for a should succeed");
            g.record_interaction("b", true, 1.0)
                .expect("test: record positive interaction for b should succeed");
        }
        g.propagate_trust();
        assert!(
            g.reputation("c")
                .expect("test: reputation for c should exist")
                .propagated_score
                > 0.0
        );
    }

    #[test]
    fn test_propagate_returns_count() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        g.add_edge("a", "b", 1.0)
            .expect("test: add_edge a->b should succeed");
        for _ in 0..20 {
            g.record_interaction("a", true, 1.0)
                .expect("test: record positive interaction for a should succeed");
        }
        let count = g.propagate_trust();
        assert!(count >= 1);
    }

    // ------- decay -------

    #[test]
    fn test_decay_scores_reduces_score() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        for _ in 0..50 {
            g.record_interaction("a", true, 1.0)
                .expect("test: record positive interaction for a should succeed");
        }
        let before = g
            .reputation("a")
            .expect("test: reputation for a should exist")
            .direct_score;
        g.decay_scores();
        let after = g
            .reputation("a")
            .expect("test: reputation for a after decay should exist")
            .direct_score;
        assert!(after <= before);
    }

    #[test]
    fn test_decay_emits_events() {
        let mut g = PeerReputationGraph::new(GraphConfig {
            trust_decay_factor: 0.5, // aggressive decay to trigger events
            ..Default::default()
        });
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        for _ in 0..50 {
            g.record_interaction("a", true, 1.0)
                .expect("test: record positive interaction for a should succeed");
        }
        let events = g.decay_scores();
        assert!(!events.is_empty());
        assert!(events
            .iter()
            .any(|e| matches!(e, ReputationEvent::ScoreDecayed { .. })));
    }

    #[test]
    fn test_decay_prunes_weak_edges() {
        let mut g = PeerReputationGraph::new(GraphConfig {
            min_edge_weight: 0.5,
            ..Default::default()
        });
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        g.add_edge("a", "b", 0.3)
            .expect("test: add weak edge a->b should succeed"); // below threshold
        g.decay_scores();
        assert_eq!(g.edge_count(), 0, "weak edge should be pruned");
    }

    #[test]
    fn test_decay_keeps_strong_edges() {
        let mut g = PeerReputationGraph::new(GraphConfig {
            min_edge_weight: 0.1,
            ..Default::default()
        });
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        g.add_edge("a", "b", 0.9)
            .expect("test: add strong edge a->b should succeed"); // above threshold
        g.decay_scores();
        assert_eq!(g.edge_count(), 1, "strong edge should remain");
    }

    // ------- reputation query -------

    #[test]
    fn test_reputation_not_found() {
        let g = make_graph();
        let err = g
            .reputation("ghost")
            .expect_err("test: expected error for nonexistent peer reputation");
        assert!(matches!(err, GraphError::PeerNotFound(_)));
    }

    #[test]
    fn test_reputation_combined_score_range() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        for _ in 0..20 {
            g.record_interaction("a", true, 0.5)
                .expect("test: record positive interaction for a should succeed");
        }
        let r = g
            .reputation("a")
            .expect("test: reputation for a should exist");
        assert!((0.0..=1.0).contains(&r.combined_score));
    }

    #[test]
    fn test_reputation_confidence_grows() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        let c0 = g
            .reputation("a")
            .expect("test: reputation for a should exist")
            .confidence;
        for _ in 0..10 {
            g.record_interaction("a", true, 0.5)
                .expect("test: record positive interaction should succeed");
        }
        let c10 = g
            .reputation("a")
            .expect("test: reputation for a should exist")
            .confidence;
        assert!(c10 > c0);
    }

    #[test]
    fn test_reputation_confidence_caps_at_one() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        for _ in 0..200 {
            g.record_interaction("a", true, 0.3)
                .expect("test: record positive interaction should succeed");
        }
        let c = g
            .reputation("a")
            .expect("test: reputation for a should exist")
            .confidence;
        assert!(c <= 1.0);
    }

    // ------- top_peers -------

    #[test]
    fn test_top_peers_empty() {
        let g = make_graph();
        assert!(g.top_peers(5).is_empty());
    }

    #[test]
    fn test_top_peers_order() {
        let mut g = make_graph();
        for name in &["a", "b", "c"] {
            g.add_peer(name.to_string())
                .expect("test: add_peer should succeed");
        }
        for _ in 0..20 {
            g.record_interaction("a", true, 1.0)
                .expect("test: record positive interaction for a should succeed");
        }
        let top = g.top_peers(3);
        assert!(!top.is_empty());
        assert_eq!(top[0].peer_id, "a");
    }

    #[test]
    fn test_top_peers_limit() {
        let mut g = make_graph();
        for i in 0..10 {
            g.add_peer(format!("peer{i}"))
                .expect("test: add_peer should succeed");
        }
        assert!(g.top_peers(5).len() <= 5);
    }

    #[test]
    fn test_top_peers_fewer_than_n() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        let top = g.top_peers(10);
        assert_eq!(top.len(), 1);
    }

    // ------- peers_trusting / peers_trusted_by -------

    #[test]
    fn test_peers_trusting() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        g.add_peer("c".to_string())
            .expect("test: add_peer c should succeed");
        g.add_edge("b", "a", 0.5)
            .expect("test: add_edge b->a should succeed");
        g.add_edge("c", "a", 0.7)
            .expect("test: add_edge c->a should succeed");
        let trusting = g.peers_trusting("a");
        assert!(trusting.contains(&"b".to_string()));
        assert!(trusting.contains(&"c".to_string()));
    }

    #[test]
    fn test_peers_trusted_by() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        g.add_peer("c".to_string())
            .expect("test: add_peer c should succeed");
        g.add_edge("a", "b", 0.5)
            .expect("test: add_edge a->b should succeed");
        g.add_edge("a", "c", 0.7)
            .expect("test: add_edge a->c should succeed");
        let trusted = g.peers_trusted_by("a");
        assert!(trusted.contains(&"b".to_string()));
        assert!(trusted.contains(&"c".to_string()));
    }

    #[test]
    fn test_peers_trusting_empty() {
        let mut g = make_graph();
        g.add_peer("loner".to_string())
            .expect("test: add_peer loner should succeed");
        assert!(g.peers_trusting("loner").is_empty());
    }

    // ------- stats -------

    #[test]
    fn test_stats_empty() {
        let g = make_graph();
        let s = g.stats();
        assert_eq!(s.peer_count, 0);
        assert_eq!(s.edge_count, 0);
        assert_eq!(s.avg_out_degree, 0.0);
        assert_eq!(s.isolated_peers, 0);
    }

    #[test]
    fn test_stats_peer_and_edge_count() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        g.add_edge("a", "b", 0.5)
            .expect("test: add_edge a->b should succeed");
        let s = g.stats();
        assert_eq!(s.peer_count, 2);
        assert_eq!(s.edge_count, 1);
    }

    #[test]
    fn test_stats_isolated_peers() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        // Both start with direct_score = 0.5, so combined ≥ 0.1.
        // Set b's score very low to make it prunable.
        for _ in 0..50 {
            g.record_interaction("b", false, 1.0)
                .expect("test: record negative interaction for b should succeed");
        }
        let s = g.stats();
        // Both peers have no edges.
        assert_eq!(s.isolated_peers, 2);
    }

    #[test]
    fn test_stats_top_peers_max_five() {
        let mut g = make_graph();
        for i in 0..10 {
            g.add_peer(format!("p{i}"))
                .expect("test: add_peer should succeed");
        }
        let s = g.stats();
        assert!(s.top_peers.len() <= 5);
    }

    // ------- prune_isolated -------

    #[test]
    fn test_prune_isolated_removes_low_score_no_edge() {
        let mut g = PeerReputationGraph::new(GraphConfig {
            trust_decay_factor: 0.5,
            ..Default::default()
        });
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        // Give 'b' a very low score.
        for _ in 0..50 {
            g.record_interaction("b", false, 1.0)
                .expect("test: record negative interaction for b should succeed");
        }
        // 'a' stays at neutral 0.5 — above the 0.1 threshold → NOT pruned.
        let pruned = g.prune_isolated();
        assert!(pruned >= 1);
        assert!(!g.contains_peer("b"));
    }

    #[test]
    fn test_prune_isolated_keeps_high_score() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        for _ in 0..50 {
            g.record_interaction("a", true, 1.0)
                .expect("test: record positive interaction for a should succeed");
        }
        let pruned = g.prune_isolated();
        // 'a' has a high score, should not be pruned.
        assert_eq!(pruned, 0);
        assert!(g.contains_peer("a"));
    }

    #[test]
    fn test_prune_isolated_keeps_connected_peers() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        g.add_edge("a", "b", 0.8)
            .expect("test: add_edge a->b should succeed");
        // Even if 'a' has a low score, it has an edge so it should NOT be pruned.
        for _ in 0..100 {
            g.record_interaction("a", false, 1.0)
                .expect("test: record negative interaction for a should succeed");
        }
        let pruned = g.prune_isolated();
        assert_eq!(pruned, 0);
        assert!(g.contains_peer("a"));
    }

    // ------- error types -------

    #[test]
    fn test_graph_error_display_peer_not_found() {
        let err = GraphError::PeerNotFound("x".to_string());
        assert!(err.to_string().contains("x"));
    }

    #[test]
    fn test_graph_error_display_edge_not_found() {
        let err = GraphError::EdgeNotFound {
            from: "a".to_string(),
            to: "b".to_string(),
        };
        let s = err.to_string();
        assert!(s.contains("a"));
        assert!(s.contains("b"));
    }

    #[test]
    fn test_graph_error_display_self_loop() {
        let err = GraphError::SelfLoop("a".to_string());
        assert!(err.to_string().contains("a"));
    }

    #[test]
    fn test_graph_error_display_invalid_weight() {
        let err = GraphError::InvalidWeight(2.5);
        assert!(err.to_string().contains("2.5"));
    }

    #[test]
    fn test_graph_error_max_peers() {
        assert_eq!(
            GraphError::MaxPeersExceeded.to_string(),
            "maximum peer count exceeded"
        );
    }

    // ------- edge interaction count -------

    #[test]
    fn test_edge_interaction_count() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        g.add_edge("a", "b", 0.5)
            .expect("test: add_edge a->b 0.5 should succeed");
        g.add_edge("a", "b", 0.6)
            .expect("test: update edge a->b to 0.6 should succeed");
        g.add_edge("a", "b", 0.7)
            .expect("test: update edge a->b to 0.7 should succeed");
        assert_eq!(
            g.edge("a", "b")
                .expect("test: edge a->b should exist")
                .interaction_count,
            3
        );
    }

    // ------- graph config defaults -------

    #[test]
    fn test_graph_config_defaults() {
        let cfg = GraphConfig::default();
        assert!((cfg.trust_decay_factor - 0.99).abs() < 1e-9);
        assert_eq!(cfg.propagation_depth, 3);
        assert!((cfg.propagation_damping - 0.5).abs() < 1e-9);
        assert_eq!(cfg.max_peers, 10_000);
    }

    // ------- xorshift64 -------

    #[test]
    fn test_xorshift64_produces_nonzero() {
        let mut state = 12345u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
        assert_ne!(state, 12345);
    }

    #[test]
    fn test_xorshift64_sequence_unique() {
        let mut state = 0xdeadbeef_u64;
        let a = xorshift64(&mut state);
        let b = xorshift64(&mut state);
        let c = xorshift64(&mut state);
        // Very unlikely to repeat with a good PRNG.
        assert_ne!(a, b);
        assert_ne!(b, c);
    }

    // ------- combined score formula -------

    #[test]
    fn test_combined_score_weights() {
        let cfg = GraphConfig {
            combination_weight_direct: 1.0,
            combination_weight_propagated: 0.0,
            ..Default::default()
        };
        let mut g = PeerReputationGraph::new(cfg);
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        for _ in 0..50 {
            g.record_interaction("a", true, 1.0)
                .expect("test: record positive interaction for a should succeed");
        }
        g.propagate_trust();
        let r = g
            .reputation("a")
            .expect("test: reputation for a should exist");
        // combined should equal direct when propagated weight = 0.
        assert!((r.combined_score - r.direct_score).abs() < 1e-9);
    }

    // ------- percentile -------

    #[test]
    fn test_percentile_single_peer() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        let r = g
            .reputation("a")
            .expect("test: reputation for a should exist");
        assert_eq!(r.percentile, 0.5);
    }

    #[test]
    fn test_percentile_ordering() {
        let mut g = make_graph();
        g.add_peer("low".to_string())
            .expect("test: add_peer low should succeed");
        g.add_peer("high".to_string())
            .expect("test: add_peer high should succeed");
        // Make 'high' have a much better score.
        for _ in 0..50 {
            g.record_interaction("high", true, 1.0)
                .expect("test: record positive interaction for high should succeed");
            g.record_interaction("low", false, 1.0)
                .expect("test: record negative interaction for low should succeed");
        }
        let low_pct = g
            .reputation("low")
            .expect("test: reputation for low should exist")
            .percentile;
        let high_pct = g
            .reputation("high")
            .expect("test: reputation for high should exist")
            .percentile;
        assert!(high_pct > low_pct);
    }

    // ------- all_edges accessor -------

    #[test]
    fn test_all_edges() {
        let mut g = make_graph();
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        g.add_peer("c".to_string())
            .expect("test: add_peer c should succeed");
        g.add_edge("a", "b", 0.5)
            .expect("test: add_edge a->b should succeed");
        g.add_edge("b", "c", 0.3)
            .expect("test: add_edge b->c should succeed");
        let edges = g.all_edges();
        assert_eq!(edges.len(), 2);
    }

    // ------- propagation damping -------

    #[test]
    fn test_propagation_damping_reduces_with_hops() {
        // Verify damping: a SINGLE high-score source 'a' propagates to 'b' (1 hop)
        // more than it alone contributes to 'd' (3 hops) when b/c/d have near-zero
        // direct scores.  We give b/c/d many negative interactions to suppress their
        // own propagation contributions.
        let cfg = GraphConfig {
            propagation_depth: 3,
            propagation_damping: 0.5,
            ..Default::default()
        };
        let mut g = PeerReputationGraph::new(cfg);
        g.add_peer("a".to_string())
            .expect("test: add_peer a should succeed");
        g.add_peer("b".to_string())
            .expect("test: add_peer b should succeed");
        g.add_peer("c".to_string())
            .expect("test: add_peer c should succeed");
        g.add_peer("d".to_string())
            .expect("test: add_peer d should succeed");
        g.add_edge("a", "b", 1.0)
            .expect("test: add_edge a->b should succeed");
        g.add_edge("b", "c", 1.0)
            .expect("test: add_edge b->c should succeed");
        g.add_edge("c", "d", 1.0)
            .expect("test: add_edge c->d should succeed");
        // Boost 'a' to near 1.0.
        for _ in 0..100 {
            g.record_interaction("a", true, 1.0)
                .expect("test: record positive interaction for a should succeed");
        }
        // Drive b/c/d to near 0.0 so they contribute almost nothing.
        for _ in 0..100 {
            g.record_interaction("b", false, 1.0)
                .expect("test: record negative interaction for b should succeed");
            g.record_interaction("c", false, 1.0)
                .expect("test: record negative interaction for c should succeed");
            g.record_interaction("d", false, 1.0)
                .expect("test: record negative interaction for d should succeed");
        }
        g.propagate_trust();
        // From 'a' alone: b_contribution = a_direct * 0.5^1, d_contribution = a_direct * 0.5^3
        // (b/c/d near-zero means their BFS contributes negligibly).
        let b_prop = g
            .reputation("b")
            .expect("test: reputation for b should exist")
            .propagated_score;
        let d_prop = g
            .reputation("d")
            .expect("test: reputation for d should exist")
            .propagated_score;
        assert!(b_prop > d_prop, "b_prop={b_prop}, d_prop={d_prop}");
    }
}
