//! Temporal Knowledge Graph — tracks how facts and relationships evolve over time.
//!
//! # Overview
//!
//! A [`TemporalKnowledgeGraph`] stores nodes and edges along with a timeline of events,
//! allowing queries that ask "what was true at time T?" as well as full history queries.
//!
//! # Quick start
//!
//! ```rust
//! use ipfrs_tensorlogic::temporal_knowledge_graph::{
//!     TemporalKnowledgeGraph, TkgQuery,
//! };
//!
//! let mut g = TemporalKnowledgeGraph::new();
//! let alice = g.add_node("Alice".to_string(), 0).expect("example: should succeed in docs");
//! let bob   = g.add_node("Bob".to_string(),   0).expect("example: should succeed in docs");
//! let edge  = g.add_edge(alice, bob, "knows".to_string(), 1.0, 0).expect("example: should succeed in docs");
//!
//! let snap = g.snapshot_at(10);
//! assert_eq!(snap.nodes.len(), 2);
//! assert_eq!(snap.edges.len(), 1);
//! ```

use std::collections::{BTreeMap, HashMap, VecDeque};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// XorShift64 PRNG — deterministic, no external dependencies.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// FNV-1a 64-bit hash.
#[inline]
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

/// Generate a 16-byte identifier seeded from `seed` mixed with a counter.
fn gen_id(seed: &mut u64, salt: u64) -> [u8; 16] {
    let a = xorshift64(seed);
    let b = fnv1a_64(&(a ^ salt).to_le_bytes());
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&a.to_le_bytes());
    out[8..].copy_from_slice(&b.to_le_bytes());
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Public newtypes / identifiers
// ─────────────────────────────────────────────────────────────────────────────

/// Unique identifier for a node (16 opaque bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub [u8; 16]);

/// Unique identifier for an edge (16 opaque bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId(pub [u8; 16]);

// ─────────────────────────────────────────────────────────────────────────────
// Core data types
// ─────────────────────────────────────────────────────────────────────────────

/// A node in the temporal knowledge graph.
#[derive(Debug, Clone)]
pub struct TkgNode {
    pub id: NodeId,
    pub label: String,
    pub properties: HashMap<String, String>,
    pub created_at: u64,
    pub deleted_at: Option<u64>,
}

impl TkgNode {
    /// Returns `true` if the node exists (has not been deleted) at time `t`.
    pub fn alive_at(&self, t: u64) -> bool {
        self.created_at <= t && self.deleted_at.is_none_or(|d| t < d)
    }
}

/// A directed edge in the temporal knowledge graph.
#[derive(Debug, Clone)]
pub struct TkgEdge {
    pub id: EdgeId,
    pub src: NodeId,
    pub dst: NodeId,
    pub relation: String,
    pub weight: f64,
    pub valid_from: u64,
    pub valid_until: Option<u64>,
}

impl TkgEdge {
    /// Returns `true` if the edge is valid at time `t`.
    pub fn valid_at(&self, t: u64) -> bool {
        self.valid_from <= t && self.valid_until.is_none_or(|v| t < v)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Events
// ─────────────────────────────────────────────────────────────────────────────

/// An event recorded in the graph timeline.
#[derive(Debug, Clone)]
pub enum TkgEvent {
    NodeAdded(NodeId),
    NodeDeleted(NodeId),
    EdgeAdded(EdgeId),
    EdgeDeleted(EdgeId),
    PropertyChanged {
        node_id: NodeId,
        key: String,
        old_val: Option<String>,
        new_val: String,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Queries
// ─────────────────────────────────────────────────────────────────────────────

/// Query types supported by [`TemporalKnowledgeGraph`].
#[derive(Debug, Clone)]
pub enum TkgQuery {
    /// All nodes alive at timestamp `t`.
    NodesAt(u64),
    /// All edges valid at timestamp `t`.
    EdgesAt(u64),
    /// Full event history for a node.
    NodeHistory(NodeId),
    /// All edges between `src` and `dst` in the time window `[from_t, to_t]`.
    EdgesBetween {
        src: NodeId,
        dst: NodeId,
        from_t: u64,
        to_t: u64,
    },
    /// Value of a node property at a specific time.
    PropertyAt {
        node_id: NodeId,
        key: String,
        at_t: u64,
    },
}

/// The result of a [`TkgQuery`].
#[derive(Debug, Clone)]
pub enum TkgQueryResult {
    Nodes(Vec<TkgNode>),
    Edges(Vec<TkgEdge>),
    Events(Vec<(u64, TkgEvent)>),
    Property(Option<String>),
    Empty,
}

// ─────────────────────────────────────────────────────────────────────────────
// Snapshot
// ─────────────────────────────────────────────────────────────────────────────

/// A point-in-time snapshot of the graph.
#[derive(Debug, Clone)]
pub struct TkgSnapshot {
    pub at: u64,
    pub nodes: Vec<TkgNode>,
    pub edges: Vec<TkgEdge>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics for the graph.
#[derive(Debug, Clone, Default)]
pub struct TkgGraphStats {
    pub total_nodes: usize,
    pub live_nodes: usize,
    pub total_edges: usize,
    pub live_edges: usize,
    pub total_events: usize,
    pub first_timestamp: Option<u64>,
    pub last_timestamp: Option<u64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Merge policy
// ─────────────────────────────────────────────────────────────────────────────

/// Conflict-resolution policy when merging two graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TkgMergePolicy {
    /// Keep this graph's data on conflict.
    KeepMine,
    /// Keep the other graph's data on conflict.
    KeepOther,
    /// Include data from both graphs (union).
    UnionAll,
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by temporal knowledge graph operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TkgError {
    #[error("node {0:?} not found")]
    NodeNotFound(NodeId),
    #[error("edge {0:?} not found")]
    EdgeNotFound(EdgeId),
    #[error("node {0:?} already deleted")]
    NodeAlreadyDeleted(NodeId),
    #[error("edge {0:?} already deleted")]
    EdgeAlreadyDeleted(EdgeId),
    #[error("source node {0:?} does not exist at the given timestamp")]
    SourceNodeInvalid(NodeId),
    #[error("destination node {0:?} does not exist at the given timestamp")]
    DestNodeInvalid(NodeId),
}

// ─────────────────────────────────────────────────────────────────────────────
// Main struct
// ─────────────────────────────────────────────────────────────────────────────

/// A temporal knowledge graph that records how facts and relationships change.
///
/// All mutations are timestamped and appended to an internal timeline, enabling
/// point-in-time queries and full history replay.
#[derive(Debug, Clone)]
pub struct TemporalKnowledgeGraph {
    nodes: HashMap<NodeId, TkgNode>,
    edges: HashMap<EdgeId, TkgEdge>,
    /// timestamp → ordered list of events at that tick.
    timeline: BTreeMap<u64, Vec<TkgEvent>>,
    /// Monotonically increasing PRNG state for ID generation.
    rng_state: u64,
    /// Counter incremented per ID generation to guarantee uniqueness.
    id_counter: u64,
}

impl Default for TemporalKnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalKnowledgeGraph {
    /// Create a new, empty temporal knowledge graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            timeline: BTreeMap::new(),
            rng_state: 0xcafe_babe_dead_beef,
            id_counter: 0,
        }
    }

    // ─────────────────────────────────────────────── ID generation ─────────

    fn next_node_id(&mut self) -> NodeId {
        self.id_counter = self.id_counter.wrapping_add(1);
        NodeId(gen_id(&mut self.rng_state, self.id_counter))
    }

    fn next_edge_id(&mut self) -> EdgeId {
        self.id_counter = self.id_counter.wrapping_add(1);
        let raw = gen_id(&mut self.rng_state, self.id_counter ^ 0xFFFF_FFFF);
        EdgeId(raw)
    }

    // ───────────────────────────────────────── Timeline helpers ─────────────

    fn push_event(&mut self, ts: u64, event: TkgEvent) {
        self.timeline.entry(ts).or_default().push(event);
    }

    // ─────────────────────────────────────────────── Mutations ──────────────

    /// Add a node with the given `label` at timestamp `ts`.
    ///
    /// Returns the new [`NodeId`] on success.
    pub fn add_node(&mut self, label: String, ts: u64) -> Result<NodeId, TkgError> {
        let id = self.next_node_id();
        let node = TkgNode {
            id,
            label,
            properties: HashMap::new(),
            created_at: ts,
            deleted_at: None,
        };
        self.nodes.insert(id, node);
        self.push_event(ts, TkgEvent::NodeAdded(id));
        Ok(id)
    }

    /// Soft-delete node `id` at timestamp `ts`.
    pub fn delete_node(&mut self, id: NodeId, ts: u64) -> Result<(), TkgError> {
        let node = self.nodes.get_mut(&id).ok_or(TkgError::NodeNotFound(id))?;
        if node.deleted_at.is_some() {
            return Err(TkgError::NodeAlreadyDeleted(id));
        }
        node.deleted_at = Some(ts);
        self.push_event(ts, TkgEvent::NodeDeleted(id));
        Ok(())
    }

    /// Add an edge from `src` to `dst` with the given `relation`, `weight`, and
    /// start timestamp `valid_from`.
    ///
    /// Both endpoint nodes must exist (not deleted) at `valid_from`.
    pub fn add_edge(
        &mut self,
        src: NodeId,
        dst: NodeId,
        relation: String,
        weight: f64,
        valid_from: u64,
    ) -> Result<EdgeId, TkgError> {
        // Validate source node.
        {
            let src_node = self
                .nodes
                .get(&src)
                .ok_or(TkgError::SourceNodeInvalid(src))?;
            if !src_node.alive_at(valid_from) {
                return Err(TkgError::SourceNodeInvalid(src));
            }
        }
        // Validate destination node.
        {
            let dst_node = self.nodes.get(&dst).ok_or(TkgError::DestNodeInvalid(dst))?;
            if !dst_node.alive_at(valid_from) {
                return Err(TkgError::DestNodeInvalid(dst));
            }
        }

        let id = self.next_edge_id();
        let edge = TkgEdge {
            id,
            src,
            dst,
            relation,
            weight,
            valid_from,
            valid_until: None,
        };
        self.edges.insert(id, edge);
        self.push_event(valid_from, TkgEvent::EdgeAdded(id));
        Ok(id)
    }

    /// Soft-delete edge `id` at timestamp `ts`.
    pub fn delete_edge(&mut self, id: EdgeId, ts: u64) -> Result<(), TkgError> {
        let edge = self.edges.get_mut(&id).ok_or(TkgError::EdgeNotFound(id))?;
        if edge.valid_until.is_some() {
            return Err(TkgError::EdgeAlreadyDeleted(id));
        }
        edge.valid_until = Some(ts);
        self.push_event(ts, TkgEvent::EdgeDeleted(id));
        Ok(())
    }

    /// Set (or overwrite) a property on node `id` at timestamp `ts`.
    pub fn set_property(
        &mut self,
        id: NodeId,
        key: String,
        value: String,
        ts: u64,
    ) -> Result<(), TkgError> {
        let node = self.nodes.get_mut(&id).ok_or(TkgError::NodeNotFound(id))?;
        let old_val = node.properties.get(&key).cloned();
        node.properties.insert(key.clone(), value.clone());
        self.push_event(
            ts,
            TkgEvent::PropertyChanged {
                node_id: id,
                key,
                old_val,
                new_val: value,
            },
        );
        Ok(())
    }

    // ─────────────────────────────────────────────────── Queries ────────────

    /// Execute a [`TkgQuery`] and return a [`TkgQueryResult`].
    pub fn query(&self, q: TkgQuery) -> TkgQueryResult {
        match q {
            TkgQuery::NodesAt(t) => {
                let nodes = self
                    .nodes
                    .values()
                    .filter(|n| n.alive_at(t))
                    .cloned()
                    .collect();
                TkgQueryResult::Nodes(nodes)
            }

            TkgQuery::EdgesAt(t) => {
                let edges = self
                    .edges
                    .values()
                    .filter(|e| e.valid_at(t))
                    .cloned()
                    .collect();
                TkgQueryResult::Edges(edges)
            }

            TkgQuery::NodeHistory(node_id) => {
                let mut events: Vec<(u64, TkgEvent)> = Vec::new();
                for (&ts, evs) in &self.timeline {
                    for ev in evs {
                        let relevant = match ev {
                            TkgEvent::NodeAdded(id) | TkgEvent::NodeDeleted(id) => *id == node_id,
                            TkgEvent::PropertyChanged { node_id: nid, .. } => *nid == node_id,
                            _ => false,
                        };
                        if relevant {
                            events.push((ts, ev.clone()));
                        }
                    }
                }
                TkgQueryResult::Events(events)
            }

            TkgQuery::EdgesBetween {
                src,
                dst,
                from_t,
                to_t,
            } => {
                let edges = self
                    .edges
                    .values()
                    .filter(|e| {
                        e.src == src
                            && e.dst == dst
                            && e.valid_from <= to_t
                            && e.valid_until.is_none_or(|u| u >= from_t)
                    })
                    .cloned()
                    .collect();
                TkgQueryResult::Edges(edges)
            }

            TkgQuery::PropertyAt { node_id, key, at_t } => {
                // Replay property-change events up to `at_t`.
                let mut current: Option<String> = None;
                for (&ts, evs) in &self.timeline {
                    if ts > at_t {
                        break;
                    }
                    for ev in evs {
                        if let TkgEvent::PropertyChanged {
                            node_id: nid,
                            key: k,
                            new_val,
                            ..
                        } = ev
                        {
                            if *nid == node_id && k == &key {
                                current = Some(new_val.clone());
                            }
                        }
                    }
                }
                TkgQueryResult::Property(current)
            }
        }
    }

    // ─────────────────────────────────────── Snapshot ───────────────────────

    /// Return a point-in-time snapshot: all nodes and edges valid at `t`.
    pub fn snapshot_at(&self, t: u64) -> TkgSnapshot {
        let nodes: Vec<TkgNode> = self
            .nodes
            .values()
            .filter(|n| n.alive_at(t))
            .cloned()
            .collect();
        let edges: Vec<TkgEdge> = self
            .edges
            .values()
            .filter(|e| e.valid_at(t))
            .cloned()
            .collect();
        TkgSnapshot {
            at: t,
            nodes,
            edges,
        }
    }

    // ─────────────────────────────────── Temporal path ──────────────────────

    /// BFS to find a path from `src` to `dst` using only edges valid at `at_t`.
    ///
    /// Returns `None` if no path exists or if either endpoint does not exist at `at_t`.
    pub fn temporal_path(&self, src: NodeId, dst: NodeId, at_t: u64) -> Option<Vec<NodeId>> {
        // Quick sanity checks.
        let src_node = self.nodes.get(&src)?;
        if !src_node.alive_at(at_t) {
            return None;
        }
        let dst_node = self.nodes.get(&dst)?;
        if !dst_node.alive_at(at_t) {
            return None;
        }

        if src == dst {
            return Some(vec![src]);
        }

        // Build adjacency list for nodes alive at `at_t`.
        let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for edge in self.edges.values() {
            if edge.valid_at(at_t) {
                adj.entry(edge.src).or_default().push(edge.dst);
            }
        }

        // BFS.
        let mut visited: HashMap<NodeId, NodeId> = HashMap::new(); // child → parent
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        queue.push_back(src);
        visited.insert(src, src); // sentinel: src's parent is itself

        'bfs: while let Some(cur) = queue.pop_front() {
            if let Some(neighbours) = adj.get(&cur) {
                for &next in neighbours {
                    if visited.contains_key(&next) {
                        continue;
                    }
                    visited.insert(next, cur);
                    if next == dst {
                        break 'bfs;
                    }
                    queue.push_back(next);
                }
            }
        }

        if !visited.contains_key(&dst) {
            return None;
        }

        // Reconstruct path.
        let mut path = Vec::new();
        let mut cur = dst;
        loop {
            path.push(cur);
            let parent = *visited.get(&cur)?;
            if parent == cur {
                break; // reached src sentinel
            }
            cur = parent;
        }
        path.reverse();
        Some(path)
    }

    // ─────────────────────────────────────────── Merge ──────────────────────

    /// Merge `other` into `self` according to `policy`.
    ///
    /// - [`TkgMergePolicy::KeepMine`]  — on ID collision keep `self`'s data.
    /// - [`TkgMergePolicy::KeepOther`] — on ID collision keep `other`'s data.
    /// - [`TkgMergePolicy::UnionAll`]  — keep all data; on ID collision prefer `other`.
    pub fn merge_graphs(&mut self, other: &Self, policy: TkgMergePolicy) {
        // Merge nodes.
        for (&id, other_node) in &other.nodes {
            match policy {
                TkgMergePolicy::KeepMine => {
                    self.nodes.entry(id).or_insert_with(|| other_node.clone());
                }
                TkgMergePolicy::KeepOther => {
                    self.nodes.insert(id, other_node.clone());
                }
                TkgMergePolicy::UnionAll => {
                    let entry = self.nodes.entry(id).or_insert_with(|| other_node.clone());
                    // Union properties.
                    for (k, v) in &other_node.properties {
                        entry
                            .properties
                            .entry(k.clone())
                            .or_insert_with(|| v.clone());
                    }
                    // Extend deleted_at to the later timestamp if both are deleted.
                    if let (Some(mine_d), Some(other_d)) = (entry.deleted_at, other_node.deleted_at)
                    {
                        entry.deleted_at = Some(mine_d.max(other_d));
                    } else if other_node.deleted_at.is_some() {
                        entry.deleted_at = other_node.deleted_at;
                    }
                }
            }
        }

        // Merge edges.
        for (&id, other_edge) in &other.edges {
            match policy {
                TkgMergePolicy::KeepMine => {
                    self.edges.entry(id).or_insert_with(|| other_edge.clone());
                }
                TkgMergePolicy::KeepOther | TkgMergePolicy::UnionAll => {
                    self.edges.insert(id, other_edge.clone());
                }
            }
        }

        // Merge timeline.
        for (&ts, other_evs) in &other.timeline {
            let entry = self.timeline.entry(ts).or_default();
            for ev in other_evs {
                entry.push(ev.clone());
            }
        }
    }

    // ─────────────────────────────────────────── Statistics ─────────────────

    /// Compute aggregate statistics for the graph.
    pub fn stats(&self) -> TkgGraphStats {
        let total_events: usize = self.timeline.values().map(|v| v.len()).sum();
        TkgGraphStats {
            total_nodes: self.nodes.len(),
            live_nodes: self
                .nodes
                .values()
                .filter(|n| n.deleted_at.is_none())
                .count(),
            total_edges: self.edges.len(),
            live_edges: self
                .edges
                .values()
                .filter(|e| e.valid_until.is_none())
                .count(),
            total_events,
            first_timestamp: self.timeline.keys().next().copied(),
            last_timestamp: self.timeline.keys().next_back().copied(),
        }
    }

    // ─────────────────────────────────────────── Accessors ──────────────────

    /// Return a reference to the node with the given id, if it exists.
    pub fn get_node(&self, id: NodeId) -> Option<&TkgNode> {
        self.nodes.get(&id)
    }

    /// Return a reference to the edge with the given id, if it exists.
    pub fn get_edge(&self, id: EdgeId) -> Option<&TkgEdge> {
        self.edges.get(&id)
    }

    /// Return all events recorded at exactly timestamp `ts`.
    pub fn events_at(&self, ts: u64) -> &[TkgEvent] {
        self.timeline.get(&ts).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Iterate over the timeline in chronological order.
    pub fn timeline_iter(&self) -> impl Iterator<Item = (u64, &[TkgEvent])> {
        self.timeline.iter().map(|(&ts, evs)| (ts, evs.as_slice()))
    }

    /// Number of nodes (including deleted).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges (including expired).
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_graph() -> (TemporalKnowledgeGraph, NodeId, NodeId) {
        let mut g = TemporalKnowledgeGraph::new();
        let a = g
            .add_node("Alice".to_string(), 0)
            .expect("test setup: add_node Alice should succeed");
        let b = g
            .add_node("Bob".to_string(), 0)
            .expect("test setup: add_node Bob should succeed");
        (g, a, b)
    }

    // ── ID & PRNG ─────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_nonzero() {
        let mut state = 1u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 1);
    }

    #[test]
    fn test_xorshift64_deterministic() {
        let mut s1 = 42u64;
        let mut s2 = 42u64;
        assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
    }

    #[test]
    fn test_fnv1a_empty() {
        let h = fnv1a_64(&[]);
        assert_eq!(h, 14_695_981_039_346_656_037u64);
    }

    #[test]
    fn test_fnv1a_deterministic() {
        let a = fnv1a_64(b"hello");
        let b = fnv1a_64(b"hello");
        assert_eq!(a, b);
    }

    #[test]
    fn test_fnv1a_different_inputs() {
        assert_ne!(fnv1a_64(b"foo"), fnv1a_64(b"bar"));
    }

    #[test]
    fn test_gen_id_unique() {
        let mut state = 0x1234_5678u64;
        let id1 = gen_id(&mut state, 1);
        let id2 = gen_id(&mut state, 2);
        assert_ne!(id1, id2);
    }

    // ── Node operations ───────────────────────────────────────────────────

    #[test]
    fn test_add_node_returns_unique_ids() {
        let mut g = TemporalKnowledgeGraph::new();
        let a = g
            .add_node("A".to_string(), 0)
            .expect("test setup: add_node A should succeed");
        let b = g
            .add_node("B".to_string(), 0)
            .expect("test setup: add_node B should succeed");
        assert_ne!(a, b);
    }

    #[test]
    fn test_add_node_stored() {
        let mut g = TemporalKnowledgeGraph::new();
        let id = g
            .add_node("X".to_string(), 5)
            .expect("test setup: add_node X should succeed");
        let node = g
            .get_node(id)
            .expect("test setup: node must exist after add_node succeeded");
        assert_eq!(node.label, "X");
        assert_eq!(node.created_at, 5);
        assert!(node.deleted_at.is_none());
    }

    #[test]
    fn test_delete_node_sets_deleted_at() {
        let (mut g, a, _) = make_graph();
        g.delete_node(a, 10)
            .expect("test setup: delete_node should succeed");
        let node = g
            .get_node(a)
            .expect("test setup: node must exist after add_node succeeded");
        assert_eq!(node.deleted_at, Some(10));
    }

    #[test]
    fn test_delete_node_not_found() {
        let mut g = TemporalKnowledgeGraph::new();
        let fake = NodeId([0u8; 16]);
        assert!(matches!(
            g.delete_node(fake, 0),
            Err(TkgError::NodeNotFound(_))
        ));
    }

    #[test]
    fn test_delete_node_twice_errors() {
        let (mut g, a, _) = make_graph();
        g.delete_node(a, 5)
            .expect("test setup: first deletion of node a must succeed");
        assert!(matches!(
            g.delete_node(a, 10),
            Err(TkgError::NodeAlreadyDeleted(_))
        ));
    }

    #[test]
    fn test_node_alive_at() {
        let node = TkgNode {
            id: NodeId([0; 16]),
            label: "x".to_string(),
            properties: HashMap::new(),
            created_at: 5,
            deleted_at: Some(15),
        };
        assert!(!node.alive_at(4));
        assert!(node.alive_at(5));
        assert!(node.alive_at(14));
        assert!(!node.alive_at(15));
    }

    #[test]
    fn test_node_alive_no_deletion() {
        let node = TkgNode {
            id: NodeId([0; 16]),
            label: "x".to_string(),
            properties: HashMap::new(),
            created_at: 0,
            deleted_at: None,
        };
        assert!(node.alive_at(u64::MAX));
    }

    // ── Edge operations ───────────────────────────────────────────────────

    #[test]
    fn test_add_edge_stored() {
        let (mut g, a, b) = make_graph();
        let eid = g
            .add_edge(a, b, "knows".to_string(), 0.9, 0)
            .expect("test setup: add edge a→b with relation 'knows'");
        let edge = g
            .get_edge(eid)
            .expect("test setup: get edge by id after successful add");
        assert_eq!(edge.src, a);
        assert_eq!(edge.dst, b);
        assert_eq!(edge.relation, "knows");
        assert!((edge.weight - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_add_edge_invalid_src() {
        let mut g = TemporalKnowledgeGraph::new();
        let fake = NodeId([0u8; 16]);
        let b = g
            .add_node("B".to_string(), 0)
            .expect("test setup: add node B to graph");
        assert!(matches!(
            g.add_edge(fake, b, "r".to_string(), 1.0, 0),
            Err(TkgError::SourceNodeInvalid(_))
        ));
    }

    #[test]
    fn test_add_edge_invalid_dst() {
        let mut g = TemporalKnowledgeGraph::new();
        let a = g
            .add_node("A".to_string(), 0)
            .expect("test setup: add node A to graph");
        let fake = NodeId([0u8; 16]);
        assert!(matches!(
            g.add_edge(a, fake, "r".to_string(), 1.0, 0),
            Err(TkgError::DestNodeInvalid(_))
        ));
    }

    #[test]
    fn test_add_edge_on_deleted_src_fails() {
        let (mut g, a, b) = make_graph();
        g.delete_node(a, 5)
            .expect("test setup: delete node a before attempting edge add");
        assert!(matches!(
            g.add_edge(a, b, "r".to_string(), 1.0, 10),
            Err(TkgError::SourceNodeInvalid(_))
        ));
    }

    #[test]
    fn test_delete_edge_sets_valid_until() {
        let (mut g, a, b) = make_graph();
        let eid = g
            .add_edge(a, b, "r".to_string(), 1.0, 0)
            .expect("test setup: add edge a→b with relation 'r'");
        g.delete_edge(eid, 20)
            .expect("test setup: delete edge at timestamp 20");
        let edge = g
            .get_edge(eid)
            .expect("test setup: get edge after deletion to inspect valid_until");
        assert_eq!(edge.valid_until, Some(20));
    }

    #[test]
    fn test_delete_edge_not_found() {
        let mut g = TemporalKnowledgeGraph::new();
        let fake = EdgeId([0u8; 16]);
        assert!(matches!(
            g.delete_edge(fake, 0),
            Err(TkgError::EdgeNotFound(_))
        ));
    }

    #[test]
    fn test_delete_edge_twice_errors() {
        let (mut g, a, b) = make_graph();
        let eid = g
            .add_edge(a, b, "r".to_string(), 1.0, 0)
            .expect("test setup: add edge for double-delete test");
        g.delete_edge(eid, 5)
            .expect("test setup: first deletion of edge must succeed");
        assert!(matches!(
            g.delete_edge(eid, 10),
            Err(TkgError::EdgeAlreadyDeleted(_))
        ));
    }

    #[test]
    fn test_edge_valid_at() {
        let edge = TkgEdge {
            id: EdgeId([0; 16]),
            src: NodeId([0; 16]),
            dst: NodeId([1; 16]),
            relation: "r".to_string(),
            weight: 1.0,
            valid_from: 10,
            valid_until: Some(20),
        };
        assert!(!edge.valid_at(9));
        assert!(edge.valid_at(10));
        assert!(edge.valid_at(19));
        assert!(!edge.valid_at(20));
    }

    // ── Properties ────────────────────────────────────────────────────────

    #[test]
    fn test_set_property_stores_value() {
        let (mut g, a, _) = make_graph();
        g.set_property(a, "age".to_string(), "30".to_string(), 1)
            .expect("test setup: set property 'age' to '30' on node a");
        let node = g
            .get_node(a)
            .expect("test setup: get node a after setting property");
        assert_eq!(node.properties.get("age").map(String::as_str), Some("30"));
    }

    #[test]
    fn test_set_property_overwrite() {
        let (mut g, a, _) = make_graph();
        g.set_property(a, "age".to_string(), "30".to_string(), 1)
            .expect("test setup: set initial property 'age' to '30'");
        g.set_property(a, "age".to_string(), "31".to_string(), 2)
            .expect("test setup: overwrite property 'age' to '31'");
        let node = g
            .get_node(a)
            .expect("test setup: get node a after property overwrite");
        assert_eq!(node.properties.get("age").map(String::as_str), Some("31"));
    }

    #[test]
    fn test_set_property_node_not_found() {
        let mut g = TemporalKnowledgeGraph::new();
        let fake = NodeId([0u8; 16]);
        assert!(matches!(
            g.set_property(fake, "k".to_string(), "v".to_string(), 0),
            Err(TkgError::NodeNotFound(_))
        ));
    }

    // ── Timeline & events ─────────────────────────────────────────────────

    #[test]
    fn test_events_at_returns_events() {
        let (g, a, _) = make_graph();
        let evs = g.events_at(0);
        assert_eq!(evs.len(), 2);
        let has_a = evs
            .iter()
            .any(|e| matches!(e, TkgEvent::NodeAdded(id) if *id == a));
        assert!(has_a);
    }

    #[test]
    fn test_events_at_empty_timestamp() {
        let g = TemporalKnowledgeGraph::new();
        assert!(g.events_at(999).is_empty());
    }

    #[test]
    fn test_timeline_iter_ordered() {
        let mut g = TemporalKnowledgeGraph::new();
        g.add_node("A".to_string(), 10)
            .expect("test setup: add node A at timestamp 10");
        g.add_node("B".to_string(), 5)
            .expect("test setup: add node B at timestamp 5");
        g.add_node("C".to_string(), 20)
            .expect("test setup: add node C at timestamp 20");
        let ts: Vec<u64> = g.timeline_iter().map(|(t, _)| t).collect();
        assert_eq!(ts, vec![5, 10, 20]);
    }

    // ── Queries ───────────────────────────────────────────────────────────

    #[test]
    fn test_query_nodes_at() {
        let (mut g, a, _) = make_graph();
        g.delete_node(a, 5)
            .expect("test setup: delete node a at timestamp 5");
        if let TkgQueryResult::Nodes(nodes) = g.query(TkgQuery::NodesAt(3)) {
            assert_eq!(nodes.len(), 2);
        } else {
            panic!("wrong variant");
        }
        if let TkgQueryResult::Nodes(nodes) = g.query(TkgQuery::NodesAt(10)) {
            assert_eq!(nodes.len(), 1);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_query_edges_at() {
        let (mut g, a, b) = make_graph();
        let eid = g
            .add_edge(a, b, "r".to_string(), 1.0, 0)
            .expect("test setup: add edge a→b at timestamp 0");
        g.delete_edge(eid, 10)
            .expect("test setup: delete edge at timestamp 10");
        if let TkgQueryResult::Edges(edges) = g.query(TkgQuery::EdgesAt(5)) {
            assert_eq!(edges.len(), 1);
        } else {
            panic!("wrong variant");
        }
        if let TkgQueryResult::Edges(edges) = g.query(TkgQuery::EdgesAt(15)) {
            assert_eq!(edges.len(), 0);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_query_node_history() {
        let (mut g, a, _) = make_graph();
        g.set_property(a, "x".to_string(), "1".to_string(), 2)
            .expect("test setup: set property 'x' to '1' at timestamp 2");
        g.set_property(a, "x".to_string(), "2".to_string(), 4)
            .expect("test setup: set property 'x' to '2' at timestamp 4");
        g.delete_node(a, 6)
            .expect("test setup: delete node a at timestamp 6 to generate delete event");
        if let TkgQueryResult::Events(events) = g.query(TkgQuery::NodeHistory(a)) {
            // NodeAdded + 2 PropertyChanged + NodeDeleted = 4
            assert_eq!(events.len(), 4);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_query_edges_between() {
        let (mut g, a, b) = make_graph();
        g.add_edge(a, b, "r".to_string(), 1.0, 5)
            .expect("test setup: add edge a→b with relation 'r' at timestamp 5");
        g.add_edge(a, b, "s".to_string(), 0.5, 15)
            .expect("test setup: add edge a→b with relation 's' at timestamp 15");
        if let TkgQueryResult::Edges(edges) = g.query(TkgQuery::EdgesBetween {
            src: a,
            dst: b,
            from_t: 0,
            to_t: 10,
        }) {
            assert_eq!(edges.len(), 1);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_query_edges_between_full_range() {
        let (mut g, a, b) = make_graph();
        g.add_edge(a, b, "r".to_string(), 1.0, 0)
            .expect("test setup: add first edge a→b with relation 'r'");
        g.add_edge(a, b, "s".to_string(), 0.5, 0)
            .expect("test setup: add second edge a→b with relation 's'");
        if let TkgQueryResult::Edges(edges) = g.query(TkgQuery::EdgesBetween {
            src: a,
            dst: b,
            from_t: 0,
            to_t: u64::MAX,
        }) {
            assert_eq!(edges.len(), 2);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_query_property_at_returns_correct_version() {
        let (mut g, a, _) = make_graph();
        g.set_property(a, "k".to_string(), "v1".to_string(), 5)
            .expect("test setup: set property 'k' to 'v1' at timestamp 5");
        g.set_property(a, "k".to_string(), "v2".to_string(), 10)
            .expect("test setup: set property 'k' to 'v2' at timestamp 10");
        if let TkgQueryResult::Property(Some(val)) = g.query(TkgQuery::PropertyAt {
            node_id: a,
            key: "k".to_string(),
            at_t: 7,
        }) {
            assert_eq!(val, "v1");
        } else {
            panic!("expected Some(v1)");
        }
        if let TkgQueryResult::Property(Some(val)) = g.query(TkgQuery::PropertyAt {
            node_id: a,
            key: "k".to_string(),
            at_t: 10,
        }) {
            assert_eq!(val, "v2");
        } else {
            panic!("expected Some(v2)");
        }
    }

    #[test]
    fn test_query_property_at_none_before_set() {
        let (mut g, a, _) = make_graph();
        g.set_property(a, "k".to_string(), "v".to_string(), 10)
            .expect("test setup: set property 'k' to 'v' at timestamp 10");
        if let TkgQueryResult::Property(val) = g.query(TkgQuery::PropertyAt {
            node_id: a,
            key: "k".to_string(),
            at_t: 5,
        }) {
            assert!(val.is_none());
        } else {
            panic!("wrong variant");
        }
    }

    // ── Snapshot ──────────────────────────────────────────────────────────

    #[test]
    fn test_snapshot_at_basic() {
        let (mut g, a, b) = make_graph();
        g.add_edge(a, b, "r".to_string(), 1.0, 0)
            .expect("test setup: add edge a→b for basic snapshot test");
        let snap = g.snapshot_at(5);
        assert_eq!(snap.at, 5);
        assert_eq!(snap.nodes.len(), 2);
        assert_eq!(snap.edges.len(), 1);
    }

    #[test]
    fn test_snapshot_excludes_deleted_nodes() {
        let (mut g, a, _) = make_graph();
        g.delete_node(a, 3)
            .expect("test setup: delete node a at timestamp 3 before snapshot at 5");
        let snap = g.snapshot_at(5);
        assert_eq!(snap.nodes.len(), 1);
    }

    #[test]
    fn test_snapshot_excludes_expired_edges() {
        let (mut g, a, b) = make_graph();
        let eid = g
            .add_edge(a, b, "r".to_string(), 1.0, 0)
            .expect("test setup: add edge for snapshot expired-edge test");
        g.delete_edge(eid, 5)
            .expect("test setup: delete edge at timestamp 5 before snapshot at 10");
        let snap = g.snapshot_at(10);
        assert_eq!(snap.edges.len(), 0);
    }

    #[test]
    fn test_snapshot_empty_graph() {
        let g = TemporalKnowledgeGraph::new();
        let snap = g.snapshot_at(100);
        assert!(snap.nodes.is_empty());
        assert!(snap.edges.is_empty());
    }

    // ── Temporal path ─────────────────────────────────────────────────────

    #[test]
    fn test_temporal_path_direct() {
        let (mut g, a, b) = make_graph();
        g.add_edge(a, b, "r".to_string(), 1.0, 0)
            .expect("test setup: add direct edge a→b for path test");
        let path = g
            .temporal_path(a, b, 5)
            .expect("test setup: direct path from a to b must exist at timestamp 5");
        assert_eq!(path, vec![a, b]);
    }

    #[test]
    fn test_temporal_path_self() {
        let (g, a, _) = make_graph();
        let path = g
            .temporal_path(a, a, 0)
            .expect("test setup: self-path from a to a must always exist");
        assert_eq!(path, vec![a]);
    }

    #[test]
    fn test_temporal_path_multi_hop() {
        let mut g = TemporalKnowledgeGraph::new();
        let a = g
            .add_node("A".to_string(), 0)
            .expect("test setup: add node A for multi-hop path test");
        let b = g
            .add_node("B".to_string(), 0)
            .expect("test setup: add node B for multi-hop path test");
        let c = g
            .add_node("C".to_string(), 0)
            .expect("test setup: add node C for multi-hop path test");
        g.add_edge(a, b, "r".to_string(), 1.0, 0)
            .expect("test setup: add edge A→B for multi-hop path");
        g.add_edge(b, c, "r".to_string(), 1.0, 0)
            .expect("test setup: add edge B→C for multi-hop path");
        let path = g
            .temporal_path(a, c, 5)
            .expect("test setup: multi-hop path A→B→C must exist at timestamp 5");
        assert_eq!(path, vec![a, b, c]);
    }

    #[test]
    fn test_temporal_path_no_path() {
        let (g, a, b) = make_graph();
        // No edges added.
        assert!(g.temporal_path(a, b, 5).is_none());
    }

    #[test]
    fn test_temporal_path_expired_edge() {
        let (mut g, a, b) = make_graph();
        let eid = g
            .add_edge(a, b, "r".to_string(), 1.0, 0)
            .expect("test setup: add edge that will be expired for path test");
        g.delete_edge(eid, 5)
            .expect("test setup: delete edge at timestamp 5 to expire it");
        // Path exists at t=3 but not t=10.
        assert!(g.temporal_path(a, b, 3).is_some());
        assert!(g.temporal_path(a, b, 10).is_none());
    }

    #[test]
    fn test_temporal_path_deleted_node() {
        let (mut g, a, b) = make_graph();
        g.add_edge(a, b, "r".to_string(), 1.0, 0)
            .expect("test setup: add edge a→b before deleting node b");
        g.delete_node(b, 5)
            .expect("test setup: delete destination node b at timestamp 5");
        // dst deleted at t=10 → no path.
        assert!(g.temporal_path(a, b, 10).is_none());
    }

    #[test]
    fn test_temporal_path_unknown_src() {
        let g = TemporalKnowledgeGraph::new();
        let fake = NodeId([0u8; 16]);
        let fake2 = NodeId([1u8; 16]);
        assert!(g.temporal_path(fake, fake2, 0).is_none());
    }

    // ── Merge ─────────────────────────────────────────────────────────────

    #[test]
    fn test_merge_keep_mine() {
        let (mut g1, a, _) = make_graph();
        g1.set_property(a, "x".to_string(), "mine".to_string(), 1)
            .expect("test setup: set property 'x' to 'mine' on node a in g1");

        let mut g2 = TemporalKnowledgeGraph::new();
        // Insert node with same id manually by building a compatible graph.
        // We'll just add a new node in g2 and merge; property should not override.
        g2.nodes.insert(
            a,
            TkgNode {
                id: a,
                label: "Alice-other".to_string(),
                properties: {
                    let mut m = HashMap::new();
                    m.insert("x".to_string(), "other".to_string());
                    m
                },
                created_at: 0,
                deleted_at: None,
            },
        );

        g1.merge_graphs(&g2, TkgMergePolicy::KeepMine);
        assert_eq!(
            g1.get_node(a)
                .expect("test setup: get node a from g1 after KeepMine merge")
                .properties
                .get("x")
                .map(String::as_str),
            Some("mine")
        );
    }

    #[test]
    fn test_merge_keep_other() {
        let (mut g1, a, _) = make_graph();
        g1.set_property(a, "x".to_string(), "mine".to_string(), 1)
            .expect("test setup: set property 'x' to 'mine' on node a before KeepOther merge");

        let mut g2 = TemporalKnowledgeGraph::new();
        g2.nodes.insert(
            a,
            TkgNode {
                id: a,
                label: "Alice-other".to_string(),
                properties: {
                    let mut m = HashMap::new();
                    m.insert("x".to_string(), "other".to_string());
                    m
                },
                created_at: 0,
                deleted_at: None,
            },
        );

        g1.merge_graphs(&g2, TkgMergePolicy::KeepOther);
        assert_eq!(
            g1.get_node(a)
                .expect("test setup: get node a from g1 after KeepOther merge")
                .properties
                .get("x")
                .map(String::as_str),
            Some("other")
        );
    }

    #[test]
    fn test_merge_union_all_new_node() {
        let (mut g1, _, _) = make_graph();
        let mut g2 = TemporalKnowledgeGraph::new();
        let c = g2
            .add_node("Charlie".to_string(), 0)
            .expect("test setup: add Charlie node to g2 for union merge test");

        g1.merge_graphs(&g2, TkgMergePolicy::UnionAll);
        assert!(g1.get_node(c).is_some());
    }

    #[test]
    fn test_merge_timeline_combined() {
        let mut g1 = TemporalKnowledgeGraph::new();
        g1.add_node("A".to_string(), 5)
            .expect("test setup: add node A at timestamp 5 in g1");

        let mut g2 = TemporalKnowledgeGraph::new();
        g2.add_node("B".to_string(), 10)
            .expect("test setup: add node B at timestamp 10 in g2");

        g1.merge_graphs(&g2, TkgMergePolicy::UnionAll);
        assert!(!g1.events_at(5).is_empty());
        assert!(!g1.events_at(10).is_empty());
    }

    #[test]
    fn test_merge_edges_keep_mine() {
        let (mut g1, a, b) = make_graph();
        let eid = g1
            .add_edge(a, b, "mine".to_string(), 1.0, 0)
            .expect("test setup: add edge with relation 'mine' in g1 before merge");

        let mut g2 = TemporalKnowledgeGraph::new();
        g2.edges.insert(
            eid,
            TkgEdge {
                id: eid,
                src: a,
                dst: b,
                relation: "other".to_string(),
                weight: 0.5,
                valid_from: 0,
                valid_until: None,
            },
        );

        g1.merge_graphs(&g2, TkgMergePolicy::KeepMine);
        assert_eq!(
            g1.get_edge(eid)
                .expect("test setup: get edge from g1 after KeepMine merge")
                .relation,
            "mine"
        );
    }

    // ── Statistics ────────────────────────────────────────────────────────

    #[test]
    fn test_stats_basic() {
        let (mut g, a, b) = make_graph();
        g.add_edge(a, b, "r".to_string(), 1.0, 0)
            .expect("test setup: add edge a→b for stats test");
        let s = g.stats();
        assert_eq!(s.total_nodes, 2);
        assert_eq!(s.live_nodes, 2);
        assert_eq!(s.total_edges, 1);
        assert_eq!(s.live_edges, 1);
    }

    #[test]
    fn test_stats_after_deletion() {
        let (mut g, a, b) = make_graph();
        let eid = g
            .add_edge(a, b, "r".to_string(), 1.0, 0)
            .expect("test setup: add edge for stats-after-deletion test");
        g.delete_node(a, 5)
            .expect("test setup: delete node a at timestamp 5 for stats test");
        g.delete_edge(eid, 5)
            .expect("test setup: delete edge at timestamp 5 for stats test");
        let s = g.stats();
        assert_eq!(s.live_nodes, 1);
        assert_eq!(s.live_edges, 0);
    }

    #[test]
    fn test_stats_timestamps() {
        let mut g = TemporalKnowledgeGraph::new();
        g.add_node("A".to_string(), 3)
            .expect("test setup: add node A at timestamp 3");
        g.add_node("B".to_string(), 7)
            .expect("test setup: add node B at timestamp 7");
        let s = g.stats();
        assert_eq!(s.first_timestamp, Some(3));
        assert_eq!(s.last_timestamp, Some(7));
    }

    #[test]
    fn test_stats_empty_graph() {
        let g = TemporalKnowledgeGraph::new();
        let s = g.stats();
        assert_eq!(s.total_nodes, 0);
        assert_eq!(s.total_events, 0);
        assert!(s.first_timestamp.is_none());
    }

    // ── Node/Edge counts ──────────────────────────────────────────────────

    #[test]
    fn test_node_count() {
        let (g, _, _) = make_graph();
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn test_edge_count() {
        let (mut g, a, b) = make_graph();
        g.add_edge(a, b, "r".to_string(), 1.0, 0)
            .expect("test setup: add single edge for edge count test");
        assert_eq!(g.edge_count(), 1);
    }

    // ── Default / Clone ───────────────────────────────────────────────────

    #[test]
    fn test_default_empty() {
        let g: TemporalKnowledgeGraph = Default::default();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_clone_independent() {
        let (mut g, a, b) = make_graph();
        let mut g2 = g.clone();
        g.add_edge(a, b, "r".to_string(), 1.0, 0)
            .expect("test setup: add edge a→b for clone test");
        assert_eq!(g.edge_count(), 1);
        assert_eq!(g2.edge_count(), 0);
        // Modifications to g2 don't affect g.
        g2.delete_node(a, 99)
            .expect("test setup: delete node a in cloned graph to verify independence");
        assert!(g
            .get_node(a)
            .expect("test setup: get node a from original graph after clone modification")
            .deleted_at
            .is_none());
    }

    // ── Error display ─────────────────────────────────────────────────────

    #[test]
    fn test_error_display_node_not_found() {
        let e = TkgError::NodeNotFound(NodeId([0u8; 16]));
        assert!(e.to_string().contains("not found"));
    }

    #[test]
    fn test_error_display_edge_not_found() {
        let e = TkgError::EdgeNotFound(EdgeId([0u8; 16]));
        assert!(e.to_string().contains("not found"));
    }

    // ── Edge uniqueness ───────────────────────────────────────────────────

    #[test]
    fn test_multiple_edges_same_pair() {
        let (mut g, a, b) = make_graph();
        let e1 = g
            .add_edge(a, b, "r1".to_string(), 1.0, 0)
            .expect("test setup: add first edge with relation 'r1'");
        let e2 = g
            .add_edge(a, b, "r2".to_string(), 0.5, 0)
            .expect("test setup: add second edge with relation 'r2'");
        assert_ne!(e1, e2);
        assert_eq!(g.edge_count(), 2);
    }

    // ── Large graph BFS ───────────────────────────────────────────────────

    #[test]
    fn test_temporal_path_long_chain() {
        let mut g = TemporalKnowledgeGraph::new();
        let n = 20usize;
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            ids.push(
                g.add_node(format!("N{i}"), 0)
                    .expect("test setup: add node N{i} in long-chain test"),
            );
        }
        for i in 0..n - 1 {
            g.add_edge(ids[i], ids[i + 1], "next".to_string(), 1.0, 0)
                .expect("test setup: add chain edge from node i to node i+1");
        }
        let path = g
            .temporal_path(ids[0], ids[n - 1], 0)
            .expect("test setup: long chain path from first to last node must exist");
        assert_eq!(path.len(), n);
        assert_eq!(path[0], ids[0]);
        assert_eq!(path[n - 1], ids[n - 1]);
    }

    // ── NodeId / EdgeId ordering ──────────────────────────────────────────

    #[test]
    fn test_node_id_ordering() {
        let a = NodeId([0u8; 16]);
        let b = NodeId([1u8; 16]);
        assert!(a < b);
    }

    #[test]
    fn test_edge_id_equality() {
        let a = EdgeId([42u8; 16]);
        let b = EdgeId([42u8; 16]);
        assert_eq!(a, b);
    }
}
