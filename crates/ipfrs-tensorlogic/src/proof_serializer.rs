//! Proof Serializer
//!
//! Serializes and deserializes distributed backward-chaining proof trees for
//! storage in IPLD and transmission over the network.
//!
//! A [`ProofTreeInput`] (logical tree) is flattened in DFS order into a
//! [`SerializedProof`] whose `proof_id` is the FNV-1a hex digest of the
//! sorted, concatenated node IDs.  Round-trip fidelity is guaranteed: every
//! field, binding, and topology is reconstructed by [`ProofSerializer::deserialize`].
//!
//! # Examples
//!
//! ```
//! use ipfrs_tensorlogic::proof_serializer::{
//!     ProofNodeInput, ProofSerializer, ProofTreeInput,
//! };
//! use std::collections::HashMap;
//!
//! // Build a trivial single-node proof.
//! let mut nodes = HashMap::new();
//! nodes.insert(
//!     "n0".to_string(),
//!     ProofNodeInput {
//!         rule_id: None,
//!         bindings: HashMap::new(),
//!         children: vec![],
//!         peer_id: None,
//!     },
//! );
//! let input = ProofTreeInput {
//!     root_goal: "n0".to_string(),
//!     nodes,
//!     proved: true,
//! };
//!
//! let ser = ProofSerializer::default();
//! let proof = ser.serialize(&input).expect("example: should succeed in docs");
//! assert!(proof.proved);
//! assert_eq!(proof.depth, 0);
//! assert_eq!(proof.edge_count, 0);
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors produced by [`ProofSerializer`].
#[derive(Debug, Error)]
pub enum ProofSerError {
    /// A node referenced by `root_goal` or a child ID is absent from `nodes`.
    #[error("missing node: {0}")]
    MissingNode(String),

    /// The proof tree contains a cycle (a node appears in its own transitive children).
    #[error("cycle detected in proof tree")]
    CycleDetected,

    /// JSON encoding/decoding error.
    #[error("JSON error: {0}")]
    JsonError(String),

    /// `serialize` was called with an empty `nodes` map.
    #[error("proof tree is empty")]
    EmptyTree,
}

// ─── Input types ─────────────────────────────────────────────────────────────

/// A single node as provided by the caller prior to serialization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProofNodeInput {
    /// Identifier of the rule applied at this node, if any.
    pub rule_id: Option<String>,

    /// Variable bindings established at this node.
    pub bindings: HashMap<String, String>,

    /// Ordered list of child node IDs.
    pub children: Vec<String>,

    /// ID of the remote peer that resolved this goal (if any).
    pub peer_id: Option<String>,
}

/// Full proof tree as produced by the distributed backward chainer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProofTreeInput {
    /// Node ID of the root goal.
    pub root_goal: String,

    /// All nodes in the tree, keyed by their node ID.
    pub nodes: HashMap<String, ProofNodeInput>,

    /// Whether the root goal was successfully proved.
    pub proved: bool,
}

// ─── Serialized / stored types ───────────────────────────────────────────────

/// A flattened representation of one node, suitable for storage and transmission.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProofNodeRecord {
    /// Unique identifier of this node within the proof.
    pub node_id: String,

    /// Rule applied at this node, if any.
    pub rule_id: Option<String>,

    /// Variable bindings at this node.
    pub bindings: HashMap<String, String>,

    /// IDs of direct child nodes (in original order).
    pub children_ids: Vec<String>,

    /// `true` when the node has no children (base-fact or axiom leaf).
    pub is_leaf: bool,

    /// Remote peer that resolved this node's goal, if any.
    pub peer_id: Option<String>,
}

/// A fully serialized proof tree, ready for IPLD storage or network transmission.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SerializedProof {
    /// FNV-1a hex digest of the sorted, concatenated node IDs.
    pub proof_id: String,

    /// Node ID of the root goal.
    pub goal: String,

    /// All nodes in DFS pre-order.
    pub nodes: Vec<ProofNodeRecord>,

    /// Total number of directed parent→child edges.
    pub edge_count: usize,

    /// Maximum depth of the tree (root = depth 0).
    pub depth: u32,

    /// Whether the root goal was proved.
    pub proved: bool,

    /// Total number of bindings across all nodes.
    pub total_bindings: usize,

    /// Unix millisecond timestamp when this struct was created.
    pub serialized_at_ms: u64,
}

// ─── Stats ───────────────────────────────────────────────────────────────────

/// A point-in-time snapshot of [`ProofSerializerStats`] counters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofSerializerStatsSnapshot {
    pub total_serialized: u64,
    pub total_deserialized: u64,
    pub total_json_encoded: u64,
    pub total_json_decoded: u64,
}

/// Atomic counters tracking [`ProofSerializer`] activity.
#[derive(Debug, Default)]
pub struct ProofSerializerStats {
    total_serialized: AtomicU64,
    total_deserialized: AtomicU64,
    total_json_encoded: AtomicU64,
    total_json_decoded: AtomicU64,
}

impl ProofSerializerStats {
    /// Returns a consistent snapshot (individual atomics read sequentially).
    pub fn snapshot(&self) -> ProofSerializerStatsSnapshot {
        ProofSerializerStatsSnapshot {
            total_serialized: self.total_serialized.load(Ordering::Relaxed),
            total_deserialized: self.total_deserialized.load(Ordering::Relaxed),
            total_json_encoded: self.total_json_encoded.load(Ordering::Relaxed),
            total_json_decoded: self.total_json_decoded.load(Ordering::Relaxed),
        }
    }
}

// ─── ProofSerializer ─────────────────────────────────────────────────────────

/// Serializes and deserializes distributed proof trees.
///
/// The serializer is stateless apart from its [`ProofSerializerStats`] counter
/// set; it can therefore be shared freely across threads via an `Arc`.
#[derive(Debug, Default)]
pub struct ProofSerializer {
    /// Cumulative operation counters.
    pub stats: ProofSerializerStats,
}

impl ProofSerializer {
    /// Creates a new [`ProofSerializer`] with zeroed counters.
    pub fn new() -> Self {
        Self::default()
    }

    // ── public API ─────────────────────────────────────────────────────────

    /// Flattens `input` into a [`SerializedProof`] using DFS pre-order traversal.
    ///
    /// # Errors
    ///
    /// - [`ProofSerError::EmptyTree`] when `nodes` is empty.
    /// - [`ProofSerError::MissingNode`] when the root or any child ID is absent.
    /// - [`ProofSerError::CycleDetected`] when the tree contains a cycle.
    pub fn serialize(&self, input: &ProofTreeInput) -> Result<SerializedProof, ProofSerError> {
        if input.nodes.is_empty() {
            return Err(ProofSerError::EmptyTree);
        }

        // DFS pre-order traversal — collect (node_id, depth) pairs.
        let order = self.dfs_order(input)?;

        // Compute aggregate metrics.
        let depth = order.iter().map(|(_, d)| *d).max().unwrap_or(0);
        let edge_count: usize = order
            .iter()
            .map(|(id, _)| input.nodes.get(id).map(|n| n.children.len()).unwrap_or(0))
            .sum();
        let total_bindings: usize = order
            .iter()
            .map(|(id, _)| input.nodes.get(id).map(|n| n.bindings.len()).unwrap_or(0))
            .sum();

        // Build flat records.
        let nodes: Vec<ProofNodeRecord> = order
            .iter()
            .map(|(id, _)| {
                let n = &input.nodes[id];
                ProofNodeRecord {
                    node_id: id.clone(),
                    rule_id: n.rule_id.clone(),
                    bindings: n.bindings.clone(),
                    children_ids: n.children.clone(),
                    is_leaf: n.children.is_empty(),
                    peer_id: n.peer_id.clone(),
                }
            })
            .collect();

        // Compute proof_id: FNV-1a of sorted, concatenated node IDs.
        let mut sorted_ids: Vec<&str> = order.iter().map(|(id, _)| id.as_str()).collect();
        sorted_ids.sort_unstable();
        let proof_id = fnv1a_hex(sorted_ids.iter().copied());

        let serialized_at_ms = current_ms();

        self.stats.total_serialized.fetch_add(1, Ordering::Relaxed);

        Ok(SerializedProof {
            proof_id,
            goal: input.root_goal.clone(),
            nodes,
            edge_count,
            depth,
            proved: input.proved,
            total_bindings,
            serialized_at_ms,
        })
    }

    /// Reconstructs a [`ProofTreeInput`] from a [`SerializedProof`].
    ///
    /// # Errors
    ///
    /// Returns [`ProofSerError::EmptyTree`] when the proof contains no nodes,
    /// or [`ProofSerError::MissingNode`] when a referenced child is absent.
    pub fn deserialize(&self, proof: &SerializedProof) -> Result<ProofTreeInput, ProofSerError> {
        if proof.nodes.is_empty() {
            return Err(ProofSerError::EmptyTree);
        }

        // Build the node map from the flat records.
        let mut nodes: HashMap<String, ProofNodeInput> = HashMap::with_capacity(proof.nodes.len());
        for record in &proof.nodes {
            nodes.insert(
                record.node_id.clone(),
                ProofNodeInput {
                    rule_id: record.rule_id.clone(),
                    bindings: record.bindings.clone(),
                    children: record.children_ids.clone(),
                    peer_id: record.peer_id.clone(),
                },
            );
        }

        // Verify all referenced children actually exist.
        for record in &proof.nodes {
            for child_id in &record.children_ids {
                if !nodes.contains_key(child_id) {
                    return Err(ProofSerError::MissingNode(child_id.clone()));
                }
            }
        }

        self.stats
            .total_deserialized
            .fetch_add(1, Ordering::Relaxed);

        Ok(ProofTreeInput {
            root_goal: proof.goal.clone(),
            nodes,
            proved: proof.proved,
        })
    }

    /// Encodes `proof` as a compact JSON string.
    pub fn to_json(&self, proof: &SerializedProof) -> Result<String, ProofSerError> {
        let json =
            serde_json::to_string(proof).map_err(|e| ProofSerError::JsonError(e.to_string()))?;
        self.stats
            .total_json_encoded
            .fetch_add(1, Ordering::Relaxed);
        Ok(json)
    }

    /// Decodes a [`SerializedProof`] from a JSON string.
    pub fn from_json(&self, json: &str) -> Result<SerializedProof, ProofSerError> {
        let proof: SerializedProof =
            serde_json::from_str(json).map_err(|e| ProofSerError::JsonError(e.to_string()))?;
        self.stats
            .total_json_decoded
            .fetch_add(1, Ordering::Relaxed);
        Ok(proof)
    }

    // ── internals ──────────────────────────────────────────────────────────

    /// Returns all node IDs in DFS pre-order together with their depth.
    ///
    /// Detects cycles by tracking the current call stack (ancestors).
    fn dfs_order(&self, input: &ProofTreeInput) -> Result<Vec<(String, u32)>, ProofSerError> {
        let root = &input.root_goal;
        if !input.nodes.contains_key(root) {
            return Err(ProofSerError::MissingNode(root.clone()));
        }

        let mut order: Vec<(String, u32)> = Vec::with_capacity(input.nodes.len());
        // `ancestors` tracks the current DFS stack to detect back-edges (cycles).
        let mut ancestors: HashSet<String> = HashSet::new();

        Self::dfs_visit(input, root, 0, &mut order, &mut ancestors)?;
        Ok(order)
    }

    fn dfs_visit(
        input: &ProofTreeInput,
        node_id: &str,
        depth: u32,
        order: &mut Vec<(String, u32)>,
        ancestors: &mut HashSet<String>,
    ) -> Result<(), ProofSerError> {
        if ancestors.contains(node_id) {
            return Err(ProofSerError::CycleDetected);
        }

        let node = input
            .nodes
            .get(node_id)
            .ok_or_else(|| ProofSerError::MissingNode(node_id.to_string()))?;

        order.push((node_id.to_string(), depth));
        ancestors.insert(node_id.to_string());

        for child_id in &node.children {
            Self::dfs_visit(input, child_id, depth + 1, order, ancestors)?;
        }

        ancestors.remove(node_id);
        Ok(())
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Computes the FNV-1a 64-bit hash of all strings in `parts` and returns a
/// lower-hex string (16 characters).
fn fnv1a_hex<'a>(parts: impl Iterator<Item = &'a str>) -> String {
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;

    let mut hash: u64 = OFFSET_BASIS;
    for part in parts {
        for byte in part.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(PRIME);
        }
    }
    format!("{:016x}", hash)
}

/// Returns the current Unix time in milliseconds, falling back to 0 on error.
fn current_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────────

    fn single_node_input() -> ProofTreeInput {
        let mut nodes = HashMap::new();
        nodes.insert(
            "n0".to_string(),
            ProofNodeInput {
                rule_id: None,
                bindings: HashMap::new(),
                children: vec![],
                peer_id: None,
            },
        );
        ProofTreeInput {
            root_goal: "n0".to_string(),
            nodes,
            proved: true,
        }
    }

    /// Builds:
    /// ```
    ///        n0 (root)
    ///       /       \
    ///     n1         n2
    ///    /  \
    ///  n3    n4
    /// ```
    fn multi_level_input() -> ProofTreeInput {
        let mut nodes = HashMap::new();
        nodes.insert(
            "n0".to_string(),
            ProofNodeInput {
                rule_id: Some("rule_A".to_string()),
                bindings: [("X".to_string(), "alice".to_string())].into(),
                children: vec!["n1".to_string(), "n2".to_string()],
                peer_id: None,
            },
        );
        nodes.insert(
            "n1".to_string(),
            ProofNodeInput {
                rule_id: Some("rule_B".to_string()),
                bindings: [("Y".to_string(), "bob".to_string())].into(),
                children: vec!["n3".to_string(), "n4".to_string()],
                peer_id: Some("peer-1".to_string()),
            },
        );
        nodes.insert(
            "n2".to_string(),
            ProofNodeInput {
                rule_id: None,
                bindings: HashMap::new(),
                children: vec![],
                peer_id: None,
            },
        );
        nodes.insert(
            "n3".to_string(),
            ProofNodeInput {
                rule_id: None,
                bindings: [("Z".to_string(), "carol".to_string())].into(),
                children: vec![],
                peer_id: Some("peer-2".to_string()),
            },
        );
        nodes.insert(
            "n4".to_string(),
            ProofNodeInput {
                rule_id: None,
                bindings: HashMap::new(),
                children: vec![],
                peer_id: None,
            },
        );
        ProofTreeInput {
            root_goal: "n0".to_string(),
            nodes,
            proved: true,
        }
    }

    // ── tests ──────────────────────────────────────────────────────────────

    #[test]
    fn test_serialize_single_node() {
        let ser = ProofSerializer::new();
        let input = single_node_input();
        let proof = ser.serialize(&input).expect("serialize should succeed");

        assert_eq!(proof.nodes.len(), 1);
        assert_eq!(proof.nodes[0].node_id, "n0");
        assert!(proof.nodes[0].is_leaf);
        assert_eq!(proof.edge_count, 0);
        assert_eq!(proof.depth, 0);
        assert!(proof.proved);
        assert_eq!(proof.total_bindings, 0);
        assert!(!proof.proof_id.is_empty());
    }

    #[test]
    fn test_serialize_multi_level() {
        let ser = ProofSerializer::new();
        let input = multi_level_input();
        let proof = ser.serialize(&input).expect("serialize multi-level");

        // 5 nodes total
        assert_eq!(proof.nodes.len(), 5);

        // depth: n0=0, n1=1, n3=2, n4=2 → max depth = 2
        assert_eq!(proof.depth, 2);

        // edges: n0→n1, n0→n2, n1→n3, n1→n4 = 4
        assert_eq!(proof.edge_count, 4);

        // DFS pre-order from n0: n0, n1, n3, n4, n2
        assert_eq!(proof.nodes[0].node_id, "n0");
        assert_eq!(proof.nodes[1].node_id, "n1");
        assert_eq!(proof.nodes[2].node_id, "n3");
        assert_eq!(proof.nodes[3].node_id, "n4");
        assert_eq!(proof.nodes[4].node_id, "n2");
    }

    #[test]
    fn test_proof_id_is_deterministic() {
        let ser = ProofSerializer::new();
        let input = multi_level_input();

        let proof1 = ser.serialize(&input).expect("first serialize");
        let proof2 = ser.serialize(&input).expect("second serialize");

        // proof_id must not depend on HashMap iteration order
        assert_eq!(proof1.proof_id, proof2.proof_id);
    }

    #[test]
    fn test_proof_id_differs_for_different_trees() {
        let ser = ProofSerializer::new();
        let input1 = single_node_input();
        let input2 = multi_level_input();

        let p1 = ser.serialize(&input1).expect("test: should succeed");
        let p2 = ser.serialize(&input2).expect("test: should succeed");

        assert_ne!(p1.proof_id, p2.proof_id);
    }

    #[test]
    fn test_deserialize_round_trip() {
        let ser = ProofSerializer::new();
        let input = multi_level_input();

        let proof = ser.serialize(&input).expect("serialize");
        let reconstructed = ser.deserialize(&proof).expect("deserialize");

        // The node map should be equal (order-agnostic).
        assert_eq!(reconstructed.root_goal, input.root_goal);
        assert_eq!(reconstructed.proved, input.proved);
        assert_eq!(reconstructed.nodes.len(), input.nodes.len());

        for (id, orig_node) in &input.nodes {
            let rec_node = reconstructed.nodes.get(id).expect("node should exist");
            assert_eq!(rec_node.rule_id, orig_node.rule_id);
            assert_eq!(rec_node.bindings, orig_node.bindings);
            assert_eq!(rec_node.children, orig_node.children);
            assert_eq!(rec_node.peer_id, orig_node.peer_id);
        }
    }

    #[test]
    fn test_json_round_trip() {
        let ser = ProofSerializer::new();
        let input = multi_level_input();
        let proof = ser.serialize(&input).expect("serialize");

        let json = ser.to_json(&proof).expect("to_json");
        let decoded = ser.from_json(&json).expect("from_json");

        assert_eq!(proof, decoded);
    }

    #[test]
    fn test_missing_root_node_returns_error() {
        let ser = ProofSerializer::new();
        let mut input = single_node_input();
        input.root_goal = "nonexistent".to_string();

        let result = ser.serialize(&input);
        assert!(matches!(result, Err(ProofSerError::MissingNode(_))));
    }

    #[test]
    fn test_empty_tree_returns_error() {
        let ser = ProofSerializer::new();
        let input = ProofTreeInput {
            root_goal: "n0".to_string(),
            nodes: HashMap::new(),
            proved: false,
        };
        let result = ser.serialize(&input);
        assert!(matches!(result, Err(ProofSerError::EmptyTree)));
    }

    #[test]
    fn test_proved_flag_preserved_false() {
        let ser = ProofSerializer::new();
        let mut input = single_node_input();
        input.proved = false;

        let proof = ser.serialize(&input).expect("serialize");
        assert!(!proof.proved);
        let reconstructed = ser.deserialize(&proof).expect("deserialize");
        assert!(!reconstructed.proved);
    }

    #[test]
    fn test_depth_computed_correctly() {
        let ser = ProofSerializer::new();
        // Chain: n0 → n1 → n2 → n3 (depth = 3)
        let mut nodes = HashMap::new();
        for i in 0..4_usize {
            nodes.insert(
                format!("n{}", i),
                ProofNodeInput {
                    rule_id: None,
                    bindings: HashMap::new(),
                    children: if i < 3 {
                        vec![format!("n{}", i + 1)]
                    } else {
                        vec![]
                    },
                    peer_id: None,
                },
            );
        }
        let input = ProofTreeInput {
            root_goal: "n0".to_string(),
            nodes,
            proved: true,
        };
        let proof = ser.serialize(&input).expect("serialize chain");
        assert_eq!(proof.depth, 3);
        assert_eq!(proof.edge_count, 3);
    }

    #[test]
    fn test_edge_count_correct() {
        let ser = ProofSerializer::new();
        let input = multi_level_input();
        let proof = ser.serialize(&input).expect("serialize");
        // Edges: n0→n1, n0→n2, n1→n3, n1→n4 = 4
        assert_eq!(proof.edge_count, 4);
    }

    #[test]
    fn test_total_bindings_correct() {
        let ser = ProofSerializer::new();
        let input = multi_level_input();
        let proof = ser.serialize(&input).expect("serialize");
        // n0: 1, n1: 1, n2: 0, n3: 1, n4: 0 → 3
        assert_eq!(proof.total_bindings, 3);
    }

    #[test]
    fn test_cycle_detection() {
        let ser = ProofSerializer::new();
        // n0 → n1 → n0  (cycle)
        let mut nodes = HashMap::new();
        nodes.insert(
            "n0".to_string(),
            ProofNodeInput {
                rule_id: None,
                bindings: HashMap::new(),
                children: vec!["n1".to_string()],
                peer_id: None,
            },
        );
        nodes.insert(
            "n1".to_string(),
            ProofNodeInput {
                rule_id: None,
                bindings: HashMap::new(),
                children: vec!["n0".to_string()],
                peer_id: None,
            },
        );
        let input = ProofTreeInput {
            root_goal: "n0".to_string(),
            nodes,
            proved: false,
        };
        let result = ser.serialize(&input);
        assert!(matches!(result, Err(ProofSerError::CycleDetected)));
    }

    #[test]
    fn test_stats_accumulation() {
        let ser = ProofSerializer::new();
        let input = single_node_input();

        // Initial state
        let snap0 = ser.stats.snapshot();
        assert_eq!(snap0.total_serialized, 0);
        assert_eq!(snap0.total_deserialized, 0);
        assert_eq!(snap0.total_json_encoded, 0);
        assert_eq!(snap0.total_json_decoded, 0);

        let proof = ser.serialize(&input).expect("test: should succeed");
        ser.serialize(&input).expect("test: should succeed");
        let snap1 = ser.stats.snapshot();
        assert_eq!(snap1.total_serialized, 2);

        ser.deserialize(&proof).expect("test: should succeed");
        let snap2 = ser.stats.snapshot();
        assert_eq!(snap2.total_deserialized, 1);

        let json = ser.to_json(&proof).expect("test: should succeed");
        let snap3 = ser.stats.snapshot();
        assert_eq!(snap3.total_json_encoded, 1);

        ser.from_json(&json).expect("test: should succeed");
        let snap4 = ser.stats.snapshot();
        assert_eq!(snap4.total_json_decoded, 1);
    }

    #[test]
    fn test_json_invalid_input_returns_error() {
        let ser = ProofSerializer::new();
        let result = ser.from_json("{ not valid json }");
        assert!(matches!(result, Err(ProofSerError::JsonError(_))));
    }

    #[test]
    fn test_is_leaf_flag_correct() {
        let ser = ProofSerializer::new();
        let input = multi_level_input();
        let proof = ser.serialize(&input).expect("serialize");

        for record in &proof.nodes {
            let expected_leaf = record.children_ids.is_empty();
            assert_eq!(
                record.is_leaf, expected_leaf,
                "is_leaf mismatch for node {}",
                record.node_id
            );
        }
    }

    #[test]
    fn test_peer_id_preserved() {
        let ser = ProofSerializer::new();
        let input = multi_level_input();
        let proof = ser.serialize(&input).expect("serialize");

        // n1 should carry peer-1
        let n1 = proof
            .nodes
            .iter()
            .find(|r| r.node_id == "n1")
            .expect("n1 must be present");
        assert_eq!(n1.peer_id.as_deref(), Some("peer-1"));
    }
}
