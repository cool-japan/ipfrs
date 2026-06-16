//! CID-linked computation graph for gradient tracking.
//!
//! This module provides:
//! - [`ComputationNode`] — a single node in the gradient computation DAG
//! - [`ComputationGraphError`] — graph-specific error type
//! - [`ComputationGraphStore`] — full DAG with topological ordering, provenance,
//!   Arrow IPC gradient storage, and checkpointing

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::GradientError;

// ── ComputationNode ────────────────────────────────────────────────────────

/// A single node in the gradient computation graph.
///
/// Each node represents one tensor operation and holds CIDs that link it to
/// the content-addressed blocks produced and consumed during that operation.
/// This makes the full backward-pass auditable via IPLD provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationNode {
    /// Unique stable identifier for this node (UUID-v4 string)
    pub id: String,
    /// Operation name, e.g. `"matmul"`, `"relu"`, `"softmax"`
    pub op: String,
    /// CIDs of the input tensor blocks consumed by this operation
    pub input_cids: Vec<String>,
    /// CID of the output tensor block produced (None until the forward pass runs)
    pub output_cid: Option<String>,
    /// CID of the gradient tensor block (None until the backward pass runs)
    pub gradient_cid: Option<String>,
    /// Arbitrary string key/value annotations (shape, dtype, device, …)
    pub metadata: HashMap<String, String>,
}

impl ComputationNode {
    /// Create a new node with the given operation and input CIDs.
    pub fn new(op: impl Into<String>, input_cids: Vec<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            op: op.into(),
            input_cids,
            output_cid: None,
            gradient_cid: None,
            metadata: HashMap::new(),
        }
    }

    /// Attach an arbitrary metadata annotation.
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ── ComputationGraphError ──────────────────────────────────────────────────

/// Errors specific to the CID-linked computation graph.
#[derive(Debug, thiserror::Error)]
pub enum ComputationGraphError {
    #[error("Node not found: {0}")]
    NodeNotFound(String),

    #[error("Circular dependency detected in computation graph")]
    CircularDependency,

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<ComputationGraphError> for GradientError {
    fn from(e: ComputationGraphError) -> Self {
        GradientError::InvalidGradient(e.to_string())
    }
}

// ── ComputationGraphStore ──────────────────────────────────────────────────

/// CID-linked computation graph store.
///
/// Maintains the directed acyclic graph (DAG) of [`ComputationNode`]s that
/// constitutes the forward pass.  After forward execution each node is linked
/// to its output CID; after the backward pass each node carries a gradient CID.
/// The graph can be checkpointed to bytes and restored for resume.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComputationGraphStore {
    /// All nodes, keyed by node-id
    nodes: HashMap<String, ComputationNode>,
    /// Directed edges: (producer_node_id, consumer_node_id)
    edges: Vec<(String, String)>,
}

impl ComputationGraphStore {
    /// Create an empty computation graph store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a node into the graph.
    ///
    /// Edges are automatically derived from `node.input_cids`: for each input
    /// CID we look for an existing node whose `output_cid` matches, and record
    /// a directed edge from that producer node to the new node.
    pub fn add_node(&mut self, node: ComputationNode) {
        // Build edges: any existing node whose output_cid is in node.input_cids
        // is a producer of this node.
        let producers: Vec<String> = self
            .nodes
            .values()
            .filter_map(|n| {
                n.output_cid.as_ref().and_then(|oc| {
                    if node.input_cids.contains(oc) {
                        Some(n.id.clone())
                    } else {
                        None
                    }
                })
            })
            .collect();

        for producer_id in producers {
            self.edges.push((producer_id, node.id.clone()));
        }

        self.nodes.insert(node.id.clone(), node);
    }

    /// Record that the forward pass for `node_id` produced `output_cid`.
    ///
    /// After recording, edges to any downstream nodes that list this CID as an
    /// input are added automatically.
    pub fn record_output(
        &mut self,
        node_id: &str,
        output_cid: String,
    ) -> Result<(), GradientError> {
        let node = self
            .nodes
            .get_mut(node_id)
            .ok_or_else(|| GradientError::InvalidGradient(format!("Node not found: {node_id}")))?;
        node.output_cid = Some(output_cid.clone());

        // Wire up any downstream nodes that already declared this CID as input.
        let consumers: Vec<String> = self
            .nodes
            .values()
            .filter(|n| n.id != node_id && n.input_cids.contains(&output_cid))
            .map(|n| n.id.clone())
            .collect();

        for consumer_id in consumers {
            let edge = (node_id.to_string(), consumer_id);
            if !self.edges.contains(&edge) {
                self.edges.push(edge);
            }
        }

        Ok(())
    }

    /// Record that the backward pass computed a gradient for `node_id`.
    pub fn record_gradient(
        &mut self,
        node_id: &str,
        grad_cid: String,
    ) -> Result<(), GradientError> {
        let node = self
            .nodes
            .get_mut(node_id)
            .ok_or_else(|| GradientError::InvalidGradient(format!("Node not found: {node_id}")))?;
        node.gradient_cid = Some(grad_cid);
        Ok(())
    }

    /// Return node IDs in topological order (Kahn's algorithm).
    ///
    /// Nodes with no predecessors come first; this is the correct execution
    /// order for the forward pass (and reverse for backward).
    pub fn topological_order(&self) -> Vec<String> {
        // Build adjacency and in-degree maps.
        let mut in_degree: HashMap<&str, usize> =
            self.nodes.keys().map(|id| (id.as_str(), 0)).collect();

        let mut successors: HashMap<&str, Vec<&str>> = self
            .nodes
            .keys()
            .map(|id| (id.as_str(), Vec::new()))
            .collect();

        for (from, to) in &self.edges {
            *in_degree.entry(to.as_str()).or_insert(0) += 1;
            successors
                .entry(from.as_str())
                .or_default()
                .push(to.as_str());
        }

        // Collect sources (in-degree == 0).
        let mut queue: std::collections::VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        // Sort queue for determinism.
        let mut queue_vec: Vec<&str> = queue.drain(..).collect();
        queue_vec.sort_unstable();
        queue.extend(queue_vec);

        let mut order: Vec<String> = Vec::with_capacity(self.nodes.len());

        while let Some(node_id) = queue.pop_front() {
            order.push(node_id.to_string());

            if let Some(succs) = successors.get(node_id) {
                let mut next: Vec<&str> = succs
                    .iter()
                    .copied()
                    .filter(|&s| {
                        let deg = in_degree.get_mut(s).map(|d| {
                            *d = d.saturating_sub(1);
                            *d
                        });
                        deg == Some(0)
                    })
                    .collect();
                next.sort_unstable();
                queue.extend(next);
            }
        }

        order
    }

    // ── Arrow IPC gradient storage ────────────────────────────────────────

    /// Serialize a gradient tensor as an Apache Arrow IPC block and store its
    /// CID in the named node's `gradient_cid` field.
    ///
    /// The gradient data is wrapped in an [`crate::arrow::ArrowTensorStore`]
    /// as a single f32 column named `"gradient"`.  Shape metadata is embedded
    /// in the Arrow field metadata so it survives a round-trip through
    /// [`Self::load_gradient_from_arrow`].
    ///
    /// The CID is computed as a DAG-CBOR CID (SHA-256) over the raw Arrow IPC
    /// bytes, matching the convention used throughout `ipld_codec`.
    ///
    /// # Returns
    ///
    /// The CID string that was recorded on the node.
    pub fn store_gradient_as_arrow(
        &mut self,
        node_id: &str,
        gradient_data: &[f32],
        shape: &[usize],
    ) -> Result<String, GradientError> {
        use crate::arrow::{ArrowTensor, ArrowTensorStore};
        use ipfrs_core::CidBuilder;

        // Encode shape into a metadata key so it survives IPC round-trips.
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");

        // Build an ArrowTensor for the gradient data; embed shape in metadata.
        let mut tensor = ArrowTensor::from_slice_f32("gradient", shape.to_vec(), gradient_data);
        tensor
            .metadata
            .custom
            .insert("gradient_shape".to_string(), shape_str);

        let mut store = ArrowTensorStore::new();
        store.insert(tensor);

        let ipc_bytes = store
            .to_bytes()
            .map_err(|e| GradientError::InvalidGradient(format!("Arrow IPC encode error: {e}")))?;

        // Compute a DAG-CBOR CID over the raw Arrow IPC bytes.
        let cid = CidBuilder::new()
            .codec(0x71) // DAG-CBOR codec
            .build(&ipc_bytes)
            .map_err(|e| GradientError::InvalidGradient(format!("CID computation error: {e}")))?;
        let cid_str = cid.to_string();

        // Record gradient CID on the node.
        let node = self
            .nodes
            .get_mut(node_id)
            .ok_or_else(|| GradientError::InvalidGradient(format!("Node not found: {node_id}")))?;
        node.gradient_cid = Some(cid_str.clone());

        Ok(cid_str)
    }

    /// Decode gradient data from raw Arrow IPC block bytes.
    ///
    /// Validates that the decoded tensor shape matches `expected_shape` before
    /// returning the f32 values.  Pass an empty `expected_shape` slice to skip
    /// shape validation.
    pub fn load_gradient_from_arrow(
        arrow_bytes: &[u8],
        expected_shape: &[usize],
    ) -> Result<Vec<f32>, GradientError> {
        use crate::arrow::ArrowTensorStore;

        let store = ArrowTensorStore::from_bytes(arrow_bytes)
            .map_err(|e| GradientError::InvalidGradient(format!("Arrow IPC decode error: {e}")))?;

        let tensor = store.get("gradient").ok_or_else(|| {
            GradientError::InvalidGradient(
                "Arrow IPC block does not contain a 'gradient' column".to_string(),
            )
        })?;

        // Validate shape if provided.
        if !expected_shape.is_empty() && tensor.metadata.shape != expected_shape {
            return Err(GradientError::ShapeMismatch {
                expected: expected_shape.to_vec(),
                actual: tensor.metadata.shape.clone(),
            });
        }

        let slice = tensor
            .as_slice_f32()
            .ok_or(GradientError::IncompatibleDtype(
                crate::arrow::TensorDtype::Float32,
            ))?;

        Ok(slice.to_vec())
    }

    /// Serialize the entire graph to JSON bytes for checkpointing.
    pub fn checkpoint(&self) -> Result<Vec<u8>, GradientError> {
        serde_json::to_vec(self)
            .map_err(|e| GradientError::InvalidGradient(format!("Checkpoint serialization: {e}")))
    }

    /// Restore a graph from checkpoint bytes produced by [`Self::checkpoint`].
    pub fn from_checkpoint(data: &[u8]) -> Result<Self, GradientError> {
        serde_json::from_slice(data)
            .map_err(|e| GradientError::InvalidGradient(format!("Checkpoint deserialization: {e}")))
    }

    /// Return all nodes that list `cid` as one of their input CIDs.
    pub fn find_consumers(&self, cid: &str) -> Vec<&ComputationNode> {
        self.nodes
            .values()
            .filter(|n| n.input_cids.iter().any(|ic| ic == cid))
            .collect()
    }

    /// Return the chain of nodes whose `output_cid` led to `output_cid` being
    /// produced, ordered from earliest producer to the node that emitted it.
    ///
    /// This performs a depth-first traversal of the reverse DAG starting from
    /// the node that owns `output_cid`.
    pub fn provenance_chain(&self, output_cid: &str) -> Vec<&ComputationNode> {
        // Find the node that produced this CID.
        let root = self
            .nodes
            .values()
            .find(|n| n.output_cid.as_deref() == Some(output_cid));

        let Some(root) = root else {
            return Vec::new();
        };

        // Walk backwards through input CIDs using a stack (DFS).
        let mut chain: Vec<&ComputationNode> = Vec::new();
        let mut visited: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut stack: Vec<&ComputationNode> = vec![root];

        while let Some(node) = stack.pop() {
            if !visited.insert(node.id.as_str()) {
                continue;
            }
            chain.push(node);

            for input_cid in &node.input_cids {
                if let Some(parent) = self
                    .nodes
                    .values()
                    .find(|n| n.output_cid.as_deref() == Some(input_cid.as_str()))
                {
                    stack.push(parent);
                }
            }
        }

        // Reverse so the chain runs earliest → latest.
        chain.reverse();
        chain
    }

    /// Immutable access to a node by id.
    pub fn get_node(&self, node_id: &str) -> Option<&ComputationNode> {
        self.nodes.get(node_id)
    }

    /// Number of nodes currently in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges currently in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Iterate over all nodes (order is unspecified).
    pub fn nodes(&self) -> impl Iterator<Item = &ComputationNode> {
        self.nodes.values()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod computation_graph_tests {
    use super::*;

    // Helper: build a linear graph  input → matmul → relu → output
    fn build_linear_graph() -> (ComputationGraphStore, String, String, String, String) {
        let mut store = ComputationGraphStore::new();

        // input node: no inputs, produces cid_a
        let mut input_node = ComputationNode::new("input", vec![]);
        let input_id = input_node.id.clone();
        input_node.output_cid = Some("cid_a".to_string());
        store.add_node(input_node);

        // matmul node: consumes cid_a, produces cid_b
        let mut matmul_node = ComputationNode::new("matmul", vec!["cid_a".to_string()]);
        let matmul_id = matmul_node.id.clone();
        matmul_node.output_cid = Some("cid_b".to_string());
        store.add_node(matmul_node);

        // relu node: consumes cid_b, produces cid_c
        let mut relu_node = ComputationNode::new("relu", vec!["cid_b".to_string()]);
        let relu_id = relu_node.id.clone();
        relu_node.output_cid = Some("cid_c".to_string());
        store.add_node(relu_node);

        // output node: consumes cid_c, produces cid_d
        let mut output_node = ComputationNode::new("output", vec!["cid_c".to_string()]);
        let output_id = output_node.id.clone();
        output_node.output_cid = Some("cid_d".to_string());
        store.add_node(output_node);

        (store, input_id, matmul_id, relu_id, output_id)
    }

    #[test]
    fn test_add_and_retrieve_node() {
        let mut store = ComputationGraphStore::new();

        let node = ComputationNode::new("relu", vec!["cid_x".to_string()])
            .with_meta("dtype", "f32")
            .with_meta("shape", "[128, 64]");

        let node_id = node.id.clone();
        store.add_node(node);

        assert_eq!(store.node_count(), 1);

        let retrieved = store.get_node(&node_id).expect("node should exist");
        assert_eq!(retrieved.op, "relu");
        assert_eq!(retrieved.input_cids, vec!["cid_x".to_string()]);
        assert_eq!(
            retrieved.metadata.get("dtype").map(|s| s.as_str()),
            Some("f32")
        );
        assert!(retrieved.output_cid.is_none());
        assert!(retrieved.gradient_cid.is_none());
    }

    #[test]
    fn test_topological_order() {
        let (store, input_id, matmul_id, relu_id, output_id) = build_linear_graph();

        let order = store.topological_order();

        assert_eq!(order.len(), 4, "all four nodes should appear");

        // Verify ordering constraints: each node precedes its dependents.
        let pos = |id: &str| order.iter().position(|x| x == id).expect("id in order");

        assert!(pos(&input_id) < pos(&matmul_id), "input before matmul");
        assert!(pos(&matmul_id) < pos(&relu_id), "matmul before relu");
        assert!(pos(&relu_id) < pos(&output_id), "relu before output");
    }

    #[test]
    fn test_record_output_and_gradient() {
        let mut store = ComputationGraphStore::new();
        let node = ComputationNode::new("softmax", vec!["cid_in".to_string()]);
        let node_id = node.id.clone();
        store.add_node(node);

        // Record forward-pass output.
        store
            .record_output(&node_id, "cid_out".to_string())
            .expect("test: should succeed");
        assert_eq!(
            store
                .get_node(&node_id)
                .expect("test: should succeed")
                .output_cid
                .as_deref(),
            Some("cid_out")
        );

        // Record backward-pass gradient.
        store
            .record_gradient(&node_id, "cid_grad".to_string())
            .expect("test: should succeed");
        assert_eq!(
            store
                .get_node(&node_id)
                .expect("test: should succeed")
                .gradient_cid
                .as_deref(),
            Some("cid_grad")
        );
    }

    #[test]
    fn test_record_output_missing_node() {
        let mut store = ComputationGraphStore::new();
        let result = store.record_output("nonexistent-id", "cid_out".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_record_gradient_missing_node() {
        let mut store = ComputationGraphStore::new();
        let result = store.record_gradient("nonexistent-id", "cid_grad".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let (store, _, _, _, _) = build_linear_graph();

        let bytes = store.checkpoint().expect("checkpoint serialization");
        let restored =
            ComputationGraphStore::from_checkpoint(&bytes).expect("checkpoint deserialization");

        assert_eq!(restored.node_count(), 4);
        assert_eq!(restored.edge_count(), store.edge_count());
    }

    #[test]
    fn test_gradient_checkpoint_save_load() {
        let (store, _, _, _, _) = build_linear_graph();

        let mut ckpt = super::super::checkpoint::GradientCheckpoint::new(store, 42)
            .with_loss_cid("cid_loss_xyz");
        ckpt.set_optimizer_state("adam_m", vec![1u8, 2, 3]);
        ckpt.set_optimizer_state("adam_v", vec![4u8, 5, 6]);

        // Use a unique temp directory to avoid collisions between parallel test runs.
        let dir = std::env::temp_dir().join(format!("ipfrs_grad_test_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("test: should succeed");
        let path = dir.join("checkpoint.json");

        ckpt.save(&path).expect("save checkpoint");

        let loaded =
            super::super::checkpoint::GradientCheckpoint::load(&path).expect("load checkpoint");

        assert_eq!(loaded.step, 42);
        assert_eq!(loaded.loss_cid.as_deref(), Some("cid_loss_xyz"));
        assert_eq!(
            loaded.optimizer_state.get("adam_m").map(|v| v.as_slice()),
            Some([1u8, 2, 3].as_slice())
        );
        assert_eq!(
            loaded.optimizer_state.get("adam_v").map(|v| v.as_slice()),
            Some([4u8, 5, 6].as_slice())
        );
        assert_eq!(loaded.graph.node_count(), 4);

        // Cleanup.
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_provenance_chain() {
        // Graph: A → B → C  (linear pipeline)
        // A produces cid_a, B consumes cid_a and produces cid_b,
        // C consumes cid_b and produces cid_c.
        let mut store = ComputationGraphStore::new();

        let mut node_a = ComputationNode::new("load", vec![]);
        node_a.output_cid = Some("cid_a".to_string());
        store.add_node(node_a);

        let mut node_b = ComputationNode::new("linear", vec!["cid_a".to_string()]);
        node_b.output_cid = Some("cid_b".to_string());
        store.add_node(node_b);

        let mut node_c = ComputationNode::new("relu", vec!["cid_b".to_string()]);
        node_c.output_cid = Some("cid_c".to_string());
        store.add_node(node_c);

        let chain = store.provenance_chain("cid_c");
        assert_eq!(chain.len(), 3, "chain should include all 3 nodes");

        // The last node in the chain should be the one that produced cid_c.
        assert_eq!(
            chain
                .last()
                .expect("test: should succeed")
                .output_cid
                .as_deref(),
            Some("cid_c")
        );

        // The first node should be the source (no inputs).
        assert!(chain
            .first()
            .expect("test: should succeed")
            .input_cids
            .is_empty());
    }

    #[test]
    fn test_provenance_chain_unknown_cid() {
        let store = ComputationGraphStore::new();
        let chain = store.provenance_chain("unknown_cid");
        assert!(chain.is_empty());
    }

    #[test]
    fn test_find_consumers() {
        let mut store = ComputationGraphStore::new();

        // Two nodes share the same input CID "shared_cid".
        let mut node_a = ComputationNode::new("op_a", vec!["shared_cid".to_string()]);
        node_a.output_cid = Some("cid_out_a".to_string());
        let id_a = node_a.id.clone();

        let mut node_b = ComputationNode::new(
            "op_b",
            vec!["shared_cid".to_string(), "other_cid".to_string()],
        );
        node_b.output_cid = Some("cid_out_b".to_string());
        let id_b = node_b.id.clone();

        // node_c does NOT consume "shared_cid".
        let node_c = ComputationNode::new("op_c", vec!["different_cid".to_string()]);

        store.add_node(node_a);
        store.add_node(node_b);
        store.add_node(node_c);

        let consumers = store.find_consumers("shared_cid");
        assert_eq!(consumers.len(), 2);

        let consumer_ids: Vec<&str> = consumers.iter().map(|n| n.id.as_str()).collect();
        assert!(consumer_ids.contains(&id_a.as_str()));
        assert!(consumer_ids.contains(&id_b.as_str()));

        // Querying for "other_cid" should return only node_b.
        let consumers_other = store.find_consumers("other_cid");
        assert_eq!(consumers_other.len(), 1);
        assert_eq!(consumers_other[0].id, id_b);
    }

    #[test]
    fn test_empty_graph_topological_order() {
        let store = ComputationGraphStore::new();
        let order = store.topological_order();
        assert!(order.is_empty());
    }

    #[test]
    fn test_single_node_graph() {
        let mut store = ComputationGraphStore::new();
        let node = ComputationNode::new("loss", vec![]);
        let id = node.id.clone();
        store.add_node(node);

        let order = store.topological_order();
        assert_eq!(order, vec![id]);
    }

    #[test]
    fn test_graph_store_node_and_edge_counts() {
        let (store, _, _, _, _) = build_linear_graph();
        // 4 nodes, 3 edges (input→matmul, matmul→relu, relu→output)
        assert_eq!(store.node_count(), 4);
        assert_eq!(store.edge_count(), 3);
    }

    // ── Arrow IPC gradient storage tests ─────────────────────────────────────

    #[test]
    fn test_store_and_load_gradient_arrow() {
        let mut graph = ComputationGraphStore::new();
        let node = ComputationNode::new("matmul", vec![]);
        let node_id = node.id.clone();
        graph.add_node(node);

        let grad_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let shape = vec![2usize, 3];

        // Store the gradient
        let cid_str = graph
            .store_gradient_as_arrow(&node_id, &grad_data, &shape)
            .expect("store_gradient_as_arrow");

        // CID must be non-empty and recorded on the node
        assert!(!cid_str.is_empty(), "CID string must not be empty");
        let node = graph.get_node(&node_id).expect("node should exist");
        assert_eq!(node.gradient_cid.as_deref(), Some(cid_str.as_str()));

        // Now encode and decode via Arrow IPC
        use crate::arrow::{ArrowTensor, ArrowTensorStore};
        let tensor = ArrowTensor::from_slice_f32("gradient", shape.clone(), &grad_data);
        let mut store = ArrowTensorStore::new();
        store.insert(tensor);
        let ipc_bytes = store.to_bytes().expect("to_bytes");

        let loaded = ComputationGraphStore::load_gradient_from_arrow(&ipc_bytes, &shape)
            .expect("load_gradient_from_arrow");

        assert_eq!(loaded, grad_data, "Loaded gradient must match original");
    }

    #[test]
    fn test_gradient_shape_preserved() {
        use crate::arrow::{ArrowTensor, ArrowTensorStore};

        // 3-D tensor: shape [2, 3, 4]
        let shape = vec![2usize, 3, 4];
        let numel: usize = shape.iter().product();
        let grad_data: Vec<f32> = (0..numel).map(|i| i as f32 * 0.01).collect();

        let tensor = ArrowTensor::from_slice_f32("gradient", shape.clone(), &grad_data);
        let mut store = ArrowTensorStore::new();
        store.insert(tensor);
        let ipc_bytes = store.to_bytes().expect("to_bytes");

        let loaded = ComputationGraphStore::load_gradient_from_arrow(&ipc_bytes, &shape)
            .expect("load_gradient_from_arrow");

        assert_eq!(loaded.len(), numel, "Element count must be preserved");
        for (i, (&orig, &loaded_val)) in grad_data.iter().zip(loaded.iter()).enumerate() {
            assert!(
                (orig - loaded_val).abs() < 1e-6,
                "Mismatch at index {}: {} vs {}",
                i,
                orig,
                loaded_val
            );
        }
    }

    #[test]
    fn test_gradient_shape_mismatch_error() {
        use crate::arrow::{ArrowTensor, ArrowTensorStore};

        let shape = vec![2usize, 3];
        let grad_data: Vec<f32> = vec![1.0; 6];
        let tensor = ArrowTensor::from_slice_f32("gradient", shape.clone(), &grad_data);
        let mut store = ArrowTensorStore::new();
        store.insert(tensor);
        let ipc_bytes = store.to_bytes().expect("to_bytes");

        // Requesting wrong shape should fail
        let wrong_shape = vec![3usize, 2];
        let result = ComputationGraphStore::load_gradient_from_arrow(&ipc_bytes, &wrong_shape);
        assert!(
            matches!(result, Err(GradientError::ShapeMismatch { .. })),
            "Expected ShapeMismatch error, got {:?}",
            result
        );
    }

    #[test]
    fn test_store_gradient_node_not_found() {
        let mut graph = ComputationGraphStore::new();
        let result = graph.store_gradient_as_arrow("nonexistent-node-id", &[1.0, 2.0], &[2]);
        assert!(result.is_err(), "Should fail for nonexistent node");
    }
}
