//! Graph Neural Network (GNN) with message-passing, edge weighting, and multi-layer propagation.
//!
//! This module implements a full message-passing GNN where nodes aggregate neighbour
//! features, transform them through stacked linear layers (with bias and activation),
//! and iteratively refine node embeddings.
//!
//! # Quick example
//!
//! ```
//! use ipfrs_tensorlogic::graph_neural_network::{
//!     GraphNeuralNetwork, GnnConfig, GnnLayer, GnnActivation, GnnAggregation,
//! };
//!
//! // One hidden layer: 2-dim input -> 4-dim output, ReLU
//! let layer = GnnLayer {
//!     weights: vec![
//!         vec![0.5, -0.3],
//!         vec![-0.1,  0.8],
//!         vec![ 0.2,  0.4],
//!         vec![-0.6,  0.1],
//!     ],
//!     bias: vec![0.0, 0.0, 0.0, 0.0],
//!     activation: GnnActivation::Relu,
//! };
//!
//! let config = GnnConfig {
//!     layers: vec![layer],
//!     aggregation: GnnAggregation::Mean,
//!     num_iterations: 2,
//! };
//!
//! let mut gnn = GraphNeuralNetwork::new(config);
//! let a = gnn.add_node(vec![1.0, 0.0]);
//! let b = gnn.add_node(vec![0.0, 1.0]);
//! gnn.add_edge(a, b, 1.0).expect("example: should succeed in docs");
//!
//! let embeddings = gnn.forward();
//! assert_eq!(embeddings.len(), 2);
//! assert_eq!(embeddings[0].len(), 4);
//! ```

use std::fmt;

// ── Activation enum ──────────────────────────────────────────────────────────

/// Activation function applied element-wise in a [`GnnLayer`].
#[derive(Debug, Clone, PartialEq)]
pub enum GnnActivation {
    /// Rectified Linear Unit: max(0, x)
    Relu,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// Pass-through: f(x) = x
    Linear,
}

impl GnnActivation {
    /// Apply the activation function to a scalar value.
    #[inline]
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            GnnActivation::Relu => x.max(0.0),
            GnnActivation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            GnnActivation::Tanh => x.tanh(),
            GnnActivation::Linear => x,
        }
    }
}

// ── Aggregation enum ─────────────────────────────────────────────────────────

/// Neighbour feature aggregation strategy used during message passing.
///
/// Note: named `GnnAggregation` (not `AggregationMethod`) to avoid collision
/// with the same-named type already exported from the `gradient` module.
#[derive(Debug, Clone, PartialEq)]
pub enum GnnAggregation {
    /// Weighted sum of neighbour features (weight from edge).
    Sum,
    /// Weighted sum divided by total edge weight; falls back to own features if
    /// the node is isolated.
    Mean,
    /// Element-wise maximum across neighbours (unweighted); falls back to own
    /// features if the node is isolated.
    Max,
}

// ── Node / Edge primitives ───────────────────────────────────────────────────

/// Newtype wrapper for node indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GnnNodeId(pub usize);

impl fmt::Display for GnnNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GnnNodeId({})", self.0)
    }
}

/// Feature vector associated with a single node.
#[derive(Debug, Clone)]
pub struct NodeFeatures {
    /// Identifier of the node.
    pub id: GnnNodeId,
    /// Feature vector of dimension D (must be consistent across all nodes).
    pub features: Vec<f64>,
}

/// Directed edge in the graph with an optional scalar weight.
#[derive(Debug, Clone)]
pub struct GnnEdge {
    /// Source node.
    pub from: GnnNodeId,
    /// Target node.
    pub to: GnnNodeId,
    /// Edge weight used for weighted aggregation strategies.
    pub weight: f64,
}

// ── Layer ────────────────────────────────────────────────────────────────────

/// A single linear + bias + activation layer inside the GNN.
///
/// `weights` is a matrix of shape `[output_dim × input_dim]`; each inner
/// `Vec<f64>` is one output row.
#[derive(Debug, Clone)]
pub struct GnnLayer {
    /// Weight matrix: `weights[out_row][in_col]`.
    pub weights: Vec<Vec<f64>>,
    /// Bias vector of length `output_dim`.
    pub bias: Vec<f64>,
    /// Activation applied after the linear transform.
    pub activation: GnnActivation,
}

// ── Config ───────────────────────────────────────────────────────────────────

/// Full configuration for a [`GraphNeuralNetwork`].
#[derive(Debug, Clone)]
pub struct GnnConfig {
    /// Ordered sequence of layers applied to each node's representation.
    pub layers: Vec<GnnLayer>,
    /// Neighbour aggregation strategy.
    pub aggregation: GnnAggregation,
    /// Number of message-passing iterations.
    pub num_iterations: usize,
}

// ── Error ────────────────────────────────────────────────────────────────────

/// Errors that can arise when working with a [`GraphNeuralNetwork`].
#[derive(Debug, Clone, PartialEq)]
pub enum GnnError {
    /// A referenced node index does not exist.
    NodeNotFound(usize),
    /// Layer input/output dimension mismatch.
    DimensionMismatch {
        /// Zero-based layer index.
        layer: usize,
        /// Expected dimension.
        expected: usize,
        /// Actual dimension encountered.
        got: usize,
    },
    /// The graph contains no nodes.
    EmptyGraph,
}

impl fmt::Display for GnnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GnnError::NodeNotFound(id) => write!(f, "node not found: {id}"),
            GnnError::DimensionMismatch {
                layer,
                expected,
                got,
            } => {
                write!(
                    f,
                    "dimension mismatch at layer {layer}: expected {expected}, got {got}"
                )
            }
            GnnError::EmptyGraph => write!(f, "the graph is empty"),
        }
    }
}

impl std::error::Error for GnnError {}

// ── Stats ────────────────────────────────────────────────────────────────────

/// Snapshot of graph and model statistics.
#[derive(Debug, Clone)]
pub struct GnnStats {
    /// Number of nodes currently in the graph.
    pub num_nodes: usize,
    /// Number of directed edges (2 per undirected edge).
    pub num_edges: usize,
    /// Average out-degree across all nodes.
    pub avg_degree: f64,
    /// Dimension of the raw node feature vectors.
    pub feature_dim: usize,
    /// Dimensionality of the final embedding (output of last layer, or feature
    /// dim if there are no layers).
    pub output_dim: usize,
    /// Total number of message-passing rounds executed via [`GraphNeuralNetwork::forward`].
    pub trained_iterations: u64,
}

// ── Main struct ──────────────────────────────────────────────────────────────

/// Message-passing Graph Neural Network.
///
/// Supports dynamic node/edge addition/removal, configurable aggregation
/// strategies, and multi-layer propagation.
pub struct GraphNeuralNetwork {
    /// GNN configuration (layers, aggregation, iterations).
    pub config: GnnConfig,
    /// Node feature vectors; index matches the logical node id.
    pub nodes: Vec<NodeFeatures>,
    /// Adjacency list: `adjacency[node_idx]` = list of `(neighbour_id, weight)`.
    pub adjacency: Vec<Vec<(GnnNodeId, f64)>>,
    /// Cumulative message-passing iterations executed so far.
    pub trained_iterations: u64,
}

impl GraphNeuralNetwork {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Create a new, empty graph neural network with the given configuration.
    pub fn new(config: GnnConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            adjacency: Vec::new(),
            trained_iterations: 0,
        }
    }

    // ── Graph modification ────────────────────────────────────────────────────

    /// Append a new node with the provided feature vector.
    ///
    /// Returns the [`GnnNodeId`] assigned to the new node.
    pub fn add_node(&mut self, features: Vec<f64>) -> GnnNodeId {
        let id = GnnNodeId(self.nodes.len());
        self.nodes.push(NodeFeatures { id, features });
        self.adjacency.push(Vec::new());
        id
    }

    /// Add an undirected edge between `from` and `to` with the given `weight`.
    ///
    /// Both directions are stored so that neighbour aggregation works
    /// transparently for undirected graphs.
    ///
    /// # Errors
    ///
    /// Returns [`GnnError::NodeNotFound`] if either node id is out of bounds.
    pub fn add_edge(
        &mut self,
        from: GnnNodeId,
        to: GnnNodeId,
        weight: f64,
    ) -> Result<(), GnnError> {
        let n = self.nodes.len();
        if from.0 >= n {
            return Err(GnnError::NodeNotFound(from.0));
        }
        if to.0 >= n {
            return Err(GnnError::NodeNotFound(to.0));
        }
        self.adjacency[from.0].push((to, weight));
        self.adjacency[to.0].push((from, weight));
        Ok(())
    }

    /// Remove the node with the given id together with all edges incident to it.
    ///
    /// Returns `true` if the node existed, `false` otherwise.
    ///
    /// **Note**: Removing a node renumbers all nodes with higher indices; any
    /// previously obtained [`GnnNodeId`]s for those nodes become stale.
    pub fn remove_node(&mut self, id: GnnNodeId) -> bool {
        let idx = id.0;
        if idx >= self.nodes.len() {
            return false;
        }

        // Remove the node and its adjacency list.
        self.nodes.remove(idx);
        self.adjacency.remove(idx);

        // Fix-up remaining adjacency lists:
        // 1. Remove any edge that pointed to the deleted node.
        // 2. Decrement indices of nodes that were renumbered (index > idx).
        for adj in self.adjacency.iter_mut() {
            adj.retain(|(nb, _)| nb.0 != idx);
            for (nb, _) in adj.iter_mut() {
                if nb.0 > idx {
                    nb.0 -= 1;
                }
            }
        }

        // Fix node ids stored inside NodeFeatures.
        for (i, node) in self.nodes.iter_mut().enumerate() {
            node.id = GnnNodeId(i);
        }

        true
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /// Run `num_iterations` rounds of message passing and return the final node
    /// embeddings.
    ///
    /// Each iteration:
    /// 1. Every node aggregates its neighbours' current feature vectors using
    ///    the configured [`GnnAggregation`].
    /// 2. The aggregated vector is passed through every [`GnnLayer`] in order.
    ///
    /// Returns one embedding `Vec<f64>` per node in node-index order.
    pub fn forward(&self) -> Vec<Vec<f64>> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        // Initialise embeddings from raw features.
        let mut embeddings: Vec<Vec<f64>> = self.nodes.iter().map(|n| n.features.clone()).collect();

        for _ in 0..self.config.num_iterations {
            let prev = embeddings.clone();
            for (node_idx, emb) in embeddings.iter_mut().enumerate() {
                let agg = self.aggregate_neighbors(GnnNodeId(node_idx), &prev);
                let mut h = agg;
                for layer in &self.config.layers {
                    h = Self::apply_layer(&h, layer);
                }
                *emb = h;
            }
        }

        embeddings
    }

    /// Aggregate the feature vectors of `node_id`'s neighbours according to
    /// the configured aggregation strategy.
    ///
    /// Falls back to the node's own features when it has no neighbours.
    pub fn aggregate_neighbors(&self, node_id: GnnNodeId, features: &[Vec<f64>]) -> Vec<f64> {
        let neighbours = &self.adjacency[node_id.0];
        if neighbours.is_empty() {
            return features[node_id.0].clone();
        }

        let feat_dim = features[node_id.0].len();
        // Guard against empty feature vectors.
        if feat_dim == 0 {
            return Vec::new();
        }

        match &self.config.aggregation {
            GnnAggregation::Sum => {
                let mut acc = vec![0.0f64; feat_dim];
                for &(nb_id, weight) in neighbours {
                    let nb_feat = &features[nb_id.0];
                    let effective_dim = nb_feat.len().min(feat_dim);
                    for i in 0..effective_dim {
                        acc[i] += weight * nb_feat[i];
                    }
                }
                acc
            }
            GnnAggregation::Mean => {
                let mut acc = vec![0.0f64; feat_dim];
                let mut total_weight = 0.0f64;
                for &(nb_id, weight) in neighbours {
                    let nb_feat = &features[nb_id.0];
                    let effective_dim = nb_feat.len().min(feat_dim);
                    for i in 0..effective_dim {
                        acc[i] += weight * nb_feat[i];
                    }
                    total_weight += weight;
                }
                if total_weight.abs() > f64::EPSILON {
                    for v in acc.iter_mut() {
                        *v /= total_weight;
                    }
                } else {
                    // All weights are zero or no neighbours: return own features.
                    return features[node_id.0].clone();
                }
                acc
            }
            GnnAggregation::Max => {
                let mut acc = vec![f64::NEG_INFINITY; feat_dim];
                for &(nb_id, _weight) in neighbours {
                    let nb_feat = &features[nb_id.0];
                    let effective_dim = nb_feat.len().min(feat_dim);
                    for i in 0..effective_dim {
                        if nb_feat[i] > acc[i] {
                            acc[i] = nb_feat[i];
                        }
                    }
                }
                // Replace any remaining -inf (dimension not covered by any neighbour)
                // with the node's own feature value.
                let own = &features[node_id.0];
                for (i, slot) in acc.iter_mut().enumerate() {
                    if *slot == f64::NEG_INFINITY {
                        *slot = *own.get(i).unwrap_or(&0.0);
                    }
                }
                acc
            }
        }
    }

    /// Apply a single layer (linear transform + bias + activation) to `input`.
    ///
    /// `layer.weights` must be `[output_dim × input_dim]`. Rows shorter than
    /// `input` are zero-padded; inputs longer than any row are silently ignored
    /// for that row (safe defaults for heterogeneous graphs).
    pub fn apply_layer(input: &[f64], layer: &GnnLayer) -> Vec<f64> {
        layer
            .weights
            .iter()
            .zip(layer.bias.iter())
            .map(|(row, &b)| {
                let dot: f64 = row.iter().zip(input.iter()).map(|(&w, &x)| w * x).sum();
                layer.activation.apply(dot + b)
            })
            .collect()
    }

    // ── Query helpers ─────────────────────────────────────────────────────────

    /// Compute the embedding for a single node.
    ///
    /// # Errors
    ///
    /// - [`GnnError::EmptyGraph`] if the graph has no nodes.
    /// - [`GnnError::NodeNotFound`] if `id` is out of range.
    pub fn node_embedding(&self, id: GnnNodeId) -> Result<Vec<f64>, GnnError> {
        if self.nodes.is_empty() {
            return Err(GnnError::EmptyGraph);
        }
        if id.0 >= self.nodes.len() {
            return Err(GnnError::NodeNotFound(id.0));
        }
        let embeddings = self.forward();
        Ok(embeddings[id.0].clone())
    }

    /// Compute the graph-level embedding as the element-wise mean of all node
    /// embeddings.
    ///
    /// Returns an empty vector if the graph has no nodes or if the node
    /// embeddings are empty.
    pub fn graph_embedding(&self) -> Vec<f64> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        let embeddings = self.forward();
        if embeddings.is_empty() {
            return Vec::new();
        }
        let dim = embeddings[0].len();
        if dim == 0 {
            return Vec::new();
        }
        let n = embeddings.len() as f64;
        let mut mean = vec![0.0f64; dim];
        for emb in &embeddings {
            for (i, &v) in emb.iter().enumerate() {
                if i < dim {
                    mean[i] += v;
                }
            }
        }
        for v in mean.iter_mut() {
            *v /= n;
        }
        mean
    }

    // ── Graph metadata ────────────────────────────────────────────────────────

    /// Number of nodes in the graph.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of directed edges (2 per undirected edge).
    pub fn num_edges(&self) -> usize {
        self.adjacency.iter().map(|adj| adj.len()).sum()
    }

    /// Return a statistics snapshot for the current graph state.
    pub fn stats(&self) -> GnnStats {
        let num_nodes = self.nodes.len();
        let num_edges = self.num_edges();
        let avg_degree = if num_nodes > 0 {
            num_edges as f64 / num_nodes as f64
        } else {
            0.0
        };
        let feature_dim = self.nodes.first().map(|n| n.features.len()).unwrap_or(0);
        let output_dim = self
            .config
            .layers
            .last()
            .map(|l| l.bias.len())
            .unwrap_or(feature_dim);
        GnnStats {
            num_nodes,
            num_edges,
            avg_degree,
            feature_dim,
            output_dim,
            trained_iterations: self.trained_iterations,
        }
    }
}

// ── PRNG helper (test support) ────────────────────────────────────────────────

/// Xorshift64 pseudo-random number generator for test vector generation.
///
/// This is intentionally not used in production code paths.
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::graph_neural_network::{
        xorshift64, GnnActivation, GnnAggregation, GnnConfig, GnnError, GnnLayer, GnnNodeId,
        GnnStats, GraphNeuralNetwork, NodeFeatures,
    };

    // ── helpers ───────────────────────────────────────────────────────────────

    fn identity_layer(dim: usize) -> GnnLayer {
        let weights: Vec<Vec<f64>> = (0..dim)
            .map(|i| {
                let mut row = vec![0.0f64; dim];
                row[i] = 1.0;
                row
            })
            .collect();
        GnnLayer {
            weights,
            bias: vec![0.0; dim],
            activation: GnnActivation::Linear,
        }
    }

    fn linear_config(dim: usize, iters: usize) -> GnnConfig {
        GnnConfig {
            layers: vec![identity_layer(dim)],
            aggregation: GnnAggregation::Mean,
            num_iterations: iters,
        }
    }

    fn two_node_gnn() -> GraphNeuralNetwork {
        let config = linear_config(2, 1);
        let mut gnn = GraphNeuralNetwork::new(config);
        gnn.add_node(vec![1.0, 0.0]);
        gnn.add_node(vec![0.0, 1.0]);
        gnn.add_edge(GnnNodeId(0), GnnNodeId(1), 1.0)
            .expect("test: should succeed");
        gnn
    }

    fn relu_layer(in_dim: usize, out_dim: usize) -> GnnLayer {
        // Use xorshift64 to generate pseudo-random weights in [-0.5, 0.5].
        let mut state: u64 = 0xDEAD_BEEF_1234_5678;
        let weights: Vec<Vec<f64>> = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| {
                        let r = xorshift64(&mut state);
                        (r as f64 / u64::MAX as f64) - 0.5
                    })
                    .collect()
            })
            .collect();
        GnnLayer {
            weights,
            bias: vec![0.0; out_dim],
            activation: GnnActivation::Relu,
        }
    }

    // ── Test 1: new() creates an empty graph ──────────────────────────────────

    #[test]
    fn test_new_empty_graph() {
        let config = GnnConfig {
            layers: vec![],
            aggregation: GnnAggregation::Mean,
            num_iterations: 1,
        };
        let gnn = GraphNeuralNetwork::new(config);
        assert_eq!(gnn.num_nodes(), 0);
        assert_eq!(gnn.num_edges(), 0);
    }

    // ── Test 2: add_node increments count ────────────────────────────────────

    #[test]
    fn test_add_node_increments_count() {
        let mut gnn = GraphNeuralNetwork::new(linear_config(3, 1));
        let id0 = gnn.add_node(vec![1.0, 2.0, 3.0]);
        let id1 = gnn.add_node(vec![4.0, 5.0, 6.0]);
        assert_eq!(id0.0, 0);
        assert_eq!(id1.0, 1);
        assert_eq!(gnn.num_nodes(), 2);
    }

    // ── Test 3: add_edge validates source node ────────────────────────────────

    #[test]
    fn test_add_edge_invalid_source() {
        let mut gnn = GraphNeuralNetwork::new(linear_config(2, 1));
        gnn.add_node(vec![1.0, 0.0]);
        let result = gnn.add_edge(GnnNodeId(5), GnnNodeId(0), 1.0);
        assert_eq!(result, Err(GnnError::NodeNotFound(5)));
    }

    // ── Test 4: add_edge validates destination node ───────────────────────────

    #[test]
    fn test_add_edge_invalid_dest() {
        let mut gnn = GraphNeuralNetwork::new(linear_config(2, 1));
        gnn.add_node(vec![1.0, 0.0]);
        let result = gnn.add_edge(GnnNodeId(0), GnnNodeId(99), 1.0);
        assert_eq!(result, Err(GnnError::NodeNotFound(99)));
    }

    // ── Test 5: add_edge is undirected ────────────────────────────────────────

    #[test]
    fn test_add_edge_undirected() {
        let gnn = two_node_gnn();
        // Each undirected edge contributes 2 directed entries.
        assert_eq!(gnn.num_edges(), 2);
    }

    // ── Test 6: remove_node returns false for missing node ────────────────────

    #[test]
    fn test_remove_node_missing() {
        let mut gnn = two_node_gnn();
        assert!(!gnn.remove_node(GnnNodeId(99)));
    }

    // ── Test 7: remove_node removes node and edges ───────────────────────────

    #[test]
    fn test_remove_node_cleans_edges() {
        let mut gnn = two_node_gnn();
        assert!(gnn.remove_node(GnnNodeId(0)));
        assert_eq!(gnn.num_nodes(), 1);
        // The remaining node's adjacency list should be empty.
        assert_eq!(gnn.num_edges(), 0);
    }

    // ── Test 8: remove_node renumbers correctly ───────────────────────────────

    #[test]
    fn test_remove_node_renumbers() {
        let mut gnn = GraphNeuralNetwork::new(linear_config(2, 1));
        gnn.add_node(vec![1.0, 0.0]);
        gnn.add_node(vec![0.0, 1.0]);
        gnn.add_node(vec![1.0, 1.0]);
        gnn.remove_node(GnnNodeId(0));
        // Previous node 1 is now node 0, previous node 2 is now node 1.
        assert_eq!(gnn.nodes[0].id, GnnNodeId(0));
        assert_eq!(gnn.nodes[1].id, GnnNodeId(1));
    }

    // ── Test 9: forward on empty graph returns empty vec ─────────────────────

    #[test]
    fn test_forward_empty_graph() {
        let gnn = GraphNeuralNetwork::new(linear_config(2, 1));
        let out = gnn.forward();
        assert!(out.is_empty());
    }

    // ── Test 10: forward with identity layer preserves features ──────────────

    #[test]
    fn test_forward_identity_layer_single_node() {
        let mut gnn = GraphNeuralNetwork::new(linear_config(3, 2));
        gnn.add_node(vec![1.0, 2.0, 3.0]);
        let out = gnn.forward();
        assert_eq!(out.len(), 1);
        // Isolated node: aggregation returns own features, identity preserves them.
        assert!((out[0][0] - 1.0).abs() < 1e-9);
        assert!((out[0][1] - 2.0).abs() < 1e-9);
        assert!((out[0][2] - 3.0).abs() < 1e-9);
    }

    // ── Test 11: forward output dimension matches layer output ────────────────

    #[test]
    fn test_forward_output_dimension() {
        let layer = relu_layer(3, 5);
        let config = GnnConfig {
            layers: vec![layer],
            aggregation: GnnAggregation::Sum,
            num_iterations: 1,
        };
        let mut gnn = GraphNeuralNetwork::new(config);
        gnn.add_node(vec![1.0, 2.0, 3.0]);
        let out = gnn.forward();
        assert_eq!(out[0].len(), 5);
    }

    // ── Test 12: aggregate_neighbors Mean with equal weights ──────────────────

    #[test]
    fn test_aggregate_mean_equal_weights() {
        let gnn = two_node_gnn();
        let features = vec![vec![2.0, 4.0], vec![6.0, 8.0]];
        // Mean of node-1's features with weight 1 → [6, 8]
        let agg = gnn.aggregate_neighbors(GnnNodeId(0), &features);
        assert!((agg[0] - 6.0).abs() < 1e-9);
        assert!((agg[1] - 8.0).abs() < 1e-9);
    }

    // ── Test 13: aggregate_neighbors Sum ─────────────────────────────────────

    #[test]
    fn test_aggregate_sum() {
        let config = GnnConfig {
            layers: vec![],
            aggregation: GnnAggregation::Sum,
            num_iterations: 1,
        };
        let mut gnn = GraphNeuralNetwork::new(config);
        gnn.add_node(vec![1.0, 0.0]);
        gnn.add_node(vec![3.0, 4.0]);
        gnn.add_edge(GnnNodeId(0), GnnNodeId(1), 2.0)
            .expect("test: should succeed");

        let features = vec![vec![1.0, 0.0], vec![3.0, 4.0]];
        let agg = gnn.aggregate_neighbors(GnnNodeId(0), &features);
        // weight=2.0, nb features=[3,4] → [6, 8]
        assert!((agg[0] - 6.0).abs() < 1e-9);
        assert!((agg[1] - 8.0).abs() < 1e-9);
    }

    // ── Test 14: aggregate_neighbors Max ─────────────────────────────────────

    #[test]
    fn test_aggregate_max() {
        let config = GnnConfig {
            layers: vec![],
            aggregation: GnnAggregation::Max,
            num_iterations: 1,
        };
        let mut gnn = GraphNeuralNetwork::new(config);
        gnn.add_node(vec![1.0, 5.0]);
        gnn.add_node(vec![3.0, 2.0]);
        gnn.add_node(vec![0.5, 7.0]);
        gnn.add_edge(GnnNodeId(0), GnnNodeId(1), 1.0)
            .expect("test: should succeed");
        gnn.add_edge(GnnNodeId(0), GnnNodeId(2), 1.0)
            .expect("test: should succeed");

        let features = vec![vec![1.0, 5.0], vec![3.0, 2.0], vec![0.5, 7.0]];
        let agg = gnn.aggregate_neighbors(GnnNodeId(0), &features);
        // Element-wise max of [3,2] and [0.5,7] → [3, 7]
        assert!((agg[0] - 3.0).abs() < 1e-9);
        assert!((agg[1] - 7.0).abs() < 1e-9);
    }

    // ── Test 15: isolated node returns own features in Mean ───────────────────

    #[test]
    fn test_isolated_node_mean_returns_own() {
        let config = GnnConfig {
            layers: vec![],
            aggregation: GnnAggregation::Mean,
            num_iterations: 1,
        };
        let mut gnn = GraphNeuralNetwork::new(config);
        gnn.add_node(vec![9.0, 8.0]);
        let features = vec![vec![9.0, 8.0]];
        let agg = gnn.aggregate_neighbors(GnnNodeId(0), &features);
        assert!((agg[0] - 9.0).abs() < 1e-9);
        assert!((agg[1] - 8.0).abs() < 1e-9);
    }

    // ── Test 16: apply_layer – linear (no activation) ────────────────────────

    #[test]
    fn test_apply_layer_linear() {
        let layer = GnnLayer {
            weights: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            bias: vec![0.5, -0.5],
            activation: GnnActivation::Linear,
        };
        let input = vec![1.0, 1.0];
        let out = GraphNeuralNetwork::apply_layer(&input, &layer);
        // Row 0: 1*1 + 2*1 + 0.5 = 3.5
        // Row 1: 3*1 + 4*1 - 0.5 = 6.5
        assert!((out[0] - 3.5).abs() < 1e-9);
        assert!((out[1] - 6.5).abs() < 1e-9);
    }

    // ── Test 17: apply_layer – ReLU clips negatives ───────────────────────────

    #[test]
    fn test_apply_layer_relu() {
        let layer = GnnLayer {
            weights: vec![vec![-1.0, 0.0], vec![1.0, 0.0]],
            bias: vec![0.0, 0.0],
            activation: GnnActivation::Relu,
        };
        let input = vec![1.0, 0.0];
        let out = GraphNeuralNetwork::apply_layer(&input, &layer);
        assert!((out[0] - 0.0).abs() < 1e-9); // -1 → clipped to 0
        assert!((out[1] - 1.0).abs() < 1e-9);
    }

    // ── Test 18: apply_layer – Sigmoid output approaches (0,1) ──────────────

    #[test]
    fn test_apply_layer_sigmoid_range() {
        let layer = GnnLayer {
            weights: vec![vec![1.0]],
            bias: vec![0.0],
            activation: GnnActivation::Sigmoid,
        };
        // Sigmoid of large negative value should be very close to 0 but ≥ 0.
        let out_neg = GraphNeuralNetwork::apply_layer(&[-20.0], &layer);
        assert!(out_neg[0] >= 0.0 && out_neg[0] < 0.01);

        // Sigmoid of large positive value should be very close to 1 but ≤ 1.
        let out_pos = GraphNeuralNetwork::apply_layer(&[20.0], &layer);
        assert!(out_pos[0] > 0.99 && out_pos[0] <= 1.0);

        // Check symmetry around 0.
        let mid = GraphNeuralNetwork::apply_layer(&[0.0], &layer);
        assert!((mid[0] - 0.5).abs() < 1e-9);
    }

    // ── Test 19: apply_layer – Tanh output approaches (-1,1) ─────────────────

    #[test]
    fn test_apply_layer_tanh_range() {
        let layer = GnnLayer {
            weights: vec![vec![1.0]],
            bias: vec![0.0],
            activation: GnnActivation::Tanh,
        };
        // Tanh of large negative should be very close to -1 but ≥ -1.
        let out_neg = GraphNeuralNetwork::apply_layer(&[-20.0], &layer);
        assert!(out_neg[0] >= -1.0 && out_neg[0] < -0.99);

        let out_zero = GraphNeuralNetwork::apply_layer(&[0.0], &layer);
        assert!((out_zero[0]).abs() < 1e-9);

        // Tanh of large positive should be very close to 1 but ≤ 1.
        let out_pos = GraphNeuralNetwork::apply_layer(&[20.0], &layer);
        assert!(out_pos[0] > 0.99 && out_pos[0] <= 1.0);
    }

    // ── Test 20: node_embedding returns correct node ──────────────────────────

    #[test]
    fn test_node_embedding_valid() {
        let gnn = two_node_gnn();
        let emb = gnn.node_embedding(GnnNodeId(0));
        assert!(emb.is_ok());
        assert_eq!(emb.expect("test: should succeed").len(), 2);
    }

    // ── Test 21: node_embedding errors on missing node ────────────────────────

    #[test]
    fn test_node_embedding_not_found() {
        let gnn = two_node_gnn();
        let result = gnn.node_embedding(GnnNodeId(10));
        assert_eq!(result, Err(GnnError::NodeNotFound(10)));
    }

    // ── Test 22: node_embedding errors on empty graph ─────────────────────────

    #[test]
    fn test_node_embedding_empty_graph() {
        let gnn = GraphNeuralNetwork::new(linear_config(2, 1));
        let result = gnn.node_embedding(GnnNodeId(0));
        assert_eq!(result, Err(GnnError::EmptyGraph));
    }

    // ── Test 23: graph_embedding returns empty vec for empty graph ────────────

    #[test]
    fn test_graph_embedding_empty() {
        let gnn = GraphNeuralNetwork::new(linear_config(2, 1));
        let emb = gnn.graph_embedding();
        assert!(emb.is_empty());
    }

    // ── Test 24: graph_embedding dimension matches layer output ───────────────

    #[test]
    fn test_graph_embedding_dimension() {
        let gnn = two_node_gnn();
        let emb = gnn.graph_embedding();
        assert_eq!(emb.len(), 2);
    }

    // ── Test 25: graph_embedding is mean of node embeddings ───────────────────

    #[test]
    fn test_graph_embedding_is_mean() {
        let config = GnnConfig {
            layers: vec![],
            aggregation: GnnAggregation::Mean,
            num_iterations: 0,
        };
        let mut gnn = GraphNeuralNetwork::new(config);
        gnn.add_node(vec![2.0, 4.0]);
        gnn.add_node(vec![6.0, 8.0]);

        // With 0 iterations the embeddings stay as raw features.
        let emb = gnn.graph_embedding();
        assert!((emb[0] - 4.0).abs() < 1e-9);
        assert!((emb[1] - 6.0).abs() < 1e-9);
    }

    // ── Test 26: num_edges counts correctly ───────────────────────────────────

    #[test]
    fn test_num_edges() {
        let mut gnn = GraphNeuralNetwork::new(linear_config(2, 1));
        let a = gnn.add_node(vec![1.0, 0.0]);
        let b = gnn.add_node(vec![0.0, 1.0]);
        let c = gnn.add_node(vec![1.0, 1.0]);
        gnn.add_edge(a, b, 1.0).expect("test: should succeed");
        gnn.add_edge(b, c, 1.0).expect("test: should succeed");
        // 2 undirected edges → 4 directed entries.
        assert_eq!(gnn.num_edges(), 4);
    }

    // ── Test 27: stats reflects correct values ────────────────────────────────

    #[test]
    fn test_stats() {
        let mut gnn = GraphNeuralNetwork::new(linear_config(2, 1));
        let a = gnn.add_node(vec![1.0, 0.0]);
        let b = gnn.add_node(vec![0.0, 1.0]);
        gnn.add_edge(a, b, 1.0).expect("test: should succeed");
        let s: GnnStats = gnn.stats();
        assert_eq!(s.num_nodes, 2);
        assert_eq!(s.num_edges, 2);
        assert!((s.avg_degree - 1.0).abs() < 1e-9);
        assert_eq!(s.feature_dim, 2);
        assert_eq!(s.output_dim, 2);
    }

    // ── Test 28: GnnError::DimensionMismatch display ──────────────────────────

    #[test]
    fn test_gnn_error_display() {
        let e = GnnError::DimensionMismatch {
            layer: 0,
            expected: 4,
            got: 3,
        };
        let s = format!("{e}");
        assert!(s.contains("dimension mismatch"));
        assert!(s.contains("layer 0"));
    }

    // ── Test 29: GnnNodeId display ────────────────────────────────────────────

    #[test]
    fn test_gnn_node_id_display() {
        let id = GnnNodeId(42);
        assert_eq!(format!("{id}"), "GnnNodeId(42)");
    }

    // ── Test 30: multi-layer forward propagation ──────────────────────────────

    #[test]
    fn test_multi_layer_forward() {
        // Two stacked layers: 2→4→2
        let l1 = relu_layer(2, 4);
        let l2 = GnnLayer {
            weights: vec![vec![0.25, 0.25, 0.25, 0.25], vec![0.25, 0.25, 0.25, 0.25]],
            bias: vec![0.0, 0.0],
            activation: GnnActivation::Linear,
        };
        let config = GnnConfig {
            layers: vec![l1, l2],
            aggregation: GnnAggregation::Mean,
            num_iterations: 2,
        };
        let mut gnn = GraphNeuralNetwork::new(config);
        let a = gnn.add_node(vec![1.0, 0.0]);
        let b = gnn.add_node(vec![0.0, 1.0]);
        gnn.add_edge(a, b, 1.0).expect("test: should succeed");
        let out = gnn.forward();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 2);
        assert_eq!(out[1].len(), 2);
    }

    // ── Test 31: xorshift64 never returns zero ────────────────────────────────

    #[test]
    fn test_xorshift64_nonzero() {
        let mut state = 1u64;
        for _ in 0..10_000 {
            let v = xorshift64(&mut state);
            assert_ne!(v, 0);
        }
    }

    // ── Test 32: xorshift64 state advances ────────────────────────────────────

    #[test]
    fn test_xorshift64_state_advances() {
        let mut state = 0xABCD_1234u64;
        let v1 = xorshift64(&mut state);
        let v2 = xorshift64(&mut state);
        assert_ne!(v1, v2);
    }

    // ── Test 33: GnnActivation::apply covers all variants ────────────────────

    #[test]
    fn test_activation_apply_all_variants() {
        assert!((GnnActivation::Relu.apply(-5.0) - 0.0).abs() < 1e-9);
        assert!((GnnActivation::Relu.apply(3.0) - 3.0).abs() < 1e-9);

        let sig = GnnActivation::Sigmoid.apply(0.0);
        assert!((sig - 0.5).abs() < 1e-9);

        let t = GnnActivation::Tanh.apply(0.0);
        assert!(t.abs() < 1e-9);

        assert!((GnnActivation::Linear.apply(7.77) - 7.77).abs() < 1e-9);
    }

    // ── Test 34: zero-weight Mean falls back to own features ─────────────────

    #[test]
    fn test_mean_zero_weight_fallback() {
        let config = GnnConfig {
            layers: vec![],
            aggregation: GnnAggregation::Mean,
            num_iterations: 1,
        };
        let mut gnn = GraphNeuralNetwork::new(config);
        gnn.add_node(vec![3.0, 7.0]);
        gnn.add_node(vec![1.0, 2.0]);
        // Add edge with zero weight.
        gnn.add_edge(GnnNodeId(0), GnnNodeId(1), 0.0)
            .expect("test: should succeed");

        let features = vec![vec![3.0, 7.0], vec![1.0, 2.0]];
        let agg = gnn.aggregate_neighbors(GnnNodeId(0), &features);
        // Total weight = 0 → fallback to own features [3, 7]
        assert!((agg[0] - 3.0).abs() < 1e-9);
        assert!((agg[1] - 7.0).abs() < 1e-9);
    }

    // ── Test 35: NodeFeatures struct accessible ───────────────────────────────

    #[test]
    fn test_node_features_accessible() {
        let nf = NodeFeatures {
            id: GnnNodeId(3),
            features: vec![0.1, 0.2, 0.3],
        };
        assert_eq!(nf.id.0, 3);
        assert!((nf.features[2] - 0.3).abs() < 1e-9);
    }

    // ── Test 36: large graph with random features ─────────────────────────────

    #[test]
    fn test_large_graph_forward() {
        let layer = relu_layer(4, 8);
        let config = GnnConfig {
            layers: vec![layer],
            aggregation: GnnAggregation::Sum,
            num_iterations: 3,
        };
        let mut gnn = GraphNeuralNetwork::new(config);
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;

        // 20 nodes with random 4-dim features.
        for _ in 0..20 {
            let features: Vec<f64> = (0..4)
                .map(|_| (xorshift64(&mut state) as f64 / u64::MAX as f64) * 2.0 - 1.0)
                .collect();
            gnn.add_node(features);
        }

        // Ring topology.
        for i in 0..20 {
            gnn.add_edge(GnnNodeId(i), GnnNodeId((i + 1) % 20), 1.0)
                .expect("test: should succeed");
        }

        let out = gnn.forward();
        assert_eq!(out.len(), 20);
        for emb in &out {
            assert_eq!(emb.len(), 8);
            // ReLU → all outputs ≥ 0
            for &v in emb {
                assert!(v >= 0.0);
            }
        }
    }

    // ── Test 37: remove_node then forward still works ─────────────────────────

    #[test]
    fn test_forward_after_remove_node() {
        let mut gnn = GraphNeuralNetwork::new(linear_config(2, 1));
        gnn.add_node(vec![1.0, 0.0]);
        gnn.add_node(vec![0.0, 1.0]);
        gnn.add_node(vec![0.5, 0.5]);
        gnn.add_edge(GnnNodeId(0), GnnNodeId(1), 1.0)
            .expect("test: should succeed");
        gnn.add_edge(GnnNodeId(1), GnnNodeId(2), 1.0)
            .expect("test: should succeed");
        gnn.remove_node(GnnNodeId(1));
        assert_eq!(gnn.num_nodes(), 2);
        // Both remaining nodes are now isolated.
        assert_eq!(gnn.num_edges(), 0);
        let out = gnn.forward();
        assert_eq!(out.len(), 2);
    }

    // ── Test 38: graph with self-loop (both endpoints identical) ─────────────

    #[test]
    fn test_self_loop_treated_as_edge() {
        let config = GnnConfig {
            layers: vec![],
            aggregation: GnnAggregation::Sum,
            num_iterations: 1,
        };
        let mut gnn = GraphNeuralNetwork::new(config);
        gnn.add_node(vec![2.0, 3.0]);
        // Self-loop: from == to; add_edge should succeed.
        let res = gnn.add_edge(GnnNodeId(0), GnnNodeId(0), 1.0);
        assert!(res.is_ok());
    }

    // ── Test 39: trained_iterations starts at zero ────────────────────────────

    #[test]
    fn test_trained_iterations_initial() {
        let gnn = GraphNeuralNetwork::new(linear_config(2, 1));
        assert_eq!(gnn.trained_iterations, 0);
        let s = gnn.stats();
        assert_eq!(s.trained_iterations, 0);
    }

    // ── Test 40: GnnError::NodeNotFound display ───────────────────────────────

    #[test]
    fn test_gnn_error_node_not_found_display() {
        let e = GnnError::NodeNotFound(7);
        let s = format!("{e}");
        assert!(s.contains("7"));
    }

    // ── Test 41: GnnError::EmptyGraph display ─────────────────────────────────

    #[test]
    fn test_gnn_error_empty_graph_display() {
        let e = GnnError::EmptyGraph;
        let s = format!("{e}");
        assert!(s.contains("empty"));
    }

    // ── Test 42: GnnError implements std::error::Error ───────────────────────

    #[test]
    fn test_gnn_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(GnnError::EmptyGraph);
        assert!(!e.to_string().is_empty());
    }

    // ── Test 43: zero iterations forward returns raw features ─────────────────

    #[test]
    fn test_zero_iterations_returns_raw_features() {
        let config = GnnConfig {
            layers: vec![],
            aggregation: GnnAggregation::Mean,
            num_iterations: 0,
        };
        let mut gnn = GraphNeuralNetwork::new(config);
        gnn.add_node(vec![5.0, 6.0]);
        gnn.add_node(vec![7.0, 8.0]);
        gnn.add_edge(GnnNodeId(0), GnnNodeId(1), 1.0)
            .expect("test: should succeed");
        let out = gnn.forward();
        // With 0 iterations embeddings equal raw features.
        assert!((out[0][0] - 5.0).abs() < 1e-9);
        assert!((out[1][1] - 8.0).abs() < 1e-9);
    }

    // ── Test 44: weighted edge affects Mean aggregation proportionally ─────────

    #[test]
    fn test_weighted_edge_mean() {
        let config = GnnConfig {
            layers: vec![],
            aggregation: GnnAggregation::Mean,
            num_iterations: 1,
        };
        let mut gnn = GraphNeuralNetwork::new(config);
        gnn.add_node(vec![0.0]);
        gnn.add_node(vec![2.0]);
        gnn.add_node(vec![8.0]);
        // Node 0 has two neighbours: node 1 with weight 1, node 2 with weight 3.
        gnn.add_edge(GnnNodeId(0), GnnNodeId(1), 1.0)
            .expect("test: should succeed");
        gnn.add_edge(GnnNodeId(0), GnnNodeId(2), 3.0)
            .expect("test: should succeed");

        let features = vec![vec![0.0], vec![2.0], vec![8.0]];
        let agg = gnn.aggregate_neighbors(GnnNodeId(0), &features);
        // Weighted mean: (1*2 + 3*8) / (1+3) = 26/4 = 6.5
        assert!((agg[0] - 6.5).abs() < 1e-9);
    }

    // ── Test 45: graph_embedding single node equals its embedding ─────────────

    #[test]
    fn test_graph_embedding_single_node() {
        let config = GnnConfig {
            layers: vec![],
            aggregation: GnnAggregation::Mean,
            num_iterations: 0,
        };
        let mut gnn = GraphNeuralNetwork::new(config);
        gnn.add_node(vec![3.0, 9.0]);
        let ge = gnn.graph_embedding();
        let ne = gnn
            .node_embedding(GnnNodeId(0))
            .expect("test: should succeed");
        for (g, n) in ge.iter().zip(ne.iter()) {
            assert!((g - n).abs() < 1e-9);
        }
    }

    // ── Test 46: stats output_dim reflects last layer ─────────────────────────

    #[test]
    fn test_stats_output_dim_last_layer() {
        let l1 = relu_layer(2, 16);
        let l2 = relu_layer(16, 4);
        let config = GnnConfig {
            layers: vec![l1, l2],
            aggregation: GnnAggregation::Mean,
            num_iterations: 1,
        };
        let mut gnn = GraphNeuralNetwork::new(config);
        gnn.add_node(vec![1.0, 2.0]);
        let s = gnn.stats();
        assert_eq!(s.output_dim, 4);
    }
}
