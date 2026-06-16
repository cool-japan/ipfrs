//! Computation graph storage and execution
//!
//! This module provides:
//! - IPLD schema for computation graphs
//! - Graph serialization and deserialization
//! - Graph optimization (CSE, constant folding, fusion)
//! - Lazy evaluation with memoization
//! - Parallel execution support
//! - Streaming execution with backpressure
//! - Distributed graph execution

use ipfrs_core::Cid;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use thiserror::Error;

/// Errors that can occur during graph operations
#[derive(Debug, Error)]
pub enum GraphError {
    #[error("Node not found: {0}")]
    NodeNotFound(String),

    #[error("Circular dependency detected")]
    CircularDependency,

    #[error("Invalid graph structure: {0}")]
    InvalidGraph(String),

    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },

    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    #[error("Missing input: {0}")]
    MissingInput(String),

    #[error("Execution error: {0}")]
    ExecutionError(String),
}

/// Tensor operation types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TensorOp {
    /// Input placeholder
    Input { name: String },

    /// Constant tensor
    Constant { value_cid: String },

    /// Matrix multiplication
    MatMul,

    /// Element-wise addition
    Add,

    /// Element-wise multiplication
    Mul,

    /// Element-wise subtraction
    Sub,

    /// Element-wise division
    Div,

    /// Einsum operation with subscript notation
    Einsum { subscripts: String },

    /// Reshape operation
    Reshape { shape: Vec<i64> },

    /// Transpose operation
    Transpose { axes: Vec<usize> },

    /// Reduce sum along axes
    ReduceSum { axes: Vec<usize>, keepdims: bool },

    /// Reduce mean along axes
    ReduceMean { axes: Vec<usize>, keepdims: bool },

    /// Activation: ReLU
    ReLU,

    /// Activation: Tanh
    Tanh,

    /// Activation: Sigmoid
    Sigmoid,

    /// Activation: GELU (Gaussian Error Linear Unit)
    GELU,

    /// Activation: Softmax along axis
    Softmax { axis: i64 },

    /// Layer normalization
    LayerNorm {
        normalized_shape: Vec<usize>,
        eps: f64,
    },

    /// Batch normalization
    BatchNorm { eps: f64, momentum: f64 },

    /// Dropout (training mode)
    Dropout { p: f64 },

    /// Element-wise exponential
    Exp,

    /// Element-wise logarithm
    Log,

    /// Element-wise power
    Pow { exponent: f64 },

    /// Element-wise square root
    Sqrt,

    /// Concatenate tensors along axis
    Concat { axis: usize },

    /// Split tensor along axis
    Split { axis: usize, sections: Vec<usize> },

    /// Gather elements along axis
    Gather { axis: usize },

    /// Scatter elements along axis
    Scatter { axis: usize },

    /// Slice tensor
    Slice {
        start: Vec<i64>,
        end: Vec<i64>,
        strides: Vec<i64>,
    },

    /// Pad tensor
    Pad {
        padding: Vec<(usize, usize)>,
        mode: String,
    },

    // Fused operations for performance
    /// Fused MatMul + Add (common in linear layers)
    FusedLinear,

    /// Fused Add + ReLU
    FusedAddReLU,

    /// Fused BatchNorm + ReLU
    FusedBatchNormReLU { eps: f64, momentum: f64 },

    /// Fused LayerNorm + Dropout
    FusedLayerNormDropout {
        normalized_shape: Vec<usize>,
        eps: f64,
        dropout_p: f64,
    },
}

impl TensorOp {
    /// Get the number of inputs required by this operation
    pub fn num_inputs(&self) -> usize {
        match self {
            TensorOp::Input { .. } | TensorOp::Constant { .. } => 0,
            TensorOp::ReLU
            | TensorOp::Tanh
            | TensorOp::Sigmoid
            | TensorOp::GELU
            | TensorOp::Softmax { .. }
            | TensorOp::LayerNorm { .. }
            | TensorOp::BatchNorm { .. }
            | TensorOp::Dropout { .. }
            | TensorOp::Exp
            | TensorOp::Log
            | TensorOp::Pow { .. }
            | TensorOp::Sqrt
            | TensorOp::Reshape { .. }
            | TensorOp::Transpose { .. }
            | TensorOp::ReduceSum { .. }
            | TensorOp::ReduceMean { .. }
            | TensorOp::Slice { .. }
            | TensorOp::Pad { .. } => 1,
            TensorOp::MatMul
            | TensorOp::Add
            | TensorOp::Mul
            | TensorOp::Sub
            | TensorOp::Div
            | TensorOp::Gather { .. }
            | TensorOp::Scatter { .. }
            | TensorOp::FusedAddReLU => 2,
            TensorOp::Einsum { .. } => 2, // Simplified for now
            TensorOp::Concat { .. } | TensorOp::Split { .. } => 1, // Variadic, but simplified
            TensorOp::FusedLinear => 3,   // input, weight, bias
            TensorOp::FusedBatchNormReLU { .. } => 1,
            TensorOp::FusedLayerNormDropout { .. } => 1,
        }
    }

    /// Check if this is a pure operation (no side effects)
    pub fn is_pure(&self) -> bool {
        true // All current ops are pure
    }

    /// Infer output shape from input shapes
    pub fn infer_output_shape(
        &self,
        input_shapes: &[Vec<usize>],
    ) -> Result<Vec<usize>, GraphError> {
        match self {
            TensorOp::Input { .. } | TensorOp::Constant { .. } => Err(GraphError::InvalidGraph(
                "Cannot infer shape for input/constant nodes without explicit shape".to_string(),
            )),
            // Unary element-wise operations preserve shape
            TensorOp::ReLU
            | TensorOp::Tanh
            | TensorOp::Sigmoid
            | TensorOp::GELU
            | TensorOp::Exp
            | TensorOp::Log
            | TensorOp::Sqrt
            | TensorOp::Dropout { .. } => {
                if input_shapes.is_empty() {
                    return Err(GraphError::MissingInput(
                        "No input shapes provided".to_string(),
                    ));
                }
                Ok(input_shapes[0].clone())
            }
            // Binary element-wise operations (broadcasting rules apply)
            TensorOp::Add | TensorOp::Mul | TensorOp::Sub | TensorOp::Div => {
                if input_shapes.len() < 2 {
                    return Err(GraphError::MissingInput(
                        "Binary operation requires 2 inputs".to_string(),
                    ));
                }
                Self::broadcast_shapes(&input_shapes[0], &input_shapes[1])
            }
            TensorOp::MatMul => {
                if input_shapes.len() < 2 {
                    return Err(GraphError::MissingInput(
                        "MatMul requires 2 inputs".to_string(),
                    ));
                }
                let a = &input_shapes[0];
                let b = &input_shapes[1];
                if a.len() < 2 || b.len() < 2 {
                    return Err(GraphError::ShapeMismatch(
                        "MatMul requires at least 2D tensors".to_string(),
                    ));
                }
                let m = a[a.len() - 2];
                let k1 = a[a.len() - 1];
                let k2 = b[b.len() - 2];
                let n = b[b.len() - 1];
                if k1 != k2 {
                    return Err(GraphError::ShapeMismatch(format!(
                        "MatMul dimension mismatch: {} vs {}",
                        k1, k2
                    )));
                }
                let mut result = a[..a.len() - 2].to_vec();
                result.push(m);
                result.push(n);
                Ok(result)
            }
            TensorOp::Reshape { shape } => {
                let new_shape: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
                Ok(new_shape)
            }
            TensorOp::Transpose { axes } => {
                if input_shapes.is_empty() {
                    return Err(GraphError::MissingInput(
                        "No input shapes provided".to_string(),
                    ));
                }
                let input_shape = &input_shapes[0];
                if axes.len() != input_shape.len() {
                    return Err(GraphError::ShapeMismatch(
                        "Transpose axes must match input dimensions".to_string(),
                    ));
                }
                let mut output_shape = vec![0; input_shape.len()];
                for (i, &axis) in axes.iter().enumerate() {
                    output_shape[i] = input_shape[axis];
                }
                Ok(output_shape)
            }
            TensorOp::ReduceSum { axes, keepdims } | TensorOp::ReduceMean { axes, keepdims } => {
                if input_shapes.is_empty() {
                    return Err(GraphError::MissingInput(
                        "No input shapes provided".to_string(),
                    ));
                }
                let input_shape = &input_shapes[0];
                if *keepdims {
                    let mut output_shape = input_shape.clone();
                    for &axis in axes {
                        if axis < output_shape.len() {
                            output_shape[axis] = 1;
                        }
                    }
                    Ok(output_shape)
                } else {
                    let output_shape: Vec<usize> = input_shape
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !axes.contains(i))
                        .map(|(_, &dim)| dim)
                        .collect();
                    Ok(output_shape)
                }
            }
            TensorOp::Softmax { .. } => {
                if input_shapes.is_empty() {
                    return Err(GraphError::MissingInput(
                        "No input shapes provided".to_string(),
                    ));
                }
                Ok(input_shapes[0].clone())
            }
            TensorOp::LayerNorm { .. }
            | TensorOp::BatchNorm { .. }
            | TensorOp::Pow { .. }
            | TensorOp::FusedBatchNormReLU { .. }
            | TensorOp::FusedLayerNormDropout { .. } => {
                if input_shapes.is_empty() {
                    return Err(GraphError::MissingInput(
                        "No input shapes provided".to_string(),
                    ));
                }
                Ok(input_shapes[0].clone())
            }
            TensorOp::Concat { axis } => {
                if input_shapes.is_empty() {
                    return Err(GraphError::MissingInput(
                        "Concat requires at least one input".to_string(),
                    ));
                }
                let mut output_shape = input_shapes[0].clone();
                if *axis >= output_shape.len() {
                    return Err(GraphError::ShapeMismatch("Invalid concat axis".to_string()));
                }
                for shape in &input_shapes[1..] {
                    if shape.len() != output_shape.len() {
                        return Err(GraphError::ShapeMismatch(
                            "Concat inputs must have same rank".to_string(),
                        ));
                    }
                    output_shape[*axis] += shape[*axis];
                }
                Ok(output_shape)
            }
            TensorOp::Slice { start, end, .. } => {
                if input_shapes.is_empty() {
                    return Err(GraphError::MissingInput(
                        "No input shapes provided".to_string(),
                    ));
                }
                let input_shape = &input_shapes[0];
                let output_shape: Vec<usize> = start
                    .iter()
                    .zip(end.iter())
                    .map(|(&s, &e)| (e - s).max(0) as usize)
                    .collect();
                if output_shape.len() != input_shape.len() {
                    return Err(GraphError::ShapeMismatch(
                        "Slice dimensions must match input".to_string(),
                    ));
                }
                Ok(output_shape)
            }
            TensorOp::Pad { padding, .. } => {
                if input_shapes.is_empty() {
                    return Err(GraphError::MissingInput(
                        "No input shapes provided".to_string(),
                    ));
                }
                let input_shape = &input_shapes[0];
                let output_shape: Vec<usize> = input_shape
                    .iter()
                    .zip(padding.iter())
                    .map(|(&dim, &(pad_before, pad_after))| dim + pad_before + pad_after)
                    .collect();
                Ok(output_shape)
            }
            TensorOp::FusedLinear => {
                if input_shapes.len() < 3 {
                    return Err(GraphError::MissingInput(
                        "FusedLinear requires 3 inputs".to_string(),
                    ));
                }
                // Similar to MatMul + Add
                let a = &input_shapes[0];
                let b = &input_shapes[1];
                if a.len() < 2 || b.len() < 2 {
                    return Err(GraphError::ShapeMismatch(
                        "Linear requires at least 2D tensors".to_string(),
                    ));
                }
                let m = a[a.len() - 2];
                let n = b[b.len() - 1];
                let mut result = a[..a.len() - 2].to_vec();
                result.push(m);
                result.push(n);
                Ok(result)
            }
            TensorOp::FusedAddReLU => {
                if input_shapes.len() < 2 {
                    return Err(GraphError::MissingInput(
                        "FusedAddReLU requires 2 inputs".to_string(),
                    ));
                }
                Self::broadcast_shapes(&input_shapes[0], &input_shapes[1])
            }
            _ => {
                // For operations not yet implemented, preserve first input shape
                if input_shapes.is_empty() {
                    return Err(GraphError::MissingInput(
                        "No input shapes provided".to_string(),
                    ));
                }
                Ok(input_shapes[0].clone())
            }
        }
    }

    /// Broadcast two shapes according to NumPy broadcasting rules
    fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, GraphError> {
        let mut result = Vec::new();
        let max_len = a.len().max(b.len());

        for i in 0..max_len {
            let dim_a = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
            let dim_b = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

            if dim_a == dim_b {
                result.push(dim_a);
            } else if dim_a == 1 {
                result.push(dim_b);
            } else if dim_b == 1 {
                result.push(dim_a);
            } else {
                return Err(GraphError::ShapeMismatch(format!(
                    "Cannot broadcast shapes: {:?} and {:?}",
                    a, b
                )));
            }
        }

        result.reverse();
        Ok(result)
    }
}

/// Node in the computation graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique node ID
    pub id: String,

    /// Operation type
    pub op: TensorOp,

    /// Input node IDs
    pub inputs: Vec<String>,

    /// Output shape (if known)
    pub output_shape: Option<Vec<usize>>,

    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl GraphNode {
    /// Create a new graph node
    pub fn new(id: String, op: TensorOp) -> Self {
        Self {
            id,
            op,
            inputs: Vec::new(),
            output_shape: None,
            metadata: HashMap::new(),
        }
    }

    /// Add an input node
    pub fn add_input(mut self, input_id: String) -> Self {
        self.inputs.push(input_id);
        self
    }

    /// Set output shape
    pub fn with_output_shape(mut self, shape: Vec<usize>) -> Self {
        self.output_shape = Some(shape);
        self
    }

    /// Add metadata
    pub fn add_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Computation graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationGraph {
    /// Graph nodes (node ID -> node)
    pub nodes: HashMap<String, GraphNode>,

    /// Input node IDs
    pub inputs: Vec<String>,

    /// Output node IDs
    pub outputs: Vec<String>,

    /// Graph metadata
    pub metadata: HashMap<String, String>,

    /// Graph CID (if stored in IPFS)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(serialize_with = "serialize_optional_cid")]
    #[serde(deserialize_with = "deserialize_optional_cid")]
    pub cid: Option<Cid>,
}

impl ComputationGraph {
    /// Create a new empty computation graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            metadata: HashMap::new(),
            cid: None,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: GraphNode) -> Result<(), GraphError> {
        let id = node.id.clone();

        // Validate inputs exist
        for input_id in &node.inputs {
            if !self.nodes.contains_key(input_id) && !self.inputs.contains(input_id) {
                return Err(GraphError::NodeNotFound(input_id.clone()));
            }
        }

        self.nodes.insert(id, node);
        Ok(())
    }

    /// Mark a node as an input
    pub fn mark_input(&mut self, node_id: String) {
        if !self.inputs.contains(&node_id) {
            self.inputs.push(node_id);
        }
    }

    /// Mark a node as an output
    pub fn mark_output(&mut self, node_id: String) {
        if !self.outputs.contains(&node_id) {
            self.outputs.push(node_id);
        }
    }

    /// Get topological order of nodes
    pub fn topological_sort(&self) -> Result<Vec<String>, GraphError> {
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut adj_list: HashMap<String, Vec<String>> = HashMap::new();

        // Build adjacency list and compute in-degrees
        for (node_id, node) in &self.nodes {
            in_degree.entry(node_id.clone()).or_insert(0);
            adj_list.entry(node_id.clone()).or_default();

            for input_id in &node.inputs {
                if self.nodes.contains_key(input_id) {
                    *in_degree.entry(node_id.clone()).or_insert(0) += 1;
                    adj_list
                        .entry(input_id.clone())
                        .or_default()
                        .push(node_id.clone());
                }
            }
        }

        // Kahn's algorithm
        let mut queue: VecDeque<String> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(id, _)| id.clone())
            .collect();

        let mut result = Vec::new();

        while let Some(node_id) = queue.pop_front() {
            result.push(node_id.clone());

            if let Some(neighbors) = adj_list.get(&node_id) {
                for neighbor in neighbors {
                    if let Some(deg) = in_degree.get_mut(neighbor) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(neighbor.clone());
                        }
                    }
                }
            }
        }

        if result.len() != self.nodes.len() {
            return Err(GraphError::CircularDependency);
        }

        Ok(result)
    }

    /// Extract a subgraph containing only the specified output nodes
    pub fn extract_subgraph(&self, output_ids: &[String]) -> Result<ComputationGraph, GraphError> {
        let mut subgraph = ComputationGraph::new();
        let mut visited = HashSet::new();
        let mut queue: VecDeque<String> = output_ids.iter().cloned().collect();

        // Backward DFS to find all dependencies
        while let Some(node_id) = queue.pop_front() {
            if visited.contains(&node_id) {
                continue;
            }

            visited.insert(node_id.clone());

            if let Some(node) = self.nodes.get(&node_id) {
                for input_id in &node.inputs {
                    if !visited.contains(input_id) {
                        queue.push_back(input_id.clone());
                    }
                }
            }
        }

        // Set inputs first (before adding nodes that depend on them)
        for input_id in &self.inputs {
            if visited.contains(input_id) {
                subgraph.mark_input(input_id.clone());
            }
        }

        // Copy relevant nodes
        for node_id in &visited {
            if let Some(node) = self.nodes.get(node_id) {
                subgraph.nodes.insert(node_id.clone(), node.clone());
            }
        }

        // Set outputs
        for output_id in output_ids {
            subgraph.mark_output(output_id.clone());
        }

        Ok(subgraph)
    }

    /// Optimize the graph using common subexpression elimination
    pub fn optimize_cse(&mut self) -> usize {
        let mut optimized_count = 0;
        let mut expr_map: HashMap<String, String> = HashMap::new();

        if let Ok(sorted) = self.topological_sort() {
            for node_id in sorted {
                if let Some(node) = self.nodes.get(&node_id) {
                    // Create expression signature
                    let signature = format!("{:?}:{:?}", node.op, node.inputs);

                    if let Some(existing_id) = expr_map.get(&signature) {
                        // Found duplicate, replace references
                        for other_node in self.nodes.values_mut() {
                            for input in &mut other_node.inputs {
                                if input == &node_id {
                                    *input = existing_id.clone();
                                    optimized_count += 1;
                                }
                            }
                        }
                    } else {
                        expr_map.insert(signature, node_id.clone());
                    }
                }
            }
        }

        optimized_count
    }

    /// Count the number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of inputs
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Get the number of outputs
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// Propagate shapes through the graph (shape inference)
    /// This method performs a topological traversal and infers output shapes for all nodes
    pub fn propagate_shapes(&mut self) -> Result<(), GraphError> {
        // Get topological order
        let topo_order = self.topological_sort()?;

        // Propagate shapes in topological order
        for node_id in topo_order {
            if let Some(node) = self.nodes.get(&node_id).cloned() {
                // Skip if shape is already known
                if node.output_shape.is_some() {
                    continue;
                }

                // Collect input shapes
                let mut input_shapes = Vec::new();
                for input_id in &node.inputs {
                    if let Some(input_node) = self.nodes.get(input_id) {
                        if let Some(shape) = &input_node.output_shape {
                            input_shapes.push(shape.clone());
                        } else {
                            return Err(GraphError::InvalidGraph(format!(
                                "Input node {} has no shape information",
                                input_id
                            )));
                        }
                    } else {
                        return Err(GraphError::NodeNotFound(input_id.clone()));
                    }
                }

                // Infer output shape
                let output_shape = node.op.infer_output_shape(&input_shapes)?;

                // Update node with inferred shape
                if let Some(node_mut) = self.nodes.get_mut(&node_id) {
                    node_mut.output_shape = Some(output_shape);
                }
            }
        }

        Ok(())
    }

    /// Validate graph structure and shapes
    pub fn validate(&self) -> Result<(), GraphError> {
        // Check all inputs exist
        for input_id in &self.inputs {
            if !self.nodes.contains_key(input_id) {
                return Err(GraphError::NodeNotFound(format!(
                    "Input node {} not found",
                    input_id
                )));
            }
        }

        // Check all outputs exist
        for output_id in &self.outputs {
            if !self.nodes.contains_key(output_id) {
                return Err(GraphError::NodeNotFound(format!(
                    "Output node {} not found",
                    output_id
                )));
            }
        }

        // Check all node inputs exist
        for (node_id, node) in &self.nodes {
            for input_id in &node.inputs {
                if !self.nodes.contains_key(input_id) && !self.inputs.contains(input_id) {
                    return Err(GraphError::NodeNotFound(format!(
                        "Node {} references non-existent input {}",
                        node_id, input_id
                    )));
                }
            }

            // Validate expected number of inputs
            let expected_inputs = node.op.num_inputs();
            if node.inputs.len() != expected_inputs && expected_inputs > 0 {
                return Err(GraphError::InvalidGraph(format!(
                    "Node {} expects {} inputs but has {}",
                    node_id,
                    expected_inputs,
                    node.inputs.len()
                )));
            }
        }

        // Check for cycles
        self.topological_sort().map(|_| ())
    }

    /// Get memory footprint estimate for the graph
    pub fn estimate_memory(&self) -> usize {
        let mut total_bytes = 0;

        for node in self.nodes.values() {
            if let Some(shape) = &node.output_shape {
                // Assume f32 (4 bytes) for simplicity
                let elements: usize = shape.iter().product();
                total_bytes += elements * 4;
            }
        }

        total_bytes
    }
}

impl Default for ComputationGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph optimizer for applying optimizations
pub struct GraphOptimizer;

impl GraphOptimizer {
    /// Apply constant folding
    pub fn constant_folding(graph: &mut ComputationGraph) -> Result<usize, GraphError> {
        let mut folded_count = 0;

        // Simplified constant folding - in a real implementation,
        // we would evaluate constant sub-expressions
        let sorted = graph.topological_sort()?;

        for node_id in sorted {
            if let Some(node) = graph.nodes.get(&node_id) {
                // Check if all inputs are constants
                let all_const = node.inputs.iter().all(|input_id| {
                    graph
                        .nodes
                        .get(input_id)
                        .map(|n| matches!(n.op, TensorOp::Constant { .. }))
                        .unwrap_or(false)
                });

                if all_const && node.op.is_pure() {
                    // In a real implementation, we would evaluate this
                    // and replace with a constant
                    folded_count += 1;
                }
            }
        }

        Ok(folded_count)
    }

    /// Fuse consecutive operations where possible
    pub fn fusion(graph: &mut ComputationGraph) -> Result<usize, GraphError> {
        let mut fused_count = 0;
        let mut nodes_to_remove = HashSet::new();
        let mut new_nodes: HashMap<String, GraphNode> = HashMap::new();

        // Build a map of node outputs to their consumers
        let mut consumers: HashMap<String, Vec<String>> = HashMap::new();
        for (node_id, node) in &graph.nodes {
            for input in &node.inputs {
                consumers
                    .entry(input.clone())
                    .or_default()
                    .push(node_id.clone());
            }
        }

        // Pattern 1: MatMul + Add -> FusedLinear
        for (node_id, node) in &graph.nodes {
            if let TensorOp::Add = node.op {
                if node.inputs.len() == 2 {
                    // Check if one of the inputs is a MatMul
                    for input_id in &node.inputs {
                        if let Some(input_node) = graph.nodes.get(input_id) {
                            if matches!(input_node.op, TensorOp::MatMul) {
                                // Only fuse if the MatMul has a single consumer
                                if let Some(input_consumers) = consumers.get(input_id) {
                                    if input_consumers.len() == 1
                                        && !nodes_to_remove.contains(node_id)
                                    {
                                        // Create fused node
                                        let fused_id = format!("{}_fused", node_id);
                                        let fused_node = GraphNode {
                                            id: fused_id.clone(),
                                            op: TensorOp::FusedLinear,
                                            inputs: vec![
                                                input_node.inputs[0].clone(),
                                                input_node.inputs[1].clone(),
                                                node.inputs
                                                    .iter()
                                                    .find(|&id| id != input_id)
                                                    .expect("BiasAdd node has 2 inputs; one differs from the MatMul input")
                                                    .clone(),
                                            ],
                                            output_shape: node.output_shape.clone(),
                                            metadata: HashMap::new(),
                                        };
                                        new_nodes.insert(fused_id, fused_node);
                                        nodes_to_remove.insert(node_id.clone());
                                        nodes_to_remove.insert(input_id.clone());
                                        fused_count += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Pattern 2: Add + ReLU -> FusedAddReLU
        for (node_id, node) in &graph.nodes {
            if let TensorOp::ReLU = node.op {
                if node.inputs.len() == 1 {
                    let input_id = &node.inputs[0];
                    if let Some(input_node) = graph.nodes.get(input_id) {
                        if matches!(input_node.op, TensorOp::Add) {
                            if let Some(input_consumers) = consumers.get(input_id) {
                                if input_consumers.len() == 1 && !nodes_to_remove.contains(node_id)
                                {
                                    let fused_id = format!("{}_fused", node_id);
                                    let fused_node = GraphNode {
                                        id: fused_id.clone(),
                                        op: TensorOp::FusedAddReLU,
                                        inputs: input_node.inputs.clone(),
                                        output_shape: node.output_shape.clone(),
                                        metadata: HashMap::new(),
                                    };
                                    new_nodes.insert(fused_id, fused_node);
                                    nodes_to_remove.insert(node_id.clone());
                                    nodes_to_remove.insert(input_id.clone());
                                    fused_count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Pattern 3: BatchNorm + ReLU -> FusedBatchNormReLU
        for (node_id, node) in &graph.nodes {
            if let TensorOp::ReLU = node.op {
                if node.inputs.len() == 1 {
                    let input_id = &node.inputs[0];
                    if let Some(input_node) = graph.nodes.get(input_id) {
                        if let TensorOp::BatchNorm { eps, momentum } = &input_node.op {
                            if let Some(input_consumers) = consumers.get(input_id) {
                                if input_consumers.len() == 1 && !nodes_to_remove.contains(node_id)
                                {
                                    let fused_id = format!("{}_fused", node_id);
                                    let fused_node = GraphNode {
                                        id: fused_id.clone(),
                                        op: TensorOp::FusedBatchNormReLU {
                                            eps: *eps,
                                            momentum: *momentum,
                                        },
                                        inputs: input_node.inputs.clone(),
                                        output_shape: node.output_shape.clone(),
                                        metadata: HashMap::new(),
                                    };
                                    new_nodes.insert(fused_id, fused_node);
                                    nodes_to_remove.insert(node_id.clone());
                                    nodes_to_remove.insert(input_id.clone());
                                    fused_count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Pattern 4: LayerNorm + Dropout -> FusedLayerNormDropout
        for (node_id, node) in &graph.nodes {
            if let TensorOp::Dropout { p } = &node.op {
                if node.inputs.len() == 1 {
                    let input_id = &node.inputs[0];
                    if let Some(input_node) = graph.nodes.get(input_id) {
                        if let TensorOp::LayerNorm {
                            normalized_shape,
                            eps,
                        } = &input_node.op
                        {
                            if let Some(input_consumers) = consumers.get(input_id) {
                                if input_consumers.len() == 1 && !nodes_to_remove.contains(node_id)
                                {
                                    let fused_id = format!("{}_fused", node_id);
                                    let fused_node = GraphNode {
                                        id: fused_id.clone(),
                                        op: TensorOp::FusedLayerNormDropout {
                                            normalized_shape: normalized_shape.clone(),
                                            eps: *eps,
                                            dropout_p: *p,
                                        },
                                        inputs: input_node.inputs.clone(),
                                        output_shape: node.output_shape.clone(),
                                        metadata: HashMap::new(),
                                    };
                                    new_nodes.insert(fused_id, fused_node);
                                    nodes_to_remove.insert(node_id.clone());
                                    nodes_to_remove.insert(input_id.clone());
                                    fused_count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Apply the fusion by removing old nodes and adding new ones
        graph.nodes.retain(|id, _| !nodes_to_remove.contains(id));
        graph.nodes.extend(new_nodes);

        // Update references to removed nodes
        // Build a mapping from old node IDs to fused node IDs
        let mut replacements: HashMap<String, String> = HashMap::new();
        for removed_id in &nodes_to_remove {
            let fused_id = format!("{}_fused", removed_id);
            if graph.nodes.contains_key(&fused_id) {
                replacements.insert(removed_id.clone(), fused_id);
            }
        }

        // Apply replacements
        let node_ids: Vec<String> = graph.nodes.keys().cloned().collect();
        for node_id in node_ids {
            if let Some(node) = graph.nodes.get_mut(&node_id) {
                for input in &mut node.inputs {
                    if let Some(replacement) = replacements.get(input) {
                        *input = replacement.clone();
                    }
                }
            }
        }

        Ok(fused_count)
    }

    /// Remove dead nodes (nodes not connected to outputs)
    pub fn remove_dead_nodes(graph: &mut ComputationGraph) -> Result<usize, GraphError> {
        let subgraph = graph.extract_subgraph(&graph.outputs.clone())?;
        let removed = graph.nodes.len() - subgraph.nodes.len();

        *graph = subgraph;

        Ok(removed)
    }

    /// Apply all optimizations
    pub fn optimize_all(graph: &mut ComputationGraph) -> Result<(), GraphError> {
        // Apply optimizations multiple times until convergence
        let mut prev_count = graph.node_count();

        for _ in 0..10 {
            Self::constant_folding(graph)?;
            graph.optimize_cse();
            Self::fusion(graph)?;
            Self::remove_dead_nodes(graph)?;

            let curr_count = graph.node_count();
            if curr_count == prev_count {
                break;
            }
            prev_count = curr_count;
        }

        Ok(())
    }
}

/// Lazy evaluation cache
#[derive(Debug, Clone)]
pub struct LazyCache {
    /// Cached results (node ID -> cached value)
    cache: HashMap<String, Vec<f32>>,

    /// Cache size limit (in number of entries)
    max_size: usize,

    /// Access order for LRU eviction
    access_order: VecDeque<String>,
}

impl LazyCache {
    /// Create a new lazy cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
            access_order: VecDeque::new(),
        }
    }

    /// Get a cached value
    pub fn get(&mut self, node_id: &str) -> Option<&Vec<f32>> {
        if self.cache.contains_key(node_id) {
            // Update access order
            self.access_order.retain(|id| id != node_id);
            self.access_order.push_back(node_id.to_string());

            self.cache.get(node_id)
        } else {
            None
        }
    }

    /// Insert a value into the cache
    pub fn insert(&mut self, node_id: String, value: Vec<f32>) {
        // Evict if necessary
        while self.cache.len() >= self.max_size && !self.access_order.is_empty() {
            if let Some(evict_id) = self.access_order.pop_front() {
                self.cache.remove(&evict_id);
            }
        }

        self.cache.insert(node_id.clone(), value);
        self.access_order.push_back(node_id);
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
    }

    /// Get cache size
    pub fn size(&self) -> usize {
        self.cache.len()
    }

    /// Get cache hit ratio (if we track statistics)
    pub fn hit_ratio(&self) -> f32 {
        // Simplified - would need counters for hits/misses
        0.0
    }
}

/// Execution batch containing independent nodes that can run in parallel
#[derive(Debug, Clone)]
pub struct ExecutionBatch {
    /// Node IDs in this batch
    pub node_ids: Vec<String>,
    /// Batch level in the dependency graph
    pub level: usize,
}

impl ExecutionBatch {
    /// Create a new execution batch
    pub fn new(level: usize) -> Self {
        Self {
            node_ids: Vec::new(),
            level,
        }
    }

    /// Add a node to the batch
    pub fn add_node(&mut self, node_id: String) {
        self.node_ids.push(node_id);
    }

    /// Get the number of nodes in the batch
    pub fn size(&self) -> usize {
        self.node_ids.len()
    }
}

/// Batch scheduler for identifying independent nodes
pub struct BatchScheduler;

impl BatchScheduler {
    /// Create execution batches from a computation graph
    /// Returns batches where all nodes in each batch can execute in parallel
    pub fn create_batches(graph: &ComputationGraph) -> Result<Vec<ExecutionBatch>, GraphError> {
        let sorted = graph.topological_sort()?;
        let mut batches: Vec<ExecutionBatch> = Vec::new();
        let mut node_to_level: HashMap<String, usize> = HashMap::new();

        // Assign levels to each node based on dependencies
        for node_id in &sorted {
            let max_input_level = if let Some(node) = graph.nodes.get(node_id) {
                node.inputs
                    .iter()
                    .filter_map(|input_id| node_to_level.get(input_id))
                    .max()
                    .copied()
                    .unwrap_or(0)
            } else {
                0
            };

            let level = if graph.inputs.contains(node_id) {
                0
            } else {
                max_input_level + 1
            };

            node_to_level.insert(node_id.clone(), level);

            // Add node to appropriate batch
            while batches.len() <= level {
                batches.push(ExecutionBatch::new(batches.len()));
            }
            batches[level].add_node(node_id.clone());
        }

        Ok(batches)
    }
}

/// Parallel executor for computation graphs
pub struct ParallelExecutor {
    /// Number of threads to use (None = use rayon default)
    thread_count: Option<usize>,
}

impl ParallelExecutor {
    /// Create a new parallel executor
    pub fn new(thread_count: Option<usize>) -> Self {
        Self { thread_count }
    }

    /// Execute a computation graph in parallel
    /// This is a simplified version that tracks execution order
    pub fn execute(&self, graph: &ComputationGraph) -> Result<Vec<String>, GraphError> {
        let batches = BatchScheduler::create_batches(graph)?;
        let mut executed = Vec::new();

        // Configure rayon thread pool if needed
        if let Some(threads) = self.thread_count {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| GraphError::ExecutionError(e.to_string()))?;
        }

        // Execute each batch in parallel
        for batch in batches {
            let batch_results: Vec<String> = batch
                .node_ids
                .par_iter()
                .map(|node_id| {
                    // In a real implementation, this would execute the actual operation
                    // For now, we just return the node ID to track execution
                    node_id.clone()
                })
                .collect();

            executed.extend(batch_results);
        }

        Ok(executed)
    }

    /// Execute a batch of nodes in parallel with a custom function
    pub fn execute_batch<F>(
        &self,
        batch: &ExecutionBatch,
        graph: &ComputationGraph,
        executor_fn: F,
    ) -> Result<Vec<(String, Vec<f32>)>, GraphError>
    where
        F: Fn(&GraphNode) -> Result<Vec<f32>, GraphError> + Sync + Send,
    {
        let results: Result<Vec<(String, Vec<f32>)>, GraphError> = batch
            .node_ids
            .par_iter()
            .map(|node_id| {
                let node = graph
                    .nodes
                    .get(node_id)
                    .ok_or_else(|| GraphError::NodeNotFound(node_id.clone()))?;
                let result = executor_fn(node)?;
                Ok((node_id.clone(), result))
            })
            .collect();

        results
    }
}

/// Stream chunk for streaming execution
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// Chunk data (node ID -> tensor data)
    pub data: HashMap<String, Vec<f32>>,
    /// Chunk index
    pub index: usize,
    /// Total number of chunks
    pub total_chunks: usize,
}

impl StreamChunk {
    /// Create a new stream chunk
    pub fn new(index: usize, total_chunks: usize) -> Self {
        Self {
            data: HashMap::new(),
            index,
            total_chunks,
        }
    }

    /// Add data for a node
    pub fn add_data(&mut self, node_id: String, data: Vec<f32>) {
        self.data.insert(node_id, data);
    }

    /// Check if this is the last chunk
    pub fn is_last(&self) -> bool {
        self.index == self.total_chunks - 1
    }
}

/// Streaming executor for processing data in chunks
pub struct StreamingExecutor {
    /// Chunk size (number of elements per chunk)
    chunk_size: usize,
    /// Maximum number of chunks to buffer
    max_buffer_size: usize,
    /// Current buffer
    buffer: Arc<Mutex<VecDeque<StreamChunk>>>,
}

impl StreamingExecutor {
    /// Create a new streaming executor
    pub fn new(chunk_size: usize, max_buffer_size: usize) -> Self {
        Self {
            chunk_size,
            max_buffer_size,
            buffer: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Split input data into chunks
    pub fn create_chunks(&self, data: Vec<f32>, node_id: &str) -> Vec<StreamChunk> {
        let total_elements = data.len();
        let total_chunks = total_elements.div_ceil(self.chunk_size);
        let mut chunks = Vec::new();

        for (i, chunk_data) in data.chunks(self.chunk_size).enumerate() {
            let mut chunk = StreamChunk::new(i, total_chunks);
            chunk.add_data(node_id.to_string(), chunk_data.to_vec());
            chunks.push(chunk);
        }

        chunks
    }

    /// Execute graph on a stream chunk
    pub fn execute_chunk(
        &self,
        _graph: &ComputationGraph,
        chunk: StreamChunk,
    ) -> Result<StreamChunk, GraphError> {
        // In a real implementation, this would:
        // 1. Execute the graph operations on the chunk data
        // 2. Apply backpressure if buffer is full
        // 3. Return the processed chunk

        // For now, return the chunk as-is
        Ok(chunk)
    }

    /// Process a stream of chunks through the graph
    pub fn process_stream(
        &self,
        graph: &ComputationGraph,
        chunks: Vec<StreamChunk>,
    ) -> Result<Vec<StreamChunk>, GraphError> {
        let mut results = Vec::new();

        for chunk in chunks {
            // Check backpressure
            {
                let buffer = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
                if buffer.len() >= self.max_buffer_size {
                    // In a real implementation, we would wait or apply backpressure
                    // For now, we just continue
                }
            }

            // Execute chunk
            let result = self.execute_chunk(graph, chunk)?;

            // Add to buffer
            {
                let mut buffer = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
                buffer.push_back(result.clone());

                // Keep buffer size in check
                while buffer.len() > self.max_buffer_size {
                    buffer.pop_front();
                }
            }

            results.push(result);
        }

        Ok(results)
    }

    /// Get the current buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Clear the buffer
    pub fn clear_buffer(&self) {
        self.buffer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// Get chunk size
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Get max buffer size
    pub fn max_buffer_size(&self) -> usize {
        self.max_buffer_size
    }
}

/// Distributed graph execution for multi-node computation
///
/// This module provides infrastructure for distributing computation graphs
/// across multiple nodes in an IPFS network.
/// Node assignment for distributed execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAssignment {
    /// Node ID in the computation graph
    pub node_id: String,
    /// Worker ID (peer ID or node identifier)
    pub worker_id: String,
    /// Execution priority
    pub priority: usize,
}

/// Graph partition for a single worker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPartition {
    /// Worker ID that will execute this partition
    pub worker_id: String,
    /// Nodes assigned to this worker
    pub nodes: Vec<String>,
    /// Input dependencies from other partitions
    pub external_inputs: HashMap<String, String>, // node_id -> source_worker_id
    /// Output nodes consumed by other partitions
    pub external_outputs: Vec<String>,
    /// Subgraph for this partition
    #[serde(skip)]
    pub subgraph: Option<ComputationGraph>,
}

impl GraphPartition {
    /// Create a new graph partition
    pub fn new(worker_id: String) -> Self {
        Self {
            worker_id,
            nodes: Vec::new(),
            external_inputs: HashMap::new(),
            external_outputs: Vec::new(),
            subgraph: None,
        }
    }

    /// Add a node to this partition
    pub fn add_node(&mut self, node_id: String) {
        if !self.nodes.contains(&node_id) {
            self.nodes.push(node_id);
        }
    }

    /// Add an external input dependency
    pub fn add_external_input(&mut self, node_id: String, source_worker_id: String) {
        self.external_inputs.insert(node_id, source_worker_id);
    }

    /// Mark a node as an external output
    pub fn mark_external_output(&mut self, node_id: String) {
        if !self.external_outputs.contains(&node_id) {
            self.external_outputs.push(node_id);
        }
    }

    /// Get the number of nodes in this partition
    pub fn size(&self) -> usize {
        self.nodes.len()
    }
}

/// Distributed executor for multi-node graph execution
pub struct DistributedExecutor {
    /// Worker assignments
    assignments: HashMap<String, NodeAssignment>,
    /// Graph partitions by worker ID
    partitions: HashMap<String, GraphPartition>,
    /// Communication timeout (milliseconds)
    timeout_ms: u64,
}

impl DistributedExecutor {
    /// Create a new distributed executor
    pub fn new() -> Self {
        Self {
            assignments: HashMap::new(),
            partitions: HashMap::new(),
            timeout_ms: 30000, // 30 seconds default
        }
    }

    /// Set communication timeout
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Partition a graph across multiple workers
    /// Uses a simple round-robin strategy for now
    pub fn partition_graph(
        &mut self,
        graph: &ComputationGraph,
        worker_ids: &[String],
    ) -> Result<(), GraphError> {
        if worker_ids.is_empty() {
            return Err(GraphError::InvalidGraph("No workers available".to_string()));
        }

        // Get topological order
        let sorted = graph.topological_sort()?;

        // Create partitions for each worker
        for worker_id in worker_ids {
            self.partitions
                .insert(worker_id.clone(), GraphPartition::new(worker_id.clone()));
        }

        // Assign nodes to workers in round-robin fashion
        for (idx, node_id) in sorted.iter().enumerate() {
            let worker_id = &worker_ids[idx % worker_ids.len()];
            let assignment = NodeAssignment {
                node_id: node_id.clone(),
                worker_id: worker_id.clone(),
                priority: idx,
            };

            self.assignments.insert(node_id.clone(), assignment);
            if let Some(partition) = self.partitions.get_mut(worker_id) {
                partition.add_node(node_id.clone());
            }
        }

        // Identify cross-partition dependencies
        for (node_id, node) in &graph.nodes {
            if let Some(assignment) = self.assignments.get(node_id) {
                for input_id in &node.inputs {
                    if let Some(input_assignment) = self.assignments.get(input_id) {
                        if input_assignment.worker_id != assignment.worker_id {
                            // Cross-partition dependency
                            if let Some(partition) = self.partitions.get_mut(&assignment.worker_id)
                            {
                                partition.add_external_input(
                                    input_id.clone(),
                                    input_assignment.worker_id.clone(),
                                );
                            }
                            if let Some(source_partition) =
                                self.partitions.get_mut(&input_assignment.worker_id)
                            {
                                source_partition.mark_external_output(input_id.clone());
                            }
                        }
                    }
                }
            }
        }

        // Build subgraphs for each partition
        for partition in self.partitions.values_mut() {
            let mut subgraph = ComputationGraph::new();

            // Add nodes belonging to this partition
            for node_id in &partition.nodes {
                if let Some(node) = graph.nodes.get(node_id) {
                    subgraph.nodes.insert(node_id.clone(), node.clone());
                }
            }

            // Mark inputs and outputs
            for input_id in partition.external_inputs.keys() {
                if subgraph.nodes.contains_key(input_id) || graph.inputs.contains(input_id) {
                    subgraph.mark_input(input_id.clone());
                }
            }

            for output_id in &partition.external_outputs {
                if subgraph.nodes.contains_key(output_id) {
                    subgraph.mark_output(output_id.clone());
                }
            }

            // Also include original graph inputs if they're in this partition
            for input_id in &graph.inputs {
                if partition.nodes.contains(input_id) {
                    subgraph.mark_input(input_id.clone());
                }
            }

            // Include original graph outputs if they're in this partition
            for output_id in &graph.outputs {
                if partition.nodes.contains(output_id) {
                    subgraph.mark_output(output_id.clone());
                }
            }

            partition.subgraph = Some(subgraph);
        }

        Ok(())
    }

    /// Get partition for a specific worker
    pub fn get_partition(&self, worker_id: &str) -> Option<&GraphPartition> {
        self.partitions.get(worker_id)
    }

    /// Get all partitions
    pub fn get_partitions(&self) -> &HashMap<String, GraphPartition> {
        &self.partitions
    }

    /// Get node assignment
    pub fn get_assignment(&self, node_id: &str) -> Option<&NodeAssignment> {
        self.assignments.get(node_id)
    }

    /// Execute a distributed graph
    /// NOTE: This is a stub that will be integrated with ipfrs-network
    pub fn execute_distributed(
        &self,
        _graph: &ComputationGraph,
    ) -> Result<HashMap<String, Vec<f32>>, GraphError> {
        // This is a placeholder for distributed execution
        // When ipfrs-network is integrated, this will:
        // 1. Send subgraphs to respective workers
        // 2. Coordinate data transfer between workers
        // 3. Collect results from workers
        // 4. Assemble final output

        Err(GraphError::ExecutionError(
            "Distributed execution requires ipfrs-network integration".to_string(),
        ))
    }

    /// Estimate communication cost for a partition
    pub fn estimate_communication_cost(&self, worker_id: &str) -> usize {
        if let Some(partition) = self.partitions.get(worker_id) {
            partition.external_inputs.len() + partition.external_outputs.len()
        } else {
            0
        }
    }

    /// Get total number of workers
    pub fn worker_count(&self) -> usize {
        self.partitions.len()
    }

    /// Get timeout in milliseconds
    pub fn timeout(&self) -> u64 {
        self.timeout_ms
    }
}

impl Default for DistributedExecutor {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions for serializing/deserializing Option<Cid>
fn serialize_optional_cid<S>(cid: &Option<Cid>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::Serialize;
    match cid {
        Some(c) => Some(c.to_string()).serialize(serializer),
        None => None::<String>.serialize(serializer),
    }
}

fn deserialize_optional_cid<'de, D>(deserializer: D) -> Result<Option<Cid>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let opt = Option::<String>::deserialize(deserializer)?;
    opt.map(|s| s.parse().map_err(serde::de::Error::custom))
        .transpose()
}

#[cfg(test)]
#[path = "computation_graph_tests.rs"]
mod tests {}
