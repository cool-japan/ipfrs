//! Visualization utilities for computation graphs and proofs.
//!
//! This module provides tools for exporting graphs and proofs to DOT format
//! (Graphviz) for visualization and debugging.
//!
//! # Examples
//!
//! ## Visualizing a Computation Graph
//!
//! ```
//! use ipfrs_tensorlogic::{ComputationGraph, GraphNode, TensorOp, GraphVisualizer};
//!
//! let mut graph = ComputationGraph::new();
//!
//! // Create nodes
//! let input = GraphNode::new("input".to_string(), TensorOp::Input {
//!     name: "x".to_string(),
//! });
//! graph.add_node(input).expect("write to String is infallible");
//! graph.mark_input("input".to_string());
//!
//! let relu = GraphNode::new("relu".to_string(), TensorOp::ReLU)
//!     .add_input("input".to_string());
//! graph.add_node(relu).expect("write to String is infallible");
//! graph.mark_output("relu".to_string());
//!
//! // Export to DOT format
//! let dot = GraphVisualizer::to_dot(&graph);
//! println!("{}", dot);
//! // Save to file: std::fs::write("graph.dot", dot).expect("write to String is infallible");
//! // Render: dot -Tpng graph.dot -o graph.png
//! ```

use crate::computation_graph::{ComputationGraph, TensorOp};
use crate::proof_storage::ProofFragment;
use std::fmt::Write as FmtWrite;

/// Visualizer for computation graphs
pub struct GraphVisualizer;

impl GraphVisualizer {
    /// Export a computation graph to DOT format
    ///
    /// The output can be rendered using Graphviz:
    /// ```bash
    /// dot -Tpng graph.dot -o graph.png
    /// dot -Tsvg graph.dot -o graph.svg
    /// ```
    pub fn to_dot(graph: &ComputationGraph) -> String {
        let mut dot = String::new();
        writeln!(dot, "digraph ComputationGraph {{").expect("write to String is infallible");
        writeln!(dot, "  rankdir=TB;").expect("write to String is infallible");
        writeln!(dot, "  node [shape=box, style=filled];").expect("write to String is infallible");
        writeln!(dot).expect("write to String is infallible");

        // Write nodes
        for (node_id, node) in &graph.nodes {
            let color = Self::node_color(&node.op);
            let shape = if graph.inputs.contains(node_id) {
                "ellipse"
            } else if graph.outputs.contains(node_id) {
                "doubleoctagon"
            } else {
                "box"
            };

            let label = Self::format_operation(&node.op);
            writeln!(
                dot,
                "  \"{}\" [label=\"{}\\n{}\", fillcolor=\"{}\", shape={}];",
                Self::escape(node_id),
                Self::escape(node_id),
                label,
                color,
                shape
            )
            .expect("write to String is infallible");
        }

        writeln!(dot).expect("write to String is infallible");

        // Write edges
        for (node_id, node) in &graph.nodes {
            for input in &node.inputs {
                writeln!(
                    dot,
                    "  \"{}\" -> \"{}\";",
                    Self::escape(input),
                    Self::escape(node_id)
                )
                .expect("write to String is infallible");
            }
        }

        // Add legend
        writeln!(dot).expect("write to String is infallible");
        writeln!(dot, "  subgraph cluster_legend {{").expect("write to String is infallible");
        writeln!(dot, "    label=\"Legend\";").expect("write to String is infallible");
        writeln!(dot, "    style=filled;").expect("write to String is infallible");
        writeln!(dot, "    fillcolor=lightgrey;").expect("write to String is infallible");
        writeln!(
            dot,
            "    legend_input [label=\"Input\", shape=ellipse, fillcolor=lightblue];"
        )
        .expect("write to String is infallible");
        writeln!(
            dot,
            "    legend_output [label=\"Output\", shape=doubleoctagon, fillcolor=lightgreen];"
        )
        .expect("write to String is infallible");
        writeln!(
            dot,
            "    legend_compute [label=\"Compute\", shape=box, fillcolor=lightyellow];"
        )
        .expect("write to String is infallible");
        writeln!(dot, "  }}").expect("write to String is infallible");

        writeln!(dot, "}}").expect("write to String is infallible");
        dot
    }

    /// Get color for a node based on operation type
    fn node_color(op: &TensorOp) -> &'static str {
        match op {
            TensorOp::Input { .. } | TensorOp::Constant { .. } => "lightblue",
            TensorOp::MatMul | TensorOp::Einsum { .. } => "orange",
            TensorOp::Add | TensorOp::Mul | TensorOp::Sub | TensorOp::Div => "yellow",
            TensorOp::ReLU
            | TensorOp::Tanh
            | TensorOp::Sigmoid
            | TensorOp::GELU
            | TensorOp::Softmax { .. } => "lightgreen",
            TensorOp::LayerNorm { .. } | TensorOp::BatchNorm { .. } => "lightcoral",
            TensorOp::Dropout { .. } => "plum",
            TensorOp::Reshape { .. } | TensorOp::Transpose { .. } | TensorOp::Slice { .. } => {
                "lightyellow"
            }
            _ => "white",
        }
    }

    /// Format operation for display
    fn format_operation(op: &TensorOp) -> String {
        match op {
            TensorOp::Input { name } => format!("Input({})", name),
            TensorOp::Constant { value_cid } => format!("Const(cid:{})", &value_cid[..8]),
            TensorOp::MatMul => "MatMul".to_string(),
            TensorOp::Einsum { subscripts } => format!("Einsum({})", subscripts),
            TensorOp::Add => "Add".to_string(),
            TensorOp::Mul => "Multiply".to_string(),
            TensorOp::Sub => "Subtract".to_string(),
            TensorOp::Div => "Divide".to_string(),
            TensorOp::ReLU => "ReLU".to_string(),
            TensorOp::Tanh => "Tanh".to_string(),
            TensorOp::Sigmoid => "Sigmoid".to_string(),
            TensorOp::GELU => "GELU".to_string(),
            TensorOp::Softmax { axis } => format!("Softmax(axis={})", axis),
            TensorOp::LayerNorm {
                normalized_shape: _,
                eps,
            } => format!("LayerNorm(ε={:.1e})", eps),
            TensorOp::BatchNorm { eps, momentum } => {
                format!("BatchNorm(ε={:.1e}, μ={:.2})", eps, momentum)
            }
            TensorOp::Dropout { p } => format!("Dropout({:.2})", p),
            TensorOp::Reshape { shape } => format!("Reshape({:?})", shape),
            TensorOp::Transpose { axes } => format!("Transpose({:?})", axes),
            TensorOp::ReduceSum { axes, keepdims: _ } => format!("ReduceSum({:?})", axes),
            TensorOp::ReduceMean { axes, keepdims: _ } => format!("ReduceMean({:?})", axes),
            TensorOp::Concat { axis } => format!("Concat(axis={})", axis),
            TensorOp::Split { axis, sections } => {
                format!("Split(axis={}, n={})", axis, sections.len())
            }
            TensorOp::Gather { axis } => format!("Gather(axis={})", axis),
            TensorOp::Scatter { axis } => format!("Scatter(axis={})", axis),
            TensorOp::Slice {
                start,
                end,
                strides,
            } => format!("Slice({:?}:{:?}:{:?})", start, end, strides),
            TensorOp::Pad { padding, mode: _ } => format!("Pad({:?})", padding),
            TensorOp::Exp => "Exp".to_string(),
            TensorOp::Log => "Log".to_string(),
            TensorOp::Pow { exponent } => format!("Pow({})", exponent),
            TensorOp::Sqrt => "Sqrt".to_string(),
            TensorOp::FusedLinear => "FusedLinear".to_string(),
            TensorOp::FusedAddReLU => "FusedAdd+ReLU".to_string(),
            TensorOp::FusedBatchNormReLU { eps, momentum } => {
                format!("FusedBN+ReLU(ε={:.1e}, μ={:.2})", eps, momentum)
            }
            TensorOp::FusedLayerNormDropout {
                normalized_shape: _,
                eps,
                dropout_p,
            } => format!("FusedLN+Dropout(ε={:.1e}, p={:.2})", eps, dropout_p),
        }
    }

    /// Escape special characters for DOT format
    fn escape(s: &str) -> String {
        s.replace('\"', "\\\"")
            .replace('\n', "\\n")
            .replace('\t', "\\t")
    }

    /// Export graph statistics
    pub fn graph_stats(graph: &ComputationGraph) -> String {
        let mut stats = String::new();
        writeln!(stats, "Graph Statistics:").expect("write to String is infallible");
        writeln!(stats, "  Total nodes: {}", graph.nodes.len())
            .expect("write to String is infallible");
        writeln!(stats, "  Input nodes: {}", graph.inputs.len())
            .expect("write to String is infallible");
        writeln!(stats, "  Output nodes: {}", graph.outputs.len())
            .expect("write to String is infallible");

        // Count operation types
        let mut op_counts = std::collections::HashMap::new();
        for node in graph.nodes.values() {
            let op_name = Self::operation_name(&node.op);
            *op_counts.entry(op_name).or_insert(0) += 1;
        }

        writeln!(stats, "  Operation counts:").expect("write to String is infallible");
        let mut ops: Vec<_> = op_counts.into_iter().collect();
        ops.sort_by_key(|a| std::cmp::Reverse(a.1));
        for (op, count) in ops {
            writeln!(stats, "    {}: {}", op, count).expect("write to String is infallible");
        }

        stats
    }

    fn operation_name(op: &TensorOp) -> &'static str {
        match op {
            TensorOp::Input { .. } => "Input",
            TensorOp::Constant { .. } => "Constant",
            TensorOp::MatMul => "MatMul",
            TensorOp::Einsum { .. } => "Einsum",
            TensorOp::Add => "Add",
            TensorOp::Mul => "Mul",
            TensorOp::Sub => "Sub",
            TensorOp::Div => "Div",
            TensorOp::ReLU => "ReLU",
            TensorOp::Tanh => "Tanh",
            TensorOp::Sigmoid => "Sigmoid",
            TensorOp::GELU => "GELU",
            TensorOp::Softmax { .. } => "Softmax",
            TensorOp::LayerNorm { .. } => "LayerNorm",
            TensorOp::BatchNorm { .. } => "BatchNorm",
            TensorOp::Dropout { .. } => "Dropout",
            TensorOp::Reshape { .. } => "Reshape",
            TensorOp::Transpose { .. } => "Transpose",
            TensorOp::ReduceSum { .. } => "ReduceSum",
            TensorOp::ReduceMean { .. } => "ReduceMean",
            TensorOp::Concat { .. } => "Concat",
            TensorOp::Split { .. } => "Split",
            TensorOp::Gather { .. } => "Gather",
            TensorOp::Scatter { .. } => "Scatter",
            TensorOp::Slice { .. } => "Slice",
            TensorOp::Pad { .. } => "Pad",
            TensorOp::Exp => "Exp",
            TensorOp::Log => "Log",
            TensorOp::Pow { .. } => "Pow",
            TensorOp::Sqrt => "Sqrt",
            TensorOp::FusedLinear => "FusedLinear",
            TensorOp::FusedAddReLU => "FusedAddReLU",
            TensorOp::FusedBatchNormReLU { .. } => "FusedBatchNormReLU",
            TensorOp::FusedLayerNormDropout { .. } => "FusedLayerNormDropout",
        }
    }
}

/// Visualizer for proof trees
pub struct ProofVisualizer;

impl ProofVisualizer {
    /// Export a proof tree to DOT format
    ///
    /// The proof is rendered as a tree with the conclusion at the top
    /// and premises as child nodes.
    pub fn to_dot(proof: &ProofFragment, id: usize) -> String {
        let mut dot = String::new();
        writeln!(dot, "digraph ProofTree {{").expect("write to String is infallible");
        writeln!(dot, "  rankdir=TB;").expect("write to String is infallible");
        writeln!(dot, "  node [shape=box, style=\"filled,rounded\"];")
            .expect("write to String is infallible");
        writeln!(dot).expect("write to String is infallible");

        let mut node_counter = 0;
        Self::write_proof_node(&mut dot, proof, id, &mut node_counter);

        writeln!(dot, "}}").expect("write to String is infallible");
        dot
    }

    fn write_proof_node(
        dot: &mut String,
        proof: &ProofFragment,
        node_id: usize,
        counter: &mut usize,
    ) {
        let color = if proof.premise_refs.is_empty() {
            "lightblue" // Fact (no premises)
        } else {
            "lightyellow" // Rule application
        };

        let conclusion_str = format!("{:?}", proof.conclusion);
        writeln!(
            dot,
            "  node_{} [label=\"{}\", fillcolor=\"{}\"];",
            node_id,
            GraphVisualizer::escape(&conclusion_str),
            color
        )
        .expect("write to String is infallible");

        // Write premise references as child nodes
        for premise_ref in &proof.premise_refs {
            *counter += 1;
            let premise_id = *counter;
            let premise_str = if let Some(ref hint) = premise_ref.conclusion_hint {
                hint.clone()
            } else {
                format!("CID: {}", premise_ref.cid)
            };
            writeln!(
                dot,
                "  node_{} [label=\"{}\", fillcolor=\"lightgray\"];",
                premise_id,
                GraphVisualizer::escape(&premise_str)
            )
            .expect("write to String is infallible");
            writeln!(dot, "  node_{} -> node_{};", node_id, premise_id)
                .expect("write to String is infallible");
        }

        // Add rule information
        if let Some(ref rule_ref) = proof.rule_applied {
            writeln!(
                dot,
                "  node_{}_rule [label=\"Rule: {}\", shape=note, fillcolor=\"lightyellow\"];",
                node_id,
                GraphVisualizer::escape(&rule_ref.rule_id)
            )
            .expect("write to String is infallible");
            writeln!(
                dot,
                "  node_{}_rule -> node_{} [style=dashed];",
                node_id, node_id
            )
            .expect("write to String is infallible");
        }
    }

    /// Generate a textual explanation of a proof
    pub fn explain(proof: &ProofFragment, depth: usize) -> String {
        let mut explanation = String::new();
        let indent = "  ".repeat(depth);

        writeln!(explanation, "{}Prove: {:?}", indent, proof.conclusion)
            .expect("write to String is infallible");

        if proof.premise_refs.is_empty() {
            writeln!(explanation, "{}  ✓ This is a known fact", indent)
                .expect("write to String is infallible");
        } else {
            if let Some(ref rule_ref) = proof.rule_applied {
                writeln!(explanation, "{}  Using rule: {}", indent, rule_ref.rule_id)
                    .expect("write to String is infallible");
            }
            writeln!(
                explanation,
                "{}  Requires proving {} premise(s):",
                indent,
                proof.premise_refs.len()
            )
            .expect("write to String is infallible");
            for (i, premise_ref) in proof.premise_refs.iter().enumerate() {
                let hint = premise_ref
                    .conclusion_hint
                    .as_deref()
                    .unwrap_or("(premise)");
                writeln!(explanation, "{}    {}. {}", indent, i + 1, hint)
                    .expect("write to String is infallible");
            }
        }

        if let Some(complexity) = proof.metadata.complexity {
            writeln!(explanation, "{}  Complexity: {} steps", indent, complexity)
                .expect("write to String is infallible");
        }
        writeln!(explanation, "{}  Depth: {}", indent, proof.metadata.depth)
            .expect("write to String is infallible");

        explanation
    }

    /// Generate a summary of proof statistics
    pub fn proof_stats(proof: &ProofFragment) -> String {
        let mut stats = String::new();
        writeln!(stats, "Proof Statistics:").expect("write to String is infallible");
        writeln!(stats, "  ID: {}", proof.id).expect("write to String is infallible");
        writeln!(stats, "  Direct premises: {}", proof.premise_refs.len())
            .expect("write to String is infallible");

        writeln!(
            stats,
            "  Complexity: {} steps",
            proof.metadata.complexity.unwrap_or(0)
        )
        .expect("write to String is infallible");
        writeln!(stats, "  Depth: {}", proof.metadata.depth)
            .expect("write to String is infallible");
        if let Some(ref created_by) = proof.metadata.created_by {
            writeln!(stats, "  Created by: {}", created_by).expect("write to String is infallible");
        }

        if proof.premise_refs.is_empty() {
            writeln!(stats, "  Type: Fact (axiom)").expect("write to String is infallible");
        } else {
            writeln!(stats, "  Type: Rule application").expect("write to String is infallible");
            if let Some(ref rule_ref) = proof.rule_applied {
                writeln!(stats, "  Rule: {}", rule_ref.rule_id)
                    .expect("write to String is infallible");
            }
        }

        if !proof.substitution.is_empty() {
            writeln!(stats, "  Substitutions: {}", proof.substitution.len())
                .expect("write to String is infallible");
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComputationGraph, GraphNode, Predicate, TensorOp, Term};

    #[test]
    fn test_graph_to_dot() {
        let mut graph = ComputationGraph::new();

        let input = GraphNode::new(
            "input".to_string(),
            TensorOp::Input {
                name: "x".to_string(),
            },
        );
        graph.add_node(input).expect("test: should succeed");
        graph.mark_input("input".to_string());

        let relu =
            GraphNode::new("relu".to_string(), TensorOp::ReLU).add_input("input".to_string());
        graph.add_node(relu).expect("test: should succeed");
        graph.mark_output("relu".to_string());

        let dot = GraphVisualizer::to_dot(&graph);

        assert!(dot.contains("digraph ComputationGraph"));
        assert!(dot.contains("\"input\""));
        assert!(dot.contains("\"relu\""));
        assert!(dot.contains("\"input\" -> \"relu\""));
    }

    #[test]
    fn test_graph_stats() {
        let mut graph = ComputationGraph::new();

        let input = GraphNode::new(
            "input".to_string(),
            TensorOp::Input {
                name: "x".to_string(),
            },
        );
        graph.add_node(input).expect("test: should succeed");

        let relu =
            GraphNode::new("relu".to_string(), TensorOp::ReLU).add_input("input".to_string());
        graph.add_node(relu).expect("test: should succeed");

        let stats = GraphVisualizer::graph_stats(&graph);

        assert!(stats.contains("Total nodes: 2"));
        assert!(stats.contains("Input: 1"));
        assert!(stats.contains("ReLU: 1"));
    }

    #[test]
    fn test_proof_to_dot() {
        use crate::proof_storage::{ProofFragmentRef, ProofMetadata, RuleRef};

        let conclusion = Predicate::new(
            "ancestor".to_string(),
            vec![
                Term::Const(crate::Constant::String("Alice".to_string())),
                Term::Const(crate::Constant::String("Bob".to_string())),
            ],
        );

        let proof = ProofFragment {
            id: "proof_1".to_string(),
            conclusion,
            rule_applied: Some(RuleRef {
                rule_id: "ancestor_rule".to_string(),
                rule_cid: None,
                rule: None,
            }),
            premise_refs: vec![ProofFragmentRef {
                cid: ipfrs_core::Cid::default(),
                conclusion_hint: Some("parent(Alice, Bob)".to_string()),
            }],
            substitution: vec![],
            metadata: ProofMetadata {
                created_at: None,
                created_by: None,
                complexity: Some(2),
                depth: 1,
                custom: std::collections::HashMap::new(),
            },
        };

        let dot = ProofVisualizer::to_dot(&proof, 0);

        assert!(dot.contains("digraph ProofTree"));
        assert!(dot.contains("ancestor"));
        assert!(dot.contains("parent"));
    }

    #[test]
    fn test_proof_explain() {
        use crate::proof_storage::ProofMetadata;

        let conclusion = Predicate::new(
            "test".to_string(),
            vec![Term::Const(crate::Constant::String("A".to_string()))],
        );

        let proof = ProofFragment {
            id: "proof_2".to_string(),
            conclusion,
            rule_applied: None,
            premise_refs: vec![],
            substitution: vec![],
            metadata: ProofMetadata {
                created_at: None,
                created_by: None,
                complexity: None,
                depth: 0,
                custom: std::collections::HashMap::new(),
            },
        };

        let explanation = ProofVisualizer::explain(&proof, 0);

        assert!(explanation.contains("Prove"));
        assert!(explanation.contains("known fact"));
    }

    #[test]
    fn test_proof_stats() {
        use crate::proof_storage::ProofMetadata;

        let conclusion = Predicate::new(
            "test".to_string(),
            vec![Term::Const(crate::Constant::String("A".to_string()))],
        );

        let proof = ProofFragment {
            id: "proof_3".to_string(),
            conclusion,
            rule_applied: None,
            premise_refs: vec![],
            substitution: vec![],
            metadata: ProofMetadata {
                created_at: None,
                created_by: None,
                complexity: None,
                depth: 0,
                custom: std::collections::HashMap::new(),
            },
        };

        let stats = ProofVisualizer::proof_stats(&proof);

        assert!(stats.contains("Proof Statistics"));
        assert!(stats.contains("Type: Fact"));
    }
}
