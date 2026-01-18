//! Visualization demo
//!
//! This example demonstrates how to visualize computation graphs and proof trees
//! using the DOT format (Graphviz).
//!
//! Run with:
//! ```bash
//! cargo run --example visualization_demo
//! ```
//!
//! To render the generated DOT files:
//! ```bash
//! dot -Tpng graph.dot -o graph.png
//! dot -Tsvg proof.dot -o proof.svg
//! ```

use ipfrs_tensorlogic::{
    ComputationGraph, Constant, GraphNode, GraphVisualizer, InferenceEngine, KnowledgeBase,
    Predicate, ProofFragment, ProofFragmentRef, ProofMetadata, ProofVisualizer, Rule, RuleRef,
    TensorOp, Term,
};
use std::collections::HashMap;
use std::fs;

fn main() {
    println!("=== IPFRS TensorLogic Visualization Demo ===\n");

    // Part 1: Visualize a computation graph
    println!("1. Creating computation graph...");
    let mut graph = ComputationGraph::new();

    // Create input
    let input = GraphNode::new(
        "input".to_string(),
        TensorOp::Input {
            name: "x".to_string(),
        },
    );
    graph.add_node(input).unwrap();
    graph.mark_input("input".to_string());

    // Create first layer (Linear: MatMul + Add)
    let weights1 = GraphNode::new(
        "weights1".to_string(),
        TensorOp::Constant {
            value_cid: "bafkreiabc123456".to_string(),
        },
    );
    graph.add_node(weights1).unwrap();

    let matmul1 = GraphNode::new("matmul1".to_string(), TensorOp::MatMul)
        .add_input("input".to_string())
        .add_input("weights1".to_string());
    graph.add_node(matmul1).unwrap();

    let bias1 = GraphNode::new(
        "bias1".to_string(),
        TensorOp::Constant {
            value_cid: "bafkreidef789012".to_string(),
        },
    );
    graph.add_node(bias1).unwrap();

    let add1 = GraphNode::new("add1".to_string(), TensorOp::Add)
        .add_input("matmul1".to_string())
        .add_input("bias1".to_string());
    graph.add_node(add1).unwrap();

    // ReLU activation
    let relu = GraphNode::new("relu".to_string(), TensorOp::ReLU).add_input("add1".to_string());
    graph.add_node(relu).unwrap();

    // Layer Normalization
    let ln = GraphNode::new(
        "layer_norm".to_string(),
        TensorOp::LayerNorm {
            normalized_shape: vec![128],
            eps: 1e-5,
        },
    )
    .add_input("relu".to_string());
    graph.add_node(ln).unwrap();

    // Dropout
    let dropout = GraphNode::new("dropout".to_string(), TensorOp::Dropout { p: 0.1 })
        .add_input("layer_norm".to_string());
    graph.add_node(dropout).unwrap();

    // Second layer
    let weights2 = GraphNode::new(
        "weights2".to_string(),
        TensorOp::Constant {
            value_cid: "bafkreighi345678".to_string(),
        },
    );
    graph.add_node(weights2).unwrap();

    let matmul2 = GraphNode::new("matmul2".to_string(), TensorOp::MatMul)
        .add_input("dropout".to_string())
        .add_input("weights2".to_string());
    graph.add_node(matmul2).unwrap();

    // Softmax output
    let softmax = GraphNode::new("output".to_string(), TensorOp::Softmax { axis: -1 })
        .add_input("matmul2".to_string());
    graph.add_node(softmax).unwrap();
    graph.mark_output("output".to_string());

    // Export to DOT
    println!("2. Exporting computation graph to DOT format...");
    let dot = GraphVisualizer::to_dot(&graph);
    fs::write("graph.dot", &dot).unwrap();
    println!("   ✓ Saved to graph.dot");

    // Print statistics
    println!("\n3. Graph Statistics:");
    let stats = GraphVisualizer::graph_stats(&graph);
    println!("{}", stats);

    // Part 2: Visualize a proof tree
    println!("\n4. Creating proof tree...");

    // Create knowledge base
    let mut kb = KnowledgeBase::new();

    // Add facts
    kb.add_fact(Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("Alice".to_string())),
            Term::Const(Constant::String("Bob".to_string())),
        ],
    ));

    kb.add_fact(Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("Bob".to_string())),
            Term::Const(Constant::String("Carol".to_string())),
        ],
    ));

    // Add ancestor rule
    kb.add_rule(Rule::new(
        Predicate::new(
            "ancestor".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        ),
        vec![Predicate::new(
            "parent".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        )],
    ));

    kb.add_rule(Rule::new(
        Predicate::new(
            "ancestor".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Z".to_string())],
        ),
        vec![
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            Predicate::new(
                "ancestor".to_string(),
                vec![Term::Var("Y".to_string()), Term::Var("Z".to_string())],
            ),
        ],
    ));

    // Run query
    let engine = InferenceEngine::new();
    let query = Predicate::new(
        "ancestor".to_string(),
        vec![
            Term::Const(Constant::String("Alice".to_string())),
            Term::Var("Y".to_string()),
        ],
    );

    let results = engine.query(&query, &kb).unwrap();
    println!("   Found {} ancestor(s) of Alice", results.len());

    // Create a sample proof fragment
    let conclusion = Predicate::new(
        "ancestor".to_string(),
        vec![
            Term::Const(Constant::String("Alice".to_string())),
            Term::Const(Constant::String("Carol".to_string())),
        ],
    );

    let proof = ProofFragment {
        id: "proof_1".to_string(),
        conclusion,
        rule_applied: Some(RuleRef {
            rule_id: "ancestor_transitive".to_string(),
            rule_cid: None,
            rule: None,
        }),
        premise_refs: vec![
            ProofFragmentRef {
                cid: ipfrs_core::Cid::default(),
                conclusion_hint: Some("parent(Alice, Bob)".to_string()),
            },
            ProofFragmentRef {
                cid: ipfrs_core::Cid::default(),
                conclusion_hint: Some("ancestor(Bob, Carol)".to_string()),
            },
        ],
        substitution: vec![
            (
                "X".to_string(),
                Term::Const(Constant::String("Alice".to_string())),
            ),
            (
                "Y".to_string(),
                Term::Const(Constant::String("Bob".to_string())),
            ),
            (
                "Z".to_string(),
                Term::Const(Constant::String("Carol".to_string())),
            ),
        ],
        metadata: ProofMetadata {
            created_at: Some(1704070800),
            created_by: Some("inference_engine".to_string()),
            complexity: Some(3),
            depth: 2,
            custom: HashMap::new(),
        },
    };

    // Export proof to DOT
    println!("\n5. Exporting proof tree to DOT format...");
    let proof_dot = ProofVisualizer::to_dot(&proof, 0);
    fs::write("proof.dot", &proof_dot).unwrap();
    println!("   ✓ Saved to proof.dot");

    // Generate proof explanation
    println!("\n6. Proof Explanation:");
    let explanation = ProofVisualizer::explain(&proof, 0);
    println!("{}", explanation);

    // Print proof statistics
    println!("\n7. Proof Statistics:");
    let proof_stats = ProofVisualizer::proof_stats(&proof);
    println!("{}", proof_stats);

    println!("\n=== Visualization Demo Complete ===");
    println!("\nTo visualize the generated files:");
    println!("  dot -Tpng graph.dot -o graph.png");
    println!("  dot -Tsvg proof.dot -o proof.svg");
    println!("\nOr open in an online viewer:");
    println!("  https://dreampuf.github.io/GraphvizOnline/");
}
