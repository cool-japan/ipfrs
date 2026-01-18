//! Distributed Graph Execution Example
//!
//! This example demonstrates how to partition and execute computation graphs
//! across multiple workers in a distributed system.
//!
//! Features demonstrated:
//! - Graph partitioning across multiple workers
//! - Cross-partition dependency tracking
//! - Communication cost estimation
//! - Subgraph extraction for each worker
//!
//! Note: This example shows the partitioning framework. Full distributed
//! execution will be available when ipfrs-network integration is complete.

use ipfrs_tensorlogic::{ComputationGraph, DistributedExecutor, GraphNode, TensorOp};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Distributed Graph Execution Example ===\n");

    // Create a computation graph representing a neural network layer
    println!("1. Creating computation graph (neural network layer)...");
    let graph = create_neural_network_graph()?;

    println!("   ✓ Graph created with {} nodes", graph.node_count());
    println!("   ✓ Inputs: {}", graph.input_count());
    println!("   ✓ Outputs: {}", graph.output_count());

    // Visualize the graph structure
    println!("\n2. Graph structure:");
    print_graph_structure(&graph);

    // Create a distributed executor
    println!("\n3. Creating distributed executor...");
    let mut executor = DistributedExecutor::new().with_timeout(60000);
    println!("   ✓ Executor created with 60s timeout");

    // Define worker nodes
    let workers = vec![
        "worker_1".to_string(),
        "worker_2".to_string(),
        "worker_3".to_string(),
    ];
    println!("   ✓ Available workers: {}", workers.len());

    // Partition the graph across workers
    println!("\n4. Partitioning graph across workers...");
    executor.partition_graph(&graph, &workers)?;
    println!("   ✓ Graph partitioned successfully");

    // Analyze partitions
    println!("\n5. Partition analysis:");
    analyze_partitions(&executor, &workers);

    // Display partition details
    println!("\n6. Detailed partition breakdown:");
    for worker_id in &workers {
        display_partition_details(&executor, worker_id);
    }

    // Estimate communication costs
    println!("\n7. Communication cost analysis:");
    let total_comm_cost = estimate_total_communication(&executor, &workers);
    println!("   Total cross-partition dependencies: {}", total_comm_cost);

    // Demonstrate subgraph extraction
    println!("\n8. Subgraph verification:");
    verify_subgraphs(&executor, &workers);

    // Show execution plan
    println!("\n9. Execution plan:");
    show_execution_plan(&executor, &workers);

    println!("\n=== Summary ===");
    println!(
        "✓ Created computation graph with {} nodes",
        graph.node_count()
    );
    println!("✓ Partitioned across {} workers", executor.worker_count());
    println!(
        "✓ Total communication overhead: {} data transfers",
        total_comm_cost
    );
    println!("✓ Framework ready for distributed execution");
    println!(
        "\nNote: Full distributed execution will be available with ipfrs-network integration."
    );

    Ok(())
}

/// Create a computation graph representing a simple neural network layer
fn create_neural_network_graph() -> Result<ComputationGraph, Box<dyn std::error::Error>> {
    let mut graph = ComputationGraph::new();

    // Input layer
    let input = GraphNode::new(
        "input".to_string(),
        TensorOp::Input {
            name: "x".to_string(),
        },
    )
    .with_output_shape(vec![128, 784]); // batch_size x input_dim

    graph.add_node(input)?;
    graph.mark_input("input".to_string());

    // Weights (constant)
    let weights = GraphNode::new(
        "weights".to_string(),
        TensorOp::Constant {
            value_cid: "QmWeights123".to_string(),
        },
    )
    .with_output_shape(vec![784, 256]); // input_dim x hidden_dim

    graph.add_node(weights)?;

    // Matrix multiplication: input @ weights
    let matmul = GraphNode::new("matmul".to_string(), TensorOp::MatMul)
        .add_input("input".to_string())
        .add_input("weights".to_string())
        .with_output_shape(vec![128, 256]);

    graph.add_node(matmul)?;

    // Bias (constant)
    let bias = GraphNode::new(
        "bias".to_string(),
        TensorOp::Constant {
            value_cid: "QmBias456".to_string(),
        },
    )
    .with_output_shape(vec![256]);

    graph.add_node(bias)?;

    // Add bias: matmul + bias
    let add_bias = GraphNode::new("add_bias".to_string(), TensorOp::Add)
        .add_input("matmul".to_string())
        .add_input("bias".to_string())
        .with_output_shape(vec![128, 256]);

    graph.add_node(add_bias)?;

    // Activation: ReLU
    let relu = GraphNode::new("relu".to_string(), TensorOp::ReLU)
        .add_input("add_bias".to_string())
        .with_output_shape(vec![128, 256]);

    graph.add_node(relu)?;

    // Batch normalization (simplified as multiply + add)
    let scale = GraphNode::new(
        "bn_scale".to_string(),
        TensorOp::Constant {
            value_cid: "QmScale789".to_string(),
        },
    )
    .with_output_shape(vec![256]);

    graph.add_node(scale)?;

    let bn_mul = GraphNode::new("bn_mul".to_string(), TensorOp::Mul)
        .add_input("relu".to_string())
        .add_input("bn_scale".to_string())
        .with_output_shape(vec![128, 256]);

    graph.add_node(bn_mul)?;

    let bn_offset = GraphNode::new(
        "bn_offset".to_string(),
        TensorOp::Constant {
            value_cid: "QmOffset101".to_string(),
        },
    )
    .with_output_shape(vec![256]);

    graph.add_node(bn_offset)?;

    let output = GraphNode::new("output".to_string(), TensorOp::Add)
        .add_input("bn_mul".to_string())
        .add_input("bn_offset".to_string())
        .with_output_shape(vec![128, 256]);

    graph.add_node(output)?;
    graph.mark_output("output".to_string());

    Ok(graph)
}

/// Print the graph structure
fn print_graph_structure(graph: &ComputationGraph) {
    println!("   Nodes:");
    for (id, node) in &graph.nodes {
        let inputs = if node.inputs.is_empty() {
            "none".to_string()
        } else {
            node.inputs.join(", ")
        };
        println!("     - {} ({:?}) <- [{}]", id, node.op, inputs);
    }
}

/// Analyze and display partition information
fn analyze_partitions(executor: &DistributedExecutor, workers: &[String]) {
    for worker_id in workers {
        if let Some(partition) = executor.get_partition(worker_id) {
            let comm_cost = executor.estimate_communication_cost(worker_id);
            println!(
                "   {} → {} nodes, {} communication edges",
                worker_id,
                partition.size(),
                comm_cost
            );
        }
    }
}

/// Display detailed partition information
fn display_partition_details(executor: &DistributedExecutor, worker_id: &str) {
    if let Some(partition) = executor.get_partition(worker_id) {
        println!("\n   Worker: {}", worker_id);
        println!("   ├─ Nodes ({}):", partition.nodes.len());
        for node_id in &partition.nodes {
            if let Some(assignment) = executor.get_assignment(node_id) {
                println!("   │  ├─ {} (priority: {})", node_id, assignment.priority);
            }
        }

        if !partition.external_inputs.is_empty() {
            println!(
                "   ├─ External inputs ({}):",
                partition.external_inputs.len()
            );
            for (input_id, source_worker) in &partition.external_inputs {
                println!("   │  ├─ {} ← from {}", input_id, source_worker);
            }
        }

        if !partition.external_outputs.is_empty() {
            println!(
                "   └─ External outputs ({}): {:?}",
                partition.external_outputs.len(),
                partition.external_outputs
            );
        }
    }
}

/// Estimate total communication cost
fn estimate_total_communication(executor: &DistributedExecutor, workers: &[String]) -> usize {
    workers
        .iter()
        .map(|w| executor.estimate_communication_cost(w))
        .sum()
}

/// Verify subgraphs are properly created
fn verify_subgraphs(executor: &DistributedExecutor, workers: &[String]) {
    for worker_id in workers {
        if let Some(partition) = executor.get_partition(worker_id) {
            if let Some(subgraph) = &partition.subgraph {
                println!(
                    "   {} → Subgraph: {} nodes, {} inputs, {} outputs",
                    worker_id,
                    subgraph.node_count(),
                    subgraph.input_count(),
                    subgraph.output_count()
                );
            }
        }
    }
}

/// Show the execution plan
fn show_execution_plan(executor: &DistributedExecutor, workers: &[String]) {
    println!("   Execution order (by priority):");

    // Collect all assignments and sort by priority
    let mut all_assignments: Vec<_> = workers
        .iter()
        .flat_map(|worker_id| {
            if let Some(partition) = executor.get_partition(worker_id) {
                partition
                    .nodes
                    .iter()
                    .filter_map(|node_id| executor.get_assignment(node_id))
                    .collect::<Vec<_>>()
            } else {
                vec![]
            }
        })
        .collect();

    all_assignments.sort_by_key(|a| a.priority);

    for (i, assignment) in all_assignments.iter().enumerate() {
        println!(
            "   {}. {} @ {} (priority: {})",
            i + 1,
            assignment.node_id,
            assignment.worker_id,
            assignment.priority
        );
    }
}
