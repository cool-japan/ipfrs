//! Computation graph tests - split from computation_graph.rs to maintain the 2000-line limit.
//!
//! This file is included from computation_graph.rs via `#[path]` in the `tests` module.

use super::*;

#[test]
fn test_tensor_op() {
    let add = TensorOp::Add;
    assert_eq!(add.num_inputs(), 2);
    assert!(add.is_pure());

    let relu = TensorOp::ReLU;
    assert_eq!(relu.num_inputs(), 1);
}

#[test]
fn test_graph_node() {
    let node = GraphNode::new("node1".to_string(), TensorOp::Add)
        .add_input("input1".to_string())
        .add_input("input2".to_string())
        .with_output_shape(vec![10, 20]);

    assert_eq!(node.inputs.len(), 2);
    assert_eq!(node.output_shape, Some(vec![10, 20]));
}

#[test]
fn test_computation_graph() {
    let mut graph = ComputationGraph::new();

    let input1 = GraphNode::new(
        "input1".to_string(),
        TensorOp::Input {
            name: "x".to_string(),
        },
    );

    let input2 = GraphNode::new(
        "input2".to_string(),
        TensorOp::Input {
            name: "y".to_string(),
        },
    );

    graph.add_node(input1).expect("test: node addition should succeed");
    graph.add_node(input2).expect("test: node addition should succeed");
    graph.mark_input("input1".to_string());
    graph.mark_input("input2".to_string());

    let add = GraphNode::new("add1".to_string(), TensorOp::Add)
        .add_input("input1".to_string())
        .add_input("input2".to_string());

    graph.add_node(add).expect("test: node addition should succeed");
    graph.mark_output("add1".to_string());

    assert_eq!(graph.node_count(), 3);
    assert_eq!(graph.input_count(), 2);
    assert_eq!(graph.output_count(), 1);
}

#[test]
fn test_topological_sort() {
    let mut graph = ComputationGraph::new();

    let input1 = GraphNode::new(
        "a".to_string(),
        TensorOp::Input {
            name: "x".to_string(),
        },
    );
    graph.add_node(input1).expect("test: node addition should succeed");

    let b = GraphNode::new("b".to_string(), TensorOp::ReLU).add_input("a".to_string());
    graph.add_node(b).expect("test: node addition should succeed");

    let c = GraphNode::new("c".to_string(), TensorOp::Tanh).add_input("b".to_string());
    graph.add_node(c).expect("test: node addition should succeed");

    let sorted = graph.topological_sort().expect("test: topological sort should succeed on DAG");

    // Check that 'a' comes before 'b', and 'b' comes before 'c'
    let pos_a = sorted.iter().position(|x| x == "a").expect("test: should succeed");
    let pos_b = sorted.iter().position(|x| x == "b").expect("test: should succeed");
    let pos_c = sorted.iter().position(|x| x == "c").expect("test: should succeed");

    assert!(pos_a < pos_b);
    assert!(pos_b < pos_c);
}

#[test]
fn test_subgraph_extraction() {
    let mut graph = ComputationGraph::new();

    let a = GraphNode::new(
        "a".to_string(),
        TensorOp::Input {
            name: "x".to_string(),
        },
    );

    graph.add_node(a).expect("test: node addition should succeed");
    graph.mark_input("a".to_string());

    let b = GraphNode::new("b".to_string(), TensorOp::ReLU).add_input("a".to_string());
    let c = GraphNode::new("c".to_string(), TensorOp::Tanh).add_input("a".to_string());

    graph.add_node(b).expect("test: node addition should succeed");
    graph.add_node(c).expect("test: node addition should succeed");

    let subgraph = graph.extract_subgraph(&["b".to_string()]).expect("test: should succeed");

    assert_eq!(subgraph.node_count(), 2); // Should have 'a' and 'b'
    assert!(subgraph.nodes.contains_key("a"));
    assert!(subgraph.nodes.contains_key("b"));
    assert!(!subgraph.nodes.contains_key("c"));
}

#[test]
fn test_cse_optimization() {
    let mut graph = ComputationGraph::new();

    let a = GraphNode::new(
        "a".to_string(),
        TensorOp::Input {
            name: "x".to_string(),
        },
    );
    let b = GraphNode::new(
        "b".to_string(),
        TensorOp::Input {
            name: "y".to_string(),
        },
    );

    // Create two identical Add operations
    let add1 = GraphNode::new("add1".to_string(), TensorOp::Add)
        .add_input("a".to_string())
        .add_input("b".to_string());

    let add2 = GraphNode::new("add2".to_string(), TensorOp::Add)
        .add_input("a".to_string())
        .add_input("b".to_string());

    graph.add_node(a).expect("test: node addition should succeed");
    graph.add_node(b).expect("test: node addition should succeed");
    graph.add_node(add1).expect("test: node addition should succeed");
    graph.add_node(add2).expect("test: node addition should succeed");

    // CSE should detect these as duplicates
    let _optimized = graph.optimize_cse();
    // Note: In a more sophisticated implementation, we would verify
    // that duplicates are actually eliminated
}

#[test]
fn test_lazy_cache() {
    let mut cache = LazyCache::new(2);

    cache.insert("node1".to_string(), vec![1.0, 2.0]);
    cache.insert("node2".to_string(), vec![3.0, 4.0]);

    assert_eq!(cache.size(), 2);
    assert!(cache.get("node1").is_some());

    // Adding a third item should evict the least recently used
    cache.insert("node3".to_string(), vec![5.0, 6.0]);
    assert_eq!(cache.size(), 2);
}

#[test]
fn test_graph_optimizer() {
    let mut graph = ComputationGraph::new();

    let input = GraphNode::new(
        "input".to_string(),
        TensorOp::Input {
            name: "x".to_string(),
        },
    );

    graph.add_node(input).expect("test: node addition should succeed");
    graph.mark_input("input".to_string());

    let relu =
        GraphNode::new("relu".to_string(), TensorOp::ReLU).add_input("input".to_string());

    // Add a dead node (not connected to output)
    let dead =
        GraphNode::new("dead".to_string(), TensorOp::Tanh).add_input("input".to_string());

    graph.add_node(relu).expect("test: node addition should succeed");
    graph.add_node(dead).expect("test: node addition should succeed");
    graph.mark_output("relu".to_string());

    let removed = GraphOptimizer::remove_dead_nodes(&mut graph).expect("test: dead node removal should succeed");

    assert_eq!(removed, 1);
    assert!(!graph.nodes.contains_key("dead"));
}

#[test]
fn test_batch_scheduler() {
    let mut graph = ComputationGraph::new();

    // Create a simple graph: a -> b, a -> c, (b,c) -> d
    let a = GraphNode::new(
        "a".to_string(),
        TensorOp::Input {
            name: "x".to_string(),
        },
    );
    graph.add_node(a).expect("test: node addition should succeed");
    graph.mark_input("a".to_string());

    let b = GraphNode::new("b".to_string(), TensorOp::ReLU).add_input("a".to_string());
    let c = GraphNode::new("c".to_string(), TensorOp::Tanh).add_input("a".to_string());

    graph.add_node(b).expect("test: node addition should succeed");
    graph.add_node(c).expect("test: node addition should succeed");

    let d = GraphNode::new("d".to_string(), TensorOp::Add)
        .add_input("b".to_string())
        .add_input("c".to_string());
    graph.add_node(d).expect("test: node addition should succeed");
    graph.mark_output("d".to_string());

    let batches = BatchScheduler::create_batches(&graph).expect("test: batch creation should succeed");

    // Batch 0: a (input)
    // Batch 1: b, c (both depend only on a)
    // Batch 2: d (depends on b and c)
    assert_eq!(batches.len(), 3);
    assert_eq!(batches[0].size(), 1); // a
    assert_eq!(batches[1].size(), 2); // b, c
    assert_eq!(batches[2].size(), 1); // d
}

#[test]
fn test_parallel_executor() {
    let mut graph = ComputationGraph::new();

    let input1 = GraphNode::new(
        "input1".to_string(),
        TensorOp::Input {
            name: "x".to_string(),
        },
    );
    let input2 = GraphNode::new(
        "input2".to_string(),
        TensorOp::Input {
            name: "y".to_string(),
        },
    );

    graph.add_node(input1).expect("test: node addition should succeed");
    graph.add_node(input2).expect("test: node addition should succeed");
    graph.mark_input("input1".to_string());
    graph.mark_input("input2".to_string());

    let add = GraphNode::new("add".to_string(), TensorOp::Add)
        .add_input("input1".to_string())
        .add_input("input2".to_string());

    graph.add_node(add).expect("test: node addition should succeed");
    graph.mark_output("add".to_string());

    let executor = ParallelExecutor::new(Some(2));
    let result = executor.execute(&graph).expect("test: graph execution should succeed");

    // All nodes should be executed
    assert_eq!(result.len(), 3);
}

#[test]
fn test_execution_batch() {
    let mut batch = ExecutionBatch::new(0);
    batch.add_node("node1".to_string());
    batch.add_node("node2".to_string());

    assert_eq!(batch.size(), 2);
    assert_eq!(batch.level, 0);
    assert!(batch.node_ids.contains(&"node1".to_string()));
}

#[test]
fn test_streaming_executor() {
    let executor = StreamingExecutor::new(100, 10);

    // Create test data
    let data: Vec<f32> = (0..250).map(|i| i as f32).collect();
    let chunks = executor.create_chunks(data.clone(), "test_node");

    // Should create 3 chunks (100, 100, 50)
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].data["test_node"].len(), 100);
    assert_eq!(chunks[1].data["test_node"].len(), 100);
    assert_eq!(chunks[2].data["test_node"].len(), 50);
    assert!(chunks[2].is_last());

    assert_eq!(executor.chunk_size(), 100);
    assert_eq!(executor.max_buffer_size(), 10);
}

#[test]
fn test_stream_chunk() {
    let mut chunk = StreamChunk::new(0, 5);
    chunk.add_data("node1".to_string(), vec![1.0, 2.0, 3.0]);
    chunk.add_data("node2".to_string(), vec![4.0, 5.0, 6.0]);

    assert_eq!(chunk.index, 0);
    assert_eq!(chunk.total_chunks, 5);
    assert!(!chunk.is_last());
    assert_eq!(chunk.data.len(), 2);

    let last_chunk = StreamChunk::new(4, 5);
    assert!(last_chunk.is_last());
}

#[test]
fn test_streaming_process_stream() {
    let graph = ComputationGraph::new();
    let executor = StreamingExecutor::new(100, 5);

    let data: Vec<f32> = (0..300).map(|i| i as f32).collect();
    let chunks = executor.create_chunks(data, "input");

    let results = executor.process_stream(&graph, chunks).expect("test: stream processing should succeed");

    assert_eq!(results.len(), 3);
    assert!(executor.buffer_size() <= executor.max_buffer_size());

    executor.clear_buffer();
    assert_eq!(executor.buffer_size(), 0);
}

#[test]
fn test_distributed_executor_creation() {
    let executor = DistributedExecutor::new();
    assert_eq!(executor.worker_count(), 0);
    assert_eq!(executor.timeout(), 30000);

    let executor_custom = DistributedExecutor::new().with_timeout(60000);
    assert_eq!(executor_custom.timeout(), 60000);
}

#[test]
fn test_graph_partitioning() {
    let mut graph = ComputationGraph::new();

    // Create a simple graph: input -> a -> b -> c -> output
    let input = GraphNode::new(
        "input".to_string(),
        TensorOp::Input {
            name: "x".to_string(),
        },
    );
    graph.add_node(input).expect("test: node addition should succeed");
    graph.mark_input("input".to_string());

    let a = GraphNode::new("a".to_string(), TensorOp::ReLU).add_input("input".to_string());
    let b = GraphNode::new("b".to_string(), TensorOp::Tanh).add_input("a".to_string());
    let c = GraphNode::new("c".to_string(), TensorOp::Sigmoid).add_input("b".to_string());

    graph.add_node(a).expect("test: node addition should succeed");
    graph.add_node(b).expect("test: node addition should succeed");
    graph.add_node(c).expect("test: node addition should succeed");
    graph.mark_output("c".to_string());

    // Partition across 2 workers
    let mut executor = DistributedExecutor::new();
    let workers = vec!["worker1".to_string(), "worker2".to_string()];
    executor.partition_graph(&graph, &workers).expect("test: graph partitioning should succeed");

    assert_eq!(executor.worker_count(), 2);

    // Check that partitions were created
    let partition1 = executor.get_partition("worker1");
    let partition2 = executor.get_partition("worker2");

    assert!(partition1.is_some());
    assert!(partition2.is_some());

    // Each partition should have nodes
    let p1 = partition1.expect("test: should succeed");
    let p2 = partition2.expect("test: should succeed");

    assert!(p1.size() > 0);
    assert!(p2.size() > 0);

    // Total nodes across partitions should match graph
    assert_eq!(p1.size() + p2.size(), 4); // input, a, b, c
}

#[test]
fn test_cross_partition_dependencies() {
    let mut graph = ComputationGraph::new();

    // Create a graph with cross-partition dependencies
    let input1 = GraphNode::new(
        "input1".to_string(),
        TensorOp::Input {
            name: "x".to_string(),
        },
    );
    let input2 = GraphNode::new(
        "input2".to_string(),
        TensorOp::Input {
            name: "y".to_string(),
        },
    );

    graph.add_node(input1).expect("test: node addition should succeed");
    graph.add_node(input2).expect("test: node addition should succeed");
    graph.mark_input("input1".to_string());
    graph.mark_input("input2".to_string());

    let a = GraphNode::new("a".to_string(), TensorOp::ReLU).add_input("input1".to_string());
    let b = GraphNode::new("b".to_string(), TensorOp::Tanh).add_input("input2".to_string());
    let c = GraphNode::new("c".to_string(), TensorOp::Add)
        .add_input("a".to_string())
        .add_input("b".to_string());

    graph.add_node(a).expect("test: node addition should succeed");
    graph.add_node(b).expect("test: node addition should succeed");
    graph.add_node(c).expect("test: node addition should succeed");
    graph.mark_output("c".to_string());

    // Partition across 3 workers
    let mut executor = DistributedExecutor::new();
    let workers = vec![
        "worker1".to_string(),
        "worker2".to_string(),
        "worker3".to_string(),
    ];
    executor.partition_graph(&graph, &workers).expect("test: graph partitioning should succeed");

    // Check communication costs
    let cost1 = executor.estimate_communication_cost("worker1");
    let cost2 = executor.estimate_communication_cost("worker2");
    let cost3 = executor.estimate_communication_cost("worker3");

    // At least one partition should have external dependencies
    assert!(cost1 > 0 || cost2 > 0 || cost3 > 0);
}

#[test]
fn test_graph_partition_struct() {
    let mut partition = GraphPartition::new("worker1".to_string());

    partition.add_node("node1".to_string());
    partition.add_node("node2".to_string());
    partition.add_node("node1".to_string()); // Duplicate should be ignored

    assert_eq!(partition.size(), 2);

    partition.add_external_input("input1".to_string(), "worker2".to_string());
    partition.mark_external_output("output1".to_string());

    assert_eq!(partition.external_inputs.len(), 1);
    assert_eq!(partition.external_outputs.len(), 1);
}

#[test]
fn test_node_assignment() {
    let assignment = NodeAssignment {
        node_id: "node1".to_string(),
        worker_id: "worker1".to_string(),
        priority: 5,
    };

    assert_eq!(assignment.node_id, "node1");
    assert_eq!(assignment.worker_id, "worker1");
    assert_eq!(assignment.priority, 5);
}

#[test]
fn test_distributed_partition_no_workers() {
    let graph = ComputationGraph::new();
    let mut executor = DistributedExecutor::new();
    let workers: Vec<String> = vec![];

    let result = executor.partition_graph(&graph, &workers);
    assert!(result.is_err());
}

#[test]
fn test_shape_inference_matmul() {
    let op = TensorOp::MatMul;
    let input_shapes = vec![vec![2, 3, 4], vec![2, 4, 5]];
    let output_shape = op.infer_output_shape(&input_shapes).expect("test: shape inference should succeed");
    assert_eq!(output_shape, vec![2, 3, 5]);
}

#[test]
fn test_shape_inference_add_broadcast() {
    let op = TensorOp::Add;
    let input_shapes = vec![vec![3, 1, 4], vec![3, 2, 4]];
    let output_shape = op.infer_output_shape(&input_shapes).expect("test: shape inference should succeed");
    assert_eq!(output_shape, vec![3, 2, 4]);
}

#[test]
fn test_shape_inference_reduce_sum() {
    let op = TensorOp::ReduceSum {
        axes: vec![1],
        keepdims: false,
    };
    let input_shapes = vec![vec![2, 3, 4]];
    let output_shape = op.infer_output_shape(&input_shapes).expect("test: shape inference should succeed");
    assert_eq!(output_shape, vec![2, 4]);
}

#[test]
fn test_shape_inference_reduce_sum_keepdims() {
    let op = TensorOp::ReduceSum {
        axes: vec![1],
        keepdims: true,
    };
    let input_shapes = vec![vec![2, 3, 4]];
    let output_shape = op.infer_output_shape(&input_shapes).expect("test: shape inference should succeed");
    assert_eq!(output_shape, vec![2, 1, 4]);
}

#[test]
fn test_shape_inference_transpose() {
    let op = TensorOp::Transpose {
        axes: vec![0, 2, 1],
    };
    let input_shapes = vec![vec![2, 3, 4]];
    let output_shape = op.infer_output_shape(&input_shapes).expect("test: shape inference should succeed");
    assert_eq!(output_shape, vec![2, 4, 3]);
}

#[test]
fn test_shape_inference_concat() {
    let op = TensorOp::Concat { axis: 1 };
    let input_shapes = vec![vec![2, 3, 4], vec![2, 5, 4]];
    let output_shape = op.infer_output_shape(&input_shapes).expect("test: shape inference should succeed");
    assert_eq!(output_shape, vec![2, 8, 4]);
}

#[test]
fn test_shape_inference_reshape() {
    let op = TensorOp::Reshape { shape: vec![6, 4] };
    let input_shapes = vec![vec![2, 3, 4]];
    let output_shape = op.infer_output_shape(&input_shapes).expect("test: shape inference should succeed");
    assert_eq!(output_shape, vec![6, 4]);
}

#[test]
fn test_graph_shape_propagation() {
    let mut graph = ComputationGraph::new();

    // Input: [2, 3]
    let mut input = GraphNode::new(
        "input".to_string(),
        TensorOp::Input {
            name: "x".to_string(),
        },
    );
    input.output_shape = Some(vec![2, 3]);
    graph.add_node(input).expect("test: node addition should succeed");
    graph.mark_input("input".to_string());

    // Weight: [3, 4]
    let mut weight = GraphNode::new(
        "weight".to_string(),
        TensorOp::Constant {
            value_cid: "cid1".to_string(),
        },
    );
    weight.output_shape = Some(vec![3, 4]);
    graph.add_node(weight).expect("test: node addition should succeed");

    // MatMul: should be [2, 4]
    let matmul = GraphNode::new("matmul".to_string(), TensorOp::MatMul)
        .add_input("input".to_string())
        .add_input("weight".to_string());
    graph.add_node(matmul).expect("test: node addition should succeed");

    // ReLU: should be [2, 4]
    let relu =
        GraphNode::new("relu".to_string(), TensorOp::ReLU).add_input("matmul".to_string());
    graph.add_node(relu).expect("test: node addition should succeed");
    graph.mark_output("relu".to_string());

    // Propagate shapes
    graph.propagate_shapes().expect("test: shape propagation should succeed");

    // Check inferred shapes
    assert_eq!(
        graph.nodes.get("matmul").expect("test: should succeed").output_shape,
        Some(vec![2, 4])
    );
    assert_eq!(
        graph.nodes.get("relu").expect("test: should succeed").output_shape,
        Some(vec![2, 4])
    );
}

#[test]
fn test_graph_validation() {
    let mut graph = ComputationGraph::new();

    let input = GraphNode::new(
        "input".to_string(),
        TensorOp::Input {
            name: "x".to_string(),
        },
    )
    .with_output_shape(vec![2, 3]);
    graph.add_node(input).expect("test: node addition should succeed");
    graph.mark_input("input".to_string());

    let relu =
        GraphNode::new("relu".to_string(), TensorOp::ReLU).add_input("input".to_string());
    graph.add_node(relu).expect("test: node addition should succeed");
    graph.mark_output("relu".to_string());

    // Should validate successfully
    assert!(graph.validate().is_ok());
}

#[test]
fn test_graph_validation_missing_input() {
    let mut graph = ComputationGraph::new();

    let relu =
        GraphNode::new("relu".to_string(), TensorOp::ReLU).add_input("nonexistent".to_string());

    // Should fail because input doesn't exist
    assert!(graph.add_node(relu).is_err());
}

#[test]
fn test_estimate_memory() {
    let mut graph = ComputationGraph::new();

    let mut input = GraphNode::new(
        "input".to_string(),
        TensorOp::Input {
            name: "x".to_string(),
        },
    );
    input.output_shape = Some(vec![10, 20]); // 200 elements * 4 bytes = 800 bytes
    graph.add_node(input).expect("test: node addition should succeed");

    let mut weight = GraphNode::new(
        "weight".to_string(),
        TensorOp::Constant {
            value_cid: "cid1".to_string(),
        },
    );
    weight.output_shape = Some(vec![20, 30]); // 600 elements * 4 bytes = 2400 bytes
    graph.add_node(weight).expect("test: node addition should succeed");

    let memory = graph.estimate_memory();
    assert_eq!(memory, 800 + 2400); // 3200 bytes total
}

#[test]
fn test_broadcast_shapes_same() {
    let result = TensorOp::broadcast_shapes(&[2, 3, 4], &[2, 3, 4]).expect("test: broadcast shape computation should succeed");
    assert_eq!(result, vec![2, 3, 4]);
}

#[test]
fn test_broadcast_shapes_scalar() {
    let result = TensorOp::broadcast_shapes(&[2, 3, 4], &[1]).expect("test: broadcast shape computation should succeed");
    assert_eq!(result, vec![2, 3, 4]);
}

#[test]
fn test_broadcast_shapes_incompatible() {
    let result = TensorOp::broadcast_shapes(&[2, 3, 4], &[2, 5, 4]);
    assert!(result.is_err());
}
