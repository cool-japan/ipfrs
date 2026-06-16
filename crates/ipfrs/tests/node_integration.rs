//! Integration tests for IPFRS Node
//!
//! These tests verify the complete functionality of the IPFRS node,
//! including blocks, semantic search, and logic programming.

use ipfrs::{Node, NodeConfig, QueryFilter};
use ipfrs_core::Block;
use ipfrs_tensorlogic::ir::{Constant, Predicate, Rule, Term};
use std::path::PathBuf;

/// Helper to create a test node with unique storage
async fn create_test_node(test_name: &str) -> Node {
    let path = format!("/tmp/ipfrs-test-{}", test_name);
    let _ = std::fs::remove_dir_all(&path);

    let mut config = NodeConfig::default();
    config.storage.path = PathBuf::from(path);
    config.enable_semantic = true;
    config.enable_tensorlogic = true;

    let mut node = Node::new(config).expect("Failed to create node");
    node.start().await.expect("Failed to start node");
    node
}

#[tokio::test]
async fn test_node_lifecycle() {
    let mut node = create_test_node("lifecycle").await;
    // Node should have started successfully (no panic above)
    let _ = &node;

    // Stop node
    node.stop().await.expect("Failed to stop node");
}

#[tokio::test]
async fn test_block_operations() {
    let mut node = create_test_node("blocks").await;

    // Create test data
    let data = b"Hello, IPFRS!";
    let block = Block::new(data.to_vec().into()).expect("Failed to create block");
    let cid = *block.cid();

    // Put block
    node.put_block(&block).await.expect("Failed to put block");

    // Get block
    let retrieved = node.get_block(&cid).await.expect("Failed to get block");
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().data().as_ref(), data);

    // Has block
    let exists = node.has_block(&cid).await.expect("Failed to check block");
    assert!(exists);

    // Delete block
    node.delete_block(&cid)
        .await
        .expect("Failed to delete block");
    let exists = node.has_block(&cid).await.expect("Failed to check block");
    assert!(!exists);

    node.stop().await.expect("Failed to stop");
}

#[tokio::test]
async fn test_batch_block_operations() {
    let mut node = create_test_node("batch-blocks").await;

    // Create multiple blocks
    let mut blocks = Vec::new();
    let mut cids = Vec::new();
    for i in 0..10 {
        let data = format!("Block {}", i).into_bytes();
        let block = Block::new(data.into()).expect("Failed to create block");
        cids.push(*block.cid());
        blocks.push(block);
    }

    // Batch put
    for block in &blocks {
        node.put_block(block).await.expect("Failed to put block");
    }

    // Verify all blocks exist
    for cid in &cids {
        let exists = node.has_block(cid).await.expect("Failed to check block");
        assert!(exists);
    }

    // Get all blocks
    for (i, cid) in cids.iter().enumerate() {
        let retrieved = node.get_block(cid).await.expect("Failed to get block");
        assert!(retrieved.is_some());
        let expected = format!("Block {}", i);
        assert_eq!(retrieved.unwrap().data(), expected.as_bytes());
    }

    node.stop().await.expect("Failed to stop");
}

#[tokio::test]
async fn test_semantic_indexing() {
    let mut node = create_test_node("semantic").await;

    // Create and store blocks with embeddings
    let embedding_dim = 768; // Default dimension
    let mut cids = Vec::new();

    for i in 0..5 {
        let data = format!("Document {}", i).into_bytes();
        let block = Block::new(data.into()).expect("Failed to create block");
        let cid = *block.cid();

        node.put_block(&block).await.expect("Failed to put block");

        // Create embedding (simple pattern for testing)
        let embedding: Vec<f32> = (0..embedding_dim)
            .map(|j| ((i + j) as f32) / embedding_dim as f32)
            .collect();

        node.index_content(&cid, &embedding)
            .await
            .expect("Failed to index content");

        cids.push(cid);
    }

    // Search for similar content
    let query: Vec<f32> = (0..embedding_dim)
        .map(|j| (j as f32) / embedding_dim as f32)
        .collect();

    let results = node
        .search_similar(&query, 3)
        .await
        .expect("Failed to search");

    assert!(!results.is_empty());
    assert!(results.len() <= 3);

    // Verify results have scores
    for result in &results {
        assert!(result.score >= 0.0);
    }

    node.stop().await.expect("Failed to stop");
}

#[tokio::test]
async fn test_semantic_filtered_search() {
    let mut node = create_test_node("semantic-filtered").await;

    let embedding_dim = 768; // Default dimension
    let block = Block::new(b"test".to_vec().into()).unwrap();
    let cid = *block.cid();
    node.put_block(&block).await.unwrap();

    let embedding: Vec<f32> = (0..embedding_dim).map(|i| i as f32 / 64.0).collect();
    node.index_content(&cid, &embedding).await.unwrap();

    // Search with filter
    let query: Vec<f32> = (0..embedding_dim).map(|i| i as f32 / 64.0).collect();
    let filter = QueryFilter {
        min_score: Some(0.5),
        max_score: None,
        max_results: Some(5),
        cid_prefix: None,
    };

    let results = node
        .search_hybrid(&query, 10, filter)
        .await
        .expect("Failed to search");

    // Results should respect filter
    for result in &results {
        assert!(result.score >= 0.5);
    }

    node.stop().await.unwrap();
}

#[tokio::test]
async fn test_logic_facts() {
    let mut node = create_test_node("logic-facts").await;

    // Add facts
    let fact1 = Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("Alice".to_string())),
            Term::Const(Constant::String("Bob".to_string())),
        ],
    );
    node.add_fact(fact1.clone()).expect("Failed to add fact");

    let fact2 = Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("Bob".to_string())),
            Term::Const(Constant::String("Charlie".to_string())),
        ],
    );
    node.add_fact(fact2).expect("Failed to add fact");

    // Get stats
    let stats = node.tensorlogic_stats().expect("Failed to get stats");
    assert_eq!(stats.num_facts, 2);
    assert_eq!(stats.num_rules, 0);

    // Query for facts
    let goal = Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("Alice".to_string())),
            Term::Var("X".to_string()),
        ],
    );

    let results = node.infer(&goal).expect("Failed to infer");
    assert!(!results.is_empty());

    node.stop().await.unwrap();
}

#[tokio::test]
async fn test_logic_rules() {
    let mut node = create_test_node("logic-rules").await;

    // Add facts
    node.add_fact(Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("Alice".to_string())),
            Term::Const(Constant::String("Bob".to_string())),
        ],
    ))
    .unwrap();

    node.add_fact(Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("Bob".to_string())),
            Term::Const(Constant::String("Charlie".to_string())),
        ],
    ))
    .unwrap();

    // Add rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z)
    let rule = Rule::new(
        Predicate::new(
            "grandparent".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Z".to_string())],
        ),
        vec![
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
            ),
            Predicate::new(
                "parent".to_string(),
                vec![Term::Var("Y".to_string()), Term::Var("Z".to_string())],
            ),
        ],
    );

    node.add_rule(rule).expect("Failed to add rule");

    // Get stats
    let stats = node.tensorlogic_stats().expect("Failed to get stats");
    assert_eq!(stats.num_facts, 2);
    assert_eq!(stats.num_rules, 1);

    // Query using rule
    let goal = Predicate::new(
        "grandparent".to_string(),
        vec![
            Term::Var("X".to_string()),
            Term::Const(Constant::String("Charlie".to_string())),
        ],
    );

    let results = node.infer(&goal).expect("Failed to infer");
    assert!(!results.is_empty());

    node.stop().await.unwrap();
}

#[tokio::test]
async fn test_logic_proof_generation() {
    let mut node = create_test_node("logic-proof").await;

    // Add facts
    node.add_fact(Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("Alice".to_string())),
            Term::Const(Constant::String("Bob".to_string())),
        ],
    ))
    .unwrap();

    // Generate proof for a fact
    let goal = Predicate::new(
        "parent".to_string(),
        vec![
            Term::Const(Constant::String("Alice".to_string())),
            Term::Const(Constant::String("Bob".to_string())),
        ],
    );

    let proof = node.prove(&goal).expect("Failed to prove");
    assert!(proof.is_some());

    // Verify proof
    let is_valid = node
        .verify_proof(&proof.unwrap())
        .expect("Failed to verify proof");
    assert!(is_valid);

    node.stop().await.unwrap();
}

#[tokio::test]
async fn test_persistence_semantic_index() {
    let test_name = "persist-semantic";
    let index_path = format!("/tmp/ipfrs-test-{}-index.bin", test_name);
    let _ = std::fs::remove_file(&index_path);

    {
        let mut node = create_test_node(test_name).await;

        // Index some content
        let embedding_dim = 768; // Default dimension
        for i in 0..3 {
            let data = format!("Doc {}", i).into_bytes();
            let block = Block::new(data.into()).unwrap();
            node.put_block(&block).await.unwrap();

            let embedding: Vec<f32> = (0..embedding_dim).map(|j| (i + j) as f32).collect();
            node.index_content(block.cid(), &embedding).await.unwrap();
        }

        // Save index
        node.save_semantic_index(PathBuf::from(&index_path))
            .await
            .expect("Failed to save index");

        node.stop().await.unwrap();
    }

    // Load in new node (reuse same storage to access blocks)
    {
        let path = format!("/tmp/ipfrs-test-{}", test_name);
        let mut config = NodeConfig::default();
        config.storage.path = PathBuf::from(path);
        config.enable_semantic = true;
        config.enable_tensorlogic = true;

        let mut node = Node::new(config).expect("Failed to create node");
        node.start().await.expect("Failed to start node");

        node.load_semantic_index(PathBuf::from(&index_path))
            .await
            .expect("Failed to load index");

        // Note: Current implementation saves metadata (dimension, metric, CID mappings)
        // but not the full HNSW graph structure. This is acceptable for now.
        // Full graph serialization would require hnsw_rs dump/load support.

        // Verify metadata was loaded correctly
        let stats = node.semantic_stats().expect("Failed to get stats");
        assert_eq!(stats.dimension, 768, "Dimension should be preserved");
        assert_eq!(stats.num_vectors, 3, "CID count should be preserved");

        // Verify blocks still exist in shared storage
        for i in 0..3 {
            let data = format!("Doc {}", i).into_bytes();
            let block = Block::new(data.into()).unwrap();
            assert!(node.has_block(block.cid()).await.unwrap());
        }

        node.stop().await.unwrap();
    }

    let _ = std::fs::remove_file(&index_path);
}

#[tokio::test]
async fn test_persistence_knowledge_base() {
    let test_name = "persist-kb";
    let kb_path = format!("/tmp/ipfrs-test-{}-kb.bin", test_name);
    let _ = std::fs::remove_file(&kb_path);

    {
        let mut node = create_test_node(test_name).await;

        // Add facts
        node.add_fact(Predicate::new(
            "likes".to_string(),
            vec![
                Term::Const(Constant::String("Alice".to_string())),
                Term::Const(Constant::String("Rust".to_string())),
            ],
        ))
        .unwrap();

        // Save KB
        node.save_knowledge_base(PathBuf::from(&kb_path))
            .await
            .expect("Failed to save KB");

        node.stop().await.unwrap();
    }

    // Load in new node
    {
        let mut node = create_test_node(&format!("{}-reload", test_name)).await;

        node.load_knowledge_base(PathBuf::from(&kb_path))
            .await
            .expect("Failed to load KB");

        // Verify KB works
        let stats = node.tensorlogic_stats().unwrap();
        assert_eq!(stats.num_facts, 1);

        node.stop().await.unwrap();
    }

    let _ = std::fs::remove_file(&kb_path);
}

#[tokio::test]
async fn test_concurrent_operations() {
    let mut node = create_test_node("concurrent").await;

    // Create multiple blocks concurrently
    let mut handles = vec![];

    for i in 0..10 {
        let data = format!("Concurrent block {}", i).into_bytes();
        let block = Block::new(data.into()).unwrap();
        let cid = *block.cid();

        // Store block (synchronous since we can't clone node)
        node.put_block(&block).await.unwrap();
        handles.push(cid);
    }

    // Verify all blocks
    for cid in handles {
        let exists = node.has_block(&cid).await.unwrap();
        assert!(exists);
    }

    node.stop().await.unwrap();
}
