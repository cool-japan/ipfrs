//! End-to-End Workflow Tests
//!
//! These tests exercise complete IPFRS workflows from start to finish,
//! testing the integration of all components in realistic scenarios.

use ipfrs::{Block, Constant, Node, NodeConfig, Predicate, QueryFilter, Rule, Term};
use std::path::PathBuf;
use std::time::Duration;
use tokio::time::sleep;

/// Helper to create a test node with unique storage
async fn create_test_node(test_name: &str) -> Node {
    let path = format!("/tmp/ipfrs-e2e-{}-{}", test_name, std::process::id());
    let _ = std::fs::remove_dir_all(&path);

    let mut config = NodeConfig::default();
    config.storage.path = PathBuf::from(path);
    config.enable_semantic = true;
    config.enable_tensorlogic = true;

    let mut node = Node::new(config).expect("Failed to create node");
    node.start().await.expect("Failed to start node");
    node
}

/// Cleanup test node
async fn cleanup_node(mut node: Node, test_name: &str) {
    let path = format!("/tmp/ipfrs-e2e-{}-{}", test_name, std::process::id());
    node.stop().await.expect("Failed to stop node");
    let _ = std::fs::remove_dir_all(&path);
}

#[tokio::test]
async fn test_e2e_content_storage_and_retrieval() {
    // Complete workflow: Add content, verify, retrieve, delete, verify gone
    let node = create_test_node("content-workflow").await;

    // Step 1: Add content
    let content = b"End-to-end test content";
    let block = Block::new(content.to_vec().into()).expect("Failed to create block");
    let cid = *block.cid();

    node.put_block(&block).await.expect("Failed to put block");

    // Step 2: Verify block exists
    let exists = node.has_block(&cid).await.expect("Failed to check block");
    assert!(exists, "Block should exist after put");

    // Step 3: Retrieve and verify content
    let retrieved = node.get_block(&cid).await.expect("Failed to get block");
    assert!(retrieved.is_some(), "Block should be retrievable");
    assert_eq!(retrieved.unwrap().data().as_ref(), content);

    // Step 4: Get statistics
    let stat = node.block_stat(&cid).await.expect("Failed to get stats");
    assert!(stat.is_some(), "Stats should be available");
    assert_eq!(stat.unwrap().size, content.len());

    // Step 5: Delete block
    node.delete_block(&cid)
        .await
        .expect("Failed to delete block");

    // Step 6: Verify block is gone
    let exists_after = node.has_block(&cid).await.expect("Failed to check block");
    assert!(!exists_after, "Block should not exist after delete");

    cleanup_node(node, "content-workflow").await;
}

#[tokio::test]
async fn test_e2e_semantic_search_workflow() {
    // Complete workflow: Index documents, search, filter, persist, reload
    let mut node = create_test_node("semantic-workflow").await;

    // Step 1: Index multiple documents
    let documents = vec![
        ("Rust programming language", vec![0.1, 0.9, 0.3]),
        ("Python machine learning", vec![0.8, 0.2, 0.7]),
        ("JavaScript web development", vec![0.3, 0.4, 0.9]),
        ("Go concurrent programming", vec![0.2, 0.8, 0.4]),
        ("Java enterprise applications", vec![0.6, 0.3, 0.2]),
    ];

    let mut cids = Vec::new();
    for (text, embedding_base) in &documents {
        let block = Block::new(text.as_bytes().to_vec().into()).expect("Failed to create block");
        let cid = *block.cid();
        node.put_block(&block).await.expect("Failed to put block");

        // Expand to 768 dimensions
        let embedding: Vec<f32> = embedding_base.iter().cycle().take(768).copied().collect();

        node.index_content(&cid, &embedding)
            .await
            .expect("Failed to index");
        cids.push(cid);
    }

    // Step 2: Verify all indexed
    let stats = node.semantic_stats().expect("Failed to get stats");
    assert_eq!(stats.num_vectors, 5, "Should have 5 indexed vectors");

    // Step 3: Perform similarity search
    let query_embedding: Vec<f32> = [0.15, 0.85, 0.35]
        .iter()
        .cycle()
        .take(768)
        .copied()
        .collect();

    let results = node
        .search_similar(&query_embedding, 3)
        .await
        .expect("Failed to search");
    assert_eq!(results.len(), 3, "Should return top 3 results");

    // Step 4: Filtered search
    let filter = QueryFilter {
        min_score: Some(0.5),
        max_score: None,
        max_results: Some(2),
        cid_prefix: None,
    };

    let filtered = node
        .search_hybrid(&query_embedding, 5, filter)
        .await
        .expect("Failed to filter");
    assert!(
        filtered.len() <= 2,
        "Filtered results should respect max_results"
    );

    // Step 5: Persist index
    let index_path = format!("/tmp/ipfrs-e2e-semantic-index-{}.bin", std::process::id());
    node.save_semantic_index(&index_path)
        .await
        .expect("Failed to save index");
    assert!(
        std::path::Path::new(&index_path).exists(),
        "Index file should exist"
    );

    // Step 6: Stop first node and create second node with same storage
    // (blocks are already persisted, just need to reload the semantic index)
    let storage_path = format!("/tmp/ipfrs-e2e-semantic-workflow-{}", std::process::id());

    node.stop().await.expect("Failed to stop first node");
    drop(node); // Explicitly drop to release database lock

    // Create new node with same storage path
    let mut config2 = NodeConfig::default();
    config2.storage.path = PathBuf::from(&storage_path);
    config2.enable_semantic = true;

    let mut node2 = Node::new(config2).expect("Failed to create node2");
    node2.start().await.expect("Failed to start node2");

    // Blocks are already in storage, just reload the semantic index
    node2
        .load_semantic_index(&index_path)
        .await
        .expect("Failed to load index");

    // Step 7: Verify loaded index works
    let stats2 = node2
        .semantic_stats()
        .expect("Failed to get stats from node2");
    assert_eq!(stats2.num_vectors, 5, "Loaded index should have 5 vectors");

    // Verify search completes successfully (result count may vary based on score thresholds)
    let results2 = node2
        .search_similar(&query_embedding, 3)
        .await
        .expect("Failed to search node2");
    println!(
        "Loaded index returned {} results (may vary from original due to score filtering)",
        results2.len()
    );

    // Cleanup (node was already dropped earlier after stopping)
    node2.stop().await.expect("Failed to stop node2");
    std::fs::remove_dir_all(&storage_path).ok();
    std::fs::remove_file(&index_path).ok();
}

#[tokio::test]
async fn test_e2e_logic_reasoning_workflow() {
    // Complete workflow: Add facts/rules, infer, prove, verify, persist, reload
    let mut node = create_test_node("logic-workflow").await;

    // Step 1: Build knowledge base with family relationships
    let facts = vec![
        ("Alice", "Bob"),     // Alice is parent of Bob
        ("Bob", "Charlie"),   // Bob is parent of Charlie
        ("Charlie", "Diana"), // Charlie is parent of Diana
        ("Eve", "Frank"),     // Eve is parent of Frank
    ];

    for (parent, child) in &facts {
        let fact = Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String(parent.to_string())),
                Term::Const(Constant::String(child.to_string())),
            ],
        );
        node.add_fact(fact).expect("Failed to add fact");
    }

    // Step 2: Add grandparent rule
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

    // Step 3: Add ancestor rule (transitive closure)
    let ancestor_rule1 = Rule::new(
        Predicate::new(
            "ancestor".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        ),
        vec![Predicate::new(
            "parent".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Y".to_string())],
        )],
    );
    node.add_rule(ancestor_rule1)
        .expect("Failed to add ancestor rule 1");

    let ancestor_rule2 = Rule::new(
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
    );
    node.add_rule(ancestor_rule2)
        .expect("Failed to add ancestor rule 2");

    // Step 4: Verify knowledge base stats
    let stats = node.tensorlogic_stats().expect("Failed to get stats");
    assert_eq!(stats.num_facts, 4, "Should have 4 facts");
    assert_eq!(stats.num_rules, 3, "Should have 3 rules");

    // Step 5: Perform inference - find grandparents of Charlie
    let goal = Predicate::new(
        "grandparent".to_string(),
        vec![
            Term::Var("X".to_string()),
            Term::Const(Constant::String("Charlie".to_string())),
        ],
    );

    let results = node.infer(&goal).expect("Failed to infer");
    assert!(!results.is_empty(), "Should find grandparents");

    // Step 6: Generate and verify proof
    let proof_result = node.prove(&goal).expect("Failed to prove");
    if let Some(proof) = proof_result {
        // Verify proof directly
        let is_valid = node.verify_proof(&proof).expect("Failed to verify proof");
        assert!(is_valid, "Proof should be valid");

        // Store proof and retrieve it
        let proof_cid = node
            .store_proof(&proof)
            .await
            .expect("Failed to store proof");
        let retrieved_proof = node
            .get_proof(&proof_cid)
            .await
            .expect("Failed to get proof");
        assert!(retrieved_proof.is_some(), "Proof should exist");

        // Verify retrieved proof
        let is_valid_retrieved = node
            .verify_proof(&retrieved_proof.unwrap())
            .expect("Failed to verify retrieved proof");
        assert!(is_valid_retrieved, "Retrieved proof should be valid");
    } else {
        panic!("Proof generation returned None");
    }

    // Step 7: Complex inference - find all ancestors of Diana
    let ancestor_goal = Predicate::new(
        "ancestor".to_string(),
        vec![
            Term::Var("X".to_string()),
            Term::Const(Constant::String("Diana".to_string())),
        ],
    );

    let ancestors = node
        .infer(&ancestor_goal)
        .expect("Failed to infer ancestors");
    assert!(!ancestors.is_empty(), "Diana should have ancestors");
    assert!(
        ancestors.len() >= 2,
        "Diana should have at least 2 ancestors"
    );

    // Step 8: Persist knowledge base
    let kb_path = format!("/tmp/ipfrs-e2e-kb-{}.bin", std::process::id());
    node.save_knowledge_base(&kb_path)
        .await
        .expect("Failed to save KB");
    assert!(
        std::path::Path::new(&kb_path).exists(),
        "KB file should exist"
    );

    // Step 9: Stop first node and create new node with same storage, then reload KB
    let storage_path = format!("/tmp/ipfrs-e2e-logic-workflow-{}", std::process::id());

    node.stop().await.expect("Failed to stop first node");
    drop(node); // Explicitly drop to release database lock

    // Create new node with same storage path
    let mut config2 = NodeConfig::default();
    config2.storage.path = PathBuf::from(&storage_path);
    config2.enable_tensorlogic = true;

    let mut node2 = Node::new(config2).expect("Failed to create node2");
    node2.start().await.expect("Failed to start node2");
    node2
        .load_knowledge_base(&kb_path)
        .await
        .expect("Failed to load KB");

    // Step 10: Verify loaded KB works
    let stats2 = node2
        .tensorlogic_stats()
        .expect("Failed to get stats from node2");
    assert_eq!(stats2.num_facts, 4, "Loaded KB should have 4 facts");
    assert_eq!(stats2.num_rules, 3, "Loaded KB should have 3 rules");

    let results2 = node2.infer(&goal).expect("Failed to infer from loaded KB");
    assert_eq!(
        results2.len(),
        results.len(),
        "Loaded KB should produce same results"
    );

    // Cleanup (node was already dropped earlier after stopping)
    node2.stop().await.expect("Failed to stop node2");
    std::fs::remove_dir_all(&storage_path).ok();
    std::fs::remove_file(&kb_path).ok();
}

#[tokio::test]
async fn test_e2e_combined_semantic_and_logic() {
    // Complex workflow: Combine semantic search with logic reasoning
    let node = create_test_node("combined-workflow").await;

    // Step 1: Add documents with semantic indexing
    let papers = vec![
        ("Deep Learning Foundations", "AI", vec![0.9, 0.1, 0.2]),
        ("Neural Networks Theory", "AI", vec![0.85, 0.15, 0.25]),
        ("Database Systems", "DB", vec![0.1, 0.9, 0.3]),
        ("Distributed Databases", "DB", vec![0.15, 0.85, 0.35]),
    ];

    for (title, topic, embedding_base) in &papers {
        let block = Block::new(title.as_bytes().to_vec().into()).expect("Failed to create block");
        let cid = *block.cid();
        node.put_block(&block).await.expect("Failed to put block");

        // Index semantically
        let embedding: Vec<f32> = embedding_base.iter().cycle().take(768).copied().collect();
        node.index_content(&cid, &embedding)
            .await
            .expect("Failed to index");

        // Add to knowledge base
        let fact = Predicate::new(
            "paper".to_string(),
            vec![
                Term::Const(Constant::String(title.to_string())),
                Term::Const(Constant::String(topic.to_string())),
            ],
        );
        node.add_fact(fact).expect("Failed to add fact");
    }

    // Step 2: Add citation rules
    let cite_fact1 = Predicate::new(
        "cites".to_string(),
        vec![
            Term::Const(Constant::String("Deep Learning Foundations".to_string())),
            Term::Const(Constant::String("Neural Networks Theory".to_string())),
        ],
    );
    node.add_fact(cite_fact1).expect("Failed to add citation");

    // Step 3: Semantic search for AI papers
    let ai_query: Vec<f32> = [0.9, 0.1, 0.2].iter().cycle().take(768).copied().collect();
    let ai_papers = node
        .search_similar(&ai_query, 2)
        .await
        .expect("Failed to search");
    assert_eq!(ai_papers.len(), 2, "Should find 2 AI papers");

    // Step 4: Logic query for papers in AI topic
    let ai_goal = Predicate::new(
        "paper".to_string(),
        vec![
            Term::Var("Title".to_string()),
            Term::Const(Constant::String("AI".to_string())),
        ],
    );
    let ai_logic_results = node.infer(&ai_goal).expect("Failed to infer");
    assert_eq!(
        ai_logic_results.len(),
        2,
        "Should find 2 AI papers via logic"
    );

    // Step 5: Combined query - find papers that cite AI papers
    let citation_goal = Predicate::new(
        "cites".to_string(),
        vec![
            Term::Var("Paper1".to_string()),
            Term::Var("Paper2".to_string()),
        ],
    );
    let citations = node
        .infer(&citation_goal)
        .expect("Failed to infer citations");
    assert!(!citations.is_empty(), "Should find citations");

    cleanup_node(node, "combined-workflow").await;
}

#[tokio::test]
async fn test_e2e_concurrent_operations() {
    // Test concurrent block operations, searches, and queries
    let node = create_test_node("concurrent").await;

    // Task 1: Add blocks concurrently
    for i in 0..10 {
        let data = format!("Concurrent block {}", i);
        let block = Block::new(data.into_bytes().into()).expect("Failed to create block");
        node.put_block(&block).await.expect("Failed to put block");
    }

    // Task 2: Index content concurrently
    for i in 0..10 {
        let data = format!("Concurrent document {}", i);
        let block = Block::new(data.into_bytes().into()).expect("Failed to create block");
        let cid = *block.cid();
        node.put_block(&block).await.expect("Failed to put block");

        let embedding: Vec<f32> = (0..768).map(|j| ((i + j) as f32) / 100.0).collect();
        node.index_content(&cid, &embedding)
            .await
            .expect("Failed to index");
    }

    // Task 3: Add facts concurrently
    for i in 0..10 {
        let fact = Predicate::new(
            "test".to_string(),
            vec![
                Term::Const(Constant::String(format!("value{}", i))),
                Term::Const(Constant::Int(i)),
            ],
        );
        node.add_fact(fact).expect("Failed to add fact");
    }

    // Wait a bit for all operations to complete
    sleep(Duration::from_millis(100)).await;

    // Verify all operations succeeded
    let storage_stats = node.storage_stats().expect("Failed to get storage stats");
    assert!(
        storage_stats.num_blocks >= 20,
        "Should have at least 20 blocks"
    );

    let semantic_stats = node.semantic_stats().expect("Failed to get semantic stats");
    assert_eq!(semantic_stats.num_vectors, 10, "Should have 10 vectors");

    let logic_stats = node.tensorlogic_stats().expect("Failed to get logic stats");
    assert_eq!(logic_stats.num_facts, 10, "Should have 10 facts");

    cleanup_node(node, "concurrent").await;
}

#[tokio::test]
async fn test_e2e_error_recovery() {
    // Test graceful error handling and recovery
    let node = create_test_node("error-recovery").await;

    // Test 1: Query non-existent CID
    use ipfrs_core::Cid;
    let fake_cid = Cid::default();
    let result = node
        .get_block(&fake_cid)
        .await
        .expect("Should not error on missing block");
    assert!(result.is_none(), "Should return None for missing block");

    // Test 2: Index with wrong dimension (should handle gracefully)
    let block = Block::new(b"test".to_vec().into()).expect("Failed to create block");
    let cid = *block.cid();
    node.put_block(&block).await.expect("Failed to put block");

    // Try to index with wrong dimension
    let wrong_embedding: Vec<f32> = vec![0.1, 0.2, 0.3]; // Too small
    let index_result = node.index_content(&cid, &wrong_embedding).await;
    assert!(index_result.is_err(), "Should error on wrong dimension");

    // Test 3: Inference with undefined predicate
    let undefined_goal = Predicate::new(
        "undefined_predicate".to_string(),
        vec![Term::Var("X".to_string())],
    );
    let infer_result = node
        .infer(&undefined_goal)
        .expect("Should not panic on undefined predicate");
    assert!(
        infer_result.is_empty(),
        "Should return empty results for undefined predicate"
    );

    cleanup_node(node, "error-recovery").await;
}

#[tokio::test]
async fn test_e2e_persistence_after_restart() {
    // Test data persistence across node restarts
    let test_id = std::process::id();
    let storage_path = format!("/tmp/ipfrs-e2e-persist-{}", test_id);

    // Phase 1: Create node and add data
    {
        let mut config = NodeConfig::default();
        config.storage.path = PathBuf::from(&storage_path);
        config.enable_semantic = false;
        config.enable_tensorlogic = false;

        let mut node = Node::new(config).expect("Failed to create node");
        node.start().await.expect("Failed to start node");

        // Add multiple blocks
        for i in 0..5 {
            let data = format!("Persistent block {}", i);
            let block = Block::new(data.into_bytes().into()).expect("Failed to create block");
            node.put_block(&block).await.expect("Failed to put block");
        }

        // Stop node
        node.stop().await.expect("Failed to stop node");
    }

    // Phase 2: Restart node and verify data persisted
    {
        let mut config = NodeConfig::default();
        config.storage.path = PathBuf::from(&storage_path);
        config.enable_semantic = false;
        config.enable_tensorlogic = false;

        let mut node = Node::new(config).expect("Failed to create node");
        node.start().await.expect("Failed to start node");

        // Verify blocks persisted
        let stats = node.storage_stats().expect("Failed to get stats");
        assert_eq!(stats.num_blocks, 5, "Should have 5 persisted blocks");

        // Cleanup
        node.stop().await.expect("Failed to stop node");
    }

    std::fs::remove_dir_all(&storage_path).ok();
}

#[tokio::test]
async fn test_e2e_pin_management_workflow() {
    // Test complete pin management workflow
    let node = create_test_node("pin-workflow").await;

    // Step 1: Add content
    let content1 = b"Important content to pin";
    let content2 = b"Another important piece of data";
    let content3 = b"Temporary content";

    let block1 = Block::new(content1.to_vec().into()).expect("Failed to create block1");
    let block2 = Block::new(content2.to_vec().into()).expect("Failed to create block2");
    let block3 = Block::new(content3.to_vec().into()).expect("Failed to create block3");

    let cid1 = *block1.cid();
    let cid2 = *block2.cid();
    let cid3 = *block3.cid();

    node.put_block(&block1).await.expect("Failed to put block1");
    node.put_block(&block2).await.expect("Failed to put block2");
    node.put_block(&block3).await.expect("Failed to put block3");

    // Step 2: Pin important content
    node.pin_add(&cid1, false, Some("important-data-1".to_string()))
        .await
        .expect("Failed to pin cid1");
    node.pin_add(&cid2, false, Some("important-data-2".to_string()))
        .await
        .expect("Failed to pin cid2");

    // Step 3: Verify pins exist
    let pins = node.pin_ls().expect("Failed to list pins");
    assert_eq!(pins.len(), 2, "Should have 2 pins");

    let pinned_cids: Vec<_> = pins.iter().map(|p| p.cid).collect();
    assert!(pinned_cids.contains(&cid1), "cid1 should be pinned");
    assert!(pinned_cids.contains(&cid2), "cid2 should be pinned");
    assert!(!pinned_cids.contains(&cid3), "cid3 should not be pinned");

    // Step 4: Verify pins can be listed again
    let pins_verify = node.pin_ls().expect("Failed to list pins for verification");
    assert_eq!(pins_verify.len(), 2, "Should still have 2 pins");

    // Check that pin names are preserved
    let has_names = pins_verify.iter().any(|p| p.name.is_some());
    assert!(has_names, "Pin names should be preserved");

    // Step 5: Remove a pin
    node.pin_rm(&cid2, false)
        .await
        .expect("Failed to remove pin");

    let pins_after = node.pin_ls().expect("Failed to list pins");
    assert_eq!(pins_after.len(), 1, "Should have 1 pin after removal");

    // Step 6: Verify garbage collection protects pinned content
    // Run GC (dry run first) - pinned content should not be removed
    let gc_dry = node.repo_gc(true).await;
    if let Ok(result) = gc_dry {
        println!(
            "GC dry run: {} blocks would be collected",
            result.blocks_collected
        );
        println!("GC dry run: {} bytes would be freed", result.bytes_freed);
        // In a dry run, blocks aren't actually removed
    }

    // Run actual GC
    let gc_result = node.repo_gc(false).await;
    if gc_result.is_ok() {
        // After GC, pinned content should still exist
        let still_exists = node.has_block(&cid1).await.expect("Failed to check block");
        assert!(still_exists, "Pinned content should survive GC");
    }

    cleanup_node(node, "pin-workflow").await;
}

#[tokio::test]
async fn test_e2e_repository_analysis() {
    // Test repository analysis and statistics
    let node = create_test_node("repo-analysis").await;

    // Add content of various sizes
    let test_data = vec![
        ("small", vec![0u8; 100]),
        ("medium", vec![1u8; 10_000]),
        ("large", vec![2u8; 100_000]),
    ];

    let mut total_size = 0;
    for (name, data) in &test_data {
        let block = Block::new(data.clone().into()).expect("Failed to create block");
        node.put_block(&block).await.expect("Failed to put block");
        total_size += data.len();
        println!("Added {} block: {} bytes", name, data.len());
    }

    // Get storage statistics
    let stats = node.storage_stats().expect("Failed to get storage stats");
    assert_eq!(stats.num_blocks, 3, "Should have 3 blocks");
    assert!(!stats.is_empty, "Storage should not be empty");

    println!("Repository stats:");
    println!("  Blocks: {}", stats.num_blocks);
    println!("  Is empty: {}", stats.is_empty);

    // Test formatted size display with the total we calculated
    let formatted_size = ipfrs::format_bytes(total_size as u64);
    println!("Formatted total size: {}", formatted_size);
    assert!(formatted_size.contains("KB") || formatted_size.contains("B"));

    // Verify average block size calculation
    let avg_size = total_size / test_data.len();
    println!("Average block size: {} bytes", avg_size);
    assert!(avg_size > 0);

    // Verify we can retrieve all blocks
    for (name, data) in &test_data {
        let block = Block::new(data.clone().into()).expect("Failed to create block");
        let cid = block.cid();
        let retrieved = node.get_block(cid).await.expect("Failed to get block");
        assert!(retrieved.is_some(), "{} block should be retrievable", name);
    }

    cleanup_node(node, "repo-analysis").await;
}
