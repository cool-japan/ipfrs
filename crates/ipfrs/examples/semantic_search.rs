//! Semantic Search Example
//!
//! This example demonstrates IPFRS semantic search capabilities:
//! - Configuring semantic router
//! - Indexing content with embeddings
//! - Performing similarity searches
//! - Using query filters

use ipfrs::{Node, NodeConfig};
use ipfrs_semantic::QueryFilter;

#[tokio::main]
async fn main() -> ipfrs::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== IPFRS Semantic Search Example ===\n");

    // Configure node with semantic search enabled
    let config = NodeConfig {
        enable_semantic: true,
        ..NodeConfig::default()
    };

    let mut node = Node::new(config)?;
    node.start().await?;
    println!("✓ Node started with semantic search enabled");

    // Verify semantic is enabled
    if !node.is_semantic_enabled() {
        eprintln!("ERROR: Semantic search not enabled!");
        return Ok(());
    }

    // Example 1: Index documents with embeddings
    println!("\n--- Example 1: Indexing Documents ---");

    // Simulated documents and their embeddings
    // In practice, use a real embedding model like sentence-transformers
    let documents = vec![
        (
            "Rust is a systems programming language.",
            generate_embedding("rust programming"),
        ),
        (
            "IPFS is a distributed file system.",
            generate_embedding("distributed storage"),
        ),
        (
            "Machine learning powers AI applications.",
            generate_embedding("machine learning ai"),
        ),
        (
            "Blockchain enables decentralized trust.",
            generate_embedding("blockchain decentralized"),
        ),
        (
            "Neural networks learn from data.",
            generate_embedding("neural network learning"),
        ),
    ];

    let mut cids = Vec::new();
    for (text, embedding) in &documents {
        let cid = node.add_bytes(text.as_bytes()).await?;
        node.index_content(&cid, embedding).await?;
        println!("Indexed: \"{}\" → {}", text, cid);
        cids.push(cid);
    }

    // Example 2: Semantic similarity search
    println!("\n--- Example 2: Similarity Search ---");

    let query = "artificial intelligence and neural networks";
    let query_embedding = generate_embedding(query);

    println!("Query: \"{}\"", query);
    let results = node.search_similar(&query_embedding, 3).await?;

    println!("\nTop 3 similar documents:");
    for (i, result) in results.iter().enumerate() {
        // Find the original text
        if let Some(data) = node.get(&result.cid).await? {
            let text = String::from_utf8_lossy(&data);
            println!("  {}. [score: {:.4}] {}", i + 1, result.score, text);
        }
    }

    // Example 3: Filtered search
    println!("\n--- Example 3: Filtered Search ---");

    let filter = QueryFilter {
        min_score: Some(0.7), // Only results with score >= 0.7
        max_score: None,      // No upper limit
        max_results: Some(2), // Limit to 2 results
        cid_prefix: None,
    };

    let filtered_results = node.search_hybrid(&query_embedding, 5, filter).await?;

    println!("Filtered results (min_score=0.7, max=2):");
    for (i, result) in filtered_results.iter().enumerate() {
        if let Some(data) = node.get(&result.cid).await? {
            let text = String::from_utf8_lossy(&data);
            println!("  {}. [score: {:.4}] {}", i + 1, result.score, text);
        }
    }

    // Example 4: Semantic statistics
    println!("\n--- Example 4: Semantic Statistics ---");

    let stats = node.semantic_stats()?;
    println!("Semantic Index Statistics:");
    println!("  Indexed vectors: {}", stats.num_vectors);
    println!("  Vector dimension: {}", stats.dimension);
    println!("  Distance metric: {:?}", stats.metric);
    println!("  Cache size: {}", stats.cache_size);
    println!("  Cache capacity: {}", stats.cache_capacity);

    // Clean shutdown
    println!("\n--- Shutting Down ---");
    node.stop().await?;
    println!("✓ Node stopped");

    println!("\n=== Example Complete ===");
    Ok(())
}

/// Generate a simulated embedding vector
///
/// In a real application, use a proper embedding model like:
/// - sentence-transformers (Python)
/// - rust-bert
/// - OpenAI Embeddings API
/// - Cohere Embeddings API
fn generate_embedding(text: &str) -> Vec<f32> {
    // This is a VERY simple hash-based pseudo-embedding for demonstration
    // DO NOT use in production! Use proper embedding models.

    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    let hash = hasher.finish();

    // Generate 384-dimensional vector from hash (terrible but deterministic)
    let mut embedding = Vec::with_capacity(384);
    for i in 0..384 {
        let val = ((hash.wrapping_add(i as u64)) % 1000) as f32 / 1000.0;
        embedding.push(val);
    }

    // Normalize to unit vector
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in &mut embedding {
            *val /= norm;
        }
    }

    embedding
}
