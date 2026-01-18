//! Example demonstrating lazy loading of semantic and tensorlogic components
//!
//! This example shows how IPFRS now uses lazy initialization to improve
//! startup time and reduce memory usage. Components are only initialized
//! when first accessed.

use ipfrs::{Node, NodeConfig};
use ipfrs_tensorlogic::{Constant, Predicate, Term};

#[tokio::main]
async fn main() -> ipfrs::Result<()> {
    println!("=== IPFRS Lazy Loading Demo ===\n");

    // Create and start node - semantic and tensorlogic are NOT initialized yet
    let mut node = Node::new(NodeConfig::default())?;
    println!("1. Created node (components not yet initialized)");

    node.start().await?;
    println!("2. Started node (still no semantic/tensorlogic initialization)");

    // Check initialization status
    println!("\n--- Initial Status ---");
    println!("Semantic enabled: {}", node.is_semantic_enabled());
    println!("Semantic initialized: {}", node.is_semantic_initialized());
    println!("TensorLogic enabled: {}", node.is_tensorlogic_enabled());
    println!(
        "TensorLogic initialized: {}",
        node.is_tensorlogic_initialized()
    );

    // First semantic operation triggers initialization
    println!("\n3. Performing first semantic operation...");
    let data = b"Hello, lazy loading!";
    let cid = node.add_bytes(data.as_ref()).await?;
    let embedding = vec![0.5; 768]; // Example 768-dim embedding
    node.index_content(&cid, &embedding).await?;
    println!("   Semantic router was initialized on first use!");

    // Check initialization status again
    println!("\n--- After First Semantic Use ---");
    println!("Semantic initialized: {}", node.is_semantic_initialized());
    println!(
        "TensorLogic initialized: {}",
        node.is_tensorlogic_initialized()
    );

    // First logic operation triggers tensorlogic initialization
    println!("\n4. Performing first logic operation...");
    let alice = Term::Const(Constant::String("Alice".to_string()));
    let fact = Predicate::new("person".to_string(), vec![alice]);
    node.add_fact(fact)?;
    println!("   TensorLogic store was initialized on first use!");

    // Check final status
    println!("\n--- Final Status ---");
    println!("Semantic initialized: {}", node.is_semantic_initialized());
    println!(
        "TensorLogic initialized: {}",
        node.is_tensorlogic_initialized()
    );

    // Clean up first node
    node.stop().await?;

    // Demonstrate warmup for predictable latency
    println!("\n=== Warmup Demo ===");
    let mut node2 = Node::new(NodeConfig::default())?;
    node2.start().await?;
    println!("Created second node");

    println!("Pre-warming all components...");
    node2.warmup()?;
    println!("All components pre-initialized!");

    println!("\n--- After Warmup ---");
    println!("Semantic initialized: {}", node2.is_semantic_initialized());
    println!(
        "TensorLogic initialized: {}",
        node2.is_tensorlogic_initialized()
    );

    // Clean up
    node2.stop().await?;

    println!("\n=== Benefits of Lazy Loading ===");
    println!("✓ Faster startup time (only initialize what you use)");
    println!("✓ Lower memory usage (components not loaded until needed)");
    println!("✓ Flexible deployment (configure features, pay only for what you use)");
    println!("✓ Optional warmup for predictable latency in production");

    Ok(())
}
