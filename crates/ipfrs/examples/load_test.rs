//! Load Testing Tool for IPFRS
//!
//! This tool performs comprehensive load testing on IPFRS to validate
//! performance under various scenarios and load patterns.
//!
//! Usage:
//!   cargo run --package ipfrs --example load_test --release
//!
//! Test Scenarios:
//! - Block operations (put, get, has)
//! - Semantic search at scale
//! - Logic inference with complex KB
//! - Concurrent operations
//! - Mixed workload patterns

use ipfrs::{Block, Constant, Node, NodeConfig, Predicate, Rule, Term};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Load test configuration
#[derive(Debug, Clone)]
struct LoadTestConfig {
    /// Number of blocks to create
    num_blocks: usize,
    /// Number of semantic vectors to index
    num_vectors: usize,
    /// Number of facts to add to KB
    num_facts: usize,
    /// Number of concurrent workers
    num_workers: usize,
    /// Vector dimension for semantic search
    vector_dim: usize,
}

impl Default for LoadTestConfig {
    fn default() -> Self {
        Self {
            num_blocks: 1000,
            num_vectors: 500,
            num_facts: 200,
            num_workers: 10,
            vector_dim: 768,
        }
    }
}

/// Test results and metrics
#[derive(Debug)]
struct TestMetrics {
    name: String,
    total_operations: usize,
    duration: Duration,
    ops_per_sec: f64,
    avg_latency_ms: f64,
    min_latency_ms: f64,
    max_latency_ms: f64,
}

impl TestMetrics {
    fn new(name: String, total_ops: usize, duration: Duration, latencies: &[Duration]) -> Self {
        let duration_secs = duration.as_secs_f64();
        let ops_per_sec = total_ops as f64 / duration_secs;

        let latency_ms: Vec<f64> = latencies.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
        let avg_latency_ms = latency_ms.iter().sum::<f64>() / latency_ms.len() as f64;
        let min_latency_ms = latency_ms.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_latency_ms = latency_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Self {
            name,
            total_operations: total_ops,
            duration,
            ops_per_sec,
            avg_latency_ms,
            min_latency_ms,
            max_latency_ms,
        }
    }

    fn print(&self) {
        println!("\n=== {} ===", self.name);
        println!("Total operations: {}", self.total_operations);
        println!("Duration: {:.2}s", self.duration.as_secs_f64());
        println!("Throughput: {:.2} ops/sec", self.ops_per_sec);
        println!("Avg latency: {:.2}ms", self.avg_latency_ms);
        println!("Min latency: {:.2}ms", self.min_latency_ms);
        println!("Max latency: {:.2}ms", self.max_latency_ms);
    }
}

#[tokio::main]
async fn main() -> ipfrs::Result<()> {
    println!("🚀 IPFRS Load Testing Tool\n");
    println!("This will test IPFRS performance under various load scenarios.");
    println!("Tests run in RELEASE mode for accurate performance measurements.\n");

    let config = LoadTestConfig::default();
    println!("Configuration:");
    println!("  Blocks: {}", config.num_blocks);
    println!("  Vectors: {}", config.num_vectors);
    println!("  Facts: {}", config.num_facts);
    println!("  Workers: {}", config.num_workers);
    println!("  Vector dimension: {}", config.vector_dim);
    println!();

    // Setup test node
    let storage_path = format!("/tmp/ipfrs-load-test-{}", std::process::id());
    let _ = std::fs::remove_dir_all(&storage_path);

    let mut node_config = NodeConfig::default();
    node_config.storage.path = PathBuf::from(&storage_path);
    node_config.enable_semantic = true;
    node_config.enable_tensorlogic = true;

    let mut node = Node::new(node_config)?;
    node.start().await?;
    println!("✓ Node initialized\n");

    // Run test scenarios
    let mut all_metrics = Vec::new();

    // Test 1: Block write performance
    let metrics = test_block_writes(&node, &config).await?;
    metrics.print();
    all_metrics.push(metrics);

    // Test 2: Block read performance
    let metrics = test_block_reads(&node, &config).await?;
    metrics.print();
    all_metrics.push(metrics);

    // Test 3: Semantic indexing performance
    let metrics = test_semantic_indexing(&node, &config).await?;
    metrics.print();
    all_metrics.push(metrics);

    // Test 4: Semantic search performance
    let metrics = test_semantic_search(&node, &config).await?;
    metrics.print();
    all_metrics.push(metrics);

    // Test 5: Logic fact insertion performance
    let metrics = test_logic_facts(&node, &config).await?;
    metrics.print();
    all_metrics.push(metrics);

    // Test 6: Logic inference performance
    let metrics = test_logic_inference(&node, &config).await?;
    metrics.print();
    all_metrics.push(metrics);

    // Test 7: Concurrent mixed workload
    let metrics = test_concurrent_mixed(&node, &config).await?;
    metrics.print();
    all_metrics.push(metrics);

    // Test 8: Persistence performance
    let metrics = test_persistence(&mut node, &config).await?;
    metrics.print();
    all_metrics.push(metrics);

    // Summary
    print_summary(&all_metrics);

    // Cleanup
    node.stop().await?;
    std::fs::remove_dir_all(&storage_path).ok();

    println!("\n✅ Load testing complete!");
    Ok(())
}

/// Test block write throughput
async fn test_block_writes(node: &Node, config: &LoadTestConfig) -> ipfrs::Result<TestMetrics> {
    let mut latencies = Vec::new();
    let start = Instant::now();

    for i in 0..config.num_blocks {
        let data = format!("Block data {}", i);
        let block = Block::new(data.into_bytes().into())?;

        let op_start = Instant::now();
        node.put_block(&block).await?;
        latencies.push(op_start.elapsed());

        if (i + 1) % 100 == 0 {
            print!("\rWrote {} / {} blocks", i + 1, config.num_blocks);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }

    println!();
    let duration = start.elapsed();
    Ok(TestMetrics::new(
        "Block Writes".to_string(),
        config.num_blocks,
        duration,
        &latencies,
    ))
}

/// Test block read throughput
async fn test_block_reads(node: &Node, config: &LoadTestConfig) -> ipfrs::Result<TestMetrics> {
    // First, collect CIDs of existing blocks
    let stats = node.storage_stats()?;
    let num_reads = config.num_blocks.min(stats.num_blocks);

    // Generate CIDs to read (we'll read the blocks we just wrote)
    let mut cids = Vec::new();
    for i in 0..num_reads {
        let data = format!("Block data {}", i);
        let block = Block::new(data.into_bytes().into())?;
        cids.push(*block.cid());
    }

    let mut latencies = Vec::new();
    let start = Instant::now();

    for (i, cid) in cids.iter().enumerate() {
        let op_start = Instant::now();
        let _ = node.get_block(cid).await?;
        latencies.push(op_start.elapsed());

        if (i + 1) % 100 == 0 {
            print!("\rRead {} / {} blocks", i + 1, num_reads);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }

    println!();
    let duration = start.elapsed();
    Ok(TestMetrics::new(
        "Block Reads".to_string(),
        num_reads,
        duration,
        &latencies,
    ))
}

/// Test semantic indexing throughput
async fn test_semantic_indexing(
    node: &Node,
    config: &LoadTestConfig,
) -> ipfrs::Result<TestMetrics> {
    let mut latencies = Vec::new();
    let start = Instant::now();

    for i in 0..config.num_vectors {
        // Create block
        let data = format!("Document {}", i);
        let block = Block::new(data.into_bytes().into())?;
        let cid = *block.cid();
        node.put_block(&block).await?;

        // Generate embedding (deterministic for testing)
        let embedding: Vec<f32> = (0..config.vector_dim)
            .map(|j| ((i + j) as f32 * 0.01) % 1.0)
            .collect();

        // Index
        let op_start = Instant::now();
        node.index_content(&cid, &embedding).await?;
        latencies.push(op_start.elapsed());

        if (i + 1) % 50 == 0 {
            print!("\rIndexed {} / {} vectors", i + 1, config.num_vectors);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }

    println!();
    let duration = start.elapsed();
    Ok(TestMetrics::new(
        "Semantic Indexing".to_string(),
        config.num_vectors,
        duration,
        &latencies,
    ))
}

/// Test semantic search throughput
async fn test_semantic_search(node: &Node, config: &LoadTestConfig) -> ipfrs::Result<TestMetrics> {
    let num_searches = 100;
    let mut latencies = Vec::new();
    let start = Instant::now();

    for i in 0..num_searches {
        // Generate query embedding
        let query: Vec<f32> = (0..config.vector_dim)
            .map(|j| ((i * 2 + j) as f32 * 0.01) % 1.0)
            .collect();

        let op_start = Instant::now();
        let _ = node.search_similar(&query, 10).await?;
        latencies.push(op_start.elapsed());

        if (i + 1) % 10 == 0 {
            print!("\rSearched {} / {} queries", i + 1, num_searches);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }

    println!();
    let duration = start.elapsed();
    Ok(TestMetrics::new(
        "Semantic Search".to_string(),
        num_searches,
        duration,
        &latencies,
    ))
}

/// Test logic fact insertion throughput
async fn test_logic_facts(node: &Node, config: &LoadTestConfig) -> ipfrs::Result<TestMetrics> {
    let mut latencies = Vec::new();
    let start = Instant::now();

    for i in 0..config.num_facts {
        let fact = Predicate::new(
            "person".to_string(),
            vec![
                Term::Const(Constant::String(format!("person_{}", i))),
                Term::Const(Constant::Int(i as i64)),
            ],
        );

        let op_start = Instant::now();
        node.add_fact(fact)?;
        latencies.push(op_start.elapsed());

        if (i + 1) % 20 == 0 {
            print!("\rAdded {} / {} facts", i + 1, config.num_facts);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }

    println!();
    let duration = start.elapsed();
    Ok(TestMetrics::new(
        "Logic Fact Insertion".to_string(),
        config.num_facts,
        duration,
        &latencies,
    ))
}

/// Test logic inference throughput
async fn test_logic_inference(node: &Node, _config: &LoadTestConfig) -> ipfrs::Result<TestMetrics> {
    // Add some rules for inference
    let rule = Rule::new(
        Predicate::new("adult".to_string(), vec![Term::Var("X".to_string())]),
        vec![Predicate::new(
            "person".to_string(),
            vec![Term::Var("X".to_string()), Term::Var("Age".to_string())],
        )],
    );
    node.add_rule(rule)?;

    let num_inferences = 50;
    let mut latencies = Vec::new();
    let start = Instant::now();

    for i in 0..num_inferences {
        let goal = if i % 2 == 0 {
            Predicate::new(
                "person".to_string(),
                vec![Term::Var("X".to_string()), Term::Var("Age".to_string())],
            )
        } else {
            Predicate::new("adult".to_string(), vec![Term::Var("X".to_string())])
        };

        let op_start = Instant::now();
        let _ = node.infer(&goal)?;
        latencies.push(op_start.elapsed());

        if (i + 1) % 10 == 0 {
            print!("\rInferred {} / {} queries", i + 1, num_inferences);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }

    println!();
    let duration = start.elapsed();
    Ok(TestMetrics::new(
        "Logic Inference".to_string(),
        num_inferences,
        duration,
        &latencies,
    ))
}

/// Test mixed workload (rapid sequential operations simulating concurrent load)
async fn test_concurrent_mixed(node: &Node, config: &LoadTestConfig) -> ipfrs::Result<TestMetrics> {
    let total_ops = 300;
    let mut latencies = Vec::new();
    let start = Instant::now();

    for idx in 0..total_ops {
        // Mix of operations
        let op_start = Instant::now();
        match idx % 3 {
            0 => {
                // Block operation
                let data = format!("Mixed block {}", idx);
                let block = Block::new(data.into_bytes().into())?;
                node.put_block(&block).await?;
            }
            1 => {
                // Semantic operation
                let data = format!("Mixed doc {}", idx);
                let block = Block::new(data.into_bytes().into())?;
                let cid = *block.cid();
                node.put_block(&block).await?;

                let embedding: Vec<f32> = (0..config.vector_dim)
                    .map(|j| ((idx + j) as f32 * 0.01) % 1.0)
                    .collect();
                node.index_content(&cid, &embedding).await?;
            }
            2 => {
                // Logic operation
                let fact = Predicate::new(
                    "mixed_fact".to_string(),
                    vec![Term::Const(Constant::Int(idx as i64))],
                );
                node.add_fact(fact)?;
            }
            _ => unreachable!(),
        }
        latencies.push(op_start.elapsed());

        if (idx + 1) % 30 == 0 {
            print!("\rProcessed {} / {} mixed operations", idx + 1, total_ops);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }

    println!();
    let duration = start.elapsed();

    Ok(TestMetrics::new(
        "Mixed Workload".to_string(),
        total_ops,
        duration,
        &latencies,
    ))
}

/// Test persistence save/load performance
async fn test_persistence(node: &mut Node, _config: &LoadTestConfig) -> ipfrs::Result<TestMetrics> {
    let mut latencies = Vec::new();

    // Test semantic index save/load
    let sem_path = "/tmp/ipfrs-load-test-semantic-index.bin";
    let start = Instant::now();
    node.save_semantic_index(sem_path).await?;
    latencies.push(start.elapsed());

    let start = Instant::now();
    node.load_semantic_index(sem_path).await?;
    latencies.push(start.elapsed());

    // Test knowledge base save/load
    let kb_path = "/tmp/ipfrs-load-test-kb.bin";
    let start = Instant::now();
    node.save_knowledge_base(kb_path).await?;
    latencies.push(start.elapsed());

    let start = Instant::now();
    node.load_knowledge_base(kb_path).await?;
    latencies.push(start.elapsed());

    print!("\r✓ Persistence operations complete");
    println!();

    // Cleanup
    std::fs::remove_file(sem_path).ok();
    std::fs::remove_file(kb_path).ok();

    let total_duration: Duration = latencies.iter().sum();
    Ok(TestMetrics::new(
        "Persistence Save/Load".to_string(),
        4, // 2 saves + 2 loads
        total_duration,
        &latencies,
    ))
}

/// Print overall summary
fn print_summary(metrics: &[TestMetrics]) {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    LOAD TEST SUMMARY                          ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");

    for metric in metrics {
        println!(
            "║ {:<30} {:>10.2} ops/s                 ║",
            metric.name, metric.ops_per_sec
        );
    }

    println!("╚═══════════════════════════════════════════════════════════════╝");

    // Calculate totals
    let total_ops: usize = metrics.iter().map(|m| m.total_operations).sum();
    let total_time: Duration = metrics.iter().map(|m| m.duration).sum();
    let overall_throughput = total_ops as f64 / total_time.as_secs_f64();

    println!("\nOverall Statistics:");
    println!("  Total operations: {}", total_ops);
    println!("  Total time: {:.2}s", total_time.as_secs_f64());
    println!("  Overall throughput: {:.2} ops/sec", overall_throughput);
}
