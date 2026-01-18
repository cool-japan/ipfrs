//! Production IPFRS Node Example
//!
//! This example demonstrates a production-ready IPFRS node with:
//! - Prometheus metrics
//! - Health check endpoints
//! - Distributed tracing (OpenTelemetry)
//! - Graceful shutdown
//! - Error recovery (retry + circuit breaker)
//!
//! # Usage
//!
//! ```bash
//! cargo run --package ipfrs --example production_node
//! ```
//!
//! Then in another terminal:
//! - Metrics: `curl http://localhost:9000/metrics`
//! - Health: Check liveness and readiness probes
//! - Shutdown: Press Ctrl+C for graceful shutdown

use ipfrs::{
    health::{HealthChecker, HealthStatus},
    metrics::{self, MetricsRegistry},
    recovery::{retry_async, CircuitBreaker, RetryPolicy},
    shutdown::{wait_for_signal, ShutdownCoordinator},
    tracing_setup::{init_tracing, TracingConfig},
    Block, Constant, Node, NodeConfig, Predicate, Rule, Term,
};
use std::path::PathBuf;
use std::time::Duration;
use tracing::{error, info, info_span, warn};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Production IPFRS Node...\n");

    // 1. Initialize Distributed Tracing
    println!("📊 Setting up observability...");
    let tracing_config = TracingConfig::new("ipfrs-production-example".to_string())
        .with_log_level("info".to_string());

    let _tracing_guard = init_tracing(tracing_config)?;
    info!("Distributed tracing initialized");

    // 2. Initialize Prometheus Metrics
    let metrics_registry = MetricsRegistry::new();
    let metrics_addr = "127.0.0.1:9000".parse()?;
    metrics_registry.init_prometheus(metrics_addr)?;
    info!(
        "Prometheus metrics available at http://{}/metrics",
        metrics_addr
    );

    // 3. Initialize Health Checker
    let health_checker = HealthChecker::new();
    info!("Health checks initialized");

    // 4. Initialize Shutdown Coordinator
    let shutdown = ShutdownCoordinator::new(Duration::from_secs(30));
    info!("Shutdown coordinator initialized");

    // 5. Create and Start IPFRS Node
    println!("\n📦 Starting IPFRS node...");
    let mut config = NodeConfig::default();
    config.storage.path = PathBuf::from("/tmp/ipfrs-production-example");
    config.enable_semantic = true;
    config.enable_tensorlogic = true;

    let mut node = Node::new(config)?;
    node.start().await?;
    info!("IPFRS node started successfully");

    // 6. Run Health Check Loop
    let health_shutdown = shutdown.clone();
    tokio::spawn(async move {
        health_check_loop(health_checker, health_shutdown).await;
    });

    // 7. Update Metrics Loop
    let metrics_shutdown = shutdown.clone();
    tokio::spawn(async move {
        metrics_update_loop(metrics_registry, metrics_shutdown).await;
    });

    // 8. Demonstrate Features (inline, not spawned)
    println!("\n✅ Production node is running!");
    println!("   - Metrics: http://localhost:9000/metrics");
    println!("   - Demonstrating features...\n");

    // Run demonstrations inline with a timeout
    let demo_shutdown = shutdown.clone();
    let mut demo_shutdown_rx = demo_shutdown.subscribe();
    tokio::select! {
        result = demonstrate_features(&mut node) => {
            if let Err(e) = result {
                error!("Error in demonstration: {}", e);
            } else {
                info!("✨ All demonstrations completed successfully");
            }
        }
        _ = demo_shutdown_rx.recv() => {
            info!("Shutdown requested during demonstration");
        }
    }

    println!("\n📊 System is operational. Press Ctrl+C to shutdown gracefully\n");

    // 9. Wait for Shutdown Signal
    let signal = wait_for_signal().await;
    info!("Received shutdown signal: {:?}", signal);

    // 10. Stop the node
    node.stop().await?;

    // 11. Initiate Graceful Shutdown
    shutdown.shutdown(signal);

    // 12. Wait for Components to Cleanup
    match shutdown.wait_for_shutdown().await {
        Ok(()) => info!("Graceful shutdown completed successfully"),
        Err(()) => warn!("Graceful shutdown timed out"),
    }

    println!("\n👋 Production node stopped\n");
    Ok(())
}

/// Demonstrate IPFRS features with production patterns
async fn demonstrate_features(node: &mut Node) -> Result<(), Box<dyn std::error::Error>> {
    info!("🎯 Demonstrating production features...");

    // Feature 1: Block Operations with Retry
    demonstrate_block_ops_with_retry(node).await?;

    // Feature 2: Semantic Search with Circuit Breaker
    demonstrate_semantic_with_circuit_breaker(node).await?;

    // Feature 3: Logic Programming with Error Handling
    demonstrate_logic_with_error_handling(node).await?;

    Ok(())
}

/// Demonstrate block operations with retry logic
async fn demonstrate_block_ops_with_retry(
    node: &mut Node,
) -> Result<(), Box<dyn std::error::Error>> {
    let span = info_span!("block_operations");
    let _guard = span.enter();

    info!("📦 Block Operations with Retry");

    // Create and store blocks with retry policy
    let retry_policy = RetryPolicy::exponential(3, Duration::from_millis(100));

    for i in 0..5 {
        let data = format!("Production Block {}", i);
        let data_len = data.len();
        let start = std::time::Instant::now();

        // Create a closure that captures only what we need
        let data_clone = data.clone();
        let cid = retry_async(retry_policy.clone(), || {
            let data_inner = data_clone.clone();
            async move {
                let block = Block::new(data_inner.into_bytes().into())
                    .map_err(|e| format!("Block creation error: {}", e))?;
                let cid = *block.cid();
                Ok::<_, String>(cid)
            }
        })
        .await?;

        // Store the block (outside retry for simplicity in this demo)
        let block = Block::new(data.into_bytes().into())?;
        node.put_block(&block).await?;

        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Record metrics
        metrics::record_block_put(data_len, duration_ms);

        info!("  ✓ Stored block {}: {} ({:.2}ms)", i, cid, duration_ms);
    }

    // Update storage metrics
    let stats = node.storage_stats()?;
    metrics::set_block_count(stats.num_blocks);
    // Note: total_size is not available in StorageStats, only recording block count
    metrics::set_storage_size_bytes(0); // Placeholder

    info!("  📊 Storage: {} blocks", stats.num_blocks);

    Ok(())
}

/// Demonstrate semantic search with circuit breaker
async fn demonstrate_semantic_with_circuit_breaker(
    node: &mut Node,
) -> Result<(), Box<dyn std::error::Error>> {
    let span = info_span!("semantic_search");
    let _guard = span.enter();

    info!("🔍 Semantic Search with Circuit Breaker");

    // Create circuit breaker for search operations
    let breaker = CircuitBreaker::new(5, 2, Duration::from_secs(60));

    // Index some content
    for i in 0..10 {
        let data = format!("Document {}: AI and machine learning content", i);
        let block = Block::new(data.into_bytes().into())?;
        let cid = *block.cid();
        node.put_block(&block).await?;

        // Generate embedding (simplified)
        let embedding: Vec<f32> = (0..768).map(|j| ((i + j) as f32) / 100.0).collect();

        let start = std::time::Instant::now();
        node.index_content(&cid, &embedding).await?;

        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        metrics::record_vector_index(768, duration_ms);
    }

    // Perform search with circuit breaker pattern
    let query: Vec<f32> = (0..768).map(|i| i as f32 / 100.0).collect();

    // Check if circuit breaker allows the operation
    let start = std::time::Instant::now();
    let results = if breaker.is_available() {
        match node.search_similar(&query, 5).await {
            Ok(results) => {
                breaker.record_success();
                results
            }
            Err(e) => {
                breaker.record_failure();
                return Err(format!("Search error: {}", e).into());
            }
        }
    } else {
        return Err("Circuit breaker is open, search unavailable".into());
    };

    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    metrics::record_similarity_search(5, results.len(), duration_ms);

    info!(
        "  ✓ Found {} similar items ({:.2}ms)",
        results.len(),
        duration_ms
    );

    // Update semantic stats
    let stats = node.semantic_stats()?;
    metrics::set_vector_count(stats.num_vectors);

    info!(
        "  📊 Vectors: {}, Dimension: {}",
        stats.num_vectors, stats.dimension
    );

    Ok(())
}

/// Demonstrate logic programming with error handling
async fn demonstrate_logic_with_error_handling(
    node: &mut Node,
) -> Result<(), Box<dyn std::error::Error>> {
    let span = info_span!("logic_programming");
    let _guard = span.enter();

    info!("🧠 Logic Programming with Error Handling");

    // Add facts with error handling
    let facts = vec![("Alice", "Bob"), ("Bob", "Charlie"), ("Charlie", "Diana")];

    for (parent, child) in &facts {
        let fact = Predicate::new(
            "parent".to_string(),
            vec![
                Term::Const(Constant::String(parent.to_string())),
                Term::Const(Constant::String(child.to_string())),
            ],
        );

        let start = std::time::Instant::now();
        node.add_fact(fact)?;

        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        metrics::record_fact_add(duration_ms);
    }

    // Add rule
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

    let start = std::time::Instant::now();
    node.add_rule(rule)?;

    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    metrics::record_rule_add(duration_ms);

    // Run inference
    let goal = Predicate::new(
        "grandparent".to_string(),
        vec![
            Term::Var("X".to_string()),
            Term::Const(Constant::String("Charlie".to_string())),
        ],
    );

    let start = std::time::Instant::now();
    let results = node.infer(&goal)?;

    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    metrics::record_inference(results.len(), duration_ms);

    info!(
        "  ✓ Inference found {} results ({:.2}ms)",
        results.len(),
        duration_ms
    );

    // Update logic stats
    let stats = node.tensorlogic_stats()?;
    metrics::set_kb_stats(stats.num_facts, stats.num_rules);

    info!(
        "  📊 Knowledge Base: {} facts, {} rules",
        stats.num_facts, stats.num_rules
    );

    Ok(())
}

/// Health check monitoring loop
async fn health_check_loop(checker: HealthChecker, shutdown: ShutdownCoordinator) {
    let mut interval = tokio::time::interval(Duration::from_secs(10));
    let mut shutdown_rx = shutdown.subscribe();

    loop {
        tokio::select! {
            _ = interval.tick() => {
                // Perform readiness check
                let health = checker.check_readiness(true, true, true, true);

                match health.status {
                    HealthStatus::Healthy => {
                        info!("❤️  Health check: HEALTHY (uptime: {}s)", health.uptime_seconds);
                    }
                    HealthStatus::Degraded => {
                        warn!("⚠️  Health check: DEGRADED (uptime: {}s)", health.uptime_seconds);
                    }
                    HealthStatus::Unhealthy => {
                        error!("❌ Health check: UNHEALTHY (uptime: {}s)", health.uptime_seconds);
                    }
                }
            }
            _ = shutdown_rx.recv() => {
                info!("Health check loop shutting down");
                break;
            }
        }
    }
}

/// Metrics update loop
async fn metrics_update_loop(registry: MetricsRegistry, shutdown: ShutdownCoordinator) {
    let mut interval = tokio::time::interval(Duration::from_secs(5));
    let mut shutdown_rx = shutdown.subscribe();

    loop {
        tokio::select! {
            _ = interval.tick() => {
                let uptime = registry.uptime_seconds();
                metrics::set_uptime_seconds(uptime);
            }
            _ = shutdown_rx.recv() => {
                info!("Metrics update loop shutting down");
                break;
            }
        }
    }
}
