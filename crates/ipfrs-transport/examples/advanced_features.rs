//! Advanced features demonstration
//!
//! This example demonstrates the Phase 11 advanced features:
//! - Request coalescing for duplicate elimination
//! - Connection migration for network changes
//! - Advanced request scheduling with multiple policies
//!
//! Run with: cargo run --example advanced_features

use bytes::Bytes;
use ipfrs_core::Cid;
use ipfrs_transport::{
    AdvancedScheduler, CoalescerConfig, ConnectionMigration, MigrationConfig, RequestCoalescer,
    SchedulePriority, ScheduledRequest, SchedulingPolicy,
};
use multihash::Multihash;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;

fn dummy_cid(seed: u64) -> Cid {
    let data = seed.to_le_bytes();
    let hash = Multihash::wrap(0x12, &data).unwrap();
    Cid::new_v1(0x55, hash)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Features Demonstration ===\n");

    // Demonstrate request coalescing
    demonstrate_request_coalescing().await?;

    println!();

    // Demonstrate connection migration
    demonstrate_connection_migration().await?;

    println!();

    // Demonstrate advanced scheduling
    demonstrate_advanced_scheduling().await?;

    println!("\n=== All demonstrations completed successfully! ===");
    Ok(())
}

async fn demonstrate_request_coalescing() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Request Coalescing Demonstration ---");

    let config = CoalescerConfig {
        coalesce_window: Duration::from_millis(20),
        max_waiters_per_request: 100,
        broadcast_capacity: 128,
        enable_stats: true,
    };

    let coalescer = Arc::new(RequestCoalescer::new(config));
    let cid = dummy_cid(12345);

    println!("Scenario: 10 concurrent requests for the same block");
    println!("Expected: Only 1 network request, 9 coalesced\n");

    // Spawn 10 concurrent "requests" for the same CID
    let mut handles = vec![];

    for i in 0..10 {
        let coalescer_clone = coalescer.clone();
        let cid_clone = cid;

        let handle = tokio::spawn(async move {
            let start = Instant::now();

            // Register the request
            let rx = coalescer_clone.register_request(&cid_clone).await.unwrap();

            if let Some(mut receiver) = rx {
                // This is a coalesced request - wait for the result
                println!("  Request {}: Waiting for coalesced result", i);
                match receiver.recv().await {
                    Ok(Ok(data)) => {
                        println!(
                            "  Request {}: Received data ({} bytes) in {:?}",
                            i,
                            data.len(),
                            start.elapsed()
                        );
                    }
                    Ok(Err(e)) => println!("  Request {}: Error: {}", i, e),
                    Err(e) => println!("  Request {}: Channel error: {}", i, e),
                }
            } else {
                // This is the first request - simulate fetching the data
                println!("  Request {}: Fetching data (first request)", i);
                sleep(Duration::from_millis(50)).await; // Simulate network delay

                let data = Bytes::from(format!("Block data for CID {}", i));
                coalescer_clone.complete_request(&cid_clone, data).await;
                println!("  Request {}: Completed fetch in {:?}", i, start.elapsed());
            }
        });

        handles.push(handle);

        // Small delay to ensure first request starts
        if i == 0 {
            sleep(Duration::from_millis(5)).await;
        }
    }

    // Wait for all requests to complete
    for handle in handles {
        handle.await?;
    }

    // Show statistics
    let stats = coalescer.stats().await;
    println!("\nCoalescing Statistics:");
    println!("  Total requests: {}", stats.total_requests);
    println!("  Unique requests: {}", stats.unique_requests);
    println!("  Coalesced requests: {}", stats.coalesced_requests);
    println!("  Efficiency: {:.1}%", stats.efficiency() * 100.0);
    println!("  Reduction ratio: {:.1}%", stats.reduction_ratio() * 100.0);
    println!(
        "  Average waiters per request: {:.1}",
        stats.avg_waiters_per_request
    );

    Ok(())
}

async fn demonstrate_connection_migration() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Connection Migration Demonstration ---");

    let config = MigrationConfig {
        enable_auto_migration: true,
        migration_timeout: Duration::from_secs(5),
        max_retries: 3,
        grace_period: Duration::from_secs(2),
        ..Default::default()
    };

    let migration = Arc::new(ConnectionMigration::new(config));

    // Setup event callback
    let event_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let count_clone = event_count.clone();

    migration
        .on_event(move |event| {
            count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            match event {
                ipfrs_transport::MigrationEvent::NetworkChangeDetected { old_addr, new_addr } => {
                    println!("  Event: Network change detected");
                    println!("    Old address: {}", old_addr);
                    println!("    New address: {}", new_addr);
                }
                ipfrs_transport::MigrationEvent::MigrationStarted {
                    connection_id,
                    from_addr,
                    to_addr,
                } => {
                    println!("  Event: Migration started");
                    println!("    Connection: {}", connection_id);
                    println!("    From: {} -> To: {}", from_addr, to_addr);
                }
                ipfrs_transport::MigrationEvent::MigrationCompleted {
                    connection_id,
                    new_addr,
                    duration,
                } => {
                    println!("  Event: Migration completed");
                    println!("    Connection: {}", connection_id);
                    println!("    New address: {}", new_addr);
                    println!("    Duration: {:?}", duration);
                }
                ipfrs_transport::MigrationEvent::MigrationFailed {
                    connection_id,
                    reason,
                    retry_count,
                } => {
                    println!("  Event: Migration failed");
                    println!("    Connection: {}", connection_id);
                    println!("    Reason: {}", reason);
                    println!("    Retry count: {}", retry_count);
                }
            }
        })
        .await;

    println!("Scenario 1: Successful migration (WiFi -> Cellular)\n");

    let old_addr = "192.168.1.100:8000".parse()?;
    let new_addr = "10.0.0.50:8000".parse()?;

    migration
        .start_migration("conn1".to_string(), old_addr, new_addr)
        .await?;

    // Simulate migration time
    sleep(Duration::from_millis(100)).await;

    migration.complete_migration("conn1").await?;

    println!("\nScenario 2: Failed migration with retry\n");

    let old_addr2 = "192.168.1.100:8001".parse()?;
    let new_addr2 = "10.0.0.51:8001".parse()?;

    migration
        .start_migration("conn2".to_string(), old_addr2, new_addr2)
        .await?;

    // Simulate failure
    sleep(Duration::from_millis(50)).await;
    let _ = migration
        .fail_migration("conn2", "Network timeout".to_string())
        .await;

    sleep(Duration::from_millis(50)).await;
    let _ = migration
        .fail_migration("conn2", "Network timeout".to_string())
        .await;

    sleep(Duration::from_millis(50)).await;
    let _ = migration
        .fail_migration("conn2", "Network timeout".to_string())
        .await;

    // Show statistics
    let stats = migration.stats().await;
    println!("\nMigration Statistics:");
    println!("  Total migrations: {}", stats.total_migrations);
    println!("  Successful: {}", stats.successful_migrations);
    println!("  Failed: {}", stats.failed_migrations);
    println!("  Success rate: {:.1}%", stats.success_rate() * 100.0);
    println!("  Average duration: {:?}", stats.avg_migration_duration);
    println!(
        "  Total events received: {}",
        event_count.load(std::sync::atomic::Ordering::SeqCst)
    );

    Ok(())
}

async fn demonstrate_advanced_scheduling() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Advanced Scheduling Demonstration ---");

    // Test multiple scheduling policies
    let policies = vec![
        (SchedulingPolicy::Fifo, "FIFO"),
        (SchedulingPolicy::ShortestJobFirst, "Shortest Job First"),
        (
            SchedulingPolicy::EarliestDeadlineFirst,
            "Earliest Deadline First",
        ),
        (
            SchedulingPolicy::WeightedFairQueueing,
            "Weighted Fair Queueing",
        ),
        (SchedulingPolicy::MultilevelFeedback, "Multi-Level Feedback"),
    ];

    for (policy, name) in policies {
        println!("\n{} Policy:", name);
        demonstrate_scheduling_policy(policy).await?;
    }

    Ok(())
}

async fn demonstrate_scheduling_policy(
    policy: SchedulingPolicy,
) -> Result<(), Box<dyn std::error::Error>> {
    let scheduler = AdvancedScheduler::new(policy);

    // Create requests with different characteristics
    let requests = vec![
        // Large, low priority, far deadline
        ScheduledRequest::new(dummy_cid(1), SchedulePriority::Low)
            .with_size(10_000_000)
            .with_deadline(Instant::now() + Duration::from_secs(60)),
        // Small, normal priority, medium deadline
        ScheduledRequest::new(dummy_cid(2), SchedulePriority::Normal)
            .with_size(1_000)
            .with_deadline(Instant::now() + Duration::from_secs(30)),
        // Medium, high priority, near deadline
        ScheduledRequest::new(dummy_cid(3), SchedulePriority::High)
            .with_size(100_000)
            .with_deadline(Instant::now() + Duration::from_secs(5)),
        // Small, urgent priority, very near deadline
        ScheduledRequest::new(dummy_cid(4), SchedulePriority::Urgent)
            .with_size(500)
            .with_deadline(Instant::now() + Duration::from_secs(1)),
        // Medium, critical priority, immediate deadline
        ScheduledRequest::new(dummy_cid(5), SchedulePriority::Critical)
            .with_size(50_000)
            .with_deadline(Instant::now() + Duration::from_millis(100)),
    ];

    // Schedule all requests
    for req in requests {
        scheduler.schedule(req).await;
    }

    println!("  Scheduled 5 requests with varying priorities, sizes, and deadlines");
    println!("  Processing order:");

    // Process requests in scheduled order
    let mut order = vec![];
    while let Some(req) = scheduler.next().await {
        let size_str = req
            .estimated_size
            .map(|s| format!("{} bytes", s))
            .unwrap_or_else(|| "unknown".to_string());
        let deadline_str = req
            .deadline
            .map(|d| {
                if d > Instant::now() {
                    format!("{}s from now", d.duration_since(Instant::now()).as_secs())
                } else {
                    "overdue".to_string()
                }
            })
            .unwrap_or_else(|| "none".to_string());

        println!(
            "    CID {:?}: priority={:?}, size={}, deadline={}",
            req.cid, req.priority, size_str, deadline_str
        );

        // Simulate processing
        let completion_time = Duration::from_millis(10);
        scheduler.mark_completed(&req, completion_time).await;

        order.push(req.cid);
    }

    // Show statistics
    let stats = scheduler.stats().await;
    println!("\n  Scheduling Statistics:");
    println!("    Total scheduled: {}", stats.total_scheduled);
    println!("    Total completed: {}", stats.total_completed);
    println!("    Average wait time: {:?}", stats.avg_wait_time);
    println!(
        "    Average completion time: {:?}",
        stats.avg_completion_time
    );
    println!("    Deadline misses: {}", stats.deadline_misses);
    println!(
        "    Deadline miss rate: {:.1}%",
        stats.deadline_miss_rate() * 100.0
    );

    Ok(())
}
