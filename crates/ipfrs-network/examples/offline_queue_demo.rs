//! Offline Queue Usage Example
//!
//! This example demonstrates how to use the offline queue for mobile and
//! intermittent connectivity scenarios:
//! 1. Creating and configuring an offline queue
//! 2. Queuing requests when offline
//! 3. Priority-based request handling
//! 4. Automatic replay when network comes online
//! 5. Retry logic for failed requests
//! 6. Statistics tracking

use ipfrs_network::{
    OfflineQueue, OfflineQueueConfig, QueuedRequest, QueuedRequestType, RequestPriority,
};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("=== Offline Queue Example ===\n");

    // Scenario 1: Default Configuration
    println!("1. Creating offline queue with default configuration");
    let queue = OfflineQueue::new(OfflineQueueConfig::default())?;
    println!("   ✓ Queue created\n");

    // Scenario 2: Queuing Requests When Offline
    println!("2. Queuing requests while offline");

    let req1 = QueuedRequest::new(
        "req1".to_string(),
        QueuedRequestType::ProvideContent("QmTest1".to_string()),
        RequestPriority::Normal,
        Duration::from_secs(60),
    );

    let req2 = QueuedRequest::new(
        "req2".to_string(),
        QueuedRequestType::FindProviders("QmTest2".to_string()),
        RequestPriority::High,
        Duration::from_secs(60),
    );

    let req3 = QueuedRequest::new(
        "req3".to_string(),
        QueuedRequestType::PutValue {
            key: "my-key".to_string(),
            value: b"my-value".to_vec(),
        },
        RequestPriority::Critical,
        Duration::from_secs(60),
    );

    queue.enqueue(req1)?;
    queue.enqueue(req2)?;
    queue.enqueue(req3)?;

    println!("   Queued 3 requests:");
    println!("     - Request 1: Normal priority (ProvideContent)");
    println!("     - Request 2: High priority (FindProviders)");
    println!("     - Request 3: Critical priority (PutValue)");
    println!("   Pending requests: {}\n", queue.pending_count());

    // Scenario 3: Priority-Based Ordering
    println!("3. Network comes online - requests processed by priority");
    queue.set_online(true);
    println!("   Network status: Online");
    println!("   Processing requests in priority order:\n");

    while let Some(request) = queue.dequeue() {
        println!(
            "   Processing: {} (Priority: {:?})",
            request.id, request.priority
        );

        // Simulate request processing
        sleep(Duration::from_millis(100)).await;

        // Simulate success/failure
        let success = request.id != "req2"; // req2 will "fail"

        if success {
            println!("     ✓ Completed successfully");
            queue.mark_completed(&request.id, true);
        } else {
            println!("     ✗ Failed - will retry");
            queue.requeue(request)?;
        }
    }
    println!();

    // Scenario 4: Mobile Configuration
    println!("4. Creating queue with mobile-optimized configuration");
    let mobile_queue = OfflineQueue::new(OfflineQueueConfig::mobile())?;
    println!("   Mobile config:");
    println!("     - Max queue size: 500");
    println!("     - Persistence: Enabled");
    println!("     - Replay batch size: 5");
    println!("     - Replay delay: 200ms\n");

    // Scenario 5: Batch Replay
    println!("5. Batch replay demonstration");

    for i in 0..10 {
        let req = QueuedRequest::new(
            format!("batch_{}", i),
            QueuedRequestType::FindProviders(format!("QmBatch{}", i)),
            RequestPriority::Normal,
            Duration::from_secs(60),
        );
        mobile_queue.enqueue(req)?;
    }

    println!("   Queued 10 requests");
    mobile_queue.set_online(true);

    let batch = mobile_queue.get_replay_batch();
    println!("   Got batch of {} requests for replay", batch.len());

    for (idx, request) in batch.iter().enumerate() {
        println!("     {}: {}", idx + 1, request.id);
    }
    println!();

    // Scenario 6: Network Disconnection
    println!("6. Handling network disconnection");
    mobile_queue.set_online(false);
    println!("   Network status: Offline");

    let new_req = QueuedRequest::new(
        "offline_req".to_string(),
        QueuedRequestType::ProvideContent("QmOffline".to_string()),
        RequestPriority::High,
        Duration::from_secs(60),
    );

    mobile_queue.enqueue(new_req)?;
    println!("   Queued request while offline");
    println!("   Pending count: {}", mobile_queue.pending_count());

    // Try to dequeue while offline
    if mobile_queue.dequeue().is_none() {
        println!("   ✓ Correctly refusing to dequeue while offline\n");
    }

    // Scenario 7: IoT Configuration
    println!("7. IoT device configuration");
    let _iot_queue = OfflineQueue::new(OfflineQueueConfig::iot())?;
    println!("   IoT config:");
    println!("     - Max queue size: 100");
    println!("     - Replay batch size: 3");
    println!("     - Replay delay: 500ms\n");

    // Scenario 8: Statistics
    println!("8. Queue statistics");

    let stats = queue.stats();
    println!("   Queue statistics:");
    println!("     - Requests queued: {}", stats.requests_queued);
    println!("     - Requests completed: {}", stats.requests_completed);
    println!("     - Requests failed: {}", stats.requests_failed);
    println!("     - Requests retried: {}", stats.requests_retried);
    println!("     - Success rate: {:.2}%", stats.success_rate() * 100.0);
    println!(
        "     - Completion rate: {:.2}%",
        stats.completion_rate() * 100.0
    );
    println!("     - Online transitions: {}", stats.online_transitions);
    println!("     - Offline transitions: {}", stats.offline_transitions);

    println!("\n=== Example completed successfully ===");

    Ok(())
}
