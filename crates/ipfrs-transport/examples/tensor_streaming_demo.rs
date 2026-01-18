//! Tensor Streaming Demo Example
//!
//! This example demonstrates tensor streaming capabilities including:
//! - Creating tensor metadata with chunks
//! - Stream request queue with priority scheduling
//! - Backpressure control
//! - Progress tracking
//!
//! Run with: cargo run --example tensor_streaming_demo

use ipfrs_core::Cid;
use ipfrs_transport::{
    BackpressureConfig, BackpressureController, StreamRequest, StreamRequestQueue, TensorMetadata,
};
use multihash::Multihash;
use std::time::{Duration, Instant};

/// Create a dummy CID for demonstration
fn create_cid(seed: u64) -> Cid {
    let data = seed.to_le_bytes();
    let hash = Multihash::wrap(0x12, &data).unwrap();
    Cid::new_v1(0x55, hash)
}

fn main() {
    println!("=== Tensor Streaming Demo ===\n");

    // 1. Create Tensor Metadata
    println!("--- Tensor Metadata ---\n");

    // Simulate a large tensor split into chunks
    let tensor_cid = create_cid(1000);
    let chunk_cids: Vec<Cid> = (0..10).map(|i| create_cid(1000 + i)).collect();

    let total_size = 1024 * 1024 * 3 * 4u64; // 12 MB (float32 = 4 bytes)

    let metadata = TensorMetadata::new(tensor_cid)
        .with_dtype("float32")
        .with_shape(vec![1024, 1024, 3]) // 1024x1024x3 tensor
        .with_size(total_size)
        .with_chunks(chunk_cids.clone())
        .with_priority_hint(750); // High priority

    println!("Created tensor metadata:");
    println!("  CID: {}", metadata.cid);
    println!("  Shape: {:?}", metadata.shape.as_ref().unwrap());
    println!(
        "  Total size: {} bytes ({:.2} MB)",
        metadata.size_bytes.unwrap(),
        metadata.size_bytes.unwrap() as f64 / 1_000_000.0
    );
    println!("  Chunks: {}", metadata.chunks.len());
    println!("  Priority hint: {}", metadata.priority_hint.unwrap());

    // 2. Stream Request Queue with Priority Scheduling
    println!("\n--- Stream Request Queue ---\n");

    let mut queue = StreamRequestQueue::new(100);

    // Add multiple stream requests with different priorities
    let critical_request = StreamRequest {
        cid: create_cid(2000),
        priority: 1000, // Critical priority
        deadline: Some(Instant::now() + Duration::from_secs(5)),
        queued_at: Instant::now(),
    };

    let high_priority_request = StreamRequest {
        cid: create_cid(2001),
        priority: 750, // High priority
        deadline: None,
        queued_at: Instant::now(),
    };

    let low_priority_request = StreamRequest {
        cid: create_cid(2002),
        priority: 250, // Low priority
        deadline: None,
        queued_at: Instant::now(),
    };

    queue.push(critical_request);
    queue.push(high_priority_request);
    queue.push(low_priority_request);

    println!("Added 3 stream requests to queue:");
    println!("  Critical priority (1000) with 5s deadline");
    println!("  High priority (750)");
    println!("  Low priority (250)");

    // Get highest priority request
    if let Some(request) = queue.pop() {
        println!("\nHighest priority stream request:");
        println!("  CID: {}", request.cid);
        println!("  Priority: {}", request.priority);
        println!("  Has deadline: {}", request.deadline.is_some());
    }

    // 3. Backpressure Control
    println!("\n--- Backpressure Control ---\n");

    let backpressure_config = BackpressureConfig {
        max_pending: 100,
        high_watermark: 80,
        low_watermark: 20,
        max_buffer_bytes: 100 * 1024 * 1024, // 100 MB
    };

    println!("Backpressure controller configured:");
    println!("  Max pending: {} chunks", backpressure_config.max_pending);
    println!(
        "  High watermark: {} pending chunks",
        backpressure_config.high_watermark
    );
    println!(
        "  Low watermark: {} pending chunks",
        backpressure_config.low_watermark
    );
    println!(
        "  Max buffer: {} MB",
        backpressure_config.max_buffer_bytes / 1024 / 1024
    );

    let mut backpressure = BackpressureController::new(backpressure_config);

    // Simulate sending chunks
    for _i in 0..50 {
        backpressure.on_send(1024 * 1024); // 1 MB chunks
    }

    println!("\nSent 50 chunks (1 MB each)");
    println!("Pending count: {}", backpressure.pending_count());
    println!(
        "Pending bytes: {} MB",
        backpressure.pending_bytes() / 1024 / 1024
    );
    println!("Paused: {}", backpressure.is_paused());
    println!("Should accept more: {}", backpressure.should_accept());

    // 4. Tensor with Dependencies (for computation graphs)
    println!("\n--- Tensor Dependencies ---\n");

    let input_tensor = create_cid(5000);
    let weight_tensor = create_cid(5001);

    let output_tensor = TensorMetadata::new(create_cid(5002))
        .with_dtype("float32")
        .with_shape(vec![128, 128])
        .with_size(128 * 128 * 4)
        .with_chunks(vec![create_cid(5003)])
        .with_dependencies(vec![input_tensor, weight_tensor])
        .with_priority_hint(500); // Normal priority

    println!("Created output tensor with dependencies:");
    println!("  Output CID: {}", output_tensor.cid);
    println!("  Depends on {} tensors:", output_tensor.dependencies.len());
    for (i, dep) in output_tensor.dependencies.iter().enumerate() {
        println!("    Dependency {}: {}", i + 1, dep);
    }

    // 5. Tensor with Deadline
    println!("\n--- Tensor with Deadline ---\n");

    let deadline = Instant::now() + Duration::from_secs(10);

    let urgent_tensor = TensorMetadata::new(create_cid(6000))
        .with_dtype("float16")
        .with_shape(vec![64, 64])
        .with_size(64 * 64 * 2)
        .with_chunks(vec![create_cid(6001)])
        .with_deadline(deadline)
        .with_priority_hint(900); // Urgent priority

    println!("Created urgent tensor with deadline:");
    println!("  CID: {}", urgent_tensor.cid);
    println!("  Priority hint: {:?}", urgent_tensor.priority_hint);
    println!("  Deadline: 10 seconds from now");
    println!("  Shape: {:?}", urgent_tensor.shape);
    println!("  Data type: {:?}", urgent_tensor.dtype);

    println!("\n✓ Tensor streaming demo completed successfully!");
    println!("\nThis example demonstrated:");
    println!("  • Creating tensor metadata with chunks and priorities");
    println!("  • Using stream request queue for priority scheduling");
    println!("  • Backpressure control for flow management");
    println!("  • Tensor dependencies for computation graphs");
    println!("  • Deadline-based priority elevation");
}
