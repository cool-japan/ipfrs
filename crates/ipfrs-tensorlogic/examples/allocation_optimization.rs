//! Example: Allocation optimization techniques
//!
//! This example demonstrates:
//! - Buffer pooling for reduced allocations
//! - Stack-based allocation for small buffers
//! - Adaptive buffers (stack/heap hybrid)
//! - Zero-copy conversions

use ipfrs_tensorlogic::{
    AdaptiveBuffer, BufferPool, StackBuffer, TypedBufferPool, ZeroCopyConverter,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Allocation Optimization Example ===\n");

    // 1. Buffer Pool
    println!("--- Buffer Pool ---");
    let pool = BufferPool::new(4096, 10); // 4KB buffers, max 10 in pool
    println!("Created buffer pool: 4KB buffers, max 10 pooled");

    // Acquire and use buffers
    let start = Instant::now();
    for i in 0..100 {
        let mut buffer = pool.acquire();
        buffer.as_mut().extend_from_slice(&[i as u8; 100]);
        // Buffer automatically returned to pool when dropped
    }
    let pool_duration = start.elapsed();
    println!("100 pooled allocations: {:?}", pool_duration);
    println!("Pool size after: {}", pool.size());

    // Compare with direct allocation
    let start = Instant::now();
    for i in 0..100 {
        let mut buffer = Vec::with_capacity(4096);
        buffer.extend_from_slice(&[i as u8; 100]);
    }
    let direct_duration = start.elapsed();
    println!("100 direct allocations: {:?}", direct_duration);
    println!(
        "Speedup: {:.2}x\n",
        direct_duration.as_nanos() as f64 / pool_duration.as_nanos() as f64
    );

    // 2. Typed Buffer Pool
    println!("--- Typed Buffer Pool ---");
    let float_pool = TypedBufferPool::<f32>::new(1024, 10);
    println!("Created typed buffer pool for f32");

    let mut buffer = float_pool.acquire();
    buffer.extend((0..100).map(|i| i as f32));
    println!("Buffer length: {}", buffer.len());
    println!("Sum: {}", buffer.iter().sum::<f32>());
    drop(buffer);
    println!("Pool size: {}\n", float_pool.size());

    // 3. Stack Buffer
    println!("--- Stack Buffer ---");
    let mut stack_buf = StackBuffer::<256>::new();
    println!(
        "Created stack buffer: {} bytes capacity",
        stack_buf.capacity()
    );

    stack_buf.write(b"Hello, ")?;
    stack_buf.write(b"World!")?;
    println!("Written: {:?}", std::str::from_utf8(stack_buf.as_slice())?);
    println!(
        "Length: {}, Remaining: {}",
        stack_buf.len(),
        stack_buf.remaining()
    );

    // Try to overflow
    let large_data = vec![0u8; 300];
    match stack_buf.write(&large_data) {
        Ok(_) => println!("Should not reach here"),
        Err(e) => println!("Overflow prevented: {}\n", e),
    }

    // 4. Adaptive Buffer
    println!("--- Adaptive Buffer ---");

    // Small data - stays on stack
    let mut adaptive_small = AdaptiveBuffer::new(10);
    adaptive_small.write(b"small")?;
    println!(
        "Small buffer (hint=10): {} bytes, still on stack",
        adaptive_small.len()
    );

    // Large data - uses heap
    let mut adaptive_large = AdaptiveBuffer::new(1000);
    adaptive_large.write(&vec![0u8; 500])?;
    println!(
        "Large buffer (hint=1000): {} bytes, using heap",
        adaptive_large.len()
    );

    // Auto-upgrade from stack to heap
    let mut adaptive_upgrade = AdaptiveBuffer::new(10);
    adaptive_upgrade.write(&[0u8; 100])?; // Small hint but large write
    adaptive_upgrade.write(&[0u8; 200])?; // Continue writing
    println!(
        "Auto-upgraded buffer: {} bytes (started on stack, upgraded to heap)\n",
        adaptive_upgrade.len()
    );

    // 5. Zero-Copy Conversions
    println!("--- Zero-Copy Conversions ---");

    let floats: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    println!(
        "Original floats: {} elements, {} bytes",
        floats.len(),
        floats.len() * 4
    );

    // Zero-copy conversion to bytes
    let start = Instant::now();
    let bytes = ZeroCopyConverter::slice_to_bytes(&floats);
    let zero_copy_duration = start.elapsed();
    println!(
        "Zero-copy to bytes: {} bytes in {:?}",
        bytes.len(),
        zero_copy_duration
    );

    // Compare with copying conversion
    let start = Instant::now();
    let copied_bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
    let copy_duration = start.elapsed();
    println!(
        "Copying to bytes: {} bytes in {:?}",
        copied_bytes.len(),
        copy_duration
    );
    println!(
        "Speedup: {:.2}x",
        copy_duration.as_nanos() as f64 / zero_copy_duration.as_nanos() as f64
    );

    // Zero-copy conversion back
    let floats_back: &[f32] = ZeroCopyConverter::bytes_to_slice(bytes);
    println!("Zero-copy back: {} elements", floats_back.len());
    println!("Data integrity: {}\n", floats == floats_back);

    // 6. Batch Processing with Pool
    println!("--- Batch Processing with Pool ---");

    let processing_pool = BufferPool::new(8192, 8);
    let num_batches = 10;

    println!("Processing {} batches with pooled buffers...", num_batches);
    let start = Instant::now();

    for batch_id in 0..num_batches {
        let mut buffer = processing_pool.acquire();

        // Simulate processing
        for i in 0..100 {
            let data = format!("batch_{}_item_{}", batch_id, i);
            buffer.as_mut().extend_from_slice(data.as_bytes());
        }

        // Process buffer (simulated)
        let _ = buffer.len();
    }

    let total_duration = start.elapsed();
    println!("Processed {} batches in {:?}", num_batches, total_duration);
    println!("Average per batch: {:?}", total_duration / num_batches);
    println!(
        "Final pool size: {} (buffers available for reuse)",
        processing_pool.size()
    );

    // Summary
    println!("\n--- Summary ---");
    println!("✓ Buffer pooling reduces allocation overhead");
    println!("✓ Stack buffers avoid heap allocation for small data");
    println!("✓ Adaptive buffers automatically choose stack or heap");
    println!("✓ Zero-copy conversions eliminate unnecessary copying");
    println!("✓ Typed pools provide type-safe buffer reuse");

    println!("\n✓ Example completed successfully!");
    Ok(())
}
