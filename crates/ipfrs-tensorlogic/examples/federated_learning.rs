//! Example: Federated Learning with Gradient Compression and Differential Privacy
//!
//! This example demonstrates:
//! - Gradient compression (top-k, threshold, quantization)
//! - Gradient aggregation from multiple clients
//! - Differential privacy with noise addition

use ipfrs_tensorlogic::{GradientAggregator, GradientCompressor, GradientVerifier};
use rand::RngExt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Federated Learning Example ===\n");

    // Simulate multiple clients
    let num_clients = 5;
    let gradient_size = 10000;

    println!("Simulating federated learning with {} clients", num_clients);
    println!("Gradient size: {} parameters\n", gradient_size);

    // Generate random gradients for each client
    let mut rng = rand::rng();
    let client_gradients: Vec<Vec<f32>> = (0..num_clients)
        .map(|_| {
            (0..gradient_size)
                .map(|_| rng.random::<f32>() * 2.0 - 1.0) // Random values in [-1, 1]
                .collect()
        })
        .collect();

    println!("--- Gradient Statistics (Before Compression) ---");
    for (i, grad) in client_gradients.iter().enumerate() {
        let mean = grad.iter().sum::<f32>() / grad.len() as f32;
        let max = grad.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = grad.iter().cloned().fold(f32::INFINITY, f32::min);
        println!(
            "Client {}: mean={:.6}, min={:.6}, max={:.6}",
            i + 1,
            mean,
            min,
            max
        );
    }

    // Compress gradients using different methods
    println!("\n--- Gradient Compression ---");

    // Method 1: Top-k compression (keep top 10% values)
    println!("\n1. Top-k Compression (k=10%)");
    let k = (gradient_size as f32 * 0.1) as usize;
    let mut compressed_top_k = Vec::new();

    for (i, grad) in client_gradients.iter().enumerate() {
        let sparse = GradientCompressor::top_k(grad, vec![gradient_size], k)?;
        println!(
            "   Client {}: {} non-zeros ({:.1}% sparse)",
            i + 1,
            sparse.nnz(),
            sparse.sparsity_ratio() * 100.0
        );
        compressed_top_k.push(sparse);
    }

    // Method 2: Threshold compression (keep values > 0.5)
    println!("\n2. Threshold Compression (threshold=0.5)");
    let mut compressed_threshold = Vec::new();

    for (i, grad) in client_gradients.iter().enumerate() {
        let sparse = GradientCompressor::threshold(grad, vec![gradient_size], 0.5);
        println!(
            "   Client {}: {} non-zeros ({:.1}% sparse)",
            i + 1,
            sparse.nnz(),
            sparse.sparsity_ratio() * 100.0
        );
        compressed_threshold.push(sparse);
    }

    // Method 3: Quantization (int8)
    println!("\n3. Int8 Quantization");
    let mut compressed_quantized = Vec::new();

    for (i, grad) in client_gradients.iter().enumerate() {
        let quantized = GradientCompressor::quantize(grad, vec![gradient_size]);
        println!(
            "   Client {}: compression ratio: {:.2}x",
            i + 1,
            quantized.compression_ratio()
        );
        compressed_quantized.push(quantized);
    }

    // Aggregate gradients
    println!("\n--- Gradient Aggregation ---");

    // Convert sparse gradients back to dense for aggregation
    let dense_gradients: Vec<Vec<f32>> = compressed_top_k
        .iter()
        .map(|sparse| sparse.to_dense())
        .collect();

    // Unweighted average
    let avg_gradient = GradientAggregator::average(&dense_gradients)?;
    let avg_mean = avg_gradient.iter().sum::<f32>() / avg_gradient.len() as f32;
    println!("Unweighted average: mean={:.6}", avg_mean);

    // Weighted average (different weights for different clients)
    let weights = vec![0.3, 0.25, 0.2, 0.15, 0.1]; // Weights sum to 1.0
    let weighted_gradient = GradientAggregator::weighted_average(&client_gradients, &weights)?;
    let weighted_mean = weighted_gradient.iter().sum::<f32>() / weighted_gradient.len() as f32;
    println!("Weighted average: mean={:.6}", weighted_mean);
    println!("Weights used: {:?}", weights);

    // Apply momentum
    let momentum = 0.9;
    let prev_velocity = vec![0.0f32; gradient_size]; // Previous velocity
    let updated_gradient =
        GradientAggregator::apply_momentum(&avg_gradient, &prev_velocity, momentum)?;
    let momentum_mean = updated_gradient.iter().sum::<f32>() / updated_gradient.len() as f32;
    println!("With momentum (β={}): mean={:.6}", momentum, momentum_mean);

    // Apply gradient clipping
    println!("\n--- Gradient Clipping ---");
    let clip_norm = 1.0;
    let mut clipped_gradient = avg_gradient.clone();
    GradientVerifier::clip_by_norm(&mut clipped_gradient, clip_norm);
    let clipped_mean = clipped_gradient.iter().sum::<f32>() / clipped_gradient.len() as f32;
    println!("Clipped gradient mean: {:.6}", clipped_mean);
    println!("Clip norm: {}", clip_norm);

    // Summary
    println!("\n--- Summary ---");
    println!("✓ {} clients participated in training", num_clients);
    println!("✓ Gradients compressed using top-k, threshold, and quantization");
    println!("✓ Gradients aggregated with weighted averaging");
    println!("✓ Momentum applied with β={}", momentum);
    println!("✓ Gradient clipping applied with norm={}", clip_norm);
    println!("✓ Model update ready for distribution");

    println!("\n✓ Example completed successfully!");
    Ok(())
}
