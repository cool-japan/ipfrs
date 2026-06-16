//! PyTorch Checkpoint Support Demo
//!
//! This example demonstrates how to work with PyTorch model checkpoints in ipfrs-tensorlogic.
//! It shows:
//! 1. Creating checkpoints programmatically
//! 2. Adding model state and metadata
//! 3. Extracting checkpoint information
//! 4. Converting checkpoints to Safetensors format
//! 5. Inspecting checkpoint contents
//!
//! Run with: cargo run --example pytorch_checkpoint_demo

use anyhow::Result;
use ipfrs_tensorlogic::pytorch_checkpoint::{CheckpointMetadata, PyTorchCheckpoint, TensorData};

fn main() -> Result<()> {
    println!("=== PyTorch Checkpoint Support Demo ===\n");

    // Example 1: Create a checkpoint from scratch
    println!("1. Creating a PyTorch checkpoint programmatically");
    println!("   Building a simple neural network checkpoint...");

    let mut checkpoint = PyTorchCheckpoint::new();

    // Add layer weights
    let layer1_weight = create_random_weights(vec![128, 784], 0.1);
    checkpoint.add_tensor("layer1.weight".to_string(), layer1_weight);

    let layer1_bias = create_zeros(vec![128]);
    checkpoint.add_tensor("layer1.bias".to_string(), layer1_bias);

    let layer2_weight = create_random_weights(vec![10, 128], 0.1);
    checkpoint.add_tensor("layer2.weight".to_string(), layer2_weight);

    let layer2_bias = create_zeros(vec![10]);
    checkpoint.add_tensor("layer2.bias".to_string(), layer2_bias);

    // Add training metadata
    checkpoint.set_epoch(10);
    checkpoint.add_metadata("model_name".to_string(), "SimpleNN".to_string());
    checkpoint.add_metadata("dataset".to_string(), "MNIST".to_string());
    checkpoint.add_metadata("accuracy".to_string(), "0.95".to_string());

    println!("   ✓ Created checkpoint with 4 layers\n");

    // Example 2: Extract and display metadata
    println!("2. Extracting checkpoint metadata");
    let metadata = checkpoint.metadata();
    display_metadata(&metadata);

    // Example 3: Inspect state dict
    println!("3. Inspecting state dictionary");
    let state_dict = checkpoint.state_dict();
    println!("   Total parameters: {}", state_dict.len());

    for (name, tensor) in state_dict.iter() {
        let num_params: usize = tensor.shape.iter().product();
        println!(
            "   - {}: {:?} ({} parameters)",
            name, tensor.shape, num_params
        );
    }
    println!();

    // Example 4: Access individual tensors
    println!("4. Accessing individual tensor data");
    if let Some(weight_tensor) = state_dict.get("layer1.weight") {
        let weights = weight_tensor.as_f32()?;
        println!("   Layer1 weight shape: {:?}", weight_tensor.shape);
        println!("   First 5 weights: {:?}", &weights[..5.min(weights.len())]);
        println!("   Weight statistics:");
        let sum: f32 = weights.iter().sum();
        let mean = sum / weights.len() as f32;
        println!("     Mean: {:.6}", mean);
        println!(
            "     Min: {:.6}",
            weights.iter().cloned().fold(f32::INFINITY, f32::min)
        );
        println!(
            "     Max: {:.6}",
            weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );
    }
    println!();

    // Example 5: Convert to Safetensors format
    println!("5. Converting to Safetensors format");
    println!("   Converting checkpoint to Safetensors...");
    let safetensors_bytes = checkpoint.to_safetensors()?;
    println!(
        "   ✓ Converted to Safetensors ({} bytes)",
        safetensors_bytes.len()
    );
    println!("   Safetensors format provides:");
    println!("     - Faster loading");
    println!("     - Better security (no pickle execution)");
    println!("     - Zero-copy memory mapping");
    println!("     - Cross-language compatibility");
    println!();

    // Example 6: Create checkpoint with training history
    println!("6. Creating checkpoint with training history");
    let mut training_checkpoint = PyTorchCheckpoint::new();

    // Add simple model
    training_checkpoint.add_tensor(
        "fc.weight".to_string(),
        create_random_weights(vec![10, 20], 0.1),
    );
    training_checkpoint.add_tensor("fc.bias".to_string(), create_zeros(vec![10]));

    // Add training information
    training_checkpoint.set_epoch(50);
    training_checkpoint.add_metadata("optimizer".to_string(), "Adam".to_string());
    training_checkpoint.add_metadata("learning_rate".to_string(), "0.001".to_string());
    training_checkpoint.add_metadata("best_val_loss".to_string(), "0.123".to_string());

    let metadata = training_checkpoint.metadata();
    println!("   Training checkpoint summary:");
    println!("     Total parameters: {}", metadata.total_parameters);
    println!("     Model size: {} bytes", metadata.total_size_bytes);
    println!("     Epoch: {:?}", metadata.epoch);
    println!();

    // Example 7: Model inspection utilities
    println!("7. Model inspection and analysis");
    analyze_model_architecture(&checkpoint);

    // Example 8: Format recommendations
    println!("8. Format recommendations");
    println!("   For production use:");
    println!("     ✓ Safetensors: Recommended for security and performance");
    println!("       - No code execution during loading");
    println!("       - Fast memory-mapped access");
    println!("       - Cross-platform compatibility");
    println!("     ✗ Pickle: Only for legacy PyTorch checkpoints");
    println!("       - Security risks with untrusted files");
    println!("       - Python-specific format");
    println!();

    println!("=== Demo Complete ===");
    println!("\nKey Takeaways:");
    println!("  1. PyTorch checkpoints can be created and manipulated in Rust");
    println!("  2. Full metadata extraction and inspection is supported");
    println!("  3. Seamless conversion to Safetensors for modern workflows");
    println!("  4. Useful for model analysis, debugging, and format conversion");

    Ok(())
}

/// Create random weights with Xavier initialization
fn create_random_weights(shape: Vec<usize>, scale: f32) -> TensorData {
    use rand::RngExt;
    let mut rng = rand::rng();

    let num_elements: usize = shape.iter().product();
    let weights: Vec<f32> = (0..num_elements)
        .map(|_| (rng.random::<f32>() - 0.5) * 2.0 * scale)
        .collect();

    TensorData::from_f32(shape, &weights)
}

/// Create zero-initialized tensor
fn create_zeros(shape: Vec<usize>) -> TensorData {
    let num_elements: usize = shape.iter().product();
    let zeros = vec![0.0f32; num_elements];
    TensorData::from_f32(shape, &zeros)
}

/// Display checkpoint metadata in a formatted way
fn display_metadata(metadata: &CheckpointMetadata) {
    println!("   Checkpoint Metadata:");
    println!("     Total parameters: {}", metadata.total_parameters);
    println!("     Number of layers: {}", metadata.layer_names.len());
    println!(
        "     Total size: {} bytes ({:.2} MB)",
        metadata.total_size_bytes,
        metadata.total_size_bytes as f64 / 1024.0 / 1024.0
    );

    println!("     Data types:");
    for (dtype, count) in &metadata.dtypes {
        println!("       - {}: {} tensors", dtype, count);
    }

    if let Some(epoch) = metadata.epoch {
        println!("     Training epoch: {}", epoch);
    }

    println!();
}

/// Analyze model architecture
fn analyze_model_architecture(checkpoint: &PyTorchCheckpoint) {
    println!("   Model Architecture Analysis:");

    let state_dict = checkpoint.state_dict();

    // Count layer types
    let mut weight_count = 0;
    let mut bias_count = 0;
    let mut other_count = 0;

    for name in state_dict.iter().map(|(n, _)| n) {
        if name.contains("weight") {
            weight_count += 1;
        } else if name.contains("bias") {
            bias_count += 1;
        } else {
            other_count += 1;
        }
    }

    println!("     Weight tensors: {}", weight_count);
    println!("     Bias tensors: {}", bias_count);
    println!("     Other tensors: {}", other_count);

    // Estimate model depth
    let max_layer_num = state_dict
        .iter()
        .filter_map(|(name, _)| {
            // Try to extract layer number from names like "layer1.weight"
            name.split('.')
                .next()
                .and_then(|s| s.trim_start_matches("layer").parse::<usize>().ok())
        })
        .max()
        .unwrap_or(1);

    println!("     Estimated depth: {} layers", max_layer_num);

    // Calculate total parameters
    let metadata = checkpoint.metadata();
    println!(
        "     Total parameters: {} ({:.2}M)",
        metadata.total_parameters,
        metadata.total_parameters as f64 / 1_000_000.0
    );

    println!();
}
