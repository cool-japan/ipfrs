//! Model Quantization Example
//!
//! This example demonstrates the comprehensive quantization support in ipfrs-tensorlogic,
//! showing how to quantize models for efficient deployment on edge devices.
//!
//! Features demonstrated:
//! - INT8 and INT4 quantization
//! - Per-tensor and per-channel quantization
//! - Symmetric and asymmetric quantization
//! - Dynamic quantization for activations
//! - Percentile-based calibration
//! - Quantization error analysis
//! - Model size reduction

use ipfrs_tensorlogic::{
    CalibrationMethod, DynamicQuantizer, QuantizationConfig, QuantizationGranularity,
    QuantizationScheme, QuantizedTensor,
};

fn main() {
    println!("=== Model Quantization Example ===\n");

    // Example 1: Per-tensor INT8 symmetric quantization
    per_tensor_int8_example();

    // Example 2: Per-channel quantization for Conv2D weights
    per_channel_quantization_example();

    // Example 3: INT4 extreme compression
    int4_compression_example();

    // Example 4: Asymmetric quantization for ReLU activations
    asymmetric_quantization_example();

    // Example 5: Dynamic quantization for activations
    dynamic_quantization_example();

    // Example 6: Percentile calibration for outlier handling
    percentile_calibration_example();

    // Example 7: Complete model quantization pipeline
    complete_model_example();
}

fn per_tensor_int8_example() {
    println!("--- Example 1: Per-Tensor INT8 Symmetric Quantization ---");

    // Simulate a fully connected layer weight matrix (128x64 = 8192 parameters)
    let weights: Vec<f32> = (0..8192)
        .map(|i| {
            // Simulate normal distribution: mean=0, std=0.1
            let x = (i as f32) / 8192.0;
            0.1 * (x - 0.5) * 2.0
        })
        .collect();

    println!(
        "Original weight shape: [128, 64], size: {} KB",
        weights.len() * 4 / 1024
    );

    // Create INT8 symmetric quantization config
    let config = QuantizationConfig::int8_symmetric();

    // Quantize
    let quantized = QuantizedTensor::quantize_per_tensor(&weights, vec![128, 64], config).unwrap();

    println!("Quantization params:");
    println!("  Scale: {:.6}", quantized.params[0].scale);
    println!("  Zero point: {}", quantized.params[0].zero_point);

    // Calculate compression ratio and error
    let compression_ratio = quantized.compression_ratio();
    let error = quantized.quantization_error(&weights);

    println!("Compression ratio: {:.2}x", compression_ratio);
    println!("Quantization error (MSE): {:.6}", error);
    println!("Quantized size: {} KB\n", quantized.size_bytes() / 1024);
}

fn per_channel_quantization_example() {
    println!("--- Example 2: Per-Channel Quantization for Conv2D ---");

    // Simulate Conv2D weights: [out_channels=64, in_channels=32, kernel_h=3, kernel_w=3]
    // Total: 64 * 32 * 3 * 3 = 18,432 parameters
    let out_channels = 64;
    let total_size = 64 * 32 * 3 * 3;

    let weights: Vec<f32> = (0..total_size)
        .map(|i| {
            let channel = i / (total_size / out_channels);
            // Each channel has different distribution
            let scale = 0.05 + (channel as f32) / 1000.0;
            let x = (i as f32) / (total_size as f32);
            scale * (x - 0.5) * 2.0
        })
        .collect();

    println!("Conv2D weights: [64, 32, 3, 3]");
    println!("Original size: {} KB", weights.len() * 4 / 1024);

    // Per-channel quantization (one scale/zero-point per output channel)
    let config = QuantizationConfig::int8_per_channel(out_channels);
    let quantized =
        QuantizedTensor::quantize_per_channel(&weights, vec![out_channels, 32 * 3 * 3], config)
            .unwrap();

    println!(
        "Quantization params per channel: {}",
        quantized.params.len()
    );
    println!(
        "Channel 0: scale={:.6}, zero_point={}",
        quantized.params[0].scale, quantized.params[0].zero_point
    );
    println!(
        "Channel 63: scale={:.6}, zero_point={}",
        quantized.params[63].scale, quantized.params[63].zero_point
    );

    let error = quantized.quantization_error(&weights);
    println!("Quantization error (MSE): {:.6}", error);
    println!("Compression ratio: {:.2}x\n", quantized.compression_ratio());
}

fn int4_compression_example() {
    println!("--- Example 3: INT4 Extreme Compression ---");

    // Simulate a large embedding matrix (10000 x 512)
    let vocab_size = 10000;
    let embedding_dim = 512;
    let total_size = vocab_size * embedding_dim;

    let embeddings: Vec<f32> = (0..total_size)
        .map(|i| {
            let x = (i as f32) / (total_size as f32);
            0.05 * (x - 0.5) * 2.0
        })
        .collect();

    println!("Embedding matrix: [{}, {}]", vocab_size, embedding_dim);
    println!(
        "Original size: {:.2} MB",
        embeddings.len() * 4 / 1024 / 1024
    );

    // INT4 quantization for extreme compression
    let config = QuantizationConfig::int4_symmetric();
    let quantized =
        QuantizedTensor::quantize_per_tensor(&embeddings, vec![vocab_size, embedding_dim], config)
            .unwrap();

    println!("Quantized with INT4");
    println!(
        "Quantized size: {:.2} MB",
        quantized.size_bytes() / 1024 / 1024
    );
    println!("Compression ratio: {:.2}x", quantized.compression_ratio());

    // Pack INT4 data (2 values per byte)
    let packed = quantized.pack_int4().unwrap();
    println!("Packed INT4 size: {} bytes", packed.len());

    let error = quantized.quantization_error(&embeddings);
    println!("Quantization error (MSE): {:.6}\n", error);
}

fn asymmetric_quantization_example() {
    println!("--- Example 4: Asymmetric Quantization for ReLU Activations ---");

    // ReLU activations are always non-negative, so asymmetric works better
    let activations: Vec<f32> = (0..1000)
        .map(|i| {
            let x = (i as f32) / 1000.0;
            (x * 10.0).max(0.0) // ReLU-like distribution
        })
        .collect();

    println!("ReLU activations (all non-negative)");
    println!(
        "Min: {:.2}",
        activations.iter().copied().fold(f32::INFINITY, f32::min)
    );
    println!(
        "Max: {:.2}",
        activations
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
    );

    // Compare symmetric vs asymmetric
    let symmetric_config = QuantizationConfig::int8_symmetric();
    let asymmetric_config = QuantizationConfig::int8_asymmetric();

    let symmetric =
        QuantizedTensor::quantize_per_tensor(&activations, vec![1000], symmetric_config).unwrap();
    let asymmetric =
        QuantizedTensor::quantize_per_tensor(&activations, vec![1000], asymmetric_config).unwrap();

    let symmetric_error = symmetric.quantization_error(&activations);
    let asymmetric_error = asymmetric.quantization_error(&activations);

    println!("\nSymmetric quantization:");
    println!("  Zero point: {}", symmetric.params[0].zero_point);
    println!("  Error (MSE): {:.6}", symmetric_error);

    println!("\nAsymmetric quantization:");
    println!("  Zero point: {}", asymmetric.params[0].zero_point);
    println!("  Error (MSE): {:.6}", asymmetric_error);
    println!(
        "  Improvement: {:.2}%\n",
        (symmetric_error - asymmetric_error) / symmetric_error * 100.0
    );
}

fn dynamic_quantization_example() {
    println!("--- Example 5: Dynamic Quantization for Activations ---");

    // Dynamic quantization quantizes at runtime
    let quantizer = DynamicQuantizer::new(QuantizationScheme::Int8, true);

    // Simulate different activation batches
    let batch1: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();
    let batch2: Vec<f32> = (0..256).map(|i| (i as f32) / 512.0).collect();

    let q1 = quantizer.quantize_activation(&batch1, vec![256]).unwrap();
    let q2 = quantizer.quantize_activation(&batch2, vec![256]).unwrap();

    println!("Batch 1 quantization:");
    println!("  Scale: {:.6}", q1.params[0].scale);
    println!("  Zero point: {}", q1.params[0].zero_point);

    println!("\nBatch 2 quantization:");
    println!("  Scale: {:.6}", q2.params[0].scale);
    println!("  Zero point: {}", q2.params[0].zero_point);

    println!("\nNote: Different batches get different quantization params\n");
}

fn percentile_calibration_example() {
    println!("--- Example 6: Percentile Calibration for Outlier Handling ---");

    // Create data with outliers
    let mut data = vec![0.0f32; 1000];
    for (i, val) in data.iter_mut().enumerate() {
        if !(10..990).contains(&i) {
            // Outliers
            *val = if i < 10 { -100.0 } else { 100.0 };
        } else {
            // Normal data: -1 to 1
            *val = ((i as f32) - 500.0) / 500.0;
        }
    }

    println!("Data with outliers:");
    println!("  Total values: {}", data.len());
    println!("  Outliers: 20 (10 at each end)");

    // Min-max calibration (affected by outliers)
    let minmax_config = QuantizationConfig {
        scheme: QuantizationScheme::Int8,
        granularity: QuantizationGranularity::PerTensor,
        symmetric: true,
        calibration: CalibrationMethod::MinMax,
    };

    // Percentile calibration (clips outliers)
    let percentile_config = QuantizationConfig {
        scheme: QuantizationScheme::Int8,
        granularity: QuantizationGranularity::PerTensor,
        symmetric: true,
        calibration: CalibrationMethod::Percentile {
            lower: 1,
            upper: 99,
        },
    };

    let minmax_q = QuantizedTensor::quantize_per_tensor(&data, vec![1000], minmax_config).unwrap();
    let percentile_q =
        QuantizedTensor::quantize_per_tensor(&data, vec![1000], percentile_config).unwrap();

    println!("\nMin-max calibration:");
    println!("  Scale: {:.6}", minmax_q.params[0].scale);

    println!("\nPercentile calibration (1-99%):");
    println!("  Scale: {:.6}", percentile_q.params[0].scale);
    println!(
        "  Scale reduction: {:.2}x (better precision for non-outliers)\n",
        minmax_q.params[0].scale / percentile_q.params[0].scale
    );
}

fn complete_model_example() {
    println!("--- Example 7: Complete Model Quantization Pipeline ---");

    // Simulate a small neural network
    struct Layer {
        name: String,
        weights: Vec<f32>,
        shape: Vec<usize>,
    }

    let layers = vec![
        Layer {
            name: "fc1".to_string(),
            weights: vec![0.1; 784 * 128], // 784 -> 128
            shape: vec![128, 784],
        },
        Layer {
            name: "fc2".to_string(),
            weights: vec![0.05; 128 * 64], // 128 -> 64
            shape: vec![64, 128],
        },
        Layer {
            name: "fc3".to_string(),
            weights: vec![0.02; 64 * 10], // 64 -> 10
            shape: vec![10, 64],
        },
    ];

    println!("Neural Network:");
    let total_params: usize = layers.iter().map(|l| l.weights.len()).sum();
    println!("  Layers: {}", layers.len());
    println!("  Total parameters: {}", total_params);
    println!("  Original size: {} KB\n", total_params * 4 / 1024);

    // Quantize all layers
    let mut quantized_layers = Vec::new();
    let mut total_quantized_size = 0;

    for layer in &layers {
        // Use per-channel quantization for fully connected layers
        let num_channels = layer.shape[0];
        let config = QuantizationConfig::int8_per_channel(num_channels);

        let quantized =
            QuantizedTensor::quantize_per_channel(&layer.weights, layer.shape.clone(), config)
                .unwrap();

        let error = quantized.quantization_error(&layer.weights);

        println!("Layer: {}", layer.name);
        println!("  Shape: {:?}", layer.shape);
        println!("  Params: {}", layer.weights.len());
        println!("  Quantization error: {:.6}", error);
        println!("  Size: {} bytes\n", quantized.size_bytes());

        total_quantized_size += quantized.size_bytes();
        quantized_layers.push(quantized);
    }

    let original_size = total_params * 4;
    let compression_ratio = original_size as f32 / total_quantized_size as f32;

    println!("Model Summary:");
    println!("  Original size: {} KB", original_size / 1024);
    println!("  Quantized size: {} KB", total_quantized_size / 1024);
    println!("  Compression ratio: {:.2}x", compression_ratio);
    println!(
        "  Size reduction: {:.1}%",
        (1.0 - total_quantized_size as f32 / original_size as f32) * 100.0
    );
}
