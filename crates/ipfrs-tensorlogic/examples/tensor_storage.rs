//! Example: Basic tensor storage and retrieval with Arrow
//!
//! This example demonstrates:
//! - Creating tensors with Arrow format
//! - Zero-copy access to tensor data
//! - IPC serialization and deserialization
//! - Tensor metadata handling

use ipfrs_tensorlogic::{ArrowTensor, ArrowTensorStore};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tensor Storage Example ===\n");

    // Create sample tensor data
    let data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1).collect();
    println!("Created tensor with {} elements", data.len());

    // Create an Arrow tensor (zero-copy)
    let tensor = ArrowTensor::from_slice_f32("model_weights", vec![10, 10, 10], &data);
    println!("Tensor shape: {:?}", tensor.metadata.shape);
    println!("Tensor dtype: {:?}", tensor.metadata.dtype);

    // Zero-copy access to the data
    let slice = tensor.as_slice_f32().expect("Failed to get f32 slice");
    println!("First 5 elements: {:?}", &slice[..5]);
    println!("Sum of all elements: {}", slice.iter().sum::<f32>());

    // Create a tensor store and add the tensor
    let mut store = ArrowTensorStore::new();
    store.insert(tensor);
    println!("\nTensor store size: {}", store.len());

    // Serialize to bytes (IPC format)
    let serialized = store.to_bytes()?;
    println!("Serialized size: {} bytes", serialized.len());

    // Deserialize from bytes
    let deserialized_store = ArrowTensorStore::from_bytes(&serialized)?;
    println!("Deserialized store size: {}", deserialized_store.len());

    // Verify the tensor
    if let Some(restored_tensor) = deserialized_store.get("model_weights") {
        let restored_slice = restored_tensor
            .as_slice_f32()
            .expect("Failed to get restored slice");
        println!("Restored first 5 elements: {:?}", &restored_slice[..5]);

        // Verify data integrity
        let original_data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1).collect();
        let restored_data: Vec<f32> = restored_slice.to_vec();
        println!("Data integrity check: {}", original_data == restored_data);
    }

    // Demonstrate different data types
    println!("\n=== Different Data Types ===");

    let i32_data: Vec<i32> = (0..100).collect();
    let i32_tensor = ArrowTensor::from_slice_i32("integers", vec![100], &i32_data);
    println!(
        "i32 tensor created with shape {:?}",
        i32_tensor.metadata.shape
    );

    let f64_data: Vec<f64> = (0..100).map(|i| i as f64 / 10.0).collect();
    let f64_tensor = ArrowTensor::from_slice_f64("doubles", vec![10, 10], &f64_data);
    println!(
        "f64 tensor created with shape {:?}",
        f64_tensor.metadata.shape
    );

    println!("\n✓ Example completed successfully!");
    Ok(())
}
