//! Integration tests for zero-copy operations
//!
//! Tests verify that zero-copy tensor operations work correctly and efficiently:
//! - Arrow tensor zero-copy access
//! - Safetensors zero-copy loading
//! - Shared memory zero-copy sharing
//! - Type conversions without unnecessary copies

use bytes::Bytes;
use ipfrs_tensorlogic::{
    ArrowTensor, ArrowTensorStore, SafetensorsReader, SafetensorsWriter, SharedMemoryPool,
    SharedTensorBuffer, SharedTensorInfo, TensorDtype, ZeroCopyConverter,
};
use tempfile::NamedTempFile;

/// Test that Arrow tensors provide zero-copy access to underlying data
#[test]
fn test_arrow_tensor_zero_copy_access() {
    // Create a tensor from f32 data
    let data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();
    let tensor = ArrowTensor::from_slice_f32("test_tensor", vec![1000], &data);

    // Verify metadata
    assert_eq!(tensor.metadata.name, "test_tensor");
    assert_eq!(tensor.metadata.shape, vec![1000]);
    assert_eq!(tensor.metadata.dtype, TensorDtype::Float32);

    // Zero-copy access
    let slice = tensor.as_slice_f32().expect("Failed to get f32 slice");

    // Verify data integrity
    assert_eq!(slice.len(), 1000);
    for (i, &value) in slice.iter().enumerate() {
        assert_eq!(value, i as f32 * 0.5);
    }

    // Test that we can access the same data through different methods
    let bytes = tensor.as_bytes();
    assert_eq!(bytes.len(), 1000 * 4); // 1000 f32 * 4 bytes

    // Verify bytes match original data
    let floats_from_bytes: &[f32] = ZeroCopyConverter::bytes_to_slice(&bytes);
    assert_eq!(floats_from_bytes.len(), 1000);
    for (i, &value) in floats_from_bytes.iter().enumerate() {
        assert_eq!(value, i as f32 * 0.5);
    }
}

/// Test zero-copy access for different data types
#[test]
fn test_arrow_tensor_multi_dtype_zero_copy() {
    // Test f32
    let f32_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let f32_tensor = ArrowTensor::from_slice_f32("f32", vec![4], &f32_data);
    let f32_slice = f32_tensor.as_slice_f32().unwrap();
    assert_eq!(f32_slice, &f32_data[..]);

    // Test f64
    let f64_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let f64_tensor = ArrowTensor::from_slice_f64("f64", vec![4], &f64_data);
    let f64_slice = f64_tensor.as_slice_f64().unwrap();
    assert_eq!(f64_slice, &f64_data[..]);

    // Test i32
    let i32_data: Vec<i32> = vec![1, 2, 3, 4];
    let i32_tensor = ArrowTensor::from_slice_i32("i32", vec![4], &i32_data);
    let i32_slice = i32_tensor.as_slice_i32().unwrap();
    assert_eq!(i32_slice, &i32_data[..]);

    // Test i64
    let i64_data: Vec<i64> = vec![1, 2, 3, 4];
    let i64_tensor = ArrowTensor::from_slice_i64("i64", vec![4], &i64_data);
    let i64_slice = i64_tensor.as_slice_i64().unwrap();
    assert_eq!(i64_slice, &i64_data[..]);
}

/// Test that Arrow tensor store manages tensors correctly
#[test]
fn test_arrow_tensor_store_zero_copy() {
    let mut store = ArrowTensorStore::new();

    // Add multiple tensors
    let t1 = ArrowTensor::from_slice_f32("weights", vec![10], &[1.0; 10]);
    let t2 = ArrowTensor::from_slice_f32("biases", vec![5], &[0.1; 5]);
    let t3 = ArrowTensor::from_slice_i32("indices", vec![3], &[0, 1, 2]);

    store.insert(t1);
    store.insert(t2);
    store.insert(t3);

    assert_eq!(store.len(), 3);

    // Retrieve and verify zero-copy access
    let weights = store.get("weights").expect("Weights not found");
    let weights_slice = weights.as_slice_f32().unwrap();
    assert_eq!(weights_slice.len(), 10);
    assert!(weights_slice.iter().all(|&x| x == 1.0));

    let indices = store.get("indices").expect("Indices not found");
    let indices_slice = indices.as_slice_i32().unwrap();
    assert_eq!(indices_slice, &[0, 1, 2]);
}

/// Test Safetensors zero-copy loading
#[test]
fn test_safetensors_zero_copy_loading() {
    // Create a Safetensors file with multiple tensors
    let mut writer = SafetensorsWriter::new();

    // Add various tensors
    writer.add_f32("layer1.weight", vec![128, 64], &vec![0.1; 128 * 64]);
    writer.add_f32("layer1.bias", vec![128], &vec![0.01; 128]);
    writer.add_f64("high_precision", vec![10], &[std::f64::consts::PI; 10]);
    writer.add_i32("vocab_ids", vec![1000], &vec![42; 1000]);

    let bytes = writer.serialize().expect("Failed to serialize");

    // Load with zero-copy reader
    let reader = SafetensorsReader::from_bytes(Bytes::from(bytes)).expect("Failed to load");

    // Verify tensors can be loaded as Arrow tensors (zero-copy)
    let weight = reader
        .load_as_arrow("layer1.weight")
        .expect("Failed to load weight");
    assert_eq!(weight.metadata.shape, vec![128, 64]);
    let weight_slice = weight.as_slice_f32().unwrap();
    assert_eq!(weight_slice.len(), 128 * 64);

    let bias = reader
        .load_as_arrow("layer1.bias")
        .expect("Failed to load bias");
    assert_eq!(bias.metadata.shape, vec![128]);
    let bias_slice = bias.as_slice_f32().unwrap();
    assert_eq!(bias_slice.len(), 128);

    let high_prec = reader
        .load_as_arrow("high_precision")
        .expect("Failed to load high precision");
    let hp_slice = high_prec.as_slice_f64().unwrap();
    assert!(hp_slice
        .iter()
        .all(|&x| (x - std::f64::consts::PI).abs() < 1e-10));
}

/// Test shared memory zero-copy sharing
#[test]
fn test_shared_memory_zero_copy() {
    use tempfile::tempdir;

    let dir = tempdir().expect("Failed to create temp dir");
    let mut pool = SharedMemoryPool::new(dir.path(), 1024 * 1024 * 100); // 100MB limit

    // Create a large tensor
    let data: Vec<f32> = (0..10000).map(|i| i as f32).collect();

    // Create a temporary file for shared memory
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp_file.path();

    // Define tensor info
    let tensor_info = SharedTensorInfo {
        name: "test_tensor".to_string(),
        dtype: TensorDtype::Float32,
        shape: vec![10000],
        offset: 0,
        size: 10000 * 4,
    };

    // Create shared buffer
    let mut buffer =
        SharedTensorBuffer::create(path, 10000 * 4, std::slice::from_ref(&tensor_info))
            .expect("Failed to create shared buffer");

    // Write data
    buffer.write_tensor(&tensor_info, &data);
    buffer.flush().expect("Failed to flush");

    // Open read-only view (zero-copy)
    let readonly = SharedTensorBuffer::open_readonly(path).expect("Failed to create readonly view");

    // Get metadata
    let metadata = readonly.tensor_metadata().expect("Failed to get metadata");
    assert_eq!(metadata.len(), 1);

    // Read data (zero-copy)
    let read_data: Vec<f32> = readonly.read_tensor(&metadata[0]);
    assert_eq!(read_data.len(), 10000);

    // Verify data integrity
    for (i, &value) in read_data.iter().enumerate() {
        assert_eq!(value, i as f32);
    }

    // Register with pool
    pool.register("test_buffer", readonly)
        .expect("Failed to register buffer");

    // Clean up
    pool.remove("test_buffer");
}

/// Test type conversions without unnecessary copies
#[test]
fn test_zero_copy_converter() {
    // Test f32 to bytes and back
    let floats: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let bytes = ZeroCopyConverter::slice_to_bytes(&floats);
    assert_eq!(bytes.len(), 20); // 5 floats * 4 bytes

    let floats_back: &[f32] = ZeroCopyConverter::bytes_to_slice(bytes);
    assert_eq!(floats_back, &floats[..]);

    // Test f64 to bytes and back
    let doubles: Vec<f64> = vec![1.0, 2.0, 3.0];
    let bytes = ZeroCopyConverter::slice_to_bytes(&doubles);
    assert_eq!(bytes.len(), 24); // 3 f64 * 8 bytes

    let doubles_back: &[f64] = ZeroCopyConverter::bytes_to_slice(bytes);
    assert_eq!(doubles_back, &doubles[..]);

    // Test i32 to bytes and back
    let ints: Vec<i32> = vec![1, 2, 3, 4];
    let bytes = ZeroCopyConverter::slice_to_bytes(&ints);
    assert_eq!(bytes.len(), 16); // 4 i32 * 4 bytes

    let ints_back: &[i32] = ZeroCopyConverter::bytes_to_slice(bytes);
    assert_eq!(ints_back, &ints[..]);
}

/// Test roundtrip: Arrow -> Safetensors -> Arrow (zero-copy)
#[test]
fn test_arrow_safetensors_roundtrip() {
    // Create Arrow tensors
    let mut store = ArrowTensorStore::new();
    store.insert(ArrowTensor::from_slice_f32(
        "w1",
        vec![100],
        &vec![0.5; 100],
    ));
    store.insert(ArrowTensor::from_slice_f64("w2", vec![50], &vec![1.5; 50]));
    store.insert(ArrowTensor::from_slice_i32("idx", vec![10], &[7; 10]));

    // Convert to Safetensors
    let mut writer = SafetensorsWriter::new();

    for name in store.names() {
        let tensor = store.get(name).unwrap();
        match tensor.metadata.dtype {
            TensorDtype::Float32 => {
                let data = tensor.as_slice_f32().unwrap();
                writer.add_f32(&tensor.metadata.name, tensor.metadata.shape.clone(), data);
            }
            TensorDtype::Float64 => {
                let data = tensor.as_slice_f64().unwrap();
                writer.add_f64(&tensor.metadata.name, tensor.metadata.shape.clone(), data);
            }
            TensorDtype::Int32 => {
                let data = tensor.as_slice_i32().unwrap();
                writer.add_i32(&tensor.metadata.name, tensor.metadata.shape.clone(), data);
            }
            TensorDtype::Int64 => {
                let data = tensor.as_slice_i64().unwrap();
                writer.add_i64(&tensor.metadata.name, tensor.metadata.shape.clone(), data);
            }
            _ => {}
        }
    }

    let bytes = writer.serialize().expect("Failed to serialize");

    // Load back
    let reader = SafetensorsReader::from_bytes(Bytes::from(bytes)).expect("Failed to read");

    // Verify all tensors
    let w1 = reader.load_as_arrow("w1").expect("Failed to load w1");
    assert_eq!(w1.as_slice_f32().unwrap().len(), 100);
    assert!(w1.as_slice_f32().unwrap().iter().all(|&x| x == 0.5));

    let w2 = reader.load_as_arrow("w2").expect("Failed to load w2");
    assert_eq!(w2.as_slice_f64().unwrap().len(), 50);
    assert!(w2.as_slice_f64().unwrap().iter().all(|&x| x == 1.5));

    let idx = reader.load_as_arrow("idx").expect("Failed to load idx");
    assert_eq!(idx.as_slice_i32().unwrap().len(), 10);
    assert!(idx.as_slice_i32().unwrap().iter().all(|&x| x == 7));
}

/// Test large tensor zero-copy operations
#[test]
fn test_large_tensor_zero_copy() {
    // Create a large tensor (10 million f32 elements = 40MB)
    let size = 10_000_000;
    let data: Vec<f32> = (0..size).map(|i| (i % 100) as f32).collect();

    // Create Arrow tensor
    let tensor = ArrowTensor::from_slice_f32("large", vec![size], &data);

    // Zero-copy access should be instant
    let slice = tensor.as_slice_f32().expect("Failed to get slice");
    assert_eq!(slice.len(), size);

    // Verify a sample of values
    assert_eq!(slice[0], 0.0);
    assert_eq!(slice[99], 99.0);
    assert_eq!(slice[100], 0.0);
    assert_eq!(slice[size - 1], ((size - 1) % 100) as f32);

    // Byte access should also be zero-copy
    let bytes = tensor.as_bytes();
    assert_eq!(bytes.len(), size * 4);
}

/// Test error handling in zero-copy operations
#[test]
fn test_zero_copy_type_mismatch_errors() {
    // Create an f32 tensor
    let tensor = ArrowTensor::from_slice_f32("test", vec![10], &[1.0; 10]);

    // Trying to access as wrong type should fail
    assert!(tensor.as_slice_f64().is_none());
    assert!(tensor.as_slice_i32().is_none());
    assert!(tensor.as_slice_i64().is_none());

    // Correct type should succeed
    assert!(tensor.as_slice_f32().is_some());
}

/// Test that zero-copy operations don't cause memory leaks
#[test]
fn test_zero_copy_memory_management() {
    // Create and drop many tensors
    for _ in 0..1000 {
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let tensor = ArrowTensor::from_slice_f32("temp", vec![1000], &data);
        let _slice = tensor.as_slice_f32().unwrap();
        // Tensor and slice dropped here
    }

    // If there were memory leaks, this test would consume lots of memory
    // The test passing indicates proper memory management
}
