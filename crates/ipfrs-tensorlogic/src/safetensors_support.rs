//! Safetensors file format support
//!
//! Provides parsing and writing of the safetensors format for:
//! - Native safetensors reading with multi-dtype support (f32, f64, i32, i64)
//! - Chunked storage for large models
//! - Lazy loading with memory mapping
//! - Metadata extraction
//! - Zero-copy Arrow tensor conversion
//!
//! ## Supported Data Types
//!
//! The reader and writer support the following data types:
//! - **Float32** (f32) - Standard precision floating point
//! - **Float64** (f64) - Double precision floating point
//! - **Int32** (i32) - 32-bit signed integers
//! - **Int64** (i64) - 64-bit signed integers
//!
//! All supported types can be loaded as Arrow tensors for zero-copy access.

use crate::arrow::{ArrowTensor, ArrowTensorStore, TensorDtype};
use bytes::Bytes;
use memmap2::Mmap;
use safetensors::tensor::{SafeTensorError, SafeTensors};
use safetensors::{Dtype, View};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Safetensors file reader with lazy loading support
pub struct SafetensorsReader {
    /// Memory-mapped file (for lazy loading)
    mmap: Option<Mmap>,
    /// Raw bytes (for in-memory loading)
    bytes: Option<Bytes>,
    /// Parsed tensor metadata
    metadata: HashMap<String, TensorInfo>,
    /// Global metadata from the file
    global_metadata: HashMap<String, String>,
}

/// Information about a tensor in a safetensors file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Tensor name
    pub name: String,
    /// Data type
    pub dtype: TensorDtype,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Byte offset in the file
    pub data_offset: usize,
    /// Size in bytes
    pub data_size: usize,
}

impl SafetensorsReader {
    /// Open a safetensors file with memory mapping (lazy loading)
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, SafetensorError> {
        let file = File::open(path.as_ref()).map_err(SafetensorError::Io)?;
        let mmap = unsafe { Mmap::map(&file).map_err(SafetensorError::Io)? };

        Self::from_mmap(mmap)
    }

    /// Create from memory-mapped data
    fn from_mmap(mmap: Mmap) -> Result<Self, SafetensorError> {
        // Parse header to get metadata
        let tensors = SafeTensors::deserialize(&mmap)?;

        let mut metadata = HashMap::new();
        let global_metadata = HashMap::new();

        // Extract tensor info
        for (name, view) in tensors.tensors() {
            let dtype = convert_safetensor_dtype(view.dtype());
            let shape = view.shape().to_vec();
            let data = view.data();

            let info = TensorInfo {
                name: name.clone(),
                dtype,
                shape,
                data_offset: data.as_ptr() as usize - mmap.as_ptr() as usize,
                data_size: data.len(),
            };
            metadata.insert(name, info);
        }

        Ok(Self {
            mmap: Some(mmap),
            bytes: None,
            metadata,
            global_metadata,
        })
    }

    /// Load from bytes
    pub fn from_bytes(bytes: Bytes) -> Result<Self, SafetensorError> {
        let tensors = SafeTensors::deserialize(&bytes)?;

        let mut metadata = HashMap::new();
        let global_metadata = HashMap::new();

        for (name, view) in tensors.tensors() {
            let dtype = convert_safetensor_dtype(view.dtype());
            let shape = view.shape().to_vec();
            let data = view.data();

            let info = TensorInfo {
                name: name.clone(),
                dtype,
                shape,
                data_offset: data.as_ptr() as usize - bytes.as_ptr() as usize,
                data_size: data.len(),
            };
            metadata.insert(name, info);
        }

        Ok(Self {
            mmap: None,
            bytes: Some(bytes),
            metadata,
            global_metadata,
        })
    }

    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.metadata.keys().map(|s| s.as_str()).collect()
    }

    /// Get tensor info by name
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.metadata.get(name)
    }

    /// Get global metadata
    pub fn global_metadata(&self) -> &HashMap<String, String> {
        &self.global_metadata
    }

    /// Get the number of tensors
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }

    /// Get raw data for a tensor (zero-copy)
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.metadata.get(name)?;
        let data = self.get_data()?;
        Some(&data[info.data_offset..info.data_offset + info.data_size])
    }

    /// Get the underlying data slice
    fn get_data(&self) -> Option<&[u8]> {
        if let Some(ref mmap) = self.mmap {
            Some(mmap.as_ref())
        } else if let Some(ref bytes) = self.bytes {
            Some(bytes.as_ref())
        } else {
            None
        }
    }

    /// Load a tensor as f32 slice
    pub fn load_f32(&self, name: &str) -> Option<Vec<f32>> {
        let info = self.tensor_info(name)?;
        if info.dtype != TensorDtype::Float32 {
            return None;
        }

        let data = self.tensor_data(name)?;
        let f32_data: Vec<f32> = data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Some(f32_data)
    }

    /// Load a tensor as f64 slice
    pub fn load_f64(&self, name: &str) -> Option<Vec<f64>> {
        let info = self.tensor_info(name)?;
        if info.dtype != TensorDtype::Float64 {
            return None;
        }

        let data = self.tensor_data(name)?;
        let f64_data: Vec<f64> = data
            .chunks_exact(8)
            .map(|chunk| {
                f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect();
        Some(f64_data)
    }

    /// Load a tensor as i32 slice
    pub fn load_i32(&self, name: &str) -> Option<Vec<i32>> {
        let info = self.tensor_info(name)?;
        if info.dtype != TensorDtype::Int32 {
            return None;
        }

        let data = self.tensor_data(name)?;
        let i32_data: Vec<i32> = data
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Some(i32_data)
    }

    /// Load a tensor as i64 slice
    pub fn load_i64(&self, name: &str) -> Option<Vec<i64>> {
        let info = self.tensor_info(name)?;
        if info.dtype != TensorDtype::Int64 {
            return None;
        }

        let data = self.tensor_data(name)?;
        let i64_data: Vec<i64> = data
            .chunks_exact(8)
            .map(|chunk| {
                i64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect();
        Some(i64_data)
    }

    /// Load a tensor as ArrowTensor
    pub fn load_as_arrow(&self, name: &str) -> Option<ArrowTensor> {
        let info = self.tensor_info(name)?;

        match info.dtype {
            TensorDtype::Float32 => {
                let data = self.load_f32(name)?;
                Some(ArrowTensor::from_slice_f32(name, info.shape.clone(), &data))
            }
            TensorDtype::Float64 => {
                let data = self.load_f64(name)?;
                Some(ArrowTensor::from_slice_f64(name, info.shape.clone(), &data))
            }
            TensorDtype::Int32 => {
                let data = self.load_i32(name)?;
                Some(ArrowTensor::from_slice_i32(name, info.shape.clone(), &data))
            }
            TensorDtype::Int64 => {
                let data = self.load_i64(name)?;
                Some(ArrowTensor::from_slice_i64(name, info.shape.clone(), &data))
            }
            _ => None, // Other dtypes not yet supported in ArrowTensor
        }
    }

    /// Load all tensors into an ArrowTensorStore
    pub fn load_all_as_arrow(&self) -> ArrowTensorStore {
        let mut store = ArrowTensorStore::new();

        for name in self.tensor_names() {
            if let Some(tensor) = self.load_as_arrow(name) {
                store.insert(tensor);
            }
        }

        store
    }

    /// Get total size of all tensors
    pub fn total_size_bytes(&self) -> usize {
        self.metadata.values().map(|info| info.data_size).sum()
    }

    /// Get a summary of the model
    pub fn summary(&self) -> ModelSummary {
        let mut dtype_counts: HashMap<TensorDtype, usize> = HashMap::new();
        let mut total_params = 0usize;
        let mut total_bytes = 0usize;

        for info in self.metadata.values() {
            *dtype_counts.entry(info.dtype).or_insert(0) += 1;
            let numel: usize = info.shape.iter().product();
            total_params += numel;
            total_bytes += info.data_size;
        }

        ModelSummary {
            num_tensors: self.metadata.len(),
            total_params,
            total_bytes,
            dtype_distribution: dtype_counts,
            metadata: self.global_metadata.clone(),
        }
    }
}

/// Summary of a model's structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSummary {
    /// Number of tensors
    pub num_tensors: usize,
    /// Total number of parameters
    pub total_params: usize,
    /// Total size in bytes
    pub total_bytes: usize,
    /// Distribution of data types
    pub dtype_distribution: HashMap<TensorDtype, usize>,
    /// Global metadata
    pub metadata: HashMap<String, String>,
}

/// Safetensors file writer
pub struct SafetensorsWriter {
    /// Tensors to write
    tensors: Vec<(String, TensorData)>,
    /// Global metadata
    metadata: HashMap<String, String>,
}

/// Tensor data for writing
struct TensorData {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

/// Reference wrapper for TensorData that implements View
struct TensorDataRef<'a>(&'a TensorData);

impl View for TensorDataRef<'_> {
    fn dtype(&self) -> Dtype {
        self.0.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        std::borrow::Cow::Borrowed(&self.0.data)
    }

    fn data_len(&self) -> usize {
        self.0.data.len()
    }
}

impl SafetensorsWriter {
    /// Create a new writer
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add global metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Add a f32 tensor
    pub fn add_f32(&mut self, name: &str, shape: Vec<usize>, data: &[f32]) {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.tensors.push((
            name.to_string(),
            TensorData {
                dtype: Dtype::F32,
                shape,
                data: bytes,
            },
        ));
    }

    /// Add a f64 tensor
    pub fn add_f64(&mut self, name: &str, shape: Vec<usize>, data: &[f64]) {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.tensors.push((
            name.to_string(),
            TensorData {
                dtype: Dtype::F64,
                shape,
                data: bytes,
            },
        ));
    }

    /// Add an i32 tensor
    pub fn add_i32(&mut self, name: &str, shape: Vec<usize>, data: &[i32]) {
        let bytes: Vec<u8> = data.iter().flat_map(|i| i.to_le_bytes()).collect();
        self.tensors.push((
            name.to_string(),
            TensorData {
                dtype: Dtype::I32,
                shape,
                data: bytes,
            },
        ));
    }

    /// Add an i64 tensor
    pub fn add_i64(&mut self, name: &str, shape: Vec<usize>, data: &[i64]) {
        let bytes: Vec<u8> = data.iter().flat_map(|i| i.to_le_bytes()).collect();
        self.tensors.push((
            name.to_string(),
            TensorData {
                dtype: Dtype::I64,
                shape,
                data: bytes,
            },
        ));
    }

    /// Add an ArrowTensor
    pub fn add_arrow_tensor(&mut self, tensor: &ArrowTensor) {
        match tensor.metadata.dtype {
            TensorDtype::Float32 => {
                if let Some(data) = tensor.as_slice_f32() {
                    self.add_f32(&tensor.metadata.name, tensor.metadata.shape.clone(), data);
                }
            }
            TensorDtype::Float64 => {
                if let Some(data) = tensor.as_slice_f64() {
                    self.add_f64(&tensor.metadata.name, tensor.metadata.shape.clone(), data);
                }
            }
            TensorDtype::Int32 => {
                if let Some(data) = tensor.as_slice_i32() {
                    self.add_i32(&tensor.metadata.name, tensor.metadata.shape.clone(), data);
                }
            }
            TensorDtype::Int64 => {
                if let Some(data) = tensor.as_slice_i64() {
                    self.add_i64(&tensor.metadata.name, tensor.metadata.shape.clone(), data);
                }
            }
            _ => {} // Other dtypes not yet supported
        }
    }

    /// Write to a file
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), SafetensorError> {
        let bytes = self.serialize()?;
        let mut file = File::create(path).map_err(SafetensorError::Io)?;
        file.write_all(&bytes).map_err(SafetensorError::Io)?;
        Ok(())
    }

    /// Serialize to bytes
    pub fn serialize(&self) -> Result<Vec<u8>, SafetensorError> {
        let tensors: Vec<(&str, TensorDataRef)> = self
            .tensors
            .iter()
            .map(|(name, data)| (name.as_str(), TensorDataRef(data)))
            .collect();

        let metadata = if self.metadata.is_empty() {
            None
        } else {
            let meta: HashMap<String, String> = self.metadata.clone();
            Some(meta)
        };

        Ok(safetensors::tensor::serialize(tensors, metadata)?)
    }
}

impl Default for SafetensorsWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Chunked model storage for large models
pub struct ChunkedModelStorage {
    /// Base path for chunks
    base_path: std::path::PathBuf,
    /// Chunk size in bytes
    chunk_size: usize,
    /// Chunk index
    chunks: Vec<ChunkInfo>,
}

/// Information about a model chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkInfo {
    /// Chunk index
    pub index: usize,
    /// Path to chunk file
    pub path: String,
    /// Tensors in this chunk
    pub tensors: Vec<String>,
    /// Size in bytes
    pub size_bytes: usize,
}

impl ChunkedModelStorage {
    /// Create a new chunked storage
    pub fn new<P: AsRef<Path>>(base_path: P, chunk_size: usize) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
            chunk_size,
            chunks: Vec::new(),
        }
    }

    /// Write a model in chunks
    #[allow(clippy::too_many_arguments)]
    pub fn write_chunked(&mut self, store: &ArrowTensorStore) -> Result<(), SafetensorError> {
        let mut current_chunk = SafetensorsWriter::new();
        let mut current_size = 0usize;
        let mut current_tensors = Vec::new();

        for name in store.names() {
            if let Some(tensor) = store.get(name) {
                let tensor_size = tensor.metadata.size_bytes();

                // Start new chunk if current would exceed limit
                if current_size + tensor_size > self.chunk_size && !current_tensors.is_empty() {
                    self.write_chunk(current_chunk, &current_tensors, current_size)?;
                    current_chunk = SafetensorsWriter::new();
                    current_tensors = Vec::new();
                    current_size = 0;
                }

                current_chunk.add_arrow_tensor(tensor);
                current_tensors.push(name.to_string());
                current_size += tensor_size;
            }
        }

        // Write final chunk
        if !current_tensors.is_empty() {
            self.write_chunk(current_chunk, &current_tensors, current_size)?;
        }

        Ok(())
    }

    fn write_chunk(
        &mut self,
        writer: SafetensorsWriter,
        tensors: &[String],
        size: usize,
    ) -> Result<(), SafetensorError> {
        let index = self.chunks.len();
        let filename = format!("chunk_{:04}.safetensors", index);
        let path = self.base_path.join(&filename);

        writer.write_to_file(&path)?;

        self.chunks.push(ChunkInfo {
            index,
            path: filename,
            tensors: tensors.to_vec(),
            size_bytes: size,
        });

        Ok(())
    }

    /// Write chunk index
    pub fn write_index(&self) -> Result<(), std::io::Error> {
        let index_path = self.base_path.join("model_index.json");
        let json = serde_json::to_string_pretty(&self.chunks)?;
        std::fs::write(index_path, json)?;
        Ok(())
    }

    /// Load chunk index
    pub fn load_index<P: AsRef<Path>>(path: P) -> Result<Vec<ChunkInfo>, std::io::Error> {
        let index_path = path.as_ref().join("model_index.json");
        let content = std::fs::read_to_string(index_path)?;
        let chunks: Vec<ChunkInfo> = serde_json::from_str(&content)?;
        Ok(chunks)
    }

    /// Get chunk containing a specific tensor
    pub fn find_tensor_chunk(&self, tensor_name: &str) -> Option<&ChunkInfo> {
        self.chunks
            .iter()
            .find(|chunk| chunk.tensors.contains(&tensor_name.to_string()))
    }
}

/// Custom error type for safetensor operations
#[derive(Debug)]
pub enum SafetensorError {
    /// IO error
    Io(std::io::Error),
    /// Parse error
    Parse(String),
    /// Safetensors library error
    Safetensors(SafeTensorError),
}

impl From<SafeTensorError> for SafetensorError {
    fn from(err: SafeTensorError) -> Self {
        SafetensorError::Safetensors(err)
    }
}

impl std::fmt::Display for SafetensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SafetensorError::Io(e) => write!(f, "IO error: {}", e),
            SafetensorError::Parse(s) => write!(f, "Parse error: {}", s),
            SafetensorError::Safetensors(e) => write!(f, "Safetensors error: {:?}", e),
        }
    }
}

impl std::error::Error for SafetensorError {}

/// Convert safetensors dtype to our dtype
fn convert_safetensor_dtype(dtype: Dtype) -> TensorDtype {
    match dtype {
        Dtype::F32 => TensorDtype::Float32,
        Dtype::F64 => TensorDtype::Float64,
        Dtype::I8 => TensorDtype::Int8,
        Dtype::I16 => TensorDtype::Int16,
        Dtype::I32 => TensorDtype::Int32,
        Dtype::I64 => TensorDtype::Int64,
        Dtype::U8 => TensorDtype::UInt8,
        Dtype::U16 => TensorDtype::UInt16,
        Dtype::U32 => TensorDtype::UInt32,
        Dtype::U64 => TensorDtype::UInt64,
        Dtype::BF16 => TensorDtype::BFloat16,
        Dtype::F16 => TensorDtype::Float16,
        _ => TensorDtype::Float32, // Default fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_writer_and_reader() {
        // Create a safetensors file
        let mut writer =
            SafetensorsWriter::new().with_metadata("format".to_string(), "test".to_string());

        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        writer.add_f32("test_tensor", vec![3, 4], &data);

        // Write to temp file
        let mut temp_file = NamedTempFile::new().expect("test: should succeed");
        let bytes = writer.serialize().expect("test: should succeed");
        temp_file.write_all(&bytes).expect("test: should succeed");
        temp_file.flush().expect("test: should succeed");

        // Read back
        let reader = SafetensorsReader::open(temp_file.path()).expect("test: should succeed");

        assert_eq!(reader.len(), 1);
        assert!(reader.tensor_info("test_tensor").is_some());

        let info = reader
            .tensor_info("test_tensor")
            .expect("test: should succeed");
        assert_eq!(info.shape, vec![3, 4]);
        assert_eq!(info.dtype, TensorDtype::Float32);

        let loaded = reader
            .load_f32("test_tensor")
            .expect("test: should succeed");
        assert_eq!(loaded, data);
    }

    #[test]
    fn test_model_summary() {
        let mut writer = SafetensorsWriter::new();
        writer.add_f32("layer1", vec![10, 10], &[0.0; 100]);
        writer.add_f32("layer2", vec![10, 5], &[0.0; 50]);

        let bytes = writer.serialize().expect("test: should succeed");
        let reader =
            SafetensorsReader::from_bytes(Bytes::from(bytes)).expect("test: should succeed");

        let summary = reader.summary();
        assert_eq!(summary.num_tensors, 2);
        assert_eq!(summary.total_params, 150);
    }

    #[test]
    fn test_arrow_conversion() {
        let mut writer = SafetensorsWriter::new();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        writer.add_f32("weights", vec![2, 3], &data);

        let bytes = writer.serialize().expect("test: should succeed");
        let reader =
            SafetensorsReader::from_bytes(Bytes::from(bytes)).expect("test: should succeed");

        let tensor = reader
            .load_as_arrow("weights")
            .expect("test: should succeed");
        assert_eq!(tensor.metadata.name, "weights");
        assert_eq!(tensor.metadata.shape, vec![2, 3]);
        assert_eq!(tensor.as_slice_f32().expect("test: should succeed"), &data);
    }

    #[test]
    fn test_f64_support() {
        let mut writer = SafetensorsWriter::new();
        let data: Vec<f64> = vec![1.5, 2.5, 3.5, 4.5];
        writer.add_f64("weights_f64", vec![2, 2], &data);

        let bytes = writer.serialize().expect("test: should succeed");
        let reader =
            SafetensorsReader::from_bytes(Bytes::from(bytes)).expect("test: should succeed");

        // Test load_f64
        let loaded = reader
            .load_f64("weights_f64")
            .expect("test: should succeed");
        assert_eq!(loaded, data);

        // Test load_as_arrow
        let tensor = reader
            .load_as_arrow("weights_f64")
            .expect("test: should succeed");
        assert_eq!(tensor.metadata.name, "weights_f64");
        assert_eq!(tensor.metadata.dtype, TensorDtype::Float64);
        assert_eq!(tensor.as_slice_f64().expect("test: should succeed"), &data);
    }

    #[test]
    fn test_i32_support() {
        let mut writer = SafetensorsWriter::new();
        let data: Vec<i32> = vec![-10, 20, -30, 40, 50, -60];
        writer.add_i32("indices", vec![2, 3], &data);

        let bytes = writer.serialize().expect("test: should succeed");
        let reader =
            SafetensorsReader::from_bytes(Bytes::from(bytes)).expect("test: should succeed");

        // Test load_i32
        let loaded = reader.load_i32("indices").expect("test: should succeed");
        assert_eq!(loaded, data);

        // Test load_as_arrow
        let tensor = reader
            .load_as_arrow("indices")
            .expect("test: should succeed");
        assert_eq!(tensor.metadata.name, "indices");
        assert_eq!(tensor.metadata.dtype, TensorDtype::Int32);
        assert_eq!(tensor.as_slice_i32().expect("test: should succeed"), &data);
    }

    #[test]
    fn test_i64_support() {
        let mut writer = SafetensorsWriter::new();
        let data: Vec<i64> = vec![-1000000000, 2000000000, -3000000000, 4000000000];
        writer.add_i64("large_indices", vec![2, 2], &data);

        let bytes = writer.serialize().expect("test: should succeed");
        let reader =
            SafetensorsReader::from_bytes(Bytes::from(bytes)).expect("test: should succeed");

        // Test load_i64
        let loaded = reader
            .load_i64("large_indices")
            .expect("test: should succeed");
        assert_eq!(loaded, data);

        // Test load_as_arrow
        let tensor = reader
            .load_as_arrow("large_indices")
            .expect("test: should succeed");
        assert_eq!(tensor.metadata.name, "large_indices");
        assert_eq!(tensor.metadata.dtype, TensorDtype::Int64);
        assert_eq!(tensor.as_slice_i64().expect("test: should succeed"), &data);
    }

    #[test]
    fn test_mixed_dtypes() {
        let mut writer = SafetensorsWriter::new();

        let f32_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let f64_data: Vec<f64> = vec![5.5, 6.5];
        let i32_data: Vec<i32> = vec![10, 20, 30];
        let i64_data: Vec<i64> = vec![100, 200];

        writer.add_f32("layer1", vec![4], &f32_data);
        writer.add_f64("layer2", vec![2], &f64_data);
        writer.add_i32("layer3", vec![3], &i32_data);
        writer.add_i64("layer4", vec![2], &i64_data);

        let bytes = writer.serialize().expect("test: should succeed");
        let reader =
            SafetensorsReader::from_bytes(Bytes::from(bytes)).expect("test: should succeed");

        assert_eq!(reader.len(), 4);

        // Verify all tensors can be loaded correctly
        assert_eq!(
            reader.load_f32("layer1").expect("test: should succeed"),
            f32_data
        );
        assert_eq!(
            reader.load_f64("layer2").expect("test: should succeed"),
            f64_data
        );
        assert_eq!(
            reader.load_i32("layer3").expect("test: should succeed"),
            i32_data
        );
        assert_eq!(
            reader.load_i64("layer4").expect("test: should succeed"),
            i64_data
        );

        // Verify all can be loaded as arrow
        assert!(reader.load_as_arrow("layer1").is_some());
        assert!(reader.load_as_arrow("layer2").is_some());
        assert!(reader.load_as_arrow("layer3").is_some());
        assert!(reader.load_as_arrow("layer4").is_some());
    }

    #[test]
    fn test_arrow_tensor_roundtrip() {
        use crate::arrow::ArrowTensor;

        // Test f64
        let f64_tensor = ArrowTensor::from_slice_f64("test_f64", vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let mut writer = SafetensorsWriter::new();
        writer.add_arrow_tensor(&f64_tensor);

        let bytes = writer.serialize().expect("test: should succeed");
        let reader =
            SafetensorsReader::from_bytes(Bytes::from(bytes)).expect("test: should succeed");
        let loaded = reader
            .load_as_arrow("test_f64")
            .expect("test: should succeed");
        assert_eq!(
            loaded.as_slice_f64().expect("test: should succeed"),
            f64_tensor.as_slice_f64().expect("test: should succeed")
        );

        // Test i32
        let i32_tensor = ArrowTensor::from_slice_i32("test_i32", vec![3], &[10, 20, 30]);
        let mut writer = SafetensorsWriter::new();
        writer.add_arrow_tensor(&i32_tensor);

        let bytes = writer.serialize().expect("test: should succeed");
        let reader =
            SafetensorsReader::from_bytes(Bytes::from(bytes)).expect("test: should succeed");
        let loaded = reader
            .load_as_arrow("test_i32")
            .expect("test: should succeed");
        assert_eq!(
            loaded.as_slice_i32().expect("test: should succeed"),
            i32_tensor.as_slice_i32().expect("test: should succeed")
        );

        // Test i64
        let i64_tensor = ArrowTensor::from_slice_i64("test_i64", vec![2], &[100, 200]);
        let mut writer = SafetensorsWriter::new();
        writer.add_arrow_tensor(&i64_tensor);

        let bytes = writer.serialize().expect("test: should succeed");
        let reader =
            SafetensorsReader::from_bytes(Bytes::from(bytes)).expect("test: should succeed");
        let loaded = reader
            .load_as_arrow("test_i64")
            .expect("test: should succeed");
        assert_eq!(
            loaded.as_slice_i64().expect("test: should succeed"),
            i64_tensor.as_slice_i64().expect("test: should succeed")
        );
    }
}
