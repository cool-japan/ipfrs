//! Apache Arrow integration for zero-copy tensor transport
//!
//! Provides Arrow memory layout for tensors, enabling:
//! - Zero-copy data access
//! - Efficient columnar data formats
//! - Interoperability with Arrow ecosystem

use arrow::array::{
    ArrayRef, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
    UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Seek, Write};
use std::sync::Arc;

/// Tensor data type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorDtype {
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    BFloat16,
    Float16,
}

impl TensorDtype {
    /// Get the size in bytes of a single element
    #[inline]
    pub fn element_size(&self) -> usize {
        match self {
            TensorDtype::Float32 => 4,
            TensorDtype::Float64 => 8,
            TensorDtype::Int8 | TensorDtype::UInt8 => 1,
            TensorDtype::Int16
            | TensorDtype::UInt16
            | TensorDtype::Float16
            | TensorDtype::BFloat16 => 2,
            TensorDtype::Int32 | TensorDtype::UInt32 => 4,
            TensorDtype::Int64 | TensorDtype::UInt64 => 8,
        }
    }

    /// Convert to Arrow DataType
    #[inline]
    pub fn to_arrow_type(&self) -> DataType {
        match self {
            TensorDtype::Float32 => DataType::Float32,
            TensorDtype::Float64 => DataType::Float64,
            TensorDtype::Int8 => DataType::Int8,
            TensorDtype::Int16 => DataType::Int16,
            TensorDtype::Int32 => DataType::Int32,
            TensorDtype::Int64 => DataType::Int64,
            TensorDtype::UInt8 => DataType::UInt8,
            TensorDtype::UInt16 => DataType::UInt16,
            TensorDtype::UInt32 => DataType::UInt32,
            TensorDtype::UInt64 => DataType::UInt64,
            // BFloat16 and Float16 are stored as UInt16 in Arrow
            TensorDtype::BFloat16 | TensorDtype::Float16 => DataType::UInt16,
        }
    }

    /// Get dtype from string representation
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "f32" | "float32" => Some(TensorDtype::Float32),
            "f64" | "float64" => Some(TensorDtype::Float64),
            "i8" | "int8" => Some(TensorDtype::Int8),
            "i16" | "int16" => Some(TensorDtype::Int16),
            "i32" | "int32" => Some(TensorDtype::Int32),
            "i64" | "int64" => Some(TensorDtype::Int64),
            "u8" | "uint8" => Some(TensorDtype::UInt8),
            "u16" | "uint16" => Some(TensorDtype::UInt16),
            "u32" | "uint32" => Some(TensorDtype::UInt32),
            "u64" | "uint64" => Some(TensorDtype::UInt64),
            "bf16" | "bfloat16" => Some(TensorDtype::BFloat16),
            "f16" | "float16" => Some(TensorDtype::Float16),
            _ => None,
        }
    }
}

impl std::fmt::Display for TensorDtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorDtype::Float32 => write!(f, "float32"),
            TensorDtype::Float64 => write!(f, "float64"),
            TensorDtype::Int8 => write!(f, "int8"),
            TensorDtype::Int16 => write!(f, "int16"),
            TensorDtype::Int32 => write!(f, "int32"),
            TensorDtype::Int64 => write!(f, "int64"),
            TensorDtype::UInt8 => write!(f, "uint8"),
            TensorDtype::UInt16 => write!(f, "uint16"),
            TensorDtype::UInt32 => write!(f, "uint32"),
            TensorDtype::UInt64 => write!(f, "uint64"),
            TensorDtype::BFloat16 => write!(f, "bfloat16"),
            TensorDtype::Float16 => write!(f, "float16"),
        }
    }
}

/// Tensor metadata for self-describing tensors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    /// Tensor name
    pub name: String,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: TensorDtype,
    /// Strides (in elements, not bytes)
    pub strides: Option<Vec<usize>>,
    /// Custom metadata fields
    pub custom: HashMap<String, String>,
}

impl TensorMetadata {
    /// Create new tensor metadata
    pub fn new(name: String, shape: Vec<usize>, dtype: TensorDtype) -> Self {
        Self {
            name,
            shape,
            dtype,
            strides: None,
            custom: HashMap::new(),
        }
    }

    /// Set strides
    pub fn with_strides(mut self, strides: Vec<usize>) -> Self {
        self.strides = Some(strides);
        self
    }

    /// Add custom metadata
    pub fn with_custom(mut self, key: String, value: String) -> Self {
        self.custom.insert(key, value);
        self
    }

    /// Get the number of elements
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the size in bytes
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.element_size()
    }

    /// Compute default strides (row-major order)
    pub fn compute_strides(&self) -> Vec<usize> {
        if self.shape.is_empty() {
            return vec![];
        }
        let mut strides = vec![1; self.shape.len()];
        for i in (0..self.shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }
        strides
    }

    /// Get strides (computed if not specified)
    pub fn get_strides(&self) -> Vec<usize> {
        self.strides
            .clone()
            .unwrap_or_else(|| self.compute_strides())
    }
}

/// Arrow-backed tensor for zero-copy access
pub struct ArrowTensor {
    /// Tensor metadata
    pub metadata: TensorMetadata,
    /// Arrow array containing the data
    array: ArrayRef,
}

impl ArrowTensor {
    /// Create a new Arrow tensor from raw data
    pub fn from_slice_f32(name: &str, shape: Vec<usize>, data: &[f32]) -> Self {
        let metadata = TensorMetadata::new(name.to_string(), shape, TensorDtype::Float32);
        let array: ArrayRef = Arc::new(Float32Array::from(data.to_vec()));
        Self { metadata, array }
    }

    /// Create a new Arrow tensor from raw f64 data
    pub fn from_slice_f64(name: &str, shape: Vec<usize>, data: &[f64]) -> Self {
        let metadata = TensorMetadata::new(name.to_string(), shape, TensorDtype::Float64);
        let array: ArrayRef = Arc::new(Float64Array::from(data.to_vec()));
        Self { metadata, array }
    }

    /// Create from i32 data
    pub fn from_slice_i32(name: &str, shape: Vec<usize>, data: &[i32]) -> Self {
        let metadata = TensorMetadata::new(name.to_string(), shape, TensorDtype::Int32);
        let array: ArrayRef = Arc::new(Int32Array::from(data.to_vec()));
        Self { metadata, array }
    }

    /// Create from i64 data
    pub fn from_slice_i64(name: &str, shape: Vec<usize>, data: &[i64]) -> Self {
        let metadata = TensorMetadata::new(name.to_string(), shape, TensorDtype::Int64);
        let array: ArrayRef = Arc::new(Int64Array::from(data.to_vec()));
        Self { metadata, array }
    }

    /// Get zero-copy view of f32 data
    #[inline]
    pub fn as_slice_f32(&self) -> Option<&[f32]> {
        self.array
            .as_any()
            .downcast_ref::<Float32Array>()
            .map(|arr| arr.values().as_ref())
    }

    /// Get zero-copy view of f64 data
    #[inline]
    pub fn as_slice_f64(&self) -> Option<&[f64]> {
        self.array
            .as_any()
            .downcast_ref::<Float64Array>()
            .map(|arr| arr.values().as_ref())
    }

    /// Get zero-copy view of i32 data
    #[inline]
    pub fn as_slice_i32(&self) -> Option<&[i32]> {
        self.array
            .as_any()
            .downcast_ref::<Int32Array>()
            .map(|arr| arr.values().as_ref())
    }

    /// Get zero-copy view of i64 data
    #[inline]
    pub fn as_slice_i64(&self) -> Option<&[i64]> {
        self.array
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(|arr| arr.values().as_ref())
    }

    /// Get raw bytes (copies data)
    pub fn as_bytes(&self) -> Vec<u8> {
        let data = self.array.to_data();
        if data.buffers().is_empty() {
            Vec::new()
        } else {
            data.buffers()[0].as_slice().to_vec()
        }
    }

    /// Get the underlying Arrow array
    #[inline]
    pub fn array(&self) -> &ArrayRef {
        &self.array
    }

    /// Get the number of elements
    #[inline]
    pub fn len(&self) -> usize {
        self.array.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.array.is_empty()
    }
}

/// Collection of tensors stored in Arrow format
pub struct ArrowTensorStore {
    /// Tensors by name
    tensors: HashMap<String, ArrowTensor>,
    /// Schema for the tensor collection
    schema: Option<Arc<Schema>>,
}

impl ArrowTensorStore {
    /// Create a new empty store
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            schema: None,
        }
    }

    /// Add a tensor to the store
    pub fn insert(&mut self, tensor: ArrowTensor) {
        self.schema = None; // Invalidate schema
        self.tensors.insert(tensor.metadata.name.clone(), tensor);
    }

    /// Get a tensor by name
    #[inline]
    pub fn get(&self, name: &str) -> Option<&ArrowTensor> {
        self.tensors.get(name)
    }

    /// List all tensor names
    pub fn names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of tensors
    #[inline]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Build Arrow schema for all tensors
    pub fn build_schema(&mut self) -> Arc<Schema> {
        if let Some(ref schema) = self.schema {
            return schema.clone();
        }

        let fields: Vec<Field> = self
            .tensors
            .values()
            .map(|t| {
                let mut metadata = HashMap::new();
                metadata.insert("shape".to_string(), format!("{:?}", t.metadata.shape));
                metadata.insert("dtype".to_string(), t.metadata.dtype.to_string());
                if let Some(ref strides) = t.metadata.strides {
                    metadata.insert("strides".to_string(), format!("{:?}", strides));
                }
                for (k, v) in &t.metadata.custom {
                    metadata.insert(k.clone(), v.clone());
                }
                Field::new(&t.metadata.name, t.metadata.dtype.to_arrow_type(), false)
                    .with_metadata(metadata)
            })
            .collect();

        let schema = Arc::new(Schema::new(fields));
        self.schema = Some(schema.clone());
        schema
    }

    /// Convert to RecordBatch for IPC
    pub fn to_record_batch(&mut self) -> Result<RecordBatch, arrow::error::ArrowError> {
        let schema = self.build_schema();
        let columns: Vec<ArrayRef> = self.tensors.values().map(|t| t.array.clone()).collect();
        RecordBatch::try_new(schema, columns)
    }

    /// Write to Arrow IPC format
    pub fn write_ipc<W: Write>(&mut self, writer: W) -> Result<(), arrow::error::ArrowError> {
        let batch = self.to_record_batch()?;
        let schema = batch.schema();
        let mut ipc_writer = FileWriter::try_new(writer, &schema)?;
        ipc_writer.write(&batch)?;
        ipc_writer.finish()?;
        Ok(())
    }

    /// Read from Arrow IPC format
    pub fn read_ipc<R: Read + Seek>(reader: R) -> Result<Self, arrow::error::ArrowError> {
        let ipc_reader = FileReader::try_new(reader, None)?;
        let schema = ipc_reader.schema();
        let mut store = Self::new();

        for batch_result in ipc_reader {
            let batch = batch_result?;
            for (i, field) in schema.fields().iter().enumerate() {
                let array = batch.column(i).clone();
                let shape = parse_shape_from_metadata(field.metadata());
                let dtype = dtype_from_arrow(field.data_type());

                let metadata = TensorMetadata::new(field.name().clone(), shape, dtype);
                store
                    .tensors
                    .insert(field.name().clone(), ArrowTensor { metadata, array });
            }
        }

        store.schema = Some(schema);
        Ok(store)
    }

    /// Serialize to bytes (Arrow IPC format)
    pub fn to_bytes(&mut self) -> Result<Bytes, arrow::error::ArrowError> {
        let mut buffer = Vec::new();
        self.write_ipc(&mut buffer)?;
        Ok(Bytes::from(buffer))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, arrow::error::ArrowError> {
        let cursor = std::io::Cursor::new(bytes);
        Self::read_ipc(cursor)
    }
}

impl Default for ArrowTensorStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse shape from field metadata
fn parse_shape_from_metadata(metadata: &HashMap<String, String>) -> Vec<usize> {
    metadata
        .get("shape")
        .and_then(|s| {
            // Parse "[1, 2, 3]" format
            let trimmed = s.trim_start_matches('[').trim_end_matches(']');
            let parts: Result<Vec<usize>, _> =
                trimmed.split(',').map(|p| p.trim().parse()).collect();
            parts.ok()
        })
        .unwrap_or_default()
}

/// Convert Arrow DataType to TensorDtype
fn dtype_from_arrow(dt: &DataType) -> TensorDtype {
    match dt {
        DataType::Float32 => TensorDtype::Float32,
        DataType::Float64 => TensorDtype::Float64,
        DataType::Int8 => TensorDtype::Int8,
        DataType::Int16 => TensorDtype::Int16,
        DataType::Int32 => TensorDtype::Int32,
        DataType::Int64 => TensorDtype::Int64,
        DataType::UInt8 => TensorDtype::UInt8,
        DataType::UInt16 => TensorDtype::UInt16,
        DataType::UInt32 => TensorDtype::UInt32,
        DataType::UInt64 => TensorDtype::UInt64,
        _ => TensorDtype::Float32, // Default
    }
}

/// Zero-copy tensor accessor trait
pub trait ZeroCopyAccessor {
    /// Get raw byte vector
    fn get_bytes(&self) -> Vec<u8>;

    /// Get length in bytes
    fn len_bytes(&self) -> usize {
        self.get_bytes().len()
    }
}

impl ZeroCopyAccessor for ArrowTensor {
    fn get_bytes(&self) -> Vec<u8> {
        ArrowTensor::as_bytes(self)
    }
}

/// Create Arrow buffer from raw bytes (zero-copy when possible)
#[allow(deprecated)]
pub fn buffer_from_bytes(bytes: Bytes) -> Buffer {
    Buffer::from(bytes)
}

/// Create typed array from buffer
#[allow(dead_code)]
fn create_array_from_buffer(buffer: Buffer, dtype: TensorDtype, _len: usize) -> ArrayRef {
    match dtype {
        TensorDtype::Float32 => Arc::new(Float32Array::new(buffer.into(), None)) as ArrayRef,
        TensorDtype::Float64 => Arc::new(Float64Array::new(buffer.into(), None)) as ArrayRef,
        TensorDtype::Int8 => Arc::new(Int8Array::new(buffer.into(), None)) as ArrayRef,
        TensorDtype::Int16 => Arc::new(Int16Array::new(buffer.into(), None)) as ArrayRef,
        TensorDtype::Int32 => Arc::new(Int32Array::new(buffer.into(), None)) as ArrayRef,
        TensorDtype::Int64 => Arc::new(Int64Array::new(buffer.into(), None)) as ArrayRef,
        TensorDtype::UInt8 => Arc::new(UInt8Array::new(buffer.into(), None)) as ArrayRef,
        TensorDtype::UInt16 => Arc::new(UInt16Array::new(buffer.into(), None)) as ArrayRef,
        TensorDtype::UInt32 => Arc::new(UInt32Array::new(buffer.into(), None)) as ArrayRef,
        TensorDtype::UInt64 => Arc::new(UInt64Array::new(buffer.into(), None)) as ArrayRef,
        // Float16/BFloat16 stored as UInt16
        TensorDtype::Float16 | TensorDtype::BFloat16 => {
            Arc::new(UInt16Array::new(buffer.into(), None)) as ArrayRef
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_metadata() {
        let meta = TensorMetadata::new("test".to_string(), vec![2, 3, 4], TensorDtype::Float32);
        assert_eq!(meta.numel(), 24);
        assert_eq!(meta.size_bytes(), 96);
        assert_eq!(meta.compute_strides(), vec![12, 4, 1]);
    }

    #[test]
    fn test_arrow_tensor_f32() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = ArrowTensor::from_slice_f32("weights", vec![2, 3], &data);

        assert_eq!(tensor.metadata.name, "weights");
        assert_eq!(tensor.metadata.shape, vec![2, 3]);
        assert_eq!(tensor.len(), 6);

        let slice = tensor.as_slice_f32().expect("test: should succeed");
        assert_eq!(slice, &data);
    }

    #[test]
    fn test_arrow_tensor_store() {
        let mut store = ArrowTensorStore::new();

        let w1 = ArrowTensor::from_slice_f32("layer1.weight", vec![4, 3], &[0.0; 12]);
        let w2 = ArrowTensor::from_slice_f32("layer2.weight", vec![2, 4], &[0.0; 8]);

        store.insert(w1);
        store.insert(w2);

        assert_eq!(store.len(), 2);
        assert!(store.get("layer1.weight").is_some());
        assert!(store.get("layer2.weight").is_some());
    }

    #[test]
    fn test_ipc_roundtrip() {
        let mut store = ArrowTensorStore::new();
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        store.insert(ArrowTensor::from_slice_f32("test", vec![3, 4], &data));

        let bytes = store.to_bytes().expect("test: should succeed");
        let loaded = ArrowTensorStore::from_bytes(&bytes).expect("test: should succeed");

        assert_eq!(loaded.len(), 1);
        let tensor = loaded.get("test").expect("test: should succeed");
        assert_eq!(tensor.as_slice_f32().expect("test: should succeed"), &data);
    }

    #[test]
    fn test_dtype_conversion() {
        assert_eq!(TensorDtype::Float32.to_arrow_type(), DataType::Float32);
        assert_eq!(TensorDtype::Int64.to_arrow_type(), DataType::Int64);
        assert_eq!(TensorDtype::Float32.element_size(), 4);
        assert_eq!(TensorDtype::Float64.element_size(), 8);
    }
}
