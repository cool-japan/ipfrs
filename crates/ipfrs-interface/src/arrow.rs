//! Apache Arrow integration for zero-copy data exchange
//!
//! This module provides Apache Arrow IPC format support for tensor data,
//! enabling efficient zero-copy data transfer for ML/data science workflows.

use arrow::array::{
    ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, UInt16Array, UInt32Array,
    UInt64Array, UInt8Array,
};
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use ipfrs_core::error::{Error, Result};
use std::sync::Arc;

use crate::tensor::TensorMetadata;

/// Convert tensor data to Apache Arrow RecordBatch
///
/// The tensor is represented as a single column in the RecordBatch,
/// with the column name "data" and appropriate Arrow data type.
pub fn tensor_to_record_batch(metadata: &TensorMetadata, data: &[u8]) -> Result<RecordBatch> {
    // Determine Arrow data type from tensor dtype string
    let arrow_dtype = match metadata.dtype.as_str() {
        "F32" | "f32" => DataType::Float32,
        "F64" | "f64" => DataType::Float64,
        "I32" | "i32" => DataType::Int32,
        "I64" | "i64" => DataType::Int64,
        "U8" | "u8" => DataType::UInt8,
        "U16" | "u16" => DataType::UInt16,
        "U32" | "u32" => DataType::UInt32,
        "U64" | "u64" => DataType::UInt64,
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype: {}",
                metadata.dtype
            )))
        }
    };

    // Create schema
    let schema = Schema::new(vec![Field::new("data", arrow_dtype.clone(), false)]);

    // Create array from raw data
    let array: ArrayRef = match metadata.dtype.as_str() {
        "F32" | "f32" => {
            let buffer = Buffer::from(data);
            Arc::new(Float32Array::new(buffer.into(), None))
        }
        "F64" | "f64" => {
            let buffer = Buffer::from(data);
            Arc::new(Float64Array::new(buffer.into(), None))
        }
        "I32" | "i32" => {
            let buffer = Buffer::from(data);
            Arc::new(Int32Array::new(buffer.into(), None))
        }
        "I64" | "i64" => {
            let buffer = Buffer::from(data);
            Arc::new(Int64Array::new(buffer.into(), None))
        }
        "U8" | "u8" => {
            let buffer = Buffer::from(data);
            Arc::new(UInt8Array::new(buffer.into(), None))
        }
        "U16" | "u16" => {
            let buffer = Buffer::from(data);
            Arc::new(UInt16Array::new(buffer.into(), None))
        }
        "U32" | "u32" => {
            let buffer = Buffer::from(data);
            Arc::new(UInt32Array::new(buffer.into(), None))
        }
        "U64" | "u64" => {
            let buffer = Buffer::from(data);
            Arc::new(UInt64Array::new(buffer.into(), None))
        }
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype: {}",
                metadata.dtype
            )))
        }
    };

    // Create record batch
    RecordBatch::try_new(Arc::new(schema), vec![array])
        .map_err(|e| Error::Internal(format!("Failed to create Arrow RecordBatch: {}", e)))
}

/// Serialize RecordBatch to Arrow IPC Stream format
///
/// Returns the serialized bytes that can be sent over HTTP
pub fn record_batch_to_ipc_bytes(batch: &RecordBatch) -> Result<Bytes> {
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &batch.schema())
            .map_err(|e| Error::Internal(format!("Failed to create Arrow StreamWriter: {}", e)))?;

        writer
            .write(batch)
            .map_err(|e| Error::Internal(format!("Failed to write Arrow batch: {}", e)))?;

        writer
            .finish()
            .map_err(|e| Error::Internal(format!("Failed to finish Arrow stream: {}", e)))?;
    }

    Ok(Bytes::from(buffer))
}

/// Create Arrow schema with metadata for tensor shape and dtype
///
/// This enriches the Arrow schema with custom metadata about the tensor dimensions
pub fn create_tensor_schema(metadata: &TensorMetadata) -> Result<Schema> {
    let arrow_dtype = match metadata.dtype.as_str() {
        "F32" | "f32" => DataType::Float32,
        "F64" | "f64" => DataType::Float64,
        "I32" | "i32" => DataType::Int32,
        "I64" | "i64" => DataType::Int64,
        "U8" | "u8" => DataType::UInt8,
        "U16" | "u16" => DataType::UInt16,
        "U32" | "u32" => DataType::UInt32,
        "U64" | "u64" => DataType::UInt64,
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype: {}",
                metadata.dtype
            )))
        }
    };

    // Create field with metadata
    let mut field = Field::new("data", arrow_dtype, false);

    // Add tensor shape as metadata
    let shape_str = metadata
        .shape
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(",");

    field = field.with_metadata(
        [
            ("tensor_shape".to_string(), shape_str),
            ("tensor_dtype".to_string(), metadata.dtype.clone()),
            (
                "tensor_layout".to_string(),
                format!("{:?}", metadata.layout),
            ),
        ]
        .into_iter()
        .collect(),
    );

    Ok(Schema::new(vec![field]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorLayout;

    #[test]
    fn test_tensor_to_record_batch_f32() {
        let metadata = TensorMetadata {
            shape: vec![2, 3],
            dtype: "F32".to_string(),
            num_elements: 6,
            size_bytes: 24,
            layout: TensorLayout::RowMajor,
        };

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes = data
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<u8>>();

        let batch =
            tensor_to_record_batch(&metadata, &bytes).expect("test: f32 tensor to record batch");
        assert_eq!(batch.num_columns(), 1);
        assert_eq!(batch.num_rows(), 6);

        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("test: downcast column to Float32Array");
        assert_eq!(array.value(0), 1.0);
        assert_eq!(array.value(5), 6.0);
    }

    #[test]
    fn test_tensor_to_record_batch_i32() {
        let metadata = TensorMetadata {
            shape: vec![4],
            dtype: "I32".to_string(),
            num_elements: 4,
            size_bytes: 16,
            layout: TensorLayout::RowMajor,
        };

        let data: Vec<i32> = vec![10, 20, 30, 40];
        let bytes = data
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect::<Vec<u8>>();

        let batch = tensor_to_record_batch(&metadata, &bytes)
            .expect("test: tensor_to_record_batch should succeed for I32 data");
        assert_eq!(batch.num_rows(), 4);

        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("test: downcast to Int32Array should succeed");
        assert_eq!(array.value(0), 10);
        assert_eq!(array.value(3), 40);
    }

    #[test]
    fn test_record_batch_to_ipc_bytes() {
        let metadata = TensorMetadata {
            shape: vec![3],
            dtype: "F32".to_string(),
            num_elements: 3,
            size_bytes: 12,
            layout: TensorLayout::RowMajor,
        };

        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let bytes = data
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<u8>>();

        let batch = tensor_to_record_batch(&metadata, &bytes)
            .expect("test: tensor_to_record_batch should succeed for F32 data");
        let ipc_bytes =
            record_batch_to_ipc_bytes(&batch).expect("test: IPC serialization should succeed");

        // IPC format should have non-trivial size (header + data)
        assert!(ipc_bytes.len() > 50);
    }

    #[test]
    fn test_create_tensor_schema() {
        let metadata = TensorMetadata {
            shape: vec![10, 20, 30],
            dtype: "F64".to_string(),
            num_elements: 6000,
            size_bytes: 48000,
            layout: TensorLayout::RowMajor,
        };

        let schema = create_tensor_schema(&metadata)
            .expect("test: schema creation should succeed for F64 tensor");
        assert_eq!(schema.fields().len(), 1);

        let field = &schema.fields()[0];
        assert_eq!(field.name(), "data");
        assert_eq!(field.data_type(), &DataType::Float64);

        let meta = field.metadata();
        assert!(meta.contains_key("tensor_shape"));
        assert_eq!(
            meta.get("tensor_shape")
                .expect("test: tensor_shape metadata key should be present"),
            "10,20,30"
        );
        assert_eq!(
            meta.get("tensor_dtype")
                .expect("test: tensor_dtype metadata key should be present"),
            "F64"
        );
    }

    #[test]
    fn test_all_dtypes() {
        let dtypes = vec!["F32", "F64", "I32", "I64", "U8", "U16", "U32", "U64"];

        for dtype in dtypes {
            let element_size = match dtype {
                "F32" | "I32" | "U32" => 4,
                "F64" | "I64" | "U64" => 8,
                "U8" => 1,
                "U16" => 2,
                _ => 4,
            };

            let metadata = TensorMetadata {
                shape: vec![4],
                dtype: dtype.to_string(),
                num_elements: 4,
                size_bytes: 4 * element_size,
                layout: TensorLayout::RowMajor,
            };

            let data = vec![0u8; metadata.size_bytes];
            let result = tensor_to_record_batch(&metadata, &data);
            assert!(result.is_ok(), "Failed for dtype: {}", dtype);
        }
    }
}
