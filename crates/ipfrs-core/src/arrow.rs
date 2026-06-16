//! Apache Arrow memory layout integration for zero-copy tensor access.
//!
//! This module provides conversions between IPFRS tensor types and Apache Arrow arrays,
//! enabling zero-copy interoperability with the Arrow ecosystem (Parquet, Flight, etc.).
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_core::arrow::{TensorBlockArrowExt, arrow_to_tensor_block};
//! use ipfrs_core::tensor::{TensorBlock, TensorDtype, TensorShape};
//! use bytes::Bytes;
//! use arrow_array::Float32Array;
//!
//! // Convert Arrow array to TensorBlock (zero-copy)
//! let arrow_array = Float32Array::from(vec![1.0f32, 2.0, 3.0, 4.0]);
//! let tensor = arrow_to_tensor_block(&arrow_array, TensorShape::new(vec![2, 2])).unwrap();
//!
//! // Convert TensorBlock back to Arrow array
//! let arrow_back = tensor.to_arrow_array().unwrap();
//! ```

use crate::error::{Error, Result};
use crate::tensor::{TensorBlock, TensorDtype, TensorShape};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, Int8Array,
    UInt32Array, UInt8Array,
};
use arrow_buffer::Buffer;
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use std::sync::Arc;

/// Extension trait for TensorBlock to provide Arrow conversions
pub trait TensorBlockArrowExt {
    /// Convert to an Arrow array (zero-copy when possible)
    fn to_arrow_array(&self) -> Result<ArrayRef>;

    /// Convert to an Arrow Field (for schema)
    fn to_arrow_field(&self, name: &str) -> Field;

    /// Convert to an Arrow Schema
    fn to_arrow_schema(&self, field_name: &str) -> Schema;
}

impl TensorBlockArrowExt for TensorBlock {
    fn to_arrow_array(&self) -> Result<ArrayRef> {
        let metadata = self.metadata();
        let data = self.data();

        match metadata.dtype {
            TensorDtype::F32 => {
                let buffer = Buffer::from(data.clone());
                let array = Float32Array::new(buffer.into(), None);
                Ok(Arc::new(array) as ArrayRef)
            }
            TensorDtype::F64 => {
                let buffer = Buffer::from(data.clone());
                let array = Float64Array::new(buffer.into(), None);
                Ok(Arc::new(array) as ArrayRef)
            }
            TensorDtype::I8 => {
                let buffer = Buffer::from(data.clone());
                let array = Int8Array::new(buffer.into(), None);
                Ok(Arc::new(array) as ArrayRef)
            }
            TensorDtype::I32 => {
                let buffer = Buffer::from(data.clone());
                let array = Int32Array::new(buffer.into(), None);
                Ok(Arc::new(array) as ArrayRef)
            }
            TensorDtype::I64 => {
                let buffer = Buffer::from(data.clone());
                let array = Int64Array::new(buffer.into(), None);
                Ok(Arc::new(array) as ArrayRef)
            }
            TensorDtype::U8 => {
                let buffer = Buffer::from(data.clone());
                let array = UInt8Array::new(buffer.into(), None);
                Ok(Arc::new(array) as ArrayRef)
            }
            TensorDtype::U32 => {
                let buffer = Buffer::from(data.clone());
                let array = UInt32Array::new(buffer.into(), None);
                Ok(Arc::new(array) as ArrayRef)
            }
            TensorDtype::Bool => {
                // Boolean arrays are stored as bit-packed in Arrow
                let bytes: Vec<u8> = data.to_vec();
                let array = BooleanArray::from(bytes.iter().map(|&b| b != 0).collect::<Vec<_>>());
                Ok(Arc::new(array) as ArrayRef)
            }
            TensorDtype::F16 => {
                // Arrow doesn't have native F16 support, convert to F32
                Err(Error::InvalidInput(
                    "F16 not directly supported by Arrow, use F32 instead".to_string(),
                ))
            }
        }
    }

    fn to_arrow_field(&self, name: &str) -> Field {
        let metadata = self.metadata();
        let arrow_dtype = tensor_dtype_to_arrow(&metadata.dtype);
        Field::new(name, arrow_dtype, false)
    }

    fn to_arrow_schema(&self, field_name: &str) -> Schema {
        Schema::new(vec![self.to_arrow_field(field_name)])
    }
}

/// Convert Arrow DataType to TensorDtype
pub fn arrow_dtype_to_tensor(dtype: &DataType) -> Result<TensorDtype> {
    match dtype {
        DataType::Float32 => Ok(TensorDtype::F32),
        DataType::Float64 => Ok(TensorDtype::F64),
        DataType::Int8 => Ok(TensorDtype::I8),
        DataType::Int32 => Ok(TensorDtype::I32),
        DataType::Int64 => Ok(TensorDtype::I64),
        DataType::UInt8 => Ok(TensorDtype::U8),
        DataType::UInt32 => Ok(TensorDtype::U32),
        DataType::Boolean => Ok(TensorDtype::Bool),
        _ => Err(Error::InvalidInput(format!(
            "Unsupported Arrow dtype: {:?}",
            dtype
        ))),
    }
}

/// Convert TensorDtype to Arrow DataType
pub fn tensor_dtype_to_arrow(dtype: &TensorDtype) -> DataType {
    match dtype {
        TensorDtype::F32 => DataType::Float32,
        TensorDtype::F64 => DataType::Float64,
        TensorDtype::I8 => DataType::Int8,
        TensorDtype::I32 => DataType::Int32,
        TensorDtype::I64 => DataType::Int64,
        TensorDtype::U8 => DataType::UInt8,
        TensorDtype::U32 => DataType::UInt32,
        TensorDtype::Bool => DataType::Boolean,
        TensorDtype::F16 => DataType::Float32, // Fallback to F32
    }
}

/// Convert an Arrow array to a TensorBlock (zero-copy)
pub fn arrow_to_tensor_block(array: &dyn Array, shape: TensorShape) -> Result<TensorBlock> {
    let dtype = arrow_dtype_to_tensor(array.data_type())?;

    // Get the raw buffer data
    let data = match array.data_type() {
        DataType::Float32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("checked: DataType::Float32 matches Float32Array");
            let buffer = arr.values();
            // Cast typed slice to &[u8] for Bytes
            let byte_slice = unsafe {
                std::slice::from_raw_parts(
                    buffer.as_ptr() as *const u8,
                    buffer.len() * std::mem::size_of::<f32>(),
                )
            };
            Bytes::copy_from_slice(byte_slice)
        }
        DataType::Float64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("checked: DataType::Float64 matches Float64Array");
            let buffer = arr.values();
            let byte_slice = unsafe {
                std::slice::from_raw_parts(
                    buffer.as_ptr() as *const u8,
                    buffer.len() * std::mem::size_of::<f64>(),
                )
            };
            Bytes::copy_from_slice(byte_slice)
        }
        DataType::Int8 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int8Array>()
                .expect("checked: DataType::Int8 matches Int8Array");
            let buffer = arr.values();
            let byte_slice =
                unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const u8, buffer.len()) };
            Bytes::copy_from_slice(byte_slice)
        }
        DataType::Int32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("checked: DataType::Int32 matches Int32Array");
            let buffer = arr.values();
            let byte_slice = unsafe {
                std::slice::from_raw_parts(
                    buffer.as_ptr() as *const u8,
                    buffer.len() * std::mem::size_of::<i32>(),
                )
            };
            Bytes::copy_from_slice(byte_slice)
        }
        DataType::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("checked: DataType::Int64 matches Int64Array");
            let buffer = arr.values();
            let byte_slice = unsafe {
                std::slice::from_raw_parts(
                    buffer.as_ptr() as *const u8,
                    buffer.len() * std::mem::size_of::<i64>(),
                )
            };
            Bytes::copy_from_slice(byte_slice)
        }
        DataType::UInt8 => {
            let arr = array
                .as_any()
                .downcast_ref::<UInt8Array>()
                .expect("checked: DataType::UInt8 matches UInt8Array");
            let buffer = arr.values();
            Bytes::copy_from_slice(buffer.as_ref())
        }
        DataType::UInt32 => {
            let arr = array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .expect("checked: DataType::UInt32 matches UInt32Array");
            let buffer = arr.values();
            let byte_slice = unsafe {
                std::slice::from_raw_parts(
                    buffer.as_ptr() as *const u8,
                    buffer.len() * std::mem::size_of::<u32>(),
                )
            };
            Bytes::copy_from_slice(byte_slice)
        }
        DataType::Boolean => {
            let arr = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("checked: DataType::Boolean matches BooleanArray");
            let bytes: Vec<u8> = (0..arr.len()).map(|i| arr.value(i) as u8).collect();
            Bytes::from(bytes)
        }
        _ => {
            return Err(Error::InvalidInput(format!(
                "Unsupported Arrow dtype: {:?}",
                array.data_type()
            )))
        }
    };

    TensorBlock::new(data, shape, dtype)
}

/// Create an Arrow RecordBatch from multiple TensorBlocks
#[allow(dead_code)]
pub fn tensors_to_record_batch(
    tensors: Vec<(&str, &TensorBlock)>,
) -> Result<arrow_array::RecordBatch> {
    let mut fields = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();

    for (name, tensor) in tensors {
        fields.push(tensor.to_arrow_field(name));
        arrays.push(tensor.to_arrow_array()?);
    }

    let schema = Arc::new(Schema::new(fields));
    arrow_array::RecordBatch::try_new(schema, arrays)
        .map_err(|e| Error::InvalidInput(format!("Failed to create RecordBatch: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_to_arrow_f32() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let bytes = Bytes::from(
            data.iter()
                .flat_map(|&f| f.to_le_bytes())
                .collect::<Vec<u8>>(),
        );

        let tensor =
            TensorBlock::new(bytes, TensorShape::new(vec![2, 2]), TensorDtype::F32).unwrap();

        let arrow_array = tensor.to_arrow_array().unwrap();
        let f32_array = arrow_array.as_any().downcast_ref::<Float32Array>().unwrap();

        assert_eq!(f32_array.len(), 4);
        assert_eq!(f32_array.value(0), 1.0);
        assert_eq!(f32_array.value(1), 2.0);
        assert_eq!(f32_array.value(2), 3.0);
        assert_eq!(f32_array.value(3), 4.0);
    }

    #[test]
    fn test_arrow_to_tensor_f32() {
        let arrow_array = Float32Array::from(vec![1.0f32, 2.0, 3.0, 4.0]);
        let tensor = arrow_to_tensor_block(&arrow_array, TensorShape::new(vec![2, 2])).unwrap();

        assert_eq!(tensor.element_count(), 4);
        assert_eq!(tensor.metadata().dtype, TensorDtype::F32);
    }

    #[test]
    fn test_tensor_to_arrow_i32() {
        let data = [1i32, 2, 3, 4];
        let bytes = Bytes::from(
            data.iter()
                .flat_map(|&i| i.to_le_bytes())
                .collect::<Vec<u8>>(),
        );

        let tensor = TensorBlock::new(bytes, TensorShape::new(vec![4]), TensorDtype::I32).unwrap();

        let arrow_array = tensor.to_arrow_array().unwrap();
        let i32_array = arrow_array.as_any().downcast_ref::<Int32Array>().unwrap();

        assert_eq!(i32_array.len(), 4);
        assert_eq!(i32_array.value(0), 1);
        assert_eq!(i32_array.value(3), 4);
    }

    #[test]
    fn test_dtype_conversions() {
        // TensorDtype to Arrow DataType
        assert_eq!(tensor_dtype_to_arrow(&TensorDtype::F32), DataType::Float32);
        assert_eq!(tensor_dtype_to_arrow(&TensorDtype::I64), DataType::Int64);
        assert_eq!(tensor_dtype_to_arrow(&TensorDtype::Bool), DataType::Boolean);

        // Arrow DataType to TensorDtype
        assert_eq!(
            arrow_dtype_to_tensor(&DataType::Float32).unwrap(),
            TensorDtype::F32
        );
        assert_eq!(
            arrow_dtype_to_tensor(&DataType::Int64).unwrap(),
            TensorDtype::I64
        );
    }

    #[test]
    fn test_arrow_schema_generation() {
        let data = Bytes::from(vec![0u8; 16]);
        let tensor = TensorBlock::new(data, TensorShape::new(vec![4]), TensorDtype::F32).unwrap();

        let schema = tensor.to_arrow_schema("tensor_data");
        assert_eq!(schema.fields().len(), 1);
        assert_eq!(schema.field(0).name(), "tensor_data");
        assert_eq!(schema.field(0).data_type(), &DataType::Float32);
    }

    #[test]
    fn test_zero_copy_roundtrip() {
        // Create Arrow array
        let original_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arrow_array = Float32Array::from(original_data.clone());

        // Convert to TensorBlock
        let tensor = arrow_to_tensor_block(&arrow_array, TensorShape::new(vec![2, 3])).unwrap();

        // Convert back to Arrow
        let arrow_back = tensor.to_arrow_array().unwrap();
        let f32_back = arrow_back.as_any().downcast_ref::<Float32Array>().unwrap();

        // Verify data integrity
        assert_eq!(f32_back.len(), original_data.len());
        for (i, &expected) in original_data.iter().enumerate() {
            assert_eq!(f32_back.value(i), expected);
        }
    }

    #[test]
    fn test_tensor_to_arrow_field() {
        let data = Bytes::from(vec![0u8; 64]); // 8 elements * 8 bytes per I64
        let tensor = TensorBlock::new(data, TensorShape::new(vec![8]), TensorDtype::I64).unwrap();

        let field = tensor.to_arrow_field("my_tensor");
        assert_eq!(field.name(), "my_tensor");
        assert_eq!(field.data_type(), &DataType::Int64);
        assert!(!field.is_nullable());
    }
}
