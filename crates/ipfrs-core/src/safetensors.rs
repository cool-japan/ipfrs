//! # Safetensors Format Support
//!
//! This module provides support for parsing and working with the Safetensors format,
//! a safe format for storing tensors that includes metadata and data in a single file.
//!
//! ## Format Structure
//!
//! ```text
//! [ 8 bytes: header length (u64 LE) ]
//! [ N bytes: JSON metadata header ]
//! [ M bytes: raw tensor data ]
//! ```
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_core::safetensors::SafetensorsFile;
//!
//! // Create a simple Safetensors file
//! # fn create_test_safetensors() -> Vec<u8> {
//! #     let metadata = serde_json::json!({"weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}});
//! #     let header_bytes = serde_json::to_vec(&metadata).unwrap();
//! #     let header_len = header_bytes.len() as u64;
//! #     let mut file = Vec::new();
//! #     file.extend_from_slice(&header_len.to_le_bytes());
//! #     file.extend_from_slice(&header_bytes);
//! #     file.extend_from_slice(&1.0f32.to_le_bytes());
//! #     file.extend_from_slice(&2.0f32.to_le_bytes());
//! #     file.extend_from_slice(&3.0f32.to_le_bytes());
//! #     file.extend_from_slice(&4.0f32.to_le_bytes());
//! #     file
//! # }
//! let data = create_test_safetensors();
//! let file = SafetensorsFile::parse(&data).unwrap();
//!
//! // Get tensor metadata
//! for (name, info) in file.tensors() {
//!     println!("Tensor: {}, shape: {:?}, dtype: {:?}", name, info.shape, info.dtype);
//! }
//! ```

use crate::{tensor::TensorDtype, Error, Ipld, Result, TensorBlock, TensorShape};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Safetensors metadata for a single tensor
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafetensorInfo {
    /// Data type of the tensor
    pub dtype: String,
    /// Shape of the tensor (dimensions)
    pub shape: Vec<usize>,
    /// Data offsets [start, end] in the data section
    pub data_offsets: [usize; 2],
}

impl SafetensorInfo {
    /// Convert Safetensors dtype string to TensorDtype
    pub fn to_tensor_dtype(&self) -> Result<TensorDtype> {
        match self.dtype.as_str() {
            "F32" => Ok(TensorDtype::F32),
            "F64" => Ok(TensorDtype::F64),
            "F16" => Ok(TensorDtype::F16),
            "I8" => Ok(TensorDtype::I8),
            "I32" => Ok(TensorDtype::I32),
            "I64" => Ok(TensorDtype::I64),
            "U8" => Ok(TensorDtype::U8),
            "U32" => Ok(TensorDtype::U32),
            "BOOL" => Ok(TensorDtype::Bool),
            _ => Err(Error::InvalidData(format!(
                "Unsupported Safetensors dtype: {}",
                self.dtype
            ))),
        }
    }

    /// Get the size in bytes of this tensor
    pub fn size_bytes(&self) -> usize {
        self.data_offsets[1] - self.data_offsets[0]
    }
}

/// A parsed Safetensors file with metadata and data
#[derive(Debug, Clone)]
pub struct SafetensorsFile {
    /// Tensor metadata mapped by tensor name
    tensors: BTreeMap<String, SafetensorInfo>,
    /// Raw data section (all tensors concatenated)
    data: Bytes,
    /// Offset where data section starts in the original file
    #[allow(dead_code)]
    data_offset: usize,
}

impl SafetensorsFile {
    /// Parse a Safetensors file from bytes
    ///
    /// # Format
    ///
    /// - First 8 bytes: header length as little-endian u64
    /// - Next N bytes: JSON metadata header
    /// - Remaining bytes: raw tensor data
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(Error::InvalidData(
                "Safetensors file too small (missing header length)".to_string(),
            ));
        }

        // Read header length (first 8 bytes, little-endian u64)
        let header_len = u64::from_le_bytes(
            bytes[0..8]
                .try_into()
                .map_err(|_| Error::InvalidData("Failed to read header length".to_string()))?,
        ) as usize;

        if bytes.len() < 8 + header_len {
            return Err(Error::InvalidData(
                "Safetensors file too small (truncated header)".to_string(),
            ));
        }

        // Parse JSON metadata
        let header_bytes = &bytes[8..8 + header_len];
        let tensors: BTreeMap<String, SafetensorInfo> = serde_json::from_slice(header_bytes)
            .map_err(|e| {
                Error::InvalidData(format!("Failed to parse Safetensors header: {}", e))
            })?;

        // Extract data section
        let data_offset = 8 + header_len;
        let data = Bytes::copy_from_slice(&bytes[data_offset..]);

        Ok(Self {
            tensors,
            data,
            data_offset,
        })
    }

    /// Get all tensor metadata
    pub fn tensors(&self) -> &BTreeMap<String, SafetensorInfo> {
        &self.tensors
    }

    /// Get metadata for a specific tensor by name
    pub fn get_tensor_info(&self, name: &str) -> Option<&SafetensorInfo> {
        self.tensors.get(name)
    }

    /// Get the raw data for a specific tensor (zero-copy slice)
    pub fn get_tensor_data(&self, name: &str) -> Result<Bytes> {
        let info = self
            .get_tensor_info(name)
            .ok_or_else(|| Error::InvalidData(format!("Tensor '{}' not found", name)))?;

        let start = info.data_offsets[0];
        let end = info.data_offsets[1];

        if end > self.data.len() {
            return Err(Error::InvalidData(format!(
                "Tensor '{}' data offset out of bounds",
                name
            )));
        }

        // Zero-copy slice
        Ok(self.data.slice(start..end))
    }

    /// Convert a tensor to a TensorBlock
    pub fn to_tensor_block(&self, name: &str) -> Result<TensorBlock> {
        let info = self
            .get_tensor_info(name)
            .ok_or_else(|| Error::InvalidData(format!("Tensor '{}' not found", name)))?;

        let data = self.get_tensor_data(name)?;
        let shape = TensorShape::new(info.shape.clone());
        let dtype = info.to_tensor_dtype()?;

        TensorBlock::new(data, shape, dtype)
    }

    /// Convert all tensors to IPLD metadata format
    ///
    /// Creates an IPLD map with:
    /// - Tensor names as keys
    /// - Metadata (shape, dtype, CID links to data) as values
    pub fn to_ipld_metadata(&self) -> Result<Ipld> {
        let mut metadata = BTreeMap::new();

        for (name, info) in &self.tensors {
            let tensor_block = self.to_tensor_block(name)?;
            let cid = *tensor_block.cid();

            let mut tensor_meta = BTreeMap::new();
            tensor_meta.insert("dtype".to_string(), Ipld::String(info.dtype.clone()));
            tensor_meta.insert(
                "shape".to_string(),
                Ipld::List(
                    info.shape
                        .iter()
                        .map(|&s| Ipld::Integer(s as i128))
                        .collect(),
                ),
            );
            tensor_meta.insert("data".to_string(), Ipld::Link(cid.into()));

            metadata.insert(name.clone(), Ipld::Map(tensor_meta));
        }

        Ok(Ipld::Map(metadata))
    }

    /// Get the total number of tensors
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Get the total size of the data section
    pub fn data_size(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_safetensors() -> Vec<u8> {
        // Create a simple Safetensors file with one F32 tensor [2, 2]
        let metadata = serde_json::json!({
            "weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 16]
            }
        });

        let header_bytes = serde_json::to_vec(&metadata).unwrap();
        let header_len = header_bytes.len() as u64;

        let mut file = Vec::new();
        // Header length (8 bytes)
        file.extend_from_slice(&header_len.to_le_bytes());
        // Header JSON
        file.extend_from_slice(&header_bytes);
        // Data: 4 floats (16 bytes)
        file.extend_from_slice(&1.0f32.to_le_bytes());
        file.extend_from_slice(&2.0f32.to_le_bytes());
        file.extend_from_slice(&3.0f32.to_le_bytes());
        file.extend_from_slice(&4.0f32.to_le_bytes());

        file
    }

    #[test]
    fn test_parse_safetensors() {
        let data = create_test_safetensors();
        let file = SafetensorsFile::parse(&data).unwrap();

        assert_eq!(file.tensor_count(), 1);
        assert!(file.get_tensor_info("weight").is_some());
    }

    #[test]
    fn test_tensor_info() {
        let data = create_test_safetensors();
        let file = SafetensorsFile::parse(&data).unwrap();

        let info = file.get_tensor_info("weight").unwrap();
        assert_eq!(info.dtype, "F32");
        assert_eq!(info.shape, vec![2, 2]);
        assert_eq!(info.data_offsets, [0, 16]);
        assert_eq!(info.size_bytes(), 16);
    }

    #[test]
    fn test_get_tensor_data() {
        let data = create_test_safetensors();
        let file = SafetensorsFile::parse(&data).unwrap();

        let tensor_data = file.get_tensor_data("weight").unwrap();
        assert_eq!(tensor_data.len(), 16);
    }

    #[test]
    fn test_to_tensor_block() {
        let data = create_test_safetensors();
        let file = SafetensorsFile::parse(&data).unwrap();

        let tensor = file.to_tensor_block("weight").unwrap();
        assert_eq!(tensor.element_count(), 4);
        assert_eq!(tensor.dtype(), TensorDtype::F32);
        assert_eq!(tensor.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_to_ipld_metadata() {
        let data = create_test_safetensors();
        let file = SafetensorsFile::parse(&data).unwrap();

        let ipld = file.to_ipld_metadata().unwrap();

        if let Ipld::Map(metadata) = ipld {
            assert!(metadata.contains_key("weight"));

            if let Some(Ipld::Map(tensor_meta)) = metadata.get("weight") {
                assert!(tensor_meta.contains_key("dtype"));
                assert!(tensor_meta.contains_key("shape"));
                assert!(tensor_meta.contains_key("data"));
            } else {
                panic!("Expected tensor metadata to be a map");
            }
        } else {
            panic!("Expected IPLD to be a map");
        }
    }

    #[test]
    fn test_invalid_safetensors() {
        // Too small
        let result = SafetensorsFile::parse(&[1, 2, 3]);
        assert!(result.is_err());

        // Invalid header
        let mut invalid = vec![0u8; 8];
        invalid.extend_from_slice(&100u64.to_le_bytes()[..8]);
        let result = SafetensorsFile::parse(&invalid);
        assert!(result.is_err());
    }

    #[test]
    fn test_dtype_conversion() {
        let info = SafetensorInfo {
            dtype: "F32".to_string(),
            shape: vec![2, 2],
            data_offsets: [0, 16],
        };

        assert_eq!(info.to_tensor_dtype().unwrap(), TensorDtype::F32);

        let invalid_info = SafetensorInfo {
            dtype: "INVALID".to_string(),
            shape: vec![2, 2],
            data_offsets: [0, 16],
        };

        assert!(invalid_info.to_tensor_dtype().is_err());
    }

    #[test]
    fn test_multiple_tensors() {
        let metadata = serde_json::json!({
            "weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 16]
            },
            "bias": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [16, 24]
            }
        });

        let header_bytes = serde_json::to_vec(&metadata).unwrap();
        let header_len = header_bytes.len() as u64;

        let mut file = Vec::new();
        file.extend_from_slice(&header_len.to_le_bytes());
        file.extend_from_slice(&header_bytes);
        // weight data: 4 floats
        file.extend_from_slice(&1.0f32.to_le_bytes());
        file.extend_from_slice(&2.0f32.to_le_bytes());
        file.extend_from_slice(&3.0f32.to_le_bytes());
        file.extend_from_slice(&4.0f32.to_le_bytes());
        // bias data: 2 floats
        file.extend_from_slice(&0.1f32.to_le_bytes());
        file.extend_from_slice(&0.2f32.to_le_bytes());

        let parsed = SafetensorsFile::parse(&file).unwrap();
        assert_eq!(parsed.tensor_count(), 2);

        let weight = parsed.to_tensor_block("weight").unwrap();
        assert_eq!(weight.element_count(), 4);

        let bias = parsed.to_tensor_block("bias").unwrap();
        assert_eq!(bias.element_count(), 2);
    }
}
