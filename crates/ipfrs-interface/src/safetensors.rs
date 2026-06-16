//! Safetensors integration for tensor serialization
//!
//! Provides utilities for working with safetensors format:
//! - Parsing and validating safetensors files
//! - Extracting tensor metadata
//! - Reading tensor data
//! - Creating safetensors from raw tensors

use bytes::Bytes;
use ipfrs_core::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Safetensors file format handler
#[derive(Debug)]
pub struct SafetensorsFile {
    /// Parsed header with tensor metadata
    header: SafetensorsHeader,
    /// Raw data bytes (header + tensors)
    data: Bytes,
    /// Header size in bytes
    header_size: usize,
}

/// Safetensors header structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsHeader {
    /// Tensor metadata indexed by name
    #[serde(flatten)]
    pub tensors: HashMap<String, TensorInfo>,
}

/// Information about a single tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Data type (e.g., "F32", "F64", "I32")
    pub dtype: String,
    /// Tensor shape (dimensions)
    pub shape: Vec<usize>,
    /// Start offset in the data section
    pub data_offsets: [usize; 2], // [start, end]
}

impl SafetensorsFile {
    /// Parse a safetensors file from bytes
    pub fn from_bytes(data: Bytes) -> Result<Self> {
        if data.len() < 8 {
            return Err(Error::InvalidInput(
                "Data too short for safetensors format".to_string(),
            ));
        }

        // First 8 bytes = header length (little-endian u64)
        let header_len = u64::from_le_bytes(
            data[0..8]
                .try_into()
                .expect("data[0..8] is exactly 8 bytes after bounds check"),
        ) as usize;

        if data.len() < 8 + header_len {
            return Err(Error::InvalidInput(
                "Incomplete safetensors header".to_string(),
            ));
        }

        // Parse JSON header
        let header_bytes = &data[8..8 + header_len];
        let header: SafetensorsHeader = serde_json::from_slice(header_bytes).map_err(|e| {
            Error::InvalidInput(format!("Failed to parse safetensors header: {}", e))
        })?;

        // Validate header
        Self::validate_header(&header, data.len() - 8 - header_len)?;

        Ok(SafetensorsFile {
            header,
            data,
            header_size: 8 + header_len,
        })
    }

    /// Validate header offsets and data integrity
    fn validate_header(header: &SafetensorsHeader, data_section_size: usize) -> Result<()> {
        for (name, info) in &header.tensors {
            let [start, end] = info.data_offsets;

            if start >= end {
                return Err(Error::InvalidInput(format!(
                    "Invalid offsets for tensor '{}': start={}, end={}",
                    name, start, end
                )));
            }

            if end > data_section_size {
                return Err(Error::InvalidInput(format!(
                    "Tensor '{}' offset {} exceeds data section size {}",
                    name, end, data_section_size
                )));
            }

            // Validate size matches shape and dtype
            let expected_size = Self::calculate_tensor_size(&info.shape, &info.dtype);
            let actual_size = end - start;

            if actual_size != expected_size {
                return Err(Error::InvalidInput(format!(
                    "Tensor '{}' size mismatch: expected {}, got {}",
                    name, expected_size, actual_size
                )));
            }
        }

        Ok(())
    }

    /// Calculate expected tensor size in bytes
    fn calculate_tensor_size(shape: &[usize], dtype: &str) -> usize {
        let num_elements: usize = shape.iter().product();
        let element_size = Self::dtype_size(dtype);
        num_elements * element_size
    }

    /// Get size of a data type in bytes
    fn dtype_size(dtype: &str) -> usize {
        match dtype {
            "F16" | "BF16" => 2,
            "F32" | "I32" | "U32" => 4,
            "F64" | "I64" | "U64" => 8,
            "I8" | "U8" => 1,
            "I16" | "U16" => 2,
            "BOOL" => 1,
            _ => 4, // Default to 4 bytes
        }
    }

    /// Get tensor data by name
    pub fn get_tensor(&self, name: &str) -> Result<TensorData> {
        let info = self.header.tensors.get(name).ok_or_else(|| {
            Error::NotFound(format!("Tensor '{}' not found in safetensors file", name))
        })?;

        let [start, end] = info.data_offsets;
        let data_start = self.header_size + start;
        let data_end = self.header_size + end;

        if data_end > self.data.len() {
            return Err(Error::InvalidInput(format!(
                "Tensor data range {}..{} exceeds file size {}",
                data_start,
                data_end,
                self.data.len()
            )));
        }

        Ok(TensorData {
            dtype: info.dtype.clone(),
            shape: info.shape.clone(),
            data: self.data.slice(data_start..data_end),
        })
    }

    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<String> {
        self.header
            .tensors
            .keys()
            .filter(|k| k.as_str() != "__metadata__")
            .cloned()
            .collect()
    }

    /// Get tensor metadata by name
    pub fn get_tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.header.tensors.get(name)
    }

    /// Get the full header
    pub fn header(&self) -> &SafetensorsHeader {
        &self.header
    }

    /// Get raw file data
    pub fn raw_data(&self) -> &Bytes {
        &self.data
    }
}

/// Tensor data extracted from safetensors
#[derive(Debug, Clone)]
pub struct TensorData {
    /// Data type
    pub dtype: String,
    /// Shape (dimensions)
    pub shape: Vec<usize>,
    /// Raw tensor data
    pub data: Bytes,
}

impl TensorData {
    /// Get the number of elements in the tensor
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Get element size in bytes
    pub fn element_size(&self) -> usize {
        if self.num_elements() == 0 {
            return 0;
        }
        self.size_bytes() / self.num_elements()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(SafetensorsFile::dtype_size("F32"), 4);
        assert_eq!(SafetensorsFile::dtype_size("F64"), 8);
        assert_eq!(SafetensorsFile::dtype_size("F16"), 2);
        assert_eq!(SafetensorsFile::dtype_size("I32"), 4);
        assert_eq!(SafetensorsFile::dtype_size("U8"), 1);
        assert_eq!(SafetensorsFile::dtype_size("BOOL"), 1);
    }

    #[test]
    fn test_calculate_tensor_size() {
        assert_eq!(
            SafetensorsFile::calculate_tensor_size(&[10, 20], "F32"),
            10 * 20 * 4
        );
        assert_eq!(
            SafetensorsFile::calculate_tensor_size(&[5, 5, 5], "F64"),
            5 * 5 * 5 * 8
        );
    }

    #[test]
    fn test_tensor_data_num_elements() {
        let data = TensorData {
            dtype: "F32".to_string(),
            shape: vec![2, 3],
            data: Bytes::from(vec![0u8; 24]), // 2*3*4 = 24 bytes
        };
        assert_eq!(data.num_elements(), 6);
        assert_eq!(data.size_bytes(), 24);
        assert_eq!(data.element_size(), 4);
    }
}
