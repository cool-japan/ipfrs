//! Tensor-aware block types for neural network data.
//!
//! This module provides specialized types for storing and managing tensor data
//! in a content-addressed manner. Tensors are the fundamental data structure
//! in machine learning frameworks like PyTorch and TensorFlow.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_core::tensor::{TensorBlock, TensorDtype, TensorShape};
//! use bytes::Bytes;
//!
//! // Create a 2x3 f32 tensor
//! let shape = TensorShape::new(vec![2, 3]);
//! let data = Bytes::from(vec![
//!     0f32.to_le_bytes(), 1f32.to_le_bytes(),
//!     2f32.to_le_bytes(), 3f32.to_le_bytes(),
//!     4f32.to_le_bytes(), 5f32.to_le_bytes(),
//! ].concat());
//!
//! let tensor = TensorBlock::new(data, shape, TensorDtype::F32).unwrap();
//! assert_eq!(tensor.element_count(), 6);
//! ```

use crate::block::Block;
use crate::error::{Error, Result};
use bytes::Bytes;
use serde::{Deserialize, Serialize};

/// Supported tensor data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorDtype {
    /// 32-bit floating point (IEEE 754)
    F32,
    /// 16-bit floating point (IEEE 754-2008)
    F16,
    /// 64-bit floating point (IEEE 754)
    F64,
    /// 8-bit signed integer
    I8,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 8-bit unsigned integer
    U8,
    /// 32-bit unsigned integer
    U32,
    /// Boolean (1 byte)
    Bool,
}

impl TensorDtype {
    /// Get the size in bytes of this data type
    #[inline]
    pub fn size_bytes(&self) -> usize {
        match self {
            TensorDtype::F32 => 4,
            TensorDtype::F16 => 2,
            TensorDtype::F64 => 8,
            TensorDtype::I8 => 1,
            TensorDtype::I32 => 4,
            TensorDtype::I64 => 8,
            TensorDtype::U8 => 1,
            TensorDtype::U32 => 4,
            TensorDtype::Bool => 1,
        }
    }

    /// Get a human-readable name for this data type
    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            TensorDtype::F32 => "float32",
            TensorDtype::F16 => "float16",
            TensorDtype::F64 => "float64",
            TensorDtype::I8 => "int8",
            TensorDtype::I32 => "int32",
            TensorDtype::I64 => "int64",
            TensorDtype::U8 => "uint8",
            TensorDtype::U32 => "uint32",
            TensorDtype::Bool => "bool",
        }
    }
}

/// Tensor shape (dimensions)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorShape {
    dims: Vec<usize>,
}

impl TensorShape {
    /// Create a new tensor shape
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Create a scalar (0-dimensional tensor)
    pub fn scalar() -> Self {
        Self { dims: vec![] }
    }

    /// Get the dimensions
    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Get the rank (number of dimensions)
    #[inline]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Calculate the total number of elements
    #[inline]
    pub fn element_count(&self) -> usize {
        if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().product()
        }
    }

    /// Check if this is a scalar
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// Check if this is a vector (1D)
    #[inline]
    pub fn is_vector(&self) -> bool {
        self.dims.len() == 1
    }

    /// Check if this is a matrix (2D)
    #[inline]
    pub fn is_matrix(&self) -> bool {
        self.dims.len() == 2
    }
}

/// Tensor metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    /// Tensor shape
    pub shape: TensorShape,
    /// Data type
    pub dtype: TensorDtype,
    /// Optional tensor name
    pub name: Option<String>,
    /// Optional additional metadata (e.g., gradient info, requires_grad)
    pub metadata: std::collections::BTreeMap<String, String>,
}

impl TensorMetadata {
    /// Create new tensor metadata
    pub fn new(shape: TensorShape, dtype: TensorDtype) -> Self {
        Self {
            shape,
            dtype,
            name: None,
            metadata: std::collections::BTreeMap::new(),
        }
    }

    /// Set the tensor name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Add custom metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Calculate expected data size in bytes
    pub fn expected_size(&self) -> usize {
        self.shape.element_count() * self.dtype.size_bytes()
    }
}

/// A content-addressed tensor block
///
/// Combines a regular [`Block`] with tensor-specific metadata like shape and dtype.
/// This allows storing neural network weights, activations, and gradients in a
/// content-addressed manner.
#[derive(Debug, Clone)]
pub struct TensorBlock {
    /// The underlying data block
    block: Block,
    /// Tensor metadata
    metadata: TensorMetadata,
}

impl TensorBlock {
    /// Create a new tensor block
    ///
    /// # Arguments
    ///
    /// * `data` - Raw tensor data (should be in native endian format)
    /// * `shape` - Tensor shape
    /// * `dtype` - Data type
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Data size doesn't match shape * dtype size
    /// - Block creation fails
    ///
    /// # Example
    ///
    /// ```rust
    /// use ipfrs_core::tensor::{TensorBlock, TensorDtype, TensorShape};
    /// use bytes::Bytes;
    ///
    /// let shape = TensorShape::new(vec![2, 2]);
    /// let data = Bytes::from(vec![1.0f32, 2.0, 3.0, 4.0]
    ///     .iter()
    ///     .flat_map(|f| f.to_le_bytes())
    ///     .collect::<Vec<u8>>());
    ///
    /// let tensor = TensorBlock::new(data, shape, TensorDtype::F32).unwrap();
    /// assert_eq!(tensor.element_count(), 4);
    /// ```
    pub fn new(data: Bytes, shape: TensorShape, dtype: TensorDtype) -> Result<Self> {
        let metadata = TensorMetadata::new(shape, dtype);

        // Validate data size
        let expected_size = metadata.expected_size();
        if data.len() != expected_size {
            return Err(Error::InvalidData(format!(
                "Tensor data size mismatch: expected {} bytes, got {}",
                expected_size,
                data.len()
            )));
        }

        // Create underlying block
        let block = Block::new(data)?;

        Ok(Self { block, metadata })
    }

    /// Create a tensor block with metadata
    pub fn with_metadata(data: Bytes, metadata: TensorMetadata) -> Result<Self> {
        let expected_size = metadata.expected_size();
        if data.len() != expected_size {
            return Err(Error::InvalidData(format!(
                "Tensor data size mismatch: expected {} bytes, got {}",
                expected_size,
                data.len()
            )));
        }

        let block = Block::new(data)?;
        Ok(Self { block, metadata })
    }

    /// Get the underlying block
    pub fn block(&self) -> &Block {
        &self.block
    }

    /// Get tensor metadata
    pub fn metadata(&self) -> &TensorMetadata {
        &self.metadata
    }

    /// Get the tensor shape
    pub fn shape(&self) -> &TensorShape {
        &self.metadata.shape
    }

    /// Get the data type
    pub fn dtype(&self) -> TensorDtype {
        self.metadata.dtype
    }

    /// Get the number of elements
    pub fn element_count(&self) -> usize {
        self.metadata.shape.element_count()
    }

    /// Get the CID of this tensor
    pub fn cid(&self) -> &crate::cid::Cid {
        self.block.cid()
    }

    /// Get the raw tensor data
    pub fn data(&self) -> &Bytes {
        self.block.data()
    }

    /// Consume and return the underlying block and metadata
    pub fn into_parts(self) -> (Block, TensorMetadata) {
        (self.block, self.metadata)
    }

    /// Verify the tensor block integrity
    pub fn verify(&self) -> Result<bool> {
        self.block.verify()
    }

    /// Reshape the tensor to a new shape (must have same element count)
    pub fn reshape(&self, new_shape: TensorShape) -> Result<Self> {
        if new_shape.element_count() != self.element_count() {
            return Err(Error::InvalidInput(format!(
                "Cannot reshape tensor with {} elements to shape with {} elements",
                self.element_count(),
                new_shape.element_count()
            )));
        }

        let new_metadata = TensorMetadata {
            shape: new_shape,
            dtype: self.metadata.dtype,
            name: self.metadata.name.clone(),
            metadata: self.metadata.metadata.clone(),
        };

        Ok(Self {
            block: self.block.clone(),
            metadata: new_metadata,
        })
    }

    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data().len()
    }

    /// Check if this is a scalar tensor (0-dimensional)
    pub fn is_scalar(&self) -> bool {
        self.shape().is_scalar()
    }

    /// Check if this is a vector (1-dimensional)
    pub fn is_vector(&self) -> bool {
        self.shape().is_vector()
    }

    /// Check if this is a matrix (2-dimensional)
    pub fn is_matrix(&self) -> bool {
        self.shape().is_matrix()
    }
}

/// Utility functions for creating tensors from typed data
impl TensorBlock {
    /// Create a tensor from a slice of f32 values
    pub fn from_f32_slice(data: &[f32], shape: TensorShape) -> Result<Self> {
        if data.len() != shape.element_count() {
            return Err(Error::InvalidInput(format!(
                "Data length {} doesn't match shape element count {}",
                data.len(),
                shape.element_count()
            )));
        }

        let bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();
        Self::new(Bytes::from(bytes), shape, TensorDtype::F32)
    }

    /// Create a tensor from a slice of f64 values
    pub fn from_f64_slice(data: &[f64], shape: TensorShape) -> Result<Self> {
        if data.len() != shape.element_count() {
            return Err(Error::InvalidInput(format!(
                "Data length {} doesn't match shape element count {}",
                data.len(),
                shape.element_count()
            )));
        }

        let bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();
        Self::new(Bytes::from(bytes), shape, TensorDtype::F64)
    }

    /// Create a tensor from a slice of i32 values
    pub fn from_i32_slice(data: &[i32], shape: TensorShape) -> Result<Self> {
        if data.len() != shape.element_count() {
            return Err(Error::InvalidInput(format!(
                "Data length {} doesn't match shape element count {}",
                data.len(),
                shape.element_count()
            )));
        }

        let bytes: Vec<u8> = data.iter().flat_map(|&i| i.to_le_bytes()).collect();
        Self::new(Bytes::from(bytes), shape, TensorDtype::I32)
    }

    /// Create a tensor from a slice of i64 values
    pub fn from_i64_slice(data: &[i64], shape: TensorShape) -> Result<Self> {
        if data.len() != shape.element_count() {
            return Err(Error::InvalidInput(format!(
                "Data length {} doesn't match shape element count {}",
                data.len(),
                shape.element_count()
            )));
        }

        let bytes: Vec<u8> = data.iter().flat_map(|&i| i.to_le_bytes()).collect();
        Self::new(Bytes::from(bytes), shape, TensorDtype::I64)
    }

    /// Create a tensor from a slice of u8 values
    pub fn from_u8_slice(data: &[u8], shape: TensorShape) -> Result<Self> {
        if data.len() != shape.element_count() {
            return Err(Error::InvalidInput(format!(
                "Data length {} doesn't match shape element count {}",
                data.len(),
                shape.element_count()
            )));
        }

        Self::new(Bytes::copy_from_slice(data), shape, TensorDtype::U8)
    }

    /// Convert tensor data to a Vec of f32 values (if dtype is F32)
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        if self.dtype() != TensorDtype::F32 {
            return Err(Error::InvalidInput(format!(
                "Cannot convert {} tensor to f32",
                self.dtype().name()
            )));
        }

        let data = self.data();
        let mut result = Vec::with_capacity(self.element_count());

        for chunk in data.chunks_exact(4) {
            let bytes: [u8; 4] = chunk
                .try_into()
                .expect("chunks_exact(4) guarantees exactly 4 bytes");
            result.push(f32::from_le_bytes(bytes));
        }

        Ok(result)
    }

    /// Convert tensor data to a Vec of f64 values (if dtype is F64)
    pub fn to_f64_vec(&self) -> Result<Vec<f64>> {
        if self.dtype() != TensorDtype::F64 {
            return Err(Error::InvalidInput(format!(
                "Cannot convert {} tensor to f64",
                self.dtype().name()
            )));
        }

        let data = self.data();
        let mut result = Vec::with_capacity(self.element_count());

        for chunk in data.chunks_exact(8) {
            let bytes: [u8; 8] = chunk
                .try_into()
                .expect("chunks_exact(8) guarantees exactly 8 bytes");
            result.push(f64::from_le_bytes(bytes));
        }

        Ok(result)
    }

    /// Convert tensor data to a Vec of i32 values (if dtype is I32)
    pub fn to_i32_vec(&self) -> Result<Vec<i32>> {
        if self.dtype() != TensorDtype::I32 {
            return Err(Error::InvalidInput(format!(
                "Cannot convert {} tensor to i32",
                self.dtype().name()
            )));
        }

        let data = self.data();
        let mut result = Vec::with_capacity(self.element_count());

        for chunk in data.chunks_exact(4) {
            let bytes: [u8; 4] = chunk
                .try_into()
                .expect("chunks_exact(4) guarantees exactly 4 bytes");
            result.push(i32::from_le_bytes(bytes));
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_dtype_sizes() {
        assert_eq!(TensorDtype::F32.size_bytes(), 4);
        assert_eq!(TensorDtype::F16.size_bytes(), 2);
        assert_eq!(TensorDtype::I8.size_bytes(), 1);
        assert_eq!(TensorDtype::I32.size_bytes(), 4);
    }

    #[test]
    fn test_tensor_shape() {
        let shape = TensorShape::new(vec![2, 3, 4]);
        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.element_count(), 24);
        assert!(!shape.is_scalar());
        assert!(!shape.is_vector());
        assert!(!shape.is_matrix());

        let scalar = TensorShape::scalar();
        assert!(scalar.is_scalar());
        assert_eq!(scalar.element_count(), 1);
    }

    #[test]
    fn test_tensor_block_creation() {
        let shape = TensorShape::new(vec![2, 2]);
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let tensor = TensorBlock::new(Bytes::from(data), shape, TensorDtype::F32).unwrap();

        assert_eq!(tensor.element_count(), 4);
        assert_eq!(tensor.dtype(), TensorDtype::F32);
        assert_eq!(tensor.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_tensor_size_validation() {
        let shape = TensorShape::new(vec![2, 2]);
        // Too small data (only 3 floats instead of 4)
        let data: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let result = TensorBlock::new(Bytes::from(data), shape, TensorDtype::F32);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_metadata() {
        let shape = TensorShape::new(vec![10, 20]);
        let metadata = TensorMetadata::new(shape, TensorDtype::F32)
            .with_name("layer1.weight".to_string())
            .with_metadata("requires_grad".to_string(), "true".to_string());

        assert_eq!(metadata.name, Some("layer1.weight".to_string()));
        assert_eq!(metadata.expected_size(), 10 * 20 * 4); // 800 bytes
    }

    #[test]
    fn test_tensor_from_f32_slice() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = TensorShape::new(vec![2, 3]);

        let tensor = TensorBlock::from_f32_slice(&data, shape).unwrap();
        assert_eq!(tensor.element_count(), 6);
        assert_eq!(tensor.dtype(), TensorDtype::F32);

        // Roundtrip test
        let recovered = tensor.to_f32_vec().unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_tensor_from_i32_slice() {
        let data = vec![10i32, 20, 30, 40];
        let shape = TensorShape::new(vec![2, 2]);

        let tensor = TensorBlock::from_i32_slice(&data, shape).unwrap();
        assert_eq!(tensor.element_count(), 4);

        let recovered = tensor.to_i32_vec().unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_tensor_reshape() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = TensorShape::new(vec![2, 3]);
        let tensor = TensorBlock::from_f32_slice(&data, shape).unwrap();

        // Reshape 2x3 to 3x2
        let reshaped = tensor.reshape(TensorShape::new(vec![3, 2])).unwrap();
        assert_eq!(reshaped.shape().dims(), &[3, 2]);
        assert_eq!(reshaped.element_count(), 6);

        // Verify data is preserved
        let recovered = reshaped.to_f32_vec().unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_tensor_reshape_invalid() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = TensorShape::new(vec![2, 2]);
        let tensor = TensorBlock::from_f32_slice(&data, shape).unwrap();

        // Try to reshape to incompatible shape
        let result = tensor.reshape(TensorShape::new(vec![3, 2])); // 6 elements != 4
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_type_checks() {
        let data = vec![1.0f32, 2.0];
        let tensor = TensorBlock::from_f32_slice(&data, TensorShape::new(vec![2])).unwrap();
        assert!(tensor.is_vector());
        assert!(!tensor.is_matrix());
        assert!(!tensor.is_scalar());

        let matrix = TensorBlock::from_f32_slice(&data, TensorShape::new(vec![1, 2])).unwrap();
        assert!(matrix.is_matrix());
    }

    #[test]
    fn test_tensor_to_vec_wrong_dtype() {
        let data = vec![1i32, 2, 3];
        let tensor = TensorBlock::from_i32_slice(&data, TensorShape::new(vec![3])).unwrap();

        // Try to convert i32 tensor to f32
        let result = tensor.to_f32_vec();
        assert!(result.is_err());
    }
}
