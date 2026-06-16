//! Safetensors format support for efficient model storage
//!
//! Provides native support for the Safetensors format:
//! - Parse .safetensors files
//! - Extract metadata and tensor information
//! - Store tensors as content-addressed blocks
//! - Chunked storage for large models (70B+ parameters)
//! - Lazy loading of model weights
//!
//! # Example
//!
//! ```rust,ignore
//! use ipfrs_storage::{SafetensorsStore, SledBlockStore, BlockStoreConfig};
//! use std::sync::Arc;
//! use std::path::PathBuf;
//!
//! # async fn example() -> ipfrs_core::Result<()> {
//! // Create block store
//! let store = Arc::new(SledBlockStore::new(BlockStoreConfig {
//!     path: PathBuf::from(".ipfrs/models"),
//!     cache_size: 1024 * 1024 * 1024, // 1GB cache
//! })?);
//!
//! // Create safetensors store
//! let safetensors_store = SafetensorsStore::new(store);
//!
//! // Load and store a safetensors file
//! let model_cid = safetensors_store.import_file("model.safetensors").await?;
//!
//! // Lazy load a specific tensor
//! let tensor_data = safetensors_store.load_tensor(&model_cid, "layer.0.weight").await?;
//! # Ok(())
//! # }
//! ```

use crate::traits::BlockStore;
use bytes::Bytes;
use ipfrs_core::{Block, Cid, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;

/// Tensor data type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DType {
    F32,
    F64,
    F16,
    BF16,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
}

impl DType {
    /// Get size in bytes for this dtype
    pub fn size(&self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F64 | DType::I64 | DType::U64 => 8,
            DType::F16 | DType::BF16 | DType::I16 | DType::U16 => 2,
            DType::I8 | DType::U8 | DType::Bool => 1,
        }
    }
}

impl FromStr for DType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "F32" => Ok(DType::F32),
            "F64" => Ok(DType::F64),
            "F16" => Ok(DType::F16),
            "BF16" => Ok(DType::BF16),
            "I8" => Ok(DType::I8),
            "I16" => Ok(DType::I16),
            "I32" => Ok(DType::I32),
            "I64" => Ok(DType::I64),
            "U8" => Ok(DType::U8),
            "U16" => Ok(DType::U16),
            "U32" => Ok(DType::U32),
            "U64" => Ok(DType::U64),
            "BOOL" => Ok(DType::Bool),
            _ => Err(format!("Unknown dtype: {s}")),
        }
    }
}

/// Tensor metadata from safetensors header
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TensorInfo {
    /// Data type of the tensor
    pub dtype: DType,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Start offset in the data section
    pub data_offsets: (usize, usize),
}

impl TensorInfo {
    /// Calculate total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Calculate total size in bytes
    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.size()
    }
}

/// Safetensors file header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsHeader {
    /// Tensor metadata by name
    pub tensors: HashMap<String, TensorInfo>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Chunked tensor storage for large tensors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkedTensor {
    /// Tensor name
    pub name: String,
    /// Tensor metadata
    pub info: TensorInfo,
    /// CIDs of chunks (in order)
    #[serde(
        serialize_with = "serialize_cid_vec",
        deserialize_with = "deserialize_cid_vec"
    )]
    pub chunk_cids: Vec<Cid>,
    /// Size of each chunk in bytes
    pub chunk_size: usize,
}

// Custom serialization for Vec<Cid>
fn serialize_cid_vec<S>(cids: &[Cid], serializer: S) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeSeq;
    let mut seq = serializer.serialize_seq(Some(cids.len()))?;
    for cid in cids {
        seq.serialize_element(&cid.to_bytes())?;
    }
    seq.end()
}

fn deserialize_cid_vec<'de, D>(deserializer: D) -> std::result::Result<Vec<Cid>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let bytes_vec: Vec<Vec<u8>> = Deserialize::deserialize(deserializer)?;
    bytes_vec
        .into_iter()
        .map(|bytes| Cid::try_from(bytes).map_err(serde::de::Error::custom))
        .collect()
}

/// Safetensors model manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsManifest {
    /// Model name
    pub name: String,
    /// Safetensors header
    pub header: SafetensorsHeader,
    /// Chunked tensors
    pub tensors: HashMap<String, ChunkedTensor>,
    /// Total model size in bytes
    pub total_size: u64,
}

/// Configuration for chunked storage
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Chunk size in bytes (default: 64MB)
    pub chunk_size: usize,
    /// Whether to compress chunks
    pub compress: bool,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 64 * 1024 * 1024, // 64MB
            compress: false,
        }
    }
}

/// Safetensors store for managing model weights
pub struct SafetensorsStore<S: BlockStore> {
    /// Underlying block store
    store: Arc<S>,
    /// Chunk configuration
    chunk_config: ChunkConfig,
}

impl<S: BlockStore> SafetensorsStore<S> {
    /// Create a new safetensors store
    pub fn new(store: Arc<S>) -> Self {
        Self {
            store,
            chunk_config: ChunkConfig::default(),
        }
    }

    /// Create with custom chunk configuration
    pub fn with_config(store: Arc<S>, chunk_config: ChunkConfig) -> Self {
        Self {
            store,
            chunk_config,
        }
    }

    /// Parse safetensors header from bytes
    pub fn parse_header(data: &[u8]) -> Result<(SafetensorsHeader, usize)> {
        if data.len() < 8 {
            return Err(Error::Storage(
                "File too small to be safetensors".to_string(),
            ));
        }

        // Read header size (8 bytes, little-endian u64)
        let header_size = u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]) as usize;

        if data.len() < 8 + header_size {
            return Err(Error::Storage("Incomplete safetensors header".to_string()));
        }

        // Parse JSON header
        let header_bytes = &data[8..8 + header_size];
        let header_json: serde_json::Value = serde_json::from_slice(header_bytes)
            .map_err(|e| Error::Serialization(format!("Failed to parse header JSON: {e}")))?;

        let mut tensors = HashMap::new();
        let mut metadata = HashMap::new();

        // Parse tensors
        if let Some(obj) = header_json.as_object() {
            for (key, value) in obj {
                if key == "__metadata__" {
                    // Parse metadata
                    if let Some(meta_obj) = value.as_object() {
                        for (k, v) in meta_obj {
                            if let Some(s) = v.as_str() {
                                metadata.insert(k.clone(), s.to_string());
                            }
                        }
                    }
                } else {
                    // Parse tensor info
                    if let Some(tensor_obj) = value.as_object() {
                        let dtype_str = tensor_obj
                            .get("dtype")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| Error::Storage("Missing dtype".to_string()))?;

                        let dtype = dtype_str.parse::<DType>().map_err(Error::Storage)?;

                        let shape: Vec<usize> = tensor_obj
                            .get("shape")
                            .and_then(|v| v.as_array())
                            .ok_or_else(|| Error::Storage("Missing shape".to_string()))?
                            .iter()
                            .filter_map(|v| v.as_u64().map(|n| n as usize))
                            .collect();

                        let data_offsets = tensor_obj
                            .get("data_offsets")
                            .and_then(|v| v.as_array())
                            .ok_or_else(|| Error::Storage("Missing data_offsets".to_string()))?;

                        let start = data_offsets[0].as_u64().ok_or_else(|| {
                            Error::Storage("Invalid data_offsets start".to_string())
                        })? as usize;
                        let end = data_offsets[1]
                            .as_u64()
                            .ok_or_else(|| Error::Storage("Invalid data_offsets end".to_string()))?
                            as usize;

                        tensors.insert(
                            key.clone(),
                            TensorInfo {
                                dtype,
                                shape,
                                data_offsets: (start, end),
                            },
                        );
                    }
                }
            }
        }

        Ok((SafetensorsHeader { tensors, metadata }, 8 + header_size))
    }

    /// Import safetensors file and store as chunks
    pub async fn import_from_bytes(&self, name: String, data: &[u8]) -> Result<Cid> {
        // Parse header
        let (header, data_offset) = Self::parse_header(data)?;

        let data_section = &data[data_offset..];
        let mut chunked_tensors = HashMap::new();
        let mut total_size = 0u64;

        // Process each tensor
        for (tensor_name, tensor_info) in &header.tensors {
            let (start, end) = tensor_info.data_offsets;
            let tensor_data = &data_section[start..end];

            // Chunk the tensor data
            let mut chunk_cids = Vec::new();
            for chunk in tensor_data.chunks(self.chunk_config.chunk_size) {
                let block = Block::new(Bytes::from(chunk.to_vec()))?;
                let cid = *block.cid();
                self.store.put(&block).await?;
                chunk_cids.push(cid);
            }

            chunked_tensors.insert(
                tensor_name.clone(),
                ChunkedTensor {
                    name: tensor_name.clone(),
                    info: tensor_info.clone(),
                    chunk_cids,
                    chunk_size: self.chunk_config.chunk_size,
                },
            );

            total_size += tensor_data.len() as u64;
        }

        // Create manifest
        let manifest = SafetensorsManifest {
            name,
            header,
            tensors: chunked_tensors,
            total_size,
        };

        // Store manifest
        let manifest_bytes = oxicode::serde::encode_to_vec(&manifest, oxicode::config::standard())
            .map_err(|e| Error::Serialization(format!("Failed to serialize manifest: {e}")))?;

        let manifest_block = Block::new(Bytes::from(manifest_bytes))?;
        let manifest_cid = *manifest_block.cid();
        self.store.put(&manifest_block).await?;

        Ok(manifest_cid)
    }

    /// Load safetensors manifest
    pub async fn load_manifest(&self, manifest_cid: &Cid) -> Result<SafetensorsManifest> {
        let block = self
            .store
            .get(manifest_cid)
            .await?
            .ok_or_else(|| Error::NotFound(format!("Manifest not found: {manifest_cid}")))?;

        let manifest: SafetensorsManifest =
            oxicode::serde::decode_owned_from_slice(block.data(), oxicode::config::standard())
                .map(|(v, _)| v)
                .map_err(|e| {
                    Error::Serialization(format!("Failed to deserialize manifest: {e}"))
                })?;

        Ok(manifest)
    }

    /// Load a specific tensor (lazy loading)
    pub async fn load_tensor(&self, manifest_cid: &Cid, tensor_name: &str) -> Result<Vec<u8>> {
        let manifest = self.load_manifest(manifest_cid).await?;

        let chunked_tensor = manifest
            .tensors
            .get(tensor_name)
            .ok_or_else(|| Error::NotFound(format!("Tensor not found: {tensor_name}")))?;

        // Load all chunks
        let mut tensor_data = Vec::with_capacity(chunked_tensor.info.size_bytes());

        for chunk_cid in &chunked_tensor.chunk_cids {
            let chunk_block = self
                .store
                .get(chunk_cid)
                .await?
                .ok_or_else(|| Error::NotFound(format!("Chunk not found: {chunk_cid}")))?;

            tensor_data.extend_from_slice(chunk_block.data());
        }

        Ok(tensor_data)
    }

    /// Load multiple tensors (batch loading for efficiency)
    pub async fn load_tensors(
        &self,
        manifest_cid: &Cid,
        tensor_names: &[&str],
    ) -> Result<HashMap<String, Vec<u8>>> {
        let _manifest = self.load_manifest(manifest_cid).await?;
        let mut result = HashMap::new();

        for &tensor_name in tensor_names {
            let tensor_data = self.load_tensor(manifest_cid, tensor_name).await?;
            result.insert(tensor_name.to_string(), tensor_data);
        }

        Ok(result)
    }

    /// Get tensor metadata without loading data
    pub async fn get_tensor_info(
        &self,
        manifest_cid: &Cid,
        tensor_name: &str,
    ) -> Result<TensorInfo> {
        let manifest = self.load_manifest(manifest_cid).await?;

        manifest
            .tensors
            .get(tensor_name)
            .map(|ct| ct.info.clone())
            .ok_or_else(|| Error::NotFound(format!("Tensor not found: {tensor_name}")))
    }

    /// List all tensors in the model
    pub async fn list_tensors(&self, manifest_cid: &Cid) -> Result<Vec<String>> {
        let manifest = self.load_manifest(manifest_cid).await?;
        Ok(manifest.tensors.keys().cloned().collect())
    }

    /// Get model statistics
    pub async fn get_model_stats(&self, manifest_cid: &Cid) -> Result<ModelStats> {
        let manifest = self.load_manifest(manifest_cid).await?;

        let tensor_count = manifest.tensors.len();
        let total_parameters: usize = manifest.tensors.values().map(|ct| ct.info.numel()).sum();

        let chunk_count: usize = manifest
            .tensors
            .values()
            .map(|ct| ct.chunk_cids.len())
            .sum();

        Ok(ModelStats {
            name: manifest.name,
            tensor_count,
            total_parameters,
            total_size_bytes: manifest.total_size,
            chunk_count,
            avg_chunk_size: if chunk_count > 0 {
                manifest.total_size / chunk_count as u64
            } else {
                0
            },
        })
    }
}

/// Model statistics
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelStats {
    /// Model name
    pub name: String,
    /// Number of tensors
    pub tensor_count: usize,
    /// Total number of parameters
    pub total_parameters: usize,
    /// Total size in bytes
    pub total_size_bytes: u64,
    /// Number of chunks
    pub chunk_count: usize,
    /// Average chunk size
    pub avg_chunk_size: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockstore::{BlockStoreConfig, SledBlockStore};

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::F32.size(), 4);
        assert_eq!(DType::F64.size(), 8);
        assert_eq!(DType::F16.size(), 2);
        assert_eq!(DType::I8.size(), 1);
    }

    #[test]
    fn test_tensor_info_numel() {
        let info = TensorInfo {
            dtype: DType::F32,
            shape: vec![2, 3, 4],
            data_offsets: (0, 96),
        };

        assert_eq!(info.numel(), 24);
        assert_eq!(info.size_bytes(), 96);
    }

    #[tokio::test]
    async fn test_safetensors_store() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-safetensors-test"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);

        let store = Arc::new(
            SledBlockStore::new(config)
                .expect("failed to create SledBlockStore for safetensors test"),
        );
        let safetensors_store = SafetensorsStore::new(store);

        // Create a minimal safetensors file
        let header = r#"{"tensor1":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let header_size = header.len() as u64;
        let mut data = Vec::new();
        data.extend_from_slice(&header_size.to_le_bytes());
        data.extend_from_slice(header.as_bytes());
        // Add tensor data (2x2 f32 = 16 bytes)
        data.extend_from_slice(&[0u8; 16]);

        let manifest_cid = safetensors_store
            .import_from_bytes("test_model".to_string(), &data)
            .await
            .expect("test: import_from_bytes should succeed");

        // Load manifest
        let manifest = safetensors_store
            .load_manifest(&manifest_cid)
            .await
            .expect("test: load_manifest should succeed");
        assert_eq!(manifest.name, "test_model");
        assert_eq!(manifest.tensors.len(), 1);

        // Get stats
        let stats = safetensors_store
            .get_model_stats(&manifest_cid)
            .await
            .expect("test: get_model_stats should succeed");
        assert_eq!(stats.tensor_count, 1);
        assert_eq!(stats.total_parameters, 4);
    }

    #[test]
    fn test_parse_header() {
        let header = r#"{"tensor1":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let header_size = header.len() as u64;
        let mut data = Vec::new();
        data.extend_from_slice(&header_size.to_le_bytes());
        data.extend_from_slice(header.as_bytes());

        let (parsed, offset) = SafetensorsStore::<SledBlockStore>::parse_header(&data)
            .expect("test: parse_header should succeed on valid header");
        assert_eq!(offset, 8 + header.len());
        assert_eq!(parsed.tensors.len(), 1);
        assert!(parsed.tensors.contains_key("tensor1"));

        let tensor_info = &parsed.tensors["tensor1"];
        assert_eq!(tensor_info.dtype, DType::F32);
        assert_eq!(tensor_info.shape, vec![2, 2]);
        assert_eq!(tensor_info.data_offsets, (0, 16));
    }
}
