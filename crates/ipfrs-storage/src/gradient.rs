//! Gradient and tensor storage with delta encoding
//!
//! Provides efficient storage for neural network gradients and tensors:
//! - Delta encoding (store changes only)
//! - Sparse gradient compression
//! - Provenance metadata tracking
//! - Integration with version control
//!
//! # Example
//!
//! ```rust,ignore
//! use ipfrs_storage::{GradientStore, DeltaEncoder, SledBlockStore, BlockStoreConfig};
//! use std::sync::Arc;
//! use std::path::PathBuf;
//!
//! # async fn example() -> ipfrs_core::Result<()> {
//! // Create block store
//! let store = Arc::new(SledBlockStore::new(BlockStoreConfig {
//!     path: PathBuf::from(".ipfrs/gradients"),
//!     cache_size: 100 * 1024 * 1024,
//! })?);
//!
//! // Create gradient store
//! let gradient_store = GradientStore::new(store);
//!
//! // Store a gradient with delta encoding
//! let gradient = vec![1.0f32, 2.0, 3.0, 4.0];
//! let metadata = ProvenanceMetadata {
//!     layer: "layer1".to_string(),
//!     timestamp: 1234567890,
//!     training_config: "lr=0.001".to_string(),
//! };
//!
//! let cid = gradient_store.store_gradient(&gradient, Some(metadata)).await?;
//! # Ok(())
//! # }
//! ```

use crate::traits::BlockStore;
use crate::vcs::VersionControl;
use bytes::Bytes;
use ipfrs_core::{Block, Cid, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Provenance metadata for tracking gradient origins
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProvenanceMetadata {
    /// Layer name or identifier
    pub layer: String,
    /// Unix timestamp when gradient was computed
    pub timestamp: u64,
    /// Training configuration (hyperparameters, etc.)
    pub training_config: String,
    /// Optional parent gradient CID (for delta encoding)
    #[serde(
        serialize_with = "serialize_option_cid",
        deserialize_with = "deserialize_option_cid"
    )]
    pub parent: Option<Cid>,
    /// Training step/epoch number
    pub step: Option<u64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

// Custom serialization for Option<Cid>
fn serialize_option_cid<S>(cid: &Option<Cid>, serializer: S) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match cid {
        Some(c) => serializer.serialize_some(&c.to_bytes()),
        None => serializer.serialize_none(),
    }
}

fn deserialize_option_cid<'de, D>(deserializer: D) -> std::result::Result<Option<Cid>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt: Option<Vec<u8>> = Deserialize::deserialize(deserializer)?;
    match opt {
        Some(bytes) => Cid::try_from(bytes)
            .map(Some)
            .map_err(serde::de::Error::custom),
        None => Ok(None),
    }
}

impl ProvenanceMetadata {
    /// Create new provenance metadata
    pub fn new(layer: String, training_config: String) -> Self {
        Self {
            layer,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time is after UNIX epoch")
                .as_secs(),
            training_config,
            parent: None,
            step: None,
            metadata: HashMap::new(),
        }
    }

    /// Set parent gradient CID
    pub fn with_parent(mut self, parent: Cid) -> Self {
        self.parent = Some(parent);
        self
    }

    /// Set training step
    pub fn with_step(mut self, step: u64) -> Self {
        self.step = Some(step);
        self
    }

    /// Add custom metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Gradient data with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientData {
    /// Shape of the gradient tensor
    pub shape: Vec<usize>,
    /// Data type (f32, f64, etc.)
    pub dtype: String,
    /// Encoded gradient data (delta or full)
    pub data: Vec<u8>,
    /// Whether this is a delta or full gradient
    pub is_delta: bool,
    /// Provenance metadata
    pub provenance: Option<ProvenanceMetadata>,
}

/// Delta encoder for efficient gradient storage
pub struct DeltaEncoder;

impl DeltaEncoder {
    /// Encode delta between base and target gradients
    ///
    /// Returns compressed delta representation
    pub fn encode_delta(base: &[f32], target: &[f32]) -> Result<Vec<u8>> {
        if base.len() != target.len() {
            return Err(Error::Storage(
                "Base and target must have same length".to_string(),
            ));
        }

        // Compute delta
        let delta: Vec<f32> = target.iter().zip(base.iter()).map(|(t, b)| t - b).collect();

        // Sparse encoding: store only non-zero deltas
        let mut sparse_delta = Vec::new();

        for (idx, &value) in delta.iter().enumerate() {
            if value.abs() > 1e-10 {
                // Threshold for sparsity
                // Store index and value
                sparse_delta.extend_from_slice(&(idx as u32).to_le_bytes());
                sparse_delta.extend_from_slice(&value.to_le_bytes());
            }
        }

        Ok(sparse_delta)
    }

    /// Decode delta and apply to base gradient
    pub fn decode_delta(base: &[f32], delta_bytes: &[u8]) -> Result<Vec<f32>> {
        let mut result = base.to_vec();

        // Read sparse delta entries
        let mut offset = 0;
        while offset + 8 <= delta_bytes.len() {
            let idx_bytes = &delta_bytes[offset..offset + 4];
            let value_bytes = &delta_bytes[offset + 4..offset + 8];

            let idx = u32::from_le_bytes([idx_bytes[0], idx_bytes[1], idx_bytes[2], idx_bytes[3]])
                as usize;
            let value = f32::from_le_bytes([
                value_bytes[0],
                value_bytes[1],
                value_bytes[2],
                value_bytes[3],
            ]);

            if idx < result.len() {
                result[idx] += value;
            }

            offset += 8;
        }

        Ok(result)
    }

    /// Compute compression ratio
    pub fn compression_ratio(original_size: usize, compressed_size: usize) -> f64 {
        if compressed_size == 0 {
            return 0.0;
        }
        original_size as f64 / compressed_size as f64
    }
}

/// Gradient store for managing gradients with delta encoding
pub struct GradientStore<S: BlockStore> {
    /// Underlying block store
    store: Arc<S>,
    /// Optional version control integration
    vcs: Option<Arc<VersionControl<S>>>,
}

impl<S: BlockStore> GradientStore<S> {
    /// Create a new gradient store
    pub fn new(store: Arc<S>) -> Self {
        Self { store, vcs: None }
    }

    /// Create with version control integration
    pub fn with_vcs(store: Arc<S>, vcs: Arc<VersionControl<S>>) -> Self {
        Self {
            store,
            vcs: Some(vcs),
        }
    }

    /// Get the version control system, if available
    pub fn vcs(&self) -> Option<&Arc<VersionControl<S>>> {
        self.vcs.as_ref()
    }

    /// Store a gradient (full)
    pub async fn store_gradient(
        &self,
        data: &[f32],
        shape: Vec<usize>,
        provenance: Option<ProvenanceMetadata>,
    ) -> Result<Cid> {
        let gradient_data = GradientData {
            shape,
            dtype: "f32".to_string(),
            data: Self::encode_f32_slice(data),
            is_delta: false,
            provenance,
        };

        self.store_gradient_data(&gradient_data).await
    }

    /// Store a gradient as delta from base
    pub async fn store_gradient_delta(
        &self,
        base_cid: &Cid,
        target: &[f32],
        shape: Vec<usize>,
        provenance: Option<ProvenanceMetadata>,
    ) -> Result<Cid> {
        // Load base gradient
        let base_data = self.load_gradient(base_cid).await?;
        let base = Self::decode_f32_slice(&base_data.data)?;

        // Encode delta
        let delta_bytes = DeltaEncoder::encode_delta(&base, target)?;

        let mut prov = provenance
            .unwrap_or_else(|| ProvenanceMetadata::new("unknown".to_string(), "delta".to_string()));
        prov.parent = Some(*base_cid);

        let gradient_data = GradientData {
            shape,
            dtype: "f32".to_string(),
            data: delta_bytes,
            is_delta: true,
            provenance: Some(prov),
        };

        self.store_gradient_data(&gradient_data).await
    }

    /// Load a gradient and reconstruct if it's a delta
    pub async fn load_gradient(&self, cid: &Cid) -> Result<GradientData> {
        let block = self
            .store
            .get(cid)
            .await?
            .ok_or_else(|| Error::NotFound(format!("Gradient not found: {cid}")))?;

        let gradient_data: GradientData =
            oxicode::serde::decode_owned_from_slice(block.data(), oxicode::config::standard())
                .map(|(v, _)| v)
                .map_err(|e| {
                    Error::Serialization(format!("Failed to deserialize gradient: {e}"))
                })?;

        Ok(gradient_data)
    }

    /// Reconstruct full gradient from delta chain
    pub fn reconstruct_gradient<'a>(
        &'a self,
        cid: &'a Cid,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<f32>>> + Send + 'a>> {
        Box::pin(async move {
            let gradient_data = self.load_gradient(cid).await?;

            if !gradient_data.is_delta {
                // Already full gradient
                return Self::decode_f32_slice(&gradient_data.data);
            }

            // Recursively reconstruct from base
            let parent_cid = gradient_data
                .provenance
                .as_ref()
                .and_then(|p| p.parent)
                .ok_or_else(|| Error::Storage("Delta gradient missing parent CID".to_string()))?;

            let base = self.reconstruct_gradient(&parent_cid).await?;
            DeltaEncoder::decode_delta(&base, &gradient_data.data)
        })
    }

    /// Store gradient data as a block
    async fn store_gradient_data(&self, gradient_data: &GradientData) -> Result<Cid> {
        let bytes = oxicode::serde::encode_to_vec(gradient_data, oxicode::config::standard())
            .map_err(|e| Error::Serialization(format!("Failed to serialize gradient: {e}")))?;

        let block = Block::new(Bytes::from(bytes))?;
        let cid = *block.cid();
        self.store.put(&block).await?;

        Ok(cid)
    }

    /// Encode f32 slice to bytes
    fn encode_f32_slice(data: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &value in data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    /// Decode bytes to f32 slice
    fn decode_f32_slice(bytes: &[u8]) -> Result<Vec<f32>> {
        if !bytes.len().is_multiple_of(4) {
            return Err(Error::Storage("Invalid f32 data length".to_string()));
        }

        let mut data = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            data.push(value);
        }

        Ok(data)
    }

    /// Get compression statistics for a gradient chain
    pub async fn compute_compression_stats(&self, cid: &Cid) -> Result<CompressionStats> {
        let gradient_data = self.load_gradient(cid).await?;

        let original_size = gradient_data.shape.iter().product::<usize>() * 4; // f32 = 4 bytes
        let compressed_size = gradient_data.data.len();

        let ratio = DeltaEncoder::compression_ratio(original_size, compressed_size);

        Ok(CompressionStats {
            original_size,
            compressed_size,
            compression_ratio: ratio,
            is_delta: gradient_data.is_delta,
        })
    }
}

/// Compression statistics
#[derive(Debug, Clone, PartialEq)]
pub struct CompressionStats {
    /// Original uncompressed size in bytes
    pub original_size: usize,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Compression ratio (original / compressed)
    pub compression_ratio: f64,
    /// Whether this is a delta encoding
    pub is_delta: bool,
}

#[cfg(all(test, feature = "sled-backend"))]
mod tests {
    use super::*;
    use crate::blockstore::{BlockStoreConfig, SledBlockStore};

    #[test]
    fn test_delta_encoding() {
        let base = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let target = vec![1.1f32, 2.0, 3.2, 4.0, 5.0];

        let delta_bytes = DeltaEncoder::encode_delta(&base, &target)
            .expect("encode_delta should succeed for valid equal-length slices");
        let reconstructed = DeltaEncoder::decode_delta(&base, &delta_bytes)
            .expect("decode_delta should reconstruct original values");

        for (i, (&orig, &recon)) in target.iter().zip(reconstructed.iter()).enumerate() {
            assert!(
                (orig - recon).abs() < 1e-5,
                "Mismatch at index {}: {} vs {}",
                i,
                orig,
                recon
            );
        }
    }

    #[test]
    fn test_sparse_delta() {
        // Sparse gradient: only a few elements change
        let base = vec![0.0f32; 1000];
        let mut target = vec![0.0f32; 1000];
        target[10] = 1.5;
        target[500] = 2.3;
        target[999] = -0.7;

        let delta_bytes = DeltaEncoder::encode_delta(&base, &target)
            .expect("encode_delta should succeed for sparse gradient");

        // Delta should be much smaller than full gradient
        let full_size = 1000 * 4; // 4000 bytes
        let delta_size = delta_bytes.len(); // Only 24 bytes (3 * 8)
        assert!(delta_size < full_size / 10, "Delta not sparse enough");

        let reconstructed = DeltaEncoder::decode_delta(&base, &delta_bytes)
            .expect("decode_delta should reconstruct sparse gradient");
        for (i, (&orig, &recon)) in target.iter().zip(reconstructed.iter()).enumerate() {
            assert!(
                (orig - recon).abs() < 1e-5,
                "Mismatch at index {}: {} vs {}",
                i,
                orig,
                recon
            );
        }
    }

    #[tokio::test]
    async fn test_gradient_store() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-gradient-test"),
            cache_size: 10 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);

        let store = Arc::new(
            SledBlockStore::new(config).expect("sled block store should initialize successfully"),
        );
        let gradient_store = GradientStore::new(store);

        let gradient = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];

        let cid = gradient_store
            .store_gradient(&gradient, shape, None)
            .await
            .expect("test: store_gradient should succeed");

        let loaded = gradient_store
            .load_gradient(&cid)
            .await
            .expect("test: load_gradient should return stored gradient");
        assert_eq!(loaded.shape, vec![2, 2]);
        assert!(!loaded.is_delta);
    }

    #[tokio::test]
    async fn test_gradient_delta_chain() {
        let config = BlockStoreConfig {
            path: std::env::temp_dir().join("ipfrs-gradient-delta-test"),
            cache_size: 10 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);

        let store = Arc::new(
            SledBlockStore::new(config).expect("test: sled block store should initialize"),
        );
        let gradient_store = GradientStore::new(store);

        // Store base gradient
        let base_grad = vec![1.0f32, 2.0, 3.0, 4.0];
        let base_cid = gradient_store
            .store_gradient(&base_grad, vec![2, 2], None)
            .await
            .expect("test: store_gradient for base should succeed");

        // Store delta
        let target_grad = vec![1.1f32, 2.0, 3.2, 4.0];
        let delta_cid = gradient_store
            .store_gradient_delta(&base_cid, &target_grad, vec![2, 2], None)
            .await
            .expect("test: store_gradient_delta should succeed");

        // Reconstruct
        let reconstructed = gradient_store
            .reconstruct_gradient(&delta_cid)
            .await
            .expect("test: reconstruct_gradient should succeed");

        for (i, (&orig, &recon)) in target_grad.iter().zip(reconstructed.iter()).enumerate() {
            assert!(
                (orig - recon).abs() < 1e-5,
                "Mismatch at index {}: {} vs {}",
                i,
                orig,
                recon
            );
        }
    }

    #[test]
    fn test_provenance_metadata() {
        let metadata = ProvenanceMetadata::new("layer1".to_string(), "lr=0.001".to_string())
            .with_step(100)
            .with_metadata("optimizer".to_string(), "adam".to_string());

        assert_eq!(metadata.layer, "layer1");
        assert_eq!(metadata.step, Some(100));
        assert_eq!(
            metadata
                .metadata
                .get("optimizer")
                .expect("test: optimizer metadata key should exist"),
            &"adam".to_string()
        );
    }
}
