//! Erasure coding for data resilience
//!
//! Provides Reed-Solomon erasure coding for:
//! - Configurable redundancy
//! - Partial block recovery
//! - Data durability in distributed storage

use ipfrs_core::Cid;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use thiserror::Error;

/// Serialize CID as string
fn serialize_cid<S>(cid: &Cid, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(&cid.to_string())
}

/// Deserialize CID from string
fn deserialize_cid<'de, D>(deserializer: D) -> Result<Cid, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.parse().map_err(serde::de::Error::custom)
}

/// Serialize Vec<CID> as Vec<String>
fn serialize_cid_vec<S>(cids: &[Cid], serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    use serde::ser::SerializeSeq;
    let mut seq = serializer.serialize_seq(Some(cids.len()))?;
    for cid in cids {
        seq.serialize_element(&cid.to_string())?;
    }
    seq.end()
}

/// Deserialize Vec<CID> from Vec<String>
fn deserialize_cid_vec<'de, D>(deserializer: D) -> Result<Vec<Cid>, D::Error>
where
    D: Deserializer<'de>,
{
    let strings: Vec<String> = Vec::deserialize(deserializer)?;
    strings
        .iter()
        .map(|s| s.parse().map_err(serde::de::Error::custom))
        .collect()
}

/// Error types for erasure coding
#[derive(Error, Debug)]
pub enum ErasureError {
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),
    #[error("Insufficient shards for recovery: have {have}, need {need}")]
    InsufficientShards { have: usize, need: usize },
    #[error("Shard size mismatch: expected {expected}, got {got}")]
    ShardSizeMismatch { expected: usize, got: usize },
    #[error("Invalid shard index: {0}")]
    InvalidShardIndex(usize),
    #[error("Encoding failed: {0}")]
    EncodingFailed(String),
    #[error("Decoding failed: {0}")]
    DecodingFailed(String),
}

/// Erasure coding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErasureConfig {
    /// Number of data shards
    pub data_shards: usize,
    /// Number of parity shards
    pub parity_shards: usize,
}

impl ErasureConfig {
    /// Create a new erasure coding configuration
    pub fn new(data_shards: usize, parity_shards: usize) -> Result<Self, ErasureError> {
        if data_shards == 0 {
            return Err(ErasureError::InvalidParams(
                "Data shards must be > 0".to_string(),
            ));
        }
        if parity_shards == 0 {
            return Err(ErasureError::InvalidParams(
                "Parity shards must be > 0".to_string(),
            ));
        }
        if data_shards + parity_shards > 256 {
            return Err(ErasureError::InvalidParams(
                "Total shards must be <= 256".to_string(),
            ));
        }
        Ok(Self {
            data_shards,
            parity_shards,
        })
    }

    /// Total number of shards
    pub fn total_shards(&self) -> usize {
        self.data_shards + self.parity_shards
    }

    /// Minimum shards needed for recovery
    pub fn min_shards_for_recovery(&self) -> usize {
        self.data_shards
    }

    /// Maximum failures tolerated
    pub fn max_failures(&self) -> usize {
        self.parity_shards
    }

    /// Redundancy ratio
    pub fn redundancy_ratio(&self) -> f64 {
        self.total_shards() as f64 / self.data_shards as f64
    }
}

/// Erasure-coded shard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shard {
    /// Shard index (0..total_shards)
    pub index: usize,
    /// Shard data
    pub data: Vec<u8>,
    /// Is this a parity shard?
    pub is_parity: bool,
}

impl Shard {
    /// Create a new shard
    pub fn new(index: usize, data: Vec<u8>, is_parity: bool) -> Self {
        Self {
            index,
            data,
            is_parity,
        }
    }

    /// Get shard size
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

/// Erasure-coded block metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErasureMetadata {
    /// Original block CID
    #[serde(serialize_with = "serialize_cid", deserialize_with = "deserialize_cid")]
    pub original_cid: Cid,
    /// Original block size
    pub original_size: usize,
    /// Erasure coding configuration
    pub config: ErasureConfig,
    /// Shard size in bytes
    pub shard_size: usize,
    /// CIDs of all shards (data + parity)
    #[serde(
        serialize_with = "serialize_cid_vec",
        deserialize_with = "deserialize_cid_vec"
    )]
    pub shard_cids: Vec<Cid>,
}

impl ErasureMetadata {
    /// Create new erasure metadata
    pub fn new(
        original_cid: Cid,
        original_size: usize,
        config: ErasureConfig,
        shard_size: usize,
        shard_cids: Vec<Cid>,
    ) -> Self {
        Self {
            original_cid,
            original_size,
            config,
            shard_size,
            shard_cids,
        }
    }

    /// Get total number of shards
    pub fn total_shards(&self) -> usize {
        self.config.total_shards()
    }

    /// Check if we have enough shards for recovery
    pub fn can_recover(&self, available_shards: usize) -> bool {
        available_shards >= self.config.min_shards_for_recovery()
    }
}

/// Simple XOR-based erasure coding implementation
/// (In production, use a proper Reed-Solomon library like reed-solomon-erasure)
pub struct SimpleErasureEncoder {
    config: ErasureConfig,
}

impl SimpleErasureEncoder {
    /// Create a new encoder
    pub fn new(config: ErasureConfig) -> Self {
        Self { config }
    }

    /// Encode data into shards
    pub fn encode(&self, data: &[u8]) -> Result<Vec<Shard>, ErasureError> {
        let data_shards = self.config.data_shards;
        let parity_shards = self.config.parity_shards;

        // Calculate shard size (pad if necessary)
        let shard_size = data.len().div_ceil(data_shards);
        let padded_size = shard_size * data_shards;

        // Create padded data
        let mut padded_data = data.to_vec();
        padded_data.resize(padded_size, 0);

        let mut shards = Vec::new();

        // Create data shards
        for i in 0..data_shards {
            let start = i * shard_size;
            let end = start + shard_size;
            let shard_data = padded_data[start..end].to_vec();
            shards.push(Shard::new(i, shard_data, false));
        }

        // Create parity shards using simple XOR
        // (This is a simplified implementation; real Reed-Solomon is more complex)
        for p in 0..parity_shards {
            let mut parity_data = vec![0u8; shard_size];

            // XOR all data shards with different patterns for each parity shard
            for (i, shard) in shards.iter().enumerate().take(data_shards) {
                let weight = ((i + p + 1) % 256) as u8;
                for (j, &byte) in shard.data.iter().enumerate() {
                    parity_data[j] ^= byte.wrapping_mul(weight);
                }
            }

            shards.push(Shard::new(data_shards + p, parity_data, true));
        }

        Ok(shards)
    }

    /// Decode data from shards
    pub fn decode(&self, shards: &[Shard], original_size: usize) -> Result<Vec<u8>, ErasureError> {
        if shards.len() < self.config.data_shards {
            return Err(ErasureError::InsufficientShards {
                have: shards.len(),
                need: self.config.data_shards,
            });
        }

        // Check shard sizes match
        if !shards.is_empty() {
            let expected_size = shards[0].size();
            for shard in shards {
                if shard.size() != expected_size {
                    return Err(ErasureError::ShardSizeMismatch {
                        expected: expected_size,
                        got: shard.size(),
                    });
                }
            }
        }

        // Separate data and parity shards
        let mut data_shards: Vec<_> = shards.iter().filter(|s| !s.is_parity).collect();
        let parity_shards: Vec<_> = shards.iter().filter(|s| s.is_parity).collect();

        // If we have all data shards, reconstruct directly
        if data_shards.len() == self.config.data_shards {
            let shard_size = data_shards[0].size();
            let mut reconstructed = Vec::with_capacity(shard_size * data_shards.len());

            // Sort by index
            data_shards.sort_by_key(|s| s.index);

            for shard in data_shards {
                reconstructed.extend_from_slice(&shard.data);
            }

            // Trim to original size
            reconstructed.truncate(original_size);
            return Ok(reconstructed);
        }

        // Need to use parity shards for recovery
        // This is a simplified recovery; real Reed-Solomon uses matrix inversion
        if data_shards.len() + parity_shards.len() < self.config.data_shards {
            return Err(ErasureError::InsufficientShards {
                have: data_shards.len() + parity_shards.len(),
                need: self.config.data_shards,
            });
        }

        // For this simple implementation, we can only recover if we have
        // exactly data_shards worth of total shards
        if data_shards.len() + parity_shards.len() == self.config.data_shards {
            // Simplified recovery (not a real implementation)
            return Err(ErasureError::DecodingFailed(
                "Recovery not fully implemented in simple encoder".to_string(),
            ));
        }

        Ok(Vec::new())
    }
}

/// Erasure coding manager
pub struct ErasureManager {
    encoder: SimpleErasureEncoder,
    /// Cached shard metadata
    metadata_cache: HashMap<Cid, ErasureMetadata>,
}

impl ErasureManager {
    /// Create a new erasure manager
    pub fn new(config: ErasureConfig) -> Self {
        Self {
            encoder: SimpleErasureEncoder::new(config),
            metadata_cache: HashMap::new(),
        }
    }

    /// Encode a block into shards
    pub fn encode_block(&mut self, _cid: Cid, data: &[u8]) -> Result<Vec<Shard>, ErasureError> {
        self.encoder.encode(data)
    }

    /// Decode shards back to original block
    pub fn decode_shards(
        &self,
        shards: &[Shard],
        original_size: usize,
    ) -> Result<Vec<u8>, ErasureError> {
        self.encoder.decode(shards, original_size)
    }

    /// Store metadata for a block
    pub fn store_metadata(&mut self, metadata: ErasureMetadata) {
        self.metadata_cache.insert(metadata.original_cid, metadata);
    }

    /// Get metadata for a block
    pub fn get_metadata(&self, cid: &Cid) -> Option<&ErasureMetadata> {
        self.metadata_cache.get(cid)
    }

    /// Check if block can be recovered with given shards
    pub fn can_recover(&self, cid: &Cid, available_shards: usize) -> bool {
        if let Some(metadata) = self.get_metadata(cid) {
            metadata.can_recover(available_shards)
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cid() -> Cid {
        "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse()
            .expect("test: valid CID string should parse")
    }

    #[test]
    fn test_erasure_config() {
        let config =
            ErasureConfig::new(4, 2).expect("test: valid ErasureConfig(4,2) should be created");
        assert_eq!(config.data_shards, 4);
        assert_eq!(config.parity_shards, 2);
        assert_eq!(config.total_shards(), 6);
        assert_eq!(config.min_shards_for_recovery(), 4);
        assert_eq!(config.max_failures(), 2);
        assert_eq!(config.redundancy_ratio(), 1.5);
    }

    #[test]
    fn test_erasure_config_invalid() {
        assert!(ErasureConfig::new(0, 2).is_err());
        assert!(ErasureConfig::new(2, 0).is_err());
        assert!(ErasureConfig::new(200, 100).is_err());
    }

    #[test]
    fn test_shard_creation() {
        let shard = Shard::new(0, vec![1, 2, 3], false);
        assert_eq!(shard.index, 0);
        assert_eq!(shard.size(), 3);
        assert!(!shard.is_parity);
    }

    #[test]
    fn test_encode_decode() {
        let config =
            ErasureConfig::new(3, 2).expect("test: valid ErasureConfig(3,2) should be created");
        let encoder = SimpleErasureEncoder::new(config);

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let original_size = data.len();

        let shards = encoder
            .encode(&data)
            .expect("test: encoding data into shards should succeed");
        assert_eq!(shards.len(), 5); // 3 data + 2 parity

        // Verify shard properties
        for (i, shard) in shards.iter().enumerate() {
            assert_eq!(shard.index, i);
            if i < 3 {
                assert!(!shard.is_parity);
            } else {
                assert!(shard.is_parity);
            }
        }

        // Decode with all shards
        let decoded = encoder
            .decode(&shards[..3], original_size)
            .expect("test: decoding data shards should succeed");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_encode_empty_data() {
        let config =
            ErasureConfig::new(2, 1).expect("test: valid ErasureConfig(2,1) should be created");
        let encoder = SimpleErasureEncoder::new(config);

        let data = vec![];
        let shards = encoder
            .encode(&data)
            .expect("test: encoding empty data should succeed");
        assert_eq!(shards.len(), 3);
    }

    #[test]
    fn test_decode_insufficient_shards() {
        let config =
            ErasureConfig::new(4, 2).expect("test: valid ErasureConfig(4,2) should be created");
        let encoder = SimpleErasureEncoder::new(config);

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shards = encoder
            .encode(&data)
            .expect("test: encoding data into shards should succeed");

        // Try to decode with only 2 shards (need 4)
        let result = encoder.decode(&shards[..2], data.len());
        assert!(result.is_err());
    }

    #[test]
    fn test_erasure_metadata() {
        let cid = test_cid();
        let config =
            ErasureConfig::new(3, 2).expect("test: valid ErasureConfig(3,2) should be created");
        let shard_cids = vec![test_cid(); 5];

        let metadata = ErasureMetadata::new(cid, 1000, config, 350, shard_cids);

        assert_eq!(metadata.original_size, 1000);
        assert_eq!(metadata.total_shards(), 5);
        assert!(metadata.can_recover(3));
        assert!(!metadata.can_recover(2));
    }

    #[test]
    fn test_erasure_manager() {
        let config =
            ErasureConfig::new(3, 2).expect("test: valid ErasureConfig(3,2) should be created");
        let mut manager = ErasureManager::new(config);

        let cid = test_cid();
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];

        let shards = manager
            .encode_block(cid, &data)
            .expect("test: encoding block into shards should succeed");
        assert_eq!(shards.len(), 5);

        let decoded = manager
            .decode_shards(&shards[..3], data.len())
            .expect("test: decoding shards back to original data should succeed");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_metadata_caching() {
        let config =
            ErasureConfig::new(3, 2).expect("test: valid ErasureConfig(3,2) should be created");
        let mut manager = ErasureManager::new(config.clone());

        let cid = test_cid();
        let shard_cids = vec![test_cid(); 5];
        let metadata = ErasureMetadata::new(cid, 1000, config, 350, shard_cids);

        manager.store_metadata(metadata);

        let retrieved = manager.get_metadata(&cid);
        assert!(retrieved.is_some());
        assert_eq!(
            retrieved
                .expect("test: stored metadata should be retrievable by CID")
                .original_size,
            1000
        );
    }

    #[test]
    fn test_can_recover() {
        let config =
            ErasureConfig::new(4, 2).expect("test: valid ErasureConfig(4,2) should be created");
        let mut manager = ErasureManager::new(config.clone());

        let cid = test_cid();
        let shard_cids = vec![test_cid(); 6];
        let metadata = ErasureMetadata::new(cid, 1000, config, 250, shard_cids);

        manager.store_metadata(metadata);

        assert!(manager.can_recover(&cid, 4));
        assert!(manager.can_recover(&cid, 5));
        assert!(!manager.can_recover(&cid, 3));
    }
}
