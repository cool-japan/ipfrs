//! Helper functions for creating storage stacks
//!
//! This module provides convenient functions for setting up common storage configurations.

use crate::{
    BlockStoreConfig, BloomBlockStore, BloomConfig, CachedBlockStore, MemoryBlockStore,
    MetricsBlockStore, SledBlockStore,
};
use crate::{ChunkingConfig, DedupBlockStore};
#[cfg(feature = "encryption")]
use crate::{Cipher, EncryptedBlockStore, EncryptionConfig};
use crate::{CoalesceConfig, CoalescingBlockStore};
#[cfg(feature = "compression")]
use crate::{CompressionAlgorithm, CompressionBlockStore, CompressionConfig};
use crate::{TtlBlockStore, TtlConfig};
use ipfrs_core::Result;
use std::path::PathBuf;
use std::time::Duration;

// Type aliases for complex storage stacks
type FullStack = BloomBlockStore<CachedBlockStore<SledBlockStore>>;
type MonitoredFullStack = MetricsBlockStore<FullStack>;

#[cfg(feature = "compression")]
type CompressedStack = CompressionBlockStore<FullStack>;
#[cfg(feature = "compression")]
type MonitoredCompressedStack = MetricsBlockStore<CompressedStack>;

#[cfg(feature = "encryption")]
type EncryptedStack = EncryptedBlockStore<FullStack>;
#[cfg(feature = "encryption")]
type MonitoredEncryptedStack = MetricsBlockStore<EncryptedStack>;

type DedupStack = DedupBlockStore<FullStack>;
type MonitoredDedupStack = MetricsBlockStore<DedupStack>;

#[cfg(feature = "compression")]
type UltimateStack = DedupBlockStore<CompressedStack>;
#[cfg(feature = "compression")]
type MonitoredUltimateStack = MetricsBlockStore<UltimateStack>;

/// Storage stack builder for easy configuration
pub struct StorageStackBuilder {
    config: BlockStoreConfig,
    enable_cache: bool,
    cache_size_mb: usize,
    enable_bloom: bool,
    bloom_expected_items: usize,
    enable_tiering: bool,
}

impl Default for StorageStackBuilder {
    fn default() -> Self {
        Self {
            config: BlockStoreConfig::default(),
            enable_cache: true,
            cache_size_mb: 100,
            enable_bloom: true,
            bloom_expected_items: 100_000,
            enable_tiering: false,
        }
    }
}

impl StorageStackBuilder {
    /// Create a new storage stack builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the base storage configuration
    pub fn with_config(mut self, config: BlockStoreConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the storage path
    pub fn with_path(mut self, path: PathBuf) -> Self {
        self.config.path = path;
        self
    }

    /// Enable LRU caching with specified size in MB
    pub fn with_cache(mut self, size_mb: usize) -> Self {
        self.enable_cache = true;
        self.cache_size_mb = size_mb;
        self
    }

    /// Disable LRU caching
    pub fn without_cache(mut self) -> Self {
        self.enable_cache = false;
        self
    }

    /// Enable bloom filter with expected number of items
    pub fn with_bloom(mut self, expected_items: usize) -> Self {
        self.enable_bloom = true;
        self.bloom_expected_items = expected_items;
        self
    }

    /// Disable bloom filter
    pub fn without_bloom(mut self) -> Self {
        self.enable_bloom = false;
        self
    }

    /// Enable hot/cold tiering
    pub fn with_tiering(mut self) -> Self {
        self.enable_tiering = true;
        self
    }

    /// Build a simple storage stack (base store only)
    pub fn build_simple(self) -> Result<SledBlockStore> {
        SledBlockStore::new(self.config)
    }

    /// Build a cached storage stack
    pub fn build_cached(self) -> Result<CachedBlockStore<SledBlockStore>> {
        use crate::CacheConfig;
        use std::num::NonZeroUsize;
        let base = SledBlockStore::new(self.config)?;
        // Translate megabyte budget into a block-count capacity (assume ~4 KiB/block).
        let block_capacity = std::cmp::max(1, self.cache_size_mb * 1024 * 1024 / 4096);
        let config = CacheConfig {
            l1_capacity: NonZeroUsize::new(block_capacity)
                .unwrap_or_else(|| NonZeroUsize::new(1024).expect("1024>0")),
            max_block_bytes: 256 * 1024,
        };
        Ok(CachedBlockStore::new(base, config))
    }

    /// Build a full storage stack with cache and bloom filter
    pub fn build_full(self) -> Result<BloomBlockStore<CachedBlockStore<SledBlockStore>>> {
        use crate::CacheConfig;
        use std::num::NonZeroUsize;
        let base = SledBlockStore::new(self.config)?;

        let block_capacity = if self.enable_cache {
            std::cmp::max(1, self.cache_size_mb * 1024 * 1024 / 4096)
        } else {
            1 // Minimal cache if disabled
        };
        let cache_config = CacheConfig {
            l1_capacity: NonZeroUsize::new(block_capacity)
                .unwrap_or_else(|| NonZeroUsize::new(1).expect("1>0")),
            max_block_bytes: 256 * 1024,
        };
        let cached = CachedBlockStore::new(base, cache_config);

        if self.enable_bloom {
            let bloom_config = BloomConfig::new(self.bloom_expected_items, 0.01);
            Ok(BloomBlockStore::with_config(cached, bloom_config))
        } else {
            // Return with minimal bloom filter if disabled
            let bloom_config = BloomConfig::new(100, 0.01);
            Ok(BloomBlockStore::with_config(cached, bloom_config))
        }
    }
}

/// Quick setup functions for common use cases
/// Create a development storage stack with caching and bloom filter
///
/// - Path: /tmp/ipfrs-dev
/// - Cache: 50MB
/// - Bloom filter: 10,000 expected items
pub fn development_stack() -> Result<BloomBlockStore<CachedBlockStore<SledBlockStore>>> {
    StorageStackBuilder::new()
        .with_config(BlockStoreConfig::development())
        .with_cache(50)
        .with_bloom(10_000)
        .build_full()
}

/// Create a production storage stack with caching and bloom filter
///
/// - Path: provided by user
/// - Cache: 500MB
/// - Bloom filter: 1,000,000 expected items
pub fn production_stack(path: PathBuf) -> Result<FullStack> {
    StorageStackBuilder::new()
        .with_config(BlockStoreConfig::production(path))
        .with_cache(500)
        .with_bloom(1_000_000)
        .build_full()
}

/// Create an embedded storage stack with minimal resource usage
///
/// - Path: provided by user
/// - Cache: 10MB
/// - Bloom filter: 5,000 expected items
pub fn embedded_stack(path: PathBuf) -> Result<BloomBlockStore<CachedBlockStore<SledBlockStore>>> {
    StorageStackBuilder::new()
        .with_config(BlockStoreConfig::embedded(path))
        .with_cache(10)
        .with_bloom(5_000)
        .build_full()
}

/// Create a testing storage stack with minimal resources
///
/// - Path: temporary directory
/// - Cache: 5MB
/// - Bloom filter: 1,000 expected items
pub fn testing_stack() -> Result<BloomBlockStore<CachedBlockStore<SledBlockStore>>> {
    StorageStackBuilder::new()
        .with_config(BlockStoreConfig::testing())
        .with_cache(5)
        .with_bloom(1_000)
        .build_full()
}

/// Create a production stack with metrics tracking
///
/// Adds comprehensive metrics on top of a production storage stack.
/// Useful for monitoring performance in production deployments.
pub fn monitored_production_stack(path: PathBuf) -> Result<MonitoredFullStack> {
    let base = production_stack(path)?;
    Ok(MetricsBlockStore::new(base))
}

/// Create a high-performance in-memory stack
///
/// Best for:
/// - Testing and development
/// - Temporary caching layers
/// - High-speed operations where persistence isn't needed
pub fn memory_stack() -> MetricsBlockStore<BloomBlockStore<MemoryBlockStore>> {
    let base = MemoryBlockStore::new();
    let bloom_config = BloomConfig::new(100_000, 0.01);
    let bloom = BloomBlockStore::with_config(base, bloom_config);
    MetricsBlockStore::new(bloom)
}

/// Create a compressed production stack (requires "compression" feature)
///
/// Uses Zstd compression to reduce storage size.
/// Best for: Large datasets where storage space is a concern
#[cfg(feature = "compression")]
pub fn compressed_production_stack(path: PathBuf) -> Result<MonitoredCompressedStack> {
    let base = production_stack(path)?;
    let compression_config = CompressionConfig {
        algorithm: CompressionAlgorithm::Zstd,
        level: 3,        // Balanced compression
        threshold: 1024, // Only compress blocks > 1KB
        max_ratio: 0.9,  // Only keep if compressed to 90% or less
    };
    let compressed = CompressionBlockStore::new(base, compression_config);
    Ok(MetricsBlockStore::new(compressed))
}

/// Create an encrypted production stack (requires "encryption" feature)
///
/// Uses ChaCha20-Poly1305 encryption for data at rest.
/// Best for: Sensitive data requiring encryption
#[cfg(feature = "encryption")]
pub fn encrypted_production_stack(
    path: PathBuf,
    password: &str,
) -> Result<MonitoredEncryptedStack> {
    use crate::EncryptionKey;

    let base = production_stack(path)?;
    let (key, _salt) =
        EncryptionKey::derive_from_password(Cipher::ChaCha20Poly1305, password.as_bytes(), None)?;
    let config = EncryptionConfig {
        cipher: Cipher::ChaCha20Poly1305,
    };
    let encrypted = EncryptedBlockStore::new(base, key, config);
    Ok(MetricsBlockStore::new(encrypted))
}

/// Create a deduplicated production stack
///
/// Uses content-defined chunking for automatic deduplication.
/// Best for: Datasets with significant redundancy
pub fn deduplicated_production_stack(path: PathBuf) -> Result<MonitoredDedupStack> {
    let base = production_stack(path)?;
    let chunking_config = ChunkingConfig::default();
    let dedup = DedupBlockStore::new(base, chunking_config);
    Ok(MetricsBlockStore::new(dedup))
}

/// Create the ultimate production stack with all optimizations (requires all features)
///
/// Combines:
/// - Compression (Zstd)
/// - Deduplication
/// - Caching
/// - Bloom filters
/// - Metrics tracking
///
/// Best for: Maximum efficiency in production with all features enabled
#[cfg(feature = "compression")]
pub fn ultimate_production_stack(path: PathBuf) -> Result<MonitoredUltimateStack> {
    let base = compressed_production_stack(path)?;
    // Remove metrics temporarily to add dedup, then re-add metrics
    let inner = base.into_inner();
    let chunking_config = ChunkingConfig::default();
    let dedup = DedupBlockStore::new(inner, chunking_config);
    Ok(MetricsBlockStore::new(dedup))
}

/// Create a production stack with TTL support for automatic expiration
///
/// Useful for:
/// - Temporary cache layers
/// - Time-limited data storage
/// - Preventing unbounded growth
///
/// # Arguments
/// * `path` - Storage directory path
/// * `default_ttl` - Default time-to-live for blocks
pub fn ttl_production_stack(
    path: PathBuf,
    default_ttl: Duration,
) -> Result<MetricsBlockStore<TtlBlockStore<FullStack>>> {
    let base = production_stack(path)?;
    let ttl_config = TtlConfig {
        default_ttl,
        auto_cleanup: true,
        cleanup_interval: Duration::from_secs(300), // 5 minutes
        max_tracked_blocks: 1_000_000,
    };
    let ttl_store = TtlBlockStore::new(base, ttl_config);
    Ok(MetricsBlockStore::new(ttl_store))
}

/// Create a production stack with automatic expiration for cache use cases
///
/// Optimized for cache workloads with:
/// - 1-hour default TTL
/// - Automatic cleanup every 5 minutes
/// - Large cache (500MB)
/// - Bloom filter for fast negative lookups
///
/// Best for: Temporary data caching with automatic expiration
pub fn cache_stack(path: PathBuf) -> Result<MetricsBlockStore<TtlBlockStore<FullStack>>> {
    ttl_production_stack(path, Duration::from_secs(3600)) // 1 hour TTL
}

/// Create a high-performance write-coalescing stack for in-memory operations
///
/// Combines:
/// - In-memory storage (no persistence)
/// - Write coalescing for batching (1000 writes per batch)
/// - 100ms flush interval
/// - Metrics tracking
///
/// Best for: Temporary high-throughput write workloads
pub fn coalescing_memory_stack() -> MetricsBlockStore<CoalescingBlockStore<MemoryBlockStore>> {
    let base = MemoryBlockStore::new();
    let coalesce_config = CoalesceConfig::new(1000, Duration::from_millis(100));
    let coalescing = CoalescingBlockStore::new(base, coalesce_config);
    MetricsBlockStore::new(coalescing)
}

/// Create a read-optimized production stack
///
/// Optimized for read-heavy workloads with:
/// - Large cache (1GB) for frequently accessed blocks
/// - Bloom filter for fast negative lookups
/// - Metrics tracking
///
/// Best for: Content delivery and read-heavy applications
pub fn read_optimized_stack(path: PathBuf) -> Result<FullStack> {
    StorageStackBuilder::new()
        .with_config(BlockStoreConfig::production(path))
        .with_cache(1024) // 1GB cache
        .with_bloom(2_000_000) // Support for 2M blocks
        .build_full()
}

/// Create a write-optimized production stack with deduplication
///
/// Optimized for write-heavy workloads with:
/// - Deduplication to reduce storage
/// - Smaller cache (100MB) to favor writes
/// - Bloom filter for existence checks
/// - Metrics tracking
///
/// Best for: Ingestion pipelines and write-heavy applications
pub fn write_optimized_stack(path: PathBuf) -> Result<MonitoredDedupStack> {
    let base = StorageStackBuilder::new()
        .with_config(BlockStoreConfig::production(path))
        .with_cache(100) // Smaller cache for writes
        .with_bloom(1_000_000)
        .build_full()?;

    let chunking_config = ChunkingConfig::default();
    let dedup = DedupBlockStore::new(base, chunking_config);
    Ok(MetricsBlockStore::new(dedup))
}

/// Create a minimal resource stack for IoT/embedded devices
///
/// Ultra-low resource usage with:
/// - 5MB cache
/// - Small bloom filter (1000 expected items)
/// - Minimal batch sizes
///
/// Best for: Raspberry Pi, embedded systems, resource-constrained environments
pub fn iot_stack(path: PathBuf) -> Result<FullStack> {
    StorageStackBuilder::new()
        .with_config(BlockStoreConfig::embedded(path))
        .with_cache(5) // 5MB cache
        .with_bloom(1_000) // Very small bloom filter
        .build_full()
}

/// Create a resilient production stack with all safety features
///
/// Combines:
/// - TTL for automatic cleanup (24 hours default)
/// - Metrics tracking for monitoring
/// - Large cache for performance
/// - Bloom filter for efficiency
///
/// Best for: Mission-critical production deployments requiring data lifecycle management
pub fn resilient_stack(path: PathBuf) -> Result<MetricsBlockStore<TtlBlockStore<FullStack>>> {
    ttl_production_stack(path, Duration::from_secs(86400)) // 24 hour TTL
}

/// Create a high-throughput ingestion stack
///
/// Optimized for maximum write throughput:
/// - Deduplication for storage efficiency
/// - Smaller cache (200MB) to favor writes
/// - Bloom filter for fast existence checks
/// - Metrics for monitoring
///
/// Best for: Data ingestion pipelines, ETL processes, bulk imports
pub fn ingestion_stack(path: PathBuf) -> Result<MonitoredDedupStack> {
    let base = StorageStackBuilder::new()
        .with_config(BlockStoreConfig::production(path))
        .with_cache(200) // Smaller cache for write optimization
        .with_bloom(2_000_000)
        .build_full()?;

    let chunking_config = ChunkingConfig::default();
    let dedup = DedupBlockStore::new(base, chunking_config);

    Ok(MetricsBlockStore::new(dedup))
}

/// Create a CDN edge cache stack
///
/// Optimized for content delivery:
/// - Very large cache (2GB) for hot content
/// - TTL for automatic expiration (1 hour default)
/// - Bloom filter for fast negative lookups
/// - Metrics for monitoring cache effectiveness
///
/// Best for: CDN edge nodes, content delivery, proxy caching
pub fn cdn_edge_stack(path: PathBuf) -> Result<MetricsBlockStore<TtlBlockStore<FullStack>>> {
    let base = StorageStackBuilder::new()
        .with_config(BlockStoreConfig::production(path))
        .with_cache(2048) // 2GB cache
        .with_bloom(5_000_000) // Support large number of blocks
        .build_full()?;

    let ttl_config = TtlConfig {
        default_ttl: Duration::from_secs(3600), // 1 hour
        auto_cleanup: true,
        cleanup_interval: Duration::from_secs(600), // 10 minutes
        max_tracked_blocks: 5_000_000,
    };
    let ttl_store = TtlBlockStore::new(base, ttl_config);

    Ok(MetricsBlockStore::new(ttl_store))
}

/// Create a scientific data archive stack
///
/// Optimized for large scientific datasets:
/// - Compression (Zstd level 5 for better compression)
/// - Deduplication for redundant data
/// - Medium cache (256MB)
/// - Bloom filter
///
/// Best for: Scientific data repositories, research archives, large dataset storage
#[cfg(feature = "compression")]
pub fn scientific_archive_stack(path: PathBuf) -> Result<MonitoredUltimateStack> {
    let base = StorageStackBuilder::new()
        .with_config(BlockStoreConfig::production(path))
        .with_cache(256)
        .with_bloom(1_000_000)
        .build_full()?;

    let compression_config = CompressionConfig {
        algorithm: CompressionAlgorithm::Zstd,
        level: 5,        // Higher compression for archives
        threshold: 512,  // Compress blocks > 512 bytes
        max_ratio: 0.95, // Keep if compressed to 95% or less
    };
    let compressed = CompressionBlockStore::new(base, compression_config);

    let chunking_config = ChunkingConfig::default();
    let dedup = DedupBlockStore::new(compressed, chunking_config);

    Ok(MetricsBlockStore::new(dedup))
}

/// Create a blockchain storage stack
///
/// Optimized for blockchain data:
/// - No TTL (permanent storage)
/// - Large bloom filter for fast lookups
/// - Medium cache for recent blocks
/// - Deduplication for transactions
/// - Metrics for monitoring
///
/// Best for: Blockchain nodes, distributed ledgers, immutable data
pub fn blockchain_stack(path: PathBuf) -> Result<MonitoredDedupStack> {
    let base = StorageStackBuilder::new()
        .with_config(BlockStoreConfig::production(path))
        .with_cache(512) // 512MB for recent blocks
        .with_bloom(10_000_000) // Large bloom for many blocks
        .build_full()?;

    let chunking_config = ChunkingConfig::default();
    let dedup = DedupBlockStore::new(base, chunking_config);

    Ok(MetricsBlockStore::new(dedup))
}

/// Create a machine learning model storage stack
///
/// Optimized for ML model storage:
/// - Large cache (1GB) for frequently accessed models
/// - Bloom filter for fast existence checks
/// - Metrics tracking
///
/// Best for: ML model repositories, model versioning, training checkpoints
pub fn ml_model_stack(path: PathBuf) -> Result<MonitoredFullStack> {
    let base = StorageStackBuilder::new()
        .with_config(BlockStoreConfig::production(path))
        .with_cache(1024) // 1GB cache for models
        .with_bloom(100_000) // Moderate bloom size
        .build_full()?;

    Ok(MetricsBlockStore::new(base))
}

/// Create a media streaming stack
///
/// Optimized for video/audio streaming:
/// - Large cache (3GB) for hot content
/// - TTL for session-based content (2 hours)
/// - Bloom filter for catalog lookups
///
/// Best for: Video streaming services, audio platforms, media servers
pub fn media_streaming_stack(path: PathBuf) -> Result<MetricsBlockStore<TtlBlockStore<FullStack>>> {
    let base = StorageStackBuilder::new()
        .with_config(BlockStoreConfig::production(path))
        .with_cache(3072) // 3GB cache
        .with_bloom(1_000_000)
        .build_full()?;

    let ttl_config = TtlConfig {
        default_ttl: Duration::from_secs(7200), // 2 hours
        auto_cleanup: true,
        cleanup_interval: Duration::from_secs(300), // 5 minutes
        max_tracked_blocks: 1_000_000,
    };
    let ttl_store = TtlBlockStore::new(base, ttl_config);

    Ok(MetricsBlockStore::new(ttl_store))
}

/// Create a distributed file system stack
///
/// Optimized for distributed filesystems:
/// - Compression for storage efficiency
/// - Deduplication for redundant files
/// - Large cache (1.5GB)
/// - Large bloom filter
///
/// Best for: Distributed filesystems, cluster storage, shared filesystems
#[cfg(feature = "compression")]
pub fn distributed_fs_stack(path: PathBuf) -> Result<MonitoredUltimateStack> {
    let base = StorageStackBuilder::new()
        .with_config(BlockStoreConfig::production(path))
        .with_cache(1536) // 1.5GB cache
        .with_bloom(5_000_000)
        .build_full()?;

    let compression_config = CompressionConfig {
        algorithm: CompressionAlgorithm::Lz4, // Fast compression for FS
        level: 1,                             // Fast compression
        threshold: 4096,                      // Only compress larger files
        max_ratio: 0.9,
    };
    let compressed = CompressionBlockStore::new(base, compression_config);

    let chunking_config = ChunkingConfig::default();
    let dedup = DedupBlockStore::new(compressed, chunking_config);

    Ok(MetricsBlockStore::new(dedup))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_simple() {
        let _stack = StorageStackBuilder::new()
            .with_path(std::env::temp_dir().join("test-simple"))
            .build_simple();
        assert!(_stack.is_ok());
    }

    #[test]
    fn test_builder_cached() {
        let _stack = StorageStackBuilder::new()
            .with_path(std::env::temp_dir().join("test-cached"))
            .with_cache(10)
            .build_cached();
        assert!(_stack.is_ok());
    }

    #[test]
    fn test_builder_full() {
        let _stack = StorageStackBuilder::new()
            .with_path(std::env::temp_dir().join("test-full"))
            .with_cache(10)
            .with_bloom(1000)
            .build_full();
        assert!(_stack.is_ok());
    }

    #[test]
    fn test_development_stack() {
        let _stack = development_stack();
        assert!(_stack.is_ok());
    }

    #[test]
    fn test_testing_stack() {
        let _stack = testing_stack();
        assert!(_stack.is_ok());
    }
}
