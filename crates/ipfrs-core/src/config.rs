//! Centralized configuration management for IPFRS
//!
//! This module provides a unified configuration system for all IPFRS operations,
//! including environment variable support, validation, and preset configurations
//! for common use cases.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_core::config::{Config, ConfigBuilder};
//! use ipfrs_core::HashAlgorithm;
//!
//! // Use default configuration
//! let config = Config::default();
//!
//! // Build custom configuration
//! let config = ConfigBuilder::new()
//!     .chunk_size(512 * 1024)
//!     .hash_algorithm(HashAlgorithm::Sha3_256)
//!     .enable_metrics(true)
//!     .build()
//!     .unwrap();
//!
//! // Use preset for high-performance scenarios
//! let config = Config::high_performance();
//!
//! println!("Chunk size: {}", config.chunk_size);
//! println!("Hash algorithm: {:?}", config.hash_algorithm);
//! ```

use crate::chunking::{ChunkingStrategy, DEFAULT_CHUNK_SIZE, MAX_CHUNK_SIZE, MIN_CHUNK_SIZE};
use crate::cid::HashAlgorithm;
use crate::error::{Error, Result};
use once_cell::sync::Lazy;
use std::sync::{Arc, RwLock};

/// Global configuration instance
pub static GLOBAL_CONFIG: Lazy<Arc<RwLock<Config>>> =
    Lazy::new(|| Arc::new(RwLock::new(Config::default())));

/// Get the global configuration
///
/// # Example
///
/// ```rust
/// use ipfrs_core::config::global_config;
///
/// let config = global_config();
/// let chunk_size = config.read().unwrap_or_else(|e| e.into_inner()).chunk_size;
/// ```
pub fn global_config() -> Arc<RwLock<Config>> {
    Arc::clone(&GLOBAL_CONFIG)
}

/// Set the global configuration
///
/// # Example
///
/// ```rust
/// use ipfrs_core::config::{set_global_config, Config};
///
/// let config = Config::high_performance();
/// set_global_config(config);
/// ```
pub fn set_global_config(config: Config) {
    *GLOBAL_CONFIG.write().unwrap_or_else(|e| e.into_inner()) = config;
}

/// Main configuration for IPFRS operations
#[derive(Debug, Clone)]
pub struct Config {
    // Chunking settings
    /// Size of each chunk in bytes
    pub chunk_size: usize,
    /// Chunking strategy to use
    pub chunking_strategy: ChunkingStrategy,
    /// Maximum links per DAG node
    pub max_links_per_node: usize,

    // Hashing settings
    /// Hash algorithm for CID generation
    pub hash_algorithm: HashAlgorithm,

    // Performance settings
    /// Number of threads for parallel operations (None = use all available)
    pub num_threads: Option<usize>,
    /// Enable parallel chunking for large files
    pub enable_parallel_chunking: bool,
    /// Threshold for switching to parallel chunking (bytes)
    pub parallel_threshold: usize,

    // Memory settings
    /// Enable memory pooling for allocations
    pub enable_pooling: bool,
    /// Maximum pool size in bytes
    pub pool_max_size: usize,

    // Metrics and observability
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Maximum metrics samples to keep
    pub metrics_max_samples: usize,

    // Validation settings
    /// Enable block verification on read
    pub verify_blocks: bool,
    /// Enable CID validation on parse
    pub validate_cids: bool,

    // Storage settings
    /// Enable compression for storage
    pub enable_compression: bool,
    /// Compression level (0-9, where 0 is no compression)
    pub compression_level: u8,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            // Chunking defaults
            chunk_size: DEFAULT_CHUNK_SIZE,
            chunking_strategy: ChunkingStrategy::FixedSize,
            max_links_per_node: 174,

            // Hashing defaults
            hash_algorithm: HashAlgorithm::Sha256,

            // Performance defaults
            num_threads: None,
            enable_parallel_chunking: true,
            parallel_threshold: 1_000_000, // 1MB

            // Memory defaults
            enable_pooling: true,
            pool_max_size: 100 * 1024 * 1024, // 100MB

            // Metrics defaults
            enable_metrics: true,
            metrics_max_samples: 10_000,

            // Validation defaults
            verify_blocks: true,
            validate_cids: true,

            // Storage defaults
            enable_compression: false,
            compression_level: 3,
        }
    }
}

impl Config {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Validate chunk size
        if self.chunk_size < MIN_CHUNK_SIZE {
            return Err(Error::InvalidInput(format!(
                "Chunk size {} is below minimum {}",
                self.chunk_size, MIN_CHUNK_SIZE
            )));
        }
        if self.chunk_size > MAX_CHUNK_SIZE {
            return Err(Error::InvalidInput(format!(
                "Chunk size {} exceeds maximum {}",
                self.chunk_size, MAX_CHUNK_SIZE
            )));
        }

        // Validate compression level
        if self.compression_level > 9 {
            return Err(Error::InvalidInput(format!(
                "Compression level {} exceeds maximum 9",
                self.compression_level
            )));
        }

        // Validate pool size
        if self.pool_max_size == 0 && self.enable_pooling {
            return Err(Error::InvalidInput(
                "Pool max size cannot be zero when pooling is enabled".to_string(),
            ));
        }

        Ok(())
    }

    // === Preset Configurations ===

    /// Configuration optimized for high performance
    ///
    /// Uses SHA3-256 hashing, parallel processing, and larger chunk sizes.
    pub fn high_performance() -> Self {
        Self {
            hash_algorithm: HashAlgorithm::Sha3_256,
            chunk_size: 512 * 1024, // 512KB
            enable_parallel_chunking: true,
            parallel_threshold: 100_000, // 100KB
            num_threads: None,           // Use all available
            chunking_strategy: ChunkingStrategy::FixedSize,
            enable_pooling: true,
            pool_max_size: 200 * 1024 * 1024, // 200MB
            enable_metrics: true,
            ..Default::default()
        }
    }

    /// Configuration optimized for storage efficiency
    ///
    /// Uses content-defined chunking for better deduplication.
    pub fn storage_optimized() -> Self {
        Self {
            chunking_strategy: ChunkingStrategy::ContentDefined,
            chunk_size: 128 * 1024, // 128KB for better deduplication
            enable_compression: true,
            compression_level: 6,
            hash_algorithm: HashAlgorithm::Sha256,
            ..Default::default()
        }
    }

    /// Configuration for embedded/resource-constrained systems
    ///
    /// Uses smaller chunks, less parallelism, and minimal memory.
    pub fn embedded() -> Self {
        Self {
            chunk_size: 64 * 1024, // 64KB
            enable_parallel_chunking: false,
            num_threads: Some(1),
            enable_pooling: false,
            pool_max_size: 10 * 1024 * 1024, // 10MB
            enable_metrics: false,
            hash_algorithm: HashAlgorithm::Sha256,
            ..Default::default()
        }
    }

    /// Configuration for testing and development
    ///
    /// Enables all validations and uses smaller sizes for faster tests.
    pub fn testing() -> Self {
        Self {
            chunk_size: 16 * 1024, // 16KB
            verify_blocks: true,
            validate_cids: true,
            enable_metrics: true,
            enable_parallel_chunking: false, // More deterministic
            hash_algorithm: HashAlgorithm::Sha256,
            ..Default::default()
        }
    }
}

/// Builder for creating custom configurations
#[derive(Debug, Default)]
pub struct ConfigBuilder {
    config: Config,
}

impl ConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: Config::default(),
        }
    }

    /// Set chunk size
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.config.chunk_size = size;
        self
    }

    /// Set chunking strategy
    pub fn chunking_strategy(mut self, strategy: ChunkingStrategy) -> Self {
        self.config.chunking_strategy = strategy;
        self
    }

    /// Set hash algorithm
    pub fn hash_algorithm(mut self, algorithm: HashAlgorithm) -> Self {
        self.config.hash_algorithm = algorithm;
        self
    }

    /// Set number of threads
    pub fn num_threads(mut self, threads: usize) -> Self {
        self.config.num_threads = Some(threads);
        self
    }

    /// Enable or disable parallel chunking
    pub fn enable_parallel_chunking(mut self, enable: bool) -> Self {
        self.config.enable_parallel_chunking = enable;
        self
    }

    /// Set parallel chunking threshold
    pub fn parallel_threshold(mut self, threshold: usize) -> Self {
        self.config.parallel_threshold = threshold;
        self
    }

    /// Enable or disable memory pooling
    pub fn enable_pooling(mut self, enable: bool) -> Self {
        self.config.enable_pooling = enable;
        self
    }

    /// Set maximum pool size
    pub fn pool_max_size(mut self, size: usize) -> Self {
        self.config.pool_max_size = size;
        self
    }

    /// Enable or disable metrics collection
    pub fn enable_metrics(mut self, enable: bool) -> Self {
        self.config.enable_metrics = enable;
        self
    }

    /// Set maximum metrics samples
    pub fn metrics_max_samples(mut self, samples: usize) -> Self {
        self.config.metrics_max_samples = samples;
        self
    }

    /// Enable or disable block verification
    pub fn verify_blocks(mut self, verify: bool) -> Self {
        self.config.verify_blocks = verify;
        self
    }

    /// Enable or disable CID validation
    pub fn validate_cids(mut self, validate: bool) -> Self {
        self.config.validate_cids = validate;
        self
    }

    /// Enable or disable compression
    pub fn enable_compression(mut self, enable: bool) -> Self {
        self.config.enable_compression = enable;
        self
    }

    /// Set compression level
    pub fn compression_level(mut self, level: u8) -> Self {
        self.config.compression_level = level;
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<Config> {
        self.config.validate()?;
        Ok(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.chunk_size, DEFAULT_CHUNK_SIZE);
        assert!(config.enable_metrics);
        assert!(config.verify_blocks);
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        assert!(config.validate().is_ok());

        // Invalid chunk size
        config.chunk_size = 100;
        assert!(config.validate().is_err());

        // Valid chunk size
        config.chunk_size = 128 * 1024;
        assert!(config.validate().is_ok());

        // Invalid compression level
        config.compression_level = 10;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_high_performance_preset() {
        let config = Config::high_performance();
        assert_eq!(config.hash_algorithm, HashAlgorithm::Sha3_256);
        assert_eq!(config.chunk_size, 512 * 1024);
        assert!(config.enable_parallel_chunking);
    }

    #[test]
    fn test_storage_optimized_preset() {
        let config = Config::storage_optimized();
        assert_eq!(config.chunking_strategy, ChunkingStrategy::ContentDefined);
        assert!(config.enable_compression);
        assert_eq!(config.compression_level, 6);
    }

    #[test]
    fn test_embedded_preset() {
        let config = Config::embedded();
        assert_eq!(config.chunk_size, 64 * 1024);
        assert!(!config.enable_parallel_chunking);
        assert_eq!(config.num_threads, Some(1));
        assert!(!config.enable_pooling);
    }

    #[test]
    fn test_testing_preset() {
        let config = Config::testing();
        assert_eq!(config.chunk_size, 16 * 1024);
        assert!(config.verify_blocks);
        assert!(config.validate_cids);
        assert!(!config.enable_parallel_chunking);
    }

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .chunk_size(256 * 1024)
            .hash_algorithm(HashAlgorithm::Sha3_256)
            .enable_metrics(true)
            .num_threads(4)
            .build()
            .unwrap();

        assert_eq!(config.chunk_size, 256 * 1024);
        assert_eq!(config.hash_algorithm, HashAlgorithm::Sha3_256);
        assert!(config.enable_metrics);
        assert_eq!(config.num_threads, Some(4));
    }

    #[test]
    fn test_config_builder_validation() {
        let result = ConfigBuilder::new().chunk_size(100).build();
        assert!(result.is_err());

        let result = ConfigBuilder::new().chunk_size(128 * 1024).build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_global_config() {
        let config = global_config();
        {
            let cfg = config.read().unwrap_or_else(|e| e.into_inner());
            assert_eq!(cfg.chunk_size, DEFAULT_CHUNK_SIZE);
        }

        // Set new global config
        set_global_config(Config::high_performance());

        {
            let cfg = config.read().unwrap_or_else(|e| e.into_inner());
            assert_eq!(cfg.hash_algorithm, HashAlgorithm::Sha3_256);
        }

        // Reset to default for other tests
        set_global_config(Config::default());
    }

    #[test]
    fn test_builder_fluent_interface() {
        let config = ConfigBuilder::new()
            .chunk_size(128 * 1024)
            .enable_parallel_chunking(true)
            .parallel_threshold(500_000)
            .enable_compression(true)
            .compression_level(5)
            .build()
            .unwrap();

        assert_eq!(config.chunk_size, 128 * 1024);
        assert!(config.enable_parallel_chunking);
        assert_eq!(config.parallel_threshold, 500_000);
        assert!(config.enable_compression);
        assert_eq!(config.compression_level, 5);
    }
}
