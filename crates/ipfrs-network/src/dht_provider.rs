//! Custom DHT provider interface for pluggable DHT implementations
//!
//! This module provides a trait-based abstraction for DHT implementations,
//! allowing IPFRS to support multiple DHT backends beyond Kademlia.
//!
//! ## Features
//!
//! - **Pluggable DHT**: Support for alternative DHT implementations
//! - **Provider Trait**: Common interface for all DHT backends
//! - **Provider Registry**: Dynamic registration and selection of DHT providers
//! - **Extensibility**: Easy integration of custom DHT algorithms
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_network::dht_provider::{DhtProvider, DhtProviderRegistry, DhtCapabilities};
//! use ipfrs_network::dht_provider::kademlia::KademliaDhtProvider;
//! use std::sync::Arc;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Register DHT providers
//! let mut registry = DhtProviderRegistry::new();
//! registry.register("kademlia", Arc::new(KademliaDhtProvider::new()));
//!
//! // Select and use a DHT provider
//! if let Some(provider) = registry.get("kademlia") {
//!     let caps = provider.capabilities();
//!     println!("Provider supports content routing: {}", caps.supports_content_routing);
//! }
//! # Ok(())
//! # }
//! ```

use cid::Cid;
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

/// Errors that can occur in DHT provider operations
#[derive(Debug, Error)]
pub enum DhtProviderError {
    /// Provider not found
    #[error("DHT provider not found: {0}")]
    ProviderNotFound(String),

    /// Operation not supported by this provider
    #[error("Operation not supported: {0}")]
    OperationNotSupported(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Query failed
    #[error("Query failed: {0}")]
    QueryFailed(String),

    /// Internal error
    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Capabilities of a DHT provider
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DhtCapabilities {
    /// Supports content routing (finding providers)
    pub supports_content_routing: bool,
    /// Supports peer routing (finding peers)
    pub supports_peer_routing: bool,
    /// Supports key-value storage
    pub supports_kv_storage: bool,
    /// Supports range queries
    pub supports_range_queries: bool,
    /// Supports semantic queries
    pub supports_semantic_queries: bool,
    /// Maximum number of hops for queries
    pub max_query_hops: Option<usize>,
    /// Supports custom routing algorithms
    pub supports_custom_routing: bool,
}

impl DhtCapabilities {
    /// Create capabilities for a basic DHT
    pub fn basic() -> Self {
        Self {
            supports_content_routing: true,
            supports_peer_routing: true,
            supports_kv_storage: false,
            supports_range_queries: false,
            supports_semantic_queries: false,
            max_query_hops: Some(20),
            supports_custom_routing: false,
        }
    }

    /// Create capabilities for an advanced DHT
    pub fn advanced() -> Self {
        Self {
            supports_content_routing: true,
            supports_peer_routing: true,
            supports_kv_storage: true,
            supports_range_queries: true,
            supports_semantic_queries: true,
            max_query_hops: Some(20),
            supports_custom_routing: true,
        }
    }
}

/// Result of a DHT query
#[derive(Debug, Clone)]
pub struct DhtQueryResult {
    /// Peers that provide the content
    pub providers: Vec<PeerId>,
    /// Number of hops taken
    pub hops: usize,
    /// Query duration in milliseconds
    pub duration_ms: u64,
    /// Whether the query was successful
    pub success: bool,
}

/// Peer information from DHT
#[derive(Debug, Clone)]
pub struct DhtPeerInfo {
    /// Peer ID
    pub peer_id: PeerId,
    /// Addresses
    pub addresses: Vec<String>,
    /// Distance metric (provider-specific)
    pub distance: Option<u64>,
}

/// Common trait for DHT providers
pub trait DhtProvider: Send + Sync {
    /// Get provider name
    fn name(&self) -> &str;

    /// Get provider version
    fn version(&self) -> &str;

    /// Get provider capabilities
    fn capabilities(&self) -> DhtCapabilities;

    /// Bootstrap the DHT with known peers
    fn bootstrap(&self, peers: Vec<PeerId>) -> Result<(), DhtProviderError>;

    /// Announce content availability
    fn provide(&self, cid: &Cid) -> Result<(), DhtProviderError>;

    /// Find providers for content
    fn find_providers(&self, cid: &Cid) -> Result<DhtQueryResult, DhtProviderError>;

    /// Find a specific peer
    fn find_peer(&self, peer_id: &PeerId) -> Result<DhtPeerInfo, DhtProviderError>;

    /// Get closest peers to a key
    fn get_closest_peers(
        &self,
        key: &[u8],
        count: usize,
    ) -> Result<Vec<DhtPeerInfo>, DhtProviderError>;

    /// Put a key-value pair (if supported)
    fn put_value(&self, key: &[u8], value: &[u8]) -> Result<(), DhtProviderError> {
        let _ = (key, value);
        Err(DhtProviderError::OperationNotSupported(
            "Key-value storage not supported".to_string(),
        ))
    }

    /// Get a value by key (if supported)
    fn get_value(&self, key: &[u8]) -> Result<Vec<u8>, DhtProviderError> {
        let _ = key;
        Err(DhtProviderError::OperationNotSupported(
            "Key-value storage not supported".to_string(),
        ))
    }

    /// Get DHT statistics
    fn stats(&self) -> DhtProviderStats;

    /// Check if DHT is healthy
    fn is_healthy(&self) -> bool {
        let stats = self.stats();
        stats.routing_table_size > 0 && stats.success_rate > 0.5
    }
}

/// Statistics for DHT provider
#[derive(Debug, Clone, Default)]
pub struct DhtProviderStats {
    /// Number of peers in routing table
    pub routing_table_size: usize,
    /// Total queries executed
    pub total_queries: u64,
    /// Successful queries
    pub successful_queries: u64,
    /// Failed queries
    pub failed_queries: u64,
    /// Average query duration in milliseconds
    pub avg_query_duration_ms: f64,
    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,
}

impl DhtProviderStats {
    /// Calculate success rate
    pub fn calculate_success_rate(&mut self) {
        if self.total_queries > 0 {
            self.success_rate = self.successful_queries as f64 / self.total_queries as f64;
        } else {
            self.success_rate = 0.0;
        }
    }
}

/// Registry for DHT providers
pub struct DhtProviderRegistry {
    /// Registered providers
    providers: HashMap<String, Arc<dyn DhtProvider>>,
    /// Active provider name
    active_provider: Option<String>,
}

impl DhtProviderRegistry {
    /// Create a new provider registry
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            active_provider: None,
        }
    }

    /// Register a DHT provider
    pub fn register(&mut self, name: impl Into<String>, provider: Arc<dyn DhtProvider>) {
        let name = name.into();
        self.providers.insert(name.clone(), provider);

        // Set as active if it's the first provider
        if self.active_provider.is_none() {
            self.active_provider = Some(name);
        }
    }

    /// Get a provider by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn DhtProvider>> {
        self.providers.get(name).cloned()
    }

    /// Get the active provider
    pub fn get_active(&self) -> Option<Arc<dyn DhtProvider>> {
        self.active_provider
            .as_ref()
            .and_then(|name| self.get(name))
    }

    /// Set active provider
    pub fn set_active(&mut self, name: impl Into<String>) -> Result<(), DhtProviderError> {
        let name = name.into();
        if self.providers.contains_key(&name) {
            self.active_provider = Some(name);
            Ok(())
        } else {
            Err(DhtProviderError::ProviderNotFound(name))
        }
    }

    /// List all registered providers
    pub fn list_providers(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }

    /// Remove a provider
    pub fn unregister(&mut self, name: &str) -> Option<Arc<dyn DhtProvider>> {
        let provider = self.providers.remove(name);

        // Clear active provider if it was removed
        if self.active_provider.as_deref() == Some(name) {
            self.active_provider = None;
        }

        provider
    }

    /// Get number of registered providers
    pub fn count(&self) -> usize {
        self.providers.len()
    }

    /// Check if a provider is registered
    pub fn has_provider(&self, name: &str) -> bool {
        self.providers.contains_key(name)
    }
}

impl Default for DhtProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Kademlia DHT provider implementation
pub mod kademlia {
    use super::*;

    /// Kademlia DHT provider
    pub struct KademliaDhtProvider {
        stats: parking_lot::RwLock<DhtProviderStats>,
        config: KademliaConfig,
    }

    /// Configuration for Kademlia DHT
    #[derive(Debug, Clone)]
    pub struct KademliaConfig {
        /// K-bucket size (number of peers per bucket)
        pub k_bucket_size: usize,
        /// Alpha parameter (concurrency)
        pub alpha: usize,
        /// Replication factor
        pub replication_factor: usize,
        /// Query timeout in seconds
        pub query_timeout_secs: u64,
    }

    impl Default for KademliaConfig {
        fn default() -> Self {
            Self {
                k_bucket_size: 20,
                alpha: 3,
                replication_factor: 20,
                query_timeout_secs: 60,
            }
        }
    }

    impl KademliaDhtProvider {
        /// Create a new Kademlia DHT provider
        pub fn new() -> Self {
            Self::with_config(KademliaConfig::default())
        }

        /// Create with custom configuration
        pub fn with_config(config: KademliaConfig) -> Self {
            Self {
                stats: parking_lot::RwLock::new(DhtProviderStats::default()),
                config,
            }
        }

        /// Get configuration
        #[allow(dead_code)]
        pub fn config(&self) -> &KademliaConfig {
            &self.config
        }
    }

    impl Default for KademliaDhtProvider {
        fn default() -> Self {
            Self::new()
        }
    }

    impl DhtProvider for KademliaDhtProvider {
        fn name(&self) -> &str {
            "kademlia"
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        fn capabilities(&self) -> DhtCapabilities {
            DhtCapabilities {
                supports_content_routing: true,
                supports_peer_routing: true,
                supports_kv_storage: true,
                supports_range_queries: false,
                supports_semantic_queries: false,
                max_query_hops: Some(20),
                supports_custom_routing: false,
            }
        }

        fn bootstrap(&self, peers: Vec<PeerId>) -> Result<(), DhtProviderError> {
            // Placeholder: In production, this would connect to bootstrap peers
            let mut stats = self.stats.write();
            stats.routing_table_size = peers.len();
            Ok(())
        }

        fn provide(&self, _cid: &Cid) -> Result<(), DhtProviderError> {
            // Placeholder: In production, this would announce to DHT
            Ok(())
        }

        fn find_providers(&self, _cid: &Cid) -> Result<DhtQueryResult, DhtProviderError> {
            // Placeholder: In production, this would query DHT
            let mut stats = self.stats.write();
            stats.total_queries += 1;
            stats.successful_queries += 1;
            stats.calculate_success_rate();

            Ok(DhtQueryResult {
                providers: vec![],
                hops: 0,
                duration_ms: 0,
                success: true,
            })
        }

        fn find_peer(&self, peer_id: &PeerId) -> Result<DhtPeerInfo, DhtProviderError> {
            // Placeholder: In production, this would query DHT
            Ok(DhtPeerInfo {
                peer_id: *peer_id,
                addresses: vec![],
                distance: None,
            })
        }

        fn get_closest_peers(
            &self,
            _key: &[u8],
            count: usize,
        ) -> Result<Vec<DhtPeerInfo>, DhtProviderError> {
            // Placeholder: In production, this would query routing table
            let _ = count;
            Ok(vec![])
        }

        fn put_value(&self, _key: &[u8], _value: &[u8]) -> Result<(), DhtProviderError> {
            // Placeholder: In production, this would store in DHT
            Ok(())
        }

        fn get_value(&self, _key: &[u8]) -> Result<Vec<u8>, DhtProviderError> {
            // Placeholder: In production, this would retrieve from DHT
            Err(DhtProviderError::QueryFailed("Not found".to_string()))
        }

        fn stats(&self) -> DhtProviderStats {
            self.stats.read().clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::kademlia::*;
    use super::*;

    #[test]
    fn test_dht_capabilities() {
        let basic = DhtCapabilities::basic();
        assert!(basic.supports_content_routing);
        assert!(basic.supports_peer_routing);
        assert!(!basic.supports_kv_storage);

        let advanced = DhtCapabilities::advanced();
        assert!(advanced.supports_content_routing);
        assert!(advanced.supports_kv_storage);
        assert!(advanced.supports_semantic_queries);
    }

    #[test]
    fn test_kademlia_provider() {
        let provider = KademliaDhtProvider::new();
        assert_eq!(provider.name(), "kademlia");
        assert_eq!(provider.version(), "1.0.0");

        let caps = provider.capabilities();
        assert!(caps.supports_content_routing);
        assert!(caps.supports_peer_routing);
        assert!(caps.supports_kv_storage);
    }

    #[test]
    fn test_provider_registry() {
        let mut registry = DhtProviderRegistry::new();
        assert_eq!(registry.count(), 0);

        let provider = Arc::new(KademliaDhtProvider::new());
        registry.register("kademlia", provider);
        assert_eq!(registry.count(), 1);
        assert!(registry.has_provider("kademlia"));
    }

    #[test]
    fn test_registry_active_provider() {
        let mut registry = DhtProviderRegistry::new();
        let provider = Arc::new(KademliaDhtProvider::new());
        registry.register("kademlia", provider);

        let active = registry.get_active();
        assert!(active.is_some());
        assert_eq!(
            active
                .expect("test: active provider should be set after registration")
                .name(),
            "kademlia"
        );
    }

    #[test]
    fn test_registry_set_active() {
        let mut registry = DhtProviderRegistry::new();
        let provider1 = Arc::new(KademliaDhtProvider::new());
        registry.register("kademlia", provider1);

        registry
            .set_active("kademlia")
            .expect("test: set_active should succeed for registered provider");
        assert_eq!(
            registry
                .get_active()
                .expect("test: active provider should be kademlia after set_active")
                .name(),
            "kademlia"
        );
    }

    #[test]
    fn test_registry_unregister() {
        let mut registry = DhtProviderRegistry::new();
        let provider = Arc::new(KademliaDhtProvider::new());
        registry.register("kademlia", provider);

        assert_eq!(registry.count(), 1);
        registry.unregister("kademlia");
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn test_provider_bootstrap() {
        let provider = KademliaDhtProvider::new();
        let peers = vec![PeerId::random(), PeerId::random()];
        let result = provider.bootstrap(peers);
        assert!(result.is_ok());
    }

    #[test]
    fn test_provider_stats() {
        let provider = KademliaDhtProvider::new();
        let stats = provider.stats();
        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.successful_queries, 0);
    }

    #[test]
    fn test_provider_health() {
        let provider = KademliaDhtProvider::new();
        // Initially unhealthy (no peers)
        assert!(!provider.is_healthy());

        // Bootstrap with peers
        provider
            .bootstrap(vec![PeerId::random()])
            .expect("test: bootstrap with one peer should succeed");

        // Perform a query to improve success rate
        let cid = Cid::default();
        provider
            .find_providers(&cid)
            .expect("test: find_providers should succeed on bootstrapped provider");

        // Should be healthy now
        assert!(provider.is_healthy());
    }

    #[test]
    fn test_list_providers() {
        let mut registry = DhtProviderRegistry::new();
        let provider = Arc::new(KademliaDhtProvider::new());
        registry.register("kademlia", provider);

        let providers = registry.list_providers();
        assert_eq!(providers.len(), 1);
        assert!(providers.contains(&"kademlia".to_string()));
    }

    #[test]
    fn test_provider_not_found() {
        let mut registry = DhtProviderRegistry::new();
        let result = registry.set_active("nonexistent");
        assert!(result.is_err());
        assert!(matches!(result, Err(DhtProviderError::ProviderNotFound(_))));
    }

    #[test]
    fn test_stats_success_rate() {
        let mut stats = DhtProviderStats {
            total_queries: 10,
            successful_queries: 7,
            ..Default::default()
        };
        stats.calculate_success_rate();
        assert!((stats.success_rate - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_kademlia_config_default() {
        let config = KademliaConfig::default();
        assert_eq!(config.k_bucket_size, 20);
        assert_eq!(config.alpha, 3);
        assert_eq!(config.replication_factor, 20);
        assert_eq!(config.query_timeout_secs, 60);
    }

    #[test]
    fn test_unsupported_operation() {
        let provider = KademliaDhtProvider::new();

        // This should work (Kademlia supports KV storage)
        let result = provider.put_value(b"key", b"value");
        assert!(result.is_ok());
    }
}
