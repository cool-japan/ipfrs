//! High-level network facade integrating all IPFRS network modules
//!
//! This module provides a unified interface to all network functionality,
//! making it easy to use advanced features without manual wiring.
//!
//! # Example
//!
//! ```rust,no_run
//! use ipfrs_network::NetworkFacadeBuilder;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut facade = NetworkFacadeBuilder::new()
//!         .with_preset_mobile()
//!         .with_semantic_dht()
//!         .with_gossipsub()
//!         .with_geo_routing()
//!         .build()?;
//!
//!     facade.start().await?;
//!
//!     // Use integrated features
//!     let health = facade.get_health();
//!     println!("Network health: {:?}", health.status);
//!
//!     Ok(())
//! }
//! ```

use crate::{
    adaptive_polling::{AdaptivePolling, AdaptivePollingConfig},
    background_mode::{BackgroundModeConfig, BackgroundModeManager},
    dht::{DhtConfig, DhtManager},
    dht_provider::{kademlia::KademliaDhtProvider, DhtProviderRegistry},
    geo_routing::{GeoRouter, GeoRouterConfig},
    gossipsub::{GossipSubConfig, GossipSubManager},
    memory_monitor::{MemoryMonitor, MemoryMonitorConfig},
    multipath_quic::{MultipathConfig, MultipathQuicManager},
    network_monitor::{NetworkMonitor, NetworkMonitorConfig},
    node::{NetworkConfig, NetworkHealthSummary, NetworkNode},
    offline_queue::{OfflineQueue, OfflineQueueConfig},
    peer::{PeerStore, PeerStoreConfig},
    peer_selector::{PeerSelector, PeerSelectorConfig},
    presets::NetworkPreset,
    quality_predictor::{QualityPredictor, QualityPredictorConfig},
    query_batcher::{QueryBatcher, QueryBatcherConfig},
    semantic_dht::{SemanticDht, SemanticDhtConfig},
    throttle::{BandwidthThrottle, ThrottleConfig},
    tor::{TorConfig, TorManager},
};
use libp2p::{Multiaddr, PeerId};
use parking_lot::RwLock;
use std::sync::Arc;
use tokio::sync::RwLock as AsyncRwLock;

type IpfrsResult<T> = ipfrs_core::error::Result<T>;

/// Comprehensive network statistics
#[derive(Debug, Clone)]
pub struct NetworkStats {
    pub peer_count: usize,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub is_healthy: bool,
    pub semantic_dht_enabled: bool,
    pub gossipsub_enabled: bool,
    pub geo_routing_enabled: bool,
    pub tor_enabled: bool,
    pub bandwidth_throttle_enabled: bool,
    pub memory_monitor_enabled: bool,
}

/// Detailed module status
#[derive(Debug, Clone)]
pub struct ModuleStatus {
    pub semantic_dht: bool,
    pub gossipsub: bool,
    pub geo_routing: bool,
    pub quality_predictor: bool,
    pub peer_selector: bool,
    pub multipath_quic: bool,
    pub tor: bool,
    pub bandwidth_throttle: bool,
    pub adaptive_polling: bool,
    pub background_mode: bool,
    pub offline_queue: bool,
    pub memory_monitor: bool,
    pub network_monitor: bool,
    pub query_batcher: bool,
}

/// High-level network facade integrating all modules
pub struct NetworkFacade {
    /// Core network node
    pub node: NetworkNode,

    /// Optional modules
    pub semantic_dht: Option<Arc<RwLock<SemanticDht>>>,
    pub gossipsub: Option<Arc<RwLock<GossipSubManager>>>,
    pub geo_router: Option<Arc<RwLock<GeoRouter>>>,
    pub quality_predictor: Option<Arc<RwLock<QualityPredictor>>>,
    pub peer_selector: Option<Arc<RwLock<PeerSelector>>>,
    pub multipath_quic: Option<Arc<RwLock<MultipathQuicManager>>>,
    pub tor_manager: Option<Arc<AsyncRwLock<TorManager>>>,
    pub bandwidth_throttle: Option<Arc<RwLock<BandwidthThrottle>>>,
    pub adaptive_polling: Option<Arc<RwLock<AdaptivePolling>>>,
    pub background_mode: Option<Arc<RwLock<BackgroundModeManager>>>,
    pub offline_queue: Option<Arc<RwLock<OfflineQueue>>>,
    pub memory_monitor: Option<Arc<RwLock<MemoryMonitor>>>,
    pub network_monitor: Option<Arc<RwLock<NetworkMonitor>>>,
    pub query_batcher: Option<Arc<RwLock<QueryBatcher>>>,

    /// Always-available supporting modules
    pub peer_store: Arc<RwLock<PeerStore>>,
    pub dht_manager: Arc<RwLock<DhtManager>>,
    pub dht_provider_registry: Arc<RwLock<DhtProviderRegistry>>,
}

impl NetworkFacade {
    /// Create a new network facade with default configuration
    pub fn new(config: NetworkConfig) -> IpfrsResult<Self> {
        let node = NetworkNode::new(config)?;
        let peer_store = Arc::new(RwLock::new(PeerStore::with_config(
            PeerStoreConfig::default(),
        )));
        let dht_manager = Arc::new(RwLock::new(DhtManager::new(DhtConfig::default())));
        let dht_provider_registry = Arc::new(RwLock::new(DhtProviderRegistry::new()));

        Ok(Self {
            node,
            semantic_dht: None,
            gossipsub: None,
            geo_router: None,
            quality_predictor: None,
            peer_selector: None,
            multipath_quic: None,
            tor_manager: None,
            bandwidth_throttle: None,
            adaptive_polling: None,
            background_mode: None,
            offline_queue: None,
            memory_monitor: None,
            network_monitor: None,
            query_batcher: None,
            peer_store,
            dht_manager,
            dht_provider_registry,
        })
    }

    /// Start the network node and all enabled modules
    pub async fn start(&mut self) -> IpfrsResult<()> {
        self.node.start().await?;

        // Start Tor if enabled
        if let Some(tor) = &self.tor_manager {
            tor.write().await.start().await.map_err(|e| {
                ipfrs_core::error::Error::Network(format!("Failed to start Tor: {}", e))
            })?;
        }

        Ok(())
    }

    /// Stop the network node and all modules
    pub async fn stop(&mut self) -> IpfrsResult<()> {
        // Stop Tor if enabled
        if let Some(tor) = &self.tor_manager {
            tor.write().await.stop().await.map_err(|e| {
                ipfrs_core::error::Error::Network(format!("Failed to stop Tor: {}", e))
            })?;
        }

        self.node.stop().await
    }

    /// Get peer ID
    pub fn peer_id(&self) -> PeerId {
        self.node.peer_id()
    }

    /// Get connected peers
    pub fn connected_peers(&self) -> Vec<PeerId> {
        self.node.connected_peers()
    }

    /// Connect to a peer
    pub async fn connect(&mut self, addr: Multiaddr) -> IpfrsResult<()> {
        self.node.connect(addr).await
    }

    /// Disconnect from a peer
    pub async fn disconnect(&mut self, peer_id: PeerId) -> IpfrsResult<()> {
        self.node.disconnect(peer_id).await
    }

    /// Provide content to the DHT
    pub async fn provide(&mut self, cid: &cid::Cid) -> IpfrsResult<()> {
        self.node.provide(cid).await
    }

    /// Find providers for content
    pub async fn find_providers(&mut self, cid: &cid::Cid) -> IpfrsResult<()> {
        self.node.find_providers(cid).await
    }

    /// Get network health summary
    pub fn get_health(&self) -> NetworkHealthSummary {
        self.node.get_network_health()
    }

    /// Check if network is healthy
    pub fn is_healthy(&self) -> bool {
        self.node.is_healthy()
    }

    /// Get peer count
    pub fn peer_count(&self) -> usize {
        self.node.get_peer_count()
    }

    /// Check if connected to peer
    pub fn is_connected_to(&self, peer_id: &PeerId) -> bool {
        self.node.is_connected_to(peer_id)
    }

    /// Get bytes sent
    pub fn bytes_sent(&self) -> u64 {
        self.node.get_bytes_sent()
    }

    /// Get bytes received
    pub fn bytes_received(&self) -> u64 {
        self.node.get_bytes_received()
    }

    /// Add a Tor manager after creation (requires async initialization)
    pub async fn with_tor_manager(
        &mut self,
        config: TorConfig,
    ) -> Result<(), crate::tor::TorError> {
        let manager = TorManager::new(config).await?;
        self.tor_manager = Some(Arc::new(AsyncRwLock::new(manager)));
        Ok(())
    }

    // ============================================================================
    // Semantic DHT Operations (when enabled)
    // ============================================================================

    /// Perform semantic search using vector embeddings (requires semantic_dht)
    ///
    /// # Returns
    /// Returns an error if semantic DHT is not enabled
    pub fn semantic_search(
        &self,
        namespace: &crate::semantic_dht::NamespaceId,
        embedding: Vec<f32>,
        top_k: usize,
    ) -> IpfrsResult<Vec<crate::semantic_dht::SemanticResult>> {
        let dht = self.semantic_dht.as_ref().ok_or_else(|| {
            ipfrs_core::error::Error::Network("Semantic DHT not enabled".to_string())
        })?;

        let query = crate::semantic_dht::SemanticQuery {
            embedding,
            namespace: namespace.clone(),
            top_k,
            metadata_filter: None,
            timeout: std::time::Duration::from_secs(30),
        };

        dht.read().query(query).map_err(|e| {
            ipfrs_core::error::Error::Network(format!("Semantic search failed: {}", e))
        })
    }

    /// Index content with semantic embedding (requires semantic_dht)
    pub fn index_content(
        &self,
        cid: cid::Cid,
        embedding: Vec<f32>,
        namespace: crate::semantic_dht::NamespaceId,
    ) -> IpfrsResult<()> {
        let dht = self.semantic_dht.as_ref().ok_or_else(|| {
            ipfrs_core::error::Error::Network("Semantic DHT not enabled".to_string())
        })?;

        dht.write()
            .index_content(cid, embedding, namespace)
            .map_err(|e| {
                ipfrs_core::error::Error::Network(format!("Semantic indexing failed: {}", e))
            })
    }

    /// Register a semantic namespace (requires semantic_dht)
    pub fn register_semantic_namespace(
        &self,
        namespace: crate::semantic_dht::NamespaceId,
        dimension: usize,
    ) -> IpfrsResult<()> {
        let dht = self.semantic_dht.as_ref().ok_or_else(|| {
            ipfrs_core::error::Error::Network("Semantic DHT not enabled".to_string())
        })?;

        let ns = crate::semantic_dht::SemanticNamespace {
            id: namespace,
            dimension,
            distance_metric: crate::semantic_dht::DistanceMetric::Cosine,
            lsh_config: Default::default(),
        };

        dht.write().register_namespace(ns).map_err(|e| {
            ipfrs_core::error::Error::Network(format!("Namespace registration failed: {}", e))
        })
    }

    // ============================================================================
    // GossipSub Operations (when enabled)
    // ============================================================================

    /// Subscribe to a topic (requires gossipsub)
    pub fn subscribe(&self, topic: &str) -> IpfrsResult<()> {
        let gossipsub = self.gossipsub.as_ref().ok_or_else(|| {
            ipfrs_core::error::Error::Network("GossipSub not enabled".to_string())
        })?;

        let topic_id = crate::gossipsub::TopicId::new(topic);
        gossipsub
            .write()
            .subscribe(topic_id)
            .map_err(|e| ipfrs_core::error::Error::Network(format!("Subscribe failed: {}", e)))
    }

    /// Unsubscribe from a topic (requires gossipsub)
    pub fn unsubscribe(&self, topic: &str) -> IpfrsResult<()> {
        let gossipsub = self.gossipsub.as_ref().ok_or_else(|| {
            ipfrs_core::error::Error::Network("GossipSub not enabled".to_string())
        })?;

        let topic_id = crate::gossipsub::TopicId::new(topic);
        gossipsub
            .write()
            .unsubscribe(&topic_id)
            .map_err(|e| ipfrs_core::error::Error::Network(format!("Unsubscribe failed: {}", e)))
    }

    /// Publish a message to a topic (requires gossipsub)
    pub fn publish(&self, topic: &str, data: Vec<u8>) -> IpfrsResult<crate::gossipsub::MessageId> {
        let gossipsub = self.gossipsub.as_ref().ok_or_else(|| {
            ipfrs_core::error::Error::Network("GossipSub not enabled".to_string())
        })?;

        let topic_id = crate::gossipsub::TopicId::new(topic);
        let source = self.peer_id();
        gossipsub
            .write()
            .publish(topic_id, data, source)
            .map_err(|e| ipfrs_core::error::Error::Network(format!("Publish failed: {}", e)))
    }

    /// Get subscribed topics (requires gossipsub)
    pub fn subscribed_topics(&self) -> IpfrsResult<Vec<String>> {
        let _gossipsub = self.gossipsub.as_ref().ok_or_else(|| {
            ipfrs_core::error::Error::Network("GossipSub not enabled".to_string())
        })?;

        // Note: subscriptions field is private, so we return an empty list for now
        // This would need to be exposed by the GossipSubManager API
        Ok(Vec::new())
    }

    // ============================================================================
    // Geographic & Quality-Based Operations (when enabled)
    // ============================================================================

    /// Find nearby peers based on geographic location (requires geo_routing)
    pub fn find_nearby_peers(
        &self,
        location: crate::geo_routing::GeoLocation,
    ) -> IpfrsResult<Vec<PeerId>> {
        let geo_router = self.geo_router.as_ref().ok_or_else(|| {
            ipfrs_core::error::Error::Network("Geographic routing not enabled".to_string())
        })?;

        Ok(geo_router
            .read()
            .get_nearby_peers(&location)
            .iter()
            .map(|p| p.peer_id)
            .collect())
    }

    /// Set location for a peer (requires geo_routing)
    pub fn set_peer_location(
        &self,
        peer_id: PeerId,
        location: crate::geo_routing::GeoLocation,
    ) -> IpfrsResult<()> {
        let geo_router = self.geo_router.as_ref().ok_or_else(|| {
            ipfrs_core::error::Error::Network("Geographic routing not enabled".to_string())
        })?;

        geo_router.read().update_peer_location(peer_id, location);
        Ok(())
    }

    /// Get best quality peers for content transfer (requires quality_predictor)
    pub fn get_best_peers(&self, peers: &[PeerId], count: usize) -> IpfrsResult<Vec<PeerId>> {
        let predictor = self.quality_predictor.as_ref().ok_or_else(|| {
            ipfrs_core::error::Error::Network("Quality predictor not enabled".to_string())
        })?;

        Ok(predictor
            .read()
            .rank_peers(peers)
            .into_iter()
            .take(count)
            .map(|(peer_id, _)| peer_id)
            .collect())
    }

    /// Select optimal peers (requires peer_selector)
    pub fn select_optimal_peers(
        &self,
        criteria: &crate::peer_selector::SelectionCriteria,
    ) -> IpfrsResult<Vec<PeerId>> {
        let selector = self.peer_selector.as_ref().ok_or_else(|| {
            ipfrs_core::error::Error::Network("Peer selector not enabled".to_string())
        })?;

        Ok(selector
            .read()
            .select_peers(criteria)
            .into_iter()
            .map(|p| p.peer_id)
            .collect())
    }

    // ============================================================================
    // Network Diagnostics & Statistics
    // ============================================================================

    /// Get comprehensive network statistics
    pub fn get_network_stats(&self) -> NetworkStats {
        NetworkStats {
            peer_count: self.peer_count(),
            bytes_sent: self.bytes_sent(),
            bytes_received: self.bytes_received(),
            is_healthy: self.is_healthy(),
            semantic_dht_enabled: self.semantic_dht.is_some(),
            gossipsub_enabled: self.gossipsub.is_some(),
            geo_routing_enabled: self.geo_router.is_some(),
            tor_enabled: self.tor_manager.is_some(),
            bandwidth_throttle_enabled: self.bandwidth_throttle.is_some(),
            memory_monitor_enabled: self.memory_monitor.is_some(),
        }
    }

    /// Get detailed module status
    pub fn get_module_status(&self) -> ModuleStatus {
        ModuleStatus {
            semantic_dht: self.semantic_dht.is_some(),
            gossipsub: self.gossipsub.is_some(),
            geo_routing: self.geo_router.is_some(),
            quality_predictor: self.quality_predictor.is_some(),
            peer_selector: self.peer_selector.is_some(),
            multipath_quic: self.multipath_quic.is_some(),
            tor: self.tor_manager.is_some(),
            bandwidth_throttle: self.bandwidth_throttle.is_some(),
            adaptive_polling: self.adaptive_polling.is_some(),
            background_mode: self.background_mode.is_some(),
            offline_queue: self.offline_queue.is_some(),
            memory_monitor: self.memory_monitor.is_some(),
            network_monitor: self.network_monitor.is_some(),
            query_batcher: self.query_batcher.is_some(),
        }
    }

    /// Get memory usage statistics (requires memory_monitor)
    pub fn get_memory_stats(&self) -> IpfrsResult<crate::memory_monitor::MemoryStats> {
        let monitor = self.memory_monitor.as_ref().ok_or_else(|| {
            ipfrs_core::error::Error::Network("Memory monitor not enabled".to_string())
        })?;

        Ok(monitor.read().stats())
    }

    // ============================================================================
    // Batch Operations
    // ============================================================================

    /// Connect to multiple peers concurrently
    pub async fn connect_batch(&mut self, addrs: Vec<Multiaddr>) -> Vec<IpfrsResult<()>> {
        self.node.connect_to_peers(addrs).await
    }

    /// Announce multiple CIDs to the DHT
    pub async fn provide_batch(&mut self, cids: Vec<cid::Cid>) -> Vec<IpfrsResult<()>> {
        let mut results = Vec::new();
        for cid in cids {
            results.push(self.provide(&cid).await);
        }
        results
    }

    /// Find providers for multiple CIDs
    pub async fn find_providers_batch(&mut self, cids: Vec<cid::Cid>) -> Vec<IpfrsResult<()>> {
        let mut results = Vec::new();
        for cid in cids {
            results.push(self.find_providers(&cid).await);
        }
        results
    }

    // ============================================================================
    // Configuration & Inspection
    // ============================================================================

    /// Check if a specific module is enabled
    pub fn is_module_enabled(&self, module: &str) -> bool {
        match module {
            "semantic_dht" => self.semantic_dht.is_some(),
            "gossipsub" => self.gossipsub.is_some(),
            "geo_routing" => self.geo_router.is_some(),
            "quality_predictor" => self.quality_predictor.is_some(),
            "peer_selector" => self.peer_selector.is_some(),
            "multipath_quic" => self.multipath_quic.is_some(),
            "tor" => self.tor_manager.is_some(),
            "bandwidth_throttle" => self.bandwidth_throttle.is_some(),
            "adaptive_polling" => self.adaptive_polling.is_some(),
            "background_mode" => self.background_mode.is_some(),
            "offline_queue" => self.offline_queue.is_some(),
            "memory_monitor" => self.memory_monitor.is_some(),
            "network_monitor" => self.network_monitor.is_some(),
            "query_batcher" => self.query_batcher.is_some(),
            _ => false,
        }
    }

    /// Get list of enabled module names
    pub fn enabled_modules(&self) -> Vec<String> {
        let modules = vec![
            ("semantic_dht", self.semantic_dht.is_some()),
            ("gossipsub", self.gossipsub.is_some()),
            ("geo_routing", self.geo_router.is_some()),
            ("quality_predictor", self.quality_predictor.is_some()),
            ("peer_selector", self.peer_selector.is_some()),
            ("multipath_quic", self.multipath_quic.is_some()),
            ("tor", self.tor_manager.is_some()),
            ("bandwidth_throttle", self.bandwidth_throttle.is_some()),
            ("adaptive_polling", self.adaptive_polling.is_some()),
            ("background_mode", self.background_mode.is_some()),
            ("offline_queue", self.offline_queue.is_some()),
            ("memory_monitor", self.memory_monitor.is_some()),
            ("network_monitor", self.network_monitor.is_some()),
            ("query_batcher", self.query_batcher.is_some()),
        ];

        modules
            .into_iter()
            .filter_map(|(name, enabled)| {
                if enabled {
                    Some(name.to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get human-readable summary of the network facade configuration
    pub fn summary(&self) -> String {
        let enabled = self.enabled_modules();
        format!(
            "NetworkFacade {{\n  Peer ID: {}\n  Peers: {}\n  Health: {:?}\n  Enabled modules ({}):\n    {}\n}}",
            self.peer_id(),
            self.peer_count(),
            self.get_health().status,
            enabled.len(),
            enabled.join(", ")
        )
    }
}

/// Builder for NetworkFacade
pub struct NetworkFacadeBuilder {
    config: NetworkConfig,
    enable_semantic_dht: bool,
    enable_gossipsub: bool,
    enable_geo_routing: bool,
    enable_quality_predictor: bool,
    enable_peer_selector: bool,
    enable_multipath_quic: bool,
    enable_tor: bool,
    enable_bandwidth_throttle: bool,
    enable_adaptive_polling: bool,
    enable_background_mode: bool,
    enable_offline_queue: bool,
    enable_memory_monitor: bool,
    enable_network_monitor: bool,
    enable_query_batcher: bool,

    semantic_dht_config: Option<SemanticDhtConfig>,
    gossipsub_config: Option<GossipSubConfig>,
    geo_router_config: Option<GeoRouterConfig>,
    quality_predictor_config: Option<QualityPredictorConfig>,
    peer_selector_config: Option<PeerSelectorConfig>,
    multipath_config: Option<MultipathConfig>,
    tor_config: Option<TorConfig>,
    throttle_config: Option<ThrottleConfig>,
    adaptive_polling_config: Option<AdaptivePollingConfig>,
    background_mode_config: Option<BackgroundModeConfig>,
    offline_queue_config: Option<OfflineQueueConfig>,
    memory_monitor_config: Option<MemoryMonitorConfig>,
    network_monitor_config: Option<NetworkMonitorConfig>,
    query_batcher_config: Option<QueryBatcherConfig>,
    peer_store_config: Option<PeerStoreConfig>,
    dht_config: Option<DhtConfig>,
}

impl NetworkFacadeBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: NetworkConfig::default(),
            enable_semantic_dht: false,
            enable_gossipsub: false,
            enable_geo_routing: false,
            enable_quality_predictor: false,
            enable_peer_selector: false,
            enable_multipath_quic: false,
            enable_tor: false,
            enable_bandwidth_throttle: false,
            enable_adaptive_polling: false,
            enable_background_mode: false,
            enable_offline_queue: false,
            enable_memory_monitor: false,
            enable_network_monitor: false,
            enable_query_batcher: false,
            semantic_dht_config: None,
            gossipsub_config: None,
            geo_router_config: None,
            quality_predictor_config: None,
            peer_selector_config: None,
            multipath_config: None,
            tor_config: None,
            throttle_config: None,
            adaptive_polling_config: None,
            background_mode_config: None,
            offline_queue_config: None,
            memory_monitor_config: None,
            network_monitor_config: None,
            query_batcher_config: None,
            peer_store_config: None,
            dht_config: None,
        }
    }

    /// Use a preset configuration
    pub fn with_preset(mut self, preset: NetworkPreset) -> Self {
        self.config = preset.network;

        if let Some(config) = preset.throttle {
            self.throttle_config = Some(config);
            self.enable_bandwidth_throttle = true;
        }

        if let Some(config) = preset.adaptive_polling {
            self.adaptive_polling_config = Some(config);
            self.enable_adaptive_polling = true;
        }

        if let Some(config) = preset.memory_monitor {
            self.memory_monitor_config = Some(config);
            self.enable_memory_monitor = true;
        }

        if let Some(config) = preset.offline_queue {
            self.offline_queue_config = Some(config);
            self.enable_offline_queue = true;
        }

        if let Some(config) = preset.background_mode {
            self.background_mode_config = Some(config);
            self.enable_background_mode = true;
        }

        if let Some(config) = preset.query_batcher {
            self.query_batcher_config = Some(config);
            self.enable_query_batcher = true;
        }

        if let Some(config) = preset.geo_router {
            self.geo_router_config = Some(config);
            self.enable_geo_routing = true;
        }

        if let Some(config) = preset.quality_predictor {
            self.quality_predictor_config = Some(config);
            self.enable_quality_predictor = true;
        }

        if let Some(config) = preset.peer_selector {
            self.peer_selector_config = Some(config);
            self.enable_peer_selector = true;
        }

        if let Some(config) = preset.multipath {
            self.multipath_config = Some(config);
            self.enable_multipath_quic = true;
        }

        if let Some(config) = preset.tor {
            self.tor_config = Some(config);
            self.enable_tor = true;
        }

        self.peer_store_config = Some(preset.peer_store);
        self.dht_config = Some(preset.dht);

        self
    }

    /// Use mobile preset
    pub fn with_preset_mobile(self) -> Self {
        self.with_preset(NetworkPreset::mobile())
    }

    /// Use IoT preset
    pub fn with_preset_iot(self) -> Self {
        self.with_preset(NetworkPreset::iot())
    }

    /// Use low-memory preset
    pub fn with_preset_low_memory(self) -> Self {
        self.with_preset(NetworkPreset::low_memory())
    }

    /// Use high-performance preset
    pub fn with_preset_high_performance(self) -> Self {
        self.with_preset(NetworkPreset::high_performance())
    }

    /// Use privacy preset
    pub fn with_preset_privacy(self) -> Self {
        self.with_preset(NetworkPreset::privacy())
    }

    /// Set custom network configuration
    pub fn with_config(mut self, config: NetworkConfig) -> Self {
        self.config = config;
        self
    }

    /// Enable semantic DHT
    pub fn with_semantic_dht(mut self) -> Self {
        self.enable_semantic_dht = true;
        self
    }

    /// Enable semantic DHT with custom config
    pub fn with_semantic_dht_config(mut self, config: SemanticDhtConfig) -> Self {
        self.enable_semantic_dht = true;
        self.semantic_dht_config = Some(config);
        self
    }

    /// Enable GossipSub
    pub fn with_gossipsub(mut self) -> Self {
        self.enable_gossipsub = true;
        self
    }

    /// Enable GossipSub with custom config
    pub fn with_gossipsub_config(mut self, config: GossipSubConfig) -> Self {
        self.enable_gossipsub = true;
        self.gossipsub_config = Some(config);
        self
    }

    /// Enable geographic routing
    pub fn with_geo_routing(mut self) -> Self {
        self.enable_geo_routing = true;
        self
    }

    /// Enable geographic routing with custom config
    pub fn with_geo_routing_config(mut self, config: GeoRouterConfig) -> Self {
        self.enable_geo_routing = true;
        self.geo_router_config = Some(config);
        self
    }

    /// Enable quality prediction
    pub fn with_quality_predictor(mut self) -> Self {
        self.enable_quality_predictor = true;
        self
    }

    /// Enable quality prediction with custom config
    pub fn with_quality_predictor_config(mut self, config: QualityPredictorConfig) -> Self {
        self.enable_quality_predictor = true;
        self.quality_predictor_config = Some(config);
        self
    }

    /// Enable intelligent peer selection
    pub fn with_peer_selector(mut self) -> Self {
        self.enable_peer_selector = true;
        self
    }

    /// Enable intelligent peer selection with custom config
    pub fn with_peer_selector_config(mut self, config: PeerSelectorConfig) -> Self {
        self.enable_peer_selector = true;
        self.peer_selector_config = Some(config);
        self
    }

    /// Enable multipath QUIC
    pub fn with_multipath_quic(mut self) -> Self {
        self.enable_multipath_quic = true;
        self
    }

    /// Enable multipath QUIC with custom config
    pub fn with_multipath_quic_config(mut self, config: MultipathConfig) -> Self {
        self.enable_multipath_quic = true;
        self.multipath_config = Some(config);
        self
    }

    /// Enable Tor integration
    pub fn with_tor(mut self) -> Self {
        self.enable_tor = true;
        self
    }

    /// Enable Tor with custom config
    pub fn with_tor_config(mut self, config: TorConfig) -> Self {
        self.enable_tor = true;
        self.tor_config = Some(config);
        self
    }

    /// Enable bandwidth throttling
    pub fn with_bandwidth_throttle(mut self) -> Self {
        self.enable_bandwidth_throttle = true;
        self
    }

    /// Enable bandwidth throttling with custom config
    pub fn with_bandwidth_throttle_config(mut self, config: ThrottleConfig) -> Self {
        self.enable_bandwidth_throttle = true;
        self.throttle_config = Some(config);
        self
    }

    /// Enable adaptive polling
    pub fn with_adaptive_polling(mut self) -> Self {
        self.enable_adaptive_polling = true;
        self
    }

    /// Enable adaptive polling with custom config
    pub fn with_adaptive_polling_config(mut self, config: AdaptivePollingConfig) -> Self {
        self.enable_adaptive_polling = true;
        self.adaptive_polling_config = Some(config);
        self
    }

    /// Enable background mode
    pub fn with_background_mode(mut self) -> Self {
        self.enable_background_mode = true;
        self
    }

    /// Enable background mode with custom config
    pub fn with_background_mode_config(mut self, config: BackgroundModeConfig) -> Self {
        self.enable_background_mode = true;
        self.background_mode_config = Some(config);
        self
    }

    /// Enable offline queue
    pub fn with_offline_queue(mut self) -> Self {
        self.enable_offline_queue = true;
        self
    }

    /// Enable offline queue with custom config
    pub fn with_offline_queue_config(mut self, config: OfflineQueueConfig) -> Self {
        self.enable_offline_queue = true;
        self.offline_queue_config = Some(config);
        self
    }

    /// Enable memory monitoring
    pub fn with_memory_monitor(mut self) -> Self {
        self.enable_memory_monitor = true;
        self
    }

    /// Enable memory monitoring with custom config
    pub fn with_memory_monitor_config(mut self, config: MemoryMonitorConfig) -> Self {
        self.enable_memory_monitor = true;
        self.memory_monitor_config = Some(config);
        self
    }

    /// Enable network monitoring
    pub fn with_network_monitor(mut self) -> Self {
        self.enable_network_monitor = true;
        self
    }

    /// Enable network monitoring with custom config
    pub fn with_network_monitor_config(mut self, config: NetworkMonitorConfig) -> Self {
        self.enable_network_monitor = true;
        self.network_monitor_config = Some(config);
        self
    }

    /// Enable query batching
    pub fn with_query_batcher(mut self) -> Self {
        self.enable_query_batcher = true;
        self
    }

    /// Enable query batching with custom config
    pub fn with_query_batcher_config(mut self, config: QueryBatcherConfig) -> Self {
        self.enable_query_batcher = true;
        self.query_batcher_config = Some(config);
        self
    }

    /// Build the network facade
    pub fn build(self) -> IpfrsResult<NetworkFacade> {
        let node = NetworkNode::new(self.config)?;

        let peer_store = Arc::new(RwLock::new(if let Some(config) = self.peer_store_config {
            PeerStore::with_config(config)
        } else {
            PeerStore::with_config(PeerStoreConfig::default())
        }));

        let dht_manager = Arc::new(RwLock::new(if let Some(config) = self.dht_config {
            DhtManager::new(config)
        } else {
            DhtManager::new(DhtConfig::default())
        }));

        let dht_provider_registry = Arc::new(RwLock::new({
            let mut registry = DhtProviderRegistry::new();
            registry.register("kademlia", Arc::new(KademliaDhtProvider::new()));
            registry
        }));

        Ok(NetworkFacade {
            node,
            semantic_dht: if self.enable_semantic_dht {
                Some(Arc::new(RwLock::new(SemanticDht::new(
                    self.semantic_dht_config.unwrap_or_default(),
                ))))
            } else {
                None
            },
            gossipsub: if self.enable_gossipsub {
                Some(Arc::new(RwLock::new(GossipSubManager::new(
                    self.gossipsub_config.unwrap_or_default(),
                ))))
            } else {
                None
            },
            geo_router: if self.enable_geo_routing {
                Some(Arc::new(RwLock::new(GeoRouter::new(
                    self.geo_router_config.unwrap_or_default(),
                ))))
            } else {
                None
            },
            quality_predictor: if self.enable_quality_predictor {
                let config = self.quality_predictor_config.unwrap_or_default();
                Some(Arc::new(RwLock::new(
                    QualityPredictor::new(config).map_err(|e| {
                        ipfrs_core::error::Error::Network(format!(
                            "Failed to create quality predictor: {}",
                            e
                        ))
                    })?,
                )))
            } else {
                None
            },
            peer_selector: if self.enable_peer_selector {
                Some(Arc::new(RwLock::new(PeerSelector::new(
                    self.peer_selector_config.unwrap_or_default(),
                ))))
            } else {
                None
            },
            multipath_quic: if self.enable_multipath_quic {
                Some(Arc::new(RwLock::new(MultipathQuicManager::new(
                    self.multipath_config.unwrap_or_default(),
                ))))
            } else {
                None
            },
            // Note: TorManager requires async initialization, so it's not created in the builder
            // Users should use with_tor_manager() after building the facade if Tor is needed
            tor_manager: None,
            bandwidth_throttle: if self.enable_bandwidth_throttle {
                let config = self.throttle_config.unwrap_or_default();
                Some(Arc::new(RwLock::new(
                    BandwidthThrottle::new(config).map_err(|e| {
                        ipfrs_core::error::Error::Network(format!(
                            "Failed to create bandwidth throttle: {}",
                            e
                        ))
                    })?,
                )))
            } else {
                None
            },
            adaptive_polling: if self.enable_adaptive_polling {
                let config = self.adaptive_polling_config.unwrap_or_default();
                Some(Arc::new(RwLock::new(
                    AdaptivePolling::new(config).map_err(|e| {
                        ipfrs_core::error::Error::Network(format!(
                            "Failed to create adaptive polling: {}",
                            e
                        ))
                    })?,
                )))
            } else {
                None
            },
            background_mode: if self.enable_background_mode {
                Some(Arc::new(RwLock::new(BackgroundModeManager::new(
                    self.background_mode_config.unwrap_or_default(),
                ))))
            } else {
                None
            },
            offline_queue: if self.enable_offline_queue {
                let config = self.offline_queue_config.unwrap_or_default();
                Some(Arc::new(RwLock::new(OfflineQueue::new(config).map_err(
                    |e| {
                        ipfrs_core::error::Error::Network(format!(
                            "Failed to create offline queue: {}",
                            e
                        ))
                    },
                )?)))
            } else {
                None
            },
            memory_monitor: if self.enable_memory_monitor {
                let config = self.memory_monitor_config.unwrap_or_default();
                Some(Arc::new(RwLock::new(MemoryMonitor::new(config).map_err(
                    |e| {
                        ipfrs_core::error::Error::Network(format!(
                            "Failed to create memory monitor: {}",
                            e
                        ))
                    },
                )?)))
            } else {
                None
            },
            network_monitor: if self.enable_network_monitor {
                let config = self.network_monitor_config.unwrap_or_default();
                Some(Arc::new(RwLock::new(NetworkMonitor::new(config))))
            } else {
                None
            },
            query_batcher: if self.enable_query_batcher {
                let config = self.query_batcher_config.unwrap_or_default();
                Some(Arc::new(RwLock::new(QueryBatcher::new(config).map_err(
                    |e| {
                        ipfrs_core::error::Error::Network(format!(
                            "Failed to create query batcher: {}",
                            e
                        ))
                    },
                )?)))
            } else {
                None
            },
            peer_store,
            dht_manager,
            dht_provider_registry,
        })
    }
}

impl Default for NetworkFacadeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_default() {
        let builder = NetworkFacadeBuilder::new();
        assert!(!builder.enable_semantic_dht);
        assert!(!builder.enable_gossipsub);
    }

    #[test]
    fn test_builder_with_features() {
        let builder = NetworkFacadeBuilder::new()
            .with_semantic_dht()
            .with_gossipsub()
            .with_geo_routing();

        assert!(builder.enable_semantic_dht);
        assert!(builder.enable_gossipsub);
        assert!(builder.enable_geo_routing);
    }

    #[test]
    fn test_builder_with_mobile_preset() {
        let builder = NetworkFacadeBuilder::new().with_preset_mobile();
        assert!(builder.enable_bandwidth_throttle);
        assert!(builder.enable_adaptive_polling);
    }

    #[test]
    fn test_builder_with_iot_preset() {
        let builder = NetworkFacadeBuilder::new().with_preset_iot();
        assert!(builder.enable_bandwidth_throttle);
        assert!(builder.enable_query_batcher);
    }

    #[test]
    fn test_builder_with_privacy_preset() {
        let builder = NetworkFacadeBuilder::new().with_preset_privacy();
        assert!(builder.enable_tor);
    }

    #[tokio::test]
    async fn test_facade_creation() {
        let result = NetworkFacadeBuilder::new().build();
        assert!(result.is_ok());

        let facade = result.expect("test: facade creation should succeed");
        assert!(facade.semantic_dht.is_none());
        assert!(facade.gossipsub.is_none());
    }

    #[tokio::test]
    async fn test_facade_with_semantic_dht() {
        let result = NetworkFacadeBuilder::new().with_semantic_dht().build();
        assert!(result.is_ok());

        let facade = result.expect("test: facade with semantic DHT should be created successfully");
        assert!(facade.semantic_dht.is_some());
    }

    #[tokio::test]
    async fn test_facade_with_all_features() {
        let result = NetworkFacadeBuilder::new()
            .with_semantic_dht()
            .with_gossipsub()
            .with_geo_routing()
            .with_quality_predictor()
            .with_bandwidth_throttle()
            .with_adaptive_polling()
            .with_memory_monitor()
            .with_network_monitor()
            .with_query_batcher()
            .build();

        assert!(result.is_ok());

        let facade = result.expect("test: facade with all features should be created successfully");
        assert!(facade.semantic_dht.is_some());
        assert!(facade.gossipsub.is_some());
        assert!(facade.geo_router.is_some());
        assert!(facade.quality_predictor.is_some());
        assert!(facade.bandwidth_throttle.is_some());
        assert!(facade.adaptive_polling.is_some());
        assert!(facade.memory_monitor.is_some());
        assert!(facade.network_monitor.is_some());
        assert!(facade.query_batcher.is_some());
    }

    #[tokio::test]
    async fn test_facade_peer_id() {
        let facade = NetworkFacadeBuilder::new()
            .build()
            .expect("test: facade build should succeed for peer_id test");
        let peer_id = facade.peer_id();
        assert!(!peer_id.to_string().is_empty());
    }

    #[tokio::test]
    async fn test_facade_connected_peers_empty() {
        let facade = NetworkFacadeBuilder::new()
            .build()
            .expect("test: facade build should succeed for connected peers test");
        let peers = facade.connected_peers();
        assert_eq!(peers.len(), 0);
    }

    #[tokio::test]
    async fn test_facade_peer_count_zero() {
        let facade = NetworkFacadeBuilder::new()
            .build()
            .expect("test: facade build should succeed for peer count test");
        assert_eq!(facade.peer_count(), 0);
    }

    #[tokio::test]
    async fn test_facade_health() {
        let facade = NetworkFacadeBuilder::new()
            .build()
            .expect("test: facade build should succeed for health test");
        let health = facade.get_health();
        assert!(matches!(health.status, _));
    }

    #[tokio::test]
    async fn test_facade_bandwidth_stats() {
        let facade = NetworkFacadeBuilder::new()
            .build()
            .expect("test: facade build should succeed for bandwidth stats test");
        assert_eq!(facade.bytes_sent(), 0);
        assert_eq!(facade.bytes_received(), 0);
    }
}
