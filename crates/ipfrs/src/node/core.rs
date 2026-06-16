//! Node core: configuration, lifecycle, status, diagnostics

use ipfrs_core::{Error, Result};
use ipfrs_network::{NetworkConfig, NetworkNode};
use ipfrs_semantic::RouterConfig;
use ipfrs_storage::{BlockStoreConfig, BlockStoreTrait, CachedBlockStore, SledBlockStore};
use once_cell::sync::OnceCell;
use std::sync::Arc;
use std::time::SystemTime;

use crate::auth::AuthManager;
use crate::diagnostics::{
    HealthStatus, NetworkDiagnostics, NodeDiagnostics, ResourceUsage, SemanticDiagnostics,
    StorageDiagnostics, TensorLogicDiagnostics,
};
use crate::pin::PinManager;
use crate::tls::{TlsConfig, TlsManager};

use super::Node;

/// IPFRS node configuration
#[derive(Debug, Clone)]
pub struct NodeConfig {
    /// Network configuration
    pub network: NetworkConfig,
    /// Storage configuration
    pub storage: BlockStoreConfig,
    /// Semantic router configuration
    pub semantic: RouterConfig,
    /// Enable semantic routing
    pub enable_semantic: bool,
    /// Enable TensorLogic integration
    pub enable_tensorlogic: bool,
    /// Authentication configuration (JWT secret)
    pub auth_jwt_secret: Option<String>,
    /// TLS configuration
    pub tls: Option<TlsConfig>,
    /// Per-session Bitswap / DHT fetch timeout in seconds.
    ///
    /// When `None`, the default of 30 seconds is used.  This value is passed
    /// to network provider queries when fetching blocks that are not found
    /// locally so that stalled DHT lookups do not block the caller indefinitely.
    pub fetch_timeout_secs: Option<u64>,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            network: NetworkConfig::default(),
            storage: BlockStoreConfig::default(),
            semantic: RouterConfig::default(),
            enable_semantic: true,
            enable_tensorlogic: true,
            auth_jwt_secret: None,
            tls: None,
            fetch_timeout_secs: None,
        }
    }
}

impl NodeConfig {
    /// Enable semantic search with custom configuration
    pub fn with_semantic(mut self, config: RouterConfig) -> Self {
        self.semantic = config;
        self.enable_semantic = true;
        self
    }

    /// Enable TensorLogic
    pub fn with_tensorlogic(mut self) -> Self {
        self.enable_tensorlogic = true;
        self
    }

    /// Enable authentication with JWT secret
    pub fn with_auth(mut self, jwt_secret: String) -> Self {
        self.auth_jwt_secret = Some(jwt_secret);
        self
    }

    /// Enable TLS with configuration
    pub fn with_tls(mut self, tls_config: TlsConfig) -> Self {
        self.tls = Some(tls_config);
        self
    }
}

/// Node status information
#[derive(Debug, Clone)]
pub struct NodeStatus {
    /// Whether the node is running
    pub running: bool,
    /// Whether network is enabled
    pub network_enabled: bool,
    /// Whether storage is enabled
    pub storage_enabled: bool,
    /// Whether semantic routing is enabled
    pub semantic_enabled: bool,
    /// Whether TensorLogic is enabled
    pub tensorlogic_enabled: bool,
}

impl Node {
    /// Create a new IPFRS node
    pub fn new(config: NodeConfig) -> Result<Self> {
        let metrics = ipfrs_interface::metrics::IpfrsMetrics::new()
            .map(Arc::new)
            .map_err(|e| {
                Error::Initialization(format!("Failed to create metrics registry: {}", e))
            })?;

        Ok(Self {
            config,
            network: None,
            storage: None,
            semantic: OnceCell::new(),
            tensorlogic: OnceCell::new(),
            auth_manager: None,
            tls_manager: None,
            pin_manager: Arc::new(PinManager::new()),
            startup_time: None,
            metrics,
        })
    }

    /// Start the IPFRS node
    pub async fn start(&mut self) -> Result<()> {
        // Record startup time
        self.startup_time = Some(SystemTime::now());

        // Initialize storage — wrap SledBlockStore in a hot-block L1 cache.
        let sled_store = SledBlockStore::new(self.config.storage.clone())?;
        let cached_store = CachedBlockStore::with_default_config(sled_store);
        let storage_arc = Arc::new(cached_store);
        self.storage = Some(storage_arc.clone());

        // Note: Semantic router and TensorLogic are now lazily initialized on first use
        // This improves startup time and reduces memory usage when not needed

        // Initialize authentication if configured
        if let Some(ref jwt_secret) = self.config.auth_jwt_secret {
            let auth_manager = AuthManager::new(jwt_secret.clone());
            self.auth_manager = Some(Arc::new(auth_manager));
        }

        // Initialize TLS if configured
        if let Some(ref tls_config) = self.config.tls {
            let tls_manager = TlsManager::new(tls_config.clone())
                .map_err(|e| Error::Initialization(format!("TLS initialization failed: {}", e)))?;
            self.tls_manager = Some(Arc::new(tls_manager));
        }

        // Initialize network
        let mut network = NetworkNode::new(self.config.network.clone())?;
        network.start().await?;
        self.network = Some(network);

        // Try to restore HNSW index from a previous snapshot (best-effort).
        // After loading the full snapshot, any adjacent incremental delta is
        // applied automatically so nodes that restart frequently do not lose
        // recent embeddings.
        if self.config.enable_semantic {
            let index_path = self.config.storage.path.join("hnsw_index.snap");
            if index_path.exists() {
                tracing::info!(
                    path = %index_path.display(),
                    "Restoring HNSW index from snapshot (with incremental delta if available)"
                );
                // Use load_index_with_delta so that any incremental delta written
                // by the previous shutdown is applied on top of the full snapshot.
                match self.semantic() {
                    Ok(sem) => {
                        if let Err(e) = sem.load_index_with_delta(&index_path).await {
                            tracing::warn!(
                                error = %e,
                                "Failed to restore HNSW snapshot – starting with empty index"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "Semantic router unavailable during HNSW restore"
                        );
                    }
                }
            }
        }

        // Try to restore TensorLogic knowledge base from a previous snapshot (best-effort)
        if self.config.enable_tensorlogic {
            let tl_path = self.config.storage.path.join("tensorlogic.snap");
            if tl_path.exists() {
                tracing::info!(
                    path = %tl_path.display(),
                    "Restoring TensorLogic snapshot"
                );
                match self.tensorlogic() {
                    Ok(tensorlogic) => match tensorlogic.load_kb(&tl_path).await {
                        Ok(()) => {
                            tracing::info!("Restored TensorLogic knowledge base from snapshot")
                        }
                        Err(e) => tracing::warn!(
                            error = %e,
                            "Could not restore TensorLogic snapshot – starting with empty knowledge base"
                        ),
                    },
                    Err(e) => tracing::warn!(
                        error = %e,
                        "TensorLogic not available for snapshot restore"
                    ),
                }
            }
        }

        Ok(())
    }

    /// Stop the IPFRS node
    pub async fn stop(&mut self) -> Result<()> {
        if let Some(mut network) = self.network.take() {
            network.stop().await?;
        }

        // Persist HNSW index snapshot before shutting down (best-effort).
        // Uses smart incremental logic: writes only a delta file when fewer
        // than 10 % of entries are dirty, otherwise falls back to a full snapshot.
        if self.config.enable_semantic && self.semantic.get().is_some() {
            let index_path = self.config.storage.path.join("hnsw_index.snap");
            tracing::info!(
                path = %index_path.display(),
                "Saving HNSW index snapshot (smart incremental)"
            );
            let save_result = match self.semantic() {
                Ok(sem) => sem.save_index_smart(&index_path).await,
                Err(e) => Err(e),
            };
            match save_result {
                Ok(()) => {
                    tracing::info!(
                        path = %index_path.display(),
                        pin_id = %ipfrs_storage::snapshot_pin_id(&index_path),
                        "HNSW snapshot pinned (GC-protected)"
                    );
                    // Pin the snapshot's block CIDs via the Sled-backed registry so
                    // the OrphanGarbageCollector never deletes them.
                    if let Some(storage) = &self.storage {
                        if let Ok(all_cids) = storage.list_cids() {
                            match storage.inner().snapshot_pin_registry() {
                                Ok(snap_reg) => {
                                    for cid in &all_cids {
                                        if let Err(e) = snap_reg.pin(cid, "hnsw-snapshot") {
                                            tracing::warn!(
                                                error = %e,
                                                cid = %cid,
                                                "Failed to pin snapshot CID"
                                            );
                                        }
                                    }
                                    tracing::info!(
                                        count = all_cids.len(),
                                        "Pinned HNSW snapshot CIDs in Sled registry"
                                    );
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        error = %e,
                                        "Could not open SledSnapshotPinRegistry for HNSW snapshot CIDs"
                                    );
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "Failed to save HNSW snapshot – index will not be restored on next start"
                    );
                }
            }
        }

        // Persist TensorLogic knowledge base snapshot before shutting down (best-effort)
        if self.config.enable_tensorlogic && self.tensorlogic.get().is_some() {
            let tl_path = self.config.storage.path.join("tensorlogic.snap");
            tracing::info!(
                path = %tl_path.display(),
                "Saving TensorLogic snapshot"
            );
            match self.tensorlogic() {
                Ok(tensorlogic) => {
                    if let Err(e) = tensorlogic.save_kb(&tl_path).await {
                        tracing::warn!(
                            error = %e,
                            "Failed to save TensorLogic snapshot – knowledge base will not be restored on next start"
                        );
                    }
                }
                Err(e) => tracing::warn!(
                    error = %e,
                    "TensorLogic not available for snapshot save"
                ),
            }
        }

        // Flush storage before stopping
        if let Some(storage) = &self.storage {
            storage.flush().await?;
        }

        // Clear all components
        // Note: OnceCell fields (semantic, tensorlogic) will be dropped automatically
        self.storage = None;
        self.auth_manager = None;
        self.tls_manager = None;

        Ok(())
    }

    /// Pre-initialize lazy components for faster first access
    ///
    /// By default, semantic router and TensorLogic store are initialized
    /// lazily on first use. This method forces their initialization upfront,
    /// which can be useful for:
    /// - Warmup scenarios where you want predictable latency
    /// - Load testing where you want to measure steady-state performance
    /// - Detecting configuration errors early at startup
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Pre-initialize all components for faster first access
    /// node.warmup()?;
    ///
    /// // Now semantic and tensorlogic calls will be instant
    /// # Ok(())
    /// # }
    /// ```
    pub fn warmup(&self) -> Result<()> {
        // Pre-initialize semantic router if enabled
        if self.config.enable_semantic {
            let _ = self.semantic()?;
        }

        // Pre-initialize TensorLogic store if enabled
        if self.config.enable_tensorlogic {
            let _ = self.tensorlogic()?;
        }

        Ok(())
    }

    /// Get node status
    pub fn status(&self) -> NodeStatus {
        NodeStatus {
            running: self.network.is_some(),
            network_enabled: self.network.is_some(),
            storage_enabled: self.storage.is_some(),
            semantic_enabled: self.semantic.get().is_some(),
            tensorlogic_enabled: self.tensorlogic.get().is_some(),
        }
    }

    /// Get comprehensive node diagnostics
    ///
    /// Collects detailed diagnostic information about node health, resource usage,
    /// and performance. This is useful for monitoring, troubleshooting, and optimization.
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig, DiagnosticAnalyzer};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Get diagnostics
    /// let diagnostics = node.diagnostics().await?;
    ///
    /// // Analyze and get recommendations
    /// let recommendations = DiagnosticAnalyzer::analyze(&diagnostics);
    /// for rec in recommendations {
    ///     println!("{:?}: {}", rec.severity, rec.message);
    /// }
    ///
    /// // Or get a human-readable report
    /// let report = DiagnosticAnalyzer::health_report(&diagnostics);
    /// println!("{}", report);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn diagnostics(&self) -> Result<NodeDiagnostics> {
        use std::time::Duration;

        // Calculate uptime
        let uptime = if let Some(startup) = self.startup_time {
            SystemTime::now()
                .duration_since(startup)
                .unwrap_or(Duration::from_secs(0))
        } else {
            Duration::from_secs(0)
        };

        // Gather storage diagnostics
        let storage_diag = if self.storage.is_some() {
            match self.storage_stats() {
                Ok(stats) => StorageDiagnostics {
                    total_blocks: stats.num_blocks as u64,
                    total_bytes: 0,    // Not tracked by current stats
                    avg_block_size: 0, // Cannot calculate without total_bytes
                    storage_path: self.config.storage.path.to_string_lossy().to_string(),
                    health: HealthStatus::Healthy,
                },
                Err(_) => StorageDiagnostics {
                    total_blocks: 0,
                    total_bytes: 0,
                    avg_block_size: 0,
                    storage_path: self.config.storage.path.to_string_lossy().to_string(),
                    health: HealthStatus::Degraded,
                },
            }
        } else {
            StorageDiagnostics {
                total_blocks: 0,
                total_bytes: 0,
                avg_block_size: 0,
                storage_path: String::new(),
                health: HealthStatus::Unknown,
            }
        };

        // Gather semantic diagnostics
        let semantic_diag = if self.config.enable_semantic && self.semantic.get().is_some() {
            match self.semantic_stats() {
                Ok(stats) => Some(SemanticDiagnostics {
                    num_vectors: stats.num_vectors,
                    dimensions: stats.dimension,
                    health: HealthStatus::Healthy,
                    cache_hit_rate: if stats.cache_size > 0 {
                        Some(stats.cache_size as f64 / stats.cache_capacity as f64)
                    } else {
                        None
                    },
                }),
                Err(_) => Some(SemanticDiagnostics {
                    num_vectors: 0,
                    dimensions: 0,
                    health: HealthStatus::Degraded,
                    cache_hit_rate: None,
                }),
            }
        } else {
            None
        };

        // Gather TensorLogic diagnostics
        let tensorlogic_diag = if self.config.enable_tensorlogic && self.tensorlogic.get().is_some()
        {
            match self.tensorlogic_stats() {
                Ok(stats) => Some(TensorLogicDiagnostics {
                    num_facts: stats.num_facts,
                    num_rules: stats.num_rules,
                    health: HealthStatus::Healthy,
                    avg_inference_ms: self.tensorlogic().ok().and_then(|tl| tl.avg_inference_ms()),
                }),
                Err(_) => Some(TensorLogicDiagnostics {
                    num_facts: 0,
                    num_rules: 0,
                    health: HealthStatus::Degraded,
                    avg_inference_ms: None,
                }),
            }
        } else {
            None
        };

        // Gather network diagnostics
        let network_diag = self.network.as_ref().map(|network| {
            let stats = network.stats();
            NetworkDiagnostics {
                connected_peers: stats.connected_peers,
                health: if stats.connected_peers > 0 {
                    HealthStatus::Healthy
                } else {
                    HealthStatus::Degraded
                },
                bytes_sent: stats.bytes_sent,
                bytes_received: stats.bytes_received,
            }
        });

        // Gather resource usage — estimate from known in-memory structures.
        // HNSW index: nodes × (dim×4 + M×8) bytes
        // TensorLogic KB: rules×500 + facts×200 bytes
        let hnsw_bytes = self
            .semantic
            .get()
            .and_then(|sem| sem.estimated_memory_bytes().ok())
            .unwrap_or(0);
        let tl_bytes = self
            .tensorlogic
            .get()
            .map(|tl| tl.estimated_memory_bytes())
            .unwrap_or(0);
        let resources = ResourceUsage {
            memory_bytes: (hnsw_bytes + tl_bytes) as u64,
            cpu_percent: None,
        };

        Ok(NodeDiagnostics {
            timestamp: SystemTime::now(),
            uptime,
            storage: storage_diag,
            semantic: semantic_diag,
            tensorlogic: tensorlogic_diag,
            network: network_diag,
            resources,
        })
    }

    /// Check if semantic routing is enabled
    ///
    /// Returns true if the semantic router is configured to be enabled.
    /// Note: The router will be lazily initialized on first use.
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// if node.is_semantic_enabled() {
    ///     println!("Semantic search is available");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_semantic_enabled(&self) -> bool {
        self.config.enable_semantic
    }

    /// Check if TensorLogic is enabled
    ///
    /// Returns true if the TensorLogic store is configured to be enabled.
    /// Note: The store will be lazily initialized on first use.
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// if node.is_tensorlogic_enabled() {
    ///     println!("Logic programming is available");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_tensorlogic_enabled(&self) -> bool {
        self.config.enable_tensorlogic
    }

    /// Check if the node is running
    ///
    /// Returns true if the node has been started and the network component
    /// is active.
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    ///
    /// assert!(!node.is_running());
    ///
    /// node.start().await?;
    /// assert!(node.is_running());
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_running(&self) -> bool {
        self.network.is_some()
    }

    /// Check if semantic router has been initialized
    ///
    /// Returns true if the semantic router has been lazily initialized.
    /// This is different from `is_semantic_enabled()` which checks if it's configured.
    pub fn is_semantic_initialized(&self) -> bool {
        self.semantic.get().is_some()
    }

    /// Check if TensorLogic store has been initialized
    ///
    /// Returns true if the TensorLogic store has been lazily initialized.
    /// This is different from `is_tensorlogic_enabled()` which checks if it's configured.
    pub fn is_tensorlogic_initialized(&self) -> bool {
        self.tensorlogic.get().is_some()
    }

    /// Get the authentication manager if enabled
    pub fn auth_manager(&self) -> Result<Arc<AuthManager>> {
        self.auth_manager
            .clone()
            .ok_or_else(|| Error::Internal("Authentication not enabled".to_string()))
    }

    /// Check if authentication is enabled
    pub fn is_auth_enabled(&self) -> bool {
        self.auth_manager.is_some()
    }

    /// Get the TLS manager if enabled
    pub fn tls_manager(&self) -> Result<Arc<TlsManager>> {
        self.tls_manager
            .clone()
            .ok_or_else(|| Error::Internal("TLS not enabled".to_string()))
    }

    /// Check if TLS is enabled
    pub fn is_tls_enabled(&self) -> bool {
        self.tls_manager.is_some()
    }
}
