//! IPFRS Node - Unified node implementation

use bytes::Bytes;
use ipfrs_core::{Block, Cid, Error, Result};
use ipfrs_network::{NetworkConfig, NetworkNode};
use ipfrs_semantic::{DistanceMetric, QueryFilter, RouterConfig, SearchResult, SemanticRouter};
use ipfrs_storage::{BlockStoreConfig, BlockStoreTrait, SledBlockStore};
use ipfrs_tensorlogic::{Predicate, Proof, Rule, Substitution, TensorLogicStore, Term};
use once_cell::sync::OnceCell;
use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

use crate::auth::{AuthManager, AuthToken, Permission, Role, User};
use crate::diagnostics::{
    HealthStatus, NetworkDiagnostics, NodeDiagnostics, ResourceUsage, SemanticDiagnostics,
    StorageDiagnostics, TensorLogicDiagnostics,
};
use crate::fsck::{FilesystemChecker, FsckConfig};
use crate::gc::{GarbageCollector, GcConfig};
use crate::pin::{PinInfo, PinManager, PinType};
use crate::tls::{TlsConfig, TlsManager};

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

/// IPFRS unified node combining all layers
pub struct Node {
    config: NodeConfig,
    network: Option<NetworkNode>,
    storage: Option<Arc<SledBlockStore>>,
    semantic: OnceCell<Arc<SemanticRouter>>,
    tensorlogic: OnceCell<Arc<TensorLogicStore<SledBlockStore>>>,
    auth_manager: Option<Arc<AuthManager>>,
    tls_manager: Option<Arc<TlsManager>>,
    pin_manager: Arc<PinManager>,
    startup_time: Option<SystemTime>,
}

impl Node {
    /// Create a new IPFRS node
    pub fn new(config: NodeConfig) -> Result<Self> {
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
        })
    }

    /// Start the IPFRS node
    pub async fn start(&mut self) -> Result<()> {
        // Record startup time
        self.startup_time = Some(SystemTime::now());

        // Initialize storage
        let storage = SledBlockStore::new(self.config.storage.clone())?;
        let storage_arc = Arc::new(storage);
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

        Ok(())
    }

    /// Stop the IPFRS node
    pub async fn stop(&mut self) -> Result<()> {
        if let Some(mut network) = self.network.take() {
            network.stop().await?;
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
                    avg_inference_ms: None, // TODO: Track inference times
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

        // Gather resource usage (simple approximation)
        let resources = ResourceUsage {
            memory_bytes: 0, // TODO: Add actual memory tracking
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

    // ==================================================================
    // Authentication & Security
    // ==================================================================

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

    /// Create a new user (requires auth enabled)
    pub fn create_user(
        &self,
        username: String,
        email: Option<String>,
        roles: std::collections::HashSet<Role>,
    ) -> Result<User> {
        let auth = self.auth_manager()?;
        auth.create_user(username, email, roles)
            .map_err(|e| Error::Internal(format!("Failed to create user: {}", e)))
    }

    /// Verify an authentication token
    pub fn verify_token(&self, token: &str) -> Result<AuthToken> {
        let auth = self.auth_manager()?;
        auth.verify_token(token)
            .map_err(|e| Error::Internal(format!("Token verification failed: {}", e)))
    }

    /// Check if a token has a specific permission
    pub fn check_permission(&self, token: &AuthToken, permission: Permission) -> Result<()> {
        let auth = self.auth_manager()?;
        auth.check_permission(token, permission)
            .map_err(|e| Error::Internal(format!("Permission check failed: {}", e)))
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

    // ==================================================================
    // File Operations
    // ==================================================================

    /// Add a file from the filesystem
    pub async fn add_file(&self, path: impl AsRef<Path>) -> Result<Cid> {
        let storage = self.storage()?;

        let data = tokio::fs::read(path.as_ref()).await?;

        let block = Block::new(Bytes::from(data))?;
        let cid = *block.cid();

        storage.put(&block).await?;

        Ok(cid)
    }

    /// Add bytes directly to storage
    pub async fn add_bytes(&self, data: impl Into<Bytes>) -> Result<Cid> {
        let storage = self.storage()?;

        let block = Block::new(data.into())?;
        let cid = *block.cid();

        storage.put(&block).await?;

        Ok(cid)
    }

    /// Add content from an async reader
    ///
    /// Reads all data from the provided reader, stores it as a block, and returns the CID.
    /// This is useful for streaming data from files, network streams, or other async sources.
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    /// use tokio::fs::File;
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// let file = File::open("data.bin").await?;
    /// let cid = node.add_reader(file).await?;
    /// println!("Stored with CID: {}", cid);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn add_reader<R>(&self, mut reader: R) -> Result<Cid>
    where
        R: tokio::io::AsyncRead + Unpin,
    {
        use tokio::io::AsyncReadExt;

        let storage = self.storage()?;

        // Read all data into a buffer
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).await?;

        let block = Block::new(Bytes::from(buffer))?;
        let cid = *block.cid();

        storage.put(&block).await?;

        Ok(cid)
    }

    /// Get content by CID
    pub async fn get(&self, cid: &Cid) -> Result<Option<Bytes>> {
        let storage = self.storage()?;

        match storage.get(cid).await? {
            Some(block) => Ok(Some(block.data().clone())),
            None => Ok(None),
        }
    }

    /// Get a byte range from content
    ///
    /// Retrieves a specific byte range from a block, similar to HTTP 206 Partial Content.
    /// This is useful for streaming large files or implementing range requests.
    ///
    /// # Parameters
    /// - `cid`: The content identifier
    /// - `offset`: Starting byte position (0-indexed)
    /// - `length`: Number of bytes to read (None for all remaining bytes)
    ///
    /// # Returns
    /// - `Ok(Some(bytes))` - The requested byte range
    /// - `Ok(None)` - Block not found
    /// - `Err(_)` - Invalid range or other error
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let cid = ipfrs_core::Cid::default();
    /// // Get bytes 100-199 (100 bytes starting at offset 100)
    /// if let Some(data) = node.get_range(&cid, 100, Some(100)).await? {
    ///     println!("Retrieved {} bytes", data.len());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_range(
        &self,
        cid: &Cid,
        offset: usize,
        length: Option<usize>,
    ) -> Result<Option<Bytes>> {
        let storage = self.storage()?;

        match storage.get(cid).await? {
            Some(block) => {
                let data = block.data();
                let total_len = data.len();

                // Validate offset
                if offset >= total_len {
                    return Err(Error::InvalidData(format!(
                        "Offset {} is beyond block size {}",
                        offset, total_len
                    )));
                }

                // Calculate end position
                let end = match length {
                    Some(len) => std::cmp::min(offset + len, total_len),
                    None => total_len,
                };

                // Extract range
                let range_data = data.slice(offset..end);
                Ok(Some(range_data))
            }
            None => Ok(None),
        }
    }

    /// Get content and write to file
    pub async fn get_to_file(&self, cid: &Cid, path: impl AsRef<Path>) -> Result<()> {
        let storage = self.storage()?;

        match storage.get(cid).await? {
            Some(block) => {
                tokio::fs::write(path.as_ref(), block.data()).await?;
                Ok(())
            }
            None => Err(Error::NotFound(format!("Block not found: {}", cid))),
        }
    }

    /// Add a directory recursively
    ///
    /// Traverses a directory tree, stores all files as blocks, and creates
    /// a directory structure using IPLD. Returns the root CID.
    ///
    /// # Directory Structure
    /// Directories are stored as IPLD maps where:
    /// - Keys are file/directory names
    /// - Values are either:
    ///   - Links to file blocks (for files)
    ///   - Nested maps (for subdirectories)
    ///
    /// # Example
    /// ```rust,ignore
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// let root_cid = node.add_directory("/path/to/directory").await?;
    /// println!("Directory stored with root CID: {}", root_cid);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn add_directory(&self, dir_path: impl AsRef<Path>) -> Result<Cid> {
        use std::collections::BTreeMap;

        let dir_path = dir_path.as_ref();

        if !dir_path.is_dir() {
            return Err(Error::InvalidData(format!(
                "Path is not a directory: {}",
                dir_path.display()
            )));
        }

        let mut entries = BTreeMap::new();
        let mut read_dir = tokio::fs::read_dir(dir_path).await?;

        while let Some(entry) = read_dir.next_entry().await? {
            let file_name = entry
                .file_name()
                .to_str()
                .ok_or_else(|| {
                    Error::InvalidData(format!("Invalid filename: {:?}", entry.file_name()))
                })?
                .to_string();

            let file_path = entry.path();
            let metadata = entry.metadata().await?;

            if metadata.is_file() {
                // Store file as block and create link
                let cid = self.add_file(&file_path).await?;
                entries.insert(file_name, ipfrs_core::Ipld::link(cid));
            } else if metadata.is_dir() {
                // Recursively add subdirectory
                let subdir_cid = self.add_directory(&file_path).await?;
                entries.insert(file_name, ipfrs_core::Ipld::link(subdir_cid));
            }
            // Skip other file types (symlinks, etc.)
        }

        // Store directory as IPLD map
        let dir_ipld = ipfrs_core::Ipld::Map(entries);
        self.dag_put(dir_ipld).await
    }

    /// Get a directory and write all files to the filesystem
    ///
    /// Retrieves a directory DAG from storage and recreates the directory
    /// structure on the filesystem.
    ///
    /// # Example
    /// ```rust,ignore
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let dir_cid = ipfrs_core::Cid::default();
    /// node.get_directory(&dir_cid, "/path/to/output").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_directory(&self, cid: &Cid, output_path: impl AsRef<Path>) -> Result<()> {
        let output_path = output_path.as_ref();

        // Create output directory
        tokio::fs::create_dir_all(output_path).await?;

        // Get directory IPLD
        let dir_ipld = self
            .dag_get(cid)
            .await?
            .ok_or_else(|| Error::NotFound(format!("Directory not found: {}", cid)))?;

        match dir_ipld {
            ipfrs_core::Ipld::Map(entries) => {
                for (name, value) in entries {
                    let entry_path = output_path.join(&name);

                    match value {
                        ipfrs_core::Ipld::Link(link) => {
                            // Try to determine if it's a file or directory
                            // by checking if it's a map
                            if let Some(ipld) = self.dag_get(&link.0).await? {
                                match ipld {
                                    ipfrs_core::Ipld::Map(_) => {
                                        // It's a directory
                                        self.get_directory(&link.0, &entry_path).await?;
                                    }
                                    _ => {
                                        // It's a file - get the raw bytes
                                        if let Some(bytes) = self.get(&link.0).await? {
                                            tokio::fs::write(&entry_path, bytes).await?;
                                        }
                                    }
                                }
                            }
                        }
                        _ => {
                            // Unexpected value type in directory
                            return Err(Error::InvalidData(format!(
                                "Unexpected value type in directory for entry: {}",
                                name
                            )));
                        }
                    }
                }
                Ok(())
            }
            _ => Err(Error::InvalidData(format!(
                "CID does not point to a directory structure: {}",
                cid
            ))),
        }
    }

    // ==================================================================
    // DAG Operations
    // ==================================================================

    /// Store an IPLD DAG node
    ///
    /// Serializes the IPLD data structure, stores it as a block, and returns the CID.
    /// This is useful for storing structured data with links to other blocks.
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    /// use ipfrs_core::Ipld;
    /// use std::collections::BTreeMap;
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert("name".to_string(), Ipld::String("Alice".to_string()));
    /// map.insert("age".to_string(), Ipld::Integer(30));
    ///
    /// let cid = node.dag_put(Ipld::Map(map)).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn dag_put(&self, data: ipfrs_core::Ipld) -> Result<Cid> {
        let storage = self.storage()?;

        // Serialize IPLD to DAG-CBOR format
        let bytes = data.to_dag_cbor()?;

        // Create block and store
        let block = Block::new(Bytes::from(bytes))?;
        let cid = *block.cid();

        storage.put(&block).await?;

        Ok(cid)
    }

    /// Retrieve an IPLD DAG node
    ///
    /// Fetches a block by CID and deserializes it as IPLD.
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let cid = ipfrs_core::Cid::default();
    /// if let Some(data) = node.dag_get(&cid).await? {
    ///     println!("Retrieved DAG node: {:?}", data);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn dag_get(&self, cid: &Cid) -> Result<Option<ipfrs_core::Ipld>> {
        let storage = self.storage()?;

        match storage.get(cid).await? {
            Some(block) => {
                let ipld = ipfrs_core::Ipld::from_dag_cbor(block.data())?;
                Ok(Some(ipld))
            }
            None => Ok(None),
        }
    }

    /// Resolve an IPLD path
    ///
    /// Navigates through IPLD structures following a path like "/key1/key2/0".
    /// Returns the CID if the path leads to a link.
    ///
    /// # Path Format
    /// - Map keys: "/key"
    /// - List indices: "/0", "/1", etc.
    /// - Nested: "/map_key/0/nested_key"
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let root_cid = ipfrs_core::Cid::default();
    /// // Resolve path: /users/0/profile
    /// if let Some(cid) = node.dag_resolve(&root_cid, "/users/0/profile").await? {
    ///     println!("Resolved to CID: {}", cid);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn dag_resolve(&self, root: &Cid, path: &str) -> Result<Option<Cid>> {
        let mut current_cid = *root;
        let parts: Vec<&str> = path
            .trim_start_matches('/')
            .split('/')
            .filter(|s| !s.is_empty())
            .collect();

        if parts.is_empty() {
            return Ok(Some(current_cid));
        }

        for part in parts {
            // Get the current node
            let ipld = match self.dag_get(&current_cid).await? {
                Some(ipld) => ipld,
                None => return Ok(None),
            };

            // Navigate to the next node
            match ipld {
                ipfrs_core::Ipld::Map(map) => {
                    match map.get(part) {
                        Some(ipfrs_core::Ipld::Link(link)) => {
                            current_cid = link.0;
                        }
                        Some(_value) => {
                            // For non-link values, we can't resolve further
                            // But we could return the value itself in a future enhancement
                            return Err(Error::InvalidData(format!(
                                "Path leads to non-link value at '{}'",
                                part
                            )));
                        }
                        None => {
                            return Ok(None);
                        }
                    }
                }
                ipfrs_core::Ipld::List(list) => {
                    let index: usize = part.parse().map_err(|_| {
                        Error::InvalidData(format!("Invalid list index: '{}'", part))
                    })?;

                    match list.get(index) {
                        Some(ipfrs_core::Ipld::Link(link)) => {
                            current_cid = link.0;
                        }
                        Some(_) => {
                            return Err(Error::InvalidData(format!(
                                "Path leads to non-link value at index {}",
                                index
                            )));
                        }
                        None => {
                            return Ok(None);
                        }
                    }
                }
                _ => {
                    return Err(Error::InvalidData(format!(
                        "Cannot navigate through non-map/non-list value at '{}'",
                        part
                    )));
                }
            }
        }

        Ok(Some(current_cid))
    }

    /// Traverse a DAG and collect all reachable CIDs
    ///
    /// Performs a breadth-first traversal starting from the root CID,
    /// following all links and collecting all reachable CIDs.
    ///
    /// # Parameters
    /// - `root`: The starting CID
    /// - `max_depth`: Maximum depth to traverse (None for unlimited)
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let root_cid = ipfrs_core::Cid::default();
    /// // Traverse up to depth 10
    /// let cids = node.dag_traverse(&root_cid, Some(10)).await?;
    /// println!("Found {} reachable blocks", cids.len());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn dag_traverse(&self, root: &Cid, max_depth: Option<usize>) -> Result<Vec<Cid>> {
        use std::collections::{HashSet, VecDeque};

        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut queue = VecDeque::new();

        queue.push_back((*root, 0usize));
        visited.insert(*root);

        while let Some((cid, depth)) = queue.pop_front() {
            // Check depth limit
            if let Some(max) = max_depth {
                if depth >= max {
                    continue;
                }
            }

            result.push(cid);

            // Get the node
            if let Some(ipld) = self.dag_get(&cid).await? {
                // Extract all links
                for link_cid in ipld.links() {
                    if visited.insert(link_cid) {
                        queue.push_back((link_cid, depth + 1));
                    }
                }
            }
        }

        Ok(result)
    }

    // ==================================================================
    // Block Operations
    // ==================================================================

    /// Store a raw block
    pub async fn put_block(&self, block: &Block) -> Result<()> {
        let storage = self.storage()?;
        storage.put(block).await
    }

    /// Store multiple blocks atomically
    pub async fn put_blocks(&self, blocks: &[Block]) -> Result<()> {
        let storage = self.storage()?;
        storage.put_many(blocks).await
    }

    /// Retrieve a block by CID
    pub async fn get_block(&self, cid: &Cid) -> Result<Option<Block>> {
        let storage = self.storage()?;
        storage.get(cid).await
    }

    /// Retrieve multiple blocks
    pub async fn get_blocks(&self, cids: &[Cid]) -> Result<Vec<Option<Block>>> {
        let storage = self.storage()?;
        storage.get_many(cids).await
    }

    /// Check if a block exists
    pub async fn has_block(&self, cid: &Cid) -> Result<bool> {
        let storage = self.storage()?;
        storage.has(cid).await
    }

    /// Check if multiple blocks exist
    pub async fn has_blocks(&self, cids: &[Cid]) -> Result<Vec<bool>> {
        let storage = self.storage()?;
        storage.has_many(cids).await
    }

    /// Delete a block
    pub async fn delete_block(&self, cid: &Cid) -> Result<()> {
        let storage = self.storage()?;
        storage.delete(cid).await
    }

    /// Delete multiple blocks
    pub async fn delete_blocks(&self, cids: &[Cid]) -> Result<()> {
        let storage = self.storage()?;
        storage.delete_many(cids).await
    }

    /// Get detailed statistics about a block
    ///
    /// Returns comprehensive information about a block including its size,
    /// CID details, and storage metadata.
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let cid = ipfrs_core::Cid::default();
    /// if let Some(stat) = node.block_stat(&cid).await? {
    ///     println!("Block size: {} bytes", stat.size);
    ///     println!("CID: {}", stat.cid);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn block_stat(&self, cid: &Cid) -> Result<Option<BlockStat>> {
        let storage = self.storage()?;

        match storage.get(cid).await? {
            Some(block) => Ok(Some(BlockStat {
                cid: *cid,
                size: block.data().len(),
            })),
            None => Ok(None),
        }
    }

    /// Remove a block from storage
    ///
    /// Removes a block if it's safe to do so. This method checks pinning status
    /// and refuses to remove pinned blocks to prevent accidental data loss.
    ///
    /// # Safety
    /// This operation is irreversible. The block will be permanently deleted
    /// from storage. Pinned blocks are protected and cannot be removed until unpinned.
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let cid = ipfrs_core::Cid::default();
    /// node.block_rm(&cid).await?;
    /// println!("Block removed");
    /// # Ok(())
    /// # }
    /// ```
    pub async fn block_rm(&self, cid: &Cid) -> Result<()> {
        // Check if block is pinned
        if self.pin_manager.is_pinned(cid) {
            return Err(Error::InvalidInput(format!(
                "Cannot remove pinned block: {}. Unpin it first.",
                cid
            )));
        }
        self.delete_block(cid).await
    }

    /// List all CIDs in storage
    pub fn list_blocks(&self) -> Result<Vec<Cid>> {
        let storage = self.storage()?;
        storage.list_cids()
    }

    // ==================================================================
    // Statistics & Management
    // ==================================================================

    /// Get storage statistics
    pub fn storage_stats(&self) -> Result<StorageStats> {
        let storage = self.storage()?;

        Ok(StorageStats {
            num_blocks: storage.len(),
            is_empty: storage.is_empty(),
        })
    }

    /// Flush pending writes to disk
    pub async fn flush(&self) -> Result<()> {
        let storage = self.storage()?;
        storage.flush().await
    }

    /// Get semantic router statistics
    ///
    /// Returns comprehensive statistics about the semantic index including
    /// vector count, dimension, distance metric, and cache performance.
    ///
    /// # Returns
    /// Statistics about the semantic router
    ///
    /// # Errors
    /// Returns error if semantic routing is not enabled
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// let stats = node.semantic_stats()?;
    /// println!("Indexed vectors: {}", stats.num_vectors);
    /// println!("Vector dimension: {}", stats.dimension);
    /// println!("Cache size: {}/{}", stats.cache_size, stats.cache_capacity);
    /// # Ok(())
    /// # }
    /// ```
    pub fn semantic_stats(&self) -> Result<SemanticStats> {
        let semantic = self.semantic()?;

        let router_stats = semantic.stats();
        let cache_stats = semantic.cache_stats();

        Ok(SemanticStats {
            num_vectors: router_stats.num_vectors,
            dimension: router_stats.dimension,
            metric: router_stats.metric,
            cache_size: cache_stats.size,
            cache_capacity: cache_stats.capacity,
        })
    }

    /// Get TensorLogic statistics
    ///
    /// Returns information about the TensorLogic store including counts of
    /// stored terms, predicates, and rules.
    ///
    /// # Returns
    /// Statistics about TensorLogic storage
    ///
    /// # Errors
    /// Returns error if TensorLogic is not enabled
    ///
    /// # Note
    /// Currently returns basic statistics. Future versions will include
    /// knowledge base statistics, inference metrics, and proof counts.
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// let stats = node.tensorlogic_stats()?;
    /// println!("TensorLogic enabled: {}", stats.enabled);
    /// # Ok(())
    /// # }
    /// ```
    pub fn tensorlogic_stats(&self) -> Result<TensorLogicStats> {
        let tensorlogic = self.tensorlogic()?;
        let kb_stats = tensorlogic.kb_stats();

        Ok(TensorLogicStats {
            enabled: true,
            num_facts: kb_stats.num_facts,
            num_rules: kb_stats.num_rules,
        })
    }

    // ==================================================================
    // Semantic Operations
    // ==================================================================

    /// Index content with its semantic embedding
    ///
    /// Adds content to the semantic index for similarity search. The embedding
    /// should be a vector representation of the content (e.g., from a sentence
    /// transformer model).
    ///
    /// # Arguments
    /// * `cid` - Content identifier to index
    /// * `embedding` - Vector embedding of the content
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let cid = ipfrs_core::Cid::default();
    /// // Index content with 768-dimensional embedding (e.g., from BERT)
    /// let embedding = vec![0.5; 768];
    /// node.index_content(&cid, &embedding).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn index_content(&self, cid: &Cid, embedding: &[f32]) -> Result<()> {
        let semantic = self.semantic()?;
        semantic.add(cid, embedding)
    }

    /// Search for similar content by semantic similarity
    ///
    /// Performs k-nearest neighbor search over indexed content using vector
    /// similarity. Returns the top k most similar items.
    ///
    /// # Arguments
    /// * `query_embedding` - Query vector to search for
    /// * `k` - Number of results to return
    ///
    /// # Returns
    /// Vector of search results ordered by similarity (highest first)
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Search for top 10 similar documents
    /// let query_embedding = vec![0.3; 768];
    /// let results = node.search_similar(&query_embedding, 10).await?;
    ///
    /// for result in results {
    ///     println!("CID: {}, Score: {}", result.cid, result.score);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search_similar(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let semantic = self.semantic()?;
        semantic.query(query_embedding, k).await
    }

    /// Search with advanced filtering options
    ///
    /// Performs semantic search with additional filters like minimum score
    /// threshold, CID prefix matching, and result limits.
    ///
    /// # Arguments
    /// * `query_embedding` - Query vector to search for
    /// * `k` - Number of results to return
    /// * `filter` - Query filter options
    ///
    /// # Returns
    /// Vector of filtered search results ordered by similarity
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig, QueryFilter};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Search with filters
    /// let query_embedding = vec![0.3; 768];
    /// let filter = QueryFilter {
    ///     min_score: Some(0.8),  // Only results with score >= 0.8
    ///     max_score: None,        // No max score filter
    ///     max_results: Some(5),   // Limit to 5 results
    ///     cid_prefix: None,       // No CID filtering
    /// };
    ///
    /// let results = node.search_hybrid(&query_embedding, 20, filter).await?;
    ///
    /// for result in results {
    ///     println!("High-confidence match: {} ({})", result.cid, result.score);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search_hybrid(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: QueryFilter,
    ) -> Result<Vec<SearchResult>> {
        let semantic = self.semantic()?;
        semantic.query_with_filter(query_embedding, k, filter).await
    }

    // ==================================================================
    // TensorLogic Operations
    // ==================================================================

    /// Store a logical term
    ///
    /// Serializes and stores a TensorLogic term as a content-addressed block.
    /// Terms can be constants, variables, functions, or references to other CIDs.
    ///
    /// # Arguments
    /// * `term` - The logical term to store
    ///
    /// # Returns
    /// CID of the stored term
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig, Term, Constant};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Store a constant term
    /// let term = Term::Const(Constant::String("Alice".to_string()));
    /// let cid = node.put_term(&term).await?;
    /// println!("Stored term with CID: {}", cid);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn put_term(&self, term: &Term) -> Result<Cid> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.store_term(term).await
    }

    /// Retrieve a logical term by CID
    ///
    /// Fetches and deserializes a TensorLogic term from storage.
    ///
    /// # Arguments
    /// * `cid` - Content identifier of the term
    ///
    /// # Returns
    /// The term if found, None otherwise
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let cid = ipfrs_core::Cid::default();
    /// if let Some(term) = node.get_term(&cid).await? {
    ///     println!("Retrieved term: {}", term);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_term(&self, cid: &Cid) -> Result<Option<Term>> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.get_term(cid).await
    }

    /// Store a logical predicate
    ///
    /// Stores a predicate (named relation with arguments) as a content-addressed block.
    ///
    /// # Arguments
    /// * `predicate` - The predicate to store
    ///
    /// # Returns
    /// CID of the stored predicate
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig, Predicate, Term, Constant};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Store a predicate: parent("Alice", "Bob")
    /// let predicate = Predicate::new(
    ///     "parent".to_string(),
    ///     vec![
    ///         Term::Const(Constant::String("Alice".to_string())),
    ///         Term::Const(Constant::String("Bob".to_string())),
    ///     ],
    /// );
    ///
    /// let cid = node.store_predicate(&predicate).await?;
    /// println!("Stored predicate with CID: {}", cid);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn store_predicate(&self, predicate: &Predicate) -> Result<Cid> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.store_predicate(predicate).await
    }

    /// Retrieve a logical predicate by CID
    ///
    /// Fetches and deserializes a predicate from storage.
    ///
    /// # Arguments
    /// * `cid` - Content identifier of the predicate
    ///
    /// # Returns
    /// The predicate if found, None otherwise
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let cid = ipfrs_core::Cid::default();
    /// if let Some(predicate) = node.get_predicate(&cid).await? {
    ///     println!("Retrieved predicate: {}", predicate);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_predicate(&self, cid: &Cid) -> Result<Option<Predicate>> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.get_predicate(cid).await
    }

    /// Store a logical rule
    ///
    /// Stores a Horn clause (head :- body) as a content-addressed block.
    ///
    /// # Arguments
    /// * `rule` - The rule to store
    ///
    /// # Returns
    /// CID of the stored rule
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig, Rule, Predicate, Term, Constant};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Store a fact: parent("Alice", "Bob")
    /// let fact = Rule::fact(Predicate::new(
    ///     "parent".to_string(),
    ///     vec![
    ///         Term::Const(Constant::String("Alice".to_string())),
    ///         Term::Const(Constant::String("Bob".to_string())),
    ///     ],
    /// ));
    ///
    /// let cid = node.store_rule(&fact).await?;
    /// println!("Stored rule with CID: {}", cid);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn store_rule(&self, rule: &Rule) -> Result<Cid> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.store_rule(rule).await
    }

    /// Retrieve a logical rule by CID
    ///
    /// Fetches and deserializes a rule from storage.
    ///
    /// # Arguments
    /// * `cid` - Content identifier of the rule
    ///
    /// # Returns
    /// The rule if found, None otherwise
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let cid = ipfrs_core::Cid::default();
    /// if let Some(rule) = node.get_rule(&cid).await? {
    ///     println!("Retrieved rule: head={}, body_len={}", rule.head, rule.body.len());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_rule(&self, cid: &Cid) -> Result<Option<Rule>> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.get_rule(cid).await
    }

    /// Add a fact to the knowledge base
    ///
    /// Adds a logical fact (predicate with no body) to the in-memory knowledge base.
    /// Facts are used during inference queries.
    ///
    /// # Arguments
    /// * `fact` - The predicate to add as a fact
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig, Predicate, Term, Constant};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Add fact: parent("Alice", "Bob")
    /// let fact = Predicate::new(
    ///     "parent".to_string(),
    ///     vec![
    ///         Term::Const(Constant::String("Alice".to_string())),
    ///         Term::Const(Constant::String("Bob".to_string())),
    ///     ],
    /// );
    ///
    /// node.add_fact(fact)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_fact(&self, fact: Predicate) -> Result<()> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.add_fact(fact)
    }

    /// Add a rule to the knowledge base
    ///
    /// Adds a logical rule (Horn clause) to the in-memory knowledge base.
    /// Rules are used during inference queries.
    ///
    /// # Arguments
    /// * `rule` - The rule to add
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig, Rule, Predicate, Term};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Add rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z)
    /// let head = Predicate::new("grandparent".to_string(), vec![
    ///     Term::Var("X".to_string()),
    ///     Term::Var("Z".to_string()),
    /// ]);
    /// let body = vec![
    ///     Predicate::new("parent".to_string(), vec![
    ///         Term::Var("X".to_string()),
    ///         Term::Var("Y".to_string()),
    ///     ]),
    ///     Predicate::new("parent".to_string(), vec![
    ///         Term::Var("Y".to_string()),
    ///         Term::Var("Z".to_string()),
    ///     ]),
    /// ];
    ///
    /// node.add_rule(Rule::new(head, body))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_rule(&self, rule: Rule) -> Result<()> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.add_rule(rule)
    }

    /// Run inference query
    ///
    /// Executes a logical query using backward chaining inference.
    /// Returns all variable substitutions that satisfy the goal.
    ///
    /// # Arguments
    /// * `goal` - The query predicate to prove
    ///
    /// # Returns
    /// Vector of variable substitutions (bindings) that satisfy the goal
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig, Predicate, Term, Constant};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Add some facts
    /// node.add_fact(Predicate::new("parent".to_string(), vec![
    ///     Term::Const(Constant::String("Alice".to_string())),
    ///     Term::Const(Constant::String("Bob".to_string())),
    /// ]))?;
    ///
    /// // Query: parent("Alice", X)?
    /// let goal = Predicate::new("parent".to_string(), vec![
    ///     Term::Const(Constant::String("Alice".to_string())),
    ///     Term::Var("X".to_string()),
    /// ]);
    ///
    /// let solutions = node.infer(&goal)?;
    /// for solution in solutions {
    ///     println!("Solution: {:?}", solution);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn infer(&self, goal: &Predicate) -> Result<Vec<Substitution>> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.infer(goal)
    }

    /// Generate proof tree
    ///
    /// Constructs a formal proof for a given goal predicate using backward chaining.
    /// Returns the proof if one can be found, None otherwise.
    ///
    /// # Arguments
    /// * `goal` - The goal to prove
    ///
    /// # Returns
    /// Proof object if goal can be proven, None otherwise
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig, Predicate, Term, Constant};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Add some facts
    /// node.add_fact(Predicate::new("parent".to_string(), vec![
    ///     Term::Const(Constant::String("Alice".to_string())),
    ///     Term::Const(Constant::String("Bob".to_string())),
    /// ]))?;
    ///
    /// // Generate proof
    /// let goal = Predicate::new("parent".to_string(), vec![
    ///     Term::Const(Constant::String("Alice".to_string())),
    ///     Term::Const(Constant::String("Bob".to_string())),
    /// ]);
    ///
    /// if let Some(proof) = node.prove(&goal)? {
    ///     println!("Proof found: {:?}", proof);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn prove(&self, goal: &Predicate) -> Result<Option<Proof>> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.prove(goal)
    }

    /// Store a proof and return its CID
    ///
    /// Serializes and stores a proof tree as a content-addressed block.
    ///
    /// # Arguments
    /// * `proof` - The proof to store
    ///
    /// # Returns
    /// CID of the stored proof
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig, Predicate, Term, Constant};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Add fact and generate proof
    /// node.add_fact(Predicate::new("parent".to_string(), vec![
    ///     Term::Const(Constant::String("Alice".to_string())),
    ///     Term::Const(Constant::String("Bob".to_string())),
    /// ]))?;
    ///
    /// let goal = Predicate::new("parent".to_string(), vec![
    ///     Term::Const(Constant::String("Alice".to_string())),
    ///     Term::Const(Constant::String("Bob".to_string())),
    /// ]);
    ///
    /// if let Some(proof) = node.prove(&goal)? {
    ///     let cid = node.store_proof(&proof).await?;
    ///     println!("Proof stored with CID: {}", cid);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn store_proof(&self, proof: &Proof) -> Result<Cid> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.store_proof(proof).await
    }

    /// Retrieve a proof by CID
    ///
    /// Fetches and deserializes a proof from storage.
    ///
    /// # Arguments
    /// * `cid` - Content identifier of the proof
    ///
    /// # Returns
    /// The proof if found, None otherwise
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let cid = ipfrs_core::Cid::default();
    /// if let Some(proof) = node.get_proof(&cid).await? {
    ///     println!("Retrieved proof for goal: {}", proof.goal);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_proof(&self, cid: &Cid) -> Result<Option<Proof>> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.get_proof(cid).await
    }

    /// Verify a proof against the current knowledge base
    ///
    /// Checks if a proof tree is valid by verifying that:
    /// - All facts exist in the knowledge base
    /// - All rules exist and are correctly applied
    /// - All subproofs are valid
    ///
    /// # Arguments
    /// * `proof` - The proof to verify
    ///
    /// # Returns
    /// `true` if the proof is valid, `false` otherwise
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig, Predicate, Term, Constant};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Add fact
    /// node.add_fact(Predicate::new("parent".to_string(), vec![
    ///     Term::Const(Constant::String("Alice".to_string())),
    ///     Term::Const(Constant::String("Bob".to_string())),
    /// ]))?;
    ///
    /// // Generate and verify proof
    /// let goal = Predicate::new("parent".to_string(), vec![
    ///     Term::Const(Constant::String("Alice".to_string())),
    ///     Term::Const(Constant::String("Bob".to_string())),
    /// ]);
    ///
    /// if let Some(proof) = node.prove(&goal)? {
    ///     let is_valid = node.verify_proof(&proof)?;
    ///     println!("Proof is valid: {}", is_valid);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn verify_proof(&self, proof: &Proof) -> Result<bool> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.verify_proof(proof)
    }

    /// Get knowledge base statistics
    ///
    /// Returns statistics about the in-memory knowledge base including
    /// counts of facts and rules.
    ///
    /// # Returns
    /// Knowledge base statistics
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// let stats = node.kb_stats()?;
    /// println!("Facts: {}, Rules: {}", stats.num_facts, stats.num_rules);
    /// # Ok(())
    /// # }
    /// ```
    pub fn kb_stats(&self) -> Result<ipfrs_tensorlogic::KnowledgeBaseStats> {
        let tensorlogic = self.tensorlogic()?;
        Ok(tensorlogic.kb_stats())
    }

    // ==================================================================
    // Persistence Operations
    // ==================================================================

    /// Save the semantic index to disk
    ///
    /// Persists the entire HNSW index including all vectors and CID mappings
    /// to a file for later loading.
    ///
    /// # Arguments
    /// * `path` - Path to save the index file
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Save the semantic index
    /// node.save_semantic_index("semantic.index").await?;
    /// println!("Semantic index saved");
    /// # Ok(())
    /// # }
    /// ```
    pub async fn save_semantic_index(&self, path: impl AsRef<Path>) -> Result<()> {
        let semantic = self.semantic()?;
        semantic.save_index(path).await
    }

    /// Load a semantic index from disk
    ///
    /// Loads a previously saved HNSW index from disk, replacing the current index.
    ///
    /// # Arguments
    /// * `path` - Path to the saved index file
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Load the semantic index
    /// node.load_semantic_index("semantic.index").await?;
    /// println!("Semantic index loaded");
    /// # Ok(())
    /// # }
    /// ```
    pub async fn load_semantic_index(&self, path: impl AsRef<Path>) -> Result<()> {
        let semantic = self.semantic()?;
        semantic.load_index(path).await
    }

    /// Save the knowledge base to disk
    ///
    /// Persists the entire knowledge base (facts and rules) to a file
    /// for later loading.
    ///
    /// # Arguments
    /// * `path` - Path to save the knowledge base file
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Save the knowledge base
    /// node.save_knowledge_base("knowledge.kb").await?;
    /// println!("Knowledge base saved");
    /// # Ok(())
    /// # }
    /// ```
    pub async fn save_knowledge_base(&self, path: impl AsRef<Path>) -> Result<()> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.save_kb(path).await
    }

    /// Load a knowledge base from disk
    ///
    /// Loads a previously saved knowledge base from disk, replacing the current KB.
    ///
    /// # Arguments
    /// * `path` - Path to the saved knowledge base file
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Load the knowledge base
    /// node.load_knowledge_base("knowledge.kb").await?;
    /// println!("Knowledge base loaded");
    /// # Ok(())
    /// # }
    /// ```
    pub async fn load_knowledge_base(&self, path: impl AsRef<Path>) -> Result<()> {
        let tensorlogic = self.tensorlogic()?;
        tensorlogic.load_kb(path).await
    }

    // ==================================================================
    // Network Operations
    // ==================================================================

    /// Get local peer ID
    pub fn peer_id(&self) -> Result<String> {
        let network = self.network()?;
        Ok(network.peer_id().to_string())
    }

    /// Get connected peers
    pub async fn peers(&self) -> Result<Vec<String>> {
        let network = self.network()?;
        let peers = network.connected_peers();
        Ok(peers.into_iter().map(|p| p.to_string()).collect())
    }

    /// Connect to a peer
    pub async fn connect(&mut self, addr: &str) -> Result<()> {
        let addr: ipfrs_network::libp2p::Multiaddr = addr
            .parse()
            .map_err(|e| Error::Network(format!("Invalid multiaddr: {}", e)))?;

        if let Some(network) = &mut self.network {
            network.connect(addr).await?;
        }
        Ok(())
    }

    /// Disconnect from a peer
    pub async fn disconnect(&mut self, peer_id: &str) -> Result<()> {
        use std::str::FromStr;
        let peer_id: ipfrs_network::libp2p::PeerId =
            ipfrs_network::libp2p::PeerId::from_str(peer_id)
                .map_err(|e| Error::Network(format!("Invalid peer ID: {}", e)))?;

        if let Some(network) = &mut self.network {
            network.disconnect(peer_id).await?;
        }
        Ok(())
    }

    /// Announce content to DHT (provide)
    pub async fn provide(&mut self, cid: &Cid) -> Result<()> {
        if let Some(network) = &mut self.network {
            network.provide(cid).await?;
        }
        Ok(())
    }

    /// Find providers for content in DHT
    pub async fn find_providers(&mut self, cid: &Cid) -> Result<()> {
        if let Some(network) = &mut self.network {
            network.find_providers(cid).await?;
        }
        Ok(())
    }

    /// Get network statistics
    pub fn network_stats(&self) -> Result<ipfrs_network::NetworkStats> {
        let network = self.network()?;
        Ok(network.stats())
    }

    /// Get bitswap statistics
    pub fn bitswap_stats(&self) -> Result<ipfrs_network::BitswapStats> {
        // Placeholder implementation - returns default stats
        Ok(ipfrs_network::BitswapStats::default())
    }

    /// Ping a peer (placeholder implementation)
    pub async fn ping(&mut self, _peer_id: &str) -> Result<()> {
        // Placeholder - actual ping would use libp2p ping protocol
        // For now, just verify we're connected
        let _network = self.network()?;
        Ok(())
    }

    /// Find a peer's addresses in the DHT (placeholder implementation)
    pub async fn find_peer(&mut self, peer_id: &str) -> Result<Vec<String>> {
        let _network = self.network()?;
        // Placeholder - actual implementation would query DHT
        Ok(vec![format!("/p2p/{}", peer_id)])
    }

    /// Get bootstrap peers
    pub fn bootstrap_peers(&self) -> Result<Vec<String>> {
        let stats = self.network_stats()?;
        Ok(stats.bootstrap_peers)
    }

    /// Add a bootstrap peer
    pub async fn add_bootstrap_peer(&mut self, addr: &str) -> Result<()> {
        let network = self.network_mut()?;
        let multiaddr: ipfrs_network::libp2p::Multiaddr = addr
            .parse()
            .map_err(|e| Error::Network(format!("Invalid multiaddr: {}", e)))?;
        network.connect(multiaddr).await?;
        Ok(())
    }

    /// Remove a bootstrap peer
    pub async fn remove_bootstrap_peer(&mut self, _addr: &str) -> Result<()> {
        // Placeholder - would update config and disconnect
        Ok(())
    }

    // ==================================================================
    // DAG Export/Import
    // ==================================================================

    /// Export a DAG to CAR (Content Addressable aRchive) format
    pub async fn dag_export(&self, root: &Cid, path: impl AsRef<Path>) -> Result<DagExportStats> {
        use ipfrs_storage::export_to_car;

        let storage = self.storage()?;
        let roots = vec![*root];

        let stats = export_to_car(storage.as_ref(), path.as_ref(), roots).await?;

        Ok(DagExportStats {
            blocks_exported: stats.blocks_written,
            bytes_exported: stats.bytes_written,
        })
    }

    /// Import blocks from a CAR file
    pub async fn dag_import(&self, path: impl AsRef<Path>) -> Result<DagImportStats> {
        use ipfrs_storage::import_from_car;

        let storage = self.storage()?;
        let stats = import_from_car(storage.as_ref(), path.as_ref()).await?;

        Ok(DagImportStats {
            blocks_imported: stats.blocks_read,
            bytes_imported: stats.bytes_read,
        })
    }

    // ==================================================================
    // Pin Management
    // ==================================================================

    /// Pin a block (prevent it from being garbage collected)
    pub async fn pin_add(&self, cid: &Cid, recursive: bool, name: Option<String>) -> Result<()> {
        // Verify the block exists
        if !self.storage()?.has(cid).await? {
            return Err(Error::NotFound(cid.to_string()));
        }

        let pin_type = if recursive {
            PinType::Recursive
        } else {
            PinType::Direct
        };

        self.pin_manager.pin(*cid, pin_type, name)?;

        // If recursive, traverse and mark all referenced blocks
        if recursive {
            self.mark_recursive_pins(cid).await?;
        }

        Ok(())
    }

    /// Mark all blocks referenced by a CID as indirectly pinned
    async fn mark_recursive_pins(&self, root: &Cid) -> Result<()> {
        let storage = self.storage()?;
        let mut to_visit = vec![*root];
        let mut visited = std::collections::HashSet::new();

        while let Some(cid) = to_visit.pop() {
            if !visited.insert(cid) {
                continue;
            }

            // Get the block and try to parse as IPLD
            if let Some(block) = storage.get(&cid).await? {
                if let Ok(ipld) = ipfrs_core::Ipld::from_dag_cbor(block.data()) {
                    // Add all links as indirect pins
                    for link_cid in ipld.links() {
                        self.pin_manager.add_indirect(*root, link_cid);
                        to_visit.push(link_cid);
                    }
                }
            }
        }

        Ok(())
    }

    /// Unpin a block
    pub async fn pin_rm(&self, cid: &Cid, recursive: bool) -> Result<()> {
        self.pin_manager.unpin(cid, recursive)
    }

    /// List all pinned blocks
    pub fn pin_ls(&self) -> Result<Vec<PinInfo>> {
        Ok(self.pin_manager.list())
    }

    /// Verify all pins are available
    pub async fn pin_verify(&self) -> Result<Vec<(Cid, bool)>> {
        let storage = self.storage()?;
        let pins = self.pin_manager.list();
        let mut results = Vec::new();

        for pin in pins {
            let exists = storage.has(&pin.cid).await?;
            results.push((pin.cid, exists));
        }

        Ok(results)
    }

    /// Save pin index to disk
    pub async fn pin_save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.pin_manager.save(path).await
    }

    /// Load pin index from disk
    pub async fn pin_load(&self, path: impl AsRef<Path>) -> Result<()> {
        self.pin_manager.load(path).await
    }

    // ==================================================================
    // Repository Management
    // ==================================================================

    /// Run garbage collection
    pub async fn repo_gc(&self, dry_run: bool) -> Result<GcResult> {
        let storage = self.storage()?;
        let gc = GarbageCollector::new(storage.clone(), self.pin_manager.clone());

        let config = GcConfig {
            dry_run,
            ..Default::default()
        };

        let stats = gc.collect(config).await?;

        Ok(GcResult {
            blocks_collected: stats.blocks_collected,
            bytes_freed: stats.bytes_freed,
            blocks_marked: stats.blocks_marked,
            blocks_scanned: stats.blocks_scanned,
            duration: stats.duration,
            cancelled: stats.cancelled,
        })
    }

    /// Verify repository integrity
    pub async fn repo_fsck(&self) -> Result<FsckResult> {
        let storage = self.storage()?;
        let fsck = FilesystemChecker::new(storage.clone());

        let config = FsckConfig::default();
        let result = fsck.check(config).await?;

        Ok(FsckResult {
            blocks_checked: result.blocks_checked,
            blocks_valid: result.blocks_valid,
            blocks_corrupt: result.blocks_corrupt,
            blocks_missing: result.blocks_missing,
        })
    }

    /// Run quick filesystem check (only verify CIDs match content)
    pub async fn repo_fsck_quick(&self) -> Result<FsckResult> {
        let storage = self.storage()?;
        let fsck = FilesystemChecker::new(storage.clone());

        let result = fsck.quick_check().await?;

        Ok(FsckResult {
            blocks_checked: result.blocks_checked,
            blocks_valid: result.blocks_valid,
            blocks_corrupt: result.blocks_corrupt,
            blocks_missing: result.blocks_missing,
        })
    }

    /// Get garbage collection statistics (without running GC)
    pub fn gc_stats(&self) -> Result<(usize, usize)> {
        let storage = self.storage()?;
        let gc = GarbageCollector::new(storage.clone(), self.pin_manager.clone());

        let unpinned_count = gc.count_unpinned()?;
        let pinned_count = self.pin_manager.count();

        Ok((pinned_count, unpinned_count))
    }

    // ==================================================================
    // Repository Analysis
    // ==================================================================

    /// Get comprehensive repository statistics
    pub async fn repo_stat(&self) -> Result<crate::repo::RepoStats> {
        let storage = self.storage()?;
        let analyzer = crate::repo::RepoAnalyzer::new(storage.clone(), self.pin_manager.clone());
        analyzer.analyze().await
    }

    /// Get block size distribution
    pub async fn block_distribution(&self) -> Result<crate::repo::BlockDistribution> {
        let storage = self.storage()?;
        let analyzer = crate::repo::RepoAnalyzer::new(storage.clone(), self.pin_manager.clone());
        analyzer.block_distribution().await
    }

    /// Find duplicate blocks (same content, different CIDs)
    pub async fn find_duplicates(&self) -> Result<Vec<Vec<Cid>>> {
        let storage = self.storage()?;
        let analyzer = crate::repo::RepoAnalyzer::new(storage.clone(), self.pin_manager.clone());
        analyzer.find_duplicates().await
    }

    /// Get largest blocks in the repository
    pub async fn largest_blocks(&self, limit: usize) -> Result<Vec<(Cid, u64)>> {
        let storage = self.storage()?;
        let analyzer = crate::repo::RepoAnalyzer::new(storage.clone(), self.pin_manager.clone());
        analyzer.largest_blocks(limit).await
    }

    /// Find orphaned blocks (not pinned and not referenced)
    pub async fn find_orphaned(&self) -> Result<Vec<Cid>> {
        let storage = self.storage()?;
        let analyzer = crate::repo::RepoAnalyzer::new(storage.clone(), self.pin_manager.clone());
        analyzer.find_orphaned_blocks().await
    }

    // ==================================================================
    // Internal Helpers
    // ==================================================================

    /// Get storage handle or return error if not started
    fn storage(&self) -> Result<&Arc<SledBlockStore>> {
        self.storage.as_ref().ok_or_else(|| {
            Error::Initialization("Node not started - call start() first".to_string())
        })
    }

    /// Get semantic router handle or return error if not enabled
    /// Lazily initializes the semantic router on first access
    fn semantic(&self) -> Result<&Arc<SemanticRouter>> {
        if !self.config.enable_semantic {
            return Err(Error::Initialization(
                "Semantic routing not enabled - set enable_semantic=true in config".to_string(),
            ));
        }

        self.semantic
            .get_or_try_init(|| SemanticRouter::new(self.config.semantic.clone()).map(Arc::new))
    }

    /// Get TensorLogic store handle or return error if not enabled
    /// Lazily initializes the TensorLogic store on first access
    fn tensorlogic(&self) -> Result<&Arc<TensorLogicStore<SledBlockStore>>> {
        if !self.config.enable_tensorlogic {
            return Err(Error::Initialization(
                "TensorLogic not enabled - set enable_tensorlogic=true in config".to_string(),
            ));
        }

        let storage = self.storage()?;

        self.tensorlogic
            .get_or_try_init(|| TensorLogicStore::new(storage.clone()).map(Arc::new))
    }

    /// Get network handle or return error if not started
    fn network(&self) -> Result<&NetworkNode> {
        self.network.as_ref().ok_or_else(|| {
            Error::Initialization("Node not started - call start() first".to_string())
        })
    }

    /// Get mutable network handle or return error if not started
    fn network_mut(&mut self) -> Result<&mut NetworkNode> {
        self.network.as_mut().ok_or_else(|| {
            Error::Initialization("Node not started - call start() first".to_string())
        })
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

/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStats {
    /// Number of blocks stored
    pub num_blocks: usize,
    /// Whether storage is empty
    pub is_empty: bool,
}

/// Block statistics
#[derive(Debug, Clone)]
pub struct BlockStat {
    /// Content identifier
    pub cid: Cid,
    /// Size in bytes
    pub size: usize,
}

/// Semantic router statistics
#[derive(Debug, Clone)]
pub struct SemanticStats {
    /// Number of indexed vectors
    pub num_vectors: usize,
    /// Vector dimension
    pub dimension: usize,
    /// Distance metric used
    pub metric: DistanceMetric,
    /// Current cache size
    pub cache_size: usize,
    /// Maximum cache capacity
    pub cache_capacity: usize,
}

/// TensorLogic statistics
#[derive(Debug, Clone)]
pub struct TensorLogicStats {
    /// Whether TensorLogic is enabled
    pub enabled: bool,
    /// Number of facts in knowledge base
    pub num_facts: usize,
    /// Number of rules in knowledge base
    pub num_rules: usize,
}

/// Result of garbage collection
#[derive(Debug, Clone)]
pub struct GcResult {
    /// Number of blocks collected (deleted)
    pub blocks_collected: u64,
    /// Bytes freed
    pub bytes_freed: u64,
    /// Number of blocks marked as reachable
    pub blocks_marked: u64,
    /// Number of blocks scanned
    pub blocks_scanned: u64,
    /// Duration of GC run
    pub duration: std::time::Duration,
    /// Whether GC was cancelled
    pub cancelled: bool,
}

/// Result of filesystem check
#[derive(Debug, Clone)]
pub struct FsckResult {
    /// Number of blocks checked
    pub blocks_checked: u64,
    /// Number of valid blocks
    pub blocks_valid: u64,
    /// List of corrupt blocks
    pub blocks_corrupt: Vec<Cid>,
    /// List of missing blocks
    pub blocks_missing: Vec<Cid>,
}

/// Result of DAG export operation
#[derive(Debug, Clone)]
pub struct DagExportStats {
    /// Number of blocks exported
    pub blocks_exported: u64,
    /// Total bytes exported
    pub bytes_exported: u64,
}

/// Result of DAG import operation
#[derive(Debug, Clone)]
pub struct DagImportStats {
    /// Number of blocks imported
    pub blocks_imported: u64,
    /// Total bytes imported
    pub bytes_imported: u64,
}
