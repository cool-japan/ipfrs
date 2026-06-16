//! HTTP Gateway for IPFRS
//!
//! Provides REST API endpoints for interacting with IPFRS storage.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use ipfrs_core::Result as CoreResult;
use ipfrs_semantic::{RouterConfig, SemanticRouter};
use ipfrs_storage::{BlockStoreConfig, SledBlockStore};
use ipfrs_tensorlogic::TensorLogicStore;
use std::sync::Arc;
use tower_http::trace::TraceLayer;
use tracing::{error, info};

use crate::auth::AuthState;
use crate::auth_handlers;
use crate::graphql::{create_schema, IpfrsSchema};
use crate::streaming;
use crate::tensor;
use crate::tls::TlsConfig;

/// Gateway server state
#[derive(Clone)]
pub struct GatewayState {
    pub(crate) store: Arc<SledBlockStore>,
    semantic: Option<Arc<SemanticRouter>>,
    tensorlogic: Option<Arc<TensorLogicStore<SledBlockStore>>>,
    network: Option<Arc<tokio::sync::Mutex<ipfrs_network::NetworkNode>>>,
    graphql_schema: Option<IpfrsSchema>,
    pub(crate) auth: Option<AuthState>,
}

impl GatewayState {
    /// Create a new gateway state with the given storage configuration
    pub fn new(config: BlockStoreConfig) -> CoreResult<Self> {
        let store = SledBlockStore::new(config)?;
        Ok(Self {
            store: Arc::new(store),
            semantic: None,
            tensorlogic: None,
            network: None,
            graphql_schema: None,
            auth: None,
        })
    }

    /// Enable authentication and authorization
    pub fn with_auth(
        mut self,
        secret: &[u8],
        default_admin_password: Option<&str>,
    ) -> CoreResult<Self> {
        let auth_state = if let Some(password) = default_admin_password {
            AuthState::with_default_admin(secret, password).map_err(|e| {
                ipfrs_core::Error::Internal(format!("Failed to create auth state: {}", e))
            })?
        } else {
            AuthState::new(secret)
        };
        self.auth = Some(auth_state);
        Ok(self)
    }

    /// Enable semantic search capabilities
    pub fn with_semantic(mut self, config: RouterConfig) -> CoreResult<Self> {
        let semantic = SemanticRouter::new(config).map_err(|e| {
            ipfrs_core::Error::Internal(format!("Failed to create semantic router: {}", e))
        })?;
        self.semantic = Some(Arc::new(semantic));
        Ok(self)
    }

    /// Enable tensorlogic capabilities
    pub fn with_tensorlogic(mut self) -> CoreResult<Self> {
        let tensorlogic = TensorLogicStore::new(Arc::clone(&self.store))?;
        self.tensorlogic = Some(Arc::new(tensorlogic));
        Ok(self)
    }

    /// Enable networking capabilities
    pub fn with_network(mut self, network: ipfrs_network::NetworkNode) -> Self {
        self.network = Some(Arc::new(tokio::sync::Mutex::new(network)));
        self
    }

    /// Enable GraphQL API
    pub fn with_graphql(mut self) -> Self {
        let schema = create_schema(
            Arc::clone(&self.store),
            self.semantic.clone(),
            self.tensorlogic.clone(),
        );
        self.graphql_schema = Some(schema);
        self
    }
}

/// Gateway server configuration
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// Listen address
    pub listen_addr: String,
    /// Storage configuration
    pub storage_config: BlockStoreConfig,
    /// Optional TLS configuration for HTTPS
    pub tls_config: Option<TlsConfig>,
    /// Compression configuration
    pub compression_config: crate::middleware::CompressionConfig,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            listen_addr: "127.0.0.1:8080".to_string(),
            storage_config: BlockStoreConfig::default(),
            tls_config: None,
            compression_config: crate::middleware::CompressionConfig::default(),
        }
    }
}

impl GatewayConfig {
    /// Create a production-ready configuration
    ///
    /// Features:
    /// - Listens on all interfaces (0.0.0.0:8080)
    /// - Maximum compression enabled (best ratio)
    /// - Larger cache (500MB)
    /// - Optimized for throughput
    pub fn production() -> Self {
        Self {
            listen_addr: "0.0.0.0:8080".to_string(),
            storage_config: BlockStoreConfig::default()
                .with_path("./ipfrs_data".into())
                .with_cache_mb(500),
            tls_config: None,
            compression_config: crate::middleware::CompressionConfig {
                enable_gzip: true,
                level: crate::middleware::CompressionLevel::Best,
                min_size: 512,
            },
        }
    }

    /// Create a development configuration
    ///
    /// Features:
    /// - Listens on localhost only (127.0.0.1:8080)
    /// - Fast compression (minimal CPU usage)
    /// - Smaller cache (50MB)
    /// - Optimized for quick iteration
    pub fn development() -> Self {
        Self {
            listen_addr: "127.0.0.1:8080".to_string(),
            storage_config: BlockStoreConfig::default()
                .with_path("./dev_data".into())
                .with_cache_mb(50),
            tls_config: None,
            compression_config: crate::middleware::CompressionConfig {
                enable_gzip: true,
                level: crate::middleware::CompressionLevel::Fastest,
                min_size: 2048,
            },
        }
    }

    /// Create a testing configuration
    ///
    /// Features:
    /// - Listens on localhost with random port (127.0.0.1:0)
    /// - Compression disabled for faster tests
    /// - Minimal cache (10MB)
    /// - In-memory or temporary storage
    pub fn testing() -> Self {
        Self {
            listen_addr: "127.0.0.1:0".to_string(),
            storage_config: BlockStoreConfig::default()
                .with_path(std::env::temp_dir().join("ipfrs_test"))
                .with_cache_mb(10),
            tls_config: None,
            compression_config: crate::middleware::CompressionConfig {
                enable_gzip: false,
                level: crate::middleware::CompressionLevel::Fastest,
                min_size: 1048576, // Only compress files > 1MB
            },
        }
    }

    /// Builder: Set the listen address
    pub fn with_listen_addr(mut self, addr: impl Into<String>) -> Self {
        self.listen_addr = addr.into();
        self
    }

    /// Builder: Set the storage path
    pub fn with_storage_path(mut self, path: impl Into<String>) -> Self {
        self.storage_config = self.storage_config.with_path(path.into().into());
        self
    }

    /// Builder: Set the cache size in MB
    pub fn with_cache_mb(mut self, size_mb: usize) -> Self {
        self.storage_config = self.storage_config.with_cache_mb(size_mb);
        self
    }

    /// Builder: Enable TLS/HTTPS
    pub fn with_tls(mut self, tls_config: TlsConfig) -> Self {
        self.tls_config = Some(tls_config);
        self
    }

    /// Builder: Set compression level
    pub fn with_compression_level(mut self, level: crate::middleware::CompressionLevel) -> Self {
        self.compression_config.level = level;
        self
    }

    /// Builder: Enable gzip compression
    pub fn with_full_compression(mut self) -> Self {
        self.compression_config.enable_gzip = true;
        self
    }

    /// Builder: Disable all compression
    pub fn without_compression(mut self) -> Self {
        self.compression_config.enable_gzip = false;
        self
    }

    /// Validate the configuration
    ///
    /// Returns an error if the configuration is invalid
    pub fn validate(&self) -> CoreResult<()> {
        // Validate listen address format
        if self.listen_addr.is_empty() {
            return Err(ipfrs_core::Error::Internal(
                "Listen address cannot be empty".to_string(),
            ));
        }

        // Validate that listen address can be parsed
        self.listen_addr
            .parse::<std::net::SocketAddr>()
            .map_err(|e| ipfrs_core::Error::Internal(format!("Invalid listen address: {}", e)))?;

        // Validate storage path
        if self.storage_config.path.as_os_str().is_empty() {
            return Err(ipfrs_core::Error::Internal(
                "Storage path cannot be empty".to_string(),
            ));
        }

        // Validate compression config
        if self.compression_config.min_size == 0 {
            return Err(ipfrs_core::Error::Internal(
                "Compression min_size must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
}

/// HTTP Gateway server
pub struct Gateway {
    config: GatewayConfig,
    state: GatewayState,
}

impl Gateway {
    /// Create a new gateway with the given configuration
    pub fn new(config: GatewayConfig) -> CoreResult<Self> {
        let state = GatewayState::new(config.storage_config.clone())?;
        Ok(Self { config, state })
    }

    /// Build the router with all endpoints
    fn router(&self) -> Router {
        let mut router = Router::new()
            // Health check (public)
            .route("/health", get(health_check))
            // Prometheus metrics (public)
            .route("/metrics", get(metrics_endpoint))
            // IPFS gateway (public for now)
            .route("/ipfs/:cid", get(get_content))
            // Authentication endpoints (public)
            .route("/api/v0/auth/login", post(auth_handlers::login_handler))
            .route(
                "/api/v0/auth/register",
                post(auth_handlers::register_handler),
            )
            // GraphQL API (public for now)
            .route("/graphql", post(graphql_handler))
            .route("/graphql", get(graphql_playground))
            // Kubo-compatible API
            .route("/api/v0/version", get(api_version))
            .route("/api/v0/add", post(api_add))
            .route("/api/v0/block/get", post(api_block_get))
            .route("/api/v0/block/put", post(api_block_put))
            .route("/api/v0/block/stat", post(api_block_stat))
            .route("/api/v0/cat", post(api_cat))
            .route("/api/v0/dag/get", post(api_dag_get))
            .route("/api/v0/dag/put", post(api_dag_put))
            .route("/api/v0/dag/resolve", post(api_dag_resolve))
            // Semantic search endpoints
            .route("/api/v0/semantic/index", post(api_semantic_index))
            .route("/api/v0/semantic/search", post(api_semantic_search))
            .route("/api/v0/semantic/stats", get(api_semantic_stats))
            .route("/api/v0/semantic/save", post(api_semantic_save))
            .route("/api/v0/semantic/load", post(api_semantic_load))
            // TensorLogic endpoints
            .route("/api/v0/logic/term", post(api_logic_store_term))
            .route("/api/v0/logic/term/:cid", get(api_logic_get_term))
            .route("/api/v0/logic/predicate", post(api_logic_store_predicate))
            .route("/api/v0/logic/rule", post(api_logic_store_rule))
            .route("/api/v0/logic/stats", get(api_logic_stats))
            .route("/api/v0/logic/fact", post(api_logic_add_fact))
            .route("/api/v0/logic/rule/add", post(api_logic_add_rule))
            .route("/api/v0/logic/infer", post(api_logic_infer))
            .route("/api/v0/logic/prove", post(api_logic_prove))
            .route("/api/v0/logic/verify", post(api_logic_verify))
            .route("/api/v0/logic/proof/:cid", get(api_logic_get_proof))
            .route("/api/v0/logic/kb/stats", get(api_logic_kb_stats))
            .route("/api/v0/logic/kb/save", post(api_logic_kb_save))
            .route("/api/v0/logic/kb/load", post(api_logic_kb_load))
            // Network endpoints
            .route("/api/v0/id", get(api_network_id))
            .route("/api/v0/swarm/peers", get(api_swarm_peers))
            .route("/api/v0/swarm/connect", post(api_swarm_connect))
            .route("/api/v0/swarm/disconnect", post(api_swarm_disconnect))
            .route("/api/v0/dht/findprovs", post(api_dht_findprovs))
            .route("/api/v0/dht/provide", post(api_dht_provide))
            // Streaming API (v1)
            .route("/v1/stream/download/:cid", get(streaming::stream_download))
            .route("/v1/stream/upload", post(streaming::stream_upload))
            .route(
                "/v1/progress/:operation_id",
                get(streaming::progress_stream),
            )
            // Batch API (v1)
            .route("/v1/block/batch/get", post(streaming::batch_get))
            .route("/v1/block/batch/put", post(streaming::batch_put))
            .route("/v1/block/batch/has", post(streaming::batch_has))
            // Zero-Copy Tensor API (v1)
            .route("/v1/tensor/:cid", get(tensor::get_tensor))
            .route("/v1/tensor/:cid/info", get(tensor::get_tensor_info))
            .route("/v1/tensor/:cid/arrow", get(tensor::get_tensor_arrow));

        // Add protected auth endpoints if auth is enabled
        if self.state.auth.is_some() {
            router = router
                .route("/api/v0/auth/me", get(auth_handlers::me_handler))
                .route(
                    "/api/v0/auth/permissions",
                    post(auth_handlers::update_permissions_handler),
                )
                .route(
                    "/api/v0/auth/deactivate/:username",
                    post(auth_handlers::deactivate_user_handler),
                )
                // API Key management endpoints
                .route(
                    "/api/v0/auth/keys",
                    post(auth_handlers::create_api_key_handler),
                )
                .route(
                    "/api/v0/auth/keys",
                    get(auth_handlers::list_api_keys_handler),
                )
                .route(
                    "/api/v0/auth/keys/:key_id/revoke",
                    post(auth_handlers::revoke_api_key_handler),
                )
                .route(
                    "/api/v0/auth/keys/:key_id",
                    axum::routing::delete(auth_handlers::delete_api_key_handler),
                );
        }

        // TODO: integrate OxiARC-based HTTP compression when available.
        // tower-http's CompressionLayer (C-backed brotli/deflate) is not used per COOLJAPAN policy.
        // compression_config.enable_gzip is preserved for future OxiARC wiring.
        router
            .with_state(self.state.clone())
            .layer(TraceLayer::new_for_http())
    }

    /// Start the gateway server (HTTP or HTTPS based on configuration)
    pub async fn start(self) -> CoreResult<()> {
        let app = self.router();

        // Print endpoint information
        self.print_endpoints();

        // Start HTTP or HTTPS server based on TLS configuration
        if let Some(ref tls_config) = self.config.tls_config {
            info!(
                "Starting IPFRS HTTPS Gateway on {}",
                self.config.listen_addr
            );

            // Build TLS server config
            let rustls_config = tls_config.build_server_config().await.map_err(|e| {
                ipfrs_core::Error::Internal(format!("TLS configuration error: {}", e))
            })?;

            // Create TLS acceptor
            let addr: std::net::SocketAddr = self
                .config
                .listen_addr
                .parse()
                .map_err(|e| ipfrs_core::Error::Internal(format!("Invalid address: {}", e)))?;

            info!("Gateway listening on https://{}", self.config.listen_addr);
            info!("TLS/SSL enabled");

            // Start HTTPS server with axum-server
            axum_server::bind_rustls(addr, rustls_config)
                .serve(app.into_make_service())
                .await
                .map_err(|e| ipfrs_core::Error::Internal(format!("HTTPS server error: {}", e)))?;
        } else {
            info!("Starting IPFRS HTTP Gateway on {}", self.config.listen_addr);

            let listener = tokio::net::TcpListener::bind(&self.config.listen_addr)
                .await
                .map_err(|e| {
                    ipfrs_core::Error::Internal(format!("Failed to bind to address: {}", e))
                })?;

            info!("Gateway listening on http://{}", self.config.listen_addr);
            info!("Warning: TLS not enabled, using plain HTTP");

            // Start HTTP server
            axum::serve(listener, app)
                .await
                .map_err(|e| ipfrs_core::Error::Internal(format!("HTTP server error: {}", e)))?;
        }

        Ok(())
    }

    /// Print available endpoints
    fn print_endpoints(&self) {
        info!("Endpoints:");
        info!("  GET  /health                      - Health check");
        info!("  GET  /ipfs/{{cid}}                 - Retrieve content");

        // Authentication endpoints (if enabled)
        if self.state.auth.is_some() {
            info!("  POST /api/v0/auth/login           - User login");
            info!("  POST /api/v0/auth/register        - User registration");
            info!("  GET  /api/v0/auth/me              - Current user info");
            info!("  POST /api/v0/auth/permissions     - Update permissions (admin)");
            info!("  POST /api/v0/auth/deactivate/:user - Deactivate user (admin)");
            info!("  POST /api/v0/auth/keys            - Create API key");
            info!("  GET  /api/v0/auth/keys            - List API keys");
            info!("  POST /api/v0/auth/keys/:id/revoke - Revoke API key");
            info!("  DEL  /api/v0/auth/keys/:id        - Delete API key");
        }

        info!("  POST /api/v0/version              - Get version");
        info!("  POST /api/v0/add                  - Upload file");
        info!("  POST /api/v0/block/get            - Get block");
        info!("  POST /api/v0/block/put            - Store raw block");
        info!("  POST /api/v0/block/stat           - Get block stats");
        info!("  POST /api/v0/cat                  - Output content");
        info!("  POST /api/v0/dag/get              - Get DAG node");
        info!("  POST /api/v0/dag/put              - Store DAG node");
        info!("  POST /api/v0/dag/resolve          - Resolve IPLD path");
        info!("  POST /api/v0/semantic/index       - Index content");
        info!("  POST /api/v0/semantic/search      - Search similar");
        info!("  GET  /api/v0/semantic/stats       - Semantic stats");
        info!("  POST /api/v0/semantic/save        - Save semantic index");
        info!("  POST /api/v0/semantic/load        - Load semantic index");
        info!("  POST /api/v0/logic/term           - Store term");
        info!("  GET  /api/v0/logic/term/{{cid}}    - Get term");
        info!("  POST /api/v0/logic/predicate      - Store predicate");
        info!("  POST /api/v0/logic/rule           - Store rule");
        info!("  GET  /api/v0/logic/stats          - Logic stats");
        info!("  POST /api/v0/logic/kb/save        - Save knowledge base");
        info!("  POST /api/v0/logic/kb/load        - Load knowledge base");
        info!("  GET  /api/v0/id                   - Show peer ID");
        info!("  GET  /api/v0/swarm/peers          - List peers");
        info!("  POST /api/v0/swarm/connect        - Connect to peer");
        info!("  POST /api/v0/swarm/disconnect     - Disconnect peer");
        info!("  POST /api/v0/dht/findprovs        - Find providers");
        info!("  POST /api/v0/dht/provide          - Announce content");

        // Streaming API (v1)
        info!("  GET  /v1/stream/download/:cid     - Stream download");
        info!("  POST /v1/stream/upload            - Stream upload");
        info!("  GET  /v1/progress/:operation_id   - SSE progress");

        // Batch API (v1)
        info!("  POST /v1/block/batch/get          - Batch get blocks");
        info!("  POST /v1/block/batch/put          - Batch put blocks");
        info!("  POST /v1/block/batch/has          - Batch check blocks");
    }

    /// Enable GraphQL API
    pub fn with_graphql(mut self) -> Self {
        self.state = self.state.with_graphql();
        self
    }

    /// Enable authentication and authorization
    pub fn with_auth(
        mut self,
        secret: &[u8],
        default_admin_password: Option<&str>,
    ) -> CoreResult<Self> {
        self.state = self.state.with_auth(secret, default_admin_password)?;
        Ok(self)
    }

    /// Enable semantic search capabilities
    pub fn with_semantic(mut self, config: RouterConfig) -> CoreResult<Self> {
        self.state = self.state.with_semantic(config)?;
        Ok(self)
    }

    /// Enable tensorlogic capabilities
    pub fn with_tensorlogic(mut self) -> CoreResult<Self> {
        self.state = self.state.with_tensorlogic()?;
        Ok(self)
    }

    /// Enable networking capabilities
    pub fn with_network(mut self, network: ipfrs_network::NetworkNode) -> Self {
        self.state = self.state.with_network(network);
        self
    }
}

pub(crate) mod routes;

#[allow(unused_imports)]
use routes::*;

// ============================================================================
// Error Handling
// ============================================================================

/// Application error types
#[derive(Debug)]
enum AppError {
    InvalidCid(String),
    BlockNotFound(String),
    NotFound(String),
    Upload(String),
    Storage(ipfrs_core::Error),
    FeatureDisabled(String),
    Semantic(String),
    Logic(String),
    Network(String),
}

impl From<ipfrs_core::Error> for AppError {
    fn from(err: ipfrs_core::Error) -> Self {
        AppError::Storage(err)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AppError::InvalidCid(cid) => (StatusCode::BAD_REQUEST, format!("Invalid CID: {}", cid)),
            AppError::BlockNotFound(cid) => {
                (StatusCode::NOT_FOUND, format!("Block not found: {}", cid))
            }
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            AppError::Upload(msg) => {
                error!("Upload error: {}", msg);
                (StatusCode::BAD_REQUEST, format!("Upload error: {}", msg))
            }
            AppError::Storage(err) => {
                error!("Storage error: {}", err);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Storage error: {}", err),
                )
            }
            AppError::FeatureDisabled(msg) => (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Feature not available: {}", msg),
            ),
            AppError::Semantic(msg) => {
                error!("Semantic error: {}", msg);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Semantic error: {}", msg),
                )
            }
            AppError::Logic(msg) => {
                error!("Logic error: {}", msg);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Logic error: {}", msg),
                )
            }
            AppError::Network(msg) => {
                error!("Network error: {}", msg);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Network error: {}", msg),
                )
            }
        };

        (status, message).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::routes::{build_multipart_response, merge_ranges, parse_multi_range, parse_range};
    use super::*;
    use crate::middleware::CacheConfig;
    use axum::http::header;

    #[test]
    fn test_parse_single_range() {
        // Standard range
        assert_eq!(parse_range("bytes=0-100", 1000), Some((0, 101)));

        // Open-ended range (to end)
        assert_eq!(parse_range("bytes=500-", 1000), Some((500, 1000)));

        // Invalid: start >= size
        assert_eq!(parse_range("bytes=1000-1100", 1000), None);

        // Invalid: start > end
        assert_eq!(parse_range("bytes=500-100", 1000), None);

        // Invalid format
        assert_eq!(parse_range("bytes=abc-100", 1000), None);
        assert_eq!(parse_range("invalid", 1000), None);
    }

    #[test]
    fn test_parse_multi_range() {
        // Single range via multi-range parser
        let ranges = parse_multi_range("bytes=0-100", 1000);
        assert_eq!(ranges, Some(vec![(0, 101)]));

        // Multiple ranges
        let ranges = parse_multi_range("bytes=0-100,200-300", 1000);
        assert_eq!(ranges, Some(vec![(0, 101), (200, 301)]));

        // Multiple ranges with spaces
        let ranges = parse_multi_range("bytes=0-100, 200-300, 500-600", 1000);
        assert_eq!(ranges, Some(vec![(0, 101), (200, 301), (500, 601)]));

        // Suffix range (last N bytes)
        let ranges = parse_multi_range("bytes=-500", 1000);
        assert_eq!(ranges, Some(vec![(500, 1000)]));

        // Invalid range
        assert_eq!(parse_multi_range("bytes=1000-1100", 1000), None);

        // Invalid format
        assert_eq!(parse_multi_range("invalid", 1000), None);
    }

    #[test]
    fn test_merge_ranges() {
        // Non-overlapping ranges (should not merge)
        let ranges = vec![(0, 100), (200, 300)];
        assert_eq!(merge_ranges(ranges), vec![(0, 100), (200, 300)]);

        // Overlapping ranges (should merge)
        let ranges = vec![(0, 150), (100, 200)];
        assert_eq!(merge_ranges(ranges), vec![(0, 200)]);

        // Adjacent ranges (should merge)
        let ranges = vec![(0, 100), (100, 200)];
        assert_eq!(merge_ranges(ranges), vec![(0, 200)]);

        // Out of order ranges (should sort and merge)
        let ranges = vec![(200, 300), (0, 100), (50, 150)];
        assert_eq!(merge_ranges(ranges), vec![(0, 150), (200, 300)]);

        // Single range (no change)
        let ranges = vec![(50, 100)];
        assert_eq!(merge_ranges(ranges), vec![(50, 100)]);

        // Empty (no change)
        let ranges: Vec<(usize, usize)> = vec![];
        assert_eq!(merge_ranges(ranges), vec![]);
    }

    #[test]
    fn test_build_multipart_response() {
        let data = b"Hello, World! This is test data for multi-range requests.";
        let ranges = vec![(0, 5), (7, 12)];
        let total_size = data.len();
        let config = CacheConfig::default();

        let response = build_multipart_response(data, &ranges, total_size, "QmTest123", &config);

        assert_eq!(response.status(), StatusCode::PARTIAL_CONTENT);

        let content_type = response
            .headers()
            .get(header::CONTENT_TYPE)
            .expect("test: CONTENT_TYPE header must be present")
            .to_str()
            .expect("test: CONTENT_TYPE header value must be valid UTF-8");
        assert!(content_type.starts_with("multipart/byteranges"));
        assert!(content_type.contains("boundary="));

        // Check caching headers are present
        assert!(response.headers().contains_key(header::ETAG));
        assert!(response.headers().contains_key(header::CACHE_CONTROL));
    }

    #[test]
    fn test_config_presets() {
        // Test production preset
        let prod = GatewayConfig::production();
        assert_eq!(prod.listen_addr, "0.0.0.0:8080");
        assert!(prod.compression_config.enable_gzip);

        // Test development preset
        let dev = GatewayConfig::development();
        assert_eq!(dev.listen_addr, "127.0.0.1:8080");
        assert!(dev.compression_config.enable_gzip);

        // Test testing preset
        let test = GatewayConfig::testing();
        assert_eq!(test.listen_addr, "127.0.0.1:0");
        assert!(!test.compression_config.enable_gzip);
    }

    #[test]
    fn test_config_builders() {
        let config = GatewayConfig::default()
            .with_listen_addr("0.0.0.0:9090")
            .with_storage_path("/custom/path")
            .with_cache_mb(200)
            .with_full_compression();

        assert_eq!(config.listen_addr, "0.0.0.0:9090");
        assert!(config.compression_config.enable_gzip);
    }

    #[test]
    fn test_config_validation() {
        // Valid config should pass
        let valid_config = GatewayConfig::default();
        assert!(valid_config.validate().is_ok());

        // Invalid address should fail
        let invalid_addr = GatewayConfig {
            listen_addr: "invalid-address".to_string(),
            ..Default::default()
        };
        assert!(invalid_addr.validate().is_err());

        // Empty address should fail
        let empty_addr = GatewayConfig {
            listen_addr: "".to_string(),
            ..Default::default()
        };
        assert!(empty_addr.validate().is_err());
    }

    #[test]
    fn test_compression_helpers() {
        let config_with = GatewayConfig::default().with_full_compression();
        assert!(config_with.compression_config.enable_gzip);

        let config_without = GatewayConfig::default().without_compression();
        assert!(!config_without.compression_config.enable_gzip);
    }
}
