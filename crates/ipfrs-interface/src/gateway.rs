//! HTTP Gateway for IPFRS
//!
//! Provides REST API endpoints for interacting with IPFRS storage.

use async_graphql::http::{playground_source, GraphQLPlaygroundConfig};
use async_graphql_axum::{GraphQLRequest, GraphQLResponse};
use axum::{
    body::Body,
    extract::{Multipart, Path, State},
    http::{header, HeaderMap, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use ipfrs_core::{Cid, Result as CoreResult};
use ipfrs_semantic::{DistanceMetric, QueryFilter, RouterConfig, SemanticRouter};
use ipfrs_storage::{BlockStoreConfig, BlockStoreTrait, SledBlockStore};
use ipfrs_tensorlogic::{Predicate, Proof, Rule, Substitution, TensorLogicStore, Term};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::compression::CompressionLayer;
use tower_http::trace::TraceLayer;
use tracing::{error, info};

use crate::middleware::{
    add_caching_headers, check_etag_match, not_modified_response, CacheConfig,
};

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
                enable_brotli: true,
                enable_deflate: true,
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
                enable_brotli: false,
                enable_deflate: false,
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
                .with_path("/tmp/ipfrs_test".into())
                .with_cache_mb(10),
            tls_config: None,
            compression_config: crate::middleware::CompressionConfig {
                enable_gzip: false,
                enable_brotli: false,
                enable_deflate: false,
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

    /// Builder: Enable all compression formats
    pub fn with_full_compression(mut self) -> Self {
        self.compression_config.enable_gzip = true;
        self.compression_config.enable_brotli = true;
        self.compression_config.enable_deflate = true;
        self
    }

    /// Builder: Disable all compression
    pub fn without_compression(mut self) -> Self {
        self.compression_config.enable_gzip = false;
        self.compression_config.enable_brotli = false;
        self.compression_config.enable_deflate = false;
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

        // Apply compression layer
        // Note: tower-http's CompressionLayer uses default compression levels.
        // The compression_config provides configuration for documentation and future enhancements.
        // Current behavior: gzip, brotli, and deflate with default levels.
        let router = if self.config.compression_config.enable_gzip
            || self.config.compression_config.enable_brotli
            || self.config.compression_config.enable_deflate
        {
            router.layer(CompressionLayer::new())
        } else {
            router
        };

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

// ============================================================================
// Handler Functions
// ============================================================================

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "service": "ipfrs-gateway",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// Prometheus metrics endpoint
///
/// Returns metrics in Prometheus text exposition format for scraping
async fn metrics_endpoint() -> impl IntoResponse {
    match crate::metrics::encode_metrics() {
        Ok(metrics) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
            metrics,
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to encode metrics: {}", e),
        )
            .into_response(),
    }
}

/// Get content via IPFS gateway with range request support
///
/// Supports:
/// - Single range requests (Range: bytes=0-100)
/// - Multi-range requests (Range: bytes=0-100,200-300)
/// - Conditional requests (If-None-Match)
/// - Caching headers (ETag, Cache-Control)
async fn get_content(
    State(state): State<GatewayState>,
    Path(cid_str): Path<String>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    let cid: Cid = cid_str
        .parse()
        .map_err(|_| AppError::InvalidCid(cid_str.clone()))?;

    // Cache configuration for CID-based content
    let cache_config = CacheConfig::default();

    // Check for conditional request (If-None-Match)
    if check_etag_match(&headers, &cid_str) {
        return Ok(not_modified_response(&cid_str, &cache_config));
    }

    match state.store.get(&cid).await? {
        Some(block) => {
            let data = block.data();
            let total_size = data.len();

            // Check for Range header
            if let Some(range_header) = headers.get(header::RANGE) {
                if let Ok(range_str) = range_header.to_str() {
                    // Try parsing multi-range first
                    if let Some(ranges) = parse_multi_range(range_str, total_size) {
                        if ranges.len() == 1 {
                            // Single range - simple response
                            let (start, end) = ranges[0];
                            let slice = &data[start..end];
                            let content_range =
                                format!("bytes {}-{}/{}", start, end - 1, total_size);

                            let mut response = Response::builder()
                                .status(StatusCode::PARTIAL_CONTENT)
                                .header(header::CONTENT_RANGE, content_range)
                                .header(header::CONTENT_LENGTH, slice.len().to_string())
                                .header(header::CONTENT_TYPE, "application/octet-stream")
                                .header(header::ACCEPT_RANGES, "bytes")
                                .body(Body::from(slice.to_vec()))
                                .unwrap();

                            // Add caching headers
                            add_caching_headers(response.headers_mut(), &cid_str, &cache_config);

                            return Ok(response);
                        } else {
                            // Multi-range response with multipart/byteranges
                            return Ok(build_multipart_response(
                                data,
                                &ranges,
                                total_size,
                                &cid_str,
                                &cache_config,
                            ));
                        }
                    }
                }
            }

            // No range request or invalid range, return full content
            let mut response = Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "application/octet-stream")
                .header(header::CONTENT_LENGTH, total_size.to_string())
                .header(header::ACCEPT_RANGES, "bytes")
                .body(Body::from(data.to_vec()))
                .unwrap();

            // Add caching headers
            add_caching_headers(response.headers_mut(), &cid_str, &cache_config);

            Ok(response)
        }
        None => Err(AppError::BlockNotFound(cid_str)),
    }
}

/// Parse HTTP Range header for single range
/// Returns (start, end) where end is exclusive
#[allow(dead_code)]
fn parse_range(range_str: &str, total_size: usize) -> Option<(usize, usize)> {
    // Expected format: "bytes=start-end" or "bytes=start-"
    let range_str = range_str.strip_prefix("bytes=")?;

    if let Some((start_str, end_str)) = range_str.split_once('-') {
        let start: usize = start_str.parse().ok()?;

        let end = if end_str.is_empty() {
            total_size
        } else {
            end_str.parse::<usize>().ok()? + 1
        };

        if start < total_size && start < end && end <= total_size {
            Some((start, end))
        } else {
            None
        }
    } else {
        None
    }
}

/// Parse HTTP Range header for multiple ranges
/// Returns Vec of (start, end) tuples where end is exclusive
/// Supports formats: "bytes=0-100", "bytes=0-100,200-300", "bytes=0-100, 200-300"
fn parse_multi_range(range_str: &str, total_size: usize) -> Option<Vec<(usize, usize)>> {
    let range_str = range_str.strip_prefix("bytes=")?;

    let mut ranges = Vec::new();

    for part in range_str.split(',') {
        let part = part.trim();
        if let Some((start_str, end_str)) = part.split_once('-') {
            // Handle suffix range (e.g., "-500" means last 500 bytes)
            if start_str.is_empty() {
                let suffix_len: usize = end_str.parse().ok()?;
                let start = total_size.saturating_sub(suffix_len);
                ranges.push((start, total_size));
                continue;
            }

            let start: usize = start_str.parse().ok()?;

            let end = if end_str.is_empty() {
                total_size
            } else {
                end_str.parse::<usize>().ok()? + 1
            };

            if start < total_size && start < end && end <= total_size {
                ranges.push((start, end));
            } else {
                return None; // Invalid range
            }
        } else {
            return None; // Invalid format
        }
    }

    if ranges.is_empty() {
        None
    } else {
        // Merge overlapping and adjacent ranges for efficiency
        Some(merge_ranges(ranges))
    }
}

/// Merge overlapping and adjacent ranges
fn merge_ranges(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if ranges.len() <= 1 {
        return ranges;
    }

    // Sort by start position
    ranges.sort_by_key(|r| r.0);

    let mut merged = Vec::new();
    let mut current = ranges[0];

    for range in ranges.into_iter().skip(1) {
        // Check if ranges overlap or are adjacent
        if range.0 <= current.1 {
            // Extend current range
            current.1 = current.1.max(range.1);
        } else {
            merged.push(current);
            current = range;
        }
    }
    merged.push(current);

    merged
}

/// Build multipart/byteranges response for multi-range requests
fn build_multipart_response(
    data: &[u8],
    ranges: &[(usize, usize)],
    total_size: usize,
    cid: &str,
    cache_config: &CacheConfig,
) -> Response {
    // Generate a boundary string
    let boundary = format!("ipfrs_boundary_{:x}", rand::random::<u64>());

    // Build multipart body
    let mut body = Vec::new();

    for (start, end) in ranges {
        // Part header
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(b"Content-Type: application/octet-stream\r\n");
        body.extend_from_slice(
            format!(
                "Content-Range: bytes {}-{}/{}\r\n\r\n",
                start,
                end - 1,
                total_size
            )
            .as_bytes(),
        );

        // Part data
        body.extend_from_slice(&data[*start..*end]);
        body.extend_from_slice(b"\r\n");
    }

    // Final boundary
    body.extend_from_slice(format!("--{}--\r\n", boundary).as_bytes());

    let content_type = format!("multipart/byteranges; boundary={}", boundary);

    let mut response = Response::builder()
        .status(StatusCode::PARTIAL_CONTENT)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CONTENT_LENGTH, body.len().to_string())
        .header(header::ACCEPT_RANGES, "bytes")
        .body(Body::from(body))
        .unwrap();

    // Add caching headers
    add_caching_headers(response.headers_mut(), cid, cache_config);

    response
}

/// Version endpoint
#[derive(Serialize)]
struct VersionResponse {
    #[serde(rename = "Version")]
    version: String,
    #[serde(rename = "System")]
    system: String,
}

async fn api_version() -> impl IntoResponse {
    Json(VersionResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
        system: "ipfrs/0.3.0".to_string(),
    })
}

/// Block get request
#[derive(Deserialize)]
struct BlockRequest {
    arg: String,
}

async fn api_block_get(
    State(state): State<GatewayState>,
    Json(req): Json<BlockRequest>,
) -> Result<Response, AppError> {
    let cid: Cid = req
        .arg
        .parse()
        .map_err(|_| AppError::InvalidCid(req.arg.clone()))?;

    match state.store.get(&cid).await? {
        Some(block) => Ok((StatusCode::OK, block.data().to_vec()).into_response()),
        None => Err(AppError::BlockNotFound(req.arg)),
    }
}

/// Block stat response
#[derive(Serialize)]
struct BlockStatResponse {
    #[serde(rename = "Key")]
    key: String,
    #[serde(rename = "Size")]
    size: u64,
}

async fn api_block_stat(
    State(state): State<GatewayState>,
    Json(req): Json<BlockRequest>,
) -> Result<Json<BlockStatResponse>, AppError> {
    let cid: Cid = req
        .arg
        .parse()
        .map_err(|_| AppError::InvalidCid(req.arg.clone()))?;

    match state.store.get(&cid).await? {
        Some(block) => Ok(Json(BlockStatResponse {
            key: req.arg,
            size: block.size(),
        })),
        None => Err(AppError::BlockNotFound(req.arg)),
    }
}

async fn api_cat(
    State(state): State<GatewayState>,
    Json(req): Json<BlockRequest>,
) -> Result<Response, AppError> {
    let cid: Cid = req
        .arg
        .parse()
        .map_err(|_| AppError::InvalidCid(req.arg.clone()))?;

    match state.store.get(&cid).await? {
        Some(block) => Ok((StatusCode::OK, block.data().to_vec()).into_response()),
        None => Err(AppError::BlockNotFound(req.arg)),
    }
}

/// Add response
#[derive(Serialize)]
struct AddResponse {
    #[serde(rename = "Name")]
    name: String,
    #[serde(rename = "Hash")]
    hash: String,
    #[serde(rename = "Size")]
    size: String,
}

async fn api_add(
    State(state): State<GatewayState>,
    mut multipart: Multipart,
) -> Result<Json<AddResponse>, AppError> {
    use bytes::Bytes;
    use ipfrs_core::Block;

    // Process the first file field
    if let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::Upload(format!("Failed to read multipart field: {}", e)))?
    {
        let name = field.file_name().unwrap_or("upload").to_string();
        let data = field
            .bytes()
            .await
            .map_err(|e| AppError::Upload(format!("Failed to read file data: {}", e)))?;

        // Create block from uploaded data
        let bytes_data = Bytes::from(data.to_vec());
        let block = Block::new(bytes_data)
            .map_err(|e| AppError::Upload(format!("Failed to create block: {}", e)))?;
        let cid = *block.cid();
        let size = block.size();

        // Store block
        state.store.put(&block).await?;

        info!("Added file '{}' as {}", name, cid);

        Ok(Json(AddResponse {
            name,
            hash: cid.to_string(),
            size: size.to_string(),
        }))
    } else {
        Err(AppError::Upload("No file provided".to_string()))
    }
}

/// Block put request with raw data
async fn api_block_put(
    State(state): State<GatewayState>,
    body: axum::body::Bytes,
) -> Result<Json<AddResponse>, AppError> {
    use ipfrs_core::Block;

    // Create block from raw bytes
    let block =
        Block::new(body).map_err(|e| AppError::Upload(format!("Failed to create block: {}", e)))?;
    let cid = *block.cid();
    let size = block.size();

    // Store block
    state.store.put(&block).await?;

    info!("Stored raw block {}", cid);

    Ok(Json(AddResponse {
        name: cid.to_string(),
        hash: cid.to_string(),
        size: size.to_string(),
    }))
}

/// DAG get request
#[derive(Deserialize)]
struct DagRequest {
    arg: String,
}

/// DAG get response
#[derive(Serialize)]
struct DagGetResponse {
    #[serde(rename = "Data")]
    data: String,
}

async fn api_dag_get(
    State(state): State<GatewayState>,
    Json(req): Json<DagRequest>,
) -> Result<Json<DagGetResponse>, AppError> {
    let cid: Cid = req
        .arg
        .parse()
        .map_err(|_| AppError::InvalidCid(req.arg.clone()))?;

    match state.store.get(&cid).await? {
        Some(block) => {
            // Convert block data to base64 for JSON transport
            let data_base64 =
                base64::Engine::encode(&base64::engine::general_purpose::STANDARD, block.data());

            Ok(Json(DagGetResponse { data: data_base64 }))
        }
        None => Err(AppError::BlockNotFound(req.arg)),
    }
}

/// DAG put response
#[derive(Serialize)]
struct DagPutResponse {
    #[serde(rename = "Cid")]
    cid: CidInfo,
}

#[derive(Serialize)]
struct CidInfo {
    #[serde(rename = "/")]
    cid: String,
}

async fn api_dag_put(
    State(state): State<GatewayState>,
    body: axum::body::Bytes,
) -> Result<Json<DagPutResponse>, AppError> {
    use ipfrs_core::Block;

    // Create block from DAG data
    let block = Block::new(body)
        .map_err(|e| AppError::Upload(format!("Failed to create DAG block: {}", e)))?;
    let cid = *block.cid();

    // Store block
    state.store.put(&block).await?;

    info!("Stored DAG node {}", cid);

    Ok(Json(DagPutResponse {
        cid: CidInfo {
            cid: cid.to_string(),
        },
    }))
}

/// DAG resolve request
#[derive(Deserialize)]
struct DagResolveRequest {
    arg: String,
}

/// DAG resolve response
#[derive(Serialize)]
struct DagResolveResponse {
    #[serde(rename = "Cid")]
    cid: CidInfo,
    #[serde(rename = "RemPath")]
    rem_path: String,
}

async fn api_dag_resolve(
    State(_state): State<GatewayState>,
    Json(req): Json<DagResolveRequest>,
) -> Result<Json<DagResolveResponse>, AppError> {
    // Parse the path (e.g., "/ipfs/Qm.../path/to/data")
    let path = req.arg.trim_start_matches("/ipfs/");
    let parts: Vec<&str> = path.splitn(2, '/').collect();

    let cid_str = parts[0];
    let cid: Cid = cid_str
        .parse()
        .map_err(|_| AppError::InvalidCid(cid_str.to_string()))?;

    let sub_path = if parts.len() > 1 { parts[1] } else { "" };

    // For now, we just return the root CID and the remainder path
    // Full IPLD path resolution would require parsing DAG-CBOR/JSON
    Ok(Json(DagResolveResponse {
        cid: CidInfo {
            cid: cid.to_string(),
        },
        rem_path: sub_path.to_string(),
    }))
}

// ============================================================================
// Semantic Search Handlers
// ============================================================================

/// Semantic index request
#[derive(Deserialize)]
struct SemanticIndexRequest {
    cid: String,
    embedding: Vec<f32>,
}

/// Semantic index response
#[derive(Serialize)]
struct SemanticIndexResponse {
    indexed: bool,
}

async fn api_semantic_index(
    State(state): State<GatewayState>,
    Json(req): Json<SemanticIndexRequest>,
) -> Result<Json<SemanticIndexResponse>, AppError> {
    let semantic = state
        .semantic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("Semantic search not enabled".to_string()))?;

    let cid: Cid = req
        .cid
        .parse()
        .map_err(|_| AppError::InvalidCid(req.cid.clone()))?;

    semantic
        .add(&cid, &req.embedding)
        .map_err(|e| AppError::Semantic(format!("Failed to index: {}", e)))?;

    info!("Indexed content {} with embedding", cid);

    Ok(Json(SemanticIndexResponse { indexed: true }))
}

/// Semantic search request
#[derive(Deserialize)]
struct SemanticSearchRequest {
    query: Vec<f32>,
    k: Option<usize>,
    filter: Option<QueryFilter>,
}

/// Semantic search response
#[derive(Serialize)]
struct SemanticSearchResponse {
    results: Vec<SearchResultJson>,
}

#[derive(Serialize)]
struct SearchResultJson {
    cid: String,
    score: f32,
}

async fn api_semantic_search(
    State(state): State<GatewayState>,
    Json(req): Json<SemanticSearchRequest>,
) -> Result<Json<SemanticSearchResponse>, AppError> {
    let semantic = state
        .semantic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("Semantic search not enabled".to_string()))?;

    let k = req.k.unwrap_or(10);

    let results = if let Some(filter) = req.filter {
        semantic
            .query_with_filter(&req.query, k, filter)
            .await
            .map_err(|e| AppError::Semantic(format!("Search failed: {}", e)))?
    } else {
        semantic
            .query(&req.query, k)
            .await
            .map_err(|e| AppError::Semantic(format!("Search failed: {}", e)))?
    };

    let results_json: Vec<SearchResultJson> = results
        .into_iter()
        .map(|r| SearchResultJson {
            cid: r.cid.to_string(),
            score: r.score,
        })
        .collect();

    Ok(Json(SemanticSearchResponse {
        results: results_json,
    }))
}

/// Semantic stats response
#[derive(Serialize)]
struct SemanticStatsResponse {
    num_vectors: usize,
    dimension: usize,
    metric: String,
    cache_size: usize,
    cache_capacity: usize,
}

async fn api_semantic_stats(
    State(state): State<GatewayState>,
) -> Result<Json<SemanticStatsResponse>, AppError> {
    let semantic = state
        .semantic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("Semantic search not enabled".to_string()))?;

    let router_stats = semantic.stats();
    let cache_stats = semantic.cache_stats();

    let metric_str = match router_stats.metric {
        DistanceMetric::Cosine => "cosine",
        DistanceMetric::L2 => "l2",
        DistanceMetric::DotProduct => "dotproduct",
    };

    Ok(Json(SemanticStatsResponse {
        num_vectors: router_stats.num_vectors,
        dimension: router_stats.dimension,
        metric: metric_str.to_string(),
        cache_size: cache_stats.size,
        cache_capacity: cache_stats.capacity,
    }))
}

/// Semantic save request
#[derive(Deserialize)]
struct SemanticSaveRequest {
    path: String,
}

/// Semantic save response
#[derive(Serialize)]
struct SemanticSaveResponse {
    success: bool,
    path: String,
}

async fn api_semantic_save(
    State(state): State<GatewayState>,
    Json(req): Json<SemanticSaveRequest>,
) -> Result<Json<SemanticSaveResponse>, AppError> {
    let semantic = state
        .semantic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("Semantic search not enabled".to_string()))?;

    semantic
        .save_index(&req.path)
        .await
        .map_err(|e| AppError::Semantic(format!("Failed to save index: {}", e)))?;

    info!("Saved semantic index to {}", req.path);

    Ok(Json(SemanticSaveResponse {
        success: true,
        path: req.path,
    }))
}

/// Semantic load request
#[derive(Deserialize)]
struct SemanticLoadRequest {
    path: String,
}

/// Semantic load response
#[derive(Serialize)]
struct SemanticLoadResponse {
    success: bool,
    path: String,
}

async fn api_semantic_load(
    State(state): State<GatewayState>,
    Json(req): Json<SemanticLoadRequest>,
) -> Result<Json<SemanticLoadResponse>, AppError> {
    let semantic = state
        .semantic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("Semantic search not enabled".to_string()))?;

    semantic
        .load_index(&req.path)
        .await
        .map_err(|e| AppError::Semantic(format!("Failed to load index: {}", e)))?;

    info!("Loaded semantic index from {}", req.path);

    Ok(Json(SemanticLoadResponse {
        success: true,
        path: req.path,
    }))
}

// ============================================================================
// TensorLogic Handlers
// ============================================================================

/// Logic store term request
#[derive(Deserialize)]
struct LogicStoreTermRequest {
    term: Term,
}

/// Logic store response
#[derive(Serialize)]
struct LogicStoreResponse {
    cid: String,
}

async fn api_logic_store_term(
    State(state): State<GatewayState>,
    Json(req): Json<LogicStoreTermRequest>,
) -> Result<Json<LogicStoreResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    let cid = tensorlogic
        .store_term(&req.term)
        .await
        .map_err(|e| AppError::Logic(format!("Failed to store term: {}", e)))?;

    info!("Stored term as {}", cid);

    Ok(Json(LogicStoreResponse {
        cid: cid.to_string(),
    }))
}

/// Logic get term response
#[derive(Serialize)]
struct LogicGetTermResponse {
    term: Term,
}

async fn api_logic_get_term(
    State(state): State<GatewayState>,
    Path(cid_str): Path<String>,
) -> Result<Json<LogicGetTermResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    let cid: Cid = cid_str
        .parse()
        .map_err(|_| AppError::InvalidCid(cid_str.clone()))?;

    let term = tensorlogic
        .get_term(&cid)
        .await
        .map_err(|e| AppError::Logic(format!("Failed to get term: {}", e)))?
        .ok_or_else(|| AppError::NotFound(format!("Term not found: {}", cid_str)))?;

    Ok(Json(LogicGetTermResponse { term }))
}

/// Logic store predicate request
#[derive(Deserialize)]
struct LogicStorePredicateRequest {
    predicate: Predicate,
}

async fn api_logic_store_predicate(
    State(state): State<GatewayState>,
    Json(req): Json<LogicStorePredicateRequest>,
) -> Result<Json<LogicStoreResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    let cid = tensorlogic
        .store_predicate(&req.predicate)
        .await
        .map_err(|e| AppError::Logic(format!("Failed to store predicate: {}", e)))?;

    info!("Stored predicate as {}", cid);

    Ok(Json(LogicStoreResponse {
        cid: cid.to_string(),
    }))
}

/// Logic store rule request
#[derive(Deserialize)]
struct LogicStoreRuleRequest {
    rule: Rule,
}

async fn api_logic_store_rule(
    State(state): State<GatewayState>,
    Json(req): Json<LogicStoreRuleRequest>,
) -> Result<Json<LogicStoreResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    let cid = tensorlogic
        .store_rule(&req.rule)
        .await
        .map_err(|e| AppError::Logic(format!("Failed to store rule: {}", e)))?;

    info!("Stored rule as {}", cid);

    Ok(Json(LogicStoreResponse {
        cid: cid.to_string(),
    }))
}

/// Logic stats response
#[derive(Serialize)]
struct LogicStatsResponse {
    enabled: bool,
}

async fn api_logic_stats(
    State(state): State<GatewayState>,
) -> Result<Json<LogicStatsResponse>, AppError> {
    let enabled = state.tensorlogic.is_some();

    Ok(Json(LogicStatsResponse { enabled }))
}

/// Add fact request
#[derive(Deserialize)]
struct AddFactRequest {
    fact: Predicate,
}

/// Add fact response
#[derive(Serialize)]
struct AddFactResponse {
    success: bool,
}

async fn api_logic_add_fact(
    State(state): State<GatewayState>,
    Json(req): Json<AddFactRequest>,
) -> Result<Json<AddFactResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    tensorlogic
        .add_fact(req.fact)
        .map_err(|e| AppError::Logic(format!("Failed to add fact: {}", e)))?;

    Ok(Json(AddFactResponse { success: true }))
}

/// Add rule request
#[derive(Deserialize)]
struct AddRuleRequest {
    rule: Rule,
}

/// Add rule response
#[derive(Serialize)]
struct AddRuleResponse {
    success: bool,
}

async fn api_logic_add_rule(
    State(state): State<GatewayState>,
    Json(req): Json<AddRuleRequest>,
) -> Result<Json<AddRuleResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    tensorlogic
        .add_rule(req.rule)
        .map_err(|e| AppError::Logic(format!("Failed to add rule: {}", e)))?;

    Ok(Json(AddRuleResponse { success: true }))
}

/// Infer request
#[derive(Deserialize)]
struct InferRequest {
    goal: Predicate,
}

/// Infer response
#[derive(Serialize)]
struct InferResponse {
    solutions: Vec<Substitution>,
}

async fn api_logic_infer(
    State(state): State<GatewayState>,
    Json(req): Json<InferRequest>,
) -> Result<Json<InferResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    let solutions = tensorlogic
        .infer(&req.goal)
        .map_err(|e| AppError::Logic(format!("Inference failed: {}", e)))?;

    Ok(Json(InferResponse { solutions }))
}

/// Prove request
#[derive(Deserialize)]
struct ProveRequest {
    goal: Predicate,
}

/// Prove response
#[derive(Serialize)]
struct ProveResponse {
    proof: Option<Proof>,
    cid: Option<String>,
}

/// Verify request
#[derive(Deserialize)]
struct VerifyRequest {
    proof: Proof,
}

/// Verify response
#[derive(Serialize)]
struct VerifyResponse {
    valid: bool,
}

async fn api_logic_prove(
    State(state): State<GatewayState>,
    Json(req): Json<ProveRequest>,
) -> Result<Json<ProveResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    let proof = tensorlogic
        .prove(&req.goal)
        .map_err(|e| AppError::Logic(format!("Proof generation failed: {}", e)))?;

    if let Some(ref p) = proof {
        let cid = tensorlogic
            .store_proof(p)
            .await
            .map_err(|e| AppError::Logic(format!("Failed to store proof: {}", e)))?;

        Ok(Json(ProveResponse {
            proof,
            cid: Some(cid.to_string()),
        }))
    } else {
        Ok(Json(ProveResponse {
            proof: None,
            cid: None,
        }))
    }
}

async fn api_logic_verify(
    State(state): State<GatewayState>,
    Json(req): Json<VerifyRequest>,
) -> Result<Json<VerifyResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    let valid = tensorlogic
        .verify_proof(&req.proof)
        .map_err(|e| AppError::Logic(format!("Proof verification failed: {}", e)))?;

    Ok(Json(VerifyResponse { valid }))
}

/// Get proof response
#[derive(Serialize)]
struct GetProofResponse {
    proof: Proof,
}

async fn api_logic_get_proof(
    State(state): State<GatewayState>,
    Path(cid_str): Path<String>,
) -> Result<Json<GetProofResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    let cid: Cid = cid_str
        .parse()
        .map_err(|e| AppError::InvalidCid(format!("Invalid CID: {}", e)))?;

    let proof = tensorlogic
        .get_proof(&cid)
        .await
        .map_err(|e| AppError::Logic(format!("Failed to get proof: {}", e)))?
        .ok_or_else(|| AppError::NotFound(format!("Proof not found: {}", cid)))?;

    Ok(Json(GetProofResponse { proof }))
}

/// KB stats response
#[derive(Serialize)]
struct KbStatsResponse {
    num_facts: usize,
    num_rules: usize,
}

async fn api_logic_kb_stats(
    State(state): State<GatewayState>,
) -> Result<Json<KbStatsResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    let kb_stats = tensorlogic.kb_stats();

    Ok(Json(KbStatsResponse {
        num_facts: kb_stats.num_facts,
        num_rules: kb_stats.num_rules,
    }))
}

/// KB save request
#[derive(Deserialize)]
struct KbSaveRequest {
    path: String,
}

/// KB save response
#[derive(Serialize)]
struct KbSaveResponse {
    success: bool,
    path: String,
}

async fn api_logic_kb_save(
    State(state): State<GatewayState>,
    Json(req): Json<KbSaveRequest>,
) -> Result<Json<KbSaveResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    tensorlogic
        .save_kb(&req.path)
        .await
        .map_err(|e| AppError::Logic(format!("Failed to save knowledge base: {}", e)))?;

    info!("Saved knowledge base to {}", req.path);

    Ok(Json(KbSaveResponse {
        success: true,
        path: req.path,
    }))
}

/// KB load request
#[derive(Deserialize)]
struct KbLoadRequest {
    path: String,
}

/// KB load response
#[derive(Serialize)]
struct KbLoadResponse {
    success: bool,
    path: String,
}

async fn api_logic_kb_load(
    State(state): State<GatewayState>,
    Json(req): Json<KbLoadRequest>,
) -> Result<Json<KbLoadResponse>, AppError> {
    let tensorlogic = state
        .tensorlogic
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("TensorLogic not enabled".to_string()))?;

    tensorlogic
        .load_kb(&req.path)
        .await
        .map_err(|e| AppError::Logic(format!("Failed to load knowledge base: {}", e)))?;

    info!("Loaded knowledge base from {}", req.path);

    Ok(Json(KbLoadResponse {
        success: true,
        path: req.path,
    }))
}

// ============================================================================
// Network Handlers
// ============================================================================

/// Network ID response
#[derive(Serialize)]
struct NetworkIdResponse {
    #[serde(rename = "ID")]
    id: String,
    #[serde(rename = "Addresses")]
    addresses: Vec<String>,
}

async fn api_network_id(
    State(state): State<GatewayState>,
) -> Result<Json<NetworkIdResponse>, AppError> {
    let network = state
        .network
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("Network not enabled".to_string()))?;

    let (peer_id, listeners) = {
        let network = network.lock().await;
        (network.peer_id().to_string(), network.listeners())
    };

    let addresses = listeners
        .iter()
        .map(|addr| format!("{}/p2p/{}", addr, peer_id))
        .collect();

    Ok(Json(NetworkIdResponse {
        id: peer_id,
        addresses,
    }))
}

/// Swarm peers response
#[derive(Serialize)]
struct SwarmPeersResponse {
    #[serde(rename = "Peers")]
    peers: Vec<PeerEntry>,
}

#[derive(Serialize)]
struct PeerEntry {
    #[serde(rename = "Peer")]
    peer: String,
}

async fn api_swarm_peers(
    State(state): State<GatewayState>,
) -> Result<Json<SwarmPeersResponse>, AppError> {
    let network = state
        .network
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("Network not enabled".to_string()))?;

    let peers = {
        let network = network.lock().await;
        network.connected_peers()
    };

    let peer_entries: Vec<PeerEntry> = peers
        .into_iter()
        .map(|p| PeerEntry {
            peer: p.to_string(),
        })
        .collect();

    Ok(Json(SwarmPeersResponse {
        peers: peer_entries,
    }))
}

/// Swarm connect request
#[derive(Deserialize)]
struct SwarmConnectRequest {
    arg: String,
}

/// Swarm connect response
#[derive(Serialize)]
struct SwarmConnectResponse {
    #[serde(rename = "Strings")]
    strings: Vec<String>,
}

async fn api_swarm_connect(
    State(state): State<GatewayState>,
    Json(req): Json<SwarmConnectRequest>,
) -> Result<Json<SwarmConnectResponse>, AppError> {
    let network = state
        .network
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("Network not enabled".to_string()))?;

    let addr: ipfrs_network::libp2p::Multiaddr = req
        .arg
        .parse()
        .map_err(|e| AppError::Network(format!("Invalid multiaddr: {}", e)))?;

    {
        let mut network = network.lock().await;
        network
            .connect(addr.clone())
            .await
            .map_err(|e| AppError::Network(format!("Connect failed: {}", e)))?;
    }

    info!("Connected to peer: {}", req.arg);

    Ok(Json(SwarmConnectResponse {
        strings: vec![format!("connect {} success", req.arg)],
    }))
}

/// Swarm disconnect request
#[derive(Deserialize)]
struct SwarmDisconnectRequest {
    arg: String,
}

/// Swarm disconnect response
#[derive(Serialize)]
struct SwarmDisconnectResponse {
    #[serde(rename = "Strings")]
    strings: Vec<String>,
}

async fn api_swarm_disconnect(
    State(state): State<GatewayState>,
    Json(req): Json<SwarmDisconnectRequest>,
) -> Result<Json<SwarmDisconnectResponse>, AppError> {
    let network = state
        .network
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("Network not enabled".to_string()))?;

    let peer_id: ipfrs_network::libp2p::PeerId = req
        .arg
        .parse()
        .map_err(|e| AppError::Network(format!("Invalid peer ID: {}", e)))?;

    {
        let mut network = network.lock().await;
        network
            .disconnect(peer_id)
            .await
            .map_err(|e| AppError::Network(format!("Disconnect failed: {}", e)))?;
    }

    info!("Disconnected from peer: {}", req.arg);

    Ok(Json(SwarmDisconnectResponse {
        strings: vec![format!("disconnect {} success", req.arg)],
    }))
}

/// DHT findprovs request
#[derive(Deserialize)]
struct DhtFindprovsRequest {
    arg: String,
}

/// DHT findprovs response
#[derive(Serialize)]
struct DhtFindprovsResponse {
    #[serde(rename = "Responses")]
    responses: Vec<DhtProviderEntry>,
}

#[derive(Serialize)]
struct DhtProviderEntry {
    #[serde(rename = "ID")]
    id: String,
}

async fn api_dht_findprovs(
    State(state): State<GatewayState>,
    Json(req): Json<DhtFindprovsRequest>,
) -> Result<Json<DhtFindprovsResponse>, AppError> {
    let network = state
        .network
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("Network not enabled".to_string()))?;

    let cid: Cid = req
        .arg
        .parse()
        .map_err(|_| AppError::InvalidCid(req.arg.clone()))?;

    {
        let mut network = network.lock().await;
        network
            .find_providers(&cid)
            .await
            .map_err(|e| AppError::Network(format!("Find providers failed: {}", e)))?;
    }

    info!("Finding providers for: {}", req.arg);

    // Note: In a real implementation, we would wait for the query to complete
    // and return the actual providers. For now, return empty list as the query
    // is asynchronous and results come via events.
    Ok(Json(DhtFindprovsResponse { responses: vec![] }))
}

/// DHT provide request
#[derive(Deserialize)]
struct DhtProvideRequest {
    arg: String,
}

/// DHT provide response
#[derive(Serialize)]
struct DhtProvideResponse {
    #[serde(rename = "ID")]
    id: String,
}

async fn api_dht_provide(
    State(state): State<GatewayState>,
    Json(req): Json<DhtProvideRequest>,
) -> Result<Json<DhtProvideResponse>, AppError> {
    let network = state
        .network
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("Network not enabled".to_string()))?;

    let cid: Cid = req
        .arg
        .parse()
        .map_err(|_| AppError::InvalidCid(req.arg.clone()))?;

    {
        let mut network = network.lock().await;
        network
            .provide(&cid)
            .await
            .map_err(|e| AppError::Network(format!("Provide failed: {}", e)))?;
    }

    info!("Announcing content to DHT: {}", req.arg);

    Ok(Json(DhtProvideResponse { id: req.arg }))
}

// ============================================================================
// GraphQL Handlers
// ============================================================================

/// GraphQL query handler
async fn graphql_handler(
    State(state): State<GatewayState>,
    req: GraphQLRequest,
) -> Result<GraphQLResponse, AppError> {
    let schema = state
        .graphql_schema
        .as_ref()
        .ok_or_else(|| AppError::FeatureDisabled("GraphQL not enabled".to_string()))?;

    Ok(schema.execute(req.into_inner()).await.into())
}

/// GraphQL playground handler
async fn graphql_playground() -> impl IntoResponse {
    Html(playground_source(GraphQLPlaygroundConfig::new("/graphql")))
}

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
    use super::*;

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
            .unwrap()
            .to_str()
            .unwrap();
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
        assert!(prod.compression_config.enable_brotli);
        assert!(prod.compression_config.enable_deflate);

        // Test development preset
        let dev = GatewayConfig::development();
        assert_eq!(dev.listen_addr, "127.0.0.1:8080");
        assert!(dev.compression_config.enable_gzip);
        assert!(!dev.compression_config.enable_brotli);

        // Test testing preset
        let test = GatewayConfig::testing();
        assert_eq!(test.listen_addr, "127.0.0.1:0");
        assert!(!test.compression_config.enable_gzip);
        assert!(!test.compression_config.enable_brotli);
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
        assert!(config.compression_config.enable_brotli);
        assert!(config.compression_config.enable_deflate);
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
        assert!(config_with.compression_config.enable_brotli);
        assert!(config_with.compression_config.enable_deflate);

        let config_without = GatewayConfig::default().without_compression();
        assert!(!config_without.compression_config.enable_gzip);
        assert!(!config_without.compression_config.enable_brotli);
        assert!(!config_without.compression_config.enable_deflate);
    }
}
