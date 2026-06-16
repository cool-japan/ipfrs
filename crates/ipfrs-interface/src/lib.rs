//! IPFRS Interface - High-performance HTTP Gateway and API Layer
//!
//! This crate provides a comprehensive API layer for IPFRS (InterPlanetary File & Reasoning System),
//! offering multiple interfaces for accessing distributed content and computation.
//!
//! # Features
//!
//! ## HTTP Gateway
//! - **Kubo-compatible API** (`/api/v0/*`) - Full compatibility with IPFS Kubo clients
//! - **High-speed v1 API** (`/v1/*`) - Optimized endpoints with batch operations
//! - **Content gateway** (`/ipfs/{cid}`) - Direct content retrieval with range requests
//! - **Multi-range support** - Efficient sparse downloads with HTTP 206
//!
//! ## gRPC Interface
//! - **BlockService** - Raw block operations (Get, Put, Has, Delete, Batch)
//! - **DagService** - DAG operations (Get, Put, Resolve, Traverse)
//! - **FileService** - File operations (Add, Get, List, Pin)
//! - **TensorService** - Zero-copy tensor operations (Get, Slice, Stream)
//! - **Streaming RPCs** - Client, server, and bidirectional streaming
//! - **Interceptors** - Authentication, logging, metrics, rate limiting
//!
//! ## WebSocket Support
//! - **Real-time events** - Block additions, peer connections, DHT queries
//! - **Pub/sub system** - Topic-based event subscriptions
//! - **Connection management** - Automatic cleanup and heartbeat
//!
//! ## Advanced Features
//! - **Zero-copy tensor API** - Efficient ML model distribution via Safetensors
//! - **Streaming uploads/downloads** - Memory-efficient large file handling
//! - **Batch operations** - Parallel processing with atomic transactions
//! - **Flow control** - Adaptive window-based congestion control
//! - **Resume/cancel** - Robust transfer management
//!
//! ## Security & Performance
//! - **Authentication** - JWT tokens, API keys, and OAuth2 (Authorization Code, Client Credentials, PKCE)
//! - **Rate limiting** - Token bucket algorithm with per-client limits
//! - **CORS** - Configurable cross-origin resource sharing
//! - **Compression** - Gzip, Brotli, and Deflate with tunable levels
//! - **HTTP caching** - ETag and Cache-Control for CDN optimization
//! - **TLS/HTTPS** - Production-ready SSL/TLS support
//!
//! # Quick Start
//!
//! ```rust,ignore
//! // Example usage (see examples/server.rs for a complete working example)
//! use ipfrs_interface::GatewayConfig;
//! use ipfrs_storage::blockstore::BlockStoreConfig;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create gateway with default configuration
//!     let storage_config = BlockStoreConfig::default();
//!     let mut gateway_config = GatewayConfig::default();
//!     gateway_config.storage_config = storage_config;
//!     gateway_config.listen_addr = "127.0.0.1:8080".to_string();
//!
//!     // See the examples directory for complete implementation
//!     Ok(())
//! }
//! ```
//!
//! # Examples
//!
//! ## Upload a file (Kubo v0 API)
//! ```bash
//! curl -X POST -F "file=@myfile.txt" http://localhost:8080/api/v0/add
//! ```
//!
//! ## Download via gateway
//! ```bash
//! curl http://localhost:8080/ipfs/<CID>
//! ```
//!
//! ## Batch block operations (v1 API)
//! ```bash
//! curl -X POST -H "Content-Type: application/json" \
//!   -d '{"cids":["<CID1>","<CID2>"]}' \
//!   http://localhost:8080/v1/block/batch/get
//! ```
//!
//! ## Get tensor slice (zero-copy)
//! ```bash
//! curl "http://localhost:8080/v1/tensor/<CID>?slice=0:10,5:15"
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐
//! │  HTTP Clients   │ (curl, browsers, IPFS clients)
//! └────────┬────────┘
//!          │
//! ┌────────▼────────┐
//! │  Axum Router    │ (HTTP/HTTPS)
//! ├─────────────────┤
//! │  Middleware     │ (CORS, Auth, Rate Limit, Compression)
//! ├─────────────────┤
//! │  API Handlers   │ (v0, v1, Gateway, WebSocket)
//! └────────┬────────┘
//!          │
//! ┌────────▼────────┐
//! │  gRPC Server    │ (Tonic)
//! ├─────────────────┤
//! │  Interceptors   │ (Auth, Logging, Metrics)
//! ├─────────────────┤
//! │  Services       │ (Block, DAG, File, Tensor)
//! └────────┬────────┘
//!          │
//! ┌────────▼────────┐
//! │  Core Layer     │ (ipfrs-core, ipfrs-storage, etc.)
//! └─────────────────┘
//! ```
//!
//! # Performance
//!
//! Target performance characteristics:
//! - Request latency: < 10ms (simple GET)
//! - Throughput: > 1GB/s (range requests)
//! - Concurrent connections: 10,000+
//! - Memory per connection: < 100KB
//!
//! # See Also
//!
//! - [`Gateway`] - Main HTTP gateway implementation
//! - [`middleware`] - CORS, rate limiting, and compression
//! - [`streaming`] - Streaming operations and flow control
//! - [`websocket`] - WebSocket real-time events

pub mod arrow;
pub mod auth;
pub mod auth_handlers;
pub mod backpressure;
pub mod binary_protocol;
pub mod ffi;
pub mod gateway;
pub mod gradient_sync;
pub mod graphql;
// The `grpc` module contains hand-written gRPC service implementations that
// depend on tonic-generated protobuf code produced by `build.rs`.  Because
// tonic-build 0.14 service generation requires proto files to be present at
// compile time and the feature-gated `grpc` build is still being stabilised,
// the module is compiled only when the `grpc` Cargo feature is enabled.
// See `crates/ipfrs-interface/build.rs` for the build configuration.
#[cfg(feature = "grpc")]
pub mod grpc;
pub mod metrics;
pub mod metrics_middleware;
pub mod middleware;
pub mod mmap;
pub mod oauth2;
pub mod python;
pub mod safetensors;
pub mod streaming;
pub mod tensor;
pub mod tls;
pub mod websocket;
pub mod zerocopy;

pub use backpressure::{
    BackpressureConfig, BackpressureController, BackpressureError, BackpressurePermit,
    BackpressureStream,
};
pub use binary_protocol::{
    BinaryMessage, ErrorResponse, GetBlockRequest, HasBlockRequest, MessageType, ProtocolError,
    PutBlockRequest, SuccessResponse, PROTOCOL_VERSION,
};
pub use gateway::{Gateway, GatewayConfig};
pub use gradient_sync::{GradientChunkResponse, GradientSyncRequest, GradientSyncService};
pub use graphql::{create_schema, IpfrsSchema};
pub use middleware::{
    cors_middleware, rate_limit_middleware, CacheConfig, CompressionConfig, CompressionLevel,
    CorsConfig, CorsState, RateLimitConfig, RateLimitState,
};
pub use mmap::{MmapCache, MmapConfig, MmapError, MmapFile};
pub use oauth2::{
    AccessToken, AuthorizationCode, CodeChallengeMethod, ErrorResponse as OAuth2ErrorResponse,
    GrantType, OAuth2Client, OAuth2ProviderConfig, OAuth2Server, RefreshToken, ResponseType, Scope,
    TokenResponse, TokenType,
};
pub use safetensors::{SafetensorsFile, SafetensorsHeader, TensorData, TensorInfo};
pub use streaming::{
    ConcurrencyConfig, FlowControlConfig, FlowController, OperationState, OperationStatus,
    OperationType, ProgressEvent, ProgressTracker, ResumeToken,
};
pub use tensor::{TensorLayout, TensorMetadata, TensorSlice};
pub use websocket::{ws_handler, RealtimeEvent, SubscriptionManager, WsMessage, WsState};
pub use zerocopy::ZeroCopyBuffer;

// gRPC exports — available only when the `grpc` feature is enabled.
// The feature gate keeps the default build 100% pure-Rust without proto
// compilation; enable `grpc` to get full tonic-backed service types.
#[cfg(feature = "grpc")]
pub use grpc::{
    backpressure_support, AuthInterceptor, BlockServiceImpl, BlockServiceServer,
    ChainedInterceptor, DagServiceImpl, DagServiceServer, FileServiceImpl, FileServiceServer,
    GrpcServiceConfig, LoggingInterceptor, MetricsInterceptor, RateLimitInterceptor,
    TensorServiceImpl, TensorServiceServer,
};
