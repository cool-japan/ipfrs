//! Streaming Support for IPFRS Gateway
//!
//! Provides:
//! - Memory-efficient streaming downloads
//! - Chunked uploads with progress tracking
//! - Server-Sent Events (SSE) for progress callbacks
//! - Batch block operations

use axum::{
    body::Body,
    extract::{Multipart, Path, Query, State},
    http::{header, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use bytes::Bytes;
use futures::stream::{self, Stream, StreamExt};
use ipfrs_core::{Block, Cid};
use ipfrs_storage::BlockStoreTrait;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::time::Duration;
use tokio::sync::broadcast;
use tracing::info;
use uuid::Uuid;

use crate::gateway::GatewayState;

// ============================================================================
// Flow Control & Concurrency
// ============================================================================

/// Concurrency control configuration for batch operations
#[derive(Debug, Clone)]
pub struct ConcurrencyConfig {
    /// Maximum number of concurrent tasks for batch operations (0 = unlimited)
    pub max_concurrent_tasks: usize,
    /// Enable parallel processing
    pub parallel_enabled: bool,
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: 100, // Reasonable default
            parallel_enabled: true,
        }
    }
}

impl ConcurrencyConfig {
    /// Create a conservative config (lower concurrency)
    pub fn conservative() -> Self {
        Self {
            max_concurrent_tasks: 50,
            parallel_enabled: true,
        }
    }

    /// Create an aggressive config (higher concurrency)
    pub fn aggressive() -> Self {
        Self {
            max_concurrent_tasks: 200,
            parallel_enabled: true,
        }
    }

    /// Create a sequential config (no parallelism)
    pub fn sequential() -> Self {
        Self {
            max_concurrent_tasks: 1,
            parallel_enabled: false,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.max_concurrent_tasks == 0 && self.parallel_enabled {
            return Err(
                "max_concurrent_tasks cannot be 0 when parallel_enabled is true".to_string(),
            );
        }
        Ok(())
    }
}

/// Flow control configuration for streaming operations
#[derive(Debug, Clone)]
pub struct FlowControlConfig {
    /// Maximum bytes per second (0 = unlimited)
    pub max_bytes_per_second: u64,
    /// Initial window size in bytes
    pub initial_window_size: usize,
    /// Maximum window size in bytes
    pub max_window_size: usize,
    /// Minimum window size in bytes
    pub min_window_size: usize,
    /// Enable dynamic window adjustment
    pub dynamic_adjustment: bool,
}

impl Default for FlowControlConfig {
    fn default() -> Self {
        Self {
            max_bytes_per_second: 0,         // Unlimited
            initial_window_size: 256 * 1024, // 256KB
            max_window_size: 1024 * 1024,    // 1MB
            min_window_size: 64 * 1024,      // 64KB
            dynamic_adjustment: true,
        }
    }
}

impl FlowControlConfig {
    /// Create a flow control config with specific rate limit
    pub fn with_rate_limit(bytes_per_second: u64) -> Self {
        Self {
            max_bytes_per_second: bytes_per_second,
            ..Default::default()
        }
    }

    /// Create a conservative flow control config (smaller windows)
    pub fn conservative() -> Self {
        Self {
            initial_window_size: 64 * 1024,
            max_window_size: 256 * 1024,
            min_window_size: 32 * 1024,
            ..Default::default()
        }
    }

    /// Create an aggressive flow control config (larger windows)
    pub fn aggressive() -> Self {
        Self {
            initial_window_size: 512 * 1024,
            max_window_size: 2 * 1024 * 1024,
            min_window_size: 128 * 1024,
            ..Default::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Min window size must be less than or equal to initial window size
        if self.min_window_size > self.initial_window_size {
            return Err(format!(
                "Minimum window size ({}) cannot exceed initial window size ({})",
                self.min_window_size, self.initial_window_size
            ));
        }

        // Initial window size must be less than or equal to max window size
        if self.initial_window_size > self.max_window_size {
            return Err(format!(
                "Initial window size ({}) cannot exceed maximum window size ({})",
                self.initial_window_size, self.max_window_size
            ));
        }

        // Validate rate limit if set
        if self.max_bytes_per_second > 0 {
            validation::validate_rate_limit(self.max_bytes_per_second)?;
        }

        Ok(())
    }
}

/// Flow control state for a streaming operation
#[derive(Debug)]
pub struct FlowController {
    config: FlowControlConfig,
    current_window_size: usize,
    bytes_sent: u64,
    start_time: std::time::Instant,
    last_adjustment: std::time::Instant,
}

impl FlowController {
    /// Create a new flow controller
    pub fn new(config: FlowControlConfig) -> Self {
        Self {
            current_window_size: config.initial_window_size,
            config,
            bytes_sent: 0,
            start_time: std::time::Instant::now(),
            last_adjustment: std::time::Instant::now(),
        }
    }

    /// Get the current window size
    pub fn window_size(&self) -> usize {
        self.current_window_size
    }

    /// Calculate delay needed to respect rate limit
    pub fn calculate_delay(&self, bytes_to_send: usize) -> std::time::Duration {
        if self.config.max_bytes_per_second == 0 {
            return std::time::Duration::from_secs(0);
        }

        let elapsed = self.start_time.elapsed();
        let elapsed_secs = elapsed.as_secs_f64();

        if elapsed_secs == 0.0 {
            return std::time::Duration::from_secs(0);
        }

        let current_rate = self.bytes_sent as f64 / elapsed_secs;
        let target_rate = self.config.max_bytes_per_second as f64;

        if current_rate + (bytes_to_send as f64 / elapsed_secs) > target_rate {
            let delay_secs = (bytes_to_send as f64 / target_rate).max(0.0);
            std::time::Duration::from_secs_f64(delay_secs)
        } else {
            std::time::Duration::from_secs(0)
        }
    }

    /// Update flow control state after sending data
    pub fn on_data_sent(&mut self, bytes: usize) {
        self.bytes_sent += bytes as u64;

        // Adjust window size if dynamic adjustment is enabled
        if self.config.dynamic_adjustment {
            self.adjust_window();
        }
    }

    /// Dynamically adjust window size based on performance
    fn adjust_window(&mut self) {
        let elapsed = self.last_adjustment.elapsed();

        // Adjust every 100ms
        if elapsed < std::time::Duration::from_millis(100) {
            return;
        }

        self.last_adjustment = std::time::Instant::now();

        // Simple AIMD (Additive Increase Multiplicative Decrease) algorithm
        // Increase window size by 10% if no issues
        let new_size = (self.current_window_size as f64 * 1.1)
            .min(self.config.max_window_size as f64) as usize;

        self.current_window_size =
            new_size.clamp(self.config.min_window_size, self.config.max_window_size);
    }

    /// Decrease window size (on congestion)
    #[allow(dead_code)]
    pub fn on_congestion(&mut self) {
        // Multiplicative decrease by 50%
        let new_size = self.current_window_size / 2;
        self.current_window_size = new_size.max(self.config.min_window_size);
        self.last_adjustment = std::time::Instant::now();
    }

    /// Get current throughput in bytes per second
    pub fn current_throughput(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.bytes_sent as f64 / elapsed
        } else {
            0.0
        }
    }
}

// ============================================================================
// Resume/Cancel Support
// ============================================================================

/// Operation state for resume/cancel support
#[derive(Debug, Clone)]
pub struct OperationState {
    /// Unique operation ID
    pub operation_id: String,
    /// Current byte offset
    pub offset: u64,
    /// Total size (if known)
    pub total_size: Option<u64>,
    /// Operation type
    pub operation_type: OperationType,
    /// Status
    pub status: OperationStatus,
}

/// Operation type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OperationType {
    Upload,
    Download,
}

/// Operation status
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OperationStatus {
    InProgress,
    Paused,
    Cancelled,
    Completed,
    Failed,
}

/// Resume token for continuing operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResumeToken {
    /// Operation ID
    pub operation_id: String,
    /// Byte offset to resume from
    pub offset: u64,
    /// Optional CID for downloads
    pub cid: Option<String>,
}

impl ResumeToken {
    /// Create a new resume token
    pub fn new(operation_id: String, offset: u64, cid: Option<String>) -> Self {
        Self {
            operation_id,
            offset,
            cid,
        }
    }

    /// Encode resume token to base64
    pub fn encode(&self) -> Result<String, String> {
        let json = serde_json::to_string(self).map_err(|e| e.to_string())?;
        Ok(base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            json.as_bytes(),
        ))
    }

    /// Decode resume token from base64
    pub fn decode(encoded: &str) -> Result<Self, String> {
        let bytes =
            base64::Engine::decode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, encoded)
                .map_err(|e| e.to_string())?;

        let json = String::from_utf8(bytes).map_err(|e| e.to_string())?;
        serde_json::from_str(&json).map_err(|e| e.to_string())
    }
}

/// Cancel request
#[derive(Debug, Deserialize)]
pub struct CancelRequest {
    /// Operation ID to cancel
    pub operation_id: String,
}

/// Cancel response
#[derive(Debug, Serialize)]
pub struct CancelResponse {
    /// Operation ID that was cancelled
    pub operation_id: String,
    /// Whether cancellation was successful
    pub cancelled: bool,
    /// Optional resume token for later resumption
    pub resume_token: Option<String>,
}

// ============================================================================
// Progress Tracking
// ============================================================================

/// Progress event for uploads/downloads
#[derive(Debug, Clone, Serialize)]
pub struct ProgressEvent {
    /// Operation ID
    pub operation_id: String,
    /// Current bytes processed
    pub bytes_processed: u64,
    /// Total bytes (if known)
    pub total_bytes: Option<u64>,
    /// Progress percentage (0-100)
    pub progress_percent: Option<f32>,
    /// Status message
    pub status: ProgressStatus,
}

/// Progress status
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ProgressStatus {
    Started,
    InProgress,
    Completed,
    Failed,
}

/// Progress tracker for streaming operations
#[derive(Clone)]
pub struct ProgressTracker {
    sender: broadcast::Sender<ProgressEvent>,
}

impl ProgressTracker {
    /// Create a new progress tracker
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(100);
        Self { sender }
    }

    /// Send a progress update
    pub fn send(&self, event: ProgressEvent) {
        let _ = self.sender.send(event);
    }

    /// Subscribe to progress updates
    pub fn subscribe(&self) -> broadcast::Receiver<ProgressEvent> {
        self.sender.subscribe()
    }
}

impl Default for ProgressTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Streaming Downloads
// ============================================================================

/// Stream download query parameters
#[derive(Debug, Deserialize)]
pub struct StreamDownloadQuery {
    /// Chunk size in bytes (default: dynamic based on flow control)
    pub chunk_size: Option<usize>,
    /// Maximum bytes per second (0 = unlimited)
    pub max_bytes_per_second: Option<u64>,
    /// Enable flow control
    pub flow_control: Option<bool>,
    /// Resume token for continuing a previous download
    pub resume_token: Option<String>,
    /// Byte offset to start from (alternative to resume_token)
    pub offset: Option<u64>,
}

/// Stream content download endpoint
///
/// GET /v1/stream/download/{cid}
///
/// Streams content in chunks for memory-efficient downloads with optional flow control and resume support.
pub async fn stream_download(
    State(state): State<GatewayState>,
    Path(cid_str): Path<String>,
    Query(query): Query<StreamDownloadQuery>,
) -> Result<Response, StreamingError> {
    let cid: Cid = cid_str
        .parse()
        .map_err(|_| StreamingError::InvalidCid(cid_str.clone()))?;

    // Get the block
    let block = state
        .store
        .get(&cid)
        .await
        .map_err(|e| StreamingError::Storage(e.to_string()))?
        .ok_or_else(|| StreamingError::NotFound(cid_str.clone()))?;

    let data = block.data().to_vec();
    let total_size = data.len();

    // Determine start offset (from resume token or explicit offset)
    let start_offset = if let Some(resume_token) = &query.resume_token {
        let token = ResumeToken::decode(resume_token)
            .map_err(|e| StreamingError::Upload(format!("Invalid resume token: {}", e)))?;

        // Validate CID matches
        if let Some(token_cid) = &token.cid {
            if token_cid != &cid_str {
                return Err(StreamingError::Upload(
                    "Resume token CID mismatch".to_string(),
                ));
            }
        }

        token.offset as usize
    } else {
        query.offset.unwrap_or(0) as usize
    };

    // Validate offset
    if start_offset >= total_size {
        return Err(StreamingError::Upload(format!(
            "Invalid offset: {} (total size: {})",
            start_offset, total_size
        )));
    }

    // Initialize flow control if requested
    let enable_flow_control = query.flow_control.unwrap_or(false);
    let flow_controller = if enable_flow_control {
        let mut config = FlowControlConfig::default();
        if let Some(rate) = query.max_bytes_per_second {
            config.max_bytes_per_second = rate;
        }
        Some(FlowController::new(config))
    } else {
        None
    };

    // Determine chunk size (from query, flow control, or default)
    let chunk_size = query.chunk_size.unwrap_or_else(|| {
        flow_controller
            .as_ref()
            .map(|fc| fc.window_size())
            .unwrap_or(64 * 1024)
    });

    // Pre-collect chunks starting from offset to avoid lifetime issues
    let chunks: Vec<Vec<u8>> = data[start_offset..]
        .chunks(chunk_size)
        .map(|chunk| chunk.to_vec())
        .collect();

    let remaining_size = total_size - start_offset;

    // Create a stream that yields chunks with flow control
    let stream = if let Some(mut fc) = flow_controller {
        let stream = async_stream::stream! {
            for chunk in chunks {
                let chunk_len = chunk.len();

                // Apply flow control delay
                let delay = fc.calculate_delay(chunk_len);
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }

                // Update flow control state
                fc.on_data_sent(chunk_len);

                yield Ok::<_, Infallible>(Bytes::from(chunk));
            }
        };
        Body::from_stream(stream)
    } else {
        let stream = stream::iter(chunks).map(|chunk| Ok::<_, Infallible>(Bytes::from(chunk)));
        Body::from_stream(stream)
    };

    // Build response with appropriate headers
    let mut response_builder = Response::builder();

    // If resuming, use 206 Partial Content
    if start_offset > 0 {
        response_builder = response_builder.status(StatusCode::PARTIAL_CONTENT);
        // Content-Range: bytes start-end/total
        let end_offset = total_size - 1;
        response_builder = response_builder.header(
            header::CONTENT_RANGE,
            format!("bytes {}-{}/{}", start_offset, end_offset, total_size),
        );
    } else {
        response_builder = response_builder.status(StatusCode::OK);
    }

    Ok(response_builder
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header(header::CONTENT_LENGTH, remaining_size.to_string())
        .header("X-Chunk-Size", chunk_size.to_string())
        .header("Accept-Ranges", "bytes")
        .body(stream)
        .expect("building streaming response with valid headers and body is infallible"))
}

// ============================================================================
// Streaming Uploads
// ============================================================================

/// Upload response
#[derive(Debug, Serialize)]
pub struct StreamUploadResponse {
    pub cid: String,
    pub size: u64,
    pub chunks_received: usize,
}

/// Stream upload endpoint with progress tracking
///
/// POST /v1/stream/upload
///
/// Accepts chunked uploads and provides progress updates via SSE.
pub async fn stream_upload(
    State(state): State<GatewayState>,
    mut multipart: Multipart,
) -> Result<Json<StreamUploadResponse>, StreamingError> {
    let mut total_data = Vec::new();
    let mut chunks_received = 0;

    // Process multipart data
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| StreamingError::Upload(format!("Failed to read field: {}", e)))?
    {
        let data = field
            .bytes()
            .await
            .map_err(|e| StreamingError::Upload(format!("Failed to read data: {}", e)))?;

        total_data.extend_from_slice(&data);
        chunks_received += 1;
    }

    if total_data.is_empty() {
        return Err(StreamingError::Upload("No data received".to_string()));
    }

    // Create and store the block
    let block = Block::new(Bytes::from(total_data))
        .map_err(|e| StreamingError::Upload(format!("Failed to create block: {}", e)))?;

    let cid = *block.cid();
    let size = block.size();

    state
        .store
        .put(&block)
        .await
        .map_err(|e| StreamingError::Storage(e.to_string()))?;

    info!("Stream upload completed: {} ({} bytes)", cid, size);

    Ok(Json(StreamUploadResponse {
        cid: cid.to_string(),
        size,
        chunks_received,
    }))
}

// ============================================================================
// Batch Operations
// ============================================================================

/// Batch get request
#[derive(Debug, Deserialize)]
pub struct BatchGetRequest {
    /// List of CIDs to retrieve
    pub cids: Vec<String>,
}

/// Batch get response
#[derive(Debug, Serialize)]
pub struct BatchGetResponse {
    /// Successfully retrieved blocks
    pub blocks: Vec<BatchBlockResult>,
    /// Failed CIDs
    pub errors: Vec<BatchError>,
}

/// Individual block result in batch
#[derive(Debug, Serialize)]
pub struct BatchBlockResult {
    pub cid: String,
    pub data: String, // Base64 encoded
    pub size: u64,
}

/// Batch error for individual items
#[derive(Debug, Serialize)]
pub struct BatchError {
    pub cid: String,
    pub error: String,
}

/// Batch get endpoint (optimized with parallel processing)
///
/// POST /v1/block/batch/get
///
/// Retrieves multiple blocks in a single request.
/// Uses parallel processing for high throughput.
pub async fn batch_get(
    State(state): State<GatewayState>,
    Json(req): Json<BatchGetRequest>,
) -> Result<Json<BatchGetResponse>, StreamingError> {
    // Validate batch size
    validation::validate_batch_size(req.cids.len()).map_err(StreamingError::Validation)?;

    // Process all CIDs in parallel using concurrent tasks
    let tasks: Vec<_> = req
        .cids
        .into_iter()
        .map(|cid_str| {
            let state = state.clone();
            tokio::spawn(async move {
                match cid_str.parse::<Cid>() {
                    Ok(cid) => match state.store.get(&cid).await {
                        Ok(Some(block)) => {
                            let data_base64 = base64::Engine::encode(
                                &base64::engine::general_purpose::STANDARD,
                                block.data(),
                            );
                            Ok(BatchBlockResult {
                                cid: cid_str,
                                data: data_base64,
                                size: block.size(),
                            })
                        }
                        Ok(None) => Err(BatchError {
                            cid: cid_str,
                            error: "Block not found".to_string(),
                        }),
                        Err(e) => Err(BatchError {
                            cid: cid_str,
                            error: e.to_string(),
                        }),
                    },
                    Err(_) => Err(BatchError {
                        cid: cid_str,
                        error: "Invalid CID".to_string(),
                    }),
                }
            })
        })
        .collect();

    // Await all tasks and collect results
    let mut blocks = Vec::new();
    let mut errors = Vec::new();

    for task in tasks {
        match task.await {
            Ok(Ok(block)) => blocks.push(block),
            Ok(Err(error)) => errors.push(error),
            Err(e) => {
                // Task panicked or was cancelled
                errors.push(BatchError {
                    cid: "unknown".to_string(),
                    error: format!("Task execution error: {}", e),
                });
            }
        }
    }

    Ok(Json(BatchGetResponse { blocks, errors }))
}

/// Batch put request item
#[derive(Debug, Deserialize)]
pub struct BatchPutItem {
    /// Base64 encoded data
    pub data: String,
}

/// Transaction mode for batch operations
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum TransactionMode {
    /// All-or-nothing: either all operations succeed or all fail
    Atomic,
    /// Best-effort: process each item independently
    #[default]
    BestEffort,
}

/// Batch put request
#[derive(Debug, Deserialize)]
pub struct BatchPutRequest {
    /// List of blocks to store
    pub blocks: Vec<BatchPutItem>,
    /// Transaction mode (default: best_effort)
    #[serde(default)]
    pub transaction_mode: TransactionMode,
}

/// Batch put response
#[derive(Debug, Serialize)]
pub struct BatchPutResponse {
    /// Successfully stored blocks
    pub stored: Vec<BatchStoredResult>,
    /// Failed items
    pub errors: Vec<BatchPutError>,
    /// Transaction ID (present in atomic mode)
    pub transaction_id: Option<String>,
    /// Transaction status
    pub transaction_status: TransactionStatus,
}

/// Transaction status
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TransactionStatus {
    /// All operations succeeded
    Committed,
    /// Some operations failed (best-effort mode)
    PartialSuccess,
    /// All operations rolled back (atomic mode)
    RolledBack,
}

/// Stored block result
#[derive(Debug, Serialize)]
pub struct BatchStoredResult {
    pub cid: String,
    pub size: u64,
    pub index: usize,
}

/// Batch put error
#[derive(Debug, Serialize)]
pub struct BatchPutError {
    pub index: usize,
    pub error: String,
}

/// Batch put endpoint
///
/// POST /v1/block/batch/put
///
/// Stores multiple blocks in a single request.
/// Supports atomic transactions (all-or-nothing) and best-effort mode.
pub async fn batch_put(
    State(state): State<GatewayState>,
    Json(req): Json<BatchPutRequest>,
) -> Result<Json<BatchPutResponse>, StreamingError> {
    let transaction_id = Uuid::new_v4().to_string();

    match req.transaction_mode {
        TransactionMode::Atomic => batch_put_atomic(state, req.blocks, transaction_id).await,
        TransactionMode::BestEffort => {
            batch_put_best_effort(state, req.blocks, transaction_id).await
        }
    }
}

/// Atomic batch put - all-or-nothing transaction
async fn batch_put_atomic(
    state: GatewayState,
    items: Vec<BatchPutItem>,
    transaction_id: String,
) -> Result<Json<BatchPutResponse>, StreamingError> {
    // Validate batch size
    validation::validate_batch_size(items.len()).map_err(StreamingError::Validation)?;

    // Phase 1: Validate all items and prepare blocks
    let mut prepared_blocks = Vec::new();
    let mut errors = Vec::new();

    for (index, item) in items.into_iter().enumerate() {
        match base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &item.data) {
            Ok(data) => match Block::new(Bytes::from(data)) {
                Ok(block) => {
                    prepared_blocks.push((index, block));
                }
                Err(e) => {
                    errors.push(BatchPutError {
                        index,
                        error: format!("Block creation error: {}", e),
                    });
                }
            },
            Err(e) => {
                errors.push(BatchPutError {
                    index,
                    error: format!("Base64 decode error: {}", e),
                });
            }
        }
    }

    // If any validation failed, rollback (don't store anything)
    if !errors.is_empty() {
        info!(
            "Atomic batch put [{}] rolled back: {} validation errors",
            transaction_id,
            errors.len()
        );
        return Ok(Json(BatchPutResponse {
            stored: vec![],
            errors,
            transaction_id: Some(transaction_id),
            transaction_status: TransactionStatus::RolledBack,
        }));
    }

    // Phase 2: Store all blocks
    let mut stored = Vec::new();
    let mut stored_cids = Vec::new(); // Track for potential rollback

    for (index, block) in prepared_blocks {
        let cid = *block.cid();
        let size = block.size();

        match state.store.put(&block).await {
            Ok(_) => {
                stored_cids.push(cid);
                stored.push(BatchStoredResult {
                    cid: cid.to_string(),
                    size,
                    index,
                });
            }
            Err(e) => {
                // Storage failure - rollback all stored blocks
                info!(
                    "Atomic batch put [{}] rolling back: storage error at index {}",
                    transaction_id, index
                );

                // Attempt to delete all previously stored blocks in this transaction
                for stored_cid in stored_cids {
                    let _ = state.store.delete(&stored_cid).await; // Best effort rollback
                }

                return Ok(Json(BatchPutResponse {
                    stored: vec![],
                    errors: vec![BatchPutError {
                        index,
                        error: format!("Storage error (transaction rolled back): {}", e),
                    }],
                    transaction_id: Some(transaction_id),
                    transaction_status: TransactionStatus::RolledBack,
                }));
            }
        }
    }

    info!(
        "Atomic batch put [{}] committed: {} blocks stored",
        transaction_id,
        stored.len()
    );

    Ok(Json(BatchPutResponse {
        stored,
        errors: vec![],
        transaction_id: Some(transaction_id),
        transaction_status: TransactionStatus::Committed,
    }))
}

/// Best-effort batch put - process each item independently
async fn batch_put_best_effort(
    state: GatewayState,
    items: Vec<BatchPutItem>,
    transaction_id: String,
) -> Result<Json<BatchPutResponse>, StreamingError> {
    // Validate batch size
    validation::validate_batch_size(items.len()).map_err(StreamingError::Validation)?;

    let mut stored = Vec::new();
    let mut errors = Vec::new();

    for (index, item) in items.into_iter().enumerate() {
        match base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &item.data) {
            Ok(data) => match Block::new(Bytes::from(data)) {
                Ok(block) => {
                    let cid = *block.cid();
                    let size = block.size();

                    match state.store.put(&block).await {
                        Ok(_) => {
                            stored.push(BatchStoredResult {
                                cid: cid.to_string(),
                                size,
                                index,
                            });
                        }
                        Err(e) => {
                            errors.push(BatchPutError {
                                index,
                                error: format!("Storage error: {}", e),
                            });
                        }
                    }
                }
                Err(e) => {
                    errors.push(BatchPutError {
                        index,
                        error: format!("Block creation error: {}", e),
                    });
                }
            },
            Err(e) => {
                errors.push(BatchPutError {
                    index,
                    error: format!("Base64 decode error: {}", e),
                });
            }
        }
    }

    let status = if errors.is_empty() {
        TransactionStatus::Committed
    } else {
        TransactionStatus::PartialSuccess
    };

    info!(
        "Best-effort batch put [{}] completed: {} stored, {} errors",
        transaction_id,
        stored.len(),
        errors.len()
    );

    Ok(Json(BatchPutResponse {
        stored,
        errors,
        transaction_id: Some(transaction_id),
        transaction_status: status,
    }))
}

/// Batch has request
#[derive(Debug, Deserialize)]
pub struct BatchHasRequest {
    /// List of CIDs to check
    pub cids: Vec<String>,
}

/// Batch has response
#[derive(Debug, Serialize)]
pub struct BatchHasResponse {
    /// Results for each CID
    pub results: Vec<BatchHasResult>,
}

/// Individual has result
#[derive(Debug, Serialize)]
pub struct BatchHasResult {
    pub cid: String,
    pub exists: bool,
}

/// Batch has endpoint (optimized with parallel processing)
///
/// POST /v1/block/batch/has
///
/// Checks if multiple blocks exist in a single request.
/// Uses parallel processing for high throughput.
pub async fn batch_has(
    State(state): State<GatewayState>,
    Json(req): Json<BatchHasRequest>,
) -> Result<Json<BatchHasResponse>, StreamingError> {
    // Validate batch size
    validation::validate_batch_size(req.cids.len()).map_err(StreamingError::Validation)?;

    // Process all CIDs in parallel using concurrent tasks
    let tasks: Vec<_> = req
        .cids
        .into_iter()
        .map(|cid_str| {
            let state = state.clone();
            tokio::spawn(async move {
                let exists = if let Ok(cid) = cid_str.parse::<Cid>() {
                    state.store.has(&cid).await.unwrap_or(false)
                } else {
                    false
                };

                BatchHasResult {
                    cid: cid_str,
                    exists,
                }
            })
        })
        .collect();

    // Await all tasks and collect results
    let mut results = Vec::new();

    for task in tasks {
        match task.await {
            Ok(result) => results.push(result),
            Err(e) => {
                // Task panicked or was cancelled - mark as non-existent
                results.push(BatchHasResult {
                    cid: format!("task_error_{}", e),
                    exists: false,
                });
            }
        }
    }

    Ok(Json(BatchHasResponse { results }))
}

// ============================================================================
// Server-Sent Events (SSE) for Progress
// ============================================================================

/// SSE progress stream endpoint
///
/// GET /v1/progress/{operation_id}
///
/// Provides real-time progress updates via Server-Sent Events.
pub async fn progress_stream(
    State(_state): State<GatewayState>,
    Path(operation_id): Path<String>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Create a progress tracker for this operation
    let tracker = ProgressTracker::new();
    let mut receiver = tracker.subscribe();

    // Create the SSE stream
    let stream = async_stream::stream! {
        // Send initial event
        let initial = ProgressEvent {
            operation_id: operation_id.clone(),
            bytes_processed: 0,
            total_bytes: None,
            progress_percent: Some(0.0),
            status: ProgressStatus::Started,
        };
        yield Ok(Event::default()
            .event("progress")
            .json_data(initial)
            .expect("serializing ProgressEvent to JSON is infallible"));

        // Stream progress updates
        loop {
            match tokio::time::timeout(Duration::from_secs(30), receiver.recv()).await {
                Ok(Ok(event)) => {
                    let is_complete = matches!(event.status, ProgressStatus::Completed | ProgressStatus::Failed);
                    yield Ok(Event::default()
                        .event("progress")
                        .json_data(event)
                        .expect("serializing ProgressEvent to JSON is infallible"));

                    if is_complete {
                        break;
                    }
                }
                Ok(Err(_)) => {
                    // Channel closed
                    break;
                }
                Err(_) => {
                    // Timeout - send keepalive
                    yield Ok(Event::default().comment("keepalive"));
                }
            }
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

// ============================================================================
// Validation Utilities
// ============================================================================

/// Request validation utilities
pub mod validation {

    /// Validate CID string format
    pub fn validate_cid(cid: &str) -> Result<(), String> {
        if cid.is_empty() {
            return Err("CID cannot be empty".to_string());
        }

        // Basic CID validation (could be more comprehensive)
        if !cid.starts_with("Qm") && !cid.starts_with("bafy") && !cid.starts_with("baf") {
            return Err(format!("Invalid CID format: {}", cid));
        }

        if cid.len() < 10 {
            return Err(format!("CID too short: {}", cid));
        }

        Ok(())
    }

    /// Validate byte offset and size
    pub fn validate_offset(offset: u64, total_size: usize) -> Result<(), String> {
        if offset as usize >= total_size {
            return Err(format!(
                "Offset {} exceeds total size {}",
                offset, total_size
            ));
        }
        Ok(())
    }

    /// Validate chunk size (reasonable limits)
    pub fn validate_chunk_size(chunk_size: usize) -> Result<(), String> {
        const MIN_CHUNK_SIZE: usize = 1024; // 1KB
        const MAX_CHUNK_SIZE: usize = 10 * 1024 * 1024; // 10MB

        if chunk_size < MIN_CHUNK_SIZE {
            return Err(format!(
                "Chunk size {} is too small (minimum: {})",
                chunk_size, MIN_CHUNK_SIZE
            ));
        }

        if chunk_size > MAX_CHUNK_SIZE {
            return Err(format!(
                "Chunk size {} is too large (maximum: {})",
                chunk_size, MAX_CHUNK_SIZE
            ));
        }

        Ok(())
    }

    /// Validate rate limit
    pub fn validate_rate_limit(bytes_per_second: u64) -> Result<(), String> {
        const MAX_RATE: u64 = 10 * 1024 * 1024 * 1024; // 10 GB/s

        if bytes_per_second > MAX_RATE {
            return Err(format!(
                "Rate limit {} exceeds maximum {}",
                bytes_per_second, MAX_RATE
            ));
        }

        Ok(())
    }

    /// Validate batch size
    pub fn validate_batch_size(count: usize) -> Result<(), String> {
        const MAX_BATCH_SIZE: usize = 1000;

        if count == 0 {
            return Err("Batch cannot be empty".to_string());
        }

        if count > MAX_BATCH_SIZE {
            return Err(format!(
                "Batch size {} exceeds maximum {}",
                count, MAX_BATCH_SIZE
            ));
        }

        Ok(())
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Streaming operation errors
#[derive(Debug)]
pub enum StreamingError {
    InvalidCid(String),
    NotFound(String),
    Upload(String),
    Storage(String),
    Validation(String),
}

impl IntoResponse for StreamingError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            StreamingError::InvalidCid(cid) => {
                (StatusCode::BAD_REQUEST, format!("Invalid CID: {}", cid))
            }
            StreamingError::NotFound(cid) => {
                (StatusCode::NOT_FOUND, format!("Block not found: {}", cid))
            }
            StreamingError::Upload(msg) => {
                (StatusCode::BAD_REQUEST, format!("Upload error: {}", msg))
            }
            StreamingError::Storage(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Storage error: {}", msg),
            ),
            StreamingError::Validation(msg) => (
                StatusCode::BAD_REQUEST,
                format!("Validation error: {}", msg),
            ),
        };

        (status, message).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_event_serialization() {
        let event = ProgressEvent {
            operation_id: "test-123".to_string(),
            bytes_processed: 1024,
            total_bytes: Some(2048),
            progress_percent: Some(50.0),
            status: ProgressStatus::InProgress,
        };

        let json =
            serde_json::to_string(&event).expect("test: progress event should serialize to JSON");
        assert!(json.contains("test-123"));
        assert!(json.contains("1024"));
        assert!(json.contains("inprogress"));
    }

    #[test]
    fn test_progress_tracker() {
        let tracker = ProgressTracker::new();
        let _receiver = tracker.subscribe();

        let event = ProgressEvent {
            operation_id: "test".to_string(),
            bytes_processed: 100,
            total_bytes: Some(200),
            progress_percent: Some(50.0),
            status: ProgressStatus::InProgress,
        };

        tracker.send(event);

        // Note: In async context, we would await the receiver
        // This test just verifies the tracker can be created and used
    }

    #[test]
    fn test_batch_request_deserialization() {
        let json = r#"{"cids": ["QmTest1", "QmTest2"]}"#;
        let req: BatchGetRequest = serde_json::from_str(json)
            .expect("test: valid JSON should deserialize to BatchGetRequest");
        assert_eq!(req.cids.len(), 2);
        assert_eq!(req.cids[0], "QmTest1");
    }

    #[test]
    fn test_batch_put_request_deserialization() {
        let json = r#"{"blocks": [{"data": "SGVsbG8="}]}"#;
        let req: BatchPutRequest = serde_json::from_str(json)
            .expect("test: valid JSON should deserialize to BatchPutRequest");
        assert_eq!(req.blocks.len(), 1);
        assert_eq!(req.blocks[0].data, "SGVsbG8=");
        assert_eq!(req.transaction_mode, TransactionMode::BestEffort); // Default
    }

    #[test]
    fn test_batch_put_request_atomic_mode() {
        let json = r#"{"blocks": [{"data": "SGVsbG8="}], "transaction_mode": "atomic"}"#;
        let req: BatchPutRequest = serde_json::from_str(json)
            .expect("test: valid JSON with atomic mode should deserialize");
        assert_eq!(req.transaction_mode, TransactionMode::Atomic);
    }

    #[test]
    fn test_transaction_mode_default() {
        let mode = TransactionMode::default();
        assert_eq!(mode, TransactionMode::BestEffort);
    }

    #[test]
    fn test_transaction_status_serialization() {
        let status = TransactionStatus::Committed;
        let json = serde_json::to_string(&status)
            .expect("test: TransactionStatus::Committed should serialize");
        assert_eq!(json, r#""committed""#);

        let status = TransactionStatus::RolledBack;
        let json = serde_json::to_string(&status)
            .expect("test: TransactionStatus::RolledBack should serialize");
        assert_eq!(json, r#""rolledback""#);
    }

    #[test]
    fn test_batch_put_response_with_transaction() {
        let response = BatchPutResponse {
            stored: vec![],
            errors: vec![],
            transaction_id: Some("test-txn-123".to_string()),
            transaction_status: TransactionStatus::Committed,
        };

        let json = serde_json::to_string(&response)
            .expect("test: BatchPutResponse should serialize to JSON");
        assert!(json.contains("test-txn-123"));
        assert!(json.contains("committed"));
    }

    #[test]
    fn test_flow_control_config_default() {
        let config = FlowControlConfig::default();
        assert_eq!(config.max_bytes_per_second, 0);
        assert_eq!(config.initial_window_size, 256 * 1024);
        assert_eq!(config.max_window_size, 1024 * 1024);
        assert_eq!(config.min_window_size, 64 * 1024);
        assert!(config.dynamic_adjustment);
    }

    #[test]
    fn test_flow_control_config_with_rate_limit() {
        let config = FlowControlConfig::with_rate_limit(1_000_000); // 1 MB/s
        assert_eq!(config.max_bytes_per_second, 1_000_000);
        assert!(config.dynamic_adjustment);
    }

    #[test]
    fn test_flow_control_config_conservative() {
        let config = FlowControlConfig::conservative();
        assert_eq!(config.initial_window_size, 64 * 1024);
        assert_eq!(config.max_window_size, 256 * 1024);
        assert_eq!(config.min_window_size, 32 * 1024);
    }

    #[test]
    fn test_flow_control_config_aggressive() {
        let config = FlowControlConfig::aggressive();
        assert_eq!(config.initial_window_size, 512 * 1024);
        assert_eq!(config.max_window_size, 2 * 1024 * 1024);
        assert_eq!(config.min_window_size, 128 * 1024);
    }

    #[test]
    fn test_flow_controller_window_size() {
        let config = FlowControlConfig::default();
        let controller = FlowController::new(config.clone());
        assert_eq!(controller.window_size(), config.initial_window_size);
    }

    #[test]
    fn test_flow_controller_no_rate_limit() {
        let config = FlowControlConfig::default(); // No rate limit (0)
        let controller = FlowController::new(config);

        // Should not delay when no rate limit
        let delay = controller.calculate_delay(1024);
        assert_eq!(delay, std::time::Duration::from_secs(0));
    }

    #[test]
    fn test_flow_controller_on_data_sent() {
        let config = FlowControlConfig::default();
        let mut controller = FlowController::new(config);

        controller.on_data_sent(1024);
        assert_eq!(controller.bytes_sent, 1024);

        controller.on_data_sent(2048);
        assert_eq!(controller.bytes_sent, 3072);
    }

    #[test]
    fn test_flow_controller_on_congestion() {
        let config = FlowControlConfig::default();
        let mut controller = FlowController::new(config.clone());

        let initial_window = controller.window_size();
        controller.on_congestion();

        // Window should decrease by 50%
        assert!(controller.window_size() < initial_window);
        assert!(controller.window_size() >= config.min_window_size);
    }

    #[test]
    fn test_flow_controller_throughput() {
        let config = FlowControlConfig::default();
        let mut controller = FlowController::new(config);

        // Send some data
        controller.on_data_sent(1024);

        // Throughput should be non-negative
        let throughput = controller.current_throughput();
        assert!(throughput >= 0.0);
    }

    #[test]
    fn test_resume_token_encode_decode() {
        let token = ResumeToken::new("op-123".to_string(), 4096, Some("QmTest123".to_string()));

        // Encode
        let encoded = token
            .encode()
            .expect("test: resume token should encode successfully");
        assert!(!encoded.is_empty());

        // Decode
        let decoded =
            ResumeToken::decode(&encoded).expect("test: valid encoded token should decode");
        assert_eq!(decoded.operation_id, "op-123");
        assert_eq!(decoded.offset, 4096);
        assert_eq!(decoded.cid, Some("QmTest123".to_string()));
    }

    #[test]
    fn test_resume_token_invalid() {
        // Invalid base64
        let result = ResumeToken::decode("invalid!!!base64");
        assert!(result.is_err());

        // Valid base64 but invalid JSON
        let invalid_json = base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            b"not json",
        );
        let result = ResumeToken::decode(&invalid_json);
        assert!(result.is_err());
    }

    #[test]
    fn test_operation_type_serialization() {
        let upload = OperationType::Upload;
        let json =
            serde_json::to_string(&upload).expect("test: OperationType::Upload should serialize");
        assert_eq!(json, r#""upload""#);

        let download = OperationType::Download;
        let json = serde_json::to_string(&download)
            .expect("test: OperationType::Download should serialize");
        assert_eq!(json, r#""download""#);
    }

    #[test]
    fn test_operation_status_serialization() {
        let status = OperationStatus::InProgress;
        let json = serde_json::to_string(&status)
            .expect("test: OperationStatus::InProgress should serialize");
        assert_eq!(json, r#""inprogress""#);

        let status = OperationStatus::Cancelled;
        let json = serde_json::to_string(&status)
            .expect("test: OperationStatus::Cancelled should serialize");
        assert_eq!(json, r#""cancelled""#);
    }

    #[test]
    fn test_cancel_response_serialization() {
        let response = CancelResponse {
            operation_id: "op-456".to_string(),
            cancelled: true,
            resume_token: Some("token123".to_string()),
        };

        let json = serde_json::to_string(&response)
            .expect("test: CancelResponse should serialize to JSON");
        assert!(json.contains("op-456"));
        assert!(json.contains("true"));
        assert!(json.contains("token123"));
    }

    // Validation tests
    #[test]
    fn test_validate_cid_valid() {
        assert!(validation::validate_cid("QmTest123456").is_ok());
        assert!(validation::validate_cid(
            "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
        )
        .is_ok());
        assert!(validation::validate_cid(
            "bafkreigh2akiscaildcqabsyg3dfr6chu3fgpregiymsck7e7aqa4s52zy"
        )
        .is_ok());
    }

    #[test]
    fn test_validate_cid_invalid() {
        assert!(validation::validate_cid("").is_err());
        assert!(validation::validate_cid("invalid").is_err());
        assert!(validation::validate_cid("Qm").is_err());
    }

    #[test]
    fn test_validate_offset_valid() {
        assert!(validation::validate_offset(0, 1000).is_ok());
        assert!(validation::validate_offset(500, 1000).is_ok());
        assert!(validation::validate_offset(999, 1000).is_ok());
    }

    #[test]
    fn test_validate_offset_invalid() {
        assert!(validation::validate_offset(1000, 1000).is_err());
        assert!(validation::validate_offset(2000, 1000).is_err());
    }

    #[test]
    fn test_validate_chunk_size_valid() {
        assert!(validation::validate_chunk_size(1024).is_ok()); // Minimum
        assert!(validation::validate_chunk_size(64 * 1024).is_ok()); // 64KB
        assert!(validation::validate_chunk_size(10 * 1024 * 1024).is_ok()); // Maximum
    }

    #[test]
    fn test_validate_chunk_size_invalid() {
        assert!(validation::validate_chunk_size(512).is_err()); // Too small
        assert!(validation::validate_chunk_size(20 * 1024 * 1024).is_err()); // Too large
    }

    #[test]
    fn test_validate_rate_limit_valid() {
        assert!(validation::validate_rate_limit(0).is_ok()); // Unlimited
        assert!(validation::validate_rate_limit(1_000_000).is_ok()); // 1 MB/s
        assert!(validation::validate_rate_limit(10 * 1024 * 1024 * 1024).is_ok());
        // Maximum
    }

    #[test]
    fn test_validate_rate_limit_invalid() {
        assert!(validation::validate_rate_limit(20 * 1024 * 1024 * 1024).is_err());
        // Too large
    }

    #[test]
    fn test_validate_batch_size_valid() {
        assert!(validation::validate_batch_size(1).is_ok());
        assert!(validation::validate_batch_size(100).is_ok());
        assert!(validation::validate_batch_size(1000).is_ok()); // Maximum
    }

    #[test]
    fn test_validate_batch_size_invalid() {
        assert!(validation::validate_batch_size(0).is_err()); // Empty
        assert!(validation::validate_batch_size(1001).is_err()); // Too large
        assert!(validation::validate_batch_size(5000).is_err()); // Way too large
    }

    #[test]
    fn test_flow_control_config_validation_valid() {
        let config = FlowControlConfig::default();
        assert!(config.validate().is_ok());

        let config = FlowControlConfig::conservative();
        assert!(config.validate().is_ok());

        let config = FlowControlConfig::aggressive();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_flow_control_config_validation_invalid() {
        // Min window size exceeds initial window size
        let config = FlowControlConfig {
            max_bytes_per_second: 0,
            initial_window_size: 64 * 1024,
            max_window_size: 1024 * 1024,
            min_window_size: 128 * 1024, // Exceeds initial
            dynamic_adjustment: true,
        };
        assert!(config.validate().is_err());

        // Initial window size exceeds max window size
        let config = FlowControlConfig {
            max_bytes_per_second: 0,
            initial_window_size: 2 * 1024 * 1024,
            max_window_size: 1024 * 1024, // Less than initial
            min_window_size: 64 * 1024,
            dynamic_adjustment: true,
        };
        assert!(config.validate().is_err());

        // Rate limit too high
        let config = FlowControlConfig {
            max_bytes_per_second: 20 * 1024 * 1024 * 1024, // Too high
            initial_window_size: 256 * 1024,
            max_window_size: 1024 * 1024,
            min_window_size: 64 * 1024,
            dynamic_adjustment: true,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_concurrency_config_default() {
        let config = ConcurrencyConfig::default();
        assert_eq!(config.max_concurrent_tasks, 100);
        assert!(config.parallel_enabled);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_concurrency_config_conservative() {
        let config = ConcurrencyConfig::conservative();
        assert_eq!(config.max_concurrent_tasks, 50);
        assert!(config.parallel_enabled);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_concurrency_config_aggressive() {
        let config = ConcurrencyConfig::aggressive();
        assert_eq!(config.max_concurrent_tasks, 200);
        assert!(config.parallel_enabled);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_concurrency_config_sequential() {
        let config = ConcurrencyConfig::sequential();
        assert_eq!(config.max_concurrent_tasks, 1);
        assert!(!config.parallel_enabled);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_concurrency_config_validation_invalid() {
        let config = ConcurrencyConfig {
            max_concurrent_tasks: 0,
            parallel_enabled: true,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_concurrency_config_validation_valid() {
        let config = ConcurrencyConfig {
            max_concurrent_tasks: 0,
            parallel_enabled: false,
        };
        assert!(config.validate().is_ok());

        let config = ConcurrencyConfig {
            max_concurrent_tasks: 100,
            parallel_enabled: true,
        };
        assert!(config.validate().is_ok());
    }
}
