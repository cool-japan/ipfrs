//! TensorSwap protocol for efficient tensor streaming
//!
//! TensorSwap extends Bitswap with optimizations for tensor data:
//! - Priority-based scheduling for computation graphs
//! - Progressive streaming for early inference
//! - Safetensors format support
//! - Computation-aware block ordering
//! - Chunked tensor transfer with backpressure
//! - Deadline-based priority elevation
//!
//! # Example
//!
//! ```
//! use ipfrs_transport::TensorMetadata;
//! use ipfrs_core::Cid;
//! use multihash::Multihash;
//! use std::time::{Duration, Instant};
//!
//! // Create a CID for the tensor
//! let hash = Multihash::wrap(0x12, &[1, 2, 3, 4]).unwrap();
//! let cid = Cid::new_v1(0x55, hash);
//!
//! // Create tensor metadata with shape and type information
//! let metadata = TensorMetadata::new(cid)
//!     .with_shape(vec![768, 768])
//!     .with_dtype("f32".to_string())
//!     .critical()
//!     .with_deadline(Instant::now() + Duration::from_secs(5));
//!
//! println!("Tensor shape: {:?}", metadata.shape);
//! println!("Data type: {:?}", metadata.dtype);
//! println!("Is critical: {}", metadata.is_critical);
//! ```

use crate::bitswap::{BitswapConfig, BitswapExchange};
use crate::peer_manager::PeerId;
use crate::want_list::Priority;
use ipfrs_core::{Block, Cid, Result};
use ipfrs_storage::traits::BlockStore;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Tensor metadata for smart scheduling
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    /// CID of the tensor block (or root CID for chunked tensors)
    pub cid: Cid,
    /// Shape information (if available)
    pub shape: Option<Vec<usize>>,
    /// Data type (f32, f16, bf16, i64, etc.)
    pub dtype: Option<String>,
    /// Dependencies (must be fetched first)
    pub dependencies: Vec<Cid>,
    /// Computation deadline (if any)
    pub deadline: Option<Instant>,
    /// Total size in bytes (if known)
    pub size_bytes: Option<u64>,
    /// Chunk CIDs for large tensors (in order)
    pub chunks: Vec<Cid>,
    /// User-defined priority hint
    pub priority_hint: Option<i32>,
    /// Layer name (for transformer models)
    pub layer_name: Option<String>,
    /// Whether this tensor is critical for computation
    pub is_critical: bool,
}

impl TensorMetadata {
    /// Create new metadata for a tensor
    pub fn new(cid: Cid) -> Self {
        Self {
            cid,
            shape: None,
            dtype: None,
            dependencies: Vec::new(),
            deadline: None,
            size_bytes: None,
            chunks: Vec::new(),
            priority_hint: None,
            layer_name: None,
            is_critical: false,
        }
    }

    /// Set shape
    pub fn with_shape(mut self, shape: Vec<usize>) -> Self {
        self.shape = Some(shape);
        self
    }

    /// Set dtype
    pub fn with_dtype(mut self, dtype: impl Into<String>) -> Self {
        self.dtype = Some(dtype.into());
        self
    }

    /// Set dependencies
    pub fn with_dependencies(mut self, deps: Vec<Cid>) -> Self {
        self.dependencies = deps;
        self
    }

    /// Set deadline
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Set size
    pub fn with_size(mut self, size: u64) -> Self {
        self.size_bytes = Some(size);
        self
    }

    /// Set chunks
    pub fn with_chunks(mut self, chunks: Vec<Cid>) -> Self {
        self.chunks = chunks;
        self
    }

    /// Set priority hint
    pub fn with_priority_hint(mut self, priority: i32) -> Self {
        self.priority_hint = Some(priority);
        self
    }

    /// Set layer name
    pub fn with_layer_name(mut self, name: impl Into<String>) -> Self {
        self.layer_name = Some(name.into());
        self
    }

    /// Mark as critical
    pub fn critical(mut self) -> Self {
        self.is_critical = true;
        self
    }

    /// Calculate total elements from shape
    pub fn num_elements(&self) -> Option<usize> {
        self.shape.as_ref().map(|s| s.iter().product())
    }

    /// Estimate size from shape and dtype
    pub fn estimated_size(&self) -> Option<u64> {
        if let Some(size) = self.size_bytes {
            return Some(size);
        }

        let elements = self.num_elements()? as u64;
        let bytes_per_element = match self.dtype.as_deref() {
            Some("f32") | Some("i32") | Some("u32") => 4,
            Some("f64") | Some("i64") | Some("u64") => 8,
            Some("f16") | Some("bf16") | Some("i16") | Some("u16") => 2,
            Some("i8") | Some("u8") | Some("bool") => 1,
            _ => return None,
        };
        Some(elements * bytes_per_element)
    }

    /// Check if this tensor is chunked
    pub fn is_chunked(&self) -> bool {
        !self.chunks.is_empty()
    }
}

/// Chunk information for streaming
#[derive(Debug, Clone)]
pub struct ChunkInfo {
    /// CID of the chunk
    pub cid: Cid,
    /// Index in the tensor (0-based)
    pub index: usize,
    /// Byte offset in the tensor
    pub offset: u64,
    /// Size of this chunk
    pub size: u64,
    /// Whether this chunk has been received
    pub received: bool,
}

/// Streaming state for a tensor
#[derive(Debug)]
pub struct TensorStream {
    /// Root CID of the tensor
    pub root_cid: Cid,
    /// Tensor metadata
    pub metadata: TensorMetadata,
    /// Chunks in order
    pub chunks: Vec<ChunkInfo>,
    /// Number of chunks received
    pub chunks_received: usize,
    /// Total bytes received
    pub bytes_received: u64,
    /// When streaming started
    pub started_at: Instant,
    /// Callback channel for progressive updates
    progress_tx: Option<mpsc::Sender<StreamProgress>>,
}

/// Progress update for streaming
#[derive(Debug, Clone)]
pub struct StreamProgress {
    /// Root CID of the tensor
    pub root_cid: Cid,
    /// Chunk index that was received
    pub chunk_index: usize,
    /// Chunk CID
    pub chunk_cid: Cid,
    /// Total chunks
    pub total_chunks: usize,
    /// Bytes received so far
    pub bytes_received: u64,
    /// Total bytes (if known)
    pub total_bytes: Option<u64>,
    /// Whether streaming is complete
    pub complete: bool,
}

impl TensorStream {
    /// Create a new tensor stream
    pub fn new(metadata: TensorMetadata) -> Self {
        let chunks: Vec<ChunkInfo> = if metadata.is_chunked() {
            let chunk_size = metadata
                .size_bytes
                .map(|s| s / metadata.chunks.len() as u64)
                .unwrap_or(1024 * 1024); // Default 1MB

            metadata
                .chunks
                .iter()
                .enumerate()
                .map(|(i, cid)| ChunkInfo {
                    cid: *cid,
                    index: i,
                    offset: i as u64 * chunk_size,
                    size: chunk_size,
                    received: false,
                })
                .collect()
        } else {
            // Single chunk for non-chunked tensors
            vec![ChunkInfo {
                cid: metadata.cid,
                index: 0,
                offset: 0,
                size: metadata.size_bytes.unwrap_or(0),
                received: false,
            }]
        };

        Self {
            root_cid: metadata.cid,
            metadata,
            chunks,
            chunks_received: 0,
            bytes_received: 0,
            started_at: Instant::now(),
            progress_tx: None,
        }
    }

    /// Set progress callback
    pub fn with_progress_channel(mut self, tx: mpsc::Sender<StreamProgress>) -> Self {
        self.progress_tx = Some(tx);
        self
    }

    /// Mark a chunk as received
    pub async fn mark_received(&mut self, cid: &Cid, size: u64) -> bool {
        if let Some(chunk) = self.chunks.iter_mut().find(|c| c.cid == *cid) {
            if !chunk.received {
                chunk.received = true;
                chunk.size = size;
                self.chunks_received += 1;
                self.bytes_received += size;

                // Send progress update if channel is set
                if let Some(tx) = &self.progress_tx {
                    let progress = StreamProgress {
                        root_cid: self.root_cid,
                        chunk_index: chunk.index,
                        chunk_cid: *cid,
                        total_chunks: self.chunks.len(),
                        bytes_received: self.bytes_received,
                        total_bytes: self.metadata.size_bytes,
                        complete: self.is_complete(),
                    };
                    let _ = tx.send(progress).await;
                }

                return true;
            }
        }
        false
    }

    /// Check if all chunks have been received
    pub fn is_complete(&self) -> bool {
        self.chunks_received >= self.chunks.len()
    }

    /// Get progress as a fraction (0.0 - 1.0)
    pub fn progress(&self) -> f64 {
        if self.chunks.is_empty() {
            return 1.0;
        }
        self.chunks_received as f64 / self.chunks.len() as f64
    }

    /// Get missing chunk CIDs
    pub fn missing_chunks(&self) -> Vec<Cid> {
        self.chunks
            .iter()
            .filter(|c| !c.received)
            .map(|c| c.cid)
            .collect()
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Get current throughput in bytes/second
    pub fn throughput(&self) -> f64 {
        let elapsed = self.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.bytes_received as f64 / elapsed
        } else {
            0.0
        }
    }
}

/// Backpressure configuration
#[derive(Debug, Clone)]
pub struct BackpressureConfig {
    /// Maximum pending chunks
    pub max_pending: usize,
    /// High watermark (pause sending above this)
    pub high_watermark: usize,
    /// Low watermark (resume sending below this)
    pub low_watermark: usize,
    /// Maximum buffer size in bytes
    pub max_buffer_bytes: usize,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            max_pending: 64,
            high_watermark: 48,
            low_watermark: 16,
            max_buffer_bytes: 64 * 1024 * 1024, // 64 MB
        }
    }
}

/// Backpressure controller for flow control
#[derive(Debug)]
pub struct BackpressureController {
    config: BackpressureConfig,
    pending_count: usize,
    pending_bytes: usize,
    paused: bool,
}

impl BackpressureController {
    /// Create a new backpressure controller
    pub fn new(config: BackpressureConfig) -> Self {
        Self {
            config,
            pending_count: 0,
            pending_bytes: 0,
            paused: false,
        }
    }

    /// Check if we should accept more data
    pub fn should_accept(&self) -> bool {
        !self.paused
            && self.pending_count < self.config.max_pending
            && self.pending_bytes < self.config.max_buffer_bytes
    }

    /// Record data being sent
    pub fn on_send(&mut self, bytes: usize) {
        self.pending_count += 1;
        self.pending_bytes += bytes;

        if self.pending_count >= self.config.high_watermark
            || self.pending_bytes >= self.config.max_buffer_bytes
        {
            self.paused = true;
        }
    }

    /// Record data acknowledgement
    pub fn on_ack(&mut self, bytes: usize) {
        self.pending_count = self.pending_count.saturating_sub(1);
        self.pending_bytes = self.pending_bytes.saturating_sub(bytes);

        if self.pending_count <= self.config.low_watermark {
            self.paused = false;
        }
    }

    /// Check if currently paused
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Get pending count
    pub fn pending_count(&self) -> usize {
        self.pending_count
    }

    /// Get pending bytes
    pub fn pending_bytes(&self) -> usize {
        self.pending_bytes
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.pending_count = 0;
        self.pending_bytes = 0;
        self.paused = false;
    }
}

/// Safetensors header information
#[derive(Debug, Clone)]
pub struct SafetensorsHeader {
    /// Header size in bytes
    pub header_size: u64,
    /// Tensor entries
    pub tensors: HashMap<String, SafetensorEntry>,
}

/// Entry in safetensors file
#[derive(Debug, Clone)]
pub struct SafetensorEntry {
    /// Tensor name
    pub name: String,
    /// Data type
    pub dtype: String,
    /// Shape
    pub shape: Vec<usize>,
    /// Byte offset in data section
    pub data_offset: u64,
    /// Byte length
    pub data_length: u64,
}

impl SafetensorsHeader {
    /// Parse safetensors header from bytes
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(ipfrs_core::Error::Deserialization(
                "Safetensors header too short".to_string(),
            ));
        }

        // First 8 bytes are header size (little-endian u64)
        let header_size = u64::from_le_bytes(data[0..8].try_into().unwrap());

        if data.len() < 8 + header_size as usize {
            return Err(ipfrs_core::Error::Deserialization(
                "Safetensors data too short for header".to_string(),
            ));
        }

        // Header is JSON
        let header_json = &data[8..8 + header_size as usize];
        let header_map: HashMap<String, serde_json::Value> = serde_json::from_slice(header_json)
            .map_err(|e| {
                ipfrs_core::Error::Deserialization(format!("Invalid safetensors header: {}", e))
            })?;

        let mut tensors = HashMap::new();

        for (name, value) in header_map {
            if name == "__metadata__" {
                continue;
            }

            if let Some(obj) = value.as_object() {
                let dtype = obj
                    .get("dtype")
                    .and_then(|v| v.as_str())
                    .unwrap_or("F32")
                    .to_string();

                let shape: Vec<usize> = obj
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as usize))
                            .collect()
                    })
                    .unwrap_or_default();

                let offsets = obj.get("data_offsets").and_then(|v| v.as_array());
                let (data_offset, data_length) = if let Some(offs) = offsets {
                    let start = offs.first().and_then(|v| v.as_u64()).unwrap_or(0);
                    let end = offs.get(1).and_then(|v| v.as_u64()).unwrap_or(start);
                    (start, end - start)
                } else {
                    (0, 0)
                };

                tensors.insert(
                    name.clone(),
                    SafetensorEntry {
                        name,
                        dtype,
                        shape,
                        data_offset,
                        data_length,
                    },
                );
            }
        }

        Ok(Self {
            header_size,
            tensors,
        })
    }

    /// Get data offset (after header)
    pub fn data_start(&self) -> u64 {
        8 + self.header_size
    }

    /// Get tensor entry by name
    pub fn get_tensor(&self, name: &str) -> Option<&SafetensorEntry> {
        self.tensors.get(name)
    }

    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }
}

/// Request queue for prioritized streaming
#[derive(Debug)]
pub struct StreamRequestQueue {
    /// Queued requests
    requests: VecDeque<StreamRequest>,
    /// Maximum queue size
    max_size: usize,
}

/// A streaming request
#[derive(Debug, Clone)]
pub struct StreamRequest {
    /// Root CID
    pub cid: Cid,
    /// Priority
    pub priority: i32,
    /// Deadline (if any)
    pub deadline: Option<Instant>,
    /// When request was queued
    pub queued_at: Instant,
}

impl StreamRequestQueue {
    /// Create a new request queue
    pub fn new(max_size: usize) -> Self {
        Self {
            requests: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    /// Add a request to the queue
    pub fn push(&mut self, request: StreamRequest) -> bool {
        if self.requests.len() >= self.max_size {
            return false;
        }

        // Insert in priority order (higher priority first)
        let pos = self
            .requests
            .iter()
            .position(|r| r.priority < request.priority)
            .unwrap_or(self.requests.len());

        self.requests.insert(pos, request);
        true
    }

    /// Pop highest priority request
    pub fn pop(&mut self) -> Option<StreamRequest> {
        self.requests.pop_front()
    }

    /// Peek at highest priority request
    pub fn peek(&self) -> Option<&StreamRequest> {
        self.requests.front()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Get queue length
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Boost priority for deadline-approaching requests
    pub fn boost_deadlines(&mut self) {
        let now = Instant::now();
        for request in &mut self.requests {
            if let Some(deadline) = request.deadline {
                if now >= deadline {
                    request.priority = request.priority.max(Priority::Critical as i32);
                } else if deadline.duration_since(now) < Duration::from_secs(1) {
                    request.priority = request.priority.max(Priority::Urgent as i32);
                }
            }
        }

        // Re-sort after priority updates
        let mut vec: Vec<_> = self.requests.drain(..).collect();
        vec.sort_by(|a, b| b.priority.cmp(&a.priority));
        self.requests = vec.into();
    }
}

/// TensorSwap configuration extending Bitswap config
#[derive(Debug, Clone)]
pub struct TensorSwapConfig {
    /// Underlying Bitswap configuration
    pub bitswap: BitswapConfig,
    /// Enable progressive streaming
    pub progressive_streaming: bool,
    /// Chunk size for streaming (bytes)
    pub chunk_size: usize,
    /// Enable deadline-based prioritization
    pub deadline_aware: bool,
    /// Backpressure configuration
    pub backpressure: BackpressureConfig,
    /// Maximum concurrent tensor streams
    pub max_concurrent_streams: usize,
    /// Priority boost for critical tensors
    pub critical_priority_boost: i32,
    /// Dependency priority boost (per level)
    pub dependency_priority_boost: i32,
}

impl Default for TensorSwapConfig {
    fn default() -> Self {
        Self {
            bitswap: BitswapConfig::default(),
            progressive_streaming: true,
            chunk_size: 1024 * 1024, // 1MB chunks
            deadline_aware: true,
            backpressure: BackpressureConfig::default(),
            max_concurrent_streams: 16,
            critical_priority_boost: 50,
            dependency_priority_boost: 10,
        }
    }
}

/// TensorSwap protocol handler
///
/// Extends Bitswap with tensor-aware optimizations for ML workloads
pub struct TensorSwap<S: BlockStore> {
    /// Underlying Bitswap exchange
    bitswap: Arc<BitswapExchange<S>>,
    /// Tensor metadata registry
    tensor_metadata: Arc<RwLock<HashMap<Cid, TensorMetadata>>>,
    /// Active tensor streams
    active_streams: Arc<RwLock<HashMap<Cid, TensorStream>>>,
    /// Backpressure controller
    backpressure: Arc<RwLock<BackpressureController>>,
    /// Configuration
    config: TensorSwapConfig,
}

impl<S: BlockStore> TensorSwap<S> {
    /// Create a new TensorSwap handler
    pub fn new(store: Arc<S>, config: TensorSwapConfig) -> Result<Self> {
        let bitswap = Arc::new(BitswapExchange::new(store, config.bitswap.clone())?);
        let backpressure = BackpressureController::new(config.backpressure.clone());

        Ok(Self {
            bitswap,
            tensor_metadata: Arc::new(RwLock::new(HashMap::new())),
            active_streams: Arc::new(RwLock::new(HashMap::new())),
            backpressure: Arc::new(RwLock::new(backpressure)),
            config,
        })
    }

    /// Create with default configuration
    pub fn with_defaults(store: Arc<S>) -> Result<Self> {
        Self::new(store, TensorSwapConfig::default())
    }

    /// Register tensor metadata for smart scheduling
    pub fn register_tensor(&self, metadata: TensorMetadata) {
        self.tensor_metadata
            .write()
            .unwrap()
            .insert(metadata.cid, metadata);
    }

    /// Request a tensor with dependency-aware scheduling
    pub fn want_tensor(&self, cid: Cid) -> Result<()> {
        let metadata = self.tensor_metadata.read().unwrap();

        // Calculate priority based on metadata
        let priority = if let Some(meta) = metadata.get(&cid) {
            self.calculate_priority(meta)
        } else {
            0 // Default priority for unknown tensors
        };

        // Request tensor via Bitswap
        self.bitswap.want(cid, priority)?;

        // Also request dependencies with higher priority
        if let Some(meta) = metadata.get(&cid) {
            for dep_cid in &meta.dependencies {
                let dep_priority = priority + self.config.dependency_priority_boost;
                self.bitswap.want(*dep_cid, dep_priority)?;
            }
        }

        Ok(())
    }

    /// Start streaming a chunked tensor
    pub fn start_stream(&self, cid: Cid) -> Result<()> {
        // Check backpressure
        if !self.backpressure.read().unwrap().should_accept() {
            return Err(ipfrs_core::Error::Internal(
                "Backpressure limit reached".to_string(),
            ));
        }

        // Check concurrent stream limit
        let active_count = self.active_streams.read().unwrap().len();
        if active_count >= self.config.max_concurrent_streams {
            return Err(ipfrs_core::Error::Internal(
                "Maximum concurrent streams reached".to_string(),
            ));
        }

        let metadata = {
            let meta_map = self.tensor_metadata.read().unwrap();
            meta_map.get(&cid).cloned()
        };

        let metadata = metadata.unwrap_or_else(|| TensorMetadata::new(cid));
        let stream = TensorStream::new(metadata.clone());

        // Request all chunks with appropriate priorities
        let base_priority = self.calculate_priority(&metadata);
        for (idx, chunk_info) in stream.chunks.iter().enumerate() {
            // Earlier chunks get higher priority for progressive streaming
            let chunk_priority = base_priority + (stream.chunks.len() - idx) as i32;
            self.bitswap.want(chunk_info.cid, chunk_priority)?;
        }

        // Store active stream
        self.active_streams.write().unwrap().insert(cid, stream);

        Ok(())
    }

    /// Start streaming with progress channel
    pub fn start_stream_with_progress(
        &self,
        cid: Cid,
        progress_tx: mpsc::Sender<StreamProgress>,
    ) -> Result<()> {
        // Check backpressure
        if !self.backpressure.read().unwrap().should_accept() {
            return Err(ipfrs_core::Error::Internal(
                "Backpressure limit reached".to_string(),
            ));
        }

        // Check concurrent stream limit
        let active_count = self.active_streams.read().unwrap().len();
        if active_count >= self.config.max_concurrent_streams {
            return Err(ipfrs_core::Error::Internal(
                "Maximum concurrent streams reached".to_string(),
            ));
        }

        let metadata = {
            let meta_map = self.tensor_metadata.read().unwrap();
            meta_map.get(&cid).cloned()
        };

        let metadata = metadata.unwrap_or_else(|| TensorMetadata::new(cid));
        let stream = TensorStream::new(metadata.clone()).with_progress_channel(progress_tx);

        // Request all chunks
        let base_priority = self.calculate_priority(&metadata);
        for (idx, chunk_info) in stream.chunks.iter().enumerate() {
            let chunk_priority = base_priority + (stream.chunks.len() - idx) as i32;
            self.bitswap.want(chunk_info.cid, chunk_priority)?;
        }

        self.active_streams.write().unwrap().insert(cid, stream);

        Ok(())
    }

    /// Get stream progress
    pub fn stream_progress(&self, cid: &Cid) -> Option<f64> {
        self.active_streams
            .read()
            .unwrap()
            .get(cid)
            .map(|s| s.progress())
    }

    /// Check if stream is complete
    pub fn is_stream_complete(&self, cid: &Cid) -> bool {
        self.active_streams
            .read()
            .unwrap()
            .get(cid)
            .map(|s| s.is_complete())
            .unwrap_or(false)
    }

    /// Cancel a tensor stream
    pub fn cancel_stream(&self, cid: &Cid) -> Result<()> {
        if let Some(stream) = self.active_streams.write().unwrap().remove(cid) {
            // Cancel all pending chunks
            for chunk in stream.missing_chunks() {
                self.bitswap.cancel_want(&chunk)?;
            }
        }
        Ok(())
    }

    /// Calculate priority based on tensor metadata
    fn calculate_priority(&self, meta: &TensorMetadata) -> i32 {
        let mut priority = meta.priority_hint.unwrap_or(0);

        // Critical tensor boost
        if meta.is_critical {
            priority += self.config.critical_priority_boost;
        }

        // Deadline-based priority
        if self.config.deadline_aware {
            if let Some(deadline) = meta.deadline {
                let now = Instant::now();
                if deadline > now {
                    let time_left = deadline.duration_since(now).as_secs();
                    // Higher priority for closer deadlines
                    priority += (100 - time_left.min(100)) as i32;
                } else {
                    // Very high priority if past deadline
                    priority += Priority::Critical as i32;
                }
            }
        }

        // Dependency depth affects priority
        priority += meta.dependencies.len() as i32 * self.config.dependency_priority_boost;

        priority
    }

    /// Receive tensor block from peer
    #[allow(clippy::await_holding_lock)]
    pub async fn receive_tensor(&self, peer_id: &PeerId, block: Block) -> Result<()> {
        let cid = *block.cid();
        let size = block.size();

        // Update active streams
        {
            let mut streams = self.active_streams.write().unwrap();
            for stream in streams.values_mut() {
                stream.mark_received(&cid, size).await;
            }
        }

        // Update backpressure
        self.backpressure.write().unwrap().on_ack(size as usize);

        self.bitswap.receive_block(peer_id, block).await
    }

    /// Send tensor block to peer
    pub async fn send_tensor(
        &self,
        peer_id: &PeerId,
        cid: &Cid,
    ) -> Result<Option<crate::messages::Message>> {
        // Check backpressure before sending
        let should_send = self.backpressure.read().unwrap().should_accept();
        if !should_send {
            return Err(ipfrs_core::Error::Internal(
                "Backpressure active".to_string(),
            ));
        }

        let result = self.bitswap.send_block(peer_id, cid).await?;

        // Update backpressure on send
        if let Some(crate::messages::Message::Block(block_msg)) = &result {
            self.backpressure
                .write()
                .unwrap()
                .on_send(block_msg.data.len());
        }

        Ok(result)
    }

    /// Cancel tensor request
    pub fn cancel_tensor(&self, cid: &Cid) -> Result<()> {
        // Also cancel any active stream
        self.cancel_stream(cid)?;
        self.bitswap.cancel_want(cid)
    }

    /// Get next tensor to fetch (highest priority)
    pub fn next_tensor(&self) -> Option<Cid> {
        self.bitswap.next_want()
    }

    /// Cleanup completed streams
    pub fn cleanup_completed_streams(&self) {
        let mut streams = self.active_streams.write().unwrap();
        streams.retain(|_, stream| !stream.is_complete());
    }

    /// Get statistics
    pub fn stats(&self) -> TensorSwapStats {
        let bitswap_stats = self.bitswap.stats();
        let streams = self.active_streams.read().unwrap();
        let backpressure = self.backpressure.read().unwrap();

        TensorSwapStats {
            want_list_size: bitswap_stats.want_list_size,
            num_tensors_registered: self.tensor_metadata.read().unwrap().len(),
            active_streams: streams.len(),
            total_bytes_sent: bitswap_stats.total_bytes_sent,
            total_bytes_recv: bitswap_stats.total_bytes_recv,
            backpressure_paused: backpressure.is_paused(),
            pending_chunks: backpressure.pending_count(),
        }
    }

    /// Access underlying Bitswap for advanced usage
    pub fn bitswap(&self) -> &Arc<BitswapExchange<S>> {
        &self.bitswap
    }

    /// Check if backpressure is active
    pub fn is_backpressure_active(&self) -> bool {
        self.backpressure.read().unwrap().is_paused()
    }

    /// Acknowledge data for backpressure
    pub fn ack_data(&self, bytes: usize) {
        self.backpressure.write().unwrap().on_ack(bytes);
    }
}

/// TensorSwap statistics
#[derive(Debug, Clone)]
pub struct TensorSwapStats {
    /// Number of tensors in want list
    pub want_list_size: usize,
    /// Number of registered tensor metadata entries
    pub num_tensors_registered: usize,
    /// Number of active tensor streams
    pub active_streams: usize,
    /// Total bytes sent
    pub total_bytes_sent: u64,
    /// Total bytes received
    pub total_bytes_recv: u64,
    /// Whether backpressure is active
    pub backpressure_paused: bool,
    /// Number of pending chunks
    pub pending_chunks: usize,
}

impl Default for TensorSwap<ipfrs_storage::SledBlockStore> {
    fn default() -> Self {
        let config = ipfrs_storage::BlockStoreConfig::default();
        let store = Arc::new(ipfrs_storage::SledBlockStore::new(config).unwrap());
        Self::with_defaults(store).unwrap()
    }
}

/// Einsum expression parser and dependency resolver
///
/// Parses einsum expressions to identify tensor dependencies and optimal fetch order
#[derive(Debug, Clone)]
pub struct EinsumExpression {
    /// Raw einsum string (e.g., "ij,jk->ik")
    pub expression: String,
    /// Input tensor identifiers
    pub inputs: Vec<String>,
    /// Output tensor identifier
    pub output: String,
}

impl EinsumExpression {
    /// Parse an einsum expression
    ///
    /// Format: "input1_indices,input2_indices->output_indices"
    /// Example: "ij,jk->ik" for matrix multiplication
    pub fn parse(expression: impl Into<String>) -> Result<Self> {
        let expr = expression.into();

        // Split on "->"
        let parts: Vec<&str> = expr.split("->").collect();
        if parts.len() != 2 {
            return Err(ipfrs_core::error::Error::InvalidInput(
                "Invalid einsum expression: missing '->'".to_string(),
            ));
        }

        let inputs_str = parts[0];
        let output_str = parts[1];

        // Parse inputs (comma-separated)
        let inputs: Vec<String> = inputs_str
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();

        if inputs.is_empty() {
            return Err(ipfrs_core::error::Error::InvalidInput(
                "Invalid einsum expression: no inputs".to_string(),
            ));
        }

        let output = output_str.trim().to_string();

        Ok(Self {
            expression: expr,
            inputs,
            output,
        })
    }

    /// Get number of input tensors
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Check if this is a reduction operation
    pub fn is_reduction(&self) -> bool {
        // If output has fewer indices than any input, it's a reduction
        let output_len = self.output.len();
        self.inputs.iter().any(|input| input.len() > output_len)
    }

    /// Check if this is a transpose operation
    pub fn is_transpose(&self) -> bool {
        self.inputs.len() == 1 && self.output.len() == self.inputs[0].len()
    }

    /// Get shared indices between inputs (for contraction)
    pub fn shared_indices(&self) -> Vec<char> {
        if self.inputs.len() < 2 {
            return Vec::new();
        }

        let first: std::collections::HashSet<char> = self.inputs[0].chars().collect();
        let second: std::collections::HashSet<char> = self.inputs[1].chars().collect();

        first.intersection(&second).copied().collect()
    }
}

/// Einsum computation graph
///
/// Manages dependencies and scheduling for einsum operations
#[derive(Debug)]
pub struct EinsumGraph {
    /// Einsum expressions to execute
    expressions: Vec<EinsumExpression>,
    /// Tensor name to CID mapping
    tensor_cids: HashMap<String, Cid>,
    /// Dependency graph: output -> inputs
    dependencies: HashMap<String, Vec<String>>,
}

impl EinsumGraph {
    /// Create a new einsum graph
    pub fn new() -> Self {
        Self {
            expressions: Vec::new(),
            tensor_cids: HashMap::new(),
            dependencies: HashMap::new(),
        }
    }

    /// Add an einsum expression to the graph
    pub fn add_expression(&mut self, expr: EinsumExpression) {
        // Record dependencies
        self.dependencies
            .insert(expr.output.clone(), expr.inputs.clone());
        self.expressions.push(expr);
    }

    /// Register a tensor CID
    pub fn register_tensor(&mut self, name: impl Into<String>, cid: Cid) {
        self.tensor_cids.insert(name.into(), cid);
    }

    /// Get all tensor CIDs
    pub fn tensor_cids(&self) -> &HashMap<String, Cid> {
        &self.tensor_cids
    }

    /// Get dependencies for a tensor
    pub fn get_dependencies(&self, tensor_name: &str) -> Option<Vec<Cid>> {
        let dep_names = self.dependencies.get(tensor_name)?;
        let cids: Option<Vec<Cid>> = dep_names
            .iter()
            .map(|name| self.tensor_cids.get(name).copied())
            .collect();
        cids
    }

    /// Compute topological order for tensor fetching
    ///
    /// Returns tensors in dependency order (leaves first)
    pub fn topological_order(&self) -> Result<Vec<(String, Cid)>> {
        let mut visited = std::collections::HashSet::new();
        let mut order = Vec::new();

        // Helper for DFS
        fn visit(
            node: &str,
            dependencies: &HashMap<String, Vec<String>>,
            tensor_cids: &HashMap<String, Cid>,
            visited: &mut std::collections::HashSet<String>,
            order: &mut Vec<(String, Cid)>,
        ) -> Result<()> {
            if visited.contains(node) {
                return Ok(());
            }

            visited.insert(node.to_string());

            // Visit dependencies first
            if let Some(deps) = dependencies.get(node) {
                for dep in deps {
                    visit(dep, dependencies, tensor_cids, visited, order)?;
                }
            }

            // Add this node
            if let Some(cid) = tensor_cids.get(node) {
                order.push((node.to_string(), *cid));
            }

            Ok(())
        }

        // Visit all tensors
        for tensor_name in self.tensor_cids.keys() {
            visit(
                tensor_name,
                &self.dependencies,
                &self.tensor_cids,
                &mut visited,
                &mut order,
            )?;
        }

        Ok(order)
    }

    /// Get priority for a tensor based on its position in the computation graph
    ///
    /// Leaf tensors (no dependencies) get highest priority
    pub fn compute_priority(&self, tensor_name: &str) -> i32 {
        let depth = self.compute_depth(tensor_name);
        // Invert depth so leaves get higher priority
        1000 - (depth as i32 * 100)
    }

    /// Compute depth of a tensor in the dependency graph
    fn compute_depth(&self, tensor_name: &str) -> usize {
        if let Some(deps) = self.dependencies.get(tensor_name) {
            if deps.is_empty() {
                return 0;
            }
            let max_dep_depth = deps
                .iter()
                .map(|d| self.compute_depth(d))
                .max()
                .unwrap_or(0);
            max_dep_depth + 1
        } else {
            0 // Leaf node
        }
    }

    /// Generate TensorMetadata with dependencies
    pub fn generate_metadata(&self, tensor_name: &str) -> Option<TensorMetadata> {
        let cid = *self.tensor_cids.get(tensor_name)?;
        let deps = self.get_dependencies(tensor_name).unwrap_or_default();
        let priority = self.compute_priority(tensor_name);

        Some(
            TensorMetadata::new(cid)
                .with_dependencies(deps)
                .with_priority_hint(priority),
        )
    }
}

impl Default for EinsumGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ipfrs_storage::{BlockStoreConfig, SledBlockStore};

    fn test_cid() -> Cid {
        "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse()
            .unwrap()
    }

    fn test_cid2() -> Cid {
        "bafybeiczsscdsbs7ffqz55asqdf3smv6klcw3gofszvwlyarci47bgf354"
            .parse()
            .unwrap()
    }

    #[tokio::test]
    async fn test_tensorswap_creation() {
        let config = BlockStoreConfig {
            path: std::path::PathBuf::from("/tmp/ipfrs-test-tensorswap-create"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).unwrap());
        let tensorswap = TensorSwap::with_defaults(store);
        assert!(tensorswap.is_ok());
    }

    #[tokio::test]
    async fn test_tensor_metadata() {
        let config = BlockStoreConfig {
            path: std::path::PathBuf::from("/tmp/ipfrs-test-tensorswap-meta"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).unwrap());
        let tensorswap = TensorSwap::with_defaults(store).unwrap();

        let cid = test_cid();
        let metadata = TensorMetadata::new(cid)
            .with_shape(vec![256, 256, 3])
            .with_dtype("f32");

        tensorswap.register_tensor(metadata);

        let stats = tensorswap.stats();
        assert_eq!(stats.num_tensors_registered, 1);
    }

    #[tokio::test]
    async fn test_dependency_scheduling() {
        let config = BlockStoreConfig {
            path: std::path::PathBuf::from("/tmp/ipfrs-test-tensorswap-dep"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).unwrap());
        let tensorswap = TensorSwap::with_defaults(store).unwrap();

        let cid1 = test_cid();
        let cid2 = test_cid2();

        // Register tensor with dependency
        let metadata = TensorMetadata::new(cid1)
            .with_shape(vec![256, 256])
            .with_dtype("f32")
            .with_dependencies(vec![cid2]);

        tensorswap.register_tensor(metadata);
        tensorswap.want_tensor(cid1).unwrap();

        // Both tensor and dependency should be wanted
        assert!(tensorswap.bitswap().is_wanted(&cid1));
        assert!(tensorswap.bitswap().is_wanted(&cid2));
    }

    #[test]
    fn test_tensor_metadata_builder() {
        let cid = test_cid();
        let metadata = TensorMetadata::new(cid)
            .with_shape(vec![1024, 768])
            .with_dtype("f32")
            .with_size(1024 * 768 * 4)
            .with_layer_name("encoder.layer.0.attention.query")
            .with_priority_hint(100)
            .critical();

        assert!(metadata.is_critical);
        assert_eq!(metadata.priority_hint, Some(100));
        assert_eq!(
            metadata.layer_name.as_deref(),
            Some("encoder.layer.0.attention.query")
        );
        assert_eq!(metadata.estimated_size(), Some(1024 * 768 * 4));
    }

    #[test]
    fn test_tensor_size_estimation() {
        let cid = test_cid();

        // f32 tensor
        let f32_meta = TensorMetadata::new(cid)
            .with_shape(vec![100, 100])
            .with_dtype("f32");
        assert_eq!(f32_meta.estimated_size(), Some(100 * 100 * 4));

        // f16 tensor
        let f16_meta = TensorMetadata::new(cid)
            .with_shape(vec![100, 100])
            .with_dtype("f16");
        assert_eq!(f16_meta.estimated_size(), Some(100 * 100 * 2));

        // i8 tensor
        let i8_meta = TensorMetadata::new(cid)
            .with_shape(vec![100, 100])
            .with_dtype("i8");
        assert_eq!(i8_meta.estimated_size(), Some(100 * 100));
    }

    #[test]
    fn test_backpressure_controller() {
        let config = BackpressureConfig {
            max_pending: 10,
            high_watermark: 8,
            low_watermark: 2,
            max_buffer_bytes: 1024,
        };

        let mut bp = BackpressureController::new(config);
        assert!(bp.should_accept());
        assert!(!bp.is_paused());

        // Send until high watermark
        for _ in 0..8 {
            bp.on_send(100);
        }
        assert!(bp.is_paused());
        assert!(!bp.should_accept());

        // Ack until low watermark
        for _ in 0..6 {
            bp.on_ack(100);
        }
        assert!(!bp.is_paused());
        assert!(bp.should_accept());
    }

    #[test]
    fn test_stream_request_queue() {
        let mut queue = StreamRequestQueue::new(10);
        let cid1 = test_cid();
        let cid2 = test_cid2();

        // Add low priority
        queue.push(StreamRequest {
            cid: cid1,
            priority: 10,
            deadline: None,
            queued_at: Instant::now(),
        });

        // Add high priority
        queue.push(StreamRequest {
            cid: cid2,
            priority: 100,
            deadline: None,
            queued_at: Instant::now(),
        });

        // High priority should come first
        let first = queue.pop().unwrap();
        assert_eq!(first.cid, cid2);
        assert_eq!(first.priority, 100);

        let second = queue.pop().unwrap();
        assert_eq!(second.cid, cid1);
    }

    #[test]
    fn test_tensor_stream() {
        let cid = test_cid();
        let chunk_cids = vec![test_cid(), test_cid2()];

        let metadata = TensorMetadata::new(cid)
            .with_chunks(chunk_cids.clone())
            .with_size(2 * 1024 * 1024);

        let stream = TensorStream::new(metadata);

        assert!(!stream.is_complete());
        assert_eq!(stream.progress(), 0.0);
        assert_eq!(stream.missing_chunks().len(), 2);
    }

    #[test]
    fn test_safetensors_header_parse() {
        // Minimal safetensors header format
        let header_json =
            r#"{"weight":{"dtype":"F32","shape":[768,768],"data_offsets":[0,2359296]}}"#;
        let header_bytes = header_json.as_bytes();
        let header_size = header_bytes.len() as u64;

        let mut data = Vec::new();
        data.extend_from_slice(&header_size.to_le_bytes());
        data.extend_from_slice(header_bytes);

        let header = SafetensorsHeader::parse(&data).unwrap();
        assert_eq!(header.tensors.len(), 1);

        let weight = header.get_tensor("weight").unwrap();
        assert_eq!(weight.dtype, "F32");
        assert_eq!(weight.shape, vec![768, 768]);
        assert_eq!(weight.data_length, 2359296);
    }

    #[tokio::test]
    async fn test_tensor_streaming() {
        let config = BlockStoreConfig {
            path: std::path::PathBuf::from("/tmp/ipfrs-test-tensorswap-stream"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&config.path);
        let store = Arc::new(SledBlockStore::new(config).unwrap());
        let tensorswap = TensorSwap::with_defaults(store).unwrap();

        let cid = test_cid();
        let metadata = TensorMetadata::new(cid)
            .with_shape(vec![1024, 1024])
            .with_dtype("f32")
            .with_size(4 * 1024 * 1024);

        tensorswap.register_tensor(metadata);

        // Start streaming
        tensorswap.start_stream(cid).unwrap();

        let stats = tensorswap.stats();
        assert_eq!(stats.active_streams, 1);
        assert!(!stats.backpressure_paused);
    }

    #[test]
    fn test_einsum_expression_parse() {
        // Matrix multiplication
        let expr = EinsumExpression::parse("ij,jk->ik").unwrap();
        assert_eq!(expr.num_inputs(), 2);
        assert_eq!(expr.inputs[0], "ij");
        assert_eq!(expr.inputs[1], "jk");
        assert_eq!(expr.output, "ik");
        assert!(!expr.is_transpose());
        assert!(!expr.is_reduction());

        let shared = expr.shared_indices();
        assert_eq!(shared.len(), 1);
        assert!(shared.contains(&'j'));

        // Reduction
        let expr2 = EinsumExpression::parse("ij->i").unwrap();
        assert!(expr2.is_reduction());
        assert_eq!(expr2.num_inputs(), 1);

        // Transpose
        let expr3 = EinsumExpression::parse("ij->ji").unwrap();
        assert!(expr3.is_transpose());
    }

    #[test]
    fn test_einsum_graph() {
        let mut graph = EinsumGraph::new();

        let cid_a = test_cid();
        let cid_b = test_cid2();
        let cid_c: Cid = "bafybeibxm2nsadl3fnxv2sxcxmxaco2jl53wpeorjdziber7rnz5gvv5h4"
            .parse()
            .unwrap();

        // Register tensors
        graph.register_tensor("A", cid_a);
        graph.register_tensor("B", cid_b);
        graph.register_tensor("C", cid_c);

        // Add expressions: A and B are inputs, C = A @ B
        let expr = EinsumExpression::parse("ij,jk->ik").unwrap();
        let mut expr_with_names = expr.clone();
        expr_with_names.inputs = vec!["A".to_string(), "B".to_string()];
        expr_with_names.output = "C".to_string();

        graph.add_expression(expr_with_names);

        // Check dependencies
        let deps = graph.get_dependencies("C").unwrap();
        assert_eq!(deps.len(), 2);
        assert!(deps.contains(&cid_a));
        assert!(deps.contains(&cid_b));

        // Check priorities (A and B are leaves, should have higher priority than C)
        let priority_a = graph.compute_priority("A");
        let priority_c = graph.compute_priority("C");
        assert!(priority_a > priority_c);

        // Check topological order
        let order = graph.topological_order().unwrap();
        assert_eq!(order.len(), 3);

        // A and B should come before C
        let pos_c = order.iter().position(|(name, _)| name == "C").unwrap();
        let pos_a = order.iter().position(|(name, _)| name == "A").unwrap();
        let pos_b = order.iter().position(|(name, _)| name == "B").unwrap();
        assert!(pos_a < pos_c);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_einsum_metadata_generation() {
        let mut graph = EinsumGraph::new();

        let cid_a = test_cid();
        let cid_b = test_cid2();
        let cid_c: Cid = "bafybeibxm2nsadl3fnxv2sxcxmxaco2jl53wpeorjdziber7rnz5gvv5h4"
            .parse()
            .unwrap();

        graph.register_tensor("A", cid_a);
        graph.register_tensor("B", cid_b);
        graph.register_tensor("C", cid_c);

        let expr = EinsumExpression {
            expression: "ij,jk->ik".to_string(),
            inputs: vec!["A".to_string(), "B".to_string()],
            output: "C".to_string(),
        };
        graph.add_expression(expr);

        // Generate metadata for C
        let metadata = graph.generate_metadata("C").unwrap();
        assert_eq!(metadata.cid, cid_c);
        assert_eq!(metadata.dependencies.len(), 2);
        assert!(metadata.priority_hint.is_some());
    }

    #[tokio::test]
    async fn test_backpressure_integration() {
        let ts_config = TensorSwapConfig {
            max_concurrent_streams: 2,
            ..Default::default()
        };

        let store_config = BlockStoreConfig {
            path: std::path::PathBuf::from("/tmp/ipfrs-test-tensorswap-bp"),
            cache_size: 100 * 1024 * 1024,
        };
        let _ = std::fs::remove_dir_all(&store_config.path);
        let store = Arc::new(SledBlockStore::new(store_config).unwrap());
        let tensorswap = TensorSwap::new(store, ts_config).unwrap();

        let cid1 = test_cid();
        let cid2 = test_cid2();

        // Start two streams (at limit)
        tensorswap.start_stream(cid1).unwrap();
        tensorswap.start_stream(cid2).unwrap();

        // Third stream should fail (at limit)
        let cid3: Cid = "bafybeibxm2nsadl3fnxv2sxcxmxaco2jl53wpeorjdziber7rnz5gvv5h4"
            .parse()
            .unwrap();
        let result = tensorswap.start_stream(cid3);
        assert!(result.is_err());

        let stats = tensorswap.stats();
        assert_eq!(stats.active_streams, 2);
    }
}
