//! Streaming types for TensorSwap: metadata, chunk management, backpressure, and request queuing.

use crate::want_list::Priority;
use ipfrs_core::Cid;
use std::collections::{HashMap, VecDeque};
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
    pub fn parse(data: &[u8]) -> ipfrs_core::Result<Self> {
        if data.len() < 8 {
            return Err(ipfrs_core::Error::Deserialization(
                "Safetensors header too short".to_string(),
            ));
        }

        // First 8 bytes are header size (little-endian u64)
        let header_size =
            u64::from_le_bytes(data[0..8].try_into().map_err(|_| {
                ipfrs_core::Error::Deserialization("header size bytes".to_string())
            })?);

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
        vec.sort_by_key(|v| std::cmp::Reverse(v.priority));
        self.requests = vec.into();
    }
}
