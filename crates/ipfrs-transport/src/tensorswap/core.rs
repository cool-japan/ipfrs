//! TensorSwap core: `TensorSwap<S>`, `TensorSwapConfig`, `TensorSwapStats`, and schema negotiation.

use crate::bitswap::{BitswapConfig, BitswapExchange};
use crate::peer_manager::PeerId;
use crate::schema_registry::{SchemaError, SchemaEvolutionFrame, SchemaRegistry, SchemaVersion};
use crate::want_list::Priority;
use ipfrs_core::{Block, Cid, Result};
use ipfrs_storage::traits::BlockStore;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tokio::sync::mpsc;

use super::streaming::{
    BackpressureConfig, BackpressureController, StreamProgress, TensorMetadata, TensorStream,
};

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
        if let Ok(mut map) = self.tensor_metadata.write() {
            map.insert(metadata.cid, metadata);
        }
    }

    /// Request a tensor with dependency-aware scheduling
    pub fn want_tensor(&self, cid: Cid) -> Result<()> {
        let metadata = self
            .tensor_metadata
            .read()
            .map_err(|_| ipfrs_core::Error::Internal("lock poisoned".to_string()))?;

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
        {
            let bp = self
                .backpressure
                .read()
                .map_err(|_| ipfrs_core::Error::Internal("lock poisoned".to_string()))?;
            if !bp.should_accept() {
                return Err(ipfrs_core::Error::Internal(
                    "Backpressure limit reached".to_string(),
                ));
            }
        }

        // Check concurrent stream limit
        {
            let active_count = self
                .active_streams
                .read()
                .map_err(|_| ipfrs_core::Error::Internal("lock poisoned".to_string()))?
                .len();
            if active_count >= self.config.max_concurrent_streams {
                return Err(ipfrs_core::Error::Internal(
                    "Maximum concurrent streams reached".to_string(),
                ));
            }
        }

        let metadata = {
            let meta_map = self
                .tensor_metadata
                .read()
                .map_err(|_| ipfrs_core::Error::Internal("lock poisoned".to_string()))?;
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
        self.active_streams
            .write()
            .map_err(|_| ipfrs_core::Error::Internal("lock poisoned".to_string()))?
            .insert(cid, stream);

        Ok(())
    }

    /// Start streaming with progress channel
    pub fn start_stream_with_progress(
        &self,
        cid: Cid,
        progress_tx: mpsc::Sender<StreamProgress>,
    ) -> Result<()> {
        // Check backpressure
        {
            let bp = self
                .backpressure
                .read()
                .map_err(|_| ipfrs_core::Error::Internal("lock poisoned".to_string()))?;
            if !bp.should_accept() {
                return Err(ipfrs_core::Error::Internal(
                    "Backpressure limit reached".to_string(),
                ));
            }
        }

        // Check concurrent stream limit
        {
            let active_count = self
                .active_streams
                .read()
                .map_err(|_| ipfrs_core::Error::Internal("lock poisoned".to_string()))?
                .len();
            if active_count >= self.config.max_concurrent_streams {
                return Err(ipfrs_core::Error::Internal(
                    "Maximum concurrent streams reached".to_string(),
                ));
            }
        }

        let metadata = {
            let meta_map = self
                .tensor_metadata
                .read()
                .map_err(|_| ipfrs_core::Error::Internal("lock poisoned".to_string()))?;
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

        self.active_streams
            .write()
            .map_err(|_| ipfrs_core::Error::Internal("lock poisoned".to_string()))?
            .insert(cid, stream);

        Ok(())
    }

    /// Get stream progress
    pub fn stream_progress(&self, cid: &Cid) -> Option<f64> {
        self.active_streams
            .read()
            .ok()?
            .get(cid)
            .map(|s| s.progress())
    }

    /// Check if stream is complete
    pub fn is_stream_complete(&self, cid: &Cid) -> bool {
        self.active_streams
            .read()
            .ok()
            .and_then(|s| s.get(cid).map(|s| s.is_complete()))
            .unwrap_or(false)
    }

    /// Cancel a tensor stream
    pub fn cancel_stream(&self, cid: &Cid) -> Result<()> {
        let stream = self
            .active_streams
            .write()
            .map_err(|_| ipfrs_core::Error::Internal("lock poisoned".to_string()))?
            .remove(cid);
        if let Some(stream) = stream {
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
            let mut streams = self
                .active_streams
                .write()
                .map_err(|_| ipfrs_core::Error::Internal("lock poisoned".to_string()))?;
            for stream in streams.values_mut() {
                stream.mark_received(&cid, size).await;
            }
        }

        // Update backpressure
        if let Ok(mut bp) = self.backpressure.write() {
            bp.on_ack(size as usize);
        }

        self.bitswap.receive_block(peer_id, block).await
    }

    /// Send tensor block to peer
    pub async fn send_tensor(
        &self,
        peer_id: &PeerId,
        cid: &Cid,
    ) -> Result<Option<crate::messages::Message>> {
        // Check backpressure before sending
        let should_send = self
            .backpressure
            .read()
            .map_err(|_| ipfrs_core::Error::Internal("lock poisoned".to_string()))?
            .should_accept();
        if !should_send {
            return Err(ipfrs_core::Error::Internal(
                "Backpressure active".to_string(),
            ));
        }

        let result = self.bitswap.send_block(peer_id, cid).await?;

        // Update backpressure on send
        if let Some(crate::messages::Message::Block(block_msg)) = &result {
            if let Ok(mut bp) = self.backpressure.write() {
                bp.on_send(block_msg.data.len());
            }
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
        if let Ok(mut streams) = self.active_streams.write() {
            streams.retain(|_, stream| !stream.is_complete());
        }
    }

    /// Get statistics
    pub fn stats(&self) -> TensorSwapStats {
        let bitswap_stats = self.bitswap.stats();
        let streams = self.active_streams.read();
        let backpressure = self.backpressure.read();

        let (active_streams, backpressure_paused, pending_chunks) =
            match (streams.as_ref(), backpressure.as_ref()) {
                (Ok(s), Ok(bp)) => (s.len(), bp.is_paused(), bp.pending_count()),
                _ => (0, false, 0),
            };

        TensorSwapStats {
            want_list_size: bitswap_stats.want_list_size,
            num_tensors_registered: self.tensor_metadata.read().map(|m| m.len()).unwrap_or(0),
            active_streams,
            total_bytes_sent: bitswap_stats.total_bytes_sent,
            total_bytes_recv: bitswap_stats.total_bytes_recv,
            backpressure_paused,
            pending_chunks,
        }
    }

    /// Access underlying Bitswap for advanced usage
    pub fn bitswap(&self) -> &Arc<BitswapExchange<S>> {
        &self.bitswap
    }

    /// Check if backpressure is active
    pub fn is_backpressure_active(&self) -> bool {
        self.backpressure
            .read()
            .map(|bp| bp.is_paused())
            .unwrap_or(false)
    }

    /// Acknowledge data for backpressure
    pub fn ack_data(&self, bytes: usize) {
        if let Ok(mut bp) = self.backpressure.write() {
            bp.on_ack(bytes);
        }
    }

    /// Negotiate schema with a remote peer at session establishment.
    ///
    /// The caller proposes a [`SchemaVersion`]; this method validates that the
    /// version is registered and returns the version that both sides agree on.
    /// If `proposed` is unknown in `registry` a [`SchemaError::NotFound`] is
    /// returned and the caller should fall back to the latest registered
    /// version (or abort the session).
    ///
    /// In a real network implementation the agreed version would be exchanged
    /// over the wire; here we perform the validation locally and return the
    /// negotiated version for the caller to use when encoding IPC batches.
    pub async fn negotiate_schema(
        &mut self,
        proposed: &SchemaVersion,
        registry: &SchemaRegistry,
    ) -> std::result::Result<SchemaVersion, SchemaError> {
        // Confirm the proposed version is actually registered.
        registry
            .get(proposed)
            .ok_or_else(|| SchemaError::NotFound(proposed.name.clone()))?;

        // Determine the latest version known locally.
        let latest = registry
            .latest_version(&proposed.name)
            .ok_or_else(|| SchemaError::NotFound(proposed.name.clone()))?;

        // If the remote proposes an older version and the local registry can
        // read it, accept the proposed version so the sender does not need to
        // upgrade immediately.
        if registry.can_read_with(proposed, &latest) {
            Ok(proposed.clone())
        } else {
            // The proposed version is newer than what we know — request downgrade
            // to the latest version we understand.
            Ok(latest)
        }
    }

    /// Evolve the session schema mid-stream.
    ///
    /// Constructs a [`SchemaEvolutionFrame`] for `new_version` and returns it
    /// as serialised bytes ready to be sent to the remote peer before the next
    /// Arrow IPC data batch.  The caller is responsible for transmitting the
    /// bytes on the wire and waiting for an acknowledgement before continuing
    /// with the new schema.
    pub async fn evolve_schema(
        &mut self,
        session_id: &str,
        old_version: &SchemaVersion,
        new_version: &SchemaVersion,
        registry: &SchemaRegistry,
    ) -> std::result::Result<Vec<u8>, SchemaError> {
        // Both versions must be registered.
        let _old_schema = registry
            .get(old_version)
            .ok_or_else(|| SchemaError::NotFound(old_version.name.clone()))?;

        let new_schema = registry
            .get(new_version)
            .ok_or_else(|| SchemaError::NotFound(new_version.name.clone()))?;

        let frame = SchemaEvolutionFrame::new(
            session_id,
            old_version.clone(),
            new_version.clone(),
            &new_schema,
        );

        frame.to_bytes()
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
        let store = Arc::new(
            ipfrs_storage::SledBlockStore::new(config)
                .expect("SledBlockStore::new with default config"),
        );
        Self::with_defaults(store).expect("TensorSwap::with_defaults")
    }
}
