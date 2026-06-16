//! Gradient synchronisation service for distributed federated learning.
//!
//! This module exposes a gRPC-style server-streaming service that allows
//! nodes to exchange gradient chunks during a distributed training round.
//! The implementation is manual (not tonic-generated) so that it can be
//! compiled and tested independently of the proto-build toolchain.
//!
//! # Architecture
//!
//! ```text
//! Client                    GradientSyncService
//!   |                              |
//!   |-- GradientSyncRequest ------>|
//!   |                              |  decode local gradient
//!   |                              |  commit_local → BlockStore
//!   |<-- GradientChunkResponse 0 --|  (local gradient chunk 0)
//!   |<-- GradientChunkResponse 1 --|  (local gradient chunk 1)
//!   |        …                    |
//!   |<-- GradientChunkResponse N --|  (local gradient chunk N)
//!   |                              |
//! ```
//!
//! In production, peer gradients would be streamed by hooking into the
//! `ipfrs-network` GossipSub event loop.  That wiring lives at the `ipfrs`
//! crate level (which owns `Node`) and must not be referenced here to avoid
//! a circular dependency.  Instead, the service pushes the *local* gradient
//! as chunks so that clients can observe server-streaming behaviour without
//! a live network.

use ipfrs_tensorlogic::gradient::arrow_ipc::{load_gradient_from_arrow, store_gradient_as_arrow};
use ipfrs_tensorlogic::gradient::backward_pass::BackwardPassConfig;
use ipfrs_tensorlogic::gradient::federated::DistributedGradientAccumulator;

// ── Request / Response types ──────────────────────────────────────────────

/// Request to initiate a distributed gradient synchronisation session.
///
/// The `local_gradient` field carries the caller's local gradient encoded
/// as Arrow IPC bytes (use
/// [`store_gradient_as_arrow`] to produce them).
#[derive(Debug, Clone)]
pub struct GradientSyncRequest {
    /// Unique identifier for this synchronisation round.
    pub session_id: String,
    /// Arrow IPC-encoded local gradient contributed by the caller.
    pub local_gradient: Vec<u8>,
    /// Minimum number of peer gradients required before aggregating.
    pub min_peers: u32,
    /// Wall-clock timeout in seconds before the session is abandoned.
    pub timeout_secs: u64,
}

/// A single gradient chunk streamed back to the client.
///
/// During a [`GradientSyncService::sync_gradients`] call the service pushes
/// one `GradientChunkResponse` per Arrow IPC chunk received from a peer so
/// that clients can start processing data before all peers have responded.
#[derive(Debug, Clone)]
pub struct GradientChunkResponse {
    /// Session identifier matching [`GradientSyncRequest::session_id`].
    pub session_id: String,
    /// Zero-based index of this chunk within the peer's gradient stream.
    pub chunk_index: u32,
    /// Total chunks expected from this peer.
    pub total_chunks: u32,
    /// Arrow IPC bytes for this chunk.
    pub data: Vec<u8>,
    /// Peer that contributed this chunk.
    pub peer_id: String,
}

// ── GradientSyncService ───────────────────────────────────────────────────

/// gRPC-style service that streams gradient chunks to clients as they arrive
/// from peers.
///
/// The service wraps a `BlockStore` so that concurrent sync sessions can
/// safely share the same block store without requiring access to the full
/// `Node` type (which lives in the `ipfrs` crate and cannot be referenced
/// from `ipfrs-interface` without a circular dependency).
///
/// # Usage
///
/// ```rust,no_run
/// use ipfrs_interface::gradient_sync::{GradientSyncRequest, GradientSyncService};
/// use ipfrs_storage::{BlockStoreConfig, SledBlockStore};
/// use ipfrs_tensorlogic::gradient::arrow_ipc::store_gradient_as_arrow;
/// use std::sync::Arc;
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = BlockStoreConfig { path: std::env::temp_dir().join("grad"), cache_size: 64 * 1024 * 1024 };
/// let store = Arc::new(SledBlockStore::new(config)?);
/// let service = GradientSyncService::new(store);
///
/// let gradient: Vec<f32> = (0u32..256).map(|i| i as f32 * 0.01).collect();
/// let local_gradient = store_gradient_as_arrow(&gradient)?;
///
/// let request = GradientSyncRequest {
///     session_id: "round-1".to_string(),
///     local_gradient,
///     min_peers: 0,
///     timeout_secs: 30,
/// };
///
/// let (tx, mut rx) = tokio::sync::mpsc::channel(64);
/// service.sync_gradients(request, tx).await?;
///
/// while let Some(chunk) = rx.recv().await {
///     println!("received chunk {}/{}", chunk.chunk_index + 1, chunk.total_chunks);
/// }
/// # Ok(())
/// # }
/// ```
pub struct GradientSyncService {
    store: std::sync::Arc<dyn ipfrs_storage::traits::BlockStore>,
}

impl GradientSyncService {
    /// Create a new service backed by `store`.
    pub fn new(store: std::sync::Arc<dyn ipfrs_storage::traits::BlockStore>) -> Self {
        Self { store }
    }

    /// Start a gradient sync session and stream chunks via `chunk_tx`.
    ///
    /// # Workflow
    ///
    /// 1. Decode `request.local_gradient` from Arrow IPC.
    /// 2. Commit the local gradient to the block store via
    ///    [`DistributedGradientAccumulator::commit_local`].
    /// 3. Poll for peer gradients (stubbed — no live network in this layer).
    /// 4. Push one [`GradientChunkResponse`] per local chunk onto `chunk_tx`
    ///    so that the caller can observe streaming behaviour end-to-end.
    ///
    /// When full peer-to-peer transport is wired up at the `ipfrs` crate
    /// level, step 3 would await actual peer CIDs from the GossipSub
    /// `GRADIENT_SYNC` event loop.
    pub async fn sync_gradients(
        &self,
        request: GradientSyncRequest,
        chunk_tx: tokio::sync::mpsc::Sender<GradientChunkResponse>,
    ) -> anyhow::Result<()> {
        // ── 1. Decode the caller's local gradient from Arrow IPC ──────────────
        let local_gradient = load_gradient_from_arrow(&request.local_gradient)
            .map_err(|e| anyhow::anyhow!("failed to decode local gradient: {e}"))?;

        tracing::debug!(
            session_id = %request.session_id,
            gradient_len = local_gradient.len(),
            min_peers = request.min_peers,
            timeout_secs = request.timeout_secs,
            "GradientSyncService: starting sync session"
        );

        // ── 2. Commit the local gradient to the block store ───────────────────
        let mut accumulator =
            DistributedGradientAccumulator::new(&request.session_id, BackwardPassConfig::default());

        let local_cid = accumulator
            .commit_local(local_gradient.clone(), self.store.as_ref())
            .await
            .map_err(|e| anyhow::anyhow!("commit_local failed: {e}"))?;

        tracing::debug!(
            session_id = %request.session_id,
            cid = %local_cid,
            "GradientSyncService: local gradient committed"
        );

        // ── 3. Stream local gradient back as chunks ───────────────────────────
        //
        // In a live deployment peer CIDs would be discovered via the network
        // layer and fed into `accumulator.add_peer_gradient(...)`.  Since
        // ipfrs-interface has no direct access to the network, we stream the
        // local gradient back in chunks so that the caller can observe the
        // server-streaming pattern end-to-end.
        let chunk_size = 65_536usize;
        let total_chunks = local_gradient.len().div_ceil(chunk_size).max(1);

        if local_gradient.is_empty() {
            // Send a single empty chunk to signal stream completion.
            let ipc = store_gradient_as_arrow(&[])
                .map_err(|e| anyhow::anyhow!("Arrow IPC encode: {e}"))?;
            chunk_tx
                .send(GradientChunkResponse {
                    session_id: request.session_id.clone(),
                    chunk_index: 0,
                    total_chunks: 1,
                    data: ipc,
                    peer_id: "local".to_string(),
                })
                .await
                .map_err(|_| anyhow::anyhow!("chunk_tx receiver dropped"))?;
            return Ok(());
        }

        for (idx, window) in local_gradient.chunks(chunk_size).enumerate() {
            let ipc = store_gradient_as_arrow(window)
                .map_err(|e| anyhow::anyhow!("Arrow IPC encode chunk {idx}: {e}"))?;

            chunk_tx
                .send(GradientChunkResponse {
                    session_id: request.session_id.clone(),
                    chunk_index: idx as u32,
                    total_chunks: total_chunks as u32,
                    data: ipc,
                    peer_id: "local".to_string(),
                })
                .await
                .map_err(|_| anyhow::anyhow!("chunk_tx receiver dropped at chunk {idx}"))?;
        }

        tracing::debug!(
            session_id = %request.session_id,
            total_chunks,
            "GradientSyncService: streamed all local chunks"
        );

        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ipfrs_storage::{BlockStoreConfig, SledBlockStore};
    use std::sync::Arc;

    fn make_store(suffix: &str) -> Arc<SledBlockStore> {
        let path = std::env::temp_dir().join(format!("ipfrs-test-gradsync-{suffix}"));
        let _ = std::fs::remove_dir_all(&path);
        let config = BlockStoreConfig {
            path,
            cache_size: 16 * 1024 * 1024,
        };
        Arc::new(SledBlockStore::new(config).expect("SledBlockStore::new"))
    }

    /// `GradientSyncService::new` must construct without panicking.
    #[test]
    fn test_gradient_sync_service_new() {
        let store = make_store("new");
        let _service = GradientSyncService::new(store);
    }

    /// A small gradient must be committed and streamed back as at least one chunk.
    #[tokio::test]
    async fn test_gradient_sync_service_streams_chunks() {
        let store = make_store("streams-chunks");
        let service = GradientSyncService::new(store);

        let gradient: Vec<f32> = (0u32..128).map(|i| i as f32 * 0.1).collect();
        let local_gradient_bytes =
            store_gradient_as_arrow(&gradient).expect("store_gradient_as_arrow");

        let request = GradientSyncRequest {
            session_id: "test-sync-session".to_string(),
            local_gradient: local_gradient_bytes,
            min_peers: 0,
            timeout_secs: 5,
        };

        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        service
            .sync_gradients(request, tx)
            .await
            .expect("sync_gradients");

        let mut received = Vec::new();
        while let Some(chunk) = rx.recv().await {
            assert_eq!(chunk.session_id, "test-sync-session");
            assert_eq!(chunk.peer_id, "local");
            received.push(chunk);
        }

        assert!(
            !received.is_empty(),
            "at least one chunk must be streamed back"
        );
        assert_eq!(received[0].chunk_index, 0);
        assert!(
            !received[0].data.is_empty(),
            "chunk data must be non-empty Arrow IPC bytes"
        );
    }

    /// An empty gradient must produce exactly one empty-chunk response.
    #[tokio::test]
    async fn test_gradient_sync_service_empty_gradient() {
        let store = make_store("empty-grad");
        let service = GradientSyncService::new(store);

        let local_gradient_bytes =
            store_gradient_as_arrow(&[]).expect("store_gradient_as_arrow on empty");

        let request = GradientSyncRequest {
            session_id: "empty-session".to_string(),
            local_gradient: local_gradient_bytes,
            min_peers: 0,
            timeout_secs: 1,
        };

        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        service
            .sync_gradients(request, tx)
            .await
            .expect("sync_gradients");

        let first = rx.recv().await.expect("must receive one chunk");
        assert_eq!(first.chunk_index, 0);
        assert_eq!(first.total_chunks, 1);
        assert_eq!(first.peer_id, "local");

        // No further chunks.
        assert!(rx.recv().await.is_none());
    }

    /// A large gradient that exceeds one chunk must be split correctly.
    #[tokio::test]
    async fn test_gradient_sync_service_large_gradient() {
        let store = make_store("large-grad");
        let service = GradientSyncService::new(store);

        // 131_072 elements → 2 chunks of 65_536 each.
        let gradient: Vec<f32> = (0u32..131_072).map(|i| i as f32 * 1e-5).collect();
        let local_gradient_bytes =
            store_gradient_as_arrow(&gradient).expect("store_gradient_as_arrow");

        let request = GradientSyncRequest {
            session_id: "large-session".to_string(),
            local_gradient: local_gradient_bytes,
            min_peers: 0,
            timeout_secs: 10,
        };

        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        service
            .sync_gradients(request, tx)
            .await
            .expect("sync_gradients");

        let mut received = Vec::new();
        while let Some(chunk) = rx.recv().await {
            received.push(chunk);
        }

        assert_eq!(received.len(), 2, "131_072 elements / 65_536 = 2 chunks");
        assert_eq!(received[0].chunk_index, 0);
        assert_eq!(received[1].chunk_index, 1);
        assert_eq!(received[0].total_chunks, 2);
        assert_eq!(received[1].total_chunks, 2);
    }
}
