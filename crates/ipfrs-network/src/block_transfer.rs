//! Streaming block transfer management with chunked, resumable transfers.
//!
//! Provides [`StreamingBlockTransfer`] for managing chunked, resumable block
//! transfers between peers with progress tracking and checksum verification.

use std::collections::HashMap;

/// Direction of a block transfer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TransferDirection {
    /// Uploading data to a remote peer.
    Upload,
    /// Downloading data from a remote peer.
    Download,
}

/// A single chunk of data in a block transfer.
#[derive(Clone, Debug)]
pub struct TransferChunk {
    /// Zero-based index of this chunk in the overall transfer.
    pub chunk_index: u32,
    /// Raw bytes of this chunk.
    pub data: Vec<u8>,
    /// Simple checksum: XOR of all bytes cast to u32.
    pub checksum: u32,
}

impl TransferChunk {
    /// Compute the expected checksum for the contained data.
    pub fn compute_checksum(data: &[u8]) -> u32 {
        data.iter().fold(0u32, |acc, &b| acc ^ u32::from(b))
    }

    /// Return `true` if the stored checksum matches the data.
    pub fn is_valid(&self) -> bool {
        self.checksum == Self::compute_checksum(&self.data)
    }
}

/// Lifecycle state of a [`BlockTransfer`].
#[derive(Clone, Debug, PartialEq)]
pub enum TransferState {
    /// Transfer has been created but not yet started.
    Pending,
    /// Transfer is actively exchanging chunks.
    InProgress {
        /// Number of chunks successfully processed so far.
        chunks_done: u32,
        /// Total number of chunks in the transfer.
        total_chunks: u32,
    },
    /// Transfer has been paused mid-way.
    Paused {
        /// Number of chunks processed before pausing.
        chunks_done: u32,
    },
    /// All chunks have been received/sent successfully.
    Completed,
    /// Transfer encountered an unrecoverable error.
    Failed {
        /// Human-readable description of the failure.
        reason: String,
    },
}

/// A single block transfer between the local node and a remote peer.
#[derive(Clone, Debug)]
pub struct BlockTransfer {
    /// Unique identifier for this transfer.
    pub transfer_id: u64,
    /// Content identifier of the block being transferred.
    pub cid: String,
    /// Identifier of the remote peer.
    pub peer_id: String,
    /// Whether this node is uploading or downloading.
    pub direction: TransferDirection,
    /// Total size of the block in bytes.
    pub total_bytes: u64,
    /// Maximum bytes per chunk (default 65536).
    pub chunk_size: u64,
    /// Current lifecycle state of the transfer.
    pub state: TransferState,
    /// Indices of chunks that have been received or sent.
    pub received_chunks: Vec<u32>,
    /// Tick at which the transfer was started.
    pub started_at_tick: u64,
}

impl BlockTransfer {
    /// Return the total number of chunks required for this transfer.
    ///
    /// Uses ceiling division so that a partial last chunk is counted.
    pub fn total_chunks(&self) -> u32 {
        if self.chunk_size == 0 {
            return 0;
        }
        self.total_bytes.div_ceil(self.chunk_size) as u32
    }

    /// Return transfer progress as a fraction in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when the transfer is [`TransferState::Pending`].
    pub fn progress(&self) -> f64 {
        let total = self.total_chunks();
        if total == 0 {
            return 1.0;
        }
        let done = match &self.state {
            TransferState::Pending => 0,
            TransferState::InProgress { chunks_done, .. } => *chunks_done,
            TransferState::Paused { chunks_done } => *chunks_done,
            TransferState::Completed => total,
            TransferState::Failed { .. } => self.received_chunks.len() as u32,
        };
        f64::from(done) / f64::from(total)
    }

    /// Return `true` if the transfer has reached [`TransferState::Completed`].
    pub fn is_complete(&self) -> bool {
        self.state == TransferState::Completed
    }
}

/// Aggregated statistics for all transfers managed by a [`StreamingBlockTransfer`].
#[derive(Clone, Debug, Default)]
pub struct TransferManagerStats {
    /// Number of transfers that are currently active (Pending, InProgress, or Paused).
    pub active_transfers: usize,
    /// Total number of transfers that reached [`TransferState::Completed`].
    pub completed_transfers: u64,
    /// Total number of transfers that reached [`TransferState::Failed`].
    pub failed_transfers: u64,
    /// Cumulative bytes transferred across all completed transfers.
    pub total_bytes_transferred: u64,
}

/// Manager for chunked, resumable block transfers between peers.
///
/// Each transfer is identified by a unique `u64` transfer ID returned by
/// [`StreamingBlockTransfer::start_transfer`].
pub struct StreamingBlockTransfer {
    /// All tracked transfers indexed by their transfer ID.
    pub transfers: HashMap<u64, BlockTransfer>,
    /// Counter used to assign the next unique transfer ID.
    pub next_id: u64,
    /// Monotonically increasing logical clock.
    pub current_tick: u64,
}

impl StreamingBlockTransfer {
    /// Create a new, empty transfer manager.
    pub fn new() -> Self {
        Self {
            transfers: HashMap::new(),
            next_id: 1,
            current_tick: 0,
        }
    }

    /// Register a new transfer and return its assigned transfer ID.
    ///
    /// The transfer begins in the [`TransferState::Pending`] state.
    /// If `chunk_size` is `0`, it defaults to `65536`.
    pub fn start_transfer(
        &mut self,
        cid: String,
        peer_id: String,
        direction: TransferDirection,
        total_bytes: u64,
        chunk_size: u64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let effective_chunk_size = if chunk_size == 0 { 65536 } else { chunk_size };

        let transfer = BlockTransfer {
            transfer_id: id,
            cid,
            peer_id,
            direction,
            total_bytes,
            chunk_size: effective_chunk_size,
            state: TransferState::Pending,
            received_chunks: Vec::new(),
            started_at_tick: self.current_tick,
        };

        self.transfers.insert(id, transfer);
        id
    }

    /// Process an incoming chunk for the given transfer.
    ///
    /// # Errors
    ///
    /// Returns an error string when:
    /// - The transfer ID is unknown.
    /// - The transfer is not in an `InProgress` or `Pending` state.
    /// - The chunk checksum does not match its data.
    /// - The chunk index has already been received.
    pub fn receive_chunk(&mut self, transfer_id: u64, chunk: TransferChunk) -> Result<(), String> {
        let transfer = self
            .transfers
            .get_mut(&transfer_id)
            .ok_or_else(|| format!("unknown transfer id: {transfer_id}"))?;

        // Validate state: only Pending and InProgress accept new chunks.
        match &transfer.state {
            TransferState::Paused { .. } => {
                return Err(format!("transfer {transfer_id} is paused"));
            }
            TransferState::Completed => {
                return Err(format!("transfer {transfer_id} is already completed"));
            }
            TransferState::Failed { reason } => {
                return Err(format!("transfer {transfer_id} has failed: {reason}"));
            }
            TransferState::Pending | TransferState::InProgress { .. } => {}
        }

        // Validate checksum.
        if !chunk.is_valid() {
            return Err(format!(
                "checksum mismatch for chunk {} of transfer {transfer_id}: expected {}, got {}",
                chunk.chunk_index,
                TransferChunk::compute_checksum(&chunk.data),
                chunk.checksum,
            ));
        }

        // Reject duplicates.
        if transfer.received_chunks.contains(&chunk.chunk_index) {
            return Err(format!(
                "chunk {} already received for transfer {transfer_id}",
                chunk.chunk_index
            ));
        }

        transfer.received_chunks.push(chunk.chunk_index);
        let chunks_done = transfer.received_chunks.len() as u32;
        let total = transfer.total_chunks();

        if chunks_done >= total {
            transfer.state = TransferState::Completed;
        } else {
            transfer.state = TransferState::InProgress {
                chunks_done,
                total_chunks: total,
            };
        }

        Ok(())
    }

    /// Pause an active or pending transfer.
    ///
    /// Returns `true` if the transfer was successfully paused, `false` otherwise
    /// (e.g., the transfer ID is unknown, or the transfer is already paused/done).
    pub fn pause(&mut self, transfer_id: u64) -> bool {
        let Some(transfer) = self.transfers.get_mut(&transfer_id) else {
            return false;
        };

        let chunks_done = match &transfer.state {
            TransferState::Pending => 0,
            TransferState::InProgress { chunks_done, .. } => *chunks_done,
            _ => return false,
        };

        transfer.state = TransferState::Paused { chunks_done };
        true
    }

    /// Resume a paused transfer.
    ///
    /// Returns `true` if the transfer was successfully resumed, `false` otherwise.
    pub fn resume(&mut self, transfer_id: u64) -> bool {
        let Some(transfer) = self.transfers.get_mut(&transfer_id) else {
            return false;
        };

        let chunks_done = match &transfer.state {
            TransferState::Paused { chunks_done } => *chunks_done,
            _ => return false,
        };

        let total_chunks = transfer.total_chunks();
        transfer.state = TransferState::InProgress {
            chunks_done,
            total_chunks,
        };
        true
    }

    /// Mark a transfer as failed with a descriptive reason.
    ///
    /// Returns `true` if the transfer was found and transitioned to
    /// [`TransferState::Failed`], `false` if the ID is unknown or the
    /// transfer has already completed/failed.
    pub fn fail(&mut self, transfer_id: u64, reason: String) -> bool {
        let Some(transfer) = self.transfers.get_mut(&transfer_id) else {
            return false;
        };

        match transfer.state {
            TransferState::Completed | TransferState::Failed { .. } => return false,
            _ => {}
        }

        transfer.state = TransferState::Failed { reason };
        true
    }

    /// Compute aggregated statistics across all managed transfers.
    pub fn stats(&self) -> TransferManagerStats {
        let mut stats = TransferManagerStats::default();

        for transfer in self.transfers.values() {
            match &transfer.state {
                TransferState::Pending
                | TransferState::InProgress { .. }
                | TransferState::Paused { .. } => {
                    stats.active_transfers += 1;
                }
                TransferState::Completed => {
                    stats.completed_transfers += 1;
                    stats.total_bytes_transferred += transfer.total_bytes;
                }
                TransferState::Failed { .. } => {
                    stats.failed_transfers += 1;
                }
            }
        }

        stats
    }

    /// Advance the logical clock by one tick.
    pub fn advance_tick(&mut self) {
        self.current_tick += 1;
    }
}

impl Default for StreamingBlockTransfer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(index: u32, data: Vec<u8>) -> TransferChunk {
        let checksum = TransferChunk::compute_checksum(&data);
        TransferChunk {
            chunk_index: index,
            data,
            checksum,
        }
    }

    // 1. start_transfer returns unique, incrementing IDs.
    #[test]
    fn test_start_transfer_unique_ids() {
        let mut mgr = StreamingBlockTransfer::new();
        let id1 = mgr.start_transfer(
            "cid1".into(),
            "peer1".into(),
            TransferDirection::Download,
            1024,
            512,
        );
        let id2 = mgr.start_transfer(
            "cid2".into(),
            "peer2".into(),
            TransferDirection::Upload,
            2048,
            512,
        );
        assert_ne!(id1, id2);
        assert!(id2 > id1);
    }

    // 2. New transfer starts in Pending state.
    #[test]
    fn test_start_transfer_pending_state() {
        let mut mgr = StreamingBlockTransfer::new();
        let id = mgr.start_transfer(
            "cid".into(),
            "peer".into(),
            TransferDirection::Download,
            100,
            64,
        );
        let t = &mgr.transfers[&id];
        assert_eq!(t.state, TransferState::Pending);
    }

    // 3. total_chunks calculation — exact multiple.
    #[test]
    fn test_total_chunks_exact_multiple() {
        let mut mgr = StreamingBlockTransfer::new();
        let id = mgr.start_transfer(
            "c".into(),
            "p".into(),
            TransferDirection::Download,
            65536,
            65536,
        );
        assert_eq!(mgr.transfers[&id].total_chunks(), 1);
    }

    // 4. total_chunks calculation — partial last chunk.
    #[test]
    fn test_total_chunks_partial() {
        let mut mgr = StreamingBlockTransfer::new();
        let id = mgr.start_transfer(
            "c".into(),
            "p".into(),
            TransferDirection::Download,
            65537,
            65536,
        );
        assert_eq!(mgr.transfers[&id].total_chunks(), 2);
    }

    // 5. total_chunks for zero bytes.
    #[test]
    fn test_total_chunks_zero_bytes() {
        let mut mgr = StreamingBlockTransfer::new();
        let id = mgr.start_transfer(
            "c".into(),
            "p".into(),
            TransferDirection::Download,
            0,
            65536,
        );
        assert_eq!(mgr.transfers[&id].total_chunks(), 0);
    }

    // 6. receive_chunk updates state to InProgress.
    #[test]
    fn test_receive_chunk_updates_state() {
        let mut mgr = StreamingBlockTransfer::new();
        // 2 chunks of 10 bytes each = 20 bytes total.
        let id = mgr.start_transfer("c".into(), "p".into(), TransferDirection::Download, 20, 10);
        let chunk = make_chunk(0, vec![1u8; 10]);
        mgr.receive_chunk(id, chunk)
            .expect("receive_chunk should succeed");
        match &mgr.transfers[&id].state {
            TransferState::InProgress {
                chunks_done,
                total_chunks,
            } => {
                assert_eq!(*chunks_done, 1);
                assert_eq!(*total_chunks, 2);
            }
            other => panic!("expected InProgress, got {other:?}"),
        }
    }

    // 7. receive_chunk with bad checksum returns Err.
    #[test]
    fn test_receive_chunk_checksum_mismatch() {
        let mut mgr = StreamingBlockTransfer::new();
        let id = mgr.start_transfer("c".into(), "p".into(), TransferDirection::Download, 10, 10);
        let chunk = TransferChunk {
            chunk_index: 0,
            data: vec![42u8; 10],
            checksum: 0xDEAD_BEEF, // deliberately wrong
        };
        let result = mgr.receive_chunk(id, chunk);
        assert!(result.is_err(), "expected error for checksum mismatch");
        assert!(result.unwrap_err().contains("checksum mismatch"));
    }

    // 8. All chunks received → Completed.
    #[test]
    fn test_all_chunks_received_completes_transfer() {
        let mut mgr = StreamingBlockTransfer::new();
        let id = mgr.start_transfer("c".into(), "p".into(), TransferDirection::Download, 30, 10);
        for i in 0..3u32 {
            let chunk = make_chunk(i, vec![i as u8; 10]);
            mgr.receive_chunk(id, chunk)
                .expect("chunk should be accepted");
        }
        assert_eq!(mgr.transfers[&id].state, TransferState::Completed);
        assert!(mgr.transfers[&id].is_complete());
    }

    // 9. Duplicate chunk rejected.
    #[test]
    fn test_duplicate_chunk_rejected() {
        let mut mgr = StreamingBlockTransfer::new();
        let id = mgr.start_transfer("c".into(), "p".into(), TransferDirection::Download, 20, 10);
        let chunk0 = make_chunk(0, vec![1u8; 10]);
        mgr.receive_chunk(id, chunk0.clone())
            .expect("first receive should succeed");
        let dup = make_chunk(0, vec![1u8; 10]);
        let result = mgr.receive_chunk(id, dup);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already received"));
    }

    // 10. pause transitions InProgress → Paused.
    #[test]
    fn test_pause_in_progress() {
        let mut mgr = StreamingBlockTransfer::new();
        let id = mgr.start_transfer("c".into(), "p".into(), TransferDirection::Download, 30, 10);
        let chunk = make_chunk(0, vec![0u8; 10]);
        mgr.receive_chunk(id, chunk)
            .expect("test: receive_chunk should accept valid chunk");
        assert!(mgr.pause(id));
        match &mgr.transfers[&id].state {
            TransferState::Paused { chunks_done } => assert_eq!(*chunks_done, 1),
            other => panic!("expected Paused, got {other:?}"),
        }
    }

    // 11. pause on Pending → Paused(0).
    #[test]
    fn test_pause_pending() {
        let mut mgr = StreamingBlockTransfer::new();
        let id = mgr.start_transfer("c".into(), "p".into(), TransferDirection::Download, 100, 10);
        assert!(mgr.pause(id));
        assert_eq!(
            mgr.transfers[&id].state,
            TransferState::Paused { chunks_done: 0 }
        );
    }

    // 12. resume transitions Paused → InProgress.
    #[test]
    fn test_resume_paused() {
        let mut mgr = StreamingBlockTransfer::new();
        let id = mgr.start_transfer("c".into(), "p".into(), TransferDirection::Download, 30, 10);
        let chunk = make_chunk(0, vec![7u8; 10]);
        mgr.receive_chunk(id, chunk)
            .expect("test: receive_chunk should accept valid chunk");
        mgr.pause(id);
        assert!(mgr.resume(id));
        match &mgr.transfers[&id].state {
            TransferState::InProgress {
                chunks_done,
                total_chunks,
            } => {
                assert_eq!(*chunks_done, 1);
                assert_eq!(*total_chunks, 3);
            }
            other => panic!("expected InProgress, got {other:?}"),
        }
    }

    // 13. resume returns false for non-paused transfer.
    #[test]
    fn test_resume_non_paused_returns_false() {
        let mut mgr = StreamingBlockTransfer::new();
        let id = mgr.start_transfer("c".into(), "p".into(), TransferDirection::Download, 10, 10);
        assert!(!mgr.resume(id)); // still Pending
    }

    // 14. fail sets Failed state.
    #[test]
    fn test_fail_sets_failed_state() {
        let mut mgr = StreamingBlockTransfer::new();
        let id = mgr.start_transfer("c".into(), "p".into(), TransferDirection::Download, 100, 10);
        assert!(mgr.fail(id, "network timeout".into()));
        match &mgr.transfers[&id].state {
            TransferState::Failed { reason } => assert_eq!(reason, "network timeout"),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    // 15. fail on completed transfer returns false.
    #[test]
    fn test_fail_on_completed_returns_false() {
        let mut mgr = StreamingBlockTransfer::new();
        let id = mgr.start_transfer("c".into(), "p".into(), TransferDirection::Download, 10, 10);
        let chunk = make_chunk(0, vec![0u8; 10]);
        mgr.receive_chunk(id, chunk)
            .expect("test: receive_chunk should accept valid chunk");
        assert!(mgr.transfers[&id].is_complete());
        assert!(!mgr.fail(id, "too late".into()));
    }

    // 16. progress before and after receiving chunks.
    #[test]
    fn test_progress_values() {
        let mut mgr = StreamingBlockTransfer::new();
        // 4 chunks of 10 bytes.
        let id = mgr.start_transfer("c".into(), "p".into(), TransferDirection::Download, 40, 10);
        assert_eq!(mgr.transfers[&id].progress(), 0.0);

        let chunk0 = make_chunk(0, vec![1u8; 10]);
        mgr.receive_chunk(id, chunk0)
            .expect("test: receive_chunk should accept chunk 0");
        let p = mgr.transfers[&id].progress();
        assert!((p - 0.25).abs() < f64::EPSILON, "expected 0.25 got {p}");

        let chunk1 = make_chunk(1, vec![2u8; 10]);
        mgr.receive_chunk(id, chunk1)
            .expect("test: receive_chunk should accept chunk 1");
        let p = mgr.transfers[&id].progress();
        assert!((p - 0.5).abs() < f64::EPSILON, "expected 0.5 got {p}");
    }

    // 17. stats counts active, completed, and failed correctly.
    #[test]
    fn test_stats_counts() {
        let mut mgr = StreamingBlockTransfer::new();

        // Two active (Pending).
        mgr.start_transfer(
            "c1".into(),
            "p1".into(),
            TransferDirection::Download,
            10,
            10,
        );
        mgr.start_transfer("c2".into(), "p2".into(), TransferDirection::Upload, 10, 10);

        // One completed.
        let id_done = mgr.start_transfer(
            "c3".into(),
            "p3".into(),
            TransferDirection::Download,
            10,
            10,
        );
        let chunk = make_chunk(0, vec![5u8; 10]);
        mgr.receive_chunk(id_done, chunk)
            .expect("test: receive_chunk should accept valid chunk for completed transfer");

        // One failed.
        let id_fail = mgr.start_transfer(
            "c4".into(),
            "p4".into(),
            TransferDirection::Download,
            10,
            10,
        );
        mgr.fail(id_fail, "oops".into());

        let stats = mgr.stats();
        assert_eq!(stats.active_transfers, 2);
        assert_eq!(stats.completed_transfers, 1);
        assert_eq!(stats.failed_transfers, 1);
        assert_eq!(stats.total_bytes_transferred, 10);
    }

    // 18. advance_tick increments current_tick.
    #[test]
    fn test_advance_tick() {
        let mut mgr = StreamingBlockTransfer::new();
        assert_eq!(mgr.current_tick, 0);
        mgr.advance_tick();
        mgr.advance_tick();
        assert_eq!(mgr.current_tick, 2);
    }

    // 19. started_at_tick is recorded from current_tick.
    #[test]
    fn test_started_at_tick_recorded() {
        let mut mgr = StreamingBlockTransfer::new();
        mgr.advance_tick();
        mgr.advance_tick();
        let id = mgr.start_transfer("c".into(), "p".into(), TransferDirection::Upload, 10, 10);
        assert_eq!(mgr.transfers[&id].started_at_tick, 2);
    }

    // 20. receive_chunk rejected for paused transfer.
    #[test]
    fn test_receive_chunk_rejected_when_paused() {
        let mut mgr = StreamingBlockTransfer::new();
        let id = mgr.start_transfer("c".into(), "p".into(), TransferDirection::Download, 20, 10);
        mgr.pause(id);
        let chunk = make_chunk(0, vec![3u8; 10]);
        let result = mgr.receive_chunk(id, chunk);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("paused"));
    }

    // 21. is_complete returns false for InProgress.
    #[test]
    fn test_is_complete_false_for_in_progress() {
        let mut mgr = StreamingBlockTransfer::new();
        let id = mgr.start_transfer("c".into(), "p".into(), TransferDirection::Download, 20, 10);
        let chunk = make_chunk(0, vec![9u8; 10]);
        mgr.receive_chunk(id, chunk)
            .expect("test: receive_chunk should accept valid chunk");
        assert!(!mgr.transfers[&id].is_complete());
    }

    // 22. Unknown transfer ID returns error from receive_chunk.
    #[test]
    fn test_receive_chunk_unknown_id() {
        let mut mgr = StreamingBlockTransfer::new();
        let chunk = make_chunk(0, vec![1u8]);
        let result = mgr.receive_chunk(9999, chunk);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unknown transfer id"));
    }
}
