//! Peer message batching for outbound message coalescing
//!
//! This module provides `PeerMessageBatcher`, which accumulates outbound messages
//! destined for the same peer and flushes them as a single batch when a size or
//! time threshold is reached.
//!
//! ## Design
//!
//! Messages are accumulated per-peer in an in-memory queue. Flushing is triggered by:
//! - **Size threshold**: accumulated payload bytes >= `max_batch_bytes`
//! - **Count threshold**: number of queued messages >= `max_batch_count`
//! - **Age threshold**: oldest message age (in ticks) >= `max_age_ticks`
//! - **Manual flush**: caller explicitly flushes a peer or all peers
//!
//! The batcher uses a logical tick counter (`u64`) rather than wall-clock time,
//! allowing deterministic testing without OS time dependencies.

use std::collections::HashMap;

/// A single outbound message queued for batched delivery.
#[derive(Debug, Clone)]
pub struct BatchMessage {
    /// Monotonic message ID assigned at push time.
    pub msg_id: u64,
    /// Destination peer identifier.
    pub peer_id: String,
    /// Raw payload bytes.
    pub payload: Vec<u8>,
    /// Logical tick counter value at the time the message was enqueued.
    pub enqueued_at: u64,
}

/// Configuration for `PeerMessageBatcher`.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Flush when accumulated payload bytes across the peer's queue reach this value.
    pub max_batch_bytes: u64,
    /// Flush when the peer's queue reaches this many messages.
    pub max_batch_count: usize,
    /// Flush when the oldest message in the peer's queue has been waiting for at
    /// least this many ticks (i.e. `tick - enqueued_at >= max_age_ticks`).
    pub max_age_ticks: u64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_bytes: 65_536,
            max_batch_count: 64,
            max_age_ticks: 100,
        }
    }
}

/// Reason why a batch was flushed.
#[derive(Debug, Clone, PartialEq)]
pub enum FlushReason {
    /// Accumulated payload bytes exceeded `max_batch_bytes`.
    SizeThreshold,
    /// Number of queued messages reached `max_batch_count`.
    CountThreshold,
    /// Oldest message waited at least `max_age_ticks` ticks.
    AgeThreshold,
    /// Caller explicitly flushed the peer's queue.
    ManualFlush,
}

/// A completed batch ready for delivery.
#[derive(Debug, Clone)]
pub struct BatchFlush {
    /// Destination peer.
    pub peer_id: String,
    /// Messages included in this batch (in insertion order).
    pub messages: Vec<BatchMessage>,
    /// What triggered the flush.
    pub reason: FlushReason,
    /// Sum of payload lengths across all messages in this batch.
    pub total_bytes: u64,
}

/// Aggregate statistics for a `PeerMessageBatcher`.
#[derive(Debug, Clone, Default)]
pub struct BatcherStats {
    /// Total number of messages pushed via `push`.
    pub total_pushed: u64,
    /// Total number of messages delivered across all flushed batches.
    pub total_flushed: u64,
    /// Total number of batches that have been flushed.
    pub total_batches: u64,
}

impl BatcherStats {
    /// Average number of messages per flushed batch.
    ///
    /// Returns `0.0` when no batches have been flushed yet.
    pub fn average_batch_size(&self) -> f64 {
        self.total_flushed as f64 / self.total_batches.max(1) as f64
    }
}

/// Batches outbound messages per-peer and flushes them when thresholds are met.
pub struct PeerMessageBatcher {
    /// Per-peer pending message queues.
    pub pending: HashMap<String, Vec<BatchMessage>>,
    /// Batching configuration.
    pub config: BatchConfig,
    /// Running statistics.
    pub stats: BatcherStats,
    /// Next message ID to assign.
    pub next_id: u64,
    /// Current logical tick counter.
    pub tick: u64,
}

impl PeerMessageBatcher {
    /// Create a new batcher with the supplied configuration.
    pub fn new(config: BatchConfig) -> Self {
        Self {
            pending: HashMap::new(),
            config,
            stats: BatcherStats::default(),
            next_id: 0,
            tick: 0,
        }
    }

    /// Push a message for `peer_id`.
    ///
    /// Assigns a monotonic `msg_id` and records `enqueued_at = tick`, then appends
    /// to the peer's queue.  If either the size or count threshold is exceeded
    /// *after* appending, the peer's queue is immediately flushed and the resulting
    /// `BatchFlush` is returned.  Otherwise `None` is returned and the message
    /// stays in the queue until the next flush trigger.
    pub fn push(&mut self, peer_id: String, payload: Vec<u8>) -> Option<BatchFlush> {
        let msg_id = self.next_id;
        self.next_id += 1;

        let msg = BatchMessage {
            msg_id,
            peer_id: peer_id.clone(),
            payload,
            enqueued_at: self.tick,
        };

        let queue = self.pending.entry(peer_id.clone()).or_default();
        queue.push(msg);
        self.stats.total_pushed += 1;

        // Check size threshold.
        let total_bytes: u64 = queue.iter().map(|m| m.payload.len() as u64).sum();
        if total_bytes >= self.config.max_batch_bytes {
            return Some(self.do_flush(&peer_id, FlushReason::SizeThreshold));
        }

        // Check count threshold.
        if queue.len() >= self.config.max_batch_count {
            return Some(self.do_flush(&peer_id, FlushReason::CountThreshold));
        }

        None
    }

    /// Advance the logical tick counter by one and check age thresholds.
    ///
    /// For each peer whose oldest pending message has been waiting for at least
    /// `max_age_ticks` ticks, the queue is flushed with `AgeThreshold`.
    ///
    /// Returns all flushes that were triggered.
    pub fn tick_advance(&mut self) -> Vec<BatchFlush> {
        self.tick += 1;

        // Collect peers that need age-triggered flushing.
        // We must collect first to avoid holding a borrow while we mutate `pending`.
        let peers_to_flush: Vec<String> = self
            .pending
            .iter()
            .filter_map(|(peer_id, queue)| {
                if let Some(oldest) = queue.first() {
                    if self.tick.saturating_sub(oldest.enqueued_at) >= self.config.max_age_ticks {
                        return Some(peer_id.clone());
                    }
                }
                None
            })
            .collect();

        peers_to_flush
            .into_iter()
            .map(|peer_id| self.do_flush(&peer_id, FlushReason::AgeThreshold))
            .collect()
    }

    /// Force-flush a specific peer's pending messages with `ManualFlush`.
    ///
    /// Returns `None` if the peer has no pending messages.
    pub fn flush_peer(&mut self, peer_id: &str) -> Option<BatchFlush> {
        match self.pending.get(peer_id) {
            Some(queue) if !queue.is_empty() => {
                Some(self.do_flush(peer_id, FlushReason::ManualFlush))
            }
            _ => None,
        }
    }

    /// Force-flush every peer that has pending messages with `ManualFlush`.
    pub fn flush_all(&mut self) -> Vec<BatchFlush> {
        let peers: Vec<String> = self
            .pending
            .iter()
            .filter(|(_, q)| !q.is_empty())
            .map(|(p, _)| p.clone())
            .collect();

        peers
            .into_iter()
            .map(|peer_id| self.do_flush(&peer_id, FlushReason::ManualFlush))
            .collect()
    }

    /// Return a reference to the current statistics.
    pub fn stats(&self) -> &BatcherStats {
        &self.stats
    }

    /// Drain the queue for `peer_id`, compute statistics, and return a `BatchFlush`.
    ///
    /// # Panics (never)
    ///
    /// This function is careful to handle missing/empty queue entries gracefully.
    fn do_flush(&mut self, peer_id: &str, reason: FlushReason) -> BatchFlush {
        let messages = self.pending.remove(peer_id).unwrap_or_default();

        let total_bytes: u64 = messages.iter().map(|m| m.payload.len() as u64).sum();
        let count = messages.len() as u64;

        self.stats.total_flushed += count;
        self.stats.total_batches += 1;

        BatchFlush {
            peer_id: peer_id.to_string(),
            messages,
            reason,
            total_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_batcher() -> PeerMessageBatcher {
        PeerMessageBatcher::new(BatchConfig::default())
    }

    // -------------------------------------------------------------------------
    // push – under threshold → None
    // -------------------------------------------------------------------------

    #[test]
    fn test_push_under_all_thresholds_returns_none() {
        let mut batcher = default_batcher();
        let result = batcher.push("peer-A".to_string(), vec![0u8; 100]);
        assert!(result.is_none(), "single small message should not flush");
    }

    #[test]
    fn test_push_increments_total_pushed() {
        let mut batcher = default_batcher();
        batcher.push("peer-A".to_string(), vec![1u8; 10]);
        batcher.push("peer-A".to_string(), vec![2u8; 10]);
        assert_eq!(batcher.stats().total_pushed, 2);
    }

    #[test]
    fn test_push_assigns_monotonic_ids() {
        let mut batcher = default_batcher();
        batcher.push("peer-A".to_string(), vec![0u8; 1]);
        batcher.push("peer-A".to_string(), vec![0u8; 1]);
        let queue = batcher.pending.get("peer-A").expect("queue must exist");
        assert_eq!(queue[0].msg_id, 0);
        assert_eq!(queue[1].msg_id, 1);
    }

    #[test]
    fn test_push_records_enqueued_at_tick() {
        let mut batcher = default_batcher();
        // Advance tick to 5 before pushing.
        for _ in 0..5 {
            batcher.tick_advance();
        }
        batcher.push("peer-A".to_string(), vec![0u8; 1]);
        let queue = batcher.pending.get("peer-A").expect("queue must exist");
        assert_eq!(queue[0].enqueued_at, 5);
    }

    // -------------------------------------------------------------------------
    // push – size threshold → Some(SizeThreshold)
    // -------------------------------------------------------------------------

    #[test]
    fn test_push_over_size_threshold_returns_size_flush() {
        let config = BatchConfig {
            max_batch_bytes: 100,
            max_batch_count: 1000,
            max_age_ticks: 10_000,
        };
        let mut batcher = PeerMessageBatcher::new(config);

        // Push 90 bytes – still under threshold.
        assert!(batcher.push("peer-B".to_string(), vec![0u8; 90]).is_none());

        // Push 10 more bytes to reach exactly 100 – should trigger SizeThreshold.
        let flush = batcher
            .push("peer-B".to_string(), vec![0u8; 10])
            .expect("should flush on size threshold");

        assert_eq!(flush.peer_id, "peer-B");
        assert_eq!(flush.reason, FlushReason::SizeThreshold);
        assert_eq!(flush.total_bytes, 100);
        assert_eq!(flush.messages.len(), 2);
    }

    #[test]
    fn test_push_size_flush_clears_peer_queue() {
        let config = BatchConfig {
            max_batch_bytes: 50,
            max_batch_count: 1000,
            max_age_ticks: 10_000,
        };
        let mut batcher = PeerMessageBatcher::new(config);
        batcher.push("peer-C".to_string(), vec![0u8; 50]);
        // Queue should be gone after flush.
        assert!(
            !batcher.pending.contains_key("peer-C")
                || batcher
                    .pending
                    .get("peer-C")
                    .map(|q| q.is_empty())
                    .unwrap_or(false)
        );
    }

    // -------------------------------------------------------------------------
    // push – count threshold → Some(CountThreshold)
    // -------------------------------------------------------------------------

    #[test]
    fn test_push_over_count_threshold_returns_count_flush() {
        let config = BatchConfig {
            max_batch_bytes: u64::MAX,
            max_batch_count: 3,
            max_age_ticks: 10_000,
        };
        let mut batcher = PeerMessageBatcher::new(config);

        assert!(batcher.push("peer-D".to_string(), vec![0u8; 1]).is_none());
        assert!(batcher.push("peer-D".to_string(), vec![0u8; 1]).is_none());

        let flush = batcher
            .push("peer-D".to_string(), vec![0u8; 1])
            .expect("should flush on count threshold");

        assert_eq!(flush.reason, FlushReason::CountThreshold);
        assert_eq!(flush.messages.len(), 3);
    }

    #[test]
    fn test_push_count_flush_updates_stats() {
        let config = BatchConfig {
            max_batch_bytes: u64::MAX,
            max_batch_count: 2,
            max_age_ticks: 10_000,
        };
        let mut batcher = PeerMessageBatcher::new(config);
        batcher.push("peer-E".to_string(), vec![0u8; 1]);
        batcher.push("peer-E".to_string(), vec![0u8; 1]);

        let stats = batcher.stats();
        assert_eq!(stats.total_flushed, 2);
        assert_eq!(stats.total_batches, 1);
    }

    // -------------------------------------------------------------------------
    // tick_advance – age threshold → AgeThreshold
    // -------------------------------------------------------------------------

    #[test]
    fn test_tick_advance_fires_age_threshold() {
        let config = BatchConfig {
            max_batch_bytes: u64::MAX,
            max_batch_count: 1000,
            max_age_ticks: 5,
        };
        let mut batcher = PeerMessageBatcher::new(config);
        batcher.push("peer-F".to_string(), vec![0u8; 1]);

        // Advance 4 ticks – should not yet flush.
        for _ in 0..4 {
            let flushes = batcher.tick_advance();
            assert!(flushes.is_empty(), "should not flush before age threshold");
        }

        // 5th advance reaches age threshold.
        let flushes = batcher.tick_advance();
        assert_eq!(flushes.len(), 1);
        assert_eq!(flushes[0].reason, FlushReason::AgeThreshold);
        assert_eq!(flushes[0].peer_id, "peer-F");
    }

    #[test]
    fn test_tick_advance_multiple_peers() {
        let config = BatchConfig {
            max_batch_bytes: u64::MAX,
            max_batch_count: 1000,
            max_age_ticks: 3,
        };
        let mut batcher = PeerMessageBatcher::new(config);
        batcher.push("peer-G1".to_string(), vec![0u8; 1]);
        batcher.push("peer-G2".to_string(), vec![0u8; 1]);

        // Advance 3 ticks.
        for _ in 0..2 {
            batcher.tick_advance();
        }
        let flushes = batcher.tick_advance();
        assert_eq!(flushes.len(), 2, "both peers should age-flush together");
        let reasons: Vec<_> = flushes.iter().map(|f| &f.reason).collect();
        assert!(reasons.iter().all(|r| **r == FlushReason::AgeThreshold));
    }

    #[test]
    fn test_tick_advance_no_flush_when_empty() {
        let mut batcher = default_batcher();
        for _ in 0..200 {
            let flushes = batcher.tick_advance();
            assert!(flushes.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // flush_peer – ManualFlush
    // -------------------------------------------------------------------------

    #[test]
    fn test_flush_peer_returns_manual_flush() {
        let mut batcher = default_batcher();
        batcher.push("peer-H".to_string(), vec![1u8; 20]);
        batcher.push("peer-H".to_string(), vec![2u8; 30]);

        let flush = batcher
            .flush_peer("peer-H")
            .expect("should return flush for non-empty peer");

        assert_eq!(flush.reason, FlushReason::ManualFlush);
        assert_eq!(flush.messages.len(), 2);
        assert_eq!(flush.total_bytes, 50);
    }

    #[test]
    fn test_flush_peer_returns_none_when_empty() {
        let mut batcher = default_batcher();
        let result = batcher.flush_peer("unknown-peer");
        assert!(result.is_none());
    }

    #[test]
    fn test_flush_peer_clears_queue() {
        let mut batcher = default_batcher();
        batcher.push("peer-I".to_string(), vec![0u8; 5]);
        batcher.flush_peer("peer-I");

        let result = batcher.flush_peer("peer-I");
        assert!(result.is_none(), "queue should be empty after flush");
    }

    // -------------------------------------------------------------------------
    // flush_all
    // -------------------------------------------------------------------------

    #[test]
    fn test_flush_all_flushes_multiple_peers() {
        let mut batcher = default_batcher();
        batcher.push("peer-J1".to_string(), vec![0u8; 10]);
        batcher.push("peer-J2".to_string(), vec![0u8; 20]);
        batcher.push("peer-J3".to_string(), vec![0u8; 30]);

        let flushes = batcher.flush_all();
        assert_eq!(flushes.len(), 3);
        assert!(flushes.iter().all(|f| f.reason == FlushReason::ManualFlush));
    }

    #[test]
    fn test_flush_all_returns_empty_vec_when_nothing_pending() {
        let mut batcher = default_batcher();
        let flushes = batcher.flush_all();
        assert!(flushes.is_empty());
    }

    #[test]
    fn test_flush_all_clears_all_queues() {
        let mut batcher = default_batcher();
        batcher.push("peer-K1".to_string(), vec![0u8; 1]);
        batcher.push("peer-K2".to_string(), vec![0u8; 1]);
        batcher.flush_all();

        let flushes_after = batcher.flush_all();
        assert!(
            flushes_after.is_empty(),
            "all queues should be empty after flush_all"
        );
    }

    // -------------------------------------------------------------------------
    // stats
    // -------------------------------------------------------------------------

    #[test]
    fn test_stats_after_multiple_flushes() {
        let config = BatchConfig {
            max_batch_bytes: u64::MAX,
            max_batch_count: 2,
            max_age_ticks: 10_000,
        };
        let mut batcher = PeerMessageBatcher::new(config);

        // First batch (peer-L, 2 messages → CountThreshold flush)
        batcher.push("peer-L".to_string(), vec![0u8; 1]);
        batcher.push("peer-L".to_string(), vec![0u8; 1]);

        // Second batch (peer-M, 2 messages → CountThreshold flush)
        batcher.push("peer-M".to_string(), vec![0u8; 1]);
        batcher.push("peer-M".to_string(), vec![0u8; 1]);

        let stats = batcher.stats();
        assert_eq!(stats.total_pushed, 4);
        assert_eq!(stats.total_flushed, 4);
        assert_eq!(stats.total_batches, 2);
    }

    #[test]
    fn test_average_batch_size_zero_when_no_batches() {
        let batcher = default_batcher();
        // total_batches = 0 → max(1) makes denominator 1 → returns 0.0
        assert_eq!(batcher.stats().average_batch_size(), 0.0);
    }

    #[test]
    fn test_average_batch_size_correct() {
        let config = BatchConfig {
            max_batch_bytes: u64::MAX,
            max_batch_count: 3,
            max_age_ticks: 10_000,
        };
        let mut batcher = PeerMessageBatcher::new(config);

        // Batch 1: 3 messages
        batcher.push("peer-N".to_string(), vec![0u8; 1]);
        batcher.push("peer-N".to_string(), vec![0u8; 1]);
        batcher.push("peer-N".to_string(), vec![0u8; 1]);

        // Batch 2: manually flush peer-N2 with 1 message
        batcher.push("peer-N2".to_string(), vec![0u8; 1]);
        batcher.flush_peer("peer-N2");

        // total_flushed = 4, total_batches = 2 → average = 2.0
        let avg = batcher.stats().average_batch_size();
        assert!(
            (avg - 2.0).abs() < f64::EPSILON,
            "expected average 2.0, got {avg}"
        );
    }

    #[test]
    fn test_do_flush_total_bytes_computed_correctly() {
        let mut batcher = default_batcher();
        batcher.push("peer-O".to_string(), vec![0u8; 100]);
        batcher.push("peer-O".to_string(), vec![0u8; 200]);
        batcher.push("peer-O".to_string(), vec![0u8; 50]);

        let flush = batcher.flush_peer("peer-O").expect("flush must succeed");
        assert_eq!(flush.total_bytes, 350);
    }

    #[test]
    fn test_separate_peer_queues_are_independent() {
        let config = BatchConfig {
            max_batch_bytes: u64::MAX,
            max_batch_count: 2,
            max_age_ticks: 10_000,
        };
        let mut batcher = PeerMessageBatcher::new(config);

        // peer-P1 gets 1 message (under count threshold).
        let r1 = batcher.push("peer-P1".to_string(), vec![0u8; 1]);
        assert!(r1.is_none());

        // peer-P2 reaches the count threshold independently.
        batcher.push("peer-P2".to_string(), vec![0u8; 1]);
        let r2 = batcher.push("peer-P2".to_string(), vec![0u8; 1]);
        assert!(r2.is_some(), "peer-P2 should flush independently");

        // peer-P1 still has its message.
        let q = batcher
            .pending
            .get("peer-P1")
            .expect("peer-P1 queue must exist");
        assert_eq!(q.len(), 1);
    }
}
