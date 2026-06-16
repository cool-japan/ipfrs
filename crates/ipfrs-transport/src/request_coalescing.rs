//! Request coalescing for deduplicating concurrent block requests
//!
//! This module provides mechanisms to coalesce multiple concurrent requests
//! for the same block into a single network request, reducing bandwidth usage
//! and improving efficiency.
//!
//! # Example
//!
//! ```
//! use ipfrs_transport::{RequestCoalescer, CoalescerConfig};
//! use ipfrs_core::Cid;
//! use bytes::Bytes;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = CoalescerConfig::default();
//! let coalescer = RequestCoalescer::new(config);
//!
//! // Multiple concurrent requests for the same CID will be coalesced
//! let cid = Cid::default();
//! # Ok(())
//! # }
//! ```

use bytes::Bytes;
use dashmap::DashMap;
use ipfrs_core::Cid;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, RwLock};

/// Configuration for request coalescing
#[derive(Debug, Clone)]
pub struct CoalescerConfig {
    /// Maximum time to wait for coalescing before making a request
    pub coalesce_window: Duration,
    /// Maximum number of waiters per request before forcing immediate request
    pub max_waiters_per_request: usize,
    /// Channel capacity for broadcasting results
    pub broadcast_capacity: usize,
    /// Enable statistics tracking
    pub enable_stats: bool,
}

impl Default for CoalescerConfig {
    fn default() -> Self {
        Self {
            coalesce_window: Duration::from_millis(10),
            max_waiters_per_request: 100,
            broadcast_capacity: 128,
            enable_stats: true,
        }
    }
}

/// Statistics for request coalescing
#[derive(Debug, Clone, Default)]
pub struct CoalescerStats {
    /// Total number of requests received
    pub total_requests: u64,
    /// Number of coalesced requests (not sent to network)
    pub coalesced_requests: u64,
    /// Number of unique requests actually sent
    pub unique_requests: u64,
    /// Average number of waiters per coalesced request
    pub avg_waiters_per_request: f64,
    /// Maximum waiters seen for a single request
    pub max_waiters_seen: usize,
}

impl CoalescerStats {
    /// Calculate coalescing efficiency (0.0 to 1.0, higher is better)
    pub fn efficiency(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.coalesced_requests as f64 / self.total_requests as f64
    }

    /// Calculate network reduction ratio (how many requests were saved)
    pub fn reduction_ratio(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        1.0 - (self.unique_requests as f64 / self.total_requests as f64)
    }
}

/// Pending request information
struct PendingRequest {
    /// Broadcast sender for results
    tx: broadcast::Sender<Result<Bytes, String>>,
    /// When the request was first received
    created_at: Instant,
    /// Number of waiters for this request
    waiter_count: usize,
}

/// Request coalescer that deduplicates concurrent requests
pub struct RequestCoalescer {
    /// Configuration
    config: CoalescerConfig,
    /// Pending requests indexed by CID
    pending: Arc<DashMap<Cid, PendingRequest>>,
    /// Statistics
    stats: Arc<RwLock<CoalescerStats>>,
}

impl RequestCoalescer {
    /// Create a new request coalescer
    pub fn new(config: CoalescerConfig) -> Self {
        Self {
            config,
            pending: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(CoalescerStats::default())),
        }
    }

    /// Register a request for a CID
    ///
    /// Returns:
    /// - Ok(Some(rx)) if this should wait for an existing request
    /// - Ok(None) if this is the first request (caller should fetch)
    pub async fn register_request(
        &self,
        cid: &Cid,
    ) -> Result<Option<broadcast::Receiver<Result<Bytes, String>>>, String> {
        // Update total requests stat
        if self.config.enable_stats {
            let mut stats = self.stats.write().await;
            stats.total_requests += 1;
        }

        // Check if there's already a pending request
        if let Some(mut entry) = self.pending.get_mut(cid) {
            // Join existing request
            entry.waiter_count += 1;
            let rx = entry.tx.subscribe();

            // Update stats
            if self.config.enable_stats {
                let mut stats = self.stats.write().await;
                stats.coalesced_requests += 1;
                if entry.waiter_count > stats.max_waiters_seen {
                    stats.max_waiters_seen = entry.waiter_count;
                }
            }

            // Check if we should force immediate fetch due to too many waiters
            if entry.waiter_count >= self.config.max_waiters_per_request {
                drop(entry); // Release lock before removing
                self.pending.remove(cid);
                return Ok(None); // Force immediate fetch
            }

            return Ok(Some(rx));
        }

        // This is the first request - create a new pending entry
        let (tx, _rx) = broadcast::channel(self.config.broadcast_capacity);
        self.pending.insert(
            *cid,
            PendingRequest {
                tx,
                created_at: Instant::now(),
                waiter_count: 1,
            },
        );

        // Update stats
        if self.config.enable_stats {
            let mut stats = self.stats.write().await;
            stats.unique_requests += 1;
        }

        Ok(None) // Caller should fetch
    }

    /// Complete a request with success
    pub async fn complete_request(&self, cid: &Cid, data: Bytes) {
        if let Some((_, pending)) = self.pending.remove(cid) {
            let waiter_count = pending.waiter_count;

            // Broadcast result to all waiters
            let _ = pending.tx.send(Ok(data));

            // Update average waiters stat
            if self.config.enable_stats && waiter_count > 1 {
                let mut stats = self.stats.write().await;
                let total_waiters = stats.avg_waiters_per_request * stats.coalesced_requests as f64;
                stats.avg_waiters_per_request =
                    (total_waiters + waiter_count as f64) / (stats.coalesced_requests + 1) as f64;
            }
        }
    }

    /// Fail a request with an error
    pub async fn fail_request(&self, cid: &Cid, error: String) {
        if let Some((_, pending)) = self.pending.remove(cid) {
            // Broadcast error to all waiters
            let _ = pending.tx.send(Err(error));
        }
    }

    /// Cancel a request (remove from pending without notifying)
    pub async fn cancel_request(&self, cid: &Cid) {
        self.pending.remove(cid);
    }

    /// Get current statistics
    pub async fn stats(&self) -> CoalescerStats {
        self.stats.read().await.clone()
    }

    /// Reset statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = CoalescerStats::default();
    }

    /// Clean up expired pending requests
    pub async fn cleanup_expired(&self) {
        let timeout = self.config.coalesce_window * 10; // 10x the coalesce window
        self.pending.retain(|_, pending| {
            if pending.created_at.elapsed() > timeout {
                // Notify waiters of timeout
                let _ = pending
                    .tx
                    .send(Err("Request timeout during coalescing".to_string()));
                false
            } else {
                true
            }
        });
    }

    /// Get number of pending requests
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cid(seed: u64) -> Cid {
        use multihash::Multihash;
        let data = seed.to_le_bytes();
        let hash = Multihash::wrap(0x12, &data).expect("test: wrap multihash from seed bytes");
        Cid::new_v1(0x55, hash)
    }

    #[tokio::test]
    async fn test_request_coalescer_creation() {
        let config = CoalescerConfig::default();
        let coalescer = RequestCoalescer::new(config);
        assert_eq!(coalescer.pending_count(), 0);
    }

    #[tokio::test]
    async fn test_first_request_returns_none() {
        let coalescer = RequestCoalescer::new(CoalescerConfig::default());
        let cid = test_cid(1);

        let result = coalescer
            .register_request(&cid)
            .await
            .expect("test: register request");
        assert!(result.is_none()); // First request should return None
        assert_eq!(coalescer.pending_count(), 1);
    }

    #[tokio::test]
    async fn test_duplicate_request_returns_receiver() {
        let coalescer = RequestCoalescer::new(CoalescerConfig::default());
        let cid = test_cid(1);

        // First request
        let first = coalescer
            .register_request(&cid)
            .await
            .expect("test: register request");
        assert!(first.is_none());

        // Duplicate request
        let second = coalescer
            .register_request(&cid)
            .await
            .expect("test: register request");
        assert!(second.is_some());
        assert_eq!(coalescer.pending_count(), 1); // Still only one pending
    }

    #[tokio::test]
    async fn test_complete_request_broadcasts_to_waiters() {
        let coalescer = RequestCoalescer::new(CoalescerConfig::default());
        let cid = test_cid(1);

        // First request (will fetch)
        let first = coalescer
            .register_request(&cid)
            .await
            .expect("test: register request");
        assert!(first.is_none());

        // Second request (will wait)
        let mut second_rx = coalescer
            .register_request(&cid)
            .await
            .expect("test: register request")
            .expect("test: receiver should be Some");

        // Third request (will wait)
        let mut third_rx = coalescer
            .register_request(&cid)
            .await
            .expect("test: register request")
            .expect("test: receiver should be Some");

        // Complete the request
        let data = Bytes::from("test data");
        coalescer.complete_request(&cid, data.clone()).await;

        // Both waiters should receive the data
        let result2 = second_rx
            .recv()
            .await
            .expect("test: receive result")
            .expect("test: inner result");
        let result3 = third_rx
            .recv()
            .await
            .expect("test: receive result")
            .expect("test: inner result");

        assert_eq!(result2, data);
        assert_eq!(result3, data);
        assert_eq!(coalescer.pending_count(), 0);
    }

    #[tokio::test]
    async fn test_fail_request_broadcasts_error() {
        let coalescer = RequestCoalescer::new(CoalescerConfig::default());
        let cid = test_cid(1);

        // First request
        coalescer
            .register_request(&cid)
            .await
            .expect("test: register request");

        // Second request (waiter)
        let mut rx = coalescer
            .register_request(&cid)
            .await
            .expect("test: register request")
            .expect("test: receiver should be Some");

        // Fail the request
        coalescer
            .fail_request(&cid, "Network error".to_string())
            .await;

        // Waiter should receive error
        let result = rx.recv().await.expect("test: receive result");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Network error");
        assert_eq!(coalescer.pending_count(), 0);
    }

    #[tokio::test]
    async fn test_cancel_request_removes_pending() {
        let coalescer = RequestCoalescer::new(CoalescerConfig::default());
        let cid = test_cid(1);

        coalescer
            .register_request(&cid)
            .await
            .expect("test: register request");
        assert_eq!(coalescer.pending_count(), 1);

        coalescer.cancel_request(&cid).await;
        assert_eq!(coalescer.pending_count(), 0);
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let config = CoalescerConfig {
            enable_stats: true,
            ..Default::default()
        };
        let coalescer = RequestCoalescer::new(config);
        let cid = test_cid(1);

        // First request
        coalescer
            .register_request(&cid)
            .await
            .expect("test: register request");

        // Duplicate requests
        coalescer
            .register_request(&cid)
            .await
            .expect("test: register request");
        coalescer
            .register_request(&cid)
            .await
            .expect("test: register request");

        let stats = coalescer.stats().await;
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.unique_requests, 1);
        assert_eq!(stats.coalesced_requests, 2);
    }

    #[tokio::test]
    async fn test_efficiency_calculation() {
        let stats = CoalescerStats {
            total_requests: 100,
            coalesced_requests: 80,
            unique_requests: 20,
            avg_waiters_per_request: 4.0,
            max_waiters_seen: 10,
        };

        assert_eq!(stats.efficiency(), 0.8); // 80/100
        assert_eq!(stats.reduction_ratio(), 0.8); // 1 - (20/100)
    }

    #[tokio::test]
    async fn test_max_waiters_limit() {
        let config = CoalescerConfig {
            max_waiters_per_request: 3,
            ..Default::default()
        };
        let coalescer = RequestCoalescer::new(config);
        let cid = test_cid(1);

        // First request
        coalescer
            .register_request(&cid)
            .await
            .expect("test: register request");

        // Add waiters
        coalescer
            .register_request(&cid)
            .await
            .expect("test: register request");
        coalescer
            .register_request(&cid)
            .await
            .expect("test: register request");

        // This should exceed max_waiters and return None (force fetch)
        let result = coalescer
            .register_request(&cid)
            .await
            .expect("test: register request");
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_reset_stats() {
        let coalescer = RequestCoalescer::new(CoalescerConfig::default());
        let cid = test_cid(1);

        coalescer
            .register_request(&cid)
            .await
            .expect("test: register request");
        coalescer
            .register_request(&cid)
            .await
            .expect("test: register request");

        let stats = coalescer.stats().await;
        assert!(stats.total_requests > 0);

        coalescer.reset_stats().await;
        let stats = coalescer.stats().await;
        assert_eq!(stats.total_requests, 0);
    }

    #[tokio::test]
    async fn test_concurrent_different_cids() {
        let coalescer = RequestCoalescer::new(CoalescerConfig::default());
        let cid1 = test_cid(1);
        let cid2 = test_cid(2);

        // Requests for different CIDs should not coalesce
        let r1 = coalescer
            .register_request(&cid1)
            .await
            .expect("test: register request");
        let r2 = coalescer
            .register_request(&cid2)
            .await
            .expect("test: register request");

        assert!(r1.is_none());
        assert!(r2.is_none());
        assert_eq!(coalescer.pending_count(), 2);
    }
}
