//! Bandwidth throttling for rate limiting
//!
//! Provides token bucket and leaky bucket algorithms for:
//! - Per-peer rate limits
//! - Global bandwidth caps
//! - QoS prioritization
//! - Burst allowance
//!
//! # Example
//!
//! ```
//! use ipfrs_transport::{TokenBucket, BandwidthThrottle, BandwidthConfig, QosPriority};
//!
//! // Create a token bucket with 1000 token capacity, refilling at 100 tokens/sec
//! let bucket = TokenBucket::new(1000, 100.0);
//!
//! // Try to consume 50 tokens
//! if bucket.try_consume(50) {
//!     println!("Request allowed");
//! } else {
//!     println!("Rate limit exceeded");
//! }
//!
//! // Create a bandwidth throttle with global limits
//! let mut config = BandwidthConfig::default();
//! config.global_upload_limit = 10_000_000; // 10 MB/s
//! config.global_download_limit = 10_000_000;
//! let throttle = BandwidthThrottle::new(config);
//!
//! // Register a peer with high priority
//! let peer_addr = "127.0.0.1:8080".parse().unwrap();
//! throttle.register_peer(peer_addr, QosPriority::High);
//! ```

use parking_lot::RwLock;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::time::sleep;
use tracing::debug;

/// Throttle error types
#[derive(Error, Debug)]
pub enum ThrottleError {
    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Quota exhausted")]
    QuotaExhausted,

    #[error("Peer not found: {0}")]
    PeerNotFound(String),
}

/// Token bucket for rate limiting
#[derive(Debug, Clone)]
pub struct TokenBucket {
    /// Maximum number of tokens (bucket capacity)
    capacity: u64,
    /// Current number of tokens
    tokens: Arc<RwLock<f64>>,
    /// Token refill rate (tokens per second)
    refill_rate: f64,
    /// Last refill time
    last_refill: Arc<RwLock<Instant>>,
}

impl TokenBucket {
    /// Create a new token bucket
    pub fn new(capacity: u64, refill_rate: f64) -> Self {
        Self {
            capacity,
            tokens: Arc::new(RwLock::new(capacity as f64)),
            refill_rate,
            last_refill: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Try to consume tokens (non-blocking)
    pub fn try_consume(&self, amount: u64) -> bool {
        self.refill();

        let mut tokens = self.tokens.write();
        if *tokens >= amount as f64 {
            *tokens -= amount as f64;
            true
        } else {
            false
        }
    }

    /// Consume tokens (blocking until available)
    pub async fn consume(&self, amount: u64) {
        loop {
            if self.try_consume(amount) {
                return;
            }

            // Calculate wait time
            let tokens = *self.tokens.read();
            let needed = amount as f64 - tokens;
            let wait_time = Duration::from_secs_f64(needed / self.refill_rate);

            sleep(wait_time.min(Duration::from_millis(100))).await;
        }
    }

    /// Refill tokens based on elapsed time
    fn refill(&self) {
        let now = Instant::now();
        let mut last_refill = self.last_refill.write();
        let elapsed = now.duration_since(*last_refill).as_secs_f64();

        if elapsed > 0.0 {
            let new_tokens = elapsed * self.refill_rate;
            let mut tokens = self.tokens.write();
            *tokens = (*tokens + new_tokens).min(self.capacity as f64);
            *last_refill = now;
        }
    }

    /// Get current token count
    pub fn available_tokens(&self) -> u64 {
        self.refill();
        *self.tokens.read() as u64
    }

    /// Get capacity
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Get refill rate
    pub fn refill_rate(&self) -> f64 {
        self.refill_rate
    }
}

/// QoS priority levels for bandwidth allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QosPriority {
    /// Best effort (lowest priority)
    BestEffort = 0,
    /// Normal priority
    Normal = 1,
    /// High priority
    High = 2,
    /// Critical (highest priority)
    Critical = 3,
}

impl QosPriority {
    /// Get bandwidth share multiplier
    pub fn multiplier(&self) -> f64 {
        match self {
            QosPriority::BestEffort => 0.5,
            QosPriority::Normal => 1.0,
            QosPriority::High => 2.0,
            QosPriority::Critical => 4.0,
        }
    }
}

/// Bandwidth throttle configuration
#[derive(Debug, Clone)]
pub struct BandwidthConfig {
    /// Global upload limit (bytes per second, 0 = unlimited)
    pub global_upload_limit: u64,
    /// Global download limit (bytes per second, 0 = unlimited)
    pub global_download_limit: u64,
    /// Per-peer upload limit (bytes per second, 0 = unlimited)
    pub peer_upload_limit: u64,
    /// Per-peer download limit (bytes per second, 0 = unlimited)
    pub peer_download_limit: u64,
    /// Allow burst transfers (use full bucket capacity)
    pub allow_burst: bool,
    /// Burst capacity multiplier
    pub burst_multiplier: f64,
}

impl Default for BandwidthConfig {
    fn default() -> Self {
        Self {
            global_upload_limit: 0,   // Unlimited
            global_download_limit: 0, // Unlimited
            peer_upload_limit: 0,     // Unlimited
            peer_download_limit: 0,   // Unlimited
            allow_burst: true,
            burst_multiplier: 2.0,
        }
    }
}

/// Statistics for bandwidth throttling
#[derive(Debug, Clone, Default)]
pub struct ThrottleStats {
    /// Total bytes uploaded
    pub bytes_uploaded: u64,
    /// Total bytes downloaded
    pub bytes_downloaded: u64,
    /// Number of times throttled
    pub throttle_count: u64,
    /// Total time spent waiting
    pub total_wait_time: Duration,
    /// Current upload rate (bytes/sec)
    pub current_upload_rate: f64,
    /// Current download rate (bytes/sec)
    pub current_download_rate: f64,
}

/// Per-peer throttle state
struct PeerThrottle {
    upload: Option<TokenBucket>,
    download: Option<TokenBucket>,
    priority: QosPriority,
}

/// Bandwidth throttle manager
pub struct BandwidthThrottle {
    config: BandwidthConfig,
    global_upload: Option<Arc<TokenBucket>>,
    global_download: Option<Arc<TokenBucket>>,
    peer_throttles: Arc<RwLock<HashMap<SocketAddr, PeerThrottle>>>,
    stats: Arc<RwLock<ThrottleStats>>,
}

impl BandwidthThrottle {
    /// Create a new bandwidth throttle
    pub fn new(config: BandwidthConfig) -> Self {
        let burst_capacity_multiplier = if config.allow_burst {
            config.burst_multiplier
        } else {
            1.0
        };

        let global_upload = if config.global_upload_limit > 0 {
            Some(Arc::new(TokenBucket::new(
                (config.global_upload_limit as f64 * burst_capacity_multiplier) as u64,
                config.global_upload_limit as f64,
            )))
        } else {
            None
        };

        let global_download = if config.global_download_limit > 0 {
            Some(Arc::new(TokenBucket::new(
                (config.global_download_limit as f64 * burst_capacity_multiplier) as u64,
                config.global_download_limit as f64,
            )))
        } else {
            None
        };

        Self {
            config,
            global_upload,
            global_download,
            peer_throttles: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(ThrottleStats::default())),
        }
    }

    /// Register a peer with optional priority
    pub fn register_peer(&self, addr: SocketAddr, priority: QosPriority) {
        let burst_multiplier = if self.config.allow_burst {
            self.config.burst_multiplier
        } else {
            1.0
        };

        let upload = if self.config.peer_upload_limit > 0 {
            let rate = self.config.peer_upload_limit as f64 * priority.multiplier();
            Some(TokenBucket::new((rate * burst_multiplier) as u64, rate))
        } else {
            None
        };

        let download = if self.config.peer_download_limit > 0 {
            let rate = self.config.peer_download_limit as f64 * priority.multiplier();
            Some(TokenBucket::new((rate * burst_multiplier) as u64, rate))
        } else {
            None
        };

        let throttle = PeerThrottle {
            upload,
            download,
            priority,
        };

        self.peer_throttles.write().insert(addr, throttle);
        debug!("Registered peer {} with priority {:?}", addr, priority);
    }

    /// Unregister a peer
    pub fn unregister_peer(&self, addr: &SocketAddr) {
        self.peer_throttles.write().remove(addr);
        debug!("Unregistered peer {}", addr);
    }

    /// Throttle upload (wait until bandwidth available)
    pub async fn throttle_upload(&self, addr: &SocketAddr, bytes: u64) {
        let start = Instant::now();

        // Global throttle
        if let Some(global) = &self.global_upload {
            global.consume(bytes).await;
        }

        // Per-peer throttle
        let upload_bucket = {
            let peer_throttles = self.peer_throttles.read();
            peer_throttles
                .get(addr)
                .and_then(|peer| peer.upload.clone())
        };

        if let Some(upload) = upload_bucket {
            upload.consume(bytes).await;
        }

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.bytes_uploaded += bytes;
            let wait_time = start.elapsed();
            if wait_time > Duration::from_millis(1) {
                stats.throttle_count += 1;
                stats.total_wait_time += wait_time;
            }
        }
    }

    /// Throttle download (wait until bandwidth available)
    pub async fn throttle_download(&self, addr: &SocketAddr, bytes: u64) {
        let start = Instant::now();

        // Global throttle
        if let Some(global) = &self.global_download {
            global.consume(bytes).await;
        }

        // Per-peer throttle
        let download_bucket = {
            let peer_throttles = self.peer_throttles.read();
            peer_throttles
                .get(addr)
                .and_then(|peer| peer.download.clone())
        };

        if let Some(download) = download_bucket {
            download.consume(bytes).await;
        }

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.bytes_downloaded += bytes;
            let wait_time = start.elapsed();
            if wait_time > Duration::from_millis(1) {
                stats.throttle_count += 1;
                stats.total_wait_time += wait_time;
            }
        }
    }

    /// Try to throttle upload (non-blocking, returns false if would block)
    pub fn try_throttle_upload(&self, addr: &SocketAddr, bytes: u64) -> bool {
        // Global throttle
        if let Some(global) = &self.global_upload {
            if !global.try_consume(bytes) {
                return false;
            }
        }

        // Per-peer throttle
        let peer_throttles = self.peer_throttles.read();
        if let Some(peer) = peer_throttles.get(addr) {
            if let Some(upload) = &peer.upload {
                if !upload.try_consume(bytes) {
                    return false;
                }
            }
        }

        // Update stats
        self.stats.write().bytes_uploaded += bytes;
        true
    }

    /// Try to throttle download (non-blocking)
    pub fn try_throttle_download(&self, addr: &SocketAddr, bytes: u64) -> bool {
        // Global throttle
        if let Some(global) = &self.global_download {
            if !global.try_consume(bytes) {
                return false;
            }
        }

        // Per-peer throttle
        let peer_throttles = self.peer_throttles.read();
        if let Some(peer) = peer_throttles.get(addr) {
            if let Some(download) = &peer.download {
                if !download.try_consume(bytes) {
                    return false;
                }
            }
        }

        // Update stats
        self.stats.write().bytes_downloaded += bytes;
        true
    }

    /// Update peer priority
    pub fn update_peer_priority(&self, addr: &SocketAddr, priority: QosPriority) {
        let mut peer_throttles = self.peer_throttles.write();
        if let Some(peer) = peer_throttles.get_mut(addr) {
            peer.priority = priority;
            debug!("Updated peer {} priority to {:?}", addr, priority);
        }
    }

    /// Get statistics
    pub fn stats(&self) -> ThrottleStats {
        self.stats.read().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        *self.stats.write() = ThrottleStats::default();
    }

    /// Get available upload bandwidth
    pub fn available_upload_bandwidth(&self) -> Option<u64> {
        self.global_upload.as_ref().map(|b| b.available_tokens())
    }

    /// Get available download bandwidth
    pub fn available_download_bandwidth(&self) -> Option<u64> {
        self.global_download.as_ref().map(|b| b.available_tokens())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_bucket() {
        let bucket = TokenBucket::new(100, 10.0);

        assert_eq!(bucket.available_tokens(), 100);
        assert!(bucket.try_consume(50));
        assert_eq!(bucket.available_tokens(), 50);
        assert!(bucket.try_consume(50));
        assert_eq!(bucket.available_tokens(), 0);
        assert!(!bucket.try_consume(1));
    }

    #[tokio::test]
    async fn test_token_bucket_refill() {
        let bucket = TokenBucket::new(100, 100.0); // 100 tokens/sec

        bucket.try_consume(100);
        assert_eq!(bucket.available_tokens(), 0);

        tokio::time::sleep(Duration::from_millis(500)).await;

        // Should have refilled ~50 tokens
        let available = bucket.available_tokens();
        assert!((45..=55).contains(&available), "Got {} tokens", available);
    }

    #[test]
    fn test_qos_priority() {
        assert_eq!(QosPriority::BestEffort.multiplier(), 0.5);
        assert_eq!(QosPriority::Normal.multiplier(), 1.0);
        assert_eq!(QosPriority::High.multiplier(), 2.0);
        assert_eq!(QosPriority::Critical.multiplier(), 4.0);
    }

    #[test]
    fn test_bandwidth_config_default() {
        let config = BandwidthConfig::default();
        assert_eq!(config.global_upload_limit, 0);
        assert!(config.allow_burst);
        assert_eq!(config.burst_multiplier, 2.0);
    }

    #[tokio::test]
    async fn test_bandwidth_throttle() {
        let config = BandwidthConfig {
            global_upload_limit: 1000,   // 1000 bytes/sec
            global_download_limit: 2000, // 2000 bytes/sec
            peer_upload_limit: 0,
            peer_download_limit: 0,
            allow_burst: false,
            burst_multiplier: 1.0,
        };

        let throttle = BandwidthThrottle::new(config);
        let addr: SocketAddr = "127.0.0.1:8080"
            .parse()
            .expect("test: parse socket address");

        throttle.register_peer(addr, QosPriority::Normal);

        // Should be able to upload immediately
        assert!(throttle.try_throttle_upload(&addr, 500));

        // Stats should be updated
        let stats = throttle.stats();
        assert_eq!(stats.bytes_uploaded, 500);
    }

    #[tokio::test]
    async fn test_peer_priority() {
        let config = BandwidthConfig {
            global_upload_limit: 0,
            global_download_limit: 0,
            peer_upload_limit: 1000, // 1000 bytes/sec
            peer_download_limit: 0,
            allow_burst: false,
            burst_multiplier: 1.0,
        };

        let throttle = BandwidthThrottle::new(config);
        let addr: SocketAddr = "127.0.0.1:8080"
            .parse()
            .expect("test: parse socket address");

        // High priority peer gets 2x bandwidth
        throttle.register_peer(addr, QosPriority::High);

        // Should be able to upload more due to higher priority
        assert!(throttle.try_throttle_upload(&addr, 1000));
    }
}
