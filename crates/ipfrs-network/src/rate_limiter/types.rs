//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use thiserror::Error;

use super::constants::TOKEN_SCALE;
use super::types_7::{PeerRateLimiterStats, RateLimiterStatsSnapshot};

/// Atomic statistics counters for `RateLimiter`.
#[derive(Debug, Default)]
pub struct AtomicRateLimiterStats {
    /// Requests that were allowed.
    pub total_allowed: AtomicU64,
    /// Requests that were throttled.
    pub total_throttled: AtomicU64,
    /// Requests that were rejected.
    pub total_rejected: AtomicU64,
}
impl AtomicRateLimiterStats {
    /// Return a point-in-time snapshot (relaxed loads — counters may drift
    /// slightly between reads, which is acceptable for monitoring purposes).
    pub fn snapshot(&self) -> RateLimiterStatsSnapshot {
        RateLimiterStatsSnapshot {
            total_allowed: self.total_allowed.load(Ordering::Relaxed),
            total_throttled: self.total_throttled.load(Ordering::Relaxed),
            total_rejected: self.total_rejected.load(Ordering::Relaxed),
        }
    }
}
/// Configuration for the [`PeerRateLimiter`].
#[derive(Debug, Clone)]
pub struct PeerRateLimiterConfig {
    /// Maximum tokens per peer bucket (default 1 000).
    pub per_peer_max_tokens: u64,
    /// Tokens added to each peer bucket per tick (default 100).
    pub per_peer_refill_rate: u64,
    /// Maximum tokens in the global bucket (default 10 000).
    pub global_max_tokens: u64,
    /// Tokens added to the global bucket per tick (default 1 000).
    pub global_refill_rate: u64,
    /// Consecutive violations that trigger an automatic peer block (default 10).
    pub auto_block_threshold: u32,
}
/// Signal emitted by [`BackpressureController`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpressureSignal {
    /// Queue is below the low watermark — consumers may speed up.
    Drain,
    /// Queue is in the normal operating range.
    Normal,
    /// Queue is approaching capacity — producers should slow down.
    Backpressure,
    /// Queue is full — producers must stop.
    Full,
}
/// Statistics tracked by the rate limiter
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RateLimiterStats {
    /// Total connection attempts
    pub total_attempts: u64,
    /// Connections allowed
    pub allowed: u64,
    /// Connections rate limited
    pub rate_limited: u64,
    /// Connections queued
    pub queued: u64,
    /// Current queue size
    pub current_queue_size: usize,
    /// Average rate (connections per second)
    pub avg_rate: f64,
    /// Current rate limit
    pub current_limit: f64,
    /// Tokens available
    pub tokens_available: f64,
}
/// Result of a rate-limit check.
#[derive(Clone, Debug, PartialEq)]
pub enum RateLimitResult {
    /// Request is permitted; `tokens_remaining` reflects the peer bucket after deduction.
    Allowed { tokens_remaining: u64 },
    /// Request is deferred; retry after `retry_after_ticks` ticks.
    Throttled { retry_after_ticks: u64 },
    /// Peer has been blocked (banned) due to repeated violations.
    Blocked,
}
/// Per-peer token bucket state.
#[derive(Debug, Clone)]
pub struct PeerLimiter {
    /// Identifier of the peer this limiter tracks.
    pub peer_id: String,
    /// Current token count.
    pub tokens: u64,
    /// Maximum token capacity.
    pub max_tokens: u64,
    /// Tokens replenished per tick.
    pub refill_rate: u64,
    /// Number of consecutive rate-limit violations.
    pub consecutive_violations: u32,
    /// Whether the peer is permanently blocked.
    pub blocked: bool,
    /// Total requests that were allowed.
    pub total_allowed: u64,
    /// Total requests that were throttled.
    pub total_throttled: u64,
}
impl PeerLimiter {
    /// Create a new peer limiter.
    pub fn new(peer_id: String, max_tokens: u64, refill_rate: u64) -> Self {
        Self {
            peer_id,
            tokens: max_tokens,
            max_tokens,
            refill_rate,
            consecutive_violations: 0,
            blocked: false,
            total_allowed: 0,
            total_throttled: 0,
        }
    }
    /// Attempt to consume `cost` tokens.
    ///
    /// Returns:
    /// - [`RateLimitResult::Blocked`] if the peer is blocked.
    /// - [`RateLimitResult::Allowed`] when sufficient tokens exist.
    /// - [`RateLimitResult::Throttled`] with the ticks-to-wait otherwise.
    ///   After 10 consecutive violations the peer is also blocked.
    pub fn try_consume(&mut self, cost: u64) -> RateLimitResult {
        if self.blocked {
            return RateLimitResult::Blocked;
        }
        if self.tokens >= cost {
            self.tokens -= cost;
            self.total_allowed += 1;
            return RateLimitResult::Allowed {
                tokens_remaining: self.tokens,
            };
        }
        self.consecutive_violations += 1;
        self.total_throttled += 1;
        if self.consecutive_violations >= 10 {
            self.blocked = true;
        }
        let shortfall = cost - self.tokens;
        let refill_rate = self.refill_rate.max(1);
        let retry_after_ticks = shortfall.div_ceil(refill_rate);
        RateLimitResult::Throttled { retry_after_ticks }
    }
    /// Replenish tokens for one tick.
    ///
    /// Also decrements `consecutive_violations` by 1 when the bucket is at
    /// least half-full and the peer is not blocked.
    pub fn refill(&mut self) {
        self.tokens = self
            .tokens
            .saturating_add(self.refill_rate)
            .min(self.max_tokens);
        if !self.blocked && self.consecutive_violations > 0 && self.tokens >= self.max_tokens / 2 {
            self.consecutive_violations -= 1;
        }
    }
}
/// Errors that can occur during rate limiting operations
#[derive(Debug, Error)]
pub enum RateLimiterError {
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("Peer blocked: {0}")]
    PeerBlocked(String),
}
/// Per-peer connection tracking
#[derive(Debug, Clone)]
pub(super) struct PeerTracking {
    /// Connection attempts in current window
    pub(super) attempts: Vec<Instant>,
    /// Successful connections
    pub(super) successes: u64,
    /// Failed connections
    pub(super) failures: u64,
    /// Last connection timestamp
    pub(super) last_connection: Option<Instant>,
}
impl PeerTracking {
    pub(super) fn new() -> Self {
        Self {
            attempts: Vec::new(),
            successes: 0,
            failures: 0,
            last_connection: None,
        }
    }
    /// Clean up old attempts outside the window
    pub(super) fn cleanup(&mut self, window: Duration) {
        let cutoff = Instant::now() - window;
        self.attempts.retain(|&t| t > cutoff);
    }
    /// Record a connection attempt
    pub(super) fn record_attempt(&mut self) {
        self.attempts.push(Instant::now());
        self.last_connection = Some(Instant::now());
    }
    /// Get current rate (attempts per second)
    pub(super) fn current_rate(&self, window: Duration) -> f64 {
        if self.attempts.is_empty() {
            return 0.0;
        }
        let now = Instant::now();
        let recent = self
            .attempts
            .iter()
            .filter(|&&t| now.duration_since(t) < window)
            .count();
        recent as f64 / window.as_secs_f64()
    }
}
/// Token bucket for rate limiting
#[derive(Debug)]
pub(super) struct TokenBucket {
    /// Current number of tokens
    pub(super) tokens: f64,
    /// Maximum tokens (burst size)
    pub(super) capacity: f64,
    /// Token refill rate (tokens per second)
    pub(super) rate: f64,
    /// Last refill timestamp
    pub(super) last_refill: Instant,
}
impl TokenBucket {
    pub(super) fn new(rate: f64, capacity: usize) -> Self {
        Self {
            tokens: capacity as f64,
            capacity: capacity as f64,
            rate,
            last_refill: Instant::now(),
        }
    }
    /// Refill tokens based on elapsed time
    pub(super) fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let new_tokens = elapsed * self.rate;
        self.tokens = (self.tokens + new_tokens).min(self.capacity);
        self.last_refill = now;
    }
    /// Try to consume a token
    pub(super) fn try_consume(&mut self, count: f64) -> bool {
        self.refill();
        if self.tokens >= count {
            self.tokens -= count;
            true
        } else {
            false
        }
    }
    /// Get current token count
    pub(super) fn available(&mut self) -> f64 {
        self.refill();
        self.tokens
    }
    /// Update rate dynamically
    pub(super) fn update_rate(&mut self, new_rate: f64) {
        self.refill();
        self.rate = new_rate;
    }
}
/// Priority level for connections
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConnectionPriority {
    /// Critical connections (bootstrap nodes, important peers)
    Critical,
    /// High priority connections
    High,
    /// Normal priority connections
    Normal,
    /// Low priority connections
    Low,
}
impl ConnectionPriority {
    /// Get the rate multiplier for this priority level
    pub fn rate_multiplier(&self) -> f64 {
        match self {
            Self::Critical => 2.0,
            Self::High => 1.5,
            Self::Normal => 1.0,
            Self::Low => 0.5,
        }
    }
}
/// Global token bucket shared across all peers.
#[derive(Debug, Clone)]
pub struct GlobalLimiter {
    /// Current token count.
    pub tokens: u64,
    /// Maximum token capacity.
    pub max_tokens: u64,
    /// Tokens replenished per tick.
    pub refill_rate: u64,
}
impl GlobalLimiter {
    /// Create a new global limiter.
    pub fn new(max_tokens: u64, refill_rate: u64) -> Self {
        Self {
            tokens: max_tokens,
            max_tokens,
            refill_rate,
        }
    }
    /// Try to consume `cost` tokens from the global bucket.
    ///
    /// Returns `false` when there are not enough tokens; deducts and returns
    /// `true` on success.
    pub fn try_consume(&mut self, cost: u64) -> bool {
        if self.tokens >= cost {
            self.tokens -= cost;
            true
        } else {
            false
        }
    }
    /// Replenish tokens for one tick.
    pub fn refill(&mut self) {
        self.tokens = self
            .tokens
            .saturating_add(self.refill_rate)
            .min(self.max_tokens);
    }
}
/// Decision returned by `RateLimiter::check`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RateLimitDecision {
    /// Request is allowed immediately.
    Allowed,
    /// Request is throttled; caller should retry after the given milliseconds.
    Throttled {
        /// Approximate milliseconds until tokens are available.
        retry_after_ms: u64,
    },
    /// Request is rejected outright (e.g., peer has never been registered and
    /// global bucket is exhausted).
    Rejected,
}
/// A token-bucket rate limiter enforcing per-peer **and** global request limits.
///
/// Every call to [`PeerRateLimiter::check`] first deducts from the shared
/// global bucket and then from the individual peer bucket.  Either bucket being
/// exhausted causes a [`RateLimitResult::Throttled`] response.
///
/// Advance time with [`PeerRateLimiter::tick`]; each tick refills both the
/// global bucket and all peer buckets.
pub struct PeerRateLimiter {
    /// Per-peer token buckets.
    pub peers: HashMap<String, PeerLimiter>,
    /// Shared global token bucket.
    pub global: GlobalLimiter,
    /// Active configuration.
    pub config: PeerRateLimiterConfig,
}
impl PeerRateLimiter {
    /// Create a new limiter with the provided configuration.
    pub fn new(config: PeerRateLimiterConfig) -> Self {
        let global = GlobalLimiter::new(config.global_max_tokens, config.global_refill_rate);
        Self {
            peers: HashMap::new(),
            global,
            config,
        }
    }
    /// Check whether a request from `peer_id` costing `cost` tokens is
    /// permitted.
    ///
    /// Auto-creates the peer limiter on first encounter.  The global bucket is
    /// checked first; if it is exhausted, [`RateLimitResult::Throttled { retry_after_ticks: 1 }`]
    /// is returned immediately without touching the peer bucket.
    pub fn check(&mut self, peer_id: &str, cost: u64) -> RateLimitResult {
        if !self.peers.contains_key(peer_id) {
            let limiter = PeerLimiter::new(
                peer_id.to_string(),
                self.config.per_peer_max_tokens,
                self.config.per_peer_refill_rate,
            );
            self.peers.insert(peer_id.to_string(), limiter);
        }
        if !self.global.try_consume(cost) {
            return RateLimitResult::Throttled {
                retry_after_ticks: 1,
            };
        }
        let peer = self
            .peers
            .get_mut(peer_id)
            .expect("peer was inserted above");
        peer.try_consume(cost)
    }
    /// Advance time by one tick: refill all peer buckets and the global bucket.
    pub fn tick(&mut self) {
        self.global.refill();
        for peer in self.peers.values_mut() {
            peer.refill();
        }
    }
    /// Immediately block `peer_id`.  Creates the peer entry if it does not
    /// exist yet.
    pub fn block_peer(&mut self, peer_id: &str) {
        let entry = self.peers.entry(peer_id.to_string()).or_insert_with(|| {
            PeerLimiter::new(
                peer_id.to_string(),
                self.config.per_peer_max_tokens,
                self.config.per_peer_refill_rate,
            )
        });
        entry.blocked = true;
    }
    /// Remove the block on `peer_id` and reset its violation counter.
    ///
    /// Returns `false` if the peer is not found.
    pub fn unblock_peer(&mut self, peer_id: &str) -> bool {
        match self.peers.get_mut(peer_id) {
            Some(peer) => {
                peer.blocked = false;
                peer.consecutive_violations = 0;
                true
            }
            None => false,
        }
    }
    /// Remove all state for `peer_id`.
    ///
    /// Returns `true` if the peer was present, `false` otherwise.
    pub fn remove_peer(&mut self, peer_id: &str) -> bool {
        self.peers.remove(peer_id).is_some()
    }
    /// Return aggregated statistics.
    pub fn stats(&self) -> PeerRateLimiterStats {
        let total_peers = self.peers.len();
        let blocked_peers = self.peers.values().filter(|p| p.blocked).count();
        let total_allowed = self.peers.values().map(|p| p.total_allowed).sum();
        let total_throttled = self.peers.values().map(|p| p.total_throttled).sum();
        PeerRateLimiterStats {
            total_peers,
            blocked_peers,
            total_allowed,
            total_throttled,
        }
    }
}
/// Connection rate limiter
pub struct ConnectionRateLimiter {
    pub(super) config: RateLimiterConfig,
    pub(super) bucket: Arc<RwLock<TokenBucket>>,
    pub(super) peer_tracking: Arc<RwLock<HashMap<String, PeerTracking>>>,
    pub(super) stats: Arc<RwLock<RateLimiterStats>>,
    pub(super) queue: Arc<RwLock<Vec<(String, ConnectionPriority, Instant)>>>,
}
impl ConnectionRateLimiter {
    /// Create a new connection rate limiter
    pub fn new(config: RateLimiterConfig) -> Self {
        let bucket = TokenBucket::new(config.max_rate, config.burst_size);
        Self {
            config,
            bucket: Arc::new(RwLock::new(bucket)),
            peer_tracking: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(RateLimiterStats::default())),
            queue: Arc::new(RwLock::new(Vec::new())),
        }
    }
    /// Check if a connection is allowed
    pub async fn allow_connection(&self, peer_id: &str) -> bool {
        self.allow_connection_with_priority(peer_id, ConnectionPriority::Normal)
            .await
    }
    /// Check if a connection is allowed with specific priority
    pub async fn allow_connection_with_priority(
        &self,
        peer_id: &str,
        priority: ConnectionPriority,
    ) -> bool {
        let mut stats = self.stats.write();
        stats.total_attempts += 1;
        if self.config.enable_per_peer_limits {
            let mut tracking = self.peer_tracking.write();
            let peer_track = tracking
                .entry(peer_id.to_string())
                .or_insert_with(PeerTracking::new);
            peer_track.cleanup(self.config.peer_window);
            let current_rate = peer_track.current_rate(self.config.peer_window);
            if current_rate >= self.config.max_per_peer_rate {
                stats.rate_limited += 1;
                return false;
            }
        }
        let cost = 1.0 / priority.rate_multiplier();
        let mut bucket = self.bucket.write();
        if bucket.try_consume(cost) {
            if self.config.enable_per_peer_limits {
                let mut tracking = self.peer_tracking.write();
                if let Some(peer_track) = tracking.get_mut(peer_id) {
                    peer_track.record_attempt();
                }
            }
            stats.allowed += 1;
            stats.tokens_available = bucket.available();
            true
        } else {
            stats.rate_limited += 1;
            if self.config.enable_queuing {
                let mut queue = self.queue.write();
                if queue.len() < self.config.max_queue_size {
                    queue.push((peer_id.to_string(), priority, Instant::now()));
                    stats.queued += 1;
                    stats.current_queue_size = queue.len();
                }
            }
            false
        }
    }
    /// Record a successful connection
    pub fn record_success(&self, peer_id: &str) {
        if !self.config.enable_per_peer_limits {
            return;
        }
        let mut tracking = self.peer_tracking.write();
        if let Some(peer_track) = tracking.get_mut(peer_id) {
            peer_track.successes += 1;
            if self.config.enable_adaptive {
                self.adapt_rate_on_success();
            }
        }
    }
    /// Record a failed connection
    pub fn record_failure(&self, peer_id: &str) {
        if !self.config.enable_per_peer_limits {
            return;
        }
        let mut tracking = self.peer_tracking.write();
        if let Some(peer_track) = tracking.get_mut(peer_id) {
            peer_track.failures += 1;
            if self.config.enable_adaptive {
                self.adapt_rate_on_failure();
            }
        }
    }
    /// Adapt rate upward on success
    pub(super) fn adapt_rate_on_success(&self) {
        let mut bucket = self.bucket.write();
        let current_rate = bucket.rate;
        let new_rate =
            (current_rate * (1.0 + self.config.adaptive_factor)).min(self.config.max_adaptive_rate);
        if new_rate != current_rate {
            bucket.update_rate(new_rate);
            let mut stats = self.stats.write();
            stats.current_limit = new_rate;
        }
    }
    /// Adapt rate downward on failure
    pub(super) fn adapt_rate_on_failure(&self) {
        let mut bucket = self.bucket.write();
        let current_rate = bucket.rate;
        let new_rate =
            (current_rate * (1.0 - self.config.adaptive_factor)).max(self.config.min_rate);
        if new_rate != current_rate {
            bucket.update_rate(new_rate);
            let mut stats = self.stats.write();
            stats.current_limit = new_rate;
        }
    }
    /// Process queued connections
    pub async fn process_queue(&self) -> Vec<String> {
        let mut queue = self.queue.write();
        let mut bucket = self.bucket.write();
        let mut allowed = Vec::new();
        queue.sort_by(|a, b| match (a.1, b.1) {
            (ConnectionPriority::Critical, ConnectionPriority::Critical) => a.2.cmp(&b.2),
            (ConnectionPriority::Critical, _) => std::cmp::Ordering::Less,
            (_, ConnectionPriority::Critical) => std::cmp::Ordering::Greater,
            (ConnectionPriority::High, ConnectionPriority::High) => a.2.cmp(&b.2),
            (ConnectionPriority::High, _) => std::cmp::Ordering::Less,
            (_, ConnectionPriority::High) => std::cmp::Ordering::Greater,
            _ => a.2.cmp(&b.2),
        });
        queue.retain(|(peer_id, priority, _)| {
            let cost = 1.0 / priority.rate_multiplier();
            if bucket.try_consume(cost) {
                allowed.push(peer_id.clone());
                false
            } else {
                true
            }
        });
        let mut stats = self.stats.write();
        stats.current_queue_size = queue.len();
        allowed
    }
    /// Get current statistics
    pub fn stats(&self) -> RateLimiterStats {
        let mut stats = self.stats.read().clone();
        let bucket = self.bucket.write();
        stats.current_limit = bucket.rate;
        stats.tokens_available = bucket.tokens;
        if stats.total_attempts > 0 {
            stats.avg_rate = stats.allowed as f64 / (stats.total_attempts as f64 / bucket.rate);
        }
        stats
    }
    /// Get per-peer statistics
    pub fn peer_stats(&self, peer_id: &str) -> Option<(u64, u64, f64)> {
        let tracking = self.peer_tracking.read();
        tracking.get(peer_id).map(|track| {
            (
                track.successes,
                track.failures,
                track.current_rate(self.config.peer_window),
            )
        })
    }
    /// Reset rate limiter state
    pub fn reset(&self) {
        let mut bucket = self.bucket.write();
        bucket.tokens = bucket.capacity;
        bucket.last_refill = Instant::now();
        self.peer_tracking.write().clear();
        self.queue.write().clear();
        let mut stats = self.stats.write();
        *stats = RateLimiterStats::default();
    }
}
/// Lock-free backpressure controller driven by an atomic queue depth counter.
///
/// Producers call [`push`](BackpressureController::push) and inspect the
/// returned [`BackpressureSignal`]; consumers call
/// [`pop`](BackpressureController::pop).
#[derive(Debug)]
pub struct BackpressureController {
    /// Current queue depth.
    pub(super) queue_depth: AtomicU64,
    /// Hard limit — `Full` above this.
    pub(super) max_queue_depth: u64,
    /// Drain signal below this.
    pub(super) low_watermark: u64,
    /// Backpressure signal at-or-above this (and below max).
    pub(super) high_watermark: u64,
}
impl BackpressureController {
    /// Create with default watermarks (low=2000, high=8000, max=10000).
    pub fn new() -> Self {
        Self::with_config(10_000, 2_000, 8_000)
    }
    /// Create with explicit parameters.
    ///
    /// # Panics
    ///
    /// Panics (in debug) if watermarks are inconsistent.
    pub fn with_config(max_queue_depth: u64, low_watermark: u64, high_watermark: u64) -> Self {
        debug_assert!(
            low_watermark <= high_watermark && high_watermark <= max_queue_depth,
            "watermarks must satisfy low <= high <= max"
        );
        Self {
            queue_depth: AtomicU64::new(0),
            max_queue_depth,
            low_watermark,
            high_watermark,
        }
    }
    /// Compute the signal for a given depth value.
    pub(super) fn signal_for(&self, depth: u64) -> BackpressureSignal {
        if depth >= self.max_queue_depth {
            BackpressureSignal::Full
        } else if depth >= self.high_watermark {
            BackpressureSignal::Backpressure
        } else if depth < self.low_watermark {
            BackpressureSignal::Drain
        } else {
            BackpressureSignal::Normal
        }
    }
    /// Increment queue depth and return the resulting signal.
    ///
    /// If the queue is already `Full` the depth is still incremented (producers
    /// are responsible for checking the returned signal and not enqueuing).
    pub fn push(&self) -> BackpressureSignal {
        let new_depth = self.queue_depth.fetch_add(1, Ordering::Relaxed) + 1;
        self.signal_for(new_depth)
    }
    /// Decrement queue depth (saturating at zero).
    pub fn pop(&self) {
        self.queue_depth
            .fetch_update(Ordering::Release, Ordering::Relaxed, |d| {
                Some(d.saturating_sub(1))
            })
            .ok();
    }
    /// Return the current queue depth.
    pub fn depth(&self) -> u64 {
        self.queue_depth.load(Ordering::Relaxed)
    }
    /// Return the current signal without modifying depth.
    pub fn signal(&self) -> BackpressureSignal {
        self.signal_for(self.depth())
    }
}
/// Configuration for the connection rate limiter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimiterConfig {
    /// Maximum connection rate (connections per second)
    pub max_rate: f64,
    /// Maximum burst size (tokens)
    pub burst_size: usize,
    /// Enable per-peer rate limiting
    pub enable_per_peer_limits: bool,
    /// Maximum connections per peer per second
    pub max_per_peer_rate: f64,
    /// Enable adaptive rate limiting
    pub enable_adaptive: bool,
    /// Adjustment factor for adaptive limiting (0.0 to 1.0)
    pub adaptive_factor: f64,
    /// Minimum rate (connections per second) when adapting
    pub min_rate: f64,
    /// Maximum rate (connections per second) when adapting
    pub max_adaptive_rate: f64,
    /// Enable connection queuing when rate limited
    pub enable_queuing: bool,
    /// Maximum queue size for pending connections
    pub max_queue_size: usize,
    /// Time window for per-peer tracking
    pub peer_window: Duration,
}
impl RateLimiterConfig {
    /// Configuration for aggressive rate limiting (low rates)
    pub fn conservative() -> Self {
        Self {
            max_rate: 5.0,
            burst_size: 10,
            max_per_peer_rate: 1.0,
            max_queue_size: 50,
            ..Default::default()
        }
    }
    /// Configuration for permissive rate limiting (high rates)
    pub fn permissive() -> Self {
        Self {
            max_rate: 50.0,
            burst_size: 100,
            max_per_peer_rate: 10.0,
            max_queue_size: 200,
            ..Default::default()
        }
    }
    /// Configuration with adaptive rate limiting enabled
    pub fn adaptive() -> Self {
        Self {
            enable_adaptive: true,
            adaptive_factor: 0.2,
            min_rate: 2.0,
            max_adaptive_rate: 50.0,
            ..Default::default()
        }
    }
}
/// Atomic token-bucket rate limiter.
///
/// Tokens are stored as fixed-point integers (1 logical token == `TOKEN_SCALE`
/// internal units) so that all state mutations can use lock-free atomics.
/// The only mutex is on `last_refill` which is updated at most once per
/// `try_acquire` call and therefore contention is low.
pub struct AtomicTokenBucket {
    /// Maximum tokens (logical).
    pub(super) capacity: u64,
    /// Current tokens stored as `tokens * TOKEN_SCALE`.
    pub(super) tokens: AtomicU64,
    /// Tokens per second (logical).
    pub(super) refill_rate: u64,
    /// Last refill timestamp protected by a mutex.
    pub(super) last_refill: Mutex<Instant>,
}
impl AtomicTokenBucket {
    /// Create a new bucket pre-filled to `capacity`.
    pub fn new(capacity: u64, refill_rate: u64) -> Self {
        Self {
            capacity,
            tokens: AtomicU64::new(capacity.saturating_mul(TOKEN_SCALE)),
            refill_rate,
            last_refill: Mutex::new(Instant::now()),
        }
    }
    /// Refill the bucket based on elapsed time since last refill.
    pub fn refill(&self) {
        let Ok(mut last) = self.last_refill.lock() else {
            return;
        };
        let now = Instant::now();
        let elapsed_secs = now.duration_since(*last).as_secs_f64();
        if elapsed_secs <= 0.0 {
            return;
        }
        *last = now;
        let new_tokens_scaled =
            (elapsed_secs * self.refill_rate as f64 * TOKEN_SCALE as f64) as u64;
        if new_tokens_scaled == 0 {
            return;
        }
        let cap_scaled = self.capacity.saturating_mul(TOKEN_SCALE);
        let mut current = self.tokens.load(Ordering::Relaxed);
        loop {
            let new_val = current.saturating_add(new_tokens_scaled).min(cap_scaled);
            match self.tokens.compare_exchange_weak(
                current,
                new_val,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }
    /// Non-blocking acquire attempt. Returns `true` if `tokens` logical tokens
    /// were successfully deducted.
    pub fn try_acquire(&self, tokens: u64) -> bool {
        self.refill();
        let cost = tokens.saturating_mul(TOKEN_SCALE);
        let mut current = self.tokens.load(Ordering::Relaxed);
        loop {
            if current < cost {
                return false;
            }
            match self.tokens.compare_exchange_weak(
                current,
                current - cost,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }
    /// Return current logical token count.
    pub fn available_tokens(&self) -> u64 {
        self.tokens.load(Ordering::Relaxed) / TOKEN_SCALE
    }
    /// Ratio of current tokens to capacity in `[0.0, 1.0]`.
    pub fn fill_ratio(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.tokens.load(Ordering::Relaxed) as f64
            / self.capacity.saturating_mul(TOKEN_SCALE) as f64
    }
}
