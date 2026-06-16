//! Peer management with scoring, selection, and blacklisting
//!
//! Implements advanced peer management for optimal block exchange:
//! - Peer scoring with latency, bandwidth, and reliability metrics
//! - Multiple peer selection strategies
//! - Automatic blacklisting of misbehaving nodes
//!
//! # Example
//!
//! ```
//! use ipfrs_transport::{PeerManager, PeerScoringConfig};
//! use std::time::Duration;
//!
//! // Create a peer manager with default configuration
//! let config = PeerScoringConfig::default();
//! let mut manager = PeerManager::new(config);
//!
//! // Add some peers
//! manager.add_peer("peer1".to_string());
//! manager.add_peer("peer2".to_string());
//!
//! // Record successful data transfer (1000 bytes, 10ms latency)
//! manager.record_success(&"peer1".to_string(), 1000, Duration::from_millis(10));
//!
//! // Record a failure
//! manager.record_failure(&"peer2".to_string());
//!
//! // Select the best peer based on score
//! if let Some(best_peer) = manager.best_peer() {
//!     println!("Selected peer: {}", best_peer);
//! }
//! ```

use ipfrs_core::Cid;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Peer identifier
pub type PeerId = String;

/// Peer scoring configuration
#[derive(Debug, Clone)]
pub struct PeerScoringConfig {
    /// Weight for latency in score calculation (0.0-1.0)
    pub latency_weight: f64,
    /// Weight for bandwidth in score calculation (0.0-1.0)
    pub bandwidth_weight: f64,
    /// Weight for reliability in score calculation (0.0-1.0)
    pub reliability_weight: f64,
    /// Decay factor for exponential moving average (0.0-1.0)
    /// Higher = more weight to recent observations
    pub ewma_alpha: f64,
    /// Score decay per second for inactive peers
    pub inactivity_decay: f64,
    /// Minimum score before automatic blacklisting
    pub min_score: f64,
    /// Maximum failures before blacklisting
    pub max_failures: u32,
}

impl Default for PeerScoringConfig {
    fn default() -> Self {
        Self {
            latency_weight: 0.3,
            bandwidth_weight: 0.4,
            reliability_weight: 0.3,
            ewma_alpha: 0.3,
            inactivity_decay: 0.01,
            min_score: 0.1,
            max_failures: 10,
        }
    }
}

/// Peer performance metrics
#[derive(Debug, Clone)]
pub struct PeerMetrics {
    /// Average round-trip latency (ms)
    pub latency_ms: f64,
    /// Average bandwidth (bytes/second)
    pub bandwidth_bps: f64,
    /// Success rate (0.0-1.0)
    pub reliability: f64,
    /// Total requests sent
    pub requests_sent: u64,
    /// Total requests completed successfully
    pub requests_completed: u64,
    /// Total requests failed
    pub requests_failed: u64,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_recv: u64,
    /// Last update time
    pub last_update: Instant,
    /// Last successful interaction
    pub last_success: Option<Instant>,
    /// Consecutive failures
    pub consecutive_failures: u32,
}

impl Default for PeerMetrics {
    fn default() -> Self {
        Self {
            latency_ms: 100.0,          // Default assumption
            bandwidth_bps: 1_000_000.0, // 1 MB/s default
            reliability: 1.0,           // Start optimistic
            requests_sent: 0,
            requests_completed: 0,
            requests_failed: 0,
            bytes_sent: 0,
            bytes_recv: 0,
            last_update: Instant::now(),
            last_success: None,
            consecutive_failures: 0,
        }
    }
}

impl PeerMetrics {
    /// Update latency with exponential moving average
    pub fn update_latency(&mut self, latency_ms: f64, alpha: f64) {
        self.latency_ms = alpha * latency_ms + (1.0 - alpha) * self.latency_ms;
        self.last_update = Instant::now();
    }

    /// Update bandwidth with exponential moving average
    pub fn update_bandwidth(&mut self, bytes: u64, duration: Duration, alpha: f64) {
        if duration.as_secs_f64() > 0.0 {
            let bps = bytes as f64 / duration.as_secs_f64();
            self.bandwidth_bps = alpha * bps + (1.0 - alpha) * self.bandwidth_bps;
        }
        self.last_update = Instant::now();
    }

    /// Record a successful request
    pub fn record_success(&mut self, bytes: u64, latency: Duration, alpha: f64) {
        self.requests_completed += 1;
        self.bytes_recv += bytes;
        self.consecutive_failures = 0;
        self.last_success = Some(Instant::now());
        self.last_update = Instant::now();

        // Update reliability
        let total = self.requests_completed + self.requests_failed;
        if total > 0 {
            self.reliability = self.requests_completed as f64 / total as f64;
        }

        // Update latency
        self.update_latency(latency.as_secs_f64() * 1000.0, alpha);

        // Update bandwidth
        self.update_bandwidth(bytes, latency, alpha);
    }

    /// Record a failed request
    pub fn record_failure(&mut self, alpha: f64) {
        self.requests_failed += 1;
        self.consecutive_failures += 1;
        self.last_update = Instant::now();

        // Update reliability
        let total = self.requests_completed + self.requests_failed;
        if total > 0 {
            self.reliability = alpha * 0.0 + (1.0 - alpha) * self.reliability;
        }
    }

    /// Calculate composite score
    pub fn score(&self, config: &PeerScoringConfig) -> f64 {
        // Normalize latency (lower is better, 10ms=1.0, 1000ms=0.01)
        let latency_score = (10.0 / self.latency_ms.max(1.0)).min(1.0);

        // Normalize bandwidth (higher is better, 10MB/s=1.0)
        let bandwidth_score = (self.bandwidth_bps / 10_000_000.0).min(1.0);

        // Reliability is already 0-1
        let reliability_score = self.reliability;

        // Apply inactivity decay
        let time_since_update = self.last_update.elapsed().as_secs_f64();
        let decay = (1.0 - config.inactivity_decay * time_since_update).max(0.1);

        // Weighted sum
        let score = (config.latency_weight * latency_score
            + config.bandwidth_weight * bandwidth_score
            + config.reliability_weight * reliability_score)
            * decay;

        score.clamp(0.0, 1.0)
    }

    /// Debt ratio (how much we owe them)
    pub fn debt_ratio(&self) -> f64 {
        if self.bytes_sent == 0 {
            return f64::INFINITY;
        }
        self.bytes_recv as f64 / self.bytes_sent as f64
    }
}

/// Blacklist entry
#[derive(Debug, Clone)]
pub struct BlacklistEntry {
    /// When the peer was blacklisted
    pub blacklisted_at: Instant,
    /// Why the peer was blacklisted
    pub reason: BlacklistReason,
    /// When the ban expires (None = permanent)
    pub expires_at: Option<Instant>,
}

/// Reasons for blacklisting
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlacklistReason {
    /// Too many consecutive failures
    RepeatedFailures,
    /// Score dropped below minimum
    LowScore,
    /// Sent invalid data
    InvalidData,
    /// Protocol violation
    ProtocolViolation,
    /// Manually blacklisted
    Manual,
}

impl std::fmt::Display for BlacklistReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlacklistReason::RepeatedFailures => write!(f, "repeated failures"),
            BlacklistReason::LowScore => write!(f, "low score"),
            BlacklistReason::InvalidData => write!(f, "invalid data"),
            BlacklistReason::ProtocolViolation => write!(f, "protocol violation"),
            BlacklistReason::Manual => write!(f, "manual"),
        }
    }
}

/// Peer selection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionStrategy {
    /// Select peers with lowest latency
    FastestFirst,
    /// Select peers with highest bandwidth
    HighestBandwidth,
    /// Select peers with best composite score
    BestScore,
    /// Round-robin for fairness
    RoundRobin,
    /// Random selection
    Random,
    /// Least loaded (fewest active requests)
    LeastLoaded,
}

/// Peer state information
#[derive(Debug, Clone)]
pub struct PeerState {
    /// Peer ID
    pub id: PeerId,
    /// Performance metrics
    pub metrics: PeerMetrics,
    /// CIDs this peer has (from HAVE messages)
    pub has_cids: HashSet<Cid>,
    /// CIDs this peer doesn't have (from DONT_HAVE messages)
    pub doesnt_have_cids: HashSet<Cid>,
    /// Number of active requests to this peer
    pub active_requests: u32,
    /// Maximum concurrent requests allowed
    pub max_concurrent: u32,
    /// Is this peer connected
    pub connected: bool,
}

impl PeerState {
    /// Create a new peer state
    pub fn new(id: PeerId) -> Self {
        Self {
            id,
            metrics: PeerMetrics::default(),
            has_cids: HashSet::new(),
            doesnt_have_cids: HashSet::new(),
            active_requests: 0,
            max_concurrent: 16, // Default max concurrent requests
            connected: true,
        }
    }

    /// Check if peer can accept more requests
    pub fn can_accept_request(&self) -> bool {
        self.connected && self.active_requests < self.max_concurrent
    }

    /// Check if peer might have a CID
    pub fn might_have(&self, cid: &Cid) -> bool {
        !self.doesnt_have_cids.contains(cid)
    }

    /// Check if peer definitely has a CID
    pub fn has(&self, cid: &Cid) -> bool {
        self.has_cids.contains(cid)
    }
}

/// Peer manager for tracking and selecting peers
pub struct PeerManager {
    /// All known peers
    peers: HashMap<PeerId, PeerState>,
    /// Blacklisted peers
    blacklist: HashMap<PeerId, BlacklistEntry>,
    /// Configuration
    config: PeerScoringConfig,
    /// Round-robin index
    round_robin_idx: usize,
    /// Recently used peers (for round-robin) - reserved for future use
    #[allow(dead_code)]
    recent_peers: VecDeque<PeerId>,
}

impl PeerManager {
    /// Create a new peer manager
    pub fn new(config: PeerScoringConfig) -> Self {
        Self {
            peers: HashMap::new(),
            blacklist: HashMap::new(),
            config,
            round_robin_idx: 0,
            recent_peers: VecDeque::with_capacity(100),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(PeerScoringConfig::default())
    }

    /// Add or update a peer
    pub fn add_peer(&mut self, id: PeerId) {
        if !self.is_blacklisted(&id) {
            self.peers
                .entry(id.clone())
                .or_insert_with(|| PeerState::new(id));
        }
    }

    /// Remove a peer
    pub fn remove_peer(&mut self, id: &PeerId) {
        self.peers.remove(id);
    }

    /// Get peer state
    pub fn get_peer(&self, id: &PeerId) -> Option<&PeerState> {
        self.peers.get(id)
    }

    /// Get mutable peer state
    pub fn get_peer_mut(&mut self, id: &PeerId) -> Option<&mut PeerState> {
        self.peers.get_mut(id)
    }

    /// Record successful request
    pub fn record_success(&mut self, peer_id: &PeerId, bytes: u64, latency: Duration) {
        if let Some(peer) = self.peers.get_mut(peer_id) {
            peer.metrics
                .record_success(bytes, latency, self.config.ewma_alpha);
            peer.active_requests = peer.active_requests.saturating_sub(1);
        }
    }

    /// Record failed request
    pub fn record_failure(&mut self, peer_id: &PeerId) {
        if let Some(peer) = self.peers.get_mut(peer_id) {
            peer.metrics.record_failure(self.config.ewma_alpha);
            peer.active_requests = peer.active_requests.saturating_sub(1);

            // Check for automatic blacklisting
            if peer.metrics.consecutive_failures >= self.config.max_failures {
                self.blacklist_peer(
                    peer_id.clone(),
                    BlacklistReason::RepeatedFailures,
                    Some(Duration::from_secs(3600)), // 1 hour ban
                );
            } else if peer.metrics.score(&self.config) < self.config.min_score {
                self.blacklist_peer(
                    peer_id.clone(),
                    BlacklistReason::LowScore,
                    Some(Duration::from_secs(1800)), // 30 minute ban
                );
            }
        }
    }

    /// Record that peer has a CID
    pub fn record_has(&mut self, peer_id: &PeerId, cid: Cid) {
        if let Some(peer) = self.peers.get_mut(peer_id) {
            peer.has_cids.insert(cid);
            peer.doesnt_have_cids.remove(&cid);
        }
    }

    /// Record that peer doesn't have a CID
    pub fn record_doesnt_have(&mut self, peer_id: &PeerId, cid: Cid) {
        if let Some(peer) = self.peers.get_mut(peer_id) {
            peer.doesnt_have_cids.insert(cid);
            peer.has_cids.remove(&cid);
        }
    }

    /// Mark request sent to peer
    pub fn mark_request_sent(&mut self, peer_id: &PeerId) {
        if let Some(peer) = self.peers.get_mut(peer_id) {
            peer.metrics.requests_sent += 1;
            peer.active_requests += 1;
        }
    }

    /// Blacklist a peer
    pub fn blacklist_peer(
        &mut self,
        peer_id: PeerId,
        reason: BlacklistReason,
        duration: Option<Duration>,
    ) {
        let expires_at = duration.map(|d| Instant::now() + d);
        self.blacklist.insert(
            peer_id.clone(),
            BlacklistEntry {
                blacklisted_at: Instant::now(),
                reason,
                expires_at,
            },
        );
        self.peers.remove(&peer_id);
    }

    /// Remove peer from blacklist
    pub fn unblacklist_peer(&mut self, peer_id: &PeerId) {
        self.blacklist.remove(peer_id);
    }

    /// Check if peer is blacklisted
    pub fn is_blacklisted(&self, peer_id: &PeerId) -> bool {
        if let Some(entry) = self.blacklist.get(peer_id) {
            // Check if ban has expired
            if let Some(expires) = entry.expires_at {
                if Instant::now() >= expires {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }

    /// Clean up expired blacklist entries
    pub fn cleanup_blacklist(&mut self) {
        let now = Instant::now();
        self.blacklist
            .retain(|_, entry| entry.expires_at.is_none_or(|exp| exp > now));
    }

    /// Select peers for a request
    pub fn select_peers(
        &mut self,
        cid: &Cid,
        count: usize,
        strategy: SelectionStrategy,
    ) -> Vec<PeerId> {
        self.cleanup_blacklist();

        // Filter available peers
        let available: Vec<_> = self
            .peers
            .values()
            .filter(|p| p.can_accept_request() && p.might_have(cid))
            .collect();

        if available.is_empty() {
            return Vec::new();
        }

        match strategy {
            SelectionStrategy::FastestFirst => {
                let mut sorted: Vec<_> = available.into_iter().collect();
                sorted.sort_by(|a, b| {
                    a.metrics
                        .latency_ms
                        .partial_cmp(&b.metrics.latency_ms)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                sorted
                    .into_iter()
                    .take(count)
                    .map(|p| p.id.clone())
                    .collect()
            }
            SelectionStrategy::HighestBandwidth => {
                let mut sorted: Vec<_> = available.into_iter().collect();
                sorted.sort_by(|a, b| {
                    b.metrics
                        .bandwidth_bps
                        .partial_cmp(&a.metrics.bandwidth_bps)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                sorted
                    .into_iter()
                    .take(count)
                    .map(|p| p.id.clone())
                    .collect()
            }
            SelectionStrategy::BestScore => {
                let mut sorted: Vec<_> = available.into_iter().collect();
                sorted.sort_by(|a, b| {
                    let score_a = a.metrics.score(&self.config);
                    let score_b = b.metrics.score(&self.config);
                    score_b
                        .partial_cmp(&score_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                sorted
                    .into_iter()
                    .take(count)
                    .map(|p| p.id.clone())
                    .collect()
            }
            SelectionStrategy::RoundRobin => {
                let mut result = Vec::with_capacity(count);
                let peer_ids: Vec<_> = available.iter().map(|p| p.id.clone()).collect();
                let len = peer_ids.len();

                for i in 0..count.min(len) {
                    let idx = (self.round_robin_idx + i) % len;
                    result.push(peer_ids[idx].clone());
                }

                self.round_robin_idx = (self.round_robin_idx + count) % len.max(1);
                result
            }
            SelectionStrategy::Random => {
                // Simple pseudo-random using time
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as usize)
                    .unwrap_or(0);

                let mut peer_ids: Vec<_> = available.iter().map(|p| p.id.clone()).collect();
                // Fisher-Yates shuffle (simplified)
                for i in (1..peer_ids.len()).rev() {
                    let j = (now.wrapping_add(i)) % (i + 1);
                    peer_ids.swap(i, j);
                }
                peer_ids.into_iter().take(count).collect()
            }
            SelectionStrategy::LeastLoaded => {
                let mut sorted: Vec<_> = available.into_iter().collect();
                sorted.sort_by_key(|p| p.active_requests);
                sorted
                    .into_iter()
                    .take(count)
                    .map(|p| p.id.clone())
                    .collect()
            }
        }
    }

    /// Select best peers that definitely have the CID
    pub fn select_providers(&mut self, cid: &Cid, count: usize) -> Vec<PeerId> {
        self.cleanup_blacklist();

        let mut providers: Vec<_> = self
            .peers
            .values()
            .filter(|p| p.can_accept_request() && p.has(cid))
            .collect();

        // Sort by score
        providers.sort_by(|a, b| {
            let score_a = a.metrics.score(&self.config);
            let score_b = b.metrics.score(&self.config);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        providers
            .into_iter()
            .take(count)
            .map(|p| p.id.clone())
            .collect()
    }

    /// Get all peer IDs
    pub fn peer_ids(&self) -> Vec<PeerId> {
        self.peers.keys().cloned().collect()
    }

    /// Get number of connected peers
    pub fn connected_count(&self) -> usize {
        self.peers.values().filter(|p| p.connected).count()
    }

    /// Get number of blacklisted peers
    pub fn blacklisted_count(&self) -> usize {
        self.blacklist.len()
    }

    /// Get peer scores
    pub fn get_scores(&self) -> HashMap<PeerId, f64> {
        self.peers
            .iter()
            .map(|(id, peer)| (id.clone(), peer.metrics.score(&self.config)))
            .collect()
    }

    /// Get peer with best score
    pub fn best_peer(&self) -> Option<&PeerId> {
        self.peers
            .iter()
            .filter(|(_, p)| p.connected)
            .max_by(|(_, a), (_, b)| {
                let score_a = a.metrics.score(&self.config);
                let score_b = b.metrics.score(&self.config);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(id, _)| id)
    }

    /// Set peer connected status
    pub fn set_connected(&mut self, peer_id: &PeerId, connected: bool) {
        if let Some(peer) = self.peers.get_mut(peer_id) {
            peer.connected = connected;
        }
    }

    /// Get statistics
    pub fn stats(&self) -> PeerManagerStats {
        let total_peers = self.peers.len();
        let connected_peers = self.peers.values().filter(|p| p.connected).count();
        let blacklisted_peers = self.blacklist.len();

        let avg_score = if total_peers > 0 {
            self.peers
                .values()
                .map(|p| p.metrics.score(&self.config))
                .sum::<f64>()
                / total_peers as f64
        } else {
            0.0
        };

        let avg_latency = if total_peers > 0 {
            self.peers
                .values()
                .map(|p| p.metrics.latency_ms)
                .sum::<f64>()
                / total_peers as f64
        } else {
            0.0
        };

        let total_requests: u64 = self.peers.values().map(|p| p.metrics.requests_sent).sum();
        let total_completed: u64 = self
            .peers
            .values()
            .map(|p| p.metrics.requests_completed)
            .sum();
        let total_failed: u64 = self.peers.values().map(|p| p.metrics.requests_failed).sum();

        PeerManagerStats {
            total_peers,
            connected_peers,
            blacklisted_peers,
            avg_score,
            avg_latency_ms: avg_latency,
            total_requests,
            total_completed,
            total_failed,
        }
    }
}

/// Peer manager statistics
#[derive(Debug, Clone)]
pub struct PeerManagerStats {
    /// Total known peers
    pub total_peers: usize,
    /// Currently connected peers
    pub connected_peers: usize,
    /// Blacklisted peers
    pub blacklisted_peers: usize,
    /// Average peer score
    pub avg_score: f64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// Total requests sent
    pub total_requests: u64,
    /// Total completed requests
    pub total_completed: u64,
    /// Total failed requests
    pub total_failed: u64,
}

/// Thread-safe peer manager
pub struct ConcurrentPeerManager {
    inner: Arc<RwLock<PeerManager>>,
}

impl ConcurrentPeerManager {
    /// Create a new concurrent peer manager
    pub fn new(config: PeerScoringConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(PeerManager::new(config))),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(PeerScoringConfig::default())
    }

    /// Add a peer
    pub fn add_peer(&self, id: PeerId) {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .add_peer(id);
    }

    /// Remove a peer
    pub fn remove_peer(&self, id: &PeerId) {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove_peer(id);
    }

    /// Record successful request
    pub fn record_success(&self, peer_id: &PeerId, bytes: u64, latency: Duration) {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .record_success(peer_id, bytes, latency);
    }

    /// Record failed request
    pub fn record_failure(&self, peer_id: &PeerId) {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .record_failure(peer_id);
    }

    /// Record HAVE message
    pub fn record_has(&self, peer_id: &PeerId, cid: Cid) {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .record_has(peer_id, cid);
    }

    /// Record DONT_HAVE message
    pub fn record_doesnt_have(&self, peer_id: &PeerId, cid: Cid) {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .record_doesnt_have(peer_id, cid);
    }

    /// Mark request sent
    pub fn mark_request_sent(&self, peer_id: &PeerId) {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .mark_request_sent(peer_id);
    }

    /// Blacklist a peer
    pub fn blacklist_peer(
        &self,
        peer_id: PeerId,
        reason: BlacklistReason,
        duration: Option<Duration>,
    ) {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .blacklist_peer(peer_id, reason, duration);
    }

    /// Check if blacklisted
    pub fn is_blacklisted(&self, peer_id: &PeerId) -> bool {
        self.inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .is_blacklisted(peer_id)
    }

    /// Select peers
    pub fn select_peers(
        &self,
        cid: &Cid,
        count: usize,
        strategy: SelectionStrategy,
    ) -> Vec<PeerId> {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .select_peers(cid, count, strategy)
    }

    /// Select providers
    pub fn select_providers(&self, cid: &Cid, count: usize) -> Vec<PeerId> {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .select_providers(cid, count)
    }

    /// Get statistics
    pub fn stats(&self) -> PeerManagerStats {
        self.inner.read().unwrap_or_else(|e| e.into_inner()).stats()
    }

    /// Get scores
    pub fn get_scores(&self) -> HashMap<PeerId, f64> {
        self.inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get_scores()
    }

    /// Set connected status
    pub fn set_connected(&self, peer_id: &PeerId, connected: bool) {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .set_connected(peer_id, connected);
    }

    /// Clone the inner Arc
    pub fn clone_inner(&self) -> Arc<RwLock<PeerManager>> {
        Arc::clone(&self.inner)
    }
}

impl Clone for ConcurrentPeerManager {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Initial backoff duration
    pub initial_backoff: Duration,
    /// Maximum backoff duration
    pub max_backoff: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Jitter factor (0.0-1.0) to prevent thundering herd
    pub jitter_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter_factor: 0.2,
        }
    }
}

/// Retry policy with exponential backoff
pub struct RetryPolicy {
    config: RetryConfig,
    attempt: u32,
    last_attempt: Option<Instant>,
}

impl RetryPolicy {
    /// Create a new retry policy
    pub fn new(config: RetryConfig) -> Self {
        Self {
            config,
            attempt: 0,
            last_attempt: None,
        }
    }

    /// Check if more retries are available
    pub fn can_retry(&self) -> bool {
        self.attempt < self.config.max_retries
    }

    /// Get the next backoff duration
    pub fn next_backoff(&mut self) -> Duration {
        self.attempt += 1;
        self.last_attempt = Some(Instant::now());

        let base_backoff = self.config.initial_backoff.as_millis() as f64
            * self.config.backoff_multiplier.powi(self.attempt as i32 - 1);

        let capped_backoff = base_backoff.min(self.config.max_backoff.as_millis() as f64);

        // Add jitter
        let jitter = if self.config.jitter_factor > 0.0 {
            use std::collections::hash_map::RandomState;
            use std::hash::BuildHasher;

            let hash = RandomState::new().hash_one(self.attempt);
            let jitter_range = capped_backoff * self.config.jitter_factor;
            ((hash % 1000) as f64 / 1000.0) * jitter_range
        } else {
            0.0
        };

        Duration::from_millis((capped_backoff + jitter) as u64)
    }

    /// Reset the retry policy
    pub fn reset(&mut self) {
        self.attempt = 0;
        self.last_attempt = None;
    }

    /// Get current attempt number
    pub fn attempt(&self) -> u32 {
        self.attempt
    }
}

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed, requests flow normally
    Closed,
    /// Circuit is open, requests are rejected
    Open,
    /// Circuit is half-open, testing if service recovered
    HalfOpen,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening circuit
    pub failure_threshold: u32,
    /// Duration to wait before entering half-open state
    pub timeout: Duration,
    /// Number of successful requests in half-open to close circuit
    pub success_threshold: u32,
    /// Window duration for counting failures
    pub window_duration: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            timeout: Duration::from_secs(30),
            success_threshold: 2,
            window_duration: Duration::from_secs(60),
        }
    }
}

/// Circuit breaker for peer connections
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: Arc<RwLock<CircuitState>>,
    failure_count: Arc<RwLock<u32>>,
    success_count: Arc<RwLock<u32>>,
    last_failure_time: Arc<RwLock<Option<Instant>>>,
    opened_at: Arc<RwLock<Option<Instant>>>,
    failure_timestamps: Arc<RwLock<VecDeque<Instant>>>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_count: Arc::new(RwLock::new(0)),
            success_count: Arc::new(RwLock::new(0)),
            last_failure_time: Arc::new(RwLock::new(None)),
            opened_at: Arc::new(RwLock::new(None)),
            failure_timestamps: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    /// Get current circuit state
    pub fn state(&self) -> CircuitState {
        *self.state.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Check if request is allowed
    pub fn is_request_allowed(&self) -> bool {
        let mut state = self.state.write().unwrap_or_else(|e| e.into_inner());

        match *state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if timeout elapsed to enter half-open
                let opened_at = self.opened_at.read().unwrap_or_else(|e| e.into_inner());
                if let Some(opened_time) = *opened_at {
                    if opened_time.elapsed() >= self.config.timeout {
                        *state = CircuitState::HalfOpen;
                        *self
                            .success_count
                            .write()
                            .unwrap_or_else(|e| e.into_inner()) = 0;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Record a successful request
    pub fn record_success(&self) {
        let mut state = self.state.write().unwrap_or_else(|e| e.into_inner());

        match *state {
            CircuitState::Closed => {
                // Reset failure count on success
                *self
                    .failure_count
                    .write()
                    .unwrap_or_else(|e| e.into_inner()) = 0;
                self.failure_timestamps
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .clear();
            }
            CircuitState::HalfOpen => {
                let mut success_count = self
                    .success_count
                    .write()
                    .unwrap_or_else(|e| e.into_inner());
                *success_count += 1;

                if *success_count >= self.config.success_threshold {
                    *state = CircuitState::Closed;
                    *self
                        .failure_count
                        .write()
                        .unwrap_or_else(|e| e.into_inner()) = 0;
                    *success_count = 0;
                    self.failure_timestamps
                        .write()
                        .unwrap_or_else(|e| e.into_inner())
                        .clear();
                }
            }
            CircuitState::Open => {}
        }
    }

    /// Record a failed request
    pub fn record_failure(&self) {
        let now = Instant::now();
        let mut state = self.state.write().unwrap_or_else(|e| e.into_inner());

        // Update failure timestamps
        {
            let mut timestamps = self
                .failure_timestamps
                .write()
                .unwrap_or_else(|e| e.into_inner());
            timestamps.push_back(now);

            // Remove old timestamps outside the window
            while let Some(&oldest) = timestamps.front() {
                if oldest.elapsed() > self.config.window_duration {
                    timestamps.pop_front();
                } else {
                    break;
                }
            }
        }

        *self
            .last_failure_time
            .write()
            .unwrap_or_else(|e| e.into_inner()) = Some(now);

        match *state {
            CircuitState::Closed => {
                let mut failure_count = self
                    .failure_count
                    .write()
                    .unwrap_or_else(|e| e.into_inner());
                *failure_count += 1;

                // Count failures in current window
                let window_failures = self
                    .failure_timestamps
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .len() as u32;

                if window_failures >= self.config.failure_threshold {
                    *state = CircuitState::Open;
                    *self.opened_at.write().unwrap_or_else(|e| e.into_inner()) = Some(now);
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open immediately reopens circuit
                *state = CircuitState::Open;
                *self.opened_at.write().unwrap_or_else(|e| e.into_inner()) = Some(now);
                *self
                    .success_count
                    .write()
                    .unwrap_or_else(|e| e.into_inner()) = 0;
            }
            CircuitState::Open => {}
        }
    }

    /// Reset the circuit breaker
    pub fn reset(&self) {
        *self.state.write().unwrap_or_else(|e| e.into_inner()) = CircuitState::Closed;
        *self
            .failure_count
            .write()
            .unwrap_or_else(|e| e.into_inner()) = 0;
        *self
            .success_count
            .write()
            .unwrap_or_else(|e| e.into_inner()) = 0;
        *self
            .last_failure_time
            .write()
            .unwrap_or_else(|e| e.into_inner()) = None;
        *self.opened_at.write().unwrap_or_else(|e| e.into_inner()) = None;
        self.failure_timestamps
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// Get statistics
    pub fn stats(&self) -> CircuitBreakerStats {
        CircuitBreakerStats {
            state: self.state(),
            failure_count: *self.failure_count.read().unwrap_or_else(|e| e.into_inner()),
            success_count: *self.success_count.read().unwrap_or_else(|e| e.into_inner()),
            window_failures: self
                .failure_timestamps
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .len() as u32,
        }
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }
}

/// Circuit breaker statistics
#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    /// Current state
    pub state: CircuitState,
    /// Total failure count
    pub failure_count: u32,
    /// Success count in half-open state
    pub success_count: u32,
    /// Failures in current window
    pub window_failures: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cid() -> Cid {
        "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse()
            .expect("test: parse CID from known-good string")
    }

    #[test]
    fn test_peer_metrics_score() {
        let config = PeerScoringConfig::default();
        let mut metrics = PeerMetrics::default();

        // Default should have reasonable score
        let initial_score = metrics.score(&config);
        assert!(initial_score > 0.0);
        assert!(initial_score <= 1.0);

        // Good performance should improve score
        metrics.latency_ms = 5.0;
        metrics.bandwidth_bps = 50_000_000.0;
        metrics.reliability = 1.0;
        let good_score = metrics.score(&config);
        assert!(good_score > initial_score);

        // Poor performance should lower score
        metrics.latency_ms = 500.0;
        metrics.bandwidth_bps = 100_000.0;
        metrics.reliability = 0.5;
        let poor_score = metrics.score(&config);
        assert!(poor_score < good_score);
    }

    #[test]
    fn test_peer_manager_add_remove() {
        let mut manager = PeerManager::with_defaults();

        manager.add_peer("peer1".to_string());
        assert!(manager.get_peer(&"peer1".to_string()).is_some());

        manager.remove_peer(&"peer1".to_string());
        assert!(manager.get_peer(&"peer1".to_string()).is_none());
    }

    #[test]
    fn test_peer_selection_fastest_first() {
        let mut manager = PeerManager::with_defaults();
        let cid = test_cid();

        manager.add_peer("slow".to_string());
        manager.add_peer("fast".to_string());

        // Set different latencies
        if let Some(peer) = manager.get_peer_mut(&"slow".to_string()) {
            peer.metrics.latency_ms = 100.0;
        }
        if let Some(peer) = manager.get_peer_mut(&"fast".to_string()) {
            peer.metrics.latency_ms = 10.0;
        }

        let selected = manager.select_peers(&cid, 1, SelectionStrategy::FastestFirst);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0], "fast");
    }

    #[test]
    fn test_blacklisting() {
        let mut manager = PeerManager::with_defaults();

        manager.add_peer("bad_peer".to_string());
        assert!(!manager.is_blacklisted(&"bad_peer".to_string()));

        manager.blacklist_peer(
            "bad_peer".to_string(),
            BlacklistReason::Manual,
            None, // Permanent
        );

        assert!(manager.is_blacklisted(&"bad_peer".to_string()));
        assert!(manager.get_peer(&"bad_peer".to_string()).is_none());
    }

    #[test]
    fn test_temporary_blacklist_expiry() {
        let mut manager = PeerManager::with_defaults();

        manager.add_peer("temp_bad".to_string());
        manager.blacklist_peer(
            "temp_bad".to_string(),
            BlacklistReason::RepeatedFailures,
            Some(Duration::from_millis(10)), // Very short ban
        );

        assert!(manager.is_blacklisted(&"temp_bad".to_string()));

        // Wait for expiry
        std::thread::sleep(Duration::from_millis(20));

        // Should no longer be blacklisted
        assert!(!manager.is_blacklisted(&"temp_bad".to_string()));
    }

    #[test]
    fn test_has_doesnt_have_tracking() {
        let mut manager = PeerManager::with_defaults();
        let cid = test_cid();

        manager.add_peer("peer1".to_string());
        manager.record_has(&"peer1".to_string(), cid);

        let peer = manager
            .get_peer(&"peer1".to_string())
            .expect("test: peer1 was just added to manager");
        assert!(peer.has(&cid));

        // Provider selection should include this peer
        let providers = manager.select_providers(&cid, 1);
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0], "peer1");
    }

    #[test]
    fn test_concurrent_peer_manager() {
        let manager = ConcurrentPeerManager::with_defaults();

        manager.add_peer("peer1".to_string());
        manager.record_success(&"peer1".to_string(), 1000, Duration::from_millis(10));

        let stats = manager.stats();
        assert_eq!(stats.total_peers, 1);
        assert_eq!(stats.total_completed, 1);
    }

    #[test]
    fn test_retry_policy() {
        let config = RetryConfig::default();
        let mut policy = RetryPolicy::new(config);

        assert!(policy.can_retry());
        assert_eq!(policy.attempt(), 0);

        let backoff1 = policy.next_backoff();
        assert!(backoff1.as_millis() >= 100);

        let backoff2 = policy.next_backoff();
        assert!(backoff2 > backoff1);

        policy.reset();
        assert_eq!(policy.attempt(), 0);
    }

    #[test]
    fn test_circuit_breaker() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            timeout: Duration::from_millis(100),
            success_threshold: 2,
            window_duration: Duration::from_secs(60),
        };

        let breaker = CircuitBreaker::new(config);

        // Initially closed
        assert_eq!(breaker.state(), CircuitState::Closed);
        assert!(breaker.is_request_allowed());

        // Record failures
        breaker.record_failure();
        breaker.record_failure();
        breaker.record_failure();

        // Should be open now
        assert_eq!(breaker.state(), CircuitState::Open);
        assert!(!breaker.is_request_allowed());

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(150));

        // Should transition to half-open
        assert!(breaker.is_request_allowed());
        assert_eq!(breaker.state(), CircuitState::HalfOpen);

        // Record successes to close
        breaker.record_success();
        breaker.record_success();

        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_stats() {
        let breaker = CircuitBreaker::default();

        breaker.record_failure();
        breaker.record_failure();

        let stats = breaker.stats();
        assert_eq!(stats.failure_count, 2);
        assert_eq!(stats.window_failures, 2);
        assert_eq!(stats.state, CircuitState::Closed);
    }
}
