//! Gossip Dissemination Protocol Engine
//!
//! A production-quality gossip protocol engine implementing epidemic broadcast with
//! pluggable fanout strategies, message deduplication, hop-count TTL, and per-peer
//! delivery scoring.  All state is synchronous and `Send + Sync`-compatible (the
//! public API takes `&mut self` for mutable operations).
//!
//! # Design notes
//! - **No external PRNG**: `xorshift64` is used where randomness is required.
//! - **No `rand` crate**: coin-flip / shuffle are done inline.
//! - **FNV-1a 64-bit** is used for deterministic message-ID generation.
//! - **Dedup cache**: bounded `VecDeque` providing O(1) push / O(n) lookup, sufficient
//!   for typical gossip workloads where the window is small (≤ 4096 messages).

use std::collections::VecDeque;

// ────────────────────────────────────────────────────────────────────────────
// Primitive helpers
// ────────────────────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash.
#[inline]
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

/// Xorshift64 PRNG – mutates `state` in-place and returns the next value.
#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

/// A gossip message travelling through the overlay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GossipMessage {
    /// FNV-1a hex of `payload ++ timestamp`.
    pub id: String,
    /// Peer-ID string of the node that originally created the message.
    pub origin_peer: String,
    /// Number of hops already traversed (incremented on each forward).
    pub hop_count: u8,
    /// Maximum number of hops before the message is dropped.
    pub max_hops: u8,
    /// Raw payload bytes.
    pub payload: Vec<u8>,
    /// Unix-epoch milliseconds at creation time.
    pub timestamp: u64,
    /// Application-level topic string.
    pub topic: String,
}

/// An active or recently-seen gossip peer.
#[derive(Debug, Clone)]
pub struct GossipPeer {
    /// Peer-ID string.
    pub id: String,
    /// Unix-epoch milliseconds of last activity.
    pub last_seen: u64,
    /// Total number of messages received from this peer.
    pub message_count: u64,
    /// Whether this peer is considered active.
    pub is_active: bool,
    /// Exponential-moving-average delivery efficiency in `[0.0, 1.0]`.
    pub fanout_score: f64,
}

impl GossipPeer {
    fn new(id: String) -> Self {
        Self {
            id,
            last_seen: 0,
            message_count: 0,
            is_active: true,
            fanout_score: 0.5,
        }
    }
}

/// Peer-selection strategy used when forwarding messages.
#[derive(Debug, Clone)]
pub enum FanoutStrategy {
    /// Always forward to the first `n` active peers (excluding origin).
    Fixed(usize),
    /// Adaptive count: `min + (active_count / 10).min(max − min)`.
    Adaptive {
        /// Minimum number of peers to forward to.
        min: usize,
        /// Maximum number of peers to forward to.
        max: usize,
    },
    /// Each active peer is included with probability `p` (0.0–1.0).
    Epidemic(f64),
    /// Deterministic shuffle via `xorshift64(seed)`, then take `fanout_size`.
    Random(u64),
    /// Sort peers descending by `fanout_score`, take `fanout_size`.
    PriorityBased,
}

/// Configuration for a `GossipProtocolEngine`.
#[derive(Debug, Clone)]
pub struct GossipConfig {
    /// Default number of peers to forward to (used by several strategies).
    pub fanout_size: usize,
    /// Maximum hop-count before a message is silently dropped.
    pub max_hops: u8,
    /// Maximum number of message IDs retained in the dedup cache.
    pub dedup_cache_size: usize,
    /// How often the engine should gossip (milliseconds); informational only.
    pub gossip_interval_ms: u64,
    /// Minimum number of active peers required before forwarding.
    pub min_peers_for_gossip: usize,
    /// Fanout strategy to apply on each forward.
    pub strategy: FanoutStrategy,
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            fanout_size: 6,
            max_hops: 7,
            dedup_cache_size: 2048,
            gossip_interval_ms: 100,
            min_peers_for_gossip: 1,
            strategy: FanoutStrategy::Fixed(6),
        }
    }
}

/// Aggregate statistics produced by a `GossipProtocolEngine`.
#[derive(Debug, Clone, Default)]
pub struct GossipStats {
    /// Total messages successfully received (not duplicates, not over TTL).
    pub messages_received: u64,
    /// Total messages forwarded to downstream peers.
    pub messages_forwarded: u64,
    /// Messages dropped because they were already seen.
    pub messages_dropped_duplicate: u64,
    /// Messages dropped because `hop_count >= max_hops`.
    pub messages_dropped_ttl: u64,
    /// Current number of active peers.
    pub active_peers: usize,
    /// Exponential-moving-average hop-count of accepted messages.
    pub avg_propagation_hops: f64,
}

/// Events emitted by a `GossipProtocolEngine`.
#[derive(Debug, Clone)]
pub enum GossipEvent {
    /// A new (non-duplicate, within-TTL) message was accepted.
    MessageReceived(GossipMessage),
    /// A message was forwarded to a set of peers.
    MessageForwarded {
        /// ID of the forwarded message.
        msg_id: String,
        /// Peer-IDs that the message was sent to.
        to_peers: Vec<String>,
    },
    /// A new peer was registered.
    PeerAdded(String),
    /// A peer was removed.
    PeerRemoved(String),
    /// A duplicate message was dropped.
    DuplicateDropped(String),
}

/// Errors returned by `GossipProtocolEngine` operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    /// No peer with the given ID is registered.
    PeerNotFound(String),
    /// Payload exceeds the allowed maximum (value is the actual size in bytes).
    MessageTooLarge(usize),
    /// There are no active peers to forward to.
    NoPeersAvailable,
    /// `hop_count` field carries an invalid value (e.g. exceeds `u8::MAX`).
    InvalidHopCount,
    /// Binary decoding of a field failed.
    DecodingError(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::PeerNotFound(id) => write!(f, "peer not found: {id}"),
            EngineError::MessageTooLarge(n) => write!(f, "message too large: {n} bytes"),
            EngineError::NoPeersAvailable => write!(f, "no active peers available"),
            EngineError::InvalidHopCount => write!(f, "invalid hop count"),
            EngineError::DecodingError(msg) => write!(f, "decoding error: {msg}"),
        }
    }
}

impl std::error::Error for EngineError {}

// ────────────────────────────────────────────────────────────────────────────
// Engine
// ────────────────────────────────────────────────────────────────────────────

/// Production-quality gossip dissemination protocol engine.
///
/// The engine is **not** async; it exposes a pure, synchronous API suitable for
/// embedding inside higher-level async event loops.  All state is owned by the
/// struct and protected via `&mut self`.
pub struct GossipProtocolEngine {
    config: GossipConfig,
    peers: Vec<GossipPeer>,
    dedup_cache: VecDeque<String>,
    stats: GossipStats,
    event_log: Vec<GossipEvent>,
    /// PRNG state for strategies that need randomness.
    rng_state: u64,
    /// EMA accumulator for avg_propagation_hops.
    hop_ema_n: u64,
}

impl GossipProtocolEngine {
    // ── Construction ──────────────────────────────────────────────────────

    /// Create a new engine with the given configuration.
    ///
    /// The PRNG seed is derived from `dedup_cache_size` XOR a constant so that
    /// default-config engines are reproducible in tests.
    pub fn new(config: GossipConfig) -> Self {
        let seed = (config.dedup_cache_size as u64)
            ^ 0xdeadbeef_cafebabe
            ^ (config.fanout_size as u64).wrapping_mul(6_364_136_223_846_793_005);
        Self {
            peers: Vec::new(),
            dedup_cache: VecDeque::with_capacity(config.dedup_cache_size),
            stats: GossipStats::default(),
            event_log: Vec::new(),
            rng_state: if seed == 0 { 1 } else { seed },
            hop_ema_n: 0,
            config,
        }
    }

    /// Create an engine with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(GossipConfig::default())
    }

    // ── Peer management ───────────────────────────────────────────────────

    /// Register a new peer.  Returns `Ok(())` if the peer was added, or if it
    /// was already present (idempotent).
    pub fn add_peer(&mut self, peer_id: String) -> Result<(), EngineError> {
        if !self.peers.iter().any(|p| p.id == peer_id) {
            self.peers.push(GossipPeer::new(peer_id.clone()));
            self.event_log.push(GossipEvent::PeerAdded(peer_id));
            self.stats.active_peers = self.active_peer_count();
        }
        Ok(())
    }

    /// Remove a registered peer.  Returns `Err(PeerNotFound)` if absent.
    pub fn remove_peer(&mut self, peer_id: &str) -> Result<(), EngineError> {
        let pos = self
            .peers
            .iter()
            .position(|p| p.id == peer_id)
            .ok_or_else(|| EngineError::PeerNotFound(peer_id.to_owned()))?;
        self.peers.remove(pos);
        self.event_log
            .push(GossipEvent::PeerRemoved(peer_id.to_owned()));
        self.stats.active_peers = self.active_peer_count();
        Ok(())
    }

    // ── Message creation ──────────────────────────────────────────────────

    /// Create a new `GossipMessage` with a computed FNV-1a ID.
    ///
    /// `timestamp` should be a monotonically-increasing value (e.g.
    /// `SystemTime::now()` milliseconds); when running tests without a real
    /// clock, pass any non-zero value.
    pub fn create_message(
        payload: Vec<u8>,
        topic: String,
        origin: String,
        timestamp: u64,
        max_hops: u8,
    ) -> Result<GossipMessage, EngineError> {
        let mut id_input = payload.clone();
        id_input.extend_from_slice(&timestamp.to_le_bytes());
        let hash = fnv1a_64(&id_input);
        let id = format!("{hash:016x}");
        Ok(GossipMessage {
            id,
            origin_peer: origin,
            hop_count: 0,
            max_hops,
            payload,
            timestamp,
            topic,
        })
    }

    // ── Receive / forward ─────────────────────────────────────────────────

    /// Process an incoming `GossipMessage`.
    ///
    /// Returns a `Vec<GossipEvent>` describing what happened:
    /// - `DuplicateDropped` if the message was already seen.
    /// - `MessageReceived` + `MessageForwarded` if accepted and forwarded.
    /// - `MessageReceived` only if there are no eligible forward peers.
    pub fn receive_message(&mut self, msg: GossipMessage) -> Result<Vec<GossipEvent>, EngineError> {
        // 1. Dedup check.
        if self.dedup_cache.contains(&msg.id) {
            self.stats.messages_dropped_duplicate += 1;
            let ev = GossipEvent::DuplicateDropped(msg.id.clone());
            self.event_log.push(ev.clone());
            return Ok(vec![ev]);
        }

        // 2. TTL check.
        if msg.hop_count >= msg.max_hops {
            self.stats.messages_dropped_ttl += 1;
            return Ok(vec![]);
        }

        // 3. Accept: add to dedup cache (bounded).
        if self.dedup_cache.len() >= self.config.dedup_cache_size {
            self.dedup_cache.pop_front();
        }
        self.dedup_cache.push_back(msg.id.clone());

        // 4. Update stats.
        self.stats.messages_received += 1;
        self.hop_ema_n += 1;
        let hops_f = msg.hop_count as f64;
        if self.hop_ema_n == 1 {
            self.stats.avg_propagation_hops = hops_f;
        } else {
            self.stats.avg_propagation_hops = self.stats.avg_propagation_hops * 0.9 + hops_f * 0.1;
        }

        let received_ev = GossipEvent::MessageReceived(msg.clone());
        self.event_log.push(received_ev.clone());
        let mut events = vec![received_ev];

        // 5. Select forwarding peers (clone strategy to avoid borrow conflict).
        let strategy = self.config.strategy.clone();
        let targets = self.select_peers(&msg.origin_peer, &strategy);

        if !targets.is_empty() {
            // 6. Forward: bump hop_count for the forwarded copy.
            let mut forwarded = msg.clone();
            forwarded.hop_count = msg.hop_count.saturating_add(1);
            self.stats.messages_forwarded += targets.len() as u64;

            let fwd_ev = GossipEvent::MessageForwarded {
                msg_id: msg.id.clone(),
                to_peers: targets.clone(),
            };
            self.event_log.push(fwd_ev.clone());
            events.push(fwd_ev);

            // Update scores: mark each target as "delivered".
            for peer_id in &targets {
                // Safe: we have a valid peer list; update_peer_score is infallible.
                self.update_peer_score(peer_id, true);
            }
        }

        self.stats.active_peers = self.active_peer_count();
        Ok(events)
    }

    // ── Peer selection ────────────────────────────────────────────────────

    /// Select peers to forward a message to, applying the given strategy.
    ///
    /// The returned list excludes `exclude` (the origin peer) and only
    /// contains active peers.
    pub fn select_peers(&mut self, exclude: &str, strategy: &FanoutStrategy) -> Vec<String> {
        // Gather active candidate IDs (excluding `exclude`).
        let candidates: Vec<(usize, f64)> = self
            .peers
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_active && p.id != exclude)
            .map(|(i, p)| (i, p.fanout_score))
            .collect();

        if candidates.is_empty() {
            return Vec::new();
        }

        let active_count = candidates.len();

        match strategy {
            FanoutStrategy::Fixed(n) => {
                // Take the first `n` active peers in insertion order.
                candidates
                    .iter()
                    .take(*n)
                    .map(|(i, _)| self.peers[*i].id.clone())
                    .collect()
            }

            FanoutStrategy::Adaptive { min, max } => {
                let extra = (active_count / 10).min(max.saturating_sub(*min));
                let target = (min + extra).min(*max).min(active_count);
                candidates
                    .iter()
                    .take(target)
                    .map(|(i, _)| self.peers[*i].id.clone())
                    .collect()
            }

            FanoutStrategy::Epidemic(p) => {
                let threshold = (*p * u64::MAX as f64) as u64;
                candidates
                    .iter()
                    .filter(|_| {
                        let r = xorshift64(&mut self.rng_state);
                        r <= threshold
                    })
                    .map(|(i, _)| self.peers[*i].id.clone())
                    .collect()
            }

            FanoutStrategy::Random(seed) => {
                // Build a local index list and shuffle with xorshift64.
                let mut indices: Vec<usize> = (0..active_count).collect();
                let mut state = if *seed == 0 { self.rng_state } else { *seed };
                // Fisher-Yates shuffle.
                for i in (1..active_count).rev() {
                    let j = (xorshift64(&mut state) % (i as u64 + 1)) as usize;
                    indices.swap(i, j);
                }
                // Advance rng_state to avoid repeated deterministic sequences.
                self.rng_state = state;
                let take = self.config.fanout_size.min(active_count);
                indices[..take]
                    .iter()
                    .map(|&ci| self.peers[candidates[ci].0].id.clone())
                    .collect()
            }

            FanoutStrategy::PriorityBased => {
                // Sort descending by fanout_score, take `fanout_size`.
                let mut sorted = candidates.clone();
                sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let take = self.config.fanout_size.min(active_count);
                sorted[..take]
                    .iter()
                    .map(|(i, _)| self.peers[*i].id.clone())
                    .collect()
            }
        }
    }

    // ── Scoring ───────────────────────────────────────────────────────────

    /// Update a peer's `fanout_score` using an EMA with α = 0.1.
    ///
    /// `delivered = true`  → positive signal (score nudges toward 1.0).
    /// `delivered = false` → negative signal (score nudges toward 0.0).
    pub fn update_peer_score(&mut self, peer_id: &str, delivered: bool) {
        if let Some(peer) = self.peers.iter_mut().find(|p| p.id == peer_id) {
            let signal = if delivered { 1.0_f64 } else { 0.0_f64 };
            peer.fanout_score = 0.9 * peer.fanout_score + 0.1 * signal;
            // Clamp to [0.0, 1.0] for numerical stability.
            peer.fanout_score = peer.fanout_score.clamp(0.0, 1.0);
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    /// Number of message IDs currently in the dedup cache.
    pub fn pending_messages(&self) -> usize {
        self.dedup_cache.len()
    }

    /// Return a snapshot of current statistics.
    pub fn stats(&self) -> GossipStats {
        let mut s = self.stats.clone();
        s.active_peers = self.active_peer_count();
        s
    }

    /// Drain and return all buffered `GossipEvent`s.
    pub fn drain_events(&mut self) -> Vec<GossipEvent> {
        std::mem::take(&mut self.event_log)
    }

    /// Borrow the current configuration.
    pub fn config(&self) -> &GossipConfig {
        &self.config
    }

    /// Number of registered peers (active + inactive).
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Get a peer by ID.
    pub fn peer(&self, peer_id: &str) -> Option<&GossipPeer> {
        self.peers.iter().find(|p| p.id == peer_id)
    }

    /// Mark a peer as inactive without removing it.
    pub fn deactivate_peer(&mut self, peer_id: &str) -> Result<(), EngineError> {
        let peer = self
            .peers
            .iter_mut()
            .find(|p| p.id == peer_id)
            .ok_or_else(|| EngineError::PeerNotFound(peer_id.to_owned()))?;
        peer.is_active = false;
        self.stats.active_peers = self.active_peer_count();
        Ok(())
    }

    /// Reactivate a previously deactivated peer.
    pub fn reactivate_peer(&mut self, peer_id: &str) -> Result<(), EngineError> {
        let peer = self
            .peers
            .iter_mut()
            .find(|p| p.id == peer_id)
            .ok_or_else(|| EngineError::PeerNotFound(peer_id.to_owned()))?;
        peer.is_active = true;
        self.stats.active_peers = self.active_peer_count();
        Ok(())
    }

    /// Reset the dedup cache without touching any other state.
    pub fn clear_dedup_cache(&mut self) {
        self.dedup_cache.clear();
    }

    // ── Private helpers ───────────────────────────────────────────────────

    fn active_peer_count(&self) -> usize {
        self.peers.iter().filter(|p| p.is_active).count()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────

    fn make_engine() -> GossipProtocolEngine {
        GossipProtocolEngine::with_defaults()
    }

    fn add_peers(engine: &mut GossipProtocolEngine, n: usize) {
        for i in 0..n {
            engine.add_peer(format!("peer{i}")).expect("add_peer");
        }
    }

    fn make_msg(id: &str, origin: &str, hop: u8, max: u8) -> GossipMessage {
        GossipMessage {
            id: id.to_owned(),
            origin_peer: origin.to_owned(),
            hop_count: hop,
            max_hops: max,
            payload: vec![1, 2, 3],
            timestamp: 1_000_000,
            topic: "test".to_owned(),
        }
    }

    // ── 1. Peer management ────────────────────────────────────────────────

    #[test]
    fn add_peer_increases_count() {
        let mut e = make_engine();
        e.add_peer("p1".into())
            .expect("test: add_peer should succeed");
        assert_eq!(e.peer_count(), 1);
    }

    #[test]
    fn add_peer_idempotent() {
        let mut e = make_engine();
        e.add_peer("p1".into())
            .expect("test: add_peer should succeed");
        e.add_peer("p1".into())
            .expect("test: add_peer idempotent should succeed");
        assert_eq!(e.peer_count(), 1);
    }

    #[test]
    fn remove_peer_decreases_count() {
        let mut e = make_engine();
        e.add_peer("p1".into())
            .expect("test: add_peer should succeed");
        e.remove_peer("p1")
            .expect("test: remove_peer should succeed");
        assert_eq!(e.peer_count(), 0);
    }

    #[test]
    fn remove_nonexistent_peer_errors() {
        let mut e = make_engine();
        let res = e.remove_peer("ghost");
        assert!(matches!(res, Err(EngineError::PeerNotFound(_))));
    }

    #[test]
    fn peer_added_event_emitted() {
        let mut e = make_engine();
        e.add_peer("p1".into())
            .expect("test: add_peer should succeed");
        let events = e.drain_events();
        assert!(events
            .iter()
            .any(|ev| matches!(ev, GossipEvent::PeerAdded(id) if id == "p1")));
    }

    #[test]
    fn peer_removed_event_emitted() {
        let mut e = make_engine();
        e.add_peer("p1".into())
            .expect("test: add_peer should succeed");
        e.drain_events();
        e.remove_peer("p1")
            .expect("test: remove_peer should succeed");
        let events = e.drain_events();
        assert!(events
            .iter()
            .any(|ev| matches!(ev, GossipEvent::PeerRemoved(id) if id == "p1")));
    }

    #[test]
    fn deactivate_peer_excludes_from_forwarding() {
        let mut e = make_engine();
        e.add_peer("p1".into())
            .expect("test: add_peer should succeed");
        e.deactivate_peer("p1")
            .expect("test: deactivate_peer should succeed");
        assert_eq!(e.active_peer_count(), 0);
    }

    #[test]
    fn reactivate_peer_includes_in_forwarding() {
        let mut e = make_engine();
        e.add_peer("p1".into())
            .expect("test: add_peer should succeed");
        e.deactivate_peer("p1")
            .expect("test: deactivate_peer should succeed");
        e.reactivate_peer("p1")
            .expect("test: reactivate_peer should succeed");
        assert_eq!(e.active_peer_count(), 1);
    }

    #[test]
    fn deactivate_nonexistent_errors() {
        let mut e = make_engine();
        assert!(matches!(
            e.deactivate_peer("ghost"),
            Err(EngineError::PeerNotFound(_))
        ));
    }

    // ── 2. Message creation ───────────────────────────────────────────────

    #[test]
    fn create_message_produces_valid_id() {
        let msg = GossipProtocolEngine::create_message(
            vec![1, 2, 3],
            "topic".into(),
            "origin".into(),
            12345,
            7,
        )
        .expect("test: create_message should succeed");
        assert_eq!(msg.id.len(), 16, "FNV-1a 64-bit hex is 16 chars");
        assert_eq!(msg.hop_count, 0);
        assert_eq!(msg.max_hops, 7);
    }

    #[test]
    fn create_message_different_timestamps_differ() {
        let m1 = GossipProtocolEngine::create_message(vec![1], "t".into(), "o".into(), 1, 5)
            .expect("test: create_message with timestamp 1 should succeed");
        let m2 = GossipProtocolEngine::create_message(vec![1], "t".into(), "o".into(), 2, 5)
            .expect("test: create_message with timestamp 2 should succeed");
        assert_ne!(m1.id, m2.id);
    }

    #[test]
    fn create_message_same_inputs_same_id() {
        let m1 =
            GossipProtocolEngine::create_message(vec![9, 8, 7], "t".into(), "o".into(), 999, 5)
                .expect("test: create_message m1 should succeed");
        let m2 =
            GossipProtocolEngine::create_message(vec![9, 8, 7], "t".into(), "o".into(), 999, 5)
                .expect("test: create_message m2 should succeed");
        assert_eq!(m1.id, m2.id);
    }

    // ── 3. Receive and forward ────────────────────────────────────────────

    #[test]
    fn receive_message_no_peers_still_accepted() {
        let mut e = make_engine();
        let msg = make_msg("msg1", "origin", 0, 7);
        let events = e
            .receive_message(msg)
            .expect("test: receive_message should succeed");
        assert!(events
            .iter()
            .any(|ev| matches!(ev, GossipEvent::MessageReceived(_))));
        assert_eq!(e.stats().messages_received, 1);
    }

    #[test]
    fn receive_message_forwards_to_peers() {
        let mut e = make_engine();
        add_peers(&mut e, 3);
        let msg = make_msg("msg1", "external", 0, 7);
        let events = e
            .receive_message(msg)
            .expect("test: receive_message should succeed");
        let fwd = events
            .iter()
            .find(|ev| matches!(ev, GossipEvent::MessageForwarded { .. }));
        assert!(fwd.is_some());
        if let Some(GossipEvent::MessageForwarded { to_peers, .. }) = fwd {
            assert!(!to_peers.is_empty());
        }
    }

    #[test]
    fn receive_excludes_origin_peer() {
        let mut e = make_engine();
        e.add_peer("p1".into())
            .expect("test: add_peer should succeed");
        let msg = make_msg("msg1", "p1", 0, 7);
        let events = e
            .receive_message(msg)
            .expect("test: receive_message should succeed");
        // p1 is the only peer and is excluded → no forward.
        assert!(!events
            .iter()
            .any(|ev| matches!(ev, GossipEvent::MessageForwarded { .. })));
    }

    #[test]
    fn receive_hop_count_incremented_on_forward() {
        let mut e = make_engine();
        add_peers(&mut e, 2);
        let msg = make_msg("msg1", "ext", 3, 7);
        let events = e
            .receive_message(msg)
            .expect("test: receive_message should succeed");
        assert!(events
            .iter()
            .any(|ev| matches!(ev, GossipEvent::MessageForwarded { .. })));
    }

    #[test]
    fn stats_messages_received_incremented() {
        let mut e = make_engine();
        let msg = make_msg("m1", "ext", 0, 7);
        e.receive_message(msg)
            .expect("test: receive_message should succeed");
        assert_eq!(e.stats().messages_received, 1);
    }

    #[test]
    fn stats_messages_forwarded_incremented() {
        let mut e = make_engine();
        add_peers(&mut e, 4);
        let msg = make_msg("m1", "ext", 0, 7);
        e.receive_message(msg)
            .expect("test: receive_message should succeed");
        assert!(e.stats().messages_forwarded > 0);
    }

    // ── 4. Deduplication ─────────────────────────────────────────────────

    #[test]
    fn duplicate_message_dropped() {
        let mut e = make_engine();
        let msg = make_msg("dup", "ext", 0, 7);
        e.receive_message(msg.clone())
            .expect("test: first receive_message should succeed");
        let events2 = e
            .receive_message(msg)
            .expect("test: second receive_message should succeed");
        assert!(events2
            .iter()
            .any(|ev| matches!(ev, GossipEvent::DuplicateDropped(_))));
    }

    #[test]
    fn duplicate_increments_dropped_counter() {
        let mut e = make_engine();
        let msg = make_msg("dup", "ext", 0, 7);
        e.receive_message(msg.clone())
            .expect("test: first receive_message should succeed");
        e.receive_message(msg)
            .expect("test: duplicate receive_message should succeed");
        assert_eq!(e.stats().messages_dropped_duplicate, 1);
    }

    #[test]
    fn duplicate_not_forwarded() {
        let mut e = make_engine();
        add_peers(&mut e, 3);
        let msg = make_msg("dup", "ext", 0, 7);
        e.receive_message(msg.clone())
            .expect("test: first receive_message should succeed");
        let before = e.stats().messages_forwarded;
        let events = e
            .receive_message(msg)
            .expect("test: duplicate receive_message should succeed");
        assert!(events
            .iter()
            .any(|ev| matches!(ev, GossipEvent::DuplicateDropped(_))));
        assert_eq!(e.stats().messages_forwarded, before);
    }

    #[test]
    fn dedup_cache_bounded_eviction() {
        let mut e = GossipProtocolEngine::new(GossipConfig {
            dedup_cache_size: 3,
            ..GossipConfig::default()
        });
        for i in 0_u32..4 {
            let msg = make_msg(&format!("m{i}"), "ext", 0, 7);
            e.receive_message(msg)
                .expect("test: receive_message should succeed");
        }
        // Cache should hold exactly 3 entries.
        assert_eq!(e.pending_messages(), 3);
    }

    #[test]
    fn evicted_message_can_be_reaccepted() {
        let mut e = GossipProtocolEngine::new(GossipConfig {
            dedup_cache_size: 2,
            ..GossipConfig::default()
        });
        e.receive_message(make_msg("m0", "ext", 0, 7))
            .expect("test: receive m0 should succeed");
        e.receive_message(make_msg("m1", "ext", 0, 7))
            .expect("test: receive m1 should succeed");
        // Evict m0 by adding m2.
        e.receive_message(make_msg("m2", "ext", 0, 7))
            .expect("test: receive m2 should succeed");
        // m0 should be re-accepted now.
        let events = e
            .receive_message(make_msg("m0", "ext", 0, 7))
            .expect("test: re-receive evicted m0 should succeed");
        assert!(events
            .iter()
            .any(|ev| matches!(ev, GossipEvent::MessageReceived(_))));
    }

    #[test]
    fn clear_dedup_cache_allows_reprocessing() {
        let mut e = make_engine();
        let msg = make_msg("m1", "ext", 0, 7);
        e.receive_message(msg.clone())
            .expect("test: first receive_message should succeed");
        e.clear_dedup_cache();
        let events = e
            .receive_message(msg)
            .expect("test: receive after cache clear should succeed");
        assert!(events
            .iter()
            .any(|ev| matches!(ev, GossipEvent::MessageReceived(_))));
    }

    // ── 5. TTL / hop-count enforcement ────────────────────────────────────

    #[test]
    fn message_at_max_hops_dropped() {
        let mut e = make_engine();
        add_peers(&mut e, 3);
        let msg = make_msg("ttl1", "ext", 7, 7); // hop_count == max_hops
        let events = e
            .receive_message(msg)
            .expect("test: receive_message should succeed even for TTL drop");
        assert!(events.is_empty());
        assert_eq!(e.stats().messages_dropped_ttl, 1);
    }

    #[test]
    fn message_below_max_hops_accepted() {
        let mut e = make_engine();
        let msg = make_msg("ttl2", "ext", 6, 7); // hop_count < max_hops
        let events = e
            .receive_message(msg)
            .expect("test: receive_message below max_hops should succeed");
        assert!(events
            .iter()
            .any(|ev| matches!(ev, GossipEvent::MessageReceived(_))));
    }

    #[test]
    fn ttl_drop_not_forwarded() {
        let mut e = make_engine();
        add_peers(&mut e, 5);
        let msg = make_msg("ttl3", "ext", 7, 7);
        e.receive_message(msg)
            .expect("test: receive_message should succeed even for TTL drop");
        assert_eq!(e.stats().messages_forwarded, 0);
    }

    #[test]
    fn max_hops_zero_drops_immediately() {
        let mut e = make_engine();
        let msg = make_msg("ttl4", "ext", 0, 0);
        let events = e
            .receive_message(msg)
            .expect("test: receive_message with max_hops=0 should succeed");
        assert!(events.is_empty());
        assert_eq!(e.stats().messages_dropped_ttl, 1);
    }

    // ── 6. Fanout strategies ──────────────────────────────────────────────

    fn engine_with_strategy(strategy: FanoutStrategy) -> GossipProtocolEngine {
        GossipProtocolEngine::new(GossipConfig {
            fanout_size: 3,
            strategy,
            ..GossipConfig::default()
        })
    }

    #[test]
    fn fanout_fixed_respects_n() {
        let mut e = engine_with_strategy(FanoutStrategy::Fixed(2));
        add_peers(&mut e, 6);
        let peers = e.select_peers("ext", &FanoutStrategy::Fixed(2));
        assert_eq!(peers.len(), 2);
    }

    #[test]
    fn fanout_fixed_capped_by_available() {
        let mut e = engine_with_strategy(FanoutStrategy::Fixed(10));
        add_peers(&mut e, 3);
        let peers = e.select_peers("ext", &FanoutStrategy::Fixed(10));
        assert_eq!(peers.len(), 3);
    }

    #[test]
    fn fanout_adaptive_min() {
        let mut e = engine_with_strategy(FanoutStrategy::Adaptive { min: 2, max: 8 });
        // 5 peers → extra = 5/10 = 0 → target = 2
        add_peers(&mut e, 5);
        let peers = e.select_peers("ext", &FanoutStrategy::Adaptive { min: 2, max: 8 });
        assert!(peers.len() >= 2);
    }

    #[test]
    fn fanout_adaptive_max_clamped() {
        let mut e = engine_with_strategy(FanoutStrategy::Adaptive { min: 1, max: 3 });
        add_peers(&mut e, 100);
        let peers = e.select_peers("ext", &FanoutStrategy::Adaptive { min: 1, max: 3 });
        assert!(peers.len() <= 3);
    }

    #[test]
    fn fanout_epidemic_probability_zero() {
        let mut e = engine_with_strategy(FanoutStrategy::Epidemic(0.0));
        add_peers(&mut e, 10);
        let peers = e.select_peers("ext", &FanoutStrategy::Epidemic(0.0));
        assert!(peers.is_empty());
    }

    #[test]
    fn fanout_epidemic_probability_one() {
        let mut e = engine_with_strategy(FanoutStrategy::Epidemic(1.0));
        add_peers(&mut e, 5);
        let peers = e.select_peers("ext", &FanoutStrategy::Epidemic(1.0));
        assert_eq!(peers.len(), 5);
    }

    #[test]
    fn fanout_random_returns_fanout_size() {
        let mut e = engine_with_strategy(FanoutStrategy::Random(42));
        add_peers(&mut e, 10);
        let peers = e.select_peers("ext", &FanoutStrategy::Random(42));
        assert_eq!(peers.len(), 3); // fanout_size = 3
    }

    #[test]
    fn fanout_random_no_duplicates() {
        let mut e = engine_with_strategy(FanoutStrategy::Random(7));
        add_peers(&mut e, 10);
        let peers = e.select_peers("ext", &FanoutStrategy::Random(7));
        let mut sorted = peers.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(peers.len(), sorted.len());
    }

    #[test]
    fn fanout_priority_based_returns_fanout_size() {
        let mut e = engine_with_strategy(FanoutStrategy::PriorityBased);
        add_peers(&mut e, 8);
        // Assign differentiated scores.
        for i in 0..8_usize {
            e.peers[i].fanout_score = i as f64 / 8.0;
        }
        let peers = e.select_peers("ext", &FanoutStrategy::PriorityBased);
        assert_eq!(peers.len(), 3); // fanout_size = 3
    }

    #[test]
    fn fanout_priority_based_orders_by_score() {
        let mut e = engine_with_strategy(FanoutStrategy::PriorityBased);
        for id in ["low", "high", "mid"] {
            e.add_peer(id.into())
                .expect("test: add_peer should succeed");
        }
        e.peers.iter_mut().for_each(|p| {
            p.fanout_score = match p.id.as_str() {
                "low" => 0.1,
                "high" => 0.9,
                "mid" => 0.5,
                _ => 0.0,
            };
        });
        let peers = e.select_peers("ext", &FanoutStrategy::PriorityBased);
        // First peer selected should be the highest-scored one.
        assert_eq!(peers[0], "high");
    }

    #[test]
    fn fanout_no_active_peers_returns_empty() {
        let mut e = engine_with_strategy(FanoutStrategy::Fixed(5));
        let peers = e.select_peers("ext", &FanoutStrategy::Fixed(5));
        assert!(peers.is_empty());
    }

    #[test]
    fn fanout_excludes_origin() {
        let mut e = engine_with_strategy(FanoutStrategy::Fixed(10));
        e.add_peer("origin".into())
            .expect("test: add_peer origin should succeed");
        e.add_peer("p2".into())
            .expect("test: add_peer p2 should succeed");
        let peers = e.select_peers("origin", &FanoutStrategy::Fixed(10));
        assert!(!peers.contains(&"origin".to_owned()));
    }

    // ── 7. Peer scoring ───────────────────────────────────────────────────

    #[test]
    fn update_score_positive_increases() {
        let mut e = make_engine();
        e.add_peer("p1".into())
            .expect("test: add_peer should succeed");
        let before = e
            .peer("p1")
            .expect("test: peer p1 should exist")
            .fanout_score;
        e.update_peer_score("p1", true);
        let after = e
            .peer("p1")
            .expect("test: peer p1 should exist after score update")
            .fanout_score;
        assert!(after > before);
    }

    #[test]
    fn update_score_negative_decreases() {
        let mut e = make_engine();
        e.add_peer("p1".into())
            .expect("test: add_peer should succeed");
        // Start at 1.0 so we can observe a decrease.
        e.peers[0].fanout_score = 1.0;
        e.update_peer_score("p1", false);
        assert!(
            e.peer("p1")
                .expect("test: peer p1 should exist")
                .fanout_score
                < 1.0
        );
    }

    #[test]
    fn update_score_clamps_to_one() {
        let mut e = make_engine();
        e.add_peer("p1".into())
            .expect("test: add_peer should succeed");
        e.peers[0].fanout_score = 0.999;
        for _ in 0..1000 {
            e.update_peer_score("p1", true);
        }
        assert!(
            e.peer("p1")
                .expect("test: peer p1 should exist")
                .fanout_score
                <= 1.0
        );
    }

    #[test]
    fn update_score_clamps_to_zero() {
        let mut e = make_engine();
        e.add_peer("p1".into())
            .expect("test: add_peer should succeed");
        e.peers[0].fanout_score = 0.001;
        for _ in 0..1000 {
            e.update_peer_score("p1", false);
        }
        assert!(
            e.peer("p1")
                .expect("test: peer p1 should exist")
                .fanout_score
                >= 0.0
        );
    }

    #[test]
    fn update_score_nonexistent_peer_is_noop() {
        let mut e = make_engine();
        // Should not panic.
        e.update_peer_score("ghost", true);
    }

    // ── 8. Stats ──────────────────────────────────────────────────────────

    #[test]
    fn stats_active_peers_reflects_add_remove() {
        let mut e = make_engine();
        e.add_peer("p1".into())
            .expect("test: add_peer should succeed");
        assert_eq!(e.stats().active_peers, 1);
        e.remove_peer("p1")
            .expect("test: remove_peer should succeed");
        assert_eq!(e.stats().active_peers, 0);
    }

    #[test]
    fn stats_avg_propagation_hops_tracks_messages() {
        let mut e = make_engine();
        e.receive_message(make_msg("m1", "ext", 2, 7))
            .expect("test: receive m1 should succeed");
        e.receive_message(make_msg("m2", "ext", 4, 7))
            .expect("test: receive m2 should succeed");
        // After two messages: first sets avg=2.0, second: 0.9*2+0.1*4 = 2.2
        let avg = e.stats().avg_propagation_hops;
        assert!((avg - 2.2).abs() < 1e-9, "avg={avg}");
    }

    #[test]
    fn stats_reset_on_clear_cache() {
        let mut e = make_engine();
        e.receive_message(make_msg("m1", "ext", 0, 7))
            .expect("test: receive m1 should succeed");
        assert_eq!(e.pending_messages(), 1);
        e.clear_dedup_cache();
        assert_eq!(e.pending_messages(), 0);
    }

    // ── 9. Event log ──────────────────────────────────────────────────────

    #[test]
    fn drain_events_returns_all_events() {
        let mut e = make_engine();
        e.add_peer("p1".into())
            .expect("test: add_peer should succeed");
        e.receive_message(make_msg("m1", "ext", 0, 7))
            .expect("test: receive_message should succeed");
        let events = e.drain_events();
        assert!(!events.is_empty());
    }

    #[test]
    fn drain_events_clears_log() {
        let mut e = make_engine();
        e.add_peer("p1".into())
            .expect("test: add_peer should succeed");
        e.drain_events();
        let events2 = e.drain_events();
        assert!(events2.is_empty());
    }

    #[test]
    fn event_log_order_preserved() {
        let mut e = make_engine();
        e.add_peer("p1".into())
            .expect("test: add_peer p1 should succeed");
        e.add_peer("p2".into())
            .expect("test: add_peer p2 should succeed");
        let events = e.drain_events();
        assert!(matches!(&events[0], GossipEvent::PeerAdded(id) if id == "p1"));
        assert!(matches!(&events[1], GossipEvent::PeerAdded(id) if id == "p2"));
    }

    // ── 10. Edge cases ────────────────────────────────────────────────────

    #[test]
    fn empty_payload_message_accepted() {
        let mut e = make_engine();
        let msg = GossipMessage {
            id: "empty".into(),
            origin_peer: "ext".into(),
            hop_count: 0,
            max_hops: 7,
            payload: vec![],
            timestamp: 1,
            topic: "t".into(),
        };
        let events = e
            .receive_message(msg)
            .expect("test: receive_message with empty payload should succeed");
        assert!(events
            .iter()
            .any(|ev| matches!(ev, GossipEvent::MessageReceived(_))));
    }

    #[test]
    fn hop_count_u8_max_minus_one_dropped() {
        let mut e = make_engine();
        // max_hops = 254, hop_count = 254 → drop
        let msg = make_msg("big", "ext", 254, 254);
        let events = e
            .receive_message(msg)
            .expect("test: receive_message at u8 max hops should succeed");
        assert!(events.is_empty());
        assert_eq!(e.stats().messages_dropped_ttl, 1);
    }

    #[test]
    fn many_messages_no_panic() {
        let mut e = GossipProtocolEngine::new(GossipConfig {
            dedup_cache_size: 128,
            ..GossipConfig::default()
        });
        add_peers(&mut e, 5);
        for i in 0_u32..256 {
            let msg = make_msg(&format!("{i:08x}"), "ext", 0, 7);
            e.receive_message(msg)
                .expect("test: receive_message in bulk loop should succeed");
        }
        assert!(e.stats().messages_received > 0);
    }

    #[test]
    fn config_accessor() {
        let e = make_engine();
        assert_eq!(e.config().fanout_size, 6);
    }

    #[test]
    fn peer_lookup_by_id() {
        let mut e = make_engine();
        e.add_peer("p42".into())
            .expect("test: add_peer p42 should succeed");
        let p = e.peer("p42").expect("test: peer p42 should be found");
        assert_eq!(p.id, "p42");
    }

    #[test]
    fn peer_lookup_missing_returns_none() {
        let e = make_engine();
        assert!(e.peer("ghost").is_none());
    }

    #[test]
    fn fnv1a_known_value() {
        // FNV-1a of empty input is the offset basis.
        assert_eq!(fnv1a_64(b""), 14_695_981_039_346_656_037);
    }

    #[test]
    fn xorshift64_produces_nonzero() {
        let mut state = 12345_u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }

    #[test]
    fn xorshift64_advances_state() {
        let mut state = 12345_u64;
        let v1 = xorshift64(&mut state);
        let v2 = xorshift64(&mut state);
        assert_ne!(v1, v2);
    }

    #[test]
    fn receive_multiple_topics_tracked_independently() {
        let mut e = make_engine();
        let m1 = GossipMessage {
            id: "t1m1".into(),
            origin_peer: "ext".into(),
            hop_count: 0,
            max_hops: 7,
            payload: vec![],
            timestamp: 1,
            topic: "news".into(),
        };
        let m2 = GossipMessage {
            id: "t2m1".into(),
            origin_peer: "ext".into(),
            hop_count: 0,
            max_hops: 7,
            payload: vec![],
            timestamp: 2,
            topic: "alerts".into(),
        };
        e.receive_message(m1)
            .expect("test: receive m1 (news topic) should succeed");
        e.receive_message(m2)
            .expect("test: receive m2 (alerts topic) should succeed");
        assert_eq!(e.stats().messages_received, 2);
    }

    #[test]
    fn adaptive_strategy_scales_with_peer_count() {
        let mut e = engine_with_strategy(FanoutStrategy::Adaptive { min: 2, max: 20 });
        add_peers(&mut e, 50);
        let peers = e.select_peers("ext", &FanoutStrategy::Adaptive { min: 2, max: 20 });
        // 50 active peers → extra = 50/10 = 5 → target = min(7, 20) = 7
        assert_eq!(peers.len(), 7);
    }

    #[test]
    fn random_strategy_seed_zero_uses_engine_rng() {
        let mut e = engine_with_strategy(FanoutStrategy::Random(0));
        add_peers(&mut e, 10);
        // Should not panic and should return some peers.
        let peers = e.select_peers("ext", &FanoutStrategy::Random(0));
        assert_eq!(peers.len(), 3); // fanout_size = 3
    }

    #[test]
    fn gossip_event_message_forwarded_contains_correct_id() {
        let mut e = make_engine();
        add_peers(&mut e, 2);
        let msg = make_msg("fwd_id_check", "ext", 0, 7);
        let events = e
            .receive_message(msg)
            .expect("test: receive_message should succeed");
        let fwd = events.iter().find_map(|ev| {
            if let GossipEvent::MessageForwarded { msg_id, .. } = ev {
                Some(msg_id.clone())
            } else {
                None
            }
        });
        assert_eq!(
            fwd.expect("test: MessageForwarded event should be present"),
            "fwd_id_check"
        );
    }

    #[test]
    fn engine_new_with_zero_fanout() {
        // When fanout_size=0 AND strategy=Fixed(0), no peers are selected.
        let mut e = GossipProtocolEngine::new(GossipConfig {
            fanout_size: 0,
            strategy: FanoutStrategy::Fixed(0),
            ..GossipConfig::default()
        });
        add_peers(&mut e, 5);
        let msg = make_msg("z", "ext", 0, 7);
        let events = e
            .receive_message(msg)
            .expect("test: receive_message with zero fanout should succeed");
        // zero fanout → no forwarding event.
        assert!(!events
            .iter()
            .any(|ev| matches!(ev, GossipEvent::MessageForwarded { .. })));
    }

    #[test]
    fn engine_default_config_fanout_six() {
        let e = GossipProtocolEngine::with_defaults();
        assert_eq!(e.config().fanout_size, 6);
    }

    #[test]
    fn statistics_snapshot_is_clone() {
        let e = make_engine();
        let s1 = e.stats();
        let s2 = s1.clone();
        assert_eq!(s1.messages_received, s2.messages_received);
    }
}
