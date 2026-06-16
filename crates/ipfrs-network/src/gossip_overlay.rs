//! Gossip Overlay Manager — application-level gossip for lightweight state distribution
//!
//! Manages distribution of peer capabilities, index statistics, and gradient round status
//! without requiring DHT lookups. Uses a fanout-based epidemic broadcast with sequence-number
//! deduplication.
//!
//! ## Architecture
//!
//! Messages flow through three stages:
//! 1. **Receive** — deduplicate via `GossipState`, enqueue for application processing
//! 2. **Fanout** — select `fanout` random peers (excluding sender), enqueue outbound copies
//! 3. **Drain** — application pulls inbound/outbound queues for processing/sending
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_network::gossip_overlay::{GossipOverlayManager, GossipMessage};
//!
//! let mgr = GossipOverlayManager::default();
//! mgr.add_peer("peer-A".to_string());
//! mgr.add_peer("peer-B".to_string());
//! mgr.add_peer("peer-C".to_string());
//!
//! let msg = GossipMessage::Heartbeat {
//!     peer_id: "peer-X".to_string(),
//!     uptime_secs: 42,
//!     sequence: 1,
//! };
//!
//! let is_new = mgr.receive(msg);
//! assert!(is_new);
//!
//! let inbound = mgr.drain_inbound();
//! assert_eq!(inbound.len(), 1);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};

// ---------------------------------------------------------------------------
// GossipMessage
// ---------------------------------------------------------------------------

/// Lightweight gossip messages distributed across the overlay without DHT lookups.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GossipMessage {
    /// Peer advertising its capabilities.
    PeerAnnounce {
        peer_id: String,
        capabilities: Vec<String>,
        sequence: u64,
    },
    /// Peer reporting its current vector index statistics.
    IndexStats {
        peer_id: String,
        vector_count: u64,
        dimensions: u32,
        sequence: u64,
    },
    /// Gradient-round status update.
    RoundStatus {
        round_id: u64,
        status: String,
        peer_id: String,
        sequence: u64,
    },
    /// Liveness heartbeat.
    Heartbeat {
        peer_id: String,
        uptime_secs: u64,
        sequence: u64,
    },
}

impl GossipMessage {
    /// Human-readable message type tag.
    pub fn message_type(&self) -> &str {
        match self {
            GossipMessage::PeerAnnounce { .. } => "peer_announce",
            GossipMessage::IndexStats { .. } => "index_stats",
            GossipMessage::RoundStatus { .. } => "round_status",
            GossipMessage::Heartbeat { .. } => "heartbeat",
        }
    }

    /// Monotonically-increasing sequence number carried by this message.
    pub fn sequence(&self) -> u64 {
        match self {
            GossipMessage::PeerAnnounce { sequence, .. } => *sequence,
            GossipMessage::IndexStats { sequence, .. } => *sequence,
            GossipMessage::RoundStatus { sequence, .. } => *sequence,
            GossipMessage::Heartbeat { sequence, .. } => *sequence,
        }
    }

    /// Originating peer identifier.
    pub fn peer_id(&self) -> &str {
        match self {
            GossipMessage::PeerAnnounce { peer_id, .. } => peer_id,
            GossipMessage::IndexStats { peer_id, .. } => peer_id,
            GossipMessage::RoundStatus { peer_id, .. } => peer_id,
            GossipMessage::Heartbeat { peer_id, .. } => peer_id,
        }
    }
}

// ---------------------------------------------------------------------------
// GossipState — per-peer sequence deduplication
// ---------------------------------------------------------------------------

/// Tracks the highest sequence number seen from each peer to deduplicate messages.
#[derive(Debug, Default)]
pub struct GossipState {
    /// Maps `peer_id` → highest `sequence` observed so far.
    pub seen: HashMap<String, u64>,
}

impl GossipState {
    /// Returns `true` when `sequence` is strictly greater than the last recorded value
    /// for `peer_id` (or when the peer has never been seen).
    pub fn is_new(&self, peer_id: &str, sequence: u64) -> bool {
        match self.seen.get(peer_id) {
            Some(&last) => sequence > last,
            None => true,
        }
    }

    /// Advance (or initialise) the tracked sequence for `peer_id`.
    /// Only updates when `sequence` is greater than the currently stored value.
    pub fn record(&mut self, peer_id: &str, sequence: u64) {
        let entry = self.seen.entry(peer_id.to_string()).or_insert(0);
        if sequence > *entry {
            *entry = sequence;
        }
    }

    /// Remove all peers whose highest recorded sequence is ≤ `threshold_seq`.
    /// Useful for cleaning up stale entries from long-gone peers.
    pub fn prune_stale(&mut self, threshold_seq: u64) {
        self.seen.retain(|_, &mut seq| seq > threshold_seq);
    }
}

// ---------------------------------------------------------------------------
// GossipFanout — peer selection for epidemic broadcast
// ---------------------------------------------------------------------------

/// Maintains the candidate peer set and selects fanout targets for each forwarded message.
#[derive(Debug)]
pub struct GossipFanout {
    /// All known candidate peers.
    pub peers: Vec<String>,
    /// Number of peers to forward each message to.
    pub fanout: usize,
}

impl Default for GossipFanout {
    fn default() -> Self {
        Self {
            peers: Vec::new(),
            fanout: 3,
        }
    }
}

impl GossipFanout {
    /// Create a new fanout structure with the given target fan-out width.
    pub fn new(fanout: usize) -> Self {
        Self {
            peers: Vec::new(),
            fanout,
        }
    }

    /// Select up to `self.fanout` targets from the peer list, excluding `exclude`.
    ///
    /// Selection is deterministic (round-robin starting offset derived from the
    /// FNV-1a hash of `exclude`) so that the same sender always produces the same
    /// forwarding set given the same peer list, while different senders produce
    /// different sets — ensuring good epidemic coverage without requiring a PRNG.
    pub fn select_targets(&self, exclude: &str) -> Vec<String> {
        let candidates: Vec<&String> = self
            .peers
            .iter()
            .filter(|p| p.as_str() != exclude)
            .collect();
        if candidates.is_empty() {
            return Vec::new();
        }

        // FNV-1a hash of `exclude` bytes for a deterministic, cheap starting offset.
        let hash = fnv1a_hash(exclude.as_bytes());
        let start = (hash as usize) % candidates.len();

        let take = self.fanout.min(candidates.len());
        let mut result = Vec::with_capacity(take);
        for i in 0..take {
            let idx = (start + i) % candidates.len();
            result.push(candidates[idx].clone());
        }
        result
    }

    /// Register a new candidate peer. Silently ignores duplicates.
    pub fn add_peer(&mut self, peer_id: String) {
        if !self.peers.contains(&peer_id) {
            self.peers.push(peer_id);
        }
    }

    /// Remove a peer from the candidate list. No-op if the peer is not present.
    pub fn remove_peer(&mut self, peer_id: &str) {
        self.peers.retain(|p| p.as_str() != peer_id);
    }

    /// Number of registered peers.
    pub fn len(&self) -> usize {
        self.peers.len()
    }

    /// Returns `true` if no peers are registered.
    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }
}

/// Simple FNV-1a 64-bit hash for deterministic offset computation.
fn fnv1a_hash(bytes: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    let mut hash = FNV_OFFSET_BASIS;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// GossipStats — lock-free atomic counters
// ---------------------------------------------------------------------------

/// Lock-free counters tracking message flow through the overlay.
#[derive(Debug, Default)]
pub struct GossipStats {
    /// Total messages received (new + duplicates).
    pub total_received: AtomicU64,
    /// Messages accepted as novel (forwarded to inbound queue).
    pub total_new: AtomicU64,
    /// Messages discarded as duplicates.
    pub total_duplicates: AtomicU64,
    /// Individual (target, message) pairs enqueued for fanout.
    pub total_fanned_out: AtomicU64,
}

/// A point-in-time snapshot of [`GossipStats`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GossipStatsSnapshot {
    pub total_received: u64,
    pub total_new: u64,
    pub total_duplicates: u64,
    pub total_fanned_out: u64,
}

impl GossipStats {
    /// Atomically read all counters into a snapshot.
    pub fn snapshot(&self) -> GossipStatsSnapshot {
        GossipStatsSnapshot {
            total_received: self.total_received.load(Ordering::Relaxed),
            total_new: self.total_new.load(Ordering::Relaxed),
            total_duplicates: self.total_duplicates.load(Ordering::Relaxed),
            total_fanned_out: self.total_fanned_out.load(Ordering::Relaxed),
        }
    }
}

// ---------------------------------------------------------------------------
// GossipOverlayManager — the top-level coordinator
// ---------------------------------------------------------------------------

/// Manages the application-level gossip overlay.
///
/// All operations are thread-safe; queues and state are protected by `Mutex`/`RwLock`.
///
/// ### Typical usage pattern
///
/// 1. Register known peers via [`add_peer`](Self::add_peer).
/// 2. On network receive, call [`receive`](Self::receive); it deduplicates and enqueues.
/// 3. Periodically call [`drain_outbound`](Self::drain_outbound) to obtain (target, msg) pairs
///    and forward them over the transport layer.
/// 4. Call [`drain_inbound`](Self::drain_inbound) to pull novel messages for local processing.
#[derive(Debug, Default)]
pub struct GossipOverlayManager {
    state: Mutex<GossipState>,
    fanout: RwLock<GossipFanout>,
    inbound_queue: Mutex<VecDeque<GossipMessage>>,
    outbound_queue: Mutex<VecDeque<(String, GossipMessage)>>,
    pub stats: GossipStats,
}

impl GossipOverlayManager {
    /// Create a manager with a custom fanout width.
    pub fn with_fanout(fanout: usize) -> Self {
        Self {
            fanout: RwLock::new(GossipFanout::new(fanout)),
            ..Default::default()
        }
    }

    // ------------------------------------------------------------------
    // Peer management (delegates to GossipFanout)
    // ------------------------------------------------------------------

    /// Register a peer as a fanout candidate.
    pub fn add_peer(&self, peer_id: String) {
        self.fanout
            .write()
            .expect("fanout RwLock poisoned")
            .add_peer(peer_id);
    }

    /// Deregister a peer from the fanout candidate list.
    pub fn remove_peer(&self, peer_id: &str) {
        self.fanout
            .write()
            .expect("fanout RwLock poisoned")
            .remove_peer(peer_id);
    }

    /// Number of registered fanout candidate peers.
    pub fn peer_count(&self) -> usize {
        self.fanout.read().expect("fanout RwLock poisoned").len()
    }

    // ------------------------------------------------------------------
    // Core message handling
    // ------------------------------------------------------------------

    /// Process an incoming message.
    ///
    /// Returns `true` when the message is novel (first time this sequence number from this
    /// peer has been seen).  When novel, the message is:
    /// - pushed to the **inbound** queue for application processing, and
    /// - pushed to the **outbound** queue for each fanout target.
    ///
    /// Returns `false` and increments the duplicate counter when the message was already seen.
    pub fn receive(&self, msg: GossipMessage) -> bool {
        self.stats.total_received.fetch_add(1, Ordering::Relaxed);

        let pid = msg.peer_id().to_string();
        let seq = msg.sequence();

        let is_new = {
            let mut state = self.state.lock().expect("state Mutex poisoned");
            if state.is_new(&pid, seq) {
                state.record(&pid, seq);
                true
            } else {
                false
            }
        };

        if !is_new {
            self.stats.total_duplicates.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        self.stats.total_new.fetch_add(1, Ordering::Relaxed);

        // Enqueue for local application processing.
        self.inbound_queue
            .lock()
            .expect("inbound_queue Mutex poisoned")
            .push_back(msg.clone());

        // Select fanout targets and enqueue outbound copies.
        let targets = self
            .fanout
            .read()
            .expect("fanout RwLock poisoned")
            .select_targets(&pid);

        let fanout_count = targets.len() as u64;
        {
            let mut out = self
                .outbound_queue
                .lock()
                .expect("outbound_queue Mutex poisoned");
            for target in targets {
                out.push_back((target, msg.clone()));
            }
        }
        self.stats
            .total_fanned_out
            .fetch_add(fanout_count, Ordering::Relaxed);

        true
    }

    /// Drain and return all messages currently in the inbound queue.
    pub fn drain_inbound(&self) -> Vec<GossipMessage> {
        let mut q = self
            .inbound_queue
            .lock()
            .expect("inbound_queue Mutex poisoned");
        q.drain(..).collect()
    }

    /// Drain and return all `(target_peer_id, message)` pairs from the outbound queue.
    pub fn drain_outbound(&self) -> Vec<(String, GossipMessage)> {
        let mut q = self
            .outbound_queue
            .lock()
            .expect("outbound_queue Mutex poisoned");
        q.drain(..).collect()
    }

    /// Broadcast `msg` to **all** registered fanout peers (ignoring the message's own
    /// `peer_id` as a sender — useful for locally-originated messages).
    pub fn broadcast(&self, msg: GossipMessage) {
        let peers: Vec<String> = self
            .fanout
            .read()
            .expect("fanout RwLock poisoned")
            .peers
            .clone();

        let count = peers.len() as u64;
        {
            let mut out = self
                .outbound_queue
                .lock()
                .expect("outbound_queue Mutex poisoned");
            for peer in peers {
                out.push_back((peer, msg.clone()));
            }
        }
        self.stats
            .total_fanned_out
            .fetch_add(count, Ordering::Relaxed);
    }

    /// A point-in-time snapshot of all gossip counters.
    pub fn stats_snapshot(&self) -> GossipStatsSnapshot {
        self.stats.snapshot()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper constructors ---------------------------------------------------

    fn make_announce(peer_id: &str, seq: u64) -> GossipMessage {
        GossipMessage::PeerAnnounce {
            peer_id: peer_id.to_string(),
            capabilities: vec!["vector-search".to_string()],
            sequence: seq,
        }
    }

    fn make_index_stats(peer_id: &str, seq: u64) -> GossipMessage {
        GossipMessage::IndexStats {
            peer_id: peer_id.to_string(),
            vector_count: 1000,
            dimensions: 768,
            sequence: seq,
        }
    }

    fn make_round_status(peer_id: &str, seq: u64) -> GossipMessage {
        GossipMessage::RoundStatus {
            round_id: 7,
            status: "running".to_string(),
            peer_id: peer_id.to_string(),
            sequence: seq,
        }
    }

    fn make_heartbeat(peer_id: &str, seq: u64) -> GossipMessage {
        GossipMessage::Heartbeat {
            peer_id: peer_id.to_string(),
            uptime_secs: 300,
            sequence: seq,
        }
    }

    // 1. message_type for PeerAnnounce
    #[test]
    fn test_message_type_peer_announce() {
        let msg = make_announce("p1", 1);
        assert_eq!(msg.message_type(), "peer_announce");
    }

    // 2. message_type for IndexStats
    #[test]
    fn test_message_type_index_stats() {
        let msg = make_index_stats("p1", 1);
        assert_eq!(msg.message_type(), "index_stats");
    }

    // 3. message_type for RoundStatus
    #[test]
    fn test_message_type_round_status() {
        let msg = make_round_status("p1", 1);
        assert_eq!(msg.message_type(), "round_status");
    }

    // 4. message_type for Heartbeat
    #[test]
    fn test_message_type_heartbeat() {
        let msg = make_heartbeat("p1", 1);
        assert_eq!(msg.message_type(), "heartbeat");
    }

    // 5. GossipMessage::sequence and peer_id accessors
    #[test]
    fn test_message_accessors() {
        let msg = make_announce("alice", 42);
        assert_eq!(msg.sequence(), 42);
        assert_eq!(msg.peer_id(), "alice");
    }

    // 6. GossipState::is_new — first message is always new
    #[test]
    fn test_gossip_state_is_new_first() {
        let state = GossipState::default();
        assert!(state.is_new("peer-A", 1));
    }

    // 7. GossipState deduplication — same sequence not new after record
    #[test]
    fn test_gossip_state_dedup() {
        let mut state = GossipState::default();
        state.record("peer-A", 5);
        assert!(!state.is_new("peer-A", 5));
        assert!(!state.is_new("peer-A", 4));
        assert!(state.is_new("peer-A", 6));
    }

    // 8. GossipState::prune_stale
    #[test]
    fn test_gossip_state_prune_stale() {
        let mut state = GossipState::default();
        state.record("peer-A", 3);
        state.record("peer-B", 7);
        state.record("peer-C", 10);
        state.prune_stale(7); // removes peer-A (3 ≤ 7) and peer-B (7 ≤ 7)
        assert!(!state.seen.contains_key("peer-A"));
        assert!(!state.seen.contains_key("peer-B"));
        assert!(state.seen.contains_key("peer-C"));
    }

    // 9. GossipFanout::select_targets excludes sender
    #[test]
    fn test_fanout_select_excludes_sender() {
        let mut fanout = GossipFanout::new(3);
        fanout.add_peer("A".to_string());
        fanout.add_peer("B".to_string());
        fanout.add_peer("C".to_string());
        fanout.add_peer("D".to_string());

        let targets = fanout.select_targets("A");
        assert!(
            !targets.contains(&"A".to_string()),
            "sender must not appear in targets"
        );
        assert!(targets.len() <= 3);
    }

    // 10. GossipFanout::select_targets respects fanout width
    #[test]
    fn test_fanout_select_width() {
        let mut fanout = GossipFanout::new(2);
        for i in 0..10u32 {
            fanout.add_peer(format!("peer-{i}"));
        }
        let targets = fanout.select_targets("peer-0");
        assert_eq!(targets.len(), 2);
    }

    // 11. receive returns true for new message, false for duplicate
    #[test]
    fn test_receive_new_vs_duplicate() {
        let mgr = GossipOverlayManager::default();
        let msg = make_heartbeat("peer-X", 1);
        assert!(mgr.receive(msg.clone()), "first receive must be new");
        assert!(!mgr.receive(msg), "second receive must be duplicate");
    }

    // 12. receive enqueues fanout messages
    #[test]
    fn test_receive_enqueues_fanout() {
        let mgr = GossipOverlayManager::default();
        mgr.add_peer("peer-A".to_string());
        mgr.add_peer("peer-B".to_string());
        mgr.add_peer("peer-C".to_string());

        let msg = make_heartbeat("sender", 1);
        mgr.receive(msg);

        let out = mgr.drain_outbound();
        // fanout default = 3, all 3 peers should receive it (sender is not in peer list)
        assert_eq!(out.len(), 3, "all 3 registered peers should receive fanout");
        for (target, _) in &out {
            assert_ne!(target, "sender");
        }
    }

    // 13. drain_inbound clears queue
    #[test]
    fn test_drain_inbound_clears_queue() {
        let mgr = GossipOverlayManager::default();
        mgr.receive(make_announce("p1", 1));
        mgr.receive(make_announce("p2", 1));
        let first_drain = mgr.drain_inbound();
        assert_eq!(first_drain.len(), 2);
        let second_drain = mgr.drain_inbound();
        assert!(second_drain.is_empty(), "queue should be empty after drain");
    }

    // 14. drain_outbound clears queue
    #[test]
    fn test_drain_outbound_clears_queue() {
        let mgr = GossipOverlayManager::default();
        mgr.add_peer("peer-A".to_string());
        mgr.receive(make_heartbeat("sender", 1));
        let first = mgr.drain_outbound();
        assert!(!first.is_empty());
        let second = mgr.drain_outbound();
        assert!(
            second.is_empty(),
            "outbound queue should be empty after drain"
        );
    }

    // 15. broadcast reaches all registered peers
    #[test]
    fn test_broadcast_reaches_all_peers() {
        let mgr = GossipOverlayManager::default();
        mgr.add_peer("peer-A".to_string());
        mgr.add_peer("peer-B".to_string());
        mgr.add_peer("peer-C".to_string());

        let msg = make_index_stats("local-node", 1);
        mgr.broadcast(msg.clone());

        let out = mgr.drain_outbound();
        assert_eq!(
            out.len(),
            3,
            "broadcast must enqueue one copy per registered peer"
        );
        let targets: Vec<&str> = out.iter().map(|(t, _)| t.as_str()).collect();
        assert!(targets.contains(&"peer-A"));
        assert!(targets.contains(&"peer-B"));
        assert!(targets.contains(&"peer-C"));
    }

    // 16. stats accumulate correctly
    #[test]
    fn test_stats_accumulation() {
        let mgr = GossipOverlayManager::default();
        mgr.add_peer("peer-A".to_string());
        mgr.add_peer("peer-B".to_string());

        let msg = make_round_status("origin", 10);
        mgr.receive(msg.clone()); // new
        mgr.receive(msg.clone()); // duplicate

        let snap = mgr.stats_snapshot();
        assert_eq!(snap.total_received, 2);
        assert_eq!(snap.total_new, 1);
        assert_eq!(snap.total_duplicates, 1);
        // fanout = 3 but only 2 peers registered → 2 outbound
        assert_eq!(snap.total_fanned_out, 2);
    }

    // 17. peer_count reflects add/remove operations
    #[test]
    fn test_peer_count() {
        let mgr = GossipOverlayManager::default();
        assert_eq!(mgr.peer_count(), 0);
        mgr.add_peer("p1".to_string());
        mgr.add_peer("p2".to_string());
        assert_eq!(mgr.peer_count(), 2);
        mgr.remove_peer("p1");
        assert_eq!(mgr.peer_count(), 1);
    }

    // 18. duplicate message does NOT enqueue to inbound
    #[test]
    fn test_duplicate_not_enqueued_to_inbound() {
        let mgr = GossipOverlayManager::default();
        let msg = make_announce("p1", 5);
        mgr.receive(msg.clone());
        mgr.receive(msg.clone()); // duplicate — should not enqueue

        let inbound = mgr.drain_inbound();
        assert_eq!(
            inbound.len(),
            1,
            "only one copy should reach the inbound queue"
        );
    }

    // 19. higher sequence after duplicate is treated as new
    #[test]
    fn test_higher_sequence_after_duplicate_is_new() {
        let mgr = GossipOverlayManager::default();
        let msg_v1 = make_heartbeat("peer-Z", 1);
        let msg_v2 = make_heartbeat("peer-Z", 2);

        assert!(mgr.receive(msg_v1));
        assert!(!mgr.receive(make_heartbeat("peer-Z", 1))); // same seq → dup
        assert!(mgr.receive(msg_v2)); // higher seq → new

        let snap = mgr.stats_snapshot();
        assert_eq!(snap.total_new, 2);
        assert_eq!(snap.total_duplicates, 1);
    }

    // 20. serialization round-trip for all variants
    #[test]
    fn test_serde_round_trip() {
        let messages = vec![
            make_announce("p1", 1),
            make_index_stats("p2", 2),
            make_round_status("p3", 3),
            make_heartbeat("p4", 4),
        ];
        for msg in messages {
            let json = serde_json::to_string(&msg).expect("serialization failed");
            let decoded: GossipMessage =
                serde_json::from_str(&json).expect("deserialization failed");
            assert_eq!(msg, decoded);
        }
    }
}
