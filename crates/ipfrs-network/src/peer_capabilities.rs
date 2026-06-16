//! Peer Capability Advertisement and Negotiation
//!
//! This module provides a high-level registry for advertising and negotiating
//! capabilities between P2P peers. Unlike the lower-level `capability_registry`
//! module (which is tick-based and protocol-oriented), this module is designed
//! around wall-clock TTL semantics, structured requirement matching
//! (`require_all` / `require_any`), and a richer set of first-class capability
//! variants aligned with the IPFRS protocol suite.
//!
//! # Key Types
//!
//! - [`PeerCapability`] — enumeration of well-known and custom capabilities.
//! - [`PeerCapabilitySet`] — a single peer's advertised capability snapshot.
//! - [`CapabilityConfig`] — local configuration: what we advertise, what we
//!   require, how many peers we track, and TTL.
//! - [`CapabilityRegistry`] — the central in-memory registry; register, query,
//!   expire, and remove peer capability sets.
//! - [`CapabilityStats`] — counters and per-capability distribution histograms.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::peer_capabilities::{
//!     CapabilityConfig, CapabilityRegistry, PeerCapability, PeerCapabilitySet,
//! };
//! use std::collections::HashSet;
//!
//! let config = CapabilityConfig {
//!     local_capabilities: vec![PeerCapability::BitswapV2, PeerCapability::DHTKademlia],
//!     require_all: vec![PeerCapability::BitswapV1],
//!     require_any: vec![PeerCapability::DHTKademlia, PeerCapability::DHTCorvus],
//!     max_peers: 256,
//!     ttl_ms: 60_000,
//! };
//!
//! let mut registry = CapabilityRegistry::new(config);
//!
//! let peer = PeerCapabilitySet {
//!     peer_id: "QmPeer1".to_string(),
//!     capabilities: [PeerCapability::BitswapV1, PeerCapability::DHTKademlia]
//!         .into_iter()
//!         .collect(),
//!     version: "0.2.0".to_string(),
//!     advertised_at: 1_000,
//!     last_verified: None,
//! };
//!
//! assert!(registry.register(peer));
//! assert_eq!(registry.peer_count(), 1);
//! ```

use std::collections::{HashMap, HashSet};

// ── PeerCapability ────────────────────────────────────────────────────────────

/// A well-known or custom capability that a peer may advertise.
///
/// Variants cover the full IPFRS protocol suite: block exchange, DHT flavours,
/// pub/sub, graph sync, semantic search, tensor logic, relay, NAT traversal,
/// and an open-ended `Custom` escape hatch.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PeerCapability {
    /// Bitswap block-exchange protocol, version 1.
    BitswapV1,
    /// Bitswap block-exchange protocol, version 2 (session-aware).
    BitswapV2,
    /// Standard Kademlia DHT routing.
    DHTKademlia,
    /// Corvus DHT — IPFRS-native DHT with semantic routing extensions.
    DHTCorvus,
    /// GossipSub pub/sub mesh messaging.
    GossipSub,
    /// GraphSync — IPLD graph synchronisation protocol.
    GraphSync,
    /// Semantic vector-similarity search over DHT keyspace.
    SemanticSearch,
    /// TensorLogic rule engine and inference capability.
    TensorLogic,
    /// Circuit-relay service (the peer acts as a relay).
    Relay,
    /// NAT traversal via AutoNAT / DCUtR / hole-punching.
    NatTraversal,
    /// Application-defined capability identified by an arbitrary string.
    Custom(String),
}

impl PeerCapability {
    /// Returns a stable ASCII key that uniquely identifies this capability
    /// variant.  Used for the stats distribution histogram.
    pub fn key(&self) -> String {
        match self {
            PeerCapability::BitswapV1 => "bitswap_v1".to_string(),
            PeerCapability::BitswapV2 => "bitswap_v2".to_string(),
            PeerCapability::DHTKademlia => "dht_kademlia".to_string(),
            PeerCapability::DHTCorvus => "dht_corvus".to_string(),
            PeerCapability::GossipSub => "gossipsub".to_string(),
            PeerCapability::GraphSync => "graphsync".to_string(),
            PeerCapability::SemanticSearch => "semantic_search".to_string(),
            PeerCapability::TensorLogic => "tensor_logic".to_string(),
            PeerCapability::Relay => "relay".to_string(),
            PeerCapability::NatTraversal => "nat_traversal".to_string(),
            PeerCapability::Custom(s) => format!("custom:{s}"),
        }
    }

    /// Returns a `&'static str` label for well-known variants.  Custom
    /// capabilities return `"custom"`.
    pub fn static_name(&self) -> &'static str {
        match self {
            PeerCapability::BitswapV1 => "bitswap_v1",
            PeerCapability::BitswapV2 => "bitswap_v2",
            PeerCapability::DHTKademlia => "dht_kademlia",
            PeerCapability::DHTCorvus => "dht_corvus",
            PeerCapability::GossipSub => "gossipsub",
            PeerCapability::GraphSync => "graphsync",
            PeerCapability::SemanticSearch => "semantic_search",
            PeerCapability::TensorLogic => "tensor_logic",
            PeerCapability::Relay => "relay",
            PeerCapability::NatTraversal => "nat_traversal",
            PeerCapability::Custom(_) => "custom",
        }
    }
}

// ── PeerCapabilitySet ─────────────────────────────────────────────────────────

/// The full capability advertisement for a single remote peer.
///
/// This is the unit stored inside [`CapabilityRegistry`].  The
/// `advertised_at` field carries a Unix-epoch millisecond timestamp which is
/// compared against the registry's `ttl_ms` to determine freshness.
#[derive(Debug, Clone)]
pub struct PeerCapabilitySet {
    /// Unique peer identifier (string form of the libp2p `PeerId`).
    pub peer_id: String,
    /// Set of capabilities this peer claims to support.
    pub capabilities: HashSet<PeerCapability>,
    /// Peer's self-reported IPFRS protocol version string, e.g. `"0.2.0"`.
    pub version: String,
    /// Unix-epoch millisecond timestamp when this record was first advertised.
    pub advertised_at: u64,
    /// Unix-epoch millisecond timestamp of the most recent successful
    /// verification, if any.
    pub last_verified: Option<u64>,
}

impl PeerCapabilitySet {
    /// Returns `true` if this peer supports `cap`.
    #[inline]
    pub fn has(&self, cap: &PeerCapability) -> bool {
        self.capabilities.contains(cap)
    }

    /// Returns `true` if this peer supports **all** capabilities in `caps`.
    pub fn has_all(&self, caps: &[PeerCapability]) -> bool {
        caps.iter().all(|c| self.capabilities.contains(c))
    }

    /// Returns `true` if this peer supports **at least one** capability in
    /// `caps`.  Returns `true` when `caps` is empty (vacuous truth).
    pub fn has_any(&self, caps: &[PeerCapability]) -> bool {
        if caps.is_empty() {
            return true;
        }
        caps.iter().any(|c| self.capabilities.contains(c))
    }

    /// Returns `true` if the record has expired given `now_ms` and `ttl_ms`.
    ///
    /// Expiry is defined as `now_ms >= advertised_at + ttl_ms`.
    #[inline]
    pub fn is_expired(&self, now_ms: u64, ttl_ms: u64) -> bool {
        now_ms >= self.advertised_at.saturating_add(ttl_ms)
    }
}

// ── CapabilityConfig ──────────────────────────────────────────────────────────

/// Configuration for the local node's capability advertisement and peer
/// filtering policy.
#[derive(Debug, Clone)]
pub struct CapabilityConfig {
    /// Capabilities that this local node advertises to remote peers.
    pub local_capabilities: Vec<PeerCapability>,
    /// Peers that do not have **all** of these capabilities are rejected during
    /// [`CapabilityRegistry::register`].
    pub require_all: Vec<PeerCapability>,
    /// Peers that have **none** of these capabilities are rejected during
    /// [`CapabilityRegistry::register`].  An empty list means no
    /// `require_any` constraint.
    pub require_any: Vec<PeerCapability>,
    /// Maximum number of peers the registry will hold simultaneously.
    /// Attempts to register beyond this limit return `false`.
    pub max_peers: usize,
    /// Time-to-live in milliseconds after which a `PeerCapabilitySet` is
    /// considered stale and eligible for purging.
    pub ttl_ms: u64,
}

impl Default for CapabilityConfig {
    fn default() -> Self {
        Self {
            local_capabilities: Vec::new(),
            require_all: Vec::new(),
            require_any: Vec::new(),
            max_peers: 1024,
            ttl_ms: 300_000, // 5 minutes
        }
    }
}

// ── CapabilityStats ───────────────────────────────────────────────────────────

/// Runtime statistics for a [`CapabilityRegistry`].
#[derive(Debug, Clone, Default)]
pub struct CapabilityStats {
    /// Total number of successful [`CapabilityRegistry::register`] calls since
    /// the registry was created.
    pub total_registered: u64,
    /// Total number of successful [`CapabilityRegistry::remove`] calls.
    pub total_removed: u64,
    /// Number of registration attempts that were rejected because the peer
    /// did not meet the configured requirements.
    pub peers_filtered_out: u64,
    /// Per-capability count of how many currently registered peers declare each
    /// capability.  Keys are produced by [`PeerCapability::key`].
    pub capability_distribution: HashMap<String, u64>,
}

// ── CapabilityRegistry ────────────────────────────────────────────────────────

/// Central registry for peer capability advertisements.
///
/// The registry enforces:
/// - **Requirement filtering** — peers must satisfy `require_all` and
///   `require_any` from the [`CapabilityConfig`] to be accepted.
/// - **Capacity limiting** — at most `max_peers` entries are held.
/// - **TTL expiry** — stale entries can be purged via [`CapabilityRegistry::purge_expired`].
///
/// All methods take `&mut self`; the registry is not internally synchronised.
/// Wrap in an `Arc<Mutex<_>>` for concurrent access.
pub struct CapabilityRegistry {
    config: CapabilityConfig,
    peers: HashMap<String, PeerCapabilitySet>,
    stats: CapabilityStats,
}

impl CapabilityRegistry {
    /// Create a new registry with the given configuration.
    pub fn new(config: CapabilityConfig) -> Self {
        Self {
            config,
            peers: HashMap::new(),
            stats: CapabilityStats::default(),
        }
    }

    // ── Registration ──────────────────────────────────────────────────────────

    /// Register `peer` in the registry.
    ///
    /// Returns `true` on success, `false` when:
    /// - The peer does not meet the configured requirements
    ///   (`meets_requirements`), or
    /// - The registry is at capacity (`max_peers`).
    ///
    /// If the same `peer_id` is already present the old entry is replaced
    /// (the stats counters still increment as if it were a new registration).
    pub fn register(&mut self, peer: PeerCapabilitySet) -> bool {
        if !self.meets_requirements(&peer) {
            self.stats.peers_filtered_out = self.stats.peers_filtered_out.saturating_add(1);
            return false;
        }

        // Enforce max_peers only when the peer is genuinely new.
        let is_new = !self.peers.contains_key(&peer.peer_id);
        if is_new && self.peers.len() >= self.config.max_peers {
            self.stats.peers_filtered_out = self.stats.peers_filtered_out.saturating_add(1);
            return false;
        }

        // If replacing an existing entry, remove the old capability counts
        // before adding the new ones.
        if let Some(old) = self.peers.get(&peer.peer_id) {
            for cap in &old.capabilities {
                let key = cap.key();
                let count = self.stats.capability_distribution.entry(key).or_insert(0);
                *count = count.saturating_sub(1);
            }
        }

        // Update the capability histogram with the new peer's capabilities.
        for cap in &peer.capabilities {
            *self
                .stats
                .capability_distribution
                .entry(cap.key())
                .or_insert(0) += 1;
        }

        self.peers.insert(peer.peer_id.clone(), peer);
        self.stats.total_registered = self.stats.total_registered.saturating_add(1);
        true
    }

    // ── Requirement Checking ──────────────────────────────────────────────────

    /// Returns `true` if `peer` satisfies the configured `require_all` and
    /// `require_any` constraints.
    ///
    /// - `require_all`: the peer must possess every listed capability.
    /// - `require_any`: the peer must possess at least one listed capability
    ///   (no constraint when the list is empty).
    pub fn meets_requirements(&self, peer: &PeerCapabilitySet) -> bool {
        // All of require_all must be present.
        if !peer.has_all(&self.config.require_all) {
            return false;
        }
        // At least one of require_any must be present (skip check when empty).
        if !self.config.require_any.is_empty() && !peer.has_any(&self.config.require_any) {
            return false;
        }
        true
    }

    // ── Removal ───────────────────────────────────────────────────────────────

    /// Remove the peer identified by `peer_id` from the registry.
    ///
    /// Returns `true` if the peer was present and removed, `false` otherwise.
    pub fn remove(&mut self, peer_id: &str) -> bool {
        if let Some(removed) = self.peers.remove(peer_id) {
            for cap in &removed.capabilities {
                let key = cap.key();
                let count = self.stats.capability_distribution.entry(key).or_insert(0);
                *count = count.saturating_sub(1);
            }
            self.stats.total_removed = self.stats.total_removed.saturating_add(1);
            true
        } else {
            false
        }
    }

    // ── Lookup ────────────────────────────────────────────────────────────────

    /// Retrieve the [`PeerCapabilitySet`] for `peer_id`, if present.
    pub fn get_peer(&self, peer_id: &str) -> Option<&PeerCapabilitySet> {
        self.peers.get(peer_id)
    }

    /// Return all peers that advertise `cap`.
    pub fn peers_with_capability(&self, cap: &PeerCapability) -> Vec<&PeerCapabilitySet> {
        self.peers.values().filter(|p| p.has(cap)).collect()
    }

    /// Return all peers that advertise **every** capability in `caps`.
    pub fn peers_with_all(&self, caps: &[PeerCapability]) -> Vec<&PeerCapabilitySet> {
        self.peers.values().filter(|p| p.has_all(caps)).collect()
    }

    /// Return all peers that advertise **at least one** capability in `caps`.
    pub fn peers_with_any(&self, caps: &[PeerCapability]) -> Vec<&PeerCapabilitySet> {
        self.peers.values().filter(|p| p.has_any(caps)).collect()
    }

    // ── Expiry ────────────────────────────────────────────────────────────────

    /// Remove all peers whose `advertised_at` is older than `now - ttl_ms`.
    ///
    /// Returns the number of entries purged.
    pub fn purge_expired(&mut self, now: u64) -> usize {
        let ttl_ms = self.config.ttl_ms;

        // Collect keys to remove first to satisfy the borrow checker.
        let expired_ids: Vec<String> = self
            .peers
            .iter()
            .filter(|(_, p)| p.is_expired(now, ttl_ms))
            .map(|(id, _)| id.clone())
            .collect();

        let count = expired_ids.len();
        for id in expired_ids {
            // Use `remove` to update stats counters correctly.
            self.remove(&id);
            // `remove` increments `total_removed`; adjust back so TTL-based
            // purges are distinguishable from explicit removals if the caller
            // tracks `total_removed` themselves.  Currently we keep them in
            // the same bucket — callers can diff the before/after value.
        }
        count
    }

    // ── Capability Naming ─────────────────────────────────────────────────────

    /// Returns a `&'static str` label for well-known capability variants.
    ///
    /// For [`PeerCapability::Custom`] the returned string is always
    /// `"custom"`.  Use [`PeerCapability::key`] when you need the full
    /// dynamic key including the custom payload.
    pub fn capability_name(cap: &PeerCapability) -> &'static str {
        cap.static_name()
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Returns the number of peers currently in the registry.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Returns a reference to the runtime statistics snapshot.
    ///
    /// The snapshot reflects the live state of the registry; the histogram
    /// entries are decremented on removal, so they track currently registered
    /// peers rather than lifetime totals.
    pub fn stats(&self) -> &CapabilityStats {
        &self.stats
    }

    /// Returns the local capabilities configured for this node.
    pub fn local_capabilities(&self) -> &[PeerCapability] {
        &self.config.local_capabilities
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn default_config() -> CapabilityConfig {
        CapabilityConfig {
            local_capabilities: vec![PeerCapability::BitswapV2, PeerCapability::DHTKademlia],
            require_all: Vec::new(),
            require_any: Vec::new(),
            max_peers: 10,
            ttl_ms: 60_000,
        }
    }

    fn make_peer(
        peer_id: &str,
        caps: impl IntoIterator<Item = PeerCapability>,
        advertised_at: u64,
    ) -> PeerCapabilitySet {
        PeerCapabilitySet {
            peer_id: peer_id.to_string(),
            capabilities: caps.into_iter().collect(),
            version: "0.2.0".to_string(),
            advertised_at,
            last_verified: None,
        }
    }

    // ── 1. Basic registration ─────────────────────────────────────────────────

    #[test]
    fn test_register_and_count() {
        let mut reg = CapabilityRegistry::new(default_config());
        let peer = make_peer("peer1", [PeerCapability::BitswapV1], 1_000);
        assert!(reg.register(peer));
        assert_eq!(reg.peer_count(), 1);
    }

    // ── 2. Stats total_registered increments ─────────────────────────────────

    #[test]
    fn test_stats_total_registered() {
        let mut reg = CapabilityRegistry::new(default_config());
        reg.register(make_peer("p1", [PeerCapability::Relay], 0));
        reg.register(make_peer("p2", [PeerCapability::GossipSub], 0));
        assert_eq!(reg.stats().total_registered, 2);
    }

    // ── 3. require_all — accept matching peer ─────────────────────────────────

    #[test]
    fn test_require_all_accept() {
        let config = CapabilityConfig {
            require_all: vec![PeerCapability::BitswapV1, PeerCapability::DHTKademlia],
            ..default_config()
        };
        let mut reg = CapabilityRegistry::new(config);
        let peer = make_peer(
            "p1",
            [PeerCapability::BitswapV1, PeerCapability::DHTKademlia],
            0,
        );
        assert!(reg.register(peer));
    }

    // ── 4. require_all — reject peer missing one capability ───────────────────

    #[test]
    fn test_require_all_reject() {
        let config = CapabilityConfig {
            require_all: vec![PeerCapability::BitswapV1, PeerCapability::DHTKademlia],
            ..default_config()
        };
        let mut reg = CapabilityRegistry::new(config);
        // Peer only has BitswapV1 — missing DHTKademlia.
        let peer = make_peer("p1", [PeerCapability::BitswapV1], 0);
        assert!(!reg.register(peer));
        assert_eq!(reg.peer_count(), 0);
        assert_eq!(reg.stats().peers_filtered_out, 1);
    }

    // ── 5. require_any — accept peer with one matching ────────────────────────

    #[test]
    fn test_require_any_accept() {
        let config = CapabilityConfig {
            require_any: vec![PeerCapability::Relay, PeerCapability::NatTraversal],
            ..default_config()
        };
        let mut reg = CapabilityRegistry::new(config);
        let peer = make_peer("p1", [PeerCapability::Relay], 0);
        assert!(reg.register(peer));
    }

    // ── 6. require_any — reject peer with none matching ───────────────────────

    #[test]
    fn test_require_any_reject() {
        let config = CapabilityConfig {
            require_any: vec![PeerCapability::Relay, PeerCapability::NatTraversal],
            ..default_config()
        };
        let mut reg = CapabilityRegistry::new(config);
        let peer = make_peer("p1", [PeerCapability::GossipSub], 0);
        assert!(!reg.register(peer));
        assert_eq!(reg.stats().peers_filtered_out, 1);
    }

    // ── 7. Empty requirements accept any peer ─────────────────────────────────

    #[test]
    fn test_empty_requirements_accept_all() {
        let mut reg = CapabilityRegistry::new(default_config());
        // No capabilities at all — should still pass with no requirements.
        let peer = make_peer("p1", [], 0);
        assert!(reg.register(peer));
    }

    // ── 8. peers_with_capability ──────────────────────────────────────────────

    #[test]
    fn test_peers_with_capability() {
        let mut reg = CapabilityRegistry::new(default_config());
        reg.register(make_peer("p1", [PeerCapability::BitswapV1], 0));
        reg.register(make_peer(
            "p2",
            [PeerCapability::BitswapV1, PeerCapability::Relay],
            0,
        ));
        reg.register(make_peer("p3", [PeerCapability::Relay], 0));

        let bitswap_peers = reg.peers_with_capability(&PeerCapability::BitswapV1);
        assert_eq!(bitswap_peers.len(), 2);

        let relay_peers = reg.peers_with_capability(&PeerCapability::Relay);
        assert_eq!(relay_peers.len(), 2);
    }

    // ── 9. peers_with_all ─────────────────────────────────────────────────────

    #[test]
    fn test_peers_with_all() {
        let mut reg = CapabilityRegistry::new(default_config());
        reg.register(make_peer(
            "p1",
            [PeerCapability::BitswapV1, PeerCapability::DHTKademlia],
            0,
        ));
        reg.register(make_peer("p2", [PeerCapability::BitswapV1], 0));
        reg.register(make_peer("p3", [PeerCapability::DHTKademlia], 0));

        let both = reg.peers_with_all(&[PeerCapability::BitswapV1, PeerCapability::DHTKademlia]);
        assert_eq!(both.len(), 1);
        assert_eq!(both[0].peer_id, "p1");
    }

    // ── 10. peers_with_any ────────────────────────────────────────────────────

    #[test]
    fn test_peers_with_any() {
        let mut reg = CapabilityRegistry::new(default_config());
        reg.register(make_peer("p1", [PeerCapability::BitswapV2], 0));
        reg.register(make_peer("p2", [PeerCapability::SemanticSearch], 0));
        reg.register(make_peer("p3", [PeerCapability::GraphSync], 0));

        let result =
            reg.peers_with_any(&[PeerCapability::BitswapV2, PeerCapability::SemanticSearch]);
        assert_eq!(result.len(), 2);
    }

    // ── 11. peers_with_any — empty slice returns all peers ────────────────────

    #[test]
    fn test_peers_with_any_empty_slice() {
        let mut reg = CapabilityRegistry::new(default_config());
        reg.register(make_peer("p1", [PeerCapability::Relay], 0));
        reg.register(make_peer("p2", [PeerCapability::GossipSub], 0));

        // Empty caps slice → vacuous truth → all peers returned.
        let result = reg.peers_with_any(&[]);
        assert_eq!(result.len(), 2);
    }

    // ── 12. peers_with_all — empty slice returns all peers ────────────────────

    #[test]
    fn test_peers_with_all_empty_slice() {
        let mut reg = CapabilityRegistry::new(default_config());
        reg.register(make_peer("p1", [PeerCapability::Relay], 0));
        reg.register(make_peer("p2", [PeerCapability::GossipSub], 0));

        // Empty caps slice → every peer has_all([]) trivially.
        let result = reg.peers_with_all(&[]);
        assert_eq!(result.len(), 2);
    }

    // ── 13. remove ────────────────────────────────────────────────────────────

    #[test]
    fn test_remove_existing() {
        let mut reg = CapabilityRegistry::new(default_config());
        reg.register(make_peer("p1", [PeerCapability::Relay], 0));
        assert_eq!(reg.peer_count(), 1);
        assert!(reg.remove("p1"));
        assert_eq!(reg.peer_count(), 0);
        assert_eq!(reg.stats().total_removed, 1);
    }

    // ── 14. remove non-existent returns false ─────────────────────────────────

    #[test]
    fn test_remove_nonexistent() {
        let mut reg = CapabilityRegistry::new(default_config());
        assert!(!reg.remove("nobody"));
        assert_eq!(reg.stats().total_removed, 0);
    }

    // ── 15. purge_expired ─────────────────────────────────────────────────────

    #[test]
    fn test_purge_expired() {
        let config = CapabilityConfig {
            ttl_ms: 1_000,
            ..default_config()
        };
        let mut reg = CapabilityRegistry::new(config);

        // Peer advertised at t=0, now is t=2000 → expired.
        reg.register(make_peer("old", [PeerCapability::Relay], 0));
        // Peer advertised at t=5000, now is t=2000 → still fresh.
        reg.register(make_peer("fresh", [PeerCapability::GossipSub], 5_000));

        let purged = reg.purge_expired(2_000);
        assert_eq!(purged, 1);
        assert_eq!(reg.peer_count(), 1);
        assert!(reg.get_peer("fresh").is_some());
        assert!(reg.get_peer("old").is_none());
    }

    // ── 16. purge_expired — nothing to purge ─────────────────────────────────

    #[test]
    fn test_purge_expired_none() {
        let mut reg = CapabilityRegistry::new(default_config());
        reg.register(make_peer("p1", [PeerCapability::Relay], 100_000));
        let purged = reg.purge_expired(1_000);
        assert_eq!(purged, 0);
        assert_eq!(reg.peer_count(), 1);
    }

    // ── 17. purge_expired — all peers expired ────────────────────────────────

    #[test]
    fn test_purge_expired_all() {
        let config = CapabilityConfig {
            ttl_ms: 500,
            ..default_config()
        };
        let mut reg = CapabilityRegistry::new(config);
        reg.register(make_peer("p1", [PeerCapability::Relay], 0));
        reg.register(make_peer("p2", [PeerCapability::GossipSub], 100));
        let purged = reg.purge_expired(10_000);
        assert_eq!(purged, 2);
        assert_eq!(reg.peer_count(), 0);
    }

    // ── 18. max_peers limit ───────────────────────────────────────────────────

    #[test]
    fn test_max_peers_limit() {
        let config = CapabilityConfig {
            max_peers: 2,
            ..default_config()
        };
        let mut reg = CapabilityRegistry::new(config);
        assert!(reg.register(make_peer("p1", [PeerCapability::Relay], 0)));
        assert!(reg.register(make_peer("p2", [PeerCapability::Relay], 0)));
        // Third peer should be rejected.
        assert!(!reg.register(make_peer("p3", [PeerCapability::Relay], 0)));
        assert_eq!(reg.peer_count(), 2);
        assert_eq!(reg.stats().peers_filtered_out, 1);
    }

    // ── 19. Re-registering the same peer_id replaces entry ───────────────────

    #[test]
    fn test_reregister_replaces_entry() {
        let mut reg = CapabilityRegistry::new(default_config());
        reg.register(make_peer("p1", [PeerCapability::BitswapV1], 100));
        // Re-register with updated capabilities.
        reg.register(make_peer(
            "p1",
            [PeerCapability::BitswapV2, PeerCapability::DHTKademlia],
            200,
        ));
        // Peer count should still be 1.
        assert_eq!(reg.peer_count(), 1);
        // The stored entry should reflect the update.
        let stored = reg.get_peer("p1").expect("peer should exist");
        assert!(stored.has(&PeerCapability::BitswapV2));
        assert!(!stored.has(&PeerCapability::BitswapV1));
    }

    // ── 20. capability distribution stats ────────────────────────────────────

    #[test]
    fn test_capability_distribution() {
        let mut reg = CapabilityRegistry::new(default_config());
        reg.register(make_peer(
            "p1",
            [PeerCapability::Relay, PeerCapability::GossipSub],
            0,
        ));
        reg.register(make_peer("p2", [PeerCapability::Relay], 0));

        let dist = &reg.stats().capability_distribution;
        assert_eq!(dist.get("relay").copied().unwrap_or(0), 2);
        assert_eq!(dist.get("gossipsub").copied().unwrap_or(0), 1);
    }

    // ── 21. Distribution decrements on remove ────────────────────────────────

    #[test]
    fn test_distribution_decrements_on_remove() {
        let mut reg = CapabilityRegistry::new(default_config());
        reg.register(make_peer("p1", [PeerCapability::Relay], 0));
        reg.register(make_peer("p2", [PeerCapability::Relay], 0));

        assert_eq!(
            reg.stats()
                .capability_distribution
                .get("relay")
                .copied()
                .unwrap_or(0),
            2
        );
        reg.remove("p1");
        assert_eq!(
            reg.stats()
                .capability_distribution
                .get("relay")
                .copied()
                .unwrap_or(0),
            1
        );
    }

    // ── 22. Custom capability ─────────────────────────────────────────────────

    #[test]
    fn test_custom_capability() {
        let mut reg = CapabilityRegistry::new(default_config());
        let custom = PeerCapability::Custom("my_protocol/1.0".to_string());
        reg.register(make_peer("p1", [custom.clone()], 0));

        let results = reg.peers_with_capability(&custom);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].peer_id, "p1");
    }

    // ── 23. capability_name static labels ────────────────────────────────────

    #[test]
    fn test_capability_name() {
        assert_eq!(
            CapabilityRegistry::capability_name(&PeerCapability::BitswapV1),
            "bitswap_v1"
        );
        assert_eq!(
            CapabilityRegistry::capability_name(&PeerCapability::DHTCorvus),
            "dht_corvus"
        );
        assert_eq!(
            CapabilityRegistry::capability_name(&PeerCapability::TensorLogic),
            "tensor_logic"
        );
        assert_eq!(
            CapabilityRegistry::capability_name(&PeerCapability::Custom("foo".to_string())),
            "custom"
        );
    }

    // ── 24. local_capabilities accessor ──────────────────────────────────────

    #[test]
    fn test_local_capabilities() {
        let config = CapabilityConfig {
            local_capabilities: vec![PeerCapability::BitswapV2, PeerCapability::SemanticSearch],
            ..default_config()
        };
        let reg = CapabilityRegistry::new(config);
        let local = reg.local_capabilities();
        assert_eq!(local.len(), 2);
        assert!(local.contains(&PeerCapability::BitswapV2));
        assert!(local.contains(&PeerCapability::SemanticSearch));
    }

    // ── 25. get_peer returns None for unknown peer ────────────────────────────

    #[test]
    fn test_get_peer_none() {
        let reg = CapabilityRegistry::new(default_config());
        assert!(reg.get_peer("nonexistent").is_none());
    }

    // ── 26. require_all + require_any combined ────────────────────────────────

    #[test]
    fn test_require_all_and_any_combined() {
        let config = CapabilityConfig {
            require_all: vec![PeerCapability::BitswapV1],
            require_any: vec![PeerCapability::Relay, PeerCapability::NatTraversal],
            ..default_config()
        };
        let mut reg = CapabilityRegistry::new(config);

        // Has BitswapV1 + Relay → should be accepted.
        let ok = make_peer("ok", [PeerCapability::BitswapV1, PeerCapability::Relay], 0);
        assert!(reg.register(ok));

        // Has BitswapV1 but neither Relay nor NatTraversal → rejected.
        let no_any = make_peer(
            "no_any",
            [PeerCapability::BitswapV1, PeerCapability::GossipSub],
            0,
        );
        assert!(!reg.register(no_any));

        // Has Relay but not BitswapV1 → rejected by require_all.
        let no_all = make_peer("no_all", [PeerCapability::Relay], 0);
        assert!(!reg.register(no_all));

        assert_eq!(reg.peer_count(), 1);
    }

    // ── 27. meets_requirements without registration ───────────────────────────

    #[test]
    fn test_meets_requirements_direct() {
        let config = CapabilityConfig {
            require_all: vec![PeerCapability::GossipSub],
            require_any: vec![],
            ..default_config()
        };
        let reg = CapabilityRegistry::new(config);

        let good = make_peer("g", [PeerCapability::GossipSub], 0);
        assert!(reg.meets_requirements(&good));

        let bad = make_peer("b", [PeerCapability::Relay], 0);
        assert!(!reg.meets_requirements(&bad));
    }

    // ── 28. TTL boundary — exactly at expiry ──────────────────────────────────

    #[test]
    fn test_ttl_boundary_exact() {
        // ttl_ms = 1000, advertised_at = 0, now = 1000 → expired (>=).
        let set = make_peer("p", [PeerCapability::Relay], 0);
        assert!(set.is_expired(1_000, 1_000));

        // now = 999 → still fresh.
        assert!(!set.is_expired(999, 1_000));
    }

    // ── 29. Distribution is correct after re-register ─────────────────────────

    #[test]
    fn test_distribution_after_reregister() {
        let mut reg = CapabilityRegistry::new(default_config());
        reg.register(make_peer("p1", [PeerCapability::Relay], 0));
        // Re-register p1 with a different capability set.
        reg.register(make_peer("p1", [PeerCapability::GossipSub], 100));

        let dist = &reg.stats().capability_distribution;
        // relay should be 0 (old entry removed), gossipsub should be 1.
        assert_eq!(dist.get("relay").copied().unwrap_or(0), 0);
        assert_eq!(dist.get("gossipsub").copied().unwrap_or(0), 1);
    }

    // ── 30. Overlap query — peers_with_all vs peers_with_any ─────────────────

    #[test]
    fn test_overlap_queries() {
        let mut reg = CapabilityRegistry::new(default_config());
        reg.register(make_peer(
            "p1",
            [PeerCapability::BitswapV1, PeerCapability::BitswapV2],
            0,
        ));
        reg.register(make_peer("p2", [PeerCapability::BitswapV1], 0));
        reg.register(make_peer("p3", [PeerCapability::BitswapV2], 0));

        let both = reg.peers_with_all(&[PeerCapability::BitswapV1, PeerCapability::BitswapV2]);
        assert_eq!(both.len(), 1);

        let either = reg.peers_with_any(&[PeerCapability::BitswapV1, PeerCapability::BitswapV2]);
        assert_eq!(either.len(), 3);
    }
}
