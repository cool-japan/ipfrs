//! Peer Capability Registry
//!
//! Tracks what capabilities each peer advertises so that queries can be routed
//! to nodes that are able to serve them (vector search, TensorLogic,
//! gradient sync, block storage, content routing, etc.).
//!
//! # Two Capability Systems
//!
//! This module contains two related but distinct capability tracking systems:
//!
//! 1. **[`NodeCapability`] / [`NodeCapabilityRegistry`]** — legacy node-level capabilities
//!    for vector search, tensor logic, gradient sync, block storage, and content
//!    routing.  These are keyed by a string name and carry structured payloads.
//!
//! 2. **[`Capability`] / [`PeerCapabilityRegistry`]** — protocol-level capabilities for
//!    Bitswap, TensorSwap, Kademlia DHT, GossipSub, content addressing, and
//!    extensible custom capabilities.  Advertisements are tick-based and support
//!    TTL-based expiry.

use std::collections::HashMap;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

// ── NodeCapability (legacy) ───────────────────────────────────────────────────

/// A single node-level capability that a peer may advertise.
///
/// This is the original capability type tracking resource capacities like
/// vector-search index sizes, rule-engine rule counts, model sizes, etc.
/// For protocol-level capabilities (Bitswap, Kademlia, GossipSub, …) see
/// [`Capability`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeCapability {
    /// HNSW / ANN vector index with declared size and dimensionality.
    VectorSearch {
        /// Number of vectors stored in the index.
        index_size: u64,
        /// Embedding dimensionality.
        dimensions: u32,
    },
    /// TensorLogic rule engine with a declared rule count.
    TensorLogic {
        /// Number of compiled rules in the engine.
        rule_count: u64,
    },
    /// Gradient synchronisation participant with declared model footprint.
    GradientSync {
        /// Serialised model size in bytes.
        model_size_bytes: u64,
    },
    /// Raw block / object storage with declared occupancy.
    BlockStorage {
        /// Bytes currently stored by this peer.
        stored_bytes: u64,
    },
    /// Peer participates in content routing (DHT provider announcements).
    ContentRouting,
}

impl NodeCapability {
    /// Returns a stable, lowercase ASCII identifier for the capability.
    ///
    /// These names are used as keys in the capability histogram and when
    /// looking up capabilities by name.
    pub fn name(&self) -> &str {
        match self {
            NodeCapability::VectorSearch { .. } => "vector_search",
            NodeCapability::TensorLogic { .. } => "tensor_logic",
            NodeCapability::GradientSync { .. } => "gradient_sync",
            NodeCapability::BlockStorage { .. } => "block_storage",
            NodeCapability::ContentRouting => "content_routing",
        }
    }
}

// ── NodeCapabilities ──────────────────────────────────────────────────────────

/// The full node-level capability advertisement sent by a single peer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeCapabilities {
    /// Unique peer identifier (string form of the libp2p `PeerId`).
    pub peer_id: String,
    /// Ordered list of capabilities the peer supports.
    pub capabilities: Vec<NodeCapability>,
    /// IPFRS protocol version string, e.g. `"0.2.0"`.
    pub protocol_version: String,
    /// Unix epoch timestamp in milliseconds at which the advertisement was
    /// created / last refreshed.
    pub announced_at: u64,
    /// Time-to-live in seconds after which this record is considered stale.
    pub ttl_secs: u64,
}

impl NodeCapabilities {
    /// Default TTL used when none is provided explicitly.
    pub const DEFAULT_TTL_SECS: u64 = 300;

    /// Create a fresh `NodeCapabilities` record with the current system time.
    ///
    /// `announced_at` is set to the current Unix time in milliseconds.
    /// `ttl_secs` defaults to [`Self::DEFAULT_TTL_SECS`] (300 s).
    pub fn new(peer_id: &str, capabilities: Vec<NodeCapability>, protocol_version: &str) -> Self {
        let announced_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            peer_id: peer_id.to_owned(),
            capabilities,
            protocol_version: protocol_version.to_owned(),
            announced_at,
            ttl_secs: Self::DEFAULT_TTL_SECS,
        }
    }

    /// Returns `true` if the peer supports the capability identified by `name`.
    pub fn has_capability(&self, name: &str) -> bool {
        self.capabilities.iter().any(|c| c.name() == name)
    }

    /// Returns `true` if the record has expired relative to `now_ms`.
    ///
    /// A record is expired when `now_ms > announced_at + ttl_secs * 1000`.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        now_ms
            > self
                .announced_at
                .saturating_add(self.ttl_secs.saturating_mul(1_000))
    }

    /// Returns a reference to the first `NodeCapability` whose name matches `name`,
    /// or `None` if no such capability is present.
    pub fn get_capability(&self, name: &str) -> Option<&NodeCapability> {
        self.capabilities.iter().find(|c| c.name() == name)
    }
}

// ── NodeCapabilityRegistry ────────────────────────────────────────────────────

/// Thread-safe registry that maps peer IDs to their node-level capability
/// advertisements.
///
/// # Example
///
/// ```rust
/// use ipfrs_network::capability_registry::{
///     NodeCapability, NodeCapabilities, NodeCapabilityRegistry,
/// };
///
/// let registry = NodeCapabilityRegistry::default();
///
/// let caps = NodeCapabilities::new(
///     "peer-1",
///     vec![NodeCapability::ContentRouting],
///     "0.2.0",
/// );
/// registry.register(caps);
///
/// let found = registry.find_by_capability("content_routing");
/// assert_eq!(found.len(), 1);
/// ```
#[derive(Debug, Default)]
pub struct NodeCapabilityRegistry {
    entries: RwLock<HashMap<String, NodeCapabilities>>,
}

impl NodeCapabilityRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register or replace the capability advertisement for the peer described
    /// by `caps.peer_id`.
    pub fn register(&self, caps: NodeCapabilities) {
        match self.entries.write() {
            Ok(mut guard) => {
                guard.insert(caps.peer_id.clone(), caps);
            }
            Err(poisoned) => {
                // Recover the lock and continue — a panicking writer should not
                // permanently block the registry.
                let mut guard = poisoned.into_inner();
                guard.insert(caps.peer_id.clone(), caps);
            }
        }
    }

    /// Remove the advertisement for `peer_id` from the registry.
    ///
    /// This is a no-op when the peer is not registered.
    pub fn unregister(&self, peer_id: &str) {
        match self.entries.write() {
            Ok(mut guard) => {
                guard.remove(peer_id);
            }
            Err(poisoned) => {
                let mut guard = poisoned.into_inner();
                guard.remove(peer_id);
            }
        }
    }

    /// Return a clone of the `NodeCapabilities` for `peer_id`, or `None` if the
    /// peer is not registered.
    pub fn get(&self, peer_id: &str) -> Option<NodeCapabilities> {
        match self.entries.read() {
            Ok(guard) => guard.get(peer_id).cloned(),
            Err(poisoned) => poisoned.into_inner().get(peer_id).cloned(),
        }
    }

    /// Return clones of all non-expired peer records that advertise the
    /// capability identified by `name`.
    ///
    /// Expiry is evaluated relative to `now_ms` obtained from the system clock
    /// at the time of the call.
    pub fn find_by_capability(&self, name: &str) -> Vec<NodeCapabilities> {
        let now_ms = current_time_ms();
        match self.entries.read() {
            Ok(guard) => guard
                .values()
                .filter(|nc| !nc.is_expired(now_ms) && nc.has_capability(name))
                .cloned()
                .collect(),
            Err(poisoned) => poisoned
                .into_inner()
                .values()
                .filter(|nc| !nc.is_expired(now_ms) && nc.has_capability(name))
                .cloned()
                .collect(),
        }
    }

    /// Remove all expired entries from the registry and return the number of
    /// entries that were evicted.
    pub fn evict_expired(&self, now_ms: u64) -> usize {
        match self.entries.write() {
            Ok(mut guard) => {
                let before = guard.len();
                guard.retain(|_, nc| !nc.is_expired(now_ms));
                before - guard.len()
            }
            Err(poisoned) => {
                let mut guard = poisoned.into_inner();
                let before = guard.len();
                guard.retain(|_, nc| !nc.is_expired(now_ms));
                before - guard.len()
            }
        }
    }

    /// Return the number of peers currently registered (including expired ones
    /// that have not yet been evicted).
    pub fn peer_count(&self) -> usize {
        match self.entries.read() {
            Ok(guard) => guard.len(),
            Err(poisoned) => poisoned.into_inner().len(),
        }
    }

    /// Return a histogram mapping each capability name to the number of
    /// currently-registered peers (including stale ones) that advertise it.
    pub fn capability_histogram(&self) -> HashMap<String, usize> {
        match self.entries.read() {
            Ok(guard) => build_histogram(guard.values()),
            Err(poisoned) => build_histogram(poisoned.into_inner().values()),
        }
    }
}

// ── Capability (protocol-level) ───────────────────────────────────────────────

/// Protocol-level capability that a peer may advertise.
///
/// These represent the networking protocols and features a peer supports,
/// enabling capability-based peer selection for routing and protocol
/// negotiation.  For resource-level capabilities (vector search, block storage,
/// etc.) see [`NodeCapability`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability {
    /// Bitswap block-exchange protocol.
    Bitswap {
        /// Bitswap protocol version.
        version: u8,
    },
    /// TensorSwap tensor-exchange protocol.
    TensorSwap {
        /// TensorSwap protocol version.
        version: u8,
    },
    /// Kademlia DHT participation.
    Kademlia,
    /// GossipSub pub/sub with a list of subscribed topic names.
    GossipSub {
        /// Topics the peer has subscribed to.
        topics: Vec<String>,
    },
    /// CID-based content addressing (IPFS-compatible object storage).
    ContentAddressing,
    /// Extensible custom capability for application-defined features.
    Custom {
        /// Capability identifier (ASCII, lowercase recommended).
        name: String,
        /// Capability version.
        version: u8,
    },
}

// ── CapabilityAdvertisement ───────────────────────────────────────────────────

/// A time-bounded advertisement of the [`Capability`] set that a peer supports.
///
/// Advertisements are valid while `current_tick < expires_at_tick`.  Once
/// expired they should be ignored for routing decisions, and can be removed by
/// calling [`PeerCapabilityRegistry::evict_expired`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityAdvertisement {
    /// Unique peer identifier.
    pub peer_id: String,
    /// Capabilities the peer currently advertises.
    pub capabilities: Vec<Capability>,
    /// Tick at which this advertisement was created.
    pub advertised_at_tick: u64,
    /// Tick after which this advertisement is considered expired.
    pub expires_at_tick: u64,
}

impl CapabilityAdvertisement {
    /// Returns `true` if this advertisement is still valid at `current_tick`.
    ///
    /// An advertisement is valid while `current_tick < expires_at_tick`.
    #[inline]
    pub fn is_valid(&self, current_tick: u64) -> bool {
        current_tick < self.expires_at_tick
    }

    /// Returns `true` if this advertisement contains `cap` (exact match via
    /// [`PartialEq`]).
    #[inline]
    pub fn has_capability(&self, cap: &Capability) -> bool {
        self.capabilities.contains(cap)
    }
}

// ── CapabilityRegistryStats ───────────────────────────────────────────────────

/// Statistics snapshot for a [`PeerCapabilityRegistry`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityRegistryStats {
    /// Total number of peers in the registry (including expired ones not yet
    /// evicted).
    pub total_peers: usize,
    /// Number of peers whose advertisement is still valid at the query tick.
    pub active_peers: usize,
    /// Sum of the capability-list lengths across all advertisements (active
    /// and expired).
    pub total_capabilities_registered: usize,
    /// Number of advertisements that have already expired at the query tick.
    pub expired_count: usize,
}

// ── PeerCapabilityRegistry ────────────────────────────────────────────────────

/// Tick-based registry tracking which protocol-level [`Capability`] set each
/// peer advertises.
///
/// This is the primary data structure for capability-based peer selection.
/// Advertisements carry a TTL expressed in ticks; callers must supply the
/// current tick to all query operations so that expired entries are correctly
/// filtered out without requiring a background eviction task.
///
/// # Example
///
/// ```rust
/// use ipfrs_network::capability_registry::{
///     Capability, PeerCapabilityRegistry,
/// };
///
/// let mut registry = PeerCapabilityRegistry::new();
/// registry.advertise(
///     "peer-a".to_string(),
///     vec![Capability::Kademlia, Capability::Bitswap { version: 1 }],
///     /*current_tick=*/ 0,
///     /*ttl_ticks=*/ 100,
/// );
///
/// let peers = registry.peers_with_capability(&Capability::Kademlia, 0);
/// assert_eq!(peers, vec!["peer-a"]);
/// ```
#[derive(Debug, Default)]
pub struct PeerCapabilityRegistry {
    /// Peer-ID → advertisement map.
    advertisements: HashMap<String, CapabilityAdvertisement>,
}

impl PeerCapabilityRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace the advertisement for `peer_id`.
    ///
    /// The advertisement expires at tick `current_tick + ttl_ticks`.
    pub fn advertise(
        &mut self,
        peer_id: String,
        capabilities: Vec<Capability>,
        current_tick: u64,
        ttl_ticks: u64,
    ) {
        let advert = CapabilityAdvertisement {
            peer_id: peer_id.clone(),
            capabilities,
            advertised_at_tick: current_tick,
            expires_at_tick: current_tick.saturating_add(ttl_ticks),
        };
        self.advertisements.insert(peer_id, advert);
    }

    /// Return a sorted list of peer IDs whose valid advertisements include `cap`.
    ///
    /// Only advertisements that pass [`CapabilityAdvertisement::is_valid`] at
    /// `current_tick` are considered.  The result is sorted alphabetically.
    pub fn peers_with_capability(&self, cap: &Capability, current_tick: u64) -> Vec<&str> {
        let mut peers: Vec<&str> = self
            .advertisements
            .values()
            .filter(|advert| advert.is_valid(current_tick) && advert.has_capability(cap))
            .map(|advert| advert.peer_id.as_str())
            .collect();
        peers.sort_unstable();
        peers
    }

    /// Return the capability slice for `peer_id` if its advertisement is valid
    /// at `current_tick`, or `None` otherwise.
    ///
    /// Returns `None` when:
    /// - the peer is not in the registry, or
    /// - the peer's advertisement has expired.
    pub fn peer_capabilities(&self, peer_id: &str, current_tick: u64) -> Option<&[Capability]> {
        self.advertisements
            .get(peer_id)
            .filter(|advert| advert.is_valid(current_tick))
            .map(|advert| advert.capabilities.as_slice())
    }

    /// Remove the advertisement for `peer_id`.
    ///
    /// Returns `true` if the peer was found and removed, `false` if the peer
    /// was not registered.
    pub fn remove_peer(&mut self, peer_id: &str) -> bool {
        self.advertisements.remove(peer_id).is_some()
    }

    /// Remove all expired advertisements from the registry.
    ///
    /// Returns the number of advertisements that were removed.
    pub fn evict_expired(&mut self, current_tick: u64) -> usize {
        let before = self.advertisements.len();
        self.advertisements
            .retain(|_, advert| advert.is_valid(current_tick));
        before - self.advertisements.len()
    }

    /// Return a [`CapabilityRegistryStats`] snapshot evaluated at `current_tick`.
    pub fn stats(&self, current_tick: u64) -> CapabilityRegistryStats {
        let total_peers = self.advertisements.len();
        let mut active_peers = 0usize;
        let mut expired_count = 0usize;
        let mut total_capabilities_registered = 0usize;

        for advert in self.advertisements.values() {
            total_capabilities_registered =
                total_capabilities_registered.saturating_add(advert.capabilities.len());
            if advert.is_valid(current_tick) {
                active_peers = active_peers.saturating_add(1);
            } else {
                expired_count = expired_count.saturating_add(1);
            }
        }

        CapabilityRegistryStats {
            total_peers,
            active_peers,
            total_capabilities_registered,
            expired_count,
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn current_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn build_histogram<'a, I>(iter: I) -> HashMap<String, usize>
where
    I: Iterator<Item = &'a NodeCapabilities>,
{
    let mut map: HashMap<String, usize> = HashMap::new();
    for nc in iter {
        for cap in &nc.capabilities {
            *map.entry(cap.name().to_owned()).or_insert(0) += 1;
        }
    }
    map
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─────────────────────────────────────────────────────────────────────────
    // NodeCapabilityRegistry legacy tests
    // ─────────────────────────────────────────────────────────────────────────

    fn make_node_caps(peer_id: &str, caps: Vec<NodeCapability>) -> NodeCapabilities {
        NodeCapabilities::new(peer_id, caps, "0.2.0")
    }

    /// Build a NodeCapabilities whose `announced_at` is in the past so it is
    /// already expired.
    fn make_expired_node_caps(peer_id: &str, caps: Vec<NodeCapability>) -> NodeCapabilities {
        let mut nc = make_node_caps(peer_id, caps);
        nc.announced_at = 0;
        nc.ttl_secs = 1;
        nc
    }

    // ── test 1: register and retrieve ────────────────────────────────────────

    #[test]
    fn test_register_and_get() {
        let registry = NodeCapabilityRegistry::new();
        let caps = make_node_caps("peer-a", vec![NodeCapability::ContentRouting]);
        registry.register(caps.clone());

        let retrieved = registry.get("peer-a").expect("should be present");
        assert_eq!(retrieved.peer_id, "peer-a");
        assert_eq!(retrieved.protocol_version, "0.2.0");
    }

    // ── test 2: get returns None for unknown peer ─────────────────────────────

    #[test]
    fn test_get_unknown_peer_returns_none() {
        let registry = NodeCapabilityRegistry::new();
        assert!(registry.get("nobody").is_none());
    }

    // ── test 3: has_capability true ───────────────────────────────────────────

    #[test]
    fn test_has_capability_true() {
        let caps = make_node_caps(
            "peer-b",
            vec![NodeCapability::VectorSearch {
                index_size: 1000,
                dimensions: 128,
            }],
        );
        assert!(caps.has_capability("vector_search"));
    }

    // ── test 4: has_capability false ──────────────────────────────────────────

    #[test]
    fn test_has_capability_false() {
        let caps = make_node_caps("peer-c", vec![NodeCapability::ContentRouting]);
        assert!(!caps.has_capability("vector_search"));
    }

    // ── test 5: find_by_capability returns correct peers ─────────────────────

    #[test]
    fn test_find_by_capability_correct_peers() {
        let registry = NodeCapabilityRegistry::new();

        registry.register(make_node_caps(
            "peer-1",
            vec![NodeCapability::BlockStorage { stored_bytes: 512 }],
        ));
        registry.register(make_node_caps(
            "peer-2",
            vec![
                NodeCapability::BlockStorage { stored_bytes: 1024 },
                NodeCapability::ContentRouting,
            ],
        ));
        registry.register(make_node_caps(
            "peer-3",
            vec![NodeCapability::ContentRouting],
        ));

        let found = registry.find_by_capability("block_storage");
        let ids: Vec<&str> = found.iter().map(|nc| nc.peer_id.as_str()).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"peer-1"));
        assert!(ids.contains(&"peer-2"));
    }

    // ── test 6: find_by_capability excludes expired peers ────────────────────

    #[test]
    fn test_find_by_capability_excludes_expired() {
        let registry = NodeCapabilityRegistry::new();

        registry.register(make_node_caps(
            "fresh",
            vec![NodeCapability::ContentRouting],
        ));
        registry.register(make_expired_node_caps(
            "stale",
            vec![NodeCapability::ContentRouting],
        ));

        let found = registry.find_by_capability("content_routing");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].peer_id, "fresh");
    }

    // ── test 7: is_expired true ───────────────────────────────────────────────

    #[test]
    fn test_is_expired_true() {
        let mut nc = make_node_caps("x", vec![]);
        nc.announced_at = 1_000_000; // 1 000 seconds in ms
        nc.ttl_secs = 60;
        let now_ms = 1_000_000 + 70_000;
        assert!(nc.is_expired(now_ms));
    }

    // ── test 8: is_expired false ──────────────────────────────────────────────

    #[test]
    fn test_is_expired_false() {
        let mut nc = make_node_caps("y", vec![]);
        nc.announced_at = 1_000_000;
        nc.ttl_secs = 300;
        let now_ms = 1_000_000 + 10_000;
        assert!(!nc.is_expired(now_ms));
    }

    // ── test 9: evict_expired removes stale entries ───────────────────────────

    #[test]
    fn test_evict_expired_removes_stale() {
        let registry = NodeCapabilityRegistry::new();

        registry.register(make_node_caps(
            "alive",
            vec![NodeCapability::ContentRouting],
        ));
        registry.register(make_expired_node_caps(
            "dead-1",
            vec![NodeCapability::ContentRouting],
        ));
        registry.register(make_expired_node_caps(
            "dead-2",
            vec![NodeCapability::ContentRouting],
        ));

        let evicted = registry.evict_expired(current_time_ms());
        assert_eq!(evicted, 2);
        assert_eq!(registry.peer_count(), 1);
    }

    // ── test 10: evict_expired keeps fresh entries ────────────────────────────

    #[test]
    fn test_evict_expired_keeps_fresh() {
        let registry = NodeCapabilityRegistry::new();

        registry.register(make_node_caps(
            "fresh-a",
            vec![NodeCapability::TensorLogic { rule_count: 10 }],
        ));
        registry.register(make_node_caps(
            "fresh-b",
            vec![NodeCapability::TensorLogic { rule_count: 20 }],
        ));

        let evicted = registry.evict_expired(current_time_ms());
        assert_eq!(evicted, 0);
        assert_eq!(registry.peer_count(), 2);
    }

    // ── test 11: capability_histogram counts correctly ────────────────────────

    #[test]
    fn test_capability_histogram() {
        let registry = NodeCapabilityRegistry::new();

        registry.register(make_node_caps(
            "h1",
            vec![
                NodeCapability::ContentRouting,
                NodeCapability::BlockStorage { stored_bytes: 0 },
            ],
        ));
        registry.register(make_node_caps("h2", vec![NodeCapability::ContentRouting]));
        registry.register(make_node_caps(
            "h3",
            vec![NodeCapability::VectorSearch {
                index_size: 50,
                dimensions: 64,
            }],
        ));

        let hist = registry.capability_histogram();
        assert_eq!(*hist.get("content_routing").unwrap_or(&0), 2);
        assert_eq!(*hist.get("block_storage").unwrap_or(&0), 1);
        assert_eq!(*hist.get("vector_search").unwrap_or(&0), 1);
        assert_eq!(*hist.get("tensor_logic").unwrap_or(&0), 0);
    }

    // ── test 12: get_capability returns correct variant ───────────────────────

    #[test]
    fn test_get_capability_returns_correct_variant() {
        let caps = make_node_caps(
            "peer-v",
            vec![
                NodeCapability::VectorSearch {
                    index_size: 2048,
                    dimensions: 256,
                },
                NodeCapability::ContentRouting,
            ],
        );

        let found = caps
            .get_capability("vector_search")
            .expect("should be found");
        match found {
            NodeCapability::VectorSearch {
                index_size,
                dimensions,
            } => {
                assert_eq!(*index_size, 2048);
                assert_eq!(*dimensions, 256);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    // ── test 13: get_capability returns None for absent name ──────────────────

    #[test]
    fn test_get_capability_absent() {
        let caps = make_node_caps("peer-z", vec![NodeCapability::ContentRouting]);
        assert!(caps.get_capability("gradient_sync").is_none());
    }

    // ── test 14: unregister removes peer ──────────────────────────────────────

    #[test]
    fn test_unregister_removes_peer() {
        let registry = NodeCapabilityRegistry::new();
        registry.register(make_node_caps(
            "to-remove",
            vec![NodeCapability::ContentRouting],
        ));
        assert_eq!(registry.peer_count(), 1);

        registry.unregister("to-remove");
        assert_eq!(registry.peer_count(), 0);
        assert!(registry.get("to-remove").is_none());
    }

    // ── test 15: unregister on unknown peer is no-op ──────────────────────────

    #[test]
    fn test_unregister_unknown_is_noop() {
        let registry = NodeCapabilityRegistry::new();
        registry.unregister("ghost"); // must not panic
        assert_eq!(registry.peer_count(), 0);
    }

    // ── test 16: re-register replaces existing entry ──────────────────────────

    #[test]
    fn test_register_replaces_existing() {
        let registry = NodeCapabilityRegistry::new();

        registry.register(make_node_caps(
            "peer-r",
            vec![NodeCapability::ContentRouting],
        ));
        registry.register(make_node_caps(
            "peer-r",
            vec![NodeCapability::BlockStorage { stored_bytes: 9999 }],
        ));

        assert_eq!(registry.peer_count(), 1);
        let nc = registry.get("peer-r").expect("peer-r should be present");
        assert!(!nc.has_capability("content_routing"));
        assert!(nc.has_capability("block_storage"));
    }

    // ── test 17: gradient_sync capability name ────────────────────────────────

    #[test]
    fn test_gradient_sync_name() {
        let cap = NodeCapability::GradientSync {
            model_size_bytes: 1_000_000,
        };
        assert_eq!(cap.name(), "gradient_sync");
    }

    // ── test 18: serde round-trip ─────────────────────────────────────────────

    #[test]
    fn test_serde_round_trip() {
        let original = make_node_caps(
            "peer-serde",
            vec![
                NodeCapability::TensorLogic { rule_count: 42 },
                NodeCapability::GradientSync {
                    model_size_bytes: 1024,
                },
            ],
        );

        let json = serde_json::to_string(&original).expect("serialise");
        let decoded: NodeCapabilities = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(original, decoded);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PeerCapabilityRegistry tests
    // ─────────────────────────────────────────────────────────────────────────

    // ── helper ────────────────────────────────────────────────────────────────

    fn make_registry() -> PeerCapabilityRegistry {
        PeerCapabilityRegistry::new()
    }

    // ── test 19: advertise creates entry ─────────────────────────────────────

    #[test]
    fn test_pcr_advertise_creates_entry() {
        let mut reg = make_registry();
        reg.advertise("peer-a".to_string(), vec![Capability::Kademlia], 0, 100);
        let caps = reg
            .peer_capabilities("peer-a", 0)
            .expect("should have capabilities");
        assert_eq!(caps, &[Capability::Kademlia]);
    }

    // ── test 20: advertise overwrites existing ────────────────────────────────

    #[test]
    fn test_pcr_advertise_overwrites_existing() {
        let mut reg = make_registry();
        reg.advertise("peer-b".to_string(), vec![Capability::Kademlia], 0, 100);
        reg.advertise(
            "peer-b".to_string(),
            vec![Capability::Bitswap { version: 2 }],
            10,
            200,
        );
        let caps = reg
            .peer_capabilities("peer-b", 10)
            .expect("should have capabilities");
        assert_eq!(caps.len(), 1);
        assert!(caps.contains(&Capability::Bitswap { version: 2 }));
        assert!(!caps.contains(&Capability::Kademlia));
    }

    // ── test 21: peers_with_capability returns correct sorted peers ───────────

    #[test]
    fn test_pcr_peers_with_capability_sorted() {
        let mut reg = make_registry();
        reg.advertise("peer-c".to_string(), vec![Capability::Kademlia], 0, 100);
        reg.advertise("peer-a".to_string(), vec![Capability::Kademlia], 0, 100);
        reg.advertise("peer-b".to_string(), vec![Capability::Kademlia], 0, 100);
        reg.advertise(
            "peer-d".to_string(),
            vec![Capability::ContentAddressing],
            0,
            100,
        );

        let peers = reg.peers_with_capability(&Capability::Kademlia, 0);
        assert_eq!(peers, vec!["peer-a", "peer-b", "peer-c"]);
    }

    // ── test 22: peers_with_capability ignores expired ────────────────────────

    #[test]
    fn test_pcr_peers_with_capability_ignores_expired() {
        let mut reg = make_registry();
        reg.advertise("fresh".to_string(), vec![Capability::Kademlia], 0, 100);
        // Expires at tick 10, query at tick 20 → expired
        reg.advertise("stale".to_string(), vec![Capability::Kademlia], 0, 10);

        let peers = reg.peers_with_capability(&Capability::Kademlia, 20);
        assert_eq!(peers, vec!["fresh"]);
    }

    // ── test 23: peer_capabilities returns None for expired ───────────────────

    #[test]
    fn test_pcr_peer_capabilities_none_for_expired() {
        let mut reg = make_registry();
        // expires_at_tick = 0 + 5 = 5; query at tick 10 → expired
        reg.advertise("peer-exp".to_string(), vec![Capability::Kademlia], 0, 5);
        assert!(reg.peer_capabilities("peer-exp", 10).is_none());
    }

    // ── test 24: peer_capabilities returns Some for valid ─────────────────────

    #[test]
    fn test_pcr_peer_capabilities_some_for_valid() {
        let mut reg = make_registry();
        reg.advertise(
            "peer-ok".to_string(),
            vec![Capability::ContentAddressing],
            0,
            100,
        );
        let caps = reg
            .peer_capabilities("peer-ok", 50)
            .expect("should be valid");
        assert_eq!(caps, &[Capability::ContentAddressing]);
    }

    // ── test 25: peer_capabilities returns None for unknown peer ─────────────

    #[test]
    fn test_pcr_peer_capabilities_none_for_unknown() {
        let reg = make_registry();
        assert!(reg.peer_capabilities("ghost", 0).is_none());
    }

    // ── test 26: remove_peer returns true when peer exists ───────────────────

    #[test]
    fn test_pcr_remove_peer_returns_true() {
        let mut reg = make_registry();
        reg.advertise("peer-del".to_string(), vec![Capability::Kademlia], 0, 100);
        assert!(reg.remove_peer("peer-del"));
        assert!(reg.peer_capabilities("peer-del", 0).is_none());
    }

    // ── test 27: remove_peer returns false when peer absent ──────────────────

    #[test]
    fn test_pcr_remove_peer_returns_false() {
        let mut reg = make_registry();
        assert!(!reg.remove_peer("nobody"));
    }

    // ── test 28: evict_expired removes expired entries ────────────────────────

    #[test]
    fn test_pcr_evict_expired_removes_expired() {
        let mut reg = make_registry();
        reg.advertise("alive".to_string(), vec![Capability::Kademlia], 0, 100);
        reg.advertise("dead-1".to_string(), vec![Capability::Kademlia], 0, 5);
        reg.advertise("dead-2".to_string(), vec![Capability::Kademlia], 0, 3);

        let evicted = reg.evict_expired(10);
        assert_eq!(evicted, 2);
        // "alive" is still valid at tick 10 (expires at 100)
        assert!(reg.peer_capabilities("alive", 10).is_some());
        assert!(reg.peer_capabilities("dead-1", 10).is_none());
        assert!(reg.peer_capabilities("dead-2", 10).is_none());
    }

    // ── test 29: evict_expired returns 0 when nothing expired ────────────────

    #[test]
    fn test_pcr_evict_expired_zero_when_none_expired() {
        let mut reg = make_registry();
        reg.advertise("p1".to_string(), vec![Capability::Kademlia], 0, 100);
        reg.advertise("p2".to_string(), vec![Capability::Kademlia], 0, 200);
        assert_eq!(reg.evict_expired(0), 0);
    }

    // ── test 30: stats active_peers vs total_peers ────────────────────────────

    #[test]
    fn test_pcr_stats_active_vs_total() {
        let mut reg = make_registry();
        reg.advertise("active-1".to_string(), vec![Capability::Kademlia], 0, 100);
        reg.advertise(
            "active-2".to_string(),
            vec![Capability::ContentAddressing],
            0,
            100,
        );
        // expires at tick 5, queried at tick 50 → expired
        reg.advertise("expired-1".to_string(), vec![Capability::Kademlia], 0, 5);

        let stats = reg.stats(50);
        assert_eq!(stats.total_peers, 3);
        assert_eq!(stats.active_peers, 2);
        assert_eq!(stats.expired_count, 1);
        assert_eq!(stats.total_capabilities_registered, 3); // 1 + 1 + 1
    }

    // ── test 31: GossipSub topic matching ────────────────────────────────────

    #[test]
    fn test_pcr_gossipsub_topic_matching_exact() {
        let mut reg = make_registry();
        let topics_a = vec!["news".to_string(), "sports".to_string()];
        let topics_b = vec!["weather".to_string()];

        reg.advertise(
            "peer-gs-a".to_string(),
            vec![Capability::GossipSub {
                topics: topics_a.clone(),
            }],
            0,
            100,
        );
        reg.advertise(
            "peer-gs-b".to_string(),
            vec![Capability::GossipSub {
                topics: topics_b.clone(),
            }],
            0,
            100,
        );

        // Exact match: must have identical topic list
        let peers_news = reg.peers_with_capability(
            &Capability::GossipSub {
                topics: topics_a.clone(),
            },
            0,
        );
        assert_eq!(peers_news, vec!["peer-gs-a"]);

        let peers_weather = reg.peers_with_capability(
            &Capability::GossipSub {
                topics: topics_b.clone(),
            },
            0,
        );
        assert_eq!(peers_weather, vec!["peer-gs-b"]);
    }

    // ── test 32: Custom capability matching ───────────────────────────────────

    #[test]
    fn test_pcr_custom_capability_matching() {
        let mut reg = make_registry();
        reg.advertise(
            "peer-custom".to_string(),
            vec![Capability::Custom {
                name: "my-proto".to_string(),
                version: 3,
            }],
            0,
            100,
        );

        // Exact match succeeds
        let peers = reg.peers_with_capability(
            &Capability::Custom {
                name: "my-proto".to_string(),
                version: 3,
            },
            0,
        );
        assert_eq!(peers, vec!["peer-custom"]);

        // Different version — no match
        let peers_v2 = reg.peers_with_capability(
            &Capability::Custom {
                name: "my-proto".to_string(),
                version: 2,
            },
            0,
        );
        assert!(peers_v2.is_empty());

        // Different name — no match
        let peers_other = reg.peers_with_capability(
            &Capability::Custom {
                name: "other-proto".to_string(),
                version: 3,
            },
            0,
        );
        assert!(peers_other.is_empty());
    }

    // ── test 33: TensorSwap version matching ──────────────────────────────────

    #[test]
    fn test_pcr_tensorswap_version_matching() {
        let mut reg = make_registry();
        reg.advertise(
            "peer-ts-v1".to_string(),
            vec![Capability::TensorSwap { version: 1 }],
            0,
            100,
        );
        reg.advertise(
            "peer-ts-v2".to_string(),
            vec![Capability::TensorSwap { version: 2 }],
            0,
            100,
        );

        let v1_peers = reg.peers_with_capability(&Capability::TensorSwap { version: 1 }, 0);
        assert_eq!(v1_peers, vec!["peer-ts-v1"]);

        let v2_peers = reg.peers_with_capability(&Capability::TensorSwap { version: 2 }, 0);
        assert_eq!(v2_peers, vec!["peer-ts-v2"]);
    }

    // ── test 34: Bitswap version exact match ─────────────────────────────────

    #[test]
    fn test_pcr_bitswap_exact_version() {
        let mut reg = make_registry();
        reg.advertise(
            "peer-bs".to_string(),
            vec![Capability::Bitswap { version: 1 }],
            0,
            100,
        );

        // Same version → match
        assert_eq!(
            reg.peers_with_capability(&Capability::Bitswap { version: 1 }, 0),
            vec!["peer-bs"]
        );

        // Different version → no match
        assert!(reg
            .peers_with_capability(&Capability::Bitswap { version: 2 }, 0)
            .is_empty());
    }

    // ── test 35: stats total_capabilities_registered counts all (incl. expired) ──

    #[test]
    fn test_pcr_stats_total_caps_includes_expired() {
        let mut reg = make_registry();
        // 2 capabilities
        reg.advertise(
            "active".to_string(),
            vec![Capability::Kademlia, Capability::ContentAddressing],
            0,
            100,
        );
        // 1 capability, expired at tick 50
        reg.advertise(
            "expired".to_string(),
            vec![Capability::Bitswap { version: 1 }],
            0,
            5,
        );

        let stats = reg.stats(50);
        assert_eq!(stats.total_capabilities_registered, 3); // 2 + 1
        assert_eq!(stats.active_peers, 1);
        assert_eq!(stats.expired_count, 1);
    }

    // ── test 36: multiple capabilities per peer ───────────────────────────────

    #[test]
    fn test_pcr_multiple_capabilities_per_peer() {
        let mut reg = make_registry();
        reg.advertise(
            "multi".to_string(),
            vec![
                Capability::Kademlia,
                Capability::Bitswap { version: 1 },
                Capability::ContentAddressing,
            ],
            0,
            100,
        );

        assert_eq!(
            reg.peers_with_capability(&Capability::Kademlia, 0),
            vec!["multi"]
        );
        assert_eq!(
            reg.peers_with_capability(&Capability::Bitswap { version: 1 }, 0),
            vec!["multi"]
        );
        assert_eq!(
            reg.peers_with_capability(&Capability::ContentAddressing, 0),
            vec!["multi"]
        );
    }

    // ── test 37: advertisement valid at boundary tick ─────────────────────────

    #[test]
    fn test_pcr_advert_valid_at_boundary() {
        let mut reg = make_registry();
        // expires_at_tick = 0 + 10 = 10
        reg.advertise("peer-bnd".to_string(), vec![Capability::Kademlia], 0, 10);

        // current_tick < expires_at_tick (9 < 10) → valid
        assert!(reg.peer_capabilities("peer-bnd", 9).is_some());
        // current_tick == expires_at_tick (10 == 10) → is_valid returns false (not strictly less)
        assert!(reg.peer_capabilities("peer-bnd", 10).is_none());
    }

    // ── test 38: empty registry stats ────────────────────────────────────────

    #[test]
    fn test_pcr_empty_registry_stats() {
        let reg = make_registry();
        let stats = reg.stats(0);
        assert_eq!(stats.total_peers, 0);
        assert_eq!(stats.active_peers, 0);
        assert_eq!(stats.expired_count, 0);
        assert_eq!(stats.total_capabilities_registered, 0);
    }

    // ── test 39: peers_with_capability empty when no peers ───────────────────

    #[test]
    fn test_pcr_peers_with_capability_empty_registry() {
        let reg = make_registry();
        assert!(reg
            .peers_with_capability(&Capability::Kademlia, 0)
            .is_empty());
    }

    // ── test 40: CapabilityAdvertisement is_valid boundary ───────────────────

    #[test]
    fn test_advert_is_valid_boundary() {
        let advert = CapabilityAdvertisement {
            peer_id: "p".to_string(),
            capabilities: vec![],
            advertised_at_tick: 0,
            expires_at_tick: 50,
        };
        assert!(advert.is_valid(49));
        assert!(!advert.is_valid(50));
        assert!(!advert.is_valid(51));
    }

    // ── test 41: CapabilityAdvertisement has_capability ──────────────────────

    #[test]
    fn test_advert_has_capability() {
        let advert = CapabilityAdvertisement {
            peer_id: "p".to_string(),
            capabilities: vec![Capability::Kademlia, Capability::ContentAddressing],
            advertised_at_tick: 0,
            expires_at_tick: 100,
        };
        assert!(advert.has_capability(&Capability::Kademlia));
        assert!(advert.has_capability(&Capability::ContentAddressing));
        assert!(!advert.has_capability(&Capability::Bitswap { version: 1 }));
    }

    // ── test 42: evict_expired then re-advertise ──────────────────────────────

    #[test]
    fn test_pcr_evict_then_re_advertise() {
        let mut reg = make_registry();
        reg.advertise("peer-evict".to_string(), vec![Capability::Kademlia], 0, 5);

        // Evict at tick 10
        let evicted = reg.evict_expired(10);
        assert_eq!(evicted, 1);

        // Re-advertise
        reg.advertise(
            "peer-evict".to_string(),
            vec![Capability::ContentAddressing],
            10,
            50,
        );
        let caps = reg
            .peer_capabilities("peer-evict", 10)
            .expect("should be present after re-advertise");
        assert!(caps.contains(&Capability::ContentAddressing));
    }

    // ── test 43: peers_with_capability when all peers expired ─────────────────

    #[test]
    fn test_pcr_peers_with_capability_all_expired() {
        let mut reg = make_registry();
        reg.advertise("peer-x".to_string(), vec![Capability::Kademlia], 0, 1);
        reg.advertise("peer-y".to_string(), vec![Capability::Kademlia], 0, 2);

        let peers = reg.peers_with_capability(&Capability::Kademlia, 100);
        assert!(peers.is_empty());
    }
}
