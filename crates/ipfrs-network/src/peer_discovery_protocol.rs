//! Multi-Strategy Peer Discovery Protocol
//!
//! [`PeerDiscoveryProtocol`] implements a production-grade, multi-strategy peer
//! discovery engine that tracks peers discovered via Bootstrap, mDNS, DHT,
//! PeerExchange, Static configuration, and Rendezvous namespaces.
//!
//! # Design overview
//!
//! * **Deduplication** – peers are keyed by `id`; adding an already-known peer
//!   merges its new addresses and refreshes TTL.
//! * **TTL-based expiry** – every peer carries a `discovered_at + ttl_us`
//!   deadline; [`PeerDiscoveryProtocol::expire_peers`] removes stale entries.
//! * **Event sourcing** – every state transition emits a [`PdpDiscoveryEvent`]
//!   that callers drain with [`PeerDiscoveryProtocol::drain_events`].
//! * **PRNG** – random peer selection uses a lock-free xorshift64 PRNG; no
//!   `rand` crate dependency.
//! * **Capability filtering** – peers are annotated with free-form capability
//!   strings so callers can select specialized subsets.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PRNG (no rand crate)
// ---------------------------------------------------------------------------

/// xorshift64 PRNG — fast, non-cryptographic, no external dependencies.
#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ---------------------------------------------------------------------------
// Discovery method
// ---------------------------------------------------------------------------

/// The mechanism that surfaced a particular [`PdpDiscoveredPeer`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PdpDiscoveryMethod {
    /// Peer came from a bootstrap service at the given URL.
    Bootstrap(String),
    /// Local-network multicast discovery via mDNS.
    Mdns {
        /// mDNS service name (e.g. `_ipfrs._tcp.local`).
        service_name: String,
    },
    /// Discovered through Kademlia DHT lookup for the given key.
    Dht {
        /// The DHT lookup key used to find the peer.
        query_key: String,
    },
    /// Another connected peer advertised this peer (PEX / peer exchange).
    PeerExchange {
        /// The peer-id of the peer that advertised this entry.
        source_peer: String,
    },
    /// Statically configured in application settings.
    Static(String),
    /// Discovered through a rendezvous point in the given namespace.
    Rendezvous {
        /// Rendezvous namespace (e.g. `"/ipfrs/mainnet"`).
        namespace: String,
    },
}

impl PdpDiscoveryMethod {
    /// Return the variant name as a lowercase string (used for stats bucketing
    /// and filtering).
    pub fn variant_name(&self) -> &'static str {
        match self {
            PdpDiscoveryMethod::Bootstrap(_) => "bootstrap",
            PdpDiscoveryMethod::Mdns { .. } => "mdns",
            PdpDiscoveryMethod::Dht { .. } => "dht",
            PdpDiscoveryMethod::PeerExchange { .. } => "peer_exchange",
            PdpDiscoveryMethod::Static(_) => "static",
            PdpDiscoveryMethod::Rendezvous { .. } => "rendezvous",
        }
    }
}

// ---------------------------------------------------------------------------
// Discovered peer
// ---------------------------------------------------------------------------

/// A peer that has been discovered through one of the supported discovery
/// strategies and is being tracked by [`PeerDiscoveryProtocol`].
#[derive(Clone, Debug)]
pub struct PdpDiscoveredPeer {
    /// Unique peer identifier (e.g., libp2p PeerId as string).
    pub id: String,
    /// Known multiaddresses for this peer.
    pub addresses: Vec<String>,
    /// Unix timestamp in microseconds when this peer was first recorded.
    pub discovered_at: u64,
    /// Which mechanism surfaced this peer.
    pub discovery_method: PdpDiscoveryMethod,
    /// Time-to-live in microseconds, measured from `discovered_at`.
    pub ttl_us: u64,
    /// Whether the peer has been verified (successfully dialled / handshaked).
    pub verified: bool,
    /// Free-form capability strings (e.g., `["bitswap/1.2.0", "relay/v2"]`).
    pub capabilities: Vec<String>,
    /// Unix timestamp in microseconds when the peer was last verified.
    /// `0` means never verified.
    pub(crate) verified_at: u64,
    /// Internal: current effective expiry = `discovered_at + ttl_us`.
    /// This is updated on refresh so we don't recompute it repeatedly.
    pub(crate) expires_at: u64,
}

impl PdpDiscoveredPeer {
    /// Construct a new [`PdpDiscoveredPeer`] and compute its initial expiry.
    pub fn new(
        id: impl Into<String>,
        addresses: Vec<String>,
        discovered_at: u64,
        discovery_method: PdpDiscoveryMethod,
        ttl_us: u64,
        capabilities: Vec<String>,
    ) -> Self {
        let expires_at = discovered_at.saturating_add(ttl_us);
        Self {
            id: id.into(),
            addresses,
            discovered_at,
            discovery_method,
            ttl_us,
            verified: false,
            capabilities,
            verified_at: 0,
            expires_at,
        }
    }

    /// Returns `true` if this peer has expired relative to `current_ts`.
    #[inline]
    pub fn is_expired(&self, current_ts: u64) -> bool {
        // Use the precomputed `expires_at` for constant-time check.
        self.expires_at < current_ts
    }
}

// ---------------------------------------------------------------------------
// Discovery config
// ---------------------------------------------------------------------------

/// Configuration for [`PeerDiscoveryProtocol`].
#[derive(Clone, Debug)]
pub struct PdpDiscoveryConfig {
    /// Maximum number of peers the table may hold simultaneously.
    pub max_peers: usize,
    /// Default TTL in microseconds for newly added peers.
    /// Defaults to 3 600 000 000 µs (1 hour).
    pub peer_ttl_us: u64,
    /// Enable mDNS-sourced peers.
    pub enable_mdns: bool,
    /// Enable DHT-sourced peers.
    pub enable_dht: bool,
    /// Enable peer-exchange-sourced peers.
    pub enable_peer_exchange: bool,
    /// Enable rendezvous-sourced peers.
    pub enable_rendezvous: bool,
    /// Bootstrap peer addresses (used to seed the peer table on startup).
    pub bootstrap_peers: Vec<String>,
    /// How often (in µs) the caller is expected to call `expire_peers`.
    /// Informational; the protocol does not schedule anything itself.
    pub refresh_interval_us: u64,
}

impl Default for PdpDiscoveryConfig {
    fn default() -> Self {
        Self {
            max_peers: 1024,
            peer_ttl_us: 3_600_000_000, // 1 hour
            enable_mdns: true,
            enable_dht: true,
            enable_peer_exchange: true,
            enable_rendezvous: true,
            bootstrap_peers: Vec::new(),
            refresh_interval_us: 60_000_000, // 1 minute
        }
    }
}

// ---------------------------------------------------------------------------
// Discovery events
// ---------------------------------------------------------------------------

/// Events emitted by [`PeerDiscoveryProtocol`] as state transitions occur.
#[derive(Clone, Debug)]
pub enum PdpDiscoveryEvent {
    /// A new peer was added to the table, or an existing peer was updated.
    PeerDiscovered(PdpDiscoveredPeer),
    /// A peer's TTL elapsed and it was removed from the table.
    PeerExpired(String),
    /// A peer was successfully verified.
    PeerVerified(String),
    /// A discovery method failed to produce results.
    DiscoveryMethodFailed {
        /// The variant name of the failing method (e.g., `"dht"`).
        method: String,
        /// Human-readable reason string.
        reason: String,
    },
    /// The peer table reached capacity; the number of active peers is reported.
    PeerTableFull(usize),
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Aggregate statistics emitted by [`PeerDiscoveryProtocol::stats`].
#[derive(Clone, Debug, Default)]
pub struct PdpDiscoveryStats {
    /// Total peers ever added (including re-additions / merges).
    pub total_discovered: u64,
    /// Number of currently active (non-expired) peers.
    pub active_peers: usize,
    /// Total peers evicted by TTL expiry.
    pub expired_peers: u64,
    /// Number of peer verification calls performed.
    pub verifications_performed: u64,
    /// Per-method counts: `(method_name, count)`.
    pub by_method: Vec<(String, u64)>,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors returned by [`PeerDiscoveryProtocol`] operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PdpDiscoveryError {
    /// No peer with the given id was found in the table.
    PeerNotFound(String),
    /// The peer table is at capacity and cannot accept new entries.
    MaxPeersExceeded,
    /// The supplied multiaddress is not valid.
    InvalidAddress(String),
    /// A peer with the same id was already present (used informally; normally
    /// re-additions are merged, not rejected).
    DuplicatePeer(String),
    /// A configuration value is invalid or inconsistent.
    ConfigurationError(String),
}

impl std::fmt::Display for PdpDiscoveryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PdpDiscoveryError::PeerNotFound(id) => write!(f, "peer not found: {id}"),
            PdpDiscoveryError::MaxPeersExceeded => write!(f, "peer table is full"),
            PdpDiscoveryError::InvalidAddress(addr) => write!(f, "invalid address: {addr}"),
            PdpDiscoveryError::DuplicatePeer(id) => write!(f, "duplicate peer: {id}"),
            PdpDiscoveryError::ConfigurationError(msg) => {
                write!(f, "configuration error: {msg}")
            }
        }
    }
}

impl std::error::Error for PdpDiscoveryError {}

// ---------------------------------------------------------------------------
// Internal per-method counter helper
// ---------------------------------------------------------------------------

/// Internal per-method discovery counter used inside [`PeerDiscoveryProtocol`].
#[derive(Debug, Default)]
struct MethodCounts {
    counts: HashMap<String, u64>,
}

impl MethodCounts {
    fn increment(&mut self, name: &str) {
        *self.counts.entry(name.to_string()).or_insert(0) += 1;
    }

    fn to_vec(&self) -> Vec<(String, u64)> {
        let mut v: Vec<(String, u64)> = self.counts.iter().map(|(k, v)| (k.clone(), *v)).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }
}

// ---------------------------------------------------------------------------
// Core protocol
// ---------------------------------------------------------------------------

/// Multi-strategy peer discovery protocol.
///
/// Maintains a bounded peer table with TTL expiry, verification tracking,
/// capability filtering, and per-method statistics.  All state transitions emit
/// [`PdpDiscoveryEvent`] values buffered internally and drained by the caller.
#[derive(Debug)]
pub struct PeerDiscoveryProtocol {
    /// Peer table keyed by peer id.
    peers: HashMap<String, PdpDiscoveredPeer>,
    /// Effective configuration.
    config: PdpDiscoveryConfig,
    /// Pending event buffer.
    events: Vec<PdpDiscoveryEvent>,
    /// Running total of peers ever added.
    total_discovered: u64,
    /// Running total of peers expired.
    expired_peers: u64,
    /// Running total of verification calls.
    verifications_performed: u64,
    /// Per-method add counts.
    method_counts: MethodCounts,
}

impl PeerDiscoveryProtocol {
    /// Create a new protocol instance with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns [`PdpDiscoveryError::ConfigurationError`] if `max_peers` is 0
    /// or `peer_ttl_us` is 0.
    pub fn new(config: PdpDiscoveryConfig) -> Result<Self, PdpDiscoveryError> {
        if config.max_peers == 0 {
            return Err(PdpDiscoveryError::ConfigurationError(
                "max_peers must be > 0".into(),
            ));
        }
        if config.peer_ttl_us == 0 {
            return Err(PdpDiscoveryError::ConfigurationError(
                "peer_ttl_us must be > 0".into(),
            ));
        }
        Ok(Self {
            peers: HashMap::new(),
            config,
            events: Vec::new(),
            total_discovered: 0,
            expired_peers: 0,
            verifications_performed: 0,
            method_counts: MethodCounts::default(),
        })
    }

    /// Create a new protocol instance with the default configuration.
    pub fn with_defaults() -> Self {
        // Safety: default config is always valid.
        Self::new(PdpDiscoveryConfig::default()).expect("default config is always valid")
    }

    // -----------------------------------------------------------------------
    // Peer management
    // -----------------------------------------------------------------------

    /// Add a peer to the table.
    ///
    /// * If the peer is **new** and the table has capacity, it is inserted and
    ///   [`PdpDiscoveryEvent::PeerDiscovered`] is emitted.
    /// * If the peer **already exists**, its addresses are merged (deduped) and
    ///   its TTL is refreshed; [`PdpDiscoveryEvent::PeerDiscovered`] is emitted
    ///   again to signal the update.
    /// * If the table is **full** and the peer is unknown,
    ///   [`PdpDiscoveryError::MaxPeersExceeded`] is returned and
    ///   [`PdpDiscoveryEvent::PeerTableFull`] is emitted.
    pub fn add_peer(
        &mut self,
        peer: PdpDiscoveredPeer,
    ) -> Result<PdpDiscoveryEvent, PdpDiscoveryError> {
        let method_name = peer.discovery_method.variant_name().to_string();

        if let Some(existing) = self.peers.get_mut(&peer.id) {
            // Merge addresses (deduplicate).
            for addr in &peer.addresses {
                if !existing.addresses.contains(addr) {
                    existing.addresses.push(addr.clone());
                }
            }
            // Refresh TTL.
            existing.ttl_us = peer.ttl_us;
            existing.expires_at = existing.discovered_at.saturating_add(peer.ttl_us);
            // Merge capabilities.
            for cap in &peer.capabilities {
                if !existing.capabilities.contains(cap) {
                    existing.capabilities.push(cap.clone());
                }
            }
            self.total_discovered += 1;
            self.method_counts.increment(&method_name);

            let event = PdpDiscoveryEvent::PeerDiscovered(existing.clone());
            self.events.push(event.clone());
            return Ok(event);
        }

        // New peer — check capacity.
        if self.peers.len() >= self.config.max_peers {
            let active = self.peers.len();
            let full_event = PdpDiscoveryEvent::PeerTableFull(active);
            self.events.push(full_event);
            return Err(PdpDiscoveryError::MaxPeersExceeded);
        }

        self.total_discovered += 1;
        self.method_counts.increment(&method_name);

        let event = PdpDiscoveryEvent::PeerDiscovered(peer.clone());
        self.events.push(event.clone());
        self.peers.insert(peer.id.clone(), peer);
        Ok(event)
    }

    /// Remove a peer by id.
    ///
    /// # Errors
    ///
    /// Returns [`PdpDiscoveryError::PeerNotFound`] if the peer does not exist.
    pub fn remove_peer(&mut self, id: &str) -> Result<(), PdpDiscoveryError> {
        self.peers
            .remove(id)
            .map(|_| ())
            .ok_or_else(|| PdpDiscoveryError::PeerNotFound(id.to_string()))
    }

    /// Mark a peer as verified and record the current timestamp.
    ///
    /// Emits [`PdpDiscoveryEvent::PeerVerified`] on success.
    ///
    /// # Errors
    ///
    /// Returns [`PdpDiscoveryError::PeerNotFound`] if the peer does not exist.
    pub fn verify_peer(
        &mut self,
        id: &str,
        current_ts: u64,
    ) -> Result<PdpDiscoveryEvent, PdpDiscoveryError> {
        let peer = self
            .peers
            .get_mut(id)
            .ok_or_else(|| PdpDiscoveryError::PeerNotFound(id.to_string()))?;
        peer.verified = true;
        peer.verified_at = current_ts;
        self.verifications_performed += 1;
        let event = PdpDiscoveryEvent::PeerVerified(id.to_string());
        self.events.push(event.clone());
        Ok(event)
    }

    /// Remove all peers whose TTL has elapsed relative to `current_ts`.
    ///
    /// Returns one [`PdpDiscoveryEvent::PeerExpired`] per removed peer.
    pub fn expire_peers(&mut self, current_ts: u64) -> Vec<PdpDiscoveryEvent> {
        let expired_ids: Vec<String> = self
            .peers
            .iter()
            .filter(|(_, p)| p.is_expired(current_ts))
            .map(|(id, _)| id.clone())
            .collect();

        let mut events = Vec::with_capacity(expired_ids.len());
        for id in &expired_ids {
            self.peers.remove(id);
            self.expired_peers += 1;
            let ev = PdpDiscoveryEvent::PeerExpired(id.clone());
            self.events.push(ev.clone());
            events.push(ev);
        }
        events
    }

    /// Reset the TTL of a known peer to `current_ts + config.peer_ttl_us`.
    ///
    /// # Errors
    ///
    /// Returns [`PdpDiscoveryError::PeerNotFound`] if the peer does not exist.
    pub fn refresh_peer(&mut self, id: &str, current_ts: u64) -> Result<(), PdpDiscoveryError> {
        let peer = self
            .peers
            .get_mut(id)
            .ok_or_else(|| PdpDiscoveryError::PeerNotFound(id.to_string()))?;
        peer.discovered_at = current_ts;
        peer.ttl_us = self.config.peer_ttl_us;
        peer.expires_at = current_ts.saturating_add(self.config.peer_ttl_us);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Lookups
    // -----------------------------------------------------------------------

    /// Retrieve an immutable reference to a peer by id.
    ///
    /// # Errors
    ///
    /// Returns [`PdpDiscoveryError::PeerNotFound`] if the peer does not exist.
    pub fn get_peer(&self, id: &str) -> Result<&PdpDiscoveredPeer, PdpDiscoveryError> {
        self.peers
            .get(id)
            .ok_or_else(|| PdpDiscoveryError::PeerNotFound(id.to_string()))
    }

    /// Return all peers discovered by the given method **variant name**.
    ///
    /// The `method` string must match one of the values returned by
    /// [`PdpDiscoveryMethod::variant_name`] (e.g., `"bootstrap"`, `"mdns"`).
    pub fn peers_by_method(&self, method: &str) -> Vec<&PdpDiscoveredPeer> {
        self.peers
            .values()
            .filter(|p| p.discovery_method.variant_name() == method)
            .collect()
    }

    /// Return all peers that have the given capability string.
    pub fn peers_with_capability(&self, capability: &str) -> Vec<&PdpDiscoveredPeer> {
        self.peers
            .values()
            .filter(|p| p.capabilities.iter().any(|c| c == capability))
            .collect()
    }

    /// Return up to `n` randomly selected peers using an xorshift64-based
    /// Fisher-Yates shuffle seeded with `seed`.
    ///
    /// If the table has fewer than `n` peers, all peers are returned in
    /// shuffled order.
    pub fn random_peers(&self, n: usize, seed: u64) -> Vec<&PdpDiscoveredPeer> {
        let mut indices: Vec<usize> = (0..self.peers.len()).collect();
        let peers_vec: Vec<&PdpDiscoveredPeer> = self.peers.values().collect();
        let mut state = if seed == 0 {
            0xDEAD_BEEF_CAFE_1337
        } else {
            seed
        };

        // Partial Fisher-Yates: shuffle only the first `take` elements.
        let take = n.min(indices.len());
        for i in 0..take {
            let rand_val = xorshift64(&mut state);
            let j = i + (rand_val as usize % (indices.len() - i));
            indices.swap(i, j);
        }

        indices
            .into_iter()
            .take(take)
            .map(|i| peers_vec[i])
            .collect()
    }

    // -----------------------------------------------------------------------
    // Bulk import (PeerExchange)
    // -----------------------------------------------------------------------

    /// Bulk-import peers from a `PeerExchange` handshake.
    ///
    /// For each peer in `other`:
    /// * Skip if already expired (`discovered_at + ttl_us < current_ts`).
    /// * If the peer is already known, merge addresses + capabilities and
    ///   refresh TTL.
    /// * If the peer is new, insert it (subject to capacity).
    ///
    /// Returns the list of events generated.
    pub fn merge_peer_table(
        &mut self,
        other: Vec<PdpDiscoveredPeer>,
        current_ts: u64,
    ) -> Vec<PdpDiscoveryEvent> {
        let mut events = Vec::new();
        for peer in other {
            // Skip already-expired peers.
            if peer.is_expired(current_ts) {
                continue;
            }
            match self.add_peer(peer) {
                Ok(ev) => events.push(ev),
                Err(PdpDiscoveryError::MaxPeersExceeded) => {
                    // PeerTableFull event was already pushed to self.events.
                    // Include it in the returned slice so the caller sees it.
                    if let Some(ev) = self.events.last().cloned() {
                        events.push(ev);
                    }
                    // Stop importing once the table is full.
                    break;
                }
                Err(_) => {
                    // Other errors during bulk import are silently skipped.
                }
            }
        }
        events
    }

    // -----------------------------------------------------------------------
    // Statistics and event drain
    // -----------------------------------------------------------------------

    /// Return a snapshot of current statistics.
    pub fn stats(&self) -> PdpDiscoveryStats {
        PdpDiscoveryStats {
            total_discovered: self.total_discovered,
            active_peers: self.peers.len(),
            expired_peers: self.expired_peers,
            verifications_performed: self.verifications_performed,
            by_method: self.method_counts.to_vec(),
        }
    }

    /// Drain and return all buffered events.
    ///
    /// After this call the internal event buffer is empty.
    pub fn drain_events(&mut self) -> Vec<PdpDiscoveryEvent> {
        std::mem::take(&mut self.events)
    }

    /// Return the current number of peers in the table.
    #[inline]
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Return an iterator over all peers currently in the table.
    pub fn all_peers(&self) -> impl Iterator<Item = &PdpDiscoveredPeer> {
        self.peers.values()
    }

    /// Record a discovery method failure, emitting
    /// [`PdpDiscoveryEvent::DiscoveryMethodFailed`].
    pub fn record_method_failure(&mut self, method: impl Into<String>, reason: impl Into<String>) {
        let ev = PdpDiscoveryEvent::DiscoveryMethodFailed {
            method: method.into(),
            reason: reason.into(),
        };
        self.events.push(ev);
    }
}

// ---------------------------------------------------------------------------
// Type aliases for the task spec  (PdpXxx → Xxx)
// ---------------------------------------------------------------------------

/// Type alias: `DiscoveredPeer` in the peer_discovery_protocol namespace.
pub type DiscoveredPeer = PdpDiscoveredPeer;

/// Type alias: `DiscoveryConfig` in the peer_discovery_protocol namespace.
pub type DiscoveryConfig = PdpDiscoveryConfig;

/// Type alias: `DiscoveryStats` in the peer_discovery_protocol namespace.
pub type DiscoveryStats = PdpDiscoveryStats;

/// Type alias: `DiscoveryEvent` in the peer_discovery_protocol namespace.
pub type DiscoveryEvent = PdpDiscoveryEvent;

/// Type alias: `DiscoveryError` in the peer_discovery_protocol namespace.
pub type DiscoveryError = PdpDiscoveryError;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_ts(offset_us: u64) -> u64 {
        1_000_000_000_000 + offset_us
    }

    fn default_config() -> PdpDiscoveryConfig {
        PdpDiscoveryConfig {
            max_peers: 16,
            peer_ttl_us: 3_600_000_000,
            ..PdpDiscoveryConfig::default()
        }
    }

    fn make_peer(id: &str, method: PdpDiscoveryMethod) -> PdpDiscoveredPeer {
        PdpDiscoveredPeer::new(
            id,
            vec![format!("/ip4/127.0.0.1/tcp/{}", id.len() * 1000)],
            make_ts(0),
            method,
            3_600_000_000,
            vec!["bitswap/1.2.0".into()],
        )
    }

    fn make_peer_with_caps(id: &str, caps: Vec<&str>) -> PdpDiscoveredPeer {
        PdpDiscoveredPeer::new(
            id,
            vec!["/ip4/10.0.0.1/tcp/4001".into()],
            make_ts(0),
            PdpDiscoveryMethod::Bootstrap("http://boot.example.com".into()),
            3_600_000_000,
            caps.into_iter().map(String::from).collect(),
        )
    }

    fn make_protocol() -> PeerDiscoveryProtocol {
        PeerDiscoveryProtocol::new(default_config()).expect("valid config")
    }

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_with_defaults() {
        let p = PeerDiscoveryProtocol::with_defaults();
        assert_eq!(p.peer_count(), 0);
    }

    #[test]
    fn test_new_with_valid_config() {
        let cfg = PdpDiscoveryConfig {
            max_peers: 100,
            peer_ttl_us: 60_000_000,
            ..PdpDiscoveryConfig::default()
        };
        let p = PeerDiscoveryProtocol::new(cfg);
        assert!(p.is_ok());
    }

    #[test]
    fn test_new_zero_max_peers_fails() {
        let cfg = PdpDiscoveryConfig {
            max_peers: 0,
            ..PdpDiscoveryConfig::default()
        };
        let err =
            PeerDiscoveryProtocol::new(cfg).expect_err("test: zero max_peers config should fail");
        assert!(matches!(err, PdpDiscoveryError::ConfigurationError(_)));
    }

    #[test]
    fn test_new_zero_ttl_fails() {
        let cfg = PdpDiscoveryConfig {
            peer_ttl_us: 0,
            ..PdpDiscoveryConfig::default()
        };
        let err = PeerDiscoveryProtocol::new(cfg).expect_err("test: zero ttl config should fail");
        assert!(matches!(err, PdpDiscoveryError::ConfigurationError(_)));
    }

    // -----------------------------------------------------------------------
    // Add peer
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_peer_basic() {
        let mut p = make_protocol();
        let peer = make_peer("peer-1", PdpDiscoveryMethod::Bootstrap("url".into()));
        let result = p.add_peer(peer);
        assert!(result.is_ok());
        assert_eq!(p.peer_count(), 1);
    }

    #[test]
    fn test_add_peer_emits_discovered_event() {
        let mut p = make_protocol();
        let peer = make_peer("peer-1", PdpDiscoveryMethod::Bootstrap("url".into()));
        let ev = p
            .add_peer(peer)
            .expect("test: add_peer should succeed for new peer");
        assert!(matches!(ev, PdpDiscoveryEvent::PeerDiscovered(_)));
    }

    #[test]
    fn test_add_peer_duplicate_merges_addresses() {
        let mut p = make_protocol();
        let peer1 = PdpDiscoveredPeer::new(
            "dup-peer",
            vec!["/ip4/1.2.3.4/tcp/4001".into()],
            make_ts(0),
            PdpDiscoveryMethod::Bootstrap("url".into()),
            3_600_000_000,
            vec![],
        );
        let peer2 = PdpDiscoveredPeer::new(
            "dup-peer",
            vec!["/ip4/5.6.7.8/tcp/4001".into()],
            make_ts(10),
            PdpDiscoveryMethod::Bootstrap("url".into()),
            3_600_000_000,
            vec![],
        );
        p.add_peer(peer1)
            .expect("test: add first peer should succeed");
        p.add_peer(peer2)
            .expect("test: add duplicate peer should succeed (merge)");
        let stored = p
            .get_peer("dup-peer")
            .expect("test: get_peer should find dup-peer");
        assert_eq!(stored.addresses.len(), 2);
        assert_eq!(p.peer_count(), 1);
    }

    #[test]
    fn test_add_peer_duplicate_deduplicates_addresses() {
        let mut p = make_protocol();
        let addr = "/ip4/1.2.3.4/tcp/4001";
        let peer1 = PdpDiscoveredPeer::new(
            "dup2",
            vec![addr.into()],
            make_ts(0),
            PdpDiscoveryMethod::Mdns {
                service_name: "svc".into(),
            },
            3_600_000_000,
            vec![],
        );
        let peer2 = PdpDiscoveredPeer::new(
            "dup2",
            vec![addr.into()], // same address
            make_ts(1),
            PdpDiscoveryMethod::Mdns {
                service_name: "svc".into(),
            },
            3_600_000_000,
            vec![],
        );
        p.add_peer(peer1)
            .expect("test: add first dup2 peer should succeed");
        p.add_peer(peer2)
            .expect("test: add duplicate dup2 peer should succeed (merge)");
        let stored = p.get_peer("dup2").expect("test: get_peer should find dup2");
        assert_eq!(stored.addresses.len(), 1);
    }

    #[test]
    fn test_add_peer_table_full_error() {
        let cfg = PdpDiscoveryConfig {
            max_peers: 2,
            peer_ttl_us: 3_600_000_000,
            ..PdpDiscoveryConfig::default()
        };
        let mut p = PeerDiscoveryProtocol::new(cfg)
            .expect("test: valid config with max_peers=2 should succeed");
        p.add_peer(make_peer("a", PdpDiscoveryMethod::Static("a".into())))
            .expect("test: add peer a should succeed");
        p.add_peer(make_peer("b", PdpDiscoveryMethod::Static("b".into())))
            .expect("test: add peer b should succeed");
        let err = p
            .add_peer(make_peer("c", PdpDiscoveryMethod::Static("c".into())))
            .expect_err("test: add peer c beyond max_peers should fail");
        assert!(matches!(err, PdpDiscoveryError::MaxPeersExceeded));
    }

    #[test]
    fn test_add_peer_table_full_emits_event() {
        let cfg = PdpDiscoveryConfig {
            max_peers: 1,
            peer_ttl_us: 3_600_000_000,
            ..PdpDiscoveryConfig::default()
        };
        let mut p = PeerDiscoveryProtocol::new(cfg)
            .expect("test: valid config with max_peers=1 should succeed");
        p.add_peer(make_peer("a", PdpDiscoveryMethod::Static("a".into())))
            .expect("test: add peer a should succeed");
        let _ = p.add_peer(make_peer("b", PdpDiscoveryMethod::Static("b".into())));
        let events = p.drain_events();
        // Should contain PeerDiscovered(a) + PeerTableFull
        assert!(events
            .iter()
            .any(|e| matches!(e, PdpDiscoveryEvent::PeerTableFull(_))));
    }

    #[test]
    fn test_add_peer_multiple_methods() {
        let mut p = make_protocol();
        p.add_peer(make_peer("p1", PdpDiscoveryMethod::Bootstrap("b".into())))
            .expect("test: add p1 bootstrap peer should succeed");
        p.add_peer(make_peer(
            "p2",
            PdpDiscoveryMethod::Mdns {
                service_name: "s".into(),
            },
        ))
        .expect("test: add p2 mdns peer should succeed");
        p.add_peer(make_peer(
            "p3",
            PdpDiscoveryMethod::Dht {
                query_key: "k".into(),
            },
        ))
        .expect("test: add p3 dht peer should succeed");
        p.add_peer(make_peer(
            "p4",
            PdpDiscoveryMethod::PeerExchange {
                source_peer: "src".into(),
            },
        ))
        .expect("test: add p4 peer_exchange peer should succeed");
        p.add_peer(make_peer("p5", PdpDiscoveryMethod::Static("addr".into())))
            .expect("test: add p5 static peer should succeed");
        p.add_peer(make_peer(
            "p6",
            PdpDiscoveryMethod::Rendezvous {
                namespace: "ns".into(),
            },
        ))
        .expect("test: add p6 rendezvous peer should succeed");
        assert_eq!(p.peer_count(), 6);
    }

    // -----------------------------------------------------------------------
    // Remove peer
    // -----------------------------------------------------------------------

    #[test]
    fn test_remove_peer_success() {
        let mut p = make_protocol();
        p.add_peer(make_peer("r1", PdpDiscoveryMethod::Static("x".into())))
            .expect("test: add r1 peer should succeed");
        assert!(p.remove_peer("r1").is_ok());
        assert_eq!(p.peer_count(), 0);
    }

    #[test]
    fn test_remove_peer_not_found() {
        let mut p = make_protocol();
        let err = p
            .remove_peer("ghost")
            .expect_err("test: remove nonexistent peer should fail");
        assert!(matches!(err, PdpDiscoveryError::PeerNotFound(_)));
    }

    #[test]
    fn test_remove_then_add_same_id() {
        let mut p = make_protocol();
        p.add_peer(make_peer("x", PdpDiscoveryMethod::Static("a".into())))
            .expect("test: add peer x should succeed");
        p.remove_peer("x")
            .expect("test: remove peer x should succeed");
        // Should be addable again.
        p.add_peer(make_peer("x", PdpDiscoveryMethod::Static("a".into())))
            .expect("test: re-add peer x after removal should succeed");
        assert_eq!(p.peer_count(), 1);
    }

    // -----------------------------------------------------------------------
    // Verify peer
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_peer_success() {
        let mut p = make_protocol();
        p.add_peer(make_peer("v1", PdpDiscoveryMethod::Bootstrap("u".into())))
            .expect("test: add v1 peer should succeed");
        let ev = p
            .verify_peer("v1", make_ts(1000))
            .expect("test: verify_peer v1 should succeed");
        assert!(matches!(ev, PdpDiscoveryEvent::PeerVerified(_)));
        assert!(
            p.get_peer("v1")
                .expect("test: get_peer v1 should succeed after verify")
                .verified
        );
    }

    #[test]
    fn test_verify_peer_not_found() {
        let mut p = make_protocol();
        let err = p
            .verify_peer("ghost", make_ts(0))
            .expect_err("test: verify nonexistent peer should fail");
        assert!(matches!(err, PdpDiscoveryError::PeerNotFound(_)));
    }

    #[test]
    fn test_verify_peer_updates_verified_at() {
        let mut p = make_protocol();
        p.add_peer(make_peer("v2", PdpDiscoveryMethod::Bootstrap("u".into())))
            .expect("test: add v2 peer should succeed");
        let ts = make_ts(500_000);
        p.verify_peer("v2", ts)
            .expect("test: verify_peer v2 should succeed");
        assert_eq!(
            p.get_peer("v2")
                .expect("test: get_peer v2 should succeed after verify")
                .verified_at,
            ts
        );
    }

    #[test]
    fn test_verify_peer_increments_stats() {
        let mut p = make_protocol();
        p.add_peer(make_peer("v3", PdpDiscoveryMethod::Bootstrap("u".into())))
            .expect("test: add v3 peer should succeed");
        p.verify_peer("v3", make_ts(0))
            .expect("test: first verify_peer v3 should succeed");
        p.verify_peer("v3", make_ts(1))
            .expect("test: second verify_peer v3 should succeed");
        assert_eq!(p.stats().verifications_performed, 2);
    }

    // -----------------------------------------------------------------------
    // TTL expiration
    // -----------------------------------------------------------------------

    #[test]
    fn test_expire_peers_removes_stale() {
        let mut p = make_protocol();
        let ttl = 1_000_000; // 1 second in µs
        let added_at = make_ts(0);
        let peer = PdpDiscoveredPeer::new(
            "exp-1",
            vec![],
            added_at,
            PdpDiscoveryMethod::Bootstrap("b".into()),
            ttl,
            vec![],
        );
        p.add_peer(peer)
            .expect("test: add exp-1 peer should succeed");
        // Current time is well past expiry.
        let evs = p.expire_peers(added_at + ttl + 1);
        assert_eq!(evs.len(), 1);
        assert!(matches!(&evs[0], PdpDiscoveryEvent::PeerExpired(id) if id == "exp-1"));
        assert_eq!(p.peer_count(), 0);
    }

    #[test]
    fn test_expire_peers_keeps_live() {
        let mut p = make_protocol();
        let ttl = 1_000_000;
        let added_at = make_ts(0);
        let peer = PdpDiscoveredPeer::new(
            "live-1",
            vec![],
            added_at,
            PdpDiscoveryMethod::Static("x".into()),
            ttl,
            vec![],
        );
        p.add_peer(peer)
            .expect("test: add live-1 peer should succeed");
        // Current time is before expiry.
        let evs = p.expire_peers(added_at + ttl - 1);
        assert_eq!(evs.len(), 0);
        assert_eq!(p.peer_count(), 1);
    }

    #[test]
    fn test_expire_peers_increments_expired_count() {
        let mut p = make_protocol();
        let ttl = 500;
        let t0 = make_ts(0);
        for i in 0..3u64 {
            let peer = PdpDiscoveredPeer::new(
                format!("ep-{i}"),
                vec![],
                t0,
                PdpDiscoveryMethod::Dht {
                    query_key: "k".into(),
                },
                ttl,
                vec![],
            );
            p.add_peer(peer)
                .expect("test: add ep-{i} peer should succeed");
        }
        p.expire_peers(t0 + ttl + 1);
        assert_eq!(p.stats().expired_peers, 3);
    }

    #[test]
    fn test_expire_peers_emits_events_in_buffer() {
        let mut p = make_protocol();
        let ttl = 100;
        let t0 = make_ts(0);
        let peer = PdpDiscoveredPeer::new(
            "buf-exp",
            vec![],
            t0,
            PdpDiscoveryMethod::Static("x".into()),
            ttl,
            vec![],
        );
        p.add_peer(peer)
            .expect("test: add buf-exp peer should succeed");
        p.expire_peers(t0 + ttl + 1);
        let events = p.drain_events();
        assert!(events
            .iter()
            .any(|e| matches!(e, PdpDiscoveryEvent::PeerExpired(_))));
    }

    #[test]
    fn test_expire_only_expired_not_live() {
        let mut p = make_protocol();
        let t0 = make_ts(0);
        let short_ttl = 1_000;
        let long_ttl = 3_600_000_000;
        let short_peer = PdpDiscoveredPeer::new(
            "short",
            vec![],
            t0,
            PdpDiscoveryMethod::Static("x".into()),
            short_ttl,
            vec![],
        );
        let long_peer = PdpDiscoveredPeer::new(
            "long",
            vec![],
            t0,
            PdpDiscoveryMethod::Static("y".into()),
            long_ttl,
            vec![],
        );
        p.add_peer(short_peer)
            .expect("test: add short-ttl peer should succeed");
        p.add_peer(long_peer)
            .expect("test: add long-ttl peer should succeed");
        let evs = p.expire_peers(t0 + short_ttl + 1);
        assert_eq!(evs.len(), 1);
        assert!(matches!(&evs[0], PdpDiscoveryEvent::PeerExpired(id) if id == "short"));
        assert_eq!(p.peer_count(), 1);
        assert!(p.get_peer("long").is_ok());
    }

    // -----------------------------------------------------------------------
    // Refresh peer
    // -----------------------------------------------------------------------

    #[test]
    fn test_refresh_peer_extends_ttl() {
        let mut p = make_protocol();
        let short_ttl = 100;
        let t0 = make_ts(0);
        let peer = PdpDiscoveredPeer::new(
            "ref-1",
            vec![],
            t0,
            PdpDiscoveryMethod::Bootstrap("b".into()),
            short_ttl,
            vec![],
        );
        p.add_peer(peer)
            .expect("test: add ref-1 peer should succeed");
        // Refresh before expiry.
        let t1 = t0 + 50;
        p.refresh_peer("ref-1", t1)
            .expect("test: refresh_peer ref-1 should succeed");
        // Expire at old deadline — should NOT expire because TTL was refreshed.
        let evs = p.expire_peers(t0 + short_ttl + 1);
        assert_eq!(evs.len(), 0);
    }

    #[test]
    fn test_refresh_peer_not_found() {
        let mut p = make_protocol();
        let err = p
            .refresh_peer("ghost", make_ts(0))
            .expect_err("test: refresh nonexistent peer should fail");
        assert!(matches!(err, PdpDiscoveryError::PeerNotFound(_)));
    }

    #[test]
    fn test_refresh_then_expire() {
        let mut p = make_protocol();
        let t0 = make_ts(0);
        let peer = PdpDiscoveredPeer::new(
            "re-exp",
            vec![],
            t0,
            PdpDiscoveryMethod::Static("x".into()),
            3_600_000_000,
            vec![],
        );
        p.add_peer(peer)
            .expect("test: add re-exp peer should succeed");
        // Refresh with tiny TTL (config default), then advance past it.
        p.refresh_peer("re-exp", t0)
            .expect("test: refresh_peer re-exp should succeed");
        // Default TTL is 3_600_000_000 us — advance past that.
        let evs = p.expire_peers(t0 + 3_600_000_001);
        assert_eq!(evs.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Get peer
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_peer_success() {
        let mut p = make_protocol();
        p.add_peer(make_peer("g1", PdpDiscoveryMethod::Static("a".into())))
            .expect("test: add g1 peer should succeed");
        assert!(p.get_peer("g1").is_ok());
    }

    #[test]
    fn test_get_peer_not_found() {
        let p = make_protocol();
        assert!(matches!(
            p.get_peer("none"),
            Err(PdpDiscoveryError::PeerNotFound(_))
        ));
    }

    // -----------------------------------------------------------------------
    // peers_by_method
    // -----------------------------------------------------------------------

    #[test]
    fn test_peers_by_method_bootstrap() {
        let mut p = make_protocol();
        p.add_peer(make_peer("b1", PdpDiscoveryMethod::Bootstrap("url".into())))
            .expect("test: add b1 bootstrap peer should succeed");
        p.add_peer(make_peer(
            "m1",
            PdpDiscoveryMethod::Mdns {
                service_name: "s".into(),
            },
        ))
        .expect("test: add m1 mdns peer should succeed");
        let boots = p.peers_by_method("bootstrap");
        assert_eq!(boots.len(), 1);
        assert_eq!(boots[0].id, "b1");
    }

    #[test]
    fn test_peers_by_method_mdns() {
        let mut p = make_protocol();
        p.add_peer(make_peer(
            "m2",
            PdpDiscoveryMethod::Mdns {
                service_name: "svc".into(),
            },
        ))
        .expect("test: add m2 mdns peer should succeed");
        p.add_peer(make_peer(
            "m3",
            PdpDiscoveryMethod::Mdns {
                service_name: "svc".into(),
            },
        ))
        .expect("test: add m3 mdns peer should succeed");
        assert_eq!(p.peers_by_method("mdns").len(), 2);
    }

    #[test]
    fn test_peers_by_method_dht() {
        let mut p = make_protocol();
        p.add_peer(make_peer(
            "d1",
            PdpDiscoveryMethod::Dht {
                query_key: "k".into(),
            },
        ))
        .expect("test: add d1 dht peer should succeed");
        assert_eq!(p.peers_by_method("dht").len(), 1);
    }

    #[test]
    fn test_peers_by_method_peer_exchange() {
        let mut p = make_protocol();
        p.add_peer(make_peer(
            "pe1",
            PdpDiscoveryMethod::PeerExchange {
                source_peer: "src".into(),
            },
        ))
        .expect("test: add pe1 peer_exchange peer should succeed");
        assert_eq!(p.peers_by_method("peer_exchange").len(), 1);
    }

    #[test]
    fn test_peers_by_method_rendezvous() {
        let mut p = make_protocol();
        p.add_peer(make_peer(
            "rv1",
            PdpDiscoveryMethod::Rendezvous {
                namespace: "ns".into(),
            },
        ))
        .expect("test: add rv1 rendezvous peer should succeed");
        assert_eq!(p.peers_by_method("rendezvous").len(), 1);
    }

    #[test]
    fn test_peers_by_method_no_match() {
        let mut p = make_protocol();
        p.add_peer(make_peer("x", PdpDiscoveryMethod::Static("a".into())))
            .expect("test: add x peer for method filter test should succeed");
        assert_eq!(p.peers_by_method("dht").len(), 0);
    }

    // -----------------------------------------------------------------------
    // peers_with_capability
    // -----------------------------------------------------------------------

    #[test]
    fn test_peers_with_capability_found() {
        let mut p = make_protocol();
        p.add_peer(make_peer_with_caps("c1", vec!["relay/v2", "bitswap"]))
            .expect("test: add c1 peer with caps should succeed");
        p.add_peer(make_peer_with_caps("c2", vec!["bitswap"]))
            .expect("test: add c2 peer with caps should succeed");
        let relay_peers = p.peers_with_capability("relay/v2");
        assert_eq!(relay_peers.len(), 1);
        assert_eq!(relay_peers[0].id, "c1");
    }

    #[test]
    fn test_peers_with_capability_not_found() {
        let mut p = make_protocol();
        p.add_peer(make_peer_with_caps("c3", vec!["bitswap"]))
            .expect("test: add c3 peer with caps should succeed");
        assert_eq!(p.peers_with_capability("relay/v2").len(), 0);
    }

    #[test]
    fn test_peers_with_capability_multiple() {
        let mut p = make_protocol();
        for i in 0..5 {
            p.add_peer(make_peer_with_caps(
                &format!("peer-cap-{i}"),
                vec!["dht/1.0"],
            ))
            .expect("test: add peer with dht/1.0 capability should succeed");
        }
        assert_eq!(p.peers_with_capability("dht/1.0").len(), 5);
    }

    #[test]
    fn test_peers_with_capability_empty_table() {
        let p = make_protocol();
        assert_eq!(p.peers_with_capability("anything").len(), 0);
    }

    // -----------------------------------------------------------------------
    // random_peers
    // -----------------------------------------------------------------------

    #[test]
    fn test_random_peers_count() {
        let mut p = make_protocol();
        for i in 0..10u32 {
            p.add_peer(make_peer(
                &format!("rp-{i}"),
                PdpDiscoveryMethod::Static("a".into()),
            ))
            .expect("test: add random peer should succeed");
        }
        let sample = p.random_peers(5, 0xDEAD);
        assert_eq!(sample.len(), 5);
    }

    #[test]
    fn test_random_peers_fewer_than_requested() {
        let mut p = make_protocol();
        for i in 0..3u32 {
            p.add_peer(make_peer(
                &format!("rp2-{i}"),
                PdpDiscoveryMethod::Static("a".into()),
            ))
            .expect("test: add rp2 peer should succeed");
        }
        let sample = p.random_peers(10, 1234);
        assert_eq!(sample.len(), 3);
    }

    #[test]
    fn test_random_peers_zero_n() {
        let mut p = make_protocol();
        p.add_peer(make_peer("z", PdpDiscoveryMethod::Static("a".into())))
            .expect("test: add z peer for random_peers test should succeed");
        let sample = p.random_peers(0, 42);
        assert_eq!(sample.len(), 0);
    }

    #[test]
    fn test_random_peers_no_duplicates() {
        let mut p = make_protocol();
        for i in 0..8u32 {
            p.add_peer(make_peer(
                &format!("rp3-{i}"),
                PdpDiscoveryMethod::Static("a".into()),
            ))
            .expect("test: add rp3 peer should succeed");
        }
        let sample = p.random_peers(8, 0xCAFE);
        let mut ids: Vec<&str> = sample.iter().map(|pr| pr.id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 8);
    }

    #[test]
    fn test_random_peers_different_seeds_different_order() {
        let mut p = make_protocol();
        for i in 0..10u32 {
            p.add_peer(make_peer(
                &format!("seed-{i}"),
                PdpDiscoveryMethod::Static("a".into()),
            ))
            .expect("test: add seed peer should succeed");
        }
        let s1: Vec<String> = p.random_peers(10, 1).iter().map(|p| p.id.clone()).collect();
        let s2: Vec<String> = p
            .random_peers(10, 99999)
            .iter()
            .map(|p| p.id.clone())
            .collect();
        // With different seeds the order should differ (not guaranteed but
        // extremely likely with 10 distinct peers).
        assert_ne!(s1, s2);
    }

    // -----------------------------------------------------------------------
    // merge_peer_table
    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_peer_table_basic() {
        let mut p = make_protocol();
        let t0 = make_ts(0);
        let peers = vec![
            make_peer("mp1", PdpDiscoveryMethod::Static("a".into())),
            make_peer("mp2", PdpDiscoveryMethod::Static("b".into())),
        ];
        let events = p.merge_peer_table(peers, t0);
        assert_eq!(events.len(), 2);
        assert_eq!(p.peer_count(), 2);
    }

    #[test]
    fn test_merge_peer_table_skips_expired() {
        let mut p = make_protocol();
        let t0 = make_ts(0);
        let ttl = 100;
        let expired = PdpDiscoveredPeer::new(
            "old-peer",
            vec![],
            t0,
            PdpDiscoveryMethod::Static("x".into()),
            ttl,
            vec![],
        );
        // current_ts is past the expiry.
        let events = p.merge_peer_table(vec![expired], t0 + ttl + 1);
        assert_eq!(events.len(), 0);
        assert_eq!(p.peer_count(), 0);
    }

    #[test]
    fn test_merge_peer_table_deduplicates() {
        let mut p = make_protocol();
        let t0 = make_ts(0);
        p.add_peer(make_peer("dup", PdpDiscoveryMethod::Static("a".into())))
            .expect("test: add dup peer before merge should succeed");
        let import = vec![make_peer("dup", PdpDiscoveryMethod::Static("a".into()))];
        p.merge_peer_table(import, t0);
        assert_eq!(p.peer_count(), 1);
    }

    #[test]
    fn test_merge_peer_table_stops_at_capacity() {
        let cfg = PdpDiscoveryConfig {
            max_peers: 2,
            peer_ttl_us: 3_600_000_000,
            ..PdpDiscoveryConfig::default()
        };
        let mut p = PeerDiscoveryProtocol::new(cfg)
            .expect("test: valid config with max_peers=2 should succeed");
        let t0 = make_ts(0);
        let peers: Vec<_> = (0..5)
            .map(|i| make_peer(&format!("mp-{i}"), PdpDiscoveryMethod::Static("a".into())))
            .collect();
        p.merge_peer_table(peers, t0);
        // Should not exceed max_peers.
        assert!(p.peer_count() <= 2);
    }

    #[test]
    fn test_merge_peer_table_returns_events() {
        let mut p = make_protocol();
        let t0 = make_ts(0);
        let peers: Vec<_> = (0..3)
            .map(|i| {
                make_peer(
                    &format!("evt-{i}"),
                    PdpDiscoveryMethod::Bootstrap("b".into()),
                )
            })
            .collect();
        let events = p.merge_peer_table(peers, t0);
        assert_eq!(events.len(), 3);
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_total_discovered() {
        let mut p = make_protocol();
        for i in 0..3u32 {
            p.add_peer(make_peer(
                &format!("st-{i}"),
                PdpDiscoveryMethod::Bootstrap("b".into()),
            ))
            .expect("test: add st peer should succeed");
        }
        assert_eq!(p.stats().total_discovered, 3);
    }

    #[test]
    fn test_stats_active_peers() {
        let mut p = make_protocol();
        for i in 0..4u32 {
            p.add_peer(make_peer(
                &format!("ap-{i}"),
                PdpDiscoveryMethod::Static("a".into()),
            ))
            .expect("test: add ap peer should succeed");
        }
        assert_eq!(p.stats().active_peers, 4);
    }

    #[test]
    fn test_stats_expired_peers() {
        let mut p = make_protocol();
        let t0 = make_ts(0);
        let short = PdpDiscoveredPeer::new(
            "s",
            vec![],
            t0,
            PdpDiscoveryMethod::Static("x".into()),
            100,
            vec![],
        );
        p.add_peer(short)
            .expect("test: add short-lived peer should succeed");
        p.expire_peers(t0 + 200);
        assert_eq!(p.stats().expired_peers, 1);
    }

    #[test]
    fn test_stats_by_method() {
        let mut p = make_protocol();
        p.add_peer(make_peer("s1", PdpDiscoveryMethod::Bootstrap("b".into())))
            .expect("test: add s1 bootstrap peer should succeed");
        p.add_peer(make_peer("s2", PdpDiscoveryMethod::Bootstrap("b".into())))
            .expect("test: add s2 bootstrap peer should succeed");
        p.add_peer(make_peer(
            "s3",
            PdpDiscoveryMethod::Mdns {
                service_name: "s".into(),
            },
        ))
        .expect("test: add s3 mdns peer should succeed");
        let stats = p.stats();
        let boot_count = stats
            .by_method
            .iter()
            .find(|(m, _)| m == "bootstrap")
            .map(|(_, c)| *c)
            .unwrap_or(0);
        let mdns_count = stats
            .by_method
            .iter()
            .find(|(m, _)| m == "mdns")
            .map(|(_, c)| *c)
            .unwrap_or(0);
        assert_eq!(boot_count, 2);
        assert_eq!(mdns_count, 1);
    }

    #[test]
    fn test_stats_verifications() {
        let mut p = make_protocol();
        p.add_peer(make_peer("vv1", PdpDiscoveryMethod::Static("a".into())))
            .expect("test: add vv1 peer should succeed");
        p.add_peer(make_peer("vv2", PdpDiscoveryMethod::Static("b".into())))
            .expect("test: add vv2 peer should succeed");
        p.verify_peer("vv1", make_ts(0))
            .expect("test: verify_peer vv1 should succeed");
        p.verify_peer("vv2", make_ts(1))
            .expect("test: verify_peer vv2 should succeed");
        assert_eq!(p.stats().verifications_performed, 2);
    }

    // -----------------------------------------------------------------------
    // Event drain
    // -----------------------------------------------------------------------

    #[test]
    fn test_drain_events_clears_buffer() {
        let mut p = make_protocol();
        p.add_peer(make_peer("ev1", PdpDiscoveryMethod::Static("a".into())))
            .expect("test: add ev1 peer should succeed");
        let first = p.drain_events();
        assert!(!first.is_empty());
        let second = p.drain_events();
        assert!(second.is_empty());
    }

    #[test]
    fn test_drain_events_accumulates() {
        let mut p = make_protocol();
        p.add_peer(make_peer("e1", PdpDiscoveryMethod::Static("a".into())))
            .expect("test: add e1 peer should succeed");
        p.add_peer(make_peer("e2", PdpDiscoveryMethod::Static("b".into())))
            .expect("test: add e2 peer should succeed");
        let evs = p.drain_events();
        assert_eq!(evs.len(), 2);
    }

    #[test]
    fn test_drain_events_after_expiry() {
        let mut p = make_protocol();
        let t0 = make_ts(0);
        let peer = PdpDiscoveredPeer::new(
            "de1",
            vec![],
            t0,
            PdpDiscoveryMethod::Static("x".into()),
            100,
            vec![],
        );
        p.add_peer(peer).expect("test: add de1 peer should succeed");
        p.drain_events(); // Clear initial events.
        p.expire_peers(t0 + 200);
        let evs = p.drain_events();
        assert_eq!(evs.len(), 1);
        assert!(matches!(&evs[0], PdpDiscoveryEvent::PeerExpired(_)));
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_display_peer_not_found() {
        let err = PdpDiscoveryError::PeerNotFound("abc".into());
        assert!(err.to_string().contains("abc"));
    }

    #[test]
    fn test_error_display_max_peers() {
        let err = PdpDiscoveryError::MaxPeersExceeded;
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn test_error_display_invalid_address() {
        let err = PdpDiscoveryError::InvalidAddress("bad-addr".into());
        assert!(err.to_string().contains("bad-addr"));
    }

    #[test]
    fn test_error_display_duplicate_peer() {
        let err = PdpDiscoveryError::DuplicatePeer("dup".into());
        assert!(err.to_string().contains("dup"));
    }

    #[test]
    fn test_error_display_config_error() {
        let err = PdpDiscoveryError::ConfigurationError("bad".into());
        assert!(err.to_string().contains("bad"));
    }

    // -----------------------------------------------------------------------
    // xorshift64 PRNG
    // -----------------------------------------------------------------------

    #[test]
    fn test_xorshift64_not_zero() {
        let mut state = 12345;
        let val = xorshift64(&mut state);
        assert_ne!(val, 0);
    }

    #[test]
    fn test_xorshift64_sequence_is_different() {
        let mut state = 1;
        let v1 = xorshift64(&mut state);
        let v2 = xorshift64(&mut state);
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_xorshift64_deterministic() {
        let mut s1 = 42u64;
        let mut s2 = 42u64;
        for _ in 0..100 {
            assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
        }
    }

    // -----------------------------------------------------------------------
    // Discovery method variant names
    // -----------------------------------------------------------------------

    #[test]
    fn test_variant_names() {
        assert_eq!(
            PdpDiscoveryMethod::Bootstrap("u".into()).variant_name(),
            "bootstrap"
        );
        assert_eq!(
            PdpDiscoveryMethod::Mdns {
                service_name: "s".into()
            }
            .variant_name(),
            "mdns"
        );
        assert_eq!(
            PdpDiscoveryMethod::Dht {
                query_key: "k".into()
            }
            .variant_name(),
            "dht"
        );
        assert_eq!(
            PdpDiscoveryMethod::PeerExchange {
                source_peer: "p".into()
            }
            .variant_name(),
            "peer_exchange"
        );
        assert_eq!(
            PdpDiscoveryMethod::Static("a".into()).variant_name(),
            "static"
        );
        assert_eq!(
            PdpDiscoveryMethod::Rendezvous {
                namespace: "n".into()
            }
            .variant_name(),
            "rendezvous"
        );
    }

    // -----------------------------------------------------------------------
    // record_method_failure
    // -----------------------------------------------------------------------

    #[test]
    fn test_record_method_failure_emits_event() {
        let mut p = make_protocol();
        p.record_method_failure("dht", "timeout");
        let evs = p.drain_events();
        assert_eq!(evs.len(), 1);
        assert!(matches!(
            &evs[0],
            PdpDiscoveryEvent::DiscoveryMethodFailed { method, reason }
            if method == "dht" && reason == "timeout"
        ));
    }

    // -----------------------------------------------------------------------
    // all_peers iterator
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_peers_iterator() {
        let mut p = make_protocol();
        for i in 0..5u32 {
            p.add_peer(make_peer(
                &format!("it-{i}"),
                PdpDiscoveryMethod::Static("a".into()),
            ))
            .expect("test: add it peer should succeed");
        }
        assert_eq!(p.all_peers().count(), 5);
    }

    // -----------------------------------------------------------------------
    // Type alias sanity
    // -----------------------------------------------------------------------

    #[test]
    fn test_type_aliases_are_usable() {
        let _: DiscoveredPeer = PdpDiscoveredPeer::new(
            "alias-test",
            vec![],
            0,
            PdpDiscoveryMethod::Static("x".into()),
            1000,
            vec![],
        );
        let _: DiscoveryConfig = PdpDiscoveryConfig::default();
        let _: DiscoveryError = PdpDiscoveryError::MaxPeersExceeded;
    }
}
