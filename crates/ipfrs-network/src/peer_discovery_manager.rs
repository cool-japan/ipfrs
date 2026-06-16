//! Multi-strategy Peer Discovery Manager
//!
//! Provides [`PeerDiscoveryManager`] — a production-grade component that tracks
//! peer candidates sourced from multiple discovery strategies, deduplicates
//! them, scores connection likelihood, and drives the connection-attempt
//! lifecycle from first-seen through disconnect.
//!
//! # Design overview
//!
//! * **Deduplication** – every candidate is keyed by its peer-id string; a
//!   second `add_candidate` call for the same peer-id merges the new addresses
//!   into the existing record rather than creating a duplicate entry.
//!
//! * **Scoring** – each candidate carries a `connect_score` (∈ [0, 10]) that
//!   rises on successful connection and falls on failure, timeout, or
//!   unreachability.  Candidates are ranked by score when the caller asks for
//!   connection targets.
//!
//! * **Backoff** – `candidates_to_try` returns only those peers whose
//!   `last_attempt` is old enough (≥ `retry_backoff_ms`) or who have never
//!   been attempted.
//!
//! * **Stale eviction** – `evict_stale` removes non-connected candidates that
//!   were both *discovered* and *last attempted* (or never attempted) beyond
//!   `stale_threshold_ms`.

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The mechanism through which a [`PeerCandidate`] was first discovered.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DiscoveryMethod {
    /// Peer came from the static bootstrap list.
    Bootstrap,
    /// Local-network multicast discovery via mDNS.
    Mdns,
    /// Kademlia DHT iterative peer lookup / random walk.
    Dht,
    /// Another connected peer advertised this peer (PEX).
    PeerExchange,
    /// Manually added by an operator or the application layer.
    Manual,
}

impl std::fmt::Display for DiscoveryMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiscoveryMethod::Bootstrap => write!(f, "Bootstrap"),
            DiscoveryMethod::Mdns => write!(f, "Mdns"),
            DiscoveryMethod::Dht => write!(f, "Dht"),
            DiscoveryMethod::PeerExchange => write!(f, "PeerExchange"),
            DiscoveryMethod::Manual => write!(f, "Manual"),
        }
    }
}

/// A single peer candidate tracked by [`PeerDiscoveryManager`].
#[derive(Clone, Debug)]
pub struct PeerCandidate {
    /// Unique peer identifier (e.g., libp2p PeerId as a string).
    pub peer_id: String,
    /// Known multiaddresses for this peer.
    pub addresses: Vec<String>,
    /// Which discovery mechanism first surfaced this peer.
    pub discovered_via: DiscoveryMethod,
    /// Unix timestamp (milliseconds) when this peer was first discovered.
    pub discovered_at: u64,
    /// Connection likelihood score.  Starts at `1.0`; capped to `[0.0, 10.0]`.
    pub connect_score: f64,
    /// Whether at least one connection attempt has been made.
    pub attempted: bool,
    /// Whether the peer is currently connected.
    pub connected: bool,
    /// Unix timestamp (milliseconds) of the most recent connection attempt.
    pub last_attempt: Option<u64>,
}

impl PeerCandidate {
    /// Create a new candidate with default scoring state.
    pub fn new(
        peer_id: impl Into<String>,
        addresses: Vec<String>,
        discovered_via: DiscoveryMethod,
        discovered_at: u64,
    ) -> Self {
        Self {
            peer_id: peer_id.into(),
            addresses,
            discovered_via,
            discovered_at,
            connect_score: 1.0,
            attempted: false,
            connected: false,
            last_attempt: None,
        }
    }
}

/// Outcome of a single connection attempt reported back to the manager.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConnectOutcome {
    /// Connection established successfully.
    Success,
    /// Remote peer actively refused the connection.
    Refused,
    /// Connection attempt timed out.
    Timeout,
    /// Peer address is not reachable at all.
    Unreachable,
}

/// Configuration knobs for [`PeerDiscoveryManager`].
#[derive(Clone, Debug)]
pub struct DiscoveryConfig {
    /// Maximum number of candidates to track at any time.
    pub max_candidates: usize,
    /// Maximum number of simultaneously connected peers.
    pub max_connected: usize,
    /// How often (ms) scores are eligible for re-evaluation (informational).
    pub rescore_interval_ms: u64,
    /// Minimum gap (ms) between successive connection attempts to the same peer.
    pub retry_backoff_ms: u64,
    /// Age threshold (ms) beyond which an unconnected, un-attempted peer is
    /// considered stale and eligible for eviction.
    pub stale_threshold_ms: u64,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            max_candidates: 500,
            max_connected: 50,
            rescore_interval_ms: 60_000,
            retry_backoff_ms: 5_000,
            stale_threshold_ms: 300_000,
        }
    }
}

/// Aggregated statistics snapshot returned by [`PeerDiscoveryManager::stats`].
#[derive(Clone, Debug, Default)]
pub struct DiscoveryStats {
    /// Total candidates currently tracked.
    pub total_candidates: usize,
    /// Number of currently connected peers.
    pub connected: usize,
    /// Number of candidates for which at least one attempt was made.
    pub attempted: usize,
    /// Average `connect_score` across all tracked candidates.
    pub avg_score: f64,
    /// Cumulative distinct peers ever added (including evicted ones).
    pub total_discovered: u64,
    /// Cumulative peers that reached `ConnectOutcome::Success`.
    pub total_connected: u64,
    /// Cumulative failed connection attempts (Refused + Timeout + Unreachable).
    pub total_failed: u64,
    /// How many candidates are tracked per discovery method (method name → count).
    pub method_distribution: HashMap<String, usize>,
}

// ---------------------------------------------------------------------------
// Score arithmetic helpers
// ---------------------------------------------------------------------------

const SCORE_MIN: f64 = 0.0;
const SCORE_MAX: f64 = 10.0;
const SCORE_SUCCESS_DELTA: f64 = 0.5;
const SCORE_REFUSED_DELTA: f64 = -0.3;
const SCORE_TIMEOUT_DELTA: f64 = -0.2;
const SCORE_UNREACHABLE_DELTA: f64 = -0.5;
const SCORE_DISCONNECT_DELTA: f64 = -0.1;

#[inline]
fn clamp_score(score: f64) -> f64 {
    score.clamp(SCORE_MIN, SCORE_MAX)
}

// ---------------------------------------------------------------------------
// PeerDiscoveryManager
// ---------------------------------------------------------------------------

/// Multi-strategy peer discovery manager.
///
/// Maintains a scored, deduplicated pool of [`PeerCandidate`]s, drives the
/// connection lifecycle, and provides ranked candidate selection with
/// configurable backoff and stale eviction.
pub struct PeerDiscoveryManager {
    /// Configuration.
    pub config: DiscoveryConfig,
    /// All tracked candidates, keyed by peer-id.
    candidates: HashMap<String, PeerCandidate>,
    /// Set of peer-ids that are currently connected.
    connected: HashSet<String>,
    /// Cumulative peers ever added.
    total_discovered: u64,
    /// Cumulative `ConnectOutcome::Success` events.
    total_connected: u64,
    /// Cumulative failed connection outcomes.
    total_failed: u64,
}

impl PeerDiscoveryManager {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new manager with the given configuration.
    pub fn new(config: DiscoveryConfig) -> Self {
        Self {
            config,
            candidates: HashMap::new(),
            connected: HashSet::new(),
            total_discovered: 0,
            total_connected: 0,
            total_failed: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Candidate lifecycle
    // -----------------------------------------------------------------------

    /// Add or update a candidate.
    ///
    /// * If the peer is already tracked, the new addresses are merged
    ///   (deduplicated) into the existing record and `true` is returned.
    /// * If the candidate pool is at `max_candidates`, the lowest-scoring
    ///   **non-connected** candidate is evicted first.  If no eviction is
    ///   possible (all slots are connected) `false` is returned.
    /// * Otherwise the candidate is inserted and `true` is returned.
    pub fn add_candidate(&mut self, candidate: PeerCandidate) -> bool {
        // Merge path: peer already known.
        if let Some(existing) = self.candidates.get_mut(&candidate.peer_id) {
            for addr in &candidate.addresses {
                if !existing.addresses.contains(addr) {
                    existing.addresses.push(addr.clone());
                }
            }
            return true;
        }

        // Evict if necessary.
        if self.candidates.len() >= self.config.max_candidates && !self.evict_lowest_non_connected()
        {
            return false;
        }

        self.total_discovered += 1;
        self.candidates.insert(candidate.peer_id.clone(), candidate);
        true
    }

    /// Record the outcome of a connection attempt.
    ///
    /// Updates `connect_score`, `attempted`, `connected`, and counters.
    /// Returns `false` when the peer is not found.
    pub fn record_outcome(&mut self, peer_id: &str, outcome: ConnectOutcome, now: u64) -> bool {
        let candidate = match self.candidates.get_mut(peer_id) {
            Some(c) => c,
            None => return false,
        };

        candidate.last_attempt = Some(now);

        match outcome {
            ConnectOutcome::Success => {
                candidate.connected = true;
                candidate.connect_score =
                    clamp_score(candidate.connect_score + SCORE_SUCCESS_DELTA);
                let pid = peer_id.to_owned();
                self.connected.insert(pid);
                self.total_connected += 1;
            }
            ConnectOutcome::Refused => {
                candidate.attempted = true;
                candidate.connect_score =
                    clamp_score(candidate.connect_score + SCORE_REFUSED_DELTA);
                self.total_failed += 1;
            }
            ConnectOutcome::Timeout => {
                candidate.attempted = true;
                candidate.connect_score =
                    clamp_score(candidate.connect_score + SCORE_TIMEOUT_DELTA);
                self.total_failed += 1;
            }
            ConnectOutcome::Unreachable => {
                candidate.attempted = true;
                candidate.connect_score =
                    clamp_score(candidate.connect_score + SCORE_UNREACHABLE_DELTA);
                self.total_failed += 1;
            }
        }

        true
    }

    /// Mark a peer as disconnected.
    ///
    /// The peer remains in the candidate pool with a slightly reduced score so
    /// it can be re-tried later.  Returns `false` when the peer is not found.
    pub fn mark_disconnected(&mut self, peer_id: &str) -> bool {
        let candidate = match self.candidates.get_mut(peer_id) {
            Some(c) => c,
            None => return false,
        };

        candidate.connected = false;
        candidate.connect_score = clamp_score(candidate.connect_score + SCORE_DISCONNECT_DELTA);
        self.connected.remove(peer_id);
        true
    }

    // -----------------------------------------------------------------------
    // Candidate selection
    // -----------------------------------------------------------------------

    /// Return up to `n` candidates that should be tried next.
    ///
    /// Eligible candidates are those that:
    /// * Are **not** currently connected.
    /// * Have `last_attempt == None` **or** `last_attempt` is at least
    ///   `retry_backoff_ms` before `now`.
    ///
    /// Results are sorted by `connect_score` descending.
    pub fn candidates_to_try(&self, n: usize, now: u64) -> Vec<&PeerCandidate> {
        let backoff = self.config.retry_backoff_ms;
        let mut eligible: Vec<&PeerCandidate> = self
            .candidates
            .values()
            .filter(|c| {
                if c.connected {
                    return false;
                }
                match c.last_attempt {
                    None => true,
                    Some(t) => now.saturating_sub(t) >= backoff,
                }
            })
            .collect();

        eligible.sort_by(|a, b| {
            b.connect_score
                .partial_cmp(&a.connect_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        eligible.truncate(n);
        eligible
    }

    /// Evict non-connected candidates that are considered stale.
    ///
    /// A candidate is stale when **both**:
    /// * `discovered_at < now - stale_threshold_ms`
    /// * `last_attempt < now - stale_threshold_ms` *or* `last_attempt == None`
    ///
    /// Returns the number of evicted candidates.
    pub fn evict_stale(&mut self, now: u64) -> usize {
        let threshold = self.config.stale_threshold_ms;
        let cutoff = now.saturating_sub(threshold);

        let stale_ids: Vec<String> = self
            .candidates
            .iter()
            .filter(|(_, c)| {
                if c.connected {
                    return false;
                }
                let old_discovery = c.discovered_at < cutoff;
                let old_attempt = match c.last_attempt {
                    None => true,
                    Some(t) => t < cutoff,
                };
                old_discovery && old_attempt
            })
            .map(|(k, _)| k.clone())
            .collect();

        let count = stale_ids.len();
        for id in stale_ids {
            self.candidates.remove(&id);
        }
        count
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// All currently connected peer candidates.
    pub fn connected_peers(&self) -> Vec<&PeerCandidate> {
        self.candidates.values().filter(|c| c.connected).collect()
    }

    /// Total number of tracked candidates.
    pub fn peer_count(&self) -> usize {
        self.candidates.len()
    }

    /// Number of currently connected peers.
    pub fn connected_count(&self) -> usize {
        self.connected.len()
    }

    /// The highest-scoring candidate that is **not** currently connected.
    pub fn best_candidate(&self) -> Option<&PeerCandidate> {
        self.candidates
            .values()
            .filter(|c| !c.connected)
            .max_by(|a, b| {
                a.connect_score
                    .partial_cmp(&b.connect_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// All candidates discovered via the given method.
    pub fn candidates_by_method(&self, method: &DiscoveryMethod) -> Vec<&PeerCandidate> {
        self.candidates
            .values()
            .filter(|c| &c.discovered_via == method)
            .collect()
    }

    /// A statistics snapshot covering the current pool state.
    pub fn stats(&self) -> DiscoveryStats {
        let total_candidates = self.candidates.len();
        let connected = self.connected.len();
        let attempted = self.candidates.values().filter(|c| c.attempted).count();

        let avg_score = if total_candidates == 0 {
            0.0
        } else {
            let sum: f64 = self.candidates.values().map(|c| c.connect_score).sum();
            sum / total_candidates as f64
        };

        let mut method_distribution: HashMap<String, usize> = HashMap::new();
        for c in self.candidates.values() {
            *method_distribution
                .entry(c.discovered_via.to_string())
                .or_insert(0) += 1;
        }

        DiscoveryStats {
            total_candidates,
            connected,
            attempted,
            avg_score,
            total_discovered: self.total_discovered,
            total_connected: self.total_connected,
            total_failed: self.total_failed,
            method_distribution,
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Evict the single lowest-scoring non-connected candidate.
    /// Returns `true` if an eviction occurred.
    fn evict_lowest_non_connected(&mut self) -> bool {
        let victim_id: Option<String> = self
            .candidates
            .iter()
            .filter(|(_, c)| !c.connected)
            .min_by(|(_, a), (_, b)| {
                a.connect_score
                    .partial_cmp(&b.connect_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(k, _)| k.clone());

        if let Some(id) = victim_id {
            self.candidates.remove(&id);
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::peer_discovery_manager::{
        ConnectOutcome, DiscoveryConfig, DiscoveryMethod, PeerCandidate, PeerDiscoveryManager,
    };

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn default_config() -> DiscoveryConfig {
        DiscoveryConfig::default()
    }

    fn make_candidate(id: &str, method: DiscoveryMethod, discovered_at: u64) -> PeerCandidate {
        PeerCandidate::new(
            id,
            vec![format!("/ip4/1.2.3.4/tcp/{id}")],
            method,
            discovered_at,
        )
    }

    fn make_candidate_with_addrs(
        id: &str,
        addrs: Vec<String>,
        method: DiscoveryMethod,
        discovered_at: u64,
    ) -> PeerCandidate {
        PeerCandidate::new(id, addrs, method, discovered_at)
    }

    // -----------------------------------------------------------------------
    // Construction / defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_empty_state() {
        let mgr = PeerDiscoveryManager::new(default_config());
        assert_eq!(mgr.peer_count(), 0);
        assert_eq!(mgr.connected_count(), 0);
    }

    #[test]
    fn test_default_config_values() {
        let cfg = DiscoveryConfig::default();
        assert_eq!(cfg.max_candidates, 500);
        assert_eq!(cfg.max_connected, 50);
        assert_eq!(cfg.rescore_interval_ms, 60_000);
        assert_eq!(cfg.retry_backoff_ms, 5_000);
        assert_eq!(cfg.stale_threshold_ms, 300_000);
    }

    #[test]
    fn test_peer_candidate_new_defaults() {
        let c = make_candidate("p1", DiscoveryMethod::Bootstrap, 1000);
        assert_eq!(c.peer_id, "p1");
        assert!((c.connect_score - 1.0).abs() < f64::EPSILON);
        assert!(!c.attempted);
        assert!(!c.connected);
        assert!(c.last_attempt.is_none());
    }

    // -----------------------------------------------------------------------
    // add_candidate
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_candidate_returns_true() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        let c = make_candidate("p1", DiscoveryMethod::Bootstrap, 1000);
        assert!(mgr.add_candidate(c));
        assert_eq!(mgr.peer_count(), 1);
    }

    #[test]
    fn test_add_duplicate_merges_addresses() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        let c1 = make_candidate_with_addrs(
            "p1",
            vec!["/ip4/1.0.0.1/tcp/4001".to_string()],
            DiscoveryMethod::Bootstrap,
            1000,
        );
        let c2 = make_candidate_with_addrs(
            "p1",
            vec![
                "/ip4/1.0.0.1/tcp/4001".to_string(), // duplicate
                "/ip4/2.0.0.2/tcp/4001".to_string(), // new
            ],
            DiscoveryMethod::Mdns,
            2000,
        );
        mgr.add_candidate(c1);
        let added = mgr.add_candidate(c2);
        assert!(added);
        assert_eq!(mgr.peer_count(), 1);
        let stored = mgr.candidates.get("p1").expect("should exist");
        assert_eq!(stored.addresses.len(), 2);
    }

    #[test]
    fn test_add_duplicate_no_address_duplication() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        let addr = "/ip4/1.0.0.1/tcp/4001".to_string();
        mgr.add_candidate(make_candidate_with_addrs(
            "p1",
            vec![addr.clone()],
            DiscoveryMethod::Bootstrap,
            0,
        ));
        mgr.add_candidate(make_candidate_with_addrs(
            "p1",
            vec![addr.clone()],
            DiscoveryMethod::Bootstrap,
            0,
        ));
        assert_eq!(
            mgr.candidates.get("p1").expect("present").addresses.len(),
            1
        );
    }

    #[test]
    fn test_add_candidate_at_capacity_evicts_lowest() {
        let cfg = DiscoveryConfig {
            max_candidates: 3,
            ..Default::default()
        };
        let mut mgr = PeerDiscoveryManager::new(cfg);

        let mut c_low = make_candidate("low", DiscoveryMethod::Bootstrap, 0);
        c_low.connect_score = 0.1;

        let mut c_mid = make_candidate("mid", DiscoveryMethod::Bootstrap, 0);
        c_mid.connect_score = 1.0;

        let mut c_high = make_candidate("high", DiscoveryMethod::Bootstrap, 0);
        c_high.connect_score = 5.0;

        mgr.add_candidate(c_low);
        mgr.add_candidate(c_mid);
        mgr.add_candidate(c_high);
        assert_eq!(mgr.peer_count(), 3);

        // Adding a 4th should evict "low"
        let c_new = make_candidate("new", DiscoveryMethod::Dht, 0);
        assert!(mgr.add_candidate(c_new));
        assert_eq!(mgr.peer_count(), 3);
        assert!(!mgr.candidates.contains_key("low"));
        assert!(mgr.candidates.contains_key("new"));
    }

    #[test]
    fn test_add_candidate_at_capacity_all_connected_returns_false() {
        let cfg = DiscoveryConfig {
            max_candidates: 2,
            ..Default::default()
        };
        let mut mgr = PeerDiscoveryManager::new(cfg);

        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        mgr.add_candidate(make_candidate("p2", DiscoveryMethod::Bootstrap, 0));

        // Mark both connected so there is nothing to evict.
        mgr.record_outcome("p1", ConnectOutcome::Success, 100);
        mgr.record_outcome("p2", ConnectOutcome::Success, 100);

        let c_new = make_candidate("p3", DiscoveryMethod::Dht, 0);
        assert!(!mgr.add_candidate(c_new));
        assert_eq!(mgr.peer_count(), 2);
    }

    #[test]
    fn test_total_discovered_increments() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        mgr.add_candidate(make_candidate("p2", DiscoveryMethod::Bootstrap, 0));
        // Duplicate should NOT increment.
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        assert_eq!(mgr.total_discovered, 2);
    }

    // -----------------------------------------------------------------------
    // record_outcome
    // -----------------------------------------------------------------------

    #[test]
    fn test_record_outcome_success_marks_connected() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        assert!(mgr.record_outcome("p1", ConnectOutcome::Success, 1000));
        let c = mgr.candidates.get("p1").expect("present");
        assert!(c.connected);
        assert_eq!(mgr.connected_count(), 1);
        assert_eq!(mgr.total_connected, 1);
    }

    #[test]
    fn test_record_outcome_success_increases_score() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        mgr.record_outcome("p1", ConnectOutcome::Success, 1000);
        let score = mgr.candidates.get("p1").expect("present").connect_score;
        assert!((score - 1.5).abs() < 1e-10, "expected 1.5, got {score}");
    }

    #[test]
    fn test_record_outcome_success_score_capped_at_10() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        let mut c = make_candidate("p1", DiscoveryMethod::Bootstrap, 0);
        c.connect_score = 9.9;
        mgr.add_candidate(c);
        mgr.record_outcome("p1", ConnectOutcome::Success, 1000);
        let score = mgr.candidates.get("p1").expect("present").connect_score;
        assert!(score <= 10.0, "score must not exceed 10.0");
        assert!((score - 10.0).abs() < 1e-10, "expected 10.0, got {score}");
    }

    #[test]
    fn test_record_outcome_refused_decreases_score() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        mgr.record_outcome("p1", ConnectOutcome::Refused, 1000);
        let c = mgr.candidates.get("p1").expect("present");
        assert!(c.attempted);
        assert!((c.connect_score - 0.7).abs() < 1e-10);
        assert_eq!(mgr.total_failed, 1);
    }

    #[test]
    fn test_record_outcome_timeout_decreases_score() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        mgr.record_outcome("p1", ConnectOutcome::Timeout, 1000);
        let c = mgr.candidates.get("p1").expect("present");
        assert!((c.connect_score - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_record_outcome_unreachable_decreases_score() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        mgr.record_outcome("p1", ConnectOutcome::Unreachable, 1000);
        let c = mgr.candidates.get("p1").expect("present");
        assert!((c.connect_score - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_record_outcome_score_floor_at_zero() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        let mut c = make_candidate("p1", DiscoveryMethod::Bootstrap, 0);
        c.connect_score = 0.1;
        mgr.add_candidate(c);
        mgr.record_outcome("p1", ConnectOutcome::Unreachable, 1000);
        let score = mgr.candidates.get("p1").expect("present").connect_score;
        assert!(score >= 0.0, "score must not go below 0");
        assert!((score - 0.0).abs() < 1e-10, "expected 0.0, got {score}");
    }

    #[test]
    fn test_record_outcome_unknown_peer_returns_false() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        assert!(!mgr.record_outcome("ghost", ConnectOutcome::Success, 0));
    }

    #[test]
    fn test_record_outcome_sets_last_attempt() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        mgr.record_outcome("p1", ConnectOutcome::Refused, 9999);
        assert_eq!(
            mgr.candidates.get("p1").expect("present").last_attempt,
            Some(9999)
        );
    }

    // -----------------------------------------------------------------------
    // mark_disconnected
    // -----------------------------------------------------------------------

    #[test]
    fn test_mark_disconnected_clears_connected_flag() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        mgr.record_outcome("p1", ConnectOutcome::Success, 100);
        assert!(mgr.mark_disconnected("p1"));
        assert!(!mgr.candidates.get("p1").expect("present").connected);
        assert_eq!(mgr.connected_count(), 0);
    }

    #[test]
    fn test_mark_disconnected_reduces_score() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        mgr.record_outcome("p1", ConnectOutcome::Success, 100);
        let score_before = mgr.candidates.get("p1").expect("present").connect_score;
        mgr.mark_disconnected("p1");
        let score_after = mgr.candidates.get("p1").expect("present").connect_score;
        assert!(score_after < score_before);
    }

    #[test]
    fn test_mark_disconnected_score_floor_zero() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        let mut c = make_candidate("p1", DiscoveryMethod::Bootstrap, 0);
        c.connect_score = 0.05;
        c.connected = true;
        mgr.add_candidate(c);
        mgr.connected.insert("p1".to_owned());
        mgr.mark_disconnected("p1");
        assert!(mgr.candidates.get("p1").expect("present").connect_score >= 0.0);
    }

    #[test]
    fn test_mark_disconnected_unknown_peer_returns_false() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        assert!(!mgr.mark_disconnected("ghost"));
    }

    // -----------------------------------------------------------------------
    // candidates_to_try
    // -----------------------------------------------------------------------

    #[test]
    fn test_candidates_to_try_excludes_connected() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        mgr.add_candidate(make_candidate("p2", DiscoveryMethod::Bootstrap, 0));
        mgr.record_outcome("p1", ConnectOutcome::Success, 500);

        let candidates = mgr.candidates_to_try(10, 1000);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].peer_id, "p2");
    }

    #[test]
    fn test_candidates_to_try_respects_backoff() {
        let cfg = DiscoveryConfig {
            retry_backoff_ms: 5_000,
            ..Default::default()
        };
        let mut mgr = PeerDiscoveryManager::new(cfg);
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        // Attempt at t=1000 → not eligible until t=6000
        mgr.record_outcome("p1", ConnectOutcome::Refused, 1000);

        let too_early = mgr.candidates_to_try(10, 5_999);
        assert!(too_early.is_empty(), "should be in backoff");

        let after_backoff = mgr.candidates_to_try(10, 6_001);
        assert_eq!(after_backoff.len(), 1);
    }

    #[test]
    fn test_candidates_to_try_sorts_by_score_descending() {
        let mut mgr = PeerDiscoveryManager::new(default_config());

        let mut c_low = make_candidate("low", DiscoveryMethod::Bootstrap, 0);
        c_low.connect_score = 0.5;
        let mut c_mid = make_candidate("mid", DiscoveryMethod::Bootstrap, 0);
        c_mid.connect_score = 2.0;
        let mut c_high = make_candidate("high", DiscoveryMethod::Bootstrap, 0);
        c_high.connect_score = 7.0;

        mgr.add_candidate(c_low);
        mgr.add_candidate(c_mid);
        mgr.add_candidate(c_high);

        let result = mgr.candidates_to_try(3, 100_000);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].peer_id, "high");
        assert_eq!(result[1].peer_id, "mid");
        assert_eq!(result[2].peer_id, "low");
    }

    #[test]
    fn test_candidates_to_try_truncates_to_n() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        for i in 0..10 {
            mgr.add_candidate(make_candidate(&format!("p{i}"), DiscoveryMethod::Dht, 0));
        }
        assert_eq!(mgr.candidates_to_try(3, 100_000).len(), 3);
    }

    #[test]
    fn test_candidates_to_try_never_attempted_always_eligible() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        // now=0 with backoff=5000: no last_attempt → eligible
        let result = mgr.candidates_to_try(10, 0);
        assert_eq!(result.len(), 1);
    }

    // -----------------------------------------------------------------------
    // evict_stale
    // -----------------------------------------------------------------------

    #[test]
    fn test_evict_stale_removes_old_unattempted() {
        let cfg = DiscoveryConfig {
            stale_threshold_ms: 10_000,
            ..Default::default()
        };
        let mut mgr = PeerDiscoveryManager::new(cfg);
        mgr.add_candidate(make_candidate("old", DiscoveryMethod::Bootstrap, 0));
        mgr.add_candidate(make_candidate("recent", DiscoveryMethod::Bootstrap, 5_000));

        // now = 20_000 → cutoff = 10_000
        // "old" discovered_at=0 < 10_000 and no last_attempt → stale
        // "recent" discovered_at=5_000 < 10_000 and no last_attempt → stale too
        let evicted = mgr.evict_stale(20_000);
        assert_eq!(evicted, 2);
        assert_eq!(mgr.peer_count(), 0);
    }

    #[test]
    fn test_evict_stale_keeps_recently_attempted() {
        let cfg = DiscoveryConfig {
            stale_threshold_ms: 10_000,
            ..Default::default()
        };
        let mut mgr = PeerDiscoveryManager::new(cfg);
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        // Attempt at t=15_000 → last_attempt=15_000, cutoff=10_000 → NOT stale
        mgr.record_outcome("p1", ConnectOutcome::Refused, 15_000);

        let evicted = mgr.evict_stale(20_000);
        assert_eq!(evicted, 0);
        assert_eq!(mgr.peer_count(), 1);
    }

    #[test]
    fn test_evict_stale_never_evicts_connected() {
        let cfg = DiscoveryConfig {
            stale_threshold_ms: 1,
            ..Default::default()
        };
        let mut mgr = PeerDiscoveryManager::new(cfg);
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        mgr.record_outcome("p1", ConnectOutcome::Success, 0);

        // Even though discovered_at=0 and stale_threshold=1, connected peers
        // must never be evicted.
        let evicted = mgr.evict_stale(100_000);
        assert_eq!(evicted, 0);
        assert_eq!(mgr.peer_count(), 1);
    }

    #[test]
    fn test_evict_stale_returns_count() {
        let cfg = DiscoveryConfig {
            stale_threshold_ms: 100,
            ..Default::default()
        };
        let mut mgr = PeerDiscoveryManager::new(cfg);
        for i in 0..5 {
            mgr.add_candidate(make_candidate(&format!("p{i}"), DiscoveryMethod::Dht, 0));
        }
        let evicted = mgr.evict_stale(200);
        assert_eq!(evicted, 5);
    }

    // -----------------------------------------------------------------------
    // connected_peers
    // -----------------------------------------------------------------------

    #[test]
    fn test_connected_peers_only_returns_connected() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        mgr.add_candidate(make_candidate("p2", DiscoveryMethod::Bootstrap, 0));
        mgr.record_outcome("p1", ConnectOutcome::Success, 100);

        let connected = mgr.connected_peers();
        assert_eq!(connected.len(), 1);
        assert_eq!(connected[0].peer_id, "p1");
    }

    #[test]
    fn test_connected_peers_empty_initially() {
        let mgr = PeerDiscoveryManager::new(default_config());
        assert!(mgr.connected_peers().is_empty());
    }

    // -----------------------------------------------------------------------
    // best_candidate
    // -----------------------------------------------------------------------

    #[test]
    fn test_best_candidate_returns_highest_score() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        let mut c_low = make_candidate("low", DiscoveryMethod::Bootstrap, 0);
        c_low.connect_score = 0.5;
        let mut c_high = make_candidate("high", DiscoveryMethod::Bootstrap, 0);
        c_high.connect_score = 8.0;
        mgr.add_candidate(c_low);
        mgr.add_candidate(c_high);

        let best = mgr.best_candidate().expect("should have a best");
        assert_eq!(best.peer_id, "high");
    }

    #[test]
    fn test_best_candidate_excludes_connected() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        let mut c1 = make_candidate("p1", DiscoveryMethod::Bootstrap, 0);
        c1.connect_score = 9.0;
        let mut c2 = make_candidate("p2", DiscoveryMethod::Bootstrap, 0);
        c2.connect_score = 3.0;
        mgr.add_candidate(c1);
        mgr.add_candidate(c2);
        mgr.record_outcome("p1", ConnectOutcome::Success, 100);

        let best = mgr.best_candidate().expect("p2 should be best");
        assert_eq!(best.peer_id, "p2");
    }

    #[test]
    fn test_best_candidate_returns_none_when_all_connected() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        mgr.record_outcome("p1", ConnectOutcome::Success, 100);
        assert!(mgr.best_candidate().is_none());
    }

    #[test]
    fn test_best_candidate_empty_manager() {
        let mgr = PeerDiscoveryManager::new(default_config());
        assert!(mgr.best_candidate().is_none());
    }

    // -----------------------------------------------------------------------
    // candidates_by_method
    // -----------------------------------------------------------------------

    #[test]
    fn test_candidates_by_method_filters_correctly() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        mgr.add_candidate(make_candidate("b1", DiscoveryMethod::Bootstrap, 0));
        mgr.add_candidate(make_candidate("b2", DiscoveryMethod::Bootstrap, 0));
        mgr.add_candidate(make_candidate("m1", DiscoveryMethod::Mdns, 0));

        let bootstrap_peers = mgr.candidates_by_method(&DiscoveryMethod::Bootstrap);
        assert_eq!(bootstrap_peers.len(), 2);

        let mdns_peers = mgr.candidates_by_method(&DiscoveryMethod::Mdns);
        assert_eq!(mdns_peers.len(), 1);

        let dht_peers = mgr.candidates_by_method(&DiscoveryMethod::Dht);
        assert!(dht_peers.is_empty());
    }

    #[test]
    fn test_candidates_by_method_all_variants() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        mgr.add_candidate(make_candidate("b", DiscoveryMethod::Bootstrap, 0));
        mgr.add_candidate(make_candidate("m", DiscoveryMethod::Mdns, 0));
        mgr.add_candidate(make_candidate("d", DiscoveryMethod::Dht, 0));
        mgr.add_candidate(make_candidate("px", DiscoveryMethod::PeerExchange, 0));
        mgr.add_candidate(make_candidate("man", DiscoveryMethod::Manual, 0));

        for method in [
            DiscoveryMethod::Bootstrap,
            DiscoveryMethod::Mdns,
            DiscoveryMethod::Dht,
            DiscoveryMethod::PeerExchange,
            DiscoveryMethod::Manual,
        ] {
            assert_eq!(mgr.candidates_by_method(&method).len(), 1);
        }
    }

    // -----------------------------------------------------------------------
    // stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_initial_empty() {
        let mgr = PeerDiscoveryManager::new(default_config());
        let s = mgr.stats();
        assert_eq!(s.total_candidates, 0);
        assert_eq!(s.connected, 0);
        assert_eq!(s.attempted, 0);
        assert!((s.avg_score - 0.0).abs() < f64::EPSILON);
        assert_eq!(s.total_discovered, 0);
    }

    #[test]
    fn test_stats_avg_score() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        let mut c1 = make_candidate("p1", DiscoveryMethod::Bootstrap, 0);
        c1.connect_score = 2.0;
        let mut c2 = make_candidate("p2", DiscoveryMethod::Bootstrap, 0);
        c2.connect_score = 4.0;
        mgr.add_candidate(c1);
        mgr.add_candidate(c2);
        let s = mgr.stats();
        assert!((s.avg_score - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_method_distribution() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        mgr.add_candidate(make_candidate("b1", DiscoveryMethod::Bootstrap, 0));
        mgr.add_candidate(make_candidate("b2", DiscoveryMethod::Bootstrap, 0));
        mgr.add_candidate(make_candidate("m1", DiscoveryMethod::Mdns, 0));
        let s = mgr.stats();
        assert_eq!(
            s.method_distribution.get("Bootstrap").copied().unwrap_or(0),
            2
        );
        assert_eq!(s.method_distribution.get("Mdns").copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_stats_totals_after_lifecycle() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        mgr.add_candidate(make_candidate("p2", DiscoveryMethod::Bootstrap, 0));
        mgr.record_outcome("p1", ConnectOutcome::Success, 100);
        mgr.record_outcome("p2", ConnectOutcome::Refused, 100);

        let s = mgr.stats();
        assert_eq!(s.total_discovered, 2);
        assert_eq!(s.total_connected, 1);
        assert_eq!(s.total_failed, 1);
        assert_eq!(s.connected, 1);
        assert_eq!(s.attempted, 1); // only p2 has attempted=true
    }

    #[test]
    fn test_stats_attempted_count() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        for i in 0..5 {
            mgr.add_candidate(make_candidate(
                &format!("p{i}"),
                DiscoveryMethod::Bootstrap,
                0,
            ));
        }
        mgr.record_outcome("p0", ConnectOutcome::Refused, 100);
        mgr.record_outcome("p1", ConnectOutcome::Timeout, 100);
        let s = mgr.stats();
        assert_eq!(s.attempted, 2);
    }

    // -----------------------------------------------------------------------
    // DiscoveryMethod display / hash
    // -----------------------------------------------------------------------

    #[test]
    fn test_discovery_method_display() {
        assert_eq!(DiscoveryMethod::Bootstrap.to_string(), "Bootstrap");
        assert_eq!(DiscoveryMethod::Mdns.to_string(), "Mdns");
        assert_eq!(DiscoveryMethod::Dht.to_string(), "Dht");
        assert_eq!(DiscoveryMethod::PeerExchange.to_string(), "PeerExchange");
        assert_eq!(DiscoveryMethod::Manual.to_string(), "Manual");
    }

    #[test]
    fn test_discovery_method_in_hashmap() {
        let mut map: HashMap<DiscoveryMethod, u32> = HashMap::new();
        map.insert(DiscoveryMethod::Bootstrap, 1);
        map.insert(DiscoveryMethod::Mdns, 2);
        assert_eq!(map[&DiscoveryMethod::Bootstrap], 1);
        assert_eq!(map[&DiscoveryMethod::Mdns], 2);
    }

    // -----------------------------------------------------------------------
    // Edge-cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_successes_cap_score() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        for i in 0..30 {
            mgr.record_outcome("p1", ConnectOutcome::Success, i * 100);
        }
        let score = mgr.candidates.get("p1").expect("present").connect_score;
        assert!(score <= 10.0, "score must never exceed 10.0");
    }

    #[test]
    fn test_multiple_failures_floor_score() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        for i in 0..30 {
            mgr.record_outcome("p1", ConnectOutcome::Unreachable, i * 100);
        }
        let score = mgr.candidates.get("p1").expect("present").connect_score;
        assert!(score >= 0.0, "score must never go below 0.0");
    }

    #[test]
    fn test_reconnect_after_disconnect_increments_total_connected() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        mgr.record_outcome("p1", ConnectOutcome::Success, 100);
        mgr.mark_disconnected("p1");
        mgr.record_outcome("p1", ConnectOutcome::Success, 200);
        assert_eq!(mgr.total_connected, 2);
    }

    #[test]
    fn test_peer_count_and_connected_count_consistent() {
        let mut mgr = PeerDiscoveryManager::new(default_config());
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        mgr.add_candidate(make_candidate("p2", DiscoveryMethod::Bootstrap, 0));
        mgr.add_candidate(make_candidate("p3", DiscoveryMethod::Bootstrap, 0));
        mgr.record_outcome("p1", ConnectOutcome::Success, 100);
        mgr.record_outcome("p2", ConnectOutcome::Success, 100);

        assert_eq!(mgr.peer_count(), 3);
        assert_eq!(mgr.connected_count(), 2);
    }

    #[test]
    fn test_evict_stale_does_not_touch_non_stale() {
        let cfg = DiscoveryConfig {
            stale_threshold_ms: 10_000,
            ..Default::default()
        };
        let mut mgr = PeerDiscoveryManager::new(cfg);
        // Discovered at 15_000 → not old enough when now=20_000, cutoff=10_000
        mgr.add_candidate(make_candidate("fresh", DiscoveryMethod::Bootstrap, 15_000));
        mgr.add_candidate(make_candidate("stale", DiscoveryMethod::Bootstrap, 0));

        let evicted = mgr.evict_stale(20_000);
        assert_eq!(evicted, 1);
        assert!(mgr.candidates.contains_key("fresh"));
        assert!(!mgr.candidates.contains_key("stale"));
    }

    #[test]
    fn test_no_candidates_to_try_when_all_in_backoff() {
        let cfg = DiscoveryConfig {
            retry_backoff_ms: 10_000,
            ..Default::default()
        };
        let mut mgr = PeerDiscoveryManager::new(cfg);
        mgr.add_candidate(make_candidate("p1", DiscoveryMethod::Bootstrap, 0));
        mgr.add_candidate(make_candidate("p2", DiscoveryMethod::Bootstrap, 0));
        mgr.record_outcome("p1", ConnectOutcome::Refused, 5_000);
        mgr.record_outcome("p2", ConnectOutcome::Timeout, 5_000);

        // at t=6_000 both are still in backoff
        assert!(mgr.candidates_to_try(10, 6_000).is_empty());
    }
}
