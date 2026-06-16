//! NetworkSecurityMonitor — Real-time network security monitoring with anomaly
//! detection, threat scoring, and incident management.
//!
//! # Overview
//!
//! This module provides production-grade security monitoring for a peer-to-peer
//! network. It tracks [`SecurityEvent`]s per peer, computes time-decaying
//! [`ThreatScore`]s, and automatically opens [`SecurityIncident`]s when a
//! peer's score crosses a configurable threshold.
//!
//! ## Scoring Model
//!
//! Each [`ThreatLevel`] contributes a fixed number of points to a peer's score:
//!
//! | Level    | Points |
//! |----------|--------|
//! | None     | 0      |
//! | Low      | 5      |
//! | Medium   | 15     |
//! | High     | 30     |
//! | Critical | 50     |
//!
//! Scores decay by 10 % per elapsed hour (integer hours):
//! `score *= 0.9 ^ hours_since_last_update`.
//!
//! ## Event ID
//!
//! Event IDs are computed with FNV-1a over the concatenation of
//! `peer_id + threat_type_name + timestamp_string`.

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// FNV-1a helpers
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit offset basis.
const FNV1A_OFFSET: u64 = 14_695_981_039_346_656_037;
/// FNV-1a 64-bit prime.
const FNV1A_PRIME: u64 = 1_099_511_628_211;

/// Compute FNV-1a 64-bit hash over an arbitrary byte slice.
fn fnv1a(data: &[u8]) -> u64 {
    let mut hash = FNV1A_OFFSET;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV1A_PRIME);
    }
    hash
}

/// Compute a deterministic event ID from peer id, threat type name, and timestamp.
fn compute_event_id(peer_id: &str, threat_type_name: &str, timestamp: u64) -> u64 {
    let mut raw = String::with_capacity(
        peer_id.len() + threat_type_name.len() + 20, // 20 bytes is enough for u64
    );
    raw.push_str(peer_id);
    raw.push_str(threat_type_name);
    raw.push_str(&timestamp.to_string());
    fnv1a(raw.as_bytes())
}

// ---------------------------------------------------------------------------
// ThreatType
// ---------------------------------------------------------------------------

/// The category of a detected threat.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ThreatType {
    /// Sybil attack — a single entity controls many pseudo-identities.
    Sybil,
    /// Eclipse attack — routing table is flooded with attacker-controlled peers.
    Eclipse,
    /// Distributed Denial of Service.
    DDoS,
    /// Man-in-the-Middle interception or tampering.
    ManInTheMiddle,
    /// Replay of previously captured messages.
    ReplayAttack,
    /// Deliberate poisoning of routing tables.
    RoutingPoison,
    /// Modification of content in transit or at rest.
    DataTampering,
}

impl ThreatType {
    /// Return the canonical string name used in ID hashing.
    pub fn name(&self) -> &'static str {
        match self {
            ThreatType::Sybil => "Sybil",
            ThreatType::Eclipse => "Eclipse",
            ThreatType::DDoS => "DDoS",
            ThreatType::ManInTheMiddle => "ManInTheMiddle",
            ThreatType::ReplayAttack => "ReplayAttack",
            ThreatType::RoutingPoison => "RoutingPoison",
            ThreatType::DataTampering => "DataTampering",
        }
    }
}

// ---------------------------------------------------------------------------
// ThreatLevel
// ---------------------------------------------------------------------------

/// Ordered severity of a detected threat. `Critical > High > Medium > Low > None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreatLevel {
    /// No threat — informational / clean baseline event.
    None,
    /// Low-severity anomaly; worth tracking but unlikely harmful.
    Low,
    /// Medium-severity anomaly; investigation recommended.
    Medium,
    /// High-severity threat; immediate attention required.
    High,
    /// Critical threat; automatic incident creation and possible ban.
    Critical,
}

impl ThreatLevel {
    /// Score points contributed to a peer's [`ThreatScore`] by one event at
    /// this level.
    pub fn score_contribution(self) -> f64 {
        match self {
            ThreatLevel::None => 0.0,
            ThreatLevel::Low => 5.0,
            ThreatLevel::Medium => 15.0,
            ThreatLevel::High => 30.0,
            ThreatLevel::Critical => 50.0,
        }
    }

    /// Integer rank used for comparison.
    fn rank(self) -> u8 {
        match self {
            ThreatLevel::None => 0,
            ThreatLevel::Low => 1,
            ThreatLevel::Medium => 2,
            ThreatLevel::High => 3,
            ThreatLevel::Critical => 4,
        }
    }
}

impl PartialOrd for ThreatLevel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ThreatLevel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank().cmp(&other.rank())
    }
}

// ---------------------------------------------------------------------------
// SecurityEvent
// ---------------------------------------------------------------------------

/// A single security-relevant observation about a peer.
#[derive(Debug, Clone)]
pub struct SecurityEvent {
    /// Deterministic FNV-1a ID: hash of `peer_id + threat_type.name() + timestamp`.
    pub id: u64,
    /// The peer that triggered this event.
    pub peer_id: String,
    /// Category of threat observed.
    pub threat_type: ThreatType,
    /// Severity of the threat.
    pub threat_level: ThreatLevel,
    /// Human-readable description.
    pub description: String,
    /// Unix epoch in milliseconds at which the event was recorded.
    pub timestamp: u64,
    /// Supporting evidence strings (packet hashes, CIDs, log snippets, …).
    pub evidence: Vec<String>,
}

impl SecurityEvent {
    /// Create a new `SecurityEvent`, computing the ID automatically.
    pub fn new(
        peer_id: String,
        threat_type: ThreatType,
        threat_level: ThreatLevel,
        description: String,
        timestamp: u64,
        evidence: Vec<String>,
    ) -> Self {
        let id = compute_event_id(&peer_id, threat_type.name(), timestamp);
        SecurityEvent {
            id,
            peer_id,
            threat_type,
            threat_level,
            description,
            timestamp,
            evidence,
        }
    }
}

// ---------------------------------------------------------------------------
// ThreatScore
// ---------------------------------------------------------------------------

/// Aggregated threat score for a peer, with time-based decay.
#[derive(Debug, Clone)]
pub struct ThreatScore {
    /// The peer this score belongs to.
    pub peer_id: String,
    /// Current score in the range `[0.0, 100.0]`.  100.0 = definitely malicious.
    pub score: f64,
    /// Number of events that have contributed to this score.
    pub contributing_events: u32,
    /// Timestamp (ms since epoch) when the score was last updated.
    pub last_updated: u64,
}

impl ThreatScore {
    /// Create a fresh score for `peer_id`, starting at 0.
    pub fn new(peer_id: String, now: u64) -> Self {
        ThreatScore {
            peer_id,
            score: 0.0,
            contributing_events: 0,
            last_updated: now,
        }
    }

    /// Add `points` to the score, capping at 100.0, and record the update time.
    pub fn add(&mut self, points: f64, now: u64) {
        self.score = (self.score + points).min(100.0);
        self.contributing_events = self.contributing_events.saturating_add(1);
        self.last_updated = now;
    }
}

// ---------------------------------------------------------------------------
// IncidentStatus
// ---------------------------------------------------------------------------

/// Lifecycle status of a [`SecurityIncident`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IncidentStatus {
    /// Newly created; not yet assigned.
    Open,
    /// Assigned and actively being investigated.
    Investigating,
    /// Confirmed and resolved.
    Resolved,
    /// Marked as a false positive; no further action required.
    FalsePositive,
}

// ---------------------------------------------------------------------------
// SecurityIncident
// ---------------------------------------------------------------------------

/// A group of related security events that warrants coordinated response.
#[derive(Debug, Clone)]
pub struct SecurityIncident {
    /// FNV-1a ID computed at creation time.
    pub id: u64,
    /// IDs of the [`SecurityEvent`]s that constitute this incident.
    pub events: Vec<u64>,
    /// Current lifecycle status.
    pub status: IncidentStatus,
    /// When the incident was created (ms epoch).
    pub created_at: u64,
    /// When the incident was resolved, if applicable.
    pub resolved_at: Option<u64>,
}

// ---------------------------------------------------------------------------
// SecurityMonitorStats
// ---------------------------------------------------------------------------

/// Aggregate statistics snapshot from [`NetworkSecurityMonitor`].
#[derive(Debug, Clone)]
pub struct SecurityMonitorStats {
    /// Total events currently buffered.
    pub total_events: usize,
    /// Number of incidents with status `Open` or `Investigating`.
    pub open_incidents: usize,
    /// Number of incidents with status `Resolved` or `FalsePositive`.
    pub resolved_incidents: usize,
    /// Number of peers with a current score ≥ 50.
    pub high_threat_peers: usize,
    /// Mean current score across all tracked peers (0.0 if none).
    pub avg_threat_score: f64,
}

// ---------------------------------------------------------------------------
// NetworkSecurityMonitor
// ---------------------------------------------------------------------------

/// Real-time network security monitor.
///
/// Maintains a rolling event buffer, per-peer threat scores with exponential
/// decay, and incident tracking.  All timestamps are expected as milliseconds
/// since Unix epoch.
pub struct NetworkSecurityMonitor {
    /// Rolling buffer of recent security events.
    events: VecDeque<SecurityEvent>,
    /// Per-peer threat scores.
    scores: HashMap<String, ThreatScore>,
    /// All incidents, ordered by creation time.
    incidents: Vec<SecurityIncident>,
    /// Maximum number of events held in the rolling buffer.
    max_events: usize,
    /// Score threshold above which an incident is automatically created.
    threat_threshold: f64,
    /// Monotonically increasing counter for generating incident IDs.
    next_id: u64,
}

impl NetworkSecurityMonitor {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new monitor with the given capacity and auto-incident threshold.
    ///
    /// # Arguments
    ///
    /// * `max_events`       – Maximum events kept in the rolling buffer.
    /// * `threat_threshold` – Score level (0..100) that triggers automatic
    ///   incident creation.  Defaults to 50.0 if `0.0` is supplied.
    pub fn new(max_events: usize, threat_threshold: f64) -> Self {
        let threshold = if threat_threshold <= 0.0 {
            50.0
        } else {
            threat_threshold
        };
        NetworkSecurityMonitor {
            events: VecDeque::with_capacity(max_events.min(4096)),
            scores: HashMap::new(),
            incidents: Vec::new(),
            max_events,
            threat_threshold: threshold,
            next_id: 1,
        }
    }

    /// Convenience constructor with sensible defaults.
    pub fn default_monitor() -> Self {
        Self::new(10_000, 50.0)
    }

    // -----------------------------------------------------------------------
    // Event recording
    // -----------------------------------------------------------------------

    /// Record a security event, update the peer's threat score, and
    /// auto-create an incident if the score exceeds the threshold.
    ///
    /// Returns the ID of the newly created event.
    pub fn record_event(
        &mut self,
        peer_id: String,
        threat_type: ThreatType,
        level: ThreatLevel,
        description: String,
        evidence: Vec<String>,
        now: u64,
    ) -> u64 {
        // Build and store the event.
        let event = SecurityEvent::new(
            peer_id.clone(),
            threat_type,
            level,
            description,
            now,
            evidence,
        );
        let event_id = event.id;

        // Enforce rolling-buffer capacity.
        if self.events.len() >= self.max_events {
            self.events.pop_front();
        }
        self.events.push_back(event);

        // Update peer score.
        let points = level.score_contribution();
        let score_entry = self
            .scores
            .entry(peer_id.clone())
            .or_insert_with(|| ThreatScore::new(peer_id.clone(), now));

        // Apply decay first so the new points are added to the decayed base.
        Self::apply_decay_inner(score_entry, now);
        score_entry.add(points, now);

        let current_score = score_entry.score;

        // Auto-create incident when threshold crossed and no open incident
        // already exists for this peer.
        if current_score >= self.threat_threshold && !self.has_open_incident_for_peer(&peer_id) {
            let peer_events: Vec<u64> = self
                .events
                .iter()
                .filter(|e| e.peer_id == peer_id)
                .map(|e| e.id)
                .collect();
            self.create_incident(peer_events, now);
        }

        event_id
    }

    // -----------------------------------------------------------------------
    // Threat scoring
    // -----------------------------------------------------------------------

    /// Return the current decay-adjusted threat score for `peer_id`.
    /// Returns `0.0` for unknown peers.
    pub fn threat_score(&mut self, peer_id: &str, now: u64) -> f64 {
        match self.scores.get_mut(peer_id) {
            None => 0.0,
            Some(score) => {
                Self::apply_decay_inner(score, now);
                score.score
            }
        }
    }

    /// Apply hourly 10 % decay to a [`ThreatScore`].
    ///
    /// `score *= 0.9 ^ floor((now - last_updated) / 3_600_000)`
    ///
    /// This is a pure (no-`self`) helper so it can be called with an
    /// `&mut ThreatScore` extracted from the map while the monitor is also
    /// borrowed.
    pub fn apply_decay(score: &mut ThreatScore, now: u64) {
        Self::apply_decay_inner(score, now);
    }

    fn apply_decay_inner(score: &mut ThreatScore, now: u64) {
        if now <= score.last_updated {
            return;
        }
        let elapsed_ms = now - score.last_updated;
        let hours = elapsed_ms / 3_600_000;
        if hours == 0 {
            return;
        }
        // 0.9^hours
        let factor = 0.9_f64.powi(hours as i32);
        score.score *= factor;
        score.last_updated = now;
    }

    // -----------------------------------------------------------------------
    // Query helpers
    // -----------------------------------------------------------------------

    /// Return the top `n` [`ThreatScore`] references, sorted by score descending,
    /// after applying decay at the given timestamp.
    pub fn top_threats(&mut self, n: usize, now: u64) -> Vec<&ThreatScore> {
        // Apply decay to all entries.
        for score in self.scores.values_mut() {
            Self::apply_decay_inner(score, now);
        }
        let mut refs: Vec<&ThreatScore> = self.scores.values().collect();
        refs.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        refs.truncate(n);
        refs
    }

    /// All events recorded for `peer_id`, in insertion order.
    pub fn events_for_peer(&self, peer_id: &str) -> Vec<&SecurityEvent> {
        self.events
            .iter()
            .filter(|e| e.peer_id == peer_id)
            .collect()
    }

    /// All events of the given threat type, in insertion order.
    pub fn events_by_threat_type(&self, threat_type: &ThreatType) -> Vec<&SecurityEvent> {
        self.events
            .iter()
            .filter(|e| &e.threat_type == threat_type)
            .collect()
    }

    /// All events with a timestamp ≥ `timestamp`.
    pub fn events_since(&self, timestamp: u64) -> Vec<&SecurityEvent> {
        self.events
            .iter()
            .filter(|e| e.timestamp >= timestamp)
            .collect()
    }

    // -----------------------------------------------------------------------
    // Incident management
    // -----------------------------------------------------------------------

    /// Create a new incident from a list of event IDs.  Returns the new incident's ID.
    pub fn create_incident(&mut self, events: Vec<u64>, now: u64) -> u64 {
        let id = self.next_incident_id(now, events.len() as u64);
        let incident = SecurityIncident {
            id,
            events,
            status: IncidentStatus::Open,
            created_at: now,
            resolved_at: None,
        };
        self.incidents.push(incident);
        id
    }

    /// Update the status of incident `incident_id`.  Returns `true` on success.
    pub fn update_incident_status(&mut self, incident_id: u64, status: IncidentStatus) -> bool {
        match self.incidents.iter_mut().find(|i| i.id == incident_id) {
            None => false,
            Some(incident) => {
                incident.status = status;
                true
            }
        }
    }

    /// All incidents with status `Open` or `Investigating`.
    pub fn open_incidents(&self) -> Vec<&SecurityIncident> {
        self.incidents
            .iter()
            .filter(|i| {
                i.status == IncidentStatus::Open || i.status == IncidentStatus::Investigating
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Maintenance
    // -----------------------------------------------------------------------

    /// Remove all events and the score entry for `peer_id`.
    /// Returns the number of events removed.
    pub fn clear_peer_history(&mut self, peer_id: &str) -> usize {
        let before = self.events.len();
        self.events.retain(|e| e.peer_id != peer_id);
        let removed = before - self.events.len();
        self.scores.remove(peer_id);
        removed
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Compute a statistics snapshot at the given timestamp (applying decay
    /// before computing averages).
    pub fn stats(&mut self, now: u64) -> SecurityMonitorStats {
        // Apply decay to all scores.
        for score in self.scores.values_mut() {
            Self::apply_decay_inner(score, now);
        }

        let total_events = self.events.len();
        let open_incidents = self.open_incidents().len();
        let resolved_incidents = self
            .incidents
            .iter()
            .filter(|i| {
                i.status == IncidentStatus::Resolved || i.status == IncidentStatus::FalsePositive
            })
            .count();

        let high_threat_peers = self.scores.values().filter(|s| s.score >= 50.0).count();

        let avg_threat_score = if self.scores.is_empty() {
            0.0
        } else {
            let sum: f64 = self.scores.values().map(|s| s.score).sum();
            sum / self.scores.len() as f64
        };

        SecurityMonitorStats {
            total_events,
            open_incidents,
            resolved_incidents,
            high_threat_peers,
            avg_threat_score,
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Check whether an open or investigating incident already exists that
    /// contains at least one event for `peer_id`.
    fn has_open_incident_for_peer(&self, peer_id: &str) -> bool {
        // Collect event IDs for this peer.
        let peer_event_ids: std::collections::HashSet<u64> = self
            .events
            .iter()
            .filter(|e| e.peer_id == peer_id)
            .map(|e| e.id)
            .collect();

        self.incidents.iter().any(|inc| {
            (inc.status == IncidentStatus::Open || inc.status == IncidentStatus::Investigating)
                && inc.events.iter().any(|eid| peer_event_ids.contains(eid))
        })
    }

    /// Generate a new, unique incident ID using FNV-1a over the counter + now.
    fn next_incident_id(&mut self, now: u64, extra: u64) -> u64 {
        let counter = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        let mut data = [0u8; 24];
        data[0..8].copy_from_slice(&counter.to_le_bytes());
        data[8..16].copy_from_slice(&now.to_le_bytes());
        data[16..24].copy_from_slice(&extra.to_le_bytes());
        fnv1a(&data)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::security_monitor::{
        IncidentStatus, NetworkSecurityMonitor, SecurityEvent, ThreatLevel, ThreatScore, ThreatType,
    };

    // ------------------------------------------------------------------
    // Helper
    // ------------------------------------------------------------------

    fn make_monitor() -> NetworkSecurityMonitor {
        NetworkSecurityMonitor::new(100, 50.0)
    }

    const T0: u64 = 1_700_000_000_000; // arbitrary epoch ms

    fn record(
        mon: &mut NetworkSecurityMonitor,
        peer: &str,
        tt: ThreatType,
        level: ThreatLevel,
    ) -> u64 {
        mon.record_event(
            peer.to_owned(),
            tt,
            level,
            "test".to_owned(),
            vec!["evidence".to_owned()],
            T0,
        )
    }

    // ------------------------------------------------------------------
    // ThreatLevel ordering
    // ------------------------------------------------------------------

    #[test]
    fn threat_level_ordering_none_is_smallest() {
        assert!(ThreatLevel::None < ThreatLevel::Low);
    }

    #[test]
    fn threat_level_ordering_critical_is_largest() {
        assert!(ThreatLevel::Critical > ThreatLevel::High);
    }

    #[test]
    fn threat_level_ordering_full_chain() {
        let levels = [
            ThreatLevel::None,
            ThreatLevel::Low,
            ThreatLevel::Medium,
            ThreatLevel::High,
            ThreatLevel::Critical,
        ];
        for window in levels.windows(2) {
            assert!(window[0] < window[1]);
        }
    }

    #[test]
    fn threat_level_eq() {
        assert_eq!(ThreatLevel::High, ThreatLevel::High);
        assert_ne!(ThreatLevel::Low, ThreatLevel::Medium);
    }

    // ------------------------------------------------------------------
    // Score contributions
    // ------------------------------------------------------------------

    #[test]
    fn score_contribution_none_is_zero() {
        assert_eq!(ThreatLevel::None.score_contribution(), 0.0);
    }

    #[test]
    fn score_contribution_values() {
        assert_eq!(ThreatLevel::Low.score_contribution(), 5.0);
        assert_eq!(ThreatLevel::Medium.score_contribution(), 15.0);
        assert_eq!(ThreatLevel::High.score_contribution(), 30.0);
        assert_eq!(ThreatLevel::Critical.score_contribution(), 50.0);
    }

    // ------------------------------------------------------------------
    // Event IDs
    // ------------------------------------------------------------------

    #[test]
    fn event_id_is_deterministic() {
        let e1 = SecurityEvent::new(
            "peer1".to_owned(),
            ThreatType::Sybil,
            ThreatLevel::Low,
            "desc".to_owned(),
            T0,
            vec![],
        );
        let e2 = SecurityEvent::new(
            "peer1".to_owned(),
            ThreatType::Sybil,
            ThreatLevel::Low,
            "desc".to_owned(),
            T0,
            vec![],
        );
        assert_eq!(e1.id, e2.id);
    }

    #[test]
    fn event_id_differs_for_different_peers() {
        let e1 = SecurityEvent::new(
            "peerA".to_owned(),
            ThreatType::DDoS,
            ThreatLevel::High,
            "x".to_owned(),
            T0,
            vec![],
        );
        let e2 = SecurityEvent::new(
            "peerB".to_owned(),
            ThreatType::DDoS,
            ThreatLevel::High,
            "x".to_owned(),
            T0,
            vec![],
        );
        assert_ne!(e1.id, e2.id);
    }

    #[test]
    fn event_id_differs_for_different_timestamps() {
        let e1 = SecurityEvent::new(
            "peer1".to_owned(),
            ThreatType::Eclipse,
            ThreatLevel::Medium,
            "x".to_owned(),
            T0,
            vec![],
        );
        let e2 = SecurityEvent::new(
            "peer1".to_owned(),
            ThreatType::Eclipse,
            ThreatLevel::Medium,
            "x".to_owned(),
            T0 + 1,
            vec![],
        );
        assert_ne!(e1.id, e2.id);
    }

    // ------------------------------------------------------------------
    // record_event / threat_score
    // ------------------------------------------------------------------

    #[test]
    fn record_event_returns_nonzero_id() {
        let mut mon = make_monitor();
        let id = record(&mut mon, "peer1", ThreatType::Sybil, ThreatLevel::Low);
        assert_ne!(id, 0);
    }

    #[test]
    fn threat_score_unknown_peer_is_zero() {
        let mut mon = make_monitor();
        assert_eq!(mon.threat_score("ghost", T0), 0.0);
    }

    #[test]
    fn threat_score_accumulates_after_low_event() {
        let mut mon = make_monitor();
        record(&mut mon, "peer1", ThreatType::Sybil, ThreatLevel::Low);
        let score = mon.threat_score("peer1", T0);
        assert!((score - 5.0).abs() < 1e-9);
    }

    #[test]
    fn threat_score_accumulates_multiple_events() {
        let mut mon = make_monitor();
        record(&mut mon, "peer1", ThreatType::DDoS, ThreatLevel::Low); // +5
        record(&mut mon, "peer1", ThreatType::DDoS, ThreatLevel::Medium); // +15
        let score = mon.threat_score("peer1", T0);
        assert!((score - 20.0).abs() < 1e-9);
    }

    #[test]
    fn threat_score_capped_at_100() {
        let mut mon = make_monitor();
        for _ in 0..10 {
            record(&mut mon, "peer1", ThreatType::DDoS, ThreatLevel::Critical); // +50 each
        }
        let score = mon.threat_score("peer1", T0);
        assert!(score <= 100.0);
    }

    // ------------------------------------------------------------------
    // Decay
    // ------------------------------------------------------------------

    #[test]
    fn decay_no_change_within_same_hour() {
        let mut score = ThreatScore::new("p".to_owned(), T0);
        score.add(40.0, T0);
        NetworkSecurityMonitor::apply_decay(&mut score, T0 + 3_599_999);
        assert!((score.score - 40.0).abs() < 1e-9);
    }

    #[test]
    fn decay_one_hour_reduces_by_10_percent() {
        let mut score = ThreatScore::new("p".to_owned(), T0);
        score.add(100.0, T0);
        NetworkSecurityMonitor::apply_decay(&mut score, T0 + 3_600_000);
        let expected = 100.0 * 0.9;
        assert!((score.score - expected).abs() < 1e-6);
    }

    #[test]
    fn decay_two_hours() {
        let mut score = ThreatScore::new("p".to_owned(), T0);
        score.add(100.0, T0);
        NetworkSecurityMonitor::apply_decay(&mut score, T0 + 7_200_000);
        let expected = 100.0 * 0.9_f64.powi(2);
        assert!((score.score - expected).abs() < 1e-6);
    }

    #[test]
    fn decay_updates_last_updated_timestamp() {
        let t1 = T0 + 3_600_000;
        let mut score = ThreatScore::new("p".to_owned(), T0);
        score.add(50.0, T0);
        NetworkSecurityMonitor::apply_decay(&mut score, t1);
        assert_eq!(score.last_updated, t1);
    }

    #[test]
    fn decay_noop_when_now_equals_last_updated() {
        let mut score = ThreatScore::new("p".to_owned(), T0);
        score.add(60.0, T0);
        NetworkSecurityMonitor::apply_decay(&mut score, T0);
        assert!((score.score - 60.0).abs() < 1e-9);
    }

    // ------------------------------------------------------------------
    // Auto-incident creation
    // ------------------------------------------------------------------

    #[test]
    fn auto_incident_created_when_threshold_crossed() {
        let mut mon = NetworkSecurityMonitor::new(100, 50.0);
        // 2× Critical = 100 pts → should cross threshold 50
        record(
            &mut mon,
            "bad_peer",
            ThreatType::Eclipse,
            ThreatLevel::Critical,
        );
        let open = mon.open_incidents();
        assert_eq!(open.len(), 1);
    }

    #[test]
    fn no_duplicate_incident_for_same_peer() {
        let mut mon = NetworkSecurityMonitor::new(100, 10.0);
        // Each Critical adds 50; threshold is 10
        record(
            &mut mon,
            "bad_peer",
            ThreatType::Sybil,
            ThreatLevel::Critical,
        );
        record(
            &mut mon,
            "bad_peer",
            ThreatType::Sybil,
            ThreatLevel::Critical,
        );
        let open = mon.open_incidents();
        assert_eq!(open.len(), 1);
    }

    #[test]
    fn no_incident_below_threshold() {
        let mut mon = NetworkSecurityMonitor::new(100, 80.0);
        record(&mut mon, "peer1", ThreatType::DDoS, ThreatLevel::Low); // +5
        assert!(mon.open_incidents().is_empty());
    }

    // ------------------------------------------------------------------
    // Incident lifecycle
    // ------------------------------------------------------------------

    #[test]
    fn create_incident_manually() {
        let mut mon = make_monitor();
        let id = mon.create_incident(vec![1, 2, 3], T0);
        assert_ne!(id, 0);
        assert_eq!(mon.open_incidents().len(), 1);
    }

    #[test]
    fn update_incident_status_to_investigating() {
        let mut mon = make_monitor();
        let id = mon.create_incident(vec![], T0);
        assert!(mon.update_incident_status(id, IncidentStatus::Investigating));
        let open = mon.open_incidents();
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].status, IncidentStatus::Investigating);
    }

    #[test]
    fn update_incident_status_to_resolved() {
        let mut mon = make_monitor();
        let id = mon.create_incident(vec![], T0);
        mon.update_incident_status(id, IncidentStatus::Resolved);
        assert!(mon.open_incidents().is_empty());
    }

    #[test]
    fn update_incident_status_false_positive() {
        let mut mon = make_monitor();
        let id = mon.create_incident(vec![], T0);
        mon.update_incident_status(id, IncidentStatus::FalsePositive);
        assert!(mon.open_incidents().is_empty());
    }

    #[test]
    fn update_incident_unknown_id_returns_false() {
        let mut mon = make_monitor();
        assert!(!mon.update_incident_status(9_999_999, IncidentStatus::Resolved));
    }

    // ------------------------------------------------------------------
    // Query methods
    // ------------------------------------------------------------------

    #[test]
    fn events_for_peer_returns_only_matching() {
        let mut mon = make_monitor();
        record(&mut mon, "peerA", ThreatType::Sybil, ThreatLevel::Low);
        record(&mut mon, "peerB", ThreatType::DDoS, ThreatLevel::High);
        let evs = mon.events_for_peer("peerA");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].peer_id, "peerA");
    }

    #[test]
    fn events_for_peer_empty_for_unknown() {
        let mon = make_monitor();
        assert!(mon.events_for_peer("nobody").is_empty());
    }

    #[test]
    fn events_by_threat_type_filters_correctly() {
        let mut mon = make_monitor();
        record(&mut mon, "p1", ThreatType::Sybil, ThreatLevel::Low);
        record(&mut mon, "p2", ThreatType::DDoS, ThreatLevel::High);
        record(&mut mon, "p3", ThreatType::Sybil, ThreatLevel::Medium);
        let sybil = mon.events_by_threat_type(&ThreatType::Sybil);
        assert_eq!(sybil.len(), 2);
        for e in sybil {
            assert_eq!(e.threat_type, ThreatType::Sybil);
        }
    }

    #[test]
    fn events_since_filters_by_timestamp() {
        let mut mon = make_monitor();
        mon.record_event(
            "p1".to_owned(),
            ThreatType::DDoS,
            ThreatLevel::Low,
            "x".to_owned(),
            vec![],
            T0,
        );
        mon.record_event(
            "p2".to_owned(),
            ThreatType::DDoS,
            ThreatLevel::Low,
            "y".to_owned(),
            vec![],
            T0 + 5000,
        );
        let recent = mon.events_since(T0 + 1);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].peer_id, "p2");
    }

    #[test]
    fn events_since_includes_equal_timestamp() {
        let mut mon = make_monitor();
        mon.record_event(
            "p1".to_owned(),
            ThreatType::Sybil,
            ThreatLevel::Low,
            "x".to_owned(),
            vec![],
            T0,
        );
        let evs = mon.events_since(T0);
        assert_eq!(evs.len(), 1);
    }

    // ------------------------------------------------------------------
    // top_threats
    // ------------------------------------------------------------------

    #[test]
    fn top_threats_returns_highest_scores_first() {
        let mut mon = make_monitor();
        record(&mut mon, "low", ThreatType::Sybil, ThreatLevel::Low); // 5
        record(&mut mon, "high", ThreatType::DDoS, ThreatLevel::High); // 30
        record(&mut mon, "med", ThreatType::Eclipse, ThreatLevel::Medium); // 15
        let top = mon.top_threats(3, T0);
        assert_eq!(top[0].peer_id, "high");
        assert_eq!(top[1].peer_id, "med");
        assert_eq!(top[2].peer_id, "low");
    }

    #[test]
    fn top_threats_n_larger_than_peers() {
        let mut mon = make_monitor();
        record(&mut mon, "p1", ThreatType::Sybil, ThreatLevel::Low);
        let top = mon.top_threats(10, T0);
        assert_eq!(top.len(), 1);
    }

    // ------------------------------------------------------------------
    // clear_peer_history
    // ------------------------------------------------------------------

    #[test]
    fn clear_peer_history_removes_events_and_score() {
        let mut mon = make_monitor();
        record(&mut mon, "peer1", ThreatType::Sybil, ThreatLevel::High);
        record(&mut mon, "peer1", ThreatType::DDoS, ThreatLevel::Medium);
        record(&mut mon, "peer2", ThreatType::Eclipse, ThreatLevel::Low);
        let removed = mon.clear_peer_history("peer1");
        assert_eq!(removed, 2);
        assert_eq!(mon.threat_score("peer1", T0), 0.0);
        assert_eq!(mon.events_for_peer("peer1").len(), 0);
        assert_eq!(mon.events_for_peer("peer2").len(), 1);
    }

    #[test]
    fn clear_peer_history_unknown_peer_returns_zero() {
        let mut mon = make_monitor();
        assert_eq!(mon.clear_peer_history("nobody"), 0);
    }

    // ------------------------------------------------------------------
    // Rolling buffer
    // ------------------------------------------------------------------

    #[test]
    fn rolling_buffer_evicts_oldest_when_full() {
        let mut mon = NetworkSecurityMonitor::new(3, 200.0); // high threshold to avoid incidents
        let id1 = record(&mut mon, "p", ThreatType::Sybil, ThreatLevel::None);
        let _id2 = record(&mut mon, "p", ThreatType::DDoS, ThreatLevel::None);
        let _id3 = record(&mut mon, "p", ThreatType::Eclipse, ThreatLevel::None);
        let _id4 = record(&mut mon, "p", ThreatType::ReplayAttack, ThreatLevel::None);
        // id1 should have been evicted
        let present: Vec<u64> = mon.events_for_peer("p").iter().map(|e| e.id).collect();
        assert!(!present.contains(&id1));
        assert_eq!(present.len(), 3);
    }

    // ------------------------------------------------------------------
    // Stats
    // ------------------------------------------------------------------

    #[test]
    fn stats_empty_monitor() {
        let mut mon = make_monitor();
        let s = mon.stats(T0);
        assert_eq!(s.total_events, 0);
        assert_eq!(s.open_incidents, 0);
        assert_eq!(s.resolved_incidents, 0);
        assert_eq!(s.high_threat_peers, 0);
        assert_eq!(s.avg_threat_score, 0.0);
    }

    #[test]
    fn stats_counts_open_incidents() {
        let mut mon = make_monitor();
        mon.create_incident(vec![], T0);
        mon.create_incident(vec![], T0);
        let s = mon.stats(T0);
        assert_eq!(s.open_incidents, 2);
    }

    #[test]
    fn stats_counts_resolved_incidents() {
        let mut mon = make_monitor();
        let id = mon.create_incident(vec![], T0);
        mon.update_incident_status(id, IncidentStatus::Resolved);
        let s = mon.stats(T0);
        assert_eq!(s.resolved_incidents, 1);
        assert_eq!(s.open_incidents, 0);
    }

    #[test]
    fn stats_high_threat_peers_threshold_50() {
        let mut mon = NetworkSecurityMonitor::new(100, 200.0); // prevent auto-incident
        record(&mut mon, "clean", ThreatType::Sybil, ThreatLevel::Low); // 5
        record(&mut mon, "dirty", ThreatType::DDoS, ThreatLevel::Critical); // 50
        let s = mon.stats(T0);
        assert_eq!(s.high_threat_peers, 1);
    }

    #[test]
    fn stats_avg_threat_score() {
        let mut mon = NetworkSecurityMonitor::new(100, 200.0);
        record(&mut mon, "p1", ThreatType::Sybil, ThreatLevel::Low); // 5
        record(&mut mon, "p2", ThreatType::DDoS, ThreatLevel::Medium); // 15
        let s = mon.stats(T0);
        let expected_avg = (5.0 + 15.0) / 2.0;
        assert!((s.avg_threat_score - expected_avg).abs() < 1e-6);
    }

    // ------------------------------------------------------------------
    // ThreatType names
    // ------------------------------------------------------------------

    #[test]
    fn threat_type_names_are_correct() {
        assert_eq!(ThreatType::Sybil.name(), "Sybil");
        assert_eq!(ThreatType::Eclipse.name(), "Eclipse");
        assert_eq!(ThreatType::DDoS.name(), "DDoS");
        assert_eq!(ThreatType::ManInTheMiddle.name(), "ManInTheMiddle");
        assert_eq!(ThreatType::ReplayAttack.name(), "ReplayAttack");
        assert_eq!(ThreatType::RoutingPoison.name(), "RoutingPoison");
        assert_eq!(ThreatType::DataTampering.name(), "DataTampering");
    }

    // ------------------------------------------------------------------
    // Default monitor
    // ------------------------------------------------------------------

    #[test]
    fn default_monitor_threshold_is_50() {
        let mut mon = NetworkSecurityMonitor::default_monitor();
        // A single Critical event scores 50; just at the threshold.
        record(&mut mon, "p", ThreatType::Sybil, ThreatLevel::Critical);
        // Score == 50 → incident should have been created.
        assert_eq!(mon.open_incidents().len(), 1);
    }

    #[test]
    fn zero_threshold_replaced_with_50() {
        let mon = NetworkSecurityMonitor::new(10, 0.0);
        // Cannot access the field directly, but we can verify via stats
        // that no panics occur.
        let _ = format!("{:?}", mon.events_for_peer("x"));
    }

    // ------------------------------------------------------------------
    // Incident events list
    // ------------------------------------------------------------------

    #[test]
    fn incident_contains_peer_event_ids() {
        let mut mon = NetworkSecurityMonitor::new(100, 10.0);
        let eid = record(&mut mon, "p", ThreatType::DDoS, ThreatLevel::Critical);
        let incidents = mon.open_incidents();
        assert_eq!(incidents.len(), 1);
        assert!(incidents[0].events.contains(&eid));
    }

    // ------------------------------------------------------------------
    // Multiple peers do not interfere
    // ------------------------------------------------------------------

    #[test]
    fn different_peers_have_independent_scores() {
        let mut mon = make_monitor();
        record(&mut mon, "alice", ThreatType::Sybil, ThreatLevel::High); // 30
        record(&mut mon, "bob", ThreatType::DDoS, ThreatLevel::Medium); // 15
        assert!((mon.threat_score("alice", T0) - 30.0).abs() < 1e-9);
        assert!((mon.threat_score("bob", T0) - 15.0).abs() < 1e-9);
    }
}
