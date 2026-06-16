//! NAT traversal system for P2P hole-punching.
//!
//! Provides NAT type detection, STUN-like binding analysis,
//! hole-punch coordination with port prediction, and relay fallback.

use std::collections::HashMap;

/// NAT type detection result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NatType {
    /// No NAT — public IP directly reachable
    Open,
    /// Full cone: any external host can send to mapped port
    FullCone,
    /// Restricted cone: only hosts the internal host has sent to
    RestrictedCone,
    /// Port restricted: restricted by both IP and port
    PortRestricted,
    /// Symmetric: different mapping per destination
    Symmetric,
    /// Could not determine NAT type
    Unknown,
}

/// Traversal strategy recommendation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraversalStrategy {
    /// Open NAT — direct connection
    Direct,
    /// Full cone / restricted — simultaneous hole punch
    HolePunch,
    /// Port restricted with predictable pattern — port prediction + punch
    PortPrediction,
    /// Symmetric — needs relay
    Relay,
}

/// Status of a hole-punch attempt
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HolePunchStatus {
    /// Waiting to start
    Pending,
    /// Currently attempting
    InProgress,
    /// Successfully established
    Success,
    /// All attempts exhausted
    Failed,
    /// Exceeded time limit
    TimedOut,
}

/// STUN-like binding result
#[derive(Debug, Clone)]
pub struct StunBinding {
    /// Local address used for the binding request
    pub local_addr: String,
    /// Mapped (external) address as seen by the STUN server
    pub mapped_addr: String,
    /// Detected NAT type from this binding
    pub nat_type: NatType,
    /// Round-trip latency in milliseconds
    pub latency_ms: u64,
    /// Unix timestamp when this binding was obtained
    pub timestamp: u64,
}

/// Tracks a single hole-punch attempt to a peer
#[derive(Debug, Clone)]
pub struct HolePunchAttempt {
    /// Peer identifier
    pub peer_id: String,
    /// Local port used for punching
    pub local_port: u16,
    /// Remote address to punch towards
    pub remote_addr: String,
    /// Number of attempts made so far
    pub attempts: u32,
    /// Maximum attempts before giving up
    pub max_attempts: u32,
    /// Current status
    pub status: HolePunchStatus,
    /// Unix timestamp when the attempt started
    pub started_at: u64,
}

/// Configuration for NAT traversal behaviour
#[derive(Debug, Clone)]
pub struct NatTraversalConfig {
    /// Timeout for STUN binding requests (ms)
    pub stun_timeout_ms: u64,
    /// Maximum hole-punch attempts per peer
    pub max_punch_attempts: u32,
    /// Interval between successive punch packets (ms)
    pub punch_interval_ms: u64,
    /// Range of ports to scan around a predicted port
    pub port_prediction_range: u16,
    /// Whether to attempt UPnP port mapping
    pub enable_upnp: bool,
    /// Whether to fall back to relay when punch fails
    pub enable_relay_fallback: bool,
}

impl Default for NatTraversalConfig {
    fn default() -> Self {
        Self {
            stun_timeout_ms: 3000,
            max_punch_attempts: 5,
            punch_interval_ms: 500,
            port_prediction_range: 10,
            enable_upnp: true,
            enable_relay_fallback: true,
        }
    }
}

/// Aggregated statistics for NAT traversal operations
#[derive(Debug, Clone, Default)]
pub struct NatTraversalStats {
    /// Total hole-punch attempts initiated
    pub total_attempts: u64,
    /// Successful punches
    pub successful: u64,
    /// Failed punches (max attempts exhausted)
    pub failed: u64,
    /// Punches that timed out
    pub timed_out: u64,
    /// Running average of successful punch time (ms)
    pub avg_punch_time_ms: f64,
}

/// Central manager for NAT traversal operations
pub struct NatTraversalManager {
    config: NatTraversalConfig,
    bindings: Vec<StunBinding>,
    active_punches: HashMap<String, HolePunchAttempt>,
    stats: NatTraversalStats,
    detected_nat_type: NatType,
}

impl NatTraversalManager {
    /// Create a new manager with the given configuration.
    pub fn new(config: NatTraversalConfig) -> Self {
        Self {
            config,
            bindings: Vec::new(),
            active_punches: HashMap::new(),
            stats: NatTraversalStats::default(),
            detected_nat_type: NatType::Unknown,
        }
    }

    /// Analyse a set of STUN bindings to determine the NAT type.
    ///
    /// The algorithm compares mapped addresses across bindings:
    /// - All identical mapped addresses => FullCone (or Open if mapped == local)
    /// - Same IP but different ports => PortRestricted
    /// - Different IPs => Symmetric
    /// - Single binding or ambiguous => RestrictedCone or Unknown
    pub fn detect_nat_type(bindings: &[StunBinding]) -> NatType {
        if bindings.is_empty() {
            return NatType::Unknown;
        }

        if bindings.len() == 1 {
            let b = &bindings[0];
            if b.local_addr == b.mapped_addr {
                return NatType::Open;
            }
            // Single binding — can only say it's behind some NAT
            return NatType::RestrictedCone;
        }

        // Extract mapped IPs and ports
        let mapped_parts: Vec<(&str, &str)> = bindings
            .iter()
            .filter_map(|b| {
                let parts: Vec<&str> = b.mapped_addr.rsplitn(2, ':').collect();
                if parts.len() == 2 {
                    Some((parts[1], parts[0]))
                } else {
                    None
                }
            })
            .collect();

        if mapped_parts.is_empty() {
            return NatType::Unknown;
        }

        // Check if all local == mapped (Open)
        let all_open = bindings.iter().all(|b| b.local_addr == b.mapped_addr);
        if all_open {
            return NatType::Open;
        }

        let first_ip = mapped_parts[0].0;
        let first_port = mapped_parts[0].1;

        let all_same_ip = mapped_parts.iter().all(|(ip, _)| *ip == first_ip);
        let all_same_port = mapped_parts.iter().all(|(_, port)| *port == first_port);

        if all_same_ip && all_same_port {
            // Same external endpoint regardless of destination => FullCone
            NatType::FullCone
        } else if all_same_ip && !all_same_port {
            // Same IP but varying port => PortRestricted
            NatType::PortRestricted
        } else {
            // Different IPs across destinations => Symmetric
            NatType::Symmetric
        }
    }

    /// Record a new STUN binding and re-detect the NAT type.
    pub fn add_binding(&mut self, binding: StunBinding) {
        self.bindings.push(binding);
        self.detected_nat_type = Self::detect_nat_type(&self.bindings);
    }

    /// Begin a hole-punch attempt to a peer.
    ///
    /// Returns an error if a punch is already active for this peer.
    pub fn initiate_punch(
        &mut self,
        peer_id: &str,
        remote_addr: &str,
        local_port: u16,
    ) -> Result<(), String> {
        if self.active_punches.contains_key(peer_id) {
            return Err(format!("Hole-punch already active for peer {}", peer_id));
        }

        let attempt = HolePunchAttempt {
            peer_id: peer_id.to_string(),
            local_port,
            remote_addr: remote_addr.to_string(),
            attempts: 0,
            max_attempts: self.config.max_punch_attempts,
            status: HolePunchStatus::Pending,
            started_at: current_timestamp(),
        };

        self.active_punches.insert(peer_id.to_string(), attempt);
        self.stats.total_attempts += 1;

        Ok(())
    }

    /// Predict the next allocated port(s) based on observed samples.
    ///
    /// If the samples show a linear pattern (constant delta), predict the next
    /// port and return a range around it. Otherwise return a range around the
    /// base port.
    pub fn predict_port(&self, base_port: u16, sample_ports: &[u16]) -> Vec<u16> {
        let range = self.config.port_prediction_range;

        if sample_ports.len() >= 2 {
            // Calculate deltas between consecutive ports
            let deltas: Vec<i32> = sample_ports
                .windows(2)
                .map(|w| i32::from(w[1]) - i32::from(w[0]))
                .collect();

            // Check if deltas are consistent (linear allocation)
            let first_delta = deltas[0];
            let is_linear = deltas.iter().all(|d| *d == first_delta);

            if is_linear && first_delta != 0 {
                let last = i32::from(*sample_ports.last().unwrap_or(&base_port));
                let predicted = last + first_delta;
                let center = predicted.clamp(0, 65535) as u16;
                return port_range(center, range);
            }
        }

        // Fallback: range around base_port
        port_range(base_port, range)
    }

    /// Update the status of an active hole-punch attempt.
    pub fn update_attempt(&mut self, peer_id: &str, status: HolePunchStatus) -> Result<(), String> {
        let attempt = self
            .active_punches
            .get_mut(peer_id)
            .ok_or_else(|| format!("No active punch for peer {}", peer_id))?;

        match &status {
            HolePunchStatus::InProgress => {
                attempt.attempts += 1;
                if attempt.attempts > attempt.max_attempts {
                    attempt.status = HolePunchStatus::Failed;
                    self.stats.failed += 1;
                    return Ok(());
                }
            }
            HolePunchStatus::Success => {
                self.stats.successful += 1;
                let elapsed = current_timestamp().saturating_sub(attempt.started_at);
                // Update running average
                let n = self.stats.successful as f64;
                self.stats.avg_punch_time_ms =
                    self.stats.avg_punch_time_ms * ((n - 1.0) / n) + (elapsed as f64) / n;
            }
            HolePunchStatus::Failed => {
                self.stats.failed += 1;
            }
            HolePunchStatus::TimedOut => {
                self.stats.timed_out += 1;
            }
            HolePunchStatus::Pending => {}
        }

        attempt.status = status;
        Ok(())
    }

    /// Look up an active punch attempt by peer ID.
    pub fn get_attempt(&self, peer_id: &str) -> Option<&HolePunchAttempt> {
        self.active_punches.get(peer_id)
    }

    /// Number of currently active (non-terminal) punch attempts.
    pub fn active_punch_count(&self) -> usize {
        self.active_punches
            .values()
            .filter(|a| {
                matches!(
                    a.status,
                    HolePunchStatus::Pending | HolePunchStatus::InProgress
                )
            })
            .count()
    }

    /// Remove expired or completed attempts older than `now - stun_timeout_ms`.
    /// Returns the number of entries removed.
    pub fn cleanup_expired(&mut self, now: u64) -> usize {
        let timeout = self.config.stun_timeout_ms;
        let before = self.active_punches.len();

        self.active_punches.retain(|_k, v| {
            let age = now.saturating_sub(v.started_at);
            let is_terminal = matches!(
                v.status,
                HolePunchStatus::Success | HolePunchStatus::Failed | HolePunchStatus::TimedOut
            );
            // Keep if not expired AND not in a terminal state
            age < timeout || !is_terminal
        });

        before - self.active_punches.len()
    }

    /// Fraction of successful punches out of total (0.0 if none attempted).
    pub fn success_rate(&self) -> f64 {
        if self.stats.total_attempts == 0 {
            return 0.0;
        }
        self.stats.successful as f64 / self.stats.total_attempts as f64
    }

    /// Reference to aggregated stats.
    pub fn stats(&self) -> &NatTraversalStats {
        &self.stats
    }

    /// Reference to the currently detected NAT type.
    pub fn detected_nat_type(&self) -> &NatType {
        &self.detected_nat_type
    }

    /// Recommend the best traversal strategy based on the detected NAT type.
    pub fn best_strategy(&self) -> TraversalStrategy {
        match &self.detected_nat_type {
            NatType::Open => TraversalStrategy::Direct,
            NatType::FullCone | NatType::RestrictedCone => TraversalStrategy::HolePunch,
            NatType::PortRestricted => TraversalStrategy::PortPrediction,
            NatType::Symmetric | NatType::Unknown => {
                if self.config.enable_relay_fallback {
                    TraversalStrategy::Relay
                } else {
                    TraversalStrategy::HolePunch
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a sorted, deduplicated range of ports around `center`.
fn port_range(center: u16, half_range: u16) -> Vec<u16> {
    let low = center.saturating_sub(half_range);
    let high = center.saturating_add(half_range);
    (low..=high).collect()
}

/// Monotonic-ish timestamp in milliseconds (uses `Instant` elapsed from a
/// fixed epoch for testability). Falls back to 0 on platforms without a clock.
fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> NatTraversalConfig {
        NatTraversalConfig::default()
    }

    fn make_binding(local: &str, mapped: &str) -> StunBinding {
        StunBinding {
            local_addr: local.to_string(),
            mapped_addr: mapped.to_string(),
            nat_type: NatType::Unknown,
            latency_ms: 10,
            timestamp: 1000,
        }
    }

    // ---- Config defaults ----

    #[test]
    fn config_default_values() {
        let c = NatTraversalConfig::default();
        assert_eq!(c.stun_timeout_ms, 3000);
        assert_eq!(c.max_punch_attempts, 5);
        assert_eq!(c.punch_interval_ms, 500);
        assert_eq!(c.port_prediction_range, 10);
        assert!(c.enable_upnp);
        assert!(c.enable_relay_fallback);
    }

    #[test]
    fn config_custom_values() {
        let c = NatTraversalConfig {
            stun_timeout_ms: 5000,
            max_punch_attempts: 10,
            punch_interval_ms: 200,
            port_prediction_range: 20,
            enable_upnp: false,
            enable_relay_fallback: false,
        };
        assert_eq!(c.stun_timeout_ms, 5000);
        assert!(!c.enable_upnp);
    }

    // ---- StunBinding creation ----

    #[test]
    fn stun_binding_fields() {
        let b = StunBinding {
            local_addr: "192.168.1.1:5000".into(),
            mapped_addr: "1.2.3.4:6000".into(),
            nat_type: NatType::FullCone,
            latency_ms: 42,
            timestamp: 999,
        };
        assert_eq!(b.local_addr, "192.168.1.1:5000");
        assert_eq!(b.mapped_addr, "1.2.3.4:6000");
        assert_eq!(b.nat_type, NatType::FullCone);
        assert_eq!(b.latency_ms, 42);
        assert_eq!(b.timestamp, 999);
    }

    #[test]
    fn stun_binding_clone() {
        let b = make_binding("10.0.0.1:80", "1.2.3.4:80");
        let b2 = b.clone();
        assert_eq!(b.local_addr, b2.local_addr);
        assert_eq!(b.mapped_addr, b2.mapped_addr);
    }

    // ---- NAT type detection ----

    #[test]
    fn detect_nat_type_empty() {
        assert_eq!(NatTraversalManager::detect_nat_type(&[]), NatType::Unknown);
    }

    #[test]
    fn detect_nat_type_open_single() {
        let bindings = [make_binding("1.2.3.4:5000", "1.2.3.4:5000")];
        assert_eq!(
            NatTraversalManager::detect_nat_type(&bindings),
            NatType::Open
        );
    }

    #[test]
    fn detect_nat_type_open_multiple() {
        let bindings = [
            make_binding("1.2.3.4:5000", "1.2.3.4:5000"),
            make_binding("1.2.3.4:5001", "1.2.3.4:5001"),
        ];
        assert_eq!(
            NatTraversalManager::detect_nat_type(&bindings),
            NatType::Open
        );
    }

    #[test]
    fn detect_nat_type_full_cone() {
        // Same mapped addr regardless of destination
        let bindings = [
            make_binding("192.168.1.1:5000", "1.2.3.4:6000"),
            make_binding("192.168.1.1:5001", "1.2.3.4:6000"),
        ];
        assert_eq!(
            NatTraversalManager::detect_nat_type(&bindings),
            NatType::FullCone
        );
    }

    #[test]
    fn detect_nat_type_port_restricted() {
        // Same IP but different ports
        let bindings = [
            make_binding("192.168.1.1:5000", "1.2.3.4:6000"),
            make_binding("192.168.1.1:5001", "1.2.3.4:6001"),
        ];
        assert_eq!(
            NatTraversalManager::detect_nat_type(&bindings),
            NatType::PortRestricted
        );
    }

    #[test]
    fn detect_nat_type_symmetric() {
        // Different IPs
        let bindings = [
            make_binding("192.168.1.1:5000", "1.2.3.4:6000"),
            make_binding("192.168.1.1:5001", "5.6.7.8:6001"),
        ];
        assert_eq!(
            NatTraversalManager::detect_nat_type(&bindings),
            NatType::Symmetric
        );
    }

    #[test]
    fn detect_nat_type_restricted_cone_single_behind_nat() {
        // Single binding, local != mapped
        let bindings = [make_binding("192.168.1.1:5000", "1.2.3.4:6000")];
        assert_eq!(
            NatTraversalManager::detect_nat_type(&bindings),
            NatType::RestrictedCone
        );
    }

    #[test]
    fn detect_nat_type_unknown_bad_format() {
        // Mapped addr without colon separator
        let bindings = [StunBinding {
            local_addr: "192.168.1.1".into(),
            mapped_addr: "no-port-here".into(),
            nat_type: NatType::Unknown,
            latency_ms: 0,
            timestamp: 0,
        }];
        // Single binding with no port parse => RestrictedCone (single-binding path)
        // Actually the filter_map will yield empty => Unknown
        // But we have len()==1 check first...
        // The single-binding check runs before mapped_parts extraction.
        // local != mapped => RestrictedCone
        assert_eq!(
            NatTraversalManager::detect_nat_type(&bindings),
            NatType::RestrictedCone
        );
    }

    // ---- add_binding ----

    #[test]
    fn add_binding_updates_nat_type() {
        let mut mgr = NatTraversalManager::new(default_config());
        assert_eq!(*mgr.detected_nat_type(), NatType::Unknown);

        mgr.add_binding(make_binding("192.168.1.1:5000", "1.2.3.4:6000"));
        // Single behind-NAT binding => RestrictedCone
        assert_eq!(*mgr.detected_nat_type(), NatType::RestrictedCone);

        mgr.add_binding(make_binding("192.168.1.1:5001", "1.2.3.4:6000"));
        // Two bindings, same mapped => FullCone
        assert_eq!(*mgr.detected_nat_type(), NatType::FullCone);
    }

    // ---- Port prediction ----

    #[test]
    fn predict_port_linear_pattern() {
        let mgr = NatTraversalManager::new(default_config());
        // Ports increasing by 2
        let predicted = mgr.predict_port(5000, &[5000, 5002, 5004]);
        // Next predicted: 5006, range ±10
        assert!(predicted.contains(&5006));
        assert!(predicted.contains(&4996));
        assert!(predicted.contains(&5016));
    }

    #[test]
    fn predict_port_no_samples() {
        let mgr = NatTraversalManager::new(default_config());
        let predicted = mgr.predict_port(8000, &[]);
        // Fallback: range around base_port
        assert!(predicted.contains(&8000));
        assert!(predicted.contains(&7990));
        assert!(predicted.contains(&8010));
    }

    #[test]
    fn predict_port_single_sample() {
        let mgr = NatTraversalManager::new(default_config());
        let predicted = mgr.predict_port(4000, &[4000]);
        // Not enough for pattern => fallback
        assert!(predicted.contains(&4000));
    }

    #[test]
    fn predict_port_random_pattern() {
        let mgr = NatTraversalManager::new(default_config());
        // Non-linear deltas
        let predicted = mgr.predict_port(3000, &[3000, 3005, 3003]);
        // Fallback to base_port range
        assert!(predicted.contains(&3000));
    }

    #[test]
    fn predict_port_near_upper_bound() {
        let mgr = NatTraversalManager::new(default_config());
        let predicted = mgr.predict_port(65530, &[65528, 65530, 65532]);
        // Predicted: 65534, range should end at 65535
        assert!(predicted.last().copied().unwrap_or(0) == 65535);
    }

    #[test]
    fn predict_port_near_lower_bound() {
        let mgr = NatTraversalManager::new(default_config());
        let predicted = mgr.predict_port(5, &[10, 5, 0]);
        // delta = -5, predicted = 0 - 5 => clamped to 0
        assert!(predicted.contains(&0));
    }

    // ---- Hole punch initiation ----

    #[test]
    fn initiate_punch_success() {
        let mut mgr = NatTraversalManager::new(default_config());
        let result = mgr.initiate_punch("peer-1", "1.2.3.4:8000", 5000);
        assert!(result.is_ok());
        assert_eq!(mgr.active_punch_count(), 1);
    }

    #[test]
    fn initiate_punch_duplicate_error() {
        let mut mgr = NatTraversalManager::new(default_config());
        let _ = mgr.initiate_punch("peer-1", "1.2.3.4:8000", 5000);
        let result = mgr.initiate_punch("peer-1", "1.2.3.4:9000", 6000);
        assert!(result.is_err());
    }

    #[test]
    fn initiate_punch_increments_total_attempts() {
        let mut mgr = NatTraversalManager::new(default_config());
        let _ = mgr.initiate_punch("p1", "1.2.3.4:80", 100);
        let _ = mgr.initiate_punch("p2", "5.6.7.8:80", 200);
        assert_eq!(mgr.stats().total_attempts, 2);
    }

    // ---- Status updates ----

    #[test]
    fn update_attempt_success() {
        let mut mgr = NatTraversalManager::new(default_config());
        let _ = mgr.initiate_punch("peer-1", "1.2.3.4:8000", 5000);
        let result = mgr.update_attempt("peer-1", HolePunchStatus::Success);
        assert!(result.is_ok());
        assert_eq!(mgr.stats().successful, 1);
    }

    #[test]
    fn update_attempt_unknown_peer() {
        let mut mgr = NatTraversalManager::new(default_config());
        let result = mgr.update_attempt("ghost", HolePunchStatus::Success);
        assert!(result.is_err());
    }

    #[test]
    fn update_attempt_failed() {
        let mut mgr = NatTraversalManager::new(default_config());
        let _ = mgr.initiate_punch("peer-1", "1.2.3.4:8000", 5000);
        let _ = mgr.update_attempt("peer-1", HolePunchStatus::Failed);
        assert_eq!(mgr.stats().failed, 1);
    }

    #[test]
    fn update_attempt_timed_out() {
        let mut mgr = NatTraversalManager::new(default_config());
        let _ = mgr.initiate_punch("peer-1", "1.2.3.4:8000", 5000);
        let _ = mgr.update_attempt("peer-1", HolePunchStatus::TimedOut);
        assert_eq!(mgr.stats().timed_out, 1);
    }

    #[test]
    fn update_attempt_max_exceeded_auto_fails() {
        let config = NatTraversalConfig {
            max_punch_attempts: 2,
            ..default_config()
        };
        let mut mgr = NatTraversalManager::new(config);
        let _ = mgr.initiate_punch("peer-1", "1.2.3.4:8000", 5000);

        // 3 InProgress updates — third should auto-fail
        let _ = mgr.update_attempt("peer-1", HolePunchStatus::InProgress);
        let _ = mgr.update_attempt("peer-1", HolePunchStatus::InProgress);
        let _ = mgr.update_attempt("peer-1", HolePunchStatus::InProgress);

        let attempt = mgr.get_attempt("peer-1");
        assert!(attempt.is_some());
        assert_eq!(attempt.map(|a| &a.status), Some(&HolePunchStatus::Failed));
        assert_eq!(mgr.stats().failed, 1);
    }

    // ---- get_attempt ----

    #[test]
    fn get_attempt_existing() {
        let mut mgr = NatTraversalManager::new(default_config());
        let _ = mgr.initiate_punch("peer-1", "1.2.3.4:80", 100);
        let a = mgr.get_attempt("peer-1");
        assert!(a.is_some());
        assert_eq!(a.map(|x| &x.peer_id).unwrap_or(&String::new()), "peer-1");
    }

    #[test]
    fn get_attempt_missing() {
        let mgr = NatTraversalManager::new(default_config());
        assert!(mgr.get_attempt("nobody").is_none());
    }

    // ---- active_punch_count ----

    #[test]
    fn active_punch_count_excludes_terminal() {
        let mut mgr = NatTraversalManager::new(default_config());
        let _ = mgr.initiate_punch("p1", "1.2.3.4:80", 100);
        let _ = mgr.initiate_punch("p2", "5.6.7.8:80", 200);
        let _ = mgr.update_attempt("p1", HolePunchStatus::Success);
        // p1 is terminal, p2 is pending
        assert_eq!(mgr.active_punch_count(), 1);
    }

    // ---- cleanup_expired ----

    #[test]
    fn cleanup_expired_removes_old_terminal() {
        let mut mgr = NatTraversalManager::new(default_config());
        let _ = mgr.initiate_punch("p1", "1.2.3.4:80", 100);
        let _ = mgr.update_attempt("p1", HolePunchStatus::Success);

        // Fake "now" far in the future
        let removed = mgr.cleanup_expired(u64::MAX);
        assert_eq!(removed, 1);
        assert!(mgr.get_attempt("p1").is_none());
    }

    #[test]
    fn cleanup_expired_keeps_recent() {
        let mut mgr = NatTraversalManager::new(default_config());
        let _ = mgr.initiate_punch("p1", "1.2.3.4:80", 100);
        // Still pending, not terminal => kept regardless of age
        let removed = mgr.cleanup_expired(u64::MAX);
        assert_eq!(removed, 0);
    }

    // ---- success_rate ----

    #[test]
    fn success_rate_no_attempts() {
        let mgr = NatTraversalManager::new(default_config());
        assert!((mgr.success_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn success_rate_half() {
        let mut mgr = NatTraversalManager::new(default_config());
        let _ = mgr.initiate_punch("p1", "1.2.3.4:80", 100);
        let _ = mgr.initiate_punch("p2", "5.6.7.8:80", 200);
        let _ = mgr.update_attempt("p1", HolePunchStatus::Success);
        let _ = mgr.update_attempt("p2", HolePunchStatus::Failed);
        assert!((mgr.success_rate() - 0.5).abs() < f64::EPSILON);
    }

    // ---- Strategy selection ----

    #[test]
    fn strategy_open() {
        let mut mgr = NatTraversalManager::new(default_config());
        mgr.add_binding(make_binding("1.2.3.4:80", "1.2.3.4:80"));
        assert_eq!(mgr.best_strategy(), TraversalStrategy::Direct);
    }

    #[test]
    fn strategy_full_cone() {
        let mut mgr = NatTraversalManager::new(default_config());
        mgr.add_binding(make_binding("192.168.1.1:80", "1.2.3.4:6000"));
        mgr.add_binding(make_binding("192.168.1.1:81", "1.2.3.4:6000"));
        assert_eq!(mgr.best_strategy(), TraversalStrategy::HolePunch);
    }

    #[test]
    fn strategy_port_restricted() {
        let mut mgr = NatTraversalManager::new(default_config());
        mgr.add_binding(make_binding("192.168.1.1:80", "1.2.3.4:6000"));
        mgr.add_binding(make_binding("192.168.1.1:81", "1.2.3.4:6001"));
        assert_eq!(mgr.best_strategy(), TraversalStrategy::PortPrediction);
    }

    #[test]
    fn strategy_symmetric_with_relay() {
        let mut mgr = NatTraversalManager::new(default_config());
        mgr.add_binding(make_binding("192.168.1.1:80", "1.2.3.4:6000"));
        mgr.add_binding(make_binding("192.168.1.1:81", "5.6.7.8:6001"));
        assert_eq!(mgr.best_strategy(), TraversalStrategy::Relay);
    }

    #[test]
    fn strategy_symmetric_no_relay() {
        let config = NatTraversalConfig {
            enable_relay_fallback: false,
            ..default_config()
        };
        let mut mgr = NatTraversalManager::new(config);
        mgr.add_binding(make_binding("192.168.1.1:80", "1.2.3.4:6000"));
        mgr.add_binding(make_binding("192.168.1.1:81", "5.6.7.8:6001"));
        assert_eq!(mgr.best_strategy(), TraversalStrategy::HolePunch);
    }

    #[test]
    fn strategy_unknown_defaults_to_relay() {
        let mgr = NatTraversalManager::new(default_config());
        assert_eq!(mgr.best_strategy(), TraversalStrategy::Relay);
    }

    // ---- Stats ----

    #[test]
    fn stats_default() {
        let s = NatTraversalStats::default();
        assert_eq!(s.total_attempts, 0);
        assert_eq!(s.successful, 0);
        assert_eq!(s.failed, 0);
        assert_eq!(s.timed_out, 0);
        assert!((s.avg_punch_time_ms - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_reference() {
        let mgr = NatTraversalManager::new(default_config());
        let s = mgr.stats();
        assert_eq!(s.total_attempts, 0);
    }

    // ---- HolePunchAttempt ----

    #[test]
    fn hole_punch_attempt_clone() {
        let a = HolePunchAttempt {
            peer_id: "p1".into(),
            local_port: 1234,
            remote_addr: "1.2.3.4:80".into(),
            attempts: 0,
            max_attempts: 5,
            status: HolePunchStatus::Pending,
            started_at: 100,
        };
        let b = a.clone();
        assert_eq!(a.peer_id, b.peer_id);
        assert_eq!(a.status, b.status);
    }

    // ---- NatType enum ----

    #[test]
    fn nat_type_equality() {
        assert_eq!(NatType::Open, NatType::Open);
        assert_ne!(NatType::Open, NatType::FullCone);
        assert_ne!(NatType::Symmetric, NatType::Unknown);
    }

    // ---- port_range helper ----

    #[test]
    fn port_range_basic() {
        let r = port_range(100, 5);
        assert_eq!(r.len(), 11); // 95..=105
        assert_eq!(r[0], 95);
        assert_eq!(*r.last().unwrap_or(&0), 105);
    }

    #[test]
    fn port_range_saturates_low() {
        let r = port_range(3, 10);
        assert_eq!(r[0], 0);
    }

    #[test]
    fn port_range_saturates_high() {
        let r = port_range(65530, 10);
        assert_eq!(*r.last().unwrap_or(&0), 65535);
    }
}
