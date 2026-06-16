//! Circuit breaker pattern for per-peer fault tolerance.
//!
//! Implements the classic Closed / Open / HalfOpen state machine to prevent
//! cascading failures in P2P networks.  Each peer gets its own independent
//! circuit breaker tracked by a [`CircuitBreakerRegistry`].
//!
//! ## State machine
//!
//! ```text
//!   ┌──────────────────────────────────────────────┐
//!   │  consecutive_failures >= failure_threshold   │
//!   ▼                                              │
//! Closed ──────────────────────────────────► Open  │
//!   ▲                                        │     │
//!   │  consecutive_successes >= success_threshold  │
//!   │                                        │     │
//!   │   now >= opened_at + timeout_ms        │     │
//!   └──────────── HalfOpen ◄────────────────┘     │
//!                    │                             │
//!                    └─── failure ────────────────►┘
//! ```

use std::collections::{HashMap, VecDeque};

// ──────────────────────────────────────────────────────────────────────────────
// CircuitState
// ──────────────────────────────────────────────────────────────────────────────

/// Internal state of a per-peer circuit breaker.
///
/// Timestamps are Unix milliseconds (or any monotonic u64 counter the caller
/// provides — the implementation never calls the system clock itself).
#[derive(Clone, Debug, PartialEq)]
pub enum CircuitBreakerState {
    /// Normal operation — all calls are allowed through.
    Closed,
    /// Circuit tripped — calls are rejected until `opened_at + timeout_ms`.
    Open {
        /// Timestamp (ms) at which the circuit was opened.
        opened_at: u64,
    },
    /// Recovery probe phase — a limited number of calls are allowed through.
    HalfOpen {
        /// Timestamp (ms) at which the half-open probe phase started.
        probe_start: u64,
    },
}

/// Public alias kept for backward-compatibility with the previous API surface.
pub type PeerCircuitState = CircuitBreakerState;

// ──────────────────────────────────────────────────────────────────────────────
// CircuitConfig
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration knobs for a [`PeerCircuitBreaker`].
#[derive(Clone, Debug)]
pub struct CircuitConfig {
    /// Consecutive failures required to transition from Closed → Open.
    pub failure_threshold: u32,
    /// Consecutive successes in HalfOpen required to transition → Closed.
    pub success_threshold: u32,
    /// Milliseconds to wait in Open before transitioning to HalfOpen.
    pub timeout_ms: u64,
    /// Maximum concurrent in-flight calls allowed while in HalfOpen.
    pub half_open_max_calls: u32,
    /// Calls that take longer than this (ms) count as failures even when
    /// the underlying operation reports success.
    pub slow_call_threshold_ms: u64,
    /// Sliding window size: the last `window_size` results are tracked for
    /// failure-rate calculation.
    pub window_size: u32,
}

impl Default for CircuitConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout_ms: 30_000,
            half_open_max_calls: 1,
            slow_call_threshold_ms: 5_000,
            window_size: 10,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CallResult
// ──────────────────────────────────────────────────────────────────────────────

/// Outcome of a single call attempt that the caller feeds back to the circuit
/// breaker via [`PeerCircuitBreaker::record_result`].
#[derive(Clone, Debug)]
pub enum CallResult {
    /// The call completed successfully.
    Success {
        /// Wall-clock duration of the call in milliseconds.
        duration_ms: u64,
    },
    /// The call returned an application-level error.
    Failure {
        /// Wall-clock duration of the call in milliseconds.
        duration_ms: u64,
        /// Human-readable description of the error.
        reason: String,
    },
    /// The call was cancelled or timed out before a response was received.
    Timeout {
        /// How long the call was in-flight before being aborted.
        duration_ms: u64,
    },
}

impl CallResult {
    /// Returns `true` if this result should be treated as a *failure* by the
    /// circuit breaker.
    ///
    /// A `Success` is treated as a failure when its `duration_ms` meets or
    /// exceeds `slow_call_threshold_ms`.
    pub fn is_failure(&self, slow_call_threshold_ms: u64) -> bool {
        match self {
            Self::Failure { .. } | Self::Timeout { .. } => true,
            Self::Success { duration_ms } => *duration_ms >= slow_call_threshold_ms,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CircuitStats
// ──────────────────────────────────────────────────────────────────────────────

/// Snapshot of statistics for a single peer's circuit breaker.
#[derive(Clone, Debug, Default)]
pub struct CircuitStats {
    /// Human-readable current state: `"Closed"`, `"Open"`, or `"HalfOpen"`.
    pub state: String,
    /// Number of failures recorded in the sliding window.
    pub failure_count: u32,
    /// Number of successes recorded in the sliding window.
    pub success_count: u32,
    /// Current run of consecutive failures (resets on any success).
    pub consecutive_failures: u32,
    /// Current run of consecutive successes (only meaningful in HalfOpen).
    pub consecutive_successes: u32,
    /// Total calls ever attempted (including rejected ones).
    pub total_calls: u64,
    /// Calls rejected because the circuit was Open (or HalfOpen at capacity).
    pub rejected_calls: u64,
    /// Timestamp (ms) of the most recent Open transition.
    pub last_opened_at: Option<u64>,
    /// Timestamp (ms) of the most recent Closed transition (recovery).
    pub last_closed_at: Option<u64>,
}

// ──────────────────────────────────────────────────────────────────────────────
// PeerCircuitBreaker
// ──────────────────────────────────────────────────────────────────────────────

/// Per-peer circuit breaker implementing the Closed / Open / HalfOpen state
/// machine.
///
/// The caller is responsible for providing monotonically increasing timestamps
/// (`now: u64`) — typically Unix milliseconds — so the implementation remains
/// fully testable without touching the system clock.
#[derive(Clone, Debug)]
pub struct PeerCircuitBreaker {
    /// Identifier of the peer this circuit is guarding.
    pub peer_id: String,
    /// Configuration.
    pub config: CircuitConfig,
    /// Current state of the circuit.
    pub state: CircuitBreakerState,
    /// Sliding window of the last `config.window_size` results.
    /// `true` = success, `false` = failure.
    pub window: VecDeque<bool>,
    /// Consecutive failures in the current Closed stretch.
    pub consecutive_failures: u32,
    /// Consecutive successes in the current HalfOpen stretch.
    pub consecutive_successes: u32,
    /// Total calls attempted (including rejected ones).
    pub total_calls: u64,
    /// Calls rejected (circuit Open or HalfOpen at capacity).
    pub rejected_calls: u64,
    /// Timestamp of the most recent Open transition.
    pub last_opened_at: Option<u64>,
    /// Timestamp of the most recent Closed (recovery) transition.
    pub last_closed_at: Option<u64>,
    /// Number of in-flight calls currently allowed in HalfOpen.
    pub half_open_calls: u32,
}

impl PeerCircuitBreaker {
    /// Create a new circuit breaker for `peer_id` with the supplied config.
    pub fn new(peer_id: String, config: CircuitConfig) -> Self {
        Self {
            peer_id,
            config,
            state: CircuitBreakerState::Closed,
            window: VecDeque::new(),
            consecutive_failures: 0,
            consecutive_successes: 0,
            total_calls: 0,
            rejected_calls: 0,
            last_opened_at: None,
            last_closed_at: None,
            half_open_calls: 0,
        }
    }

    // ── State query ──────────────────────────────────────────────────────────

    /// Returns `true` if a call should be allowed right now.
    ///
    /// Side-effect: if the circuit is Open and the timeout has elapsed, it is
    /// silently transitioned to HalfOpen before returning `true`.
    pub fn can_call(&mut self, now: u64) -> bool {
        match &self.state.clone() {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open { opened_at } => {
                if now >= opened_at + self.config.timeout_ms {
                    // Transition Open → HalfOpen and allow this call.
                    self.state = CircuitBreakerState::HalfOpen { probe_start: now };
                    self.half_open_calls = 0;
                    self.consecutive_successes = 0;
                    true
                } else {
                    false
                }
            }
            CircuitBreakerState::HalfOpen { .. } => {
                self.half_open_calls < self.config.half_open_max_calls
            }
        }
    }

    // ── Recording results ────────────────────────────────────────────────────

    /// Feed back the outcome of a call.
    ///
    /// Updates the sliding window, counters, and triggers state transitions.
    pub fn record_result(&mut self, result: CallResult, now: u64) {
        let is_failure = result.is_failure(self.config.slow_call_threshold_ms);
        self.push_window(!is_failure);

        match self.state.clone() {
            CircuitBreakerState::Closed => {
                self.record_closed(is_failure, now);
            }
            CircuitBreakerState::HalfOpen { .. } => {
                self.record_half_open(is_failure, now);
            }
            CircuitBreakerState::Open { .. } => {
                // Calls should not reach here in normal flow (can_call returns
                // false while Open), but we handle it gracefully.
            }
        }
    }

    /// Process a result while the circuit is Closed.
    fn record_closed(&mut self, is_failure: bool, now: u64) {
        if is_failure {
            self.consecutive_failures += 1;
            self.consecutive_successes = 0;
            if self.consecutive_failures >= self.config.failure_threshold {
                self.trip_open(now);
            }
        } else {
            self.consecutive_failures = 0;
            self.consecutive_successes += 1;
        }
    }

    /// Process a result while the circuit is HalfOpen.
    fn record_half_open(&mut self, is_failure: bool, now: u64) {
        if is_failure {
            // Any failure in HalfOpen immediately re-trips the circuit.
            self.consecutive_successes = 0;
            self.trip_open(now);
        } else {
            self.consecutive_successes += 1;
            self.consecutive_failures = 0;
            if self.consecutive_successes >= self.config.success_threshold {
                self.close(now);
            }
        }
    }

    /// Transition to Open state and record the timestamp.
    fn trip_open(&mut self, now: u64) {
        self.state = CircuitBreakerState::Open { opened_at: now };
        self.last_opened_at = Some(now);
        self.half_open_calls = 0;
    }

    /// Transition to Closed state and clear counters.
    fn close(&mut self, now: u64) {
        self.state = CircuitBreakerState::Closed;
        self.last_closed_at = Some(now);
        self.consecutive_failures = 0;
        self.consecutive_successes = 0;
        self.half_open_calls = 0;
    }

    // ── Sliding window helpers ───────────────────────────────────────────────

    /// Push a result into the sliding window, evicting the oldest entry when
    /// the window is full.
    fn push_window(&mut self, success: bool) {
        if self.window.len() >= self.config.window_size as usize {
            self.window.pop_front();
        }
        self.window.push_back(success);
    }

    // ── Metrics ─────────────────────────────────────────────────────────────

    /// Fraction of failures in the current sliding window.
    ///
    /// Returns `0.0` when the window is empty.
    pub fn failure_rate(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        let failures = self.window.iter().filter(|&&s| !s).count();
        failures as f64 / self.window.len() as f64
    }

    /// Number of successes in the current sliding window.
    fn window_success_count(&self) -> u32 {
        self.window.iter().filter(|&&s| s).count() as u32
    }

    /// Number of failures in the current sliding window.
    fn window_failure_count(&self) -> u32 {
        self.window.iter().filter(|&&s| !s).count() as u32
    }

    /// Force the circuit to Closed and reset all counters.
    pub fn reset(&mut self, now: u64) {
        self.state = CircuitBreakerState::Closed;
        self.window.clear();
        self.consecutive_failures = 0;
        self.consecutive_successes = 0;
        self.half_open_calls = 0;
        self.last_closed_at = Some(now);
    }

    /// Return a snapshot of current statistics.
    pub fn stats(&self) -> CircuitStats {
        let state_str = match &self.state {
            CircuitBreakerState::Closed => "Closed",
            CircuitBreakerState::Open { .. } => "Open",
            CircuitBreakerState::HalfOpen { .. } => "HalfOpen",
        };
        CircuitStats {
            state: state_str.to_string(),
            failure_count: self.window_failure_count(),
            success_count: self.window_success_count(),
            consecutive_failures: self.consecutive_failures,
            consecutive_successes: self.consecutive_successes,
            total_calls: self.total_calls,
            rejected_calls: self.rejected_calls,
            last_opened_at: self.last_opened_at,
            last_closed_at: self.last_closed_at,
        }
    }

    /// Convenience: return `true` if the circuit is currently Closed.
    pub fn is_closed(&self) -> bool {
        matches!(self.state, CircuitBreakerState::Closed)
    }

    /// Convenience: return `true` if the circuit is currently Open.
    pub fn is_open(&self) -> bool {
        matches!(self.state, CircuitBreakerState::Open { .. })
    }

    /// Convenience: return `true` if the circuit is currently HalfOpen.
    pub fn is_half_open(&self) -> bool {
        matches!(self.state, CircuitBreakerState::HalfOpen { .. })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// RegistryStats
// ──────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics across all peers in a [`CircuitBreakerRegistry`].
#[derive(Clone, Debug, Default)]
pub struct RegistryStats {
    /// Total number of tracked peers.
    pub total_peers: usize,
    /// Peers currently in Closed state.
    pub closed_count: usize,
    /// Peers currently in Open state.
    pub open_count: usize,
    /// Peers currently in HalfOpen state.
    pub half_open_count: usize,
    /// Sum of rejected calls across all peers.
    pub total_rejected_calls: u64,
}

// ──────────────────────────────────────────────────────────────────────────────
// PeerCircuit — backward-compatible thin wrapper
// ──────────────────────────────────────────────────────────────────────────────

/// Backward-compatible public struct that wraps the per-peer circuit data.
///
/// New code should prefer [`PeerCircuitBreaker`] directly.
#[derive(Clone, Debug)]
pub struct PeerCircuit {
    /// Peer identifier.
    pub peer_id: String,
    /// Current state of the circuit.
    pub state: CircuitBreakerState,
    /// Consecutive failures accumulated while Closed.
    pub consecutive_failures: u32,
    /// Consecutive successes accumulated while HalfOpen.
    pub probe_successes: u32,
    /// Accumulated totals.
    pub total_calls: u64,
    /// Rejected call count.
    pub rejected_calls: u64,
}

impl From<&PeerCircuitBreaker> for PeerCircuit {
    fn from(b: &PeerCircuitBreaker) -> Self {
        Self {
            peer_id: b.peer_id.clone(),
            state: b.state.clone(),
            consecutive_failures: b.consecutive_failures,
            probe_successes: b.consecutive_successes,
            total_calls: b.total_calls,
            rejected_calls: b.rejected_calls,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CircuitBreakerRegistry
// ──────────────────────────────────────────────────────────────────────────────

/// Registry that manages one [`PeerCircuitBreaker`] per peer.
///
/// All mutating methods take `now: u64` (caller-supplied timestamp in ms) so
/// the registry is fully testable without touching the system clock.
pub struct CircuitBreakerRegistry {
    breakers: HashMap<String, PeerCircuitBreaker>,
    default_config: CircuitConfig,
}

impl CircuitBreakerRegistry {
    /// Create an empty registry with the supplied default configuration.
    pub fn new(default_config: CircuitConfig) -> Self {
        Self {
            breakers: HashMap::new(),
            default_config,
        }
    }

    /// Return a mutable reference to the breaker for `peer_id`, creating one
    /// with the default configuration if it does not exist yet.
    pub fn get_or_create(&mut self, peer_id: &str) -> &mut PeerCircuitBreaker {
        let config = self.default_config.clone();
        self.breakers
            .entry(peer_id.to_string())
            .or_insert_with(|| PeerCircuitBreaker::new(peer_id.to_string(), config))
    }

    /// Return `true` if a call to `peer_id` is currently permitted.
    ///
    /// If the peer has no breaker yet it is implicitly Closed.
    pub fn can_call(&mut self, peer_id: &str, now: u64) -> bool {
        self.get_or_create(peer_id).can_call(now)
    }

    /// Record the outcome of a call to `peer_id`.
    pub fn record_result(&mut self, peer_id: &str, result: CallResult, now: u64) {
        let breaker = self.get_or_create(peer_id);
        breaker.total_calls += 1;
        breaker.record_result(result, now);
    }

    /// IDs of peers whose circuit is currently Open (still blocking calls).
    pub fn open_peers(&mut self, now: u64) -> Vec<String> {
        self.breakers
            .iter_mut()
            .filter_map(|(id, b)| {
                if let CircuitBreakerState::Open { opened_at } = &b.state {
                    // Only return if the circuit is *still* open (timeout not yet elapsed).
                    if now < opened_at + b.config.timeout_ms {
                        return Some(id.clone());
                    }
                }
                None
            })
            .collect()
    }

    /// IDs of peers whose circuit is currently HalfOpen.
    pub fn half_open_peers(&self) -> Vec<String> {
        self.breakers
            .iter()
            .filter_map(|(id, b)| {
                if matches!(b.state, CircuitBreakerState::HalfOpen { .. }) {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Force the circuit for `peer_id` to Closed.
    ///
    /// Returns `false` if no breaker exists for the peer.
    pub fn reset_peer(&mut self, peer_id: &str, now: u64) -> bool {
        match self.breakers.get_mut(peer_id) {
            Some(b) => {
                b.reset(now);
                true
            }
            None => false,
        }
    }

    /// Remove Closed peers that have fewer than `min_calls` total calls.
    ///
    /// Returns the number of peers evicted.
    pub fn evict_closed_peers(&mut self, min_calls: u64) -> usize {
        let before = self.breakers.len();
        self.breakers.retain(|_, b| {
            // Keep if: not Closed, OR has enough calls to be worth retaining.
            !matches!(b.state, CircuitBreakerState::Closed) || b.total_calls >= min_calls
        });
        before - self.breakers.len()
    }

    /// Aggregate statistics for the entire registry.
    pub fn registry_stats(&mut self, now: u64) -> RegistryStats {
        let mut stats = RegistryStats {
            total_peers: self.breakers.len(),
            ..Default::default()
        };
        for b in self.breakers.values_mut() {
            // Trigger any pending Open → HalfOpen transitions so counts are accurate.
            if let CircuitBreakerState::Open { opened_at } = &b.state {
                if now >= opened_at + b.config.timeout_ms {
                    let probe_start = now;
                    b.state = CircuitBreakerState::HalfOpen { probe_start };
                    b.half_open_calls = 0;
                    b.consecutive_successes = 0;
                }
            }
            match &b.state {
                CircuitBreakerState::Closed => stats.closed_count += 1,
                CircuitBreakerState::Open { .. } => stats.open_count += 1,
                CircuitBreakerState::HalfOpen { .. } => stats.half_open_count += 1,
            }
            stats.total_rejected_calls += b.rejected_calls;
        }
        stats
    }

    /// Number of peers currently tracked.
    pub fn len(&self) -> usize {
        self.breakers.len()
    }

    /// Returns `true` when no peers are tracked.
    pub fn is_empty(&self) -> bool {
        self.breakers.is_empty()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::circuit_breaker::{
        CallResult, CircuitBreakerRegistry, CircuitConfig, PeerCircuitBreaker, RegistryStats,
    };

    // ── helpers ──────────────────────────────────────────────────────────────

    fn default_config() -> CircuitConfig {
        CircuitConfig::default()
    }

    fn make_breaker(peer_id: &str) -> PeerCircuitBreaker {
        PeerCircuitBreaker::new(peer_id.to_string(), default_config())
    }

    fn success(ms: u64) -> CallResult {
        CallResult::Success { duration_ms: ms }
    }

    fn failure(ms: u64) -> CallResult {
        CallResult::Failure {
            duration_ms: ms,
            reason: "err".to_string(),
        }
    }

    fn timeout_result(ms: u64) -> CallResult {
        CallResult::Timeout { duration_ms: ms }
    }

    /// Push `n` failures into `b` at timestamp `now`.
    fn inject_failures(b: &mut PeerCircuitBreaker, n: u32, now: u64) {
        for _ in 0..n {
            b.record_result(failure(1), now);
        }
    }

    /// Trip the breaker to Open by injecting `failure_threshold` failures.
    fn trip_open(b: &mut PeerCircuitBreaker, now: u64) {
        inject_failures(b, b.config.failure_threshold, now);
    }

    // ── 1. New breaker is Closed ──────────────────────────────────────────────

    #[test]
    fn new_breaker_is_closed() {
        let b = make_breaker("p1");
        assert!(b.is_closed());
    }

    // ── 2. can_call returns true while Closed ─────────────────────────────────

    #[test]
    fn can_call_while_closed() {
        let mut b = make_breaker("p2");
        assert!(b.can_call(0));
    }

    // ── 3. Consecutive failures trip the circuit Open ─────────────────────────

    #[test]
    fn consecutive_failures_trip_open() {
        let mut b = make_breaker("p3");
        inject_failures(&mut b, 4, 0); // one below threshold
        assert!(b.is_closed(), "should still be Closed after 4 failures");
        b.record_result(failure(1), 0); // 5th failure
        assert!(
            b.is_open(),
            "should be Open after hitting failure_threshold"
        );
    }

    // ── 4. Open circuit rejects can_call ─────────────────────────────────────

    #[test]
    fn open_circuit_rejects_can_call() {
        let mut b = make_breaker("p4");
        trip_open(&mut b, 0);
        assert!(!b.can_call(1000), "should be rejected while Open");
    }

    // ── 5. Open transitions to HalfOpen after timeout ─────────────────────────

    #[test]
    fn open_transitions_to_half_open_after_timeout() {
        let mut b = make_breaker("p5");
        trip_open(&mut b, 0);
        // timeout_ms default = 30_000
        let result = b.can_call(30_000);
        assert!(result, "should allow call after timeout elapses");
        assert!(b.is_half_open(), "should be HalfOpen");
    }

    // ── 6. Open stays Open before timeout ────────────────────────────────────

    #[test]
    fn open_stays_open_before_timeout() {
        let mut b = make_breaker("p6");
        trip_open(&mut b, 0);
        assert!(
            !b.can_call(29_999),
            "should still be blocked before timeout"
        );
        assert!(b.is_open());
    }

    // ── 7. HalfOpen allows limited calls ─────────────────────────────────────

    #[test]
    fn half_open_allows_limited_calls() {
        let mut b = PeerCircuitBreaker::new(
            "p7".to_string(),
            CircuitConfig {
                half_open_max_calls: 2,
                ..default_config()
            },
        );
        trip_open(&mut b, 0);
        b.can_call(30_000); // triggers Open → HalfOpen
        assert!(b.is_half_open());

        b.half_open_calls = 1; // simulate one in-flight call
        assert!(b.can_call(30_001), "second call should be allowed");

        b.half_open_calls = 2; // at capacity
        assert!(!b.can_call(30_002), "third call should be rejected");
    }

    // ── 8. Enough successes in HalfOpen close the circuit ────────────────────

    #[test]
    fn half_open_successes_close_circuit() {
        let mut b = make_breaker("p8");
        trip_open(&mut b, 0);
        b.can_call(30_000); // → HalfOpen
        b.record_result(success(100), 30_001);
        assert!(b.is_half_open(), "still HalfOpen after 1 success (need 2)");
        b.record_result(success(100), 30_002);
        assert!(
            b.is_closed(),
            "should be Closed after success_threshold reached"
        );
    }

    // ── 9. Failure in HalfOpen re-trips the circuit ───────────────────────────

    #[test]
    fn half_open_failure_reopens() {
        let mut b = make_breaker("p9");
        trip_open(&mut b, 0);
        b.can_call(30_000); // → HalfOpen
        b.record_result(failure(1), 30_001);
        assert!(b.is_open(), "failure in HalfOpen should re-open circuit");
    }

    // ── 10. Slow success counts as failure ────────────────────────────────────

    #[test]
    fn slow_success_counts_as_failure() {
        let mut b = make_breaker("p10");
        // slow_call_threshold_ms default = 5_000; duration >= threshold = failure
        for _ in 0..5 {
            b.record_result(success(5_000), 0);
        }
        assert!(b.is_open(), "slow calls should trip the circuit");
    }

    // ── 11. Fast success does not count as failure ────────────────────────────

    #[test]
    fn fast_success_is_not_a_failure() {
        let mut b = make_breaker("p11");
        b.record_result(success(4_999), 0);
        assert!(b.is_closed());
        assert_eq!(b.consecutive_failures, 0);
    }

    // ── 12. failure_rate is 0.0 on empty window ───────────────────────────────

    #[test]
    fn failure_rate_empty_window() {
        let b = make_breaker("p12");
        assert_eq!(b.failure_rate(), 0.0);
    }

    // ── 13. failure_rate calculation ──────────────────────────────────────────

    #[test]
    fn failure_rate_calculation() {
        let mut b = make_breaker("p13");
        b.record_result(success(1), 0);
        b.record_result(failure(1), 0);
        b.record_result(failure(1), 0);
        b.record_result(success(1), 0);
        // window: [true, false, false, true] → 2/4 = 0.5
        let rate = b.failure_rate();
        assert!((rate - 0.5).abs() < 1e-9, "rate={rate}");
    }

    // ── 14. Sliding window evicts oldest on overflow ──────────────────────────

    #[test]
    fn sliding_window_evicts_oldest() {
        let mut b = PeerCircuitBreaker::new(
            "p14".to_string(),
            CircuitConfig {
                window_size: 3,
                failure_threshold: 100, // prevent tripping
                ..default_config()
            },
        );
        b.record_result(failure(1), 0);
        b.record_result(failure(1), 0);
        b.record_result(failure(1), 0);
        assert_eq!(b.window.len(), 3);
        // Adding one more should evict the oldest failure.
        b.record_result(success(1), 0);
        assert_eq!(b.window.len(), 3);
        // Now 2 failures + 1 success remain.
        let rate = b.failure_rate();
        assert!((rate - 2.0 / 3.0).abs() < 1e-9, "rate={rate}");
    }

    // ── 15. reset() clears all state ─────────────────────────────────────────

    #[test]
    fn reset_clears_state() {
        let mut b = make_breaker("p15");
        trip_open(&mut b, 0);
        b.reset(1000);
        assert!(b.is_closed());
        assert_eq!(b.consecutive_failures, 0);
        assert_eq!(b.consecutive_successes, 0);
        assert!(b.window.is_empty());
        assert_eq!(b.last_closed_at, Some(1000));
    }

    // ── 16. stats() reflects current state ───────────────────────────────────

    #[test]
    fn stats_reflects_current_state() {
        let mut b = make_breaker("p16");
        b.total_calls = 10;
        b.rejected_calls = 2;
        b.record_result(failure(1), 0);
        let s = b.stats();
        assert_eq!(s.state, "Closed");
        assert_eq!(s.total_calls, 10);
        assert_eq!(s.rejected_calls, 2);
        assert_eq!(s.consecutive_failures, 1);
    }

    // ── 17. stats() shows Open state ──────────────────────────────────────────

    #[test]
    fn stats_shows_open_state() {
        let mut b = make_breaker("p17");
        trip_open(&mut b, 42);
        let s = b.stats();
        assert_eq!(s.state, "Open");
        assert_eq!(s.last_opened_at, Some(42));
    }

    // ── 18. last_opened_at set on trip ───────────────────────────────────────

    #[test]
    fn last_opened_at_set_on_trip() {
        let mut b = make_breaker("p18");
        trip_open(&mut b, 9999);
        assert_eq!(b.last_opened_at, Some(9999));
    }

    // ── 19. last_closed_at set on recovery ────────────────────────────────────

    #[test]
    fn last_closed_at_set_on_recovery() {
        let mut b = make_breaker("p19");
        trip_open(&mut b, 0);
        b.can_call(30_000); // → HalfOpen
        b.record_result(success(1), 30_001);
        b.record_result(success(1), 30_002);
        assert!(b.is_closed());
        assert_eq!(b.last_closed_at, Some(30_002));
    }

    // ── 20. Timeout CallResult is always a failure ────────────────────────────

    #[test]
    fn timeout_result_is_failure() {
        let r = timeout_result(100);
        assert!(r.is_failure(5_000), "Timeout should always be a failure");
    }

    // ── 21. Failure CallResult is always a failure ────────────────────────────

    #[test]
    fn failure_result_is_failure() {
        let r = failure(1);
        assert!(r.is_failure(5_000));
    }

    // ── 22. Success with duration < threshold is not a failure ────────────────

    #[test]
    fn success_below_threshold_not_failure() {
        let r = success(4_999);
        assert!(!r.is_failure(5_000));
    }

    // ── 23. Success with duration == threshold is a failure ───────────────────

    #[test]
    fn success_at_threshold_is_failure() {
        let r = success(5_000);
        assert!(r.is_failure(5_000));
    }

    // ── 24. Registry: get_or_create creates new breaker ──────────────────────

    #[test]
    fn registry_creates_new_breaker() {
        let mut reg = CircuitBreakerRegistry::new(default_config());
        let b = reg.get_or_create("r1");
        assert!(b.is_closed());
    }

    // ── 25. Registry: can_call delegates correctly ────────────────────────────

    #[test]
    fn registry_can_call_delegates() {
        let mut reg = CircuitBreakerRegistry::new(default_config());
        assert!(reg.can_call("r2", 0));
    }

    // ── 26. Registry: record_result increments total_calls ───────────────────

    #[test]
    fn registry_record_result_increments_total_calls() {
        let mut reg = CircuitBreakerRegistry::new(default_config());
        reg.record_result("r3", success(1), 0);
        reg.record_result("r3", success(1), 0);
        assert_eq!(reg.get_or_create("r3").total_calls, 2);
    }

    // ── 27. Registry: open_peers returns only Open peers ─────────────────────

    #[test]
    fn registry_open_peers() {
        let mut reg = CircuitBreakerRegistry::new(default_config());
        // Trip r4 but not r5.
        for _ in 0..5 {
            reg.record_result("r4", failure(1), 0);
        }
        reg.record_result("r5", success(1), 0);
        let open = reg.open_peers(0);
        assert!(open.contains(&"r4".to_string()));
        assert!(!open.contains(&"r5".to_string()));
    }

    // ── 28. Registry: half_open_peers returns only HalfOpen peers ────────────

    #[test]
    fn registry_half_open_peers() {
        let mut reg = CircuitBreakerRegistry::new(default_config());
        // Trip then allow timeout.
        for _ in 0..5 {
            reg.record_result("r6", failure(1), 0);
        }
        reg.can_call("r6", 30_000); // triggers → HalfOpen
        let ho = reg.half_open_peers();
        assert!(ho.contains(&"r6".to_string()));
    }

    // ── 29. Registry: reset_peer returns false for unknown peer ───────────────

    #[test]
    fn registry_reset_peer_unknown() {
        let mut reg = CircuitBreakerRegistry::new(default_config());
        assert!(!reg.reset_peer("nobody", 0));
    }

    // ── 30. Registry: reset_peer closes an Open circuit ──────────────────────

    #[test]
    fn registry_reset_peer_closes_open() {
        let mut reg = CircuitBreakerRegistry::new(default_config());
        for _ in 0..5 {
            reg.record_result("r7", failure(1), 0);
        }
        assert!(reg.reset_peer("r7", 100));
        assert!(reg.get_or_create("r7").is_closed());
    }

    // ── 31. Registry: evict_closed_peers removes low-traffic peers ────────────

    #[test]
    fn registry_evict_closed_peers() {
        let mut reg = CircuitBreakerRegistry::new(default_config());
        reg.record_result("low", success(1), 0); // 1 total call
        reg.record_result("high", success(1), 0);
        reg.record_result("high", success(1), 0);
        reg.record_result("high", success(1), 0); // 3 total calls

        let evicted = reg.evict_closed_peers(2); // min_calls = 2
        assert_eq!(evicted, 1, "should evict 'low' (1 call < 2)");
        assert!(!reg.breakers.contains_key("low"));
        assert!(reg.breakers.contains_key("high"));
    }

    // ── 32. Registry: evict does NOT remove Open peers ────────────────────────

    #[test]
    fn registry_evict_skips_open_peers() {
        let mut reg = CircuitBreakerRegistry::new(default_config());
        for _ in 0..5 {
            reg.record_result("open-peer", failure(1), 0);
        }
        // open-peer has 5 calls but is Open; min_calls = 10 would evict it if
        // Closed, but it should be kept because it is Open.
        let evicted = reg.evict_closed_peers(10);
        assert_eq!(evicted, 0, "should not evict Open peer");
        assert!(reg.breakers.contains_key("open-peer"));
    }

    // ── 33. Registry: registry_stats counts states correctly ─────────────────

    #[test]
    fn registry_stats_counts() {
        let mut reg = CircuitBreakerRegistry::new(default_config());
        reg.record_result("s1", success(1), 0); // Closed
        for _ in 0..5 {
            reg.record_result("s2", failure(1), 0); // Open
        }
        let stats: RegistryStats = reg.registry_stats(0);
        assert_eq!(stats.total_peers, 2);
        assert_eq!(stats.closed_count, 1);
        assert_eq!(stats.open_count, 1);
        assert_eq!(stats.half_open_count, 0);
    }

    // ── 34. Registry: total_rejected_calls aggregates correctly ───────────────

    #[test]
    fn registry_stats_total_rejected() {
        let mut reg = CircuitBreakerRegistry::new(default_config());
        // Trip peer and then record rejections manually.
        for _ in 0..5 {
            reg.record_result("rj", failure(1), 0);
        }
        {
            let b = reg.get_or_create("rj");
            b.rejected_calls = 7;
        }
        let stats = reg.registry_stats(0);
        assert_eq!(stats.total_rejected_calls, 7);
    }

    // ── 35. Multiple trips accumulate last_opened_at ──────────────────────────

    #[test]
    fn multiple_trips_update_last_opened_at() {
        let mut b = make_breaker("p35");
        trip_open(&mut b, 100);
        assert_eq!(b.last_opened_at, Some(100));

        // Recover.
        b.can_call(130_100); // → HalfOpen
        b.record_result(success(1), 130_101);
        b.record_result(success(1), 130_102); // → Closed

        // Trip again at a later time.
        trip_open(&mut b, 200_000);
        assert_eq!(b.last_opened_at, Some(200_000));
    }

    // ── 36. Timeout CallResult trips the circuit ──────────────────────────────

    #[test]
    fn timeout_result_trips_circuit() {
        let mut b = make_breaker("p36");
        for _ in 0..5 {
            b.record_result(timeout_result(10_000), 0);
        }
        assert!(b.is_open(), "repeated timeouts should trip circuit Open");
    }

    // ── 37. Window counts match after mixed results ───────────────────────────

    #[test]
    fn window_counts_after_mixed_results() {
        let mut b = PeerCircuitBreaker::new(
            "p37".to_string(),
            CircuitConfig {
                window_size: 6,
                failure_threshold: 100,
                ..default_config()
            },
        );
        for _ in 0..3 {
            b.record_result(success(1), 0);
        }
        for _ in 0..3 {
            b.record_result(failure(1), 0);
        }
        let s = b.stats();
        assert_eq!(s.success_count, 3);
        assert_eq!(s.failure_count, 3);
    }

    // ── 38. CircuitStats: last_opened_at / last_closed_at from stats() ────────

    #[test]
    fn stats_timestamps_propagated() {
        let mut b = make_breaker("p38");
        trip_open(&mut b, 5555);
        b.can_call(35_555); // → HalfOpen
        b.record_result(success(1), 35_556);
        b.record_result(success(1), 35_557); // → Closed

        let s = b.stats();
        assert_eq!(s.last_opened_at, Some(5555));
        assert_eq!(s.last_closed_at, Some(35_557));
    }

    // ── 39. is_closed / is_open / is_half_open helpers ───────────────────────

    #[test]
    fn state_helpers_correct() {
        let mut b = make_breaker("p39");
        assert!(b.is_closed());
        assert!(!b.is_open());
        assert!(!b.is_half_open());

        trip_open(&mut b, 0);
        assert!(!b.is_closed());
        assert!(b.is_open());
        assert!(!b.is_half_open());

        b.can_call(30_000);
        assert!(!b.is_closed());
        assert!(!b.is_open());
        assert!(b.is_half_open());
    }

    // ── 40. Registry is_empty / len ──────────────────────────────────────────

    #[test]
    fn registry_len_and_is_empty() {
        let mut reg = CircuitBreakerRegistry::new(default_config());
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        reg.can_call("x", 0);
        assert!(!reg.is_empty());
        assert_eq!(reg.len(), 1);
    }
}
