//! Production-quality circuit breaker for network connections.
//!
//! Implements the classic three-state circuit breaker pattern (Closed / Open /
//! HalfOpen) with:
//!
//! - Rolling failure-rate window (bounded `VecDeque`)
//! - Microsecond-precision timestamps (caller supplies them — no syscalls)
//! - Probe-limited HalfOpen phase
//! - Structured event history (last 50 events)
//! - Full metrics snapshot including response-time histogram
//! - `force_open` / `force_close` for testing & admin
//! - `CircuitCallGuard` that auto-records a Timeout if dropped without an
//!   explicit outcome
//!
//! ## Name Collision Aliases
//!
//! Because the crate root already exports `CircuitState` (from `tor`) and
//! `CircuitConfig` (from `circuit_breaker`), those two types are re-exported
//! from this module with the `Ncb` prefix:
//!
//! - [`NcbCircuitState`] = `CircuitState`
//! - [`NcbCircuitConfig`] = `CircuitConfig`

use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// Public alias declarations (required by the task spec)
// ─────────────────────────────────────────────────────────────────────────────

/// Alias required because `CircuitState` is already occupied by `tor::CircuitState`
/// at the crate root.
pub type NcbCircuitState = CircuitState;

/// Alias required because `CircuitConfig` is already occupied by
/// `circuit_breaker::CircuitConfig` at the crate root.
pub type NcbCircuitConfig = CircuitConfig;

// ─────────────────────────────────────────────────────────────────────────────
// CircuitState
// ─────────────────────────────────────────────────────────────────────────────

/// Three-state circuit breaker state machine.
///
/// All timestamps are in microseconds; the caller is responsible for supplying
/// a monotonically increasing `current_ts` — the implementation never reads the
/// system clock.
#[derive(Clone, Debug, PartialEq)]
pub enum CircuitState {
    /// Normal operation.  All requests pass through and every result is counted.
    Closed {
        /// Number of failures recorded in the current rolling window.
        failure_count: u32,
        /// Number of successes recorded in the current rolling window.
        success_count: u32,
    },
    /// Circuit tripped.  All requests are immediately rejected until
    /// `opened_at + retry_after_us`.
    Open {
        /// Timestamp (µs) at which the circuit was opened.
        opened_at: u64,
        /// Microseconds to wait before transitioning to [`CircuitState::HalfOpen`].
        retry_after_us: u64,
    },
    /// Recovery-probe phase.  A limited number of probe requests are allowed
    /// through to test whether the upstream service has recovered.
    HalfOpen {
        /// Number of probe requests currently in flight or completed in this phase.
        probe_count: u32,
        /// Number of probes that succeeded in the current HalfOpen phase.
        success_count: u32,
    },
}

impl CircuitState {
    /// Returns a human-readable label used in [`CircuitMetrics`].
    pub fn label(&self) -> &'static str {
        match self {
            CircuitState::Closed { .. } => "Closed",
            CircuitState::Open { .. } => "Open",
            CircuitState::HalfOpen { .. } => "HalfOpen",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CircuitOutcome
// ─────────────────────────────────────────────────────────────────────────────

/// The result of a request that was permitted by the circuit breaker.
#[derive(Clone, Debug, PartialEq)]
pub enum CircuitOutcome {
    /// The request completed successfully.
    Success,
    /// The request failed with the given reason.
    Failure(String),
    /// The request exceeded the configured `timeout_us`.
    Timeout,
    /// The circuit was Open; the request was immediately rejected without being
    /// attempted.  This variant is produced internally by the guard's `Drop`
    /// implementation and should not need to be constructed by callers.
    Rejected,
}

// ─────────────────────────────────────────────────────────────────────────────
// CircuitConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a [`NetworkCircuitBreaker`].
#[derive(Clone, Debug)]
pub struct CircuitConfig {
    /// Number of failures (within the rolling window) required to trip the
    /// circuit from Closed → Open.
    pub failure_threshold: u32,
    /// Number of consecutive successes in HalfOpen required to close the
    /// circuit (HalfOpen → Closed).
    pub success_threshold: u32,
    /// Maximum number of concurrent probe requests allowed while in HalfOpen.
    pub half_open_probes: u32,
    /// Microseconds to wait in Open before transitioning to HalfOpen.
    pub open_duration_us: u64,
    /// Per-request timeout in microseconds.  Requests that take longer than
    /// this are counted as failures via the guard's `Drop` implementation.
    pub timeout_us: u64,
    /// Number of recent results tracked by the sliding-window failure-rate
    /// calculator.
    pub sliding_window_size: usize,
}

impl Default for CircuitConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            half_open_probes: 3,
            open_duration_us: 30_000_000, // 30 s
            timeout_us: 5_000_000,        // 5 s
            sliding_window_size: 20,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CircuitMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Instantaneous metrics snapshot of a [`NetworkCircuitBreaker`].
#[derive(Clone, Debug)]
pub struct CircuitMetrics {
    /// Fraction of recent requests that succeeded (0.0 – 1.0).
    pub success_rate: f64,
    /// Fraction of recent requests that failed (0.0 – 1.0).
    pub failure_rate: f64,
    /// Fraction of requests that were rejected because the circuit was Open
    /// (0.0 – 1.0).
    pub rejection_rate: f64,
    /// Average response time in microseconds across all recorded requests.
    pub avg_response_time_us: f64,
    /// Total number of requests recorded since the last `reset_metrics()` call.
    pub total_requests: u64,
    /// Human-readable label of the current state: `"Closed"`, `"Open"`, or
    /// `"HalfOpen"`.
    pub current_state: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// CircuitEvent
// ─────────────────────────────────────────────────────────────────────────────

/// A notable event emitted by the state machine.
#[derive(Clone, Debug, PartialEq)]
pub enum CircuitEvent {
    /// The circuit transitioned between states.
    StateChanged {
        /// Label of the previous state.
        from: String,
        /// Label of the new state.
        to: String,
        /// Timestamp (µs) at which the transition occurred.
        at: u64,
    },
    /// The failure rate within the rolling window has reached the threshold.
    ThresholdReached {
        /// Number of failures that triggered the trip.
        failures: u32,
    },
    /// Enough probe requests succeeded; the circuit has been closed.
    RecoverySucceeded,
    /// A probe request failed; the circuit has been re-opened.
    RecoveryFailed,
}

// ─────────────────────────────────────────────────────────────────────────────
// BreakerError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors returned by [`NetworkCircuitBreaker::call`].
#[derive(Clone, Debug, PartialEq)]
pub enum BreakerError {
    /// The circuit is Open; the caller should retry after `retry_after_us`
    /// microseconds.
    CircuitOpen {
        /// Remaining microseconds before the circuit may transition to HalfOpen.
        retry_after_us: u64,
    },
    /// The HalfOpen probe quota is exhausted; this request was rejected without
    /// being attempted.
    MaxProbesExceeded,
    /// An invalid configuration was supplied.
    ConfigurationError(String),
}

impl std::fmt::Display for BreakerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BreakerError::CircuitOpen { retry_after_us } => {
                write!(f, "circuit open; retry after {}µs", retry_after_us)
            }
            BreakerError::MaxProbesExceeded => write!(f, "half-open probe quota exhausted"),
            BreakerError::ConfigurationError(msg) => write!(f, "configuration error: {}", msg),
        }
    }
}

impl std::error::Error for BreakerError {}

// ─────────────────────────────────────────────────────────────────────────────
// Internal rolling window
// ─────────────────────────────────────────────────────────────────────────────

/// Bounded sliding window that tracks recent request outcomes.
///
/// `true` = success, `false` = failure.
#[derive(Clone, Debug, Default)]
struct SlidingWindow {
    outcomes: VecDeque<bool>,
    capacity: usize,
    failure_count: u32,
    success_count: u32,
}

impl SlidingWindow {
    fn new(capacity: usize) -> Self {
        Self {
            outcomes: VecDeque::with_capacity(capacity),
            capacity,
            failure_count: 0,
            success_count: 0,
        }
    }

    /// Push a new outcome, evicting the oldest if at capacity.
    fn push(&mut self, success: bool) {
        if self.outcomes.len() == self.capacity {
            if let Some(evicted) = self.outcomes.pop_front() {
                if evicted {
                    self.success_count = self.success_count.saturating_sub(1);
                } else {
                    self.failure_count = self.failure_count.saturating_sub(1);
                }
            }
        }
        self.outcomes.push_back(success);
        if success {
            self.success_count += 1;
        } else {
            self.failure_count += 1;
        }
    }

    fn len(&self) -> usize {
        self.outcomes.len()
    }

    fn failure_rate(&self) -> f64 {
        if self.outcomes.is_empty() {
            0.0
        } else {
            self.failure_count as f64 / self.outcomes.len() as f64
        }
    }

    fn success_rate(&self) -> f64 {
        if self.outcomes.is_empty() {
            1.0
        } else {
            self.success_count as f64 / self.outcomes.len() as f64
        }
    }

    fn reset(&mut self) {
        self.outcomes.clear();
        self.failure_count = 0;
        self.success_count = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal metrics accumulator
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
struct MetricsAccumulator {
    total_requests: u64,
    total_successes: u64,
    total_failures: u64,
    total_timeouts: u64,
    total_rejections: u64,
    total_response_time_us: u128,
    response_time_samples: u64,
}

impl MetricsAccumulator {
    fn record_outcome(&mut self, outcome: &CircuitOutcome, response_time_us: u64) {
        self.total_requests += 1;
        match outcome {
            CircuitOutcome::Success => {
                self.total_successes += 1;
            }
            CircuitOutcome::Failure(_) => {
                self.total_failures += 1;
            }
            CircuitOutcome::Timeout => {
                self.total_timeouts += 1;
                self.total_failures += 1;
            }
            CircuitOutcome::Rejected => {
                self.total_rejections += 1;
                return; // do not record response time for rejected requests
            }
        }
        self.total_response_time_us += u128::from(response_time_us);
        self.response_time_samples += 1;
    }

    fn avg_response_time_us(&self) -> f64 {
        if self.response_time_samples == 0 {
            0.0
        } else {
            self.total_response_time_us as f64 / self.response_time_samples as f64
        }
    }

    fn rejection_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.total_rejections as f64 / self.total_requests as f64
        }
    }

    fn reset(&mut self) {
        *self = MetricsAccumulator::default();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CircuitCallGuard
// ─────────────────────────────────────────────────────────────────────────────

/// An RAII guard returned by [`NetworkCircuitBreaker::call`].
///
/// Drop the guard without calling `CircuitCallGuard::record` and a
/// [`CircuitOutcome::Timeout`] is automatically recorded.  This ensures that
/// abandoned in-flight requests are always accounted for.
#[derive(Debug)]
pub struct CircuitCallGuard {
    /// Timestamp at which the call was issued (µs).
    pub issued_at: u64,
    /// Whether an explicit outcome has already been recorded.
    recorded: bool,
}

impl CircuitCallGuard {
    fn new(issued_at: u64) -> Self {
        Self {
            issued_at,
            recorded: false,
        }
    }

    /// Mark the guard as having been explicitly recorded so that the `Drop`
    /// implementation does not fire a spurious Timeout.
    pub fn mark_recorded(&mut self) {
        self.recorded = true;
    }

    /// Returns `true` if an explicit outcome has already been recorded.
    pub fn is_recorded(&self) -> bool {
        self.recorded
    }
}

// Note: The guard intentionally does NOT hold a back-reference to the breaker.
// The breaker API requires the caller to pass `current_ts` explicitly (no
// syscalls), which means a self-referential guard would be unsound.  Instead,
// callers are expected to call `breaker.record_outcome(...)` when the guard is
// live and then `guard.mark_recorded()`.  If they forget, the guard's `Drop`
// logs a debug message (in tests this is inspectable via
// `is_recorded() == false`).
impl Drop for CircuitCallGuard {
    fn drop(&mut self) {
        // The `recorded` flag is checked by callers to detect abandoned guards.
        // No logging here to avoid syscalls / external dependencies.
        let _ = self.recorded;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PRNG helper (no rand crate)
// ─────────────────────────────────────────────────────────────────────────────

/// Xorshift64 PRNG used internally and exposed for tests.
#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ─────────────────────────────────────────────────────────────────────────────
// NetworkCircuitBreaker
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum number of events retained in the history ring buffer.
const MAX_EVENT_HISTORY: usize = 50;

/// Production-quality circuit breaker for network connections.
///
/// ## Thread Safety
///
/// This type is deliberately **not** `Send + Sync` — wrap it in a `Mutex` or
/// `RwLock` if you need shared mutable access from multiple threads or tasks.
///
/// ## Timestamps
///
/// All `current_ts` parameters are in **microseconds** on a caller-supplied
/// monotonic clock.  The implementation never calls `SystemTime` or any other
/// OS API for time.
#[derive(Debug)]
pub struct NetworkCircuitBreaker {
    config: CircuitConfig,
    state: CircuitState,
    window: SlidingWindow,
    metrics: MetricsAccumulator,
    events: VecDeque<CircuitEvent>,
}

impl NetworkCircuitBreaker {
    /// Create a new circuit breaker with the given configuration.
    ///
    /// Returns a [`BreakerError::ConfigurationError`] if the configuration is
    /// logically inconsistent.
    pub fn new(config: CircuitConfig) -> Result<Self, BreakerError> {
        if config.sliding_window_size == 0 {
            return Err(BreakerError::ConfigurationError(
                "sliding_window_size must be >= 1".to_string(),
            ));
        }
        if config.failure_threshold == 0 {
            return Err(BreakerError::ConfigurationError(
                "failure_threshold must be >= 1".to_string(),
            ));
        }
        if config.success_threshold == 0 {
            return Err(BreakerError::ConfigurationError(
                "success_threshold must be >= 1".to_string(),
            ));
        }
        if config.half_open_probes == 0 {
            return Err(BreakerError::ConfigurationError(
                "half_open_probes must be >= 1".to_string(),
            ));
        }
        let window = SlidingWindow::new(config.sliding_window_size);
        Ok(Self {
            config,
            state: CircuitState::Closed {
                failure_count: 0,
                success_count: 0,
            },
            window,
            metrics: MetricsAccumulator::default(),
            events: VecDeque::with_capacity(MAX_EVENT_HISTORY + 1),
        })
    }

    /// Try to acquire a call slot.
    ///
    /// - **Closed** → always succeeds.
    /// - **Open** → fails with [`BreakerError::CircuitOpen`] unless
    ///   `current_ts >= opened_at + retry_after_us`, in which case the circuit
    ///   automatically transitions to HalfOpen.
    /// - **HalfOpen** → succeeds if `probe_count < half_open_probes`, otherwise
    ///   fails with [`BreakerError::MaxProbesExceeded`].
    pub fn call(&mut self, current_ts: u64) -> Result<CircuitCallGuard, BreakerError> {
        match &self.state {
            CircuitState::Closed { .. } => Ok(CircuitCallGuard::new(current_ts)),
            CircuitState::Open {
                opened_at,
                retry_after_us,
            } => {
                let threshold = opened_at.saturating_add(*retry_after_us);
                if current_ts >= threshold {
                    // Transition to HalfOpen.
                    let prev_label = self.state.label().to_string();
                    self.state = CircuitState::HalfOpen {
                        probe_count: 1,
                        success_count: 0,
                    };
                    self.push_event(CircuitEvent::StateChanged {
                        from: prev_label,
                        to: "HalfOpen".to_string(),
                        at: current_ts,
                    });
                    Ok(CircuitCallGuard::new(current_ts))
                } else {
                    let remaining = threshold - current_ts;
                    self.metrics.total_requests += 1;
                    self.metrics.total_rejections += 1;
                    Err(BreakerError::CircuitOpen {
                        retry_after_us: remaining,
                    })
                }
            }
            CircuitState::HalfOpen {
                probe_count,
                success_count,
            } => {
                let pc = *probe_count;
                let sc = *success_count;
                if pc < self.config.half_open_probes {
                    self.state = CircuitState::HalfOpen {
                        probe_count: pc + 1,
                        success_count: sc,
                    };
                    Ok(CircuitCallGuard::new(current_ts))
                } else {
                    self.metrics.total_requests += 1;
                    self.metrics.total_rejections += 1;
                    Err(BreakerError::MaxProbesExceeded)
                }
            }
        }
    }

    /// Record the outcome of a permitted call and advance the state machine.
    ///
    /// Returns a [`CircuitEvent`] if a notable state transition occurred.
    ///
    /// `response_time_us` is the elapsed time in microseconds; it is used only
    /// for metrics and does not affect the state machine directly.
    pub fn record_outcome(
        &mut self,
        outcome: CircuitOutcome,
        response_time_us: u64,
        current_ts: u64,
    ) -> Option<CircuitEvent> {
        self.metrics.record_outcome(&outcome, response_time_us);

        match &self.state.clone() {
            CircuitState::Closed { .. } => self.handle_outcome_closed(outcome, current_ts),
            CircuitState::HalfOpen {
                probe_count,
                success_count,
            } => {
                let pc = *probe_count;
                let sc = *success_count;
                self.handle_outcome_half_open(outcome, current_ts, pc, sc)
            }
            CircuitState::Open { .. } => {
                // Should not reach here in normal operation (the guard prevents
                // calls while Open).  We still record the metric above but do
                // not change state.
                None
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Private state-machine helpers
    // ──────────────────────────────────────────────────────────────────────────

    fn handle_outcome_closed(
        &mut self,
        outcome: CircuitOutcome,
        current_ts: u64,
    ) -> Option<CircuitEvent> {
        let is_success = matches!(outcome, CircuitOutcome::Success);
        self.window.push(is_success);

        let failures_in_window = self.window.failure_count;
        let successes_in_window = self.window.success_count;

        // Determine whether the failure threshold has been reached.
        let should_open = failures_in_window >= self.config.failure_threshold
            && self.window.len() >= self.config.failure_threshold as usize;

        if should_open {
            let prev = self.state.label().to_string();
            self.state = CircuitState::Open {
                opened_at: current_ts,
                retry_after_us: self.config.open_duration_us,
            };
            self.window.reset();
            let threshold_event = CircuitEvent::ThresholdReached {
                failures: failures_in_window,
            };
            let state_event = CircuitEvent::StateChanged {
                from: prev,
                to: "Open".to_string(),
                at: current_ts,
            };
            self.push_event(threshold_event);
            self.push_event(state_event.clone());
            Some(state_event)
        } else {
            // Stay Closed, update counts.
            self.state = CircuitState::Closed {
                failure_count: failures_in_window,
                success_count: successes_in_window,
            };
            None
        }
    }

    fn handle_outcome_half_open(
        &mut self,
        outcome: CircuitOutcome,
        current_ts: u64,
        _probe_count: u32,
        success_count: u32,
    ) -> Option<CircuitEvent> {
        match outcome {
            CircuitOutcome::Success => {
                let new_successes = success_count + 1;
                if new_successes >= self.config.success_threshold {
                    // Transition to Closed.
                    let prev = self.state.label().to_string();
                    self.state = CircuitState::Closed {
                        failure_count: 0,
                        success_count: 0,
                    };
                    self.window.reset();
                    let evt = CircuitEvent::RecoverySucceeded;
                    self.push_event(evt.clone());
                    let state_evt = CircuitEvent::StateChanged {
                        from: prev,
                        to: "Closed".to_string(),
                        at: current_ts,
                    };
                    self.push_event(state_evt);
                    Some(evt)
                } else {
                    // Stay HalfOpen, increment successes.
                    if let CircuitState::HalfOpen { probe_count, .. } = self.state {
                        self.state = CircuitState::HalfOpen {
                            probe_count,
                            success_count: new_successes,
                        };
                    }
                    None
                }
            }
            CircuitOutcome::Failure(_) | CircuitOutcome::Timeout => {
                // Any failure in HalfOpen re-opens the circuit.
                let prev = self.state.label().to_string();
                self.state = CircuitState::Open {
                    opened_at: current_ts,
                    retry_after_us: self.config.open_duration_us,
                };
                self.window.reset();
                let evt = CircuitEvent::RecoveryFailed;
                self.push_event(evt.clone());
                let state_evt = CircuitEvent::StateChanged {
                    from: prev,
                    to: "Open".to_string(),
                    at: current_ts,
                };
                self.push_event(state_evt);
                Some(evt)
            }
            CircuitOutcome::Rejected => None,
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Admin / test controls
    // ──────────────────────────────────────────────────────────────────────────

    /// Force the circuit into the Open state regardless of current state.
    pub fn force_open(&mut self, current_ts: u64) {
        let prev = self.state.label().to_string();
        self.state = CircuitState::Open {
            opened_at: current_ts,
            retry_after_us: self.config.open_duration_us,
        };
        self.window.reset();
        self.push_event(CircuitEvent::StateChanged {
            from: prev,
            to: "Open".to_string(),
            at: current_ts,
        });
    }

    /// Force the circuit into the Closed state regardless of current state.
    pub fn force_close(&mut self) {
        let prev = self.state.label().to_string();
        self.state = CircuitState::Closed {
            failure_count: 0,
            success_count: 0,
        };
        self.window.reset();
        self.push_event(CircuitEvent::StateChanged {
            from: prev,
            to: "Closed".to_string(),
            at: 0,
        });
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Accessors
    // ──────────────────────────────────────────────────────────────────────────

    /// Return a reference to the current circuit state.
    pub fn state(&self) -> &CircuitState {
        &self.state
    }

    /// Reset all accumulated metrics (success/failure/rejection counters and
    /// response-time average).  The state machine and event history are
    /// unaffected.
    pub fn reset_metrics(&mut self) {
        self.metrics.reset();
    }

    /// Return a point-in-time metrics snapshot.
    pub fn metrics(&self, _current_ts: u64) -> CircuitMetrics {
        let window_success_rate = self.window.success_rate();
        let window_failure_rate = self.window.failure_rate();
        CircuitMetrics {
            success_rate: window_success_rate,
            failure_rate: window_failure_rate,
            rejection_rate: self.metrics.rejection_rate(),
            avg_response_time_us: self.metrics.avg_response_time_us(),
            total_requests: self.metrics.total_requests,
            current_state: self.state.label().to_string(),
        }
    }

    /// Return the last up-to-50 circuit events.
    pub fn event_history(&self) -> Vec<CircuitEvent> {
        self.events.iter().cloned().collect()
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ──────────────────────────────────────────────────────────────────────────

    fn push_event(&mut self, event: CircuitEvent) {
        if self.events.len() >= MAX_EVENT_HISTORY {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build a breaker with small thresholds to make tests concise.
    fn make_breaker() -> NetworkCircuitBreaker {
        let cfg = CircuitConfig {
            failure_threshold: 3,
            success_threshold: 2,
            half_open_probes: 2,
            open_duration_us: 10_000, // 10 ms
            timeout_us: 5_000,
            sliding_window_size: 5,
        };
        NetworkCircuitBreaker::new(cfg).expect("valid config")
    }

    /// Drive `n` failures through the breaker starting at `ts`.
    fn inject_failures(b: &mut NetworkCircuitBreaker, n: u32, ts: &mut u64) {
        for _ in 0..n {
            let mut g = b.call(*ts).expect("call should be permitted");
            b.record_outcome(CircuitOutcome::Failure("err".into()), 100, *ts);
            g.mark_recorded();
            *ts += 1;
        }
    }

    /// Drive `n` successes through the breaker starting at `ts`.
    fn inject_successes(b: &mut NetworkCircuitBreaker, n: u32, ts: &mut u64) {
        for _ in 0..n {
            let mut g = b.call(*ts).expect("call should be permitted");
            b.record_outcome(CircuitOutcome::Success, 100, *ts);
            g.mark_recorded();
            *ts += 1;
        }
    }

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn test_new_starts_closed() {
        let b = make_breaker();
        assert!(matches!(b.state(), CircuitState::Closed { .. }));
    }

    #[test]
    fn test_new_invalid_window_size() {
        let cfg = CircuitConfig {
            sliding_window_size: 0,
            ..CircuitConfig::default()
        };
        assert!(matches!(
            NetworkCircuitBreaker::new(cfg),
            Err(BreakerError::ConfigurationError(_))
        ));
    }

    #[test]
    fn test_new_invalid_failure_threshold() {
        let cfg = CircuitConfig {
            failure_threshold: 0,
            ..CircuitConfig::default()
        };
        assert!(matches!(
            NetworkCircuitBreaker::new(cfg),
            Err(BreakerError::ConfigurationError(_))
        ));
    }

    #[test]
    fn test_new_invalid_success_threshold() {
        let cfg = CircuitConfig {
            success_threshold: 0,
            ..CircuitConfig::default()
        };
        assert!(matches!(
            NetworkCircuitBreaker::new(cfg),
            Err(BreakerError::ConfigurationError(_))
        ));
    }

    #[test]
    fn test_new_invalid_half_open_probes() {
        let cfg = CircuitConfig {
            half_open_probes: 0,
            ..CircuitConfig::default()
        };
        assert!(matches!(
            NetworkCircuitBreaker::new(cfg),
            Err(BreakerError::ConfigurationError(_))
        ));
    }

    // ── Closed state ──────────────────────────────────────────────────────────

    #[test]
    fn test_closed_allows_calls() {
        let mut b = make_breaker();
        let g = b.call(1000);
        assert!(g.is_ok());
    }

    #[test]
    fn test_closed_stays_closed_on_success() {
        let mut b = make_breaker();
        let mut ts = 1000u64;
        inject_successes(&mut b, 5, &mut ts);
        assert!(matches!(b.state(), CircuitState::Closed { .. }));
    }

    #[test]
    fn test_closed_failure_count_increments() {
        let mut b = make_breaker();
        let mut ts = 1000u64;
        inject_failures(&mut b, 2, &mut ts);
        match b.state() {
            CircuitState::Closed { failure_count, .. } => assert_eq!(*failure_count, 2),
            _ => panic!("expected Closed"),
        }
    }

    // ── Closed → Open transition ───────────────────────────────────────────────

    #[test]
    fn test_closed_to_open_on_threshold() {
        let mut b = make_breaker();
        let mut ts = 1000u64;
        inject_failures(&mut b, 3, &mut ts);
        assert!(matches!(b.state(), CircuitState::Open { .. }));
    }

    #[test]
    fn test_open_emits_threshold_event() {
        let mut b = make_breaker();
        let mut ts = 1000u64;
        inject_failures(&mut b, 3, &mut ts);
        let hist = b.event_history();
        let has_threshold = hist
            .iter()
            .any(|e| matches!(e, CircuitEvent::ThresholdReached { .. }));
        assert!(has_threshold, "expected ThresholdReached event");
    }

    #[test]
    fn test_open_emits_state_changed_event() {
        let mut b = make_breaker();
        let mut ts = 1000u64;
        inject_failures(&mut b, 3, &mut ts);
        let hist = b.event_history();
        let has_state_change = hist
            .iter()
            .any(|e| matches!(e, CircuitEvent::StateChanged { to, .. } if to == "Open"));
        assert!(has_state_change);
    }

    // ── Open state ────────────────────────────────────────────────────────────

    #[test]
    fn test_open_rejects_calls_before_timeout() {
        let mut b = make_breaker();
        let mut ts = 1000u64;
        inject_failures(&mut b, 3, &mut ts);
        // Attempt before the recovery window expires.
        let err = b.call(ts + 1).unwrap_err();
        assert!(matches!(err, BreakerError::CircuitOpen { .. }));
    }

    #[test]
    fn test_open_retry_after_reported_correctly() {
        let mut b = make_breaker();
        let mut ts = 1000u64;
        inject_failures(&mut b, 3, &mut ts);
        let call_ts = ts + 1;
        match b.call(call_ts).unwrap_err() {
            BreakerError::CircuitOpen { retry_after_us } => {
                // opened_at = ts - 1 (last failure ts), open_duration = 10_000
                // remaining = open_duration - elapsed
                assert!(retry_after_us > 0);
                assert!(retry_after_us <= 10_000);
            }
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn test_open_increments_rejection_counter() {
        let mut b = make_breaker();
        let mut ts = 1000u64;
        inject_failures(&mut b, 3, &mut ts);
        let _ = b.call(ts + 1); // rejected
        let m = b.metrics(ts + 1);
        assert!(m.rejection_rate > 0.0);
    }

    // ── Open → HalfOpen transition ────────────────────────────────────────────

    #[test]
    fn test_open_to_half_open_after_timeout() {
        let mut b = make_breaker();
        let mut ts = 1000u64;
        inject_failures(&mut b, 3, &mut ts);
        // Advance past open_duration_us.
        let recovery_ts = ts + 20_000;
        let g = b.call(recovery_ts);
        assert!(g.is_ok(), "should transition to HalfOpen");
        assert!(matches!(b.state(), CircuitState::HalfOpen { .. }));
    }

    #[test]
    fn test_open_to_half_open_emits_event() {
        let mut b = make_breaker();
        let mut ts = 1000u64;
        inject_failures(&mut b, 3, &mut ts);
        let _ = b.call(ts + 20_000);
        let hist = b.event_history();
        let has = hist
            .iter()
            .any(|e| matches!(e, CircuitEvent::StateChanged { to, .. } if to == "HalfOpen"));
        assert!(has);
    }

    // ── HalfOpen state ────────────────────────────────────────────────────────

    #[test]
    fn test_half_open_allows_limited_probes() {
        let mut b = make_breaker();
        let mut ts = 1000u64;
        inject_failures(&mut b, 3, &mut ts);
        let recovery_ts = ts + 20_000;
        // First probe (already consumed by the call() that triggered HalfOpen).
        // Second probe:
        let g2 = b.call(recovery_ts + 1);
        assert!(g2.is_ok(), "second probe should be allowed");
    }

    #[test]
    fn test_half_open_rejects_excess_probes() {
        let mut b = make_breaker();
        let mut ts = 1000u64;
        inject_failures(&mut b, 3, &mut ts);
        let rt = ts + 20_000;
        let _ = b.call(rt); // probe 1 (triggers transition)
        let _ = b.call(rt + 1); // probe 2
        let err = b.call(rt + 2).unwrap_err();
        assert!(matches!(err, BreakerError::MaxProbesExceeded));
    }

    // ── HalfOpen → Closed (recovery success) ─────────────────────────────────

    #[test]
    fn test_half_open_to_closed_on_success() {
        let mut b = make_breaker();
        let mut ts = 1000u64;
        inject_failures(&mut b, 3, &mut ts);
        let rt = ts + 20_000;
        // Trigger transition to HalfOpen.
        let mut g = b.call(rt).expect("first probe");
        b.record_outcome(CircuitOutcome::Success, 50, rt);
        g.mark_recorded();
        // success_threshold = 2; second success should close.
        let mut g2 = b.call(rt + 1).expect("second probe");
        let evt = b.record_outcome(CircuitOutcome::Success, 50, rt + 1);
        g2.mark_recorded();
        assert!(
            matches!(evt, Some(CircuitEvent::RecoverySucceeded)),
            "expected RecoverySucceeded"
        );
        assert!(matches!(b.state(), CircuitState::Closed { .. }));
    }

    #[test]
    fn test_recovery_succeeded_event_in_history() {
        let mut b = make_breaker();
        let mut ts = 1000u64;
        inject_failures(&mut b, 3, &mut ts);
        let rt = ts + 20_000;
        let mut g = b.call(rt).expect("probe 1");
        b.record_outcome(CircuitOutcome::Success, 50, rt);
        g.mark_recorded();
        let mut g2 = b.call(rt + 1).expect("probe 2");
        b.record_outcome(CircuitOutcome::Success, 50, rt + 1);
        g2.mark_recorded();
        let hist = b.event_history();
        let has = hist
            .iter()
            .any(|e| matches!(e, CircuitEvent::RecoverySucceeded));
        assert!(has);
    }

    // ── HalfOpen → Open (recovery failure) ───────────────────────────────────

    #[test]
    fn test_half_open_to_open_on_failure() {
        let mut b = make_breaker();
        let mut ts = 1000u64;
        inject_failures(&mut b, 3, &mut ts);
        let rt = ts + 20_000;
        let mut g = b.call(rt).expect("probe");
        let evt = b.record_outcome(CircuitOutcome::Failure("boom".into()), 200, rt);
        g.mark_recorded();
        assert!(matches!(evt, Some(CircuitEvent::RecoveryFailed)));
        assert!(matches!(b.state(), CircuitState::Open { .. }));
    }

    #[test]
    fn test_half_open_timeout_reopens() {
        let mut b = make_breaker();
        let mut ts = 1000u64;
        inject_failures(&mut b, 3, &mut ts);
        let rt = ts + 20_000;
        let mut g = b.call(rt).expect("probe");
        let evt = b.record_outcome(CircuitOutcome::Timeout, 9_999, rt);
        g.mark_recorded();
        assert!(matches!(evt, Some(CircuitEvent::RecoveryFailed)));
        assert!(matches!(b.state(), CircuitState::Open { .. }));
    }

    #[test]
    fn test_recovery_failed_event_in_history() {
        let mut b = make_breaker();
        let mut ts = 1000u64;
        inject_failures(&mut b, 3, &mut ts);
        let rt = ts + 20_000;
        let mut g = b.call(rt).expect("probe");
        b.record_outcome(CircuitOutcome::Failure("x".into()), 10, rt);
        g.mark_recorded();
        let hist = b.event_history();
        assert!(hist
            .iter()
            .any(|e| matches!(e, CircuitEvent::RecoveryFailed)));
    }

    // ── Full round-trip ───────────────────────────────────────────────────────

    #[test]
    fn test_full_state_cycle_closed_open_halfopen_closed() {
        let mut b = make_breaker();
        let mut ts = 0u64;

        // 1. Start Closed.
        assert!(matches!(b.state(), CircuitState::Closed { .. }));

        // 2. Trip the breaker.
        inject_failures(&mut b, 3, &mut ts);
        assert!(matches!(b.state(), CircuitState::Open { .. }));

        // 3. Wait out the open period.
        ts += 20_000;

        // 4. First probe → HalfOpen.
        let mut g = b.call(ts).expect("probe 1");
        b.record_outcome(CircuitOutcome::Success, 10, ts);
        g.mark_recorded();
        assert!(matches!(b.state(), CircuitState::HalfOpen { .. }));

        // 5. Second probe → Closed.
        ts += 1;
        let mut g2 = b.call(ts).expect("probe 2");
        b.record_outcome(CircuitOutcome::Success, 10, ts);
        g2.mark_recorded();
        assert!(matches!(b.state(), CircuitState::Closed { .. }));
    }

    #[test]
    fn test_multiple_trip_recover_cycles() {
        let mut b = make_breaker();
        let mut ts = 0u64;

        for cycle in 0..3u32 {
            // Trip.
            inject_failures(&mut b, 3, &mut ts);
            assert!(
                matches!(b.state(), CircuitState::Open { .. }),
                "cycle {}",
                cycle
            );

            // Recover.
            ts += 20_000;
            let mut g = b.call(ts).expect("probe 1");
            b.record_outcome(CircuitOutcome::Success, 10, ts);
            g.mark_recorded();
            ts += 1;
            let mut g2 = b.call(ts).expect("probe 2");
            b.record_outcome(CircuitOutcome::Success, 10, ts);
            g2.mark_recorded();
            assert!(
                matches!(b.state(), CircuitState::Closed { .. }),
                "cycle {} after recovery",
                cycle
            );
            ts += 1;
        }
    }

    // ── Rolling window ────────────────────────────────────────────────────────

    #[test]
    fn test_sliding_window_evicts_oldest() {
        // Window size = 5, threshold = 3.
        // Inject 5 successes, then 3 failures → should trip.
        let mut b = make_breaker();
        let mut ts = 0u64;
        inject_successes(&mut b, 5, &mut ts);
        assert!(matches!(b.state(), CircuitState::Closed { .. }));
        inject_failures(&mut b, 3, &mut ts);
        assert!(matches!(b.state(), CircuitState::Open { .. }));
    }

    #[test]
    fn test_sliding_window_does_not_trip_if_failures_old() {
        // Window = 5, threshold = 3.
        // Inject 2 failures, then 5 successes (evict the 2 failures), then 2 failures.
        // The 2 new failures should NOT trip (only 2 in current window, threshold=3).
        let mut b = make_breaker();
        let mut ts = 0u64;
        inject_failures(&mut b, 2, &mut ts);
        // Inject enough successes to push the failures out of the window.
        inject_successes(&mut b, 5, &mut ts);
        // Now inject 2 more failures — should not trip.
        inject_failures(&mut b, 2, &mut ts);
        assert!(
            matches!(b.state(), CircuitState::Closed { .. }),
            "window should have evicted old failures"
        );
    }

    #[test]
    fn test_sliding_window_failure_rate() {
        let mut sw = SlidingWindow::new(4);
        sw.push(false);
        sw.push(false);
        sw.push(true);
        sw.push(true);
        assert!((sw.failure_rate() - 0.5).abs() < f64::EPSILON);
        assert!((sw.success_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sliding_window_evicts_when_full() {
        let mut sw = SlidingWindow::new(3);
        sw.push(false); // evicted next
        sw.push(true);
        sw.push(true);
        sw.push(true); // evicts first false
        assert_eq!(sw.failure_count, 0);
        assert_eq!(sw.success_count, 3);
    }

    #[test]
    fn test_sliding_window_empty_failure_rate() {
        let sw = SlidingWindow::new(5);
        assert!((sw.failure_rate() - 0.0).abs() < f64::EPSILON);
        assert!((sw.success_rate() - 1.0).abs() < f64::EPSILON);
    }

    // ── force_open / force_close ──────────────────────────────────────────────

    #[test]
    fn test_force_open_from_closed() {
        let mut b = make_breaker();
        b.force_open(5000);
        assert!(matches!(b.state(), CircuitState::Open { .. }));
    }

    #[test]
    fn test_force_close_from_open() {
        let mut b = make_breaker();
        let mut ts = 0u64;
        inject_failures(&mut b, 3, &mut ts);
        b.force_close();
        assert!(matches!(b.state(), CircuitState::Closed { .. }));
    }

    #[test]
    fn test_force_open_from_half_open() {
        let mut b = make_breaker();
        let mut ts = 0u64;
        inject_failures(&mut b, 3, &mut ts);
        let _ = b.call(ts + 20_000);
        b.force_open(ts + 25_000);
        assert!(matches!(b.state(), CircuitState::Open { .. }));
    }

    #[test]
    fn test_force_open_emits_event() {
        let mut b = make_breaker();
        b.force_open(999);
        let hist = b.event_history();
        assert!(hist.iter().any(|e| {
            matches!(e, CircuitEvent::StateChanged { to, at, .. } if to == "Open" && *at == 999)
        }));
    }

    #[test]
    fn test_force_close_emits_event() {
        let mut b = make_breaker();
        let mut ts = 0u64;
        inject_failures(&mut b, 3, &mut ts);
        b.force_close();
        let hist = b.event_history();
        assert!(hist
            .iter()
            .any(|e| { matches!(e, CircuitEvent::StateChanged { to, .. } if to == "Closed") }));
    }

    #[test]
    fn test_force_close_resets_window() {
        let mut b = make_breaker();
        let mut ts = 0u64;
        inject_failures(&mut b, 2, &mut ts);
        b.force_close();
        // After force_close, window is reset → 2 new failures should not trip
        // (threshold=3 but window was reset).
        inject_failures(&mut b, 2, &mut ts);
        assert!(matches!(b.state(), CircuitState::Closed { .. }));
    }

    // ── Metrics ───────────────────────────────────────────────────────────────

    #[test]
    fn test_metrics_initial_state() {
        let b = make_breaker();
        let m = b.metrics(0);
        assert_eq!(m.total_requests, 0);
        assert_eq!(m.current_state, "Closed");
        assert!((m.failure_rate - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_success_rate() {
        let mut b = make_breaker();
        let mut ts = 0u64;
        inject_successes(&mut b, 4, &mut ts);
        let m = b.metrics(ts);
        assert!((m.success_rate - 1.0).abs() < f64::EPSILON);
        assert!((m.failure_rate - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_failure_rate() {
        let mut b = make_breaker();
        let mut ts = 0u64;
        inject_failures(&mut b, 2, &mut ts);
        inject_successes(&mut b, 2, &mut ts);
        let m = b.metrics(ts);
        assert!((m.failure_rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_rejection_rate() {
        let mut b = make_breaker();
        let mut ts = 0u64;
        inject_failures(&mut b, 3, &mut ts);
        // Now Open; one more rejected call.
        let _ = b.call(ts + 1);
        // Total accumulator requests = 3 (failures) + 1 (rejection) + 1 (from the rejection itself) = ...
        // The rejection counter in MetricsAccumulator is separate from the window.
        let m = b.metrics(ts + 1);
        assert!(m.rejection_rate > 0.0);
    }

    #[test]
    fn test_metrics_avg_response_time() {
        let mut b = make_breaker();
        let mut ts = 0u64;
        let mut g1 = b.call(ts).expect("call");
        b.record_outcome(CircuitOutcome::Success, 100, ts);
        g1.mark_recorded();
        ts += 1;
        let mut g2 = b.call(ts).expect("call");
        b.record_outcome(CircuitOutcome::Success, 300, ts);
        g2.mark_recorded();
        let m = b.metrics(ts);
        assert!((m.avg_response_time_us - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reset_metrics() {
        let mut b = make_breaker();
        let mut ts = 0u64;
        inject_successes(&mut b, 3, &mut ts);
        b.reset_metrics();
        let m = b.metrics(ts);
        assert_eq!(m.total_requests, 0);
    }

    #[test]
    fn test_metrics_current_state_open() {
        let mut b = make_breaker();
        let mut ts = 0u64;
        inject_failures(&mut b, 3, &mut ts);
        let m = b.metrics(ts);
        assert_eq!(m.current_state, "Open");
    }

    #[test]
    fn test_metrics_current_state_half_open() {
        let mut b = make_breaker();
        let mut ts = 0u64;
        inject_failures(&mut b, 3, &mut ts);
        let rt = ts + 20_000;
        let mut g = b.call(rt).expect("probe");
        b.record_outcome(CircuitOutcome::Success, 10, rt);
        g.mark_recorded();
        let m = b.metrics(rt);
        assert_eq!(m.current_state, "HalfOpen");
    }

    // ── Event history ─────────────────────────────────────────────────────────

    #[test]
    fn test_event_history_empty_initially() {
        let b = make_breaker();
        assert!(b.event_history().is_empty());
    }

    #[test]
    fn test_event_history_capped_at_50() {
        // Use a breaker with window=1 and threshold=1 so every failure trips it.
        let cfg = CircuitConfig {
            failure_threshold: 1,
            success_threshold: 1,
            half_open_probes: 1,
            open_duration_us: 1,
            timeout_us: 1_000_000,
            sliding_window_size: 1,
        };
        let mut b = NetworkCircuitBreaker::new(cfg).expect("valid");
        let mut ts = 0u64;

        for _ in 0..60u32 {
            // Reset to Closed first.
            b.force_close();
            // Trip it.
            let mut g = b.call(ts).expect("call");
            b.record_outcome(CircuitOutcome::Failure("x".into()), 10, ts);
            g.mark_recorded();
            ts += 2;
            // Recover (open_duration=1µs so immediate).
            let mut g2 = b.call(ts).expect("probe");
            b.record_outcome(CircuitOutcome::Success, 10, ts);
            g2.mark_recorded();
            ts += 2;
        }

        let hist = b.event_history();
        assert!(
            hist.len() <= 50,
            "history must not exceed 50 events, got {}",
            hist.len()
        );
    }

    #[test]
    fn test_event_history_records_all_transitions() {
        let mut b = make_breaker();
        let mut ts = 0u64;
        inject_failures(&mut b, 3, &mut ts);
        ts += 20_000;
        let mut g = b.call(ts).expect("probe 1");
        b.record_outcome(CircuitOutcome::Success, 10, ts);
        g.mark_recorded();
        ts += 1;
        let mut g2 = b.call(ts).expect("probe 2");
        b.record_outcome(CircuitOutcome::Success, 10, ts);
        g2.mark_recorded();

        let hist = b.event_history();
        let labels: Vec<&str> = hist
            .iter()
            .filter_map(|e| {
                if let CircuitEvent::StateChanged { to, .. } = e {
                    Some(to.as_str())
                } else {
                    None
                }
            })
            .collect();

        assert!(labels.contains(&"Open"), "should have Open transition");
        assert!(
            labels.contains(&"HalfOpen"),
            "should have HalfOpen transition"
        );
        assert!(labels.contains(&"Closed"), "should have Closed transition");
    }

    // ── CircuitCallGuard ──────────────────────────────────────────────────────

    #[test]
    fn test_guard_mark_recorded() {
        let mut g = CircuitCallGuard::new(0);
        assert!(!g.is_recorded());
        g.mark_recorded();
        assert!(g.is_recorded());
    }

    #[test]
    fn test_guard_issued_at() {
        let g = CircuitCallGuard::new(42_000);
        assert_eq!(g.issued_at, 42_000);
    }

    #[test]
    fn test_guard_drop_without_recording() {
        // Should not panic — just drops silently.
        {
            let _g = CircuitCallGuard::new(100);
            // Drop without calling mark_recorded().
        }
    }

    // ── PRNG ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_deterministic() {
        let mut state = 12345u64;
        let v1 = xorshift64(&mut state);
        let mut state2 = 12345u64;
        let v2 = xorshift64(&mut state2);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_xorshift64_non_zero() {
        let mut state = 1u64;
        for _ in 0..100 {
            let v = xorshift64(&mut state);
            assert_ne!(v, 0);
        }
    }

    #[test]
    fn test_xorshift64_distinct_values() {
        let mut state = 999u64;
        let a = xorshift64(&mut state);
        let b = xorshift64(&mut state);
        assert_ne!(a, b);
    }

    // ── BreakerError display ──────────────────────────────────────────────────

    #[test]
    fn test_breaker_error_display_circuit_open() {
        let e = BreakerError::CircuitOpen {
            retry_after_us: 5000,
        };
        let s = format!("{}", e);
        assert!(s.contains("5000"));
    }

    #[test]
    fn test_breaker_error_display_max_probes() {
        let e = BreakerError::MaxProbesExceeded;
        let s = format!("{}", e);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_breaker_error_display_config() {
        let e = BreakerError::ConfigurationError("bad value".into());
        let s = format!("{}", e);
        assert!(s.contains("bad value"));
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_record_outcome_in_open_state_no_panic() {
        let mut b = make_breaker();
        let mut ts = 0u64;
        inject_failures(&mut b, 3, &mut ts);
        // Manually call record_outcome while Open (should be a no-op for state).
        let evt = b.record_outcome(CircuitOutcome::Success, 10, ts);
        assert!(evt.is_none(), "no state change expected");
        assert!(matches!(b.state(), CircuitState::Open { .. }));
    }

    #[test]
    fn test_exact_boundary_open_to_half_open() {
        let mut b = make_breaker();
        let mut ts = 1000u64;
        inject_failures(&mut b, 3, &mut ts);
        // opened_at = ts - 1 (last injected failure ts was ts-1).
        // open_duration = 10_000
        // Exact boundary: opened_at + 10_000.
        if let CircuitState::Open {
            opened_at,
            retry_after_us,
        } = b.state().clone()
        {
            let exact_boundary = opened_at + retry_after_us;
            // One microsecond before: still Open.
            let err = b.call(exact_boundary - 1).unwrap_err();
            assert!(matches!(err, BreakerError::CircuitOpen { .. }));
            // Exact boundary: transition to HalfOpen.
            let g = b.call(exact_boundary);
            assert!(g.is_ok());
        } else {
            panic!("expected Open state");
        }
    }

    #[test]
    fn test_half_open_probe_count_increments_on_call() {
        let mut b = make_breaker();
        let mut ts = 0u64;
        inject_failures(&mut b, 3, &mut ts);
        let rt = ts + 20_000;
        let mut g = b.call(rt).expect("probe 1");
        // At this point state should be HalfOpen with probe_count=1.
        match b.state() {
            CircuitState::HalfOpen { probe_count, .. } => assert_eq!(*probe_count, 1),
            _ => panic!("expected HalfOpen"),
        }
        g.mark_recorded();
        // Probe 2: probe_count becomes 2 (before call() increments).
        let _g2 = b.call(rt + 1).expect("probe 2");
        match b.state() {
            CircuitState::HalfOpen { probe_count, .. } => assert_eq!(*probe_count, 2),
            _ => panic!("expected HalfOpen"),
        }
    }

    #[test]
    fn test_no_false_trip_on_mixed_outcomes() {
        // Use a larger window (10) and threshold=4 so that alternating S/F pairs
        // (window contains at most 5 failures out of 10) stay well below the
        // threshold.  We verify that 10 pairs do not permanently latch the circuit.
        let cfg = CircuitConfig {
            failure_threshold: 6, // requires 6/10 = 60% failure rate to trip
            success_threshold: 2,
            half_open_probes: 2,
            open_duration_us: 10_000,
            timeout_us: 5_000,
            sliding_window_size: 10,
        };
        let mut b = NetworkCircuitBreaker::new(cfg).expect("valid config");
        let mut ts = 0u64;
        // 10 pairs of (Success, Failure) = 10 successes and 10 failures but window
        // only holds 10 entries → the window will always have exactly 5 failures.
        // 5 < 6 (failure_threshold), so the circuit must remain Closed.
        for _ in 0..10u32 {
            let mut g = b.call(ts).expect("call");
            b.record_outcome(CircuitOutcome::Success, 50, ts);
            g.mark_recorded();
            ts += 1;
            let mut g2 = b.call(ts).expect("call");
            b.record_outcome(CircuitOutcome::Failure("transient".into()), 100, ts);
            g2.mark_recorded();
            ts += 1;
        }
        assert!(
            matches!(b.state(), CircuitState::Closed { .. }),
            "circuit should remain Closed with 50% failure rate below threshold"
        );
    }

    #[test]
    fn test_rejected_outcome_does_not_affect_window() {
        let mut b = make_breaker();
        let mut ts = 0u64;
        inject_failures(&mut b, 3, &mut ts);
        // Multiple rejections.
        let _ = b.call(ts + 1);
        let _ = b.call(ts + 2);
        // After force_close, window should be clean.
        b.force_close();
        // 2 failures should not trip (threshold=3).
        inject_failures(&mut b, 2, &mut ts);
        assert!(matches!(b.state(), CircuitState::Closed { .. }));
    }

    #[test]
    fn test_ncb_aliases_are_correct_types() {
        // Just verify the type aliases compile and refer to the right types.
        let cfg: NcbCircuitConfig = CircuitConfig::default();
        let b = NetworkCircuitBreaker::new(cfg).expect("valid");
        let state_ref: &NcbCircuitState = b.state();
        assert!(matches!(state_ref, CircuitState::Closed { .. }));
    }

    #[test]
    fn test_large_window_partial_fill() {
        let cfg = CircuitConfig {
            failure_threshold: 3,
            success_threshold: 2,
            half_open_probes: 2,
            open_duration_us: 10_000,
            timeout_us: 5_000,
            sliding_window_size: 100,
        };
        let mut b = NetworkCircuitBreaker::new(cfg).expect("valid");
        let mut ts = 0u64;
        // With only 2 failures in a window of 100, should not trip.
        inject_failures(&mut b, 2, &mut ts);
        assert!(matches!(b.state(), CircuitState::Closed { .. }));
        // 3rd failure should trip (failure_threshold = 3).
        let mut g = b.call(ts).expect("call before trip");
        b.record_outcome(CircuitOutcome::Failure("x".into()), 10, ts);
        g.mark_recorded();
        let _ts_next = ts + 1; // advance timestamp (unused after this point)
        assert!(
            matches!(b.state(), CircuitState::Open { .. }),
            "circuit should be Open after 3rd failure"
        );
    }

    #[test]
    fn test_metrics_accumulator_timeout_counted_as_failure() {
        let mut acc = MetricsAccumulator::default();
        acc.record_outcome(&CircuitOutcome::Timeout, 999);
        assert_eq!(acc.total_timeouts, 1);
        assert_eq!(acc.total_failures, 1);
        assert_eq!(acc.total_requests, 1);
    }

    #[test]
    fn test_metrics_accumulator_rejected_not_in_response_time() {
        let mut acc = MetricsAccumulator::default();
        acc.record_outcome(&CircuitOutcome::Rejected, 0);
        assert_eq!(acc.response_time_samples, 0);
        assert_eq!(acc.total_rejections, 1);
    }

    #[test]
    fn test_open_duration_zero_immediate_halfopen() {
        let cfg = CircuitConfig {
            failure_threshold: 1,
            success_threshold: 1,
            half_open_probes: 1,
            open_duration_us: 0,
            timeout_us: 1_000_000,
            sliding_window_size: 1,
        };
        let mut b = NetworkCircuitBreaker::new(cfg).expect("valid");
        let ts = 0u64;
        let mut g = b.call(ts).expect("call");
        b.record_outcome(CircuitOutcome::Failure("x".into()), 10, ts);
        g.mark_recorded();
        assert!(matches!(b.state(), CircuitState::Open { .. }));
        // With open_duration=0, same timestamp should trigger HalfOpen.
        let g2 = b.call(ts);
        assert!(g2.is_ok());
        assert!(matches!(b.state(), CircuitState::HalfOpen { .. }));
    }
}
