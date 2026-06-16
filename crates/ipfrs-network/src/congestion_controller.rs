//! TCP-inspired multi-algorithm congestion controller for IPFRS peer data streams.
//!
//! Implements SlowStart, CongestionAvoidance, FastRecovery, and Idle phases with
//! five distinct algorithms: Reno, Cubic, BBR, Vegas, and Westwood.
//!
//! # Legacy API
//!
//! The original per-peer `PeerCongestionController` and `MultiPeerCongestionManager`
//! are retained for backwards compatibility.
//!
//! # New API
//!
//! Use [`CongestionController`] for full multi-algorithm support.

use std::collections::{HashMap, VecDeque};

// ─── type aliases ────────────────────────────────────────────────────────────

/// Connection identifier type alias.
pub type CccConnId = u64;
/// Main congestion controller type alias.
pub type CccCongestionController = CongestionController;
/// Decision returned from ack/loss/timeout handlers.
pub type CccDecision = Decision;

// ─── inline PRNG for deterministic jitter ────────────────────────────────────

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ─── Algorithm ───────────────────────────────────────────────────────────────

/// Congestion control algorithm selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CccAlgorithm {
    /// TCP Reno: AIMD additive-increase/multiplicative-decrease.
    #[default]
    Reno,
    /// TCP CUBIC: cubic growth function for high-BDP networks.
    Cubic,
    /// BBR: Bottleneck Bandwidth and RTT model-based control.
    Bbr,
    /// TCP Vegas: RTT-based proactive congestion avoidance.
    Vegas,
    /// Westwood+: bandwidth estimation on loss for fast recovery.
    Westwood,
}

// ─── State ───────────────────────────────────────────────────────────────────

/// Congestion controller FSM state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CccState {
    /// Exponential window growth until ssthresh is reached.
    SlowStart,
    /// Linear (or algorithm-defined) window growth.
    #[default]
    CongestionAvoidance,
    /// Recovering from packet loss via limited retransmit.
    FastRecovery,
    /// Connection is quiescent (no in-flight data).
    Idle,
}

// ─── EventType ───────────────────────────────────────────────────────────────

/// Kinds of congestion events logged in the event ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CccEventType {
    /// An ACK arrived.
    AckReceived,
    /// A packet was declared lost.
    PacketLost,
    /// Retransmission timeout triggered.
    Timeout,
    /// Fast retransmit triggered (3 DUP-ACKs).
    FastRetransmit,
    /// Controller entered slow-start.
    SlowStartEnter,
    /// Controller entered congestion avoidance.
    CaEnter,
}

// ─── Event ───────────────────────────────────────────────────────────────────

/// A single congestion event record.
#[derive(Clone, Debug)]
pub struct CccEvent {
    /// Monotonic timestamp (ms since controller creation, from connection `last_ts`).
    pub ts: u64,
    /// Connection this event belongs to.
    pub conn_id: CccConnId,
    /// What happened.
    pub event_type: CccEventType,
    /// Congestion window before the event.
    pub cwnd_before: u64,
    /// Congestion window after the event.
    pub cwnd_after: u64,
}

// ─── Decision ────────────────────────────────────────────────────────────────

/// Result returned by `on_ack`, `on_loss`, and `on_timeout`.
#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    /// New congestion window (bytes).
    pub new_cwnd: u64,
    /// New slow-start threshold (bytes).
    pub new_ssthresh: u64,
    /// New FSM state.
    pub new_state: CccState,
    /// Recommended sending rate (bytes/sec), if RTT is known.
    pub sending_rate: Option<f64>,
}

// ─── Connection ──────────────────────────────────────────────────────────────

/// Per-connection congestion state.
#[derive(Clone, Debug)]
pub struct CccConnection {
    /// Unique connection identifier.
    pub id: CccConnId,
    /// Current congestion window (bytes).
    pub cwnd: u64,
    /// Slow-start threshold (bytes).
    pub ssthresh: u64,
    /// Current FSM state.
    pub state: CccState,
    /// Smoothed RTT (ms), exponential weighted moving average.
    pub rtt_ms: f64,
    /// RTT variance (ms).
    pub rtt_var: f64,
    /// Bytes currently in flight.
    pub in_flight: u64,
    /// Total bytes acknowledged.
    pub bytes_acked: u64,
    /// Total bytes lost.
    pub bytes_lost: u64,
    /// Timestamp of the last event (monotonic ms counter, internal).
    pub last_ts: u64,
    // ── Cubic-specific ──────────────────────────────────────────────────────
    /// Time (in RTTs) since last congestion event (for Cubic).
    cubic_k: f64,
    /// Window size at last congestion event (for Cubic).
    cubic_w_max: f64,
    // ── BBR-specific ─────────────────────────────────────────────────────────
    /// Estimated bottleneck bandwidth (bytes/sec).
    bbr_bw: f64,
    /// Minimum observed RTT for BBR (ms).
    bbr_min_rtt: f64,
    // ── Vegas-specific ────────────────────────────────────────────────────────
    /// Minimum observed RTT for Vegas (ms).
    vegas_base_rtt: f64,
    // ── Westwood-specific ────────────────────────────────────────────────────
    /// Bandwidth estimate (bytes/sec) for Westwood.
    westwood_bw: f64,
    /// Internal PRNG state for jitter.
    prng_state: u64,
}

impl CccConnection {
    fn new(id: CccConnId, config: &CccControllerConfig) -> Self {
        Self {
            id,
            cwnd: config.initial_cwnd,
            ssthresh: config.ssthresh,
            state: CccState::SlowStart,
            rtt_ms: 0.0,
            rtt_var: 0.0,
            in_flight: 0,
            bytes_acked: 0,
            bytes_lost: 0,
            last_ts: 0,
            cubic_k: 0.0,
            cubic_w_max: config.initial_cwnd as f64,
            bbr_bw: 0.0,
            bbr_min_rtt: f64::MAX,
            vegas_base_rtt: 0.0,
            westwood_bw: 0.0,
            prng_state: id.wrapping_add(1).max(1),
        }
    }

    /// Update the EWMA-smoothed RTT.
    fn update_rtt(&mut self, new_rtt_ms: f64, alpha: f64) {
        if self.rtt_ms == 0.0 {
            self.rtt_ms = new_rtt_ms;
            self.rtt_var = new_rtt_ms / 2.0;
        } else {
            let diff = (new_rtt_ms - self.rtt_ms).abs();
            self.rtt_var = (1.0 - 0.25) * self.rtt_var + 0.25 * diff;
            self.rtt_ms = (1.0 - alpha) * self.rtt_ms + alpha * new_rtt_ms;
        }
        if new_rtt_ms < self.bbr_min_rtt {
            self.bbr_min_rtt = new_rtt_ms;
        }
        if self.vegas_base_rtt == 0.0 || new_rtt_ms < self.vegas_base_rtt {
            self.vegas_base_rtt = new_rtt_ms;
        }
    }

    /// Sending rate in bytes/sec given current cwnd and RTT.
    fn rate(&self) -> Option<f64> {
        if self.rtt_ms > 0.0 {
            Some(self.cwnd as f64 / (self.rtt_ms / 1000.0))
        } else {
            None
        }
    }

    /// Apply a small random jitter (±1 %) to the congestion window.
    ///
    /// Used by algorithms that need to break synchronisation among multiple
    /// flows.  The magnitude is bounded so cwnd never leaves `[min, max]`.
    pub fn apply_cwnd_jitter(&mut self, min_cwnd: u64, max_cwnd: u64) {
        let r = xorshift64(&mut self.prng_state);
        // Map to ±1 % of cwnd.
        let pct = (r % 201) as i64 - 100; // -100 … +100
        let delta = (self.cwnd as i64 * pct / 10_000).unsigned_abs();
        if pct >= 0 {
            self.cwnd = self.cwnd.saturating_add(delta).min(max_cwnd);
        } else {
            self.cwnd = self.cwnd.saturating_sub(delta).max(min_cwnd);
        }
    }
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the multi-connection [`CongestionController`].
#[derive(Clone, Debug)]
pub struct CccControllerConfig {
    /// Congestion control algorithm to apply.
    pub algorithm: CccAlgorithm,
    /// Initial congestion window (bytes). Default: 64 KB.
    pub initial_cwnd: u64,
    /// Minimum congestion window (bytes). Default: 1 MTU = 1 448 B.
    pub min_cwnd: u64,
    /// Maximum congestion window (bytes). Default: 16 MB.
    pub max_cwnd: u64,
    /// Initial slow-start threshold (bytes). Default: 1 MB.
    pub ssthresh: u64,
    /// EWMA smoothing factor for RTT. Typical: 0.125.
    pub rtt_alpha: f64,
}

impl Default for CccControllerConfig {
    fn default() -> Self {
        Self {
            algorithm: CccAlgorithm::Reno,
            initial_cwnd: 65_536,
            min_cwnd: 1_448,
            max_cwnd: 16_777_216,
            ssthresh: 1_048_576,
            rtt_alpha: 0.125,
        }
    }
}

// ─── Stats ───────────────────────────────────────────────────────────────────

/// Aggregate statistics for the entire controller.
#[derive(Clone, Debug, Default)]
pub struct CccControllerStats {
    /// Total ACK events processed across all connections.
    pub total_acks: u64,
    /// Total loss events processed across all connections.
    pub total_losses: u64,
    /// Average congestion window across active connections (bytes).
    pub avg_cwnd: f64,
    /// Average smoothed RTT across active connections (ms).
    pub avg_rtt: f64,
    /// Number of active connections.
    pub active_connections: usize,
}

// ─── CongestionController ────────────────────────────────────────────────────

/// Multi-connection, multi-algorithm TCP-inspired congestion controller.
///
/// Manages per-connection congestion state and emits [`Decision`] structs
/// that callers use to pace their sends.
pub struct CongestionController {
    /// Per-connection state.
    connections: HashMap<CccConnId, CccConnection>,
    /// Bounded ring buffer of recent events (max 1 000).
    events: VecDeque<CccEvent>,
    /// Controller-wide configuration.
    config: CccControllerConfig,
    /// Aggregate ACK counter.
    total_acks: u64,
    /// Aggregate loss counter.
    total_losses: u64,
    /// Monotonic internal tick (incremented on every mutating call).
    tick: u64,
}

impl CongestionController {
    /// Create a new controller with the given configuration.
    pub fn new(config: CccControllerConfig) -> Self {
        Self {
            connections: HashMap::new(),
            events: VecDeque::with_capacity(1_000),
            config,
            total_acks: 0,
            total_losses: 0,
            tick: 0,
        }
    }

    /// Create a new controller with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(CccControllerConfig::default())
    }

    // ── Connection management ────────────────────────────────────────────────

    /// Register a new connection.  If the connection already exists this is a no-op.
    pub fn add_connection(&mut self, id: CccConnId) {
        self.connections
            .entry(id)
            .or_insert_with(|| CccConnection::new(id, &self.config));
    }

    /// Remove a tracked connection.  Returns `true` if it existed.
    pub fn remove_connection(&mut self, id: CccConnId) -> bool {
        self.connections.remove(&id).is_some()
    }

    /// Reset a connection to its initial state (keeps the entry).
    ///
    /// Returns an error if the connection does not exist.
    pub fn reset_connection(&mut self, id: CccConnId) -> Result<(), &'static str> {
        let config = &self.config;
        let conn = self
            .connections
            .get_mut(&id)
            .ok_or("connection not found")?;
        *conn = CccConnection::new(id, config);
        Ok(())
    }

    // ── Core operations ──────────────────────────────────────────────────────

    /// Process an ACK for `bytes_acked` bytes on `conn_id` with the measured RTT.
    ///
    /// Returns the new [`Decision`], or an error if `conn_id` is unknown.
    pub fn on_ack(
        &mut self,
        conn_id: CccConnId,
        bytes_acked: u64,
        rtt_ms: f64,
    ) -> Result<CccDecision, &'static str> {
        self.tick = self.tick.wrapping_add(1);
        let ts = self.tick;

        let conn = self
            .connections
            .get_mut(&conn_id)
            .ok_or("connection not found")?;

        let cwnd_before = conn.cwnd;
        conn.update_rtt(rtt_ms, self.config.rtt_alpha);
        conn.bytes_acked = conn.bytes_acked.saturating_add(bytes_acked);
        conn.last_ts = ts;

        // Update bandwidth estimate (Westwood / BBR).
        if rtt_ms > 0.0 {
            let sample_bw = bytes_acked as f64 / (rtt_ms / 1_000.0);
            if conn.westwood_bw == 0.0 {
                conn.westwood_bw = sample_bw;
            } else {
                conn.westwood_bw = 0.875 * conn.westwood_bw + 0.125 * sample_bw;
            }
            if conn.bbr_bw == 0.0 {
                conn.bbr_bw = sample_bw;
            } else {
                conn.bbr_bw = conn.bbr_bw.max(sample_bw);
            }
        }

        let min_cwnd = self.config.min_cwnd;
        let max_cwnd = self.config.max_cwnd;

        let new_cwnd = match self.config.algorithm {
            CccAlgorithm::Reno => reno_on_ack(conn, bytes_acked, min_cwnd, max_cwnd),
            CccAlgorithm::Cubic => cubic_on_ack(conn, bytes_acked, rtt_ms, min_cwnd, max_cwnd),
            CccAlgorithm::Bbr => bbr_on_ack(conn, min_cwnd, max_cwnd),
            CccAlgorithm::Vegas => vegas_on_ack(conn, min_cwnd, max_cwnd),
            CccAlgorithm::Westwood => westwood_on_ack(conn, bytes_acked, min_cwnd, max_cwnd),
        };

        conn.cwnd = new_cwnd.clamp(min_cwnd, max_cwnd);
        self.total_acks += 1;

        // Extract values before releasing the borrow.
        let cwnd_after = conn.cwnd;
        let new_ssthresh = conn.ssthresh;
        let new_state = conn.state;
        let sending_rate = conn.rate();

        self.push_event(CccEvent {
            ts,
            conn_id,
            event_type: CccEventType::AckReceived,
            cwnd_before,
            cwnd_after,
        });

        Ok(Decision {
            new_cwnd: cwnd_after,
            new_ssthresh,
            new_state,
            sending_rate,
        })
    }

    /// Process a packet loss event on `conn_id`.
    ///
    /// Applies multiplicative decrease and enters fast recovery.
    pub fn on_loss(
        &mut self,
        conn_id: CccConnId,
        lost_bytes: u64,
    ) -> Result<CccDecision, &'static str> {
        self.tick = self.tick.wrapping_add(1);
        let ts = self.tick;

        let conn = self
            .connections
            .get_mut(&conn_id)
            .ok_or("connection not found")?;

        let cwnd_before = conn.cwnd;
        conn.bytes_lost = conn.bytes_lost.saturating_add(lost_bytes);
        conn.last_ts = ts;

        let min_cwnd = self.config.min_cwnd;
        let max_cwnd = self.config.max_cwnd;

        let new_cwnd = match self.config.algorithm {
            CccAlgorithm::Reno => reno_on_loss(conn, min_cwnd),
            CccAlgorithm::Cubic => cubic_on_loss(conn, min_cwnd),
            CccAlgorithm::Bbr => bbr_on_loss(conn, min_cwnd),
            CccAlgorithm::Vegas => vegas_on_loss(conn, min_cwnd),
            CccAlgorithm::Westwood => westwood_on_loss(conn, min_cwnd),
        };

        conn.cwnd = new_cwnd.clamp(min_cwnd, max_cwnd);
        conn.state = CccState::FastRecovery;
        self.total_losses += 1;

        // Extract values before releasing the borrow.
        let cwnd_after = conn.cwnd;
        let new_ssthresh = conn.ssthresh;
        let sending_rate = conn.rate();

        self.push_event(CccEvent {
            ts,
            conn_id,
            event_type: CccEventType::PacketLost,
            cwnd_before,
            cwnd_after,
        });

        Ok(Decision {
            new_cwnd: cwnd_after,
            new_ssthresh,
            new_state: CccState::FastRecovery,
            sending_rate,
        })
    }

    /// Process a retransmission timeout on `conn_id`.
    ///
    /// Resets cwnd to min_cwnd, halves ssthresh, re-enters slow start.
    pub fn on_timeout(&mut self, conn_id: CccConnId) -> Result<CccDecision, &'static str> {
        self.tick = self.tick.wrapping_add(1);
        let ts = self.tick;

        let conn = self
            .connections
            .get_mut(&conn_id)
            .ok_or("connection not found")?;

        let cwnd_before = conn.cwnd;
        conn.last_ts = ts;

        let min_cwnd = self.config.min_cwnd;

        conn.ssthresh = (conn.cwnd / 2).max(min_cwnd);
        conn.cwnd = min_cwnd;
        conn.state = CccState::SlowStart;
        // Reset Cubic state on timeout.
        conn.cubic_k = 0.0;
        conn.cubic_w_max = min_cwnd as f64;
        self.total_losses += 1;

        // Extract values before releasing the borrow.
        let cwnd_after = conn.cwnd;
        let new_ssthresh = conn.ssthresh;
        let sending_rate = conn.rate();

        self.push_event(CccEvent {
            ts,
            conn_id,
            event_type: CccEventType::Timeout,
            cwnd_before,
            cwnd_after,
        });
        self.push_event(CccEvent {
            ts,
            conn_id,
            event_type: CccEventType::SlowStartEnter,
            cwnd_before: cwnd_after,
            cwnd_after,
        });

        Ok(Decision {
            new_cwnd: cwnd_after,
            new_ssthresh,
            new_state: CccState::SlowStart,
            sending_rate,
        })
    }

    // ── Queries ──────────────────────────────────────────────────────────────

    /// Current sending rate for a connection, in bytes/sec.
    ///
    /// Returns `None` if the connection is unknown or RTT has not been measured.
    pub fn sending_rate(&self, conn_id: CccConnId) -> Option<f64> {
        self.connections.get(&conn_id).and_then(|c| c.rate())
    }

    /// Aggregate statistics across all active connections.
    pub fn controller_stats(&self) -> CccControllerStats {
        let n = self.connections.len();
        if n == 0 {
            return CccControllerStats {
                total_acks: self.total_acks,
                total_losses: self.total_losses,
                avg_cwnd: 0.0,
                avg_rtt: 0.0,
                active_connections: 0,
            };
        }
        let sum_cwnd: u64 = self.connections.values().map(|c| c.cwnd).sum();
        let sum_rtt: f64 = self.connections.values().map(|c| c.rtt_ms).sum();
        CccControllerStats {
            total_acks: self.total_acks,
            total_losses: self.total_losses,
            avg_cwnd: sum_cwnd as f64 / n as f64,
            avg_rtt: sum_rtt / n as f64,
            active_connections: n,
        }
    }

    /// Immutable reference to a connection, if it exists.
    pub fn connection(&self, conn_id: CccConnId) -> Option<&CccConnection> {
        self.connections.get(&conn_id)
    }

    /// Read-only view of the event log.
    pub fn events(&self) -> &VecDeque<CccEvent> {
        &self.events
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    fn push_event(&mut self, event: CccEvent) {
        if self.events.len() >= 1_000 {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }
}

// ─── Reno ────────────────────────────────────────────────────────────────────

/// Reno on-ACK: AIMD with slow-start below ssthresh.
fn reno_on_ack(conn: &mut CccConnection, bytes_acked: u64, min_cwnd: u64, max_cwnd: u64) -> u64 {
    match conn.state {
        CccState::SlowStart | CccState::Idle => {
            let new_cwnd = conn.cwnd.saturating_add(bytes_acked).min(max_cwnd);
            if new_cwnd >= conn.ssthresh {
                conn.state = CccState::CongestionAvoidance;
            }
            new_cwnd
        }
        CccState::CongestionAvoidance => {
            // Additive increase: +MSS per RTT ≈ MSS² / cwnd per ACK.
            let mss: u64 = 1_448;
            let increase = if conn.cwnd > 0 {
                (mss.saturating_mul(mss)).saturating_div(conn.cwnd).max(1)
            } else {
                mss
            };
            conn.cwnd.saturating_add(increase).min(max_cwnd)
        }
        CccState::FastRecovery => {
            // Deflate window: exit recovery once cwnd climbs back to ssthresh.
            let new_cwnd = conn.cwnd.saturating_add(bytes_acked / 2).min(max_cwnd);
            if new_cwnd >= conn.ssthresh {
                conn.state = CccState::CongestionAvoidance;
            }
            new_cwnd
        }
    }
    .max(min_cwnd)
}

/// Reno on-loss: multiplicative decrease, halve cwnd.
fn reno_on_loss(conn: &mut CccConnection, min_cwnd: u64) -> u64 {
    conn.ssthresh = (conn.cwnd / 2).max(min_cwnd);
    conn.ssthresh
}

// ─── Cubic ───────────────────────────────────────────────────────────────────

/// Cubic growth constant C (standard value 0.4).
const CUBIC_C: f64 = 0.4;
/// Cubic beta (standard 0.7 for CUBIC).
const CUBIC_BETA: f64 = 0.7;

/// Cubic on-ACK: cubic growth function around W_max.
fn cubic_on_ack(
    conn: &mut CccConnection,
    bytes_acked: u64,
    rtt_ms: f64,
    min_cwnd: u64,
    max_cwnd: u64,
) -> u64 {
    match conn.state {
        CccState::SlowStart | CccState::Idle => {
            let new_cwnd = conn.cwnd.saturating_add(bytes_acked).min(max_cwnd);
            if new_cwnd >= conn.ssthresh {
                conn.state = CccState::CongestionAvoidance;
                // Compute K: cubic root of (W_max * (1-beta) / C).
                let w_max = conn.cubic_w_max;
                conn.cubic_k = ((w_max * (1.0 - CUBIC_BETA)) / CUBIC_C).cbrt();
            }
            new_cwnd.max(min_cwnd)
        }
        CccState::CongestionAvoidance => {
            let rtt_s = (rtt_ms / 1_000.0).max(0.001);
            // t = elapsed RTTs since last congestion (approximated by one RTT per call).
            let t = rtt_s;
            let k = conn.cubic_k;
            let w_max = conn.cubic_w_max;
            // W_cubic(t) = C*(t-K)^3 + W_max
            let delta = t - k;
            let w_cubic = CUBIC_C * delta * delta * delta + w_max;
            let target = w_cubic.max(conn.cwnd as f64);
            let new_cwnd = (target as u64).clamp(min_cwnd, max_cwnd);
            if new_cwnd >= conn.ssthresh {
                conn.state = CccState::CongestionAvoidance; // stays
            }
            new_cwnd
        }
        CccState::FastRecovery => {
            let new_cwnd = conn.cwnd.saturating_add(bytes_acked / 2).min(max_cwnd);
            if new_cwnd >= conn.ssthresh {
                conn.state = CccState::CongestionAvoidance;
            }
            new_cwnd.max(min_cwnd)
        }
    }
}

/// Cubic on-loss: W_max = current cwnd, new cwnd = cwnd * beta.
fn cubic_on_loss(conn: &mut CccConnection, min_cwnd: u64) -> u64 {
    conn.cubic_w_max = conn.cwnd as f64;
    let new_cwnd = ((conn.cwnd as f64 * CUBIC_BETA) as u64).max(min_cwnd);
    conn.ssthresh = new_cwnd;
    // Reset cubic K after loss so next CA phase recomputes it.
    conn.cubic_k = 0.0;
    new_cwnd
}

// ─── BBR ─────────────────────────────────────────────────────────────────────

/// BBR on-ACK: pacing window = BDP = bw × min_rtt.
fn bbr_on_ack(conn: &mut CccConnection, min_cwnd: u64, max_cwnd: u64) -> u64 {
    if conn.bbr_bw > 0.0 && conn.bbr_min_rtt < f64::MAX && conn.bbr_min_rtt > 0.0 {
        let bdp = conn.bbr_bw * (conn.bbr_min_rtt / 1_000.0);
        // Add 1.25× gain for probing.
        let target = (bdp * 1.25) as u64;
        // BBR never reduces cwnd on an ACK: the BDP estimate can only raise it.
        let new_cwnd = target.max(conn.cwnd).clamp(min_cwnd, max_cwnd);
        if conn.state == CccState::SlowStart && new_cwnd >= conn.ssthresh {
            conn.state = CccState::CongestionAvoidance;
        }
        new_cwnd
    } else {
        // Fallback to Reno slow-start until we have measurements.
        let new_cwnd = conn.cwnd.saturating_add(1_448).min(max_cwnd);
        if new_cwnd >= conn.ssthresh {
            conn.state = CccState::CongestionAvoidance;
        }
        new_cwnd.max(min_cwnd)
    }
}

/// BBR on-loss: do not reduce cwnd; BBR relies on rate not loss.
/// We do a mild 10% reduction to prevent starvation.
fn bbr_on_loss(conn: &mut CccConnection, min_cwnd: u64) -> u64 {
    conn.ssthresh = (conn.cwnd * 9 / 10).max(min_cwnd);
    conn.ssthresh
}

// ─── Vegas ───────────────────────────────────────────────────────────────────

/// Vegas alpha/beta thresholds (packets).
const VEGAS_ALPHA: f64 = 2.0;
const VEGAS_BETA: f64 = 4.0;

/// Vegas on-ACK: compare actual throughput to expected throughput.
fn vegas_on_ack(conn: &mut CccConnection, min_cwnd: u64, max_cwnd: u64) -> u64 {
    if conn.vegas_base_rtt == 0.0 || conn.rtt_ms == 0.0 {
        // No RTT sample yet — behave like Reno slow-start.
        let new_cwnd = conn.cwnd.saturating_add(1_448).min(max_cwnd);
        if new_cwnd >= conn.ssthresh {
            conn.state = CccState::CongestionAvoidance;
        }
        return new_cwnd.max(min_cwnd);
    }

    let expected = conn.cwnd as f64 / conn.vegas_base_rtt;
    let actual = conn.cwnd as f64 / conn.rtt_ms;
    let diff = expected - actual;
    let mss: f64 = 1_448.0;

    let new_cwnd: u64 = match conn.state {
        CccState::SlowStart | CccState::Idle => {
            let grown = conn.cwnd.saturating_add(1_448).min(max_cwnd);
            if grown >= conn.ssthresh {
                conn.state = CccState::CongestionAvoidance;
            }
            grown
        }
        CccState::CongestionAvoidance => {
            if diff < VEGAS_ALPHA {
                // Too little queuing — increase.
                (conn.cwnd as f64 + mss * mss / conn.cwnd as f64) as u64
            } else if diff > VEGAS_BETA {
                // Too much queuing — decrease.
                conn.cwnd
                    .saturating_sub((mss * mss / conn.cwnd as f64) as u64)
            } else {
                conn.cwnd
            }
        }
        CccState::FastRecovery => {
            let grown = conn.cwnd.saturating_add(1_448 / 2).min(max_cwnd);
            if grown >= conn.ssthresh {
                conn.state = CccState::CongestionAvoidance;
            }
            grown
        }
    };
    new_cwnd.clamp(min_cwnd, max_cwnd)
}

/// Vegas on-loss: same as Reno multiplicative decrease.
fn vegas_on_loss(conn: &mut CccConnection, min_cwnd: u64) -> u64 {
    conn.ssthresh = (conn.cwnd / 2).max(min_cwnd);
    conn.ssthresh
}

// ─── Westwood ────────────────────────────────────────────────────────────────

/// Westwood on-ACK: AIMD but uses bandwidth estimate.
fn westwood_on_ack(
    conn: &mut CccConnection,
    bytes_acked: u64,
    min_cwnd: u64,
    max_cwnd: u64,
) -> u64 {
    match conn.state {
        CccState::SlowStart | CccState::Idle => {
            let new_cwnd = conn.cwnd.saturating_add(bytes_acked).min(max_cwnd);
            if new_cwnd >= conn.ssthresh {
                conn.state = CccState::CongestionAvoidance;
            }
            new_cwnd.max(min_cwnd)
        }
        CccState::CongestionAvoidance => {
            let mss: u64 = 1_448;
            let increase = if conn.cwnd > 0 {
                (mss.saturating_mul(mss)).saturating_div(conn.cwnd).max(1)
            } else {
                mss
            };
            conn.cwnd
                .saturating_add(increase)
                .min(max_cwnd)
                .max(min_cwnd)
        }
        CccState::FastRecovery => {
            let new_cwnd = conn.cwnd.saturating_add(bytes_acked / 2).min(max_cwnd);
            if new_cwnd >= conn.ssthresh {
                conn.state = CccState::CongestionAvoidance;
            }
            new_cwnd.max(min_cwnd)
        }
    }
}

/// Westwood on-loss: set ssthresh = BDP estimate.
fn westwood_on_loss(conn: &mut CccConnection, min_cwnd: u64) -> u64 {
    if conn.westwood_bw > 0.0 && conn.rtt_ms > 0.0 {
        let bdp = (conn.westwood_bw * conn.rtt_ms / 1_000.0) as u64;
        conn.ssthresh = bdp.max(min_cwnd);
    } else {
        conn.ssthresh = (conn.cwnd / 2).max(min_cwnd);
    }
    conn.ssthresh
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEGACY API (retained for backwards compatibility)
// ═══════════════════════════════════════════════════════════════════════════════

/// Phases of the congestion control state machine (legacy).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CongestionState {
    /// Exponential growth phase: window doubles per RTT.
    SlowStart,
    /// Linear growth phase: additive increase.
    CongestionAvoidance,
    /// Recovering from a detected packet loss event.
    FastRecovery,
}

/// Events that drive congestion controller state transitions (legacy).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CongestionEvent {
    /// A successful acknowledgment was received for `bytes` of data.
    AckReceived {
        /// Number of bytes acknowledged.
        bytes: u64,
    },
    /// Packet loss was detected (triggers window reduction).
    PacketLoss,
    /// Retransmit timeout — severe reduction, resets to SlowStart.
    Timeout,
    /// Explicit Congestion Notification mark received.
    EcnMark,
}

/// A snapshot of the current congestion window state for a peer (legacy).
#[derive(Clone, Debug)]
pub struct WindowStats {
    /// Current congestion window size in bytes.
    pub current_window: u64,
    /// Slow-start threshold in bytes.
    pub slow_start_threshold: u64,
    /// Current congestion control phase.
    pub state: CongestionState,
    /// Cumulative number of ACK events processed.
    pub total_acks: u64,
    /// Cumulative number of loss/ECN events processed.
    pub total_losses: u64,
}

impl WindowStats {
    /// Returns the fraction of the total capacity represented by the current window.
    ///
    /// Returns `0.0` when both `current_window` and `slow_start_threshold` are zero.
    pub fn utilization(&self) -> f64 {
        let denom = self.current_window + self.slow_start_threshold;
        if denom == 0 {
            return 0.0;
        }
        self.current_window as f64 / denom as f64
    }
}

/// Configuration parameters for a congestion controller (legacy).
#[derive(Clone, Debug)]
pub struct CongestionConfig {
    /// Initial congestion window size (bytes). Default: 64 KB.
    pub initial_window: u64,
    /// Maximum congestion window size (bytes). Default: 16 MB.
    pub max_window: u64,
    /// Minimum congestion window size (bytes). Default: 1 MTU (1 448 B).
    pub min_window: u64,
    /// Initial slow-start threshold (bytes). Default: 1 MB.
    pub slow_start_threshold: u64,
}

impl Default for CongestionConfig {
    fn default() -> Self {
        Self {
            initial_window: 65_536,
            max_window: 16_777_216,
            min_window: 1_448,
            slow_start_threshold: 1_048_576,
        }
    }
}

/// Per-peer CUBIC-inspired congestion controller (legacy).
pub struct PeerCongestionController {
    /// Identifier of the remote peer.
    pub peer_id: String,
    /// Current congestion window (bytes).
    pub window: u64,
    /// Slow-start threshold (bytes).
    pub ssthresh: u64,
    /// Current state of the congestion control state machine.
    pub state: CongestionState,
    /// Configuration parameters.
    pub config: CongestionConfig,
    /// Total number of ACK events received.
    pub total_acks: u64,
    /// Total number of loss/ECN events received.
    pub total_losses: u64,
}

impl PeerCongestionController {
    /// Create a new controller for `peer_id`, starting in [`CongestionState::SlowStart`].
    pub fn new(peer_id: String, config: CongestionConfig) -> Self {
        let window = config.initial_window;
        let ssthresh = config.slow_start_threshold;
        Self {
            peer_id,
            window,
            ssthresh,
            state: CongestionState::SlowStart,
            config,
            total_acks: 0,
            total_losses: 0,
        }
    }

    /// Process a network event and update window / state accordingly.
    pub fn on_event(&mut self, event: CongestionEvent) {
        match event {
            CongestionEvent::AckReceived { bytes } => self.handle_ack(bytes),
            CongestionEvent::PacketLoss => self.handle_packet_loss(),
            CongestionEvent::Timeout => self.handle_timeout(),
            CongestionEvent::EcnMark => self.handle_ecn(),
        }
    }

    fn handle_ack(&mut self, bytes: u64) {
        match self.state {
            CongestionState::SlowStart => {
                self.window = self
                    .window
                    .saturating_add(bytes)
                    .min(self.config.max_window);
                if self.window >= self.ssthresh {
                    self.state = CongestionState::CongestionAvoidance;
                }
                self.total_acks += 1;
            }
            CongestionState::CongestionAvoidance => {
                let increase = if self.window > 0 {
                    (bytes.saturating_mul(bytes)).saturating_div(self.window)
                } else {
                    bytes
                };
                self.window = self
                    .window
                    .saturating_add(increase)
                    .min(self.config.max_window);
                self.total_acks += 1;
            }
            CongestionState::FastRecovery => {
                self.window = self
                    .window
                    .saturating_add(bytes / 2)
                    .min(self.config.max_window);
                if self.window >= self.ssthresh {
                    self.state = CongestionState::CongestionAvoidance;
                }
                self.total_acks += 1;
            }
        }
    }

    fn handle_packet_loss(&mut self) {
        self.ssthresh = (self.window / 2).max(self.config.min_window);
        self.window = self.ssthresh;
        self.state = CongestionState::FastRecovery;
        self.total_losses += 1;
    }

    fn handle_timeout(&mut self) {
        self.ssthresh = (self.window / 2).max(self.config.min_window);
        self.window = self.config.initial_window.max(self.config.min_window);
        self.state = CongestionState::SlowStart;
        self.total_losses += 1;
    }

    fn handle_ecn(&mut self) {
        self.ssthresh = ((self.window * 7) / 8).max(self.config.min_window);
        self.window = self.ssthresh;
        self.state = CongestionState::CongestionAvoidance;
        self.total_losses += 1;
    }

    /// Return a snapshot of the current window statistics.
    pub fn window_stats(&self) -> WindowStats {
        WindowStats {
            current_window: self.window,
            slow_start_threshold: self.ssthresh,
            state: self.state,
            total_acks: self.total_acks,
            total_losses: self.total_losses,
        }
    }

    /// Returns `true` if the caller may send `bytes` given the current window.
    pub fn can_send(&self, bytes: u64) -> bool {
        bytes <= self.window
    }
}

/// Manages [`PeerCongestionController`] instances for multiple peers (legacy).
pub struct MultiPeerCongestionManager {
    /// Per-peer controllers, keyed by peer ID string.
    pub controllers: HashMap<String, PeerCongestionController>,
    /// Default configuration applied to newly created controllers.
    pub config: CongestionConfig,
}

impl MultiPeerCongestionManager {
    /// Create a new manager with the given default configuration.
    pub fn new(config: CongestionConfig) -> Self {
        Self {
            controllers: HashMap::new(),
            config,
        }
    }

    /// Return a mutable reference to the controller for `peer_id`.
    pub fn get_or_create(&mut self, peer_id: &str) -> &mut PeerCongestionController {
        self.controllers
            .entry(peer_id.to_owned())
            .or_insert_with(|| {
                PeerCongestionController::new(peer_id.to_owned(), self.config.clone())
            })
    }

    /// Deliver a congestion event to the controller for `peer_id`.
    pub fn on_event(&mut self, peer_id: &str, event: CongestionEvent) {
        let ctrl = self.get_or_create(peer_id);
        ctrl.on_event(event);
    }

    /// Remove the controller for `peer_id`.
    pub fn remove_peer(&mut self, peer_id: &str) -> bool {
        self.controllers.remove(peer_id).is_some()
    }

    /// Return the sum of all managed peer windows.
    pub fn total_window(&self) -> u64 {
        self.controllers.values().map(|c| c.window).sum()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn default_config() -> CongestionConfig {
        CongestionConfig::default()
    }

    fn legacy_ctrl(peer: &str) -> PeerCongestionController {
        PeerCongestionController::new(peer.to_owned(), default_config())
    }

    fn make_ctrl(algo: CccAlgorithm) -> CongestionController {
        CongestionController::new(CccControllerConfig {
            algorithm: algo,
            ..CccControllerConfig::default()
        })
    }

    // ══ LEGACY TESTS (1–24) ══════════════════════════════════════════════════

    #[test]
    fn test_new_starts_in_slow_start() {
        let ctrl = legacy_ctrl("peer-a");
        assert_eq!(ctrl.state, CongestionState::SlowStart);
        assert_eq!(ctrl.window, 65_536);
        assert_eq!(ctrl.ssthresh, 1_048_576);
    }

    #[test]
    fn test_slow_start_ack_grows_window() {
        let mut ctrl = legacy_ctrl("p");
        let before = ctrl.window;
        ctrl.on_event(CongestionEvent::AckReceived { bytes: 1_000 });
        assert_eq!(ctrl.window, before + 1_000);
    }

    #[test]
    fn test_slow_start_caps_at_max() {
        let cfg = CongestionConfig {
            initial_window: 16_777_000,
            ..Default::default()
        };
        let mut ctrl = PeerCongestionController::new("p".into(), cfg.clone());
        ctrl.on_event(CongestionEvent::AckReceived { bytes: 1_000_000 });
        assert_eq!(ctrl.window, cfg.max_window);
    }

    #[test]
    fn test_slow_start_transitions_at_ssthresh() {
        let cfg = CongestionConfig {
            initial_window: 500_000,
            slow_start_threshold: 600_000,
            ..Default::default()
        };
        let mut ctrl = PeerCongestionController::new("p".into(), cfg);
        ctrl.on_event(CongestionEvent::AckReceived { bytes: 200_000 });
        assert_eq!(ctrl.state, CongestionState::CongestionAvoidance);
    }

    #[test]
    fn test_congestion_avoidance_increase_smaller() {
        let cfg = CongestionConfig {
            initial_window: 100_000,
            slow_start_threshold: 50_000,
            ..Default::default()
        };
        let mut ctrl = PeerCongestionController::new("p".into(), cfg);
        ctrl.state = CongestionState::CongestionAvoidance;
        let bytes: u64 = 1_000;
        let before = ctrl.window;
        ctrl.on_event(CongestionEvent::AckReceived { bytes });
        let ca_increase = ctrl.window - before;
        assert!(ca_increase < bytes);
        assert_eq!(ca_increase, (bytes * bytes) / 100_000);
    }

    #[test]
    fn test_fast_recovery_ack_grows_half() {
        let mut ctrl = legacy_ctrl("p");
        ctrl.state = CongestionState::FastRecovery;
        ctrl.ssthresh = ctrl.window + 100_000;
        let before = ctrl.window;
        ctrl.on_event(CongestionEvent::AckReceived { bytes: 2_000 });
        assert_eq!(ctrl.window, before + 1_000);
    }

    #[test]
    fn test_fast_recovery_transitions_to_ca() {
        let cfg = CongestionConfig {
            initial_window: 50_000,
            slow_start_threshold: 60_000,
            ..Default::default()
        };
        let mut ctrl = PeerCongestionController::new("p".into(), cfg);
        ctrl.state = CongestionState::FastRecovery;
        ctrl.ssthresh = 51_000;
        ctrl.on_event(CongestionEvent::AckReceived { bytes: 10_000 });
        assert_eq!(ctrl.state, CongestionState::CongestionAvoidance);
    }

    #[test]
    fn test_packet_loss_sets_ssthresh_and_state() {
        let mut ctrl = legacy_ctrl("p");
        let orig_window = ctrl.window;
        ctrl.on_event(CongestionEvent::PacketLoss);
        assert_eq!(ctrl.ssthresh, orig_window / 2);
        assert_eq!(ctrl.window, orig_window / 2);
        assert_eq!(ctrl.state, CongestionState::FastRecovery);
    }

    #[test]
    fn test_packet_loss_floors_at_min() {
        let cfg = CongestionConfig {
            initial_window: 2_000,
            min_window: 1_448,
            ..Default::default()
        };
        let mut ctrl = PeerCongestionController::new("p".into(), cfg.clone());
        ctrl.on_event(CongestionEvent::PacketLoss);
        assert_eq!(ctrl.window, cfg.min_window);
        assert_eq!(ctrl.ssthresh, cfg.min_window);
    }

    #[test]
    fn test_timeout_resets_window_and_state() {
        let mut ctrl = legacy_ctrl("p");
        ctrl.window = 500_000;
        ctrl.on_event(CongestionEvent::Timeout);
        assert_eq!(ctrl.window, 65_536);
        assert_eq!(ctrl.state, CongestionState::SlowStart);
    }

    #[test]
    fn test_ecn_mark() {
        let mut ctrl = legacy_ctrl("p");
        ctrl.window = 800_000;
        ctrl.on_event(CongestionEvent::EcnMark);
        assert_eq!(ctrl.ssthresh, (800_000u64 * 7) / 8);
        assert_eq!(ctrl.window, ctrl.ssthresh);
        assert_eq!(ctrl.state, CongestionState::CongestionAvoidance);
    }

    #[test]
    fn test_total_acks_increments() {
        let mut ctrl = legacy_ctrl("p");
        assert_eq!(ctrl.total_acks, 0);
        ctrl.on_event(CongestionEvent::AckReceived { bytes: 100 });
        ctrl.on_event(CongestionEvent::AckReceived { bytes: 100 });
        assert_eq!(ctrl.total_acks, 2);
    }

    #[test]
    fn test_total_losses_increments() {
        let mut ctrl = legacy_ctrl("p");
        assert_eq!(ctrl.total_losses, 0);
        ctrl.on_event(CongestionEvent::PacketLoss);
        ctrl.on_event(CongestionEvent::EcnMark);
        ctrl.on_event(CongestionEvent::Timeout);
        assert_eq!(ctrl.total_losses, 3);
    }

    #[test]
    fn test_can_send_within_window() {
        let ctrl = legacy_ctrl("p");
        assert!(ctrl.can_send(ctrl.window));
        assert!(ctrl.can_send(1));
    }

    #[test]
    fn test_can_send_exceeds_window() {
        let ctrl = legacy_ctrl("p");
        assert!(!ctrl.can_send(ctrl.window + 1));
    }

    #[test]
    fn test_utilization() {
        let ctrl = legacy_ctrl("p");
        let stats = ctrl.window_stats();
        let expected = stats.current_window as f64
            / (stats.current_window + stats.slow_start_threshold) as f64;
        let diff = (stats.utilization() - expected).abs();
        assert!(diff < 1e-12);
    }

    #[test]
    fn test_utilization_zero() {
        let stats = WindowStats {
            current_window: 0,
            slow_start_threshold: 0,
            state: CongestionState::SlowStart,
            total_acks: 0,
            total_losses: 0,
        };
        assert_eq!(stats.utilization(), 0.0);
    }

    #[test]
    fn test_manager_creates_on_first_access() {
        let mut mgr = MultiPeerCongestionManager::new(default_config());
        let ctrl = mgr.get_or_create("peer-1");
        assert_eq!(ctrl.state, CongestionState::SlowStart);
        assert_eq!(ctrl.window, 65_536);
    }

    #[test]
    fn test_manager_routes_event() {
        let mut mgr = MultiPeerCongestionManager::new(default_config());
        mgr.get_or_create("a");
        mgr.get_or_create("b");
        let initial_a = mgr.controllers["a"].window;
        let initial_b = mgr.controllers["b"].window;
        mgr.on_event("a", CongestionEvent::AckReceived { bytes: 5_000 });
        assert_eq!(mgr.controllers["a"].window, initial_a + 5_000);
        assert_eq!(mgr.controllers["b"].window, initial_b);
    }

    #[test]
    fn test_remove_peer() {
        let mut mgr = MultiPeerCongestionManager::new(default_config());
        mgr.get_or_create("x");
        assert!(mgr.remove_peer("x"));
        assert!(!mgr.remove_peer("x"));
    }

    #[test]
    fn test_total_window() {
        let mut mgr = MultiPeerCongestionManager::new(default_config());
        mgr.get_or_create("a");
        mgr.get_or_create("b");
        let expected = mgr.controllers["a"].window + mgr.controllers["b"].window;
        assert_eq!(mgr.total_window(), expected);
    }

    #[test]
    fn test_ssthresh_floor_on_loss() {
        let cfg = CongestionConfig {
            initial_window: 1_500,
            min_window: 1_448,
            ..Default::default()
        };
        let mut ctrl = PeerCongestionController::new("p".into(), cfg.clone());
        ctrl.on_event(CongestionEvent::PacketLoss);
        assert!(ctrl.ssthresh >= cfg.min_window);
        assert!(ctrl.window >= cfg.min_window);
    }

    #[test]
    fn test_state_machine_full_cycle() {
        let mut ctrl = legacy_ctrl("p");
        assert_eq!(ctrl.state, CongestionState::SlowStart);
        for _ in 0..20 {
            ctrl.on_event(CongestionEvent::AckReceived { bytes: 100_000 });
        }
        assert_eq!(ctrl.state, CongestionState::CongestionAvoidance);
        ctrl.on_event(CongestionEvent::PacketLoss);
        assert_eq!(ctrl.state, CongestionState::FastRecovery);
        for _ in 0..30 {
            ctrl.on_event(CongestionEvent::AckReceived { bytes: 100_000 });
        }
        assert_eq!(ctrl.state, CongestionState::CongestionAvoidance);
        ctrl.on_event(CongestionEvent::Timeout);
        assert_eq!(ctrl.state, CongestionState::SlowStart);
    }

    #[test]
    fn test_ca_aimd_formula() {
        let cfg = CongestionConfig {
            initial_window: 200_000,
            slow_start_threshold: 100_000,
            ..Default::default()
        };
        let mut ctrl = PeerCongestionController::new("p".into(), cfg);
        ctrl.state = CongestionState::CongestionAvoidance;
        let window_before = ctrl.window;
        let bytes: u64 = 4_000;
        ctrl.on_event(CongestionEvent::AckReceived { bytes });
        let expected_increase = (bytes * bytes) / window_before;
        assert_eq!(ctrl.window, window_before + expected_increase);
    }

    // ══ NEW API TESTS (25+) ══════════════════════════════════════════════════

    // ── add / remove / reset connection ─────────────────────────────────────

    #[test]
    fn test_add_connection_creates_entry() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        assert!(cc.connection(1).is_some());
    }

    #[test]
    fn test_add_connection_idempotent() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        let cwnd1 = cc.connection(1).map(|c| c.cwnd);
        cc.add_connection(1); // should not reset
        let cwnd2 = cc.connection(1).map(|c| c.cwnd);
        assert_eq!(cwnd1, cwnd2);
    }

    #[test]
    fn test_remove_connection_returns_true() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(10);
        assert!(cc.remove_connection(10));
    }

    #[test]
    fn test_remove_connection_missing_returns_false() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        assert!(!cc.remove_connection(99));
    }

    #[test]
    fn test_reset_connection_restores_defaults() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(5);
        // Drive cwnd up.
        cc.on_ack(5, 500_000, 10.0).expect("ack ok");
        cc.reset_connection(5).expect("reset ok");
        let conn = cc.connection(5).expect("still exists");
        assert_eq!(conn.cwnd, 65_536);
        assert_eq!(conn.state, CccState::SlowStart);
    }

    #[test]
    fn test_reset_connection_unknown_errors() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        assert!(cc.reset_connection(999).is_err());
    }

    // ── on_ack ───────────────────────────────────────────────────────────────

    #[test]
    fn test_on_ack_unknown_errors() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        assert!(cc.on_ack(42, 1_000, 10.0).is_err());
    }

    #[test]
    fn test_reno_slow_start_ack_increases_cwnd() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        let before = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        cc.on_ack(1, 1_000, 20.0).expect("ok");
        let after = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        assert!(after > before);
    }

    #[test]
    fn test_reno_transitions_to_ca_after_ssthresh() {
        let mut cc = CongestionController::new(CccControllerConfig {
            algorithm: CccAlgorithm::Reno,
            initial_cwnd: 900_000,
            ssthresh: 1_000_000,
            ..Default::default()
        });
        cc.add_connection(1);
        cc.on_ack(1, 200_000, 10.0).expect("ok");
        let state = cc
            .connection(1)
            .map(|c| c.state)
            .expect("test: connection 1 should exist after add_connection");
        assert_eq!(state, CccState::CongestionAvoidance);
    }

    #[test]
    fn test_reno_ca_smaller_increase_than_ss() {
        let mut cc = CongestionController::new(CccControllerConfig {
            algorithm: CccAlgorithm::Reno,
            initial_cwnd: 200_000,
            ssthresh: 100_000,
            ..Default::default()
        });
        cc.add_connection(1);
        // Force into CA.
        if let Some(c) = cc.connections.get_mut(&1) {
            c.state = CccState::CongestionAvoidance;
        }
        let before = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        cc.on_ack(1, 5_000, 10.0).expect("ok");
        let after = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        // In CA the increase should be much smaller than 5_000.
        let delta = after - before;
        assert!(delta < 5_000, "delta={delta}");
    }

    #[test]
    fn test_reno_cwnd_capped_at_max() {
        let mut cc = CongestionController::new(CccControllerConfig {
            algorithm: CccAlgorithm::Reno,
            initial_cwnd: 16_776_000,
            max_cwnd: 16_777_216,
            ..Default::default()
        });
        cc.add_connection(1);
        cc.on_ack(1, 500_000, 10.0).expect("ok");
        let cwnd = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        assert!(cwnd <= 16_777_216);
    }

    // ── on_loss ──────────────────────────────────────────────────────────────

    #[test]
    fn test_on_loss_unknown_errors() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        assert!(cc.on_loss(42, 1_000).is_err());
    }

    #[test]
    fn test_reno_on_loss_halves_cwnd() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        let before = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        cc.on_loss(1, 1_000).expect("ok");
        let after = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        assert!(after <= before / 2 + 1); // allow for min_cwnd floor
    }

    #[test]
    fn test_on_loss_enters_fast_recovery() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        cc.on_loss(1, 100).expect("ok");
        assert_eq!(
            cc.connection(1).map(|c| c.state),
            Some(CccState::FastRecovery)
        );
    }

    #[test]
    fn test_on_loss_increments_total_losses() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        cc.on_loss(1, 100).expect("ok");
        cc.on_loss(1, 100).expect("ok");
        assert_eq!(cc.controller_stats().total_losses, 2);
    }

    #[test]
    fn test_on_loss_cwnd_never_below_min() {
        let mut cc = CongestionController::new(CccControllerConfig {
            algorithm: CccAlgorithm::Reno,
            initial_cwnd: 1_448,
            min_cwnd: 1_448,
            ..Default::default()
        });
        cc.add_connection(1);
        cc.on_loss(1, 100).expect("ok");
        let cwnd = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        assert!(cwnd >= 1_448);
    }

    // ── on_timeout ───────────────────────────────────────────────────────────

    #[test]
    fn test_on_timeout_unknown_errors() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        assert!(cc.on_timeout(42).is_err());
    }

    #[test]
    fn test_on_timeout_resets_to_slow_start() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        cc.on_ack(1, 500_000, 10.0).expect("ok");
        cc.on_timeout(1).expect("ok");
        assert_eq!(cc.connection(1).map(|c| c.state), Some(CccState::SlowStart));
    }

    #[test]
    fn test_on_timeout_resets_cwnd_to_min() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        cc.on_timeout(1).expect("ok");
        let cwnd = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        assert_eq!(cwnd, 1_448); // min_cwnd
    }

    #[test]
    fn test_on_timeout_halves_ssthresh() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        let before_ssthresh = cc.connection(1).map(|c| c.ssthresh).unwrap_or(0);
        let before_cwnd = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        cc.on_timeout(1).expect("ok");
        let after_ssthresh = cc.connection(1).map(|c| c.ssthresh).unwrap_or(0);
        assert!(after_ssthresh <= (before_cwnd / 2).max(1_448));
        // ssthresh must be <= what it was (initial cwnd is < initial ssthresh).
        assert!(after_ssthresh <= before_ssthresh);
    }

    #[test]
    fn test_on_timeout_increments_total_losses() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        cc.on_timeout(1).expect("ok");
        assert_eq!(cc.controller_stats().total_losses, 1);
    }

    // ── sending_rate ─────────────────────────────────────────────────────────

    #[test]
    fn test_sending_rate_none_before_rtt() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        assert_eq!(cc.sending_rate(1), None);
    }

    #[test]
    fn test_sending_rate_some_after_ack() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        cc.on_ack(1, 1_000, 20.0).expect("ok");
        assert!(cc.sending_rate(1).is_some());
    }

    #[test]
    fn test_sending_rate_unknown_none() {
        let cc = make_ctrl(CccAlgorithm::Reno);
        assert_eq!(cc.sending_rate(999), None);
    }

    #[test]
    fn test_sending_rate_positive() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        cc.on_ack(1, 10_000, 100.0).expect("ok");
        let rate = cc.sending_rate(1).unwrap_or(0.0);
        assert!(rate > 0.0);
    }

    // ── controller_stats ─────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty_controller() {
        let cc = make_ctrl(CccAlgorithm::Reno);
        let stats = cc.controller_stats();
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.total_acks, 0);
        assert_eq!(stats.total_losses, 0);
        assert_eq!(stats.avg_cwnd, 0.0);
        assert_eq!(stats.avg_rtt, 0.0);
    }

    #[test]
    fn test_stats_counts_connections() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        cc.add_connection(2);
        assert_eq!(cc.controller_stats().active_connections, 2);
    }

    #[test]
    fn test_stats_total_acks_aggregated() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        cc.add_connection(2);
        cc.on_ack(1, 1_000, 10.0).expect("ok");
        cc.on_ack(2, 1_000, 10.0).expect("ok");
        assert_eq!(cc.controller_stats().total_acks, 2);
    }

    #[test]
    fn test_stats_avg_cwnd_reasonable() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        let stats = cc.controller_stats();
        assert!(stats.avg_cwnd > 0.0);
    }

    // ── event log ────────────────────────────────────────────────────────────

    #[test]
    fn test_events_populated_on_ack() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        cc.on_ack(1, 1_000, 10.0).expect("ok");
        assert!(!cc.events().is_empty());
    }

    #[test]
    fn test_events_populated_on_loss() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        cc.on_loss(1, 100).expect("ok");
        let has_loss = cc
            .events()
            .iter()
            .any(|e| e.event_type == CccEventType::PacketLost);
        assert!(has_loss);
    }

    #[test]
    fn test_events_bounded_at_1000() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        for _ in 0..1_100u32 {
            cc.on_ack(1, 100, 5.0).expect("ok");
        }
        assert!(cc.events().len() <= 1_000);
    }

    #[test]
    fn test_events_timeout_type() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        cc.on_timeout(1).expect("ok");
        let has_timeout = cc
            .events()
            .iter()
            .any(|e| e.event_type == CccEventType::Timeout);
        assert!(has_timeout);
    }

    #[test]
    fn test_events_record_cwnd_change() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        cc.on_ack(1, 10_000, 10.0).expect("ok");
        let event = cc.events().iter().last().expect("event");
        // cwnd_before and cwnd_after are both valid u64 values.
        assert!(event.cwnd_after > 0);
    }

    // ── Cubic algorithm ──────────────────────────────────────────────────────

    #[test]
    fn test_cubic_slow_start_grows() {
        let mut cc = make_ctrl(CccAlgorithm::Cubic);
        cc.add_connection(1);
        let before = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        cc.on_ack(1, 10_000, 20.0).expect("ok");
        let after = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        assert!(after > before);
    }

    #[test]
    fn test_cubic_loss_reduces_cwnd() {
        let mut cc = make_ctrl(CccAlgorithm::Cubic);
        cc.add_connection(1);
        let before = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        cc.on_loss(1, 1_000).expect("ok");
        let after = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        assert!(after <= before);
    }

    #[test]
    fn test_cubic_timeout_resets_to_slowstart() {
        let mut cc = make_ctrl(CccAlgorithm::Cubic);
        cc.add_connection(1);
        cc.on_timeout(1).expect("ok");
        assert_eq!(cc.connection(1).map(|c| c.state), Some(CccState::SlowStart));
    }

    // ── BBR algorithm ────────────────────────────────────────────────────────

    #[test]
    fn test_bbr_grows_cwnd() {
        let mut cc = make_ctrl(CccAlgorithm::Bbr);
        cc.add_connection(1);
        let before = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        cc.on_ack(1, 10_000, 20.0).expect("ok");
        let after = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        assert!(after >= before);
    }

    #[test]
    fn test_bbr_loss_mild_reduction() {
        let mut cc = make_ctrl(CccAlgorithm::Bbr);
        cc.add_connection(1);
        let before = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        cc.on_loss(1, 1_000).expect("ok");
        let after = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        // BBR does a mild 10% reduction, so after should be > 0 and <= before.
        assert!(after > 0 && after <= before);
    }

    #[test]
    fn test_bbr_has_sending_rate_after_ack() {
        let mut cc = make_ctrl(CccAlgorithm::Bbr);
        cc.add_connection(1);
        cc.on_ack(1, 50_000, 15.0).expect("ok");
        assert!(cc.sending_rate(1).is_some());
    }

    // ── Vegas algorithm ──────────────────────────────────────────────────────

    #[test]
    fn test_vegas_grows_cwnd_without_rtt() {
        let mut cc = make_ctrl(CccAlgorithm::Vegas);
        cc.add_connection(1);
        let before = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        cc.on_ack(1, 1_448, 0.0).expect("ok");
        let after = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        assert!(after >= before);
    }

    #[test]
    fn test_vegas_loss_reduces_cwnd() {
        let mut cc = make_ctrl(CccAlgorithm::Vegas);
        cc.add_connection(1);
        let before = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        cc.on_loss(1, 1_000).expect("ok");
        let after = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        assert!(after <= before);
    }

    // ── Westwood algorithm ───────────────────────────────────────────────────

    #[test]
    fn test_westwood_slow_start_grows() {
        let mut cc = make_ctrl(CccAlgorithm::Westwood);
        cc.add_connection(1);
        let before = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        cc.on_ack(1, 5_000, 10.0).expect("ok");
        let after = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        assert!(after > before);
    }

    #[test]
    fn test_westwood_loss_uses_bw_estimate() {
        let mut cc = make_ctrl(CccAlgorithm::Westwood);
        cc.add_connection(1);
        // Give it some bandwidth history.
        cc.on_ack(1, 50_000, 20.0).expect("ok");
        cc.on_loss(1, 1_000).expect("ok");
        // After loss with a BW estimate, ssthresh should be the BDP.
        let ssthresh = cc.connection(1).map(|c| c.ssthresh).unwrap_or(0);
        assert!(ssthresh >= 1_448);
    }

    // ── xorshift64 ────────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_produces_nonzero() {
        let mut state: u64 = 12345;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }

    #[test]
    fn test_xorshift64_changes_state() {
        let mut state: u64 = 9999;
        let v1 = xorshift64(&mut state);
        let v2 = xorshift64(&mut state);
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_xorshift64_deterministic() {
        let mut s1: u64 = 42;
        let mut s2: u64 = 42;
        assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
    }

    // ── Decision struct ──────────────────────────────────────────────────────

    #[test]
    fn test_decision_new_cwnd_matches_connection() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        let d = cc.on_ack(1, 1_000, 10.0).expect("ok");
        let actual = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        assert_eq!(d.new_cwnd, actual);
    }

    #[test]
    fn test_decision_state_matches_connection() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        let d = cc.on_loss(1, 100).expect("ok");
        assert_eq!(d.new_state, CccState::FastRecovery);
    }

    #[test]
    fn test_decision_sending_rate_consistency() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        let d = cc.on_ack(1, 1_000, 50.0).expect("ok");
        // Decision sending_rate should match controller's sending_rate.
        assert_eq!(d.sending_rate, cc.sending_rate(1));
    }

    // ── RTT tracking ─────────────────────────────────────────────────────────

    #[test]
    fn test_rtt_updated_on_ack() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        cc.on_ack(1, 1_000, 30.0).expect("ok");
        let rtt = cc.connection(1).map(|c| c.rtt_ms).unwrap_or(0.0);
        assert!(rtt > 0.0);
    }

    #[test]
    fn test_rtt_ewma_converges() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        for _ in 0..20 {
            cc.on_ack(1, 1_000, 100.0).expect("ok");
        }
        let rtt = cc.connection(1).map(|c| c.rtt_ms).unwrap_or(0.0);
        // After 20 samples of 100ms, EWMA should be close to 100ms.
        assert!((rtt - 100.0).abs() < 20.0, "rtt={rtt}");
    }

    // ── bytes_acked / bytes_lost accounting ───────────────────────────────────

    #[test]
    fn test_bytes_acked_accumulates() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        cc.on_ack(1, 5_000, 10.0).expect("ok");
        cc.on_ack(1, 3_000, 10.0).expect("ok");
        let ba = cc.connection(1).map(|c| c.bytes_acked).unwrap_or(0);
        assert_eq!(ba, 8_000);
    }

    #[test]
    fn test_bytes_lost_accumulates() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        cc.on_loss(1, 1_000).expect("ok");
        cc.on_loss(1, 500).expect("ok");
        let bl = cc.connection(1).map(|c| c.bytes_lost).unwrap_or(0);
        assert_eq!(bl, 1_500);
    }

    // ── min_cwnd floor guarantee ──────────────────────────────────────────────

    #[test]
    fn test_cwnd_never_below_min_after_many_losses() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        for _ in 0..50 {
            cc.on_loss(1, 100_000).expect("ok");
        }
        let cwnd = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        assert!(cwnd >= 1_448);
    }

    // ── max_cwnd ceiling guarantee ────────────────────────────────────────────

    #[test]
    fn test_cwnd_never_above_max_after_many_acks() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        for _ in 0..5_000u32 {
            cc.on_ack(1, 100_000, 5.0).expect("ok");
        }
        let cwnd = cc.connection(1).map(|c| c.cwnd).unwrap_or(0);
        assert!(cwnd <= 16_777_216);
    }

    // ── multi-connection isolation ────────────────────────────────────────────

    #[test]
    fn test_connections_isolated() {
        let mut cc = make_ctrl(CccAlgorithm::Reno);
        cc.add_connection(1);
        cc.add_connection(2);
        let before_2 = cc.connection(2).map(|c| c.cwnd).unwrap_or(0);
        cc.on_loss(1, 10_000).expect("ok");
        let after_2 = cc.connection(2).map(|c| c.cwnd).unwrap_or(0);
        assert_eq!(before_2, after_2);
    }

    // ── CccAlgorithm default ──────────────────────────────────────────────────

    #[test]
    fn test_algorithm_default_is_reno() {
        assert_eq!(CccAlgorithm::default(), CccAlgorithm::Reno);
    }

    // ── CccState default ─────────────────────────────────────────────────────

    #[test]
    fn test_state_default_is_ca() {
        assert_eq!(CccState::default(), CccState::CongestionAvoidance);
    }

    // ── type alias usability ──────────────────────────────────────────────────

    #[test]
    fn test_type_aliases_usable() {
        let mut cc: CccCongestionController = CccCongestionController::with_defaults();
        cc.add_connection(1);
        let d: CccDecision = cc.on_ack(1, 1_000, 10.0).expect("ok");
        assert!(d.new_cwnd > 0);
    }
}
