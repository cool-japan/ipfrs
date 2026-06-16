//! WebRTC peer connection state machine.
//!
//! # Overview
//!
//! [`PeerStateMachine`] models the lifecycle of a single WebRTC peer connection
//! as a finite state machine with the following states:
//!
//! ```text
//!                      start_connecting(now_ms)
//!   New ──────────────────────────────────────► Connecting { started_at_ms }
//!                                                     │
//!                             mark_connected(peer_id, now_ms)
//!                                                     ▼
//!                              Connected { peer_id, established_at_ms }
//!                                                     │
//!                            mark_disconnected(reason, now_ms)
//!                                                     ▼
//!                             Disconnected { peer_id, reason }
//!
//!   (any state) ──── mark_failed(error) ──────► Failed { error }
//!   (any state) ──── reset() ─────────────────► New
//! ```
//!
//! Every transition is logged in a `transition_log` with a caller-supplied
//! millisecond timestamp so that the state history can be inspected for
//! debugging or telemetry purposes.
//!
//! # Design notes
//!
//! - All transition methods return `Result<(), String>`.  Invalid transitions
//!   (e.g. calling `mark_connected` while already in the `Connected` state)
//!   return `Err` with a descriptive message; they do **not** panic.
//! - Timestamps are supplied by the caller (`now_ms: u64`) rather than read
//!   from the system clock so that unit tests remain deterministic and the
//!   code stays `no_std`-compatible.
//! - The `transition_log` grows without bound; callers that need bounded memory
//!   should call `reset()` periodically or replace the machine entirely.

// ---------------------------------------------------------------------------
// PeerConnectionState
// ---------------------------------------------------------------------------

/// The current state of a WebRTC peer connection.
#[derive(Debug, Clone, PartialEq)]
pub enum PeerConnectionState {
    /// Initial state — no connection attempt has been made yet.
    New,
    /// A connection attempt is in progress (offer/answer exchange, ICE gather).
    Connecting {
        /// Wall-clock time (ms since epoch) when `start_connecting` was called.
        started_at_ms: u64,
    },
    /// The data channel is open and the connection is usable.
    Connected {
        /// Application-level identifier of the remote peer.
        peer_id: String,
        /// Wall-clock time (ms since epoch) when the connection was established.
        established_at_ms: u64,
    },
    /// The previously connected peer has disconnected.
    Disconnected {
        /// Application-level identifier of the peer that disconnected.
        peer_id: String,
        /// Human-readable reason for the disconnection.
        reason: String,
    },
    /// An unrecoverable error has occurred.
    Failed {
        /// Description of the error.
        error: String,
    },
}

// ---------------------------------------------------------------------------
// PeerStateMachine
// ---------------------------------------------------------------------------

/// Finite state machine that tracks the lifecycle of a single WebRTC peer
/// connection.
///
/// Construct with [`PeerStateMachine::new`], then drive with the transition
/// methods.  All invalid transitions return `Err` rather than panicking.
pub struct PeerStateMachine {
    state: PeerConnectionState,
    /// Append-only log of `(timestamp_ms, new_state)` pairs, one per transition.
    transition_log: Vec<(u64, PeerConnectionState)>,
}

impl PeerStateMachine {
    /// Create a new state machine in the [`PeerConnectionState::New`] state.
    pub fn new() -> Self {
        Self {
            state: PeerConnectionState::New,
            transition_log: Vec::new(),
        }
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Return a reference to the current state.
    pub fn state(&self) -> &PeerConnectionState {
        &self.state
    }

    /// Return `true` iff the current state is [`PeerConnectionState::Connected`].
    pub fn is_connected(&self) -> bool {
        matches!(&self.state, PeerConnectionState::Connected { .. })
    }

    /// Return the remote peer's identifier when in the `Connected` state,
    /// or `None` otherwise.
    pub fn peer_id(&self) -> Option<&str> {
        match &self.state {
            PeerConnectionState::Connected { peer_id, .. } => Some(peer_id.as_str()),
            _ => None,
        }
    }

    /// Return the total number of state transitions that have occurred.
    pub fn transition_count(&self) -> usize {
        self.transition_log.len()
    }

    /// Return the timestamp (ms) of the most recent transition, or `None` if
    /// no transitions have occurred yet.
    pub fn last_transition_ms(&self) -> Option<u64> {
        self.transition_log.last().map(|(ts, _)| *ts)
    }

    // ------------------------------------------------------------------
    // Transitions
    // ------------------------------------------------------------------

    /// Transition from `New` → `Connecting`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the current state is not `New`.
    pub fn start_connecting(&mut self, now_ms: u64) -> Result<(), String> {
        match &self.state {
            PeerConnectionState::New => {
                let new_state = PeerConnectionState::Connecting {
                    started_at_ms: now_ms,
                };
                self.apply_transition(now_ms, new_state);
                Ok(())
            }
            other => Err(format!(
                "start_connecting: invalid transition from {other:?}; expected New"
            )),
        }
    }

    /// Transition from `Connecting` → `Connected`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the current state is not `Connecting`.
    pub fn mark_connected(&mut self, peer_id: String, now_ms: u64) -> Result<(), String> {
        match &self.state {
            PeerConnectionState::Connecting { .. } => {
                let new_state = PeerConnectionState::Connected {
                    peer_id,
                    established_at_ms: now_ms,
                };
                self.apply_transition(now_ms, new_state);
                Ok(())
            }
            other => Err(format!(
                "mark_connected: invalid transition from {other:?}; expected Connecting"
            )),
        }
    }

    /// Transition from `Connected` → `Disconnected`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the current state is not `Connected`.
    pub fn mark_disconnected(&mut self, reason: String, now_ms: u64) -> Result<(), String> {
        match &self.state {
            PeerConnectionState::Connected { peer_id, .. } => {
                let peer_id_clone = peer_id.clone();
                let new_state = PeerConnectionState::Disconnected {
                    peer_id: peer_id_clone,
                    reason,
                };
                self.apply_transition(now_ms, new_state);
                Ok(())
            }
            other => Err(format!(
                "mark_disconnected: invalid transition from {other:?}; expected Connected"
            )),
        }
    }

    /// Transition to `Failed` from any state.
    ///
    /// This transition is always valid regardless of the current state.
    pub fn mark_failed(&mut self, error: String) -> Result<(), String> {
        // Capture a synthetic timestamp of 0 — callers that need a real
        // timestamp should use a variant that accepts `now_ms`.  Providing
        // a fixed value keeps the signature simple and matches the spec.
        let new_state = PeerConnectionState::Failed { error };
        self.apply_transition(0, new_state);
        Ok(())
    }

    /// Reset to the `New` state, clearing the transition log.
    pub fn reset(&mut self) {
        self.state = PeerConnectionState::New;
        self.transition_log.clear();
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn apply_transition(&mut self, now_ms: u64, new_state: PeerConnectionState) {
        self.state = new_state.clone();
        self.transition_log.push((now_ms, new_state));
    }
}

impl Default for PeerStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::{PeerConnectionState, PeerStateMachine};

    // ------------------------------------------------------------------
    // Initial state
    // ------------------------------------------------------------------

    #[test]
    fn test_initial_state_is_new() {
        let machine = PeerStateMachine::new();
        assert_eq!(
            machine.state(),
            &PeerConnectionState::New,
            "fresh machine must be in New state"
        );
        assert_eq!(machine.transition_count(), 0);
        assert!(machine.last_transition_ms().is_none());
    }

    // ------------------------------------------------------------------
    // New → Connecting
    // ------------------------------------------------------------------

    #[test]
    fn test_new_to_connecting() {
        let mut machine = PeerStateMachine::new();
        machine
            .start_connecting(1000)
            .expect("New → Connecting must succeed");
        assert_eq!(
            machine.state(),
            &PeerConnectionState::Connecting {
                started_at_ms: 1000
            },
        );
        assert_eq!(machine.transition_count(), 1);
        assert_eq!(machine.last_transition_ms(), Some(1000));
    }

    // ------------------------------------------------------------------
    // Invalid: Connecting from Connected
    // ------------------------------------------------------------------

    #[test]
    fn test_invalid_transition_connecting_from_connected() {
        let mut machine = PeerStateMachine::new();
        machine.start_connecting(100).expect("New → Connecting");
        machine
            .mark_connected("peer-x".to_string(), 200)
            .expect("Connecting → Connected");

        let result = machine.start_connecting(300);
        assert!(
            result.is_err(),
            "start_connecting from Connected must return Err"
        );
        // State must be unchanged.
        assert!(machine.is_connected());
    }

    // ------------------------------------------------------------------
    // Connecting → Connected
    // ------------------------------------------------------------------

    #[test]
    fn test_connecting_to_connected() {
        let mut machine = PeerStateMachine::new();
        machine.start_connecting(500).expect("New → Connecting");
        machine
            .mark_connected("peer-alpha".to_string(), 750)
            .expect("Connecting → Connected");

        assert!(machine.is_connected());
        assert_eq!(machine.peer_id(), Some("peer-alpha"));
        assert_eq!(machine.transition_count(), 2);
    }

    // ------------------------------------------------------------------
    // Connected → Disconnected
    // ------------------------------------------------------------------

    #[test]
    fn test_connected_to_disconnected() {
        let mut machine = PeerStateMachine::new();
        machine.start_connecting(0).expect("step 1");
        machine
            .mark_connected("peer-beta".to_string(), 100)
            .expect("step 2");
        machine
            .mark_disconnected("network timeout".to_string(), 200)
            .expect("Connected → Disconnected");

        assert!(!machine.is_connected());
        assert_eq!(
            machine.state(),
            &PeerConnectionState::Disconnected {
                peer_id: "peer-beta".to_string(),
                reason: "network timeout".to_string(),
            }
        );
        assert_eq!(machine.transition_count(), 3);
    }

    // ------------------------------------------------------------------
    // Any state → Failed
    // ------------------------------------------------------------------

    #[test]
    fn test_any_state_to_failed() {
        // From New
        let mut m1 = PeerStateMachine::new();
        m1.mark_failed("fatal error from New".to_string())
            .expect("New → Failed must succeed");
        assert_eq!(
            m1.state(),
            &PeerConnectionState::Failed {
                error: "fatal error from New".to_string()
            }
        );

        // From Connecting
        let mut m2 = PeerStateMachine::new();
        m2.start_connecting(0).expect("step");
        m2.mark_failed("fatal error from Connecting".to_string())
            .expect("Connecting → Failed must succeed");
        assert!(matches!(m2.state(), PeerConnectionState::Failed { .. }));

        // From Connected
        let mut m3 = PeerStateMachine::new();
        m3.start_connecting(0).expect("step");
        m3.mark_connected("p".to_string(), 1).expect("step");
        m3.mark_failed("fatal error from Connected".to_string())
            .expect("Connected → Failed must succeed");
        assert!(matches!(m3.state(), PeerConnectionState::Failed { .. }));

        // From Disconnected
        let mut m4 = PeerStateMachine::new();
        m4.start_connecting(0).expect("step");
        m4.mark_connected("p".to_string(), 1).expect("step");
        m4.mark_disconnected("bye".to_string(), 2).expect("step");
        m4.mark_failed("fatal error from Disconnected".to_string())
            .expect("Disconnected → Failed must succeed");
        assert!(matches!(m4.state(), PeerConnectionState::Failed { .. }));
    }

    // ------------------------------------------------------------------
    // reset
    // ------------------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut machine = PeerStateMachine::new();
        machine.start_connecting(10).expect("step");
        machine
            .mark_connected("peer-z".to_string(), 20)
            .expect("step");
        assert_eq!(machine.transition_count(), 2);

        machine.reset();

        assert_eq!(machine.state(), &PeerConnectionState::New);
        assert_eq!(
            machine.transition_count(),
            0,
            "transition log must be cleared after reset"
        );
        assert!(machine.last_transition_ms().is_none());

        // Must be able to start a new connection after reset.
        machine
            .start_connecting(30)
            .expect("must be able to reconnect after reset");
        assert_eq!(machine.transition_count(), 1);
    }

    // ------------------------------------------------------------------
    // is_connected
    // ------------------------------------------------------------------

    #[test]
    fn test_is_connected() {
        let mut machine = PeerStateMachine::new();
        assert!(!machine.is_connected(), "New is not connected");

        machine.start_connecting(0).expect("step");
        assert!(!machine.is_connected(), "Connecting is not connected");

        machine
            .mark_connected("peer-q".to_string(), 1)
            .expect("step");
        assert!(machine.is_connected(), "Connected must be connected");

        machine
            .mark_disconnected("done".to_string(), 2)
            .expect("step");
        assert!(!machine.is_connected(), "Disconnected is not connected");
    }

    // ------------------------------------------------------------------
    // peer_id
    // ------------------------------------------------------------------

    #[test]
    fn test_peer_id() {
        let mut machine = PeerStateMachine::new();
        assert_eq!(machine.peer_id(), None, "New has no peer_id");

        machine.start_connecting(0).expect("step");
        assert_eq!(machine.peer_id(), None, "Connecting has no peer_id");

        machine
            .mark_connected("my-peer".to_string(), 1)
            .expect("step");
        assert_eq!(machine.peer_id(), Some("my-peer"));

        machine
            .mark_disconnected("bye".to_string(), 2)
            .expect("step");
        assert_eq!(machine.peer_id(), None, "Disconnected has no peer_id");
    }

    // ------------------------------------------------------------------
    // transition log
    // ------------------------------------------------------------------

    #[test]
    fn test_transition_log() {
        let mut machine = PeerStateMachine::new();
        assert_eq!(machine.transition_count(), 0);
        assert!(machine.last_transition_ms().is_none());

        machine.start_connecting(1_000).expect("t=1000");
        assert_eq!(machine.transition_count(), 1);
        assert_eq!(machine.last_transition_ms(), Some(1_000));

        machine
            .mark_connected("log-peer".to_string(), 2_000)
            .expect("t=2000");
        assert_eq!(machine.transition_count(), 2);
        assert_eq!(machine.last_transition_ms(), Some(2_000));

        machine
            .mark_disconnected("log test".to_string(), 3_000)
            .expect("t=3000");
        assert_eq!(machine.transition_count(), 3);
        assert_eq!(machine.last_transition_ms(), Some(3_000));
    }
}
