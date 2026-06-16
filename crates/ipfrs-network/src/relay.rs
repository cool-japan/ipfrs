//! Circuit Relay v2 reservation management
//!
//! This module provides [`RelayManager`] which tracks active Circuit Relay v2
//! reservations, enforces a maximum-reservation cap, and evicts stale/expired
//! entries so the node always has a fresh set of relay contacts for NAT
//! traversal.

use parking_lot::RwLock;
use std::collections::HashMap;

// ─── Data types ──────────────────────────────────────────────────────────────

/// State captured when a circuit relay v2 reservation is accepted.
#[derive(Debug, Clone)]
pub struct RelayReservation {
    /// Peer ID of the relay (string form of [`libp2p::PeerId`]).
    pub relay_peer_id: String,
    /// Multiaddr string of the relay.
    pub relay_addr: String,
    /// Unix timestamp (ms) when the reservation was created.
    pub reserved_at_ms: u64,
    /// Unix timestamp (ms) when the reservation expires (TTL from relay).
    pub expires_at_ms: u64,
    /// Optional relay voucher bytes (`None` when the relay did not provide one).
    pub voucher: Option<Vec<u8>>,
}

impl RelayReservation {
    /// Returns `true` when the reservation has passed its expiry time.
    #[inline]
    pub fn is_expired(&self, now_ms: u64) -> bool {
        now_ms >= self.expires_at_ms
    }

    /// Remaining lifetime in milliseconds, or `0` if already expired.
    #[inline]
    pub fn ttl_remaining_ms(&self, now_ms: u64) -> u64 {
        self.expires_at_ms.saturating_sub(now_ms)
    }
}

// ─── Error type ──────────────────────────────────────────────────────────────

/// Errors returned by [`RelayManager`] operations.
#[derive(Debug)]
pub enum RelayError {
    /// The manager has reached its reservation cap.
    AtCapacity {
        /// Maximum allowed reservations.
        max: usize,
    },
    /// A reservation for this relay peer already exists.
    AlreadyExists {
        /// The peer that was already registered.
        relay_peer_id: String,
    },
}

impl std::fmt::Display for RelayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AtCapacity { max } => {
                write!(f, "relay manager at capacity (max {max} reservations)")
            }
            Self::AlreadyExists { relay_peer_id } => {
                write!(
                    f,
                    "relay reservation already exists for peer {relay_peer_id}"
                )
            }
        }
    }
}

impl std::error::Error for RelayError {}

// ─── RelayManager ────────────────────────────────────────────────────────────

/// Manages circuit relay v2 reservations.
///
/// All public methods are safe to call from multiple threads concurrently; the
/// internal [`RwLock`] is a `parking_lot` lock and never blocks async tasks
/// for long.
pub struct RelayManager {
    /// Current reservations, keyed by relay peer ID.
    reservations: RwLock<HashMap<String, RelayReservation>>,
    /// Maximum number of simultaneously held reservations.
    max_reservations: usize,
}

impl RelayManager {
    /// Create a new manager with the given capacity limit.
    pub fn new(max_reservations: usize) -> Self {
        Self {
            reservations: RwLock::new(HashMap::new()),
            max_reservations,
        }
    }

    /// Insert a new reservation.
    ///
    /// Fails with:
    /// - [`RelayError::AlreadyExists`] if `reservation.relay_peer_id` is
    ///   already tracked.
    /// - [`RelayError::AtCapacity`] if the cap would be exceeded.
    pub fn add_reservation(&self, reservation: RelayReservation) -> Result<(), RelayError> {
        let mut map = self.reservations.write();
        if map.contains_key(&reservation.relay_peer_id) {
            return Err(RelayError::AlreadyExists {
                relay_peer_id: reservation.relay_peer_id.clone(),
            });
        }
        if map.len() >= self.max_reservations {
            return Err(RelayError::AtCapacity {
                max: self.max_reservations,
            });
        }
        map.insert(reservation.relay_peer_id.clone(), reservation);
        Ok(())
    }

    /// Remove the reservation for `relay_peer_id`.
    ///
    /// Returns `true` if a reservation existed and was removed.
    pub fn remove_reservation(&self, relay_peer_id: &str) -> bool {
        self.reservations.write().remove(relay_peer_id).is_some()
    }

    /// Look up a reservation by relay peer ID (cloned snapshot).
    pub fn get_reservation(&self, relay_peer_id: &str) -> Option<RelayReservation> {
        self.reservations.read().get(relay_peer_id).cloned()
    }

    /// Return all non-expired reservations as of `now_ms`.
    pub fn active_reservations(&self, now_ms: u64) -> Vec<RelayReservation> {
        self.reservations
            .read()
            .values()
            .filter(|r| !r.is_expired(now_ms))
            .cloned()
            .collect()
    }

    /// Return the peer IDs of all expired reservations.
    pub fn expired_reservations(&self, now_ms: u64) -> Vec<String> {
        self.reservations
            .read()
            .values()
            .filter(|r| r.is_expired(now_ms))
            .map(|r| r.relay_peer_id.clone())
            .collect()
    }

    /// Remove every expired reservation and return the count of removed entries.
    pub fn prune_expired(&self, now_ms: u64) -> usize {
        let mut map = self.reservations.write();
        let before = map.len();
        map.retain(|_, r| !r.is_expired(now_ms));
        before - map.len()
    }

    /// Total number of tracked reservations (active + expired).
    pub fn reservation_count(&self) -> usize {
        self.reservations.read().len()
    }

    /// Returns `true` when the manager cannot accept any more reservations.
    pub fn is_at_capacity(&self) -> bool {
        self.reservations.read().len() >= self.max_reservations
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_reservation(peer_id: &str, reserved_at: u64, expires_at: u64) -> RelayReservation {
        RelayReservation {
            relay_peer_id: peer_id.to_string(),
            relay_addr: format!("/ip4/127.0.0.1/tcp/4001/p2p/{peer_id}"),
            reserved_at_ms: reserved_at,
            expires_at_ms: expires_at,
            voucher: None,
        }
    }

    #[test]
    fn test_add_and_get_reservation() {
        let mgr = RelayManager::new(4);
        let res = make_reservation("peer1", 1000, 5000);
        mgr.add_reservation(res).expect("should succeed");
        let got = mgr.get_reservation("peer1").expect("should be present");
        assert_eq!(got.relay_peer_id, "peer1");
        assert_eq!(got.reserved_at_ms, 1000);
        assert_eq!(got.expires_at_ms, 5000);
    }

    #[test]
    fn test_remove_reservation() {
        let mgr = RelayManager::new(4);
        mgr.add_reservation(make_reservation("peer1", 0, 9999))
            .expect("add ok");
        assert!(mgr.remove_reservation("peer1"));
        assert!(!mgr.remove_reservation("peer1")); // second remove returns false
        assert!(mgr.get_reservation("peer1").is_none());
    }

    #[test]
    fn test_prune_expired() {
        let mgr = RelayManager::new(10);
        mgr.add_reservation(make_reservation("alive", 0, 10_000))
            .expect("ok");
        mgr.add_reservation(make_reservation("dead1", 0, 100))
            .expect("ok");
        mgr.add_reservation(make_reservation("dead2", 0, 200))
            .expect("ok");

        let removed = mgr.prune_expired(500);
        assert_eq!(removed, 2);
        assert_eq!(mgr.reservation_count(), 1);
        assert!(mgr.get_reservation("alive").is_some());
    }

    #[test]
    fn test_active_reservations_filters_expired() {
        let mgr = RelayManager::new(10);
        mgr.add_reservation(make_reservation("alive", 0, 10_000))
            .expect("ok");
        mgr.add_reservation(make_reservation("expired", 0, 50))
            .expect("ok");

        let active = mgr.active_reservations(100);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].relay_peer_id, "alive");
    }

    #[test]
    fn test_capacity_limit() {
        let mgr = RelayManager::new(2);
        mgr.add_reservation(make_reservation("p1", 0, 9999))
            .expect("ok");
        mgr.add_reservation(make_reservation("p2", 0, 9999))
            .expect("ok");
        let err = mgr
            .add_reservation(make_reservation("p3", 0, 9999))
            .expect_err("should fail at capacity");
        assert!(
            matches!(err, RelayError::AtCapacity { max: 2 }),
            "expected AtCapacity, got {err}"
        );
    }

    #[test]
    fn test_duplicate_reservation_error() {
        let mgr = RelayManager::new(4);
        mgr.add_reservation(make_reservation("peer1", 0, 9999))
            .expect("first add ok");
        let err = mgr
            .add_reservation(make_reservation("peer1", 100, 9999))
            .expect_err("duplicate should fail");
        assert!(
            matches!(err, RelayError::AlreadyExists { .. }),
            "expected AlreadyExists, got {err}"
        );
    }

    #[test]
    fn test_reservation_ttl_remaining() {
        let res = make_reservation("p", 0, 5000);
        assert_eq!(res.ttl_remaining_ms(3000), 2000);
        assert_eq!(res.ttl_remaining_ms(5000), 0); // exactly at expiry
        assert_eq!(res.ttl_remaining_ms(6000), 0); // saturating_sub
    }

    #[test]
    fn test_reservation_is_expired() {
        let res = make_reservation("p", 0, 5000);
        assert!(!res.is_expired(4999));
        assert!(res.is_expired(5000));
        assert!(res.is_expired(9999));
    }
}
