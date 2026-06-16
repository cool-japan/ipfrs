//! Connection Pool
//!
//! Provides per-peer connection pooling with idle eviction, state transitions,
//! capacity enforcement, and detailed statistics.
//!
//! # Overview
//!
//! Maintaining a dedicated connection per request is expensive. This module reuses
//! existing connections from a per-peer pool and evicts idle ones that have
//! exceeded the configured `idle_timeout`.
//!
//! # Example
//!
//! ```
//! use ipfrs_network::connection_pool::{ConnectionPool, PoolConfig};
//! use std::time::{Duration, Instant};
//!
//! let config = PoolConfig {
//!     max_connections_per_peer: 4,
//!     max_total_connections: 100,
//!     idle_timeout: Duration::from_secs(60),
//!     connect_timeout: Duration::from_secs(10),
//! };
//!
//! let pool = ConnectionPool::new(config);
//!
//! // Acquire (or create) a connection slot for "peer-1".
//! let conn_id = pool.acquire("peer-1").expect("acquire failed");
//!
//! // Transition to active once the handshake completes.
//! pool.mark_active(conn_id).expect("mark_active failed");
//!
//! // Return to idle when the request finishes.
//! pool.mark_idle(conn_id).expect("mark_idle failed");
//!
//! // Evict connections that have been idle too long.
//! let evicted = pool.evict_idle(Instant::now());
//! println!("Evicted {} idle connections", evicted);
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by [`ConnectionPool`] operations.
#[derive(Debug, Error)]
pub enum PoolError {
    /// The per-peer connection limit has been reached.
    #[error("peer capacity exceeded for '{peer_id}' (max {max})")]
    PeerCapacityExceeded {
        /// Peer identifier.
        peer_id: String,
        /// Configured per-peer maximum.
        max: usize,
    },

    /// The global connection limit has been reached.
    #[error("global capacity exceeded (max {max})")]
    GlobalCapacityExceeded {
        /// Configured global maximum.
        max: usize,
    },

    /// No connection with the given ID exists in the pool.
    #[error("connection {0} not found in pool")]
    ConnectionNotFound(u64),
}

// ---------------------------------------------------------------------------
// ConnectionState
// ---------------------------------------------------------------------------

/// Lifecycle state of a single pooled connection.
#[derive(Debug, Clone)]
pub enum ConnectionState {
    /// Connection is idle and available for reuse.
    Idle {
        /// Timestamp when the connection became idle.
        idle_since: Instant,
    },
    /// Connection is currently handling a request.
    Active {
        /// Timestamp when the active phase started.
        since: Instant,
    },
    /// Connection is being established (handshake in progress).
    Connecting {
        /// Timestamp when the connecting phase started.
        since: Instant,
    },
    /// Connection attempt or in-flight operation failed.
    Failed {
        /// Human-readable failure reason.
        reason: String,
        /// Timestamp when the failure was recorded.
        at: Instant,
    },
}

impl ConnectionState {
    /// Returns `true` if the connection is in the [`Idle`](ConnectionState::Idle) state.
    pub fn is_idle(&self) -> bool {
        matches!(self, ConnectionState::Idle { .. })
    }

    /// Returns `true` if the connection is in the [`Active`](ConnectionState::Active) state.
    pub fn is_active(&self) -> bool {
        matches!(self, ConnectionState::Active { .. })
    }
}

// ---------------------------------------------------------------------------
// PooledConnection
// ---------------------------------------------------------------------------

/// A single connection entry managed by the pool.
#[derive(Debug, Clone)]
pub struct PooledConnection {
    /// Identifier of the remote peer.
    pub peer_id: String,
    /// Pool-assigned unique connection identifier.
    pub connection_id: u64,
    /// Current lifecycle state.
    pub state: ConnectionState,
    /// Timestamp when the pool slot was created.
    pub created_at: Instant,
    /// Cumulative bytes sent over this connection.
    pub bytes_sent: u64,
    /// Cumulative bytes received over this connection.
    pub bytes_received: u64,
}

impl PooledConnection {
    /// Returns `true` when the connection is idle and available for reuse.
    pub fn is_idle(&self) -> bool {
        self.state.is_idle()
    }

    /// Returns `true` when the connection is actively handling a request.
    pub fn is_active(&self) -> bool {
        self.state.is_active()
    }

    /// Returns how long the connection has been idle, or `None` if it is not currently idle.
    pub fn idle_duration(&self, now: Instant) -> Option<Duration> {
        if let ConnectionState::Idle { idle_since } = self.state {
            Some(now.duration_since(idle_since))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// PoolConfig
// ---------------------------------------------------------------------------

/// Configuration knobs for [`ConnectionPool`].
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of connections per remote peer (default: 4).
    pub max_connections_per_peer: usize,
    /// Maximum total connections across all peers (default: 100).
    pub max_total_connections: usize,
    /// Duration a connection may remain idle before eviction (default: 60 s).
    pub idle_timeout: Duration,
    /// Maximum time allowed to establish a new connection (default: 10 s).
    pub connect_timeout: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections_per_peer: 4,
            max_total_connections: 100,
            idle_timeout: Duration::from_secs(60),
            connect_timeout: Duration::from_secs(10),
        }
    }
}

// ---------------------------------------------------------------------------
// PoolStats
// ---------------------------------------------------------------------------

/// Atomic counters tracking pool-wide activity.
#[derive(Debug, Default)]
pub struct PoolStats {
    /// Total successful `acquire` calls (reuse + new slots).
    pub total_acquired: AtomicU64,
    /// Total `release` calls.
    pub total_released: AtomicU64,
    /// Total connections removed by [`ConnectionPool::evict_idle`].
    pub total_evicted: AtomicU64,
    /// Total connections that transitioned to the `Failed` state.
    pub total_failed: AtomicU64,
    /// Total new connection slots created (excludes reuse of idle slots).
    pub total_created: AtomicU64,
}

impl PoolStats {
    /// Take an immutable snapshot of the current counters.
    pub fn snapshot(&self) -> PoolStatsSnapshot {
        PoolStatsSnapshot {
            total_acquired: self.total_acquired.load(Ordering::Relaxed),
            total_released: self.total_released.load(Ordering::Relaxed),
            total_evicted: self.total_evicted.load(Ordering::Relaxed),
            total_failed: self.total_failed.load(Ordering::Relaxed),
            total_created: self.total_created.load(Ordering::Relaxed),
        }
    }
}

/// Immutable snapshot of [`PoolStats`] suitable for logging and export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolStatsSnapshot {
    /// See [`PoolStats::total_acquired`].
    pub total_acquired: u64,
    /// See [`PoolStats::total_released`].
    pub total_released: u64,
    /// See [`PoolStats::total_evicted`].
    pub total_evicted: u64,
    /// See [`PoolStats::total_failed`].
    pub total_failed: u64,
    /// See [`PoolStats::total_created`].
    pub total_created: u64,
}

// ---------------------------------------------------------------------------
// ConnectionPool
// ---------------------------------------------------------------------------

/// Per-peer connection pool with idle eviction and capacity enforcement.
///
/// The pool stores connections grouped by `peer_id`.  On [`acquire`](Self::acquire)
/// it first looks for an existing idle connection; if none is available it
/// creates a new `Connecting` slot, subject to both per-peer and global limits.
pub struct ConnectionPool {
    /// All connections, keyed by peer identifier.
    connections: RwLock<HashMap<String, Vec<PooledConnection>>>,
    /// Monotonically increasing connection-ID generator.
    next_id: AtomicU64,
    /// Pool configuration.
    config: PoolConfig,
    /// Live counters.
    stats: PoolStats,
}

impl ConnectionPool {
    /// Create a new pool with the given configuration.
    pub fn new(config: PoolConfig) -> Self {
        Self {
            connections: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            config,
            stats: PoolStats::default(),
        }
    }

    /// Create a new pool with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(PoolConfig::default())
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Count the total number of connections across all peers (read-lock held by caller).
    fn total_count_locked(map: &HashMap<String, Vec<PooledConnection>>) -> usize {
        map.values().map(|v| v.len()).sum()
    }

    /// Find the first idle connection for `peer_id` and transition it to
    /// `Connecting`, returning its ID.  Returns `None` when no idle slot exists.
    fn reuse_idle(map: &mut HashMap<String, Vec<PooledConnection>>, peer_id: &str) -> Option<u64> {
        let conns = map.get_mut(peer_id)?;
        let slot = conns.iter_mut().find(|c| c.is_idle())?;
        let id = slot.connection_id;
        slot.state = ConnectionState::Connecting {
            since: Instant::now(),
        };
        Some(id)
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Acquire a connection to `peer_id`.
    ///
    /// 1. If an idle connection exists for the peer it is transitioned to
    ///    `Connecting` and its ID is returned.
    /// 2. Otherwise a new slot in `Connecting` state is created, provided
    ///    neither the per-peer limit nor the global limit would be exceeded.
    ///
    /// # Errors
    ///
    /// - [`PoolError::PeerCapacityExceeded`] when the per-peer limit is reached.
    /// - [`PoolError::GlobalCapacityExceeded`] when the global limit is reached.
    pub fn acquire(&self, peer_id: &str) -> Result<u64, PoolError> {
        let mut map = self
            .connections
            .write()
            .expect("connection pool lock poisoned");

        // Try to reuse an idle slot first.
        if let Some(id) = Self::reuse_idle(&mut map, peer_id) {
            self.stats.total_acquired.fetch_add(1, Ordering::Relaxed);
            return Ok(id);
        }

        // Check per-peer capacity.
        let peer_count = map.get(peer_id).map(|v| v.len()).unwrap_or(0);
        if peer_count >= self.config.max_connections_per_peer {
            return Err(PoolError::PeerCapacityExceeded {
                peer_id: peer_id.to_string(),
                max: self.config.max_connections_per_peer,
            });
        }

        // Check global capacity.
        let total = Self::total_count_locked(&map);
        if total >= self.config.max_total_connections {
            return Err(PoolError::GlobalCapacityExceeded {
                max: self.config.max_total_connections,
            });
        }

        // Allocate a new slot.
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let conn = PooledConnection {
            peer_id: peer_id.to_string(),
            connection_id: id,
            state: ConnectionState::Connecting {
                since: Instant::now(),
            },
            created_at: Instant::now(),
            bytes_sent: 0,
            bytes_received: 0,
        };
        map.entry(peer_id.to_string()).or_default().push(conn);

        self.stats.total_acquired.fetch_add(1, Ordering::Relaxed);
        self.stats.total_created.fetch_add(1, Ordering::Relaxed);
        Ok(id)
    }

    /// Transition `connection_id` to [`Active`](ConnectionState::Active).
    ///
    /// # Errors
    ///
    /// Returns [`PoolError::ConnectionNotFound`] if no connection with the given
    /// ID exists.
    pub fn mark_active(&self, connection_id: u64) -> Result<(), PoolError> {
        let mut map = self
            .connections
            .write()
            .expect("connection pool lock poisoned");

        for conns in map.values_mut() {
            if let Some(c) = conns.iter_mut().find(|c| c.connection_id == connection_id) {
                c.state = ConnectionState::Active {
                    since: Instant::now(),
                };
                return Ok(());
            }
        }
        Err(PoolError::ConnectionNotFound(connection_id))
    }

    /// Transition `connection_id` to [`Idle`](ConnectionState::Idle).
    ///
    /// # Errors
    ///
    /// Returns [`PoolError::ConnectionNotFound`] if no connection with the given
    /// ID exists.
    pub fn mark_idle(&self, connection_id: u64) -> Result<(), PoolError> {
        let mut map = self
            .connections
            .write()
            .expect("connection pool lock poisoned");

        for conns in map.values_mut() {
            if let Some(c) = conns.iter_mut().find(|c| c.connection_id == connection_id) {
                c.state = ConnectionState::Idle {
                    idle_since: Instant::now(),
                };
                return Ok(());
            }
        }
        Err(PoolError::ConnectionNotFound(connection_id))
    }

    /// Transition `connection_id` to [`Failed`](ConnectionState::Failed) with
    /// the supplied human-readable `reason`.
    ///
    /// # Errors
    ///
    /// Returns [`PoolError::ConnectionNotFound`] if no connection with the given
    /// ID exists.
    pub fn mark_failed(&self, connection_id: u64, reason: &str) -> Result<(), PoolError> {
        let mut map = self
            .connections
            .write()
            .expect("connection pool lock poisoned");

        for conns in map.values_mut() {
            if let Some(c) = conns.iter_mut().find(|c| c.connection_id == connection_id) {
                c.state = ConnectionState::Failed {
                    reason: reason.to_string(),
                    at: Instant::now(),
                };
                self.stats.total_failed.fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }
        }
        Err(PoolError::ConnectionNotFound(connection_id))
    }

    /// Remove `connection_id` from the pool entirely.
    ///
    /// This is a no-op when the connection does not exist.  Empty peer buckets
    /// are cleaned up automatically.
    pub fn release(&self, connection_id: u64) {
        let mut map = self
            .connections
            .write()
            .expect("connection pool lock poisoned");

        let mut removed = false;
        for conns in map.values_mut() {
            let before = conns.len();
            conns.retain(|c| c.connection_id != connection_id);
            if conns.len() < before {
                removed = true;
                break;
            }
        }

        // Remove empty peer buckets to keep the map tidy.
        map.retain(|_, v| !v.is_empty());

        if removed {
            self.stats.total_released.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Remove all connections that have been idle longer than
    /// [`PoolConfig::idle_timeout`], measured against `now`.
    ///
    /// Returns the number of connections evicted.
    pub fn evict_idle(&self, now: Instant) -> usize {
        let mut map = self
            .connections
            .write()
            .expect("connection pool lock poisoned");

        let timeout = self.config.idle_timeout;
        let mut evicted: usize = 0;

        for conns in map.values_mut() {
            let before = conns.len();
            conns.retain(|c| {
                if let Some(dur) = c.idle_duration(now) {
                    dur <= timeout
                } else {
                    true
                }
            });
            evicted += before - conns.len();
        }

        map.retain(|_, v| !v.is_empty());

        self.stats
            .total_evicted
            .fetch_add(evicted as u64, Ordering::Relaxed);
        evicted
    }

    // ------------------------------------------------------------------
    // Aggregate queries
    // ------------------------------------------------------------------

    /// Total number of connections currently in the pool (all states).
    pub fn connection_count(&self) -> usize {
        let map = self
            .connections
            .read()
            .expect("connection pool lock poisoned");
        Self::total_count_locked(&map)
    }

    /// Number of connections in the [`Idle`](ConnectionState::Idle) state.
    pub fn idle_count(&self) -> usize {
        let map = self
            .connections
            .read()
            .expect("connection pool lock poisoned");
        map.values()
            .flat_map(|v| v.iter())
            .filter(|c| c.is_idle())
            .count()
    }

    /// Number of connections in the [`Active`](ConnectionState::Active) state.
    pub fn active_count(&self) -> usize {
        let map = self
            .connections
            .read()
            .expect("connection pool lock poisoned");
        map.values()
            .flat_map(|v| v.iter())
            .filter(|c| c.is_active())
            .count()
    }

    /// Return an immutable snapshot of the pool-wide statistics.
    pub fn stats(&self) -> PoolStatsSnapshot {
        self.stats.snapshot()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    fn pool_with_limits(per_peer: usize, total: usize) -> ConnectionPool {
        ConnectionPool::new(PoolConfig {
            max_connections_per_peer: per_peer,
            max_total_connections: total,
            idle_timeout: Duration::from_secs(60),
            connect_timeout: Duration::from_secs(10),
        })
    }

    // -----------------------------------------------------------------------
    // 1. acquire returns a valid connection_id
    // -----------------------------------------------------------------------
    #[test]
    fn acquire_returns_connection_id() {
        let pool = pool_with_limits(4, 100);
        let id = pool.acquire("peer-a").expect("acquire should succeed");
        assert!(id > 0, "connection_id must be > 0");
    }

    // -----------------------------------------------------------------------
    // 2. acquire increments total_acquired stat
    // -----------------------------------------------------------------------
    #[test]
    fn acquire_increments_stats() {
        let pool = pool_with_limits(4, 100);
        pool.acquire("peer-a").expect("first acquire");
        pool.acquire("peer-a").expect("second acquire");
        let snap = pool.stats();
        assert_eq!(snap.total_acquired, 2);
    }

    // -----------------------------------------------------------------------
    // 3. New slot starts in Connecting state
    // -----------------------------------------------------------------------
    #[test]
    fn new_connection_starts_in_connecting_state() {
        let pool = pool_with_limits(4, 100);
        let id = pool.acquire("peer-b").expect("acquire");
        let map = pool.connections.read().expect("lock");
        let conn = map["peer-b"]
            .iter()
            .find(|c| c.connection_id == id)
            .expect("connection should be present");
        assert!(
            matches!(conn.state, ConnectionState::Connecting { .. }),
            "expected Connecting state after acquire"
        );
    }

    // -----------------------------------------------------------------------
    // 4. mark_active transitions state
    // -----------------------------------------------------------------------
    #[test]
    fn mark_active_transitions_state() {
        let pool = pool_with_limits(4, 100);
        let id = pool.acquire("peer-c").expect("acquire");
        pool.mark_active(id).expect("mark_active");

        let map = pool.connections.read().expect("lock");
        let conn = map["peer-c"]
            .iter()
            .find(|c| c.connection_id == id)
            .expect("connection should exist");
        assert!(conn.is_active(), "state should be Active after mark_active");
    }

    // -----------------------------------------------------------------------
    // 5. mark_idle transitions state
    // -----------------------------------------------------------------------
    #[test]
    fn mark_idle_transitions_state() {
        let pool = pool_with_limits(4, 100);
        let id = pool.acquire("peer-d").expect("acquire");
        pool.mark_active(id).expect("mark_active");
        pool.mark_idle(id).expect("mark_idle");

        let map = pool.connections.read().expect("lock");
        let conn = map["peer-d"]
            .iter()
            .find(|c| c.connection_id == id)
            .expect("connection should exist");
        assert!(conn.is_idle(), "state should be Idle after mark_idle");
    }

    // -----------------------------------------------------------------------
    // 6. mark_failed records reason
    // -----------------------------------------------------------------------
    #[test]
    fn mark_failed_records_reason() {
        let pool = pool_with_limits(4, 100);
        let id = pool.acquire("peer-e").expect("acquire");
        pool.mark_failed(id, "timeout").expect("mark_failed");

        let map = pool.connections.read().expect("lock");
        let conn = map["peer-e"]
            .iter()
            .find(|c| c.connection_id == id)
            .expect("connection should exist");
        match &conn.state {
            ConnectionState::Failed { reason, .. } => {
                assert_eq!(reason, "timeout");
            }
            other => panic!("expected Failed state, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // 7. mark_failed increments total_failed stat
    // -----------------------------------------------------------------------
    #[test]
    fn mark_failed_increments_stats() {
        let pool = pool_with_limits(4, 100);
        let id = pool.acquire("peer-f").expect("acquire");
        pool.mark_failed(id, "connection refused")
            .expect("mark_failed");
        assert_eq!(pool.stats().total_failed, 1);
    }

    // -----------------------------------------------------------------------
    // 8. release removes connection from pool
    // -----------------------------------------------------------------------
    #[test]
    fn release_removes_connection() {
        let pool = pool_with_limits(4, 100);
        let id = pool.acquire("peer-g").expect("acquire");
        assert_eq!(pool.connection_count(), 1);
        pool.release(id);
        assert_eq!(pool.connection_count(), 0);
    }

    // -----------------------------------------------------------------------
    // 9. release increments total_released stat
    // -----------------------------------------------------------------------
    #[test]
    fn release_increments_stats() {
        let pool = pool_with_limits(4, 100);
        let id = pool.acquire("peer-h").expect("acquire");
        pool.release(id);
        assert_eq!(pool.stats().total_released, 1);
    }

    // -----------------------------------------------------------------------
    // 10. Per-peer capacity enforcement
    // -----------------------------------------------------------------------
    #[test]
    fn per_peer_capacity_enforced() {
        let pool = pool_with_limits(2, 100);
        pool.acquire("peer-i").expect("first");
        pool.acquire("peer-i").expect("second");
        let err = pool.acquire("peer-i").expect_err("third should fail");
        assert!(
            matches!(err, PoolError::PeerCapacityExceeded { max: 2, .. }),
            "unexpected error: {err}"
        );
    }

    // -----------------------------------------------------------------------
    // 11. Global capacity enforcement
    // -----------------------------------------------------------------------
    #[test]
    fn global_capacity_enforced() {
        let pool = pool_with_limits(10, 3);
        pool.acquire("peer-j").expect("slot 1");
        pool.acquire("peer-k").expect("slot 2");
        pool.acquire("peer-l").expect("slot 3");
        let err = pool.acquire("peer-m").expect_err("slot 4 should fail");
        assert!(
            matches!(err, PoolError::GlobalCapacityExceeded { max: 3 }),
            "unexpected error: {err}"
        );
    }

    // -----------------------------------------------------------------------
    // 12. idle connection is reused on acquire (no new slot created)
    // -----------------------------------------------------------------------
    #[test]
    fn idle_connection_is_reused_on_acquire() {
        let pool = pool_with_limits(4, 100);
        let id1 = pool.acquire("peer-n").expect("first acquire");
        pool.mark_active(id1).expect("mark_active");
        pool.mark_idle(id1).expect("mark_idle");

        // There is now one idle connection.  A second acquire should reuse it.
        let id2 = pool.acquire("peer-n").expect("second acquire");
        assert_eq!(id1, id2, "should reuse the existing idle connection");

        // Still only one connection in the pool.
        assert_eq!(pool.connection_count(), 1);

        let snap = pool.stats();
        // total_created should be 1 (only the first slot creation), not 2.
        assert_eq!(snap.total_created, 1);
        assert_eq!(snap.total_acquired, 2);
    }

    // -----------------------------------------------------------------------
    // 13. evict_idle removes timed-out connections
    // -----------------------------------------------------------------------
    #[test]
    fn evict_idle_removes_timed_out_connections() {
        let pool = ConnectionPool::new(PoolConfig {
            max_connections_per_peer: 4,
            max_total_connections: 100,
            idle_timeout: Duration::from_millis(50),
            connect_timeout: Duration::from_secs(10),
        });

        let id = pool.acquire("peer-o").expect("acquire");
        pool.mark_active(id).expect("mark_active");
        pool.mark_idle(id).expect("mark_idle");

        // Sleep past the idle_timeout.
        thread::sleep(Duration::from_millis(100));

        let evicted = pool.evict_idle(Instant::now());
        assert_eq!(evicted, 1, "one connection should have been evicted");
        assert_eq!(pool.connection_count(), 0);
        assert_eq!(pool.stats().total_evicted, 1);
    }

    // -----------------------------------------------------------------------
    // 14. evict_idle does NOT remove non-idle connections
    // -----------------------------------------------------------------------
    #[test]
    fn evict_idle_skips_active_connections() {
        let pool = ConnectionPool::new(PoolConfig {
            max_connections_per_peer: 4,
            max_total_connections: 100,
            idle_timeout: Duration::from_millis(1),
            connect_timeout: Duration::from_secs(10),
        });

        let id = pool.acquire("peer-p").expect("acquire");
        pool.mark_active(id).expect("mark_active");

        // Even after sleeping, active connections should not be evicted.
        thread::sleep(Duration::from_millis(10));
        let evicted = pool.evict_idle(Instant::now());
        assert_eq!(evicted, 0);
        assert_eq!(pool.connection_count(), 1);
    }

    // -----------------------------------------------------------------------
    // 15. idle_count / active_count correctness
    // -----------------------------------------------------------------------
    #[test]
    fn idle_and_active_counts_are_correct() {
        let pool = pool_with_limits(10, 100);

        let id1 = pool.acquire("peer-q").expect("conn1");
        let id2 = pool.acquire("peer-q").expect("conn2");
        let id3 = pool.acquire("peer-q").expect("conn3");

        pool.mark_active(id1).expect("active id1");
        pool.mark_active(id2).expect("active id2");
        pool.mark_idle(id2).expect("idle id2");
        // id3 stays in Connecting state (neither idle nor active).

        assert_eq!(pool.active_count(), 1, "only id1 is active");
        assert_eq!(pool.idle_count(), 1, "only id2 is idle");
        assert_eq!(pool.connection_count(), 3);

        // Release id3 and check totals.
        pool.release(id3);
        assert_eq!(pool.connection_count(), 2);
    }

    // -----------------------------------------------------------------------
    // 16. Unknown connection_id returns ConnectionNotFound
    // -----------------------------------------------------------------------
    #[test]
    fn unknown_connection_id_returns_not_found() {
        let pool = pool_with_limits(4, 100);
        assert!(matches!(
            pool.mark_active(9999),
            Err(PoolError::ConnectionNotFound(9999))
        ));
        assert!(matches!(
            pool.mark_idle(9999),
            Err(PoolError::ConnectionNotFound(9999))
        ));
        assert!(matches!(
            pool.mark_failed(9999, "x"),
            Err(PoolError::ConnectionNotFound(9999))
        ));
    }

    // -----------------------------------------------------------------------
    // 17. Stats accumulation across mixed operations
    // -----------------------------------------------------------------------
    #[test]
    fn stats_accumulate_correctly() {
        let pool = pool_with_limits(10, 100);

        let id1 = pool.acquire("peer-r").expect("a1");
        let id2 = pool.acquire("peer-r").expect("a2");
        pool.mark_active(id1).expect("active1");
        pool.mark_failed(id2, "refused").expect("failed2");
        pool.release(id1);

        let snap = pool.stats();
        assert_eq!(snap.total_acquired, 2);
        assert_eq!(snap.total_created, 2);
        assert_eq!(snap.total_failed, 1);
        assert_eq!(snap.total_released, 1);
    }

    // -----------------------------------------------------------------------
    // 18. release on non-existent id is a no-op (no panic)
    // -----------------------------------------------------------------------
    #[test]
    fn release_nonexistent_is_noop() {
        let pool = pool_with_limits(4, 100);
        pool.release(42); // should not panic or change released counter
        assert_eq!(pool.stats().total_released, 0);
    }

    // -----------------------------------------------------------------------
    // 19. idle_duration returns None for non-idle states
    // -----------------------------------------------------------------------
    #[test]
    fn idle_duration_returns_none_for_active() {
        let conn = PooledConnection {
            peer_id: "peer-s".to_string(),
            connection_id: 1,
            state: ConnectionState::Active {
                since: Instant::now(),
            },
            created_at: Instant::now(),
            bytes_sent: 0,
            bytes_received: 0,
        };
        assert!(conn.idle_duration(Instant::now()).is_none());
    }

    // -----------------------------------------------------------------------
    // 20. Connections from different peers are independent
    // -----------------------------------------------------------------------
    #[test]
    fn different_peers_are_independent() {
        let pool = pool_with_limits(2, 100);

        // Fill peer-t to capacity.
        pool.acquire("peer-t").expect("t1");
        pool.acquire("peer-t").expect("t2");
        let err = pool.acquire("peer-t").expect_err("t3 over limit");
        assert!(matches!(err, PoolError::PeerCapacityExceeded { .. }));

        // peer-u should still be able to acquire.
        pool.acquire("peer-u").expect("u1 should succeed");
    }
}

// ===========================================================================
// PeerConnectionPool — tick-based, lock-free, pure-ownership pool
// ===========================================================================

/// Lifecycle state for a connection managed by [`PeerConnectionPool`].
///
/// Named `PoolConnectionState` to avoid collision with the [`ConnectionState`]
/// used by the async [`ConnectionPool`] above.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolConnectionState {
    /// Available for reuse by a caller.
    Idle,
    /// Currently acquired by a caller.
    InUse,
    /// Marked for teardown; will be removed on the next [`PeerConnectionPool::evict_idle`] call.
    Closing,
}

/// A single connection entry managed by [`PeerConnectionPool`].
#[derive(Debug, Clone)]
pub struct PeerPooledConnection {
    /// Unique identifier for this connection within the pool.
    pub conn_id: u64,
    /// Remote peer this connection is associated with.
    pub peer_id: String,
    /// Current lifecycle state.
    pub state: PoolConnectionState,
    /// Logical tick at which this connection was created.
    pub created_at_tick: u64,
    /// Logical tick at which this connection was last used (acquired or released).
    pub last_used_tick: u64,
    /// Total number of times this connection has been acquired.
    pub use_count: u64,
}

/// Configuration for [`PeerConnectionPool`].
#[derive(Debug, Clone)]
pub struct PeerPoolConfig {
    /// Maximum number of connections allowed per peer (default: 4).
    pub max_per_peer: usize,
    /// Close idle connections that have not been used for this many ticks (default: 60).
    pub idle_timeout_ticks: u64,
    /// Global cap across all peers (default: 100).
    pub max_total: usize,
}

impl Default for PeerPoolConfig {
    fn default() -> Self {
        Self {
            max_per_peer: 4,
            idle_timeout_ticks: 60,
            max_total: 100,
        }
    }
}

/// A point-in-time snapshot of [`PeerConnectionPool`] statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerPoolStats {
    /// Total connections currently tracked by the pool.
    pub total_connections: usize,
    /// Connections currently in the [`Idle`](PoolConnectionState::Idle) state.
    pub idle_connections: usize,
    /// Connections currently in the [`InUse`](PoolConnectionState::InUse) state.
    pub in_use_connections: usize,
    /// Number of distinct peers with at least one connection.
    pub total_peers: usize,
    /// Cumulative number of successful acquire calls.
    pub total_acquired: u64,
    /// Cumulative number of successful release calls.
    pub total_released: u64,
    /// Cumulative number of connections removed by eviction.
    pub total_evicted: u64,
}

/// Tick-based, single-threaded peer connection pool.
///
/// Manages a set of reusable [`PeerPooledConnection`]s keyed by a monotonically
/// increasing `conn_id`.  Callers drive time by passing an explicit `tick`
/// counter; the pool does not use wall-clock time.
///
/// # Example
///
/// ```
/// use ipfrs_network::connection_pool::{PeerConnectionPool, PeerPoolConfig};
///
/// let mut pool = PeerConnectionPool::new(PeerPoolConfig::default());
/// let tick = 0_u64;
///
/// // Acquire (or create) a connection for "peer-1".
/// let conn_id = pool.acquire("peer-1", tick).expect("should succeed");
///
/// // Return the connection to the idle pool.
/// assert!(pool.release(conn_id, tick + 1));
///
/// // Evict connections that have been idle for too long.
/// pool.evict_idle(tick + 100);
/// ```
pub struct PeerConnectionPool {
    /// All connections, keyed by their unique `conn_id`.
    pub connections: HashMap<u64, PeerPooledConnection>,
    /// Monotonically increasing counter for assigning connection IDs.
    pub next_conn_id: u64,
    /// Pool configuration.
    pub config: PeerPoolConfig,
    /// Cumulative successful acquires.
    pub total_acquired: u64,
    /// Cumulative successful releases.
    pub total_released: u64,
    /// Cumulative evictions.
    pub total_evicted: u64,
}

impl PeerConnectionPool {
    /// Create a new, empty pool with the given configuration.
    pub fn new(config: PeerPoolConfig) -> Self {
        Self {
            connections: HashMap::new(),
            next_conn_id: 1,
            config,
            total_acquired: 0,
            total_released: 0,
            total_evicted: 0,
        }
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Count connections belonging to `peer_id`.
    fn peer_connection_count(&self, peer_id: &str) -> usize {
        self.connections
            .values()
            .filter(|c| c.peer_id == peer_id)
            .count()
    }

    /// Allocate the next connection ID.
    fn alloc_id(&mut self) -> u64 {
        let id = self.next_conn_id;
        self.next_conn_id += 1;
        id
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Acquire a connection for `peer_id` at the given logical `tick`.
    ///
    /// Returns `Some(conn_id)` on success:
    /// - If an [`Idle`](PoolConnectionState::Idle) connection exists for the
    ///   peer it is reused (set to [`InUse`](PoolConnectionState::InUse)).
    /// - Otherwise a new connection is created if the per-peer and global caps
    ///   allow it.
    ///
    /// Returns `None` when all capacity limits are exhausted.
    pub fn acquire(&mut self, peer_id: &str, tick: u64) -> Option<u64> {
        // 1. Reuse an existing idle connection for this peer.
        let idle_id = self
            .connections
            .values()
            .find(|c| c.peer_id == peer_id && c.state == PoolConnectionState::Idle)
            .map(|c| c.conn_id);

        if let Some(id) = idle_id {
            let conn = self.connections.get_mut(&id)?;
            conn.state = PoolConnectionState::InUse;
            conn.last_used_tick = tick;
            conn.use_count += 1;
            self.total_acquired += 1;
            return Some(id);
        }

        // 2. Check capacity before creating a new connection.
        let peer_count = self.peer_connection_count(peer_id);
        if peer_count >= self.config.max_per_peer {
            return None;
        }
        if self.connections.len() >= self.config.max_total {
            return None;
        }

        // 3. Create a new InUse connection.
        let id = self.alloc_id();
        let conn = PeerPooledConnection {
            conn_id: id,
            peer_id: peer_id.to_string(),
            state: PoolConnectionState::InUse,
            created_at_tick: tick,
            last_used_tick: tick,
            use_count: 1,
        };
        self.connections.insert(id, conn);
        self.total_acquired += 1;
        Some(id)
    }

    /// Return a connection to the idle pool.
    ///
    /// Returns `true` on success, `false` if the connection was not found or
    /// was not in the [`InUse`](PoolConnectionState::InUse) state.
    pub fn release(&mut self, conn_id: u64, tick: u64) -> bool {
        match self.connections.get_mut(&conn_id) {
            Some(conn) if conn.state == PoolConnectionState::InUse => {
                conn.state = PoolConnectionState::Idle;
                conn.last_used_tick = tick;
                self.total_released += 1;
                true
            }
            _ => false,
        }
    }

    /// Mark a connection as [`Closing`](PoolConnectionState::Closing).
    ///
    /// Returns `false` if no connection with the given ID exists.
    pub fn close(&mut self, conn_id: u64) -> bool {
        match self.connections.get_mut(&conn_id) {
            Some(conn) => {
                conn.state = PoolConnectionState::Closing;
                true
            }
            None => false,
        }
    }

    /// Evict connections that are stale or marked for teardown.
    ///
    /// Removes:
    /// - [`Idle`](PoolConnectionState::Idle) connections whose
    ///   `last_used_tick` is at least `idle_timeout_ticks` ticks in the past.
    /// - All [`Closing`](PoolConnectionState::Closing) connections.
    ///
    /// [`total_evicted`](Self::total_evicted) is incremented for each removed
    /// connection.
    pub fn evict_idle(&mut self, tick: u64) {
        let timeout = self.config.idle_timeout_ticks;
        let mut evicted: u64 = 0;

        self.connections.retain(|_, conn| {
            let should_remove = match conn.state {
                PoolConnectionState::Idle => tick.saturating_sub(conn.last_used_tick) >= timeout,
                PoolConnectionState::Closing => true,
                PoolConnectionState::InUse => false,
            };
            if should_remove {
                evicted += 1;
            }
            !should_remove
        });

        self.total_evicted += evicted;
    }

    /// Return all connections associated with `peer_id`, sorted ascending by
    /// `conn_id`.
    pub fn connections_for_peer<'a>(&'a self, peer_id: &str) -> Vec<&'a PeerPooledConnection> {
        let mut result: Vec<&PeerPooledConnection> = self
            .connections
            .values()
            .filter(|c| c.peer_id == peer_id)
            .collect();
        result.sort_by_key(|c| c.conn_id);
        result
    }

    /// Return a point-in-time snapshot of pool statistics.
    pub fn stats(&self) -> PeerPoolStats {
        let total_connections = self.connections.len();
        let idle_connections = self
            .connections
            .values()
            .filter(|c| c.state == PoolConnectionState::Idle)
            .count();
        let in_use_connections = self
            .connections
            .values()
            .filter(|c| c.state == PoolConnectionState::InUse)
            .count();

        // Count distinct peers that have at least one connection.
        let mut peer_set = std::collections::HashSet::new();
        for conn in self.connections.values() {
            peer_set.insert(conn.peer_id.as_str());
        }
        let total_peers = peer_set.len();

        PeerPoolStats {
            total_connections,
            idle_connections,
            in_use_connections,
            total_peers,
            total_acquired: self.total_acquired,
            total_released: self.total_released,
            total_evicted: self.total_evicted,
        }
    }
}

// ===========================================================================
// PeerConnectionPool tests
// ===========================================================================

#[cfg(test)]
mod peer_pool_tests {
    use super::{PeerConnectionPool, PeerPoolConfig, PoolConnectionState};

    fn default_pool() -> PeerConnectionPool {
        PeerConnectionPool::new(PeerPoolConfig::default())
    }

    fn small_pool(max_per_peer: usize, max_total: usize) -> PeerConnectionPool {
        PeerConnectionPool::new(PeerPoolConfig {
            max_per_peer,
            idle_timeout_ticks: 60,
            max_total,
        })
    }

    // -----------------------------------------------------------------------
    // 1. new() starts empty
    // -----------------------------------------------------------------------
    #[test]
    fn new_starts_empty() {
        let pool = default_pool();
        let stats = pool.stats();
        assert_eq!(stats.total_connections, 0);
        assert_eq!(stats.idle_connections, 0);
        assert_eq!(stats.in_use_connections, 0);
        assert_eq!(stats.total_peers, 0);
        assert_eq!(stats.total_acquired, 0);
        assert_eq!(stats.total_released, 0);
        assert_eq!(stats.total_evicted, 0);
    }

    // -----------------------------------------------------------------------
    // 2. acquire creates new connection for a new peer
    // -----------------------------------------------------------------------
    #[test]
    fn acquire_creates_new_connection_for_new_peer() {
        let mut pool = default_pool();
        let id = pool.acquire("peer-a", 0);
        assert!(id.is_some());
        assert_eq!(pool.connections.len(), 1);
    }

    // -----------------------------------------------------------------------
    // 3. acquire reuses idle connection
    // -----------------------------------------------------------------------
    #[test]
    fn acquire_reuses_idle_connection() {
        let mut pool = default_pool();
        let id1 = pool.acquire("peer-b", 0).expect("first acquire");
        pool.release(id1, 1);
        let id2 = pool.acquire("peer-b", 2).expect("second acquire");
        // Should reuse the same slot, not create a new one.
        assert_eq!(id1, id2);
        assert_eq!(pool.connections.len(), 1);
    }

    // -----------------------------------------------------------------------
    // 4. acquire returns None when peer is at max_per_peer
    // -----------------------------------------------------------------------
    #[test]
    fn acquire_returns_none_at_max_per_peer() {
        let mut pool = small_pool(2, 100);
        let _a = pool.acquire("peer-c", 0).expect("slot 1");
        let _b = pool.acquire("peer-c", 0).expect("slot 2");
        let result = pool.acquire("peer-c", 0);
        assert!(result.is_none(), "should be None when at per-peer cap");
    }

    // -----------------------------------------------------------------------
    // 5. acquire returns None when total is at max_total
    // -----------------------------------------------------------------------
    #[test]
    fn acquire_returns_none_at_max_total() {
        let mut pool = small_pool(10, 2);
        pool.acquire("peer-d", 0).expect("slot 1");
        pool.acquire("peer-e", 0).expect("slot 2");
        let result = pool.acquire("peer-f", 0);
        assert!(result.is_none(), "should be None when at global cap");
    }

    // -----------------------------------------------------------------------
    // 6. acquire sets InUse state
    // -----------------------------------------------------------------------
    #[test]
    fn acquire_sets_in_use_state() {
        let mut pool = default_pool();
        let id = pool.acquire("peer-g", 0).expect("acquire");
        let conn = pool.connections.get(&id).expect("conn exists");
        assert_eq!(conn.state, PoolConnectionState::InUse);
    }

    // -----------------------------------------------------------------------
    // 7. acquire increments use_count
    // -----------------------------------------------------------------------
    #[test]
    fn acquire_increments_use_count() {
        let mut pool = default_pool();
        let id = pool.acquire("peer-h", 0).expect("first acquire");
        pool.release(id, 1);
        pool.acquire("peer-h", 2).expect("second acquire");
        let conn = pool.connections.get(&id).expect("conn exists");
        assert_eq!(conn.use_count, 2);
    }

    // -----------------------------------------------------------------------
    // 8. release sets Idle state
    // -----------------------------------------------------------------------
    #[test]
    fn release_sets_idle_state() {
        let mut pool = default_pool();
        let id = pool.acquire("peer-i", 0).expect("acquire");
        assert!(pool.release(id, 1));
        let conn = pool.connections.get(&id).expect("conn exists");
        assert_eq!(conn.state, PoolConnectionState::Idle);
    }

    // -----------------------------------------------------------------------
    // 9. release updates last_used_tick
    // -----------------------------------------------------------------------
    #[test]
    fn release_updates_last_used_tick() {
        let mut pool = default_pool();
        let id = pool.acquire("peer-j", 0).expect("acquire");
        assert!(pool.release(id, 42));
        let conn = pool.connections.get(&id).expect("conn exists");
        assert_eq!(conn.last_used_tick, 42);
    }

    // -----------------------------------------------------------------------
    // 10. release returns false if not found
    // -----------------------------------------------------------------------
    #[test]
    fn release_false_if_not_found() {
        let mut pool = default_pool();
        assert!(!pool.release(9999, 0));
    }

    // -----------------------------------------------------------------------
    // 11. release returns false if already Idle
    // -----------------------------------------------------------------------
    #[test]
    fn release_false_if_already_idle() {
        let mut pool = default_pool();
        let id = pool.acquire("peer-k", 0).expect("acquire");
        assert!(pool.release(id, 1)); // first release succeeds
        assert!(!pool.release(id, 2)); // second release on Idle fails
    }

    // -----------------------------------------------------------------------
    // 12. close sets Closing state
    // -----------------------------------------------------------------------
    #[test]
    fn close_sets_closing_state() {
        let mut pool = default_pool();
        let id = pool.acquire("peer-l", 0).expect("acquire");
        assert!(pool.close(id));
        let conn = pool.connections.get(&id).expect("conn exists");
        assert_eq!(conn.state, PoolConnectionState::Closing);
    }

    // -----------------------------------------------------------------------
    // 13. close returns false if not found
    // -----------------------------------------------------------------------
    #[test]
    fn close_false_if_not_found() {
        let mut pool = default_pool();
        assert!(!pool.close(9999));
    }

    // -----------------------------------------------------------------------
    // 14. evict_idle removes old idle connections
    // -----------------------------------------------------------------------
    #[test]
    fn evict_idle_removes_old_idle() {
        let mut pool = PeerConnectionPool::new(PeerPoolConfig {
            max_per_peer: 4,
            idle_timeout_ticks: 10,
            max_total: 100,
        });
        let id = pool.acquire("peer-m", 0).expect("acquire");
        pool.release(id, 0);
        // Advance tick beyond idle_timeout_ticks.
        pool.evict_idle(11);
        assert!(pool.connections.is_empty());
    }

    // -----------------------------------------------------------------------
    // 15. evict_idle keeps recent idle connections
    // -----------------------------------------------------------------------
    #[test]
    fn evict_idle_keeps_recent_idle() {
        let mut pool = PeerConnectionPool::new(PeerPoolConfig {
            max_per_peer: 4,
            idle_timeout_ticks: 10,
            max_total: 100,
        });
        let id = pool.acquire("peer-n", 0).expect("acquire");
        pool.release(id, 5);
        // Only 4 ticks elapsed — below the threshold.
        pool.evict_idle(9);
        assert_eq!(pool.connections.len(), 1);
    }

    // -----------------------------------------------------------------------
    // 16. evict_idle removes Closing connections
    // -----------------------------------------------------------------------
    #[test]
    fn evict_idle_removes_closing_connections() {
        let mut pool = default_pool();
        let id = pool.acquire("peer-o", 0).expect("acquire");
        pool.close(id);
        pool.evict_idle(0); // tick does not matter for Closing
        assert!(pool.connections.is_empty());
    }

    // -----------------------------------------------------------------------
    // 17. evict_idle increments total_evicted
    // -----------------------------------------------------------------------
    #[test]
    fn evict_idle_increments_total_evicted() {
        let mut pool = PeerConnectionPool::new(PeerPoolConfig {
            max_per_peer: 4,
            idle_timeout_ticks: 10,
            max_total: 100,
        });
        let a = pool.acquire("peer-p", 0).expect("a");
        let b = pool.acquire("peer-p", 0).expect("b");
        pool.release(a, 0);
        pool.release(b, 0);
        pool.evict_idle(100);
        assert_eq!(pool.total_evicted, 2);
    }

    // -----------------------------------------------------------------------
    // 18. connections_for_peer filters correctly
    // -----------------------------------------------------------------------
    #[test]
    fn connections_for_peer_filters_correctly() {
        let mut pool = default_pool();
        pool.acquire("peer-q", 0).expect("q1");
        pool.acquire("peer-q", 0).expect("q2");
        pool.acquire("peer-r", 0).expect("r1");

        let q_conns = pool.connections_for_peer("peer-q");
        assert_eq!(q_conns.len(), 2);
        for c in &q_conns {
            assert_eq!(c.peer_id, "peer-q");
        }
    }

    // -----------------------------------------------------------------------
    // 19. connections_for_peer sorted by conn_id
    // -----------------------------------------------------------------------
    #[test]
    fn connections_for_peer_sorted_by_conn_id() {
        let mut pool = default_pool();
        pool.acquire("peer-s", 0).expect("s1");
        pool.acquire("peer-t", 0).expect("t1"); // interleave other peer
        pool.acquire("peer-s", 0).expect("s2");

        let s_conns = pool.connections_for_peer("peer-s");
        assert_eq!(s_conns.len(), 2);
        assert!(s_conns[0].conn_id < s_conns[1].conn_id);
    }

    // -----------------------------------------------------------------------
    // 20. stats total_connections correct
    // -----------------------------------------------------------------------
    #[test]
    fn stats_total_connections_correct() {
        let mut pool = default_pool();
        pool.acquire("peer-u", 0).expect("u1");
        pool.acquire("peer-u", 0).expect("u2");
        pool.acquire("peer-v", 0).expect("v1");
        assert_eq!(pool.stats().total_connections, 3);
    }

    // -----------------------------------------------------------------------
    // 21. stats idle/in_use counts
    // -----------------------------------------------------------------------
    #[test]
    fn stats_idle_in_use_counts() {
        let mut pool = default_pool();
        let id1 = pool.acquire("peer-w", 0).expect("w1");
        let _id2 = pool.acquire("peer-w", 0).expect("w2");
        pool.release(id1, 1);

        let s = pool.stats();
        assert_eq!(s.idle_connections, 1);
        assert_eq!(s.in_use_connections, 1);
        assert_eq!(s.total_connections, 2);
        assert_eq!(s.total_peers, 1);
    }

    // -----------------------------------------------------------------------
    // 22. stats total_acquired/released/evicted
    // -----------------------------------------------------------------------
    #[test]
    fn stats_total_acquired_released_evicted() {
        let mut pool = PeerConnectionPool::new(PeerPoolConfig {
            max_per_peer: 4,
            idle_timeout_ticks: 5,
            max_total: 100,
        });

        let a = pool.acquire("peer-x", 0).expect("a");
        let b = pool.acquire("peer-x", 0).expect("b");
        pool.release(a, 1);
        pool.release(b, 1);
        pool.evict_idle(10);

        let s = pool.stats();
        assert_eq!(s.total_acquired, 2);
        assert_eq!(s.total_released, 2);
        assert_eq!(s.total_evicted, 2);
    }

    // -----------------------------------------------------------------------
    // Bonus: idle boundary — exactly at timeout tick is evicted
    // -----------------------------------------------------------------------
    #[test]
    fn evict_idle_boundary_exactly_at_timeout() {
        let mut pool = PeerConnectionPool::new(PeerPoolConfig {
            max_per_peer: 4,
            idle_timeout_ticks: 10,
            max_total: 100,
        });
        let id = pool.acquire("peer-y", 0).expect("acquire");
        pool.release(id, 0);
        // tick - last_used = 10 == idle_timeout_ticks → should evict.
        pool.evict_idle(10);
        assert!(pool.connections.is_empty());
    }

    // -----------------------------------------------------------------------
    // Bonus: InUse connections are never evicted
    // -----------------------------------------------------------------------
    #[test]
    fn evict_idle_does_not_remove_in_use() {
        let mut pool = PeerConnectionPool::new(PeerPoolConfig {
            max_per_peer: 4,
            idle_timeout_ticks: 1,
            max_total: 100,
        });
        pool.acquire("peer-z", 0).expect("acquire"); // left InUse
        pool.evict_idle(1000);
        assert_eq!(pool.connections.len(), 1);
    }
}
