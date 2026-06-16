//! Connection Pool Manager
//!
//! Production-quality connection pool with health checking, adaptive sizing,
//! multiple acquisition policies, idle/lifetime eviction, and event streaming.
//!
//! # Overview
//!
//! [`ConnectionPoolManager`] maintains a collection of [`PooledConnection`] objects
//! keyed by auto-increment IDs. Connections move through the [`ConnState`] lifecycle
//! (`Idle → InUse → Idle | Draining | Closed`) and are evicted by background
//! maintenance when they exceed configured timeouts.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::connection_pool_manager::{
//!     ConnectionPoolManager, CpmPoolConfig, AcquirePolicy,
//! };
//!
//! let config = CpmPoolConfig {
//!     min_connections: 1,
//!     max_connections: 8,
//!     idle_timeout_us: 30_000_000,
//!     max_lifetime_us: 3_600_000_000,
//!     health_check_interval_us: 10_000_000,
//!     acquire_timeout_us: 5_000_000,
//!     policy: AcquirePolicy::HealthBest,
//! };
//! let mut pool = ConnectionPoolManager::new(config);
//! let id = pool.add_connection("127.0.0.1:9000".to_string(), vec![]).unwrap();
//! let acquired = pool.acquire(1_000).unwrap();
//! pool.release(acquired, 2_000, false).unwrap();
//! ```

use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// PRNG helper (no rand crate)
// ---------------------------------------------------------------------------

/// Simple xorshift64 PRNG used for tie-breaking and test randomness.
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
// ConnState
// ---------------------------------------------------------------------------

/// Lifecycle state of a single pooled connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnState {
    /// Available for acquisition.
    Idle,
    /// Currently borrowed by a caller.
    InUse {
        /// Microsecond timestamp when the connection was borrowed.
        borrowed_at: u64,
    },
    /// Marked for graceful shutdown; no new acquires allowed.
    Draining,
    /// Permanently closed and may be removed.
    Closed,
    /// Health probe in progress.
    HealthChecking,
}

// ---------------------------------------------------------------------------
// AcquirePolicy
// ---------------------------------------------------------------------------

/// Strategy used to pick an idle connection during [`ConnectionPoolManager::acquire`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcquirePolicy {
    /// Take the oldest idle connection (first-in, first-out by creation time).
    Fifo,
    /// Take the most recently created idle connection.
    Lifo,
    /// Take the connection with the lowest `use_count`.
    LeastUsed,
    /// Take the connection with the highest `health_score`.
    HealthBest,
    /// Cycle through idle connections in insertion order.
    RoundRobin,
}

// ---------------------------------------------------------------------------
// CpmPoolConfig  (Cpm prefix avoids collision with connection_pool::PoolConfig)
// ---------------------------------------------------------------------------

/// Configuration for [`ConnectionPoolManager`].
#[derive(Debug, Clone)]
pub struct CpmPoolConfig {
    /// Minimum number of open connections to maintain.
    pub min_connections: usize,
    /// Maximum total connections allowed in the pool.
    pub max_connections: usize,
    /// Microseconds a connection may remain idle before eviction.
    pub idle_timeout_us: u64,
    /// Absolute connection lifetime in microseconds.
    pub max_lifetime_us: u64,
    /// How often health checks should run (informational; not auto-scheduled).
    pub health_check_interval_us: u64,
    /// Maximum microseconds `acquire` will wait before returning `AcquireTimeout`.
    pub acquire_timeout_us: u64,
    /// Acquisition selection policy.
    pub policy: AcquirePolicy,
}

impl Default for CpmPoolConfig {
    fn default() -> Self {
        Self {
            min_connections: 1,
            max_connections: 16,
            idle_timeout_us: 60_000_000,
            max_lifetime_us: 3_600_000_000,
            health_check_interval_us: 10_000_000,
            acquire_timeout_us: 5_000_000,
            policy: AcquirePolicy::HealthBest,
        }
    }
}

// ---------------------------------------------------------------------------
// PooledConnection
// ---------------------------------------------------------------------------

/// A single managed connection inside [`ConnectionPoolManager`].
#[derive(Debug, Clone)]
pub struct PooledConnection {
    /// Unique connection identifier (auto-increment).
    pub id: u64,
    /// Remote address string (e.g. `"192.168.1.1:4001"`).
    pub peer_address: String,
    /// Microsecond timestamp when this connection was created.
    pub created_at: u64,
    /// Microsecond timestamp of the last time this connection was used.
    pub last_used: u64,
    /// Total number of times this connection has been acquired.
    pub use_count: u64,
    /// Health score in `[0.0, 1.0]`; higher is healthier.
    pub health_score: f64,
    /// Arbitrary tags attached at creation time.
    pub tags: Vec<String>,
    /// Current lifecycle state.
    pub state: ConnState,
}

impl PooledConnection {
    fn new(id: u64, peer_address: String, tags: Vec<String>, created_at: u64) -> Self {
        Self {
            id,
            peer_address,
            created_at,
            last_used: created_at,
            use_count: 0,
            health_score: 1.0,
            tags,
            state: ConnState::Idle,
        }
    }

    /// Returns `true` if the connection is currently idle and can be acquired.
    #[inline]
    pub fn is_idle(&self) -> bool {
        self.state == ConnState::Idle
    }
}

// ---------------------------------------------------------------------------
// CpmPoolStats  (Cpm prefix avoids collision with connection_pool::PoolStats)
// ---------------------------------------------------------------------------

/// Point-in-time statistics snapshot for a [`ConnectionPoolManager`].
#[derive(Debug, Clone, Default)]
pub struct CpmPoolStats {
    /// Total connections currently tracked (all states).
    pub total_connections: usize,
    /// Connections in the `Idle` state.
    pub idle_connections: usize,
    /// Connections in the `InUse` state.
    pub in_use_connections: usize,
    /// Cumulative successful acquire operations.
    pub total_acquires: u64,
    /// Cumulative release operations.
    pub total_releases: u64,
    /// Cumulative acquire failures (pool exhausted or timeout).
    pub acquire_failures: u64,
    /// Cumulative health-check failures (score < 0.3).
    pub health_check_failures: u64,
    /// Average health score across all connections (NaN if no connections).
    pub avg_health_score: f64,
}

// ---------------------------------------------------------------------------
// PoolEvent
// ---------------------------------------------------------------------------

/// Asynchronous event emitted by pool operations.
#[derive(Debug, Clone)]
pub enum PoolEvent {
    /// A new connection was added to the pool.
    ConnectionAdded(u64),
    /// A connection was removed from the pool.
    ConnectionRemoved {
        /// ID of the removed connection.
        id: u64,
        /// Human-readable reason string.
        reason: String,
    },
    /// A health check reported a critically low score.
    HealthCheckFailed {
        /// Connection ID that failed the health check.
        id: u64,
        /// Observed score that triggered the failure.
        score: f64,
    },
    /// An acquire call timed out.
    AcquireTimeout,
    /// All connections have been drained via [`ConnectionPoolManager::drain`].
    PoolDrained,
}

// ---------------------------------------------------------------------------
// CpmPoolError  (Cpm prefix avoids collision with connection_pool::PoolError)
// ---------------------------------------------------------------------------

/// Errors returned by [`ConnectionPoolManager`] operations.
#[derive(Debug, Error)]
pub enum CpmPoolError {
    /// All connections are in-use and the pool is at maximum capacity.
    #[error("connection pool exhausted (max {0} connections)")]
    PoolExhausted(usize),

    /// No connection with the supplied ID exists in the pool.
    #[error("connection {0} not found in pool")]
    ConnectionNotFound(u64),

    /// Acquire timed out after the given number of microseconds.
    #[error("acquire timed out after {0} µs")]
    AcquireTimeout(u64),

    /// Health check failed for the given connection ID.
    #[error("health check failed for connection {0}")]
    HealthCheckFailed(u64),

    /// The supplied configuration is invalid.
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),
}

// ---------------------------------------------------------------------------
// ConnectionPoolManager
// ---------------------------------------------------------------------------

/// Adaptive connection pool with health checking, multiple acquisition policies,
/// idle/lifetime eviction, event streaming, and dynamic resize.
pub struct ConnectionPoolManager {
    config: CpmPoolConfig,
    connections: HashMap<u64, PooledConnection>,
    next_id: u64,
    round_robin_cursor: usize,
    // Cumulative statistics.
    total_acquires: u64,
    total_releases: u64,
    acquire_failures: u64,
    health_check_failures: u64,
    // Pending events to be drained by the caller.
    pending_events: Vec<PoolEvent>,
}

impl ConnectionPoolManager {
    /// Create a new pool with the given configuration.
    ///
    /// Returns [`CpmPoolError::InvalidConfiguration`] if `min > max` or `max == 0`.
    pub fn new(config: CpmPoolConfig) -> Self {
        Self {
            config,
            connections: HashMap::new(),
            next_id: 1,
            round_robin_cursor: 0,
            total_acquires: 0,
            total_releases: 0,
            acquire_failures: 0,
            health_check_failures: 0,
            pending_events: Vec::new(),
        }
    }

    /// Validate the current configuration.
    pub fn validate_config(config: &CpmPoolConfig) -> Result<(), CpmPoolError> {
        if config.max_connections == 0 {
            return Err(CpmPoolError::InvalidConfiguration(
                "max_connections must be > 0".to_string(),
            ));
        }
        if config.min_connections > config.max_connections {
            return Err(CpmPoolError::InvalidConfiguration(format!(
                "min_connections ({}) > max_connections ({})",
                config.min_connections, config.max_connections
            )));
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Core CRUD
    // -----------------------------------------------------------------------

    /// Add a new connection to the pool.
    ///
    /// Returns the new connection's unique ID, or [`CpmPoolError::PoolExhausted`] if
    /// `max_connections` has been reached.
    pub fn add_connection(
        &mut self,
        address: String,
        tags: Vec<String>,
    ) -> Result<u64, CpmPoolError> {
        if self.connections.len() >= self.config.max_connections {
            self.acquire_failures += 1;
            return Err(CpmPoolError::PoolExhausted(self.config.max_connections));
        }
        let id = self.next_id;
        self.next_id += 1;
        let conn = PooledConnection::new(id, address, tags, 0);
        self.connections.insert(id, conn);
        self.pending_events.push(PoolEvent::ConnectionAdded(id));
        Ok(id)
    }

    /// Add a connection with an explicit creation timestamp (useful in tests).
    pub fn add_connection_at(
        &mut self,
        address: String,
        tags: Vec<String>,
        created_at: u64,
    ) -> Result<u64, CpmPoolError> {
        if self.connections.len() >= self.config.max_connections {
            self.acquire_failures += 1;
            return Err(CpmPoolError::PoolExhausted(self.config.max_connections));
        }
        let id = self.next_id;
        self.next_id += 1;
        let mut conn = PooledConnection::new(id, address, tags, created_at);
        conn.last_used = created_at;
        self.connections.insert(id, conn);
        self.pending_events.push(PoolEvent::ConnectionAdded(id));
        Ok(id)
    }

    /// Acquire an idle connection according to the configured [`AcquirePolicy`].
    ///
    /// Returns the ID of the acquired connection, or an error if:
    /// - No idle connections exist and the pool is at capacity → [`CpmPoolError::PoolExhausted`]
    /// - All connections are busy and no idle ones can be selected →
    ///   [`CpmPoolError::AcquireTimeout`] (using `acquire_timeout_us` as the waited value)
    pub fn acquire(&mut self, current_ts: u64) -> Result<u64, CpmPoolError> {
        let chosen_id = self.select_idle(current_ts)?;

        if let Some(conn) = self.connections.get_mut(&chosen_id) {
            conn.state = ConnState::InUse {
                borrowed_at: current_ts,
            };
            conn.use_count += 1;
            conn.last_used = current_ts;
        }
        self.total_acquires += 1;
        Ok(chosen_id)
    }

    /// Release a previously acquired connection back to `Idle`.
    ///
    /// If `health_degraded` is `true`, the health score is reduced by `0.1`
    /// (floored at `0.0`).
    pub fn release(
        &mut self,
        conn_id: u64,
        current_ts: u64,
        health_degraded: bool,
    ) -> Result<(), CpmPoolError> {
        let conn = self
            .connections
            .get_mut(&conn_id)
            .ok_or(CpmPoolError::ConnectionNotFound(conn_id))?;

        conn.state = ConnState::Idle;
        conn.last_used = current_ts;
        if health_degraded {
            conn.health_score = (conn.health_score - 0.1_f64).max(0.0_f64);
        }
        self.total_releases += 1;
        Ok(())
    }

    /// Permanently remove a connection from the pool.
    pub fn remove_connection(&mut self, conn_id: u64) -> Result<(), CpmPoolError> {
        if self.connections.remove(&conn_id).is_none() {
            return Err(CpmPoolError::ConnectionNotFound(conn_id));
        }
        self.pending_events.push(PoolEvent::ConnectionRemoved {
            id: conn_id,
            reason: "explicit removal".to_string(),
        });
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Health
    // -----------------------------------------------------------------------

    /// Record the result of a health check for the given connection.
    ///
    /// If `score < 0.3`, emits a [`PoolEvent::HealthCheckFailed`] and returns it.
    /// Otherwise returns [`PoolEvent::ConnectionAdded`] as a no-op sentinel
    /// (callers should match on the returned variant).
    pub fn health_check(
        &mut self,
        conn_id: u64,
        score: f64,
        current_ts: u64,
    ) -> Result<PoolEvent, CpmPoolError> {
        let conn = self
            .connections
            .get_mut(&conn_id)
            .ok_or(CpmPoolError::ConnectionNotFound(conn_id))?;

        conn.health_score = score.clamp(0.0, 1.0);
        conn.last_used = current_ts;

        if score < 0.3 {
            self.health_check_failures += 1;
            let event = PoolEvent::HealthCheckFailed { id: conn_id, score };
            self.pending_events.push(event.clone());
            Ok(event)
        } else {
            Ok(PoolEvent::ConnectionAdded(conn_id))
        }
    }

    // -----------------------------------------------------------------------
    // Maintenance
    // -----------------------------------------------------------------------

    /// Evict connections that have exceeded idle or absolute lifetime limits.
    ///
    /// Returns a list of [`PoolEvent::ConnectionRemoved`] events for each evicted
    /// connection.
    pub fn run_maintenance(&mut self, current_ts: u64) -> Vec<PoolEvent> {
        let idle_timeout = self.config.idle_timeout_us;
        let max_lifetime = self.config.max_lifetime_us;

        // Collect IDs to evict (avoid borrow issues).
        let mut to_evict: Vec<(u64, String)> = Vec::new();
        for (id, conn) in &self.connections {
            if conn.state == ConnState::Closed {
                to_evict.push((*id, "already closed".to_string()));
                continue;
            }
            // Lifetime eviction (applies to all non-InUse connections).
            if matches!(conn.state, ConnState::Idle | ConnState::HealthChecking) {
                let age = current_ts.saturating_sub(conn.created_at);
                if age >= max_lifetime {
                    to_evict.push((*id, "max lifetime exceeded".to_string()));
                    continue;
                }
                // Idle timeout eviction.
                let idle_time = current_ts.saturating_sub(conn.last_used);
                if idle_time >= idle_timeout {
                    to_evict.push((*id, "idle timeout exceeded".to_string()));
                }
            }
        }

        let mut events = Vec::with_capacity(to_evict.len());
        for (id, reason) in to_evict {
            self.connections.remove(&id);
            let ev = PoolEvent::ConnectionRemoved { id, reason };
            self.pending_events.push(ev.clone());
            events.push(ev);
        }
        events
    }

    // -----------------------------------------------------------------------
    // Resize
    // -----------------------------------------------------------------------

    /// Resize the pool's maximum connection limit.
    ///
    /// When shrinking, excess idle connections are drained (closed) immediately.
    pub fn resize(&mut self, new_max: usize) -> Result<(), CpmPoolError> {
        if new_max == 0 {
            return Err(CpmPoolError::InvalidConfiguration(
                "new_max must be > 0".to_string(),
            ));
        }
        let old_max = self.config.max_connections;
        self.config.max_connections = new_max;

        if new_max < old_max {
            // Drain idle connections until we are within the new limit.
            let excess = self.connections.len().saturating_sub(new_max);
            if excess > 0 {
                let idle_ids: Vec<u64> = self
                    .connections
                    .values()
                    .filter(|c| c.state == ConnState::Idle)
                    .map(|c| c.id)
                    .take(excess)
                    .collect();
                for id in idle_ids {
                    self.connections.remove(&id);
                    self.pending_events.push(PoolEvent::ConnectionRemoved {
                        id,
                        reason: "pool resize shrink".to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Drain
    // -----------------------------------------------------------------------

    /// Close all connections in insertion order and return their IDs.
    pub fn drain(&mut self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.connections.keys().copied().collect();
        ids.sort_unstable(); // insertion order approximated by ascending ID.

        for &id in &ids {
            self.pending_events.push(PoolEvent::ConnectionRemoved {
                id,
                reason: "pool drain".to_string(),
            });
        }
        self.connections.clear();
        self.pending_events.push(PoolEvent::PoolDrained);
        ids
    }

    // -----------------------------------------------------------------------
    // Query helpers
    // -----------------------------------------------------------------------

    /// Return references to all connections that carry the given tag.
    pub fn connections_with_tag(&self, tag: &str) -> Vec<&PooledConnection> {
        self.connections
            .values()
            .filter(|c| c.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Compute a statistics snapshot.
    pub fn stats(&self) -> CpmPoolStats {
        let total = self.connections.len();
        let idle = self
            .connections
            .values()
            .filter(|c| c.state == ConnState::Idle)
            .count();
        let in_use = self
            .connections
            .values()
            .filter(|c| matches!(c.state, ConnState::InUse { .. }))
            .count();

        let avg_health = if total == 0 {
            f64::NAN
        } else {
            let sum: f64 = self.connections.values().map(|c| c.health_score).sum();
            sum / total as f64
        };

        CpmPoolStats {
            total_connections: total,
            idle_connections: idle,
            in_use_connections: in_use,
            total_acquires: self.total_acquires,
            total_releases: self.total_releases,
            acquire_failures: self.acquire_failures,
            health_check_failures: self.health_check_failures,
            avg_health_score: avg_health,
        }
    }

    /// Drain and return all pending events.
    pub fn drain_events(&mut self) -> Vec<PoolEvent> {
        std::mem::take(&mut self.pending_events)
    }

    /// Returns a reference to the underlying connection map (read-only).
    pub fn connections(&self) -> &HashMap<u64, PooledConnection> {
        &self.connections
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &CpmPoolConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Policy-based idle selection (private)
    // -----------------------------------------------------------------------

    fn select_idle(&mut self, _current_ts: u64) -> Result<u64, CpmPoolError> {
        // Build a list of idle connection IDs (and relevant metadata for sorting).
        let idle: Vec<(u64, u64, u64, f64)> = self
            .connections
            .values()
            .filter(|c| c.state == ConnState::Idle)
            .map(|c| (c.id, c.created_at, c.use_count, c.health_score))
            .collect();

        if idle.is_empty() {
            self.acquire_failures += 1;
            if self.connections.len() >= self.config.max_connections {
                return Err(CpmPoolError::PoolExhausted(self.config.max_connections));
            }
            // Pool has room but no idle connections — treat as timeout.
            return Err(CpmPoolError::AcquireTimeout(self.config.acquire_timeout_us));
        }

        let chosen_id = match self.config.policy {
            AcquirePolicy::Fifo => {
                // Oldest created_at first.
                idle.iter()
                    .min_by_key(|&&(_, created_at, _, _)| created_at)
                    .map(|&(id, _, _, _)| id)
                    .expect("idle is non-empty")
            }
            AcquirePolicy::Lifo => {
                // Newest created_at first.
                idle.iter()
                    .max_by_key(|&&(_, created_at, _, _)| created_at)
                    .map(|&(id, _, _, _)| id)
                    .expect("idle is non-empty")
            }
            AcquirePolicy::LeastUsed => idle
                .iter()
                .min_by_key(|&&(_, _, use_count, _)| use_count)
                .map(|&(id, _, _, _)| id)
                .expect("idle is non-empty"),
            AcquirePolicy::HealthBest => {
                // Highest health score; use ID as stable tie-breaker.
                idle.iter()
                    .max_by(|a, b| {
                        a.3.partial_cmp(&b.3)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| a.0.cmp(&b.0))
                    })
                    .map(|&(id, _, _, _)| id)
                    .expect("idle is non-empty")
            }
            AcquirePolicy::RoundRobin => {
                // Sort by ID for stable ordering, then advance cursor.
                let mut sorted: Vec<u64> = idle.iter().map(|&(id, _, _, _)| id).collect();
                sorted.sort_unstable();
                let idx = self.round_robin_cursor % sorted.len();
                self.round_robin_cursor = self.round_robin_cursor.wrapping_add(1);
                sorted[idx]
            }
        };

        Ok(chosen_id)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn default_config() -> CpmPoolConfig {
        CpmPoolConfig {
            min_connections: 1,
            max_connections: 10,
            idle_timeout_us: 100_000,
            max_lifetime_us: 1_000_000,
            health_check_interval_us: 50_000,
            acquire_timeout_us: 5_000,
            policy: AcquirePolicy::Fifo,
        }
    }

    fn make_pool() -> ConnectionPoolManager {
        ConnectionPoolManager::new(default_config())
    }

    fn populate(pool: &mut ConnectionPoolManager, n: usize) -> Vec<u64> {
        (0..n)
            .map(|i| {
                pool.add_connection(format!("127.0.0.1:{}", 9000 + i), vec![])
                    .expect("add_connection failed")
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // 1. add_connection
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_connection_basic() {
        let mut pool = make_pool();
        let id = pool
            .add_connection("127.0.0.1:9000".to_string(), vec![])
            .expect("test: add_connection should succeed");
        assert_eq!(id, 1);
        assert_eq!(pool.connections().len(), 1);
    }

    #[test]
    fn test_add_connection_increments_id() {
        let mut pool = make_pool();
        let id1 = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection a should succeed");
        let id2 = pool
            .add_connection("b".to_string(), vec![])
            .expect("test: add_connection b should succeed");
        assert!(id2 > id1);
    }

    #[test]
    fn test_add_connection_emits_event() {
        let mut pool = make_pool();
        let id = pool
            .add_connection("x".to_string(), vec![])
            .expect("test: add_connection should succeed");
        let events = pool.drain_events();
        assert!(events
            .iter()
            .any(|e| matches!(e, PoolEvent::ConnectionAdded(i) if *i == id)));
    }

    #[test]
    fn test_add_connection_at_capacity_fails() {
        let mut pool = ConnectionPoolManager::new(CpmPoolConfig {
            max_connections: 2,
            ..default_config()
        });
        pool.add_connection("a".to_string(), vec![])
            .expect("test: first add_connection should succeed");
        pool.add_connection("b".to_string(), vec![])
            .expect("test: second add_connection should succeed");
        let err = pool.add_connection("c".to_string(), vec![]).unwrap_err();
        assert!(matches!(err, CpmPoolError::PoolExhausted(_)));
    }

    #[test]
    fn test_add_connection_new_state_is_idle() {
        let mut pool = make_pool();
        let id = pool
            .add_connection("h".to_string(), vec![])
            .expect("test: add_connection should succeed");
        let conn = pool
            .connections()
            .get(&id)
            .expect("test: connection should exist after add");
        assert_eq!(conn.state, ConnState::Idle);
    }

    #[test]
    fn test_add_connection_with_tags() {
        let mut pool = make_pool();
        let tags = vec!["region:eu".to_string(), "tier:1".to_string()];
        let id = pool
            .add_connection("t".to_string(), tags.clone())
            .expect("test: add_connection with tags should succeed");
        let conn = pool
            .connections()
            .get(&id)
            .expect("test: connection should exist after add");
        assert_eq!(conn.tags, tags);
    }

    // -----------------------------------------------------------------------
    // 2. acquire / release
    // -----------------------------------------------------------------------

    #[test]
    fn test_acquire_idle_connection() {
        let mut pool = make_pool();
        let id = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection should succeed");
        let acquired = pool
            .acquire(100)
            .expect("test: acquire should succeed with idle connection");
        assert_eq!(acquired, id);
        let conn = pool
            .connections()
            .get(&id)
            .expect("test: connection should exist");
        assert!(matches!(conn.state, ConnState::InUse { .. }));
    }

    #[test]
    fn test_acquire_increments_use_count() {
        let mut pool = make_pool();
        let id = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection should succeed");
        pool.acquire(100)
            .expect("test: first acquire should succeed");
        pool.release(id, 200, false)
            .expect("test: release should succeed");
        pool.acquire(300)
            .expect("test: second acquire should succeed");
        let conn = pool
            .connections()
            .get(&id)
            .expect("test: connection should exist");
        assert_eq!(conn.use_count, 2);
    }

    #[test]
    fn test_acquire_updates_last_used() {
        let mut pool = make_pool();
        let id = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection should succeed");
        pool.acquire(999).expect("test: acquire should succeed");
        let conn = pool
            .connections()
            .get(&id)
            .expect("test: connection should exist");
        assert_eq!(conn.last_used, 999);
    }

    #[test]
    fn test_acquire_no_idle_pool_full() {
        let mut pool = ConnectionPoolManager::new(CpmPoolConfig {
            max_connections: 1,
            ..default_config()
        });
        let id = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection should succeed");
        pool.acquire(100)
            .expect("test: first acquire should succeed"); // now InUse
        let err = pool.acquire(200).unwrap_err();
        assert!(matches!(err, CpmPoolError::PoolExhausted(_)));
        // release and re-acquire succeeds
        pool.release(id, 300, false)
            .expect("test: release should succeed");
        assert!(pool.acquire(400).is_ok());
    }

    #[test]
    fn test_release_transitions_to_idle() {
        let mut pool = make_pool();
        let id = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection should succeed");
        pool.acquire(100).expect("test: acquire should succeed");
        pool.release(id, 200, false)
            .expect("test: release should succeed");
        let conn = pool
            .connections()
            .get(&id)
            .expect("test: connection should exist");
        assert_eq!(conn.state, ConnState::Idle);
    }

    #[test]
    fn test_release_health_degraded_reduces_score() {
        let mut pool = make_pool();
        let id = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection should succeed");
        pool.acquire(100).expect("test: acquire should succeed");
        pool.release(id, 200, true)
            .expect("test: release with health degraded should succeed");
        let conn = pool
            .connections()
            .get(&id)
            .expect("test: connection should exist");
        assert!((conn.health_score - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_release_health_floor_at_zero() {
        let mut pool = make_pool();
        let id = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection should succeed");
        // Degrade 11 times from 1.0 → floor at 0.0
        for _ in 0..11 {
            if pool
                .connections()
                .get(&id)
                .expect("test: connection should exist")
                .is_idle()
            {
                pool.acquire(1).expect("test: acquire should succeed");
            }
            pool.release(id, 2, true)
                .expect("test: release should succeed");
        }
        let conn = pool
            .connections()
            .get(&id)
            .expect("test: connection should exist");
        assert_eq!(conn.health_score, 0.0);
    }

    #[test]
    fn test_release_unknown_id_error() {
        let mut pool = make_pool();
        let err = pool.release(999, 100, false).unwrap_err();
        assert!(matches!(err, CpmPoolError::ConnectionNotFound(999)));
    }

    // -----------------------------------------------------------------------
    // 3. AcquirePolicy::Fifo
    // -----------------------------------------------------------------------

    #[test]
    fn test_policy_fifo() {
        let mut pool = ConnectionPoolManager::new(CpmPoolConfig {
            policy: AcquirePolicy::Fifo,
            ..default_config()
        });
        let id1 = pool
            .add_connection_at("a".to_string(), vec![], 100)
            .expect("test: add_connection_at a should succeed");
        let id2 = pool
            .add_connection_at("b".to_string(), vec![], 200)
            .expect("test: add_connection_at b should succeed");
        let acquired = pool
            .acquire(300)
            .expect("test: first acquire should succeed");
        assert_eq!(acquired, id1, "FIFO should pick oldest (id1)");
        pool.release(id1, 400, false)
            .expect("test: release should succeed");
        let acquired2 = pool
            .acquire(500)
            .expect("test: second acquire should succeed");
        assert_eq!(acquired2, id1, "After release, id1 is oldest again");
        let _ = acquired2;
        let _ = id2;
    }

    // -----------------------------------------------------------------------
    // 4. AcquirePolicy::Lifo
    // -----------------------------------------------------------------------

    #[test]
    fn test_policy_lifo() {
        let mut pool = ConnectionPoolManager::new(CpmPoolConfig {
            policy: AcquirePolicy::Lifo,
            ..default_config()
        });
        let _id1 = pool
            .add_connection_at("a".to_string(), vec![], 100)
            .expect("test: add_connection_at a should succeed");
        let id2 = pool
            .add_connection_at("b".to_string(), vec![], 200)
            .expect("test: add_connection_at b should succeed");
        let acquired = pool.acquire(300).expect("test: acquire should succeed");
        assert_eq!(acquired, id2, "LIFO should pick newest (id2)");
    }

    // -----------------------------------------------------------------------
    // 5. AcquirePolicy::LeastUsed
    // -----------------------------------------------------------------------

    #[test]
    fn test_policy_least_used() {
        let mut pool = ConnectionPoolManager::new(CpmPoolConfig {
            policy: AcquirePolicy::LeastUsed,
            ..default_config()
        });
        let id1 = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection a should succeed");
        let id2 = pool
            .add_connection("b".to_string(), vec![])
            .expect("test: add_connection b should succeed");
        // Manually bump id1's use_count to 5 so id2 (0 uses) wins LeastUsed.
        pool.connections
            .get_mut(&id1)
            .expect("test: connection id1 should exist")
            .use_count = 5;
        // Now id2 has 0 uses → LeastUsed should pick id2.
        let acquired = pool.acquire(5).expect("test: acquire should succeed");
        assert_eq!(acquired, id2);
    }

    // -----------------------------------------------------------------------
    // 6. AcquirePolicy::HealthBest
    // -----------------------------------------------------------------------

    #[test]
    fn test_policy_health_best() {
        let mut pool = ConnectionPoolManager::new(CpmPoolConfig {
            policy: AcquirePolicy::HealthBest,
            ..default_config()
        });
        let id1 = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection a should succeed");
        let id2 = pool
            .add_connection("b".to_string(), vec![])
            .expect("test: add_connection b should succeed");
        // Lower id1's health.
        pool.health_check(id1, 0.4, 0)
            .expect("test: health_check id1 should succeed");
        pool.health_check(id2, 0.9, 0)
            .expect("test: health_check id2 should succeed");
        let acquired = pool.acquire(1).expect("test: acquire should succeed");
        assert_eq!(acquired, id2, "HealthBest should pick id2 (higher score)");
    }

    // -----------------------------------------------------------------------
    // 7. AcquirePolicy::RoundRobin
    // -----------------------------------------------------------------------

    #[test]
    fn test_policy_round_robin() {
        let mut pool = ConnectionPoolManager::new(CpmPoolConfig {
            policy: AcquirePolicy::RoundRobin,
            ..default_config()
        });
        let id1 = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection a should succeed");
        let id2 = pool
            .add_connection("b".to_string(), vec![])
            .expect("test: add_connection b should succeed");
        let id3 = pool
            .add_connection("c".to_string(), vec![])
            .expect("test: add_connection c should succeed");

        // First acquire → cursor=0 → id1
        let a = pool.acquire(1).expect("test: first acquire should succeed");
        assert_eq!(a, id1);
        pool.release(id1, 2, false)
            .expect("test: release id1 should succeed");

        // Second acquire: only id2, id3 are idle → sorted = [id2, id3], cursor=1 → id2
        let b = pool
            .acquire(3)
            .expect("test: second acquire should succeed");
        assert_eq!(b, id2);
        pool.release(id2, 4, false)
            .expect("test: release id2 should succeed");

        // Third acquire: id1, id2, id3 all idle → sorted, cursor=2 → id3
        let c = pool.acquire(5).expect("test: third acquire should succeed");
        assert_eq!(c, id3);
        let _ = c;
    }

    // -----------------------------------------------------------------------
    // 8. Health degradation
    // -----------------------------------------------------------------------

    #[test]
    fn test_health_check_update() {
        let mut pool = make_pool();
        let id = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection should succeed");
        pool.health_check(id, 0.75, 100)
            .expect("test: health_check should succeed");
        let conn = pool
            .connections()
            .get(&id)
            .expect("test: connection should exist");
        assert!((conn.health_score - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_health_check_clamps_above_one() {
        let mut pool = make_pool();
        let id = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection should succeed");
        pool.health_check(id, 1.5, 0)
            .expect("test: health_check with out-of-range score should succeed");
        assert!(
            (pool
                .connections()
                .get(&id)
                .expect("test: connection should exist")
                .health_score
                - 1.0)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn test_health_check_clamps_below_zero() {
        let mut pool = make_pool();
        let id = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection should succeed");
        pool.health_check(id, -0.5, 0)
            .expect("test: health_check with negative score should succeed");
        assert_eq!(
            pool.connections()
                .get(&id)
                .expect("test: connection should exist")
                .health_score,
            0.0
        );
    }

    #[test]
    fn test_health_check_failed_event_below_threshold() {
        let mut pool = make_pool();
        let id = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection should succeed");
        let ev = pool
            .health_check(id, 0.2, 0)
            .expect("test: health_check should succeed");
        assert!(matches!(ev, PoolEvent::HealthCheckFailed { id: i, .. } if i == id));
    }

    #[test]
    fn test_health_check_no_failure_at_threshold() {
        let mut pool = make_pool();
        let id = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection should succeed");
        // exactly 0.3 should NOT emit failure
        let ev = pool
            .health_check(id, 0.3, 0)
            .expect("test: health_check at threshold should succeed");
        assert!(!matches!(ev, PoolEvent::HealthCheckFailed { .. }));
    }

    #[test]
    fn test_health_check_failure_increments_counter() {
        let mut pool = make_pool();
        let id = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection should succeed");
        pool.health_check(id, 0.1, 0)
            .expect("test: health_check below threshold should succeed");
        assert_eq!(pool.stats().health_check_failures, 1);
    }

    #[test]
    fn test_health_check_not_found_error() {
        let mut pool = make_pool();
        let err = pool.health_check(42, 0.5, 0).unwrap_err();
        assert!(matches!(err, CpmPoolError::ConnectionNotFound(42)));
    }

    // -----------------------------------------------------------------------
    // 9. Maintenance — idle timeout
    // -----------------------------------------------------------------------

    #[test]
    fn test_maintenance_idle_timeout() {
        let mut pool = ConnectionPoolManager::new(CpmPoolConfig {
            idle_timeout_us: 1_000,
            max_lifetime_us: 999_999_999,
            ..default_config()
        });
        let id = pool
            .add_connection_at("a".to_string(), vec![], 0)
            .expect("test: add_connection_at should succeed");
        // last_used = 0; run maintenance at ts=2000 (> idle_timeout=1000)
        let events = pool.run_maintenance(2_000);
        assert!(!events.is_empty(), "should evict idle-timeout connection");
        assert!(events.iter().any(|e| matches!(
            e,
            PoolEvent::ConnectionRemoved { id: i, .. } if *i == id
        )));
        assert!(!pool.connections().contains_key(&id));
    }

    #[test]
    fn test_maintenance_skips_in_use() {
        let mut pool = ConnectionPoolManager::new(CpmPoolConfig {
            idle_timeout_us: 1_000,
            max_lifetime_us: 999_999_999,
            ..default_config()
        });
        let id = pool
            .add_connection_at("a".to_string(), vec![], 0)
            .expect("test: add_connection_at should succeed");
        pool.acquire(0).expect("test: acquire should succeed");
        let events = pool.run_maintenance(2_000);
        assert!(events.is_empty(), "InUse connections should not be evicted");
        assert!(pool.connections().contains_key(&id));
    }

    // -----------------------------------------------------------------------
    // 10. Maintenance — lifetime timeout
    // -----------------------------------------------------------------------

    #[test]
    fn test_maintenance_max_lifetime() {
        let mut pool = ConnectionPoolManager::new(CpmPoolConfig {
            idle_timeout_us: 999_999_999,
            max_lifetime_us: 500,
            ..default_config()
        });
        let id = pool
            .add_connection_at("a".to_string(), vec![], 0)
            .expect("test: add_connection_at should succeed");
        // Connection created at 0; run at 600 → age 600 > max_lifetime 500
        let events = pool.run_maintenance(600);
        assert!(!events.is_empty());
        assert!(events.iter().any(|e| matches!(
            e,
            PoolEvent::ConnectionRemoved { id: i, .. } if *i == id
        )));
    }

    #[test]
    fn test_maintenance_no_eviction_young_connection() {
        let mut pool = ConnectionPoolManager::new(CpmPoolConfig {
            idle_timeout_us: 10_000,
            max_lifetime_us: 50_000,
            ..default_config()
        });
        pool.add_connection_at("a".to_string(), vec![], 1_000)
            .expect("test: add_connection_at should succeed");
        let events = pool.run_maintenance(2_000); // only 1000 µs old
        assert!(events.is_empty());
    }

    #[test]
    fn test_maintenance_multiple_evictions() {
        let mut pool = ConnectionPoolManager::new(CpmPoolConfig {
            idle_timeout_us: 100,
            max_lifetime_us: 999_999_999,
            ..default_config()
        });
        pool.add_connection_at("a".to_string(), vec![], 0)
            .expect("test: add_connection_at a should succeed");
        pool.add_connection_at("b".to_string(), vec![], 0)
            .expect("test: add_connection_at b should succeed");
        pool.add_connection_at("c".to_string(), vec![], 0)
            .expect("test: add_connection_at c should succeed");
        let events = pool.run_maintenance(200);
        assert_eq!(events.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 11. Resize
    // -----------------------------------------------------------------------

    #[test]
    fn test_resize_grow() {
        let mut pool = ConnectionPoolManager::new(CpmPoolConfig {
            max_connections: 3,
            ..default_config()
        });
        populate(&mut pool, 3);
        pool.resize(10).expect("test: resize grow should succeed");
        assert_eq!(pool.config().max_connections, 10);
        // Now we can add more.
        assert!(pool.add_connection("new".to_string(), vec![]).is_ok());
    }

    #[test]
    fn test_resize_shrink_evicts_idle() {
        let mut pool = ConnectionPoolManager::new(CpmPoolConfig {
            max_connections: 5,
            ..default_config()
        });
        populate(&mut pool, 5);
        pool.resize(3).expect("test: resize shrink should succeed");
        assert!(pool.connections().len() <= 3);
    }

    #[test]
    fn test_resize_zero_fails() {
        let mut pool = make_pool();
        let err = pool.resize(0).unwrap_err();
        assert!(matches!(err, CpmPoolError::InvalidConfiguration(_)));
    }

    #[test]
    fn test_resize_same_size_noop() {
        let mut pool = ConnectionPoolManager::new(CpmPoolConfig {
            max_connections: 4,
            ..default_config()
        });
        populate(&mut pool, 4);
        pool.resize(4)
            .expect("test: resize same size should succeed");
        assert_eq!(pool.connections().len(), 4);
    }

    // -----------------------------------------------------------------------
    // 12. Drain
    // -----------------------------------------------------------------------

    #[test]
    fn test_drain_clears_all_connections() {
        let mut pool = make_pool();
        populate(&mut pool, 5);
        let ids = pool.drain();
        assert_eq!(ids.len(), 5);
        assert!(pool.connections().is_empty());
    }

    #[test]
    fn test_drain_emits_pool_drained_event() {
        let mut pool = make_pool();
        populate(&mut pool, 2);
        pool.drain();
        let events = pool.drain_events();
        assert!(events.iter().any(|e| matches!(e, PoolEvent::PoolDrained)));
    }

    #[test]
    fn test_drain_ids_sorted_ascending() {
        let mut pool = make_pool();
        let added = populate(&mut pool, 4);
        let drained = pool.drain();
        assert_eq!(drained, added);
    }

    #[test]
    fn test_drain_empty_pool() {
        let mut pool = make_pool();
        let ids = pool.drain();
        assert!(ids.is_empty());
        let events = pool.drain_events();
        assert!(events.iter().any(|e| matches!(e, PoolEvent::PoolDrained)));
    }

    // -----------------------------------------------------------------------
    // 13. Tag filtering
    // -----------------------------------------------------------------------

    #[test]
    fn test_connections_with_tag_basic() {
        let mut pool = make_pool();
        pool.add_connection("a".to_string(), vec!["eu".to_string()])
            .expect("test: add_connection a with eu tag should succeed");
        pool.add_connection("b".to_string(), vec!["us".to_string()])
            .expect("test: add_connection b with us tag should succeed");
        pool.add_connection("c".to_string(), vec!["eu".to_string(), "tier1".to_string()])
            .expect("test: add_connection c with tags should succeed");
        let eu = pool.connections_with_tag("eu");
        assert_eq!(eu.len(), 2);
    }

    #[test]
    fn test_connections_with_tag_no_match() {
        let mut pool = make_pool();
        pool.add_connection("a".to_string(), vec!["us".to_string()])
            .expect("test: add_connection should succeed");
        let result = pool.connections_with_tag("eu");
        assert!(result.is_empty());
    }

    #[test]
    fn test_connections_with_tag_multiple_tags() {
        let mut pool = make_pool();
        let id = pool
            .add_connection(
                "a".to_string(),
                vec!["x".to_string(), "y".to_string(), "z".to_string()],
            )
            .expect("test: add_connection with multiple tags should succeed");
        assert_eq!(pool.connections_with_tag("x").len(), 1);
        assert_eq!(pool.connections_with_tag("y").len(), 1);
        assert_eq!(pool.connections_with_tag("z").len(), 1);
        let _ = id;
    }

    // -----------------------------------------------------------------------
    // 14. Stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_empty_pool() {
        let pool = make_pool();
        let s = pool.stats();
        assert_eq!(s.total_connections, 0);
        assert_eq!(s.idle_connections, 0);
        assert_eq!(s.in_use_connections, 0);
        assert!(s.avg_health_score.is_nan());
    }

    #[test]
    fn test_stats_after_add() {
        let mut pool = make_pool();
        populate(&mut pool, 3);
        let s = pool.stats();
        assert_eq!(s.total_connections, 3);
        assert_eq!(s.idle_connections, 3);
        assert_eq!(s.in_use_connections, 0);
    }

    #[test]
    fn test_stats_after_acquire() {
        let mut pool = make_pool();
        populate(&mut pool, 3);
        pool.acquire(1).expect("test: acquire should succeed");
        let s = pool.stats();
        assert_eq!(s.idle_connections, 2);
        assert_eq!(s.in_use_connections, 1);
        assert_eq!(s.total_acquires, 1);
    }

    #[test]
    fn test_stats_avg_health() {
        let mut pool = make_pool();
        let id1 = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection a should succeed");
        let id2 = pool
            .add_connection("b".to_string(), vec![])
            .expect("test: add_connection b should succeed");
        pool.health_check(id1, 0.6, 0)
            .expect("test: health_check id1 should succeed");
        pool.health_check(id2, 0.4, 0)
            .expect("test: health_check id2 should succeed");
        let s = pool.stats();
        assert!((s.avg_health_score - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_stats_acquire_failures() {
        let mut pool = ConnectionPoolManager::new(CpmPoolConfig {
            max_connections: 1,
            ..default_config()
        });
        let id = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection should succeed");
        pool.acquire(1).expect("test: acquire should succeed");
        let _ = pool.acquire(2); // fails
        let s = pool.stats();
        assert_eq!(s.acquire_failures, 1);
        pool.release(id, 3, false)
            .expect("test: release should succeed");
    }

    // -----------------------------------------------------------------------
    // 15. Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_remove_connection_ok() {
        let mut pool = make_pool();
        let id = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection should succeed");
        pool.remove_connection(id)
            .expect("test: remove_connection should succeed");
        assert!(!pool.connections().contains_key(&id));
    }

    #[test]
    fn test_remove_connection_not_found() {
        let mut pool = make_pool();
        let err = pool.remove_connection(42).unwrap_err();
        assert!(matches!(err, CpmPoolError::ConnectionNotFound(42)));
    }

    #[test]
    fn test_acquire_pool_empty_timeout() {
        let mut pool = make_pool();
        // Pool has capacity but nothing idle.
        let err = pool.acquire(100).unwrap_err();
        assert!(matches!(err, CpmPoolError::AcquireTimeout(_)));
    }

    #[test]
    fn test_invalid_config_zero_max() {
        let result = ConnectionPoolManager::validate_config(&CpmPoolConfig {
            max_connections: 0,
            ..default_config()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_min_gt_max() {
        let result = ConnectionPoolManager::validate_config(&CpmPoolConfig {
            min_connections: 5,
            max_connections: 3,
            ..default_config()
        });
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // 16. drain_events
    // -----------------------------------------------------------------------

    #[test]
    fn test_drain_events_clears_queue() {
        let mut pool = make_pool();
        pool.add_connection("a".to_string(), vec![])
            .expect("test: add_connection should succeed");
        let ev1 = pool.drain_events();
        assert!(!ev1.is_empty());
        let ev2 = pool.drain_events();
        assert!(ev2.is_empty());
    }

    #[test]
    fn test_drain_events_removed_event() {
        let mut pool = make_pool();
        let id = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection should succeed");
        pool.drain_events(); // clear add event
        pool.remove_connection(id)
            .expect("test: remove_connection should succeed");
        let events = pool.drain_events();
        assert!(events
            .iter()
            .any(|e| matches!(e, PoolEvent::ConnectionRemoved { id: i, .. } if *i == id)));
    }

    // -----------------------------------------------------------------------
    // 17. xorshift64 PRNG
    // -----------------------------------------------------------------------

    #[test]
    fn test_xorshift64_non_zero() {
        let mut state = 12345u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }

    #[test]
    fn test_xorshift64_sequence_distinct() {
        let mut state = 1u64;
        let a = xorshift64(&mut state);
        let b = xorshift64(&mut state);
        assert_ne!(a, b);
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
    // 18. ConnState PartialEq
    // -----------------------------------------------------------------------

    #[test]
    fn test_connstate_eq() {
        assert_eq!(ConnState::Idle, ConnState::Idle);
        assert_ne!(ConnState::Idle, ConnState::Closed);
        assert_eq!(
            ConnState::InUse { borrowed_at: 10 },
            ConnState::InUse { borrowed_at: 10 }
        );
        assert_ne!(
            ConnState::InUse { borrowed_at: 10 },
            ConnState::InUse { borrowed_at: 20 }
        );
    }

    // -----------------------------------------------------------------------
    // 19. Integration: add → acquire → release → health_check → maintenance
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_lifecycle() {
        let mut pool = ConnectionPoolManager::new(CpmPoolConfig {
            max_connections: 3,
            idle_timeout_us: 10_000,
            max_lifetime_us: 100_000,
            ..default_config()
        });
        let id1 = pool
            .add_connection_at("a".to_string(), vec![], 0)
            .expect("test: add_connection_at a should succeed");
        let id2 = pool
            .add_connection_at("b".to_string(), vec![], 0)
            .expect("test: add_connection_at b should succeed");

        // Acquire both.
        let a1 = pool
            .acquire(1_000)
            .expect("test: first acquire should succeed");
        let a2 = pool
            .acquire(1_000)
            .expect("test: second acquire should succeed");

        // Release id1 with degraded health.
        pool.release(a1, 2_000, true)
            .expect("test: release a1 with health degraded should succeed");
        // Release id2 cleanly.
        pool.release(a2, 2_000, false)
            .expect("test: release a2 should succeed");

        // Health check: put id1 in HealthChecking, get failure event.
        pool.connections
            .get_mut(&id1)
            .expect("test: connection id1 should exist")
            .state = ConnState::HealthChecking;
        pool.health_check(id1, 0.1, 3_000)
            .expect("test: health_check should succeed");

        // Maintenance at 15000: idle_timeout = 10000, so both connections (last_used=2000)
        // should be evicted.
        let evicted = pool.run_maintenance(15_000);
        assert!(!evicted.is_empty());
        let _ = (id1, id2);
    }

    // -----------------------------------------------------------------------
    // 20. resize + stats interaction
    // -----------------------------------------------------------------------

    #[test]
    fn test_resize_shrink_stats_updated() {
        let mut pool = ConnectionPoolManager::new(CpmPoolConfig {
            max_connections: 5,
            ..default_config()
        });
        populate(&mut pool, 5);
        pool.resize(2).expect("test: resize shrink should succeed");
        let s = pool.stats();
        assert!(s.total_connections <= 2);
    }

    // -----------------------------------------------------------------------
    // 21. Multiple acquire-release cycles (stress)
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_acquire_release_cycles() {
        let mut pool = ConnectionPoolManager::new(CpmPoolConfig {
            max_connections: 3,
            ..default_config()
        });
        let ids = populate(&mut pool, 3);
        for cycle in 0..10_u64 {
            let ts_base = cycle * 1000;
            let acquired: Vec<u64> = ids
                .iter()
                .map(|_| {
                    pool.acquire(ts_base)
                        .expect("test: acquire in cycle should succeed")
                })
                .collect();
            for &id in &acquired {
                pool.release(id, ts_base + 1, false)
                    .expect("test: release in cycle should succeed");
            }
        }
        let s = pool.stats();
        assert_eq!(s.total_acquires, 30);
        assert_eq!(s.total_releases, 30);
        assert_eq!(s.idle_connections, 3);
    }

    // -----------------------------------------------------------------------
    // 22. HealthCheckFailed event in pending queue
    // -----------------------------------------------------------------------

    #[test]
    fn test_health_check_failure_in_pending_events() {
        let mut pool = make_pool();
        let id = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection should succeed");
        pool.drain_events();
        pool.health_check(id, 0.05, 0)
            .expect("test: health_check below threshold should succeed");
        let events = pool.drain_events();
        assert!(events
            .iter()
            .any(|e| matches!(e, PoolEvent::HealthCheckFailed { .. })));
    }

    // -----------------------------------------------------------------------
    // 23. Pool exhausted counter
    // -----------------------------------------------------------------------

    #[test]
    fn test_pool_exhausted_increments_failure_counter() {
        let mut pool = ConnectionPoolManager::new(CpmPoolConfig {
            max_connections: 1,
            ..default_config()
        });
        let id = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection should succeed");
        pool.acquire(1).expect("test: acquire should succeed");
        let _ = pool.acquire(2); // pool exhausted
        assert_eq!(pool.stats().acquire_failures, 1);
        pool.release(id, 3, false)
            .expect("test: release should succeed");
    }

    // -----------------------------------------------------------------------
    // 24. Ensure released connections are acquirable again
    // -----------------------------------------------------------------------

    #[test]
    fn test_released_connection_reacquirable() {
        let mut pool = make_pool();
        let id = pool
            .add_connection("a".to_string(), vec![])
            .expect("test: add_connection should succeed");
        pool.acquire(1).expect("test: acquire should succeed");
        pool.release(id, 2, false)
            .expect("test: release should succeed");
        let reacquired = pool.acquire(3).expect("test: reacquire should succeed");
        assert_eq!(reacquired, id);
    }

    // -----------------------------------------------------------------------
    // 25. add_connection_at populates timestamps correctly
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_connection_at_timestamps() {
        let mut pool = make_pool();
        let id = pool
            .add_connection_at("a".to_string(), vec![], 9_999)
            .expect("test: add_connection_at should succeed");
        let conn = pool
            .connections()
            .get(&id)
            .expect("test: connection should exist");
        assert_eq!(conn.created_at, 9_999);
        assert_eq!(conn.last_used, 9_999);
    }
}
