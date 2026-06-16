//! Connection Session Management
//!
//! This module provides utilities for tracking connection sessions and their lifecycles.
//!
//! # Features
//!
//! - Track active connection sessions with metadata
//! - Session lifecycle management (created, active, idle, closing, closed)
//! - Idle timeout detection
//! - Session statistics and metrics
//! - Callback hooks for lifecycle events
//!
//! # Example
//!
//! ```
//! use ipfrs_network::session::{SessionManager, SessionConfig, SessionState};
//! use libp2p::PeerId;
//! use std::time::Duration;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = SessionConfig {
//!     idle_timeout: Duration::from_secs(300),
//!     max_sessions: 1000,
//!     ..Default::default()
//! };
//!
//! let mut manager = SessionManager::new(config);
//!
//! // Create a new session
//! let peer_id = PeerId::random();
//! manager.create_session(peer_id);
//!
//! // Check session state
//! if let Some(session) = manager.get_session(&peer_id) {
//!     println!("Session state: {:?}", session.state);
//!     println!("Duration: {:?}", session.duration());
//! }
//!
//! // Update activity
//! manager.mark_activity(&peer_id);
//!
//! // Close session
//! manager.close_session(&peer_id);
//! # Ok(())
//! # }
//! ```

use dashmap::DashMap;
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Session state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionState {
    /// Session is being created
    Creating,
    /// Session is active
    Active,
    /// Session is idle (no recent activity)
    Idle,
    /// Session is being closed
    Closing,
    /// Session is closed
    Closed,
}

impl SessionState {
    /// Check if session is active (creating or active)
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Creating | Self::Active)
    }

    /// Check if session is terminated (closing or closed)
    pub fn is_terminated(&self) -> bool {
        matches!(self, Self::Closing | Self::Closed)
    }
}

/// Connection session information
#[derive(Debug, Clone)]
pub struct Session {
    /// Peer ID
    pub peer_id: PeerId,
    /// Session state
    pub state: SessionState,
    /// Session creation time
    pub created_at: Instant,
    /// Last activity time
    pub last_activity: Instant,
    /// Session closed time (if closed)
    pub closed_at: Option<Instant>,
    /// Bytes sent in this session
    pub bytes_sent: u64,
    /// Bytes received in this session
    pub bytes_received: u64,
    /// Number of messages sent
    pub messages_sent: u64,
    /// Number of messages received
    pub messages_received: u64,
    /// Custom session metadata
    pub metadata: SessionMetadata,
}

impl Session {
    /// Create a new session
    pub fn new(peer_id: PeerId) -> Self {
        let now = Instant::now();
        Self {
            peer_id,
            state: SessionState::Creating,
            created_at: now,
            last_activity: now,
            closed_at: None,
            bytes_sent: 0,
            bytes_received: 0,
            messages_sent: 0,
            messages_received: 0,
            metadata: SessionMetadata::default(),
        }
    }

    /// Get session duration
    pub fn duration(&self) -> Duration {
        if let Some(closed_at) = self.closed_at {
            closed_at.duration_since(self.created_at)
        } else {
            Instant::now().duration_since(self.created_at)
        }
    }

    /// Get time since last activity
    pub fn idle_duration(&self) -> Duration {
        Instant::now().duration_since(self.last_activity)
    }

    /// Check if session is idle (exceeds timeout)
    pub fn is_idle(&self, timeout: Duration) -> bool {
        self.idle_duration() > timeout
    }

    /// Mark activity
    pub fn mark_activity(&mut self) {
        self.last_activity = Instant::now();
        if self.state == SessionState::Idle {
            self.state = SessionState::Active;
        }
    }

    /// Update sent bytes
    pub fn add_bytes_sent(&mut self, bytes: u64) {
        self.bytes_sent += bytes;
        self.mark_activity();
    }

    /// Update received bytes
    pub fn add_bytes_received(&mut self, bytes: u64) {
        self.bytes_received += bytes;
        self.mark_activity();
    }

    /// Record message sent
    pub fn record_message_sent(&mut self) {
        self.messages_sent += 1;
        self.mark_activity();
    }

    /// Record message received
    pub fn record_message_received(&mut self) {
        self.messages_received += 1;
        self.mark_activity();
    }
}

/// Session metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Connection endpoint (address)
    pub endpoint: Option<String>,
    /// Connection protocol
    pub protocol: Option<String>,
    /// Session tags
    pub tags: Vec<String>,
    /// Connection quality score (0.0-1.0)
    pub quality_score: Option<f64>,
}

/// Session configuration
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Idle timeout before marking session as idle
    pub idle_timeout: Duration,
    /// Maximum number of concurrent sessions
    pub max_sessions: usize,
    /// Enable automatic cleanup of closed sessions
    pub auto_cleanup: bool,
    /// Cleanup interval for closed sessions
    pub cleanup_interval: Duration,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            idle_timeout: Duration::from_secs(300), // 5 minutes
            max_sessions: 1000,
            auto_cleanup: true,
            cleanup_interval: Duration::from_secs(60), // 1 minute
        }
    }
}

/// Session statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionStats {
    /// Total sessions created
    pub total_created: u64,
    /// Currently active sessions
    pub active_sessions: usize,
    /// Currently idle sessions
    pub idle_sessions: usize,
    /// Total sessions closed
    pub total_closed: u64,
    /// Total bytes sent across all sessions
    pub total_bytes_sent: u64,
    /// Total bytes received across all sessions
    pub total_bytes_received: u64,
    /// Total messages sent
    pub total_messages_sent: u64,
    /// Total messages received
    pub total_messages_received: u64,
    /// Average session duration
    pub avg_session_duration: Duration,
}

/// Session manager
pub struct SessionManager {
    /// Configuration
    config: SessionConfig,
    /// Active sessions
    sessions: Arc<DashMap<PeerId, Session>>,
    /// Statistics
    stats: Arc<parking_lot::RwLock<SessionStats>>,
}

impl SessionManager {
    /// Create a new session manager
    pub fn new(config: SessionConfig) -> Self {
        Self {
            config,
            sessions: Arc::new(DashMap::new()),
            stats: Arc::new(parking_lot::RwLock::new(SessionStats::default())),
        }
    }

    /// Create a new session
    pub fn create_session(&self, peer_id: PeerId) -> bool {
        // Check max sessions limit
        if self.sessions.len() >= self.config.max_sessions {
            return false;
        }

        let session = Session::new(peer_id);
        let inserted = self.sessions.insert(peer_id, session).is_none();

        if inserted {
            let mut stats = self.stats.write();
            stats.total_created += 1;
        }

        inserted
    }

    /// Get a session
    pub fn get_session(&self, peer_id: &PeerId) -> Option<Session> {
        self.sessions.get(peer_id).map(|s| s.clone())
    }

    /// Mark session as active
    pub fn activate_session(&self, peer_id: &PeerId) {
        if let Some(mut session) = self.sessions.get_mut(peer_id) {
            session.state = SessionState::Active;
            session.mark_activity();
        }
    }

    /// Mark activity on a session
    pub fn mark_activity(&self, peer_id: &PeerId) {
        if let Some(mut session) = self.sessions.get_mut(peer_id) {
            session.mark_activity();
        }
    }

    /// Update session bandwidth
    pub fn update_bandwidth(&self, peer_id: &PeerId, sent: u64, received: u64) {
        if let Some(mut session) = self.sessions.get_mut(peer_id) {
            session.add_bytes_sent(sent);
            session.add_bytes_received(received);

            let mut stats = self.stats.write();
            stats.total_bytes_sent += sent;
            stats.total_bytes_received += received;
        }
    }

    /// Record message activity
    pub fn record_message(&self, peer_id: &PeerId, sent: bool) {
        if let Some(mut session) = self.sessions.get_mut(peer_id) {
            if sent {
                session.record_message_sent();
                let mut stats = self.stats.write();
                stats.total_messages_sent += 1;
            } else {
                session.record_message_received();
                let mut stats = self.stats.write();
                stats.total_messages_received += 1;
            }
        }
    }

    /// Update session metadata
    pub fn update_metadata(&self, peer_id: &PeerId, metadata: SessionMetadata) {
        if let Some(mut session) = self.sessions.get_mut(peer_id) {
            session.metadata = metadata;
        }
    }

    /// Close a session
    pub fn close_session(&self, peer_id: &PeerId) {
        if let Some(mut session) = self.sessions.get_mut(peer_id) {
            session.state = SessionState::Closing;
            session.closed_at = Some(Instant::now());

            let mut stats = self.stats.write();
            stats.total_closed += 1;

            // Update average duration
            let total_duration = stats.avg_session_duration.as_secs_f64()
                * (stats.total_closed - 1) as f64
                + session.duration().as_secs_f64();
            stats.avg_session_duration =
                Duration::from_secs_f64(total_duration / stats.total_closed as f64);
        }
    }

    /// Remove a closed session
    pub fn remove_session(&self, peer_id: &PeerId) -> Option<Session> {
        self.sessions.remove(peer_id).map(|(_, s)| s)
    }

    /// Get all sessions
    pub fn get_all_sessions(&self) -> Vec<Session> {
        self.sessions.iter().map(|entry| entry.clone()).collect()
    }

    /// Get sessions by state
    pub fn get_sessions_by_state(&self, state: SessionState) -> Vec<Session> {
        self.sessions
            .iter()
            .filter(|entry| entry.state == state)
            .map(|entry| entry.clone())
            .collect()
    }

    /// Check for idle sessions and mark them
    pub fn check_idle_sessions(&self) -> Vec<PeerId> {
        let mut idle_peers = Vec::new();

        for mut entry in self.sessions.iter_mut() {
            if entry.state == SessionState::Active && entry.is_idle(self.config.idle_timeout) {
                entry.state = SessionState::Idle;
                idle_peers.push(entry.peer_id);
            }
        }

        idle_peers
    }

    /// Cleanup closed sessions
    pub fn cleanup_closed_sessions(&self) -> usize {
        let closed_sessions: Vec<PeerId> = self
            .sessions
            .iter()
            .filter(|entry| entry.state == SessionState::Closed)
            .map(|entry| entry.peer_id)
            .collect();

        let count = closed_sessions.len();
        for peer_id in closed_sessions {
            self.sessions.remove(&peer_id);
        }

        count
    }

    /// Get statistics
    pub fn stats(&self) -> SessionStats {
        let mut stats = self.stats.read().clone();

        // Update current session counts
        stats.active_sessions = self
            .sessions
            .iter()
            .filter(|e| e.state == SessionState::Active)
            .count();
        stats.idle_sessions = self
            .sessions
            .iter()
            .filter(|e| e.state == SessionState::Idle)
            .count();

        stats
    }

    /// Get session count
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Get active session count
    pub fn active_session_count(&self) -> usize {
        self.sessions.iter().filter(|e| e.state.is_active()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation() {
        let peer_id = PeerId::random();
        let session = Session::new(peer_id);

        assert_eq!(session.peer_id, peer_id);
        assert_eq!(session.state, SessionState::Creating);
        assert_eq!(session.bytes_sent, 0);
        assert_eq!(session.bytes_received, 0);
    }

    #[test]
    fn test_session_state() {
        assert!(SessionState::Creating.is_active());
        assert!(SessionState::Active.is_active());
        assert!(!SessionState::Idle.is_active());
        assert!(SessionState::Closing.is_terminated());
        assert!(SessionState::Closed.is_terminated());
    }

    #[test]
    fn test_session_activity() {
        let peer_id = PeerId::random();
        let mut session = Session::new(peer_id);

        let initial_time = session.last_activity;
        std::thread::sleep(std::time::Duration::from_millis(10));

        session.mark_activity();
        assert!(session.last_activity > initial_time);
    }

    #[test]
    fn test_session_bandwidth() {
        let peer_id = PeerId::random();
        let mut session = Session::new(peer_id);

        session.add_bytes_sent(1024);
        session.add_bytes_received(2048);

        assert_eq!(session.bytes_sent, 1024);
        assert_eq!(session.bytes_received, 2048);
    }

    #[test]
    fn test_session_messages() {
        let peer_id = PeerId::random();
        let mut session = Session::new(peer_id);

        session.record_message_sent();
        session.record_message_sent();
        session.record_message_received();

        assert_eq!(session.messages_sent, 2);
        assert_eq!(session.messages_received, 1);
    }

    #[test]
    fn test_session_manager_creation() {
        let config = SessionConfig::default();
        let manager = SessionManager::new(config);

        let peer_id = PeerId::random();
        assert!(manager.create_session(peer_id));
        assert_eq!(manager.session_count(), 1);

        let stats = manager.stats();
        assert_eq!(stats.total_created, 1);
    }

    #[test]
    fn test_session_manager_max_sessions() {
        let config = SessionConfig {
            max_sessions: 2,
            ..Default::default()
        };
        let manager = SessionManager::new(config);

        let peer1 = PeerId::random();
        let peer2 = PeerId::random();
        let peer3 = PeerId::random();

        assert!(manager.create_session(peer1));
        assert!(manager.create_session(peer2));
        assert!(!manager.create_session(peer3)); // Should fail (max reached)

        assert_eq!(manager.session_count(), 2);
    }

    #[test]
    fn test_session_manager_activity() {
        let manager = SessionManager::new(SessionConfig::default());
        let peer_id = PeerId::random();

        manager.create_session(peer_id);
        manager.activate_session(&peer_id);

        let session = manager
            .get_session(&peer_id)
            .expect("test: session should exist after activate_session");
        assert_eq!(session.state, SessionState::Active);
    }

    #[test]
    fn test_session_manager_bandwidth() {
        let manager = SessionManager::new(SessionConfig::default());
        let peer_id = PeerId::random();

        manager.create_session(peer_id);
        manager.update_bandwidth(&peer_id, 1024, 2048);

        let session = manager
            .get_session(&peer_id)
            .expect("test: session should exist after update_bandwidth");
        assert_eq!(session.bytes_sent, 1024);
        assert_eq!(session.bytes_received, 2048);

        let stats = manager.stats();
        assert_eq!(stats.total_bytes_sent, 1024);
        assert_eq!(stats.total_bytes_received, 2048);
    }

    #[test]
    fn test_session_manager_close() {
        let manager = SessionManager::new(SessionConfig::default());
        let peer_id = PeerId::random();

        manager.create_session(peer_id);
        manager.close_session(&peer_id);

        let session = manager
            .get_session(&peer_id)
            .expect("test: session should exist after close_session");
        assert_eq!(session.state, SessionState::Closing);
        assert!(session.closed_at.is_some());

        let stats = manager.stats();
        assert_eq!(stats.total_closed, 1);
    }

    #[test]
    fn test_session_manager_remove() {
        let manager = SessionManager::new(SessionConfig::default());
        let peer_id = PeerId::random();

        manager.create_session(peer_id);
        assert_eq!(manager.session_count(), 1);

        manager.remove_session(&peer_id);
        assert_eq!(manager.session_count(), 0);
    }

    #[test]
    fn test_session_manager_filter_by_state() {
        let manager = SessionManager::new(SessionConfig::default());

        let peer1 = PeerId::random();
        let peer2 = PeerId::random();

        manager.create_session(peer1);
        manager.create_session(peer2);
        manager.activate_session(&peer1);

        let active = manager.get_sessions_by_state(SessionState::Active);
        assert_eq!(active.len(), 1);

        let creating = manager.get_sessions_by_state(SessionState::Creating);
        assert_eq!(creating.len(), 1);
    }

    #[test]
    fn test_session_manager_cleanup() {
        let manager = SessionManager::new(SessionConfig::default());

        let peer1 = PeerId::random();
        let peer2 = PeerId::random();

        manager.create_session(peer1);
        manager.create_session(peer2);

        manager.close_session(&peer1);
        if let Some(mut session) = manager.sessions.get_mut(&peer1) {
            session.state = SessionState::Closed;
        }

        let cleaned = manager.cleanup_closed_sessions();
        assert_eq!(cleaned, 1);
        assert_eq!(manager.session_count(), 1);
    }

    #[test]
    fn test_session_idle_detection() {
        let peer_id = PeerId::random();
        let session = Session::new(peer_id);

        // Should not be idle immediately
        assert!(!session.is_idle(Duration::from_secs(1)));

        // Create session with old last_activity
        let mut old_session = Session::new(peer_id);
        old_session.last_activity = Instant::now() - Duration::from_secs(10);

        // Should be idle after 5 seconds
        assert!(old_session.is_idle(Duration::from_secs(5)));
    }
}
