//! Session management for grouping related block requests
//!
//! Sessions provide a way to group related block requests together,
//! enabling features like:
//! - Batch completion notifications
//! - Session-level prioritization
//! - Coordinated cancellation
//! - Progress tracking across multiple blocks
//!
//! # Example
//!
//! ```
//! use ipfrs_transport::{SessionManager, SessionConfig, Priority};
//! use ipfrs_core::Cid;
//! use multihash::Multihash;
//!
//! // Create a session manager
//! let manager = SessionManager::new();
//!
//! // Create test CIDs
//! let hash1 = Multihash::wrap(0x12, &[1, 2, 3]).expect("valid multihash bytes");
//! let cid1 = Cid::new_v1(0x55, hash1);
//! let hash2 = Multihash::wrap(0x12, &[4, 5, 6]).expect("valid multihash bytes");
//! let cid2 = Cid::new_v1(0x55, hash2);
//!
//! // Create a session
//! let config = SessionConfig::default();
//! let session = manager.create_session(config);
//!
//! // Add blocks to the session
//! session.add_block(cid1, Some(Priority::Normal)).unwrap();
//! session.add_block(cid2, Some(Priority::High)).unwrap();
//!
//! // Check session status
//! let stats = session.stats();
//! println!("Total blocks: {}, received: {}", stats.total_blocks, stats.blocks_received);
//! ```

use crate::want_list::Priority;
use bytes::Bytes;
use cid::Cid;
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::{mpsc, watch};
use tracing::{debug, info, warn};

/// Session ID type
pub type SessionId = u64;

/// Session error types
#[derive(Error, Debug)]
pub enum SessionError {
    #[error("Session not found: {0}")]
    NotFound(SessionId),

    #[error("Session already exists: {0}")]
    AlreadyExists(SessionId),

    #[error("Session closed: {0}")]
    Closed(SessionId),

    #[error("Block not in session: {0}")]
    BlockNotInSession(String),

    #[error("Timeout waiting for session completion")]
    Timeout,
}

/// Session state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Session is active and accepting requests
    Active,
    /// Session is paused (not sending new requests)
    Paused,
    /// Session is completing (no new blocks, waiting for pending)
    Completing,
    /// Session is completed
    Completed,
    /// Session was cancelled
    Cancelled,
}

/// Session configuration
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Session timeout (0 = no timeout)
    pub timeout: Duration,
    /// Default priority for blocks in this session
    pub default_priority: Priority,
    /// Maximum concurrent blocks per session
    pub max_concurrent_blocks: usize,
    /// Enable progress notifications
    pub progress_notifications: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(300), // 5 minutes
            default_priority: Priority::Normal,
            max_concurrent_blocks: 100,
            progress_notifications: true,
        }
    }
}

/// Session statistics
#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    /// Total blocks requested
    pub total_blocks: usize,
    /// Blocks received
    pub blocks_received: usize,
    /// Blocks failed
    pub blocks_failed: usize,
    /// Total bytes transferred
    pub bytes_transferred: u64,
    /// Session start time
    pub started_at: Option<Instant>,
    /// Session end time
    pub completed_at: Option<Instant>,
    /// Average block fetch time
    pub avg_block_time: Option<Duration>,
}

impl SessionStats {
    /// Calculate progress percentage
    pub fn progress(&self) -> f64 {
        if self.total_blocks == 0 {
            return 0.0;
        }
        (self.blocks_received as f64 / self.total_blocks as f64) * 100.0
    }

    /// Calculate throughput in bytes per second
    pub fn throughput(&self) -> Option<f64> {
        if let (Some(started), Some(completed)) = (self.started_at, self.completed_at) {
            let duration = completed.duration_since(started).as_secs_f64();
            if duration > 0.0 {
                return Some(self.bytes_transferred as f64 / duration);
            }
        }
        None
    }

    /// Check if session is complete
    pub fn is_complete(&self) -> bool {
        self.blocks_received + self.blocks_failed >= self.total_blocks
    }
}

/// Block request within a session
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BlockRequest {
    cid: Cid,
    priority: Priority,
    requested_at: Instant,
    completed_at: Option<Instant>,
    size: Option<usize>,
}

/// Session progress event
#[derive(Debug, Clone)]
pub enum SessionEvent {
    /// Session started
    Started { session_id: SessionId },
    /// Block received
    BlockReceived {
        session_id: SessionId,
        cid: Cid,
        size: usize,
    },
    /// Block failed
    BlockFailed {
        session_id: SessionId,
        cid: Cid,
        error: String,
    },
    /// Session progress update
    Progress {
        session_id: SessionId,
        stats: SessionStats,
    },
    /// Session completed
    Completed {
        session_id: SessionId,
        stats: SessionStats,
    },
    /// Session cancelled
    Cancelled { session_id: SessionId },
}

/// A session for grouping related block requests
pub struct Session {
    id: SessionId,
    config: SessionConfig,
    state: Arc<RwLock<SessionState>>,
    blocks: Arc<DashMap<Cid, BlockRequest>>,
    stats: Arc<RwLock<SessionStats>>,
    event_tx: Option<mpsc::UnboundedSender<SessionEvent>>,
    state_rx: watch::Receiver<SessionState>,
    state_tx: watch::Sender<SessionState>,
}

impl Session {
    /// Create a new session
    pub fn new(
        id: SessionId,
        config: SessionConfig,
        event_tx: Option<mpsc::UnboundedSender<SessionEvent>>,
    ) -> Self {
        let (state_tx, state_rx) = watch::channel(SessionState::Active);

        let session = Self {
            id,
            config,
            state: Arc::new(RwLock::new(SessionState::Active)),
            blocks: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(SessionStats {
                started_at: Some(Instant::now()),
                ..Default::default()
            })),
            event_tx,
            state_rx,
            state_tx,
        };

        // Send started event
        if let Some(tx) = &session.event_tx {
            let _ = tx.send(SessionEvent::Started { session_id: id });
        }

        session
    }

    /// Get session ID
    pub fn id(&self) -> SessionId {
        self.id
    }

    /// Get current state
    pub fn state(&self) -> SessionState {
        *self.state.read()
    }

    /// Add a block to the session
    pub fn add_block(&self, cid: Cid, priority: Option<Priority>) -> Result<(), SessionError> {
        let state = *self.state.read();
        if state != SessionState::Active {
            return Err(SessionError::Closed(self.id));
        }

        let priority = priority.unwrap_or(self.config.default_priority);

        let request = BlockRequest {
            cid,
            priority,
            requested_at: Instant::now(),
            completed_at: None,
            size: None,
        };

        self.blocks.insert(cid, request);

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_blocks += 1;
        }

        debug!("Added block {} to session {}", cid, self.id);

        Ok(())
    }

    /// Add multiple blocks to the session
    pub fn add_blocks(&self, cids: &[Cid], priority: Option<Priority>) -> Result<(), SessionError> {
        for cid in cids {
            self.add_block(*cid, priority)?;
        }
        Ok(())
    }

    /// Mark a block as received
    pub fn mark_received(&self, cid: &Cid, data: &Bytes) -> Result<(), SessionError> {
        let mut block = self
            .blocks
            .get_mut(cid)
            .ok_or_else(|| SessionError::BlockNotInSession(cid.to_string()))?;

        block.completed_at = Some(Instant::now());
        block.size = Some(data.len());

        // Update stats and check for completion
        let should_complete = {
            let mut stats = self.stats.write();
            stats.blocks_received += 1;
            stats.bytes_transferred += data.len() as u64;

            // Update average block time
            let fetch_time = block
                .completed_at
                .expect("completed_at was just set above")
                .duration_since(block.requested_at);
            stats.avg_block_time = Some(
                stats
                    .avg_block_time
                    .map(|avg| (avg + fetch_time) / 2)
                    .unwrap_or(fetch_time),
            );

            // Check if session is complete
            let is_complete = stats.is_complete() && self.state() == SessionState::Active;
            if is_complete {
                stats.completed_at = Some(Instant::now());
            }
            is_complete
        }; // Release stats write lock here

        // Transition state outside the stats lock to avoid deadlock
        if should_complete {
            self.transition_state(SessionState::Completed);
        }

        // Send events
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(SessionEvent::BlockReceived {
                session_id: self.id,
                cid: *cid,
                size: data.len(),
            });

            if self.config.progress_notifications {
                let _ = tx.send(SessionEvent::Progress {
                    session_id: self.id,
                    stats: self.stats.read().clone(),
                });
            }
        }

        debug!("Block {} received in session {}", cid, self.id);

        Ok(())
    }

    /// Mark a block as failed
    pub fn mark_failed(&self, cid: &Cid, error: String) -> Result<(), SessionError> {
        let _block = self
            .blocks
            .get(cid)
            .ok_or_else(|| SessionError::BlockNotInSession(cid.to_string()))?;

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.blocks_failed += 1;

            // Check if session is complete
            if stats.is_complete() && self.state() == SessionState::Active {
                stats.completed_at = Some(Instant::now());
                self.transition_state(SessionState::Completed);
            }
        }

        // Send events
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(SessionEvent::BlockFailed {
                session_id: self.id,
                cid: *cid,
                error: error.clone(),
            });
        }

        warn!("Block {} failed in session {}: {}", cid, self.id, error);

        Ok(())
    }

    /// Pause the session
    pub fn pause(&self) {
        self.transition_state(SessionState::Paused);
        info!("Session {} paused", self.id);
    }

    /// Resume the session
    pub fn resume(&self) {
        self.transition_state(SessionState::Active);
        info!("Session {} resumed", self.id);
    }

    /// Cancel the session
    pub fn cancel(&self) {
        self.transition_state(SessionState::Cancelled);

        if let Some(tx) = &self.event_tx {
            let _ = tx.send(SessionEvent::Cancelled {
                session_id: self.id,
            });
        }

        info!("Session {} cancelled", self.id);
    }

    /// Get session statistics
    pub fn stats(&self) -> SessionStats {
        self.stats.read().clone()
    }

    /// Get pending blocks (not yet received)
    pub fn pending_blocks(&self) -> Vec<Cid> {
        self.blocks
            .iter()
            .filter(|entry| entry.value().completed_at.is_none())
            .map(|entry| *entry.key())
            .collect()
    }

    /// Wait for session completion
    pub async fn wait_completion(&self) -> Result<SessionStats, SessionError> {
        let mut rx = self.state_rx.clone();

        // Check if already complete
        let state = *self.state.read();
        if state == SessionState::Completed || state == SessionState::Cancelled {
            return Ok(self.stats.read().clone());
        }

        // Wait for state change
        loop {
            if rx.changed().await.is_err() {
                return Err(SessionError::Closed(self.id));
            }

            let state = *rx.borrow();
            if state == SessionState::Completed || state == SessionState::Cancelled {
                return Ok(self.stats.read().clone());
            }
        }
    }

    /// Transition to a new state
    fn transition_state(&self, new_state: SessionState) {
        *self.state.write() = new_state;
        let _ = self.state_tx.send(new_state);

        if new_state == SessionState::Completed {
            if let Some(tx) = &self.event_tx {
                let _ = tx.send(SessionEvent::Completed {
                    session_id: self.id,
                    stats: self.stats.read().clone(),
                });
            }
        }
    }
}

/// Session manager for managing multiple sessions
pub struct SessionManager {
    sessions: Arc<DashMap<SessionId, Arc<Session>>>,
    next_session_id: Arc<RwLock<SessionId>>,
    event_tx: mpsc::UnboundedSender<SessionEvent>,
    event_rx: Arc<RwLock<mpsc::UnboundedReceiver<SessionEvent>>>,
}

impl SessionManager {
    /// Create a new session manager
    pub fn new() -> Self {
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        Self {
            sessions: Arc::new(DashMap::new()),
            next_session_id: Arc::new(RwLock::new(1)),
            event_tx,
            event_rx: Arc::new(RwLock::new(event_rx)),
        }
    }

    /// Create a new session
    pub fn create_session(&self, config: SessionConfig) -> Arc<Session> {
        let session_id = {
            let mut id = self.next_session_id.write();
            let current = *id;
            *id += 1;
            current
        };

        let session = Arc::new(Session::new(
            session_id,
            config,
            Some(self.event_tx.clone()),
        ));
        self.sessions.insert(session_id, session.clone());

        info!("Created session {}", session_id);

        session
    }

    /// Get a session by ID
    pub fn get_session(&self, session_id: SessionId) -> Option<Arc<Session>> {
        self.sessions.get(&session_id).map(|s| s.clone())
    }

    /// Remove a session
    pub fn remove_session(&self, session_id: SessionId) -> Option<Arc<Session>> {
        self.sessions.remove(&session_id).map(|(_, s)| s)
    }

    /// Get all active sessions
    pub fn active_sessions(&self) -> Vec<Arc<Session>> {
        self.sessions
            .iter()
            .filter(|entry| entry.value().state() == SessionState::Active)
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Clean up completed sessions
    pub fn cleanup_completed(&self) -> usize {
        let to_remove: Vec<_> = self
            .sessions
            .iter()
            .filter(|entry| {
                let state = entry.value().state();
                state == SessionState::Completed || state == SessionState::Cancelled
            })
            .map(|entry| *entry.key())
            .collect();

        let count = to_remove.len();
        for session_id in to_remove {
            self.sessions.remove(&session_id);
        }

        if count > 0 {
            info!("Cleaned up {} completed sessions", count);
        }

        count
    }

    /// Receive session events
    #[allow(clippy::await_holding_lock)]
    pub async fn recv_event(&self) -> Option<SessionEvent> {
        self.event_rx.write().recv().await
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_cid(n: u8) -> Cid {
        let data = vec![n; 32];
        Cid::new_v1(
            0x55,
            multihash::Multihash::wrap(0x12, &data).expect("test: create multihash"),
        )
    }

    #[test]
    fn test_session_creation() {
        let manager = SessionManager::new();
        let session = manager.create_session(SessionConfig::default());

        assert_eq!(session.state(), SessionState::Active);
        assert_eq!(session.stats().total_blocks, 0);
    }

    #[test]
    fn test_add_blocks() {
        let manager = SessionManager::new();
        let session = manager.create_session(SessionConfig::default());

        let cid1 = dummy_cid(1);
        let cid2 = dummy_cid(2);

        session
            .add_block(cid1, None)
            .expect("test: add block to session");
        session
            .add_block(cid2, Some(Priority::High))
            .expect("test: add block to session");

        let stats = session.stats();
        assert_eq!(stats.total_blocks, 2);
        assert_eq!(stats.blocks_received, 0);
    }

    #[test]
    fn test_mark_received() {
        let manager = SessionManager::new();
        let session = manager.create_session(SessionConfig::default());

        let cid = dummy_cid(1);
        session
            .add_block(cid, None)
            .expect("test: add block to session");

        let data = Bytes::from(vec![1, 2, 3, 4]);
        session
            .mark_received(&cid, &data)
            .expect("test: mark received");

        let stats = session.stats();
        assert_eq!(stats.blocks_received, 1);
        assert_eq!(stats.bytes_transferred, 4);
        assert!(stats.is_complete());
    }

    #[test]
    fn test_session_progress() {
        let manager = SessionManager::new();
        let session = manager.create_session(SessionConfig::default());

        session
            .add_blocks(&[dummy_cid(1), dummy_cid(2), dummy_cid(3)], None)
            .expect("test: add blocks to session");

        session
            .mark_received(&dummy_cid(1), &Bytes::from(vec![1]))
            .expect("test: mark received");
        let progress1 = session.stats().progress();
        assert!(
            (progress1 - 100.0 / 3.0).abs() < 1e-6,
            "Expected ~33.33%, got {}",
            progress1
        );

        session
            .mark_received(&dummy_cid(2), &Bytes::from(vec![2]))
            .expect("test: mark received");
        let progress2 = session.stats().progress();
        assert!(
            (progress2 - 200.0 / 3.0).abs() < 1e-6,
            "Expected ~66.67%, got {}",
            progress2
        );

        session
            .mark_received(&dummy_cid(3), &Bytes::from(vec![3]))
            .expect("test: mark received");
        let progress3 = session.stats().progress();
        assert!(
            (progress3 - 100.0).abs() < 1e-6,
            "Expected 100%, got {}",
            progress3
        );
        assert_eq!(session.state(), SessionState::Completed);
    }

    #[test]
    fn test_pause_resume() {
        let manager = SessionManager::new();
        let session = manager.create_session(SessionConfig::default());

        assert_eq!(session.state(), SessionState::Active);

        session.pause();
        assert_eq!(session.state(), SessionState::Paused);

        session.resume();
        assert_eq!(session.state(), SessionState::Active);
    }

    #[test]
    fn test_cancel() {
        let manager = SessionManager::new();
        let session = manager.create_session(SessionConfig::default());

        session.cancel();
        assert_eq!(session.state(), SessionState::Cancelled);

        // Should not be able to add blocks to cancelled session
        assert!(session.add_block(dummy_cid(1), None).is_err());
    }

    #[tokio::test]
    async fn test_wait_completion() {
        let manager = SessionManager::new();
        let session = manager.create_session(SessionConfig::default());

        session
            .add_block(dummy_cid(1), None)
            .expect("test: add block to session");

        let session_clone = session.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            session_clone
                .mark_received(&dummy_cid(1), &Bytes::from(vec![1]))
                .expect("test: mark received");
        });

        let stats = session
            .wait_completion()
            .await
            .expect("test: wait completion");
        assert_eq!(stats.blocks_received, 1);
    }
}
