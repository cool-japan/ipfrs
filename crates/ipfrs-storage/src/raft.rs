//! RAFT Consensus Protocol Implementation
//!
//! This module implements the RAFT consensus algorithm for distributed storage.
//! RAFT provides strong consistency guarantees through leader election and log replication.
//!
//! # Architecture
//!
//! - **RaftNode**: Main RAFT node that participates in consensus
//! - **RaftLog**: Append-only log of operations
//! - **StateMachine**: Applies committed operations to the underlying BlockStore
//! - **RPC Protocol**: AppendEntries and RequestVote for node communication
//!
//! # Example
//!
//! ```ignore
//! use ipfrs_storage::raft::{RaftNode, RaftConfig, NodeId};
//! use ipfrs_storage::sled::SledBlockStore;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let store = SledBlockStore::new(std::env::temp_dir().join("raft-node-1"))?;
//!     let config = RaftConfig::default();
//!
//!     let mut node = RaftNode::new(
//!         NodeId(1),
//!         vec![NodeId(2), NodeId(3)],
//!         store,
//!         config,
//!     )?;
//!
//!     node.start().await?;
//!     Ok(())
//! }
//! ```

use crate::traits::BlockStore;
use ipfrs_core::{Block, Cid, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio::time;
use tracing::{debug, info};

/// Unique identifier for a RAFT node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u64);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

/// RAFT term number (monotonically increasing)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Term(pub u64);

impl Term {
    pub fn increment(&mut self) {
        self.0 += 1;
    }
}

/// Index in the RAFT log
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub struct LogIndex(pub u64);

impl LogIndex {
    pub fn increment(&mut self) {
        self.0 += 1;
    }
}

/// Node state in RAFT protocol
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeState {
    /// Follower state (default)
    Follower,
    /// Candidate state (during election)
    Candidate,
    /// Leader state (elected leader)
    Leader,
}

/// RAFT log entry containing a command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Term when entry was received by leader
    pub term: Term,
    /// Index in the log
    pub index: LogIndex,
    /// Command to execute on state machine
    pub command: Command,
}

/// Command that can be applied to the state machine (BlockStore)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Command {
    /// Put a block (stores CID and data separately)
    Put { cid_bytes: Vec<u8>, data: Vec<u8> },
    /// Delete a block (CID stored as bytes)
    Delete { cid_bytes: Vec<u8> },
    /// No-op (used for leader election)
    NoOp,
}

/// AppendEntries RPC request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesRequest {
    /// Leader's term
    pub term: Term,
    /// Leader's ID (so follower can redirect clients)
    pub leader_id: NodeId,
    /// Index of log entry immediately preceding new ones
    pub prev_log_index: LogIndex,
    /// Term of prev_log_index entry
    pub prev_log_term: Term,
    /// Log entries to store (empty for heartbeat)
    pub entries: Vec<LogEntry>,
    /// Leader's commit index
    pub leader_commit: LogIndex,
}

/// AppendEntries RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesResponse {
    /// Current term, for leader to update itself
    pub term: Term,
    /// True if follower contained entry matching prev_log_index and prev_log_term
    pub success: bool,
    /// Hint for leader to backtrack (next index to try)
    pub conflict_index: Option<LogIndex>,
}

/// RequestVote RPC request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVoteRequest {
    /// Candidate's term
    pub term: Term,
    /// Candidate requesting vote
    pub candidate_id: NodeId,
    /// Index of candidate's last log entry
    pub last_log_index: LogIndex,
    /// Term of candidate's last log entry
    pub last_log_term: Term,
}

/// RequestVote RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVoteResponse {
    /// Current term, for candidate to update itself
    pub term: Term,
    /// True means candidate received vote
    pub vote_granted: bool,
}

/// Configuration for RAFT node
#[derive(Debug, Clone)]
pub struct RaftConfig {
    /// Heartbeat interval (when leader)
    pub heartbeat_interval: Duration,
    /// Election timeout range (randomized to avoid split votes)
    pub election_timeout_min: Duration,
    pub election_timeout_max: Duration,
    /// Maximum number of entries to send in one AppendEntries RPC
    pub max_entries_per_append: usize,
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval: Duration::from_millis(50),
            election_timeout_min: Duration::from_millis(150),
            election_timeout_max: Duration::from_millis(300),
            max_entries_per_append: 100,
        }
    }
}

/// Persistent state on all servers (must survive restarts)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistentState {
    /// Latest term server has seen
    current_term: Term,
    /// Candidate that received vote in current term
    voted_for: Option<NodeId>,
}

impl Default for PersistentState {
    fn default() -> Self {
        Self {
            current_term: Term(0),
            voted_for: None,
        }
    }
}

/// Volatile state on all servers
#[derive(Debug, Default)]
struct VolatileState {
    /// Index of highest log entry known to be committed
    commit_index: LogIndex,
    /// Index of highest log entry applied to state machine
    last_applied: LogIndex,
}

/// Volatile state on leaders (reinitialized after election)
#[derive(Debug)]
#[allow(dead_code)]
struct LeaderState {
    /// For each server, index of next log entry to send
    next_index: HashMap<NodeId, LogIndex>,
    /// For each server, index of highest log entry known to be replicated
    match_index: HashMap<NodeId, LogIndex>,
}

/// RAFT node that participates in consensus
pub struct RaftNode<S: BlockStore> {
    /// This node's ID
    id: NodeId,
    /// Other nodes in the cluster
    peers: Vec<NodeId>,
    /// Current state (Follower, Candidate, Leader)
    state: Arc<RwLock<NodeState>>,
    /// Persistent state
    persistent: Arc<RwLock<PersistentState>>,
    /// Volatile state
    volatile: Arc<RwLock<VolatileState>>,
    /// Leader state (only valid when Leader)
    #[allow(dead_code)]
    leader_state: Arc<RwLock<Option<LeaderState>>>,
    /// RAFT log
    log: Arc<RwLock<Vec<LogEntry>>>,
    /// Underlying block store (state machine)
    store: Arc<S>,
    /// Configuration
    config: RaftConfig,
    /// Last time we heard from leader (for election timeout)
    last_heartbeat: Arc<RwLock<Instant>>,
    /// Current leader (if known)
    current_leader: Arc<RwLock<Option<NodeId>>>,
    /// Channel for RPC requests
    rpc_tx: mpsc::UnboundedSender<RpcMessage>,
    rpc_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<RpcMessage>>>>,
}

/// RPC message for internal communication
#[derive(Debug)]
#[allow(dead_code)]
enum RpcMessage {
    AppendEntries {
        request: AppendEntriesRequest,
        response_tx: oneshot::Sender<AppendEntriesResponse>,
    },
    RequestVote {
        request: RequestVoteRequest,
        response_tx: oneshot::Sender<RequestVoteResponse>,
    },
}

impl<S: BlockStore + Send + Sync + 'static> RaftNode<S> {
    /// Create a new RAFT node
    pub fn new(id: NodeId, peers: Vec<NodeId>, store: S, config: RaftConfig) -> Result<Self> {
        let (rpc_tx, rpc_rx) = mpsc::unbounded_channel();

        Ok(Self {
            id,
            peers,
            state: Arc::new(RwLock::new(NodeState::Follower)),
            persistent: Arc::new(RwLock::new(PersistentState::default())),
            volatile: Arc::new(RwLock::new(VolatileState::default())),
            leader_state: Arc::new(RwLock::new(None)),
            log: Arc::new(RwLock::new(Vec::new())),
            store: Arc::new(store),
            config,
            last_heartbeat: Arc::new(RwLock::new(Instant::now())),
            current_leader: Arc::new(RwLock::new(None)),
            rpc_tx,
            rpc_rx: Arc::new(RwLock::new(Some(rpc_rx))),
        })
    }

    /// Start the RAFT node
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting RAFT node {}", self.id);

        // Take the receiver out of the option
        let mut rpc_rx = self
            .rpc_rx
            .write()
            .take()
            .ok_or_else(|| ipfrs_core::Error::Internal("Node already started".to_string()))?;

        // Spawn election timer
        let _election_handle = self.spawn_election_timer();

        // Main event loop
        loop {
            tokio::select! {
                // Handle RPC messages
                Some(msg) = rpc_rx.recv() => {
                    self.handle_rpc(msg).await?;
                }
                // Periodic tasks (apply committed entries)
                _ = time::sleep(Duration::from_millis(10)) => {
                    self.apply_committed_entries().await?;
                }
            }
        }
    }

    /// Spawn election timer task
    fn spawn_election_timer(&self) -> tokio::task::JoinHandle<()> {
        let id = self.id;
        let state = Arc::clone(&self.state);
        let persistent = Arc::clone(&self.persistent);
        let last_heartbeat = Arc::clone(&self.last_heartbeat);
        let config = self.config.clone();
        let _peers = self.peers.clone();
        let _log = Arc::clone(&self.log);
        let _rpc_tx = self.rpc_tx.clone();

        tokio::spawn(async move {
            loop {
                // Calculate randomized election timeout
                let timeout = Self::random_election_timeout(&config);
                time::sleep(timeout).await;

                // Check if we should start an election
                let current_state = *state.read();
                let elapsed = last_heartbeat.read().elapsed();

                if current_state != NodeState::Leader && elapsed >= timeout {
                    info!("{}: Election timeout, starting election", id);
                    // Start election (simplified - would need to send RequestVote RPCs)
                    *state.write() = NodeState::Candidate;
                    persistent.write().current_term.increment();
                    persistent.write().voted_for = Some(id);
                }
            }
        })
    }

    /// Get a random election timeout
    fn random_election_timeout(config: &RaftConfig) -> Duration {
        use rand::RngExt;
        let min = config.election_timeout_min.as_millis() as u64;
        let max = config.election_timeout_max.as_millis() as u64;
        let timeout_ms = rand::rng().random_range(min..=max);
        Duration::from_millis(timeout_ms)
    }

    /// Handle incoming RPC message
    async fn handle_rpc(&self, msg: RpcMessage) -> Result<()> {
        match msg {
            RpcMessage::AppendEntries {
                request,
                response_tx,
            } => {
                let response = self.handle_append_entries(request).await?;
                let _ = response_tx.send(response);
            }
            RpcMessage::RequestVote {
                request,
                response_tx,
            } => {
                let response = self.handle_request_vote(request).await?;
                let _ = response_tx.send(response);
            }
        }
        Ok(())
    }

    /// Handle AppendEntries RPC
    #[allow(clippy::unused_async)]
    async fn handle_append_entries(
        &self,
        request: AppendEntriesRequest,
    ) -> Result<AppendEntriesResponse> {
        let mut persistent = self.persistent.write();
        let current_term = persistent.current_term;

        // Reply false if term < currentTerm
        if request.term < current_term {
            return Ok(AppendEntriesResponse {
                term: current_term,
                success: false,
                conflict_index: None,
            });
        }

        // Update term if we see a higher one
        if request.term > current_term {
            persistent.current_term = request.term;
            persistent.voted_for = None;
            *self.state.write() = NodeState::Follower;
        }

        // Reset election timer (we heard from leader)
        *self.last_heartbeat.write() = Instant::now();
        *self.current_leader.write() = Some(request.leader_id);

        let mut log = self.log.write();

        // Reply false if log doesn't contain entry at prev_log_index with prev_log_term
        if request.prev_log_index.0 > 0 {
            if request.prev_log_index.0 > log.len() as u64 {
                return Ok(AppendEntriesResponse {
                    term: persistent.current_term,
                    success: false,
                    conflict_index: Some(LogIndex(log.len() as u64)),
                });
            }

            let prev_entry = &log[(request.prev_log_index.0 - 1) as usize];
            if prev_entry.term != request.prev_log_term {
                // Find conflicting term's first index
                let conflict_term = prev_entry.term;
                let mut conflict_index = request.prev_log_index.0;
                for entry in log.iter().rev() {
                    if entry.term != conflict_term {
                        break;
                    }
                    conflict_index = entry.index.0;
                }

                return Ok(AppendEntriesResponse {
                    term: persistent.current_term,
                    success: false,
                    conflict_index: Some(LogIndex(conflict_index)),
                });
            }
        }

        // Append new entries
        for entry in request.entries {
            let index = (entry.index.0 - 1) as usize;
            if index >= log.len() {
                log.push(entry);
            } else if log[index].term != entry.term {
                // Delete conflicting entry and all that follow
                log.truncate(index);
                log.push(entry);
            }
        }

        // Update commit index
        if request.leader_commit.0 > self.volatile.read().commit_index.0 {
            let new_commit = request.leader_commit.0.min(log.len() as u64);
            self.volatile.write().commit_index = LogIndex(new_commit);
        }

        Ok(AppendEntriesResponse {
            term: persistent.current_term,
            success: true,
            conflict_index: None,
        })
    }

    /// Handle RequestVote RPC
    #[allow(clippy::unused_async)]
    async fn handle_request_vote(
        &self,
        request: RequestVoteRequest,
    ) -> Result<RequestVoteResponse> {
        let mut persistent = self.persistent.write();
        let current_term = persistent.current_term;

        // Reply false if term < currentTerm
        if request.term < current_term {
            return Ok(RequestVoteResponse {
                term: current_term,
                vote_granted: false,
            });
        }

        // Update term if we see a higher one
        if request.term > current_term {
            persistent.current_term = request.term;
            persistent.voted_for = None;
            *self.state.write() = NodeState::Follower;
        }

        // Grant vote if we haven't voted or voted for this candidate
        let vote_granted = if persistent.voted_for.is_none()
            || persistent.voted_for == Some(request.candidate_id)
        {
            // Check if candidate's log is at least as up-to-date
            let log = self.log.read();
            let last_log_index = log.len() as u64;
            let last_log_term = log.last().map(|e| e.term).unwrap_or(Term(0));

            let log_ok = request.last_log_term > last_log_term
                || (request.last_log_term == last_log_term
                    && request.last_log_index.0 >= last_log_index);

            if log_ok {
                persistent.voted_for = Some(request.candidate_id);
                true
            } else {
                false
            }
        } else {
            false
        };

        Ok(RequestVoteResponse {
            term: persistent.current_term,
            vote_granted,
        })
    }

    /// Apply committed entries to the state machine
    async fn apply_committed_entries(&self) -> Result<()> {
        let commit_index = self.volatile.read().commit_index;

        loop {
            // Extract the command while holding the lock
            let command = {
                let mut volatile = self.volatile.write();

                if volatile.last_applied.0 >= commit_index.0 {
                    break;
                }

                volatile.last_applied.0 += 1;
                let entry = &self.log.read()[(volatile.last_applied.0 - 1) as usize];
                entry.command.clone()
            }; // Lock is dropped here

            // Apply command to state machine (without holding the lock)
            match command {
                Command::Put { cid_bytes, data } => {
                    // Reconstruct CID and Block
                    if let Ok(cid) = Cid::try_from(cid_bytes.as_slice()) {
                        let block = Block::from_parts(cid, bytes::Bytes::from(data));
                        self.store.put(&block).await?;
                        debug!("Applied PUT: {}", block.cid());
                    }
                }
                Command::Delete { cid_bytes } => {
                    // Deserialize CID from bytes
                    if let Ok(cid) = Cid::try_from(cid_bytes.as_slice()) {
                        self.store.delete(&cid).await?;
                        debug!("Applied DELETE: {}", cid);
                    }
                }
                Command::NoOp => {
                    debug!("Applied NoOp");
                }
            }
        }

        Ok(())
    }

    /// Append a new entry to the log (leader only)
    #[allow(clippy::unused_async)]
    pub async fn append_entry(&self, command: Command) -> Result<LogIndex> {
        let state = *self.state.read();
        if state != NodeState::Leader {
            return Err(ipfrs_core::Error::Internal("Not the leader".to_string()));
        }

        let mut log = self.log.write();
        let index = LogIndex((log.len() + 1) as u64);
        let term = self.persistent.read().current_term;

        let entry = LogEntry {
            term,
            index,
            command,
        };

        log.push(entry);
        Ok(index)
    }

    /// Get the current leader ID
    pub fn current_leader(&self) -> Option<NodeId> {
        *self.current_leader.read()
    }

    /// Check if this node is the leader
    pub fn is_leader(&self) -> bool {
        *self.state.read() == NodeState::Leader
    }

    /// Get current term
    pub fn current_term(&self) -> Term {
        self.persistent.read().current_term
    }
}

/// Statistics about the RAFT node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftStats {
    /// Node ID
    pub node_id: NodeId,
    /// Current state
    pub state: String,
    /// Current term
    pub term: Term,
    /// Current leader (if known)
    pub leader: Option<NodeId>,
    /// Log size
    pub log_size: usize,
    /// Commit index
    pub commit_index: LogIndex,
    /// Last applied index
    pub last_applied: LogIndex,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryBlockStore;

    #[tokio::test]
    async fn test_node_creation() {
        let store = MemoryBlockStore::new();
        let config = RaftConfig::default();
        let node = RaftNode::new(NodeId(1), vec![NodeId(2), NodeId(3)], store, config);
        assert!(node.is_ok());
    }

    #[tokio::test]
    async fn test_append_entries_lower_term() {
        let store = MemoryBlockStore::new();
        let config = RaftConfig::default();
        let node = RaftNode::new(NodeId(1), vec![NodeId(2), NodeId(3)], store, config).unwrap();

        // Set current term to 5
        node.persistent.write().current_term = Term(5);

        let request = AppendEntriesRequest {
            term: Term(3),
            leader_id: NodeId(2),
            prev_log_index: LogIndex(0),
            prev_log_term: Term(0),
            entries: vec![],
            leader_commit: LogIndex(0),
        };

        let response = node.handle_append_entries(request).await.unwrap();
        assert!(!response.success);
        assert_eq!(response.term, Term(5));
    }

    #[tokio::test]
    async fn test_request_vote_grant() {
        let store = MemoryBlockStore::new();
        let config = RaftConfig::default();
        let node = RaftNode::new(NodeId(1), vec![NodeId(2), NodeId(3)], store, config).unwrap();

        let request = RequestVoteRequest {
            term: Term(1),
            candidate_id: NodeId(2),
            last_log_index: LogIndex(0),
            last_log_term: Term(0),
        };

        let response = node.handle_request_vote(request).await.unwrap();
        assert!(response.vote_granted);
        assert_eq!(node.persistent.read().voted_for, Some(NodeId(2)));
    }

    #[tokio::test]
    async fn test_request_vote_deny_already_voted() {
        let store = MemoryBlockStore::new();
        let config = RaftConfig::default();
        let node = RaftNode::new(NodeId(1), vec![NodeId(2), NodeId(3)], store, config).unwrap();

        // Vote for node 2
        node.persistent.write().voted_for = Some(NodeId(2));
        node.persistent.write().current_term = Term(1);

        // Node 3 requests vote
        let request = RequestVoteRequest {
            term: Term(1),
            candidate_id: NodeId(3),
            last_log_index: LogIndex(0),
            last_log_term: Term(0),
        };

        let response = node.handle_request_vote(request).await.unwrap();
        assert!(!response.vote_granted);
    }
}
