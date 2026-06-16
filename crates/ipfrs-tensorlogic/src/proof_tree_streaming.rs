//! Distributed proof tree streaming via incremental update events.
//!
//! This module provides [`ProofTreeStreamer`] which emits [`ProofTreeUpdate`]
//! events as nodes of a [`crate::proof_tree::ProofTree`] resolve.  Updates are
//! sent through an unbounded MPSC channel so consumers can process them
//! asynchronously or aggregate them into a [`ProofTreeStreamSummary`].
//!
//! ## Design
//!
//! - Each derivation session is identified by a `session_id` string shared
//!   across all updates.
//! - Sequence numbers (`seq`) are assigned atomically and monotonically so
//!   receivers can detect gaps or re-ordering.
//! - A "final" update (`is_final = true`) signals that the session is
//!   complete.  [`ProofTreeStreamSummary::apply`] sets `is_complete = true`
//!   when it encounters such an update.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

// ─── Public update event ─────────────────────────────────────────────────────

/// A single update event in an ongoing proof tree derivation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProofTreeUpdate {
    /// Session identifier shared across all updates for this derivation.
    pub session_id: String,
    /// The goal being proved at this update, rendered as a `Display` string.
    pub goal: String,
    /// Depth of this node in the proof tree.
    pub depth: usize,
    /// Whether this node resolved successfully.
    pub resolved: bool,
    /// The peer that resolved this node (`None` = local).
    pub peer: Option<String>,
    /// Sequence number within this session (monotonically increasing).
    pub seq: u64,
    /// `true` if this is the final update for the session.
    pub is_final: bool,
}

// ─── ProofTreeUpdateSink ──────────────────────────────────────────────────────

/// Receives streamed proof tree updates via an unbounded channel.
pub struct ProofTreeUpdateSink {
    sender: UnboundedSender<ProofTreeUpdate>,
}

impl ProofTreeUpdateSink {
    /// Create a new sink together with the receiving end of the channel.
    pub fn new() -> (Self, UnboundedReceiver<ProofTreeUpdate>) {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        (Self { sender }, receiver)
    }

    /// Send an update.  Silently drops the update if the channel is closed.
    pub fn send(&self, update: ProofTreeUpdate) {
        // Ignore send errors — the receiver may have been dropped intentionally.
        let _ = self.sender.send(update);
    }

    /// Returns `true` if the receiving end of the channel has been dropped.
    pub fn is_closed(&self) -> bool {
        self.sender.is_closed()
    }
}

// ─── ProofTreeStreamer ────────────────────────────────────────────────────────

/// Streams partial proof tree updates as nodes resolve.
pub struct ProofTreeStreamer {
    session_id: String,
    seq: AtomicU64,
    sink: Arc<ProofTreeUpdateSink>,
}

impl ProofTreeStreamer {
    /// Create a new streamer bound to `session_id` and `sink`.
    pub fn new(session_id: impl Into<String>, sink: Arc<ProofTreeUpdateSink>) -> Self {
        Self {
            session_id: session_id.into(),
            seq: AtomicU64::new(0),
            sink,
        }
    }

    /// Return the session identifier for this streamer.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Emit an update for a successfully resolved node.
    pub fn emit_resolved(&self, goal: &str, depth: usize, peer: Option<String>) {
        let seq = self.next_seq();
        self.sink.send(ProofTreeUpdate {
            session_id: self.session_id.clone(),
            goal: goal.to_string(),
            depth,
            resolved: true,
            peer,
            seq,
            is_final: false,
        });
    }

    /// Emit an update for a node that could not be resolved.
    pub fn emit_unresolved(&self, goal: &str, depth: usize) {
        let seq = self.next_seq();
        self.sink.send(ProofTreeUpdate {
            session_id: self.session_id.clone(),
            goal: goal.to_string(),
            depth,
            resolved: false,
            peer: None,
            seq,
            is_final: false,
        });
    }

    /// Emit the final summary update, marking the session as complete.
    pub fn emit_final(&self, goal: &str, depth: usize, resolved: bool, peer: Option<String>) {
        let seq = self.next_seq();
        self.sink.send(ProofTreeUpdate {
            session_id: self.session_id.clone(),
            goal: goal.to_string(),
            depth,
            resolved,
            peer,
            seq,
            is_final: true,
        });
    }

    /// Walk a completed [`crate::proof_tree::ProofTree`] and emit one update
    /// per node in BFS (pre-order breadth-first) order, followed by a final
    /// update that reflects the overall tree completion status.
    pub fn stream_tree(&self, tree: &crate::proof_tree::ProofTree) {
        let mut queue: VecDeque<&crate::proof_tree::ProofNode> = VecDeque::new();
        queue.push_back(&tree.root);

        while let Some(node) = queue.pop_front() {
            let goal_str = format!("{}", node.goal);
            if node.resolved {
                self.emit_resolved(&goal_str, node.depth, node.peer.clone());
            } else {
                self.emit_unresolved(&goal_str, node.depth);
            }
            for child in &node.children {
                queue.push_back(child);
            }
        }

        // Emit the final event summarising the whole tree.
        let root_goal = format!("{}", tree.query);
        self.emit_final(&root_goal, 0, tree.is_complete, None);
    }

    /// Return the current sequence counter value (number of updates emitted so far).
    pub fn seq(&self) -> u64 {
        self.seq.load(Ordering::SeqCst)
    }

    // ── private ──────────────────────────────────────────────────────────────

    fn next_seq(&self) -> u64 {
        // fetch_add returns the *old* value, so seq() will equal the number of
        // updates that have been emitted once the method returns.
        self.seq.fetch_add(1, Ordering::SeqCst)
    }
}

// ─── ProofTreeStreamSummary ───────────────────────────────────────────────────

/// Aggregated statistics collected by folding a stream of [`ProofTreeUpdate`]s.
#[derive(Debug, Clone)]
pub struct ProofTreeStreamSummary {
    /// Session this summary belongs to.
    pub session_id: String,
    /// Total number of updates applied so far.
    pub total_updates: u64,
    /// Number of updates with `resolved = true`.
    pub resolved_count: u64,
    /// Number of updates with `resolved = false`.
    pub unresolved_count: u64,
    /// Unique peer IDs seen in updates (deduplicated on demand via
    /// [`ProofTreeStreamSummary::dedup_peers`]).
    pub contributing_peers: Vec<String>,
    /// Maximum `depth` value seen across all updates.
    pub max_depth: usize,
    /// `true` once an update with `is_final = true` has been applied.
    pub is_complete: bool,
}

impl ProofTreeStreamSummary {
    /// Construct an empty summary for `session_id`.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            total_updates: 0,
            resolved_count: 0,
            unresolved_count: 0,
            contributing_peers: Vec::new(),
            max_depth: 0,
            is_complete: false,
        }
    }

    /// Fold a single [`ProofTreeUpdate`] into this summary.
    pub fn apply(&mut self, update: &ProofTreeUpdate) {
        self.total_updates += 1;

        if update.resolved {
            self.resolved_count += 1;
        } else {
            self.unresolved_count += 1;
        }

        if let Some(ref peer) = update.peer {
            self.contributing_peers.push(peer.clone());
        }

        if update.depth > self.max_depth {
            self.max_depth = update.depth;
        }

        if update.is_final {
            self.is_complete = true;
        }
    }

    /// Sort and deduplicate `contributing_peers` in place.
    pub fn dedup_peers(&mut self) {
        self.contributing_peers.sort_unstable();
        self.contributing_peers.dedup();
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Term;
    use crate::proof_tree::{ProofNode, ProofTree};
    use std::collections::HashMap;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_streamer(id: &str) -> (ProofTreeStreamer, UnboundedReceiver<ProofTreeUpdate>) {
        let (sink, rx) = ProofTreeUpdateSink::new();
        let streamer = ProofTreeStreamer::new(id, Arc::new(sink));
        (streamer, rx)
    }

    fn atom(s: &str) -> Term {
        Term::Const(crate::ir::Constant::String(s.to_string()))
    }

    /// Drain all currently pending messages from an unbounded receiver without
    /// blocking.
    fn drain(rx: &mut UnboundedReceiver<ProofTreeUpdate>) -> Vec<ProofTreeUpdate> {
        let mut updates = Vec::new();
        while let Ok(u) = rx.try_recv() {
            updates.push(u);
        }
        updates
    }

    // ── individual emit tests ─────────────────────────────────────────────────

    #[test]
    fn test_emit_resolved_increments_seq() {
        let (streamer, mut rx) = make_streamer("session-1");
        streamer.emit_resolved("a(X)", 0, None);
        streamer.emit_resolved("b(X)", 1, None);
        streamer.emit_resolved("c(X)", 2, None);

        let updates = drain(&mut rx);
        assert_eq!(updates.len(), 3);
        assert_eq!(streamer.seq(), 3);
        assert_eq!(updates[0].seq, 0);
        assert_eq!(updates[1].seq, 1);
        assert_eq!(updates[2].seq, 2);
    }

    #[test]
    fn test_emit_unresolved() {
        let (streamer, mut rx) = make_streamer("session-2");
        streamer.emit_unresolved("fail(X)", 1);

        let updates = drain(&mut rx);
        assert_eq!(updates.len(), 1);
        assert!(
            !updates[0].resolved,
            "unresolved update should have resolved=false"
        );
        assert!(updates[0].peer.is_none());
        assert!(!updates[0].is_final);
    }

    #[test]
    fn test_emit_final_is_final() {
        let (streamer, mut rx) = make_streamer("session-3");
        streamer.emit_final("query(a)", 0, true, None);

        let updates = drain(&mut rx);
        assert_eq!(updates.len(), 1);
        assert!(updates[0].is_final, "final update must have is_final=true");
        assert!(updates[0].resolved);
    }

    // ── stream_tree tests ─────────────────────────────────────────────────────

    #[test]
    fn test_stream_tree_bfs_order() {
        // Build: root(goal=A) -> child1(B), child2(C)
        let child1 = ProofNode::fact(atom("B"), 1, None);
        let child2 = ProofNode::fact(atom("C"), 1, None);
        let root = ProofNode {
            goal: atom("A"),
            rule_cid: None,
            peer: None,
            children: vec![child1, child2],
            resolved: true,
            depth: 0,
        };
        let tree = ProofTree {
            root,
            query: atom("A"),
            bindings: HashMap::new(),
            is_complete: true,
        };

        let (streamer, mut rx) = make_streamer("session-4");
        streamer.stream_tree(&tree);

        let updates = drain(&mut rx);
        // 3 nodes + 1 final = 4 updates
        assert_eq!(
            updates.len(),
            4,
            "expected 3 node updates + 1 final, got {}",
            updates.len()
        );

        // First three should not be final
        for u in &updates[..3] {
            assert!(!u.is_final, "only the 4th update should be final");
        }
        // Last must be final
        assert!(updates[3].is_final, "last update must be final");
    }

    // ── summary tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_summary_apply_counts() {
        let mut summary = ProofTreeStreamSummary::new("session-5");

        let make = |resolved: bool, is_final: bool| ProofTreeUpdate {
            session_id: "session-5".to_string(),
            goal: "g".to_string(),
            depth: 0,
            resolved,
            peer: None,
            seq: 0,
            is_final,
        };

        for _ in 0..3 {
            summary.apply(&make(true, false));
        }
        for _ in 0..2 {
            summary.apply(&make(false, false));
        }

        assert_eq!(summary.total_updates, 5);
        assert_eq!(summary.resolved_count, 3);
        assert_eq!(summary.unresolved_count, 2);
        assert!(!summary.is_complete, "no final update yet");
    }

    #[test]
    fn test_summary_contributing_peers() {
        let mut summary = ProofTreeStreamSummary::new("session-6");

        let peers = ["peer-A", "peer-B", "peer-A", "peer-C", "peer-B"];
        for peer in &peers {
            summary.apply(&ProofTreeUpdate {
                session_id: "session-6".to_string(),
                goal: "g".to_string(),
                depth: 0,
                resolved: true,
                peer: Some(peer.to_string()),
                seq: 0,
                is_final: false,
            });
        }

        summary.dedup_peers();
        assert_eq!(
            summary.contributing_peers.len(),
            3,
            "dedup should leave 3 unique peers"
        );
        assert_eq!(
            summary.contributing_peers,
            vec!["peer-A", "peer-B", "peer-C"]
        );
    }

    #[test]
    fn test_summary_is_complete_after_final() {
        let mut summary = ProofTreeStreamSummary::new("session-7");

        let non_final = ProofTreeUpdate {
            session_id: "session-7".to_string(),
            goal: "g".to_string(),
            depth: 0,
            resolved: true,
            peer: None,
            seq: 0,
            is_final: false,
        };
        summary.apply(&non_final);
        assert!(!summary.is_complete, "should not be complete before final");

        let final_update = ProofTreeUpdate {
            is_final: true,
            seq: 1,
            ..non_final
        };
        summary.apply(&final_update);
        assert!(summary.is_complete, "should be complete after final update");
    }

    #[test]
    fn test_sink_closed_no_panic() {
        let (sink, rx) = ProofTreeUpdateSink::new();
        // Drop the receiver to close the channel.
        drop(rx);

        let streamer = ProofTreeStreamer::new("session-8", Arc::new(sink));
        // Sending after close must not panic.
        streamer.emit_resolved("goal", 0, None);
        streamer.emit_unresolved("goal", 0);
        streamer.emit_final("goal", 0, true, None);
        // If we get here without panicking the test passes.
    }

    #[test]
    fn test_session_id_propagated() {
        let session = "my-unique-session-id";
        let (streamer, mut rx) = make_streamer(session);

        streamer.emit_resolved("a", 0, None);
        streamer.emit_unresolved("b", 1);
        streamer.emit_final("c", 0, false, None);

        let updates = drain(&mut rx);
        for u in &updates {
            assert_eq!(
                u.session_id, session,
                "all updates must carry the session_id '{}'",
                session
            );
        }
    }

    #[test]
    fn test_max_depth_tracking() {
        let mut summary = ProofTreeStreamSummary::new("session-9");

        let depths = [0usize, 3, 7, 2, 5];
        for (i, &d) in depths.iter().enumerate() {
            summary.apply(&ProofTreeUpdate {
                session_id: "session-9".to_string(),
                goal: "g".to_string(),
                depth: d,
                resolved: true,
                peer: None,
                seq: i as u64,
                is_final: false,
            });
        }

        assert_eq!(summary.max_depth, 7, "max depth should be 7");
    }
}
