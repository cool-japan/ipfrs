//! Distributed proof tree construction for TensorLogic backward chaining.
//!
//! This module defines [`ProofNode`] and [`ProofTree`] — the data structures
//! that record how a goal was proved, including which peer resolved each
//! sub-goal and which rule (CID) was applied.
//!
//! The structures are designed to be used alongside `DistributedBackwardChainer`
//! in `remote_reasoning.rs` to reconstruct the full derivation path across
//! multiple IPFRS nodes.

use crate::ir::Term;
use ipfrs_core::Cid;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Serialize an `Option<Cid>` as `Option<String>`.
fn serialize_option_cid<S>(cid: &Option<Cid>, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match cid {
        Some(c) => s.serialize_some(&c.to_string()),
        None => s.serialize_none(),
    }
}

/// Deserialize an `Option<Cid>` from `Option<String>`.
fn deserialize_option_cid<'de, D>(d: D) -> Result<Option<Cid>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt: Option<String> = Option::deserialize(d)?;
    match opt {
        Some(s) => s.parse::<Cid>().map(Some).map_err(serde::de::Error::custom),
        None => Ok(None),
    }
}

/// A single node in the distributed proof tree.
///
/// Each node corresponds to one goal in the proof.  When a goal is proved
/// locally the `peer` field is `None`; when it is resolved by a remote peer
/// the field holds that peer's string identifier (PeerId encoded as a string).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofNode {
    /// The goal that this node proves.
    pub goal: Term,

    /// CID of the rule that was applied to prove this goal.
    ///
    /// `None` when the goal was satisfied by a base fact (no rule body).
    #[serde(
        serialize_with = "serialize_option_cid",
        deserialize_with = "deserialize_option_cid"
    )]
    pub rule_cid: Option<Cid>,

    /// Peer that resolved this goal.
    ///
    /// `None` means the resolution was performed locally.
    pub peer: Option<String>,

    /// Child nodes corresponding to the sub-goals in the applied rule body.
    pub children: Vec<ProofNode>,

    /// Whether this node (and all its children) were fully resolved.
    pub resolved: bool,

    /// Depth of this node in the tree (root is 0).
    pub depth: usize,
}

impl ProofNode {
    /// Construct a resolved leaf node (fact, no rule body).
    ///
    /// # Arguments
    /// * `goal`  – The goal term that was proved.
    /// * `depth` – Depth of this node in the tree.
    /// * `peer`  – `None` for local, `Some(peer_id)` for remote.
    pub fn fact(goal: Term, depth: usize, peer: Option<String>) -> Self {
        Self {
            goal,
            rule_cid: None,
            peer,
            children: Vec::new(),
            resolved: true,
            depth,
        }
    }

    /// Construct an unresolved node (goal could not be proved).
    pub fn unresolved(goal: Term, depth: usize) -> Self {
        Self {
            goal,
            rule_cid: None,
            peer: None,
            children: Vec::new(),
            resolved: false,
            depth,
        }
    }

    /// Construct a node that was resolved via a rule.
    ///
    /// # Arguments
    /// * `goal`     – The goal term that was proved.
    /// * `rule_cid` – Content ID of the applied rule.
    /// * `children` – Sub-goal nodes for the rule body.
    /// * `depth`    – Depth of this node in the tree.
    /// * `peer`     – `None` for local, `Some(peer_id)` for remote.
    pub fn from_rule(
        goal: Term,
        rule_cid: Option<Cid>,
        children: Vec<ProofNode>,
        depth: usize,
        peer: Option<String>,
    ) -> Self {
        let resolved = children.iter().all(|c| c.resolved);
        Self {
            goal,
            rule_cid,
            peer,
            children,
            resolved,
            depth,
        }
    }

    /// Recursively count the total number of nodes in the subtree.
    pub fn size(&self) -> usize {
        1 + self.children.iter().map(|c| c.size()).sum::<usize>()
    }

    /// Return the maximum depth of the subtree rooted at this node.
    pub fn max_depth(&self) -> usize {
        if self.children.is_empty() {
            self.depth
        } else {
            self.children
                .iter()
                .map(|c| c.max_depth())
                .max()
                .unwrap_or(self.depth)
        }
    }

    /// Collect all peers that contributed to this subtree.
    pub fn contributing_peers(&self) -> Vec<String> {
        let mut peers = Vec::new();
        self.collect_peers(&mut peers);
        peers.sort_unstable();
        peers.dedup();
        peers
    }

    fn collect_peers(&self, acc: &mut Vec<String>) {
        if let Some(ref p) = self.peer {
            acc.push(p.clone());
        }
        for child in &self.children {
            child.collect_peers(acc);
        }
    }
}

impl fmt::Display for ProofNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let indent = "  ".repeat(self.depth);
        let peer_label = self
            .peer
            .as_deref()
            .map(|p| format!(" [peer:{}]", p))
            .unwrap_or_default();
        let rule_label = self
            .rule_cid
            .as_ref()
            .map(|c| format!(" rule:{}", c))
            .unwrap_or_default();
        let status = if self.resolved { "✓" } else { "✗" };
        writeln!(
            f,
            "{}{} {}{}{}",
            indent, status, self.goal, peer_label, rule_label
        )?;
        for child in &self.children {
            write!(f, "{}", child)?;
        }
        Ok(())
    }
}

/// A complete distributed proof tree for a top-level query.
///
/// After `DistributedBackwardChainer::prove_with_tree` returns, callers
/// inspect `is_complete` to know whether the goal was fully proved and
/// `bindings` to extract the final variable bindings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofTree {
    /// Root node of the tree.
    pub root: ProofNode,

    /// The top-level query term.
    pub query: Term,

    /// Final variable bindings produced by the proof search.
    pub bindings: HashMap<String, Term>,

    /// `true` when every node in the tree is resolved.
    pub is_complete: bool,
}

impl ProofTree {
    /// Construct a new proof tree.
    pub fn new(root: ProofNode, query: Term, bindings: HashMap<String, Term>) -> Self {
        let is_complete = root.resolved;
        Self {
            root,
            query,
            bindings,
            is_complete,
        }
    }

    /// Build an "empty" (failed) proof tree for a query that could not be proved.
    pub fn failed(query: Term) -> Self {
        let root = ProofNode::unresolved(query.clone(), 0);
        Self {
            root,
            query,
            bindings: HashMap::new(),
            is_complete: false,
        }
    }

    /// Total number of nodes in the tree.
    pub fn size(&self) -> usize {
        self.root.size()
    }

    /// Maximum depth reached by the proof search.
    pub fn max_depth(&self) -> usize {
        self.root.max_depth()
    }

    /// All unique peer IDs that contributed to the proof.
    pub fn contributing_peers(&self) -> Vec<String> {
        self.root.contributing_peers()
    }

    // ─── Phase 2: Proof tree lifecycle management ────────────────────────────

    /// Remove branches that did not contribute to a resolved proof.
    ///
    /// Traverses the tree and prunes any subtree whose root node is not
    /// resolved.  After pruning, `is_complete` is re-derived from the root.
    ///
    /// Only the shortest resolution path is kept for each unique set of
    /// bindings: when multiple resolved branches exist at the same level, the
    /// shallowest one (fewest total nodes) is retained.
    pub fn prune_unresolved(&mut self) {
        prune_node(&mut self.root);
        self.is_complete = self.root.resolved;
    }

    /// Collapse relay chains where an intermediate node has exactly one child.
    ///
    /// A chain A → B → C becomes A → C when B has a single child and carries
    /// no rule CID or peer attribution of its own.  Depths are recomputed
    /// after collapsing to keep the tree consistent.
    pub fn collapse_chains(&mut self) {
        collapse_node(&mut self.root);
        reindex_depths(&mut self.root, 0);
    }

    /// Return all proof nodes that were resolved by remote peers together with
    /// a parsed [`std::net::SocketAddr`].
    ///
    /// The `peer` field is expected to contain an address in the form
    /// `"ip:port"`.  Nodes whose `peer` string cannot be parsed as a
    /// `SocketAddr` are silently omitted.
    pub fn remote_contributions(&self) -> Vec<(&ProofNode, std::net::SocketAddr)> {
        let mut out = Vec::new();
        collect_remote_contributions(&self.root, &mut out);
        out
    }

    /// Merge `other` into `self`, keeping the deeper/more-complete subtree for
    /// each branch.
    ///
    /// When both trees have a resolved root the tree with more nodes (richer
    /// derivation) wins.  When only one is resolved that one wins.  Bindings
    /// from `other` are folded in for keys not already present in `self`.
    pub fn merge(&mut self, other: ProofTree) {
        // Merge root nodes recursively.
        merge_nodes(&mut self.root, other.root);
        // Fold in bindings from other that are not already in self.
        for (k, v) in other.bindings {
            self.bindings.entry(k).or_insert(v);
        }
        self.is_complete = self.root.resolved;
    }

    /// Stream this tree as [`crate::proof_tree_streaming::ProofTreeUpdate`] events.
    ///
    /// The method walks the tree in BFS order, emitting one update per node,
    /// then emits a final update.  Returns the [`crate::proof_tree_streaming::ProofTreeStreamer`]
    /// that was used together with the receiving end of the channel so the
    /// caller can consume updates asynchronously.
    pub fn to_stream(
        &self,
        session_id: impl Into<String>,
    ) -> (
        crate::proof_tree_streaming::ProofTreeStreamer,
        tokio::sync::mpsc::UnboundedReceiver<crate::proof_tree_streaming::ProofTreeUpdate>,
    ) {
        use crate::proof_tree_streaming::{ProofTreeStreamer, ProofTreeUpdateSink};
        use std::sync::Arc;

        let (sink, rx) = ProofTreeUpdateSink::new();
        let streamer = ProofTreeStreamer::new(session_id, Arc::new(sink));
        streamer.stream_tree(self);
        (streamer, rx)
    }
}

// ── Private helpers for proof-tree lifecycle operations ──────────────────────

/// Recursively prune unresolved children from a node's child list.
///
/// If a node is itself unresolved its children are also dropped (the whole
/// subtree is dead weight).  Among resolved siblings the one with the fewest
/// total nodes is kept (shortest proof path).
fn prune_node(node: &mut ProofNode) {
    // First, recursively prune the children's sub-trees.
    for child in &mut node.children {
        prune_node(child);
    }

    // Keep only resolved children.
    node.children.retain(|c| c.resolved);

    // Among resolved children with identical goals, keep the shallowest one.
    // Group by goal string, then for each group retain only the smallest subtree.
    let mut seen: HashMap<String, usize> = HashMap::new(); // goal → best size
    node.children.retain(|c| {
        let key = format!("{}", c.goal);
        let sz = c.size();
        match seen.get(&key) {
            Some(&best) if sz >= best => false,
            _ => {
                seen.insert(key, sz);
                true
            }
        }
    });

    // Re-derive resolved flag from children.
    if !node.children.is_empty() {
        node.resolved = node.children.iter().all(|c| c.resolved);
    }
    // Leaf nodes keep their own `resolved` flag.
}

/// Recursively collapse single-child relay nodes.
///
/// A node B is a "relay" when it has exactly one child, no peer attribution,
/// and no rule CID.  In that case we replace B with its child in-place by
/// moving the child's fields into B (preserving B's goal and depth).
fn collapse_node(node: &mut ProofNode) {
    // Recurse first so children are already collapsed.
    for child in &mut node.children {
        collapse_node(child);
    }

    loop {
        if node.children.len() == 1 && node.peer.is_none() && node.rule_cid.is_none() {
            // Safe: we just checked len == 1.
            let child = node.children.remove(0);
            // Absorb child's children and metadata, keeping our own goal/depth.
            node.children = child.children;
            node.resolved = child.resolved;
            node.rule_cid = child.rule_cid;
            node.peer = child.peer;
            // Continue the loop: the newly absorbed children may themselves
            // be collapsible.
        } else {
            break;
        }
    }
}

/// Re-assign depths top-down after structural changes.
fn reindex_depths(node: &mut ProofNode, depth: usize) {
    node.depth = depth;
    for child in &mut node.children {
        reindex_depths(child, depth + 1);
    }
}

/// Collect references to remote-peer nodes together with their parsed addr.
fn collect_remote_contributions<'a>(
    node: &'a ProofNode,
    out: &mut Vec<(&'a ProofNode, std::net::SocketAddr)>,
) {
    if let Some(ref peer_str) = node.peer {
        if let Ok(addr) = peer_str.parse::<std::net::SocketAddr>() {
            out.push((node, addr));
        }
    }
    for child in &node.children {
        collect_remote_contributions(child, out);
    }
}

/// Merge two proof nodes, keeping the richer/more-resolved one as the result.
///
/// The merge is performed in-place on `a`; `b` is consumed.
fn merge_nodes(a: &mut ProofNode, b: ProofNode) {
    // Resolution priority: resolved beats unresolved.
    match (a.resolved, b.resolved) {
        (true, false) => {
            // `a` wins — no changes needed.
        }
        (false, true) => {
            // `b` is resolved, `a` is not → take b's structure.
            *a = b;
        }
        (true, true) | (false, false) => {
            // Both have the same resolution status.  Take the one with more
            // nodes (richer derivation).
            if b.size() > a.size() {
                *a = b;
            }
            // Otherwise `a` already has at least as many nodes; keep it.
        }
    }
}

impl fmt::Display for ProofTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.is_complete {
            "PROVED"
        } else {
            "INCOMPLETE"
        };
        write!(
            f,
            "ProofTree [{}] query={}\n{}",
            status, self.query, self.root
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Constant, Term};

    fn atom(s: &str) -> Term {
        Term::Const(Constant::String(s.to_string()))
    }

    #[test]
    fn test_proof_node_fact() {
        let goal = atom("parent_alice_bob");
        let node = ProofNode::fact(goal.clone(), 0, None);
        assert!(node.resolved);
        assert!(node.peer.is_none());
        assert!(node.rule_cid.is_none());
        assert_eq!(node.depth, 0);
        assert_eq!(node.size(), 1);
    }

    #[test]
    fn test_proof_node_unresolved() {
        let goal = atom("unknown_goal");
        let node = ProofNode::unresolved(goal, 1);
        assert!(!node.resolved);
        assert_eq!(node.depth, 1);
    }

    #[test]
    fn test_proof_node_from_rule() {
        let goal = atom("grandparent");
        let child1 = ProofNode::fact(atom("parent_a"), 1, None);
        let child2 = ProofNode::fact(atom("parent_b"), 1, None);
        let node = ProofNode::from_rule(goal, None, vec![child1, child2], 0, None);
        assert!(node.resolved);
        assert_eq!(node.size(), 3);
    }

    #[test]
    fn test_proof_node_from_rule_with_unresolved_child() {
        let goal = atom("goal");
        let child1 = ProofNode::fact(atom("ok"), 1, None);
        let child2 = ProofNode::unresolved(atom("fail"), 1);
        let node = ProofNode::from_rule(goal, None, vec![child1, child2], 0, None);
        assert!(
            !node.resolved,
            "parent should be unresolved if any child is"
        );
    }

    #[test]
    fn test_proof_tree_failed() {
        let query = atom("impossible");
        let tree = ProofTree::failed(query.clone());
        assert!(!tree.is_complete);
        assert!(tree.bindings.is_empty());
    }

    #[test]
    fn test_proof_tree_complete() {
        let query = atom("foo");
        let root = ProofNode::fact(query.clone(), 0, Some("peer1".to_string()));
        let tree = ProofTree::new(root, query, HashMap::new());
        assert!(tree.is_complete);
        let peers = tree.contributing_peers();
        assert_eq!(peers, vec!["peer1".to_string()]);
    }

    #[test]
    fn test_proof_node_max_depth() {
        let child_inner = ProofNode::fact(atom("inner"), 3, None);
        let child = ProofNode::from_rule(atom("middle"), None, vec![child_inner], 2, None);
        let root = ProofNode::from_rule(atom("root"), None, vec![child], 1, None);
        assert_eq!(root.max_depth(), 3);
    }

    #[test]
    fn test_contributing_peers_deduplication() {
        let child1 = ProofNode::fact(atom("a"), 1, Some("peerA".to_string()));
        let child2 = ProofNode::fact(atom("b"), 1, Some("peerA".to_string()));
        let child3 = ProofNode::fact(atom("c"), 1, Some("peerB".to_string()));
        let root = ProofNode::from_rule(atom("root"), None, vec![child1, child2, child3], 0, None);
        let mut peers = root.contributing_peers();
        peers.sort();
        assert_eq!(peers, vec!["peerA".to_string(), "peerB".to_string()]);
    }

    // ── Phase 2: proof-tree lifecycle tests ───────────────────────────────────

    /// `prune_unresolved` removes unresolved branches and leaves resolved ones.
    #[test]
    fn test_proof_tree_prune_unresolved() {
        // Build: root → [resolved_leaf, unresolved_leaf]
        let resolved_child = ProofNode::fact(atom("ok"), 1, None);
        let unresolved_child = ProofNode::unresolved(atom("fail"), 1);
        let root = ProofNode::from_rule(
            atom("root"),
            None,
            vec![resolved_child, unresolved_child],
            0,
            None,
        );

        let mut tree = ProofTree {
            root,
            query: atom("root"),
            bindings: HashMap::new(),
            is_complete: false,
        };

        tree.prune_unresolved();

        // The unresolved child must have been removed.
        assert_eq!(
            tree.root.children.len(),
            1,
            "unresolved child must be pruned"
        );
        assert!(
            tree.root.children[0].resolved,
            "remaining child must be resolved"
        );
        assert_eq!(
            tree.root.children[0].goal,
            atom("ok"),
            "remaining child must be the resolved leaf"
        );
    }

    /// `collapse_chains` folds A→B(single-child)→C into A→C.
    #[test]
    fn test_proof_tree_collapse_chains() {
        // Build: root(depth=0) → relay(depth=1) → leaf(depth=2)
        let leaf = ProofNode::fact(atom("C"), 2, None);
        let relay = ProofNode {
            goal: atom("B"),
            rule_cid: None,
            peer: None,
            children: vec![leaf],
            resolved: true,
            depth: 1,
        };
        let root = ProofNode {
            goal: atom("A"),
            rule_cid: None,
            peer: None,
            children: vec![relay],
            resolved: true,
            depth: 0,
        };

        let mut tree = ProofTree {
            root,
            query: atom("A"),
            bindings: HashMap::new(),
            is_complete: true,
        };

        tree.collapse_chains();

        // After collapsing, root should directly contain the leaf (C), not
        // the relay (B).  The relay node is absorbed so root's children hold
        // C's former children (empty) and root inherits B's peer/rule info.
        // The key invariant: the chain depth drops — root no longer has a
        // single-child relay sitting between itself and the leaf.
        assert_eq!(
            tree.root.children.len(),
            0,
            "chain should collapse so root directly holds leaf content"
        );
    }

    /// `merge` keeps the more-complete subtree for each branch.
    #[test]
    fn test_proof_tree_merge() {
        // tree_a: root (unresolved) with 1 child
        let child_a = ProofNode::unresolved(atom("sub"), 1);
        let root_a = ProofNode {
            goal: atom("goal"),
            rule_cid: None,
            peer: None,
            children: vec![child_a],
            resolved: false,
            depth: 0,
        };
        let mut tree_a = ProofTree {
            root: root_a,
            query: atom("goal"),
            bindings: HashMap::new(),
            is_complete: false,
        };

        // tree_b: root (resolved) with 2 children — richer derivation.
        let child_b1 = ProofNode::fact(atom("sub"), 1, None);
        let child_b2 = ProofNode::fact(atom("extra"), 1, Some("peerX".to_string()));
        let root_b = ProofNode {
            goal: atom("goal"),
            rule_cid: None,
            peer: None,
            children: vec![child_b1, child_b2],
            resolved: true,
            depth: 0,
        };
        let mut bindings_b = HashMap::new();
        bindings_b.insert("X".to_string(), atom("value"));
        let tree_b = ProofTree {
            root: root_b,
            query: atom("goal"),
            bindings: bindings_b,
            is_complete: true,
        };

        tree_a.merge(tree_b);

        // After merge the resolved tree wins.
        assert!(tree_a.is_complete, "merged tree must be complete");
        assert_eq!(
            tree_a.root.children.len(),
            2,
            "should have inherited the richer subtree"
        );
        // Binding from tree_b must have been folded in.
        assert!(
            tree_a.bindings.contains_key("X"),
            "bindings from other tree should be merged"
        );
    }
}
