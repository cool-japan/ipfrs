//! Multi-hop rule resolution with loop detection for the IPFRS distributed logic engine.
//!
//! This module extends distributed backward chaining with multi-hop traversal across
//! peer nodes, tracking which `(peer_id, goal_hash)` pairs have been visited to
//! prevent infinite loops in circular rule configurations.
//!
//! # Overview
//!
//! The central entry point is [`MultiHopResolver::resolve`], which:
//!
//! 1. Attempts local KB resolution (facts + one rule level) first.
//! 2. On failure, discovers remote peers via the `find_providers` callback.
//! 3. Delegates the goal to each peer via `remote_query`, recording each hop in a
//!    [`HopTrace`].
//! 4. Returns a [`MultiHopResult`] indicating whether the goal was resolved, the
//!    full hop trace, and any top-level variable bindings.
//!
//! Loop detection is handled by [`VisitedSet`]: before each local or remote
//! attempt the `(peer_id, goal_hash)` pair is inserted; if it already exists
//! the attempt is skipped and `false` is returned immediately.

use crate::ir::{KnowledgeBase, Predicate, Term};
use crate::reasoning::{apply_subst_predicate, rename_rule_vars, unify_predicates, Substitution};
use futures::future::BoxFuture;
use ipfrs_core::Cid;
use std::collections::{HashMap, HashSet};

// ── FNV-1a hash ───────────────────────────────────────────────────────────────

/// Compute a simple FNV-1a 64-bit hash of the `Debug` representation of `term`.
///
/// FNV-1a is deterministic, allocation-free on the hash side, and fast enough
/// for the visited-set use case without pulling in extra dependencies.
fn fnv1a_hash_term(term: &Term) -> u64 {
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;

    let repr = format!("{:?}", term);
    let mut hash = OFFSET_BASIS;
    for byte in repr.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

// ── VisitedSet ────────────────────────────────────────────────────────────────

/// Tracks which `(peer_id, goal_hash)` pairs have been visited to prevent loops.
///
/// When a `(peer_id, goal)` pair is first visited `try_visit` returns `true`.
/// Subsequent visits return `false`, signalling a loop.
pub struct VisitedSet {
    entries: HashSet<(String, u64)>,
    max_entries: usize,
}

impl VisitedSet {
    /// Create a new [`VisitedSet`] that holds at most `max_entries` pairs.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashSet::new(),
            max_entries,
        }
    }

    /// Attempt to record a visit for `(peer_id, goal)`.
    ///
    /// Returns `true` if this is the **first** visit (no loop), or `false` if
    /// the pair has already been seen (loop detected) **or** the set is full.
    pub fn try_visit(&mut self, peer_id: &str, goal: &Term) -> bool {
        if self.entries.len() >= self.max_entries {
            return false;
        }
        let hash = fnv1a_hash_term(goal);
        self.entries.insert((peer_id.to_string(), hash))
    }

    /// Return the number of recorded visits.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if no visits have been recorded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Check whether `(peer_id, goal)` is already in the visited set without
    /// modifying it.
    pub fn contains(&self, peer_id: &str, goal: &Term) -> bool {
        let hash = fnv1a_hash_term(goal);
        self.entries.contains(&(peer_id.to_string(), hash))
    }
}

// ── HopRecord ─────────────────────────────────────────────────────────────────

/// Records one hop in a multi-hop derivation chain.
#[derive(Debug, Clone)]
pub struct HopRecord {
    /// Index in the chain: `0` = local, `1` = first remote, and so on.
    pub hop_index: usize,
    /// `None` for local resolution; `Some(peer_id)` for remote.
    pub peer_id: Option<String>,
    /// Display representation of the goal term at this hop.
    pub goal: String,
    /// Whether this hop resolved the goal.
    pub resolved: bool,
}

// ── HopTrace ──────────────────────────────────────────────────────────────────

/// Complete trace of all hops taken during a multi-hop resolution attempt.
#[derive(Debug, Clone)]
pub struct HopTrace {
    hops: Vec<HopRecord>,
}

impl HopTrace {
    /// Create an empty trace.
    pub fn new() -> Self {
        Self { hops: Vec::new() }
    }

    /// Append a [`HopRecord`] to the trace.
    pub fn push(&mut self, record: HopRecord) {
        self.hops.push(record);
    }

    /// Number of recorded hops.
    pub fn len(&self) -> usize {
        self.hops.len()
    }

    /// Return `true` if the trace is empty.
    pub fn is_empty(&self) -> bool {
        self.hops.is_empty()
    }

    /// Slice over all recorded hops.
    pub fn hops(&self) -> &[HopRecord] {
        &self.hops
    }

    /// References to hops that involved a remote peer (`peer_id` is `Some`).
    pub fn remote_hops(&self) -> Vec<&HopRecord> {
        self.hops.iter().filter(|h| h.peer_id.is_some()).collect()
    }

    /// Total number of hops (alias for `len`).
    pub fn max_depth(&self) -> usize {
        self.hops.len()
    }

    /// Number of hops whose `resolved` field is `true`.
    pub fn resolved_count(&self) -> usize {
        self.hops.iter().filter(|h| h.resolved).count()
    }

    /// Deduplicated list of all remote peer IDs present in the trace.
    pub fn all_peers(&self) -> Vec<String> {
        let mut seen: HashSet<String> = HashSet::new();
        let mut peers: Vec<String> = Vec::new();
        for hop in &self.hops {
            if let Some(ref pid) = hop.peer_id {
                if seen.insert(pid.clone()) {
                    peers.push(pid.clone());
                }
            }
        }
        peers
    }
}

impl Default for HopTrace {
    fn default() -> Self {
        Self::new()
    }
}

// ── MultiHopConfig ────────────────────────────────────────────────────────────

/// Configuration governing multi-hop resolution behaviour.
pub struct MultiHopConfig {
    /// Maximum number of remote hops before giving up (default: 5).
    pub max_hops: usize,
    /// Maximum local backward-chaining depth per hop (default: 10).
    pub max_depth: usize,
    /// Maximum number of `(peer_id, goal)` entries in the visited set (default: 1000).
    pub max_visited: usize,
    /// Maximum number of remote peers to contact per unresolved sub-goal (default: 3).
    pub max_remote_peers: usize,
    /// Per-hop timeout in milliseconds (default: 15 000).
    pub timeout_ms: u64,
}

impl Default for MultiHopConfig {
    fn default() -> Self {
        Self {
            max_hops: 5,
            max_depth: 10,
            max_visited: 1000,
            max_remote_peers: 3,
            timeout_ms: 15_000,
        }
    }
}

// ── MultiHopResult ────────────────────────────────────────────────────────────

/// Result of a multi-hop resolution attempt.
pub struct MultiHopResult {
    /// Whether the goal was ultimately resolved.
    pub resolved: bool,
    /// Full hop trace.
    pub trace: HopTrace,
    /// Top-level variable bindings collected during resolution.
    pub bindings: HashMap<String, Term>,
}

// ── Local resolution helper ───────────────────────────────────────────────────

/// Try to resolve `goal` against `kb` using a simple single-pass backward chain.
///
/// This is synchronous — it checks:
/// 1. Whether any fact in the KB unifies with `goal` directly.
/// 2. Whether any rule head unifies with `goal` **and** every body predicate is
///    itself satisfiable by a fact in the KB (one level of rule application).
fn try_local(goal: &Term, kb: &KnowledgeBase, max_depth: usize) -> bool {
    try_local_recursive(goal, kb, 0, max_depth)
}

fn try_local_recursive(goal: &Term, kb: &KnowledgeBase, depth: usize, max_depth: usize) -> bool {
    if depth > max_depth {
        return false;
    }

    let pred = match term_to_predicate_local(goal) {
        Some(p) => p,
        None => return goal.is_ground(),
    };

    let empty_subst = Substitution::new();
    let pred = apply_subst_predicate(&pred, &empty_subst);

    // 1. Try local facts
    let facts: Vec<_> = kb.get_predicates(&pred.name).into_iter().cloned().collect();
    for fact in &facts {
        if unify_predicates(&pred, fact, &empty_subst).is_some() {
            return true;
        }
    }

    // 2. Try local rules (one level deep)
    let rules: Vec<_> = kb.get_rules(&pred.name).into_iter().cloned().collect();
    for rule in &rules {
        let renamed = rename_rule_vars(rule, depth);
        if let Some(_new_subst) = unify_predicates(&pred, &renamed.head, &empty_subst) {
            // Check every body predicate is resolvable
            let all_body_ok = renamed.body.iter().all(|body_pred| {
                let body_term = Term::Fun(body_pred.name.clone(), body_pred.args.clone());
                try_local_recursive(&body_term, kb, depth + 1, max_depth)
            });
            if all_body_ok {
                return true;
            }
        }
    }

    false
}

fn term_to_predicate_local(term: &Term) -> Option<Predicate> {
    match term {
        Term::Fun(name, args) => Some(Predicate::new(name.clone(), args.clone())),
        Term::Const(crate::ir::Constant::String(s)) => Some(Predicate::new(s.clone(), Vec::new())),
        _ => None,
    }
}

// ── MultiHopResolver ─────────────────────────────────────────────────────────

/// Bundles mutable traversal state with the two async callbacks so that
/// `resolve_inner` stays within clippy's argument-count limit.
struct ResolveCtx<'a, FP, FQ> {
    visited: &'a mut VisitedSet,
    trace: &'a mut HopTrace,
    find_providers: &'a FP,
    remote_query: &'a FQ,
}

/// Multi-hop backward-chaining resolver with loop detection.
///
/// Unlike [`crate::distributed_backward_chainer::DistributedBackwardChainer`],
/// which records a full [`crate::proof_tree::ProofTree`], this resolver focuses
/// on *how many hops* were taken and whether any hop triggered a loop.
pub struct MultiHopResolver {
    config: MultiHopConfig,
}

impl MultiHopResolver {
    /// Create a resolver with the given configuration.
    pub fn new(config: MultiHopConfig) -> Self {
        Self { config }
    }

    /// Attempt to resolve `goal` using multi-hop backward chaining.
    ///
    /// The two callbacks follow the same conventions as
    /// `DistributedBackwardChainer::prove_with_tree`:
    ///
    /// - `find_providers(cid)` → list of peer IDs that might hold rules for `cid`.
    /// - `remote_query(peer_id, goal)` → `Some(bindings)` on success, `None` on
    ///   failure / timeout.
    ///
    /// Both callbacks take owned values and return `'static` futures so that
    /// they can be used inside the recursive async implementation without
    /// complex higher-ranked lifetime constraints.
    pub async fn resolve<FP, FQ>(
        &self,
        goal: &Term,
        local_kb: &KnowledgeBase,
        find_providers: &FP,
        remote_query: &FQ,
    ) -> MultiHopResult
    where
        FP: Fn(Cid) -> BoxFuture<'static, Vec<String>> + Send + Sync,
        FQ: Fn(String, Term) -> BoxFuture<'static, Option<Vec<HashMap<String, Term>>>>
            + Send
            + Sync,
    {
        let mut visited = VisitedSet::new(self.config.max_visited);
        let mut trace = HopTrace::new();

        let resolved = {
            let mut ctx = ResolveCtx {
                visited: &mut visited,
                trace: &mut trace,
                find_providers,
                remote_query,
            };
            self.resolve_inner(goal, local_kb, 0, &mut ctx).await
        };

        MultiHopResult {
            resolved,
            trace,
            bindings: HashMap::new(),
        }
    }

    // Recursive inner implementation — each call represents one "hop level".
    //
    // Mutable traversal state (`visited`, `trace`) and the two async callbacks
    // are bundled in `ResolveCtx` to keep the argument count within clippy's
    // default limit of 7.
    fn resolve_inner<'a, FP, FQ>(
        &'a self,
        goal: &'a Term,
        kb: &'a KnowledgeBase,
        hop: usize,
        ctx: &'a mut ResolveCtx<'a, FP, FQ>,
    ) -> BoxFuture<'a, bool>
    where
        FP: Fn(Cid) -> BoxFuture<'static, Vec<String>> + Send + Sync,
        FQ: Fn(String, Term) -> BoxFuture<'static, Option<Vec<HashMap<String, Term>>>>
            + Send
            + Sync,
    {
        Box::pin(async move {
            // Depth guard
            if hop > self.config.max_hops {
                return false;
            }

            // Loop detection for local attempt
            if !ctx.visited.try_visit("local", goal) {
                return false;
            }

            // 1. Try local KB
            if try_local(goal, kb, self.config.max_depth) {
                ctx.trace.push(HopRecord {
                    hop_index: hop,
                    peer_id: None,
                    goal: format!("{:?}", goal),
                    resolved: true,
                });
                return true;
            }

            // 2. Find remote providers via rule-CID index
            let peer_ids = self.collect_peer_ids(goal, kb, ctx.find_providers).await;

            // 3. Try each peer
            for peer_id in peer_ids {
                // Loop detection: same goal must not be sent to the same peer twice
                if !ctx.visited.try_visit(&peer_id, goal) {
                    continue;
                }

                let result = (ctx.remote_query)(peer_id.clone(), goal.clone()).await;
                if let Some(bindings_list) = result {
                    if !bindings_list.is_empty() {
                        ctx.trace.push(HopRecord {
                            hop_index: hop,
                            peer_id: Some(peer_id),
                            goal: format!("{:?}", goal),
                            resolved: true,
                        });
                        return true;
                    }
                }

                ctx.trace.push(HopRecord {
                    hop_index: hop,
                    peer_id: Some(peer_id),
                    goal: format!("{:?}", goal),
                    resolved: false,
                });
            }

            // Nothing worked at this hop level
            ctx.trace.push(HopRecord {
                hop_index: hop,
                peer_id: None,
                goal: format!("{:?}", goal),
                resolved: false,
            });
            false
        })
    }

    /// Collect up to `max_remote_peers` peer IDs that may have rules relevant
    /// to `goal`, using the same CID-based lookup strategy as
    /// [`DistributedBackwardChainer`].
    async fn collect_peer_ids<FP>(
        &self,
        goal: &Term,
        kb: &KnowledgeBase,
        find_providers: &FP,
    ) -> Vec<String>
    where
        FP: Fn(Cid) -> BoxFuture<'static, Vec<String>> + Send + Sync,
    {
        let pred_name = match goal {
            Term::Fun(name, _) => name.clone(),
            Term::Const(crate::ir::Constant::String(s)) => s.clone(),
            _ => return Vec::new(),
        };

        let local_index = kb.index_rules_by_predicate_local();
        let rule_indices = local_index.get(&pred_name).cloned().unwrap_or_default();

        let mut candidate_cids: Vec<Cid> = Vec::new();
        for rule_idx in rule_indices {
            if let Some(rule) = kb.rules.get(rule_idx) {
                use crate::ipld_codec::{rule_cid, rule_to_rule_ipld};
                if let Ok(rule_ipld) = rule_to_rule_ipld(rule) {
                    if let Ok(cid) = rule_cid(&rule_ipld) {
                        candidate_cids.push(cid);
                    }
                }
            }
        }

        let mut peer_ids: Vec<String> = Vec::new();
        for cid in candidate_cids {
            if peer_ids.len() >= self.config.max_remote_peers {
                break;
            }
            let providers = find_providers(cid).await;
            for p in providers {
                if !peer_ids.contains(&p) {
                    peer_ids.push(p.clone());
                    if peer_ids.len() >= self.config.max_remote_peers {
                        break;
                    }
                }
            }
        }

        peer_ids
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Constant, Predicate, Rule, Term};
    use std::collections::HashMap;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn atom(s: &str) -> Term {
        Term::Const(Constant::String(s.to_string()))
    }

    fn var(s: &str) -> Term {
        Term::Var(s.to_string())
    }

    fn fun(name: &str, args: Vec<Term>) -> Term {
        Term::Fun(name.to_string(), args)
    }

    fn pred(name: &str, args: Vec<Term>) -> Predicate {
        Predicate::new(name.to_string(), args)
    }

    /// `find_providers` that always returns an empty peer list.
    fn no_providers() -> impl Fn(Cid) -> BoxFuture<'static, Vec<String>> + Send + Sync {
        |_cid| Box::pin(async { vec![] })
    }

    /// `remote_query` that always returns `None` (no remote knowledge).
    fn no_remote(
    ) -> impl Fn(String, Term) -> BoxFuture<'static, Option<Vec<HashMap<String, Term>>>> + Send + Sync
    {
        |_peer, _goal| Box::pin(async { None })
    }

    // ── VisitedSet tests ──────────────────────────────────────────────────────

    #[test]
    fn test_visited_set_first_visit() {
        let mut vs = VisitedSet::new(100);
        let goal = fun("p", vec![atom("a")]);
        assert!(
            vs.try_visit("local", &goal),
            "first visit should return true"
        );
    }

    #[test]
    fn test_visited_set_loop_detection() {
        let mut vs = VisitedSet::new(100);
        let goal = fun("p", vec![atom("a")]);
        vs.try_visit("local", &goal);
        assert!(
            !vs.try_visit("local", &goal),
            "second visit of same goal should return false (loop detected)"
        );
    }

    #[test]
    fn test_visited_set_different_goals() {
        let mut vs = VisitedSet::new(100);
        let g1 = fun("p", vec![atom("a")]);
        let g2 = fun("p", vec![atom("b")]);
        assert!(
            vs.try_visit("local", &g1),
            "first goal should be first visit"
        );
        assert!(
            vs.try_visit("local", &g2),
            "different goal should not be a loop"
        );
    }

    #[test]
    fn test_visited_set_different_peers() {
        let mut vs = VisitedSet::new(100);
        let goal = fun("p", vec![atom("a")]);
        assert!(vs.try_visit("peer-1", &goal), "peer-1 first visit");
        assert!(
            vs.try_visit("peer-2", &goal),
            "same goal but different peer is not a loop"
        );
        assert!(
            !vs.try_visit("peer-1", &goal),
            "peer-1 second visit is a loop"
        );
    }

    // ── HopTrace tests ────────────────────────────────────────────────────────

    #[test]
    fn test_hop_trace_push_and_len() {
        let mut trace = HopTrace::new();
        assert_eq!(trace.len(), 0);
        assert!(trace.is_empty());

        trace.push(HopRecord {
            hop_index: 0,
            peer_id: None,
            goal: "p(a)".to_string(),
            resolved: true,
        });
        assert_eq!(trace.len(), 1);
        assert!(!trace.is_empty());

        trace.push(HopRecord {
            hop_index: 1,
            peer_id: Some("peer-1".to_string()),
            goal: "q(b)".to_string(),
            resolved: false,
        });
        assert_eq!(trace.len(), 2);
    }

    #[test]
    fn test_hop_trace_remote_hops() {
        let mut trace = HopTrace::new();
        trace.push(HopRecord {
            hop_index: 0,
            peer_id: None,
            goal: "p(a)".to_string(),
            resolved: false,
        });
        trace.push(HopRecord {
            hop_index: 1,
            peer_id: Some("peer-1".to_string()),
            goal: "p(a)".to_string(),
            resolved: true,
        });
        trace.push(HopRecord {
            hop_index: 2,
            peer_id: Some("peer-2".to_string()),
            goal: "q(b)".to_string(),
            resolved: false,
        });

        let remotes = trace.remote_hops();
        assert_eq!(remotes.len(), 2, "only hops with Some(peer_id) are remote");
        assert_eq!(remotes[0].peer_id.as_deref(), Some("peer-1"));
        assert_eq!(remotes[1].peer_id.as_deref(), Some("peer-2"));
    }

    #[test]
    fn test_hop_trace_all_peers() {
        let mut trace = HopTrace::new();
        trace.push(HopRecord {
            hop_index: 0,
            peer_id: Some("peer-A".to_string()),
            goal: "p(a)".to_string(),
            resolved: false,
        });
        trace.push(HopRecord {
            hop_index: 1,
            peer_id: Some("peer-B".to_string()),
            goal: "p(a)".to_string(),
            resolved: false,
        });
        // duplicate
        trace.push(HopRecord {
            hop_index: 2,
            peer_id: Some("peer-A".to_string()),
            goal: "q(x)".to_string(),
            resolved: true,
        });

        let peers = trace.all_peers();
        assert_eq!(peers.len(), 2, "should deduplicate peer IDs");
        assert!(peers.contains(&"peer-A".to_string()));
        assert!(peers.contains(&"peer-B".to_string()));
    }

    // ── MultiHopResolver tests ────────────────────────────────────────────────

    fn build_local_kb() -> KnowledgeBase {
        let mut kb = KnowledgeBase::new();
        // a(alice) — base fact
        kb.add_fact(pred("a", vec![atom("alice")]));
        // b(X) :- a(X)
        kb.add_rule(Rule::new(
            pred("b", vec![var("X")]),
            vec![pred("a", vec![var("X")])],
        ));
        // c(X) :- b(X)
        kb.add_rule(Rule::new(
            pred("c", vec![var("X")]),
            vec![pred("b", vec![var("X")])],
        ));
        kb
    }

    #[tokio::test]
    async fn test_multi_hop_local_resolution() {
        let kb = build_local_kb();
        let resolver = MultiHopResolver::new(MultiHopConfig::default());
        let goal = fun("b", vec![atom("alice")]);

        let result = resolver
            .resolve(&goal, &kb, &no_providers(), &no_remote())
            .await;

        assert!(
            result.resolved,
            "goal b(alice) should resolve locally via rule b(X):-a(X)"
        );
        assert!(
            result.trace.remote_hops().is_empty(),
            "no remote hops expected for purely local resolution"
        );
    }

    #[tokio::test]
    async fn test_multi_hop_loop_prevention() {
        // Circular rule: p(X) :- p(X)  — should NOT loop forever
        let mut kb = KnowledgeBase::new();
        kb.add_rule(Rule::new(
            pred("p", vec![var("X")]),
            vec![pred("p", vec![var("X")])],
        ));

        let resolver = MultiHopResolver::new(MultiHopConfig {
            max_hops: 3,
            max_depth: 3,
            ..Default::default()
        });
        let goal = fun("p", vec![atom("x")]);

        // This must return without hanging or stack-overflowing
        let result = resolver
            .resolve(&goal, &kb, &no_providers(), &no_remote())
            .await;

        assert!(
            !result.resolved,
            "circular rule with no base fact should not resolve"
        );
    }

    #[tokio::test]
    async fn test_multi_hop_max_hops_limit() {
        // KB has no rules at all — goal is completely unknown
        let kb = KnowledgeBase::new();
        let resolver = MultiHopResolver::new(MultiHopConfig {
            max_hops: 2,
            ..Default::default()
        });

        // Provide a remote peer that always says "not found" after max_hops
        let goal = fun("unknown_pred", vec![atom("x")]);
        let result = resolver
            .resolve(&goal, &kb, &no_providers(), &no_remote())
            .await;

        assert!(
            !result.resolved,
            "should return unresolved when max_hops exceeded"
        );
    }

    #[test]
    fn test_default_config_values() {
        let cfg = MultiHopConfig::default();
        assert_eq!(cfg.max_hops, 5);
        assert_eq!(cfg.max_depth, 10);
        assert_eq!(cfg.max_visited, 1000);
        assert_eq!(cfg.max_remote_peers, 3);
        assert_eq!(cfg.timeout_ms, 15_000);
    }

    #[test]
    fn test_multi_hop_result_struct() {
        let trace = HopTrace::new();
        let result = MultiHopResult {
            resolved: false,
            trace,
            bindings: HashMap::new(),
        };
        assert!(!result.resolved);
        assert!(result.trace.is_empty());
        assert!(result.bindings.is_empty());
    }
}
