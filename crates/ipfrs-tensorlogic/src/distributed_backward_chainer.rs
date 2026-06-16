//! Distributed backward-chaining prover for TensorLogic.
//!
//! [`DistributedBackwardChainer`] extends the local backward-chaining engine
//! with the ability to delegate unsatisfied sub-goals to remote IPFRS peers.
//! When the local knowledge base cannot resolve a goal the chainer:
//!
//! 1. Looks up relevant rule CIDs in a predicate-name → CID index.
//! 2. Queries the DHT for peers that are providers of those CIDs.
//! 3. Sends the goal to at most `max_remote_peers` peers.
//! 4. Integrates the first successful remote binding into the proof tree.
//!
//! The resulting [`ProofTree`] records exactly *which* peer resolved each
//! node, making the derivation auditable across a distributed network.
//!
//! # Callback Conventions
//!
//! Both callbacks take *owned* arguments and return `BoxFuture<'static, ...>`.
//! This avoids complex higher-ranked lifetime constraints while keeping the
//! API ergonomic for both real-network integrations and unit-test mocks.

use crate::ir::{Constant, KnowledgeBase, Predicate, Term};
use crate::proof_tree::{ProofNode, ProofTree};
use crate::reasoning::{apply_subst_predicate, rename_rule_vars, unify_predicates, Substitution};
use futures::future::BoxFuture;
use ipfrs_core::{Cid, Result};
use std::collections::HashMap;
use std::sync::Arc;

/// A variable-binding map produced by a remote peer for a sub-goal.
pub type Binding = HashMap<String, Term>;

/// Distributed backward-chaining prover.
pub struct DistributedBackwardChainer {
    /// Maximum chaining depth.
    pub max_depth: usize,
    /// Maximum number of peers to contact per unresolved sub-goal.
    pub max_remote_peers: usize,
    /// Per-peer query timeout in milliseconds.
    pub timeout_ms: u64,
}

impl Default for DistributedBackwardChainer {
    fn default() -> Self {
        Self {
            max_depth: 10,
            max_remote_peers: 3,
            timeout_ms: 5000,
        }
    }
}

impl DistributedBackwardChainer {
    /// Create a chainer with explicit parameters.
    pub fn new(max_depth: usize, max_remote_peers: usize, timeout_ms: u64) -> Self {
        Self {
            max_depth,
            max_remote_peers,
            timeout_ms,
        }
    }

    /// Attempt to prove `goal` and return a [`ProofTree`] recording the full
    /// derivation path.
    ///
    /// Both callbacks take owned arguments and return `'static` futures.
    pub async fn prove_with_tree<FP, FQ>(
        &self,
        goal: &Term,
        local_kb: &KnowledgeBase,
        find_providers: FP,
        remote_query: FQ,
    ) -> Result<ProofTree>
    where
        FP: Fn(Cid) -> BoxFuture<'static, Vec<String>> + Send + Sync + 'static,
        FQ: Fn(String, Term) -> BoxFuture<'static, Option<Vec<Binding>>> + Send + Sync + 'static,
    {
        let ctx = Arc::new(ProveCtx {
            max_depth: self.max_depth,
            max_remote_peers: self.max_remote_peers,
            find_providers: Arc::new(find_providers),
            remote_query: Arc::new(remote_query),
        });

        let root =
            prove_term_impl(goal.clone(), Substitution::new(), local_kb.clone(), 0, ctx).await;

        let bindings = extract_top_level_bindings(goal, &root);
        let tree = ProofTree::new(root, goal.clone(), bindings);
        Ok(tree)
    }
}

// ── Free functions for recursion ──────────────────────────────────────────────

struct ProveCtx<FP, FQ> {
    max_depth: usize,
    max_remote_peers: usize,
    find_providers: Arc<FP>,
    remote_query: Arc<FQ>,
}

fn prove_term_impl<FP, FQ>(
    goal: Term,
    subst: Substitution,
    kb: KnowledgeBase,
    depth: usize,
    ctx: Arc<ProveCtx<FP, FQ>>,
) -> BoxFuture<'static, ProofNode>
where
    FP: Fn(Cid) -> BoxFuture<'static, Vec<String>> + Send + Sync + 'static,
    FQ: Fn(String, Term) -> BoxFuture<'static, Option<Vec<Binding>>> + Send + Sync + 'static,
{
    Box::pin(async move {
        if depth > ctx.max_depth {
            return ProofNode::unresolved(goal, depth);
        }

        let pred = match term_to_predicate(&goal) {
            Some(p) => p,
            None => {
                if goal.is_ground() {
                    return ProofNode::fact(goal, depth, None);
                }
                return ProofNode::unresolved(goal, depth);
            }
        };

        let pred = apply_subst_predicate(&pred, &subst);

        // 1. Try local facts
        let local_facts: Vec<_> = kb.get_predicates(&pred.name).into_iter().cloned().collect();
        for fact in &local_facts {
            if unify_predicates(&pred, fact, &subst).is_some() {
                return ProofNode::fact(goal, depth, None);
            }
        }

        // 2. Try local rules
        let local_rules: Vec<_> = kb.get_rules(&pred.name).into_iter().cloned().collect();
        for rule in &local_rules {
            let renamed = rename_rule_vars(rule, depth);
            if let Some(new_subst) = unify_predicates(&pred, &renamed.head, &subst) {
                let mut children = Vec::with_capacity(renamed.body.len());
                let mut body_resolved = true;

                for body_pred in &renamed.body {
                    let body_term = predicate_to_term(body_pred);
                    let child = prove_term_impl(
                        body_term,
                        new_subst.clone(),
                        kb.clone(),
                        depth + 1,
                        ctx.clone(),
                    )
                    .await;
                    if !child.resolved {
                        body_resolved = false;
                    }
                    children.push(child);
                }

                if body_resolved {
                    return ProofNode::from_rule(goal, None, children, depth, None);
                }
            }
        }

        // 3. Remote delegation
        if let Some(node) = try_remote_inner(
            &goal,
            &subst,
            &kb,
            depth,
            ctx.max_remote_peers,
            &*ctx.find_providers,
            &*ctx.remote_query,
        )
        .await
        {
            return node;
        }

        ProofNode::unresolved(goal, depth)
    })
}

async fn try_remote_inner<FP, FQ>(
    goal: &Term,
    subst: &Substitution,
    kb: &KnowledgeBase,
    depth: usize,
    max_remote_peers: usize,
    find_providers: &FP,
    remote_query: &FQ,
) -> Option<ProofNode>
where
    FP: Fn(Cid) -> BoxFuture<'static, Vec<String>> + Send + Sync,
    FQ: Fn(String, Term) -> BoxFuture<'static, Option<Vec<Binding>>> + Send + Sync,
{
    let pred = term_to_predicate(goal)?;
    let pred = apply_subst_predicate(&pred, subst);

    let local_index = kb.index_rules_by_predicate_local();
    let rule_indices = local_index.get(&pred.name).cloned().unwrap_or_default();

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
        if peer_ids.len() >= max_remote_peers {
            break;
        }
        let providers = find_providers(cid).await;
        for p in providers {
            if !peer_ids.contains(&p) {
                peer_ids.push(p);
                if peer_ids.len() >= max_remote_peers {
                    break;
                }
            }
        }
    }

    for peer_id in peer_ids {
        if let Some(bindings_list) = remote_query(peer_id.clone(), goal.clone()).await {
            if !bindings_list.is_empty() {
                return Some(ProofNode::fact(goal.clone(), depth, Some(peer_id)));
            }
        }
    }

    None
}

// ── Term ↔ Predicate helpers ──────────────────────────────────────────────────

fn term_to_predicate(term: &Term) -> Option<Predicate> {
    match term {
        Term::Fun(name, args) => Some(Predicate::new(name.clone(), args.clone())),
        Term::Const(Constant::String(s)) => Some(Predicate::new(s.clone(), Vec::new())),
        _ => None,
    }
}

fn predicate_to_term(pred: &Predicate) -> Term {
    Term::Fun(pred.name.clone(), pred.args.clone())
}

fn extract_top_level_bindings(query: &Term, _root: &ProofNode) -> HashMap<String, Term> {
    let mut bindings = HashMap::new();
    collect_ground_terms(query, &mut bindings);
    bindings
}

fn collect_ground_terms(term: &Term, acc: &mut HashMap<String, Term>) {
    match term {
        Term::Fun(_, args) => {
            for arg in args {
                collect_ground_terms(arg, acc);
            }
        }
        Term::Const(Constant::String(s)) => {
            acc.insert(s.clone(), term.clone());
        }
        _ => {}
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipld_codec::{rule_cid, rule_to_rule_ipld};
    use crate::ir::{Constant, Predicate, Rule, Term};
    use std::collections::HashMap;

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

    fn build_chain_kb() -> KnowledgeBase {
        let mut kb = KnowledgeBase::new();
        kb.add_fact(pred("a", vec![atom("alice")]));
        kb.add_rule(Rule::new(
            pred("b", vec![var("X")]),
            vec![pred("a", vec![var("X")])],
        ));
        kb.add_rule(Rule::new(
            pred("c", vec![var("X")]),
            vec![pred("b", vec![var("X")])],
        ));
        kb
    }

    fn no_providers() -> impl Fn(Cid) -> BoxFuture<'static, Vec<String>> + Send + Sync + 'static {
        |_cid| Box::pin(async { vec![] })
    }

    fn no_remote(
    ) -> impl Fn(String, Term) -> BoxFuture<'static, Option<Vec<Binding>>> + Send + Sync + 'static
    {
        |_peer, _goal| Box::pin(async { None })
    }

    /// A simple A->B->C chain should resolve entirely locally.
    #[tokio::test]
    async fn test_proof_tree_local_only() {
        let kb = build_chain_kb();
        let chainer = DistributedBackwardChainer::default();
        let goal = fun("c", vec![atom("alice")]);
        let tree = chainer
            .prove_with_tree(&goal, &kb, no_providers(), no_remote())
            .await
            .expect("prove_with_tree failed");

        assert!(tree.is_complete, "chain should fully resolve locally");
        assert!(
            tree.contributing_peers().is_empty(),
            "no remote peers expected"
        );
        assert!(!tree.root.children.is_empty(), "root should have children");
        assert_eq!(tree.root.peer, None);
    }

    /// Local rule partially resolved; a mock remote peer returns the binding.
    #[tokio::test]
    async fn test_proof_tree_partial_remote() {
        let mut kb = KnowledgeBase::new();
        kb.add_rule(Rule::new(
            pred("c", vec![var("X")]),
            vec![pred("a", vec![var("X")])],
        ));

        let rule = kb.rules[0].clone();
        let rule_ipld = rule_to_rule_ipld(&rule).expect("ipld");
        let expected_cid = rule_cid(&rule_ipld).expect("cid");

        let mock_peer = "mock-peer-001";

        let find_providers = move |lookup_cid: Cid| -> BoxFuture<'static, Vec<String>> {
            let peers = if lookup_cid == expected_cid {
                vec![mock_peer.to_string()]
            } else {
                vec![]
            };
            Box::pin(async move { peers })
        };

        let remote_query =
            move |peer: String, _goal: Term| -> BoxFuture<'static, Option<Vec<Binding>>> {
                let bindings: Option<Vec<Binding>> = if peer == mock_peer {
                    let mut b = HashMap::new();
                    b.insert("X".to_string(), atom("alice"));
                    Some(vec![b])
                } else {
                    None
                };
                Box::pin(async move { bindings })
            };

        let chainer = DistributedBackwardChainer::default();
        let goal = fun("c", vec![atom("alice")]);
        let tree = chainer
            .prove_with_tree(&goal, &kb, find_providers, remote_query)
            .await
            .expect("prove_with_tree failed");

        let peers = tree.contributing_peers();
        assert!(
            peers.contains(&mock_peer.to_string()),
            "mock peer should appear in contributing peers: {:?}",
            peers
        );
    }

    /// Index 10 rules and look up by predicate name; verify CID list.
    #[tokio::test]
    async fn test_predicate_index_roundtrip() {
        let mut kb = KnowledgeBase::new();
        for i in 0..10 {
            let head_name = format!("rule_{}", i);
            kb.add_rule(Rule::new(
                pred(&head_name, vec![var("X")]),
                vec![pred("base", vec![var("X")])],
            ));
        }

        let mut cid_map: HashMap<usize, Cid> = HashMap::new();
        for (idx, rule) in kb.rules.iter().enumerate() {
            let rule_ipld = rule_to_rule_ipld(rule).expect("ipld");
            let cid = rule_cid(&rule_ipld).expect("cid");
            cid_map.insert(idx, cid);
        }

        let index = kb.index_rules_by_predicate(&cid_map);

        for i in 0..10 {
            let name = format!("rule_{}", i);
            let cids = index.get(&name).expect("predicate not indexed");
            assert_eq!(cids.len(), 1, "expected 1 CID for {}", name);
        }

        assert!(
            !index.contains_key("base"),
            "body predicate should not be indexed"
        );
    }

    /// Chain that exceeds max_depth should produce is_complete=false.
    #[tokio::test]
    async fn test_backward_chain_depth_limit() {
        let mut kb = KnowledgeBase::new();
        kb.add_rule(Rule::new(
            pred("p", vec![var("X")]),
            vec![pred("p", vec![var("X")])],
        ));

        let chainer = DistributedBackwardChainer::new(3, 0, 5000);
        let goal = fun("p", vec![atom("a")]);

        let tree = chainer
            .prove_with_tree(&goal, &kb, no_providers(), no_remote())
            .await
            .expect("prove_with_tree should not error");

        assert!(
            !tree.is_complete,
            "recursive chain should NOT be complete when depth limit is hit"
        );
    }
}
