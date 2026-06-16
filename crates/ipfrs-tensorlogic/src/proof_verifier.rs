//! Proof Verifier
//!
//! Verifies serialized proof trees by re-checking each node's rule application.
//! Uses memoization to avoid re-verifying shared sub-proofs, and detects
//! cycles via an in-progress tracking set during DFS traversal.
//!
//! # Examples
//!
//! ```
//! use ipfrs_tensorlogic::proof_verifier::{ProofNode, ProofVerifier, RuleSpec};
//!
//! let mut verifier = ProofVerifier::new();
//!
//! // Register an axiom rule (arity = 0, no premises required)
//! verifier.register_rule(RuleSpec {
//!     rule_id: "axiom".to_string(),
//!     head_pattern: "fact:".to_string(),
//!     arity: 0,
//! });
//!
//! // A single axiom node
//! let nodes = vec![ProofNode {
//!     node_id: 1,
//!     rule_id: "axiom".to_string(),
//!     goal: "fact:alice-is-human".to_string(),
//!     premise_ids: vec![],
//!     depth: 0,
//! }];
//!
//! let result = verifier.verify(&nodes);
//! assert!(result.is_valid());
//! assert_eq!(result.nodes_checked, 1);
//! ```

use std::collections::{HashMap, HashSet};

use thiserror::Error;

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors produced during proof verification.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum VerificationError {
    /// The rule ID referenced by a node is not registered.
    #[error("unknown rule: {0}")]
    UnknownRule(String),

    /// The node's goal does not match the rule's head pattern.
    #[error("goal mismatch at node {node_id}: expected pattern {expected:?}, got {actual:?}")]
    GoalMismatch {
        node_id: u64,
        expected: String,
        actual: String,
    },

    /// The number of premises at a node does not match the rule's arity.
    #[error("premise mismatch at node {node_id}, premise index {index}")]
    PremiseMismatch { node_id: u64, index: usize },

    /// A cycle was detected in the proof tree at the given node.
    #[error("cyclic proof detected at node {0}")]
    CyclicProof(u64),

    /// The proof contains no nodes.
    #[error("proof is empty")]
    EmptyProof,
}

// ─── Result ──────────────────────────────────────────────────────────────────

/// Summary of a completed proof verification pass.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether verification completed without encountering any errors.
    pub verified: bool,

    /// Total number of nodes that were actively checked.
    pub nodes_checked: usize,

    /// Number of nodes whose result was served from the memo cache (cache hits).
    pub nodes_memoized: usize,

    /// Maximum proof tree depth encountered during traversal.
    pub max_depth: usize,

    /// All errors collected during verification.
    pub errors: Vec<VerificationError>,
}

impl VerificationResult {
    /// Returns `true` iff the proof is verified and no errors were collected.
    pub fn is_valid(&self) -> bool {
        self.verified && self.errors.is_empty()
    }
}

// ─── Proof node ──────────────────────────────────────────────────────────────

/// A single node in a serialized proof tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofNode {
    /// Unique identifier for this node.
    pub node_id: u64,

    /// Identifier of the inference rule applied at this node.
    pub rule_id: String,

    /// The logical goal that this node proves.
    pub goal: String,

    /// IDs of the premise (child) nodes consumed by the rule application.
    pub premise_ids: Vec<u64>,

    /// Depth of this node in the proof tree (root = 0).
    pub depth: usize,
}

// ─── Rule spec ───────────────────────────────────────────────────────────────

/// Specification of an inference rule used to validate proof nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleSpec {
    /// Unique identifier for this rule.
    pub rule_id: String,

    /// Prefix pattern that the goal must start with for this rule to apply.
    pub head_pattern: String,

    /// Expected number of premises (children) for this rule.
    pub arity: usize,
}

// ─── Verifier ────────────────────────────────────────────────────────────────

/// Verifies serialized proof trees by re-checking each node's rule application.
///
/// Maintains a memoization cache so that shared sub-proofs (DAG structure) are
/// not re-verified on subsequent traversals.  A fresh [`ProofVerifier`] starts
/// with an empty rule registry and an empty memo cache.
pub struct ProofVerifier {
    /// Registered inference rules, keyed by rule ID.
    pub rules: HashMap<String, RuleSpec>,

    /// Memoized verification results: `node_id → verified`.
    pub memo: HashMap<u64, bool>,

    /// Nodes currently on the DFS stack — used for cycle detection.
    pub in_progress: HashSet<u64>,
}

impl ProofVerifier {
    /// Create a new, empty `ProofVerifier`.
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
            memo: HashMap::new(),
            in_progress: HashSet::new(),
        }
    }

    /// Register an inference rule.  Replaces any existing rule with the same ID.
    pub fn register_rule(&mut self, spec: RuleSpec) {
        self.rules.insert(spec.rule_id.clone(), spec);
    }

    /// Verify a slice of [`ProofNode`]s.
    ///
    /// Identifies the root as the node whose ID does not appear as any other
    /// node's premise.  If there is no unique root, the first node is used.
    ///
    /// DFS from the root re-checks every rule application.  Results are
    /// memoized so shared sub-proofs are counted as cache hits.
    pub fn verify(&mut self, nodes: &[ProofNode]) -> VerificationResult {
        if nodes.is_empty() {
            return VerificationResult {
                verified: false,
                nodes_checked: 0,
                nodes_memoized: 0,
                max_depth: 0,
                errors: vec![VerificationError::EmptyProof],
            };
        }

        // Build a quick lookup map.
        let node_map: HashMap<u64, &ProofNode> = nodes.iter().map(|n| (n.node_id, n)).collect();

        // Find the root: node whose ID does not appear as any premise.
        let all_premise_ids: HashSet<u64> = nodes
            .iter()
            .flat_map(|n| n.premise_ids.iter().copied())
            .collect();

        let roots: Vec<u64> = nodes
            .iter()
            .map(|n| n.node_id)
            .filter(|id| !all_premise_ids.contains(id))
            .collect();

        let root_id = if roots.len() == 1 {
            roots[0]
        } else {
            // No unique root — fall back to first node.
            nodes[0].node_id
        };

        // Accumulators shared across recursive calls via owned state on `self`.
        let mut errors: Vec<VerificationError> = Vec::new();
        let mut nodes_checked: usize = 0;
        let mut nodes_memoized: usize = 0;
        let mut max_depth: usize = 0;

        let ok = self.verify_node(
            root_id,
            &node_map,
            &mut errors,
            &mut nodes_checked,
            &mut nodes_memoized,
            &mut max_depth,
        );

        // Clean up in-progress set (should already be empty after DFS).
        self.in_progress.clear();

        VerificationResult {
            verified: ok,
            nodes_checked,
            nodes_memoized,
            max_depth,
            errors,
        }
    }

    /// Recursive DFS verification for a single node.
    ///
    /// Returns `true` if this subtree is valid.
    fn verify_node(
        &mut self,
        node_id: u64,
        node_map: &HashMap<u64, &ProofNode>,
        errors: &mut Vec<VerificationError>,
        nodes_checked: &mut usize,
        nodes_memoized: &mut usize,
        max_depth: &mut usize,
    ) -> bool {
        // Cycle detection.
        if self.in_progress.contains(&node_id) {
            errors.push(VerificationError::CyclicProof(node_id));
            return false;
        }

        // Memo cache hit.
        if let Some(&cached) = self.memo.get(&node_id) {
            *nodes_memoized += 1;
            return cached;
        }

        // Retrieve the node.
        let node = match node_map.get(&node_id) {
            Some(n) => *n,
            None => {
                // Referenced node not found — treat as a failed check.
                errors.push(VerificationError::PremiseMismatch { node_id, index: 0 });
                self.memo.insert(node_id, false);
                return false;
            }
        };

        // Update max depth.
        if node.depth > *max_depth {
            *max_depth = node.depth;
        }

        // Mark as in-progress.
        self.in_progress.insert(node_id);
        *nodes_checked += 1;

        // Look up the rule.
        let rule = match self.rules.get(&node.rule_id) {
            Some(r) => r.clone(), // clone to avoid borrow conflict
            None => {
                errors.push(VerificationError::UnknownRule(node.rule_id.clone()));
                self.in_progress.remove(&node_id);
                self.memo.insert(node_id, false);
                return false;
            }
        };

        // Check goal matches head pattern.
        if !node.goal.starts_with(&rule.head_pattern) {
            errors.push(VerificationError::GoalMismatch {
                node_id,
                expected: rule.head_pattern.clone(),
                actual: node.goal.clone(),
            });
            self.in_progress.remove(&node_id);
            self.memo.insert(node_id, false);
            return false;
        }

        // Check arity (number of premises).
        if node.premise_ids.len() != rule.arity {
            // Report a PremiseMismatch at the first mismatched index.
            let index = node.premise_ids.len().min(rule.arity);
            errors.push(VerificationError::PremiseMismatch { node_id, index });
            self.in_progress.remove(&node_id);
            self.memo.insert(node_id, false);
            return false;
        }

        // Recurse into premises.
        let mut all_ok = true;
        let premise_ids: Vec<u64> = node.premise_ids.clone();
        for (i, &premise_id) in premise_ids.iter().enumerate() {
            if !node_map.contains_key(&premise_id) {
                errors.push(VerificationError::PremiseMismatch { node_id, index: i });
                all_ok = false;
            } else {
                let premise_ok = self.verify_node(
                    premise_id,
                    node_map,
                    errors,
                    nodes_checked,
                    nodes_memoized,
                    max_depth,
                );
                if !premise_ok {
                    all_ok = false;
                }
            }
        }

        self.in_progress.remove(&node_id);
        self.memo.insert(node_id, all_ok);
        all_ok
    }

    /// Reset memo cache and in-progress set.
    pub fn clear_memo(&mut self) {
        self.memo.clear();
        self.in_progress.clear();
    }

    /// Returns `(rules_registered, memo_entries)`.
    pub fn stats(&self) -> (usize, usize) {
        (self.rules.len(), self.memo.len())
    }
}

impl Default for ProofVerifier {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a verifier with common rules.
    fn make_verifier() -> ProofVerifier {
        let mut v = ProofVerifier::new();
        // axiom: no premises, goal must start with "fact:"
        v.register_rule(RuleSpec {
            rule_id: "axiom".to_string(),
            head_pattern: "fact:".to_string(),
            arity: 0,
        });
        // modus_ponens: 1 premise, goal must start with "derived:"
        v.register_rule(RuleSpec {
            rule_id: "modus_ponens".to_string(),
            head_pattern: "derived:".to_string(),
            arity: 1,
        });
        // conjunction: 2 premises, goal must start with "and:"
        v.register_rule(RuleSpec {
            rule_id: "conjunction".to_string(),
            head_pattern: "and:".to_string(),
            arity: 2,
        });
        v
    }

    // 1. Empty proof returns EmptyProof error.
    #[test]
    fn test_empty_proof() {
        let mut v = make_verifier();
        let result = v.verify(&[]);
        assert!(!result.is_valid());
        assert_eq!(result.errors.len(), 1);
        assert!(matches!(result.errors[0], VerificationError::EmptyProof));
    }

    // 2. Single axiom node (no premises) is verified successfully.
    #[test]
    fn test_single_axiom() {
        let mut v = make_verifier();
        let nodes = vec![ProofNode {
            node_id: 1,
            rule_id: "axiom".to_string(),
            goal: "fact:alice-is-human".to_string(),
            premise_ids: vec![],
            depth: 0,
        }];
        let result = v.verify(&nodes);
        assert!(result.is_valid(), "{:?}", result.errors);
        assert_eq!(result.nodes_checked, 1);
        assert_eq!(result.nodes_memoized, 0);
        assert_eq!(result.max_depth, 0);
    }

    // 3. Two-node proof (rule with 1 premise) is verified.
    #[test]
    fn test_two_node_proof() {
        let mut v = make_verifier();
        let nodes = vec![
            ProofNode {
                node_id: 1,
                rule_id: "axiom".to_string(),
                goal: "fact:alice-is-human".to_string(),
                premise_ids: vec![],
                depth: 1,
            },
            ProofNode {
                node_id: 2,
                rule_id: "modus_ponens".to_string(),
                goal: "derived:alice-is-mortal".to_string(),
                premise_ids: vec![1],
                depth: 0,
            },
        ];
        let result = v.verify(&nodes);
        assert!(result.is_valid(), "{:?}", result.errors);
        assert_eq!(result.nodes_checked, 2);
        assert_eq!(result.max_depth, 1);
    }

    // 4. Three-level chain is verified.
    #[test]
    fn test_three_level_chain() {
        let mut v = make_verifier();
        // chain: axiom(1) -> modus_ponens(2) -> modus_ponens(3)
        let nodes = vec![
            ProofNode {
                node_id: 1,
                rule_id: "axiom".to_string(),
                goal: "fact:base".to_string(),
                premise_ids: vec![],
                depth: 2,
            },
            ProofNode {
                node_id: 2,
                rule_id: "modus_ponens".to_string(),
                goal: "derived:mid".to_string(),
                premise_ids: vec![1],
                depth: 1,
            },
            ProofNode {
                node_id: 3,
                rule_id: "modus_ponens".to_string(),
                goal: "derived:top".to_string(),
                premise_ids: vec![2],
                depth: 0,
            },
        ];
        let result = v.verify(&nodes);
        assert!(result.is_valid(), "{:?}", result.errors);
        assert_eq!(result.nodes_checked, 3);
        assert_eq!(result.max_depth, 2);
    }

    // 5. Unknown rule returns UnknownRule error.
    #[test]
    fn test_unknown_rule() {
        let mut v = make_verifier();
        let nodes = vec![ProofNode {
            node_id: 1,
            rule_id: "no_such_rule".to_string(),
            goal: "fact:something".to_string(),
            premise_ids: vec![],
            depth: 0,
        }];
        let result = v.verify(&nodes);
        assert!(!result.is_valid());
        assert!(result.errors.iter().any(|e| matches!(
            e,
            VerificationError::UnknownRule(id) if id == "no_such_rule"
        )));
    }

    // 6. Arity mismatch returns PremiseMismatch error.
    #[test]
    fn test_arity_mismatch() {
        let mut v = make_verifier();
        // modus_ponens expects 1 premise but we give 0.
        let nodes = vec![ProofNode {
            node_id: 1,
            rule_id: "modus_ponens".to_string(),
            goal: "derived:something".to_string(),
            premise_ids: vec![], // wrong: should be 1 premise
            depth: 0,
        }];
        let result = v.verify(&nodes);
        assert!(!result.is_valid());
        assert!(result
            .errors
            .iter()
            .any(|e| matches!(e, VerificationError::PremiseMismatch { node_id: 1, .. })));
    }

    // 7. Goal mismatch returns GoalMismatch error.
    #[test]
    fn test_goal_mismatch() {
        let mut v = make_verifier();
        let nodes = vec![ProofNode {
            node_id: 1,
            rule_id: "axiom".to_string(),
            goal: "wrong_prefix:something".to_string(), // axiom expects "fact:"
            premise_ids: vec![],
            depth: 0,
        }];
        let result = v.verify(&nodes);
        assert!(!result.is_valid());
        assert!(result
            .errors
            .iter()
            .any(|e| matches!(e, VerificationError::GoalMismatch { node_id: 1, .. })));
    }

    // 8. Cycle detection returns CyclicProof error.
    #[test]
    fn test_cycle_detection() {
        let mut v = make_verifier();
        // node 1 has premise 2, and node 2 has premise 1 — a cycle.
        // We need a rule with arity 1.
        let nodes = vec![
            ProofNode {
                node_id: 1,
                rule_id: "modus_ponens".to_string(),
                goal: "derived:a".to_string(),
                premise_ids: vec![2],
                depth: 0,
            },
            ProofNode {
                node_id: 2,
                rule_id: "modus_ponens".to_string(),
                goal: "derived:b".to_string(),
                premise_ids: vec![1],
                depth: 1,
            },
        ];
        let result = v.verify(&nodes);
        assert!(!result.is_valid());
        assert!(result
            .errors
            .iter()
            .any(|e| matches!(e, VerificationError::CyclicProof(_))));
    }

    // 9. Memo cache hits are counted correctly.
    #[test]
    fn test_memo_cache_hits() {
        let mut v = make_verifier();
        // Diamond DAG: root(3) has premises [1, 2]; both 1 and 2 share premise 0.
        // But ProofNode.premise_ids are IDs, and `verify` traverses from root only.
        // We create: axiom(0), mp(1, premise=0), mp(2, premise=0), conj(3, premises=[1,2]).
        let nodes = vec![
            ProofNode {
                node_id: 0,
                rule_id: "axiom".to_string(),
                goal: "fact:shared".to_string(),
                premise_ids: vec![],
                depth: 2,
            },
            ProofNode {
                node_id: 1,
                rule_id: "modus_ponens".to_string(),
                goal: "derived:left".to_string(),
                premise_ids: vec![0],
                depth: 1,
            },
            ProofNode {
                node_id: 2,
                rule_id: "modus_ponens".to_string(),
                goal: "derived:right".to_string(),
                premise_ids: vec![0],
                depth: 1,
            },
            ProofNode {
                node_id: 3,
                rule_id: "conjunction".to_string(),
                goal: "and:both".to_string(),
                premise_ids: vec![1, 2],
                depth: 0,
            },
        ];
        let result = v.verify(&nodes);
        assert!(result.is_valid(), "{:?}", result.errors);
        // Node 0 is reached via node 1 first (checked), then via node 2 (memoized).
        assert_eq!(result.nodes_memoized, 1);
        // nodes_checked = 4 (3, 1, 0, 2) — 0 is checked once, memoized once.
        assert_eq!(result.nodes_checked, 4);
    }

    // 10. clear_memo resets memo and in_progress state.
    #[test]
    fn test_clear_memo() {
        let mut v = make_verifier();
        let nodes = vec![ProofNode {
            node_id: 42,
            rule_id: "axiom".to_string(),
            goal: "fact:x".to_string(),
            premise_ids: vec![],
            depth: 0,
        }];
        let _ = v.verify(&nodes);
        assert!(!v.memo.is_empty());

        v.clear_memo();
        assert!(v.memo.is_empty());
        assert!(v.in_progress.is_empty());
    }

    // 11. stats() returns correct counts.
    #[test]
    fn test_stats() {
        let mut v = make_verifier();
        let (rules, memo) = v.stats();
        assert_eq!(rules, 3); // axiom, modus_ponens, conjunction
        assert_eq!(memo, 0);

        let nodes = vec![ProofNode {
            node_id: 99,
            rule_id: "axiom".to_string(),
            goal: "fact:y".to_string(),
            premise_ids: vec![],
            depth: 0,
        }];
        let _ = v.verify(&nodes);
        let (rules2, memo2) = v.stats();
        assert_eq!(rules2, 3);
        assert_eq!(memo2, 1);
    }

    // 12. is_valid() returns false when errors are present.
    #[test]
    fn test_is_valid_false_on_errors() {
        let result = VerificationResult {
            verified: true,
            nodes_checked: 1,
            nodes_memoized: 0,
            max_depth: 0,
            errors: vec![VerificationError::EmptyProof],
        };
        assert!(!result.is_valid());
    }

    // 13. Multiple errors are collected (not short-circuited at root).
    #[test]
    fn test_multiple_errors_collected() {
        let mut v = make_verifier();
        // conjunction has 2 premises; supply two nodes both with unknown rules.
        let nodes = vec![
            ProofNode {
                node_id: 1,
                rule_id: "bad_rule_a".to_string(),
                goal: "fact:a".to_string(),
                premise_ids: vec![],
                depth: 1,
            },
            ProofNode {
                node_id: 2,
                rule_id: "bad_rule_b".to_string(),
                goal: "fact:b".to_string(),
                premise_ids: vec![],
                depth: 1,
            },
            ProofNode {
                node_id: 3,
                rule_id: "conjunction".to_string(),
                goal: "and:ab".to_string(),
                premise_ids: vec![1, 2],
                depth: 0,
            },
        ];
        let result = v.verify(&nodes);
        assert!(!result.is_valid());
        // At least two UnknownRule errors: one for each bad premise.
        let unknown_count = result
            .errors
            .iter()
            .filter(|e| matches!(e, VerificationError::UnknownRule(_)))
            .count();
        assert!(
            unknown_count >= 2,
            "expected >=2 UnknownRule errors, got {unknown_count}"
        );
    }

    // 14. max_depth computed correctly.
    #[test]
    fn test_max_depth_computed() {
        let mut v = make_verifier();
        let nodes = vec![
            ProofNode {
                node_id: 1,
                rule_id: "axiom".to_string(),
                goal: "fact:a".to_string(),
                premise_ids: vec![],
                depth: 5,
            },
            ProofNode {
                node_id: 2,
                rule_id: "modus_ponens".to_string(),
                goal: "derived:b".to_string(),
                premise_ids: vec![1],
                depth: 2,
            },
        ];
        let result = v.verify(&nodes);
        assert!(result.is_valid(), "{:?}", result.errors);
        assert_eq!(result.max_depth, 5);
    }

    // 15. register_rule replaces existing rule.
    #[test]
    fn test_register_rule_replaces() {
        let mut v = ProofVerifier::new();
        v.register_rule(RuleSpec {
            rule_id: "my_rule".to_string(),
            head_pattern: "old:".to_string(),
            arity: 0,
        });
        // Now replace with a different pattern.
        v.register_rule(RuleSpec {
            rule_id: "my_rule".to_string(),
            head_pattern: "new:".to_string(),
            arity: 0,
        });
        assert_eq!(v.rules.len(), 1);
        assert_eq!(v.rules["my_rule"].head_pattern, "new:");

        // Verify that the new pattern is used.
        let nodes = vec![ProofNode {
            node_id: 1,
            rule_id: "my_rule".to_string(),
            goal: "new:something".to_string(),
            premise_ids: vec![],
            depth: 0,
        }];
        let result = v.verify(&nodes);
        assert!(result.is_valid(), "{:?}", result.errors);
    }

    // 16. Large flat proof (all axioms) verifies correctly.
    #[test]
    fn test_large_flat_proof() {
        let mut v = ProofVerifier::new();
        // A "tree" rule that takes N premises would require special setup.
        // Instead we register a wide conjunction-style rule with arity 0
        // and verify 1000 independent axiom nodes.
        v.register_rule(RuleSpec {
            rule_id: "axiom".to_string(),
            head_pattern: "fact:".to_string(),
            arity: 0,
        });

        // Create 1000 independent axiom nodes.
        let nodes: Vec<ProofNode> = (0u64..1000)
            .map(|i| ProofNode {
                node_id: i,
                rule_id: "axiom".to_string(),
                goal: format!("fact:item-{i}"),
                premise_ids: vec![],
                depth: 0,
            })
            .collect();

        // verify uses the first node as root when there's no single root
        // (all 1000 are roots since none are each other's premises).
        // We just call verify and confirm no errors on the root chain.
        let result = v.verify(&nodes);
        // root node 0 has no premises — should be valid on its own.
        assert!(result.is_valid(), "{:?}", result.errors);
        assert_eq!(result.nodes_checked, 1);
        assert_eq!(result.nodes_memoized, 0);
    }
}
