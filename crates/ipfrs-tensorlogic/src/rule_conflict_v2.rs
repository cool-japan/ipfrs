//! Extended Rule Conflict Resolution V2
//!
//! This module provides comprehensive conflict detection and resolution for
//! logic rule sets with support for multi-party conflicts, priority chains,
//! and detailed conflict reports.
//!
//! # Conflict Types
//!
//! | Variant | Meaning |
//! |---------|---------|
//! | [`ConflictType::HeadOverlap`] | Two rules prove the same goal pattern (prefix match). |
//! | [`ConflictType::CycleDetected`] | Rules form a dependency cycle. |
//! | [`ConflictType::PriorityTie`] | Same priority value; resolution is ambiguous. |
//! | [`ConflictType::ContradictoryConstraints`] | Rules impose contradictory head conditions. |
//!
//! # Resolution Strategies
//!
//! | Variant | Meaning |
//! |---------|---------|
//! | [`ResolutionStrategy::HigherPriority`] | Prefer rule with higher priority value. |
//! | [`ResolutionStrategy::LaterTimestamp`] | Prefer more recently created rule. |
//! | [`ResolutionStrategy::Alphabetical`] | Prefer lexicographically smaller rule_id. |
//! | [`ResolutionStrategy::FirstRegistered`] | Prefer rule registered first (index order). |
//!
//! # Examples
//!
//! ```
//! use ipfrs_tensorlogic::rule_conflict_v2::{
//!     ResolutionStrategy, RuleConflictResolverV2, RuleV2,
//! };
//!
//! let mut resolver = RuleConflictResolverV2::new(ResolutionStrategy::HigherPriority);
//!
//! resolver.register(RuleV2 {
//!     rule_id: "r1".to_string(),
//!     head: "parent".to_string(),
//!     body: vec![],
//!     priority: 10,
//!     author: "alice".to_string(),
//!     created_at_secs: 1000,
//! });
//!
//! resolver.register(RuleV2 {
//!     rule_id: "r2".to_string(),
//!     head: "parent_of".to_string(),
//!     body: vec![],
//!     priority: 5,
//!     author: "bob".to_string(),
//!     created_at_secs: 2000,
//! });
//!
//! let reports = resolver.resolve_all();
//! // "parent" is a prefix of "parent_of" → HeadOverlap, r1 wins (higher priority)
//! assert_eq!(reports.len(), 1);
//! assert_eq!(reports[0].winner, Some("r1".to_string()));
//! ```

use std::collections::HashMap;

// ─── ConflictType ─────────────────────────────────────────────────────────────

/// Categorises the nature of a detected conflict between rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictType {
    /// Two rules prove the same goal pattern (head prefix overlap).
    HeadOverlap,
    /// Rules form a dependency cycle.
    CycleDetected,
    /// Two or more rules share the same priority value; the winner cannot be
    /// determined unambiguously by priority alone.
    PriorityTie,
    /// Rules impose contradictory conditions on the same head predicate.
    ContradictoryConstraints,
}

// ─── RuleV2 ───────────────────────────────────────────────────────────────────

/// A versioned logic rule with conflict-resolution metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleV2 {
    /// Unique identifier for this rule.
    pub rule_id: String,
    /// Head predicate pattern (what the rule proves).
    pub head: String,
    /// Body predicates (what the rule depends on).
    pub body: Vec<String>,
    /// Priority value — higher is preferred.
    pub priority: i32,
    /// Author / owner of this rule.
    pub author: String,
    /// Unix timestamp (seconds) of rule creation.
    pub created_at_secs: u64,
}

// ─── ConflictReport ──────────────────────────────────────────────────────────

/// A structured report describing a single detected conflict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConflictReport {
    /// The kind of conflict detected.
    pub conflict_type: ConflictType,
    /// All rule IDs involved in the conflict.
    pub rule_ids: Vec<String>,
    /// The rule ID that wins (if it can be determined by the strategy).
    pub winner: Option<String>,
    /// Human-readable explanation of the conflict and its resolution.
    pub resolution: String,
}

// ─── ResolutionStrategy ──────────────────────────────────────────────────────

/// Strategy used to select the winning rule when a conflict is detected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolutionStrategy {
    /// Prefer the rule with the higher `priority` value.
    HigherPriority,
    /// Prefer the more recently created rule (largest `created_at_secs`).
    LaterTimestamp,
    /// Prefer the rule with the lexicographically smaller `rule_id` (deterministic).
    Alphabetical,
    /// Prefer the rule that was registered first (smallest index in `rules` vec).
    FirstRegistered,
}

// ─── RuleConflictResolverV2 ───────────────────────────────────────────────────

/// Extended conflict resolver for logic rule sets.
///
/// Supports multi-party conflict detection, priority chains, cycle detection,
/// and detailed conflict reporting.
#[derive(Debug, Clone)]
pub struct RuleConflictResolverV2 {
    /// All registered rules, in registration order.
    pub rules: Vec<RuleV2>,
    /// Strategy used to choose the winner in a pairwise conflict.
    pub strategy: ResolutionStrategy,
}

impl RuleConflictResolverV2 {
    /// Create a new resolver with the given resolution strategy.
    pub fn new(strategy: ResolutionStrategy) -> Self {
        Self {
            rules: Vec::new(),
            strategy,
        }
    }

    /// Register a rule with the resolver.
    ///
    /// Rules are appended in registration order; this order is used by
    /// [`ResolutionStrategy::FirstRegistered`].
    pub fn register(&mut self, rule: RuleV2) {
        self.rules.push(rule);
    }

    /// Determine the winner between two rules (identified by their slice
    /// indices) using the configured strategy.
    ///
    /// Returns `None` when the strategy cannot break the tie (e.g. equal
    /// priority under [`ResolutionStrategy::HigherPriority`]).
    fn pick_winner_indices(&self, idx_a: usize, idx_b: usize) -> Option<usize> {
        let a = &self.rules[idx_a];
        let b = &self.rules[idx_b];
        match self.strategy {
            ResolutionStrategy::HigherPriority => {
                if a.priority > b.priority {
                    Some(idx_a)
                } else if b.priority > a.priority {
                    Some(idx_b)
                } else {
                    None // tie
                }
            }
            ResolutionStrategy::LaterTimestamp => {
                if a.created_at_secs > b.created_at_secs {
                    Some(idx_a)
                } else if b.created_at_secs > a.created_at_secs {
                    Some(idx_b)
                } else {
                    None
                }
            }
            ResolutionStrategy::Alphabetical => {
                if a.rule_id <= b.rule_id {
                    Some(idx_a)
                } else {
                    Some(idx_b)
                }
            }
            ResolutionStrategy::FirstRegistered => {
                // idx_a < idx_b by construction in detect_head_overlaps, but
                // guard anyway.
                if idx_a <= idx_b {
                    Some(idx_a)
                } else {
                    Some(idx_b)
                }
            }
        }
    }

    /// Detect pairs of rules whose heads share a prefix (one is a prefix of
    /// the other, or they are equal).
    ///
    /// For each such pair, a [`ConflictReport`] is produced.  When the
    /// strategy resolves the tie the `winner` field is set; when the
    /// priorities are equal under [`ResolutionStrategy::HigherPriority`] the
    /// conflict type is [`ConflictType::PriorityTie`].
    pub fn detect_head_overlaps(&self) -> Vec<ConflictReport> {
        let mut reports = Vec::new();

        for i in 0..self.rules.len() {
            for j in (i + 1)..self.rules.len() {
                let head_a = &self.rules[i].head;
                let head_b = &self.rules[j].head;

                let overlaps =
                    head_a.starts_with(head_b.as_str()) || head_b.starts_with(head_a.as_str());
                if !overlaps {
                    continue;
                }

                // Determine conflict type and winner.
                let (conflict_type, winner_id, resolution) =
                    if matches!(self.strategy, ResolutionStrategy::HigherPriority)
                        && self.rules[i].priority == self.rules[j].priority
                    {
                        let res = format!(
                            "Rules '{}' and '{}' have overlapping heads ('{}' / '{}') \
                             and equal priority {}; cannot determine winner.",
                            self.rules[i].rule_id,
                            self.rules[j].rule_id,
                            head_a,
                            head_b,
                            self.rules[i].priority
                        );
                        (ConflictType::PriorityTie, None, res)
                    } else {
                        let winner_idx = self.pick_winner_indices(i, j);
                        let winner_id = winner_idx.map(|idx| self.rules[idx].rule_id.clone());
                        let res = match &winner_id {
                            Some(w) => format!(
                                "Rules '{}' and '{}' have overlapping heads ('{}' / '{}'). \
                                 Winner: '{}' selected by {:?} strategy.",
                                self.rules[i].rule_id,
                                self.rules[j].rule_id,
                                head_a,
                                head_b,
                                w,
                                self.strategy
                            ),
                            None => format!(
                                "Rules '{}' and '{}' have overlapping heads ('{}' / '{}'). \
                                 Strategy {:?} could not break the tie.",
                                self.rules[i].rule_id,
                                self.rules[j].rule_id,
                                head_a,
                                head_b,
                                self.strategy
                            ),
                        };
                        (ConflictType::HeadOverlap, winner_id, res)
                    };

                reports.push(ConflictReport {
                    conflict_type,
                    rule_ids: vec![self.rules[i].rule_id.clone(), self.rules[j].rule_id.clone()],
                    winner: winner_id,
                    resolution,
                });
            }
        }

        reports
    }

    /// Detect dependency cycles among the registered rules.
    ///
    /// A dependency edge A → B exists when any predicate in A's body matches
    /// B's head (exact match: `body_pred == B.head`).
    ///
    /// Cycle detection uses iterative DFS with 3-colour marking:
    /// - `0` = unvisited
    /// - `1` = in the current DFS stack (grey)
    /// - `2` = fully processed (black)
    ///
    /// Each distinct cycle is reported once.
    pub fn detect_cycles(&self) -> Vec<ConflictReport> {
        // Build: head → list of rule indices that have this head.
        let mut head_to_indices: HashMap<&str, Vec<usize>> = HashMap::new();
        for (idx, rule) in self.rules.iter().enumerate() {
            head_to_indices
                .entry(rule.head.as_str())
                .or_default()
                .push(idx);
        }

        // Build adjacency list: rule index → set of rule indices it depends on.
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); self.rules.len()];
        for (src_idx, rule) in self.rules.iter().enumerate() {
            for body_pred in &rule.body {
                if let Some(targets) = head_to_indices.get(body_pred.as_str()) {
                    for &tgt_idx in targets {
                        if tgt_idx != src_idx {
                            adj[src_idx].push(tgt_idx);
                        }
                    }
                }
            }
        }

        let n = self.rules.len();
        // 0 = unvisited, 1 = in-progress, 2 = done
        let mut color: Vec<u8> = vec![0u8; n];
        // parent[i] = index of node that discovered i in DFS
        let mut parent: Vec<Option<usize>> = vec![None; n];
        let mut reports: Vec<ConflictReport> = Vec::new();
        // Track cycle signatures already reported to deduplicate.
        let mut reported_cycles: Vec<Vec<usize>> = Vec::new();

        for start in 0..n {
            if color[start] != 0 {
                continue;
            }
            // Iterative DFS using an explicit stack.
            // Stack items: (node_index, iterator_position_into_adj[node])
            let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
            color[start] = 1;

            'dfs: while let Some((node, child_pos)) = stack.last_mut() {
                let node = *node;
                let adj_list = &adj[node];

                if *child_pos < adj_list.len() {
                    let child = adj_list[*child_pos];
                    *child_pos += 1;

                    if color[child] == 1 {
                        // Back edge found: reconstruct cycle path.
                        let mut cycle_indices: Vec<usize> = Vec::new();
                        cycle_indices.push(child);
                        // Walk the stack backwards until we reach `child` again.
                        let mut found = false;
                        for &(frame_node, _) in stack.iter().rev() {
                            if found {
                                break;
                            }
                            cycle_indices.push(frame_node);
                            if frame_node == child {
                                found = true;
                            }
                        }
                        cycle_indices.dedup();

                        // Normalise the cycle for deduplication: rotate so that
                        // the minimum index is first, then check set equality.
                        let mut cycle_sorted = cycle_indices.clone();
                        cycle_sorted.sort_unstable();
                        cycle_sorted.dedup();

                        let already = reported_cycles.contains(&cycle_sorted);

                        if !already {
                            reported_cycles.push(cycle_sorted);
                            let rule_ids: Vec<String> = cycle_indices
                                .iter()
                                .map(|&idx| self.rules[idx].rule_id.clone())
                                .collect();
                            let resolution = format!(
                                "Dependency cycle detected among rules: {}.",
                                rule_ids.join(" → ")
                            );
                            reports.push(ConflictReport {
                                conflict_type: ConflictType::CycleDetected,
                                rule_ids,
                                winner: None,
                                resolution,
                            });
                        }
                    } else if color[child] == 0 {
                        color[child] = 1;
                        parent[child] = Some(node);
                        stack.push((child, 0));
                    }
                    // color[child] == 2: already fully processed, skip.
                } else {
                    // All children of `node` have been visited.
                    color[node] = 2;
                    stack.pop();
                    let _ = &adj; // satisfy borrow checker
                    continue 'dfs;
                }
            }
        }

        // Suppress the unused variable warning for `parent` (it's tracked for
        // potential future use in detailed path reconstruction).
        let _ = parent;

        reports
    }

    /// Run all conflict detectors and return a deduplicated list of reports.
    ///
    /// Combines results from [`detect_head_overlaps`][Self::detect_head_overlaps]
    /// and [`detect_cycles`][Self::detect_cycles].
    pub fn resolve_all(&self) -> Vec<ConflictReport> {
        let mut reports = self.detect_head_overlaps();
        let cycles = self.detect_cycles();
        for cycle in cycles {
            // Deduplicate by rule_ids set.
            let already = reports
                .iter()
                .any(|r| r.conflict_type == cycle.conflict_type && r.rule_ids == cycle.rule_ids);
            if !already {
                reports.push(cycle);
            }
        }
        reports
    }

    /// Find the winning rule for a given goal string.
    ///
    /// Candidates are all rules whose head is a prefix of `goal` (i.e.
    /// `goal.starts_with(&rule.head)`).  The configured strategy is then
    /// applied to select the single best candidate.
    ///
    /// Returns `None` when no rule matches or after degenerate ties that
    /// cannot be broken.
    pub fn winner_for_goal(&self, goal: &str) -> Option<&RuleV2> {
        // Collect (index, rule) pairs whose head matches.
        let candidates: Vec<usize> = self
            .rules
            .iter()
            .enumerate()
            .filter(|(_, r)| goal.starts_with(r.head.as_str()))
            .map(|(idx, _)| idx)
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Fold over candidates to pick the best one.
        let best_idx = candidates.iter().copied().reduce(|acc, idx| {
            match self.pick_winner_indices(acc, idx) {
                Some(winner) => winner,
                // Tie-break: keep the earlier index for stability.
                None => acc,
            }
        })?;

        self.rules.get(best_idx)
    }

    /// Return all rules registered by a specific author.
    pub fn rules_for_author(&self, author: &str) -> Vec<&RuleV2> {
        self.rules.iter().filter(|r| r.author == author).collect()
    }

    /// Return all rules sorted by descending priority.
    ///
    /// Rules with the same priority retain their original registration order
    /// (stable sort).
    pub fn sorted_by_priority(&self) -> Vec<&RuleV2> {
        let mut sorted: Vec<&RuleV2> = self.rules.iter().collect();
        sorted.sort_by_key(|r| std::cmp::Reverse(r.priority)); // descending
        sorted
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_rule(
        id: &str,
        head: &str,
        body: Vec<&str>,
        priority: i32,
        author: &str,
        ts: u64,
    ) -> RuleV2 {
        RuleV2 {
            rule_id: id.to_string(),
            head: head.to_string(),
            body: body.iter().map(|s| s.to_string()).collect(),
            priority,
            author: author.to_string(),
            created_at_secs: ts,
        }
    }

    // ── 1: register adds rule ─────────────────────────────────────────────────

    #[test]
    fn test_register_adds_rule() {
        let mut resolver = RuleConflictResolverV2::new(ResolutionStrategy::HigherPriority);
        assert_eq!(resolver.rules.len(), 0);
        resolver.register(make_rule("r1", "parent", vec![], 5, "alice", 1000));
        assert_eq!(resolver.rules.len(), 1);
        resolver.register(make_rule("r2", "child", vec![], 3, "bob", 2000));
        assert_eq!(resolver.rules.len(), 2);
    }

    // ── 2: no conflicts for non-overlapping rules ─────────────────────────────

    #[test]
    fn test_no_conflicts_non_overlapping() {
        let mut resolver = RuleConflictResolverV2::new(ResolutionStrategy::HigherPriority);
        resolver.register(make_rule("r1", "ancestor", vec![], 5, "alice", 1000));
        resolver.register(make_rule("r2", "sibling", vec![], 3, "bob", 2000));
        let reports = resolver.detect_head_overlaps();
        assert!(reports.is_empty());
    }

    // ── 3: head overlap detected for prefix match ─────────────────────────────

    #[test]
    fn test_head_overlap_prefix_match() {
        let mut resolver = RuleConflictResolverV2::new(ResolutionStrategy::HigherPriority);
        resolver.register(make_rule("r1", "parent", vec![], 10, "alice", 1000));
        resolver.register(make_rule("r2", "parent_of", vec![], 5, "bob", 2000));
        let reports = resolver.detect_head_overlaps();
        assert_eq!(reports.len(), 1);
        assert!(matches!(
            reports[0].conflict_type,
            ConflictType::HeadOverlap
        ));
        assert!(reports[0].rule_ids.contains(&"r1".to_string()));
        assert!(reports[0].rule_ids.contains(&"r2".to_string()));
    }

    // ── 4: higher priority wins with HigherPriority strategy ──────────────────

    #[test]
    fn test_higher_priority_wins() {
        let mut resolver = RuleConflictResolverV2::new(ResolutionStrategy::HigherPriority);
        resolver.register(make_rule("r1", "parent", vec![], 10, "alice", 1000));
        resolver.register(make_rule("r2", "parent_of", vec![], 5, "bob", 2000));
        let reports = resolver.detect_head_overlaps();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].winner, Some("r1".to_string()));
    }

    // ── 5: priority tie generates PriorityTie report ──────────────────────────

    #[test]
    fn test_priority_tie_generates_report() {
        let mut resolver = RuleConflictResolverV2::new(ResolutionStrategy::HigherPriority);
        resolver.register(make_rule("r1", "parent", vec![], 5, "alice", 1000));
        resolver.register(make_rule("r2", "parent_of", vec![], 5, "bob", 2000));
        let reports = resolver.detect_head_overlaps();
        assert_eq!(reports.len(), 1);
        assert!(matches!(
            reports[0].conflict_type,
            ConflictType::PriorityTie
        ));
        assert_eq!(reports[0].winner, None);
    }

    // ── 6: LaterTimestamp strategy picks newer rule ───────────────────────────

    #[test]
    fn test_later_timestamp_strategy() {
        let mut resolver = RuleConflictResolverV2::new(ResolutionStrategy::LaterTimestamp);
        resolver.register(make_rule("r1", "parent", vec![], 5, "alice", 1000));
        resolver.register(make_rule("r2", "parent_of", vec![], 5, "bob", 9999));
        let reports = resolver.detect_head_overlaps();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].winner, Some("r2".to_string()));
    }

    // ── 7: Alphabetical strategy picks lexicographically first ────────────────

    #[test]
    fn test_alphabetical_strategy() {
        let mut resolver = RuleConflictResolverV2::new(ResolutionStrategy::Alphabetical);
        resolver.register(make_rule("zebra_rule", "parent", vec![], 5, "alice", 1000));
        resolver.register(make_rule("alpha_rule", "parent_of", vec![], 5, "bob", 2000));
        let reports = resolver.detect_head_overlaps();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].winner, Some("alpha_rule".to_string()));
    }

    // ── 8: FirstRegistered strategy picks first by index ─────────────────────

    #[test]
    fn test_first_registered_strategy() {
        let mut resolver = RuleConflictResolverV2::new(ResolutionStrategy::FirstRegistered);
        resolver.register(make_rule("r1", "parent", vec![], 5, "alice", 1000));
        resolver.register(make_rule("r2", "parent_of", vec![], 5, "bob", 2000));
        let reports = resolver.detect_head_overlaps();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].winner, Some("r1".to_string()));
    }

    // ── 9: cycle detection A→B→A ─────────────────────────────────────────────

    #[test]
    fn test_cycle_ab_ba() {
        let mut resolver = RuleConflictResolverV2::new(ResolutionStrategy::HigherPriority);
        // A's body contains B's head; B's body contains A's head.
        resolver.register(make_rule("A", "head_a", vec!["head_b"], 5, "alice", 1000));
        resolver.register(make_rule("B", "head_b", vec!["head_a"], 5, "alice", 2000));
        let reports = resolver.detect_cycles();
        assert!(!reports.is_empty());
        assert!(matches!(
            reports[0].conflict_type,
            ConflictType::CycleDetected
        ));
        // Both rules should be present.
        let ids: Vec<&str> = reports[0].rule_ids.iter().map(|s| s.as_str()).collect();
        assert!(ids.contains(&"A"));
        assert!(ids.contains(&"B"));
    }

    // ── 10: cycle detection A→B→C→A ──────────────────────────────────────────

    #[test]
    fn test_cycle_abc() {
        let mut resolver = RuleConflictResolverV2::new(ResolutionStrategy::HigherPriority);
        resolver.register(make_rule("A", "head_a", vec!["head_b"], 5, "alice", 1000));
        resolver.register(make_rule("B", "head_b", vec!["head_c"], 5, "alice", 2000));
        resolver.register(make_rule("C", "head_c", vec!["head_a"], 5, "alice", 3000));
        let reports = resolver.detect_cycles();
        assert!(!reports.is_empty());
        let ids: Vec<&str> = reports[0].rule_ids.iter().map(|s| s.as_str()).collect();
        assert!(ids.contains(&"A"));
        assert!(ids.contains(&"B"));
        assert!(ids.contains(&"C"));
    }

    // ── 11: no cycle for chain A→B→C (no back edge) ──────────────────────────

    #[test]
    fn test_no_cycle_chain_abc() {
        let mut resolver = RuleConflictResolverV2::new(ResolutionStrategy::HigherPriority);
        resolver.register(make_rule("A", "head_a", vec!["head_b"], 5, "alice", 1000));
        resolver.register(make_rule("B", "head_b", vec!["head_c"], 5, "alice", 2000));
        resolver.register(make_rule("C", "head_c", vec![], 5, "alice", 3000));
        let reports = resolver.detect_cycles();
        assert!(reports.is_empty());
    }

    // ── 12: resolve_all returns both overlaps and cycles ─────────────────────

    #[test]
    fn test_resolve_all_combines_reports() {
        let mut resolver = RuleConflictResolverV2::new(ResolutionStrategy::HigherPriority);
        // Head overlap: r1 / r2
        resolver.register(make_rule("r1", "parent", vec!["head_b"], 10, "alice", 1000));
        resolver.register(make_rule("r2", "parent_of", vec!["head_a"], 5, "bob", 2000));
        // Cycle: r1 → r2 (r1's body has "head_b" which is r2's ... actually let's make a separate cycle)
        // Additional dedicated cycle rules.
        resolver.register(make_rule(
            "cA",
            "cycle_a",
            vec!["cycle_b"],
            1,
            "carol",
            3000,
        ));
        resolver.register(make_rule(
            "cB",
            "cycle_b",
            vec!["cycle_a"],
            1,
            "carol",
            4000,
        ));
        let reports = resolver.resolve_all();
        let has_overlap = reports.iter().any(|r| {
            r.conflict_type == ConflictType::HeadOverlap
                || r.conflict_type == ConflictType::PriorityTie
        });
        let has_cycle = reports
            .iter()
            .any(|r| r.conflict_type == ConflictType::CycleDetected);
        assert!(
            has_overlap,
            "Expected at least one HeadOverlap or PriorityTie report"
        );
        assert!(has_cycle, "Expected at least one CycleDetected report");
    }

    // ── 13: winner_for_goal returns highest priority matching rule ─────────────

    #[test]
    fn test_winner_for_goal_highest_priority() {
        let mut resolver = RuleConflictResolverV2::new(ResolutionStrategy::HigherPriority);
        resolver.register(make_rule("r1", "parent", vec![], 5, "alice", 1000));
        resolver.register(make_rule("r2", "parent", vec![], 10, "bob", 2000));
        resolver.register(make_rule("r3", "parent_extra", vec![], 3, "carol", 3000));
        let winner = resolver.winner_for_goal("parent(X,Y)");
        assert!(winner.is_some());
        assert_eq!(winner.map(|r| r.rule_id.as_str()), Some("r2"));
    }

    // ── 14: winner_for_goal returns None for unknown goal ─────────────────────

    #[test]
    fn test_winner_for_goal_unknown() {
        let mut resolver = RuleConflictResolverV2::new(ResolutionStrategy::HigherPriority);
        resolver.register(make_rule("r1", "parent", vec![], 5, "alice", 1000));
        let winner = resolver.winner_for_goal("ancestor(X,Y)");
        assert!(winner.is_none());
    }

    // ── 15: rules_for_author filters correctly ────────────────────────────────

    #[test]
    fn test_rules_for_author() {
        let mut resolver = RuleConflictResolverV2::new(ResolutionStrategy::HigherPriority);
        resolver.register(make_rule("r1", "parent", vec![], 5, "alice", 1000));
        resolver.register(make_rule("r2", "child", vec![], 3, "bob", 2000));
        resolver.register(make_rule("r3", "sibling", vec![], 4, "alice", 3000));
        let alice_rules = resolver.rules_for_author("alice");
        assert_eq!(alice_rules.len(), 2);
        let ids: Vec<&str> = alice_rules.iter().map(|r| r.rule_id.as_str()).collect();
        assert!(ids.contains(&"r1"));
        assert!(ids.contains(&"r3"));
        let bob_rules = resolver.rules_for_author("bob");
        assert_eq!(bob_rules.len(), 1);
    }

    // ── 16: sorted_by_priority descending ────────────────────────────────────

    #[test]
    fn test_sorted_by_priority_descending() {
        let mut resolver = RuleConflictResolverV2::new(ResolutionStrategy::HigherPriority);
        resolver.register(make_rule("r1", "a", vec![], 3, "alice", 1000));
        resolver.register(make_rule("r2", "b", vec![], 10, "bob", 2000));
        resolver.register(make_rule("r3", "c", vec![], 7, "carol", 3000));
        let sorted = resolver.sorted_by_priority();
        assert_eq!(sorted[0].priority, 10);
        assert_eq!(sorted[1].priority, 7);
        assert_eq!(sorted[2].priority, 3);
    }

    // ── 17: ConflictReport has winner field set correctly ─────────────────────

    #[test]
    fn test_conflict_report_winner_field() {
        let mut resolver = RuleConflictResolverV2::new(ResolutionStrategy::LaterTimestamp);
        resolver.register(make_rule("old_rule", "fact", vec![], 5, "alice", 100));
        resolver.register(make_rule(
            "new_rule",
            "fact_detail",
            vec![],
            5,
            "alice",
            9999,
        ));
        let reports = resolver.detect_head_overlaps();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].winner, Some("new_rule".to_string()));
        assert!(!reports[0].resolution.is_empty());
    }

    // ── 18: head overlap with same head (exact match) ─────────────────────────

    #[test]
    fn test_head_overlap_exact_match() {
        let mut resolver = RuleConflictResolverV2::new(ResolutionStrategy::HigherPriority);
        resolver.register(make_rule("r1", "same_head", vec![], 5, "alice", 1000));
        resolver.register(make_rule("r2", "same_head", vec![], 8, "bob", 2000));
        let reports = resolver.detect_head_overlaps();
        assert_eq!(reports.len(), 1);
        // "same_head".starts_with("same_head") is true, so this is an overlap.
        assert!(matches!(
            reports[0].conflict_type,
            ConflictType::HeadOverlap | ConflictType::PriorityTie
        ));
        // r2 has higher priority.
        assert_eq!(reports[0].winner, Some("r2".to_string()));
    }
}
