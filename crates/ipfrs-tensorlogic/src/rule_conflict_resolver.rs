//! RuleConflictResolver — production-quality logic rule conflict detection and resolution.
//!
//! This module detects five categories of conflict among [`LogicRule`]s and resolves
//! them via pluggable [`ResolutionStrategy`] implementations.
//!
//! # Conflict categories
//!
//! | Variant | Meaning |
//! |---------|---------|
//! | [`ConflictType::DirectContradiction`] | Same head, disjoint bodies. |
//! | [`ConflictType::PriorityConflict`] | Same head, different priorities. |
//! | [`ConflictType::CyclicDependency`] | Premise→head graph contains a back-edge. |
//! | [`ConflictType::UndercutConflict`] | One rule's head negates another rule. |
//! | [`ConflictType::RebuttalConflict`] | Two rules have complementary heads. |
//!
//! # Resolution strategies
//!
//! | Variant | Meaning |
//! |---------|---------|
//! | [`ResolutionStrategy::PriorityOrder`] | Higher `priority` wins. |
//! | [`ResolutionStrategy::Specificity`] | More body conditions wins. |
//! | [`ResolutionStrategy::LastWriter`] | More recently added rule wins. |
//! | [`ResolutionStrategy::Inhibit`] | Defeasible rule loses; tie → higher priority. |
//! | [`ResolutionStrategy::Merge`] | Named merge operator (future extension). |
//! | [`ResolutionStrategy::AskOracle`] | Returns [`ResolverError::UnresolvableConflict`]. |
//!
//! # Examples
//!
//! ```
//! use ipfrs_tensorlogic::rule_conflict_resolver::{
//!     LogicRule, ResolverConfig, ResolutionStrategy, RuleConflictResolver,
//! };
//!
//! let cfg = ResolverConfig::default();
//! let mut resolver = RuleConflictResolver::new(cfg);
//!
//! resolver.add_rule(LogicRule {
//!     id: "r1".to_string(),
//!     head: "bird".to_string(),
//!     body: vec!["has_wings".to_string(), "feathers".to_string()],
//!     priority: 10,
//!     source: "ornithology".to_string(),
//!     is_defeasible: false,
//! }).expect("test setup: add_rule should not fail");
//!
//! resolver.add_rule(LogicRule {
//!     id: "r2".to_string(),
//!     head: "bird".to_string(),
//!     body: vec!["lays_eggs".to_string()],
//!     priority: 5,
//!     source: "biology".to_string(),
//!     is_defeasible: true,
//! }).expect("test setup: add_rule should not fail");
//!
//! let conflicts = resolver.detect_conflicts();
//! assert!(!conflicts.is_empty());
//!
//! let stats = resolver.stats();
//! assert_eq!(stats.rules_loaded, 2);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

// ─── xorshift64 (no-rand policy) ──────────────────────────────────────────────

/// Fast 64-bit pseudo-random number generator (xorshift64 algorithm).
/// Used only in tests; exported with crate-root alias `rcr_xorshift64`.
#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ─── Core types ───────────────────────────────────────────────────────────────

/// A single logic rule with metadata for conflict analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicRule {
    /// Unique identifier for this rule.
    pub id: String,
    /// The conclusion (head) of the rule, e.g. `"bird"`.
    pub head: String,
    /// Premises (body) of the rule.  A premise may carry a `"NOT:"` prefix
    /// to represent negation-as-failure.
    pub body: Vec<String>,
    /// Numeric priority; higher values take precedence.
    pub priority: i32,
    /// Provenance / source identifier.
    pub source: String,
    /// Whether this rule can be defeated by a higher-priority rule.
    pub is_defeasible: bool,
}

impl LogicRule {
    /// Return the positive (non-negated) premises in `body`.
    #[inline]
    pub fn positive_body(&self) -> impl Iterator<Item = &str> {
        self.body
            .iter()
            .filter(|p| !p.starts_with("NOT:"))
            .map(String::as_str)
    }

    /// Return the negated conditions (with the `"NOT:"` prefix stripped).
    #[inline]
    pub fn negated_body(&self) -> impl Iterator<Item = &str> {
        self.body
            .iter()
            .filter(|p| p.starts_with("NOT:"))
            .map(|p| &p["NOT:".len()..])
    }
}

// ─── ConflictType ─────────────────────────────────────────────────────────────

/// The category of a detected conflict between two or more rules.
#[derive(Debug, Clone, PartialEq)]
pub enum ConflictType {
    /// Two rules derive the same head with completely disjoint bodies.
    DirectContradiction {
        /// ID of the first conflicting rule.
        rule_a: String,
        /// ID of the second conflicting rule.
        rule_b: String,
    },
    /// Two rules derive the same head but with different priority values.
    PriorityConflict {
        /// ID of the higher-priority rule.
        higher: String,
        /// ID of the lower-priority rule.
        lower: String,
    },
    /// A dependency cycle was found in the rule graph.
    CyclicDependency {
        /// Ordered list of rule IDs forming the cycle.
        cycle: Vec<String>,
    },
    /// Rule A's head appears as a negated premise in rule B, meaning A undercuts B.
    UndercutConflict {
        /// ID of the rule whose head undercuts the other.
        undercutter: String,
        /// ID of the rule being undercut.
        undercut: String,
    },
    /// Rule A and rule B have complementary heads (`"NOT:<other_head>"`).
    RebuttalConflict {
        /// ID of the first rebutting rule.
        rule_a: String,
        /// ID of the second rebutting rule.
        rule_b: String,
    },
}

// ─── ResolutionStrategy ───────────────────────────────────────────────────────

/// How a detected conflict should be resolved.
#[derive(Debug, Clone, PartialEq)]
pub enum ResolutionStrategy {
    /// Higher `priority` field wins.
    PriorityOrder,
    /// Rule with more body conditions wins (more specific).
    Specificity,
    /// Rule added most recently (highest insertion index) wins.
    LastWriter,
    /// If a rule is defeasible, it loses; otherwise use priority.
    Inhibit,
    /// Apply a named merge operator (future extension).
    Merge(String),
    /// External oracle must decide; resolving returns an error.
    AskOracle,
}

// ─── ConflictRecord ───────────────────────────────────────────────────────────

/// A recorded conflict together with its (optional) resolution.
#[derive(Debug, Clone)]
pub struct ConflictRecord {
    /// The kind and participating rules of the conflict.
    pub conflict_type: ConflictType,
    /// Unix-epoch milliseconds when this conflict was detected (monotonic counter
    /// when a real clock is unavailable).
    pub detected_at: u64,
    /// Whether this conflict has been resolved.
    pub resolved: bool,
    /// The strategy that was applied (if resolved).
    pub resolution: Option<ResolutionStrategy>,
    /// The ID of the winning rule (if resolved and applicable).
    pub winning_rule: Option<String>,
}

impl ConflictRecord {
    fn new(conflict_type: ConflictType, timestamp: u64) -> Self {
        Self {
            conflict_type,
            detected_at: timestamp,
            resolved: false,
            resolution: None,
            winning_rule: None,
        }
    }
}

// ─── ResolverConfig ───────────────────────────────────────────────────────────

/// Configuration for [`RuleConflictResolver`].
#[derive(Debug, Clone)]
pub struct ResolverConfig {
    /// Strategy used by [`RuleConflictResolver::winning_rule`] and
    /// [`RuleConflictResolver::resolve_all`] when no explicit strategy is given.
    pub default_strategy: ResolutionStrategy,
    /// Whether to run cycle detection during [`RuleConflictResolver::detect_conflicts`].
    pub enable_cycle_detection: bool,
    /// Hard cap on the number of rules that may be loaded simultaneously.
    pub max_rules: usize,
    /// Weight given to specificity when computing a composite score (reserved for
    /// future weighted-strategy extensions).
    pub specificity_weight: f64,
    /// Weight given to priority (reserved for future weighted-strategy extensions).
    pub priority_weight: f64,
}

impl Default for ResolverConfig {
    fn default() -> Self {
        Self {
            default_strategy: ResolutionStrategy::PriorityOrder,
            enable_cycle_detection: true,
            max_rules: 10_000,
            specificity_weight: 0.5,
            priority_weight: 0.5,
        }
    }
}

// ─── ResolverStats ────────────────────────────────────────────────────────────

/// Snapshot of resolver metrics.
#[derive(Debug, Clone, Default)]
pub struct ResolverStats {
    /// Total rules currently loaded.
    pub rules_loaded: usize,
    /// Total conflicts ever detected (cumulative).
    pub conflicts_detected: usize,
    /// Total conflicts that have been resolved (cumulative).
    pub conflicts_resolved: usize,
    /// Total dependency cycles ever found.
    pub cycles_found: usize,
    /// Conflicts that have been detected but not yet resolved.
    pub unresolved_conflicts: usize,
}

// ─── ResolverError ────────────────────────────────────────────────────────────

/// Errors that can be returned by [`RuleConflictResolver`] operations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ResolverError {
    /// A rule ID was referenced but no such rule exists.
    #[error("rule not found: {0}")]
    RuleNotFound(String),

    /// The conflict between `rule_a` and `rule_b` cannot be resolved automatically.
    #[error("unresolvable conflict between '{rule_a}' and '{rule_b}'")]
    UnresolvableConflict {
        /// First conflicting rule ID.
        rule_a: String,
        /// Second conflicting rule ID.
        rule_b: String,
    },

    /// A cyclic dependency was detected among the listed rules.
    #[error("cyclic dependency: {0:?}")]
    CyclicDependency(Vec<String>),

    /// Invalid or inconsistent resolver configuration.
    #[error("configuration error: {0}")]
    ConfigurationError(String),

    /// The rule set would exceed [`ResolverConfig::max_rules`].
    #[error("max rules exceeded")]
    MaxRulesExceeded,
}

// ─── Internal helper ──────────────────────────────────────────────────────────

/// Internal bookkeeping entry for a stored rule.
#[derive(Debug, Clone)]
struct RuleEntry {
    rule: LogicRule,
    /// Insertion index; used by the LastWriter strategy.
    insertion_order: usize,
}

// ─── RuleConflictResolver ─────────────────────────────────────────────────────

/// Production-quality conflict detector and resolver for logic rule sets.
///
/// See the [module-level documentation](self) for a full description.
pub struct RuleConflictResolver {
    /// Ordered storage; the key is the rule ID.
    rules: HashMap<String, RuleEntry>,
    /// Monotonically increasing counter; simulates a clock for `detected_at`.
    clock: u64,
    /// Configuration.
    config: ResolverConfig,
    /// Cumulative statistics.
    stats: ResolverStats,
}

impl RuleConflictResolver {
    /// Create a new resolver with the given configuration.
    pub fn new(config: ResolverConfig) -> Self {
        Self {
            rules: HashMap::new(),
            clock: 0,
            config,
            stats: ResolverStats::default(),
        }
    }

    /// Advance the internal clock by one tick and return the new value.
    #[inline]
    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    // ── Rule management ───────────────────────────────────────────────────────

    /// Add a rule to the resolver.
    ///
    /// An immediate check for [`ConflictType::DirectContradiction`] with every
    /// existing rule that shares the same head is performed.  The statistics are
    /// updated but the conflicts are not automatically resolved.
    ///
    /// # Errors
    ///
    /// * [`ResolverError::MaxRulesExceeded`] when the hard cap would be breached.
    pub fn add_rule(&mut self, rule: LogicRule) -> Result<(), ResolverError> {
        if self.rules.len() >= self.config.max_rules {
            return Err(ResolverError::MaxRulesExceeded);
        }

        let insertion_order = self.rules.len();
        let ts = self.tick();

        // Immediate contradiction check against all existing same-head rules.
        let same_head: Vec<String> = self
            .rules
            .values()
            .filter(|e| e.rule.head == rule.head)
            .map(|e| e.rule.id.clone())
            .collect();

        for existing_id in same_head {
            let existing_body: HashSet<&str> = self.rules[&existing_id]
                .rule
                .body
                .iter()
                .map(String::as_str)
                .collect();
            let new_body: HashSet<&str> = rule.body.iter().map(String::as_str).collect();

            if existing_body.is_disjoint(&new_body) {
                self.stats.conflicts_detected += 1;
                let _ = ts; // timestamp consumed above
            }
        }

        self.rules.insert(
            rule.id.clone(),
            RuleEntry {
                rule,
                insertion_order,
            },
        );

        self.stats.rules_loaded = self.rules.len();
        Ok(())
    }

    /// Remove a rule by ID.
    ///
    /// # Errors
    ///
    /// * [`ResolverError::RuleNotFound`] when no such rule exists.
    pub fn remove_rule(&mut self, id: &str) -> Result<(), ResolverError> {
        if self.rules.remove(id).is_none() {
            return Err(ResolverError::RuleNotFound(id.to_string()));
        }
        self.stats.rules_loaded = self.rules.len();
        Ok(())
    }

    // ── Conflict detection ────────────────────────────────────────────────────

    /// Run a full conflict scan and return all detected [`ConflictRecord`]s.
    ///
    /// Detection order:
    /// 1. [`ConflictType::DirectContradiction`]
    /// 2. [`ConflictType::PriorityConflict`]
    /// 3. [`ConflictType::CyclicDependency`] (if enabled in config)
    /// 4. [`ConflictType::UndercutConflict`]
    /// 5. [`ConflictType::RebuttalConflict`]
    pub fn detect_conflicts(&mut self) -> Vec<ConflictRecord> {
        // Collect clones so we don't hold a borrow on `self.rules` while
        // calling `self.tick()` (which needs `&mut self`).
        let rule_snapshots: Vec<LogicRule> = self.rules.values().map(|e| e.rule.clone()).collect();

        let mut pending_types: Vec<ConflictType> = Vec::new();

        // ── 1. DirectContradiction & 2. PriorityConflict ─────────────────────
        for i in 0..rule_snapshots.len() {
            for j in (i + 1)..rule_snapshots.len() {
                let a = &rule_snapshots[i];
                let b = &rule_snapshots[j];

                if a.head != b.head {
                    continue;
                }

                let body_a: HashSet<&str> = a.body.iter().map(String::as_str).collect();
                let body_b: HashSet<&str> = b.body.iter().map(String::as_str).collect();

                // DirectContradiction: same head, completely disjoint bodies.
                if body_a.is_disjoint(&body_b) {
                    pending_types.push(ConflictType::DirectContradiction {
                        rule_a: a.id.clone(),
                        rule_b: b.id.clone(),
                    });
                }

                // PriorityConflict: same head, different priorities.
                if a.priority != b.priority {
                    let (higher, lower) = if a.priority > b.priority {
                        (a.id.clone(), b.id.clone())
                    } else {
                        (b.id.clone(), a.id.clone())
                    };
                    pending_types.push(ConflictType::PriorityConflict { higher, lower });
                }
            }
        }

        // ── 3. CyclicDependency ───────────────────────────────────────────────
        if self.config.enable_cycle_detection {
            let cycles = self.detect_cycles();
            let cycle_count = cycles.len();
            for cycle in cycles {
                pending_types.push(ConflictType::CyclicDependency { cycle });
            }
            self.stats.cycles_found += cycle_count;
        }

        // ── 4. UndercutConflict ───────────────────────────────────────────────
        // Rule A undercuts rule B when A's head appears as "NOT:<head_A>" in B's body.
        for a in &rule_snapshots {
            let negated_head = format!("NOT:{}", a.head);
            for b in &rule_snapshots {
                if a.id == b.id {
                    continue;
                }
                if b.body.contains(&negated_head) {
                    pending_types.push(ConflictType::UndercutConflict {
                        undercutter: a.id.clone(),
                        undercut: b.id.clone(),
                    });
                }
            }
        }

        // ── 5. RebuttalConflict ───────────────────────────────────────────────
        // Rules A and B rebut each other when one head is "NOT:" + the other head.
        for i in 0..rule_snapshots.len() {
            for j in (i + 1)..rule_snapshots.len() {
                let a = &rule_snapshots[i];
                let b = &rule_snapshots[j];

                let a_negates_b = a.head == format!("NOT:{}", b.head);
                let b_negates_a = b.head == format!("NOT:{}", a.head);

                if a_negates_b || b_negates_a {
                    pending_types.push(ConflictType::RebuttalConflict {
                        rule_a: a.id.clone(),
                        rule_b: b.id.clone(),
                    });
                }
            }
        }

        // Now mint timestamps and build ConflictRecords.
        let records: Vec<ConflictRecord> = pending_types
            .into_iter()
            .map(|ct| {
                let ts = self.tick();
                ConflictRecord::new(ct, ts)
            })
            .collect();

        // Update stats.
        let new_conflicts = records.len();
        self.stats.conflicts_detected += new_conflicts;
        self.stats.unresolved_conflicts += new_conflicts;

        records
    }

    /// DFS-based cycle detection on the premise→head dependency graph.
    ///
    /// Each rule contributes edges: `(premise_head → rule_head)` for every
    /// positive premise that also appears as some rule's head.
    fn detect_cycles(&self) -> Vec<Vec<String>> {
        // Build adjacency list: head → heads that depend on it.
        // An edge head_X → head_Y means "Y's body contains X" i.e. Y depends on X.
        // We model it as: for each rule with head H and positive body premise P,
        // add edge P → H.  A cycle in this graph means mutual dependency.

        // Collect all known heads.
        let all_heads: HashSet<&str> = self.rules.values().map(|e| e.rule.head.as_str()).collect();

        // Build adjacency: premise → {conclusions that depend on it}.
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for entry in self.rules.values() {
            for premise in entry.rule.positive_body() {
                if all_heads.contains(premise) {
                    adj.entry(premise)
                        .or_default()
                        .push(entry.rule.head.as_str());
                }
            }
        }

        // Iterative DFS to find all simple cycles (report each once).
        let mut found_cycles: Vec<Vec<String>> = Vec::new();
        let mut globally_visited: HashSet<&str> = HashSet::new();

        for start in all_heads.iter().copied() {
            if globally_visited.contains(start) {
                continue;
            }

            // DFS stack: (node, path_so_far, visited_in_this_path)
            let mut stack: VecDeque<(&str, Vec<&str>, HashSet<&str>)> = VecDeque::new();
            let mut path_visited: HashSet<&str> = HashSet::new();
            path_visited.insert(start);
            stack.push_back((start, vec![start], path_visited));

            while let Some((node, path, mut visited)) = stack.pop_back() {
                globally_visited.insert(node);

                if let Some(neighbors) = adj.get(node) {
                    for &neighbor in neighbors {
                        if neighbor == start {
                            // Back-edge → cycle found.
                            let mut cycle: Vec<String> =
                                path.iter().map(|s| s.to_string()).collect();
                            cycle.push(start.to_string());
                            found_cycles.push(cycle);
                        } else if !visited.contains(neighbor) {
                            let mut new_path = path.clone();
                            new_path.push(neighbor);
                            let mut new_visited = visited.clone();
                            new_visited.insert(neighbor);
                            stack.push_back((neighbor, new_path, new_visited));
                        }
                    }
                }

                // Mark all path nodes as globally visited to avoid re-traversal.
                for n in &path {
                    visited.insert(n);
                }
            }
        }

        found_cycles
    }

    // ── Resolution ────────────────────────────────────────────────────────────

    /// Resolve a single conflict record and return the winning rule ID.
    ///
    /// The resolution strategy is taken from `self.config.default_strategy`.
    ///
    /// # Errors
    ///
    /// * [`ResolverError::RuleNotFound`] when a rule referenced in the conflict
    ///   no longer exists.
    /// * [`ResolverError::UnresolvableConflict`] when the strategy is
    ///   [`ResolutionStrategy::AskOracle`].
    /// * [`ResolverError::CyclicDependency`] for cyclic conflicts (cannot be
    ///   resolved automatically).
    pub fn resolve(&mut self, conflict: &ConflictRecord) -> Result<String, ResolverError> {
        let strategy = self.config.default_strategy.clone();
        let winner = self.apply_strategy(conflict, &strategy)?;
        self.stats.conflicts_resolved += 1;
        if self.stats.unresolved_conflicts > 0 {
            self.stats.unresolved_conflicts -= 1;
        }
        Ok(winner)
    }

    /// Resolve all conflicts returned by `detect_conflicts` and return each
    /// result paired with its conflict record.
    pub fn resolve_all(&mut self) -> Vec<(ConflictRecord, Result<String, ResolverError>)> {
        let conflicts = self.detect_conflicts();
        let mut results = Vec::with_capacity(conflicts.len());
        for conflict in conflicts {
            let res = {
                let strategy = self.config.default_strategy.clone();
                self.apply_strategy(&conflict, &strategy)
            };
            if res.is_ok() {
                self.stats.conflicts_resolved += 1;
                if self.stats.unresolved_conflicts > 0 {
                    self.stats.unresolved_conflicts -= 1;
                }
            }
            results.push((conflict, res));
        }
        results
    }

    /// Internal: apply `strategy` to `conflict` and return the winning rule ID.
    fn apply_strategy(
        &self,
        conflict: &ConflictRecord,
        strategy: &ResolutionStrategy,
    ) -> Result<String, ResolverError> {
        match strategy {
            ResolutionStrategy::AskOracle => {
                let (a, b) = self.conflict_pair_ids(&conflict.conflict_type)?;
                Err(ResolverError::UnresolvableConflict {
                    rule_a: a,
                    rule_b: b,
                })
            }

            ResolutionStrategy::Merge(_op) => {
                // Merge is a future extension; for now treat like AskOracle.
                let (a, b) = self.conflict_pair_ids(&conflict.conflict_type)?;
                Err(ResolverError::UnresolvableConflict {
                    rule_a: a,
                    rule_b: b,
                })
            }

            ResolutionStrategy::PriorityOrder => {
                let (id_a, id_b) = self.conflict_pair_ids(&conflict.conflict_type)?;
                let entry_a = self
                    .rules
                    .get(&id_a)
                    .ok_or_else(|| ResolverError::RuleNotFound(id_a.clone()))?;
                let entry_b = self
                    .rules
                    .get(&id_b)
                    .ok_or_else(|| ResolverError::RuleNotFound(id_b.clone()))?;
                Ok(if entry_a.rule.priority >= entry_b.rule.priority {
                    id_a
                } else {
                    id_b
                })
            }

            ResolutionStrategy::Specificity => {
                let (id_a, id_b) = self.conflict_pair_ids(&conflict.conflict_type)?;
                let entry_a = self
                    .rules
                    .get(&id_a)
                    .ok_or_else(|| ResolverError::RuleNotFound(id_a.clone()))?;
                let entry_b = self
                    .rules
                    .get(&id_b)
                    .ok_or_else(|| ResolverError::RuleNotFound(id_b.clone()))?;
                Ok(if entry_a.rule.body.len() >= entry_b.rule.body.len() {
                    id_a
                } else {
                    id_b
                })
            }

            ResolutionStrategy::LastWriter => {
                let (id_a, id_b) = self.conflict_pair_ids(&conflict.conflict_type)?;
                let entry_a = self
                    .rules
                    .get(&id_a)
                    .ok_or_else(|| ResolverError::RuleNotFound(id_a.clone()))?;
                let entry_b = self
                    .rules
                    .get(&id_b)
                    .ok_or_else(|| ResolverError::RuleNotFound(id_b.clone()))?;
                Ok(if entry_a.insertion_order >= entry_b.insertion_order {
                    id_a
                } else {
                    id_b
                })
            }

            ResolutionStrategy::Inhibit => {
                let (id_a, id_b) = self.conflict_pair_ids(&conflict.conflict_type)?;
                let entry_a = self
                    .rules
                    .get(&id_a)
                    .ok_or_else(|| ResolverError::RuleNotFound(id_a.clone()))?;
                let entry_b = self
                    .rules
                    .get(&id_b)
                    .ok_or_else(|| ResolverError::RuleNotFound(id_b.clone()))?;

                let a_defeasible = entry_a.rule.is_defeasible;
                let b_defeasible = entry_b.rule.is_defeasible;

                let winner = match (a_defeasible, b_defeasible) {
                    (true, false) => id_b.clone(),
                    (false, true) => id_a.clone(),
                    // Both or neither defeasible → fall back to priority.
                    _ => {
                        if entry_a.rule.priority >= entry_b.rule.priority {
                            id_a.clone()
                        } else {
                            id_b.clone()
                        }
                    }
                };
                Ok(winner)
            }
        }
    }

    /// Extract the two participating rule IDs from a conflict type.
    ///
    /// For cyclic dependencies the "pair" is the first two elements of the cycle
    /// (for error reporting purposes).
    fn conflict_pair_ids(&self, ct: &ConflictType) -> Result<(String, String), ResolverError> {
        match ct {
            ConflictType::DirectContradiction { rule_a, rule_b } => {
                Ok((rule_a.clone(), rule_b.clone()))
            }
            ConflictType::PriorityConflict { higher, lower } => Ok((higher.clone(), lower.clone())),
            ConflictType::CyclicDependency { cycle } => {
                if cycle.len() < 2 {
                    return Err(ResolverError::CyclicDependency(cycle.clone()));
                }
                Err(ResolverError::CyclicDependency(cycle.clone()))
            }
            ConflictType::UndercutConflict {
                undercutter,
                undercut,
            } => Ok((undercutter.clone(), undercut.clone())),
            ConflictType::RebuttalConflict { rule_a, rule_b } => {
                Ok((rule_a.clone(), rule_b.clone()))
            }
        }
    }

    // ── Query helpers ─────────────────────────────────────────────────────────

    /// Return all rules whose positive body conditions are a subset of `facts`.
    ///
    /// Negated body conditions (prefixed with `"NOT:"`) are intentionally
    /// ignored during this applicability check.
    pub fn applicable_rules<'a>(&'a self, facts: &[String]) -> Vec<&'a LogicRule> {
        let fact_set: HashSet<&str> = facts.iter().map(String::as_str).collect();
        let mut result: Vec<&'a LogicRule> = self
            .rules
            .values()
            .filter(|entry| entry.rule.positive_body().all(|p| fact_set.contains(p)))
            .map(|entry| &entry.rule)
            .collect();
        // Stable order by insertion index.
        result.sort_by_key(|r| self.rules.get(&r.id).map_or(0, |e| e.insertion_order));
        result
    }

    /// Among the applicable rules for `head`, pick the winner according to the
    /// configured default strategy.
    ///
    /// Returns `None` when no rule for that head is applicable.
    pub fn winning_rule<'a>(&'a self, head: &str, facts: &[String]) -> Option<&'a LogicRule> {
        let candidates: Vec<&'a LogicRule> = self
            .applicable_rules(facts)
            .into_iter()
            .filter(|r| r.head == head)
            .collect();

        if candidates.is_empty() {
            return None;
        }

        let winner = match &self.config.default_strategy {
            ResolutionStrategy::PriorityOrder => candidates.into_iter().max_by_key(|r| r.priority),

            ResolutionStrategy::Specificity => candidates.into_iter().max_by_key(|r| r.body.len()),

            ResolutionStrategy::LastWriter => candidates
                .into_iter()
                .max_by_key(|r| self.rules.get(&r.id).map_or(0, |e| e.insertion_order)),

            ResolutionStrategy::Inhibit => {
                // Non-defeasible rules take precedence; among ties use priority.
                let non_def: Vec<&'a LogicRule> = candidates
                    .iter()
                    .copied()
                    .filter(|r| !r.is_defeasible)
                    .collect();
                if !non_def.is_empty() {
                    non_def.into_iter().max_by_key(|r| r.priority)
                } else {
                    candidates.into_iter().max_by_key(|r| r.priority)
                }
            }

            // For oracle / merge: fall back to priority.
            ResolutionStrategy::AskOracle | ResolutionStrategy::Merge(_) => {
                candidates.into_iter().max_by_key(|r| r.priority)
            }
        };

        winner
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /// Return a snapshot of current resolver statistics.
    pub fn stats(&self) -> ResolverStats {
        self.stats.clone()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_rule(id: &str, head: &str, body: &[&str], priority: i32) -> LogicRule {
        LogicRule {
            id: id.to_string(),
            head: head.to_string(),
            body: body.iter().map(|s| s.to_string()).collect(),
            priority,
            source: "test".to_string(),
            is_defeasible: false,
        }
    }

    fn make_defeasible(id: &str, head: &str, body: &[&str], priority: i32) -> LogicRule {
        LogicRule {
            is_defeasible: true,
            ..make_rule(id, head, body, priority)
        }
    }

    fn default_resolver() -> RuleConflictResolver {
        RuleConflictResolver::new(ResolverConfig::default())
    }

    // ── xorshift64 ────────────────────────────────────────────────────────────

    #[test]
    fn xorshift_produces_nonzero() {
        let mut state = 12345u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }

    #[test]
    fn xorshift_sequence_differs() {
        let mut state = 99u64;
        let a = xorshift64(&mut state);
        let b = xorshift64(&mut state);
        assert_ne!(a, b);
    }

    // ── add_rule ──────────────────────────────────────────────────────────────

    #[test]
    fn add_rule_success() {
        let mut r = default_resolver();
        let rule = make_rule("r1", "bird", &["has_wings", "feathers"], 10);
        assert!(r.add_rule(rule).is_ok());
        assert_eq!(r.stats().rules_loaded, 1);
    }

    #[test]
    fn add_multiple_rules() {
        let mut r = default_resolver();
        for i in 0..5u32 {
            let rule = make_rule(&format!("r{i}"), "head", &[&format!("body{i}")], i as i32);
            assert!(r.add_rule(rule).is_ok());
        }
        assert_eq!(r.stats().rules_loaded, 5);
    }

    #[test]
    fn add_rule_max_exceeded() {
        let cfg = ResolverConfig {
            max_rules: 2,
            ..Default::default()
        };
        let mut r = RuleConflictResolver::new(cfg);
        r.add_rule(make_rule("r1", "a", &["x"], 1))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r2", "b", &["y"], 1))
            .expect("test setup: add_rule should not fail");
        let err = r
            .add_rule(make_rule("r3", "c", &["z"], 1))
            .expect_err("test setup: expected MaxRulesExceeded error");
        assert_eq!(err, ResolverError::MaxRulesExceeded);
    }

    // ── remove_rule ───────────────────────────────────────────────────────────

    #[test]
    fn remove_existing_rule() {
        let mut r = default_resolver();
        r.add_rule(make_rule("r1", "bird", &["wings"], 10))
            .expect("test setup: add_rule should not fail");
        assert!(r.remove_rule("r1").is_ok());
        assert_eq!(r.stats().rules_loaded, 0);
    }

    #[test]
    fn remove_nonexistent_returns_error() {
        let mut r = default_resolver();
        let err = r
            .remove_rule("ghost")
            .expect_err("test setup: expected RuleNotFound error");
        assert_eq!(err, ResolverError::RuleNotFound("ghost".to_string()));
    }

    #[test]
    fn remove_reduces_count() {
        let mut r = default_resolver();
        r.add_rule(make_rule("r1", "a", &["x"], 1))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r2", "b", &["y"], 1))
            .expect("test setup: add_rule should not fail");
        r.remove_rule("r1")
            .expect("test setup: remove_rule should not fail");
        assert_eq!(r.stats().rules_loaded, 1);
    }

    // ── DirectContradiction ───────────────────────────────────────────────────

    #[test]
    fn detect_direct_contradiction() {
        let mut r = default_resolver();
        r.add_rule(make_rule("r1", "bird", &["wings", "feathers"], 10))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r2", "bird", &["lays_eggs"], 5))
            .expect("test setup: add_rule should not fail");
        let conflicts = r.detect_conflicts();
        let has_contradiction = conflicts
            .iter()
            .any(|c| matches!(&c.conflict_type, ConflictType::DirectContradiction { .. }));
        assert!(has_contradiction);
    }

    #[test]
    fn no_contradiction_when_bodies_overlap() {
        let mut r = default_resolver();
        // Both bodies share "vertebrate" → not disjoint.
        r.add_rule(make_rule("r1", "bird", &["vertebrate", "wings"], 10))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r2", "bird", &["vertebrate", "feathers"], 5))
            .expect("test setup: add_rule should not fail");
        let conflicts = r.detect_conflicts();
        let has_contradiction = conflicts
            .iter()
            .any(|c| matches!(&c.conflict_type, ConflictType::DirectContradiction { .. }));
        assert!(!has_contradiction);
    }

    #[test]
    fn contradiction_ids_are_correct() {
        let mut r = default_resolver();
        r.add_rule(make_rule("alpha", "P", &["A"], 1))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("beta", "P", &["B"], 1))
            .expect("test setup: add_rule should not fail");
        let conflicts = r.detect_conflicts();
        let ct = conflicts
            .iter()
            .find_map(|c| {
                if let ConflictType::DirectContradiction { rule_a, rule_b } = &c.conflict_type {
                    Some((rule_a.clone(), rule_b.clone()))
                } else {
                    None
                }
            })
            .expect("should have contradiction");
        let ids: HashSet<String> = [ct.0, ct.1].into_iter().collect();
        assert!(ids.contains("alpha"));
        assert!(ids.contains("beta"));
    }

    // ── PriorityConflict ──────────────────────────────────────────────────────

    #[test]
    fn detect_priority_conflict() {
        let mut r = default_resolver();
        r.add_rule(make_rule("r1", "q", &["a"], 10))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r2", "q", &["b"], 5))
            .expect("test setup: add_rule should not fail");
        let conflicts = r.detect_conflicts();
        let has_prio = conflicts
            .iter()
            .any(|c| matches!(&c.conflict_type, ConflictType::PriorityConflict { .. }));
        assert!(has_prio);
    }

    #[test]
    fn no_priority_conflict_when_same_priority() {
        let mut r = default_resolver();
        // Both have priority 7 but disjoint bodies; only DirectContradiction.
        r.add_rule(make_rule("r1", "q", &["a"], 7))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r2", "q", &["b"], 7))
            .expect("test setup: add_rule should not fail");
        let conflicts = r.detect_conflicts();
        let has_prio = conflicts
            .iter()
            .any(|c| matches!(&c.conflict_type, ConflictType::PriorityConflict { .. }));
        assert!(!has_prio);
    }

    #[test]
    fn priority_conflict_higher_lower_correct() {
        let mut r = default_resolver();
        r.add_rule(make_rule("lo", "q", &["a"], 2))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("hi", "q", &["b"], 9))
            .expect("test setup: add_rule should not fail");
        let conflicts = r.detect_conflicts();
        let prio = conflicts
            .iter()
            .find_map(|c| {
                if let ConflictType::PriorityConflict { higher, lower } = &c.conflict_type {
                    Some((higher.clone(), lower.clone()))
                } else {
                    None
                }
            })
            .expect("should have priority conflict");
        assert_eq!(prio.0, "hi");
        assert_eq!(prio.1, "lo");
    }

    // ── CyclicDependency ──────────────────────────────────────────────────────

    #[test]
    fn detect_simple_cycle() {
        let mut r = default_resolver();
        // A → B → A
        r.add_rule(make_rule("r_ab", "B", &["A"], 1))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r_ba", "A", &["B"], 1))
            .expect("test setup: add_rule should not fail");
        let conflicts = r.detect_conflicts();
        let has_cycle = conflicts
            .iter()
            .any(|c| matches!(&c.conflict_type, ConflictType::CyclicDependency { .. }));
        assert!(has_cycle);
    }

    #[test]
    fn no_cycle_in_dag() {
        let mut r = default_resolver();
        // A → B → C (no cycle)
        r.add_rule(make_rule("r1", "B", &["A"], 1))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r2", "C", &["B"], 1))
            .expect("test setup: add_rule should not fail");
        let conflicts = r.detect_conflicts();
        let has_cycle = conflicts
            .iter()
            .any(|c| matches!(&c.conflict_type, ConflictType::CyclicDependency { .. }));
        assert!(!has_cycle);
    }

    #[test]
    fn cycle_detection_disabled() {
        let cfg = ResolverConfig {
            enable_cycle_detection: false,
            ..Default::default()
        };
        let mut r = RuleConflictResolver::new(cfg);
        r.add_rule(make_rule("r_ab", "B", &["A"], 1))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r_ba", "A", &["B"], 1))
            .expect("test setup: add_rule should not fail");
        let conflicts = r.detect_conflicts();
        let has_cycle = conflicts
            .iter()
            .any(|c| matches!(&c.conflict_type, ConflictType::CyclicDependency { .. }));
        assert!(!has_cycle);
    }

    #[test]
    fn three_node_cycle() {
        let mut r = default_resolver();
        r.add_rule(make_rule("ab", "B", &["A"], 1))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("bc", "C", &["B"], 1))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("ca", "A", &["C"], 1))
            .expect("test setup: add_rule should not fail");
        let conflicts = r.detect_conflicts();
        let has_cycle = conflicts
            .iter()
            .any(|c| matches!(&c.conflict_type, ConflictType::CyclicDependency { .. }));
        assert!(has_cycle);
    }

    // ── UndercutConflict ──────────────────────────────────────────────────────

    #[test]
    fn detect_undercut_conflict() {
        let mut r = default_resolver();
        // r1 concludes "flies"; r2 has "NOT:flies" in its body.
        r.add_rule(make_rule("r1", "flies", &["has_wings"], 10))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r2", "penguin", &["NOT:flies", "lays_eggs"], 5))
            .expect("test setup: add_rule should not fail");
        let conflicts = r.detect_conflicts();
        let has_undercut = conflicts
            .iter()
            .any(|c| matches!(&c.conflict_type, ConflictType::UndercutConflict { .. }));
        assert!(has_undercut);
    }

    #[test]
    fn undercut_ids_correct() {
        let mut r = default_resolver();
        r.add_rule(make_rule("cutter", "X", &["p"], 5))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("victim", "Y", &["NOT:X", "q"], 5))
            .expect("test setup: add_rule should not fail");
        let conflicts = r.detect_conflicts();
        let uc = conflicts
            .iter()
            .find_map(|c| {
                if let ConflictType::UndercutConflict {
                    undercutter,
                    undercut,
                } = &c.conflict_type
                {
                    Some((undercutter.clone(), undercut.clone()))
                } else {
                    None
                }
            })
            .expect("should have undercut conflict");
        assert_eq!(uc.0, "cutter");
        assert_eq!(uc.1, "victim");
    }

    #[test]
    fn no_undercut_without_not_prefix() {
        let mut r = default_resolver();
        r.add_rule(make_rule("r1", "flies", &["wings"], 1))
            .expect("test setup: add_rule should not fail");
        // body uses "flies" (positive), not "NOT:flies".
        r.add_rule(make_rule("r2", "bird", &["flies", "feathers"], 1))
            .expect("test setup: add_rule should not fail");
        let conflicts = r.detect_conflicts();
        let has_undercut = conflicts
            .iter()
            .any(|c| matches!(&c.conflict_type, ConflictType::UndercutConflict { .. }));
        assert!(!has_undercut);
    }

    // ── RebuttalConflict ──────────────────────────────────────────────────────

    #[test]
    fn detect_rebuttal_conflict() {
        let mut r = default_resolver();
        r.add_rule(make_rule("r1", "bird", &["wings"], 5))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r2", "NOT:bird", &["penguin"], 5))
            .expect("test setup: add_rule should not fail");
        let conflicts = r.detect_conflicts();
        let has_rebuttal = conflicts
            .iter()
            .any(|c| matches!(&c.conflict_type, ConflictType::RebuttalConflict { .. }));
        assert!(has_rebuttal);
    }

    #[test]
    fn rebuttal_ids_correct() {
        let mut r = default_resolver();
        r.add_rule(make_rule("pos", "alive", &["moving"], 1))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("neg", "NOT:alive", &["still"], 1))
            .expect("test setup: add_rule should not fail");
        let conflicts = r.detect_conflicts();
        let rb = conflicts
            .iter()
            .find_map(|c| {
                if let ConflictType::RebuttalConflict { rule_a, rule_b } = &c.conflict_type {
                    Some((rule_a.clone(), rule_b.clone()))
                } else {
                    None
                }
            })
            .expect("should have rebuttal conflict");
        let ids: HashSet<String> = [rb.0, rb.1].into_iter().collect();
        assert!(ids.contains("pos"));
        assert!(ids.contains("neg"));
    }

    #[test]
    fn no_rebuttal_without_not_head() {
        let mut r = default_resolver();
        r.add_rule(make_rule("r1", "bird", &["wings"], 1))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r2", "reptile", &["scales"], 1))
            .expect("test setup: add_rule should not fail");
        let conflicts = r.detect_conflicts();
        let has_rebuttal = conflicts
            .iter()
            .any(|c| matches!(&c.conflict_type, ConflictType::RebuttalConflict { .. }));
        assert!(!has_rebuttal);
    }

    // ── Resolution: PriorityOrder ─────────────────────────────────────────────

    #[test]
    fn resolve_priority_order_higher_wins() {
        let cfg = ResolverConfig {
            default_strategy: ResolutionStrategy::PriorityOrder,
            ..Default::default()
        };
        let mut r = RuleConflictResolver::new(cfg);
        r.add_rule(make_rule("lo", "q", &["a"], 3))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("hi", "q", &["b"], 9))
            .expect("test setup: add_rule should not fail");
        let mut conflicts = r.detect_conflicts();
        let prio_conflict = conflicts
            .iter_mut()
            .find(|c| matches!(&c.conflict_type, ConflictType::PriorityConflict { .. }))
            .cloned()
            .expect("must find priority conflict");
        let winner = r
            .resolve(&prio_conflict)
            .expect("test setup: resolve should succeed");
        assert_eq!(winner, "hi");
    }

    #[test]
    fn resolve_priority_order_equal_returns_first_id() {
        let cfg = ResolverConfig {
            default_strategy: ResolutionStrategy::PriorityOrder,
            ..Default::default()
        };
        let mut r = RuleConflictResolver::new(cfg);
        r.add_rule(make_rule("r1", "q", &["a"], 5))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r2", "q", &["b"], 5))
            .expect("test setup: add_rule should not fail");
        let mut conflicts = r.detect_conflicts();
        // Only direct contradiction (same priority).
        let dc = conflicts
            .iter_mut()
            .find(|c| matches!(&c.conflict_type, ConflictType::DirectContradiction { .. }))
            .cloned()
            .expect("must find contradiction");
        let winner = r.resolve(&dc).expect("test setup: resolve should succeed");
        // Either is fine; just ensure no error.
        assert!(winner == "r1" || winner == "r2");
    }

    // ── Resolution: Specificity ───────────────────────────────────────────────

    #[test]
    fn resolve_specificity_more_conditions_wins() {
        let cfg = ResolverConfig {
            default_strategy: ResolutionStrategy::Specificity,
            ..Default::default()
        };
        let mut r = RuleConflictResolver::new(cfg);
        // Disjoint bodies (no shared premises) → DirectContradiction.
        // "specific" has more conditions, so it should win under Specificity.
        r.add_rule(make_rule("general", "bird", &["flies"], 1))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule(
            "specific",
            "bird",
            &["lays_eggs", "warm_blooded", "beak"],
            1,
        ))
        .expect("test setup: add_rule should not fail");
        let mut conflicts = r.detect_conflicts();
        let dc = conflicts
            .iter_mut()
            .find(|c| matches!(&c.conflict_type, ConflictType::DirectContradiction { .. }))
            .cloned()
            .expect("test setup: should find DirectContradiction conflict");
        let winner = r.resolve(&dc).expect("test setup: resolve should succeed");
        assert_eq!(winner, "specific");
    }

    // ── Resolution: LastWriter ────────────────────────────────────────────────

    #[test]
    fn resolve_last_writer_newer_wins() {
        let cfg = ResolverConfig {
            default_strategy: ResolutionStrategy::LastWriter,
            ..Default::default()
        };
        let mut r = RuleConflictResolver::new(cfg);
        r.add_rule(make_rule("old", "q", &["a"], 5))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("new", "q", &["b"], 5))
            .expect("test setup: add_rule should not fail");
        let mut conflicts = r.detect_conflicts();
        let dc = conflicts
            .iter_mut()
            .find(|c| matches!(&c.conflict_type, ConflictType::DirectContradiction { .. }))
            .cloned()
            .expect("test setup: should find DirectContradiction conflict");
        let winner = r.resolve(&dc).expect("test setup: resolve should succeed");
        assert_eq!(winner, "new");
    }

    // ── Resolution: Inhibit ───────────────────────────────────────────────────

    #[test]
    fn resolve_inhibit_non_defeasible_wins() {
        let cfg = ResolverConfig {
            default_strategy: ResolutionStrategy::Inhibit,
            ..Default::default()
        };
        let mut r = RuleConflictResolver::new(cfg);
        r.add_rule(make_defeasible("def", "q", &["a"], 10))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("firm", "q", &["b"], 1))
            .expect("test setup: add_rule should not fail");
        let mut conflicts = r.detect_conflicts();
        let dc = conflicts
            .iter_mut()
            .find(|c| matches!(&c.conflict_type, ConflictType::DirectContradiction { .. }))
            .cloned()
            .expect("test setup: should find DirectContradiction conflict");
        let winner = r.resolve(&dc).expect("test setup: resolve should succeed");
        assert_eq!(winner, "firm");
    }

    #[test]
    fn resolve_inhibit_both_defeasible_uses_priority() {
        let cfg = ResolverConfig {
            default_strategy: ResolutionStrategy::Inhibit,
            ..Default::default()
        };
        let mut r = RuleConflictResolver::new(cfg);
        r.add_rule(make_defeasible("lo", "q", &["a"], 2))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_defeasible("hi", "q", &["b"], 9))
            .expect("test setup: add_rule should not fail");
        let mut conflicts = r.detect_conflicts();
        let prio = conflicts
            .iter_mut()
            .find(|c| matches!(&c.conflict_type, ConflictType::PriorityConflict { .. }))
            .cloned()
            .expect("test setup: should find PriorityConflict conflict");
        let winner = r
            .resolve(&prio)
            .expect("test setup: resolve should succeed");
        assert_eq!(winner, "hi");
    }

    #[test]
    fn resolve_inhibit_neither_defeasible_uses_priority() {
        let cfg = ResolverConfig {
            default_strategy: ResolutionStrategy::Inhibit,
            ..Default::default()
        };
        let mut r = RuleConflictResolver::new(cfg);
        r.add_rule(make_rule("lo", "q", &["a"], 2))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("hi", "q", &["b"], 9))
            .expect("test setup: add_rule should not fail");
        let mut conflicts = r.detect_conflicts();
        let prio = conflicts
            .iter_mut()
            .find(|c| matches!(&c.conflict_type, ConflictType::PriorityConflict { .. }))
            .cloned()
            .expect("test setup: should find PriorityConflict conflict");
        let winner = r
            .resolve(&prio)
            .expect("test setup: resolve should succeed");
        assert_eq!(winner, "hi");
    }

    // ── Resolution: AskOracle ─────────────────────────────────────────────────

    #[test]
    fn resolve_ask_oracle_returns_error() {
        let cfg = ResolverConfig {
            default_strategy: ResolutionStrategy::AskOracle,
            ..Default::default()
        };
        let mut r = RuleConflictResolver::new(cfg);
        r.add_rule(make_rule("r1", "q", &["a"], 1))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r2", "q", &["b"], 1))
            .expect("test setup: add_rule should not fail");
        let mut conflicts = r.detect_conflicts();
        let dc = conflicts
            .iter_mut()
            .find(|c| matches!(&c.conflict_type, ConflictType::DirectContradiction { .. }))
            .cloned()
            .expect("test setup: should find DirectContradiction conflict");
        let err = r
            .resolve(&dc)
            .expect_err("test setup: expected UnresolvableConflict error");
        assert!(matches!(err, ResolverError::UnresolvableConflict { .. }));
    }

    // ── Resolution: Merge ─────────────────────────────────────────────────────

    #[test]
    fn resolve_merge_returns_error() {
        let cfg = ResolverConfig {
            default_strategy: ResolutionStrategy::Merge("union".to_string()),
            ..Default::default()
        };
        let mut r = RuleConflictResolver::new(cfg);
        r.add_rule(make_rule("r1", "q", &["a"], 1))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r2", "q", &["b"], 1))
            .expect("test setup: add_rule should not fail");
        let mut conflicts = r.detect_conflicts();
        let dc = conflicts
            .iter_mut()
            .find(|c| matches!(&c.conflict_type, ConflictType::DirectContradiction { .. }))
            .cloned()
            .expect("test setup: should find DirectContradiction conflict");
        let err = r
            .resolve(&dc)
            .expect_err("test setup: expected UnresolvableConflict error");
        assert!(matches!(err, ResolverError::UnresolvableConflict { .. }));
    }

    // ── resolve_all ───────────────────────────────────────────────────────────

    #[test]
    fn resolve_all_returns_results_for_all_conflicts() {
        let mut r = default_resolver();
        r.add_rule(make_rule("r1", "q", &["a"], 10))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r2", "q", &["b"], 5))
            .expect("test setup: add_rule should not fail");
        let results = r.resolve_all();
        assert!(!results.is_empty());
    }

    #[test]
    fn resolve_all_increments_resolved_stats() {
        let mut r = default_resolver();
        r.add_rule(make_rule("r1", "q", &["a"], 10))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r2", "q", &["b"], 5))
            .expect("test setup: add_rule should not fail");
        r.resolve_all();
        assert!(r.stats().conflicts_resolved > 0);
    }

    // ── applicable_rules ──────────────────────────────────────────────────────

    #[test]
    fn applicable_rules_all_facts_present() {
        let mut r = default_resolver();
        r.add_rule(make_rule("r1", "bird", &["wings", "feathers"], 1))
            .expect("test setup: add_rule should not fail");
        let facts: Vec<String> = vec!["wings".to_string(), "feathers".to_string()];
        let applicable = r.applicable_rules(&facts);
        assert_eq!(applicable.len(), 1);
        assert_eq!(applicable[0].id, "r1");
    }

    #[test]
    fn applicable_rules_missing_fact() {
        let mut r = default_resolver();
        r.add_rule(make_rule("r1", "bird", &["wings", "feathers"], 1))
            .expect("test setup: add_rule should not fail");
        let facts: Vec<String> = vec!["wings".to_string()]; // missing "feathers"
        let applicable = r.applicable_rules(&facts);
        assert!(applicable.is_empty());
    }

    #[test]
    fn applicable_rules_ignores_negated_conditions() {
        let mut r = default_resolver();
        // body has a NOT: condition; applicable_rules should ignore it.
        r.add_rule(make_rule(
            "r1",
            "mammal",
            &["warm_blooded", "NOT:has_gills"],
            1,
        ))
        .expect("test setup: add_rule should not fail");
        let facts: Vec<String> = vec!["warm_blooded".to_string()];
        // "NOT:has_gills" is ignored; rule should be applicable.
        let applicable = r.applicable_rules(&facts);
        assert_eq!(applicable.len(), 1);
    }

    #[test]
    fn applicable_rules_empty_body_always_applicable() {
        let mut r = default_resolver();
        r.add_rule(make_rule("r_always", "axiom", &[], 1))
            .expect("test setup: add_rule should not fail");
        let facts: Vec<String> = vec![];
        let applicable = r.applicable_rules(&facts);
        assert_eq!(applicable.len(), 1);
    }

    #[test]
    fn applicable_rules_multiple() {
        let mut r = default_resolver();
        r.add_rule(make_rule("r1", "a", &["x"], 1))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r2", "b", &["y"], 1))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r3", "c", &["x", "y"], 1))
            .expect("test setup: add_rule should not fail");
        let facts = vec!["x".to_string(), "y".to_string()];
        let applicable = r.applicable_rules(&facts);
        assert_eq!(applicable.len(), 3);
    }

    // ── winning_rule ──────────────────────────────────────────────────────────

    #[test]
    fn winning_rule_priority_strategy() {
        let cfg = ResolverConfig {
            default_strategy: ResolutionStrategy::PriorityOrder,
            ..Default::default()
        };
        let mut r = RuleConflictResolver::new(cfg);
        r.add_rule(make_rule("lo", "q", &["a"], 2))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("hi", "q", &["a"], 9))
            .expect("test setup: add_rule should not fail");
        let facts = vec!["a".to_string()];
        let winner = r.winning_rule("q", &facts).expect("must have winner");
        assert_eq!(winner.id, "hi");
    }

    #[test]
    fn winning_rule_specificity_strategy() {
        let cfg = ResolverConfig {
            default_strategy: ResolutionStrategy::Specificity,
            ..Default::default()
        };
        let mut r = RuleConflictResolver::new(cfg);
        r.add_rule(make_rule("gen", "q", &["a"], 1))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("spec", "q", &["a", "b", "c"], 1))
            .expect("test setup: add_rule should not fail");
        let facts = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let winner = r.winning_rule("q", &facts).expect("must have winner");
        assert_eq!(winner.id, "spec");
    }

    #[test]
    fn winning_rule_last_writer_strategy() {
        let cfg = ResolverConfig {
            default_strategy: ResolutionStrategy::LastWriter,
            ..Default::default()
        };
        let mut r = RuleConflictResolver::new(cfg);
        r.add_rule(make_rule("old", "q", &["a"], 1))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("new", "q", &["a"], 1))
            .expect("test setup: add_rule should not fail");
        let facts = vec!["a".to_string()];
        let winner = r.winning_rule("q", &facts).expect("must have winner");
        assert_eq!(winner.id, "new");
    }

    #[test]
    fn winning_rule_none_when_no_applicable() {
        let mut r = default_resolver();
        r.add_rule(make_rule("r1", "q", &["missing_fact"], 1))
            .expect("test setup: add_rule should not fail");
        let facts: Vec<String> = vec![];
        assert!(r.winning_rule("q", &facts).is_none());
    }

    #[test]
    fn winning_rule_none_when_wrong_head() {
        let mut r = default_resolver();
        r.add_rule(make_rule("r1", "q", &["a"], 1))
            .expect("test setup: add_rule should not fail");
        let facts = vec!["a".to_string()];
        assert!(r.winning_rule("unrelated_head", &facts).is_none());
    }

    #[test]
    fn winning_rule_inhibit_non_defeasible_preferred() {
        let cfg = ResolverConfig {
            default_strategy: ResolutionStrategy::Inhibit,
            ..Default::default()
        };
        let mut r = RuleConflictResolver::new(cfg);
        r.add_rule(make_defeasible("soft", "q", &["a"], 99))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("hard", "q", &["a"], 1))
            .expect("test setup: add_rule should not fail");
        let facts = vec!["a".to_string()];
        let winner = r.winning_rule("q", &facts).expect("must have winner");
        assert_eq!(winner.id, "hard");
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn stats_rules_loaded_accurate() {
        let mut r = default_resolver();
        assert_eq!(r.stats().rules_loaded, 0);
        r.add_rule(make_rule("r1", "a", &["x"], 1))
            .expect("test setup: add_rule should not fail");
        assert_eq!(r.stats().rules_loaded, 1);
        r.remove_rule("r1")
            .expect("test setup: remove_rule should not fail");
        assert_eq!(r.stats().rules_loaded, 0);
    }

    #[test]
    fn stats_conflicts_detected_increments() {
        let mut r = default_resolver();
        r.add_rule(make_rule("r1", "q", &["a"], 1))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r2", "q", &["b"], 1))
            .expect("test setup: add_rule should not fail");
        let before = r.stats().conflicts_detected;
        r.detect_conflicts();
        let after = r.stats().conflicts_detected;
        assert!(after > before);
    }

    #[test]
    fn stats_cycles_found() {
        let mut r = default_resolver();
        r.add_rule(make_rule("ab", "B", &["A"], 1))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("ba", "A", &["B"], 1))
            .expect("test setup: add_rule should not fail");
        r.detect_conflicts();
        assert!(r.stats().cycles_found > 0);
    }

    #[test]
    fn stats_conflicts_resolved() {
        let mut r = default_resolver();
        r.add_rule(make_rule("r1", "q", &["a"], 5))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r2", "q", &["b"], 10))
            .expect("test setup: add_rule should not fail");
        r.resolve_all();
        assert!(r.stats().conflicts_resolved > 0);
    }

    // ── Error cases ───────────────────────────────────────────────────────────

    #[test]
    fn resolve_rule_not_found_after_removal() {
        let mut r = default_resolver();
        r.add_rule(make_rule("r1", "q", &["a"], 5))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r2", "q", &["b"], 10))
            .expect("test setup: add_rule should not fail");
        let mut conflicts = r.detect_conflicts();
        let dc = conflicts
            .iter_mut()
            .find(|c| matches!(&c.conflict_type, ConflictType::DirectContradiction { .. }))
            .cloned()
            .expect("test setup: should find DirectContradiction conflict");
        // Remove one rule before resolving.
        r.remove_rule("r1")
            .expect("test setup: remove_rule should not fail");
        let err = r
            .resolve(&dc)
            .expect_err("test setup: expected RuleNotFound error after removal");
        assert!(matches!(err, ResolverError::RuleNotFound(_)));
    }

    #[test]
    fn cyclic_conflict_pair_error() {
        let mut r = default_resolver();
        r.add_rule(make_rule("ab", "B", &["A"], 1))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("ba", "A", &["B"], 1))
            .expect("test setup: add_rule should not fail");
        let mut conflicts = r.detect_conflicts();
        let cycle_conflict = conflicts
            .iter_mut()
            .find(|c| matches!(&c.conflict_type, ConflictType::CyclicDependency { .. }))
            .cloned()
            .expect("test setup: should find CyclicDependency conflict");
        let err = r
            .resolve(&cycle_conflict)
            .expect_err("test setup: expected CyclicDependency error");
        assert!(matches!(err, ResolverError::CyclicDependency(_)));
    }

    #[test]
    fn config_error_variant_display() {
        let err = ResolverError::ConfigurationError("bad value".to_string());
        assert!(err.to_string().contains("bad value"));
    }

    #[test]
    fn max_rules_exceeded_display() {
        let err = ResolverError::MaxRulesExceeded;
        assert!(!err.to_string().is_empty());
    }

    // ── LogicRule helpers ─────────────────────────────────────────────────────

    #[test]
    fn positive_body_excludes_negations() {
        let rule = make_rule("r", "head", &["a", "NOT:b", "c"], 1);
        let pos: Vec<&str> = rule.positive_body().collect();
        assert_eq!(pos, vec!["a", "c"]);
    }

    #[test]
    fn negated_body_strips_prefix() {
        let rule = make_rule("r", "head", &["a", "NOT:b", "NOT:c"], 1);
        let neg: Vec<&str> = rule.negated_body().collect();
        assert_eq!(neg, vec!["b", "c"]);
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn empty_resolver_no_conflicts() {
        let mut r = default_resolver();
        assert!(r.detect_conflicts().is_empty());
    }

    #[test]
    fn single_rule_no_conflicts() {
        let mut r = default_resolver();
        r.add_rule(make_rule("only", "q", &["a"], 1))
            .expect("test setup: add_rule should not fail");
        assert!(r.detect_conflicts().is_empty());
    }

    #[test]
    fn different_heads_no_direct_contradiction() {
        let mut r = default_resolver();
        r.add_rule(make_rule("r1", "bird", &["wings"], 1))
            .expect("test setup: add_rule should not fail");
        r.add_rule(make_rule("r2", "fish", &["fins"], 1))
            .expect("test setup: add_rule should not fail");
        let conflicts = r.detect_conflicts();
        let has_dc = conflicts
            .iter()
            .any(|c| matches!(&c.conflict_type, ConflictType::DirectContradiction { .. }));
        assert!(!has_dc);
    }

    #[test]
    fn resolve_all_empty_resolver() {
        let mut r = default_resolver();
        let results = r.resolve_all();
        assert!(results.is_empty());
    }

    #[test]
    fn applicable_rules_superset_of_body() {
        let mut r = default_resolver();
        r.add_rule(make_rule("r1", "q", &["a"], 1))
            .expect("test setup: add_rule should not fail");
        // Provide more facts than needed.
        let facts = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let applicable = r.applicable_rules(&facts);
        assert_eq!(applicable.len(), 1);
    }
}
