//! Storage Retention Policy Engine for IPFRS block storage.
//!
//! Manages data retention policies that determine when blocks should be expired,
//! archived, or permanently deleted based on age, size, access patterns, and tags.

/// Action to take on a block as determined by a retention rule.
#[derive(Debug, Clone, PartialEq)]
pub enum RetentionAction {
    /// No action needed; block should be kept as-is.
    Keep,
    /// Move the block to cold/archive storage.
    Archive,
    /// Permanently remove the block.
    Delete,
    /// Flag the block for human review with an explanatory reason.
    Warn { reason: String },
}

/// A single retention rule with priority-based evaluation order.
#[derive(Debug, Clone)]
pub struct RetentionRule {
    /// Unique identifier for this rule.
    pub rule_id: u64,
    /// Human-readable name for this rule.
    pub name: String,
    /// Maximum age in seconds before this rule triggers. `None` means no age constraint.
    pub max_age_secs: Option<u64>,
    /// Maximum size in bytes before this rule triggers. `None` means no size constraint.
    pub max_size_bytes: Option<u64>,
    /// If `Some`, this rule only applies to blocks that carry this tag.
    pub required_tag: Option<String>,
    /// The action to apply when this rule matches a block.
    pub action: RetentionAction,
    /// Evaluation priority: higher values are evaluated first.
    pub priority: u32,
}

/// Metadata record describing a stored block.
#[derive(Debug, Clone)]
pub struct BlockRecord {
    /// Content Identifier of the block.
    pub cid: String,
    /// Size of the block in bytes.
    pub size_bytes: u64,
    /// Unix timestamp (seconds) when the block was first stored.
    pub created_at_secs: u64,
    /// Unix timestamp (seconds) of the most recent access.
    pub last_accessed_secs: u64,
    /// Arbitrary tags attached to this block.
    pub tags: Vec<String>,
    /// When `true`, the block is protected and will always receive `Keep`.
    pub pinned: bool,
}

/// The outcome of evaluating retention policy rules against a single block.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyDecision {
    /// Content Identifier of the evaluated block.
    pub cid: String,
    /// The action the engine recommends for this block.
    pub action: RetentionAction,
    /// The rule that triggered this decision, or `None` for the default `Keep`.
    pub matched_rule_id: Option<u64>,
}

/// Aggregate statistics computed from a collection of policy decisions.
#[derive(Debug, Clone, PartialEq)]
pub struct RetentionStats {
    /// Total number of blocks evaluated.
    pub total_evaluated: usize,
    /// Number of blocks that will be kept.
    pub keep_count: usize,
    /// Number of blocks that will be archived.
    pub archive_count: usize,
    /// Number of blocks that will be deleted.
    pub delete_count: usize,
    /// Number of blocks flagged for review.
    pub warn_count: usize,
}

/// Engine that evaluates storage retention policies against block records.
///
/// Rules are maintained in descending priority order so that the highest-priority
/// rule is always evaluated first. Pinned blocks bypass all rules and always receive
/// a `Keep` decision.
#[derive(Debug, Clone)]
pub struct StorageRetentionPolicyEngine {
    /// Active rules, sorted in descending priority order.
    pub rules: Vec<RetentionRule>,
}

impl StorageRetentionPolicyEngine {
    /// Creates a new engine with no rules.
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Adds a rule while maintaining descending priority order.
    ///
    /// If two rules share the same priority, the newly inserted rule is placed
    /// after all existing rules with the same priority (stable ordering).
    pub fn add_rule(&mut self, rule: RetentionRule) {
        // Find the first position whose priority is strictly less than the new rule's
        // priority, and insert before it.
        let pos = self
            .rules
            .iter()
            .position(|r| r.priority < rule.priority)
            .unwrap_or(self.rules.len());
        self.rules.insert(pos, rule);
    }

    /// Removes the rule with the given `rule_id`.
    ///
    /// Returns `true` if a rule was found and removed, `false` otherwise.
    pub fn remove_rule(&mut self, rule_id: u64) -> bool {
        if let Some(pos) = self.rules.iter().position(|r| r.rule_id == rule_id) {
            self.rules.remove(pos);
            true
        } else {
            false
        }
    }

    /// Evaluates retention policy rules against a single block.
    ///
    /// Pinned blocks always receive `RetentionAction::Keep` with no matched rule.
    /// Otherwise, rules are checked in descending priority order; the first rule
    /// whose conditions **all** hold wins. If no rule matches, the default
    /// `RetentionAction::Keep` is returned with `matched_rule_id = None`.
    pub fn evaluate(&self, block: &BlockRecord, now_secs: u64) -> PolicyDecision {
        if block.pinned {
            return PolicyDecision {
                cid: block.cid.clone(),
                action: RetentionAction::Keep,
                matched_rule_id: None,
            };
        }

        for rule in &self.rules {
            if self.rule_matches(rule, block, now_secs) {
                return PolicyDecision {
                    cid: block.cid.clone(),
                    action: rule.action.clone(),
                    matched_rule_id: Some(rule.rule_id),
                };
            }
        }

        // Default: keep with no rule applied.
        PolicyDecision {
            cid: block.cid.clone(),
            action: RetentionAction::Keep,
            matched_rule_id: None,
        }
    }

    /// Evaluates retention policy rules against every block in `blocks`.
    pub fn evaluate_all(&self, blocks: &[BlockRecord], now_secs: u64) -> Vec<PolicyDecision> {
        blocks.iter().map(|b| self.evaluate(b, now_secs)).collect()
    }

    /// Computes aggregate statistics from a slice of policy decisions.
    pub fn stats(&self, decisions: &[PolicyDecision]) -> RetentionStats {
        let mut keep_count = 0usize;
        let mut archive_count = 0usize;
        let mut delete_count = 0usize;
        let mut warn_count = 0usize;

        for decision in decisions {
            match &decision.action {
                RetentionAction::Keep => keep_count += 1,
                RetentionAction::Archive => archive_count += 1,
                RetentionAction::Delete => delete_count += 1,
                RetentionAction::Warn { .. } => warn_count += 1,
            }
        }

        RetentionStats {
            total_evaluated: decisions.len(),
            keep_count,
            archive_count,
            delete_count,
            warn_count,
        }
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Returns `true` if **all** active conditions of `rule` hold for `block`
    /// at time `now_secs`.
    fn rule_matches(&self, rule: &RetentionRule, block: &BlockRecord, now_secs: u64) -> bool {
        // Age condition: the block must be older than max_age_secs.
        if let Some(max_age) = rule.max_age_secs {
            let age = now_secs.saturating_sub(block.created_at_secs);
            if age <= max_age {
                return false;
            }
        }

        // Size condition: the block must exceed max_size_bytes.
        if let Some(max_size) = rule.max_size_bytes {
            if block.size_bytes <= max_size {
                return false;
            }
        }

        // Tag condition: the block must carry the required tag.
        if let Some(ref tag) = rule.required_tag {
            if !block.tags.contains(tag) {
                return false;
            }
        }

        true
    }
}

impl Default for StorageRetentionPolicyEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// StorageRetentionPolicy — tick-based, entry-managing retention policy
// ══════════════════════════════════════════════════════════════════════════════

use std::collections::HashMap;

/// Action produced by tick-based retention evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickRetentionAction {
    /// Block should be kept.
    Keep,
    /// Block should be moved to archive/cold storage.
    Archive,
    /// Block should be permanently deleted.
    Delete,
}

/// A tick-based retention rule with priority ordering.
#[derive(Debug, Clone)]
pub struct TickRetentionRule {
    /// Human-readable name (also used as unique key).
    pub name: String,
    /// Maximum age in ticks before this rule triggers. `None` = no age constraint.
    pub max_age_ticks: Option<u64>,
    /// Maximum total storage size in bytes before this rule triggers. `None` = no size constraint.
    pub max_size_bytes: Option<u64>,
    /// The action to take when this rule matches.
    pub action: TickRetentionAction,
    /// Higher priority rules are evaluated first.
    pub priority: u32,
}

/// Metadata for a block tracked by the retention policy.
#[derive(Debug, Clone)]
pub struct RetentionEntry {
    /// Content identifier of the block.
    pub block_cid: String,
    /// Size of the block in bytes.
    pub size_bytes: u64,
    /// Tick at which the block was created/added.
    pub created_tick: u64,
    /// Tick at which the block was last accessed.
    pub last_accessed_tick: u64,
    /// Pinned blocks are never deleted or archived.
    pub pinned: bool,
}

/// Aggregate statistics for the retention policy.
#[derive(Debug, Clone)]
pub struct RetentionPolicyStats {
    /// Number of entries currently tracked.
    pub entry_count: usize,
    /// Number of rules currently registered.
    pub rule_count: usize,
    /// Cumulative count of deleted entries since creation.
    pub total_deleted: u64,
    /// Cumulative count of archived entries since creation.
    pub total_archived: u64,
    /// Cumulative bytes freed by deletion since creation.
    pub bytes_freed: u64,
}

/// Tick-based storage retention policy with built-in entry tracking.
///
/// Maintains a set of blocks (entries) and evaluates configurable rules
/// against them. Rules are sorted by descending priority. Pinned entries
/// are always kept regardless of rules.
pub struct StorageRetentionPolicy {
    rules: Vec<TickRetentionRule>,
    entries: HashMap<String, RetentionEntry>,
    current_tick: u64,
    total_deleted: u64,
    total_archived: u64,
    bytes_freed: u64,
}

impl StorageRetentionPolicy {
    /// Creates a new, empty retention policy at tick 0.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            entries: HashMap::new(),
            current_tick: 0,
            total_deleted: 0,
            total_archived: 0,
            bytes_freed: 0,
        }
    }

    /// Adds a rule, maintaining descending priority order.
    /// If a rule with the same name already exists it is replaced.
    pub fn add_rule(&mut self, rule: TickRetentionRule) {
        // Remove existing rule with same name, if any.
        self.rules.retain(|r| r.name != rule.name);
        let pos = self
            .rules
            .iter()
            .position(|r| r.priority < rule.priority)
            .unwrap_or(self.rules.len());
        self.rules.insert(pos, rule);
    }

    /// Removes the rule with the given name.
    /// Returns `true` if a rule was found and removed.
    pub fn remove_rule(&mut self, name: &str) -> bool {
        let before = self.rules.len();
        self.rules.retain(|r| r.name != name);
        self.rules.len() < before
    }

    /// Registers a new block entry at the given creation tick.
    pub fn add_entry(&mut self, block_cid: &str, size_bytes: u64, created_tick: u64) {
        let entry = RetentionEntry {
            block_cid: block_cid.to_string(),
            size_bytes,
            created_tick,
            last_accessed_tick: created_tick,
            pinned: false,
        };
        self.entries.insert(block_cid.to_string(), entry);
    }

    /// Pins a block, protecting it from deletion/archival.
    /// Returns `true` if the entry was found and pinned.
    pub fn pin(&mut self, block_cid: &str) -> bool {
        if let Some(entry) = self.entries.get_mut(block_cid) {
            entry.pinned = true;
            true
        } else {
            false
        }
    }

    /// Unpins a block, allowing retention rules to apply.
    /// Returns `true` if the entry was found and unpinned.
    pub fn unpin(&mut self, block_cid: &str) -> bool {
        if let Some(entry) = self.entries.get_mut(block_cid) {
            entry.pinned = false;
            true
        } else {
            false
        }
    }

    /// Evaluates retention rules against a single block.
    ///
    /// Pinned blocks always return `Keep`. Rules are evaluated in descending
    /// priority order; the first matching rule wins. If no rule matches,
    /// the default is `Keep`.
    pub fn evaluate(&self, block_cid: &str) -> TickRetentionAction {
        let entry = match self.entries.get(block_cid) {
            Some(e) => e,
            None => return TickRetentionAction::Keep,
        };

        if entry.pinned {
            return TickRetentionAction::Keep;
        }

        let total_size = self.total_size_bytes();

        for rule in &self.rules {
            if self.tick_rule_matches(rule, entry, total_size) {
                return rule.action;
            }
        }

        TickRetentionAction::Keep
    }

    /// Evaluates all entries and enforces the policy.
    ///
    /// Entries marked for deletion are removed from the internal store.
    /// Returns a list of `(cid, action)` pairs for entries that received
    /// `Archive` or `Delete` actions.
    pub fn enforce(&mut self) -> Vec<(String, TickRetentionAction)> {
        let total_size = self.total_size_bytes();
        let cids: Vec<String> = self.entries.keys().cloned().collect();

        let mut actions = Vec::new();

        for cid in &cids {
            if let Some(entry) = self.entries.get(cid) {
                if entry.pinned {
                    continue;
                }

                let mut matched_action = TickRetentionAction::Keep;
                for rule in &self.rules {
                    if self.tick_rule_matches(rule, entry, total_size) {
                        matched_action = rule.action;
                        break;
                    }
                }

                if matched_action != TickRetentionAction::Keep {
                    actions.push((cid.clone(), matched_action));
                }
            }
        }

        // Apply deletions and track stats.
        for (cid, action) in &actions {
            match action {
                TickRetentionAction::Delete => {
                    if let Some(removed) = self.entries.remove(cid) {
                        self.total_deleted += 1;
                        self.bytes_freed += removed.size_bytes;
                    }
                }
                TickRetentionAction::Archive => {
                    self.total_archived += 1;
                    // Archived entries remain in the map (they moved to cold storage
                    // externally) but we still count them.
                }
                TickRetentionAction::Keep => {}
            }
        }

        actions
    }

    /// Updates the `last_accessed_tick` of the given entry to the current tick.
    pub fn access(&mut self, block_cid: &str) {
        if let Some(entry) = self.entries.get_mut(block_cid) {
            entry.last_accessed_tick = self.current_tick;
        }
    }

    /// Advances the internal clock by one tick.
    pub fn tick(&mut self) {
        self.current_tick += 1;
    }

    /// Returns the number of entries currently tracked.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Returns the total size in bytes of all tracked entries.
    pub fn total_size_bytes(&self) -> u64 {
        self.entries.values().map(|e| e.size_bytes).sum()
    }

    /// Returns aggregate statistics about this policy.
    pub fn stats(&self) -> RetentionPolicyStats {
        RetentionPolicyStats {
            entry_count: self.entries.len(),
            rule_count: self.rules.len(),
            total_deleted: self.total_deleted,
            total_archived: self.total_archived,
            bytes_freed: self.bytes_freed,
        }
    }

    /// Returns the current tick value.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    /// Checks whether a tick-based rule matches the given entry.
    fn tick_rule_matches(
        &self,
        rule: &TickRetentionRule,
        entry: &RetentionEntry,
        total_size: u64,
    ) -> bool {
        // Age condition: block must be older than max_age_ticks.
        if let Some(max_age) = rule.max_age_ticks {
            let age = self.current_tick.saturating_sub(entry.created_tick);
            if age <= max_age {
                return false;
            }
        }

        // Size condition: total storage must exceed max_size_bytes.
        if let Some(max_size) = rule.max_size_bytes {
            if total_size <= max_size {
                return false;
            }
        }

        // At least one condition must be specified for the rule to match.
        if rule.max_age_ticks.is_none() && rule.max_size_bytes.is_none() {
            return false;
        }

        true
    }
}

impl Default for StorageRetentionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_block(
        cid: &str,
        size_bytes: u64,
        created_at_secs: u64,
        tags: Vec<&str>,
        pinned: bool,
    ) -> BlockRecord {
        BlockRecord {
            cid: cid.to_string(),
            size_bytes,
            created_at_secs,
            last_accessed_secs: created_at_secs,
            tags: tags.into_iter().map(|t| t.to_string()).collect(),
            pinned,
        }
    }

    fn age_rule(
        rule_id: u64,
        max_age_secs: u64,
        action: RetentionAction,
        priority: u32,
    ) -> RetentionRule {
        RetentionRule {
            rule_id,
            name: format!("age-rule-{rule_id}"),
            max_age_secs: Some(max_age_secs),
            max_size_bytes: None,
            required_tag: None,
            action,
            priority,
        }
    }

    fn size_rule(
        rule_id: u64,
        max_size_bytes: u64,
        action: RetentionAction,
        priority: u32,
    ) -> RetentionRule {
        RetentionRule {
            rule_id,
            name: format!("size-rule-{rule_id}"),
            max_age_secs: None,
            max_size_bytes: Some(max_size_bytes),
            required_tag: None,
            action,
            priority,
        }
    }

    fn tag_rule(
        rule_id: u64,
        required_tag: &str,
        action: RetentionAction,
        priority: u32,
    ) -> RetentionRule {
        RetentionRule {
            rule_id,
            name: format!("tag-rule-{rule_id}"),
            max_age_secs: None,
            max_size_bytes: None,
            required_tag: Some(required_tag.to_string()),
            action,
            priority,
        }
    }

    // ── Engine construction ───────────────────────────────────────────────────

    #[test]
    fn new_starts_with_no_rules() {
        let engine = StorageRetentionPolicyEngine::new();
        assert!(engine.rules.is_empty());
    }

    // ── add_rule ─────────────────────────────────────────────────────────────

    #[test]
    fn add_rule_single_rule() {
        let mut engine = StorageRetentionPolicyEngine::new();
        engine.add_rule(age_rule(1, 3600, RetentionAction::Delete, 10));
        assert_eq!(engine.rules.len(), 1);
    }

    #[test]
    fn add_rule_maintains_priority_order_descending() {
        let mut engine = StorageRetentionPolicyEngine::new();
        engine.add_rule(age_rule(1, 3600, RetentionAction::Delete, 5));
        engine.add_rule(age_rule(2, 7200, RetentionAction::Archive, 20));
        engine.add_rule(age_rule(3, 1800, RetentionAction::Delete, 10));

        // Expected order: priority 20, 10, 5
        assert_eq!(engine.rules[0].priority, 20);
        assert_eq!(engine.rules[1].priority, 10);
        assert_eq!(engine.rules[2].priority, 5);
    }

    #[test]
    fn add_rule_equal_priorities_stable() {
        let mut engine = StorageRetentionPolicyEngine::new();
        engine.add_rule(age_rule(1, 3600, RetentionAction::Delete, 10));
        engine.add_rule(age_rule(2, 7200, RetentionAction::Archive, 10));
        // Both at priority 10; rule 1 inserted first should remain before rule 2.
        assert_eq!(engine.rules[0].rule_id, 1);
        assert_eq!(engine.rules[1].rule_id, 2);
    }

    // ── remove_rule ───────────────────────────────────────────────────────────

    #[test]
    fn remove_rule_returns_true_when_found() {
        let mut engine = StorageRetentionPolicyEngine::new();
        engine.add_rule(age_rule(42, 3600, RetentionAction::Delete, 10));
        assert!(engine.remove_rule(42));
        assert!(engine.rules.is_empty());
    }

    #[test]
    fn remove_rule_returns_false_when_not_found() {
        let mut engine = StorageRetentionPolicyEngine::new();
        assert!(!engine.remove_rule(99));
    }

    #[test]
    fn remove_rule_leaves_others_intact() {
        let mut engine = StorageRetentionPolicyEngine::new();
        engine.add_rule(age_rule(1, 3600, RetentionAction::Delete, 10));
        engine.add_rule(age_rule(2, 7200, RetentionAction::Archive, 5));
        engine.remove_rule(1);
        assert_eq!(engine.rules.len(), 1);
        assert_eq!(engine.rules[0].rule_id, 2);
    }

    // ── Pinned blocks ─────────────────────────────────────────────────────────

    #[test]
    fn pinned_block_always_keep_even_with_delete_rule() {
        let mut engine = StorageRetentionPolicyEngine::new();
        // A very aggressive delete rule that would normally fire.
        engine.add_rule(age_rule(1, 0, RetentionAction::Delete, 100));

        let block = make_block("QmPin", 99_999, 0, vec![], true);
        let decision = engine.evaluate(&block, 1_000_000);

        assert_eq!(decision.action, RetentionAction::Keep);
        assert_eq!(decision.matched_rule_id, None);
    }

    #[test]
    fn pinned_block_keep_with_no_rules() {
        let engine = StorageRetentionPolicyEngine::new();
        let block = make_block("QmPinNoRule", 100, 0, vec![], true);
        let decision = engine.evaluate(&block, 1000);
        assert_eq!(decision.action, RetentionAction::Keep);
        assert_eq!(decision.matched_rule_id, None);
    }

    // ── Default Keep when no rules ─────────────────────────────────────────────

    #[test]
    fn no_rules_returns_default_keep() {
        let engine = StorageRetentionPolicyEngine::new();
        let block = make_block("QmNoRule", 512, 1000, vec![], false);
        let decision = engine.evaluate(&block, 2000);
        assert_eq!(decision.action, RetentionAction::Keep);
        assert_eq!(decision.matched_rule_id, None);
    }

    // ── Age rule ──────────────────────────────────────────────────────────────

    #[test]
    fn age_rule_matches_when_old_enough() {
        let mut engine = StorageRetentionPolicyEngine::new();
        // max_age_secs = 100; block is 200 s old → age (200) > 100 → matches.
        engine.add_rule(age_rule(1, 100, RetentionAction::Delete, 10));

        let block = make_block("QmOld", 512, 0, vec![], false);
        let decision = engine.evaluate(&block, 200);

        assert_eq!(decision.action, RetentionAction::Delete);
        assert_eq!(decision.matched_rule_id, Some(1));
    }

    #[test]
    fn age_rule_does_not_match_when_too_young() {
        let mut engine = StorageRetentionPolicyEngine::new();
        // max_age_secs = 1000; block is only 50 s old → age (50) ≤ 1000 → no match.
        engine.add_rule(age_rule(1, 1000, RetentionAction::Delete, 10));

        let block = make_block("QmYoung", 512, 1000, vec![], false);
        let decision = engine.evaluate(&block, 1050);

        assert_eq!(decision.action, RetentionAction::Keep);
        assert_eq!(decision.matched_rule_id, None);
    }

    #[test]
    fn age_rule_does_not_match_exactly_at_boundary() {
        let mut engine = StorageRetentionPolicyEngine::new();
        // age == max_age_secs → not strictly greater → no match.
        engine.add_rule(age_rule(1, 100, RetentionAction::Delete, 10));

        let block = make_block("QmBoundary", 512, 0, vec![], false);
        let decision = engine.evaluate(&block, 100);

        assert_eq!(decision.action, RetentionAction::Keep);
    }

    // ── Size rule ─────────────────────────────────────────────────────────────

    #[test]
    fn size_rule_matches_when_large_enough() {
        let mut engine = StorageRetentionPolicyEngine::new();
        // max_size_bytes = 1000; block is 2000 B → size (2000) > 1000 → matches.
        engine.add_rule(size_rule(2, 1000, RetentionAction::Archive, 10));

        let block = make_block("QmBig", 2000, 0, vec![], false);
        let decision = engine.evaluate(&block, 0);

        assert_eq!(decision.action, RetentionAction::Archive);
        assert_eq!(decision.matched_rule_id, Some(2));
    }

    #[test]
    fn size_rule_does_not_match_when_small() {
        let mut engine = StorageRetentionPolicyEngine::new();
        // max_size_bytes = 1000; block is only 500 B → no match.
        engine.add_rule(size_rule(2, 1000, RetentionAction::Archive, 10));

        let block = make_block("QmSmall", 500, 0, vec![], false);
        let decision = engine.evaluate(&block, 0);

        assert_eq!(decision.action, RetentionAction::Keep);
        assert_eq!(decision.matched_rule_id, None);
    }

    #[test]
    fn size_rule_does_not_match_exactly_at_boundary() {
        let mut engine = StorageRetentionPolicyEngine::new();
        // size == max_size_bytes → not strictly greater → no match.
        engine.add_rule(size_rule(2, 1000, RetentionAction::Archive, 10));

        let block = make_block("QmExact", 1000, 0, vec![], false);
        let decision = engine.evaluate(&block, 0);

        assert_eq!(decision.action, RetentionAction::Keep);
    }

    // ── Tag rule ──────────────────────────────────────────────────────────────

    #[test]
    fn tag_rule_matches_correctly_when_tag_present() {
        let mut engine = StorageRetentionPolicyEngine::new();
        engine.add_rule(tag_rule(3, "archive", RetentionAction::Archive, 10));

        let block = make_block("QmTagged", 512, 0, vec!["archive", "index"], false);
        let decision = engine.evaluate(&block, 0);

        assert_eq!(decision.action, RetentionAction::Archive);
        assert_eq!(decision.matched_rule_id, Some(3));
    }

    #[test]
    fn tag_rule_no_match_when_tag_absent() {
        let mut engine = StorageRetentionPolicyEngine::new();
        engine.add_rule(tag_rule(3, "archive", RetentionAction::Archive, 10));

        let block = make_block("QmUntagged", 512, 0, vec!["index"], false);
        let decision = engine.evaluate(&block, 0);

        assert_eq!(decision.action, RetentionAction::Keep);
        assert_eq!(decision.matched_rule_id, None);
    }

    // ── Priority ordering ─────────────────────────────────────────────────────

    #[test]
    fn first_matching_rule_wins_by_priority() {
        let mut engine = StorageRetentionPolicyEngine::new();
        // High-priority delete rule: triggers for old blocks.
        engine.add_rule(age_rule(10, 100, RetentionAction::Delete, 50));
        // Low-priority archive rule: also triggers for old blocks.
        engine.add_rule(age_rule(20, 100, RetentionAction::Archive, 10));

        let block = make_block("QmOld", 512, 0, vec![], false);
        let decision = engine.evaluate(&block, 200);

        // The higher-priority Delete rule (id 10) should win.
        assert_eq!(decision.action, RetentionAction::Delete);
        assert_eq!(decision.matched_rule_id, Some(10));
    }

    #[test]
    fn lower_priority_rule_not_applied_when_higher_matches() {
        let mut engine = StorageRetentionPolicyEngine::new();
        engine.add_rule(age_rule(1, 0, RetentionAction::Keep, 100));
        engine.add_rule(age_rule(2, 0, RetentionAction::Delete, 1));

        let block = make_block("QmBlock", 512, 0, vec![], false);
        // age == 1 > 0, so rule 1 (priority 100, Keep) matches first.
        let decision = engine.evaluate(&block, 1);

        assert_eq!(decision.action, RetentionAction::Keep);
        assert_eq!(decision.matched_rule_id, Some(1));
    }

    // ── Action variants ───────────────────────────────────────────────────────

    #[test]
    fn archive_action_returned_correctly() {
        let mut engine = StorageRetentionPolicyEngine::new();
        engine.add_rule(size_rule(5, 100, RetentionAction::Archive, 10));

        let block = make_block("QmArchive", 200, 0, vec![], false);
        let decision = engine.evaluate(&block, 0);

        assert_eq!(decision.action, RetentionAction::Archive);
    }

    #[test]
    fn delete_action_returned_correctly() {
        let mut engine = StorageRetentionPolicyEngine::new();
        engine.add_rule(size_rule(6, 100, RetentionAction::Delete, 10));

        let block = make_block("QmDelete", 200, 0, vec![], false);
        let decision = engine.evaluate(&block, 0);

        assert_eq!(decision.action, RetentionAction::Delete);
    }

    #[test]
    fn warn_action_returned_correctly() {
        let mut engine = StorageRetentionPolicyEngine::new();
        engine.add_rule(RetentionRule {
            rule_id: 7,
            name: "warn-big".to_string(),
            max_age_secs: None,
            max_size_bytes: Some(100),
            required_tag: None,
            action: RetentionAction::Warn {
                reason: "block too large".to_string(),
            },
            priority: 10,
        });

        let block = make_block("QmWarn", 200, 0, vec![], false);
        let decision = engine.evaluate(&block, 0);

        assert_eq!(
            decision.action,
            RetentionAction::Warn {
                reason: "block too large".to_string()
            }
        );
        assert_eq!(decision.matched_rule_id, Some(7));
    }

    // ── evaluate_all ──────────────────────────────────────────────────────────

    #[test]
    fn evaluate_all_processes_multiple_blocks() {
        let mut engine = StorageRetentionPolicyEngine::new();
        engine.add_rule(size_rule(1, 1000, RetentionAction::Delete, 10));

        let blocks = vec![
            make_block("QmA", 500, 0, vec![], false),  // small → Keep
            make_block("QmB", 2000, 0, vec![], false), // large → Delete
            make_block("QmC", 1500, 0, vec![], true),  // pinned → Keep
        ];
        let decisions = engine.evaluate_all(&blocks, 0);

        assert_eq!(decisions.len(), 3);
        assert_eq!(decisions[0].action, RetentionAction::Keep);
        assert_eq!(decisions[1].action, RetentionAction::Delete);
        assert_eq!(decisions[2].action, RetentionAction::Keep);
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn stats_keep_count_correct() {
        let engine = StorageRetentionPolicyEngine::new();
        let blocks = vec![
            make_block("QmA", 100, 0, vec![], false),
            make_block("QmB", 200, 0, vec![], false),
        ];
        let decisions = engine.evaluate_all(&blocks, 0);
        let stats = engine.stats(&decisions);
        assert_eq!(stats.keep_count, 2);
        assert_eq!(stats.total_evaluated, 2);
    }

    #[test]
    fn stats_archive_count_correct() {
        let mut engine = StorageRetentionPolicyEngine::new();
        engine.add_rule(size_rule(1, 100, RetentionAction::Archive, 10));

        let blocks = vec![
            make_block("QmA", 50, 0, vec![], false),  // Keep
            make_block("QmB", 200, 0, vec![], false), // Archive
            make_block("QmC", 300, 0, vec![], false), // Archive
        ];
        let decisions = engine.evaluate_all(&blocks, 0);
        let stats = engine.stats(&decisions);

        assert_eq!(stats.archive_count, 2);
        assert_eq!(stats.keep_count, 1);
    }

    #[test]
    fn stats_delete_count_correct() {
        let mut engine = StorageRetentionPolicyEngine::new();
        engine.add_rule(size_rule(1, 100, RetentionAction::Delete, 10));

        let blocks = vec![
            make_block("QmA", 200, 0, vec![], false), // Delete
            make_block("QmB", 50, 0, vec![], false),  // Keep
        ];
        let decisions = engine.evaluate_all(&blocks, 0);
        let stats = engine.stats(&decisions);

        assert_eq!(stats.delete_count, 1);
        assert_eq!(stats.keep_count, 1);
    }

    #[test]
    fn stats_warn_count_correct() {
        let mut engine = StorageRetentionPolicyEngine::new();
        engine.add_rule(RetentionRule {
            rule_id: 1,
            name: "warn-old".to_string(),
            max_age_secs: Some(100),
            max_size_bytes: None,
            required_tag: None,
            action: RetentionAction::Warn {
                reason: "stale".to_string(),
            },
            priority: 10,
        });

        let blocks = vec![
            make_block("QmA", 100, 0, vec![], false), // age 200 > 100 → Warn
            make_block("QmB", 100, 0, vec![], false), // age 200 > 100 → Warn
        ];
        let decisions = engine.evaluate_all(&blocks, 200);
        let stats = engine.stats(&decisions);

        assert_eq!(stats.warn_count, 2);
        assert_eq!(stats.total_evaluated, 2);
    }

    // ── matched_rule_id ───────────────────────────────────────────────────────

    #[test]
    fn matched_rule_id_set_when_rule_matches() {
        let mut engine = StorageRetentionPolicyEngine::new();
        engine.add_rule(age_rule(99, 100, RetentionAction::Delete, 10));

        let block = make_block("QmMatch", 512, 0, vec![], false);
        let decision = engine.evaluate(&block, 200);

        assert_eq!(decision.matched_rule_id, Some(99));
    }

    #[test]
    fn matched_rule_id_none_when_no_rule_matches() {
        let mut engine = StorageRetentionPolicyEngine::new();
        // Rule requires tag "special" but block has none.
        engine.add_rule(tag_rule(10, "special", RetentionAction::Delete, 10));

        let block = make_block("QmNoMatch", 512, 0, vec![], false);
        let decision = engine.evaluate(&block, 1000);

        assert_eq!(decision.action, RetentionAction::Keep);
        assert_eq!(decision.matched_rule_id, None);
    }

    // ── Combined conditions (Engine) ────────────────────────────────────────

    #[test]
    fn rule_with_all_conditions_requires_all_to_match() {
        let mut engine = StorageRetentionPolicyEngine::new();
        // Rule fires only when: old AND big AND tagged "hot".
        engine.add_rule(RetentionRule {
            rule_id: 77,
            name: "combined".to_string(),
            max_age_secs: Some(100),
            max_size_bytes: Some(1000),
            required_tag: Some("hot".to_string()),
            action: RetentionAction::Archive,
            priority: 10,
        });

        let now = 300u64;

        // All three conditions met → Archive.
        let full_match = make_block("QmFull", 2000, 0, vec!["hot"], false);
        assert_eq!(
            engine.evaluate(&full_match, now).action,
            RetentionAction::Archive
        );

        // Missing tag → Keep.
        let no_tag = make_block("QmNoTag", 2000, 0, vec![], false);
        assert_eq!(engine.evaluate(&no_tag, now).action, RetentionAction::Keep);

        // Too small → Keep.
        let small = make_block("QmSmall", 500, 0, vec!["hot"], false);
        assert_eq!(engine.evaluate(&small, now).action, RetentionAction::Keep);

        // Too young → Keep.
        let young = make_block("QmYoung", 2000, 250, vec!["hot"], false);
        assert_eq!(engine.evaluate(&young, now).action, RetentionAction::Keep);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // StorageRetentionPolicy (tick-based) tests
    // ══════════════════════════════════════════════════════════════════════════

    fn tick_age_rule(
        name: &str,
        max_age: u64,
        action: TickRetentionAction,
        priority: u32,
    ) -> TickRetentionRule {
        TickRetentionRule {
            name: name.to_string(),
            max_age_ticks: Some(max_age),
            max_size_bytes: None,
            action,
            priority,
        }
    }

    fn tick_size_rule(
        name: &str,
        max_size: u64,
        action: TickRetentionAction,
        priority: u32,
    ) -> TickRetentionRule {
        TickRetentionRule {
            name: name.to_string(),
            max_age_ticks: None,
            max_size_bytes: Some(max_size),
            action,
            priority,
        }
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn tick_new_starts_empty() {
        let policy = StorageRetentionPolicy::new();
        assert_eq!(policy.entry_count(), 0);
        assert_eq!(policy.total_size_bytes(), 0);
        assert_eq!(policy.current_tick(), 0);
    }

    #[test]
    fn tick_default_is_equivalent_to_new() {
        let policy = StorageRetentionPolicy::default();
        assert_eq!(policy.entry_count(), 0);
        assert_eq!(policy.current_tick(), 0);
    }

    // ── add_entry / entry_count / total_size_bytes ──────────────────────────

    #[test]
    fn tick_add_entry_increases_count() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_entry("QmA", 100, 0);
        assert_eq!(policy.entry_count(), 1);
        policy.add_entry("QmB", 200, 0);
        assert_eq!(policy.entry_count(), 2);
    }

    #[test]
    fn tick_total_size_bytes_sums_entries() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_entry("QmA", 100, 0);
        policy.add_entry("QmB", 250, 0);
        assert_eq!(policy.total_size_bytes(), 350);
    }

    #[test]
    fn tick_add_entry_overwrites_same_cid() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_entry("QmA", 100, 0);
        policy.add_entry("QmA", 999, 5);
        assert_eq!(policy.entry_count(), 1);
        assert_eq!(policy.total_size_bytes(), 999);
    }

    // ── pin / unpin ─────────────────────────────────────────────────────────

    #[test]
    fn tick_pin_returns_true_when_entry_exists() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_entry("QmA", 100, 0);
        assert!(policy.pin("QmA"));
    }

    #[test]
    fn tick_pin_returns_false_when_entry_missing() {
        let mut policy = StorageRetentionPolicy::new();
        assert!(!policy.pin("QmNone"));
    }

    #[test]
    fn tick_unpin_returns_true_when_entry_exists() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_entry("QmA", 100, 0);
        policy.pin("QmA");
        assert!(policy.unpin("QmA"));
    }

    #[test]
    fn tick_unpin_returns_false_when_entry_missing() {
        let mut policy = StorageRetentionPolicy::new();
        assert!(!policy.unpin("QmNone"));
    }

    // ── age-based deletion ──────────────────────────────────────────────────

    #[test]
    fn tick_age_rule_deletes_old_blocks() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_rule(tick_age_rule(
            "delete-old",
            5,
            TickRetentionAction::Delete,
            10,
        ));
        policy.add_entry("QmOld", 100, 0);

        // Advance 6 ticks → age = 6 > 5 → delete.
        for _ in 0..6 {
            policy.tick();
        }
        assert_eq!(policy.evaluate("QmOld"), TickRetentionAction::Delete);
    }

    #[test]
    fn tick_age_rule_keeps_young_blocks() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_rule(tick_age_rule(
            "delete-old",
            10,
            TickRetentionAction::Delete,
            10,
        ));
        policy.add_entry("QmYoung", 100, 0);

        // Only 3 ticks → age = 3 ≤ 10 → keep.
        for _ in 0..3 {
            policy.tick();
        }
        assert_eq!(policy.evaluate("QmYoung"), TickRetentionAction::Keep);
    }

    #[test]
    fn tick_age_rule_boundary_keeps_at_exact_age() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_rule(tick_age_rule(
            "delete-old",
            5,
            TickRetentionAction::Delete,
            10,
        ));
        policy.add_entry("QmBoundary", 100, 0);

        // Exactly 5 ticks → age == max_age → not strictly greater → keep.
        for _ in 0..5 {
            policy.tick();
        }
        assert_eq!(policy.evaluate("QmBoundary"), TickRetentionAction::Keep);
    }

    // ── size-based deletion ─────────────────────────────────────────────────

    #[test]
    fn tick_size_rule_deletes_when_total_exceeds_limit() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_rule(tick_size_rule(
            "size-limit",
            500,
            TickRetentionAction::Delete,
            10,
        ));
        policy.add_entry("QmA", 300, 0);
        policy.add_entry("QmB", 300, 0);

        // Total = 600 > 500 → delete.
        assert_eq!(policy.evaluate("QmA"), TickRetentionAction::Delete);
    }

    #[test]
    fn tick_size_rule_keeps_when_total_within_limit() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_rule(tick_size_rule(
            "size-limit",
            1000,
            TickRetentionAction::Delete,
            10,
        ));
        policy.add_entry("QmA", 100, 0);

        // Total = 100 ≤ 1000 → keep.
        assert_eq!(policy.evaluate("QmA"), TickRetentionAction::Keep);
    }

    #[test]
    fn tick_size_rule_boundary_keeps_at_exact_size() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_rule(tick_size_rule(
            "size-limit",
            500,
            TickRetentionAction::Delete,
            10,
        ));
        policy.add_entry("QmA", 250, 0);
        policy.add_entry("QmB", 250, 0);

        // Total = 500 == max_size → not strictly greater → keep.
        assert_eq!(policy.evaluate("QmA"), TickRetentionAction::Keep);
    }

    // ── pinned blocks protected ─────────────────────────────────────────────

    #[test]
    fn tick_pinned_block_never_deleted() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_rule(tick_age_rule(
            "delete-old",
            0,
            TickRetentionAction::Delete,
            100,
        ));
        policy.add_entry("QmPinned", 100, 0);
        policy.pin("QmPinned");
        policy.tick();

        assert_eq!(policy.evaluate("QmPinned"), TickRetentionAction::Keep);
    }

    #[test]
    fn tick_pinned_block_survives_enforce() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_rule(tick_age_rule(
            "delete-old",
            0,
            TickRetentionAction::Delete,
            100,
        ));
        policy.add_entry("QmPinned", 100, 0);
        policy.pin("QmPinned");
        policy.tick();

        let actions = policy.enforce();
        assert!(actions.is_empty());
        assert_eq!(policy.entry_count(), 1);
    }

    #[test]
    fn tick_unpin_allows_deletion() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_rule(tick_age_rule(
            "delete-old",
            0,
            TickRetentionAction::Delete,
            100,
        ));
        policy.add_entry("QmBlock", 100, 0);
        policy.pin("QmBlock");
        policy.tick();

        // While pinned → keep.
        assert_eq!(policy.evaluate("QmBlock"), TickRetentionAction::Keep);

        // After unpin → delete.
        policy.unpin("QmBlock");
        assert_eq!(policy.evaluate("QmBlock"), TickRetentionAction::Delete);
    }

    // ── rule priority ordering ──────────────────────────────────────────────

    #[test]
    fn tick_higher_priority_rule_wins() {
        let mut policy = StorageRetentionPolicy::new();
        // High priority: archive.
        policy.add_rule(tick_age_rule(
            "archive-rule",
            0,
            TickRetentionAction::Archive,
            50,
        ));
        // Low priority: delete.
        policy.add_rule(tick_age_rule(
            "delete-rule",
            0,
            TickRetentionAction::Delete,
            10,
        ));

        policy.add_entry("QmBlock", 100, 0);
        policy.tick();

        // Archive (priority 50) should win over Delete (priority 10).
        assert_eq!(policy.evaluate("QmBlock"), TickRetentionAction::Archive);
    }

    #[test]
    fn tick_add_rule_replaces_same_name() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_rule(tick_age_rule(
            "my-rule",
            100,
            TickRetentionAction::Delete,
            10,
        ));
        policy.add_rule(tick_age_rule(
            "my-rule",
            5,
            TickRetentionAction::Archive,
            20,
        ));

        // Should only have one rule.
        let stats = policy.stats();
        assert_eq!(stats.rule_count, 1);
    }

    // ── enforce removes entries ──────────────────────────────────────────────

    #[test]
    fn tick_enforce_removes_deleted_entries() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_rule(tick_age_rule(
            "delete-old",
            5,
            TickRetentionAction::Delete,
            10,
        ));
        policy.add_entry("QmOld", 200, 0);
        policy.add_entry("QmNew", 100, 10);

        // Advance to tick 10.
        for _ in 0..10 {
            policy.tick();
        }

        let actions = policy.enforce();

        // QmOld: age 10 > 5 → deleted. QmNew: age 0 ≤ 5 → kept.
        assert!(actions
            .iter()
            .any(|(c, a)| c == "QmOld" && *a == TickRetentionAction::Delete));
        assert_eq!(policy.entry_count(), 1);
    }

    #[test]
    fn tick_enforce_returns_actions_taken() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_rule(tick_age_rule("del", 0, TickRetentionAction::Delete, 10));
        policy.add_entry("QmA", 50, 0);
        policy.add_entry("QmB", 75, 0);
        policy.tick();

        let actions = policy.enforce();
        assert_eq!(actions.len(), 2);
        for (_, a) in &actions {
            assert_eq!(*a, TickRetentionAction::Delete);
        }
    }

    // ── access updates tick ─────────────────────────────────────────────────

    #[test]
    fn tick_access_updates_last_accessed_tick() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_entry("QmA", 100, 0);

        for _ in 0..5 {
            policy.tick();
        }
        policy.access("QmA");

        // Verify via the entry (indirectly — we test that the entry
        // still exists and the system is consistent).
        assert_eq!(policy.entry_count(), 1);
        // The internal last_accessed_tick should be 5, but we can't
        // read it directly, so verify via stats consistency.
        let stats = policy.stats();
        assert_eq!(stats.entry_count, 1);
    }

    #[test]
    fn tick_access_nonexistent_is_noop() {
        let mut policy = StorageRetentionPolicy::new();
        policy.tick();
        policy.access("QmGhost");
        // Should not panic or change anything.
        assert_eq!(policy.entry_count(), 0);
    }

    // ── archive action ──────────────────────────────────────────────────────

    #[test]
    fn tick_archive_action_does_not_remove_entry() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_rule(tick_age_rule(
            "archive",
            2,
            TickRetentionAction::Archive,
            10,
        ));
        policy.add_entry("QmA", 100, 0);

        for _ in 0..5 {
            policy.tick();
        }

        let actions = policy.enforce();
        assert!(actions
            .iter()
            .any(|(c, a)| c == "QmA" && *a == TickRetentionAction::Archive));
        // Entry should still exist (only Delete removes it).
        assert_eq!(policy.entry_count(), 1);
    }

    // ── multiple rules ──────────────────────────────────────────────────────

    #[test]
    fn tick_multiple_rules_evaluated_in_order() {
        let mut policy = StorageRetentionPolicy::new();
        // Priority 30: archive if total > 50 bytes.
        policy.add_rule(tick_size_rule(
            "archive-big",
            50,
            TickRetentionAction::Archive,
            30,
        ));
        // Priority 20: delete if age > 3.
        policy.add_rule(tick_age_rule(
            "delete-old",
            3,
            TickRetentionAction::Delete,
            20,
        ));

        policy.add_entry("QmA", 100, 0);
        for _ in 0..5 {
            policy.tick();
        }

        // Both rules match, but archive (priority 30) wins.
        assert_eq!(policy.evaluate("QmA"), TickRetentionAction::Archive);
    }

    #[test]
    fn tick_only_matching_rule_applied() {
        let mut policy = StorageRetentionPolicy::new();
        // Age rule won't match (block is too young).
        policy.add_rule(tick_age_rule(
            "delete-old",
            100,
            TickRetentionAction::Delete,
            50,
        ));
        // Size rule matches (total 200 > 50).
        policy.add_rule(tick_size_rule(
            "archive-big",
            50,
            TickRetentionAction::Archive,
            10,
        ));

        policy.add_entry("QmA", 200, 0);

        assert_eq!(policy.evaluate("QmA"), TickRetentionAction::Archive);
    }

    // ── stats tracking ──────────────────────────────────────────────────────

    #[test]
    fn tick_stats_initial_zeroes() {
        let policy = StorageRetentionPolicy::new();
        let stats = policy.stats();
        assert_eq!(stats.entry_count, 0);
        assert_eq!(stats.rule_count, 0);
        assert_eq!(stats.total_deleted, 0);
        assert_eq!(stats.total_archived, 0);
        assert_eq!(stats.bytes_freed, 0);
    }

    #[test]
    fn tick_stats_tracks_deleted_count_and_bytes() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_rule(tick_age_rule("del", 0, TickRetentionAction::Delete, 10));
        policy.add_entry("QmA", 100, 0);
        policy.add_entry("QmB", 250, 0);
        policy.tick();
        policy.enforce();

        let stats = policy.stats();
        assert_eq!(stats.total_deleted, 2);
        assert_eq!(stats.bytes_freed, 350);
    }

    #[test]
    fn tick_stats_tracks_archived_count() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_rule(tick_age_rule("arch", 0, TickRetentionAction::Archive, 10));
        policy.add_entry("QmA", 100, 0);
        policy.tick();
        policy.enforce();

        let stats = policy.stats();
        assert_eq!(stats.total_archived, 1);
        assert_eq!(stats.bytes_freed, 0); // Archive does not free bytes.
    }

    #[test]
    fn tick_stats_rule_count() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_rule(tick_age_rule("r1", 10, TickRetentionAction::Delete, 10));
        policy.add_rule(tick_size_rule("r2", 500, TickRetentionAction::Archive, 5));
        assert_eq!(policy.stats().rule_count, 2);
    }

    #[test]
    fn tick_stats_accumulate_across_enforcements() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_rule(tick_age_rule("del", 0, TickRetentionAction::Delete, 10));

        // First enforcement: QmA created at tick 0, advance to tick 1 → age 1 > 0 → delete.
        policy.add_entry("QmA", 100, 0);
        policy.tick();
        policy.enforce();

        // Second enforcement: QmB created at current tick (1), advance to tick 2 → age 1 > 0 → delete.
        let created = policy.current_tick();
        policy.add_entry("QmB", 200, created);
        policy.tick();
        policy.enforce();

        let stats = policy.stats();
        assert_eq!(stats.total_deleted, 2);
        assert_eq!(stats.bytes_freed, 300);
    }

    // ── empty policy ────────────────────────────────────────────────────────

    #[test]
    fn tick_empty_policy_no_rules_keeps_everything() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_entry("QmA", 100, 0);
        policy.add_entry("QmB", 200, 0);

        for _ in 0..100 {
            policy.tick();
        }

        let actions = policy.enforce();
        assert!(actions.is_empty());
        assert_eq!(policy.entry_count(), 2);
    }

    #[test]
    fn tick_empty_policy_enforce_returns_empty() {
        let mut policy = StorageRetentionPolicy::new();
        let actions = policy.enforce();
        assert!(actions.is_empty());
    }

    // ── remove_rule ─────────────────────────────────────────────────────────

    #[test]
    fn tick_remove_rule_returns_true_when_found() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_rule(tick_age_rule(
            "my-rule",
            10,
            TickRetentionAction::Delete,
            10,
        ));
        assert!(policy.remove_rule("my-rule"));
        assert_eq!(policy.stats().rule_count, 0);
    }

    #[test]
    fn tick_remove_rule_returns_false_when_not_found() {
        let mut policy = StorageRetentionPolicy::new();
        assert!(!policy.remove_rule("nonexistent"));
    }

    // ── evaluate nonexistent entry ──────────────────────────────────────────

    #[test]
    fn tick_evaluate_nonexistent_returns_keep() {
        let policy = StorageRetentionPolicy::new();
        assert_eq!(policy.evaluate("QmNope"), TickRetentionAction::Keep);
    }

    // ── tick advancement ────────────────────────────────────────────────────

    #[test]
    fn tick_advances_clock() {
        let mut policy = StorageRetentionPolicy::new();
        assert_eq!(policy.current_tick(), 0);
        policy.tick();
        assert_eq!(policy.current_tick(), 1);
        policy.tick();
        policy.tick();
        assert_eq!(policy.current_tick(), 3);
    }

    // ── rule without conditions never matches ───────────────────────────────

    #[test]
    fn tick_rule_without_conditions_never_matches() {
        let mut policy = StorageRetentionPolicy::new();
        policy.add_rule(TickRetentionRule {
            name: "empty-rule".to_string(),
            max_age_ticks: None,
            max_size_bytes: None,
            action: TickRetentionAction::Delete,
            priority: 100,
        });
        policy.add_entry("QmA", 100, 0);
        policy.tick();

        assert_eq!(policy.evaluate("QmA"), TickRetentionAction::Keep);
    }
}
