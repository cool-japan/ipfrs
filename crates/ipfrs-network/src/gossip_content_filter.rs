//! Content-based gossip message filtering for bandwidth reduction.
//!
//! Provides [`ContentGossipFilter`] which evaluates incoming GossipSub messages
//! against a configurable rule chain.  Each [`FilterRule`] can match on topic
//! substring, message size bounds, and peer allowlists.  Rules are evaluated in
//! insertion order; the first rule that returns [`FilterAction::Accept`] or
//! [`FilterAction::Reject`] wins.  [`FilterAction::Defer`] causes the next rule
//! to be consulted.  If every rule defers (or no rules exist), the default
//! action is [`FilterAction::Accept`].
//!
//! # Example
//!
//! ```rust
//! use ipfrs_network::gossip_content_filter::{
//!     ContentGossipFilter, FilterAction, FilterRule, GossipFilterStats,
//! };
//!
//! let mut filter = ContentGossipFilter::new();
//!
//! // Reject any message larger than 1 MiB
//! filter.add_rule(FilterRule {
//!     name: "max-1mb".to_string(),
//!     topic_pattern: None,
//!     max_size_bytes: Some(1_048_576),
//!     min_size_bytes: None,
//!     peer_allowlist: None,
//! });
//!
//! assert_eq!(
//!     filter.evaluate("blocks", "peer-A", 2_000_000),
//!     FilterAction::Reject,
//! );
//! assert_eq!(
//!     filter.evaluate("blocks", "peer-A", 512),
//!     FilterAction::Accept,
//! );
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// Public data types
// ─────────────────────────────────────────────────────────────────────────────

/// Decision returned by a single rule or the overall filter chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterAction {
    /// Message should be processed normally.
    Accept,
    /// Message should be dropped — saves bandwidth.
    Reject,
    /// This rule does not have an opinion; let subsequent rules decide.
    Defer,
}

/// A single content-based filter rule evaluated against incoming gossip
/// messages.
///
/// A rule **rejects** if any of its active constraints are violated:
///
/// - `topic_pattern` — if set, the topic must contain this substring for the
///   rule to apply at all.  If the topic does *not* match, the rule defers.
/// - `max_size_bytes` — reject if `size_bytes > max`.
/// - `min_size_bytes` — reject if `size_bytes < min`.
/// - `peer_allowlist` — reject if the peer is *not* in the list.
///
/// If the rule applies (topic matches or no topic constraint) and none of the
/// constraints trigger a reject, the rule **accepts**.
#[derive(Debug, Clone)]
pub struct FilterRule {
    /// Human-readable name used for lookups and removal.
    pub name: String,
    /// Optional substring that must appear in the topic for this rule to apply.
    /// When `None`, the rule applies to all topics.
    pub topic_pattern: Option<String>,
    /// Reject messages strictly larger than this threshold.
    pub max_size_bytes: Option<u64>,
    /// Reject messages strictly smaller than this threshold.
    pub min_size_bytes: Option<u64>,
    /// If set, only peers whose ID appears in this list are accepted.
    pub peer_allowlist: Option<Vec<String>>,
}

impl FilterRule {
    /// Evaluate this rule against a single message descriptor.
    ///
    /// Returns [`FilterAction::Defer`] when the rule's topic pattern does not
    /// match the incoming topic (the rule simply does not apply).
    fn evaluate(&self, topic: &str, peer_id: &str, size_bytes: u64) -> FilterAction {
        // ── Topic gate ───────────────────────────────────────────────────────
        if let Some(ref pattern) = self.topic_pattern {
            if !topic.contains(pattern.as_str()) {
                return FilterAction::Defer;
            }
        }

        // ── Size bounds ──────────────────────────────────────────────────────
        if let Some(max) = self.max_size_bytes {
            if size_bytes > max {
                return FilterAction::Reject;
            }
        }
        if let Some(min) = self.min_size_bytes {
            if size_bytes < min {
                return FilterAction::Reject;
            }
        }

        // ── Peer allowlist ───────────────────────────────────────────────────
        if let Some(ref allowlist) = self.peer_allowlist {
            if !allowlist.iter().any(|p| p == peer_id) {
                return FilterAction::Reject;
            }
        }

        FilterAction::Accept
    }
}

/// Cumulative statistics for the content filter.
#[derive(Debug, Clone)]
pub struct GossipFilterStats {
    /// Number of rules currently registered.
    pub rule_count: usize,
    /// Messages that ultimately received [`FilterAction::Accept`].
    pub messages_accepted: u64,
    /// Messages that ultimately received [`FilterAction::Reject`].
    pub messages_rejected: u64,
    /// Messages where every rule deferred (counted as accepted but tracked
    /// separately).
    pub messages_deferred: u64,
    /// Total payload bytes saved by rejecting messages.
    pub bytes_saved: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// ContentGossipFilter
// ─────────────────────────────────────────────────────────────────────────────

/// Content-based gossip message filter.
///
/// Rules are evaluated in the order they were added.  The first rule returning
/// [`FilterAction::Accept`] or [`FilterAction::Reject`] determines the final
/// outcome.  If all rules defer, the message is accepted by default.
pub struct ContentGossipFilter {
    rules: Vec<FilterRule>,
    messages_accepted: u64,
    messages_rejected: u64,
    messages_deferred: u64,
    bytes_saved: u64,
}

impl ContentGossipFilter {
    /// Create a new filter with no rules.  Without rules every message is
    /// accepted.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            messages_accepted: 0,
            messages_rejected: 0,
            messages_deferred: 0,
            bytes_saved: 0,
        }
    }

    /// Append a rule to the end of the evaluation chain.
    pub fn add_rule(&mut self, rule: FilterRule) {
        self.rules.push(rule);
    }

    /// Remove the first rule whose `name` matches.  Returns `true` if a rule
    /// was removed.
    pub fn remove_rule(&mut self, name: &str) -> bool {
        if let Some(idx) = self.rules.iter().position(|r| r.name == name) {
            self.rules.remove(idx);
            true
        } else {
            false
        }
    }

    /// Evaluate a message against the rule chain.
    ///
    /// Returns the winning [`FilterAction`].  Side-effects: counters and
    /// `bytes_saved` are updated.
    pub fn evaluate(&mut self, topic: &str, peer_id: &str, size_bytes: u64) -> FilterAction {
        for rule in &self.rules {
            match rule.evaluate(topic, peer_id, size_bytes) {
                FilterAction::Accept => {
                    self.messages_accepted += 1;
                    return FilterAction::Accept;
                }
                FilterAction::Reject => {
                    self.messages_rejected += 1;
                    self.bytes_saved += size_bytes;
                    return FilterAction::Reject;
                }
                FilterAction::Defer => {
                    // continue to next rule
                }
            }
        }

        // Every rule deferred (or there are no rules).
        self.messages_deferred += 1;
        self.messages_accepted += 1;
        FilterAction::Accept
    }

    /// Number of rules currently registered.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Look up a rule by name.
    pub fn get_rule(&self, name: &str) -> Option<&FilterRule> {
        self.rules.iter().find(|r| r.name == name)
    }

    /// Remove all rules.
    pub fn clear_rules(&mut self) {
        self.rules.clear();
    }

    /// Return a snapshot of cumulative statistics.
    pub fn stats(&self) -> GossipFilterStats {
        GossipFilterStats {
            rule_count: self.rules.len(),
            messages_accepted: self.messages_accepted,
            messages_rejected: self.messages_rejected,
            messages_deferred: self.messages_deferred,
            bytes_saved: self.bytes_saved,
        }
    }
}

impl Default for ContentGossipFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn size_rule(name: &str, max: Option<u64>, min: Option<u64>) -> FilterRule {
        FilterRule {
            name: name.to_string(),
            topic_pattern: None,
            max_size_bytes: max,
            min_size_bytes: min,
            peer_allowlist: None,
        }
    }

    fn topic_rule(name: &str, pattern: &str) -> FilterRule {
        FilterRule {
            name: name.to_string(),
            topic_pattern: Some(pattern.to_string()),
            max_size_bytes: None,
            min_size_bytes: None,
            peer_allowlist: None,
        }
    }

    fn allowlist_rule(name: &str, peers: Vec<&str>) -> FilterRule {
        FilterRule {
            name: name.to_string(),
            topic_pattern: None,
            max_size_bytes: None,
            min_size_bytes: None,
            peer_allowlist: Some(peers.into_iter().map(|s| s.to_string()).collect()),
        }
    }

    // ── 1. Accept by default (no rules) ──────────────────────────────────────

    #[test]
    fn test_accept_by_default_no_rules() {
        let mut f = ContentGossipFilter::new();
        assert_eq!(
            f.evaluate("any-topic", "any-peer", 1024),
            FilterAction::Accept
        );
    }

    #[test]
    fn test_accept_by_default_stats() {
        let mut f = ContentGossipFilter::new();
        f.evaluate("t", "p", 100);
        let s = f.stats();
        assert_eq!(s.messages_accepted, 1);
        assert_eq!(s.messages_deferred, 1);
    }

    // ── 2. Reject by max size ────────────────────────────────────────────────

    #[test]
    fn test_reject_exceeds_max_size() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("max-1k", Some(1024), None));
        assert_eq!(f.evaluate("t", "p", 2048), FilterAction::Reject);
    }

    #[test]
    fn test_accept_within_max_size() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("max-1k", Some(1024), None));
        assert_eq!(f.evaluate("t", "p", 512), FilterAction::Accept);
    }

    #[test]
    fn test_accept_at_exact_max_size() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("max-1k", Some(1024), None));
        assert_eq!(f.evaluate("t", "p", 1024), FilterAction::Accept);
    }

    // ── 3. Reject by min size ────────────────────────────────────────────────

    #[test]
    fn test_reject_below_min_size() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("min-100", None, Some(100)));
        assert_eq!(f.evaluate("t", "p", 50), FilterAction::Reject);
    }

    #[test]
    fn test_accept_above_min_size() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("min-100", None, Some(100)));
        assert_eq!(f.evaluate("t", "p", 200), FilterAction::Accept);
    }

    #[test]
    fn test_accept_at_exact_min_size() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("min-100", None, Some(100)));
        assert_eq!(f.evaluate("t", "p", 100), FilterAction::Accept);
    }

    // ── 4. Topic pattern matching ────────────────────────────────────────────

    #[test]
    fn test_topic_pattern_match_accepts() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(topic_rule("blocks-only", "blocks"));
        assert_eq!(f.evaluate("new-blocks", "p", 100), FilterAction::Accept);
    }

    #[test]
    fn test_topic_pattern_no_match_defers() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(topic_rule("blocks-only", "blocks"));
        // Topic does not contain "blocks" — rule defers, default accept.
        assert_eq!(f.evaluate("transactions", "p", 100), FilterAction::Accept);
    }

    #[test]
    fn test_topic_pattern_substring() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(topic_rule("tx-rule", "tx"));
        assert_eq!(
            f.evaluate("pending-tx-pool", "p", 100),
            FilterAction::Accept
        );
    }

    #[test]
    fn test_topic_pattern_with_size_constraint() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(FilterRule {
            name: "block-max-size".to_string(),
            topic_pattern: Some("block".to_string()),
            max_size_bytes: Some(500),
            min_size_bytes: None,
            peer_allowlist: None,
        });
        // Topic matches, size exceeds → reject.
        assert_eq!(
            f.evaluate("block-announce", "p", 1000),
            FilterAction::Reject
        );
        // Topic matches, size within → accept.
        assert_eq!(f.evaluate("block-announce", "p", 400), FilterAction::Accept);
        // Topic does not match → defer → default accept.
        assert_eq!(f.evaluate("mempool", "p", 9999), FilterAction::Accept);
    }

    // ── 5. Peer allowlist ────────────────────────────────────────────────────

    #[test]
    fn test_peer_allowlist_accepts_listed_peer() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(allowlist_rule("trusted", vec!["peer-A", "peer-B"]));
        assert_eq!(f.evaluate("t", "peer-A", 100), FilterAction::Accept);
    }

    #[test]
    fn test_peer_allowlist_rejects_unlisted_peer() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(allowlist_rule("trusted", vec!["peer-A", "peer-B"]));
        assert_eq!(f.evaluate("t", "peer-C", 100), FilterAction::Reject);
    }

    #[test]
    fn test_peer_allowlist_empty_rejects_all() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(allowlist_rule("empty-list", vec![]));
        assert_eq!(f.evaluate("t", "anyone", 100), FilterAction::Reject);
    }

    // ── 6. Rule ordering — first non-Defer wins ─────────────────────────────

    #[test]
    fn test_first_accept_wins() {
        let mut f = ContentGossipFilter::new();
        // Rule 1: accept everything (no constraints).
        f.add_rule(size_rule("accept-all", None, None));
        // Rule 2: reject everything over 0 bytes.
        f.add_rule(size_rule("reject-all", Some(0), None));
        assert_eq!(f.evaluate("t", "p", 9999), FilterAction::Accept);
    }

    #[test]
    fn test_first_reject_wins() {
        let mut f = ContentGossipFilter::new();
        // Rule 1: reject everything over 0 bytes.
        f.add_rule(size_rule("reject-all", Some(0), None));
        // Rule 2: accept everything.
        f.add_rule(size_rule("accept-all", None, None));
        assert_eq!(f.evaluate("t", "p", 100), FilterAction::Reject);
    }

    #[test]
    fn test_defer_falls_through_to_next_rule() {
        let mut f = ContentGossipFilter::new();
        // Rule 1: topic "block" — defers for non-block topics.
        f.add_rule(FilterRule {
            name: "block-max".to_string(),
            topic_pattern: Some("block".to_string()),
            max_size_bytes: Some(500),
            min_size_bytes: None,
            peer_allowlist: None,
        });
        // Rule 2: no topic constraint — rejects large messages.
        f.add_rule(size_rule("global-max", Some(1000), None));

        // "tx" topic → rule 1 defers → rule 2 rejects (2000 > 1000).
        assert_eq!(f.evaluate("tx", "p", 2000), FilterAction::Reject);
    }

    // ── 7. Remove rule ───────────────────────────────────────────────────────

    #[test]
    fn test_remove_existing_rule() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("my-rule", Some(100), None));
        assert!(f.remove_rule("my-rule"));
        assert_eq!(f.rule_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_rule_returns_false() {
        let mut f = ContentGossipFilter::new();
        assert!(!f.remove_rule("no-such-rule"));
    }

    #[test]
    fn test_remove_restores_default_accept() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("strict", Some(0), None));
        assert_eq!(f.evaluate("t", "p", 100), FilterAction::Reject);
        f.remove_rule("strict");
        assert_eq!(f.evaluate("t", "p", 100), FilterAction::Accept);
    }

    // ── 8. Clear rules ───────────────────────────────────────────────────────

    #[test]
    fn test_clear_rules() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("r1", Some(100), None));
        f.add_rule(size_rule("r2", Some(200), None));
        f.clear_rules();
        assert_eq!(f.rule_count(), 0);
    }

    #[test]
    fn test_clear_then_accept_by_default() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("strict", Some(0), None));
        f.clear_rules();
        assert_eq!(f.evaluate("t", "p", 999), FilterAction::Accept);
    }

    // ── 9. Stats tracking ────────────────────────────────────────────────────

    #[test]
    fn test_stats_accepted_count() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("max-1k", Some(1024), None));
        f.evaluate("t", "p", 100);
        f.evaluate("t", "p", 200);
        assert_eq!(f.stats().messages_accepted, 2);
    }

    #[test]
    fn test_stats_rejected_count() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("max-1k", Some(1024), None));
        f.evaluate("t", "p", 2000);
        f.evaluate("t", "p", 3000);
        assert_eq!(f.stats().messages_rejected, 2);
    }

    #[test]
    fn test_stats_bytes_saved() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("max-1k", Some(1024), None));
        f.evaluate("t", "p", 2000);
        f.evaluate("t", "p", 3000);
        assert_eq!(f.stats().bytes_saved, 5000);
    }

    #[test]
    fn test_stats_deferred_count() {
        let mut f = ContentGossipFilter::new();
        // No rules → everything defers.
        f.evaluate("t", "p", 100);
        assert_eq!(f.stats().messages_deferred, 1);
        // Also counted as accepted.
        assert_eq!(f.stats().messages_accepted, 1);
    }

    #[test]
    fn test_stats_rule_count() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("a", None, None));
        f.add_rule(size_rule("b", None, None));
        assert_eq!(f.stats().rule_count, 2);
    }

    // ── 10. Multiple rules interaction ───────────────────────────────────────

    #[test]
    fn test_multiple_topic_rules() {
        let mut f = ContentGossipFilter::new();
        // Rule for "block" topic: max 1 KiB.
        f.add_rule(FilterRule {
            name: "block-limit".to_string(),
            topic_pattern: Some("block".to_string()),
            max_size_bytes: Some(1024),
            min_size_bytes: None,
            peer_allowlist: None,
        });
        // Rule for "tx" topic: max 256 bytes.
        f.add_rule(FilterRule {
            name: "tx-limit".to_string(),
            topic_pattern: Some("tx".to_string()),
            max_size_bytes: Some(256),
            min_size_bytes: None,
            peer_allowlist: None,
        });

        // Block topic, within limit.
        assert_eq!(f.evaluate("block", "p", 500), FilterAction::Accept);
        // Block topic, exceeds limit.
        assert_eq!(f.evaluate("block", "p", 2000), FilterAction::Reject);
        // Tx topic, within limit.
        assert_eq!(f.evaluate("tx", "p", 100), FilterAction::Accept);
        // Tx topic, exceeds limit.
        assert_eq!(f.evaluate("tx", "p", 300), FilterAction::Reject);
    }

    #[test]
    fn test_allowlist_plus_size_rule() {
        let mut f = ContentGossipFilter::new();
        // Only trusted peers.
        f.add_rule(allowlist_rule("trusted-only", vec!["peer-A"]));
        // Plus size limit.
        f.add_rule(size_rule("max-1k", Some(1024), None));

        // Trusted peer, small message → first rule accepts.
        assert_eq!(f.evaluate("t", "peer-A", 100), FilterAction::Accept);
        // Untrusted peer → first rule rejects immediately.
        assert_eq!(f.evaluate("t", "peer-B", 100), FilterAction::Reject);
    }

    // ── 11. Defer semantics ──────────────────────────────────────────────────

    #[test]
    fn test_all_rules_defer_results_in_accept() {
        let mut f = ContentGossipFilter::new();
        // Two rules with topic patterns that won't match.
        f.add_rule(topic_rule("blocks", "block"));
        f.add_rule(topic_rule("txs", "tx"));
        // Topic "state" matches neither → both defer → accept.
        assert_eq!(f.evaluate("state", "p", 100), FilterAction::Accept);
    }

    #[test]
    fn test_defer_does_not_increment_reject() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(topic_rule("blocks", "block"));
        f.evaluate("state", "p", 100);
        assert_eq!(f.stats().messages_rejected, 0);
    }

    #[test]
    fn test_defer_then_accept_by_second_rule() {
        let mut f = ContentGossipFilter::new();
        // Rule 1 defers for non-"block" topics.
        f.add_rule(topic_rule("block-rule", "block"));
        // Rule 2 unconditionally accepts.
        f.add_rule(size_rule("accept-all", None, None));
        assert_eq!(f.evaluate("state", "p", 100), FilterAction::Accept);
    }

    // ── 12. get_rule ─────────────────────────────────────────────────────────

    #[test]
    fn test_get_rule_found() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("my-rule", Some(500), None));
        let r = f.get_rule("my-rule");
        assert!(r.is_some());
        assert_eq!(r.map(|r| r.max_size_bytes), Some(Some(500)));
    }

    #[test]
    fn test_get_rule_not_found() {
        let f = ContentGossipFilter::new();
        assert!(f.get_rule("missing").is_none());
    }

    // ── 13. rule_count ───────────────────────────────────────────────────────

    #[test]
    fn test_rule_count_empty() {
        let f = ContentGossipFilter::new();
        assert_eq!(f.rule_count(), 0);
    }

    #[test]
    fn test_rule_count_after_add_and_remove() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("a", None, None));
        f.add_rule(size_rule("b", None, None));
        assert_eq!(f.rule_count(), 2);
        f.remove_rule("a");
        assert_eq!(f.rule_count(), 1);
    }

    // ── 14. Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn test_zero_size_message_accepted_without_min() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("max-1k", Some(1024), None));
        assert_eq!(f.evaluate("t", "p", 0), FilterAction::Accept);
    }

    #[test]
    fn test_zero_size_message_rejected_with_min() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("min-1", None, Some(1)));
        assert_eq!(f.evaluate("t", "p", 0), FilterAction::Reject);
    }

    #[test]
    fn test_u64_max_size_accepted_when_max_is_max() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("huge-max", Some(u64::MAX), None));
        assert_eq!(f.evaluate("t", "p", u64::MAX), FilterAction::Accept);
    }

    #[test]
    fn test_empty_topic_string() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(topic_rule("anything", ""));
        // Empty pattern matches everything (substring of any string).
        assert_eq!(f.evaluate("any-topic", "p", 100), FilterAction::Accept);
    }

    #[test]
    fn test_empty_peer_id_string() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(allowlist_rule("has-empty", vec![""]));
        assert_eq!(f.evaluate("t", "", 100), FilterAction::Accept);
        assert_eq!(f.evaluate("t", "nonempty", 100), FilterAction::Reject);
    }

    // ── 15. Min and max combined ─────────────────────────────────────────────

    #[test]
    fn test_min_and_max_combined_accept_in_range() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("range", Some(1000), Some(100)));
        assert_eq!(f.evaluate("t", "p", 500), FilterAction::Accept);
    }

    #[test]
    fn test_min_and_max_combined_reject_below() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("range", Some(1000), Some(100)));
        assert_eq!(f.evaluate("t", "p", 50), FilterAction::Reject);
    }

    #[test]
    fn test_min_and_max_combined_reject_above() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("range", Some(1000), Some(100)));
        assert_eq!(f.evaluate("t", "p", 2000), FilterAction::Reject);
    }

    // ── 16. Default trait ────────────────────────────────────────────────────

    #[test]
    fn test_default_creates_empty_filter() {
        let f = ContentGossipFilter::default();
        assert_eq!(f.rule_count(), 0);
    }

    // ── 17. Stats after mixed operations ─────────────────────────────────────

    #[test]
    fn test_stats_after_mixed_operations() {
        let mut f = ContentGossipFilter::new();
        f.add_rule(size_rule("max-500", Some(500), None));

        f.evaluate("t", "p", 100); // accept
        f.evaluate("t", "p", 200); // accept
        f.evaluate("t", "p", 600); // reject (600 bytes saved)
        f.evaluate("t", "p", 800); // reject (800 bytes saved)
        f.evaluate("t", "p", 300); // accept

        let s = f.stats();
        assert_eq!(s.messages_accepted, 3);
        assert_eq!(s.messages_rejected, 2);
        assert_eq!(s.bytes_saved, 1400);
        assert_eq!(s.rule_count, 1);
    }
}
