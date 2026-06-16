//! Retention Policy Engine for IPFRS block storage.
//!
//! Evaluates which blocks to retain or expire based on configurable retention policies,
//! supporting pinning, TTLs, access-frequency thresholds, and size limits.

/// Decision made by the retention policy engine for a block.
#[derive(Debug, Clone, PartialEq)]
pub enum RetentionDecision {
    /// Keep the block with the given reason.
    Keep { reason: String },
    /// Expire (delete) the block with the given reason.
    Expire { reason: String },
    /// Defer the decision until the given Unix timestamp (seconds).
    Defer { until_secs: u64 },
}

/// Record describing a stored block and its metadata.
#[derive(Debug, Clone)]
pub struct BlockRecord {
    /// Content Identifier of the block.
    pub cid: String,
    /// Size of the block in bytes.
    pub size_bytes: u64,
    /// Unix timestamp (seconds) when the block was created.
    pub created_at_secs: u64,
    /// Unix timestamp (seconds) when the block was last accessed.
    pub last_accessed_secs: u64,
    /// Number of times this block has been accessed.
    pub access_count: u64,
    /// Whether this block is pinned (protected from GC).
    pub is_pinned: bool,
    /// Arbitrary tags attached to this block (e.g., "index", "checkpoint").
    pub tags: Vec<String>,
}

impl BlockRecord {
    /// Returns the age of the block in seconds relative to `now_secs`.
    /// Uses saturating subtraction so it never underflows.
    pub fn age_secs(&self, now_secs: u64) -> u64 {
        now_secs.saturating_sub(self.created_at_secs)
    }
}

/// A single retention rule that can match a block and produce a decision.
#[derive(Debug, Clone)]
pub enum RetentionRule {
    /// Keep any block that is pinned.
    PinProtected,
    /// Expire blocks whose age exceeds the given number of seconds.
    MaxAge { secs: u64 },
    /// Expire blocks whose access count is below the threshold AND that are older than 3600 s.
    MinAccessCount { count: u64 },
    /// Expire blocks that exceed the given size in bytes.
    MaxSize { bytes: u64 },
    /// Keep if the block has the required tag; expire otherwise.
    TagRequires { tag: String },
    /// Expire if the block has the excluded tag; no match (continue) otherwise.
    TagExcludes { tag: String },
}

/// Configuration for the `RetentionPolicyEngine`.
#[derive(Debug, Clone)]
pub struct PolicyConfig {
    /// Ordered list of rules; the first matching rule wins.
    pub rules: Vec<RetentionRule>,
    /// Decision to return when no rule matches.
    pub default_decision: RetentionDecision,
    /// How far into the future (seconds) a `Defer` decision should defer to.
    pub defer_window_secs: u64,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            default_decision: RetentionDecision::Keep {
                reason: "default".to_string(),
            },
            defer_window_secs: 3600,
        }
    }
}

/// Aggregate statistics for the retention engine since last reset.
#[derive(Debug, Clone, Default)]
pub struct RetentionEngineStats {
    /// Total number of blocks evaluated.
    pub evaluated: u64,
    /// Number of blocks that received a `Keep` decision.
    pub kept: u64,
    /// Number of blocks that received an `Expire` decision.
    pub expired: u64,
    /// Number of blocks that received a `Defer` decision.
    pub deferred: u64,
}

/// Engine that evaluates block retention decisions against a `PolicyConfig`.
#[derive(Debug)]
pub struct RetentionPolicyEngine {
    /// Policy configuration driving all evaluations.
    pub config: PolicyConfig,
    /// Running statistics since the last `reset_stats()`.
    pub stats: RetentionEngineStats,
}

impl RetentionPolicyEngine {
    /// Create a new engine with the given configuration.
    pub fn new(config: PolicyConfig) -> Self {
        Self {
            config,
            stats: RetentionEngineStats::default(),
        }
    }

    /// Evaluate the retention decision for a single `BlockRecord`.
    ///
    /// Rules are evaluated in order; the first matching rule wins.
    /// If no rule matches, `config.default_decision` is returned.
    /// Statistics are updated after every call.
    pub fn evaluate(&mut self, record: &BlockRecord, now_secs: u64) -> RetentionDecision {
        let decision = self.evaluate_inner(record, now_secs);
        self.stats.evaluated += 1;
        match &decision {
            RetentionDecision::Keep { .. } => self.stats.kept += 1,
            RetentionDecision::Expire { .. } => self.stats.expired += 1,
            RetentionDecision::Defer { .. } => self.stats.deferred += 1,
        }
        decision
    }

    /// Inner evaluation logic (does not touch stats).
    fn evaluate_inner(&self, record: &BlockRecord, now_secs: u64) -> RetentionDecision {
        let age = record.age_secs(now_secs);

        for rule in &self.config.rules {
            match rule {
                RetentionRule::PinProtected => {
                    if record.is_pinned {
                        return RetentionDecision::Keep {
                            reason: "pinned".to_string(),
                        };
                    }
                    // Not pinned — rule does not match; continue.
                }

                RetentionRule::MaxAge { secs } => {
                    if age > *secs {
                        return RetentionDecision::Expire {
                            reason: "max_age exceeded".to_string(),
                        };
                    }
                    // Young enough — rule does not match; continue.
                }

                RetentionRule::MinAccessCount { count } => {
                    if record.access_count < *count && age > 3600 {
                        return RetentionDecision::Expire {
                            reason: "low access count".to_string(),
                        };
                    }
                    // Either access_count is sufficient or block is too young — no match; continue.
                }

                RetentionRule::MaxSize { bytes } => {
                    if record.size_bytes > *bytes {
                        return RetentionDecision::Expire {
                            reason: "oversized block".to_string(),
                        };
                    }
                    // Within size limit — no match; continue.
                }

                RetentionRule::TagRequires { tag } => {
                    // This rule always matches (either keep or expire).
                    if record.tags.contains(tag) {
                        return RetentionDecision::Keep {
                            reason: "has required tag".to_string(),
                        };
                    } else {
                        return RetentionDecision::Expire {
                            reason: "missing required tag".to_string(),
                        };
                    }
                }

                RetentionRule::TagExcludes { tag } => {
                    if record.tags.contains(tag) {
                        return RetentionDecision::Expire {
                            reason: "excluded tag".to_string(),
                        };
                    }
                    // Does not have the excluded tag — no match; continue.
                }
            }
        }

        // No rule matched.
        self.config.default_decision.clone()
    }

    /// Evaluate all records in `records` and return `(cid, decision)` pairs.
    pub fn evaluate_batch(
        &mut self,
        records: &[BlockRecord],
        now_secs: u64,
    ) -> Vec<(String, RetentionDecision)> {
        records
            .iter()
            .map(|r| {
                let decision = self.evaluate(r, now_secs);
                (r.cid.clone(), decision)
            })
            .collect()
    }

    /// Return references to all blocks that should be expired.
    pub fn blocks_to_expire<'a>(
        &mut self,
        records: &'a [BlockRecord],
        now_secs: u64,
    ) -> Vec<&'a BlockRecord> {
        records
            .iter()
            .filter(|r| matches!(self.evaluate(r, now_secs), RetentionDecision::Expire { .. }))
            .collect()
    }

    /// Return a reference to the current statistics.
    pub fn stats(&self) -> &RetentionEngineStats {
        &self.stats
    }

    /// Reset all statistics counters to zero.
    pub fn reset_stats(&mut self) {
        self.stats = RetentionEngineStats::default();
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience builder for `BlockRecord` with sensible defaults.
    fn make_record(cid: &str) -> BlockRecord {
        BlockRecord {
            cid: cid.to_string(),
            size_bytes: 1024,
            created_at_secs: 0,
            last_accessed_secs: 0,
            access_count: 10,
            is_pinned: false,
            tags: Vec::new(),
        }
    }

    // ── new() ────────────────────────────────────────────────────────────────

    #[test]
    fn test_new_with_config() {
        let config = PolicyConfig::default();
        let engine = RetentionPolicyEngine::new(config);
        assert_eq!(engine.stats().evaluated, 0);
        assert_eq!(engine.stats().kept, 0);
        assert_eq!(engine.stats().expired, 0);
        assert_eq!(engine.stats().deferred, 0);
    }

    // ── PinProtected ─────────────────────────────────────────────────────────

    #[test]
    fn test_pin_protected_keeps_pinned_block() {
        let config = PolicyConfig {
            rules: vec![RetentionRule::PinProtected],
            default_decision: RetentionDecision::Expire {
                reason: "default".to_string(),
            },
            defer_window_secs: 3600,
        };
        let mut engine = RetentionPolicyEngine::new(config);
        let mut record = make_record("cid-pinned");
        record.is_pinned = true;

        let decision = engine.evaluate(&record, 9999);
        assert!(matches!(decision, RetentionDecision::Keep { .. }));
        if let RetentionDecision::Keep { reason } = decision {
            assert_eq!(reason, "pinned");
        }
    }

    #[test]
    fn test_pin_protected_skips_non_pinned() {
        let config = PolicyConfig {
            rules: vec![RetentionRule::PinProtected],
            default_decision: RetentionDecision::Expire {
                reason: "default expire".to_string(),
            },
            defer_window_secs: 3600,
        };
        let mut engine = RetentionPolicyEngine::new(config);
        let record = make_record("cid-unpinned"); // is_pinned == false

        let decision = engine.evaluate(&record, 9999);
        // PinProtected does not match → falls through to default (Expire)
        assert!(matches!(decision, RetentionDecision::Expire { .. }));
    }

    // ── MaxAge ────────────────────────────────────────────────────────────────

    #[test]
    fn test_max_age_expires_old_block() {
        let config = PolicyConfig {
            rules: vec![RetentionRule::MaxAge { secs: 3600 }],
            default_decision: RetentionDecision::Keep {
                reason: "default".to_string(),
            },
            defer_window_secs: 3600,
        };
        let mut engine = RetentionPolicyEngine::new(config);
        let mut record = make_record("cid-old");
        record.created_at_secs = 0;

        let now = 7201; // age = 7201 > 3600
        let decision = engine.evaluate(&record, now);
        assert!(matches!(decision, RetentionDecision::Expire { .. }));
        if let RetentionDecision::Expire { reason } = decision {
            assert_eq!(reason, "max_age exceeded");
        }
    }

    #[test]
    fn test_max_age_keeps_young_block() {
        let config = PolicyConfig {
            rules: vec![RetentionRule::MaxAge { secs: 3600 }],
            default_decision: RetentionDecision::Keep {
                reason: "default".to_string(),
            },
            defer_window_secs: 3600,
        };
        let mut engine = RetentionPolicyEngine::new(config);
        let mut record = make_record("cid-young");
        record.created_at_secs = 1000;

        let now = 2000; // age = 1000 <= 3600
        let decision = engine.evaluate(&record, now);
        // MaxAge does not match → falls through to Keep default
        assert!(matches!(decision, RetentionDecision::Keep { .. }));
    }

    // ── MinAccessCount ────────────────────────────────────────────────────────

    #[test]
    fn test_min_access_count_expires_low_access_old_block() {
        let config = PolicyConfig {
            rules: vec![RetentionRule::MinAccessCount { count: 5 }],
            default_decision: RetentionDecision::Keep {
                reason: "default".to_string(),
            },
            defer_window_secs: 3600,
        };
        let mut engine = RetentionPolicyEngine::new(config);
        let mut record = make_record("cid-low-access");
        record.access_count = 2; // < 5
        record.created_at_secs = 0;

        let now = 7200; // age = 7200 > 3600
        let decision = engine.evaluate(&record, now);
        assert!(matches!(decision, RetentionDecision::Expire { .. }));
        if let RetentionDecision::Expire { reason } = decision {
            assert_eq!(reason, "low access count");
        }
    }

    #[test]
    fn test_min_access_count_keeps_recently_created_block() {
        let config = PolicyConfig {
            rules: vec![RetentionRule::MinAccessCount { count: 5 }],
            default_decision: RetentionDecision::Keep {
                reason: "default".to_string(),
            },
            defer_window_secs: 3600,
        };
        let mut engine = RetentionPolicyEngine::new(config);
        let mut record = make_record("cid-new");
        record.access_count = 2; // < 5, but age <= 3600
        record.created_at_secs = 500;

        let now = 1000; // age = 500 <= 3600
        let decision = engine.evaluate(&record, now);
        // Rule does not match (age <= 3600) → default Keep
        assert!(matches!(decision, RetentionDecision::Keep { .. }));
    }

    // ── MaxSize ───────────────────────────────────────────────────────────────

    #[test]
    fn test_max_size_expires_large_block() {
        let config = PolicyConfig {
            rules: vec![RetentionRule::MaxSize { bytes: 512 }],
            default_decision: RetentionDecision::Keep {
                reason: "default".to_string(),
            },
            defer_window_secs: 3600,
        };
        let mut engine = RetentionPolicyEngine::new(config);
        let mut record = make_record("cid-large");
        record.size_bytes = 1024; // > 512

        let decision = engine.evaluate(&record, 9999);
        assert!(matches!(decision, RetentionDecision::Expire { .. }));
        if let RetentionDecision::Expire { reason } = decision {
            assert_eq!(reason, "oversized block");
        }
    }

    // ── TagRequires ───────────────────────────────────────────────────────────

    #[test]
    fn test_tag_requires_keeps_matching_block() {
        let config = PolicyConfig {
            rules: vec![RetentionRule::TagRequires {
                tag: "checkpoint".to_string(),
            }],
            default_decision: RetentionDecision::Expire {
                reason: "default".to_string(),
            },
            defer_window_secs: 3600,
        };
        let mut engine = RetentionPolicyEngine::new(config);
        let mut record = make_record("cid-checkpoint");
        record.tags = vec!["checkpoint".to_string(), "index".to_string()];

        let decision = engine.evaluate(&record, 9999);
        assert!(matches!(decision, RetentionDecision::Keep { .. }));
        if let RetentionDecision::Keep { reason } = decision {
            assert_eq!(reason, "has required tag");
        }
    }

    #[test]
    fn test_tag_requires_expires_non_matching() {
        let config = PolicyConfig {
            rules: vec![RetentionRule::TagRequires {
                tag: "checkpoint".to_string(),
            }],
            default_decision: RetentionDecision::Keep {
                reason: "default".to_string(),
            },
            defer_window_secs: 3600,
        };
        let mut engine = RetentionPolicyEngine::new(config);
        let record = make_record("cid-no-checkpoint"); // no tags

        let decision = engine.evaluate(&record, 9999);
        assert!(matches!(decision, RetentionDecision::Expire { .. }));
        if let RetentionDecision::Expire { reason } = decision {
            assert_eq!(reason, "missing required tag");
        }
    }

    // ── TagExcludes ───────────────────────────────────────────────────────────

    #[test]
    fn test_tag_excludes_expires_matching_block() {
        let config = PolicyConfig {
            rules: vec![RetentionRule::TagExcludes {
                tag: "temp".to_string(),
            }],
            default_decision: RetentionDecision::Keep {
                reason: "default".to_string(),
            },
            defer_window_secs: 3600,
        };
        let mut engine = RetentionPolicyEngine::new(config);
        let mut record = make_record("cid-temp");
        record.tags = vec!["temp".to_string()];

        let decision = engine.evaluate(&record, 9999);
        assert!(matches!(decision, RetentionDecision::Expire { .. }));
        if let RetentionDecision::Expire { reason } = decision {
            assert_eq!(reason, "excluded tag");
        }
    }

    #[test]
    fn test_tag_excludes_skips_non_matching_continues() {
        let config = PolicyConfig {
            rules: vec![
                RetentionRule::TagExcludes {
                    tag: "temp".to_string(),
                },
                RetentionRule::PinProtected,
            ],
            default_decision: RetentionDecision::Expire {
                reason: "default".to_string(),
            },
            defer_window_secs: 3600,
        };
        let mut engine = RetentionPolicyEngine::new(config);
        let mut record = make_record("cid-pinned-no-temp");
        record.is_pinned = true;
        // No "temp" tag → TagExcludes does not match → PinProtected matches → Keep

        let decision = engine.evaluate(&record, 9999);
        assert!(matches!(decision, RetentionDecision::Keep { .. }));
    }

    // ── No rules / default ────────────────────────────────────────────────────

    #[test]
    fn test_no_rules_returns_default_decision() {
        let config = PolicyConfig {
            rules: Vec::new(),
            default_decision: RetentionDecision::Defer { until_secs: 99999 },
            defer_window_secs: 3600,
        };
        let mut engine = RetentionPolicyEngine::new(config);
        let record = make_record("cid-any");

        let decision = engine.evaluate(&record, 0);
        assert!(matches!(
            decision,
            RetentionDecision::Defer { until_secs: 99999 }
        ));
    }

    // ── Rule order ────────────────────────────────────────────────────────────

    #[test]
    fn test_rule_order_first_match_wins() {
        // PinProtected appears before MaxSize; pinned block should be Kept, not Expired.
        let config = PolicyConfig {
            rules: vec![
                RetentionRule::PinProtected,
                RetentionRule::MaxSize { bytes: 0 }, // would expire everything
            ],
            default_decision: RetentionDecision::Expire {
                reason: "default".to_string(),
            },
            defer_window_secs: 3600,
        };
        let mut engine = RetentionPolicyEngine::new(config);
        let mut record = make_record("cid-pinned-large");
        record.is_pinned = true;
        record.size_bytes = 99999;

        let decision = engine.evaluate(&record, 9999);
        assert!(matches!(decision, RetentionDecision::Keep { .. }));
    }

    // ── evaluate_batch ────────────────────────────────────────────────────────

    #[test]
    fn test_evaluate_batch_returns_correct_pairs() {
        let config = PolicyConfig {
            rules: vec![RetentionRule::MaxSize { bytes: 500 }],
            default_decision: RetentionDecision::Keep {
                reason: "default".to_string(),
            },
            defer_window_secs: 3600,
        };
        let mut engine = RetentionPolicyEngine::new(config);

        let mut small = make_record("small");
        small.size_bytes = 100;
        let mut large = make_record("large");
        large.size_bytes = 1000;

        let results = engine.evaluate_batch(&[small, large], 9999);
        assert_eq!(results.len(), 2);

        let small_result = results.iter().find(|(cid, _)| cid == "small").unwrap();
        assert!(matches!(small_result.1, RetentionDecision::Keep { .. }));

        let large_result = results.iter().find(|(cid, _)| cid == "large").unwrap();
        assert!(matches!(large_result.1, RetentionDecision::Expire { .. }));
    }

    // ── blocks_to_expire ──────────────────────────────────────────────────────

    #[test]
    fn test_blocks_to_expire_filters_correctly() {
        let config = PolicyConfig {
            rules: vec![RetentionRule::MaxSize { bytes: 500 }],
            default_decision: RetentionDecision::Keep {
                reason: "default".to_string(),
            },
            defer_window_secs: 3600,
        };
        let mut engine = RetentionPolicyEngine::new(config);

        let mut small = make_record("small");
        small.size_bytes = 100;
        let mut large = make_record("large");
        large.size_bytes = 1000;

        let records = vec![small, large];
        let to_expire = engine.blocks_to_expire(&records, 9999);
        assert_eq!(to_expire.len(), 1);
        assert_eq!(to_expire[0].cid, "large");
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_updated_correctly() {
        let config = PolicyConfig {
            rules: vec![
                RetentionRule::PinProtected,
                RetentionRule::MaxSize { bytes: 500 },
            ],
            default_decision: RetentionDecision::Defer { until_secs: 9999 },
            defer_window_secs: 3600,
        };
        let mut engine = RetentionPolicyEngine::new(config);

        // Block 1: pinned → Keep
        let mut pinned = make_record("cid-pinned");
        pinned.is_pinned = true;
        engine.evaluate(&pinned, 0);

        // Block 2: large → Expire
        let mut large = make_record("cid-large");
        large.size_bytes = 1000;
        engine.evaluate(&large, 0);

        // Block 3: small, not pinned → Defer (default)
        let mut small = make_record("cid-small");
        small.size_bytes = 100;
        engine.evaluate(&small, 0);

        let stats = engine.stats();
        assert_eq!(stats.evaluated, 3);
        assert_eq!(stats.kept, 1);
        assert_eq!(stats.expired, 1);
        assert_eq!(stats.deferred, 1);
    }

    #[test]
    fn test_reset_stats_zeroes_counters() {
        let config = PolicyConfig {
            rules: vec![RetentionRule::PinProtected],
            default_decision: RetentionDecision::Keep {
                reason: "default".to_string(),
            },
            defer_window_secs: 3600,
        };
        let mut engine = RetentionPolicyEngine::new(config);

        let record = make_record("cid-any");
        engine.evaluate(&record, 0);
        engine.evaluate(&record, 0);

        assert_eq!(engine.stats().evaluated, 2);

        engine.reset_stats();

        let stats = engine.stats();
        assert_eq!(stats.evaluated, 0);
        assert_eq!(stats.kept, 0);
        assert_eq!(stats.expired, 0);
        assert_eq!(stats.deferred, 0);
    }
}
