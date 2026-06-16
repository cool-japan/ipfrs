//! Hot-reload manager for logic rule sets.
//!
//! Manages live swapping of versioned rule sets without interrupting running
//! inference sessions.  Sessions can be pinned to a specific version and
//! optionally migrated to the latest version on every `commit`.
//!
//! ## Lifecycle
//!
//! ```text
//! load_rule_set(rules)   ──► pending_version set, RuleSetLoaded logged
//!        │
//! commit()               ──► pending → current, sessions migrated, RuleSetSwapped logged
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// FNV-1a helpers (pure Rust, no external crate)
// ---------------------------------------------------------------------------

const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;

/// Compute a 64-bit FNV-1a checksum over a byte slice.
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Compute the FNV-1a checksum for a slice of rule-id strings.
///
/// Rule IDs are concatenated in order (no separator) as described in the spec.
fn checksum_rules(rule_ids: &[String]) -> u64 {
    let mut combined: Vec<u8> = Vec::new();
    for id in rule_ids {
        combined.extend_from_slice(id.as_bytes());
    }
    fnv1a_64(&combined)
}

// ---------------------------------------------------------------------------
// ReloadEvent
// ---------------------------------------------------------------------------

/// Events emitted during the hot-reload lifecycle.
#[derive(Clone, Debug)]
pub enum ReloadEvent {
    /// A new rule set has been loaded into the pending slot.
    RuleSetLoaded {
        /// The version number assigned to this rule set.
        version: u64,
        /// How many rules are in the set.
        rule_count: usize,
    },
    /// The pending rule set has been committed and swapped into the active slot.
    RuleSetSwapped {
        /// The version that was previously active.
        old_version: u64,
        /// The version that is now active.
        new_version: u64,
    },
    /// A session has been migrated from one version to another.
    SessionMigrated {
        /// The session that was migrated.
        session_id: String,
        /// The version the session was pinned to before migration.
        from_version: u64,
        /// The version the session is now pinned to.
        to_version: u64,
    },
    /// A reload attempt failed.
    ReloadFailed {
        /// Human-readable description of the failure.
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// VersionedRuleSet
// ---------------------------------------------------------------------------

/// An immutable, versioned snapshot of a rule set.
///
/// The `checksum` is the FNV-1a hash computed over all `rules` concatenated in
/// their original order.
#[derive(Clone, Debug)]
pub struct VersionedRuleSet {
    /// Monotonically increasing version counter.
    pub version: u64,
    /// Rule IDs that belong to this version.
    pub rules: Vec<String>,
    /// Unix timestamp (seconds) when this version was loaded.
    pub loaded_at_secs: u64,
    /// FNV-1a checksum of all rule IDs concatenated in order.
    pub checksum: u64,
}

impl VersionedRuleSet {
    /// Returns the number of rules in this version.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Returns `true` if `rule_id` is present in this rule set.
    pub fn contains(&self, rule_id: &str) -> bool {
        self.rules.iter().any(|r| r == rule_id)
    }
}

// ---------------------------------------------------------------------------
// LiveSession
// ---------------------------------------------------------------------------

/// Tracks a single running inference session and the rule-set version it uses.
#[derive(Clone, Debug)]
pub struct LiveSession {
    /// Unique identifier for this session.
    pub session_id: String,
    /// The rule-set version this session is currently locked to.
    pub pinned_version: u64,
    /// Unix timestamp (seconds) when the session started.
    pub started_at_secs: u64,
    /// When `true` this session may be automatically migrated to the latest
    /// rule-set version on the next `commit`.
    pub migrateable: bool,
}

// ---------------------------------------------------------------------------
// ReloadStats
// ---------------------------------------------------------------------------

/// Cumulative statistics for the `RuleHotReloadManager`.
#[derive(Clone, Debug, Default)]
pub struct ReloadStats {
    /// Total number of `commit` calls attempted.
    pub total_reloads: u64,
    /// Number of commits that completed successfully.
    pub successful_reloads: u64,
    /// Number of commits that failed.
    pub failed_reloads: u64,
    /// Total number of sessions migrated across all commits.
    pub sessions_migrated: u64,
}

impl ReloadStats {
    /// Returns the fraction of reload attempts that succeeded.
    ///
    /// Returns `0.0` if no reloads have been attempted yet.
    pub fn success_rate(&self) -> f64 {
        if self.total_reloads == 0 {
            0.0
        } else {
            self.successful_reloads as f64 / self.total_reloads as f64
        }
    }
}

// ---------------------------------------------------------------------------
// RuleHotReloadManager
// ---------------------------------------------------------------------------

/// Maximum number of events retained in the in-memory event log.
const MAX_EVENT_LOG: usize = 100;

/// Hot-reload manager that atomically swaps versioned rule sets while
/// keeping running inference sessions stable.
pub struct RuleHotReloadManager {
    /// The rule set that is currently active (served to sessions).
    pub current_version: Option<VersionedRuleSet>,
    /// The next rule set waiting to be committed.
    pub pending_version: Option<VersionedRuleSet>,
    /// All currently registered inference sessions, keyed by session ID.
    pub live_sessions: HashMap<String, LiveSession>,
    /// Bounded ring of the last `MAX_EVENT_LOG` lifecycle events.
    pub event_log: Vec<ReloadEvent>,
    /// Cumulative reload statistics.
    pub stats: ReloadStats,
    /// Counter used to assign the next version number.
    next_version: u64,
}

impl RuleHotReloadManager {
    /// Creates a new, empty manager with no rule sets loaded.
    pub fn new() -> Self {
        Self {
            current_version: None,
            pending_version: None,
            live_sessions: HashMap::new(),
            event_log: Vec::new(),
            stats: ReloadStats::default(),
            next_version: 1,
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Append an event to the bounded event log.
    ///
    /// When the log reaches `MAX_EVENT_LOG` entries the oldest entry is
    /// discarded to make room for the new one.
    fn push_event(&mut self, event: ReloadEvent) {
        if self.event_log.len() >= MAX_EVENT_LOG {
            self.event_log.remove(0);
        }
        self.event_log.push(event);
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Load a new rule set into the *pending* slot.
    ///
    /// Computes the FNV-1a checksum over the supplied rule IDs, assigns the
    /// next available version number, stores the result as `pending_version`,
    /// and appends a [`ReloadEvent::RuleSetLoaded`] event.
    ///
    /// Returns the version number assigned to the new rule set.
    pub fn load_rule_set(&mut self, rules: Vec<String>, now_secs: u64) -> u64 {
        let version = self.next_version;
        self.next_version += 1;

        let checksum = checksum_rules(&rules);
        let rule_count = rules.len();

        self.pending_version = Some(VersionedRuleSet {
            version,
            rules,
            loaded_at_secs: now_secs,
            checksum,
        });

        self.push_event(ReloadEvent::RuleSetLoaded {
            version,
            rule_count,
        });

        version
    }

    /// Commit the pending rule set to active.
    ///
    /// - If there is no pending rule set, returns `None` immediately.
    /// - Otherwise the pending set becomes the new `current_version`.
    /// - A [`ReloadEvent::RuleSetSwapped`] event is appended.
    /// - Every *migrateable* session is updated to the new version and a
    ///   [`ReloadEvent::SessionMigrated`] is appended for each one.
    /// - `stats.successful_reloads` and `stats.total_reloads` are incremented.
    ///
    /// Returns the new current version number on success.
    pub fn commit(&mut self) -> Option<u64> {
        let pending = self.pending_version.take()?;
        let new_version = pending.version;
        let old_version = self.current_version.as_ref().map(|v| v.version).unwrap_or(0);

        self.current_version = Some(pending);

        self.push_event(ReloadEvent::RuleSetSwapped {
            old_version,
            new_version,
        });

        // Migrate all migrateable sessions.
        let migrateable_ids: Vec<String> = self
            .live_sessions
            .values()
            .filter(|s| s.migrateable && s.pinned_version != new_version)
            .map(|s| s.session_id.clone())
            .collect();

        for session_id in migrateable_ids {
            if let Some(session) = self.live_sessions.get_mut(&session_id) {
                let from_version = session.pinned_version;
                session.pinned_version = new_version;
                self.stats.sessions_migrated += 1;
                // We must push the event after updating stats; borrow of
                // `self.live_sessions` was already released above.
                let event = ReloadEvent::SessionMigrated {
                    session_id: session_id.clone(),
                    from_version,
                    to_version: new_version,
                };
                // Push event — borrow checker note: `self.live_sessions` is
                // NOT borrowed here because we only borrow `self.event_log`.
                if self.event_log.len() >= MAX_EVENT_LOG {
                    self.event_log.remove(0);
                }
                self.event_log.push(event);
            }
        }

        self.stats.total_reloads += 1;
        self.stats.successful_reloads += 1;

        Some(new_version)
    }

    /// Register a new inference session.
    ///
    /// The session is pinned to `pinned_version`.  When `migrateable` is
    /// `true` the session will be automatically updated to the latest version
    /// on every subsequent `commit`.
    pub fn register_session(
        &mut self,
        session_id: String,
        pinned_version: u64,
        started_at_secs: u64,
        migrateable: bool,
    ) {
        self.live_sessions.insert(
            session_id.clone(),
            LiveSession {
                session_id,
                pinned_version,
                started_at_secs,
                migrateable,
            },
        );
    }

    /// Remove a session by ID.
    ///
    /// Returns `true` if the session existed and was removed, `false` if it
    /// was not found.
    pub fn unregister_session(&mut self, session_id: &str) -> bool {
        self.live_sessions.remove(session_id).is_some()
    }

    /// Returns all sessions whose `pinned_version` differs from the current
    /// active version.
    ///
    /// When no rule set has been committed yet (`current_version` is `None`)
    /// *all* sessions are considered to be on an "old" version.
    pub fn sessions_on_old_version(&self) -> Vec<&LiveSession> {
        match &self.current_version {
            None => self.live_sessions.values().collect(),
            Some(current) => self
                .live_sessions
                .values()
                .filter(|s| s.pinned_version != current.version)
                .collect(),
        }
    }

    /// Returns the version number of the currently active rule set, or `None`
    /// if no rule set has been committed yet.
    pub fn current_version(&self) -> Option<u64> {
        self.current_version.as_ref().map(|v| v.version)
    }

    /// Returns a reference to the cumulative reload statistics.
    pub fn stats(&self) -> &ReloadStats {
        &self.stats
    }

    /// Returns the last `n` events from the event log.
    ///
    /// If `n` exceeds the number of events stored, all stored events are
    /// returned.
    pub fn recent_events(&self, n: usize) -> &[ReloadEvent] {
        let len = self.event_log.len();
        if n >= len {
            &self.event_log
        } else {
            &self.event_log[len - n..]
        }
    }
}

impl Default for RuleHotReloadManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a manager and load a rule set with deterministic time.
    fn make_manager_with_rules(rules: Vec<&str>) -> (RuleHotReloadManager, u64) {
        let mut mgr = RuleHotReloadManager::new();
        let rule_strings: Vec<String> = rules.into_iter().map(|s| s.to_string()).collect();
        let version = mgr.load_rule_set(rule_strings, 1_000);
        (mgr, version)
    }

    // 1. new() creates an empty manager.
    #[test]
    fn test_new_is_empty() {
        let mgr = RuleHotReloadManager::new();
        assert!(mgr.current_version.is_none());
        assert!(mgr.pending_version.is_none());
        assert!(mgr.live_sessions.is_empty());
        assert!(mgr.event_log.is_empty());
        assert_eq!(mgr.stats.total_reloads, 0);
    }

    // 2. load_rule_set returns the correct version and sets pending.
    #[test]
    fn test_load_rule_set_returns_version_and_sets_pending() {
        let (mgr, version) = make_manager_with_rules(vec!["rule_a", "rule_b"]);
        assert_eq!(version, 1);
        let pending = mgr.pending_version.as_ref().expect("pending should be set");
        assert_eq!(pending.version, 1);
        assert_eq!(pending.rules.len(), 2);
    }

    // 3. load_rule_set computes a non-zero checksum.
    #[test]
    fn test_load_rule_set_checksum_computed() {
        let (mgr, _) = make_manager_with_rules(vec!["rule_a", "rule_b"]);
        let pending = mgr.pending_version.as_ref().expect("test: should succeed");
        // FNV-1a over non-empty input is never 0 in practice; also verify it
        // matches the manually computed value.
        let expected = checksum_rules(&[
            "rule_a".to_string(),
            "rule_b".to_string(),
        ]);
        assert_eq!(pending.checksum, expected);
        assert_ne!(pending.checksum, 0);
    }

    // 4. commit returns None when there is no pending rule set.
    #[test]
    fn test_commit_none_if_no_pending() {
        let mut mgr = RuleHotReloadManager::new();
        assert!(mgr.commit().is_none());
    }

    // 5. commit swaps pending into current.
    #[test]
    fn test_commit_swaps_pending_to_current() {
        let (mut mgr, version) = make_manager_with_rules(vec!["rule_x"]);
        let committed = mgr.commit().expect("commit should succeed");
        assert_eq!(committed, version);
        assert!(mgr.current_version.is_some());
        assert!(mgr.pending_version.is_none());
        assert_eq!(mgr.current_version.as_ref().expect("test: should succeed").version, version);
    }

    // 6. commit logs a RuleSetSwapped event.
    #[test]
    fn test_commit_logs_rule_set_swapped() {
        let (mut mgr, _) = make_manager_with_rules(vec!["rule_y"]);
        mgr.commit();
        let swapped = mgr.event_log.iter().any(|e| {
            matches!(e, ReloadEvent::RuleSetSwapped { new_version: 1, .. })
        });
        assert!(swapped, "expected RuleSetSwapped in event log");
    }

    // 7. commit migrates migrateable sessions.
    #[test]
    fn test_commit_migrates_migrateable_sessions() {
        let (mut mgr, v1) = make_manager_with_rules(vec!["r1"]);
        mgr.commit();

        // Load v2 and register a migrateable session pinned to v1.
        let v2 = mgr.load_rule_set(vec!["r1".to_string(), "r2".to_string()], 2_000);
        mgr.register_session("sess-1".to_string(), v1, 1_000, true);

        mgr.commit();

        let session = mgr.live_sessions.get("sess-1").expect("test: should succeed");
        assert_eq!(session.pinned_version, v2);
    }

    // 8. commit does NOT migrate non-migrateable sessions.
    #[test]
    fn test_commit_does_not_migrate_non_migrateable_sessions() {
        let (mut mgr, v1) = make_manager_with_rules(vec!["r1"]);
        mgr.commit();

        let v2 = mgr.load_rule_set(vec!["r1".to_string(), "r2".to_string()], 2_000);
        mgr.register_session("sess-locked".to_string(), v1, 1_000, false);

        mgr.commit();

        let session = mgr.live_sessions.get("sess-locked").expect("test: should succeed");
        // Should still be pinned to v1, not v2.
        assert_eq!(session.pinned_version, v1);
        assert_ne!(session.pinned_version, v2);
    }

    // 9. register_session adds a session to live_sessions.
    #[test]
    fn test_register_session_adds_to_map() {
        let mut mgr = RuleHotReloadManager::new();
        mgr.register_session("my-session".to_string(), 1, 500, true);
        assert!(mgr.live_sessions.contains_key("my-session"));
    }

    // 10. unregister_session removes and returns true; false for missing.
    #[test]
    fn test_unregister_session_removes_and_returns_correct_bool() {
        let mut mgr = RuleHotReloadManager::new();
        mgr.register_session("s1".to_string(), 1, 100, false);

        assert!(mgr.unregister_session("s1"));
        assert!(!mgr.live_sessions.contains_key("s1"));
        // Removing again should return false.
        assert!(!mgr.unregister_session("s1"));
    }

    // 11. sessions_on_old_version lists sessions with stale pinned_version.
    #[test]
    fn test_sessions_on_old_version_after_commit() {
        let (mut mgr, v1) = make_manager_with_rules(vec!["r1"]);
        mgr.commit();

        // Register an un-migrateable session pinned to v1.
        mgr.register_session("old-sess".to_string(), v1, 1_000, false);

        // Load and commit v2.
        mgr.load_rule_set(vec!["r1".to_string(), "r2".to_string()], 2_000);
        mgr.commit();

        let old = mgr.sessions_on_old_version();
        let ids: Vec<&str> = old.iter().map(|s| s.session_id.as_str()).collect();
        assert!(ids.contains(&"old-sess"), "old-sess should appear in stale list");
    }

    // 12. current_version is None before commit, Some after.
    #[test]
    fn test_current_version_none_before_commit_some_after() {
        let (mut mgr, version) = make_manager_with_rules(vec!["r1"]);
        assert!(mgr.current_version().is_none());
        mgr.commit();
        assert_eq!(mgr.current_version(), Some(version));
    }

    // 13. successful_reloads incremented after each commit.
    #[test]
    fn test_stats_successful_reloads_incremented() {
        let mut mgr = RuleHotReloadManager::new();
        mgr.load_rule_set(vec!["r1".to_string()], 1_000);
        mgr.commit();
        assert_eq!(mgr.stats().successful_reloads, 1);
        mgr.load_rule_set(vec!["r2".to_string()], 2_000);
        mgr.commit();
        assert_eq!(mgr.stats().successful_reloads, 2);
    }

    // 14. success_rate calculation.
    #[test]
    fn test_stats_success_rate() {
        let mut stats = ReloadStats {
            total_reloads: 4,
            successful_reloads: 3,
            failed_reloads: 1,
            sessions_migrated: 0,
        };
        let rate = stats.success_rate();
        assert!((rate - 0.75).abs() < f64::EPSILON);

        stats.total_reloads = 0;
        assert_eq!(stats.success_rate(), 0.0);
    }

    // 15. event_log is bounded at MAX_EVENT_LOG (100).
    #[test]
    fn test_event_log_bounded_at_100() {
        let mut mgr = RuleHotReloadManager::new();
        // Each load_rule_set produces 1 event (RuleSetLoaded).
        // Each commit produces ≥1 event (RuleSetSwapped).
        // Push more than 100 events by alternating load + commit.
        for i in 0u64..60 {
            mgr.load_rule_set(vec![format!("rule_{i}")], i * 10);
            mgr.commit();
        }
        assert!(
            mgr.event_log.len() <= MAX_EVENT_LOG,
            "event log length {} exceeds MAX_EVENT_LOG {}",
            mgr.event_log.len(),
            MAX_EVENT_LOG
        );
    }

    // 16. recent_events returns the last n events.
    #[test]
    fn test_recent_events_returns_last_n() {
        let mut mgr = RuleHotReloadManager::new();
        for i in 0u64..5 {
            mgr.load_rule_set(vec![format!("rule_{i}")], i * 10);
            mgr.commit();
        }
        // 5 loads → 5 RuleSetLoaded events
        // 5 commits → 5 RuleSetSwapped events
        // total = 10 events
        let all = mgr.event_log.len();
        let last3 = mgr.recent_events(3);
        assert_eq!(last3.len(), 3);
        // The 3 events returned should be the same as the last 3 in event_log.
        let expected = &mgr.event_log[all - 3..];
        // Compare discriminants via Debug since ReloadEvent is not PartialEq.
        for (a, b) in last3.iter().zip(expected.iter()) {
            assert_eq!(format!("{a:?}"), format!("{b:?}"));
        }
    }

    // 17. load_rule_set twice increments version counter.
    #[test]
    fn test_load_twice_version_increments() {
        let mut mgr = RuleHotReloadManager::new();
        let v1 = mgr.load_rule_set(vec!["r1".to_string()], 100);
        mgr.commit();
        let v2 = mgr.load_rule_set(vec!["r2".to_string()], 200);
        assert_eq!(v2, v1 + 1, "second load should get next version");
    }

    // Bonus: VersionedRuleSet::contains and rule_count.
    #[test]
    fn test_versioned_ruleset_contains_and_rule_count() {
        let vrs = VersionedRuleSet {
            version: 1,
            rules: vec!["alpha".to_string(), "beta".to_string()],
            loaded_at_secs: 0,
            checksum: 0,
        };
        assert_eq!(vrs.rule_count(), 2);
        assert!(vrs.contains("alpha"));
        assert!(!vrs.contains("gamma"));
    }

    // Bonus: sessions_on_old_version when current is None returns all sessions.
    #[test]
    fn test_sessions_on_old_version_when_no_current_returns_all() {
        let mut mgr = RuleHotReloadManager::new();
        mgr.register_session("s1".to_string(), 1, 0, true);
        mgr.register_session("s2".to_string(), 2, 0, false);
        let old = mgr.sessions_on_old_version();
        assert_eq!(old.len(), 2);
    }
}
