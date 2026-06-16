//! Session Replay Engine — records and replays inference sessions for debugging,
//! regression testing, and audit purposes.

/// A single recorded event within an inference session.
#[derive(Clone, Debug, PartialEq)]
pub enum ReplayEvent {
    /// A term was asserted into the knowledge base.
    Assert { term: String, rule_set: String },
    /// A term was retracted from the knowledge base.
    Retract { term: String },
    /// A query was submitted.
    Query { goal: String, max_depth: usize },
    /// The result of a query.
    QueryResult {
        goal: String,
        success: bool,
        bindings_count: usize,
    },
    /// A rule was loaded into the engine.
    RuleLoaded { rule_id: String, head: String },
    /// The session was started.
    SessionStart {
        session_id: String,
        timestamp_secs: u64,
    },
    /// The session ended.
    SessionEnd {
        session_id: String,
        timestamp_secs: u64,
    },
}

/// A recorded inference session consisting of an ordered list of events.
#[derive(Clone, Debug)]
pub struct ReplaySession {
    /// Unique identifier for this session.
    pub session_id: String,
    /// Ordered list of events recorded during the session.
    pub events: Vec<ReplayEvent>,
    /// Unix timestamp (seconds) when the session started.
    pub started_at_secs: u64,
}

impl ReplaySession {
    /// Creates a new empty session.
    fn new(session_id: String) -> Self {
        Self {
            session_id,
            events: Vec::new(),
            started_at_secs: 0,
        }
    }

    /// Returns the duration in seconds between `SessionStart` and `SessionEnd` events.
    /// Returns 0 if no `SessionEnd` event is present.
    pub fn duration_secs(&self) -> u64 {
        let start = self.events.iter().find_map(|e| {
            if let ReplayEvent::SessionStart { timestamp_secs, .. } = e {
                Some(*timestamp_secs)
            } else {
                None
            }
        });

        let end = self.events.iter().find_map(|e| {
            if let ReplayEvent::SessionEnd { timestamp_secs, .. } = e {
                Some(*timestamp_secs)
            } else {
                None
            }
        });

        match (start, end) {
            (Some(s), Some(e)) => e.saturating_sub(s),
            _ => 0,
        }
    }

    /// Returns the total number of events in this session.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Returns the number of `Query` events in this session.
    pub fn query_count(&self) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e, ReplayEvent::Query { .. }))
            .count()
    }

    /// Returns the fraction of `QueryResult` events that were successful.
    /// Returns `0.0` if there are no `QueryResult` events.
    pub fn success_rate(&self) -> f64 {
        let results: Vec<bool> = self
            .events
            .iter()
            .filter_map(|e| {
                if let ReplayEvent::QueryResult { success, .. } = e {
                    Some(*success)
                } else {
                    None
                }
            })
            .collect();

        if results.is_empty() {
            return 0.0;
        }

        let successes = results.iter().filter(|&&s| s).count();
        successes as f64 / results.len() as f64
    }
}

/// Filter criteria for selecting sessions from the engine.
#[derive(Clone, Debug, PartialEq)]
pub enum ReplayFilter {
    /// Return all sessions.
    All,
    /// Return sessions that contain at least one `Query` or `QueryResult` event.
    QueriesOnly,
    /// Return sessions that contain at least one `Assert` event.
    AssertionsOnly,
    /// Return the single session matching the given session ID.
    SessionId(String),
    /// Return sessions whose `started_at_secs` is within `[from_secs, to_secs]`.
    TimeRange { from_secs: u64, to_secs: u64 },
}

/// Aggregate statistics across all recorded sessions.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplayStats {
    /// Total number of recorded sessions.
    pub total_sessions: usize,
    /// Total events across all sessions.
    pub total_events: usize,
    /// Total `Query` events across all sessions.
    pub total_queries: usize,
    /// Total `Assert` events across all sessions.
    pub total_assertions: usize,
    /// Average session duration in seconds.
    pub avg_session_duration_secs: f64,
}

/// Engine that records inference session events and allows structured replay and analysis.
pub struct SessionReplayEngine {
    /// All recorded sessions, oldest-first.
    pub sessions: Vec<ReplaySession>,
    /// Maximum number of sessions to retain; oldest session is dropped when exceeded.
    pub max_sessions: usize,
}

impl SessionReplayEngine {
    /// Creates a new engine with the specified maximum session capacity.
    pub fn new(max_sessions: usize) -> Self {
        Self {
            sessions: Vec::new(),
            max_sessions,
        }
    }

    /// Records `event` for the session identified by `session_id`.
    ///
    /// If no session with that ID exists yet, a new one is created.  When the
    /// engine is already at capacity (`max_sessions`) and a new session must be
    /// created, the oldest session is dropped first.
    ///
    /// If the event is a `SessionStart`, the session's `started_at_secs` is
    /// initialised from the event's `timestamp_secs`.
    pub fn record_event(&mut self, session_id: &str, event: ReplayEvent) {
        // Check if the session already exists.
        let pos = self
            .sessions
            .iter()
            .position(|s| s.session_id == session_id);

        if let Some(idx) = pos {
            // Existing session — just append.
            if let ReplayEvent::SessionStart { timestamp_secs, .. } = &event {
                if self.sessions[idx].started_at_secs == 0 {
                    self.sessions[idx].started_at_secs = *timestamp_secs;
                }
            }
            self.sessions[idx].events.push(event);
        } else {
            // New session — enforce capacity first.
            if self.sessions.len() >= self.max_sessions {
                self.sessions.remove(0);
            }

            let mut session = ReplaySession::new(session_id.to_string());

            // Pre-populate started_at_secs if the first event is a SessionStart.
            if let ReplayEvent::SessionStart { timestamp_secs, .. } = &event {
                session.started_at_secs = *timestamp_secs;
            }

            session.events.push(event);
            self.sessions.push(session);
        }
    }

    /// Returns a reference to the session with the given ID, if it exists.
    pub fn get_session(&self, session_id: &str) -> Option<&ReplaySession> {
        self.sessions.iter().find(|s| s.session_id == session_id)
    }

    /// Returns all sessions that match the given filter.
    pub fn filter_sessions(&self, filter: &ReplayFilter) -> Vec<&ReplaySession> {
        self.sessions
            .iter()
            .filter(|s| match filter {
                ReplayFilter::All => true,
                ReplayFilter::QueriesOnly => s.events.iter().any(|e| {
                    matches!(
                        e,
                        ReplayEvent::Query { .. } | ReplayEvent::QueryResult { .. }
                    )
                }),
                ReplayFilter::AssertionsOnly => s
                    .events
                    .iter()
                    .any(|e| matches!(e, ReplayEvent::Assert { .. })),
                ReplayFilter::SessionId(id) => &s.session_id == id,
                ReplayFilter::TimeRange { from_secs, to_secs } => {
                    s.started_at_secs >= *from_secs && s.started_at_secs <= *to_secs
                }
            })
            .collect()
    }

    /// Returns only the `Query` and `QueryResult` events for the given session.
    pub fn replay_queries(&self, session_id: &str) -> Vec<&ReplayEvent> {
        match self.get_session(session_id) {
            None => Vec::new(),
            Some(session) => session
                .events
                .iter()
                .filter(|e| {
                    matches!(
                        e,
                        ReplayEvent::Query { .. } | ReplayEvent::QueryResult { .. }
                    )
                })
                .collect(),
        }
    }

    /// Computes aggregate statistics across all recorded sessions.
    pub fn stats(&self) -> ReplayStats {
        let total_sessions = self.sessions.len();
        let total_events: usize = self.sessions.iter().map(|s| s.events.len()).sum();

        let total_queries: usize = self.sessions.iter().map(|s| s.query_count()).sum();

        let total_assertions: usize = self
            .sessions
            .iter()
            .map(|s| {
                s.events
                    .iter()
                    .filter(|e| matches!(e, ReplayEvent::Assert { .. }))
                    .count()
            })
            .sum();

        let avg_session_duration_secs = if total_sessions == 0 {
            0.0
        } else {
            let total_dur: u64 = self.sessions.iter().map(|s| s.duration_secs()).sum();
            total_dur as f64 / total_sessions as f64
        };

        ReplayStats {
            total_sessions,
            total_events,
            total_queries,
            total_assertions,
            avg_session_duration_secs,
        }
    }

    /// Exports a summary of the session as a JSON-like string.
    ///
    /// Returns `None` if the session is not found.
    pub fn export_session(&self, session_id: &str) -> Option<String> {
        let session = self.get_session(session_id)?;
        let events = session.event_count();
        let queries = session.query_count();
        let success_rate = session.success_rate();
        Some(format!(
            r#"{{"session_id":"{id}","events":{events},"queries":{queries},"success_rate":{success_rate}}}"#,
            id = session.session_id,
            events = events,
            queries = queries,
            success_rate = success_rate,
        ))
    }

    /// Removes all recorded sessions from the engine.
    pub fn clear(&mut self) {
        self.sessions.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine() -> SessionReplayEngine {
        SessionReplayEngine::new(1000)
    }

    fn session_start(session_id: &str, ts: u64) -> ReplayEvent {
        ReplayEvent::SessionStart {
            session_id: session_id.to_string(),
            timestamp_secs: ts,
        }
    }

    fn session_end(session_id: &str, ts: u64) -> ReplayEvent {
        ReplayEvent::SessionEnd {
            session_id: session_id.to_string(),
            timestamp_secs: ts,
        }
    }

    fn query(goal: &str) -> ReplayEvent {
        ReplayEvent::Query {
            goal: goal.to_string(),
            max_depth: 10,
        }
    }

    fn query_result(goal: &str, success: bool, bindings_count: usize) -> ReplayEvent {
        ReplayEvent::QueryResult {
            goal: goal.to_string(),
            success,
            bindings_count,
        }
    }

    fn assert_event(term: &str) -> ReplayEvent {
        ReplayEvent::Assert {
            term: term.to_string(),
            rule_set: "default".to_string(),
        }
    }

    // 1. new() produces an empty engine
    #[test]
    fn test_new_empty() {
        let engine = make_engine();
        assert_eq!(engine.sessions.len(), 0);
        assert_eq!(engine.max_sessions, 1000);
    }

    // 2. record_event creates a session on first event
    #[test]
    fn test_record_event_creates_session() {
        let mut engine = make_engine();
        engine.record_event("s1", session_start("s1", 100));
        assert_eq!(engine.sessions.len(), 1);
        assert_eq!(engine.sessions[0].session_id, "s1");
    }

    // 3. record_event appends to an existing session
    #[test]
    fn test_record_event_appends() {
        let mut engine = make_engine();
        engine.record_event("s1", session_start("s1", 100));
        engine.record_event("s1", query("parent(X, Y)"));
        engine.record_event("s1", query_result("parent(X, Y)", true, 2));
        assert_eq!(engine.sessions.len(), 1);
        assert_eq!(engine.sessions[0].events.len(), 3);
    }

    // 4. record_event enforces max_sessions by dropping the oldest
    #[test]
    fn test_record_event_max_sessions_drops_oldest() {
        let mut engine = SessionReplayEngine::new(2);
        engine.record_event("s1", session_start("s1", 10));
        engine.record_event("s2", session_start("s2", 20));
        // Adding a third session should evict s1
        engine.record_event("s3", session_start("s3", 30));
        assert_eq!(engine.sessions.len(), 2);
        assert!(engine.get_session("s1").is_none());
        assert!(engine.get_session("s2").is_some());
        assert!(engine.get_session("s3").is_some());
    }

    // 5. get_session: found case
    #[test]
    fn test_get_session_found() {
        let mut engine = make_engine();
        engine.record_event("s1", session_start("s1", 100));
        let s = engine.get_session("s1");
        assert!(s.is_some());
        assert_eq!(s.expect("test: should succeed").session_id, "s1");
    }

    // 6. get_session: not found case
    #[test]
    fn test_get_session_not_found() {
        let engine = make_engine();
        assert!(engine.get_session("missing").is_none());
    }

    // 7. filter_sessions All returns all
    #[test]
    fn test_filter_all() {
        let mut engine = make_engine();
        engine.record_event("s1", session_start("s1", 1));
        engine.record_event("s2", session_start("s2", 2));
        let results = engine.filter_sessions(&ReplayFilter::All);
        assert_eq!(results.len(), 2);
    }

    // 8. filter_sessions QueriesOnly
    #[test]
    fn test_filter_queries_only() {
        let mut engine = make_engine();
        engine.record_event("s1", session_start("s1", 1));
        engine.record_event("s1", query("foo(X)"));
        engine.record_event("s2", session_start("s2", 2));
        engine.record_event("s2", assert_event("bar(a)"));
        let results = engine.filter_sessions(&ReplayFilter::QueriesOnly);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session_id, "s1");
    }

    // 9. filter_sessions AssertionsOnly
    #[test]
    fn test_filter_assertions_only() {
        let mut engine = make_engine();
        engine.record_event("s1", session_start("s1", 1));
        engine.record_event("s1", assert_event("parent(a, b)"));
        engine.record_event("s2", session_start("s2", 2));
        engine.record_event("s2", query("foo(X)"));
        let results = engine.filter_sessions(&ReplayFilter::AssertionsOnly);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session_id, "s1");
    }

    // 10. filter_sessions SessionId
    #[test]
    fn test_filter_session_id() {
        let mut engine = make_engine();
        engine.record_event("s1", session_start("s1", 1));
        engine.record_event("s2", session_start("s2", 2));
        let results = engine.filter_sessions(&ReplayFilter::SessionId("s2".to_string()));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session_id, "s2");
    }

    // 11. filter_sessions TimeRange
    #[test]
    fn test_filter_time_range() {
        let mut engine = make_engine();
        engine.record_event("s1", session_start("s1", 100));
        engine.record_event("s2", session_start("s2", 200));
        engine.record_event("s3", session_start("s3", 300));
        let results = engine.filter_sessions(&ReplayFilter::TimeRange {
            from_secs: 150,
            to_secs: 250,
        });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session_id, "s2");
    }

    // 12. replay_queries returns only Query + QueryResult events
    #[test]
    fn test_replay_queries_filters_correctly() {
        let mut engine = make_engine();
        engine.record_event("s1", session_start("s1", 1));
        engine.record_event("s1", assert_event("foo(a)"));
        engine.record_event("s1", query("foo(X)"));
        engine.record_event("s1", query_result("foo(X)", true, 1));
        engine.record_event("s1", session_end("s1", 100));

        let events = engine.replay_queries("s1");
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], ReplayEvent::Query { .. }));
        assert!(matches!(events[1], ReplayEvent::QueryResult { .. }));
    }

    // 13. duration_secs: correct difference between SessionStart and SessionEnd
    #[test]
    fn test_duration_secs() {
        let mut engine = make_engine();
        engine.record_event("s1", session_start("s1", 1000));
        engine.record_event("s1", session_end("s1", 1045));
        let session = engine.get_session("s1").expect("test: should succeed");
        assert_eq!(session.duration_secs(), 45);
    }

    // 14. success_rate: 1 success + 1 failure => 0.5
    #[test]
    fn test_success_rate_half() {
        let mut engine = make_engine();
        engine.record_event("s1", session_start("s1", 1));
        engine.record_event("s1", query_result("a(X)", true, 1));
        engine.record_event("s1", query_result("b(X)", false, 0));
        let session = engine.get_session("s1").expect("test: should succeed");
        let rate = session.success_rate();
        assert!((rate - 0.5).abs() < f64::EPSILON);
    }

    // 15. query_count is correct
    #[test]
    fn test_query_count() {
        let mut engine = make_engine();
        engine.record_event("s1", session_start("s1", 1));
        engine.record_event("s1", query("a(X)"));
        engine.record_event("s1", query("b(X)"));
        engine.record_event("s1", query_result("a(X)", true, 1));
        let session = engine.get_session("s1").expect("test: should succeed");
        assert_eq!(session.query_count(), 2);
    }

    // 16. stats() returns correct totals
    #[test]
    fn test_stats_totals() {
        let mut engine = make_engine();
        // session s1: 1 assert, 1 query, duration 50s
        engine.record_event("s1", session_start("s1", 0));
        engine.record_event("s1", assert_event("foo(a)"));
        engine.record_event("s1", query("foo(X)"));
        engine.record_event("s1", session_end("s1", 50));

        // session s2: 2 asserts, 2 queries, duration 100s
        engine.record_event("s2", session_start("s2", 200));
        engine.record_event("s2", assert_event("bar(b)"));
        engine.record_event("s2", assert_event("baz(c)"));
        engine.record_event("s2", query("bar(X)"));
        engine.record_event("s2", query("baz(X)"));
        engine.record_event("s2", session_end("s2", 300));

        let stats = engine.stats();
        assert_eq!(stats.total_sessions, 2);
        assert_eq!(stats.total_events, 4 + 6); // 10 total
        assert_eq!(stats.total_queries, 3);
        assert_eq!(stats.total_assertions, 3);
        // avg duration = (50 + 100) / 2 = 75.0
        assert!((stats.avg_session_duration_secs - 75.0).abs() < f64::EPSILON);
    }

    // 17. export_session: Some for known session, None for unknown
    #[test]
    fn test_export_session() {
        let mut engine = make_engine();
        engine.record_event("s1", session_start("s1", 1));
        engine.record_event("s1", query("foo(X)"));
        engine.record_event("s1", query_result("foo(X)", true, 2));

        let exported = engine.export_session("s1");
        assert!(exported.is_some());
        let s = exported.expect("test: should succeed");
        assert!(s.contains("\"session_id\":\"s1\""));
        assert!(s.contains("\"events\":3"));
        assert!(s.contains("\"queries\":1"));

        assert!(engine.export_session("nonexistent").is_none());
    }

    // 18. clear() resets everything
    #[test]
    fn test_clear() {
        let mut engine = make_engine();
        engine.record_event("s1", session_start("s1", 1));
        engine.record_event("s2", session_start("s2", 2));
        assert_eq!(engine.sessions.len(), 2);
        engine.clear();
        assert_eq!(engine.sessions.len(), 0);
        assert!(engine.get_session("s1").is_none());
    }
}
