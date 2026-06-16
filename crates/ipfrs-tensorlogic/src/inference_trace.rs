//! Inference trace recorder for debugging and performance analysis.
//!
//! Records detailed traces of inference execution: which rules fired, in what
//! order, with what bindings, and summary statistics.

/// A single event recorded during inference execution.
#[derive(Clone, Debug)]
pub enum TraceEvent {
    /// A rule was fired during backward/forward chaining.
    RuleFired {
        rule_id: String,
        goal: String,
        depth: usize,
        bindings: Vec<(String, String)>,
    },
    /// A goal was resolved (successfully or not).
    GoalResolved {
        goal: String,
        depth: usize,
        success: bool,
    },
    /// A backtrack point was recorded with the number of alternatives.
    BacktrackPoint { goal: String, alternatives: usize },
    /// The inference result was served from cache.
    CacheHit { goal: String, depth: usize },
    /// The maximum recursion depth was reached for a goal.
    MaxDepthReached { goal: String, depth: usize },
}

/// A named span grouping a contiguous window of trace events.
#[derive(Clone, Debug)]
pub struct TraceSpan {
    /// Monotonically increasing identifier assigned at span creation.
    pub span_id: u64,
    /// Human-readable label for this span.
    pub label: String,
    /// Index into the events vec at the time the span was opened.
    pub start_event_idx: usize,
    /// Index into the events vec at the time the span was closed.
    /// `None` while the span is still open.
    pub end_event_idx: Option<usize>,
}

impl TraceSpan {
    /// Number of events captured inside this span.
    ///
    /// Returns `0` for an unclosed span.
    pub fn duration_events(&self) -> usize {
        match self.end_event_idx {
            Some(end) => end.saturating_sub(self.start_event_idx),
            None => 0,
        }
    }
}

/// Aggregated counters derived from recorded events.
#[derive(Clone, Debug, Default)]
pub struct TraceStats {
    pub total_events: u64,
    pub rules_fired: u64,
    pub goals_resolved: u64,
    pub cache_hits: u64,
    pub backtracks: u64,
    pub max_depth_reached: u64,
}

impl TraceStats {
    /// Fraction of resolved goals that were served from cache.
    ///
    /// Returns `0.0` when no goals have been resolved yet.
    pub fn cache_hit_rate(&self) -> f64 {
        self.cache_hits as f64 / self.goals_resolved.max(1) as f64
    }
}

/// Records and queries a bounded trace of inference execution events.
pub struct InferenceTraceRecorder {
    /// Recorded events in order of occurrence.
    pub events: Vec<TraceEvent>,
    /// Named spans.
    pub spans: Vec<TraceSpan>,
    /// Counter used to hand out monotonically increasing span IDs.
    pub next_span_id: u64,
    /// Aggregated statistics.
    pub stats: TraceStats,
    /// Maximum number of events to retain (oldest are dropped when exceeded).
    pub max_events: usize,
}

impl InferenceTraceRecorder {
    /// Create a new recorder with the given event capacity cap.
    pub fn new(max_events: usize) -> Self {
        Self {
            events: Vec::new(),
            spans: Vec::new(),
            next_span_id: 0,
            stats: TraceStats::default(),
            max_events,
        }
    }

    /// Append an event, updating statistics.
    ///
    /// If the number of events would exceed `max_events`, the oldest event is
    /// removed first (and span indices that referenced it are **not** adjusted
    /// — callers should treat indices as approximate after overflow).
    pub fn record(&mut self, event: TraceEvent) {
        // Update counters before the potential removal so totals stay accurate.
        self.stats.total_events += 1;
        match &event {
            TraceEvent::RuleFired { .. } => self.stats.rules_fired += 1,
            TraceEvent::GoalResolved { .. } => self.stats.goals_resolved += 1,
            TraceEvent::CacheHit { .. } => self.stats.cache_hits += 1,
            TraceEvent::BacktrackPoint { .. } => self.stats.backtracks += 1,
            TraceEvent::MaxDepthReached { .. } => self.stats.max_depth_reached += 1,
        }

        if self.events.len() >= self.max_events {
            self.events.remove(0);
        }
        self.events.push(event);
    }

    /// Open a new named span and return its ID.
    pub fn begin_span(&mut self, label: String) -> u64 {
        let id = self.next_span_id;
        self.next_span_id += 1;
        self.spans.push(TraceSpan {
            span_id: id,
            label,
            start_event_idx: self.events.len(),
            end_event_idx: None,
        });
        id
    }

    /// Close the span identified by `span_id`, recording the current event
    /// count as the end boundary.
    pub fn end_span(&mut self, span_id: u64) {
        let current_len = self.events.len();
        if let Some(span) = self.spans.iter_mut().find(|s| s.span_id == span_id) {
            span.end_event_idx = Some(current_len);
        }
    }

    /// Return the slice of events that fall within the given span.
    ///
    /// Returns an empty slice when the span is unknown or not yet closed.
    pub fn events_in_span(&self, span_id: u64) -> &[TraceEvent] {
        let span = match self.spans.iter().find(|s| s.span_id == span_id) {
            Some(s) => s,
            None => return &[],
        };
        let end = match span.end_event_idx {
            Some(e) => e,
            None => return &[],
        };
        let start = span.start_event_idx.min(self.events.len());
        let end = end.min(self.events.len());
        if start >= end {
            return &[];
        }
        &self.events[start..end]
    }

    /// Return all events for which `pred` returns `true`.
    pub fn filter_events(&self, pred: impl Fn(&TraceEvent) -> bool) -> Vec<&TraceEvent> {
        self.events.iter().filter(|e| pred(e)).collect()
    }

    /// Return only the `RuleFired` events that fall inside the given span.
    pub fn rules_fired_in_span(&self, span_id: u64) -> Vec<&TraceEvent> {
        self.events_in_span(span_id)
            .iter()
            .filter(|e| matches!(e, TraceEvent::RuleFired { .. }))
            .collect()
    }

    /// Borrow the accumulated statistics.
    pub fn stats(&self) -> &TraceStats {
        &self.stats
    }

    /// Reset all recorded data and statistics.
    pub fn clear(&mut self) {
        self.events.clear();
        self.spans.clear();
        self.next_span_id = 0;
        self.stats = TraceStats::default();
    }

    /// Produce a one-line human-readable summary of the trace statistics.
    pub fn export_summary(&self) -> String {
        format!(
            "events={} rules={} cache_hits={} backtracks={}",
            self.stats.total_events,
            self.stats.rules_fired,
            self.stats.cache_hits,
            self.stats.backtracks,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule_fired(rule_id: &str, goal: &str) -> TraceEvent {
        TraceEvent::RuleFired {
            rule_id: rule_id.to_string(),
            goal: goal.to_string(),
            depth: 1,
            bindings: vec![("X".to_string(), "alice".to_string())],
        }
    }

    fn goal_resolved(goal: &str, success: bool) -> TraceEvent {
        TraceEvent::GoalResolved {
            goal: goal.to_string(),
            depth: 1,
            success,
        }
    }

    fn cache_hit(goal: &str) -> TraceEvent {
        TraceEvent::CacheHit {
            goal: goal.to_string(),
            depth: 1,
        }
    }

    fn backtrack(goal: &str) -> TraceEvent {
        TraceEvent::BacktrackPoint {
            goal: goal.to_string(),
            alternatives: 3,
        }
    }

    fn max_depth(goal: &str) -> TraceEvent {
        TraceEvent::MaxDepthReached {
            goal: goal.to_string(),
            depth: 10,
        }
    }

    // 1. new() produces empty state
    #[test]
    fn test_new_empty_state() {
        let recorder = InferenceTraceRecorder::new(100);
        assert!(recorder.events.is_empty());
        assert!(recorder.spans.is_empty());
        assert_eq!(recorder.next_span_id, 0);
        assert_eq!(recorder.stats.total_events, 0);
        assert_eq!(recorder.max_events, 100);
    }

    // 2. record RuleFired updates stats.rules_fired
    #[test]
    fn test_record_rule_fired_stats() {
        let mut r = InferenceTraceRecorder::new(100);
        r.record(rule_fired("r1", "parent(X,Y)"));
        assert_eq!(r.stats.rules_fired, 1);
        assert_eq!(r.stats.total_events, 1);
    }

    // 3. record GoalResolved updates stats.goals_resolved
    #[test]
    fn test_record_goal_resolved_stats() {
        let mut r = InferenceTraceRecorder::new(100);
        r.record(goal_resolved("parent(alice,bob)", true));
        assert_eq!(r.stats.goals_resolved, 1);
        assert_eq!(r.stats.total_events, 1);
    }

    // 4. record CacheHit updates stats.cache_hits
    #[test]
    fn test_record_cache_hit_stats() {
        let mut r = InferenceTraceRecorder::new(100);
        r.record(cache_hit("ancestor(X,Y)"));
        assert_eq!(r.stats.cache_hits, 1);
        assert_eq!(r.stats.total_events, 1);
    }

    // 5. record BacktrackPoint updates stats.backtracks
    #[test]
    fn test_record_backtrack_stats() {
        let mut r = InferenceTraceRecorder::new(100);
        r.record(backtrack("foo(X)"));
        assert_eq!(r.stats.backtracks, 1);
        assert_eq!(r.stats.total_events, 1);
    }

    // 6. record MaxDepthReached updates stats.max_depth_reached
    #[test]
    fn test_record_max_depth_stats() {
        let mut r = InferenceTraceRecorder::new(100);
        r.record(max_depth("loop(X)"));
        assert_eq!(r.stats.max_depth_reached, 1);
        assert_eq!(r.stats.total_events, 1);
    }

    // 7. max_events cap drops oldest event
    #[test]
    fn test_max_events_cap_drops_oldest() {
        let mut r = InferenceTraceRecorder::new(3);
        r.record(rule_fired("r1", "goal1"));
        r.record(rule_fired("r2", "goal2"));
        r.record(rule_fired("r3", "goal3"));
        // All three should be present
        assert_eq!(r.events.len(), 3);

        // Adding a fourth should drop the first
        r.record(rule_fired("r4", "goal4"));
        assert_eq!(r.events.len(), 3);
        // The surviving events should be r2, r3, r4
        let goals: Vec<String> = r
            .events
            .iter()
            .map(|e| match e {
                TraceEvent::RuleFired { goal, .. } => goal.clone(),
                _ => String::new(),
            })
            .collect();
        assert_eq!(goals, vec!["goal2", "goal3", "goal4"]);
        // Total events counter is still 4
        assert_eq!(r.stats.total_events, 4);
    }

    // 8. begin_span returns monotonic ids
    #[test]
    fn test_begin_span_monotonic_ids() {
        let mut r = InferenceTraceRecorder::new(100);
        let id0 = r.begin_span("span-a".to_string());
        let id1 = r.begin_span("span-b".to_string());
        let id2 = r.begin_span("span-c".to_string());
        assert!(id0 < id1);
        assert!(id1 < id2);
    }

    // 9. end_span sets end_event_idx
    #[test]
    fn test_end_span_sets_end_event_idx() {
        let mut r = InferenceTraceRecorder::new(100);
        let id = r.begin_span("my-span".to_string());
        r.record(rule_fired("r1", "g1"));
        r.end_span(id);
        let span = r
            .spans
            .iter()
            .find(|s| s.span_id == id)
            .expect("test: should succeed");
        assert_eq!(span.end_event_idx, Some(1));
    }

    // 10. events_in_span returns correct slice
    #[test]
    fn test_events_in_span_correct_slice() {
        let mut r = InferenceTraceRecorder::new(100);
        // event before span
        r.record(rule_fired("r0", "before"));
        let id = r.begin_span("my-span".to_string());
        r.record(rule_fired("r1", "inside1"));
        r.record(goal_resolved("inside2", true));
        r.end_span(id);
        // event after span
        r.record(rule_fired("r2", "after"));

        let slice = r.events_in_span(id);
        assert_eq!(slice.len(), 2);
        match &slice[0] {
            TraceEvent::RuleFired { goal, .. } => assert_eq!(goal, "inside1"),
            _ => panic!("unexpected event"),
        }
        match &slice[1] {
            TraceEvent::GoalResolved { goal, .. } => assert_eq!(goal, "inside2"),
            _ => panic!("unexpected event"),
        }
    }

    // 11. events_in_span unknown span_id returns empty
    #[test]
    fn test_events_in_span_unknown_returns_empty() {
        let r = InferenceTraceRecorder::new(100);
        assert!(r.events_in_span(9999).is_empty());
    }

    // 12. filter_events by closure
    #[test]
    fn test_filter_events_by_closure() {
        let mut r = InferenceTraceRecorder::new(100);
        r.record(rule_fired("r1", "g1"));
        r.record(goal_resolved("g2", true));
        r.record(cache_hit("g3"));
        r.record(rule_fired("r2", "g4"));

        let fired = r.filter_events(|e| matches!(e, TraceEvent::RuleFired { .. }));
        assert_eq!(fired.len(), 2);
    }

    // 13. rules_fired_in_span filters correctly
    #[test]
    fn test_rules_fired_in_span() {
        let mut r = InferenceTraceRecorder::new(100);
        let id = r.begin_span("test".to_string());
        r.record(rule_fired("r1", "g1"));
        r.record(goal_resolved("g2", true));
        r.record(cache_hit("g3"));
        r.record(rule_fired("r2", "g4"));
        r.end_span(id);

        let fired = r.rules_fired_in_span(id);
        assert_eq!(fired.len(), 2);
        for e in fired {
            assert!(matches!(e, TraceEvent::RuleFired { .. }));
        }
    }

    // 14. cache_hit_rate calculation
    #[test]
    fn test_cache_hit_rate() {
        let mut r = InferenceTraceRecorder::new(100);
        // 2 goals resolved, 1 cache hit → 0.5
        r.record(goal_resolved("g1", true));
        r.record(goal_resolved("g2", false));
        r.record(cache_hit("g3"));
        let rate = r.stats().cache_hit_rate();
        assert!((rate - 0.5).abs() < f64::EPSILON);
    }

    // 15. clear() resets everything
    #[test]
    fn test_clear_resets_everything() {
        let mut r = InferenceTraceRecorder::new(100);
        r.record(rule_fired("r1", "g1"));
        let _ = r.begin_span("s1".to_string());
        r.clear();
        assert!(r.events.is_empty());
        assert!(r.spans.is_empty());
        assert_eq!(r.next_span_id, 0);
        assert_eq!(r.stats.total_events, 0);
        assert_eq!(r.stats.rules_fired, 0);
        assert_eq!(r.stats.goals_resolved, 0);
        assert_eq!(r.stats.cache_hits, 0);
        assert_eq!(r.stats.backtracks, 0);
        assert_eq!(r.stats.max_depth_reached, 0);
    }

    // 16. export_summary format correct
    #[test]
    fn test_export_summary_format() {
        let mut r = InferenceTraceRecorder::new(100);
        r.record(rule_fired("r1", "g1"));
        r.record(cache_hit("g2"));
        r.record(backtrack("g3"));
        let summary = r.export_summary();
        assert_eq!(summary, "events=3 rules=1 cache_hits=1 backtracks=1");
    }

    // 17. duration_events: closed vs open span
    #[test]
    fn test_duration_events_closed_vs_open() {
        let mut r = InferenceTraceRecorder::new(100);
        let open_id = r.begin_span("open".to_string());
        r.record(rule_fired("r1", "g1"));
        r.record(rule_fired("r2", "g2"));

        let closed_id = r.begin_span("closed".to_string());
        r.record(rule_fired("r3", "g3"));
        r.end_span(closed_id);

        // open span: end_event_idx is None → duration_events() == 0
        let open_span = r
            .spans
            .iter()
            .find(|s| s.span_id == open_id)
            .expect("test: should succeed");
        assert_eq!(open_span.duration_events(), 0);

        // closed span: captured exactly 1 event
        let closed_span = r
            .spans
            .iter()
            .find(|s| s.span_id == closed_id)
            .expect("test: should succeed");
        assert_eq!(closed_span.duration_events(), 1);
    }

    // Bonus: cache_hit_rate when no goals resolved (avoid division by zero)
    #[test]
    fn test_cache_hit_rate_no_goals() {
        let r = InferenceTraceRecorder::new(100);
        assert_eq!(r.stats().cache_hit_rate(), 0.0);
    }
}
