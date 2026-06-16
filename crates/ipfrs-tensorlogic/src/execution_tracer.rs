//! TensorExecutionTracer — records detailed execution traces of TensorLogic
//! inference operations for debugging, profiling, and replay purposes.

/// Kinds of events that can be recorded in an execution trace.
#[derive(Clone, Debug, PartialEq)]
pub enum TraceEventKind {
    /// A rule evaluation has started.
    RuleEvalStart { rule_id: u64 },
    /// A rule evaluation has finished.
    RuleEvalEnd { rule_id: u64, matched: bool },
    /// A fact lookup was performed.
    FactLookup { predicate: String, found: bool },
    /// A tensor was read from storage.
    TensorRead { tensor_id: u64, size_bytes: u64 },
    /// A tensor was written to storage.
    TensorWrite { tensor_id: u64, size_bytes: u64 },
    /// An inference goal evaluation has started.
    InferenceStart { goal: String },
    /// An inference goal evaluation has finished.
    InferenceEnd { goal: String, success: bool },
}

/// A single recorded event in the execution trace.
#[derive(Clone, Debug, PartialEq)]
pub struct TraceEvent {
    /// Globally unique identifier for this event (monotonically increasing).
    pub event_id: u64,
    /// Logical clock tick at which the event was recorded.
    pub tick: u64,
    /// The kind (payload) of the event.
    pub kind: TraceEventKind,
    /// The session that produced this event.
    pub session_id: u64,
}

/// Aggregated summary statistics for a single session.
#[derive(Clone, Debug, PartialEq)]
pub struct TraceSummary {
    /// The session being summarised.
    pub session_id: u64,
    /// Total number of events in this session.
    pub event_count: usize,
    /// Number of `RuleEvalStart` events.
    pub rule_evaluations: usize,
    /// Number of `RuleEvalEnd` events where `matched == true`.
    pub successful_rules: usize,
    /// Total bytes read across all `TensorRead` events.
    pub tensor_reads: u64,
    /// Total bytes written across all `TensorWrite` events.
    pub tensor_writes: u64,
}

impl TraceSummary {
    /// Returns the fraction of rule evaluations that matched.
    ///
    /// Returns `0.0` when no rule evaluations have been recorded.
    pub fn rule_match_rate(&self) -> f64 {
        if self.rule_evaluations == 0 {
            0.0
        } else {
            self.successful_rules as f64 / self.rule_evaluations as f64
        }
    }
}

/// Configuration for [`TensorExecutionTracer`].
#[derive(Clone, Debug, PartialEq)]
pub struct TracerConfig {
    /// Maximum number of events retained in the ring buffer.  Oldest events
    /// are evicted when this limit is reached.
    pub max_events: usize,
    /// When `false` all `record` calls are no-ops.
    pub enabled: bool,
}

impl Default for TracerConfig {
    fn default() -> Self {
        Self {
            max_events: 10_000,
            enabled: true,
        }
    }
}

/// Whole-tracer statistics (across all sessions).
#[derive(Clone, Debug, PartialEq)]
pub struct TracerStats {
    /// Total number of events currently held.
    pub total_events: usize,
    /// Number of distinct session identifiers present.
    pub total_sessions: usize,
    /// Number of events that were evicted due to the `max_events` cap.
    pub dropped_events: u64,
}

/// Records detailed execution traces of TensorLogic inference operations.
///
/// The tracer maintains a capped ring buffer of [`TraceEvent`]s.  When the
/// buffer reaches `config.max_events`, the oldest event is evicted and
/// `dropped_events` is incremented.
pub struct TensorExecutionTracer {
    /// Retained events (oldest-first).
    pub events: Vec<TraceEvent>,
    /// The `event_id` that will be assigned to the next recorded event.
    pub next_event_id: u64,
    /// Tracer configuration.
    pub config: TracerConfig,
    /// Count of events dropped due to the capacity cap.
    dropped_events: u64,
}

impl TensorExecutionTracer {
    /// Creates a new tracer with the supplied configuration.
    pub fn new(config: TracerConfig) -> Self {
        Self {
            events: Vec::new(),
            next_event_id: 0,
            config,
            dropped_events: 0,
        }
    }

    /// Records an event.
    ///
    /// * If `config.enabled` is `false` the call is a no-op.
    /// * If the buffer is already at capacity the oldest event is removed and
    ///   `dropped_events` is incremented before the new event is appended.
    pub fn record(&mut self, session_id: u64, tick: u64, kind: TraceEventKind) {
        if !self.config.enabled {
            return;
        }

        if self.events.len() >= self.config.max_events {
            self.events.remove(0);
            self.dropped_events += 1;
        }

        self.events.push(TraceEvent {
            event_id: self.next_event_id,
            tick,
            kind,
            session_id,
        });
        self.next_event_id += 1;
    }

    /// Returns all events belonging to `session_id`, in chronological order.
    pub fn events_for_session(&self, session_id: u64) -> Vec<&TraceEvent> {
        self.events
            .iter()
            .filter(|e| e.session_id == session_id)
            .collect()
    }

    /// Computes aggregated statistics for `session_id`.
    pub fn summarize_session(&self, session_id: u64) -> TraceSummary {
        let mut summary = TraceSummary {
            session_id,
            event_count: 0,
            rule_evaluations: 0,
            successful_rules: 0,
            tensor_reads: 0,
            tensor_writes: 0,
        };

        for event in self.events.iter().filter(|e| e.session_id == session_id) {
            summary.event_count += 1;
            match &event.kind {
                TraceEventKind::RuleEvalStart { .. } => {
                    summary.rule_evaluations += 1;
                }
                TraceEventKind::RuleEvalEnd { matched, .. } if *matched => {
                    summary.successful_rules += 1;
                }
                TraceEventKind::RuleEvalEnd { .. } => {}
                TraceEventKind::TensorRead { size_bytes, .. } => {
                    summary.tensor_reads += size_bytes;
                }
                TraceEventKind::TensorWrite { size_bytes, .. } => {
                    summary.tensor_writes += size_bytes;
                }
                _ => {}
            }
        }

        summary
    }

    /// Removes all events that belong to `session_id`.
    pub fn clear_session(&mut self, session_id: u64) {
        self.events.retain(|e| e.session_id != session_id);
    }

    /// Returns a sorted, deduplicated list of all session identifiers present
    /// in the current event buffer.
    pub fn all_sessions(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.events.iter().map(|e| e.session_id).collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Returns aggregate statistics across all sessions.
    pub fn stats(&self) -> TracerStats {
        let total_sessions = {
            let mut ids: Vec<u64> = self.events.iter().map(|e| e.session_id).collect();
            ids.sort_unstable();
            ids.dedup();
            ids.len()
        };

        TracerStats {
            total_events: self.events.len(),
            total_sessions,
            dropped_events: self.dropped_events,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn default_tracer() -> TensorExecutionTracer {
        TensorExecutionTracer::new(TracerConfig::default())
    }

    fn rule_start(rule_id: u64) -> TraceEventKind {
        TraceEventKind::RuleEvalStart { rule_id }
    }

    fn rule_end(rule_id: u64, matched: bool) -> TraceEventKind {
        TraceEventKind::RuleEvalEnd { rule_id, matched }
    }

    fn fact_lookup(predicate: &str, found: bool) -> TraceEventKind {
        TraceEventKind::FactLookup {
            predicate: predicate.to_string(),
            found,
        }
    }

    fn tensor_read(tensor_id: u64, size_bytes: u64) -> TraceEventKind {
        TraceEventKind::TensorRead {
            tensor_id,
            size_bytes,
        }
    }

    fn tensor_write(tensor_id: u64, size_bytes: u64) -> TraceEventKind {
        TraceEventKind::TensorWrite {
            tensor_id,
            size_bytes,
        }
    }

    fn inf_start(goal: &str) -> TraceEventKind {
        TraceEventKind::InferenceStart {
            goal: goal.to_string(),
        }
    }

    fn inf_end(goal: &str, success: bool) -> TraceEventKind {
        TraceEventKind::InferenceEnd {
            goal: goal.to_string(),
            success,
        }
    }

    // ── 1. new() starts empty ────────────────────────────────────────────────

    #[test]
    fn test_new_starts_empty() {
        let tracer = default_tracer();
        assert!(tracer.events.is_empty());
        assert_eq!(tracer.next_event_id, 0);
    }

    // ── 2. record adds event ─────────────────────────────────────────────────

    #[test]
    fn test_record_adds_event() {
        let mut tracer = default_tracer();
        tracer.record(1, 0, rule_start(42));
        assert_eq!(tracer.events.len(), 1);
        assert_eq!(tracer.events[0].session_id, 1);
        assert_eq!(tracer.events[0].tick, 0);
        assert_eq!(tracer.events[0].kind, rule_start(42));
    }

    // ── 3. record is no-op when disabled ────────────────────────────────────

    #[test]
    fn test_record_noop_when_disabled() {
        let mut tracer = TensorExecutionTracer::new(TracerConfig {
            max_events: 100,
            enabled: false,
        });
        tracer.record(1, 0, rule_start(1));
        tracer.record(1, 1, rule_end(1, true));
        assert!(tracer.events.is_empty());
        assert_eq!(tracer.next_event_id, 0);
    }

    // ── 4. record evicts oldest at max_events ────────────────────────────────

    #[test]
    fn test_record_evicts_oldest_at_max_events() {
        let mut tracer = TensorExecutionTracer::new(TracerConfig {
            max_events: 3,
            enabled: true,
        });
        tracer.record(1, 0, rule_start(1));
        tracer.record(1, 1, rule_start(2));
        tracer.record(1, 2, rule_start(3));
        // Buffer is full; next record evicts event_id=0
        tracer.record(1, 3, rule_start(4));
        assert_eq!(tracer.events.len(), 3);
        assert_eq!(tracer.events[0].event_id, 1); // oldest now
        assert_eq!(tracer.events[2].event_id, 3); // newest
    }

    // ── 5. record increments next_event_id ──────────────────────────────────

    #[test]
    fn test_record_increments_event_id() {
        let mut tracer = default_tracer();
        tracer.record(1, 0, rule_start(1));
        tracer.record(1, 1, rule_start(2));
        assert_eq!(tracer.next_event_id, 2);
        assert_eq!(tracer.events[0].event_id, 0);
        assert_eq!(tracer.events[1].event_id, 1);
    }

    // ── 6. events_for_session returns correct events in order ────────────────

    #[test]
    fn test_events_for_session_order() {
        let mut tracer = default_tracer();
        tracer.record(1, 0, rule_start(1));
        tracer.record(2, 1, rule_start(99));
        tracer.record(1, 2, rule_end(1, true));
        let events = tracer.events_for_session(1);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].tick, 0);
        assert_eq!(events[1].tick, 2);
    }

    // ── 7. events_for_session empty for unknown session ──────────────────────

    #[test]
    fn test_events_for_session_unknown() {
        let mut tracer = default_tracer();
        tracer.record(1, 0, rule_start(1));
        assert!(tracer.events_for_session(999).is_empty());
    }

    // ── 8. summarize_session event_count ────────────────────────────────────

    #[test]
    fn test_summarize_event_count() {
        let mut tracer = default_tracer();
        tracer.record(1, 0, rule_start(1));
        tracer.record(1, 1, rule_end(1, false));
        tracer.record(1, 2, fact_lookup("parent", true));
        tracer.record(2, 3, rule_start(2)); // different session
        let summary = tracer.summarize_session(1);
        assert_eq!(summary.event_count, 3);
    }

    // ── 9. summarize_session rule_evaluations count ──────────────────────────

    #[test]
    fn test_summarize_rule_evaluations() {
        let mut tracer = default_tracer();
        tracer.record(1, 0, rule_start(1));
        tracer.record(1, 1, rule_start(2));
        tracer.record(1, 2, rule_end(1, true));
        let summary = tracer.summarize_session(1);
        assert_eq!(summary.rule_evaluations, 2);
    }

    // ── 10. summarize_session successful_rules count ─────────────────────────

    #[test]
    fn test_summarize_successful_rules() {
        let mut tracer = default_tracer();
        tracer.record(1, 0, rule_end(1, true));
        tracer.record(1, 1, rule_end(2, false));
        tracer.record(1, 2, rule_end(3, true));
        let summary = tracer.summarize_session(1);
        assert_eq!(summary.successful_rules, 2);
    }

    // ── 11. summarize_session tensor_reads bytes ─────────────────────────────

    #[test]
    fn test_summarize_tensor_reads() {
        let mut tracer = default_tracer();
        tracer.record(1, 0, tensor_read(10, 1024));
        tracer.record(1, 1, tensor_read(11, 2048));
        let summary = tracer.summarize_session(1);
        assert_eq!(summary.tensor_reads, 3072);
    }

    // ── 12. summarize_session tensor_writes bytes ────────────────────────────

    #[test]
    fn test_summarize_tensor_writes() {
        let mut tracer = default_tracer();
        tracer.record(1, 0, tensor_write(10, 512));
        tracer.record(1, 1, tensor_write(11, 256));
        let summary = tracer.summarize_session(1);
        assert_eq!(summary.tensor_writes, 768);
    }

    // ── 13. rule_match_rate computed correctly ───────────────────────────────

    #[test]
    fn test_rule_match_rate() {
        let mut tracer = default_tracer();
        tracer.record(1, 0, rule_start(1));
        tracer.record(1, 1, rule_start(2));
        tracer.record(1, 2, rule_end(1, true));
        tracer.record(1, 3, rule_end(2, false));
        let summary = tracer.summarize_session(1);
        // 2 evals, 1 match → 0.5
        assert!((summary.rule_match_rate() - 0.5).abs() < f64::EPSILON);
    }

    // ── 14. rule_match_rate 0.0 when no evals ───────────────────────────────

    #[test]
    fn test_rule_match_rate_zero_when_no_evals() {
        let tracer = default_tracer();
        let summary = tracer.summarize_session(99);
        assert_eq!(summary.rule_match_rate(), 0.0);
    }

    // ── 15. clear_session removes events ────────────────────────────────────

    #[test]
    fn test_clear_session() {
        let mut tracer = default_tracer();
        tracer.record(1, 0, rule_start(1));
        tracer.record(2, 1, rule_start(2));
        tracer.record(1, 2, rule_end(1, true));
        tracer.clear_session(1);
        assert_eq!(tracer.events.len(), 1);
        assert_eq!(tracer.events[0].session_id, 2);
    }

    // ── 16. all_sessions unique sorted ids ──────────────────────────────────

    #[test]
    fn test_all_sessions_unique_sorted() {
        let mut tracer = default_tracer();
        tracer.record(3, 0, rule_start(1));
        tracer.record(1, 1, rule_start(2));
        tracer.record(3, 2, rule_end(1, false));
        tracer.record(2, 3, fact_lookup("foo", true));
        let sessions = tracer.all_sessions();
        assert_eq!(sessions, vec![1, 2, 3]);
    }

    // ── 17. stats total_events correct ──────────────────────────────────────

    #[test]
    fn test_stats_total_events() {
        let mut tracer = default_tracer();
        tracer.record(1, 0, rule_start(1));
        tracer.record(2, 1, rule_start(2));
        assert_eq!(tracer.stats().total_events, 2);
    }

    // ── 18. stats total_sessions correct ────────────────────────────────────

    #[test]
    fn test_stats_total_sessions() {
        let mut tracer = default_tracer();
        tracer.record(1, 0, rule_start(1));
        tracer.record(2, 1, rule_start(2));
        tracer.record(1, 2, rule_end(1, true));
        assert_eq!(tracer.stats().total_sessions, 2);
    }

    // ── 19. stats dropped_events increments on eviction ─────────────────────

    #[test]
    fn test_stats_dropped_events() {
        let mut tracer = TensorExecutionTracer::new(TracerConfig {
            max_events: 2,
            enabled: true,
        });
        tracer.record(1, 0, rule_start(1));
        tracer.record(1, 1, rule_start(2));
        assert_eq!(tracer.stats().dropped_events, 0);
        tracer.record(1, 2, rule_start(3)); // evicts first
        assert_eq!(tracer.stats().dropped_events, 1);
        tracer.record(1, 3, rule_start(4)); // evicts second
        assert_eq!(tracer.stats().dropped_events, 2);
    }

    // ── 20. multiple sessions interleaved ────────────────────────────────────

    #[test]
    fn test_multiple_sessions_interleaved() {
        let mut tracer = default_tracer();
        tracer.record(1, 0, inf_start("goal_a"));
        tracer.record(2, 1, inf_start("goal_b"));
        tracer.record(1, 2, tensor_read(5, 256));
        tracer.record(2, 3, tensor_write(6, 512));
        tracer.record(1, 4, inf_end("goal_a", true));
        tracer.record(2, 5, inf_end("goal_b", false));

        let s1 = tracer.summarize_session(1);
        assert_eq!(s1.event_count, 3);
        assert_eq!(s1.tensor_reads, 256);

        let s2 = tracer.summarize_session(2);
        assert_eq!(s2.event_count, 3);
        assert_eq!(s2.tensor_writes, 512);
    }

    // ── 21. FactLookup events counted in event_count ─────────────────────────

    #[test]
    fn test_fact_lookup_in_event_count() {
        let mut tracer = default_tracer();
        tracer.record(1, 0, fact_lookup("edge", true));
        tracer.record(1, 1, fact_lookup("edge", false));
        tracer.record(1, 2, fact_lookup("node", true));
        let summary = tracer.summarize_session(1);
        assert_eq!(summary.event_count, 3);
        // FactLookup events do not affect rule or tensor counters
        assert_eq!(summary.rule_evaluations, 0);
        assert_eq!(summary.tensor_reads, 0);
        assert_eq!(summary.tensor_writes, 0);
    }

    // ── 22. TracerConfig enabled=false suppresses all ────────────────────────

    #[test]
    fn test_config_disabled_suppresses_all_variants() {
        let mut tracer = TensorExecutionTracer::new(TracerConfig {
            max_events: 1000,
            enabled: false,
        });
        tracer.record(1, 0, rule_start(1));
        tracer.record(1, 1, rule_end(1, true));
        tracer.record(1, 2, fact_lookup("p", false));
        tracer.record(1, 3, tensor_read(7, 128));
        tracer.record(1, 4, tensor_write(8, 64));
        tracer.record(1, 5, inf_start("g"));
        tracer.record(1, 6, inf_end("g", true));
        assert!(tracer.events.is_empty());
        assert_eq!(tracer.next_event_id, 0);
        let stats = tracer.stats();
        assert_eq!(stats.total_events, 0);
        assert_eq!(stats.dropped_events, 0);
    }

    // ── bonus: summarize returns zero-valued struct for unknown session ───────

    #[test]
    fn test_summarize_unknown_session_zeros() {
        let tracer = default_tracer();
        let summary = tracer.summarize_session(42);
        assert_eq!(summary.event_count, 0);
        assert_eq!(summary.rule_evaluations, 0);
        assert_eq!(summary.successful_rules, 0);
        assert_eq!(summary.tensor_reads, 0);
        assert_eq!(summary.tensor_writes, 0);
    }

    // ── bonus: all_sessions returns empty when no events ─────────────────────

    #[test]
    fn test_all_sessions_empty() {
        let tracer = default_tracer();
        assert!(tracer.all_sessions().is_empty());
    }

    // ── bonus: event_id is stable across sessions ─────────────────────────────

    #[test]
    fn test_event_ids_across_sessions() {
        let mut tracer = default_tracer();
        tracer.record(10, 0, rule_start(1));
        tracer.record(20, 1, rule_start(2));
        tracer.record(10, 2, rule_end(1, true));
        assert_eq!(tracer.events[0].event_id, 0);
        assert_eq!(tracer.events[1].event_id, 1);
        assert_eq!(tracer.events[2].event_id, 2);
    }

    // ── bonus: clear_session then re-record works ─────────────────────────────

    #[test]
    fn test_clear_then_rerecord() {
        let mut tracer = default_tracer();
        tracer.record(1, 0, rule_start(1));
        tracer.clear_session(1);
        assert!(tracer.events.is_empty());
        tracer.record(1, 1, rule_start(2));
        assert_eq!(tracer.events.len(), 1);
        assert_eq!(tracer.summarize_session(1).event_count, 1);
    }
}
