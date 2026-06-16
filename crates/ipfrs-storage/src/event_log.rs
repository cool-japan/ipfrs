//! Structured event logging for storage operations.
//!
//! Provides a bounded, queryable event log that tracks storage operations
//! such as block additions, deletions, compaction events, quota violations,
//! and errors. Events are tagged with type, severity, and logical tick,
//! enabling efficient filtering and diagnostics.

use std::collections::{HashMap, VecDeque};

/// The kind of storage operation that generated an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    /// A new block was added to storage.
    BlockAdded,
    /// A block was deleted from storage.
    BlockDeleted,
    /// A block was read / accessed.
    BlockAccessed,
    /// A compaction cycle started.
    CompactionStarted,
    /// A compaction cycle completed.
    CompactionCompleted,
    /// A quota limit was exceeded.
    QuotaExceeded,
    /// An error occurred during a storage operation.
    Error,
}

/// Severity level of a storage event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventSeverity {
    /// Informational – normal operation.
    Info,
    /// Warning – something unusual but not fatal.
    Warning,
    /// Error – an operation failed.
    Error,
    /// Critical – requires immediate attention.
    Critical,
}

/// A single storage event record.
#[derive(Debug, Clone)]
pub struct StorageEvent {
    /// Unique, monotonically increasing event identifier.
    pub id: u64,
    /// The type of storage operation.
    pub event_type: EventType,
    /// Severity level.
    pub severity: EventSeverity,
    /// Human-readable description.
    pub message: String,
    /// Optional CID of the block involved.
    pub block_cid: Option<String>,
    /// Logical tick at which the event was recorded.
    pub tick: u64,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

/// Aggregate statistics about the event log.
#[derive(Debug, Clone)]
pub struct EventLogStats {
    /// Current number of events in the log.
    pub total_events: usize,
    /// Counts keyed by severity name.
    pub events_by_severity: HashMap<String, u64>,
    /// Tick of the oldest event, if any.
    pub oldest_tick: Option<u64>,
    /// Tick of the newest event, if any.
    pub newest_tick: Option<u64>,
}

/// A bounded, queryable log of [`StorageEvent`]s.
///
/// When the number of stored events exceeds `max_events`, the oldest
/// event is evicted on each new insertion.
pub struct StorageEventLog {
    events: VecDeque<StorageEvent>,
    max_events: usize,
    next_id: u64,
    current_tick: u64,
    counts_by_type: HashMap<EventType, u64>,
}

impl StorageEventLog {
    /// Create a new event log that retains at most `max_events` entries.
    pub fn new(max_events: usize) -> Self {
        Self {
            events: VecDeque::new(),
            max_events,
            next_id: 0,
            current_tick: 0,
            counts_by_type: HashMap::new(),
        }
    }

    /// Append a new event and return its unique id.
    ///
    /// If the log has reached `max_events`, the oldest event is evicted first.
    pub fn log(
        &mut self,
        event_type: EventType,
        severity: EventSeverity,
        message: &str,
        block_cid: Option<&str>,
        metadata: HashMap<String, String>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);

        let event = StorageEvent {
            id,
            event_type,
            severity,
            message: message.to_string(),
            block_cid: block_cid.map(|s| s.to_string()),
            tick: self.current_tick,
            metadata,
        };

        // Evict oldest if at capacity.
        if self.events.len() >= self.max_events {
            self.events.pop_front();
        }

        *self.counts_by_type.entry(event_type).or_insert(0) += 1;
        self.events.push_back(event);

        id
    }

    /// Look up an event by its id.
    pub fn get_event(&self, id: u64) -> Option<&StorageEvent> {
        self.events.iter().find(|e| e.id == id)
    }

    /// Return all events matching the given [`EventType`].
    pub fn events_by_type(&self, event_type: EventType) -> Vec<&StorageEvent> {
        self.events
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    /// Return all events matching the given [`EventSeverity`].
    pub fn events_by_severity(&self, severity: EventSeverity) -> Vec<&StorageEvent> {
        self.events
            .iter()
            .filter(|e| e.severity == severity)
            .collect()
    }

    /// Return all events whose tick is strictly greater than `tick`.
    pub fn events_since(&self, tick: u64) -> Vec<&StorageEvent> {
        self.events.iter().filter(|e| e.tick > tick).collect()
    }

    /// Return the cumulative count of events of the given type
    /// (including evicted events).
    pub fn count_by_type(&self, event_type: EventType) -> u64 {
        self.counts_by_type.get(&event_type).copied().unwrap_or(0)
    }

    /// Return the last `n` events (or fewer if the log is smaller).
    pub fn recent(&self, n: usize) -> Vec<&StorageEvent> {
        let len = self.events.len();
        let skip = len.saturating_sub(n);
        self.events.iter().skip(skip).collect()
    }

    /// Advance the logical tick by one.
    pub fn tick(&mut self) {
        self.current_tick = self.current_tick.saturating_add(1);
    }

    /// Remove all events from the log.
    ///
    /// Cumulative type counts and the current tick are preserved.
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Return the number of events currently in the log.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Compute aggregate statistics over the current event log.
    pub fn stats(&self) -> EventLogStats {
        let mut events_by_severity: HashMap<String, u64> = HashMap::new();
        for event in &self.events {
            let key = format!("{:?}", event.severity);
            *events_by_severity.entry(key).or_insert(0) += 1;
        }

        let oldest_tick = self.events.front().map(|e| e.tick);
        let newest_tick = self.events.back().map(|e| e.tick);

        EventLogStats {
            total_events: self.events.len(),
            events_by_severity,
            oldest_tick,
            newest_tick,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_meta() -> HashMap<String, String> {
        HashMap::new()
    }

    // 1. Log a single event and verify id.
    #[test]
    fn test_log_returns_id() {
        let mut log = StorageEventLog::new(100);
        let id = log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "added",
            None,
            empty_meta(),
        );
        assert_eq!(id, 0);
    }

    // 2. Sequential ids.
    #[test]
    fn test_sequential_ids() {
        let mut log = StorageEventLog::new(100);
        let a = log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "a",
            None,
            empty_meta(),
        );
        let b = log.log(
            EventType::BlockDeleted,
            EventSeverity::Info,
            "b",
            None,
            empty_meta(),
        );
        assert_eq!(a, 0);
        assert_eq!(b, 1);
    }

    // 3. get_event returns correct event.
    #[test]
    fn test_get_event() {
        let mut log = StorageEventLog::new(100);
        let id = log.log(
            EventType::BlockAccessed,
            EventSeverity::Warning,
            "access",
            Some("Qm123"),
            empty_meta(),
        );
        let evt = log.get_event(id).expect("event should exist");
        assert_eq!(evt.event_type, EventType::BlockAccessed);
        assert_eq!(evt.severity, EventSeverity::Warning);
        assert_eq!(evt.message, "access");
        assert_eq!(evt.block_cid.as_deref(), Some("Qm123"));
    }

    // 4. get_event returns None for missing id.
    #[test]
    fn test_get_event_missing() {
        let log = StorageEventLog::new(100);
        assert!(log.get_event(42).is_none());
    }

    // 5. events_by_type filters correctly.
    #[test]
    fn test_events_by_type() {
        let mut log = StorageEventLog::new(100);
        log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "a",
            None,
            empty_meta(),
        );
        log.log(
            EventType::BlockDeleted,
            EventSeverity::Info,
            "b",
            None,
            empty_meta(),
        );
        log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "c",
            None,
            empty_meta(),
        );
        let added = log.events_by_type(EventType::BlockAdded);
        assert_eq!(added.len(), 2);
    }

    // 6. events_by_severity filters correctly.
    #[test]
    fn test_events_by_severity() {
        let mut log = StorageEventLog::new(100);
        log.log(
            EventType::Error,
            EventSeverity::Error,
            "err1",
            None,
            empty_meta(),
        );
        log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "ok",
            None,
            empty_meta(),
        );
        log.log(
            EventType::Error,
            EventSeverity::Error,
            "err2",
            None,
            empty_meta(),
        );
        let errors = log.events_by_severity(EventSeverity::Error);
        assert_eq!(errors.len(), 2);
    }

    // 7. events_since returns events after tick.
    #[test]
    fn test_events_since() {
        let mut log = StorageEventLog::new(100);
        log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "t0",
            None,
            empty_meta(),
        );
        log.tick();
        log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "t1",
            None,
            empty_meta(),
        );
        log.tick();
        log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "t2",
            None,
            empty_meta(),
        );
        let since = log.events_since(0);
        assert_eq!(since.len(), 2);
    }

    // 8. count_by_type tracks cumulative counts.
    #[test]
    fn test_count_by_type() {
        let mut log = StorageEventLog::new(100);
        log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "a",
            None,
            empty_meta(),
        );
        log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "b",
            None,
            empty_meta(),
        );
        log.log(
            EventType::BlockDeleted,
            EventSeverity::Info,
            "c",
            None,
            empty_meta(),
        );
        assert_eq!(log.count_by_type(EventType::BlockAdded), 2);
        assert_eq!(log.count_by_type(EventType::BlockDeleted), 1);
        assert_eq!(log.count_by_type(EventType::Error), 0);
    }

    // 9. recent returns last n events.
    #[test]
    fn test_recent() {
        let mut log = StorageEventLog::new(100);
        for i in 0..5 {
            log.log(
                EventType::BlockAdded,
                EventSeverity::Info,
                &format!("e{i}"),
                None,
                empty_meta(),
            );
        }
        let r = log.recent(3);
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].id, 2);
        assert_eq!(r[2].id, 4);
    }

    // 10. recent with n > event_count.
    #[test]
    fn test_recent_exceeds_count() {
        let mut log = StorageEventLog::new(100);
        log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "only",
            None,
            empty_meta(),
        );
        let r = log.recent(10);
        assert_eq!(r.len(), 1);
    }

    // 11. Max events eviction.
    #[test]
    fn test_max_events_eviction() {
        let mut log = StorageEventLog::new(3);
        for i in 0..5 {
            log.log(
                EventType::BlockAdded,
                EventSeverity::Info,
                &format!("e{i}"),
                None,
                empty_meta(),
            );
        }
        assert_eq!(log.event_count(), 3);
        // Oldest remaining should be id=2.
        assert!(log.get_event(0).is_none());
        assert!(log.get_event(1).is_none());
        assert!(log.get_event(2).is_some());
    }

    // 12. Eviction preserves cumulative counts.
    #[test]
    fn test_eviction_preserves_counts() {
        let mut log = StorageEventLog::new(2);
        for _ in 0..5 {
            log.log(
                EventType::BlockAdded,
                EventSeverity::Info,
                "x",
                None,
                empty_meta(),
            );
        }
        assert_eq!(log.count_by_type(EventType::BlockAdded), 5);
    }

    // 13. clear removes all events.
    #[test]
    fn test_clear() {
        let mut log = StorageEventLog::new(100);
        log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "a",
            None,
            empty_meta(),
        );
        log.log(
            EventType::BlockDeleted,
            EventSeverity::Info,
            "b",
            None,
            empty_meta(),
        );
        log.clear();
        assert_eq!(log.event_count(), 0);
    }

    // 14. clear preserves cumulative counts.
    #[test]
    fn test_clear_preserves_counts() {
        let mut log = StorageEventLog::new(100);
        log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "a",
            None,
            empty_meta(),
        );
        log.clear();
        assert_eq!(log.count_by_type(EventType::BlockAdded), 1);
    }

    // 15. stats total_events.
    #[test]
    fn test_stats_total() {
        let mut log = StorageEventLog::new(100);
        log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "a",
            None,
            empty_meta(),
        );
        log.log(
            EventType::Error,
            EventSeverity::Error,
            "e",
            None,
            empty_meta(),
        );
        let s = log.stats();
        assert_eq!(s.total_events, 2);
    }

    // 16. stats events_by_severity.
    #[test]
    fn test_stats_severity_counts() {
        let mut log = StorageEventLog::new(100);
        log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "a",
            None,
            empty_meta(),
        );
        log.log(
            EventType::Error,
            EventSeverity::Error,
            "e",
            None,
            empty_meta(),
        );
        log.log(
            EventType::QuotaExceeded,
            EventSeverity::Warning,
            "q",
            None,
            empty_meta(),
        );
        let s = log.stats();
        assert_eq!(s.events_by_severity.get("Info").copied(), Some(1));
        assert_eq!(s.events_by_severity.get("Error").copied(), Some(1));
        assert_eq!(s.events_by_severity.get("Warning").copied(), Some(1));
    }

    // 17. stats tick range.
    #[test]
    fn test_stats_tick_range() {
        let mut log = StorageEventLog::new(100);
        log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "t0",
            None,
            empty_meta(),
        );
        log.tick();
        log.tick();
        log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "t2",
            None,
            empty_meta(),
        );
        let s = log.stats();
        assert_eq!(s.oldest_tick, Some(0));
        assert_eq!(s.newest_tick, Some(2));
    }

    // 18. stats on empty log.
    #[test]
    fn test_stats_empty() {
        let log = StorageEventLog::new(100);
        let s = log.stats();
        assert_eq!(s.total_events, 0);
        assert!(s.oldest_tick.is_none());
        assert!(s.newest_tick.is_none());
    }

    // 19. Multiple event types.
    #[test]
    fn test_multiple_event_types() {
        let mut log = StorageEventLog::new(100);
        log.log(
            EventType::CompactionStarted,
            EventSeverity::Info,
            "start",
            None,
            empty_meta(),
        );
        log.log(
            EventType::CompactionCompleted,
            EventSeverity::Info,
            "done",
            None,
            empty_meta(),
        );
        assert_eq!(log.events_by_type(EventType::CompactionStarted).len(), 1);
        assert_eq!(log.events_by_type(EventType::CompactionCompleted).len(), 1);
    }

    // 20. Tick tracking.
    #[test]
    fn test_tick_tracking() {
        let mut log = StorageEventLog::new(100);
        log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "t0",
            None,
            empty_meta(),
        );
        log.tick();
        log.tick();
        log.tick();
        log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "t3",
            None,
            empty_meta(),
        );
        let evt = log.get_event(1).expect("should exist");
        assert_eq!(evt.tick, 3);
    }

    // 21. Metadata preserved.
    #[test]
    fn test_metadata_preserved() {
        let mut log = StorageEventLog::new(100);
        let mut meta = HashMap::new();
        meta.insert("size".to_string(), "1024".to_string());
        meta.insert("codec".to_string(), "dag-pb".to_string());
        let id = log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "m",
            Some("QmABC"),
            meta,
        );
        let evt = log.get_event(id).expect("should exist");
        assert_eq!(evt.metadata.get("size").map(|s| s.as_str()), Some("1024"));
        assert_eq!(
            evt.metadata.get("codec").map(|s| s.as_str()),
            Some("dag-pb")
        );
    }

    // 22. event_count after multiple operations.
    #[test]
    fn test_event_count() {
        let mut log = StorageEventLog::new(100);
        assert_eq!(log.event_count(), 0);
        log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "a",
            None,
            empty_meta(),
        );
        assert_eq!(log.event_count(), 1);
        log.log(
            EventType::BlockDeleted,
            EventSeverity::Info,
            "b",
            None,
            empty_meta(),
        );
        assert_eq!(log.event_count(), 2);
    }

    // 23. Critical severity.
    #[test]
    fn test_critical_severity() {
        let mut log = StorageEventLog::new(100);
        log.log(
            EventType::Error,
            EventSeverity::Critical,
            "disk full",
            None,
            empty_meta(),
        );
        let critical = log.events_by_severity(EventSeverity::Critical);
        assert_eq!(critical.len(), 1);
        assert_eq!(critical[0].message, "disk full");
    }

    // 24. events_since with no matches.
    #[test]
    fn test_events_since_no_matches() {
        let mut log = StorageEventLog::new(100);
        log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "old",
            None,
            empty_meta(),
        );
        let since = log.events_since(0);
        assert!(since.is_empty());
    }

    // 25. QuotaExceeded event type.
    #[test]
    fn test_quota_exceeded_type() {
        let mut log = StorageEventLog::new(100);
        log.log(
            EventType::QuotaExceeded,
            EventSeverity::Warning,
            "over limit",
            None,
            empty_meta(),
        );
        assert_eq!(log.count_by_type(EventType::QuotaExceeded), 1);
        let by_type = log.events_by_type(EventType::QuotaExceeded);
        assert_eq!(by_type.len(), 1);
    }

    // 26. block_cid None.
    #[test]
    fn test_block_cid_none() {
        let mut log = StorageEventLog::new(100);
        let id = log.log(
            EventType::CompactionStarted,
            EventSeverity::Info,
            "compact",
            None,
            empty_meta(),
        );
        let evt = log.get_event(id).expect("should exist");
        assert!(evt.block_cid.is_none());
    }

    // 27. Max events = 1.
    #[test]
    fn test_max_events_one() {
        let mut log = StorageEventLog::new(1);
        log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "first",
            None,
            empty_meta(),
        );
        log.log(
            EventType::BlockDeleted,
            EventSeverity::Info,
            "second",
            None,
            empty_meta(),
        );
        assert_eq!(log.event_count(), 1);
        assert!(log.get_event(0).is_none());
        assert!(log.get_event(1).is_some());
    }

    // 28. Tick does not affect existing events.
    #[test]
    fn test_tick_immutable_on_existing() {
        let mut log = StorageEventLog::new(100);
        log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "before",
            None,
            empty_meta(),
        );
        log.tick();
        let evt = log.get_event(0).expect("should exist");
        assert_eq!(evt.tick, 0);
    }

    // 29. recent(0) returns empty.
    #[test]
    fn test_recent_zero() {
        let mut log = StorageEventLog::new(100);
        log.log(
            EventType::BlockAdded,
            EventSeverity::Info,
            "a",
            None,
            empty_meta(),
        );
        assert!(log.recent(0).is_empty());
    }

    // 30. Combined filter: type then severity check.
    #[test]
    fn test_combined_filter() {
        let mut log = StorageEventLog::new(100);
        log.log(
            EventType::Error,
            EventSeverity::Error,
            "e1",
            None,
            empty_meta(),
        );
        log.log(
            EventType::Error,
            EventSeverity::Critical,
            "e2",
            None,
            empty_meta(),
        );
        log.log(
            EventType::BlockAdded,
            EventSeverity::Error,
            "e3",
            None,
            empty_meta(),
        );
        let errors_of_type: Vec<_> = log
            .events_by_type(EventType::Error)
            .into_iter()
            .filter(|e| e.severity == EventSeverity::Critical)
            .collect();
        assert_eq!(errors_of_type.len(), 1);
        assert_eq!(errors_of_type[0].message, "e2");
    }
}
