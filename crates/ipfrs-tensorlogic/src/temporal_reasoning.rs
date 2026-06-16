//! Temporal Reasoning Engine — temporal logic for time-based constraints, intervals, and event ordering.
//!
//! Implements Allen's interval algebra (13 relations) for expressive temporal reasoning,
//! providing event storage, constraint checking, and graph-based chain analysis.

use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Core temporal types
// ---------------------------------------------------------------------------

/// Unix timestamp in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TimePoint {
    pub t: i64,
}

impl TimePoint {
    /// Create a new `TimePoint` from a millisecond timestamp.
    pub fn new(t: i64) -> Self {
        Self { t }
    }
}

impl From<i64> for TimePoint {
    fn from(t: i64) -> Self {
        Self { t }
    }
}

// ---------------------------------------------------------------------------
// TimeInterval
// ---------------------------------------------------------------------------

/// A closed time interval `[start, end]` with the invariant `start ≤ end`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeInterval {
    pub start: TimePoint,
    pub end: TimePoint,
}

impl TimeInterval {
    /// Construct a new `TimeInterval`, returning an error when `start > end`.
    pub fn new(start: TimePoint, end: TimePoint) -> Result<Self, TemporalError> {
        if start > end {
            return Err(TemporalError::InvalidInterval {
                start: start.t,
                end: end.t,
            });
        }
        Ok(Self { start, end })
    }

    /// Duration in milliseconds.  Always `≥ 0`.
    pub fn duration_ms(&self) -> u64 {
        (self.end.t - self.start.t) as u64
    }

    /// Returns `true` if the two intervals share at least one point (Allen's
    /// *overlaps*, *starts*, *during*, *finishes*, *equals*, and their inverses
    /// all count as overlapping in the common sense).  Specifically we test
    /// `self.start ≤ other.end && other.start ≤ self.end`.
    pub fn overlaps(&self, other: &TimeInterval) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    /// Returns `true` if `t` lies within `[start, end]`.
    pub fn contains_point(&self, t: &TimePoint) -> bool {
        self.start <= *t && *t <= self.end
    }

    /// Allen's *precedes*: `self.end < other.start`.
    pub fn before(&self, other: &TimeInterval) -> bool {
        self.end < other.start
    }

    /// Allen's *preceded-by*: `other.end < self.start`.
    pub fn after(&self, other: &TimeInterval) -> bool {
        other.end < self.start
    }

    /// Allen's *meets*: `self.end == other.start`.
    pub fn meets(&self, other: &TimeInterval) -> bool {
        self.end == other.start
    }

    /// Allen's *during* (proper): `self.start >= other.start && self.end <= other.end`.
    pub fn during(&self, other: &TimeInterval) -> bool {
        self.start >= other.start && self.end <= other.end
    }
}

// ---------------------------------------------------------------------------
// Allen's interval algebra — full 13-relation classification
// ---------------------------------------------------------------------------

/// The 13 relations of Allen's interval algebra.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllenRelation {
    /// `a` ends strictly before `b` starts.
    Precedes,
    /// `a` ends exactly when `b` starts.
    Meets,
    /// `a` starts before `b` and ends inside `b`.
    Overlaps,
    /// `a` starts before `b` and ends exactly when `b` ends.
    FinishedBy,
    /// `a` starts before `b` starts and ends after `b` ends.
    Contains,
    /// `a` and `b` start at the same point; `a` ends before `b`.
    Starts,
    /// `a` and `b` are identical.
    Equals,
    /// `a` and `b` start at the same point; `b` ends before `a`.
    StartedBy,
    /// `a` starts after `b` starts and ends before `b` ends (proper subset).
    During,
    /// `a` starts after `b` starts and ends exactly when `b` ends.
    Finishes,
    /// `b` starts before `a` and ends inside `a`.
    OverlappedBy,
    /// `a` starts exactly when `b` ends.
    MetBy,
    /// `a` starts strictly after `b` ends.
    PrecededBy,
}

impl AllenRelation {
    /// Classify two intervals into exactly one Allen relation.
    pub fn classify(a: &TimeInterval, b: &TimeInterval) -> AllenRelation {
        let as_ = a.start.t;
        let ae = a.end.t;
        let bs = b.start.t;
        let be = b.end.t;

        if ae < bs {
            AllenRelation::Precedes
        } else if ae == bs {
            AllenRelation::Meets
        } else if as_ < bs && ae < be {
            AllenRelation::Overlaps
        } else if as_ < bs && ae == be {
            AllenRelation::FinishedBy
        } else if as_ < bs && ae > be {
            AllenRelation::Contains
        } else if as_ == bs && ae < be {
            AllenRelation::Starts
        } else if as_ == bs && ae == be {
            AllenRelation::Equals
        } else if as_ == bs && ae > be {
            AllenRelation::StartedBy
        } else if as_ > bs && ae < be {
            AllenRelation::During
        } else if as_ > bs && ae == be {
            AllenRelation::Finishes
        } else if as_ > bs && ae > be && as_ < be {
            AllenRelation::OverlappedBy
        } else if as_ == be {
            AllenRelation::MetBy
        } else {
            // as_ > be
            AllenRelation::PrecededBy
        }
    }

    /// Returns `true` when the two intervals share any common point under this
    /// relation (i.e. the relation is *not* a strict precede or strict follow).
    pub fn is_overlapping_kind(self) -> bool {
        !matches!(
            self,
            AllenRelation::Precedes
                | AllenRelation::PrecededBy
                | AllenRelation::Meets
                | AllenRelation::MetBy
        )
    }
}

// ---------------------------------------------------------------------------
// TemporalEvent
// ---------------------------------------------------------------------------

/// A named event occupying a time interval with optional tags and payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalEvent {
    pub id: String,
    pub interval: TimeInterval,
    pub tags: Vec<String>,
    pub payload: String,
}

impl TemporalEvent {
    /// Convenience constructor.
    pub fn new(
        id: impl Into<String>,
        interval: TimeInterval,
        tags: Vec<String>,
        payload: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            interval,
            tags,
            payload: payload.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// TemporalConstraint
// ---------------------------------------------------------------------------

/// Constraints that can be enforced across events.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemporalConstraint {
    /// Event `a` must end before event `b` starts.
    Before { a: String, b: String },
    /// Event `a` must start after event `b` ends.
    After { a: String, b: String },
    /// Events `a` and `b` must share at least one point in time.
    Overlapping { a: String, b: String },
    /// Event `inner` must be entirely contained within event `outer`.
    During { inner: String, outer: String },
    /// Event `event` must overlap with the given fixed `window`.
    Within { event: String, window: TimeInterval },
}

// ---------------------------------------------------------------------------
// ConstraintViolation
// ---------------------------------------------------------------------------

/// Describes a single constraint violation.
#[derive(Debug, Clone)]
pub struct ConstraintViolation {
    /// Human-readable description of the violated constraint.
    pub constraint: String,
    /// First event involved in the violation.
    pub event_a: String,
    /// Second event (may equal `event_a` for unary constraints).
    pub event_b: String,
    /// The Allen relation that was actually observed.
    pub relation: AllenRelation,
}

// ---------------------------------------------------------------------------
// TemporalError
// ---------------------------------------------------------------------------

/// Errors produced by the temporal reasoning engine.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TemporalError {
    #[error("event not found: {0}")]
    EventNotFound(String),

    #[error("duplicate event id: {0}")]
    DuplicateEventId(String),

    #[error("maximum event capacity reached")]
    MaxEventsReached,

    #[error("invalid interval: start ({start}) > end ({end})")]
    InvalidInterval { start: i64, end: i64 },
}

// ---------------------------------------------------------------------------
// TemporalStats
// ---------------------------------------------------------------------------

/// Aggregate statistics about the engine's current state.
#[derive(Debug, Clone)]
pub struct TemporalStats {
    pub event_count: usize,
    pub constraint_count: usize,
    /// Span in ms from the earliest event start to the latest event end.
    /// Zero if there are no events.
    pub timeline_span_ms: u64,
    /// Mean event duration in ms.  Zero if there are no events.
    pub avg_event_duration_ms: f64,
}

// ---------------------------------------------------------------------------
// TemporalReasoningEngine
// ---------------------------------------------------------------------------

/// A production-grade temporal logic reasoning system.
///
/// Stores a collection of [`TemporalEvent`]s and a set of [`TemporalConstraint`]s,
/// and provides rich queries over them including Allen's interval algebra
/// classification, windowed search, tag-based lookup, chain analysis, and
/// full constraint checking.
#[derive(Debug)]
pub struct TemporalReasoningEngine {
    events: HashMap<String, TemporalEvent>,
    constraints: Vec<TemporalConstraint>,
    max_events: usize,
}

impl TemporalReasoningEngine {
    /// Create a new engine with the given event capacity.
    pub fn new(max_events: usize) -> Self {
        Self {
            events: HashMap::new(),
            constraints: Vec::new(),
            max_events,
        }
    }

    // -----------------------------------------------------------------------
    // Event management
    // -----------------------------------------------------------------------

    /// Add an event.  Returns `Err` if the id already exists or if the engine
    /// is at capacity.
    pub fn add_event(&mut self, event: TemporalEvent) -> Result<(), TemporalError> {
        if self.events.contains_key(&event.id) {
            return Err(TemporalError::DuplicateEventId(event.id.clone()));
        }
        if self.events.len() >= self.max_events {
            return Err(TemporalError::MaxEventsReached);
        }
        self.events.insert(event.id.clone(), event);
        Ok(())
    }

    /// Remove an event by id.  Returns `true` if it existed.
    pub fn remove_event(&mut self, id: &str) -> bool {
        self.events.remove(id).is_some()
    }

    /// Retrieve a reference to an event by id.
    pub fn get_event(&self, id: &str) -> Option<&TemporalEvent> {
        self.events.get(id)
    }

    // -----------------------------------------------------------------------
    // Temporal queries
    // -----------------------------------------------------------------------

    /// Return all events whose interval overlaps `window`, sorted by start time
    /// (then end time as tiebreaker).
    pub fn events_in_window(&self, window: &TimeInterval) -> Vec<&TemporalEvent> {
        let mut result: Vec<&TemporalEvent> = self
            .events
            .values()
            .filter(|e| e.interval.overlaps(window))
            .collect();
        result.sort_by(|a, b| {
            a.interval
                .start
                .cmp(&b.interval.start)
                .then_with(|| a.interval.end.cmp(&b.interval.end))
        });
        result
    }

    /// Return all events that carry `tag`, sorted by start time.
    pub fn events_with_tag(&self, tag: &str) -> Vec<&TemporalEvent> {
        let mut result: Vec<&TemporalEvent> = self
            .events
            .values()
            .filter(|e| e.tags.iter().any(|t| t == tag))
            .collect();
        result.sort_by(|a, b| {
            a.interval
                .start
                .cmp(&b.interval.start)
                .then_with(|| a.interval.end.cmp(&b.interval.end))
        });
        result
    }

    /// Return all events sorted chronologically (by start, then end).
    pub fn timeline(&self) -> Vec<&TemporalEvent> {
        let mut events: Vec<&TemporalEvent> = self.events.values().collect();
        events.sort_by(|a, b| {
            a.interval
                .start
                .cmp(&b.interval.start)
                .then_with(|| a.interval.end.cmp(&b.interval.end))
        });
        events
    }

    /// Return all events that overlap with the event identified by `id`,
    /// excluding the event itself.  Returns an empty `Vec` if `id` not found.
    pub fn concurrent_events(&self, id: &str) -> Vec<&TemporalEvent> {
        let target = match self.events.get(id) {
            Some(e) => e,
            None => return Vec::new(),
        };
        let mut result: Vec<&TemporalEvent> = self
            .events
            .values()
            .filter(|e| e.id != id && e.interval.overlaps(&target.interval))
            .collect();
        result.sort_by(|a, b| {
            a.interval
                .start
                .cmp(&b.interval.start)
                .then_with(|| a.interval.end.cmp(&b.interval.end))
        });
        result
    }

    // -----------------------------------------------------------------------
    // Allen's relation
    // -----------------------------------------------------------------------

    /// Classify the Allen relation between two events.  Returns `None` if
    /// either event is not found.
    pub fn allen_relation(&self, a: &str, b: &str) -> Option<AllenRelation> {
        let ea = self.events.get(a)?;
        let eb = self.events.get(b)?;
        Some(AllenRelation::classify(&ea.interval, &eb.interval))
    }

    // -----------------------------------------------------------------------
    // Constraints
    // -----------------------------------------------------------------------

    /// Register a constraint.  Constraints are not validated on insertion;
    /// call [`check_constraints`](TemporalReasoningEngine::check_constraints)
    /// to evaluate them.
    pub fn add_constraint(&mut self, c: TemporalConstraint) {
        self.constraints.push(c);
    }

    /// Evaluate all registered constraints and return a list of violations.
    pub fn check_constraints(&self) -> Vec<ConstraintViolation> {
        let mut violations = Vec::new();
        for constraint in &self.constraints {
            match constraint {
                TemporalConstraint::Before { a, b } => {
                    self.check_before(a, b, &mut violations);
                }
                TemporalConstraint::After { a, b } => {
                    self.check_after(a, b, &mut violations);
                }
                TemporalConstraint::Overlapping { a, b } => {
                    self.check_overlapping(a, b, &mut violations);
                }
                TemporalConstraint::During { inner, outer } => {
                    self.check_during(inner, outer, &mut violations);
                }
                TemporalConstraint::Within { event, window } => {
                    self.check_within(event, window, &mut violations);
                }
            }
        }
        violations
    }

    // -----------------------------------------------------------------------
    // Event chains (connected components via overlap)
    // -----------------------------------------------------------------------

    /// Compute connected components of the overlap graph.
    ///
    /// Two events are in the same component if they overlap (directly or
    /// transitively).  Returns chains sorted by the start time of their
    /// first event; within each chain events are also sorted by start time.
    pub fn event_chains(&self) -> Vec<Vec<&TemporalEvent>> {
        // Collect ids in a stable order for BFS.
        let ids: Vec<&str> = self.events.keys().map(String::as_str).collect();
        let n = ids.len();
        if n == 0 {
            return Vec::new();
        }

        // Build an id → index map.
        let idx: HashMap<&str, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        // Adjacency list — symmetric overlap.
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..n {
            let ei = self.events.get(ids[i]).expect("known key");
            for j in (i + 1)..n {
                let ej = self.events.get(ids[j]).expect("known key");
                if ei.interval.overlaps(&ej.interval) {
                    adj[i].push(j);
                    adj[j].push(i);
                }
            }
        }

        // BFS to find components.
        let mut visited = vec![false; n];
        let mut components: Vec<Vec<usize>> = Vec::new();

        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut component = Vec::new();
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(start);
            visited[start] = true;
            while let Some(cur) = queue.pop_front() {
                component.push(cur);
                for &nb in &adj[cur] {
                    if !visited[nb] {
                        visited[nb] = true;
                        queue.push_back(nb);
                    }
                }
            }
            components.push(component);
        }

        // Build result: sort within each component by start time, then sort
        // components by the start time of their first event.
        let sort_key = |idx_val: usize| -> (i64, i64) {
            let e = self.events.get(ids[idx_val]).expect("known key");
            (e.interval.start.t, e.interval.end.t)
        };

        let mut chains: Vec<Vec<&TemporalEvent>> = components
            .into_iter()
            .map(|mut comp| {
                comp.sort_by_key(|&i| sort_key(i));
                comp.iter()
                    .map(|&i| self.events.get(ids[i]).expect("known key"))
                    .collect()
            })
            .collect();

        chains.sort_by(|a, b| {
            let ka = a
                .first()
                .map(|e| (e.interval.start.t, e.interval.end.t))
                .unwrap_or((i64::MAX, i64::MAX));
            let kb = b
                .first()
                .map(|e| (e.interval.start.t, e.interval.end.t))
                .unwrap_or((i64::MAX, i64::MAX));
            ka.cmp(&kb)
        });

        // Drop the idx map — it was only used inside idx for the adj build.
        let _ = idx;

        chains
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Compute aggregate statistics about the current engine state.
    pub fn stats(&self) -> TemporalStats {
        let event_count = self.events.len();
        let constraint_count = self.constraints.len();

        if event_count == 0 {
            return TemporalStats {
                event_count: 0,
                constraint_count,
                timeline_span_ms: 0,
                avg_event_duration_ms: 0.0,
            };
        }

        let mut earliest = i64::MAX;
        let mut latest = i64::MIN;
        let mut total_duration: u64 = 0;

        for e in self.events.values() {
            if e.interval.start.t < earliest {
                earliest = e.interval.start.t;
            }
            if e.interval.end.t > latest {
                latest = e.interval.end.t;
            }
            total_duration += e.interval.duration_ms();
        }

        let timeline_span_ms = if latest >= earliest {
            (latest - earliest) as u64
        } else {
            0
        };

        let avg_event_duration_ms = total_duration as f64 / event_count as f64;

        TemporalStats {
            event_count,
            constraint_count,
            timeline_span_ms,
            avg_event_duration_ms,
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers for constraint checking
    // -----------------------------------------------------------------------

    fn check_before(&self, a: &str, b: &str, violations: &mut Vec<ConstraintViolation>) {
        let (ea, eb) = match (self.events.get(a), self.events.get(b)) {
            (Some(x), Some(y)) => (x, y),
            _ => return,
        };
        let rel = AllenRelation::classify(&ea.interval, &eb.interval);
        // Satisfied only by Precedes (strict before).
        if rel != AllenRelation::Precedes {
            violations.push(ConstraintViolation {
                constraint: format!("Before({a}, {b})"),
                event_a: a.to_owned(),
                event_b: b.to_owned(),
                relation: rel,
            });
        }
    }

    fn check_after(&self, a: &str, b: &str, violations: &mut Vec<ConstraintViolation>) {
        let (ea, eb) = match (self.events.get(a), self.events.get(b)) {
            (Some(x), Some(y)) => (x, y),
            _ => return,
        };
        let rel = AllenRelation::classify(&ea.interval, &eb.interval);
        // Satisfied only by PrecededBy.
        if rel != AllenRelation::PrecededBy {
            violations.push(ConstraintViolation {
                constraint: format!("After({a}, {b})"),
                event_a: a.to_owned(),
                event_b: b.to_owned(),
                relation: rel,
            });
        }
    }

    fn check_overlapping(&self, a: &str, b: &str, violations: &mut Vec<ConstraintViolation>) {
        let (ea, eb) = match (self.events.get(a), self.events.get(b)) {
            (Some(x), Some(y)) => (x, y),
            _ => return,
        };
        let rel = AllenRelation::classify(&ea.interval, &eb.interval);
        if !ea.interval.overlaps(&eb.interval) {
            violations.push(ConstraintViolation {
                constraint: format!("Overlapping({a}, {b})"),
                event_a: a.to_owned(),
                event_b: b.to_owned(),
                relation: rel,
            });
        }
    }

    fn check_during(&self, inner: &str, outer: &str, violations: &mut Vec<ConstraintViolation>) {
        let (ei, eo) = match (self.events.get(inner), self.events.get(outer)) {
            (Some(x), Some(y)) => (x, y),
            _ => return,
        };
        let rel = AllenRelation::classify(&ei.interval, &eo.interval);
        // Satisfied by During, Starts, Finishes, Equals (inner fully inside outer).
        let satisfied = matches!(
            rel,
            AllenRelation::During
                | AllenRelation::Starts
                | AllenRelation::Finishes
                | AllenRelation::Equals
        );
        if !satisfied {
            violations.push(ConstraintViolation {
                constraint: format!("During(inner={inner}, outer={outer})"),
                event_a: inner.to_owned(),
                event_b: outer.to_owned(),
                relation: rel,
            });
        }
    }

    fn check_within(
        &self,
        event_id: &str,
        window: &TimeInterval,
        violations: &mut Vec<ConstraintViolation>,
    ) {
        let ev = match self.events.get(event_id) {
            Some(e) => e,
            None => return,
        };
        // Synthesise a virtual "window" event for Allen classification.
        let window_event = TemporalEvent::new("__window__", window.clone(), Vec::new(), "");
        let rel = AllenRelation::classify(&ev.interval, &window_event.interval);
        if !ev.interval.overlaps(window) {
            violations.push(ConstraintViolation {
                constraint: format!("Within({event_id}, window)"),
                event_a: event_id.to_owned(),
                event_b: "__window__".to_owned(),
                relation: rel,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::temporal_reasoning::{
        AllenRelation, ConstraintViolation, TemporalConstraint, TemporalError, TemporalEvent,
        TemporalReasoningEngine, TimeInterval, TimePoint,
    };

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn tp(t: i64) -> TimePoint {
        TimePoint::new(t)
    }

    fn iv(s: i64, e: i64) -> TimeInterval {
        TimeInterval::new(tp(s), tp(e)).expect("valid interval in test helper")
    }

    fn ev(id: &str, s: i64, e: i64) -> TemporalEvent {
        TemporalEvent::new(id, iv(s, e), Vec::new(), "")
    }

    fn ev_tagged(id: &str, s: i64, e: i64, tags: &[&str]) -> TemporalEvent {
        TemporalEvent::new(
            id,
            iv(s, e),
            tags.iter().map(|t| t.to_string()).collect(),
            "",
        )
    }

    // ------------------------------------------------------------------
    // TimePoint
    // ------------------------------------------------------------------

    #[test]
    fn test_time_point_ordering() {
        assert!(tp(10) < tp(20));
        assert!(tp(20) > tp(10));
        assert_eq!(tp(5), tp(5));
    }

    #[test]
    fn test_time_point_from_i64() {
        let p: TimePoint = 42_i64.into();
        assert_eq!(p.t, 42);
    }

    // ------------------------------------------------------------------
    // TimeInterval construction & invariant
    // ------------------------------------------------------------------

    #[test]
    fn test_interval_valid() {
        let i = iv(0, 100);
        assert_eq!(i.duration_ms(), 100);
    }

    #[test]
    fn test_interval_invalid_returns_error() {
        let result = TimeInterval::new(tp(100), tp(0));
        assert_eq!(
            result,
            Err(TemporalError::InvalidInterval { start: 100, end: 0 })
        );
    }

    #[test]
    fn test_interval_zero_duration() {
        let i = iv(50, 50);
        assert_eq!(i.duration_ms(), 0);
    }

    // ------------------------------------------------------------------
    // TimeInterval predicates
    // ------------------------------------------------------------------

    #[test]
    fn test_contains_point() {
        let i = iv(10, 20);
        assert!(i.contains_point(&tp(10)));
        assert!(i.contains_point(&tp(15)));
        assert!(i.contains_point(&tp(20)));
        assert!(!i.contains_point(&tp(9)));
        assert!(!i.contains_point(&tp(21)));
    }

    #[test]
    fn test_before_after() {
        let a = iv(0, 10);
        let b = iv(20, 30);
        assert!(a.before(&b));
        assert!(b.after(&a));
        assert!(!b.before(&a));
        assert!(!a.after(&b));
    }

    #[test]
    fn test_meets() {
        let a = iv(0, 10);
        let b = iv(10, 20);
        assert!(a.meets(&b));
        assert!(!b.meets(&a));
    }

    #[test]
    fn test_during() {
        let outer = iv(0, 100);
        let inner = iv(20, 80);
        assert!(inner.during(&outer));
        assert!(!outer.during(&inner));
    }

    #[test]
    fn test_overlaps_various() {
        assert!(iv(0, 10).overlaps(&iv(5, 15)));
        assert!(iv(5, 15).overlaps(&iv(0, 10)));
        assert!(iv(0, 10).overlaps(&iv(10, 20))); // touching = overlapping in common sense
        assert!(!iv(0, 10).overlaps(&iv(11, 20)));
    }

    // ------------------------------------------------------------------
    // Allen's interval algebra
    // ------------------------------------------------------------------

    #[test]
    fn test_allen_precedes() {
        let rel = AllenRelation::classify(&iv(0, 5), &iv(10, 20));
        assert_eq!(rel, AllenRelation::Precedes);
    }

    #[test]
    fn test_allen_preceded_by() {
        let rel = AllenRelation::classify(&iv(10, 20), &iv(0, 5));
        assert_eq!(rel, AllenRelation::PrecededBy);
    }

    #[test]
    fn test_allen_meets() {
        let rel = AllenRelation::classify(&iv(0, 10), &iv(10, 20));
        assert_eq!(rel, AllenRelation::Meets);
    }

    #[test]
    fn test_allen_met_by() {
        let rel = AllenRelation::classify(&iv(10, 20), &iv(0, 10));
        assert_eq!(rel, AllenRelation::MetBy);
    }

    #[test]
    fn test_allen_overlaps() {
        let rel = AllenRelation::classify(&iv(0, 15), &iv(10, 25));
        assert_eq!(rel, AllenRelation::Overlaps);
    }

    #[test]
    fn test_allen_overlapped_by() {
        let rel = AllenRelation::classify(&iv(10, 25), &iv(0, 15));
        assert_eq!(rel, AllenRelation::OverlappedBy);
    }

    #[test]
    fn test_allen_starts() {
        let rel = AllenRelation::classify(&iv(0, 10), &iv(0, 20));
        assert_eq!(rel, AllenRelation::Starts);
    }

    #[test]
    fn test_allen_started_by() {
        let rel = AllenRelation::classify(&iv(0, 20), &iv(0, 10));
        assert_eq!(rel, AllenRelation::StartedBy);
    }

    #[test]
    fn test_allen_finishes() {
        let rel = AllenRelation::classify(&iv(10, 20), &iv(0, 20));
        assert_eq!(rel, AllenRelation::Finishes);
    }

    #[test]
    fn test_allen_finished_by() {
        let rel = AllenRelation::classify(&iv(0, 20), &iv(10, 20));
        assert_eq!(rel, AllenRelation::FinishedBy);
    }

    #[test]
    fn test_allen_during() {
        let rel = AllenRelation::classify(&iv(10, 15), &iv(0, 20));
        assert_eq!(rel, AllenRelation::During);
    }

    #[test]
    fn test_allen_contains() {
        let rel = AllenRelation::classify(&iv(0, 20), &iv(10, 15));
        assert_eq!(rel, AllenRelation::Contains);
    }

    #[test]
    fn test_allen_equals() {
        let rel = AllenRelation::classify(&iv(5, 15), &iv(5, 15));
        assert_eq!(rel, AllenRelation::Equals);
    }

    #[test]
    fn test_allen_overlapping_kind_filter() {
        // Overlapping relations
        for rel in [
            AllenRelation::Overlaps,
            AllenRelation::FinishedBy,
            AllenRelation::Contains,
            AllenRelation::Starts,
            AllenRelation::Equals,
            AllenRelation::StartedBy,
            AllenRelation::During,
            AllenRelation::Finishes,
            AllenRelation::OverlappedBy,
        ] {
            assert!(rel.is_overlapping_kind(), "{rel:?} should be overlapping");
        }
        // Non-overlapping relations
        for rel in [
            AllenRelation::Precedes,
            AllenRelation::PrecededBy,
            AllenRelation::Meets,
            AllenRelation::MetBy,
        ] {
            assert!(
                !rel.is_overlapping_kind(),
                "{rel:?} should not be overlapping"
            );
        }
    }

    // ------------------------------------------------------------------
    // Engine: basic event CRUD
    // ------------------------------------------------------------------

    #[test]
    fn test_add_and_get_event() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev("e1", 0, 100))
            .expect("test: should succeed");
        let e = engine.get_event("e1").expect("test: should succeed");
        assert_eq!(e.id, "e1");
    }

    #[test]
    fn test_duplicate_event_error() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev("e1", 0, 100))
            .expect("test: should succeed");
        let res = engine.add_event(ev("e1", 50, 150));
        assert_eq!(res, Err(TemporalError::DuplicateEventId("e1".to_owned())));
    }

    #[test]
    fn test_max_events_error() {
        let mut engine = TemporalReasoningEngine::new(2);
        engine
            .add_event(ev("e1", 0, 10))
            .expect("test: should succeed");
        engine
            .add_event(ev("e2", 20, 30))
            .expect("test: should succeed");
        let res = engine.add_event(ev("e3", 40, 50));
        assert_eq!(res, Err(TemporalError::MaxEventsReached));
    }

    #[test]
    fn test_remove_event() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev("e1", 0, 100))
            .expect("test: should succeed");
        assert!(engine.remove_event("e1"));
        assert!(!engine.remove_event("e1"));
        assert!(engine.get_event("e1").is_none());
    }

    // ------------------------------------------------------------------
    // Engine: queries
    // ------------------------------------------------------------------

    #[test]
    fn test_events_in_window() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev("a", 0, 10))
            .expect("test: should succeed"); // before window
        engine
            .add_event(ev("b", 5, 15))
            .expect("test: should succeed"); // overlaps
        engine
            .add_event(ev("c", 12, 20))
            .expect("test: should succeed"); // inside window
        engine
            .add_event(ev("d", 19, 30))
            .expect("test: should succeed"); // overlaps end
        engine
            .add_event(ev("e", 25, 35))
            .expect("test: should succeed"); // after window
        let window = iv(11, 22);
        let result = engine.events_in_window(&window);
        let ids: Vec<&str> = result.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"b"), "b should overlap window [11,22]");
        assert!(ids.contains(&"c"), "c should be inside window");
        assert!(ids.contains(&"d"), "d should overlap window end");
        assert!(!ids.contains(&"a"), "a should not overlap window");
        assert!(!ids.contains(&"e"), "e should not overlap window");
        // Sorted by start
        for i in 1..result.len() {
            assert!(result[i - 1].interval.start <= result[i].interval.start);
        }
    }

    #[test]
    fn test_events_with_tag() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev_tagged("a", 0, 10, &["foo", "bar"]))
            .expect("test: should succeed");
        engine
            .add_event(ev_tagged("b", 5, 15, &["bar"]))
            .expect("test: should succeed");
        engine
            .add_event(ev("c", 20, 30))
            .expect("test: should succeed");

        let foo = engine.events_with_tag("foo");
        assert_eq!(foo.len(), 1);
        assert_eq!(foo[0].id, "a");

        let bar = engine.events_with_tag("bar");
        assert_eq!(bar.len(), 2);
        assert_eq!(bar[0].id, "a");
        assert_eq!(bar[1].id, "b");
    }

    #[test]
    fn test_timeline_sorted() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev("c", 50, 70))
            .expect("test: should succeed");
        engine
            .add_event(ev("a", 0, 20))
            .expect("test: should succeed");
        engine
            .add_event(ev("b", 10, 40))
            .expect("test: should succeed");
        let tl = engine.timeline();
        assert_eq!(tl[0].id, "a");
        assert_eq!(tl[1].id, "b");
        assert_eq!(tl[2].id, "c");
    }

    #[test]
    fn test_concurrent_events() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev("a", 0, 50))
            .expect("test: should succeed");
        engine
            .add_event(ev("b", 30, 80))
            .expect("test: should succeed");
        engine
            .add_event(ev("c", 60, 100))
            .expect("test: should succeed");
        engine
            .add_event(ev("d", 200, 300))
            .expect("test: should succeed");

        let conc = engine.concurrent_events("a");
        let ids: Vec<&str> = conc.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"b"));
        assert!(!ids.contains(&"c"));
        assert!(!ids.contains(&"d"));
        assert!(!ids.contains(&"a")); // self excluded
    }

    #[test]
    fn test_concurrent_events_unknown_id() {
        let engine = TemporalReasoningEngine::new(100);
        assert!(engine.concurrent_events("nonexistent").is_empty());
    }

    #[test]
    fn test_allen_relation_via_engine() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev("a", 0, 10))
            .expect("test: should succeed");
        engine
            .add_event(ev("b", 20, 30))
            .expect("test: should succeed");
        assert_eq!(
            engine.allen_relation("a", "b"),
            Some(AllenRelation::Precedes)
        );
        assert_eq!(engine.allen_relation("a", "missing"), None);
    }

    // ------------------------------------------------------------------
    // Constraints
    // ------------------------------------------------------------------

    #[test]
    fn test_constraint_before_satisfied() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev("a", 0, 10))
            .expect("test: should succeed");
        engine
            .add_event(ev("b", 20, 30))
            .expect("test: should succeed");
        engine.add_constraint(TemporalConstraint::Before {
            a: "a".into(),
            b: "b".into(),
        });
        assert!(engine.check_constraints().is_empty());
    }

    #[test]
    fn test_constraint_before_violated() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev("a", 0, 20))
            .expect("test: should succeed"); // overlaps b
        engine
            .add_event(ev("b", 10, 30))
            .expect("test: should succeed");
        engine.add_constraint(TemporalConstraint::Before {
            a: "a".into(),
            b: "b".into(),
        });
        let violations = engine.check_constraints();
        assert!(!violations.is_empty());
        assert_eq!(violations[0].event_a, "a");
    }

    #[test]
    fn test_constraint_after_satisfied() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev("a", 50, 100))
            .expect("test: should succeed");
        engine
            .add_event(ev("b", 0, 30))
            .expect("test: should succeed");
        engine.add_constraint(TemporalConstraint::After {
            a: "a".into(),
            b: "b".into(),
        });
        assert!(engine.check_constraints().is_empty());
    }

    #[test]
    fn test_constraint_after_violated() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev("a", 0, 20))
            .expect("test: should succeed");
        engine
            .add_event(ev("b", 0, 30))
            .expect("test: should succeed");
        engine.add_constraint(TemporalConstraint::After {
            a: "a".into(),
            b: "b".into(),
        });
        let violations = engine.check_constraints();
        assert!(!violations.is_empty());
    }

    #[test]
    fn test_constraint_overlapping_satisfied() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev("a", 0, 20))
            .expect("test: should succeed");
        engine
            .add_event(ev("b", 10, 30))
            .expect("test: should succeed");
        engine.add_constraint(TemporalConstraint::Overlapping {
            a: "a".into(),
            b: "b".into(),
        });
        assert!(engine.check_constraints().is_empty());
    }

    #[test]
    fn test_constraint_overlapping_violated() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev("a", 0, 5))
            .expect("test: should succeed");
        engine
            .add_event(ev("b", 10, 20))
            .expect("test: should succeed");
        engine.add_constraint(TemporalConstraint::Overlapping {
            a: "a".into(),
            b: "b".into(),
        });
        let violations = engine.check_constraints();
        assert!(!violations.is_empty());
    }

    #[test]
    fn test_constraint_during_satisfied() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev("inner", 10, 20))
            .expect("test: should succeed");
        engine
            .add_event(ev("outer", 0, 50))
            .expect("test: should succeed");
        engine.add_constraint(TemporalConstraint::During {
            inner: "inner".into(),
            outer: "outer".into(),
        });
        assert!(engine.check_constraints().is_empty());
    }

    #[test]
    fn test_constraint_during_violated() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev("inner", 0, 100))
            .expect("test: should succeed");
        engine
            .add_event(ev("outer", 10, 50))
            .expect("test: should succeed");
        engine.add_constraint(TemporalConstraint::During {
            inner: "inner".into(),
            outer: "outer".into(),
        });
        let violations = engine.check_constraints();
        assert!(!violations.is_empty());
    }

    #[test]
    fn test_constraint_within_satisfied() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev("e", 10, 20))
            .expect("test: should succeed");
        let window = iv(0, 50);
        engine.add_constraint(TemporalConstraint::Within {
            event: "e".into(),
            window,
        });
        assert!(engine.check_constraints().is_empty());
    }

    #[test]
    fn test_constraint_within_violated() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev("e", 100, 200))
            .expect("test: should succeed");
        let window = iv(0, 50);
        engine.add_constraint(TemporalConstraint::Within {
            event: "e".into(),
            window,
        });
        let violations = engine.check_constraints();
        assert!(!violations.is_empty());
        assert_eq!(violations[0].event_a, "e");
    }

    #[test]
    fn test_multiple_constraints_mixed() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev("a", 0, 10))
            .expect("test: should succeed");
        engine
            .add_event(ev("b", 20, 30))
            .expect("test: should succeed");
        engine
            .add_event(ev("c", 5, 15))
            .expect("test: should succeed");
        // a before b — satisfied
        engine.add_constraint(TemporalConstraint::Before {
            a: "a".into(),
            b: "b".into(),
        });
        // a before c — violated (a overlaps c)
        engine.add_constraint(TemporalConstraint::Before {
            a: "a".into(),
            b: "c".into(),
        });
        let violations = engine.check_constraints();
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].event_b, "c");
    }

    // ------------------------------------------------------------------
    // Event chains
    // ------------------------------------------------------------------

    #[test]
    fn test_event_chains_empty() {
        let engine = TemporalReasoningEngine::new(100);
        assert!(engine.event_chains().is_empty());
    }

    #[test]
    fn test_event_chains_single() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev("a", 0, 10))
            .expect("test: should succeed");
        let chains = engine.event_chains();
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].len(), 1);
    }

    #[test]
    fn test_event_chains_two_isolated() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev("a", 0, 10))
            .expect("test: should succeed");
        engine
            .add_event(ev("b", 100, 200))
            .expect("test: should succeed");
        let chains = engine.event_chains();
        assert_eq!(chains.len(), 2);
    }

    #[test]
    fn test_event_chains_transitive() {
        let mut engine = TemporalReasoningEngine::new(100);
        // a overlaps b; b overlaps c; a does NOT directly overlap c
        engine
            .add_event(ev("a", 0, 20))
            .expect("test: should succeed");
        engine
            .add_event(ev("b", 15, 35))
            .expect("test: should succeed");
        engine
            .add_event(ev("c", 30, 50))
            .expect("test: should succeed");
        engine
            .add_event(ev("d", 200, 300))
            .expect("test: should succeed"); // isolated
        let chains = engine.event_chains();
        // {a,b,c} is one component; {d} is another
        assert_eq!(chains.len(), 2);
        let big = chains
            .iter()
            .find(|ch| ch.len() == 3)
            .expect("test: should succeed");
        let ids: Vec<&str> = big.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"a") && ids.contains(&"b") && ids.contains(&"c"));
    }

    // ------------------------------------------------------------------
    // Statistics
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_empty() {
        let engine = TemporalReasoningEngine::new(100);
        let s = engine.stats();
        assert_eq!(s.event_count, 0);
        assert_eq!(s.constraint_count, 0);
        assert_eq!(s.timeline_span_ms, 0);
        assert_eq!(s.avg_event_duration_ms, 0.0);
    }

    #[test]
    fn test_stats_with_events() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev("a", 0, 100))
            .expect("test: should succeed");
        engine
            .add_event(ev("b", 50, 200))
            .expect("test: should succeed");
        engine.add_constraint(TemporalConstraint::Overlapping {
            a: "a".into(),
            b: "b".into(),
        });
        let s = engine.stats();
        assert_eq!(s.event_count, 2);
        assert_eq!(s.constraint_count, 1);
        assert_eq!(s.timeline_span_ms, 200); // 0..200
                                             // avg = (100 + 150) / 2 = 125
        assert!((s.avg_event_duration_ms - 125.0).abs() < f64::EPSILON);
    }

    // ------------------------------------------------------------------
    // ConstraintViolation fields
    // ------------------------------------------------------------------

    #[test]
    fn test_violation_contains_relation() {
        let mut engine = TemporalReasoningEngine::new(100);
        engine
            .add_event(ev("x", 0, 50))
            .expect("test: should succeed");
        engine
            .add_event(ev("y", 40, 80))
            .expect("test: should succeed");
        engine.add_constraint(TemporalConstraint::Before {
            a: "x".into(),
            b: "y".into(),
        });
        let violations = engine.check_constraints();
        assert_eq!(violations.len(), 1);
        let v: &ConstraintViolation = &violations[0];
        assert_eq!(v.relation, AllenRelation::Overlaps);
        assert!(v.constraint.contains("Before"));
    }
}
