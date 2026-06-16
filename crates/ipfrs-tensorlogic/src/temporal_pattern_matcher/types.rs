//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};
use thiserror::Error;

pub(super) enum AdvanceOutcome {
    /// Pattern fully completed.
    Completed(MatchResult),
    /// Pattern completed AND a fork-branch is kept for more repetitions.
    CompletedAndFork(MatchResult, NfaState),
    /// State advanced to next step (no completion yet).
    Advanced(NfaState),
    /// Fork: keep both the repeating branch and the advancing branch.
    Fork(NfaState, NfaState),
    /// Event did not match; state retained unchanged.
    Retained(NfaState),
    /// State terminated (negation violated or contiguous-mode interruption).
    Discarded,
}
/// A named, multi-step temporal pattern.
#[derive(Clone, Debug)]
pub struct TemporalPattern {
    /// Human-readable name for this pattern.
    pub name: String,
    /// Ordered sequence of steps.
    pub steps: Vec<PatternStep>,
    /// When `false`, matched events must be contiguous in the event stream
    /// (no gaps between matched steps are allowed).
    pub allow_gaps: bool,
}
impl TemporalPattern {
    /// Construct a new `TemporalPattern`.
    pub fn new(name: impl Into<String>, steps: Vec<PatternStep>, allow_gaps: bool) -> Self {
        Self {
            name: name.into(),
            steps,
            allow_gaps,
        }
    }
}
/// Repetition specification for a `PatternStep`.
#[derive(Clone, Debug, PartialEq)]
pub enum RepeatSpec {
    /// Must match exactly `n` times.
    Exactly(usize),
    /// Must match at least `n` times.
    AtLeast(usize),
    /// Must match at most `n` times (0 is valid, meaning the step is optional).
    AtMost(usize),
    /// Must match between `min` and `max` times (inclusive).
    Between(usize, usize),
}
impl RepeatSpec {
    /// The minimum number of repetitions required.
    pub fn min_count(&self) -> usize {
        match self {
            Self::Exactly(n) => *n,
            Self::AtLeast(n) => *n,
            Self::AtMost(_) => 0,
            Self::Between(min, _) => *min,
        }
    }
    /// The maximum number of repetitions allowed (`usize::MAX` for unbounded).
    pub fn max_count(&self) -> usize {
        match self {
            Self::Exactly(n) => *n,
            Self::AtLeast(_) => usize::MAX,
            Self::AtMost(n) => *n,
            Self::Between(_, max) => *max,
        }
    }
    /// Returns `true` if `count` repetitions satisfies this spec.
    pub fn is_satisfied(&self, count: usize) -> bool {
        count >= self.min_count() && count <= self.max_count()
    }
    /// Returns `true` if further repetitions are possible given `count` so far.
    pub fn can_repeat(&self, count: usize) -> bool {
        count < self.max_count()
    }
}
/// Timing constraint between consecutive pattern steps.
///
/// Note: `TemporalConstraint` already exists in `temporal_reasoning`; this
/// type is exported from `lib.rs` under the alias `TpmTemporalConstraint`.
#[derive(Clone, Debug, PartialEq)]
pub enum TemporalConstraint {
    /// Next event must follow within `max_gap_us` microseconds.
    Within { max_gap_us: u64 },
    /// Next event must follow at least `min_gap_us` microseconds later.
    After { min_gap_us: u64 },
    /// Next event must follow in [`min_gap_us`, `max_gap_us`].
    Between { min_gap_us: u64, max_gap_us: u64 },
    /// Events within `tolerance_us` of the same timestamp are simultaneous.
    Simultaneous { tolerance_us: u64 },
    /// No timing constraint.
    Unbounded,
}
impl TemporalConstraint {
    /// Returns `true` when `gap` (in microseconds since the previous event)
    /// satisfies this constraint.
    pub fn satisfied(&self, gap: u64) -> bool {
        match self {
            Self::Within { max_gap_us } => gap <= *max_gap_us,
            Self::After { min_gap_us } => gap >= *min_gap_us,
            Self::Between {
                min_gap_us,
                max_gap_us,
            } => gap >= *min_gap_us && gap <= *max_gap_us,
            Self::Simultaneous { tolerance_us } => gap <= *tolerance_us,
            Self::Unbounded => true,
        }
    }
    /// Computes the per-step timing deviation in [0.0, 1.0].
    /// 0.0 = perfect (or unbounded), 1.0 = at constraint edge.
    pub fn deviation(&self, gap: u64) -> f64 {
        match self {
            Self::Within { max_gap_us } => {
                if *max_gap_us == 0 {
                    0.0
                } else {
                    (gap as f64 / *max_gap_us as f64).min(1.0)
                }
            }
            Self::After { min_gap_us } => {
                if *min_gap_us == 0 || gap == 0 {
                    0.0
                } else {
                    (1.0 - (*min_gap_us as f64 / gap as f64)).clamp(0.0, 1.0)
                }
            }
            Self::Between {
                min_gap_us,
                max_gap_us,
            } => {
                let range = max_gap_us.saturating_sub(*min_gap_us) as f64;
                if range == 0.0 {
                    return 0.0;
                }
                let mid = (*min_gap_us + max_gap_us) / 2;
                let d = (gap as i64 - mid as i64).unsigned_abs() as f64;
                (d / (range / 2.0)).min(1.0)
            }
            Self::Simultaneous { tolerance_us } => {
                if *tolerance_us == 0 {
                    0.0
                } else {
                    (gap as f64 / *tolerance_us as f64).min(1.0)
                }
            }
            Self::Unbounded => 0.0,
        }
    }
}
/// One step in a `TemporalPattern`.
#[derive(Clone, Debug)]
pub struct PatternStep {
    /// The event label that this step matches.
    pub label: EventLabel,
    /// Timing constraint relative to the previous matched event.
    pub constraint: TemporalConstraint,
    /// Optional repetition specification (defaults to `Exactly(1)` when absent).
    pub repeat: Option<RepeatSpec>,
    /// When `true`, this step represents a *must-not-occur* assertion.
    pub negation: bool,
}
impl PatternStep {
    /// Construct a simple (non-negated, non-repeating) step.
    pub fn new(label: impl Into<String>, constraint: TemporalConstraint) -> Self {
        Self {
            label: EventLabel::new(label),
            constraint,
            repeat: None,
            negation: false,
        }
    }
    /// Builder: set repetition spec.
    pub fn with_repeat(mut self, repeat: RepeatSpec) -> Self {
        self.repeat = Some(repeat);
        self
    }
    /// Builder: mark step as negated.
    pub fn negated(mut self) -> Self {
        self.negation = true;
        self
    }
    /// Minimum required matches for this step.
    pub fn min_count(&self) -> usize {
        match &self.repeat {
            Some(r) => r.min_count(),
            None => {
                if self.negation {
                    0
                } else {
                    1
                }
            }
        }
    }
    /// Maximum allowed matches for this step.
    pub fn max_count(&self) -> usize {
        match &self.repeat {
            Some(r) => r.max_count(),
            None => 1,
        }
    }
}
/// A completed pattern match.
#[derive(Clone, Debug)]
pub struct MatchResult {
    /// Name of the matched pattern.
    pub pattern_name: String,
    /// The events that were matched, in order.
    pub matched_events: Vec<TimedEvent>,
    /// Timestamp of the first matched event.
    pub start_ts: u64,
    /// Timestamp of the last matched event.
    pub end_ts: u64,
    /// Total duration in microseconds.
    pub duration_us: u64,
    /// Confidence score in [0.0, 1.0] based on timing tightness.
    pub confidence: f64,
}
/// NFA-based temporal sequence pattern matcher.
///
/// Matches event streams against registered `TemporalPattern`s, tracking
/// multiple concurrent NFA states (one per active partial match), and emitting
/// `MatchResult`s when patterns complete.
pub struct TemporalPatternMatcher {
    pub(super) config: MatcherConfig,
    pub(super) patterns: HashMap<String, TemporalPattern>,
    /// Per-pattern list of active NFA states.
    pub(super) active_states: HashMap<String, Vec<NfaState>>,
    /// Sliding window of raw events for contiguity enforcement.
    pub(super) event_buffer: VecDeque<TimedEvent>,
    pub(super) stats: MatcherStats,
    /// Timestamp of the last processed event (for out-of-order detection).
    pub(super) last_event_ts: Option<u64>,
    /// Running sum of per-call latencies for average computation.
    pub(super) latency_sum_us: f64,
    pub(super) latency_count: u64,
}
impl TemporalPatternMatcher {
    /// Create a new matcher with the provided configuration.
    pub fn new(config: MatcherConfig) -> Self {
        Self {
            config,
            patterns: HashMap::new(),
            active_states: HashMap::new(),
            event_buffer: VecDeque::new(),
            stats: MatcherStats::default(),
            last_event_ts: None,
            latency_sum_us: 0.0,
            latency_count: 0,
        }
    }
    /// Create a matcher with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(MatcherConfig::default())
    }
    /// Register a pattern. Returns `MatcherError::InvalidPattern` when the
    /// pattern has no steps or any `Between` repeat has `min > max`.
    pub fn register_pattern(&mut self, pattern: TemporalPattern) -> Result<(), MatcherError> {
        if pattern.steps.is_empty() {
            return Err(MatcherError::InvalidPattern(format!(
                "pattern '{}' has no steps",
                pattern.name
            )));
        }
        for (i, step) in pattern.steps.iter().enumerate() {
            if let Some(RepeatSpec::Between(min, max)) = &step.repeat {
                if min > max {
                    return Err(MatcherError::InvalidPattern(format!(
                        "pattern '{}' step {i}: repeat min ({min}) > max ({max})",
                        pattern.name
                    )));
                }
            }
        }
        let name = pattern.name.clone();
        self.patterns.insert(name.clone(), pattern);
        self.active_states.entry(name).or_default();
        self.stats.patterns_registered = self.patterns.len();
        Ok(())
    }
    /// Remove a previously registered pattern and its in-flight NFA states.
    pub fn unregister_pattern(&mut self, name: &str) -> Result<(), MatcherError> {
        if self.patterns.remove(name).is_none() {
            return Err(MatcherError::PatternNotFound(name.to_owned()));
        }
        self.active_states.remove(name);
        self.stats.patterns_registered = self.patterns.len();
        Ok(())
    }
    /// Feed a single event into the matcher.
    ///
    /// Returns all `MatchResult`s that were completed by this event.
    pub fn feed_event(&mut self, event: TimedEvent) -> Result<Vec<MatchResult>, MatcherError> {
        if let Some(last) = self.last_event_ts {
            if event.timestamp < last {
                return Err(MatcherError::TimestampOutOfOrder {
                    received: event.timestamp,
                    last,
                });
            }
        }
        self.last_event_ts = Some(event.timestamp);
        if self.event_buffer.len() >= self.config.max_events_buffered {
            self.event_buffer.pop_front();
        }
        self.event_buffer.push_back(event.clone());
        let start_us = Self::monotonic_us();
        let results = self.advance_all_nfa(&event);
        let elapsed_us = Self::elapsed_us(start_us);
        self.latency_sum_us += elapsed_us;
        self.latency_count += 1;
        self.stats.avg_match_latency_us = self.latency_sum_us / self.latency_count as f64;
        self.stats.events_processed += 1;
        self.stats.matches_found += results.len() as u64;
        Ok(results)
    }
    /// Force-complete any partial matches that have timed out.
    ///
    /// A partial match is emitted only when `current_step >= pattern.steps.len()`,
    /// meaning all steps have been consumed at least to their minimum counts.
    pub fn flush(&mut self) -> Vec<MatchResult> {
        let now = self
            .last_event_ts
            .unwrap_or(0)
            .saturating_add(self.config.max_window_us);
        let mut results = Vec::new();
        for (pattern_name, states) in &mut self.active_states {
            let pattern = match self.patterns.get(pattern_name) {
                Some(p) => p,
                None => continue,
            };
            let mut remaining = Vec::new();
            for state in states.drain(..) {
                let age = now.saturating_sub(state.last_ts);
                if age <= self.config.max_window_us {
                    remaining.push(state);
                    continue;
                }
                if state.current_step >= pattern.steps.len() {
                    if let Some(r) = Self::build_match_result(pattern_name, &state) {
                        results.push(r);
                    }
                }
            }
            *states = remaining;
        }
        self.stats.matches_found += results.len() as u64;
        results
    }
    /// Return a snapshot of the current statistics.
    pub fn stats(&self) -> MatcherStats {
        self.stats.clone()
    }
    /// Return the total number of in-progress NFA states across all patterns.
    pub fn pending_matches(&self) -> usize {
        self.active_states.values().map(|v| v.len()).sum()
    }
    pub(super) fn advance_all_nfa(&mut self, event: &TimedEvent) -> Vec<MatchResult> {
        let mut completed: Vec<MatchResult> = Vec::new();
        let pattern_names: Vec<String> = self.patterns.keys().cloned().collect();
        for pname in pattern_names {
            let pattern = match self.patterns.get(&pname) {
                Some(p) => p.clone(),
                None => continue,
            };
            let states = self.active_states.entry(pname.clone()).or_default();
            states
                .retain(|s| event.timestamp.saturating_sub(s.last_ts) <= self.config.max_window_us);
            let old_states: Vec<NfaState> = std::mem::take(states);
            let mut next_states: Vec<NfaState> = Vec::new();
            for state in old_states {
                let outcome = Self::try_advance_state(&pattern, state, event, &self.config);
                match outcome {
                    AdvanceOutcome::Completed(r) => {
                        completed.push(r);
                    }
                    AdvanceOutcome::CompletedAndFork(r, keep) => {
                        completed.push(r);
                        next_states.push(keep);
                    }
                    AdvanceOutcome::Advanced(s) => {
                        next_states.push(s);
                    }
                    AdvanceOutcome::Fork(keep, advance) => {
                        next_states.push(keep);
                        next_states.push(advance);
                    }
                    AdvanceOutcome::Retained(s) => {
                        next_states.push(s);
                    }
                    AdvanceOutcome::Discarded => {
                        self.stats.false_positives += 1;
                    }
                }
            }
            *self.active_states.get_mut(&pname).expect("key exists") = next_states;
            let should_start = self.config.enable_overlapping_matches
                || self
                    .active_states
                    .get(&pname)
                    .map(|v| v.is_empty())
                    .unwrap_or(true);
            if should_start {
                let first_step = match pattern.steps.first() {
                    Some(s) => s,
                    None => continue,
                };
                if !first_step.negation && event.label == first_step.label {
                    let first_gap_ok = match &first_step.constraint {
                        TemporalConstraint::After { min_gap_us } => *min_gap_us == 0,
                        other => other.satisfied(0),
                    };
                    if first_gap_ok {
                        let max_step0 = first_step.max_count();
                        if max_step0 == 0 {
                        } else if pattern.steps.len() == 1 {
                            let dev = first_step.constraint.deviation(0);
                            let mr = MatchResult {
                                pattern_name: pname.clone(),
                                matched_events: vec![event.clone()],
                                start_ts: event.timestamp,
                                end_ts: event.timestamp,
                                duration_us: 0,
                                confidence: (1.0 - dev).clamp(0.0, 1.0),
                            };
                            completed.push(mr);
                            self.stats.matches_found += 1;
                        } else {
                            let new_state =
                                NfaState::after_first_step(pname.clone(), event.clone());
                            let adjusted = if first_step.min_count() > 1 {
                                NfaState {
                                    current_step: 0,
                                    step_repeat_count: 1,
                                    ..new_state
                                }
                            } else {
                                new_state
                            };
                            self.active_states
                                .get_mut(&pname)
                                .expect("key exists")
                                .push(adjusted);
                        }
                    }
                }
            }
        }
        completed
    }
    /// Attempt to advance a single NFA `state` given the new `event`.
    pub(super) fn try_advance_state(
        pattern: &TemporalPattern,
        mut state: NfaState,
        event: &TimedEvent,
        config: &MatcherConfig,
    ) -> AdvanceOutcome {
        let step_idx = state.current_step;
        if step_idx >= pattern.steps.len() {
            return AdvanceOutcome::Retained(state);
        }
        let step = &pattern.steps[step_idx];
        let gap = event.timestamp.saturating_sub(state.last_ts);
        if gap > config.max_window_us {
            return AdvanceOutcome::Discarded;
        }
        if step.negation {
            if event.label == step.label && step.constraint.satisfied(gap) {
                return AdvanceOutcome::Discarded;
            }
            return AdvanceOutcome::Retained(state);
        }
        if event.label != step.label {
            if !pattern.allow_gaps && step_idx > 0 {
                return AdvanceOutcome::Discarded;
            }
            return AdvanceOutcome::Retained(state);
        }
        if !step.constraint.satisfied(gap) {
            return AdvanceOutcome::Retained(state);
        }
        let dev = step.constraint.deviation(gap);
        state.matched_so_far.push(event.clone());
        state.last_ts = event.timestamp;
        state.timing_deviations.push(dev);
        state.step_repeat_count += 1;
        let repeat_count = state.step_repeat_count;
        let step_min = step.min_count();
        let step_max = step.max_count();
        let step_satisfied = repeat_count >= step_min;
        let can_repeat_more = repeat_count < step_max;
        if step_satisfied && can_repeat_more {
            let mut advance_state = state.clone();
            advance_state.current_step = step_idx + 1;
            advance_state.step_repeat_count = 0;
            let keep_state = NfaState { ..state };
            if advance_state.current_step >= pattern.steps.len() {
                match Self::build_match_result(&advance_state.pattern_name, &advance_state) {
                    Some(r) => return AdvanceOutcome::CompletedAndFork(r, keep_state),
                    None => return AdvanceOutcome::Retained(keep_state),
                }
            }
            return AdvanceOutcome::Fork(keep_state, advance_state);
        }
        if step_satisfied && !can_repeat_more {
            state.current_step = step_idx + 1;
            state.step_repeat_count = 0;
            if state.current_step >= pattern.steps.len() {
                return match Self::build_match_result(&state.pattern_name, &state) {
                    Some(r) => AdvanceOutcome::Completed(r),
                    None => AdvanceOutcome::Discarded,
                };
            }
            return AdvanceOutcome::Advanced(state);
        }
        AdvanceOutcome::Advanced(state)
    }
    /// Build a `MatchResult` from a completed NFA state.
    pub(super) fn build_match_result(pattern_name: &str, state: &NfaState) -> Option<MatchResult> {
        if state.matched_so_far.is_empty() {
            return None;
        }
        let start_ts = state.matched_so_far.first()?.timestamp;
        let end_ts = state.matched_so_far.last()?.timestamp;
        let duration_us = end_ts.saturating_sub(start_ts);
        let confidence = state.confidence();
        Some(MatchResult {
            pattern_name: pattern_name.to_owned(),
            matched_events: state.matched_so_far.clone(),
            start_ts,
            end_ts,
            duration_us,
            confidence,
        })
    }
    /// Returns current monotonic microsecond timestamp.
    pub(super) fn monotonic_us() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0)
    }
    /// Elapsed microseconds since the given start value.
    pub(super) fn elapsed_us(start: u64) -> f64 {
        let now = Self::monotonic_us();
        now.saturating_sub(start) as f64
    }
}
/// A single event with a label, microsecond timestamp, and opaque payload.
#[derive(Clone, Debug)]
pub struct TimedEvent {
    /// Identifier for the event class.
    pub label: EventLabel,
    /// Microsecond timestamp (monotonic origin assumed by caller).
    pub timestamp: u64,
    /// Opaque application payload.
    pub payload: Vec<u8>,
}
impl TimedEvent {
    /// Construct a new `TimedEvent`.
    pub fn new(label: impl Into<String>, timestamp: u64, payload: Vec<u8>) -> Self {
        Self {
            label: EventLabel::new(label),
            timestamp,
            payload,
        }
    }
    /// Convenience constructor with empty payload.
    pub fn simple(label: impl Into<String>, timestamp: u64) -> Self {
        Self::new(label, timestamp, Vec::new())
    }
}
/// A label identifying a class of events.
#[derive(Clone, Hash, Eq, PartialEq, Debug)]
pub struct EventLabel(pub String);
impl EventLabel {
    /// Create a new `EventLabel`.
    pub fn new(label: impl Into<String>) -> Self {
        Self(label.into())
    }
}
/// Configuration for `TemporalPatternMatcher`.
#[derive(Clone, Debug)]
pub struct MatcherConfig {
    /// Maximum allowed time window (µs) between first and last event of a match.
    pub max_window_us: u64,
    /// Maximum number of raw events kept in the sliding buffer.
    pub max_events_buffered: usize,
    /// When `true`, overlapping matches for the same pattern are tracked.
    pub enable_overlapping_matches: bool,
}
/// Active NFA state tracking progress through one pattern.
///
/// Invariant: `current_step` is the *next* step index to match.
/// After step `i` is fully consumed, `current_step` becomes `i + 1`.
#[derive(Clone, Debug)]
pub struct NfaState {
    /// Name of the pattern being matched.
    pub pattern_name: String,
    /// Index of the NEXT step to match (0-based).
    pub current_step: usize,
    /// Events matched so far.
    pub matched_so_far: Vec<TimedEvent>,
    /// Timestamp of the most recently matched event.
    pub last_ts: u64,
    /// How many times the *current* step has been repeated so far (0 = none yet).
    pub step_repeat_count: usize,
    /// Accumulated per-step timing deviations for confidence computation.
    pub timing_deviations: Vec<f64>,
}
impl NfaState {
    /// Create a fresh state after the first-step event has been matched.
    ///
    /// `current_step` is set to `1` because step 0 has been consumed.
    /// `step_repeat_count` is set to `0` because we are about to attempt step 1.
    pub(super) fn after_first_step(pattern_name: String, first_event: TimedEvent) -> Self {
        let ts = first_event.timestamp;
        Self {
            pattern_name,
            current_step: 1,
            matched_so_far: vec![first_event],
            last_ts: ts,
            step_repeat_count: 0,
            timing_deviations: Vec::new(),
        }
    }
    /// Compute overall confidence from accumulated deviations.
    pub(super) fn confidence(&self) -> f64 {
        if self.timing_deviations.is_empty() {
            return 1.0;
        }
        let mean_dev =
            self.timing_deviations.iter().sum::<f64>() / self.timing_deviations.len() as f64;
        (1.0 - mean_dev).clamp(0.0, 1.0)
    }
}
/// Errors returned by `TemporalPatternMatcher`.
#[derive(Debug, Error)]
pub enum MatcherError {
    #[error("pattern not found: {0}")]
    PatternNotFound(String),
    #[error("invalid pattern: {0}")]
    InvalidPattern(String),
    #[error("event buffer overflow (max_events_buffered exceeded)")]
    BufferOverflow,
    #[error("timestamp out of order: received {received}, last was {last}")]
    TimestampOutOfOrder { received: u64, last: u64 },
}
/// Runtime statistics for a `TemporalPatternMatcher`.
#[derive(Clone, Debug, Default)]
pub struct MatcherStats {
    /// Number of patterns currently registered.
    pub patterns_registered: usize,
    /// Total events processed since creation.
    pub events_processed: u64,
    /// Total complete matches found since creation.
    pub matches_found: u64,
    /// Total false positives detected (negation steps that were triggered).
    pub false_positives: u64,
    /// Average latency (µs) of a single `feed_event` call.
    pub avg_match_latency_us: f64,
}
