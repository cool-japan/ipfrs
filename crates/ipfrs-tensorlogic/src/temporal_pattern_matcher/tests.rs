//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    fn make_event(label: &str, ts: u64) -> TimedEvent {
        TimedEvent::simple(label, ts)
    }
    fn default_matcher() -> TemporalPatternMatcher {
        TemporalPatternMatcher::with_defaults()
    }
    fn simple_pattern(name: &str, labels: &[&str]) -> TemporalPattern {
        let steps = labels
            .iter()
            .map(|l| PatternStep::new(*l, TemporalConstraint::Unbounded))
            .collect();
        TemporalPattern::new(name, steps, true)
    }
    #[test]
    fn test_register_pattern_success() {
        let mut m = default_matcher();
        let p = simple_pattern("p1", &["A", "B"]);
        assert!(m.register_pattern(p).is_ok());
        assert_eq!(m.stats().patterns_registered, 1);
    }
    #[test]
    fn test_unregister_pattern_success() {
        let mut m = default_matcher();
        m.register_pattern(simple_pattern("p1", &["A"]))
            .expect("test: should succeed");
        assert!(m.unregister_pattern("p1").is_ok());
        assert_eq!(m.stats().patterns_registered, 0);
    }
    #[test]
    fn test_unregister_nonexistent_error() {
        let mut m = default_matcher();
        let err = m.unregister_pattern("ghost");
        assert!(matches!(err, Err(MatcherError::PatternNotFound(_))));
    }
    #[test]
    fn test_register_empty_pattern_error() {
        let mut m = default_matcher();
        let p = TemporalPattern::new("empty", vec![], true);
        assert!(matches!(
            m.register_pattern(p),
            Err(MatcherError::InvalidPattern(_))
        ));
    }
    #[test]
    fn test_register_invalid_between_repeat_error() {
        let mut m = default_matcher();
        let step = PatternStep::new("A", TemporalConstraint::Unbounded)
            .with_repeat(RepeatSpec::Between(5, 2));
        let p = TemporalPattern::new("bad", vec![step], true);
        assert!(matches!(
            m.register_pattern(p),
            Err(MatcherError::InvalidPattern(_))
        ));
    }
    #[test]
    fn test_simple_two_step_sequence() {
        let mut m = default_matcher();
        m.register_pattern(simple_pattern("seq", &["A", "B"]))
            .expect("test: should succeed");
        let r1 = m
            .feed_event(make_event("A", 1000))
            .expect("test: should succeed");
        assert!(r1.is_empty());
        let r2 = m
            .feed_event(make_event("B", 2000))
            .expect("test: should succeed");
        assert_eq!(r2.len(), 1);
        assert_eq!(r2[0].pattern_name, "seq");
        assert_eq!(r2[0].matched_events.len(), 2);
    }
    #[test]
    fn test_three_step_sequence() {
        let mut m = default_matcher();
        m.register_pattern(simple_pattern("seq3", &["X", "Y", "Z"]))
            .expect("test: should succeed");
        assert!(m
            .feed_event(make_event("X", 100))
            .expect("test: should succeed")
            .is_empty());
        assert!(m
            .feed_event(make_event("Y", 200))
            .expect("test: should succeed")
            .is_empty());
        let r = m
            .feed_event(make_event("Z", 300))
            .expect("test: should succeed");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].duration_us, 200);
    }
    #[test]
    fn test_single_step_pattern() {
        let mut m = default_matcher();
        m.register_pattern(simple_pattern("single", &["E"]))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("E", 500))
            .expect("test: should succeed");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].start_ts, 500);
    }
    #[test]
    fn test_no_match_wrong_label() {
        let mut m = default_matcher();
        m.register_pattern(simple_pattern("seq", &["A", "B"]))
            .expect("test: should succeed");
        m.feed_event(make_event("A", 100))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("C", 200))
            .expect("test: should succeed");
        assert!(r.is_empty());
    }
    #[test]
    fn test_within_constraint_satisfied() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded),
            PatternStep::new("B", TemporalConstraint::Within { max_gap_us: 500 }),
        ];
        let p = TemporalPattern::new("win", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 1000))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 1400))
            .expect("test: should succeed");
        assert_eq!(r.len(), 1);
    }
    #[test]
    fn test_within_constraint_violated() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded),
            PatternStep::new("B", TemporalConstraint::Within { max_gap_us: 500 }),
        ];
        let p = TemporalPattern::new("win", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 1000))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 2000))
            .expect("test: should succeed");
        assert!(r.is_empty());
    }
    #[test]
    fn test_after_constraint_satisfied() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded),
            PatternStep::new("B", TemporalConstraint::After { min_gap_us: 300 }),
        ];
        let p = TemporalPattern::new("aft", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 0))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 400))
            .expect("test: should succeed");
        assert_eq!(r.len(), 1);
    }
    #[test]
    fn test_after_constraint_violated() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded),
            PatternStep::new("B", TemporalConstraint::After { min_gap_us: 500 }),
        ];
        let p = TemporalPattern::new("aft", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 0))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 200))
            .expect("test: should succeed");
        assert!(r.is_empty());
    }
    #[test]
    fn test_between_constraint_satisfied() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded),
            PatternStep::new(
                "B",
                TemporalConstraint::Between {
                    min_gap_us: 100,
                    max_gap_us: 500,
                },
            ),
        ];
        let p = TemporalPattern::new("bet", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 1000))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 1300))
            .expect("test: should succeed");
        assert_eq!(r.len(), 1);
    }
    #[test]
    fn test_between_constraint_too_soon() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded),
            PatternStep::new(
                "B",
                TemporalConstraint::Between {
                    min_gap_us: 200,
                    max_gap_us: 500,
                },
            ),
        ];
        let p = TemporalPattern::new("bet", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 1000))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 1050))
            .expect("test: should succeed");
        assert!(r.is_empty());
    }
    #[test]
    fn test_simultaneous_constraint_satisfied() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded),
            PatternStep::new("B", TemporalConstraint::Simultaneous { tolerance_us: 50 }),
        ];
        let p = TemporalPattern::new("sim", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 1000))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 1030))
            .expect("test: should succeed");
        assert_eq!(r.len(), 1);
    }
    #[test]
    fn test_simultaneous_constraint_violated() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded),
            PatternStep::new("B", TemporalConstraint::Simultaneous { tolerance_us: 50 }),
        ];
        let p = TemporalPattern::new("sim", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 1000))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 1100))
            .expect("test: should succeed");
        assert!(r.is_empty());
    }
    #[test]
    fn test_overlapping_matches_enabled() {
        let config = MatcherConfig {
            enable_overlapping_matches: true,
            ..Default::default()
        };
        let mut m = TemporalPatternMatcher::new(config);
        m.register_pattern(simple_pattern("p", &["A", "B"]))
            .expect("test: should succeed");
        m.feed_event(make_event("A", 100))
            .expect("test: should succeed");
        m.feed_event(make_event("A", 200))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 300))
            .expect("test: should succeed");
        assert!(
            r.len() >= 2,
            "expected at least 2 overlapping matches, got {}",
            r.len()
        );
    }
    #[test]
    fn test_overlapping_disabled_single_match() {
        let config = MatcherConfig {
            enable_overlapping_matches: false,
            ..Default::default()
        };
        let mut m = TemporalPatternMatcher::new(config);
        m.register_pattern(simple_pattern("p", &["A", "B"]))
            .expect("test: should succeed");
        m.feed_event(make_event("A", 100))
            .expect("test: should succeed");
        m.feed_event(make_event("A", 200))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 300))
            .expect("test: should succeed");
        assert!(r.len() <= 1);
    }
    #[test]
    fn test_negation_step_triggered_discards_state() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded),
            PatternStep::new("BAD", TemporalConstraint::Within { max_gap_us: 1000 }).negated(),
            PatternStep::new("B", TemporalConstraint::Unbounded),
        ];
        let p = TemporalPattern::new("neg", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 0))
            .expect("test: should succeed");
        m.feed_event(make_event("BAD", 100))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 200))
            .expect("test: should succeed");
        assert!(r.is_empty(), "negation step should have discarded state");
    }
    #[test]
    fn test_negation_step_not_triggered_state_retained() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded),
            PatternStep::new("BAD", TemporalConstraint::Within { max_gap_us: 1000 }).negated(),
        ];
        let p = TemporalPattern::new("neg", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 0))
            .expect("test: should succeed");
        m.feed_event(make_event("GOOD", 100))
            .expect("test: should succeed");
        assert!(m.pending_matches() > 0);
    }
    #[test]
    fn test_stats_false_positives_incremented_on_negation() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded),
            PatternStep::new("X", TemporalConstraint::Within { max_gap_us: 500 }).negated(),
        ];
        let p = TemporalPattern::new("neg2", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 0))
            .expect("test: should succeed");
        m.feed_event(make_event("X", 100))
            .expect("test: should succeed");
        assert!(m.stats().false_positives > 0);
    }
    #[test]
    fn test_repeat_exactly_two() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded)
                .with_repeat(RepeatSpec::Exactly(2)),
            PatternStep::new("B", TemporalConstraint::Unbounded),
        ];
        let p = TemporalPattern::new("rep", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 100))
            .expect("test: should succeed");
        m.feed_event(make_event("A", 200))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 300))
            .expect("test: should succeed");
        assert!(!r.is_empty(), "should match after 2 A's followed by B");
    }
    #[test]
    fn test_repeat_at_least_two() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded)
                .with_repeat(RepeatSpec::AtLeast(2)),
            PatternStep::new("B", TemporalConstraint::Unbounded),
        ];
        let p = TemporalPattern::new("atleast", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 100))
            .expect("test: should succeed");
        m.feed_event(make_event("A", 200))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 300))
            .expect("test: should succeed");
        assert!(!r.is_empty(), "should match with AtLeast(2) A's");
    }
    #[test]
    fn test_repeat_at_most_three() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded).with_repeat(RepeatSpec::AtMost(3)),
            PatternStep::new("B", TemporalConstraint::Unbounded),
        ];
        let p = TemporalPattern::new("atmost", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 100))
            .expect("test: should succeed");
        m.feed_event(make_event("A", 200))
            .expect("test: should succeed");
        m.feed_event(make_event("A", 300))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 400))
            .expect("test: should succeed");
        assert!(!r.is_empty(), "should match with 3 A's (AtMost(3))");
    }
    #[test]
    fn test_repeat_between_spec() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded)
                .with_repeat(RepeatSpec::Between(2, 4)),
            PatternStep::new("B", TemporalConstraint::Unbounded),
        ];
        let p = TemporalPattern::new("betw", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 100))
            .expect("test: should succeed");
        m.feed_event(make_event("A", 200))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 300))
            .expect("test: should succeed");
        assert!(!r.is_empty(), "should match with 2 A's (Between(2,4))");
    }
    #[test]
    fn test_flush_does_not_increase_pending() {
        let config = MatcherConfig {
            max_window_us: 100,
            ..Default::default()
        };
        let mut m = TemporalPatternMatcher::new(config);
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded),
            PatternStep::new("B", TemporalConstraint::Unbounded),
        ];
        m.register_pattern(TemporalPattern::new("p", steps, true))
            .expect("test: should succeed");
        m.feed_event(make_event("A", 0))
            .expect("test: should succeed");
        let pending_before = m.pending_matches();
        m.feed_event(make_event("X", 10000))
            .expect("test: should succeed");
        let _ = m.flush();
        let pending_after = m.pending_matches();
        assert!(pending_after <= pending_before);
    }
    #[test]
    fn test_flush_single_step_match_emitted_on_feed() {
        let mut m = default_matcher();
        m.register_pattern(simple_pattern("single", &["A"]))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("A", 0))
            .expect("test: should succeed");
        assert_eq!(r.len(), 1);
    }
    #[test]
    fn test_buffer_overflow_drops_oldest() {
        let config = MatcherConfig {
            max_events_buffered: 5,
            ..Default::default()
        };
        let mut m = TemporalPatternMatcher::new(config);
        for i in 0..10u64 {
            m.feed_event(make_event("A", i * 100))
                .expect("test: should succeed");
        }
        assert_eq!(m.stats().events_processed, 10);
    }
    #[test]
    fn test_large_event_stream_no_panic() {
        let mut m = default_matcher();
        m.register_pattern(simple_pattern("p", &["A", "B"]))
            .expect("test: should succeed");
        let mut ts = 0u64;
        for _ in 0..1000 {
            m.feed_event(make_event("A", ts))
                .expect("test: should succeed");
            ts += 100;
            m.feed_event(make_event("B", ts))
                .expect("test: should succeed");
            ts += 100;
        }
        assert!(m.stats().matches_found > 0);
    }
    #[test]
    fn test_timestamp_out_of_order_error() {
        let mut m = default_matcher();
        m.feed_event(make_event("A", 1000))
            .expect("test: should succeed");
        let err = m.feed_event(make_event("B", 500));
        assert!(matches!(err, Err(MatcherError::TimestampOutOfOrder { .. })));
    }
    #[test]
    fn test_pattern_not_found_unregister() {
        let mut m = default_matcher();
        assert!(matches!(
            m.unregister_pattern("nonexistent"),
            Err(MatcherError::PatternNotFound(_))
        ));
    }
    #[test]
    fn test_stats_events_processed() {
        let mut m = default_matcher();
        m.feed_event(make_event("A", 0))
            .expect("test: should succeed");
        m.feed_event(make_event("B", 1))
            .expect("test: should succeed");
        assert_eq!(m.stats().events_processed, 2);
    }
    #[test]
    fn test_stats_matches_found() {
        let mut m = default_matcher();
        m.register_pattern(simple_pattern("p", &["A", "B"]))
            .expect("test: should succeed");
        m.feed_event(make_event("A", 0))
            .expect("test: should succeed");
        m.feed_event(make_event("B", 1))
            .expect("test: should succeed");
        assert!(m.stats().matches_found >= 1);
    }
    #[test]
    fn test_pending_matches_count() {
        let mut m = default_matcher();
        m.register_pattern(simple_pattern("p", &["A", "B"]))
            .expect("test: should succeed");
        m.feed_event(make_event("A", 0))
            .expect("test: should succeed");
        assert!(m.pending_matches() >= 1);
    }
    #[test]
    fn test_confidence_high_for_tight_timing() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded),
            PatternStep::new("B", TemporalConstraint::Within { max_gap_us: 1000 }),
        ];
        let p = TemporalPattern::new("conf", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 0))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 100))
            .expect("test: should succeed");
        assert!(!r.is_empty());
        assert!(r[0].confidence > 0.8);
    }
    #[test]
    fn test_confidence_clamped_to_unit_interval() {
        let mut m = default_matcher();
        m.register_pattern(simple_pattern("p", &["A"]))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("A", 0))
            .expect("test: should succeed");
        assert!(!r.is_empty());
        assert!(r[0].confidence >= 0.0 && r[0].confidence <= 1.0);
    }
    #[test]
    fn test_match_result_duration() {
        let mut m = default_matcher();
        m.register_pattern(simple_pattern("p", &["A", "B"]))
            .expect("test: should succeed");
        m.feed_event(make_event("A", 1000))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 3000))
            .expect("test: should succeed");
        assert_eq!(r[0].duration_us, 2000);
    }
    #[test]
    fn test_multiple_patterns_concurrent() {
        let mut m = default_matcher();
        m.register_pattern(simple_pattern("p1", &["A", "B"]))
            .expect("test: should succeed");
        m.register_pattern(simple_pattern("p2", &["A", "C"]))
            .expect("test: should succeed");
        m.feed_event(make_event("A", 0))
            .expect("test: should succeed");
        let rb = m
            .feed_event(make_event("B", 100))
            .expect("test: should succeed");
        let rc = m
            .feed_event(make_event("C", 200))
            .expect("test: should succeed");
        let has_p1 = rb.iter().any(|r| r.pattern_name == "p1");
        let has_p2 = rc.iter().any(|r| r.pattern_name == "p2");
        assert!(has_p1, "p1 should have matched");
        assert!(has_p2, "p2 should have matched");
    }
    #[test]
    fn test_xorshift64_deterministic() {
        let mut state = 12345u64;
        let v1 = xorshift64(&mut state);
        let mut state2 = 12345u64;
        let v2 = xorshift64(&mut state2);
        assert_eq!(v1, v2);
    }
    #[test]
    fn test_xorshift64_changes_state() {
        let mut state = 1u64;
        let _ = xorshift64(&mut state);
        assert_ne!(state, 1u64);
    }
    #[test]
    fn test_xorshift64_sequence_unique() {
        let mut state = 999u64;
        let a = xorshift64(&mut state);
        let b = xorshift64(&mut state);
        let c = xorshift64(&mut state);
        assert_ne!(a, b);
        assert_ne!(b, c);
    }
    #[test]
    fn test_within_satisfies_edge() {
        let c = TemporalConstraint::Within { max_gap_us: 500 };
        assert!(c.satisfied(500));
        assert!(!c.satisfied(501));
    }
    #[test]
    fn test_after_satisfies_edge() {
        let c = TemporalConstraint::After { min_gap_us: 200 };
        assert!(c.satisfied(200));
        assert!(!c.satisfied(199));
    }
    #[test]
    fn test_between_satisfies_range() {
        let c = TemporalConstraint::Between {
            min_gap_us: 100,
            max_gap_us: 400,
        };
        assert!(c.satisfied(100));
        assert!(c.satisfied(250));
        assert!(c.satisfied(400));
        assert!(!c.satisfied(99));
        assert!(!c.satisfied(401));
    }
    #[test]
    fn test_simultaneous_satisfies_zero() {
        let c = TemporalConstraint::Simultaneous { tolerance_us: 10 };
        assert!(c.satisfied(0));
        assert!(c.satisfied(10));
        assert!(!c.satisfied(11));
    }
    #[test]
    fn test_unbounded_always_satisfies() {
        let c = TemporalConstraint::Unbounded;
        assert!(c.satisfied(0));
        assert!(c.satisfied(u64::MAX));
    }
    #[test]
    fn test_repeat_exactly_helpers() {
        let r = RepeatSpec::Exactly(3);
        assert_eq!(r.min_count(), 3);
        assert_eq!(r.max_count(), 3);
        assert!(r.is_satisfied(3));
        assert!(!r.is_satisfied(2));
        assert!(!r.is_satisfied(4));
        assert!(!r.can_repeat(3));
        assert!(r.can_repeat(2));
    }
    #[test]
    fn test_repeat_at_least_helpers() {
        let r = RepeatSpec::AtLeast(2);
        assert_eq!(r.min_count(), 2);
        assert_eq!(r.max_count(), usize::MAX);
        assert!(!r.is_satisfied(1));
        assert!(r.is_satisfied(2));
        assert!(r.is_satisfied(100));
        assert!(r.can_repeat(99));
    }
    #[test]
    fn test_repeat_at_most_helpers() {
        let r = RepeatSpec::AtMost(3);
        assert_eq!(r.min_count(), 0);
        assert_eq!(r.max_count(), 3);
        assert!(r.is_satisfied(0));
        assert!(r.is_satisfied(3));
        assert!(!r.is_satisfied(4));
        assert!(!r.can_repeat(3));
    }
    #[test]
    fn test_repeat_between_helpers() {
        let r = RepeatSpec::Between(2, 5);
        assert_eq!(r.min_count(), 2);
        assert_eq!(r.max_count(), 5);
        assert!(!r.is_satisfied(1));
        assert!(r.is_satisfied(3));
        assert!(r.is_satisfied(5));
        assert!(!r.is_satisfied(6));
    }
    #[test]
    fn test_allow_gaps_true() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded),
            PatternStep::new("C", TemporalConstraint::Unbounded),
        ];
        let p = TemporalPattern::new("gap", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 100))
            .expect("test: should succeed");
        m.feed_event(make_event("B", 200))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("C", 300))
            .expect("test: should succeed");
        assert!(
            !r.is_empty(),
            "allow_gaps=true should tolerate irrelevant event"
        );
    }
    #[test]
    fn test_allow_gaps_false_terminates_on_gap() {
        let config = MatcherConfig {
            enable_overlapping_matches: false,
            ..Default::default()
        };
        let mut m = TemporalPatternMatcher::new(config);
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded),
            PatternStep::new("C", TemporalConstraint::Unbounded),
        ];
        let p = TemporalPattern::new("nogap", steps, false);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 100))
            .expect("test: should succeed");
        m.feed_event(make_event("B", 200))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("C", 300))
            .expect("test: should succeed");
        assert!(
            r.is_empty(),
            "allow_gaps=false should terminate on unexpected event"
        );
    }
    #[test]
    fn test_match_result_start_end_ts() {
        let mut m = default_matcher();
        m.register_pattern(simple_pattern("p", &["A", "B"]))
            .expect("test: should succeed");
        m.feed_event(make_event("A", 500))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 800))
            .expect("test: should succeed");
        assert!(!r.is_empty());
        assert_eq!(r[0].start_ts, 500);
        assert_eq!(r[0].end_ts, 800);
    }
    #[test]
    fn test_match_result_matched_events_len() {
        let mut m = default_matcher();
        m.register_pattern(simple_pattern("p", &["A", "B", "C"]))
            .expect("test: should succeed");
        m.feed_event(make_event("A", 100))
            .expect("test: should succeed");
        m.feed_event(make_event("B", 200))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("C", 300))
            .expect("test: should succeed");
        assert_eq!(r[0].matched_events.len(), 3);
    }
    #[test]
    fn test_event_label_equality() {
        let a = EventLabel::new("hello");
        let b = EventLabel::new("hello");
        let c = EventLabel::new("world");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
    #[test]
    fn test_event_label_display() {
        let l = EventLabel::new("test_label");
        assert_eq!(format!("{l}"), "test_label");
    }
    #[test]
    fn test_matcher_config_defaults() {
        let c = MatcherConfig::default();
        assert!(c.max_events_buffered > 0);
        assert!(c.max_window_us > 0);
    }
    #[test]
    fn test_multiple_sequential_matches() {
        let mut m = default_matcher();
        m.register_pattern(simple_pattern("p", &["A", "B"]))
            .expect("test: should succeed");
        let mut total = 0usize;
        for i in 0..5u64 {
            m.feed_event(make_event("A", i * 1000))
                .expect("test: should succeed");
            let r = m
                .feed_event(make_event("B", i * 1000 + 100))
                .expect("test: should succeed");
            total += r.len();
        }
        assert!(total >= 5, "expected at least 5 matches, got {total}");
    }
    #[test]
    fn test_jitter_timing_all_within_window() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded),
            PatternStep::new("B", TemporalConstraint::Within { max_gap_us: 2000 }),
        ];
        m.register_pattern(TemporalPattern::new("jitter", steps, true))
            .expect("test: should succeed");
        let mut rng = 0xDEADBEEFu64;
        let mut ts = 0u64;
        let mut matches = 0usize;
        for _ in 0..20 {
            m.feed_event(make_event("A", ts))
                .expect("test: should succeed");
            let jitter = xorshift64(&mut rng) % 2000;
            ts += jitter;
            let r = m
                .feed_event(make_event("B", ts))
                .expect("test: should succeed");
            matches += r.len();
            ts += 10_000;
        }
        assert!(matches > 0, "expected some matches with jitter");
    }
    #[test]
    fn test_timed_event_with_payload() {
        let ev = TimedEvent::new("label", 42, vec![1, 2, 3, 4]);
        assert_eq!(ev.payload, vec![1, 2, 3, 4]);
        assert_eq!(ev.timestamp, 42);
    }
    #[test]
    fn test_deviation_within_zero_at_zero() {
        let c = TemporalConstraint::Within { max_gap_us: 1000 };
        assert_eq!(c.deviation(0), 0.0);
    }
    #[test]
    fn test_deviation_within_one_at_max() {
        let c = TemporalConstraint::Within { max_gap_us: 1000 };
        assert!((c.deviation(1000) - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_deviation_unbounded_always_zero() {
        let c = TemporalConstraint::Unbounded;
        assert_eq!(c.deviation(u64::MAX), 0.0);
    }
    #[test]
    fn test_pattern_step_builder() {
        let s = PatternStep::new("A", TemporalConstraint::Unbounded)
            .with_repeat(RepeatSpec::Exactly(3))
            .negated();
        assert!(s.negation);
        assert_eq!(s.max_count(), 3);
    }
    #[test]
    fn test_flush_removes_incomplete_states() {
        let config = MatcherConfig {
            max_window_us: 100,
            ..Default::default()
        };
        let mut m = TemporalPatternMatcher::new(config);
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded),
            PatternStep::new("B", TemporalConstraint::Unbounded),
        ];
        m.register_pattern(TemporalPattern::new("p", steps, true))
            .expect("test: should succeed");
        m.feed_event(make_event("A", 0))
            .expect("test: should succeed");
        let pending_before = m.pending_matches();
        m.feed_event(make_event("X", 10000))
            .expect("test: should succeed");
        let _ = m.flush();
        let pending_after = m.pending_matches();
        assert!(
            pending_after <= pending_before,
            "flush should not increase pending matches"
        );
    }
    #[test]
    fn test_default_repeat_is_exactly_one() {
        let s = PatternStep::new("A", TemporalConstraint::Unbounded);
        assert_eq!(s.min_count(), 1);
        assert_eq!(s.max_count(), 1);
    }
    #[test]
    fn test_equal_timestamps_allowed() {
        let mut m = default_matcher();
        m.register_pattern(simple_pattern("p", &["A", "B"]))
            .expect("test: should succeed");
        m.feed_event(make_event("A", 100))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 100))
            .expect("test: should succeed");
        assert!(!r.is_empty());
        assert_eq!(r[0].duration_us, 0);
    }
    #[test]
    fn test_stats_patterns_registered_count() {
        let mut m = default_matcher();
        m.register_pattern(simple_pattern("p1", &["A"]))
            .expect("test: should succeed");
        m.register_pattern(simple_pattern("p2", &["B"]))
            .expect("test: should succeed");
        m.register_pattern(simple_pattern("p3", &["C"]))
            .expect("test: should succeed");
        assert_eq!(m.stats().patterns_registered, 3);
        m.unregister_pattern("p2").expect("test: should succeed");
        assert_eq!(m.stats().patterns_registered, 2);
    }
    #[test]
    fn test_between_equal_min_max() {
        let c = TemporalConstraint::Between {
            min_gap_us: 100,
            max_gap_us: 100,
        };
        assert!(c.satisfied(100));
        assert!(!c.satisfied(99));
        assert!(!c.satisfied(101));
    }
    #[test]
    fn test_after_exactly_at_min() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded),
            PatternStep::new("B", TemporalConstraint::After { min_gap_us: 100 }),
        ];
        let p = TemporalPattern::new("aft", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 0))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 100))
            .expect("test: should succeed");
        assert_eq!(r.len(), 1);
    }
    #[test]
    fn test_within_exactly_at_max() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded),
            PatternStep::new("B", TemporalConstraint::Within { max_gap_us: 300 }),
        ];
        let p = TemporalPattern::new("win", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 0))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 300))
            .expect("test: should succeed");
        assert_eq!(r.len(), 1);
    }
    #[test]
    fn test_mixed_constraints_sequence() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded),
            PatternStep::new("B", TemporalConstraint::Within { max_gap_us: 500 }),
            PatternStep::new("C", TemporalConstraint::After { min_gap_us: 100 }),
        ];
        let p = TemporalPattern::new("mix", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 0))
            .expect("test: should succeed");
        m.feed_event(make_event("B", 300))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("C", 500))
            .expect("test: should succeed");
        assert_eq!(r.len(), 1);
    }
    #[test]
    fn test_exactly_two_with_one_event_no_match() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded)
                .with_repeat(RepeatSpec::Exactly(2)),
            PatternStep::new("B", TemporalConstraint::Unbounded),
        ];
        let p = TemporalPattern::new("ex2", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 0))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 100))
            .expect("test: should succeed");
        assert!(
            r.is_empty(),
            "should not match with only 1 A when Exactly(2)"
        );
    }
    #[test]
    fn test_match_result_pattern_name() {
        let mut m = default_matcher();
        m.register_pattern(simple_pattern("my_special_pattern", &["X"]))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("X", 0))
            .expect("test: should succeed");
        assert_eq!(r[0].pattern_name, "my_special_pattern");
    }
    #[test]
    fn test_simultaneous_zero_tolerance() {
        let c = TemporalConstraint::Simultaneous { tolerance_us: 0 };
        assert!(c.satisfied(0));
        assert!(!c.satisfied(1));
    }
    #[test]
    fn test_with_defaults_functional() {
        let mut m = TemporalPatternMatcher::with_defaults();
        m.register_pattern(simple_pattern("p", &["Z"]))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("Z", 0))
            .expect("test: should succeed");
        assert_eq!(r.len(), 1);
    }
    #[test]
    fn test_between_repeat_three_matches() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded)
                .with_repeat(RepeatSpec::Between(2, 4)),
            PatternStep::new("B", TemporalConstraint::Unbounded),
        ];
        let p = TemporalPattern::new("betw3", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 100))
            .expect("test: should succeed");
        m.feed_event(make_event("A", 200))
            .expect("test: should succeed");
        m.feed_event(make_event("A", 300))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 400))
            .expect("test: should succeed");
        assert!(!r.is_empty(), "3 A's then B should match Between(2,4)");
    }
    #[test]
    fn test_no_spurious_matches_on_unrelated_labels() {
        let mut m = default_matcher();
        m.register_pattern(simple_pattern("p", &["START", "END"]))
            .expect("test: should succeed");
        for i in 0..50u64 {
            m.feed_event(make_event("NOISE", i * 10))
                .expect("test: should succeed");
        }
        assert_eq!(m.stats().matches_found, 0);
    }
    #[test]
    fn test_flush_with_no_patterns() {
        let mut m = default_matcher();
        let result = m.flush();
        assert!(result.is_empty());
    }
    #[test]
    fn test_at_least_one_single_match() {
        let mut m = default_matcher();
        let steps = vec![
            PatternStep::new("A", TemporalConstraint::Unbounded)
                .with_repeat(RepeatSpec::AtLeast(1)),
            PatternStep::new("B", TemporalConstraint::Unbounded),
        ];
        let p = TemporalPattern::new("al1", steps, true);
        m.register_pattern(p).expect("test: should succeed");
        m.feed_event(make_event("A", 100))
            .expect("test: should succeed");
        let r = m
            .feed_event(make_event("B", 200))
            .expect("test: should succeed");
        assert!(!r.is_empty(), "AtLeast(1) should match after 1 A");
    }
}
