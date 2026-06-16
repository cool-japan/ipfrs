//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    fn make_entry(id: u64, class: TrafficClass, bytes: usize, ts: u64) -> QueueEntry {
        QueueEntry {
            id,
            data_bytes: bytes,
            class,
            enqueued_at: ts,
            source: "src".to_string(),
            destination: "dst".to_string(),
        }
    }
    fn fifo_shaper(depth: usize) -> TrafficShaper {
        TrafficShaper::new(ShaperConfig {
            discipline: QueuingDiscipline::Fifo,
            max_queue_depth: depth,
            rate_limit_bps: 0,
            burst_allowance_bytes: 0,
            drop_policy: DropPolicy::Tail,
        })
        .expect("valid config")
    }
    fn pq_shaper(bands: u8, depth: usize) -> TrafficShaper {
        TrafficShaper::new(ShaperConfig {
            discipline: QueuingDiscipline::PriorityQueue(bands),
            max_queue_depth: depth,
            rate_limit_bps: 0,
            burst_allowance_bytes: 0,
            drop_policy: DropPolicy::Tail,
        })
        .expect("valid config")
    }
    fn wfq_shaper(depth: usize) -> TrafficShaper {
        TrafficShaper::new(ShaperConfig {
            discipline: QueuingDiscipline::WeightedFairQueuing {
                weights: vec![
                    (TrafficClass::RealTime, 8),
                    (TrafficClass::BulkData, 4),
                    (TrafficClass::Background, 1),
                ],
            },
            max_queue_depth: depth,
            rate_limit_bps: 0,
            burst_allowance_bytes: 0,
            drop_policy: DropPolicy::Tail,
        })
        .expect("valid config")
    }
    fn token_shaper(rate_bps: u64, burst_bytes: u64, depth: usize) -> TrafficShaper {
        TrafficShaper::new(ShaperConfig {
            discipline: QueuingDiscipline::TokenBucket {
                rate_bps,
                burst_bytes,
            },
            max_queue_depth: depth,
            rate_limit_bps: rate_bps,
            burst_allowance_bytes: burst_bytes,
            drop_policy: DropPolicy::Tail,
        })
        .expect("valid config")
    }
    fn leaky_shaper(drain_rate_bps: u64, bucket_bytes: u64, depth: usize) -> TrafficShaper {
        TrafficShaper::new(ShaperConfig {
            discipline: QueuingDiscipline::LeakyBucket {
                drain_rate_bps,
                bucket_bytes,
            },
            max_queue_depth: depth,
            rate_limit_bps: drain_rate_bps,
            burst_allowance_bytes: bucket_bytes,
            drop_policy: DropPolicy::Tail,
        })
        .expect("valid config")
    }
    fn diffserv_shaper(depth: usize) -> TrafficShaper {
        TrafficShaper::new(ShaperConfig {
            discipline: QueuingDiscipline::DiffServ {
                dscp_map: vec![
                    (46, TrafficClass::RealTime),
                    (34, TrafficClass::Interactive),
                    (0, TrafficClass::BulkData),
                    (8, TrafficClass::Background),
                ],
            },
            max_queue_depth: depth,
            rate_limit_bps: 0,
            burst_allowance_bytes: 0,
            drop_policy: DropPolicy::Tail,
        })
        .expect("valid config")
    }
    #[test]
    fn test_fifo_basic_enqueue_dequeue() {
        let mut s = fifo_shaper(10);
        s.enqueue(make_entry(1, TrafficClass::BulkData, 100, 1000), 1000)
            .expect("test: enqueue entry 1");
        s.enqueue(make_entry(2, TrafficClass::BulkData, 200, 2000), 2000)
            .expect("test: enqueue entry 2");
        let e = s.dequeue(3000).expect("test: dequeue first entry");
        assert_eq!(e.id, 1);
        let e2 = s.dequeue(4000).expect("test: dequeue second entry");
        assert_eq!(e2.id, 2);
    }
    #[test]
    fn test_fifo_dequeue_empty_returns_none() {
        let mut s = fifo_shaper(10);
        assert!(s.dequeue(1000).is_none());
    }
    #[test]
    fn test_fifo_queue_depth() {
        let mut s = fifo_shaper(10);
        assert_eq!(s.queue_depth(), 0);
        s.enqueue(make_entry(1, TrafficClass::BulkData, 100, 1000), 1000)
            .expect("test: enqueue entry");
        assert_eq!(s.queue_depth(), 1);
        s.dequeue(2000);
        assert_eq!(s.queue_depth(), 0);
    }
    #[test]
    fn test_fifo_stats_enqueued_dequeued() {
        let mut s = fifo_shaper(10);
        s.enqueue(make_entry(1, TrafficClass::BulkData, 512, 1000), 1000)
            .expect("test: enqueue entry");
        s.dequeue(2000);
        let st = s.stats();
        assert_eq!(st.enqueued, 1);
        assert_eq!(st.dequeued, 1);
        assert_eq!(st.bytes_shaped, 512);
    }
    #[test]
    fn test_tail_drop_policy() {
        let mut s = fifo_shaper(2);
        s.enqueue(make_entry(1, TrafficClass::BulkData, 100, 1000), 1000)
            .expect("test: enqueue entry");
        s.enqueue(make_entry(2, TrafficClass::BulkData, 100, 2000), 2000)
            .expect("test: enqueue entry");
        let result = s.enqueue(make_entry(3, TrafficClass::BulkData, 100, 3000), 3000);
        assert!(matches!(result, Err(ShaperError::QueueFull(_))));
        assert_eq!(s.queue_depth(), 2);
        let st = s.stats();
        assert_eq!(st.dropped, 1);
    }
    #[test]
    fn test_tail_drop_event_emitted() {
        let mut s = fifo_shaper(1);
        s.enqueue(make_entry(1, TrafficClass::BulkData, 100, 1000), 1000)
            .expect("test: enqueue entry");
        let _ = s.enqueue(make_entry(2, TrafficClass::BulkData, 100, 2000), 2000);
        let events = s.drain_events();
        let dropped = events
            .iter()
            .any(|e| matches!(e, ShaperEvent::PacketDropped { id: 2, .. }));
        assert!(dropped, "PacketDropped event expected");
    }
    #[test]
    fn test_head_drop_policy() {
        let mut s = TrafficShaper::new(ShaperConfig {
            discipline: QueuingDiscipline::Fifo,
            max_queue_depth: 2,
            rate_limit_bps: 0,
            burst_allowance_bytes: 0,
            drop_policy: DropPolicy::Head,
        })
        .expect("test: create shaper");
        s.enqueue(make_entry(1, TrafficClass::BulkData, 100, 1000), 1000)
            .expect("test: enqueue entry");
        s.enqueue(make_entry(2, TrafficClass::BulkData, 100, 2000), 2000)
            .expect("test: enqueue entry");
        s.enqueue(make_entry(3, TrafficClass::BulkData, 100, 3000), 3000)
            .expect("test: enqueue entry");
        assert_eq!(s.queue_depth(), 2);
        let e = s.dequeue(4000).expect("test: dequeue entry");
        assert_eq!(e.id, 2, "oldest (id=1) should have been evicted");
    }
    #[test]
    fn test_head_drop_event_emitted() {
        let mut s = TrafficShaper::new(ShaperConfig {
            discipline: QueuingDiscipline::Fifo,
            max_queue_depth: 1,
            rate_limit_bps: 0,
            burst_allowance_bytes: 0,
            drop_policy: DropPolicy::Head,
        })
        .expect("test: create shaper");
        s.enqueue(make_entry(1, TrafficClass::BulkData, 100, 1000), 1000)
            .expect("test: enqueue entry");
        s.enqueue(make_entry(2, TrafficClass::BulkData, 100, 2000), 2000)
            .expect("test: enqueue entry");
        let events = s.drain_events();
        let dropped = events
            .iter()
            .any(|e| matches!(e, ShaperEvent::PacketDropped { id: 1, .. }));
        assert!(
            dropped,
            "PacketDropped(id=1) event expected for head eviction"
        );
    }
    #[test]
    fn test_red_no_drop_below_min() {
        let mut s = TrafficShaper::new(ShaperConfig {
            discipline: QueuingDiscipline::Fifo,
            max_queue_depth: 100,
            rate_limit_bps: 0,
            burst_allowance_bytes: 0,
            drop_policy: DropPolicy::RED {
                min_threshold: 50,
                max_threshold: 90,
            },
        })
        .expect("test: create shaper");
        for i in 0..10u64 {
            s.enqueue(
                make_entry(i, TrafficClass::BulkData, 10, i * 1000),
                i * 1000,
            )
            .expect("test: enqueue entry");
        }
        assert_eq!(s.stats().dropped, 0);
    }
    #[test]
    fn test_red_forced_drop_at_max() {
        let mut s = TrafficShaper::new(ShaperConfig {
            discipline: QueuingDiscipline::Fifo,
            max_queue_depth: 100,
            rate_limit_bps: 0,
            burst_allowance_bytes: 0,
            drop_policy: DropPolicy::RED {
                min_threshold: 9,
                max_threshold: 10,
            },
        })
        .expect("test: create shaper");
        for i in 0..9u64 {
            s.enqueue(make_entry(i, TrafficClass::BulkData, 10, 0), 0)
                .expect("test: enqueue entry");
        }
        s.enqueue(make_entry(9, TrafficClass::BulkData, 10, 0), 0)
            .expect("test: enqueue entry");
        let result = s.enqueue(make_entry(99, TrafficClass::BulkData, 10, 0), 0);
        assert!(matches!(result, Err(ShaperError::QueueFull(_))));
    }
    #[test]
    fn test_red_probabilistic_region() {
        let mut s = TrafficShaper::new(ShaperConfig {
            discipline: QueuingDiscipline::Fifo,
            max_queue_depth: 200,
            rate_limit_bps: 0,
            burst_allowance_bytes: 0,
            drop_policy: DropPolicy::RED {
                min_threshold: 10,
                max_threshold: 100,
            },
        })
        .expect("test: create shaper");
        for i in 0..10u64 {
            s.enqueue(make_entry(i, TrafficClass::BulkData, 10, 0), 0)
                .expect("test: enqueue entry");
        }
        let mut accepted = 0u64;
        let mut dropped = 0u64;
        for i in 10..110u64 {
            match s.enqueue(make_entry(i, TrafficClass::BulkData, 10, 0), 0) {
                Ok(_) => accepted += 1,
                Err(_) => dropped += 1,
            }
        }
        assert!(accepted > 0, "some must be accepted in RED zone");
        assert!(dropped > 0, "some must be dropped in RED zone");
    }
    #[test]
    fn test_red_invalid_config_rejected() {
        let result = TrafficShaper::new(ShaperConfig {
            discipline: QueuingDiscipline::Fifo,
            max_queue_depth: 100,
            rate_limit_bps: 0,
            burst_allowance_bytes: 0,
            drop_policy: DropPolicy::RED {
                min_threshold: 50,
                max_threshold: 30,
            },
        });
        assert!(matches!(result, Err(ShaperError::InvalidConfig(_))));
    }
    #[test]
    fn test_priority_queue_high_before_low() {
        let mut s = pq_shaper(4, 20);
        s.enqueue(make_entry(1, TrafficClass::Background, 100, 1000), 1000)
            .expect("test: enqueue entry");
        s.enqueue(make_entry(2, TrafficClass::RealTime, 100, 2000), 2000)
            .expect("test: enqueue entry");
        let first = s.dequeue(3000).expect("test: dequeue entry");
        assert_eq!(first.id, 2, "RealTime should dequeue before Background");
    }
    #[test]
    fn test_priority_queue_fifo_within_band() {
        let mut s = pq_shaper(4, 20);
        s.enqueue(make_entry(10, TrafficClass::BulkData, 100, 1000), 1000)
            .expect("test: enqueue entry");
        s.enqueue(make_entry(11, TrafficClass::BulkData, 100, 2000), 2000)
            .expect("test: enqueue entry");
        let a = s.dequeue(3000).expect("test: dequeue entry");
        let b = s.dequeue(4000).expect("test: dequeue entry");
        assert_eq!(a.id, 10);
        assert_eq!(b.id, 11);
    }
    #[test]
    fn test_priority_queue_single_band() {
        let mut s = pq_shaper(1, 10);
        s.enqueue(make_entry(1, TrafficClass::BulkData, 10, 1000), 1000)
            .expect("test: enqueue entry");
        assert_eq!(s.queue_depth(), 1);
        s.dequeue(2000).expect("test: dequeue entry");
        assert_eq!(s.queue_depth(), 0);
    }
    #[test]
    fn test_priority_queue_drain_class() {
        let mut s = pq_shaper(4, 20);
        s.enqueue(make_entry(1, TrafficClass::Background, 10, 1000), 1000)
            .expect("test: enqueue entry");
        s.enqueue(make_entry(2, TrafficClass::Background, 10, 2000), 2000)
            .expect("test: enqueue entry");
        s.enqueue(make_entry(3, TrafficClass::RealTime, 10, 3000), 3000)
            .expect("test: enqueue entry");
        let drained = s.drain_class(TrafficClass::Background, 10);
        assert_eq!(drained.len(), 2);
        assert_eq!(s.queue_depth(), 1);
    }
    #[test]
    fn test_wfq_dequeues_all_classes() {
        let mut s = wfq_shaper(20);
        s.enqueue(make_entry(1, TrafficClass::RealTime, 100, 1000), 1000)
            .expect("test: enqueue entry");
        s.enqueue(make_entry(2, TrafficClass::BulkData, 100, 1000), 1000)
            .expect("test: enqueue entry");
        s.enqueue(make_entry(3, TrafficClass::Background, 100, 1000), 1000)
            .expect("test: enqueue entry");
        let mut seen = std::collections::HashSet::new();
        for _ in 0..3 {
            if let Some(e) = s.dequeue(2000) {
                seen.insert(e.id);
            }
        }
        assert!(seen.contains(&1));
        assert!(seen.contains(&2));
        assert!(seen.contains(&3));
    }
    #[test]
    fn test_wfq_weighted_ordering() {
        let mut s = wfq_shaper(100);
        for i in 0..30u64 {
            s.enqueue(make_entry(i, TrafficClass::RealTime, 100, i * 100), i * 100)
                .expect("test: enqueue entry");
        }
        for i in 30..60u64 {
            s.enqueue(make_entry(i, TrafficClass::BulkData, 100, i * 100), i * 100)
                .expect("test: enqueue entry");
        }
        for i in 60..90u64 {
            s.enqueue(
                make_entry(i, TrafficClass::Background, 100, i * 100),
                i * 100,
            )
            .expect("test: enqueue entry");
        }
        let mut rt_count = 0u32;
        let mut bulk_count = 0u32;
        let mut bg_count = 0u32;
        for _ in 0..90 {
            if let Some(e) = s.dequeue(1_000_000) {
                match e.class {
                    TrafficClass::RealTime => rt_count += 1,
                    TrafficClass::BulkData => bulk_count += 1,
                    TrafficClass::Background => bg_count += 1,
                    _ => {}
                }
            }
        }
        assert!(
            rt_count >= bulk_count,
            "RealTime({rt_count}) >= BulkData({bulk_count})"
        );
        assert!(
            bulk_count >= bg_count,
            "BulkData({bulk_count}) >= Background({bg_count})"
        );
    }
    #[test]
    fn test_wfq_drain_class() {
        let mut s = wfq_shaper(20);
        for i in 0..5u64 {
            s.enqueue(make_entry(i, TrafficClass::RealTime, 100, i * 100), i * 100)
                .expect("test: enqueue entry");
        }
        for i in 5..10u64 {
            s.enqueue(
                make_entry(i, TrafficClass::Background, 100, i * 100),
                i * 100,
            )
            .expect("test: enqueue entry");
        }
        let drained = s.drain_class(TrafficClass::RealTime, 3);
        assert_eq!(drained.len(), 3);
        assert_eq!(s.queue_depth(), 7);
    }
    #[test]
    fn test_wfq_unknown_class_falls_to_last() {
        let mut s = wfq_shaper(20);
        let result = s.enqueue(make_entry(1, TrafficClass::Management, 100, 1000), 1000);
        assert!(result.is_ok());
        assert_eq!(s.queue_depth(), 1);
    }
    #[test]
    fn test_token_bucket_allows_burst() {
        let mut s = token_shaper(1_000_000, 10_240, 100);
        for i in 0..10u64 {
            let result = s.enqueue(make_entry(i, TrafficClass::BulkData, 1024, 0), 0);
            assert!(result.is_ok(), "entry {i} should fit in burst");
        }
    }
    #[test]
    fn test_token_bucket_rate_limit_exceeded() {
        let mut s = token_shaper(1_000_000, 1000, 100);
        let result = s.enqueue(make_entry(1, TrafficClass::BulkData, 1000, 0), 0);
        assert!(result.is_ok());
        let result2 = s.enqueue(make_entry(2, TrafficClass::BulkData, 100, 0), 0);
        assert!(matches!(result2, Err(ShaperError::RateLimitExceeded(_))));
    }
    #[test]
    fn test_token_bucket_replenish_after_time() {
        let mut s = token_shaper(1_000_000, 200, 100);
        s.enqueue(make_entry(1, TrafficClass::BulkData, 200, 0), 0)
            .expect("test: enqueue entry");
        let result = s.enqueue(make_entry(2, TrafficClass::BulkData, 200, 1_600), 1_600);
        assert!(result.is_ok(), "should have tokens after replenishment");
    }
    #[test]
    fn test_token_bucket_replenish_tokens_fn() {
        let mut s = token_shaper(8_000_000, 1000, 100);
        s.tokens = 0;
        s.last_replenish_us = 0;
        let tokens = s.replenish_tokens(1_000);
        assert_eq!(tokens, 1000);
    }
    #[test]
    fn test_token_bucket_update_rate() {
        let mut s = token_shaper(1_000_000, 10_000, 100);
        s.update_rate(2_000_000).expect("test: update rate");
        if let QueuingDiscipline::TokenBucket { rate_bps, .. } = &s.config.discipline {
            assert_eq!(*rate_bps, 2_000_000);
        } else {
            panic!("expected TokenBucket discipline");
        }
    }
    #[test]
    fn test_token_bucket_dequeue_order() {
        let mut s = token_shaper(1_000_000, 10_000, 100);
        s.enqueue(make_entry(1, TrafficClass::BulkData, 100, 0), 0)
            .expect("test: enqueue entry");
        s.enqueue(make_entry(2, TrafficClass::BulkData, 100, 1000), 1000)
            .expect("test: enqueue entry");
        let e1 = s.dequeue(2000).expect("test: dequeue entry");
        assert_eq!(e1.id, 1);
    }
    #[test]
    fn test_leaky_bucket_allows_within_capacity() {
        let mut s = leaky_shaper(1_000_000, 1000, 100);
        let result = s.enqueue(make_entry(1, TrafficClass::BulkData, 999, 0), 0);
        assert!(result.is_ok());
    }
    #[test]
    fn test_leaky_bucket_overflow_rejected() {
        let mut s = leaky_shaper(1_000_000, 500, 100);
        s.enqueue(make_entry(1, TrafficClass::BulkData, 500, 0), 0)
            .expect("test: enqueue entry");
        let result = s.enqueue(make_entry(2, TrafficClass::BulkData, 100, 0), 0);
        assert!(matches!(result, Err(ShaperError::RateLimitExceeded(_))));
    }
    #[test]
    fn test_leaky_bucket_drains_over_time() {
        let mut s = leaky_shaper(8_000_000, 500, 100);
        s.enqueue(make_entry(1, TrafficClass::BulkData, 500, 0), 0)
            .expect("test: enqueue entry");
        let result = s.enqueue(make_entry(2, TrafficClass::BulkData, 500, 500), 500);
        assert!(result.is_ok(), "leaky bucket should drain over time");
    }
    #[test]
    fn test_leaky_bucket_update_rate() {
        let mut s = leaky_shaper(1_000_000, 10_000, 100);
        s.update_rate(500_000).expect("test: update rate");
        if let QueuingDiscipline::LeakyBucket { drain_rate_bps, .. } = &s.config.discipline {
            assert_eq!(*drain_rate_bps, 500_000);
        } else {
            panic!("expected LeakyBucket discipline");
        }
    }
    #[test]
    fn test_diffserv_priority_order() {
        let mut s = diffserv_shaper(20);
        s.enqueue(make_entry(1, TrafficClass::Background, 100, 1000), 1000)
            .expect("test: enqueue entry");
        s.enqueue(make_entry(2, TrafficClass::RealTime, 100, 1000), 1000)
            .expect("test: enqueue entry");
        s.enqueue(make_entry(3, TrafficClass::BulkData, 100, 1000), 1000)
            .expect("test: enqueue entry");
        let first = s.dequeue(2000).expect("test: dequeue entry");
        assert_eq!(first.id, 2, "RealTime should dequeue first");
        let second = s.dequeue(3000).expect("test: dequeue entry");
        assert_ne!(
            second.id, 1,
            "Background should not dequeue before BulkData"
        );
    }
    #[test]
    fn test_diffserv_resolve_dscp() {
        let s = diffserv_shaper(20);
        assert_eq!(s.resolve_dscp(46), TrafficClass::RealTime);
        assert_eq!(s.resolve_dscp(0), TrafficClass::BulkData);
        assert_eq!(s.resolve_dscp(99), TrafficClass::BulkData);
    }
    #[test]
    fn test_diffserv_drain_class() {
        let mut s = diffserv_shaper(20);
        for i in 0..3u64 {
            s.enqueue(make_entry(i, TrafficClass::RealTime, 100, i * 100), i * 100)
                .expect("test: enqueue entry");
        }
        s.enqueue(make_entry(10, TrafficClass::Background, 100, 1000), 1000)
            .expect("test: enqueue entry");
        let drained = s.drain_class(TrafficClass::RealTime, 2);
        assert_eq!(drained.len(), 2);
        assert_eq!(s.queue_depth(), 2);
    }
    #[test]
    fn test_peek_does_not_remove() {
        let mut s = fifo_shaper(10);
        s.enqueue(make_entry(1, TrafficClass::BulkData, 100, 1000), 1000)
            .expect("test: enqueue entry");
        let peeked = s.peek().map(|e| e.id);
        assert_eq!(peeked, Some(1));
        assert_eq!(s.queue_depth(), 1);
    }
    #[test]
    fn test_peek_empty_returns_none() {
        let s = fifo_shaper(10);
        assert!(s.peek().is_none());
    }
    #[test]
    fn test_drain_class_fifo() {
        let mut s = fifo_shaper(20);
        s.enqueue(make_entry(1, TrafficClass::RealTime, 100, 1000), 1000)
            .expect("test: enqueue entry");
        s.enqueue(make_entry(2, TrafficClass::BulkData, 100, 2000), 2000)
            .expect("test: enqueue entry");
        s.enqueue(make_entry(3, TrafficClass::RealTime, 100, 3000), 3000)
            .expect("test: enqueue entry");
        let drained = s.drain_class(TrafficClass::RealTime, 10);
        assert_eq!(drained.len(), 2);
        assert_eq!(s.queue_depth(), 1);
        let remaining = s.dequeue(4000).expect("test: dequeue entry");
        assert_eq!(remaining.id, 2);
    }
    #[test]
    fn test_drain_class_respects_n() {
        let mut s = fifo_shaper(20);
        for i in 0..5u64 {
            s.enqueue(
                make_entry(i, TrafficClass::Background, 100, i * 1000),
                i * 1000,
            )
            .expect("test: enqueue entry");
        }
        let drained = s.drain_class(TrafficClass::Background, 3);
        assert_eq!(drained.len(), 3);
        assert_eq!(s.queue_depth(), 2);
    }
    #[test]
    fn test_drain_class_empty_returns_empty() {
        let mut s = fifo_shaper(10);
        let drained = s.drain_class(TrafficClass::RealTime, 5);
        assert!(drained.is_empty());
    }
    #[test]
    fn test_stats_dropped_count() {
        let mut s = fifo_shaper(2);
        s.enqueue(make_entry(1, TrafficClass::BulkData, 100, 1000), 1000)
            .expect("test: enqueue entry");
        s.enqueue(make_entry(2, TrafficClass::BulkData, 100, 2000), 2000)
            .expect("test: enqueue entry");
        let _ = s.enqueue(make_entry(3, TrafficClass::BulkData, 100, 3000), 3000);
        let _ = s.enqueue(make_entry(4, TrafficClass::BulkData, 100, 4000), 4000);
        let st = s.stats();
        assert_eq!(st.dropped, 2);
    }
    #[test]
    fn test_stats_class_stats_populated() {
        let mut s = fifo_shaper(10);
        s.enqueue(make_entry(1, TrafficClass::RealTime, 100, 1000), 1000)
            .expect("test: enqueue entry");
        s.enqueue(make_entry(2, TrafficClass::BulkData, 100, 2000), 2000)
            .expect("test: enqueue entry");
        s.dequeue(3000);
        s.dequeue(4000);
        let st = s.stats();
        assert!(!st.class_stats.is_empty());
        let rt = st.class_stats.iter().find(|(k, _)| k == "RealTime");
        assert!(rt.is_some());
    }
    #[test]
    fn test_stats_avg_latency_nonzero() {
        let mut s = fifo_shaper(10);
        s.enqueue(
            make_entry(1, TrafficClass::BulkData, 100, 1_000_000),
            1_000_000,
        )
        .expect("test: enqueue entry");
        s.dequeue(2_000_000);
        let st = s.stats();
        assert!(st.avg_latency_us > 0.0, "avg latency should be nonzero");
    }
    #[test]
    fn test_stats_avg_queue_depth_tracks() {
        let mut s = fifo_shaper(10);
        for i in 0..5u64 {
            s.enqueue(
                make_entry(i, TrafficClass::BulkData, 100, i * 1000),
                i * 1000,
            )
            .expect("test: enqueue entry");
        }
        let st = s.stats();
        assert!(st.avg_queue_depth > 0.0);
    }
    #[test]
    fn test_events_enqueue_dequeue_sequence() {
        let mut s = fifo_shaper(10);
        s.enqueue(make_entry(7, TrafficClass::BulkData, 100, 1000), 1000)
            .expect("test: enqueue entry");
        s.dequeue(2000);
        let events = s.drain_events();
        let has_enq = events
            .iter()
            .any(|e| matches!(e, ShaperEvent::PacketEnqueued(7)));
        let has_deq = events
            .iter()
            .any(|e| matches!(e, ShaperEvent::PacketDequeued(7)));
        assert!(has_enq);
        assert!(has_deq);
    }
    #[test]
    fn test_drain_events_clears_buffer() {
        let mut s = fifo_shaper(10);
        s.enqueue(make_entry(1, TrafficClass::BulkData, 100, 1000), 1000)
            .expect("test: enqueue entry");
        let e1 = s.drain_events();
        assert!(!e1.is_empty());
        let e2 = s.drain_events();
        assert!(e2.is_empty(), "second drain should return empty");
    }
    #[test]
    fn test_rate_limit_hit_event() {
        let mut s = token_shaper(1_000_000, 100, 10);
        s.enqueue(make_entry(1, TrafficClass::BulkData, 100, 0), 0)
            .expect("test: enqueue entry");
        let _ = s.enqueue(make_entry(2, TrafficClass::BulkData, 100, 0), 0);
        let events = s.drain_events();
        assert!(events
            .iter()
            .any(|e| matches!(e, ShaperEvent::RateLimitHit)));
    }
    #[test]
    fn test_invalid_config_zero_depth() {
        let result = TrafficShaper::new(ShaperConfig {
            discipline: QueuingDiscipline::Fifo,
            max_queue_depth: 0,
            rate_limit_bps: 0,
            burst_allowance_bytes: 0,
            drop_policy: DropPolicy::Tail,
        });
        assert!(matches!(result, Err(ShaperError::InvalidConfig(_))));
    }
    #[test]
    fn test_update_rate_non_rate_discipline_fails() {
        let mut s = fifo_shaper(10);
        let result = s.update_rate(1_000_000);
        assert!(matches!(result, Err(ShaperError::InvalidConfig(_))));
    }
    #[test]
    fn test_shaper_error_display() {
        assert!(!ShaperError::QueueFull(5).to_string().is_empty());
        assert!(!ShaperError::RateLimitExceeded(1000).to_string().is_empty());
        assert!(!ShaperError::InvalidConfig("bad".to_string())
            .to_string()
            .is_empty());
        assert!(!ShaperError::EntryNotFound(99).to_string().is_empty());
    }
    #[test]
    fn test_tokens_available_helper() {
        let t = tokens_available(0, 8_000_000, 500, 10_000);
        assert_eq!(t, 500);
    }
    #[test]
    fn test_tokens_available_capped_at_burst() {
        let t = tokens_available(0, 8_000_000, 100_000, 200);
        assert_eq!(t, 200, "should be capped at burst_bytes");
    }
    #[test]
    fn test_tokens_available_accumulates() {
        let t = tokens_available(50, 8_000_000, 50, 10_000);
        assert_eq!(t, 100);
    }
    #[test]
    fn test_replenish_tokens_updates_last_ts() {
        let mut s = token_shaper(8_000_000, 1000, 100);
        s.tokens = 0;
        s.last_replenish_us = 0;
        s.replenish_tokens(300);
        assert_eq!(s.last_replenish_us, 300);
    }
    #[test]
    fn test_xorshift64_non_zero_output() {
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
        assert_ne!(state, 0x1234_5678_9ABC_DEF0);
    }
    #[test]
    fn test_xorshift_f64_range() {
        let mut state = 0xDEAD_BEEF_CAFE_BABEu64;
        for _ in 0..1000 {
            let v = xorshift_f64(&mut state);
            assert!((0.0..1.0).contains(&v), "f64 out of [0,1): {v}");
        }
    }
    #[test]
    fn test_traffic_class_priority_order() {
        assert!(TrafficClass::RealTime.priority() > TrafficClass::Interactive.priority());
        assert!(TrafficClass::Interactive.priority() > TrafficClass::BulkData.priority());
        assert!(TrafficClass::BulkData.priority() > TrafficClass::Background.priority());
    }
    #[test]
    fn test_traffic_class_labels_unique() {
        let labels: Vec<_> = [
            TrafficClass::RealTime,
            TrafficClass::Interactive,
            TrafficClass::BulkData,
            TrafficClass::Background,
            TrafficClass::Management,
        ]
        .iter()
        .map(|c| c.label())
        .collect();
        let unique: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(labels.len(), unique.len());
    }
    fn make_token(peer: &str, class: PeerTrafficClass, bytes: u64, queued_at: u64) -> TrafficToken {
        TrafficToken {
            peer_id: peer.to_string(),
            class,
            bytes,
            queued_at,
        }
    }
    #[test]
    fn test_peer_shaper_new_empty() {
        let shaper = PeerTrafficShaper::new(65_536, 1_048_576);
        assert!(shaper.buckets.is_empty());
    }
    #[test]
    fn test_peer_shaper_enqueue_creates_bucket() {
        let mut shaper = PeerTrafficShaper::new(65_536, 1_048_576);
        shaper.buckets.insert(
            "pa".to_string(),
            PeerTokenBucket::new("pa".to_string(), 65_536, 1_048_576),
        );
        shaper
            .buckets
            .get_mut("pa")
            .expect("test: get mutable peer bucket")
            .tokens = 1000;
        shaper.enqueue(make_token("pa", PeerTrafficClass::DataTransfer, 100, 1));
        assert_eq!(shaper.buckets["pa"].queued.len(), 1);
    }
    #[test]
    fn test_peer_shaper_tick_refills() {
        let mut shaper = PeerTrafficShaper::new(65_536, 0);
        let mut b = PeerTokenBucket::new("p1".to_string(), 65_536, 1000);
        b.tokens = 0;
        shaper.buckets.insert("p1".to_string(), b);
        shaper.tick();
        assert_eq!(shaper.buckets["p1"].tokens, 1000);
    }
    #[test]
    fn test_peer_shaper_dequeue_all_priority() {
        let mut shaper = PeerTrafficShaper::new(65_536, 0);
        let mut bucket = PeerTokenBucket::new("p".to_string(), 65_536, 0);
        bucket.tokens = 10_000;
        bucket
            .queued
            .push(make_token("p", PeerTrafficClass::LowBackground, 10, 1));
        bucket
            .queued
            .push(make_token("p", PeerTrafficClass::Critical, 20, 2));
        bucket
            .queued
            .sort_by_key(|a| std::cmp::Reverse(a.class.weight()));
        shaper.buckets.insert("p".to_string(), bucket);
        let drained = shaper.dequeue_all();
        assert_eq!(drained[0].class, PeerTrafficClass::Critical);
    }
    #[test]
    fn test_peer_shaper_stats() {
        let mut shaper = PeerTrafficShaper::new(65_536, 0);
        shaper.buckets.insert(
            "a".to_string(),
            PeerTokenBucket::new("a".to_string(), 65_536, 0),
        );
        shaper.buckets.insert(
            "b".to_string(),
            PeerTokenBucket::new("b".to_string(), 65_536, 0),
        );
        let st = shaper.stats();
        assert_eq!(st.active_peers, 2);
    }
    #[test]
    fn test_peer_shaper_remove_peer() {
        let mut shaper = PeerTrafficShaper::new(65_536, 0);
        shaper.buckets.insert(
            "rm".to_string(),
            PeerTokenBucket::new("rm".to_string(), 65_536, 0),
        );
        assert!(shaper.remove_peer("rm"));
        assert!(!shaper.remove_peer("rm"));
    }
    #[test]
    fn test_pq_stats_after_dequeue() {
        let mut s = pq_shaper(4, 20);
        s.enqueue(make_entry(1, TrafficClass::RealTime, 256, 1000), 1000)
            .expect("test: enqueue entry");
        s.enqueue(make_entry(2, TrafficClass::Background, 64, 2000), 2000)
            .expect("test: enqueue entry");
        s.dequeue(3000);
        s.dequeue(4000);
        let st = s.stats();
        assert_eq!(st.dequeued, 2);
        assert_eq!(st.bytes_shaped, 320);
    }
    #[test]
    fn test_wfq_stats_after_dequeue() {
        let mut s = wfq_shaper(20);
        s.enqueue(make_entry(1, TrafficClass::RealTime, 100, 1000), 1000)
            .expect("test: enqueue entry");
        s.dequeue(2000);
        let st = s.stats();
        assert_eq!(st.dequeued, 1);
        assert_eq!(st.bytes_shaped, 100);
    }
    #[test]
    fn test_diffserv_stats_accumulate() {
        let mut s = diffserv_shaper(20);
        for i in 0..3u64 {
            s.enqueue(
                make_entry(i, TrafficClass::Interactive, 100, i * 1000),
                i * 1000,
            )
            .expect("test: enqueue entry");
        }
        for _ in 0..3 {
            s.dequeue(10_000);
        }
        let st = s.stats();
        assert_eq!(st.enqueued, 3);
        assert_eq!(st.dequeued, 3);
    }
    #[test]
    fn test_peek_pq() {
        let mut s = pq_shaper(4, 20);
        s.enqueue(make_entry(1, TrafficClass::Background, 10, 1000), 1000)
            .expect("test: enqueue entry");
        s.enqueue(make_entry(2, TrafficClass::RealTime, 10, 2000), 2000)
            .expect("test: enqueue entry");
        let peeked_id = s.peek().map(|e| e.id);
        assert_eq!(
            peeked_id,
            Some(2),
            "peek should see RealTime (highest priority)"
        );
    }
    #[test]
    fn test_peek_diffserv() {
        let mut s = diffserv_shaper(20);
        s.enqueue(make_entry(1, TrafficClass::Background, 10, 1000), 1000)
            .expect("test: enqueue entry");
        s.enqueue(make_entry(2, TrafficClass::Interactive, 10, 1000), 1000)
            .expect("test: enqueue entry");
        let peeked_id = s.peek().map(|e| e.id);
        assert_eq!(
            peeked_id,
            Some(2),
            "peek should see Interactive before Background"
        );
    }
}
