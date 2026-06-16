//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use std::time::{Duration, Instant};

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    use tokio::time::sleep;
    #[tokio::test]
    async fn test_rate_limiter_creation() {
        let limiter = ConnectionRateLimiter::new(RateLimiterConfig::default());
        let stats = limiter.stats();
        assert_eq!(stats.total_attempts, 0);
    }
    #[tokio::test]
    async fn test_allow_connection() {
        let limiter = ConnectionRateLimiter::new(RateLimiterConfig::default());
        let allowed = limiter.allow_connection("peer1").await;
        assert!(allowed);
        let stats = limiter.stats();
        assert_eq!(stats.allowed, 1);
    }
    #[tokio::test]
    async fn test_rate_limiting() {
        let config = RateLimiterConfig {
            max_rate: 10.0,
            burst_size: 5,
            ..Default::default()
        };
        let limiter = ConnectionRateLimiter::new(config);
        for _ in 0..5 {
            assert!(limiter.allow_connection("peer1").await);
        }
        let allowed = limiter.allow_connection("peer1").await;
        assert!(!allowed);
    }
    #[tokio::test]
    async fn test_per_peer_limits() {
        let config = RateLimiterConfig {
            max_rate: 100.0,
            burst_size: 100,
            enable_per_peer_limits: true,
            max_per_peer_rate: 5.0,
            peer_window: Duration::from_secs(1),
            ..Default::default()
        };
        let limiter = ConnectionRateLimiter::new(config);
        for _ in 0..5 {
            assert!(limiter.allow_connection("peer1").await);
        }
        let allowed = limiter.allow_connection("peer1").await;
        assert!(!allowed);
        assert!(limiter.allow_connection("peer2").await);
    }
    #[tokio::test]
    async fn test_priority() {
        let config = RateLimiterConfig {
            max_rate: 10.0,
            burst_size: 2,
            ..Default::default()
        };
        let limiter = ConnectionRateLimiter::new(config);
        assert!(
            limiter
                .allow_connection_with_priority("peer1", ConnectionPriority::Critical)
                .await
        );
        assert!(
            limiter
                .allow_connection_with_priority("peer2", ConnectionPriority::Critical)
                .await
        );
        let stats = limiter.stats();
        assert!(stats.tokens_available > 0.0);
    }
    #[tokio::test]
    async fn test_queuing() {
        let config = RateLimiterConfig {
            max_rate: 1.0,
            burst_size: 1,
            enable_queuing: true,
            max_queue_size: 10,
            ..Default::default()
        };
        let limiter = ConnectionRateLimiter::new(config);
        assert!(limiter.allow_connection("peer1").await);
        assert!(!limiter.allow_connection("peer2").await);
        assert!(!limiter.allow_connection("peer3").await);
        let stats = limiter.stats();
        assert_eq!(stats.queued, 2);
    }
    #[tokio::test]
    async fn test_process_queue() {
        let config = RateLimiterConfig {
            max_rate: 10.0,
            burst_size: 1,
            enable_queuing: true,
            ..Default::default()
        };
        let limiter = ConnectionRateLimiter::new(config);
        limiter.allow_connection("peer1").await;
        limiter.allow_connection("peer2").await;
        limiter.allow_connection("peer3").await;
        sleep(Duration::from_millis(200)).await;
        let allowed = limiter.process_queue().await;
        assert!(!allowed.is_empty());
    }
    #[tokio::test]
    async fn test_success_failure_recording() {
        let config = RateLimiterConfig {
            enable_per_peer_limits: true,
            ..Default::default()
        };
        let limiter = ConnectionRateLimiter::new(config);
        limiter.allow_connection("peer1").await;
        limiter.record_success("peer1");
        let (successes, failures, _) = limiter
            .peer_stats("peer1")
            .expect("test: peer1 stats should exist after allow_connection");
        assert_eq!(successes, 1);
        assert_eq!(failures, 0);
    }
    #[tokio::test]
    async fn test_config_presets() {
        let conservative = RateLimiterConfig::conservative();
        assert!(conservative.max_rate < 10.0);
        let permissive = RateLimiterConfig::permissive();
        assert!(permissive.max_rate > 10.0);
        let adaptive = RateLimiterConfig::adaptive();
        assert!(adaptive.enable_adaptive);
    }
    #[tokio::test]
    async fn test_reset() {
        let limiter = ConnectionRateLimiter::new(RateLimiterConfig::default());
        limiter.allow_connection("peer1").await;
        assert_eq!(limiter.stats().allowed, 1);
        limiter.reset();
        assert_eq!(limiter.stats().allowed, 0);
    }
    #[tokio::test]
    async fn test_token_refill() {
        let config = RateLimiterConfig {
            max_rate: 10.0,
            burst_size: 5,
            ..Default::default()
        };
        let limiter = ConnectionRateLimiter::new(config);
        for _ in 0..5 {
            limiter.allow_connection("peer1").await;
        }
        sleep(Duration::from_millis(200)).await;
        assert!(limiter.allow_connection("peer1").await);
    }
    #[test]
    fn test_atomic_token_bucket_try_acquire_success() {
        let bucket = AtomicTokenBucket::new(100, 10);
        assert!(
            bucket.try_acquire(50),
            "should succeed when tokens available"
        );
        assert_eq!(bucket.available_tokens(), 50);
    }
    #[test]
    fn test_atomic_token_bucket_try_acquire_failure() {
        let bucket = AtomicTokenBucket::new(10, 1);
        assert!(bucket.try_acquire(10));
        assert!(!bucket.try_acquire(1), "should fail when no tokens remain");
    }
    #[test]
    fn test_atomic_token_bucket_try_acquire_exact() {
        let bucket = AtomicTokenBucket::new(5, 1);
        assert!(bucket.try_acquire(5));
        assert_eq!(bucket.available_tokens(), 0);
        assert!(!bucket.try_acquire(1));
    }
    #[test]
    fn test_atomic_token_bucket_refill() {
        let bucket = AtomicTokenBucket::new(100, 1_000);
        bucket.try_acquire(100);
        assert_eq!(bucket.available_tokens(), 0);
        {
            let mut last = bucket.last_refill.lock().unwrap_or_else(|e| e.into_inner());
            *last = Instant::now() - Duration::from_secs(1);
        }
        bucket.refill();
        assert_eq!(bucket.available_tokens(), 100);
    }
    #[test]
    fn test_atomic_token_bucket_fill_ratio_full() {
        let bucket = AtomicTokenBucket::new(100, 10);
        let ratio = bucket.fill_ratio();
        assert!(
            (ratio - 1.0).abs() < 1e-6,
            "full bucket should have ratio 1.0"
        );
    }
    #[test]
    fn test_atomic_token_bucket_fill_ratio_half() {
        let bucket = AtomicTokenBucket::new(100, 10);
        bucket.try_acquire(50);
        let ratio = bucket.fill_ratio();
        assert!(
            (ratio - 0.5).abs() < 0.01,
            "ratio should be ~0.5 after draining half"
        );
    }
    #[test]
    fn test_atomic_token_bucket_fill_ratio_empty() {
        let bucket = AtomicTokenBucket::new(100, 10);
        bucket.try_acquire(100);
        let ratio = bucket.fill_ratio();
        assert!(ratio < 0.01, "empty bucket fill_ratio should be near 0");
    }
    #[test]
    fn test_rate_limiter_allowed() {
        let rl = RateLimiter::new(1_000, 100);
        let decision = rl.check("peer_a", 1);
        assert_eq!(decision, RateLimitDecision::Allowed);
        assert_eq!(rl.stats_snapshot().total_allowed, 1);
    }
    #[test]
    fn test_rate_limiter_global_exhaustion_returns_throttled() {
        let rl = RateLimiter::new(1, 1);
        assert_eq!(rl.check("peer_a", 1), RateLimitDecision::Allowed);
        let decision = rl.check("peer_a", 1);
        assert!(
            matches!(decision, RateLimitDecision::Throttled { .. }),
            "expected Throttled, got {decision:?}"
        );
        assert_eq!(rl.stats_snapshot().total_throttled, 1);
    }
    #[test]
    fn test_rate_limiter_per_peer_throttled() {
        let rl = RateLimiter::with_defaults(10_000, 100, 2, 1);
        assert_eq!(rl.check("peer_b", 1), RateLimitDecision::Allowed);
        assert_eq!(rl.check("peer_b", 1), RateLimitDecision::Allowed);
        let decision = rl.check("peer_b", 1);
        assert!(
            matches!(decision, RateLimitDecision::Throttled { .. }),
            "expected Throttled on per-peer exhaustion, got {decision:?}"
        );
    }
    #[test]
    fn test_rate_limiter_per_peer_isolation() {
        let rl = RateLimiter::with_defaults(10_000, 100, 2, 1);
        rl.check("peer_c", 2);
        assert_eq!(rl.check("peer_d", 1), RateLimitDecision::Allowed);
    }
    #[test]
    fn test_rate_limiter_global_limits_regardless_of_peer() {
        let rl = RateLimiter::with_defaults(1, 1, 10_000, 100);
        assert_eq!(rl.check("peer_e", 1), RateLimitDecision::Allowed);
        let decision = rl.check("peer_f", 1);
        assert!(
            matches!(decision, RateLimitDecision::Throttled { .. }),
            "global limit should block fresh peer, got {decision:?}"
        );
    }
    #[test]
    fn test_rate_limiter_register_peer() {
        let rl = RateLimiter::new(1_000, 100);
        rl.register_peer("vip", 5_000, 500);
        assert_eq!(rl.check("vip", 1_000), RateLimitDecision::Allowed);
        assert_eq!(rl.peer_count(), 1);
    }
    #[test]
    fn test_rate_limiter_remove_peer() {
        let rl = RateLimiter::new(1_000, 100);
        rl.register_peer("temp", 100, 10);
        assert_eq!(rl.peer_count(), 1);
        rl.remove_peer("temp");
        assert_eq!(rl.peer_count(), 0);
    }
    #[test]
    fn test_rate_limiter_stats_accumulation() {
        let rl = RateLimiter::with_defaults(10, 1, 10, 1);
        for _ in 0..10 {
            rl.check("peer_s", 1);
        }
        let snap = rl.stats_snapshot();
        assert_eq!(snap.total_allowed, 10);
        assert_eq!(snap.total_throttled, 0);
        assert_eq!(snap.total_rejected, 0);
    }
    #[test]
    fn test_backpressure_initial_state_drain() {
        let bp = BackpressureController::new();
        assert_eq!(bp.signal(), BackpressureSignal::Drain);
        assert_eq!(bp.depth(), 0);
    }
    #[test]
    fn test_backpressure_push_pop_depth() {
        let bp = BackpressureController::new();
        bp.push();
        bp.push();
        assert_eq!(bp.depth(), 2);
        bp.pop();
        assert_eq!(bp.depth(), 1);
        bp.pop();
        assert_eq!(bp.depth(), 0);
    }
    #[test]
    fn test_backpressure_pop_saturating() {
        let bp = BackpressureController::new();
        bp.pop();
        assert_eq!(bp.depth(), 0);
    }
    #[test]
    fn test_backpressure_signal_normal() {
        let bp = BackpressureController::new();
        for _ in 0..3_000 {
            bp.push();
        }
        assert_eq!(bp.signal(), BackpressureSignal::Normal);
    }
    #[test]
    fn test_backpressure_signal_drain() {
        let bp = BackpressureController::new();
        assert_eq!(bp.signal(), BackpressureSignal::Drain);
    }
    #[test]
    fn test_backpressure_signal_backpressure() {
        let bp = BackpressureController::new();
        for _ in 0..8_000 {
            bp.push();
        }
        assert_eq!(bp.signal(), BackpressureSignal::Backpressure);
    }
    #[test]
    fn test_backpressure_signal_full() {
        let bp = BackpressureController::new();
        for _ in 0..10_000 {
            bp.push();
        }
        assert_eq!(bp.signal(), BackpressureSignal::Full);
    }
    #[test]
    fn test_backpressure_push_returns_correct_signal() {
        let bp = BackpressureController::with_config(10, 2, 7);
        assert_eq!(bp.push(), BackpressureSignal::Drain);
        assert_eq!(bp.push(), BackpressureSignal::Normal);
        for _ in 0..5 {
            bp.push();
        }
        assert_eq!(bp.signal(), BackpressureSignal::Backpressure);
        for _ in 0..3 {
            bp.push();
        }
        assert_eq!(bp.signal(), BackpressureSignal::Full);
    }
    #[test]
    fn test_backpressure_custom_config() {
        let bp = BackpressureController::with_config(100, 10, 80);
        assert_eq!(bp.signal(), BackpressureSignal::Drain);
        for _ in 0..50 {
            bp.push();
        }
        assert_eq!(bp.signal(), BackpressureSignal::Normal);
    }
}
#[cfg(test)]
mod peer_rate_limiter_tests {
    use super::*;
    fn default_limiter() -> PeerRateLimiter {
        PeerRateLimiter::new(PeerRateLimiterConfig::default())
    }
    #[test]
    fn test_new_starts_empty() {
        let rl = default_limiter();
        assert_eq!(rl.peers.len(), 0);
    }
    #[test]
    fn test_check_auto_creates_peer() {
        let mut rl = default_limiter();
        let _ = rl.check("peer_a", 1);
        assert!(rl.peers.contains_key("peer_a"));
    }
    #[test]
    fn test_check_allowed() {
        let mut rl = default_limiter();
        let result = rl.check("peer_b", 1);
        assert!(matches!(result, RateLimitResult::Allowed { .. }));
    }
    #[test]
    fn test_check_throttled_peer() {
        let config = PeerRateLimiterConfig {
            per_peer_max_tokens: 5,
            per_peer_refill_rate: 1,
            global_max_tokens: 100_000,
            global_refill_rate: 10_000,
            auto_block_threshold: 10,
        };
        let mut rl = PeerRateLimiter::new(config);
        let _ = rl.check("peer_c", 5);
        let result = rl.check("peer_c", 1);
        assert!(matches!(result, RateLimitResult::Throttled { .. }));
    }
    #[test]
    fn test_check_blocked_peer() {
        let mut rl = default_limiter();
        rl.block_peer("peer_d");
        let result = rl.check("peer_d", 1);
        assert_eq!(result, RateLimitResult::Blocked);
    }
    #[test]
    fn test_global_throttles_when_exhausted() {
        let config = PeerRateLimiterConfig {
            per_peer_max_tokens: 100_000,
            per_peer_refill_rate: 10_000,
            global_max_tokens: 3,
            global_refill_rate: 1,
            auto_block_threshold: 10,
        };
        let mut rl = PeerRateLimiter::new(config);
        let _ = rl.check("peer_e", 3);
        let result = rl.check("peer_e", 1);
        assert_eq!(
            result,
            RateLimitResult::Throttled {
                retry_after_ticks: 1
            }
        );
    }
    #[test]
    fn test_auto_block_at_10_violations() {
        let config = PeerRateLimiterConfig {
            per_peer_max_tokens: 0,
            per_peer_refill_rate: 0,
            global_max_tokens: 100_000,
            global_refill_rate: 10_000,
            auto_block_threshold: 10,
        };
        let mut rl = PeerRateLimiter::new(config);
        for _ in 0..10 {
            rl.check("peer_f", 1);
        }
        let result = rl.check("peer_f", 1);
        assert_eq!(result, RateLimitResult::Blocked);
    }
    #[test]
    fn test_refill_adds_tokens_up_to_max() {
        let mut peer = PeerLimiter::new("p".to_string(), 100, 50);
        peer.tokens = 0;
        peer.refill();
        assert_eq!(peer.tokens, 50);
        peer.refill();
        assert_eq!(peer.tokens, 100);
        peer.refill();
        assert_eq!(peer.tokens, 100);
    }
    #[test]
    fn test_refill_decrements_violations_when_half_full() {
        let mut peer = PeerLimiter::new("p".to_string(), 100, 100);
        peer.consecutive_violations = 3;
        peer.tokens = 0;
        peer.refill();
        assert_eq!(peer.consecutive_violations, 2);
    }
    #[test]
    fn test_tick_refills_all_peers() {
        let config = PeerRateLimiterConfig {
            per_peer_max_tokens: 100,
            per_peer_refill_rate: 10,
            global_max_tokens: 100_000,
            global_refill_rate: 10_000,
            auto_block_threshold: 10,
        };
        let mut rl = PeerRateLimiter::new(config);
        let _ = rl.check("p1", 100);
        let _ = rl.check("p2", 100);
        rl.tick();
        assert_eq!(rl.peers["p1"].tokens, 10);
        assert_eq!(rl.peers["p2"].tokens, 10);
    }
    #[test]
    fn test_block_peer_sets_blocked() {
        let mut rl = default_limiter();
        rl.block_peer("peer_g");
        assert!(rl.peers["peer_g"].blocked);
    }
    #[test]
    fn test_unblock_peer_resets() {
        let mut rl = default_limiter();
        rl.block_peer("peer_h");
        rl.peers
            .get_mut("peer_h")
            .expect("test: peer_h should exist after block_peer")
            .consecutive_violations = 5;
        let ok = rl.unblock_peer("peer_h");
        assert!(ok);
        assert!(!rl.peers["peer_h"].blocked);
        assert_eq!(rl.peers["peer_h"].consecutive_violations, 0);
    }
    #[test]
    fn test_unblock_peer_unknown() {
        let mut rl = default_limiter();
        assert!(!rl.unblock_peer("ghost"));
    }
    #[test]
    fn test_remove_peer_true_false() {
        let mut rl = default_limiter();
        let _ = rl.check("peer_i", 1);
        assert!(rl.remove_peer("peer_i"));
        assert!(!rl.remove_peer("peer_i"));
    }
    #[test]
    fn test_retry_after_ticks_correct() {
        let mut peer = PeerLimiter::new("p".to_string(), 10, 3);
        peer.tokens = 0;
        let result = peer.try_consume(10);
        assert_eq!(
            result,
            RateLimitResult::Throttled {
                retry_after_ticks: 4
            }
        );
    }
    #[test]
    fn test_total_allowed_increments() {
        let mut rl = default_limiter();
        rl.check("peer_j", 1);
        rl.check("peer_j", 1);
        assert_eq!(rl.peers["peer_j"].total_allowed, 2);
    }
    #[test]
    fn test_total_throttled_increments() {
        let config = PeerRateLimiterConfig {
            per_peer_max_tokens: 1,
            per_peer_refill_rate: 1,
            global_max_tokens: 100_000,
            global_refill_rate: 10_000,
            auto_block_threshold: 10,
        };
        let mut rl = PeerRateLimiter::new(config);
        rl.check("peer_k", 1);
        rl.check("peer_k", 1);
        rl.check("peer_k", 1);
        assert_eq!(rl.peers["peer_k"].total_throttled, 2);
    }
    #[test]
    fn test_stats_total_peers() {
        let mut rl = default_limiter();
        rl.check("p1", 1);
        rl.check("p2", 1);
        rl.check("p3", 1);
        assert_eq!(rl.stats().total_peers, 3);
    }
    #[test]
    fn test_stats_blocked_peers() {
        let mut rl = default_limiter();
        rl.check("p1", 1);
        rl.check("p2", 1);
        rl.block_peer("p1");
        assert_eq!(rl.stats().blocked_peers, 1);
    }
    #[test]
    fn test_stats_aggregated() {
        let config = PeerRateLimiterConfig {
            per_peer_max_tokens: 2,
            per_peer_refill_rate: 1,
            global_max_tokens: 100_000,
            global_refill_rate: 10_000,
            auto_block_threshold: 10,
        };
        let mut rl = PeerRateLimiter::new(config);
        rl.check("p1", 1);
        rl.check("p1", 1);
        rl.check("p1", 1);
        rl.check("p2", 1);
        let s = rl.stats();
        assert_eq!(s.total_allowed, 3);
        assert_eq!(s.total_throttled, 1);
    }
    #[test]
    fn test_try_consume_ceil_division() {
        let mut peer = PeerLimiter::new("p".to_string(), 100, 7);
        peer.tokens = 1;
        let result = peer.try_consume(8);
        assert_eq!(
            result,
            RateLimitResult::Throttled {
                retry_after_ticks: 1
            }
        );
        let mut peer2 = PeerLimiter::new("p".to_string(), 100, 7);
        peer2.tokens = 0;
        let result2 = peer2.try_consume(8);
        assert_eq!(
            result2,
            RateLimitResult::Throttled {
                retry_after_ticks: 2
            }
        );
    }
    #[test]
    fn test_allowed_tokens_remaining() {
        let mut peer = PeerLimiter::new("p".to_string(), 100, 10);
        let result = peer.try_consume(30);
        assert_eq!(
            result,
            RateLimitResult::Allowed {
                tokens_remaining: 70
            }
        );
    }
}
