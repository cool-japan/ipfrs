//! DAG-aware Block Prefetch Scheduler V2
//!
//! This module provides an improved prefetch scheduler that uses DAG-aware look-ahead:
//! when a block is accessed, it enqueues blocks referenced by the DAG links of that block.
//!
//! Features:
//! - Priority-based queuing (Critical, High, Normal, Low)
//! - TTL-based expiration of stale prefetch requests
//! - Deduplication with priority promotion
//! - Capacity management with eviction of low-priority entries
//! - Comprehensive statistics tracking

use std::collections::HashSet;

/// Priority level for a prefetch request.
///
/// Higher numeric value = higher priority. Used for ordering in the prefetch queue.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrefetchPriority {
    /// Background speculation — lowest priority
    Low = 0,
    /// Speculative three hops from trigger
    Normal = 1,
    /// Two hops away from trigger
    High = 2,
    /// Directly linked from current access — highest priority
    Critical = 3,
}

/// A single prefetch request for a CID.
#[derive(Clone, Debug)]
pub struct PrefetchRequest {
    /// Content identifier of the block to prefetch
    pub cid: String,
    /// Priority of this prefetch request
    pub priority: PrefetchPriority,
    /// Hop distance from the trigger block
    pub depth: usize,
    /// Monotonic tick counter at time of request
    pub requested_at: u64,
}

impl PrefetchRequest {
    /// Returns `true` if this request has exceeded its TTL.
    ///
    /// # Arguments
    /// * `ttl_ticks` - Number of ticks before a request expires
    /// * `now_ticks` - Current tick counter value
    pub fn is_expired(&self, ttl_ticks: u64, now_ticks: u64) -> bool {
        now_ticks.saturating_sub(self.requested_at) >= ttl_ticks
    }
}

/// Outcome of a prefetch operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrefetchResult {
    /// Block was successfully loaded into cache
    Satisfied { cid: String },
    /// Block was already present in cache when fetched
    AlreadyCached { cid: String },
    /// Request was dequeued without fetching due to capacity pressure
    Evicted { cid: String },
    /// Fetch attempt failed
    Failed { cid: String, reason: String },
}

/// Aggregate statistics for the prefetch scheduler.
#[derive(Clone, Debug, Default)]
pub struct PrefetchStats {
    /// Total number of requests enqueued
    pub enqueued: u64,
    /// Number of requests that resulted in successful cache population
    pub satisfied: u64,
    /// Number of requests where the block was already cached
    pub already_cached: u64,
    /// Number of requests evicted due to capacity or expiry
    pub evicted: u64,
    /// Number of requests that failed during fetch
    pub failed: u64,
}

impl PrefetchStats {
    /// Cache utilization ratio: (satisfied + already_cached) / max(enqueued, 1)
    pub fn utilization(&self) -> f64 {
        (self.satisfied + self.already_cached) as f64 / self.enqueued.max(1) as f64
    }
}

/// DAG-aware block prefetch scheduler (V2).
///
/// When a block is accessed, its DAG links are enqueued for background prefetching
/// at decreasing priority levels based on their hop distance from the trigger.
pub struct BlockPrefetchSchedulerV2 {
    /// Pending prefetch requests
    queue: Vec<PrefetchRequest>,
    /// CIDs already in cache — skip re-enqueueing
    cached: HashSet<String>,
    /// CIDs currently being fetched — skip re-enqueueing
    in_flight: HashSet<String>,
    /// Maximum number of entries allowed in the queue
    max_queue_size: usize,
    /// Maximum hop depth to consider for prefetching
    max_depth: usize,
    /// Number of ticks after which a queued request expires
    ttl_ticks: u64,
    /// Aggregate statistics
    stats: PrefetchStats,
    /// Monotonic tick counter
    tick_counter: u64,
}

impl BlockPrefetchSchedulerV2 {
    /// Create a new scheduler with explicit parameters.
    ///
    /// # Arguments
    /// * `max_queue_size` - Maximum pending requests (default: 500)
    /// * `max_depth` - Maximum DAG hop depth to prefetch (default: 3)
    /// * `ttl_ticks` - Ticks before a queued request expires (default: 1000)
    pub fn new(max_queue_size: usize, max_depth: usize, ttl_ticks: u64) -> Self {
        Self {
            queue: Vec::new(),
            cached: HashSet::new(),
            in_flight: HashSet::new(),
            max_queue_size,
            max_depth,
            ttl_ticks,
            stats: PrefetchStats::default(),
            tick_counter: 0,
        }
    }

    /// Record a block access and enqueue its direct DAG links for prefetching.
    ///
    /// Only depth-1 links (directly referenced) are enqueued at `Critical` priority.
    /// Links beyond `max_depth` are silently skipped.
    pub fn on_access(&mut self, _cid: &str, dag_links: &[String]) {
        let depth = 1usize;
        if depth > self.max_depth {
            return;
        }
        let priority = match depth {
            1 => PrefetchPriority::Critical,
            2 => PrefetchPriority::High,
            3 => PrefetchPriority::Normal,
            _ => PrefetchPriority::Low,
        };
        for link in dag_links {
            let req = PrefetchRequest {
                cid: link.clone(),
                priority: priority.clone(),
                depth,
                requested_at: self.tick_counter,
            };
            self.enqueue(req);
        }
    }

    /// Enqueue a prefetch request.
    ///
    /// Deduplicates by CID — if the CID already exists in the queue, the entry
    /// with the higher priority is kept. Applies capacity management by evicting
    /// a `Low`-priority entry when the queue is at capacity.
    pub fn enqueue(&mut self, req: PrefetchRequest) {
        // Skip if already in cache or in-flight
        if self.cached.contains(&req.cid) || self.in_flight.contains(&req.cid) {
            return;
        }

        // Deduplicate: if the CID is already queued, keep the higher priority
        if let Some(pos) = self.queue.iter().position(|r| r.cid == req.cid) {
            if req.priority > self.queue[pos].priority {
                self.queue[pos].priority = req.priority;
            }
            // Either way, do not count as a new enqueue
            return;
        }

        // Enforce capacity
        if self.queue.len() >= self.max_queue_size {
            // Try to evict a Low-priority entry
            if let Some(low_pos) = self.queue.iter().position(|r| r.priority == PrefetchPriority::Low) {
                self.queue.swap_remove(low_pos);
                self.stats.evicted += 1;
            } else {
                // No Low entry to evict; drop the new request
                return;
            }
        }

        self.stats.enqueued += 1;
        self.queue.push(req);
    }

    /// Dequeue up to `n` requests, ordered by priority descending then age ascending.
    ///
    /// Expired entries are removed and counted as evicted. The returned entries
    /// are marked as in-flight.
    pub fn dequeue_batch(&mut self, n: usize) -> Vec<PrefetchRequest> {
        // Remove expired entries first
        let ttl = self.ttl_ticks;
        let now = self.tick_counter;
        let mut evicted_count = 0u64;
        self.queue.retain(|r| {
            if r.is_expired(ttl, now) {
                evicted_count += 1;
                false
            } else {
                true
            }
        });
        self.stats.evicted += evicted_count;

        // Sort: higher priority first; for equal priority, earlier request first
        self.queue.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| a.requested_at.cmp(&b.requested_at))
        });

        // Take up to n
        let take = n.min(self.queue.len());
        let batch: Vec<PrefetchRequest> = self.queue.drain(..take).collect();

        for req in &batch {
            self.in_flight.insert(req.cid.clone());
        }

        batch
    }

    /// Mark a CID as successfully fetched and cached.
    pub fn mark_satisfied(&mut self, cid: &str) {
        self.in_flight.remove(cid);
        self.cached.insert(cid.to_owned());
        self.stats.satisfied += 1;
    }

    /// Mark a CID as already present in cache (no fetch needed).
    pub fn mark_cached(&mut self, cid: &str) {
        self.cached.insert(cid.to_owned());
        self.stats.already_cached += 1;
    }

    /// Mark a CID as failed during fetch.
    pub fn mark_failed(&mut self, cid: &str, _reason: String) {
        self.in_flight.remove(cid);
        self.stats.failed += 1;
    }

    /// Advance the internal tick counter by one.
    pub fn tick(&mut self) {
        self.tick_counter += 1;
    }

    /// Returns the current number of pending prefetch requests.
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Returns a reference to the aggregate statistics.
    pub fn stats(&self) -> &PrefetchStats {
        &self.stats
    }
}

impl Default for BlockPrefetchSchedulerV2 {
    fn default() -> Self {
        Self::new(500, 3, 1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scheduler() -> BlockPrefetchSchedulerV2 {
        BlockPrefetchSchedulerV2::new(500, 3, 1000)
    }

    // Test 1: new() produces empty scheduler
    #[test]
    fn test_new_empty() {
        let s = make_scheduler();
        assert_eq!(s.queue_len(), 0);
        assert_eq!(s.stats().enqueued, 0);
        assert_eq!(s.stats().satisfied, 0);
        assert_eq!(s.stats().already_cached, 0);
        assert_eq!(s.stats().evicted, 0);
        assert_eq!(s.stats().failed, 0);
        assert_eq!(s.tick_counter, 0);
    }

    // Test 2: on_access enqueues dag_links at Critical priority
    #[test]
    fn test_on_access_enqueues_critical() {
        let mut s = make_scheduler();
        let links = vec!["cid1".to_string(), "cid2".to_string()];
        s.on_access("root", &links);
        assert_eq!(s.queue_len(), 2);
        assert_eq!(s.stats().enqueued, 2);
        // Check priorities
        for req in &s.queue {
            assert_eq!(req.priority, PrefetchPriority::Critical);
            assert_eq!(req.depth, 1);
        }
    }

    // Test 3: on_access skips links already in cache
    #[test]
    fn test_on_access_skip_cached() {
        let mut s = make_scheduler();
        s.cached.insert("cid1".to_string());
        let links = vec!["cid1".to_string(), "cid2".to_string()];
        s.on_access("root", &links);
        assert_eq!(s.queue_len(), 1);
        assert_eq!(s.stats().enqueued, 1);
    }

    // Test 4: on_access skips links already in_flight
    #[test]
    fn test_on_access_skip_in_flight() {
        let mut s = make_scheduler();
        s.in_flight.insert("cid1".to_string());
        let links = vec!["cid1".to_string(), "cid2".to_string()];
        s.on_access("root", &links);
        assert_eq!(s.queue_len(), 1);
        assert_eq!(s.stats().enqueued, 1);
    }

    // Test 5: enqueue deduplicates by cid, keeps higher priority
    #[test]
    fn test_enqueue_dedup_keep_higher_priority() {
        let mut s = make_scheduler();
        let req_low = PrefetchRequest {
            cid: "cid1".to_string(),
            priority: PrefetchPriority::Low,
            depth: 3,
            requested_at: 0,
        };
        let req_high = PrefetchRequest {
            cid: "cid1".to_string(),
            priority: PrefetchPriority::High,
            depth: 2,
            requested_at: 1,
        };
        s.enqueue(req_low);
        assert_eq!(s.queue_len(), 1);
        assert_eq!(s.stats().enqueued, 1);

        // Enqueue higher priority — no new entry, but priority is updated
        s.enqueue(req_high);
        assert_eq!(s.queue_len(), 1);
        // enqueued stays at 1 (dedup, no new entry counted)
        assert_eq!(s.stats().enqueued, 1);
        assert_eq!(s.queue[0].priority, PrefetchPriority::High);
    }

    // Test 6: enqueue evicts Low-priority entry when at max_queue_size
    #[test]
    fn test_enqueue_max_queue_evicts_low() {
        let mut s = BlockPrefetchSchedulerV2::new(3, 3, 1000);
        for i in 0..3 {
            s.enqueue(PrefetchRequest {
                cid: format!("low{i}"),
                priority: PrefetchPriority::Low,
                depth: 3,
                requested_at: i as u64,
            });
        }
        assert_eq!(s.queue_len(), 3);
        // Add one more high-priority entry — should evict one Low
        s.enqueue(PrefetchRequest {
            cid: "high1".to_string(),
            priority: PrefetchPriority::Critical,
            depth: 1,
            requested_at: 3,
        });
        assert_eq!(s.queue_len(), 3);
        assert_eq!(s.stats().evicted, 1);
        assert_eq!(s.stats().enqueued, 4);
    }

    // Test 7: enqueue skips new entry when full with no Low entries
    #[test]
    fn test_enqueue_skip_when_full_no_low() {
        let mut s = BlockPrefetchSchedulerV2::new(2, 3, 1000);
        s.enqueue(PrefetchRequest {
            cid: "crit1".to_string(),
            priority: PrefetchPriority::Critical,
            depth: 1,
            requested_at: 0,
        });
        s.enqueue(PrefetchRequest {
            cid: "crit2".to_string(),
            priority: PrefetchPriority::Critical,
            depth: 1,
            requested_at: 1,
        });
        // Queue is full with Critical entries; new Normal entry should be dropped
        s.enqueue(PrefetchRequest {
            cid: "norm1".to_string(),
            priority: PrefetchPriority::Normal,
            depth: 3,
            requested_at: 2,
        });
        assert_eq!(s.queue_len(), 2);
        assert_eq!(s.stats().enqueued, 2); // third was not enqueued
        assert_eq!(s.stats().evicted, 0);
    }

    // Test 8: dequeue_batch returns entries sorted by priority desc
    #[test]
    fn test_dequeue_batch_sorted_by_priority() {
        let mut s = make_scheduler();
        s.enqueue(PrefetchRequest {
            cid: "low".to_string(),
            priority: PrefetchPriority::Low,
            depth: 3,
            requested_at: 0,
        });
        s.enqueue(PrefetchRequest {
            cid: "critical".to_string(),
            priority: PrefetchPriority::Critical,
            depth: 1,
            requested_at: 1,
        });
        s.enqueue(PrefetchRequest {
            cid: "normal".to_string(),
            priority: PrefetchPriority::Normal,
            depth: 3,
            requested_at: 2,
        });
        let batch = s.dequeue_batch(3);
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0].priority, PrefetchPriority::Critical);
        assert_eq!(batch[1].priority, PrefetchPriority::Normal);
        assert_eq!(batch[2].priority, PrefetchPriority::Low);
    }

    // Test 9: dequeue_batch removes expired entries (counted as evicted)
    #[test]
    fn test_dequeue_batch_removes_expired() {
        let mut s = BlockPrefetchSchedulerV2::new(500, 3, 10);
        // Enqueue at tick 0
        s.enqueue(PrefetchRequest {
            cid: "old".to_string(),
            priority: PrefetchPriority::Normal,
            depth: 1,
            requested_at: 0,
        });
        // Advance ticks beyond TTL
        for _ in 0..10 {
            s.tick();
        }
        s.enqueue(PrefetchRequest {
            cid: "fresh".to_string(),
            priority: PrefetchPriority::Critical,
            depth: 1,
            requested_at: s.tick_counter,
        });
        let batch = s.dequeue_batch(10);
        // "old" should have been expired and evicted
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].cid, "fresh");
        assert_eq!(s.stats().evicted, 1);
    }

    // Test 10: dequeue_batch marks returned entries as in_flight
    #[test]
    fn test_dequeue_batch_marks_in_flight() {
        let mut s = make_scheduler();
        s.enqueue(PrefetchRequest {
            cid: "cid1".to_string(),
            priority: PrefetchPriority::High,
            depth: 2,
            requested_at: 0,
        });
        let batch = s.dequeue_batch(1);
        assert_eq!(batch.len(), 1);
        assert!(s.in_flight.contains("cid1"));
        assert_eq!(s.queue_len(), 0);
    }

    // Test 11: mark_satisfied moves cid to cached and increments stats
    #[test]
    fn test_mark_satisfied() {
        let mut s = make_scheduler();
        s.in_flight.insert("cid1".to_string());
        s.mark_satisfied("cid1");
        assert!(!s.in_flight.contains("cid1"));
        assert!(s.cached.contains("cid1"));
        assert_eq!(s.stats().satisfied, 1);
    }

    // Test 12: mark_cached increments already_cached
    #[test]
    fn test_mark_cached() {
        let mut s = make_scheduler();
        s.mark_cached("cid1");
        assert!(s.cached.contains("cid1"));
        assert_eq!(s.stats().already_cached, 1);
    }

    // Test 13: mark_failed removes from in_flight and increments failed
    #[test]
    fn test_mark_failed() {
        let mut s = make_scheduler();
        s.in_flight.insert("cid1".to_string());
        s.mark_failed("cid1", "network error".to_string());
        assert!(!s.in_flight.contains("cid1"));
        assert_eq!(s.stats().failed, 1);
    }

    // Test 14: tick increments tick_counter
    #[test]
    fn test_tick_increments_counter() {
        let mut s = make_scheduler();
        assert_eq!(s.tick_counter, 0);
        s.tick();
        assert_eq!(s.tick_counter, 1);
        s.tick();
        assert_eq!(s.tick_counter, 2);
    }

    // Test 15: is_expired before and after TTL
    #[test]
    fn test_is_expired_before_and_after_ttl() {
        let req = PrefetchRequest {
            cid: "cid1".to_string(),
            priority: PrefetchPriority::Normal,
            depth: 1,
            requested_at: 100,
        };
        // Not yet expired: now = 109, ttl = 10 => elapsed = 9 < 10
        assert!(!req.is_expired(10, 109));
        // Exactly at TTL boundary: now = 110, elapsed = 10 >= 10 => expired
        assert!(req.is_expired(10, 110));
        // Well past TTL
        assert!(req.is_expired(10, 200));
    }

    // Test 16: utilization calculation
    #[test]
    fn test_utilization_calculation() {
        let mut stats = PrefetchStats::default();
        // 0 enqueued => utilization = 0/1 = 0.0
        assert!((stats.utilization() - 0.0).abs() < f64::EPSILON);

        stats.enqueued = 10;
        stats.satisfied = 6;
        stats.already_cached = 2;
        // (6 + 2) / 10 = 0.8
        assert!((stats.utilization() - 0.8).abs() < f64::EPSILON);
    }

    // Test 17: queue_len decreases after dequeue
    #[test]
    fn test_queue_len_decreases_after_dequeue() {
        let mut s = make_scheduler();
        for i in 0..5 {
            s.enqueue(PrefetchRequest {
                cid: format!("cid{i}"),
                priority: PrefetchPriority::Normal,
                depth: 1,
                requested_at: i as u64,
            });
        }
        assert_eq!(s.queue_len(), 5);
        s.dequeue_batch(3);
        assert_eq!(s.queue_len(), 2);
        s.dequeue_batch(10);
        assert_eq!(s.queue_len(), 0);
    }

    // Test 18: on_access with empty dag_links enqueues nothing
    #[test]
    fn test_on_access_empty_links() {
        let mut s = make_scheduler();
        s.on_access("root", &[]);
        assert_eq!(s.queue_len(), 0);
        assert_eq!(s.stats().enqueued, 0);
    }

    // Test 19: dequeue_batch with n=0 returns empty vec
    #[test]
    fn test_dequeue_batch_zero() {
        let mut s = make_scheduler();
        s.enqueue(PrefetchRequest {
            cid: "cid1".to_string(),
            priority: PrefetchPriority::Critical,
            depth: 1,
            requested_at: 0,
        });
        let batch = s.dequeue_batch(0);
        assert!(batch.is_empty());
        assert_eq!(s.queue_len(), 1);
    }
}
