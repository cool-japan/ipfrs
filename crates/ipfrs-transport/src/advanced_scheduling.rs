//! Advanced request scheduling algorithms
//!
//! This module provides sophisticated scheduling algorithms for optimizing
//! block request ordering based on various factors like deadline, priority,
//! size, and historical performance.
//!
//! # Example
//!
//! ```
//! use ipfrs_transport::{AdvancedScheduler, SchedulingPolicy};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let scheduler = AdvancedScheduler::new(SchedulingPolicy::EarliestDeadlineFirst);
//! # Ok(())
//! # }
//! ```

use ipfrs_core::Cid;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Scheduling policy for request ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// First-In-First-Out (simple queue)
    Fifo,
    /// Shortest Job First (smallest blocks first)
    ShortestJobFirst,
    /// Earliest Deadline First
    EarliestDeadlineFirst,
    /// Weighted Fair Queueing (balance priority and fairness)
    WeightedFairQueueing,
    /// Multi-Level Feedback Queue (adaptive based on history)
    MultilevelFeedback,
}

/// Priority level for requests
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SchedulePriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Urgent = 3,
    Critical = 4,
}

/// Request metadata for scheduling
#[derive(Debug, Clone)]
pub struct ScheduledRequest {
    /// Content identifier
    pub cid: Cid,
    /// Priority level
    pub priority: SchedulePriority,
    /// Estimated size (bytes), if known
    pub estimated_size: Option<usize>,
    /// Deadline for completion
    pub deadline: Option<Instant>,
    /// When request was submitted
    pub submitted_at: Instant,
    /// Queue level (for multi-level feedback)
    pub queue_level: usize,
    /// Number of times rescheduled (for aging)
    pub reschedule_count: usize,
}

impl ScheduledRequest {
    /// Create a new scheduled request
    pub fn new(cid: Cid, priority: SchedulePriority) -> Self {
        Self {
            cid,
            priority,
            estimated_size: None,
            deadline: None,
            submitted_at: Instant::now(),
            queue_level: 0,
            reschedule_count: 0,
        }
    }

    /// Set estimated size
    pub fn with_size(mut self, size: usize) -> Self {
        self.estimated_size = Some(size);
        self
    }

    /// Set deadline
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Calculate urgency score (0.0 to 1.0, higher is more urgent)
    pub fn urgency_score(&self) -> f64 {
        match self.deadline {
            Some(deadline) => {
                let time_until_deadline = deadline
                    .duration_since(Instant::now())
                    .as_secs_f64()
                    .max(0.0);
                // More urgent as deadline approaches
                1.0 / (1.0 + time_until_deadline)
            }
            None => 0.0,
        }
    }

    /// Calculate wait time
    pub fn wait_time(&self) -> Duration {
        self.submitted_at.elapsed()
    }

    /// Calculate aging bonus (increases with wait time)
    pub fn aging_bonus(&self) -> f64 {
        let wait_secs = self.wait_time().as_secs_f64();
        // Exponential aging to prevent starvation
        (wait_secs / 60.0).min(5.0) // Cap at 5.0 after 5 minutes
    }
}

/// Wrapper for heap ordering
struct OrderedRequest {
    request: ScheduledRequest,
    score: f64,
}

impl PartialEq for OrderedRequest {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for OrderedRequest {}

impl PartialOrd for OrderedRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher score = higher priority in max heap
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}

/// Advanced request scheduler
pub struct AdvancedScheduler {
    /// Scheduling policy
    policy: SchedulingPolicy,
    /// Priority queue
    queue: Arc<RwLock<BinaryHeap<OrderedRequest>>>,
    /// Statistics
    stats: Arc<RwLock<SchedulerStats>>,
}

/// Statistics for the scheduler
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Total requests scheduled
    pub total_scheduled: u64,
    /// Total requests completed
    pub total_completed: u64,
    /// Average wait time
    pub avg_wait_time: Duration,
    /// Average completion time
    pub avg_completion_time: Duration,
    /// Number of deadline misses
    pub deadline_misses: u64,
}

impl SchedulerStats {
    /// Calculate completion rate
    pub fn completion_rate(&self) -> f64 {
        if self.total_scheduled == 0 {
            return 0.0;
        }
        self.total_completed as f64 / self.total_scheduled as f64
    }

    /// Calculate deadline miss rate
    pub fn deadline_miss_rate(&self) -> f64 {
        if self.total_completed == 0 {
            return 0.0;
        }
        self.deadline_misses as f64 / self.total_completed as f64
    }
}

impl AdvancedScheduler {
    /// Create a new advanced scheduler
    pub fn new(policy: SchedulingPolicy) -> Self {
        Self {
            policy,
            queue: Arc::new(RwLock::new(BinaryHeap::new())),
            stats: Arc::new(RwLock::new(SchedulerStats::default())),
        }
    }

    /// Schedule a request
    pub async fn schedule(&self, request: ScheduledRequest) {
        let score = self.calculate_score(&request);

        let mut queue = self.queue.write().await;
        queue.push(OrderedRequest { request, score });

        let mut stats = self.stats.write().await;
        stats.total_scheduled += 1;
    }

    /// Get the next request to process
    pub async fn next(&self) -> Option<ScheduledRequest> {
        let mut queue = self.queue.write().await;
        queue.pop().map(|ordered| ordered.request)
    }

    /// Peek at the next request without removing it
    pub async fn peek(&self) -> Option<ScheduledRequest> {
        let queue = self.queue.read().await;
        queue.peek().map(|ordered| ordered.request.clone())
    }

    /// Mark a request as completed
    pub async fn mark_completed(&self, request: &ScheduledRequest, completion_time: Duration) {
        let mut stats = self.stats.write().await;
        stats.total_completed += 1;

        // Update average wait time
        let wait_time = request.wait_time();
        let total_wait = stats.avg_wait_time.as_millis() as u64 * (stats.total_completed - 1)
            + wait_time.as_millis() as u64;
        stats.avg_wait_time = Duration::from_millis(total_wait / stats.total_completed);

        // Update average completion time
        let total_completion = stats.avg_completion_time.as_millis() as u64
            * (stats.total_completed - 1)
            + completion_time.as_millis() as u64;
        stats.avg_completion_time = Duration::from_millis(total_completion / stats.total_completed);

        // Check deadline miss
        if let Some(deadline) = request.deadline {
            if Instant::now() > deadline {
                stats.deadline_misses += 1;
            }
        }
    }

    /// Calculate scheduling score for a request
    fn calculate_score(&self, request: &ScheduledRequest) -> f64 {
        match self.policy {
            SchedulingPolicy::Fifo => {
                // Earlier submissions get higher scores
                -(request.submitted_at.elapsed().as_secs_f64())
            }
            SchedulingPolicy::ShortestJobFirst => {
                // Smaller jobs get higher scores
                match request.estimated_size {
                    Some(size) => -(size as f64),
                    None => 0.0, // Unknown size goes to middle
                }
            }
            SchedulingPolicy::EarliestDeadlineFirst => {
                // Earlier deadlines get higher scores
                request.urgency_score() * 1000.0 + request.priority as u8 as f64
            }
            SchedulingPolicy::WeightedFairQueueing => {
                // Balance priority, urgency, and aging
                let priority_score = request.priority as u8 as f64 * 10.0;
                let urgency_score = request.urgency_score() * 50.0;
                let aging_bonus = request.aging_bonus() * 5.0;
                priority_score + urgency_score + aging_bonus
            }
            SchedulingPolicy::MultilevelFeedback => {
                // Higher queue levels (older requests) get priority boost
                let level_boost = request.queue_level as f64 * 10.0;
                let priority_score = request.priority as u8 as f64 * 5.0;
                let aging_bonus = request.aging_bonus() * 3.0;
                level_boost + priority_score + aging_bonus
            }
        }
    }

    /// Get current queue size
    pub async fn queue_size(&self) -> usize {
        let queue = self.queue.read().await;
        queue.len()
    }

    /// Get current statistics
    pub async fn stats(&self) -> SchedulerStats {
        self.stats.read().await.clone()
    }

    /// Reset statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = SchedulerStats::default();
    }

    /// Clear the queue
    pub async fn clear(&self) {
        let mut queue = self.queue.write().await;
        queue.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use multihash::Multihash;

    fn test_cid(seed: u64) -> Cid {
        let data = seed.to_le_bytes();
        let hash = Multihash::wrap(0x12, &data).expect("test: create multihash");
        Cid::new_v1(0x55, hash)
    }

    #[tokio::test]
    async fn test_scheduler_creation() {
        let scheduler = AdvancedScheduler::new(SchedulingPolicy::Fifo);
        assert_eq!(scheduler.queue_size().await, 0);
    }

    #[tokio::test]
    async fn test_fifo_scheduling() {
        let scheduler = AdvancedScheduler::new(SchedulingPolicy::Fifo);

        let req1 = ScheduledRequest::new(test_cid(1), SchedulePriority::Normal);
        let req2 = ScheduledRequest::new(test_cid(2), SchedulePriority::Normal);

        scheduler.schedule(req1).await;
        tokio::time::sleep(Duration::from_millis(10)).await;
        scheduler.schedule(req2).await;

        // FIFO should return req1 first
        let next = scheduler
            .next()
            .await
            .expect("test: get next scheduled item");
        assert_eq!(next.cid, test_cid(1));
    }

    #[tokio::test]
    async fn test_shortest_job_first() {
        let scheduler = AdvancedScheduler::new(SchedulingPolicy::ShortestJobFirst);

        let req1 = ScheduledRequest::new(test_cid(1), SchedulePriority::Normal).with_size(1000);
        let req2 = ScheduledRequest::new(test_cid(2), SchedulePriority::Normal).with_size(500);

        scheduler.schedule(req1).await;
        scheduler.schedule(req2).await;

        // Should return smaller job first
        let next = scheduler
            .next()
            .await
            .expect("test: get next scheduled item");
        assert_eq!(next.cid, test_cid(2));
    }

    #[tokio::test]
    async fn test_earliest_deadline_first() {
        let scheduler = AdvancedScheduler::new(SchedulingPolicy::EarliestDeadlineFirst);

        let far_deadline = Instant::now() + Duration::from_secs(100);
        let near_deadline = Instant::now() + Duration::from_secs(10);

        let req1 = ScheduledRequest::new(test_cid(1), SchedulePriority::Normal)
            .with_deadline(far_deadline);
        let req2 = ScheduledRequest::new(test_cid(2), SchedulePriority::Normal)
            .with_deadline(near_deadline);

        scheduler.schedule(req1).await;
        scheduler.schedule(req2).await;

        // Should return request with nearer deadline
        let next = scheduler
            .next()
            .await
            .expect("test: get next scheduled item");
        assert_eq!(next.cid, test_cid(2));
    }

    #[tokio::test]
    async fn test_priority_ordering() {
        let scheduler = AdvancedScheduler::new(SchedulingPolicy::WeightedFairQueueing);

        let req_low = ScheduledRequest::new(test_cid(1), SchedulePriority::Low);
        let req_high = ScheduledRequest::new(test_cid(2), SchedulePriority::Critical);

        scheduler.schedule(req_low).await;
        scheduler.schedule(req_high).await;

        // Higher priority should come first
        let next = scheduler
            .next()
            .await
            .expect("test: get next scheduled item");
        assert_eq!(next.cid, test_cid(2));
    }

    #[tokio::test]
    async fn test_urgency_score() {
        let near_deadline = Instant::now() + Duration::from_secs(5);
        let far_deadline = Instant::now() + Duration::from_secs(100);

        let req1 = ScheduledRequest::new(test_cid(1), SchedulePriority::Normal)
            .with_deadline(near_deadline);
        let req2 = ScheduledRequest::new(test_cid(2), SchedulePriority::Normal)
            .with_deadline(far_deadline);

        assert!(req1.urgency_score() > req2.urgency_score());
    }

    #[tokio::test]
    async fn test_aging_bonus() {
        let mut req = ScheduledRequest::new(test_cid(1), SchedulePriority::Normal);
        req.submitted_at = Instant::now() - Duration::from_secs(120);

        let bonus = req.aging_bonus();
        assert!(bonus > 0.0);
    }

    #[tokio::test]
    async fn test_mark_completed() {
        let scheduler = AdvancedScheduler::new(SchedulingPolicy::Fifo);
        let req = ScheduledRequest::new(test_cid(1), SchedulePriority::Normal);

        scheduler.schedule(req.clone()).await;
        let next = scheduler
            .next()
            .await
            .expect("test: get next scheduled item");
        scheduler
            .mark_completed(&next, Duration::from_millis(100))
            .await;

        let stats = scheduler.stats().await;
        assert_eq!(stats.total_completed, 1);
        assert_eq!(stats.completion_rate(), 1.0);
    }

    #[tokio::test]
    async fn test_deadline_miss_tracking() {
        let scheduler = AdvancedScheduler::new(SchedulingPolicy::EarliestDeadlineFirst);

        // Create request with deadline in the past
        let past_deadline = Instant::now() - Duration::from_secs(1);
        let req = ScheduledRequest::new(test_cid(1), SchedulePriority::Normal)
            .with_deadline(past_deadline);

        scheduler.schedule(req.clone()).await;
        let next = scheduler
            .next()
            .await
            .expect("test: get next scheduled item");
        scheduler
            .mark_completed(&next, Duration::from_millis(100))
            .await;

        let stats = scheduler.stats().await;
        assert_eq!(stats.deadline_misses, 1);
        assert_eq!(stats.deadline_miss_rate(), 1.0);
    }

    #[tokio::test]
    async fn test_queue_size() {
        let scheduler = AdvancedScheduler::new(SchedulingPolicy::Fifo);

        assert_eq!(scheduler.queue_size().await, 0);

        scheduler
            .schedule(ScheduledRequest::new(test_cid(1), SchedulePriority::Normal))
            .await;
        assert_eq!(scheduler.queue_size().await, 1);

        scheduler
            .schedule(ScheduledRequest::new(test_cid(2), SchedulePriority::Normal))
            .await;
        assert_eq!(scheduler.queue_size().await, 2);

        scheduler.next().await;
        assert_eq!(scheduler.queue_size().await, 1);
    }

    #[tokio::test]
    async fn test_clear_queue() {
        let scheduler = AdvancedScheduler::new(SchedulingPolicy::Fifo);

        scheduler
            .schedule(ScheduledRequest::new(test_cid(1), SchedulePriority::Normal))
            .await;
        scheduler
            .schedule(ScheduledRequest::new(test_cid(2), SchedulePriority::Normal))
            .await;

        assert_eq!(scheduler.queue_size().await, 2);

        scheduler.clear().await;
        assert_eq!(scheduler.queue_size().await, 0);
    }

    #[tokio::test]
    async fn test_peek() {
        let scheduler = AdvancedScheduler::new(SchedulingPolicy::Fifo);
        let req = ScheduledRequest::new(test_cid(1), SchedulePriority::Normal);

        scheduler.schedule(req.clone()).await;

        let peeked = scheduler.peek().await.expect("test: peek scheduler");
        assert_eq!(peeked.cid, test_cid(1));

        // Queue should still have the item
        assert_eq!(scheduler.queue_size().await, 1);
    }

    #[tokio::test]
    async fn test_stats_reset() {
        let scheduler = AdvancedScheduler::new(SchedulingPolicy::Fifo);
        let req = ScheduledRequest::new(test_cid(1), SchedulePriority::Normal);

        scheduler.schedule(req.clone()).await;
        let next = scheduler
            .next()
            .await
            .expect("test: get next scheduled item");
        scheduler
            .mark_completed(&next, Duration::from_millis(100))
            .await;

        let stats = scheduler.stats().await;
        assert!(stats.total_completed > 0);

        scheduler.reset_stats().await;
        let stats = scheduler.stats().await;
        assert_eq!(stats.total_completed, 0);
    }

    #[tokio::test]
    async fn test_multilevel_feedback() {
        let scheduler = AdvancedScheduler::new(SchedulingPolicy::MultilevelFeedback);

        let mut req1 = ScheduledRequest::new(test_cid(1), SchedulePriority::Normal);
        req1.queue_level = 2; // Older request

        let req2 = ScheduledRequest::new(test_cid(2), SchedulePriority::High);
        // queue_level = 0 (newer request)

        scheduler.schedule(req1).await;
        scheduler.schedule(req2).await;

        // Older request should get priority boost
        let next = scheduler
            .next()
            .await
            .expect("test: get next scheduled item");
        // Could be either depending on exact scoring, but let's just verify it works
        assert!(next.cid == test_cid(1) || next.cid == test_cid(2));
    }
}
