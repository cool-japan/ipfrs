//! Priority-based work queue for processing peer requests.
//!
//! Provides [`PeerRequestQueue`] — a multi-level priority queue keyed by
//! [`RequestPriority`] (Critical > High > Normal > Low).  Within each priority
//! level messages are dequeued in strict FIFO order.
//!
//! The queue tracks a monotonic logical clock advanced via [`PeerRequestQueue::tick`],
//! stamps every incoming request with the current tick, and accumulates lifetime
//! statistics in [`PriorityQueueStats`].

use std::collections::{BTreeMap, VecDeque};

// ─── RequestPriority ────────────────────────────────────────────────────────

/// Priority level for a queued peer request.
///
/// Variants are ordered so that [`Ord`] yields `Low < Normal < High < Critical`,
/// matching the dequeue precedence: Critical is popped first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum RequestPriority {
    /// Lowest priority — only processed when all higher tiers are empty.
    Low = 0,
    /// Default priority.
    Normal = 1,
    /// Elevated priority.
    High = 2,
    /// Highest priority — always processed first.
    Critical = 3,
}

// ─── PeerRequest ────────────────────────────────────────────────────────────

/// A single request waiting in the queue.
#[derive(Debug, Clone)]
pub struct PeerRequest {
    /// Unique identifier assigned by the queue on enqueue.
    pub id: u64,
    /// Identifier of the peer that originated the request.
    pub peer_id: String,
    /// Priority tier governing dequeue order.
    pub priority: RequestPriority,
    /// Size of the request payload in bytes.
    pub payload_size: u64,
    /// Logical clock value at the time the request was enqueued.
    pub enqueued_tick: u64,
}

// ─── PriorityQueueStats ─────────────────────────────────────────────────────

/// Snapshot of queue statistics.
#[derive(Debug, Clone)]
pub struct PriorityQueueStats {
    /// Total requests enqueued over the lifetime of the queue.
    pub total_enqueued: u64,
    /// Total requests dequeued over the lifetime of the queue.
    pub total_dequeued: u64,
    /// Current number of pending requests across all priority levels.
    pub current_size: usize,
    /// Pending requests at [`RequestPriority::Critical`].
    pub critical_count: usize,
    /// Pending requests at [`RequestPriority::High`].
    pub high_count: usize,
    /// Pending requests at [`RequestPriority::Normal`].
    pub normal_count: usize,
    /// Pending requests at [`RequestPriority::Low`].
    pub low_count: usize,
}

// ─── PeerRequestQueue ───────────────────────────────────────────────────────

/// Priority-based work queue for peer requests.
///
/// Requests are placed into per-priority FIFO sub-queues stored in a
/// [`BTreeMap`].  On [`dequeue`](Self::dequeue), the highest non-empty
/// priority level is drained first (Critical → High → Normal → Low).
pub struct PeerRequestQueue {
    /// Per-priority FIFO sub-queues.
    queues: BTreeMap<RequestPriority, VecDeque<PeerRequest>>,
    /// Next unique identifier to assign.
    next_id: u64,
    /// Monotonic logical clock.
    current_tick: u64,
    /// Lifetime enqueue counter.
    total_enqueued: u64,
    /// Lifetime dequeue counter.
    total_dequeued: u64,
}

impl PeerRequestQueue {
    /// Create a new, empty queue.
    pub fn new() -> Self {
        let mut queues = BTreeMap::new();
        queues.insert(RequestPriority::Critical, VecDeque::new());
        queues.insert(RequestPriority::High, VecDeque::new());
        queues.insert(RequestPriority::Normal, VecDeque::new());
        queues.insert(RequestPriority::Low, VecDeque::new());

        Self {
            queues,
            next_id: 0,
            current_tick: 0,
            total_enqueued: 0,
            total_dequeued: 0,
        }
    }

    /// Enqueue a new request and return its unique identifier.
    pub fn enqueue(&mut self, peer_id: &str, priority: RequestPriority, payload_size: u64) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);

        let request = PeerRequest {
            id,
            peer_id: peer_id.to_string(),
            priority,
            payload_size,
            enqueued_tick: self.current_tick,
        };

        // The sub-queue is guaranteed to exist because we seed all four in `new()`.
        if let Some(q) = self.queues.get_mut(&priority) {
            q.push_back(request);
        }

        self.total_enqueued += 1;
        id
    }

    /// Pop the highest-priority request (FIFO within each level).
    ///
    /// Returns `None` when every sub-queue is empty.
    pub fn dequeue(&mut self) -> Option<PeerRequest> {
        // Iterate in *reverse* key order because `BTreeMap` orders by `Ord` and
        // `Critical(3) > High(2) > Normal(1) > Low(0)`.
        for (_, q) in self.queues.iter_mut().rev() {
            if let Some(req) = q.pop_front() {
                self.total_dequeued += 1;
                return Some(req);
            }
        }
        None
    }

    /// Peek at the highest-priority request without removing it.
    pub fn peek(&self) -> Option<&PeerRequest> {
        for (_, q) in self.queues.iter().rev() {
            if let Some(req) = q.front() {
                return Some(req);
            }
        }
        None
    }

    /// Total number of requests across all priority levels.
    pub fn len(&self) -> usize {
        self.queues.values().map(VecDeque::len).sum()
    }

    /// Returns `true` when the queue contains no requests.
    pub fn is_empty(&self) -> bool {
        self.queues.values().all(VecDeque::is_empty)
    }

    /// Remove and return all requests at the given priority level.
    pub fn drain_priority(&mut self, priority: RequestPriority) -> Vec<PeerRequest> {
        match self.queues.get_mut(&priority) {
            Some(q) => {
                let drained: Vec<PeerRequest> = q.drain(..).collect();
                self.total_dequeued += drained.len() as u64;
                drained
            }
            None => Vec::new(),
        }
    }

    /// Cancel a specific request by its unique identifier.
    ///
    /// Returns `true` if the request was found and removed.
    pub fn cancel(&mut self, request_id: u64) -> bool {
        for (_, q) in self.queues.iter_mut() {
            if let Some(pos) = q.iter().position(|r| r.id == request_id) {
                q.remove(pos);
                return true;
            }
        }
        false
    }

    /// Advance the logical clock by one tick.
    pub fn tick(&mut self) {
        self.current_tick = self.current_tick.wrapping_add(1);
    }

    /// Number of pending requests at the given priority level.
    pub fn count_by_priority(&self, priority: RequestPriority) -> usize {
        self.queues.get(&priority).map_or(0, VecDeque::len)
    }

    /// Sum of `payload_size` across all pending requests.
    pub fn total_payload_bytes(&self) -> u64 {
        self.queues
            .values()
            .flat_map(|q| q.iter())
            .map(|r| r.payload_size)
            .sum()
    }

    /// Snapshot of current queue statistics.
    pub fn stats(&self) -> PriorityQueueStats {
        PriorityQueueStats {
            total_enqueued: self.total_enqueued,
            total_dequeued: self.total_dequeued,
            current_size: self.len(),
            critical_count: self.count_by_priority(RequestPriority::Critical),
            high_count: self.count_by_priority(RequestPriority::High),
            normal_count: self.count_by_priority(RequestPriority::Normal),
            low_count: self.count_by_priority(RequestPriority::Low),
        }
    }
}

impl Default for PeerRequestQueue {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ─────────────────────────────────────────────────────────────

    fn new_queue() -> PeerRequestQueue {
        PeerRequestQueue::new()
    }

    // ── basic enqueue / dequeue ─────────────────────────────────────────────

    #[test]
    fn enqueue_returns_unique_ids() {
        let mut q = new_queue();
        let id0 = q.enqueue("peer-a", RequestPriority::Normal, 100);
        let id1 = q.enqueue("peer-a", RequestPriority::Normal, 200);
        assert_ne!(id0, id1);
    }

    #[test]
    fn dequeue_returns_none_when_empty() {
        let mut q = new_queue();
        assert!(q.dequeue().is_none());
    }

    #[test]
    fn enqueue_then_dequeue_returns_same_request() {
        let mut q = new_queue();
        let id = q.enqueue("peer-a", RequestPriority::Normal, 512);
        let req = q.dequeue().expect("should have one request");
        assert_eq!(req.id, id);
        assert_eq!(req.peer_id, "peer-a");
        assert_eq!(req.payload_size, 512);
    }

    // ── priority ordering ───────────────────────────────────────────────────

    #[test]
    fn critical_dequeued_before_high() {
        let mut q = new_queue();
        q.enqueue("peer-a", RequestPriority::High, 10);
        q.enqueue("peer-a", RequestPriority::Critical, 10);
        let first = q.dequeue().expect("should have request");
        assert_eq!(first.priority, RequestPriority::Critical);
    }

    #[test]
    fn high_dequeued_before_normal() {
        let mut q = new_queue();
        q.enqueue("peer-a", RequestPriority::Normal, 10);
        q.enqueue("peer-a", RequestPriority::High, 10);
        let first = q.dequeue().expect("should have request");
        assert_eq!(first.priority, RequestPriority::High);
    }

    #[test]
    fn normal_dequeued_before_low() {
        let mut q = new_queue();
        q.enqueue("peer-a", RequestPriority::Low, 10);
        q.enqueue("peer-a", RequestPriority::Normal, 10);
        let first = q.dequeue().expect("should have request");
        assert_eq!(first.priority, RequestPriority::Normal);
    }

    #[test]
    fn full_priority_ordering() {
        let mut q = new_queue();
        q.enqueue("peer-a", RequestPriority::Low, 10);
        q.enqueue("peer-a", RequestPriority::Normal, 10);
        q.enqueue("peer-a", RequestPriority::Critical, 10);
        q.enqueue("peer-a", RequestPriority::High, 10);

        let r1 = q.dequeue().expect("msg");
        let r2 = q.dequeue().expect("msg");
        let r3 = q.dequeue().expect("msg");
        let r4 = q.dequeue().expect("msg");

        assert_eq!(r1.priority, RequestPriority::Critical);
        assert_eq!(r2.priority, RequestPriority::High);
        assert_eq!(r3.priority, RequestPriority::Normal);
        assert_eq!(r4.priority, RequestPriority::Low);
        assert!(q.dequeue().is_none());
    }

    // ── FIFO within same priority ───────────────────────────────────────────

    #[test]
    fn fifo_within_critical() {
        let mut q = new_queue();
        let id0 = q.enqueue("peer-a", RequestPriority::Critical, 10);
        let id1 = q.enqueue("peer-a", RequestPriority::Critical, 20);
        let id2 = q.enqueue("peer-a", RequestPriority::Critical, 30);
        assert_eq!(q.dequeue().expect("msg").id, id0);
        assert_eq!(q.dequeue().expect("msg").id, id1);
        assert_eq!(q.dequeue().expect("msg").id, id2);
    }

    #[test]
    fn fifo_within_high() {
        let mut q = new_queue();
        let id0 = q.enqueue("peer-b", RequestPriority::High, 10);
        let id1 = q.enqueue("peer-b", RequestPriority::High, 20);
        assert_eq!(q.dequeue().expect("msg").id, id0);
        assert_eq!(q.dequeue().expect("msg").id, id1);
    }

    #[test]
    fn fifo_within_normal() {
        let mut q = new_queue();
        let id0 = q.enqueue("peer-c", RequestPriority::Normal, 10);
        let id1 = q.enqueue("peer-c", RequestPriority::Normal, 20);
        let id2 = q.enqueue("peer-c", RequestPriority::Normal, 30);
        assert_eq!(q.dequeue().expect("msg").id, id0);
        assert_eq!(q.dequeue().expect("msg").id, id1);
        assert_eq!(q.dequeue().expect("msg").id, id2);
    }

    #[test]
    fn fifo_within_low() {
        let mut q = new_queue();
        let id0 = q.enqueue("peer-d", RequestPriority::Low, 10);
        let id1 = q.enqueue("peer-d", RequestPriority::Low, 20);
        assert_eq!(q.dequeue().expect("msg").id, id0);
        assert_eq!(q.dequeue().expect("msg").id, id1);
    }

    // ── peek ────────────────────────────────────────────────────────────────

    #[test]
    fn peek_returns_none_when_empty() {
        let q = new_queue();
        assert!(q.peek().is_none());
    }

    #[test]
    fn peek_does_not_remove() {
        let mut q = new_queue();
        q.enqueue("peer-a", RequestPriority::Normal, 100);
        assert!(q.peek().is_some());
        assert_eq!(q.len(), 1);
        assert!(q.peek().is_some());
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn peek_returns_highest_priority() {
        let mut q = new_queue();
        q.enqueue("peer-a", RequestPriority::Low, 10);
        q.enqueue("peer-a", RequestPriority::High, 20);
        let peeked = q.peek().expect("should have request");
        assert_eq!(peeked.priority, RequestPriority::High);
    }

    // ── len / is_empty ──────────────────────────────────────────────────────

    #[test]
    fn len_counts_all_priorities() {
        let mut q = new_queue();
        q.enqueue("peer-a", RequestPriority::Critical, 10);
        q.enqueue("peer-a", RequestPriority::High, 10);
        q.enqueue("peer-a", RequestPriority::Normal, 10);
        q.enqueue("peer-a", RequestPriority::Low, 10);
        assert_eq!(q.len(), 4);
    }

    #[test]
    fn is_empty_when_new() {
        let q = new_queue();
        assert!(q.is_empty());
    }

    #[test]
    fn is_empty_false_after_enqueue() {
        let mut q = new_queue();
        q.enqueue("peer-a", RequestPriority::Normal, 10);
        assert!(!q.is_empty());
    }

    #[test]
    fn is_empty_true_after_all_dequeued() {
        let mut q = new_queue();
        q.enqueue("peer-a", RequestPriority::Normal, 10);
        let _ = q.dequeue();
        assert!(q.is_empty());
    }

    // ── drain_priority ──────────────────────────────────────────────────────

    #[test]
    fn drain_priority_removes_all_at_level() {
        let mut q = new_queue();
        q.enqueue("peer-a", RequestPriority::High, 10);
        q.enqueue("peer-b", RequestPriority::High, 20);
        q.enqueue("peer-a", RequestPriority::Normal, 30);

        let drained = q.drain_priority(RequestPriority::High);
        assert_eq!(drained.len(), 2);
        assert_eq!(q.len(), 1);
        assert_eq!(q.count_by_priority(RequestPriority::High), 0);
    }

    #[test]
    fn drain_priority_returns_empty_when_none() {
        let mut q = new_queue();
        q.enqueue("peer-a", RequestPriority::Normal, 10);
        let drained = q.drain_priority(RequestPriority::Critical);
        assert!(drained.is_empty());
    }

    #[test]
    fn drain_preserves_fifo_order() {
        let mut q = new_queue();
        let id0 = q.enqueue("peer-a", RequestPriority::Normal, 10);
        let id1 = q.enqueue("peer-b", RequestPriority::Normal, 20);
        let id2 = q.enqueue("peer-c", RequestPriority::Normal, 30);

        let drained = q.drain_priority(RequestPriority::Normal);
        assert_eq!(drained.len(), 3);
        assert_eq!(drained[0].id, id0);
        assert_eq!(drained[1].id, id1);
        assert_eq!(drained[2].id, id2);
    }

    // ── cancel ──────────────────────────────────────────────────────────────

    #[test]
    fn cancel_removes_specific_request() {
        let mut q = new_queue();
        let _id0 = q.enqueue("peer-a", RequestPriority::Normal, 10);
        let id1 = q.enqueue("peer-b", RequestPriority::Normal, 20);
        let _id2 = q.enqueue("peer-c", RequestPriority::Normal, 30);

        assert!(q.cancel(id1));
        assert_eq!(q.len(), 2);

        // Remaining requests should be id0 and id2 in FIFO order.
        let r0 = q.dequeue().expect("msg");
        let r2 = q.dequeue().expect("msg");
        assert_eq!(r0.id, _id0);
        assert_eq!(r2.id, _id2);
    }

    #[test]
    fn cancel_returns_false_for_nonexistent_id() {
        let mut q = new_queue();
        q.enqueue("peer-a", RequestPriority::Normal, 10);
        assert!(!q.cancel(9999));
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn cancel_across_priorities() {
        let mut q = new_queue();
        let _id0 = q.enqueue("peer-a", RequestPriority::Critical, 10);
        let id1 = q.enqueue("peer-b", RequestPriority::Low, 20);

        assert!(q.cancel(id1));
        assert_eq!(q.len(), 1);
        let remaining = q.dequeue().expect("msg");
        assert_eq!(remaining.priority, RequestPriority::Critical);
    }

    // ── tick ────────────────────────────────────────────────────────────────

    #[test]
    fn tick_advances_clock() {
        let mut q = new_queue();
        q.tick();
        q.tick();
        let id = q.enqueue("peer-a", RequestPriority::Normal, 10);
        let req = q.dequeue().expect("msg");
        assert_eq!(req.id, id);
        assert_eq!(req.enqueued_tick, 2);
    }

    #[test]
    fn enqueued_tick_reflects_current_clock() {
        let mut q = new_queue();
        q.enqueue("peer-a", RequestPriority::Normal, 10);
        q.tick();
        q.tick();
        q.tick();
        q.enqueue("peer-b", RequestPriority::Normal, 20);

        let r0 = q.dequeue().expect("msg");
        let r1 = q.dequeue().expect("msg");
        assert_eq!(r0.enqueued_tick, 0);
        assert_eq!(r1.enqueued_tick, 3);
    }

    // ── count_by_priority ───────────────────────────────────────────────────

    #[test]
    fn count_by_priority_tracks_correctly() {
        let mut q = new_queue();
        q.enqueue("peer-a", RequestPriority::Critical, 10);
        q.enqueue("peer-b", RequestPriority::Critical, 10);
        q.enqueue("peer-c", RequestPriority::Normal, 10);

        assert_eq!(q.count_by_priority(RequestPriority::Critical), 2);
        assert_eq!(q.count_by_priority(RequestPriority::High), 0);
        assert_eq!(q.count_by_priority(RequestPriority::Normal), 1);
        assert_eq!(q.count_by_priority(RequestPriority::Low), 0);
    }

    // ── total_payload_bytes ─────────────────────────────────────────────────

    #[test]
    fn total_payload_bytes_sums_all() {
        let mut q = new_queue();
        q.enqueue("peer-a", RequestPriority::Critical, 100);
        q.enqueue("peer-b", RequestPriority::Normal, 200);
        q.enqueue("peer-c", RequestPriority::Low, 300);
        assert_eq!(q.total_payload_bytes(), 600);
    }

    #[test]
    fn total_payload_bytes_decreases_after_dequeue() {
        let mut q = new_queue();
        q.enqueue("peer-a", RequestPriority::Normal, 500);
        q.enqueue("peer-b", RequestPriority::Normal, 300);
        let _ = q.dequeue();
        assert_eq!(q.total_payload_bytes(), 300);
    }

    #[test]
    fn total_payload_bytes_zero_when_empty() {
        let q = new_queue();
        assert_eq!(q.total_payload_bytes(), 0);
    }

    // ── stats ───────────────────────────────────────────────────────────────

    #[test]
    fn stats_tracks_enqueued_and_dequeued() {
        let mut q = new_queue();
        q.enqueue("peer-a", RequestPriority::Normal, 10);
        q.enqueue("peer-b", RequestPriority::High, 20);
        q.enqueue("peer-c", RequestPriority::Low, 30);
        let _ = q.dequeue();

        let s = q.stats();
        assert_eq!(s.total_enqueued, 3);
        assert_eq!(s.total_dequeued, 1);
        assert_eq!(s.current_size, 2);
    }

    #[test]
    fn stats_priority_counts() {
        let mut q = new_queue();
        q.enqueue("peer-a", RequestPriority::Critical, 10);
        q.enqueue("peer-b", RequestPriority::Critical, 10);
        q.enqueue("peer-c", RequestPriority::High, 10);
        q.enqueue("peer-d", RequestPriority::Normal, 10);
        q.enqueue("peer-e", RequestPriority::Low, 10);
        q.enqueue("peer-f", RequestPriority::Low, 10);

        let s = q.stats();
        assert_eq!(s.critical_count, 2);
        assert_eq!(s.high_count, 1);
        assert_eq!(s.normal_count, 1);
        assert_eq!(s.low_count, 2);
    }

    #[test]
    fn stats_after_drain() {
        let mut q = new_queue();
        q.enqueue("peer-a", RequestPriority::High, 10);
        q.enqueue("peer-b", RequestPriority::High, 10);
        q.enqueue("peer-c", RequestPriority::Low, 10);

        let _ = q.drain_priority(RequestPriority::High);
        let s = q.stats();
        assert_eq!(s.total_dequeued, 2);
        assert_eq!(s.high_count, 0);
        assert_eq!(s.low_count, 1);
        assert_eq!(s.current_size, 1);
    }

    // ── mixed operations ────────────────────────────────────────────────────

    #[test]
    fn mixed_enqueue_dequeue_cancel_ordering() {
        let mut q = new_queue();
        let id_low = q.enqueue("peer-a", RequestPriority::Low, 10);
        let _id_crit = q.enqueue("peer-b", RequestPriority::Critical, 20);
        let id_normal = q.enqueue("peer-c", RequestPriority::Normal, 30);

        // Cancel the normal-priority request.
        assert!(q.cancel(id_normal));
        assert_eq!(q.len(), 2);

        // Critical should come first, then low.
        let r1 = q.dequeue().expect("msg");
        assert_eq!(r1.priority, RequestPriority::Critical);
        let r2 = q.dequeue().expect("msg");
        assert_eq!(r2.id, id_low);
        assert!(q.is_empty());
    }

    #[test]
    fn interleaved_enqueue_dequeue() {
        let mut q = new_queue();
        q.enqueue("peer-a", RequestPriority::Normal, 10);
        q.enqueue("peer-b", RequestPriority::High, 20);

        // Dequeue should get High first.
        let r1 = q.dequeue().expect("msg");
        assert_eq!(r1.priority, RequestPriority::High);

        // Enqueue a Critical while Normal is still pending.
        q.enqueue("peer-c", RequestPriority::Critical, 30);

        // Critical should come next.
        let r2 = q.dequeue().expect("msg");
        assert_eq!(r2.priority, RequestPriority::Critical);

        // Then Normal.
        let r3 = q.dequeue().expect("msg");
        assert_eq!(r3.priority, RequestPriority::Normal);
        assert!(q.is_empty());
    }

    #[test]
    fn multiple_peers_same_priority() {
        let mut q = new_queue();
        let id_a = q.enqueue("peer-a", RequestPriority::Normal, 10);
        let id_b = q.enqueue("peer-b", RequestPriority::Normal, 20);
        let id_c = q.enqueue("peer-c", RequestPriority::Normal, 30);

        // Should come out in FIFO order regardless of peer_id.
        assert_eq!(q.dequeue().expect("msg").id, id_a);
        assert_eq!(q.dequeue().expect("msg").id, id_b);
        assert_eq!(q.dequeue().expect("msg").id, id_c);
    }

    #[test]
    fn default_impl_creates_empty_queue() {
        let q = PeerRequestQueue::default();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
        assert_eq!(q.total_payload_bytes(), 0);
    }
}
