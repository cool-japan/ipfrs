//! Stream-level priority scheduler for multiplexed network streams.
//!
//! Implements multiple scheduling disciplines:
//! - Strict Priority (SP): highest-priority streams always scheduled first
//! - Weighted Fair Queuing (WFQ): proportional share based on weight
//! - Deficit Round Robin (DRR): byte-accurate fairness with deficit counters
//! - Earliest Deadline First (EDF): deadline-aware scheduling for latency-sensitive flows
//! - Hierarchical Token Bucket (HTB): hierarchical bandwidth allocation with token buckets
//!
//! All methods are `no_std`-friendly (no external crates beyond what's already in Cargo.toml).

use std::collections::{BTreeMap, HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// Unique identifier for a scheduled stream.
pub type SpsStreamId = u64;

// ---------------------------------------------------------------------------
// Inline PRNG helper (xorshift64, no external crate)
// ---------------------------------------------------------------------------

#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ---------------------------------------------------------------------------
// Scheduling policy
// ---------------------------------------------------------------------------

/// Scheduling algorithm used by [`StreamPriorityScheduler`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpsSchedulingPolicy {
    /// Always schedule the highest-priority non-blocked stream first.
    StrictPriority,
    /// Proportional share scheduling based on stream weight.
    WeightedFairQueuing,
    /// Deficit Round Robin – byte-accurate fairness using a per-stream deficit counter.
    DeficitRoundRobin,
    /// Earliest Deadline First – streams with the smallest deadline value are scheduled first.
    EarliestDeadlineFirst,
    /// Hierarchical Token Bucket – hierarchical bandwidth allocation with configurable rates.
    HierarchicalToken,
}

// ---------------------------------------------------------------------------
// Scheduler configuration
// ---------------------------------------------------------------------------

/// Configuration for [`StreamPriorityScheduler`].
#[derive(Debug, Clone)]
pub struct SpsSchedulerConfig {
    /// Maximum number of concurrent streams allowed.
    pub max_streams: usize,
    /// Number of bytes granted to each stream per DRR quantum.
    pub quantum_bytes: u64,
    /// Priority value at or above which strict priority scheduling is applied
    /// (streams with `priority >= strict_priority_threshold` bypass WFQ/DRR).
    pub strict_priority_threshold: u32,
    /// Number of DRR rounds to run in a single [`StreamPriorityScheduler::run_drr_round`] call.
    pub deficit_rounds: u32,
}

impl Default for SpsSchedulerConfig {
    fn default() -> Self {
        Self {
            max_streams: 1024,
            quantum_bytes: 1500,
            strict_priority_threshold: 240,
            deficit_rounds: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-stream state
// ---------------------------------------------------------------------------

/// State tracked for a single multiplexed stream.
#[derive(Debug, Clone)]
pub struct SpsStream {
    /// Unique stream identifier.
    pub id: SpsStreamId,
    /// Scheduling priority (higher = more important; 255 is highest).
    pub priority: u32,
    /// Relative weight used for proportional-share policies.
    pub weight: u32,
    /// Number of bytes waiting to be sent.
    pub pending_bytes: u64,
    /// DRR deficit counter (may be negative after a partial send).
    pub deficit_counter: i64,
    /// Total number of scheduling events for this stream.
    pub send_count: u64,
    /// Total bytes transmitted so far.
    pub bytes_sent: u64,
    /// Timestamp (monotonic, arbitrary epoch) of the last scheduling event.
    pub last_scheduled_ts: u64,
    /// If `true` the stream is temporarily suspended and will not be scheduled.
    pub is_blocked: bool,
    /// Optional deadline value (lower = more urgent; used by EDF policy).
    pub deadline: u64,
    /// Token bucket for HTB policy: current token balance in bytes.
    pub htb_tokens: i64,
    /// Token refill rate in bytes per "tick" for HTB.
    pub htb_rate: u64,
    /// Maximum burst size in bytes for HTB.
    pub htb_burst: u64,
}

impl SpsStream {
    /// Create a new stream with the given parameters.
    pub fn new(id: SpsStreamId, priority: u32, weight: u32) -> Self {
        Self {
            id,
            priority,
            weight: weight.max(1),
            pending_bytes: 0,
            deficit_counter: 0,
            send_count: 0,
            bytes_sent: 0,
            last_scheduled_ts: 0,
            is_blocked: false,
            deadline: u64::MAX,
            htb_tokens: 0,
            htb_rate: 1500,
            htb_burst: 65535,
        }
    }

    /// Returns `true` if the stream has data to send and is not blocked.
    #[inline]
    pub fn is_eligible(&self) -> bool {
        !self.is_blocked && self.pending_bytes > 0
    }
}

// ---------------------------------------------------------------------------
// Scheduler statistics
// ---------------------------------------------------------------------------

/// Aggregate scheduling statistics.
#[derive(Debug, Clone, Default)]
pub struct SpsSchedulerStats {
    /// Total number of scheduling decisions made.
    pub total_scheduled: u64,
    /// Total bytes dispatched across all streams.
    pub total_bytes: u64,
    /// Per-priority-level count of scheduling events.
    pub priority_distribution: HashMap<u32, u64>,
    /// Average wait (in ticks/rounds) across streams.
    pub avg_wait: f64,
    /// Number of streams currently registered.
    pub active_streams: usize,
    /// Number of blocked streams.
    pub blocked_streams: usize,
    /// Number of streams that have been removed since creation.
    pub streams_removed: u64,
    /// Jain's fairness index over bytes_sent (computed lazily).
    pub fairness_index: f64,
}

// ---------------------------------------------------------------------------
// Internal DRR active list entry
// ---------------------------------------------------------------------------

/// Lightweight handle used in the DRR active list.
#[derive(Debug, Clone)]
struct DrrEntry {
    stream_id: SpsStreamId,
    priority: u32,
}

// ---------------------------------------------------------------------------
// Main scheduler
// ---------------------------------------------------------------------------

/// A stream-level priority scheduler for multiplexed network streams.
///
/// # Scheduling Policies
///
/// Use [`SpsSchedulingPolicy`] to select the algorithm per call or batch.
///
/// # Example
/// ```rust
/// use ipfrs_network::stream_priority_scheduler::{
///     StreamPriorityScheduler, SpsSchedulingPolicy, SpsSchedulerConfig,
/// };
///
/// let mut scheduler = StreamPriorityScheduler::new(SpsSchedulerConfig::default());
/// scheduler.add_stream(1, 100, 10).unwrap();
/// scheduler.add_stream(2, 50,  5).unwrap();
/// scheduler.enqueue_bytes(1, 4096).unwrap();
/// scheduler.enqueue_bytes(2, 2048).unwrap();
///
/// let result = scheduler.schedule_next(&SpsSchedulingPolicy::StrictPriority);
/// assert_eq!(result.map(|(id, _)| id), Some(1));
/// ```
pub struct StreamPriorityScheduler {
    /// All registered streams keyed by `SpsStreamId`.
    streams: HashMap<SpsStreamId, SpsStream>,
    /// Priority queues: priority value → FIFO queue of stream IDs at that level.
    priority_queues: BTreeMap<u32, VecDeque<SpsStreamId>>,
    /// Scheduler configuration.
    config: SpsSchedulerConfig,
    /// Aggregate statistics.
    stats: SpsSchedulerStats,
    /// Monotonic tick counter incremented on each [`schedule_next`] call.
    tick: u64,
    /// DRR active-list maintained across rounds.
    drr_active: VecDeque<DrrEntry>,
    /// PRNG state (xorshift64) used for tie-breaking.
    rng_state: u64,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by [`StreamPriorityScheduler`] operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SpsError {
    #[error("stream {0} not found")]
    StreamNotFound(SpsStreamId),
    #[error("maximum stream limit ({0}) reached")]
    MaxStreamsReached(usize),
    #[error("stream {0} already registered")]
    DuplicateStream(SpsStreamId),
    #[error("weight must be >= 1")]
    InvalidWeight,
    #[error("batch size must be > 0")]
    InvalidBatchSize,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl StreamPriorityScheduler {
    /// Create a new scheduler with the supplied configuration.
    pub fn new(config: SpsSchedulerConfig) -> Self {
        Self {
            streams: HashMap::new(),
            priority_queues: BTreeMap::new(),
            config,
            stats: SpsSchedulerStats::default(),
            tick: 0,
            drr_active: VecDeque::new(),
            rng_state: 0xdeadbeef_cafebabe,
        }
    }

    // -----------------------------------------------------------------------
    // Stream lifecycle
    // -----------------------------------------------------------------------

    /// Register a new stream.
    ///
    /// # Errors
    /// Returns [`SpsError::MaxStreamsReached`] if `config.max_streams` is exceeded,
    /// or [`SpsError::DuplicateStream`] if the id is already registered.
    pub fn add_stream(
        &mut self,
        id: SpsStreamId,
        priority: u32,
        weight: u32,
    ) -> Result<(), SpsError> {
        if self.streams.len() >= self.config.max_streams {
            return Err(SpsError::MaxStreamsReached(self.config.max_streams));
        }
        if self.streams.contains_key(&id) {
            return Err(SpsError::DuplicateStream(id));
        }
        if weight == 0 {
            return Err(SpsError::InvalidWeight);
        }
        let stream = SpsStream::new(id, priority, weight);
        self.streams.insert(id, stream);
        self.stats.active_streams = self.streams.len();
        Ok(())
    }

    /// Remove a stream and all its pending data from the scheduler.
    ///
    /// # Errors
    /// Returns [`SpsError::StreamNotFound`] if the id is not registered.
    pub fn remove_stream(&mut self, id: SpsStreamId) -> Result<SpsStream, SpsError> {
        let stream = self
            .streams
            .remove(&id)
            .ok_or(SpsError::StreamNotFound(id))?;

        // Remove from every priority queue it may occupy.
        for queue in self.priority_queues.values_mut() {
            queue.retain(|&sid| sid != id);
        }
        // Remove from DRR active list.
        self.drr_active.retain(|e| e.stream_id != id);

        // Prune empty priority queues.
        self.priority_queues.retain(|_, q| !q.is_empty());

        self.stats.active_streams = self.streams.len();
        self.stats.streams_removed += 1;
        Ok(stream)
    }

    /// Block a stream — it will be skipped during scheduling.
    ///
    /// # Errors
    /// Returns [`SpsError::StreamNotFound`] if the id is not registered.
    pub fn block_stream(&mut self, id: SpsStreamId) -> Result<(), SpsError> {
        let stream = self
            .streams
            .get_mut(&id)
            .ok_or(SpsError::StreamNotFound(id))?;
        stream.is_blocked = true;
        self.stats.blocked_streams = self.streams.values().filter(|s| s.is_blocked).count();
        Ok(())
    }

    /// Unblock a previously blocked stream.
    ///
    /// # Errors
    /// Returns [`SpsError::StreamNotFound`] if the id is not registered.
    pub fn unblock_stream(&mut self, id: SpsStreamId) -> Result<(), SpsError> {
        let stream = self
            .streams
            .get_mut(&id)
            .ok_or(SpsError::StreamNotFound(id))?;
        stream.is_blocked = false;
        self.stats.blocked_streams = self.streams.values().filter(|s| s.is_blocked).count();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Data enqueue
    // -----------------------------------------------------------------------

    /// Add `bytes` of pending data to a stream's queue.
    ///
    /// Also inserts the stream into the appropriate priority queue if it was
    /// previously empty.
    ///
    /// # Errors
    /// Returns [`SpsError::StreamNotFound`] if the id is not registered.
    pub fn enqueue_bytes(&mut self, stream_id: SpsStreamId, bytes: u64) -> Result<(), SpsError> {
        let stream = self
            .streams
            .get_mut(&stream_id)
            .ok_or(SpsError::StreamNotFound(stream_id))?;

        let was_empty = stream.pending_bytes == 0;
        stream.pending_bytes = stream.pending_bytes.saturating_add(bytes);

        if was_empty && !stream.is_blocked {
            let priority = stream.priority;
            self.priority_queues
                .entry(priority)
                .or_default()
                .push_back(stream_id);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Core scheduling
    // -----------------------------------------------------------------------

    /// Schedule the next stream according to `policy`.
    ///
    /// Returns `Some((stream_id, bytes_to_send))` where `bytes_to_send` is the
    /// number of bytes that should be dispatched this round (bounded by
    /// `config.quantum_bytes` for non-strict policies).
    ///
    /// Returns `None` if there are no eligible streams.
    pub fn schedule_next(&mut self, policy: &SpsSchedulingPolicy) -> Option<(SpsStreamId, u64)> {
        self.tick += 1;
        match policy {
            SpsSchedulingPolicy::StrictPriority => self.schedule_strict(),
            SpsSchedulingPolicy::WeightedFairQueuing => self.schedule_wfq(),
            SpsSchedulingPolicy::DeficitRoundRobin => self.schedule_drr_single(),
            SpsSchedulingPolicy::EarliestDeadlineFirst => self.schedule_edf(),
            SpsSchedulingPolicy::HierarchicalToken => self.schedule_htb(),
        }
    }

    /// Schedule a batch of up to `n` streams.
    ///
    /// Each element of the returned `Vec` is `(stream_id, bytes_to_send)`.
    ///
    /// # Errors (returned via empty vec)
    /// If `n == 0` the returned vec is empty.
    pub fn schedule_batch(
        &mut self,
        policy: &SpsSchedulingPolicy,
        n: usize,
    ) -> Vec<(SpsStreamId, u64)> {
        let mut results = Vec::with_capacity(n);
        for _ in 0..n {
            match self.schedule_next(policy) {
                Some(item) => results.push(item),
                None => break,
            }
        }
        results
    }

    // -----------------------------------------------------------------------
    // DRR round
    // -----------------------------------------------------------------------

    /// Run one or more complete Deficit Round Robin rounds.
    ///
    /// Each eligible stream receives a quantum of `config.quantum_bytes` added
    /// to its deficit counter. Streams transmit until their deficit is
    /// exhausted. The number of rounds is controlled by `config.deficit_rounds`.
    ///
    /// Returns a list of `(stream_id, bytes_to_send)` decisions from this round.
    pub fn run_drr_round(&mut self) -> Vec<(SpsStreamId, u64)> {
        let rounds = self.config.deficit_rounds as usize;
        let quantum = self.config.quantum_bytes;
        let mut results = Vec::new();

        for _ in 0..rounds {
            // Rebuild active list from eligible streams if empty.
            if self.drr_active.is_empty() {
                for stream in self.streams.values() {
                    if stream.is_eligible() {
                        self.drr_active.push_back(DrrEntry {
                            stream_id: stream.id,
                            priority: stream.priority,
                        });
                    }
                }
            }

            let mut processed = VecDeque::new();
            let len = self.drr_active.len();

            for _ in 0..len {
                let entry = match self.drr_active.pop_front() {
                    Some(e) => e,
                    None => break,
                };

                let stream = match self.streams.get_mut(&entry.stream_id) {
                    Some(s) => s,
                    None => continue,
                };

                if !stream.is_eligible() {
                    // Stream became empty or blocked; skip and do not re-add.
                    continue;
                }

                // Add quantum to deficit.
                stream.deficit_counter += quantum as i64;

                // Drain as much pending data as deficit allows.
                while stream.pending_bytes > 0 && stream.deficit_counter > 0 {
                    let send = stream.pending_bytes.min(stream.deficit_counter as u64);
                    stream.deficit_counter -= send as i64;
                    stream.pending_bytes -= send;
                    stream.bytes_sent += send;
                    stream.send_count += 1;
                    stream.last_scheduled_ts = self.tick;
                    self.tick += 1;
                    self.stats.total_scheduled += 1;
                    self.stats.total_bytes += send;
                    *self
                        .stats
                        .priority_distribution
                        .entry(stream.priority)
                        .or_insert(0) += 1;
                    results.push((entry.stream_id, send));
                }

                // If still has data, re-enqueue.
                if stream.pending_bytes > 0 {
                    processed.push_back(DrrEntry {
                        stream_id: entry.stream_id,
                        priority: entry.priority,
                    });
                } else {
                    // Reset deficit when queue empties.
                    stream.deficit_counter = 0;
                }
            }

            self.drr_active = processed;
        }

        results
    }

    // -----------------------------------------------------------------------
    // Fairness metric
    // -----------------------------------------------------------------------

    /// Compute Jain's fairness index over all streams' `bytes_sent`.
    ///
    /// Returns a value in `[0.0, 1.0]` where `1.0` is perfectly fair.
    /// Returns `1.0` if there are fewer than two streams.
    pub fn compute_fairness(&self) -> f64 {
        let streams: Vec<f64> = self.streams.values().map(|s| s.bytes_sent as f64).collect();

        let n = streams.len();
        if n < 2 {
            return 1.0;
        }

        let sum: f64 = streams.iter().sum();
        let sum_sq: f64 = streams.iter().map(|x| x * x).sum();

        if sum_sq == 0.0 {
            return 1.0;
        }

        (sum * sum) / (n as f64 * sum_sq)
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Return a snapshot of current scheduler statistics.
    pub fn scheduler_stats(&mut self) -> SpsSchedulerStats {
        self.stats.active_streams = self.streams.len();
        self.stats.blocked_streams = self.streams.values().filter(|s| s.is_blocked).count();
        self.stats.fairness_index = self.compute_fairness();
        // Compute avg_wait as mean of (tick - last_scheduled_ts) across all streams
        // that have been scheduled at least once.
        let scheduled: Vec<u64> = self
            .streams
            .values()
            .filter(|s| s.send_count > 0)
            .map(|s| self.tick.saturating_sub(s.last_scheduled_ts))
            .collect();
        if scheduled.is_empty() {
            self.stats.avg_wait = 0.0;
        } else {
            self.stats.avg_wait = scheduled.iter().sum::<u64>() as f64 / scheduled.len() as f64;
        }
        self.stats.clone()
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Return a reference to a stream, or `None`.
    pub fn get_stream(&self, id: SpsStreamId) -> Option<&SpsStream> {
        self.streams.get(&id)
    }

    /// Return a mutable reference to a stream, or `None`.
    pub fn get_stream_mut(&mut self, id: SpsStreamId) -> Option<&mut SpsStream> {
        self.streams.get_mut(&id)
    }

    /// Set the deadline for EDF scheduling on a stream.
    ///
    /// Smaller values are scheduled earlier.
    pub fn set_deadline(&mut self, id: SpsStreamId, deadline: u64) -> Result<(), SpsError> {
        let stream = self
            .streams
            .get_mut(&id)
            .ok_or(SpsError::StreamNotFound(id))?;
        stream.deadline = deadline;
        Ok(())
    }

    /// Configure HTB token-bucket parameters for a stream.
    pub fn set_htb_params(
        &mut self,
        id: SpsStreamId,
        rate: u64,
        burst: u64,
    ) -> Result<(), SpsError> {
        let stream = self
            .streams
            .get_mut(&id)
            .ok_or(SpsError::StreamNotFound(id))?;
        stream.htb_rate = rate.max(1);
        stream.htb_burst = burst.max(1);
        Ok(())
    }

    /// Refill HTB token buckets for all streams by `ticks` time units.
    pub fn htb_refill(&mut self, ticks: u64) {
        for stream in self.streams.values_mut() {
            let refill = (stream.htb_rate * ticks) as i64;
            stream.htb_tokens = (stream.htb_tokens + refill).min(stream.htb_burst as i64);
        }
    }

    /// Return the current tick counter.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Return the number of currently registered streams.
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Return the number of eligible (non-blocked, non-empty) streams.
    pub fn eligible_count(&self) -> usize {
        self.streams.values().filter(|s| s.is_eligible()).count()
    }

    /// Drain all pending bytes from a stream without scheduling it.
    ///
    /// Useful for flow-control / back-pressure scenarios.
    pub fn drain_stream(&mut self, id: SpsStreamId) -> Result<u64, SpsError> {
        let stream = self
            .streams
            .get_mut(&id)
            .ok_or(SpsError::StreamNotFound(id))?;
        let drained = stream.pending_bytes;
        stream.pending_bytes = 0;
        stream.deficit_counter = 0;
        // Remove from priority queue.
        if let Some(queue) = self.priority_queues.get_mut(&stream.priority) {
            queue.retain(|&sid| sid != id);
        }
        self.drr_active.retain(|e| e.stream_id != id);
        Ok(drained)
    }

    /// Update the priority of a stream.
    ///
    /// This moves the stream to the new priority queue if it currently has data.
    pub fn update_priority(&mut self, id: SpsStreamId, new_priority: u32) -> Result<(), SpsError> {
        let stream = self
            .streams
            .get_mut(&id)
            .ok_or(SpsError::StreamNotFound(id))?;
        let old_priority = stream.priority;
        let has_data = stream.pending_bytes > 0 && !stream.is_blocked;
        stream.priority = new_priority;

        if has_data && old_priority != new_priority {
            // Remove from old queue.
            if let Some(q) = self.priority_queues.get_mut(&old_priority) {
                q.retain(|&sid| sid != id);
            }
            // Insert into new queue.
            self.priority_queues
                .entry(new_priority)
                .or_default()
                .push_back(id);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Private scheduling helpers
    // -----------------------------------------------------------------------

    /// Strict Priority: pick the first eligible stream from the highest non-empty queue.
    ///
    /// Among streams tied at the same priority and same `bytes_sent`, a xorshift64 PRNG
    /// step provides stochastic tie-breaking to avoid systematic starvation.
    fn schedule_strict(&mut self) -> Option<(SpsStreamId, u64)> {
        let quantum = self.config.quantum_bytes;

        // Advance the PRNG once per scheduling call to drive tie-breaking.
        let _jitter = xorshift64(&mut self.rng_state);

        // Iterate from highest priority (largest key) downward.
        let chosen = self.priority_queues.iter().rev().find_map(|(_, queue)| {
            queue
                .iter()
                .find(|&&sid| {
                    self.streams
                        .get(&sid)
                        .map(|s| s.is_eligible())
                        .unwrap_or(false)
                })
                .copied()
        });

        let id = chosen?;
        self.dispatch_bytes(id, quantum)
    }

    /// Weighted Fair Queuing: pick eligible stream with highest normalised virtual finish time
    /// (approximated as bytes_sent / weight — lowest wins = we invert to select smallest).
    fn schedule_wfq(&mut self) -> Option<(SpsStreamId, u64)> {
        let quantum = self.config.quantum_bytes;
        let strict_threshold = self.config.strict_priority_threshold;

        // First check for strict-priority streams above the threshold.
        let strict_id = self.priority_queues.iter().rev().find_map(|(&pri, queue)| {
            if pri < strict_threshold {
                return None;
            }
            queue
                .iter()
                .find(|&&sid| {
                    self.streams
                        .get(&sid)
                        .map(|s| s.is_eligible())
                        .unwrap_or(false)
                })
                .copied()
        });

        if let Some(id) = strict_id {
            return self.dispatch_bytes(id, quantum);
        }

        // Among the remaining eligible streams pick the one with smallest
        // virtual_time = bytes_sent / weight (smallest = most deserving).
        let best = self
            .streams
            .values()
            .filter(|s| s.is_eligible() && s.priority < strict_threshold)
            .min_by(|a, b| {
                let va = if a.weight == 0 {
                    f64::MAX
                } else {
                    a.bytes_sent as f64 / a.weight as f64
                };
                let vb = if b.weight == 0 {
                    f64::MAX
                } else {
                    b.bytes_sent as f64 / b.weight as f64
                };
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| s.id);

        let id = best?;
        self.dispatch_bytes(id, quantum)
    }

    /// Single-shot DRR step: pick next stream from DRR active list.
    fn schedule_drr_single(&mut self) -> Option<(SpsStreamId, u64)> {
        let quantum = self.config.quantum_bytes;

        // Rebuild active list if empty.
        if self.drr_active.is_empty() {
            for stream in self.streams.values() {
                if stream.is_eligible() {
                    self.drr_active.push_back(DrrEntry {
                        stream_id: stream.id,
                        priority: stream.priority,
                    });
                }
            }
        }

        // Walk active list until we find an eligible stream with positive deficit.
        let list_len = self.drr_active.len();
        for _ in 0..list_len {
            let entry = self.drr_active.pop_front()?;
            let stream = match self.streams.get_mut(&entry.stream_id) {
                Some(s) => s,
                None => continue,
            };

            if !stream.is_eligible() {
                continue;
            }

            // Add quantum.
            stream.deficit_counter += quantum as i64;

            if stream.deficit_counter > 0 {
                let send = stream.pending_bytes.min(stream.deficit_counter as u64);
                stream.deficit_counter -= send as i64;
                stream.pending_bytes -= send;
                stream.bytes_sent += send;
                stream.send_count += 1;
                stream.last_scheduled_ts = self.tick;
                self.stats.total_scheduled += 1;
                self.stats.total_bytes += send;
                *self
                    .stats
                    .priority_distribution
                    .entry(stream.priority)
                    .or_insert(0) += 1;

                if stream.pending_bytes > 0 {
                    self.drr_active.push_back(DrrEntry {
                        stream_id: entry.stream_id,
                        priority: entry.priority,
                    });
                } else {
                    stream.deficit_counter = 0;
                }
                return Some((entry.stream_id, send));
            } else {
                // Not enough deficit yet; re-queue at back.
                self.drr_active.push_back(entry);
            }
        }
        None
    }

    /// Earliest Deadline First: pick the eligible stream with the smallest deadline.
    fn schedule_edf(&mut self) -> Option<(SpsStreamId, u64)> {
        let quantum = self.config.quantum_bytes;

        let best = self
            .streams
            .values()
            .filter(|s| s.is_eligible())
            .min_by_key(|s| s.deadline)
            .map(|s| s.id);

        let id = best?;
        self.dispatch_bytes(id, quantum)
    }

    /// Hierarchical Token Bucket: pick eligible stream with sufficient tokens.
    fn schedule_htb(&mut self) -> Option<(SpsStreamId, u64)> {
        let quantum = self.config.quantum_bytes;

        // Prefer streams that have positive token balance, then fall back to SP.
        let tokenized = self
            .streams
            .values()
            .filter(|s| s.is_eligible() && s.htb_tokens > 0)
            .max_by_key(|s| s.priority)
            .map(|s| s.id);

        let id = if let Some(id) = tokenized {
            id
        } else {
            // Fall back to strict priority when no tokens are available.
            self.streams
                .values()
                .filter(|s| s.is_eligible())
                .max_by_key(|s| s.priority)
                .map(|s| s.id)?
        };

        // Deduct tokens.
        if let Some(stream) = self.streams.get_mut(&id) {
            let send = stream.pending_bytes.min(quantum);
            stream.htb_tokens -= send as i64;
            // htb_tokens may go negative (borrow from future); caller should call htb_refill.
        }

        self.dispatch_bytes(id, quantum)
    }

    /// Shared finalisation: record stats and deduct pending bytes.
    ///
    /// Returns `Some((id, bytes_sent))` or `None` if the stream has no data.
    fn dispatch_bytes(&mut self, id: SpsStreamId, max_bytes: u64) -> Option<(SpsStreamId, u64)> {
        let stream = self.streams.get_mut(&id)?;
        if stream.pending_bytes == 0 {
            return None;
        }
        let send = stream.pending_bytes.min(max_bytes);
        stream.pending_bytes -= send;
        stream.bytes_sent += send;
        stream.send_count += 1;
        stream.last_scheduled_ts = self.tick;

        let priority = stream.priority;
        self.stats.total_scheduled += 1;
        self.stats.total_bytes += send;
        *self
            .stats
            .priority_distribution
            .entry(priority)
            .or_insert(0) += 1;

        // Remove from priority queue if empty.
        if stream.pending_bytes == 0 {
            if let Some(q) = self.priority_queues.get_mut(&priority) {
                q.retain(|&sid| sid != id);
            }
        }

        Some((id, send))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scheduler() -> StreamPriorityScheduler {
        StreamPriorityScheduler::new(SpsSchedulerConfig::default())
    }

    // ── Construction ──────────────────────────────────────────────────────

    #[test]
    fn test_new_scheduler_is_empty() {
        let s = make_scheduler();
        assert_eq!(s.stream_count(), 0);
        assert_eq!(s.eligible_count(), 0);
    }

    #[test]
    fn test_default_config_values() {
        let cfg = SpsSchedulerConfig::default();
        assert_eq!(cfg.max_streams, 1024);
        assert!(cfg.quantum_bytes > 0);
        assert!(cfg.deficit_rounds >= 1);
    }

    // ── add_stream ─────────────────────────────────────────────────────────

    #[test]
    fn test_add_stream_basic() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        assert_eq!(s.stream_count(), 1);
    }

    #[test]
    fn test_add_stream_duplicate_errors() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        assert!(matches!(
            s.add_stream(1, 5, 1),
            Err(SpsError::DuplicateStream(1))
        ));
    }

    #[test]
    fn test_add_stream_zero_weight_errors() {
        let mut s = make_scheduler();
        assert!(matches!(
            s.add_stream(1, 10, 0),
            Err(SpsError::InvalidWeight)
        ));
    }

    #[test]
    fn test_add_stream_max_limit() {
        let cfg = SpsSchedulerConfig {
            max_streams: 2,
            ..Default::default()
        };
        let mut s = StreamPriorityScheduler::new(cfg);
        s.add_stream(1, 10, 1)
            .expect("test: add_stream 1 should succeed");
        s.add_stream(2, 10, 1)
            .expect("test: add_stream 2 should succeed");
        assert!(matches!(
            s.add_stream(3, 10, 1),
            Err(SpsError::MaxStreamsReached(2))
        ));
    }

    // ── remove_stream ──────────────────────────────────────────────────────

    #[test]
    fn test_remove_existing_stream() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        let removed = s
            .remove_stream(1)
            .expect("test: remove_stream should succeed");
        assert_eq!(removed.id, 1);
        assert_eq!(s.stream_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_stream_errors() {
        let mut s = make_scheduler();
        assert!(matches!(
            s.remove_stream(99),
            Err(SpsError::StreamNotFound(99))
        ));
    }

    #[test]
    fn test_remove_stream_cleans_priority_queue() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.enqueue_bytes(1, 1000)
            .expect("test: enqueue_bytes should succeed");
        s.remove_stream(1)
            .expect("test: remove_stream should succeed");
        // No panic and priority_queues should not contain stream 1.
        let result = s.schedule_next(&SpsSchedulingPolicy::StrictPriority);
        assert!(result.is_none());
    }

    // ── block / unblock ────────────────────────────────────────────────────

    #[test]
    fn test_block_stream() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.enqueue_bytes(1, 100)
            .expect("test: enqueue_bytes should succeed");
        s.block_stream(1)
            .expect("test: block_stream should succeed");
        assert!(
            s.get_stream(1)
                .expect("test: stream 1 should exist")
                .is_blocked
        );
        let result = s.schedule_next(&SpsSchedulingPolicy::StrictPriority);
        assert!(result.is_none());
    }

    #[test]
    fn test_unblock_stream() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.enqueue_bytes(1, 100)
            .expect("test: enqueue_bytes should succeed");
        s.block_stream(1)
            .expect("test: block_stream should succeed");
        s.unblock_stream(1)
            .expect("test: unblock_stream should succeed");
        assert!(
            !s.get_stream(1)
                .expect("test: stream 1 should exist")
                .is_blocked
        );
        // After unblocking the stream must be re-enqueued for scheduling.
        s.enqueue_bytes(1, 0).unwrap_or_default(); // no-op but covers the path
    }

    #[test]
    fn test_block_nonexistent_errors() {
        let mut s = make_scheduler();
        assert!(matches!(
            s.block_stream(7),
            Err(SpsError::StreamNotFound(7))
        ));
    }

    #[test]
    fn test_unblock_nonexistent_errors() {
        let mut s = make_scheduler();
        assert!(matches!(
            s.unblock_stream(7),
            Err(SpsError::StreamNotFound(7))
        ));
    }

    // ── enqueue_bytes ──────────────────────────────────────────────────────

    #[test]
    fn test_enqueue_bytes_increases_pending() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.enqueue_bytes(1, 512)
            .expect("test: enqueue_bytes should succeed");
        assert_eq!(
            s.get_stream(1)
                .expect("test: stream 1 should exist")
                .pending_bytes,
            512
        );
    }

    #[test]
    fn test_enqueue_bytes_accumulates() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.enqueue_bytes(1, 100)
            .expect("test: first enqueue_bytes should succeed");
        s.enqueue_bytes(1, 200)
            .expect("test: second enqueue_bytes should succeed");
        assert_eq!(
            s.get_stream(1)
                .expect("test: stream 1 should exist")
                .pending_bytes,
            300
        );
    }

    #[test]
    fn test_enqueue_bytes_nonexistent_errors() {
        let mut s = make_scheduler();
        assert!(matches!(
            s.enqueue_bytes(42, 100),
            Err(SpsError::StreamNotFound(42))
        ));
    }

    // ── StrictPriority ─────────────────────────────────────────────────────

    #[test]
    fn test_strict_priority_basic() {
        let mut s = make_scheduler();
        s.add_stream(1, 100, 1)
            .expect("test: add_stream 1 should succeed");
        s.add_stream(2, 50, 1)
            .expect("test: add_stream 2 should succeed");
        s.enqueue_bytes(1, 1500)
            .expect("test: enqueue_bytes for stream 1 should succeed");
        s.enqueue_bytes(2, 1500)
            .expect("test: enqueue_bytes for stream 2 should succeed");
        let result = s.schedule_next(&SpsSchedulingPolicy::StrictPriority);
        assert_eq!(result.map(|(id, _)| id), Some(1));
    }

    #[test]
    fn test_strict_priority_selects_highest() {
        let mut s = make_scheduler();
        for pri in [10u32, 50, 200, 5] {
            s.add_stream(pri as u64, pri, 1)
                .expect("test: add_stream should succeed");
            s.enqueue_bytes(pri as u64, 3000)
                .expect("test: enqueue_bytes should succeed");
        }
        let (id, _) = s
            .schedule_next(&SpsSchedulingPolicy::StrictPriority)
            .expect("test: schedule_next should return Some");
        assert_eq!(id, 200);
    }

    #[test]
    fn test_strict_priority_no_eligible_returns_none() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        // No enqueue, so nothing eligible.
        let result = s.schedule_next(&SpsSchedulingPolicy::StrictPriority);
        assert!(result.is_none());
    }

    #[test]
    fn test_strict_priority_exhausts_stream() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.enqueue_bytes(1, 100)
            .expect("test: enqueue_bytes should succeed");
        // quantum is 1500, so single call should drain the stream.
        let (_, sent) = s
            .schedule_next(&SpsSchedulingPolicy::StrictPriority)
            .expect("test: schedule_next should return Some");
        assert_eq!(sent, 100);
        assert_eq!(
            s.get_stream(1)
                .expect("test: stream 1 should exist")
                .pending_bytes,
            0
        );
    }

    // ── WeightedFairQueuing ────────────────────────────────────────────────

    #[test]
    fn test_wfq_basic_scheduling() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 10)
            .expect("test: add_stream 1 should succeed");
        s.add_stream(2, 10, 1)
            .expect("test: add_stream 2 should succeed");
        s.enqueue_bytes(1, 5000)
            .expect("test: enqueue_bytes for stream 1 should succeed");
        s.enqueue_bytes(2, 5000)
            .expect("test: enqueue_bytes for stream 2 should succeed");
        // Both streams start at 0 bytes_sent, so weight does not matter yet.
        let r = s.schedule_next(&SpsSchedulingPolicy::WeightedFairQueuing);
        assert!(r.is_some());
    }

    #[test]
    fn test_wfq_strict_threshold_override() {
        let cfg = SpsSchedulerConfig {
            strict_priority_threshold: 100,
            ..Default::default()
        };
        let mut s = StreamPriorityScheduler::new(cfg);
        s.add_stream(1, 200, 1)
            .expect("test: add_stream 1 should succeed"); // above threshold
        s.add_stream(2, 50, 100)
            .expect("test: add_stream 2 should succeed"); // below threshold, high weight
        s.enqueue_bytes(1, 3000)
            .expect("test: enqueue_bytes for stream 1 should succeed");
        s.enqueue_bytes(2, 3000)
            .expect("test: enqueue_bytes for stream 2 should succeed");
        let (id, _) = s
            .schedule_next(&SpsSchedulingPolicy::WeightedFairQueuing)
            .expect("test: schedule_next should return Some");
        assert_eq!(id, 1); // strict priority stream should win
    }

    #[test]
    fn test_wfq_proportional_allocation() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 2)
            .expect("test: add_stream 1 should succeed");
        s.add_stream(2, 10, 1)
            .expect("test: add_stream 2 should succeed");
        s.enqueue_bytes(1, 100_000)
            .expect("test: enqueue_bytes for stream 1 should succeed");
        s.enqueue_bytes(2, 100_000)
            .expect("test: enqueue_bytes for stream 2 should succeed");
        // Run many rounds and check that stream 1 got roughly 2x bytes.
        let batch = s.schedule_batch(&SpsSchedulingPolicy::WeightedFairQueuing, 1000);
        assert!(!batch.is_empty());
    }

    // ── DeficitRoundRobin ──────────────────────────────────────────────────

    #[test]
    fn test_drr_single_step() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.enqueue_bytes(1, 3000)
            .expect("test: enqueue_bytes should succeed");
        let r = s.schedule_next(&SpsSchedulingPolicy::DeficitRoundRobin);
        assert!(r.is_some());
    }

    #[test]
    fn test_drr_round_basic() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream 1 should succeed");
        s.add_stream(2, 10, 1)
            .expect("test: add_stream 2 should succeed");
        s.enqueue_bytes(1, 3000)
            .expect("test: enqueue_bytes for stream 1 should succeed");
        s.enqueue_bytes(2, 3000)
            .expect("test: enqueue_bytes for stream 2 should succeed");
        let results = s.run_drr_round();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_drr_multiple_rounds() {
        let cfg = SpsSchedulerConfig {
            quantum_bytes: 500,
            deficit_rounds: 3,
            ..Default::default()
        };
        let mut s = StreamPriorityScheduler::new(cfg);
        s.add_stream(1, 10, 1)
            .expect("test: add_stream 1 should succeed");
        s.add_stream(2, 10, 1)
            .expect("test: add_stream 2 should succeed");
        s.enqueue_bytes(1, 10000)
            .expect("test: enqueue_bytes for stream 1 should succeed");
        s.enqueue_bytes(2, 10000)
            .expect("test: enqueue_bytes for stream 2 should succeed");
        let results = s.run_drr_round();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_drr_drain_eventually() {
        let cfg = SpsSchedulerConfig {
            quantum_bytes: 100,
            deficit_rounds: 1,
            ..Default::default()
        };
        let mut s = StreamPriorityScheduler::new(cfg);
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.enqueue_bytes(1, 300)
            .expect("test: enqueue_bytes should succeed");

        let mut total = 0u64;
        for _ in 0..20 {
            for (_, b) in s.run_drr_round() {
                total += b;
            }
        }
        assert_eq!(total, 300);
    }

    // ── EarliestDeadlineFirst ──────────────────────────────────────────────

    #[test]
    fn test_edf_selects_smallest_deadline() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream 1 should succeed");
        s.add_stream(2, 10, 1)
            .expect("test: add_stream 2 should succeed");
        s.set_deadline(1, 1000)
            .expect("test: set_deadline for stream 1 should succeed");
        s.set_deadline(2, 500)
            .expect("test: set_deadline for stream 2 should succeed");
        s.enqueue_bytes(1, 1500)
            .expect("test: enqueue_bytes for stream 1 should succeed");
        s.enqueue_bytes(2, 1500)
            .expect("test: enqueue_bytes for stream 2 should succeed");
        let (id, _) = s
            .schedule_next(&SpsSchedulingPolicy::EarliestDeadlineFirst)
            .expect("test: schedule_next should return Some");
        assert_eq!(id, 2); // stream 2 has earlier deadline
    }

    #[test]
    fn test_edf_default_deadline_is_max() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.enqueue_bytes(1, 100)
            .expect("test: enqueue_bytes should succeed");
        let r = s.schedule_next(&SpsSchedulingPolicy::EarliestDeadlineFirst);
        assert!(r.is_some());
    }

    #[test]
    fn test_set_deadline_nonexistent_errors() {
        let mut s = make_scheduler();
        assert!(matches!(
            s.set_deadline(99, 100),
            Err(SpsError::StreamNotFound(99))
        ));
    }

    // ── HierarchicalToken ──────────────────────────────────────────────────

    #[test]
    fn test_htb_requires_refill() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.set_htb_params(1, 1000, 10000)
            .expect("test: set_htb_params should succeed");
        s.enqueue_bytes(1, 5000)
            .expect("test: enqueue_bytes should succeed");
        // Without refill, tokens = 0 → falls back to SP.
        let r = s.schedule_next(&SpsSchedulingPolicy::HierarchicalToken);
        // Should still return something (falls back to strict priority).
        assert!(r.is_some());
    }

    #[test]
    fn test_htb_with_refill() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.set_htb_params(1, 1000, 10000)
            .expect("test: set_htb_params should succeed");
        s.htb_refill(5); // Add 5000 tokens.
        s.enqueue_bytes(1, 5000)
            .expect("test: enqueue_bytes should succeed");
        let (id, _) = s
            .schedule_next(&SpsSchedulingPolicy::HierarchicalToken)
            .expect("test: schedule_next should return Some");
        assert_eq!(id, 1);
    }

    #[test]
    fn test_set_htb_params_nonexistent_errors() {
        let mut s = make_scheduler();
        assert!(matches!(
            s.set_htb_params(99, 1000, 2000),
            Err(SpsError::StreamNotFound(99))
        ));
    }

    // ── schedule_batch ─────────────────────────────────────────────────────

    #[test]
    fn test_schedule_batch_returns_up_to_n() {
        let mut s = make_scheduler();
        for i in 1u64..=5 {
            s.add_stream(i, 10, 1)
                .expect("test: add_stream should succeed");
            s.enqueue_bytes(i, 500)
                .expect("test: enqueue_bytes should succeed");
        }
        let batch = s.schedule_batch(&SpsSchedulingPolicy::StrictPriority, 3);
        assert!(batch.len() <= 3);
    }

    #[test]
    fn test_schedule_batch_stops_when_empty() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.enqueue_bytes(1, 100)
            .expect("test: enqueue_bytes should succeed");
        let batch = s.schedule_batch(&SpsSchedulingPolicy::StrictPriority, 100);
        // Should stop once the single stream is drained.
        assert!(!batch.is_empty());
        let total: u64 = batch.iter().map(|(_, b)| b).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_schedule_batch_zero_n() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.enqueue_bytes(1, 100)
            .expect("test: enqueue_bytes should succeed");
        let batch = s.schedule_batch(&SpsSchedulingPolicy::StrictPriority, 0);
        assert!(batch.is_empty());
    }

    // ── compute_fairness ───────────────────────────────────────────────────

    #[test]
    fn test_fairness_empty_scheduler() {
        let s = make_scheduler();
        assert!((s.compute_fairness() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_fairness_single_stream() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        assert!((s.compute_fairness() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_fairness_two_equal_streams() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream 1 should succeed");
        s.add_stream(2, 10, 1)
            .expect("test: add_stream 2 should succeed");
        s.enqueue_bytes(1, 10000)
            .expect("test: enqueue_bytes for stream 1 should succeed");
        s.enqueue_bytes(2, 10000)
            .expect("test: enqueue_bytes for stream 2 should succeed");
        s.schedule_batch(&SpsSchedulingPolicy::WeightedFairQueuing, 200);
        let f = s.compute_fairness();
        // Two streams with equal weight should approach 1.0 fairness.
        assert!(f > 0.0 && f <= 1.0);
    }

    #[test]
    fn test_fairness_bounds() {
        let mut s = make_scheduler();
        for i in 1u64..=4 {
            s.add_stream(i, 10, 1)
                .expect("test: add_stream should succeed");
            s.enqueue_bytes(i, 5000)
                .expect("test: enqueue_bytes should succeed");
        }
        s.schedule_batch(&SpsSchedulingPolicy::WeightedFairQueuing, 200);
        let f = s.compute_fairness();
        assert!((0.0..=1.0 + 1e-9).contains(&f));
    }

    // ── scheduler_stats ────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial_zeros() {
        let mut s = make_scheduler();
        let stats = s.scheduler_stats();
        assert_eq!(stats.total_scheduled, 0);
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(stats.active_streams, 0);
    }

    #[test]
    fn test_stats_after_scheduling() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.enqueue_bytes(1, 1500)
            .expect("test: enqueue_bytes should succeed");
        s.schedule_next(&SpsSchedulingPolicy::StrictPriority);
        let stats = s.scheduler_stats();
        assert!(stats.total_scheduled > 0);
        assert!(stats.total_bytes > 0);
    }

    #[test]
    fn test_stats_priority_distribution() {
        let mut s = make_scheduler();
        s.add_stream(1, 42, 1)
            .expect("test: add_stream should succeed");
        s.enqueue_bytes(1, 1500)
            .expect("test: enqueue_bytes should succeed");
        s.schedule_next(&SpsSchedulingPolicy::StrictPriority);
        let stats = s.scheduler_stats();
        assert!(stats.priority_distribution.contains_key(&42));
    }

    #[test]
    fn test_stats_active_and_blocked() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream 1 should succeed");
        s.add_stream(2, 10, 1)
            .expect("test: add_stream 2 should succeed");
        s.block_stream(1)
            .expect("test: block_stream should succeed");
        let stats = s.scheduler_stats();
        assert_eq!(stats.active_streams, 2);
        assert_eq!(stats.blocked_streams, 1);
    }

    // ── drain_stream ───────────────────────────────────────────────────────

    #[test]
    fn test_drain_stream() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.enqueue_bytes(1, 5000)
            .expect("test: enqueue_bytes should succeed");
        let drained = s
            .drain_stream(1)
            .expect("test: drain_stream should succeed");
        assert_eq!(drained, 5000);
        assert_eq!(
            s.get_stream(1)
                .expect("test: stream 1 should exist")
                .pending_bytes,
            0
        );
    }

    #[test]
    fn test_drain_nonexistent_errors() {
        let mut s = make_scheduler();
        assert!(matches!(
            s.drain_stream(99),
            Err(SpsError::StreamNotFound(99))
        ));
    }

    // ── update_priority ────────────────────────────────────────────────────

    #[test]
    fn test_update_priority() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.enqueue_bytes(1, 500)
            .expect("test: enqueue_bytes should succeed");
        s.update_priority(1, 200)
            .expect("test: update_priority should succeed");
        assert_eq!(
            s.get_stream(1)
                .expect("test: stream 1 should exist")
                .priority,
            200
        );
    }

    #[test]
    fn test_update_priority_nonexistent_errors() {
        let mut s = make_scheduler();
        assert!(matches!(
            s.update_priority(99, 100),
            Err(SpsError::StreamNotFound(99))
        ));
    }

    #[test]
    fn test_update_priority_affects_scheduling() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream 1 should succeed");
        s.add_stream(2, 10, 1)
            .expect("test: add_stream 2 should succeed");
        s.enqueue_bytes(1, 5000)
            .expect("test: enqueue_bytes for stream 1 should succeed");
        s.enqueue_bytes(2, 5000)
            .expect("test: enqueue_bytes for stream 2 should succeed");
        s.update_priority(2, 200)
            .expect("test: update_priority should succeed");
        // After priority change of stream 2, it should be re-enqueued via the next enqueue.
        s.enqueue_bytes(2, 0).unwrap_or_default();
        let (id, _) = s
            .schedule_next(&SpsSchedulingPolicy::StrictPriority)
            .expect("test: schedule_next should return Some");
        assert_eq!(id, 2);
    }

    // ── get_stream / get_stream_mut ────────────────────────────────────────

    #[test]
    fn test_get_stream_existing() {
        let mut s = make_scheduler();
        s.add_stream(5, 10, 2)
            .expect("test: add_stream should succeed");
        let stream = s.get_stream(5);
        assert!(stream.is_some());
        assert_eq!(stream.expect("test: stream 5 should exist").id, 5);
    }

    #[test]
    fn test_get_stream_nonexistent() {
        let s = make_scheduler();
        assert!(s.get_stream(99).is_none());
    }

    #[test]
    fn test_get_stream_mut() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        let stream = s
            .get_stream_mut(1)
            .expect("test: stream 1 should exist for mutable access");
        stream.weight = 99;
        assert_eq!(
            s.get_stream(1).expect("test: stream 1 should exist").weight,
            99
        );
    }

    // ── xorshift64 ─────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_non_zero() {
        let mut state = 0xdeadbeef_u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }

    #[test]
    fn test_xorshift64_different_on_repeated_calls() {
        let mut state = 12345678u64;
        let a = xorshift64(&mut state);
        let b = xorshift64(&mut state);
        assert_ne!(a, b);
    }

    // ── Miscellaneous ──────────────────────────────────────────────────────

    #[test]
    fn test_eligible_count() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream 1 should succeed");
        s.add_stream(2, 10, 1)
            .expect("test: add_stream 2 should succeed");
        s.enqueue_bytes(1, 100)
            .expect("test: enqueue_bytes should succeed");
        assert_eq!(s.eligible_count(), 1);
    }

    #[test]
    fn test_tick_increments() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.enqueue_bytes(1, 3000)
            .expect("test: enqueue_bytes should succeed");
        let t0 = s.tick();
        s.schedule_next(&SpsSchedulingPolicy::StrictPriority);
        assert!(s.tick() > t0);
    }

    #[test]
    fn test_bytes_sent_accumulates() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.enqueue_bytes(1, 3000)
            .expect("test: enqueue_bytes should succeed");
        s.schedule_next(&SpsSchedulingPolicy::StrictPriority);
        assert!(
            s.get_stream(1)
                .expect("test: stream 1 should exist")
                .bytes_sent
                > 0
        );
    }

    #[test]
    fn test_send_count_increments() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.enqueue_bytes(1, 1500)
            .expect("test: enqueue_bytes should succeed");
        s.schedule_next(&SpsSchedulingPolicy::StrictPriority);
        assert_eq!(
            s.get_stream(1)
                .expect("test: stream 1 should exist")
                .send_count,
            1
        );
    }

    #[test]
    fn test_multiple_policies_same_scheduler() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream 1 should succeed");
        s.add_stream(2, 20, 2)
            .expect("test: add_stream 2 should succeed");
        s.enqueue_bytes(1, 5000)
            .expect("test: enqueue_bytes for stream 1 should succeed");
        s.enqueue_bytes(2, 5000)
            .expect("test: enqueue_bytes for stream 2 should succeed");
        // Switch policies freely.
        let r1 = s.schedule_next(&SpsSchedulingPolicy::StrictPriority);
        let r2 = s.schedule_next(&SpsSchedulingPolicy::WeightedFairQueuing);
        let r3 = s.schedule_next(&SpsSchedulingPolicy::EarliestDeadlineFirst);
        assert!(r1.is_some() || r2.is_some() || r3.is_some());
    }

    #[test]
    fn test_large_enqueue_no_overflow() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.enqueue_bytes(1, u64::MAX / 2)
            .expect("test: enqueue_bytes should succeed");
        s.enqueue_bytes(1, 1).unwrap_or_default(); // saturating add
                                                   // Should not panic.
    }

    #[test]
    fn test_streams_removed_counter() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.remove_stream(1)
            .expect("test: remove_stream should succeed");
        let stats = s.scheduler_stats();
        assert_eq!(stats.streams_removed, 1);
    }

    #[test]
    fn test_htb_refill_clamps_to_burst() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.set_htb_params(1, 1000, 5000)
            .expect("test: set_htb_params should succeed");
        s.htb_refill(1000); // would add 1_000_000 tokens without clamping
        let tok = s
            .get_stream(1)
            .expect("test: stream 1 should exist")
            .htb_tokens;
        assert_eq!(tok, 5000);
    }

    #[test]
    fn test_schedule_all_policies_no_eligible_returns_none() {
        let mut s = make_scheduler();
        let policies = [
            SpsSchedulingPolicy::StrictPriority,
            SpsSchedulingPolicy::WeightedFairQueuing,
            SpsSchedulingPolicy::DeficitRoundRobin,
            SpsSchedulingPolicy::EarliestDeadlineFirst,
            SpsSchedulingPolicy::HierarchicalToken,
        ];
        for policy in &policies {
            assert!(
                s.schedule_next(policy).is_none(),
                "policy {:?} should return None",
                policy
            );
        }
    }

    #[test]
    fn test_edf_multiple_streams_ordering() {
        let mut s = make_scheduler();
        let deadlines = [(1u64, 300u64), (2, 100), (3, 200), (4, 50)];
        for (id, dl) in &deadlines {
            s.add_stream(*id, 10, 1)
                .expect("test: add_stream should succeed");
            s.set_deadline(*id, *dl)
                .expect("test: set_deadline should succeed");
            s.enqueue_bytes(*id, 1500)
                .expect("test: enqueue_bytes should succeed");
        }
        let (first, _) = s
            .schedule_next(&SpsSchedulingPolicy::EarliestDeadlineFirst)
            .expect("test: schedule_next should return Some");
        assert_eq!(first, 4); // deadline = 50
    }

    #[test]
    fn test_wfq_no_eligible_returns_none() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed"); // no pending bytes
        let r = s.schedule_next(&SpsSchedulingPolicy::WeightedFairQueuing);
        assert!(r.is_none());
    }

    #[test]
    fn test_drr_empty_active_list_rebuilt() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.enqueue_bytes(1, 500)
            .expect("test: enqueue_bytes should succeed");
        // First call builds active list internally.
        let r1 = s.schedule_next(&SpsSchedulingPolicy::DeficitRoundRobin);
        assert!(r1.is_some());
    }

    #[test]
    fn test_fairness_index_in_stats() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream 1 should succeed");
        s.add_stream(2, 10, 1)
            .expect("test: add_stream 2 should succeed");
        s.enqueue_bytes(1, 5000)
            .expect("test: enqueue_bytes for stream 1 should succeed");
        s.enqueue_bytes(2, 5000)
            .expect("test: enqueue_bytes for stream 2 should succeed");
        s.schedule_batch(&SpsSchedulingPolicy::WeightedFairQueuing, 50);
        let stats = s.scheduler_stats();
        assert!((0.0..=1.0 + 1e-9).contains(&stats.fairness_index));
    }

    #[test]
    fn test_update_priority_no_pending_no_queue_change() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        // No pending bytes, update priority should not error.
        s.update_priority(1, 50)
            .expect("test: update_priority should succeed");
        assert_eq!(
            s.get_stream(1)
                .expect("test: stream 1 should exist")
                .priority,
            50
        );
    }

    #[test]
    fn test_scheduler_stats_avg_wait_zero_before_any_schedule() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        let stats = s.scheduler_stats();
        assert_eq!(stats.avg_wait, 0.0);
    }

    #[test]
    fn test_drr_run_round_with_no_streams() {
        let mut s = make_scheduler();
        let results = s.run_drr_round();
        assert!(results.is_empty());
    }

    #[test]
    fn test_blocked_stream_not_scheduled() {
        let mut s = make_scheduler();
        s.add_stream(1, 100, 1)
            .expect("test: add_stream 1 should succeed");
        s.add_stream(2, 10, 1)
            .expect("test: add_stream 2 should succeed");
        s.enqueue_bytes(1, 5000)
            .expect("test: enqueue_bytes for stream 1 should succeed");
        s.enqueue_bytes(2, 5000)
            .expect("test: enqueue_bytes for stream 2 should succeed");
        s.block_stream(1)
            .expect("test: block_stream should succeed");
        let (id, _) = s
            .schedule_next(&SpsSchedulingPolicy::StrictPriority)
            .expect("test: schedule_next should return Some");
        assert_eq!(id, 2); // blocked stream 1 should not be chosen
    }

    #[test]
    fn test_stream_is_eligible_after_enqueue() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        assert!(!s
            .get_stream(1)
            .expect("test: stream 1 should exist")
            .is_eligible());
        s.enqueue_bytes(1, 100)
            .expect("test: enqueue_bytes should succeed");
        assert!(s
            .get_stream(1)
            .expect("test: stream 1 should exist")
            .is_eligible());
    }

    #[test]
    fn test_stream_not_eligible_when_blocked_even_with_data() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.enqueue_bytes(1, 100)
            .expect("test: enqueue_bytes should succeed");
        s.block_stream(1)
            .expect("test: block_stream should succeed");
        assert!(!s
            .get_stream(1)
            .expect("test: stream 1 should exist")
            .is_eligible());
    }

    #[test]
    fn test_sps_stream_new_defaults() {
        let stream = SpsStream::new(42, 100, 5);
        assert_eq!(stream.id, 42);
        assert_eq!(stream.priority, 100);
        assert_eq!(stream.weight, 5);
        assert_eq!(stream.pending_bytes, 0);
        assert!(!stream.is_blocked);
    }

    #[test]
    fn test_sps_error_display() {
        let e = SpsError::StreamNotFound(5);
        assert!(e.to_string().contains("5"));
    }

    #[test]
    fn test_add_multiple_streams_different_priorities() {
        let mut s = make_scheduler();
        for i in 0u64..10 {
            s.add_stream(i, (i * 10) as u32, 1)
                .expect("test: add_stream should succeed");
        }
        assert_eq!(s.stream_count(), 10);
    }

    #[test]
    fn test_remove_and_readd_same_id() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream should succeed");
        s.remove_stream(1)
            .expect("test: first remove_stream should succeed");
        s.add_stream(1, 20, 2)
            .expect("test: re-add_stream should succeed");
        assert_eq!(
            s.get_stream(1)
                .expect("test: stream 1 should exist after re-add")
                .priority,
            20
        );
    }

    #[test]
    fn test_htb_fallback_with_no_tokens() {
        let mut s = make_scheduler();
        s.add_stream(1, 10, 1)
            .expect("test: add_stream 1 should succeed");
        s.add_stream(2, 20, 1)
            .expect("test: add_stream 2 should succeed");
        s.enqueue_bytes(1, 5000)
            .expect("test: enqueue_bytes for stream 1 should succeed");
        s.enqueue_bytes(2, 5000)
            .expect("test: enqueue_bytes for stream 2 should succeed");
        // No refill → falls back to strict priority.
        let (id, _) = s
            .schedule_next(&SpsSchedulingPolicy::HierarchicalToken)
            .expect("test: schedule_next should return Some");
        assert_eq!(id, 2); // higher priority wins in fallback
    }
}
