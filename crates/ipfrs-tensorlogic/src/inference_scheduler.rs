//! TensorInferenceScheduler — deadline-aware priority scheduling for TensorLogic inference jobs.
//!
//! Schedules inference jobs across available compute slots with priority queuing,
//! resource budgets, and deadline enforcement.

use std::collections::HashMap;

/// Status of an inference job in the scheduler lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobStatus {
    /// Queued but not yet started.
    Pending,
    /// Currently executing on a compute slot.
    Running,
    /// Finished successfully.
    Completed,
    /// Deadline passed before the job could complete.
    Expired,
    /// Explicitly cancelled by the caller.
    Cancelled,
}

/// A single inference job managed by the scheduler.
#[derive(Clone, Debug)]
pub struct InferenceJob {
    /// Unique identifier assigned at submission time.
    pub job_id: u64,
    /// The inference goal string.
    pub goal: String,
    /// Higher values are scheduled first.
    pub priority: u32,
    /// Optional logical tick after which the job is considered expired.
    pub deadline_tick: Option<u64>,
    /// Estimated execution time in milliseconds (advisory).
    pub estimated_cost_ms: u64,
    /// Current lifecycle status.
    pub status: JobStatus,
    /// Logical tick at which the job was submitted.
    pub submitted_at_tick: u64,
    /// Logical tick at which execution began, if started.
    pub started_at_tick: Option<u64>,
    /// Logical tick at which execution finished, if completed.
    pub completed_at_tick: Option<u64>,
}

impl InferenceJob {
    /// Returns the end-to-end latency in ticks if the job has completed.
    pub fn latency_ticks(&self) -> Option<u64> {
        match (self.status, self.completed_at_tick) {
            (JobStatus::Completed, Some(done)) => Some(done - self.submitted_at_tick),
            _ => None,
        }
    }

    /// Returns `true` if the job is in a terminal state (no further transitions possible).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            JobStatus::Completed | JobStatus::Expired | JobStatus::Cancelled
        )
    }
}

/// Configuration for [`TensorInferenceScheduler`].
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    /// Maximum number of jobs that may be Running simultaneously.
    pub max_concurrent: usize,
    /// Maximum number of jobs that may be Pending at one time.
    pub max_queue_size: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 4,
            max_queue_size: 256,
        }
    }
}

/// Cumulative statistics maintained by the scheduler.
#[derive(Clone, Debug, Default)]
pub struct SchedulerStats {
    /// Total jobs ever submitted (including those that were rejected — note: rejected jobs
    /// are *not* counted; only accepted submissions increment this).
    pub total_submitted: u64,
    /// Total jobs that reached [`JobStatus::Completed`].
    pub total_completed: u64,
    /// Total jobs that reached [`JobStatus::Expired`].
    pub total_expired: u64,
    /// Total jobs that reached [`JobStatus::Cancelled`].
    pub total_cancelled: u64,
    /// Current number of Pending jobs.
    pub queue_depth: usize,
    /// Current number of Running jobs.
    pub running_count: usize,
}

impl SchedulerStats {
    /// Fraction of terminal jobs that completed successfully.
    ///
    /// Returns `0.0` when no jobs have reached a terminal state.
    pub fn completion_rate(&self) -> f64 {
        let denominator = self.total_completed + self.total_expired + self.total_cancelled;
        if denominator == 0 {
            0.0
        } else {
            self.total_completed as f64 / denominator as f64
        }
    }
}

/// Deadline-aware priority scheduler for TensorLogic inference jobs.
///
/// Uses a logical tick counter (driven by the caller) for deadline evaluation
/// and ordering. All operations are synchronous and single-threaded; wrap with
/// a mutex for multi-threaded use.
pub struct TensorInferenceScheduler {
    /// All known jobs, keyed by job_id.
    pub jobs: HashMap<u64, InferenceJob>,
    /// Monotonically increasing counter used to assign job IDs.
    pub next_job_id: u64,
    /// Immutable configuration.
    pub config: SchedulerConfig,
    /// Incrementally maintained statistics.
    pub stats: SchedulerStats,
}

impl TensorInferenceScheduler {
    /// Creates a new scheduler with the given configuration.
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            jobs: HashMap::new(),
            next_job_id: 0,
            config,
            stats: SchedulerStats::default(),
        }
    }

    /// Submits a new inference job.
    ///
    /// Returns `Some(job_id)` on success, or `None` if the pending queue is full.
    pub fn submit(
        &mut self,
        goal: &str,
        priority: u32,
        deadline_tick: Option<u64>,
        estimated_cost_ms: u64,
        tick: u64,
    ) -> Option<u64> {
        // Reject if pending queue is at capacity.
        let pending_count = self
            .jobs
            .values()
            .filter(|j| j.status == JobStatus::Pending)
            .count();
        if pending_count >= self.config.max_queue_size {
            return None;
        }

        let job_id = self.next_job_id;
        self.next_job_id += 1;

        let job = InferenceJob {
            job_id,
            goal: goal.to_string(),
            priority,
            deadline_tick,
            estimated_cost_ms,
            status: JobStatus::Pending,
            submitted_at_tick: tick,
            started_at_tick: None,
            completed_at_tick: None,
        };

        self.jobs.insert(job_id, job);
        self.stats.total_submitted += 1;
        self.stats.queue_depth += 1;

        Some(job_id)
    }

    /// Advances the scheduler to `current_tick`.
    ///
    /// 1. Expires any Running or Pending jobs whose deadline has passed.
    /// 2. Promotes Pending jobs (highest priority first, ties broken by job_id ascending)
    ///    into Running state until `max_concurrent` slots are filled.
    pub fn tick(&mut self, current_tick: u64) {
        // --- Phase 1: expire overdue jobs ---
        let mut expired_ids: Vec<u64> = Vec::new();
        for job in self.jobs.values() {
            if let Some(dl) = job.deadline_tick {
                if dl < current_tick
                    && (job.status == JobStatus::Running || job.status == JobStatus::Pending)
                {
                    expired_ids.push(job.job_id);
                }
            }
        }
        for id in expired_ids {
            if let Some(job) = self.jobs.get_mut(&id) {
                let was_pending = job.status == JobStatus::Pending;
                let was_running = job.status == JobStatus::Running;
                job.status = JobStatus::Expired;
                self.stats.total_expired += 1;
                if was_pending {
                    self.stats.queue_depth = self.stats.queue_depth.saturating_sub(1);
                }
                if was_running {
                    self.stats.running_count = self.stats.running_count.saturating_sub(1);
                }
            }
        }

        // --- Phase 2: promote pending jobs into running slots ---
        let running_count = self
            .jobs
            .values()
            .filter(|j| j.status == JobStatus::Running)
            .count();
        let available_slots = self.config.max_concurrent.saturating_sub(running_count);
        if available_slots == 0 {
            return;
        }

        // Collect candidates: Pending jobs sorted by (priority DESC, job_id ASC).
        let mut candidates: Vec<u64> = self
            .jobs
            .values()
            .filter(|j| j.status == JobStatus::Pending)
            .map(|j| j.job_id)
            .collect();

        candidates.sort_by(|&a, &b| {
            let ja = &self.jobs[&a];
            let jb = &self.jobs[&b];
            // Higher priority first; equal priority → lower job_id first.
            jb.priority
                .cmp(&ja.priority)
                .then_with(|| ja.job_id.cmp(&jb.job_id))
        });

        for id in candidates.into_iter().take(available_slots) {
            if let Some(job) = self.jobs.get_mut(&id) {
                job.status = JobStatus::Running;
                job.started_at_tick = Some(current_tick);
                self.stats.queue_depth = self.stats.queue_depth.saturating_sub(1);
                self.stats.running_count += 1;
            }
        }
    }

    /// Marks a Running job as Completed.
    ///
    /// Returns `false` if the job is not found or is not currently Running.
    pub fn complete(&mut self, job_id: u64, tick: u64) -> bool {
        match self.jobs.get_mut(&job_id) {
            Some(job) if job.status == JobStatus::Running => {
                job.status = JobStatus::Completed;
                job.completed_at_tick = Some(tick);
                self.stats.total_completed += 1;
                self.stats.running_count = self.stats.running_count.saturating_sub(1);
                true
            }
            _ => false,
        }
    }

    /// Cancels a Pending or Running job.
    ///
    /// Returns `false` if the job is not found or is already in a terminal state.
    pub fn cancel(&mut self, job_id: u64) -> bool {
        match self.jobs.get_mut(&job_id) {
            Some(job) if !job.is_terminal() => {
                let was_pending = job.status == JobStatus::Pending;
                let was_running = job.status == JobStatus::Running;
                job.status = JobStatus::Cancelled;
                self.stats.total_cancelled += 1;
                if was_pending {
                    self.stats.queue_depth = self.stats.queue_depth.saturating_sub(1);
                }
                if was_running {
                    self.stats.running_count = self.stats.running_count.saturating_sub(1);
                }
                true
            }
            _ => false,
        }
    }

    /// Returns the number of jobs currently in Pending state.
    pub fn queue_depth(&self) -> usize {
        self.jobs
            .values()
            .filter(|j| j.status == JobStatus::Pending)
            .count()
    }

    /// Returns all currently Running jobs, sorted ascending by job_id.
    pub fn running_jobs(&self) -> Vec<&InferenceJob> {
        let mut running: Vec<&InferenceJob> = self
            .jobs
            .values()
            .filter(|j| j.status == JobStatus::Running)
            .collect();
        running.sort_by_key(|j| j.job_id);
        running
    }

    /// Looks up a job by ID.
    pub fn job(&self, job_id: u64) -> Option<&InferenceJob> {
        self.jobs.get(&job_id)
    }

    /// Returns a reference to the current scheduler statistics.
    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_scheduler() -> TensorInferenceScheduler {
        TensorInferenceScheduler::new(SchedulerConfig::default())
    }

    // 1. new() starts empty
    #[test]
    fn test_new_starts_empty() {
        let sched = default_scheduler();
        assert!(sched.jobs.is_empty());
        assert_eq!(sched.next_job_id, 0);
        assert_eq!(sched.queue_depth(), 0);
        assert!(sched.running_jobs().is_empty());
    }

    // 2. submit creates Pending job
    #[test]
    fn test_submit_creates_pending_job() {
        let mut sched = default_scheduler();
        let id = sched
            .submit("parent(X,Y)", 10, None, 50, 0)
            .expect("test: should succeed");
        let job = sched.job(id).expect("test: should succeed");
        assert_eq!(job.status, JobStatus::Pending);
        assert_eq!(job.goal, "parent(X,Y)");
        assert_eq!(job.priority, 10);
        assert_eq!(job.submitted_at_tick, 0);
        assert!(job.started_at_tick.is_none());
        assert!(job.completed_at_tick.is_none());
    }

    // 3. submit returns None when queue full
    #[test]
    fn test_submit_returns_none_when_queue_full() {
        let config = SchedulerConfig {
            max_concurrent: 4,
            max_queue_size: 2,
        };
        let mut sched = TensorInferenceScheduler::new(config);
        assert!(sched.submit("goal1", 1, None, 10, 0).is_some());
        assert!(sched.submit("goal2", 1, None, 10, 0).is_some());
        // Third submission should be rejected
        assert!(sched.submit("goal3", 1, None, 10, 0).is_none());
    }

    // 4. submit increments total_submitted
    #[test]
    fn test_submit_increments_total_submitted() {
        let mut sched = default_scheduler();
        sched.submit("g1", 1, None, 10, 0);
        assert_eq!(sched.stats().total_submitted, 1);
        sched.submit("g2", 1, None, 10, 0);
        assert_eq!(sched.stats().total_submitted, 2);
    }

    // 5. tick starts Pending jobs up to max_concurrent
    #[test]
    fn test_tick_starts_pending_up_to_max_concurrent() {
        let config = SchedulerConfig {
            max_concurrent: 2,
            max_queue_size: 256,
        };
        let mut sched = TensorInferenceScheduler::new(config);
        sched.submit("g1", 1, None, 10, 0);
        sched.submit("g2", 1, None, 10, 0);
        sched.submit("g3", 1, None, 10, 0);
        sched.tick(1);
        let running = sched.running_jobs();
        assert_eq!(running.len(), 2);
        assert_eq!(sched.queue_depth(), 1);
    }

    // 6. tick respects priority order
    #[test]
    fn test_tick_respects_priority_order() {
        let config = SchedulerConfig {
            max_concurrent: 1,
            max_queue_size: 256,
        };
        let mut sched = TensorInferenceScheduler::new(config);
        let low_id = sched
            .submit("low", 1, None, 10, 0)
            .expect("test: should succeed");
        let high_id = sched
            .submit("high", 100, None, 10, 0)
            .expect("test: should succeed");
        sched.tick(1);
        // High priority job should be running
        assert_eq!(
            sched.job(high_id).expect("test: should succeed").status,
            JobStatus::Running
        );
        assert_eq!(
            sched.job(low_id).expect("test: should succeed").status,
            JobStatus::Pending
        );
    }

    // 7. tick breaks ties by job_id ascending
    #[test]
    fn test_tick_breaks_ties_by_job_id_ascending() {
        let config = SchedulerConfig {
            max_concurrent: 1,
            max_queue_size: 256,
        };
        let mut sched = TensorInferenceScheduler::new(config);
        let first_id = sched
            .submit("first", 5, None, 10, 0)
            .expect("test: should succeed");
        let second_id = sched
            .submit("second", 5, None, 10, 0)
            .expect("test: should succeed");
        sched.tick(1);
        // Lower job_id should win on tie
        assert_eq!(
            sched.job(first_id).expect("test: should succeed").status,
            JobStatus::Running
        );
        assert_eq!(
            sched.job(second_id).expect("test: should succeed").status,
            JobStatus::Pending
        );
    }

    // 8. tick expires Running jobs past deadline
    #[test]
    fn test_tick_expires_running_jobs_past_deadline() {
        let mut sched = default_scheduler();
        let id = sched
            .submit("goal", 1, Some(5), 10, 0)
            .expect("test: should succeed");
        sched.tick(1); // starts the job
        assert_eq!(
            sched.job(id).expect("test: should succeed").status,
            JobStatus::Running
        );
        sched.tick(6); // deadline 5 < current_tick 6 → expire
        assert_eq!(
            sched.job(id).expect("test: should succeed").status,
            JobStatus::Expired
        );
    }

    // 9. tick expires Pending jobs past deadline
    #[test]
    fn test_tick_expires_pending_jobs_past_deadline() {
        let config = SchedulerConfig {
            max_concurrent: 0, // no slots → jobs stay Pending
            max_queue_size: 256,
        };
        let mut sched = TensorInferenceScheduler::new(config);
        let id = sched
            .submit("goal", 1, Some(3), 10, 0)
            .expect("test: should succeed");
        sched.tick(4); // deadline 3 < 4 → expire
        assert_eq!(
            sched.job(id).expect("test: should succeed").status,
            JobStatus::Expired
        );
    }

    // 10. tick does not start expired jobs
    #[test]
    fn test_tick_does_not_start_expired_jobs() {
        let mut sched = default_scheduler();
        let id = sched
            .submit("goal", 1, Some(2), 10, 0)
            .expect("test: should succeed");
        sched.tick(3); // expire and attempt to start in same tick
        assert_eq!(
            sched.job(id).expect("test: should succeed").status,
            JobStatus::Expired
        );
    }

    // 11. complete sets Completed
    #[test]
    fn test_complete_sets_completed() {
        let mut sched = default_scheduler();
        let id = sched
            .submit("goal", 1, None, 10, 0)
            .expect("test: should succeed");
        sched.tick(1);
        assert!(sched.complete(id, 5));
        assert_eq!(
            sched.job(id).expect("test: should succeed").status,
            JobStatus::Completed
        );
        assert_eq!(
            sched
                .job(id)
                .expect("test: should succeed")
                .completed_at_tick,
            Some(5)
        );
    }

    // 12. complete returns false for unknown job
    #[test]
    fn test_complete_false_for_unknown_job() {
        let mut sched = default_scheduler();
        assert!(!sched.complete(9999, 1));
    }

    // 13. complete returns false for non-Running job
    #[test]
    fn test_complete_false_for_non_running_job() {
        let mut sched = default_scheduler();
        let id = sched
            .submit("goal", 1, None, 10, 0)
            .expect("test: should succeed");
        // Job is still Pending — not started yet
        assert!(!sched.complete(id, 1));
    }

    // 14. cancel Pending job
    #[test]
    fn test_cancel_pending_job() {
        let mut sched = default_scheduler();
        let id = sched
            .submit("goal", 1, None, 10, 0)
            .expect("test: should succeed");
        assert!(sched.cancel(id));
        assert_eq!(
            sched.job(id).expect("test: should succeed").status,
            JobStatus::Cancelled
        );
        assert_eq!(sched.stats().total_cancelled, 1);
    }

    // 15. cancel Running job
    #[test]
    fn test_cancel_running_job() {
        let mut sched = default_scheduler();
        let id = sched
            .submit("goal", 1, None, 10, 0)
            .expect("test: should succeed");
        sched.tick(1);
        assert_eq!(
            sched.job(id).expect("test: should succeed").status,
            JobStatus::Running
        );
        assert!(sched.cancel(id));
        assert_eq!(
            sched.job(id).expect("test: should succeed").status,
            JobStatus::Cancelled
        );
    }

    // 16. cancel returns false for terminal job
    #[test]
    fn test_cancel_false_for_terminal_job() {
        let mut sched = default_scheduler();
        let id = sched
            .submit("goal", 1, None, 10, 0)
            .expect("test: should succeed");
        sched.tick(1);
        sched.complete(id, 2);
        assert!(!sched.cancel(id)); // already Completed
    }

    // 17. latency_ticks computed correctly
    #[test]
    fn test_latency_ticks_correct() {
        let mut sched = default_scheduler();
        let id = sched
            .submit("goal", 1, None, 10, 0)
            .expect("test: should succeed");
        sched.tick(3);
        sched.complete(id, 10);
        let latency = sched.job(id).expect("test: should succeed").latency_ticks();
        assert_eq!(latency, Some(10)); // 10 - 0 = 10
    }

    // 18. is_terminal for each terminal status
    #[test]
    fn test_is_terminal_for_each_status() {
        let make_job = |status: JobStatus| InferenceJob {
            job_id: 0,
            goal: "g".to_string(),
            priority: 0,
            deadline_tick: None,
            estimated_cost_ms: 0,
            status,
            submitted_at_tick: 0,
            started_at_tick: None,
            completed_at_tick: None,
        };

        assert!(!make_job(JobStatus::Pending).is_terminal());
        assert!(!make_job(JobStatus::Running).is_terminal());
        assert!(make_job(JobStatus::Completed).is_terminal());
        assert!(make_job(JobStatus::Expired).is_terminal());
        assert!(make_job(JobStatus::Cancelled).is_terminal());
    }

    // 19. queue_depth counts Pending only
    #[test]
    fn test_queue_depth_counts_pending_only() {
        let mut sched = default_scheduler();
        sched.submit("g1", 1, None, 10, 0);
        sched.submit("g2", 1, None, 10, 0);
        assert_eq!(sched.queue_depth(), 2);
        sched.tick(1); // starts up to max_concurrent (4) — both start
                       // Both should now be Running, queue_depth = 0
        assert_eq!(sched.queue_depth(), 0);
    }

    // 20. running_jobs sorted by job_id
    #[test]
    fn test_running_jobs_sorted_by_job_id() {
        let mut sched = default_scheduler();
        sched.submit("g1", 5, None, 10, 0);
        sched.submit("g2", 10, None, 10, 0); // higher priority → started first
        sched.submit("g3", 1, None, 10, 0);
        sched.tick(1);
        let running = sched.running_jobs();
        let ids: Vec<u64> = running.iter().map(|j| j.job_id).collect();
        // Should be sorted ascending regardless of start order
        let mut expected = ids.clone();
        expected.sort();
        assert_eq!(ids, expected);
    }

    // 21. stats completion_rate correct
    #[test]
    fn test_stats_completion_rate_correct() {
        let mut sched = default_scheduler();
        let id1 = sched
            .submit("g1", 1, None, 10, 0)
            .expect("test: should succeed");
        let id2 = sched
            .submit("g2", 1, None, 10, 0)
            .expect("test: should succeed");
        sched.tick(1);
        sched.complete(id1, 2);
        sched.cancel(id2);
        // 1 completed, 0 expired, 1 cancelled → rate = 0.5
        let rate = sched.stats().completion_rate();
        assert!((rate - 0.5).abs() < f64::EPSILON);
    }

    // 22. stats total_expired increments
    #[test]
    fn test_stats_total_expired_increments() {
        let mut sched = default_scheduler();
        sched.submit("g1", 1, Some(2), 10, 0);
        sched.submit("g2", 1, Some(2), 10, 0);
        assert_eq!(sched.stats().total_expired, 0);
        sched.tick(3);
        assert_eq!(sched.stats().total_expired, 2);
    }

    // 23. completion_rate returns 0.0 when no terminal jobs
    #[test]
    fn test_completion_rate_zero_when_no_terminal() {
        let sched = default_scheduler();
        assert_eq!(sched.stats().completion_rate(), 0.0);
    }

    // 24. stats queue_depth tracks correctly through lifecycle
    #[test]
    fn test_stats_queue_depth_lifecycle() {
        let mut sched = default_scheduler();
        sched.submit("g1", 1, None, 10, 0);
        sched.submit("g2", 1, None, 10, 0);
        assert_eq!(sched.stats.queue_depth, 2);
        sched.tick(1); // both start
        assert_eq!(sched.stats.queue_depth, 0);
    }

    // 25. stats running_count tracks correctly
    #[test]
    fn test_stats_running_count_tracks() {
        let mut sched = default_scheduler();
        let id1 = sched
            .submit("g1", 1, None, 10, 0)
            .expect("test: should succeed");
        sched.tick(1);
        assert_eq!(sched.stats.running_count, 1);
        sched.complete(id1, 2);
        assert_eq!(sched.stats.running_count, 0);
    }
}
