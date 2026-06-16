//! Tensor operation scheduler with priority, dependency tracking, and resource accounting.
//!
//! [`TensorOpScheduler`] manages a set of [`TensorOp`] entries. Each operation
//! carries a [`OpPriority`], a list of dependency op-ids that must complete
//! before the operation is eligible to run, and a resource estimate
//! (`estimated_flops`). The scheduler advances a monotonic tick counter and
//! transitions operations through the [`OpStatus`] state machine:
//!
//! ```text
//! Pending  ──(all deps Completed)──►  Ready
//!                                       │
//!            ◄── fail_op ───────────────┤
//!                                       │ start_op
//!                                       ▼
//!                                    Running
//!                                       │
//!            ◄── fail_op ───────────────┤
//!                                       │ complete_op
//!                                       ▼
//!                                   Completed
//! ```
//!
//! # Example
//!
//! ```
//! use ipfrs_tensorlogic::op_scheduler::{TensorOpScheduler, OpPriority, OpStatus};
//!
//! let mut sched = TensorOpScheduler::new();
//!
//! // Enqueue a root operation (no deps).
//! let root = sched.enqueue("matmul".to_string(), OpPriority::High, vec![], 1_000_000);
//!
//! // Enqueue a dependent operation.
//! let dep = sched.enqueue("relu".to_string(), OpPriority::Normal, vec![root], 500_000);
//!
//! // Root is immediately ready (no deps). Advance tick to flush.
//! sched.advance_tick();
//! assert_eq!(sched.ops[&root].status, OpStatus::Ready);
//! // dep still pending because root is not yet Completed.
//! assert_eq!(sched.ops[&dep].status, OpStatus::Pending);
//!
//! // Run and complete root.
//! assert!(sched.start_op(root));
//! assert!(sched.complete_op(root));
//!
//! // Now dep can become Ready.
//! sched.advance_tick();
//! assert_eq!(sched.ops[&dep].status, OpStatus::Ready);
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// OpPriority
// ---------------------------------------------------------------------------

/// Scheduling priority for a tensor operation.
///
/// Higher numeric value = higher urgency. Used by [`TensorOpScheduler::next_ready`]
/// to pick the most urgent ready operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OpPriority {
    /// Background / best-effort work.
    Low = 0,
    /// Default priority for most operations.
    Normal = 1,
    /// Elevated priority; preferred over `Normal` and `Low`.
    High = 2,
    /// Highest priority; must run before everything else.
    Critical = 3,
}

// ---------------------------------------------------------------------------
// OpStatus
// ---------------------------------------------------------------------------

/// Lifecycle state of a scheduled tensor operation.
#[derive(Clone, Debug, PartialEq)]
pub enum OpStatus {
    /// Waiting for one or more dependency operations to complete.
    Pending,
    /// All dependencies are satisfied; operation may be started.
    Ready,
    /// Operation is currently executing.
    Running,
    /// Operation finished successfully.
    Completed,
    /// Operation encountered an error.
    Failed {
        /// Human-readable description of the failure.
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// TensorOp
// ---------------------------------------------------------------------------

/// A single tensor operation tracked by the scheduler.
#[derive(Clone, Debug)]
pub struct TensorOp {
    /// Monotonically increasing identifier assigned at enqueue time.
    pub op_id: u64,
    /// Human-readable operation name (e.g. `"matmul"`, `"relu"`).
    pub name: String,
    /// Scheduling priority of this operation.
    pub priority: OpPriority,
    /// Identifiers of operations that must reach [`OpStatus::Completed`] before
    /// this operation may transition to [`OpStatus::Ready`].
    pub deps: Vec<u64>,
    /// Estimated floating-point operations (used for resource accounting).
    pub estimated_flops: u64,
    /// Current lifecycle state.
    pub status: OpStatus,
    /// Scheduler tick at which this operation was enqueued.
    pub enqueued_at: u64,
    /// Scheduler tick at which this operation transitioned to `Running`.
    pub started_at: Option<u64>,
    /// Scheduler tick at which this operation reached `Completed` or `Failed`.
    pub completed_at: Option<u64>,
}

impl TensorOp {
    /// Returns the number of ticks the operation spent running, or `None` if
    /// the operation has not both started and completed.
    #[must_use]
    pub fn duration_ticks(&self) -> Option<u64> {
        match (self.started_at, self.completed_at) {
            (Some(s), Some(c)) => Some(c.saturating_sub(s)),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// SchedulerStats
// ---------------------------------------------------------------------------

/// Aggregate statistics produced by [`TensorOpScheduler::stats`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchedulerStats {
    /// Total number of operations enqueued since the scheduler was created.
    pub total_enqueued: u64,
    /// Number of operations in [`OpStatus::Completed`] state.
    pub completed: u64,
    /// Number of operations in [`OpStatus::Failed`] state.
    pub failed: u64,
    /// Number of operations currently in [`OpStatus::Pending`] state.
    pub pending: usize,
    /// Number of operations currently in [`OpStatus::Ready`] state.
    pub ready: usize,
    /// Number of operations currently in [`OpStatus::Running`] state.
    pub running: usize,
    /// Sum of `estimated_flops` for all `Completed` operations.
    pub total_flops_completed: u64,
}

// ---------------------------------------------------------------------------
// TensorOpScheduler
// ---------------------------------------------------------------------------

/// Priority-based tensor operation scheduler with dependency tracking.
///
/// See the [module-level documentation](self) for a full usage example.
pub struct TensorOpScheduler {
    /// All registered operations keyed by `op_id`.
    pub ops: HashMap<u64, TensorOp>,
    /// Next op_id to assign.
    next_id: u64,
    /// Current monotonic tick counter.
    tick: u64,
}

impl TensorOpScheduler {
    /// Creates an empty scheduler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ops: HashMap::new(),
            next_id: 0,
            tick: 0,
        }
    }

    /// Enqueues a new operation and returns its assigned `op_id`.
    ///
    /// The operation starts in [`OpStatus::Pending`] regardless of whether its
    /// dependency list is empty. Call [`advance_tick`](Self::advance_tick) to
    /// promote operations with satisfied dependencies to [`OpStatus::Ready`].
    ///
    /// # Arguments
    ///
    /// * `name`     – Human-readable operation name.
    /// * `priority` – Scheduling priority.
    /// * `deps`     – IDs of operations that must complete before this one runs.
    /// * `flops`    – Estimated floating-point operations for resource accounting.
    pub fn enqueue(
        &mut self,
        name: String,
        priority: OpPriority,
        deps: Vec<u64>,
        flops: u64,
    ) -> u64 {
        let op_id = self.next_id;
        self.next_id += 1;

        let op = TensorOp {
            op_id,
            name,
            priority,
            deps,
            estimated_flops: flops,
            status: OpStatus::Pending,
            enqueued_at: self.tick,
            started_at: None,
            completed_at: None,
        };
        self.ops.insert(op_id, op);
        self.tick += 1;
        op_id
    }

    /// Advances the scheduler tick by one and re-evaluates dependency readiness.
    ///
    /// Each [`OpStatus::Pending`] operation whose every dependency is in
    /// [`OpStatus::Completed`] is promoted to [`OpStatus::Ready`].
    pub fn advance_tick(&mut self) {
        self.tick += 1;

        // Collect ids that should transition Pending → Ready.
        // We cannot mutate `self.ops` while iterating it, so we gather the ids
        // first.
        let ids_to_ready: Vec<u64> = self
            .ops
            .iter()
            .filter_map(|(&id, op)| {
                if op.status != OpStatus::Pending {
                    return None;
                }
                let all_done = op.deps.iter().all(|dep_id| {
                    self.ops
                        .get(dep_id)
                        .map(|d| d.status == OpStatus::Completed)
                        .unwrap_or(false)
                });
                if all_done {
                    Some(id)
                } else {
                    None
                }
            })
            .collect();

        for id in ids_to_ready {
            if let Some(op) = self.ops.get_mut(&id) {
                op.status = OpStatus::Ready;
            }
        }
    }

    /// Attempts to transition the operation to [`OpStatus::Running`].
    ///
    /// Returns `true` on success. Returns `false` if the operation does not
    /// exist or is not in [`OpStatus::Ready`] state.
    pub fn start_op(&mut self, op_id: u64) -> bool {
        match self.ops.get_mut(&op_id) {
            Some(op) if op.status == OpStatus::Ready => {
                op.status = OpStatus::Running;
                op.started_at = Some(self.tick);
                true
            }
            _ => false,
        }
    }

    /// Attempts to transition the operation to [`OpStatus::Completed`].
    ///
    /// Returns `true` on success. Returns `false` if the operation does not
    /// exist or is not in [`OpStatus::Running`] state.
    pub fn complete_op(&mut self, op_id: u64) -> bool {
        match self.ops.get_mut(&op_id) {
            Some(op) if op.status == OpStatus::Running => {
                op.status = OpStatus::Completed;
                op.completed_at = Some(self.tick);
                true
            }
            _ => false,
        }
    }

    /// Attempts to transition the operation to [`OpStatus::Failed`].
    ///
    /// The operation must be in [`OpStatus::Running`] or [`OpStatus::Ready`]
    /// state. Returns `true` on success, `false` otherwise.
    pub fn fail_op(&mut self, op_id: u64, reason: String) -> bool {
        match self.ops.get_mut(&op_id) {
            Some(op) if op.status == OpStatus::Running || op.status == OpStatus::Ready => {
                op.status = OpStatus::Failed { reason };
                op.completed_at = Some(self.tick);
                true
            }
            _ => false,
        }
    }

    /// Returns the `op_id` of the highest-priority [`OpStatus::Ready`] operation.
    ///
    /// Ties in priority are broken by the lowest `op_id` (FIFO within priority).
    /// Returns `None` if no operations are currently in `Ready` state.
    #[must_use]
    pub fn next_ready(&self) -> Option<u64> {
        self.ops
            .values()
            .filter(|op| op.status == OpStatus::Ready)
            .max_by(|a, b| {
                // Primary: higher priority wins.
                // Secondary: lower op_id wins (FIFO).
                a.priority
                    .cmp(&b.priority)
                    .then_with(|| b.op_id.cmp(&a.op_id))
            })
            .map(|op| op.op_id)
    }

    /// Returns a snapshot of aggregate scheduler statistics.
    #[must_use]
    pub fn stats(&self) -> SchedulerStats {
        let mut completed: u64 = 0;
        let mut failed: u64 = 0;
        let mut pending: usize = 0;
        let mut ready: usize = 0;
        let mut running: usize = 0;
        let mut total_flops_completed: u64 = 0;

        for op in self.ops.values() {
            match &op.status {
                OpStatus::Pending => pending += 1,
                OpStatus::Ready => ready += 1,
                OpStatus::Running => running += 1,
                OpStatus::Completed => {
                    completed += 1;
                    total_flops_completed =
                        total_flops_completed.saturating_add(op.estimated_flops);
                }
                OpStatus::Failed { .. } => failed += 1,
            }
        }

        SchedulerStats {
            total_enqueued: self.next_id,
            completed,
            failed,
            pending,
            ready,
            running,
            total_flops_completed,
        }
    }
}

impl Default for TensorOpScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helper: create a scheduler, enqueue an op, advance, start, complete.
    // ------------------------------------------------------------------

    fn make_sched() -> TensorOpScheduler {
        TensorOpScheduler::new()
    }

    // 1. Enqueue returns sequential IDs starting from 0.
    #[test]
    fn test_enqueue_sequential_ids() {
        let mut s = make_sched();
        let id0 = s.enqueue("a".to_string(), OpPriority::Normal, vec![], 100);
        let id1 = s.enqueue("b".to_string(), OpPriority::Normal, vec![], 100);
        let id2 = s.enqueue("c".to_string(), OpPriority::Normal, vec![], 100);
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    // 2. Fresh op starts as Pending.
    #[test]
    fn test_enqueue_status_pending() {
        let mut s = make_sched();
        let id = s.enqueue("op".to_string(), OpPriority::Normal, vec![], 0);
        assert_eq!(s.ops[&id].status, OpStatus::Pending);
    }

    // 3. advance_tick promotes dep-free Pending op to Ready.
    #[test]
    fn test_advance_tick_no_deps_becomes_ready() {
        let mut s = make_sched();
        let id = s.enqueue("op".to_string(), OpPriority::High, vec![], 500);
        s.advance_tick();
        assert_eq!(s.ops[&id].status, OpStatus::Ready);
    }

    // 4. Pending op with unsatisfied dep stays Pending after advance_tick.
    #[test]
    fn test_advance_tick_pending_dep_stays_pending() {
        let mut s = make_sched();
        let root = s.enqueue("root".to_string(), OpPriority::Normal, vec![], 0);
        let child = s.enqueue("child".to_string(), OpPriority::Normal, vec![root], 0);
        s.advance_tick();
        // root: no deps → Ready
        assert_eq!(s.ops[&root].status, OpStatus::Ready);
        // child depends on root which is still Ready (not Completed) → stays Pending
        assert_eq!(s.ops[&child].status, OpStatus::Pending);
    }

    // 5. Dependent op becomes Ready only after dep is Completed.
    #[test]
    fn test_advance_tick_after_dep_completed() {
        let mut s = make_sched();
        let root = s.enqueue("root".to_string(), OpPriority::Normal, vec![], 0);
        let child = s.enqueue("child".to_string(), OpPriority::Normal, vec![root], 0);
        s.advance_tick(); // root → Ready, child still Pending
        assert!(s.start_op(root)); // root → Running
        assert!(s.complete_op(root)); // root → Completed
        s.advance_tick(); // child → Ready
        assert_eq!(s.ops[&child].status, OpStatus::Ready);
    }

    // 6. start_op transitions Ready → Running.
    #[test]
    fn test_start_op_ready_to_running() {
        let mut s = make_sched();
        let id = s.enqueue("op".to_string(), OpPriority::Normal, vec![], 0);
        s.advance_tick();
        assert_eq!(s.ops[&id].status, OpStatus::Ready);
        let ok = s.start_op(id);
        assert!(ok);
        assert_eq!(s.ops[&id].status, OpStatus::Running);
    }

    // 7. start_op on Pending op returns false.
    #[test]
    fn test_start_op_pending_fails() {
        let mut s = make_sched();
        let id = s.enqueue("op".to_string(), OpPriority::Normal, vec![], 0);
        // do not advance_tick, op is still Pending
        let ok = s.start_op(id);
        assert!(!ok);
        assert_eq!(s.ops[&id].status, OpStatus::Pending);
    }

    // 8. complete_op transitions Running → Completed.
    #[test]
    fn test_complete_op_running_to_completed() {
        let mut s = make_sched();
        let id = s.enqueue("op".to_string(), OpPriority::Normal, vec![], 42);
        s.advance_tick();
        s.start_op(id);
        let ok = s.complete_op(id);
        assert!(ok);
        assert_eq!(s.ops[&id].status, OpStatus::Completed);
    }

    // 9. complete_op on non-Running op returns false.
    #[test]
    fn test_complete_op_non_running_fails() {
        let mut s = make_sched();
        let id = s.enqueue("op".to_string(), OpPriority::Normal, vec![], 0);
        s.advance_tick(); // op is Ready, not Running
        let ok = s.complete_op(id);
        assert!(!ok);
    }

    // 10. complete_op increments total_flops_completed in stats.
    #[test]
    fn test_complete_op_increments_flops() {
        let mut s = make_sched();
        let id = s.enqueue("op".to_string(), OpPriority::Normal, vec![], 1_000_000);
        s.advance_tick();
        s.start_op(id);
        s.complete_op(id);
        let stats = s.stats();
        assert_eq!(stats.total_flops_completed, 1_000_000);
    }

    // 11. fail_op on Running op transitions to Failed with reason.
    #[test]
    fn test_fail_op_running_to_failed() {
        let mut s = make_sched();
        let id = s.enqueue("op".to_string(), OpPriority::Normal, vec![], 0);
        s.advance_tick();
        s.start_op(id);
        let ok = s.fail_op(id, "OOM".to_string());
        assert!(ok);
        assert_eq!(
            s.ops[&id].status,
            OpStatus::Failed {
                reason: "OOM".to_string()
            }
        );
    }

    // 12. fail_op on Ready op also succeeds.
    #[test]
    fn test_fail_op_ready_to_failed() {
        let mut s = make_sched();
        let id = s.enqueue("op".to_string(), OpPriority::High, vec![], 0);
        s.advance_tick();
        assert_eq!(s.ops[&id].status, OpStatus::Ready);
        let ok = s.fail_op(id, "device lost".to_string());
        assert!(ok);
        assert_eq!(
            s.ops[&id].status,
            OpStatus::Failed {
                reason: "device lost".to_string()
            }
        );
    }

    // 13. fail_op on Pending op returns false.
    #[test]
    fn test_fail_op_pending_fails() {
        let mut s = make_sched();
        let id = s.enqueue("op".to_string(), OpPriority::Normal, vec![], 0);
        // no advance_tick → Pending
        let ok = s.fail_op(id, "err".to_string());
        assert!(!ok);
        assert_eq!(s.ops[&id].status, OpStatus::Pending);
    }

    // 14. next_ready picks highest-priority Ready op.
    #[test]
    fn test_next_ready_highest_priority() {
        let mut s = make_sched();
        let low = s.enqueue("low".to_string(), OpPriority::Low, vec![], 0);
        let high = s.enqueue("high".to_string(), OpPriority::High, vec![], 0);
        let normal = s.enqueue("normal".to_string(), OpPriority::Normal, vec![], 0);
        s.advance_tick();
        let next = s.next_ready().expect("should have a ready op");
        assert_eq!(next, high, "High priority op should be selected");
        // suppress unused warnings
        let _ = (low, normal);
    }

    // 15. next_ready tie-breaking: lower op_id wins (FIFO).
    #[test]
    fn test_next_ready_fifo_tiebreak() {
        let mut s = make_sched();
        // Enqueue three Normal-priority ops; they should be ready after advance.
        let id0 = s.enqueue("first".to_string(), OpPriority::Normal, vec![], 0);
        let id1 = s.enqueue("second".to_string(), OpPriority::Normal, vec![], 0);
        let _id2 = s.enqueue("third".to_string(), OpPriority::Normal, vec![], 0);
        s.advance_tick();
        let next = s.next_ready().expect("should have ready op");
        assert_eq!(next, id0, "Lowest op_id should win tie");
        // Start and complete id0, then check again.
        s.start_op(id0);
        s.complete_op(id0);
        let next2 = s.next_ready().expect("should still have ready ops");
        assert_eq!(next2, id1);
    }

    // 16. next_ready returns None when no ops are Ready.
    #[test]
    fn test_next_ready_none_when_empty() {
        let s = make_sched();
        assert_eq!(s.next_ready(), None);
    }

    // 17. stats counts reflect all status categories correctly.
    #[test]
    fn test_stats_counts() {
        let mut s = make_sched();
        // Enqueue 4 ops
        let a = s.enqueue("a".to_string(), OpPriority::Normal, vec![], 100);
        let b = s.enqueue("b".to_string(), OpPriority::Normal, vec![], 200);
        let _c = s.enqueue("c".to_string(), OpPriority::Normal, vec![a], 300);
        let _d = s.enqueue("d".to_string(), OpPriority::Normal, vec![], 400);
        // a, b, d: no deps → all Pending initially
        s.advance_tick(); // a, b, d → Ready; c still Pending (dep on a)
        s.start_op(a); // a → Running
        s.complete_op(a); // a → Completed
        s.start_op(b); // b → Running
        s.fail_op(b, "timeout".to_string()); // b → Failed
                                             // Now: a=Completed, b=Failed, c=Pending, d=Ready
        s.advance_tick(); // c → Ready (a is Completed)
                          // d still Ready, c now Ready
        let stats = s.stats();
        assert_eq!(stats.total_enqueued, 4);
        assert_eq!(stats.completed, 1);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.pending, 0);
        assert_eq!(stats.ready, 2); // c and d
        assert_eq!(stats.running, 0);
        assert_eq!(stats.total_flops_completed, 100); // only 'a' completed
    }

    // 18. duration_ticks returns None when not started.
    #[test]
    fn test_duration_ticks_not_started() {
        let mut s = make_sched();
        let id = s.enqueue("op".to_string(), OpPriority::Normal, vec![], 0);
        assert_eq!(s.ops[&id].duration_ticks(), None);
    }

    // 19. duration_ticks returns None when started but not completed.
    #[test]
    fn test_duration_ticks_started_not_completed() {
        let mut s = make_sched();
        let id = s.enqueue("op".to_string(), OpPriority::Normal, vec![], 0);
        s.advance_tick();
        s.start_op(id);
        assert_eq!(s.ops[&id].duration_ticks(), None);
    }

    // 20. duration_ticks returns correct delta after completion.
    #[test]
    fn test_duration_ticks_completed() {
        let mut s = make_sched();
        let id = s.enqueue("op".to_string(), OpPriority::Normal, vec![], 0);
        s.advance_tick(); // tick advances; op becomes Ready
        s.start_op(id); // started_at = current tick
        let started_tick = s.ops[&id].started_at.expect("started_at must be set");
        s.advance_tick(); // move tick forward
        s.advance_tick(); // move tick forward again
                          // complete_op uses current tick as completed_at
        s.complete_op(id);
        let completed_tick = s.ops[&id].completed_at.expect("completed_at must be set");
        let expected = completed_tick - started_tick;
        assert_eq!(s.ops[&id].duration_ticks(), Some(expected));
        assert!(expected > 0);
    }

    // 21. Critical priority beats High, Normal, Low.
    #[test]
    fn test_next_ready_critical_beats_all() {
        let mut s = make_sched();
        let _low = s.enqueue("low".to_string(), OpPriority::Low, vec![], 0);
        let _norm = s.enqueue("norm".to_string(), OpPriority::Normal, vec![], 0);
        let _high = s.enqueue("high".to_string(), OpPriority::High, vec![], 0);
        let crit = s.enqueue("crit".to_string(), OpPriority::Critical, vec![], 0);
        s.advance_tick();
        assert_eq!(s.next_ready(), Some(crit));
    }

    // 22. Chain of 3 ops completes sequentially.
    #[test]
    fn test_chain_three_ops() {
        let mut s = make_sched();
        let a = s.enqueue("a".to_string(), OpPriority::Normal, vec![], 10);
        let b = s.enqueue("b".to_string(), OpPriority::Normal, vec![a], 20);
        let c = s.enqueue("c".to_string(), OpPriority::Normal, vec![b], 30);

        s.advance_tick();
        assert_eq!(s.ops[&a].status, OpStatus::Ready);
        assert_eq!(s.ops[&b].status, OpStatus::Pending);
        assert_eq!(s.ops[&c].status, OpStatus::Pending);

        s.start_op(a);
        s.complete_op(a);
        s.advance_tick();
        assert_eq!(s.ops[&b].status, OpStatus::Ready);
        assert_eq!(s.ops[&c].status, OpStatus::Pending);

        s.start_op(b);
        s.complete_op(b);
        s.advance_tick();
        assert_eq!(s.ops[&c].status, OpStatus::Ready);

        s.start_op(c);
        s.complete_op(c);

        let stats = s.stats();
        assert_eq!(stats.completed, 3);
        assert_eq!(stats.total_flops_completed, 60);
    }
}
