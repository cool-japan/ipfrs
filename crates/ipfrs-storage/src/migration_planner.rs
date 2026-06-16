//! Storage migration planner: schedules and tracks block movements between tiers.
//!
//! This module provides [`StorageMigrationPlanner`] which plans migrations of
//! blocks between storage tiers (NVMe → SSD → HDD → Archive), computes cost
//! estimates, resolves dependency ordering, and supports rollback of failed tasks.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Ordered storage tiers from fastest/most-expensive to slowest/cheapest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StorageTier {
    /// NVMe — fastest and most expensive.
    Nvme = 0,
    /// Solid-state drive.
    Ssd = 1,
    /// Hard-disk drive.
    Hdd = 2,
    /// Long-term archival storage — slowest and cheapest.
    Archive = 3,
}

/// Whether a migration moves a block toward a faster or slower tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MigrationDirection {
    /// Move to a faster (lower-numbered) tier.
    Promote,
    /// Move to a slower (higher-numbered) tier, or stay at the same tier.
    Demote,
}

/// A single unit of migration work: moving one block from one tier to another.
#[derive(Clone, Debug)]
pub struct MigrationTask {
    /// Unique identifier for this task.
    pub task_id: u64,
    /// The block being migrated.
    pub block_id: u64,
    /// Content identifier of the block.
    pub cid: String,
    /// Tier the block is currently stored on.
    pub from_tier: StorageTier,
    /// Tier the block should be moved to.
    pub to_tier: StorageTier,
    /// Whether the migration promotes or demotes the block.
    pub direction: MigrationDirection,
    /// Block size in bytes.
    pub size_bytes: u64,
    /// Estimated migration duration in milliseconds.
    pub estimated_cost_ms: u64,
    /// IDs of tasks that must reach [`MigrationStatus::Completed`] before this
    /// task may start.
    pub depends_on: Vec<u64>,
}

impl MigrationTask {
    /// Returns `true` when the task moves the block toward a faster tier.
    pub fn is_promotion(&self) -> bool {
        self.direction == MigrationDirection::Promote
    }
}

/// Lifecycle state of a [`MigrationTask`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MigrationStatus {
    /// Waiting to be started (dependencies may or may not be resolved).
    Pending,
    /// Currently executing.
    InProgress,
    /// Finished successfully.
    Completed,
    /// Finished with an error.
    Failed,
    /// Was in [`Failed`](MigrationStatus::Failed) and has been rolled back.
    RolledBack,
}

/// A task together with its current execution metadata.
#[derive(Clone, Debug)]
pub struct MigrationRecord {
    /// The task being tracked.
    pub task: MigrationTask,
    /// Current lifecycle state.
    pub status: MigrationStatus,
    /// Logical clock value at which this task transitioned to
    /// [`InProgress`](MigrationStatus::InProgress).
    pub started_at_tick: Option<u64>,
    /// Logical clock value at which this task reached a terminal state.
    pub completed_at_tick: Option<u64>,
}

/// Aggregate counters produced by [`StorageMigrationPlanner::stats`].
#[derive(Clone, Debug, Default)]
pub struct PlannerStats {
    /// Total number of tasks ever planned.
    pub total_tasks: usize,
    /// Tasks in [`Pending`](MigrationStatus::Pending) state.
    pub pending: usize,
    /// Tasks in [`InProgress`](MigrationStatus::InProgress) state.
    pub in_progress: usize,
    /// Tasks in [`Completed`](MigrationStatus::Completed) state.
    pub completed: usize,
    /// Tasks in [`Failed`](MigrationStatus::Failed) or
    /// [`RolledBack`](MigrationStatus::RolledBack) state.
    pub failed: usize,
    /// Sum of `size_bytes` for every completed task.
    pub total_bytes_migrated: u64,
    /// Number of completed promotion tasks.
    pub promotions: u64,
    /// Number of completed demotion tasks.
    pub demotions: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Core planner
// ─────────────────────────────────────────────────────────────────────────────

/// Plans and tracks migrations of blocks across storage tiers.
///
/// # Example
///
/// ```
/// use ipfrs_storage::migration_planner::{StorageMigrationPlanner, StorageTier};
///
/// let mut planner = StorageMigrationPlanner::new();
/// let id = planner.plan_migration(1, "bafyexample".into(), StorageTier::Nvme, StorageTier::Hdd, 4_000_000, vec![]);
/// assert!(planner.start_task(id, 0));
/// assert!(planner.complete_task(id, 10));
/// ```
pub struct StorageMigrationPlanner {
    /// All records keyed by their `task_id`.
    pub records: HashMap<u64, MigrationRecord>,
    /// Counter used to assign the next unique task identifier.
    pub next_task_id: u64,
}

impl StorageMigrationPlanner {
    /// Creates an empty planner with no tasks.
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
            next_task_id: 0,
        }
    }

    /// Plans a migration and returns the new task's ID.
    ///
    /// The direction is determined by comparing `from` and `to`:
    /// - `from > to` → [`Promote`](MigrationDirection::Promote)
    /// - `from <= to` → [`Demote`](MigrationDirection::Demote)
    ///
    /// The cost estimate uses the formula `size_bytes / 1_000_000 + 10` (ms).
    pub fn plan_migration(
        &mut self,
        block_id: u64,
        cid: String,
        from: StorageTier,
        to: StorageTier,
        size_bytes: u64,
        depends_on: Vec<u64>,
    ) -> u64 {
        let task_id = self.next_task_id;
        self.next_task_id += 1;

        let direction = if from > to {
            MigrationDirection::Promote
        } else {
            MigrationDirection::Demote
        };

        let estimated_cost_ms = size_bytes / 1_000_000 + 10;

        let task = MigrationTask {
            task_id,
            block_id,
            cid,
            from_tier: from,
            to_tier: to,
            direction,
            size_bytes,
            estimated_cost_ms,
            depends_on,
        };

        let record = MigrationRecord {
            task,
            status: MigrationStatus::Pending,
            started_at_tick: None,
            completed_at_tick: None,
        };

        self.records.insert(task_id, record);
        task_id
    }

    /// Transitions a task from [`Pending`](MigrationStatus::Pending) to
    /// [`InProgress`](MigrationStatus::InProgress).
    ///
    /// Returns `false` if:
    /// - the task does not exist,
    /// - the task is not in `Pending` state, or
    /// - any dependency is not yet [`Completed`](MigrationStatus::Completed).
    pub fn start_task(&mut self, task_id: u64, current_tick: u64) -> bool {
        // Collect dependency ids first to avoid holding an immutable borrow.
        let depends_on: Vec<u64> = match self.records.get(&task_id) {
            Some(r) if r.status == MigrationStatus::Pending => r.task.depends_on.clone(),
            _ => return false,
        };

        // Verify every dependency has completed.
        for dep_id in &depends_on {
            match self.records.get(dep_id) {
                Some(dep) if dep.status == MigrationStatus::Completed => {}
                _ => return false,
            }
        }

        if let Some(record) = self.records.get_mut(&task_id) {
            record.status = MigrationStatus::InProgress;
            record.started_at_tick = Some(current_tick);
            true
        } else {
            false
        }
    }

    /// Transitions a task from [`InProgress`](MigrationStatus::InProgress) to
    /// [`Completed`](MigrationStatus::Completed).
    ///
    /// Returns `false` if the task is not found or not `InProgress`.
    pub fn complete_task(&mut self, task_id: u64, current_tick: u64) -> bool {
        match self.records.get_mut(&task_id) {
            Some(r) if r.status == MigrationStatus::InProgress => {
                r.status = MigrationStatus::Completed;
                r.completed_at_tick = Some(current_tick);
                true
            }
            _ => false,
        }
    }

    /// Transitions a task to [`Failed`](MigrationStatus::Failed).
    ///
    /// Accepted from either [`Pending`](MigrationStatus::Pending) or
    /// [`InProgress`](MigrationStatus::InProgress). Returns `false` if the task
    /// is not found or already in a terminal state.
    pub fn fail_task(&mut self, task_id: u64) -> bool {
        match self.records.get_mut(&task_id) {
            Some(r)
                if r.status == MigrationStatus::Pending
                    || r.status == MigrationStatus::InProgress =>
            {
                r.status = MigrationStatus::Failed;
                true
            }
            _ => false,
        }
    }

    /// Transitions a task from [`Failed`](MigrationStatus::Failed) to
    /// [`RolledBack`](MigrationStatus::RolledBack).
    ///
    /// Returns `false` if the task is not found or not in `Failed` state.
    pub fn rollback_task(&mut self, task_id: u64) -> bool {
        match self.records.get_mut(&task_id) {
            Some(r) if r.status == MigrationStatus::Failed => {
                r.status = MigrationStatus::RolledBack;
                true
            }
            _ => false,
        }
    }

    /// Returns the IDs of all tasks that are [`Pending`](MigrationStatus::Pending)
    /// and whose every dependency has [`Completed`](MigrationStatus::Completed).
    ///
    /// The list is sorted in ascending order of `task_id`.
    pub fn ready_tasks(&self) -> Vec<u64> {
        let mut ready: Vec<u64> = self
            .records
            .values()
            .filter(|r| {
                if r.status != MigrationStatus::Pending {
                    return false;
                }
                r.task.depends_on.iter().all(|dep_id| {
                    self.records
                        .get(dep_id)
                        .is_some_and(|d| d.status == MigrationStatus::Completed)
                })
            })
            .map(|r| r.task.task_id)
            .collect();

        ready.sort_unstable();
        ready
    }

    /// Returns an immutable reference to the record for `task_id`, or `None`.
    pub fn get_record(&self, task_id: u64) -> Option<&MigrationRecord> {
        self.records.get(&task_id)
    }

    /// Computes aggregate statistics over all tracked records.
    pub fn stats(&self) -> PlannerStats {
        let mut stats = PlannerStats {
            total_tasks: self.records.len(),
            ..Default::default()
        };

        for record in self.records.values() {
            match record.status {
                MigrationStatus::Pending => stats.pending += 1,
                MigrationStatus::InProgress => stats.in_progress += 1,
                MigrationStatus::Completed => {
                    stats.completed += 1;
                    stats.total_bytes_migrated += record.task.size_bytes;
                    match record.task.direction {
                        MigrationDirection::Promote => stats.promotions += 1,
                        MigrationDirection::Demote => stats.demotions += 1,
                    }
                }
                MigrationStatus::Failed | MigrationStatus::RolledBack => stats.failed += 1,
            }
        }

        stats
    }
}

impl Default for StorageMigrationPlanner {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ───────────────────────────────────────────────────────────────

    fn make_planner() -> StorageMigrationPlanner {
        StorageMigrationPlanner::new()
    }

    fn add_simple(
        p: &mut StorageMigrationPlanner,
        from: StorageTier,
        to: StorageTier,
        size: u64,
    ) -> u64 {
        p.plan_migration(1, "bafytest".into(), from, to, size, vec![])
    }

    // ── plan_migration ───────────────────────────────────────────────────────

    #[test]
    fn plan_migration_returns_sequential_ids() {
        let mut p = make_planner();
        let id0 = add_simple(&mut p, StorageTier::Nvme, StorageTier::Hdd, 0);
        let id1 = add_simple(&mut p, StorageTier::Nvme, StorageTier::Hdd, 0);
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
    }

    #[test]
    fn plan_migration_demote_direction() {
        let mut p = make_planner();
        let id = add_simple(&mut p, StorageTier::Nvme, StorageTier::Hdd, 0);
        let record = p.get_record(id).expect("record must exist");
        assert_eq!(record.task.direction, MigrationDirection::Demote);
        assert!(!record.task.is_promotion());
    }

    #[test]
    fn plan_migration_promote_direction() {
        let mut p = make_planner();
        let id = add_simple(&mut p, StorageTier::Archive, StorageTier::Ssd, 0);
        let record = p.get_record(id).expect("record must exist");
        assert_eq!(record.task.direction, MigrationDirection::Promote);
        assert!(record.task.is_promotion());
    }

    #[test]
    fn plan_migration_same_tier_is_demote() {
        let mut p = make_planner();
        let id = add_simple(&mut p, StorageTier::Ssd, StorageTier::Ssd, 0);
        let record = p.get_record(id).expect("record must exist");
        assert_eq!(record.task.direction, MigrationDirection::Demote);
    }

    #[test]
    fn plan_migration_cost_estimate_small() {
        let mut p = make_planner();
        // 500_000 bytes → 500_000 / 1_000_000 = 0, + 10 = 10
        let id = add_simple(&mut p, StorageTier::Nvme, StorageTier::Archive, 500_000);
        let record = p.get_record(id).unwrap();
        assert_eq!(record.task.estimated_cost_ms, 10);
    }

    #[test]
    fn plan_migration_cost_estimate_large() {
        let mut p = make_planner();
        // 5_000_000 bytes → 5 + 10 = 15
        let id = add_simple(&mut p, StorageTier::Ssd, StorageTier::Hdd, 5_000_000);
        let record = p.get_record(id).unwrap();
        assert_eq!(record.task.estimated_cost_ms, 15);
    }

    #[test]
    fn plan_migration_status_is_pending() {
        let mut p = make_planner();
        let id = add_simple(&mut p, StorageTier::Nvme, StorageTier::Hdd, 0);
        let record = p.get_record(id).unwrap();
        assert_eq!(record.status, MigrationStatus::Pending);
        assert!(record.started_at_tick.is_none());
        assert!(record.completed_at_tick.is_none());
    }

    #[test]
    fn plan_migration_stores_block_fields() {
        let mut p = make_planner();
        let id = p.plan_migration(
            42,
            "bafycid".into(),
            StorageTier::Hdd,
            StorageTier::Nvme,
            1_000,
            vec![],
        );
        let record = p.get_record(id).unwrap();
        assert_eq!(record.task.block_id, 42);
        assert_eq!(record.task.cid, "bafycid");
        assert_eq!(record.task.from_tier, StorageTier::Hdd);
        assert_eq!(record.task.to_tier, StorageTier::Nvme);
        assert_eq!(record.task.size_bytes, 1_000);
    }

    // ── start_task ───────────────────────────────────────────────────────────

    #[test]
    fn start_task_sets_in_progress() {
        let mut p = make_planner();
        let id = add_simple(&mut p, StorageTier::Ssd, StorageTier::Hdd, 0);
        assert!(p.start_task(id, 5));
        let record = p.get_record(id).unwrap();
        assert_eq!(record.status, MigrationStatus::InProgress);
        assert_eq!(record.started_at_tick, Some(5));
    }

    #[test]
    fn start_task_unknown_id_returns_false() {
        let mut p = make_planner();
        assert!(!p.start_task(999, 0));
    }

    #[test]
    fn start_task_already_in_progress_returns_false() {
        let mut p = make_planner();
        let id = add_simple(&mut p, StorageTier::Ssd, StorageTier::Hdd, 0);
        assert!(p.start_task(id, 1));
        assert!(!p.start_task(id, 2));
    }

    #[test]
    fn start_task_blocked_by_pending_dependency() {
        let mut p = make_planner();
        let dep = add_simple(&mut p, StorageTier::Nvme, StorageTier::Ssd, 0);
        let child = p.plan_migration(
            2,
            "baf".into(),
            StorageTier::Ssd,
            StorageTier::Hdd,
            0,
            vec![dep],
        );
        assert!(!p.start_task(child, 0));
    }

    #[test]
    fn start_task_blocked_by_in_progress_dependency() {
        let mut p = make_planner();
        let dep = add_simple(&mut p, StorageTier::Nvme, StorageTier::Ssd, 0);
        let child = p.plan_migration(
            2,
            "baf".into(),
            StorageTier::Ssd,
            StorageTier::Hdd,
            0,
            vec![dep],
        );
        p.start_task(dep, 0);
        assert!(!p.start_task(child, 1));
    }

    #[test]
    fn start_task_unblocked_when_dependency_completes() {
        let mut p = make_planner();
        let dep = add_simple(&mut p, StorageTier::Nvme, StorageTier::Ssd, 0);
        let child = p.plan_migration(
            2,
            "baf".into(),
            StorageTier::Ssd,
            StorageTier::Hdd,
            0,
            vec![dep],
        );
        p.start_task(dep, 0);
        p.complete_task(dep, 1);
        assert!(p.start_task(child, 2));
    }

    // ── complete_task ────────────────────────────────────────────────────────

    #[test]
    fn complete_task_sets_completed() {
        let mut p = make_planner();
        let id = add_simple(&mut p, StorageTier::Ssd, StorageTier::Archive, 0);
        p.start_task(id, 0);
        assert!(p.complete_task(id, 10));
        let record = p.get_record(id).unwrap();
        assert_eq!(record.status, MigrationStatus::Completed);
        assert_eq!(record.completed_at_tick, Some(10));
    }

    #[test]
    fn complete_task_not_in_progress_returns_false() {
        let mut p = make_planner();
        let id = add_simple(&mut p, StorageTier::Ssd, StorageTier::Archive, 0);
        assert!(!p.complete_task(id, 1));
    }

    #[test]
    fn complete_task_unknown_id_returns_false() {
        let mut p = make_planner();
        assert!(!p.complete_task(999, 0));
    }

    // ── fail_task ────────────────────────────────────────────────────────────

    #[test]
    fn fail_task_from_pending() {
        let mut p = make_planner();
        let id = add_simple(&mut p, StorageTier::Nvme, StorageTier::Hdd, 0);
        assert!(p.fail_task(id));
        assert_eq!(p.get_record(id).unwrap().status, MigrationStatus::Failed);
    }

    #[test]
    fn fail_task_from_in_progress() {
        let mut p = make_planner();
        let id = add_simple(&mut p, StorageTier::Nvme, StorageTier::Hdd, 0);
        p.start_task(id, 0);
        assert!(p.fail_task(id));
        assert_eq!(p.get_record(id).unwrap().status, MigrationStatus::Failed);
    }

    #[test]
    fn fail_task_from_completed_returns_false() {
        let mut p = make_planner();
        let id = add_simple(&mut p, StorageTier::Nvme, StorageTier::Hdd, 0);
        p.start_task(id, 0);
        p.complete_task(id, 1);
        assert!(!p.fail_task(id));
    }

    #[test]
    fn fail_task_unknown_id_returns_false() {
        let mut p = make_planner();
        assert!(!p.fail_task(999));
    }

    // ── rollback_task ────────────────────────────────────────────────────────

    #[test]
    fn rollback_task_from_failed_succeeds() {
        let mut p = make_planner();
        let id = add_simple(&mut p, StorageTier::Nvme, StorageTier::Hdd, 0);
        p.fail_task(id);
        assert!(p.rollback_task(id));
        assert_eq!(
            p.get_record(id).unwrap().status,
            MigrationStatus::RolledBack
        );
    }

    #[test]
    fn rollback_task_from_pending_returns_false() {
        let mut p = make_planner();
        let id = add_simple(&mut p, StorageTier::Nvme, StorageTier::Hdd, 0);
        assert!(!p.rollback_task(id));
    }

    #[test]
    fn rollback_task_from_completed_returns_false() {
        let mut p = make_planner();
        let id = add_simple(&mut p, StorageTier::Nvme, StorageTier::Hdd, 0);
        p.start_task(id, 0);
        p.complete_task(id, 1);
        assert!(!p.rollback_task(id));
    }

    #[test]
    fn rollback_task_idempotency_returns_false() {
        // After RolledBack, a second rollback attempt must fail.
        let mut p = make_planner();
        let id = add_simple(&mut p, StorageTier::Nvme, StorageTier::Hdd, 0);
        p.fail_task(id);
        p.rollback_task(id);
        assert!(!p.rollback_task(id));
    }

    // ── ready_tasks ──────────────────────────────────────────────────────────

    #[test]
    fn ready_tasks_no_deps_all_pending() {
        let mut p = make_planner();
        let id0 = add_simple(&mut p, StorageTier::Nvme, StorageTier::Hdd, 0);
        let id1 = add_simple(&mut p, StorageTier::Ssd, StorageTier::Archive, 0);
        let ready = p.ready_tasks();
        assert_eq!(ready, vec![id0, id1]);
    }

    #[test]
    fn ready_tasks_sorted_ascending() {
        let mut p = make_planner();
        for _ in 0..5 {
            add_simple(&mut p, StorageTier::Nvme, StorageTier::Hdd, 0);
        }
        let ready = p.ready_tasks();
        let mut sorted = ready.clone();
        sorted.sort_unstable();
        assert_eq!(ready, sorted);
    }

    #[test]
    fn ready_tasks_excludes_in_progress() {
        let mut p = make_planner();
        let id0 = add_simple(&mut p, StorageTier::Nvme, StorageTier::Hdd, 0);
        let id1 = add_simple(&mut p, StorageTier::Ssd, StorageTier::Archive, 0);
        p.start_task(id0, 0);
        let ready = p.ready_tasks();
        assert_eq!(ready, vec![id1]);
    }

    #[test]
    fn ready_tasks_excludes_completed() {
        let mut p = make_planner();
        let id0 = add_simple(&mut p, StorageTier::Nvme, StorageTier::Hdd, 0);
        p.start_task(id0, 0);
        p.complete_task(id0, 1);
        assert!(p.ready_tasks().is_empty());
    }

    #[test]
    fn ready_tasks_with_dependency_chain() {
        let mut p = make_planner();
        let a = add_simple(&mut p, StorageTier::Nvme, StorageTier::Ssd, 0);
        let b = p.plan_migration(
            2,
            "b".into(),
            StorageTier::Ssd,
            StorageTier::Hdd,
            0,
            vec![a],
        );
        let c = p.plan_migration(
            3,
            "c".into(),
            StorageTier::Hdd,
            StorageTier::Archive,
            0,
            vec![b],
        );

        // Only `a` is ready initially.
        assert_eq!(p.ready_tasks(), vec![a]);

        p.start_task(a, 0);
        p.complete_task(a, 1);
        // Now `b` is ready.
        assert_eq!(p.ready_tasks(), vec![b]);

        p.start_task(b, 2);
        p.complete_task(b, 3);
        // Now `c` is ready.
        assert_eq!(p.ready_tasks(), vec![c]);
    }

    // ── stats ────────────────────────────────────────────────────────────────

    #[test]
    fn stats_counts_by_status() {
        let mut p = make_planner();
        let a = add_simple(&mut p, StorageTier::Nvme, StorageTier::Hdd, 1_000);
        let b = add_simple(&mut p, StorageTier::Ssd, StorageTier::Archive, 2_000);
        let c = add_simple(&mut p, StorageTier::Hdd, StorageTier::Nvme, 3_000);
        let d = add_simple(&mut p, StorageTier::Archive, StorageTier::Ssd, 500);

        p.start_task(a, 0);
        p.complete_task(a, 1);
        p.start_task(b, 2);
        p.fail_task(b);
        // c stays Pending; d stays Pending

        let s = p.stats();
        assert_eq!(s.total_tasks, 4);
        assert_eq!(s.completed, 1);
        assert_eq!(s.failed, 1);
        assert_eq!(s.pending, 2);
        assert_eq!(s.in_progress, 0);
        let _ = c;
        let _ = d;
    }

    #[test]
    fn stats_total_bytes_migrated_accumulates() {
        let mut p = make_planner();
        let a = add_simple(&mut p, StorageTier::Nvme, StorageTier::Hdd, 1_000_000);
        let b = add_simple(&mut p, StorageTier::Ssd, StorageTier::Archive, 2_000_000);

        p.start_task(a, 0);
        p.complete_task(a, 1);
        p.start_task(b, 2);
        p.complete_task(b, 3);

        assert_eq!(p.stats().total_bytes_migrated, 3_000_000);
    }

    #[test]
    fn stats_bytes_not_counted_for_failed() {
        let mut p = make_planner();
        let a = add_simple(&mut p, StorageTier::Nvme, StorageTier::Hdd, 5_000_000);
        p.start_task(a, 0);
        p.fail_task(a);

        assert_eq!(p.stats().total_bytes_migrated, 0);
    }

    #[test]
    fn stats_promotions_and_demotions() {
        let mut p = make_planner();
        // Demote
        let d0 = add_simple(&mut p, StorageTier::Nvme, StorageTier::Hdd, 0);
        let d1 = add_simple(&mut p, StorageTier::Ssd, StorageTier::Archive, 0);
        // Promote
        let p0 = add_simple(&mut p, StorageTier::Hdd, StorageTier::Ssd, 0);

        for id in [d0, d1, p0] {
            p.start_task(id, 0);
            p.complete_task(id, 1);
        }

        let s = p.stats();
        assert_eq!(s.demotions, 2);
        assert_eq!(s.promotions, 1);
    }

    #[test]
    fn stats_rolledback_counted_as_failed() {
        let mut p = make_planner();
        let id = add_simple(&mut p, StorageTier::Nvme, StorageTier::Hdd, 0);
        p.fail_task(id);
        p.rollback_task(id);

        let s = p.stats();
        assert_eq!(s.failed, 1);
        assert_eq!(s.completed, 0);
    }

    #[test]
    fn stats_empty_planner() {
        let p = make_planner();
        let s = p.stats();
        assert_eq!(s.total_tasks, 0);
        assert_eq!(s.pending, 0);
        assert_eq!(s.total_bytes_migrated, 0);
    }

    // ── get_record ───────────────────────────────────────────────────────────

    #[test]
    fn get_record_missing_returns_none() {
        let p = make_planner();
        assert!(p.get_record(0).is_none());
    }

    // ── edge cases ───────────────────────────────────────────────────────────

    #[test]
    fn multiple_deps_all_must_complete() {
        let mut p = make_planner();
        let a = add_simple(&mut p, StorageTier::Nvme, StorageTier::Ssd, 0);
        let b = add_simple(&mut p, StorageTier::Nvme, StorageTier::Hdd, 0);
        let c = p.plan_migration(
            10,
            "c".into(),
            StorageTier::Ssd,
            StorageTier::Archive,
            0,
            vec![a, b],
        );

        // Both a and b pending: c not ready.
        assert!(!p.ready_tasks().contains(&c));

        p.start_task(a, 0);
        p.complete_task(a, 1);
        // Only a done: c still not ready.
        assert!(!p.ready_tasks().contains(&c));

        p.start_task(b, 2);
        p.complete_task(b, 3);
        // Both done: c ready.
        assert!(p.ready_tasks().contains(&c));
    }

    #[test]
    fn default_impl_is_empty() {
        let p = StorageMigrationPlanner::default();
        assert_eq!(p.stats().total_tasks, 0);
    }
}
