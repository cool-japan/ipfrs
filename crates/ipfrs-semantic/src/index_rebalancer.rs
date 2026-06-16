//! Embedding Index Rebalancer
//!
//! Plans and tracks the execution of a rebalancing operation across HNSW index
//! shards, moving vectors from overloaded shards to underloaded ones.
//!
//! # Overview
//!
//! The rebalancer operates in three phases:
//!
//! 1. **Analysis** – [`EmbeddingIndexRebalancer::plan_rebalance`] inspects the
//!    load factor of each shard and identifies those that are either overloaded
//!    (above [`RebalancerConfig::overload_threshold`]) or underloaded (below
//!    [`RebalancerConfig::underload_threshold`]).
//!
//! 2. **Planning** – For every overloaded shard the planner pairs it with the
//!    underloaded shard that has the most available capacity, creates one or more
//!    [`MoveTask`]s (each bounded by [`RebalancerConfig::max_moves_per_task`]),
//!    and assembles them into a [`RebalancePlan`].
//!
//! 3. **Tracking** – Callers drive the plan forward by calling
//!    [`EmbeddingIndexRebalancer::update_task_status`] as each task progresses
//!    through [`MoveStatus::Pending`] → [`MoveStatus::InProgress`] →
//!    [`MoveStatus::Completed`] (or [`MoveStatus::Failed`]).

// ---------------------------------------------------------------------------
// MoveStatus
// ---------------------------------------------------------------------------

/// Lifecycle state of a single [`MoveTask`].
#[derive(Clone, Debug, PartialEq)]
pub enum MoveStatus {
    /// The task has been created but not yet started.
    Pending,
    /// The task is currently being executed.
    InProgress {
        /// Unix timestamp (seconds) at which execution began.
        started_at_secs: u64,
    },
    /// The task finished successfully.
    Completed {
        /// Unix timestamp (seconds) at which execution finished.
        finished_at_secs: u64,
    },
    /// The task failed.
    Failed {
        /// Human-readable reason for the failure.
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// ShardLoad
// ---------------------------------------------------------------------------

/// Snapshot of the load on a single HNSW index shard.
#[derive(Debug, Clone)]
pub struct ShardLoad {
    /// Unique identifier for the shard.
    pub shard_id: String,
    /// Number of vectors currently stored in the shard.
    pub vector_count: u64,
    /// Maximum number of vectors the shard can hold.
    pub capacity: u64,
}

impl ShardLoad {
    /// Returns the fraction of capacity that is currently in use.
    ///
    /// The denominator is clamped to at least 1 so that zero-capacity shards
    /// return 0.0 rather than NaN.
    pub fn load_factor(&self) -> f64 {
        self.vector_count as f64 / self.capacity.max(1) as f64
    }

    /// Returns `true` when the shard's load factor exceeds `threshold`.
    pub fn is_overloaded(&self, threshold: f64) -> bool {
        self.load_factor() > threshold
    }

    /// Returns `true` when the shard's load factor is below `threshold`.
    pub fn is_underloaded(&self, threshold: f64) -> bool {
        self.load_factor() < threshold
    }

    /// Returns the number of additional vectors the shard can accept.
    ///
    /// Uses saturating subtraction so that over-capacity shards return 0.
    pub fn available_capacity(&self) -> u64 {
        self.capacity.saturating_sub(self.vector_count)
    }
}

// ---------------------------------------------------------------------------
// MoveTask
// ---------------------------------------------------------------------------

/// A single unit of rebalancing work: move `vector_count` vectors from one
/// shard to another.
#[derive(Debug, Clone)]
pub struct MoveTask {
    /// Monotonically increasing identifier assigned by the rebalancer.
    pub task_id: u64,
    /// Source shard from which vectors will be moved.
    pub from_shard: String,
    /// Destination shard to which vectors will be moved.
    pub to_shard: String,
    /// Number of vectors to be moved by this task.
    pub vector_count: u64,
    /// Current lifecycle state of the task.
    pub status: MoveStatus,
}

// ---------------------------------------------------------------------------
// RebalancePlan
// ---------------------------------------------------------------------------

/// A set of [`MoveTask`]s that collectively rebalance a cluster.
#[derive(Debug, Clone)]
pub struct RebalancePlan {
    /// Ordered list of tasks in this plan.
    pub tasks: Vec<MoveTask>,
    /// Total number of vectors that will be moved if the plan completes fully.
    pub estimated_moves: u64,
}

impl RebalancePlan {
    /// Returns the number of tasks that are still [`MoveStatus::Pending`].
    pub fn pending_tasks(&self) -> usize {
        self.tasks
            .iter()
            .filter(|t| t.status == MoveStatus::Pending)
            .count()
    }

    /// Returns the number of tasks that have reached [`MoveStatus::Completed`].
    pub fn completed_tasks(&self) -> usize {
        self.tasks
            .iter()
            .filter(|t| matches!(t.status, MoveStatus::Completed { .. }))
            .count()
    }

    /// Returns `true` when every task is either [`MoveStatus::Completed`] or
    /// [`MoveStatus::Failed`] (i.e. no more work remains).
    pub fn is_complete(&self) -> bool {
        self.tasks.iter().all(|t| {
            matches!(
                t.status,
                MoveStatus::Completed { .. } | MoveStatus::Failed { .. }
            )
        })
    }
}

// ---------------------------------------------------------------------------
// RebalancerConfig
// ---------------------------------------------------------------------------

/// Tuning knobs for the [`EmbeddingIndexRebalancer`].
#[derive(Debug, Clone)]
pub struct RebalancerConfig {
    /// Load factor above which a shard is considered overloaded (default: 0.85).
    pub overload_threshold: f64,
    /// Load factor below which a shard is considered underloaded (default: 0.40).
    pub underload_threshold: f64,
    /// Upper bound on vectors that a single [`MoveTask`] may move (default:
    /// 10 000).  Larger excesses are split into multiple tasks.
    pub max_moves_per_task: u64,
}

impl Default for RebalancerConfig {
    fn default() -> Self {
        Self {
            overload_threshold: 0.85,
            underload_threshold: 0.40,
            max_moves_per_task: 10_000,
        }
    }
}

// ---------------------------------------------------------------------------
// EmbeddingIndexRebalancer
// ---------------------------------------------------------------------------

/// Plans and tracks rebalancing operations across HNSW index shards.
///
/// # Example
///
/// ```rust
/// use ipfrs_semantic::index_rebalancer::{
///     EmbeddingIndexRebalancer, RebalancerConfig, ShardLoad,
/// };
///
/// let mut rebalancer = EmbeddingIndexRebalancer::new(RebalancerConfig::default());
///
/// let shards = vec![
///     ShardLoad { shard_id: "s0".into(), vector_count: 9_000, capacity: 10_000 },
///     ShardLoad { shard_id: "s1".into(), vector_count: 1_000, capacity: 10_000 },
/// ];
///
/// let plan = rebalancer.plan_rebalance(&shards);
/// assert!(!plan.tasks.is_empty());
/// ```
#[derive(Debug)]
pub struct EmbeddingIndexRebalancer {
    /// Configuration used when producing plans.
    pub config: RebalancerConfig,
    /// History of all plans produced by this rebalancer (newest last).
    pub plans: Vec<RebalancePlan>,
    /// The task ID that will be assigned to the next [`MoveTask`] created.
    pub next_task_id: u64,
}

impl EmbeddingIndexRebalancer {
    /// Creates a new rebalancer with the supplied configuration.
    pub fn new(config: RebalancerConfig) -> Self {
        Self {
            config,
            plans: Vec::new(),
            next_task_id: 0,
        }
    }

    /// Analyses `shards` and produces a [`RebalancePlan`] that moves vectors
    /// from overloaded shards to underloaded ones.
    ///
    /// The plan is also appended to [`Self::plans`] for historical tracking.
    ///
    /// ## Algorithm
    ///
    /// For each overloaded shard the planner:
    ///
    /// 1. Computes the *excess* – the number of vectors above the underload
    ///    threshold: `excess = (load_factor − underload_threshold) × capacity`.
    /// 2. Sorts underloaded shards by available capacity descending and picks
    ///    the one with the most room.
    /// 3. Emits [`MoveTask`]s of at most `max_moves_per_task` vectors until
    ///    the excess is exhausted.
    pub fn plan_rebalance(&mut self, shards: &[ShardLoad]) -> RebalancePlan {
        let overloaded: Vec<&ShardLoad> = shards
            .iter()
            .filter(|s| s.is_overloaded(self.config.overload_threshold))
            .collect();

        // Collect underloaded shards sorted by available capacity descending so
        // that the most spacious shard is preferred.
        let mut underloaded: Vec<ShardLoad> = shards
            .iter()
            .filter(|s| s.is_underloaded(self.config.underload_threshold))
            .cloned()
            .collect();
        underloaded.sort_by_key(|b| std::cmp::Reverse(b.available_capacity()));

        let mut tasks: Vec<MoveTask> = Vec::new();

        'overloaded: for src in &overloaded {
            // Number of vectors we want to move away from this shard.
            let excess = {
                let e = (src.load_factor() - self.config.underload_threshold) * src.capacity as f64;
                e.ceil() as u64
            };

            let mut remaining = excess;

            // Pair with underloaded shards in order of available capacity.
            for dst in underloaded.iter_mut() {
                if remaining == 0 {
                    break;
                }

                let available = dst.available_capacity();
                if available == 0 {
                    continue;
                }

                // How many vectors will actually flow into this destination?
                let to_move = remaining.min(available);
                remaining = remaining.saturating_sub(to_move);

                // Reduce the tracked available capacity so subsequent overloaded
                // shards see an accurate picture.
                dst.vector_count = dst.vector_count.saturating_add(to_move).min(dst.capacity);

                // Split into tasks bounded by max_moves_per_task.
                let cap = self.config.max_moves_per_task;
                let mut left = to_move;
                while left > 0 {
                    let chunk = left.min(cap);
                    left = left.saturating_sub(chunk);

                    tasks.push(MoveTask {
                        task_id: self.next_task_id,
                        from_shard: src.shard_id.clone(),
                        to_shard: dst.shard_id.clone(),
                        vector_count: chunk,
                        status: MoveStatus::Pending,
                    });
                    self.next_task_id += 1;
                }

                if remaining == 0 {
                    continue 'overloaded;
                }
            }
            // If remaining > 0 we ran out of underloaded destinations; that is
            // acceptable – we emit as many tasks as we can.
        }

        let estimated_moves: u64 = tasks.iter().map(|t| t.vector_count).sum();

        let plan = RebalancePlan {
            tasks,
            estimated_moves,
        };

        self.plans.push(plan.clone());
        plan
    }

    /// Updates the status of the task identified by `task_id` across all
    /// historical plans.
    ///
    /// Returns `true` when the task was found and updated, `false` otherwise.
    pub fn update_task_status(&mut self, task_id: u64, status: MoveStatus) -> bool {
        for plan in self.plans.iter_mut() {
            for task in plan.tasks.iter_mut() {
                if task.task_id == task_id {
                    task.status = status;
                    return true;
                }
            }
        }
        false
    }

    /// Returns a reference to the most recent plan if it has not yet completed.
    pub fn active_plan(&self) -> Option<&RebalancePlan> {
        self.plans
            .last()
            .and_then(|p| if p.is_complete() { None } else { Some(p) })
    }

    /// Returns the number of plans that have reached a terminal state (i.e.
    /// all tasks are [`MoveStatus::Completed`] or [`MoveStatus::Failed`]).
    pub fn completed_plans(&self) -> usize {
        self.plans.iter().filter(|p| p.is_complete()).count()
    }

    /// Returns aggregate statistics: `(total_tasks, total_moves_planned)`.
    pub fn stats(&self) -> (usize, u64) {
        let total_tasks: usize = self.plans.iter().map(|p| p.tasks.len()).sum();
        let total_moves: u64 = self.plans.iter().map(|p| p.estimated_moves).sum();
        (total_tasks, total_moves)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_rebalancer() -> EmbeddingIndexRebalancer {
        EmbeddingIndexRebalancer::new(RebalancerConfig::default())
    }

    // 1. new() initialises with the supplied config
    #[test]
    fn test_new_with_config() {
        let cfg = RebalancerConfig {
            overload_threshold: 0.9,
            underload_threshold: 0.3,
            max_moves_per_task: 5_000,
        };
        let rb = EmbeddingIndexRebalancer::new(cfg.clone());
        assert_eq!(rb.config.overload_threshold, cfg.overload_threshold);
        assert_eq!(rb.config.underload_threshold, cfg.underload_threshold);
        assert_eq!(rb.config.max_moves_per_task, cfg.max_moves_per_task);
        assert!(rb.plans.is_empty());
        assert_eq!(rb.next_task_id, 0);
    }

    // 2. load_factor returns vector_count / capacity
    #[test]
    fn test_load_factor() {
        let s = ShardLoad {
            shard_id: "s0".into(),
            vector_count: 500,
            capacity: 1_000,
        };
        assert!((s.load_factor() - 0.5).abs() < f64::EPSILON);
    }

    // 3. load_factor with zero capacity does not divide by zero
    #[test]
    fn test_load_factor_zero_capacity() {
        let s = ShardLoad {
            shard_id: "s0".into(),
            vector_count: 0,
            capacity: 0,
        };
        assert_eq!(s.load_factor(), 0.0);
    }

    // 4. is_overloaded
    #[test]
    fn test_is_overloaded() {
        let s = ShardLoad {
            shard_id: "s0".into(),
            vector_count: 900,
            capacity: 1_000,
        };
        assert!(s.is_overloaded(0.85));
        assert!(!s.is_overloaded(0.95));
    }

    // 5. is_underloaded
    #[test]
    fn test_is_underloaded() {
        let s = ShardLoad {
            shard_id: "s0".into(),
            vector_count: 300,
            capacity: 1_000,
        };
        assert!(s.is_underloaded(0.40));
        assert!(!s.is_underloaded(0.20));
    }

    // 6. available_capacity uses saturating_sub
    #[test]
    fn test_available_capacity_saturating() {
        let s = ShardLoad {
            shard_id: "s0".into(),
            vector_count: 1_200,
            capacity: 1_000,
        };
        assert_eq!(s.available_capacity(), 0);

        let s2 = ShardLoad {
            shard_id: "s1".into(),
            vector_count: 400,
            capacity: 1_000,
        };
        assert_eq!(s2.available_capacity(), 600);
    }

    // 7. plan_rebalance: no overloaded or underloaded shards → empty task list
    #[test]
    fn test_plan_no_action_needed() {
        let mut rb = default_rebalancer();
        let shards = vec![
            ShardLoad {
                shard_id: "s0".into(),
                vector_count: 600,
                capacity: 1_000,
            },
            ShardLoad {
                shard_id: "s1".into(),
                vector_count: 650,
                capacity: 1_000,
            },
        ];
        let plan = rb.plan_rebalance(&shards);
        assert!(plan.tasks.is_empty());
        assert_eq!(plan.estimated_moves, 0);
    }

    // 8. plan_rebalance: one overloaded + one underloaded → at least one task
    #[test]
    fn test_plan_one_overloaded_one_underloaded() {
        let mut rb = default_rebalancer();
        let shards = vec![
            ShardLoad {
                shard_id: "over".into(),
                vector_count: 9_000,
                capacity: 10_000,
            }, // 0.9 > 0.85
            ShardLoad {
                shard_id: "under".into(),
                vector_count: 1_000,
                capacity: 10_000,
            }, // 0.1 < 0.40
        ];
        let plan = rb.plan_rebalance(&shards);
        assert!(!plan.tasks.is_empty());
        assert_eq!(plan.tasks[0].from_shard, "over");
        assert_eq!(plan.tasks[0].to_shard, "under");
        assert!(plan.estimated_moves > 0);
    }

    // 9. plan_rebalance: excess > max_moves_per_task → multiple tasks
    #[test]
    fn test_plan_excess_splits_into_multiple_tasks() {
        let cfg = RebalancerConfig {
            overload_threshold: 0.5,
            underload_threshold: 0.1,
            max_moves_per_task: 1_000,
        };
        let mut rb = EmbeddingIndexRebalancer::new(cfg);
        let shards = vec![
            ShardLoad {
                shard_id: "src".into(),
                vector_count: 8_000,
                capacity: 10_000,
            }, // 0.8 > 0.5
            ShardLoad {
                shard_id: "dst".into(),
                vector_count: 500,
                capacity: 10_000,
            }, // 0.05 < 0.1
        ];
        let plan = rb.plan_rebalance(&shards);
        // excess ≈ (0.8 − 0.1) × 10 000 = 7 000 vectors → 7 tasks of 1 000
        assert!(plan.tasks.len() >= 2, "expected multiple tasks");
        for t in &plan.tasks {
            assert!(t.vector_count <= 1_000, "task exceeds max_moves_per_task");
        }
    }

    // 10. task_ids are monotonically increasing across plans
    #[test]
    fn test_task_ids_monotonically_increasing() {
        let mut rb = default_rebalancer();
        let shards = vec![
            ShardLoad {
                shard_id: "over".into(),
                vector_count: 9_000,
                capacity: 10_000,
            },
            ShardLoad {
                shard_id: "under".into(),
                vector_count: 1_000,
                capacity: 10_000,
            },
        ];
        let plan1 = rb.plan_rebalance(&shards);
        let plan2 = rb.plan_rebalance(&shards);

        let ids1: Vec<u64> = plan1.tasks.iter().map(|t| t.task_id).collect();
        let ids2: Vec<u64> = plan2.tasks.iter().map(|t| t.task_id).collect();

        // IDs within each plan are strictly increasing
        for w in ids1.windows(2) {
            assert!(w[0] < w[1]);
        }
        for w in ids2.windows(2) {
            assert!(w[0] < w[1]);
        }

        // The second plan's first ID is strictly greater than the first plan's
        // last ID (global monotonicity).
        if let (Some(&last1), Some(&first2)) = (ids1.last(), ids2.first()) {
            assert!(first2 > last1);
        }
    }

    // 11. estimated_moves equals sum of vector_count over all tasks
    #[test]
    fn test_estimated_moves_correct() {
        let mut rb = default_rebalancer();
        let shards = vec![
            ShardLoad {
                shard_id: "over".into(),
                vector_count: 9_000,
                capacity: 10_000,
            },
            ShardLoad {
                shard_id: "under".into(),
                vector_count: 1_000,
                capacity: 10_000,
            },
        ];
        let plan = rb.plan_rebalance(&shards);
        let sum: u64 = plan.tasks.iter().map(|t| t.vector_count).sum();
        assert_eq!(plan.estimated_moves, sum);
    }

    // 12. plan is pushed to the history
    #[test]
    fn test_plan_history_pushed() {
        let mut rb = default_rebalancer();
        let shards = vec![ShardLoad {
            shard_id: "s0".into(),
            vector_count: 600,
            capacity: 1_000,
        }];
        rb.plan_rebalance(&shards);
        rb.plan_rebalance(&shards);
        assert_eq!(rb.plans.len(), 2);
    }

    // 13. update_task_status: known task → returns true
    #[test]
    fn test_update_task_status_found() {
        let mut rb = default_rebalancer();
        let shards = vec![
            ShardLoad {
                shard_id: "over".into(),
                vector_count: 9_000,
                capacity: 10_000,
            },
            ShardLoad {
                shard_id: "under".into(),
                vector_count: 1_000,
                capacity: 10_000,
            },
        ];
        let plan = rb.plan_rebalance(&shards);
        let tid = plan.tasks[0].task_id;
        let found = rb.update_task_status(
            tid,
            MoveStatus::Completed {
                finished_at_secs: 1_000,
            },
        );
        assert!(found);
        // Verify the change was actually applied.
        let updated = rb
            .plans
            .iter()
            .flat_map(|p| p.tasks.iter())
            .find(|t| t.task_id == tid)
            .expect("task must exist");
        assert!(matches!(
            updated.status,
            MoveStatus::Completed {
                finished_at_secs: 1_000
            }
        ));
    }

    // 14. update_task_status: unknown task → returns false
    #[test]
    fn test_update_task_status_not_found() {
        let mut rb = default_rebalancer();
        let found = rb.update_task_status(9999, MoveStatus::Pending);
        assert!(!found);
    }

    // 15. is_complete: all tasks Completed → true
    #[test]
    fn test_is_complete_all_completed() {
        let tasks = vec![
            MoveTask {
                task_id: 0,
                from_shard: "a".into(),
                to_shard: "b".into(),
                vector_count: 100,
                status: MoveStatus::Completed {
                    finished_at_secs: 1,
                },
            },
            MoveTask {
                task_id: 1,
                from_shard: "a".into(),
                to_shard: "b".into(),
                vector_count: 200,
                status: MoveStatus::Failed {
                    reason: "timeout".into(),
                },
            },
        ];
        let plan = RebalancePlan {
            estimated_moves: 300,
            tasks,
        };
        assert!(plan.is_complete());
    }

    // 16. is_complete: pending tasks → false
    #[test]
    fn test_is_complete_with_pending() {
        let tasks = vec![MoveTask {
            task_id: 0,
            from_shard: "a".into(),
            to_shard: "b".into(),
            vector_count: 100,
            status: MoveStatus::Pending,
        }];
        let plan = RebalancePlan {
            estimated_moves: 100,
            tasks,
        };
        assert!(!plan.is_complete());
    }

    // 17. pending_tasks / completed_tasks counts
    #[test]
    fn test_pending_and_completed_counts() {
        let tasks = vec![
            MoveTask {
                task_id: 0,
                from_shard: "a".into(),
                to_shard: "b".into(),
                vector_count: 100,
                status: MoveStatus::Pending,
            },
            MoveTask {
                task_id: 1,
                from_shard: "a".into(),
                to_shard: "b".into(),
                vector_count: 100,
                status: MoveStatus::Pending,
            },
            MoveTask {
                task_id: 2,
                from_shard: "a".into(),
                to_shard: "b".into(),
                vector_count: 100,
                status: MoveStatus::Completed {
                    finished_at_secs: 42,
                },
            },
        ];
        let plan = RebalancePlan {
            estimated_moves: 300,
            tasks,
        };
        assert_eq!(plan.pending_tasks(), 2);
        assert_eq!(plan.completed_tasks(), 1);
    }

    // 18. active_plan returns None when the most recent plan is complete
    #[test]
    fn test_active_plan_none_when_complete() {
        let mut rb = default_rebalancer();
        let shards = vec![
            ShardLoad {
                shard_id: "over".into(),
                vector_count: 9_000,
                capacity: 10_000,
            },
            ShardLoad {
                shard_id: "under".into(),
                vector_count: 1_000,
                capacity: 10_000,
            },
        ];
        let plan = rb.plan_rebalance(&shards);

        // Mark all tasks as completed.
        for t in &plan.tasks {
            rb.update_task_status(
                t.task_id,
                MoveStatus::Completed {
                    finished_at_secs: 100,
                },
            );
        }

        assert!(rb.active_plan().is_none());
    }

    // 19. active_plan returns Some when the most recent plan has pending tasks
    #[test]
    fn test_active_plan_some_when_pending() {
        let mut rb = default_rebalancer();
        let shards = vec![
            ShardLoad {
                shard_id: "over".into(),
                vector_count: 9_000,
                capacity: 10_000,
            },
            ShardLoad {
                shard_id: "under".into(),
                vector_count: 1_000,
                capacity: 10_000,
            },
        ];
        rb.plan_rebalance(&shards);
        assert!(rb.active_plan().is_some());
    }

    // 20. completed_plans count
    #[test]
    fn test_completed_plans_count() {
        let mut rb = default_rebalancer();
        // Empty plan (no overloaded shards) → is_complete() = true immediately.
        let shards_balanced = vec![ShardLoad {
            shard_id: "s0".into(),
            vector_count: 600,
            capacity: 1_000,
        }];
        rb.plan_rebalance(&shards_balanced);
        rb.plan_rebalance(&shards_balanced);
        // Both plans have zero tasks, so is_complete() returns true for each.
        assert_eq!(rb.completed_plans(), 2);
    }

    // 21. stats returns correct totals
    #[test]
    fn test_stats() {
        let mut rb = default_rebalancer();
        let shards = vec![
            ShardLoad {
                shard_id: "over".into(),
                vector_count: 9_000,
                capacity: 10_000,
            },
            ShardLoad {
                shard_id: "under".into(),
                vector_count: 1_000,
                capacity: 10_000,
            },
        ];
        rb.plan_rebalance(&shards);
        rb.plan_rebalance(&shards);

        let (total_tasks, total_moves) = rb.stats();
        let expected_tasks: usize = rb.plans.iter().map(|p| p.tasks.len()).sum();
        let expected_moves: u64 = rb.plans.iter().map(|p| p.estimated_moves).sum();
        assert_eq!(total_tasks, expected_tasks);
        assert_eq!(total_moves, expected_moves);
    }
}
