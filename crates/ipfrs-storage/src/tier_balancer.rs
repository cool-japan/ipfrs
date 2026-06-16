//! StorageTierBalancer — monitors utilization across storage tiers and generates
//! rebalancing plans that move blocks to meet target utilization ratios.
//!
//! The balancer operates over four tier kinds (NVMe → SSD → HDD → Archive),
//! each assigned a capacity, current usage, and a desired utilization ratio.
//! When a tier is over its target the balancer selects blocks from that tier
//! and schedules them for migration to the most-available under-target tier.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_storage::tier_balancer::{
//!     StorageTierBalancer, TierKind, TierStatus,
//! };
//!
//! let mut balancer = StorageTierBalancer::new();
//!
//! balancer.add_tier(TierStatus {
//!     kind: TierKind::Nvme,
//!     capacity_bytes: 1_000,
//!     used_bytes: 900,
//!     target_ratio: 0.7,
//! });
//! balancer.add_tier(TierStatus {
//!     kind: TierKind::Ssd,
//!     capacity_bytes: 10_000,
//!     used_bytes: 1_000,
//!     target_ratio: 0.8,
//! });
//!
//! let candidates = vec![
//!     ("bafy1".to_string(), 200_u64, TierKind::Nvme),
//! ];
//! let tasks = balancer.plan_rebalance(candidates);
//! assert_eq!(tasks.len(), 1);
//! assert_eq!(tasks[0].from_tier, TierKind::Nvme);
//! assert_eq!(tasks[0].to_tier, TierKind::Ssd);
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// TierKind
// ---------------------------------------------------------------------------

/// The four storage tier kinds ordered from fastest/most-expensive to
/// slowest/cheapest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TierKind {
    /// NVMe — fastest, most expensive.
    Nvme,
    /// Solid-state drive.
    Ssd,
    /// Hard-disk drive.
    Hdd,
    /// Archive (tape / deep cold storage) — slowest, cheapest.
    Archive,
}

// ---------------------------------------------------------------------------
// TierStatus
// ---------------------------------------------------------------------------

/// Live utilization snapshot for a single storage tier.
#[derive(Debug, Clone)]
pub struct TierStatus {
    /// Which tier this status belongs to.
    pub kind: TierKind,
    /// Total capacity of the tier in bytes.
    pub capacity_bytes: u64,
    /// Number of bytes currently in use.
    pub used_bytes: u64,
    /// Desired utilization ratio in `[0.0, 1.0]`.
    pub target_ratio: f64,
}

impl TierStatus {
    /// Returns the current utilization ratio (`used / capacity`).
    /// Returns `0.0` when `capacity_bytes` is zero.
    pub fn utilization(&self) -> f64 {
        if self.capacity_bytes == 0 {
            return 0.0;
        }
        self.used_bytes as f64 / self.capacity_bytes as f64
    }

    /// Returns the number of bytes still available (`capacity - used`),
    /// saturating at zero.
    pub fn free_bytes(&self) -> u64 {
        self.capacity_bytes.saturating_sub(self.used_bytes)
    }

    /// Returns `true` when the tier is above its target utilization ratio.
    pub fn is_over_target(&self) -> bool {
        self.utilization() > self.target_ratio
    }

    /// Returns how many bytes exceed the target usage level.
    ///
    /// When the tier is at or below its target the result is `0`.
    pub fn excess_bytes(&self) -> u64 {
        if !self.is_over_target() {
            return 0;
        }
        let target_used = (self.capacity_bytes as f64 * self.target_ratio) as u64;
        self.used_bytes.saturating_sub(target_used)
    }
}

// ---------------------------------------------------------------------------
// MoveTask
// ---------------------------------------------------------------------------

/// A scheduled request to migrate one block from one tier to another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoveTask {
    /// Unique identifier assigned by the balancer.
    pub task_id: u64,
    /// Content identifier of the block to be moved.
    pub cid: String,
    /// Size of the block in bytes.
    pub size_bytes: u64,
    /// Source tier.
    pub from_tier: TierKind,
    /// Destination tier.
    pub to_tier: TierKind,
    /// Scheduling priority — higher value means the task should run first.
    pub priority: u32,
}

// ---------------------------------------------------------------------------
// BalancerStats
// ---------------------------------------------------------------------------

/// Aggregate statistics about the current state of the balancer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BalancerStats {
    /// Total number of tiers registered with the balancer.
    pub total_tiers: usize,
    /// Number of tiers that are currently above their target utilization.
    pub over_target_tiers: usize,
    /// Number of move tasks waiting in the pending queue.
    pub total_move_tasks: usize,
    /// Sum of `size_bytes` across all pending move tasks.
    pub total_bytes_to_move: u64,
}

// ---------------------------------------------------------------------------
// StorageTierBalancer
// ---------------------------------------------------------------------------

/// Balances data across storage tiers by monitoring utilization and generating
/// rebalancing plans.
///
/// Rebalancing proceeds tier-by-tier: for each over-target tier the balancer
/// selects candidate blocks (in the order supplied by the caller) until the
/// estimated excess is covered, creating a [`MoveTask`] for each block.  The
/// destination is always the under-target tier with the most free bytes.
pub struct StorageTierBalancer {
    /// Registered tiers indexed by [`TierKind`].
    pub tiers: HashMap<TierKind, TierStatus>,
    /// Outstanding move tasks, sorted by `priority` descending (highest first).
    pub pending_tasks: Vec<MoveTask>,
    /// Monotonically increasing counter used to generate unique task IDs.
    pub next_task_id: u64,
    /// Total number of tasks that have been completed via [`Self::complete_task`].
    pub total_completed_tasks: u64,
}

impl StorageTierBalancer {
    /// Creates a new, empty balancer.
    pub fn new() -> Self {
        Self {
            tiers: HashMap::new(),
            pending_tasks: Vec::new(),
            next_task_id: 1,
            total_completed_tasks: 0,
        }
    }

    /// Registers a tier with the balancer.  Overwrites any existing entry for
    /// the same [`TierKind`].
    pub fn add_tier(&mut self, status: TierStatus) {
        self.tiers.insert(status.kind, status);
    }

    /// Updates the `used_bytes` counter for an existing tier.
    ///
    /// If the tier has not been registered the call is a no-op.
    pub fn update_usage(&mut self, kind: TierKind, used_bytes: u64) {
        if let Some(tier) = self.tiers.get_mut(&kind) {
            tier.used_bytes = used_bytes;
        }
    }

    /// Plans a rebalancing run given a list of candidate blocks.
    ///
    /// # Parameters
    ///
    /// * `candidates` — `(cid, size_bytes, current_tier)` tuples describing
    ///   blocks eligible for migration.
    ///
    /// # Returns
    ///
    /// Shared references to the newly created [`MoveTask`]s in their sorted
    /// (priority-descending) order within [`Self::pending_tasks`].
    pub fn plan_rebalance(&mut self, candidates: Vec<(String, u64, TierKind)>) -> Vec<&MoveTask> {
        // Snapshot which tiers are over target and by how much.
        // We also maintain a *virtual* free-bytes map so that as we schedule
        // moves we account for the bytes we are about to add to the destination.
        let mut virtual_free: HashMap<TierKind, u64> = self
            .tiers
            .iter()
            .map(|(k, v)| (*k, v.free_bytes()))
            .collect();

        // Compute per-tier excess at the start of the plan (static snapshot).
        let over_target_kinds: Vec<TierKind> = {
            let mut ks: Vec<TierKind> = self
                .tiers
                .values()
                .filter(|t| t.is_over_target())
                .map(|t| t.kind)
                .collect();
            ks.sort(); // deterministic order (NVMe → Archive)
            ks
        };

        let mut new_tasks: Vec<MoveTask> = Vec::new();

        for src_kind in over_target_kinds {
            let excess = match self.tiers.get(&src_kind) {
                Some(t) => t.excess_bytes(),
                None => continue,
            };
            if excess == 0 {
                continue;
            }

            let mut remaining = excess;

            // Walk candidates that live in this tier.
            for (cid, size, tier) in &candidates {
                if *tier != src_kind {
                    continue;
                }
                if remaining == 0 {
                    break;
                }

                // Pick the under-target destination with the most virtual free bytes,
                // excluding the source tier itself.
                let dest_kind = self
                    .tiers
                    .values()
                    .filter(|t| t.kind != src_kind && !t.is_over_target())
                    .max_by(|a, b| {
                        let fa = virtual_free.get(&a.kind).copied().unwrap_or(0);
                        let fb = virtual_free.get(&b.kind).copied().unwrap_or(0);
                        fa.cmp(&fb)
                    })
                    .map(|t| t.kind);

                let dest_kind = match dest_kind {
                    Some(d) => d,
                    None => break, // no room anywhere — skip rest of this source tier
                };

                let priority = move_priority(src_kind, dest_kind);
                let task_id = self.next_task_id;
                self.next_task_id += 1;

                // Update virtual free bytes for destination.
                let dest_free = virtual_free.entry(dest_kind).or_insert(0);
                *dest_free = dest_free.saturating_sub(*size);

                remaining = remaining.saturating_sub(*size);

                new_tasks.push(MoveTask {
                    task_id,
                    cid: cid.clone(),
                    size_bytes: *size,
                    from_tier: src_kind,
                    to_tier: dest_kind,
                    priority,
                });
            }
        }

        if new_tasks.is_empty() {
            return Vec::new();
        }

        // Record the task IDs we are about to add so we can return references.
        let new_ids: Vec<u64> = new_tasks.iter().map(|t| t.task_id).collect();

        // Merge into pending_tasks and re-sort by priority descending.
        self.pending_tasks.extend(new_tasks);
        self.pending_tasks
            .sort_by_key(|t| std::cmp::Reverse(t.priority));

        // Return references to the newly added tasks (by task_id).
        self.pending_tasks
            .iter()
            .filter(|t| new_ids.contains(&t.task_id))
            .collect()
    }

    /// Marks a task as completed and removes it from the pending queue.
    ///
    /// Returns `true` when the task was found and removed, `false` otherwise.
    pub fn complete_task(&mut self, task_id: u64) -> bool {
        if let Some(pos) = self.pending_tasks.iter().position(|t| t.task_id == task_id) {
            self.pending_tasks.remove(pos);
            self.total_completed_tasks += 1;
            true
        } else {
            false
        }
    }

    /// Returns a reference to the [`TierStatus`] for the given kind, if any.
    pub fn tier_status(&self, kind: TierKind) -> Option<&TierStatus> {
        self.tiers.get(&kind)
    }

    /// Returns aggregate statistics about the balancer's current state.
    pub fn stats(&self) -> BalancerStats {
        let over_target_tiers = self.tiers.values().filter(|t| t.is_over_target()).count();
        let total_bytes_to_move = self.pending_tasks.iter().map(|t| t.size_bytes).sum();
        BalancerStats {
            total_tiers: self.tiers.len(),
            over_target_tiers,
            total_move_tasks: self.pending_tasks.len(),
            total_bytes_to_move,
        }
    }
}

impl Default for StorageTierBalancer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the scheduling priority for a move from `src` to `dst`.
///
/// Priority rules (higher = schedule first):
/// - NVMe → SSD : 10
/// - SSD  → HDD : 5
/// - HDD  → Archive : 1
/// - anything else : 1 (treat as lowest)
fn move_priority(src: TierKind, dst: TierKind) -> u32 {
    match (src, dst) {
        (TierKind::Nvme, TierKind::Ssd) => 10,
        (TierKind::Ssd, TierKind::Hdd) => 5,
        (TierKind::Hdd, TierKind::Archive) => 1,
        _ => 1,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn nvme_status(capacity: u64, used: u64, target: f64) -> TierStatus {
        TierStatus {
            kind: TierKind::Nvme,
            capacity_bytes: capacity,
            used_bytes: used,
            target_ratio: target,
        }
    }

    fn ssd_status(capacity: u64, used: u64, target: f64) -> TierStatus {
        TierStatus {
            kind: TierKind::Ssd,
            capacity_bytes: capacity,
            used_bytes: used,
            target_ratio: target,
        }
    }

    fn hdd_status(capacity: u64, used: u64, target: f64) -> TierStatus {
        TierStatus {
            kind: TierKind::Hdd,
            capacity_bytes: capacity,
            used_bytes: used,
            target_ratio: target,
        }
    }

    fn archive_status(capacity: u64, used: u64, target: f64) -> TierStatus {
        TierStatus {
            kind: TierKind::Archive,
            capacity_bytes: capacity,
            used_bytes: used,
            target_ratio: target,
        }
    }

    // -------------------------------------------------------------------------
    // 1. new() starts empty
    // -------------------------------------------------------------------------
    #[test]
    fn test_new_starts_empty() {
        let b = StorageTierBalancer::new();
        assert!(b.tiers.is_empty());
        assert!(b.pending_tasks.is_empty());
        assert_eq!(b.next_task_id, 1);
        assert_eq!(b.total_completed_tasks, 0);
    }

    // -------------------------------------------------------------------------
    // 2. add_tier stores correctly
    // -------------------------------------------------------------------------
    #[test]
    fn test_add_tier_stores() {
        let mut b = StorageTierBalancer::new();
        b.add_tier(nvme_status(1000, 500, 0.8));
        assert_eq!(b.tiers.len(), 1);
        let t = b.tier_status(TierKind::Nvme).expect("nvme should exist");
        assert_eq!(t.capacity_bytes, 1000);
        assert_eq!(t.used_bytes, 500);
    }

    // -------------------------------------------------------------------------
    // 3. add_tier overwrites existing
    // -------------------------------------------------------------------------
    #[test]
    fn test_add_tier_overwrites() {
        let mut b = StorageTierBalancer::new();
        b.add_tier(nvme_status(1000, 500, 0.8));
        b.add_tier(nvme_status(2000, 100, 0.5));
        assert_eq!(b.tiers.len(), 1);
        let t = b.tier_status(TierKind::Nvme).expect("nvme should exist");
        assert_eq!(t.capacity_bytes, 2000);
        assert_eq!(t.used_bytes, 100);
    }

    // -------------------------------------------------------------------------
    // 4. update_usage changes used_bytes
    // -------------------------------------------------------------------------
    #[test]
    fn test_update_usage_changes_used_bytes() {
        let mut b = StorageTierBalancer::new();
        b.add_tier(nvme_status(1000, 500, 0.8));
        b.update_usage(TierKind::Nvme, 700);
        assert_eq!(b.tier_status(TierKind::Nvme).unwrap().used_bytes, 700);
    }

    // -------------------------------------------------------------------------
    // 5. update_usage no-op for unknown tier
    // -------------------------------------------------------------------------
    #[test]
    fn test_update_usage_noop_unknown() {
        let mut b = StorageTierBalancer::new();
        // Should not panic
        b.update_usage(TierKind::Ssd, 9999);
        assert!(b.tiers.is_empty());
    }

    // -------------------------------------------------------------------------
    // 6. utilization() computed correctly
    // -------------------------------------------------------------------------
    #[test]
    fn test_utilization_computed_correctly() {
        let t = nvme_status(1000, 750, 0.8);
        assert!((t.utilization() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_utilization_zero_capacity() {
        let t = nvme_status(0, 0, 0.8);
        assert_eq!(t.utilization(), 0.0);
    }

    // -------------------------------------------------------------------------
    // 7. free_bytes saturating
    // -------------------------------------------------------------------------
    #[test]
    fn test_free_bytes_saturating() {
        let t = nvme_status(1000, 1200, 0.8);
        assert_eq!(t.free_bytes(), 0); // saturates at 0
        let t2 = nvme_status(1000, 400, 0.8);
        assert_eq!(t2.free_bytes(), 600);
    }

    // -------------------------------------------------------------------------
    // 8. is_over_target true/false
    // -------------------------------------------------------------------------
    #[test]
    fn test_is_over_target_true() {
        let t = nvme_status(1000, 900, 0.8); // utilization=0.9 > 0.8
        assert!(t.is_over_target());
    }

    #[test]
    fn test_is_over_target_false() {
        let t = nvme_status(1000, 700, 0.8); // utilization=0.7 < 0.8
        assert!(!t.is_over_target());
    }

    #[test]
    fn test_is_over_target_exactly_at_boundary() {
        let t = nvme_status(1000, 800, 0.8); // utilization == target_ratio → not over
        assert!(!t.is_over_target());
    }

    // -------------------------------------------------------------------------
    // 9. excess_bytes computed correctly
    // -------------------------------------------------------------------------
    #[test]
    fn test_excess_bytes_over_target() {
        // capacity=1000, used=900, target=0.7 → target_used=700, excess=200
        let t = nvme_status(1000, 900, 0.7);
        assert_eq!(t.excess_bytes(), 200);
    }

    // -------------------------------------------------------------------------
    // 10. excess_bytes 0 when not over target
    // -------------------------------------------------------------------------
    #[test]
    fn test_excess_bytes_zero_when_not_over() {
        let t = nvme_status(1000, 700, 0.8);
        assert_eq!(t.excess_bytes(), 0);
    }

    // -------------------------------------------------------------------------
    // 11. plan_rebalance generates tasks for over-target tier
    // -------------------------------------------------------------------------
    #[test]
    fn test_plan_rebalance_generates_tasks() {
        let mut b = StorageTierBalancer::new();
        // NVMe over target (used=900, target=0.7 → excess=200)
        b.add_tier(nvme_status(1000, 900, 0.7));
        // SSD under target (lots of room)
        b.add_tier(ssd_status(10_000, 1_000, 0.8));

        let candidates = vec![("cid1".to_string(), 250_u64, TierKind::Nvme)];
        let tasks = b.plan_rebalance(candidates);
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].cid, "cid1");
    }

    // -------------------------------------------------------------------------
    // 12. plan_rebalance picks under-target destination
    // -------------------------------------------------------------------------
    #[test]
    fn test_plan_rebalance_picks_under_target_destination() {
        let mut b = StorageTierBalancer::new();
        b.add_tier(nvme_status(1000, 900, 0.7));
        // SSD: under target (util ≈ 0.1, target 0.8)
        b.add_tier(ssd_status(10_000, 1_000, 0.8));
        // HDD: over target (used=9_500, capacity=10_000, util=0.95 > target=0.9)
        b.add_tier(hdd_status(10_000, 9_500, 0.9));

        let candidates = vec![("cid1".to_string(), 100_u64, TierKind::Nvme)];
        let tasks = b.plan_rebalance(candidates);
        assert_eq!(tasks.len(), 1);
        // HDD is over target so SSD must be chosen as destination.
        assert_eq!(tasks[0].to_tier, TierKind::Ssd);
    }

    // -------------------------------------------------------------------------
    // 13. plan_rebalance stops when excess covered
    // -------------------------------------------------------------------------
    #[test]
    fn test_plan_rebalance_stops_when_excess_covered() {
        let mut b = StorageTierBalancer::new();
        // excess = 900 - 700 = 200 bytes
        b.add_tier(nvme_status(1000, 900, 0.7));
        b.add_tier(ssd_status(10_000, 1_000, 0.8));

        // Three candidates; only enough excess for the first one (300 > 200).
        let candidates = vec![
            ("cid1".to_string(), 300_u64, TierKind::Nvme),
            ("cid2".to_string(), 300_u64, TierKind::Nvme),
            ("cid3".to_string(), 300_u64, TierKind::Nvme),
        ];
        let tasks = b.plan_rebalance(candidates);
        // Once the first candidate covers the 200-byte excess, remaining becomes 0
        // and the loop breaks — only 1 task generated.
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].cid, "cid1");
    }

    // -------------------------------------------------------------------------
    // 14. plan_rebalance skips if no under-target destination
    // -------------------------------------------------------------------------
    #[test]
    fn test_plan_rebalance_skips_no_under_target_destination() {
        let mut b = StorageTierBalancer::new();
        b.add_tier(nvme_status(1000, 900, 0.7)); // over target
                                                 // All other tiers are also over target — nowhere to move data.
        b.add_tier(ssd_status(1000, 900, 0.8)); // util=0.9 > 0.8 → over target

        let candidates = vec![("cid1".to_string(), 100_u64, TierKind::Nvme)];
        let tasks = b.plan_rebalance(candidates);
        assert!(tasks.is_empty());
    }

    // -------------------------------------------------------------------------
    // 15. MoveTask priority set correctly (Nvme→Ssd=10)
    // -------------------------------------------------------------------------
    #[test]
    fn test_move_task_priority_nvme_to_ssd() {
        let mut b = StorageTierBalancer::new();
        b.add_tier(nvme_status(1000, 900, 0.7));
        b.add_tier(ssd_status(10_000, 1_000, 0.8));

        let candidates = vec![("cid1".to_string(), 250_u64, TierKind::Nvme)];
        let tasks = b.plan_rebalance(candidates);
        assert_eq!(tasks[0].priority, 10);
    }

    #[test]
    fn test_move_task_priority_ssd_to_hdd() {
        let mut b = StorageTierBalancer::new();
        b.add_tier(ssd_status(1000, 900, 0.7));
        b.add_tier(hdd_status(100_000, 1_000, 0.8));

        let candidates = vec![("cid1".to_string(), 250_u64, TierKind::Ssd)];
        let tasks = b.plan_rebalance(candidates);
        assert_eq!(tasks[0].priority, 5);
    }

    #[test]
    fn test_move_task_priority_hdd_to_archive() {
        let mut b = StorageTierBalancer::new();
        b.add_tier(hdd_status(1000, 900, 0.7));
        b.add_tier(archive_status(1_000_000, 1_000, 0.8));

        let candidates = vec![("cid1".to_string(), 250_u64, TierKind::Hdd)];
        let tasks = b.plan_rebalance(candidates);
        assert_eq!(tasks[0].priority, 1);
    }

    // -------------------------------------------------------------------------
    // 16. complete_task removes from pending
    // -------------------------------------------------------------------------
    #[test]
    fn test_complete_task_removes_from_pending() {
        let mut b = StorageTierBalancer::new();
        b.add_tier(nvme_status(1000, 900, 0.7));
        b.add_tier(ssd_status(10_000, 1_000, 0.8));

        let candidates = vec![("cid1".to_string(), 250_u64, TierKind::Nvme)];
        let tasks = b.plan_rebalance(candidates);
        let task_id = tasks[0].task_id;

        assert_eq!(b.pending_tasks.len(), 1);
        let removed = b.complete_task(task_id);
        assert!(removed);
        assert!(b.pending_tasks.is_empty());
    }

    // -------------------------------------------------------------------------
    // 17. complete_task false for unknown id
    // -------------------------------------------------------------------------
    #[test]
    fn test_complete_task_false_for_unknown() {
        let mut b = StorageTierBalancer::new();
        let result = b.complete_task(9999);
        assert!(!result);
    }

    // -------------------------------------------------------------------------
    // 18. total_completed_tasks increments
    // -------------------------------------------------------------------------
    #[test]
    fn test_total_completed_tasks_increments() {
        let mut b = StorageTierBalancer::new();
        b.add_tier(nvme_status(1000, 900, 0.7));
        b.add_tier(ssd_status(10_000, 1_000, 0.8));

        let candidates = vec![
            ("cid1".to_string(), 100_u64, TierKind::Nvme),
            ("cid2".to_string(), 100_u64, TierKind::Nvme),
        ];
        let tasks = b.plan_rebalance(candidates);
        let id1 = tasks[0].task_id;
        let id2 = tasks[1].task_id;

        b.complete_task(id1);
        assert_eq!(b.total_completed_tasks, 1);
        b.complete_task(id2);
        assert_eq!(b.total_completed_tasks, 2);
    }

    // -------------------------------------------------------------------------
    // 19. stats over_target_tiers count
    // -------------------------------------------------------------------------
    #[test]
    fn test_stats_over_target_tiers_count() {
        let mut b = StorageTierBalancer::new();
        b.add_tier(nvme_status(1000, 900, 0.7)); // over
        b.add_tier(ssd_status(1000, 500, 0.8)); // not over
        b.add_tier(hdd_status(1000, 950, 0.9)); // over

        let stats = b.stats();
        assert_eq!(stats.total_tiers, 3);
        assert_eq!(stats.over_target_tiers, 2);
    }

    // -------------------------------------------------------------------------
    // 20. stats total_bytes_to_move
    // -------------------------------------------------------------------------
    #[test]
    fn test_stats_total_bytes_to_move() {
        let mut b = StorageTierBalancer::new();
        // NVMe over target: excess = 900 - 700 = 200
        b.add_tier(nvme_status(1000, 900, 0.7));
        // SSD under target
        b.add_tier(ssd_status(10_000, 1_000, 0.8));

        let candidates = vec![
            ("cid1".to_string(), 120_u64, TierKind::Nvme),
            ("cid2".to_string(), 120_u64, TierKind::Nvme),
        ];
        b.plan_rebalance(candidates);

        let stats = b.stats();
        // Both candidates are scheduled because 120 < 200 excess, and then
        // 120+120=240 covers the 200-byte excess (second candidate runs because
        // remaining=80 > 0 after first).
        assert_eq!(stats.total_bytes_to_move, 240);
    }

    // -------------------------------------------------------------------------
    // 21. tier_status returns None for unregistered tier
    // -------------------------------------------------------------------------
    #[test]
    fn test_tier_status_none_for_unregistered() {
        let b = StorageTierBalancer::new();
        assert!(b.tier_status(TierKind::Hdd).is_none());
    }

    // -------------------------------------------------------------------------
    // 22. pending_tasks sorted by priority descending
    // -------------------------------------------------------------------------
    #[test]
    fn test_pending_tasks_sorted_by_priority_desc() {
        let mut b = StorageTierBalancer::new();
        // NVMe over target → priority 10 for NVMe→SSD moves
        b.add_tier(nvme_status(1000, 950, 0.7));
        // SSD over target → priority 5 for SSD→HDD moves
        b.add_tier(ssd_status(1000, 950, 0.7));
        // HDD under target (large free space so it wins)
        b.add_tier(hdd_status(1_000_000, 1_000, 0.8));

        let candidates = vec![
            ("cid_nvme".to_string(), 100_u64, TierKind::Nvme),
            ("cid_ssd".to_string(), 100_u64, TierKind::Ssd),
        ];
        b.plan_rebalance(candidates);

        // pending_tasks must be sorted priority desc: 10 before 5
        let priorities: Vec<u32> = b.pending_tasks.iter().map(|t| t.priority).collect();
        for w in priorities.windows(2) {
            assert!(w[0] >= w[1], "tasks not sorted: {:?}", priorities);
        }
    }
}
