//! Storage Replication Tracker
//!
//! Tracks replication factor of stored blocks across peers, detects
//! under-replicated blocks, and generates replication tasks to restore
//! desired redundancy.

use std::collections::HashMap;

// ── ReplicaLocation ──────────────────────────────────────────────────────────

/// Records where a block replica lives and when it was last confirmed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplicaLocation {
    /// Peer that holds this replica.
    pub peer_id: String,
    /// Logical tick at which the replica was last confirmed present.
    pub confirmed_at_tick: u64,
    /// Whether this peer is the designated primary for the block.
    pub is_primary: bool,
}

// ── ReplicationStatus ────────────────────────────────────────────────────────

/// Health status of a block's replication level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplicationStatus {
    /// Actual replicas equal desired replicas.
    Healthy,
    /// Actual replicas are fewer than desired but at least one exists.
    UnderReplicated,
    /// Actual replicas exceed desired replicas.
    OverReplicated,
    /// No replicas exist at all.
    Critical,
}

// ── BlockReplicationEntry ────────────────────────────────────────────────────

/// Replication record for a single content block.
#[derive(Clone, Debug)]
pub struct BlockReplicationEntry {
    /// Stable numeric identifier for this block.
    pub block_id: u64,
    /// Content identifier (CID) of the block.
    pub cid: String,
    /// How many replicas this block should have.
    pub desired_replicas: usize,
    /// Currently known replica locations.
    pub replicas: Vec<ReplicaLocation>,
}

impl BlockReplicationEntry {
    /// Returns the current number of replicas.
    pub fn actual_replicas(&self) -> usize {
        self.replicas.len()
    }

    /// Derives the health status from actual vs. desired replica counts.
    pub fn status(&self) -> ReplicationStatus {
        let actual = self.replicas.len();
        if actual == 0 {
            ReplicationStatus::Critical
        } else if actual < self.desired_replicas {
            ReplicationStatus::UnderReplicated
        } else if actual > self.desired_replicas {
            ReplicationStatus::OverReplicated
        } else {
            ReplicationStatus::Healthy
        }
    }

    /// Number of additional replicas needed to reach the desired count.
    ///
    /// Returns `0` when already at or above the desired level.
    pub fn deficit(&self) -> usize {
        self.desired_replicas.saturating_sub(self.replicas.len())
    }

    /// Number of replicas above the desired count.
    ///
    /// Returns `0` when at or below the desired level.
    pub fn surplus(&self) -> usize {
        self.replicas.len().saturating_sub(self.desired_replicas)
    }
}

// ── ReplicationTask ──────────────────────────────────────────────────────────

/// Work item requesting that additional copies of a block be created.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplicationTask {
    /// Block to be replicated.
    pub block_id: u64,
    /// Content identifier of the block.
    pub cid: String,
    /// How many new copies must be produced.
    pub needed_copies: usize,
    /// Scheduling priority: Critical = 100, UnderReplicated = 50.
    pub priority: u32,
}

// ── ReplicationStats ─────────────────────────────────────────────────────────

/// Aggregate replication statistics across all tracked blocks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplicationStats {
    /// Total number of tracked blocks.
    pub total_blocks: usize,
    /// Blocks with status [`ReplicationStatus::Healthy`].
    pub healthy: usize,
    /// Blocks with status [`ReplicationStatus::UnderReplicated`].
    pub under_replicated: usize,
    /// Blocks with status [`ReplicationStatus::OverReplicated`].
    pub over_replicated: usize,
    /// Blocks with status [`ReplicationStatus::Critical`].
    pub critical: usize,
    /// Total replica count across all blocks.
    pub total_replicas: usize,
}

// ── StorageReplicationTracker ─────────────────────────────────────────────────

/// Tracks replication of content blocks across a peer network.
pub struct StorageReplicationTracker {
    /// All tracked block entries keyed by block_id.
    pub entries: HashMap<u64, BlockReplicationEntry>,
    /// Counter used to issue monotonically increasing block IDs.
    pub next_block_id: u64,
    /// Replication factor applied when `register_block` receives `None`.
    pub default_desired_replicas: usize,
}

impl StorageReplicationTracker {
    /// Creates a new tracker with the given default replication factor.
    pub fn new(default_desired_replicas: usize) -> Self {
        Self {
            entries: HashMap::new(),
            next_block_id: 0,
            default_desired_replicas,
        }
    }

    /// Registers a new block and returns its assigned `block_id`.
    ///
    /// If `desired_replicas` is `None`, the tracker's default is used.
    pub fn register_block(&mut self, cid: String, desired_replicas: Option<usize>) -> u64 {
        let block_id = self.next_block_id;
        self.next_block_id += 1;
        let desired = desired_replicas.unwrap_or(self.default_desired_replicas);
        self.entries.insert(
            block_id,
            BlockReplicationEntry {
                block_id,
                cid,
                desired_replicas: desired,
                replicas: Vec::new(),
            },
        );
        block_id
    }

    /// Records that `peer_id` holds a replica of `block_id` at `current_tick`.
    ///
    /// - Returns `false` if `block_id` is not tracked.
    /// - If the peer already has a replica, its tick is updated; otherwise a
    ///   new [`ReplicaLocation`] is appended.
    /// - Returns `true` on success.
    pub fn add_replica(
        &mut self,
        block_id: u64,
        peer_id: String,
        current_tick: u64,
        is_primary: bool,
    ) -> bool {
        let Some(entry) = self.entries.get_mut(&block_id) else {
            return false;
        };

        if let Some(loc) = entry.replicas.iter_mut().find(|l| l.peer_id == peer_id) {
            loc.confirmed_at_tick = current_tick;
            loc.is_primary = is_primary;
        } else {
            entry.replicas.push(ReplicaLocation {
                peer_id,
                confirmed_at_tick: current_tick,
                is_primary,
            });
        }
        true
    }

    /// Removes the replica held by `peer_id` from `block_id`.
    ///
    /// Returns `false` when the block or peer is not found.
    pub fn remove_replica(&mut self, block_id: u64, peer_id: &str) -> bool {
        let Some(entry) = self.entries.get_mut(&block_id) else {
            return false;
        };
        let before = entry.replicas.len();
        entry.replicas.retain(|l| l.peer_id != peer_id);
        entry.replicas.len() < before
    }

    /// Returns all blocks that are [`ReplicationStatus::UnderReplicated`] or
    /// [`ReplicationStatus::Critical`], sorted by `block_id` ascending.
    pub fn under_replicated_blocks(&self) -> Vec<&BlockReplicationEntry> {
        let mut blocks: Vec<&BlockReplicationEntry> = self
            .entries
            .values()
            .filter(|e| {
                matches!(
                    e.status(),
                    ReplicationStatus::UnderReplicated | ReplicationStatus::Critical
                )
            })
            .collect();
        blocks.sort_by_key(|e| e.block_id);
        blocks
    }

    /// Generates replication tasks for every block with a positive deficit.
    ///
    /// Tasks are ordered by priority descending then `block_id` ascending.
    pub fn generate_tasks(&self) -> Vec<ReplicationTask> {
        let mut tasks: Vec<ReplicationTask> = self
            .entries
            .values()
            .filter(|e| e.deficit() > 0)
            .map(|e| {
                let priority = match e.status() {
                    ReplicationStatus::Critical => 100,
                    _ => 50,
                };
                ReplicationTask {
                    block_id: e.block_id,
                    cid: e.cid.clone(),
                    needed_copies: e.deficit(),
                    priority,
                }
            })
            .collect();

        tasks.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| a.block_id.cmp(&b.block_id))
        });
        tasks
    }

    /// Returns a reference to the entry for `block_id`, if tracked.
    pub fn get_entry(&self, block_id: u64) -> Option<&BlockReplicationEntry> {
        self.entries.get(&block_id)
    }

    /// Computes aggregate statistics across all tracked blocks.
    pub fn stats(&self) -> ReplicationStats {
        let mut stats = ReplicationStats {
            total_blocks: self.entries.len(),
            healthy: 0,
            under_replicated: 0,
            over_replicated: 0,
            critical: 0,
            total_replicas: 0,
        };
        for entry in self.entries.values() {
            stats.total_replicas += entry.replicas.len();
            match entry.status() {
                ReplicationStatus::Healthy => stats.healthy += 1,
                ReplicationStatus::UnderReplicated => stats.under_replicated += 1,
                ReplicationStatus::OverReplicated => stats.over_replicated += 1,
                ReplicationStatus::Critical => stats.critical += 1,
            }
        }
        stats
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tracker() -> StorageReplicationTracker {
        StorageReplicationTracker::new(3)
    }

    // ── register_block ────────────────────────────────────────────────────────

    #[test]
    fn test_register_block_creates_entry() {
        let mut t = tracker();
        let id = t.register_block("cid-a".into(), None);
        assert!(t.entries.contains_key(&id));
    }

    #[test]
    fn test_register_block_returns_incrementing_ids() {
        let mut t = tracker();
        let id0 = t.register_block("cid-0".into(), None);
        let id1 = t.register_block("cid-1".into(), None);
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
    }

    #[test]
    fn test_register_block_uses_default_desired_replicas_when_none() {
        let mut t = StorageReplicationTracker::new(5);
        let id = t.register_block("cid-x".into(), None);
        assert_eq!(t.entries[&id].desired_replicas, 5);
    }

    #[test]
    fn test_register_block_uses_provided_desired_replicas() {
        let mut t = tracker();
        let id = t.register_block("cid-x".into(), Some(7));
        assert_eq!(t.entries[&id].desired_replicas, 7);
    }

    #[test]
    fn test_register_block_starts_with_no_replicas() {
        let mut t = tracker();
        let id = t.register_block("cid-x".into(), None);
        assert!(t.entries[&id].replicas.is_empty());
    }

    #[test]
    fn test_register_block_stores_cid() {
        let mut t = tracker();
        let id = t.register_block("my-cid".into(), None);
        assert_eq!(t.entries[&id].cid, "my-cid");
    }

    // ── add_replica ───────────────────────────────────────────────────────────

    #[test]
    fn test_add_replica_appends_new_location() {
        let mut t = tracker();
        let id = t.register_block("c".into(), None);
        assert!(t.add_replica(id, "peer-1".into(), 10, false));
        assert_eq!(t.entries[&id].replicas.len(), 1);
        assert_eq!(t.entries[&id].replicas[0].peer_id, "peer-1");
    }

    #[test]
    fn test_add_replica_returns_false_for_unknown_block_id() {
        let mut t = tracker();
        assert!(!t.add_replica(999, "peer-1".into(), 1, false));
    }

    #[test]
    fn test_add_replica_updates_existing_peer_tick() {
        let mut t = tracker();
        let id = t.register_block("c".into(), None);
        t.add_replica(id, "peer-1".into(), 10, false);
        t.add_replica(id, "peer-1".into(), 99, true);
        // still only one entry
        assert_eq!(t.entries[&id].replicas.len(), 1);
        assert_eq!(t.entries[&id].replicas[0].confirmed_at_tick, 99);
        assert!(t.entries[&id].replicas[0].is_primary);
    }

    #[test]
    fn test_add_replica_multiple_peers() {
        let mut t = tracker();
        let id = t.register_block("c".into(), None);
        t.add_replica(id, "peer-1".into(), 1, true);
        t.add_replica(id, "peer-2".into(), 2, false);
        t.add_replica(id, "peer-3".into(), 3, false);
        assert_eq!(t.entries[&id].replicas.len(), 3);
    }

    // ── remove_replica ────────────────────────────────────────────────────────

    #[test]
    fn test_remove_replica_removes_entry() {
        let mut t = tracker();
        let id = t.register_block("c".into(), None);
        t.add_replica(id, "peer-1".into(), 1, false);
        assert!(t.remove_replica(id, "peer-1"));
        assert!(t.entries[&id].replicas.is_empty());
    }

    #[test]
    fn test_remove_replica_returns_false_for_unknown_block() {
        let mut t = tracker();
        assert!(!t.remove_replica(42, "peer-1"));
    }

    #[test]
    fn test_remove_replica_returns_false_for_unknown_peer() {
        let mut t = tracker();
        let id = t.register_block("c".into(), None);
        assert!(!t.remove_replica(id, "ghost-peer"));
    }

    // ── status ────────────────────────────────────────────────────────────────

    #[test]
    fn test_status_critical_when_zero_replicas() {
        let mut t = tracker();
        let id = t.register_block("c".into(), Some(3));
        assert_eq!(t.entries[&id].status(), ReplicationStatus::Critical);
    }

    #[test]
    fn test_status_under_replicated_when_below_desired() {
        let mut t = tracker();
        let id = t.register_block("c".into(), Some(3));
        t.add_replica(id, "p1".into(), 1, false);
        t.add_replica(id, "p2".into(), 2, false);
        assert_eq!(t.entries[&id].status(), ReplicationStatus::UnderReplicated);
    }

    #[test]
    fn test_status_healthy_when_equal_desired() {
        let mut t = tracker();
        let id = t.register_block("c".into(), Some(2));
        t.add_replica(id, "p1".into(), 1, true);
        t.add_replica(id, "p2".into(), 2, false);
        assert_eq!(t.entries[&id].status(), ReplicationStatus::Healthy);
    }

    #[test]
    fn test_status_over_replicated_when_above_desired() {
        let mut t = tracker();
        let id = t.register_block("c".into(), Some(1));
        t.add_replica(id, "p1".into(), 1, true);
        t.add_replica(id, "p2".into(), 2, false);
        assert_eq!(t.entries[&id].status(), ReplicationStatus::OverReplicated);
    }

    // ── deficit / surplus ─────────────────────────────────────────────────────

    #[test]
    fn test_deficit_correct_value() {
        let mut t = tracker();
        let id = t.register_block("c".into(), Some(5));
        t.add_replica(id, "p1".into(), 1, false);
        t.add_replica(id, "p2".into(), 2, false);
        assert_eq!(t.entries[&id].deficit(), 3);
    }

    #[test]
    fn test_deficit_zero_when_healthy() {
        let mut t = tracker();
        let id = t.register_block("c".into(), Some(2));
        t.add_replica(id, "p1".into(), 1, false);
        t.add_replica(id, "p2".into(), 2, false);
        assert_eq!(t.entries[&id].deficit(), 0);
    }

    #[test]
    fn test_deficit_zero_when_over_replicated() {
        let mut t = tracker();
        let id = t.register_block("c".into(), Some(1));
        t.add_replica(id, "p1".into(), 1, false);
        t.add_replica(id, "p2".into(), 2, false);
        assert_eq!(t.entries[&id].deficit(), 0);
    }

    #[test]
    fn test_surplus_correct_value() {
        let mut t = tracker();
        let id = t.register_block("c".into(), Some(1));
        t.add_replica(id, "p1".into(), 1, false);
        t.add_replica(id, "p2".into(), 2, false);
        t.add_replica(id, "p3".into(), 3, false);
        assert_eq!(t.entries[&id].surplus(), 2);
    }

    #[test]
    fn test_surplus_zero_when_healthy() {
        let mut t = tracker();
        let id = t.register_block("c".into(), Some(2));
        t.add_replica(id, "p1".into(), 1, false);
        t.add_replica(id, "p2".into(), 2, false);
        assert_eq!(t.entries[&id].surplus(), 0);
    }

    // ── under_replicated_blocks ───────────────────────────────────────────────

    #[test]
    fn test_under_replicated_blocks_sorted_by_block_id_asc() {
        let mut t = tracker();
        // block ids 0,1,2,3
        let id0 = t.register_block("c0".into(), Some(3)); // Critical (0 replicas)
        let id1 = t.register_block("c1".into(), Some(3)); // UnderReplicated (1 replica)
        let id2 = t.register_block("c2".into(), Some(2)); // Healthy (2 replicas)
        let id3 = t.register_block("c3".into(), Some(3)); // UnderReplicated (2 replicas)

        t.add_replica(id1, "p1".into(), 1, false);
        t.add_replica(id2, "p1".into(), 1, false);
        t.add_replica(id2, "p2".into(), 2, false);
        t.add_replica(id3, "p1".into(), 1, false);
        t.add_replica(id3, "p2".into(), 2, false);

        let under = t.under_replicated_blocks();
        assert_eq!(under.len(), 3); // id0 (Critical), id1 (Under), id3 (Under)
        assert_eq!(under[0].block_id, id0);
        assert_eq!(under[1].block_id, id1);
        assert_eq!(under[2].block_id, id3);
    }

    #[test]
    fn test_under_replicated_blocks_excludes_healthy_and_over_replicated() {
        let mut t = tracker();
        let id_healthy = t.register_block("h".into(), Some(1));
        let id_over = t.register_block("o".into(), Some(1));
        t.add_replica(id_healthy, "p1".into(), 1, false);
        t.add_replica(id_over, "p1".into(), 1, false);
        t.add_replica(id_over, "p2".into(), 2, false);
        assert!(t.under_replicated_blocks().is_empty());
    }

    // ── generate_tasks ────────────────────────────────────────────────────────

    #[test]
    fn test_generate_tasks_priority_critical_before_under_replicated() {
        let mut t = tracker();
        let id_under = t.register_block("under".into(), Some(3));
        let id_crit = t.register_block("crit".into(), Some(2));
        // id_under: 1 replica, deficit 2, UnderReplicated → priority 50
        t.add_replica(id_under, "p1".into(), 1, false);
        // id_crit: 0 replicas, deficit 2, Critical → priority 100

        let tasks = t.generate_tasks();
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].block_id, id_crit);
        assert_eq!(tasks[0].priority, 100);
        assert_eq!(tasks[1].block_id, id_under);
        assert_eq!(tasks[1].priority, 50);
    }

    #[test]
    fn test_generate_tasks_needed_copies_equals_deficit() {
        let mut t = tracker();
        let id = t.register_block("c".into(), Some(5));
        t.add_replica(id, "p1".into(), 1, false);
        t.add_replica(id, "p2".into(), 2, false);
        let tasks = t.generate_tasks();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].needed_copies, 3);
    }

    #[test]
    fn test_generate_tasks_sorted_by_block_id_within_same_priority() {
        let mut t = tracker();
        // Both Critical (0 replicas, desired > 0)
        let id0 = t.register_block("c0".into(), Some(2));
        let id1 = t.register_block("c1".into(), Some(3));
        let tasks = t.generate_tasks();
        assert_eq!(tasks.len(), 2);
        assert!(tasks[0].block_id <= tasks[1].block_id);
        let _ = id0;
        let _ = id1;
    }

    #[test]
    fn test_generate_tasks_excludes_healthy_blocks() {
        let mut t = tracker();
        let id = t.register_block("c".into(), Some(1));
        t.add_replica(id, "p1".into(), 1, false);
        assert!(t.generate_tasks().is_empty());
    }

    #[test]
    fn test_generate_tasks_excludes_over_replicated_blocks() {
        let mut t = tracker();
        let id = t.register_block("c".into(), Some(1));
        t.add_replica(id, "p1".into(), 1, false);
        t.add_replica(id, "p2".into(), 2, false);
        assert!(t.generate_tasks().is_empty());
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty_tracker() {
        let t = tracker();
        let s = t.stats();
        assert_eq!(s.total_blocks, 0);
        assert_eq!(s.healthy, 0);
        assert_eq!(s.critical, 0);
        assert_eq!(s.total_replicas, 0);
    }

    #[test]
    fn test_stats_counts_by_status() {
        let mut t = tracker();
        // Critical: 0 replicas, desired 2
        let id_crit = t.register_block("crit".into(), Some(2));
        // UnderReplicated: 1 of 2
        let id_under = t.register_block("under".into(), Some(2));
        t.add_replica(id_under, "p1".into(), 1, false);
        // Healthy: 2 of 2
        let id_healthy = t.register_block("healthy".into(), Some(2));
        t.add_replica(id_healthy, "p1".into(), 1, false);
        t.add_replica(id_healthy, "p2".into(), 2, false);
        // OverReplicated: 3 of 2
        let id_over = t.register_block("over".into(), Some(2));
        t.add_replica(id_over, "p1".into(), 1, false);
        t.add_replica(id_over, "p2".into(), 2, false);
        t.add_replica(id_over, "p3".into(), 3, false);
        let _ = id_crit;

        let s = t.stats();
        assert_eq!(s.total_blocks, 4);
        assert_eq!(s.critical, 1);
        assert_eq!(s.under_replicated, 1);
        assert_eq!(s.healthy, 1);
        assert_eq!(s.over_replicated, 1);
        assert_eq!(s.total_replicas, 6); // 0+1+2+3
    }

    // ── get_entry ─────────────────────────────────────────────────────────────

    #[test]
    fn test_get_entry_returns_some_for_known_block() {
        let mut t = tracker();
        let id = t.register_block("c".into(), None);
        assert!(t.get_entry(id).is_some());
    }

    #[test]
    fn test_get_entry_returns_none_for_unknown_block() {
        let t = tracker();
        assert!(t.get_entry(9999).is_none());
    }
}
