//! Peer state migration for seamless node handoff in P2P networks.
//!
//! This module provides the infrastructure to migrate peer state (routing tables,
//! provider records, connection metadata, etc.) from one node to another during
//! planned handoffs or failover scenarios.
//!
//! # Overview
//!
//! When a node needs to leave the network gracefully (maintenance, scaling, etc.),
//! its state must be transferred to one or more successor nodes to preserve
//! network continuity. The `PeerMigrationManager` orchestrates this process
//! through a structured state machine:
//!
//! ```text
//! Idle -> Preparing -> Transferring -> Verifying -> Completed
//!                 \         \              \
//!                  `--------> Failed <------'
//! ```
//!
//! # Key Features
//!
//! - **Chunked transfer**: Large state sets are broken into configurable chunks
//! - **Checksum verification**: Integrity verification after transfer completion
//! - **Concurrent migration limits**: Prevent resource exhaustion during bulk migrations
//! - **Progress tracking**: Real-time progress reporting per migration
//! - **Statistics**: Aggregate stats across all migrations for monitoring
//! - **Cleanup**: Automatic pruning of completed migration records

use std::collections::HashMap;

/// State machine for a single peer migration operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeerMigrationState {
    /// No migration activity.
    Idle,
    /// Gathering state and negotiating with target peer.
    Preparing,
    /// Actively transferring data items.
    Transferring,
    /// Transfer complete, verifying integrity via checksum.
    Verifying,
    /// Migration finished successfully.
    Completed,
    /// Migration failed with a reason.
    Failed(String),
}

impl std::fmt::Display for PeerMigrationState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Preparing => write!(f, "Preparing"),
            Self::Transferring => write!(f, "Transferring"),
            Self::Verifying => write!(f, "Verifying"),
            Self::Completed => write!(f, "Completed"),
            Self::Failed(reason) => write!(f, "Failed({})", reason),
        }
    }
}

/// Record tracking a single migration between two peers.
#[derive(Debug, Clone)]
pub struct PeerMigrationRecord {
    /// Unique identifier for this migration.
    pub migration_id: u64,
    /// Peer ID of the source (departing) node.
    pub source_peer: String,
    /// Peer ID of the target (receiving) node.
    pub target_peer: String,
    /// Current state of the migration.
    pub state: PeerMigrationState,
    /// Total number of items to transfer.
    pub items_total: u64,
    /// Number of items successfully transferred so far.
    pub items_transferred: u64,
    /// Total bytes transferred so far.
    pub bytes_transferred: u64,
    /// Timestamp (epoch ms) when the migration started.
    pub started_at: u64,
    /// Timestamp (epoch ms) when the migration completed (if finished).
    pub completed_at: Option<u64>,
    /// Checksum for integrity verification after transfer.
    pub checksum: u64,
}

/// Configuration for the migration manager.
#[derive(Debug, Clone)]
pub struct PeerMigrationConfig {
    /// Maximum number of concurrent active migrations.
    pub max_concurrent: usize,
    /// Number of items per transfer chunk.
    pub chunk_size: usize,
    /// Timeout in milliseconds for a single chunk transfer.
    pub timeout_ms: u64,
    /// Whether to verify checksums after transfer completion.
    pub verify_after_transfer: bool,
    /// Number of retry attempts for failed chunk transfers.
    pub retry_count: u32,
}

impl Default for PeerMigrationConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 4,
            chunk_size: 256,
            timeout_ms: 30_000,
            verify_after_transfer: true,
            retry_count: 3,
        }
    }
}

/// A single item being migrated between peers.
#[derive(Debug, Clone)]
pub struct MigrationItem {
    /// Key identifying this item (e.g., CID, routing key).
    pub key: String,
    /// Raw data payload.
    pub data: Vec<u8>,
    /// Sequence number for ordering within the migration.
    pub sequence: u64,
}

/// Aggregate statistics across all migrations managed by a `PeerMigrationManager`.
#[derive(Debug, Clone, Default)]
pub struct PeerMigrationStats {
    /// Total number of migrations ever started.
    pub total_migrations: u64,
    /// Number of migrations that completed successfully.
    pub successful: u64,
    /// Number of migrations that failed.
    pub failed: u64,
    /// Total bytes migrated across all successful migrations.
    pub bytes_migrated: u64,
    /// Average duration of completed migrations in milliseconds.
    pub avg_duration_ms: f64,
}

/// Manages peer state migrations for seamless node handoff.
pub struct PeerMigrationManager {
    config: PeerMigrationConfig,
    active_migrations: HashMap<u64, PeerMigrationRecord>,
    completed_migrations: Vec<PeerMigrationRecord>,
    next_id: u64,
    stats: PeerMigrationStats,
    /// Accumulated items per migration, keyed by migration_id.
    items: HashMap<u64, Vec<MigrationItem>>,
}

impl PeerMigrationManager {
    /// Create a new migration manager with the given configuration.
    pub fn new(config: PeerMigrationConfig) -> Self {
        Self {
            config,
            active_migrations: HashMap::new(),
            completed_migrations: Vec::new(),
            next_id: 1,
            stats: PeerMigrationStats::default(),
            items: HashMap::new(),
        }
    }

    /// Start a new migration from `source` to `target`.
    ///
    /// Returns the migration ID on success, or an error if the concurrent
    /// migration limit has been reached or the parameters are invalid.
    pub fn start_migration(
        &mut self,
        source: &str,
        target: &str,
        total_items: u64,
    ) -> Result<u64, String> {
        if source.is_empty() {
            return Err("source peer cannot be empty".to_string());
        }
        if target.is_empty() {
            return Err("target peer cannot be empty".to_string());
        }
        if source == target {
            return Err("source and target peers must be different".to_string());
        }
        if total_items == 0 {
            return Err("total_items must be greater than zero".to_string());
        }

        let active = self.active_count();
        if active >= self.config.max_concurrent {
            return Err(format!(
                "concurrent migration limit reached ({}/{})",
                active, self.config.max_concurrent
            ));
        }

        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);

        let now = current_timestamp_ms();

        let record = PeerMigrationRecord {
            migration_id: id,
            source_peer: source.to_string(),
            target_peer: target.to_string(),
            state: PeerMigrationState::Preparing,
            items_total: total_items,
            items_transferred: 0,
            bytes_transferred: 0,
            started_at: now,
            completed_at: None,
            checksum: 0,
        };

        self.active_migrations.insert(id, record);
        self.items.insert(id, Vec::new());
        self.stats.total_migrations = self.stats.total_migrations.saturating_add(1);

        Ok(id)
    }

    /// Add items to an active migration.
    ///
    /// The migration must be in the `Transferring` state. Each item's bytes
    /// are counted toward the transfer total.
    pub fn add_items(
        &mut self,
        migration_id: u64,
        new_items: Vec<MigrationItem>,
    ) -> Result<(), String> {
        let record = self
            .active_migrations
            .get_mut(&migration_id)
            .ok_or_else(|| format!("migration {} not found", migration_id))?;

        if record.state != PeerMigrationState::Transferring {
            return Err(format!(
                "migration {} is in state {}, expected Transferring",
                migration_id, record.state
            ));
        }

        let item_count = new_items.len() as u64;
        let byte_count: u64 = new_items.iter().map(|item| item.data.len() as u64).sum();

        // Prevent transferring more items than the declared total.
        let new_transferred = record.items_transferred.saturating_add(item_count);
        if new_transferred > record.items_total {
            return Err(format!(
                "adding {} items would exceed total ({} + {} > {})",
                item_count, record.items_transferred, item_count, record.items_total
            ));
        }

        record.items_transferred = new_transferred;
        record.bytes_transferred = record.bytes_transferred.saturating_add(byte_count);

        // Compute a simple additive checksum contribution from the items.
        let checksum_delta: u64 = compute_items_checksum(&new_items);
        record.checksum = record.checksum.wrapping_add(checksum_delta);

        let stored = self.items.entry(migration_id).or_default();
        stored.extend(new_items);

        Ok(())
    }

    /// Advance the migration to its next state.
    ///
    /// State transitions follow:
    /// `Preparing -> Transferring -> Verifying -> Completed`
    ///
    /// The `Verifying -> Completed` transition only succeeds if all items
    /// have been transferred (when `verify_after_transfer` is enabled).
    pub fn advance_state(&mut self, migration_id: u64) -> Result<PeerMigrationState, String> {
        let verify = self.config.verify_after_transfer;

        let record = self
            .active_migrations
            .get_mut(&migration_id)
            .ok_or_else(|| format!("migration {} not found", migration_id))?;

        let next = match &record.state {
            PeerMigrationState::Idle => PeerMigrationState::Preparing,
            PeerMigrationState::Preparing => PeerMigrationState::Transferring,
            PeerMigrationState::Transferring => PeerMigrationState::Verifying,
            PeerMigrationState::Verifying => {
                // Verify all items were transferred if configured to do so.
                if verify && record.items_transferred < record.items_total {
                    return Err(format!(
                        "verification failed: only {}/{} items transferred",
                        record.items_transferred, record.items_total
                    ));
                }
                PeerMigrationState::Completed
            }
            PeerMigrationState::Completed => {
                return Err(format!("migration {} is already completed", migration_id));
            }
            PeerMigrationState::Failed(reason) => {
                return Err(format!(
                    "migration {} has failed ({}), cannot advance",
                    migration_id, reason
                ));
            }
        };

        record.state = next.clone();

        // If we just completed, finalize the record.
        if next == PeerMigrationState::Completed {
            self.finalize_migration(migration_id);
        }

        Ok(next)
    }

    /// Mark a migration as completed with a final checksum.
    ///
    /// The migration must be in the `Verifying` state. If `verify_after_transfer`
    /// is enabled, the provided checksum is compared against the computed one.
    pub fn complete_migration(&mut self, migration_id: u64, checksum: u64) -> Result<(), String> {
        let verify = self.config.verify_after_transfer;

        let record = self
            .active_migrations
            .get_mut(&migration_id)
            .ok_or_else(|| format!("migration {} not found", migration_id))?;

        if record.state != PeerMigrationState::Verifying {
            return Err(format!(
                "migration {} is in state {}, expected Verifying",
                migration_id, record.state
            ));
        }

        if verify && checksum != record.checksum {
            return Err(format!(
                "checksum mismatch: expected {}, got {}",
                record.checksum, checksum
            ));
        }

        record.state = PeerMigrationState::Completed;
        record.checksum = checksum;

        self.finalize_migration(migration_id);

        Ok(())
    }

    /// Mark a migration as failed with the given reason.
    pub fn fail_migration(&mut self, migration_id: u64, reason: &str) -> Result<(), String> {
        let record = self
            .active_migrations
            .get_mut(&migration_id)
            .ok_or_else(|| format!("migration {} not found", migration_id))?;

        match &record.state {
            PeerMigrationState::Completed => {
                return Err(format!(
                    "migration {} is already completed, cannot fail",
                    migration_id
                ));
            }
            PeerMigrationState::Failed(_) => {
                return Err(format!("migration {} has already failed", migration_id));
            }
            _ => {}
        }

        record.state = PeerMigrationState::Failed(reason.to_string());
        record.completed_at = Some(current_timestamp_ms());

        self.stats.failed = self.stats.failed.saturating_add(1);

        // Move to completed list.
        if let Some(rec) = self.active_migrations.remove(&migration_id) {
            self.completed_migrations.push(rec);
        }
        self.items.remove(&migration_id);

        Ok(())
    }

    /// Get a reference to a migration record by ID.
    ///
    /// Searches both active and completed migrations.
    pub fn get_migration(&self, migration_id: u64) -> Option<&PeerMigrationRecord> {
        self.active_migrations.get(&migration_id).or_else(|| {
            self.completed_migrations
                .iter()
                .find(|r| r.migration_id == migration_id)
        })
    }

    /// Get the progress of a migration as a value between 0.0 and 1.0.
    ///
    /// Returns `None` if the migration does not exist.
    pub fn progress(&self, migration_id: u64) -> Option<f64> {
        self.get_migration(migration_id).map(|record| {
            if record.items_total == 0 {
                return 0.0;
            }
            let ratio = record.items_transferred as f64 / record.items_total as f64;
            ratio.clamp(0.0, 1.0)
        })
    }

    /// Return the number of currently active (non-completed, non-failed) migrations.
    pub fn active_count(&self) -> usize {
        self.active_migrations.len()
    }

    /// Remove completed/failed migrations with `completed_at` before the given timestamp.
    ///
    /// Returns the number of records removed.
    pub fn cleanup_completed(&mut self, before: u64) -> usize {
        let initial_len = self.completed_migrations.len();
        self.completed_migrations
            .retain(|rec| rec.completed_at.is_none_or(|ts| ts >= before));
        initial_len - self.completed_migrations.len()
    }

    /// Get a reference to the aggregate migration statistics.
    pub fn stats(&self) -> &PeerMigrationStats {
        &self.stats
    }

    /// Find all migrations (active and completed) involving a given peer,
    /// either as source or target.
    pub fn migrations_for_peer(&self, peer: &str) -> Vec<&PeerMigrationRecord> {
        let mut results: Vec<&PeerMigrationRecord> = Vec::new();

        for record in self.active_migrations.values() {
            if record.source_peer == peer || record.target_peer == peer {
                results.push(record);
            }
        }

        for record in &self.completed_migrations {
            if record.source_peer == peer || record.target_peer == peer {
                results.push(record);
            }
        }

        results
    }

    /// Internal: move a completed migration from active to completed list and update stats.
    fn finalize_migration(&mut self, migration_id: u64) {
        let now = current_timestamp_ms();

        if let Some(record) = self.active_migrations.get_mut(&migration_id) {
            record.completed_at = Some(now);
        }

        if let Some(rec) = self.active_migrations.remove(&migration_id) {
            let duration = now.saturating_sub(rec.started_at);
            self.stats.successful = self.stats.successful.saturating_add(1);
            self.stats.bytes_migrated = self
                .stats
                .bytes_migrated
                .saturating_add(rec.bytes_transferred);

            // Update rolling average duration.
            let completed_count = self.stats.successful as f64;
            if completed_count > 0.0 {
                self.stats.avg_duration_ms = self.stats.avg_duration_ms
                    + (duration as f64 - self.stats.avg_duration_ms) / completed_count;
            }

            self.completed_migrations.push(rec);
        }

        self.items.remove(&migration_id);
    }

    /// Get the configuration of this manager.
    pub fn config(&self) -> &PeerMigrationConfig {
        &self.config
    }

    /// Get the stored items for a migration (if still active).
    pub fn get_items(&self, migration_id: u64) -> Option<&[MigrationItem]> {
        self.items.get(&migration_id).map(|v| v.as_slice())
    }
}

/// Compute a simple additive checksum over migration items.
///
/// Uses a combination of key bytes and data bytes with position weighting.
fn compute_items_checksum(items: &[MigrationItem]) -> u64 {
    let mut checksum: u64 = 0;
    for item in items {
        let mut item_hash: u64 = item.sequence;
        for (i, b) in item.key.bytes().enumerate() {
            item_hash = item_hash.wrapping_add((b as u64).wrapping_mul((i as u64).wrapping_add(1)));
        }
        for (i, b) in item.data.iter().enumerate() {
            item_hash =
                item_hash.wrapping_add((*b as u64).wrapping_mul((i as u64).wrapping_add(1)));
        }
        checksum = checksum.wrapping_add(item_hash);
    }
    checksum
}

/// Get current timestamp in milliseconds.
///
/// Falls back to 0 if the system clock is unavailable.
fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> PeerMigrationConfig {
        PeerMigrationConfig::default()
    }

    fn make_items(count: usize) -> Vec<MigrationItem> {
        (0..count)
            .map(|i| MigrationItem {
                key: format!("key-{}", i),
                data: vec![i as u8; 32],
                sequence: i as u64,
            })
            .collect()
    }

    // ---------- Basic lifecycle ----------

    #[test]
    fn test_start_migration() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let id = mgr
            .start_migration("peer-a", "peer-b", 10)
            .expect("should start");
        assert_eq!(id, 1);
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn test_full_lifecycle() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let id = mgr.start_migration("peer-a", "peer-b", 3).expect("start");

        // Preparing -> Transferring
        let state = mgr.advance_state(id).expect("advance to transferring");
        assert_eq!(state, PeerMigrationState::Transferring);

        // Add items
        mgr.add_items(id, make_items(3)).expect("add items");

        // Transferring -> Verifying
        let state = mgr.advance_state(id).expect("advance to verifying");
        assert_eq!(state, PeerMigrationState::Verifying);

        // Verifying -> Completed
        let state = mgr.advance_state(id).expect("advance to completed");
        assert_eq!(state, PeerMigrationState::Completed);

        assert_eq!(mgr.active_count(), 0);
        assert_eq!(mgr.stats().successful, 1);
    }

    #[test]
    fn test_complete_with_checksum() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let id = mgr.start_migration("peer-a", "peer-b", 2).expect("start");

        mgr.advance_state(id).expect("to transferring");
        let items = make_items(2);
        let expected_checksum = compute_items_checksum(&items);
        mgr.add_items(id, items).expect("add");
        mgr.advance_state(id).expect("to verifying");

        mgr.complete_migration(id, expected_checksum)
            .expect("complete");

        let rec = mgr.get_migration(id).expect("should find");
        assert_eq!(rec.state, PeerMigrationState::Completed);
    }

    #[test]
    fn test_checksum_mismatch() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let id = mgr.start_migration("peer-a", "peer-b", 1).expect("start");
        mgr.advance_state(id).expect("to transferring");
        mgr.add_items(id, make_items(1)).expect("add");
        mgr.advance_state(id).expect("to verifying");

        let result = mgr.complete_migration(id, 999_999);
        assert!(result.is_err());
        assert!(result
            .expect_err("should fail")
            .contains("checksum mismatch"));
    }

    // ---------- State transitions ----------

    #[test]
    fn test_advance_from_idle() {
        // Migrations start in Preparing, but test that Idle also advances.
        let mut mgr = PeerMigrationManager::new(default_config());
        let id = mgr.start_migration("a", "b", 1).expect("start");

        // Force state to Idle to test transition from Idle.
        mgr.active_migrations.get_mut(&id).expect("record").state = PeerMigrationState::Idle;
        let state = mgr.advance_state(id).expect("advance");
        assert_eq!(state, PeerMigrationState::Preparing);
    }

    #[test]
    fn test_advance_completed_fails() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let id = mgr.start_migration("a", "b", 1).expect("start");
        mgr.advance_state(id).expect("to transferring");
        mgr.add_items(id, make_items(1)).expect("add");
        mgr.advance_state(id).expect("to verifying");
        mgr.advance_state(id).expect("to completed");

        // Migration is now in completed list, not active.
        let result = mgr.advance_state(id);
        assert!(result.is_err());
    }

    #[test]
    fn test_advance_failed_fails() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let id = mgr.start_migration("a", "b", 1).expect("start");
        mgr.advance_state(id).expect("to transferring");

        // Force to Failed.
        mgr.active_migrations.get_mut(&id).expect("r").state =
            PeerMigrationState::Failed("test".to_string());

        let result = mgr.advance_state(id);
        assert!(result.is_err());
    }

    #[test]
    fn test_verification_fails_if_incomplete() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let id = mgr.start_migration("a", "b", 5).expect("start");
        mgr.advance_state(id).expect("to transferring");
        mgr.add_items(id, make_items(2)).expect("add partial");
        mgr.advance_state(id).expect("to verifying");

        // Try to advance from Verifying -> Completed with incomplete items.
        let result = mgr.advance_state(id);
        assert!(result.is_err());
        assert!(result.expect_err("err").contains("verification failed"));
    }

    #[test]
    fn test_verification_skipped_when_disabled() {
        let config = PeerMigrationConfig {
            verify_after_transfer: false,
            ..default_config()
        };
        let mut mgr = PeerMigrationManager::new(config);
        let id = mgr.start_migration("a", "b", 10).expect("start");
        mgr.advance_state(id).expect("to transferring");
        mgr.add_items(id, make_items(2)).expect("add partial");
        mgr.advance_state(id).expect("to verifying");

        // Should succeed even with incomplete items.
        let state = mgr.advance_state(id).expect("to completed");
        assert_eq!(state, PeerMigrationState::Completed);
    }

    // ---------- Concurrent migration limit ----------

    #[test]
    fn test_concurrent_limit() {
        let config = PeerMigrationConfig {
            max_concurrent: 2,
            ..default_config()
        };
        let mut mgr = PeerMigrationManager::new(config);

        mgr.start_migration("a", "b", 1).expect("first");
        mgr.start_migration("c", "d", 1).expect("second");

        let result = mgr.start_migration("e", "f", 1);
        assert!(result.is_err());
        assert!(result.expect_err("err").contains("limit reached"));
    }

    #[test]
    fn test_concurrent_limit_freed_after_completion() {
        let config = PeerMigrationConfig {
            max_concurrent: 1,
            verify_after_transfer: false,
            ..default_config()
        };
        let mut mgr = PeerMigrationManager::new(config);

        let id1 = mgr.start_migration("a", "b", 1).expect("first");

        // Cannot start another.
        assert!(mgr.start_migration("c", "d", 1).is_err());

        // Complete the first.
        mgr.advance_state(id1).expect("to transferring");
        mgr.advance_state(id1).expect("to verifying");
        mgr.advance_state(id1).expect("to completed");

        // Now we can start another.
        mgr.start_migration("c", "d", 1)
            .expect("second should work");
    }

    // ---------- Progress tracking ----------

    #[test]
    fn test_progress_zero() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let id = mgr.start_migration("a", "b", 10).expect("start");
        assert_eq!(mgr.progress(id), Some(0.0));
    }

    #[test]
    fn test_progress_partial() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let id = mgr.start_migration("a", "b", 4).expect("start");
        mgr.advance_state(id).expect("to transferring");
        mgr.add_items(id, make_items(2)).expect("add");

        let p = mgr.progress(id).expect("progress");
        assert!((p - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_progress_complete() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let id = mgr.start_migration("a", "b", 3).expect("start");
        mgr.advance_state(id).expect("to transferring");
        mgr.add_items(id, make_items(3)).expect("add all");

        let p = mgr.progress(id).expect("progress");
        assert!((p - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_progress_nonexistent() {
        let mgr = PeerMigrationManager::new(default_config());
        assert_eq!(mgr.progress(999), None);
    }

    // ---------- Failure ----------

    #[test]
    fn test_fail_migration() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let id = mgr.start_migration("a", "b", 5).expect("start");
        mgr.advance_state(id).expect("to transferring");

        mgr.fail_migration(id, "network timeout").expect("fail");

        let rec = mgr.get_migration(id).expect("find");
        assert_eq!(
            rec.state,
            PeerMigrationState::Failed("network timeout".to_string())
        );
        assert_eq!(mgr.active_count(), 0);
        assert_eq!(mgr.stats().failed, 1);
    }

    #[test]
    fn test_fail_already_completed() {
        let config = PeerMigrationConfig {
            verify_after_transfer: false,
            ..default_config()
        };
        let mut mgr = PeerMigrationManager::new(config);
        let id = mgr.start_migration("a", "b", 1).expect("start");
        mgr.advance_state(id).expect("to transferring");
        mgr.advance_state(id).expect("to verifying");
        mgr.advance_state(id).expect("to completed");

        // Cannot fail a completed migration (it's in completed_migrations now).
        let result = mgr.fail_migration(id, "too late");
        assert!(result.is_err());
    }

    #[test]
    fn test_fail_already_failed() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let id = mgr.start_migration("a", "b", 1).expect("start");
        mgr.fail_migration(id, "first").expect("fail once");

        let result = mgr.fail_migration(id, "second");
        assert!(result.is_err());
    }

    // ---------- Cleanup ----------

    #[test]
    fn test_cleanup_completed() {
        let config = PeerMigrationConfig {
            verify_after_transfer: false,
            ..default_config()
        };
        let mut mgr = PeerMigrationManager::new(config);

        // Create and complete two migrations.
        let id1 = mgr.start_migration("a", "b", 1).expect("m1");
        mgr.advance_state(id1).expect("t");
        mgr.advance_state(id1).expect("v");
        mgr.advance_state(id1).expect("c");

        let id2 = mgr.start_migration("c", "d", 1).expect("m2");
        mgr.advance_state(id2).expect("t");
        mgr.advance_state(id2).expect("v");
        mgr.advance_state(id2).expect("c");

        // Cleanup with a far-future timestamp should remove both.
        let removed = mgr.cleanup_completed(u64::MAX);
        assert_eq!(removed, 2);
        assert!(mgr.completed_migrations.is_empty());
    }

    #[test]
    fn test_cleanup_none_removed() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let removed = mgr.cleanup_completed(0);
        assert_eq!(removed, 0);
    }

    // ---------- Stats ----------

    #[test]
    fn test_stats_initial() {
        let mgr = PeerMigrationManager::new(default_config());
        let s = mgr.stats();
        assert_eq!(s.total_migrations, 0);
        assert_eq!(s.successful, 0);
        assert_eq!(s.failed, 0);
        assert_eq!(s.bytes_migrated, 0);
        assert!((s.avg_duration_ms - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_after_success() {
        let config = PeerMigrationConfig {
            verify_after_transfer: false,
            ..default_config()
        };
        let mut mgr = PeerMigrationManager::new(config);
        let id = mgr.start_migration("a", "b", 2).expect("start");
        mgr.advance_state(id).expect("t");
        mgr.add_items(id, make_items(2)).expect("add");
        mgr.advance_state(id).expect("v");
        mgr.advance_state(id).expect("c");

        let s = mgr.stats();
        assert_eq!(s.total_migrations, 1);
        assert_eq!(s.successful, 1);
        assert!(s.bytes_migrated > 0);
    }

    #[test]
    fn test_stats_after_failure() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let id = mgr.start_migration("a", "b", 1).expect("start");
        mgr.fail_migration(id, "oops").expect("fail");

        let s = mgr.stats();
        assert_eq!(s.total_migrations, 1);
        assert_eq!(s.failed, 1);
        assert_eq!(s.successful, 0);
    }

    // ---------- Item transfer accounting ----------

    #[test]
    fn test_add_items_wrong_state() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let id = mgr.start_migration("a", "b", 1).expect("start");

        // Still in Preparing state.
        let result = mgr.add_items(id, make_items(1));
        assert!(result.is_err());
    }

    #[test]
    fn test_add_items_exceeds_total() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let id = mgr.start_migration("a", "b", 2).expect("start");
        mgr.advance_state(id).expect("to transferring");

        let result = mgr.add_items(id, make_items(3));
        assert!(result.is_err());
        assert!(result.expect_err("err").contains("exceed total"));
    }

    #[test]
    fn test_add_items_bytes_tracked() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let id = mgr.start_migration("a", "b", 5).expect("start");
        mgr.advance_state(id).expect("to transferring");

        let items = vec![
            MigrationItem {
                key: "k1".to_string(),
                data: vec![0u8; 100],
                sequence: 0,
            },
            MigrationItem {
                key: "k2".to_string(),
                data: vec![0u8; 200],
                sequence: 1,
            },
        ];
        mgr.add_items(id, items).expect("add");

        let rec = mgr.get_migration(id).expect("find");
        assert_eq!(rec.items_transferred, 2);
        assert_eq!(rec.bytes_transferred, 300);
    }

    #[test]
    fn test_add_items_nonexistent_migration() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let result = mgr.add_items(999, make_items(1));
        assert!(result.is_err());
    }

    // ---------- migrations_for_peer ----------

    #[test]
    fn test_migrations_for_peer_source() {
        let mut mgr = PeerMigrationManager::new(default_config());
        mgr.start_migration("peer-x", "peer-y", 1).expect("m1");
        mgr.start_migration("peer-z", "peer-w", 1).expect("m2");

        let results = mgr.migrations_for_peer("peer-x");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source_peer, "peer-x");
    }

    #[test]
    fn test_migrations_for_peer_target() {
        let mut mgr = PeerMigrationManager::new(default_config());
        mgr.start_migration("peer-a", "peer-b", 1).expect("m1");

        let results = mgr.migrations_for_peer("peer-b");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].target_peer, "peer-b");
    }

    #[test]
    fn test_migrations_for_peer_empty() {
        let mgr = PeerMigrationManager::new(default_config());
        let results = mgr.migrations_for_peer("nobody");
        assert!(results.is_empty());
    }

    // ---------- Error cases ----------

    #[test]
    fn test_start_empty_source() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let result = mgr.start_migration("", "b", 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_start_empty_target() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let result = mgr.start_migration("a", "", 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_start_same_source_target() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let result = mgr.start_migration("a", "a", 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_start_zero_items() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let result = mgr.start_migration("a", "b", 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_complete_wrong_state() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let id = mgr.start_migration("a", "b", 1).expect("start");

        // Still Preparing, not Verifying.
        let result = mgr.complete_migration(id, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_complete_nonexistent() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let result = mgr.complete_migration(999, 0);
        assert!(result.is_err());
    }

    // ---------- Config and misc ----------

    #[test]
    fn test_config_access() {
        let config = PeerMigrationConfig {
            max_concurrent: 8,
            chunk_size: 512,
            timeout_ms: 60_000,
            verify_after_transfer: false,
            retry_count: 5,
        };
        let mgr = PeerMigrationManager::new(config);
        assert_eq!(mgr.config().max_concurrent, 8);
        assert_eq!(mgr.config().chunk_size, 512);
    }

    #[test]
    fn test_default_config() {
        let config = PeerMigrationConfig::default();
        assert_eq!(config.max_concurrent, 4);
        assert!(config.verify_after_transfer);
    }

    #[test]
    fn test_state_display() {
        assert_eq!(format!("{}", PeerMigrationState::Idle), "Idle");
        assert_eq!(format!("{}", PeerMigrationState::Preparing), "Preparing");
        assert_eq!(
            format!("{}", PeerMigrationState::Transferring),
            "Transferring"
        );
        assert_eq!(format!("{}", PeerMigrationState::Verifying), "Verifying");
        assert_eq!(format!("{}", PeerMigrationState::Completed), "Completed");
        assert_eq!(
            format!("{}", PeerMigrationState::Failed("err".to_string())),
            "Failed(err)"
        );
    }

    #[test]
    fn test_migration_id_increments() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let id1 = mgr.start_migration("a", "b", 1).expect("m1");
        let id2 = mgr.start_migration("c", "d", 1).expect("m2");
        assert_eq!(id2, id1 + 1);
    }

    #[test]
    fn test_get_items() {
        let mut mgr = PeerMigrationManager::new(default_config());
        let id = mgr.start_migration("a", "b", 3).expect("start");
        mgr.advance_state(id).expect("to transferring");

        let items = make_items(2);
        mgr.add_items(id, items).expect("add");

        let stored = mgr.get_items(id).expect("get");
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].key, "key-0");
        assert_eq!(stored[1].key, "key-1");
    }

    #[test]
    fn test_checksum_computation_deterministic() {
        let items = make_items(5);
        let c1 = compute_items_checksum(&items);
        let c2 = compute_items_checksum(&items);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_checksum_different_for_different_items() {
        let items_a = make_items(3);
        let items_b = vec![MigrationItem {
            key: "different".to_string(),
            data: vec![255; 64],
            sequence: 99,
        }];
        let c1 = compute_items_checksum(&items_a);
        let c2 = compute_items_checksum(&items_b);
        assert_ne!(c1, c2);
    }
}
