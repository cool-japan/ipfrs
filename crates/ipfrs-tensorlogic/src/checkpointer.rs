//! TensorCheckpointer — periodic checkpointing of tensor computation state with rollback.
//!
//! Provides a configurable checkpoint manager that snapshots tensor values at
//! regular intervals (or on demand), maintains a bounded rolling window of
//! checkpoints, and supports rollback to any previously saved state.

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the [`TensorCheckpointer`].
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Maximum number of checkpoints to retain (FIFO eviction).
    pub max_checkpoints: usize,
    /// Number of ticks between automatic checkpoint triggers.
    pub auto_checkpoint_interval: u64,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            max_checkpoints: 5,
            auto_checkpoint_interval: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

/// A single snapshot of tensor computation state.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Unique, monotonically increasing identifier.
    pub id: u64,
    /// Tick at which this checkpoint was created.
    pub tick: u64,
    /// Human-readable label.
    pub label: String,
    /// Snapshot of tensor values — each inner `Vec<f64>` is one tensor.
    pub tensor_data: Vec<Vec<f64>>,
    /// Arbitrary key-value metadata attached to the checkpoint.
    pub metadata: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Aggregate statistics about the checkpointer state.
#[derive(Debug, Clone)]
pub struct CheckpointerStats {
    /// Number of checkpoints currently stored.
    pub total_checkpoints: usize,
    /// Tick of the oldest checkpoint, if any.
    pub oldest_tick: Option<u64>,
    /// Tick of the newest checkpoint, if any.
    pub newest_tick: Option<u64>,
    /// Total number of rollbacks performed since creation.
    pub rollbacks_performed: u64,
}

// ---------------------------------------------------------------------------
// TensorCheckpointer
// ---------------------------------------------------------------------------

/// Manages a bounded, rolling window of tensor computation checkpoints with
/// rollback support.
pub struct TensorCheckpointer {
    config: CheckpointConfig,
    checkpoints: VecDeque<Checkpoint>,
    next_id: u64,
    current_tick: u64,
    last_checkpoint_tick: u64,
    rollbacks_performed: u64,
}

impl TensorCheckpointer {
    /// Create a new checkpointer with the given configuration.
    pub fn new(config: CheckpointConfig) -> Self {
        Self {
            config,
            checkpoints: VecDeque::new(),
            next_id: 0,
            current_tick: 0,
            last_checkpoint_tick: 0,
            rollbacks_performed: 0,
        }
    }

    /// Create a checkpoint with the given label, tensor data, and metadata.
    ///
    /// If the number of stored checkpoints exceeds `max_checkpoints`, the
    /// oldest checkpoint is evicted (FIFO).
    ///
    /// Returns the unique id of the newly created checkpoint.
    pub fn create_checkpoint(
        &mut self,
        label: &str,
        tensor_data: Vec<Vec<f64>>,
        metadata: HashMap<String, String>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let checkpoint = Checkpoint {
            id,
            tick: self.current_tick,
            label: label.to_string(),
            tensor_data,
            metadata,
        };

        self.checkpoints.push_back(checkpoint);
        self.last_checkpoint_tick = self.current_tick;

        // Evict oldest if over capacity.
        while self.checkpoints.len() > self.config.max_checkpoints {
            self.checkpoints.pop_front();
        }

        id
    }

    /// Roll back to the checkpoint with the given `checkpoint_id`.
    ///
    /// Returns the checkpoint data and removes all checkpoints that were
    /// created *after* the target checkpoint.  The target checkpoint itself is
    /// also consumed (removed) so callers can restore from the returned data.
    ///
    /// Increments `rollbacks_performed`.
    pub fn rollback(&mut self, checkpoint_id: u64) -> Result<Checkpoint, String> {
        let pos = self
            .checkpoints
            .iter()
            .position(|c| c.id == checkpoint_id)
            .ok_or_else(|| format!("checkpoint id {} not found", checkpoint_id))?;

        // Remove everything after the target checkpoint.
        self.checkpoints.truncate(pos + 1);

        // Pop the target checkpoint itself so we can return it by value.
        let checkpoint = self
            .checkpoints
            .pop_back()
            .ok_or_else(|| "internal error: checkpoint disappeared".to_string())?;

        self.rollbacks_performed += 1;

        Ok(checkpoint)
    }

    /// Return a reference to the most recent checkpoint, if any.
    pub fn latest_checkpoint(&self) -> Option<&Checkpoint> {
        self.checkpoints.back()
    }

    /// Look up a checkpoint by its id.
    pub fn get_checkpoint(&self, id: u64) -> Option<&Checkpoint> {
        self.checkpoints.iter().find(|c| c.id == id)
    }

    /// Number of checkpoints currently stored.
    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    /// Returns `true` if the number of ticks since the last checkpoint is
    /// greater than or equal to `auto_checkpoint_interval`.
    pub fn should_auto_checkpoint(&self) -> bool {
        self.current_tick.saturating_sub(self.last_checkpoint_tick)
            >= self.config.auto_checkpoint_interval
    }

    /// Advance the internal tick clock by one.
    pub fn tick(&mut self) {
        self.current_tick += 1;
    }

    /// List all checkpoints as `(id, label, tick)` triples, ordered oldest to
    /// newest.
    pub fn list_checkpoints(&self) -> Vec<(u64, String, u64)> {
        self.checkpoints
            .iter()
            .map(|c| (c.id, c.label.clone(), c.tick))
            .collect()
    }

    /// Remove all stored checkpoints.
    pub fn clear_all(&mut self) {
        self.checkpoints.clear();
    }

    /// Compute aggregate statistics.
    pub fn stats(&self) -> CheckpointerStats {
        CheckpointerStats {
            total_checkpoints: self.checkpoints.len(),
            oldest_tick: self.checkpoints.front().map(|c| c.tick),
            newest_tick: self.checkpoints.back().map(|c| c.tick),
            rollbacks_performed: self.rollbacks_performed,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> CheckpointConfig {
        CheckpointConfig::default()
    }

    fn config_with_max(max: usize) -> CheckpointConfig {
        CheckpointConfig {
            max_checkpoints: max,
            ..Default::default()
        }
    }

    fn empty_meta() -> HashMap<String, String> {
        HashMap::new()
    }

    fn sample_tensors() -> Vec<Vec<f64>> {
        vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0]]
    }

    // -- basic creation -----------------------------------------------------

    #[test]
    fn test_create_checkpoint_returns_unique_ids() {
        let mut cp = TensorCheckpointer::new(default_config());
        let id0 = cp.create_checkpoint("a", vec![], empty_meta());
        let id1 = cp.create_checkpoint("b", vec![], empty_meta());
        assert_ne!(id0, id1);
        assert_eq!(cp.checkpoint_count(), 2);
    }

    #[test]
    fn test_create_checkpoint_stores_label() {
        let mut cp = TensorCheckpointer::new(default_config());
        let id = cp.create_checkpoint("my_label", vec![], empty_meta());
        let c = cp.get_checkpoint(id);
        assert!(c.is_some());
        assert_eq!(c.map(|c| c.label.as_str()), Some("my_label"));
    }

    #[test]
    fn test_create_checkpoint_stores_tensor_data() {
        let mut cp = TensorCheckpointer::new(default_config());
        let data = sample_tensors();
        let id = cp.create_checkpoint("t", data.clone(), empty_meta());
        let c = cp.get_checkpoint(id).expect("checkpoint should exist");
        assert_eq!(c.tensor_data, data);
    }

    #[test]
    fn test_create_checkpoint_records_tick() {
        let mut cp = TensorCheckpointer::new(default_config());
        cp.tick();
        cp.tick();
        let id = cp.create_checkpoint("at_tick_2", vec![], empty_meta());
        let c = cp.get_checkpoint(id).expect("checkpoint should exist");
        assert_eq!(c.tick, 2);
    }

    // -- eviction -----------------------------------------------------------

    #[test]
    fn test_max_checkpoints_eviction_fifo() {
        let mut cp = TensorCheckpointer::new(config_with_max(3));
        let id0 = cp.create_checkpoint("0", vec![], empty_meta());
        let _id1 = cp.create_checkpoint("1", vec![], empty_meta());
        let _id2 = cp.create_checkpoint("2", vec![], empty_meta());
        assert_eq!(cp.checkpoint_count(), 3);

        // Adding a 4th should evict the oldest (id0).
        let _id3 = cp.create_checkpoint("3", vec![], empty_meta());
        assert_eq!(cp.checkpoint_count(), 3);
        assert!(cp.get_checkpoint(id0).is_none());
    }

    #[test]
    fn test_eviction_preserves_newest() {
        let mut cp = TensorCheckpointer::new(config_with_max(2));
        let _id0 = cp.create_checkpoint("a", vec![], empty_meta());
        let _id1 = cp.create_checkpoint("b", vec![], empty_meta());
        let id2 = cp.create_checkpoint("c", vec![], empty_meta());

        let latest = cp.latest_checkpoint().expect("should have latest");
        assert_eq!(latest.id, id2);
    }

    #[test]
    fn test_eviction_with_max_one() {
        let mut cp = TensorCheckpointer::new(config_with_max(1));
        let _id0 = cp.create_checkpoint("first", vec![], empty_meta());
        let id1 = cp.create_checkpoint("second", vec![], empty_meta());
        assert_eq!(cp.checkpoint_count(), 1);
        assert!(cp.get_checkpoint(id1).is_some());
    }

    // -- rollback -----------------------------------------------------------

    #[test]
    fn test_rollback_returns_checkpoint_data() {
        let mut cp = TensorCheckpointer::new(default_config());
        let data = sample_tensors();
        let id = cp.create_checkpoint("snap", data.clone(), empty_meta());
        let restored = cp.rollback(id).expect("rollback should succeed");
        assert_eq!(restored.tensor_data, data);
        assert_eq!(restored.label, "snap");
    }

    #[test]
    fn test_rollback_removes_newer_checkpoints() {
        let mut cp = TensorCheckpointer::new(default_config());
        let id0 = cp.create_checkpoint("0", vec![], empty_meta());
        let id1 = cp.create_checkpoint("1", vec![], empty_meta());
        let id2 = cp.create_checkpoint("2", vec![], empty_meta());

        let _restored = cp.rollback(id1).expect("rollback should succeed");
        // id1 itself is consumed, id2 removed; only id0 remains.
        assert_eq!(cp.checkpoint_count(), 1);
        assert!(cp.get_checkpoint(id0).is_some());
        assert!(cp.get_checkpoint(id1).is_none());
        assert!(cp.get_checkpoint(id2).is_none());
    }

    #[test]
    fn test_rollback_unknown_id_errors() {
        let mut cp = TensorCheckpointer::new(default_config());
        let result = cp.rollback(999);
        assert!(result.is_err());
        let msg = result.expect_err("should be error");
        assert!(msg.contains("999"));
    }

    #[test]
    fn test_rollback_increments_counter() {
        let mut cp = TensorCheckpointer::new(default_config());
        let id = cp.create_checkpoint("x", vec![], empty_meta());
        assert_eq!(cp.stats().rollbacks_performed, 0);
        let _ = cp.rollback(id);
        assert_eq!(cp.stats().rollbacks_performed, 1);
    }

    #[test]
    fn test_rollback_to_oldest() {
        let mut cp = TensorCheckpointer::new(default_config());
        let id0 = cp.create_checkpoint("oldest", vec![vec![0.0]], empty_meta());
        let _id1 = cp.create_checkpoint("mid", vec![], empty_meta());
        let _id2 = cp.create_checkpoint("newest", vec![], empty_meta());

        let restored = cp.rollback(id0).expect("rollback should succeed");
        assert_eq!(restored.label, "oldest");
        assert_eq!(cp.checkpoint_count(), 0);
    }

    #[test]
    fn test_rollback_to_latest() {
        let mut cp = TensorCheckpointer::new(default_config());
        let _id0 = cp.create_checkpoint("a", vec![], empty_meta());
        let id1 = cp.create_checkpoint("b", vec![], empty_meta());

        let restored = cp.rollback(id1).expect("rollback should succeed");
        assert_eq!(restored.label, "b");
        assert_eq!(cp.checkpoint_count(), 1); // id0 remains
    }

    #[test]
    fn test_multiple_rollbacks() {
        let mut cp = TensorCheckpointer::new(default_config());
        let _id0 = cp.create_checkpoint("a", vec![], empty_meta());
        let id1 = cp.create_checkpoint("b", vec![], empty_meta());
        let _ = cp.rollback(id1);
        assert_eq!(cp.stats().rollbacks_performed, 1);

        let id2 = cp.create_checkpoint("c", vec![], empty_meta());
        let _ = cp.rollback(id2);
        assert_eq!(cp.stats().rollbacks_performed, 2);
    }

    // -- latest / get -------------------------------------------------------

    #[test]
    fn test_latest_checkpoint_empty() {
        let cp = TensorCheckpointer::new(default_config());
        assert!(cp.latest_checkpoint().is_none());
    }

    #[test]
    fn test_latest_checkpoint_returns_last_added() {
        let mut cp = TensorCheckpointer::new(default_config());
        let _id0 = cp.create_checkpoint("first", vec![], empty_meta());
        let id1 = cp.create_checkpoint("second", vec![], empty_meta());
        let latest = cp.latest_checkpoint().expect("should exist");
        assert_eq!(latest.id, id1);
    }

    #[test]
    fn test_get_checkpoint_nonexistent() {
        let cp = TensorCheckpointer::new(default_config());
        assert!(cp.get_checkpoint(42).is_none());
    }

    // -- auto checkpoint timing ---------------------------------------------

    #[test]
    fn test_should_auto_checkpoint_initially_true_when_interval_zero() {
        let config = CheckpointConfig {
            auto_checkpoint_interval: 0,
            ..Default::default()
        };
        let cp = TensorCheckpointer::new(config);
        assert!(cp.should_auto_checkpoint());
    }

    #[test]
    fn test_should_auto_checkpoint_false_initially() {
        let cp = TensorCheckpointer::new(default_config()); // interval = 100
        assert!(!cp.should_auto_checkpoint());
    }

    #[test]
    fn test_should_auto_checkpoint_after_enough_ticks() {
        let config = CheckpointConfig {
            auto_checkpoint_interval: 5,
            ..Default::default()
        };
        let mut cp = TensorCheckpointer::new(config);
        for _ in 0..4 {
            cp.tick();
            assert!(!cp.should_auto_checkpoint());
        }
        cp.tick(); // tick 5
        assert!(cp.should_auto_checkpoint());
    }

    #[test]
    fn test_auto_checkpoint_resets_after_create() {
        let config = CheckpointConfig {
            auto_checkpoint_interval: 3,
            ..Default::default()
        };
        let mut cp = TensorCheckpointer::new(config);
        for _ in 0..3 {
            cp.tick();
        }
        assert!(cp.should_auto_checkpoint());
        cp.create_checkpoint("auto", vec![], empty_meta());
        assert!(!cp.should_auto_checkpoint());
    }

    // -- list checkpoints ---------------------------------------------------

    #[test]
    fn test_list_checkpoints_empty() {
        let cp = TensorCheckpointer::new(default_config());
        assert!(cp.list_checkpoints().is_empty());
    }

    #[test]
    fn test_list_checkpoints_order() {
        let mut cp = TensorCheckpointer::new(default_config());
        cp.create_checkpoint("a", vec![], empty_meta());
        cp.tick();
        cp.create_checkpoint("b", vec![], empty_meta());
        let list = cp.list_checkpoints();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].1, "a");
        assert_eq!(list[1].1, "b");
        assert!(list[0].2 < list[1].2);
    }

    #[test]
    fn test_list_checkpoints_after_eviction() {
        let mut cp = TensorCheckpointer::new(config_with_max(2));
        cp.create_checkpoint("a", vec![], empty_meta());
        cp.create_checkpoint("b", vec![], empty_meta());
        cp.create_checkpoint("c", vec![], empty_meta());
        let list = cp.list_checkpoints();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].1, "b");
        assert_eq!(list[1].1, "c");
    }

    // -- clear_all ----------------------------------------------------------

    #[test]
    fn test_clear_all() {
        let mut cp = TensorCheckpointer::new(default_config());
        cp.create_checkpoint("a", vec![], empty_meta());
        cp.create_checkpoint("b", vec![], empty_meta());
        assert_eq!(cp.checkpoint_count(), 2);
        cp.clear_all();
        assert_eq!(cp.checkpoint_count(), 0);
        assert!(cp.latest_checkpoint().is_none());
    }

    #[test]
    fn test_clear_all_on_empty() {
        let mut cp = TensorCheckpointer::new(default_config());
        cp.clear_all(); // should not panic
        assert_eq!(cp.checkpoint_count(), 0);
    }

    // -- stats --------------------------------------------------------------

    #[test]
    fn test_stats_empty() {
        let cp = TensorCheckpointer::new(default_config());
        let s = cp.stats();
        assert_eq!(s.total_checkpoints, 0);
        assert!(s.oldest_tick.is_none());
        assert!(s.newest_tick.is_none());
        assert_eq!(s.rollbacks_performed, 0);
    }

    #[test]
    fn test_stats_with_checkpoints() {
        let mut cp = TensorCheckpointer::new(default_config());
        cp.create_checkpoint("a", vec![], empty_meta());
        cp.tick();
        cp.tick();
        cp.create_checkpoint("b", vec![], empty_meta());
        let s = cp.stats();
        assert_eq!(s.total_checkpoints, 2);
        assert_eq!(s.oldest_tick, Some(0));
        assert_eq!(s.newest_tick, Some(2));
    }

    #[test]
    fn test_stats_after_rollback() {
        let mut cp = TensorCheckpointer::new(default_config());
        let id0 = cp.create_checkpoint("a", vec![], empty_meta());
        cp.create_checkpoint("b", vec![], empty_meta());
        let _ = cp.rollback(id0);
        let s = cp.stats();
        assert_eq!(s.total_checkpoints, 0);
        assert_eq!(s.rollbacks_performed, 1);
    }

    // -- metadata -----------------------------------------------------------

    #[test]
    fn test_metadata_preserved() {
        let mut cp = TensorCheckpointer::new(default_config());
        let mut meta = HashMap::new();
        meta.insert("epoch".to_string(), "42".to_string());
        meta.insert("loss".to_string(), "0.01".to_string());
        let id = cp.create_checkpoint("meta_test", vec![], meta.clone());
        let c = cp.get_checkpoint(id).expect("should exist");
        assert_eq!(c.metadata, meta);
    }

    #[test]
    fn test_metadata_preserved_after_rollback() {
        let mut cp = TensorCheckpointer::new(default_config());
        let mut meta = HashMap::new();
        meta.insert("key".to_string(), "value".to_string());
        let id = cp.create_checkpoint("m", vec![], meta.clone());
        let restored = cp.rollback(id).expect("rollback should succeed");
        assert_eq!(restored.metadata, meta);
    }

    // -- edge cases ---------------------------------------------------------

    #[test]
    fn test_empty_checkpointer_count() {
        let cp = TensorCheckpointer::new(default_config());
        assert_eq!(cp.checkpoint_count(), 0);
    }

    #[test]
    fn test_tick_without_checkpoints() {
        let mut cp = TensorCheckpointer::new(default_config());
        for _ in 0..200 {
            cp.tick();
        }
        // Should not panic, and auto-checkpoint should trigger.
        assert!(cp.should_auto_checkpoint());
    }

    #[test]
    fn test_create_after_clear() {
        let mut cp = TensorCheckpointer::new(default_config());
        cp.create_checkpoint("before", vec![], empty_meta());
        cp.clear_all();
        let id = cp.create_checkpoint("after", vec![], empty_meta());
        assert_eq!(cp.checkpoint_count(), 1);
        let c = cp.get_checkpoint(id).expect("should exist");
        assert_eq!(c.label, "after");
    }

    #[test]
    fn test_rollback_then_create() {
        let mut cp = TensorCheckpointer::new(default_config());
        let id0 = cp.create_checkpoint("a", vec![], empty_meta());
        cp.create_checkpoint("b", vec![], empty_meta());
        let _ = cp.rollback(id0);

        // After rollback, we can still create new checkpoints.
        let id_new = cp.create_checkpoint("c", vec![], empty_meta());
        assert_eq!(cp.checkpoint_count(), 1);
        let c = cp.get_checkpoint(id_new).expect("should exist");
        assert_eq!(c.label, "c");
    }

    #[test]
    fn test_large_tensor_data() {
        let mut cp = TensorCheckpointer::new(default_config());
        let big = vec![vec![0.5; 1000]; 10];
        let id = cp.create_checkpoint("large", big.clone(), empty_meta());
        let c = cp.get_checkpoint(id).expect("should exist");
        assert_eq!(c.tensor_data.len(), 10);
        assert_eq!(c.tensor_data[0].len(), 1000);
        assert!((c.tensor_data[0][500] - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_checkpoint_ids_are_monotonic() {
        let mut cp = TensorCheckpointer::new(default_config());
        let mut prev = cp.create_checkpoint("0", vec![], empty_meta());
        for i in 1..10 {
            let id = cp.create_checkpoint(&i.to_string(), vec![], empty_meta());
            assert!(id > prev);
            prev = id;
        }
    }

    #[test]
    fn test_default_config_values() {
        let cfg = CheckpointConfig::default();
        assert_eq!(cfg.max_checkpoints, 5);
        assert_eq!(cfg.auto_checkpoint_interval, 100);
    }
}
