//! Distributed gradient optimizer — coordinates gradient aggregation across
//! multiple distributed training workers with staleness handling,
//! compression-friendly interfaces, and fault-tolerant worker management.
//!
//! # Overview
//!
//! [`DistributedOptimizer`] is the central coordinator. Workers register,
//! submit gradient updates for named layers, and the optimizer aggregates
//! them according to the chosen [`AggregationStrategy`].
//!
//! Four aggregation strategies are supported:
//!
//! - **Synchronous** — classical barrier-based: every active worker must
//!   submit before aggregation proceeds.
//! - **Asynchronous** — accepts updates within a configurable staleness
//!   window; stale updates are rejected at submission time.
//! - **FederatedAverage** — aggregates as soon as at least `rounds` updates
//!   have been collected for a layer.
//! - **GossipAverage** — aggregates as soon as at least `fanout` updates are
//!   available, mimicking gossip protocol averaging.

use std::collections::HashMap;
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Opaque identifier for a training worker.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkerId(pub String);

impl WorkerId {
    /// Create a new [`WorkerId`] from any string-like value.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl fmt::Display for WorkerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gradient update
// ─────────────────────────────────────────────────────────────────────────────

/// A single gradient update submitted by one worker for one named layer.
#[derive(Debug, Clone)]
pub struct GradientUpdate {
    /// The worker that produced this update.
    pub worker_id: WorkerId,
    /// The layer these gradients belong to.
    pub layer_id: String,
    /// Gradient values (one per parameter in the layer).
    pub gradients: Vec<f64>,
    /// The training step at which these gradients were computed.
    pub step: u64,
    /// Wall-clock timestamp (milliseconds since epoch, or any monotonic
    /// counter) at which the update was created.
    pub timestamp: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Aggregation strategies
// ─────────────────────────────────────────────────────────────────────────────

/// Determines how and when pending updates are aggregated.
#[derive(Debug, Clone)]
pub enum AggregationStrategy {
    /// Classic barrier: all active workers must submit at the current step
    /// before any aggregation can proceed.
    Synchronous,

    /// Asynchronous aggregation: updates within `staleness_threshold` steps
    /// of the current step are accepted; older updates are rejected.
    Asynchronous {
        /// Maximum number of steps behind the global step that a worker may be.
        staleness_threshold: u64,
    },

    /// Aggregate every time at least `rounds` worker updates have accumulated
    /// for a layer (federated learning style).
    FederatedAverage {
        /// Minimum number of updates needed before aggregation.
        rounds: u32,
    },

    /// Aggregate when at least `fanout` updates are available (gossip style).
    GossipAverage {
        /// Minimum number of updates needed before aggregation.
        fanout: usize,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Output types
// ─────────────────────────────────────────────────────────────────────────────

/// The result of aggregating gradient updates for a single layer.
#[derive(Debug, Clone)]
pub struct AggregatedGradient {
    /// The layer these gradients belong to.
    pub layer_id: String,
    /// Element-wise mean of all contributing worker gradients.
    pub values: Vec<f64>,
    /// How many workers contributed to this aggregation.
    pub contributing_workers: usize,
    /// The global step at which this aggregation was produced.
    pub step: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Worker state
// ─────────────────────────────────────────────────────────────────────────────

/// Liveness and bookkeeping state for a single registered worker.
#[derive(Debug, Clone)]
pub struct WorkerState {
    /// The worker's identifier.
    pub worker_id: WorkerId,
    /// The most recent training step for which this worker has submitted an
    /// update.
    pub latest_step: u64,
    /// Wall-clock timestamp of the most recent update from this worker.
    pub last_seen: u64,
    /// Whether the worker is currently considered active.
    pub active: bool,
    /// Cumulative number of updates submitted since registration.
    pub total_updates: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise from [`DistributedOptimizer`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizerDistError {
    /// The worker is not registered with this optimizer.
    WorkerNotFound(String),
    /// The submitted update is too far behind the current global step.
    StaleUpdate {
        worker: String,
        step: u64,
        current: u64,
    },
    /// The gradient dimensions do not match the expected layer width.
    DimensionMismatch {
        layer: String,
        expected: usize,
        got: usize,
    },
}

impl fmt::Display for OptimizerDistError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkerNotFound(id) => write!(f, "worker not found: {id}"),
            Self::StaleUpdate {
                worker,
                step,
                current,
            } => {
                write!(
                    f,
                    "stale update from worker {worker}: update step {step}, current step {current}"
                )
            }
            Self::DimensionMismatch {
                layer,
                expected,
                got,
            } => {
                write!(
                    f,
                    "dimension mismatch for layer {layer}: expected {expected}, got {got}"
                )
            }
        }
    }
}

impl std::error::Error for OptimizerDistError {}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics snapshot
// ─────────────────────────────────────────────────────────────────────────────

/// A point-in-time snapshot of optimizer statistics.
#[derive(Debug, Clone)]
pub struct DistOptimizerStats {
    pub total_workers: usize,
    pub active_workers: usize,
    pub current_step: u64,
    pub total_aggregations: u64,
    pub dropped_updates: u64,
    pub pending_layers: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Core optimizer struct
// ─────────────────────────────────────────────────────────────────────────────

/// Coordinator for distributed gradient aggregation across multiple workers.
pub struct DistributedOptimizer {
    /// Aggregation strategy that governs when and how updates are combined.
    pub strategy: AggregationStrategy,
    /// Registered workers indexed by their identifier.
    workers: HashMap<WorkerId, WorkerState>,
    /// Pending gradient updates indexed by layer ID.
    pending: HashMap<String, Vec<GradientUpdate>>,
    /// Most recently aggregated gradients indexed by layer ID.
    aggregated: HashMap<String, AggregatedGradient>,
    /// The global training step counter.
    pub current_step: u64,
    /// Number of updates that were rejected (staleness or other).
    pub dropped_updates: u64,
    /// Cumulative number of successful aggregations.
    total_aggregations: u64,
}

impl DistributedOptimizer {
    // ─────────────────────────────────────────────────────────────
    // Construction
    // ─────────────────────────────────────────────────────────────

    /// Create a new optimizer with the given aggregation strategy.
    pub fn new(strategy: AggregationStrategy) -> Self {
        Self {
            strategy,
            workers: HashMap::new(),
            pending: HashMap::new(),
            aggregated: HashMap::new(),
            current_step: 0,
            dropped_updates: 0,
            total_aggregations: 0,
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Worker lifecycle
    // ─────────────────────────────────────────────────────────────

    /// Register a worker.
    ///
    /// Returns `true` if the worker was newly registered, `false` if it was
    /// already present (even if inactive).
    pub fn register_worker(&mut self, worker_id: WorkerId) -> bool {
        if self.workers.contains_key(&worker_id) {
            return false;
        }
        self.workers.insert(
            worker_id.clone(),
            WorkerState {
                worker_id,
                latest_step: 0,
                last_seen: 0,
                active: true,
                total_updates: 0,
            },
        );
        true
    }

    /// Mark a worker as inactive (soft-deregister).
    ///
    /// Returns `true` if the worker existed and was marked inactive,
    /// `false` if it was not found.
    pub fn deregister_worker(&mut self, worker_id: &WorkerId) -> bool {
        match self.workers.get_mut(worker_id) {
            Some(state) => {
                state.active = false;
                true
            }
            None => false,
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Update submission
    // ─────────────────────────────────────────────────────────────

    /// Submit a gradient update from a worker.
    ///
    /// # Errors
    ///
    /// - [`OptimizerDistError::WorkerNotFound`] if the worker is not registered.
    /// - [`OptimizerDistError::StaleUpdate`] for async strategy when the
    ///   update is too far behind the global step.
    /// - [`OptimizerDistError::DimensionMismatch`] when the gradient dimension
    ///   conflicts with an already-pending update for the same layer.
    pub fn submit_update(
        &mut self,
        update: GradientUpdate,
        now: u64,
    ) -> Result<(), OptimizerDistError> {
        // Validate worker is registered (active or not).
        if !self.workers.contains_key(&update.worker_id) {
            return Err(OptimizerDistError::WorkerNotFound(
                update.worker_id.0.clone(),
            ));
        }

        // Staleness check for async strategy.
        if let AggregationStrategy::Asynchronous {
            staleness_threshold,
        } = self.strategy
        {
            let staleness = self.current_step.saturating_sub(update.step);
            if staleness > staleness_threshold {
                self.dropped_updates += 1;
                return Err(OptimizerDistError::StaleUpdate {
                    worker: update.worker_id.0.clone(),
                    step: update.step,
                    current: self.current_step,
                });
            }
        }

        // Dimension consistency check: verify against any existing pending
        // gradients for this layer.
        let grad_len = update.gradients.len();
        if let Some(existing) = self.pending.get(&update.layer_id) {
            if let Some(first) = existing.first() {
                let expected = first.gradients.len();
                if grad_len != expected {
                    return Err(OptimizerDistError::DimensionMismatch {
                        layer: update.layer_id.clone(),
                        expected,
                        got: grad_len,
                    });
                }
            }
        }

        // Update worker state.
        let worker_id = update.worker_id.clone();
        if let Some(state) = self.workers.get_mut(&worker_id) {
            if update.step > state.latest_step {
                state.latest_step = update.step;
            }
            state.last_seen = now;
            state.total_updates += 1;
        }

        // Append to pending.
        self.pending
            .entry(update.layer_id.clone())
            .or_default()
            .push(update);

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────
    // Aggregation
    // ─────────────────────────────────────────────────────────────

    /// Attempt to aggregate pending updates for `layer_id`.
    ///
    /// Returns `Some(AggregatedGradient)` if the strategy's quorum condition
    /// is satisfied, `None` otherwise.  On success the pending queue for that
    /// layer is cleared and the result is stored in `aggregated`.
    pub fn try_aggregate(&mut self, layer_id: &str) -> Option<AggregatedGradient> {
        let pending = self.pending.get(layer_id)?;
        if pending.is_empty() {
            return None;
        }

        let can_aggregate = match &self.strategy {
            AggregationStrategy::Synchronous => {
                // All active workers must have submitted for this layer at
                // the current step.
                let active_count = self.workers.values().filter(|w| w.active).count();

                if active_count == 0 {
                    return None;
                }

                let submitted_for_step: std::collections::HashSet<&WorkerId> = pending
                    .iter()
                    .filter(|u| u.step == self.current_step)
                    .map(|u| &u.worker_id)
                    .collect();

                let active_have_submitted = self
                    .workers
                    .values()
                    .filter(|w| w.active)
                    .all(|w| submitted_for_step.contains(&w.worker_id));

                active_have_submitted
            }
            AggregationStrategy::Asynchronous { .. } => {
                // Aggregate everything that has been accepted (staleness was
                // already checked at submission time).
                !pending.is_empty()
            }
            AggregationStrategy::FederatedAverage { rounds } => pending.len() >= *rounds as usize,
            AggregationStrategy::GossipAverage { fanout } => pending.len() >= *fanout,
        };

        if !can_aggregate {
            return None;
        }

        // Perform element-wise mean.
        let updates: Vec<GradientUpdate> = self.pending.remove(layer_id).unwrap_or_default();

        let contributing_workers = updates.len();
        let values = Self::aggregate_gradients(&updates);
        let result = AggregatedGradient {
            layer_id: layer_id.to_string(),
            values,
            contributing_workers,
            step: self.current_step,
        };

        self.aggregated.insert(layer_id.to_string(), result.clone());
        self.total_aggregations += 1;
        Some(result)
    }

    /// Compute the element-wise mean of a slice of [`GradientUpdate`]s.
    ///
    /// Returns an empty vector if `updates` is empty or all gradient vectors
    /// are empty.
    pub fn aggregate_gradients(updates: &[GradientUpdate]) -> Vec<f64> {
        if updates.is_empty() {
            return Vec::new();
        }

        let len = updates.iter().map(|u| u.gradients.len()).max().unwrap_or(0);

        if len == 0 {
            return Vec::new();
        }

        let n = updates.len() as f64;
        let mut sums = vec![0.0_f64; len];
        for update in updates {
            for (i, &g) in update.gradients.iter().enumerate() {
                if i < len {
                    sums[i] += g;
                }
            }
        }
        sums.iter_mut().for_each(|s| *s /= n);
        sums
    }

    // ─────────────────────────────────────────────────────────────
    // Step management
    // ─────────────────────────────────────────────────────────────

    /// Advance the global training step by one.
    ///
    /// For the [`AggregationStrategy::Synchronous`] strategy this also
    /// clears all pending updates left over from the previous step (workers
    /// that did not submit in time are implicitly skipped).
    pub fn advance_step(&mut self) {
        if matches!(self.strategy, AggregationStrategy::Synchronous) {
            self.pending.clear();
        }
        self.current_step += 1;
    }

    // ─────────────────────────────────────────────────────────────
    // Queries
    // ─────────────────────────────────────────────────────────────

    /// Returns references to all active workers.
    pub fn active_workers(&self) -> Vec<&WorkerState> {
        self.workers.values().filter(|w| w.active).collect()
    }

    /// Total number of registered workers (active and inactive).
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Number of currently active workers.
    pub fn active_worker_count(&self) -> usize {
        self.workers.values().filter(|w| w.active).count()
    }

    /// Number of pending (not-yet-aggregated) updates for a layer.
    pub fn pending_updates_for(&self, layer_id: &str) -> usize {
        self.pending.get(layer_id).map_or(0, |v| v.len())
    }

    /// Returns the most recently aggregated gradient for a layer, if any.
    pub fn last_aggregated(&self, layer_id: &str) -> Option<&AggregatedGradient> {
        self.aggregated.get(layer_id)
    }

    // ─────────────────────────────────────────────────────────────
    // Fault tolerance
    // ─────────────────────────────────────────────────────────────

    /// Mark workers inactive if their `last_seen` timestamp is older than
    /// `now - max_age_ms`.
    ///
    /// Returns the number of workers evicted.
    pub fn evict_stale_workers(&mut self, max_age_ms: u64, now: u64) -> usize {
        let cutoff = now.saturating_sub(max_age_ms);
        let mut count = 0usize;
        for state in self.workers.values_mut() {
            if state.active && state.last_seen < cutoff {
                state.active = false;
                count += 1;
            }
        }
        count
    }

    // ─────────────────────────────────────────────────────────────
    // Statistics
    // ─────────────────────────────────────────────────────────────

    /// Snapshot current optimizer statistics.
    pub fn stats(&self) -> DistOptimizerStats {
        DistOptimizerStats {
            total_workers: self.workers.len(),
            active_workers: self.active_worker_count(),
            current_step: self.current_step,
            total_aggregations: self.total_aggregations,
            dropped_updates: self.dropped_updates,
            pending_layers: self.pending.len(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        AggregationStrategy, DistributedOptimizer, GradientUpdate, OptimizerDistError, WorkerId,
    };

    // ── helpers ──────────────────────────────────────────────────

    fn wid(s: &str) -> WorkerId {
        WorkerId::new(s)
    }

    fn make_update(worker: &str, layer: &str, grads: Vec<f64>, step: u64) -> GradientUpdate {
        GradientUpdate {
            worker_id: wid(worker),
            layer_id: layer.to_string(),
            gradients: grads,
            step,
            timestamp: step * 1000,
        }
    }

    fn sync_optimizer() -> DistributedOptimizer {
        DistributedOptimizer::new(AggregationStrategy::Synchronous)
    }

    fn async_optimizer(threshold: u64) -> DistributedOptimizer {
        DistributedOptimizer::new(AggregationStrategy::Asynchronous {
            staleness_threshold: threshold,
        })
    }

    fn fedavg_optimizer(rounds: u32) -> DistributedOptimizer {
        DistributedOptimizer::new(AggregationStrategy::FederatedAverage { rounds })
    }

    fn gossip_optimizer(fanout: usize) -> DistributedOptimizer {
        DistributedOptimizer::new(AggregationStrategy::GossipAverage { fanout })
    }

    // ── 1. WorkerId ──────────────────────────────────────────────

    #[test]
    fn worker_id_equality() {
        assert_eq!(wid("a"), wid("a"));
        assert_ne!(wid("a"), wid("b"));
    }

    #[test]
    fn worker_id_display() {
        assert_eq!(format!("{}", wid("worker-1")), "worker-1");
    }

    #[test]
    fn worker_id_ordering() {
        let mut ids = vec![wid("c"), wid("a"), wid("b")];
        ids.sort();
        assert_eq!(ids, vec![wid("a"), wid("b"), wid("c")]);
    }

    // ── 2. Registration ──────────────────────────────────────────

    #[test]
    fn register_new_worker_returns_true() {
        let mut opt = sync_optimizer();
        assert!(opt.register_worker(wid("w1")));
    }

    #[test]
    fn register_duplicate_returns_false() {
        let mut opt = sync_optimizer();
        opt.register_worker(wid("w1"));
        assert!(!opt.register_worker(wid("w1")));
    }

    #[test]
    fn worker_count_after_registration() {
        let mut opt = sync_optimizer();
        opt.register_worker(wid("w1"));
        opt.register_worker(wid("w2"));
        assert_eq!(opt.worker_count(), 2);
    }

    #[test]
    fn active_worker_count_initial() {
        let mut opt = sync_optimizer();
        opt.register_worker(wid("w1"));
        opt.register_worker(wid("w2"));
        assert_eq!(opt.active_worker_count(), 2);
    }

    // ── 3. Deregistration ────────────────────────────────────────

    #[test]
    fn deregister_known_worker() {
        let mut opt = sync_optimizer();
        opt.register_worker(wid("w1"));
        assert!(opt.deregister_worker(&wid("w1")));
        assert_eq!(opt.active_worker_count(), 0);
    }

    #[test]
    fn deregister_unknown_worker_returns_false() {
        let mut opt = sync_optimizer();
        assert!(!opt.deregister_worker(&wid("ghost")));
    }

    #[test]
    fn deregister_does_not_remove_worker_from_map() {
        let mut opt = sync_optimizer();
        opt.register_worker(wid("w1"));
        opt.deregister_worker(&wid("w1"));
        assert_eq!(opt.worker_count(), 1);
        assert_eq!(opt.active_worker_count(), 0);
    }

    // ── 4. Update submission — general ──────────────────────────

    #[test]
    fn submit_unknown_worker_errors() {
        let mut opt = sync_optimizer();
        let upd = make_update("ghost", "layer0", vec![1.0], 0);
        assert!(matches!(
            opt.submit_update(upd, 0),
            Err(OptimizerDistError::WorkerNotFound(_))
        ));
    }

    #[test]
    fn submit_update_increments_pending() {
        let mut opt = sync_optimizer();
        opt.register_worker(wid("w1"));
        opt.submit_update(make_update("w1", "layer0", vec![1.0], 0), 1000)
            .expect("should succeed");
        assert_eq!(opt.pending_updates_for("layer0"), 1);
    }

    #[test]
    fn submit_update_tracks_worker_state() {
        let mut opt = sync_optimizer();
        opt.register_worker(wid("w1"));
        opt.submit_update(make_update("w1", "layer0", vec![0.5], 2), 5000)
            .expect("ok");
        let active = opt.active_workers();
        let w = active
            .iter()
            .find(|w| w.worker_id == wid("w1"))
            .expect("found");
        assert_eq!(w.latest_step, 2);
        assert_eq!(w.last_seen, 5000);
        assert_eq!(w.total_updates, 1);
    }

    #[test]
    fn submit_dimension_mismatch_errors() {
        let mut opt = async_optimizer(10);
        opt.register_worker(wid("w1"));
        opt.register_worker(wid("w2"));
        opt.submit_update(make_update("w1", "layer0", vec![1.0, 2.0], 0), 0)
            .expect("ok");
        let err = opt
            .submit_update(make_update("w2", "layer0", vec![1.0], 0), 0)
            .unwrap_err();
        assert!(matches!(err, OptimizerDistError::DimensionMismatch { .. }));
    }

    // ── 5. Staleness (Async) ─────────────────────────────────────

    #[test]
    fn async_accepts_fresh_update() {
        let mut opt = async_optimizer(2);
        opt.register_worker(wid("w1"));
        opt.advance_step(); // current = 1
        opt.advance_step(); // current = 2
                            // step 0 is 2 steps behind — exactly at threshold
        let res = opt.submit_update(make_update("w1", "l0", vec![1.0], 0), 100);
        assert!(res.is_ok());
    }

    #[test]
    fn async_rejects_stale_update() {
        let mut opt = async_optimizer(1);
        opt.register_worker(wid("w1"));
        opt.advance_step(); // current = 1
        opt.advance_step(); // current = 2
                            // step 0 is 2 steps behind — exceeds threshold of 1
        let res = opt.submit_update(make_update("w1", "l0", vec![1.0], 0), 100);
        assert!(matches!(res, Err(OptimizerDistError::StaleUpdate { .. })));
    }

    #[test]
    fn async_increments_dropped_on_stale() {
        let mut opt = async_optimizer(0);
        opt.register_worker(wid("w1"));
        opt.advance_step(); // current = 1
        opt.submit_update(make_update("w1", "l0", vec![1.0], 0), 0)
            .ok();
        assert_eq!(opt.dropped_updates, 1);
    }

    // ── 6. Synchronous aggregation ──────────────────────────────

    #[test]
    fn sync_no_aggregate_until_all_workers_submit() {
        let mut opt = sync_optimizer();
        opt.register_worker(wid("w1"));
        opt.register_worker(wid("w2"));
        opt.submit_update(make_update("w1", "l0", vec![1.0, 2.0], 0), 0)
            .expect("ok");
        assert!(opt.try_aggregate("l0").is_none());
    }

    #[test]
    fn sync_aggregates_when_all_workers_submit() {
        let mut opt = sync_optimizer();
        opt.register_worker(wid("w1"));
        opt.register_worker(wid("w2"));
        opt.submit_update(make_update("w1", "l0", vec![1.0, 2.0], 0), 0)
            .expect("ok");
        opt.submit_update(make_update("w2", "l0", vec![3.0, 4.0], 0), 0)
            .expect("ok");
        let agg = opt.try_aggregate("l0").expect("should aggregate");
        assert_eq!(agg.contributing_workers, 2);
        assert!((agg.values[0] - 2.0).abs() < 1e-10); // (1+3)/2
        assert!((agg.values[1] - 3.0).abs() < 1e-10); // (2+4)/2
    }

    #[test]
    fn sync_clears_pending_on_advance_step() {
        let mut opt = sync_optimizer();
        opt.register_worker(wid("w1"));
        opt.submit_update(make_update("w1", "l0", vec![1.0], 0), 0)
            .expect("ok");
        opt.advance_step();
        assert_eq!(opt.pending_updates_for("l0"), 0);
    }

    #[test]
    fn sync_ignores_inactive_workers_for_quorum() {
        let mut opt = sync_optimizer();
        opt.register_worker(wid("w1"));
        opt.register_worker(wid("w2"));
        opt.deregister_worker(&wid("w2"));
        opt.submit_update(make_update("w1", "l0", vec![1.0], 0), 0)
            .expect("ok");
        // Only w1 active, quorum of 1 satisfied
        let agg = opt.try_aggregate("l0");
        assert!(agg.is_some());
    }

    // ── 7. FederatedAverage aggregation ─────────────────────────

    #[test]
    fn fedavg_no_aggregate_below_rounds() {
        let mut opt = fedavg_optimizer(3);
        for i in 0..3 {
            opt.register_worker(wid(&format!("w{i}")));
        }
        for i in 0..2usize {
            opt.submit_update(make_update(&format!("w{i}"), "l0", vec![1.0], 0), 0)
                .expect("ok");
        }
        assert!(opt.try_aggregate("l0").is_none());
    }

    #[test]
    fn fedavg_aggregates_at_rounds() {
        let mut opt = fedavg_optimizer(2);
        opt.register_worker(wid("w0"));
        opt.register_worker(wid("w1"));
        opt.submit_update(make_update("w0", "l0", vec![2.0], 0), 0)
            .expect("ok");
        opt.submit_update(make_update("w1", "l0", vec![4.0], 0), 0)
            .expect("ok");
        let agg = opt.try_aggregate("l0").expect("should aggregate");
        assert_eq!(agg.contributing_workers, 2);
        assert!((agg.values[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn fedavg_pending_cleared_after_aggregate() {
        let mut opt = fedavg_optimizer(1);
        opt.register_worker(wid("w0"));
        opt.submit_update(make_update("w0", "l0", vec![5.0], 0), 0)
            .expect("ok");
        opt.try_aggregate("l0");
        assert_eq!(opt.pending_updates_for("l0"), 0);
    }

    // ── 8. GossipAverage aggregation ────────────────────────────

    #[test]
    fn gossip_no_aggregate_below_fanout() {
        let mut opt = gossip_optimizer(3);
        opt.register_worker(wid("w0"));
        opt.register_worker(wid("w1"));
        opt.submit_update(make_update("w0", "l0", vec![1.0], 0), 0)
            .expect("ok");
        opt.submit_update(make_update("w1", "l0", vec![2.0], 0), 0)
            .expect("ok");
        assert!(opt.try_aggregate("l0").is_none());
    }

    #[test]
    fn gossip_aggregates_at_fanout() {
        let mut opt = gossip_optimizer(2);
        opt.register_worker(wid("w0"));
        opt.register_worker(wid("w1"));
        opt.submit_update(make_update("w0", "l0", vec![0.0], 0), 0)
            .expect("ok");
        opt.submit_update(make_update("w1", "l0", vec![2.0], 0), 0)
            .expect("ok");
        let agg = opt.try_aggregate("l0").expect("should aggregate");
        assert!((agg.values[0] - 1.0).abs() < 1e-10);
    }

    // ── 9. aggregate_gradients ───────────────────────────────────

    #[test]
    fn aggregate_empty_returns_empty() {
        let result = DistributedOptimizer::aggregate_gradients(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn aggregate_single_update() {
        let upd = make_update("w1", "l0", vec![1.0, 2.0, 3.0], 0);
        let result = DistributedOptimizer::aggregate_gradients(&[upd]);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn aggregate_mean_two_workers() {
        let u1 = make_update("w1", "l0", vec![0.0, 10.0], 0);
        let u2 = make_update("w2", "l0", vec![4.0, 6.0], 0);
        let result = DistributedOptimizer::aggregate_gradients(&[u1, u2]);
        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn aggregate_mean_three_workers() {
        let u1 = make_update("w1", "l0", vec![3.0], 0);
        let u2 = make_update("w2", "l0", vec![6.0], 0);
        let u3 = make_update("w3", "l0", vec![9.0], 0);
        let result = DistributedOptimizer::aggregate_gradients(&[u1, u2, u3]);
        assert!((result[0] - 6.0).abs() < 1e-10);
    }

    // ── 10. advance_step ─────────────────────────────────────────

    #[test]
    fn advance_step_increments_counter() {
        let mut opt = sync_optimizer();
        assert_eq!(opt.current_step, 0);
        opt.advance_step();
        assert_eq!(opt.current_step, 1);
        opt.advance_step();
        assert_eq!(opt.current_step, 2);
    }

    #[test]
    fn advance_step_async_preserves_pending() {
        let mut opt = async_optimizer(5);
        opt.register_worker(wid("w1"));
        opt.submit_update(make_update("w1", "l0", vec![1.0], 0), 0)
            .expect("ok");
        opt.advance_step();
        // Async does NOT clear pending on step advance
        assert_eq!(opt.pending_updates_for("l0"), 1);
    }

    // ── 11. evict_stale_workers ──────────────────────────────────

    #[test]
    fn evict_stale_workers_marks_inactive() {
        let mut opt = sync_optimizer();
        opt.register_worker(wid("w1"));
        // Simulate w1 last seen at t=100, evict anything older than 500ms
        // with now=1000 → cutoff=500 → 100 < 500 → evict
        opt.submit_update(make_update("w1", "l0", vec![1.0], 0), 100)
            .expect("ok");
        let evicted = opt.evict_stale_workers(500, 1000);
        assert_eq!(evicted, 1);
        assert_eq!(opt.active_worker_count(), 0);
    }

    #[test]
    fn evict_fresh_worker_not_evicted() {
        let mut opt = sync_optimizer();
        opt.register_worker(wid("w1"));
        opt.submit_update(make_update("w1", "l0", vec![1.0], 0), 900)
            .expect("ok");
        let evicted = opt.evict_stale_workers(500, 1000); // cutoff = 500, 900 > 500
        assert_eq!(evicted, 0);
        assert_eq!(opt.active_worker_count(), 1);
    }

    #[test]
    fn evict_already_inactive_worker_not_counted() {
        let mut opt = sync_optimizer();
        opt.register_worker(wid("w1"));
        opt.deregister_worker(&wid("w1"));
        let evicted = opt.evict_stale_workers(0, 1000);
        assert_eq!(evicted, 0);
    }

    // ── 12. Statistics ───────────────────────────────────────────

    #[test]
    fn stats_initial_state() {
        let opt = sync_optimizer();
        let s = opt.stats();
        assert_eq!(s.total_workers, 0);
        assert_eq!(s.active_workers, 0);
        assert_eq!(s.current_step, 0);
        assert_eq!(s.total_aggregations, 0);
        assert_eq!(s.dropped_updates, 0);
        assert_eq!(s.pending_layers, 0);
    }

    #[test]
    fn stats_total_aggregations_increments() {
        let mut opt = fedavg_optimizer(1);
        opt.register_worker(wid("w0"));
        opt.submit_update(make_update("w0", "l0", vec![1.0], 0), 0)
            .expect("ok");
        opt.try_aggregate("l0");
        let s = opt.stats();
        assert_eq!(s.total_aggregations, 1);
    }

    #[test]
    fn stats_pending_layers_counts_unique_layers() {
        let mut opt = fedavg_optimizer(99); // high threshold → never aggregate
        opt.register_worker(wid("w0"));
        opt.submit_update(make_update("w0", "l0", vec![1.0], 0), 0)
            .expect("ok");
        opt.submit_update(make_update("w0", "l1", vec![2.0], 0), 0)
            .expect("ok");
        let s = opt.stats();
        assert_eq!(s.pending_layers, 2);
    }

    // ── 13. last_aggregated ──────────────────────────────────────

    #[test]
    fn last_aggregated_none_before_aggregation() {
        let opt = sync_optimizer();
        assert!(opt.last_aggregated("l0").is_none());
    }

    #[test]
    fn last_aggregated_returns_result_after_aggregation() {
        let mut opt = fedavg_optimizer(1);
        opt.register_worker(wid("w0"));
        opt.submit_update(make_update("w0", "l0", vec![7.0], 0), 0)
            .expect("ok");
        opt.try_aggregate("l0");
        let agg = opt.last_aggregated("l0").expect("exists");
        assert!((agg.values[0] - 7.0).abs() < 1e-10);
    }

    // ── 14. Multi-layer independence ─────────────────────────────

    #[test]
    fn different_layers_aggregated_independently() {
        let mut opt = fedavg_optimizer(1);
        opt.register_worker(wid("w0"));
        opt.submit_update(make_update("w0", "l0", vec![1.0], 0), 0)
            .expect("ok");
        opt.submit_update(make_update("w0", "l1", vec![2.0], 0), 0)
            .expect("ok");
        let agg0 = opt.try_aggregate("l0").expect("l0 aggregated");
        let agg1 = opt.try_aggregate("l1").expect("l1 aggregated");
        assert!((agg0.values[0] - 1.0).abs() < 1e-10);
        assert!((agg1.values[0] - 2.0).abs() < 1e-10);
    }

    // ── 15. OptimizerDistError display ──────────────────────────

    #[test]
    fn error_worker_not_found_display() {
        let e = OptimizerDistError::WorkerNotFound("bob".to_string());
        assert!(e.to_string().contains("bob"));
    }

    #[test]
    fn error_stale_update_display() {
        let e = OptimizerDistError::StaleUpdate {
            worker: "w1".to_string(),
            step: 0,
            current: 5,
        };
        assert!(e.to_string().contains("w1"));
        assert!(e.to_string().contains('5'));
    }

    #[test]
    fn error_dimension_mismatch_display() {
        let e = OptimizerDistError::DimensionMismatch {
            layer: "conv1".to_string(),
            expected: 4,
            got: 3,
        };
        assert!(e.to_string().contains("conv1"));
        assert!(e.to_string().contains('4'));
        assert!(e.to_string().contains('3'));
    }

    // ── 16. Active workers list ──────────────────────────────────

    #[test]
    fn active_workers_returns_only_active() {
        let mut opt = sync_optimizer();
        opt.register_worker(wid("w1"));
        opt.register_worker(wid("w2"));
        opt.deregister_worker(&wid("w1"));
        let active = opt.active_workers();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].worker_id, wid("w2"));
    }

    // ── 17. Async aggregation produces correct mean ──────────────

    #[test]
    fn async_aggregate_produces_mean() {
        let mut opt = async_optimizer(10);
        opt.register_worker(wid("w1"));
        opt.register_worker(wid("w2"));
        opt.submit_update(make_update("w1", "l0", vec![0.0, 2.0], 0), 0)
            .expect("ok");
        opt.submit_update(make_update("w2", "l0", vec![4.0, 6.0], 0), 0)
            .expect("ok");
        let agg = opt.try_aggregate("l0").expect("aggregated");
        assert!((agg.values[0] - 2.0).abs() < 1e-10);
        assert!((agg.values[1] - 4.0).abs() < 1e-10);
    }

    // ── 18. Zero-active workers, sync never aggregates ───────────

    #[test]
    fn sync_with_no_active_workers_never_aggregates() {
        let mut opt = sync_optimizer();
        opt.register_worker(wid("w1"));
        opt.deregister_worker(&wid("w1"));
        // Even with a submitted update (from inactive), quorum cannot be met
        // because there are zero active workers.
        // Submitting to an inactive worker is still allowed (not deregistered).
        opt.submit_update(make_update("w1", "l0", vec![1.0], 0), 0)
            .expect("ok");
        assert!(opt.try_aggregate("l0").is_none());
    }

    // ── 19. Multiple advance_step calls ─────────────────────────

    #[test]
    fn multiple_advance_steps() {
        let mut opt = sync_optimizer();
        for _ in 0..10 {
            opt.advance_step();
        }
        assert_eq!(opt.current_step, 10);
    }

    // ── 20. Aggregation step field matches current_step ──────────

    #[test]
    fn aggregated_step_matches_current_step() {
        let mut opt = fedavg_optimizer(1);
        opt.register_worker(wid("w0"));
        opt.advance_step();
        opt.advance_step(); // current_step = 2
        opt.submit_update(make_update("w0", "l0", vec![1.0], 2), 200)
            .expect("ok");
        let agg = opt.try_aggregate("l0").expect("aggregated");
        assert_eq!(agg.step, 2);
    }

    // ── 21. Re-register after deregister (same slot) ─────────────

    #[test]
    fn cannot_re_register_after_deregister() {
        let mut opt = sync_optimizer();
        opt.register_worker(wid("w1"));
        opt.deregister_worker(&wid("w1"));
        // The worker still exists in the map → re-registration fails
        let result = opt.register_worker(wid("w1"));
        assert!(!result);
    }

    // ── 22. Gossip with extra updates above fanout ────────────────

    #[test]
    fn gossip_aggregates_all_pending_above_fanout() {
        let mut opt = gossip_optimizer(2);
        opt.register_worker(wid("w0"));
        opt.register_worker(wid("w1"));
        opt.register_worker(wid("w2"));
        for i in 0..3usize {
            opt.submit_update(make_update(&format!("w{i}"), "l0", vec![3.0], 0), 0)
                .expect("ok");
        }
        let agg = opt.try_aggregate("l0").expect("aggregated");
        assert_eq!(agg.contributing_workers, 3);
    }

    // ── 23. FedAvg accumulates stats correctly ───────────────────

    #[test]
    fn fedavg_total_aggregations_increments_across_rounds() {
        let mut opt = fedavg_optimizer(1);
        opt.register_worker(wid("w0"));
        for step in 0..5u64 {
            opt.submit_update(make_update("w0", "l0", vec![1.0], step), 0)
                .expect("ok");
            opt.try_aggregate("l0");
        }
        assert_eq!(opt.stats().total_aggregations, 5);
    }

    // ── 24. Evict multiple stale workers ────────────────────────

    #[test]
    fn evict_multiple_stale_workers() {
        let mut opt = sync_optimizer();
        for i in 0..4usize {
            opt.register_worker(wid(&format!("w{i}")));
            // first two workers were last seen at t=10, rest at t=800
            let ts = if i < 2 { 10 } else { 800 };
            opt.submit_update(make_update(&format!("w{i}"), "l0", vec![1.0], 0), ts)
                .expect("ok");
        }
        // now=1000, max_age=500 → cutoff=500 → t=10 < 500 → evict
        let evicted = opt.evict_stale_workers(500, 1000);
        assert_eq!(evicted, 2);
        assert_eq!(opt.active_worker_count(), 2);
    }
}
