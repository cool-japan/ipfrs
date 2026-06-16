//! Backward pass coordination via TensorSwap.
//!
//! This module provides the pure-data structures that coordinate distributed
//! backward-pass gradient streaming.  No async or network I/O lives here;
//! the transport layer calls these types as CIDs arrive.

use serde::{Deserialize, Serialize};

use super::tensor::GradientAggregator;
use super::GradientError;

// ── Standalone helpers (also re-exported from mod.rs) ─────────────────────

/// Compute the unweighted mean of a collection of gradient vectors.
///
/// All vectors must have the same length.  Returns [`GradientError::EmptyGradients`]
/// for an empty input and [`GradientError::DimensionMismatch`] for vectors of
/// differing lengths.
pub fn federated_average(gradients: &[Vec<f32>]) -> Result<Vec<f32>, GradientError> {
    if gradients.is_empty() {
        return Err(GradientError::EmptyGradients);
    }
    let dim = gradients[0].len();
    if gradients.iter().any(|g| g.len() != dim) {
        return Err(GradientError::DimensionMismatch);
    }
    let n = gradients.len() as f32;
    let mut avg = vec![0.0f32; dim];
    for grad in gradients {
        for (a, &g) in avg.iter_mut().zip(grad.iter()) {
            *a += g / n;
        }
    }
    Ok(avg)
}

/// Clip a gradient vector in-place so that its L2 norm does not exceed `max_norm`.
///
/// If the current norm is already ≤ `max_norm` the vector is left unchanged.
pub fn clip_gradient_norm(gradient: &mut [f32], max_norm: f32) {
    let norm: f32 = gradient.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for x in gradient.iter_mut() {
            *x *= scale;
        }
    }
}

// ── BackwardStepStatus ─────────────────────────────────────────────────────

/// Coordination status for a distributed backward pass step
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackwardStepStatus {
    /// Waiting for gradient from this peer
    Pending,
    /// Gradient has been requested from the peer
    GradientRequested { peer_id: String },
    /// Gradient CID has been received
    GradientReceived { cid: String },
    /// Gradient has been included in aggregation
    Aggregated,
    /// This peer's gradient could not be collected
    Failed { reason: String },
}

// ── BackwardPassStep ───────────────────────────────────────────────────────

/// Tracks one layer's backward pass across multiple peers
#[derive(Debug)]
pub struct BackwardPassStep {
    /// Node (layer) identifier
    pub node_id: String,
    /// Operation type (e.g. "matmul", "relu")
    pub op: String,
    /// Per-peer status map
    pub peer_contributions: std::collections::HashMap<String, BackwardStepStatus>,
    /// CID of the aggregated gradient (set after aggregation)
    pub aggregated_gradient_cid: Option<String>,
    /// Wall-clock start time of this step
    pub started_at: std::time::Instant,
}

impl BackwardPassStep {
    /// Create a new step with no peers yet
    pub fn new(node_id: String, op: String) -> Self {
        Self {
            node_id,
            op,
            peer_contributions: std::collections::HashMap::new(),
            aggregated_gradient_cid: None,
            started_at: std::time::Instant::now(),
        }
    }

    /// Register a new peer as a contributor for this step
    pub fn add_peer(&mut self, peer_id: &str) {
        self.peer_contributions
            .entry(peer_id.to_string())
            .or_insert(BackwardStepStatus::Pending);
    }

    /// Record that a gradient CID was received from `peer_id`
    pub fn record_gradient_received(&mut self, peer_id: &str, cid: &str) {
        self.peer_contributions.insert(
            peer_id.to_string(),
            BackwardStepStatus::GradientReceived {
                cid: cid.to_string(),
            },
        );
    }

    /// Record a failure for `peer_id`
    pub fn record_gradient_failed(&mut self, peer_id: &str, reason: &str) {
        self.peer_contributions.insert(
            peer_id.to_string(),
            BackwardStepStatus::Failed {
                reason: reason.to_string(),
            },
        );
    }

    /// Returns `true` when every peer has either received or failed
    pub fn is_complete(&self) -> bool {
        self.peer_contributions.values().all(|s| {
            matches!(
                s,
                BackwardStepStatus::GradientReceived { .. }
                    | BackwardStepStatus::Aggregated
                    | BackwardStepStatus::Failed { .. }
            )
        })
    }

    /// Returns `true` when all peers have received (no pending/failed)
    pub fn ready_to_aggregate(&self) -> bool {
        !self.peer_contributions.is_empty()
            && self.peer_contributions.values().all(|s| {
                matches!(
                    s,
                    BackwardStepStatus::GradientReceived { .. } | BackwardStepStatus::Aggregated
                )
            })
    }

    /// Number of peers whose gradient has been received
    pub fn received_count(&self) -> usize {
        self.peer_contributions
            .values()
            .filter(|s| {
                matches!(
                    s,
                    BackwardStepStatus::GradientReceived { .. } | BackwardStepStatus::Aggregated
                )
            })
            .count()
    }

    /// Number of peers that have failed
    pub fn failed_count(&self) -> usize {
        self.peer_contributions
            .values()
            .filter(|s| matches!(s, BackwardStepStatus::Failed { .. }))
            .count()
    }
}

// ── AggregationMethod ──────────────────────────────────────────────────────

/// Aggregation strategy for combining gradients from multiple peers
#[derive(Debug, Clone, PartialEq)]
pub enum AggregationMethod {
    /// Sum all peer gradients
    Sum,
    /// Unweighted arithmetic mean
    Mean,
    /// Weighted mean (weights must sum to > 0)
    WeightedMean { weights: Vec<f32> },
    /// Federated Averaging (identical to Mean for gradient tensors)
    FedAvg,
}

// ── BackwardPassConfig ─────────────────────────────────────────────────────

/// Tuning knobs for a `BackwardPassCoordinator`
#[derive(Debug, Clone)]
pub struct BackwardPassConfig {
    /// Maximum number of participating peers (informational; not enforced as hard limit)
    pub max_peers: usize,
    /// Aggregation strategy
    pub aggregation: AggregationMethod,
    /// Per-step timeout; steps older than this are considered timed-out
    pub timeout: std::time::Duration,
    /// Optional L2 gradient-norm clipping threshold
    pub gradient_clipping: Option<f32>,
}

impl Default for BackwardPassConfig {
    fn default() -> Self {
        Self {
            max_peers: 8,
            aggregation: AggregationMethod::FedAvg,
            timeout: std::time::Duration::from_secs(60),
            gradient_clipping: None,
        }
    }
}

// ── BackwardPassStats ──────────────────────────────────────────────────────

/// Runtime statistics snapshot for a [`BackwardPassCoordinator`]
#[derive(Debug, Default)]
pub struct BackwardPassStats {
    /// Total number of scheduled steps
    pub total_steps: usize,
    /// Steps where all peers have been aggregated
    pub completed_steps: usize,
    /// Steps still waiting for peer contributions
    pub pending_steps: usize,
    /// Steps that have at least one failed peer
    pub failed_steps: usize,
    /// Bytes held in the accumulation buffer
    pub total_gradient_bytes: usize,
    /// Number of distinct peers registered across all steps
    pub participating_peers: usize,
}

// ── BackwardPassCoordinator ────────────────────────────────────────────────

/// Coordinates backward pass gradient streaming via TensorSwap.
///
/// This is a **pure data structure** — no async or network I/O lives here.
/// The transport layer calls `receive_gradient` as CIDs arrive and calls
/// `aggregate_gradients` once a step is [`BackwardPassStep::ready_to_aggregate`].
pub struct BackwardPassCoordinator {
    /// Steps in topological order (output layer first — reverse of forward pass)
    steps: Vec<BackwardPassStep>,
    /// Set of peer IDs that participate in the backward pass
    participating_peers: std::collections::HashSet<String>,
    /// Global learning rate applied during [`apply_gradient`]
    learning_rate: f32,
    /// Per-node accumulated gradient buffer (keyed by `node_id`)
    accumulation_buffer: std::collections::HashMap<String, Vec<f32>>,
    /// Coordinator configuration
    config: BackwardPassConfig,
}

impl BackwardPassCoordinator {
    /// Create a new coordinator with `learning_rate = 0.01` and the supplied config
    pub fn new(config: BackwardPassConfig) -> Self {
        Self {
            steps: Vec::new(),
            participating_peers: std::collections::HashSet::new(),
            learning_rate: 0.01,
            accumulation_buffer: std::collections::HashMap::new(),
            config,
        }
    }

    /// Override the default learning rate
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Register a computation node and its expected peer contributors
    pub fn schedule_step(&mut self, node_id: &str, op: &str, peers: &[&str]) {
        let mut step = BackwardPassStep::new(node_id.to_string(), op.to_string());
        for &peer in peers {
            step.add_peer(peer);
            self.participating_peers.insert(peer.to_string());
        }
        self.steps.push(step);
    }

    /// Record that `peer_id` has sent a gradient with content-address `gradient_cid`
    /// for the step identified by `node_id`.
    pub fn receive_gradient(
        &mut self,
        node_id: &str,
        peer_id: &str,
        gradient_cid: &str,
    ) -> Result<(), GradientError> {
        let step = self
            .steps
            .iter_mut()
            .find(|s| s.node_id == node_id)
            .ok_or_else(|| GradientError::NodeNotFound(node_id.to_string()))?;

        if !step.peer_contributions.contains_key(peer_id) {
            return Err(GradientError::PeerNotFound(peer_id.to_string()));
        }

        step.record_gradient_received(peer_id, gradient_cid);
        Ok(())
    }

    /// Aggregate a set of `(peer_id, gradient_values)` pairs for `node_id`.
    ///
    /// Applies the configured [`AggregationMethod`] and optional gradient clipping,
    /// then stores the result in the accumulation buffer.
    pub fn aggregate_gradients(
        &mut self,
        node_id: &str,
        gradient_data: Vec<(String, Vec<f32>)>,
    ) -> Result<Vec<f32>, GradientError> {
        if gradient_data.is_empty() {
            return Err(GradientError::EmptyGradients);
        }

        let dim = gradient_data[0].1.len();
        if gradient_data.iter().any(|(_, g)| g.len() != dim) {
            return Err(GradientError::DimensionMismatch);
        }

        let gradients: Vec<Vec<f32>> = gradient_data.into_iter().map(|(_, g)| g).collect();

        let mut aggregated = match &self.config.aggregation {
            AggregationMethod::Sum => {
                let mut sum = vec![0.0f32; dim];
                for grad in &gradients {
                    for (a, &g) in sum.iter_mut().zip(grad.iter()) {
                        *a += g;
                    }
                }
                sum
            }
            AggregationMethod::Mean | AggregationMethod::FedAvg => federated_average(&gradients)?,
            AggregationMethod::WeightedMean { weights } => {
                let w: Vec<f32> = weights.clone();
                GradientAggregator::weighted_average(&gradients, &w)?
            }
        };

        // Apply gradient clipping before storing
        self.clip_gradients(&mut aggregated);

        // Mark contributions as aggregated
        if let Some(step) = self.steps.iter_mut().find(|s| s.node_id == node_id) {
            for status in step.peer_contributions.values_mut() {
                if matches!(status, BackwardStepStatus::GradientReceived { .. }) {
                    *status = BackwardStepStatus::Aggregated;
                }
            }
        }

        self.accumulation_buffer
            .insert(node_id.to_string(), aggregated.clone());

        Ok(aggregated)
    }

    /// Apply L2 gradient-norm clipping if configured
    pub fn clip_gradients(&self, gradients: &mut [f32]) {
        if let Some(max_norm) = self.config.gradient_clipping {
            clip_gradient_norm(gradients, max_norm);
        }
    }

    /// Apply aggregated gradient to a parameter vector using the stored learning rate
    pub fn apply_gradient(
        &self,
        params: &mut [f32],
        gradient: &[f32],
    ) -> Result<(), GradientError> {
        if params.len() != gradient.len() {
            return Err(GradientError::DimensionMismatch);
        }
        for (p, &g) in params.iter_mut().zip(gradient.iter()) {
            *p -= self.learning_rate * g;
        }
        Ok(())
    }

    /// Return a reference to the first step that is ready to aggregate
    pub fn next_ready_step(&self) -> Option<&BackwardPassStep> {
        self.steps.iter().find(|s| s.ready_to_aggregate())
    }

    /// Collect statistics about the current backward pass state
    pub fn stats(&self) -> BackwardPassStats {
        let total_steps = self.steps.len();
        let completed_steps = self
            .steps
            .iter()
            .filter(|s| {
                s.peer_contributions
                    .values()
                    .all(|st| matches!(st, BackwardStepStatus::Aggregated))
                    && !s.peer_contributions.is_empty()
            })
            .count();
        let failed_steps = self
            .steps
            .iter()
            .filter(|s| {
                s.peer_contributions
                    .values()
                    .any(|st| matches!(st, BackwardStepStatus::Failed { .. }))
            })
            .count();
        let pending_steps = total_steps - completed_steps - failed_steps;

        let total_gradient_bytes = self
            .accumulation_buffer
            .values()
            .map(|v| v.len() * std::mem::size_of::<f32>())
            .sum();

        BackwardPassStats {
            total_steps,
            completed_steps,
            pending_steps,
            failed_steps,
            total_gradient_bytes,
            participating_peers: self.participating_peers.len(),
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod backward_pass_tests {
    use super::*;
    use std::time::Duration;

    fn default_config(method: AggregationMethod) -> BackwardPassConfig {
        BackwardPassConfig {
            max_peers: 4,
            aggregation: method,
            timeout: Duration::from_secs(30),
            gradient_clipping: None,
        }
    }

    // ── schedule_step + receive_gradient + next_ready_step ────────────────

    #[test]
    fn test_schedule_and_receive_gradient() {
        let config = BackwardPassConfig {
            max_peers: 3,
            aggregation: AggregationMethod::Mean,
            timeout: Duration::from_secs(30),
            gradient_clipping: None,
        };
        let mut coord = BackwardPassCoordinator::new(config);
        coord.schedule_step("layer1", "matmul", &["peer1", "peer2"]);

        coord
            .receive_gradient("layer1", "peer1", "cid_abc")
            .expect("peer1 receive");
        coord
            .receive_gradient("layer1", "peer2", "cid_def")
            .expect("peer2 receive");

        assert!(
            coord.next_ready_step().is_some(),
            "step should be ready after both peers reported"
        );
    }

    #[test]
    fn test_receive_gradient_unknown_node() {
        let mut coord = BackwardPassCoordinator::new(BackwardPassConfig::default());
        let result = coord.receive_gradient("ghost_layer", "peer1", "cid_x");
        assert!(matches!(result, Err(GradientError::NodeNotFound(_))));
    }

    #[test]
    fn test_receive_gradient_unknown_peer() {
        let mut coord = BackwardPassCoordinator::new(BackwardPassConfig::default());
        coord.schedule_step("layer1", "relu", &["peer1"]);
        let result = coord.receive_gradient("layer1", "peer_unknown", "cid_x");
        assert!(matches!(result, Err(GradientError::PeerNotFound(_))));
    }

    // ── federated_average (re-tested here as integration) ─────────────────

    #[test]
    fn test_federated_average() {
        let g1 = vec![1.0f32, 2.0, 3.0];
        let g2 = vec![3.0f32, 4.0, 5.0];
        let avg = federated_average(&[g1, g2]).expect("federated_average");
        assert!((avg[0] - 2.0).abs() < 1e-6, "avg[0] = {}", avg[0]);
        assert!((avg[1] - 3.0).abs() < 1e-6, "avg[1] = {}", avg[1]);
        assert!((avg[2] - 4.0).abs() < 1e-6, "avg[2] = {}", avg[2]);
    }

    #[test]
    fn test_federated_average_single() {
        let g = vec![1.0f32, 2.0, 3.0];
        let avg = federated_average(std::slice::from_ref(&g)).expect("single gradient average");
        assert_eq!(avg, g);
    }

    #[test]
    fn test_federated_average_empty() {
        let result = federated_average(&[]);
        assert!(matches!(result, Err(GradientError::EmptyGradients)));
    }

    #[test]
    fn test_federated_average_dimension_mismatch() {
        let g1 = vec![1.0f32, 2.0];
        let g2 = vec![1.0f32, 2.0, 3.0];
        let result = federated_average(&[g1, g2]);
        assert!(matches!(result, Err(GradientError::DimensionMismatch)));
    }

    // ── clip_gradient_norm ────────────────────────────────────────────────

    #[test]
    fn test_gradient_clipping() {
        let mut g = vec![3.0f32, 4.0]; // L2 norm = 5.0
        clip_gradient_norm(&mut g, 1.0);
        let norm: f32 = g.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "clipped norm should be 1.0, got {norm}"
        );
    }

    #[test]
    fn test_gradient_clipping_no_op_when_within_bound() {
        let mut g = vec![0.3f32, 0.4]; // norm = 0.5 < 1.0
        let original = g.clone();
        clip_gradient_norm(&mut g, 1.0);
        assert_eq!(
            g, original,
            "gradient must be unchanged when norm < max_norm"
        );
    }

    // ── apply_gradient ────────────────────────────────────────────────────

    #[test]
    fn test_apply_gradient_with_lr() {
        let config = BackwardPassConfig {
            max_peers: 2,
            aggregation: AggregationMethod::Mean,
            timeout: Duration::from_secs(30),
            gradient_clipping: None,
        };
        let coord = BackwardPassCoordinator::new(config).with_learning_rate(0.1);
        let mut params = vec![1.0f32, 2.0, 3.0];
        let gradient = vec![0.5f32, 1.0, 1.5];
        coord
            .apply_gradient(&mut params, &gradient)
            .expect("apply_gradient");

        // params[i] = params[i] - 0.1 * gradient[i]
        assert!((params[0] - 0.95).abs() < 1e-6, "params[0] = {}", params[0]);
        assert!((params[1] - 1.90).abs() < 1e-6, "params[1] = {}", params[1]);
        assert!((params[2] - 2.85).abs() < 1e-6, "params[2] = {}", params[2]);
    }

    #[test]
    fn test_apply_gradient_dimension_mismatch() {
        let coord = BackwardPassCoordinator::new(BackwardPassConfig::default());
        let mut params = vec![1.0f32, 2.0];
        let gradient = vec![0.5f32, 1.0, 1.5];
        let result = coord.apply_gradient(&mut params, &gradient);
        assert!(matches!(result, Err(GradientError::DimensionMismatch)));
    }

    // ── stats ─────────────────────────────────────────────────────────────

    #[test]
    fn test_backward_pass_stats() {
        let mut coord = BackwardPassCoordinator::new(default_config(AggregationMethod::FedAvg));
        coord.schedule_step("layer1", "matmul", &["peer1", "peer2"]);
        coord.schedule_step("layer2", "relu", &["peer1", "peer2"]);

        let stats = coord.stats();
        assert_eq!(stats.total_steps, 2);
        assert_eq!(stats.participating_peers, 2);
        assert_eq!(stats.completed_steps, 0);
    }

    // ── aggregation methods ───────────────────────────────────────────────

    #[test]
    fn test_aggregation_methods() {
        // Sum
        let mut coord = BackwardPassCoordinator::new(default_config(AggregationMethod::Sum));
        coord.schedule_step("l1", "op", &["p1", "p2"]);
        coord
            .receive_gradient("l1", "p1", "cid1")
            .expect("receive p1");
        coord
            .receive_gradient("l1", "p2", "cid2")
            .expect("receive p2");

        let data = vec![
            ("p1".to_string(), vec![1.0f32, 2.0]),
            ("p2".to_string(), vec![3.0f32, 4.0]),
        ];
        let agg = coord
            .aggregate_gradients("l1", data)
            .expect("aggregate sum");
        assert!((agg[0] - 4.0).abs() < 1e-6, "sum[0] = {}", agg[0]);
        assert!((agg[1] - 6.0).abs() < 1e-6, "sum[1] = {}", agg[1]);
    }

    #[test]
    fn test_aggregation_weighted_mean() {
        let config = BackwardPassConfig {
            max_peers: 2,
            aggregation: AggregationMethod::WeightedMean {
                weights: vec![1.0, 3.0],
            },
            timeout: Duration::from_secs(30),
            gradient_clipping: None,
        };
        let mut coord = BackwardPassCoordinator::new(config);
        coord.schedule_step("l1", "op", &["p1", "p2"]);
        coord.receive_gradient("l1", "p1", "c1").expect("p1");
        coord.receive_gradient("l1", "p2", "c2").expect("p2");

        let data = vec![
            ("p1".to_string(), vec![0.0f32]),
            ("p2".to_string(), vec![4.0f32]),
        ];
        // weighted mean = (1*0 + 3*4) / 4 = 3.0
        let agg = coord
            .aggregate_gradients("l1", data)
            .expect("weighted mean");
        assert!(
            (agg[0] - 3.0).abs() < 1e-5,
            "weighted mean = {}, expected 3.0",
            agg[0]
        );
    }

    #[test]
    fn test_aggregation_with_clipping() {
        let config = BackwardPassConfig {
            max_peers: 1,
            aggregation: AggregationMethod::Mean,
            timeout: Duration::from_secs(30),
            gradient_clipping: Some(1.0),
        };
        let mut coord = BackwardPassCoordinator::new(config);
        coord.schedule_step("l1", "op", &["p1"]);
        coord.receive_gradient("l1", "p1", "c1").expect("p1");

        let data = vec![("p1".to_string(), vec![3.0f32, 4.0])]; // norm = 5
        let agg = coord.aggregate_gradients("l1", data).expect("aggregate");
        let norm: f32 = agg.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "clipped norm = {norm}");
    }

    // ── step completion tracking ──────────────────────────────────────────

    #[test]
    fn test_step_completion_tracking() {
        let mut step = BackwardPassStep::new("layer1".to_string(), "matmul".to_string());
        step.add_peer("p1");
        step.add_peer("p2");
        step.add_peer("p3");

        assert!(!step.is_complete(), "not complete yet");
        assert_eq!(step.received_count(), 0);
        assert_eq!(step.failed_count(), 0);

        step.record_gradient_received("p1", "cid1");
        step.record_gradient_received("p2", "cid2");
        assert!(!step.is_complete(), "still waiting for p3");
        assert_eq!(step.received_count(), 2);

        step.record_gradient_failed("p3", "timeout");
        assert!(step.is_complete(), "complete after failure");
        assert_eq!(step.failed_count(), 1);
        assert!(
            !step.ready_to_aggregate(),
            "not ready_to_aggregate with failure"
        );
    }

    #[test]
    fn test_step_ready_to_aggregate_all_received() {
        let mut step = BackwardPassStep::new("l".to_string(), "op".to_string());
        step.add_peer("p1");
        step.add_peer("p2");

        assert!(!step.ready_to_aggregate());

        step.record_gradient_received("p1", "c1");
        step.record_gradient_received("p2", "c2");
        assert!(step.ready_to_aggregate());
    }
}
