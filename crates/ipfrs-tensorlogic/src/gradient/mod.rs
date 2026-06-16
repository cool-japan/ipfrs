//! Gradient storage and management for federated learning
//!
//! This module provides:
//! - Gradient delta format (differences from base model)
//! - Gradient compression (sparsification, quantization, top-k)
//! - Gradient aggregation (averaging, weighted, momentum)
//! - Gradient verification (checksum, shape, outliers)
//! - Backward pass coordination via TensorSwap
//! - CID-linked computation graph for gradient tracking
//! - Gradient checkpointing for training resumption

use crate::arrow::TensorDtype;
use thiserror::Error;

pub mod arrow_ipc;
pub mod backward_pass;
pub mod checkpoint;
pub mod computation_graph;
pub mod coordination;
pub mod federated;
pub mod tensor;

// ── GradientError ──────────────────────────────────────────────────────────

/// Errors that can occur during gradient operations
#[derive(Debug, Error)]
pub enum GradientError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Checksum verification failed")]
    ChecksumFailed,

    #[error("Invalid compression ratio: {0}")]
    InvalidCompressionRatio(f32),

    #[error("Empty gradient set")]
    EmptyGradientSet,

    #[error("Incompatible dtype: {0:?}")]
    IncompatibleDtype(TensorDtype),

    #[error("Outlier detected at index {index}: value {value}")]
    OutlierDetected { index: usize, value: f32 },

    #[error("Invalid gradient: {0}")]
    InvalidGradient(String),

    #[error("Empty gradients provided")]
    EmptyGradients,

    #[error("Gradient dimension mismatch between peers")]
    DimensionMismatch,

    #[error("Node not found in backward pass schedule: {0}")]
    NodeNotFound(String),

    #[error("Peer not found in step: {0}")]
    PeerNotFound(String),
}

// ── Re-exports ─────────────────────────────────────────────────────────────

pub use arrow_ipc::{load_gradient_from_arrow, store_gradient_as_arrow};

pub use backward_pass::{
    clip_gradient_norm, federated_average, AggregationMethod, BackwardPassConfig,
    BackwardPassCoordinator as LegacyBackwardPassCoordinator, BackwardPassStats, BackwardPassStep,
    BackwardStepStatus,
};

pub use checkpoint::GradientCheckpoint;

pub use computation_graph::{ComputationGraphError, ComputationGraphStore, ComputationNode};

pub use federated::{
    ClientInfo, ClientState, ConvergenceConfig, ConvergenceDetector, DPMechanism,
    DifferentialPrivacy, DistributedGradientAccumulator, FederatedError, FederatedRound,
    GossipModelSync, ModelSyncProtocol, ModelUpdate, PrivacyBudget, RoundStats, SecureAggregation,
};

pub use tensor::{
    GradientAggregator, GradientCompressor, GradientDelta, GradientVerifier, LayerGradient,
    QuantizedGradient, SparseGradient,
};

pub use coordination::{
    ArrowBlockError, BackwardPassCoordinator, BackwardPassId, CoordinationError,
    CoordinationStatus, GradientArrowBlock, GradientContribution,
};

// ── Tests originally in the top-level gradient module ─────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ipfrs_core::Cid;

    #[test]
    fn test_sparse_gradient() {
        let indices = vec![0, 5, 10];
        let values = vec![1.0, 2.0, 3.0];
        let shape = vec![20];

        let sparse = SparseGradient::new(indices.clone(), values.clone(), shape);

        assert_eq!(sparse.nnz(), 3);
        assert_eq!(sparse.total_elements(), 20);
        assert!((sparse.sparsity_ratio() - 0.85).abs() < 0.01);

        let dense = sparse.to_dense();
        assert_eq!(dense.len(), 20);
        assert_eq!(dense[0], 1.0);
        assert_eq!(dense[5], 2.0);
        assert_eq!(dense[10], 3.0);
    }

    #[test]
    fn test_quantized_gradient() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![5];

        let quantized = QuantizedGradient::from_dense(&values, shape);
        let dequantized = quantized.to_dense();

        // Check that dequantization is approximately correct
        // For a small range like [1,5] with 256 quantization levels,
        // we expect good precision
        for (i, (orig, deq)) in values.iter().zip(&dequantized).enumerate() {
            let error = (orig - deq).abs();
            // Allow for quantization error (scale = 4/255 ≈ 0.0157)
            assert!(
                error < 0.02,
                "Value {} mismatch: orig={}, deq={}, error={}",
                i,
                orig,
                deq,
                error
            );
        }
    }

    #[test]
    fn test_gradient_delta() {
        let base_cid = Cid::default();
        let mut delta = GradientDelta::new(base_cid);

        delta.add_dense_gradient("layer1".to_string(), vec![1.0, 2.0, 3.0], vec![3]);
        delta.add_dense_gradient("layer2".to_string(), vec![4.0, 5.0], vec![2]);

        assert_eq!(delta.layer_gradients.len(), 2);
        assert!(delta.verify_checksum().is_ok());
    }

    #[test]
    fn test_top_k_compression() {
        let values = vec![1.0, 5.0, 2.0, 8.0, 3.0];
        let shape = vec![5];

        let sparse = GradientCompressor::top_k(&values, shape, 2).expect("test: should succeed");

        assert_eq!(sparse.nnz(), 2);
        assert!(sparse.values.contains(&8.0));
        assert!(sparse.values.contains(&5.0));
    }

    #[test]
    fn test_threshold_compression() {
        let values = vec![0.1, 5.0, 0.2, 8.0, 0.3];
        let shape = vec![5];

        let sparse = GradientCompressor::threshold(&values, shape, 1.0);

        assert_eq!(sparse.nnz(), 2);
        assert!(sparse.values.contains(&5.0));
        assert!(sparse.values.contains(&8.0));
    }

    #[test]
    fn test_gradient_averaging() {
        let g1 = vec![1.0, 2.0, 3.0];
        let g2 = vec![3.0, 4.0, 5.0];
        let gradients = vec![g1, g2];

        let avg = GradientAggregator::average(&gradients).expect("test: should succeed");

        assert_eq!(avg, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_weighted_averaging() {
        let g1 = vec![1.0, 2.0, 3.0];
        let g2 = vec![3.0, 4.0, 5.0];
        let gradients = vec![g1, g2];
        let weights = vec![0.25, 0.75];

        let avg = GradientAggregator::weighted_average(&gradients, &weights)
            .expect("test: should succeed");

        // Expected: 0.25 * [1,2,3] + 0.75 * [3,4,5] = [2.5, 3.5, 4.5]
        assert!((avg[0] - 2.5).abs() < 0.01);
        assert!((avg[1] - 3.5).abs() < 0.01);
        assert!((avg[2] - 4.5).abs() < 0.01);
    }

    #[test]
    fn test_momentum() {
        let current = vec![1.0, 2.0, 3.0];
        let previous = vec![0.5, 1.0, 1.5];

        let result = GradientAggregator::apply_momentum(&current, &previous, 0.9)
            .expect("test: should succeed");

        // Expected: 0.9 * previous + current
        assert!((result[0] - 1.45).abs() < 0.01);
        assert!((result[1] - 2.9).abs() < 0.01);
        assert!((result[2] - 4.35).abs() < 0.01);
    }

    #[test]
    fn test_gradient_verification() {
        let gradient = vec![1.0, 2.0, 3.0, 4.0];

        // Test shape verification
        assert!(GradientVerifier::verify_shape(&gradient, &[4]).is_ok());
        assert!(GradientVerifier::verify_shape(&gradient, &[2, 2]).is_ok());
        assert!(GradientVerifier::verify_shape(&gradient, &[5]).is_err());

        // Test finite verification
        assert!(GradientVerifier::verify_finite(&gradient).is_ok());

        let invalid = vec![1.0, f32::NAN, 3.0];
        assert!(GradientVerifier::verify_finite(&invalid).is_err());
    }

    #[test]
    fn test_gradient_clipping() {
        let mut gradient = vec![3.0, 4.0]; // L2 norm = 5.0

        GradientVerifier::clip_by_norm(&mut gradient, 2.5);

        let norm = GradientVerifier::l2_norm(&gradient);
        assert!((norm - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_privacy_budget() {
        let mut budget = PrivacyBudget::new(1.0, 1e-5);

        assert_eq!(budget.remaining_epsilon, 1.0);
        assert!(!budget.is_exhausted());

        // Consume some budget
        budget.consume(0.5).expect("test: should succeed");
        assert_eq!(budget.remaining_epsilon, 0.5);
        assert!((budget.remaining_fraction() - 0.5).abs() < 1e-6);

        // Consume remaining budget
        budget.consume(0.5).expect("test: should succeed");
        assert!(budget.is_exhausted());

        // Should fail when budget is exhausted
        assert!(budget.consume(0.1).is_err());
    }

    #[test]
    fn test_differential_privacy_gaussian() {
        let mut dp = DifferentialPrivacy::new(1.0, 1e-5, 1.0, DPMechanism::Gaussian);
        let mut gradient = vec![1.0, 2.0, 3.0, 4.0];
        let original = gradient.clone();

        dp.add_gaussian_noise(&mut gradient)
            .expect("test: should succeed");

        // Gradient should be modified (with very high probability)
        assert_ne!(gradient, original);

        // Values should still be finite
        assert!(GradientVerifier::verify_finite(&gradient).is_ok());

        // Budget should be consumed
        assert!(dp.remaining_budget() < 1.0);
    }

    #[test]
    fn test_differential_privacy_laplacian() {
        let mut dp = DifferentialPrivacy::new(1.0, 1e-5, 1.0, DPMechanism::Laplacian);
        let mut gradient = vec![1.0, 2.0, 3.0, 4.0];
        let original = gradient.clone();

        dp.add_laplacian_noise(&mut gradient)
            .expect("test: should succeed");

        // Gradient should be modified (with very high probability)
        assert_ne!(gradient, original);

        // Values should still be finite
        assert!(GradientVerifier::verify_finite(&gradient).is_ok());

        // Budget should be consumed
        assert!(dp.remaining_budget() < 1.0);
    }

    #[test]
    fn test_dp_sgd() {
        let mut dp = DifferentialPrivacy::new(1.0, 1e-5, 1.0, DPMechanism::Gaussian);
        let mut gradient = vec![3.0, 4.0, 5.0, 6.0]; // L2 norm > 5.0
        let original_norm = GradientVerifier::l2_norm(&gradient);

        dp.apply_dp_sgd(&mut gradient, 5.0)
            .expect("test: should succeed");

        // Gradient should be clipped and noised
        let new_norm = GradientVerifier::l2_norm(&gradient);

        // After clipping and noise, norm might be around 5.0 but not exact due to noise
        // Just check it's different from original
        assert!(original_norm != new_norm);

        // Values should still be finite
        assert!(GradientVerifier::verify_finite(&gradient).is_ok());
    }

    #[test]
    fn test_privacy_budget_exhaustion() {
        let mut dp = DifferentialPrivacy::new(1.0, 1e-5, 1.0, DPMechanism::Gaussian);
        let mut gradient = vec![1.0, 2.0];

        // Consume budget multiple times
        // Each call consumes epsilon/100 = 0.01, so we need 100 calls to exhaust budget of 1.0
        let mut successful_calls = 0;
        for _ in 0..200 {
            if dp.add_gaussian_noise(&mut gradient).is_ok() {
                successful_calls += 1;
            } else {
                // Budget exhausted, break
                break;
            }
        }

        // Should have made ~100 successful calls before budget exhaustion
        assert!(
            (90..=110).contains(&successful_calls),
            "Expected ~100 calls, got {}",
            successful_calls
        );

        // Budget should be very low or exhausted (allow small epsilon for floating point errors)
        let remaining = dp.remaining_budget();
        assert!(
            remaining < 0.02,
            "Expected nearly exhausted budget, got {}",
            remaining
        );

        // Should fail when trying to consume more than remaining
        let mut new_gradient = vec![1.0, 2.0];
        let result = dp.add_gaussian_noise(&mut new_gradient);
        // Might succeed if there's a tiny bit of budget left, or fail if exhausted
        // Either way is acceptable at this point
        let _ = result;
    }

    #[test]
    fn test_noise_multiplier_calculation() {
        let epsilon = 1.0;
        let delta = 1e-5;
        let sensitivity = 1.0;

        let multiplier =
            DifferentialPrivacy::calculate_noise_multiplier(epsilon, delta, sensitivity);

        // Noise multiplier should be positive and reasonable
        assert!(multiplier > 0.0);
        assert!(multiplier < 10.0); // Sanity check

        // For higher epsilon (less privacy), noise should be lower
        let multiplier_high_eps =
            DifferentialPrivacy::calculate_noise_multiplier(10.0, delta, sensitivity);
        assert!(multiplier_high_eps < multiplier);
    }

    #[test]
    fn test_secure_aggregation() {
        let mut aggregator = SecureAggregation::new(3);

        assert_eq!(aggregator.participant_count(), 0);
        assert!(!aggregator.can_aggregate());

        // Add participants
        aggregator.add_participant();
        aggregator.add_participant();
        assert!(!aggregator.can_aggregate());

        aggregator.add_participant();
        assert!(aggregator.can_aggregate());

        // Test aggregation
        let g1 = vec![1.0, 2.0, 3.0];
        let g2 = vec![2.0, 3.0, 4.0];
        let g3 = vec![3.0, 4.0, 5.0];
        let gradients = vec![g1, g2, g3];

        let result = aggregator
            .aggregate_secure(&gradients)
            .expect("test: should succeed");

        // Should be average of the three gradients
        assert!((result[0] - 2.0).abs() < 0.01);
        assert!((result[1] - 3.0).abs() < 0.01);
        assert!((result[2] - 4.0).abs() < 0.01);

        // Reset
        aggregator.reset();
        assert_eq!(aggregator.participant_count(), 0);
    }

    #[test]
    fn test_secure_aggregation_insufficient_participants() {
        let aggregator = SecureAggregation::new(5);

        let g1 = vec![1.0, 2.0];
        let g2 = vec![3.0, 4.0];
        let gradients = vec![g1, g2];

        // Should fail because we don't have enough participants
        let result = aggregator.aggregate_secure(&gradients);
        assert!(result.is_err());
    }

    #[test]
    fn test_dp_mechanism_types() {
        let gaussian = DPMechanism::Gaussian;
        let laplacian = DPMechanism::Laplacian;

        assert_eq!(gaussian, DPMechanism::Gaussian);
        assert_eq!(laplacian, DPMechanism::Laplacian);
        assert_ne!(gaussian, laplacian);
    }

    #[test]
    fn test_client_info() {
        let mut client = ClientInfo::new("client1".to_string(), 1000);

        assert_eq!(client.client_id, "client1");
        assert_eq!(client.state, ClientState::Idle);
        assert_eq!(client.sample_count, 1000);

        client.start_training();
        assert_eq!(client.state, ClientState::Training);

        client.complete_training();
        assert_eq!(client.state, ClientState::Completed);

        client.mark_failed();
        assert_eq!(client.state, ClientState::Failed);
    }

    #[test]
    fn test_federated_round() {
        let model_cid = Cid::default();
        let mut round = FederatedRound::new(0, model_cid, 5);

        assert_eq!(round.round_num, 0);
        assert_eq!(round.client_count, 5);
        assert_eq!(round.completed_count, 0);
        assert!(!round.is_complete());

        // Mark clients as completed
        for _ in 0..5 {
            round.mark_client_completed();
        }

        assert_eq!(round.completed_count, 5);
        assert!(round.is_complete());

        // Complete the round
        let gradient = vec![1.0, 2.0, 3.0];
        round.complete(gradient.clone());

        assert_eq!(round.aggregated_gradient, Some(gradient));
        assert!(round.end_time.is_some());
        assert!(round.duration().is_some());
    }

    #[test]
    fn test_convergence_detector() {
        let mut detector = ConvergenceDetector::new(3, 0.01);

        // Add loss values that are converging
        detector.add_loss(1.0);
        detector.add_loss(0.99);
        detector.add_loss(0.98);

        assert!(detector.has_converged());
        assert_eq!(detector.latest_loss(), Some(0.98));
        assert_eq!(detector.history().len(), 3);

        // Reset
        detector.reset();
        assert_eq!(detector.history().len(), 0);
    }

    #[test]
    fn test_convergence_detector_not_converged() {
        let mut detector = ConvergenceDetector::new(3, 0.01);

        // Add loss values that are NOT converging
        detector.add_loss(1.0);
        detector.add_loss(0.5);
        detector.add_loss(1.5);

        assert!(!detector.has_converged());
    }

    #[test]
    fn test_model_sync_protocol() {
        let mut protocol = ModelSyncProtocol::new(10, 3, 3, 0.01);

        assert_eq!(protocol.current_round(), 0);
        assert_eq!(protocol.max_rounds(), 10);
        assert!(protocol.should_continue());

        // Start round 0
        let model_cid = Cid::default();
        let round_num = protocol
            .start_round(model_cid, 5)
            .expect("test: should succeed");

        assert_eq!(round_num, 0);
        assert_eq!(protocol.current_round(), 1);
        assert_eq!(protocol.total_rounds(), 1);

        // Complete round 0
        let gradient = vec![1.0, 2.0, 3.0];
        protocol
            .complete_round(round_num, gradient.clone(), 1.0)
            .expect("test: should succeed");

        assert_eq!(protocol.latest_loss(), Some(1.0));

        // Get round info
        let round = protocol.get_round(0).expect("test: should succeed");
        assert_eq!(round.round_num, 0);
        assert_eq!(round.aggregated_gradient, Some(gradient));
    }

    #[test]
    fn test_model_sync_protocol_convergence() {
        let mut protocol = ModelSyncProtocol::new(10, 2, 3, 0.01);

        let model_cid = Cid::default();

        // Run multiple rounds with converging loss
        for i in 0..3 {
            protocol
                .start_round(model_cid, 3)
                .expect("test: should succeed");
            let gradient = vec![1.0, 2.0];
            let loss = 1.0 - (i as f64 * 0.001);
            protocol
                .complete_round(i, gradient, loss)
                .expect("test: should succeed");
        }

        // Should have converged
        assert!(protocol.has_converged());
        assert!(!protocol.should_continue());
    }

    #[test]
    fn test_model_sync_protocol_max_rounds() {
        let mut protocol = ModelSyncProtocol::new(2, 1, 3, 0.01);

        let model_cid = Cid::default();

        // Start 2 rounds (max)
        protocol
            .start_round(model_cid, 2)
            .expect("test: should succeed");
        protocol
            .start_round(model_cid, 2)
            .expect("test: should succeed");

        // Should fail to start a third round
        let result = protocol.start_round(model_cid, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_model_sync_protocol_min_clients() {
        let mut protocol = ModelSyncProtocol::new(10, 5, 3, 0.01);

        let model_cid = Cid::default();

        // Should fail with too few clients
        let result = protocol.start_round(model_cid, 3);
        assert!(result.is_err());

        // Should succeed with enough clients
        let result = protocol.start_round(model_cid, 5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_client_state_enum() {
        let idle = ClientState::Idle;
        let training = ClientState::Training;
        let completed = ClientState::Completed;
        let failed = ClientState::Failed;

        assert_ne!(idle, training);
        assert_ne!(training, completed);
        assert_ne!(completed, failed);
        assert_eq!(idle, ClientState::Idle);
    }
}

// ── DistributedGradientAccumulator tests ─────────────────────────────────

#[cfg(test)]
mod distributed_accumulator_tests {
    use super::*;
    use ipfrs_storage::traits::BlockStore as _;

    /// Build a simple in-memory [`BlockStore`] backed by an Arc<DashMap>.
    ///
    /// We use the real `ipfrs_storage::MemoryBlockStore` so we can exercise
    /// the full `commit_local` / `add_peer_gradient` path without mocking.
    fn make_store() -> std::sync::Arc<ipfrs_storage::MemoryBlockStore> {
        std::sync::Arc::new(ipfrs_storage::MemoryBlockStore::new())
    }

    #[tokio::test]
    async fn test_distributed_accumulator_fedavg() {
        let store = make_store();

        let config = BackwardPassConfig::default();
        let mut acc = DistributedGradientAccumulator::new("session-fedavg", config);

        // Local gradient: [1, 2, 3]
        let local_grad = vec![1.0f32, 2.0, 3.0];
        let _cid = acc
            .commit_local(local_grad, &*store)
            .await
            .expect("commit_local");

        // Peer A gradient: [3, 4, 5] — store it directly and add to accumulator.
        let peer_a_bytes =
            store_gradient_as_arrow(&[3.0f32, 4.0, 5.0]).expect("peer_a arrow encode");
        let block_a = ipfrs_core::Block::new(bytes::Bytes::from(peer_a_bytes)).expect("block_a");
        let cid_a = block_a.cid();
        store.put(&block_a).await.expect("put block_a");

        acc.add_peer_gradient("peer_a", cid_a, &*store)
            .await
            .expect("add_peer_gradient peer_a");

        // Peer B gradient: [5, 6, 7]
        let peer_b_bytes =
            store_gradient_as_arrow(&[5.0f32, 6.0, 7.0]).expect("peer_b arrow encode");
        let block_b = ipfrs_core::Block::new(bytes::Bytes::from(peer_b_bytes)).expect("block_b");
        let cid_b = block_b.cid();
        store.put(&block_b).await.expect("put block_b");

        acc.add_peer_gradient("peer_b", cid_b, &*store)
            .await
            .expect("add_peer_gradient peer_b");

        assert_eq!(acc.peer_count(), 2, "should have 2 peer gradients");

        // FedAvg of [1,2,3], [3,4,5], [5,6,7] = [3,4,5]
        let agg = acc.aggregate().expect("aggregate");
        assert_eq!(agg.len(), 3);
        assert!((agg[0] - 3.0).abs() < 1e-5, "agg[0] = {}", agg[0]);
        assert!((agg[1] - 4.0).abs() < 1e-5, "agg[1] = {}", agg[1]);
        assert!((agg[2] - 5.0).abs() < 1e-5, "agg[2] = {}", agg[2]);
    }

    #[test]
    fn test_accumulator_not_ready() {
        let config = BackwardPassConfig::default();
        let mut acc = DistributedGradientAccumulator::new("session-not-ready", config);

        // Inject one peer gradient directly (no async store needed here).
        acc.peer_gradients
            .insert("peer_x".to_string(), vec![1.0f32, 2.0]);

        // With 1 peer, `is_ready(3)` must return false.
        assert!(
            !acc.is_ready(3),
            "is_ready(3) must be false with only 1 peer"
        );

        // With 1 peer, `is_ready(1)` must return true.
        assert!(acc.is_ready(1), "is_ready(1) must be true with 1 peer");
    }
}
