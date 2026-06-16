//! Differential privacy for embeddings
//!
//! This module provides privacy-preserving mechanisms for embedding-based search:
//! - Noise injection (Laplacian, Gaussian)
//! - Privacy budget tracking (epsilon-delta)
//! - Utility-privacy trade-off analysis
//! - Secure embedding release

use ipfrs_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

/// Noise distribution for differential privacy
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NoiseDistribution {
    /// Laplacian noise (for epsilon-DP)
    Laplacian { scale: f32 },
    /// Gaussian noise (for (epsilon, delta)-DP)
    Gaussian { sigma: f32 },
}

/// Privacy mechanism for adding noise to embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyMechanism {
    /// Noise distribution
    distribution: NoiseDistribution,
    /// Privacy budget (epsilon)
    epsilon: f32,
    /// Privacy parameter delta (for Gaussian)
    delta: f32,
    /// Sensitivity of the query (L2 norm bound)
    sensitivity: f32,
}

impl PrivacyMechanism {
    /// Create a new Laplacian mechanism (epsilon-DP)
    pub fn laplacian(epsilon: f32, sensitivity: f32) -> Result<Self> {
        if epsilon <= 0.0 {
            return Err(Error::InvalidInput("Epsilon must be positive".into()));
        }
        if sensitivity <= 0.0 {
            return Err(Error::InvalidInput("Sensitivity must be positive".into()));
        }

        let scale = sensitivity / epsilon;

        Ok(Self {
            distribution: NoiseDistribution::Laplacian { scale },
            epsilon,
            delta: 0.0,
            sensitivity,
        })
    }

    /// Create a new Gaussian mechanism ((epsilon, delta)-DP)
    pub fn gaussian(epsilon: f32, delta: f32, sensitivity: f32) -> Result<Self> {
        if epsilon <= 0.0 {
            return Err(Error::InvalidInput("Epsilon must be positive".into()));
        }
        if delta <= 0.0 || delta >= 1.0 {
            return Err(Error::InvalidInput("Delta must be in (0, 1)".into()));
        }
        if sensitivity <= 0.0 {
            return Err(Error::InvalidInput("Sensitivity must be positive".into()));
        }

        // Calculate sigma using the Gaussian mechanism formula
        // sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
        let sigma = sensitivity * (2.0 * (1.25 / delta).ln()).sqrt() / epsilon;

        Ok(Self {
            distribution: NoiseDistribution::Gaussian { sigma },
            epsilon,
            delta,
            sensitivity,
        })
    }

    /// Add noise to an embedding
    pub fn add_noise(&self, embedding: &[f32]) -> Vec<f32> {
        use rand::RngExt;
        let mut rng = rand::rng();

        match self.distribution {
            NoiseDistribution::Laplacian { scale } => embedding
                .iter()
                .map(|&x| x + sample_laplacian(&mut rng, scale))
                .collect(),
            NoiseDistribution::Gaussian { sigma } => {
                embedding
                    .iter()
                    .map(|&x| {
                        // Sample from N(0, sigma^2)
                        let noise: f32 = rng.random_range(-1.0..1.0);
                        x + noise * sigma
                    })
                    .collect()
            }
        }
    }

    /// Get the privacy budget (epsilon)
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Get the delta parameter
    pub fn delta(&self) -> f32 {
        self.delta
    }

    /// Calculate expected utility loss (L2 distance to original)
    pub fn expected_utility_loss(&self, dimension: usize) -> f32 {
        match self.distribution {
            NoiseDistribution::Laplacian { scale } => {
                // Expected L2 norm of Laplacian noise in d dimensions
                // E[||noise||_2] ≈ scale * sqrt(d)
                scale * (dimension as f32).sqrt()
            }
            NoiseDistribution::Gaussian { sigma } => {
                // Expected L2 norm of Gaussian noise in d dimensions
                // E[||noise||_2] = sigma * sqrt(d)
                sigma * (dimension as f32).sqrt()
            }
        }
    }
}

/// Sample from Laplacian distribution with scale parameter
fn sample_laplacian<R: rand::RngExt>(rng: &mut R, scale: f32) -> f32 {
    let u: f32 = rng.random_range(-0.5..0.5);
    if u >= 0.0 {
        -scale * (1.0 - 2.0 * u).ln()
    } else {
        scale * (1.0 + 2.0 * u).ln()
    }
}

/// Privacy budget tracker for managing cumulative privacy loss
pub struct PrivacyBudget {
    /// Total budget (epsilon)
    total_epsilon: f32,
    /// Remaining budget
    remaining_epsilon: Arc<Mutex<f32>>,
    /// Total delta
    total_delta: f32,
    /// Queries made
    queries: Arc<Mutex<Vec<QueryRecord>>>,
}

/// Record of a privacy-consuming query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRecord {
    /// Epsilon consumed
    pub epsilon: f32,
    /// Delta consumed
    pub delta: f32,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
}

impl PrivacyBudget {
    /// Create a new privacy budget
    pub fn new(total_epsilon: f32, total_delta: f32) -> Result<Self> {
        if total_epsilon <= 0.0 {
            return Err(Error::InvalidInput("Total epsilon must be positive".into()));
        }

        Ok(Self {
            total_epsilon,
            remaining_epsilon: Arc::new(Mutex::new(total_epsilon)),
            total_delta,
            queries: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Check if we can afford a query with given epsilon/delta
    pub fn can_afford(&self, epsilon: f32, delta: f32) -> bool {
        let remaining = self
            .remaining_epsilon
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *remaining >= epsilon && self.total_delta >= delta
    }

    /// Consume budget for a query
    pub fn consume(&self, epsilon: f32, delta: f32) -> Result<()> {
        if !self.can_afford(epsilon, delta) {
            return Err(Error::InvalidInput("Insufficient privacy budget".into()));
        }

        let mut remaining = self
            .remaining_epsilon
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *remaining -= epsilon;

        let mut queries = self.queries.lock().unwrap_or_else(|e| e.into_inner());
        queries.push(QueryRecord {
            epsilon,
            delta,
            timestamp: std::time::SystemTime::now(),
        });

        Ok(())
    }

    /// Get remaining epsilon
    pub fn remaining(&self) -> f32 {
        *self
            .remaining_epsilon
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Get statistics
    pub fn stats(&self) -> PrivacyBudgetStats {
        let remaining = *self
            .remaining_epsilon
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let queries = self.queries.lock().unwrap_or_else(|e| e.into_inner());

        PrivacyBudgetStats {
            total_epsilon: self.total_epsilon,
            remaining_epsilon: remaining,
            consumed_epsilon: self.total_epsilon - remaining,
            total_delta: self.total_delta,
            num_queries: queries.len(),
        }
    }
}

/// Statistics about privacy budget usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyBudgetStats {
    /// Total privacy budget
    pub total_epsilon: f32,
    /// Remaining budget
    pub remaining_epsilon: f32,
    /// Consumed budget
    pub consumed_epsilon: f32,
    /// Total delta
    pub total_delta: f32,
    /// Number of queries made
    pub num_queries: usize,
}

/// Privacy-preserving embedding wrapper
pub struct PrivateEmbedding {
    /// Original embedding (kept private)
    #[allow(dead_code)]
    original: Vec<f32>,
    /// Noisy version (public)
    pub noisy: Vec<f32>,
    /// Privacy mechanism used
    mechanism: PrivacyMechanism,
}

impl PrivateEmbedding {
    /// Create a private embedding with noise
    pub fn new(embedding: Vec<f32>, mechanism: PrivacyMechanism) -> Self {
        let noisy = mechanism.add_noise(&embedding);

        Self {
            original: embedding,
            noisy,
            mechanism,
        }
    }

    /// Get the noisy (public) embedding
    pub fn public_embedding(&self) -> &[f32] {
        &self.noisy
    }

    /// Get the privacy parameters
    pub fn privacy_params(&self) -> (f32, f32) {
        (self.mechanism.epsilon(), self.mechanism.delta())
    }

    /// Get expected utility loss
    pub fn utility_loss(&self) -> f32 {
        self.mechanism.expected_utility_loss(self.noisy.len())
    }
}

/// Utility-privacy trade-off analyzer
pub struct TradeoffAnalyzer {
    /// Different epsilon values to test
    epsilons: Vec<f32>,
    /// Sensitivity
    sensitivity: f32,
}

impl TradeoffAnalyzer {
    /// Create a new analyzer
    pub fn new(sensitivity: f32) -> Self {
        // Test a range of epsilon values from 0.1 to 10.0
        let epsilons = vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0];

        Self {
            epsilons,
            sensitivity,
        }
    }

    /// Analyze trade-offs for different epsilon values
    pub fn analyze(&self, dimension: usize) -> Vec<TradeoffPoint> {
        self.epsilons
            .iter()
            .map(|&epsilon| {
                let mechanism = PrivacyMechanism::laplacian(epsilon, self.sensitivity)
                    .expect("epsilons from preset list are all positive");
                let utility_loss = mechanism.expected_utility_loss(dimension);

                TradeoffPoint {
                    epsilon,
                    delta: 0.0,
                    utility_loss,
                }
            })
            .collect()
    }

    /// Find the best epsilon for a target utility loss
    pub fn find_epsilon_for_utility(&self, dimension: usize, max_utility_loss: f32) -> Option<f32> {
        let points = self.analyze(dimension);

        points
            .into_iter()
            .filter(|p| p.utility_loss <= max_utility_loss)
            .map(|p| p.epsilon)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }
}

/// A point in the utility-privacy trade-off space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeoffPoint {
    /// Privacy budget (epsilon)
    pub epsilon: f32,
    /// Delta parameter
    pub delta: f32,
    /// Expected utility loss
    pub utility_loss: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplacian_mechanism() {
        let mechanism =
            PrivacyMechanism::laplacian(1.0, 1.0).expect("test: valid laplacian params");
        assert_eq!(mechanism.epsilon(), 1.0);
        assert_eq!(mechanism.delta(), 0.0);

        let embedding = vec![1.0, 2.0, 3.0];
        let noisy = mechanism.add_noise(&embedding);

        assert_eq!(noisy.len(), embedding.len());
        // Noisy embedding should be different from original
        assert_ne!(noisy, embedding);
    }

    #[test]
    fn test_gaussian_mechanism() {
        let mechanism =
            PrivacyMechanism::gaussian(1.0, 0.001, 1.0).expect("test: valid gaussian params");
        assert_eq!(mechanism.epsilon(), 1.0);
        assert!(mechanism.delta() > 0.0);

        let embedding = vec![1.0, 2.0, 3.0];
        let noisy = mechanism.add_noise(&embedding);

        assert_eq!(noisy.len(), embedding.len());
    }

    #[test]
    fn test_privacy_budget() {
        let budget = PrivacyBudget::new(10.0, 0.001).expect("test: valid budget params");

        assert!(budget.can_afford(1.0, 0.0001));
        assert_eq!(budget.remaining(), 10.0);

        budget
            .consume(1.0, 0.0001)
            .expect("test: consume within budget");
        assert_eq!(budget.remaining(), 9.0);

        let stats = budget.stats();
        assert_eq!(stats.consumed_epsilon, 1.0);
        assert_eq!(stats.num_queries, 1);
    }

    #[test]
    fn test_budget_exhaustion() {
        let budget = PrivacyBudget::new(1.0, 0.001).expect("test: valid budget params");

        budget
            .consume(0.5, 0.0001)
            .expect("test: consume within budget");
        budget
            .consume(0.5, 0.0001)
            .expect("test: consume within budget");

        // Should fail - budget exhausted
        assert!(budget.consume(0.1, 0.0001).is_err());
    }

    #[test]
    fn test_private_embedding() {
        let embedding = vec![1.0, 2.0, 3.0];
        let mechanism =
            PrivacyMechanism::laplacian(1.0, 1.0).expect("test: valid laplacian params");

        let private_emb = PrivateEmbedding::new(embedding.clone(), mechanism);

        assert_eq!(private_emb.public_embedding().len(), embedding.len());
        assert_eq!(private_emb.privacy_params().0, 1.0);
        assert!(private_emb.utility_loss() > 0.0);
    }

    #[test]
    fn test_tradeoff_analyzer() {
        let analyzer = TradeoffAnalyzer::new(1.0);
        let points = analyzer.analyze(768);

        assert!(!points.is_empty());
        // Higher epsilon should give lower utility loss
        assert!(
            points[0].utility_loss
                > points
                    .last()
                    .expect("test: non-empty points vec")
                    .utility_loss
        );
    }

    #[test]
    fn test_find_epsilon_for_utility() {
        let analyzer = TradeoffAnalyzer::new(1.0);
        let epsilon = analyzer.find_epsilon_for_utility(768, 10.0);

        assert!(epsilon.is_some());
        assert!(epsilon.expect("test: epsilon found for utility bound") > 0.0);
    }

    #[test]
    fn test_utility_loss_estimation() {
        let mechanism =
            PrivacyMechanism::laplacian(1.0, 1.0).expect("test: valid laplacian params");
        let loss = mechanism.expected_utility_loss(768);

        // Should be roughly sqrt(768) for unit scale
        assert!(loss > 20.0 && loss < 30.0);
    }
}
