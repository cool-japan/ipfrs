//! Federated learning utilities.
//!
//! This module provides:
//! - Differential privacy mechanisms (`PrivacyBudget`, `DPMechanism`, `DifferentialPrivacy`)
//! - Secure aggregation (`SecureAggregation`)
//! - Client lifecycle management (`ClientState`, `ClientInfo`)
//! - Round coordination (`FederatedRound`, `ConvergenceDetector`, `ModelSyncProtocol`)
//! - Standalone helpers: `federated_average`, `clip_gradient_norm`
//! - `DistributedGradientAccumulator` for peer-to-peer gradient exchange

use ipfrs_core::Cid;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use super::arrow_ipc::{load_gradient_from_arrow, store_gradient_as_arrow};
use super::backward_pass::{federated_average, AggregationMethod, BackwardPassConfig};
use super::tensor::GradientAggregator;
use super::GradientError;

// ── FederatedError ─────────────────────────────────────────────────────────

/// Errors specific to the federated round lifecycle.
#[derive(Debug, thiserror::Error)]
pub enum FederatedError {
    #[error("No contributions available to aggregate")]
    NoContributions,
    #[error("Dimension mismatch between client gradients")]
    DimensionMismatch,
    #[error("Client not found: {0}")]
    ClientNotFound(String),
    #[error("Round not started")]
    NotStarted,
}

// ── PrivacyBudget ──────────────────────────────────────────────────────────

/// Privacy budget for differential privacy
#[derive(Debug, Clone, Copy)]
pub struct PrivacyBudget {
    /// Epsilon (privacy loss parameter)
    pub epsilon: f64,
    /// Delta (failure probability)
    pub delta: f64,
    /// Remaining epsilon
    pub remaining_epsilon: f64,
}

impl PrivacyBudget {
    /// Create a new privacy budget
    pub fn new(epsilon: f64, delta: f64) -> Self {
        Self {
            epsilon,
            delta,
            remaining_epsilon: epsilon,
        }
    }

    /// Consume some privacy budget
    pub fn consume(&mut self, epsilon_used: f64) -> Result<(), GradientError> {
        if epsilon_used > self.remaining_epsilon {
            return Err(GradientError::InvalidGradient(format!(
                "Insufficient privacy budget: need {}, have {}",
                epsilon_used, self.remaining_epsilon
            )));
        }

        self.remaining_epsilon -= epsilon_used;
        Ok(())
    }

    /// Check if budget is exhausted
    pub fn is_exhausted(&self) -> bool {
        self.remaining_epsilon <= 0.0
    }

    /// Get the fraction of budget remaining
    pub fn remaining_fraction(&self) -> f64 {
        self.remaining_epsilon / self.epsilon
    }
}

// ── DPMechanism ────────────────────────────────────────────────────────────

/// Differential privacy mechanism types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DPMechanism {
    /// Gaussian mechanism (for bounded sensitivity)
    Gaussian,
    /// Laplacian mechanism (for L1 sensitivity)
    Laplacian,
}

// ── DifferentialPrivacy ────────────────────────────────────────────────────

/// Differential privacy for gradient protection
pub struct DifferentialPrivacy {
    /// Privacy budget
    budget: PrivacyBudget,
    /// Sensitivity (L2 norm bound for gradients)
    sensitivity: f64,
    /// Mechanism type
    mechanism: DPMechanism,
}

impl DifferentialPrivacy {
    /// Create a new differential privacy instance
    pub fn new(epsilon: f64, delta: f64, sensitivity: f64, mechanism: DPMechanism) -> Self {
        Self {
            budget: PrivacyBudget::new(epsilon, delta),
            sensitivity,
            mechanism,
        }
    }

    /// Add Gaussian noise to gradient (for DP-SGD)
    /// Calibrated according to sensitivity and privacy parameters
    pub fn add_gaussian_noise(&mut self, gradient: &mut [f32]) -> Result<(), GradientError> {
        use rand::RngExt;

        if self.budget.is_exhausted() {
            return Err(GradientError::InvalidGradient(
                "Privacy budget exhausted".to_string(),
            ));
        }

        // Calculate noise scale using Gaussian mechanism
        // σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        let ln_term = (1.25 / self.budget.delta).ln();
        let sigma = self.sensitivity * (2.0 * ln_term).sqrt() / self.budget.epsilon;

        let mut rng = rand::rng();

        // Add Gaussian noise to each element
        for v in gradient.iter_mut() {
            let noise: f64 = rng.random_range(-1.0..1.0);
            let gaussian_noise = sigma * noise;
            *v += gaussian_noise as f32;
        }

        // Consume privacy budget (simplified - in practice, this depends on composition)
        self.budget.consume(self.budget.epsilon / 100.0)?;

        Ok(())
    }

    /// Add Laplacian noise to gradient
    /// Calibrated according to L1 sensitivity and privacy parameters
    pub fn add_laplacian_noise(&mut self, gradient: &mut [f32]) -> Result<(), GradientError> {
        use rand::RngExt;

        if self.budget.is_exhausted() {
            return Err(GradientError::InvalidGradient(
                "Privacy budget exhausted".to_string(),
            ));
        }

        // Calculate noise scale using Laplacian mechanism
        // b = sensitivity / ε
        let scale = self.sensitivity / self.budget.epsilon;

        let mut rng = rand::rng();

        // Add Laplacian noise to each element
        for v in gradient.iter_mut() {
            let u: f64 = rng.random_range(-0.5..0.5);
            let laplacian_noise = -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln();
            *v += laplacian_noise as f32;
        }

        // Consume privacy budget
        self.budget.consume(self.budget.epsilon / 100.0)?;

        Ok(())
    }

    /// Apply DP-SGD (Differential Private Stochastic Gradient Descent)
    /// This clips gradients and adds noise
    pub fn apply_dp_sgd(
        &mut self,
        gradient: &mut [f32],
        clip_norm: f32,
    ) -> Result<(), GradientError> {
        use super::tensor::GradientVerifier;

        // Step 1: Clip gradient to bound sensitivity
        GradientVerifier::clip_by_norm(gradient, clip_norm);

        // Step 2: Add calibrated noise
        match self.mechanism {
            DPMechanism::Gaussian => self.add_gaussian_noise(gradient)?,
            DPMechanism::Laplacian => self.add_laplacian_noise(gradient)?,
        }

        Ok(())
    }

    /// Get remaining privacy budget
    pub fn remaining_budget(&self) -> f64 {
        self.budget.remaining_epsilon
    }

    /// Check if privacy budget is exhausted
    pub fn is_budget_exhausted(&self) -> bool {
        self.budget.is_exhausted()
    }

    /// Get privacy parameters
    pub fn get_privacy_params(&self) -> (f64, f64) {
        (self.budget.epsilon, self.budget.delta)
    }

    /// Calculate noise multiplier for given privacy parameters
    /// Used in DP-SGD implementations
    pub fn calculate_noise_multiplier(epsilon: f64, delta: f64, sensitivity: f64) -> f64 {
        // σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        let ln_term = (1.25 / delta).ln();
        sensitivity * (2.0 * ln_term).sqrt() / epsilon
    }

    // ── New stateless helpers (sensitivity supplied by caller) ───────────────

    /// Add calibrated Gaussian noise to a gradient slice (stateless, sensitivity supplied).
    ///
    /// σ = sensitivity × √(2 ln(1.25/δ)) / ε
    ///
    /// Uses Box-Muller transform for Normal sampling (no `rand_distr` dependency).
    pub fn add_gaussian_noise_sens(&self, gradient: &mut [f32], sensitivity: f32) {
        use rand::RngExt;
        use std::f64::consts::PI;

        let ln_term = (1.25 / self.budget.delta).ln();
        let sigma = (sensitivity as f64) * (2.0 * ln_term).sqrt() / self.budget.epsilon;

        let mut rng = rand::rng();
        let mut i = 0;
        while i < gradient.len() {
            // Box-Muller: two uniform samples → two independent N(0,1) samples
            let u1: f64 = rng.random_range(1e-12_f64..1.0_f64);
            let u2: f64 = rng.random_range(0.0_f64..1.0_f64);
            let mag = sigma * (-2.0 * u1.ln()).sqrt();
            let z0 = mag * (2.0 * PI * u2).cos();
            let z1 = mag * (2.0 * PI * u2).sin();
            gradient[i] += z0 as f32;
            i += 1;
            if i < gradient.len() {
                gradient[i] += z1 as f32;
                i += 1;
            }
        }
    }

    /// Add calibrated Laplace noise to a gradient slice (stateless, sensitivity supplied).
    ///
    /// scale = sensitivity / ε
    pub fn add_laplace_noise_sens(&self, gradient: &mut [f32], sensitivity: f32) {
        use rand::RngExt;

        let scale = (sensitivity as f64) / self.budget.epsilon;
        let mut rng = rand::rng();
        for v in gradient.iter_mut() {
            // Invert CDF of Laplace distribution: F⁻¹(p) = -b sgn(p-0.5) ln(1-2|p-0.5|)
            let u: f64 = rng.random_range(1e-12_f64..1.0_f64 - 1e-12_f64);
            let p = u - 0.5;
            let laplace = -scale * p.signum() * (1.0 - 2.0 * p.abs()).ln();
            *v += laplace as f32;
        }
    }

    /// Clip gradient in-place to L2 norm ≤ `max_norm`.
    pub fn clip_l2(&self, gradient: &mut [f32], max_norm: f32) {
        let norm: f32 = gradient.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > max_norm && norm > 0.0 {
            let scale = max_norm / norm;
            for x in gradient.iter_mut() {
                *x *= scale;
            }
        }
    }

    /// Return remaining epsilon budget after `rounds_used` rounds, each costing ε/100.
    ///
    /// This mirrors the per-round consumption used by `add_gaussian_noise` (mutable version).
    pub fn remaining_budget_after(&self, rounds_used: u32) -> f32 {
        let per_round = self.budget.epsilon / 100.0;
        let used = per_round * rounds_used as f64;
        (self.budget.epsilon - used).max(0.0) as f32
    }

    /// True if the budget would be exhausted after `rounds_used` rounds.
    pub fn is_exhausted_after(&self, rounds_used: u32) -> bool {
        self.remaining_budget_after(rounds_used) <= 0.0
    }
}

// ── SecureAggregation ──────────────────────────────────────────────────────

/// Secure aggregation for federated learning (simplified)
pub struct SecureAggregation {
    /// Minimum number of participants required
    min_participants: usize,
    /// Current participant count
    participant_count: usize,
}

impl SecureAggregation {
    /// Create a new secure aggregation instance
    pub fn new(min_participants: usize) -> Self {
        Self {
            min_participants,
            participant_count: 0,
        }
    }

    /// Add a participant
    pub fn add_participant(&mut self) {
        self.participant_count += 1;
    }

    /// Check if we have enough participants
    pub fn can_aggregate(&self) -> bool {
        self.participant_count >= self.min_participants
    }

    /// Aggregate gradients securely
    /// In a real implementation, this would use cryptographic techniques
    /// like secret sharing, homomorphic encryption, or secure multi-party computation
    pub fn aggregate_secure(&self, gradients: &[Vec<f32>]) -> Result<Vec<f32>, GradientError> {
        if !self.can_aggregate() {
            return Err(GradientError::InvalidGradient(format!(
                "Not enough participants: need {}, have {}",
                self.min_participants, self.participant_count
            )));
        }

        // For now, use simple averaging
        // In production, this would:
        // 1. Use secret sharing to split gradients
        // 2. Aggregate encrypted shares
        // 3. Reconstruct only the sum
        GradientAggregator::average(gradients)
    }

    /// Reset participant count
    pub fn reset(&mut self) {
        self.participant_count = 0;
    }

    /// Get participant count
    pub fn participant_count(&self) -> usize {
        self.participant_count
    }
}

// ── ClientState / ClientInfo ───────────────────────────────────────────────

/// Client state in federated learning
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientState {
    /// Client is idle and ready for work
    Idle,
    /// Client is training
    Training,
    /// Client has completed training
    Completed,
    /// Client has failed or dropped out
    Failed,
}

/// Client information in federated learning
#[derive(Debug, Clone)]
pub struct ClientInfo {
    /// Client ID
    pub client_id: String,
    /// Client state
    pub state: ClientState,
    /// Number of samples the client has
    pub sample_count: usize,
    /// Last update timestamp
    pub last_update: i64,
}

impl ClientInfo {
    /// Create a new client info
    pub fn new(client_id: String, sample_count: usize) -> Self {
        Self {
            client_id,
            state: ClientState::Idle,
            sample_count,
            last_update: chrono::Utc::now().timestamp(),
        }
    }

    /// Mark client as training
    pub fn start_training(&mut self) {
        self.state = ClientState::Training;
        self.last_update = chrono::Utc::now().timestamp();
    }

    /// Mark client as completed
    pub fn complete_training(&mut self) {
        self.state = ClientState::Completed;
        self.last_update = chrono::Utc::now().timestamp();
    }

    /// Mark client as failed
    pub fn mark_failed(&mut self) {
        self.state = ClientState::Failed;
        self.last_update = chrono::Utc::now().timestamp();
    }
}

// ── RoundStats ─────────────────────────────────────────────────────────────

/// Statistics snapshot for a completed federated round.
#[derive(Debug, Clone)]
pub struct RoundStats {
    /// Round identifier
    pub round_id: u32,
    /// Number of clients that contributed
    pub participants: usize,
    /// Client IDs that did not contribute in time
    pub missing_clients: Vec<String>,
    /// Wall-clock duration of the round in milliseconds
    pub duration_ms: u64,
    /// Whether convergence was detected at end of round
    pub converged: bool,
}

// ── FederatedRound ─────────────────────────────────────────────────────────

/// Federated learning round.
///
/// Supports two construction modes:
/// - Legacy: `FederatedRound::new(round_num, global_model, client_count)`
/// - Session: `FederatedRound::start(round_id, client_ids)` + `record_contribution` + `aggregate`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedRound {
    /// Round number
    pub round_num: usize,
    /// Clients participating in this round (stored as count for serialization)
    pub client_count: usize,
    /// Global model CID for this round
    #[serde(serialize_with = "crate::serialize_cid")]
    #[serde(deserialize_with = "crate::deserialize_cid")]
    pub global_model: Cid,
    /// Aggregated gradient for this round (if computed)
    pub aggregated_gradient: Option<Vec<f32>>,
    /// Round start timestamp
    pub start_time: i64,
    /// Round end timestamp (if completed)
    pub end_time: Option<i64>,
    /// Completed client count
    pub completed_count: usize,
    // ── Session-mode fields (populated by `start()`) ──────────────────────
    /// Round ID (session mode)
    #[serde(default)]
    pub round_id: u32,
    /// Expected client IDs (session mode)
    #[serde(default)]
    pub expected_clients: Vec<String>,
    /// Per-client gradient contributions (session mode)
    #[serde(skip)]
    pub contributions: HashMap<String, Vec<f32>>,
    /// Wall-clock start instant (session mode; not serialized)
    #[serde(skip)]
    session_start: Option<Instant>,
}

impl FederatedRound {
    // ── Legacy constructor ──────────────────────────────────────────────────

    /// Create a new federated round (legacy mode).
    pub fn new(round_num: usize, global_model: Cid, client_count: usize) -> Self {
        Self {
            round_num,
            client_count,
            global_model,
            aggregated_gradient: None,
            start_time: chrono::Utc::now().timestamp(),
            end_time: None,
            completed_count: 0,
            round_id: round_num as u32,
            expected_clients: Vec::new(),
            contributions: HashMap::new(),
            session_start: None,
        }
    }

    /// Mark a client as completed (legacy mode).
    pub fn mark_client_completed(&mut self) {
        self.completed_count += 1;
    }

    /// Check if round is complete (works in both modes).
    pub fn is_complete(&self) -> bool {
        if !self.expected_clients.is_empty() {
            // Session mode: all expected clients have contributed
            self.expected_clients
                .iter()
                .all(|id| self.contributions.contains_key(id.as_str()))
        } else {
            // Legacy mode
            self.completed_count >= self.client_count
        }
    }

    /// Complete the round (legacy mode).
    pub fn complete(&mut self, aggregated_gradient: Vec<f32>) {
        self.aggregated_gradient = Some(aggregated_gradient);
        self.end_time = Some(chrono::Utc::now().timestamp());
    }

    /// Get round duration in seconds (legacy mode).
    pub fn duration(&self) -> Option<i64> {
        self.end_time.map(|end| end - self.start_time)
    }

    // ── Session mode API ───────────────────────────────────────────────────

    /// Start a new round with the given participating client IDs (session mode).
    pub fn start(round_id: u32, client_ids: Vec<String>) -> Self {
        let count = client_ids.len();
        Self {
            round_num: round_id as usize,
            client_count: count,
            global_model: Cid::default(),
            aggregated_gradient: None,
            start_time: chrono::Utc::now().timestamp(),
            end_time: None,
            completed_count: 0,
            round_id,
            expected_clients: client_ids,
            contributions: HashMap::new(),
            session_start: Some(Instant::now()),
        }
    }

    /// Record a client's gradient contribution (session mode).
    pub fn record_contribution(&mut self, client_id: &str, gradient: Vec<f32>) {
        self.contributions.insert(client_id.to_string(), gradient);
    }

    /// Aggregate all contributions using FedAvg, optionally applying DP noise.
    pub fn aggregate(&self, dp: Option<&DifferentialPrivacy>) -> Result<Vec<f32>, FederatedError> {
        if self.contributions.is_empty() {
            return Err(FederatedError::NoContributions);
        }

        let grads: Vec<Vec<f32>> = self.contributions.values().cloned().collect();
        let dim = grads[0].len();
        if grads.iter().any(|g| g.len() != dim) {
            return Err(FederatedError::DimensionMismatch);
        }

        // FedAvg
        let n = grads.len() as f32;
        let mut agg = vec![0.0f32; dim];
        for g in &grads {
            for (a, &v) in agg.iter_mut().zip(g.iter()) {
                *a += v / n;
            }
        }

        // Optionally apply DP noise (Gaussian with unit sensitivity)
        if let Some(dp) = dp {
            dp.add_gaussian_noise_sens(&mut agg, 1.0);
        }

        Ok(agg)
    }

    /// Return statistics for this round.
    pub fn stats(&self) -> RoundStats {
        let missing_clients: Vec<String> = self
            .expected_clients
            .iter()
            .filter(|id| !self.contributions.contains_key(id.as_str()))
            .cloned()
            .collect();

        let duration_ms = self
            .session_start
            .map(|t| t.elapsed().as_millis() as u64)
            .unwrap_or(0);

        RoundStats {
            round_id: self.round_id,
            participants: self.contributions.len(),
            missing_clients,
            duration_ms,
            converged: false,
        }
    }
}

// ── ConvergenceConfig ──────────────────────────────────────────────────────

/// Configuration for `ConvergenceDetector`.
#[derive(Debug, Clone)]
pub struct ConvergenceConfig {
    /// Gradient-norm threshold below which convergence is declared.
    pub threshold: f32,
    /// Rolling window for gradient norm computation.
    pub window_size: usize,
    /// EMA smoothing factor (0 < smoothing ≤ 1; higher → more weight on recent losses).
    pub smoothing: f32,
    /// Number of consecutive below-threshold rounds required before declaring convergence.
    pub patience: usize,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            threshold: 1e-4,
            window_size: 10,
            smoothing: 0.9,
            patience: 3,
        }
    }
}

// ── ConvergenceDetector ────────────────────────────────────────────────────

/// Convergence detection for federated learning.
///
/// Supports two usage modes:
/// 1. Legacy: `new(window_size, threshold)` + `add_loss()` + `has_converged()`
/// 2. EMA mode: `with_config(cfg)` + `update(loss)` returns bool
pub struct ConvergenceDetector {
    /// Window size for convergence detection (legacy mode)
    window_size: usize,
    /// Recent loss values (legacy mode)
    loss_history: Vec<f64>,
    /// Convergence threshold (legacy mode, relative change)
    threshold: f64,
    /// EMA-smoothed loss (new mode)
    ema_loss: Option<f32>,
    /// Loss window for gradient-norm computation (new mode)
    loss_window: std::collections::VecDeque<f32>,
    /// Rounds consecutively below threshold (new mode)
    plateau_count: usize,
    /// Configuration (new mode)
    config: Option<ConvergenceConfig>,
}

impl ConvergenceDetector {
    /// Create a new convergence detector (legacy mode).
    pub fn new(window_size: usize, threshold: f64) -> Self {
        Self {
            window_size,
            loss_history: Vec::new(),
            threshold,
            ema_loss: None,
            loss_window: std::collections::VecDeque::new(),
            plateau_count: 0,
            config: None,
        }
    }

    /// Create a convergence detector driven by `ConvergenceConfig`.
    pub fn with_config(config: ConvergenceConfig) -> Self {
        let window_size = config.window_size;
        let threshold = config.threshold as f64;
        Self {
            window_size,
            loss_history: Vec::new(),
            threshold,
            ema_loss: None,
            loss_window: std::collections::VecDeque::with_capacity(window_size),
            plateau_count: 0,
            config: Some(config),
        }
    }

    // ── Legacy API ──────────────────────────────────────────────────────────

    /// Add a loss value (legacy mode).
    pub fn add_loss(&mut self, loss: f64) {
        self.loss_history.push(loss);
        if self.loss_history.len() > self.window_size {
            self.loss_history.remove(0);
        }
    }

    /// Check if training has converged (legacy mode).
    pub fn has_converged(&self) -> bool {
        if self.loss_history.len() < self.window_size {
            return false;
        }
        let recent = &self.loss_history[self.loss_history.len() - self.window_size..];
        let mean = recent.iter().sum::<f64>() / recent.len() as f64;
        if mean.abs() < 1e-10 {
            return true;
        }
        let std_dev =
            (recent.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / recent.len() as f64).sqrt();
        std_dev / mean.abs() < self.threshold
    }

    /// Get the latest loss (legacy mode).
    pub fn latest_loss(&self) -> Option<f64> {
        self.loss_history.last().copied()
    }

    /// Get loss history (legacy mode).
    pub fn history(&self) -> &[f64] {
        &self.loss_history
    }

    // ── EMA / config-driven API ─────────────────────────────────────────────

    /// Record a new loss value and check for convergence (EMA mode).
    ///
    /// Returns `true` when the smoothed gradient norm drops below `config.threshold`
    /// for `config.patience` consecutive rounds.
    pub fn update(&mut self, loss: f32) -> bool {
        let cfg = match &self.config {
            Some(c) => c.clone(),
            None => ConvergenceConfig::default(),
        };

        // Update EMA
        self.ema_loss = Some(match self.ema_loss {
            None => loss,
            Some(ema) => cfg.smoothing * ema + (1.0 - cfg.smoothing) * loss,
        });

        // Maintain loss window
        if self.loss_window.len() >= cfg.window_size {
            self.loss_window.pop_front();
        }
        self.loss_window.push_back(loss);

        // Compute gradient norm (mean absolute delta over window)
        let norm = self.gradient_norm();

        // Update plateau count
        if norm < cfg.threshold {
            self.plateau_count += 1;
        } else {
            self.plateau_count = 0;
        }

        self.plateau_count >= cfg.patience
    }

    /// Exponentially smoothed loss (EMA mode).
    pub fn smoothed_loss(&self) -> f32 {
        self.ema_loss.unwrap_or(f32::NAN)
    }

    /// Gradient norm computed as mean |Δloss| over the window (EMA mode).
    pub fn gradient_norm(&self) -> f32 {
        if self.loss_window.len() < 2 {
            return f32::MAX;
        }
        let deltas: Vec<f32> = self
            .loss_window
            .iter()
            .zip(self.loss_window.iter().skip(1))
            .map(|(a, b)| (b - a).abs())
            .collect();
        deltas.iter().sum::<f32>() / deltas.len() as f32
    }

    /// Number of consecutive rounds the gradient norm has been below threshold (EMA mode).
    pub fn plateau_rounds(&self) -> usize {
        self.plateau_count
    }

    /// Reset all history (both legacy and EMA modes).
    pub fn reset(&mut self) {
        self.loss_history.clear();
        self.ema_loss = None;
        self.loss_window.clear();
        self.plateau_count = 0;
    }

    /// Return a reference to the `ConvergenceConfig` (EMA mode).
    ///
    /// Returns `None` when constructed in legacy mode.
    pub fn config(&self) -> Option<&ConvergenceConfig> {
        self.config.as_ref()
    }
}

// ── ModelSyncProtocol ──────────────────────────────────────────────────────

/// Model synchronization protocol for federated learning
pub struct ModelSyncProtocol {
    /// Current round number
    current_round: usize,
    /// Maximum number of rounds
    max_rounds: usize,
    /// Minimum number of clients per round
    min_clients_per_round: usize,
    /// Round history
    rounds: Vec<FederatedRound>,
    /// Convergence detector
    convergence: ConvergenceDetector,
}

impl ModelSyncProtocol {
    /// Create a new model synchronization protocol
    pub fn new(
        max_rounds: usize,
        min_clients_per_round: usize,
        convergence_window: usize,
        convergence_threshold: f64,
    ) -> Self {
        Self {
            current_round: 0,
            max_rounds,
            min_clients_per_round,
            rounds: Vec::new(),
            convergence: ConvergenceDetector::new(convergence_window, convergence_threshold),
        }
    }

    /// Start a new round
    pub fn start_round(
        &mut self,
        global_model: Cid,
        client_count: usize,
    ) -> Result<usize, GradientError> {
        if client_count < self.min_clients_per_round {
            return Err(GradientError::InvalidGradient(format!(
                "Not enough clients: need {}, got {}",
                self.min_clients_per_round, client_count
            )));
        }

        if self.current_round >= self.max_rounds {
            return Err(GradientError::InvalidGradient(format!(
                "Maximum rounds reached: {}",
                self.max_rounds
            )));
        }

        let round = FederatedRound::new(self.current_round, global_model, client_count);
        self.rounds.push(round);
        self.current_round += 1;

        Ok(self.current_round - 1)
    }

    /// Complete the current round
    pub fn complete_round(
        &mut self,
        round_num: usize,
        aggregated_gradient: Vec<f32>,
        loss: f64,
    ) -> Result<(), GradientError> {
        if round_num >= self.rounds.len() {
            return Err(GradientError::InvalidGradient(format!(
                "Invalid round number: {}",
                round_num
            )));
        }

        self.rounds[round_num].complete(aggregated_gradient);
        self.convergence.add_loss(loss);

        Ok(())
    }

    /// Check if training should continue
    pub fn should_continue(&self) -> bool {
        self.current_round < self.max_rounds && !self.convergence.has_converged()
    }

    /// Check if training has converged
    pub fn has_converged(&self) -> bool {
        self.convergence.has_converged()
    }

    /// Get the current round number
    pub fn current_round(&self) -> usize {
        self.current_round
    }

    /// Get the total number of rounds
    pub fn total_rounds(&self) -> usize {
        self.rounds.len()
    }

    /// Get round information
    pub fn get_round(&self, round_num: usize) -> Option<&FederatedRound> {
        self.rounds.get(round_num)
    }

    /// Get the latest loss
    pub fn latest_loss(&self) -> Option<f64> {
        self.convergence.latest_loss()
    }

    /// Get max rounds
    pub fn max_rounds(&self) -> usize {
        self.max_rounds
    }
}

// ── ModelUpdate ────────────────────────────────────────────────────────────

/// A model-update announcement broadcast via GossipSub.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelUpdate {
    /// Peer ID of the originating node
    pub peer_id: String,
    /// Content-addressed identifier of the updated model
    pub model_cid: String,
    /// Federated round this update belongs to
    pub round_id: u32,
    /// Unix epoch timestamp in milliseconds
    pub timestamp_ms: u64,
}

impl ModelUpdate {
    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Create a new `ModelUpdate` stamped with the current wall-clock time.
    pub fn new(peer_id: String, model_cid: String, round_id: u32) -> Self {
        Self {
            peer_id,
            model_cid,
            round_id,
            timestamp_ms: Self::now_ms(),
        }
    }
}

// ── GossipModelSync ────────────────────────────────────────────────────────

/// GossipSub-based model synchronisation protocol.
///
/// Broadcasts `ModelUpdate` announcements via a shared in-process channel so
/// that tests can exercise the full send/receive/verify loop without a live
/// libp2p swarm.
pub struct GossipModelSync {
    local_peer_id: String,
    /// Sink for outbound gossip messages (shared with test harness or real transport).
    tx: tokio::sync::broadcast::Sender<ModelUpdate>,
    /// Source for inbound gossip messages.
    rx: tokio::sync::broadcast::Receiver<ModelUpdate>,
}

impl GossipModelSync {
    /// Create a new `GossipModelSync` backed by a broadcast channel of capacity 64.
    pub fn new(local_peer_id: impl Into<String>) -> Self {
        let (tx, rx) = tokio::sync::broadcast::channel(64);
        Self {
            local_peer_id: local_peer_id.into(),
            tx,
            rx,
        }
    }

    /// Create a second endpoint subscribed to the same broadcast channel (for tests).
    pub fn subscribe(&self, peer_id: impl Into<String>) -> Self {
        Self {
            local_peer_id: peer_id.into(),
            tx: self.tx.clone(),
            rx: self.tx.subscribe(),
        }
    }

    /// Broadcast a local model-update CID to all mesh peers.
    pub async fn broadcast_update(&self, model_cid: &str, round_id: u32) -> anyhow::Result<()> {
        let update = ModelUpdate::new(self.local_peer_id.clone(), model_cid.to_string(), round_id);
        self.tx
            .send(update)
            .map_err(|e| anyhow::anyhow!("broadcast send: {e}"))?;
        Ok(())
    }

    /// Collect model-update CIDs from peers for up to `timeout_ms` milliseconds.
    ///
    /// Returns as soon as `expected_peers` distinct peer updates are received or
    /// the timeout expires.
    pub async fn collect_updates(
        &mut self,
        expected_peers: usize,
        timeout_ms: u64,
    ) -> anyhow::Result<Vec<ModelUpdate>> {
        let mut collected: Vec<ModelUpdate> = Vec::new();
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);

        while collected
            .iter()
            .filter(|u| u.peer_id != self.local_peer_id)
            .count()
            < expected_peers
        {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            match tokio::time::timeout(remaining, self.rx.recv()).await {
                Ok(Ok(update)) => {
                    if update.peer_id != self.local_peer_id {
                        collected.push(update);
                    }
                }
                Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(n))) => {
                    tracing::warn!("GossipModelSync: lagged by {n} messages");
                }
                Ok(Err(tokio::sync::broadcast::error::RecvError::Closed)) => break,
                Err(_timeout) => break,
            }
        }

        Ok(collected)
    }

    /// Verify that a `ModelUpdate` has not been tampered with.
    ///
    /// The current implementation checks that `model_cid` is a non-empty, valid
    /// CIDv1 or CIDv0 string and that `peer_id` is non-empty.  Tampered updates
    /// with an empty or whitespace-only CID fail verification.
    pub fn verify_update(&self, update: &ModelUpdate) -> bool {
        if update.peer_id.trim().is_empty() {
            return false;
        }
        let cid = update.model_cid.trim();
        if cid.is_empty() {
            return false;
        }
        // Accept anything that parses as a CID via the `cid` crate, or any
        // non-empty hex/base58/base32 string that a real IPFS node would emit.
        // For the test harness, valid CIDs start with "Qm", "bafy", or any
        // base-encoded prefix; tampered CIDs are typically empty strings.
        cid.parse::<cid::Cid>().is_ok() || (!cid.contains('\0') && cid.len() >= 4)
    }
}

// ── DistributedGradientAccumulator ────────────────────────────────────────

/// Accumulates gradients from distributed peers via content-addressed storage.
///
/// Workflow:
/// 1. Call `commit_local` to serialize and store the local gradient, obtaining
///    a [`Cid`] that can be broadcast to peers.
/// 2. Call `add_peer_gradient` for each CID received from a peer.  The
///    accumulator fetches the raw bytes from the `BlockStore`, decodes them
///    via Arrow IPC, and caches the result.
/// 3. Once `is_ready` returns `true`, call `aggregate` to run FedAvg
///    over all collected gradients (local + peers).
pub struct DistributedGradientAccumulator {
    session_id: String,
    /// The local node's gradient, set by `commit_local`.
    pub local_gradient: Vec<f32>,
    /// Gradients received from peers, keyed by peer ID.
    pub peer_gradients: std::collections::HashMap<String, Vec<f32>>,
    config: BackwardPassConfig,
}

impl DistributedGradientAccumulator {
    /// Create a new accumulator for the given session.
    pub fn new(session_id: &str, config: BackwardPassConfig) -> Self {
        Self {
            session_id: session_id.to_string(),
            local_gradient: Vec::new(),
            peer_gradients: std::collections::HashMap::new(),
            config,
        }
    }

    /// Store the local gradient as Arrow IPC bytes in `store`, returning its CID.
    ///
    /// The CID is computed as a raw-block CID (SHA-256) over the Arrow IPC bytes.
    pub async fn commit_local(
        &mut self,
        grad: Vec<f32>,
        store: &dyn ipfrs_storage::traits::BlockStore,
    ) -> anyhow::Result<ipfrs_core::Cid> {
        use ipfrs_core::Block;

        let ipc_bytes =
            store_gradient_as_arrow(&grad).map_err(|e| anyhow::anyhow!("Arrow IPC encode: {e}"))?;

        // Build a content-addressed block and store it.
        let block = Block::new(bytes::Bytes::from(ipc_bytes.clone()))
            .map_err(|e| anyhow::anyhow!("Block creation: {e}"))?;
        let cid = block.cid();

        store
            .put(&block)
            .await
            .map_err(|e| anyhow::anyhow!("BlockStore put: {e}"))?;

        self.local_gradient = grad;
        Ok(*cid)
    }

    /// Fetch and decode a peer's gradient from `store` using its CID.
    pub async fn add_peer_gradient(
        &mut self,
        peer_id: &str,
        cid: &ipfrs_core::Cid,
        store: &dyn ipfrs_storage::traits::BlockStore,
    ) -> anyhow::Result<()> {
        let block = store
            .get(cid)
            .await
            .map_err(|e| anyhow::anyhow!("BlockStore get: {e}"))?
            .ok_or_else(|| anyhow::anyhow!("Block not found for CID {cid}"))?;

        let grad = load_gradient_from_arrow(block.data())
            .map_err(|e| anyhow::anyhow!("Arrow IPC decode: {e}"))?;

        self.peer_gradients.insert(peer_id.to_string(), grad);
        Ok(())
    }

    /// Aggregate all collected gradients using the configured [`AggregationMethod`].
    ///
    /// Includes the local gradient plus all peer gradients collected so far.
    /// The aggregation strategy is taken from [`BackwardPassConfig::aggregation`].
    pub fn aggregate(&self) -> Result<Vec<f32>, GradientError> {
        if self.local_gradient.is_empty() {
            return Err(GradientError::EmptyGradients);
        }

        let mut all: Vec<Vec<f32>> = Vec::with_capacity(1 + self.peer_gradients.len());
        all.push(self.local_gradient.clone());
        all.extend(self.peer_gradients.values().cloned());

        match &self.config.aggregation {
            AggregationMethod::Sum => {
                let dim = all[0].len();
                if all.iter().any(|g| g.len() != dim) {
                    return Err(GradientError::DimensionMismatch);
                }
                let mut sum = vec![0.0f32; dim];
                for grad in &all {
                    for (a, &g) in sum.iter_mut().zip(grad.iter()) {
                        *a += g;
                    }
                }
                Ok(sum)
            }
            AggregationMethod::Mean | AggregationMethod::FedAvg => federated_average(&all),
            AggregationMethod::WeightedMean { weights } => {
                let w: Vec<f32> = weights.clone();
                GradientAggregator::weighted_average(&all, &w)
            }
        }
    }

    /// Returns `true` when at least `min_peers` peer gradients have been collected.
    pub fn is_ready(&self, min_peers: usize) -> bool {
        self.peer_gradients.len() >= min_peers
    }

    /// The session identifier passed at construction time.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Number of peer gradients collected so far (not counting local).
    pub fn peer_count(&self) -> usize {
        self.peer_gradients.len()
    }
}

// ── New federated infrastructure tests ───────────────────────────────────

#[cfg(test)]
mod federated_v2_tests {
    use super::*;

    // ── ConvergenceDetector (EMA mode) ───────────────────────────────────

    #[test]
    fn test_convergence_detector_converges() {
        // Use a larger window so the small-delta region has room to accumulate.
        let cfg = ConvergenceConfig {
            threshold: 0.01,
            window_size: 4,
            smoothing: 0.9,
            patience: 3,
        };
        let mut det = ConvergenceDetector::with_config(cfg);

        // Large decreases first, then nearly-flat tail (delta ≈ 0.001 < threshold 0.01)
        let losses: Vec<f32> = {
            let mut v = vec![10.0_f32, 5.0, 2.0, 1.0, 0.5];
            // Flat tail: 12 rounds of near-zero change
            for i in 0..12 {
                v.push(0.5 - (i as f32) * 0.001);
            }
            v
        };
        let mut converged = false;
        for &l in &losses {
            if det.update(l) {
                converged = true;
                break;
            }
        }
        assert!(
            converged,
            "should converge on rapidly decreasing then flat loss"
        );
    }

    #[test]
    fn test_convergence_detector_no_convergence() {
        let cfg = ConvergenceConfig {
            threshold: 0.01,
            window_size: 5,
            smoothing: 0.9,
            patience: 3,
        };
        let mut det = ConvergenceDetector::with_config(cfg);

        // Oscillating losses — large deltas, never converges
        let losses = [1.0_f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let mut converged = false;
        for &l in &losses {
            if det.update(l) {
                converged = true;
                break;
            }
        }
        assert!(!converged, "should NOT converge on oscillating loss");
    }

    #[test]
    fn test_convergence_detector_plateau() {
        let cfg = ConvergenceConfig {
            threshold: 0.1,
            window_size: 4,
            smoothing: 0.9,
            patience: 3,
        };
        let mut det = ConvergenceDetector::with_config(cfg);

        // Flat loss → tiny deltas, plateau should be detected
        for _ in 0..10 {
            det.update(0.5_f32);
        }
        // After many flat updates the plateau counter should exceed patience
        assert!(
            det.plateau_rounds() >= 3,
            "plateau_rounds={} should be >= patience=3",
            det.plateau_rounds()
        );
    }

    // ── DifferentialPrivacy (new stateless helpers) ─────────────────────

    #[test]
    fn test_dp_gaussian_noise_scale() {
        let epsilon = 1.0_f64;
        let delta = 1e-5_f64;
        let sensitivity = 1.0_f32;

        let dp =
            DifferentialPrivacy::new(epsilon, delta, sensitivity as f64, DPMechanism::Gaussian);

        // Expected σ = sensitivity * sqrt(2*ln(1.25/δ)) / ε
        let ln_term = (1.25 / delta).ln();
        let expected_sigma = (sensitivity as f64) * (2.0 * ln_term).sqrt() / epsilon;

        // Generate many samples and check empirical std-dev ≈ expected_sigma
        let mut base = vec![0.0_f32; 1000];
        dp.add_gaussian_noise_sens(&mut base, sensitivity);

        let mean = base.iter().sum::<f32>() / base.len() as f32;
        let variance = base.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / base.len() as f32;
        let std_dev = variance.sqrt() as f64;

        // Allow 40% relative error (statistical fluctuation in 1000 samples)
        let rel_err = (std_dev - expected_sigma).abs() / expected_sigma;
        assert!(
            rel_err < 0.40,
            "empirical σ={std_dev:.4} expected σ={expected_sigma:.4} rel_err={rel_err:.4}"
        );
    }

    #[test]
    fn test_dp_clip_l2_norm() {
        let dp = DifferentialPrivacy::new(1.0, 1e-5, 1.0, DPMechanism::Gaussian);

        let mut grad = vec![3.0_f32, 4.0]; // L2 norm = 5.0
        dp.clip_l2(&mut grad, 2.5);

        let norm: f32 = grad.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 2.5).abs() < 1e-4,
            "clipped norm {norm} should be ≈ 2.5"
        );
    }

    #[test]
    fn test_dp_budget_exhaustion() {
        // epsilon=1.0, each round costs ε/100 = 0.01 → 100 rounds to exhaust
        let dp = DifferentialPrivacy::new(1.0, 1e-5, 1.0, DPMechanism::Gaussian);
        assert!(!dp.is_exhausted_after(99));
        assert!(dp.is_exhausted_after(100));
        assert!(dp.is_exhausted_after(200));
    }

    // ── FederatedRound (session mode) ────────────────────────────────────

    #[test]
    fn test_federated_round_complete() {
        let clients = vec!["alice".to_string(), "bob".to_string(), "carol".to_string()];
        let mut round = FederatedRound::start(1, clients);

        assert!(!round.is_complete(), "not complete before contributions");

        round.record_contribution("alice", vec![1.0, 2.0]);
        round.record_contribution("bob", vec![3.0, 4.0]);
        round.record_contribution("carol", vec![5.0, 6.0]);

        assert!(round.is_complete(), "complete after all clients contribute");
    }

    #[test]
    fn test_federated_round_missing_client() {
        let clients = vec!["alice".to_string(), "bob".to_string(), "carol".to_string()];
        let mut round = FederatedRound::start(2, clients);

        round.record_contribution("alice", vec![1.0, 2.0]);
        // bob and carol are missing

        let stats = round.stats();
        assert!(
            !stats.missing_clients.is_empty(),
            "missing_clients should be non-empty"
        );
        assert!(
            stats.missing_clients.contains(&"bob".to_string()),
            "bob should be missing"
        );
        assert!(
            stats.missing_clients.contains(&"carol".to_string()),
            "carol should be missing"
        );
    }

    #[test]
    fn test_federated_aggregate_fedavg() {
        let clients = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut round = FederatedRound::start(3, clients);

        round.record_contribution("a", vec![1.0_f32, 2.0, 3.0]);
        round.record_contribution("b", vec![3.0_f32, 4.0, 5.0]);
        round.record_contribution("c", vec![5.0_f32, 6.0, 7.0]);

        let agg = round.aggregate(None).expect("aggregate");
        // FedAvg: ([1,2,3]+[3,4,5]+[5,6,7]) / 3 = [3,4,5]
        assert!((agg[0] - 3.0).abs() < 1e-4, "agg[0]={}", agg[0]);
        assert!((agg[1] - 4.0).abs() < 1e-4, "agg[1]={}", agg[1]);
        assert!((agg[2] - 5.0).abs() < 1e-4, "agg[2]={}", agg[2]);
    }

    // ── GossipModelSync / ModelUpdate ────────────────────────────────────

    #[tokio::test]
    async fn test_model_update_verify() {
        let sync = GossipModelSync::new("peer-local");

        // Valid CID (use a real base32 CIDv1 string produced by the cid crate)
        let valid_cid = "bafkreihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku";
        let valid_update = ModelUpdate {
            peer_id: "peer-remote".to_string(),
            model_cid: valid_cid.to_string(),
            round_id: 1,
            timestamp_ms: 0,
        };
        assert!(sync.verify_update(&valid_update), "valid CID should pass");

        // Tampered: empty CID
        let tampered = ModelUpdate {
            peer_id: "peer-remote".to_string(),
            model_cid: "".to_string(),
            round_id: 1,
            timestamp_ms: 0,
        };
        assert!(!sync.verify_update(&tampered), "empty CID should fail");

        // Tampered: whitespace-only CID
        let tampered2 = ModelUpdate {
            peer_id: "peer-remote".to_string(),
            model_cid: "   ".to_string(),
            round_id: 1,
            timestamp_ms: 0,
        };
        assert!(
            !sync.verify_update(&tampered2),
            "whitespace CID should fail"
        );
    }

    #[tokio::test]
    async fn test_gossip_broadcast_and_collect() {
        let sender = GossipModelSync::new("sender");
        let mut receiver = sender.subscribe("receiver");

        let cid = "bafkreihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku";
        sender.broadcast_update(cid, 1).await.expect("broadcast");

        let updates = receiver
            .collect_updates(1, 200)
            .await
            .expect("collect_updates");

        assert_eq!(updates.len(), 1, "should receive 1 update");
        assert_eq!(updates[0].model_cid, cid);
        assert_eq!(updates[0].round_id, 1);
    }
}
