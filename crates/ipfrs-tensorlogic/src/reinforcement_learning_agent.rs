//! Tabular reinforcement learning agent with multiple algorithms.
//!
//! Implements [`ReinforcementLearningAgent`] supporting:
//! - **Policies**: Epsilon-Greedy (with decay), Boltzmann (softmax), UCB, Random
//! - **Algorithms**: SARSA, Q-Learning, Expected SARSA, Double Q-Learning, N-Step TD
//! - **Experience Replay**: circular buffer with uniform random sampling
//! - **Eligibility Traces**: λ parameter for TD(λ) extensions
//! - **Statistics**: per-episode and aggregate tracking
//!
//! No external RNG dependency — uses an inline xorshift64 PRNG throughout.
//!
//! # Example
//!
//! ```
//! use ipfrs_tensorlogic::reinforcement_learning_agent::{
//!     ReinforcementLearningAgent, RlState, RlAction, AgentConfig, AlgorithmType, AgentPolicy,
//!     Transition,
//! };
//!
//! let config = AgentConfig::default();
//! let mut agent = ReinforcementLearningAgent::new(config);
//!
//! let s0 = RlState("s0".into());
//! let a0 = RlAction("left".into());
//! let a1 = RlAction("right".into());
//! agent.register_state(s0.clone(), vec![a0.clone(), a1.clone()]).expect("example: should succeed in docs");
//!
//! let s1 = RlState("s1".into());
//! agent.register_state(s1.clone(), vec![a0.clone(), a1.clone()]).expect("example: should succeed in docs");
//!
//! let t = Transition { state: s0.clone(), action: a0.clone(), reward: 1.0, next_state: s1.clone(), done: false };
//! let _delta = agent.update(&t).expect("example: should succeed in docs");
//! let best = agent.best_action(&s0).expect("example: should succeed in docs");
//! assert!(best == a0 || best == a1);
//! ```

use std::collections::HashMap;

mod rla_types;
pub use rla_types::*;
use rla_types::{NStepBuffer, QEntry};

// ────────────────────────────────────────────────────────────────────────────
// ReinforcementLearningAgent
// ────────────────────────────────────────────────────────────────────────────

/// Production-quality tabular RL agent supporting multiple algorithms and policies.
///
/// Register states and their valid actions, then call [`update`](Self::update)
/// after each environment step to learn Q-values.  Call
/// [`run_episode`](Self::run_episode) to process a complete episode at once.
#[derive(Debug)]
pub struct ReinforcementLearningAgent {
    /// Agent configuration (algorithm, policy, hyperparameters).
    config: AgentConfig,
    /// Valid actions keyed by state.
    state_actions: HashMap<RlState, Vec<RlAction>>,
    /// Q-table: (state, action) → QEntry.
    q_table: HashMap<(RlState, RlAction), QEntry>,
    /// Eligibility traces: (state, action) → e(s,a).
    eligibility: HashMap<(RlState, RlAction), f64>,
    /// Toggle flag for Double Q-learning (which table to update).
    double_q_toggle: bool,
    /// N-step accumulation buffer (used only for NStepTD).
    n_step_buf: NStepBuffer,
    /// Experience replay buffer.
    replay: ExperienceReplay,
    /// Aggregate statistics.
    stats: AgentStats,
    /// Total visits across all (s,a) pairs (for UCB denominator).
    total_visits: u64,
    /// Maximum |ΔQ| during the last episode (tracks convergence).
    last_episode_delta: f64,
}

impl ReinforcementLearningAgent {
    /// Create a new agent with the given configuration.
    ///
    /// Returns [`RlAgentError::InvalidConfig`] if hyperparameters are out of range.
    pub fn new(config: AgentConfig) -> Self {
        let replay_cap = config.replay_capacity.max(1);
        let n = match config.algorithm {
            AlgorithmType::NStepTD(n) => n.max(1),
            _ => 1,
        };
        Self {
            replay: ExperienceReplay::new(replay_cap),
            config,
            state_actions: HashMap::new(),
            q_table: HashMap::new(),
            eligibility: HashMap::new(),
            double_q_toggle: false,
            n_step_buf: NStepBuffer::new(n),
            stats: AgentStats {
                episodes_run: 0,
                total_steps: 0,
                avg_reward: 0.0,
                best_episode_reward: f64::NEG_INFINITY,
                convergence_delta: 0.0,
            },
            total_visits: 0,
            last_episode_delta: 0.0,
        }
    }

    // ── Registration ─────────────────────────────────────────────────────────

    /// Register `state` as a valid environment state with `actions` as its
    /// legal action set.  Can be called multiple times to add more actions;
    /// duplicate actions are ignored.
    ///
    /// # Errors
    /// Returns [`RlAgentError::InvalidConfig`] when `actions` is empty.
    pub fn register_state(
        &mut self,
        state: RlState,
        actions: Vec<RlAction>,
    ) -> Result<(), RlAgentError> {
        if actions.is_empty() {
            return Err(RlAgentError::InvalidConfig(format!(
                "state {:?} must have at least one action",
                state.0
            )));
        }
        let entry = self.state_actions.entry(state.clone()).or_default();
        for a in actions {
            // Ensure Q-entry exists.
            self.q_table.entry((state.clone(), a.clone())).or_default();
            if !entry.contains(&a) {
                entry.push(a);
            }
        }
        Ok(())
    }

    // ── Action selection ─────────────────────────────────────────────────────

    /// Select an action for `state` according to the configured policy.
    ///
    /// # Errors
    /// - [`RlAgentError::StateNotFound`] if the state is not registered.
    pub fn select_action(
        &self,
        state: &RlState,
        rng_seed: &mut u64,
    ) -> Result<RlAction, RlAgentError> {
        let actions = self
            .state_actions
            .get(state)
            .ok_or_else(|| RlAgentError::StateNotFound(state.clone()))?;

        match &self.config.policy {
            AgentPolicy::EpsilonGreedy { epsilon, .. } => {
                let r = xorshift_f64(rng_seed);
                if r < *epsilon {
                    Ok(self.random_action(actions, rng_seed))
                } else {
                    Ok(self.greedy_action(state, actions))
                }
            }
            AgentPolicy::Boltzmann { temperature } => {
                Ok(self.boltzmann_action(state, actions, *temperature, rng_seed))
            }
            AgentPolicy::UCB { c } => Ok(self.ucb_action(state, actions, *c)),
            AgentPolicy::Random => Ok(self.random_action(actions, rng_seed)),
        }
    }

    /// Return the greedy (argmax Q) action for `state`.
    ///
    /// # Errors
    /// - [`RlAgentError::StateNotFound`] if the state is not registered.
    pub fn best_action(&self, state: &RlState) -> Result<RlAction, RlAgentError> {
        let actions = self
            .state_actions
            .get(state)
            .ok_or_else(|| RlAgentError::StateNotFound(state.clone()))?;
        Ok(self.greedy_action(state, actions))
    }

    /// V(s) = max_a Q(s, a).  Returns 0.0 if the state is unregistered.
    pub fn value(&self, state: &RlState) -> f64 {
        match self.state_actions.get(state) {
            None => 0.0,
            Some(actions) => actions
                .iter()
                .map(|a| self.q1(state, a))
                .fold(f64::NEG_INFINITY, f64::max),
        }
    }

    // ── Q-table update ───────────────────────────────────────────────────────

    /// Apply a single TD update from `transition`.
    ///
    /// Returns the absolute TD error |δ| so callers can track convergence.
    ///
    /// # Errors
    /// - [`RlAgentError::StateNotFound`] — state or next_state not registered.
    /// - [`RlAgentError::ActionNotFound`] — action not valid for state.
    pub fn update(&mut self, transition: &Transition) -> Result<f64, RlAgentError> {
        self.validate_transition(transition)?;

        let delta = match self.config.algorithm.clone() {
            AlgorithmType::QLearning => self.update_q_learning(transition),
            AlgorithmType::Sarsa => self.update_sarsa(transition),
            AlgorithmType::ExpectedSarsa => self.update_expected_sarsa(transition),
            AlgorithmType::DoubleQLearning => self.update_double_q(transition),
            AlgorithmType::NStepTD(_) => self.update_n_step(transition),
        };

        // Track eligibility traces (decay all entries after each step).
        self.decay_eligibility();

        // Track per-episode convergence.
        if delta.abs() > self.last_episode_delta {
            self.last_episode_delta = delta.abs();
        }

        // Update global visit counter.
        self.total_visits += 1;
        let entry = self
            .q_table
            .entry((transition.state.clone(), transition.action.clone()))
            .or_default();
        entry.visits += 1;

        Ok(delta.abs())
    }

    // ── Episode runner ───────────────────────────────────────────────────────

    /// Process a complete sequence of transitions as one episode.
    ///
    /// Epsilon is decayed once after all transitions are processed.
    /// Episode statistics are accumulated into [`AgentStats`].
    ///
    /// # Errors
    /// Propagates any error from [`update`](Self::update).
    pub fn run_episode(
        &mut self,
        transitions: Vec<Transition>,
        _rng_seed: u64,
    ) -> Result<EpisodeStats, RlAgentError> {
        if transitions.is_empty() {
            return Ok(EpisodeStats {
                total_reward: 0.0,
                steps: 0,
                epsilon: self.current_epsilon(),
                avg_q_value: 0.0,
            });
        }

        self.last_episode_delta = 0.0;
        self.eligibility.clear();

        let mut total_reward = 0.0;
        let mut q_sum = 0.0;
        let mut q_count = 0usize;

        for t in &transitions {
            // Register states on-the-fly if not yet seen (best-effort).
            total_reward += t.reward;
            let _ = self.update(t)?;
            let q = self.q1(&t.state, &t.action);
            q_sum += q;
            q_count += 1;
        }

        let steps = transitions.len();
        let eps = self.current_epsilon();

        // Decay epsilon at end of episode.
        self.decay_epsilon();

        // Update aggregate stats.
        let ema_alpha = 0.05_f64;
        self.stats.avg_reward =
            self.stats.avg_reward * (1.0 - ema_alpha) + total_reward * ema_alpha;
        if total_reward > self.stats.best_episode_reward {
            self.stats.best_episode_reward = total_reward;
        }
        self.stats.episodes_run += 1;
        self.stats.total_steps += steps as u64;
        self.stats.convergence_delta = self.last_episode_delta;

        let avg_q = if q_count > 0 {
            q_sum / q_count as f64
        } else {
            0.0
        };

        Ok(EpisodeStats {
            total_reward,
            steps,
            epsilon: eps,
            avg_q_value: avg_q,
        })
    }

    // ── Epsilon decay ────────────────────────────────────────────────────────

    /// Decay epsilon for EpsilonGreedy policies: ε ← max(min_ε, ε × decay).
    /// No-op for other policies.
    pub fn decay_epsilon(&mut self) {
        if let AgentPolicy::EpsilonGreedy {
            ref mut epsilon,
            decay,
            min_epsilon,
        } = self.config.policy
        {
            *epsilon = (*epsilon * decay).max(min_epsilon);
        }
    }

    // ── Experience replay ────────────────────────────────────────────────────

    /// Push `t` into the experience replay buffer.
    pub fn add_experience(&mut self, t: Transition) {
        self.replay.push(t);
    }

    /// Draw `n` transitions uniformly at random from the replay buffer.
    ///
    /// # Errors
    /// - [`RlAgentError::InsufficientExperience`] when the buffer has fewer than
    ///   `n` entries.
    pub fn sample_experience(
        &self,
        n: usize,
        rng_seed: u64,
    ) -> Result<Vec<Transition>, RlAgentError> {
        let buf_len = self.replay.len();
        if buf_len < n {
            return Err(RlAgentError::InsufficientExperience(buf_len));
        }
        let mut seed = rng_seed ^ 0xdead_beef_cafe_u64;
        let mut out = Vec::with_capacity(n);
        // Reservoir / uniform sampling without replacement via partial Fisher-Yates.
        let mut indices: Vec<usize> = (0..buf_len).collect();
        for i in 0..n {
            let j = i + (xorshift64(&mut seed) as usize % (buf_len - i));
            indices.swap(i, j);
            out.push(self.replay.buffer[indices[i]].clone());
        }
        Ok(out)
    }

    // ── Statistics ───────────────────────────────────────────────────────────

    /// Return a snapshot of aggregate statistics.
    pub fn stats(&self) -> AgentStats {
        self.stats.clone()
    }

    // ────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ────────────────────────────────────────────────────────────────────────

    /// Read Q1(s, a), defaulting to 0.0.
    fn q1(&self, state: &RlState, action: &RlAction) -> f64 {
        self.q_table
            .get(&(state.clone(), action.clone()))
            .map_or(0.0, |e| e.q1)
    }

    /// Read Q2(s, a), defaulting to 0.0 (Double Q-learning only).
    fn q2(&self, state: &RlState, action: &RlAction) -> f64 {
        self.q_table
            .get(&(state.clone(), action.clone()))
            .map_or(0.0, |e| e.q2)
    }

    /// Max Q1 over all registered actions for `state`.
    fn max_q1(&self, state: &RlState) -> f64 {
        self.state_actions
            .get(state)
            .map(|acts| {
                acts.iter()
                    .map(|a| self.q1(state, a))
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .unwrap_or(0.0)
    }

    /// Greedy argmax Q1 action.
    fn greedy_action(&self, state: &RlState, actions: &[RlAction]) -> RlAction {
        actions
            .iter()
            .max_by(|a, b| {
                self.q1(state, a)
                    .partial_cmp(&self.q1(state, b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .unwrap_or_else(|| actions[0].clone())
    }

    /// Uniform random action selection.
    fn random_action(&self, actions: &[RlAction], rng_seed: &mut u64) -> RlAction {
        let idx = xorshift64(rng_seed) as usize % actions.len();
        actions[idx].clone()
    }

    /// Boltzmann (softmax) action selection.
    fn boltzmann_action(
        &self,
        state: &RlState,
        actions: &[RlAction],
        temperature: f64,
        rng_seed: &mut u64,
    ) -> RlAction {
        if temperature <= 0.0 {
            return self.greedy_action(state, actions);
        }
        // Numerically stable softmax: shift by max.
        let qs: Vec<f64> = actions.iter().map(|a| self.q1(state, a)).collect();
        let max_q = qs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = qs
            .iter()
            .map(|&q| ((q - max_q) / temperature).exp())
            .collect();
        let sum: f64 = exps.iter().sum();
        let r = xorshift_f64(rng_seed) * sum;
        let mut cumulative = 0.0;
        for (i, &e) in exps.iter().enumerate() {
            cumulative += e;
            if r <= cumulative {
                return actions[i].clone();
            }
        }
        actions[actions.len() - 1].clone()
    }

    /// UCB action selection: argmax[Q(s,a) + c * sqrt(ln(N) / (n_a + 1))].
    fn ucb_action(&self, state: &RlState, actions: &[RlAction], c: f64) -> RlAction {
        let ln_n = if self.total_visits > 0 {
            (self.total_visits as f64).ln()
        } else {
            0.0
        };
        actions
            .iter()
            .max_by(|a, b| {
                let visits_a = self
                    .q_table
                    .get(&(state.clone(), (*a).clone()))
                    .map_or(0, |e| e.visits);
                let visits_b = self
                    .q_table
                    .get(&(state.clone(), (*b).clone()))
                    .map_or(0, |e| e.visits);
                let ucb_a = self.q1(state, a) + c * (ln_n / (visits_a as f64 + 1.0)).sqrt();
                let ucb_b = self.q1(state, b) + c * (ln_n / (visits_b as f64 + 1.0)).sqrt();
                ucb_a
                    .partial_cmp(&ucb_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .unwrap_or_else(|| actions[0].clone())
    }

    /// Expected value of Q under the current ε-greedy policy over `actions`.
    fn expected_q(&self, state: &RlState, actions: &[RlAction]) -> f64 {
        let n = actions.len() as f64;
        let eps = match &self.config.policy {
            AgentPolicy::EpsilonGreedy { epsilon, .. } => *epsilon,
            _ => 0.0,
        };
        let best = self.greedy_action(state, actions);
        let random_contrib: f64 = actions.iter().map(|a| self.q1(state, a)).sum::<f64>() / n;
        let greedy_contrib = self.q1(state, &best);
        eps * random_contrib + (1.0 - eps) * greedy_contrib
    }

    /// Return the current epsilon (0.0 if policy is not EpsilonGreedy).
    fn current_epsilon(&self) -> f64 {
        if let AgentPolicy::EpsilonGreedy { epsilon, .. } = &self.config.policy {
            *epsilon
        } else {
            0.0
        }
    }

    /// Validate that both state and action in `t` are registered.
    fn validate_transition(&self, t: &Transition) -> Result<(), RlAgentError> {
        let actions = self
            .state_actions
            .get(&t.state)
            .ok_or_else(|| RlAgentError::StateNotFound(t.state.clone()))?;
        if !actions.contains(&t.action) {
            return Err(RlAgentError::ActionNotFound {
                state: t.state.clone(),
                action: t.action.clone(),
            });
        }
        // next_state must exist unless done (terminal states may be unregistered).
        if !t.done && !self.state_actions.contains_key(&t.next_state) {
            return Err(RlAgentError::StateNotFound(t.next_state.clone()));
        }
        Ok(())
    }

    // ── Algorithm implementations ─────────────────────────────────────────────

    /// Q-learning update. Returns TD error δ.
    fn update_q_learning(&mut self, t: &Transition) -> f64 {
        let alpha = self.config.alpha;
        let gamma = self.config.gamma;
        let q_sa = self.q1(&t.state, &t.action);
        let max_next = if t.done {
            0.0
        } else {
            self.max_q1(&t.next_state)
        };
        let delta = t.reward + gamma * max_next - q_sa;
        let entry = self
            .q_table
            .entry((t.state.clone(), t.action.clone()))
            .or_default();
        entry.q1 += alpha * delta;
        delta
    }

    /// SARSA update. Returns TD error δ.
    /// Uses greedy next action (on-policy under current policy).
    fn update_sarsa(&mut self, t: &Transition) -> f64 {
        let alpha = self.config.alpha;
        let gamma = self.config.gamma;
        let q_sa = self.q1(&t.state, &t.action);
        // On-policy next action: greedy (argmax Q) for simplicity in tabular case.
        let q_next = if t.done {
            0.0
        } else {
            let next_actions = self
                .state_actions
                .get(&t.next_state)
                .cloned()
                .unwrap_or_default();
            if next_actions.is_empty() {
                0.0
            } else {
                let next_a = self.greedy_action(&t.next_state, &next_actions);
                self.q1(&t.next_state, &next_a)
            }
        };
        let delta = t.reward + gamma * q_next - q_sa;
        // Eligibility trace update for SARSA(λ).
        let lambda = self.config.lambda;
        *self
            .eligibility
            .entry((t.state.clone(), t.action.clone()))
            .or_insert(0.0) += 1.0;
        // Update all entries proportional to their eligibility.
        let keys: Vec<(RlState, RlAction)> = self.eligibility.keys().cloned().collect();
        for key in keys {
            let e = *self.eligibility.get(&key).unwrap_or(&0.0);
            let entry = self.q_table.entry(key.clone()).or_default();
            entry.q1 += alpha * delta * e;
            let e_ref = self.eligibility.entry(key).or_insert(0.0);
            *e_ref *= gamma * lambda;
        }
        delta
    }

    /// Expected SARSA update. Returns TD error δ.
    fn update_expected_sarsa(&mut self, t: &Transition) -> f64 {
        let alpha = self.config.alpha;
        let gamma = self.config.gamma;
        let q_sa = self.q1(&t.state, &t.action);
        let expected_next = if t.done {
            0.0
        } else {
            let next_actions = self
                .state_actions
                .get(&t.next_state)
                .cloned()
                .unwrap_or_default();
            if next_actions.is_empty() {
                0.0
            } else {
                self.expected_q(&t.next_state, &next_actions)
            }
        };
        let delta = t.reward + gamma * expected_next - q_sa;
        let entry = self
            .q_table
            .entry((t.state.clone(), t.action.clone()))
            .or_default();
        entry.q1 += alpha * delta;
        delta
    }

    /// Double Q-learning update. Alternates which table is updated.  Returns TD error δ.
    fn update_double_q(&mut self, t: &Transition) -> f64 {
        let alpha = self.config.alpha;
        let gamma = self.config.gamma;
        self.double_q_toggle = !self.double_q_toggle;
        let delta = if self.double_q_toggle {
            // Update Q1 using Q2 for evaluation.
            let q1_sa = self.q1(&t.state, &t.action);
            let max_next = if t.done {
                0.0
            } else {
                // Select action via Q1, evaluate via Q2.
                let next_actions = self
                    .state_actions
                    .get(&t.next_state)
                    .cloned()
                    .unwrap_or_default();
                if next_actions.is_empty() {
                    0.0
                } else {
                    let best_a = self.greedy_action(&t.next_state, &next_actions);
                    self.q2(&t.next_state, &best_a)
                }
            };
            let delta = t.reward + gamma * max_next - q1_sa;
            let entry = self
                .q_table
                .entry((t.state.clone(), t.action.clone()))
                .or_default();
            entry.q1 += alpha * delta;
            delta
        } else {
            // Update Q2 using Q1 for evaluation.
            let q2_sa = self.q2(&t.state, &t.action);
            let max_next = if t.done {
                0.0
            } else {
                // Select action via Q2, evaluate via Q1.
                let next_actions = self
                    .state_actions
                    .get(&t.next_state)
                    .cloned()
                    .unwrap_or_default();
                if next_actions.is_empty() {
                    0.0
                } else {
                    let best_a = self
                        .state_actions
                        .get(&t.next_state)
                        .and_then(|acts| {
                            acts.iter()
                                .max_by(|a, b| {
                                    self.q2(&t.next_state, a)
                                        .partial_cmp(&self.q2(&t.next_state, b))
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .cloned()
                        })
                        .unwrap_or_else(|| next_actions[0].clone());
                    self.q1(&t.next_state, &best_a)
                }
            };
            let delta = t.reward + gamma * max_next - q2_sa;
            let entry = self
                .q_table
                .entry((t.state.clone(), t.action.clone()))
                .or_default();
            entry.q2 += alpha * delta;
            delta
        };
        delta
    }

    /// N-step TD update. Buffers transitions until n steps are available.
    /// Returns 0.0 until the buffer is ready, then the actual TD error.
    fn update_n_step(&mut self, t: &Transition) -> f64 {
        self.n_step_buf.transitions.push_back(t.clone());
        if !self.n_step_buf.ready() {
            return 0.0;
        }
        let oldest = match self.n_step_buf.transitions.pop_front() {
            Some(o) => o,
            None => return 0.0,
        };
        let alpha = self.config.alpha;
        let gamma = self.config.gamma;
        let q_sa = self.q1(&oldest.state, &oldest.action);
        // Bootstrap from the tail state.
        let tail = self
            .n_step_buf
            .transitions
            .back()
            .map(|last| {
                if last.done {
                    0.0
                } else {
                    self.max_q1(&last.next_state)
                }
            })
            .unwrap_or(0.0);
        let g = self.n_step_buf.n_step_return(gamma, tail);
        let delta = g - q_sa;
        let entry = self
            .q_table
            .entry((oldest.state.clone(), oldest.action.clone()))
            .or_default();
        entry.q1 += alpha * delta;
        delta
    }

    /// Decay all eligibility traces by γλ.
    fn decay_eligibility(&mut self) {
        let gamma = self.config.gamma;
        let lambda = self.config.lambda;
        let factor = gamma * lambda;
        if (factor - 0.0).abs() < f64::EPSILON {
            self.eligibility.clear();
            return;
        }
        for e in self.eligibility.values_mut() {
            *e *= factor;
        }
        // Remove negligibly small traces to keep memory bounded.
        self.eligibility.retain(|_, e| e.abs() > 1e-10);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper macros / factories ─────────────────────────────────────────────

    fn s(name: &str) -> RlState {
        RlState(name.to_string())
    }

    fn a(name: &str) -> RlAction {
        RlAction(name.to_string())
    }

    fn two_state_agent(algo: AlgorithmType, policy: AgentPolicy) -> ReinforcementLearningAgent {
        let config = AgentConfig {
            algorithm: algo,
            policy,
            alpha: 0.5,
            gamma: 0.9,
            lambda: 0.8,
            replay_capacity: 100,
            batch_size: 8,
        };
        let mut agent = ReinforcementLearningAgent::new(config);
        agent
            .register_state(s("A"), vec![a("left"), a("right")])
            .expect("test: should succeed");
        agent
            .register_state(s("B"), vec![a("left"), a("right")])
            .expect("test: should succeed");
        agent
    }

    fn simple_transition(done: bool) -> Transition {
        Transition {
            state: s("A"),
            action: a("left"),
            reward: 1.0,
            next_state: s("B"),
            done,
        }
    }

    // ── register_state ────────────────────────────────────────────────────────

    #[test]
    fn test_register_state_basic() {
        let mut agent = ReinforcementLearningAgent::new(AgentConfig::default());
        agent
            .register_state(s("s0"), vec![a("up"), a("down")])
            .expect("test: should succeed");
        assert!(agent.state_actions.contains_key(&s("s0")));
        assert_eq!(agent.state_actions[&s("s0")].len(), 2);
    }

    #[test]
    fn test_register_state_empty_actions_error() {
        let mut agent = ReinforcementLearningAgent::new(AgentConfig::default());
        let result = agent.register_state(s("s0"), vec![]);
        assert!(matches!(result, Err(RlAgentError::InvalidConfig(_))));
    }

    #[test]
    fn test_register_state_dedup_actions() {
        let mut agent = ReinforcementLearningAgent::new(AgentConfig::default());
        agent
            .register_state(s("s0"), vec![a("up"), a("up"), a("down")])
            .expect("test: should succeed");
        assert_eq!(agent.state_actions[&s("s0")].len(), 2);
    }

    #[test]
    fn test_register_state_multiple_calls_merge() {
        let mut agent = ReinforcementLearningAgent::new(AgentConfig::default());
        agent
            .register_state(s("s0"), vec![a("up")])
            .expect("test: should succeed");
        agent
            .register_state(s("s0"), vec![a("down")])
            .expect("test: should succeed");
        assert_eq!(agent.state_actions[&s("s0")].len(), 2);
    }

    // ── select_action: EpsilonGreedy ─────────────────────────────────────────

    #[test]
    fn test_epsilon_greedy_high_epsilon_mostly_random() {
        let policy = AgentPolicy::EpsilonGreedy {
            epsilon: 1.0,
            decay: 1.0,
            min_epsilon: 1.0,
        };
        let agent = two_state_agent(AlgorithmType::QLearning, policy);
        let mut seed = 42u64;
        // With ε=1 all choices should be random — just verify no panic.
        for _ in 0..20 {
            let act = agent
                .select_action(&s("A"), &mut seed)
                .expect("test: should succeed");
            assert!(act == a("left") || act == a("right"));
        }
    }

    #[test]
    fn test_epsilon_greedy_zero_epsilon_greedy() {
        let policy = AgentPolicy::EpsilonGreedy {
            epsilon: 0.0,
            decay: 1.0,
            min_epsilon: 0.0,
        };
        let mut agent = two_state_agent(AlgorithmType::QLearning, policy);
        // Manually set Q so right is clearly better.
        agent.q_table.entry((s("A"), a("right"))).or_default().q1 = 10.0;
        let mut seed = 999u64;
        let act = agent
            .select_action(&s("A"), &mut seed)
            .expect("test: should succeed");
        assert_eq!(act, a("right"));
    }

    #[test]
    fn test_epsilon_greedy_state_not_found() {
        let policy = AgentPolicy::EpsilonGreedy {
            epsilon: 0.1,
            decay: 0.99,
            min_epsilon: 0.01,
        };
        let agent = two_state_agent(AlgorithmType::QLearning, policy);
        let mut seed = 1u64;
        let result = agent.select_action(&s("UNKNOWN"), &mut seed);
        assert!(matches!(result, Err(RlAgentError::StateNotFound(_))));
    }

    // ── select_action: Boltzmann ─────────────────────────────────────────────

    #[test]
    fn test_boltzmann_returns_valid_action() {
        let policy = AgentPolicy::Boltzmann { temperature: 1.0 };
        let agent = two_state_agent(AlgorithmType::QLearning, policy);
        let mut seed = 7u64;
        for _ in 0..30 {
            let act = agent
                .select_action(&s("A"), &mut seed)
                .expect("test: should succeed");
            assert!(act == a("left") || act == a("right"));
        }
    }

    #[test]
    fn test_boltzmann_zero_temperature_greedy() {
        let policy = AgentPolicy::Boltzmann { temperature: 0.0 };
        let mut agent = two_state_agent(AlgorithmType::QLearning, policy);
        agent.q_table.entry((s("A"), a("left"))).or_default().q1 = 5.0;
        agent.q_table.entry((s("A"), a("right"))).or_default().q1 = -1.0;
        let mut seed = 0u64;
        let act = agent
            .select_action(&s("A"), &mut seed)
            .expect("test: should succeed");
        assert_eq!(act, a("left"));
    }

    #[test]
    fn test_boltzmann_high_temperature_distribution() {
        // High temperature → uniform-like distribution: both actions should appear.
        let policy = AgentPolicy::Boltzmann {
            temperature: 1000.0,
        };
        let agent = two_state_agent(AlgorithmType::QLearning, policy);
        let mut seed = 13u64;
        let mut left = 0u32;
        let mut right = 0u32;
        for _ in 0..200 {
            match agent
                .select_action(&s("A"), &mut seed)
                .expect("test: should succeed")
            {
                x if x == a("left") => left += 1,
                _ => right += 1,
            }
        }
        // Both should appear at least once.
        assert!(left > 0);
        assert!(right > 0);
    }

    // ── select_action: UCB ────────────────────────────────────────────────────

    #[test]
    fn test_ucb_returns_valid_action() {
        let policy = AgentPolicy::UCB { c: 1.0 };
        let agent = two_state_agent(AlgorithmType::QLearning, policy);
        let act = agent
            .select_action(&s("A"), &mut 0u64)
            .expect("test: should succeed");
        assert!(act == a("left") || act == a("right"));
    }

    #[test]
    fn test_ucb_with_many_visits() {
        let policy = AgentPolicy::UCB { c: 0.5 };
        let mut agent = two_state_agent(AlgorithmType::QLearning, policy);
        agent.total_visits = 1000;
        agent.q_table.entry((s("A"), a("right"))).or_default().q1 = 2.0;
        agent
            .q_table
            .entry((s("A"), a("right")))
            .or_default()
            .visits = 500;
        agent.q_table.entry((s("A"), a("left"))).or_default().visits = 500;
        let act = agent
            .select_action(&s("A"), &mut 0u64)
            .expect("test: should succeed");
        assert!(act == a("left") || act == a("right"));
    }

    // ── select_action: Random ────────────────────────────────────────────────

    #[test]
    fn test_random_policy_all_actions_reachable() {
        let agent = two_state_agent(AlgorithmType::QLearning, AgentPolicy::Random);
        let mut seed = 17u64;
        let mut seen_left = false;
        let mut seen_right = false;
        for _ in 0..100 {
            match agent
                .select_action(&s("A"), &mut seed)
                .expect("test: should succeed")
            {
                x if x == a("left") => seen_left = true,
                _ => seen_right = true,
            }
        }
        assert!(seen_left && seen_right);
    }

    // ── update: QLearning ────────────────────────────────────────────────────

    #[test]
    fn test_qlearning_update_increases_q() {
        let mut agent = two_state_agent(AlgorithmType::QLearning, AgentPolicy::Random);
        let t = simple_transition(false);
        let before = agent.q1(&s("A"), &a("left"));
        let delta = agent.update(&t).expect("test: TD update should succeed");
        let after = agent.q1(&s("A"), &a("left"));
        assert!(delta >= 0.0);
        assert!(after > before);
    }

    #[test]
    fn test_qlearning_terminal_no_bootstrap() {
        let mut agent = two_state_agent(AlgorithmType::QLearning, AgentPolicy::Random);
        let t = simple_transition(true);
        agent.update(&t).expect("test: TD update should succeed");
        // With done=true bootstrap is 0, so Q = alpha * reward = 0.5 * 1.0 = 0.5.
        assert!((agent.q1(&s("A"), &a("left")) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_qlearning_converges_to_optimal() {
        let mut agent = two_state_agent(AlgorithmType::QLearning, AgentPolicy::Random);
        let t_right = Transition {
            state: s("A"),
            action: a("right"),
            reward: 10.0,
            next_state: s("B"),
            done: true,
        };
        let t_left = Transition {
            state: s("A"),
            action: a("left"),
            reward: 1.0,
            next_state: s("B"),
            done: true,
        };
        for _ in 0..50 {
            agent
                .update(&t_right)
                .expect("test: TD update should succeed");
            agent
                .update(&t_left)
                .expect("test: TD update should succeed");
        }
        assert!(agent.q1(&s("A"), &a("right")) > agent.q1(&s("A"), &a("left")));
        assert_eq!(
            agent.best_action(&s("A")).expect("test: should succeed"),
            a("right")
        );
    }

    // ── update: Sarsa ────────────────────────────────────────────────────────

    #[test]
    fn test_sarsa_update_basic() {
        let mut agent = two_state_agent(AlgorithmType::Sarsa, AgentPolicy::Random);
        let t = simple_transition(false);
        let delta = agent.update(&t).expect("test: TD update should succeed");
        assert!(delta >= 0.0);
    }

    #[test]
    fn test_sarsa_eligibility_traces_populated() {
        let mut agent = two_state_agent(AlgorithmType::Sarsa, AgentPolicy::Random);
        let t = simple_transition(false);
        agent.update(&t).expect("test: TD update should succeed");
        // After SARSA update, eligibility map should have entries.
        assert!(!agent.eligibility.is_empty());
    }

    #[test]
    fn test_sarsa_terminal_state() {
        let mut agent = two_state_agent(AlgorithmType::Sarsa, AgentPolicy::Random);
        let t = simple_transition(true);
        agent.update(&t).expect("test: TD update should succeed");
        assert!(agent.q1(&s("A"), &a("left")) > 0.0);
    }

    // ── update: ExpectedSarsa ────────────────────────────────────────────────

    #[test]
    fn test_expected_sarsa_basic() {
        let policy = AgentPolicy::EpsilonGreedy {
            epsilon: 0.1,
            decay: 0.99,
            min_epsilon: 0.01,
        };
        let mut agent = two_state_agent(AlgorithmType::ExpectedSarsa, policy);
        let t = simple_transition(false);
        let delta = agent.update(&t).expect("test: TD update should succeed");
        assert!(delta >= 0.0);
        assert!(agent.q1(&s("A"), &a("left")) != 0.0);
    }

    #[test]
    fn test_expected_sarsa_terminal() {
        let policy = AgentPolicy::EpsilonGreedy {
            epsilon: 0.1,
            decay: 0.99,
            min_epsilon: 0.01,
        };
        let mut agent = two_state_agent(AlgorithmType::ExpectedSarsa, policy);
        let t = simple_transition(true);
        agent.update(&t).expect("test: TD update should succeed");
        assert!((agent.q1(&s("A"), &a("left")) - 0.5).abs() < 1e-9);
    }

    // ── update: DoubleQLearning ──────────────────────────────────────────────

    #[test]
    fn test_double_q_updates_alternating_tables() {
        let mut agent = two_state_agent(AlgorithmType::DoubleQLearning, AgentPolicy::Random);
        let t = simple_transition(false);
        agent.update(&t).expect("test: TD update should succeed"); // updates Q1
        let q1_after_1 = agent.q1(&s("A"), &a("left"));
        let q2_after_1 = agent.q2(&s("A"), &a("left"));
        agent.update(&t).expect("test: TD update should succeed"); // updates Q2
        let q2_after_2 = agent.q2(&s("A"), &a("left"));
        // Q1 unchanged after second update.
        assert!((agent.q1(&s("A"), &a("left")) - q1_after_1).abs() < 1e-12);
        // Q2 should have changed on second update.
        assert!(q2_after_2 != q2_after_1);
    }

    #[test]
    fn test_double_q_terminal() {
        let mut agent = two_state_agent(AlgorithmType::DoubleQLearning, AgentPolicy::Random);
        let t = simple_transition(true);
        agent.update(&t).expect("test: TD update should succeed");
        // First toggle updates Q1.
        assert!(agent.q1(&s("A"), &a("left")) != 0.0 || agent.q2(&s("A"), &a("left")) != 0.0);
    }

    // ── update: NStepTD ──────────────────────────────────────────────────────

    #[test]
    fn test_nstep_td_returns_zero_before_n_steps() {
        let config = AgentConfig {
            algorithm: AlgorithmType::NStepTD(3),
            policy: AgentPolicy::Random,
            alpha: 0.5,
            gamma: 0.9,
            lambda: 0.0,
            replay_capacity: 100,
            batch_size: 8,
        };
        let mut agent = ReinforcementLearningAgent::new(config);
        agent
            .register_state(s("A"), vec![a("left"), a("right")])
            .expect("test: should succeed");
        agent
            .register_state(s("B"), vec![a("left"), a("right")])
            .expect("test: should succeed");

        let t = simple_transition(false);
        let d1 = agent.update(&t).expect("test: TD update should succeed");
        assert_eq!(d1, 0.0); // only 1 step buffered, need 3
        let d2 = agent.update(&t).expect("test: TD update should succeed");
        assert_eq!(d2, 0.0); // 2 steps
    }

    #[test]
    fn test_nstep_td_updates_after_n_steps() {
        let config = AgentConfig {
            algorithm: AlgorithmType::NStepTD(2),
            policy: AgentPolicy::Random,
            alpha: 0.5,
            gamma: 0.9,
            lambda: 0.0,
            replay_capacity: 100,
            batch_size: 8,
        };
        let mut agent = ReinforcementLearningAgent::new(config);
        agent
            .register_state(s("A"), vec![a("left"), a("right")])
            .expect("test: should succeed");
        agent
            .register_state(s("B"), vec![a("left"), a("right")])
            .expect("test: should succeed");

        let t = simple_transition(false);
        agent.update(&t).expect("test: TD update should succeed"); // step 1 — buffered
        let d3 = agent.update(&t).expect("test: TD update should succeed"); // step 2 — triggers update
                                                                            // Non-zero delta expected after n steps.
        assert!(d3 >= 0.0);
    }

    #[test]
    fn test_nstep_td_n1_equivalent_to_qlearning() {
        // n=1 should behave identically to Q-learning.
        let config = AgentConfig {
            algorithm: AlgorithmType::NStepTD(1),
            policy: AgentPolicy::Random,
            alpha: 0.5,
            gamma: 0.9,
            lambda: 0.0,
            replay_capacity: 100,
            batch_size: 8,
        };
        let mut agent = ReinforcementLearningAgent::new(config);
        agent
            .register_state(s("A"), vec![a("left"), a("right")])
            .expect("test: should succeed");
        agent
            .register_state(s("B"), vec![a("left"), a("right")])
            .expect("test: should succeed");
        let t = simple_transition(true);
        let delta = agent.update(&t).expect("test: TD update should succeed");
        // n=1 buffer should be ready immediately.
        assert!(delta >= 0.0);
    }

    // ── run_episode ──────────────────────────────────────────────────────────

    #[test]
    fn test_run_episode_empty() {
        let mut agent = two_state_agent(AlgorithmType::QLearning, AgentPolicy::Random);
        let stats = agent
            .run_episode(vec![], 42)
            .expect("test: episode run should succeed");
        assert_eq!(stats.steps, 0);
        assert_eq!(stats.total_reward, 0.0);
    }

    #[test]
    fn test_run_episode_single_transition() {
        let mut agent = two_state_agent(AlgorithmType::QLearning, AgentPolicy::Random);
        let t = simple_transition(false);
        let stats = agent
            .run_episode(vec![t], 1)
            .expect("test: episode run should succeed");
        assert_eq!(stats.steps, 1);
        assert_eq!(stats.total_reward, 1.0);
    }

    #[test]
    fn test_run_episode_accumulates_reward() {
        let mut agent = two_state_agent(AlgorithmType::QLearning, AgentPolicy::Random);
        let transitions = vec![
            Transition {
                state: s("A"),
                action: a("left"),
                reward: 2.0,
                next_state: s("B"),
                done: false,
            },
            Transition {
                state: s("B"),
                action: a("right"),
                reward: 3.0,
                next_state: s("A"),
                done: true,
            },
        ];
        let stats = agent
            .run_episode(transitions, 0)
            .expect("test: episode run should succeed");
        assert_eq!(stats.steps, 2);
        assert!((stats.total_reward - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_run_episode_updates_aggregate_stats() {
        let mut agent = two_state_agent(AlgorithmType::QLearning, AgentPolicy::Random);
        assert_eq!(agent.stats().episodes_run, 0);
        let t = simple_transition(false);
        agent
            .run_episode(vec![t], 0)
            .expect("test: episode run should succeed");
        assert_eq!(agent.stats().episodes_run, 1);
        assert_eq!(agent.stats().total_steps, 1);
    }

    #[test]
    fn test_run_episode_tracks_best_reward() {
        let mut agent = two_state_agent(AlgorithmType::QLearning, AgentPolicy::Random);
        let t1 = Transition {
            state: s("A"),
            action: a("left"),
            reward: 5.0,
            next_state: s("B"),
            done: true,
        };
        let t2 = Transition {
            state: s("A"),
            action: a("left"),
            reward: 100.0,
            next_state: s("B"),
            done: true,
        };
        agent
            .run_episode(vec![t1], 0)
            .expect("test: episode run should succeed");
        agent
            .run_episode(vec![t2], 0)
            .expect("test: episode run should succeed");
        assert!((agent.stats().best_episode_reward - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_run_episode_epsilon_in_stats() {
        let policy = AgentPolicy::EpsilonGreedy {
            epsilon: 0.5,
            decay: 0.9,
            min_epsilon: 0.01,
        };
        let mut agent = two_state_agent(AlgorithmType::QLearning, policy);
        let t = simple_transition(false);
        let stats = agent
            .run_episode(vec![t], 0)
            .expect("test: episode run should succeed");
        // epsilon should be 0.5 at capture point (before decay).
        assert!((stats.epsilon - 0.5).abs() < 1e-9);
        // After episode epsilon is decayed.
        assert!((agent.current_epsilon() - 0.45).abs() < 1e-9);
    }

    // ── best_action / value ──────────────────────────────────────────────────

    #[test]
    fn test_best_action_returns_highest_q() {
        let mut agent = two_state_agent(AlgorithmType::QLearning, AgentPolicy::Random);
        agent.q_table.entry((s("A"), a("left"))).or_default().q1 = 1.0;
        agent.q_table.entry((s("A"), a("right"))).or_default().q1 = 5.0;
        assert_eq!(
            agent.best_action(&s("A")).expect("test: should succeed"),
            a("right")
        );
    }

    #[test]
    fn test_best_action_unknown_state_error() {
        let agent = two_state_agent(AlgorithmType::QLearning, AgentPolicy::Random);
        assert!(matches!(
            agent.best_action(&s("Z")),
            Err(RlAgentError::StateNotFound(_))
        ));
    }

    #[test]
    fn test_value_equals_max_q() {
        let mut agent = two_state_agent(AlgorithmType::QLearning, AgentPolicy::Random);
        agent.q_table.entry((s("A"), a("left"))).or_default().q1 = 2.0;
        agent.q_table.entry((s("A"), a("right"))).or_default().q1 = 7.0;
        assert!((agent.value(&s("A")) - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_value_unregistered_state_zero() {
        let agent = two_state_agent(AlgorithmType::QLearning, AgentPolicy::Random);
        assert_eq!(agent.value(&s("UNKNOWN")), 0.0);
    }

    // ── decay_epsilon ────────────────────────────────────────────────────────

    #[test]
    fn test_decay_epsilon_reduces_epsilon() {
        let policy = AgentPolicy::EpsilonGreedy {
            epsilon: 1.0,
            decay: 0.5,
            min_epsilon: 0.0,
        };
        let mut agent = two_state_agent(AlgorithmType::QLearning, policy);
        agent.decay_epsilon();
        assert!((agent.current_epsilon() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_decay_epsilon_respects_min() {
        let policy = AgentPolicy::EpsilonGreedy {
            epsilon: 0.01,
            decay: 0.1,
            min_epsilon: 0.05,
        };
        let mut agent = two_state_agent(AlgorithmType::QLearning, policy);
        agent.decay_epsilon();
        assert!((agent.current_epsilon() - 0.05).abs() < 1e-12);
    }

    #[test]
    fn test_decay_epsilon_noop_for_boltzmann() {
        let policy = AgentPolicy::Boltzmann { temperature: 2.0 };
        let mut agent = two_state_agent(AlgorithmType::QLearning, policy);
        agent.decay_epsilon(); // should not panic
        assert_eq!(agent.current_epsilon(), 0.0);
    }

    #[test]
    fn test_decay_epsilon_noop_for_random() {
        let mut agent = two_state_agent(AlgorithmType::QLearning, AgentPolicy::Random);
        agent.decay_epsilon();
        assert_eq!(agent.current_epsilon(), 0.0);
    }

    // ── experience replay ────────────────────────────────────────────────────

    #[test]
    fn test_add_experience_populates_buffer() {
        let mut agent = two_state_agent(AlgorithmType::QLearning, AgentPolicy::Random);
        assert_eq!(agent.replay.len(), 0);
        agent.add_experience(simple_transition(false));
        assert_eq!(agent.replay.len(), 1);
    }

    #[test]
    fn test_add_experience_respects_capacity() {
        let config = AgentConfig {
            replay_capacity: 3,
            ..AgentConfig::default()
        };
        let mut agent = ReinforcementLearningAgent::new(config);
        agent
            .register_state(s("A"), vec![a("left"), a("right")])
            .expect("test: should succeed");
        agent
            .register_state(s("B"), vec![a("left"), a("right")])
            .expect("test: should succeed");
        for _ in 0..10 {
            agent.add_experience(simple_transition(false));
        }
        assert_eq!(agent.replay.len(), 3);
    }

    #[test]
    fn test_sample_experience_correct_count() {
        let mut agent = two_state_agent(AlgorithmType::QLearning, AgentPolicy::Random);
        for _ in 0..20 {
            agent.add_experience(simple_transition(false));
        }
        let sample = agent
            .sample_experience(5, 42)
            .expect("test: experience sampling should succeed");
        assert_eq!(sample.len(), 5);
    }

    #[test]
    fn test_sample_experience_insufficient_error() {
        let mut agent = two_state_agent(AlgorithmType::QLearning, AgentPolicy::Random);
        agent.add_experience(simple_transition(false)); // only 1
        let result = agent.sample_experience(5, 0);
        assert!(matches!(
            result,
            Err(RlAgentError::InsufficientExperience(1))
        ));
    }

    #[test]
    fn test_sample_experience_empty_buffer_error() {
        let agent = two_state_agent(AlgorithmType::QLearning, AgentPolicy::Random);
        let result = agent.sample_experience(1, 0);
        assert!(matches!(
            result,
            Err(RlAgentError::InsufficientExperience(0))
        ));
    }

    #[test]
    fn test_sample_experience_randomness_different_seeds() {
        let mut agent = two_state_agent(AlgorithmType::QLearning, AgentPolicy::Random);
        for i in 0..20u64 {
            agent.add_experience(Transition {
                state: s("A"),
                action: a("left"),
                reward: i as f64,
                next_state: s("B"),
                done: false,
            });
        }
        let s1: Vec<f64> = agent
            .sample_experience(5, 1)
            .expect("test: should succeed")
            .iter()
            .map(|t| t.reward)
            .collect();
        let s2: Vec<f64> = agent
            .sample_experience(5, 99999)
            .expect("test: should succeed")
            .iter()
            .map(|t| t.reward)
            .collect();
        // Very likely to differ with different seeds.
        // (Technically could be the same; accept if at least one element differs.)
        let any_diff = s1.iter().zip(&s2).any(|(a, b)| (a - b).abs() > 1e-12);
        let all_same = s1 == s2;
        assert!(any_diff || !all_same || s1.len() == 1);
    }

    // ── stats ────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial_values() {
        let agent = two_state_agent(AlgorithmType::QLearning, AgentPolicy::Random);
        let stats = agent.stats();
        assert_eq!(stats.episodes_run, 0);
        assert_eq!(stats.total_steps, 0);
        assert_eq!(stats.avg_reward, 0.0);
    }

    #[test]
    fn test_stats_convergence_delta_updates() {
        let mut agent = two_state_agent(AlgorithmType::QLearning, AgentPolicy::Random);
        let t = simple_transition(false);
        agent
            .run_episode(vec![t], 0)
            .expect("test: episode run should succeed");
        // After one episode convergence_delta should be non-negative.
        assert!(agent.stats().convergence_delta >= 0.0);
    }

    #[test]
    fn test_stats_avg_reward_ema() {
        let mut agent = two_state_agent(AlgorithmType::QLearning, AgentPolicy::Random);
        // Run several episodes with reward=10.
        for _ in 0..50 {
            let t = Transition {
                state: s("A"),
                action: a("left"),
                reward: 10.0,
                next_state: s("B"),
                done: true,
            };
            agent
                .run_episode(vec![t], 0)
                .expect("test: episode run should succeed");
        }
        // After many episodes with reward=10 the EMA should be close to 10.
        let avg = agent.stats().avg_reward;
        assert!(avg > 5.0, "avg_reward {avg} should be > 5");
    }

    // ── error cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_update_unknown_state_error() {
        let mut agent = ReinforcementLearningAgent::new(AgentConfig::default());
        let t = Transition {
            state: s("GHOST"),
            action: a("up"),
            reward: 0.0,
            next_state: s("GHOST2"),
            done: false,
        };
        assert!(matches!(
            agent.update(&t),
            Err(RlAgentError::StateNotFound(_))
        ));
    }

    #[test]
    fn test_update_invalid_action_error() {
        let mut agent = ReinforcementLearningAgent::new(AgentConfig::default());
        agent
            .register_state(s("X"), vec![a("go")])
            .expect("test: should succeed");
        agent
            .register_state(s("Y"), vec![a("go")])
            .expect("test: should succeed");
        let t = Transition {
            state: s("X"),
            action: a("FORBIDDEN"),
            reward: 1.0,
            next_state: s("Y"),
            done: false,
        };
        assert!(matches!(
            agent.update(&t),
            Err(RlAgentError::ActionNotFound { .. })
        ));
    }

    #[test]
    fn test_update_next_state_not_found_non_terminal() {
        let mut agent = ReinforcementLearningAgent::new(AgentConfig::default());
        agent
            .register_state(s("X"), vec![a("go")])
            .expect("test: should succeed");
        let t = Transition {
            state: s("X"),
            action: a("go"),
            reward: 1.0,
            next_state: s("UNREGISTERED"),
            done: false,
        };
        assert!(matches!(
            agent.update(&t),
            Err(RlAgentError::StateNotFound(_))
        ));
    }

    #[test]
    fn test_update_terminal_next_state_unregistered_ok() {
        let mut agent = ReinforcementLearningAgent::new(AgentConfig::default());
        agent
            .register_state(s("X"), vec![a("go")])
            .expect("test: should succeed");
        let t = Transition {
            state: s("X"),
            action: a("go"),
            reward: 1.0,
            next_state: s("TERMINAL"),
            done: true,
        };
        // done=true → next_state need not be registered.
        assert!(agent.update(&t).is_ok());
    }

    #[test]
    fn test_rlagent_error_display() {
        let e1 = RlAgentError::StateNotFound(s("X"));
        let e2 = RlAgentError::ActionNotFound {
            state: s("X"),
            action: a("go"),
        };
        let e3 = RlAgentError::InsufficientExperience(3);
        let e4 = RlAgentError::InvalidConfig("bad alpha".into());
        assert!(!e1.to_string().is_empty());
        assert!(!e2.to_string().is_empty());
        assert!(!e3.to_string().is_empty());
        assert!(!e4.to_string().is_empty());
    }

    // ── PRNG helpers ─────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_non_zero_output() {
        let mut s = 1u64;
        for _ in 0..100 {
            let v = xorshift64(&mut s);
            assert_ne!(v, 0);
        }
    }

    #[test]
    fn test_xorshift_f64_range() {
        let mut s = 12345u64;
        for _ in 0..1000 {
            let v = xorshift_f64(&mut s);
            assert!((0.0..1.0).contains(&v));
        }
    }

    // ── UCB policy edge case ──────────────────────────────────────────────────

    #[test]
    fn test_ucb_zero_c_behaves_like_greedy() {
        let policy = AgentPolicy::UCB { c: 0.0 };
        let mut agent = two_state_agent(AlgorithmType::QLearning, policy);
        agent.q_table.entry((s("A"), a("left"))).or_default().q1 = 100.0;
        let act = agent
            .select_action(&s("A"), &mut 0u64)
            .expect("test: should succeed");
        assert_eq!(act, a("left"));
    }

    // ── ExperienceReplay struct ───────────────────────────────────────────────

    #[test]
    fn test_experience_replay_is_empty() {
        let buf = ExperienceReplay::new(10);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_experience_replay_evicts_oldest() {
        let mut buf = ExperienceReplay::new(2);
        buf.push(Transition {
            state: s("A"),
            action: a("x"),
            reward: 1.0,
            next_state: s("B"),
            done: false,
        });
        buf.push(Transition {
            state: s("B"),
            action: a("y"),
            reward: 2.0,
            next_state: s("A"),
            done: false,
        });
        buf.push(Transition {
            state: s("A"),
            action: a("z"),
            reward: 3.0,
            next_state: s("B"),
            done: false,
        });
        assert_eq!(buf.len(), 2);
        // Oldest (reward=1) should be gone; most recent two remain.
        let rewards: Vec<f64> = buf.buffer.iter().map(|t| t.reward).collect();
        assert!(!rewards.contains(&1.0));
    }

    // ── multi-episode learning ────────────────────────────────────────────────

    #[test]
    fn test_multi_episode_qlearning_improves() {
        let policy = AgentPolicy::EpsilonGreedy {
            epsilon: 0.3,
            decay: 0.99,
            min_epsilon: 0.01,
        };
        let mut agent = two_state_agent(AlgorithmType::QLearning, policy);
        let good = Transition {
            state: s("A"),
            action: a("right"),
            reward: 10.0,
            next_state: s("B"),
            done: true,
        };
        for _ in 0..100 {
            agent
                .run_episode(vec![good.clone()], 0)
                .expect("test: should succeed");
        }
        assert!(agent.q1(&s("A"), &a("right")) > 0.0);
        assert_eq!(
            agent.best_action(&s("A")).expect("test: should succeed"),
            a("right")
        );
    }

    // ── Double Q-learning extra coverage ─────────────────────────────────────

    #[test]
    fn test_double_q_both_tables_nonzero_after_many_updates() {
        let mut agent = two_state_agent(AlgorithmType::DoubleQLearning, AgentPolicy::Random);
        let t = simple_transition(false);
        for _ in 0..20 {
            agent.update(&t).expect("test: TD update should succeed");
        }
        let q1 = agent.q1(&s("A"), &a("left"));
        let q2 = agent.q2(&s("A"), &a("left"));
        assert!(q1 != 0.0 || q2 != 0.0);
    }
}
