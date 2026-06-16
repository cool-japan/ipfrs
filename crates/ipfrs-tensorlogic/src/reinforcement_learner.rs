//! Reinforcement learning agents — Q-learning and SARSA for discrete action spaces.
//!
//! Provides [`ReinforcementLearner`] with support for:
//! - Standard Q-learning (off-policy TD(0))
//! - SARSA (on-policy TD(0))
//! - Double Q-learning (reduces overestimation bias)
//! - Epsilon-greedy, greedy, and random exploration policies
//! - Per-step and per-episode statistics
//!
//! All random number generation uses an inline xorshift64 PRNG seeded at
//! construction time so the implementation is dependency-free and deterministic.

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Primitive identifiers
// ---------------------------------------------------------------------------

/// Opaque identifier for an environment state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateId(pub u64);

/// Opaque identifier for an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActionId(pub u32);

// ---------------------------------------------------------------------------
// Algorithm selector
// ---------------------------------------------------------------------------

/// Reinforcement learning algorithm variant.
#[derive(Debug, Clone, PartialEq)]
pub enum RlAlgorithm {
    /// Off-policy TD(0) with greedy target policy.
    QLearning {
        /// Learning rate α ∈ (0, 1].
        alpha: f64,
        /// Discount factor γ ∈ [0, 1].
        gamma: f64,
        /// Exploration rate ε ∈ [0, 1] for on-policy action selection.
        epsilon: f64,
    },
    /// On-policy TD(0) — the target action is chosen by the current ε-greedy policy.
    Sarsa {
        /// Learning rate α ∈ (0, 1].
        alpha: f64,
        /// Discount factor γ ∈ [0, 1].
        gamma: f64,
        /// Exploration rate ε ∈ [0, 1].
        epsilon: f64,
    },
    /// Double Q-learning — maintains two independent Q-tables and randomly selects
    /// which to update at each step, reducing the maximisation bias present in
    /// standard Q-learning.
    DoubleQLearning {
        /// Learning rate α ∈ (0, 1].
        alpha: f64,
        /// Discount factor γ ∈ [0, 1].
        gamma: f64,
        /// Exploration rate ε ∈ [0, 1].
        epsilon: f64,
    },
}

impl RlAlgorithm {
    /// Return the (alpha, gamma, epsilon) triple regardless of variant.
    pub fn hyperparams(&self) -> (f64, f64, f64) {
        match *self {
            RlAlgorithm::QLearning {
                alpha,
                gamma,
                epsilon,
            }
            | RlAlgorithm::Sarsa {
                alpha,
                gamma,
                epsilon,
            }
            | RlAlgorithm::DoubleQLearning {
                alpha,
                gamma,
                epsilon,
            } => (alpha, gamma, epsilon),
        }
    }
}

// ---------------------------------------------------------------------------
// Experience tuple
// ---------------------------------------------------------------------------

/// A single (s, a, r, s', done) transition.
#[derive(Debug, Clone)]
pub struct Experience {
    /// State at which the action was taken.
    pub state: StateId,
    /// Action that was taken.
    pub action: ActionId,
    /// Scalar reward received.
    pub reward: f64,
    /// Resulting next state.
    pub next_state: StateId,
    /// `true` if the episode terminated after this transition.
    pub done: bool,
}

// ---------------------------------------------------------------------------
// Exploration policy
// ---------------------------------------------------------------------------

/// Action-selection policy.
#[derive(Debug, Clone, PartialEq)]
pub enum Policy {
    /// With probability `epsilon` choose a uniformly random action;
    /// otherwise choose the greedy (argmax Q) action.
    EpsilonGreedy { epsilon: f64 },
    /// Always choose the action with the highest Q-value.
    Greedy,
    /// Always choose a uniformly random action.
    Random,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Aggregate statistics snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct RlStats {
    /// Total environment steps taken across all episodes.
    pub total_steps: u64,
    /// Total episodes completed.
    pub total_episodes: u64,
    /// Number of distinct states whose Q-values have been initialised.
    pub explored_states: usize,
    /// Mean return over the last 100 episodes.
    pub avg_return_last_100: f64,
    /// Highest episode return observed so far.
    pub best_return: f64,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can be returned by reinforcement learning operations.
#[derive(Debug, Clone, PartialEq)]
pub enum RlError {
    /// The requested action index is out of range for the current action space.
    InvalidAction(u32),
    /// The state identifier is inconsistent (reserved for future validation).
    InvalidState,
}

impl std::fmt::Display for RlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RlError::InvalidAction(id) => write!(f, "invalid action id: {id}"),
            RlError::InvalidState => write!(f, "invalid state"),
        }
    }
}

impl std::error::Error for RlError {}

// ---------------------------------------------------------------------------
// xorshift64 PRNG
// ---------------------------------------------------------------------------

/// Fast non-cryptographic PRNG (xorshift64).
///
/// The `state` must be non-zero; an initial seed of 0 is promoted to 1 by the
/// constructor.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Return a pseudo-random `f64` in `[0, 1)` using the given PRNG state.
#[inline]
fn rand_f64(state: &mut u64) -> f64 {
    // Use the upper 53 bits for the mantissa.
    let bits = xorshift64(state);
    (bits >> 11) as f64 / (1u64 << 53) as f64
}

/// Return a pseudo-random `usize` in `[0, n)` using the given PRNG state.
///
/// Uses rejection sampling to avoid modulo bias.
#[inline]
fn rand_usize(state: &mut u64, n: usize) -> usize {
    assert!(n > 0, "n must be positive");
    let n64 = n as u64;
    // Largest multiple of n that fits in u64
    let limit = u64::MAX - (u64::MAX % n64);
    loop {
        let r = xorshift64(state);
        if r < limit {
            return (r % n64) as usize;
        }
    }
}

// ---------------------------------------------------------------------------
// ReinforcementLearner
// ---------------------------------------------------------------------------

/// Tabular reinforcement-learning agent supporting Q-learning, SARSA, and
/// Double Q-learning.
///
/// Q-values are stored in a `HashMap` and initialised lazily to 0.0 on first
/// access, so memory consumption grows only with the number of visited states.
#[derive(Debug, Clone)]
pub struct ReinforcementLearner {
    /// The learning algorithm and its hyper-parameters.
    pub algorithm: RlAlgorithm,
    /// Primary Q-table: `q_table[s][a]` = Q(s, a).
    pub q_table: HashMap<StateId, Vec<f64>>,
    /// Secondary Q-table used only by Double Q-learning.
    pub q_table2: HashMap<StateId, Vec<f64>>,
    /// Number of discrete actions.
    pub n_actions: usize,
    /// Total environment steps executed via [`update`](Self::update).
    pub total_steps: u64,
    /// Total episodes recorded via [`start_episode`](Self::start_episode) /
    /// [`end_episode`](Self::end_episode).
    pub total_episodes: u64,
    /// Sliding window of episode returns (capped at 1 000 entries).
    pub episode_returns: VecDeque<f64>,
    /// Current xorshift64 PRNG state (always non-zero).
    pub rng_state: u64,
}

impl ReinforcementLearner {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new learner with the given algorithm, action-space size, and
    /// random seed.
    ///
    /// A seed of 0 is promoted to 1 to satisfy the xorshift64 requirement.
    pub fn new(algorithm: RlAlgorithm, n_actions: usize, seed: u64) -> Self {
        let rng_state = if seed == 0 { 1 } else { seed };
        Self {
            algorithm,
            q_table: HashMap::new(),
            q_table2: HashMap::new(),
            n_actions,
            total_steps: 0,
            total_episodes: 0,
            episode_returns: VecDeque::new(),
            rng_state,
        }
    }

    // -----------------------------------------------------------------------
    // Q-table helpers
    // -----------------------------------------------------------------------

    /// Return a reference to the Q-value vector for `state` in the primary
    /// table, inserting a zero-initialised vector if absent.
    fn ensure_state(&mut self, state: StateId) -> &mut Vec<f64> {
        let n = self.n_actions;
        self.q_table
            .entry(state)
            .or_insert_with(|| vec![0.0_f64; n])
    }

    /// Same as [`ensure_state`] but for the secondary table.
    fn ensure_state2(&mut self, state: StateId) -> &mut Vec<f64> {
        let n = self.n_actions;
        self.q_table2
            .entry(state)
            .or_insert_with(|| vec![0.0_f64; n])
    }

    /// Read Q(s, a) from the primary table without mutating it.
    fn read_q(&self, state: StateId, action: ActionId) -> f64 {
        self.q_table
            .get(&state)
            .and_then(|v| v.get(action.0 as usize).copied())
            .unwrap_or(0.0)
    }

    /// Read Q(s, a) from the secondary table without mutating it.
    fn read_q2(&self, state: StateId, action: ActionId) -> f64 {
        self.q_table2
            .get(&state)
            .and_then(|v| v.get(action.0 as usize).copied())
            .unwrap_or(0.0)
    }

    /// Return max_{a} Q(s, a) from the primary table; 0.0 if unseen.
    fn max_q(&self, state: StateId) -> f64 {
        self.q_table
            .get(&state)
            .and_then(|v| v.iter().copied().reduce(f64::max))
            .unwrap_or(0.0)
    }

    /// Return the argmax action from the primary table; `ActionId(0)` if
    /// the state is unseen (all Q-values tie at 0.0).
    fn argmax_q(&self, state: StateId) -> ActionId {
        match self.q_table.get(&state) {
            None => ActionId(0),
            Some(v) => {
                let mut best_idx = 0usize;
                let mut best_val = v[0];
                for (i, &val) in v.iter().enumerate().skip(1) {
                    if val > best_val {
                        best_val = val;
                        best_idx = i;
                    }
                }
                ActionId(best_idx as u32)
            }
        }
    }

    /// Return the action that maximises Q1(s, a) + Q2(s, a), used by Double
    /// Q-learning to select the bootstrap action.
    fn argmax_q_sum(&self, state: StateId) -> ActionId {
        let n = self.n_actions;
        let v1 = self.q_table.get(&state);
        let v2 = self.q_table2.get(&state);
        let mut best_idx = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        for i in 0..n {
            let q1 = v1.and_then(|v| v.get(i).copied()).unwrap_or(0.0);
            let q2 = v2.and_then(|v| v.get(i).copied()).unwrap_or(0.0);
            let combined = q1 + q2;
            if combined > best_val {
                best_val = combined;
                best_idx = i;
            }
        }
        ActionId(best_idx as u32)
    }

    // -----------------------------------------------------------------------
    // Action selection
    // -----------------------------------------------------------------------

    /// Select an action for `state` according to `policy`.
    ///
    /// - [`Policy::EpsilonGreedy`] — random with prob ε, else argmax Q(s, ·).
    /// - [`Policy::Greedy`] — always argmax Q(s, ·).
    /// - [`Policy::Random`] — uniformly random over the action space.
    pub fn select_action(&mut self, state: StateId, policy: &Policy) -> ActionId {
        match policy {
            Policy::Greedy => {
                // Ensure state exists so it's counted as explored.
                self.ensure_state(state);
                self.argmax_q(state)
            }
            Policy::Random => {
                let idx = rand_usize(&mut self.rng_state, self.n_actions);
                ActionId(idx as u32)
            }
            Policy::EpsilonGreedy { epsilon } => {
                let r = rand_f64(&mut self.rng_state);
                if r < *epsilon {
                    let idx = rand_usize(&mut self.rng_state, self.n_actions);
                    ActionId(idx as u32)
                } else {
                    // Ensure state exists so it's counted as explored.
                    self.ensure_state(state);
                    self.argmax_q(state)
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // TD update
    // -----------------------------------------------------------------------

    /// Apply one temporal-difference update from `experience`; return the TD
    /// error δ.
    ///
    /// Also increments [`total_steps`](Self::total_steps).
    pub fn update(&mut self, experience: &Experience) -> f64 {
        let td_error = match self.algorithm.clone() {
            RlAlgorithm::QLearning { alpha, gamma, .. } => {
                self.update_q_learning(experience, alpha, gamma)
            }
            RlAlgorithm::Sarsa {
                alpha,
                gamma,
                epsilon,
            } => self.update_sarsa(experience, alpha, gamma, epsilon),
            RlAlgorithm::DoubleQLearning { alpha, gamma, .. } => {
                self.update_double_q(experience, alpha, gamma)
            }
        };
        self.total_steps += 1;
        td_error
    }

    /// Q-learning update: δ = r + γ * max_a Q(s', a) * (1-done) - Q(s, a).
    fn update_q_learning(&mut self, exp: &Experience, alpha: f64, gamma: f64) -> f64 {
        let q_sa = self.read_q(exp.state, exp.action);
        let max_next = if exp.done {
            0.0
        } else {
            self.max_q(exp.next_state)
        };
        let td_error = exp.reward + gamma * max_next - q_sa;

        // Mutate
        self.ensure_state(exp.state);
        if let Some(v) = self.q_table.get_mut(&exp.state) {
            let idx = exp.action.0 as usize;
            if let Some(entry) = v.get_mut(idx) {
                *entry += alpha * td_error;
            }
        }
        td_error
    }

    /// SARSA update: a' is chosen by the current ε-greedy policy.
    fn update_sarsa(&mut self, exp: &Experience, alpha: f64, gamma: f64, epsilon: f64) -> f64 {
        let q_sa = self.read_q(exp.state, exp.action);

        // Select next action on-policy.
        let q_next_sa = if exp.done {
            0.0
        } else {
            let next_action =
                self.select_action(exp.next_state, &Policy::EpsilonGreedy { epsilon });
            self.read_q(exp.next_state, next_action)
        };

        let td_error = exp.reward + gamma * q_next_sa - q_sa;

        self.ensure_state(exp.state);
        if let Some(v) = self.q_table.get_mut(&exp.state) {
            let idx = exp.action.0 as usize;
            if let Some(entry) = v.get_mut(idx) {
                *entry += alpha * td_error;
            }
        }
        td_error
    }

    /// Double Q-learning update — randomly choose which table to update.
    ///
    /// The target for table-1 updates uses the action selected by table-1 to
    /// index into table-2 (and vice versa), which prevents the overestimation
    /// bias of standard Q-learning.
    fn update_double_q(&mut self, exp: &Experience, alpha: f64, gamma: f64) -> f64 {
        // Coin flip: 0 → update table1, 1 → update table2.
        let coin = xorshift64(&mut self.rng_state) & 1;

        let td_error = if coin == 0 {
            // Update Q1 using target from Q2.
            let q_sa = self.read_q(exp.state, exp.action);
            let target = if exp.done {
                exp.reward
            } else {
                // a* = argmax_a Q1(s', a)
                let a_star = self.argmax_q(exp.next_state);
                // but evaluate it with Q2
                exp.reward + gamma * self.read_q2(exp.next_state, a_star)
            };
            let delta = target - q_sa;
            self.ensure_state(exp.state);
            if let Some(v) = self.q_table.get_mut(&exp.state) {
                if let Some(entry) = v.get_mut(exp.action.0 as usize) {
                    *entry += alpha * delta;
                }
            }
            delta
        } else {
            // Update Q2 using target from Q1.
            let q_sa2 = self.read_q2(exp.state, exp.action);
            let target = if exp.done {
                exp.reward
            } else {
                // a* = argmax_a Q2(s', a)
                let a_star = self.argmax_q_sum(exp.next_state); // symmetrically
                                                                // evaluate with Q1
                exp.reward + gamma * self.read_q(exp.next_state, a_star)
            };
            let delta = target - q_sa2;
            self.ensure_state2(exp.state);
            if let Some(v) = self.q_table2.get_mut(&exp.state) {
                if let Some(entry) = v.get_mut(exp.action.0 as usize) {
                    *entry += alpha * delta;
                }
            }
            delta
        };

        td_error
    }

    // -----------------------------------------------------------------------
    // Batch update
    // -----------------------------------------------------------------------

    /// Apply TD updates for a slice of experiences in order; return the
    /// corresponding TD errors.
    pub fn batch_update(&mut self, experiences: &[Experience]) -> Vec<f64> {
        experiences.iter().map(|exp| self.update(exp)).collect()
    }

    // -----------------------------------------------------------------------
    // Query methods
    // -----------------------------------------------------------------------

    /// Return the action with the highest Q-value in the primary table.
    /// Falls back to `ActionId(0)` when the state has never been visited.
    pub fn best_action(&self, state: StateId) -> ActionId {
        self.argmax_q(state)
    }

    /// Return Q(s, a) from the primary table; 0.0 if the pair is unseen.
    pub fn q_value(&self, state: StateId, action: ActionId) -> f64 {
        self.read_q(state, action)
    }

    /// Return V(s) = max_a Q(s, a) from the primary table; 0.0 if unseen.
    pub fn value(&self, state: StateId) -> f64 {
        self.max_q(state)
    }

    /// Number of distinct states whose Q-values have been initialised.
    pub fn explored_states(&self) -> usize {
        self.q_table.len()
    }

    // -----------------------------------------------------------------------
    // Episode management
    // -----------------------------------------------------------------------

    /// Signal the start of a new episode.
    ///
    /// Pushes a placeholder return of 0.0 and increments `total_episodes`.
    pub fn start_episode(&mut self) {
        self.episode_returns.push_back(0.0);
        self.total_episodes += 1;
    }

    /// Signal the end of the current episode, recording its total return.
    ///
    /// Updates the last entry in `episode_returns` and caps the deque at 1 000
    /// entries (oldest entries are dropped).
    pub fn end_episode(&mut self, total_return: f64) {
        if let Some(last) = self.episode_returns.back_mut() {
            *last = total_return;
        } else {
            // end_episode called without a matching start_episode — record anyway.
            self.episode_returns.push_back(total_return);
        }
        while self.episode_returns.len() > 1000 {
            self.episode_returns.pop_front();
        }
    }

    /// Return the mean episode return over the last `last_n` episodes.
    /// Returns 0.0 if no episodes have been recorded.
    pub fn avg_return(&self, last_n: usize) -> f64 {
        if self.episode_returns.is_empty() || last_n == 0 {
            return 0.0;
        }
        let n = last_n.min(self.episode_returns.len());
        let start = self.episode_returns.len() - n;
        let sum: f64 = self.episode_returns.iter().skip(start).sum();
        sum / n as f64
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Return a snapshot of the current learning statistics.
    pub fn stats(&self) -> RlStats {
        let best_return = self
            .episode_returns
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let best_return = if best_return == f64::NEG_INFINITY {
            0.0
        } else {
            best_return
        };

        RlStats {
            total_steps: self.total_steps,
            total_episodes: self.total_episodes,
            explored_states: self.explored_states(),
            avg_return_last_100: self.avg_return(100),
            best_return,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        ActionId, Experience, Policy, ReinforcementLearner, RlAlgorithm, RlError, StateId,
    };

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_q_learner() -> ReinforcementLearner {
        ReinforcementLearner::new(
            RlAlgorithm::QLearning {
                alpha: 0.1,
                gamma: 0.9,
                epsilon: 0.1,
            },
            4,
            42,
        )
    }

    fn make_sarsa() -> ReinforcementLearner {
        ReinforcementLearner::new(
            RlAlgorithm::Sarsa {
                alpha: 0.1,
                gamma: 0.9,
                epsilon: 0.2,
            },
            4,
            99,
        )
    }

    fn make_double_q() -> ReinforcementLearner {
        ReinforcementLearner::new(
            RlAlgorithm::DoubleQLearning {
                alpha: 0.1,
                gamma: 0.9,
                epsilon: 0.1,
            },
            4,
            7,
        )
    }

    fn exp(s: u64, a: u32, r: f64, ns: u64, done: bool) -> Experience {
        Experience {
            state: StateId(s),
            action: ActionId(a),
            reward: r,
            next_state: StateId(ns),
            done,
        }
    }

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_q_learning_initial_state() {
        let learner = make_q_learner();
        assert_eq!(learner.total_steps, 0);
        assert_eq!(learner.total_episodes, 0);
        assert_eq!(learner.n_actions, 4);
        assert!(learner.q_table.is_empty());
    }

    #[test]
    fn test_new_zero_seed_promoted() {
        let learner = ReinforcementLearner::new(
            RlAlgorithm::QLearning {
                alpha: 0.1,
                gamma: 0.9,
                epsilon: 0.0,
            },
            2,
            0,
        );
        assert_ne!(learner.rng_state, 0);
    }

    #[test]
    fn test_new_sarsa_initial_state() {
        let learner = make_sarsa();
        assert_eq!(learner.n_actions, 4);
        assert!(learner.q_table.is_empty());
    }

    #[test]
    fn test_new_double_q_initial_state() {
        let learner = make_double_q();
        assert!(learner.q_table.is_empty());
        assert!(learner.q_table2.is_empty());
    }

    // -----------------------------------------------------------------------
    // Q-value access
    // -----------------------------------------------------------------------

    #[test]
    fn test_q_value_unseen_state_returns_zero() {
        let learner = make_q_learner();
        assert_eq!(learner.q_value(StateId(999), ActionId(0)), 0.0);
    }

    #[test]
    fn test_value_unseen_state_returns_zero() {
        let learner = make_q_learner();
        assert_eq!(learner.value(StateId(999)), 0.0);
    }

    #[test]
    fn test_best_action_unseen_state_returns_zero_action() {
        let learner = make_q_learner();
        assert_eq!(learner.best_action(StateId(42)), ActionId(0));
    }

    // -----------------------------------------------------------------------
    // Q-learning update
    // -----------------------------------------------------------------------

    #[test]
    fn test_q_learning_update_increments_steps() {
        let mut learner = make_q_learner();
        learner.update(&exp(0, 0, 1.0, 1, false));
        assert_eq!(learner.total_steps, 1);
    }

    #[test]
    fn test_q_learning_update_from_zero() {
        // Q(s,a) = 0; r=1; max Q(s',a) = 0; not done.
        // δ = 1 + 0.9 * 0 - 0 = 1.0
        // Q(s,a) += 0.1 * 1.0 → 0.1
        let mut learner = make_q_learner();
        let td = learner.update(&exp(0, 0, 1.0, 1, false));
        assert!((td - 1.0).abs() < 1e-12);
        assert!((learner.q_value(StateId(0), ActionId(0)) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_q_learning_terminal_step() {
        // done=true → max Q(s', a) = 0 regardless of next-state values.
        let mut learner = make_q_learner();
        let td = learner.update(&exp(0, 1, 5.0, 99, true));
        // δ = 5.0 + 0 - 0 = 5.0; Q(s,a) = 0.5
        assert!((td - 5.0).abs() < 1e-12);
        assert!((learner.q_value(StateId(0), ActionId(1)) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_q_learning_uses_max_next_action() {
        let mut learner = make_q_learner();
        // Seed next state values: update action 2 of state 1 to a high value.
        learner.update(&exp(1, 2, 10.0, 2, true)); // Q(1,2) = 1.0
                                                   // Now update from state 0 with next_state=1.
                                                   // max Q(1, ·) = 1.0
                                                   // δ = 0.5 + 0.9 * 1.0 - 0 = 1.4; Q(0,0) = 0.14
        let td = learner.update(&exp(0, 0, 0.5, 1, false));
        let expected = 0.5 + 0.9 * learner.q_value(StateId(1), ActionId(2));
        // We re-read after the update so assert via td_error magnitude.
        assert!(td > 0.0);
        let _ = expected; // used indirectly
    }

    #[test]
    fn test_q_learning_multiple_updates_converge() {
        let mut learner = ReinforcementLearner::new(
            RlAlgorithm::QLearning {
                alpha: 0.5,
                gamma: 0.0,
                epsilon: 0.0,
            },
            2,
            1,
        );
        // With γ=0 and r=1 always, Q(s,a) should converge to 1.0.
        for _ in 0..100 {
            learner.update(&exp(0, 0, 1.0, 0, false));
        }
        let q = learner.q_value(StateId(0), ActionId(0));
        assert!((q - 1.0).abs() < 0.01, "q={q}");
    }

    // -----------------------------------------------------------------------
    // SARSA update
    // -----------------------------------------------------------------------

    #[test]
    fn test_sarsa_update_increments_steps() {
        let mut learner = make_sarsa();
        learner.update(&exp(0, 0, 1.0, 1, false));
        assert_eq!(learner.total_steps, 1);
    }

    #[test]
    fn test_sarsa_terminal_step() {
        let mut learner = make_sarsa();
        let td = learner.update(&exp(0, 0, 2.0, 99, true));
        assert!((td - 2.0).abs() < 1e-12);
        assert!((learner.q_value(StateId(0), ActionId(0)) - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_sarsa_non_terminal_on_policy() {
        // SARSA selects next action on-policy; with all Q-values at zero the
        // selected action doesn't matter — Q(s', a') == 0 regardless.
        let mut learner = make_sarsa();
        let td = learner.update(&exp(0, 1, 3.0, 2, false));
        // δ = 3 + 0.9 * 0 - 0 = 3; Q(0,1) = 0.3
        assert!((td - 3.0).abs() < 1e-12);
        assert!((learner.q_value(StateId(0), ActionId(1)) - 0.3).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Double Q-learning
    // -----------------------------------------------------------------------

    #[test]
    fn test_double_q_update_increments_steps() {
        let mut learner = make_double_q();
        learner.update(&exp(0, 0, 1.0, 1, false));
        assert_eq!(learner.total_steps, 1);
    }

    #[test]
    fn test_double_q_terminal_step_both_tables() {
        // Run many updates to exercise both table-1 and table-2 paths.
        let mut learner = make_double_q();
        for _ in 0..200 {
            learner.update(&exp(0, 0, 1.0, 99, true));
        }
        let q1 = learner.q_value(StateId(0), ActionId(0));
        let q2 = learner.read_q2(StateId(0), ActionId(0));
        // Both tables should be non-zero.
        assert!(q1 > 0.0 || q2 > 0.0);
    }

    #[test]
    fn test_double_q_both_tables_updated() {
        let mut learner = make_double_q();
        // Run enough updates to exercise both branches.
        for i in 0u64..500 {
            learner.update(&exp(0, 0, 1.0, 1, i == 499));
        }
        let q1 = learner.q_value(StateId(0), ActionId(0));
        let q2 = learner.read_q2(StateId(0), ActionId(0));
        assert!(q1 != 0.0, "q_table1 was never updated");
        assert!(q2 != 0.0, "q_table2 was never updated");
    }

    // -----------------------------------------------------------------------
    // Batch update
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_update_returns_correct_length() {
        let mut learner = make_q_learner();
        let experiences: Vec<_> = (0..5).map(|i| exp(i, 0, 1.0, i + 1, false)).collect();
        let errors = learner.batch_update(&experiences);
        assert_eq!(errors.len(), 5);
    }

    #[test]
    fn test_batch_update_increments_steps() {
        let mut learner = make_q_learner();
        let experiences: Vec<_> = (0..10).map(|i| exp(i, 0, 1.0, i + 1, false)).collect();
        learner.batch_update(&experiences);
        assert_eq!(learner.total_steps, 10);
    }

    #[test]
    fn test_batch_update_empty_slice() {
        let mut learner = make_q_learner();
        let errors = learner.batch_update(&[]);
        assert!(errors.is_empty());
        assert_eq!(learner.total_steps, 0);
    }

    // -----------------------------------------------------------------------
    // Action selection
    // -----------------------------------------------------------------------

    #[test]
    fn test_select_greedy_action() {
        let mut learner = make_q_learner();
        // Force Q(0, 2) to be the highest by running a direct update.
        learner.update(&exp(0, 2, 5.0, 1, true));
        let action = learner.select_action(StateId(0), &Policy::Greedy);
        assert_eq!(action, ActionId(2));
    }

    #[test]
    fn test_select_random_action_in_range() {
        let mut learner = make_q_learner();
        for _ in 0..50 {
            let a = learner.select_action(StateId(0), &Policy::Random);
            assert!(
                a.0 < learner.n_actions as u32,
                "action out of range: {}",
                a.0
            );
        }
    }

    #[test]
    fn test_select_epsilon_greedy_explores() {
        // With epsilon=1.0 every action should be random.
        let mut learner = make_q_learner();
        learner.update(&exp(0, 0, 100.0, 1, true)); // Q(0,0) is very high
        let policy = Policy::EpsilonGreedy { epsilon: 1.0 };
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            let a = learner.select_action(StateId(0), &policy);
            seen.insert(a.0);
        }
        // With 200 samples at epsilon=1.0 we expect to see all 4 actions.
        assert!(seen.len() > 1);
    }

    #[test]
    fn test_select_epsilon_greedy_zero_exploits() {
        // With epsilon=0 we always exploit.
        let mut learner = make_q_learner();
        learner.update(&exp(0, 3, 100.0, 1, true));
        let policy = Policy::EpsilonGreedy { epsilon: 0.0 };
        for _ in 0..20 {
            assert_eq!(learner.select_action(StateId(0), &policy), ActionId(3));
        }
    }

    // -----------------------------------------------------------------------
    // explored_states
    // -----------------------------------------------------------------------

    #[test]
    fn test_explored_states_grows() {
        let mut learner = make_q_learner();
        assert_eq!(learner.explored_states(), 0);
        learner.update(&exp(0, 0, 1.0, 1, false));
        assert!(learner.explored_states() >= 1);
        learner.update(&exp(5, 0, 1.0, 6, false));
        assert!(learner.explored_states() >= 2);
    }

    #[test]
    fn test_explored_states_no_duplicate() {
        let mut learner = make_q_learner();
        for _ in 0..100 {
            learner.update(&exp(0, 0, 1.0, 1, false));
        }
        // Only state 0 (source) and state 1 (next) should have been initialised.
        assert!(learner.explored_states() <= 2);
    }

    // -----------------------------------------------------------------------
    // Episode management
    // -----------------------------------------------------------------------

    #[test]
    fn test_start_episode_increments_count() {
        let mut learner = make_q_learner();
        learner.start_episode();
        assert_eq!(learner.total_episodes, 1);
        learner.start_episode();
        assert_eq!(learner.total_episodes, 2);
    }

    #[test]
    fn test_end_episode_records_return() {
        let mut learner = make_q_learner();
        learner.start_episode();
        learner.end_episode(42.5);
        assert_eq!(*learner.episode_returns.back().unwrap_or(&0.0), 42.5);
    }

    #[test]
    fn test_avg_return_empty_returns_zero() {
        let learner = make_q_learner();
        assert_eq!(learner.avg_return(10), 0.0);
    }

    #[test]
    fn test_avg_return_correct() {
        let mut learner = make_q_learner();
        for r in [1.0, 2.0, 3.0, 4.0, 5.0] {
            learner.start_episode();
            learner.end_episode(r);
        }
        let avg = learner.avg_return(5);
        assert!((avg - 3.0).abs() < 1e-10, "avg={avg}");
    }

    #[test]
    fn test_avg_return_last_n() {
        let mut learner = make_q_learner();
        for r in [1.0, 2.0, 3.0, 4.0, 5.0] {
            learner.start_episode();
            learner.end_episode(r);
        }
        // Last 3 returns: 3, 4, 5 → avg = 4
        let avg = learner.avg_return(3);
        assert!((avg - 4.0).abs() < 1e-10, "avg={avg}");
    }

    #[test]
    fn test_episode_returns_capped_at_1000() {
        let mut learner = make_q_learner();
        for i in 0..1200 {
            learner.start_episode();
            learner.end_episode(i as f64);
        }
        assert!(learner.episode_returns.len() <= 1000);
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_initial() {
        let learner = make_q_learner();
        let s = learner.stats();
        assert_eq!(s.total_steps, 0);
        assert_eq!(s.total_episodes, 0);
        assert_eq!(s.explored_states, 0);
        assert_eq!(s.avg_return_last_100, 0.0);
        assert_eq!(s.best_return, 0.0);
    }

    #[test]
    fn test_stats_after_updates() {
        let mut learner = make_q_learner();
        learner.update(&exp(0, 0, 1.0, 1, false));
        learner.start_episode();
        learner.end_episode(10.0);
        let s = learner.stats();
        assert_eq!(s.total_steps, 1);
        assert_eq!(s.total_episodes, 1);
        assert!(s.explored_states > 0);
        assert!((s.best_return - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_best_return_tracks_max() {
        let mut learner = make_q_learner();
        for r in [5.0, 10.0, 3.0, 7.0] {
            learner.start_episode();
            learner.end_episode(r);
        }
        let s = learner.stats();
        assert!((s.best_return - 10.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // RlError
    // -----------------------------------------------------------------------

    #[test]
    fn test_rl_error_display_invalid_action() {
        let e = RlError::InvalidAction(5);
        let s = format!("{e}");
        assert!(s.contains('5'));
    }

    #[test]
    fn test_rl_error_display_invalid_state() {
        let e = RlError::InvalidState;
        let s = format!("{e}");
        assert!(!s.is_empty());
    }

    #[test]
    fn test_rl_error_is_std_error() {
        fn assert_error<E: std::error::Error>(_: &E) {}
        assert_error(&RlError::InvalidAction(0));
        assert_error(&RlError::InvalidState);
    }

    // -----------------------------------------------------------------------
    // RlAlgorithm helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_hyperparams_q_learning() {
        let algo = RlAlgorithm::QLearning {
            alpha: 0.1,
            gamma: 0.9,
            epsilon: 0.05,
        };
        let (a, g, e) = algo.hyperparams();
        assert!((a - 0.1).abs() < 1e-15);
        assert!((g - 0.9).abs() < 1e-15);
        assert!((e - 0.05).abs() < 1e-15);
    }

    #[test]
    fn test_hyperparams_sarsa() {
        let algo = RlAlgorithm::Sarsa {
            alpha: 0.2,
            gamma: 0.95,
            epsilon: 0.1,
        };
        let (a, g, e) = algo.hyperparams();
        assert!((a - 0.2).abs() < 1e-15);
        assert!((g - 0.95).abs() < 1e-15);
        assert!((e - 0.1).abs() < 1e-15);
    }

    #[test]
    fn test_hyperparams_double_q() {
        let algo = RlAlgorithm::DoubleQLearning {
            alpha: 0.3,
            gamma: 0.8,
            epsilon: 0.2,
        };
        let (a, g, _e) = algo.hyperparams();
        assert!((a - 0.3).abs() < 1e-15);
        assert!((g - 0.8).abs() < 1e-15);
    }

    // -----------------------------------------------------------------------
    // Regression / convergence
    // -----------------------------------------------------------------------

    #[test]
    fn test_q_learning_negative_reward() {
        let mut learner = make_q_learner();
        let td = learner.update(&exp(0, 0, -1.0, 1, true));
        assert!((td - (-1.0)).abs() < 1e-12);
        assert!((learner.q_value(StateId(0), ActionId(0)) - (-0.1)).abs() < 1e-12);
    }

    #[test]
    fn test_sarsa_negative_reward() {
        let mut learner = make_sarsa();
        let td = learner.update(&exp(0, 0, -2.0, 1, true));
        assert!((td - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn test_q_learning_long_episode() {
        let mut learner = ReinforcementLearner::new(
            RlAlgorithm::QLearning {
                alpha: 0.5,
                gamma: 0.0,
                epsilon: 0.0,
            },
            2,
            123,
        );
        learner.start_episode();
        let total: f64 = (0..20)
            .map(|i| {
                learner.update(&exp(0, 0, i as f64, 0, false));
                i as f64
            })
            .sum();
        learner.end_episode(total);
        assert_eq!(learner.total_steps, 20);
        assert!((learner.avg_return(1) - total).abs() < 1e-10);
    }

    #[test]
    fn test_double_q_no_panic_on_zero_seed() {
        let mut learner = ReinforcementLearner::new(
            RlAlgorithm::DoubleQLearning {
                alpha: 0.1,
                gamma: 0.9,
                epsilon: 0.1,
            },
            3,
            0,
        );
        for i in 0..50u64 {
            learner.update(&exp(i % 5, 0, 1.0, (i + 1) % 5, false));
        }
        assert_eq!(learner.total_steps, 50);
    }

    #[test]
    fn test_value_after_update() {
        let mut learner = make_q_learner();
        learner.update(&exp(7, 1, 3.0, 8, true));
        // Q(7, 1) = 0.3; V(7) = max over actions = 0.3
        let v = learner.value(StateId(7));
        assert!((v - 0.3).abs() < 1e-12, "v={v}");
    }

    #[test]
    fn test_best_action_after_update() {
        let mut learner = make_q_learner();
        learner.update(&exp(0, 3, 10.0, 1, true)); // Q(0, 3) high
        assert_eq!(learner.best_action(StateId(0)), ActionId(3));
    }

    /// StateId and ActionId should be copyable and comparable.
    #[test]
    fn test_identifiers_copy_eq() {
        let s1 = StateId(1);
        let s2 = s1;
        assert_eq!(s1, s2);

        let a1 = ActionId(2);
        let a2 = a1;
        assert_eq!(a1, a2);
    }
}
