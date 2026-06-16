//! Types, structs, enums, and utility functions for the RL agent.
//!
//! This module is declared by `reinforcement_learning_agent.rs` via `mod rla_types`.

use std::collections::VecDeque;

// ────────────────────────────────────────────────────────────────────────────
// PRNG (no rand crate)
// ────────────────────────────────────────────────────────────────────────────

/// Fast non-cryptographic xorshift64 PRNG step.
#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Draw a uniform f64 in [0, 1) from the xorshift64 state.
#[inline]
pub fn xorshift_f64(state: &mut u64) -> f64 {
    (xorshift64(state) >> 11) as f64 / (1u64 << 53) as f64
}

// ────────────────────────────────────────────────────────────────────────────
// Core newtypes
// ────────────────────────────────────────────────────────────────────────────

/// A named environment state (newtype over String).
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct RlState(pub String);

/// A named action (newtype over String).
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct RlAction(pub String);

// ────────────────────────────────────────────────────────────────────────────
// Transition
// ────────────────────────────────────────────────────────────────────────────

/// A single (s, a, r, s', done) experience tuple.
#[derive(Debug, Clone)]
pub struct Transition {
    /// State in which the action was taken.
    pub state: RlState,
    /// Action taken.
    pub action: RlAction,
    /// Immediate reward received.
    pub reward: f64,
    /// Next state observed.
    pub next_state: RlState,
    /// Whether the episode terminated after this transition.
    pub done: bool,
}

// ────────────────────────────────────────────────────────────────────────────
// Experience replay buffer
// ────────────────────────────────────────────────────────────────────────────

/// Circular replay buffer of environment transitions.
#[derive(Debug, Clone)]
pub struct ExperienceReplay {
    /// Stored transitions (oldest at front).
    pub buffer: VecDeque<Transition>,
    /// Maximum capacity before oldest entries are dropped.
    pub capacity: usize,
}

impl ExperienceReplay {
    /// Create a new replay buffer with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity.min(4096)),
            capacity,
        }
    }

    /// Push a transition, evicting the oldest if at capacity.
    pub fn push(&mut self, t: Transition) {
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(t);
    }

    /// Current number of stored transitions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Return `true` if the buffer contains no transitions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Policy
// ────────────────────────────────────────────────────────────────────────────

/// Action-selection policy used by the agent.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentPolicy {
    /// With probability `epsilon` pick a random action; otherwise pick argmax Q(s,·).
    EpsilonGreedy {
        /// Current exploration probability ∈ [0, 1].
        epsilon: f64,
        /// Multiplicative decay applied each call to `decay_epsilon`.
        decay: f64,
        /// Lower bound for epsilon after decay.
        min_epsilon: f64,
    },
    /// Softmax action selection with temperature τ > 0.
    Boltzmann {
        /// Temperature τ — higher → more uniform; lower → more greedy.
        temperature: f64,
    },
    /// Upper-Confidence-Bound action selection.
    UCB {
        /// Exploration coefficient c ≥ 0.
        c: f64,
    },
    /// Uniform random action selection (pure exploration).
    Random,
}

// ────────────────────────────────────────────────────────────────────────────
// Algorithm
// ────────────────────────────────────────────────────────────────────────────

/// Tabular RL algorithm implemented by the agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlgorithmType {
    /// On-policy TD(0): Q(s,a) += α[r + γ Q(s',a') - Q(s,a)] where a' ~ π.
    Sarsa,
    /// Off-policy TD(0): Q(s,a) += α[r + γ max_a' Q(s',a') - Q(s,a)].
    QLearning,
    /// Expected SARSA: target uses the expectation over π(·|s').
    ExpectedSarsa,
    /// Maintains two Q-tables; alternates which is updated.
    DoubleQLearning,
    /// N-step TD: accumulate n transitions before bootstrapping.
    NStepTD(u8),
}

// ────────────────────────────────────────────────────────────────────────────
// Agent configuration
// ────────────────────────────────────────────────────────────────────────────

/// Full configuration for [`super::ReinforcementLearningAgent`].
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Learning algorithm.
    pub algorithm: AlgorithmType,
    /// Exploration / action-selection policy.
    pub policy: AgentPolicy,
    /// Learning rate α ∈ (0, 1].
    pub alpha: f64,
    /// Discount factor γ ∈ [0, 1].
    pub gamma: f64,
    /// Eligibility-trace decay λ ∈ [0, 1].
    pub lambda: f64,
    /// Replay buffer capacity.
    pub replay_capacity: usize,
    /// Mini-batch size for experience replay sampling.
    pub batch_size: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            algorithm: AlgorithmType::QLearning,
            policy: AgentPolicy::EpsilonGreedy {
                epsilon: 0.1,
                decay: 0.995,
                min_epsilon: 0.01,
            },
            alpha: 0.1,
            gamma: 0.99,
            lambda: 0.9,
            replay_capacity: 10_000,
            batch_size: 32,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Statistics
// ────────────────────────────────────────────────────────────────────────────

/// Statistics snapshot for a single episode.
#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeStats {
    /// Undiscounted sum of rewards in the episode.
    pub total_reward: f64,
    /// Number of steps (transitions) in the episode.
    pub steps: usize,
    /// Current epsilon value at end of episode (0.0 for non-ε-greedy policies).
    pub epsilon: f64,
    /// Mean Q-value over all visited (state, action) pairs.
    pub avg_q_value: f64,
}

/// Aggregate statistics across all episodes.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentStats {
    /// Total episodes processed.
    pub episodes_run: u64,
    /// Total environment steps across all episodes.
    pub total_steps: u64,
    /// Exponential moving average of episode reward.
    pub avg_reward: f64,
    /// Highest single-episode reward observed.
    pub best_episode_reward: f64,
    /// Maximum |ΔQ| observed in the most recent episode.
    pub convergence_delta: f64,
}

// ────────────────────────────────────────────────────────────────────────────
// Error
// ────────────────────────────────────────────────────────────────────────────

/// Errors produced by [`super::ReinforcementLearningAgent`].
#[derive(Debug, Clone, PartialEq)]
pub enum RlAgentError {
    /// The requested state has not been registered.
    StateNotFound(RlState),
    /// The requested action is not valid for the given state.
    ActionNotFound {
        /// State in which the invalid action was requested.
        state: RlState,
        /// The invalid action.
        action: RlAction,
    },
    /// The replay buffer holds fewer entries than the requested sample size.
    InsufficientExperience(usize),
    /// A configuration parameter is invalid.
    InvalidConfig(String),
}

impl std::fmt::Display for RlAgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StateNotFound(s) => write!(f, "state not found: {:?}", s.0),
            Self::ActionNotFound { state, action } => {
                write!(f, "action {:?} not valid for state {:?}", action.0, state.0)
            }
            Self::InsufficientExperience(n) => {
                write!(
                    f,
                    "insufficient replay experience: only {n} transitions stored"
                )
            }
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for RlAgentError {}

// ────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────────────────

/// Per-(state, action) pair visitation and Q-value record.
#[derive(Debug, Clone, Default)]
pub(super) struct QEntry {
    /// Primary Q-value (also used as Q1 for Double Q-learning).
    pub(super) q1: f64,
    /// Secondary Q-value for Double Q-learning (unused for other algorithms).
    pub(super) q2: f64,
    /// Number of times this (s,a) pair has been visited.
    pub(super) visits: u64,
}

/// Pending n-step TD accumulation buffer.
#[derive(Debug, Clone)]
pub(super) struct NStepBuffer {
    /// Transitions accumulated so far.
    pub(super) transitions: VecDeque<Transition>,
    /// N steps to accumulate before bootstrapping.
    pub(super) n: u8,
}

impl NStepBuffer {
    pub(super) fn new(n: u8) -> Self {
        Self {
            transitions: VecDeque::with_capacity(n as usize + 1),
            n,
        }
    }

    /// Return `true` when we have accumulated `n` transitions.
    pub(super) fn ready(&self) -> bool {
        self.transitions.len() >= self.n as usize
    }

    /// Compute G_t^(n) = Σ_{k=0}^{n-1} γ^k r_{t+k} + γ^n Q(s_{t+n}, best).
    pub(super) fn n_step_return(&self, gamma: f64, bootstrap_q: f64) -> f64 {
        let mut g = 0.0;
        let mut discount = 1.0;
        for t in &self.transitions {
            g += discount * t.reward;
            discount *= gamma;
            if t.done {
                return g;
            }
        }
        g + discount * bootstrap_q
    }
}
