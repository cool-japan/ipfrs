//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::functions::{xorshift64, xorshift_f64};

/// Summary statistics for an MDP instance.
#[derive(Debug, Clone)]
pub struct MdpStats {
    pub num_states: usize,
    pub num_actions: usize,
    /// Total number of individual `Transition` objects stored.
    pub num_transitions: usize,
    pub terminal_states: usize,
    /// Average number of successor states per (state, action) pair that has
    /// at least one transition.
    pub avg_branching_factor: f64,
}
/// Metadata about a single state.
#[derive(Debug, Clone)]
pub struct MdpState {
    /// Unique identifier.
    pub id: MdpStateId,
    /// Human-readable name (may be empty).
    pub name: String,
    /// Whether this is an absorbing terminal state.
    pub is_terminal: bool,
}
/// Index of a state in the MDP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MdpStateId(pub usize);
/// Configuration for MDP solving algorithms.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Discount factor γ ∈ (0, 1].
    pub gamma: f64,
    /// Convergence threshold ε.
    pub epsilon: f64,
    /// Maximum number of iterations before giving up.
    pub max_iterations: usize,
    /// Which solver algorithm to employ.
    pub solver: SolverType,
}
/// Selects which MDP algorithm to use in [`MarkovDecisionProcess::solve`].
#[derive(Debug, Clone)]
pub enum SolverType {
    /// Synchronous value iteration (Bellman optimality update until convergence).
    ValueIteration,
    /// Classic policy iteration: full policy evaluation then greedy improvement.
    PolicyIteration,
    /// Modified policy iteration: policy evaluation runs only `k` update sweeps.
    ModifiedPolicyIteration(usize),
    /// Tabular Q-learning with ε-greedy exploration.
    Qlearning {
        /// Step-size / learning rate α ∈ (0, 1].
        alpha: f64,
        /// Exploration rate ε ∈ [0, 1].
        epsilon: f64,
    },
}
/// A deterministic policy: maps each state to the chosen action.
#[derive(Debug, Clone, Default)]
pub struct MdpPolicy {
    /// `state_id → action_id`.
    pub actions: HashMap<usize, usize>,
}
impl MdpPolicy {
    /// Return the action prescribed for `state`, if any.
    pub fn action_for(&self, state: MdpStateId) -> Option<MdpActionId> {
        self.actions.get(&state.0).copied().map(MdpActionId)
    }
    /// Assign an action for a state.
    pub fn set(&mut self, state: MdpStateId, action: MdpActionId) {
        self.actions.insert(state.0, action.0);
    }
}
/// A single probabilistic transition in the MDP.
#[derive(Debug, Clone)]
pub struct Transition {
    /// Destination state.
    pub to_state: MdpStateId,
    /// Transition probability.  Must be in [0, 1].
    pub probability: f64,
    /// Immediate reward received on this transition.
    pub reward: f64,
}
/// State-value function V(s).
#[derive(Debug, Clone)]
pub struct ValueFunction {
    /// Value for each state (indexed by `MdpStateId.0`).
    pub values: Vec<f64>,
}
impl ValueFunction {
    /// Create a zero-initialised value function for `n` states.
    pub fn zeros(n: usize) -> Self {
        Self {
            values: vec![0.0; n],
        }
    }
    /// Return V(s).
    pub fn get(&self, state: MdpStateId) -> f64 {
        self.values.get(state.0).copied().unwrap_or(0.0)
    }
    /// Set V(s) = v.
    pub fn set(&mut self, state: MdpStateId, v: f64) {
        if let Some(slot) = self.values.get_mut(state.0) {
            *slot = v;
        }
    }
    /// Compute max_s |self(s) - other(s)|.
    pub fn max_diff(&self, other: &ValueFunction) -> f64 {
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
    }
}
/// Output metadata from a solver run.
#[derive(Debug, Clone)]
pub struct SolverResult {
    /// `true` if the algorithm converged within `max_iterations`.
    pub converged: bool,
    /// Number of full sweeps performed.
    pub iterations: usize,
    /// The Bellman residual (max-norm) at termination.
    pub final_epsilon: f64,
}
/// Index of an action in the MDP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MdpActionId(pub usize);
/// Errors returned by MDP operations.
#[derive(Debug, Clone, PartialEq)]
pub enum MdpError {
    /// A state index was out of range.
    StateOutOfRange(usize),
    /// An action index was out of range.
    ActionOutOfRange(usize),
    /// A transition probability was not in [0, 1].
    InvalidProbability(f64),
    /// No transitions are registered for this (state, action) pair.
    NoTransitions { state: usize, action: usize },
}
/// Tabular Markov Decision Process.
///
/// Supports Value Iteration, Policy Iteration, and greedy policy extraction.
#[derive(Debug, Clone)]
pub struct MarkovDecisionProcess {
    /// All states, indexed by `MdpStateId.0`.
    pub states: Vec<MdpState>,
    /// Action names, indexed by `MdpActionId.0`.
    pub actions: Vec<String>,
    /// Transitions: `(from_state.0, action.0)` → list of `Transition`.
    pub transitions: HashMap<(usize, usize), Vec<Transition>>,
    /// Number of states (cached for ergonomics).
    pub num_states: usize,
    /// Number of actions (cached for ergonomics).
    pub num_actions: usize,
}
impl MarkovDecisionProcess {
    /// Create a new MDP with `num_states` unnamed non-terminal states and
    /// `num_actions` unnamed actions.
    pub fn new(num_states: usize, num_actions: usize) -> Self {
        let states = (0..num_states)
            .map(|i| MdpState {
                id: MdpStateId(i),
                name: String::new(),
                is_terminal: false,
            })
            .collect();
        let actions = vec![String::new(); num_actions];
        Self {
            states,
            actions,
            transitions: HashMap::new(),
            num_states,
            num_actions,
        }
    }
    /// Assign a human-readable name to a state.
    pub fn set_state_name(&mut self, state: MdpStateId, name: String) -> Result<(), MdpError> {
        self.states
            .get_mut(state.0)
            .ok_or(MdpError::StateOutOfRange(state.0))
            .map(|s| s.name = name)
    }
    /// Mark a state as terminal (absorbing).
    pub fn set_terminal(&mut self, state: MdpStateId, terminal: bool) -> Result<(), MdpError> {
        self.states
            .get_mut(state.0)
            .ok_or(MdpError::StateOutOfRange(state.0))
            .map(|s| s.is_terminal = terminal)
    }
    /// Assign a human-readable name to an action.
    pub fn set_action_name(&mut self, action: MdpActionId, name: String) -> Result<(), MdpError> {
        self.actions
            .get_mut(action.0)
            .ok_or(MdpError::ActionOutOfRange(action.0))
            .map(|a| *a = name)
    }
    /// Register a single transition `(from, action) → t`.
    ///
    /// Validates:
    /// - `from` is a valid state index
    /// - `action` is a valid action index
    /// - `t.to_state` is a valid state index
    /// - `t.probability ∈ [0, 1]`
    pub fn add_transition(
        &mut self,
        from: MdpStateId,
        action: MdpActionId,
        t: Transition,
    ) -> Result<(), MdpError> {
        if from.0 >= self.num_states {
            return Err(MdpError::StateOutOfRange(from.0));
        }
        if action.0 >= self.num_actions {
            return Err(MdpError::ActionOutOfRange(action.0));
        }
        if t.to_state.0 >= self.num_states {
            return Err(MdpError::StateOutOfRange(t.to_state.0));
        }
        if !(0.0..=1.0).contains(&t.probability) {
            return Err(MdpError::InvalidProbability(t.probability));
        }
        self.transitions
            .entry((from.0, action.0))
            .or_default()
            .push(t);
        Ok(())
    }
    /// Return all transitions from `from` under `action`.
    pub fn transitions_for(&self, from: MdpStateId, action: MdpActionId) -> &[Transition] {
        self.transitions
            .get(&(from.0, action.0))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
    /// Compute Q(s, a) = Σ_t p(t) · (r(t) + γ · V(t.to_state)) for every action.
    ///
    /// Returns a `Vec<f64>` of length `num_actions`.
    pub fn q_values(&self, vf: &ValueFunction, state: MdpStateId, gamma: f64) -> Vec<f64> {
        (0..self.num_actions)
            .map(|a| {
                self.transitions_for(state, MdpActionId(a))
                    .iter()
                    .map(|t| t.probability * (t.reward + gamma * vf.get(t.to_state)))
                    .sum()
            })
            .collect()
    }
    /// Run synchronous value iteration to convergence.
    ///
    /// Terminal states always have V(s) = 0.
    ///
    /// Returns `(ValueFunction, SolverResult)`.
    pub fn value_iteration(&self, config: &SolverConfig) -> (ValueFunction, SolverResult) {
        let mut vf = ValueFunction::zeros(self.num_states);
        let mut iterations = 0_usize;
        let mut final_epsilon = 0.0_f64;
        let mut converged = false;
        for _ in 0..config.max_iterations {
            let mut new_vf = ValueFunction::zeros(self.num_states);
            for s in 0..self.num_states {
                let state = MdpStateId(s);
                if self.states[s].is_terminal {
                    new_vf.set(state, 0.0);
                    continue;
                }
                let qs = self.q_values(&vf, state, config.gamma);
                let best = qs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let best = if best.is_finite() { best } else { 0.0 };
                new_vf.set(state, best);
            }
            let delta = vf.max_diff(&new_vf);
            iterations += 1;
            vf = new_vf;
            final_epsilon = delta;
            if delta < config.epsilon {
                converged = true;
                break;
            }
        }
        (
            vf,
            SolverResult {
                converged,
                iterations,
                final_epsilon,
            },
        )
    }
    /// Extract the greedy policy from a value function.
    pub fn extract_policy(&self, vf: &ValueFunction, config: &SolverConfig) -> MdpPolicy {
        let mut policy = MdpPolicy::default();
        for s in 0..self.num_states {
            let state = MdpStateId(s);
            if self.states[s].is_terminal {
                continue;
            }
            let qs = self.q_values(vf, state, config.gamma);
            let best_action = qs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            policy.set(state, MdpActionId(best_action));
        }
        policy
    }
    /// Evaluate a fixed policy until convergence or `max_iterations`.
    pub fn policy_evaluation(&self, policy: &MdpPolicy, config: &SolverConfig) -> ValueFunction {
        let mut vf = ValueFunction::zeros(self.num_states);
        for _ in 0..config.max_iterations {
            let mut new_vf = ValueFunction::zeros(self.num_states);
            for s in 0..self.num_states {
                let state = MdpStateId(s);
                if self.states[s].is_terminal {
                    new_vf.set(state, 0.0);
                    continue;
                }
                let action = match policy.action_for(state) {
                    Some(a) => a,
                    None => {
                        new_vf.set(state, 0.0);
                        continue;
                    }
                };
                let v: f64 = self
                    .transitions_for(state, action)
                    .iter()
                    .map(|t| t.probability * (t.reward + config.gamma * vf.get(t.to_state)))
                    .sum();
                new_vf.set(state, v);
            }
            let delta = vf.max_diff(&new_vf);
            vf = new_vf;
            if delta < config.epsilon {
                break;
            }
        }
        vf
    }
    /// Run policy iteration to convergence.
    ///
    /// Initialises with action 0 for every non-terminal state, then alternates
    /// between policy evaluation and greedy policy improvement.
    ///
    /// Returns `(MdpPolicy, ValueFunction, SolverResult)`.
    pub fn policy_iteration(
        &self,
        config: &SolverConfig,
    ) -> (MdpPolicy, ValueFunction, SolverResult) {
        let mut policy = MdpPolicy::default();
        for s in 0..self.num_states {
            if !self.states[s].is_terminal {
                policy.set(MdpStateId(s), MdpActionId(0));
            }
        }
        let mut iterations = 0_usize;
        let mut final_epsilon = 0.0_f64;
        let mut converged = false;
        let mut vf = ValueFunction::zeros(self.num_states);
        for _ in 0..config.max_iterations {
            let new_vf = self.policy_evaluation(&policy, config);
            let new_policy = self.extract_policy(&new_vf, config);
            let policy_changed = (0..self.num_states).any(|s| {
                let state = MdpStateId(s);
                if self.states[s].is_terminal {
                    return false;
                }
                policy.action_for(state) != new_policy.action_for(state)
            });
            final_epsilon = vf.max_diff(&new_vf);
            vf = new_vf;
            policy = new_policy;
            iterations += 1;
            if !policy_changed {
                converged = true;
                break;
            }
        }
        (
            policy,
            vf,
            SolverResult {
                converged,
                iterations,
                final_epsilon,
            },
        )
    }
    /// Compute V_π(start) — the expected return from `start` under `policy`.
    ///
    /// This simply reads the already-evaluated value function.  If no entry
    /// exists for `start` in the provided value function it returns 0.
    pub fn expected_return(
        &self,
        vf: &ValueFunction,
        _policy: &MdpPolicy,
        start: MdpStateId,
        _gamma: f64,
    ) -> f64 {
        vf.get(start)
    }
    /// Summarise the MDP structure.
    pub fn stats(&self) -> MdpStats {
        let num_transitions: usize = self.transitions.values().map(Vec::len).sum();
        let terminal_states = self.states.iter().filter(|s| s.is_terminal).count();
        let state_action_pairs = self.transitions.len();
        let avg_branching_factor = if state_action_pairs > 0 {
            num_transitions as f64 / state_action_pairs as f64
        } else {
            0.0
        };
        MdpStats {
            num_states: self.num_states,
            num_actions: self.num_actions,
            num_transitions,
            terminal_states,
            avg_branching_factor,
        }
    }
}
impl MarkovDecisionProcess {
    /// Append a new state and return its `MdpStateId`.
    pub fn add_state(&mut self, is_terminal: bool) -> MdpStateId {
        let id = MdpStateId(self.num_states);
        self.states.push(MdpState {
            id,
            name: String::new(),
            is_terminal,
        });
        self.num_states += 1;
        id
    }
    /// Append a new unnamed action and return its `MdpActionId`.
    pub fn add_action(&mut self) -> MdpActionId {
        let id = MdpActionId(self.num_actions);
        self.actions.push(String::new());
        self.num_actions += 1;
        id
    }
    /// Validate that every registered (state, action) pair has transition
    /// probabilities that sum to 1.0 ± 1e-6.
    ///
    /// Returns `Ok(())` when valid or the first violating pair as an error.
    pub fn validate(&self) -> Result<(), MdpError> {
        for (&(s, a), ts) in &self.transitions {
            let sum: f64 = ts.iter().map(|t| t.probability).sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(MdpError::InvalidProbability(sum));
            }
            let _ = (s, a);
        }
        Ok(())
    }
    /// Dispatch to the solver indicated by `config.solver` and return
    /// `(ValueFunction, MdpPolicy, SolverResult)`.
    pub fn solve(
        &self,
        config: &SolverConfig,
    ) -> Result<(ValueFunction, MdpPolicy, SolverResult), MdpError> {
        match &config.solver {
            SolverType::ValueIteration => {
                let (vf, result) = self.value_iteration(config);
                let policy = self.extract_policy(&vf, config);
                Ok((vf, policy, result))
            }
            SolverType::PolicyIteration => {
                let (policy, vf, result) = self.policy_iteration(config);
                Ok((vf, policy, result))
            }
            SolverType::ModifiedPolicyIteration(k) => {
                let (vf, result) = self.modified_policy_iteration(config, *k);
                let policy = self.extract_policy(&vf, config);
                Ok((vf, policy, result))
            }
            SolverType::Qlearning { alpha, epsilon } => {
                let (vf, policy, result) = self.q_learning(config, *alpha, *epsilon);
                Ok((vf, policy, result))
            }
        }
    }
    /// Return the greedy action for `state` given a value function.
    ///
    /// Errors if `state` is out of range.
    pub fn best_action(
        &self,
        state: MdpStateId,
        vf: &ValueFunction,
        gamma: f64,
    ) -> Result<MdpActionId, MdpError> {
        if state.0 >= self.num_states {
            return Err(MdpError::StateOutOfRange(state.0));
        }
        let qs = self.q_values(vf, state, gamma);
        let best = qs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| MdpActionId(i))
            .ok_or(MdpError::ActionOutOfRange(0))?;
        Ok(best)
    }
    /// Simulate a trajectory of at most `steps` steps starting from `start`
    /// following `policy`.
    ///
    /// Uses xorshift64 seeded with `rng_seed` for stochastic transitions.
    /// Terminates early if a terminal state is reached.
    ///
    /// Returns a `Vec` of `(state, action, reward)` triples (one per step
    /// taken).  If the start state is terminal or has no policy entry the
    /// vector is empty.
    pub fn simulate(
        &self,
        policy: &MdpPolicy,
        start: MdpStateId,
        steps: usize,
        rng_seed: u64,
    ) -> Vec<(MdpStateId, MdpActionId, f64)> {
        let mut rng = if rng_seed == 0 { 1 } else { rng_seed };
        let mut trajectory = Vec::with_capacity(steps);
        let mut current = start;
        for _ in 0..steps {
            if current.0 >= self.num_states || self.states[current.0].is_terminal {
                break;
            }
            let action = match policy.action_for(current) {
                Some(a) => a,
                None => break,
            };
            let ts = self.transitions_for(current, action);
            if ts.is_empty() {
                break;
            }
            let u = xorshift_f64(&mut rng);
            let mut cumulative = 0.0_f64;
            let mut next_state = ts[ts.len() - 1].to_state;
            let mut sampled_reward = ts[ts.len() - 1].reward;
            for t in ts {
                cumulative += t.probability;
                if u < cumulative {
                    next_state = t.to_state;
                    sampled_reward = t.reward;
                    break;
                }
            }
            trajectory.push((current, action, sampled_reward));
            current = next_state;
        }
        trajectory
    }
    /// Modified policy iteration: truncated policy evaluation (at most `k`
    /// sweeps) followed by greedy improvement until stable.
    pub fn modified_policy_iteration(
        &self,
        config: &SolverConfig,
        k: usize,
    ) -> (ValueFunction, SolverResult) {
        let eval_steps = if k == 0 { 1 } else { k };
        let mut policy = MdpPolicy::default();
        for s in 0..self.num_states {
            if !self.states[s].is_terminal {
                policy.set(MdpStateId(s), MdpActionId(0));
            }
        }
        let mut vf = ValueFunction::zeros(self.num_states);
        let mut iterations = 0_usize;
        let mut final_epsilon = 0.0_f64;
        let mut converged = false;
        for _ in 0..config.max_iterations {
            for _ in 0..eval_steps {
                let mut new_vf = ValueFunction::zeros(self.num_states);
                for s in 0..self.num_states {
                    let state = MdpStateId(s);
                    if self.states[s].is_terminal {
                        continue;
                    }
                    let action = match policy.action_for(state) {
                        Some(a) => a,
                        None => continue,
                    };
                    let v: f64 = self
                        .transitions_for(state, action)
                        .iter()
                        .map(|t| t.probability * (t.reward + config.gamma * vf.get(t.to_state)))
                        .sum();
                    new_vf.set(state, v);
                }
                final_epsilon = vf.max_diff(&new_vf);
                vf = new_vf;
            }
            let new_policy = self.extract_policy(&vf, config);
            let stable = !(0..self.num_states).any(|s| {
                let state = MdpStateId(s);
                !self.states[s].is_terminal
                    && policy.action_for(state) != new_policy.action_for(state)
            });
            policy = new_policy;
            iterations += 1;
            if stable {
                converged = true;
                break;
            }
        }
        (
            vf,
            SolverResult {
                converged,
                iterations,
                final_epsilon,
            },
        )
    }
    /// Tabular Q-learning with ε-greedy exploration.
    ///
    /// Runs `config.max_iterations` episodes.  Each episode starts from a
    /// random non-terminal state and follows the ε-greedy policy until a
    /// terminal state or a fixed intra-episode step cap is reached.
    ///
    /// Returns `(ValueFunction, MdpPolicy, SolverResult)`.
    pub fn q_learning(
        &self,
        config: &SolverConfig,
        alpha: f64,
        explore_eps: f64,
    ) -> (ValueFunction, MdpPolicy, SolverResult) {
        let mut q: Vec<Vec<f64>> = vec![vec![0.0; self.num_actions]; self.num_states];
        let mut rng: u64 = 12345;
        let intra_cap = 200_usize;
        let non_terminal_ids: Vec<usize> = (0..self.num_states)
            .filter(|&s| !self.states[s].is_terminal)
            .collect();
        if non_terminal_ids.is_empty() {
            let vf = ValueFunction::zeros(self.num_states);
            let policy = MdpPolicy::default();
            return (
                vf,
                policy,
                SolverResult {
                    converged: false,
                    iterations: 0,
                    final_epsilon: 0.0,
                },
            );
        }
        for ep in 0..config.max_iterations {
            let idx = (xorshift64(&mut rng) as usize) % non_terminal_ids.len();
            let mut s = non_terminal_ids[idx];
            for _ in 0..intra_cap {
                if self.states[s].is_terminal {
                    break;
                }
                let action = if xorshift_f64(&mut rng) < explore_eps {
                    (xorshift64(&mut rng) as usize) % self.num_actions
                } else {
                    let row = &q[s];
                    row.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                };
                let ts = self.transitions_for(MdpStateId(s), MdpActionId(action));
                if ts.is_empty() {
                    break;
                }
                let u = xorshift_f64(&mut rng);
                let mut cum = 0.0_f64;
                let last = &ts[ts.len() - 1];
                let mut next_s = last.to_state.0;
                let mut reward = last.reward;
                for t in ts {
                    cum += t.probability;
                    if u < cum {
                        next_s = t.to_state.0;
                        reward = t.reward;
                        break;
                    }
                }
                let max_next_q = if self.states[next_s].is_terminal {
                    0.0
                } else {
                    q[next_s]
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max)
                        .max(0.0)
                };
                let td_error = reward + config.gamma * max_next_q - q[s][action];
                q[s][action] += alpha * td_error;
                s = next_s;
            }
            let _ = ep;
        }
        let mut vf = ValueFunction::zeros(self.num_states);
        let mut policy = MdpPolicy::default();
        for (s, row) in q.iter().enumerate().take(self.num_states) {
            if self.states[s].is_terminal {
                continue;
            }
            let (best_a, best_q) = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, v)| (i, *v))
                .unwrap_or((0, 0.0));
            vf.set(MdpStateId(s), best_q);
            policy.set(MdpStateId(s), MdpActionId(best_a));
        }
        let result = SolverResult {
            converged: true,
            iterations: config.max_iterations,
            final_epsilon: 0.0,
        };
        (vf, policy, result)
    }
}
