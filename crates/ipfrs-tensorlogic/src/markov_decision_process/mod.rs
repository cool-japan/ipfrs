//! Markov Decision Process (MDP) solver.
//!
//! Implements tabular MDP solving via Value Iteration, Policy Iteration,
//! and Q-learning with full convergence tracking.
//!
//! # Example
//!
//! ```
//! use ipfrs_tensorlogic::markov_decision_process::{
//!     MarkovDecisionProcess, MdpStateId, MdpActionId, Transition, SolverConfig,
//! };
//!
//! let mut mdp = MarkovDecisionProcess::new(3, 2);
//! mdp.set_terminal(MdpStateId(2), true).expect("example: should succeed in docs");
//! mdp.add_transition(
//!     MdpStateId(0),
//!     MdpActionId(0),
//!     Transition { to_state: MdpStateId(2), probability: 1.0, reward: 1.0 },
//! ).expect("example: should succeed in docs");
//! let config = SolverConfig::default();
//! let (vf, result) = mdp.value_iteration(&config);
//! assert!(result.converged);
//! let _ = vf;
//! ```

pub mod functions;
pub mod mdperror_traits;
pub mod solverconfig_traits;
pub mod types;

// Re-export all types
pub use functions::*;
pub use types::*;

#[cfg(test)]
mod tests;
