//! Command handler modules for IPFRS CLI
//!
//! This module organizes all command implementations into logical groups:
//!
//! - [`daemon`] - Daemon management (start, stop, status, etc.)
//! - [`file`](mod@file) - File operations (add, get, cat, ls)
//! - [`block`] - Raw block operations
//! - [`dag`] - DAG node operations
//! - [`pin`] - Pin management
//! - [`repo`] - Repository management (gc, stat, fsck)
//! - [`network`] - Network operations (swarm, dht, bootstrap)
//! - [`tensor`] - Tensor operations
//! - [`logic`] - Logic programming operations
//! - [`semantic`] - Semantic search operations
//! - [`query`] - Hybrid query (semantic + logic)
//! - [`model`] - Model management
//! - [`gradient`] - Gradient operations for federated learning
//! - [`stats`] - Statistics display
//! - [`gateway`] - HTTP gateway
//! - [`config`] - Configuration management

pub mod block;
pub mod config;
pub mod daemon;
pub mod dag;
pub mod diag;
pub mod file;
pub mod gateway;
pub mod gradient;
pub mod ipld;
pub mod logic;
pub mod metrics;
pub mod model;
pub mod network;
pub mod pin;
pub mod query;
pub mod repo;
pub mod semantic;
pub mod stats;
pub mod tensor;

// Common utilities shared across command modules
pub mod common;

pub use block::*;
pub use config::*;
pub use daemon::*;
pub use dag::*;
pub use diag::*;
pub use file::*;
pub use gateway::*;
pub use gradient::*;
pub use logic::*;
pub use metrics::*;
pub use model::*;
pub use network::*;
pub use pin::*;
pub use query::*;
pub use repo::*;
pub use semantic::*;
pub use stats::*;
pub use tensor::*;

// ipld module is NOT glob-re-exported because its symbols would conflict with
// other modules.  Callers use the full path: crate::commands::ipld::*
