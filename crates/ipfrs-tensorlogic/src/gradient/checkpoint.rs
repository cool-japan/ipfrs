//! Gradient checkpointing — snapshot and restore training state.
//!
//! [`GradientCheckpoint`] wraps a [`ComputationGraphStore`] with optimizer
//! metadata so that a training run can be interrupted and resumed without
//! recomputing the full forward pass from scratch.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::computation_graph::ComputationGraphStore;
use super::GradientError;

/// A serializable snapshot of the gradient computation state.
///
/// Contains the full [`ComputationGraphStore`] plus training metadata so that
/// a training run can be interrupted and resumed without recomputing the
/// forward pass from scratch.
#[derive(Debug, Serialize, Deserialize)]
pub struct GradientCheckpoint {
    /// The computation graph at the time of checkpointing
    pub graph: ComputationGraphStore,
    /// Optimizer step counter
    pub step: u64,
    /// CID of the loss tensor (None before the first loss is computed)
    pub loss_cid: Option<String>,
    /// Raw optimizer state blobs, keyed by parameter name
    pub optimizer_state: HashMap<String, Vec<u8>>,
    /// Unix timestamp (seconds) when this checkpoint was created
    pub timestamp: u64,
}

impl GradientCheckpoint {
    /// Create a new checkpoint at the given step.
    pub fn new(graph: ComputationGraphStore, step: u64) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            graph,
            step,
            loss_cid: None,
            optimizer_state: HashMap::new(),
            timestamp,
        }
    }

    /// Attach the CID of the current loss tensor.
    pub fn with_loss_cid(mut self, cid: impl Into<String>) -> Self {
        self.loss_cid = Some(cid.into());
        self
    }

    /// Store a raw optimizer state blob for the given parameter.
    pub fn set_optimizer_state(&mut self, param: impl Into<String>, state: Vec<u8>) {
        self.optimizer_state.insert(param.into(), state);
    }

    /// Persist this checkpoint to a file at `path`.
    ///
    /// The file is written as JSON so it is human-inspectable and
    /// version-control friendly.
    pub fn save(&self, path: &std::path::Path) -> Result<(), GradientError> {
        let bytes = serde_json::to_vec_pretty(self).map_err(|e| {
            GradientError::InvalidGradient(format!("Checkpoint save serialization: {e}"))
        })?;

        // Use a temp-file-then-rename strategy for atomicity.
        let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
        let tmp_path = parent.join(format!(".ckpt_tmp_{}.json", uuid::Uuid::new_v4()));

        std::fs::write(&tmp_path, &bytes)
            .map_err(|e| GradientError::InvalidGradient(format!("Checkpoint write: {e}")))?;

        std::fs::rename(&tmp_path, path)
            .map_err(|e| GradientError::InvalidGradient(format!("Checkpoint rename: {e}")))?;

        Ok(())
    }

    /// Load a checkpoint from the file at `path`.
    pub fn load(path: &std::path::Path) -> Result<Self, GradientError> {
        let bytes = std::fs::read(path)
            .map_err(|e| GradientError::InvalidGradient(format!("Checkpoint read: {e}")))?;

        serde_json::from_slice(&bytes)
            .map_err(|e| GradientError::InvalidGradient(format!("Checkpoint load parse: {e}")))
    }
}
