//! Gradient coordination protocol for distributed backward passes.
//!
//! This module provides content-addressed Arrow IPC gradient storage and a
//! `BackwardPassCoordinator` that tracks multi-peer gradient contributions for
//! federated backward pass rounds.

use std::collections::HashMap;

// ── BackwardPassId ─────────────────────────────────────────────────────────

/// Identifies a backward pass round across distributed nodes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BackwardPassId {
    pub round: u64,
    pub initiator_peer: String,
}

impl BackwardPassId {
    pub fn new(round: u64, initiator_peer: impl Into<String>) -> Self {
        Self {
            round,
            initiator_peer: initiator_peer.into(),
        }
    }

    pub fn as_key(&self) -> String {
        format!("{}-{}", self.round, self.initiator_peer)
    }
}

// ── CoordinationStatus ─────────────────────────────────────────────────────

/// Status of a coordinated backward pass.
#[derive(Debug, Clone, PartialEq)]
pub enum CoordinationStatus {
    Pending,
    CollectingGradients { received: usize, expected: usize },
    Aggregating,
    Complete { aggregated_cid: String },
    Failed { reason: String },
}

impl CoordinationStatus {
    fn is_terminal(&self) -> bool {
        matches!(self, Self::Complete { .. } | Self::Failed { .. })
    }

    fn display_name(&self) -> &'static str {
        match self {
            Self::Pending => "Pending",
            Self::CollectingGradients { .. } => "CollectingGradients",
            Self::Aggregating => "Aggregating",
            Self::Complete { .. } => "Complete",
            Self::Failed { .. } => "Failed",
        }
    }
}

// ── GradientContribution ───────────────────────────────────────────────────

/// A gradient contribution from one participant peer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GradientContribution {
    pub pass_id: BackwardPassId,
    pub peer_id: String,
    pub gradient_cid: String,
    pub layer_name: String,
    pub num_samples: usize,
    pub submitted_at_ms: u64,
}

// ── CoordinationError ──────────────────────────────────────────────────────

#[derive(Debug)]
pub enum CoordinationError {
    PassNotFound { pass_id: String },
    PassAlreadyTerminal { status: String },
}

impl std::fmt::Display for CoordinationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PassNotFound { pass_id } => {
                write!(f, "Backward pass not found: {pass_id}")
            }
            Self::PassAlreadyTerminal { status } => {
                write!(f, "Backward pass is already terminal (status: {status})")
            }
        }
    }
}

impl std::error::Error for CoordinationError {}

// ── BackwardPassCoordinator ────────────────────────────────────────────────

/// Coordinates gradient collection and aggregation across peers.
pub struct BackwardPassCoordinator {
    /// pass_id_key → (status, contributions)
    passes: parking_lot::RwLock<HashMap<String, (CoordinationStatus, Vec<GradientContribution>)>>,
    expected_peers: usize,
    timeout_ms: u64,
}

impl BackwardPassCoordinator {
    pub fn new(expected_peers: usize, timeout_ms: u64) -> Self {
        Self {
            passes: parking_lot::RwLock::new(HashMap::new()),
            expected_peers,
            timeout_ms,
        }
    }

    /// Start a new backward pass round. Returns `false` if `pass_id` already exists.
    pub fn start_pass(&self, pass_id: BackwardPassId) -> bool {
        let key = pass_id.as_key();
        let mut passes = self.passes.write();
        if passes.contains_key(&key) {
            return false;
        }
        passes.insert(key, (CoordinationStatus::Pending, Vec::new()));
        true
    }

    /// Submit a gradient contribution for a pass.
    ///
    /// Returns an error if the pass is not found or is already Complete/Failed.
    pub fn submit_contribution(
        &self,
        contribution: GradientContribution,
    ) -> Result<CoordinationStatus, CoordinationError> {
        let key = contribution.pass_id.as_key();
        let mut passes = self.passes.write();
        let entry = passes
            .get_mut(&key)
            .ok_or_else(|| CoordinationError::PassNotFound {
                pass_id: key.clone(),
            })?;

        let (status, contributions) = entry;
        if status.is_terminal() {
            return Err(CoordinationError::PassAlreadyTerminal {
                status: status.display_name().to_string(),
            });
        }

        contributions.push(contribution);
        let received = contributions.len();
        let expected = self.expected_peers;
        *status = CoordinationStatus::CollectingGradients { received, expected };
        Ok(status.clone())
    }

    /// Returns `true` when all expected peers have submitted contributions.
    pub fn is_ready_to_aggregate(&self, pass_id: &BackwardPassId) -> bool {
        let key = pass_id.as_key();
        let passes = self.passes.read();
        passes
            .get(&key)
            .map(|(_, contributions)| contributions.len() >= self.expected_peers)
            .unwrap_or(false)
    }

    /// Mark aggregation as complete with result CID.
    pub fn mark_complete(
        &self,
        pass_id: &BackwardPassId,
        aggregated_cid: impl Into<String>,
    ) -> bool {
        let key = pass_id.as_key();
        let mut passes = self.passes.write();
        if let Some((status, _)) = passes.get_mut(&key) {
            *status = CoordinationStatus::Complete {
                aggregated_cid: aggregated_cid.into(),
            };
            true
        } else {
            false
        }
    }

    /// Mark pass as failed with a reason string.
    pub fn mark_failed(&self, pass_id: &BackwardPassId, reason: impl Into<String>) -> bool {
        let key = pass_id.as_key();
        let mut passes = self.passes.write();
        if let Some((status, _)) = passes.get_mut(&key) {
            *status = CoordinationStatus::Failed {
                reason: reason.into(),
            };
            true
        } else {
            false
        }
    }

    /// Get current status of a pass.
    pub fn status(&self, pass_id: &BackwardPassId) -> Option<CoordinationStatus> {
        let key = pass_id.as_key();
        let passes = self.passes.read();
        passes.get(&key).map(|(status, _)| status.clone())
    }

    /// Get all contributions for a pass.
    pub fn contributions(&self, pass_id: &BackwardPassId) -> Vec<GradientContribution> {
        let key = pass_id.as_key();
        let passes = self.passes.read();
        passes
            .get(&key)
            .map(|(_, contributions)| contributions.clone())
            .unwrap_or_default()
    }

    /// Total active (non-complete, non-failed) passes.
    pub fn active_count(&self) -> usize {
        let passes = self.passes.read();
        passes
            .values()
            .filter(|(status, _)| !status.is_terminal())
            .count()
    }

    /// Remove completed/failed passes with `pass_id.round < round`.
    /// Returns the number of passes removed.
    pub fn gc_before_round(&self, round: u64) -> usize {
        let mut passes = self.passes.write();
        let before = passes.len();
        passes.retain(|key, (status, _)| {
            if !status.is_terminal() {
                return true;
            }
            // Extract round from key: "{round}-{initiator_peer}"
            let round_num: Option<u64> = key.split_once('-').and_then(|(r, _)| r.parse().ok());
            match round_num {
                Some(r) => r >= round,
                None => true,
            }
        });
        before - passes.len()
    }

    /// Expose timeout setting (useful for callers).
    #[inline]
    pub fn timeout_ms(&self) -> u64 {
        self.timeout_ms
    }
}

// ── ArrowBlockError ────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum ArrowBlockError {
    EmptyInput,
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    InvalidMagic,
    TruncatedData,
}

impl std::fmt::Display for ArrowBlockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "No gradient blocks provided"),
            Self::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {expected:?}, got {got:?}")
            }
            Self::InvalidMagic => write!(f, "Invalid magic bytes in Arrow block"),
            Self::TruncatedData => write!(f, "Truncated data in Arrow block"),
        }
    }
}

impl std::error::Error for ArrowBlockError {}

// ── GradientArrowBlock ─────────────────────────────────────────────────────

const MAGIC: &[u8; 4] = b"GARW";

/// A gradient stored as an Arrow IPC block with content-addressing.
pub struct GradientArrowBlock {
    pub cid: String,
    pub layer_name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
    pub num_samples: usize,
}

impl GradientArrowBlock {
    /// Serialize gradient values to custom Arrow-flavoured IPC bytes.
    ///
    /// Layout:
    /// ```text
    /// 4  bytes  magic "GARW"
    /// 4  bytes  shape_len  (u32 LE)
    /// 4n bytes  shape      (u32 LE each)
    /// 4  bytes  values_len (u32 LE)
    /// 4m bytes  values     (f32 LE each)
    /// 4  bytes  num_samples (u32 LE)
    /// ```
    pub fn to_arrow_bytes(&self) -> Vec<u8> {
        let shape_len = self.shape.len() as u32;
        let values_len = self.values.len() as u32;
        let cap = 4 + 4 + 4 * self.shape.len() + 4 + 4 * self.values.len() + 4;
        let mut buf = Vec::with_capacity(cap);

        buf.extend_from_slice(MAGIC);

        buf.extend_from_slice(&shape_len.to_le_bytes());
        for &dim in &self.shape {
            buf.extend_from_slice(&(dim as u32).to_le_bytes());
        }

        buf.extend_from_slice(&values_len.to_le_bytes());
        for &v in &self.values {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        buf.extend_from_slice(&(self.num_samples as u32).to_le_bytes());
        buf
    }

    /// Deserialize from bytes produced by [`Self::to_arrow_bytes`].
    pub fn from_arrow_bytes(
        cid: String,
        layer_name: String,
        data: &[u8],
    ) -> Result<Self, ArrowBlockError> {
        let mut pos = 0;

        // Magic
        if data.len() < 4 {
            return Err(ArrowBlockError::TruncatedData);
        }
        if &data[pos..pos + 4] != MAGIC {
            return Err(ArrowBlockError::InvalidMagic);
        }
        pos += 4;

        // shape_len
        if data.len() < pos + 4 {
            return Err(ArrowBlockError::TruncatedData);
        }
        let shape_len = u32::from_le_bytes(
            data[pos..pos + 4]
                .try_into()
                .map_err(|_| ArrowBlockError::TruncatedData)?,
        ) as usize;
        pos += 4;

        // shape
        if data.len() < pos + 4 * shape_len {
            return Err(ArrowBlockError::TruncatedData);
        }
        let mut shape = Vec::with_capacity(shape_len);
        for _ in 0..shape_len {
            let dim = u32::from_le_bytes(
                data[pos..pos + 4]
                    .try_into()
                    .map_err(|_| ArrowBlockError::TruncatedData)?,
            ) as usize;
            shape.push(dim);
            pos += 4;
        }

        // values_len
        if data.len() < pos + 4 {
            return Err(ArrowBlockError::TruncatedData);
        }
        let values_len = u32::from_le_bytes(
            data[pos..pos + 4]
                .try_into()
                .map_err(|_| ArrowBlockError::TruncatedData)?,
        ) as usize;
        pos += 4;

        // values
        if data.len() < pos + 4 * values_len {
            return Err(ArrowBlockError::TruncatedData);
        }
        let mut values = Vec::with_capacity(values_len);
        for _ in 0..values_len {
            let v = f32::from_le_bytes(
                data[pos..pos + 4]
                    .try_into()
                    .map_err(|_| ArrowBlockError::TruncatedData)?,
            );
            values.push(v);
            pos += 4;
        }

        // num_samples
        if data.len() < pos + 4 {
            return Err(ArrowBlockError::TruncatedData);
        }
        let num_samples = u32::from_le_bytes(
            data[pos..pos + 4]
                .try_into()
                .map_err(|_| ArrowBlockError::TruncatedData)?,
        ) as usize;

        Ok(Self {
            cid,
            layer_name,
            shape,
            values,
            num_samples,
        })
    }

    /// Compute a deterministic CID string: `"grad-" + hex(FNV-1a hash of serialized bytes)`.
    pub fn compute_cid(layer_name: &str, values: &[f32], num_samples: usize) -> String {
        // Build a temporary block for hashing purposes
        let tmp = Self {
            cid: String::new(),
            layer_name: layer_name.to_string(),
            // Derive shape from values length: single-dimensional
            shape: vec![values.len()],
            values: values.to_vec(),
            num_samples,
        };
        let bytes = tmp.to_arrow_bytes();
        let hash = fnv1a_hash(&bytes);
        format!("grad-{hash:016x}")
    }

    /// FedAvg: weighted average of multiple gradient blocks.
    ///
    /// Weights are proportional to `num_samples`. All blocks must have the same
    /// shape. Returns `ArrowBlockError::EmptyInput` for empty input and
    /// `ArrowBlockError::ShapeMismatch` if shapes differ.
    pub fn fedavg(blocks: &[GradientArrowBlock]) -> Result<Vec<f32>, ArrowBlockError> {
        if blocks.is_empty() {
            return Err(ArrowBlockError::EmptyInput);
        }

        let expected_shape = &blocks[0].shape;
        for block in blocks.iter().skip(1) {
            if &block.shape != expected_shape {
                return Err(ArrowBlockError::ShapeMismatch {
                    expected: expected_shape.clone(),
                    got: block.shape.clone(),
                });
            }
        }

        let total_samples: usize = blocks.iter().map(|b| b.num_samples).sum();
        let dim = blocks[0].values.len();
        let mut avg = vec![0.0f32; dim];

        if total_samples == 0 {
            // Uniform average fallback when all num_samples == 0
            let n = blocks.len() as f32;
            for block in blocks {
                for (a, &v) in avg.iter_mut().zip(block.values.iter()) {
                    *a += v / n;
                }
            }
        } else {
            let total = total_samples as f32;
            for block in blocks {
                let weight = block.num_samples as f32 / total;
                for (a, &v) in avg.iter_mut().zip(block.values.iter()) {
                    *a += v * weight;
                }
            }
        }

        Ok(avg)
    }
}

// ── FNV-1a hash (pure Rust, no external deps) ─────────────────────────────

fn fnv1a_hash(data: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pass_id(round: u64) -> BackwardPassId {
        BackwardPassId::new(round, format!("peer-{round}"))
    }

    fn make_contribution(pass_id: BackwardPassId, peer_id: &str) -> GradientContribution {
        GradientContribution {
            pass_id,
            peer_id: peer_id.to_string(),
            gradient_cid: format!("grad-cid-{peer_id}"),
            layer_name: "layer0".to_string(),
            num_samples: 100,
            submitted_at_ms: 0,
        }
    }

    // ── BackwardPassCoordinator tests ──────────────────────────────────────

    #[test]
    fn test_start_pass() {
        let coord = BackwardPassCoordinator::new(2, 5000);
        let pass_id = make_pass_id(1);
        assert!(coord.start_pass(pass_id.clone()));
        assert_eq!(coord.status(&pass_id), Some(CoordinationStatus::Pending));
    }

    #[test]
    fn test_start_pass_duplicate_returns_false() {
        let coord = BackwardPassCoordinator::new(2, 5000);
        let pass_id = make_pass_id(1);
        assert!(coord.start_pass(pass_id.clone()));
        assert!(!coord.start_pass(pass_id));
    }

    #[test]
    fn test_submit_contribution_updates_status() {
        let coord = BackwardPassCoordinator::new(2, 5000);
        let pass_id = make_pass_id(2);
        coord.start_pass(pass_id.clone());

        let contribution = make_contribution(pass_id.clone(), "peer-a");
        let status = coord
            .submit_contribution(contribution)
            .expect("submit should succeed");

        assert_eq!(
            status,
            CoordinationStatus::CollectingGradients {
                received: 1,
                expected: 2
            }
        );
    }

    #[test]
    fn test_is_ready_to_aggregate_after_all_peers() {
        let coord = BackwardPassCoordinator::new(2, 5000);
        let pass_id = make_pass_id(3);
        coord.start_pass(pass_id.clone());

        assert!(!coord.is_ready_to_aggregate(&pass_id));

        coord
            .submit_contribution(make_contribution(pass_id.clone(), "peer-a"))
            .expect("peer-a submit");
        assert!(!coord.is_ready_to_aggregate(&pass_id));

        coord
            .submit_contribution(make_contribution(pass_id.clone(), "peer-b"))
            .expect("peer-b submit");
        assert!(coord.is_ready_to_aggregate(&pass_id));
    }

    #[test]
    fn test_mark_complete() {
        let coord = BackwardPassCoordinator::new(1, 5000);
        let pass_id = make_pass_id(4);
        coord.start_pass(pass_id.clone());

        assert!(coord.mark_complete(&pass_id, "agg-cid-42"));
        assert_eq!(
            coord.status(&pass_id),
            Some(CoordinationStatus::Complete {
                aggregated_cid: "agg-cid-42".to_string()
            })
        );
    }

    #[test]
    fn test_mark_failed() {
        let coord = BackwardPassCoordinator::new(1, 5000);
        let pass_id = make_pass_id(5);
        coord.start_pass(pass_id.clone());

        assert!(coord.mark_failed(&pass_id, "timeout"));
        assert_eq!(
            coord.status(&pass_id),
            Some(CoordinationStatus::Failed {
                reason: "timeout".to_string()
            })
        );
    }

    #[test]
    fn test_submit_to_terminal_pass_returns_error() {
        let coord = BackwardPassCoordinator::new(1, 5000);
        let pass_id = make_pass_id(6);
        coord.start_pass(pass_id.clone());
        coord.mark_failed(&pass_id, "network error");

        let result = coord.submit_contribution(make_contribution(pass_id, "late-peer"));
        assert!(matches!(
            result,
            Err(CoordinationError::PassAlreadyTerminal { .. })
        ));
    }

    #[test]
    fn test_gc_before_round() {
        let coord = BackwardPassCoordinator::new(1, 5000);

        // Round 1 — complete
        let p1 = make_pass_id(1);
        coord.start_pass(p1.clone());
        coord.mark_complete(&p1, "cid-1");

        // Round 5 — complete
        let p5 = make_pass_id(5);
        coord.start_pass(p5.clone());
        coord.mark_complete(&p5, "cid-5");

        // Round 10 — active
        let p10 = make_pass_id(10);
        coord.start_pass(p10.clone());

        // GC passes with round < 5 (i.e. round 1 should be removed)
        let removed = coord.gc_before_round(5);
        assert_eq!(removed, 1, "only round-1 should be removed");

        assert!(coord.status(&p1).is_none(), "round-1 gone");
        assert!(coord.status(&p5).is_some(), "round-5 remains");
        assert!(coord.status(&p10).is_some(), "round-10 remains (active)");
    }

    #[test]
    fn test_active_count() {
        let coord = BackwardPassCoordinator::new(1, 5000);

        let p1 = make_pass_id(1);
        let p2 = make_pass_id(2);
        let p3 = make_pass_id(3);

        coord.start_pass(p1.clone());
        coord.start_pass(p2.clone());
        coord.start_pass(p3.clone());

        assert_eq!(coord.active_count(), 3);

        coord.mark_complete(&p1, "cid");
        assert_eq!(coord.active_count(), 2);

        coord.mark_failed(&p2, "err");
        assert_eq!(coord.active_count(), 1);
    }

    // ── GradientArrowBlock tests ───────────────────────────────────────────

    #[test]
    fn test_gradient_arrow_block_roundtrip() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let cid = GradientArrowBlock::compute_cid("layer1", &values, 50);

        let block = GradientArrowBlock {
            cid: cid.clone(),
            layer_name: "layer1".to_string(),
            shape: vec![4],
            values: values.clone(),
            num_samples: 50,
        };

        let bytes = block.to_arrow_bytes();
        let decoded =
            GradientArrowBlock::from_arrow_bytes(cid.clone(), "layer1".to_string(), &bytes)
                .expect("roundtrip");

        assert_eq!(decoded.shape, vec![4]);
        assert_eq!(decoded.values, values);
        assert_eq!(decoded.num_samples, 50);
        assert_eq!(decoded.cid, cid);
    }

    #[test]
    fn test_fedavg_weighted() {
        // Block A: values [0.0, 0.0], 100 samples
        // Block B: values [2.0, 4.0], 300 samples
        // Expected weighted avg: 0*0.25 + 2*0.75 = 1.5 and 0*0.25 + 4*0.75 = 3.0
        let block_a = GradientArrowBlock {
            cid: "a".to_string(),
            layer_name: "l".to_string(),
            shape: vec![2],
            values: vec![0.0, 0.0],
            num_samples: 100,
        };
        let block_b = GradientArrowBlock {
            cid: "b".to_string(),
            layer_name: "l".to_string(),
            shape: vec![2],
            values: vec![2.0, 4.0],
            num_samples: 300,
        };
        let avg = GradientArrowBlock::fedavg(&[block_a, block_b]).expect("fedavg");
        assert!((avg[0] - 1.5).abs() < 1e-5, "avg[0] = {}", avg[0]);
        assert!((avg[1] - 3.0).abs() < 1e-5, "avg[1] = {}", avg[1]);
    }

    #[test]
    fn test_fedavg_empty_returns_error() {
        let result = GradientArrowBlock::fedavg(&[]);
        assert!(matches!(result, Err(ArrowBlockError::EmptyInput)));
    }

    #[test]
    fn test_fedavg_shape_mismatch() {
        let block_a = GradientArrowBlock {
            cid: "a".to_string(),
            layer_name: "l".to_string(),
            shape: vec![3],
            values: vec![1.0, 2.0, 3.0],
            num_samples: 10,
        };
        let block_b = GradientArrowBlock {
            cid: "b".to_string(),
            layer_name: "l".to_string(),
            shape: vec![2],
            values: vec![1.0, 2.0],
            num_samples: 10,
        };
        let result = GradientArrowBlock::fedavg(&[block_a, block_b]);
        assert!(matches!(result, Err(ArrowBlockError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_compute_cid_deterministic() {
        let values = vec![0.5f32, 1.0, 1.5];
        let cid1 = GradientArrowBlock::compute_cid("layer2", &values, 64);
        let cid2 = GradientArrowBlock::compute_cid("layer2", &values, 64);
        assert_eq!(cid1, cid2, "CIDs must be deterministic");
        assert!(cid1.starts_with("grad-"), "CID prefix");
    }

    #[test]
    fn test_from_arrow_bytes_bad_magic() {
        let mut bad = vec![0u8; 20];
        bad[0] = b'X'; // corrupt magic
        let result = GradientArrowBlock::from_arrow_bytes("x".to_string(), "l".to_string(), &bad);
        assert!(matches!(result, Err(ArrowBlockError::InvalidMagic)));
    }
}
