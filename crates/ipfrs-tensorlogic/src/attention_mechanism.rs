//! Scaled dot-product attention and multi-head attention for transformer-style models.
//!
//! This module provides the fundamental attention building blocks used in modern
//! transformer architectures (Vaswani et al. 2017 "Attention Is All You Need"):
//!
//! * **Scaled dot-product attention** — computes Q·Kᵀ / scale, applies an
//!   optional boolean mask (e.g. causal / padding), runs row-wise softmax, then
//!   blends the values: output = softmax(QKᵀ/scale) · V.
//! * **Multi-head attention wrapper** — splits the embedding dimension across
//!   `num_heads` independent heads, runs scaled dot-product attention on each,
//!   then concatenates the results.
//! * **Causal mask generation** — produces the upper-triangular boolean mask
//!   required for autoregressive (decoder-style) generation.
//! * **Primitive matrix operations** — `matmul`, `transpose`, and numerically
//!   stable row-wise `softmax_1d`, all implemented purely in terms of
//!   `Vec<Vec<f64>>` with no external linear-algebra dependency.
//! * **AttentionMatrix** — row-major 2-D matrix with full operator support.
//! * **AttentionMechanism** — production-grade multi-head attention with learned
//!   projection matrices, sinusoidal positional encoding, causal masking, entropy
//!   and peak attention analysis, and a `forward()` pass.
//!
//! # Example
//!
//! ```rust
//! use ipfrs_tensorlogic::{AttentionConfig, AttentionMechanism};
//!
//! let cfg = AttentionConfig {
//!     num_heads: 2,
//!     head_dim: 4,
//!     dropout_rate: 0.0,
//!     use_causal_mask: false,
//! };
//! let mut attn = AttentionMechanism::new(cfg, 64);
//!
//! // 3 tokens, d_model = 8 (num_heads * head_dim)
//! use ipfrs_tensorlogic::AttentionMatrix;
//! let input = AttentionMatrix::zeros(3, 8);
//! let out = attn.forward(&input).expect("example: should succeed in docs");
//! assert_eq!(out.output.rows, 3);
//! assert_eq!(out.output.cols, 8);
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during attention computation.
#[derive(Debug, Clone)]
pub enum AttnError {
    /// A matrix operation received incompatible dimensions.
    DimensionMismatch {
        op: String,
        expected: String,
        got: String,
    },
    /// The input sequence has zero tokens.
    EmptyInput,
    /// The `AttentionConfig` is invalid.
    InvalidConfig(String),
}

impl std::fmt::Display for AttnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { op, expected, got } => {
                write!(
                    f,
                    "DimensionMismatch in {op}: expected {expected}, got {got}"
                )
            }
            Self::EmptyInput => write!(f, "EmptyInput: sequence length is 0"),
            Self::InvalidConfig(msg) => write!(f, "InvalidConfig: {msg}"),
        }
    }
}

impl std::error::Error for AttnError {}

// ─────────────────────────────────────────────────────────────────────────────
// AttentionMatrix — row-major 2-D matrix
// ─────────────────────────────────────────────────────────────────────────────

/// A row-major 2-D matrix used throughout the attention computation.
#[derive(Debug, Clone)]
pub struct AttentionMatrix {
    /// Flat row-major storage; length == `rows * cols`.
    pub values: Vec<f64>,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
}

impl AttentionMatrix {
    /// Construct a zero-filled matrix of the given dimensions.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            values: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Get the value at `(row, col)`.  Returns `0.0` for out-of-bounds access.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row < self.rows && col < self.cols {
            self.values[row * self.cols + col]
        } else {
            0.0
        }
    }

    /// Set the value at `(row, col)`.  No-ops for out-of-bounds.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, v: f64) {
        if row < self.rows && col < self.cols {
            self.values[row * self.cols + col] = v;
        }
    }

    /// Matrix multiply `a (m×k)` by `b (k×n)` → `m×n`.
    ///
    /// Returns an error if inner dimensions do not match.
    pub fn matmul(a: &AttentionMatrix, b: &AttentionMatrix) -> Result<AttentionMatrix, AttnError> {
        if a.cols != b.rows {
            return Err(AttnError::DimensionMismatch {
                op: "AttentionMatrix::matmul".to_string(),
                expected: format!("b.rows == {}", a.cols),
                got: format!("b.rows == {}", b.rows),
            });
        }
        let m = a.rows;
        let k = a.cols;
        let n = b.cols;
        let mut out = AttentionMatrix::zeros(m, n);
        for i in 0..m {
            for p in 0..k {
                let a_val = a.values[i * k + p];
                if a_val == 0.0 {
                    continue;
                }
                for j in 0..n {
                    out.values[i * n + j] += a_val * b.values[p * n + j];
                }
            }
        }
        Ok(out)
    }

    /// Return the transpose of this matrix: `(rows×cols)` → `(cols×rows)`.
    pub fn transpose(&self) -> AttentionMatrix {
        let mut out = AttentionMatrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.values[c * self.rows + r] = self.values[r * self.cols + c];
            }
        }
        out
    }

    /// Apply row-wise softmax with numerically stable max-subtraction.
    pub fn softmax_rows(&self) -> AttentionMatrix {
        let mut out = AttentionMatrix::zeros(self.rows, self.cols);
        for r in 0..self.rows {
            let start = r * self.cols;
            let end = start + self.cols;
            let row = &self.values[start..end];
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = row.iter().map(|x| (x - max_val).exp()).collect();
            let sum: f64 = exps.iter().sum();
            let denom = if sum == 0.0 { 1.0 } else { sum };
            for (c, exp_val) in exps.iter().enumerate() {
                out.values[start + c] = exp_val / denom;
            }
        }
        out
    }

    /// Add positional encoding rows (element-wise) to the first `seq_len` rows.
    ///
    /// Both matrices must have the same number of columns.
    fn add_pos_enc(&self, pos_enc: &AttentionMatrix) -> Result<AttentionMatrix, AttnError> {
        if self.cols != pos_enc.cols {
            return Err(AttnError::DimensionMismatch {
                op: "add_pos_enc".to_string(),
                expected: format!("pos_enc.cols == {}", self.cols),
                got: format!("pos_enc.cols == {}", pos_enc.cols),
            });
        }
        let seq_len = self.rows.min(pos_enc.rows);
        let mut out = self.clone();
        for r in 0..seq_len {
            for c in 0..self.cols {
                out.values[r * self.cols + c] += pos_enc.values[r * pos_enc.cols + c];
            }
        }
        Ok(out)
    }

    /// Horizontally concatenate a slice of matrices (all with the same number of rows).
    ///
    /// Returns an error if any matrix has a different row count.
    fn hconcat(mats: &[AttentionMatrix]) -> Result<AttentionMatrix, AttnError> {
        if mats.is_empty() {
            return Ok(AttentionMatrix::zeros(0, 0));
        }
        let rows = mats[0].rows;
        let total_cols: usize = mats.iter().map(|m| m.cols).sum();
        for m in mats.iter().skip(1) {
            if m.rows != rows {
                return Err(AttnError::DimensionMismatch {
                    op: "hconcat".to_string(),
                    expected: format!("rows == {rows}"),
                    got: format!("rows == {}", m.rows),
                });
            }
        }
        let mut out = AttentionMatrix::zeros(rows, total_cols);
        let mut col_offset = 0usize;
        for m in mats {
            for r in 0..rows {
                for c in 0..m.cols {
                    out.values[r * total_cols + col_offset + c] = m.values[r * m.cols + c];
                }
            }
            col_offset += m.cols;
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the production-grade [`AttentionMechanism`].
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension of each head.  `model_dim = num_heads * head_dim`.
    pub head_dim: usize,
    /// Dropout rate (stored for future stochastic implementations).
    pub dropout_rate: f64,
    /// When `true` a causal upper-triangular mask is applied so that each
    /// position can only attend to itself and earlier positions.
    pub use_causal_mask: bool,
}

impl AttentionConfig {
    /// Return the full model dimension: `num_heads * head_dim`.
    pub fn model_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AttentionHead
// ─────────────────────────────────────────────────────────────────────────────

/// A single attention head holding three projection matrices.
///
/// Each projection is `[head_dim × model_dim]`.
#[derive(Debug, Clone)]
pub struct AttentionHead {
    /// Query projection `W_Q ∈ ℝ^{head_dim × model_dim}`.
    pub query_proj: AttentionMatrix,
    /// Key projection `W_K ∈ ℝ^{head_dim × model_dim}`.
    pub key_proj: AttentionMatrix,
    /// Value projection `W_V ∈ ℝ^{head_dim × model_dim}`.
    pub value_proj: AttentionMatrix,
}

// ─────────────────────────────────────────────────────────────────────────────
// AttentionOutput
// ─────────────────────────────────────────────────────────────────────────────

/// Result of one `forward` pass through [`AttentionMechanism`].
#[derive(Debug, Clone)]
pub struct AttentionOutput {
    /// Final output after concatenation and output projection.
    /// Shape: `seq_len × model_dim`.
    pub output: AttentionMatrix,
    /// Per-head attention weight matrices. Length = `num_heads`.
    /// Each entry has shape `seq_len × seq_len`.
    pub attention_weights: Vec<AttentionMatrix>,
    /// Per-head output matrices before concatenation.  Length = `num_heads`.
    /// Each entry has shape `seq_len × head_dim`.
    pub head_outputs: Vec<AttentionMatrix>,
}

// ─────────────────────────────────────────────────────────────────────────────
// PositionalEncoding
// ─────────────────────────────────────────────────────────────────────────────

/// Sinusoidal positional encoding table (Vaswani et al. 2017).
///
/// `PE[pos][2i]   = sin(pos / 10000^(2i/d))`
/// `PE[pos][2i+1] = cos(pos / 10000^(2i/d))`
#[derive(Debug, Clone)]
pub struct PositionalEncoding {
    /// Maximum sequence length supported.
    pub max_seq_len: usize,
    /// Dimensionality of the encoding vectors (= `model_dim`).
    pub encoding_dim: usize,
    /// Pre-computed encoding table — shape `max_seq_len × encoding_dim`.
    pub encodings: AttentionMatrix,
}

impl PositionalEncoding {
    /// Compute the sinusoidal encoding table.
    pub fn new(max_seq_len: usize, encoding_dim: usize) -> Self {
        let mut enc = AttentionMatrix::zeros(max_seq_len, encoding_dim);
        for pos in 0..max_seq_len {
            for i in 0..encoding_dim {
                let half_i = (i / 2) as f64;
                let denom = 10000_f64.powf(2.0 * half_i / encoding_dim.max(1) as f64);
                let angle = pos as f64 / denom;
                let v = if i % 2 == 0 { angle.sin() } else { angle.cos() };
                enc.set(pos, i, v);
            }
        }
        Self {
            max_seq_len,
            encoding_dim,
            encodings: enc,
        }
    }

    /// Extract the first `seq_len` rows as an `AttentionMatrix`.
    pub fn slice(&self, seq_len: usize) -> AttentionMatrix {
        let n = seq_len.min(self.max_seq_len);
        let mut out = AttentionMatrix::zeros(n, self.encoding_dim);
        for r in 0..n {
            for c in 0..self.encoding_dim {
                out.values[r * self.encoding_dim + c] =
                    self.encodings.values[r * self.encoding_dim + c];
            }
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AttnStats
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime statistics for [`AttentionMechanism`].
#[derive(Debug, Clone)]
pub struct AttnStats {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Per-head dimensionality.
    pub head_dim: usize,
    /// Full model dimension: `num_heads * head_dim`.
    pub model_dim: usize,
    /// Number of times `forward` has been called successfully.
    pub forward_count: u64,
    /// Maximum sequence length supported by the positional encoding.
    pub max_seq_len: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// AttentionMechanism — production-grade multi-head attention
// ─────────────────────────────────────────────────────────────────────────────

/// Production-grade multi-head scaled dot-product attention with sinusoidal
/// positional encoding, causal masking, learned projection matrices, and
/// attention pattern analysis utilities.
pub struct AttentionMechanism {
    /// Configuration (num_heads, head_dim, …).
    pub config: AttentionConfig,
    /// Per-head projection matrices (`W_Q`, `W_K`, `W_V`).
    pub heads: Vec<AttentionHead>,
    /// Output projection `W_O ∈ ℝ^{model_dim × model_dim}`.
    pub output_proj: AttentionMatrix,
    /// Pre-computed sinusoidal positional encoding.
    pub pos_enc: PositionalEncoding,
    /// Monotonically increasing counter of successful `forward` calls.
    pub forward_count: u64,
}

impl AttentionMechanism {
    /// Construct a new `AttentionMechanism` with constant-initialised weights.
    ///
    /// All projection matrices are initialised to `1.0 / sqrt(model_dim)` for
    /// deterministic behaviour (no randomness dependency).
    pub fn new(config: AttentionConfig, max_seq_len: usize) -> Self {
        let model_dim = config.model_dim();
        let head_dim = config.head_dim;
        let init_val = if model_dim > 0 {
            1.0 / (model_dim as f64).sqrt()
        } else {
            0.0
        };

        let make_proj = |rows: usize, cols: usize| {
            let mut m = AttentionMatrix::zeros(rows, cols);
            for v in m.values.iter_mut() {
                *v = init_val;
            }
            m
        };

        let heads: Vec<AttentionHead> = (0..config.num_heads)
            .map(|_| AttentionHead {
                query_proj: make_proj(head_dim, model_dim),
                key_proj: make_proj(head_dim, model_dim),
                value_proj: make_proj(head_dim, model_dim),
            })
            .collect();

        let output_proj = make_proj(model_dim, model_dim);
        let pos_enc = PositionalEncoding::new(max_seq_len, model_dim);

        Self {
            config,
            heads,
            output_proj,
            pos_enc,
            forward_count: 0,
        }
    }

    /// Return a statistics snapshot.
    pub fn stats(&self) -> AttnStats {
        AttnStats {
            num_heads: self.config.num_heads,
            head_dim: self.config.head_dim,
            model_dim: self.config.model_dim(),
            forward_count: self.forward_count,
            max_seq_len: self.pos_enc.max_seq_len,
        }
    }

    /// Scaled dot-product attention: `softmax(Q·Kᵀ / sqrt(head_dim)) · V`.
    ///
    /// # Arguments
    ///
    /// * `q` — shape `seq_len × head_dim`.
    /// * `k` — shape `seq_len × head_dim`.
    /// * `v` — shape `seq_len × head_dim`.
    /// * `mask` — optional `seq_len × seq_len` matrix; positions where the
    ///   value equals `1.0` receive a score of `-1e9` before the softmax.
    ///
    /// # Returns
    ///
    /// `(output [seq_len × head_dim], weights [seq_len × seq_len])`.
    pub fn scaled_dot_product(
        &self,
        q: &AttentionMatrix,
        k: &AttentionMatrix,
        v: &AttentionMatrix,
        mask: Option<&AttentionMatrix>,
    ) -> Result<(AttentionMatrix, AttentionMatrix), AttnError> {
        let seq_len = q.rows;
        let scale = if self.config.head_dim > 0 {
            (self.config.head_dim as f64).sqrt()
        } else {
            1.0
        };

        // scores = Q @ K^T  →  seq_len × seq_len
        let k_t = k.transpose();
        let mut scores = AttentionMatrix::matmul(q, &k_t)?;

        // Scale and optional masking.
        for r in 0..seq_len {
            for c in 0..seq_len {
                let idx = r * seq_len + c;
                scores.values[idx] /= scale;
                if let Some(m) = mask {
                    if m.get(r, c) == 1.0 {
                        scores.values[idx] = -1e9;
                    }
                }
            }
        }

        // Row-wise softmax.
        let weights = scores.softmax_rows();

        // output = weights @ V  →  seq_len × head_dim
        let output = AttentionMatrix::matmul(&weights, v)?;

        Ok((output, weights))
    }

    /// Build a causal (upper-triangular) mask where `mask[i][j] = 1.0` iff `j > i`.
    pub fn causal_mask(seq_len: usize) -> AttentionMatrix {
        let mut m = AttentionMatrix::zeros(seq_len, seq_len);
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                m.set(i, j, 1.0);
            }
        }
        m
    }

    /// Run a full multi-head attention forward pass.
    ///
    /// 1. Adds sinusoidal positional encoding to the input.
    /// 2. For each head: projects to Q, K, V; runs scaled dot-product attention.
    /// 3. Concatenates head outputs horizontally.
    /// 4. Applies the output projection.
    /// 5. Increments `forward_count`.
    pub fn forward(&mut self, input: &AttentionMatrix) -> Result<AttentionOutput, AttnError> {
        let seq_len = input.rows;
        if seq_len == 0 {
            return Err(AttnError::EmptyInput);
        }
        let model_dim = self.config.model_dim();
        if input.cols != model_dim {
            return Err(AttnError::DimensionMismatch {
                op: "forward".to_string(),
                expected: format!("input.cols == {model_dim}"),
                got: format!("input.cols == {}", input.cols),
            });
        }

        // Add positional encoding.
        let pos_slice = self.pos_enc.slice(seq_len);
        let x = input.add_pos_enc(&pos_slice)?;

        let mask_opt: Option<AttentionMatrix> = if self.config.use_causal_mask {
            Some(Self::causal_mask(seq_len))
        } else {
            None
        };

        let mut head_out_list: Vec<AttentionMatrix> = Vec::with_capacity(self.config.num_heads);
        let mut weight_list: Vec<AttentionMatrix> = Vec::with_capacity(self.config.num_heads);

        for head in &self.heads {
            // Q = x @ W_Q^T  →  seq_len × head_dim
            let wq_t = head.query_proj.transpose();
            let wk_t = head.key_proj.transpose();
            let wv_t = head.value_proj.transpose();

            let q = AttentionMatrix::matmul(&x, &wq_t)?;
            let k = AttentionMatrix::matmul(&x, &wk_t)?;
            let v = AttentionMatrix::matmul(&x, &wv_t)?;

            let (h_out, h_weights) = self.scaled_dot_product(&q, &k, &v, mask_opt.as_ref())?;

            head_out_list.push(h_out);
            weight_list.push(h_weights);
        }

        // Concatenate head outputs → seq_len × model_dim.
        let concat = AttentionMatrix::hconcat(&head_out_list)?;

        // Apply output projection: final = concat @ W_O^T  → seq_len × model_dim.
        let wo_t = self.output_proj.transpose();
        let final_output = AttentionMatrix::matmul(&concat, &wo_t)?;

        self.forward_count += 1;

        Ok(AttentionOutput {
            output: final_output,
            attention_weights: weight_list,
            head_outputs: head_out_list,
        })
    }

    /// Compute per-row Shannon entropy of an attention weight matrix.
    ///
    /// `H[i] = -Σ_j  w[i][j] * log(w[i][j] + 1e-10)`
    pub fn attention_entropy(weights: &AttentionMatrix) -> Vec<f64> {
        (0..weights.rows)
            .map(|r| {
                let start = r * weights.cols;
                let end = start + weights.cols;
                weights.values[start..end]
                    .iter()
                    .map(|&w| -w * (w + 1e-10_f64).ln())
                    .sum()
            })
            .collect()
    }

    /// Return the column index of the maximum weight in each row (argmax).
    pub fn peak_attention(weights: &AttentionMatrix) -> Vec<usize> {
        (0..weights.rows)
            .map(|r| {
                let start = r * weights.cols;
                let end = start + weights.cols;
                weights.values[start..end]
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Lightweight / simple API (kept for backward compatibility and convenience)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`SimpleAttentionMechanism`].
#[derive(Debug, Clone)]
pub struct SimpleAttentionConfig {
    /// Number of attention heads for multi-head attention.
    pub num_heads: usize,
    /// Dimension of each individual head.
    pub head_dim: usize,
    /// Dropout rate (currently stored for future use; no stochastic drop
    /// is applied during deterministic inference).
    pub dropout_rate: f64,
    /// When `true` a causal (upper-triangular) mask is applied so that each
    /// position can only attend to itself and earlier positions.
    pub causal_mask: bool,
    /// Override the default scale factor `1 / sqrt(head_dim)`.
    pub scale: Option<f64>,
}

/// Result of one simple attention computation.
#[derive(Debug, Clone)]
pub struct SimpleAttentionOutput {
    /// Shape: `seq_len × d_model`.
    pub output: Vec<Vec<f64>>,
    /// Averaged attention weights across heads. Shape: `seq_len × seq_len`.
    pub attention_weights: Vec<Vec<f64>>,
}

/// Running statistics collected by [`SimpleAttentionMechanism`].
#[derive(Debug, Clone, Default)]
pub struct SimpleAttentionStats {
    /// Total number of times [`SimpleAttentionMechanism::attend`] was called.
    pub total_calls: u64,
    /// Cumulative token count across all calls (= sum of sequence lengths).
    pub total_tokens: u64,
    /// Rolling arithmetic mean of sequence lengths seen so far.
    pub avg_seq_len: f64,
}

/// Lightweight multi-head attention engine.
///
/// Manages configuration and accumulates runtime statistics. Stateless with
/// respect to learned weights — callers supply their own Q, K, V projections.
pub struct SimpleAttentionMechanism {
    config: SimpleAttentionConfig,
    stats: SimpleAttentionStats,
}

impl SimpleAttentionMechanism {
    /// Construct a new mechanism from the given configuration.
    pub fn new(config: SimpleAttentionConfig) -> Self {
        Self {
            config,
            stats: SimpleAttentionStats::default(),
        }
    }

    /// Return a reference to the accumulated runtime statistics.
    pub fn stats(&self) -> &SimpleAttentionStats {
        &self.stats
    }

    /// Run multi-head attention using the stored configuration.
    ///
    /// `queries`, `keys`, and `values` must all have shape
    /// `seq_len × d_model` where `d_model = num_heads × head_dim`.
    ///
    /// The returned [`SimpleAttentionOutput`] contains:
    /// * `output` — the concatenated per-head results, shape `seq_len × d_model`.
    /// * `attention_weights` — the **mean** attention weight matrix across all
    ///   heads, shape `seq_len × seq_len`.
    pub fn attend(
        &mut self,
        queries: &[Vec<f64>],
        keys: &[Vec<f64>],
        values: &[Vec<f64>],
    ) -> SimpleAttentionOutput {
        let seq_len = queries.len();

        self.stats.total_calls += 1;
        self.stats.total_tokens += seq_len as u64;
        let n = self.stats.total_calls as f64;
        self.stats.avg_seq_len += (seq_len as f64 - self.stats.avg_seq_len) / n;

        if seq_len == 0 {
            return SimpleAttentionOutput {
                output: vec![],
                attention_weights: vec![],
            };
        }

        let scale = self
            .config
            .scale
            .unwrap_or_else(|| 1.0 / (self.config.head_dim as f64).sqrt());

        let causal = if self.config.causal_mask {
            Some(causal_mask(seq_len))
        } else {
            None
        };
        let mask_ref = causal.as_deref();

        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let d_model = num_heads * head_dim;

        let mut head_outputs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(num_heads);
        let mut weight_sum: Vec<Vec<f64>> = vec![vec![0.0; seq_len]; seq_len];

        for h in 0..num_heads {
            let col_start = h * head_dim;
            let col_end = col_start + head_dim;

            let q_h = slice_cols(queries, col_start, col_end);
            let k_h = slice_cols(keys, col_start, col_end);
            let v_h = slice_cols(values, col_start, col_end);

            let out_h = scaled_dot_product_attention(&q_h, &k_h, &v_h, scale, mask_ref);

            for (i, row) in weight_sum.iter_mut().enumerate().take(seq_len) {
                for (j, cell) in row.iter_mut().enumerate().take(seq_len) {
                    *cell += out_h.attention_weights[i].get(j).copied().unwrap_or(0.0);
                }
            }

            head_outputs.push(out_h.output);
        }

        let n_heads_f = num_heads as f64;
        let attention_weights: Vec<Vec<f64>> = weight_sum
            .iter()
            .map(|row| row.iter().map(|w| w / n_heads_f).collect())
            .collect();

        let mut output = vec![vec![0.0; d_model]; seq_len];
        for (h, head_out) in head_outputs.iter().enumerate() {
            let col_start = h * head_dim;
            for (i, row) in head_out.iter().enumerate() {
                for (j, val) in row.iter().enumerate() {
                    output[i][col_start + j] = *val;
                }
            }
        }

        SimpleAttentionOutput {
            output,
            attention_weights,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Free-standing core functions (public utility API)
// ─────────────────────────────────────────────────────────────────────────────

/// Scaled dot-product attention (free function, `Vec<Vec<f64>>` API).
///
/// Computes `softmax(Q · Kᵀ / scale) · V`.
///
/// Positions where `mask[i][j] == true` are set to `-1e9` before the softmax.
pub fn scaled_dot_product_attention(
    queries: &[Vec<f64>],
    keys: &[Vec<f64>],
    values: &[Vec<f64>],
    scale: f64,
    mask: Option<&[Vec<bool>]>,
) -> SimpleAttentionOutput {
    let seq_len = queries.len();
    if seq_len == 0 {
        return SimpleAttentionOutput {
            output: vec![],
            attention_weights: vec![],
        };
    }

    let k_t = transpose(keys);
    let mut scores = matmul(queries, &k_t);

    let safe_scale = if scale.abs() < 1e-12 { 1.0 } else { scale };
    for (i, row) in scores.iter_mut().enumerate().take(seq_len) {
        for (j, cell) in row.iter_mut().enumerate().take(seq_len) {
            *cell /= safe_scale;
            if let Some(m) = mask {
                if m.get(i).and_then(|r| r.get(j)).copied().unwrap_or(false) {
                    *cell = -1e9;
                }
            }
        }
    }

    let attention_weights: Vec<Vec<f64>> = scores.iter().map(|row| softmax_1d(row)).collect();
    let output = matmul(&attention_weights, values);

    SimpleAttentionOutput {
        output,
        attention_weights,
    }
}

/// Numerically stable softmax over a 1-D slice of logits.
///
/// Uses the max-subtraction trick to avoid floating-point overflow.
pub fn softmax_1d(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return vec![];
    }

    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let exps: Vec<f64> = logits.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();

    if sum == 0.0 {
        let n = logits.len() as f64;
        return vec![1.0 / n; logits.len()];
    }

    exps.iter().map(|e| e / sum).collect()
}

/// Standard matrix multiplication: `C = A · B`.
///
/// `A` must be `m × k`, `B` must be `k × n`; returns `m × n`.
pub fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 || b.is_empty() {
        return vec![];
    }

    let k = b.len();
    let n = b.first().map(|r| r.len()).unwrap_or(0);

    let mut result = vec![vec![0.0; n]; m];
    for i in 0..m {
        let a_row = &a[i];
        let a_len = a_row.len().min(k);
        for p in 0..a_len {
            let a_val = a_row[p];
            if a_val == 0.0 {
                continue;
            }
            let b_row = &b[p];
            let b_len = b_row.len().min(n);
            for j in 0..b_len {
                result[i][j] += a_val * b_row[j];
            }
        }
    }
    result
}

/// Transpose a 2-D matrix represented as `Vec<Vec<f64>>`.
pub fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = m.len();
    if rows == 0 {
        return vec![];
    }

    let cols = m.iter().map(|r| r.len()).max().unwrap_or(0);
    if cols == 0 {
        return vec![];
    }

    let mut out = vec![vec![0.0; rows]; cols];
    for (i, row) in m.iter().enumerate() {
        for (j, val) in row.iter().enumerate() {
            out[j][i] = *val;
        }
    }
    out
}

/// Build a causal (autoregressive) boolean mask of size `seq_len × seq_len`.
///
/// `mask[i][j] == true` iff `j > i`.
pub fn causal_mask(seq_len: usize) -> Vec<Vec<bool>> {
    (0..seq_len)
        .map(|i| (0..seq_len).map(|j| j > i).collect())
        .collect()
}

// ── Private helpers ─────────────────────────────────────────────────────────

fn slice_cols(m: &[Vec<f64>], col_start: usize, col_end: usize) -> Vec<Vec<f64>> {
    m.iter()
        .map(|row| {
            (col_start..col_end)
                .map(|c| row.get(c).copied().unwrap_or(0.0))
                .collect()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::attention_mechanism::{
        causal_mask, matmul, scaled_dot_product_attention, softmax_1d, transpose, AttentionConfig,
        AttentionMatrix, AttentionMechanism, AttnError, PositionalEncoding, SimpleAttentionConfig,
        SimpleAttentionMechanism,
    };

    // ── softmax_1d ────────────────────────────────────────────────────────────

    #[test]
    fn softmax_sums_to_one_uniform() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let result = softmax_1d(&logits);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "softmax sum = {sum}");
    }

    #[test]
    fn softmax_sums_to_one_negative_values() {
        let logits = vec![-100.0, -50.0, -1.0];
        let result = softmax_1d(&logits);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "softmax sum = {sum}");
    }

    #[test]
    fn softmax_numerical_stability_large_values() {
        let logits = vec![1e308, 1e308 + 1.0, 1e308 + 2.0];
        let result = softmax_1d(&logits);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "softmax sum = {sum}");
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn softmax_numerical_stability_very_negative() {
        let logits = vec![-1e308, -1e308, -1e308];
        let result = softmax_1d(&logits);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "softmax sum = {sum}");
    }

    #[test]
    fn softmax_single_element() {
        let result = softmax_1d(&[42.0]);
        assert!((result[0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn softmax_empty() {
        assert!(softmax_1d(&[]).is_empty());
    }

    #[test]
    fn softmax_monotone_order() {
        let logits = vec![1.0, 3.0, 2.0];
        let result = softmax_1d(&logits);
        assert!(result[1] > result[2]);
        assert!(result[2] > result[0]);
    }

    // ── matmul ────────────────────────────────────────────────────────────────

    #[test]
    fn matmul_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let c = matmul(&a, &b);
        assert!((c[0][0] - 5.0).abs() < 1e-15);
        assert!((c[0][1] - 6.0).abs() < 1e-15);
        assert!((c[1][0] - 7.0).abs() < 1e-15);
        assert!((c[1][1] - 8.0).abs() < 1e-15);
    }

    #[test]
    fn matmul_known_values() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let c = matmul(&a, &b);
        assert!((c[0][0] - 19.0).abs() < 1e-12);
        assert!((c[0][1] - 22.0).abs() < 1e-12);
        assert!((c[1][0] - 43.0).abs() < 1e-12);
        assert!((c[1][1] - 50.0).abs() < 1e-12);
    }

    #[test]
    fn matmul_non_square() {
        let a = vec![vec![1.0, 0.0, 2.0], vec![0.0, 3.0, 1.0]];
        let b = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![2.0, 3.0]];
        let c = matmul(&a, &b);
        assert!((c[0][0] - 5.0).abs() < 1e-12);
        assert!((c[0][1] - 6.0).abs() < 1e-12);
        assert!((c[1][0] - 2.0).abs() < 1e-12);
        assert!((c[1][1] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn matmul_empty_returns_empty() {
        let empty: Vec<Vec<f64>> = vec![];
        assert!(matmul(&empty, &empty).is_empty());
    }

    // ── transpose ─────────────────────────────────────────────────────────────

    #[test]
    fn transpose_square() {
        let m = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let t = transpose(&m);
        assert!((t[0][0] - 1.0).abs() < 1e-15);
        assert!((t[0][1] - 3.0).abs() < 1e-15);
        assert!((t[1][0] - 2.0).abs() < 1e-15);
        assert!((t[1][1] - 4.0).abs() < 1e-15);
    }

    #[test]
    fn transpose_rectangular() {
        let m = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let t = transpose(&m);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].len(), 2);
        assert!((t[1][1] - 5.0).abs() < 1e-15);
        assert!((t[2][0] - 3.0).abs() < 1e-15);
    }

    #[test]
    fn transpose_double_returns_original() {
        let m = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let tt = transpose(&transpose(&m));
        for (r, row) in m.iter().enumerate() {
            for (c, val) in row.iter().enumerate() {
                assert!((tt[r][c] - val).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn transpose_empty() {
        let empty: Vec<Vec<f64>> = vec![];
        assert!(transpose(&empty).is_empty());
    }

    // ── causal_mask ───────────────────────────────────────────────────────────

    #[test]
    fn causal_mask_upper_triangle_masked() {
        let mask = causal_mask(4);
        for (i, row) in mask.iter().enumerate() {
            for (j, &masked) in row.iter().enumerate().take(i + 1) {
                assert!(!masked, "position ({i},{j}) should NOT be masked");
            }
        }
        for (i, row) in mask.iter().enumerate() {
            for (j, &masked) in row.iter().enumerate().skip(i + 1) {
                assert!(masked, "position ({i},{j}) should be masked");
            }
        }
    }

    #[test]
    fn causal_mask_size_one() {
        let mask = causal_mask(1);
        assert_eq!(mask.len(), 1);
        assert!(!mask[0][0]);
    }

    #[test]
    fn causal_mask_dimensions() {
        let n = 6;
        let mask = causal_mask(n);
        assert_eq!(mask.len(), n);
        assert!(mask.iter().all(|row| row.len() == n));
    }

    // ── scaled_dot_product_attention ──────────────────────────────────────────

    #[test]
    fn sdp_output_shape() {
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let k = q.clone();
        let v = q.clone();
        let out = scaled_dot_product_attention(&q, &k, &v, 1.0, None);
        assert_eq!(out.output.len(), 3);
        assert_eq!(out.output[0].len(), 2);
        assert_eq!(out.attention_weights.len(), 3);
        assert_eq!(out.attention_weights[0].len(), 3);
    }

    #[test]
    fn sdp_attention_weights_sum_to_one_per_row() {
        let q = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let k = q.clone();
        let v = q.clone();
        let out = scaled_dot_product_attention(&q, &k, &v, 1.0, None);
        for (i, row) in out.attention_weights.iter().enumerate() {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "row {i} sums to {s}");
        }
    }

    #[test]
    fn sdp_causal_mask_suppresses_future() {
        let q = vec![vec![1.0], vec![1.0], vec![1.0]];
        let k = q.clone();
        let v = vec![vec![10.0], vec![20.0], vec![30.0]];
        let mask = causal_mask(3);
        let out = scaled_dot_product_attention(&q, &k, &v, 1.0, Some(&mask));
        assert!(out.attention_weights[0][1] < 1e-6);
        assert!(out.attention_weights[0][2] < 1e-6);
        assert!(out.attention_weights[2][0] > 1e-6);
        assert!(out.attention_weights[2][1] > 1e-6);
    }

    #[test]
    fn sdp_single_token() {
        let q = vec![vec![1.0, 2.0, 3.0]];
        let k = q.clone();
        let v = vec![vec![5.0, 6.0, 7.0]];
        let out = scaled_dot_product_attention(&q, &k, &v, 1.0, None);
        assert_eq!(out.output.len(), 1);
        assert!((out.attention_weights[0][0] - 1.0).abs() < 1e-12);
        assert!((out.output[0][0] - 5.0).abs() < 1e-12);
    }

    // ── SimpleAttentionMechanism ───────────────────────────────────────────────

    fn make_simple(heads: usize, head_dim: usize, causal: bool) -> SimpleAttentionMechanism {
        SimpleAttentionMechanism::new(SimpleAttentionConfig {
            num_heads: heads,
            head_dim,
            dropout_rate: 0.0,
            causal_mask: causal,
            scale: None,
        })
    }

    #[test]
    fn simple_attend_output_shape() {
        let mut attn = make_simple(2, 4, false);
        let d_model = 8;
        let seq_len = 5;
        let q = vec![vec![1.0; d_model]; seq_len];
        let out = attn.attend(&q, &q, &q);
        assert_eq!(out.output.len(), seq_len);
        assert_eq!(out.output[0].len(), d_model);
        assert_eq!(out.attention_weights.len(), seq_len);
    }

    #[test]
    fn simple_attend_stats_tracking() {
        let mut attn = make_simple(1, 4, false);
        let q = vec![vec![1.0; 4]; 3];
        attn.attend(&q, &q, &q);
        attn.attend(&q, &q, &q);
        assert_eq!(attn.stats().total_calls, 2);
        assert_eq!(attn.stats().total_tokens, 6);
    }

    // ── AttentionMatrix ───────────────────────────────────────────────────────

    #[test]
    fn attn_matrix_zeros() {
        let m = AttentionMatrix::zeros(3, 4);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 4);
        assert!(m.values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn attn_matrix_get_set() {
        let mut m = AttentionMatrix::zeros(2, 3);
        m.set(0, 1, 7.0);
        assert!((m.get(0, 1) - 7.0).abs() < 1e-15);
        assert_eq!(m.get(0, 0), 0.0);
        // Out-of-bounds get returns 0.0, set is no-op.
        assert_eq!(m.get(10, 10), 0.0);
        m.set(10, 10, 99.0);
    }

    #[test]
    fn attn_matrix_matmul_correct() {
        let mut a = AttentionMatrix::zeros(2, 2);
        a.set(0, 0, 1.0);
        a.set(0, 1, 2.0);
        a.set(1, 0, 3.0);
        a.set(1, 1, 4.0);
        let mut b = AttentionMatrix::zeros(2, 2);
        b.set(0, 0, 5.0);
        b.set(0, 1, 6.0);
        b.set(1, 0, 7.0);
        b.set(1, 1, 8.0);
        let c = AttentionMatrix::matmul(&a, &b).expect("test: should succeed");
        assert!((c.get(0, 0) - 19.0).abs() < 1e-12);
        assert!((c.get(0, 1) - 22.0).abs() < 1e-12);
        assert!((c.get(1, 0) - 43.0).abs() < 1e-12);
        assert!((c.get(1, 1) - 50.0).abs() < 1e-12);
    }

    #[test]
    fn attn_matrix_matmul_dim_mismatch() {
        let a = AttentionMatrix::zeros(2, 3);
        let b = AttentionMatrix::zeros(2, 2); // inner dims don't match
        let result = AttentionMatrix::matmul(&a, &b);
        assert!(matches!(result, Err(AttnError::DimensionMismatch { .. })));
    }

    #[test]
    fn attn_matrix_transpose() {
        let mut m = AttentionMatrix::zeros(2, 3);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(0, 2, 3.0);
        m.set(1, 0, 4.0);
        m.set(1, 1, 5.0);
        m.set(1, 2, 6.0);
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert!((t.get(0, 0) - 1.0).abs() < 1e-15);
        assert!((t.get(1, 0) - 2.0).abs() < 1e-15);
        assert!((t.get(2, 1) - 6.0).abs() < 1e-15);
    }

    #[test]
    fn attn_matrix_softmax_rows_sums_to_one() {
        let mut m = AttentionMatrix::zeros(3, 4);
        for r in 0..3 {
            for c in 0..4 {
                m.set(r, c, ((r * 4 + c) as f64) * 0.5);
            }
        }
        let s = m.softmax_rows();
        for r in 0..3 {
            let row_sum: f64 = (0..4).map(|c| s.get(r, c)).sum();
            assert!((row_sum - 1.0).abs() < 1e-12, "row {r} sum = {row_sum}");
        }
    }

    // ── PositionalEncoding ────────────────────────────────────────────────────

    #[test]
    fn pos_enc_shape() {
        let pe = PositionalEncoding::new(64, 8);
        assert_eq!(pe.encodings.rows, 64);
        assert_eq!(pe.encodings.cols, 8);
    }

    #[test]
    fn pos_enc_position_zero_even_dims_zero() {
        // sin(0) = 0 for all even dimensions at position 0.
        let pe = PositionalEncoding::new(10, 8);
        for i in 0..4 {
            let val = pe.encodings.get(0, i * 2);
            assert!(val.abs() < 1e-12, "PE[0][{i}*2] = {val}");
        }
    }

    #[test]
    fn pos_enc_position_zero_odd_dims_one() {
        // cos(0) = 1.0 for all odd dimensions at position 0.
        let pe = PositionalEncoding::new(10, 8);
        for i in 0..4 {
            let val = pe.encodings.get(0, i * 2 + 1);
            assert!((val - 1.0).abs() < 1e-12, "PE[0][{i}*2+1] = {val}");
        }
    }

    #[test]
    fn pos_enc_slice_correct_rows() {
        let pe = PositionalEncoding::new(64, 8);
        let sliced = pe.slice(5);
        assert_eq!(sliced.rows, 5);
        assert_eq!(sliced.cols, 8);
    }

    #[test]
    fn pos_enc_values_bounded() {
        // Sinusoidal encodings must lie in [-1, 1].
        let pe = PositionalEncoding::new(100, 16);
        for v in &pe.encodings.values {
            assert!(
                *v >= -1.0 - 1e-12 && *v <= 1.0 + 1e-12,
                "PE value out of bounds: {v}"
            );
        }
    }

    // ── AttentionMechanism (production) ───────────────────────────────────────

    fn make_attn(
        heads: usize,
        head_dim: usize,
        causal: bool,
        max_len: usize,
    ) -> AttentionMechanism {
        AttentionMechanism::new(
            AttentionConfig {
                num_heads: heads,
                head_dim,
                dropout_rate: 0.0,
                use_causal_mask: causal,
            },
            max_len,
        )
    }

    #[test]
    fn attn_config_model_dim() {
        let cfg = AttentionConfig {
            num_heads: 4,
            head_dim: 8,
            dropout_rate: 0.0,
            use_causal_mask: false,
        };
        assert_eq!(cfg.model_dim(), 32);
    }

    #[test]
    fn attn_forward_output_shape() {
        let mut attn = make_attn(2, 4, false, 64);
        let input = AttentionMatrix::zeros(3, 8);
        let out = attn.forward(&input).expect("test: should succeed");
        assert_eq!(out.output.rows, 3);
        assert_eq!(out.output.cols, 8);
        assert_eq!(out.attention_weights.len(), 2);
        assert_eq!(out.head_outputs.len(), 2);
    }

    #[test]
    fn attn_forward_weight_shape() {
        let mut attn = make_attn(3, 4, false, 32);
        let input = AttentionMatrix::zeros(5, 12);
        let out = attn.forward(&input).expect("test: should succeed");
        for w in &out.attention_weights {
            assert_eq!(w.rows, 5);
            assert_eq!(w.cols, 5);
        }
    }

    #[test]
    fn attn_forward_weights_sum_to_one() {
        let mut attn = make_attn(2, 4, false, 64);
        let mut input = AttentionMatrix::zeros(4, 8);
        for i in 0..4 {
            for j in 0..8 {
                input.set(i, j, (i * 8 + j) as f64 * 0.01);
            }
        }
        let out = attn.forward(&input).expect("test: should succeed");
        for (h, w) in out.attention_weights.iter().enumerate() {
            for r in 0..w.rows {
                let sum: f64 = (0..w.cols).map(|c| w.get(r, c)).sum();
                assert!((sum - 1.0).abs() < 1e-10, "head {h} row {r} sum = {sum}");
            }
        }
    }

    #[test]
    fn attn_forward_increments_count() {
        let mut attn = make_attn(1, 4, false, 16);
        let input = AttentionMatrix::zeros(2, 4);
        attn.forward(&input).expect("test: should succeed");
        attn.forward(&input).expect("test: should succeed");
        assert_eq!(attn.forward_count, 2);
    }

    #[test]
    fn attn_forward_stats() {
        let attn = make_attn(2, 4, false, 32);
        let s = attn.stats();
        assert_eq!(s.num_heads, 2);
        assert_eq!(s.head_dim, 4);
        assert_eq!(s.model_dim, 8);
        assert_eq!(s.forward_count, 0);
        assert_eq!(s.max_seq_len, 32);
    }

    #[test]
    fn attn_forward_empty_input_error() {
        let mut attn = make_attn(1, 4, false, 16);
        let empty = AttentionMatrix::zeros(0, 4);
        let result = attn.forward(&empty);
        assert!(matches!(result, Err(AttnError::EmptyInput)));
    }

    #[test]
    fn attn_forward_dim_mismatch_error() {
        let mut attn = make_attn(2, 4, false, 16);
        // model_dim should be 8, but we pass 6 cols.
        let bad = AttentionMatrix::zeros(3, 6);
        let result = attn.forward(&bad);
        assert!(matches!(result, Err(AttnError::DimensionMismatch { .. })));
    }

    #[test]
    fn attn_forward_causal_mask() {
        let mut attn = make_attn(1, 4, true, 16);
        let mut input = AttentionMatrix::zeros(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                input.set(i, j, 1.0);
            }
        }
        let out = attn.forward(&input).expect("test: should succeed");
        let w = &out.attention_weights[0];
        // Token 0 should not attend to future tokens (weights should be near 0).
        for j in 1..4 {
            assert!(w.get(0, j) < 1e-5, "causal: w[0][{j}] = {}", w.get(0, j));
        }
    }

    #[test]
    fn attn_causal_mask_matrix() {
        let m = AttentionMechanism::causal_mask(4);
        assert_eq!(m.rows, 4);
        assert_eq!(m.cols, 4);
        for i in 0..4 {
            for j in 0..=i {
                assert_eq!(m.get(i, j), 0.0, "({i},{j}) should be 0.0");
            }
            for j in (i + 1)..4 {
                assert_eq!(m.get(i, j), 1.0, "({i},{j}) should be 1.0");
            }
        }
    }

    #[test]
    fn attn_entropy_non_negative() {
        let mut attn = make_attn(1, 4, false, 16);
        let input = AttentionMatrix::zeros(3, 4);
        let out = attn.forward(&input).expect("test: should succeed");
        let h = AttentionMechanism::attention_entropy(&out.attention_weights[0]);
        assert_eq!(h.len(), 3);
        assert!(
            h.iter().all(|&e| e >= 0.0),
            "entropy must be non-negative: {h:?}"
        );
    }

    #[test]
    fn attn_entropy_uniform_distribution_is_max() {
        // Uniform distribution has maximum entropy = ln(n).
        let mut uniform = AttentionMatrix::zeros(1, 4);
        for c in 0..4 {
            uniform.set(0, c, 0.25);
        }
        let h = AttentionMechanism::attention_entropy(&uniform);
        let expected = (4_f64).ln();
        assert!(
            (h[0] - expected).abs() < 0.01,
            "entropy = {}, expected ≈ {expected}",
            h[0]
        );
    }

    #[test]
    fn attn_peak_attention_argmax() {
        let mut m = AttentionMatrix::zeros(2, 4);
        m.set(0, 2, 1.0); // row 0 peak at col 2
        m.set(1, 0, 1.0); // row 1 peak at col 0
        let peaks = AttentionMechanism::peak_attention(&m);
        assert_eq!(peaks[0], 2);
        assert_eq!(peaks[1], 0);
    }

    #[test]
    fn attn_head_count_in_output() {
        let num_heads = 4;
        let mut attn = make_attn(num_heads, 4, false, 32);
        let input = AttentionMatrix::zeros(3, 16);
        let out = attn.forward(&input).expect("test: should succeed");
        assert_eq!(out.attention_weights.len(), num_heads);
        assert_eq!(out.head_outputs.len(), num_heads);
    }

    #[test]
    fn attn_scaled_dot_product_output_shape() {
        let attn = make_attn(1, 4, false, 16);
        let q = AttentionMatrix::zeros(3, 4);
        let k = AttentionMatrix::zeros(3, 4);
        let v = AttentionMatrix::zeros(3, 4);
        let (out, weights) = attn
            .scaled_dot_product(&q, &k, &v, None)
            .expect("test: should succeed");
        assert_eq!(out.rows, 3);
        assert_eq!(out.cols, 4);
        assert_eq!(weights.rows, 3);
        assert_eq!(weights.cols, 3);
    }

    #[test]
    fn attn_error_display_empty_input() {
        let e = AttnError::EmptyInput;
        let s = e.to_string();
        assert!(s.contains("EmptyInput"));
    }

    #[test]
    fn attn_error_display_dim_mismatch() {
        let e = AttnError::DimensionMismatch {
            op: "test".to_string(),
            expected: "4".to_string(),
            got: "8".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("DimensionMismatch"));
    }

    #[test]
    fn attn_error_display_invalid_config() {
        let e = AttnError::InvalidConfig("num_heads must be > 0".to_string());
        let s = e.to_string();
        assert!(s.contains("InvalidConfig"));
    }

    #[test]
    fn attn_forward_large_sequence() {
        let mut attn = make_attn(2, 8, false, 256);
        let input = AttentionMatrix::zeros(32, 16);
        let out = attn.forward(&input).expect("test: should succeed");
        assert_eq!(out.output.rows, 32);
        assert_eq!(out.output.cols, 16);
    }

    #[test]
    fn attn_head_output_dim() {
        let mut attn = make_attn(3, 5, false, 32);
        let input = AttentionMatrix::zeros(4, 15);
        let out = attn.forward(&input).expect("test: should succeed");
        for h in &out.head_outputs {
            assert_eq!(h.rows, 4);
            assert_eq!(h.cols, 5);
        }
    }
}
