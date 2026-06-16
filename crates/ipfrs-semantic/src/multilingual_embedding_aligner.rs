//! Multilingual Embedding Aligner
//!
//! Maps embeddings from different language spaces to a shared cross-lingual space
//! using Procrustes alignment, Linear Regression, CCA, or Identity passthrough.

use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

use thiserror::Error;

// ─── Type aliases ─────────────────────────────────────────────────────────────

/// 8-byte language identifier (e.g. b"en\0\0\0\0\0\0")
pub type LangId = [u8; 8];

/// Convenience alias for `MultilingualEmbeddingAligner`
pub type MeaMultilingualEmbeddingAligner = MultilingualEmbeddingAligner;
/// Convenience alias for `LanguageSpace`
pub type MeaLanguageSpace = LanguageSpace;
/// Convenience alias for `AlignmentMatrix`
pub type MeaAlignmentMatrix = AlignmentMatrix;
/// Convenience alias for `MeaAlignerConfig`
pub type MeaAlignerCfg = MeaAlignerConfig;
/// Convenience alias for `MeaAlignerStats`
pub type MeaStats = MeaAlignerStats;

// ─── Errors ───────────────────────────────────────────────────────────────────

/// Errors produced by the multilingual embedding aligner.
#[derive(Debug, Error)]
pub enum MeaError {
    #[error("language '{0}' not found")]
    LanguageNotFound(String),

    #[error("embedding id {0} not found in language '{1}'")]
    EmbeddingNotFound(u64, String),

    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("not enough anchor pairs: need at least {min}, got {got}")]
    NotEnoughAnchors { min: usize, got: usize },

    #[error("alignment matrix not found for ({0}, {1})")]
    AlignmentNotFound(String, String),

    #[error("SVD power iteration did not converge")]
    SvdNotConverged,

    #[error("arithmetic error: {0}")]
    Arithmetic(String),

    #[error("empty embedding vector")]
    EmptyEmbedding,
}

// ─── Enums ────────────────────────────────────────────────────────────────────

/// Strategy used to compute the alignment matrix from anchor pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeaAlignmentMethod {
    /// Procrustes orthogonal alignment via SVD approximation (power iteration)
    #[default]
    Procrustes,
    /// Least-squares linear regression (W = (X^T X)^{-1} X^T Y)
    LinearRegression,
    /// Canonical Correlation Analysis (approximate)
    Cca,
    /// No-op: return the source vector unchanged (useful for same-language pairs)
    IdentityPassthrough,
}

// ─── Core data structures ─────────────────────────────────────────────────────

/// A named embedding space associated with one language.
#[derive(Debug, Clone)]
pub struct LanguageSpace {
    pub id: LangId,
    pub name: String,
    pub dim: usize,
    /// (embedding_id, vector)
    pub embeddings: Vec<(u64, Vec<f64>)>,
    pub centroid: Option<Vec<f64>>,
}

impl LanguageSpace {
    fn new(id: LangId, name: String, dim: usize) -> Self {
        Self {
            id,
            name,
            dim,
            embeddings: Vec::new(),
            centroid: None,
        }
    }

    fn lang_name(&self) -> String {
        String::from_utf8_lossy(&self.id)
            .trim_end_matches('\0')
            .to_string()
    }
}

/// Alignment transformation from one language space to another (row-major dim×dim).
#[derive(Debug, Clone)]
pub struct AlignmentMatrix {
    pub src_lang: LangId,
    pub tgt_lang: LangId,
    /// Row-major dim×dim matrix
    pub matrix: Vec<f64>,
    /// Mean cosine similarity on the anchor pairs after alignment
    pub quality: f64,
}

impl AlignmentMatrix {
    /// Apply matrix-vector product (dim×dim matrix × dim vector → dim vector).
    fn apply(&self, v: &[f64]) -> Result<Vec<f64>, MeaError> {
        let dim = v.len();
        let expected_len = dim * dim;
        if self.matrix.len() != expected_len {
            return Err(MeaError::DimensionMismatch {
                expected: expected_len,
                actual: self.matrix.len(),
            });
        }
        let mut out = vec![0.0_f64; dim];
        for (i, out_i) in out.iter_mut().enumerate() {
            let mut acc = 0.0_f64;
            let row_off = i * dim;
            for (j, vj) in v.iter().enumerate() {
                acc += self.matrix[row_off + j] * vj;
            }
            *out_i = acc;
        }
        Ok(out)
    }
}

/// Timestamped record of a past alignment operation.
#[derive(Debug, Clone)]
pub struct AlignmentRecord {
    pub ts: u64,
    pub src_lang: LangId,
    pub tgt_lang: LangId,
    pub n_anchors: usize,
    pub quality: f64,
}

/// Aggregate statistics about the aligner.
#[derive(Debug, Clone, Default)]
pub struct MeaAlignerStats {
    pub n_languages: usize,
    pub n_alignments: usize,
    pub n_history_records: usize,
    pub total_embeddings: usize,
    pub avg_alignment_quality: f64,
}

/// Configuration for `MultilingualEmbeddingAligner`.
#[derive(Debug, Clone)]
pub struct MeaAlignerConfig {
    pub dim: usize,
    pub normalize_embeddings: bool,
    pub alignment_method: MeaAlignmentMethod,
    pub min_anchor_pairs: usize,
}

impl Default for MeaAlignerConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            normalize_embeddings: true,
            alignment_method: MeaAlignmentMethod::Procrustes,
            min_anchor_pairs: 5,
        }
    }
}

// ─── PRNG helpers (pure Rust, no external deps) ───────────────────────────────

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ─── Math helpers ─────────────────────────────────────────────────────────────

#[inline]
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        0.0
    } else {
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }
}

/// L2-normalise a vector in-place. Returns false if the norm is zero.
#[inline]
fn normalize_vec(v: &mut [f64]) -> bool {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-12 {
        return false;
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
    true
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn lang_name_from_id(id: &LangId) -> String {
    String::from_utf8_lossy(id)
        .trim_end_matches('\0')
        .to_string()
}

// ─── Linear algebra helpers ───────────────────────────────────────────────────

/// Matrix multiply C = A * B where A is m×k, B is k×n (all row-major).
fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * n];
    for i in 0..m {
        for l in 0..k {
            let a_il = a[i * k + l];
            for j in 0..n {
                c[i * n + j] += a_il * b[l * n + j];
            }
        }
    }
    c
}

/// Transpose an m×n matrix (row-major) → n×m matrix (row-major).
fn transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut t = vec![0.0_f64; n * m];
    for i in 0..m {
        for j in 0..n {
            t[j * m + i] = a[i * n + j];
        }
    }
    t
}

/// Build the identity matrix of size d×d (row-major).
fn identity(d: usize) -> Vec<f64> {
    let mut m = vec![0.0_f64; d * d];
    for i in 0..d {
        m[i * d + i] = 1.0;
    }
    m
}

/// SVD result: (U matrix, singular values S, Vt matrix)
type SvdResult = Result<(Vec<f64>, Vec<f64>, Vec<f64>), MeaError>;

/// Power-iteration SVD to find the top-k singular triplets of an m×n matrix A.
/// Returns (U, S, V^T) each as flattened row-major matrices:
///   U  : m × k
///   S  : k (diagonal values)
///   Vt : k × n
fn svd_power_iteration(
    a: &[f64],
    m: usize,
    n: usize,
    k: usize,
    max_iter: usize,
    tol: f64,
    rng: &mut u64,
) -> SvdResult {
    // Deflation-based approach: extract one singular triplet at a time.
    let mut u_cols: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut sigmas: Vec<f64> = Vec::with_capacity(k);
    let mut v_cols: Vec<Vec<f64>> = Vec::with_capacity(k);

    // Working copy of A that we deflate.
    let mut a_work = a.to_vec();

    for _ in 0..k {
        // Random init for v (n-dim).
        let mut v = (0..n)
            .map(|_| {
                let r = xorshift64(rng);
                // Map to [-1, 1]
                (r as i64 as f64) / (i64::MAX as f64)
            })
            .collect::<Vec<f64>>();
        // Normalise.
        let vn: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if vn < 1e-12 {
            return Err(MeaError::SvdNotConverged);
        }
        for x in v.iter_mut() {
            *x /= vn;
        }

        let mut sigma = 0.0_f64;
        let mut u = vec![0.0_f64; m];

        for iter in 0..max_iter {
            // u = A * v (m-dim)
            for i in 0..m {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += a_work[i * n + j] * v[j];
                }
                u[i] = acc;
            }
            let new_sigma: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
            if new_sigma < 1e-12 {
                break;
            }
            for x in u.iter_mut() {
                *x /= new_sigma;
            }

            // v_new = A^T * u (n-dim)
            let mut v_new = vec![0.0_f64; n];
            for j in 0..n {
                let mut acc = 0.0;
                for i in 0..m {
                    acc += a_work[i * n + j] * u[i];
                }
                v_new[j] = acc;
            }
            let v_norm: f64 = v_new.iter().map(|x| x * x).sum::<f64>().sqrt();
            if v_norm < 1e-12 {
                break;
            }
            for x in v_new.iter_mut() {
                *x /= v_norm;
            }

            // Check convergence.
            let diff: f64 = v_new.iter().zip(v.iter()).map(|(a, b)| (a - b).abs()).sum();
            v = v_new;
            sigma = new_sigma;
            if iter > 0 && diff < tol {
                break;
            }
        }

        if sigma < 1e-12 {
            // Remaining rank is zero; fill with identity-like vectors.
            let mut ui = vec![0.0_f64; m];
            let mut vi = vec![0.0_f64; n];
            let idx = u_cols.len().min(m - 1);
            if idx < m {
                ui[idx] = 1.0;
            }
            let idx = u_cols.len().min(n - 1);
            if idx < n {
                vi[idx] = 1.0;
            }
            u_cols.push(ui);
            sigmas.push(0.0);
            v_cols.push(vi);
            continue;
        }

        // Deflate: A = A - sigma * u * v^T
        for i in 0..m {
            for j in 0..n {
                a_work[i * n + j] -= sigma * u[i] * v[j];
            }
        }

        u_cols.push(u);
        sigmas.push(sigma);
        v_cols.push(v);
    }

    // Pack results.
    let mut u_mat = vec![0.0_f64; m * k];
    let mut vt_mat = vec![0.0_f64; k * n];
    for t in 0..k {
        for i in 0..m {
            u_mat[i * k + t] = u_cols[t][i];
        }
        for j in 0..n {
            vt_mat[t * n + j] = v_cols[t][j];
        }
    }

    Ok((u_mat, sigmas, vt_mat))
}

/// Solve the Procrustes problem: find orthogonal W that minimises ||X W - Y||_F.
/// Uses SVD of X^T Y via power iteration.
fn procrustes(
    x: &[f64], // n_pairs × dim, row-major
    y: &[f64], // n_pairs × dim, row-major
    n: usize,
    dim: usize,
    rng: &mut u64,
) -> Result<Vec<f64>, MeaError> {
    // M = X^T * Y   (dim × dim)
    let xt = transpose(x, n, dim); // dim × n
    let m = matmul(&xt, y, dim, n, dim); // dim × dim

    // SVD of M: U, S, Vt  (all dim × dim here since k = dim)
    let k = dim.min(n); // can't extract more singular triplets than min(dim, n)
    let (u, _s, vt) = svd_power_iteration(&m, dim, dim, k, 200, 1e-7, rng)?;

    // Pad U and Vt to full dim if k < dim.
    let u_full = if k == dim {
        u
    } else {
        let mut uf = identity(dim);
        for i in 0..dim {
            for t in 0..k {
                uf[i * dim + t] = u[i * k + t];
            }
        }
        uf
    };
    let vt_full = if k == dim {
        vt
    } else {
        let mut vtf = identity(dim);
        for t in 0..k {
            for j in 0..dim {
                vtf[t * dim + j] = vt[t * k + j];
            }
        }
        vtf
    };

    // W = V * U^T = (Vt)^T * U^T
    let v_full = transpose(&vt_full, dim, dim); // dim × dim
    let ut_full = transpose(&u_full, dim, dim); // dim × dim
    let w = matmul(&v_full, &ut_full, dim, dim, dim); // dim × dim
    Ok(w)
}

/// Least-squares linear regression: W = (X^T X)^{-1} X^T Y.
/// Uses a simple Gauss-Jordan inversion for the Gram matrix.
fn linear_regression(x: &[f64], y: &[f64], n: usize, dim: usize) -> Result<Vec<f64>, MeaError> {
    let xt = transpose(x, n, dim); // dim × n
    let xtx = matmul(&xt, x, dim, n, dim); // dim × dim
    let xty = matmul(&xt, y, dim, n, dim); // dim × dim

    // Invert xtx via Gauss-Jordan.
    let inv_xtx = gauss_jordan_invert(&xtx, dim)?;
    let w = matmul(&inv_xtx, &xty, dim, dim, dim);
    Ok(w)
}

/// Gauss-Jordan matrix inversion for a d×d matrix.
fn gauss_jordan_invert(a: &[f64], d: usize) -> Result<Vec<f64>, MeaError> {
    // Augmented matrix [A | I]
    let mut aug = vec![0.0_f64; d * 2 * d];
    for i in 0..d {
        for j in 0..d {
            aug[i * 2 * d + j] = a[i * d + j];
        }
        aug[i * 2 * d + d + i] = 1.0;
    }

    for col in 0..d {
        // Find pivot.
        let mut max_row = col;
        let mut max_val = aug[col * 2 * d + col].abs();
        for row in (col + 1)..d {
            let v = aug[row * 2 * d + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(MeaError::Arithmetic(
                "singular matrix in Gauss-Jordan".to_string(),
            ));
        }
        // Swap rows.
        if max_row != col {
            for j in 0..2 * d {
                aug.swap(col * 2 * d + j, max_row * 2 * d + j);
            }
        }
        let pivot = aug[col * 2 * d + col];
        for j in 0..2 * d {
            aug[col * 2 * d + j] /= pivot;
        }
        for row in 0..d {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * d + col];
            for j in 0..2 * d {
                let v = aug[col * 2 * d + j];
                aug[row * 2 * d + j] -= factor * v;
            }
        }
    }

    let mut inv = vec![0.0_f64; d * d];
    for i in 0..d {
        for j in 0..d {
            inv[i * d + j] = aug[i * 2 * d + d + j];
        }
    }
    Ok(inv)
}

/// Approximate CCA alignment: whiten both sides then Procrustes.
fn cca_alignment(
    x: &[f64],
    y: &[f64],
    n: usize,
    dim: usize,
    rng: &mut u64,
) -> Result<Vec<f64>, MeaError> {
    // Whiten X.
    let wx = whiten(x, n, dim)?;
    // Whiten Y.
    let wy = whiten(y, n, dim)?;
    // Procrustes on whitened pair.
    procrustes(&wx, &wy, n, dim, rng)
}

/// Whiten a matrix X (n × dim) so that X^T X ≈ I.
fn whiten(x: &[f64], n: usize, dim: usize) -> Result<Vec<f64>, MeaError> {
    let xt = transpose(x, n, dim);
    let cov = matmul(&xt, x, dim, n, dim);
    // Compute the symmetric square-root inverse via Gauss-Jordan on the Gram.
    // For approximate whitening we do: W = diag(1/sqrt(diag(cov))).
    let mut w_diag = vec![0.0_f64; dim];
    for i in 0..dim {
        let v = cov[i * dim + i];
        w_diag[i] = if v > 1e-14 { 1.0 / v.sqrt() } else { 1.0 };
    }
    // X_white[row][col] = X[row][col] * w_diag[col]
    let mut xw = vec![0.0_f64; n * dim];
    for i in 0..n {
        for j in 0..dim {
            xw[i * dim + j] = x[i * dim + j] * w_diag[j];
        }
    }
    Ok(xw)
}

// ─── Main struct ──────────────────────────────────────────────────────────────

/// Multilingual embedding aligner: maps embeddings from different language
/// spaces to a shared cross-lingual embedding space.
pub struct MultilingualEmbeddingAligner {
    pub config: MeaAlignerConfig,
    pub language_spaces: HashMap<LangId, LanguageSpace>,
    pub alignment_matrices: HashMap<(LangId, LangId), AlignmentMatrix>,
    pub alignment_history: VecDeque<AlignmentRecord>,
    rng: u64,
}

impl MultilingualEmbeddingAligner {
    const HISTORY_CAP: usize = 500;

    /// Create a new aligner with the given configuration.
    pub fn new(config: MeaAlignerConfig) -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64 + d.as_secs() * 1_000_000_000)
            .unwrap_or(0x123456789abcdef0);
        Self {
            config,
            language_spaces: HashMap::new(),
            alignment_matrices: HashMap::new(),
            alignment_history: VecDeque::with_capacity(Self::HISTORY_CAP),
            rng: if seed == 0 { 0xdeadbeefcafe_u64 } else { seed },
        }
    }

    /// Create an aligner with default config.
    pub fn with_defaults() -> Self {
        Self::new(MeaAlignerConfig::default())
    }

    // ── Language management ───────────────────────────────────────────────────

    /// Register a new language space.
    pub fn add_language(&mut self, id: LangId, name: String, dim: usize) {
        self.language_spaces
            .entry(id)
            .or_insert_with(|| LanguageSpace::new(id, name, dim));
    }

    /// Remove a language space and all alignment matrices involving it.
    pub fn remove_language(&mut self, id: LangId) {
        self.language_spaces.remove(&id);
        self.alignment_matrices
            .retain(|(src, tgt), _| src != &id && tgt != &id);
    }

    // ── Embedding management ──────────────────────────────────────────────────

    /// Add an embedding to a language space.
    pub fn add_embedding(
        &mut self,
        lang_id: LangId,
        emb_id: u64,
        mut vector: Vec<f64>,
    ) -> Result<(), MeaError> {
        let space = self
            .language_spaces
            .get_mut(&lang_id)
            .ok_or_else(|| MeaError::LanguageNotFound(lang_name_from_id(&lang_id)))?;
        if vector.is_empty() {
            return Err(MeaError::EmptyEmbedding);
        }
        if vector.len() != space.dim {
            return Err(MeaError::DimensionMismatch {
                expected: space.dim,
                actual: vector.len(),
            });
        }
        if self.config.normalize_embeddings {
            normalize_vec(&mut vector);
        }
        // Upsert.
        if let Some(entry) = space.embeddings.iter_mut().find(|(id, _)| *id == emb_id) {
            entry.1 = vector;
        } else {
            space.embeddings.push((emb_id, vector));
        }
        // Invalidate centroid.
        space.centroid = None;
        Ok(())
    }

    /// Remove an embedding from a language space.
    pub fn remove_embedding(&mut self, lang_id: LangId, emb_id: u64) -> Result<(), MeaError> {
        let space = self
            .language_spaces
            .get_mut(&lang_id)
            .ok_or_else(|| MeaError::LanguageNotFound(lang_name_from_id(&lang_id)))?;
        let before = space.embeddings.len();
        space.embeddings.retain(|(id, _)| *id != emb_id);
        if space.embeddings.len() == before {
            return Err(MeaError::EmbeddingNotFound(emb_id, space.lang_name()));
        }
        space.centroid = None;
        Ok(())
    }

    // ── Alignment ─────────────────────────────────────────────────────────────

    /// Compute and store an alignment matrix from `src` → `tgt`.
    ///
    /// `anchors` is a list of `(src_emb_id, tgt_emb_id)` pairs that are
    /// known to be translation equivalents.
    pub fn compute_alignment(
        &mut self,
        src: LangId,
        tgt: LangId,
        anchors: &[(u64, u64)],
    ) -> Result<(), MeaError> {
        let min = self.config.min_anchor_pairs;
        if anchors.len() < min {
            return Err(MeaError::NotEnoughAnchors {
                min,
                got: anchors.len(),
            });
        }

        let dim = self.config.dim;

        // Collect anchor vectors.
        let src_vecs: Vec<Vec<f64>> = {
            let space = self
                .language_spaces
                .get(&src)
                .ok_or_else(|| MeaError::LanguageNotFound(lang_name_from_id(&src)))?;
            anchors
                .iter()
                .map(|(sid, _)| {
                    space
                        .embeddings
                        .iter()
                        .find(|(id, _)| id == sid)
                        .map(|(_, v)| v.clone())
                        .ok_or_else(|| MeaError::EmbeddingNotFound(*sid, space.lang_name()))
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        let tgt_vecs: Vec<Vec<f64>> = {
            let space = self
                .language_spaces
                .get(&tgt)
                .ok_or_else(|| MeaError::LanguageNotFound(lang_name_from_id(&tgt)))?;
            anchors
                .iter()
                .map(|(_, tid)| {
                    space
                        .embeddings
                        .iter()
                        .find(|(id, _)| id == tid)
                        .map(|(_, v)| v.clone())
                        .ok_or_else(|| MeaError::EmbeddingNotFound(*tid, space.lang_name()))
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        // Validate dimensions.
        for v in src_vecs.iter().chain(tgt_vecs.iter()) {
            if v.len() != dim {
                return Err(MeaError::DimensionMismatch {
                    expected: dim,
                    actual: v.len(),
                });
            }
        }

        let n = anchors.len();

        // Flatten to row-major matrices.
        let x: Vec<f64> = src_vecs.iter().flat_map(|v| v.iter().copied()).collect();
        let y: Vec<f64> = tgt_vecs.iter().flat_map(|v| v.iter().copied()).collect();

        let matrix = match self.config.alignment_method {
            MeaAlignmentMethod::IdentityPassthrough => identity(dim),
            MeaAlignmentMethod::Procrustes => procrustes(&x, &y, n, dim, &mut self.rng)?,
            MeaAlignmentMethod::LinearRegression => linear_regression(&x, &y, n, dim)?,
            MeaAlignmentMethod::Cca => cca_alignment(&x, &y, n, dim, &mut self.rng)?,
        };

        // Measure alignment quality: mean cosine(W*x_i, y_i).
        let quality = self.measure_quality(&matrix, &src_vecs, &tgt_vecs);

        let record = AlignmentRecord {
            ts: now_secs(),
            src_lang: src,
            tgt_lang: tgt,
            n_anchors: n,
            quality,
        };

        self.alignment_matrices.insert(
            (src, tgt),
            AlignmentMatrix {
                src_lang: src,
                tgt_lang: tgt,
                matrix,
                quality,
            },
        );

        if self.alignment_history.len() >= Self::HISTORY_CAP {
            self.alignment_history.pop_front();
        }
        self.alignment_history.push_back(record);

        Ok(())
    }

    fn measure_quality(&self, matrix: &[f64], src_vecs: &[Vec<f64>], tgt_vecs: &[Vec<f64>]) -> f64 {
        let n = src_vecs.len();
        if n == 0 {
            return 0.0;
        }
        let am = AlignmentMatrix {
            src_lang: [0; 8],
            tgt_lang: [0; 8],
            matrix: matrix.to_vec(),
            quality: 0.0,
        };
        let total: f64 = src_vecs
            .iter()
            .zip(tgt_vecs.iter())
            .map(|(sv, tv)| {
                am.apply(sv)
                    .map(|aligned| cosine_similarity(&aligned, tv))
                    .unwrap_or(0.0)
            })
            .sum();
        total / n as f64
    }

    /// Apply the stored alignment matrix to transform `vector` from `src` space to `tgt` space.
    pub fn align_embedding(
        &self,
        src: LangId,
        tgt: LangId,
        vector: &[f64],
    ) -> Result<Vec<f64>, MeaError> {
        if vector.is_empty() {
            return Err(MeaError::EmptyEmbedding);
        }
        if vector.len() != self.config.dim {
            return Err(MeaError::DimensionMismatch {
                expected: self.config.dim,
                actual: vector.len(),
            });
        }
        let am = self.alignment_matrices.get(&(src, tgt)).ok_or_else(|| {
            MeaError::AlignmentNotFound(lang_name_from_id(&src), lang_name_from_id(&tgt))
        })?;
        let mut out = am.apply(vector)?;
        if self.config.normalize_embeddings {
            normalize_vec(&mut out);
        }
        Ok(out)
    }

    // ── Cross-lingual search ──────────────────────────────────────────────────

    /// Search in `target_lang`'s embedding space using a query from `query_lang`.
    ///
    /// Returns top-k `(emb_id, cosine_score)` pairs, sorted by descending score.
    pub fn cross_lingual_search(
        &self,
        query_lang: LangId,
        query: &[f64],
        target_lang: LangId,
        top_k: usize,
    ) -> Result<Vec<(u64, f64)>, MeaError> {
        if query.is_empty() {
            return Err(MeaError::EmptyEmbedding);
        }

        // Validate that the target language exists before attempting alignment.
        if !self.language_spaces.contains_key(&target_lang) {
            return Err(MeaError::LanguageNotFound(lang_name_from_id(&target_lang)));
        }

        let aligned_query = if query_lang == target_lang {
            query.to_vec()
        } else {
            self.align_embedding(query_lang, target_lang, query)?
        };

        let tgt_space = self
            .language_spaces
            .get(&target_lang)
            .ok_or_else(|| MeaError::LanguageNotFound(lang_name_from_id(&target_lang)))?;

        let mut scores: Vec<(u64, f64)> = tgt_space
            .embeddings
            .iter()
            .map(|(id, v)| (*id, cosine_similarity(&aligned_query, v)))
            .collect();

        scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        Ok(scores)
    }

    // ── Quality & stats ───────────────────────────────────────────────────────

    /// Return the stored alignment quality for a (src, tgt) pair.
    pub fn alignment_quality(&self, src: LangId, tgt: LangId) -> Option<f64> {
        self.alignment_matrices
            .get(&(src, tgt))
            .map(|am| am.quality)
    }

    /// Compute and cache the centroid of a language space.
    pub fn compute_centroid(&mut self, lang_id: LangId) -> Result<Vec<f64>, MeaError> {
        let space = self
            .language_spaces
            .get_mut(&lang_id)
            .ok_or_else(|| MeaError::LanguageNotFound(lang_name_from_id(&lang_id)))?;

        if space.embeddings.is_empty() {
            return Err(MeaError::Arithmetic(
                "no embeddings to compute centroid from".to_string(),
            ));
        }

        let dim = space.dim;
        let mut c = vec![0.0_f64; dim];
        for (_, v) in &space.embeddings {
            for (ci, vi) in c.iter_mut().zip(v.iter()) {
                *ci += vi;
            }
        }
        let n = space.embeddings.len() as f64;
        for ci in c.iter_mut() {
            *ci /= n;
        }
        space.centroid = Some(c.clone());
        Ok(c)
    }

    /// Aggregate statistics about the aligner state.
    pub fn aligner_stats(&self) -> MeaAlignerStats {
        let n_alignments = self.alignment_matrices.len();
        let avg_quality = if n_alignments == 0 {
            0.0
        } else {
            let sum: f64 = self.alignment_matrices.values().map(|am| am.quality).sum();
            sum / n_alignments as f64
        };
        MeaAlignerStats {
            n_languages: self.language_spaces.len(),
            n_alignments,
            n_history_records: self.alignment_history.len(),
            total_embeddings: self
                .language_spaces
                .values()
                .map(|s| s.embeddings.len())
                .sum(),
            avg_alignment_quality: avg_quality,
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Number of embeddings in a language space.
    pub fn embedding_count(&self, lang_id: LangId) -> Option<usize> {
        self.language_spaces
            .get(&lang_id)
            .map(|s| s.embeddings.len())
    }

    /// Retrieve a stored alignment matrix (immutable borrow).
    pub fn get_alignment_matrix(&self, src: LangId, tgt: LangId) -> Option<&AlignmentMatrix> {
        self.alignment_matrices.get(&(src, tgt))
    }

    /// List all registered language IDs.
    pub fn language_ids(&self) -> Vec<LangId> {
        self.language_spaces.keys().copied().collect()
    }

    /// Check whether an alignment exists for the given pair.
    pub fn has_alignment(&self, src: LangId, tgt: LangId) -> bool {
        self.alignment_matrices.contains_key(&(src, tgt))
    }

    /// Retrieve the centroid of a language space without recomputing.
    pub fn centroid(&self, lang_id: LangId) -> Option<&Vec<f64>> {
        self.language_spaces.get(&lang_id)?.centroid.as_ref()
    }

    /// Recent alignment history entries (newest first).
    pub fn history(&self, limit: usize) -> Vec<&AlignmentRecord> {
        self.alignment_history.iter().rev().take(limit).collect()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_lang_id(s: &str) -> LangId {
        let mut id = [0u8; 8];
        for (i, b) in s.as_bytes().iter().take(8).enumerate() {
            id[i] = *b;
        }
        id
    }

    fn make_aligner(dim: usize, method: MeaAlignmentMethod) -> MultilingualEmbeddingAligner {
        MultilingualEmbeddingAligner::new(MeaAlignerConfig {
            dim,
            normalize_embeddings: false,
            alignment_method: method,
            min_anchor_pairs: 3,
        })
    }

    fn add_lang(aligner: &mut MultilingualEmbeddingAligner, name: &str, dim: usize) -> LangId {
        let id = make_lang_id(name);
        aligner.add_language(id, name.to_string(), dim);
        id
    }

    fn add_emb(aligner: &mut MultilingualEmbeddingAligner, lang: LangId, emb_id: u64, v: Vec<f64>) {
        aligner
            .add_embedding(lang, emb_id, v)
            .expect("add_embedding failed");
    }

    // ── 1: basic construction ─────────────────────────────────────────────────

    #[test]
    fn test_new_aligner_default() {
        let a = MultilingualEmbeddingAligner::with_defaults();
        assert!(a.language_spaces.is_empty());
        assert!(a.alignment_matrices.is_empty());
        assert!(a.alignment_history.is_empty());
    }

    #[test]
    fn test_new_aligner_custom_config() {
        let cfg = MeaAlignerConfig {
            dim: 32,
            normalize_embeddings: true,
            alignment_method: MeaAlignmentMethod::LinearRegression,
            min_anchor_pairs: 10,
        };
        let a = MultilingualEmbeddingAligner::new(cfg.clone());
        assert_eq!(a.config.dim, 32);
        assert_eq!(a.config.min_anchor_pairs, 10);
        assert!(a.config.normalize_embeddings);
    }

    // ── 2: language management ────────────────────────────────────────────────

    #[test]
    fn test_add_language() {
        let mut a = make_aligner(4, MeaAlignmentMethod::IdentityPassthrough);
        let en = make_lang_id("en");
        a.add_language(en, "English".to_string(), 4);
        assert_eq!(a.language_spaces.len(), 1);
        assert_eq!(a.language_spaces[&en].name, "English");
    }

    #[test]
    fn test_add_language_idempotent() {
        let mut a = make_aligner(4, MeaAlignmentMethod::IdentityPassthrough);
        let en = make_lang_id("en");
        a.add_language(en, "English".to_string(), 4);
        a.add_language(en, "English2".to_string(), 4);
        // Second add should be a no-op (entry already exists).
        assert_eq!(a.language_spaces[&en].name, "English");
    }

    #[test]
    fn test_remove_language() {
        let mut a = make_aligner(4, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", 4);
        a.remove_language(en);
        assert!(a.language_spaces.is_empty());
    }

    #[test]
    fn test_remove_language_cleans_alignments() {
        let mut a = make_aligner(4, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", 4);
        let fr = add_lang(&mut a, "fr", 4);
        // Manually insert a dummy alignment.
        a.alignment_matrices.insert(
            (en, fr),
            AlignmentMatrix {
                src_lang: en,
                tgt_lang: fr,
                matrix: identity(4),
                quality: 1.0,
            },
        );
        a.remove_language(en);
        assert!(!a.alignment_matrices.contains_key(&(en, fr)));
    }

    // ── 3: embedding management ───────────────────────────────────────────────

    #[test]
    fn test_add_embedding_ok() {
        let mut a = make_aligner(4, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", 4);
        add_emb(&mut a, en, 1, vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(a.embedding_count(en), Some(1));
    }

    #[test]
    fn test_add_embedding_dim_mismatch() {
        let mut a = make_aligner(4, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", 4);
        let res = a.add_embedding(en, 1, vec![1.0, 0.0]);
        assert!(matches!(res, Err(MeaError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_add_embedding_unknown_lang() {
        let mut a = make_aligner(4, MeaAlignmentMethod::IdentityPassthrough);
        let unknown = make_lang_id("xx");
        let res = a.add_embedding(unknown, 1, vec![1.0, 0.0, 0.0, 0.0]);
        assert!(matches!(res, Err(MeaError::LanguageNotFound(_))));
    }

    #[test]
    fn test_add_embedding_empty_vector() {
        let mut a = make_aligner(4, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", 4);
        let res = a.add_embedding(en, 1, vec![]);
        assert!(matches!(res, Err(MeaError::EmptyEmbedding)));
    }

    #[test]
    fn test_add_embedding_upsert() {
        let mut a = make_aligner(4, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", 4);
        add_emb(&mut a, en, 1, vec![1.0, 0.0, 0.0, 0.0]);
        add_emb(&mut a, en, 1, vec![0.0, 1.0, 0.0, 0.0]);
        assert_eq!(a.embedding_count(en), Some(1));
        let v = &a.language_spaces[&en].embeddings[0].1;
        assert!((v[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_remove_embedding_ok() {
        let mut a = make_aligner(4, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", 4);
        add_emb(&mut a, en, 1, vec![1.0, 0.0, 0.0, 0.0]);
        a.remove_embedding(en, 1).expect("remove failed");
        assert_eq!(a.embedding_count(en), Some(0));
    }

    #[test]
    fn test_remove_embedding_not_found() {
        let mut a = make_aligner(4, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", 4);
        let res = a.remove_embedding(en, 99);
        assert!(matches!(res, Err(MeaError::EmbeddingNotFound(99, _))));
    }

    #[test]
    fn test_remove_embedding_unknown_lang() {
        let mut a = make_aligner(4, MeaAlignmentMethod::IdentityPassthrough);
        let unknown = make_lang_id("xx");
        let res = a.remove_embedding(unknown, 1);
        assert!(matches!(res, Err(MeaError::LanguageNotFound(_))));
    }

    // ── 4: identity alignment ─────────────────────────────────────────────────

    #[test]
    fn test_compute_alignment_identity() {
        let dim = 4;
        let mut a = make_aligner(dim, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", dim);
        let fr = add_lang(&mut a, "fr", dim);
        for i in 0..5u64 {
            let v: Vec<f64> = (0..dim)
                .map(|j| ((i * dim as u64 + j as u64) % 7) as f64)
                .collect();
            add_emb(&mut a, en, i, v.clone());
            add_emb(&mut a, fr, i, v);
        }
        let anchors: Vec<(u64, u64)> = (0..5).map(|i| (i, i)).collect();
        a.compute_alignment(en, fr, &anchors)
            .expect("alignment failed");
        assert!(a.has_alignment(en, fr));
    }

    #[test]
    fn test_compute_alignment_too_few_anchors() {
        let mut a = make_aligner(4, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", 4);
        let fr = add_lang(&mut a, "fr", 4);
        let anchors: Vec<(u64, u64)> = vec![(0, 0), (1, 1)]; // only 2 < min 3
        let res = a.compute_alignment(en, fr, &anchors);
        assert!(matches!(res, Err(MeaError::NotEnoughAnchors { .. })));
    }

    #[test]
    fn test_compute_alignment_unknown_src() {
        let mut a = make_aligner(4, MeaAlignmentMethod::IdentityPassthrough);
        let xx = make_lang_id("xx");
        let fr = add_lang(&mut a, "fr", 4);
        let anchors: Vec<(u64, u64)> = (0..5).map(|i| (i, i)).collect();
        let res = a.compute_alignment(xx, fr, &anchors);
        assert!(matches!(res, Err(MeaError::LanguageNotFound(_))));
    }

    #[test]
    fn test_compute_alignment_unknown_tgt() {
        let mut a = make_aligner(4, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", 4);
        let xx = make_lang_id("xx");
        for i in 0..5u64 {
            add_emb(&mut a, en, i, vec![i as f64, 0.0, 0.0, 0.0]);
        }
        let anchors: Vec<(u64, u64)> = (0..5).map(|i| (i, i)).collect();
        let res = a.compute_alignment(en, xx, &anchors);
        assert!(matches!(res, Err(MeaError::LanguageNotFound(_))));
    }

    #[test]
    fn test_alignment_quality_stored() {
        let dim = 4;
        let mut a = make_aligner(dim, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", dim);
        let fr = add_lang(&mut a, "fr", dim);
        for i in 0..5u64 {
            let v = vec![i as f64 + 1.0, 0.0, 0.0, 0.0];
            add_emb(&mut a, en, i, v.clone());
            add_emb(&mut a, fr, i, v);
        }
        let anchors: Vec<(u64, u64)> = (0..5).map(|i| (i, i)).collect();
        a.compute_alignment(en, fr, &anchors)
            .expect("test: compute_alignment failed");
        // Identity + identical vectors → quality ≈ 1.0
        let q = a
            .alignment_quality(en, fr)
            .expect("test: alignment_quality not found");
        assert!(q > 0.99, "quality={q}");
    }

    #[test]
    fn test_alignment_quality_none_for_missing() {
        let a = make_aligner(4, MeaAlignmentMethod::IdentityPassthrough);
        let en = make_lang_id("en");
        let fr = make_lang_id("fr");
        assert!(a.alignment_quality(en, fr).is_none());
    }

    // ── 5: align_embedding ────────────────────────────────────────────────────

    #[test]
    fn test_align_embedding_identity() {
        let dim = 4;
        let mut a = make_aligner(dim, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", dim);
        let fr = add_lang(&mut a, "fr", dim);
        for i in 0..5u64 {
            let v = vec![i as f64 + 1.0, 0.0, 0.0, 0.0];
            add_emb(&mut a, en, i, v.clone());
            add_emb(&mut a, fr, i, v);
        }
        let anchors: Vec<(u64, u64)> = (0..5).map(|i| (i, i)).collect();
        a.compute_alignment(en, fr, &anchors)
            .expect("test: compute_alignment identity passthrough failed");
        let q = vec![2.0, 0.0, 0.0, 0.0];
        let aligned = a
            .align_embedding(en, fr, &q)
            .expect("test: align_embedding identity passthrough failed");
        // Identity: output ≈ input
        assert!((aligned[0] - 2.0).abs() < 1e-6, "aligned={aligned:?}");
    }

    #[test]
    fn test_align_embedding_no_matrix() {
        let a = make_aligner(4, MeaAlignmentMethod::IdentityPassthrough);
        let en = make_lang_id("en");
        let fr = make_lang_id("fr");
        let res = a.align_embedding(en, fr, &[1.0, 0.0, 0.0, 0.0]);
        assert!(matches!(res, Err(MeaError::AlignmentNotFound(_, _))));
    }

    #[test]
    fn test_align_embedding_dim_mismatch() {
        let dim = 4;
        let mut a = make_aligner(dim, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", dim);
        let fr = add_lang(&mut a, "fr", dim);
        for i in 0..5u64 {
            let v = vec![i as f64 + 1.0, 0.0, 0.0, 0.0];
            add_emb(&mut a, en, i, v.clone());
            add_emb(&mut a, fr, i, v);
        }
        let anchors: Vec<(u64, u64)> = (0..5).map(|i| (i, i)).collect();
        a.compute_alignment(en, fr, &anchors)
            .expect("test: compute_alignment for dim mismatch test failed");
        let res = a.align_embedding(en, fr, &[1.0, 0.0]); // wrong dim
        assert!(matches!(res, Err(MeaError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_align_embedding_empty() {
        let dim = 4;
        let mut a = make_aligner(dim, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", dim);
        let fr = add_lang(&mut a, "fr", dim);
        for i in 0..5u64 {
            add_emb(&mut a, en, i, vec![i as f64 + 1.0, 0.0, 0.0, 0.0]);
            add_emb(&mut a, fr, i, vec![i as f64 + 1.0, 0.0, 0.0, 0.0]);
        }
        let anchors: Vec<(u64, u64)> = (0..5).map(|i| (i, i)).collect();
        a.compute_alignment(en, fr, &anchors)
            .expect("test: compute_alignment for empty align test failed");
        let res = a.align_embedding(en, fr, &[]);
        assert!(matches!(res, Err(MeaError::EmptyEmbedding)));
    }

    // ── 6: cross-lingual search ───────────────────────────────────────────────

    #[test]
    fn test_cross_lingual_search_same_lang() {
        let dim = 4;
        let mut a = make_aligner(dim, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", dim);
        add_emb(&mut a, en, 1, vec![1.0, 0.0, 0.0, 0.0]);
        add_emb(&mut a, en, 2, vec![0.0, 1.0, 0.0, 0.0]);
        let res = a
            .cross_lingual_search(en, &[1.0, 0.0, 0.0, 0.0], en, 2)
            .expect("test: cross_lingual_search same lang failed");
        assert_eq!(res[0].0, 1);
    }

    #[test]
    fn test_cross_lingual_search_top_k() {
        let dim = 4;
        let mut a = make_aligner(dim, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", dim);
        for i in 1..=10u64 {
            add_emb(&mut a, en, i, vec![i as f64, 0.0, 0.0, 0.0]);
        }
        let res = a
            .cross_lingual_search(en, &[10.0, 0.0, 0.0, 0.0], en, 3)
            .expect("test: cross_lingual_search top k failed");
        assert_eq!(res.len(), 3);
    }

    #[test]
    fn test_cross_lingual_search_sorted_by_score() {
        let dim = 4;
        let mut a = make_aligner(dim, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", dim);
        add_emb(&mut a, en, 1, vec![1.0, 0.0, 0.0, 0.0]);
        add_emb(&mut a, en, 2, vec![0.9, 0.1, 0.0, 0.0]);
        add_emb(&mut a, en, 3, vec![0.0, 0.0, 1.0, 0.0]);
        let res = a
            .cross_lingual_search(en, &[1.0, 0.0, 0.0, 0.0], en, 3)
            .expect("test: cross_lingual_search sorted failed");
        // Scores must be non-increasing.
        for w in res.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn test_cross_lingual_search_cross_lang() {
        let dim = 4;
        let mut a = make_aligner(dim, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", dim);
        let fr = add_lang(&mut a, "fr", dim);
        for i in 0..5u64 {
            let v = vec![i as f64 + 1.0, 0.0, 0.0, 0.0];
            add_emb(&mut a, en, i, v.clone());
            add_emb(&mut a, fr, i, v);
        }
        let anchors: Vec<(u64, u64)> = (0..5).map(|i| (i, i)).collect();
        a.compute_alignment(en, fr, &anchors)
            .expect("test: compute_alignment cross lang search failed");
        let res = a
            .cross_lingual_search(en, &[5.0, 0.0, 0.0, 0.0], fr, 1)
            .expect("test: cross_lingual_search cross lang failed");
        assert_eq!(res.len(), 1);
    }

    #[test]
    fn test_cross_lingual_search_empty_query() {
        let dim = 4;
        let mut a = make_aligner(dim, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", dim);
        let res = a.cross_lingual_search(en, &[], en, 1);
        assert!(matches!(res, Err(MeaError::EmptyEmbedding)));
    }

    #[test]
    fn test_cross_lingual_search_unknown_target() {
        let dim = 4;
        let mut a = make_aligner(dim, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", dim);
        let xx = make_lang_id("xx");
        add_emb(&mut a, en, 1, vec![1.0, 0.0, 0.0, 0.0]);
        // Same lang → skip alignment, so unknown tgt triggers LanguageNotFound.
        let res = a.cross_lingual_search(en, &[1.0, 0.0, 0.0, 0.0], xx, 1);
        assert!(matches!(res, Err(MeaError::LanguageNotFound(_))));
    }

    // ── 7: centroid ───────────────────────────────────────────────────────────

    #[test]
    fn test_compute_centroid_basic() {
        let dim = 4;
        let mut a = make_aligner(dim, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", dim);
        add_emb(&mut a, en, 1, vec![2.0, 0.0, 0.0, 0.0]);
        add_emb(&mut a, en, 2, vec![4.0, 0.0, 0.0, 0.0]);
        let c = a
            .compute_centroid(en)
            .expect("test: compute_centroid basic failed");
        assert!((c[0] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_centroid_cached() {
        let dim = 4;
        let mut a = make_aligner(dim, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", dim);
        add_emb(&mut a, en, 1, vec![2.0, 0.0, 0.0, 0.0]);
        a.compute_centroid(en)
            .expect("test: compute_centroid cached failed");
        assert!(a.centroid(en).is_some());
    }

    #[test]
    fn test_compute_centroid_empty() {
        let dim = 4;
        let mut a = make_aligner(dim, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", dim);
        let res = a.compute_centroid(en);
        assert!(matches!(res, Err(MeaError::Arithmetic(_))));
    }

    #[test]
    fn test_compute_centroid_unknown_lang() {
        let mut a = make_aligner(4, MeaAlignmentMethod::IdentityPassthrough);
        let xx = make_lang_id("xx");
        let res = a.compute_centroid(xx);
        assert!(matches!(res, Err(MeaError::LanguageNotFound(_))));
    }

    #[test]
    fn test_centroid_invalidated_after_add() {
        let dim = 4;
        let mut a = make_aligner(dim, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", dim);
        add_emb(&mut a, en, 1, vec![2.0, 0.0, 0.0, 0.0]);
        a.compute_centroid(en)
            .expect("test: compute_centroid for invalidation test failed");
        // Adding a new embedding invalidates the centroid.
        add_emb(&mut a, en, 2, vec![4.0, 0.0, 0.0, 0.0]);
        assert!(a.centroid(en).is_none());
    }

    // ── 8: statistics ─────────────────────────────────────────────────────────

    #[test]
    fn test_aligner_stats_empty() {
        let a = make_aligner(4, MeaAlignmentMethod::IdentityPassthrough);
        let stats = a.aligner_stats();
        assert_eq!(stats.n_languages, 0);
        assert_eq!(stats.n_alignments, 0);
        assert_eq!(stats.total_embeddings, 0);
    }

    #[test]
    fn test_aligner_stats_after_ops() {
        let dim = 4;
        let mut a = make_aligner(dim, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", dim);
        let fr = add_lang(&mut a, "fr", dim);
        for i in 0..5u64 {
            let v = vec![i as f64 + 1.0, 0.0, 0.0, 0.0];
            add_emb(&mut a, en, i, v.clone());
            add_emb(&mut a, fr, i, v);
        }
        let anchors: Vec<(u64, u64)> = (0..5).map(|i| (i, i)).collect();
        a.compute_alignment(en, fr, &anchors)
            .expect("test: compute_alignment for stats test failed");
        let stats = a.aligner_stats();
        assert_eq!(stats.n_languages, 2);
        assert_eq!(stats.n_alignments, 1);
        assert_eq!(stats.total_embeddings, 10);
        assert_eq!(stats.n_history_records, 1);
    }

    #[test]
    fn test_history_bounded() {
        let dim = 2;
        let mut a = MultilingualEmbeddingAligner::new(MeaAlignerConfig {
            dim,
            normalize_embeddings: false,
            alignment_method: MeaAlignmentMethod::IdentityPassthrough,
            min_anchor_pairs: 1,
        });
        let en = add_lang(&mut a, "en", dim);
        let fr = add_lang(&mut a, "fr", dim);
        for i in 0..510u64 {
            // Overwrite so IDs 0..5 remain valid.
            let _ = a.add_embedding(en, i % 5, vec![1.0, 0.0]);
            let _ = a.add_embedding(fr, i % 5, vec![1.0, 0.0]);
            let anchors = vec![(i % 5, i % 5)];
            let _ = a.compute_alignment(en, fr, &anchors);
        }
        assert!(a.alignment_history.len() <= MultilingualEmbeddingAligner::HISTORY_CAP);
    }

    // ── 9: Procrustes alignment ───────────────────────────────────────────────

    #[test]
    fn test_procrustes_identity_mapping() {
        let dim = 4;
        let mut a = make_aligner(dim, MeaAlignmentMethod::Procrustes);
        let en = add_lang(&mut a, "en", dim);
        let fr = add_lang(&mut a, "fr", dim);
        // Same embeddings → should learn near-identity
        let vecs: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0, 0.0],
        ];
        for (i, v) in vecs.iter().enumerate() {
            add_emb(&mut a, en, i as u64, v.clone());
            add_emb(&mut a, fr, i as u64, v.clone());
        }
        let anchors: Vec<(u64, u64)> = (0..5).map(|i| (i, i)).collect();
        a.compute_alignment(en, fr, &anchors)
            .expect("test: compute_alignment procrustes failed");
        let q = a
            .alignment_quality(en, fr)
            .expect("test: alignment_quality procrustes not found");
        assert!(q > 0.9, "Procrustes quality on identity={q}");
    }

    #[test]
    fn test_procrustes_rotation() {
        let dim = 2;
        let mut a = make_aligner(dim, MeaAlignmentMethod::Procrustes);
        let en = add_lang(&mut a, "en", dim);
        let fr = add_lang(&mut a, "fr", dim);
        // 90° rotation: (x,y) → (-y, x)
        let src_vecs: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 0.0],
            vec![0.0, 2.0],
        ];
        let tgt_vecs: Vec<Vec<f64>> = src_vecs.iter().map(|v| vec![-v[1], v[0]]).collect();
        for (i, (sv, tv)) in src_vecs.iter().zip(tgt_vecs.iter()).enumerate() {
            add_emb(&mut a, en, i as u64, sv.clone());
            add_emb(&mut a, fr, i as u64, tv.clone());
        }
        let anchors: Vec<(u64, u64)> = (0..5).map(|i| (i, i)).collect();
        a.compute_alignment(en, fr, &anchors)
            .expect("test: compute_alignment procrustes rotation failed");
        let q = a
            .alignment_quality(en, fr)
            .expect("test: alignment_quality procrustes rotation not found");
        assert!(q > 0.9, "Procrustes rotation quality={q}");
    }

    // ── 10: linear regression ─────────────────────────────────────────────────

    #[test]
    fn test_linear_regression_identity() {
        let dim = 4;
        let mut a = make_aligner(dim, MeaAlignmentMethod::LinearRegression);
        let en = add_lang(&mut a, "en", dim);
        let fr = add_lang(&mut a, "fr", dim);
        let vecs: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.0, 0.0],
        ];
        for (i, v) in vecs.iter().enumerate() {
            add_emb(&mut a, en, i as u64, v.clone());
            add_emb(&mut a, fr, i as u64, v.clone());
        }
        let anchors: Vec<(u64, u64)> = (0..5).map(|i| (i, i)).collect();
        a.compute_alignment(en, fr, &anchors)
            .expect("test: compute_alignment linreg identity failed");
        let q = a
            .alignment_quality(en, fr)
            .expect("test: alignment_quality linreg identity not found");
        assert!(q > 0.9, "LinReg quality={q}");
    }

    #[test]
    fn test_linear_regression_scaling() {
        let dim = 2;
        let mut a = make_aligner(dim, MeaAlignmentMethod::LinearRegression);
        let en = add_lang(&mut a, "en", dim);
        let fr = add_lang(&mut a, "fr", dim);
        let src: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 0.0],
            vec![0.0, 2.0],
        ];
        // Scale x by 2, y by 3
        let tgt: Vec<Vec<f64>> = src.iter().map(|v| vec![v[0] * 2.0, v[1] * 3.0]).collect();
        for (i, (sv, tv)) in src.iter().zip(tgt.iter()).enumerate() {
            add_emb(&mut a, en, i as u64, sv.clone());
            add_emb(&mut a, fr, i as u64, tv.clone());
        }
        let anchors: Vec<(u64, u64)> = (0..5).map(|i| (i, i)).collect();
        a.compute_alignment(en, fr, &anchors)
            .expect("test: compute_alignment linreg scaling failed");
        // Test that alignment maps [1,0] → [2,0] approximately
        let out = a
            .align_embedding(en, fr, &[1.0, 0.0])
            .expect("test: align_embedding linreg scaling failed");
        assert!((out[0] - 2.0).abs() < 0.1, "out={out:?}");
    }

    // ── 11: CCA ───────────────────────────────────────────────────────────────

    #[test]
    fn test_cca_alignment() {
        let dim = 4;
        let mut a = make_aligner(dim, MeaAlignmentMethod::Cca);
        let en = add_lang(&mut a, "en", dim);
        let fr = add_lang(&mut a, "fr", dim);
        let vecs: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.5, 0.5],
        ];
        for (i, v) in vecs.iter().enumerate() {
            add_emb(&mut a, en, i as u64, v.clone());
            add_emb(&mut a, fr, i as u64, v.clone());
        }
        let anchors: Vec<(u64, u64)> = (0..5).map(|i| (i, i)).collect();
        a.compute_alignment(en, fr, &anchors)
            .expect("test: compute_alignment CCA failed");
        assert!(a.has_alignment(en, fr));
    }

    // ── 12: misc helpers ──────────────────────────────────────────────────────

    #[test]
    fn test_language_ids() {
        let mut a = make_aligner(4, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", 4);
        let fr = add_lang(&mut a, "fr", 4);
        let ids = a.language_ids();
        assert!(ids.contains(&en));
        assert!(ids.contains(&fr));
    }

    #[test]
    fn test_has_alignment_false() {
        let a = make_aligner(4, MeaAlignmentMethod::IdentityPassthrough);
        assert!(!a.has_alignment(make_lang_id("en"), make_lang_id("fr")));
    }

    #[test]
    fn test_get_alignment_matrix() {
        let dim = 4;
        let mut a = make_aligner(dim, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", dim);
        let fr = add_lang(&mut a, "fr", dim);
        for i in 0..5u64 {
            let v = vec![i as f64 + 1.0, 0.0, 0.0, 0.0];
            add_emb(&mut a, en, i, v.clone());
            add_emb(&mut a, fr, i, v);
        }
        let anchors: Vec<(u64, u64)> = (0..5).map(|i| (i, i)).collect();
        a.compute_alignment(en, fr, &anchors)
            .expect("test: compute_alignment for get_alignment_matrix test failed");
        assert!(a.get_alignment_matrix(en, fr).is_some());
        assert!(a.get_alignment_matrix(fr, en).is_none());
    }

    #[test]
    fn test_history_recent_first() {
        let dim = 2;
        let mut a = MultilingualEmbeddingAligner::new(MeaAlignerConfig {
            dim,
            normalize_embeddings: false,
            alignment_method: MeaAlignmentMethod::IdentityPassthrough,
            min_anchor_pairs: 1,
        });
        let en = add_lang(&mut a, "en", dim);
        let fr = add_lang(&mut a, "fr", dim);
        let de = add_lang(&mut a, "de", dim);
        for i in 0..3u64 {
            let _ = a.add_embedding(en, i, vec![1.0, 0.0]);
            let _ = a.add_embedding(fr, i, vec![1.0, 0.0]);
            let _ = a.add_embedding(de, i, vec![1.0, 0.0]);
        }
        a.compute_alignment(en, fr, &[(0, 0)])
            .expect("test: compute_alignment en->fr history failed");
        a.compute_alignment(en, de, &[(0, 0)])
            .expect("test: compute_alignment en->de history failed");
        let hist = a.history(2);
        // Most recent = de alignment
        assert_eq!(hist[0].tgt_lang, de);
    }

    // ── 13: cosine helpers ────────────────────────────────────────────────────

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-9);
    }

    // ── 14: xorshift ─────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift_not_zero() {
        let mut state = 0xdeadbeef_u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }

    #[test]
    fn test_xorshift_deterministic() {
        let mut s1 = 42u64;
        let mut s2 = 42u64;
        assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
    }

    #[test]
    fn test_xorshift_state_changes() {
        let mut state = 1u64;
        let v1 = xorshift64(&mut state);
        let v2 = xorshift64(&mut state);
        assert_ne!(v1, v2);
    }

    // ── 15: normalize ─────────────────────────────────────────────────────────

    #[test]
    fn test_normalize_vec_unit() {
        let mut v = vec![3.0, 4.0];
        normalize_vec(&mut v);
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_normalize_vec_zero_returns_false() {
        let mut v = vec![0.0, 0.0];
        assert!(!normalize_vec(&mut v));
    }

    // ── 16: normalization flag ────────────────────────────────────────────────

    #[test]
    fn test_normalize_flag_normalizes_on_insert() {
        let dim = 2;
        let mut a = MultilingualEmbeddingAligner::new(MeaAlignerConfig {
            dim,
            normalize_embeddings: true,
            alignment_method: MeaAlignmentMethod::IdentityPassthrough,
            min_anchor_pairs: 1,
        });
        let en = add_lang(&mut a, "en", dim);
        a.add_embedding(en, 1, vec![3.0, 4.0])
            .expect("test: add_embedding normalize flag failed");
        let v = &a.language_spaces[&en].embeddings[0].1;
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
    }

    // ── 17: matrix math ───────────────────────────────────────────────────────

    #[test]
    fn test_matmul_identity() {
        let i = identity(3);
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let out = matmul(&i, &v, 3, 3, 3);
        for (a, b) in out.iter().zip(v.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_transpose_correctness() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        let t = transpose(&a, 2, 3); // should be 3×2
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_gauss_jordan_2x2() {
        // [[2, 0], [0, 4]] → [[0.5, 0], [0, 0.25]]
        let a = vec![2.0, 0.0, 0.0, 4.0];
        let inv = gauss_jordan_invert(&a, 2).expect("test: gauss_jordan_invert 2x2 failed");
        assert!((inv[0] - 0.5).abs() < 1e-9);
        assert!((inv[3] - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_gauss_jordan_singular() {
        let a = vec![1.0, 2.0, 2.0, 4.0]; // singular
        let res = gauss_jordan_invert(&a, 2);
        assert!(matches!(res, Err(MeaError::Arithmetic(_))));
    }

    // ── 18: alignment_matrix apply ───────────────────────────────────────────

    #[test]
    fn test_alignment_matrix_apply_identity() {
        let dim = 3;
        let am = AlignmentMatrix {
            src_lang: [0; 8],
            tgt_lang: [0; 8],
            matrix: identity(dim),
            quality: 1.0,
        };
        let v = vec![1.0, 2.0, 3.0];
        let out = am
            .apply(&v)
            .expect("test: AlignmentMatrix::apply on identity matrix failed");
        assert_eq!(out, v);
    }

    #[test]
    fn test_alignment_matrix_apply_dim_mismatch() {
        let am = AlignmentMatrix {
            src_lang: [0; 8],
            tgt_lang: [0; 8],
            matrix: identity(3), // 3×3 = 9 elements
            quality: 1.0,
        };
        // Applying to a 4-element vector: matrix.len()=9 ≠ 4*4=16
        let res = am.apply(&[1.0, 2.0, 3.0, 4.0]);
        assert!(matches!(res, Err(MeaError::DimensionMismatch { .. })));
    }

    // ── 19: multi-hop alignment chains ───────────────────────────────────────

    #[test]
    fn test_multiple_language_pairs() {
        let dim = 4;
        let mut a = make_aligner(dim, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", dim);
        let fr = add_lang(&mut a, "fr", dim);
        let de = add_lang(&mut a, "de", dim);
        for i in 0..5u64 {
            let v = vec![i as f64 + 1.0, 0.0, 0.0, 0.0];
            add_emb(&mut a, en, i, v.clone());
            add_emb(&mut a, fr, i, v.clone());
            add_emb(&mut a, de, i, v);
        }
        let anch: Vec<(u64, u64)> = (0..5).map(|i| (i, i)).collect();
        a.compute_alignment(en, fr, &anch)
            .expect("test: compute_alignment en->fr for multiple pairs failed");
        a.compute_alignment(en, de, &anch)
            .expect("test: compute_alignment en->de for multiple pairs failed");
        assert_eq!(a.alignment_matrices.len(), 2);
    }

    // ── 20: embedding_count helper ────────────────────────────────────────────

    #[test]
    fn test_embedding_count_none_for_unknown_lang() {
        let a = make_aligner(4, MeaAlignmentMethod::IdentityPassthrough);
        assert!(a.embedding_count(make_lang_id("xx")).is_none());
    }

    #[test]
    fn test_embedding_count_increments() {
        let dim = 4;
        let mut a = make_aligner(dim, MeaAlignmentMethod::IdentityPassthrough);
        let en = add_lang(&mut a, "en", dim);
        add_emb(&mut a, en, 1, vec![1.0, 0.0, 0.0, 0.0]);
        add_emb(&mut a, en, 2, vec![0.0, 1.0, 0.0, 0.0]);
        assert_eq!(a.embedding_count(en), Some(2));
    }

    // ── 21: type alias availability ───────────────────────────────────────────

    #[test]
    fn test_type_aliases() {
        let _: MeaMultilingualEmbeddingAligner = MultilingualEmbeddingAligner::with_defaults();
        let _id: LangId = make_lang_id("test");
    }
}
