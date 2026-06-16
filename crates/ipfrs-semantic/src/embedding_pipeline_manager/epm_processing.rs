//! Internal processing functions and state for the Embedding Pipeline Manager.

use std::collections::HashMap;

use super::epm_types::EpmReductionMethod;

// ---------------------------------------------------------------------------
// PRNG — xorshift64, no `rand` crate dependency
// ---------------------------------------------------------------------------

#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ---------------------------------------------------------------------------
// Core math helpers (public — re-exported from mod.rs)
// ---------------------------------------------------------------------------

/// L2-normalise a vector in-place. Vectors with near-zero norm are left unchanged.
pub fn l2_normalize(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Element-wise mean of a slice of equal-length vectors.
/// Returns an empty vector when `vecs` is empty.
pub fn mean_pool(vecs: &[Vec<f64>]) -> Vec<f64> {
    if vecs.is_empty() {
        return vec![];
    }
    let dim = vecs[0].len();
    let mut result = vec![0.0_f64; dim];
    for v in vecs {
        for (r, x) in result.iter_mut().zip(v.iter()) {
            *r += x;
        }
    }
    let n = vecs.len() as f64;
    for r in result.iter_mut() {
        *r /= n;
    }
    result
}

/// Project `v` into `target_dim` dimensions using a deterministic random matrix
/// seeded with `seed` (Johnson-Lindenstrauss style).
pub fn random_projection(v: &[f64], target_dim: usize, seed: u64) -> Vec<f64> {
    if target_dim == 0 || v.is_empty() {
        return vec![0.0; target_dim];
    }
    let mut rng = seed;
    let mut result = vec![0.0_f64; target_dim];
    for (i, res) in result.iter_mut().enumerate() {
        let mut proj = 0.0_f64;
        let row_seed = xorshift64(&mut rng) ^ (i as u64);
        let mut row_rng = row_seed;
        for &x in v {
            let r = (xorshift64(&mut row_rng) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
            proj += r * x;
        }
        *res = proj / (target_dim as f64).sqrt();
    }
    result
}

// ---------------------------------------------------------------------------
// Internal mutable state
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub(super) struct PipelineState {
    pub(super) batches_processed: u64,
    pub(super) total_inputs: u64,
    pub(super) total_batch_time_us: u64,
    // stage_name -> (total_time_us, total_processed)
    pub(super) stage_time: HashMap<String, (u64, u64)>,
}

impl PipelineState {
    pub(super) fn record_batch(&mut self, n_inputs: usize, batch_us: u64) {
        self.batches_processed += 1;
        self.total_inputs += n_inputs as u64;
        self.total_batch_time_us += batch_us;
    }

    pub(super) fn record_stage(&mut self, name: &str, time_us: u64, n: usize) {
        let entry = self.stage_time.entry(name.to_string()).or_insert((0, 0));
        entry.0 += time_us;
        entry.1 += n as u64;
    }

    pub(super) fn avg_batch_time_us(&self) -> f64 {
        if self.batches_processed == 0 {
            0.0
        } else {
            self.total_batch_time_us as f64 / self.batches_processed as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Tokenisation helpers (pure Rust)
// ---------------------------------------------------------------------------

pub(super) fn tokenize_text(text: &str, lowercase: bool, strip_punct: bool) -> Vec<String> {
    let processed: String = if lowercase {
        text.to_lowercase()
    } else {
        text.to_string()
    };
    processed
        .split_whitespace()
        .map(|tok| {
            if strip_punct {
                tok.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
            } else {
                tok.to_string()
            }
        })
        .filter(|t| !t.is_empty())
        .collect()
}

pub(super) fn apply_stop_word_filter(tokens: Vec<String>, stop_words: &[String]) -> Vec<String> {
    let set: std::collections::HashSet<&str> = stop_words.iter().map(String::as_str).collect();
    tokens
        .into_iter()
        .filter(|t| !set.contains(t.as_str()))
        .collect()
}

pub(super) fn apply_ngram(tokens: &[String], n: usize) -> Vec<String> {
    if n <= 1 || tokens.len() < n {
        return tokens.to_vec();
    }
    tokens.windows(n).map(|w| w.join("_")).collect()
}

// ---------------------------------------------------------------------------
// TF-IDF helpers
// ---------------------------------------------------------------------------

/// Compute term frequency map for a token list.
pub(super) fn term_frequencies(tokens: &[String]) -> HashMap<String, f64> {
    let mut counts: HashMap<String, u64> = HashMap::new();
    for t in tokens {
        *counts.entry(t.clone()).or_insert(0) += 1;
    }
    let total = tokens.len().max(1) as f64;
    counts
        .into_iter()
        .map(|(k, v)| (k, v as f64 / total))
        .collect()
}

/// Compute IDF values from a collection of token lists (one per document).
pub(super) fn inverse_document_frequencies(all_tokens: &[Vec<String>]) -> HashMap<String, f64> {
    let n_docs = all_tokens.len() as f64;
    let mut doc_count: HashMap<String, u64> = HashMap::new();
    for doc in all_tokens {
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for t in doc {
            if seen.insert(t.as_str()) {
                *doc_count.entry(t.clone()).or_insert(0) += 1;
            }
        }
    }
    doc_count
        .into_iter()
        .map(|(term, df)| {
            let idf = ((n_docs + 1.0) / (df as f64 + 1.0)).ln() + 1.0;
            (term, idf)
        })
        .collect()
}

/// Build a sorted vocabulary from a collection of TF maps.
pub(super) fn build_vocab(tf_maps: &[HashMap<String, f64>]) -> Vec<String> {
    let mut vocab_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for tf in tf_maps {
        for k in tf.keys() {
            vocab_set.insert(k.clone());
        }
    }
    vocab_set.into_iter().collect()
}

/// Convert a TF map + IDF map into a dense vocabulary-length vector.
pub fn tfidf_vector(
    tf: &HashMap<String, f64>,
    idf: &HashMap<String, f64>,
    vocab: &[String],
) -> Vec<f64> {
    vocab
        .iter()
        .map(|term| {
            let tf_val = tf.get(term).copied().unwrap_or(0.0);
            let idf_val = idf.get(term).copied().unwrap_or(1.0);
            tf_val * idf_val
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Positional encoding
// ---------------------------------------------------------------------------

pub(super) fn add_positional_encoding(v: &mut [f64], position: usize, max_len: usize) {
    let dim = v.len();
    if dim == 0 || max_len == 0 {
        return;
    }
    let pos = position.min(max_len - 1) as f64;
    for (i, val) in v.iter_mut().enumerate() {
        let freq = 1.0 / (10000.0_f64).powf((2 * (i / 2)) as f64 / dim as f64);
        let enc = if i % 2 == 0 {
            (pos * freq).sin()
        } else {
            (pos * freq).cos()
        };
        *val += enc;
    }
}

// ---------------------------------------------------------------------------
// Dimension reduction
// ---------------------------------------------------------------------------

pub(super) fn reduce_dimensions(
    v: &[f64],
    target_dim: usize,
    method: &EpmReductionMethod,
    corpus_stats: Option<&CorpusStats>,
) -> Vec<f64> {
    let src_dim = v.len();
    if target_dim >= src_dim {
        // Pad with zeros if target is larger.
        let mut out = v.to_vec();
        out.resize(target_dim, 0.0);
        return out;
    }
    match method {
        EpmReductionMethod::TruncateDims => v[..target_dim].to_vec(),
        EpmReductionMethod::RandomProjection(seed) => random_projection(v, target_dim, *seed),
        EpmReductionMethod::MeanPooling => {
            // Split v into `target_dim` chunks and average each.
            let chunk = src_dim / target_dim;
            if chunk == 0 {
                return v[..target_dim].to_vec();
            }
            (0..target_dim)
                .map(|i| {
                    let start = i * chunk;
                    let end = if i + 1 == target_dim {
                        src_dim
                    } else {
                        start + chunk
                    };
                    let slice = &v[start..end];
                    slice.iter().sum::<f64>() / slice.len() as f64
                })
                .collect()
        }
        EpmReductionMethod::PCA => {
            // Simplified PCA: subtract per-dimension mean (if available) and
            // keep first target_dim dimensions sorted by variance contribution.
            if let Some(stats) = corpus_stats {
                let centred: Vec<f64> = v
                    .iter()
                    .zip(stats.mean.iter())
                    .map(|(x, m)| x - m)
                    .collect();
                // Sort dimension indices by descending variance and take top `target_dim`.
                let mut indices: Vec<usize> = (0..src_dim).collect();
                indices.sort_by(|&a, &b| {
                    stats.variance[b]
                        .partial_cmp(&stats.variance[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                indices
                    .into_iter()
                    .take(target_dim)
                    .map(|i| centred[i])
                    .collect()
            } else {
                // No corpus stats — fall back to truncation.
                v[..target_dim].to_vec()
            }
        }
    }
}

/// Per-dimension mean and variance computed from a batch for PCA fallback.
#[derive(Debug, Clone, Default)]
pub(super) struct CorpusStats {
    pub(super) mean: Vec<f64>,
    pub(super) variance: Vec<f64>,
}

impl CorpusStats {
    pub(super) fn from_embeddings(embeddings: &[Vec<f64>]) -> Option<Self> {
        if embeddings.is_empty() {
            return None;
        }
        let dim = embeddings[0].len();
        if dim == 0 {
            return None;
        }
        let n = embeddings.len() as f64;
        let mut mean = vec![0.0_f64; dim];
        for v in embeddings {
            for (m, x) in mean.iter_mut().zip(v.iter()) {
                *m += x;
            }
        }
        for m in mean.iter_mut() {
            *m /= n;
        }
        let mut variance = vec![0.0_f64; dim];
        for v in embeddings {
            for (var, (x, m)) in variance.iter_mut().zip(v.iter().zip(mean.iter())) {
                let diff = x - m;
                *var += diff * diff;
            }
        }
        for var in variance.iter_mut() {
            *var /= n;
        }
        Some(Self { mean, variance })
    }
}

// ---------------------------------------------------------------------------
// QuantizeToByte
// ---------------------------------------------------------------------------

pub(super) fn quantize_to_byte(v: &[f64]) -> Vec<f64> {
    if v.is_empty() {
        return vec![];
    }
    let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    if range < 1e-10 {
        return vec![0.0; v.len()];
    }
    v.iter()
        .map(|x| (((x - min) / range) * 255.0).round())
        .collect()
}
