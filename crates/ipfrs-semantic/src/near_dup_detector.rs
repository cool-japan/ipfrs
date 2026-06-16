//! Semantic Near-Duplicate Detector
//!
//! Detects near-duplicate embeddings using locality-sensitive hashing (LSH)
//! with random projection bands, enabling sub-linear duplicate detection at
//! scale.  The projection is fully deterministic — seeded by (band_id, row,
//! dimension) via FNV-1a — so no PRNG dependency is required.

use std::collections::HashMap;

// ── FNV-1a constants ──────────────────────────────────────────────────────────

const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;

/// Perform one step of FNV-1a hashing on a single byte.
#[inline]
fn fnv1a_step(hash: u64, byte: u8) -> u64 {
    (hash ^ (byte as u64)).wrapping_mul(FNV_PRIME)
}

/// Hash a single `u64` value using FNV-1a and return the resulting hash.
#[inline]
fn fnv1a_u64(value: u64) -> u64 {
    let mut h = FNV_OFFSET_BASIS;
    for b in value.to_le_bytes() {
        h = fnv1a_step(h, b);
    }
    h
}

// ── Cosine similarity helper ──────────────────────────────────────────────────

/// Compute cosine similarity between two embedding vectors.
///
/// Returns `0.0` if either vector has zero magnitude (< 1e-8).
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0_f32;
    let mut mag_a = 0.0_f32;
    let mut mag_b = 0.0_f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        mag_a += x * x;
        mag_b += y * y;
    }

    let mag_a = mag_a.sqrt();
    let mag_b = mag_b.sqrt();

    if mag_a < 1e-8 || mag_b < 1e-8 {
        return 0.0;
    }

    dot / (mag_a * mag_b)
}

// ── LSH projection ────────────────────────────────────────────────────────────

/// Compute the deterministic LSH band hash for `embedding` given a specific
/// band and configuration.
///
/// Algorithm (per requirement §4):
/// - For each row `r` in `0..rows_per_band`:
///   - For each dimension `d`: derive a projection sign ∈ {−1, +1} from
///     `fnv1a(band_id * 1000 + r * 100 + d)` mod 2; compute a single bit from
///     the sign of `projection_sign * val.signum()`.
///   - Pack the per-dimension bits into a `u64` row hash (bits past 63 are
///     ignored — embeddings longer than 64 dims wrap around bit positions).
/// - XOR all row hashes together to form the band hash.
fn compute_band_hash(embedding: &[f32], band_id: usize, rows_per_band: usize) -> u64 {
    let mut band_hash: u64 = 0;

    for r in 0..rows_per_band {
        let mut row_hash: u64 = 0;

        for (d, &val) in embedding.iter().enumerate() {
            let seed = (band_id as u64)
                .wrapping_mul(1000)
                .wrapping_add((r as u64).wrapping_mul(100))
                .wrapping_add(d as u64);

            let projection_bit = (fnv1a_u64(seed) % 2) as i32 * 2 - 1; // +1 or -1

            // val.signum() ∈ {-1, 0, 1}; treat 0 as non-negative (maps to bit 1)
            let val_sign = if val >= 0.0 { 1_i32 } else { -1_i32 };

            let bit: u64 = if projection_bit * val_sign >= 0 { 1 } else { 0 };

            // Accumulate: shift bit into the row hash at position (d % 64)
            row_hash |= bit << (d % 64);
        }

        band_hash ^= row_hash;
    }

    band_hash
}

// ── DupCandidate ──────────────────────────────────────────────────────────────

/// A near-duplicate candidate pair identified by the LSH detector.
#[derive(Clone, Debug, PartialEq)]
pub struct DupCandidate {
    /// Identifier of the first embedding in the pair.
    pub id_a: u64,
    /// Identifier of the second embedding in the pair.
    pub id_b: u64,
    /// Cosine similarity between the two embeddings.
    pub similarity: f32,
}

impl DupCandidate {
    /// Returns a canonical (sorted) key for the pair so that `(a, b)` and
    /// `(b, a)` map to the same key.
    #[inline]
    pub fn pair_key(&self) -> (u64, u64) {
        (self.id_a.min(self.id_b), self.id_a.max(self.id_b))
    }
}

// ── LshBand ───────────────────────────────────────────────────────────────────

/// A single LSH band containing hash buckets that map `u64` hashes to lists
/// of embedding IDs that hashed into each bucket.
pub struct LshBand {
    /// Zero-based index of this band.
    pub band_id: usize,
    /// Mapping from bucket hash to the IDs of embeddings assigned to it.
    pub buckets: HashMap<u64, Vec<u64>>,
}

impl LshBand {
    /// Create a new empty band with the given `band_id`.
    pub fn new(band_id: usize) -> Self {
        Self {
            band_id,
            buckets: HashMap::new(),
        }
    }

    /// Insert `embedding_id` into the bucket for `hash`.
    pub fn insert(&mut self, embedding_id: u64, hash: u64) {
        self.buckets.entry(hash).or_default().push(embedding_id);
    }

    /// Return the slice of embedding IDs in the bucket for `hash`, or an empty
    /// slice if the hash has no bucket.
    pub fn candidates_for(&self, hash: u64) -> &[u64] {
        match self.buckets.get(&hash) {
            Some(ids) => ids.as_slice(),
            None => &[],
        }
    }

    /// Remove `id` from every bucket in this band.
    fn remove_id(&mut self, id: u64) {
        for ids in self.buckets.values_mut() {
            ids.retain(|&existing| existing != id);
        }
        // Drop now-empty buckets to keep memory lean.
        self.buckets.retain(|_, ids| !ids.is_empty());
    }

    /// Total number of entries across all buckets.
    fn total_entries(&self) -> usize {
        self.buckets.values().map(|v| v.len()).sum()
    }

    /// Number of non-empty buckets.
    fn non_empty_bucket_count(&self) -> usize {
        self.buckets.values().filter(|v| !v.is_empty()).count()
    }
}

// ── NearDupConfig ─────────────────────────────────────────────────────────────

/// Configuration for [`SemanticNearDupDetector`].
#[derive(Clone, Debug)]
pub struct NearDupConfig {
    /// Number of LSH bands (default 10).
    pub num_bands: usize,
    /// Number of hash rows per band (default 4).
    pub rows_per_band: usize,
    /// Minimum cosine similarity for a pair to be reported as a near-duplicate
    /// (default 0.90).
    pub similarity_threshold: f32,
}

impl Default for NearDupConfig {
    fn default() -> Self {
        Self {
            num_bands: 10,
            rows_per_band: 4,
            similarity_threshold: 0.90,
        }
    }
}

// ── DupDetectorStats ──────────────────────────────────────────────────────────

/// Aggregate statistics for [`SemanticNearDupDetector`].
#[derive(Clone, Debug)]
pub struct DupDetectorStats {
    /// Total number of embeddings currently stored.
    pub total_embeddings: usize,
    /// Sum of all bucket sizes across every band.
    pub total_band_entries: usize,
    /// Average bucket size over non-empty buckets; `0.0` when no buckets exist.
    pub avg_bucket_size: f64,
}

// ── SemanticNearDupDetector ───────────────────────────────────────────────────

/// Detects near-duplicate embeddings using locality-sensitive hashing (LSH).
///
/// Embeddings are indexed into `num_bands` LSH bands.  Two embeddings that
/// collide (share the same bucket hash) in at least one band are considered
/// *candidates* and are verified with an exact cosine-similarity check against
/// [`NearDupConfig::similarity_threshold`].
///
/// The projection used to derive bucket hashes is fully deterministic — each
/// projection vector is synthesised from (band_id, row, dimension) via FNV-1a —
/// so results are reproducible without any PRNG state.
pub struct SemanticNearDupDetector {
    /// LSH bands.
    pub bands: Vec<LshBand>,
    /// Stored embeddings keyed by their ID.
    pub embeddings: HashMap<u64, Vec<f32>>,
    /// Detector configuration.
    pub config: NearDupConfig,
}

impl SemanticNearDupDetector {
    /// Create a new detector initialised with `config.num_bands` empty bands.
    pub fn new(config: NearDupConfig) -> Self {
        let bands = (0..config.num_bands).map(LshBand::new).collect();

        Self {
            bands,
            embeddings: HashMap::new(),
            config,
        }
    }

    /// Store `embedding` under `id` and insert it into every LSH band.
    pub fn insert(&mut self, id: u64, embedding: Vec<f32>) {
        // Store the embedding.
        self.embeddings.insert(id, embedding.clone());

        // Insert into every band.
        let rows_per_band = self.config.rows_per_band;
        for band in self.bands.iter_mut() {
            let hash = compute_band_hash(&embedding, band.band_id, rows_per_band);
            band.insert(id, hash);
        }
    }

    /// Find all stored embeddings that are near-duplicates of `query`.
    ///
    /// For each LSH band the band hash is computed for `query`; every ID in
    /// the corresponding bucket is treated as a candidate.  Candidates are
    /// deduplicated, their exact cosine similarity with `query` is computed,
    /// and only those ≥ `similarity_threshold` are returned.
    ///
    /// Results are sorted by similarity descending.
    pub fn find_candidates(&self, query: &[f32]) -> Vec<DupCandidate> {
        let rows_per_band = self.config.rows_per_band;
        let mut seen: HashMap<u64, f32> = HashMap::new();

        for band in &self.bands {
            let hash = compute_band_hash(query, band.band_id, rows_per_band);
            for &cand_id in band.candidates_for(hash) {
                if seen.contains_key(&cand_id) {
                    continue;
                }
                if let Some(stored) = self.embeddings.get(&cand_id) {
                    let sim = cosine_sim(query, stored);
                    seen.insert(cand_id, sim);
                }
            }
        }

        let threshold = self.config.similarity_threshold;
        let mut results: Vec<DupCandidate> = seen
            .into_iter()
            .filter(|&(_, sim)| sim >= threshold)
            .map(|(id, sim)| DupCandidate {
                id_a: 0, // query has no stored ID; use 0 as sentinel
                id_b: id,
                similarity: sim,
            })
            .collect();

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Detect all near-duplicate pairs within the stored embedding set.
    ///
    /// For every LSH band bucket with ≥ 2 IDs, all pairwise cosine similarities
    /// are computed.  Pairs that meet the threshold are collected, deduplicated
    /// by canonical [`DupCandidate::pair_key`], and sorted by similarity
    /// descending.
    pub fn find_duplicates_in_set(&self) -> Vec<DupCandidate> {
        let threshold = self.config.similarity_threshold;
        let mut seen: HashMap<(u64, u64), f32> = HashMap::new();

        for band in &self.bands {
            for ids in band.buckets.values() {
                if ids.len() < 2 {
                    continue;
                }
                // All pairs within this bucket.
                for i in 0..ids.len() {
                    for j in (i + 1)..ids.len() {
                        let id_a = ids[i];
                        let id_b = ids[j];
                        let key = (id_a.min(id_b), id_a.max(id_b));

                        if seen.contains_key(&key) {
                            continue;
                        }

                        let sim = match (self.embeddings.get(&id_a), self.embeddings.get(&id_b)) {
                            (Some(ea), Some(eb)) => cosine_sim(ea, eb),
                            _ => continue,
                        };

                        if sim >= threshold {
                            seen.insert(key, sim);
                        }
                    }
                }
            }
        }

        let mut results: Vec<DupCandidate> = seen
            .into_iter()
            .map(|((a, b), sim)| DupCandidate {
                id_a: a,
                id_b: b,
                similarity: sim,
            })
            .collect();

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Remove the embedding with `id` from the store and from every band bucket.
    pub fn remove(&mut self, id: u64) {
        self.embeddings.remove(&id);
        for band in self.bands.iter_mut() {
            band.remove_id(id);
        }
    }

    /// Return aggregate statistics for the detector.
    pub fn stats(&self) -> DupDetectorStats {
        let total_embeddings = self.embeddings.len();

        let total_band_entries: usize = self.bands.iter().map(|b| b.total_entries()).sum();

        let non_empty: usize = self.bands.iter().map(|b| b.non_empty_bucket_count()).sum();

        let avg_bucket_size = if non_empty == 0 {
            0.0
        } else {
            total_band_entries as f64 / non_empty as f64
        };

        DupDetectorStats {
            total_embeddings,
            total_band_entries,
            avg_bucket_size,
        }
    }
}

// ── MinHash Near-Duplicate Detection ─────────────────────────────────────────

/// Configuration for [`MinHashNearDupDetector`].
#[derive(Debug, Clone)]
pub struct MinHashConfig {
    /// Number of hash functions (default 128).
    pub num_hashes: usize,
    /// Jaccard similarity threshold for duplicate detection (default 0.8).
    pub similarity_threshold: f64,
    /// Seed for reproducible hash generation.
    pub seed: u64,
}

impl Default for MinHashConfig {
    fn default() -> Self {
        Self {
            num_hashes: 128,
            similarity_threshold: 0.8,
            seed: 42,
        }
    }
}

/// A MinHash signature representing a document's shingle set.
#[derive(Debug, Clone)]
pub struct MinHashSignature {
    /// Document identifier.
    pub doc_id: String,
    /// Min-hash values (one per hash function).
    pub hashes: Vec<u64>,
}

/// A pair of documents identified as near-duplicates.
#[derive(Debug, Clone)]
pub struct DuplicatePair {
    /// First document identifier.
    pub doc_a: String,
    /// Second document identifier.
    pub doc_b: String,
    /// Estimated Jaccard similarity from MinHash signatures.
    pub estimated_similarity: f64,
}

/// Aggregate statistics for [`MinHashNearDupDetector`].
#[derive(Debug, Clone)]
pub struct NearDupDetectorStats {
    /// Number of documents currently stored.
    pub document_count: usize,
    /// Number of hash functions configured.
    pub num_hashes: usize,
    /// Number of times `find_duplicates` or `find_duplicates_for` has been called.
    pub detections_run: u64,
}

/// Near-duplicate document detector using MinHash signatures.
///
/// MinHash approximates Jaccard similarity between token sets by computing
/// multiple hash functions over each document's tokens and retaining the
/// minimum hash value per function. The fraction of matching minimums between
/// two signatures estimates their Jaccard similarity.
///
/// Hash function `i` for a token is computed as FNV-1a over the token bytes
/// seeded with `config.seed + i`.
pub struct MinHashNearDupDetector {
    config: MinHashConfig,
    signatures: HashMap<String, MinHashSignature>,
    detections_run: u64,
}

impl MinHashNearDupDetector {
    /// Create a new detector with the given configuration.
    pub fn new(config: MinHashConfig) -> Self {
        Self {
            config,
            signatures: HashMap::new(),
            detections_run: 0,
        }
    }

    /// Compute a MinHash signature for the given document tokens.
    ///
    /// For each of the `num_hashes` hash functions, every token is hashed
    /// using FNV-1a seeded with `config.seed + hash_index`, and the minimum
    /// hash value is retained. Empty token sets produce `u64::MAX` for all
    /// hash positions.
    pub fn compute_signature(&self, doc_id: &str, tokens: &[&str]) -> MinHashSignature {
        let num = self.config.num_hashes;
        let mut hashes = vec![u64::MAX; num];

        for (hash_idx, slot) in hashes.iter_mut().enumerate() {
            let seed = self.config.seed.wrapping_add(hash_idx as u64);
            for &token in tokens {
                let h = Self::hash_token(token, seed);
                if h < *slot {
                    *slot = h;
                }
            }
        }

        MinHashSignature {
            doc_id: doc_id.to_string(),
            hashes,
        }
    }

    /// Compute and store a signature for the document.
    pub fn add_document(&mut self, doc_id: &str, tokens: &[&str]) {
        let sig = self.compute_signature(doc_id, tokens);
        self.signatures.insert(doc_id.to_string(), sig);
    }

    /// Remove a document's signature. Returns `true` if it existed.
    pub fn remove_document(&mut self, doc_id: &str) -> bool {
        self.signatures.remove(doc_id).is_some()
    }

    /// Estimate the Jaccard similarity between two stored documents.
    ///
    /// Returns `None` if either document is not found.
    pub fn estimate_similarity(&self, doc_a: &str, doc_b: &str) -> Option<f64> {
        let sig_a = self.signatures.get(doc_a)?;
        let sig_b = self.signatures.get(doc_b)?;
        Some(Self::jaccard_similarity(&sig_a.hashes, &sig_b.hashes))
    }

    /// Find all near-duplicate pairs among stored documents whose estimated
    /// similarity meets or exceeds the configured threshold.
    pub fn find_duplicates(&mut self) -> Vec<DuplicatePair> {
        self.detections_run += 1;
        let threshold = self.config.similarity_threshold;

        let ids: Vec<String> = self.signatures.keys().cloned().collect();
        let mut pairs = Vec::new();

        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                if let (Some(sa), Some(sb)) =
                    (self.signatures.get(&ids[i]), self.signatures.get(&ids[j]))
                {
                    let sim = Self::jaccard_similarity(&sa.hashes, &sb.hashes);
                    if sim >= threshold {
                        pairs.push(DuplicatePair {
                            doc_a: ids[i].clone(),
                            doc_b: ids[j].clone(),
                            estimated_similarity: sim,
                        });
                    }
                }
            }
        }

        pairs.sort_by(|a, b| {
            b.estimated_similarity
                .partial_cmp(&a.estimated_similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        pairs
    }

    /// Find near-duplicate pairs involving a specific document.
    pub fn find_duplicates_for(&mut self, doc_id: &str) -> Vec<DuplicatePair> {
        self.detections_run += 1;
        let threshold = self.config.similarity_threshold;

        let target_sig = match self.signatures.get(doc_id) {
            Some(s) => s.hashes.clone(),
            None => return Vec::new(),
        };

        let mut pairs = Vec::new();
        for (other_id, other_sig) in &self.signatures {
            if other_id == doc_id {
                continue;
            }
            let sim = Self::jaccard_similarity(&target_sig, &other_sig.hashes);
            if sim >= threshold {
                pairs.push(DuplicatePair {
                    doc_a: doc_id.to_string(),
                    doc_b: other_id.clone(),
                    estimated_similarity: sim,
                });
            }
        }

        pairs.sort_by(|a, b| {
            b.estimated_similarity
                .partial_cmp(&a.estimated_similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        pairs
    }

    /// Compute Jaccard similarity between two MinHash signatures.
    ///
    /// Returns the fraction of hash positions where the two signatures agree.
    pub fn jaccard_similarity(sig_a: &[u64], sig_b: &[u64]) -> f64 {
        if sig_a.is_empty() || sig_b.is_empty() {
            return 0.0;
        }
        let total = sig_a.len().min(sig_b.len());
        let matching = sig_a
            .iter()
            .zip(sig_b.iter())
            .filter(|(a, b)| a == b)
            .count();
        matching as f64 / total as f64
    }

    /// Number of documents currently stored.
    pub fn document_count(&self) -> usize {
        self.signatures.len()
    }

    /// Return aggregate statistics.
    pub fn stats(&self) -> NearDupDetectorStats {
        NearDupDetectorStats {
            document_count: self.signatures.len(),
            num_hashes: self.config.num_hashes,
            detections_run: self.detections_run,
        }
    }

    /// Hash a token using FNV-1a with a seed.
    #[inline]
    fn hash_token(token: &str, seed: u64) -> u64 {
        // Seed the hash by mixing `seed` into the FNV offset basis.
        let mut h = FNV_OFFSET_BASIS;
        for b in seed.to_le_bytes() {
            h = fnv1a_step(h, b);
        }
        for b in token.as_bytes() {
            h = fnv1a_step(h, *b);
        }
        h
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn default_config() -> NearDupConfig {
        NearDupConfig::default()
    }

    /// Unit vector in `dim`-dimensional space with only position `active` set.
    fn unit_vec(dim: usize, active: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; dim];
        if active < dim {
            v[active] = 1.0;
        }
        v
    }

    /// A vector almost identical to `base` with a tiny perturbation.
    fn near_vec(base: &[f32], delta: f32) -> Vec<f32> {
        let mut v = base.to_vec();
        // Add delta to first element, which changes the direction very little.
        v[0] += delta;
        v
    }

    // ── 1. new() creates correct number of bands ──────────────────────────────

    #[test]
    fn test_new_creates_correct_number_of_bands() {
        let det = SemanticNearDupDetector::new(NearDupConfig {
            num_bands: 7,
            ..default_config()
        });
        assert_eq!(det.bands.len(), 7);
    }

    // ── 2. insert stores embedding ────────────────────────────────────────────

    #[test]
    fn test_insert_stores_embedding() {
        let mut det = SemanticNearDupDetector::new(default_config());
        let emb = unit_vec(8, 0);
        det.insert(42, emb.clone());
        assert_eq!(det.embeddings.get(&42), Some(&emb));
    }

    // ── 3. find_candidates returns self-match ─────────────────────────────────

    #[test]
    fn test_find_candidates_self_match() {
        let mut det = SemanticNearDupDetector::new(default_config());
        let emb = unit_vec(8, 0);
        det.insert(1, emb.clone());
        let candidates = det.find_candidates(&emb);
        // Should find id=1 with similarity 1.0
        assert!(
            candidates
                .iter()
                .any(|c| c.id_b == 1 && (c.similarity - 1.0).abs() < 1e-5),
            "self-match must be found; got {:?}",
            candidates
        );
    }

    // ── 4. find_candidates returns empty for orthogonal vectors ───────────────

    #[test]
    fn test_find_candidates_empty_for_orthogonal() {
        let mut det = SemanticNearDupDetector::new(NearDupConfig {
            similarity_threshold: 0.90,
            ..default_config()
        });
        // Insert 16-dim unit vector along axis 0
        det.insert(1, unit_vec(16, 0));
        // Query with axis 1 (orthogonal, cosine_sim == 0.0)
        let candidates = det.find_candidates(&unit_vec(16, 1));
        let above_threshold: Vec<_> = candidates.iter().filter(|c| c.similarity >= 0.90).collect();
        assert!(
            above_threshold.is_empty(),
            "orthogonal query must yield no candidates above threshold"
        );
    }

    // ── 5. find_duplicates_in_set detects near-dup pair ───────────────────────

    #[test]
    fn test_find_duplicates_in_set_detects_near_dup() {
        let mut det = SemanticNearDupDetector::new(NearDupConfig {
            num_bands: 20,
            rows_per_band: 2,
            similarity_threshold: 0.90,
        });

        // Two almost-identical vectors
        let base = vec![1.0_f32; 32];
        let close = near_vec(&base, 0.01);

        det.insert(10, base);
        det.insert(11, close);

        let dups = det.find_duplicates_in_set();
        assert!(!dups.is_empty(), "near-duplicate pair must be detected");
        let pair_keys: Vec<_> = dups.iter().map(|d| d.pair_key()).collect();
        assert!(
            pair_keys.contains(&(10, 11)),
            "pair (10,11) must be in results"
        );
    }

    // ── 6. find_duplicates_in_set excludes dissimilar pairs ───────────────────

    #[test]
    fn test_find_duplicates_in_set_excludes_dissimilar() {
        let mut det = SemanticNearDupDetector::new(NearDupConfig {
            similarity_threshold: 0.90,
            ..default_config()
        });

        // Orthogonal 16-dim unit vectors → cosine_sim == 0.0
        det.insert(1, unit_vec(16, 0));
        det.insert(2, unit_vec(16, 1));
        det.insert(3, unit_vec(16, 2));

        let dups = det.find_duplicates_in_set();
        assert!(
            dups.is_empty(),
            "orthogonal vectors must not be reported as near-duplicates"
        );
    }

    // ── 7. DupCandidate pair_key canonical ordering ───────────────────────────

    #[test]
    fn test_dup_candidate_pair_key_canonical() {
        let c1 = DupCandidate {
            id_a: 5,
            id_b: 3,
            similarity: 0.95,
        };
        let c2 = DupCandidate {
            id_a: 3,
            id_b: 5,
            similarity: 0.95,
        };
        assert_eq!(c1.pair_key(), (3, 5));
        assert_eq!(c2.pair_key(), (3, 5));
    }

    // ── 8. pair_key deduplication: (a,b) and (b,a) yield the same key ─────────

    #[test]
    fn test_pair_key_deduplication() {
        let a = DupCandidate {
            id_a: 100,
            id_b: 200,
            similarity: 0.92,
        };
        let b = DupCandidate {
            id_a: 200,
            id_b: 100,
            similarity: 0.92,
        };
        assert_eq!(
            a.pair_key(),
            b.pair_key(),
            "(a,b) and (b,a) must have the same pair_key"
        );
    }

    // ── 9. remove clears from embeddings ─────────────────────────────────────

    #[test]
    fn test_remove_clears_from_embeddings() {
        let mut det = SemanticNearDupDetector::new(default_config());
        det.insert(7, unit_vec(8, 0));
        assert!(det.embeddings.contains_key(&7));
        det.remove(7);
        assert!(
            !det.embeddings.contains_key(&7),
            "embedding must be removed after remove()"
        );
    }

    // ── 10. stats total_embeddings ────────────────────────────────────────────

    #[test]
    fn test_stats_total_embeddings() {
        let mut det = SemanticNearDupDetector::new(default_config());
        det.insert(1, unit_vec(8, 0));
        det.insert(2, unit_vec(8, 1));
        det.insert(3, unit_vec(8, 2));
        assert_eq!(det.stats().total_embeddings, 3);
    }

    // ── 11. stats total_band_entries ──────────────────────────────────────────

    #[test]
    fn test_stats_total_band_entries() {
        let num_bands = 5;
        let mut det = SemanticNearDupDetector::new(NearDupConfig {
            num_bands,
            ..default_config()
        });
        det.insert(1, unit_vec(8, 0));
        // One embedding → one entry per band
        assert_eq!(det.stats().total_band_entries, num_bands);
    }

    // ── 12. LshBand insert and candidates_for ─────────────────────────────────

    #[test]
    fn test_lsh_band_insert_and_candidates_for() {
        let mut band = LshBand::new(0);
        band.insert(42, 0xDEAD_BEEF);
        band.insert(99, 0xDEAD_BEEF);
        let cands = band.candidates_for(0xDEAD_BEEF);
        assert!(cands.contains(&42), "id 42 must be in bucket");
        assert!(cands.contains(&99), "id 99 must be in bucket");
    }

    // ── 13. LshBand candidates_for returns empty for unknown hash ─────────────

    #[test]
    fn test_lsh_band_candidates_for_unknown_hash() {
        let band = LshBand::new(0);
        assert_eq!(band.candidates_for(0xCAFE_BABE), &[] as &[u64]);
    }

    // ── 14. cosine_sim identical vectors returns 1.0 ──────────────────────────

    #[test]
    fn test_cosine_sim_identical_returns_one() {
        let v = vec![0.3_f32, 0.4, 0.0, 0.866];
        let sim = cosine_sim(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "cosine_sim of identical vectors must be ~1.0, got {sim}"
        );
    }

    // ── 15. cosine_sim orthogonal vectors returns 0.0 ─────────────────────────

    #[test]
    fn test_cosine_sim_orthogonal_returns_zero() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        let sim = cosine_sim(&a, &b);
        assert!(
            sim.abs() < 1e-5,
            "cosine_sim of orthogonal vectors must be ~0.0, got {sim}"
        );
    }

    // ── 16. find_candidates sorted by similarity descending ───────────────────

    #[test]
    fn test_find_candidates_sorted_by_similarity_desc() {
        let mut det = SemanticNearDupDetector::new(NearDupConfig {
            num_bands: 20,
            rows_per_band: 2,
            similarity_threshold: 0.80,
        });

        let base = vec![1.0_f32; 16];
        det.insert(1, base.clone()); // sim ≈ 1.0
        det.insert(2, near_vec(&base, 0.1)); // sim slightly < 1.0
        det.insert(3, near_vec(&base, 0.5)); // sim lower still

        let candidates = det.find_candidates(&base);

        // Verify descending order
        for window in candidates.windows(2) {
            assert!(
                window[0].similarity >= window[1].similarity,
                "candidates must be sorted by similarity descending"
            );
        }
    }

    // ── 17. threshold filtering works ────────────────────────────────────────

    #[test]
    fn test_threshold_filtering() {
        let mut det = SemanticNearDupDetector::new(NearDupConfig {
            num_bands: 20,
            rows_per_band: 2,
            similarity_threshold: 0.95,
        });

        let base = vec![1.0_f32; 32];
        det.insert(1, base.clone()); // similarity with itself = 1.0 ✓
        det.insert(2, near_vec(&base, 5.0)); // large perturbation → lower similarity

        let candidates = det.find_candidates(&base);

        // Every returned candidate must meet the threshold
        for c in &candidates {
            assert!(
                c.similarity >= 0.95,
                "candidate with similarity {} is below threshold",
                c.similarity
            );
        }
    }

    // ── 18. multiple inserts accumulate ──────────────────────────────────────

    #[test]
    fn test_multiple_inserts_accumulate() {
        let mut det = SemanticNearDupDetector::new(default_config());

        for i in 0..10u64 {
            det.insert(i, unit_vec(16, (i as usize) % 16));
        }

        assert_eq!(det.stats().total_embeddings, 10);
        // Each embedding appears once per band
        assert_eq!(
            det.stats().total_band_entries,
            10 * det.config.num_bands,
            "total band entries must equal num_embeddings * num_bands"
        );
    }

    // ── 19. remove also clears from band buckets ──────────────────────────────

    #[test]
    fn test_remove_clears_from_band_buckets() {
        let mut det = SemanticNearDupDetector::new(default_config());
        det.insert(5, unit_vec(8, 0));

        let entries_before = det.stats().total_band_entries;
        assert_eq!(entries_before, det.config.num_bands);

        det.remove(5);

        assert_eq!(
            det.stats().total_band_entries,
            0,
            "band entries must be zero after removing the only embedding"
        );
    }

    // ── 20. stats avg_bucket_size is 0.0 when no embeddings ──────────────────

    #[test]
    fn test_stats_avg_bucket_size_empty() {
        let det = SemanticNearDupDetector::new(default_config());
        assert_eq!(det.stats().avg_bucket_size, 0.0);
    }

    // ── 21. stats avg_bucket_size satisfies definition ───────────────────────

    #[test]
    fn test_stats_avg_bucket_size_definition() {
        let mut det = SemanticNearDupDetector::new(default_config());
        det.insert(0, unit_vec(32, 0));
        det.insert(1, unit_vec(32, 1));

        let s = det.stats();
        // Verify the invariant: avg = total_entries / non_empty_buckets
        let non_empty: usize = det.bands.iter().map(|b| b.non_empty_bucket_count()).sum();
        let expected_avg = if non_empty == 0 {
            0.0
        } else {
            s.total_band_entries as f64 / non_empty as f64
        };
        assert!(
            (s.avg_bucket_size - expected_avg).abs() < 1e-9,
            "avg_bucket_size ({}) must equal total_band_entries / non_empty_buckets ({})",
            s.avg_bucket_size,
            expected_avg
        );
        // avg must be >= 1.0 and <= num_embeddings
        assert!(s.avg_bucket_size >= 1.0, "avg_bucket_size must be >= 1.0");
        assert!(
            s.avg_bucket_size <= 2.0,
            "avg_bucket_size with 2 embeddings must be <= 2.0"
        );
    }

    // ── 22. cosine_sim zero-magnitude vectors returns 0.0 ─────────────────────

    #[test]
    fn test_cosine_sim_zero_magnitude() {
        let zero = vec![0.0_f32, 0.0, 0.0];
        let unit = vec![1.0_f32, 0.0, 0.0];
        assert_eq!(cosine_sim(&zero, &unit), 0.0);
        assert_eq!(cosine_sim(&unit, &zero), 0.0);
        assert_eq!(cosine_sim(&zero, &zero), 0.0);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // MinHash Near-Duplicate Detector tests
    // ══════════════════════════════════════════════════════════════════════════

    fn minhash_config() -> MinHashConfig {
        MinHashConfig::default()
    }

    // ── 23. identical documents have similarity ~1.0 ─────────────────────────

    #[test]
    fn test_minhash_identical_documents_similarity_one() {
        let mut det = MinHashNearDupDetector::new(minhash_config());
        let tokens = vec!["the", "quick", "brown", "fox", "jumps"];
        det.add_document("doc1", &tokens);
        det.add_document("doc2", &tokens);
        let sim = det.estimate_similarity("doc1", "doc2");
        assert!(sim.is_some());
        let sim = sim.unwrap_or(0.0);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "identical documents must have similarity 1.0, got {sim}"
        );
    }

    // ── 24. disjoint documents have similarity ~0.0 ──────────────────────────

    #[test]
    fn test_minhash_disjoint_documents_similarity_zero() {
        let mut det = MinHashNearDupDetector::new(MinHashConfig {
            num_hashes: 256,
            ..minhash_config()
        });
        let tokens_a = vec!["alpha", "beta", "gamma", "delta", "epsilon"];
        let tokens_b = vec!["one", "two", "three", "four", "five"];
        det.add_document("a", &tokens_a);
        det.add_document("b", &tokens_b);
        let sim = det.estimate_similarity("a", "b").unwrap_or(1.0);
        assert!(
            sim < 0.15,
            "disjoint documents must have near-zero similarity, got {sim}"
        );
    }

    // ── 25. overlapping tokens produce intermediate similarity ───────────────

    #[test]
    fn test_minhash_overlapping_intermediate_similarity() {
        let mut det = MinHashNearDupDetector::new(MinHashConfig {
            num_hashes: 512,
            ..minhash_config()
        });
        // 3 shared out of 5 union => Jaccard ≈ 0.6
        let tokens_a = vec!["a", "b", "c", "d", "e"];
        let tokens_b = vec!["a", "b", "c", "x", "y"];
        det.add_document("a", &tokens_a);
        det.add_document("b", &tokens_b);
        let sim = det.estimate_similarity("a", "b").unwrap_or(0.0);
        // True Jaccard = 3/7 ≈ 0.4286; MinHash is an unbiased estimator with variance
        assert!(
            sim > 0.15 && sim < 0.75,
            "overlapping tokens must produce intermediate similarity, got {sim}"
        );
    }

    // ── 26. find_duplicates returns pairs above threshold ─────────────────────

    #[test]
    fn test_minhash_find_duplicates_above_threshold() {
        let mut det = MinHashNearDupDetector::new(MinHashConfig {
            similarity_threshold: 0.8,
            ..minhash_config()
        });
        let tokens = vec!["hello", "world", "foo", "bar"];
        det.add_document("d1", &tokens);
        det.add_document("d2", &tokens);
        det.add_document("d3", &["completely", "different", "text"]);

        let dups = det.find_duplicates();
        assert!(
            dups.iter().any(|d| {
                (d.doc_a == "d1" && d.doc_b == "d2") || (d.doc_a == "d2" && d.doc_b == "d1")
            }),
            "identical docs d1 and d2 must be in duplicates"
        );
        // d3 should not pair with d1 or d2
        for d in &dups {
            assert!(
                !(d.doc_a == "d3" || d.doc_b == "d3"),
                "d3 must not appear in duplicate pairs, got {:?}",
                d
            );
        }
    }

    // ── 27. find_duplicates_for specific doc ─────────────────────────────────

    #[test]
    fn test_minhash_find_duplicates_for_specific() {
        let mut det = MinHashNearDupDetector::new(MinHashConfig {
            similarity_threshold: 0.8,
            ..minhash_config()
        });
        let tokens = vec!["rust", "is", "great"];
        det.add_document("x", &tokens);
        det.add_document("y", &tokens);
        det.add_document("z", &["python", "is", "nice"]);

        let pairs = det.find_duplicates_for("x");
        assert!(
            pairs.iter().any(|p| p.doc_b == "y"),
            "find_duplicates_for('x') must include y"
        );
        assert!(
            !pairs.iter().any(|p| p.doc_b == "z"),
            "find_duplicates_for('x') must not include z"
        );
    }

    // ── 28. add and remove document ──────────────────────────────────────────

    #[test]
    fn test_minhash_add_remove_document() {
        let mut det = MinHashNearDupDetector::new(minhash_config());
        det.add_document("doc1", &["a", "b"]);
        assert_eq!(det.document_count(), 1);
        let removed = det.remove_document("doc1");
        assert!(removed);
        assert_eq!(det.document_count(), 0);
    }

    // ── 29. remove nonexistent returns false ─────────────────────────────────

    #[test]
    fn test_minhash_remove_nonexistent() {
        let mut det = MinHashNearDupDetector::new(minhash_config());
        assert!(!det.remove_document("no_such_doc"));
    }

    // ── 30. deterministic with same seed ─────────────────────────────────────

    #[test]
    fn test_minhash_deterministic_same_seed() {
        let config = MinHashConfig {
            seed: 12345,
            ..minhash_config()
        };
        let det1 = MinHashNearDupDetector::new(config.clone());
        let det2 = MinHashNearDupDetector::new(config);
        let tokens = vec!["hello", "world"];
        let sig1 = det1.compute_signature("d", &tokens);
        let sig2 = det2.compute_signature("d", &tokens);
        assert_eq!(
            sig1.hashes, sig2.hashes,
            "same seed must produce identical signatures"
        );
    }

    // ── 31. different seeds produce different signatures ─────────────────────

    #[test]
    fn test_minhash_different_seeds_different_signatures() {
        let det1 = MinHashNearDupDetector::new(MinHashConfig {
            seed: 100,
            ..minhash_config()
        });
        let det2 = MinHashNearDupDetector::new(MinHashConfig {
            seed: 200,
            ..minhash_config()
        });
        let tokens = vec!["hello", "world"];
        let sig1 = det1.compute_signature("d", &tokens);
        let sig2 = det2.compute_signature("d", &tokens);
        assert_ne!(
            sig1.hashes, sig2.hashes,
            "different seeds must produce different signatures"
        );
    }

    // ── 32. empty tokens produce u64::MAX for all hashes ─────────────────────

    #[test]
    fn test_minhash_empty_tokens() {
        let det = MinHashNearDupDetector::new(minhash_config());
        let sig = det.compute_signature("empty", &[]);
        assert_eq!(sig.hashes.len(), 128);
        for &h in &sig.hashes {
            assert_eq!(h, u64::MAX, "empty tokens must yield u64::MAX");
        }
    }

    // ── 33. single token ─────────────────────────────────────────────────────

    #[test]
    fn test_minhash_single_token() {
        let det = MinHashNearDupDetector::new(minhash_config());
        let sig = det.compute_signature("one", &["only"]);
        assert_eq!(sig.hashes.len(), 128);
        // Each hash should be the single hash value (not u64::MAX unless collision)
        for &h in &sig.hashes {
            assert_ne!(h, u64::MAX, "single token hash must not be u64::MAX");
        }
    }

    // ── 34. stats returns correct values ─────────────────────────────────────

    #[test]
    fn test_minhash_stats() {
        let mut det = MinHashNearDupDetector::new(MinHashConfig {
            num_hashes: 64,
            ..minhash_config()
        });
        det.add_document("a", &["x"]);
        det.add_document("b", &["y"]);
        let _ = det.find_duplicates();
        let _ = det.find_duplicates();

        let s = det.stats();
        assert_eq!(s.document_count, 2);
        assert_eq!(s.num_hashes, 64);
        assert_eq!(s.detections_run, 2);
    }

    // ── 35. jaccard_similarity static method ─────────────────────────────────

    #[test]
    fn test_minhash_jaccard_similarity_static() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 2, 3, 4, 5];
        assert!((MinHashNearDupDetector::jaccard_similarity(&a, &b) - 1.0).abs() < 1e-10);

        let c = vec![6, 7, 8, 9, 10];
        assert!((MinHashNearDupDetector::jaccard_similarity(&a, &c)).abs() < 1e-10);
    }

    // ── 36. jaccard_similarity partial match ─────────────────────────────────

    #[test]
    fn test_minhash_jaccard_partial_match() {
        let a = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let b = vec![1, 2, 3, 4, 5, 11, 12, 13, 14, 15];
        let sim = MinHashNearDupDetector::jaccard_similarity(&a, &b);
        assert!(
            (sim - 0.5).abs() < 1e-10,
            "5/10 matches should give 0.5, got {sim}"
        );
    }

    // ── 37. jaccard_similarity empty slices ──────────────────────────────────

    #[test]
    fn test_minhash_jaccard_empty() {
        assert_eq!(
            MinHashNearDupDetector::jaccard_similarity(&[], &[1, 2]),
            0.0
        );
        assert_eq!(
            MinHashNearDupDetector::jaccard_similarity(&[1, 2], &[]),
            0.0
        );
        assert_eq!(MinHashNearDupDetector::jaccard_similarity(&[], &[]), 0.0);
    }

    // ── 38. estimate_similarity returns None for missing doc ─────────────────

    #[test]
    fn test_minhash_estimate_similarity_missing_doc() {
        let det = MinHashNearDupDetector::new(minhash_config());
        assert!(det.estimate_similarity("x", "y").is_none());
    }

    // ── 39. find_duplicates_for nonexistent doc returns empty ─────────────────

    #[test]
    fn test_minhash_find_duplicates_for_missing() {
        let mut det = MinHashNearDupDetector::new(minhash_config());
        det.add_document("a", &["x"]);
        let pairs = det.find_duplicates_for("nonexistent");
        assert!(pairs.is_empty());
    }

    // ── 40. document_count tracks correctly ──────────────────────────────────

    #[test]
    fn test_minhash_document_count() {
        let mut det = MinHashNearDupDetector::new(minhash_config());
        assert_eq!(det.document_count(), 0);
        det.add_document("a", &["x"]);
        assert_eq!(det.document_count(), 1);
        det.add_document("b", &["y"]);
        assert_eq!(det.document_count(), 2);
        det.remove_document("a");
        assert_eq!(det.document_count(), 1);
    }

    // ── 41. compute_signature preserves doc_id ───────────────────────────────

    #[test]
    fn test_minhash_signature_preserves_doc_id() {
        let det = MinHashNearDupDetector::new(minhash_config());
        let sig = det.compute_signature("my-doc-42", &["a", "b"]);
        assert_eq!(sig.doc_id, "my-doc-42");
    }

    // ── 42. signature length matches num_hashes ──────────────────────────────

    #[test]
    fn test_minhash_signature_length() {
        let det = MinHashNearDupDetector::new(MinHashConfig {
            num_hashes: 256,
            ..minhash_config()
        });
        let sig = det.compute_signature("d", &["token"]);
        assert_eq!(sig.hashes.len(), 256);
    }

    // ── 43. find_duplicates increments detections_run ─────────────────────────

    #[test]
    fn test_minhash_detections_run_increments() {
        let mut det = MinHashNearDupDetector::new(minhash_config());
        assert_eq!(det.stats().detections_run, 0);
        let _ = det.find_duplicates();
        assert_eq!(det.stats().detections_run, 1);
        let _ = det.find_duplicates_for("x");
        assert_eq!(det.stats().detections_run, 2);
    }

    // ── 44. find_duplicates sorted by similarity descending ──────────────────

    #[test]
    fn test_minhash_find_duplicates_sorted() {
        let mut det = MinHashNearDupDetector::new(MinHashConfig {
            num_hashes: 512,
            similarity_threshold: 0.0,
            ..minhash_config()
        });
        det.add_document("a", &["x", "y", "z"]);
        det.add_document("b", &["x", "y", "z"]); // identical to a
        det.add_document("c", &["x", "p", "q"]); // partial overlap

        let dups = det.find_duplicates();
        for window in dups.windows(2) {
            assert!(
                window[0].estimated_similarity >= window[1].estimated_similarity,
                "results must be sorted descending by similarity"
            );
        }
    }

    // ── 45. adding doc with same id overwrites ───────────────────────────────

    #[test]
    fn test_minhash_overwrite_document() {
        let mut det = MinHashNearDupDetector::new(minhash_config());
        det.add_document("doc", &["old", "tokens"]);
        det.add_document("doc", &["new", "tokens"]);
        assert_eq!(det.document_count(), 1);
        let sig = det
            .signatures
            .get("doc")
            .expect("doc must exist after overwrite");
        // Verify it uses the new tokens by computing fresh
        let fresh = det.compute_signature("doc", &["new", "tokens"]);
        assert_eq!(sig.hashes, fresh.hashes);
    }

    // ── 46. many documents find_duplicates scales ────────────────────────────

    #[test]
    fn test_minhash_many_documents() {
        let mut det = MinHashNearDupDetector::new(MinHashConfig {
            num_hashes: 64,
            similarity_threshold: 0.9,
            ..minhash_config()
        });
        // Add 20 unique documents
        for i in 0..20 {
            det.add_document(
                &format!("doc_{i}"),
                &[&format!("unique_token_{i}"), "shared"],
            );
        }
        // Add a duplicate of doc_0
        det.add_document("doc_0_dup", &["unique_token_0", "shared"]);

        let dups = det.find_duplicates();
        assert!(
            dups.iter().any(|d| {
                (d.doc_a.contains("doc_0") && d.doc_b.contains("doc_0"))
                    || (d.doc_b.contains("doc_0") && d.doc_a.contains("doc_0"))
            }),
            "duplicate of doc_0 must be detected"
        );
    }

    // ── 47. token order does not affect signature ────────────────────────────

    #[test]
    fn test_minhash_token_order_invariant() {
        let det = MinHashNearDupDetector::new(minhash_config());
        let sig1 = det.compute_signature("d1", &["a", "b", "c"]);
        let sig2 = det.compute_signature("d2", &["c", "a", "b"]);
        assert_eq!(
            sig1.hashes, sig2.hashes,
            "token order must not affect MinHash signature"
        );
    }
}
