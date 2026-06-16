//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::constants::{BM25_B, BM25_K1};
use super::functions::{cosine_similarity, tokenize};

/// Raw content payload for a single modality slot in an [`MmiIndexedDocument`].
#[derive(Debug, Clone, PartialEq)]
pub enum ModalityData {
    /// Natural-language text.
    Text(String),
    /// Dense float vector (for approximate nearest-neighbour search).
    Vector(Vec<f64>),
    /// Structured key–value pairs (exact-match filtering).
    Structured(Vec<(String, String)>),
    /// Opaque binary blob (stored but not indexed for search).
    Binary(Vec<u8>),
    /// Scalar numeric value.
    Numeric(f64),
}
/// Inverted text index maintained by [`MultiModalIndexer`].
#[derive(Debug, Default)]
pub(super) struct TextIndex {
    /// Posting lists: term → list of doc_ids containing that term.
    pub(super) postings: HashMap<String, Vec<String>>,
    /// Per-document term statistics.
    pub(super) doc_stats: HashMap<String, TextDocStats>,
}
impl TextIndex {
    /// Add (or replace) a document's text content in the index.
    pub(super) fn add_or_replace(&mut self, doc_id: &str, text: &str) {
        self.remove(doc_id);
        let tokens = tokenize(text);
        let term_count = tokens.len();
        let mut term_freqs: HashMap<String, usize> = HashMap::new();
        for token in &tokens {
            *term_freqs.entry(token.clone()).or_insert(0) += 1;
        }
        for term in term_freqs.keys() {
            self.postings
                .entry(term.clone())
                .or_default()
                .push(doc_id.to_string());
        }
        self.doc_stats.insert(
            doc_id.to_string(),
            TextDocStats {
                term_count,
                term_freqs,
            },
        );
    }
    /// Remove a document from the text index.
    pub(super) fn remove(&mut self, doc_id: &str) {
        if let Some(stats) = self.doc_stats.remove(doc_id) {
            for term in stats.term_freqs.keys() {
                if let Some(list) = self.postings.get_mut(term) {
                    list.retain(|id| id != doc_id);
                }
            }
        }
    }
    /// Compute BM25 scores for `query_terms` against all indexed documents.
    ///
    /// Returns `HashMap<doc_id, score>` for documents that match at least one term.
    pub(super) fn bm25_scores(&self, query_terms: &[String]) -> HashMap<String, f64> {
        let n = self.doc_stats.len() as f64;
        if n == 0.0 || query_terms.is_empty() {
            return HashMap::new();
        }
        let avg_dl: f64 = self
            .doc_stats
            .values()
            .map(|s| s.term_count as f64)
            .sum::<f64>()
            / n;
        let mut scores: HashMap<String, f64> = HashMap::new();
        for term in query_terms {
            let Some(posting) = self.postings.get(term) else {
                continue;
            };
            let df = posting.len() as f64;
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for doc_id in posting {
                let Some(stats) = self.doc_stats.get(doc_id) else {
                    continue;
                };
                let tf = *stats.term_freqs.get(term).unwrap_or(&0) as f64;
                let dl = stats.term_count as f64;
                let numerator = tf * (BM25_K1 + 1.0);
                let denominator = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avg_dl);
                let contribution = idf * numerator / denominator;
                *scores.entry(doc_id.clone()).or_insert(0.0) += contribution;
            }
        }
        scores
    }
}
/// A document that may contain several named modality payloads.
#[derive(Debug, Clone, PartialEq)]
pub struct MmiIndexedDocument {
    /// Globally unique identifier.
    pub id: String,
    /// Named modality slots: `(modality_name, payload)`.
    pub modalities: Vec<(String, ModalityData)>,
    /// Arbitrary string metadata (not used for scoring but returned with results).
    pub metadata: Vec<(String, String)>,
    /// Unix-epoch seconds at which this document was indexed.
    pub indexed_at: u64,
    /// Monotonically increasing version counter (starts at 1, incremented on update).
    pub version: u32,
}
impl MmiIndexedDocument {
    /// Construct a new document with `version = 1` and an arbitrary timestamp.
    pub fn new(
        id: impl Into<String>,
        modalities: Vec<(String, ModalityData)>,
        metadata: Vec<(String, String)>,
        indexed_at: u64,
    ) -> Self {
        Self {
            id: id.into(),
            modalities,
            metadata,
            indexed_at,
            version: 1,
        }
    }
}
/// A single search result with a combined score and per-modality breakdown.
#[derive(Debug, Clone, PartialEq)]
pub struct MmiSearchResult {
    /// Document identifier.
    pub doc_id: String,
    /// Combined (weighted) relevance score.
    pub score: f64,
    /// Per-modality score contributions: `(modality_label, raw_score)`.
    pub score_breakdown: Vec<(String, f64)>,
    /// Names of the modalities that contributed to this result.
    pub matched_modalities: Vec<String>,
}
/// Structured (exact-match) inverted index.
#[derive(Debug, Default)]
pub(super) struct StructuredIndex {
    /// Posting lists: `(field, value)` → list of `doc_id`.
    pub(super) postings: HashMap<(String, String), Vec<String>>,
    /// Per-document set of `(field, value)` pairs.
    pub(super) doc_pairs: HashMap<String, Vec<(String, String)>>,
}
impl StructuredIndex {
    pub(super) fn add_or_replace(&mut self, doc_id: &str, pairs: &[(String, String)]) {
        self.remove(doc_id);
        for (k, v) in pairs {
            self.postings
                .entry((k.clone(), v.clone()))
                .or_default()
                .push(doc_id.to_string());
        }
        self.doc_pairs.insert(doc_id.to_string(), pairs.to_vec());
    }
    pub(super) fn remove(&mut self, doc_id: &str) {
        if let Some(pairs) = self.doc_pairs.remove(doc_id) {
            for (k, v) in &pairs {
                if let Some(list) = self.postings.get_mut(&(k.clone(), v.clone())) {
                    list.retain(|id| id != doc_id);
                }
            }
        }
    }
    /// Return the set of `doc_id`s that match **all** supplied `filters`.
    ///
    /// Returns `None` when `filters` is empty (caller decides semantics).
    pub(super) fn matching_docs(&self, filters: &[(String, String)]) -> Option<Vec<String>> {
        if filters.is_empty() {
            return None;
        }
        let mut candidate_set: Option<std::collections::HashSet<String>> = None;
        for (k, v) in filters {
            let posting = self
                .postings
                .get(&(k.clone(), v.clone()))
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            let current: std::collections::HashSet<String> = posting.iter().cloned().collect();
            candidate_set = Some(match candidate_set {
                None => current,
                Some(prev) => prev.intersection(&current).cloned().collect(),
            });
        }
        candidate_set.map(|s| s.into_iter().collect())
    }
}
/// Configuration for [`MultiModalIndexer`].
#[derive(Debug, Clone)]
pub struct MmiIndexConfig {
    /// Enable the inverted text index.
    pub enable_text_index: bool,
    /// Enable the brute-force vector index.
    pub enable_vector_index: bool,
    /// Enable the exact-match structured index.
    pub enable_structured_index: bool,
    /// Expected vector dimensionality (validated on insert when `Some`).
    pub vector_dim: Option<usize>,
    /// Minimum BM25 score for a text result to be included before blending.
    pub text_similarity_threshold: f64,
    /// Minimum cosine similarity for a vector result to be included before blending.
    pub vector_similarity_threshold: f64,
    /// Hard upper limit on the number of stored documents.
    pub max_documents: usize,
}
/// Per-document statistics required for BM25 scoring.
#[derive(Debug, Clone)]
pub(super) struct TextDocStats {
    /// Total term count for this document.
    pub(super) term_count: usize,
    /// Term frequency map: term → count.
    pub(super) term_freqs: HashMap<String, usize>,
}
/// Per-document vector entry.
#[derive(Debug, Clone)]
pub(super) struct VectorEntry {
    pub(super) modality_name: String,
    pub(super) vector: Vec<f64>,
}
/// Brute-force dense vector index.
#[derive(Debug, Default)]
pub(super) struct VectorIndex {
    pub(super) entries: HashMap<String, Vec<VectorEntry>>,
}
impl VectorIndex {
    pub(super) fn add_or_replace(&mut self, doc_id: &str, modality_name: &str, vector: Vec<f64>) {
        let slot = self.entries.entry(doc_id.to_string()).or_default();
        for entry in slot.iter_mut() {
            if entry.modality_name == modality_name {
                entry.vector = vector;
                return;
            }
        }
        slot.push(VectorEntry {
            modality_name: modality_name.to_string(),
            vector,
        });
    }
    pub(super) fn remove(&mut self, doc_id: &str) {
        self.entries.remove(doc_id);
    }
    pub(super) fn remove_modality(&mut self, doc_id: &str, modality_name: &str) {
        if let Some(slot) = self.entries.get_mut(doc_id) {
            slot.retain(|e| e.modality_name != modality_name);
        }
    }
    /// Compute the maximum cosine similarity between `query` and any vector stored
    /// for `doc_id`.
    pub(super) fn best_cosine(&self, doc_id: &str, query: &[f64]) -> f64 {
        self.entries
            .get(doc_id)
            .map(|slot| {
                slot.iter()
                    .map(|e| cosine_similarity(&e.vector, query))
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .unwrap_or(0.0)
    }
    /// Returns `true` if any vector is stored for `doc_id`.
    #[allow(dead_code)]
    pub(super) fn has_doc(&self, doc_id: &str) -> bool {
        self.entries.contains_key(doc_id)
    }
    pub(super) fn all_doc_ids(&self) -> impl Iterator<Item = &String> {
        self.entries.keys()
    }
}
/// Unified multi-modal content indexer.
///
/// Manages three sub-indexes (text / vector / structured) and provides
/// combined BM25 + cosine + exact-match search with configurable weight blending.
#[derive(Debug)]
pub struct MultiModalIndexer {
    pub(super) config: MmiIndexConfig,
    pub(super) documents: HashMap<String, MmiIndexedDocument>,
    pub(super) text_index: TextIndex,
    pub(super) vector_index: VectorIndex,
    pub(super) structured_index: StructuredIndex,
    pub(super) search_count: u64,
}
impl MultiModalIndexer {
    /// Create a new indexer with the supplied configuration.
    pub fn new(config: MmiIndexConfig) -> Self {
        Self {
            config,
            documents: HashMap::new(),
            text_index: TextIndex::default(),
            vector_index: VectorIndex::default(),
            structured_index: StructuredIndex::default(),
            search_count: 0,
        }
    }
    /// Create an indexer with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(MmiIndexConfig::default())
    }
    /// Validate a vector's dimensionality against `config.vector_dim`.
    pub(super) fn validate_vector_dim(&self, v: &[f64]) -> Result<(), MmiIndexError> {
        if let Some(expected) = self.config.vector_dim {
            if v.len() != expected {
                return Err(MmiIndexError::DimensionMismatch {
                    expected,
                    got: v.len(),
                });
            }
        }
        Ok(())
    }
    /// Add (or update) a single modality slot across all applicable sub-indexes.
    pub(super) fn index_modality(
        &mut self,
        doc_id: &str,
        modality_name: &str,
        data: &ModalityData,
    ) -> Result<(), MmiIndexError> {
        match data {
            ModalityData::Text(text) => {
                if self.config.enable_text_index {
                    self.text_index.add_or_replace(doc_id, text);
                }
            }
            ModalityData::Vector(v) => {
                self.validate_vector_dim(v)?;
                if self.config.enable_vector_index {
                    self.vector_index
                        .add_or_replace(doc_id, modality_name, v.clone());
                }
            }
            ModalityData::Structured(pairs) => {
                if self.config.enable_structured_index {
                    self.structured_index.add_or_replace(doc_id, pairs);
                }
            }
            ModalityData::Numeric(n) => {
                if self.config.enable_structured_index {
                    let pairs = vec![(modality_name.to_string(), n.to_string())];
                    self.structured_index.add_or_replace(doc_id, &pairs);
                }
            }
            ModalityData::Binary(_) => {}
        }
        Ok(())
    }
    /// Remove all sub-index entries for `doc_id`.
    pub(super) fn deindex_document(&mut self, doc_id: &str) {
        self.text_index.remove(doc_id);
        self.vector_index.remove(doc_id);
        self.structured_index.remove(doc_id);
    }
    /// Index a new document, or update an existing one.
    ///
    /// If a document with the same `id` already exists, its `version` is
    /// incremented and all sub-index entries are refreshed.
    pub fn index_document(&mut self, mut doc: MmiIndexedDocument) -> Result<(), MmiIndexError> {
        let is_new = !self.documents.contains_key(&doc.id);
        if is_new && self.documents.len() >= self.config.max_documents {
            return Err(MmiIndexError::MaxDocumentsExceeded);
        }
        if let Some(existing) = self.documents.get(&doc.id) {
            doc.version = existing.version + 1;
            self.deindex_document(&doc.id);
        }
        let modalities = doc.modalities.clone();
        for (name, data) in &modalities {
            self.index_modality(&doc.id, name, data)?;
        }
        self.documents.insert(doc.id.clone(), doc);
        Ok(())
    }
    /// Remove a document from the index.
    ///
    /// Returns `Err(DocumentNotFound)` if the document does not exist.
    pub fn remove_document(&mut self, id: &str) -> Result<(), MmiIndexError> {
        if !self.documents.contains_key(id) {
            return Err(MmiIndexError::DocumentNotFound(id.to_string()));
        }
        self.deindex_document(id);
        self.documents.remove(id);
        Ok(())
    }
    /// Retrieve an immutable reference to a stored document.
    pub fn get_document(&self, id: &str) -> Result<&MmiIndexedDocument, MmiIndexError> {
        self.documents
            .get(id)
            .ok_or_else(|| MmiIndexError::DocumentNotFound(id.to_string()))
    }
    /// Return references to all documents that contain the named modality.
    pub fn documents_with_modality(&self, modality_name: &str) -> Vec<&MmiIndexedDocument> {
        self.documents
            .values()
            .filter(|doc| doc.modalities.iter().any(|(name, _)| name == modality_name))
            .collect()
    }
    /// Update (replace) a single modality slot for an existing document.
    ///
    /// The document's `version` is incremented and the sub-index is refreshed
    /// for the affected modality.
    pub fn update_modality(
        &mut self,
        doc_id: &str,
        modality_name: String,
        data: ModalityData,
    ) -> Result<(), MmiIndexError> {
        if let ModalityData::Vector(ref v) = data {
            self.validate_vector_dim(v)?;
        }
        let doc = self
            .documents
            .get_mut(doc_id)
            .ok_or_else(|| MmiIndexError::DocumentNotFound(doc_id.to_string()))?;
        doc.version += 1;
        let mut replaced = false;
        for (name, payload) in doc.modalities.iter_mut() {
            if *name == modality_name {
                *payload = data.clone();
                replaced = true;
                break;
            }
        }
        if !replaced {
            doc.modalities.push((modality_name.clone(), data.clone()));
        }
        match &data {
            ModalityData::Text(_text) => {
                if self.config.enable_text_index {
                    self.text_index.remove(doc_id);
                    let doc_ref = self.documents.get(doc_id).expect("just updated");
                    let combined: String = doc_ref
                        .modalities
                        .iter()
                        .filter_map(|(_, d)| {
                            if let ModalityData::Text(t) = d {
                                Some(t.as_str())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    self.text_index.add_or_replace(doc_id, &combined);
                }
            }
            ModalityData::Vector(v) => {
                if self.config.enable_vector_index {
                    self.vector_index
                        .add_or_replace(doc_id, &modality_name, v.clone());
                }
            }
            ModalityData::Structured(_pairs) => {
                if self.config.enable_structured_index {
                    self.structured_index.remove(doc_id);
                    let doc_ref = self.documents.get(doc_id).expect("just updated");
                    let all_pairs: Vec<(String, String)> = doc_ref
                        .modalities
                        .iter()
                        .flat_map(|(_, d)| {
                            if let ModalityData::Structured(p) = d {
                                p.clone()
                            } else {
                                vec![]
                            }
                        })
                        .collect();
                    self.structured_index.add_or_replace(doc_id, &all_pairs);
                }
            }
            ModalityData::Numeric(n) => {
                if self.config.enable_structured_index {
                    self.structured_index.remove(doc_id);
                    let doc_ref = self.documents.get(doc_id).expect("just updated");
                    let mut all_pairs: Vec<(String, String)> = Vec::new();
                    for (name, d) in &doc_ref.modalities {
                        match d {
                            ModalityData::Structured(p) => all_pairs.extend(p.clone()),
                            ModalityData::Numeric(v) => {
                                all_pairs.push((name.clone(), v.to_string()))
                            }
                            _ => {}
                        }
                    }
                    self.structured_index.add_or_replace(doc_id, &all_pairs);
                }
                let _ = n;
            }
            ModalityData::Binary(_) => {
                self.vector_index.remove_modality(doc_id, &modality_name);
            }
        }
        Ok(())
    }
    /// Execute a multi-modal search query.
    ///
    /// The pipeline:
    ///
    /// 1. **Text** — BM25 scored per-document.
    /// 2. **Vector** — maximum cosine similarity per-document.
    /// 3. **Structured** — exact-match AND filter; matching docs receive score 1.0.
    /// 4. **Required modalities** — documents not containing *all* listed modalities
    ///    are pruned.
    /// 5. **Combine** — weighted blend using only the active signal types
    ///    (renormalised weights sum to 1.0).
    /// 6. **Filter + rank** — discard below `min_score`, sort descending, take `top_k`.
    pub fn search(
        &mut self,
        query: &MmiSearchQuery,
    ) -> Result<Vec<MmiSearchResult>, MmiIndexError> {
        self.search_count += 1;
        if self.documents.is_empty() {
            return Ok(vec![]);
        }
        let top_k = if query.top_k == 0 { 10 } else { query.top_k };
        let text_scores: HashMap<String, f64> = if let Some(ref q) = query.text_query {
            let terms = tokenize(q);
            let raw = self.text_index.bm25_scores(&terms);
            raw.into_iter()
                .filter(|(_, s)| *s >= self.config.text_similarity_threshold)
                .collect()
        } else {
            HashMap::new()
        };
        let vector_scores: HashMap<String, f64> = if let Some(ref qv) = query.vector_query {
            self.validate_vector_dim(qv)?;
            self.vector_index
                .all_doc_ids()
                .cloned()
                .collect::<Vec<_>>()
                .iter()
                .filter_map(|id| {
                    let sim = self.vector_index.best_cosine(id, qv);
                    if sim >= self.config.vector_similarity_threshold {
                        Some((id.clone(), sim))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            HashMap::new()
        };
        let structured_set: Option<std::collections::HashSet<String>> =
            if !query.structured_filters.is_empty() && self.config.enable_structured_index {
                let matched = self
                    .structured_index
                    .matching_docs(&query.structured_filters)
                    .unwrap_or_default();
                Some(matched.into_iter().collect())
            } else {
                None
            };
        let mut candidate_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        let has_text_query = query.text_query.is_some();
        let has_vector_query = query.vector_query.is_some();
        let has_struct_filter = !query.structured_filters.is_empty();
        if has_text_query {
            candidate_ids.extend(text_scores.keys().cloned());
        }
        if has_vector_query {
            candidate_ids.extend(vector_scores.keys().cloned());
        }
        if let Some(ref set) = structured_set {
            if has_struct_filter && !has_text_query && !has_vector_query {
                candidate_ids.extend(set.iter().cloned());
            }
        }
        if !has_text_query && !has_vector_query && !has_struct_filter {
            return Ok(vec![]);
        }
        if let Some(ref set) = structured_set {
            candidate_ids.retain(|id| set.contains(id));
        }
        if !query.modalities_required.is_empty() {
            candidate_ids.retain(|id| {
                if let Some(doc) = self.documents.get(id) {
                    query
                        .modalities_required
                        .iter()
                        .all(|req| doc.modalities.iter().any(|(name, _)| name == req))
                } else {
                    false
                }
            });
        }
        candidate_ids.retain(|id| self.documents.contains_key(id));
        let text_raw_w: f64 = if has_text_query { 0.4 } else { 0.0 };
        let vec_raw_w: f64 = if has_vector_query { 0.4 } else { 0.0 };
        let str_raw_w: f64 = if has_struct_filter { 0.2 } else { 0.0 };
        let total_w = text_raw_w + vec_raw_w + str_raw_w;
        let (text_w, vec_w, str_w) = if total_w < 1e-12 {
            (0.0, 0.0, 0.0)
        } else {
            (
                text_raw_w / total_w,
                vec_raw_w / total_w,
                str_raw_w / total_w,
            )
        };
        let max_bm25: f64 = text_scores.values().cloned().fold(0.0_f64, f64::max);
        let mut results: Vec<MmiSearchResult> = candidate_ids
            .iter()
            .filter_map(|id| {
                let t_raw = text_scores.get(id).cloned().unwrap_or(0.0);
                let t_norm = if max_bm25 > 1e-12 {
                    t_raw / max_bm25
                } else {
                    t_raw
                };
                let v_score = vector_scores.get(id).cloned().unwrap_or(0.0);
                let s_score = if let Some(ref set) = structured_set {
                    if set.contains(id) {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                let combined = text_w * t_norm + vec_w * v_score + str_w * s_score;
                if combined < query.min_score {
                    return None;
                }
                let mut score_breakdown: Vec<(String, f64)> = Vec::new();
                let mut matched_modalities: Vec<String> = Vec::new();
                if has_text_query {
                    score_breakdown.push(("text".to_string(), t_norm));
                    if t_norm > 0.0 {
                        matched_modalities.push("text".to_string());
                    }
                }
                if has_vector_query {
                    score_breakdown.push(("vector".to_string(), v_score));
                    if v_score > 0.0 {
                        matched_modalities.push("vector".to_string());
                    }
                }
                if has_struct_filter {
                    score_breakdown.push(("structured".to_string(), s_score));
                    if s_score > 0.0 {
                        matched_modalities.push("structured".to_string());
                    }
                }
                Some(MmiSearchResult {
                    doc_id: id.clone(),
                    score: combined,
                    score_breakdown,
                    matched_modalities,
                })
            })
            .collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        results.truncate(top_k);
        Ok(results)
    }
    /// Return current index statistics.
    pub fn stats(&self) -> MmiIndexStats {
        let total = self.documents.len();
        let mut modality_map: HashMap<String, usize> = HashMap::new();
        let mut total_modality_slots: usize = 0;
        for doc in self.documents.values() {
            for (name, _) in &doc.modalities {
                *modality_map.entry(name.clone()).or_insert(0) += 1;
                total_modality_slots += 1;
            }
        }
        let mut modality_counts: Vec<(String, usize)> = modality_map.into_iter().collect();
        modality_counts.sort_by(|a, b| a.0.cmp(&b.0));
        let avg_modalities_per_doc = if total == 0 {
            0.0
        } else {
            total_modality_slots as f64 / total as f64
        };
        let base: u64 = self
            .documents
            .values()
            .map(|d| {
                let slot_bytes: u64 = d
                    .modalities
                    .iter()
                    .map(|(name, data)| {
                        let data_bytes: u64 = match data {
                            ModalityData::Text(t) => t.len() as u64,
                            ModalityData::Vector(v) => (v.len() * 8) as u64,
                            ModalityData::Structured(p) => {
                                p.iter().map(|(k, v)| (k.len() + v.len()) as u64).sum()
                            }
                            ModalityData::Binary(b) => b.len() as u64,
                            ModalityData::Numeric(_) => 8,
                        };
                        name.len() as u64 + data_bytes
                    })
                    .sum::<u64>();
                d.id.len() as u64 + slot_bytes
            })
            .sum();
        let posting_overhead: u64 = self
            .text_index
            .postings
            .values()
            .map(|v| (v.len() * 24) as u64)
            .sum::<u64>()
            + self
                .structured_index
                .postings
                .values()
                .map(|v| (v.len() * 24) as u64)
                .sum::<u64>();
        MmiIndexStats {
            total_documents: total,
            modality_counts,
            avg_modalities_per_doc,
            index_size_estimate_bytes: base + posting_overhead,
            search_count: self.search_count,
        }
    }
}
/// Errors returned by [`MultiModalIndexer`] operations.
#[derive(Debug, Clone, PartialEq)]
pub enum MmiIndexError {
    /// No document with the given identifier exists.
    DocumentNotFound(String),
    /// A vector payload has a different dimension than the configured expectation.
    DimensionMismatch {
        /// The expected dimension (from `IndexConfig::vector_dim`).
        expected: usize,
        /// The actual dimension in the supplied vector.
        got: usize,
    },
    /// The index has reached `IndexConfig::max_documents`.
    MaxDocumentsExceeded,
    /// A modality name or type is not valid in the current configuration.
    InvalidModality(String),
    /// A configuration parameter is inconsistent or out of range.
    ConfigurationError(String),
}
/// A multi-modal search query.
#[derive(Debug, Clone, Default)]
pub struct MmiSearchQuery {
    /// Optional free-text query (triggers BM25 text scoring).
    pub text_query: Option<String>,
    /// Optional query vector (triggers cosine-similarity vector scoring).
    pub vector_query: Option<Vec<f64>>,
    /// Exact-match field=value filters (AND semantics across all entries).
    pub structured_filters: Vec<(String, String)>,
    /// If non-empty, only documents possessing *all* listed modality names are returned.
    pub modalities_required: Vec<String>,
    /// Maximum number of results to return.
    pub top_k: usize,
    /// Minimum combined score threshold; results below this are discarded.
    pub min_score: f64,
}
/// Runtime statistics for [`MultiModalIndexer`].
#[derive(Debug, Clone, Default)]
pub struct MmiIndexStats {
    /// Total number of documents currently in the index.
    pub total_documents: usize,
    /// How many documents contain each named modality: `(modality_name, count)`.
    pub modality_counts: Vec<(String, usize)>,
    /// Average number of modality slots per document.
    pub avg_modalities_per_doc: f64,
    /// Rough estimate of index heap usage in bytes.
    pub index_size_estimate_bytes: u64,
    /// Total number of searches performed since creation.
    pub search_count: u64,
}
