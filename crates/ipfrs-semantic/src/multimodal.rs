//! Multi-modal embedding support for unified semantic search across text, image, and audio.
//!
//! This module provides infrastructure for:
//! - Unified embedding space across modalities
//! - Cross-modal similarity search
//! - Modality-specific distance metrics
//! - Embedding projection and alignment

use crate::{DistanceMetric, VectorIndex};
use ipfrs_core::{Cid, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported modalities for embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Text embeddings (e.g., from BERT, GPT)
    Text,
    /// Image embeddings (e.g., from ResNet, CLIP)
    Image,
    /// Audio embeddings (e.g., from Wav2Vec, CLAP)
    Audio,
    /// Video embeddings (e.g., from VideoMAE)
    Video,
    /// Code embeddings (e.g., from CodeBERT)
    Code,
}

impl Modality {
    /// Get the default embedding dimension for this modality
    pub fn default_dim(&self) -> usize {
        match self {
            Modality::Text => 768,  // BERT-base
            Modality::Image => 512, // ResNet-50
            Modality::Audio => 768, // Wav2Vec 2.0
            Modality::Video => 768, // VideoMAE
            Modality::Code => 768,  // CodeBERT
        }
    }

    /// Get the recommended distance metric for this modality
    pub fn default_metric(&self) -> DistanceMetric {
        match self {
            Modality::Text => DistanceMetric::Cosine,
            Modality::Image => DistanceMetric::L2,
            Modality::Audio => DistanceMetric::Cosine,
            Modality::Video => DistanceMetric::L2,
            Modality::Code => DistanceMetric::Cosine,
        }
    }
}

/// Multi-modal embedding with modality information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalEmbedding {
    /// The embedding vector
    pub vector: Vec<f32>,
    /// The modality this embedding belongs to
    pub modality: Modality,
    /// Optional metadata about the embedding source
    pub metadata: HashMap<String, String>,
}

impl MultiModalEmbedding {
    /// Create a new multi-modal embedding
    pub fn new(vector: Vec<f32>, modality: Modality) -> Self {
        Self {
            vector,
            modality,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the embedding
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get the dimension of the embedding
    pub fn dim(&self) -> usize {
        self.vector.len()
    }
}

/// Configuration for multi-modal index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalConfig {
    /// Target dimension for unified embedding space
    pub unified_dim: usize,
    /// Whether to project embeddings to unified dimension
    pub project_to_unified: bool,
    /// Modality-specific weights for cross-modal search
    pub modality_weights: HashMap<Modality, f32>,
}

impl Default for MultiModalConfig {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert(Modality::Text, 1.0);
        weights.insert(Modality::Image, 1.0);
        weights.insert(Modality::Audio, 1.0);
        weights.insert(Modality::Video, 1.0);
        weights.insert(Modality::Code, 1.0);

        Self {
            unified_dim: 768,
            project_to_unified: false,
            modality_weights: weights,
        }
    }
}

/// Multi-modal index for unified semantic search
pub struct MultiModalIndex {
    /// Separate indices for each modality
    indices: HashMap<Modality, VectorIndex>,
    /// Configuration
    config: MultiModalConfig,
    /// Projection matrices for each modality (if using unified embedding space)
    projections: HashMap<Modality, Vec<Vec<f32>>>,
}

impl MultiModalIndex {
    /// Create a new multi-modal index
    pub fn new(config: MultiModalConfig) -> Self {
        Self {
            indices: HashMap::new(),
            config,
            projections: HashMap::new(),
        }
    }

    /// Register a modality with the index
    pub fn register_modality(&mut self, modality: Modality, dim: usize) -> Result<()> {
        let metric = modality.default_metric();

        // If projection is enabled, create index with unified dimension
        // Otherwise, use the modality's dimension
        let index_dim = if self.config.project_to_unified {
            self.config.unified_dim
        } else {
            dim
        };

        let index = VectorIndex::new(index_dim, metric, 16, 200)?;
        self.indices.insert(modality, index);

        // Initialize projection matrix if needed
        if self.config.project_to_unified && dim != self.config.unified_dim {
            self.init_projection(modality, dim)?;
        }

        Ok(())
    }

    /// Initialize random projection matrix for dimensionality reduction/expansion
    fn init_projection(&mut self, modality: Modality, from_dim: usize) -> Result<()> {
        let to_dim = self.config.unified_dim;

        // Use random projection (Johnson-Lindenstrauss lemma)
        // Each element ~ N(0, 1/to_dim)
        let mut projection = Vec::with_capacity(from_dim);

        use rand::RngExt;
        let mut rng = rand::rng();
        let scale = (1.0 / to_dim as f32).sqrt();

        for _ in 0..from_dim {
            let mut row = Vec::with_capacity(to_dim);
            for _ in 0..to_dim {
                // Sample from standard normal, then scale
                let val: f32 = rng.random_range(-1.0..1.0);
                row.push(val * scale);
            }
            projection.push(row);
        }

        self.projections.insert(modality, projection);
        Ok(())
    }

    /// Project embedding to unified dimension
    fn project_embedding(&self, embedding: &[f32], modality: Modality) -> Vec<f32> {
        if !self.config.project_to_unified {
            return embedding.to_vec();
        }

        if let Some(projection) = self.projections.get(&modality) {
            let mut result = vec![0.0; self.config.unified_dim];

            for (i, row) in projection.iter().enumerate() {
                if i >= embedding.len() {
                    break;
                }
                for (j, &proj_val) in row.iter().enumerate() {
                    result[j] += embedding[i] * proj_val;
                }
            }

            // Normalize
            let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut result {
                    *val /= norm;
                }
            }

            result
        } else {
            embedding.to_vec()
        }
    }

    /// Add an embedding to the index
    pub fn add(&mut self, cid: Cid, embedding: MultiModalEmbedding) -> Result<()> {
        // Project embedding first to avoid borrowing issues
        let projected = self.project_embedding(&embedding.vector, embedding.modality);

        let index = self.indices.get_mut(&embedding.modality).ok_or_else(|| {
            Error::InvalidInput(format!("Modality {:?} not registered", embedding.modality))
        })?;

        index.insert(&cid, &projected)?;

        Ok(())
    }

    /// Search within a specific modality
    pub fn search_modality(
        &self,
        query: &MultiModalEmbedding,
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<(Cid, f32)>> {
        let index = self.indices.get(&query.modality).ok_or_else(|| {
            Error::InvalidInput(format!("Modality {:?} not registered", query.modality))
        })?;

        let projected = self.project_embedding(&query.vector, query.modality);
        let ef_search = ef_search.unwrap_or(50);

        let results = index.search(&projected, k, ef_search)?;
        Ok(results.into_iter().map(|r| (r.cid, r.score)).collect())
    }

    /// Cross-modal search: search across all modalities
    pub fn search_cross_modal(
        &self,
        query: &MultiModalEmbedding,
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<(Cid, f32, Modality)>> {
        let mut all_results = Vec::new();
        let projected_query = self.project_embedding(&query.vector, query.modality);
        let ef_search = ef_search.unwrap_or(50);

        // Search each modality
        for (modality, index) in &self.indices {
            let weight = self
                .config
                .modality_weights
                .get(modality)
                .copied()
                .unwrap_or(1.0);

            match index.search(&projected_query, k * 2, ef_search) {
                Ok(results) => {
                    for result in results {
                        // Apply modality weight to score
                        let weighted_score = result.score * weight;
                        all_results.push((result.cid, weighted_score, *modality));
                    }
                }
                Err(_) => continue,
            }
        }

        // Sort by score and take top k
        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(k);

        Ok(all_results)
    }

    /// Get statistics for each modality
    pub fn stats(&self) -> HashMap<Modality, ModalityStats> {
        let mut stats = HashMap::new();

        for (modality, index) in &self.indices {
            stats.insert(
                *modality,
                ModalityStats {
                    num_embeddings: index.len(),
                    dimension: index.dimension(),
                    metric: modality.default_metric(),
                },
            );
        }

        stats
    }

    /// Get the number of embeddings in a specific modality
    pub fn len_for_modality(&self, modality: Modality) -> usize {
        self.indices
            .get(&modality)
            .map(|idx| idx.len())
            .unwrap_or(0)
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.indices.values().all(|idx| idx.is_empty())
    }

    /// Total number of embeddings across all modalities
    pub fn total_len(&self) -> usize {
        self.indices.values().map(|idx| idx.len()).sum()
    }
}

/// Statistics for a specific modality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityStats {
    /// Number of embeddings in this modality
    pub num_embeddings: usize,
    /// Dimension of embeddings
    pub dimension: usize,
    /// Distance metric used
    pub metric: DistanceMetric,
}

/// Alignment between two modalities for cross-modal search
pub struct ModalityAlignment {
    /// Source modality
    #[allow(dead_code)]
    source: Modality,
    /// Target modality
    #[allow(dead_code)]
    target: Modality,
    /// Learned transformation matrix (source_dim × target_dim)
    transform: Vec<Vec<f32>>,
}

impl ModalityAlignment {
    /// Create a new modality alignment
    pub fn new(source: Modality, target: Modality, source_dim: usize, target_dim: usize) -> Self {
        // Initialize with identity-like transformation
        let mut transform = vec![vec![0.0; target_dim]; source_dim];
        let min_dim = source_dim.min(target_dim);

        for (i, row) in transform.iter_mut().enumerate().take(min_dim) {
            row[i] = 1.0;
        }

        Self {
            source,
            target,
            transform,
        }
    }

    /// Learn alignment from paired examples
    ///
    /// This is a simplified version - in practice, you'd use CCA, GCCA, or neural networks
    pub fn learn_from_pairs(&mut self, pairs: &[(Vec<f32>, Vec<f32>)]) -> Result<()> {
        if pairs.is_empty() {
            return Err(Error::InvalidInput("No pairs provided".into()));
        }

        // Simplified learning: use average mapping
        // In practice, use Canonical Correlation Analysis (CCA) or neural networks
        let source_dim = pairs[0].0.len();
        let target_dim = pairs[0].1.len();

        let mut transform = vec![vec![0.0; target_dim]; source_dim];

        for (source_vec, target_vec) in pairs {
            for (i, &source_val) in source_vec.iter().enumerate().take(source_dim) {
                for (j, &target_val) in target_vec.iter().enumerate().take(target_dim) {
                    transform[i][j] += source_val * target_val;
                }
            }
        }

        // Normalize by number of pairs
        let n = pairs.len() as f32;
        for row in &mut transform {
            for val in row {
                *val /= n;
            }
        }

        self.transform = transform;
        Ok(())
    }

    /// Transform a source embedding to target modality space
    pub fn transform_embedding(&self, source: &[f32]) -> Vec<f32> {
        let target_dim = self.transform[0].len();
        let mut result = vec![0.0; target_dim];

        for (i, row) in self.transform.iter().enumerate() {
            if i >= source.len() {
                break;
            }
            for (j, &val) in row.iter().enumerate() {
                result[j] += source[i] * val;
            }
        }

        // Normalize
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut result {
                *val /= norm;
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_cid(index: usize) -> Cid {
        use multihash_codetable::{Code, MultihashDigest};
        let data = format!("multimodal_test_{}", index);
        let hash = Code::Sha2_256.digest(data.as_bytes());
        Cid::new_v1(0x55, hash)
    }

    #[test]
    fn test_modality_defaults() {
        assert_eq!(Modality::Text.default_dim(), 768);
        assert_eq!(Modality::Image.default_dim(), 512);
        assert_eq!(Modality::Text.default_metric(), DistanceMetric::Cosine);
    }

    #[test]
    fn test_multimodal_embedding_creation() {
        let vec = vec![0.1, 0.2, 0.3];
        let emb = MultiModalEmbedding::new(vec.clone(), Modality::Text);

        assert_eq!(emb.vector, vec);
        assert_eq!(emb.modality, Modality::Text);
        assert_eq!(emb.dim(), 3);
    }

    #[test]
    fn test_multimodal_index_creation() {
        let config = MultiModalConfig::default();
        let mut index = MultiModalIndex::new(config);

        assert!(index.is_empty());
        assert_eq!(index.total_len(), 0);

        // Register modalities
        index
            .register_modality(Modality::Text, 768)
            .expect("test: register Text modality dim 768 should succeed");
        index
            .register_modality(Modality::Image, 512)
            .expect("test: register Image modality dim 512 should succeed");

        assert_eq!(index.len_for_modality(Modality::Text), 0);
        assert_eq!(index.len_for_modality(Modality::Image), 0);
    }

    #[test]
    fn test_add_and_search_single_modality() {
        let config = MultiModalConfig::default();
        let mut index = MultiModalIndex::new(config);
        index
            .register_modality(Modality::Text, 3)
            .expect("test: register Text modality dim 3 should succeed");

        // Add embeddings
        let cid1 = generate_test_cid(1);
        let emb1 = MultiModalEmbedding::new(vec![1.0, 0.0, 0.0], Modality::Text);
        index
            .add(cid1, emb1)
            .expect("test: add Text embedding cid1 should succeed");

        let cid2 = generate_test_cid(2);
        let emb2 = MultiModalEmbedding::new(vec![0.0, 1.0, 0.0], Modality::Text);
        index
            .add(cid2, emb2)
            .expect("test: add Text embedding cid2 should succeed");

        assert_eq!(index.len_for_modality(Modality::Text), 2);

        // Search
        let query = MultiModalEmbedding::new(vec![0.9, 0.1, 0.0], Modality::Text);
        let results = index
            .search_modality(&query, 1, None)
            .expect("test: single-modality search should succeed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, cid1);
    }

    #[test]
    fn test_cross_modal_search() {
        let config = MultiModalConfig::default();
        let mut index = MultiModalIndex::new(config);

        index
            .register_modality(Modality::Text, 3)
            .expect("test: register Text modality dim 3 should succeed");
        index
            .register_modality(Modality::Image, 3)
            .expect("test: register Image modality dim 3 should succeed");

        // Add text embedding
        let cid1 = generate_test_cid(3);
        let emb1 = MultiModalEmbedding::new(vec![1.0, 0.0, 0.0], Modality::Text);
        index
            .add(cid1, emb1)
            .expect("test: add Text embedding cid1 should succeed");

        // Add image embedding
        let cid2 = generate_test_cid(4);
        let emb2 = MultiModalEmbedding::new(vec![0.0, 1.0, 0.0], Modality::Image);
        index
            .add(cid2, emb2)
            .expect("test: add Image embedding cid2 should succeed");

        // Cross-modal search from text
        let query = MultiModalEmbedding::new(vec![0.9, 0.1, 0.0], Modality::Text);
        let results = index
            .search_cross_modal(&query, 2, None)
            .expect("test: cross-modal search should succeed");

        assert!(!results.is_empty());
    }

    #[test]
    fn test_modality_alignment() {
        let mut alignment = ModalityAlignment::new(Modality::Text, Modality::Image, 3, 3);

        // Create some paired examples
        let pairs = vec![
            (vec![1.0, 0.0, 0.0], vec![0.9, 0.1, 0.0]),
            (vec![0.0, 1.0, 0.0], vec![0.1, 0.9, 0.0]),
        ];

        alignment
            .learn_from_pairs(&pairs)
            .expect("test: learn_from_pairs with valid aligned pairs should succeed");

        // Transform a source embedding
        let source = vec![1.0, 0.0, 0.0];
        let transformed = alignment.transform_embedding(&source);

        assert_eq!(transformed.len(), 3);
        assert!(transformed[0] > 0.5); // Should be close to target space
    }

    #[test]
    fn test_modality_stats() {
        let config = MultiModalConfig::default();
        let mut index = MultiModalIndex::new(config);

        index
            .register_modality(Modality::Text, 768)
            .expect("test: register Text modality dim 768 should succeed");
        index
            .register_modality(Modality::Image, 512)
            .expect("test: register Image modality dim 512 should succeed");

        let stats = index.stats();

        assert_eq!(stats.len(), 2);
        assert_eq!(
            stats
                .get(&Modality::Text)
                .expect("test: Text modality should be present in stats")
                .dimension,
            768
        );
        assert_eq!(
            stats
                .get(&Modality::Image)
                .expect("test: Image modality should be present in stats")
                .dimension,
            512
        );
    }

    #[test]
    fn test_projection() {
        let config = MultiModalConfig {
            project_to_unified: true,
            unified_dim: 512,
            ..Default::default()
        };

        let mut index = MultiModalIndex::new(config);
        index
            .register_modality(Modality::Text, 768)
            .expect("test: register Text modality dim 768 should succeed");

        // Add an embedding (should be projected from 768 to 512)
        let cid = generate_test_cid(5);
        let emb = MultiModalEmbedding::new(vec![0.5; 768], Modality::Text);
        index
            .add(cid, emb)
            .expect("test: add Text embedding with projection should succeed");

        assert_eq!(index.len_for_modality(Modality::Text), 1);
    }
}
