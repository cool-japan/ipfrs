//! Utility functions and helpers for common semantic search workflows
//!
//! This module provides convenience functions that combine multiple features
//! and simplify common patterns in semantic search applications.

use crate::{
    analyze_quality, diagnose_index, HybridIndex, Metadata, SemanticRouter, VectorIndex,
    VectorQuality,
};
use ipfrs_core::{Cid, Result};
use std::collections::HashMap;

/// Statistics for a batch embedding operation
#[derive(Debug, Clone)]
pub struct BatchEmbeddingStats {
    /// Total number of embeddings processed
    pub total: usize,
    /// Number of valid embeddings
    pub valid: usize,
    /// Number of invalid embeddings (failed quality check)
    pub invalid: usize,
    /// Average quality score
    pub avg_quality: f32,
    /// Minimum quality score
    pub min_quality: f32,
    /// Maximum quality score
    pub max_quality: f32,
}

/// Result of a batch index operation with statistics
#[derive(Debug)]
pub struct BatchIndexResult {
    /// Number of items successfully indexed
    pub indexed: usize,
    /// Number of items that failed indexing
    pub failed: usize,
    /// CIDs that failed (with error messages)
    pub failures: Vec<(Cid, String)>,
    /// Statistics about the batch
    pub stats: BatchEmbeddingStats,
}

/// Validate and index embeddings in a single operation with quality checking
///
/// This helper combines quality analysis with indexing, automatically
/// filtering out low-quality embeddings.
///
/// # Arguments
/// * `router` - The semantic router to index into
/// * `items` - Vector of (CID, embedding) pairs to index
/// * `min_quality` - Minimum quality score (0.0-1.0) required for indexing
///
/// # Returns
/// Statistics about the batch operation including success/failure counts
///
/// # Example
/// ```
/// use ipfrs_semantic::{SemanticRouter, utils::index_with_quality_check};
/// use ipfrs_core::Cid;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let router = SemanticRouter::with_defaults()?;
///
/// let items = vec![
///     ("bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi".parse::<Cid>()?, vec![0.1; 768]),
///     ("bafybeihpjhkeuiq3k6nqa3fkgeigeri7iebtrsuyuey5y6vy36n345xmbi".parse::<Cid>()?, vec![0.2; 768]),
/// ];
///
/// let result = index_with_quality_check(&router, &items, 0.5)?;
/// println!("Indexed: {}, Failed: {}", result.indexed, result.failed);
/// println!("Average quality: {:.2}", result.stats.avg_quality);
/// # Ok(())
/// # }
/// ```
pub fn index_with_quality_check(
    router: &SemanticRouter,
    items: &[(Cid, Vec<f32>)],
    min_quality: f32,
) -> Result<BatchIndexResult> {
    let mut indexed = 0;
    let mut failed = 0;
    let mut failures = Vec::new();
    let mut qualities = Vec::new();

    for (cid, embedding) in items {
        let quality = analyze_quality(embedding);
        qualities.push(quality.quality_score);

        if quality.quality_score >= min_quality && quality.is_valid {
            match router.add(cid, embedding) {
                Ok(_) => indexed += 1,
                Err(e) => {
                    failed += 1;
                    failures.push((*cid, e.to_string()));
                }
            }
        } else {
            failed += 1;
            failures.push((
                *cid,
                format!("Quality check failed: score={:.2}", quality.quality_score),
            ));
        }
    }

    let stats = if qualities.is_empty() {
        BatchEmbeddingStats {
            total: items.len(),
            valid: 0,
            invalid: items.len(),
            avg_quality: 0.0,
            min_quality: 0.0,
            max_quality: 0.0,
        }
    } else {
        let avg = qualities.iter().sum::<f32>() / qualities.len() as f32;
        let min = qualities.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = qualities.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        BatchEmbeddingStats {
            total: items.len(),
            valid: indexed,
            invalid: failed,
            avg_quality: avg,
            min_quality: min,
            max_quality: max,
        }
    };

    Ok(BatchIndexResult {
        indexed,
        failed,
        failures,
        stats,
    })
}

/// Validate a batch of embeddings and return quality reports
///
/// # Example
/// ```
/// use ipfrs_semantic::utils::validate_embeddings;
///
/// let embeddings = vec![
///     vec![0.1, 0.2, 0.3],
///     vec![0.4, 0.5, 0.6],
///     vec![f32::NAN, 0.1, 0.2],  // Invalid (contains NaN)
/// ];
///
/// let reports = validate_embeddings(&embeddings);
/// assert_eq!(reports.len(), 3);
/// assert!(reports[0].is_valid);
/// assert!(reports[1].is_valid);
/// assert!(!reports[2].is_valid);  // Contains NaN
/// ```
pub fn validate_embeddings(embeddings: &[Vec<f32>]) -> Vec<VectorQuality> {
    embeddings.iter().map(|e| analyze_quality(e)).collect()
}

/// Create a hybrid index with metadata extracted from a CID mapping
///
/// This is a convenience function for creating a hybrid index when you have
/// a mapping of CIDs to both embeddings and metadata.
///
/// # Example
/// ```
/// use ipfrs_semantic::{utils::create_hybrid_index_from_map, Metadata, MetadataValue};
/// use ipfrs_core::Cid;
/// use std::collections::HashMap;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut data = HashMap::new();
///
/// let cid: Cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi".parse()?;
/// let embedding = vec![0.5; 768];
/// let mut metadata = Metadata::new();
/// metadata.set("type", MetadataValue::String("document".to_string()));
///
/// data.insert(cid, (embedding, Some(metadata)));
///
/// let index = create_hybrid_index_from_map(768, data)?;
/// # Ok(())
/// # }
/// ```
pub fn create_hybrid_index_from_map(
    dimension: usize,
    data: HashMap<Cid, (Vec<f32>, Option<Metadata>)>,
) -> Result<HybridIndex> {
    let index = HybridIndex::new(crate::HybridConfig {
        dimension,
        ..Default::default()
    })?;

    for (cid, (embedding, metadata)) in data {
        index.insert(&cid, &embedding, metadata)?;
    }

    Ok(index)
}

/// Health check result for a semantic router
#[derive(Debug)]
pub struct HealthCheckResult {
    /// Is the index healthy?
    pub is_healthy: bool,
    /// Number of vectors in the index
    pub vector_count: usize,
    /// Estimated memory usage in bytes
    pub memory_bytes: usize,
    /// Issues detected (if any)
    pub issues: Vec<String>,
    /// Recommendations for optimization
    pub recommendations: Vec<String>,
}

/// Perform a comprehensive health check on a vector index
///
/// # Example
/// ```
/// use ipfrs_semantic::{VectorIndex, utils::health_check};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let index = VectorIndex::with_defaults(768)?;
/// let health = health_check(&index);
///
/// if health.is_healthy {
///     println!("Index is healthy with {} vectors", health.vector_count);
/// } else {
///     println!("Issues found: {:?}", health.issues);
/// }
/// # Ok(())
/// # }
/// ```
pub fn health_check(index: &VectorIndex) -> HealthCheckResult {
    let report = diagnose_index(index);

    HealthCheckResult {
        is_healthy: matches!(report.status, crate::diagnostics::HealthStatus::Healthy),
        vector_count: report.size,
        memory_bytes: report.memory_usage,
        issues: report
            .issues
            .iter()
            .map(|i| i.description.clone())
            .collect(),
        recommendations: report.recommendations,
    }
}

/// Normalize a vector to unit length (L2 norm = 1)
///
/// This is useful for cosine similarity searches, as normalized vectors
/// allow using dot product instead of cosine distance.
///
/// # Example
/// ```
/// use ipfrs_semantic::utils::normalize_vector;
///
/// let mut vec = vec![3.0, 4.0];
/// normalize_vector(&mut vec);
///
/// let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
/// assert!((norm - 1.0).abs() < 1e-6);
/// ```
pub fn normalize_vector(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in vector.iter_mut() {
            *x /= norm;
        }
    }
}

/// Normalize a batch of vectors in place
///
/// # Example
/// ```
/// use ipfrs_semantic::utils::normalize_vectors;
///
/// let mut vectors = vec![
///     vec![3.0, 4.0],
///     vec![1.0, 0.0],
/// ];
///
/// normalize_vectors(&mut vectors);
///
/// for vec in &vectors {
///     let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
///     assert!((norm - 1.0).abs() < 1e-6);
/// }
/// ```
pub fn normalize_vectors(vectors: &mut [Vec<f32>]) {
    for vector in vectors.iter_mut() {
        normalize_vector(vector);
    }
}

/// Calculate the average embedding from a set of embeddings
///
/// Useful for creating centroid embeddings or aggregate representations.
///
/// Returns None if the input is empty or embeddings have inconsistent dimensions.
///
/// # Example
/// ```
/// use ipfrs_semantic::utils::average_embedding;
///
/// let embeddings = vec![
///     vec![1.0, 2.0, 3.0],
///     vec![2.0, 3.0, 4.0],
///     vec![3.0, 4.0, 5.0],
/// ];
///
/// let avg = average_embedding(&embeddings).unwrap();
/// assert_eq!(avg, vec![2.0, 3.0, 4.0]);
/// ```
pub fn average_embedding(embeddings: &[Vec<f32>]) -> Option<Vec<f32>> {
    if embeddings.is_empty() {
        return None;
    }

    let dim = embeddings[0].len();
    if embeddings.iter().any(|e| e.len() != dim) {
        return None;
    }

    let mut result = vec![0.0; dim];
    for embedding in embeddings {
        for (i, &val) in embedding.iter().enumerate() {
            result[i] += val;
        }
    }

    let count = embeddings.len() as f32;
    for val in result.iter_mut() {
        *val /= count;
    }

    Some(result)
}

/// Result of a batch deletion operation
#[derive(Debug, Clone)]
pub struct BatchDeletionResult {
    /// Number of CIDs successfully deleted
    pub deleted: usize,
    /// Number of CIDs not found in the index
    pub not_found: usize,
    /// Number of CIDs that failed to delete
    pub failed: usize,
    /// CIDs that were not found
    pub not_found_cids: Vec<Cid>,
    /// CIDs that failed deletion (with error messages)
    pub failures: Vec<(Cid, String)>,
}

/// Delete multiple CIDs from a vector index in batch
///
/// This function efficiently deletes multiple CIDs and provides detailed
/// statistics about the operation.
///
/// # Arguments
/// * `index` - The vector index to delete from
/// * `cids` - List of CIDs to delete
///
/// # Returns
/// Statistics about the deletion operation
///
/// # Example
/// ```
/// use ipfrs_semantic::{VectorIndex, utils::batch_delete};
/// use ipfrs_core::Cid;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut index = VectorIndex::with_defaults(768)?;
///
/// // Add some vectors
/// let cid1: Cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi".parse()?;
/// let cid2: Cid = "bafybeihpjhkeuiq3k6nqa3fkgeigeri7iebtrsuyuey5y6vy36n345xmbi".parse()?;
/// index.insert(&cid1, &vec![0.1; 768])?;
/// index.insert(&cid2, &vec![0.2; 768])?;
///
/// // Delete them in batch
/// let result = batch_delete(&mut index, &[cid1, cid2])?;
/// assert_eq!(result.deleted, 2);
/// assert_eq!(result.not_found, 0);
/// # Ok(())
/// # }
/// ```
pub fn batch_delete(index: &mut VectorIndex, cids: &[Cid]) -> Result<BatchDeletionResult> {
    let mut deleted = 0;
    let mut not_found = 0;
    let mut failed = 0;
    let mut not_found_cids = Vec::new();
    let mut failures = Vec::new();

    for cid in cids {
        if !index.contains(cid) {
            not_found += 1;
            not_found_cids.push(*cid);
            continue;
        }

        match index.delete(cid) {
            Ok(_) => deleted += 1,
            Err(e) => {
                failed += 1;
                failures.push((*cid, e.to_string()));
            }
        }
    }

    Ok(BatchDeletionResult {
        deleted,
        not_found,
        failed,
        not_found_cids,
        failures,
    })
}

/// Calculate cosine similarity between two embeddings
///
/// Returns a value between -1.0 and 1.0, where:
/// - 1.0 means vectors point in the same direction (most similar)
/// - 0.0 means vectors are orthogonal (no similarity)
/// - -1.0 means vectors point in opposite directions (most dissimilar)
///
/// Returns None if embeddings have different dimensions or are zero vectors.
///
/// # Example
/// ```
/// use ipfrs_semantic::utils::cosine_similarity;
///
/// let embedding1 = vec![1.0, 2.0, 3.0];
/// let embedding2 = vec![2.0, 4.0, 6.0]; // Parallel to embedding1
///
/// let similarity = cosine_similarity(&embedding1, &embedding2).unwrap();
/// assert!((similarity - 1.0).abs() < 1e-6); // Should be 1.0 (perfectly similar)
/// ```
pub fn cosine_similarity(embedding1: &[f32], embedding2: &[f32]) -> Option<f32> {
    if embedding1.len() != embedding2.len() {
        return None;
    }

    let dot_product: f32 = embedding1
        .iter()
        .zip(embedding2.iter())
        .map(|(a, b)| a * b)
        .sum();

    let norm1: f32 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = embedding2.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm1 == 0.0 || norm2 == 0.0 {
        return None;
    }

    Some(dot_product / (norm1 * norm2))
}

/// Calculate pairwise cosine similarities between a query and multiple embeddings
///
/// This is useful for finding the most similar embeddings from a set without
/// indexing them first.
///
/// # Arguments
/// * `query` - Query embedding
/// * `embeddings` - List of embeddings to compare against
///
/// # Returns
/// Vector of (index, similarity) pairs, sorted by similarity (descending)
///
/// # Example
/// ```
/// use ipfrs_semantic::utils::pairwise_similarities;
///
/// let query = vec![1.0, 0.0, 0.0];
/// let embeddings = vec![
///     vec![1.0, 0.0, 0.0],  // Same as query
///     vec![0.0, 1.0, 0.0],  // Orthogonal
///     vec![0.7, 0.7, 0.0],  // Partially similar
/// ];
///
/// let similarities = pairwise_similarities(&query, &embeddings);
/// assert_eq!(similarities.len(), 3);
/// assert_eq!(similarities[0].0, 0);  // First embedding is most similar
/// assert!((similarities[0].1 - 1.0).abs() < 1e-6);
/// ```
pub fn pairwise_similarities(query: &[f32], embeddings: &[Vec<f32>]) -> Vec<(usize, f32)> {
    let mut results: Vec<(usize, f32)> = embeddings
        .iter()
        .enumerate()
        .filter_map(|(idx, emb)| cosine_similarity(query, emb).map(|sim| (idx, sim)))
        .collect();

    // Sort by similarity descending
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    results
}

/// Export index statistics to a JSON-serializable structure
///
/// This function extracts comprehensive statistics from an index for
/// monitoring, debugging, or export purposes.
///
/// # Example
/// ```
/// use ipfrs_semantic::{VectorIndex, utils::export_index_stats};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut index = VectorIndex::with_defaults(768)?;
/// let cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi".parse()?;
/// index.insert(&cid, &vec![0.5; 768])?;
///
/// let stats = export_index_stats(&index);
/// assert_eq!(stats.dimension, 768);
/// assert_eq!(stats.vector_count, 1);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, serde::Serialize)]
pub struct IndexStats {
    /// Embedding dimension
    pub dimension: usize,
    /// Number of vectors in index
    pub vector_count: usize,
    /// Distance metric used
    pub metric: String,
    /// Estimated memory usage in bytes
    pub memory_bytes: usize,
    /// Health status
    pub health_status: String,
    /// Issues detected
    pub issues: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

pub fn export_index_stats(index: &VectorIndex) -> IndexStats {
    let health = health_check(index);
    let metric = format!("{:?}", index.metric());

    IndexStats {
        dimension: index.dimension(),
        vector_count: index.len(),
        metric,
        memory_bytes: health.memory_bytes,
        health_status: if health.is_healthy {
            "Healthy".to_string()
        } else {
            "Issues Detected".to_string()
        },
        issues: health.issues,
        recommendations: health.recommendations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_vector() {
        let mut vec = vec![3.0, 4.0];
        normalize_vector(&mut vec);

        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((vec[0] - 0.6).abs() < 1e-6);
        assert!((vec[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let mut vec = vec![0.0, 0.0];
        normalize_vector(&mut vec);
        assert_eq!(vec, vec![0.0, 0.0]);
    }

    #[test]
    fn test_normalize_vectors() {
        let mut vectors = vec![vec![3.0, 4.0], vec![1.0, 0.0]];

        normalize_vectors(&mut vectors);

        for vec in &vectors {
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_average_embedding() {
        let embeddings = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
        ];

        let avg = average_embedding(&embeddings)
            .expect("test: non-empty uniform-dim embeddings should average successfully");
        assert_eq!(avg, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_average_embedding_empty() {
        let embeddings: Vec<Vec<f32>> = vec![];
        assert!(average_embedding(&embeddings).is_none());
    }

    #[test]
    fn test_average_embedding_inconsistent_dims() {
        let embeddings = vec![vec![1.0, 2.0], vec![3.0, 4.0, 5.0]];
        assert!(average_embedding(&embeddings).is_none());
    }

    #[test]
    fn test_validate_embeddings() {
        let embeddings = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![f32::NAN, 0.1, 0.2],
        ];

        let reports = validate_embeddings(&embeddings);
        assert_eq!(reports.len(), 3);
        assert!(reports[0].is_valid);
        assert!(reports[1].is_valid);
        assert!(!reports[2].is_valid); // NaN is invalid
    }

    #[test]
    fn test_health_check() {
        let index = VectorIndex::with_defaults(128)
            .expect("test: VectorIndex::with_defaults should succeed");
        let health = health_check(&index);

        // Empty index may not be healthy depending on implementation
        // At minimum, it should report 0 vectors
        assert_eq!(health.vector_count, 0);
    }

    #[test]
    fn test_batch_delete() {
        use multihash_codetable::{Code, MultihashDigest};

        let mut index = VectorIndex::with_defaults(768)
            .expect("test: VectorIndex::with_defaults should succeed");

        // Insert some test vectors
        let mut cids = Vec::new();
        for i in 0..5 {
            let data = format!("test_vector_{}", i);
            let hash = Code::Sha2_256.digest(data.as_bytes());
            let cid = Cid::new_v1(0x55, hash);
            index
                .insert(&cid, &vec![i as f32 * 0.1; 768])
                .expect("test: inserting valid vector into index should succeed");
            cids.push(cid);
        }

        // Delete first 3 CIDs
        let to_delete = &cids[0..3];
        let result = batch_delete(&mut index, to_delete)
            .expect("test: batch_delete should succeed for existing CIDs");

        assert_eq!(result.deleted, 3);
        assert_eq!(result.not_found, 0);
        assert_eq!(result.failed, 0);
        assert_eq!(index.len(), 2); // 2 remaining
    }

    #[test]
    fn test_batch_delete_not_found() {
        use multihash_codetable::{Code, MultihashDigest};

        let mut index = VectorIndex::with_defaults(768)
            .expect("test: VectorIndex::with_defaults should succeed");

        // Create a CID that's not in the index
        let data = "nonexistent";
        let hash = Code::Sha2_256.digest(data.as_bytes());
        let cid = Cid::new_v1(0x55, hash);

        let result = batch_delete(&mut index, &[cid])
            .expect("test: batch_delete should succeed even when CID not found");

        assert_eq!(result.deleted, 0);
        assert_eq!(result.not_found, 1);
        assert_eq!(result.not_found_cids.len(), 1);
    }

    #[test]
    fn test_cosine_similarity() {
        // Test identical vectors
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&vec1, &vec2)
            .expect("test: cosine_similarity of same-dimension vectors should return Some");
        assert!((sim - 1.0).abs() < 1e-6);

        // Test orthogonal vectors
        let vec3 = vec![1.0, 0.0, 0.0];
        let vec4 = vec![0.0, 1.0, 0.0];
        let sim2 = cosine_similarity(&vec3, &vec4).expect(
            "test: cosine_similarity of same-dimension orthogonal vectors should return Some",
        );
        assert!(sim2.abs() < 1e-6); // Should be ~0

        // Test parallel vectors (same direction, different magnitude)
        let vec5 = vec![1.0, 2.0, 3.0];
        let vec6 = vec![2.0, 4.0, 6.0];
        let sim3 = cosine_similarity(&vec5, &vec6).expect(
            "test: cosine_similarity of same-dimension parallel vectors should return Some",
        );
        assert!((sim3 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_different_dims() {
        let vec1 = vec![1.0, 2.0];
        let vec2 = vec![1.0, 2.0, 3.0];
        assert!(cosine_similarity(&vec1, &vec2).is_none());
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let vec1 = vec![0.0, 0.0, 0.0];
        let vec2 = vec![1.0, 2.0, 3.0];
        assert!(cosine_similarity(&vec1, &vec2).is_none());
    }

    #[test]
    fn test_pairwise_similarities() {
        let query = vec![1.0, 0.0, 0.0];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0], // Same as query
            vec![0.0, 1.0, 0.0], // Orthogonal
            vec![0.7, 0.7, 0.0], // Partially similar
        ];

        let similarities = pairwise_similarities(&query, &embeddings);

        assert_eq!(similarities.len(), 3);
        assert_eq!(similarities[0].0, 0); // First embedding is most similar
        assert!((similarities[0].1 - 1.0).abs() < 1e-6);
        assert!(similarities[1].1 > similarities[2].1); // vec[2] (orthogonal) should be least similar
    }

    #[test]
    fn test_export_index_stats() {
        use multihash_codetable::{Code, MultihashDigest};

        let mut index = VectorIndex::with_defaults(768)
            .expect("test: VectorIndex::with_defaults should succeed");

        // Add a vector
        let data = "test_vector";
        let hash = Code::Sha2_256.digest(data.as_bytes());
        let cid = Cid::new_v1(0x55, hash);
        index
            .insert(&cid, &vec![0.5; 768])
            .expect("test: inserting valid vector into index should succeed");

        let stats = export_index_stats(&index);

        assert_eq!(stats.dimension, 768);
        assert_eq!(stats.vector_count, 1);
        assert!(!stats.metric.is_empty());
    }
}
