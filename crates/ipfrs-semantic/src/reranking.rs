//! Query result re-ranking
//!
//! This module provides re-ranking capabilities for search results based on
//! multiple criteria including semantic similarity, metadata scores, recency,
//! and custom scoring functions.

use crate::hnsw::SearchResult;
use crate::metadata::{Metadata, MetadataValue};
use ipfrs_core::Cid;
use std::collections::HashMap;

/// Re-ranking strategy for search results
#[derive(Debug, Clone)]
pub enum ReRankingStrategy {
    /// Weighted combination of multiple scores
    WeightedCombination(Vec<(ScoreComponent, f32)>),
    /// Reciprocal Rank Fusion (RRF)
    ReciprocalRankFusion { k: f32 },
    /// Learning to Rank (placeholder for custom models)
    LearnToRank { model_name: String },
    /// Custom scoring function
    Custom,
}

/// Components that contribute to the final score
#[derive(Debug, Clone)]
pub enum ScoreComponent {
    /// Original vector similarity score
    VectorSimilarity,
    /// Metadata-based score (requires metadata lookup)
    MetadataScore { field: String },
    /// Recency score (requires timestamp metadata)
    Recency { decay_factor: f32 },
    /// Popularity score (requires popularity metadata)
    Popularity,
    /// Diversity score (penalize similar results)
    Diversity { threshold: f32 },
    /// Custom score (requires external scoring function)
    Custom { name: String },
}

/// Re-ranking configuration
#[derive(Debug, Clone)]
pub struct ReRankingConfig {
    /// Strategy to use for re-ranking
    pub strategy: ReRankingStrategy,
    /// Whether to normalize scores before combining
    pub normalize_scores: bool,
    /// Maximum number of results to re-rank (re-rank top-k only)
    pub top_k: Option<usize>,
}

impl Default for ReRankingConfig {
    fn default() -> Self {
        Self {
            strategy: ReRankingStrategy::WeightedCombination(vec![(
                ScoreComponent::VectorSimilarity,
                1.0,
            )]),
            normalize_scores: true,
            top_k: Some(100), // Re-rank top 100 only for efficiency
        }
    }
}

/// Result with multiple score components
#[derive(Debug, Clone)]
pub struct ScoredResult {
    /// The search result
    pub result: SearchResult,
    /// Individual score components
    pub score_components: HashMap<String, f32>,
    /// Final combined score
    pub final_score: f32,
}

/// Re-ranker for search results
pub struct ReRanker {
    config: ReRankingConfig,
    metadata_cache: HashMap<Cid, Metadata>,
}

impl ReRanker {
    /// Create a new re-ranker with the given configuration
    pub fn new(config: ReRankingConfig) -> Self {
        Self {
            config,
            metadata_cache: HashMap::new(),
        }
    }

    /// Create a re-ranker with default configuration
    pub fn with_defaults() -> Self {
        Self::new(ReRankingConfig::default())
    }

    /// Add metadata for a CID (for metadata-based scoring)
    pub fn add_metadata(&mut self, cid: Cid, metadata: Metadata) {
        self.metadata_cache.insert(cid, metadata);
    }

    /// Re-rank search results
    pub fn rerank(&self, results: Vec<SearchResult>) -> Vec<ScoredResult> {
        let limit = self
            .config
            .top_k
            .unwrap_or(results.len())
            .min(results.len());
        let mut to_rerank: Vec<SearchResult> = results.into_iter().take(limit).collect();

        match &self.config.strategy {
            ReRankingStrategy::WeightedCombination(weights) => {
                self.rerank_weighted(&mut to_rerank, weights)
            }
            ReRankingStrategy::ReciprocalRankFusion { k } => self.rerank_rrf(&mut to_rerank, *k),
            ReRankingStrategy::LearnToRank { model_name: _ } => {
                // Placeholder - would integrate with external model
                self.rerank_placeholder(&mut to_rerank)
            }
            ReRankingStrategy::Custom => self.rerank_placeholder(&mut to_rerank),
        }
    }

    /// Re-rank using weighted combination of scores
    fn rerank_weighted(
        &self,
        results: &mut [SearchResult],
        weights: &[(ScoreComponent, f32)],
    ) -> Vec<ScoredResult> {
        let mut scored_results: Vec<ScoredResult> = results
            .iter()
            .map(|r| {
                let mut score_components = HashMap::new();
                let mut final_score = 0.0;

                for (component, weight) in weights {
                    let component_score = self.compute_component_score(r, component);
                    let component_name = self.component_name(component);
                    score_components.insert(component_name, component_score);
                    final_score += component_score * weight;
                }

                ScoredResult {
                    result: r.clone(),
                    score_components,
                    final_score,
                }
            })
            .collect();

        // Normalize if requested
        if self.config.normalize_scores {
            self.normalize_scores(&mut scored_results);
        }

        // Sort by final score (descending)
        scored_results.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scored_results
    }

    /// Re-rank using Reciprocal Rank Fusion
    fn rerank_rrf(&self, results: &mut [SearchResult], k: f32) -> Vec<ScoredResult> {
        let scored_results: Vec<ScoredResult> = results
            .iter()
            .enumerate()
            .map(|(rank, r)| {
                let rrf_score = 1.0 / (k + rank as f32 + 1.0);
                let mut score_components = HashMap::new();
                score_components.insert("vector_similarity".to_string(), r.score);
                score_components.insert("rrf_score".to_string(), rrf_score);

                ScoredResult {
                    result: r.clone(),
                    score_components,
                    final_score: rrf_score,
                }
            })
            .collect();

        scored_results
    }

    /// Placeholder re-ranking (just return as-is)
    fn rerank_placeholder(&self, results: &mut [SearchResult]) -> Vec<ScoredResult> {
        results
            .iter()
            .map(|r| {
                let mut score_components = HashMap::new();
                score_components.insert("vector_similarity".to_string(), r.score);

                ScoredResult {
                    result: r.clone(),
                    score_components,
                    final_score: r.score,
                }
            })
            .collect()
    }

    /// Compute score for a single component
    fn compute_component_score(&self, result: &SearchResult, component: &ScoreComponent) -> f32 {
        match component {
            ScoreComponent::VectorSimilarity => result.score,
            ScoreComponent::MetadataScore { field } => {
                // Get metadata score from cached metadata
                if let Some(metadata) = self.metadata_cache.get(&result.cid) {
                    if let Some(value) = metadata.get(field) {
                        return self.metadata_value_to_score(value);
                    }
                }
                0.0
            }
            ScoreComponent::Recency { decay_factor } => {
                // Compute recency score from timestamp
                if let Some(metadata) = self.metadata_cache.get(&result.cid) {
                    if let Some(MetadataValue::Integer(timestamp)) = metadata.get("timestamp") {
                        // Simple exponential decay
                        let age = Self::current_timestamp() - timestamp;
                        return (-(age as f32) * decay_factor).exp();
                    }
                }
                0.0
            }
            ScoreComponent::Popularity => {
                // Get popularity from metadata
                if let Some(metadata) = self.metadata_cache.get(&result.cid) {
                    if let Some(value) = metadata.get("popularity") {
                        return self.metadata_value_to_score(value);
                    }
                }
                0.0
            }
            ScoreComponent::Diversity { threshold: _ } => {
                // Diversity scoring requires comparing with other results
                // Placeholder for now
                0.0
            }
            ScoreComponent::Custom { name: _ } => {
                // Custom scoring would call external function
                0.0
            }
        }
    }

    /// Convert metadata value to a score
    fn metadata_value_to_score(&self, value: &MetadataValue) -> f32 {
        match value {
            MetadataValue::Integer(i) => *i as f32,
            MetadataValue::Float(f) => *f as f32,
            MetadataValue::Boolean(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
            MetadataValue::Timestamp(t) => *t as f32,
            MetadataValue::String(_) | MetadataValue::StringArray(_) | MetadataValue::Null => 0.0,
        }
    }

    /// Get component name for display
    fn component_name(&self, component: &ScoreComponent) -> String {
        match component {
            ScoreComponent::VectorSimilarity => "vector_similarity".to_string(),
            ScoreComponent::MetadataScore { field } => format!("metadata_{}", field),
            ScoreComponent::Recency { .. } => "recency".to_string(),
            ScoreComponent::Popularity => "popularity".to_string(),
            ScoreComponent::Diversity { .. } => "diversity".to_string(),
            ScoreComponent::Custom { name } => format!("custom_{}", name),
        }
    }

    /// Normalize scores to [0, 1] range
    fn normalize_scores(&self, results: &mut [ScoredResult]) {
        if results.is_empty() {
            return;
        }

        // Find min and max scores
        let min_score = results
            .iter()
            .map(|r| r.final_score)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let max_score = results
            .iter()
            .map(|r| r.final_score)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(1.0);

        let range = max_score - min_score;

        if range > 0.0 {
            for result in results.iter_mut() {
                result.final_score = (result.final_score - min_score) / range;
            }
        }
    }

    /// Get current timestamp (seconds since epoch)
    fn current_timestamp() -> i64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time is after UNIX epoch")
            .as_secs() as i64
    }

    /// Create a weighted combination strategy
    pub fn weighted(components: Vec<(ScoreComponent, f32)>) -> ReRankingConfig {
        ReRankingConfig {
            strategy: ReRankingStrategy::WeightedCombination(components),
            normalize_scores: true,
            top_k: Some(100),
        }
    }

    /// Create a reciprocal rank fusion strategy
    pub fn reciprocal_rank_fusion(k: f32) -> ReRankingConfig {
        ReRankingConfig {
            strategy: ReRankingStrategy::ReciprocalRankFusion { k },
            normalize_scores: false,
            top_k: Some(100),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reranker_creation() {
        let reranker = ReRanker::with_defaults();
        assert!(matches!(
            reranker.config.strategy,
            ReRankingStrategy::WeightedCombination(_)
        ));
    }

    #[test]
    fn test_weighted_reranking() {
        let config = ReRanker::weighted(vec![
            (ScoreComponent::VectorSimilarity, 0.7),
            (ScoreComponent::Popularity, 0.3),
        ]);

        let mut reranker = ReRanker::new(config);

        // Create test results
        let cid1 = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: CID string is a valid base32 CIDv1");
        let cid2 = "bafybeihpjhkeuiq3k6nqa3fkgeigeri7iebtrsuyuey5y6vy36n345xmbi"
            .parse::<Cid>()
            .expect("test: CID string is a valid base32 CIDv1");

        // Add metadata
        let mut metadata1 = Metadata::new();
        metadata1.set("popularity", MetadataValue::Float(0.5));
        reranker.add_metadata(cid1, metadata1);

        let mut metadata2 = Metadata::new();
        metadata2.set("popularity", MetadataValue::Float(0.9));
        reranker.add_metadata(cid2, metadata2);

        let results = vec![
            SearchResult {
                cid: cid1,
                score: 0.9,
            },
            SearchResult {
                cid: cid2,
                score: 0.7,
            },
        ];

        let reranked = reranker.rerank(results);
        assert_eq!(reranked.len(), 2);

        // First result should still be cid1 (0.9*0.7 + 0.5*0.3 = 0.78)
        // vs cid2 (0.7*0.7 + 0.9*0.3 = 0.76)
        assert_eq!(reranked[0].result.cid, cid1);
    }

    #[test]
    fn test_rrf_reranking() {
        let config = ReRanker::reciprocal_rank_fusion(60.0);
        let reranker = ReRanker::new(config);

        let cid1 = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: CID string is a valid base32 CIDv1");
        let cid2 = "bafybeihpjhkeuiq3k6nqa3fkgeigeri7iebtrsuyuey5y6vy36n345xmbi"
            .parse::<Cid>()
            .expect("test: CID string is a valid base32 CIDv1");

        let results = vec![
            SearchResult {
                cid: cid1,
                score: 0.9,
            },
            SearchResult {
                cid: cid2,
                score: 0.7,
            },
        ];

        let reranked = reranker.rerank(results);
        assert_eq!(reranked.len(), 2);

        // Check RRF scores
        assert!(reranked[0].final_score > reranked[1].final_score);
    }

    #[test]
    fn test_recency_scoring() {
        let config = ReRanker::weighted(vec![
            (ScoreComponent::VectorSimilarity, 0.5),
            (
                ScoreComponent::Recency {
                    decay_factor: 0.0001,
                },
                0.5,
            ),
        ]);

        let mut reranker = ReRanker::new(config);

        let cid1 = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: CID string is a valid base32 CIDv1");

        let current_time = ReRanker::current_timestamp();

        let mut metadata = Metadata::new();
        metadata.set("timestamp", MetadataValue::Integer(current_time - 100));
        reranker.add_metadata(cid1, metadata);

        let results = vec![SearchResult {
            cid: cid1,
            score: 0.8,
        }];

        let reranked = reranker.rerank(results);
        assert_eq!(reranked.len(), 1);
        assert!(reranked[0].score_components.contains_key("recency"));
    }

    #[test]
    fn test_normalize_scores() {
        let config = ReRankingConfig {
            strategy: ReRankingStrategy::WeightedCombination(vec![(
                ScoreComponent::VectorSimilarity,
                1.0,
            )]),
            normalize_scores: true,
            top_k: None,
        };

        let reranker = ReRanker::new(config);

        let cid1 = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: CID string is a valid base32 CIDv1");
        let cid2 = "bafybeihpjhkeuiq3k6nqa3fkgeigeri7iebtrsuyuey5y6vy36n345xmbi"
            .parse::<Cid>()
            .expect("test: CID string is a valid base32 CIDv1");

        let results = vec![
            SearchResult {
                cid: cid1,
                score: 0.9,
            },
            SearchResult {
                cid: cid2,
                score: 0.5,
            },
        ];

        let reranked = reranker.rerank(results);

        // Normalized scores should be in [0, 1]
        assert!(reranked[0].final_score >= 0.0 && reranked[0].final_score <= 1.0);
        assert!(reranked[1].final_score >= 0.0 && reranked[1].final_score <= 1.0);
    }

    #[test]
    fn test_top_k_reranking() {
        let config = ReRankingConfig {
            strategy: ReRankingStrategy::WeightedCombination(vec![(
                ScoreComponent::VectorSimilarity,
                1.0,
            )]),
            normalize_scores: false,
            top_k: Some(2), // Only rerank top 2
        };

        let reranker = ReRanker::new(config);

        let cid1 = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
            .parse::<Cid>()
            .expect("test: CID string is a valid base32 CIDv1");
        let cid2 = "bafybeihpjhkeuiq3k6nqa3fkgeigeri7iebtrsuyuey5y6vy36n345xmbi"
            .parse::<Cid>()
            .expect("test: CID string is a valid base32 CIDv1");
        let cid3 = "bafybeif2pall7dybz7vecqka3zo24irdwabwdi4wc55jznaq75q7eaavvu"
            .parse::<Cid>()
            .expect("test: CID string is a valid base32 CIDv1");

        let results = vec![
            SearchResult {
                cid: cid1,
                score: 0.9,
            },
            SearchResult {
                cid: cid2,
                score: 0.7,
            },
            SearchResult {
                cid: cid3,
                score: 0.5,
            },
        ];

        let reranked = reranker.rerank(results);

        // Should only return top 2
        assert_eq!(reranked.len(), 2);
    }
}
