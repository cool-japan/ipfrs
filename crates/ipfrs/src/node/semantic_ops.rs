//! Semantic search and vector index operations for Node

use ipfrs_core::{Cid, Result};
use ipfrs_semantic::{QueryFilter, SearchResult};
use std::path::Path;

use super::{Node, SemanticStats};

impl Node {
    /// Get semantic router statistics
    ///
    /// Returns comprehensive statistics about the semantic index including
    /// vector count, dimension, distance metric, and cache performance.
    ///
    /// # Returns
    /// Statistics about the semantic router
    ///
    /// # Errors
    /// Returns error if semantic routing is not enabled
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// let stats = node.semantic_stats()?;
    /// println!("Indexed vectors: {}", stats.num_vectors);
    /// println!("Vector dimension: {}", stats.dimension);
    /// println!("Cache size: {}/{}", stats.cache_size, stats.cache_capacity);
    /// # Ok(())
    /// # }
    /// ```
    pub fn semantic_stats(&self) -> Result<SemanticStats> {
        let semantic = self.semantic()?;

        let router_stats = semantic.stats();
        let cache_stats = semantic.cache_stats();

        Ok(SemanticStats {
            num_vectors: router_stats.num_vectors,
            dimension: router_stats.dimension,
            metric: router_stats.metric,
            cache_size: cache_stats.size,
            cache_capacity: cache_stats.capacity,
        })
    }

    /// Index content with its semantic embedding
    ///
    /// Adds content to the semantic index for similarity search. The embedding
    /// should be a vector representation of the content (e.g., from a sentence
    /// transformer model).
    ///
    /// # Arguments
    /// * `cid` - Content identifier to index
    /// * `embedding` - Vector embedding of the content
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// # let cid = ipfrs_core::Cid::default();
    /// // Index content with 768-dimensional embedding (e.g., from BERT)
    /// let embedding = vec![0.5; 768];
    /// node.index_content(&cid, &embedding).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn index_content(&self, cid: &Cid, embedding: &[f32]) -> Result<()> {
        let semantic = self.semantic()?;
        semantic.add(cid, embedding)
    }

    /// Search for similar content by semantic similarity
    ///
    /// Performs k-nearest neighbor search over indexed content using vector
    /// similarity. Returns the top k most similar items.
    ///
    /// # Arguments
    /// * `query_embedding` - Query vector to search for
    /// * `k` - Number of results to return
    ///
    /// # Returns
    /// Vector of search results ordered by similarity (highest first)
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Search for top 10 similar documents
    /// let query_embedding = vec![0.3; 768];
    /// let results = node.search_similar(&query_embedding, 10).await?;
    ///
    /// for result in results {
    ///     println!("CID: {}, Score: {}", result.cid, result.score);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search_similar(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let semantic = self.semantic()?;
        semantic.query(query_embedding, k).await
    }

    /// Search with advanced filtering options
    ///
    /// Performs semantic search with additional filters like minimum score
    /// threshold, CID prefix matching, and result limits.
    ///
    /// # Arguments
    /// * `query_embedding` - Query vector to search for
    /// * `k` - Number of results to return
    /// * `filter` - Query filter options
    ///
    /// # Returns
    /// Vector of filtered search results ordered by similarity
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig, QueryFilter};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Search with filters
    /// let query_embedding = vec![0.3; 768];
    /// let filter = QueryFilter {
    ///     min_score: Some(0.8),  // Only results with score >= 0.8
    ///     max_score: None,        // No max score filter
    ///     max_results: Some(5),   // Limit to 5 results
    ///     cid_prefix: None,       // No CID filtering
    /// };
    ///
    /// let results = node.search_hybrid(&query_embedding, 20, filter).await?;
    ///
    /// for result in results {
    ///     println!("High-confidence match: {} ({})", result.cid, result.score);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search_hybrid(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: QueryFilter,
    ) -> Result<Vec<SearchResult>> {
        let semantic = self.semantic()?;
        semantic.query_with_filter(query_embedding, k, filter).await
    }

    /// Save the semantic index to disk
    ///
    /// Persists the entire HNSW index including all vectors and CID mappings
    /// to a file for later loading.
    ///
    /// # Arguments
    /// * `path` - Path to save the index file
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Save the semantic index
    /// node.save_semantic_index("semantic.index").await?;
    /// println!("Semantic index saved");
    /// # Ok(())
    /// # }
    /// ```
    pub async fn save_semantic_index(&self, path: impl AsRef<Path>) -> Result<()> {
        let semantic = self.semantic()?;
        semantic.save_index(path).await
    }

    /// Load a semantic index from disk
    ///
    /// Loads a previously saved HNSW index from disk, replacing the current index.
    ///
    /// # Arguments
    /// * `path` - Path to the saved index file
    ///
    /// # Example
    /// ```rust,no_run
    /// use ipfrs::{Node, NodeConfig};
    ///
    /// # async fn example() -> ipfrs::Result<()> {
    /// let mut node = Node::new(NodeConfig::default())?;
    /// node.start().await?;
    ///
    /// // Load the semantic index
    /// node.load_semantic_index("semantic.index").await?;
    /// println!("Semantic index loaded");
    /// # Ok(())
    /// # }
    /// ```
    pub async fn load_semantic_index(&self, path: impl AsRef<Path>) -> Result<()> {
        let semantic = self.semantic()?;
        semantic.load_index(path).await
    }
}
