//! Hybrid search combining vector similarity with metadata filtering
//!
//! This module provides a unified search interface that combines
//! semantic vector search with attribute-based filtering.

use crate::hnsw::{DistanceMetric, SearchResult, VectorIndex};
use crate::metadata::{Metadata, MetadataFilter, MetadataStore, TemporalOptions};
use crate::stats::{IndexHealth, IndexStats, MemoryUsage, PerfTimer, StatsSnapshot};
use ipfrs_core::{Cid, Error, Result};
use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::{Arc, RwLock};

/// Hybrid search configuration
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Vector dimension
    pub dimension: usize,
    /// Distance metric
    pub metric: DistanceMetric,
    /// HNSW max connections
    pub max_connections: usize,
    /// HNSW ef_construction
    pub ef_construction: usize,
    /// Default ef_search
    pub ef_search: usize,
    /// Query cache size
    pub cache_size: usize,
    /// Enable statistics collection
    pub collect_stats: bool,
    /// Filtering strategy
    pub filter_strategy: FilterStrategy,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            dimension: 768,
            metric: DistanceMetric::Cosine,
            max_connections: 16,
            ef_construction: 200,
            ef_search: 50,
            cache_size: 1000,
            collect_stats: true,
            filter_strategy: FilterStrategy::Auto,
        }
    }
}

/// Strategy for applying filters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterStrategy {
    /// Automatically choose based on selectivity
    Auto,
    /// Filter before vector search (pre-filtering)
    PreFilter,
    /// Filter after vector search (post-filtering)
    PostFilter,
}

/// Hybrid search query
#[derive(Debug, Clone)]
pub struct HybridQuery {
    /// Query vector
    pub vector: Vec<f32>,
    /// Number of results to return
    pub k: usize,
    /// Metadata filter (optional)
    pub filter: Option<MetadataFilter>,
    /// Temporal options (optional)
    pub temporal: Option<TemporalOptions>,
    /// Minimum similarity score
    pub min_score: Option<f32>,
    /// Override ef_search parameter
    pub ef_search: Option<usize>,
    /// Include metadata in results
    pub include_metadata: bool,
}

impl HybridQuery {
    /// Create a simple k-NN query
    pub fn knn(vector: Vec<f32>, k: usize) -> Self {
        Self {
            vector,
            k,
            filter: None,
            temporal: None,
            min_score: None,
            ef_search: None,
            include_metadata: false,
        }
    }

    /// Add a metadata filter
    pub fn with_filter(mut self, filter: MetadataFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Add temporal options
    pub fn with_temporal(mut self, temporal: TemporalOptions) -> Self {
        self.temporal = Some(temporal);
        self
    }

    /// Set minimum score threshold
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = Some(min_score);
        self
    }

    /// Include metadata in results
    pub fn with_metadata(mut self) -> Self {
        self.include_metadata = true;
        self
    }

    /// Override ef_search parameter
    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = Some(ef_search);
        self
    }
}

/// Hybrid search result with optional metadata
#[derive(Debug, Clone)]
pub struct HybridResult {
    /// Content identifier
    pub cid: Cid,
    /// Similarity score
    pub score: f32,
    /// Metadata (if requested)
    pub metadata: Option<Metadata>,
}

impl From<SearchResult> for HybridResult {
    fn from(result: SearchResult) -> Self {
        Self {
            cid: result.cid,
            score: result.score,
            metadata: None,
        }
    }
}

/// Hybrid search response
#[derive(Debug, Clone)]
pub struct HybridResponse {
    /// Search results
    pub results: Vec<HybridResult>,
    /// Total candidates evaluated
    pub total_evaluated: usize,
    /// Search latency in microseconds
    pub latency_us: u64,
    /// Filter strategy used
    pub strategy_used: FilterStrategy,
}

/// Hybrid search index combining HNSW with metadata
pub struct HybridIndex {
    /// Vector index
    vector_index: Arc<RwLock<VectorIndex>>,
    /// Metadata store
    metadata_store: Arc<MetadataStore>,
    /// Configuration
    config: HybridConfig,
    /// Statistics
    stats: Arc<IndexStats>,
    /// Query cache
    cache: Arc<RwLock<LruCache<u64, Vec<HybridResult>>>>,
}

impl HybridIndex {
    /// Create a new hybrid index
    pub fn new(config: HybridConfig) -> Result<Self> {
        let vector_index = VectorIndex::new(
            config.dimension,
            config.metric,
            config.max_connections,
            config.ef_construction,
        )?;

        let cache_size = NonZeroUsize::new(config.cache_size)
            .unwrap_or(NonZeroUsize::new(1000).expect("1000 > 0"));

        Ok(Self {
            vector_index: Arc::new(RwLock::new(vector_index)),
            metadata_store: Arc::new(MetadataStore::new()),
            config,
            stats: Arc::new(IndexStats::new()),
            cache: Arc::new(RwLock::new(LruCache::new(cache_size))),
        })
    }

    /// Create with default configuration
    pub fn with_defaults() -> Result<Self> {
        Self::new(HybridConfig::default())
    }

    /// Insert a vector with metadata
    pub fn insert(&self, cid: &Cid, vector: &[f32], metadata: Option<Metadata>) -> Result<()> {
        let timer = PerfTimer::start();

        // Insert into vector index
        self.vector_index
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(cid, vector)?;

        // Insert metadata if provided
        if let Some(meta) = metadata {
            self.metadata_store.insert(*cid, meta)?;
        } else {
            // Create minimal metadata with timestamp
            self.metadata_store.insert(*cid, Metadata::new())?;
        }

        if self.config.collect_stats {
            self.stats.record_insert(timer.stop());
        }

        // Invalidate cache
        self.cache
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .clear();

        Ok(())
    }

    /// Insert multiple vectors with metadata in batch
    pub fn insert_batch(&self, items: &[(Cid, Vec<f32>, Option<Metadata>)]) -> Result<()> {
        for (cid, vector, metadata) in items {
            self.insert(cid, vector, metadata.clone())?;
        }
        Ok(())
    }

    /// Delete a vector and its metadata
    pub fn delete(&self, cid: &Cid) -> Result<()> {
        self.vector_index
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .delete(cid)?;
        self.metadata_store.remove(cid)?;

        if self.config.collect_stats {
            self.stats.record_delete();
        }

        // Invalidate cache
        self.cache
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .clear();

        Ok(())
    }

    /// Perform hybrid search
    pub async fn search(&self, query: HybridQuery) -> Result<HybridResponse> {
        let timer = PerfTimer::start();

        // Determine filter strategy
        let strategy = self.determine_strategy(&query);
        let mut total_evaluated = 0;

        let results = match strategy {
            FilterStrategy::PreFilter => {
                self.search_pre_filter(&query, &mut total_evaluated).await?
            }
            FilterStrategy::PostFilter | FilterStrategy::Auto => {
                self.search_post_filter(&query, &mut total_evaluated)
                    .await?
            }
        };

        let latency = timer.stop();

        if self.config.collect_stats {
            self.stats.record_search(latency, query.k, results.len());
        }

        Ok(HybridResponse {
            results,
            total_evaluated,
            latency_us: latency.as_micros() as u64,
            strategy_used: strategy,
        })
    }

    /// Pre-filter strategy: filter first, then search on subset
    async fn search_pre_filter(
        &self,
        query: &HybridQuery,
        total_evaluated: &mut usize,
    ) -> Result<Vec<HybridResult>> {
        // Get candidate CIDs from filter
        let candidates: HashSet<Cid> = if let Some(ref filter) = query.filter {
            self.metadata_store.filter(filter).into_iter().collect()
        } else {
            // No filter, use all CIDs
            self.metadata_store.cids().into_iter().collect()
        };

        // Apply temporal filter if present
        let candidates = if let Some(ref temporal) = query.temporal {
            let time_filtered = self
                .metadata_store
                .get_by_time_range(temporal.start, temporal.end);
            candidates
                .intersection(&time_filtered.into_iter().collect())
                .copied()
                .collect()
        } else {
            candidates
        };

        *total_evaluated = candidates.len();

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Search vector index
        let ef_search = query.ef_search.unwrap_or(self.config.ef_search);
        let fetch_k = (query.k * 3).max(100); // Fetch more to account for filtering

        let search_results = self
            .vector_index
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .search(&query.vector, fetch_k, ef_search)?;

        // Filter results to candidates
        let mut results: Vec<HybridResult> = search_results
            .into_iter()
            .filter(|r| candidates.contains(&r.cid))
            .map(|r| {
                let mut hr = HybridResult::from(r);
                // Apply recency boost
                if let Some(ref temporal) = query.temporal {
                    if let Some(meta) = self.metadata_store.get(&hr.cid) {
                        let boost = temporal.recency_multiplier(meta.created_at);
                        hr.score *= boost;
                    }
                }
                hr
            })
            .collect();

        // Apply min score filter
        if let Some(min_score) = query.min_score {
            results.retain(|r| r.score >= min_score);
        }

        // Sort by score and truncate
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(query.k);

        // Add metadata if requested
        if query.include_metadata {
            for result in &mut results {
                result.metadata = self.metadata_store.get(&result.cid);
            }
        }

        Ok(results)
    }

    /// Post-filter strategy: search first, then filter results
    async fn search_post_filter(
        &self,
        query: &HybridQuery,
        total_evaluated: &mut usize,
    ) -> Result<Vec<HybridResult>> {
        let ef_search = query.ef_search.unwrap_or(self.config.ef_search);

        // Fetch more results to account for filtering
        let fetch_k = if query.filter.is_some() || query.temporal.is_some() {
            (query.k * 5).max(100)
        } else {
            query.k
        };

        let search_results = self
            .vector_index
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .search(&query.vector, fetch_k, ef_search)?;

        *total_evaluated = search_results.len();

        let mut results: Vec<HybridResult> = search_results
            .into_iter()
            .filter_map(|r| {
                // Apply metadata filter
                if let Some(ref filter) = query.filter {
                    if let Some(meta) = self.metadata_store.get(&r.cid) {
                        if !filter.matches(&meta) {
                            return None;
                        }
                    } else {
                        return None; // No metadata, filter out
                    }
                }

                // Apply temporal filter
                if let Some(ref temporal) = query.temporal {
                    if let Some(meta) = self.metadata_store.get(&r.cid) {
                        if let (Some(start), Some(end)) = (temporal.start, temporal.end) {
                            if meta.created_at < start || meta.created_at > end {
                                return None;
                            }
                        }
                    }
                }

                let mut hr = HybridResult::from(r);

                // Apply recency boost
                if let Some(ref temporal) = query.temporal {
                    if let Some(meta) = self.metadata_store.get(&hr.cid) {
                        let boost = temporal.recency_multiplier(meta.created_at);
                        hr.score *= boost;
                    }
                }

                Some(hr)
            })
            .collect();

        // Apply min score filter
        if let Some(min_score) = query.min_score {
            results.retain(|r| r.score >= min_score);
        }

        // Re-sort if recency boost was applied
        if query.temporal.is_some() {
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        results.truncate(query.k);

        // Add metadata if requested
        if query.include_metadata {
            for result in &mut results {
                result.metadata = self.metadata_store.get(&result.cid);
            }
        }

        Ok(results)
    }

    /// Determine the best filter strategy
    fn determine_strategy(&self, query: &HybridQuery) -> FilterStrategy {
        if self.config.filter_strategy != FilterStrategy::Auto {
            return self.config.filter_strategy;
        }

        // Estimate selectivity
        let total_count = self.metadata_store.len();
        if total_count == 0 {
            return FilterStrategy::PostFilter;
        }

        // If no filter, use post-filter (simpler path)
        if query.filter.is_none() && query.temporal.is_none() {
            return FilterStrategy::PostFilter;
        }

        // Estimate filter selectivity
        let filtered_count = if let Some(ref filter) = query.filter {
            self.metadata_store.filter(filter).len()
        } else {
            total_count
        };

        let selectivity = filtered_count as f64 / total_count as f64;

        // Pre-filter if highly selective (< 10% of data)
        // Post-filter if less selective (more data passes)
        if selectivity < 0.1 {
            FilterStrategy::PreFilter
        } else {
            FilterStrategy::PostFilter
        }
    }

    /// Get the number of indexed vectors
    pub fn len(&self) -> usize {
        self.vector_index
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if a CID exists
    pub fn contains(&self, cid: &Cid) -> bool {
        self.vector_index
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains(cid)
    }

    /// Get metadata for a CID
    pub fn get_metadata(&self, cid: &Cid) -> Option<Metadata> {
        self.metadata_store.get(cid)
    }

    /// Update metadata for a CID (without changing the vector)
    pub fn update_metadata(&self, cid: &Cid, metadata: Metadata) -> Result<()> {
        if !self.contains(cid) {
            return Err(Error::NotFound(format!("CID not in index: {}", cid)));
        }
        self.metadata_store.insert(*cid, metadata)?;
        Ok(())
    }

    /// Get statistics snapshot
    pub fn stats(&self) -> StatsSnapshot {
        self.stats.snapshot()
    }

    /// Get index health metrics
    pub fn health(&self) -> IndexHealth {
        let stats = self.stats.snapshot();
        IndexHealth::analyze(self.len(), self.config.dimension, Some(&stats))
    }

    /// Get memory usage estimate
    pub fn memory_usage(&self) -> MemoryUsage {
        MemoryUsage::estimate(
            self.len(),
            self.config.dimension,
            self.metadata_store.len(),
            self.config.cache_size,
        )
    }

    /// Get facet counts for a field
    pub fn facet_counts(&self, field: &str) -> std::collections::HashMap<String, usize> {
        self.metadata_store.get_facet_counts(field)
    }

    /// Clear the search cache
    pub fn clear_cache(&self) {
        self.cache
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        self.stats.reset();
    }

    /// Save the index to a path
    pub async fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.vector_index
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .save(path)
    }

    /// Clear all data
    pub fn clear(&self) -> Result<()> {
        // Create new empty vector index
        let new_index = VectorIndex::new(
            self.config.dimension,
            self.config.metric,
            self.config.max_connections,
            self.config.ef_construction,
        )?;

        *self.vector_index.write().unwrap_or_else(|e| e.into_inner()) = new_index;
        self.metadata_store.clear();
        self.cache
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.stats.reset();

        Ok(())
    }

    /// Prune entries older than the given TTL (time-to-live in seconds)
    ///
    /// Removes vectors and metadata for entries that were created more than
    /// `ttl_seconds` ago.
    ///
    /// # Arguments
    /// * `ttl_seconds` - Maximum age in seconds for entries to keep
    ///
    /// # Returns
    /// Number of entries pruned
    pub fn prune_by_ttl(&self, ttl_seconds: u64) -> Result<usize> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let cutoff = now.saturating_sub(ttl_seconds);

        self.prune_older_than(cutoff)
    }

    /// Prune entries created before a specific timestamp
    ///
    /// # Arguments
    /// * `timestamp` - Unix timestamp; entries created before this are removed
    ///
    /// # Returns
    /// Number of entries pruned
    pub fn prune_older_than(&self, timestamp: u64) -> Result<usize> {
        // Find CIDs to remove
        let cids_to_remove: Vec<Cid> = self
            .metadata_store
            .cids()
            .into_iter()
            .filter(|cid| {
                self.metadata_store
                    .get(cid)
                    .map(|m| m.created_at < timestamp)
                    .unwrap_or(false)
            })
            .collect();

        let count = cids_to_remove.len();

        // Remove from both indexes
        for cid in &cids_to_remove {
            let _ = self
                .vector_index
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .delete(cid);
            let _ = self.metadata_store.remove(cid);
        }

        // Clear cache since data has changed
        self.cache
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .clear();

        Ok(count)
    }

    /// Prune entries keeping only the N most recently created
    ///
    /// # Arguments
    /// * `max_entries` - Maximum number of entries to keep
    ///
    /// # Returns
    /// Number of entries pruned
    pub fn prune_to_max_entries(&self, max_entries: usize) -> Result<usize> {
        let current_count = self.len();
        if current_count <= max_entries {
            return Ok(0);
        }

        // Get all CIDs with their creation timestamps
        let mut entries: Vec<(Cid, u64)> = self
            .metadata_store
            .cids()
            .into_iter()
            .filter_map(|cid| self.metadata_store.get(&cid).map(|m| (cid, m.created_at)))
            .collect();

        // Sort by creation time (oldest first)
        entries.sort_by_key(|(_, ts)| *ts);

        // Calculate how many to remove
        let to_remove = current_count - max_entries;

        // Remove the oldest entries
        for (cid, _) in entries.iter().take(to_remove) {
            let _ = self
                .vector_index
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .delete(cid);
            let _ = self.metadata_store.remove(cid);
        }

        // Clear cache
        self.cache
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .clear();

        Ok(to_remove)
    }

    /// Prune entries by LRU (Least Recently Updated)
    ///
    /// Removes entries that haven't been updated recently, keeping
    /// only the most recently updated entries.
    ///
    /// # Arguments
    /// * `max_entries` - Maximum number of entries to keep
    ///
    /// # Returns
    /// Number of entries pruned
    pub fn prune_lru(&self, max_entries: usize) -> Result<usize> {
        let current_count = self.len();
        if current_count <= max_entries {
            return Ok(0);
        }

        // Get all CIDs with their update timestamps
        let mut entries: Vec<(Cid, u64)> = self
            .metadata_store
            .cids()
            .into_iter()
            .filter_map(|cid| self.metadata_store.get(&cid).map(|m| (cid, m.updated_at)))
            .collect();

        // Sort by update time (least recent first)
        entries.sort_by_key(|(_, ts)| *ts);

        // Calculate how many to remove
        let to_remove = current_count - max_entries;

        // Remove the least recently updated entries
        for (cid, _) in entries.iter().take(to_remove) {
            let _ = self
                .vector_index
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .delete(cid);
            let _ = self.metadata_store.remove(cid);
        }

        // Clear cache
        self.cache
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .clear();

        Ok(to_remove)
    }

    /// Get pruning statistics
    pub fn pruning_stats(&self) -> PruningStats {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let entries: Vec<(u64, u64)> = self
            .metadata_store
            .cids()
            .into_iter()
            .filter_map(|cid| {
                self.metadata_store
                    .get(&cid)
                    .map(|m| (m.created_at, m.updated_at))
            })
            .collect();

        if entries.is_empty() {
            return PruningStats::default();
        }

        let oldest_created = entries.iter().map(|(c, _)| *c).min().unwrap_or(now);
        let newest_created = entries.iter().map(|(c, _)| *c).max().unwrap_or(now);
        let oldest_updated = entries.iter().map(|(_, u)| *u).min().unwrap_or(now);

        let age_1day = entries.iter().filter(|(c, _)| now - *c < 86400).count();
        let age_7days = entries.iter().filter(|(c, _)| now - *c < 86400 * 7).count();
        let age_30days = entries
            .iter()
            .filter(|(c, _)| now - *c < 86400 * 30)
            .count();

        PruningStats {
            total_entries: entries.len(),
            oldest_entry_age: now.saturating_sub(oldest_created),
            newest_entry_age: now.saturating_sub(newest_created),
            oldest_update_age: now.saturating_sub(oldest_updated),
            entries_last_day: age_1day,
            entries_last_week: age_7days,
            entries_last_month: age_30days,
        }
    }
}

/// Pruning statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PruningStats {
    /// Total number of entries
    pub total_entries: usize,
    /// Age of the oldest entry in seconds
    pub oldest_entry_age: u64,
    /// Age of the newest entry in seconds
    pub newest_entry_age: u64,
    /// Age of the least recently updated entry in seconds
    pub oldest_update_age: u64,
    /// Number of entries created in the last day
    pub entries_last_day: usize,
    /// Number of entries created in the last week
    pub entries_last_week: usize,
    /// Number of entries created in the last month
    pub entries_last_month: usize,
}

impl PruningStats {
    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "Total: {}, Last day: {}, Last week: {}, Last month: {}, Oldest: {}s ago",
            self.total_entries,
            self.entries_last_day,
            self.entries_last_week,
            self.entries_last_month,
            self.oldest_entry_age
        )
    }

    /// Estimate entries that would be pruned for a given TTL
    pub fn would_prune_for_ttl(&self, ttl_seconds: u64) -> usize {
        // Approximate based on time buckets
        if ttl_seconds < 86400 {
            self.total_entries - self.entries_last_day
        } else if ttl_seconds < 86400 * 7 {
            self.total_entries - self.entries_last_week
        } else if ttl_seconds < 86400 * 30 {
            self.total_entries - self.entries_last_month
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::MetadataValue;

    fn test_cid(n: u8) -> Cid {
        // Use different valid CID strings
        let cids = [
            "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi",
            "bafybeiczsscdsbs7ffqz55asqdf3smv6klcw3gofszvwlyarci47bgf354",
            "bafybeibvfkifsqbapirjrj7zbfwddz5qz5awvbftjgktpcqcxjkzstszlm",
        ];
        cids[n as usize % cids.len()]
            .parse()
            .expect("test: parse test cid")
    }

    #[tokio::test]
    async fn test_hybrid_index_basic() {
        let config = HybridConfig {
            dimension: 4,
            ..Default::default()
        };

        let index = HybridIndex::new(config).expect("test: create hybrid index basic");

        let cid1 = test_cid(0);
        let vec1 = vec![1.0, 0.0, 0.0, 0.0];
        let meta1 = Metadata::new().with_string("type", "image");

        index
            .insert(&cid1, &vec1, Some(meta1))
            .expect("test: insert cid1 basic");

        assert_eq!(index.len(), 1);
        assert!(index.contains(&cid1));
    }

    #[tokio::test]
    async fn test_hybrid_search() {
        let config = HybridConfig {
            dimension: 4,
            ..Default::default()
        };

        let index = HybridIndex::new(config).expect("test: create hybrid index for search");

        // Insert some vectors with metadata (more vectors for better HNSW graph connectivity)
        let cid1 = test_cid(0);
        let vec1 = vec![1.0, 0.0, 0.0, 0.0];
        let meta1 = Metadata::new()
            .with_string("type", "image")
            .with_integer("size", 1024);

        let cid2 = test_cid(1);
        let vec2 = vec![0.9, 0.1, 0.0, 0.0];
        let meta2 = Metadata::new()
            .with_string("type", "document")
            .with_integer("size", 2048);

        let cid3 = test_cid(2);
        let vec3 = vec![0.0, 1.0, 0.0, 0.0];
        let meta3 = Metadata::new()
            .with_string("type", "audio")
            .with_integer("size", 512);

        index
            .insert(&cid1, &vec1, Some(meta1))
            .expect("test: insert cid1 for search");
        index
            .insert(&cid2, &vec2, Some(meta2))
            .expect("test: insert cid2 for search");
        index
            .insert(&cid3, &vec3, Some(meta3))
            .expect("test: insert cid3 for search");

        // Simple k-NN search with explicit ef_search to ensure results are found
        let mut query = HybridQuery::knn(vec![1.0, 0.0, 0.0, 0.0], 2);
        query.ef_search = Some(50); // Ensure we search enough candidates
        let response = index.search(query).await.expect("test: hybrid search");

        assert!(
            !response.results.is_empty(),
            "Expected at least 1 result, got {}",
            response.results.len()
        );
        // With 3 vectors and k=2, we should get 2 results
        assert!(
            !response.results.is_empty() && response.results.len() <= 2,
            "Expected 1-2 results, got {}",
            response.results.len()
        );
        // First result should be exact match (cid1)
        assert_eq!(response.results[0].cid, cid1);
    }

    #[tokio::test]
    async fn test_filtered_search() {
        let config = HybridConfig {
            dimension: 4,
            ..Default::default()
        };

        let index =
            HybridIndex::new(config).expect("test: create hybrid index for filtered search");

        let cid1 = test_cid(0);
        let vec1 = vec![1.0, 0.0, 0.0, 0.0];
        let meta1 = Metadata::new().with_string("category", "tech");

        let cid2 = test_cid(1);
        let vec2 = vec![0.9, 0.1, 0.0, 0.0];
        let meta2 = Metadata::new().with_string("category", "science");

        index
            .insert(&cid1, &vec1, Some(meta1))
            .expect("test: insert cid1 filtered");
        index
            .insert(&cid2, &vec2, Some(meta2))
            .expect("test: insert cid2 filtered");

        // Search with filter
        let filter = MetadataFilter::eq("category", MetadataValue::String("tech".to_string()));
        let query = HybridQuery::knn(vec![0.9, 0.1, 0.0, 0.0], 10).with_filter(filter);
        let response = index.search(query).await.expect("test: filtered search");

        // Should only return tech category
        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].cid, cid1);
    }

    #[tokio::test]
    async fn test_search_with_metadata() {
        let config = HybridConfig {
            dimension: 4,
            ..Default::default()
        };

        let index = HybridIndex::new(config).expect("test: create hybrid index with metadata");

        let cid1 = test_cid(0);
        let vec1 = vec![1.0, 0.0, 0.0, 0.0];
        let meta1 = Metadata::new().with_string("title", "Test Document");

        index
            .insert(&cid1, &vec1, Some(meta1))
            .expect("test: insert cid1 with metadata");

        let query = HybridQuery::knn(vec![1.0, 0.0, 0.0, 0.0], 1).with_metadata();
        let response = index
            .search(query)
            .await
            .expect("test: search with metadata");

        assert_eq!(response.results.len(), 1);
        assert!(response.results[0].metadata.is_some());

        let meta = response.results[0]
            .metadata
            .as_ref()
            .expect("test: result should have metadata");
        assert_eq!(
            meta.get("title"),
            Some(&MetadataValue::String("Test Document".to_string()))
        );
    }

    #[test]
    fn test_health_and_stats() {
        let config = HybridConfig {
            dimension: 4,
            ..Default::default()
        };

        let index = HybridIndex::new(config).expect("test: create hybrid index for health stats");

        let health = index.health();
        assert_eq!(health.size, 0);

        let stats = index.stats();
        assert_eq!(stats.search_count, 0);
    }

    #[test]
    fn test_pruning_to_max_entries() {
        let config = HybridConfig {
            dimension: 4,
            ..Default::default()
        };

        let index = HybridIndex::new(config).expect("test: create hybrid index for pruning");

        // Insert 3 entries
        for i in 0..3 {
            let cid = test_cid(i);
            let vec = vec![i as f32, 0.0, 0.0, 0.0];
            let meta = Metadata::new().with_integer("order", i as i64);
            index
                .insert(&cid, &vec, Some(meta))
                .expect("test: insert vector for pruning");
        }

        assert_eq!(index.len(), 3);

        // Prune to max 2 entries
        let pruned = index
            .prune_to_max_entries(2)
            .expect("test: prune to max entries");
        assert_eq!(pruned, 1);
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn test_pruning_stats() {
        let config = HybridConfig {
            dimension: 4,
            ..Default::default()
        };

        let index = HybridIndex::new(config).expect("test: create hybrid index for pruning stats");

        // Insert some entries
        for i in 0..3 {
            let cid = test_cid(i);
            let vec = vec![i as f32, 0.0, 0.0, 0.0];
            index
                .insert(&cid, &vec, None)
                .expect("test: insert vector for pruning stats");
        }

        let stats = index.pruning_stats();
        assert_eq!(stats.total_entries, 3);
        // All entries should be recent (created just now)
        assert_eq!(stats.entries_last_day, 3);
        assert_eq!(stats.entries_last_week, 3);
    }
}
