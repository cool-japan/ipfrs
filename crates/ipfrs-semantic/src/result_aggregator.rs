//! Multi-source search result aggregation and ranking.
//!
//! This module provides facilities for combining search results from multiple
//! sources (e.g., HNSW, DiskANN, federated peers) into a single ranked list
//! using various aggregation strategies including Reciprocal Rank Fusion,
//! score summation, weighted combination, and more.

use std::collections::HashMap;

/// Strategy for aggregating search results from multiple sources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AggregationStrategy {
    /// Sum scores across all sources for each document.
    ScoreSum,
    /// Take the maximum score across all sources for each document.
    ScoreMax,
    /// Average scores across all sources for each document.
    ScoreAverage,
    /// Reciprocal Rank Fusion: score = sum(1/(k+rank)).
    RankFusion,
    /// Weighted combination using per-source weights.
    WeightedCombination,
}

/// A single search result from one source.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Unique document identifier.
    pub doc_id: String,
    /// Relevance score (higher is better).
    pub score: f64,
    /// Name of the source that produced this result.
    pub source: String,
    /// Arbitrary metadata associated with the result.
    pub metadata: HashMap<String, String>,
}

/// An aggregated result combining information from multiple sources.
#[derive(Debug, Clone)]
pub struct AggregatedResult {
    /// Unique document identifier.
    pub doc_id: String,
    /// Final aggregated score.
    pub final_score: f64,
    /// List of sources that contributed to this result.
    pub sources: Vec<String>,
    /// Per-source scores as (source_name, score) pairs.
    pub source_scores: Vec<(String, f64)>,
    /// Rank in the final result list (1-based).
    pub rank: usize,
}

/// Configuration for the result aggregator.
#[derive(Debug, Clone)]
pub struct AggregatorConfig {
    /// Strategy used to combine scores.
    pub strategy: AggregationStrategy,
    /// Maximum number of results to return.
    pub max_results: usize,
    /// Minimum score threshold; results below this are filtered out.
    pub min_score_threshold: f64,
    /// Per-source weights for `WeightedCombination` strategy.
    pub source_weights: HashMap<String, f64>,
    /// RRF constant k (default 60.0). Higher values reduce the impact of rank differences.
    pub rrf_k: f64,
}

impl Default for AggregatorConfig {
    fn default() -> Self {
        Self {
            strategy: AggregationStrategy::RankFusion,
            max_results: 100,
            min_score_threshold: 0.0,
            source_weights: HashMap::new(),
            rrf_k: 60.0,
        }
    }
}

/// Tracks statistics about aggregation operations.
#[derive(Debug, Clone, Default)]
pub struct AggregatorStats {
    /// Total number of aggregation operations performed.
    pub aggregations_performed: u64,
    /// Total number of input results across all aggregations.
    pub total_input_results: u64,
    /// Total number of output results across all aggregations.
    pub total_output_results: u64,
    /// Running average compression ratio (input/output).
    pub avg_compression_ratio: f64,
}

/// Multi-source search result aggregator.
///
/// Collects search results from multiple sources and combines them using
/// a configurable aggregation strategy.
pub struct ResultAggregator {
    config: AggregatorConfig,
    result_sets: HashMap<String, Vec<SearchResult>>,
    stats: AggregatorStats,
}

impl ResultAggregator {
    /// Create a new `ResultAggregator` with the given configuration.
    pub fn new(config: AggregatorConfig) -> Self {
        Self {
            config,
            result_sets: HashMap::new(),
            stats: AggregatorStats::default(),
        }
    }

    /// Add a batch of results from a named source.
    pub fn add_results(&mut self, source: &str, results: Vec<SearchResult>) {
        self.result_sets
            .entry(source.to_string())
            .or_default()
            .extend(results);
    }

    /// Aggregate all added result sets using the configured strategy.
    ///
    /// Returns a sorted, deduplicated, ranked list of aggregated results.
    pub fn aggregate(&mut self) -> Vec<AggregatedResult> {
        let input_count: u64 = self.result_sets.values().map(|v| v.len() as u64).sum();

        let mut results = match self.config.strategy {
            AggregationStrategy::ScoreSum => Self::aggregate_score_sum(&self.result_sets),
            AggregationStrategy::ScoreMax => Self::aggregate_score_max(&self.result_sets),
            AggregationStrategy::ScoreAverage => Self::aggregate_score_avg(&self.result_sets),
            AggregationStrategy::RankFusion => {
                Self::aggregate_rrf(&self.result_sets, self.config.rrf_k)
            }
            AggregationStrategy::WeightedCombination => {
                Self::aggregate_weighted(&self.result_sets, &self.config.source_weights)
            }
        };

        self.apply_threshold(&mut results);

        // Sort descending by score, then truncate
        results.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if results.len() > self.config.max_results {
            results.truncate(self.config.max_results);
        }

        // Assign ranks (1-based)
        for (i, r) in results.iter_mut().enumerate() {
            r.rank = i + 1;
        }

        // Update stats
        let output_count = results.len() as u64;
        self.stats.aggregations_performed += 1;
        self.stats.total_input_results += input_count;
        self.stats.total_output_results += output_count;

        let n = self.stats.aggregations_performed as f64;
        let ratio = if output_count > 0 {
            input_count as f64 / output_count as f64
        } else if input_count > 0 {
            input_count as f64
        } else {
            1.0
        };
        // Running average
        self.stats.avg_compression_ratio =
            self.stats.avg_compression_ratio * ((n - 1.0) / n) + ratio / n;

        results
    }

    /// Reciprocal Rank Fusion aggregation.
    ///
    /// For each source, results are ranked by descending score. The RRF score
    /// for a document is `sum(1 / (k + rank))` across all sources.
    pub fn aggregate_rrf(
        result_sets: &HashMap<String, Vec<SearchResult>>,
        k: f64,
    ) -> Vec<AggregatedResult> {
        // Build per-source rankings (sorted descending by score)
        let mut doc_rrf_scores: HashMap<String, f64> = HashMap::new();
        let mut doc_sources: HashMap<String, Vec<String>> = HashMap::new();
        let mut doc_source_scores: HashMap<String, Vec<(String, f64)>> = HashMap::new();

        for (source, results) in result_sets {
            // Sort by descending score to determine rank
            let mut sorted: Vec<&SearchResult> = results.iter().collect();
            sorted.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Deduplicate within source (keep highest score)
            let mut seen_in_source: HashMap<&str, bool> = HashMap::new();
            let mut rank: usize = 0;

            for res in &sorted {
                if seen_in_source.contains_key(res.doc_id.as_str()) {
                    continue;
                }
                seen_in_source.insert(&res.doc_id, true);
                rank += 1;

                let rrf_score = 1.0 / (k + rank as f64);
                *doc_rrf_scores.entry(res.doc_id.clone()).or_default() += rrf_score;

                doc_sources
                    .entry(res.doc_id.clone())
                    .or_default()
                    .push(source.clone());

                doc_source_scores
                    .entry(res.doc_id.clone())
                    .or_default()
                    .push((source.clone(), res.score));
            }
        }

        Self::build_aggregated_results(doc_rrf_scores, doc_sources, doc_source_scores)
    }

    /// Sum scores across all sources for each document.
    pub fn aggregate_score_sum(
        result_sets: &HashMap<String, Vec<SearchResult>>,
    ) -> Vec<AggregatedResult> {
        let mut doc_scores: HashMap<String, f64> = HashMap::new();
        let mut doc_sources: HashMap<String, Vec<String>> = HashMap::new();
        let mut doc_source_scores: HashMap<String, Vec<(String, f64)>> = HashMap::new();

        for (source, results) in result_sets {
            let mut seen: HashMap<&str, bool> = HashMap::new();
            for res in results {
                if seen.contains_key(res.doc_id.as_str()) {
                    continue;
                }
                seen.insert(&res.doc_id, true);

                *doc_scores.entry(res.doc_id.clone()).or_default() += res.score;

                doc_sources
                    .entry(res.doc_id.clone())
                    .or_default()
                    .push(source.clone());

                doc_source_scores
                    .entry(res.doc_id.clone())
                    .or_default()
                    .push((source.clone(), res.score));
            }
        }

        Self::build_aggregated_results(doc_scores, doc_sources, doc_source_scores)
    }

    /// Take the maximum score across all sources for each document.
    pub fn aggregate_score_max(
        result_sets: &HashMap<String, Vec<SearchResult>>,
    ) -> Vec<AggregatedResult> {
        let mut doc_scores: HashMap<String, f64> = HashMap::new();
        let mut doc_sources: HashMap<String, Vec<String>> = HashMap::new();
        let mut doc_source_scores: HashMap<String, Vec<(String, f64)>> = HashMap::new();

        for (source, results) in result_sets {
            let mut seen: HashMap<&str, bool> = HashMap::new();
            for res in results {
                if seen.contains_key(res.doc_id.as_str()) {
                    continue;
                }
                seen.insert(&res.doc_id, true);

                let entry = doc_scores
                    .entry(res.doc_id.clone())
                    .or_insert(f64::NEG_INFINITY);
                if res.score > *entry {
                    *entry = res.score;
                }

                doc_sources
                    .entry(res.doc_id.clone())
                    .or_default()
                    .push(source.clone());

                doc_source_scores
                    .entry(res.doc_id.clone())
                    .or_default()
                    .push((source.clone(), res.score));
            }
        }

        Self::build_aggregated_results(doc_scores, doc_sources, doc_source_scores)
    }

    /// Average scores across all sources for each document.
    pub fn aggregate_score_avg(
        result_sets: &HashMap<String, Vec<SearchResult>>,
    ) -> Vec<AggregatedResult> {
        let mut doc_score_sums: HashMap<String, f64> = HashMap::new();
        let mut doc_score_counts: HashMap<String, usize> = HashMap::new();
        let mut doc_sources: HashMap<String, Vec<String>> = HashMap::new();
        let mut doc_source_scores: HashMap<String, Vec<(String, f64)>> = HashMap::new();

        for (source, results) in result_sets {
            let mut seen: HashMap<&str, bool> = HashMap::new();
            for res in results {
                if seen.contains_key(res.doc_id.as_str()) {
                    continue;
                }
                seen.insert(&res.doc_id, true);

                *doc_score_sums.entry(res.doc_id.clone()).or_default() += res.score;
                *doc_score_counts.entry(res.doc_id.clone()).or_default() += 1;

                doc_sources
                    .entry(res.doc_id.clone())
                    .or_default()
                    .push(source.clone());

                doc_source_scores
                    .entry(res.doc_id.clone())
                    .or_default()
                    .push((source.clone(), res.score));
            }
        }

        let doc_scores: HashMap<String, f64> = doc_score_sums
            .into_iter()
            .map(|(doc_id, sum)| {
                let count = doc_score_counts.get(&doc_id).copied().unwrap_or(1);
                (doc_id, sum / count as f64)
            })
            .collect();

        Self::build_aggregated_results(doc_scores, doc_sources, doc_source_scores)
    }

    /// Weighted combination of scores using per-source weights.
    ///
    /// If a source has no configured weight, it defaults to 1.0.
    pub fn aggregate_weighted(
        result_sets: &HashMap<String, Vec<SearchResult>>,
        weights: &HashMap<String, f64>,
    ) -> Vec<AggregatedResult> {
        let mut doc_scores: HashMap<String, f64> = HashMap::new();
        let mut doc_sources: HashMap<String, Vec<String>> = HashMap::new();
        let mut doc_source_scores: HashMap<String, Vec<(String, f64)>> = HashMap::new();

        for (source, results) in result_sets {
            let weight = weights.get(source).copied().unwrap_or(1.0);
            let mut seen: HashMap<&str, bool> = HashMap::new();

            for res in results {
                if seen.contains_key(res.doc_id.as_str()) {
                    continue;
                }
                seen.insert(&res.doc_id, true);

                *doc_scores.entry(res.doc_id.clone()).or_default() += res.score * weight;

                doc_sources
                    .entry(res.doc_id.clone())
                    .or_default()
                    .push(source.clone());

                doc_source_scores
                    .entry(res.doc_id.clone())
                    .or_default()
                    .push((source.clone(), res.score));
            }
        }

        Self::build_aggregated_results(doc_scores, doc_sources, doc_source_scores)
    }

    /// Remove all result sets.
    pub fn clear(&mut self) {
        self.result_sets.clear();
    }

    /// Return the number of distinct sources currently held.
    pub fn source_count(&self) -> usize {
        self.result_sets.len()
    }

    /// Return the total number of individual results across all sources.
    pub fn total_results(&self) -> usize {
        self.result_sets.values().map(|v| v.len()).sum()
    }

    /// Return a reference to the current aggregation statistics.
    pub fn stats(&self) -> &AggregatorStats {
        &self.stats
    }

    /// Filter out results whose `final_score` is below the configured threshold.
    pub fn apply_threshold(&self, results: &mut Vec<AggregatedResult>) {
        results.retain(|r| r.final_score >= self.config.min_score_threshold);
    }

    // ---- internal helpers ----

    /// Build the final `AggregatedResult` list from the intermediate maps.
    fn build_aggregated_results(
        doc_scores: HashMap<String, f64>,
        doc_sources: HashMap<String, Vec<String>>,
        doc_source_scores: HashMap<String, Vec<(String, f64)>>,
    ) -> Vec<AggregatedResult> {
        let mut results: Vec<AggregatedResult> = doc_scores
            .into_iter()
            .map(|(doc_id, final_score)| {
                let sources = doc_sources.get(&doc_id).cloned().unwrap_or_default();
                let source_scores = doc_source_scores.get(&doc_id).cloned().unwrap_or_default();
                AggregatedResult {
                    doc_id,
                    final_score,
                    sources,
                    source_scores,
                    rank: 0, // assigned later
                }
            })
            .collect();

        // Sort descending by score for consistent ordering
        results.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(doc_id: &str, score: f64, source: &str) -> SearchResult {
        SearchResult {
            doc_id: doc_id.to_string(),
            score,
            source: source.to_string(),
            metadata: HashMap::new(),
        }
    }

    fn make_result_with_meta(
        doc_id: &str,
        score: f64,
        source: &str,
        meta: Vec<(&str, &str)>,
    ) -> SearchResult {
        let metadata: HashMap<String, String> = meta
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        SearchResult {
            doc_id: doc_id.to_string(),
            score,
            source: source.to_string(),
            metadata,
        }
    }

    // ---- ScoreSum strategy ----

    #[test]
    fn test_score_sum_basic() {
        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::ScoreSum,
            ..Default::default()
        });
        agg.add_results(
            "a",
            vec![make_result("d1", 0.5, "a"), make_result("d2", 0.3, "a")],
        );
        agg.add_results(
            "b",
            vec![make_result("d1", 0.4, "b"), make_result("d3", 0.6, "b")],
        );

        let results = agg.aggregate();
        let d1 = results
            .iter()
            .find(|r| r.doc_id == "d1")
            .expect("d1 present");
        assert!(
            (d1.final_score - 0.9).abs() < 1e-9,
            "d1 sum = 0.5+0.4 = 0.9"
        );
    }

    #[test]
    fn test_score_sum_single_source() {
        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::ScoreSum,
            ..Default::default()
        });
        agg.add_results("a", vec![make_result("d1", 0.7, "a")]);
        let results = agg.aggregate();
        assert_eq!(results.len(), 1);
        assert!((results[0].final_score - 0.7).abs() < 1e-9);
    }

    // ---- ScoreMax strategy ----

    #[test]
    fn test_score_max_basic() {
        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::ScoreMax,
            ..Default::default()
        });
        agg.add_results("a", vec![make_result("d1", 0.5, "a")]);
        agg.add_results("b", vec![make_result("d1", 0.9, "b")]);

        let results = agg.aggregate();
        let d1 = results
            .iter()
            .find(|r| r.doc_id == "d1")
            .expect("d1 present");
        assert!((d1.final_score - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_score_max_picks_highest() {
        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::ScoreMax,
            ..Default::default()
        });
        agg.add_results("a", vec![make_result("d1", 0.1, "a")]);
        agg.add_results("b", vec![make_result("d1", 0.3, "b")]);
        agg.add_results("c", vec![make_result("d1", 0.2, "c")]);

        let results = agg.aggregate();
        assert!((results[0].final_score - 0.3).abs() < 1e-9);
    }

    // ---- ScoreAverage strategy ----

    #[test]
    fn test_score_avg_basic() {
        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::ScoreAverage,
            ..Default::default()
        });
        agg.add_results("a", vec![make_result("d1", 0.6, "a")]);
        agg.add_results("b", vec![make_result("d1", 0.4, "b")]);

        let results = agg.aggregate();
        let d1 = results
            .iter()
            .find(|r| r.doc_id == "d1")
            .expect("d1 present");
        assert!((d1.final_score - 0.5).abs() < 1e-9, "avg(0.6, 0.4) = 0.5");
    }

    #[test]
    fn test_score_avg_three_sources() {
        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::ScoreAverage,
            ..Default::default()
        });
        agg.add_results("a", vec![make_result("d1", 0.3, "a")]);
        agg.add_results("b", vec![make_result("d1", 0.6, "b")]);
        agg.add_results("c", vec![make_result("d1", 0.9, "c")]);

        let results = agg.aggregate();
        assert!((results[0].final_score - 0.6).abs() < 1e-9);
    }

    // ---- RRF strategy ----

    #[test]
    fn test_rrf_formula() {
        // With k=60, rank 1 gives 1/(60+1) = 1/61
        let mut sets: HashMap<String, Vec<SearchResult>> = HashMap::new();
        sets.insert("a".to_string(), vec![make_result("d1", 1.0, "a")]);

        let results = ResultAggregator::aggregate_rrf(&sets, 60.0);
        let expected = 1.0 / 61.0;
        assert!(
            (results[0].final_score - expected).abs() < 1e-9,
            "RRF score for rank-1 doc with k=60 should be 1/61"
        );
    }

    #[test]
    fn test_rrf_multi_source() {
        let mut sets: HashMap<String, Vec<SearchResult>> = HashMap::new();
        // d1 is rank 1 in both sources
        sets.insert(
            "a".to_string(),
            vec![make_result("d1", 1.0, "a"), make_result("d2", 0.5, "a")],
        );
        sets.insert(
            "b".to_string(),
            vec![make_result("d1", 0.9, "b"), make_result("d3", 0.8, "b")],
        );

        let results = ResultAggregator::aggregate_rrf(&sets, 60.0);
        let d1 = results
            .iter()
            .find(|r| r.doc_id == "d1")
            .expect("d1 present");
        let expected = 2.0 / 61.0; // rank 1 in both sources
        assert!((d1.final_score - expected).abs() < 1e-9);
    }

    #[test]
    fn test_rrf_respects_k_parameter() {
        let mut sets: HashMap<String, Vec<SearchResult>> = HashMap::new();
        sets.insert("a".to_string(), vec![make_result("d1", 1.0, "a")]);

        let results_low_k = ResultAggregator::aggregate_rrf(&sets, 10.0);
        let results_high_k = ResultAggregator::aggregate_rrf(&sets, 100.0);

        // Lower k means higher score for the same rank
        assert!(results_low_k[0].final_score > results_high_k[0].final_score);
    }

    #[test]
    fn test_rrf_rank_ordering() {
        // d1 has score 1.0 (rank 1), d2 has score 0.5 (rank 2)
        let mut sets: HashMap<String, Vec<SearchResult>> = HashMap::new();
        sets.insert(
            "a".to_string(),
            vec![make_result("d1", 1.0, "a"), make_result("d2", 0.5, "a")],
        );

        let results = ResultAggregator::aggregate_rrf(&sets, 60.0);
        assert!(results[0].final_score > results[1].final_score);
        // rank 1: 1/61, rank 2: 1/62
        let expected_d1 = 1.0 / 61.0;
        let expected_d2 = 1.0 / 62.0;
        assert!((results[0].final_score - expected_d1).abs() < 1e-9);
        assert!((results[1].final_score - expected_d2).abs() < 1e-9);
    }

    // ---- WeightedCombination strategy ----

    #[test]
    fn test_weighted_basic() {
        let mut weights = HashMap::new();
        weights.insert("a".to_string(), 2.0);
        weights.insert("b".to_string(), 1.0);

        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::WeightedCombination,
            source_weights: weights,
            ..Default::default()
        });
        agg.add_results("a", vec![make_result("d1", 0.5, "a")]);
        agg.add_results("b", vec![make_result("d1", 0.5, "b")]);

        let results = agg.aggregate();
        let d1 = results
            .iter()
            .find(|r| r.doc_id == "d1")
            .expect("d1 present");
        // 0.5*2.0 + 0.5*1.0 = 1.5
        assert!((d1.final_score - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_default_weight() {
        // Source not in weights map should default to 1.0
        let weights = HashMap::new();

        let mut sets: HashMap<String, Vec<SearchResult>> = HashMap::new();
        sets.insert(
            "unknown".to_string(),
            vec![make_result("d1", 0.7, "unknown")],
        );

        let results = ResultAggregator::aggregate_weighted(&sets, &weights);
        assert!((results[0].final_score - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_zero_weight() {
        let mut weights = HashMap::new();
        weights.insert("a".to_string(), 0.0);

        let mut sets: HashMap<String, Vec<SearchResult>> = HashMap::new();
        sets.insert("a".to_string(), vec![make_result("d1", 0.9, "a")]);

        let results = ResultAggregator::aggregate_weighted(&sets, &weights);
        assert!(
            (results[0].final_score).abs() < 1e-9,
            "zero weight => zero score"
        );
    }

    // ---- Threshold filtering ----

    #[test]
    fn test_threshold_filters_low_scores() {
        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::ScoreSum,
            min_score_threshold: 0.5,
            ..Default::default()
        });
        agg.add_results(
            "a",
            vec![
                make_result("d1", 0.8, "a"),
                make_result("d2", 0.3, "a"),
                make_result("d3", 0.5, "a"),
            ],
        );

        let results = agg.aggregate();
        assert_eq!(results.len(), 2); // d2 filtered out
        assert!(results.iter().all(|r| r.final_score >= 0.5));
    }

    #[test]
    fn test_threshold_zero_passes_all() {
        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::ScoreSum,
            min_score_threshold: 0.0,
            ..Default::default()
        });
        agg.add_results("a", vec![make_result("d1", 0.001, "a")]);
        let results = agg.aggregate();
        assert_eq!(results.len(), 1);
    }

    // ---- max_results limit ----

    #[test]
    fn test_max_results_truncation() {
        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::ScoreSum,
            max_results: 2,
            ..Default::default()
        });
        agg.add_results(
            "a",
            vec![
                make_result("d1", 0.9, "a"),
                make_result("d2", 0.8, "a"),
                make_result("d3", 0.7, "a"),
            ],
        );

        let results = agg.aggregate();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].doc_id, "d1");
        assert_eq!(results[1].doc_id, "d2");
    }

    // ---- Deduplication by doc_id ----

    #[test]
    fn test_dedup_within_source() {
        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::ScoreSum,
            ..Default::default()
        });
        // Same doc_id twice in one source (first occurrence wins for that source)
        agg.add_results(
            "a",
            vec![make_result("d1", 0.5, "a"), make_result("d1", 0.3, "a")],
        );
        let results = agg.aggregate();
        assert_eq!(results.len(), 1);
        // Only the first d1 from source "a" is counted
        assert!((results[0].final_score - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_dedup_across_sources_merges() {
        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::ScoreSum,
            ..Default::default()
        });
        agg.add_results("a", vec![make_result("d1", 0.5, "a")]);
        agg.add_results("b", vec![make_result("d1", 0.4, "b")]);

        let results = agg.aggregate();
        assert_eq!(results.len(), 1); // merged into one
        assert!(results[0].sources.len() >= 2);
    }

    // ---- Empty sources ----

    #[test]
    fn test_empty_no_sources() {
        let mut agg = ResultAggregator::new(AggregatorConfig::default());
        let results = agg.aggregate();
        assert!(results.is_empty());
    }

    #[test]
    fn test_empty_source_list() {
        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::ScoreSum,
            ..Default::default()
        });
        agg.add_results("a", vec![]);
        let results = agg.aggregate();
        assert!(results.is_empty());
    }

    // ---- Single source passthrough ----

    #[test]
    fn test_single_source_passthrough() {
        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::ScoreSum,
            ..Default::default()
        });
        agg.add_results(
            "only",
            vec![
                make_result("d1", 0.9, "only"),
                make_result("d2", 0.7, "only"),
            ],
        );
        let results = agg.aggregate();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].doc_id, "d1");
        assert_eq!(results[1].doc_id, "d2");
        assert_eq!(results[0].sources, vec!["only"]);
    }

    // ---- Stats tracking ----

    #[test]
    fn test_stats_initial() {
        let agg = ResultAggregator::new(AggregatorConfig::default());
        assert_eq!(agg.stats().aggregations_performed, 0);
        assert_eq!(agg.stats().total_input_results, 0);
        assert_eq!(agg.stats().total_output_results, 0);
    }

    #[test]
    fn test_stats_after_aggregate() {
        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::ScoreSum,
            ..Default::default()
        });
        agg.add_results(
            "a",
            vec![make_result("d1", 0.5, "a"), make_result("d2", 0.3, "a")],
        );
        agg.add_results("b", vec![make_result("d1", 0.4, "b")]);

        let _results = agg.aggregate();
        assert_eq!(agg.stats().aggregations_performed, 1);
        assert_eq!(agg.stats().total_input_results, 3);
        assert_eq!(agg.stats().total_output_results, 2);
    }

    #[test]
    fn test_stats_multiple_aggregations() {
        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::ScoreSum,
            ..Default::default()
        });

        agg.add_results("a", vec![make_result("d1", 0.5, "a")]);
        let _r1 = agg.aggregate();

        agg.clear();
        agg.add_results(
            "b",
            vec![make_result("d2", 0.8, "b"), make_result("d3", 0.3, "b")],
        );
        let _r2 = agg.aggregate();

        assert_eq!(agg.stats().aggregations_performed, 2);
        assert_eq!(agg.stats().total_input_results, 3); // 1 + 2
        assert_eq!(agg.stats().total_output_results, 3); // 1 + 2
    }

    // ---- Source weights ----

    #[test]
    fn test_source_weights_high_boost() {
        let mut weights = HashMap::new();
        weights.insert("premium".to_string(), 10.0);
        weights.insert("basic".to_string(), 1.0);

        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::WeightedCombination,
            source_weights: weights,
            ..Default::default()
        });
        agg.add_results("premium", vec![make_result("d1", 0.3, "premium")]);
        agg.add_results("basic", vec![make_result("d2", 0.9, "basic")]);

        let results = agg.aggregate();
        // d1: 0.3*10 = 3.0, d2: 0.9*1 = 0.9
        assert_eq!(results[0].doc_id, "d1");
        assert!((results[0].final_score - 3.0).abs() < 1e-9);
    }

    // ---- Clear and re-aggregate ----

    #[test]
    fn test_clear_resets_results() {
        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::ScoreSum,
            ..Default::default()
        });
        agg.add_results("a", vec![make_result("d1", 0.5, "a")]);
        assert_eq!(agg.source_count(), 1);
        assert_eq!(agg.total_results(), 1);

        agg.clear();
        assert_eq!(agg.source_count(), 0);
        assert_eq!(agg.total_results(), 0);

        let results = agg.aggregate();
        assert!(results.is_empty());
    }

    #[test]
    fn test_clear_and_reaggregate() {
        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::ScoreSum,
            ..Default::default()
        });

        agg.add_results("a", vec![make_result("d1", 0.5, "a")]);
        let r1 = agg.aggregate();
        assert_eq!(r1.len(), 1);

        agg.clear();
        agg.add_results("b", vec![make_result("d2", 0.8, "b")]);
        let r2 = agg.aggregate();
        assert_eq!(r2.len(), 1);
        assert_eq!(r2[0].doc_id, "d2");
    }

    // ---- Rank assignment ----

    #[test]
    fn test_rank_assignment() {
        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::ScoreSum,
            ..Default::default()
        });
        agg.add_results(
            "a",
            vec![
                make_result("d1", 0.9, "a"),
                make_result("d2", 0.7, "a"),
                make_result("d3", 0.5, "a"),
            ],
        );

        let results = agg.aggregate();
        assert_eq!(results[0].rank, 1);
        assert_eq!(results[1].rank, 2);
        assert_eq!(results[2].rank, 3);
    }

    // ---- Multi-source merge ----

    #[test]
    fn test_multi_source_merge_three() {
        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::ScoreSum,
            ..Default::default()
        });
        agg.add_results("a", vec![make_result("d1", 0.3, "a")]);
        agg.add_results("b", vec![make_result("d1", 0.3, "b")]);
        agg.add_results("c", vec![make_result("d1", 0.3, "c")]);

        let results = agg.aggregate();
        assert_eq!(results.len(), 1);
        assert!((results[0].final_score - 0.9).abs() < 1e-9);
        assert_eq!(results[0].sources.len(), 3);
    }

    // ---- source_count and total_results ----

    #[test]
    fn test_source_count() {
        let mut agg = ResultAggregator::new(AggregatorConfig::default());
        assert_eq!(agg.source_count(), 0);
        agg.add_results("a", vec![make_result("d1", 0.5, "a")]);
        assert_eq!(agg.source_count(), 1);
        agg.add_results("b", vec![make_result("d2", 0.3, "b")]);
        assert_eq!(agg.source_count(), 2);
    }

    #[test]
    fn test_total_results() {
        let mut agg = ResultAggregator::new(AggregatorConfig::default());
        assert_eq!(agg.total_results(), 0);
        agg.add_results(
            "a",
            vec![make_result("d1", 0.5, "a"), make_result("d2", 0.3, "a")],
        );
        assert_eq!(agg.total_results(), 2);
        agg.add_results("b", vec![make_result("d3", 0.4, "b")]);
        assert_eq!(agg.total_results(), 3);
    }

    // ---- Metadata preservation ----

    #[test]
    fn test_metadata_preserved() {
        let r = make_result_with_meta("d1", 0.5, "a", vec![("key", "value")]);
        assert_eq!(r.metadata.get("key").map(|s| s.as_str()), Some("value"));
    }

    // ---- source_scores tracking ----

    #[test]
    fn test_source_scores_tracked() {
        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::ScoreSum,
            ..Default::default()
        });
        agg.add_results("a", vec![make_result("d1", 0.5, "a")]);
        agg.add_results("b", vec![make_result("d1", 0.3, "b")]);

        let results = agg.aggregate();
        let d1 = results
            .iter()
            .find(|r| r.doc_id == "d1")
            .expect("d1 present");
        assert_eq!(d1.source_scores.len(), 2);
    }

    // ---- Default config ----

    #[test]
    fn test_default_config() {
        let config = AggregatorConfig::default();
        assert_eq!(config.strategy, AggregationStrategy::RankFusion);
        assert_eq!(config.max_results, 100);
        assert!((config.min_score_threshold).abs() < 1e-9);
        assert!((config.rrf_k - 60.0).abs() < 1e-9);
    }

    // ---- Aggregation strategy via aggregate() dispatch ----

    #[test]
    fn test_aggregate_dispatches_rrf() {
        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::RankFusion,
            rrf_k: 60.0,
            ..Default::default()
        });
        agg.add_results("a", vec![make_result("d1", 1.0, "a")]);
        let results = agg.aggregate();
        let expected = 1.0 / 61.0;
        assert!((results[0].final_score - expected).abs() < 1e-9);
    }

    #[test]
    fn test_aggregate_dispatches_weighted() {
        let mut weights = HashMap::new();
        weights.insert("a".to_string(), 3.0);

        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::WeightedCombination,
            source_weights: weights,
            ..Default::default()
        });
        agg.add_results("a", vec![make_result("d1", 0.5, "a")]);
        let results = agg.aggregate();
        assert!((results[0].final_score - 1.5).abs() < 1e-9);
    }

    // ---- Compression ratio ----

    #[test]
    fn test_compression_ratio() {
        let mut agg = ResultAggregator::new(AggregatorConfig {
            strategy: AggregationStrategy::ScoreSum,
            ..Default::default()
        });
        // 4 input results, 2 unique doc_ids => output 2 => ratio 4/2 = 2.0
        agg.add_results(
            "a",
            vec![make_result("d1", 0.5, "a"), make_result("d2", 0.3, "a")],
        );
        agg.add_results(
            "b",
            vec![make_result("d1", 0.4, "b"), make_result("d2", 0.2, "b")],
        );
        let _results = agg.aggregate();
        assert!((agg.stats().avg_compression_ratio - 2.0).abs() < 1e-9);
    }
}
