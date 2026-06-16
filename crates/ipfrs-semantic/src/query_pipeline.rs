//! Semantic Query Pipeline — composable multi-stage query processing.
//!
//! Chains pre-processing, expansion, retrieval simulation, ranking, and
//! post-filtering into a single, configurable pipeline that records
//! per-stage metrics for every run.

// ---------------------------------------------------------------------------
// PipelineStageKind
// ---------------------------------------------------------------------------

/// Identifies the kind of processing performed by a pipeline stage.
#[derive(Clone, Debug, PartialEq)]
pub enum PipelineStageKind {
    /// Normalise / clean the query embedding (L2-normalisation).
    Preprocess,
    /// Generate `max_variants` synthetic query variants from the embedding.
    Expand { max_variants: usize },
    /// Simulate retrieval and return `top_k` scored results.
    Retrieve { top_k: usize },
    /// Re-rank all results by score descending.
    Rank,
    /// Discard results whose score falls below `min_score`.
    Filter { min_score: f32 },
}

// ---------------------------------------------------------------------------
// QueryResult
// ---------------------------------------------------------------------------

/// A single result emitted or transformed by the pipeline.
#[derive(Clone, Debug)]
pub struct QueryResult {
    /// Opaque numeric identifier for this result.
    pub result_id: u64,
    /// Relevance score assigned by the producing stage.
    pub score: f32,
    /// Content Identifier string for the matching document / chunk.
    pub cid: String,
    /// Name of the stage that introduced this result into the pipeline.
    pub stage_added: String,
}

// ---------------------------------------------------------------------------
// StageMetrics
// ---------------------------------------------------------------------------

/// Execution metrics collected for a single pipeline stage during one run.
#[derive(Clone, Debug)]
pub struct StageMetrics {
    /// Human-readable name of the stage (matches its variant name).
    pub stage_name: String,
    /// Number of results entering this stage.
    pub input_count: usize,
    /// Number of results leaving this stage.
    pub output_count: usize,
    /// Simulated duration expressed as the zero-based stage index.
    pub duration_ticks: u64,
}

// ---------------------------------------------------------------------------
// PipelineRun
// ---------------------------------------------------------------------------

/// The complete output of a single pipeline execution.
pub struct PipelineRun {
    /// The (possibly normalised) query embedding used for this run.
    pub query_embedding: Vec<f32>,
    /// Results that survived all stages.
    pub results: Vec<QueryResult>,
    /// Per-stage metrics recorded during the run.
    pub stage_metrics: Vec<StageMetrics>,
    /// Total number of stages executed.
    pub total_stages: usize,
}

impl PipelineRun {
    /// Returns the number of results in this run.
    pub fn result_count(&self) -> usize {
        self.results.len()
    }

    /// Returns the result with the highest score, or `None` if results is empty.
    pub fn top_result(&self) -> Option<&QueryResult> {
        self.results.iter().max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

// ---------------------------------------------------------------------------
// PipelineConfig
// ---------------------------------------------------------------------------

/// Describes the ordered sequence of stages that form the pipeline.
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// Ordered list of stages to execute on each run.
    pub stages: Vec<PipelineStageKind>,
}

impl PipelineConfig {
    /// Returns the number of stages configured.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

// ---------------------------------------------------------------------------
// PipelineStats
// ---------------------------------------------------------------------------

/// Cumulative statistics aggregated across all runs of a pipeline instance.
#[derive(Clone, Debug, Default)]
pub struct PipelineStats {
    /// Total number of times `run()` has been called.
    pub total_runs: u64,
    /// Total number of results returned across all runs.
    pub total_results_returned: u64,
    /// Average number of results returned per run.
    pub avg_results_per_run: f64,
}

// ---------------------------------------------------------------------------
// SemanticQueryPipeline
// ---------------------------------------------------------------------------

/// A composable pipeline that chains multiple semantic query processing stages.
///
/// Stages are executed in the order defined by [`PipelineConfig`]. Metrics are
/// recorded for every stage on every run, and cumulative statistics are kept
/// across all invocations of [`run`](Self::run).
pub struct SemanticQueryPipeline {
    /// Configuration describing which stages to run and their parameters.
    pub config: PipelineConfig,
    /// Running statistics updated after each call to [`run`](Self::run).
    pub stats: PipelineStats,
}

impl SemanticQueryPipeline {
    /// Create a new pipeline with the given configuration and zeroed statistics.
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            stats: PipelineStats::default(),
        }
    }

    /// Execute the pipeline for one query embedding, collecting per-stage metrics.
    ///
    /// The `query_embedding` is consumed and may be modified in-place by the
    /// `Preprocess` stage before being stored in the returned [`PipelineRun`].
    pub fn run(&mut self, query_embedding: Vec<f32>) -> PipelineRun {
        let mut embedding = query_embedding;
        let mut results: Vec<QueryResult> = Vec::new();
        let mut stage_metrics: Vec<StageMetrics> = Vec::new();
        let total_stages = self.config.stages.len();

        for (stage_index, stage) in self.config.stages.clone().iter().enumerate() {
            let input_count = results.len();
            let tick = stage_index as u64;

            match stage {
                PipelineStageKind::Preprocess => {
                    // L2-normalise the query embedding in-place.
                    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 1e-9 {
                        for v in embedding.iter_mut() {
                            *v /= norm;
                        }
                    }
                    let output_count = results.len();
                    stage_metrics.push(StageMetrics {
                        stage_name: "Preprocess".to_string(),
                        input_count,
                        output_count,
                        duration_ticks: tick,
                    });
                }

                PipelineStageKind::Expand { max_variants } => {
                    for i in 0..*max_variants {
                        results.push(QueryResult {
                            result_id: i as u64,
                            score: 0.8 - (i as f32 * 0.05),
                            cid: format!("expand_{i}"),
                            stage_added: "Expand".to_string(),
                        });
                    }
                    let output_count = results.len();
                    stage_metrics.push(StageMetrics {
                        stage_name: "Expand".to_string(),
                        input_count,
                        output_count,
                        duration_ticks: tick,
                    });
                }

                PipelineStageKind::Retrieve { top_k } => {
                    for i in 0..*top_k {
                        results.push(QueryResult {
                            result_id: i as u64,
                            score: 1.0 - (i as f32 * 0.1),
                            cid: format!("retrieve_{i}"),
                            stage_added: "Retrieve".to_string(),
                        });
                    }
                    let output_count = results.len();
                    stage_metrics.push(StageMetrics {
                        stage_name: "Retrieve".to_string(),
                        input_count,
                        output_count,
                        duration_ticks: tick,
                    });
                }

                PipelineStageKind::Rank => {
                    results.sort_by(|a, b| {
                        b.score
                            .partial_cmp(&a.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let output_count = results.len();
                    stage_metrics.push(StageMetrics {
                        stage_name: "Rank".to_string(),
                        input_count,
                        output_count,
                        duration_ticks: tick,
                    });
                }

                PipelineStageKind::Filter { min_score } => {
                    let threshold = *min_score;
                    results.retain(|r| r.score >= threshold);
                    let output_count = results.len();
                    stage_metrics.push(StageMetrics {
                        stage_name: "Filter".to_string(),
                        input_count,
                        output_count,
                        duration_ticks: tick,
                    });
                }
            }
        }

        // Update cumulative statistics.
        self.stats.total_runs += 1;
        self.stats.total_results_returned += results.len() as u64;
        self.stats.avg_results_per_run =
            self.stats.total_results_returned as f64 / self.stats.total_runs as f64;

        PipelineRun {
            query_embedding: embedding,
            results,
            stage_metrics,
            total_stages,
        }
    }

    /// Return a reference to the cumulative pipeline statistics.
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pipeline(stages: Vec<PipelineStageKind>) -> SemanticQueryPipeline {
        SemanticQueryPipeline::new(PipelineConfig { stages })
    }

    fn unit_embedding(dim: usize) -> Vec<f32> {
        vec![1.0_f32 / (dim as f32).sqrt(); dim]
    }

    // 1. new() starts with zero stats
    #[test]
    fn test_new_stats_zeroed() {
        let pipeline = make_pipeline(vec![]);
        let s = pipeline.stats();
        assert_eq!(s.total_runs, 0);
        assert_eq!(s.total_results_returned, 0);
        assert_eq!(s.avg_results_per_run, 0.0);
    }

    // 2. run with empty stages returns empty results
    #[test]
    fn test_empty_pipeline_empty_results() {
        let mut pipeline = make_pipeline(vec![]);
        let run = pipeline.run(vec![0.5, 0.5]);
        assert_eq!(run.result_count(), 0);
        assert!(run.results.is_empty());
        assert!(run.stage_metrics.is_empty());
    }

    // 3. Preprocess normalizes embedding (no change to result count)
    #[test]
    fn test_preprocess_normalizes() {
        let mut pipeline = make_pipeline(vec![PipelineStageKind::Preprocess]);
        let run = pipeline.run(vec![3.0, 4.0]);
        // 3-4-5 triangle: norm = 5 → [0.6, 0.8]
        let emb = &run.query_embedding;
        assert!((emb[0] - 0.6).abs() < 1e-5, "expected 0.6 got {}", emb[0]);
        assert!((emb[1] - 0.8).abs() < 1e-5, "expected 0.8 got {}", emb[1]);
    }

    // 4. Preprocess does not change result count
    #[test]
    fn test_preprocess_no_result_change() {
        let mut pipeline = make_pipeline(vec![PipelineStageKind::Preprocess]);
        let run = pipeline.run(vec![1.0, 0.0]);
        assert_eq!(run.result_count(), 0);
    }

    // 5. Expand generates correct number of results
    #[test]
    fn test_expand_result_count() {
        let mut pipeline = make_pipeline(vec![PipelineStageKind::Expand { max_variants: 5 }]);
        let run = pipeline.run(unit_embedding(4));
        assert_eq!(run.result_count(), 5);
    }

    // 6. Expand result scores correctly generated (0.8, 0.75, 0.70, ...)
    #[test]
    fn test_expand_scores() {
        let mut pipeline = make_pipeline(vec![PipelineStageKind::Expand { max_variants: 3 }]);
        let run = pipeline.run(unit_embedding(4));
        let scores: Vec<f32> = run.results.iter().map(|r| r.score).collect();
        assert!((scores[0] - 0.8).abs() < 1e-5, "score[0]={}", scores[0]);
        assert!((scores[1] - 0.75).abs() < 1e-5, "score[1]={}", scores[1]);
        assert!((scores[2] - 0.70).abs() < 1e-5, "score[2]={}", scores[2]);
    }

    // 7. Expand result cids use "expand_{i}" format
    #[test]
    fn test_expand_cids() {
        let mut pipeline = make_pipeline(vec![PipelineStageKind::Expand { max_variants: 2 }]);
        let run = pipeline.run(unit_embedding(4));
        assert_eq!(run.results[0].cid, "expand_0");
        assert_eq!(run.results[1].cid, "expand_1");
    }

    // 8. Retrieve generates top_k results
    #[test]
    fn test_retrieve_result_count() {
        let mut pipeline = make_pipeline(vec![PipelineStageKind::Retrieve { top_k: 7 }]);
        let run = pipeline.run(unit_embedding(4));
        assert_eq!(run.result_count(), 7);
    }

    // 9. Retrieve result scores 1.0, 0.9, 0.8 etc.
    #[test]
    fn test_retrieve_scores() {
        let mut pipeline = make_pipeline(vec![PipelineStageKind::Retrieve { top_k: 3 }]);
        let run = pipeline.run(unit_embedding(4));
        assert!((run.results[0].score - 1.0).abs() < 1e-5);
        assert!((run.results[1].score - 0.9).abs() < 1e-5);
        assert!((run.results[2].score - 0.8).abs() < 1e-5);
    }

    // 10. Retrieve cids use "retrieve_{i}" format
    #[test]
    fn test_retrieve_cids() {
        let mut pipeline = make_pipeline(vec![PipelineStageKind::Retrieve { top_k: 2 }]);
        let run = pipeline.run(unit_embedding(4));
        assert_eq!(run.results[0].cid, "retrieve_0");
        assert_eq!(run.results[1].cid, "retrieve_1");
    }

    // 11. Rank sorts results by score descending
    #[test]
    fn test_rank_sorts_descending() {
        // Expand adds scores 0.8, 0.75; Retrieve adds 1.0, 0.9; Rank sorts all.
        let mut pipeline = make_pipeline(vec![
            PipelineStageKind::Expand { max_variants: 2 },
            PipelineStageKind::Retrieve { top_k: 2 },
            PipelineStageKind::Rank,
        ]);
        let run = pipeline.run(unit_embedding(4));
        let scores: Vec<f32> = run.results.iter().map(|r| r.score).collect();
        for w in scores.windows(2) {
            assert!(
                w[0] >= w[1],
                "Scores not sorted descending: {} < {}",
                w[0],
                w[1]
            );
        }
    }

    // 12. Filter removes results below threshold
    #[test]
    fn test_filter_removes_below_threshold() {
        let mut pipeline = make_pipeline(vec![
            PipelineStageKind::Retrieve { top_k: 5 },
            PipelineStageKind::Filter { min_score: 0.85 },
        ]);
        let run = pipeline.run(unit_embedding(4));
        // scores: 1.0, 0.9, 0.8, 0.7, 0.6 → only 1.0 and 0.9 pass
        assert_eq!(run.result_count(), 2);
        for r in &run.results {
            assert!(r.score >= 0.85, "score {} below threshold", r.score);
        }
    }

    // 13. Filter keeps results above threshold
    #[test]
    fn test_filter_keeps_above_threshold() {
        let mut pipeline = make_pipeline(vec![
            PipelineStageKind::Retrieve { top_k: 3 },
            PipelineStageKind::Filter { min_score: 0.5 },
        ]);
        let run = pipeline.run(unit_embedding(4));
        // scores 1.0, 0.9, 0.8 all pass 0.5
        assert_eq!(run.result_count(), 3);
    }

    // 14. Multi-stage pipeline: Retrieve → Rank → Filter
    #[test]
    fn test_multi_stage_pipeline() {
        let mut pipeline = make_pipeline(vec![
            PipelineStageKind::Retrieve { top_k: 10 },
            PipelineStageKind::Rank,
            PipelineStageKind::Filter { min_score: 0.75 },
        ]);
        let run = pipeline.run(unit_embedding(4));
        // Retrieve: 1.0, 0.9, 0.8, 0.7, … (10 items)
        // Filter ≥ 0.75: 1.0, 0.9, 0.8 → 3 items
        assert_eq!(run.result_count(), 3);
        // Should still be sorted after Rank
        let scores: Vec<f32> = run.results.iter().map(|r| r.score).collect();
        for w in scores.windows(2) {
            assert!(w[0] >= w[1]);
        }
    }

    // 15. stage_metrics recorded for each stage
    #[test]
    fn test_stage_metrics_count() {
        let mut pipeline = make_pipeline(vec![
            PipelineStageKind::Preprocess,
            PipelineStageKind::Expand { max_variants: 3 },
            PipelineStageKind::Rank,
        ]);
        let run = pipeline.run(unit_embedding(4));
        assert_eq!(run.stage_metrics.len(), 3);
    }

    // 16. StageMetrics input/output counts correct for Expand
    #[test]
    fn test_stage_metrics_expand_counts() {
        let mut pipeline = make_pipeline(vec![PipelineStageKind::Expand { max_variants: 4 }]);
        let run = pipeline.run(unit_embedding(4));
        let m = &run.stage_metrics[0];
        assert_eq!(m.input_count, 0, "Expand should see 0 inputs");
        assert_eq!(m.output_count, 4, "Expand should produce 4 outputs");
    }

    // 17. StageMetrics input/output counts correct for Filter
    #[test]
    fn test_stage_metrics_filter_counts() {
        let mut pipeline = make_pipeline(vec![
            PipelineStageKind::Retrieve { top_k: 5 },
            PipelineStageKind::Filter { min_score: 0.85 },
        ]);
        let run = pipeline.run(unit_embedding(4));
        let filter_m = &run.stage_metrics[1];
        assert_eq!(filter_m.input_count, 5);
        assert_eq!(filter_m.output_count, 2); // 1.0 and 0.9 pass
    }

    // 18. StageMetrics duration_ticks equals stage index
    #[test]
    fn test_stage_metrics_duration_ticks() {
        let mut pipeline = make_pipeline(vec![
            PipelineStageKind::Preprocess,
            PipelineStageKind::Retrieve { top_k: 2 },
            PipelineStageKind::Rank,
            PipelineStageKind::Filter { min_score: 0.5 },
        ]);
        let run = pipeline.run(unit_embedding(4));
        for (idx, m) in run.stage_metrics.iter().enumerate() {
            assert_eq!(
                m.duration_ticks, idx as u64,
                "stage {} ticks should be {}",
                idx, idx
            );
        }
    }

    // 19. PipelineRun total_stages correct
    #[test]
    fn test_pipeline_run_total_stages() {
        let mut pipeline = make_pipeline(vec![
            PipelineStageKind::Expand { max_variants: 2 },
            PipelineStageKind::Rank,
        ]);
        let run = pipeline.run(unit_embedding(4));
        assert_eq!(run.total_stages, 2);
    }

    // 20. PipelineRun top_result() returns highest score
    #[test]
    fn test_top_result_highest_score() {
        let mut pipeline = make_pipeline(vec![
            PipelineStageKind::Expand { max_variants: 3 },
            PipelineStageKind::Retrieve { top_k: 3 },
        ]);
        let run = pipeline.run(unit_embedding(4));
        let top = run.top_result().expect("should have a top result");
        let max_score = run
            .results
            .iter()
            .map(|r| r.score)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (top.score - max_score).abs() < 1e-6,
            "top_result score {} != max {}",
            top.score,
            max_score
        );
    }

    // 21. PipelineRun result_count() correct
    #[test]
    fn test_result_count_method() {
        let mut pipeline = make_pipeline(vec![PipelineStageKind::Retrieve { top_k: 6 }]);
        let run = pipeline.run(unit_embedding(4));
        assert_eq!(run.result_count(), 6);
        assert_eq!(run.result_count(), run.results.len());
    }

    // 22. stats total_runs increments
    #[test]
    fn test_stats_total_runs_increments() {
        let mut pipeline = make_pipeline(vec![]);
        pipeline.run(vec![1.0]);
        pipeline.run(vec![1.0]);
        pipeline.run(vec![1.0]);
        assert_eq!(pipeline.stats().total_runs, 3);
    }

    // 23. stats total_results_returned accumulates
    #[test]
    fn test_stats_total_results_accumulates() {
        let mut pipeline = make_pipeline(vec![PipelineStageKind::Retrieve { top_k: 3 }]);
        pipeline.run(unit_embedding(4)); // +3
        pipeline.run(unit_embedding(4)); // +3
        assert_eq!(pipeline.stats().total_results_returned, 6);
    }

    // 24. stats avg_results_per_run correct
    #[test]
    fn test_stats_avg_results_per_run() {
        let mut pipeline = make_pipeline(vec![PipelineStageKind::Retrieve { top_k: 4 }]);
        pipeline.run(unit_embedding(4)); // 4 results
        pipeline.run(unit_embedding(4)); // 4 results
        let avg = pipeline.stats().avg_results_per_run;
        assert!((avg - 4.0).abs() < 1e-9, "expected avg 4.0, got {}", avg);
    }

    // 25. top_result returns None for empty results
    #[test]
    fn test_top_result_none_when_empty() {
        let mut pipeline = make_pipeline(vec![]);
        let run = pipeline.run(vec![]);
        assert!(run.top_result().is_none());
    }

    // 26. Expand stage_added field is "Expand"
    #[test]
    fn test_expand_stage_added_field() {
        let mut pipeline = make_pipeline(vec![PipelineStageKind::Expand { max_variants: 2 }]);
        let run = pipeline.run(unit_embedding(4));
        for r in &run.results {
            assert_eq!(r.stage_added, "Expand");
        }
    }

    // 27. Retrieve stage_added field is "Retrieve"
    #[test]
    fn test_retrieve_stage_added_field() {
        let mut pipeline = make_pipeline(vec![PipelineStageKind::Retrieve { top_k: 2 }]);
        let run = pipeline.run(unit_embedding(4));
        for r in &run.results {
            assert_eq!(r.stage_added, "Retrieve");
        }
    }

    // 28. PipelineConfig::stage_count matches stages length
    #[test]
    fn test_pipeline_config_stage_count() {
        let config = PipelineConfig {
            stages: vec![
                PipelineStageKind::Preprocess,
                PipelineStageKind::Expand { max_variants: 1 },
                PipelineStageKind::Rank,
            ],
        };
        assert_eq!(config.stage_count(), 3);
    }

    // 29. Preprocess with zero vector stays zero (no panic)
    #[test]
    fn test_preprocess_zero_vector_no_panic() {
        let mut pipeline = make_pipeline(vec![PipelineStageKind::Preprocess]);
        let run = pipeline.run(vec![0.0, 0.0, 0.0]);
        // Should remain zeroed and not panic.
        assert!(run.query_embedding.iter().all(|&v| v == 0.0));
    }

    // 30. Rank with single result leaves it unchanged
    #[test]
    fn test_rank_single_result() {
        let mut pipeline = make_pipeline(vec![
            PipelineStageKind::Retrieve { top_k: 1 },
            PipelineStageKind::Rank,
        ]);
        let run = pipeline.run(unit_embedding(4));
        assert_eq!(run.result_count(), 1);
        assert!((run.results[0].score - 1.0).abs() < 1e-5);
    }
}
