//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::functions::normalize_in_place;
use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    fn make_doc(id: &str, embedding: Vec<f64>) -> SearchDoc {
        SearchDoc {
            id: id.to_string(),
            embedding,
            metadata: vec![("key".to_string(), "val".to_string())],
        }
    }
    fn uniform_index(n: usize, dim: usize) -> ContextualEmbeddingSearch {
        let mut engine = ContextualEmbeddingSearch::new();
        for i in 0..n {
            let mut emb = vec![0.0f64; dim];
            let angle = std::f64::consts::PI * 2.0 * (i as f64) / (n as f64);
            emb[0] = angle.cos();
            if dim > 1 {
                emb[1] = angle.sin();
            }
            engine
                .add_document(make_doc(&format!("doc{i}"), emb))
                .expect("test: add_document should succeed for uniform index doc");
        }
        engine
    }
    fn default_context() -> SearchContext {
        SearchContext::new("test-session", 5)
    }
    fn default_config() -> SearchConfig {
        SearchConfig {
            top_k: 5,
            rerank_top_n: 20,
            ..Default::default()
        }
    }
    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-9, "identical vectors: {sim}");
    }
    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-9);
    }
    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-9);
    }
    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
    #[test]
    fn test_cosine_similarity_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
    #[test]
    fn test_weighted_sum_single() {
        let v = vec![1.0, 2.0, 3.0];
        let result = weighted_sum(&[(&v, 2.0)]);
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }
    #[test]
    fn test_weighted_sum_two() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let result = weighted_sum(&[(&a, 0.5), (&b, 0.5)]);
        assert!((result[0] - 0.5).abs() < 1e-9);
        assert!((result[1] - 0.5).abs() < 1e-9);
    }
    #[test]
    fn test_weighted_sum_empty() {
        let result = weighted_sum(&[] as &[(&[f64], f64)]);
        assert!(result.is_empty());
    }
    #[test]
    fn test_add_document_sets_dimension() {
        let mut engine = ContextualEmbeddingSearch::new();
        engine
            .add_document(make_doc("a", vec![1.0, 2.0]))
            .expect("test: add_document should succeed for first doc");
        assert_eq!(engine.dimension(), Some(2));
    }
    #[test]
    fn test_add_document_dimension_mismatch() {
        let mut engine = ContextualEmbeddingSearch::new();
        engine
            .add_document(make_doc("a", vec![1.0, 2.0]))
            .expect("test: add_document should succeed for initial doc");
        let err = engine
            .add_document(make_doc("b", vec![1.0]))
            .expect_err("test: dimension mismatch should produce an error");
        assert_eq!(
            err,
            SearchError::DimensionMismatch {
                expected: 2,
                got: 1
            }
        );
    }
    #[test]
    fn test_add_duplicate_overwrites() {
        let mut engine = ContextualEmbeddingSearch::new();
        engine
            .add_document(make_doc("a", vec![1.0, 0.0]))
            .expect("test: add_document should succeed for first insert");
        engine
            .add_document(make_doc("a", vec![0.0, 1.0]))
            .expect("test: add_document should succeed for duplicate overwrite");
        assert_eq!(engine.len(), 1);
    }
    #[test]
    fn test_remove_document() {
        let mut engine = ContextualEmbeddingSearch::new();
        engine
            .add_document(make_doc("a", vec![1.0, 0.0]))
            .expect("test: add_document should succeed before remove");
        engine
            .remove_document("a")
            .expect("test: remove_document should succeed for existing doc");
        assert_eq!(engine.len(), 0);
    }
    #[test]
    fn test_remove_nonexistent() {
        let mut engine = ContextualEmbeddingSearch::new();
        let err = engine
            .remove_document("ghost")
            .expect_err("test: removing nonexistent doc should fail");
        matches!(err, SearchError::ConfigurationError(_));
    }
    #[test]
    fn test_add_empty_embedding() {
        let mut engine = ContextualEmbeddingSearch::new();
        let err = engine
            .add_document(make_doc("empty", vec![]))
            .expect_err("test: empty embedding should produce an error");
        matches!(err, SearchError::ConfigurationError(_));
    }
    #[test]
    fn test_is_empty_initially() {
        let engine = ContextualEmbeddingSearch::new();
        assert!(engine.is_empty());
    }
    #[test]
    fn test_len_after_adds() {
        let mut engine = ContextualEmbeddingSearch::new();
        for i in 0..5 {
            engine
                .add_document(make_doc(&format!("d{i}"), vec![i as f64, 0.0]))
                .expect("test: add_document should succeed for each doc in loop");
        }
        assert_eq!(engine.len(), 5);
    }
    #[test]
    fn test_search_empty_index() {
        let mut engine = ContextualEmbeddingSearch::new();
        let ctx = default_context();
        let cfg = default_config();
        let err = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect_err("test: search on empty index should fail");
        assert_eq!(err, SearchError::IndexEmpty);
    }
    #[test]
    fn test_search_returns_top_k() {
        let mut engine = uniform_index(10, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 3,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: search should succeed and return results");
        assert_eq!(results.len(), 3);
    }
    #[test]
    fn test_search_ranks_are_sequential() {
        let mut engine = uniform_index(5, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 5,
            rerank_top_n: 5,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: search should succeed returning ranked results");
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.rank, i + 1);
        }
    }
    #[test]
    fn test_search_query_dimension_mismatch() {
        let mut engine = uniform_index(3, 3);
        let ctx = default_context();
        let cfg = default_config();
        let err = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect_err("test: mismatched query dimension should fail");
        assert_eq!(
            err,
            SearchError::DimensionMismatch {
                expected: 3,
                got: 2
            }
        );
    }
    #[test]
    fn test_search_config_top_k_zero() {
        let mut engine = uniform_index(3, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 0,
            ..Default::default()
        };
        let err = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect_err("test: top_k=0 config should fail");
        matches!(err, SearchError::ConfigurationError(_));
    }
    #[test]
    fn test_search_config_rerank_top_n_zero() {
        let mut engine = uniform_index(3, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            rerank_top_n: 0,
            ..Default::default()
        };
        let err = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect_err("test: rerank_top_n=0 config should fail");
        matches!(err, SearchError::ConfigurationError(_));
    }
    #[test]
    fn test_search_top_k_capped_at_index_size() {
        let mut engine = uniform_index(3, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 100,
            rerank_top_n: 100,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: search should succeed with top_k capped at index size");
        assert!(results.len() <= 3);
    }
    #[test]
    fn test_search_best_result_is_most_similar() {
        let mut engine = ContextualEmbeddingSearch::new();
        engine
            .add_document(make_doc("close", vec![1.0, 0.0]))
            .expect("test: add_document should succeed for close doc");
        engine
            .add_document(make_doc("far", vec![-1.0, 0.0]))
            .expect("test: add_document should succeed for far doc");
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 2,
            rerank_top_n: 2,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            min_relevance: -1.0,
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: search should succeed returning best result");
        assert_eq!(results[0].doc_id, "close");
    }
    #[test]
    fn test_search_min_relevance_filters() {
        let mut engine = uniform_index(8, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 10,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            min_relevance: 0.9,
            ..Default::default()
        };
        let results = engine.search(&[1.0, 0.0], &ctx, &cfg);
        match results {
            Ok(r) => {
                for res in &r {
                    assert!(res.relevance_score >= 0.9 - 1e-6);
                }
            }
            Err(SearchError::InsufficientResults(_)) => {}
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
    #[test]
    fn test_expansion_no_alpha() {
        let engine = ContextualEmbeddingSearch::new();
        let ctx = default_context();
        let cfg = SearchConfig {
            expansion_alpha: 0.0,
            ..Default::default()
        };
        let eq = engine.expand_query(&[1.0, 0.0], &ctx, &cfg);
        assert_eq!(eq.original, vec![1.0, 0.0]);
        assert_eq!(eq.expanded, vec![1.0, 0.0]);
        assert!((eq.expansion_weight).abs() < 1e-9);
    }
    #[test]
    fn test_expansion_shifts_query_toward_history() {
        let mut ctx = SearchContext::new("s", 5);
        let engine = ContextualEmbeddingSearch::new();
        ctx.query_embeddings.push(vec![0.0, 1.0]);
        ctx.query_embeddings.push(vec![0.0, 1.0]);
        let cfg = SearchConfig {
            expansion_alpha: 0.5,
            ..Default::default()
        };
        let eq = engine.expand_query(&[1.0, 0.0], &ctx, &cfg);
        assert!(
            eq.expanded[1] > 0.01,
            "expected y > 0, got {:?}",
            eq.expanded
        );
    }
    #[test]
    fn test_expansion_with_positive_examples() {
        let mut ctx = SearchContext::new("s", 5);
        let engine = ContextualEmbeddingSearch::new();
        ctx.positive_examples.push(vec![0.0, 1.0]);
        let cfg = SearchConfig {
            expansion_alpha: 0.5,
            ..Default::default()
        };
        let eq = engine.expand_query(&[1.0, 0.0], &ctx, &cfg);
        assert!(eq.expanded[1] > 0.0);
    }
    #[test]
    fn test_expansion_weight_stored() {
        let engine = ContextualEmbeddingSearch::new();
        let ctx = default_context();
        let cfg = SearchConfig {
            expansion_alpha: 0.4,
            ..Default::default()
        };
        let eq = engine.expand_query(&[1.0, 0.0], &ctx, &cfg);
        assert!((eq.expansion_weight - 0.4).abs() < 1e-9);
    }
    #[test]
    fn test_expansion_history_weight_zero_when_no_history() {
        let engine = ContextualEmbeddingSearch::new();
        let ctx = default_context();
        let cfg = SearchConfig {
            expansion_alpha: 0.5,
            ..Default::default()
        };
        let eq = engine.expand_query(&[1.0, 0.0], &ctx, &cfg);
        assert!((eq.history_weight).abs() < 1e-9);
    }
    #[test]
    fn test_negative_suppression_reduces_projection() {
        let engine = ContextualEmbeddingSearch::new();
        let mut ctx = SearchContext::new("s", 5);
        ctx.negative_examples.push(vec![0.0, 1.0]);
        let mut query = vec![0.5, 0.5];
        normalize_in_place(&mut query);
        let original_y = query[1];
        engine.suppress_negatives(&mut query, &ctx);
        assert!(
            query[1] < original_y,
            "Y component should decrease after suppression"
        );
    }
    #[test]
    fn test_negative_suppression_no_effect_orthogonal() {
        let engine = ContextualEmbeddingSearch::new();
        let mut ctx = SearchContext::new("s", 5);
        ctx.negative_examples.push(vec![0.0, 1.0]);
        let mut query = vec![1.0, 0.0];
        engine.suppress_negatives(&mut query, &ctx);
        assert!((query[0] - 1.0).abs() < 1e-6);
        assert!(query[1].abs() < 1e-6);
    }
    #[test]
    fn test_negative_suppression_uses_config() {
        let mut engine = uniform_index(5, 2);
        let mut ctx = SearchContext::new("s", 5);
        ctx.negative_examples.push(vec![-1.0, 0.0]);
        let cfg_with = SearchConfig {
            use_negative_examples: true,
            expansion_alpha: 0.0,
            top_k: 5,
            rerank_top_n: 5,
            diversity_strategy: DiversityStrategy::None,
            min_relevance: -1.0,
        };
        let cfg_without = SearchConfig {
            use_negative_examples: false,
            ..cfg_with.clone()
        };
        engine
            .search(&[1.0, 0.0], &ctx, &cfg_with)
            .expect("test: search with negative examples enabled should succeed");
        engine
            .search(&[1.0, 0.0], &ctx, &cfg_without)
            .expect("test: search with negative examples disabled should succeed");
    }
    #[test]
    fn test_diversity_none_sorted_by_relevance() {
        let mut engine = uniform_index(6, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 4,
            rerank_top_n: 6,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: search with None strategy should succeed");
        for w in results.windows(2) {
            assert!(
                w[0].relevance_score >= w[1].relevance_score - 1e-9,
                "not sorted by relevance"
            );
        }
    }
    #[test]
    fn test_mmr_lambda_1_is_pure_relevance() {
        let mut engine = uniform_index(8, 2);
        let ctx = default_context();
        let mk = |strategy| SearchConfig {
            top_k: 4,
            rerank_top_n: 8,
            expansion_alpha: 0.0,
            diversity_strategy: strategy,
            ..Default::default()
        };
        let r_none = engine
            .search(&[1.0, 0.0], &ctx, &mk(DiversityStrategy::None))
            .expect("test: search with None diversity should succeed");
        let r_mmr = engine
            .search(
                &[1.0, 0.0],
                &ctx,
                &mk(DiversityStrategy::MaxMarginalRelevance(1.0)),
            )
            .expect("test: search with MMR lambda=1 should succeed");
        assert_eq!(r_none[0].doc_id, r_mmr[0].doc_id);
    }
    #[test]
    fn test_mmr_lambda_0_maximises_diversity() {
        let mut engine = uniform_index(10, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 5,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::MaxMarginalRelevance(0.0),
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: MMR lambda=0 search should succeed");
        assert_eq!(results.len(), 5);
    }
    #[test]
    fn test_mmr_diversity_scores_present() {
        let mut engine = uniform_index(10, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 5,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::MaxMarginalRelevance(0.5),
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: MMR search should succeed returning diversity scores");
        for r in &results {
            assert!(r.diversity_score >= 0.0 && r.diversity_score <= 1.0 + 1e-6);
        }
    }
    #[test]
    fn test_mmr_correct_first_pick() {
        let mut engine = ContextualEmbeddingSearch::new();
        engine
            .add_document(make_doc("best", vec![1.0, 0.0]))
            .expect("test: add_document should succeed for best doc");
        engine
            .add_document(make_doc("second", vec![0.7, 0.7]))
            .expect("test: add_document should succeed for second doc");
        engine
            .add_document(make_doc("third", vec![-1.0, 0.0]))
            .expect("test: add_document should succeed for third doc");
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 3,
            rerank_top_n: 3,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::MaxMarginalRelevance(0.5),
            min_relevance: -1.0,
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: MMR search should succeed identifying best first pick");
        assert_eq!(results[0].doc_id, "best");
    }
    #[test]
    fn test_mmr_single_doc() {
        let mut engine = ContextualEmbeddingSearch::new();
        engine
            .add_document(make_doc("only", vec![1.0, 0.0]))
            .expect("test: add_document should succeed for single doc");
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 1,
            rerank_top_n: 1,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::MaxMarginalRelevance(0.5),
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: MMR search with single doc should succeed");
        assert_eq!(results.len(), 1);
    }
    #[test]
    fn test_greedy_diversify_basic() {
        let mut engine = uniform_index(10, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 5,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::GreedyDiversify(0.1),
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: greedy diversify search should succeed");
        assert!(!results.is_empty());
    }
    #[test]
    fn test_greedy_diversify_strict_threshold_backfills() {
        let mut engine = uniform_index(10, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 5,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::GreedyDiversify(999.0),
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: greedy diversify with strict threshold should succeed");
        assert_eq!(results.len(), 5);
    }
    #[test]
    fn test_greedy_diversify_zero_threshold_like_none() {
        let mut engine = uniform_index(8, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 4,
            rerank_top_n: 8,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::GreedyDiversify(0.0),
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: greedy diversify with zero threshold should succeed");
        assert_eq!(results.len(), 4);
    }
    #[test]
    fn test_dpp_basic() {
        let mut engine = uniform_index(10, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 5,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::DeterminantalPointProcess,
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: DPP search should succeed");
        assert_eq!(results.len(), 5);
    }
    #[test]
    fn test_dpp_scores_in_range() {
        let mut engine = uniform_index(10, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 5,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::DeterminantalPointProcess,
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: DPP search should succeed returning scored results");
        for r in &results {
            assert!((0.0..=1.0 + 1e-6).contains(&r.diversity_score));
        }
    }
    #[test]
    fn test_dpp_single_doc() {
        let mut engine = ContextualEmbeddingSearch::new();
        engine
            .add_document(make_doc("a", vec![1.0, 0.0]))
            .expect("test: add_document should succeed for single DPP doc");
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 1,
            rerank_top_n: 1,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::DeterminantalPointProcess,
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: DPP search with single doc should succeed");
        assert_eq!(results.len(), 1);
    }
    #[test]
    fn test_final_score_is_average() {
        let mut engine = uniform_index(5, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 3,
            rerank_top_n: 5,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            ..Default::default()
        };
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: search should succeed for final score verification");
        for r in &results {
            let expected = (r.relevance_score + r.diversity_score) / 2.0;
            assert!((r.final_score - expected).abs() < 1e-9);
        }
    }
    #[test]
    fn test_explanation_contains_features() {
        let mut engine = uniform_index(5, 2);
        let ctx = default_context();
        let cfg = default_config();
        let results = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: search should succeed to check explanation features");
        let keys: Vec<&str> = results[0]
            .explanation
            .iter()
            .map(|(k, _)| k.as_str())
            .collect();
        assert!(keys.contains(&"relevance"));
        assert!(keys.contains(&"diversity"));
        assert!(keys.contains(&"expansion_alpha"));
    }
    #[test]
    fn test_update_context_adds_history() {
        let engine = ContextualEmbeddingSearch::new();
        let mut ctx = default_context();
        engine.update_context(&mut ctx, &[1.0, 0.0], "first query".to_string());
        assert_eq!(ctx.query_history.len(), 1);
        assert_eq!(ctx.query_embeddings.len(), 1);
    }
    #[test]
    fn test_update_context_multiple_queries() {
        let engine = ContextualEmbeddingSearch::new();
        let mut ctx = default_context();
        for i in 0..5 {
            engine.update_context(&mut ctx, &[i as f64, 0.0], format!("query {i}"));
        }
        assert_eq!(ctx.query_history.len(), 5);
        assert_eq!(ctx.query_embeddings.len(), 5);
    }
    #[test]
    fn test_update_context_then_search_uses_history() {
        let mut engine = uniform_index(10, 2);
        let mut ctx = SearchContext::new("s", 5);
        let helper = ContextualEmbeddingSearch::new();
        helper.update_context(&mut ctx, &[0.0, 1.0], "q1".to_string());
        helper.update_context(&mut ctx, &[0.0, 1.0], "q2".to_string());
        let cfg = SearchConfig {
            top_k: 3,
            rerank_top_n: 10,
            expansion_alpha: 0.5,
            diversity_strategy: DiversityStrategy::None,
            ..Default::default()
        };
        engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: search with expanded context history should succeed");
    }
    #[test]
    fn test_batch_search_empty_queries() {
        let mut engine = uniform_index(5, 2);
        let ctx = default_context();
        let cfg = default_config();
        let results = engine
            .batch_search(&[], &ctx, &cfg)
            .expect("test: batch_search with empty queries should succeed");
        assert!(results.is_empty());
    }
    #[test]
    fn test_batch_search_multiple_queries() {
        let mut engine = uniform_index(10, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 3,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            ..Default::default()
        };
        let queries = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
        let results = engine
            .batch_search(&queries, &ctx, &cfg)
            .expect("test: batch_search with multiple queries should succeed");
        assert_eq!(results.len(), 3);
        for r in &results {
            assert_eq!(r.len(), 3);
        }
    }
    #[test]
    fn test_batch_search_propagates_error() {
        let mut engine = ContextualEmbeddingSearch::new();
        let ctx = default_context();
        let cfg = default_config();
        let queries = vec![vec![1.0, 0.0]];
        let err = engine
            .batch_search(&queries, &ctx, &cfg)
            .expect_err("test: batch_search on empty index should fail");
        assert_eq!(err, SearchError::IndexEmpty);
    }
    #[test]
    fn test_batch_search_independent_results() {
        let mut engine = uniform_index(10, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 3,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            ..Default::default()
        };
        let q1 = vec![1.0, 0.0];
        let q2 = vec![-1.0, 0.0];
        let batch = engine
            .batch_search(&[q1.clone(), q2.clone()], &ctx, &cfg)
            .expect("test: batch_search should succeed for independent results");
        let single1 = engine
            .search(&q1, &ctx, &cfg)
            .expect("test: single search should succeed for comparison");
        assert_eq!(batch[0][0].doc_id, single1[0].doc_id);
    }
    #[test]
    fn test_stats_initial_zero() {
        let engine = ContextualEmbeddingSearch::new();
        let s = engine.stats();
        assert_eq!(s.queries_processed, 0);
        assert_eq!(s.cache_hits, 0);
    }
    #[test]
    fn test_stats_queries_processed_increments() {
        let mut engine = uniform_index(5, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 3,
            rerank_top_n: 5,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            ..Default::default()
        };
        engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: first search should succeed for stats tracking");
        engine
            .search(&[0.0, 1.0], &ctx, &cfg)
            .expect("test: second search should succeed for stats tracking");
        assert_eq!(engine.stats().queries_processed, 2);
    }
    #[test]
    fn test_stats_avg_expansion_similarity_updates() {
        let mut engine = uniform_index(5, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 3,
            rerank_top_n: 5,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            ..Default::default()
        };
        engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: search should succeed for avg expansion similarity check");
        assert!((engine.stats().avg_expansion_similarity - 1.0).abs() < 1e-6);
    }
    #[test]
    fn test_stats_batch_updates_correctly() {
        let mut engine = uniform_index(5, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 3,
            rerank_top_n: 5,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::None,
            ..Default::default()
        };
        let queries: Vec<Vec<f64>> = vec![vec![1.0, 0.0]; 4];
        engine
            .batch_search(&queries, &ctx, &cfg)
            .expect("test: batch_search should succeed for stats update check");
        assert_eq!(engine.stats().queries_processed, 4);
    }
    #[test]
    fn test_error_display_index_empty() {
        let e = SearchError::IndexEmpty;
        assert!(!e.to_string().is_empty());
    }
    #[test]
    fn test_error_display_dimension_mismatch() {
        let e = SearchError::DimensionMismatch {
            expected: 3,
            got: 2,
        };
        assert!(e.to_string().contains('3'));
        assert!(e.to_string().contains('2'));
    }
    #[test]
    fn test_error_display_insufficient_results() {
        let e = SearchError::InsufficientResults(5);
        assert!(e.to_string().contains('5'));
    }
    #[test]
    fn test_error_display_configuration() {
        let e = SearchError::ConfigurationError("bad value".to_string());
        assert!(e.to_string().contains("bad value"));
    }
    #[test]
    fn test_search_context_new() {
        let ctx = SearchContext::new("my-session", 10);
        assert_eq!(ctx.session_id, "my-session");
        assert_eq!(ctx.context_window, 10);
        assert!(ctx.query_history.is_empty());
    }
    #[test]
    fn test_search_context_positive_negative() {
        let mut ctx = SearchContext::new("s", 5);
        ctx.positive_examples.push(vec![1.0, 0.0]);
        ctx.negative_examples.push(vec![-1.0, 0.0]);
        assert_eq!(ctx.positive_examples.len(), 1);
        assert_eq!(ctx.negative_examples.len(), 1);
    }
    #[test]
    fn test_search_deterministic() {
        let mut engine = uniform_index(10, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 5,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::MaxMarginalRelevance(0.5),
            ..Default::default()
        };
        let r1 = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: first deterministic search should succeed");
        let r2 = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: second deterministic search should succeed");
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.doc_id, b.doc_id);
        }
    }
    #[test]
    fn test_dpp_deterministic() {
        let mut engine = uniform_index(10, 2);
        let ctx = default_context();
        let cfg = SearchConfig {
            top_k: 5,
            rerank_top_n: 10,
            expansion_alpha: 0.0,
            diversity_strategy: DiversityStrategy::DeterminantalPointProcess,
            ..Default::default()
        };
        let r1 = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: first DPP deterministic search should succeed");
        let r2 = engine
            .search(&[1.0, 0.0], &ctx, &cfg)
            .expect("test: second DPP deterministic search should succeed");
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.doc_id, b.doc_id);
        }
    }
    #[test]
    fn test_default_search_config() {
        let cfg = SearchConfig::default();
        assert_eq!(cfg.top_k, 10);
        assert!(cfg.use_negative_examples);
    }
    #[test]
    fn test_default_contextual_embedding_search() {
        let engine = ContextualEmbeddingSearch::default();
        assert!(engine.is_empty());
    }
}
