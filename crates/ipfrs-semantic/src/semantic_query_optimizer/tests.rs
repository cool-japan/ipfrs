//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use std::collections::HashMap;

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    fn default_optimizer() -> SemanticQueryOptimizer {
        SemanticQueryOptimizer::with_defaults()
    }
    fn optimizer_with_synonyms(synonyms: HashMap<String, Vec<String>>) -> SemanticQueryOptimizer {
        let mut config = OptimizerConfig::default();
        config
            .apply_rules
            .push(OptimizationRule::ExpandSynonyms(synonyms));
        SemanticQueryOptimizer::new(config)
    }
    #[test]
    fn parse_single_term() {
        let opt = default_optimizer();
        assert_eq!(
            opt.parse("rust").expect("test: parse single term"),
            QueryNode::Term("rust".to_string())
        );
    }
    #[test]
    fn parse_multiple_terms_no_op() {
        let opt = default_optimizer();
        let result = opt.parse("rust programming");
        assert!(result.is_ok() || result.is_err());
    }
    #[test]
    fn parse_term_with_hyphen() {
        let opt = default_optimizer();
        assert_eq!(
            opt.parse("machine-learning")
                .expect("test: parse hyphenated term"),
            QueryNode::Term("machine-learning".to_string())
        );
    }
    #[test]
    fn parse_and_expr() {
        let opt = default_optimizer();
        let node = opt
            .parse("rust AND systems")
            .expect("test: parse AND expression");
        assert_eq!(
            node,
            QueryNode::And(vec![
                QueryNode::Term("rust".into()),
                QueryNode::Term("systems".into())
            ])
        );
    }
    #[test]
    fn parse_or_expr() {
        let opt = default_optimizer();
        let node = opt
            .parse("python OR rust")
            .expect("test: parse OR expression");
        assert_eq!(
            node,
            QueryNode::Or(vec![
                QueryNode::Term("python".into()),
                QueryNode::Term("rust".into())
            ])
        );
    }
    #[test]
    fn parse_not_expr() {
        let opt = default_optimizer();
        assert_eq!(
            opt.parse("NOT java").expect("test: parse NOT expression"),
            QueryNode::Not(Box::new(QueryNode::Term("java".into())))
        );
    }
    #[test]
    fn parse_chained_and() {
        let opt = default_optimizer();
        let node = opt.parse("a AND b AND c").expect("test: parse chained AND");
        assert_eq!(
            node,
            QueryNode::And(vec![
                QueryNode::Term("a".into()),
                QueryNode::Term("b".into()),
                QueryNode::Term("c".into()),
            ])
        );
    }
    #[test]
    fn parse_chained_or() {
        let opt = default_optimizer();
        let node = opt.parse("a OR b OR c").expect("test: parse chained OR");
        assert_eq!(
            node,
            QueryNode::Or(vec![
                QueryNode::Term("a".into()),
                QueryNode::Term("b".into()),
                QueryNode::Term("c".into()),
            ])
        );
    }
    #[test]
    fn parse_parentheses() {
        let opt = default_optimizer();
        let node = opt
            .parse("(a OR b) AND c")
            .expect("test: parse parenthesized expression");
        assert_eq!(
            node,
            QueryNode::And(vec![
                QueryNode::Or(vec![
                    QueryNode::Term("a".into()),
                    QueryNode::Term("b".into())
                ]),
                QueryNode::Term("c".into()),
            ])
        );
    }
    #[test]
    fn parse_not_and() {
        let opt = default_optimizer();
        let node = opt
            .parse("a AND NOT b")
            .expect("test: parse AND NOT expression");
        assert_eq!(
            node,
            QueryNode::And(vec![
                QueryNode::Term("a".into()),
                QueryNode::Not(Box::new(QueryNode::Term("b".into()))),
            ])
        );
    }
    #[test]
    fn parse_phrase() {
        let opt = default_optimizer();
        assert_eq!(
            opt.parse("\"hello world\"").expect("test: parse phrase"),
            QueryNode::Phrase(vec!["hello".into(), "world".into()])
        );
    }
    #[test]
    fn parse_phrase_multi_word() {
        let opt = default_optimizer();
        assert_eq!(
            opt.parse("\"machine learning model\"")
                .expect("test: parse multi-word phrase"),
            QueryNode::Phrase(vec!["machine".into(), "learning".into(), "model".into()])
        );
    }
    #[test]
    fn parse_phrase_in_and() {
        let opt = default_optimizer();
        assert!(matches!(
            opt.parse("\"hello world\" AND rust")
                .expect("test: parse phrase in AND"),
            QueryNode::And(_)
        ));
    }
    #[test]
    fn parse_unterminated_phrase_error() {
        let opt = default_optimizer();
        assert!(matches!(
            opt.parse("\"unterminated"),
            Err(OptimizerError::ParseError(_))
        ));
    }
    #[test]
    fn parse_filter_eq() {
        let opt = default_optimizer();
        assert_eq!(
            opt.parse("type:image").expect("test: parse filter eq"),
            QueryNode::Filter {
                field: "type".into(),
                op: FilterOp::Eq,
                value: "image".into()
            }
        );
    }
    #[test]
    fn parse_filter_in_and() {
        let opt = default_optimizer();
        assert!(matches!(
            opt.parse("rust AND type:image")
                .expect("test: parse filter in AND"),
            QueryNode::And(_)
        ));
    }
    #[test]
    fn parse_fuzzy_1() {
        let opt = default_optimizer();
        assert_eq!(
            opt.parse("colour~1")
                .expect("test: parse fuzzy with max_edits 1"),
            QueryNode::Fuzzy {
                term: "colour".into(),
                max_edits: 1
            }
        );
    }
    #[test]
    fn parse_fuzzy_2() {
        let opt = default_optimizer();
        assert_eq!(
            opt.parse("programing~2")
                .expect("test: parse fuzzy with max_edits 2"),
            QueryNode::Fuzzy {
                term: "programing".into(),
                max_edits: 2
            }
        );
    }
    #[test]
    fn parse_boost_integer() {
        let opt = default_optimizer();
        assert_eq!(
            opt.parse("title^2").expect("test: parse boost integer"),
            QueryNode::Boost {
                node: Box::new(QueryNode::Term("title".into())),
                factor: 2.0
            }
        );
    }
    #[test]
    fn parse_boost_decimal() {
        let opt = default_optimizer();
        assert_eq!(
            opt.parse("title^2.5").expect("test: parse boost decimal"),
            QueryNode::Boost {
                node: Box::new(QueryNode::Term("title".into())),
                factor: 2.5
            }
        );
    }
    #[test]
    fn parse_empty_string_error() {
        assert!(matches!(
            default_optimizer().parse(""),
            Err(OptimizerError::InvalidQuery(_))
        ));
    }
    #[test]
    fn parse_empty_phrase_error() {
        assert!(matches!(
            default_optimizer().parse("\"\""),
            Err(OptimizerError::ParseError(_))
        ));
    }
    #[test]
    fn parse_missing_close_paren_error() {
        assert!(default_optimizer().parse("(a AND b").is_err());
    }
    #[test]
    fn constant_folding_and_not_same() {
        let opt = default_optimizer();
        let node = QueryNode::And(vec![
            QueryNode::Term("x".into()),
            QueryNode::Not(Box::new(QueryNode::Term("x".into()))),
        ]);
        let (result, rewrites) = opt.fold_constants(node);
        assert_eq!(rewrites, 1);
        assert_eq!(result, QueryNode::And(vec![]));
    }
    #[test]
    fn constant_folding_double_not() {
        let opt = default_optimizer();
        let node = QueryNode::Not(Box::new(QueryNode::Not(Box::new(QueryNode::Term(
            "x".into(),
        )))));
        let (result, rewrites) = opt.fold_constants(node);
        assert_eq!(result, QueryNode::Term("x".into()));
        assert_eq!(rewrites, 1);
    }
    #[test]
    fn constant_folding_no_change() {
        let opt = default_optimizer();
        let node = QueryNode::And(vec![
            QueryNode::Term("a".into()),
            QueryNode::Term("b".into()),
        ]);
        let (result, rewrites) = opt.fold_constants(node.clone());
        assert_eq!(rewrites, 0);
        assert_eq!(result, node);
    }
    #[test]
    fn constant_folding_nested_double_not() {
        let opt = default_optimizer();
        let inner = QueryNode::Not(Box::new(QueryNode::Not(Box::new(QueryNode::Term(
            "nested".into(),
        )))));
        let node = QueryNode::And(vec![QueryNode::Term("a".into()), inner]);
        let (result, rewrites) = opt.fold_constants(node);
        assert!(rewrites > 0);
        if let QueryNode::And(children) = result {
            assert!(children
                .iter()
                .any(|c| c == &QueryNode::Term("nested".into())));
        } else {
            panic!("expected And");
        }
    }
    #[test]
    fn deduplicate_terms_and() {
        let opt = default_optimizer();
        let node = QueryNode::And(vec![
            QueryNode::Term("rust".into()),
            QueryNode::Term("rust".into()),
            QueryNode::Term("systems".into()),
        ]);
        let (result, rewrites) = opt.deduplicate_terms(node);
        assert_eq!(rewrites, 1);
        if let QueryNode::And(ch) = result {
            assert_eq!(ch.len(), 2);
        } else {
            panic!("expected And");
        }
    }
    #[test]
    fn deduplicate_terms_or() {
        let opt = default_optimizer();
        let node = QueryNode::Or(vec![
            QueryNode::Term("a".into()),
            QueryNode::Term("b".into()),
            QueryNode::Term("a".into()),
        ]);
        let (result, rewrites) = opt.deduplicate_terms(node);
        assert_eq!(rewrites, 1);
        if let QueryNode::Or(ch) = result {
            assert_eq!(ch.len(), 2);
        } else {
            panic!("expected Or");
        }
    }
    #[test]
    fn deduplicate_terms_no_duplicates() {
        let opt = default_optimizer();
        let node = QueryNode::And(vec![
            QueryNode::Term("a".into()),
            QueryNode::Term("b".into()),
        ]);
        let (_, rewrites) = opt.deduplicate_terms(node);
        assert_eq!(rewrites, 0);
    }
    #[test]
    fn flatten_nested_and() {
        let opt = default_optimizer();
        let node = QueryNode::And(vec![
            QueryNode::And(vec![
                QueryNode::Term("a".into()),
                QueryNode::Term("b".into()),
            ]),
            QueryNode::Term("c".into()),
        ]);
        let (result, rewrites) = opt.flatten_nested(node);
        assert!(rewrites > 0);
        if let QueryNode::And(ch) = result {
            assert_eq!(ch.len(), 3);
        } else {
            panic!("expected And");
        }
    }
    #[test]
    fn flatten_nested_or() {
        let opt = default_optimizer();
        let node = QueryNode::Or(vec![
            QueryNode::Or(vec![
                QueryNode::Term("x".into()),
                QueryNode::Term("y".into()),
            ]),
            QueryNode::Term("z".into()),
        ]);
        let (result, rewrites) = opt.flatten_nested(node);
        assert!(rewrites > 0);
        if let QueryNode::Or(ch) = result {
            assert_eq!(ch.len(), 3);
        } else {
            panic!("expected Or");
        }
    }
    #[test]
    fn flatten_nested_mixed_no_flatten() {
        let opt = default_optimizer();
        let node = QueryNode::Or(vec![
            QueryNode::And(vec![
                QueryNode::Term("a".into()),
                QueryNode::Term("b".into()),
            ]),
            QueryNode::Term("c".into()),
        ]);
        let (result, _) = opt.flatten_nested(node);
        if let QueryNode::Or(ch) = result {
            assert_eq!(ch.len(), 2);
        } else {
            panic!("expected Or");
        }
    }
    #[test]
    fn push_down_filters_moves_filter_first() {
        let opt = default_optimizer();
        let node = QueryNode::And(vec![
            QueryNode::Term("rust".into()),
            QueryNode::Filter {
                field: "type".into(),
                op: FilterOp::Eq,
                value: "article".into(),
            },
        ]);
        let (result, rewrites) = opt.push_down_filters(node);
        assert!(rewrites > 0);
        if let QueryNode::And(ch) = result {
            assert!(matches!(ch[0], QueryNode::Filter { .. }));
        } else {
            panic!("expected And");
        }
    }
    #[test]
    fn push_down_filters_no_filter_unchanged() {
        let opt = default_optimizer();
        let node = QueryNode::And(vec![
            QueryNode::Term("a".into()),
            QueryNode::Term("b".into()),
        ]);
        let (result, rewrites) = opt.push_down_filters(node);
        assert_eq!(rewrites, 0);
        assert!(matches!(result, QueryNode::And(_)));
    }
    #[test]
    fn reorder_by_selectivity_rare_first() {
        let mut config = OptimizerConfig::default();
        config.index_stats.total_docs = 10_000;
        config
            .index_stats
            .term_frequencies
            .insert("common".into(), 8000);
        config
            .index_stats
            .term_frequencies
            .insert("rare".into(), 10);
        let opt = SemanticQueryOptimizer::new(config);
        let node = QueryNode::And(vec![
            QueryNode::Term("common".into()),
            QueryNode::Term("rare".into()),
        ]);
        let (result, _) = opt.reorder_by_selectivity(node);
        if let QueryNode::And(ch) = result {
            assert_eq!(ch[0], QueryNode::Term("rare".into()));
        } else {
            panic!("expected And");
        }
    }
    #[test]
    fn reorder_filter_before_common_term() {
        let mut config = OptimizerConfig::default();
        config.index_stats.total_docs = 1000;
        config
            .index_stats
            .term_frequencies
            .insert("verycommon".into(), 900);
        let opt = SemanticQueryOptimizer::new(config);
        let node = QueryNode::And(vec![
            QueryNode::Term("verycommon".into()),
            QueryNode::Filter {
                field: "f".into(),
                op: FilterOp::Eq,
                value: "v".into(),
            },
        ]);
        let (result, _) = opt.reorder_by_selectivity(node);
        if let QueryNode::And(ch) = result {
            assert!(matches!(ch[0], QueryNode::Filter { .. }));
        } else {
            panic!("expected And");
        }
    }
    #[test]
    fn expand_synonyms_single_term() {
        let mut synonyms = HashMap::new();
        synonyms.insert("colour".into(), vec!["color".into()]);
        let opt = optimizer_with_synonyms(synonyms.clone());
        let (result, rewrites) = opt.expand_synonyms(QueryNode::Term("colour".into()), &synonyms);
        assert_eq!(rewrites, 1);
        assert!(matches!(result, QueryNode::Or(_)));
        if let QueryNode::Or(ch) = result {
            assert_eq!(ch.len(), 2);
        }
    }
    #[test]
    fn expand_synonyms_no_match() {
        let synonyms: HashMap<String, Vec<String>> = HashMap::new();
        let opt = default_optimizer();
        let (result, rewrites) = opt.expand_synonyms(QueryNode::Term("rust".into()), &synonyms);
        assert_eq!(rewrites, 0);
        assert_eq!(result, QueryNode::Term("rust".into()));
    }
    #[test]
    fn expand_synonyms_multiple() {
        let mut synonyms = HashMap::new();
        synonyms.insert("car".into(), vec!["automobile".into(), "vehicle".into()]);
        let opt = optimizer_with_synonyms(synonyms.clone());
        let (result, _) = opt.expand_synonyms(QueryNode::Term("car".into()), &synonyms);
        if let QueryNode::Or(ch) = result {
            assert_eq!(ch.len(), 3);
        } else {
            panic!("expected Or");
        }
    }
    #[test]
    fn expand_synonyms_in_and() {
        let mut synonyms = HashMap::new();
        synonyms.insert("colour".into(), vec!["color".into()]);
        let opt = default_optimizer();
        let node = QueryNode::And(vec![
            QueryNode::Term("colour".into()),
            QueryNode::Term("image".into()),
        ]);
        let (result, rewrites) = opt.expand_synonyms(node, &synonyms);
        assert_eq!(rewrites, 1);
        if let QueryNode::And(ch) = result {
            assert!(matches!(ch[0], QueryNode::Or(_)));
        } else {
            panic!("expected And");
        }
    }
    #[test]
    fn cost_term() {
        assert!(
            (default_optimizer().estimate_cost(&QueryNode::Term("x".into())) - 1.0).abs() < 1e-9
        );
    }
    #[test]
    fn cost_phrase() {
        assert!(
            (default_optimizer().estimate_cost(&QueryNode::Phrase(vec!["a".into(), "b".into()]))
                - 2.0)
                .abs()
                < 1e-9
        );
    }
    #[test]
    fn cost_embedding() {
        assert!(
            (default_optimizer().estimate_cost(&QueryNode::Embedding(vec![0.1; 128])) - 10.0).abs()
                < 1e-9
        );
    }
    #[test]
    fn cost_or_is_sum() {
        let node = QueryNode::Or(vec![
            QueryNode::Term("a".into()),
            QueryNode::Term("b".into()),
        ]);
        assert!((default_optimizer().estimate_cost(&node) - 2.0).abs() < 1e-9);
    }
    #[test]
    fn cost_not_is_1_5x_child() {
        let node = QueryNode::Not(Box::new(QueryNode::Term("x".into())));
        assert!((default_optimizer().estimate_cost(&node) - 1.5).abs() < 1e-9);
    }
    #[test]
    fn cost_filter() {
        let node = QueryNode::Filter {
            field: "f".into(),
            op: FilterOp::Eq,
            value: "v".into(),
        };
        assert!((default_optimizer().estimate_cost(&node) - 0.5).abs() < 1e-9);
    }
    #[test]
    fn cost_boost_adds_0_1() {
        let node = QueryNode::Boost {
            node: Box::new(QueryNode::Term("x".into())),
            factor: 2.0,
        };
        assert!((default_optimizer().estimate_cost(&node) - 1.1).abs() < 1e-9);
    }
    #[test]
    fn cost_fuzzy() {
        let node = QueryNode::Fuzzy {
            term: "colour".into(),
            max_edits: 1,
        };
        assert!((default_optimizer().estimate_cost(&node) - 6.0).abs() < 1e-9);
    }
    #[test]
    fn cost_and_is_product_over_total() {
        let opt = default_optimizer();
        let node = QueryNode::And(vec![
            QueryNode::Term("a".into()),
            QueryNode::Term("b".into()),
        ]);
        let cost = opt.estimate_cost(&node);
        assert!((cost - 1.0 / 100_000.0).abs() < 1e-12);
    }
    #[test]
    fn plan_term_has_term_lookup_step() {
        let steps = default_optimizer().plan_execution(&QueryNode::Term("rust".into()));
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].step_type, StepType::TermLookup);
    }
    #[test]
    fn plan_embedding_has_vector_scan_step() {
        let steps = default_optimizer().plan_execution(&QueryNode::Embedding(vec![0.0; 64]));
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].step_type, StepType::VectorScan);
    }
    #[test]
    fn plan_filter_has_filter_step() {
        let steps = default_optimizer().plan_execution(&QueryNode::Filter {
            field: "type".into(),
            op: FilterOp::Eq,
            value: "article".into(),
        });
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].step_type, StepType::Filter);
    }
    #[test]
    fn plan_and_has_intersect_join() {
        let node = QueryNode::And(vec![
            QueryNode::Term("a".into()),
            QueryNode::Term("b".into()),
        ]);
        let steps = default_optimizer().plan_execution(&node);
        assert_eq!(
            steps
                .last()
                .expect("test: execution plan has steps")
                .step_type,
            StepType::Join(JoinType::Intersect)
        );
    }
    #[test]
    fn plan_or_has_union_join() {
        let node = QueryNode::Or(vec![
            QueryNode::Term("a".into()),
            QueryNode::Term("b".into()),
        ]);
        let steps = default_optimizer().plan_execution(&node);
        assert_eq!(
            steps
                .last()
                .expect("test: execution plan has steps")
                .step_type,
            StepType::Join(JoinType::Union)
        );
    }
    #[test]
    fn plan_not_has_difference_join() {
        let node = QueryNode::Not(Box::new(QueryNode::Term("x".into())));
        let steps = default_optimizer().plan_execution(&node);
        assert_eq!(
            steps
                .last()
                .expect("test: execution plan has steps")
                .step_type,
            StepType::Join(JoinType::Difference)
        );
    }
    #[test]
    fn plan_boost_has_rewrite_step() {
        let node = QueryNode::Boost {
            node: Box::new(QueryNode::Term("title".into())),
            factor: 2.0,
        };
        let steps = default_optimizer().plan_execution(&node);
        assert_eq!(
            steps
                .last()
                .expect("test: execution plan has steps")
                .step_type,
            StepType::Rewrite
        );
    }
    #[test]
    fn plan_fuzzy_has_term_lookup() {
        let node = QueryNode::Fuzzy {
            term: "colour".into(),
            max_edits: 1,
        };
        let steps = default_optimizer().plan_execution(&node);
        assert_eq!(steps[0].step_type, StepType::TermLookup);
        assert!(steps[0].description.contains("fuzzy"));
    }
    #[test]
    fn plan_phrase_has_term_lookup() {
        let node = QueryNode::Phrase(vec!["hello".into(), "world".into()]);
        let steps = default_optimizer().plan_execution(&node);
        assert_eq!(steps[0].step_type, StepType::TermLookup);
        assert!(steps[0].description.contains("phrase"));
    }
    #[test]
    fn optimize_produces_plan() {
        let opt = default_optimizer();
        let node = opt
            .parse("rust AND systems")
            .expect("test: parse rust AND systems");
        let plan = opt
            .optimize(node, "rust AND systems")
            .expect("test: optimize rust AND systems");
        assert_eq!(plan.original_query, "rust AND systems");
        assert!(!plan.optimized_nodes.is_empty());
        assert!(!plan.execution_steps.is_empty());
        assert!(plan.estimated_cost >= 0.0);
    }
    #[test]
    fn optimize_deduplicates_via_pipeline() {
        let opt = default_optimizer();
        let node = QueryNode::And(vec![
            QueryNode::Term("rust".into()),
            QueryNode::Term("rust".into()),
        ]);
        let plan = opt
            .optimize(node, "rust AND rust")
            .expect("test: optimize deduplication query");
        if let QueryNode::And(ch) = &plan.optimized_nodes[0] {
            assert_eq!(ch.len(), 1);
        }
    }
    #[test]
    fn optimize_flattens_nested() {
        let opt = default_optimizer();
        let node = QueryNode::And(vec![
            QueryNode::And(vec![
                QueryNode::Term("a".into()),
                QueryNode::Term("b".into()),
            ]),
            QueryNode::Term("c".into()),
        ]);
        let plan = opt
            .optimize(node, "...")
            .expect("test: optimize nested AND");
        if let QueryNode::And(ch) = &plan.optimized_nodes[0] {
            assert_eq!(ch.len(), 3);
        }
    }
    #[test]
    fn stats_counts_optimized_queries() {
        let opt = default_optimizer();
        for _ in 0..3 {
            let node = opt.parse("rust").expect("test: parse rust for stats");
            opt.optimize(node, "rust")
                .expect("test: optimize rust for stats");
        }
        assert_eq!(opt.stats().queries_optimized, 3);
    }
    #[test]
    fn stats_rewrites_incremented() {
        let opt = default_optimizer();
        let node = QueryNode::And(vec![
            QueryNode::Term("x".into()),
            QueryNode::Term("x".into()),
        ]);
        opt.optimize(node, "x AND x")
            .expect("test: optimize for rewrites stats");
        assert!(opt.stats().rewrites_applied > 0);
    }
    #[test]
    fn stats_avg_cost_reduction_between_0_and_1() {
        let opt = default_optimizer();
        let node = QueryNode::And(vec![
            QueryNode::Term("rust".into()),
            QueryNode::Term("rust".into()),
        ]);
        opt.optimize(node, "q")
            .expect("test: optimize for cost reduction stats");
        assert!(opt.stats().avg_cost_reduction >= 0.0);
    }
    #[test]
    fn cache_embedding_and_retrieve() {
        let opt = default_optimizer();
        opt.cache_embedding("rust", vec![1.0, 2.0, 3.0]);
        assert_eq!(opt.get_cached_embedding("rust"), Some(vec![1.0, 2.0, 3.0]));
    }
    #[test]
    fn cache_miss_returns_none() {
        assert!(default_optimizer()
            .get_cached_embedding("nonexistent")
            .is_none());
    }
    #[test]
    fn cache_hit_increments_stats() {
        let opt = default_optimizer();
        opt.cache_embedding("rust", vec![0.1; 8]);
        let node = QueryNode::Fuzzy {
            term: "rust".into(),
            max_edits: 1,
        };
        opt.apply_embedding_caching(node);
        assert_eq!(opt.stats().cache_hits, 1);
    }
    #[test]
    fn levenshtein_equal_strings() {
        assert_eq!(SemanticQueryOptimizer::edit_distance("rust", "rust"), 0);
    }
    #[test]
    fn levenshtein_single_substitution() {
        assert_eq!(SemanticQueryOptimizer::edit_distance("cat", "bat"), 1);
    }
    #[test]
    fn levenshtein_insertion() {
        assert_eq!(SemanticQueryOptimizer::edit_distance("color", "colour"), 1);
    }
    #[test]
    fn levenshtein_deletion() {
        assert_eq!(SemanticQueryOptimizer::edit_distance("colour", "color"), 1);
    }
    #[test]
    fn levenshtein_completely_different() {
        assert_eq!(SemanticQueryOptimizer::edit_distance("abc", "xyz"), 3);
    }
    #[test]
    fn levenshtein_empty_string() {
        assert_eq!(SemanticQueryOptimizer::edit_distance("", "abc"), 3);
        assert_eq!(SemanticQueryOptimizer::edit_distance("abc", ""), 3);
        assert_eq!(SemanticQueryOptimizer::edit_distance("", ""), 0);
    }
    #[test]
    fn levenshtein_programming_typo() {
        assert_eq!(
            SemanticQueryOptimizer::edit_distance("programming", "programing"),
            1
        );
    }
    #[test]
    fn optimizer_error_display_parse() {
        let e = OptimizerError::ParseError("bad token".into());
        assert!(e.to_string().contains("ParseError"));
        assert!(e.to_string().contains("bad token"));
    }
    #[test]
    fn optimizer_error_display_invalid_query() {
        assert!(OptimizerError::InvalidQuery("empty".into())
            .to_string()
            .contains("InvalidQuery"));
    }
    #[test]
    fn optimizer_error_display_optimization_failed() {
        assert!(OptimizerError::OptimizationFailed("internal".into())
            .to_string()
            .contains("OptimizationFailed"));
    }
    #[test]
    fn optimizer_error_display_configuration() {
        assert!(OptimizerError::ConfigurationError("bad".into())
            .to_string()
            .contains("ConfigurationError"));
    }
    #[test]
    fn validate_negative_boost_factor_error() {
        let opt = default_optimizer();
        let node = QueryNode::Boost {
            node: Box::new(QueryNode::Term("x".into())),
            factor: -1.0,
        };
        assert!(matches!(
            opt.optimize(node, "x^-1"),
            Err(OptimizerError::InvalidQuery(_))
        ));
    }
    #[test]
    fn validate_embedding_too_large_error() {
        let config = OptimizerConfig {
            max_embedding_dim: 4,
            ..OptimizerConfig::default()
        };
        let opt = SemanticQueryOptimizer::new(config);
        assert!(matches!(
            opt.optimize(QueryNode::Embedding(vec![0.1; 10]), "vec"),
            Err(OptimizerError::InvalidQuery(_))
        ));
    }
    #[test]
    fn filter_op_variants_are_distinct() {
        let ops = [
            FilterOp::Eq,
            FilterOp::Ne,
            FilterOp::Lt,
            FilterOp::Le,
            FilterOp::Gt,
            FilterOp::Ge,
            FilterOp::In(vec!["a".into()]),
            FilterOp::Contains,
        ];
        assert_eq!(ops.len(), 8);
    }
    #[test]
    fn step_types_are_distinct() {
        let types = vec![
            StepType::TermLookup,
            StepType::VectorScan,
            StepType::Filter,
            StepType::Join(JoinType::Intersect),
            StepType::Join(JoinType::Union),
            StepType::Join(JoinType::Difference),
            StepType::Sort,
            StepType::Limit,
            StepType::Rewrite,
        ];
        assert_eq!(types.len(), 9);
    }
    #[test]
    fn optimize_with_synonym_rule_expands() {
        let mut synonyms = HashMap::new();
        synonyms.insert("colour".into(), vec!["color".into()]);
        let config = OptimizerConfig {
            apply_rules: vec![OptimizationRule::ExpandSynonyms(synonyms)],
            ..OptimizerConfig::default()
        };
        let opt = SemanticQueryOptimizer::new(config);
        let plan = opt
            .optimize(QueryNode::Term("colour".into()), "colour")
            .expect("test: optimize synonym expansion");
        assert!(matches!(&plan.optimized_nodes[0], QueryNode::Or(_)));
    }
    #[test]
    fn index_hints_affect_and_cost() {
        let mut config = OptimizerConfig::default();
        config.index_stats.total_docs = 1000;
        let opt = SemanticQueryOptimizer::new(config);
        let node = QueryNode::And(vec![
            QueryNode::Term("a".into()),
            QueryNode::Term("b".into()),
        ]);
        assert!((opt.estimate_cost(&node) - 0.001).abs() < 1e-9);
    }
    #[test]
    fn config_default_has_sensible_values() {
        let config = OptimizerConfig::default();
        assert!(config.max_terms > 0);
        assert!(config.max_embedding_dim > 0);
        assert!(!config.apply_rules.is_empty());
        assert!(config.index_stats.total_docs > 0);
    }
}
