//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::functions::{
    bootstrap_indices, weighted_bootstrap, xorshift64, xorshift_f64, xorshift_usize,
};
use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    fn binary_xor_samples(n: usize, seed: u64) -> Vec<ElSample> {
        let mut rng = seed;
        (0..n)
            .map(|_| {
                let a = if xorshift64(&mut rng).is_multiple_of(2) {
                    0.0
                } else {
                    1.0
                };
                let b = if xorshift64(&mut rng).is_multiple_of(2) {
                    0.0
                } else {
                    1.0
                };
                let label = if (a > 0.5) ^ (b > 0.5) { 1.0 } else { -1.0 };
                ElSample::new(vec![a, b], label)
            })
            .collect()
    }
    fn linearly_separable_samples(n: usize) -> Vec<ElSample> {
        (0..n)
            .map(|i| {
                let x = i as f64 / n as f64;
                let label = if x > 0.5 { 1.0 } else { -1.0 };
                ElSample::new(vec![x], label)
            })
            .collect()
    }
    fn regression_samples(n: usize) -> Vec<ElSample> {
        (0..n)
            .map(|i| {
                let x = i as f64 / n as f64;
                let y = 2.0 * x + 1.0;
                ElSample::new(vec![x], y)
            })
            .collect()
    }
    fn two_feature_samples(n: usize) -> Vec<ElSample> {
        let mut rng = 123u64;
        (0..n)
            .map(|_| {
                let x1 = xorshift_f64(&mut rng);
                let x2 = xorshift_f64(&mut rng);
                let label = if x1 + x2 > 1.0 { 1.0 } else { -1.0 };
                ElSample::new(vec![x1, x2], label)
            })
            .collect()
    }
    #[test]
    fn test_default_config() {
        let cfg = ElLearnerConfig::default();
        assert_eq!(cfg.n_estimators, 100);
        assert_eq!(cfg.method, ElMethod::Bagging);
        assert!(cfg.validate().is_ok());
    }
    #[test]
    fn test_config_invalid_n_estimators() {
        let cfg = ElLearnerConfig {
            n_estimators: 0,
            ..ElLearnerConfig::default()
        };
        assert!(cfg.validate().is_err());
    }
    #[test]
    fn test_config_invalid_learning_rate_zero() {
        let cfg = ElLearnerConfig {
            learning_rate: 0.0,
            ..ElLearnerConfig::default()
        };
        assert!(cfg.validate().is_err());
    }
    #[test]
    fn test_config_invalid_learning_rate_too_large() {
        let cfg = ElLearnerConfig {
            learning_rate: 1.1,
            ..ElLearnerConfig::default()
        };
        assert!(cfg.validate().is_err());
    }
    #[test]
    fn test_config_invalid_subsample() {
        let cfg = ElLearnerConfig {
            subsample: 0.0,
            ..ElLearnerConfig::default()
        };
        assert!(cfg.validate().is_err());
    }
    #[test]
    fn test_config_valid_all_methods() {
        for method in [
            ElMethod::Bagging,
            ElMethod::AdaBoost,
            ElMethod::GradientBoosting,
            ElMethod::RandomForest,
            ElMethod::Stacking,
        ] {
            let cfg = ElLearnerConfig {
                method,
                ..ElLearnerConfig::default()
            };
            assert!(cfg.validate().is_ok());
        }
    }
    #[test]
    fn test_sample_new() {
        let s = ElSample::new(vec![1.0, 2.0], 1.0);
        assert_eq!(s.features.len(), 2);
        assert_eq!(s.label, 1.0);
    }
    #[test]
    fn test_decision_stump_predict_direction_true() {
        let m = ElBaseModel::DecisionStump {
            feature_index: 0,
            threshold: 0.5,
            direction: true,
            weight: 1.0,
        };
        assert_eq!(m.predict_raw(&[0.3]).expect("test: should succeed"), 1.0);
        assert_eq!(m.predict_raw(&[0.7]).expect("test: should succeed"), -1.0);
    }
    #[test]
    fn test_decision_stump_predict_direction_false() {
        let m = ElBaseModel::DecisionStump {
            feature_index: 0,
            threshold: 0.5,
            direction: false,
            weight: 1.0,
        };
        assert_eq!(m.predict_raw(&[0.3]).expect("test: should succeed"), -1.0);
        assert_eq!(m.predict_raw(&[0.7]).expect("test: should succeed"), 1.0);
    }
    #[test]
    fn test_decision_stump_dim_mismatch() {
        let m = ElBaseModel::DecisionStump {
            feature_index: 5,
            threshold: 0.0,
            direction: true,
            weight: 1.0,
        };
        assert!(m.predict_raw(&[0.0]).is_err());
    }
    #[test]
    fn test_perceptron_predict() {
        let m = ElBaseModel::Perceptron {
            weights: vec![1.0, 0.0],
            bias: -0.5,
            weight: 1.0,
        };
        let r = m.predict_raw(&[1.0, 0.0]).expect("test: should succeed");
        assert!(r > 0.0);
    }
    #[test]
    fn test_perceptron_dim_mismatch() {
        let m = ElBaseModel::Perceptron {
            weights: vec![1.0, 0.0],
            bias: 0.0,
            weight: 1.0,
        };
        assert!(m.predict_raw(&[1.0]).is_err());
    }
    #[test]
    fn test_base_model_weight_stump() {
        let m = ElBaseModel::DecisionStump {
            feature_index: 0,
            threshold: 0.0,
            direction: true,
            weight: 2.5,
        };
        assert_eq!(m.weight(), 2.5);
    }
    #[test]
    fn test_base_model_weight_perceptron() {
        let m = ElBaseModel::Perceptron {
            weights: vec![0.0],
            bias: 0.0,
            weight: 3.7,
        };
        assert_eq!(m.weight(), 3.7);
    }
    #[test]
    fn test_fit_empty_returns_error() {
        let mut el = EnsembleLearner::new(ElLearnerConfig::default());
        assert!(el.fit(&[]).is_err());
    }
    #[test]
    fn test_predict_before_fit() {
        let mut el = EnsembleLearner::new(ElLearnerConfig::default());
        assert!(el.predict(&[0.5]).is_err());
    }
    #[test]
    fn test_predict_dim_mismatch() {
        let samples = linearly_separable_samples(20);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            n_estimators: 5,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        assert!(el.predict(&[0.5, 0.5]).is_err());
    }
    #[test]
    fn test_fit_zero_features_error() {
        let mut el = EnsembleLearner::new(ElLearnerConfig::default());
        let bad = vec![ElSample::new(vec![], 1.0)];
        assert!(el.fit(&bad).is_err());
    }
    #[test]
    fn test_bagging_fits_and_predicts() {
        let samples = linearly_separable_samples(40);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::Bagging,
            n_estimators: 10,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        let pred = el.predict(&[0.8]).expect("test: should succeed");
        assert_eq!(pred.value.signum(), 1.0);
    }
    #[test]
    fn test_bagging_n_models() {
        let samples = linearly_separable_samples(30);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::Bagging,
            n_estimators: 7,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        assert_eq!(el.models().len(), 7);
    }
    #[test]
    fn test_bagging_stats_trained() {
        let samples = linearly_separable_samples(20);
        let mut el = EnsembleLearner::default_bagging(5);
        el.fit(&samples).expect("test: should succeed");
        let stats = el.learner_stats();
        assert!(stats.is_trained);
        assert_eq!(stats.n_models, 5);
    }
    #[test]
    fn test_bagging_predict_negative_class() {
        let samples = linearly_separable_samples(40);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::Bagging,
            n_estimators: 10,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        let pred = el.predict(&[0.1]).expect("test: should succeed");
        assert_eq!(pred.value.signum(), -1.0);
    }
    #[test]
    fn test_bagging_confidence_range() {
        let samples = linearly_separable_samples(30);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            n_estimators: 10,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        let pred = el.predict(&[0.7]).expect("test: should succeed");
        assert!((0.0..=1.0).contains(&pred.confidence));
    }
    #[test]
    fn test_bagging_n_models_in_prediction() {
        let samples = linearly_separable_samples(30);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            n_estimators: 8,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        let pred = el.predict(&[0.5]).expect("test: should succeed");
        assert_eq!(pred.n_models, 8);
    }
    #[test]
    fn test_adaboost_fits() {
        let samples = linearly_separable_samples(40);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::AdaBoost,
            n_estimators: 10,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        assert!(el.learner_stats().is_trained);
    }
    #[test]
    fn test_adaboost_positive_class() {
        let samples = linearly_separable_samples(50);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::AdaBoost,
            n_estimators: 15,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        let pred = el.predict(&[0.9]).expect("test: should succeed");
        assert!(
            pred.value > 0.0,
            "expected positive prediction, got {}",
            pred.value
        );
    }
    #[test]
    fn test_adaboost_negative_class() {
        let samples = linearly_separable_samples(50);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::AdaBoost,
            n_estimators: 15,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        let pred = el.predict(&[0.1]).expect("test: should succeed");
        assert!(
            pred.value < 0.0,
            "expected negative prediction, got {}",
            pred.value
        );
    }
    #[test]
    fn test_adaboost_weights_are_positive() {
        let samples = linearly_separable_samples(30);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::AdaBoost,
            n_estimators: 5,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        for m in el.models() {
            assert!(m.weight() >= 0.0);
        }
    }
    #[test]
    fn test_adaboost_history_populated() {
        let samples = linearly_separable_samples(30);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::AdaBoost,
            n_estimators: 5,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        let stats = el.learner_stats();
        assert!(stats.history_len > 0);
    }
    #[test]
    fn test_gb_fits_regression() {
        let samples = regression_samples(30);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::GradientBoosting,
            n_estimators: 20,
            learning_rate: 0.1,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        assert!(el.learner_stats().is_trained);
    }
    #[test]
    fn test_gb_prediction_direction() {
        let samples = regression_samples(40);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::GradientBoosting,
            n_estimators: 30,
            learning_rate: 0.3,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        let p_high = el.predict(&[0.9]).expect("test: should succeed").value;
        let p_low = el.predict(&[0.1]).expect("test: should succeed").value;
        assert!(p_high > p_low, "high x should give higher prediction");
    }
    #[test]
    fn test_gb_confidence_range() {
        let samples = regression_samples(30);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::GradientBoosting,
            n_estimators: 10,
            learning_rate: 0.1,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        let pred = el.predict(&[0.5]).expect("test: should succeed");
        assert!((0.0..=1.0).contains(&pred.confidence));
    }
    #[test]
    fn test_gb_leaf_values_populated() {
        let samples = regression_samples(20);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::GradientBoosting,
            n_estimators: 5,
            learning_rate: 0.1,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        assert!(!el.gb_leaf_values.is_empty());
    }
    #[test]
    fn test_gb_n_models_matches_estimators() {
        let samples = regression_samples(25);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::GradientBoosting,
            n_estimators: 12,
            learning_rate: 0.2,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        assert!(el.models().len() <= 12);
    }
    #[test]
    fn test_rf_fits_and_predicts() {
        let samples = two_feature_samples(40);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::RandomForest,
            n_estimators: 10,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        let pred = el.predict(&[0.8, 0.8]).expect("test: should succeed");
        assert_eq!(pred.value.signum(), 1.0);
    }
    #[test]
    fn test_rf_n_models() {
        let samples = two_feature_samples(30);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::RandomForest,
            n_estimators: 15,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        assert_eq!(el.models().len(), 15);
    }
    #[test]
    fn test_rf_negative_region() {
        let samples = two_feature_samples(50);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::RandomForest,
            n_estimators: 20,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        let pred = el.predict(&[0.1, 0.1]).expect("test: should succeed");
        assert_eq!(pred.value.signum(), -1.0);
    }
    #[test]
    fn test_stacking_fits() {
        let samples = linearly_separable_samples(40);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::Stacking,
            n_estimators: 6,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        assert!(el.learner_stats().is_trained);
    }
    #[test]
    fn test_stacking_predicts_without_crash() {
        let samples = linearly_separable_samples(40);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::Stacking,
            n_estimators: 4,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        let pred = el.predict(&[0.9]);
        assert!(pred.is_ok());
    }
    #[test]
    fn test_stacking_meta_weights_populated() {
        let samples = linearly_separable_samples(50);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::Stacking,
            n_estimators: 4,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        assert!(!el.meta_weights.is_empty());
    }
    #[test]
    fn test_predict_batch_returns_correct_count() {
        let samples = linearly_separable_samples(30);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            n_estimators: 5,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        let features = vec![vec![0.2], vec![0.5], vec![0.8]];
        let results = el.predict_batch(&features);
        assert_eq!(results.len(), 3);
    }
    #[test]
    fn test_predict_batch_all_ok() {
        let samples = linearly_separable_samples(30);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            n_estimators: 5,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        let features = vec![vec![0.1], vec![0.9]];
        let results = el.predict_batch(&features);
        assert!(results.iter().all(|r| r.is_ok()));
    }
    #[test]
    fn test_predict_batch_signs() {
        let samples = linearly_separable_samples(50);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            n_estimators: 15,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        let features = vec![vec![0.1], vec![0.9]];
        let results = el.predict_batch(&features);
        assert_eq!(
            results[0]
                .as_ref()
                .expect("test: should succeed")
                .value
                .signum(),
            -1.0
        );
        assert_eq!(
            results[1]
                .as_ref()
                .expect("test: should succeed")
                .value
                .signum(),
            1.0
        );
    }
    #[test]
    fn test_feature_importance_sums_to_one() {
        let samples = two_feature_samples(40);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            n_estimators: 10,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        let imp = el.feature_importance();
        assert_eq!(imp.len(), 2);
        let sum: f64 = imp.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }
    #[test]
    fn test_feature_importance_non_negative() {
        let samples = two_feature_samples(40);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            n_estimators: 10,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        for &imp in el.feature_importance().iter() {
            assert!(imp >= 0.0);
        }
    }
    #[test]
    fn test_feature_importance_empty_before_fit() {
        let el = EnsembleLearner::new(ElLearnerConfig::default());
        assert!(el.feature_importance().is_empty());
    }
    #[test]
    fn test_feature_importance_single_feature() {
        let samples = linearly_separable_samples(20);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            n_estimators: 5,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        let imp = el.feature_importance();
        assert_eq!(imp.len(), 1);
        assert!((imp[0] - 1.0).abs() < 1e-9);
    }
    #[test]
    fn test_oob_score_bagging_range() {
        let samples = linearly_separable_samples(40);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::Bagging,
            n_estimators: 20,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        let oob = el.oob_score(&samples);
        assert!((0.0..=1.0).contains(&oob));
    }
    #[test]
    fn test_oob_score_rf_range() {
        let samples = two_feature_samples(40);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::RandomForest,
            n_estimators: 20,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        let oob = el.oob_score(&samples);
        assert!((0.0..=1.0).contains(&oob));
    }
    #[test]
    fn test_oob_score_zero_for_non_bagging() {
        let samples = linearly_separable_samples(30);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::AdaBoost,
            n_estimators: 5,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        assert_eq!(el.oob_score(&samples), 0.0);
    }
    #[test]
    fn test_oob_score_before_fit_is_zero() {
        let el = EnsembleLearner::new(ElLearnerConfig::default());
        let samples = linearly_separable_samples(20);
        assert_eq!(el.oob_score(&samples), 0.0);
    }
    #[test]
    fn test_stats_before_fit() {
        let el = EnsembleLearner::new(ElLearnerConfig::default());
        let stats = el.learner_stats();
        assert!(!stats.is_trained);
        assert_eq!(stats.n_models, 0);
        assert_eq!(stats.total_predictions, 0);
    }
    #[test]
    fn test_stats_after_fit_n_models() {
        let samples = linearly_separable_samples(20);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            n_estimators: 6,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        assert_eq!(el.learner_stats().n_models, 6);
    }
    #[test]
    fn test_stats_total_predictions_increments() {
        let samples = linearly_separable_samples(20);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            n_estimators: 5,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        el.predict(&[0.5]).expect("test: should succeed");
        el.predict(&[0.3]).expect("test: should succeed");
        assert_eq!(el.learner_stats().total_predictions, 2);
    }
    #[test]
    fn test_stats_min_max_weight() {
        let samples = linearly_separable_samples(30);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::AdaBoost,
            n_estimators: 5,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        let stats = el.learner_stats();
        assert!(stats.min_weight <= stats.max_weight);
    }
    #[test]
    fn test_xorshift64_not_zero() {
        let mut state = 1u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }
    #[test]
    fn test_xorshift_f64_range() {
        let mut state = 999u64;
        for _ in 0..100 {
            let v = xorshift_f64(&mut state);
            assert!((0.0..1.0).contains(&v));
        }
    }
    #[test]
    fn test_xorshift_usize_range() {
        let mut state = 42u64;
        for _ in 0..100 {
            let v = xorshift_usize(&mut state, 7);
            assert!(v < 7);
        }
    }
    #[test]
    fn test_bootstrap_indices_length() {
        let mut rng = 1u64;
        let idxs = bootstrap_indices(&mut rng, 20, 10);
        assert_eq!(idxs.len(), 10);
    }
    #[test]
    fn test_bootstrap_indices_in_range() {
        let mut rng = 1u64;
        let idxs = bootstrap_indices(&mut rng, 20, 15);
        for &i in &idxs {
            assert!(i < 20);
        }
    }
    #[test]
    fn test_bagging_determinism() {
        let samples = two_feature_samples(30);
        let cfg = ElLearnerConfig {
            method: ElMethod::Bagging,
            n_estimators: 10,
            seed: 7,
            ..Default::default()
        };
        let mut el1 = EnsembleLearner::new(cfg.clone());
        let mut el2 = EnsembleLearner::new(cfg);
        el1.fit(&samples).expect("test: should succeed");
        el2.fit(&samples).expect("test: should succeed");
        let p1 = el1
            .predict(&[0.5, 0.5])
            .expect("test: should succeed")
            .value;
        let p2 = el2
            .predict(&[0.5, 0.5])
            .expect("test: should succeed")
            .value;
        assert!((p1 - p2).abs() < 1e-12);
    }
    #[test]
    fn test_rf_determinism() {
        let samples = two_feature_samples(30);
        let cfg = ElLearnerConfig {
            method: ElMethod::RandomForest,
            n_estimators: 8,
            seed: 999,
            ..Default::default()
        };
        let mut el1 = EnsembleLearner::new(cfg.clone());
        let mut el2 = EnsembleLearner::new(cfg);
        el1.fit(&samples).expect("test: should succeed");
        el2.fit(&samples).expect("test: should succeed");
        let p1 = el1
            .predict(&[0.3, 0.7])
            .expect("test: should succeed")
            .value;
        let p2 = el2
            .predict(&[0.3, 0.7])
            .expect("test: should succeed")
            .value;
        assert!((p1 - p2).abs() < 1e-12);
    }
    #[test]
    fn test_el_method_display() {
        assert_eq!(ElMethod::Bagging.to_string(), "Bagging");
        assert_eq!(ElMethod::AdaBoost.to_string(), "AdaBoost");
        assert_eq!(ElMethod::GradientBoosting.to_string(), "GradientBoosting");
        assert_eq!(ElMethod::RandomForest.to_string(), "RandomForest");
        assert_eq!(ElMethod::Stacking.to_string(), "Stacking");
    }
    #[test]
    fn test_history_bounded_at_100() {
        let samples = linearly_separable_samples(30);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::Bagging,
            n_estimators: 120,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        assert!(el.training_history.len() <= 100);
    }
    #[test]
    fn test_history_records_have_correct_round_numbers() {
        let samples = linearly_separable_samples(20);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::Bagging,
            n_estimators: 5,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        let records: Vec<_> = el.training_history.iter().collect();
        for (i, r) in records.iter().enumerate() {
            assert_eq!(r.round, i);
        }
    }
    #[test]
    fn test_weighted_bootstrap_length() {
        let mut rng = 1u64;
        let weights = vec![0.5, 0.3, 0.2];
        let idxs = weighted_bootstrap(&mut rng, &weights, 10);
        assert_eq!(idxs.len(), 10);
    }
    #[test]
    fn test_weighted_bootstrap_in_range() {
        let mut rng = 1u64;
        let weights = vec![0.5, 0.3, 0.2];
        let idxs = weighted_bootstrap(&mut rng, &weights, 20);
        for &i in &idxs {
            assert!(i < 3);
        }
    }
    #[test]
    fn test_type_alias_works() {
        let _el: ElEnsembleLearner = EnsembleLearner::new(ElLearnerConfig::default());
    }
    #[test]
    fn test_adaboost_xor_dataset_runs() {
        let samples = binary_xor_samples(60, 77);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::AdaBoost,
            n_estimators: 20,
            ..Default::default()
        });
        let result = el.fit(&samples);
        assert!(result.is_ok());
    }
    #[test]
    fn test_bagging_xor_dataset_runs() {
        let samples = binary_xor_samples(60, 88);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::Bagging,
            n_estimators: 15,
            ..Default::default()
        });
        assert!(el.fit(&samples).is_ok());
    }
    #[test]
    fn test_bagging_subsample_half() {
        let samples = linearly_separable_samples(40);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            method: ElMethod::Bagging,
            n_estimators: 10,
            subsample: 0.5,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        assert_eq!(el.models().len(), 10);
    }
    #[test]
    fn test_refit_clears_old_models() {
        let samples = linearly_separable_samples(30);
        let mut el = EnsembleLearner::new(ElLearnerConfig {
            n_estimators: 5,
            ..Default::default()
        });
        el.fit(&samples).expect("test: should succeed");
        assert_eq!(el.models().len(), 5);
        el.config = ElLearnerConfig {
            n_estimators: 3,
            ..ElLearnerConfig::default()
        };
        el.fit(&samples).expect("test: should succeed");
        assert_eq!(el.models().len(), 3);
    }
}
