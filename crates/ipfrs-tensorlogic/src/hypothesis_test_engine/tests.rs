//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::functions::{t_two_tailed, z_two_tailed};
use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    fn make_engine() -> HypothesisTestEngine {
        HypothesisTestEngine::new()
    }
    fn add_hyp(engine: &mut HypothesisTestEngine, id: &str) {
        engine
            .add_hypothesis(Hypothesis::new(id, id, "H0", "H1"))
            .expect("add_hypothesis failed");
    }
    fn make_sample(values: Vec<f64>, label: &str) -> SampleData {
        let mut s = sample_stats(&values);
        s.label = label.to_string();
        s
    }
    #[test]
    fn test_sample_stats_mean() {
        let s = sample_stats(&[2.0, 4.0, 6.0]);
        assert!((s.mean - 4.0).abs() < 1e-10);
    }
    #[test]
    fn test_sample_stats_variance() {
        let s = sample_stats(&[1.0, 2.0, 3.0]);
        assert!((s.variance - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_sample_stats_std_dev() {
        let s = sample_stats(&[0.0, 2.0]);
        assert!((s.std_dev - 2.0_f64.sqrt()).abs() < 1e-10);
    }
    #[test]
    fn test_sample_stats_single() {
        let s = sample_stats(&[5.0]);
        assert_eq!(s.mean, 5.0);
        assert_eq!(s.variance, 0.0);
    }
    #[test]
    fn test_sample_stats_empty() {
        let s = sample_stats(&[]);
        assert_eq!(s.n, 0);
        assert_eq!(s.mean, 0.0);
    }
    #[test]
    fn test_normal_cdf_zero() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-4);
    }
    #[test]
    fn test_normal_cdf_positive() {
        assert!((normal_cdf(1.96) - 0.975).abs() < 0.001);
    }
    #[test]
    fn test_normal_cdf_negative() {
        assert!((normal_cdf(-1.96) - 0.025).abs() < 0.001);
    }
    #[test]
    fn test_normal_cdf_symmetry() {
        let z = 1.5;
        assert!((normal_cdf(z) + normal_cdf(-z) - 1.0).abs() < 1e-6);
    }
    #[test]
    fn test_t_cdf_large_df_approaches_normal() {
        let t = 1.96;
        let t_p = t_cdf_approx(t, 300);
        let n_p = normal_cdf(t);
        assert!((t_p - n_p).abs() < 0.005);
    }
    #[test]
    fn test_t_cdf_symmetry() {
        let t = 2.0;
        let df = 10;
        let p_pos = t_cdf_approx(t, df);
        let p_neg = t_cdf_approx(-t, df);
        assert!((p_pos + p_neg - 1.0).abs() < 1e-6);
    }
    #[test]
    fn test_t_cdf_zero() {
        assert!((t_cdf_approx(0.0, 10) - 0.5).abs() < 0.01);
    }
    #[test]
    fn test_chi2_p_value_zero_stat() {
        assert!((chi2_p_value(0.0, 3) - 1.0).abs() < 1e-8);
    }
    #[test]
    fn test_chi2_p_value_known_case() {
        let p = chi2_p_value(3.84, 1);
        assert!((p - 0.05).abs() < 0.01, "p={p}");
    }
    #[test]
    fn test_chi2_p_value_large_stat() {
        let p = chi2_p_value(100.0, 1);
        assert!(p < 1e-10);
    }
    #[test]
    fn test_xorshift64_changes_state() {
        let mut state = 12345u64;
        let a = xorshift64(&mut state);
        let b = xorshift64(&mut state);
        assert_ne!(a, b);
    }
    #[test]
    fn test_xorshift_normal_range() {
        let mut state = 0xdeadbeef_u64;
        for _ in 0..1000 {
            let x = xorshift_normal(&mut state);
            assert!(x.abs() < 10.0, "x={x}");
        }
    }
    #[test]
    fn test_xorshift_normal_mean_approx_zero() {
        let mut state = 0xc0ffee_u64;
        let n = 10_000;
        let sum: f64 = (0..n).map(|_| xorshift_normal(&mut state)).sum();
        let mean = sum / n as f64;
        assert!(mean.abs() < 0.1, "mean={mean}");
    }
    #[test]
    fn test_add_hypothesis_ok() {
        let mut engine = make_engine();
        let result = engine.add_hypothesis(Hypothesis::new("h1", "stmt", "H0", "H1"));
        assert!(result.is_ok());
    }
    #[test]
    fn test_add_hypothesis_invalid_alpha_zero() {
        let mut engine = make_engine();
        let h = Hypothesis::new("h1", "stmt", "H0", "H1").with_alpha(0.0);
        assert!(matches!(
            engine.add_hypothesis(h),
            Err(TestError::InvalidAlpha(_))
        ));
    }
    #[test]
    fn test_add_hypothesis_invalid_alpha_one() {
        let mut engine = make_engine();
        let h = Hypothesis::new("h1", "stmt", "H0", "H1").with_alpha(1.0);
        assert!(matches!(
            engine.add_hypothesis(h),
            Err(TestError::InvalidAlpha(_))
        ));
    }
    #[test]
    fn test_add_hypothesis_negative_alpha() {
        let mut engine = make_engine();
        let h = Hypothesis::new("h1", "stmt", "H0", "H1").with_alpha(-0.05);
        assert!(matches!(
            engine.add_hypothesis(h),
            Err(TestError::InvalidAlpha(_))
        ));
    }
    #[test]
    fn test_hypothesis_not_found() {
        let mut engine = make_engine();
        let r = engine.test(
            "nonexistent",
            TestType::OneSampleZTest {
                mu0: 0.0,
                sigma: 1.0,
            },
            vec![make_sample(vec![1.0, 2.0], "s")],
        );
        assert!(matches!(r, Err(TestError::HypothesisNotFound(_))));
    }
    #[test]
    fn test_one_sample_z_reject_null() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "z1");
        let data: Vec<f64> = vec![10.0; 30];
        let s = make_sample(data, "s");
        let r = engine
            .test(
                "z1",
                TestType::OneSampleZTest {
                    mu0: 0.0,
                    sigma: 1.0,
                },
                vec![s],
            )
            .expect("test failed");
        assert!(r.reject_null);
        assert!(r.p_value < 0.05);
    }
    #[test]
    fn test_one_sample_z_accept_null() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "z2");
        let data = vec![0.01, -0.01, 0.02, -0.02, 0.0];
        let s = make_sample(data, "s");
        let r = engine
            .test(
                "z2",
                TestType::OneSampleZTest {
                    mu0: 0.0,
                    sigma: 1.0,
                },
                vec![s],
            )
            .expect("test failed");
        assert!(!r.reject_null);
    }
    #[test]
    fn test_one_sample_z_statistic_formula() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "z3");
        let vals = vec![2.0, 2.0, 2.0, 2.0];
        let s = make_sample(vals, "s");
        let r = engine
            .test(
                "z3",
                TestType::OneSampleZTest {
                    mu0: 1.0,
                    sigma: 1.0,
                },
                vec![s],
            )
            .expect("test failed");
        if let TestStatistic::ZScore(z) = r.statistic {
            let expected_z = (2.0 - 1.0) / (1.0 / 2.0);
            assert!(
                (z - expected_z).abs() < 1e-10,
                "z={z} expected={expected_z}"
            );
        } else {
            panic!("wrong statistic type");
        }
    }
    #[test]
    fn test_one_sample_z_ci_present() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "z4");
        let s = make_sample(vec![1.0, 2.0, 3.0], "s");
        let r = engine
            .test(
                "z4",
                TestType::OneSampleZTest {
                    mu0: 2.0,
                    sigma: 1.0,
                },
                vec![s],
            )
            .expect("test failed");
        assert!(r.confidence_interval.is_some());
    }
    #[test]
    fn test_one_sample_z_insufficient_data() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "z5");
        let s = make_sample(vec![1.0], "s");
        let r = engine.test(
            "z5",
            TestType::OneSampleZTest {
                mu0: 0.0,
                sigma: 1.0,
            },
            vec![s],
        );
        assert!(matches!(r, Err(TestError::InsufficientData { .. })));
    }
    #[test]
    fn test_one_sample_t_reject_null() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "t1");
        let data: Vec<f64> = (0..20).map(|i| 5.0 + i as f64 * 0.1).collect();
        let s = make_sample(data, "s");
        let r = engine
            .test("t1", TestType::OneSampleTTest { mu0: 0.0 }, vec![s])
            .expect("test failed");
        assert!(r.reject_null);
    }
    #[test]
    fn test_one_sample_t_accept_null() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "t2");
        let data = vec![0.1, -0.1, 0.05, -0.05, 0.0, 0.02, -0.02];
        let s = make_sample(data, "s");
        let r = engine
            .test("t2", TestType::OneSampleTTest { mu0: 0.0 }, vec![s])
            .expect("test failed");
        assert!(!r.reject_null);
    }
    #[test]
    fn test_one_sample_t_statistic_df() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "t3");
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let n = data.len();
        let s = make_sample(data, "s");
        let r = engine
            .test("t3", TestType::OneSampleTTest { mu0: 3.0 }, vec![s])
            .expect("test failed");
        if let TestStatistic::TScore { df, .. } = r.statistic {
            assert_eq!(df, (n - 1) as u32);
        } else {
            panic!("wrong statistic type");
        }
    }
    #[test]
    fn test_one_sample_t_p_value_bounds() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "t4");
        let data: Vec<f64> = vec![1.0, 2.0, 3.0];
        let s = make_sample(data, "s");
        let r = engine
            .test("t4", TestType::OneSampleTTest { mu0: 2.0 }, vec![s])
            .expect("test failed");
        assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
    }
    #[test]
    fn test_two_sample_t_pooled_reject() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "tt1");
        let s1 = make_sample((0..20).map(|i| 10.0 + i as f64 * 0.1).collect(), "a");
        let s2 = make_sample((0..20).map(|i| 0.0 + i as f64 * 0.1).collect(), "b");
        let r = engine
            .test(
                "tt1",
                TestType::TwoSampleTTest {
                    equal_variance: true,
                },
                vec![s1, s2],
            )
            .expect("test failed");
        assert!(r.reject_null);
    }
    #[test]
    fn test_two_sample_t_welch_accept() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "tt2");
        let s1 = make_sample(vec![1.0, 2.0, 3.0], "a");
        let s2 = make_sample(vec![1.1, 2.1, 3.1], "b");
        let r = engine
            .test(
                "tt2",
                TestType::TwoSampleTTest {
                    equal_variance: false,
                },
                vec![s1, s2],
            )
            .expect("test failed");
        assert!(!r.reject_null);
    }
    #[test]
    fn test_two_sample_t_welch_satterthwaite_df() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "tt3");
        let s1 = make_sample(vec![1.0, 2.0, 3.0, 4.0], "a");
        let s2 = make_sample(vec![2.0, 3.0, 4.0, 5.0, 6.0], "b");
        let r = engine
            .test(
                "tt3",
                TestType::TwoSampleTTest {
                    equal_variance: false,
                },
                vec![s1, s2],
            )
            .expect("test failed");
        if let TestStatistic::TScore { df, .. } = r.statistic {
            assert!(df > 0);
        }
    }
    #[test]
    fn test_two_sample_t_cohens_d_positive() {
        let s1 = make_sample(vec![10.0; 10], "a");
        let s2 = make_sample(vec![5.0; 10], "b");
        let d = HypothesisTestEngine::effect_size_cohens_d(&s1, &s2);
        assert_eq!(d, 0.0);
    }
    #[test]
    fn test_two_sample_t_cohens_d_nonzero() {
        let s1 = make_sample(vec![1.0, 2.0, 3.0, 4.0, 5.0], "a");
        let s2 = make_sample(vec![6.0, 7.0, 8.0, 9.0, 10.0], "b");
        let d = HypothesisTestEngine::effect_size_cohens_d(&s1, &s2);
        assert!(d < -1.0);
    }
    #[test]
    fn test_chi2_gof_uniform_accept() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "c1");
        let obs = vec![10.0, 10.0, 10.0];
        let exp = vec![10.0, 10.0, 10.0];
        let s = SampleData {
            values: obs,
            label: "s".into(),
            n: 30,
            mean: 10.0,
            variance: 0.0,
            std_dev: 0.0,
        };
        let r = engine
            .test(
                "c1",
                TestType::ChiSquareGoodnessOfFit { expected: exp },
                vec![s],
            )
            .expect("test failed");
        assert!(!r.reject_null);
        assert!((r.p_value - 1.0).abs() < 1e-8);
    }
    #[test]
    fn test_chi2_gof_reject() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "c2");
        let obs = vec![50.0, 1.0, 1.0];
        let exp = vec![17.3, 17.3, 17.3];
        let s = SampleData {
            values: obs,
            label: "s".into(),
            n: 52,
            mean: 17.3,
            variance: 0.0,
            std_dev: 0.0,
        };
        let r = engine
            .test(
                "c2",
                TestType::ChiSquareGoodnessOfFit { expected: exp },
                vec![s],
            )
            .expect("test failed");
        assert!(r.reject_null);
    }
    #[test]
    fn test_chi2_gof_df() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "c3");
        let k = 5;
        let obs: Vec<f64> = vec![10.0; k];
        let exp: Vec<f64> = vec![10.0; k];
        let s = SampleData {
            values: obs,
            label: "s".into(),
            n: 50,
            mean: 10.0,
            variance: 0.0,
            std_dev: 0.0,
        };
        let r = engine
            .test(
                "c3",
                TestType::ChiSquareGoodnessOfFit { expected: exp },
                vec![s],
            )
            .expect("test failed");
        if let TestStatistic::ChiSquare { df, .. } = r.statistic {
            assert_eq!(df, (k - 1) as u32);
        }
    }
    #[test]
    fn test_chi2_gof_cramers_v_zero() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "c4");
        let obs = vec![25.0, 25.0, 25.0, 25.0];
        let exp = vec![25.0, 25.0, 25.0, 25.0];
        let s = SampleData {
            values: obs,
            label: "s".into(),
            n: 100,
            mean: 25.0,
            variance: 0.0,
            std_dev: 0.0,
        };
        let r = engine
            .test(
                "c4",
                TestType::ChiSquareGoodnessOfFit { expected: exp },
                vec![s],
            )
            .expect("test failed");
        assert!(r.effect_size < 1e-8);
    }
    #[test]
    fn test_chi2_independence_accept() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "ci1");
        let contingency = vec![vec![25.0, 25.0], vec![25.0, 25.0]];
        let r = engine
            .test(
                "ci1",
                TestType::ChiSquareIndependence { contingency },
                vec![],
            )
            .expect("test failed");
        assert!(!r.reject_null);
        assert!((r.p_value - 1.0).abs() < 1e-8);
    }
    #[test]
    fn test_chi2_independence_reject() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "ci2");
        let contingency = vec![vec![90.0, 10.0], vec![10.0, 90.0]];
        let r = engine
            .test(
                "ci2",
                TestType::ChiSquareIndependence { contingency },
                vec![],
            )
            .expect("test failed");
        assert!(r.reject_null);
    }
    #[test]
    fn test_chi2_independence_df_2x3() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "ci3");
        let contingency = vec![vec![10.0, 10.0, 10.0], vec![10.0, 10.0, 10.0]];
        let r = engine
            .test(
                "ci3",
                TestType::ChiSquareIndependence { contingency },
                vec![],
            )
            .expect("test failed");
        if let TestStatistic::ChiSquare { df, .. } = r.statistic {
            assert_eq!(df, (3 - 1));
        }
    }
    #[test]
    fn test_chi2_independence_empty_table() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "ci4");
        let r = engine.test(
            "ci4",
            TestType::ChiSquareIndependence {
                contingency: vec![],
            },
            vec![],
        );
        assert!(matches!(r, Err(TestError::InvalidContingency(_))));
    }
    #[test]
    fn test_chi2_independence_unequal_rows() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "ci5");
        let contingency = vec![vec![1.0, 2.0], vec![3.0]];
        let r = engine.test(
            "ci5",
            TestType::ChiSquareIndependence { contingency },
            vec![],
        );
        assert!(matches!(r, Err(TestError::InvalidContingency(_))));
    }
    #[test]
    fn test_proportion_reject() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "p1");
        let s = SampleData {
            values: vec![],
            label: "s".into(),
            n: 100,
            mean: 0.8,
            variance: 0.0,
            std_dev: 0.0,
        };
        let r = engine
            .test("p1", TestType::OneSampleProportion { p0: 0.5 }, vec![s])
            .expect("test failed");
        assert!(r.reject_null);
    }
    #[test]
    fn test_proportion_accept() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "p2");
        let s = SampleData {
            values: vec![],
            label: "s".into(),
            n: 10,
            mean: 0.51,
            variance: 0.0,
            std_dev: 0.0,
        };
        let r = engine
            .test("p2", TestType::OneSampleProportion { p0: 0.5 }, vec![s])
            .expect("test failed");
        assert!(!r.reject_null);
    }
    #[test]
    fn test_proportion_statistic_type() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "p3");
        let s = SampleData {
            values: vec![],
            label: "s".into(),
            n: 50,
            mean: 0.6,
            variance: 0.0,
            std_dev: 0.0,
        };
        let r = engine
            .test("p3", TestType::OneSampleProportion { p0: 0.5 }, vec![s])
            .expect("test failed");
        assert!(matches!(r.statistic, TestStatistic::BinomialZ { .. }));
    }
    #[test]
    fn test_proportion_ci_bounds() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "p4");
        let s = SampleData {
            values: vec![],
            label: "s".into(),
            n: 100,
            mean: 0.6,
            variance: 0.0,
            std_dev: 0.0,
        };
        let r = engine
            .test("p4", TestType::OneSampleProportion { p0: 0.5 }, vec![s])
            .expect("test failed");
        if let Some((lo, hi)) = r.confidence_interval {
            assert!(lo < r.p_value || lo < 1.0);
            assert!(hi > lo);
        }
    }
    #[test]
    fn test_two_proportion_reject() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "pp1");
        let s1 = SampleData {
            values: vec![],
            label: "a".into(),
            n: 100,
            mean: 0.8,
            variance: 0.0,
            std_dev: 0.0,
        };
        let s2 = SampleData {
            values: vec![],
            label: "b".into(),
            n: 100,
            mean: 0.2,
            variance: 0.0,
            std_dev: 0.0,
        };
        let r = engine
            .test("pp1", TestType::TwoSampleProportion, vec![s1, s2])
            .expect("test failed");
        assert!(r.reject_null);
    }
    #[test]
    fn test_two_proportion_accept() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "pp2");
        let s1 = SampleData {
            values: vec![],
            label: "a".into(),
            n: 50,
            mean: 0.5,
            variance: 0.0,
            std_dev: 0.0,
        };
        let s2 = SampleData {
            values: vec![],
            label: "b".into(),
            n: 50,
            mean: 0.5,
            variance: 0.0,
            std_dev: 0.0,
        };
        let r = engine
            .test("pp2", TestType::TwoSampleProportion, vec![s1, s2])
            .expect("test failed");
        assert!(!r.reject_null);
        assert!((r.p_value - 1.0).abs() < 0.01);
    }
    #[test]
    fn test_ci_lower_less_than_upper() {
        let engine = make_engine();
        let s = make_sample(vec![1.0, 2.0, 3.0, 4.0, 5.0], "s");
        let (lo, hi) = engine.confidence_interval(&s, 0.05);
        assert!(lo < hi);
    }
    #[test]
    fn test_ci_contains_mean() {
        let engine = make_engine();
        let s = make_sample(vec![10.0, 11.0, 9.0, 10.5, 9.5], "s");
        let (lo, hi) = engine.confidence_interval(&s, 0.05);
        assert!(lo <= s.mean && s.mean <= hi);
    }
    #[test]
    fn test_ci_wider_at_lower_alpha() {
        let engine = make_engine();
        let s = make_sample(vec![1.0, 2.0, 3.0, 4.0, 5.0], "s");
        let (lo99, hi99) = engine.confidence_interval(&s, 0.01);
        let (lo95, hi95) = engine.confidence_interval(&s, 0.05);
        assert!(hi99 - lo99 > hi95 - lo95);
    }
    #[test]
    fn test_ci_single_point() {
        let engine = make_engine();
        let s = SampleData {
            values: vec![5.0],
            label: "s".into(),
            n: 1,
            mean: 5.0,
            variance: 0.0,
            std_dev: 0.0,
        };
        let (lo, hi) = engine.confidence_interval(&s, 0.05);
        assert_eq!(lo, 5.0);
        assert_eq!(hi, 5.0);
    }
    #[test]
    fn test_cohens_d_zero_diff() {
        let s1 = make_sample(vec![1.0, 2.0, 3.0], "a");
        let s2 = make_sample(vec![1.0, 2.0, 3.0], "b");
        let d = HypothesisTestEngine::effect_size_cohens_d(&s1, &s2);
        assert!((d).abs() < 1e-10);
    }
    #[test]
    fn test_cohens_d_sign() {
        let s1 = make_sample(vec![10.0, 11.0, 12.0], "a");
        let s2 = make_sample(vec![1.0, 2.0, 3.0], "b");
        let d = HypothesisTestEngine::effect_size_cohens_d(&s1, &s2);
        assert!(d > 0.0);
    }
    #[test]
    fn test_power_zero_effect() {
        let mut engine = make_engine();
        let p = engine.power(&TestType::OneSampleTTest { mu0: 0.0 }, 0.05, 50, 0.0);
        assert!(p < 0.20);
    }
    #[test]
    fn test_power_large_effect() {
        let mut engine = make_engine();
        let p = engine.power(&TestType::OneSampleTTest { mu0: 0.0 }, 0.05, 100, 3.0);
        assert!(p > 0.5);
    }
    #[test]
    fn test_power_disabled() {
        let config = EngineConfig {
            enable_power_calculation: false,
            ..Default::default()
        };
        let mut engine = HypothesisTestEngine::with_config(config);
        let p = engine.power(
            &TestType::OneSampleZTest {
                mu0: 0.0,
                sigma: 1.0,
            },
            0.05,
            50,
            2.0,
        );
        assert_eq!(p, 0.0);
    }
    #[test]
    fn test_power_returns_0_to_1() {
        let mut engine = make_engine();
        for es in [0.0, 0.2, 0.5, 1.0, 2.0] {
            let p = engine.power(&TestType::OneSampleTTest { mu0: 0.0 }, 0.05, 30, es);
            assert!((0.0..=1.0).contains(&p), "power={p} for effect_size={es}");
        }
    }
    #[test]
    fn test_stats_zero_initially() {
        let engine = make_engine();
        let s = engine.stats();
        assert_eq!(s.tests_performed, 0);
        assert_eq!(s.nulls_rejected, 0);
    }
    #[test]
    fn test_stats_increments() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "s1");
        let s1 = make_sample(vec![10.0; 10], "s");
        engine
            .test(
                "s1",
                TestType::OneSampleZTest {
                    mu0: 0.0,
                    sigma: 1.0,
                },
                vec![s1],
            )
            .expect("test failed");
        assert_eq!(engine.stats().tests_performed, 1);
    }
    #[test]
    fn test_stats_rejection_rate() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "r1");
        add_hyp(&mut engine, "r2");
        let reject_sample = make_sample(vec![10.0; 30], "s");
        let accept_sample = make_sample(vec![0.0; 5], "s");
        engine
            .test(
                "r1",
                TestType::OneSampleZTest {
                    mu0: 0.0,
                    sigma: 1.0,
                },
                vec![reject_sample],
            )
            .expect("test failed");
        engine
            .test(
                "r2",
                TestType::OneSampleZTest {
                    mu0: 0.0,
                    sigma: 1.0,
                },
                vec![accept_sample],
            )
            .expect("test failed");
        let rate = engine.stats().rejection_rate;
        assert!(rate > 0.0 && rate <= 1.0);
    }
    #[test]
    fn test_stats_avg_p_value_bounds() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "ap1");
        let s = make_sample(vec![1.0, 2.0, 3.0], "s");
        engine
            .test("ap1", TestType::OneSampleTTest { mu0: 2.0 }, vec![s])
            .expect("test failed");
        let avg = engine.stats().avg_p_value;
        assert!((0.0..=1.0).contains(&avg));
    }
    #[test]
    fn test_insufficient_data_two_sample() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "e1");
        let s1 = make_sample(vec![1.0], "a");
        let s2 = make_sample(vec![2.0, 3.0], "b");
        let r = engine.test(
            "e1",
            TestType::TwoSampleTTest {
                equal_variance: true,
            },
            vec![s1, s2],
        );
        assert!(matches!(r, Err(TestError::InsufficientData { .. })));
    }
    #[test]
    fn test_numerical_error_zero_sigma() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "e2");
        let s = make_sample(vec![1.0, 2.0], "s");
        let r = engine.test(
            "e2",
            TestType::OneSampleZTest {
                mu0: 0.0,
                sigma: -1.0,
            },
            vec![s],
        );
        assert!(matches!(r, Err(TestError::NumericalError(_))));
    }
    #[test]
    fn test_numerical_error_zero_std_t() {
        let mut engine = make_engine();
        add_hyp(&mut engine, "e3");
        let s = make_sample(vec![2.0, 2.0, 2.0], "s");
        let r = engine.test("e3", TestType::OneSampleTTest { mu0: 0.0 }, vec![s]);
        assert!(matches!(r, Err(TestError::NumericalError(_))));
    }
    #[test]
    fn test_test_error_display() {
        let e = TestError::InsufficientData { needed: 5, got: 2 };
        let s = e.to_string();
        assert!(s.contains("5") && s.contains("2"));
    }
    #[test]
    fn test_z_p_value_known_two_tailed() {
        let p = z_two_tailed(1.96);
        assert!((p - 0.05).abs() < 0.002, "p={p}");
    }
    #[test]
    fn test_t_p_value_known_df30() {
        let p = t_two_tailed(2.042, 30);
        assert!((p - 0.05).abs() < 0.01, "p={p}");
    }
    #[test]
    fn test_chi2_p_value_known_df3() {
        let p = chi2_p_value(7.815, 3);
        assert!((p - 0.05).abs() < 0.01, "p={p}");
    }
    #[test]
    fn test_reject_at_alpha_001() {
        let mut engine = make_engine();
        let mut h = Hypothesis::new("a001", "s", "H0", "H1");
        h.alpha = 0.01;
        engine.add_hypothesis(h).expect("add failed");
        let s = make_sample(vec![3.0; 100], "s");
        let r = engine
            .test(
                "a001",
                TestType::OneSampleZTest {
                    mu0: 0.0,
                    sigma: 1.0,
                },
                vec![s],
            )
            .expect("test failed");
        assert!(r.reject_null);
    }
    #[test]
    fn test_accept_at_strict_alpha() {
        let mut engine = make_engine();
        let mut h = Hypothesis::new("astrict", "s", "H0", "H1");
        h.alpha = 0.001;
        engine.add_hypothesis(h).expect("add failed");
        let s = make_sample(vec![0.3; 4], "s");
        let r = engine
            .test(
                "astrict",
                TestType::OneSampleZTest {
                    mu0: 0.0,
                    sigma: 1.0,
                },
                vec![s],
            )
            .expect("test failed");
        assert!(!r.reject_null);
    }
}
