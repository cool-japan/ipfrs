//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    /// Inline xorshift64 PRNG — no rand crate needed.
    fn xorshift64(state: &mut u64) -> f64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        (x as f64) / (u64::MAX as f64)
    }
    /// Build a minimal temperature-input → fan-speed-output engine.
    fn build_temp_engine(method: DefuzzMethod) -> FuzzyLogicEngine {
        let mut temp = FuzzyVariable::new("temp");
        temp.sets.push(FuzzySet::new(
            "cold",
            MembershipFunction::Triangle {
                a: 0.0,
                b: 10.0,
                c: 20.0,
            },
            0.0,
            100.0,
        ));
        temp.sets.push(FuzzySet::new(
            "warm",
            MembershipFunction::Triangle {
                a: 15.0,
                b: 25.0,
                c: 35.0,
            },
            0.0,
            100.0,
        ));
        temp.sets.push(FuzzySet::new(
            "hot",
            MembershipFunction::Triangle {
                a: 30.0,
                b: 50.0,
                c: 70.0,
            },
            0.0,
            100.0,
        ));
        let mut fan = FuzzyVariable::new("fan");
        fan.sets.push(FuzzySet::new(
            "slow",
            MembershipFunction::Triangle {
                a: 0.0,
                b: 25.0,
                c: 50.0,
            },
            0.0,
            100.0,
        ));
        fan.sets.push(FuzzySet::new(
            "fast",
            MembershipFunction::Triangle {
                a: 50.0,
                b: 75.0,
                c: 100.0,
            },
            0.0,
            100.0,
        ));
        let r1 = FuzzyRule::new(
            "r1",
            FuzzyExpr::Is {
                var: "temp".into(),
                set: "cold".into(),
            },
            FuzzyExpr::Is {
                var: "fan".into(),
                set: "slow".into(),
            },
            1.0,
        );
        let r2 = FuzzyRule::new(
            "r2",
            FuzzyExpr::Is {
                var: "temp".into(),
                set: "hot".into(),
            },
            FuzzyExpr::Is {
                var: "fan".into(),
                set: "fast".into(),
            },
            1.0,
        );
        let cfg = EngineConfig::new(vec![temp], vec![fan], vec![r1, r2], method);
        FuzzyLogicEngine::new(cfg).expect("valid engine")
    }
    #[test]
    fn triangle_peak() {
        let mf = MembershipFunction::Triangle {
            a: 0.0,
            b: 5.0,
            c: 10.0,
        };
        assert!((mf.evaluate(5.0) - 1.0).abs() < 1e-10);
    }
    #[test]
    fn triangle_left_foot() {
        let mf = MembershipFunction::Triangle {
            a: 0.0,
            b: 5.0,
            c: 10.0,
        };
        assert_eq!(mf.evaluate(0.0), 0.0);
    }
    #[test]
    fn triangle_right_foot() {
        let mf = MembershipFunction::Triangle {
            a: 0.0,
            b: 5.0,
            c: 10.0,
        };
        assert_eq!(mf.evaluate(10.0), 0.0);
    }
    #[test]
    fn triangle_midpoint_left() {
        let mf = MembershipFunction::Triangle {
            a: 0.0,
            b: 10.0,
            c: 20.0,
        };
        assert!((mf.evaluate(5.0) - 0.5).abs() < 1e-10);
    }
    #[test]
    fn triangle_midpoint_right() {
        let mf = MembershipFunction::Triangle {
            a: 0.0,
            b: 10.0,
            c: 20.0,
        };
        assert!((mf.evaluate(15.0) - 0.5).abs() < 1e-10);
    }
    #[test]
    fn triangle_outside_range() {
        let mf = MembershipFunction::Triangle {
            a: 2.0,
            b: 5.0,
            c: 8.0,
        };
        assert_eq!(mf.evaluate(1.0), 0.0);
        assert_eq!(mf.evaluate(9.0), 0.0);
    }
    #[test]
    fn trapezoid_flat_top() {
        let mf = MembershipFunction::Trapezoid {
            a: 0.0,
            b: 2.0,
            c: 6.0,
            d: 8.0,
        };
        assert_eq!(mf.evaluate(4.0), 1.0);
    }
    #[test]
    fn trapezoid_rising_slope() {
        let mf = MembershipFunction::Trapezoid {
            a: 0.0,
            b: 4.0,
            c: 6.0,
            d: 10.0,
        };
        assert!((mf.evaluate(2.0) - 0.5).abs() < 1e-10);
    }
    #[test]
    fn trapezoid_falling_slope() {
        let mf = MembershipFunction::Trapezoid {
            a: 0.0,
            b: 4.0,
            c: 6.0,
            d: 10.0,
        };
        assert!((mf.evaluate(8.0) - 0.5).abs() < 1e-10);
    }
    #[test]
    fn trapezoid_outside() {
        let mf = MembershipFunction::Trapezoid {
            a: 1.0,
            b: 3.0,
            c: 7.0,
            d: 9.0,
        };
        assert_eq!(mf.evaluate(0.5), 0.0);
        assert_eq!(mf.evaluate(9.5), 0.0);
    }
    #[test]
    fn gaussian_peak_at_mean() {
        let mf = MembershipFunction::Gaussian {
            mean: 5.0,
            sigma: 1.0,
        };
        assert!((mf.evaluate(5.0) - 1.0).abs() < 1e-10);
    }
    #[test]
    fn gaussian_half_sigma() {
        let mf = MembershipFunction::Gaussian {
            mean: 0.0,
            sigma: 1.0,
        };
        let expected = (-0.5_f64).exp();
        assert!((mf.evaluate(1.0) - expected).abs() < 1e-9);
    }
    #[test]
    fn gaussian_zero_sigma_at_mean() {
        let mf = MembershipFunction::Gaussian {
            mean: 3.0,
            sigma: 0.0,
        };
        assert_eq!(mf.evaluate(3.0), 1.0);
        assert_eq!(mf.evaluate(3.1), 0.0);
    }
    #[test]
    fn bell_peak_at_centre() {
        let mf = MembershipFunction::Bell {
            a: 2.0,
            b: 4.0,
            c: 5.0,
        };
        assert!((mf.evaluate(5.0) - 1.0).abs() < 1e-10);
    }
    #[test]
    fn bell_half_at_a_distance() {
        let mf = MembershipFunction::Bell {
            a: 3.0,
            b: 1.0,
            c: 0.0,
        };
        assert!((mf.evaluate(3.0) - 0.5).abs() < 1e-10);
    }
    #[test]
    fn sigmoid_inflection_is_half() {
        let mf = MembershipFunction::Sigmoid { a: 2.0, c: 5.0 };
        assert!((mf.evaluate(5.0) - 0.5).abs() < 1e-10);
    }
    #[test]
    fn sigmoid_rising() {
        let mf = MembershipFunction::Sigmoid { a: 1.0, c: 0.0 };
        assert!(mf.evaluate(5.0) > 0.9);
    }
    #[test]
    fn sigmoid_falling() {
        let mf = MembershipFunction::Sigmoid { a: -1.0, c: 0.0 };
        assert!(mf.evaluate(5.0) < 0.1);
    }
    #[test]
    fn singleton_at_value() {
        let mf = MembershipFunction::Singleton(7.5);
        assert_eq!(mf.evaluate(7.5), 1.0);
    }
    #[test]
    fn singleton_away() {
        let mf = MembershipFunction::Singleton(7.5);
        assert_eq!(mf.evaluate(7.6), 0.0);
    }
    #[test]
    fn linear_midpoint() {
        let mf = MembershipFunction::Linear {
            x0: 0.0,
            x1: 10.0,
            y0: 0.0,
            y1: 1.0,
        };
        assert!((mf.evaluate(5.0) - 0.5).abs() < 1e-10);
    }
    #[test]
    fn linear_clamp_below() {
        let mf = MembershipFunction::Linear {
            x0: 0.0,
            x1: 10.0,
            y0: 0.3,
            y1: 0.8,
        };
        assert!(mf.evaluate(-5.0) >= 0.0);
    }
    #[test]
    fn linear_clamp_above() {
        let mf = MembershipFunction::Linear {
            x0: 0.0,
            x1: 10.0,
            y0: 0.3,
            y1: 1.5,
        };
        assert_eq!(mf.evaluate(10.0), 1.0);
    }
    fn make_engine_for_expr() -> FuzzyLogicEngine {
        let mut a = FuzzyVariable::new("a");
        a.sets.push(FuzzySet::new(
            "high",
            MembershipFunction::Singleton(1.0),
            0.0,
            1.0,
        ));
        a.sets.push(FuzzySet::new(
            "low",
            MembershipFunction::Singleton(0.0),
            0.0,
            1.0,
        ));
        let mut b = FuzzyVariable::new("b");
        b.sets.push(FuzzySet::new(
            "high",
            MembershipFunction::Singleton(1.0),
            0.0,
            1.0,
        ));
        b.sets.push(FuzzySet::new(
            "low",
            MembershipFunction::Singleton(0.0),
            0.0,
            1.0,
        ));
        let mut out = FuzzyVariable::new("out");
        out.sets.push(FuzzySet::new(
            "yes",
            MembershipFunction::Triangle {
                a: 0.0,
                b: 0.5,
                c: 1.0,
            },
            0.0,
            1.0,
        ));
        let rule = FuzzyRule::new(
            "dummy",
            FuzzyExpr::Is {
                var: "a".into(),
                set: "high".into(),
            },
            FuzzyExpr::Is {
                var: "out".into(),
                set: "yes".into(),
            },
            1.0,
        );
        FuzzyLogicEngine::new(EngineConfig::new(
            vec![a, b],
            vec![out],
            vec![rule],
            DefuzzMethod::Centroid,
        ))
        .expect("valid engine")
    }
    #[test]
    fn expr_is_high() {
        let mut e = make_engine_for_expr();
        e.set_input("a", 1.0).expect("test: should succeed");
        let rule = FuzzyRule::new(
            "t",
            FuzzyExpr::Is {
                var: "a".into(),
                set: "high".into(),
            },
            FuzzyExpr::Is {
                var: "out".into(),
                set: "yes".into(),
            },
            1.0,
        );
        assert!((e.evaluate_rule(&rule).expect("test: should succeed") - 1.0).abs() < 1e-10);
    }
    #[test]
    fn expr_is_low() {
        let mut e = make_engine_for_expr();
        e.set_input("a", 0.5).expect("test: should succeed");
        let rule = FuzzyRule::new(
            "t",
            FuzzyExpr::Is {
                var: "a".into(),
                set: "high".into(),
            },
            FuzzyExpr::Is {
                var: "out".into(),
                set: "yes".into(),
            },
            1.0,
        );
        assert_eq!(e.evaluate_rule(&rule).expect("test: should succeed"), 0.0);
    }
    #[test]
    fn expr_and() {
        let mut e = make_engine_for_expr();
        e.set_input("a", 1.0).expect("test: should succeed");
        e.set_input("b", 1.0).expect("test: should succeed");
        let rule = FuzzyRule::new(
            "t",
            FuzzyExpr::And(
                Box::new(FuzzyExpr::Is {
                    var: "a".into(),
                    set: "high".into(),
                }),
                Box::new(FuzzyExpr::Is {
                    var: "b".into(),
                    set: "high".into(),
                }),
            ),
            FuzzyExpr::Is {
                var: "out".into(),
                set: "yes".into(),
            },
            1.0,
        );
        assert!((e.evaluate_rule(&rule).expect("test: should succeed") - 1.0).abs() < 1e-10);
    }
    #[test]
    fn expr_and_mixed() {
        let mut e = make_engine_for_expr();
        e.set_input("a", 1.0).expect("test: should succeed");
        e.set_input("b", 0.5).expect("test: should succeed");
        let rule = FuzzyRule::new(
            "t",
            FuzzyExpr::And(
                Box::new(FuzzyExpr::Is {
                    var: "a".into(),
                    set: "high".into(),
                }),
                Box::new(FuzzyExpr::Is {
                    var: "b".into(),
                    set: "high".into(),
                }),
            ),
            FuzzyExpr::Is {
                var: "out".into(),
                set: "yes".into(),
            },
            1.0,
        );
        assert_eq!(e.evaluate_rule(&rule).expect("test: should succeed"), 0.0);
    }
    #[test]
    fn expr_or() {
        let mut e = make_engine_for_expr();
        e.set_input("a", 0.5).expect("test: should succeed");
        e.set_input("b", 1.0).expect("test: should succeed");
        let rule = FuzzyRule::new(
            "t",
            FuzzyExpr::Or(
                Box::new(FuzzyExpr::Is {
                    var: "a".into(),
                    set: "high".into(),
                }),
                Box::new(FuzzyExpr::Is {
                    var: "b".into(),
                    set: "high".into(),
                }),
            ),
            FuzzyExpr::Is {
                var: "out".into(),
                set: "yes".into(),
            },
            1.0,
        );
        assert!((e.evaluate_rule(&rule).expect("test: should succeed") - 1.0).abs() < 1e-10);
    }
    #[test]
    fn expr_not() {
        let mut e = make_engine_for_expr();
        e.set_input("a", 1.0).expect("test: should succeed");
        let rule = FuzzyRule::new(
            "t",
            FuzzyExpr::Not(Box::new(FuzzyExpr::Is {
                var: "a".into(),
                set: "high".into(),
            })),
            FuzzyExpr::Is {
                var: "out".into(),
                set: "yes".into(),
            },
            1.0,
        );
        assert_eq!(e.evaluate_rule(&rule).expect("test: should succeed"), 0.0);
    }
    #[test]
    fn expr_very_concentration() {
        let mut temp = FuzzyVariable::new("t");
        temp.sets.push(FuzzySet::new(
            "med",
            MembershipFunction::Gaussian {
                mean: 0.0,
                sigma: 1.0,
            },
            -5.0,
            5.0,
        ));
        let mut out = FuzzyVariable::new("out");
        out.sets.push(FuzzySet::new(
            "yes",
            MembershipFunction::Triangle {
                a: 0.0,
                b: 0.5,
                c: 1.0,
            },
            0.0,
            1.0,
        ));
        let rule = FuzzyRule::new(
            "r",
            FuzzyExpr::Very(Box::new(FuzzyExpr::Is {
                var: "t".into(),
                set: "med".into(),
            })),
            FuzzyExpr::Is {
                var: "out".into(),
                set: "yes".into(),
            },
            1.0,
        );
        let cfg = EngineConfig::new(
            vec![temp],
            vec![out],
            vec![rule.clone()],
            DefuzzMethod::Centroid,
        );
        let mut e = FuzzyLogicEngine::new(cfg).expect("test: should succeed");
        e.set_input("t", 1.0).expect("test: should succeed");
        let alpha = e.evaluate_rule(&rule).expect("test: should succeed");
        let expected = (-0.5_f64).exp().powi(2);
        assert!((alpha - expected).abs() < 1e-6);
    }
    #[test]
    fn expr_somewhat_dilation() {
        let mut temp = FuzzyVariable::new("t");
        temp.sets.push(FuzzySet::new(
            "med",
            MembershipFunction::Gaussian {
                mean: 0.0,
                sigma: 1.0,
            },
            -5.0,
            5.0,
        ));
        let mut out = FuzzyVariable::new("out");
        out.sets.push(FuzzySet::new(
            "yes",
            MembershipFunction::Triangle {
                a: 0.0,
                b: 0.5,
                c: 1.0,
            },
            0.0,
            1.0,
        ));
        let rule = FuzzyRule::new(
            "r",
            FuzzyExpr::Somewhat(Box::new(FuzzyExpr::Is {
                var: "t".into(),
                set: "med".into(),
            })),
            FuzzyExpr::Is {
                var: "out".into(),
                set: "yes".into(),
            },
            1.0,
        );
        let cfg = EngineConfig::new(
            vec![temp],
            vec![out],
            vec![rule.clone()],
            DefuzzMethod::Centroid,
        );
        let mut e = FuzzyLogicEngine::new(cfg).expect("test: should succeed");
        e.set_input("t", 1.0).expect("test: should succeed");
        let alpha = e.evaluate_rule(&rule).expect("test: should succeed");
        let expected = (-0.5_f64).exp().sqrt();
        assert!((alpha - expected).abs() < 1e-6);
    }
    #[test]
    fn mamdani_cold_input_activates_slow_fan() {
        let mut e = build_temp_engine(DefuzzMethod::Centroid);
        e.set_input("temp", 10.0).expect("test: should succeed");
        let result = e.infer_single("fan").expect("test: should succeed");
        assert!(
            result.crisp_value < 50.0,
            "expected slow fan, got {}",
            result.crisp_value
        );
    }
    #[test]
    fn mamdani_hot_input_activates_fast_fan() {
        let mut e = build_temp_engine(DefuzzMethod::Centroid);
        e.set_input("temp", 50.0).expect("test: should succeed");
        let result = e.infer_single("fan").expect("test: should succeed");
        assert!(
            result.crisp_value > 50.0,
            "expected fast fan, got {}",
            result.crisp_value
        );
    }
    #[test]
    fn mamdani_result_has_correct_output_var() {
        let mut e = build_temp_engine(DefuzzMethod::Centroid);
        e.set_input("temp", 10.0).expect("test: should succeed");
        let result = e.infer_single("fan").expect("test: should succeed");
        assert_eq!(result.output_var, "fan");
    }
    #[test]
    fn mamdani_activation_map_populated() {
        let mut e = build_temp_engine(DefuzzMethod::Centroid);
        e.set_input("temp", 10.0).expect("test: should succeed");
        let result = e.infer_single("fan").expect("test: should succeed");
        assert_eq!(result.activation_map.len(), 2);
    }
    #[test]
    fn mamdani_dominant_set_non_empty() {
        let mut e = build_temp_engine(DefuzzMethod::Centroid);
        e.set_input("temp", 50.0).expect("test: should succeed");
        let result = e.infer_single("fan").expect("test: should succeed");
        assert!(!result.dominant_set.is_empty());
    }
    #[test]
    fn mamdani_infer_all_output_vars() {
        let mut e = build_temp_engine(DefuzzMethod::Centroid);
        e.set_input("temp", 10.0).expect("test: should succeed");
        let results = e.infer().expect("test: should succeed");
        assert_eq!(results.len(), 1);
    }
    #[test]
    fn defuzz_centroid() {
        let mut e = build_temp_engine(DefuzzMethod::Centroid);
        e.set_input("temp", 10.0).expect("test: should succeed");
        let r = e.infer_single("fan").expect("test: should succeed");
        assert!(r.crisp_value > 0.0 && r.crisp_value < 100.0);
    }
    #[test]
    fn defuzz_bisector() {
        let mut e = build_temp_engine(DefuzzMethod::Bisector);
        e.set_input("temp", 10.0).expect("test: should succeed");
        let r = e.infer_single("fan").expect("test: should succeed");
        assert!(r.crisp_value > 0.0 && r.crisp_value < 100.0);
    }
    #[test]
    fn defuzz_mean_of_maxima() {
        let mut e = build_temp_engine(DefuzzMethod::MeanOfMaxima);
        e.set_input("temp", 10.0).expect("test: should succeed");
        let r = e.infer_single("fan").expect("test: should succeed");
        assert!(r.crisp_value > 0.0 && r.crisp_value < 100.0);
    }
    #[test]
    fn defuzz_largest_of_maxima() {
        let mut e = build_temp_engine(DefuzzMethod::LargestOfMaxima);
        e.set_input("temp", 10.0).expect("test: should succeed");
        let r = e.infer_single("fan").expect("test: should succeed");
        assert!(r.crisp_value > 0.0 && r.crisp_value < 100.0);
    }
    #[test]
    fn defuzz_smallest_of_maxima() {
        let mut e = build_temp_engine(DefuzzMethod::SmallestOfMaxima);
        e.set_input("temp", 10.0).expect("test: should succeed");
        let r = e.infer_single("fan").expect("test: should succeed");
        assert!(r.crisp_value > 0.0 && r.crisp_value < 100.0);
    }
    #[test]
    fn defuzz_som_lom_ordering() {
        let mut som = build_temp_engine(DefuzzMethod::SmallestOfMaxima);
        let mut lom = build_temp_engine(DefuzzMethod::LargestOfMaxima);
        som.set_input("temp", 50.0).expect("test: should succeed");
        lom.set_input("temp", 50.0).expect("test: should succeed");
        let r_som = som.infer_single("fan").expect("test: should succeed");
        let r_lom = lom.infer_single("fan").expect("test: should succeed");
        assert!(r_som.crisp_value <= r_lom.crisp_value);
    }
    #[test]
    fn set_input_ok() {
        let mut e = build_temp_engine(DefuzzMethod::Centroid);
        assert!(e.set_input("temp", 25.0).is_ok());
    }
    #[test]
    fn set_input_unknown_var_error() {
        let mut e = build_temp_engine(DefuzzMethod::Centroid);
        assert!(matches!(
            e.set_input("humidity", 50.0),
            Err(FuzzyError::VariableNotFound(_))
        ));
    }
    #[test]
    fn membership_cold() {
        let mut e = build_temp_engine(DefuzzMethod::Centroid);
        e.set_input("temp", 10.0).expect("test: should succeed");
        let m = e.membership("temp", "cold").expect("test: should succeed");
        assert!((m - 1.0).abs() < 1e-10);
    }
    #[test]
    fn membership_unknown_var() {
        let e = build_temp_engine(DefuzzMethod::Centroid);
        assert!(matches!(
            e.membership("pressure", "low"),
            Err(FuzzyError::VariableNotFound(_))
        ));
    }
    #[test]
    fn membership_unknown_set() {
        let e = build_temp_engine(DefuzzMethod::Centroid);
        assert!(matches!(
            e.membership("temp", "freezing"),
            Err(FuzzyError::SetNotFound { .. })
        ));
    }
    #[test]
    fn add_rule_valid() {
        let mut e = build_temp_engine(DefuzzMethod::Centroid);
        let r = FuzzyRule::new(
            "r3",
            FuzzyExpr::Is {
                var: "temp".into(),
                set: "warm".into(),
            },
            FuzzyExpr::Is {
                var: "fan".into(),
                set: "slow".into(),
            },
            0.5,
        );
        assert!(e.add_rule(r).is_ok());
    }
    #[test]
    fn add_rule_invalid_var() {
        let mut e = build_temp_engine(DefuzzMethod::Centroid);
        let r = FuzzyRule::new(
            "bad",
            FuzzyExpr::Is {
                var: "humidity".into(),
                set: "warm".into(),
            },
            FuzzyExpr::Is {
                var: "fan".into(),
                set: "slow".into(),
            },
            1.0,
        );
        assert!(matches!(
            e.add_rule(r),
            Err(FuzzyError::VariableNotFound(_))
        ));
    }
    #[test]
    fn add_rule_invalid_set() {
        let mut e = build_temp_engine(DefuzzMethod::Centroid);
        let r = FuzzyRule::new(
            "bad",
            FuzzyExpr::Is {
                var: "temp".into(),
                set: "freezing".into(),
            },
            FuzzyExpr::Is {
                var: "fan".into(),
                set: "slow".into(),
            },
            1.0,
        );
        assert!(matches!(e.add_rule(r), Err(FuzzyError::SetNotFound { .. })));
    }
    #[test]
    fn error_no_rules_activated() {
        let mut e = build_temp_engine(DefuzzMethod::Centroid);
        e.set_input("temp", -100.0).expect("test: should succeed");
        let res = e.infer_single("fan");
        assert!(matches!(res, Err(FuzzyError::NoRulesActivated)));
    }
    #[test]
    fn error_output_var_not_found() {
        let mut e = build_temp_engine(DefuzzMethod::Centroid);
        e.set_input("temp", 10.0).expect("test: should succeed");
        let res = e.infer_single("pressure");
        assert!(matches!(res, Err(FuzzyError::VariableNotFound(_))));
    }
    #[test]
    fn error_config_zero_resolution() {
        let mut temp = FuzzyVariable::new("t");
        temp.sets.push(FuzzySet::new(
            "low",
            MembershipFunction::Triangle {
                a: 0.0,
                b: 5.0,
                c: 10.0,
            },
            0.0,
            10.0,
        ));
        let mut out = FuzzyVariable::new("out");
        out.sets.push(FuzzySet::new(
            "yes",
            MembershipFunction::Triangle {
                a: 0.0,
                b: 5.0,
                c: 10.0,
            },
            0.0,
            10.0,
        ));
        let mut cfg = EngineConfig::new(vec![temp], vec![out], vec![], DefuzzMethod::Centroid);
        cfg.resolution = 0;
        let res = FuzzyLogicEngine::new(cfg);
        assert!(matches!(res, Err(FuzzyError::ConfigurationError(_))));
    }
    #[test]
    fn error_rule_ref_missing_var() {
        let mut temp = FuzzyVariable::new("t");
        temp.sets.push(FuzzySet::new(
            "low",
            MembershipFunction::Triangle {
                a: 0.0,
                b: 5.0,
                c: 10.0,
            },
            0.0,
            10.0,
        ));
        let mut out = FuzzyVariable::new("out");
        out.sets.push(FuzzySet::new(
            "yes",
            MembershipFunction::Triangle {
                a: 0.0,
                b: 5.0,
                c: 10.0,
            },
            0.0,
            10.0,
        ));
        let bad_rule = FuzzyRule::new(
            "bad",
            FuzzyExpr::Is {
                var: "nonexistent".into(),
                set: "low".into(),
            },
            FuzzyExpr::Is {
                var: "out".into(),
                set: "yes".into(),
            },
            1.0,
        );
        let cfg = EngineConfig::new(
            vec![temp],
            vec![out],
            vec![bad_rule],
            DefuzzMethod::Centroid,
        );
        assert!(matches!(
            FuzzyLogicEngine::new(cfg),
            Err(FuzzyError::VariableNotFound(_))
        ));
    }
    #[test]
    fn stats_defuzz_calls_incremented() {
        let mut e = build_temp_engine(DefuzzMethod::Centroid);
        e.set_input("temp", 10.0).expect("test: should succeed");
        e.infer_single("fan").expect("test: should succeed");
        assert_eq!(e.stats().defuzz_calls, 1);
    }
    #[test]
    fn stats_rules_evaluated_positive() {
        let mut e = build_temp_engine(DefuzzMethod::Centroid);
        e.set_input("temp", 10.0).expect("test: should succeed");
        e.infer_single("fan").expect("test: should succeed");
        assert!(e.stats().rules_evaluated > 0);
    }
    #[test]
    fn stats_avg_activation_bounded() {
        let mut e = build_temp_engine(DefuzzMethod::Centroid);
        e.set_input("temp", 10.0).expect("test: should succeed");
        e.infer_single("fan").expect("test: should succeed");
        let avg = e.stats().avg_activation;
        assert!(avg >= 0.0);
    }
    #[test]
    fn weighted_rule_reduces_activation() {
        let mut e = build_temp_engine(DefuzzMethod::Centroid);
        e.set_input("temp", 10.0).expect("test: should succeed");
        let e_half = build_temp_engine(DefuzzMethod::Centroid);
        let mut temp = FuzzyVariable::new("temp");
        temp.sets.push(FuzzySet::new(
            "cold",
            MembershipFunction::Triangle {
                a: 0.0,
                b: 10.0,
                c: 20.0,
            },
            0.0,
            100.0,
        ));
        let mut fan = FuzzyVariable::new("fan");
        fan.sets.push(FuzzySet::new(
            "slow",
            MembershipFunction::Triangle {
                a: 0.0,
                b: 25.0,
                c: 50.0,
            },
            0.0,
            100.0,
        ));
        let r = FuzzyRule::new(
            "rw",
            FuzzyExpr::Is {
                var: "temp".into(),
                set: "cold".into(),
            },
            FuzzyExpr::Is {
                var: "fan".into(),
                set: "slow".into(),
            },
            0.5,
        );
        let cfg = EngineConfig::new(vec![temp], vec![fan], vec![r], DefuzzMethod::Centroid);
        let mut ew = FuzzyLogicEngine::new(cfg).expect("test: should succeed");
        ew.set_input("temp", 10.0).expect("test: should succeed");
        let rule = ew.rules.first().cloned().expect("test: should succeed");
        let act = ew.evaluate_rule(&rule).expect("test: should succeed");
        assert!((act - 0.5).abs() < 1e-10, "expected 0.5, got {act}");
        let _ = e_half;
    }
    #[test]
    fn random_inputs_produce_valid_outputs() {
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let mut e = build_temp_engine(DefuzzMethod::Centroid);
        for _ in 0..20 {
            let t = xorshift64(&mut state) * 70.0;
            e.set_input("temp", t).expect("test: should succeed");
            let result = e.infer_single("fan");
            if let Ok(r) = result {
                assert!(r.crisp_value >= 0.0 && r.crisp_value <= 100.0);
            }
        }
    }
    #[test]
    fn fuzzy_set_degree_delegates_to_mf() {
        let fs = FuzzySet::new(
            "mid",
            MembershipFunction::Triangle {
                a: 0.0,
                b: 5.0,
                c: 10.0,
            },
            0.0,
            10.0,
        );
        assert!((fs.degree(5.0) - 1.0).abs() < 1e-10);
        assert_eq!(fs.degree(0.0), 0.0);
    }
    #[test]
    fn fuzzy_variable_get_set_found() {
        let mut v = FuzzyVariable::new("x");
        v.sets.push(FuzzySet::new(
            "alpha",
            MembershipFunction::Singleton(1.0),
            0.0,
            2.0,
        ));
        assert!(v.get_set("alpha").is_some());
    }
    #[test]
    fn fuzzy_variable_get_set_not_found() {
        let v = FuzzyVariable::new("x");
        assert!(v.get_set("beta").is_none());
    }
    #[test]
    fn defuzz_method_default_is_centroid() {
        assert_eq!(DefuzzMethod::default(), DefuzzMethod::Centroid);
    }
    #[test]
    fn engine_config_default_resolution() {
        let cfg = EngineConfig::new(vec![], vec![], vec![], DefuzzMethod::Centroid);
        assert_eq!(cfg.resolution, 100);
    }
}
