    

    // ── helpers ───────────────────────────────────────────────────────────────

    fn bool_var(id: &str) -> RandomVariable {
        RandomVariable {
            id: id.to_string(),
            states: vec!["T".to_string(), "F".to_string()],
            cardinality: 2,
        }
    }

    /// Build a simple Rain → WetGrass network.
    ///
    /// P(Rain=T) = 0.2
    /// P(Wet=T | Rain=T) = 0.9, P(Wet=T | Rain=F) = 0.2
    fn rain_wet_network() -> BayesianNetwork {
        let mut variables = HashMap::new();
        variables.insert("Rain".to_string(), bool_var("Rain"));
        variables.insert("Wet".to_string(), bool_var("Wet"));

        let f_rain = Factor {
            id: "f_rain".into(),
            variables: vec!["Rain".into()],
            values: vec![0.2, 0.8],
            shape: vec![2],
        };
        let cpt_rain = ConditionalProbabilityTable {
            variable: "Rain".into(),
            parents: vec![],
            factor: f_rain,
        };

        // shape: [Rain=2, Wet=2]  row-major: Rain=T,Wet=T / Rain=T,Wet=F / Rain=F,Wet=T / Rain=F,Wet=F
        let f_wet = Factor {
            id: "f_wet".into(),
            variables: vec!["Rain".into(), "Wet".into()],
            values: vec![0.9, 0.1, 0.2, 0.8],
            shape: vec![2, 2],
        };
        let cpt_wet = ConditionalProbabilityTable {
            variable: "Wet".into(),
            parents: vec!["Rain".into()],
            factor: f_wet,
        };

        let mut adjacency = HashMap::new();
        adjacency.insert("Wet".to_string(), vec!["Rain".to_string()]);

        BayesianNetwork {
            variables,
            cpts: vec![cpt_rain, cpt_wet],
            adjacency,
        }
    }

    /// Build a three-variable chain: A → B → C
    fn chain_network() -> BayesianNetwork {
        let mut variables = HashMap::new();
        variables.insert("A".into(), bool_var("A"));
        variables.insert("B".into(), bool_var("B"));
        variables.insert("C".into(), bool_var("C"));

        let f_a = Factor {
            id: "f_a".into(),
            variables: vec!["A".into()],
            values: vec![0.4, 0.6],
            shape: vec![2],
        };
        let cpt_a = ConditionalProbabilityTable {
            variable: "A".into(),
            parents: vec![],
            factor: f_a,
        };

        let f_b = Factor {
            id: "f_b".into(),
            variables: vec!["A".into(), "B".into()],
            values: vec![0.7, 0.3, 0.1, 0.9],
            shape: vec![2, 2],
        };
        let cpt_b = ConditionalProbabilityTable {
            variable: "B".into(),
            parents: vec!["A".into()],
            factor: f_b,
        };

        let f_c = Factor {
            id: "f_c".into(),
            variables: vec!["B".into(), "C".into()],
            values: vec![0.8, 0.2, 0.3, 0.7],
            shape: vec![2, 2],
        };
        let cpt_c = ConditionalProbabilityTable {
            variable: "C".into(),
            parents: vec!["B".into()],
            factor: f_c,
        };

        let mut adjacency = HashMap::new();
        adjacency.insert("B".into(), vec!["A".into()]);
        adjacency.insert("C".into(), vec!["B".into()]);

        BayesianNetwork {
            variables,
            cpts: vec![cpt_a, cpt_b, cpt_c],
            adjacency,
        }
    }

    // ── Factor tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_factor_normalize_basic() {
        let mut f = Factor {
            id: "f".into(),
            variables: vec!["X".into()],
            values: vec![2.0, 6.0],
            shape: vec![2],
        };
        f.normalize();
        assert!((f.values[0] - 0.25).abs() < 1e-10);
        assert!((f.values[1] - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_factor_normalize_zero_sum() {
        let mut f = Factor {
            id: "f".into(),
            variables: vec!["X".into()],
            values: vec![0.0, 0.0],
            shape: vec![2],
        };
        f.normalize();
        // Should remain zero (no division by zero).
        assert_eq!(f.values, vec![0.0, 0.0]);
    }

    #[test]
    fn test_factor_product_same_scope() {
        let f1 = Factor {
            id: "f1".into(),
            variables: vec!["X".into()],
            values: vec![0.3, 0.7],
            shape: vec![2],
        };
        let f2 = Factor {
            id: "f2".into(),
            variables: vec!["X".into()],
            values: vec![0.5, 0.5],
            shape: vec![2],
        };
        let prod = f1.product(&f2);
        assert_eq!(prod.variables, vec!["X"]);
        assert!((prod.values[0] - 0.15).abs() < 1e-10);
        assert!((prod.values[1] - 0.35).abs() < 1e-10);
    }

    #[test]
    fn test_factor_product_disjoint_scope() {
        let f1 = Factor {
            id: "f1".into(),
            variables: vec!["X".into()],
            values: vec![0.4, 0.6],
            shape: vec![2],
        };
        let f2 = Factor {
            id: "f2".into(),
            variables: vec!["Y".into()],
            values: vec![0.3, 0.7],
            shape: vec![2],
        };
        let prod = f1.product(&f2);
        assert_eq!(prod.variables.len(), 2);
        assert_eq!(prod.values.len(), 4);
        // [X=0,Y=0]=0.12, [X=0,Y=1]=0.28, [X=1,Y=0]=0.18, [X=1,Y=1]=0.42
        assert!((prod.values[0] - 0.12).abs() < 1e-10);
        assert!((prod.values[1] - 0.28).abs() < 1e-10);
        assert!((prod.values[2] - 0.18).abs() < 1e-10);
        assert!((prod.values[3] - 0.42).abs() < 1e-10);
    }

    #[test]
    fn test_factor_product_overlapping_scope() {
        // Joint factor: f(A,B) * f(B,C)
        let f1 = Factor {
            id: "f1".into(),
            variables: vec!["A".into(), "B".into()],
            values: vec![0.5, 0.5, 0.5, 0.5],
            shape: vec![2, 2],
        };
        let f2 = Factor {
            id: "f2".into(),
            variables: vec!["B".into(), "C".into()],
            values: vec![0.8, 0.2, 0.3, 0.7],
            shape: vec![2, 2],
        };
        let prod = f1.product(&f2);
        assert_eq!(prod.variables.len(), 3);
        assert_eq!(prod.values.len(), 8);
    }

    #[test]
    fn test_factor_marginalize_single_var() {
        // f(X, Y) — marginalize out Y to get f(X)
        let f = Factor {
            id: "f".into(),
            variables: vec!["X".into(), "Y".into()],
            values: vec![0.2, 0.3, 0.1, 0.4],
            shape: vec![2, 2],
        };
        let dummy = HashMap::new();
        let marg = f.marginalize("Y", &dummy);
        assert_eq!(marg.variables, vec!["X"]);
        assert!((marg.values[0] - 0.5).abs() < 1e-10); // 0.2 + 0.3
        assert!((marg.values[1] - 0.5).abs() < 1e-10); // 0.1 + 0.4
    }

    #[test]
    fn test_factor_marginalize_absent_var() {
        let f = Factor {
            id: "f".into(),
            variables: vec!["X".into()],
            values: vec![0.4, 0.6],
            shape: vec![2],
        };
        let dummy = HashMap::new();
        let marg = f.marginalize("Z", &dummy);
        assert_eq!(marg.variables, vec!["X"]);
        assert_eq!(marg.values, vec![0.4, 0.6]);
    }

    #[test]
    fn test_factor_marginalize_all_vars() {
        let f = Factor {
            id: "f".into(),
            variables: vec!["X".into()],
            values: vec![0.4, 0.6],
            shape: vec![2],
        };
        let dummy = HashMap::new();
        let marg = f.marginalize("X", &dummy);
        assert!(marg.variables.is_empty());
        assert!((marg.values[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_factor_reduce_basic() {
        let f = Factor {
            id: "f".into(),
            variables: vec!["X".into(), "Y".into()],
            values: vec![0.1, 0.4, 0.2, 0.3],
            shape: vec![2, 2],
        };
        let dummy = HashMap::new();
        // Reduce X=0: keep rows where X=0 → values [0.1, 0.4]
        let red = f.reduce("X", 0, &dummy);
        assert_eq!(red.variables, vec!["Y"]);
        assert!((red.values[0] - 0.1).abs() < 1e-10);
        assert!((red.values[1] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_factor_reduce_absent_var() {
        let f = Factor {
            id: "f".into(),
            variables: vec!["X".into()],
            values: vec![0.4, 0.6],
            shape: vec![2],
        };
        let dummy = HashMap::new();
        let red = f.reduce("Z", 0, &dummy);
        assert_eq!(red.values, vec![0.4, 0.6]);
    }

    #[test]
    fn test_factor_reduce_second_state() {
        let f = Factor {
            id: "f".into(),
            variables: vec!["Rain".into(), "Wet".into()],
            values: vec![0.9, 0.1, 0.2, 0.8],
            shape: vec![2, 2],
        };
        let dummy = HashMap::new();
        // Rain=F (state 1): rows 2 & 3 → [0.2, 0.8]
        let red = f.reduce("Rain", 1, &dummy);
        assert_eq!(red.variables, vec!["Wet"]);
        assert!((red.values[0] - 0.2).abs() < 1e-10);
        assert!((red.values[1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_factor_entropy_uniform() {
        let mut f = Factor {
            id: "f".into(),
            variables: vec!["X".into()],
            values: vec![0.5, 0.5],
            shape: vec![2],
        };
        f.normalize();
        // H({0.5,0.5}) = 1 bit
        assert!((f.entropy() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_factor_entropy_certain() {
        let f = Factor {
            id: "f".into(),
            variables: vec!["X".into()],
            values: vec![1.0, 0.0],
            shape: vec![2],
        };
        assert!(f.entropy().abs() < 1e-10);
    }

    #[test]
    fn test_factor_contains_variable() {
        let f = Factor {
            id: "f".into(),
            variables: vec!["X".into(), "Y".into()],
            values: vec![0.5; 4],
            shape: vec![2, 2],
        };
        assert!(f.contains_variable("X"));
        assert!(f.contains_variable("Y"));
        assert!(!f.contains_variable("Z"));
    }

    // ── multi-dim index helpers ───────────────────────────────────────────────

    #[test]
    fn test_flat_multi_index_roundtrip() {
        let shape = vec![3usize, 4, 2];
        for flat in 0..24 {
            let multi = Factor::multi_index(flat, &shape);
            let back = Factor::flat_index(&multi, &shape);
            assert_eq!(flat, back, "roundtrip failed for flat={flat}");
        }
    }

    // ── BayesianNetworkInference construction ─────────────────────────────────

    #[test]
    fn test_engine_construction_valid() {
        let net = rain_wet_network();
        let engine = BayesianNetworkInference::new(net, BniConfig::default());
        assert!(engine.is_ok());
    }

    #[test]
    fn test_engine_rejects_cyclic_network() {
        let mut variables = HashMap::new();
        variables.insert("A".into(), bool_var("A"));
        variables.insert("B".into(), bool_var("B"));

        let f_a = Factor {
            id: "fa".into(),
            variables: vec!["B".into(), "A".into()],
            values: vec![0.5; 4],
            shape: vec![2, 2],
        };
        let cpt_a = ConditionalProbabilityTable {
            variable: "A".into(),
            parents: vec!["B".into()],
            factor: f_a,
        };

        let f_b = Factor {
            id: "fb".into(),
            variables: vec!["A".into(), "B".into()],
            values: vec![0.5; 4],
            shape: vec![2, 2],
        };
        let cpt_b = ConditionalProbabilityTable {
            variable: "B".into(),
            parents: vec!["A".into()],
            factor: f_b,
        };

        let mut adjacency = HashMap::new();
        adjacency.insert("A".into(), vec!["B".into()]);
        adjacency.insert("B".into(), vec!["A".into()]);

        let net = BayesianNetwork { variables, cpts: vec![cpt_a, cpt_b], adjacency };
        let result = BayesianNetworkInference::new(net, BniConfig::default());
        assert!(matches!(result, Err(BniError::CyclicNetwork(_))));
    }

    #[test]
    fn test_engine_rejects_invalid_cpt_shape() {
        let mut variables = HashMap::new();
        variables.insert("Rain".into(), bool_var("Rain"));

        // Wrong number of values.
        let f = Factor {
            id: "f".into(),
            variables: vec!["Rain".into()],
            values: vec![0.3, 0.3, 0.4], // length 3, but Rain has cardinality 2
            shape: vec![2],
        };
        let cpt = ConditionalProbabilityTable {
            variable: "Rain".into(),
            parents: vec![],
            factor: f,
        };
        let net = BayesianNetwork {
            variables,
            cpts: vec![cpt],
            adjacency: HashMap::new(),
        };
        let result = BayesianNetworkInference::new(net, BniConfig::default());
        assert!(matches!(result, Err(BniError::InvalidCPT { .. })));
    }

    // ── Variable Elimination ──────────────────────────────────────────────────

    #[test]
    fn test_ve_prior_rain() {
        let net = rain_wet_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");

        let result = engine.prior_marginal("Rain").expect("test: prior marginal query should succeed");
        assert_eq!(result.variable, "Rain");
        assert!((result.distribution[0].1 - 0.2).abs() < 1e-9, "P(Rain=T)");
        assert!((result.distribution[1].1 - 0.8).abs() < 1e-9, "P(Rain=F)");
    }

    #[test]
    fn test_ve_prior_wet_marginal() {
        // P(Wet=T) = P(Wet=T|Rain=T)*P(Rain=T) + P(Wet=T|Rain=F)*P(Rain=F)
        //          = 0.9*0.2 + 0.2*0.8 = 0.18 + 0.16 = 0.34
        let net = rain_wet_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");

        let result = engine.prior_marginal("Wet").expect("test: prior marginal query should succeed");
        assert!((result.distribution[0].1 - 0.34).abs() < 1e-9, "P(Wet=T) expected 0.34, got {}", result.distribution[0].1);
    }

    #[test]
    fn test_ve_with_evidence_rain_true() {
        // P(Wet | Rain=T): should be [0.9, 0.1]
        let net = rain_wet_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");

        let q = InferenceQuery {
            query_variables: vec!["Wet".into()],
            evidence: vec![Evidence { variable: "Rain".into(), observed_state: "T".into() }],
            algorithm: InferenceAlgorithm::VariableElimination,
        };
        let results = engine.query(&q).expect("test: BNI query should succeed");
        let dist = &results[0].distribution;
        assert!((dist[0].1 - 0.9).abs() < 1e-9, "P(Wet=T|Rain=T)");
        assert!((dist[1].1 - 0.1).abs() < 1e-9, "P(Wet=F|Rain=T)");
    }

    #[test]
    fn test_ve_with_evidence_wet_true() {
        // P(Rain=T | Wet=T) via Bayes:
        //   numerator   = P(Wet=T|Rain=T)*P(Rain=T) = 0.9*0.2 = 0.18
        //   denominator = 0.34
        //   posterior   = 0.18/0.34 ≈ 0.5294
        let net = rain_wet_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");

        let q = InferenceQuery {
            query_variables: vec!["Rain".into()],
            evidence: vec![Evidence { variable: "Wet".into(), observed_state: "T".into() }],
            algorithm: InferenceAlgorithm::VariableElimination,
        };
        let results = engine.query(&q).expect("test: BNI query should succeed");
        let p_rain_t = results[0].distribution[0].1;
        let expected = 0.18 / 0.34;
        assert!(
            (p_rain_t - expected).abs() < 1e-9,
            "P(Rain=T|Wet=T) expected {expected:.6}, got {p_rain_t:.6}"
        );
    }

    #[test]
    fn test_ve_chain_prior() {
        // Chain A → B → C; check P(A) = [0.4, 0.6]
        let net = chain_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        let result = engine.prior_marginal("A").expect("test: prior marginal query should succeed");
        assert!((result.distribution[0].1 - 0.4).abs() < 1e-9);
        assert!((result.distribution[1].1 - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_ve_chain_b_marginal() {
        // P(B=T) = P(B=T|A=T)*P(A=T) + P(B=T|A=F)*P(A=F)
        //        = 0.7*0.4 + 0.1*0.6 = 0.28 + 0.06 = 0.34
        let net = chain_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        let result = engine.prior_marginal("B").expect("test: prior marginal query should succeed");
        assert!((result.distribution[0].1 - 0.34).abs() < 1e-9, "P(B=T) {}", result.distribution[0].1);
    }

    #[test]
    fn test_ve_chain_with_evidence() {
        // P(C | A=T) — B is eliminated
        let net = chain_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        let q = InferenceQuery {
            query_variables: vec!["C".into()],
            evidence: vec![Evidence { variable: "A".into(), observed_state: "T".into() }],
            algorithm: InferenceAlgorithm::VariableElimination,
        };
        let results = engine.query(&q).expect("test: BNI query should succeed");
        // P(C=T|A=T) = P(C=T|B=T)*P(B=T|A=T) + P(C=T|B=F)*P(B=F|A=T)
        //            = 0.8*0.7 + 0.3*0.3 = 0.56 + 0.09 = 0.65
        assert!((results[0].distribution[0].1 - 0.65).abs() < 1e-9, "P(C=T|A=T) {}", results[0].distribution[0].1);
    }

    #[test]
    fn test_ve_multi_query() {
        let net = chain_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        let q = InferenceQuery {
            query_variables: vec!["A".into(), "C".into()],
            evidence: vec![],
            algorithm: InferenceAlgorithm::VariableElimination,
        };
        let results = engine.query(&q).expect("test: BNI query should succeed");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ve_min_degree_ordering() {
        let net = chain_network();
        let config = BniConfig { elimination_ordering: EliminationOrder::MinDegree, ..Default::default() };
        let mut engine = BayesianNetworkInference::new(net, config).expect("test: BNI engine construction should succeed");
        let result = engine.prior_marginal("C").expect("test: prior marginal query should succeed");
        assert!(result.distribution[0].1 > 0.0);
        assert!(result.distribution[1].1 > 0.0);
    }

    #[test]
    fn test_ve_sequential_ordering() {
        let net = chain_network();
        let config = BniConfig { elimination_ordering: EliminationOrder::Sequential, ..Default::default() };
        let mut engine = BayesianNetworkInference::new(net, config).expect("test: BNI engine construction should succeed");
        let result = engine.prior_marginal("C").expect("test: prior marginal query should succeed");
        assert!(result.distribution[0].1 >= 0.0);
    }

    // ── Belief Propagation ────────────────────────────────────────────────────

    #[test]
    fn test_bp_prior_rain() {
        let net = rain_wet_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        let q = InferenceQuery {
            query_variables: vec!["Rain".into()],
            evidence: vec![],
            algorithm: InferenceAlgorithm::BeliefPropagation,
        };
        let results = engine.query(&q).expect("test: BNI query should succeed");
        assert!((results[0].distribution[0].1 - 0.2).abs() < 1e-6, "BP P(Rain=T)={}", results[0].distribution[0].1);
    }

    #[test]
    fn test_bp_with_evidence() {
        let net = rain_wet_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        let q = InferenceQuery {
            query_variables: vec!["Wet".into()],
            evidence: vec![Evidence { variable: "Rain".into(), observed_state: "T".into() }],
            algorithm: InferenceAlgorithm::BeliefPropagation,
        };
        let results = engine.query(&q).expect("test: BNI query should succeed");
        // With Rain=T, P(Wet=T|Rain=T)=0.9
        assert!((results[0].distribution[0].1 - 0.9).abs() < 1e-6, "BP P(Wet=T|Rain=T)={}", results[0].distribution[0].1);
    }

    // ── Sampling ──────────────────────────────────────────────────────────────

    #[test]
    fn test_sampling_prior_rain_approximate() {
        // With enough samples, should be close to 0.2.
        let net = rain_wet_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        let q = InferenceQuery {
            query_variables: vec!["Rain".into()],
            evidence: vec![],
            algorithm: InferenceAlgorithm::Sampling { n_samples: 50_000, seed: 42 },
        };
        let results = engine.query(&q).expect("test: BNI query should succeed");
        let p = results[0].distribution[0].1;
        assert!((p - 0.2).abs() < 0.03, "P(Rain=T)≈{p:.4}");
    }

    #[test]
    fn test_sampling_with_evidence() {
        let net = rain_wet_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        let q = InferenceQuery {
            query_variables: vec!["Wet".into()],
            evidence: vec![Evidence { variable: "Rain".into(), observed_state: "T".into() }],
            algorithm: InferenceAlgorithm::Sampling { n_samples: 50_000, seed: 123 },
        };
        let results = engine.query(&q).expect("test: BNI query should succeed");
        let p = results[0].distribution[0].1;
        // P(Wet=T | Rain=T) = 0.9 (exact), sampling should be within 5%
        assert!((p - 0.9).abs() < 0.05, "Sampled P(Wet=T|Rain=T)≈{p:.4}");
    }

    #[test]
    fn test_sampling_seed_reproducibility() {
        let net = rain_wet_network();
        let mut e1 = BayesianNetworkInference::new(net.clone(), BniConfig::default()).expect("test: should succeed");
        let mut e2 = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        let q = InferenceQuery {
            query_variables: vec!["Rain".into()],
            evidence: vec![],
            algorithm: InferenceAlgorithm::Sampling { n_samples: 1000, seed: 99 },
        };
        let r1 = e1.query(&q).expect("test: BNI query should succeed");
        let r2 = e2.query(&q).expect("test: BNI query should succeed");
        assert!((r1[0].distribution[0].1 - r2[0].distribution[0].1).abs() < 1e-12);
    }

    #[test]
    fn test_sampling_zero_seed_uses_default() {
        let net = rain_wet_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        let q = InferenceQuery {
            query_variables: vec!["Rain".into()],
            evidence: vec![],
            algorithm: InferenceAlgorithm::Sampling { n_samples: 1000, seed: 0 },
        };
        let results = engine.query(&q);
        assert!(results.is_ok());
    }

    // ── QueryResult ───────────────────────────────────────────────────────────

    #[test]
    fn test_query_result_most_likely_state() {
        let net = rain_wet_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        let result = engine.prior_marginal("Rain").expect("test: prior marginal query should succeed");
        // P(Rain=F)=0.8 > P(Rain=T)=0.2
        assert_eq!(result.most_likely_state, "F");
    }

    #[test]
    fn test_query_result_entropy_range() {
        let net = rain_wet_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        let result = engine.prior_marginal("Rain").expect("test: prior marginal query should succeed");
        assert!(result.marginal_entropy >= 0.0);
        assert!(result.marginal_entropy <= 1.0 + 1e-9); // max 1 bit for binary var
    }

    // ── Evidence validation ───────────────────────────────────────────────────

    #[test]
    fn test_evidence_unknown_variable() {
        let net = rain_wet_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        let q = InferenceQuery {
            query_variables: vec!["Rain".into()],
            evidence: vec![Evidence { variable: "Unknown".into(), observed_state: "T".into() }],
            algorithm: InferenceAlgorithm::VariableElimination,
        };
        assert!(matches!(engine.query(&q), Err(BniError::VariableNotFound(_))));
    }

    #[test]
    fn test_evidence_unknown_state() {
        let net = rain_wet_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        let q = InferenceQuery {
            query_variables: vec!["Wet".into()],
            evidence: vec![Evidence { variable: "Rain".into(), observed_state: "MAYBE".into() }],
            algorithm: InferenceAlgorithm::VariableElimination,
        };
        assert!(matches!(engine.query(&q), Err(BniError::InvalidCPT { .. })));
    }

    #[test]
    fn test_evidence_conflict() {
        let net = rain_wet_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        let q = InferenceQuery {
            query_variables: vec!["Wet".into()],
            evidence: vec![
                Evidence { variable: "Rain".into(), observed_state: "T".into() },
                Evidence { variable: "Rain".into(), observed_state: "F".into() },
            ],
            algorithm: InferenceAlgorithm::VariableElimination,
        };
        assert!(matches!(engine.query(&q), Err(BniError::EvidenceConflict(_))));
    }

    #[test]
    fn test_evidence_duplicate_consistent() {
        // Same evidence twice — OK (no conflict).
        let net = rain_wet_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        let q = InferenceQuery {
            query_variables: vec!["Wet".into()],
            evidence: vec![
                Evidence { variable: "Rain".into(), observed_state: "T".into() },
                Evidence { variable: "Rain".into(), observed_state: "T".into() },
            ],
            algorithm: InferenceAlgorithm::VariableElimination,
        };
        assert!(engine.query(&q).is_ok());
    }

    // ── d-separation ──────────────────────────────────────────────────────────

    #[test]
    fn test_d_sep_chain_head_to_tail() {
        // A → B → C: A ⊥ C | B
        let net = chain_network();
        let engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        assert!(engine.d_separated("A", "C", &["B".to_string()]).expect("test: should succeed"));
    }

    #[test]
    fn test_d_sep_chain_not_separated() {
        // A → B → C: A not ⊥ C | {}
        let net = chain_network();
        let engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        assert!(!engine.d_separated("A", "C", &[]).expect("test: d-separation check should succeed"));
    }

    #[test]
    fn test_d_sep_unknown_variable() {
        let net = chain_network();
        let engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        let result = engine.d_separated("X_unknown", "A", &[]);
        assert!(matches!(result, Err(BniError::VariableNotFound(_))));
    }

    // ── add_cpt ───────────────────────────────────────────────────────────────

    #[test]
    fn test_add_cpt_replaces_existing() {
        let net = rain_wet_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");

        // Replace P(Rain) with uniform.
        let new_factor = Factor {
            id: "f_rain2".into(),
            variables: vec!["Rain".into()],
            values: vec![0.5, 0.5],
            shape: vec![2],
        };
        let new_cpt = ConditionalProbabilityTable {
            variable: "Rain".into(),
            parents: vec![],
            factor: new_factor,
        };
        assert!(engine.add_cpt(new_cpt).is_ok());

        let result = engine.prior_marginal("Rain").expect("test: prior marginal query should succeed");
        assert!((result.distribution[0].1 - 0.5).abs() < 1e-9);
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_increment() {
        let net = rain_wet_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        assert_eq!(engine.stats().queries_answered, 0);
        let _ = engine.prior_marginal("Rain").expect("test: prior marginal query should succeed");
        // prior_marginal calls query internally → 1 query answered.
        assert_eq!(engine.stats().queries_answered, 1);
        // A second direct query increments to 2.
        let q = InferenceQuery {
            query_variables: vec!["Rain".into()],
            evidence: vec![],
            algorithm: InferenceAlgorithm::VariableElimination,
        };
        let _ = engine.query(&q).expect("test: BNI query should succeed");
        assert_eq!(engine.stats().queries_answered, 2);
    }

    #[test]
    fn test_stats_avg_factors() {
        let net = chain_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        let _ = engine.prior_marginal("C").expect("test: prior marginal query should succeed");
        assert!(engine.stats().avg_factors_eliminated >= 0.0);
    }

    // ── xorshift64 ────────────────────────────────────────────────────────────

    #[test]
    fn test_bni_xorshift64_non_zero() {
        let mut state = 1u64;
        for _ in 0..100 {
            let v = bni_xorshift64(&mut state);
            assert!(v > 0, "xorshift64 should never produce 0 for non-zero seed");
        }
    }

    #[test]
    fn test_bni_xorshift64_deterministic() {
        let mut s1 = 42u64;
        let mut s2 = 42u64;
        let v1 = bni_xorshift64(&mut s1);
        let v2 = bni_xorshift64(&mut s2);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_sample_categorical_sums_to_one() {
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let mut state = 7u64;
        let mut counts = [0usize; 4];
        for _ in 0..10_000 {
            let idx = sample_categorical(&probs, &mut state);
            counts[idx] += 1;
        }
        // Each bucket should have been hit.
        for (i, &c) in counts.iter().enumerate() {
            assert!(c > 0, "category {i} was never sampled");
        }
    }

    // ── error-path tests ──────────────────────────────────────────────────────

    #[test]
    fn test_query_unknown_variable() {
        let net = rain_wet_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        let q = InferenceQuery {
            query_variables: vec!["Unicorn".into()],
            evidence: vec![],
            algorithm: InferenceAlgorithm::VariableElimination,
        };
        assert!(matches!(engine.query(&q), Err(BniError::VariableNotFound(_))));
    }

    #[test]
    fn test_prior_marginal_unknown_variable() {
        let net = rain_wet_network();
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        assert!(matches!(
            engine.prior_marginal("Unicorn"),
            Err(BniError::VariableNotFound(_))
        ));
    }

    #[test]
    fn test_d_sep_unknown_z_variable() {
        let net = chain_network();
        let engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");
        let result = engine.d_separated("A", "C", &["Z_unknown".to_string()]);
        assert!(matches!(result, Err(BniError::VariableNotFound(_))));
    }

    #[test]
    fn test_engine_max_variables_exceeded() {
        let net = rain_wet_network();
        let config = BniConfig {
            max_variables: 1,
            ..Default::default()
        };
        assert!(matches!(
            BayesianNetworkInference::new(net, config),
            Err(BniError::InferenceError(_))
        ));
    }

    // ── BniConfig defaults ────────────────────────────────────────────────────

    #[test]
    fn test_bni_config_defaults() {
        let cfg = BniConfig::default();
        assert_eq!(cfg.max_variables, 256);
        assert_eq!(cfg.max_states_per_variable, 1024);
        assert_eq!(cfg.elimination_ordering, EliminationOrder::MinFill);
    }

    // ── three-variable V-structure ────────────────────────────────────────────

    #[test]
    fn test_ve_v_structure() {
        // A → C ← B (collider C).
        let mut variables = HashMap::new();
        variables.insert("A".into(), bool_var("A"));
        variables.insert("B".into(), bool_var("B"));
        variables.insert("C".into(), bool_var("C"));

        let f_a = Factor { id: "fa".into(), variables: vec!["A".into()], values: vec![0.5, 0.5], shape: vec![2] };
        let cpt_a = ConditionalProbabilityTable { variable: "A".into(), parents: vec![], factor: f_a };

        let f_b = Factor { id: "fb".into(), variables: vec!["B".into()], values: vec![0.5, 0.5], shape: vec![2] };
        let cpt_b = ConditionalProbabilityTable { variable: "B".into(), parents: vec![], factor: f_b };

        // P(C|A,B): A=0,B=0→high; A=1,B=1→low; shape [A=2, B=2, C=2]
        let f_c = Factor {
            id: "fc".into(),
            variables: vec!["A".into(), "B".into(), "C".into()],
            values: vec![0.9, 0.1, 0.6, 0.4, 0.6, 0.4, 0.1, 0.9],
            shape: vec![2, 2, 2],
        };
        let cpt_c = ConditionalProbabilityTable {
            variable: "C".into(),
            parents: vec!["A".into(), "B".into()],
            factor: f_c,
        };

        let mut adjacency = HashMap::new();
        adjacency.insert("C".into(), vec!["A".into(), "B".into()]);

        let net = BayesianNetwork { variables, cpts: vec![cpt_a, cpt_b, cpt_c], adjacency };
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");

        let result = engine.prior_marginal("C").expect("test: prior marginal query should succeed");
        // P(C=T) = 0.25*(0.9+0.6+0.6+0.1) = 0.25*2.2 = 0.55
        assert!((result.distribution[0].1 - 0.55).abs() < 1e-9, "P(C=T)={}", result.distribution[0].1);
    }

    #[test]
    fn test_ve_v_structure_with_evidence_on_collider() {
        // A → C ← B: after observing C, A and B become dependent.
        let mut variables = HashMap::new();
        variables.insert("A".into(), bool_var("A"));
        variables.insert("B".into(), bool_var("B"));
        variables.insert("C".into(), bool_var("C"));

        let f_a = Factor { id: "fa".into(), variables: vec!["A".into()], values: vec![0.5, 0.5], shape: vec![2] };
        let cpt_a = ConditionalProbabilityTable { variable: "A".into(), parents: vec![], factor: f_a };
        let f_b = Factor { id: "fb".into(), variables: vec!["B".into()], values: vec![0.5, 0.5], shape: vec![2] };
        let cpt_b = ConditionalProbabilityTable { variable: "B".into(), parents: vec![], factor: f_b };

        let f_c = Factor {
            id: "fc".into(),
            variables: vec!["A".into(), "B".into(), "C".into()],
            values: vec![0.9, 0.1, 0.6, 0.4, 0.6, 0.4, 0.1, 0.9],
            shape: vec![2, 2, 2],
        };
        let cpt_c = ConditionalProbabilityTable {
            variable: "C".into(),
            parents: vec!["A".into(), "B".into()],
            factor: f_c,
        };

        let mut adjacency = HashMap::new();
        adjacency.insert("C".into(), vec!["A".into(), "B".into()]);

        let net = BayesianNetwork { variables, cpts: vec![cpt_a, cpt_b, cpt_c], adjacency };
        let mut engine = BayesianNetworkInference::new(net, BniConfig::default()).expect("test: should succeed");

        // After observing C=T, A should not be uniform.
        let q = InferenceQuery {
            query_variables: vec!["A".into()],
            evidence: vec![Evidence { variable: "C".into(), observed_state: "T".into() }],
            algorithm: InferenceAlgorithm::VariableElimination,
        };
        let results = engine.query(&q).expect("test: BNI query should succeed");
        // Just check it's a valid distribution.
        let sum: f64 = results[0].distribution.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-9, "P(A|C=T) sums to {sum}");
    }

    // ── BniStats display ──────────────────────────────────────────────────────

    #[test]
    fn test_stats_default() {
        let s = BniStats::default();
        assert_eq!(s.queries_answered, 0);
        assert_eq!(s.cache_hits, 0);
        assert!(s.avg_factors_eliminated.abs() < 1e-15);
    }

    // ── min_fill / min_degree on larger graph ─────────────────────────────────

    #[test]
    fn test_min_fill_larger_graph() {
        // A network with 4 variables to stress-test ordering.
        let mut variables = HashMap::new();
        for name in ["A", "B", "C", "D"] {
            variables.insert(name.to_string(), bool_var(name));
        }

        let mk = |id: &str, vars: Vec<&str>, vals: Vec<f64>| -> Factor {
            let shape: Vec<usize> = vars.iter().map(|_| 2).collect();
            Factor { id: id.into(), variables: vars.into_iter().map(String::from).collect(), values: vals, shape }
        };

        let cpts = vec![
            ConditionalProbabilityTable { variable: "A".into(), parents: vec![], factor: mk("fa", vec!["A"], vec![0.6, 0.4]) },
            ConditionalProbabilityTable { variable: "B".into(), parents: vec!["A".into()], factor: mk("fb", vec!["A", "B"], vec![0.7, 0.3, 0.2, 0.8]) },
            ConditionalProbabilityTable { variable: "C".into(), parents: vec!["A".into()], factor: mk("fc", vec!["A", "C"], vec![0.5, 0.5, 0.9, 0.1]) },
            ConditionalProbabilityTable { variable: "D".into(), parents: vec!["B".into(), "C".into()], factor: mk("fd", vec!["B", "C", "D"], vec![0.8,0.2, 0.6,0.4, 0.4,0.6, 0.1,0.9]) },
        ];
        let mut adjacency = HashMap::new();
        adjacency.insert("B".into(), vec!["A".into()]);
        adjacency.insert("C".into(), vec!["A".into()]);
        adjacency.insert("D".into(), vec!["B".into(), "C".into()]);

        let net = BayesianNetwork { variables, cpts, adjacency };

        for ordering in [EliminationOrder::MinFill, EliminationOrder::MinDegree, EliminationOrder::Sequential] {
            let config = BniConfig { elimination_ordering: ordering, ..Default::default() };
            let mut engine = BayesianNetworkInference::new(net.clone(), config).expect("test: should succeed");
            let result = engine.prior_marginal("D").expect("test: prior marginal query should succeed");
            let sum: f64 = result.distribution.iter().map(|(_, p)| p).sum();
            assert!((sum - 1.0).abs() < 1e-8, "sum={sum}");
        }
    }
