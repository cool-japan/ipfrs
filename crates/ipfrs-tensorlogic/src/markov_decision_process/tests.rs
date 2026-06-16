//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::{
        xorshift64, xorshift_f64, MarkovDecisionProcess, MdpActionId, MdpError, MdpPolicy,
        MdpStateId, SolverConfig, SolverType, Transition, ValueFunction,
    };
    /// Simple two-state chain: s0 --a0(r=1)--> s1(terminal)
    fn two_state_mdp() -> MarkovDecisionProcess {
        let mut mdp = MarkovDecisionProcess::new(2, 1);
        mdp.set_terminal(MdpStateId(1), true)
            .expect("test: should succeed");
        mdp.add_transition(
            MdpStateId(0),
            MdpActionId(0),
            Transition {
                to_state: MdpStateId(1),
                probability: 1.0,
                reward: 1.0,
            },
        )
        .expect("test: should succeed");
        mdp
    }
    /// Grid-world-like 4-state MDP:
    ///   s0 -a0-> s1 (r=0)
    ///   s0 -a1-> s2 (r=0)
    ///   s1 -a0-> s3 (r=1)  [terminal]
    ///   s2 -a0-> s3 (r=5)  [terminal]
    fn four_state_mdp() -> MarkovDecisionProcess {
        let mut mdp = MarkovDecisionProcess::new(4, 2);
        mdp.set_terminal(MdpStateId(3), true)
            .expect("test: should succeed");
        mdp.add_transition(
            MdpStateId(0),
            MdpActionId(0),
            Transition {
                to_state: MdpStateId(1),
                probability: 1.0,
                reward: 0.0,
            },
        )
        .expect("test: should succeed");
        mdp.add_transition(
            MdpStateId(0),
            MdpActionId(1),
            Transition {
                to_state: MdpStateId(2),
                probability: 1.0,
                reward: 0.0,
            },
        )
        .expect("test: should succeed");
        mdp.add_transition(
            MdpStateId(1),
            MdpActionId(0),
            Transition {
                to_state: MdpStateId(3),
                probability: 1.0,
                reward: 1.0,
            },
        )
        .expect("test: should succeed");
        mdp.add_transition(
            MdpStateId(1),
            MdpActionId(1),
            Transition {
                to_state: MdpStateId(3),
                probability: 1.0,
                reward: 1.0,
            },
        )
        .expect("test: should succeed");
        mdp.add_transition(
            MdpStateId(2),
            MdpActionId(0),
            Transition {
                to_state: MdpStateId(3),
                probability: 1.0,
                reward: 5.0,
            },
        )
        .expect("test: should succeed");
        mdp.add_transition(
            MdpStateId(2),
            MdpActionId(1),
            Transition {
                to_state: MdpStateId(3),
                probability: 1.0,
                reward: 5.0,
            },
        )
        .expect("test: should succeed");
        mdp
    }
    #[test]
    fn test_state_id_newtype() {
        let s = MdpStateId(3);
        assert_eq!(s.0, 3);
    }
    #[test]
    fn test_action_id_newtype() {
        let a = MdpActionId(7);
        assert_eq!(a.0, 7);
    }
    #[test]
    fn test_state_id_equality() {
        assert_eq!(MdpStateId(1), MdpStateId(1));
        assert_ne!(MdpStateId(1), MdpStateId(2));
    }
    #[test]
    fn test_action_id_ordering() {
        assert!(MdpActionId(0) < MdpActionId(1));
    }
    #[test]
    fn test_value_function_zeros() {
        let vf = ValueFunction::zeros(5);
        for i in 0..5 {
            assert_eq!(vf.get(MdpStateId(i)), 0.0);
        }
    }
    #[test]
    fn test_value_function_set_get() {
        let mut vf = ValueFunction::zeros(3);
        vf.set(MdpStateId(1), 42.0);
        assert_eq!(vf.get(MdpStateId(0)), 0.0);
        assert_eq!(vf.get(MdpStateId(1)), 42.0);
        assert_eq!(vf.get(MdpStateId(2)), 0.0);
    }
    #[test]
    fn test_value_function_max_diff_same() {
        let vf = ValueFunction::zeros(3);
        assert_eq!(vf.max_diff(&vf.clone()), 0.0);
    }
    #[test]
    fn test_value_function_max_diff_nonzero() {
        let mut a = ValueFunction::zeros(3);
        let mut b = ValueFunction::zeros(3);
        a.set(MdpStateId(0), 1.0);
        b.set(MdpStateId(0), 3.0);
        a.set(MdpStateId(2), 10.0);
        b.set(MdpStateId(2), 5.0);
        assert!((a.max_diff(&b) - 5.0).abs() < 1e-10);
    }
    #[test]
    fn test_value_function_out_of_bounds_returns_zero() {
        let vf = ValueFunction::zeros(2);
        assert_eq!(vf.get(MdpStateId(99)), 0.0);
    }
    #[test]
    fn test_policy_set_and_get() {
        let mut p = MdpPolicy::default();
        p.set(MdpStateId(0), MdpActionId(1));
        assert_eq!(p.action_for(MdpStateId(0)), Some(MdpActionId(1)));
    }
    #[test]
    fn test_policy_missing_state_returns_none() {
        let p = MdpPolicy::default();
        assert_eq!(p.action_for(MdpStateId(99)), None);
    }
    #[test]
    fn test_new_creates_correct_sizes() {
        let mdp = MarkovDecisionProcess::new(5, 3);
        assert_eq!(mdp.num_states, 5);
        assert_eq!(mdp.num_actions, 3);
        assert_eq!(mdp.states.len(), 5);
        assert_eq!(mdp.actions.len(), 3);
    }
    #[test]
    fn test_new_states_non_terminal_by_default() {
        let mdp = MarkovDecisionProcess::new(4, 2);
        assert!(mdp.states.iter().all(|s| !s.is_terminal));
    }
    #[test]
    fn test_set_state_name_ok() {
        let mut mdp = MarkovDecisionProcess::new(3, 1);
        mdp.set_state_name(MdpStateId(1), "goal".to_string())
            .expect("test: should succeed");
        assert_eq!(mdp.states[1].name, "goal");
    }
    #[test]
    fn test_set_state_name_out_of_range() {
        let mut mdp = MarkovDecisionProcess::new(3, 1);
        let err = mdp
            .set_state_name(MdpStateId(10), "x".to_string())
            .unwrap_err();
        assert_eq!(err, MdpError::StateOutOfRange(10));
    }
    #[test]
    fn test_set_terminal_ok() {
        let mut mdp = MarkovDecisionProcess::new(3, 1);
        mdp.set_terminal(MdpStateId(2), true)
            .expect("test: should succeed");
        assert!(mdp.states[2].is_terminal);
    }
    #[test]
    fn test_set_terminal_out_of_range() {
        let mut mdp = MarkovDecisionProcess::new(3, 1);
        let err = mdp.set_terminal(MdpStateId(5), true).unwrap_err();
        assert_eq!(err, MdpError::StateOutOfRange(5));
    }
    #[test]
    fn test_set_action_name_ok() {
        let mut mdp = MarkovDecisionProcess::new(2, 2);
        mdp.set_action_name(MdpActionId(0), "left".to_string())
            .expect("test: should succeed");
        assert_eq!(mdp.actions[0], "left");
    }
    #[test]
    fn test_set_action_name_out_of_range() {
        let mut mdp = MarkovDecisionProcess::new(2, 2);
        let err = mdp
            .set_action_name(MdpActionId(99), "x".to_string())
            .unwrap_err();
        assert_eq!(err, MdpError::ActionOutOfRange(99));
    }
    #[test]
    fn test_add_transition_ok() {
        let mut mdp = MarkovDecisionProcess::new(3, 2);
        mdp.add_transition(
            MdpStateId(0),
            MdpActionId(1),
            Transition {
                to_state: MdpStateId(2),
                probability: 0.5,
                reward: 2.0,
            },
        )
        .expect("test: should succeed");
        let ts = mdp.transitions_for(MdpStateId(0), MdpActionId(1));
        assert_eq!(ts.len(), 1);
        assert!((ts[0].probability - 0.5).abs() < 1e-10);
    }
    #[test]
    fn test_add_transition_invalid_from() {
        let mut mdp = MarkovDecisionProcess::new(2, 1);
        let err = mdp
            .add_transition(
                MdpStateId(99),
                MdpActionId(0),
                Transition {
                    to_state: MdpStateId(0),
                    probability: 1.0,
                    reward: 0.0,
                },
            )
            .unwrap_err();
        assert_eq!(err, MdpError::StateOutOfRange(99));
    }
    #[test]
    fn test_add_transition_invalid_action() {
        let mut mdp = MarkovDecisionProcess::new(2, 1);
        let err = mdp
            .add_transition(
                MdpStateId(0),
                MdpActionId(99),
                Transition {
                    to_state: MdpStateId(1),
                    probability: 1.0,
                    reward: 0.0,
                },
            )
            .unwrap_err();
        assert_eq!(err, MdpError::ActionOutOfRange(99));
    }
    #[test]
    fn test_add_transition_invalid_to_state() {
        let mut mdp = MarkovDecisionProcess::new(2, 1);
        let err = mdp
            .add_transition(
                MdpStateId(0),
                MdpActionId(0),
                Transition {
                    to_state: MdpStateId(99),
                    probability: 1.0,
                    reward: 0.0,
                },
            )
            .unwrap_err();
        assert_eq!(err, MdpError::StateOutOfRange(99));
    }
    #[test]
    fn test_add_transition_invalid_probability_negative() {
        let mut mdp = MarkovDecisionProcess::new(2, 1);
        let err = mdp
            .add_transition(
                MdpStateId(0),
                MdpActionId(0),
                Transition {
                    to_state: MdpStateId(1),
                    probability: -0.1,
                    reward: 0.0,
                },
            )
            .unwrap_err();
        assert!(matches!(err, MdpError::InvalidProbability(_)));
    }
    #[test]
    fn test_add_transition_invalid_probability_gt_one() {
        let mut mdp = MarkovDecisionProcess::new(2, 1);
        let err = mdp
            .add_transition(
                MdpStateId(0),
                MdpActionId(0),
                Transition {
                    to_state: MdpStateId(1),
                    probability: 1.5,
                    reward: 0.0,
                },
            )
            .unwrap_err();
        assert!(matches!(err, MdpError::InvalidProbability(_)));
    }
    #[test]
    fn test_transitions_for_empty() {
        let mdp = MarkovDecisionProcess::new(3, 2);
        let ts = mdp.transitions_for(MdpStateId(0), MdpActionId(0));
        assert!(ts.is_empty());
    }
    #[test]
    fn test_value_iteration_two_state_converges() {
        let mdp = two_state_mdp();
        let config = SolverConfig::default();
        let (vf, result) = mdp.value_iteration(&config);
        assert!(result.converged);
        assert!((vf.get(MdpStateId(1)) - 0.0).abs() < 1e-6);
        assert!(vf.get(MdpStateId(0)) > 0.9);
    }
    #[test]
    fn test_value_iteration_terminal_state_always_zero() {
        let mdp = two_state_mdp();
        let config = SolverConfig::default();
        let (vf, _) = mdp.value_iteration(&config);
        assert_eq!(vf.get(MdpStateId(1)), 0.0);
    }
    #[test]
    fn test_value_iteration_four_state_policy_takes_high_reward() {
        let mdp = four_state_mdp();
        let config = SolverConfig::default();
        let (vf, result) = mdp.value_iteration(&config);
        assert!(result.converged);
        let v0 = vf.get(MdpStateId(0));
        let v1 = vf.get(MdpStateId(1));
        let v2 = vf.get(MdpStateId(2));
        assert!(v2 > v1, "V(s2)={v2} should be > V(s1)={v1}");
        assert!(
            v0 > 4.0,
            "V(s0)={v0} should be > 4.0 when optimal path chosen"
        );
    }
    #[test]
    fn test_value_iteration_no_transitions_stays_zero() {
        let mdp = MarkovDecisionProcess::new(3, 2);
        let config = SolverConfig::default();
        let (vf, result) = mdp.value_iteration(&config);
        assert!(result.converged);
        for i in 0..3 {
            assert_eq!(vf.get(MdpStateId(i)), 0.0);
        }
    }
    #[test]
    fn test_value_iteration_reports_iterations() {
        let mdp = two_state_mdp();
        let config = SolverConfig::default();
        let (_, result) = mdp.value_iteration(&config);
        assert!(result.iterations > 0);
    }
    #[test]
    fn test_extract_policy_four_state_selects_high_reward_action() {
        let mdp = four_state_mdp();
        let config = SolverConfig::default();
        let (vf, _) = mdp.value_iteration(&config);
        let policy = mdp.extract_policy(&vf, &config);
        assert_eq!(policy.action_for(MdpStateId(0)), Some(MdpActionId(1)));
    }
    #[test]
    fn test_extract_policy_terminal_states_have_no_action() {
        let mdp = four_state_mdp();
        let config = SolverConfig::default();
        let (vf, _) = mdp.value_iteration(&config);
        let policy = mdp.extract_policy(&vf, &config);
        assert_eq!(policy.action_for(MdpStateId(3)), None);
    }
    #[test]
    fn test_policy_evaluation_two_state() {
        let mdp = two_state_mdp();
        let config = SolverConfig::default();
        let mut policy = MdpPolicy::default();
        policy.set(MdpStateId(0), MdpActionId(0));
        let vf = mdp.policy_evaluation(&policy, &config);
        assert!(vf.get(MdpStateId(0)) > 0.9);
        assert_eq!(vf.get(MdpStateId(1)), 0.0);
    }
    #[test]
    fn test_policy_evaluation_no_transitions_zero() {
        let mdp = MarkovDecisionProcess::new(3, 2);
        let config = SolverConfig::default();
        let policy = MdpPolicy::default();
        let vf = mdp.policy_evaluation(&policy, &config);
        for i in 0..3 {
            assert_eq!(vf.get(MdpStateId(i)), 0.0);
        }
    }
    #[test]
    fn test_policy_iteration_converges_two_state() {
        let mdp = two_state_mdp();
        let config = SolverConfig::default();
        let (policy, vf, result) = mdp.policy_iteration(&config);
        assert!(result.converged);
        assert_eq!(policy.action_for(MdpStateId(0)), Some(MdpActionId(0)));
        assert!(vf.get(MdpStateId(0)) > 0.9);
    }
    #[test]
    fn test_policy_iteration_matches_value_iteration_four_state() {
        let mdp = four_state_mdp();
        let config = SolverConfig {
            gamma: 0.9,
            ..Default::default()
        };
        let (pi_policy, pi_vf, pi_result) = mdp.policy_iteration(&config);
        let (vi_vf, vi_result) = mdp.value_iteration(&config);
        assert!(pi_result.converged);
        assert!(vi_result.converged);
        let vi_policy = mdp.extract_policy(&vi_vf, &config);
        assert_eq!(
            pi_policy.action_for(MdpStateId(0)),
            vi_policy.action_for(MdpStateId(0))
        );
        let diff = pi_vf.max_diff(&vi_vf);
        assert!(diff < 1e-4, "pi_vf vs vi_vf differ by {diff}");
    }
    #[test]
    fn test_policy_iteration_terminal_states_excluded() {
        let mdp = four_state_mdp();
        let config = SolverConfig::default();
        let (policy, _, _) = mdp.policy_iteration(&config);
        assert_eq!(policy.action_for(MdpStateId(3)), None);
    }
    #[test]
    fn test_q_values_length() {
        let mdp = MarkovDecisionProcess::new(4, 3);
        let vf = ValueFunction::zeros(4);
        let qs = mdp.q_values(&vf, MdpStateId(0), 0.99);
        assert_eq!(qs.len(), 3);
    }
    #[test]
    fn test_q_values_no_transitions_all_zero() {
        let mdp = MarkovDecisionProcess::new(4, 3);
        let vf = ValueFunction::zeros(4);
        let qs = mdp.q_values(&vf, MdpStateId(0), 0.99);
        assert!(qs.iter().all(|&q| q == 0.0));
    }
    #[test]
    fn test_q_values_correct_calculation() {
        let mut mdp = MarkovDecisionProcess::new(3, 2);
        mdp.set_terminal(MdpStateId(2), true)
            .expect("test: should succeed");
        mdp.add_transition(
            MdpStateId(0),
            MdpActionId(0),
            Transition {
                to_state: MdpStateId(2),
                probability: 1.0,
                reward: 10.0,
            },
        )
        .expect("test: should succeed");
        let mut vf = ValueFunction::zeros(3);
        vf.set(MdpStateId(2), 0.0);
        let qs = mdp.q_values(&vf, MdpStateId(0), 0.9);
        assert!((qs[0] - 10.0).abs() < 1e-10);
        assert_eq!(qs[1], 0.0);
    }
    #[test]
    fn test_expected_return_reads_value_function() {
        let mdp = two_state_mdp();
        let config = SolverConfig::default();
        let (vf, _) = mdp.value_iteration(&config);
        let policy = mdp.extract_policy(&vf, &config);
        let ret = mdp.expected_return(&vf, &policy, MdpStateId(0), config.gamma);
        assert!(ret > 0.9);
    }
    #[test]
    fn test_expected_return_terminal_is_zero() {
        let mdp = two_state_mdp();
        let config = SolverConfig::default();
        let (vf, _) = mdp.value_iteration(&config);
        let policy = MdpPolicy::default();
        let ret = mdp.expected_return(&vf, &policy, MdpStateId(1), config.gamma);
        assert_eq!(ret, 0.0);
    }
    #[test]
    fn test_stats_empty_mdp() {
        let mdp = MarkovDecisionProcess::new(3, 2);
        let s = mdp.stats();
        assert_eq!(s.num_states, 3);
        assert_eq!(s.num_actions, 2);
        assert_eq!(s.num_transitions, 0);
        assert_eq!(s.terminal_states, 0);
        assert_eq!(s.avg_branching_factor, 0.0);
    }
    #[test]
    fn test_stats_with_transitions() {
        let mdp = four_state_mdp();
        let s = mdp.stats();
        assert_eq!(s.num_states, 4);
        assert_eq!(s.num_actions, 2);
        assert_eq!(s.terminal_states, 1);
        assert!(s.num_transitions > 0);
        assert!(s.avg_branching_factor > 0.0);
    }
    #[test]
    fn test_stats_terminal_count() {
        let mut mdp = MarkovDecisionProcess::new(5, 2);
        mdp.set_terminal(MdpStateId(2), true)
            .expect("test: should succeed");
        mdp.set_terminal(MdpStateId(4), true)
            .expect("test: should succeed");
        assert_eq!(mdp.stats().terminal_states, 2);
    }
    #[test]
    fn test_solver_config_defaults() {
        let cfg = SolverConfig::default();
        assert_eq!(cfg.gamma, 0.99);
        assert_eq!(cfg.epsilon, 1e-6);
        assert_eq!(cfg.max_iterations, 1000);
    }
    #[test]
    fn test_mdp_error_display_state_out_of_range() {
        let e = MdpError::StateOutOfRange(42);
        assert!(e.to_string().contains("42"));
    }
    #[test]
    fn test_mdp_error_display_action_out_of_range() {
        let e = MdpError::ActionOutOfRange(7);
        assert!(e.to_string().contains("7"));
    }
    #[test]
    fn test_mdp_error_display_invalid_probability() {
        let e = MdpError::InvalidProbability(-0.5);
        assert!(e.to_string().contains("-0.5"));
    }
    #[test]
    fn test_mdp_error_display_no_transitions() {
        let e = MdpError::NoTransitions {
            state: 1,
            action: 2,
        };
        let s = e.to_string();
        assert!(s.contains("1") && s.contains("2"));
    }
    #[test]
    fn test_stochastic_q_value_correctly_weighted() {
        let mut mdp = MarkovDecisionProcess::new(3, 1);
        mdp.set_terminal(MdpStateId(1), true)
            .expect("test: should succeed");
        mdp.set_terminal(MdpStateId(2), true)
            .expect("test: should succeed");
        mdp.add_transition(
            MdpStateId(0),
            MdpActionId(0),
            Transition {
                to_state: MdpStateId(1),
                probability: 0.7,
                reward: 0.0,
            },
        )
        .expect("test: should succeed");
        mdp.add_transition(
            MdpStateId(0),
            MdpActionId(0),
            Transition {
                to_state: MdpStateId(2),
                probability: 0.3,
                reward: 10.0,
            },
        )
        .expect("test: should succeed");
        let vf = ValueFunction::zeros(3);
        let qs = mdp.q_values(&vf, MdpStateId(0), 0.9);
        assert!((qs[0] - 3.0).abs() < 1e-10);
    }
    #[test]
    fn test_value_iteration_stochastic_mdp_converges() {
        let mut mdp = MarkovDecisionProcess::new(3, 1);
        mdp.set_terminal(MdpStateId(1), true)
            .expect("test: should succeed");
        mdp.set_terminal(MdpStateId(2), true)
            .expect("test: should succeed");
        mdp.add_transition(
            MdpStateId(0),
            MdpActionId(0),
            Transition {
                to_state: MdpStateId(1),
                probability: 0.7,
                reward: 0.0,
            },
        )
        .expect("test: should succeed");
        mdp.add_transition(
            MdpStateId(0),
            MdpActionId(0),
            Transition {
                to_state: MdpStateId(2),
                probability: 0.3,
                reward: 10.0,
            },
        )
        .expect("test: should succeed");
        let config = SolverConfig::default();
        let (_, result) = mdp.value_iteration(&config);
        assert!(result.converged);
    }
    #[test]
    fn test_solve_value_iteration_dispatch() {
        let mdp = two_state_mdp();
        let config = SolverConfig {
            solver: SolverType::ValueIteration,
            ..Default::default()
        };
        let (vf, policy, result) = mdp.solve(&config).expect("test: should succeed");
        assert!(result.converged);
        assert!(vf.get(MdpStateId(0)) > 0.9);
        assert_eq!(policy.action_for(MdpStateId(0)), Some(MdpActionId(0)));
    }
    #[test]
    fn test_solve_policy_iteration_dispatch() {
        let mdp = four_state_mdp();
        let config = SolverConfig {
            solver: SolverType::PolicyIteration,
            ..Default::default()
        };
        let (vf, policy, result) = mdp.solve(&config).expect("test: should succeed");
        assert!(result.converged);
        assert_eq!(policy.action_for(MdpStateId(0)), Some(MdpActionId(1)));
        assert!(vf.get(MdpStateId(0)) > 4.0);
    }
    #[test]
    fn test_solve_modified_policy_iteration_dispatch() {
        let mdp = four_state_mdp();
        let config = SolverConfig {
            solver: SolverType::ModifiedPolicyIteration(5),
            gamma: 0.9,
            ..Default::default()
        };
        let (vf, policy, result) = mdp.solve(&config).expect("test: should succeed");
        assert!(result.converged);
        assert_eq!(policy.action_for(MdpStateId(0)), Some(MdpActionId(1)));
        assert!(vf.get(MdpStateId(0)) > 3.0);
    }
    #[test]
    fn test_solve_qlearning_dispatch() {
        let mdp = four_state_mdp();
        let config = SolverConfig {
            solver: SolverType::Qlearning {
                alpha: 0.1,
                epsilon: 0.2,
            },
            max_iterations: 5000,
            gamma: 0.9,
            ..Default::default()
        };
        let (vf, _policy, _result) = mdp.solve(&config).expect("test: should succeed");
        assert!(
            vf.get(MdpStateId(0)) >= 0.0,
            "V(s0)={}",
            vf.get(MdpStateId(0))
        );
    }
    #[test]
    fn test_add_state_increments_count() {
        let mut mdp = MarkovDecisionProcess::new(2, 1);
        let s = mdp.add_state(false);
        assert_eq!(s, MdpStateId(2));
        assert_eq!(mdp.num_states, 3);
    }
    #[test]
    fn test_add_state_terminal_flag() {
        let mut mdp = MarkovDecisionProcess::new(1, 1);
        let s = mdp.add_state(true);
        assert!(mdp.states[s.0].is_terminal);
    }
    #[test]
    fn test_add_action_increments_count() {
        let mut mdp = MarkovDecisionProcess::new(2, 1);
        let a = mdp.add_action();
        assert_eq!(a, MdpActionId(1));
        assert_eq!(mdp.num_actions, 2);
    }
    #[test]
    fn test_add_state_then_transition() {
        let mut mdp = MarkovDecisionProcess::new(1, 1);
        let s1 = mdp.add_state(true);
        mdp.add_transition(
            MdpStateId(0),
            MdpActionId(0),
            Transition {
                to_state: s1,
                probability: 1.0,
                reward: 7.0,
            },
        )
        .expect("test: should succeed");
        let ts = mdp.transitions_for(MdpStateId(0), MdpActionId(0));
        assert_eq!(ts.len(), 1);
        assert!((ts[0].reward - 7.0).abs() < 1e-10);
    }
    #[test]
    fn test_validate_ok_deterministic() {
        let mdp = two_state_mdp();
        assert!(mdp.validate().is_ok());
    }
    #[test]
    fn test_validate_ok_stochastic() {
        let mut mdp = MarkovDecisionProcess::new(3, 1);
        mdp.set_terminal(MdpStateId(1), true)
            .expect("test: should succeed");
        mdp.set_terminal(MdpStateId(2), true)
            .expect("test: should succeed");
        mdp.add_transition(
            MdpStateId(0),
            MdpActionId(0),
            Transition {
                to_state: MdpStateId(1),
                probability: 0.4,
                reward: 1.0,
            },
        )
        .expect("test: should succeed");
        mdp.add_transition(
            MdpStateId(0),
            MdpActionId(0),
            Transition {
                to_state: MdpStateId(2),
                probability: 0.6,
                reward: 2.0,
            },
        )
        .expect("test: should succeed");
        assert!(mdp.validate().is_ok());
    }
    #[test]
    fn test_validate_fails_probability_sum_not_one() {
        let mut mdp = MarkovDecisionProcess::new(3, 1);
        mdp.set_terminal(MdpStateId(1), true)
            .expect("test: should succeed");
        mdp.set_terminal(MdpStateId(2), true)
            .expect("test: should succeed");
        mdp.add_transition(
            MdpStateId(0),
            MdpActionId(0),
            Transition {
                to_state: MdpStateId(1),
                probability: 0.3,
                reward: 1.0,
            },
        )
        .expect("test: should succeed");
        mdp.add_transition(
            MdpStateId(0),
            MdpActionId(0),
            Transition {
                to_state: MdpStateId(2),
                probability: 0.6,
                reward: 2.0,
            },
        )
        .expect("test: should succeed");
        let err = mdp.validate();
        assert!(err.is_err(), "expected validation to fail");
    }
    #[test]
    fn test_validate_empty_mdp_ok() {
        let mdp = MarkovDecisionProcess::new(5, 3);
        assert!(mdp.validate().is_ok());
    }
    #[test]
    fn test_simulate_deterministic_trajectory() {
        let mdp = two_state_mdp();
        let config = SolverConfig::default();
        let (vf, _) = mdp.value_iteration(&config);
        let policy = mdp.extract_policy(&vf, &config);
        let traj = mdp.simulate(&policy, MdpStateId(0), 10, 42);
        assert_eq!(traj.len(), 1);
        assert_eq!(traj[0].0, MdpStateId(0));
        assert_eq!(traj[0].1, MdpActionId(0));
        assert!((traj[0].2 - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_simulate_starts_at_terminal_returns_empty() {
        let mdp = two_state_mdp();
        let policy = MdpPolicy::default();
        let traj = mdp.simulate(&policy, MdpStateId(1), 10, 1);
        assert!(traj.is_empty());
    }
    #[test]
    fn test_simulate_no_policy_entry_returns_empty() {
        let mdp = two_state_mdp();
        let policy = MdpPolicy::default();
        let traj = mdp.simulate(&policy, MdpStateId(0), 5, 1);
        assert!(traj.is_empty());
    }
    #[test]
    fn test_simulate_respects_step_cap() {
        let mut mdp = MarkovDecisionProcess::new(1, 1);
        mdp.add_transition(
            MdpStateId(0),
            MdpActionId(0),
            Transition {
                to_state: MdpStateId(0),
                probability: 1.0,
                reward: 1.0,
            },
        )
        .expect("test: should succeed");
        let mut policy = MdpPolicy::default();
        policy.set(MdpStateId(0), MdpActionId(0));
        let traj = mdp.simulate(&policy, MdpStateId(0), 7, 99);
        assert_eq!(traj.len(), 7);
    }
    #[test]
    fn test_simulate_stochastic_reaches_terminal() {
        let mut mdp = MarkovDecisionProcess::new(3, 1);
        mdp.set_terminal(MdpStateId(1), true)
            .expect("test: should succeed");
        mdp.set_terminal(MdpStateId(2), true)
            .expect("test: should succeed");
        mdp.add_transition(
            MdpStateId(0),
            MdpActionId(0),
            Transition {
                to_state: MdpStateId(1),
                probability: 0.5,
                reward: 0.0,
            },
        )
        .expect("test: should succeed");
        mdp.add_transition(
            MdpStateId(0),
            MdpActionId(0),
            Transition {
                to_state: MdpStateId(2),
                probability: 0.5,
                reward: 1.0,
            },
        )
        .expect("test: should succeed");
        let mut policy = MdpPolicy::default();
        policy.set(MdpStateId(0), MdpActionId(0));
        let traj = mdp.simulate(&policy, MdpStateId(0), 100, 42);
        assert_eq!(traj.len(), 1);
    }
    #[test]
    fn test_best_action_two_state() {
        let mdp = two_state_mdp();
        let config = SolverConfig::default();
        let (vf, _) = mdp.value_iteration(&config);
        let a = mdp
            .best_action(MdpStateId(0), &vf, config.gamma)
            .expect("test: should succeed");
        assert_eq!(a, MdpActionId(0));
    }
    #[test]
    fn test_best_action_out_of_range_errors() {
        let mdp = two_state_mdp();
        let vf = ValueFunction::zeros(2);
        let err = mdp.best_action(MdpStateId(99), &vf, 0.9);
        assert!(err.is_err());
    }
    #[test]
    fn test_best_action_selects_high_reward_path() {
        let mdp = four_state_mdp();
        let config = SolverConfig::default();
        let (vf, _) = mdp.value_iteration(&config);
        let a = mdp
            .best_action(MdpStateId(0), &vf, config.gamma)
            .expect("test: should succeed");
        assert_eq!(a, MdpActionId(1));
    }
    #[test]
    fn test_xorshift64_changes_state() {
        let mut s: u64 = 1;
        let v1 = xorshift64(&mut s);
        let v2 = xorshift64(&mut s);
        assert_ne!(v1, v2);
    }
    #[test]
    fn test_xorshift_f64_in_range() {
        let mut s: u64 = 987654321;
        for _ in 0..1000 {
            let v = xorshift_f64(&mut s);
            assert!((0.0..1.0).contains(&v));
        }
    }
    #[test]
    fn test_xorshift64_deterministic_with_same_seed() {
        let mut s1: u64 = 42;
        let mut s2: u64 = 42;
        for _ in 0..20 {
            assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
        }
    }
    #[test]
    fn test_modified_pi_converges_k1() {
        let mdp = four_state_mdp();
        let config = SolverConfig {
            gamma: 0.9,
            ..Default::default()
        };
        let (vf, result) = mdp.modified_policy_iteration(&config, 1);
        assert!(result.converged);
        assert!(vf.get(MdpStateId(0)) > 3.0);
    }
    #[test]
    fn test_modified_pi_converges_k10() {
        let mdp = four_state_mdp();
        let config = SolverConfig {
            gamma: 0.9,
            ..Default::default()
        };
        let (vf, result) = mdp.modified_policy_iteration(&config, 10);
        assert!(result.converged);
        assert!(vf.get(MdpStateId(0)) > 3.0);
    }
    #[test]
    fn test_q_learning_positive_values_after_training() {
        let mdp = four_state_mdp();
        let config = SolverConfig {
            max_iterations: 2000,
            gamma: 0.9,
            ..Default::default()
        };
        let (vf, _policy, result) = mdp.q_learning(&config, 0.1, 0.3);
        assert!(result.iterations > 0);
        assert!(vf.get(MdpStateId(0)) >= 0.0);
    }
    #[test]
    fn test_q_learning_all_terminal_mdp() {
        let mut mdp = MarkovDecisionProcess::new(2, 1);
        mdp.set_terminal(MdpStateId(0), true)
            .expect("test: should succeed");
        mdp.set_terminal(MdpStateId(1), true)
            .expect("test: should succeed");
        let config = SolverConfig::default();
        let (vf, _policy, result) = mdp.q_learning(&config, 0.1, 0.1);
        assert!(!result.converged);
        assert_eq!(vf.get(MdpStateId(0)), 0.0);
    }
    #[test]
    fn test_solver_config_default_is_value_iteration() {
        let cfg = SolverConfig::default();
        assert!(matches!(cfg.solver, SolverType::ValueIteration));
    }
    #[test]
    fn test_solver_config_policy_iteration_variant() {
        let cfg = SolverConfig {
            solver: SolverType::PolicyIteration,
            ..Default::default()
        };
        assert!(matches!(cfg.solver, SolverType::PolicyIteration));
    }
    #[test]
    fn test_solver_config_modified_pi_variant() {
        let cfg = SolverConfig {
            solver: SolverType::ModifiedPolicyIteration(3),
            ..Default::default()
        };
        if let SolverType::ModifiedPolicyIteration(k) = cfg.solver {
            assert_eq!(k, 3);
        } else {
            panic!("wrong variant");
        }
    }
    #[test]
    fn test_solver_config_qlearning_variant() {
        let cfg = SolverConfig {
            solver: SolverType::Qlearning {
                alpha: 0.05,
                epsilon: 0.1,
            },
            ..Default::default()
        };
        if let SolverType::Qlearning { alpha, epsilon } = cfg.solver {
            assert!((alpha - 0.05).abs() < 1e-12);
            assert!((epsilon - 0.1).abs() < 1e-12);
        } else {
            panic!("wrong variant");
        }
    }
}
