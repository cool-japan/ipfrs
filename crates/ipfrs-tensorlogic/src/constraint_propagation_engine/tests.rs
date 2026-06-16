//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::functions::{ceil_div, floor_div, xorshift64};
use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    fn make_engine() -> ConstraintPropagationEngine {
        ConstraintPropagationEngine::default_engine()
    }
    fn interval(lo: i64, hi: i64) -> CpeDomain {
        CpeDomain::Interval { lo, hi }
    }
    fn finite(vals: &[i64]) -> CpeDomain {
        CpeDomain::Finite(vals.iter().copied().collect())
    }
    #[test]
    fn test_domain_interval_size() {
        assert_eq!(interval(1, 5).size(), 5);
        assert_eq!(interval(3, 3).size(), 1);
        assert_eq!(interval(5, 1).size(), 0);
    }
    #[test]
    fn test_domain_interval_contains() {
        let d = interval(0, 4);
        assert!(d.contains(0));
        assert!(d.contains(4));
        assert!(!d.contains(5));
        assert!(!d.contains(-1));
    }
    #[test]
    fn test_domain_finite_size() {
        assert_eq!(finite(&[1, 3, 5, 7]).size(), 4);
        assert_eq!(finite(&[]).size(), 0);
    }
    #[test]
    fn test_domain_boolean() {
        let d = CpeDomain::Boolean;
        assert_eq!(d.size(), 2);
        assert!(d.contains(0));
        assert!(d.contains(1));
        assert!(!d.contains(2));
    }
    #[test]
    fn test_domain_remove_lo_bound() {
        let mut d = interval(1, 5);
        assert!(d.remove(1));
        assert_eq!(d.min_val(), Some(2));
    }
    #[test]
    fn test_domain_remove_hi_bound() {
        let mut d = interval(1, 5);
        assert!(d.remove(5));
        assert_eq!(d.max_val(), Some(4));
    }
    #[test]
    fn test_domain_remove_middle_materialises() {
        let mut d = interval(1, 5);
        assert!(d.remove(3));
        let vals = d.values();
        assert!(!vals.contains(&3));
        assert!(vals.contains(&1));
        assert!(vals.contains(&5));
    }
    #[test]
    fn test_domain_tighten_lo() {
        let mut d = interval(1, 10);
        assert!(d.tighten_lo(5));
        assert_eq!(d.min_val(), Some(5));
    }
    #[test]
    fn test_domain_tighten_hi() {
        let mut d = interval(1, 10);
        assert!(d.tighten_hi(7));
        assert_eq!(d.max_val(), Some(7));
    }
    #[test]
    fn test_domain_singleton() {
        let d = interval(3, 3);
        assert_eq!(d.singleton(), Some(3));
    }
    #[test]
    fn test_domain_assign_value() {
        let mut d = interval(1, 10);
        d.assign_value(6);
        assert_eq!(d.singleton(), Some(6));
    }
    #[test]
    fn test_domain_is_empty_interval() {
        assert!(interval(5, 3).is_empty());
        assert!(!interval(3, 5).is_empty());
    }
    #[test]
    fn test_domain_is_empty_finite() {
        assert!(finite(&[]).is_empty());
        assert!(!finite(&[1]).is_empty());
    }
    #[test]
    fn test_domain_values_interval() {
        let d = interval(2, 5);
        assert_eq!(d.values(), vec![2, 3, 4, 5]);
    }
    #[test]
    fn test_domain_boolean_values() {
        let d = CpeDomain::Boolean;
        let v = d.values();
        assert_eq!(v, vec![0, 1]);
    }
    #[test]
    fn test_add_variable_returns_sequential_ids() {
        let mut e = make_engine();
        let a = e.add_variable("a".to_string(), interval(0, 5));
        let b = e.add_variable("b".to_string(), interval(0, 5));
        assert_eq!(a + 1, b);
    }
    #[test]
    fn test_domain_size_after_add() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(1, 10));
        assert_eq!(e.domain_size(x), Some(10));
    }
    #[test]
    fn test_domain_size_unknown_var() {
        let e = make_engine();
        assert_eq!(e.domain_size(999), None);
    }
    #[test]
    fn test_assign_valid_value() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(1, 5));
        assert!(e.assign(x, 3).is_ok());
        assert_eq!(e.value_of(x), Some(3));
    }
    #[test]
    fn test_assign_invalid_value() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(1, 5));
        let err = e.assign(x, 10);
        assert!(err.is_err());
    }
    #[test]
    fn test_assign_unknown_var() {
        let mut e = make_engine();
        let err = e.assign(999, 1);
        assert!(matches!(err, Err(CpeError::UnknownVariable(999))));
    }
    #[test]
    fn test_propagate_empty_engine() {
        let mut e = make_engine();
        let r = e.propagate();
        assert!(matches!(r, Ok(CpePropagationResult::Solved)));
    }
    #[test]
    fn test_is_consistent_initial() {
        let mut e = make_engine();
        e.add_variable("x".to_string(), interval(1, 5));
        assert!(e.is_consistent());
    }
    #[test]
    fn test_equal_constraint_propagation() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(1, 5));
        let y = e.add_variable("y".to_string(), interval(3, 8));
        e.add_constraint(CpeConstraint::Equal(x, y));
        let r = e.propagate().expect("test: should succeed");
        assert_ne!(r, CpePropagationResult::Infeasible);
        assert!(e.domain_size(x).expect("test: should succeed") <= 3);
    }
    #[test]
    fn test_equal_constraint_infeasible() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), finite(&[1, 2]));
        let y = e.add_variable("y".to_string(), finite(&[3, 4]));
        e.add_constraint(CpeConstraint::Equal(x, y));
        let r = e.propagate();
        assert!(
            matches!(r, Ok(CpePropagationResult::Infeasible))
                || matches!(r, Err(CpeError::DomainEmpty(_)))
        );
    }
    #[test]
    fn test_not_equal_constraint() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), finite(&[1]));
        let y = e.add_variable("y".to_string(), finite(&[1, 2]));
        e.add_constraint(CpeConstraint::NotEqual(x, y));
        e.propagate().expect("test: should succeed");
        assert_eq!(e.domain_size(y), Some(1));
        assert_eq!(e.value_of(y), Some(2));
    }
    #[test]
    fn test_less_than_constraint() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(1, 10));
        let y = e.add_variable("y".to_string(), interval(1, 10));
        e.add_constraint(CpeConstraint::LessThan(x, y));
        e.assign(y, 3).expect("test: should succeed");
        assert!(e.domain_size(x).expect("test: should succeed") <= 2);
    }
    #[test]
    fn test_less_equal_constraint() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(1, 10));
        let y = e.add_variable("y".to_string(), interval(1, 10));
        e.add_constraint(CpeConstraint::LessEqual(x, y));
        e.assign(y, 3).expect("test: should succeed");
        assert!(e.domain_size(x).expect("test: should succeed") <= 3);
    }
    #[test]
    fn test_all_different_basic() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), finite(&[1, 2]));
        let y = e.add_variable("y".to_string(), finite(&[1, 2]));
        e.add_constraint(CpeConstraint::AllDifferent(vec![x, y]));
        e.assign(x, 1).expect("test: should succeed");
        assert_eq!(e.value_of(y), Some(2));
    }
    #[test]
    fn test_sum_constraint_basic() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(1, 5));
        let y = e.add_variable("y".to_string(), interval(1, 5));
        e.add_constraint(CpeConstraint::Sum {
            vars: vec![x, y],
            total: 7,
        });
        e.assign(x, 3).expect("test: should succeed");
        assert_eq!(e.value_of(y), Some(4));
    }
    #[test]
    fn test_in_domain_constraint() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(1, 10));
        e.add_constraint(CpeConstraint::InDomain(x, vec![2, 4, 6, 8]));
        e.propagate().expect("test: should succeed");
        let vals = e
            .variables
            .get(&x)
            .expect("test: should succeed")
            .domain
            .values();
        assert!(vals.iter().all(|v| v % 2 == 0));
    }
    #[test]
    fn test_abs_constraint_basic() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(-5, 5));
        let y = e.add_variable("y".to_string(), interval(0, 10));
        e.add_constraint(CpeConstraint::Abs(x, y));
        e.assign(x, -3).expect("test: should succeed");
        assert_eq!(e.value_of(y), Some(3));
    }
    #[test]
    fn test_linear_expr_constraint() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(0, 10));
        let y = e.add_variable("y".to_string(), interval(0, 10));
        e.add_constraint(CpeConstraint::LinearExpr {
            coeffs: vec![(x, 2), (y, 3)],
            rhs: 12,
        });
        e.assign(x, 3).expect("test: should succeed");
        assert_eq!(e.value_of(y), Some(2));
    }
    #[test]
    fn test_is_solved_after_all_assigned() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), finite(&[3]));
        let y = e.add_variable("y".to_string(), finite(&[7]));
        e.propagate().expect("test: should succeed");
        assert!(e.is_solved());
        let _ = (x, y);
    }
    #[test]
    fn test_propagation_stats_incremented() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(1, 5));
        let y = e.add_variable("y".to_string(), interval(1, 5));
        e.add_constraint(CpeConstraint::Equal(x, y));
        e.propagate().expect("test: should succeed");
        let stats = e.propagation_stats();
        assert!(stats.total_propagations > 0);
        let _ = (x, y);
    }
    #[test]
    fn test_backtrack_solve_simple() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), finite(&[1, 2, 3]));
        let y = e.add_variable("y".to_string(), finite(&[1, 2, 3]));
        e.add_constraint(CpeConstraint::AllDifferent(vec![x, y]));
        e.add_constraint(CpeConstraint::LessThan(x, y));
        let sol = e.backtrack_solve();
        assert!(sol.is_some());
        let s = sol.expect("test: should succeed");
        assert!(s[&x] < s[&y]);
        assert_ne!(s[&x], s[&y]);
    }
    #[test]
    fn test_backtrack_solve_unsatisfiable() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), finite(&[1]));
        let y = e.add_variable("y".to_string(), finite(&[1]));
        e.add_constraint(CpeConstraint::AllDifferent(vec![x, y]));
        let sol = e.backtrack_solve();
        assert!(sol.is_none());
    }
    #[test]
    fn test_backtrack_solve_n_queens_3() {
        let mut e = make_engine();
        let q1 = e.add_variable("q1".to_string(), interval(0, 2));
        let q2 = e.add_variable("q2".to_string(), interval(0, 2));
        let q3 = e.add_variable("q3".to_string(), interval(0, 2));
        e.add_constraint(CpeConstraint::AllDifferent(vec![q1, q2, q3]));
        e.add_constraint(CpeConstraint::NotEqual(q1, q2));
        let sol = e.backtrack_solve();
        let _ = sol;
    }
    #[test]
    fn test_backtrack_solve_sum_constraint() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(1, 5));
        let y = e.add_variable("y".to_string(), interval(1, 5));
        e.add_constraint(CpeConstraint::Sum {
            vars: vec![x, y],
            total: 6,
        });
        let sol = e.backtrack_solve();
        assert!(sol.is_some());
        let s = sol.expect("test: should succeed");
        assert_eq!(s[&x] + s[&y], 6);
    }
    #[test]
    fn test_backtrack_solve_in_domain() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(1, 10));
        e.add_constraint(CpeConstraint::InDomain(x, vec![3, 6, 9]));
        let sol = e.backtrack_solve();
        assert!(sol.is_some());
        let s = sol.expect("test: should succeed");
        assert!([3i64, 6, 9].contains(&s[&x]));
    }
    #[test]
    fn test_backtrack_solve_abs() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(-5, 5));
        let y = e.add_variable("y".to_string(), interval(0, 5));
        e.add_constraint(CpeConstraint::Abs(x, y));
        e.add_constraint(CpeConstraint::InDomain(y, vec![3]));
        let sol = e.backtrack_solve();
        assert!(sol.is_some());
        let s = sol.expect("test: should succeed");
        assert_eq!(s[&x].abs(), s[&y]);
    }
    #[test]
    fn test_config_default() {
        let cfg = CpeEngineConfig::default();
        assert_eq!(cfg.max_iterations, 10_000);
        assert!(cfg.use_bounds_propagation);
        assert!(cfg.fail_first);
        assert_eq!(cfg.arc_consistency_level, AcLevel::Ac3);
    }
    #[test]
    fn test_config_ac4() {
        let cfg = CpeEngineConfig {
            arc_consistency_level: AcLevel::Ac4,
            ..Default::default()
        };
        let mut e = ConstraintPropagationEngine::new(cfg);
        let x = e.add_variable("x".to_string(), finite(&[1, 2]));
        let y = e.add_variable("y".to_string(), finite(&[2, 3]));
        e.add_constraint(CpeConstraint::Equal(x, y));
        let r = e.propagate().expect("test: should succeed");
        assert_ne!(r, CpePropagationResult::Infeasible);
        let _ = (x, y);
    }
    #[test]
    fn test_config_ac6() {
        let cfg = CpeEngineConfig {
            arc_consistency_level: AcLevel::Ac6,
            ..Default::default()
        };
        let mut e = ConstraintPropagationEngine::new(cfg);
        let x = e.add_variable("x".to_string(), finite(&[5]));
        let y = e.add_variable("y".to_string(), interval(1, 10));
        e.add_constraint(CpeConstraint::LessEqual(x, y));
        let r = e.propagate().expect("test: should succeed");
        assert_ne!(r, CpePropagationResult::Infeasible);
        let _ = (x, y);
    }
    #[test]
    fn test_no_fail_first() {
        let cfg = CpeEngineConfig {
            fail_first: false,
            ..Default::default()
        };
        let mut e = ConstraintPropagationEngine::new(cfg);
        let x = e.add_variable("x".to_string(), finite(&[1, 2]));
        let y = e.add_variable("y".to_string(), finite(&[1, 2]));
        e.add_constraint(CpeConstraint::AllDifferent(vec![x, y]));
        let sol = e.backtrack_solve();
        assert!(sol.is_some());
    }
    #[test]
    fn test_constraint_variables_all_diff() {
        let c = CpeConstraint::AllDifferent(vec![0, 1, 2]);
        let v = c.variables();
        assert_eq!(v.len(), 3);
    }
    #[test]
    fn test_constraint_involves() {
        let c = CpeConstraint::Equal(0, 1);
        assert!(c.involves(0));
        assert!(c.involves(1));
        assert!(!c.involves(2));
    }
    #[test]
    fn test_constraint_not_equal_involves() {
        let c = CpeConstraint::NotEqual(3, 5);
        assert!(c.involves(3));
        assert!(c.involves(5));
        assert!(!c.involves(4));
    }
    #[test]
    fn test_constraint_sum_variables() {
        let c = CpeConstraint::Sum {
            vars: vec![0, 1, 2],
            total: 10,
        };
        let v = c.variables();
        assert!(v.contains(&0));
        assert!(v.contains(&1));
        assert!(v.contains(&2));
    }
    #[test]
    fn test_constraint_linear_variables() {
        let c = CpeConstraint::LinearExpr {
            coeffs: vec![(0, 2), (1, -1)],
            rhs: 5,
        };
        let v = c.variables();
        assert!(v.contains(&0));
        assert!(v.contains(&1));
    }
    #[test]
    fn test_constraint_in_domain_involves() {
        let c = CpeConstraint::InDomain(7, vec![1, 2, 3]);
        assert!(c.involves(7));
        assert!(!c.involves(0));
    }
    #[test]
    fn test_error_domain_empty_display() {
        let e = CpeError::DomainEmpty(5);
        let s = e.to_string();
        assert!(s.contains("5"));
    }
    #[test]
    fn test_error_value_not_in_domain() {
        let e = CpeError::ValueNotInDomain {
            var_id: 2,
            value: 99,
        };
        let s = e.to_string();
        assert!(s.contains("99"));
    }
    #[test]
    fn test_error_max_iterations() {
        let e = CpeError::MaxIterationsExceeded(42);
        let s = e.to_string();
        assert!(s.contains("42"));
    }
    #[test]
    fn test_type_alias_works() {
        let _e: CpeConstraintPropagationEngine = ConstraintPropagationEngine::default_engine();
    }
    #[test]
    fn test_bounds_propagation_less_than() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(1, 100));
        let y = e.add_variable("y".to_string(), interval(1, 10));
        e.add_constraint(CpeConstraint::LessThan(x, y));
        e.propagate().expect("test: should succeed");
        assert!(e.domain_size(x).expect("test: should succeed") <= 9);
    }
    #[test]
    fn test_bounds_propagation_sum() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(0, 10));
        let y = e.add_variable("y".to_string(), interval(0, 10));
        e.add_constraint(CpeConstraint::Sum {
            vars: vec![x, y],
            total: 5,
        });
        e.propagate().expect("test: should succeed");
        assert!(e.domain_size(x).expect("test: should succeed") <= 6);
        assert!(e.domain_size(y).expect("test: should succeed") <= 6);
    }
    #[test]
    fn test_bounds_propagation_linear_positive_coeff() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(0, 20));
        let y = e.add_variable("y".to_string(), interval(0, 20));
        e.add_constraint(CpeConstraint::LinearExpr {
            coeffs: vec![(x, 2), (y, 1)],
            rhs: 10,
        });
        e.propagate().expect("test: should succeed");
        assert!(e.domain_size(x).expect("test: should succeed") <= 6);
    }
    #[test]
    fn test_multiple_constraints_interact() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(1, 5));
        let y = e.add_variable("y".to_string(), interval(1, 5));
        let z = e.add_variable("z".to_string(), interval(1, 5));
        e.add_constraint(CpeConstraint::AllDifferent(vec![x, y, z]));
        e.add_constraint(CpeConstraint::Sum {
            vars: vec![x, y, z],
            total: 6,
        });
        let sol = e.backtrack_solve();
        if let Some(s) = sol {
            assert_eq!(s[&x] + s[&y] + s[&z], 6);
            let mut vals: Vec<i64> = vec![s[&x], s[&y], s[&z]];
            vals.sort_unstable();
            vals.dedup();
            assert_eq!(vals.len(), 3);
        }
    }
    #[test]
    fn test_propagation_result_is_solved_single_var() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), finite(&[42]));
        let r = e.propagate().expect("test: should succeed");
        assert_eq!(r, CpePropagationResult::Solved);
        let _ = x;
    }
    #[test]
    fn test_stats_values_removed() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(1, 5));
        let y = e.add_variable("y".to_string(), finite(&[3]));
        e.add_constraint(CpeConstraint::Equal(x, y));
        e.propagate().expect("test: should succeed");
        assert!(
            e.propagation_stats().values_removed > 0
                || e.propagation_stats().bounds_tightenings > 0
        );
        let _ = (x, y);
    }
    #[test]
    fn test_abs_constraint_y_nonneg() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(-10, 10));
        let y = e.add_variable("y".to_string(), interval(-5, 10));
        e.add_constraint(CpeConstraint::Abs(x, y));
        e.propagate().expect("test: should succeed");
        assert!(
            e.variables
                .get(&y)
                .expect("test: should succeed")
                .domain
                .min_val()
                .unwrap_or(-1)
                >= 0
        );
    }
    #[test]
    fn test_all_different_three_vars_infeasible() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), finite(&[1, 2]));
        let y = e.add_variable("y".to_string(), finite(&[1, 2]));
        let z = e.add_variable("z".to_string(), finite(&[1, 2]));
        e.add_constraint(CpeConstraint::AllDifferent(vec![x, y, z]));
        let sol = e.backtrack_solve();
        assert!(sol.is_none());
    }
    #[test]
    fn test_value_of_unassigned() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(1, 5));
        assert_eq!(e.value_of(x), None);
    }
    #[test]
    fn test_value_of_unknown() {
        let e = make_engine();
        assert_eq!(e.value_of(42), None);
    }
    #[test]
    fn test_xorshift64_not_zero_after_nonzero_seed() {
        let mut state: u64 = 12345;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }
    #[test]
    fn test_floor_ceil_div_positive() {
        assert_eq!(floor_div(7, 2), 3);
        assert_eq!(ceil_div(7, 2), 4);
    }
    #[test]
    fn test_floor_ceil_div_negative() {
        assert_eq!(floor_div(-7, 2), -4);
        assert_eq!(ceil_div(-7, 2), -3);
    }
    #[test]
    fn test_floor_div_exact() {
        assert_eq!(floor_div(6, 2), 3);
        assert_eq!(ceil_div(6, 2), 3);
    }
    #[test]
    fn test_backtrack_stats_incremented() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), finite(&[1, 2]));
        let y = e.add_variable("y".to_string(), finite(&[1, 2]));
        e.add_constraint(CpeConstraint::AllDifferent(vec![x, y]));
        e.backtrack_solve();
        assert!(e.propagation_stats().backtrack_nodes > 0);
    }
    #[test]
    fn test_engine_many_variables() {
        let mut e = make_engine();
        let vars: Vec<CpeVarId> = (0..10)
            .map(|i| e.add_variable(format!("x{i}"), interval(0, 9)))
            .collect();
        e.add_constraint(CpeConstraint::AllDifferent(vars.clone()));
        let sol = e.backtrack_solve();
        assert!(sol.is_some());
        let s = sol.expect("test: should succeed");
        let mut vals: Vec<i64> = vars.iter().map(|v| s[v]).collect();
        vals.sort_unstable();
        vals.dedup();
        assert_eq!(vals.len(), 10);
    }
    #[test]
    fn test_sum_infeasible() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(5, 10));
        let y = e.add_variable("y".to_string(), interval(5, 10));
        e.add_constraint(CpeConstraint::Sum {
            vars: vec![x, y],
            total: 3,
        });
        let sol = e.backtrack_solve();
        assert!(sol.is_none());
    }
    #[test]
    fn test_linear_infeasible() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(10, 20));
        e.add_constraint(CpeConstraint::LinearExpr {
            coeffs: vec![(x, 1)],
            rhs: 5,
        });
        let sol = e.backtrack_solve();
        assert!(sol.is_none());
    }
    #[test]
    fn test_boolean_domain_constraint() {
        let mut e = make_engine();
        let b = e.add_variable("b".to_string(), CpeDomain::Boolean);
        e.add_constraint(CpeConstraint::InDomain(b, vec![1]));
        e.propagate().expect("test: should succeed");
        assert_eq!(e.value_of(b), Some(1));
    }
    #[test]
    fn test_assign_marks_is_assigned() {
        let mut e = make_engine();
        let x = e.add_variable("x".to_string(), interval(1, 5));
        e.assign(x, 3).expect("test: should succeed");
        assert!(
            e.variables
                .get(&x)
                .expect("test: should succeed")
                .is_assigned
        );
    }
    #[test]
    fn test_propagation_stats_fields() {
        let stats = CpePropagationStats::default();
        assert_eq!(stats.arc_revisions, 0);
        assert_eq!(stats.values_removed, 0);
        assert_eq!(stats.passes, 0);
        assert_eq!(stats.total_propagations, 0);
        assert_eq!(stats.bounds_tightenings, 0);
        assert_eq!(stats.backtrack_nodes, 0);
    }
}
