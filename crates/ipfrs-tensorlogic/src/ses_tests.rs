//! Tests for symbolic_expression_simplifier — split out to keep the main
//! implementation file under 2000 lines.

use super::{
    SesError, SesExpr, SesSimplifierConfig, SesSymbolicExpressionSimplifier,
    SymbolicExpressionSimplifier,
};
use std::collections::HashMap;

fn simp() -> SymbolicExpressionSimplifier {
    SymbolicExpressionSimplifier::new()
}

// --- SesExpr constructors ---
#[test]
fn test_num_constructor() {
    let e = SesExpr::num(3.0);
    assert_eq!(e, SesExpr::Num(3.0));
}

#[test]
fn test_var_constructor() {
    let e = SesExpr::var("x");
    assert_eq!(e, SesExpr::Var("x".to_string()));
}

#[test]
fn test_add_constructor() {
    let e = SesExpr::add(SesExpr::num(1.0), SesExpr::num(2.0));
    assert!(matches!(e, SesExpr::Add(..)));
}

#[test]
fn test_mul_constructor() {
    let e = SesExpr::mul(SesExpr::num(3.0), SesExpr::var("x"));
    assert!(matches!(e, SesExpr::Mul(..)));
}

// --- Display ---
#[test]
fn test_display_num() {
    assert_eq!(SesExpr::num(5.0).to_string(), "5");
}

#[test]
fn test_display_var() {
    assert_eq!(SesExpr::var("y").to_string(), "y");
}

#[test]
fn test_display_add() {
    let e = SesExpr::add(SesExpr::var("x"), SesExpr::num(1.0));
    assert_eq!(e.to_string(), "x + 1");
}

#[test]
fn test_display_neg() {
    let e = SesExpr::neg(SesExpr::var("x"));
    assert_eq!(e.to_string(), "-x");
}

#[test]
fn test_display_sin() {
    let e = SesExpr::Sin(Box::new(SesExpr::var("x")));
    assert_eq!(e.to_string(), "sin(x)");
}

#[test]
fn test_display_abs() {
    let e = SesExpr::Abs(Box::new(SesExpr::var("x")));
    assert_eq!(e.to_string(), "|x|");
}

#[test]
fn test_display_pow() {
    let e = SesExpr::pow(SesExpr::var("x"), SesExpr::num(2.0));
    assert_eq!(e.to_string(), "x ^ 2");
}

// --- Evaluate ---
#[test]
fn test_eval_num() {
    let mut s = simp();
    let vars = HashMap::new();
    assert_eq!(
        s.evaluate(&SesExpr::num(7.0), &vars)
            .expect("test: should succeed"),
        7.0
    );
}

#[test]
fn test_eval_var() {
    let mut s = simp();
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);
    assert_eq!(
        s.evaluate(&SesExpr::var("x"), &vars)
            .expect("test: should succeed"),
        3.0
    );
}

#[test]
fn test_eval_unbound_var() {
    let mut s = simp();
    let vars = HashMap::new();
    let result = s.evaluate(&SesExpr::var("z"), &vars);
    assert!(matches!(result, Err(SesError::UnboundVariable(_))));
}

#[test]
fn test_eval_add() {
    let mut s = simp();
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.0);
    let e = SesExpr::add(SesExpr::var("x"), SesExpr::num(3.0));
    assert_eq!(
        s.evaluate(&e, &vars)
            .expect("test: symbolic expression evaluation should succeed"),
        5.0
    );
}

#[test]
fn test_eval_mul() {
    let mut s = simp();
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 4.0);
    let e = SesExpr::mul(SesExpr::var("x"), SesExpr::num(2.0));
    assert_eq!(
        s.evaluate(&e, &vars)
            .expect("test: symbolic expression evaluation should succeed"),
        8.0
    );
}

#[test]
fn test_eval_div_by_zero() {
    let mut s = simp();
    let vars = HashMap::new();
    let e = SesExpr::div(SesExpr::num(1.0), SesExpr::num(0.0));
    assert!(matches!(
        s.evaluate(&e, &vars),
        Err(SesError::DivisionByZero)
    ));
}

#[test]
fn test_eval_sqrt_negative() {
    let mut s = simp();
    let vars = HashMap::new();
    let e = SesExpr::Sqrt(Box::new(SesExpr::num(-1.0)));
    assert!(matches!(
        s.evaluate(&e, &vars),
        Err(SesError::MathDomainError(_))
    ));
}

#[test]
fn test_eval_ln_negative() {
    let mut s = simp();
    let vars = HashMap::new();
    let e = SesExpr::Ln(Box::new(SesExpr::num(-1.0)));
    assert!(matches!(
        s.evaluate(&e, &vars),
        Err(SesError::MathDomainError(_))
    ));
}

#[test]
fn test_eval_pow() {
    let mut s = simp();
    let vars = HashMap::new();
    let e = SesExpr::pow(SesExpr::num(2.0), SesExpr::num(10.0));
    assert!(
        (s.evaluate(&e, &vars)
            .expect("test: symbolic expression evaluation should succeed")
            - 1024.0)
            .abs()
            < 1e-9
    );
}

#[test]
fn test_eval_neg() {
    let mut s = simp();
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 5.0);
    let e = SesExpr::neg(SesExpr::var("x"));
    assert_eq!(
        s.evaluate(&e, &vars)
            .expect("test: symbolic expression evaluation should succeed"),
        -5.0
    );
}

#[test]
fn test_eval_sin_cos() {
    let mut s = simp();
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 0.0);
    let sin_e = SesExpr::Sin(Box::new(SesExpr::var("x")));
    let cos_e = SesExpr::Cos(Box::new(SesExpr::var("x")));
    assert!(
        (s.evaluate(&sin_e, &vars)
            .expect("test: symbolic expression evaluation should succeed")
            - 0.0)
            .abs()
            < 1e-10
    );
    assert!(
        (s.evaluate(&cos_e, &vars)
            .expect("test: symbolic expression evaluation should succeed")
            - 1.0)
            .abs()
            < 1e-10
    );
}

// --- Simplification rules ---
#[test]
fn test_simp_add_zero() {
    let mut s = simp();
    let e = SesExpr::add(SesExpr::var("x"), SesExpr::num(0.0));
    assert_eq!(s.simplify(&e), SesExpr::var("x"));
}

#[test]
fn test_simp_zero_add() {
    let mut s = simp();
    let e = SesExpr::add(SesExpr::num(0.0), SesExpr::var("y"));
    assert_eq!(s.simplify(&e), SesExpr::var("y"));
}

#[test]
fn test_simp_mul_one() {
    let mut s = simp();
    let e = SesExpr::mul(SesExpr::var("x"), SesExpr::num(1.0));
    assert_eq!(s.simplify(&e), SesExpr::var("x"));
}

#[test]
fn test_simp_one_mul() {
    let mut s = simp();
    let e = SesExpr::mul(SesExpr::num(1.0), SesExpr::var("y"));
    assert_eq!(s.simplify(&e), SesExpr::var("y"));
}

#[test]
fn test_simp_mul_zero() {
    let mut s = simp();
    let e = SesExpr::mul(SesExpr::var("x"), SesExpr::num(0.0));
    assert_eq!(s.simplify(&e), SesExpr::num(0.0));
}

#[test]
fn test_simp_zero_mul() {
    let mut s = simp();
    let e = SesExpr::mul(SesExpr::num(0.0), SesExpr::var("y"));
    assert_eq!(s.simplify(&e), SesExpr::num(0.0));
}

#[test]
fn test_simp_pow_one() {
    let mut s = simp();
    let e = SesExpr::pow(SesExpr::var("x"), SesExpr::num(1.0));
    assert_eq!(s.simplify(&e), SesExpr::var("x"));
}

#[test]
fn test_simp_pow_zero() {
    let mut s = simp();
    let e = SesExpr::pow(SesExpr::var("x"), SesExpr::num(0.0));
    assert_eq!(s.simplify(&e), SesExpr::num(1.0));
}

#[test]
fn test_simp_double_neg() {
    let mut s = simp();
    let e = SesExpr::neg(SesExpr::neg(SesExpr::var("x")));
    assert_eq!(s.simplify(&e), SesExpr::var("x"));
}

#[test]
fn test_simp_sub_zero() {
    let mut s = simp();
    let e = SesExpr::sub(SesExpr::var("x"), SesExpr::num(0.0));
    assert_eq!(s.simplify(&e), SesExpr::var("x"));
}

#[test]
fn test_simp_sub_self() {
    let mut s = simp();
    let e = SesExpr::sub(SesExpr::var("x"), SesExpr::var("x"));
    assert_eq!(s.simplify(&e), SesExpr::num(0.0));
}

#[test]
fn test_simp_div_one() {
    let mut s = simp();
    let e = SesExpr::div(SesExpr::var("x"), SesExpr::num(1.0));
    assert_eq!(s.simplify(&e), SesExpr::var("x"));
}

#[test]
fn test_simp_div_self() {
    let mut s = simp();
    let e = SesExpr::div(SesExpr::var("x"), SesExpr::var("x"));
    assert_eq!(s.simplify(&e), SesExpr::num(1.0));
}

#[test]
fn test_simp_constant_folding() {
    let mut s = simp();
    let e = SesExpr::add(SesExpr::num(3.0), SesExpr::num(4.0));
    assert_eq!(s.simplify(&e), SesExpr::num(7.0));
}

#[test]
fn test_simp_nested_constant() {
    let mut s = simp();
    let e = SesExpr::mul(
        SesExpr::add(SesExpr::num(2.0), SesExpr::num(3.0)),
        SesExpr::num(4.0),
    );
    assert_eq!(s.simplify(&e), SesExpr::num(20.0));
}

#[test]
fn test_simp_exp_zero() {
    let mut s = simp();
    let e = SesExpr::Exp(Box::new(SesExpr::num(0.0)));
    assert_eq!(s.simplify(&e), SesExpr::num(1.0));
}

#[test]
fn test_simp_ln_one() {
    let mut s = simp();
    let e = SesExpr::Ln(Box::new(SesExpr::num(1.0)));
    assert_eq!(s.simplify(&e), SesExpr::num(0.0));
}

#[test]
fn test_simp_exp_ln() {
    let mut s = simp();
    let e = SesExpr::Exp(Box::new(SesExpr::Ln(Box::new(SesExpr::var("x")))));
    assert_eq!(s.simplify(&e), SesExpr::var("x"));
}

#[test]
fn test_simp_ln_exp() {
    let mut s = simp();
    let e = SesExpr::Ln(Box::new(SesExpr::Exp(Box::new(SesExpr::var("x")))));
    assert_eq!(s.simplify(&e), SesExpr::var("x"));
}

#[test]
fn test_simp_sqrt_zero() {
    let mut s = simp();
    let e = SesExpr::Sqrt(Box::new(SesExpr::num(0.0)));
    assert_eq!(s.simplify(&e), SesExpr::num(0.0));
}

#[test]
fn test_simp_sqrt_one() {
    let mut s = simp();
    let e = SesExpr::Sqrt(Box::new(SesExpr::num(1.0)));
    assert_eq!(s.simplify(&e), SesExpr::num(1.0));
}

#[test]
fn test_simp_sqrt_sq() {
    let mut s = simp();
    let e = SesExpr::Sqrt(Box::new(SesExpr::pow(SesExpr::var("x"), SesExpr::num(2.0))));
    assert_eq!(s.simplify(&e), SesExpr::Abs(Box::new(SesExpr::var("x"))));
}

#[test]
fn test_simp_neg_zero() {
    let mut s = simp();
    let e = SesExpr::neg(SesExpr::num(0.0));
    assert_eq!(s.simplify(&e), SesExpr::num(0.0));
}

// --- sin²+cos² trig identity ---
#[test]
fn test_simp_sin2_cos2() {
    let mut s = simp();
    let sin2 = SesExpr::pow(SesExpr::Sin(Box::new(SesExpr::var("x"))), SesExpr::num(2.0));
    let cos2 = SesExpr::pow(SesExpr::Cos(Box::new(SesExpr::var("x"))), SesExpr::num(2.0));
    let e = SesExpr::add(sin2, cos2);
    assert_eq!(s.simplify(&e), SesExpr::num(1.0));
}

#[test]
fn test_simp_cos2_sin2() {
    let mut s = simp();
    let sin2 = SesExpr::pow(SesExpr::Sin(Box::new(SesExpr::var("t"))), SesExpr::num(2.0));
    let cos2 = SesExpr::pow(SesExpr::Cos(Box::new(SesExpr::var("t"))), SesExpr::num(2.0));
    let e = SesExpr::add(cos2, sin2); // reversed order
    assert_eq!(s.simplify(&e), SesExpr::num(1.0));
}

// --- Differentiation ---
#[test]
fn test_diff_const() {
    let mut s = simp();
    let e = SesExpr::num(5.0);
    let d = s.differentiate(&e, "x");
    assert_eq!(d, SesExpr::num(0.0));
}

#[test]
fn test_diff_var_same() {
    let mut s = simp();
    let e = SesExpr::var("x");
    let d = s.differentiate(&e, "x");
    assert_eq!(d, SesExpr::num(1.0));
}

#[test]
fn test_diff_var_other() {
    let mut s = simp();
    let e = SesExpr::var("y");
    let d = s.differentiate(&e, "x");
    assert_eq!(d, SesExpr::num(0.0));
}

#[test]
fn test_diff_add() {
    let mut s = simp();
    let e = SesExpr::add(SesExpr::var("x"), SesExpr::var("x"));
    let d = s.differentiate(&e, "x");
    let d_simp = s.simplify(&d);
    assert_eq!(d_simp, SesExpr::num(2.0));
}

#[test]
fn test_diff_mul_product_rule() {
    let mut s = simp();
    let e = SesExpr::mul(SesExpr::var("x"), SesExpr::var("x"));
    let d = s.differentiate(&e, "x");
    let d_simp = s.simplify(&d);
    let vars: HashMap<String, f64> = [("x".to_string(), 3.0)].into();
    assert!(
        (s.evaluate(&d_simp, &vars)
            .expect("test: symbolic expression evaluation should succeed")
            - 6.0)
            .abs()
            < 1e-9
    );
}

#[test]
fn test_diff_sin() {
    let mut s = simp();
    let e = SesExpr::Sin(Box::new(SesExpr::var("x")));
    let d = s.differentiate(&e, "x");
    let d_simp = s.simplify(&d);
    let vars: HashMap<String, f64> = [("x".to_string(), 0.0)].into();
    assert!(
        (s.evaluate(&d_simp, &vars)
            .expect("test: symbolic expression evaluation should succeed")
            - 1.0)
            .abs()
            < 1e-9
    );
}

#[test]
fn test_diff_cos() {
    let mut s = simp();
    let e = SesExpr::Cos(Box::new(SesExpr::var("x")));
    let d = s.differentiate(&e, "x");
    let vars: HashMap<String, f64> = [("x".to_string(), 0.0)].into();
    assert!(
        (s.evaluate(&d, &vars)
            .expect("test: symbolic expression evaluation should succeed")
            - 0.0)
            .abs()
            < 1e-9
    );
}

#[test]
fn test_diff_exp() {
    let mut s = simp();
    let e = SesExpr::Exp(Box::new(SesExpr::var("x")));
    let d = s.differentiate(&e, "x");
    let d_simp = s.simplify(&d);
    let vars: HashMap<String, f64> = [("x".to_string(), 0.0)].into();
    assert!(
        (s.evaluate(&d_simp, &vars)
            .expect("test: symbolic expression evaluation should succeed")
            - 1.0)
            .abs()
            < 1e-9
    );
}

#[test]
fn test_diff_ln() {
    let mut s = simp();
    let e = SesExpr::Ln(Box::new(SesExpr::var("x")));
    let d = s.differentiate(&e, "x");
    let d_simp = s.simplify(&d);
    let vars: HashMap<String, f64> = [("x".to_string(), 2.0)].into();
    assert!(
        (s.evaluate(&d_simp, &vars)
            .expect("test: symbolic expression evaluation should succeed")
            - 0.5)
            .abs()
            < 1e-9
    );
}

#[test]
fn test_diff_pow_const_exp() {
    let mut s = simp();
    let e = SesExpr::pow(SesExpr::var("x"), SesExpr::num(3.0));
    let d = s.differentiate(&e, "x");
    let d_simp = s.simplify(&d);
    let vars: HashMap<String, f64> = [("x".to_string(), 2.0)].into();
    assert!(
        (s.evaluate(&d_simp, &vars)
            .expect("test: symbolic expression evaluation should succeed")
            - 12.0)
            .abs()
            < 1e-9
    );
}

// --- Substitution ---
#[test]
fn test_substitute_var() {
    let s = simp();
    let e = SesExpr::var("x");
    let result = s.substitute(&e, "x", &SesExpr::num(5.0));
    assert_eq!(result, SesExpr::num(5.0));
}

#[test]
fn test_substitute_no_match() {
    let s = simp();
    let e = SesExpr::var("y");
    let result = s.substitute(&e, "x", &SesExpr::num(5.0));
    assert_eq!(result, SesExpr::var("y"));
}

#[test]
fn test_substitute_nested() {
    let s = simp();
    let e = SesExpr::add(
        SesExpr::var("x"),
        SesExpr::mul(SesExpr::var("x"), SesExpr::num(2.0)),
    );
    let result = s.substitute(&e, "x", &SesExpr::num(3.0));
    let mut simp_inst = simp();
    let vars = HashMap::new();
    let v = simp_inst
        .evaluate(&result, &vars)
        .expect("test: symbolic expression evaluation should succeed");
    assert!((v - 9.0).abs() < 1e-9);
}

// --- Variable collection ---
#[test]
fn test_collect_vars() {
    let s = simp();
    let e = SesExpr::add(
        SesExpr::var("x"),
        SesExpr::mul(SesExpr::var("y"), SesExpr::var("x")),
    );
    let vars = s.collect_vars(&e);
    assert!(vars.contains("x"));
    assert!(vars.contains("y"));
    assert_eq!(vars.len(), 2);
}

#[test]
fn test_collect_vars_empty() {
    let s = simp();
    let e = SesExpr::num(42.0);
    assert!(s.collect_vars(&e).is_empty());
}

#[test]
fn test_contains_var_true() {
    let s = simp();
    let e = SesExpr::add(SesExpr::var("x"), SesExpr::num(1.0));
    assert!(s.contains_var(&e, "x"));
}

#[test]
fn test_contains_var_false() {
    let s = simp();
    let e = SesExpr::add(SesExpr::var("y"), SesExpr::num(1.0));
    assert!(!s.contains_var(&e, "x"));
}

// --- Parser ---
#[test]
fn test_parse_num() {
    let e = SymbolicExpressionSimplifier::parse("3.14159265358979")
        .expect("test: expression parsing should succeed");
    assert!(matches!(e, SesExpr::Num(v) if (v - std::f64::consts::PI).abs() < 1e-10));
}

#[test]
fn test_parse_var() {
    let e = SymbolicExpressionSimplifier::parse("xyz")
        .expect("test: expression parsing should succeed");
    assert_eq!(e, SesExpr::Var("xyz".to_string()));
}

#[test]
fn test_parse_add() {
    let e = SymbolicExpressionSimplifier::parse("x + 1")
        .expect("test: expression parsing should succeed");
    assert!(matches!(e, SesExpr::Add(..)));
}

#[test]
fn test_parse_sub() {
    let e = SymbolicExpressionSimplifier::parse("x - y")
        .expect("test: expression parsing should succeed");
    assert!(matches!(e, SesExpr::Sub(..)));
}

#[test]
fn test_parse_mul() {
    let e = SymbolicExpressionSimplifier::parse("x * 2")
        .expect("test: expression parsing should succeed");
    assert!(matches!(e, SesExpr::Mul(..)));
}

#[test]
fn test_parse_div() {
    let e = SymbolicExpressionSimplifier::parse("x / 2")
        .expect("test: expression parsing should succeed");
    assert!(matches!(e, SesExpr::Div(..)));
}

#[test]
fn test_parse_pow() {
    let e = SymbolicExpressionSimplifier::parse("x ^ 3")
        .expect("test: expression parsing should succeed");
    assert!(matches!(e, SesExpr::Pow(..)));
}

#[test]
fn test_parse_unary_neg() {
    let e =
        SymbolicExpressionSimplifier::parse("-x").expect("test: expression parsing should succeed");
    assert!(matches!(e, SesExpr::Neg(..)));
}

#[test]
fn test_parse_paren() {
    let e = SymbolicExpressionSimplifier::parse("(x + 1) * 2").expect("test: should succeed");
    assert!(matches!(e, SesExpr::Mul(..)));
}

#[test]
fn test_parse_abs() {
    let e = SymbolicExpressionSimplifier::parse("|x|")
        .expect("test: expression parsing should succeed");
    assert!(matches!(e, SesExpr::Abs(..)));
}

#[test]
fn test_parse_sin() {
    let e = SymbolicExpressionSimplifier::parse("sin(x)").expect("test: should succeed");
    assert!(matches!(e, SesExpr::Sin(..)));
}

#[test]
fn test_parse_cos() {
    let e = SymbolicExpressionSimplifier::parse("cos(x)").expect("test: should succeed");
    assert!(matches!(e, SesExpr::Cos(..)));
}

#[test]
fn test_parse_exp() {
    let e = SymbolicExpressionSimplifier::parse("exp(x)").expect("test: should succeed");
    assert!(matches!(e, SesExpr::Exp(..)));
}

#[test]
fn test_parse_ln() {
    let e = SymbolicExpressionSimplifier::parse("ln(x)").expect("test: should succeed");
    assert!(matches!(e, SesExpr::Ln(..)));
}

#[test]
fn test_parse_sqrt() {
    let e = SymbolicExpressionSimplifier::parse("sqrt(x)").expect("test: should succeed");
    assert!(matches!(e, SesExpr::Sqrt(..)));
}

#[test]
fn test_parse_complex_expr() {
    let e = SymbolicExpressionSimplifier::parse("x ^ 2 + 2 * x + 1")
        .expect("test: expression parsing should succeed");
    let mut s = simp();
    let vars: HashMap<String, f64> = [("x".to_string(), 3.0)].into();
    let v = s
        .evaluate(&e, &vars)
        .expect("test: symbolic expression evaluation should succeed");
    assert!((v - 16.0).abs() < 1e-9);
}

#[test]
fn test_parse_error_unknown_func() {
    let r = SymbolicExpressionSimplifier::parse("foo(x)");
    assert!(matches!(r, Err(SesError::ParseError(_))));
}

#[test]
fn test_parse_error_unexpected_char() {
    let r = SymbolicExpressionSimplifier::parse("x @ y");
    assert!(matches!(r, Err(SesError::ParseError(_))));
}

// --- Stats ---
#[test]
fn test_stats_after_simplify() {
    let mut s = simp();
    let e = SesExpr::add(SesExpr::var("x"), SesExpr::num(0.0));
    s.simplify(&e);
    let stats = s.simplifier_stats();
    assert!(stats.simplify_calls >= 1);
}

#[test]
fn test_stats_evaluate_calls() {
    let mut s = simp();
    let vars = HashMap::new();
    let _ = s.evaluate(&SesExpr::num(1.0), &vars);
    let _ = s.evaluate(&SesExpr::num(2.0), &vars);
    assert_eq!(s.simplifier_stats().evaluate_calls, 2);
}

#[test]
fn test_stats_differentiate_calls() {
    let mut s = simp();
    s.differentiate(&SesExpr::var("x"), "x");
    assert_eq!(s.simplifier_stats().differentiate_calls, 1);
}

// --- History ---
#[test]
fn test_history_recorded() {
    let mut s = simp();
    let e = SesExpr::add(SesExpr::var("x"), SesExpr::num(0.0));
    s.simplify(&e);
    assert!(!s.history().is_empty());
}

#[test]
fn test_history_clear() {
    let mut s = simp();
    let e = SesExpr::add(SesExpr::var("x"), SesExpr::num(0.0));
    s.simplify(&e);
    s.clear_history();
    assert!(s.history().is_empty());
}

// --- Config ---
#[test]
fn test_custom_config() {
    let config = SesSimplifierConfig {
        max_passes: 5,
        enable_constant_folding: false,
        enable_algebraic_rules: true,
        enable_trig_rules: false,
    };
    let mut s = SymbolicExpressionSimplifier::with_config(config);
    let e = SesExpr::add(SesExpr::var("x"), SesExpr::num(0.0));
    assert_eq!(s.simplify(&e), SesExpr::var("x"));
}

#[test]
fn test_rule_count() {
    let s = simp();
    assert!(s.rule_count() >= 20);
}

// --- Type alias ---
#[test]
fn test_type_alias() {
    let _: SesSymbolicExpressionSimplifier = SymbolicExpressionSimplifier::new();
}

// --- Error display ---
#[test]
fn test_error_display() {
    let e = SesError::UnboundVariable("z".to_string());
    assert!(e.to_string().contains("z"));
    let e2 = SesError::DivisionByZero;
    assert!(e2.to_string().contains("zero"));
}

// --- zero_sub rule ---
#[test]
fn test_simp_zero_sub() {
    let mut s = simp();
    let e = SesExpr::sub(SesExpr::num(0.0), SesExpr::var("x"));
    let result = s.simplify(&e);
    assert_eq!(result, SesExpr::Neg(Box::new(SesExpr::var("x"))));
}

// --- mul_self_to_pow2 ---
#[test]
fn test_simp_mul_self_to_pow2() {
    let mut s = simp();
    let e = SesExpr::mul(SesExpr::var("x"), SesExpr::var("x"));
    let result = s.simplify(&e);
    assert_eq!(result, SesExpr::pow(SesExpr::var("x"), SesExpr::num(2.0)));
}

// --- add_self ---
#[test]
fn test_simp_add_self() {
    let mut s = simp();
    let e = SesExpr::add(SesExpr::var("x"), SesExpr::var("x"));
    let result = s.simplify(&e);
    assert!(matches!(result, SesExpr::Mul(..)));
}

// --- Parse and evaluate roundtrip ---
#[test]
fn test_parse_eval_roundtrip() {
    let e =
        SymbolicExpressionSimplifier::parse("sin(x)^2 + cos(x)^2").expect("test: should succeed");
    let mut s = simp();
    let simplified = s.simplify(&e);
    assert_eq!(simplified, SesExpr::num(1.0));
}

// --- Fixpoint statistics ---
#[test]
fn test_fixpoint_stat() {
    let mut s = simp();
    let e = SesExpr::var("x");
    s.simplify(&e);
    assert!(s.simplifier_stats().fixpoint_reached >= 1);
}

// --- one_pow ---
#[test]
fn test_simp_one_pow() {
    let mut s = simp();
    let e = SesExpr::pow(SesExpr::num(1.0), SesExpr::var("x"));
    assert_eq!(s.simplify(&e), SesExpr::num(1.0));
}

// --- abs eval ---
#[test]
fn test_eval_abs() {
    let mut s = simp();
    let e = SesExpr::Abs(Box::new(SesExpr::num(-3.0)));
    let vars = HashMap::new();
    assert_eq!(
        s.evaluate(&e, &vars)
            .expect("test: symbolic expression evaluation should succeed"),
        3.0
    );
}

// --- display for complex ---
#[test]
fn test_display_ln() {
    let e = SesExpr::Ln(Box::new(SesExpr::var("x")));
    assert_eq!(e.to_string(), "ln(x)");
}

#[test]
fn test_display_exp() {
    let e = SesExpr::Exp(Box::new(SesExpr::var("x")));
    assert_eq!(e.to_string(), "exp(x)");
}

#[test]
fn test_display_sqrt() {
    let e = SesExpr::Sqrt(Box::new(SesExpr::var("x")));
    assert_eq!(e.to_string(), "sqrt(x)");
}

// --- SesSimplificationStep fields ---
#[test]
fn test_step_fields() {
    let mut s = simp();
    let e = SesExpr::add(SesExpr::var("x"), SesExpr::num(0.0));
    s.simplify(&e);
    let h = s.history();
    if let Some(step) = h.front() {
        assert!(!step.rule_applied.is_empty());
        assert!(!step.before.is_empty());
        assert!(!step.after.is_empty());
    }
}

// --- zero_div ---
#[test]
fn test_simp_zero_div() {
    let mut s = simp();
    let e = SesExpr::div(SesExpr::num(0.0), SesExpr::var("x"));
    assert_eq!(s.simplify(&e), SesExpr::num(0.0));
}
