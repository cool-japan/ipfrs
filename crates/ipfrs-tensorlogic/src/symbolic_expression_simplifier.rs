//! Symbolic Expression Simplifier — multi-pass rewriting engine for symbolic math expressions.
//!
//! Provides:
//! - A rich expression type (`SesExpr`) covering arithmetic, power, trig, exp/ln
//! - Multi-pass fixpoint simplification with configurable rule sets
//! - Symbolic differentiation and substitution
//! - A recursive-descent parser for infix expressions
//! - Full numeric evaluation with variable bindings
//! - Simplification history tracking (bounded VecDeque)

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during symbolic expression operations.
#[derive(Debug, Clone, PartialEq)]
pub enum SesError {
    /// A variable was not found in the provided bindings.
    UnboundVariable(String),
    /// A numeric operation produced an invalid result (e.g. ln of negative).
    MathDomainError(String),
    /// A parse error with a descriptive message.
    ParseError(String),
    /// Division by zero attempted during evaluation.
    DivisionByZero,
}

impl fmt::Display for SesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SesError::UnboundVariable(v) => write!(f, "unbound variable: {v}"),
            SesError::MathDomainError(m) => write!(f, "math domain error: {m}"),
            SesError::ParseError(m) => write!(f, "parse error: {m}"),
            SesError::DivisionByZero => write!(f, "division by zero"),
        }
    }
}

impl std::error::Error for SesError {}

// ---------------------------------------------------------------------------
// Core expression type
// ---------------------------------------------------------------------------

/// A symbolic mathematical expression.
#[derive(Debug, Clone, PartialEq)]
pub enum SesExpr {
    /// A numeric literal.
    Num(f64),
    /// A named variable.
    Var(String),
    /// Addition: left + right.
    Add(Box<SesExpr>, Box<SesExpr>),
    /// Subtraction: left - right.
    Sub(Box<SesExpr>, Box<SesExpr>),
    /// Multiplication: left * right.
    Mul(Box<SesExpr>, Box<SesExpr>),
    /// Division: left / right.
    Div(Box<SesExpr>, Box<SesExpr>),
    /// Exponentiation: base ^ exponent.
    Pow(Box<SesExpr>, Box<SesExpr>),
    /// Negation: -expr.
    Neg(Box<SesExpr>),
    /// Absolute value: |expr|.
    Abs(Box<SesExpr>),
    /// Square root: √expr.
    Sqrt(Box<SesExpr>),
    /// Sine: sin(expr).
    Sin(Box<SesExpr>),
    /// Cosine: cos(expr).
    Cos(Box<SesExpr>),
    /// Natural exponential: e^expr.
    Exp(Box<SesExpr>),
    /// Natural logarithm: ln(expr).
    Ln(Box<SesExpr>),
}

impl SesExpr {
    /// Convenience constructor for a boxed `Num`.
    #[inline]
    pub fn num(v: f64) -> Self {
        SesExpr::Num(v)
    }
    /// Convenience constructor for `Var`.
    #[inline]
    pub fn var(name: impl Into<String>) -> Self {
        SesExpr::Var(name.into())
    }
    /// Convenience constructor for `Add`.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn add(l: SesExpr, r: SesExpr) -> Self {
        SesExpr::Add(Box::new(l), Box::new(r))
    }
    /// Convenience constructor for `Sub`.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn sub(l: SesExpr, r: SesExpr) -> Self {
        SesExpr::Sub(Box::new(l), Box::new(r))
    }
    /// Convenience constructor for `Mul`.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn mul(l: SesExpr, r: SesExpr) -> Self {
        SesExpr::Mul(Box::new(l), Box::new(r))
    }
    /// Convenience constructor for `Div`.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn div(l: SesExpr, r: SesExpr) -> Self {
        SesExpr::Div(Box::new(l), Box::new(r))
    }
    /// Convenience constructor for `Pow`.
    #[inline]
    pub fn pow(b: SesExpr, e: SesExpr) -> Self {
        SesExpr::Pow(Box::new(b), Box::new(e))
    }
    /// Convenience constructor for `Neg`.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn neg(e: SesExpr) -> Self {
        SesExpr::Neg(Box::new(e))
    }
}

// ---------------------------------------------------------------------------
// Display / infix notation
// ---------------------------------------------------------------------------

impl fmt::Display for SesExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", infix_str(self, Precedence::Lowest))
    }
}

/// Internal precedence levels for minimal-parentheses infix printing.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Precedence {
    Lowest,
    AddSub,
    MulDiv,
    Unary,
    Pow,
}

fn infix_str(e: &SesExpr, parent_prec: Precedence) -> String {
    match e {
        SesExpr::Num(v) => {
            if v.fract() == 0.0 && v.abs() < 1e15 {
                format!("{}", *v as i64)
            } else {
                format!("{v}")
            }
        }
        SesExpr::Var(name) => name.clone(),
        SesExpr::Add(l, r) => paren_if(
            &format!(
                "{} + {}",
                infix_str(l, Precedence::AddSub),
                infix_str(r, Precedence::AddSub)
            ),
            Precedence::AddSub,
            parent_prec,
        ),
        SesExpr::Sub(l, r) => paren_if(
            &format!(
                "{} - {}",
                infix_str(l, Precedence::AddSub),
                infix_str(r, Precedence::MulDiv)
            ),
            Precedence::AddSub,
            parent_prec,
        ),
        SesExpr::Mul(l, r) => paren_if(
            &format!(
                "{} * {}",
                infix_str(l, Precedence::MulDiv),
                infix_str(r, Precedence::MulDiv)
            ),
            Precedence::MulDiv,
            parent_prec,
        ),
        SesExpr::Div(l, r) => paren_if(
            &format!(
                "{} / {}",
                infix_str(l, Precedence::MulDiv),
                infix_str(r, Precedence::Unary)
            ),
            Precedence::MulDiv,
            parent_prec,
        ),
        SesExpr::Pow(b, exp) => paren_if(
            &format!(
                "{} ^ {}",
                infix_str(b, Precedence::Unary),
                infix_str(exp, Precedence::Pow)
            ),
            Precedence::Pow,
            parent_prec,
        ),
        SesExpr::Neg(inner) => paren_if(
            &format!("-{}", infix_str(inner, Precedence::Unary)),
            Precedence::Unary,
            parent_prec,
        ),
        SesExpr::Abs(inner) => format!("|{}|", infix_str(inner, Precedence::Lowest)),
        SesExpr::Sqrt(inner) => format!("sqrt({})", infix_str(inner, Precedence::Lowest)),
        SesExpr::Sin(inner) => format!("sin({})", infix_str(inner, Precedence::Lowest)),
        SesExpr::Cos(inner) => format!("cos({})", infix_str(inner, Precedence::Lowest)),
        SesExpr::Exp(inner) => format!("exp({})", infix_str(inner, Precedence::Lowest)),
        SesExpr::Ln(inner) => format!("ln({})", infix_str(inner, Precedence::Lowest)),
    }
}

fn paren_if(s: &str, my_prec: Precedence, parent_prec: Precedence) -> String {
    if my_prec < parent_prec {
        format!("({s})")
    } else {
        s.to_string()
    }
}

// ---------------------------------------------------------------------------
// Simplifier configuration
// ---------------------------------------------------------------------------

/// Configuration for the `SymbolicExpressionSimplifier`.
#[derive(Debug, Clone)]
pub struct SesSimplifierConfig {
    /// Maximum number of simplification passes before giving up.
    pub max_passes: usize,
    /// Enable constant folding (evaluate sub-expressions with no variables).
    pub enable_constant_folding: bool,
    /// Enable algebraic identity rules (e.g. x+0, x*1).
    pub enable_algebraic_rules: bool,
    /// Enable trigonometric identities (e.g. sin²+cos²=1).
    pub enable_trig_rules: bool,
}

impl Default for SesSimplifierConfig {
    fn default() -> Self {
        SesSimplifierConfig {
            max_passes: 32,
            enable_constant_folding: true,
            enable_algebraic_rules: true,
            enable_trig_rules: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Rewrite rule
// ---------------------------------------------------------------------------

/// A named rewrite rule for the simplifier.
pub struct SesRewriteRule {
    /// Human-readable name for the rule.
    pub name: String,
    /// A short pattern description string (informational only).
    pub pattern: String,
    /// Predicate: returns `true` when this rule can be applied to `expr`.
    pub applies: fn(&SesExpr) -> bool,
    /// Transformation: consumes the expression and returns the simplified form.
    pub transform: Box<dyn Fn(SesExpr) -> SesExpr + Send + Sync>,
}

impl fmt::Debug for SesRewriteRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SesRewriteRule")
            .field("name", &self.name)
            .field("pattern", &self.pattern)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Simplification step (history entry)
// ---------------------------------------------------------------------------

/// A single recorded simplification step.
#[derive(Debug, Clone)]
pub struct SesSimplificationStep {
    /// Which pass (0-indexed) this step occurred on.
    pub pass: usize,
    /// Name of the rule that was applied.
    pub rule_applied: String,
    /// Infix string of the expression *before* the rewrite.
    pub before: String,
    /// Infix string of the expression *after* the rewrite.
    pub after: String,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Summary statistics for the simplifier.
#[derive(Debug, Clone, Default)]
pub struct SesSimplifierStats {
    /// Total number of `simplify` calls made.
    pub simplify_calls: usize,
    /// Total simplification passes executed across all calls.
    pub total_passes: usize,
    /// Total number of individual rule applications.
    pub total_rule_applications: usize,
    /// Number of simplifications that reached fixpoint before max_passes.
    pub fixpoint_reached: usize,
    /// Total `evaluate` calls.
    pub evaluate_calls: usize,
    /// Total `differentiate` calls.
    pub differentiate_calls: usize,
    /// History entries currently stored.
    pub history_len: usize,
}

// ---------------------------------------------------------------------------
// Type aliases (as required by the task)
// ---------------------------------------------------------------------------

/// Alias for `SymbolicExpressionSimplifier`.
pub type SesSymbolicExpressionSimplifier = SymbolicExpressionSimplifier;

// ---------------------------------------------------------------------------
// Main struct
// ---------------------------------------------------------------------------

/// A multi-pass symbolic expression simplifier with rewriting rules and
/// canonical forms.
pub struct SymbolicExpressionSimplifier {
    /// Ordered set of rewrite rules.
    rules: Vec<SesRewriteRule>,
    /// Bounded history of simplification steps (max 500 entries).
    history: VecDeque<SesSimplificationStep>,
    /// Configuration.
    config: SesSimplifierConfig,
    /// Accumulated statistics.
    stats: SesSimplifierStats,
}

impl SymbolicExpressionSimplifier {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new simplifier with the default configuration and built-in rules.
    pub fn new() -> Self {
        let config = SesSimplifierConfig::default();
        Self::with_config(config)
    }

    /// Create a new simplifier with a custom configuration.
    pub fn with_config(config: SesSimplifierConfig) -> Self {
        let mut s = SymbolicExpressionSimplifier {
            rules: Vec::new(),
            history: VecDeque::with_capacity(500),
            config,
            stats: SesSimplifierStats::default(),
        };
        s.register_builtin_rules();
        s
    }

    /// Add a custom rewrite rule.
    pub fn add_rule(&mut self, rule: SesRewriteRule) {
        self.rules.push(rule);
    }

    // -----------------------------------------------------------------------
    // Built-in rules
    // -----------------------------------------------------------------------

    fn register_builtin_rules(&mut self) {
        // x + 0 → x
        self.rules.push(SesRewriteRule {
            name: "add_zero_right".into(),
            pattern: "x + 0 => x".into(),
            applies: |e| matches!(e, SesExpr::Add(_, r) if is_zero(r)),
            transform: Box::new(|e| {
                if let SesExpr::Add(l, r) = e {
                    if is_zero(&r) {
                        return *l;
                    }
                    SesExpr::Add(l, r)
                } else {
                    e
                }
            }),
        });
        // 0 + x → x
        self.rules.push(SesRewriteRule {
            name: "add_zero_left".into(),
            pattern: "0 + x => x".into(),
            applies: |e| matches!(e, SesExpr::Add(l, _) if is_zero(l)),
            transform: Box::new(|e| {
                if let SesExpr::Add(l, r) = e {
                    if is_zero(&l) {
                        return *r;
                    }
                    SesExpr::Add(l, r)
                } else {
                    e
                }
            }),
        });
        // x - 0 → x
        self.rules.push(SesRewriteRule {
            name: "sub_zero".into(),
            pattern: "x - 0 => x".into(),
            applies: |e| matches!(e, SesExpr::Sub(_, r) if is_zero(r)),
            transform: Box::new(|e| {
                if let SesExpr::Sub(l, r) = e {
                    if is_zero(&r) {
                        return *l;
                    }
                    SesExpr::Sub(l, r)
                } else {
                    e
                }
            }),
        });
        // 0 - x → -x
        self.rules.push(SesRewriteRule {
            name: "zero_sub".into(),
            pattern: "0 - x => -x".into(),
            applies: |e| matches!(e, SesExpr::Sub(l, _) if is_zero(l)),
            transform: Box::new(|e| {
                if let SesExpr::Sub(l, r) = e {
                    if is_zero(&l) {
                        return SesExpr::Neg(r);
                    }
                    SesExpr::Sub(l, r)
                } else {
                    e
                }
            }),
        });
        // x * 1 → x
        self.rules.push(SesRewriteRule {
            name: "mul_one_right".into(),
            pattern: "x * 1 => x".into(),
            applies: |e| matches!(e, SesExpr::Mul(_, r) if is_one(r)),
            transform: Box::new(|e| {
                if let SesExpr::Mul(l, r) = e {
                    if is_one(&r) {
                        return *l;
                    }
                    SesExpr::Mul(l, r)
                } else {
                    e
                }
            }),
        });
        // 1 * x → x
        self.rules.push(SesRewriteRule {
            name: "mul_one_left".into(),
            pattern: "1 * x => x".into(),
            applies: |e| matches!(e, SesExpr::Mul(l, _) if is_one(l)),
            transform: Box::new(|e| {
                if let SesExpr::Mul(l, r) = e {
                    if is_one(&l) {
                        return *r;
                    }
                    SesExpr::Mul(l, r)
                } else {
                    e
                }
            }),
        });
        // x * 0 → 0
        self.rules.push(SesRewriteRule {
            name: "mul_zero_right".into(),
            pattern: "x * 0 => 0".into(),
            applies: |e| matches!(e, SesExpr::Mul(_, r) if is_zero(r)),
            transform: Box::new(|e| {
                if let SesExpr::Mul(_, r) = e {
                    if is_zero(&r) {
                        return SesExpr::Num(0.0);
                    }
                    SesExpr::Mul(Box::new(SesExpr::Num(0.0)), r)
                } else {
                    e
                }
            }),
        });
        // 0 * x → 0
        self.rules.push(SesRewriteRule {
            name: "mul_zero_left".into(),
            pattern: "0 * x => 0".into(),
            applies: |e| matches!(e, SesExpr::Mul(l, _) if is_zero(l)),
            transform: Box::new(|e| {
                if let SesExpr::Mul(l, _) = e {
                    if is_zero(&l) {
                        return SesExpr::Num(0.0);
                    }
                    SesExpr::Mul(l, Box::new(SesExpr::Num(0.0)))
                } else {
                    e
                }
            }),
        });
        // x / 1 → x
        self.rules.push(SesRewriteRule {
            name: "div_one".into(),
            pattern: "x / 1 => x".into(),
            applies: |e| matches!(e, SesExpr::Div(_, r) if is_one(r)),
            transform: Box::new(|e| {
                if let SesExpr::Div(l, r) = e {
                    if is_one(&r) {
                        return *l;
                    }
                    SesExpr::Div(l, r)
                } else {
                    e
                }
            }),
        });
        // 0 / x → 0  (x ≠ 0 structurally — we do it when denominator is non-zero literal)
        self.rules.push(SesRewriteRule {
            name: "zero_div".into(),
            pattern: "0 / x => 0 (x != 0)".into(),
            applies: |e| matches!(e, SesExpr::Div(l, r) if is_zero(l) && !is_zero(r)),
            transform: Box::new(|e| {
                if let SesExpr::Div(l, r) = e {
                    if is_zero(&l) && !is_zero(&r) {
                        return SesExpr::Num(0.0);
                    }
                    SesExpr::Div(l, r)
                } else {
                    e
                }
            }),
        });
        // x ^ 1 → x
        self.rules.push(SesRewriteRule {
            name: "pow_one".into(),
            pattern: "x ^ 1 => x".into(),
            applies: |e| matches!(e, SesExpr::Pow(_, r) if is_one(r)),
            transform: Box::new(|e| {
                if let SesExpr::Pow(b, r) = e {
                    if is_one(&r) {
                        return *b;
                    }
                    SesExpr::Pow(b, r)
                } else {
                    e
                }
            }),
        });
        // x ^ 0 → 1
        self.rules.push(SesRewriteRule {
            name: "pow_zero".into(),
            pattern: "x ^ 0 => 1".into(),
            applies: |e| matches!(e, SesExpr::Pow(_, r) if is_zero(r)),
            transform: Box::new(|e| {
                if let SesExpr::Pow(_, r) = e {
                    if is_zero(&r) {
                        return SesExpr::Num(1.0);
                    }
                    SesExpr::Pow(Box::new(SesExpr::Num(0.0)), r)
                } else {
                    e
                }
            }),
        });
        // 1 ^ x → 1
        self.rules.push(SesRewriteRule {
            name: "one_pow".into(),
            pattern: "1 ^ x => 1".into(),
            applies: |e| matches!(e, SesExpr::Pow(b, _) if is_one(b)),
            transform: Box::new(|e| {
                if let SesExpr::Pow(b, _) = e {
                    if is_one(&b) {
                        return SesExpr::Num(1.0);
                    }
                    SesExpr::Pow(b, Box::new(SesExpr::Num(0.0)))
                } else {
                    e
                }
            }),
        });
        // --x → x  (double negation)
        self.rules.push(SesRewriteRule {
            name: "double_neg".into(),
            pattern: "--x => x".into(),
            applies: |e| matches!(e, SesExpr::Neg(inner) if matches!(**inner, SesExpr::Neg(_))),
            transform: Box::new(|e| {
                if let SesExpr::Neg(inner) = e {
                    if let SesExpr::Neg(inner2) = *inner {
                        return *inner2;
                    }
                    SesExpr::Neg(inner)
                } else {
                    e
                }
            }),
        });
        // x - x → 0
        self.rules.push(SesRewriteRule {
            name: "sub_self".into(),
            pattern: "x - x => 0".into(),
            applies: |e| matches!(e, SesExpr::Sub(l, r) if exprs_equal(l, r)),
            transform: Box::new(|e| {
                if let SesExpr::Sub(l, r) = e {
                    if exprs_equal(&l, &r) {
                        return SesExpr::Num(0.0);
                    }
                    SesExpr::Sub(l, r)
                } else {
                    e
                }
            }),
        });
        // x / x → 1  (when x is non-trivially non-zero in structure)
        self.rules.push(SesRewriteRule {
            name: "div_self".into(),
            pattern: "x / x => 1".into(),
            applies: |e| matches!(e, SesExpr::Div(l, r) if exprs_equal(l, r) && !is_zero(l)),
            transform: Box::new(|e| {
                if let SesExpr::Div(l, r) = e {
                    if exprs_equal(&l, &r) && !is_zero(&l) {
                        return SesExpr::Num(1.0);
                    }
                    SesExpr::Div(l, r)
                } else {
                    e
                }
            }),
        });
        // sqrt(x^2) → |x|
        self.rules.push(SesRewriteRule {
            name: "sqrt_sq".into(),
            pattern: "sqrt(x^2) => |x|".into(),
            applies: |e| matches!(e, SesExpr::Sqrt(inner) if matches!(inner.as_ref(), SesExpr::Pow(_, exp) if is_two(exp))),
            transform: Box::new(|e| {
                if let SesExpr::Sqrt(inner) = e {
                    if let SesExpr::Pow(base, exp) = *inner {
                        if is_two(&exp) { return SesExpr::Abs(base); }
                        return SesExpr::Sqrt(Box::new(SesExpr::Pow(base, exp)));
                    }
                    SesExpr::Sqrt(inner)
                } else { e }
            }),
        });
        // exp(ln(x)) → x
        self.rules.push(SesRewriteRule {
            name: "exp_ln".into(),
            pattern: "exp(ln(x)) => x".into(),
            applies: |e| matches!(e, SesExpr::Exp(inner) if matches!(**inner, SesExpr::Ln(_))),
            transform: Box::new(|e| {
                if let SesExpr::Exp(inner) = e {
                    if let SesExpr::Ln(x) = *inner {
                        return *x;
                    }
                    SesExpr::Exp(inner)
                } else {
                    e
                }
            }),
        });
        // ln(exp(x)) → x
        self.rules.push(SesRewriteRule {
            name: "ln_exp".into(),
            pattern: "ln(exp(x)) => x".into(),
            applies: |e| matches!(e, SesExpr::Ln(inner) if matches!(**inner, SesExpr::Exp(_))),
            transform: Box::new(|e| {
                if let SesExpr::Ln(inner) = e {
                    if let SesExpr::Exp(x) = *inner {
                        return *x;
                    }
                    SesExpr::Ln(inner)
                } else {
                    e
                }
            }),
        });
        // ln(1) → 0
        self.rules.push(SesRewriteRule {
            name: "ln_one".into(),
            pattern: "ln(1) => 0".into(),
            applies: |e| matches!(e, SesExpr::Ln(inner) if is_one(inner)),
            transform: Box::new(|e| {
                if let SesExpr::Ln(inner) = e {
                    if is_one(&inner) {
                        return SesExpr::Num(0.0);
                    }
                    SesExpr::Ln(inner)
                } else {
                    e
                }
            }),
        });
        // exp(0) → 1
        self.rules.push(SesRewriteRule {
            name: "exp_zero".into(),
            pattern: "exp(0) => 1".into(),
            applies: |e| matches!(e, SesExpr::Exp(inner) if is_zero(inner)),
            transform: Box::new(|e| {
                if let SesExpr::Exp(inner) = e {
                    if is_zero(&inner) {
                        return SesExpr::Num(1.0);
                    }
                    SesExpr::Exp(inner)
                } else {
                    e
                }
            }),
        });
        // sqrt(0) → 0
        self.rules.push(SesRewriteRule {
            name: "sqrt_zero".into(),
            pattern: "sqrt(0) => 0".into(),
            applies: |e| matches!(e, SesExpr::Sqrt(inner) if is_zero(inner)),
            transform: Box::new(|e| {
                if let SesExpr::Sqrt(inner) = e {
                    if is_zero(&inner) {
                        return SesExpr::Num(0.0);
                    }
                    SesExpr::Sqrt(inner)
                } else {
                    e
                }
            }),
        });
        // sqrt(1) → 1
        self.rules.push(SesRewriteRule {
            name: "sqrt_one".into(),
            pattern: "sqrt(1) => 1".into(),
            applies: |e| matches!(e, SesExpr::Sqrt(inner) if is_one(inner)),
            transform: Box::new(|e| {
                if let SesExpr::Sqrt(inner) = e {
                    if is_one(&inner) {
                        return SesExpr::Num(1.0);
                    }
                    SesExpr::Sqrt(inner)
                } else {
                    e
                }
            }),
        });
        // -(-x) handled by double_neg; also: neg(0) → 0
        self.rules.push(SesRewriteRule {
            name: "neg_zero".into(),
            pattern: "-0 => 0".into(),
            applies: |e| matches!(e, SesExpr::Neg(inner) if is_zero(inner)),
            transform: Box::new(|e| {
                if let SesExpr::Neg(inner) = e {
                    if is_zero(&inner) {
                        return SesExpr::Num(0.0);
                    }
                    SesExpr::Neg(inner)
                } else {
                    e
                }
            }),
        });
        // sin²(x) + cos²(x) → 1  (trig identity)
        self.rules.push(SesRewriteRule {
            name: "sin2_cos2".into(),
            pattern: "sin(x)^2 + cos(x)^2 => 1".into(),
            applies: |e| is_sin2_plus_cos2(e),
            transform: Box::new(|e| {
                if is_sin2_plus_cos2(&e) {
                    SesExpr::Num(1.0)
                } else {
                    e
                }
            }),
        });
        // x * x → x^2
        self.rules.push(SesRewriteRule {
            name: "mul_self_to_pow2".into(),
            pattern: "x * x => x^2".into(),
            applies: |e| matches!(e, SesExpr::Mul(l, r) if exprs_equal(l, r) && !matches!(**l, SesExpr::Num(_))),
            transform: Box::new(|e| {
                if let SesExpr::Mul(l, r) = e {
                    if exprs_equal(&l, &r) && !matches!(*l, SesExpr::Num(_)) {
                        return SesExpr::Pow(l, Box::new(SesExpr::Num(2.0)));
                    }
                    SesExpr::Mul(l, r)
                } else { e }
            }),
        });
        // x + x → 2*x
        self.rules.push(SesRewriteRule {
            name: "add_self".into(),
            pattern: "x + x => 2*x".into(),
            applies: |e| matches!(e, SesExpr::Add(l, r) if exprs_equal(l, r) && !matches!(**l, SesExpr::Num(_))),
            transform: Box::new(|e| {
                if let SesExpr::Add(l, r) = e {
                    if exprs_equal(&l, &r) && !matches!(*l, SesExpr::Num(_)) {
                        return SesExpr::Mul(Box::new(SesExpr::Num(2.0)), l);
                    }
                    SesExpr::Add(l, r)
                } else { e }
            }),
        });
    }

    // -----------------------------------------------------------------------
    // Core operations
    // -----------------------------------------------------------------------

    /// Simplify `expr` using multi-pass rewriting until fixpoint or `max_passes`.
    pub fn simplify(&mut self, expr: &SesExpr) -> SesExpr {
        self.stats.simplify_calls += 1;
        let mut current = expr.clone();
        let mut pass = 0;
        let max = self.config.max_passes;
        loop {
            let before_pass = current.clone();
            current = self.one_pass(current, pass);
            pass += 1;
            self.stats.total_passes += 1;
            if exprs_equal(&current, &before_pass) {
                self.stats.fixpoint_reached += 1;
                break;
            }
            if pass >= max {
                break;
            }
        }
        current
    }

    /// Perform one full bottom-up rewriting pass.
    fn one_pass(&mut self, expr: SesExpr, pass: usize) -> SesExpr {
        // First recurse into children
        let expr = self.recurse_children(expr, pass);
        // Then try to apply constant folding
        let expr = if self.config.enable_constant_folding {
            self.fold_constants(expr, pass)
        } else {
            expr
        };
        // Then try each rule in order
        self.apply_rules(expr, pass)
    }

    /// Recursively simplify all children of an expression.
    fn recurse_children(&mut self, expr: SesExpr, pass: usize) -> SesExpr {
        match expr {
            SesExpr::Add(l, r) => SesExpr::Add(
                Box::new(self.one_pass(*l, pass)),
                Box::new(self.one_pass(*r, pass)),
            ),
            SesExpr::Sub(l, r) => SesExpr::Sub(
                Box::new(self.one_pass(*l, pass)),
                Box::new(self.one_pass(*r, pass)),
            ),
            SesExpr::Mul(l, r) => SesExpr::Mul(
                Box::new(self.one_pass(*l, pass)),
                Box::new(self.one_pass(*r, pass)),
            ),
            SesExpr::Div(l, r) => SesExpr::Div(
                Box::new(self.one_pass(*l, pass)),
                Box::new(self.one_pass(*r, pass)),
            ),
            SesExpr::Pow(b, e) => SesExpr::Pow(
                Box::new(self.one_pass(*b, pass)),
                Box::new(self.one_pass(*e, pass)),
            ),
            SesExpr::Neg(inner) => SesExpr::Neg(Box::new(self.one_pass(*inner, pass))),
            SesExpr::Abs(inner) => SesExpr::Abs(Box::new(self.one_pass(*inner, pass))),
            SesExpr::Sqrt(inner) => SesExpr::Sqrt(Box::new(self.one_pass(*inner, pass))),
            SesExpr::Sin(inner) => SesExpr::Sin(Box::new(self.one_pass(*inner, pass))),
            SesExpr::Cos(inner) => SesExpr::Cos(Box::new(self.one_pass(*inner, pass))),
            SesExpr::Exp(inner) => SesExpr::Exp(Box::new(self.one_pass(*inner, pass))),
            SesExpr::Ln(inner) => SesExpr::Ln(Box::new(self.one_pass(*inner, pass))),
            atom => atom,
        }
    }

    /// Try to fold constant sub-expressions.
    fn fold_constants(&mut self, expr: SesExpr, pass: usize) -> SesExpr {
        if !is_constant(&expr) {
            return expr;
        }
        // Evaluate with an empty binding map (no variables)
        let vars: HashMap<String, f64> = HashMap::new();
        if let Ok(v) = self.evaluate_inner(&expr, &vars) {
            if v.is_finite() {
                let before = expr.to_string();
                let folded = SesExpr::Num(v);
                self.record_step(pass, "constant_folding", &before, &folded.to_string());
                return folded;
            }
        }
        expr
    }

    /// Apply the first matching rule to `expr`.
    fn apply_rules(&mut self, expr: SesExpr, pass: usize) -> SesExpr {
        // Collect indices of applicable rules to avoid borrow issues
        let applicable: Vec<usize> = self
            .rules
            .iter()
            .enumerate()
            .filter(|(_, r)| {
                let skip = (r.name.starts_with("sin2_cos2") && !self.config.enable_trig_rules)
                    || (!self.config.enable_algebraic_rules && !r.name.starts_with("constant"));
                if skip {
                    return false;
                }
                (r.applies)(&expr)
            })
            .map(|(i, _)| i)
            .collect();

        if let Some(&idx) = applicable.first() {
            let rule_name = self.rules[idx].name.clone();
            let before = expr.to_string();
            let result = (self.rules[idx].transform)(expr);
            let after = result.to_string();
            if before != after {
                self.stats.total_rule_applications += 1;
                self.record_step(pass, &rule_name, &before, &after);
            }
            result
        } else {
            expr
        }
    }

    fn record_step(&mut self, pass: usize, rule: &str, before: &str, after: &str) {
        if self.history.len() >= 500 {
            self.history.pop_front();
        }
        self.history.push_back(SesSimplificationStep {
            pass,
            rule_applied: rule.to_string(),
            before: before.to_string(),
            after: after.to_string(),
        });
        self.stats.history_len = self.history.len();
    }

    // -----------------------------------------------------------------------
    // Numeric evaluation
    // -----------------------------------------------------------------------

    /// Evaluate `expr` numerically given variable bindings.
    pub fn evaluate(
        &mut self,
        expr: &SesExpr,
        vars: &HashMap<String, f64>,
    ) -> Result<f64, SesError> {
        self.stats.evaluate_calls += 1;
        self.evaluate_inner(expr, vars)
    }

    fn evaluate_inner(&self, expr: &SesExpr, vars: &HashMap<String, f64>) -> Result<f64, SesError> {
        match expr {
            SesExpr::Num(v) => Ok(*v),
            SesExpr::Var(name) => vars
                .get(name.as_str())
                .copied()
                .ok_or_else(|| SesError::UnboundVariable(name.clone())),
            SesExpr::Add(l, r) => Ok(self.evaluate_inner(l, vars)? + self.evaluate_inner(r, vars)?),
            SesExpr::Sub(l, r) => Ok(self.evaluate_inner(l, vars)? - self.evaluate_inner(r, vars)?),
            SesExpr::Mul(l, r) => Ok(self.evaluate_inner(l, vars)? * self.evaluate_inner(r, vars)?),
            SesExpr::Div(l, r) => {
                let denom = self.evaluate_inner(r, vars)?;
                if denom == 0.0 {
                    return Err(SesError::DivisionByZero);
                }
                Ok(self.evaluate_inner(l, vars)? / denom)
            }
            SesExpr::Pow(b, e) => {
                let base = self.evaluate_inner(b, vars)?;
                let exp = self.evaluate_inner(e, vars)?;
                Ok(base.powf(exp))
            }
            SesExpr::Neg(inner) => Ok(-self.evaluate_inner(inner, vars)?),
            SesExpr::Abs(inner) => Ok(self.evaluate_inner(inner, vars)?.abs()),
            SesExpr::Sqrt(inner) => {
                let v = self.evaluate_inner(inner, vars)?;
                if v < 0.0 {
                    return Err(SesError::MathDomainError("sqrt of negative number".into()));
                }
                Ok(v.sqrt())
            }
            SesExpr::Sin(inner) => Ok(self.evaluate_inner(inner, vars)?.sin()),
            SesExpr::Cos(inner) => Ok(self.evaluate_inner(inner, vars)?.cos()),
            SesExpr::Exp(inner) => Ok(self.evaluate_inner(inner, vars)?.exp()),
            SesExpr::Ln(inner) => {
                let v = self.evaluate_inner(inner, vars)?;
                if v <= 0.0 {
                    return Err(SesError::MathDomainError(
                        "ln of non-positive number".into(),
                    ));
                }
                Ok(v.ln())
            }
        }
    }

    // -----------------------------------------------------------------------
    // Symbolic differentiation
    // -----------------------------------------------------------------------

    /// Compute the symbolic derivative of `expr` with respect to `var`.
    pub fn differentiate(&mut self, expr: &SesExpr, var: &str) -> SesExpr {
        self.stats.differentiate_calls += 1;
        self.diff(expr, var)
    }

    fn diff(&self, expr: &SesExpr, var: &str) -> SesExpr {
        match expr {
            SesExpr::Num(_) => SesExpr::Num(0.0),
            SesExpr::Var(name) => {
                if name == var {
                    SesExpr::Num(1.0)
                } else {
                    SesExpr::Num(0.0)
                }
            }
            // (f + g)' = f' + g'
            SesExpr::Add(l, r) => SesExpr::add(self.diff(l, var), self.diff(r, var)),
            // (f - g)' = f' - g'
            SesExpr::Sub(l, r) => SesExpr::sub(self.diff(l, var), self.diff(r, var)),
            // (f * g)' = f'g + fg'
            SesExpr::Mul(l, r) => SesExpr::add(
                SesExpr::mul(self.diff(l, var), *r.clone()),
                SesExpr::mul(*l.clone(), self.diff(r, var)),
            ),
            // (f / g)' = (f'g - fg') / g²
            SesExpr::Div(l, r) => {
                let f_prime = self.diff(l, var);
                let g_prime = self.diff(r, var);
                let num = SesExpr::sub(
                    SesExpr::mul(f_prime, *r.clone()),
                    SesExpr::mul(*l.clone(), g_prime),
                );
                let denom = SesExpr::pow(*r.clone(), SesExpr::Num(2.0));
                SesExpr::div(num, denom)
            }
            // (f^g)' — general case via logarithmic differentiation
            // (f^g)' = f^g * (g' * ln(f) + g * f'/f)
            SesExpr::Pow(base, exp) => {
                let base_dep = self.contains_var(base, var);
                let exp_dep = self.contains_var(exp, var);
                match (base_dep, exp_dep) {
                    (false, false) => SesExpr::Num(0.0),
                    // f^n where n is constant: n * f^(n-1) * f'
                    (true, false) => {
                        let n = *exp.clone();
                        let n_minus_1 = SesExpr::sub(n.clone(), SesExpr::Num(1.0));
                        let chain = self.diff(base, var);
                        SesExpr::mul(
                            SesExpr::mul(n, SesExpr::pow(*base.clone(), n_minus_1)),
                            chain,
                        )
                    }
                    // a^g where a is constant: a^g * ln(a) * g'
                    (false, true) => {
                        let chain = self.diff(exp, var);
                        SesExpr::mul(
                            SesExpr::mul(
                                SesExpr::pow(*base.clone(), *exp.clone()),
                                SesExpr::Ln(base.clone()),
                            ),
                            chain,
                        )
                    }
                    // General: f^g * (g' * ln(f) + g * f'/f)
                    (true, true) => {
                        let f_prime = self.diff(base, var);
                        let g_prime = self.diff(exp, var);
                        let term1 = SesExpr::mul(g_prime, SesExpr::Ln(base.clone()));
                        let term2 =
                            SesExpr::mul(*exp.clone(), SesExpr::div(f_prime, *base.clone()));
                        SesExpr::mul(
                            SesExpr::pow(*base.clone(), *exp.clone()),
                            SesExpr::add(term1, term2),
                        )
                    }
                }
            }
            // (-f)' = -f'
            SesExpr::Neg(inner) => SesExpr::neg(self.diff(inner, var)),
            // |f|' = f * f' / |f|  (we represent it symbolically)
            SesExpr::Abs(inner) => {
                let f_prime = self.diff(inner, var);
                SesExpr::div(
                    SesExpr::mul(*inner.clone(), f_prime),
                    SesExpr::Abs(inner.clone()),
                )
            }
            // sqrt(f)' = f' / (2 * sqrt(f))
            SesExpr::Sqrt(inner) => {
                let f_prime = self.diff(inner, var);
                SesExpr::div(
                    f_prime,
                    SesExpr::mul(SesExpr::Num(2.0), SesExpr::Sqrt(inner.clone())),
                )
            }
            // sin(f)' = cos(f) * f'
            SesExpr::Sin(inner) => {
                let f_prime = self.diff(inner, var);
                SesExpr::mul(SesExpr::Cos(inner.clone()), f_prime)
            }
            // cos(f)' = -sin(f) * f'
            SesExpr::Cos(inner) => {
                let f_prime = self.diff(inner, var);
                SesExpr::neg(SesExpr::mul(SesExpr::Sin(inner.clone()), f_prime))
            }
            // exp(f)' = exp(f) * f'
            SesExpr::Exp(inner) => {
                let f_prime = self.diff(inner, var);
                SesExpr::mul(SesExpr::Exp(inner.clone()), f_prime)
            }
            // ln(f)' = f' / f
            SesExpr::Ln(inner) => {
                let f_prime = self.diff(inner, var);
                SesExpr::div(f_prime, *inner.clone())
            }
        }
    }

    // -----------------------------------------------------------------------
    // Substitution
    // -----------------------------------------------------------------------

    /// Substitute all occurrences of `var` in `expr` with `replacement`.
    pub fn substitute(&self, expr: &SesExpr, var: &str, replacement: &SesExpr) -> SesExpr {
        match expr {
            SesExpr::Var(name) if name == var => replacement.clone(),
            SesExpr::Num(_) | SesExpr::Var(_) => expr.clone(),
            SesExpr::Add(l, r) => SesExpr::add(
                self.substitute(l, var, replacement),
                self.substitute(r, var, replacement),
            ),
            SesExpr::Sub(l, r) => SesExpr::sub(
                self.substitute(l, var, replacement),
                self.substitute(r, var, replacement),
            ),
            SesExpr::Mul(l, r) => SesExpr::mul(
                self.substitute(l, var, replacement),
                self.substitute(r, var, replacement),
            ),
            SesExpr::Div(l, r) => SesExpr::div(
                self.substitute(l, var, replacement),
                self.substitute(r, var, replacement),
            ),
            SesExpr::Pow(b, e) => SesExpr::pow(
                self.substitute(b, var, replacement),
                self.substitute(e, var, replacement),
            ),
            SesExpr::Neg(inner) => SesExpr::neg(self.substitute(inner, var, replacement)),
            SesExpr::Abs(inner) => SesExpr::Abs(Box::new(self.substitute(inner, var, replacement))),
            SesExpr::Sqrt(inner) => {
                SesExpr::Sqrt(Box::new(self.substitute(inner, var, replacement)))
            }
            SesExpr::Sin(inner) => SesExpr::Sin(Box::new(self.substitute(inner, var, replacement))),
            SesExpr::Cos(inner) => SesExpr::Cos(Box::new(self.substitute(inner, var, replacement))),
            SesExpr::Exp(inner) => SesExpr::Exp(Box::new(self.substitute(inner, var, replacement))),
            SesExpr::Ln(inner) => SesExpr::Ln(Box::new(self.substitute(inner, var, replacement))),
        }
    }

    // -----------------------------------------------------------------------
    // Variable helpers
    // -----------------------------------------------------------------------

    /// Returns `true` if `expr` contains the variable `var`.
    pub fn contains_var(&self, expr: &SesExpr, var: &str) -> bool {
        match expr {
            SesExpr::Var(name) => name == var,
            SesExpr::Num(_) => false,
            SesExpr::Add(l, r)
            | SesExpr::Sub(l, r)
            | SesExpr::Mul(l, r)
            | SesExpr::Div(l, r)
            | SesExpr::Pow(l, r) => self.contains_var(l, var) || self.contains_var(r, var),
            SesExpr::Neg(i)
            | SesExpr::Abs(i)
            | SesExpr::Sqrt(i)
            | SesExpr::Sin(i)
            | SesExpr::Cos(i)
            | SesExpr::Exp(i)
            | SesExpr::Ln(i) => self.contains_var(i, var),
        }
    }

    /// Collect all distinct variable names appearing in `expr`.
    pub fn collect_vars(&self, expr: &SesExpr) -> HashSet<String> {
        let mut set = HashSet::new();
        collect_vars_inner(expr, &mut set);
        set
    }

    // -----------------------------------------------------------------------
    // String conversion
    // -----------------------------------------------------------------------

    /// Convert `expr` to infix string notation with minimal parentheses.
    pub fn to_string(&self, expr: &SesExpr) -> String {
        expr.to_string()
    }

    // -----------------------------------------------------------------------
    // Parser
    // -----------------------------------------------------------------------

    /// Parse an infix expression string into a `SesExpr`.
    ///
    /// Supports: numbers, identifiers (variables), `+`, `-`, `*`, `/`, `^`,
    /// unary `-`, and function calls `sin`, `cos`, `exp`, `ln`, `sqrt`, `abs`.
    /// Parentheses and `|...|` for absolute value are also supported.
    pub fn parse(s: &str) -> Result<SesExpr, SesError> {
        let tokens = tokenize(s)?;
        let mut parser = Parser::new(tokens);
        let expr = parser.parse_expr()?;
        if !parser.is_at_end() {
            return Err(SesError::ParseError(format!(
                "unexpected token at position {}",
                parser.pos
            )));
        }
        Ok(expr)
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    /// Return current simplifier statistics.
    pub fn simplifier_stats(&self) -> SesSimplifierStats {
        let mut stats = self.stats.clone();
        stats.history_len = self.history.len();
        stats
    }

    /// Return a reference to the simplification history.
    pub fn history(&self) -> &VecDeque<SesSimplificationStep> {
        &self.history
    }

    /// Clear the simplification history.
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.stats.history_len = 0;
    }

    /// Return a reference to the current configuration.
    pub fn config(&self) -> &SesSimplifierConfig {
        &self.config
    }

    /// Return number of registered rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

impl Default for SymbolicExpressionSimplifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helper predicates
// ---------------------------------------------------------------------------

fn is_zero(e: &SesExpr) -> bool {
    matches!(e, SesExpr::Num(v) if *v == 0.0)
}

fn is_one(e: &SesExpr) -> bool {
    matches!(e, SesExpr::Num(v) if *v == 1.0)
}

fn is_two(e: &SesExpr) -> bool {
    matches!(e, SesExpr::Num(v) if *v == 2.0)
}

/// Returns `true` if `expr` contains no variables (fully constant).
fn is_constant(e: &SesExpr) -> bool {
    match e {
        SesExpr::Num(_) => true,
        SesExpr::Var(_) => false,
        SesExpr::Add(l, r)
        | SesExpr::Sub(l, r)
        | SesExpr::Mul(l, r)
        | SesExpr::Div(l, r)
        | SesExpr::Pow(l, r) => is_constant(l) && is_constant(r),
        SesExpr::Neg(i)
        | SesExpr::Abs(i)
        | SesExpr::Sqrt(i)
        | SesExpr::Sin(i)
        | SesExpr::Cos(i)
        | SesExpr::Exp(i)
        | SesExpr::Ln(i) => is_constant(i),
    }
}

/// Structural equality of two expressions (sufficient for rewrite rules).
fn exprs_equal(a: &SesExpr, b: &SesExpr) -> bool {
    a == b
}

/// Check if `e` is of the form `sin(x)^2 + cos(x)^2`.
fn is_sin2_plus_cos2(e: &SesExpr) -> bool {
    let SesExpr::Add(l, r) = e else {
        return false;
    };
    is_sin_squared(l) && is_cos_squared(r) || is_cos_squared(l) && is_sin_squared(r)
}

fn is_sin_squared(e: &SesExpr) -> bool {
    if let SesExpr::Pow(base, exp) = e {
        is_two(exp) && matches!(base.as_ref(), SesExpr::Sin(_))
    } else {
        false
    }
}

fn is_cos_squared(e: &SesExpr) -> bool {
    if let SesExpr::Pow(base, exp) = e {
        is_two(exp) && matches!(base.as_ref(), SesExpr::Cos(_))
    } else {
        false
    }
}

fn collect_vars_inner(e: &SesExpr, set: &mut HashSet<String>) {
    match e {
        SesExpr::Var(name) => {
            set.insert(name.clone());
        }
        SesExpr::Num(_) => {}
        SesExpr::Add(l, r)
        | SesExpr::Sub(l, r)
        | SesExpr::Mul(l, r)
        | SesExpr::Div(l, r)
        | SesExpr::Pow(l, r) => {
            collect_vars_inner(l, set);
            collect_vars_inner(r, set);
        }
        SesExpr::Neg(i)
        | SesExpr::Abs(i)
        | SesExpr::Sqrt(i)
        | SesExpr::Sin(i)
        | SesExpr::Cos(i)
        | SesExpr::Exp(i)
        | SesExpr::Ln(i) => collect_vars_inner(i, set),
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Num(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Pipe,
}

fn tokenize(s: &str) -> Result<Vec<Token>, SesError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit())
        {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            // Allow 'e'/'E' for scientific notation
            if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                i += 1;
                if i < chars.len() && (chars[i] == '+' || chars[i] == '-') {
                    i += 1;
                }
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let num_str: String = chars[start..i].iter().collect();
            let v = num_str
                .parse::<f64>()
                .map_err(|_| SesError::ParseError(format!("invalid number: {num_str}")))?;
            tokens.push(Token::Num(v));
        } else if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let ident: String = chars[start..i].iter().collect();
            tokens.push(Token::Ident(ident));
        } else {
            let tok = match c {
                '+' => Token::Plus,
                '-' => Token::Minus,
                '*' => Token::Star,
                '/' => Token::Slash,
                '^' => Token::Caret,
                '(' => Token::LParen,
                ')' => Token::RParen,
                '|' => Token::Pipe,
                _ => return Err(SesError::ParseError(format!("unexpected character: '{c}'"))),
            };
            tokens.push(tok);
            i += 1;
        }
    }
    Ok(tokens)
}

// ---------------------------------------------------------------------------
// Recursive-descent parser
// ---------------------------------------------------------------------------

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn consume(&mut self) -> Option<Token> {
        if self.pos < self.tokens.len() {
            let t = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(t)
        } else {
            None
        }
    }

    fn expect(&mut self, expected: &Token) -> Result<(), SesError> {
        match self.consume() {
            Some(ref t) if t == expected => Ok(()),
            Some(t) => Err(SesError::ParseError(format!(
                "expected {expected:?}, got {t:?}"
            ))),
            None => Err(SesError::ParseError(format!(
                "expected {expected:?}, got EOF"
            ))),
        }
    }

    fn is_at_end(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    /// expr = additive
    fn parse_expr(&mut self) -> Result<SesExpr, SesError> {
        self.parse_additive()
    }

    /// additive = multiplicative (('+' | '-') multiplicative)*
    fn parse_additive(&mut self) -> Result<SesExpr, SesError> {
        let mut lhs = self.parse_multiplicative()?;
        loop {
            match self.peek() {
                Some(Token::Plus) => {
                    self.consume();
                    let rhs = self.parse_multiplicative()?;
                    lhs = SesExpr::add(lhs, rhs);
                }
                Some(Token::Minus) => {
                    self.consume();
                    let rhs = self.parse_multiplicative()?;
                    lhs = SesExpr::sub(lhs, rhs);
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    /// multiplicative = power (('*' | '/') power)*
    fn parse_multiplicative(&mut self) -> Result<SesExpr, SesError> {
        let mut lhs = self.parse_power()?;
        loop {
            match self.peek() {
                Some(Token::Star) => {
                    self.consume();
                    let rhs = self.parse_power()?;
                    lhs = SesExpr::mul(lhs, rhs);
                }
                Some(Token::Slash) => {
                    self.consume();
                    let rhs = self.parse_power()?;
                    lhs = SesExpr::div(lhs, rhs);
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    /// power = unary ('^' power)?   (right-associative)
    fn parse_power(&mut self) -> Result<SesExpr, SesError> {
        let base = self.parse_unary()?;
        if matches!(self.peek(), Some(Token::Caret)) {
            self.consume();
            let exp = self.parse_power()?; // right-associative
            Ok(SesExpr::pow(base, exp))
        } else {
            Ok(base)
        }
    }

    /// unary = '-' unary | atom
    fn parse_unary(&mut self) -> Result<SesExpr, SesError> {
        if matches!(self.peek(), Some(Token::Minus)) {
            self.consume();
            let inner = self.parse_unary()?;
            Ok(SesExpr::neg(inner))
        } else {
            self.parse_atom()
        }
    }

    /// atom = num | ident | ident '(' expr ')' | '(' expr ')' | '|' expr '|'
    fn parse_atom(&mut self) -> Result<SesExpr, SesError> {
        match self.peek().cloned() {
            Some(Token::Num(v)) => {
                self.consume();
                Ok(SesExpr::Num(v))
            }
            Some(Token::Ident(name)) => {
                self.consume();
                // Function call?
                if matches!(self.peek(), Some(Token::LParen)) {
                    self.consume(); // consume '('
                    let arg = self.parse_expr()?;
                    self.expect(&Token::RParen)?;
                    match name.to_lowercase().as_str() {
                        "sin" => Ok(SesExpr::Sin(Box::new(arg))),
                        "cos" => Ok(SesExpr::Cos(Box::new(arg))),
                        "exp" => Ok(SesExpr::Exp(Box::new(arg))),
                        "ln" | "log" => Ok(SesExpr::Ln(Box::new(arg))),
                        "sqrt" => Ok(SesExpr::Sqrt(Box::new(arg))),
                        "abs" => Ok(SesExpr::Abs(Box::new(arg))),
                        "neg" => Ok(SesExpr::neg(arg)),
                        other => Err(SesError::ParseError(format!("unknown function: {other}"))),
                    }
                } else {
                    Ok(SesExpr::Var(name))
                }
            }
            Some(Token::LParen) => {
                self.consume();
                let inner = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(inner)
            }
            Some(Token::Pipe) => {
                self.consume();
                let inner = self.parse_expr()?;
                self.expect(&Token::Pipe)?;
                Ok(SesExpr::Abs(Box::new(inner)))
            }
            Some(t) => Err(SesError::ParseError(format!("unexpected token: {t:?}"))),
            None => Err(SesError::ParseError("unexpected end of input".into())),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — split into ses_tests.rs to keep this file under 2000 lines.
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ses_tests.rs"]
mod tests;
