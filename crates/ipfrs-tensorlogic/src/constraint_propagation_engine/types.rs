//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{BTreeSet, HashMap};

use super::type_aliases::CpeVarId;

/// Which arc-consistency algorithm to apply during propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcLevel {
    /// AC3 — standard worklist algorithm (O(ed³))
    Ac3,
    /// AC4 — support-based, optimal O(ed²) but higher constant
    Ac4,
    /// AC6 — one-direction support, O(ed²) with lower constant than AC4
    Ac6,
}
/// The domain of a CSP variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpeDomain {
    /// A contiguous integer interval [lo, hi] (inclusive).
    Interval { lo: i64, hi: i64 },
    /// An explicit finite set of integer values.
    Finite(BTreeSet<i64>),
    /// Boolean domain {0, 1}.
    Boolean,
}
impl CpeDomain {
    /// Returns `true` if the domain is empty (no values possible).
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Interval { lo, hi } => lo > hi,
            Self::Finite(s) => s.is_empty(),
            Self::Boolean => false,
        }
    }
    /// Number of values in the domain.
    pub fn size(&self) -> usize {
        match self {
            Self::Interval { lo, hi } => {
                if lo > hi {
                    0
                } else {
                    (hi - lo + 1) as usize
                }
            }
            Self::Finite(s) => s.len(),
            Self::Boolean => 2,
        }
    }
    /// Minimum value in the domain, if any.
    pub fn min_val(&self) -> Option<i64> {
        match self {
            Self::Interval { lo, hi } => {
                if lo <= hi {
                    Some(*lo)
                } else {
                    None
                }
            }
            Self::Finite(s) => s.iter().next().copied(),
            Self::Boolean => Some(0),
        }
    }
    /// Maximum value in the domain, if any.
    pub fn max_val(&self) -> Option<i64> {
        match self {
            Self::Interval { lo, hi } => {
                if lo <= hi {
                    Some(*hi)
                } else {
                    None
                }
            }
            Self::Finite(s) => s.iter().next_back().copied(),
            Self::Boolean => Some(1),
        }
    }
    /// Returns `true` if the given value is in the domain.
    pub fn contains(&self, v: i64) -> bool {
        match self {
            Self::Interval { lo, hi } => v >= *lo && v <= *hi,
            Self::Finite(s) => s.contains(&v),
            Self::Boolean => v == 0 || v == 1,
        }
    }
    /// Remove a single value from the domain (returns true if domain changed).
    pub fn remove(&mut self, v: i64) -> bool {
        match self {
            Self::Interval { lo, hi } => {
                if v < *lo || v > *hi {
                    return false;
                }
                if v == *lo {
                    *lo += 1;
                    true
                } else if v == *hi {
                    *hi -= 1;
                    true
                } else {
                    let lo_val = *lo;
                    let hi_val = *hi;
                    let mut set = BTreeSet::new();
                    for x in lo_val..=hi_val {
                        if x != v {
                            set.insert(x);
                        }
                    }
                    *self = Self::Finite(set);
                    true
                }
            }
            Self::Finite(s) => s.remove(&v),
            Self::Boolean => {
                if v == 0 || v == 1 {
                    let keep = if v == 0 { 1i64 } else { 0i64 };
                    let mut set = BTreeSet::new();
                    set.insert(keep);
                    *self = Self::Finite(set);
                    true
                } else {
                    false
                }
            }
        }
    }
    /// Restrict the lower bound (remove all values < new_lo).
    pub fn tighten_lo(&mut self, new_lo: i64) -> bool {
        match self {
            Self::Interval { lo, hi: _ } => {
                if new_lo > *lo {
                    *lo = new_lo;
                    true
                } else {
                    false
                }
            }
            Self::Finite(s) => {
                let before = s.len();
                s.retain(|&x| x >= new_lo);
                s.len() < before
            }
            Self::Boolean => {
                if new_lo > 1 {
                    *self = Self::Finite(BTreeSet::new());
                    true
                } else if new_lo == 1 {
                    let mut set = BTreeSet::new();
                    set.insert(1i64);
                    *self = Self::Finite(set);
                    true
                } else {
                    false
                }
            }
        }
    }
    /// Restrict the upper bound (remove all values > new_hi).
    pub fn tighten_hi(&mut self, new_hi: i64) -> bool {
        match self {
            Self::Interval { lo, hi } => {
                let _ = lo;
                if new_hi < *hi {
                    *hi = new_hi;
                    true
                } else {
                    false
                }
            }
            Self::Finite(s) => {
                let before = s.len();
                s.retain(|&x| x <= new_hi);
                s.len() < before
            }
            Self::Boolean => {
                if new_hi < 0 {
                    *self = Self::Finite(BTreeSet::new());
                    true
                } else if new_hi == 0 {
                    let mut set = BTreeSet::new();
                    set.insert(0i64);
                    *self = Self::Finite(set);
                    true
                } else {
                    false
                }
            }
        }
    }
    /// Return an iterator over all values (materialises interval if needed).
    pub fn values(&self) -> Vec<i64> {
        match self {
            Self::Interval { lo, hi } => {
                if lo > hi {
                    vec![]
                } else {
                    (*lo..=*hi).collect()
                }
            }
            Self::Finite(s) => s.iter().copied().collect(),
            Self::Boolean => vec![0, 1],
        }
    }
    /// Assign a single value (set domain to singleton).
    pub fn assign_value(&mut self, v: i64) {
        let mut set = BTreeSet::new();
        set.insert(v);
        *self = Self::Finite(set);
    }
    /// Return `Some(v)` if domain is a singleton.
    pub fn singleton(&self) -> Option<i64> {
        match self {
            Self::Interval { lo, hi } if lo == hi => Some(*lo),
            Self::Finite(s) if s.len() == 1 => s.iter().next().copied(),
            _ => None,
        }
    }
}
/// A variable in the constraint propagation engine.
#[derive(Debug, Clone)]
pub struct CpeVariable {
    /// Unique identifier.
    pub id: CpeVarId,
    /// Human-readable name.
    pub name: String,
    /// Current domain.
    pub domain: CpeDomain,
    /// Whether the variable has been assigned a definite value.
    pub is_assigned: bool,
}
impl CpeVariable {
    pub(super) fn new(id: CpeVarId, name: String, domain: CpeDomain) -> Self {
        Self {
            id,
            name,
            domain,
            is_assigned: false,
        }
    }
}
/// Errors produced by the constraint propagation engine.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CpeError {
    /// The specified variable ID does not exist.
    #[error("variable {0} not found")]
    UnknownVariable(CpeVarId),
    /// Assigning a value that is not in the current domain.
    #[error("value {value} not in domain of variable {var_id}")]
    ValueNotInDomain { var_id: CpeVarId, value: i64 },
    /// A domain became empty during propagation (infeasible sub-problem).
    #[error("domain of variable {0} became empty — constraint infeasible")]
    DomainEmpty(CpeVarId),
    /// Maximum iteration count reached without convergence.
    #[error("maximum iterations ({0}) reached without convergence")]
    MaxIterationsExceeded(usize),
}
#[derive(Clone)]
pub(super) struct EngineSnapshot {
    pub(super) variables: HashMap<CpeVarId, CpeVariable>,
    pub(super) stats: CpePropagationStats,
}
/// Result of a single `propagate()` call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpePropagationResult {
    /// All domains are non-empty and consistent.
    Consistent,
    /// At least one domain became empty — the problem is infeasible.
    Infeasible,
    /// The problem is fully solved (all variables assigned).
    Solved,
}
/// Configuration for `ConstraintPropagationEngine`.
#[derive(Debug, Clone)]
pub struct CpeEngineConfig {
    /// Maximum number of propagation iterations before giving up.
    pub max_iterations: usize,
    /// Arc consistency level to apply.
    pub arc_consistency_level: AcLevel,
    /// Whether to run bounds propagation in addition to AC.
    pub use_bounds_propagation: bool,
    /// Use Fail-First (MRV) variable ordering in backtracking.
    pub fail_first: bool,
}
/// Statistics collected during a single propagation run.
#[derive(Debug, Clone, Default)]
pub struct CpePropagationStats {
    /// Number of arc-revision operations performed.
    pub arc_revisions: u64,
    /// Number of domain values removed.
    pub values_removed: u64,
    /// Number of full propagation passes.
    pub passes: u64,
    /// Total propagation calls since engine creation.
    pub total_propagations: u64,
    /// Number of bounds tightenings.
    pub bounds_tightenings: u64,
    /// Number of backtrack solver nodes explored.
    pub backtrack_nodes: u64,
}
/// A constraint over one or more variables.
#[derive(Debug, Clone)]
pub enum CpeConstraint {
    /// All variables must take distinct values.
    AllDifferent(Vec<CpeVarId>),
    /// x == y
    Equal(CpeVarId, CpeVarId),
    /// x != y
    NotEqual(CpeVarId, CpeVarId),
    /// x < y
    LessThan(CpeVarId, CpeVarId),
    /// x <= y
    LessEqual(CpeVarId, CpeVarId),
    /// sum(vars) == total
    Sum { vars: Vec<CpeVarId>, total: i64 },
    /// Σ coeff_i * x_i == rhs
    LinearExpr {
        coeffs: Vec<(CpeVarId, i64)>,
        rhs: i64,
    },
    /// x must belong to the given list of values
    InDomain(CpeVarId, Vec<i64>),
    /// |x| == y  (x may be negative; y must be non-negative)
    Abs(CpeVarId, CpeVarId),
}
impl CpeConstraint {
    /// Returns all variable IDs mentioned by this constraint.
    pub fn variables(&self) -> Vec<CpeVarId> {
        match self {
            Self::AllDifferent(v) => v.clone(),
            Self::Equal(a, b)
            | Self::NotEqual(a, b)
            | Self::LessThan(a, b)
            | Self::LessEqual(a, b)
            | Self::Abs(a, b) => vec![*a, *b],
            Self::Sum { vars, .. } => vars.clone(),
            Self::LinearExpr { coeffs, .. } => coeffs.iter().map(|(id, _)| *id).collect(),
            Self::InDomain(x, _) => vec![*x],
        }
    }
    /// Returns `true` if this constraint mentions the given variable.
    pub fn involves(&self, var: CpeVarId) -> bool {
        self.variables().contains(&var)
    }
}
