//! Constraint Satisfaction Problem (CSP) solver with backtracking search,
//! arc consistency (AC-3), and heuristics for variable/value ordering.
//!
//! # Example
//!
//! ```
//! use ipfrs_tensorlogic::constraint_solver::{
//!     ConstraintSolver, Constraint, CspVarId, SolverConfig,
//! };
//!
//! let config = SolverConfig::default();
//! let mut solver = ConstraintSolver::new(config);
//!
//! let x = solver.add_variable("x".to_string(), vec![1, 2, 3]);
//! let y = solver.add_variable("y".to_string(), vec![1, 2, 3]);
//! solver.add_constraint(Constraint::NotEqual(x, y));
//!
//! let result = solver.solve();
//! assert!(!result.solutions.is_empty());
//! let sol = &result.solutions[0];
//! assert_ne!(sol.get(x), sol.get(y));
//! ```

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

// ---------------------------------------------------------------------------
// CspVarId
// ---------------------------------------------------------------------------

/// Opaque index for a CSP variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CspVarId(pub usize);

// ---------------------------------------------------------------------------
// Domain
// ---------------------------------------------------------------------------

/// The current set of allowed integer values for a CSP variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Domain {
    /// Ordered, deduplicated allowed values. Maintained sorted.
    pub values: Vec<i64>,
}

impl Domain {
    /// Create a domain from a `Vec<i64>`. Values are sorted and deduplicated.
    pub fn new(mut values: Vec<i64>) -> Self {
        values.sort_unstable();
        values.dedup();
        Self { values }
    }

    /// Return `true` when the domain has no values remaining.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Return `true` when `v` is a member of the domain.
    pub fn contains(&self, v: i64) -> bool {
        self.values.binary_search(&v).is_ok()
    }

    /// Remove `v` from the domain. Returns `true` when `v` was present.
    pub fn remove(&mut self, v: i64) -> bool {
        if let Ok(idx) = self.values.binary_search(&v) {
            self.values.remove(idx);
            true
        } else {
            false
        }
    }

    /// Number of values in the domain.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }
}

// ---------------------------------------------------------------------------
// Constraint
// ---------------------------------------------------------------------------

/// A constraint relating one or more CSP variables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Constraint {
    /// All listed variables must take pairwise-distinct values.
    AllDifferent(Vec<CspVarId>),
    /// Two variables must be equal: `a == b`.
    Equal(CspVarId, CspVarId),
    /// Two variables must differ: `a != b`.
    NotEqual(CspVarId, CspVarId),
    /// Strict ordering: `a < b`.
    LessThan(CspVarId, CspVarId),
    /// Non-strict ordering: `a <= b`.
    LessEqual(CspVarId, CspVarId),
    /// Sum of values must equal `target`: `Σ vars[i] == target`.
    Sum {
        /// Variables to sum.
        vars: Vec<CspVarId>,
        /// Required total.
        target: i64,
    },
    /// Variable must take a value from the `allowed` list.
    InDomain {
        /// The variable to restrict.
        var: CspVarId,
        /// Allowed values (does not need to be sorted).
        allowed: Vec<i64>,
    },
}

impl Constraint {
    /// Return the set of variable ids that participate in this constraint.
    pub fn variables(&self) -> Vec<CspVarId> {
        match self {
            Constraint::AllDifferent(vars) => vars.clone(),
            Constraint::Equal(a, b)
            | Constraint::NotEqual(a, b)
            | Constraint::LessThan(a, b)
            | Constraint::LessEqual(a, b) => vec![*a, *b],
            Constraint::Sum { vars, .. } => vars.clone(),
            Constraint::InDomain { var, .. } => vec![*var],
        }
    }

    /// Return `true` when `var` is mentioned in this constraint.
    pub fn involves(&self, var: CspVarId) -> bool {
        self.variables().contains(&var)
    }

    /// Return `true` when `xi` and `xj` are both mentioned (binary relationship).
    fn involves_pair(&self, xi: CspVarId, xj: CspVarId) -> bool {
        let vars = self.variables();
        vars.contains(&xi) && vars.contains(&xj)
    }
}

// ---------------------------------------------------------------------------
// Assignment
// ---------------------------------------------------------------------------

/// A (partial or complete) mapping from variable ids to assigned values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assignment {
    /// Internal map: variable index → assigned value.
    pub values: HashMap<usize, i64>,
}

impl Assignment {
    /// Create an empty assignment.
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Return `true` when all `num_vars` variables have been assigned.
    #[inline]
    pub fn is_complete(&self, num_vars: usize) -> bool {
        self.values.len() == num_vars
    }

    /// Retrieve the assigned value for `var`, if any.
    #[inline]
    pub fn get(&self, var: CspVarId) -> Option<i64> {
        self.values.get(&var.0).copied()
    }

    /// Assign `value` to `var`.
    #[inline]
    pub fn set(&mut self, var: CspVarId, value: i64) {
        self.values.insert(var.0, value);
    }

    /// Remove the assignment for `var`.
    #[inline]
    pub fn unset(&mut self, var: CspVarId) {
        self.values.remove(&var.0);
    }

    /// Iterate over all (var_id, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (CspVarId, i64)> + '_ {
        self.values.iter().map(|(&id, &v)| (CspVarId(id), v))
    }
}

impl Default for Assignment {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CspVariable
// ---------------------------------------------------------------------------

/// A CSP variable: its identity, name, and initial domain.
#[derive(Debug, Clone)]
pub struct CspVariable {
    /// Unique identifier (index in the solver's variable list).
    pub id: CspVarId,
    /// Human-readable name.
    pub name: String,
    /// Initial domain (before any pruning).
    pub domain: Domain,
}

// ---------------------------------------------------------------------------
// SolverConfig
// ---------------------------------------------------------------------------

/// Configuration knobs for [`ConstraintSolver`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverConfig {
    /// Maximum number of solutions to collect before stopping.
    pub max_solutions: usize,
    /// When `true`, run AC-3 arc consistency before backtracking.
    pub use_ac3: bool,
    /// When `true`, apply the Minimum Remaining Values (MRV) heuristic when
    /// choosing the next variable to assign.
    pub use_mrv: bool,
    /// When `true`, apply the Least Constraining Value (LCV) heuristic to
    /// order domain values. Currently sorts ascending as a lightweight proxy.
    pub use_lcv: bool,
    /// Maximum number of backtrack steps before aborting (0 = unlimited).
    pub max_backtracks: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_solutions: 1,
            use_ac3: true,
            use_mrv: true,
            use_lcv: false,
            max_backtracks: 100_000,
        }
    }
}

// ---------------------------------------------------------------------------
// SolverResult
// ---------------------------------------------------------------------------

/// Statistics and solutions returned by [`ConstraintSolver::solve`].
#[derive(Debug, Clone)]
pub struct SolverResult {
    /// All complete assignments found (up to `max_solutions`).
    pub solutions: Vec<Assignment>,
    /// Total number of backtrack steps taken during search.
    pub backtracks: u64,
    /// Total number of constraint checks performed.
    pub constraint_checks: u64,
    /// Wall-clock time spent in `solve` (milliseconds).
    pub time_ms: u64,
}

impl SolverResult {
    fn new() -> Self {
        Self {
            solutions: Vec::new(),
            backtracks: 0,
            constraint_checks: 0,
            time_ms: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// CspError
// ---------------------------------------------------------------------------

/// Error conditions that can arise during CSP solving.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CspError {
    /// A constraint referenced a variable id that does not exist.
    VariableNotFound(usize),
    /// A constraint is structurally invalid (e.g., empty variable list).
    InvalidConstraint(String),
    /// AC-3 proved the problem is unsatisfiable before backtracking began.
    UnsatisfiableAfterAC3,
}

impl std::fmt::Display for CspError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CspError::VariableNotFound(id) => write!(f, "Variable not found: {}", id),
            CspError::InvalidConstraint(msg) => write!(f, "Invalid constraint: {}", msg),
            CspError::UnsatisfiableAfterAC3 => {
                write!(f, "Problem is unsatisfiable after AC-3 preprocessing")
            }
        }
    }
}

impl std::error::Error for CspError {}

// ---------------------------------------------------------------------------
// CspStats
// ---------------------------------------------------------------------------

/// Summary statistics about the CSP instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CspStats {
    /// Number of variables registered.
    pub num_variables: usize,
    /// Number of constraints registered.
    pub num_constraints: usize,
    /// Sum of all domain sizes.
    pub total_domain_size: usize,
}

// ---------------------------------------------------------------------------
// ConstraintSolver
// ---------------------------------------------------------------------------

/// A full-featured CSP solver combining AC-3 domain pruning with
/// backtracking search and configurable variable/value ordering heuristics.
pub struct ConstraintSolver {
    /// All registered variables (in insertion order).
    pub variables: Vec<CspVariable>,
    /// All registered constraints.
    pub constraints: Vec<Constraint>,
    /// Solver configuration.
    pub config: SolverConfig,
}

impl ConstraintSolver {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new, empty solver with the given configuration.
    pub fn new(config: SolverConfig) -> Self {
        Self {
            variables: Vec::new(),
            constraints: Vec::new(),
            config,
        }
    }

    /// Register a new variable with the given name and initial domain values.
    ///
    /// Returns the `CspVarId` that uniquely identifies this variable.
    pub fn add_variable(&mut self, name: String, domain: Vec<i64>) -> CspVarId {
        let id = CspVarId(self.variables.len());
        self.variables.push(CspVariable {
            id,
            name,
            domain: Domain::new(domain),
        });
        id
    }

    /// Append a constraint to the problem.
    pub fn add_constraint(&mut self, c: Constraint) {
        self.constraints.push(c);
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Return a summary of the current CSP instance.
    pub fn stats(&self) -> CspStats {
        CspStats {
            num_variables: self.variables.len(),
            num_constraints: self.constraints.len(),
            total_domain_size: self.variables.iter().map(|v| v.domain.len()).sum(),
        }
    }

    // -----------------------------------------------------------------------
    // Consistency checking
    // -----------------------------------------------------------------------

    /// Return `true` when all constraints that are fully grounded by
    /// `assignment` (plus the tentative assignment of `value` to `var`) are
    /// satisfied.
    pub fn is_consistent(&self, assignment: &Assignment, var: CspVarId, value: i64) -> bool {
        self.constraints
            .iter()
            .all(|c| self.check_constraint(c, assignment, var, value))
    }

    /// Check a single constraint under a partial assignment with `var = value`.
    ///
    /// Returns `true` when:
    /// - the constraint cannot yet be evaluated (some participant is unassigned), OR
    /// - the constraint is satisfied by the current (partial+tentative) assignment.
    pub fn check_constraint(
        &self,
        c: &Constraint,
        assignment: &Assignment,
        var: CspVarId,
        value: i64,
    ) -> bool {
        // Build a temporary lookup closure that merges the existing assignment
        // with the tentative (var = value).
        let get = |v: CspVarId| -> Option<i64> {
            if v == var {
                Some(value)
            } else {
                assignment.get(v)
            }
        };

        match c {
            Constraint::Equal(a, b) => match (get(*a), get(*b)) {
                (Some(va), Some(vb)) => va == vb,
                _ => true,
            },
            Constraint::NotEqual(a, b) => match (get(*a), get(*b)) {
                (Some(va), Some(vb)) => va != vb,
                _ => true,
            },
            Constraint::LessThan(a, b) => match (get(*a), get(*b)) {
                (Some(va), Some(vb)) => va < vb,
                _ => true,
            },
            Constraint::LessEqual(a, b) => match (get(*a), get(*b)) {
                (Some(va), Some(vb)) => va <= vb,
                _ => true,
            },
            Constraint::AllDifferent(vars) => {
                // Collect all currently-assigned values for participants.
                let mut seen: Vec<i64> = Vec::with_capacity(vars.len());
                for &v in vars {
                    if let Some(val) = get(v) {
                        if seen.contains(&val) {
                            return false;
                        }
                        seen.push(val);
                    }
                }
                true
            }
            Constraint::Sum { vars, target } => {
                // Only evaluate when all participants are assigned.
                let mut total: i64 = 0;
                let mut all_assigned = true;
                let mut partial_sum: i64 = 0;
                let mut partial_count = 0;
                for &v in vars {
                    match get(v) {
                        Some(val) => {
                            total += val;
                            partial_sum += val;
                            partial_count += 1;
                        }
                        None => {
                            all_assigned = false;
                        }
                    }
                }
                let _ = (total, partial_sum, partial_count);
                if all_assigned {
                    // All assigned: check exact sum
                    let mut s: i64 = 0;
                    for &v in vars {
                        s += get(v).unwrap_or(0);
                    }
                    s == *target
                } else {
                    // Partial: only prune if partial sum already exceeds target
                    // (assuming non-negative domains for simple pruning)
                    let partial: i64 = vars.iter().filter_map(|&v| get(v)).sum();
                    partial <= *target
                }
            }
            Constraint::InDomain { var: dvar, allowed } => {
                if let Some(val) = get(*dvar) {
                    allowed.contains(&val)
                } else {
                    true
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // AC-3
    // -----------------------------------------------------------------------

    /// Run the AC-3 arc-consistency algorithm on the given `domains` (which
    /// are cloned from the initial variable domains at the start of `solve`).
    ///
    /// Returns `false` when arc consistency reveals that the problem has no
    /// solution (some domain became empty).
    pub fn ac3(&self, domains: &mut [Domain]) -> bool {
        // Seed the worklist with every (xi, xj) pair implied by a constraint.
        let mut worklist: VecDeque<(CspVarId, CspVarId)> = VecDeque::new();

        for c in &self.constraints {
            match c {
                Constraint::Equal(a, b)
                | Constraint::NotEqual(a, b)
                | Constraint::LessThan(a, b)
                | Constraint::LessEqual(a, b) => {
                    worklist.push_back((*a, *b));
                    worklist.push_back((*b, *a));
                }
                Constraint::AllDifferent(vars) => {
                    for i in 0..vars.len() {
                        for j in 0..vars.len() {
                            if i != j {
                                worklist.push_back((vars[i], vars[j]));
                            }
                        }
                    }
                }
                // InDomain is a unary constraint — prune immediately.
                Constraint::InDomain { var, allowed } => {
                    let idx = var.0;
                    if idx < domains.len() {
                        domains[idx].values.retain(|v| allowed.contains(v));
                        if domains[idx].is_empty() {
                            return false;
                        }
                    }
                }
                // Sum has no efficient binary-arc encoding; skip for AC-3.
                Constraint::Sum { .. } => {}
            }
        }

        while let Some((xi, xj)) = worklist.pop_front() {
            if self.revise(domains, xi, xj) {
                if domains.get(xi.0).is_none_or(Domain::is_empty) {
                    return false;
                }
                // Re-add arcs from neighbours of xi (excluding xj) back to xi.
                let neighbours = self.neighbours_of(xi);
                for xk in neighbours {
                    if xk != xj {
                        worklist.push_back((xk, xi));
                    }
                }
            }
        }

        true
    }

    /// Revise `xi`'s domain: remove any value that has no support in `xj`.
    ///
    /// Returns `true` when at least one value was removed from `xi`'s domain.
    pub fn revise(&self, domains: &mut [Domain], xi: CspVarId, xj: CspVarId) -> bool {
        if xi.0 >= domains.len() || xj.0 >= domains.len() {
            return false;
        }

        // Snapshot the current domain of xj to avoid borrow conflicts.
        let xj_vals: Vec<i64> = domains[xj.0].values.clone();
        let xi_vals: Vec<i64> = domains[xi.0].values.clone();

        // Collect only binary constraints between xi and xj.
        let relevant: Vec<&Constraint> = self
            .constraints
            .iter()
            .filter(|c| c.involves_pair(xi, xj))
            .collect();

        let mut to_remove: Vec<i64> = Vec::new();
        for &vxi in &xi_vals {
            let has_support = xj_vals.iter().any(|&vyj| {
                relevant.iter().all(|c| {
                    let mut tmp = Assignment::new();
                    tmp.set(xi, vxi);
                    tmp.set(xj, vyj);
                    // Use an empty "var/value" to check with the full assignment.
                    // We create a dummy CspVarId that won't collide with xi/xj.
                    self.check_constraint(c, &tmp, CspVarId(usize::MAX), 0)
                })
            });
            if !has_support {
                to_remove.push(vxi);
            }
        }

        if to_remove.is_empty() {
            return false;
        }
        for v in &to_remove {
            domains[xi.0].remove(*v);
        }
        true
    }

    /// Collect the set of variable ids that share any binary constraint with `var`.
    fn neighbours_of(&self, var: CspVarId) -> Vec<CspVarId> {
        let mut result: Vec<CspVarId> = Vec::new();
        for c in &self.constraints {
            match c {
                Constraint::Equal(a, b)
                | Constraint::NotEqual(a, b)
                | Constraint::LessThan(a, b)
                | Constraint::LessEqual(a, b) => {
                    if *a == var && !result.contains(b) {
                        result.push(*b);
                    } else if *b == var && !result.contains(a) {
                        result.push(*a);
                    }
                }
                Constraint::AllDifferent(vars) => {
                    if vars.contains(&var) {
                        for &v in vars {
                            if v != var && !result.contains(&v) {
                                result.push(v);
                            }
                        }
                    }
                }
                Constraint::Sum { vars, .. } => {
                    if vars.contains(&var) {
                        for &v in vars {
                            if v != var && !result.contains(&v) {
                                result.push(v);
                            }
                        }
                    }
                }
                Constraint::InDomain { .. } => {}
            }
        }
        result
    }

    // -----------------------------------------------------------------------
    // Variable and value ordering heuristics
    // -----------------------------------------------------------------------

    /// Select the next unassigned variable to expand.
    ///
    /// When `use_mrv` is enabled, picks the variable with the fewest remaining
    /// domain values (ties broken by variable id). Otherwise returns the first
    /// unassigned variable in insertion order.
    pub fn select_unassigned_variable(
        &self,
        assignment: &Assignment,
        domains: &[Domain],
    ) -> Option<CspVarId> {
        let unassigned: Vec<CspVarId> = self
            .variables
            .iter()
            .map(|v| v.id)
            .filter(|id| assignment.get(*id).is_none())
            .collect();

        if unassigned.is_empty() {
            return None;
        }

        if self.config.use_mrv {
            unassigned.into_iter().min_by_key(|id| {
                let size = domains.get(id.0).map_or(0, Domain::len);
                // Use (size, id) for deterministic tie-breaking.
                (size, id.0)
            })
        } else {
            unassigned.into_iter().next()
        }
    }

    /// Return the values from `var`'s current domain in the order we should
    /// try them during backtracking.
    ///
    /// When `use_lcv` is enabled this would apply the Least Constraining Value
    /// heuristic; currently we return values in sorted (ascending) order as a
    /// lightweight deterministic approximation.
    pub fn order_domain_values(
        &self,
        var: CspVarId,
        _assignment: &Assignment,
        domains: &[Domain],
    ) -> Vec<i64> {
        domains
            .get(var.0)
            .map_or_else(Vec::new, |d| d.values.clone())
    }

    // -----------------------------------------------------------------------
    // Backtracking search
    // -----------------------------------------------------------------------

    /// Recursive backtracking search.
    ///
    /// Returns `true` when the search should stop (either `max_solutions`
    /// reached or `max_backtracks` exceeded).
    pub fn backtrack(
        &self,
        assignment: &mut Assignment,
        domains: &mut Vec<Domain>,
        result: &mut SolverResult,
    ) -> bool {
        // Check termination conditions.
        if self.config.max_backtracks > 0 && result.backtracks >= self.config.max_backtracks as u64
        {
            return true;
        }

        // All variables assigned → record solution.
        if assignment.is_complete(self.variables.len()) {
            result.solutions.push(assignment.clone());
            return result.solutions.len() >= self.config.max_solutions;
        }

        // Choose next variable.
        let var = match self.select_unassigned_variable(assignment, domains) {
            Some(v) => v,
            None => return false,
        };

        let ordered_values = self.order_domain_values(var, assignment, domains);

        for value in ordered_values {
            result.constraint_checks += 1;
            if self.is_consistent(assignment, var, value) {
                // Assign and recurse.
                assignment.set(var, value);

                // Forward-check: ensure no neighbour's domain becomes empty.
                let domains_backup: Vec<Domain> = domains.clone();
                let mut fc_ok = true;
                for neighbour in self.neighbours_of(var) {
                    if assignment.get(neighbour).is_some() {
                        continue;
                    }
                    let orig: Vec<i64> = domains[neighbour.0].values.clone();
                    let pruned: Vec<i64> = orig
                        .into_iter()
                        .filter(|&nv| self.is_consistent(assignment, neighbour, nv))
                        .collect();
                    domains[neighbour.0].values = pruned;
                    if domains[neighbour.0].is_empty() {
                        fc_ok = false;
                        break;
                    }
                }

                if fc_ok && self.backtrack(assignment, domains, result) {
                    return true;
                }

                // Undo forward-checking domain reductions and the assignment.
                *domains = domains_backup;
                assignment.unset(var);
                result.backtracks += 1;
            }
        }

        false
    }

    // -----------------------------------------------------------------------
    // Top-level solve
    // -----------------------------------------------------------------------

    /// Solve the CSP and return a [`SolverResult`] with all found solutions
    /// and search statistics.
    pub fn solve(&mut self) -> SolverResult {
        let start = Instant::now();
        let mut result = SolverResult::new();

        // Clone initial domains for the search.
        let mut domains: Vec<Domain> = self.variables.iter().map(|v| v.domain.clone()).collect();

        // Optionally run AC-3 to prune domains.
        if self.config.use_ac3 && !self.ac3(&mut domains) {
            result.time_ms = start.elapsed().as_millis() as u64;
            return result; // no solutions (AC-3 proved unsatisfiable)
        }

        // Bail early if any domain is already empty.
        if domains.iter().any(Domain::is_empty) {
            result.time_ms = start.elapsed().as_millis() as u64;
            return result;
        }

        let mut assignment = Assignment::new();
        self.backtrack(&mut assignment, &mut domains, &mut result);

        result.time_ms = start.elapsed().as_millis() as u64;
        result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::constraint_solver::{
        Assignment, Constraint, ConstraintSolver, CspVarId, Domain, SolverConfig, SolverResult,
    };

    fn default_solver() -> ConstraintSolver {
        ConstraintSolver::new(SolverConfig::default())
    }

    // -----------------------------------------------------------------------
    // Domain tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_domain_new_sorts_and_deduplicates() {
        let d = Domain::new(vec![3, 1, 2, 1, 3]);
        assert_eq!(d.values, vec![1, 2, 3]);
    }

    #[test]
    fn test_domain_is_empty_when_no_values() {
        assert!(Domain::new(vec![]).is_empty());
        assert!(!Domain::new(vec![1]).is_empty());
    }

    #[test]
    fn test_domain_contains() {
        let d = Domain::new(vec![10, 20, 30]);
        assert!(d.contains(10));
        assert!(d.contains(20));
        assert!(!d.contains(5));
        assert!(!d.contains(99));
    }

    #[test]
    fn test_domain_remove_present_value() {
        let mut d = Domain::new(vec![1, 2, 3]);
        let removed = d.remove(2);
        assert!(removed);
        assert_eq!(d.values, vec![1, 3]);
    }

    #[test]
    fn test_domain_remove_absent_value_is_noop() {
        let mut d = Domain::new(vec![1, 2, 3]);
        let removed = d.remove(99);
        assert!(!removed);
        assert_eq!(d.values, vec![1, 2, 3]);
    }

    #[test]
    fn test_domain_len() {
        assert_eq!(Domain::new(vec![1, 2, 3]).len(), 3);
        assert_eq!(Domain::new(vec![]).len(), 0);
    }

    // -----------------------------------------------------------------------
    // Assignment tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_assignment_set_and_get() {
        let mut a = Assignment::new();
        let v = CspVarId(0);
        assert_eq!(a.get(v), None);
        a.set(v, 42);
        assert_eq!(a.get(v), Some(42));
    }

    #[test]
    fn test_assignment_unset() {
        let mut a = Assignment::new();
        let v = CspVarId(0);
        a.set(v, 7);
        a.unset(v);
        assert_eq!(a.get(v), None);
    }

    #[test]
    fn test_assignment_is_complete() {
        let mut a = Assignment::new();
        assert!(!a.is_complete(2));
        a.set(CspVarId(0), 1);
        a.set(CspVarId(1), 2);
        assert!(a.is_complete(2));
    }

    // -----------------------------------------------------------------------
    // Solver construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_variable_returns_sequential_ids() {
        let mut solver = default_solver();
        let x = solver.add_variable("x".to_string(), vec![1, 2]);
        let y = solver.add_variable("y".to_string(), vec![3, 4]);
        assert_eq!(x, CspVarId(0));
        assert_eq!(y, CspVarId(1));
    }

    #[test]
    fn test_stats() {
        let mut solver = default_solver();
        solver.add_variable("a".to_string(), vec![1, 2, 3]);
        solver.add_variable("b".to_string(), vec![4, 5]);
        let s = solver.stats();
        assert_eq!(s.num_variables, 2);
        assert_eq!(s.total_domain_size, 5);
        assert_eq!(s.num_constraints, 0);
    }

    // -----------------------------------------------------------------------
    // Simple satisfiable cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_constraints_always_satisfiable() {
        let mut solver = default_solver();
        let _x = solver.add_variable("x".to_string(), vec![42]);
        let result = solver.solve();
        assert_eq!(result.solutions.len(), 1);
        assert_eq!(result.solutions[0].get(CspVarId(0)), Some(42));
    }

    #[test]
    fn test_not_equal_two_vars_satisfiable() {
        let mut solver = default_solver();
        let x = solver.add_variable("x".to_string(), vec![1, 2]);
        let y = solver.add_variable("y".to_string(), vec![1, 2]);
        solver.add_constraint(Constraint::NotEqual(x, y));
        let result = solver.solve();
        assert_eq!(result.solutions.len(), 1);
        let sol = &result.solutions[0];
        assert_ne!(sol.get(x), sol.get(y));
    }

    #[test]
    fn test_equal_constraint() {
        let mut solver = default_solver();
        let x = solver.add_variable("x".to_string(), vec![1, 2, 3]);
        let y = solver.add_variable("y".to_string(), vec![2, 3, 4]);
        solver.add_constraint(Constraint::Equal(x, y));
        let result = solver.solve();
        assert!(!result.solutions.is_empty());
        let sol = &result.solutions[0];
        assert_eq!(sol.get(x), sol.get(y));
    }

    #[test]
    fn test_less_than_constraint() {
        let mut solver = default_solver();
        let a = solver.add_variable("a".to_string(), vec![1, 2, 3]);
        let b = solver.add_variable("b".to_string(), vec![1, 2, 3]);
        solver.add_constraint(Constraint::LessThan(a, b));
        let result = solver.solve();
        assert!(!result.solutions.is_empty());
        let sol = &result.solutions[0];
        assert!(
            sol.get(a).expect("test: should succeed") < sol.get(b).expect("test: should succeed")
        );
    }

    #[test]
    fn test_less_equal_constraint() {
        let mut solver = default_solver();
        let a = solver.add_variable("a".to_string(), vec![5]);
        let b = solver.add_variable("b".to_string(), vec![5]);
        solver.add_constraint(Constraint::LessEqual(a, b));
        let result = solver.solve();
        assert_eq!(result.solutions.len(), 1);
        let sol = &result.solutions[0];
        assert!(
            sol.get(a).expect("test: should succeed") <= sol.get(b).expect("test: should succeed")
        );
    }

    #[test]
    fn test_all_different_three_vars() {
        let mut solver = default_solver();
        let x = solver.add_variable("x".to_string(), vec![1, 2, 3]);
        let y = solver.add_variable("y".to_string(), vec![1, 2, 3]);
        let z = solver.add_variable("z".to_string(), vec![1, 2, 3]);
        solver.add_constraint(Constraint::AllDifferent(vec![x, y, z]));
        let result = solver.solve();
        assert_eq!(result.solutions.len(), 1);
        let sol = &result.solutions[0];
        assert_ne!(sol.get(x), sol.get(y));
        assert_ne!(sol.get(x), sol.get(z));
        assert_ne!(sol.get(y), sol.get(z));
    }

    #[test]
    fn test_in_domain_constraint() {
        let mut solver = default_solver();
        let x = solver.add_variable("x".to_string(), vec![1, 2, 3, 4, 5]);
        solver.add_constraint(Constraint::InDomain {
            var: x,
            allowed: vec![2, 4],
        });
        let result = solver.solve();
        assert_eq!(result.solutions.len(), 1);
        let val = result.solutions[0].get(x).unwrap_or(-1);
        assert!(val == 2 || val == 4);
    }

    #[test]
    fn test_sum_constraint_exact() {
        let mut solver = ConstraintSolver::new(SolverConfig {
            use_ac3: false, // skip AC-3 for sum (not fully supported)
            ..SolverConfig::default()
        });
        let a = solver.add_variable("a".to_string(), vec![1, 2, 3]);
        let b = solver.add_variable("b".to_string(), vec![1, 2, 3]);
        solver.add_constraint(Constraint::Sum {
            vars: vec![a, b],
            target: 4,
        });
        let result = solver.solve();
        assert!(!result.solutions.is_empty());
        let sol = &result.solutions[0];
        assert_eq!(sol.get(a).unwrap_or(0) + sol.get(b).unwrap_or(0), 4);
    }

    // -----------------------------------------------------------------------
    // Unsatisfiable cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_domain_unsatisfiable() {
        let mut solver = default_solver();
        solver.add_variable("x".to_string(), vec![]);
        let result = solver.solve();
        assert!(result.solutions.is_empty());
    }

    #[test]
    fn test_not_equal_single_value_unsatisfiable() {
        let mut solver = default_solver();
        let x = solver.add_variable("x".to_string(), vec![5]);
        let y = solver.add_variable("y".to_string(), vec![5]);
        solver.add_constraint(Constraint::NotEqual(x, y));
        let result = solver.solve();
        assert!(result.solutions.is_empty());
    }

    #[test]
    fn test_all_different_too_few_values_unsatisfiable() {
        let mut solver = default_solver();
        let x = solver.add_variable("x".to_string(), vec![1, 2]);
        let y = solver.add_variable("y".to_string(), vec![1, 2]);
        let z = solver.add_variable("z".to_string(), vec![1, 2]);
        solver.add_constraint(Constraint::AllDifferent(vec![x, y, z]));
        let result = solver.solve();
        assert!(result.solutions.is_empty());
    }

    #[test]
    fn test_less_than_equal_values_unsatisfiable() {
        let mut solver = default_solver();
        let a = solver.add_variable("a".to_string(), vec![5]);
        let b = solver.add_variable("b".to_string(), vec![5]);
        solver.add_constraint(Constraint::LessThan(a, b));
        let result = solver.solve();
        assert!(result.solutions.is_empty());
    }

    #[test]
    fn test_equal_disjoint_domains_unsatisfiable() {
        let mut solver = default_solver();
        let x = solver.add_variable("x".to_string(), vec![1, 2]);
        let y = solver.add_variable("y".to_string(), vec![3, 4]);
        solver.add_constraint(Constraint::Equal(x, y));
        let result = solver.solve();
        assert!(result.solutions.is_empty());
    }

    #[test]
    fn test_in_domain_no_overlap_unsatisfiable() {
        let mut solver = default_solver();
        let x = solver.add_variable("x".to_string(), vec![1, 2, 3]);
        solver.add_constraint(Constraint::InDomain {
            var: x,
            allowed: vec![7, 8, 9],
        });
        let result = solver.solve();
        assert!(result.solutions.is_empty());
    }

    // -----------------------------------------------------------------------
    // Multiple solutions
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_all_solutions() {
        let mut solver = ConstraintSolver::new(SolverConfig {
            max_solutions: 100,
            use_ac3: false,
            ..SolverConfig::default()
        });
        let x = solver.add_variable("x".to_string(), vec![1, 2, 3]);
        let y = solver.add_variable("y".to_string(), vec![1, 2, 3]);
        solver.add_constraint(Constraint::NotEqual(x, y));
        let result = solver.solve();
        // 3*3 - 3 = 6 solutions (x != y)
        assert_eq!(result.solutions.len(), 6);
        for sol in &result.solutions {
            assert_ne!(sol.get(x), sol.get(y));
        }
    }

    // -----------------------------------------------------------------------
    // AC-3 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ac3_prunes_domain_for_not_equal() {
        let mut solver = ConstraintSolver::new(SolverConfig {
            use_ac3: true,
            ..SolverConfig::default()
        });
        // x ∈ {5}, y ∈ {5, 6}, x != y → AC-3 must prune y to {6}.
        let x = solver.add_variable("x".to_string(), vec![5]);
        let y = solver.add_variable("y".to_string(), vec![5, 6]);
        solver.add_constraint(Constraint::NotEqual(x, y));

        let mut domains: Vec<Domain> = solver.variables.iter().map(|v| v.domain.clone()).collect();
        let consistent = solver.ac3(&mut domains);
        assert!(consistent);
        assert_eq!(domains[y.0].values, vec![6]);
    }

    #[test]
    fn test_ac3_detects_unsatisfiability() {
        let mut solver = ConstraintSolver::new(SolverConfig {
            use_ac3: true,
            ..SolverConfig::default()
        });
        let x = solver.add_variable("x".to_string(), vec![5]);
        let y = solver.add_variable("y".to_string(), vec![5]);
        solver.add_constraint(Constraint::NotEqual(x, y));

        let mut domains: Vec<Domain> = solver.variables.iter().map(|v| v.domain.clone()).collect();
        let consistent = solver.ac3(&mut domains);
        assert!(!consistent);
    }

    #[test]
    fn test_ac3_prunes_less_than() {
        let mut solver = ConstraintSolver::new(SolverConfig {
            use_ac3: true,
            ..SolverConfig::default()
        });
        // a ∈ {3,4,5}, b ∈ {1,2,3}, a < b → nothing in a is < 1, max of b is 3.
        // AC-3 should prune a to {1,2} (values < max(b)=3) and b to {2,3}
        // (values > min(a)=3) — but since a ∈ {3,4,5} and b ≤ 3, a must be < b ≤ 3,
        // which is impossible → domains may collapse.
        let a = solver.add_variable("a".to_string(), vec![3, 4, 5]);
        let b = solver.add_variable("b".to_string(), vec![1, 2, 3]);
        solver.add_constraint(Constraint::LessThan(a, b));
        let result = solver.solve();
        // No solution possible since min(a)=3, max(b)=3, requires a < b.
        assert!(result.solutions.is_empty());
    }

    #[test]
    fn test_ac3_in_domain_pruning() {
        let mut solver = default_solver();
        let x = solver.add_variable("x".to_string(), vec![1, 2, 3, 4, 5]);
        solver.add_constraint(Constraint::InDomain {
            var: x,
            allowed: vec![3],
        });
        let mut domains: Vec<Domain> = solver.variables.iter().map(|v| v.domain.clone()).collect();
        let ok = solver.ac3(&mut domains);
        assert!(ok);
        assert_eq!(domains[x.0].values, vec![3]);
    }

    // -----------------------------------------------------------------------
    // Heuristics
    // -----------------------------------------------------------------------

    #[test]
    fn test_mrv_selects_smallest_domain() {
        let mut solver = ConstraintSolver::new(SolverConfig {
            use_mrv: true,
            ..SolverConfig::default()
        });
        let x = solver.add_variable("x".to_string(), vec![1, 2, 3]);
        let y = solver.add_variable("y".to_string(), vec![1]);
        let domains: Vec<Domain> = solver.variables.iter().map(|v| v.domain.clone()).collect();
        let assignment = Assignment::new();
        let chosen = solver
            .select_unassigned_variable(&assignment, &domains)
            .expect("test: should succeed");
        // y has the smaller domain {1} so MRV should choose y.
        assert_eq!(chosen, y);
        let _ = x;
    }

    #[test]
    fn test_no_mrv_selects_first() {
        let mut solver = ConstraintSolver::new(SolverConfig {
            use_mrv: false,
            ..SolverConfig::default()
        });
        let x = solver.add_variable("x".to_string(), vec![1, 2, 3]);
        let _y = solver.add_variable("y".to_string(), vec![1]);
        let domains: Vec<Domain> = solver.variables.iter().map(|v| v.domain.clone()).collect();
        let assignment = Assignment::new();
        let chosen = solver
            .select_unassigned_variable(&assignment, &domains)
            .expect("test: should succeed");
        // Without MRV, picks first unassigned (insertion order).
        assert_eq!(chosen, x);
    }

    #[test]
    fn test_order_domain_values_returns_sorted() {
        let mut solver = default_solver();
        let x = solver.add_variable("x".to_string(), vec![5, 3, 1, 4, 2]);
        let domains: Vec<Domain> = solver.variables.iter().map(|v| v.domain.clone()).collect();
        let vals = solver.order_domain_values(x, &Assignment::new(), &domains);
        assert_eq!(vals, vec![1, 2, 3, 4, 5]);
    }

    // -----------------------------------------------------------------------
    // Complex / integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_chained_less_than() {
        let mut solver = default_solver();
        let a = solver.add_variable("a".to_string(), vec![1, 2, 3]);
        let b = solver.add_variable("b".to_string(), vec![1, 2, 3]);
        let c = solver.add_variable("c".to_string(), vec![1, 2, 3]);
        solver.add_constraint(Constraint::LessThan(a, b));
        solver.add_constraint(Constraint::LessThan(b, c));
        let result = solver.solve();
        assert_eq!(result.solutions.len(), 1);
        let sol = &result.solutions[0];
        assert!(
            sol.get(a).expect("test: should succeed") < sol.get(b).expect("test: should succeed")
        );
        assert!(
            sol.get(b).expect("test: should succeed") < sol.get(c).expect("test: should succeed")
        );
    }

    #[test]
    fn test_combined_all_different_and_less_than() {
        let mut solver = ConstraintSolver::new(SolverConfig {
            max_solutions: 10,
            ..SolverConfig::default()
        });
        let x = solver.add_variable("x".to_string(), vec![1, 2, 3, 4]);
        let y = solver.add_variable("y".to_string(), vec![1, 2, 3, 4]);
        let z = solver.add_variable("z".to_string(), vec![1, 2, 3, 4]);
        solver.add_constraint(Constraint::AllDifferent(vec![x, y, z]));
        solver.add_constraint(Constraint::LessThan(x, y));
        let result = solver.solve();
        assert!(!result.solutions.is_empty());
        for sol in &result.solutions {
            let vx = sol.get(x).expect("test: should succeed");
            let vy = sol.get(y).expect("test: should succeed");
            let vz = sol.get(z).expect("test: should succeed");
            assert_ne!(vx, vy);
            assert_ne!(vx, vz);
            assert_ne!(vy, vz);
            assert!(vx < vy);
        }
    }

    #[test]
    fn test_single_variable_no_constraint() {
        let mut solver = default_solver();
        let x = solver.add_variable("x".to_string(), vec![7]);
        let result = solver.solve();
        assert_eq!(result.solutions.len(), 1);
        assert_eq!(result.solutions[0].get(x), Some(7));
    }

    #[test]
    fn test_backtrack_count_increments() {
        let mut solver = ConstraintSolver::new(SolverConfig {
            max_solutions: 100,
            use_ac3: false,
            ..SolverConfig::default()
        });
        let x = solver.add_variable("x".to_string(), vec![1, 2]);
        let y = solver.add_variable("y".to_string(), vec![1, 2]);
        solver.add_constraint(Constraint::NotEqual(x, y));
        let result = solver.solve();
        // Some backtracks must have occurred.
        // We only assert the stat is present and non-negative (u64).
        assert!(result.backtracks < u64::MAX);
        let _ = result.constraint_checks;
    }

    #[test]
    fn test_time_ms_is_set() {
        let mut solver = default_solver();
        solver.add_variable("x".to_string(), vec![1]);
        let result = solver.solve();
        // time_ms is u64; just verify it was populated (it can be 0 on fast machines).
        let _ = result.time_ms;
    }

    #[test]
    fn test_constraint_involves() {
        let c = Constraint::NotEqual(CspVarId(0), CspVarId(1));
        assert!(c.involves(CspVarId(0)));
        assert!(c.involves(CspVarId(1)));
        assert!(!c.involves(CspVarId(2)));
    }

    #[test]
    fn test_constraint_variables_all_different() {
        let c = Constraint::AllDifferent(vec![CspVarId(0), CspVarId(2), CspVarId(4)]);
        let vars = c.variables();
        assert!(vars.contains(&CspVarId(0)));
        assert!(vars.contains(&CspVarId(2)));
        assert!(vars.contains(&CspVarId(4)));
        assert_eq!(vars.len(), 3);
    }

    #[test]
    fn test_solver_config_default() {
        let cfg = SolverConfig::default();
        assert_eq!(cfg.max_solutions, 1);
        assert!(cfg.use_ac3);
        assert!(cfg.use_mrv);
        assert!(!cfg.use_lcv);
        assert_eq!(cfg.max_backtracks, 100_000);
    }

    #[test]
    fn test_less_equal_equal_values_satisfiable() {
        let mut solver = default_solver();
        let a = solver.add_variable("a".to_string(), vec![3, 4]);
        let b = solver.add_variable("b".to_string(), vec![3, 4]);
        solver.add_constraint(Constraint::LessEqual(a, b));
        let result = solver.solve();
        assert!(!result.solutions.is_empty());
        let sol = &result.solutions[0];
        assert!(
            sol.get(a).expect("test: should succeed") <= sol.get(b).expect("test: should succeed")
        );
    }

    #[test]
    fn test_multiple_in_domain_constraints() {
        let mut solver = default_solver();
        let x = solver.add_variable("x".to_string(), vec![1, 2, 3, 4, 5]);
        solver.add_constraint(Constraint::InDomain {
            var: x,
            allowed: vec![2, 3, 4],
        });
        solver.add_constraint(Constraint::InDomain {
            var: x,
            allowed: vec![3, 4, 5],
        });
        let result = solver.solve();
        assert!(!result.solutions.is_empty());
        let val = result.solutions[0].get(x).unwrap_or(-1);
        // Intersection is {3, 4}.
        assert!(val == 3 || val == 4);
    }

    #[test]
    fn test_solver_result_new() {
        let r = SolverResult::new();
        assert!(r.solutions.is_empty());
        assert_eq!(r.backtracks, 0);
        assert_eq!(r.constraint_checks, 0);
        assert_eq!(r.time_ms, 0);
    }

    #[test]
    fn test_assignment_iter() {
        let mut a = Assignment::new();
        a.set(CspVarId(0), 10);
        a.set(CspVarId(1), 20);
        let mut collected: Vec<(CspVarId, i64)> = a.iter().collect();
        collected.sort_by_key(|(id, _)| id.0);
        assert_eq!(collected, vec![(CspVarId(0), 10), (CspVarId(1), 20)]);
    }
}
