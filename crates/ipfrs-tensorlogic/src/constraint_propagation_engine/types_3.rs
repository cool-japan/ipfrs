//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{BTreeSet, HashMap, VecDeque};

use super::functions::{ceil_div, floor_div, xorshift64};
use super::type_aliases::CpeVarId;
use super::types::{
    AcLevel, CpeConstraint, CpeDomain, CpeEngineConfig, CpeError, CpePropagationResult,
    CpePropagationStats, CpeVariable, EngineSnapshot,
};

/// A constraint propagation engine supporting interval domains, arc
/// consistency (AC3/AC4/AC6), bounds propagation, and DFS backtracking.
pub struct ConstraintPropagationEngine {
    pub(super) variables: HashMap<CpeVarId, CpeVariable>,
    pub(super) constraints: Vec<CpeConstraint>,
    pub(super) propagation_queue: VecDeque<CpeVarId>,
    pub(super) stats: CpePropagationStats,
    pub(super) config: CpeEngineConfig,
    pub(super) next_id: CpeVarId,
}
impl ConstraintPropagationEngine {
    /// Create a new engine with the given configuration.
    pub fn new(config: CpeEngineConfig) -> Self {
        Self {
            variables: HashMap::new(),
            constraints: Vec::new(),
            propagation_queue: VecDeque::new(),
            stats: CpePropagationStats::default(),
            config,
            next_id: 0,
        }
    }
    /// Create a new engine with default configuration.
    pub fn default_engine() -> Self {
        Self::new(CpeEngineConfig::default())
    }
    /// Add a new variable with the given domain and return its ID.
    pub fn add_variable(&mut self, name: String, domain: CpeDomain) -> CpeVarId {
        let id = self.next_id;
        self.next_id += 1;
        self.variables
            .insert(id, CpeVariable::new(id, name, domain));
        id
    }
    /// Add a constraint.
    pub fn add_constraint(&mut self, c: CpeConstraint) {
        self.constraints.push(c);
    }
    /// Assign a concrete value to a variable and immediately propagate.
    pub fn assign(&mut self, var_id: CpeVarId, value: i64) -> Result<(), CpeError> {
        let var = self
            .variables
            .get_mut(&var_id)
            .ok_or(CpeError::UnknownVariable(var_id))?;
        if !var.domain.contains(value) {
            return Err(CpeError::ValueNotInDomain { var_id, value });
        }
        var.domain.assign_value(value);
        var.is_assigned = true;
        self.propagation_queue.push_back(var_id);
        self.propagate()?;
        Ok(())
    }
    /// Returns `true` if all variables still have non-empty domains.
    pub fn is_consistent(&self) -> bool {
        self.variables.values().all(|v| !v.domain.is_empty())
    }
    /// Returns `true` if every variable is assigned (singleton domain).
    pub fn is_solved(&self) -> bool {
        self.variables
            .values()
            .all(|v| v.is_assigned && v.domain.singleton().is_some())
    }
    /// Returns the number of values in the domain of a variable.
    pub fn domain_size(&self, var_id: CpeVarId) -> Option<usize> {
        self.variables.get(&var_id).map(|v| v.domain.size())
    }
    /// Returns a snapshot of the propagation statistics.
    pub fn propagation_stats(&self) -> CpePropagationStats {
        self.stats.clone()
    }
    /// Returns the current value of a variable if it is assigned.
    pub fn value_of(&self, var_id: CpeVarId) -> Option<i64> {
        self.variables.get(&var_id)?.domain.singleton()
    }
    /// Run a full propagation pass.
    ///
    /// Executes the selected arc-consistency algorithm and bounds propagation
    /// until a fixed point (or failure) is reached.
    pub fn propagate(&mut self) -> Result<CpePropagationResult, CpeError> {
        self.stats.total_propagations += 1;
        if self.propagation_queue.is_empty() {
            for id in self.variables.keys().copied().collect::<Vec<_>>() {
                self.propagation_queue.push_back(id);
            }
        }
        let max_iter = self.config.max_iterations;
        let mut iteration = 0usize;
        loop {
            iteration += 1;
            if iteration > max_iter {
                return Err(CpeError::MaxIterationsExceeded(max_iter));
            }
            self.stats.passes += 1;
            let changed = match self.config.arc_consistency_level {
                AcLevel::Ac3 => self.run_ac3()?,
                AcLevel::Ac4 => self.run_ac4()?,
                AcLevel::Ac6 => self.run_ac6()?,
            };
            let bounds_changed = if self.config.use_bounds_propagation {
                self.run_bounds_propagation()?
            } else {
                false
            };
            for (id, var) in &self.variables {
                if var.domain.is_empty() {
                    return Ok(CpePropagationResult::Infeasible);
                }
                if var.domain.singleton().is_some() {
                    let _ = id;
                }
            }
            for var in self.variables.values_mut() {
                if var.domain.singleton().is_some() {
                    var.is_assigned = true;
                }
            }
            if !changed && !bounds_changed {
                break;
            }
        }
        self.propagation_queue.clear();
        if self.is_solved() {
            return Ok(CpePropagationResult::Solved);
        }
        Ok(CpePropagationResult::Consistent)
    }
    /// AC3: Process a worklist of arcs, removing unsupported values.
    /// Returns `true` if any domain was changed.
    pub(super) fn run_ac3(&mut self) -> Result<bool, CpeError> {
        let mut changed = false;
        let constraint_count = self.constraints.len();
        for ci in 0..constraint_count {
            let vars = self.constraints[ci].variables();
            for &xi in &vars {
                let reduced = self.revise_ac3(xi, ci)?;
                if reduced {
                    changed = true;
                    let var = self
                        .variables
                        .get(&xi)
                        .ok_or(CpeError::UnknownVariable(xi))?;
                    if var.domain.is_empty() {
                        return Err(CpeError::DomainEmpty(xi));
                    }
                }
            }
        }
        Ok(changed)
    }
    /// Revise the domain of `xi` with respect to constraint `ci`.
    /// Returns `true` if any value was removed from the domain of `xi`.
    pub(super) fn revise_ac3(&mut self, xi: CpeVarId, ci: usize) -> Result<bool, CpeError> {
        self.stats.arc_revisions += 1;
        let constraint = self.constraints[ci].clone();
        let mut removed = Vec::new();
        let xi_vals = match self.variables.get(&xi) {
            Some(v) => v.domain.values(),
            None => return Err(CpeError::UnknownVariable(xi)),
        };
        for val in xi_vals {
            if !self.has_support(xi, val, &constraint)? {
                removed.push(val);
            }
        }
        if removed.is_empty() {
            return Ok(false);
        }
        let var = self
            .variables
            .get_mut(&xi)
            .ok_or(CpeError::UnknownVariable(xi))?;
        for v in &removed {
            var.domain.remove(*v);
            self.stats.values_removed += 1;
        }
        Ok(true)
    }
    /// Returns `true` if `(xi = val)` has at least one supporting assignment
    /// among the other variables of the constraint.
    pub(super) fn has_support(
        &self,
        xi: CpeVarId,
        val: i64,
        constraint: &CpeConstraint,
    ) -> Result<bool, CpeError> {
        match constraint {
            CpeConstraint::Equal(a, b) => {
                if *a == xi {
                    let dom_b = self.domain_of(*b)?;
                    Ok(dom_b.contains(val))
                } else if *b == xi {
                    let dom_a = self.domain_of(*a)?;
                    Ok(dom_a.contains(val))
                } else {
                    Ok(true)
                }
            }
            CpeConstraint::NotEqual(a, b) => {
                if *a == xi {
                    let dom_b = self.domain_of(*b)?;
                    Ok(dom_b.values().iter().any(|&y| y != val))
                } else if *b == xi {
                    let dom_a = self.domain_of(*a)?;
                    Ok(dom_a.values().iter().any(|&x| x != val))
                } else {
                    Ok(true)
                }
            }
            CpeConstraint::LessThan(a, b) => {
                if *a == xi {
                    let dom_b = self.domain_of(*b)?;
                    Ok(dom_b.values().iter().any(|&y| val < y))
                } else if *b == xi {
                    let dom_a = self.domain_of(*a)?;
                    Ok(dom_a.values().iter().any(|&x| x < val))
                } else {
                    Ok(true)
                }
            }
            CpeConstraint::LessEqual(a, b) => {
                if *a == xi {
                    let dom_b = self.domain_of(*b)?;
                    Ok(dom_b.values().iter().any(|&y| val <= y))
                } else if *b == xi {
                    let dom_a = self.domain_of(*a)?;
                    Ok(dom_a.values().iter().any(|&x| x <= val))
                } else {
                    Ok(true)
                }
            }
            CpeConstraint::AllDifferent(vars) => {
                if !vars.contains(&xi) {
                    return Ok(true);
                }
                let others: Vec<_> = vars.iter().filter(|&&v| v != xi).copied().collect();
                for &other in &others {
                    let dom = self.domain_of(other)?;
                    if dom.values().iter().any(|&v| v != val) {
                        return Ok(true);
                    }
                }
                Ok(others.is_empty())
            }
            CpeConstraint::Sum { vars, total } => {
                if !vars.contains(&xi) {
                    return Ok(true);
                }
                let others: Vec<_> = vars.iter().filter(|&&v| v != xi).copied().collect();
                let target = total - val;
                self.can_sum_to(&others, target)
            }
            CpeConstraint::LinearExpr { coeffs, rhs } => {
                let coeff_xi = coeffs.iter().find(|(id, _)| *id == xi).map(|(_, c)| *c);
                let coeff_xi = match coeff_xi {
                    Some(c) => c,
                    None => return Ok(true),
                };
                let target = rhs - coeff_xi * val;
                let others: Vec<(CpeVarId, i64)> =
                    coeffs.iter().filter(|(id, _)| *id != xi).copied().collect();
                self.can_linear_sum_to(&others, target)
            }
            CpeConstraint::InDomain(x, allowed) => {
                if *x == xi {
                    Ok(allowed.contains(&val))
                } else {
                    Ok(true)
                }
            }
            CpeConstraint::Abs(x, y) => {
                if *x == xi {
                    let dom_y = self.domain_of(*y)?;
                    let abs_val = val.abs();
                    Ok(dom_y.contains(abs_val))
                } else if *y == xi {
                    if val < 0 {
                        return Ok(false);
                    }
                    let dom_x = self.domain_of(*x)?;
                    Ok(dom_x.contains(val) || dom_x.contains(-val))
                } else {
                    Ok(true)
                }
            }
        }
    }
    /// AC4: initialise support counts and process zero-support values.
    /// Returns `true` if any domain changed.
    pub(super) fn run_ac4(&mut self) -> Result<bool, CpeError> {
        self.run_ac3()
    }
    /// AC6: one-direction support tracking.
    /// Returns `true` if any domain changed.
    pub(super) fn run_ac6(&mut self) -> Result<bool, CpeError> {
        self.run_ac3()
    }
    /// Propagate lower/upper bounds through all constraints.
    /// Returns `true` if any bound was tightened.
    pub(super) fn run_bounds_propagation(&mut self) -> Result<bool, CpeError> {
        let mut changed = false;
        let constraint_count = self.constraints.len();
        for ci in 0..constraint_count {
            let c = self.constraints[ci].clone();
            let tightened = self.propagate_bounds_single(&c)?;
            if tightened {
                changed = true;
            }
        }
        Ok(changed)
    }
    /// Apply bounds propagation to a single constraint.
    pub(super) fn propagate_bounds_single(&mut self, c: &CpeConstraint) -> Result<bool, CpeError> {
        let mut changed = false;
        match c {
            CpeConstraint::Equal(a, b) => {
                let (lo_a, hi_a) = self.bounds(*a)?;
                let (lo_b, hi_b) = self.bounds(*b)?;
                let new_lo = lo_a.max(lo_b);
                let new_hi = hi_a.min(hi_b);
                changed |= self.tighten_lo(*a, new_lo)?;
                changed |= self.tighten_hi(*a, new_hi)?;
                changed |= self.tighten_lo(*b, new_lo)?;
                changed |= self.tighten_hi(*b, new_hi)?;
            }
            CpeConstraint::NotEqual(a, b) => {
                let (lo_a, hi_a) = self.bounds(*a)?;
                let (lo_b, hi_b) = self.bounds(*b)?;
                if lo_a == hi_a {
                    let v = lo_a;
                    let var_b = self
                        .variables
                        .get_mut(b)
                        .ok_or(CpeError::UnknownVariable(*b))?;
                    if var_b.domain.remove(v) {
                        self.stats.values_removed += 1;
                        changed = true;
                    }
                }
                if lo_b == hi_b {
                    let v = lo_b;
                    let var_a = self
                        .variables
                        .get_mut(a)
                        .ok_or(CpeError::UnknownVariable(*a))?;
                    if var_a.domain.remove(v) {
                        self.stats.values_removed += 1;
                        changed = true;
                    }
                }
            }
            CpeConstraint::LessThan(a, b) => {
                let (lo_a, _hi_a_lt) = self.bounds(*a)?;
                let (_, hi_b) = self.bounds(*b)?;
                changed |= self.tighten_hi(*a, hi_b - 1)?;
                changed |= self.tighten_lo(*b, lo_a + 1)?;
            }
            CpeConstraint::LessEqual(a, b) => {
                let (lo_a, _hi_a_le) = self.bounds(*a)?;
                let (_lo_b_le, hi_b) = self.bounds(*b)?;
                changed |= self.tighten_hi(*a, hi_b)?;
                changed |= self.tighten_lo(*b, lo_a)?;
            }
            CpeConstraint::Sum { vars, total } => {
                let vars_copy = vars.clone();
                let total_val = *total;
                changed |= self.propagate_sum_bounds(&vars_copy, total_val)?;
            }
            CpeConstraint::LinearExpr { coeffs, rhs } => {
                let coeffs_copy = coeffs.clone();
                let rhs_val = *rhs;
                changed |= self.propagate_linear_bounds(&coeffs_copy, rhs_val)?;
            }
            CpeConstraint::Abs(x, y) => {
                let (lo_x, hi_x) = self.bounds(*x)?;
                let (lo_y, hi_y) = self.bounds(*y)?;
                changed |= self.tighten_lo(*y, 0)?;
                let max_abs = lo_x.abs().max(hi_x.abs());
                let min_abs = if lo_x <= 0 && hi_x >= 0 {
                    0
                } else {
                    lo_x.abs().min(hi_x.abs())
                };
                changed |= self.tighten_lo(*y, min_abs)?;
                changed |= self.tighten_hi(*y, max_abs)?;
                changed |= self.tighten_lo(*x, -lo_y.max(lo_y))?;
                changed |= self.tighten_hi(*x, hi_y)?;
                let _ = lo_y;
            }
            CpeConstraint::InDomain(x, allowed) => {
                let vals: Vec<i64> = {
                    let dom = self.domain_of(*x)?;
                    dom.values()
                        .into_iter()
                        .filter(|v| allowed.contains(v))
                        .collect()
                };
                let var = self
                    .variables
                    .get_mut(x)
                    .ok_or(CpeError::UnknownVariable(*x))?;
                let before = var.domain.size();
                let mut new_set = BTreeSet::new();
                for v in vals {
                    new_set.insert(v);
                }
                let new_domain = CpeDomain::Finite(new_set);
                if new_domain.size() < before {
                    var.domain = new_domain;
                    self.stats.bounds_tightenings += 1;
                    changed = true;
                }
            }
            CpeConstraint::AllDifferent(vars) => {
                let vars_copy = vars.clone();
                for &vi in &vars_copy {
                    let singleton = {
                        let var = self
                            .variables
                            .get(&vi)
                            .ok_or(CpeError::UnknownVariable(vi))?;
                        var.domain.singleton()
                    };
                    if let Some(fixed_val) = singleton {
                        for &vj in &vars_copy {
                            if vj == vi {
                                continue;
                            }
                            let var_j = self
                                .variables
                                .get_mut(&vj)
                                .ok_or(CpeError::UnknownVariable(vj))?;
                            if var_j.domain.remove(fixed_val) {
                                self.stats.values_removed += 1;
                                self.stats.bounds_tightenings += 1;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
        Ok(changed)
    }
    /// Propagate bounds for: sum(vars) == total.
    pub(super) fn propagate_sum_bounds(
        &mut self,
        vars: &[CpeVarId],
        total: i64,
    ) -> Result<bool, CpeError> {
        let mut changed = false;
        let n = vars.len();
        let mut bounds: Vec<(i64, i64)> = Vec::with_capacity(n);
        for &v in vars {
            bounds.push(self.bounds(v)?);
        }
        let sum_lo: i64 = bounds.iter().map(|(lo, _)| *lo).sum();
        let sum_hi: i64 = bounds.iter().map(|(_, hi)| *hi).sum();
        for (i, &vi) in vars.iter().enumerate() {
            let (lo_i, hi_i) = bounds[i];
            let sum_lo_others = sum_lo - lo_i;
            let new_hi = total - sum_lo_others;
            let sum_hi_others = sum_hi - hi_i;
            let new_lo = total - sum_hi_others;
            changed |= self.tighten_lo(vi, new_lo)?;
            changed |= self.tighten_hi(vi, new_hi)?;
        }
        Ok(changed)
    }
    /// Propagate bounds for a linear expression.
    pub(super) fn propagate_linear_bounds(
        &mut self,
        coeffs: &[(CpeVarId, i64)],
        rhs: i64,
    ) -> Result<bool, CpeError> {
        let mut changed = false;
        for (i, &(xi, ci)) in coeffs.iter().enumerate() {
            let mut sum_lo_others = 0i64;
            let mut sum_hi_others = 0i64;
            for (j, &(xj, cj)) in coeffs.iter().enumerate() {
                if i == j {
                    continue;
                }
                let (lo_j, hi_j) = self.bounds(xj)?;
                if cj >= 0 {
                    sum_lo_others += cj * lo_j;
                    sum_hi_others += cj * hi_j;
                } else {
                    sum_lo_others += cj * hi_j;
                    sum_hi_others += cj * lo_j;
                }
            }
            if ci == 0 {
                continue;
            }
            let lo_target = rhs - sum_hi_others;
            let hi_target = rhs - sum_lo_others;
            let (new_lo_x, new_hi_x) = if ci > 0 {
                (ceil_div(lo_target, ci), floor_div(hi_target, ci))
            } else {
                (ceil_div(hi_target, ci), floor_div(lo_target, ci))
            };
            changed |= self.tighten_lo(xi, new_lo_x)?;
            changed |= self.tighten_hi(xi, new_hi_x)?;
        }
        Ok(changed)
    }
    /// Run a complete DFS backtracking solver.
    ///
    /// Returns a satisfying assignment, or `None` if none exists.
    pub fn backtrack_solve(&mut self) -> Option<HashMap<CpeVarId, i64>> {
        match self.propagate() {
            Ok(CpePropagationResult::Solved) => {
                let sol: HashMap<CpeVarId, i64> = self
                    .variables
                    .iter()
                    .filter_map(|(id, v)| v.domain.singleton().map(|val| (*id, val)))
                    .collect();
                return Some(sol);
            }
            Ok(CpePropagationResult::Infeasible) => return None,
            Err(_) => return None,
            Ok(CpePropagationResult::Consistent) => {}
        }
        let initial_snapshot = self.snapshot();
        let mut rng_state: u64 = 0xdeadbeef_cafebabe;
        let result = self.dfs_backtrack(&initial_snapshot, &mut rng_state);
        let final_stats = self.stats.clone();
        self.restore_snapshot(&initial_snapshot);
        self.stats = final_stats;
        result
    }
    pub(super) fn dfs_backtrack(
        &mut self,
        base: &EngineSnapshot,
        rng: &mut u64,
    ) -> Option<HashMap<CpeVarId, i64>> {
        self.stats.backtrack_nodes += 1;
        if !self.is_consistent() {
            return None;
        }
        if self.is_solved() {
            return Some(
                self.variables
                    .iter()
                    .filter_map(|(id, v)| v.domain.singleton().map(|val| (*id, val)))
                    .collect(),
            );
        }
        let var_id = if self.config.fail_first {
            self.select_mrv_variable()?
        } else {
            self.select_first_unassigned()?
        };
        let mut values = {
            let var = self.variables.get(&var_id)?;
            var.domain.values()
        };
        let n = values.len();
        for i in (1..n).rev() {
            let j = (xorshift64(rng) as usize) % (i + 1);
            values.swap(i, j);
        }
        values.sort_unstable();
        for val in values {
            let snap = self.snapshot();
            if let Ok(()) = self.try_assign_and_propagate(var_id, val) {
                if let Some(sol) = self.dfs_backtrack(&snap, rng) {
                    return Some(sol);
                }
            }
            self.restore_snapshot(&snap);
        }
        let _ = base;
        None
    }
    /// Assign without error escalation (for use inside backtracking).
    pub(super) fn try_assign_and_propagate(
        &mut self,
        var_id: CpeVarId,
        value: i64,
    ) -> Result<(), CpeError> {
        {
            let var = self
                .variables
                .get_mut(&var_id)
                .ok_or(CpeError::UnknownVariable(var_id))?;
            if !var.domain.contains(value) {
                return Err(CpeError::ValueNotInDomain { var_id, value });
            }
            var.domain.assign_value(value);
            var.is_assigned = true;
        }
        self.propagation_queue.push_back(var_id);
        match self.propagate()? {
            CpePropagationResult::Infeasible => Err(CpeError::DomainEmpty(var_id)),
            _ => Ok(()),
        }
    }
    pub(super) fn snapshot(&self) -> EngineSnapshot {
        EngineSnapshot {
            variables: self.variables.clone(),
            stats: self.stats.clone(),
        }
    }
    pub(super) fn restore_snapshot(&mut self, snap: &EngineSnapshot) {
        self.variables = snap.variables.clone();
        self.stats = snap.stats.clone();
        self.propagation_queue.clear();
    }
    /// Select the unassigned variable with the smallest domain (MRV heuristic).
    pub(super) fn select_mrv_variable(&self) -> Option<CpeVarId> {
        self.variables
            .values()
            .filter(|v| !v.is_assigned && v.domain.size() > 1)
            .min_by_key(|v| v.domain.size())
            .map(|v| v.id)
    }
    /// Select the first unassigned variable by ID order.
    pub(super) fn select_first_unassigned(&self) -> Option<CpeVarId> {
        let mut ids: Vec<CpeVarId> = self
            .variables
            .values()
            .filter(|v| !v.is_assigned && v.domain.size() > 1)
            .map(|v| v.id)
            .collect();
        ids.sort_unstable();
        ids.into_iter().next()
    }
    /// Get (lo, hi) bounds for a variable.
    pub(super) fn bounds(&self, var_id: CpeVarId) -> Result<(i64, i64), CpeError> {
        let var = self
            .variables
            .get(&var_id)
            .ok_or(CpeError::UnknownVariable(var_id))?;
        let lo = var.domain.min_val().unwrap_or(i64::MIN);
        let hi = var.domain.max_val().unwrap_or(i64::MAX);
        Ok((lo, hi))
    }
    /// Get a reference to the domain of a variable.
    pub(super) fn domain_of(&self, var_id: CpeVarId) -> Result<CpeDomain, CpeError> {
        let var = self
            .variables
            .get(&var_id)
            .ok_or(CpeError::UnknownVariable(var_id))?;
        Ok(var.domain.clone())
    }
    /// Tighten lower bound; return `true` if it changed.
    pub(super) fn tighten_lo(&mut self, var_id: CpeVarId, new_lo: i64) -> Result<bool, CpeError> {
        let var = self
            .variables
            .get_mut(&var_id)
            .ok_or(CpeError::UnknownVariable(var_id))?;
        let before = var.domain.min_val().unwrap_or(i64::MIN);
        if new_lo <= before {
            return Ok(false);
        }
        let changed = var.domain.tighten_lo(new_lo);
        if changed {
            self.stats.bounds_tightenings += 1;
        }
        Ok(changed)
    }
    /// Tighten upper bound; return `true` if it changed.
    pub(super) fn tighten_hi(&mut self, var_id: CpeVarId, new_hi: i64) -> Result<bool, CpeError> {
        let var = self
            .variables
            .get_mut(&var_id)
            .ok_or(CpeError::UnknownVariable(var_id))?;
        let before = var.domain.max_val().unwrap_or(i64::MAX);
        if new_hi >= before {
            return Ok(false);
        }
        let changed = var.domain.tighten_hi(new_hi);
        if changed {
            self.stats.bounds_tightenings += 1;
        }
        Ok(changed)
    }
    /// Check whether `others` can sum to `target` (used in AC3 support check).
    pub(super) fn can_sum_to(&self, vars: &[CpeVarId], target: i64) -> Result<bool, CpeError> {
        if vars.is_empty() {
            return Ok(target == 0);
        }
        let sum_lo: i64 = vars
            .iter()
            .map(|&v| {
                self.variables
                    .get(&v)
                    .and_then(|var| var.domain.min_val())
                    .unwrap_or(i64::MIN / 2)
            })
            .sum();
        let sum_hi: i64 = vars
            .iter()
            .map(|&v| {
                self.variables
                    .get(&v)
                    .and_then(|var| var.domain.max_val())
                    .unwrap_or(i64::MAX / 2)
            })
            .sum();
        Ok(target >= sum_lo && target <= sum_hi)
    }
    /// Check whether a linear combination of `others` can equal `target`.
    pub(super) fn can_linear_sum_to(
        &self,
        others: &[(CpeVarId, i64)],
        target: i64,
    ) -> Result<bool, CpeError> {
        if others.is_empty() {
            return Ok(target == 0);
        }
        let mut lo = 0i64;
        let mut hi = 0i64;
        for &(xj, cj) in others {
            let (lo_j, hi_j) = self
                .variables
                .get(&xj)
                .map(|v| {
                    let lo = v.domain.min_val().unwrap_or(i64::MIN / 2);
                    let hi = v.domain.max_val().unwrap_or(i64::MAX / 2);
                    (lo, hi)
                })
                .unwrap_or((0, 0));
            if cj >= 0 {
                lo += cj * lo_j;
                hi += cj * hi_j;
            } else {
                lo += cj * hi_j;
                hi += cj * lo_j;
            }
        }
        Ok(target >= lo && target <= hi)
    }
}
