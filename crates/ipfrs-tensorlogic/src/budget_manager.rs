//! Tensor Budget Manager — computational budget tracking and enforcement per inference session.
//!
//! Manages FLOP, memory, and time budgets for tensor operations.  Each
//! inference session gets an independent [`SessionBudget`]; the
//! [`TensorBudgetManager`] orchestrates multiple concurrent sessions and
//! exposes aggregated statistics.

use std::collections::HashMap;

// ─── Resource type ────────────────────────────────────────────────────────────

/// The kind of computational resource being budgeted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceType {
    /// Floating-point operations (FLOPs).
    Flops,
    /// Memory consumption in bytes.
    MemoryBytes,
    /// Wall-clock time in milliseconds.
    TimeMs,
}

// ─── Budget ───────────────────────────────────────────────────────────────────

/// A single resource budget: limit, utilisation tracking.
#[derive(Clone, Debug)]
pub struct Budget {
    /// Which resource this budget governs.
    pub resource: ResourceType,
    /// Maximum allowed units.
    pub limit: u64,
    /// Units consumed so far.
    pub used: u64,
}

impl Budget {
    /// Create a new budget with a given limit (and zero usage).
    pub fn new(resource: ResourceType, limit: u64) -> Self {
        Self {
            resource,
            limit,
            used: 0,
        }
    }

    /// How many units remain before the limit is hit.
    pub fn remaining(&self) -> u64 {
        self.limit.saturating_sub(self.used)
    }

    /// Fraction of the budget that has been consumed (`used / limit`).
    ///
    /// Returns `0.0` when `limit == 0` to avoid division by zero.
    pub fn utilization(&self) -> f64 {
        self.used as f64 / self.limit.max(1) as f64
    }

    /// Whether the budget is fully consumed (`used >= limit`).
    pub fn is_exhausted(&self) -> bool {
        self.used >= self.limit
    }
}

// ─── Session budget ───────────────────────────────────────────────────────────

/// Per-session budget container.  Holds one [`Budget`] per [`ResourceType`].
#[derive(Clone, Debug)]
pub struct SessionBudget {
    /// Unique identifier for the owning session.
    pub session_id: u64,
    /// Per-resource budgets.
    pub budgets: HashMap<ResourceType, Budget>,
}

impl SessionBudget {
    /// Create a new session budget with pre-defined limits.
    ///
    /// Resources not listed in `limits` will be auto-created on first use
    /// with a limit of [`u64::MAX`].
    pub fn new(session_id: u64, limits: Vec<(ResourceType, u64)>) -> Self {
        let mut budgets = HashMap::new();
        for (resource, limit) in limits {
            budgets.insert(resource, Budget::new(resource, limit));
        }
        Self {
            session_id,
            budgets,
        }
    }

    /// Consume `amount` units of `resource`.
    ///
    /// # Errors
    ///
    /// Returns `Err("budget exceeded")` if the consumption would push `used`
    /// past `limit`.  The budget is **not** modified in that case.
    ///
    /// If no budget exists for `resource`, one is auto-created with
    /// `limit = u64::MAX` before the consumption is applied.
    pub fn consume(&mut self, resource: ResourceType, amount: u64) -> Result<(), String> {
        let budget = self
            .budgets
            .entry(resource)
            .or_insert_with(|| Budget::new(resource, u64::MAX));

        // Saturating check: would adding `amount` exceed the limit?
        if budget.used.saturating_add(amount) > budget.limit {
            return Err("budget exceeded".to_string());
        }
        budget.used = budget.used.saturating_add(amount);
        Ok(())
    }

    /// Remaining units for `resource`.
    ///
    /// Returns [`u64::MAX`] when the resource has not been registered (no
    /// limit has been set — it is effectively unlimited).
    pub fn remaining(&self, resource: ResourceType) -> u64 {
        self.budgets
            .get(&resource)
            .map(|b| b.remaining())
            .unwrap_or(u64::MAX)
    }

    /// Whether **any** registered resource is exhausted.
    pub fn is_any_exhausted(&self) -> bool {
        self.budgets.values().any(|b| b.is_exhausted())
    }
}

// ─── Budget manager statistics ────────────────────────────────────────────────

/// Aggregate statistics across all sessions managed by a [`TensorBudgetManager`].
#[derive(Clone, Debug)]
pub struct BudgetManagerStats {
    /// Total number of sessions ever created (including closed ones).
    pub total_sessions: usize,
    /// Number of sessions that have at least one exhausted resource.
    pub exhausted_sessions: usize,
}

impl BudgetManagerStats {
    /// Fraction of sessions that have at least one exhausted resource.
    ///
    /// Returns `0.0` when no sessions exist.
    pub fn exhaustion_rate(&self) -> f64 {
        if self.total_sessions == 0 {
            return 0.0;
        }
        self.exhausted_sessions as f64 / self.total_sessions as f64
    }
}

// ─── Tensor budget manager ────────────────────────────────────────────────────

/// Orchestrates computational budgets across multiple concurrent inference
/// sessions.
///
/// # Example
///
/// ```
/// use ipfrs_tensorlogic::budget_manager::{TensorBudgetManager, ResourceType};
///
/// let mut mgr = TensorBudgetManager::new();
/// let sid = mgr.open_session(vec![
///     (ResourceType::Flops,       1_000_000),
///     (ResourceType::MemoryBytes, 256 * 1024 * 1024),
/// ]);
///
/// mgr.consume(sid, ResourceType::Flops, 500_000).expect("example: should succeed in docs");
/// assert!(mgr.close_session(sid));
/// ```
#[derive(Debug, Default)]
pub struct TensorBudgetManager {
    /// Active sessions keyed by session id.
    pub sessions: HashMap<u64, SessionBudget>,
    /// Monotonically-increasing session id counter.
    pub next_session_id: u64,
}

impl TensorBudgetManager {
    /// Create a new, empty budget manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Open a new session with the supplied resource limits.
    ///
    /// Returns the unique session id assigned to the new session.
    pub fn open_session(&mut self, limits: Vec<(ResourceType, u64)>) -> u64 {
        let id = self.next_session_id;
        self.next_session_id = self.next_session_id.saturating_add(1);
        self.sessions.insert(id, SessionBudget::new(id, limits));
        id
    }

    /// Consume `amount` units of `resource` for the given session.
    ///
    /// # Errors
    ///
    /// - `Err("session not found")` — unknown `session_id`.
    /// - `Err("budget exceeded")` — the resource budget would be exceeded.
    pub fn consume(
        &mut self,
        session_id: u64,
        resource: ResourceType,
        amount: u64,
    ) -> Result<(), String> {
        self.sessions
            .get_mut(&session_id)
            .ok_or_else(|| "session not found".to_string())?
            .consume(resource, amount)
    }

    /// Close (remove) a session.
    ///
    /// Returns `true` if the session existed and was removed, `false`
    /// otherwise.
    pub fn close_session(&mut self, session_id: u64) -> bool {
        self.sessions.remove(&session_id).is_some()
    }

    /// Compute aggregate statistics across **all currently open** sessions.
    pub fn stats(&self) -> BudgetManagerStats {
        let total_sessions = self.sessions.len();
        let exhausted_sessions = self
            .sessions
            .values()
            .filter(|s| s.is_any_exhausted())
            .count();
        BudgetManagerStats {
            total_sessions,
            exhausted_sessions,
        }
    }

    /// Return per-resource utilisation for the given session.
    ///
    /// Returns `None` if the session does not exist.
    pub fn session_utilization(&self, session_id: u64) -> Option<Vec<(ResourceType, f64)>> {
        let session = self.sessions.get(&session_id)?;
        let utilizations = session
            .budgets
            .iter()
            .map(|(&resource, budget)| (resource, budget.utilization()))
            .collect();
        Some(utilizations)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: open a session with common limits.
    fn make_manager_with_session() -> (TensorBudgetManager, u64) {
        let mut mgr = TensorBudgetManager::new();
        let sid = mgr.open_session(vec![
            (ResourceType::Flops, 1_000),
            (ResourceType::MemoryBytes, 4_096),
            (ResourceType::TimeMs, 500),
        ]);
        (mgr, sid)
    }

    // ── 1. open_session returns a valid id and creates budgets ─────────────
    #[test]
    fn test_open_session_creates_budgets() {
        let (mgr, sid) = make_manager_with_session();
        assert!(mgr.sessions.contains_key(&sid));
        let session = &mgr.sessions[&sid];
        assert_eq!(session.session_id, sid);
        assert_eq!(session.budgets[&ResourceType::Flops].limit, 1_000);
        assert_eq!(session.budgets[&ResourceType::MemoryBytes].limit, 4_096);
        assert_eq!(session.budgets[&ResourceType::TimeMs].limit, 500);
    }

    // ── 2. open_session assigns distinct ids ──────────────────────────────
    #[test]
    fn test_open_session_distinct_ids() {
        let mut mgr = TensorBudgetManager::new();
        let a = mgr.open_session(vec![]);
        let b = mgr.open_session(vec![]);
        let c = mgr.open_session(vec![]);
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    // ── 3. consume within budget succeeds ─────────────────────────────────
    #[test]
    fn test_consume_within_budget_ok() {
        let (mut mgr, sid) = make_manager_with_session();
        assert!(mgr.consume(sid, ResourceType::Flops, 500).is_ok());
        assert_eq!(mgr.sessions[&sid].budgets[&ResourceType::Flops].used, 500);
    }

    // ── 4. consume exactly at the limit is allowed ────────────────────────
    #[test]
    fn test_consume_at_limit_ok() {
        let (mut mgr, sid) = make_manager_with_session();
        assert!(mgr.consume(sid, ResourceType::Flops, 1_000).is_ok());
        assert!(mgr.sessions[&sid].budgets[&ResourceType::Flops].is_exhausted());
    }

    // ── 5. consume exceeding limit returns Err ────────────────────────────
    #[test]
    fn test_consume_exceeds_budget_err() {
        let (mut mgr, sid) = make_manager_with_session();
        let result = mgr.consume(sid, ResourceType::Flops, 1_001);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "budget exceeded");
    }

    // ── 6. budget not modified on Err ─────────────────────────────────────
    #[test]
    fn test_consume_budget_unchanged_on_err() {
        let (mut mgr, sid) = make_manager_with_session();
        let _ = mgr.consume(sid, ResourceType::Flops, 999);
        let _ = mgr.consume(sid, ResourceType::Flops, 999); // would exceed
        assert_eq!(mgr.sessions[&sid].budgets[&ResourceType::Flops].used, 999);
    }

    // ── 7. auto-create unlimited budget for unknown resource ──────────────
    #[test]
    fn test_auto_create_unlimited_budget() {
        let mut mgr = TensorBudgetManager::new();
        // open session with NO limits for Flops
        let sid = mgr.open_session(vec![(ResourceType::MemoryBytes, 1_024)]);
        // consuming Flops should succeed (auto-created with u64::MAX limit)
        assert!(mgr.consume(sid, ResourceType::Flops, u64::MAX / 2).is_ok());
        assert_eq!(
            mgr.sessions[&sid].budgets[&ResourceType::Flops].limit,
            u64::MAX
        );
    }

    // ── 8. remaining after consume ────────────────────────────────────────
    #[test]
    fn test_remaining_after_consume() {
        let (mut mgr, sid) = make_manager_with_session();
        mgr.consume(sid, ResourceType::TimeMs, 200)
            .expect("test: should succeed");
        let session = &mgr.sessions[&sid];
        assert_eq!(session.remaining(ResourceType::TimeMs), 300);
    }

    // ── 9. remaining for unregistered resource is u64::MAX ────────────────
    #[test]
    fn test_remaining_unregistered_resource() {
        let mut mgr = TensorBudgetManager::new();
        let sid = mgr.open_session(vec![]); // no resources registered
        let session = &mgr.sessions[&sid];
        assert_eq!(session.remaining(ResourceType::Flops), u64::MAX);
    }

    // ── 10. is_exhausted ─────────────────────────────────────────────────
    #[test]
    fn test_budget_is_exhausted() {
        let mut b = Budget::new(ResourceType::Flops, 100);
        assert!(!b.is_exhausted());
        b.used = 100;
        assert!(b.is_exhausted());
        b.used = 101;
        assert!(b.is_exhausted());
    }

    // ── 11. utilization calculation ───────────────────────────────────────
    #[test]
    fn test_budget_utilization() {
        let mut b = Budget::new(ResourceType::MemoryBytes, 200);
        assert!((b.utilization() - 0.0).abs() < f64::EPSILON);
        b.used = 100;
        assert!((b.utilization() - 0.5).abs() < f64::EPSILON);
        b.used = 200;
        assert!((b.utilization() - 1.0).abs() < f64::EPSILON);
    }

    // ── 12. utilization when limit == 0 ───────────────────────────────────
    #[test]
    fn test_budget_utilization_zero_limit() {
        let b = Budget::new(ResourceType::Flops, 0);
        // limit.max(1) == 1, so utilization = 0/1 = 0.0
        assert!((b.utilization() - 0.0).abs() < f64::EPSILON);
    }

    // ── 13. is_any_exhausted ─────────────────────────────────────────────
    #[test]
    fn test_session_is_any_exhausted() {
        let mut mgr = TensorBudgetManager::new();
        let sid = mgr.open_session(vec![
            (ResourceType::Flops, 10),
            (ResourceType::MemoryBytes, 1_000),
        ]);
        assert!(!mgr.sessions[&sid].is_any_exhausted());
        mgr.consume(sid, ResourceType::Flops, 10)
            .expect("test: should succeed");
        assert!(mgr.sessions[&sid].is_any_exhausted());
    }

    // ── 14. close_session removes session and returns true ────────────────
    #[test]
    fn test_close_session_removes() {
        let (mut mgr, sid) = make_manager_with_session();
        assert!(mgr.close_session(sid));
        assert!(!mgr.sessions.contains_key(&sid));
    }

    // ── 15. close_session on unknown id returns false ─────────────────────
    #[test]
    fn test_close_session_unknown_returns_false() {
        let mut mgr = TensorBudgetManager::new();
        assert!(!mgr.close_session(9999));
    }

    // ── 16. stats: exhausted_sessions count ──────────────────────────────
    #[test]
    fn test_stats_exhausted_sessions() {
        let mut mgr = TensorBudgetManager::new();
        let s1 = mgr.open_session(vec![(ResourceType::Flops, 10)]);
        let s2 = mgr.open_session(vec![(ResourceType::Flops, 10)]);
        let _s3 = mgr.open_session(vec![(ResourceType::Flops, 10)]);

        // exhaust s1 and s2
        mgr.consume(s1, ResourceType::Flops, 10)
            .expect("test: should succeed");
        mgr.consume(s2, ResourceType::Flops, 10)
            .expect("test: should succeed");

        let stats = mgr.stats();
        assert_eq!(stats.total_sessions, 3);
        assert_eq!(stats.exhausted_sessions, 2);
    }

    // ── 17. exhaustion_rate ───────────────────────────────────────────────
    #[test]
    fn test_exhaustion_rate() {
        let mut mgr = TensorBudgetManager::new();
        let s1 = mgr.open_session(vec![(ResourceType::Flops, 10)]);
        let _s2 = mgr.open_session(vec![(ResourceType::Flops, 10)]);
        mgr.consume(s1, ResourceType::Flops, 10)
            .expect("test: should succeed");

        let stats = mgr.stats();
        assert!((stats.exhaustion_rate() - 0.5).abs() < f64::EPSILON);
    }

    // ── 18. exhaustion_rate with no sessions ─────────────────────────────
    #[test]
    fn test_exhaustion_rate_no_sessions() {
        let mgr = TensorBudgetManager::new();
        let stats = mgr.stats();
        assert_eq!(stats.total_sessions, 0);
        assert!((stats.exhaustion_rate() - 0.0).abs() < f64::EPSILON);
    }

    // ── 19. session_utilization returns correct values ────────────────────
    #[test]
    fn test_session_utilization() {
        let mut mgr = TensorBudgetManager::new();
        let sid = mgr.open_session(vec![
            (ResourceType::Flops, 1_000),
            (ResourceType::MemoryBytes, 2_000),
        ]);
        mgr.consume(sid, ResourceType::Flops, 250)
            .expect("test: should succeed");
        mgr.consume(sid, ResourceType::MemoryBytes, 1_000)
            .expect("test: should succeed");

        let utils = mgr.session_utilization(sid).expect("session must exist");
        let mut map: HashMap<ResourceType, f64> = utils.into_iter().collect();
        let flops_util = map.remove(&ResourceType::Flops).expect("Flops missing");
        let mem_util = map
            .remove(&ResourceType::MemoryBytes)
            .expect("MemoryBytes missing");

        assert!((flops_util - 0.25).abs() < 1e-9);
        assert!((mem_util - 0.5).abs() < 1e-9);
    }

    // ── 20. session_utilization returns None for unknown session ──────────
    #[test]
    fn test_session_utilization_unknown() {
        let mgr = TensorBudgetManager::new();
        assert!(mgr.session_utilization(42).is_none());
    }

    // ── 21. consume on unknown session returns Err ────────────────────────
    #[test]
    fn test_consume_unknown_session_err() {
        let mut mgr = TensorBudgetManager::new();
        let result = mgr.consume(9999, ResourceType::Flops, 1);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "session not found");
    }

    // ── 22. multiple resources tracked independently ──────────────────────
    #[test]
    fn test_multiple_resources_independent() {
        let (mut mgr, sid) = make_manager_with_session();
        mgr.consume(sid, ResourceType::Flops, 900)
            .expect("test: should succeed");
        mgr.consume(sid, ResourceType::MemoryBytes, 100)
            .expect("test: should succeed");

        let session = &mgr.sessions[&sid];
        assert_eq!(session.budgets[&ResourceType::Flops].used, 900);
        assert_eq!(session.budgets[&ResourceType::MemoryBytes].used, 100);
        assert_eq!(session.budgets[&ResourceType::TimeMs].used, 0);
    }
}
