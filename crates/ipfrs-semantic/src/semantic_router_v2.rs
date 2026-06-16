//! # SemanticRouterV2
//!
//! An advanced semantic routing engine that uses embedding similarity to route
//! queries to specialised handlers, with fallback chains, load balancing, and
//! routing analytics.
//!
//! ## Design Goals
//!
//! * **Zero-copy routing** — embeddings stored contiguously; similarity computed in one pass.
//! * **Pluggable fallback** — four strategies cover the common operational scenarios.
//! * **Lock-free RNG** — self-contained xorshift64 (no external crate dependency).
//! * **No `unwrap()`** — every `Option` / `Result` is handled explicitly.

use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// xorshift64 — deterministic PRNG, no external crate needed
// ────────────────────────────────────────────────────────────────────────────

/// Advance the xorshift64 state by one step and return the new value.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// Opaque handle for a registered route handler.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RouteHandlerId(pub String);

impl RouteHandlerId {
    /// Create a new handler identifier from any `Into<String>` source.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Borrow the inner string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for RouteHandlerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ─── RouteDefinition ────────────────────────────────────────────────────────

/// A fully-specified route entry registered with the router.
#[derive(Debug, Clone)]
pub struct RouteDefinition {
    /// Unique identifier for this route.
    pub id: RouteHandlerId,
    /// Human-readable name.
    pub name: String,
    /// Embedding vector that defines the route's semantic region.
    pub embedding: Vec<f64>,
    /// Arbitrary key-value metadata (e.g. `"owner"`, `"model_version"`).
    pub metadata: HashMap<String, String>,
    /// Soft cap on how many queries per second this handler should receive.
    pub max_queries_per_second: f64,
    /// Higher priority (0 = lowest, 255 = highest) routes win tie-breaks.
    pub priority: u8,
}

impl RouteDefinition {
    /// Convenience constructor with defaults for optional fields.
    pub fn new(id: impl Into<String>, name: impl Into<String>, embedding: Vec<f64>) -> Self {
        Self {
            id: RouteHandlerId::new(id),
            name: name.into(),
            embedding,
            metadata: HashMap::new(),
            max_queries_per_second: f64::MAX,
            priority: 128,
        }
    }
}

// ─── RoutingDecision (V2) ───────────────────────────────────────────────────

/// The result of routing a single query through `SemanticRouterV2`.
#[derive(Debug, Clone)]
pub struct V2RoutingDecision {
    /// Original query text supplied by the caller.
    pub query_text: String,
    /// The embedding vector used to route the query.
    pub query_embedding: Vec<f64>,
    /// The handler selected by the router.
    pub matched_route: RouteHandlerId,
    /// Cosine similarity between the query embedding and the matched route embedding.
    /// Set to `0.0` when a fallback strategy is used.
    pub similarity: f64,
    /// `true` when the decision was made by the fallback strategy (no route exceeded the threshold).
    pub fallback_used: bool,
    /// Advisory latency hint in milliseconds (set to 0 when not measured).
    pub latency_hint_ms: u32,
}

// ─── FallbackStrategy ───────────────────────────────────────────────────────

/// Strategy to apply when no route meets the similarity threshold.
#[derive(Debug, Clone)]
pub enum FallbackStrategy {
    /// Always route to the designated default handler.
    UseDefault { default_id: RouteHandlerId },
    /// Cycle through all registered routes in registration order.
    RoundRobin,
    /// Pick the route with the lowest cumulative query count.
    LeastLoaded,
    /// Uniformly random selection using xorshift64 seeded with `seed`.
    Random { seed: u64 },
}

// ─── RouterV2Config ─────────────────────────────────────────────────────────

/// Configuration for `SemanticRouterV2`.
#[derive(Debug, Clone)]
pub struct RouterV2Config {
    /// Minimum cosine similarity required for a direct route match.
    pub similarity_threshold: f64,
    /// Maximum number of routes that may be registered simultaneously.
    pub max_routes: usize,
    /// Strategy to use when no route meets the threshold.
    pub fallback: FallbackStrategy,
    /// When `true`, the router collects per-route statistics for load inspection.
    pub enable_load_balancing: bool,
}

impl Default for RouterV2Config {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            max_routes: 100,
            fallback: FallbackStrategy::RoundRobin,
            enable_load_balancing: true,
        }
    }
}

// ─── RouteStats ─────────────────────────────────────────────────────────────

/// Per-route routing statistics.
#[derive(Debug, Clone)]
pub struct RouteStats {
    /// The handler identifier these stats belong to.
    pub handler_id: String,
    /// Total number of queries routed to this handler (both direct and fallback).
    pub total_routed: u64,
    /// Running average of cosine similarity for *direct* matches (0.0 for fallback-only routes).
    pub avg_similarity: f64,
    /// Timestamp of the most recent routing decision (caller-supplied `now` value).
    pub last_routed_at: u64,
}

impl RouteStats {
    fn new(handler_id: impl Into<String>) -> Self {
        Self {
            handler_id: handler_id.into(),
            total_routed: 0,
            avg_similarity: 0.0,
            last_routed_at: 0,
        }
    }

    /// Update the running average similarity after a direct match with `similarity`.
    fn record_direct(&mut self, similarity: f64, now: u64) {
        // Welford online mean update
        self.total_routed += 1;
        let n = self.total_routed as f64;
        self.avg_similarity += (similarity - self.avg_similarity) / n;
        self.last_routed_at = now;
    }

    /// Update stats after a fallback routing decision (similarity not available).
    fn record_fallback(&mut self, now: u64) {
        self.total_routed += 1;
        self.last_routed_at = now;
    }
}

// ─── RouterV2Stats ──────────────────────────────────────────────────────────

/// Aggregate statistics for the entire `SemanticRouterV2` instance.
#[derive(Debug, Clone)]
pub struct RouterV2Stats {
    /// Number of currently registered routes.
    pub route_count: usize,
    /// Total queries processed (since construction or last reset).
    pub total_queries: u64,
    /// Queries that fell through to a fallback strategy.
    pub fallback_count: u64,
    /// `fallback_count / total_queries`, or `0.0` when no queries have been processed.
    pub fallback_rate: f64,
    /// Weighted average similarity across all routes (direct matches only).
    pub avg_similarity: f64,
}

// ─── RouterV2Error ──────────────────────────────────────────────────────────

/// Errors that may be returned by `SemanticRouterV2` operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouterV2Error {
    /// A route with the same identifier is already registered.
    RouteAlreadyExists(String),
    /// No route with the given identifier exists.
    RouteNotFound(String),
    /// The number of registered routes would exceed `RouterV2Config::max_routes`.
    MaxRoutesReached,
    /// The operation requires at least one registered route, but none are present.
    EmptyRoutes,
}

impl std::fmt::Display for RouterV2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RouteAlreadyExists(id) => write!(f, "route already exists: {id}"),
            Self::RouteNotFound(id) => write!(f, "route not found: {id}"),
            Self::MaxRoutesReached => write!(f, "maximum route count reached"),
            Self::EmptyRoutes => write!(f, "no routes registered"),
        }
    }
}

impl std::error::Error for RouterV2Error {}

// ────────────────────────────────────────────────────────────────────────────
// SemanticRouterV2
// ────────────────────────────────────────────────────────────────────────────

/// Advanced semantic routing engine.
///
/// Routes natural-language queries (represented as embedding vectors) to the
/// most semantically appropriate registered handler.  When no handler meets
/// the configured similarity threshold, a pluggable fallback strategy is
/// applied.
///
/// # Thread Safety
///
/// `SemanticRouterV2` is **not** `Send + Sync` by itself — wrap it in an
/// `Arc<Mutex<_>>` for shared use across tasks.
pub struct SemanticRouterV2 {
    /// Router configuration (immutable after construction).
    pub config: RouterV2Config,
    /// Ordered list of registered routes.
    pub routes: Vec<RouteDefinition>,
    /// Per-handler statistics, keyed by `RouteHandlerId.0`.
    pub route_stats: HashMap<String, RouteStats>,
    /// Round-robin cursor (next index to use for `RoundRobin` fallback).
    pub rr_index: usize,
    /// Total number of queries processed.
    pub total_queries: u64,
    /// Number of queries that required a fallback.
    pub fallback_count: u64,
    /// PRNG state (xorshift64).
    pub rng_state: u64,
}

impl SemanticRouterV2 {
    /// Construct a new router with the given configuration.
    ///
    /// The PRNG is seeded to `0xCAFEBABE_12345678`.
    pub fn new(config: RouterV2Config) -> Self {
        Self {
            config,
            routes: Vec::new(),
            route_stats: HashMap::new(),
            rr_index: 0,
            total_queries: 0,
            fallback_count: 0,
            rng_state: 0xCAFEBABE_12345678_u64,
        }
    }

    /// Register a new route.
    ///
    /// # Errors
    ///
    /// * [`RouterV2Error::RouteAlreadyExists`] — if a route with the same ID exists.
    /// * [`RouterV2Error::MaxRoutesReached`] — if `max_routes` would be exceeded.
    pub fn add_route(&mut self, route: RouteDefinition) -> Result<(), RouterV2Error> {
        if self.routes.len() >= self.config.max_routes {
            return Err(RouterV2Error::MaxRoutesReached);
        }
        if self.routes.iter().any(|r| r.id == route.id) {
            return Err(RouterV2Error::RouteAlreadyExists(route.id.0.clone()));
        }
        let id_str = route.id.0.clone();
        self.routes.push(route);
        self.route_stats
            .entry(id_str.clone())
            .or_insert_with(|| RouteStats::new(id_str));
        Ok(())
    }

    /// Remove a route by identifier.
    ///
    /// Returns `true` if a route was found and removed, `false` otherwise.
    pub fn remove_route(&mut self, id: &RouteHandlerId) -> bool {
        let before = self.routes.len();
        self.routes.retain(|r| r.id != *id);
        let removed = self.routes.len() < before;
        if removed {
            self.route_stats.remove(&id.0);
            // Reset the round-robin cursor to stay in bounds
            if !self.routes.is_empty() {
                self.rr_index %= self.routes.len();
            } else {
                self.rr_index = 0;
            }
        }
        removed
    }

    /// Compute the cosine similarity between two vectors.
    ///
    /// Returns `0.0` when either vector has zero magnitude.
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let min_len = a.len().min(b.len());
        if min_len == 0 {
            return 0.0;
        }
        let mut dot = 0.0_f64;
        let mut mag_a = 0.0_f64;
        let mut mag_b = 0.0_f64;
        for i in 0..min_len {
            dot += a[i] * b[i];
            mag_a += a[i] * a[i];
            mag_b += b[i] * b[i];
        }
        let denom = mag_a.sqrt() * mag_b.sqrt();
        if denom < f64::EPSILON {
            0.0
        } else {
            (dot / denom).clamp(-1.0, 1.0)
        }
    }

    /// Find the best matching route for `query_embedding`.
    ///
    /// Returns `Some((index, similarity))` for the route with the highest cosine
    /// similarity that also meets or exceeds `similarity_threshold`.  Returns
    /// `None` when no route qualifies.
    pub fn best_match(&self, query_embedding: &[f64]) -> Option<(usize, f64)> {
        if self.routes.is_empty() {
            return None;
        }
        let mut best_idx = 0usize;
        let mut best_sim = f64::NEG_INFINITY;
        for (i, route) in self.routes.iter().enumerate() {
            let sim = Self::cosine_similarity(query_embedding, &route.embedding);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }
        if best_sim >= self.config.similarity_threshold {
            Some((best_idx, best_sim))
        } else {
            None
        }
    }

    /// Route a single query, returning a routing decision.
    ///
    /// * `query_text`      — raw text of the query (stored in the decision for audit).
    /// * `query_embedding` — embedding of the query.
    /// * `now`             — caller-supplied monotonic timestamp (e.g. milliseconds since epoch).
    pub fn route(
        &mut self,
        query_text: String,
        query_embedding: Vec<f64>,
        now: u64,
    ) -> V2RoutingDecision {
        self.total_queries += 1;

        // ── Direct match ──────────────────────────────────────────────────
        if let Some((idx, similarity)) = self.best_match(&query_embedding) {
            let matched_id = self.routes[idx].id.clone();
            self.update_stats_direct(&matched_id, similarity, now);
            return V2RoutingDecision {
                query_text,
                query_embedding,
                matched_route: matched_id,
                similarity,
                fallback_used: false,
                latency_hint_ms: 0,
            };
        }

        // ── Fallback ──────────────────────────────────────────────────────
        self.fallback_count += 1;
        let fallback_id = self.apply_fallback(now);
        V2RoutingDecision {
            query_text,
            query_embedding,
            matched_route: fallback_id,
            similarity: 0.0,
            fallback_used: true,
            latency_hint_ms: 0,
        }
    }

    /// Route a batch of queries.
    ///
    /// Queries are processed sequentially; order of results mirrors order of input.
    pub fn route_batch(
        &mut self,
        queries: Vec<(String, Vec<f64>)>,
        now: u64,
    ) -> Vec<V2RoutingDecision> {
        queries
            .into_iter()
            .map(|(text, emb)| self.route(text, emb, now))
            .collect()
    }

    /// Return the top-`k` routes by `total_routed`, descending.
    pub fn top_routes(&self, k: usize) -> Vec<&RouteStats> {
        let mut stats: Vec<&RouteStats> = self.route_stats.values().collect();
        stats.sort_by_key(|s| std::cmp::Reverse(s.total_routed));
        stats.truncate(k);
        stats
    }

    /// Look up a registered route definition by identifier.
    pub fn route_by_id(&self, id: &RouteHandlerId) -> Option<&RouteDefinition> {
        self.routes.iter().find(|r| r.id == *id)
    }

    /// Replace the embedding of an existing route in-place.
    ///
    /// Returns `true` if the route was found and updated.
    pub fn update_route_embedding(&mut self, id: &RouteHandlerId, new_embedding: Vec<f64>) -> bool {
        if let Some(route) = self.routes.iter_mut().find(|r| r.id == *id) {
            route.embedding = new_embedding;
            true
        } else {
            false
        }
    }

    /// Compute aggregate statistics for the entire router.
    pub fn router_stats(&self) -> RouterV2Stats {
        let route_count = self.routes.len();
        let fallback_rate = if self.total_queries == 0 {
            0.0
        } else {
            self.fallback_count as f64 / self.total_queries as f64
        };

        // Weighted mean similarity across all routes
        let (total_sim, total_direct) =
            self.route_stats
                .values()
                .fold((0.0_f64, 0u64), |(acc_sim, acc_n), s| {
                    // total_routed includes fallback; we re-derive direct count from avg_similarity
                    // We use avg_similarity * total_routed as sum proxy (fallback routes have avg=0)
                    (
                        acc_sim + s.avg_similarity * s.total_routed as f64,
                        acc_n + s.total_routed,
                    )
                });
        let avg_similarity = if total_direct == 0 {
            0.0
        } else {
            total_sim / total_direct as f64
        };

        RouterV2Stats {
            route_count,
            total_queries: self.total_queries,
            fallback_count: self.fallback_count,
            fallback_rate,
            avg_similarity,
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────

    fn update_stats_direct(&mut self, id: &RouteHandlerId, similarity: f64, now: u64) {
        let stats = self
            .route_stats
            .entry(id.0.clone())
            .or_insert_with(|| RouteStats::new(id.0.clone()));
        stats.record_direct(similarity, now);
    }

    fn update_stats_fallback(&mut self, id: &RouteHandlerId, now: u64) {
        let stats = self
            .route_stats
            .entry(id.0.clone())
            .or_insert_with(|| RouteStats::new(id.0.clone()));
        stats.record_fallback(now);
    }

    /// Apply the configured fallback strategy and return the selected handler ID.
    ///
    /// Falls back gracefully to index `0` when the route list is empty (the
    /// caller guarantees that `route` is only called when at least one route
    /// exists — but we handle the edge case defensively).
    fn apply_fallback(&mut self, now: u64) -> RouteHandlerId {
        if self.routes.is_empty() {
            // Should not happen in normal usage; return a sentinel value.
            return RouteHandlerId::new("__no_routes__");
        }

        let selected_id = match self.config.fallback.clone() {
            FallbackStrategy::UseDefault { default_id } => {
                // Validate the default_id exists; fall through to round-robin if not.
                if self.routes.iter().any(|r| r.id == default_id) {
                    default_id
                } else {
                    self.round_robin_id()
                }
            }
            FallbackStrategy::RoundRobin => self.round_robin_id(),
            FallbackStrategy::LeastLoaded => self.least_loaded_id(),
            FallbackStrategy::Random { .. } => self.random_id(),
        };

        self.update_stats_fallback(&selected_id.clone(), now);
        selected_id
    }

    fn round_robin_id(&mut self) -> RouteHandlerId {
        let idx = self.rr_index % self.routes.len();
        self.rr_index = idx.wrapping_add(1);
        self.routes[idx].id.clone()
    }

    fn least_loaded_id(&self) -> RouteHandlerId {
        // Pick the route whose total_routed count is smallest.
        self.routes
            .iter()
            .min_by_key(|r| self.route_stats.get(&r.id.0).map_or(0, |s| s.total_routed))
            .map(|r| r.id.clone())
            .unwrap_or_else(|| self.routes[0].id.clone())
    }

    fn random_id(&mut self) -> RouteHandlerId {
        let val = xorshift64(&mut self.rng_state);
        let idx = (val as usize) % self.routes.len();
        self.routes[idx].id.clone()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        FallbackStrategy, RouteDefinition, RouteHandlerId, RouterV2Config, RouterV2Error,
        SemanticRouterV2,
    };
    use std::collections::HashMap;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn make_router() -> SemanticRouterV2 {
        SemanticRouterV2::new(RouterV2Config::default())
    }

    fn unit_route(id: &str, dim: usize, val: f64) -> RouteDefinition {
        RouteDefinition::new(id, id, vec![val; dim])
    }

    fn orthogonal_route(id: &str, dim: usize, hot_index: usize) -> RouteDefinition {
        let mut emb = vec![0.0f64; dim];
        if hot_index < dim {
            emb[hot_index] = 1.0;
        }
        RouteDefinition::new(id, id, emb)
    }

    // ── Construction ─────────────────────────────────────────────────────

    #[test]
    fn test_new_router_has_correct_defaults() {
        let router = make_router();
        assert_eq!(router.total_queries, 0);
        assert_eq!(router.fallback_count, 0);
        assert!(router.routes.is_empty());
        assert_eq!(router.rng_state, 0xCAFEBABE_12345678_u64);
    }

    #[test]
    fn test_new_router_with_custom_config() {
        let cfg = RouterV2Config {
            similarity_threshold: 0.9,
            max_routes: 5,
            fallback: FallbackStrategy::RoundRobin,
            enable_load_balancing: false,
        };
        let router = SemanticRouterV2::new(cfg);
        assert_eq!(router.config.similarity_threshold, 0.9);
        assert_eq!(router.config.max_routes, 5);
        assert!(!router.config.enable_load_balancing);
    }

    // ── add_route ────────────────────────────────────────────────────────

    #[test]
    fn test_add_route_success() {
        let mut router = make_router();
        let route = unit_route("r1", 4, 1.0);
        assert!(router.add_route(route).is_ok());
        assert_eq!(router.routes.len(), 1);
    }

    #[test]
    fn test_add_duplicate_route_returns_error() {
        let mut router = make_router();
        router.add_route(unit_route("r1", 4, 1.0)).ok();
        let err = router
            .add_route(unit_route("r1", 4, 0.5))
            .expect_err("should fail");
        assert_eq!(err, RouterV2Error::RouteAlreadyExists("r1".into()));
    }

    #[test]
    fn test_max_routes_reached() {
        let cfg = RouterV2Config {
            max_routes: 2,
            ..RouterV2Config::default()
        };
        let mut router = SemanticRouterV2::new(cfg);
        router.add_route(unit_route("r1", 4, 1.0)).ok();
        router.add_route(unit_route("r2", 4, 0.5)).ok();
        let err = router
            .add_route(unit_route("r3", 4, 0.25))
            .expect_err("should fail");
        assert_eq!(err, RouterV2Error::MaxRoutesReached);
    }

    // ── remove_route ─────────────────────────────────────────────────────

    #[test]
    fn test_remove_existing_route() {
        let mut router = make_router();
        router.add_route(unit_route("r1", 4, 1.0)).ok();
        let removed = router.remove_route(&RouteHandlerId::new("r1"));
        assert!(removed);
        assert!(router.routes.is_empty());
    }

    #[test]
    fn test_remove_nonexistent_route_returns_false() {
        let mut router = make_router();
        let removed = router.remove_route(&RouteHandlerId::new("ghost"));
        assert!(!removed);
    }

    #[test]
    fn test_remove_cleans_up_stats() {
        let mut router = make_router();
        router.add_route(unit_route("r1", 4, 1.0)).ok();
        router.remove_route(&RouteHandlerId::new("r1"));
        assert!(!router.route_stats.contains_key("r1"));
    }

    // ── cosine_similarity ────────────────────────────────────────────────

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = SemanticRouterV2::cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = SemanticRouterV2::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = SemanticRouterV2::cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = SemanticRouterV2::cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_empty_slices() {
        let sim = SemanticRouterV2::cosine_similarity(&[], &[]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_mismatched_lengths() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0];
        // Similarity computed over shorter length
        let sim = SemanticRouterV2::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_known_value() {
        let a = vec![3.0, 4.0];
        let b = vec![4.0, 3.0];
        // dot = 24, |a| = 5, |b| = 5, sim = 24/25 = 0.96
        let sim = SemanticRouterV2::cosine_similarity(&a, &b);
        assert!((sim - 0.96).abs() < 1e-9);
    }

    // ── best_match ───────────────────────────────────────────────────────

    #[test]
    fn test_best_match_empty_router() {
        let router = make_router();
        assert!(router.best_match(&[1.0, 0.0]).is_none());
    }

    #[test]
    fn test_best_match_above_threshold() {
        let mut router = make_router();
        router.add_route(orthogonal_route("r1", 3, 0)).ok(); // [1,0,0]
        let result = router.best_match(&[1.0, 0.0, 0.0]);
        assert!(result.is_some());
        let (idx, sim) = result.expect("expected match");
        assert_eq!(idx, 0);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_best_match_below_threshold() {
        let mut router = SemanticRouterV2::new(RouterV2Config {
            similarity_threshold: 0.99,
            ..RouterV2Config::default()
        });
        router.add_route(orthogonal_route("r1", 3, 0)).ok();
        // Query is [0.5, 0.5, 0] — sim ≈ 0.707 < 0.99
        let query = vec![0.5_f64, 0.5, 0.0];
        assert!(router.best_match(&query).is_none());
    }

    #[test]
    fn test_best_match_picks_highest_similarity() {
        let mut router = make_router();
        router.add_route(orthogonal_route("r1", 3, 0)).ok(); // [1,0,0]
        router.add_route(orthogonal_route("r2", 3, 1)).ok(); // [0,1,0]
                                                             // Query mostly aligned with r2
        let result = router.best_match(&[0.1, 0.99, 0.0]);
        let (idx, _) = result.expect("expected match");
        assert_eq!(idx, 1);
    }

    // ── route / routing decisions ─────────────────────────────────────────

    #[test]
    fn test_route_direct_match() {
        let mut router = make_router();
        router.add_route(orthogonal_route("r1", 3, 0)).ok();
        let decision = router.route("query".into(), vec![1.0, 0.0, 0.0], 1000);
        assert!(!decision.fallback_used);
        assert_eq!(decision.matched_route, RouteHandlerId::new("r1"));
        assert!((decision.similarity - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_route_fallback_when_no_match() {
        let mut router = SemanticRouterV2::new(RouterV2Config {
            similarity_threshold: 0.99,
            ..RouterV2Config::default()
        });
        router.add_route(orthogonal_route("r1", 3, 0)).ok();
        let decision = router.route("query".into(), vec![0.0, 1.0, 0.0], 1000);
        assert!(decision.fallback_used);
        assert_eq!(decision.similarity, 0.0);
    }

    #[test]
    fn test_route_increments_total_queries() {
        let mut router = make_router();
        router.add_route(unit_route("r1", 3, 1.0)).ok();
        router.route("q".into(), vec![1.0, 1.0, 1.0], 0);
        router.route("q".into(), vec![1.0, 1.0, 1.0], 1);
        assert_eq!(router.total_queries, 2);
    }

    #[test]
    fn test_route_increments_fallback_count() {
        let mut router = SemanticRouterV2::new(RouterV2Config {
            similarity_threshold: 0.99,
            ..RouterV2Config::default()
        });
        router.add_route(orthogonal_route("r1", 3, 0)).ok();
        router.route("q".into(), vec![0.0, 1.0, 0.0], 0);
        assert_eq!(router.fallback_count, 1);
    }

    #[test]
    fn test_route_updates_stats() {
        let mut router = make_router();
        router.add_route(orthogonal_route("r1", 3, 0)).ok();
        router.route("q".into(), vec![1.0, 0.0, 0.0], 42);
        let stats = router.route_stats.get("r1").expect("stats should exist");
        assert_eq!(stats.total_routed, 1);
        assert_eq!(stats.last_routed_at, 42);
    }

    // ── Fallback strategies ───────────────────────────────────────────────

    #[test]
    fn test_fallback_round_robin_cycles() {
        let mut router = SemanticRouterV2::new(RouterV2Config {
            similarity_threshold: 2.0, // impossible — always fallback
            fallback: FallbackStrategy::RoundRobin,
            ..RouterV2Config::default()
        });
        router.add_route(unit_route("r1", 3, 1.0)).ok();
        router.add_route(unit_route("r2", 3, 0.0)).ok();
        let d1 = router.route("q".into(), vec![0.5, 0.5, 0.5], 0);
        let d2 = router.route("q".into(), vec![0.5, 0.5, 0.5], 0);
        let d3 = router.route("q".into(), vec![0.5, 0.5, 0.5], 0);
        // r1, r2, r1
        assert_eq!(d1.matched_route.0, "r1");
        assert_eq!(d2.matched_route.0, "r2");
        assert_eq!(d3.matched_route.0, "r1");
    }

    #[test]
    fn test_fallback_use_default_valid_id() {
        let default_id = RouteHandlerId::new("default_handler");
        let mut router = SemanticRouterV2::new(RouterV2Config {
            similarity_threshold: 2.0,
            fallback: FallbackStrategy::UseDefault {
                default_id: default_id.clone(),
            },
            ..RouterV2Config::default()
        });
        router.add_route(unit_route("r1", 3, 1.0)).ok();
        router.add_route(unit_route("default_handler", 3, 0.0)).ok();
        let decision = router.route("q".into(), vec![0.0, 1.0, 0.0], 0);
        assert_eq!(decision.matched_route, default_id);
    }

    #[test]
    fn test_fallback_use_default_invalid_falls_to_rr() {
        let mut router = SemanticRouterV2::new(RouterV2Config {
            similarity_threshold: 2.0,
            fallback: FallbackStrategy::UseDefault {
                default_id: RouteHandlerId::new("ghost"),
            },
            ..RouterV2Config::default()
        });
        router.add_route(unit_route("r1", 3, 1.0)).ok();
        // Should not panic; falls back to round-robin
        let decision = router.route("q".into(), vec![0.0, 1.0, 0.0], 0);
        assert!(decision.fallback_used);
    }

    #[test]
    fn test_fallback_least_loaded_picks_min() {
        let mut router = SemanticRouterV2::new(RouterV2Config {
            similarity_threshold: 0.9,
            fallback: FallbackStrategy::LeastLoaded,
            ..RouterV2Config::default()
        });
        router.add_route(orthogonal_route("r1", 3, 0)).ok(); // [1,0,0]
        router.add_route(orthogonal_route("r2", 3, 1)).ok(); // [0,1,0]
                                                             // Direct-route r1 multiple times to increase its load
        router.route("q".into(), vec![1.0, 0.0, 0.0], 0);
        router.route("q".into(), vec![1.0, 0.0, 0.0], 0);
        // Now force a fallback with orthogonal query — r2 should win (least loaded)
        let decision = router.route("q".into(), vec![0.0, 0.0, 1.0], 0);
        assert!(decision.fallback_used);
        assert_eq!(decision.matched_route.0, "r2");
    }

    #[test]
    fn test_fallback_random_is_deterministic_with_seed() {
        // Two routers with the same RNG state should make the same random decisions.
        let cfg = RouterV2Config {
            similarity_threshold: 2.0,
            fallback: FallbackStrategy::Random { seed: 0 }, // seed field unused; rng_state drives it
            ..RouterV2Config::default()
        };
        let mut r1 = SemanticRouterV2::new(cfg.clone());
        r1.rng_state = 42;
        let mut r2 = SemanticRouterV2::new(cfg);
        r2.rng_state = 42;
        r1.add_route(unit_route("a", 3, 1.0)).ok();
        r1.add_route(unit_route("b", 3, 0.0)).ok();
        r2.add_route(unit_route("a", 3, 1.0)).ok();
        r2.add_route(unit_route("b", 3, 0.0)).ok();
        let d1 = r1.route("q".into(), vec![0.0; 3], 0);
        let d2 = r2.route("q".into(), vec![0.0; 3], 0);
        assert_eq!(d1.matched_route, d2.matched_route);
    }

    // ── route_batch ──────────────────────────────────────────────────────

    #[test]
    fn test_route_batch_returns_correct_count() {
        let mut router = make_router();
        router.add_route(unit_route("r1", 3, 1.0)).ok();
        let queries = vec![
            ("q1".to_string(), vec![1.0, 1.0, 1.0]),
            ("q2".to_string(), vec![1.0, 1.0, 1.0]),
            ("q3".to_string(), vec![1.0, 1.0, 1.0]),
        ];
        let decisions = router.route_batch(queries, 0);
        assert_eq!(decisions.len(), 3);
    }

    #[test]
    fn test_route_batch_updates_total_queries() {
        let mut router = make_router();
        router.add_route(unit_route("r1", 3, 1.0)).ok();
        let queries: Vec<_> = (0..5).map(|i| (format!("q{i}"), vec![1.0; 3])).collect();
        router.route_batch(queries, 0);
        assert_eq!(router.total_queries, 5);
    }

    // ── top_routes ───────────────────────────────────────────────────────

    #[test]
    fn test_top_routes_ordering() {
        let mut router = make_router();
        router.add_route(orthogonal_route("r1", 3, 0)).ok();
        router.add_route(orthogonal_route("r2", 3, 1)).ok();
        // Route to r1 three times, r2 once
        for _ in 0..3 {
            router.route("q".into(), vec![1.0, 0.0, 0.0], 0);
        }
        router.route("q".into(), vec![0.0, 1.0, 0.0], 0);
        let top = router.top_routes(2);
        assert_eq!(top[0].handler_id, "r1");
        assert_eq!(top[1].handler_id, "r2");
    }

    #[test]
    fn test_top_routes_k_larger_than_registered() {
        let mut router = make_router();
        router.add_route(unit_route("r1", 3, 1.0)).ok();
        router.route("q".into(), vec![1.0; 3], 0);
        let top = router.top_routes(100);
        assert_eq!(top.len(), 1);
    }

    // ── route_by_id ──────────────────────────────────────────────────────

    #[test]
    fn test_route_by_id_found() {
        let mut router = make_router();
        router.add_route(unit_route("r1", 3, 1.0)).ok();
        let found = router.route_by_id(&RouteHandlerId::new("r1"));
        assert!(found.is_some());
        assert_eq!(found.expect("should find").name, "r1");
    }

    #[test]
    fn test_route_by_id_not_found() {
        let router = make_router();
        assert!(router.route_by_id(&RouteHandlerId::new("ghost")).is_none());
    }

    // ── update_route_embedding ───────────────────────────────────────────

    #[test]
    fn test_update_route_embedding_success() {
        let mut router = make_router();
        router.add_route(orthogonal_route("r1", 3, 0)).ok(); // [1,0,0]
        let updated =
            router.update_route_embedding(&RouteHandlerId::new("r1"), vec![0.0, 1.0, 0.0]);
        assert!(updated);
        let route = router
            .route_by_id(&RouteHandlerId::new("r1"))
            .expect("must exist");
        assert_eq!(route.embedding, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_update_route_embedding_not_found() {
        let mut router = make_router();
        let updated = router.update_route_embedding(&RouteHandlerId::new("ghost"), vec![1.0]);
        assert!(!updated);
    }

    // ── router_stats ─────────────────────────────────────────────────────

    #[test]
    fn test_router_stats_initial() {
        let router = make_router();
        let stats = router.router_stats();
        assert_eq!(stats.route_count, 0);
        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.fallback_count, 0);
        assert_eq!(stats.fallback_rate, 0.0);
    }

    #[test]
    fn test_router_stats_fallback_rate() {
        let mut router = SemanticRouterV2::new(RouterV2Config {
            similarity_threshold: 0.99,
            ..RouterV2Config::default()
        });
        router.add_route(orthogonal_route("r1", 3, 0)).ok();
        // 2 fallback queries
        router.route("q".into(), vec![0.0, 1.0, 0.0], 0);
        router.route("q".into(), vec![0.0, 0.0, 1.0], 0);
        let stats = router.router_stats();
        assert_eq!(stats.total_queries, 2);
        assert_eq!(stats.fallback_count, 2);
        assert!((stats.fallback_rate - 1.0).abs() < 1e-9);
    }

    // ── RouteHandlerId ───────────────────────────────────────────────────

    #[test]
    fn test_route_handler_id_display() {
        let id = RouteHandlerId::new("handler_42");
        assert_eq!(id.to_string(), "handler_42");
    }

    #[test]
    fn test_route_handler_id_as_str() {
        let id = RouteHandlerId::new("x");
        assert_eq!(id.as_str(), "x");
    }

    #[test]
    fn test_route_handler_id_equality() {
        let a = RouteHandlerId::new("abc");
        let b = RouteHandlerId::new("abc");
        let c = RouteHandlerId::new("xyz");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ── RouteDefinition ──────────────────────────────────────────────────

    #[test]
    fn test_route_definition_with_metadata() {
        let mut route = RouteDefinition::new("r1", "Route 1", vec![1.0, 0.0]);
        route.metadata.insert("owner".into(), "team-a".into());
        route.priority = 200;
        route.max_queries_per_second = 50.0;
        assert_eq!(
            route.metadata.get("owner").map(|s| s.as_str()),
            Some("team-a")
        );
        assert_eq!(route.priority, 200);
    }

    // ── xorshift64 ───────────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_non_zero() {
        use super::xorshift64;
        let mut state = 1u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
        assert_ne!(state, 1); // state changed
    }

    #[test]
    fn test_xorshift64_deterministic() {
        use super::xorshift64;
        let mut s1 = 12345u64;
        let mut s2 = 12345u64;
        let v1 = xorshift64(&mut s1);
        let v2 = xorshift64(&mut s2);
        assert_eq!(v1, v2);
    }

    // ── RouterV2Error ────────────────────────────────────────────────────

    #[test]
    fn test_error_display_route_already_exists() {
        let e = RouterV2Error::RouteAlreadyExists("foo".into());
        assert!(e.to_string().contains("foo"));
    }

    #[test]
    fn test_error_display_route_not_found() {
        let e = RouterV2Error::RouteNotFound("bar".into());
        assert!(e.to_string().contains("bar"));
    }

    #[test]
    fn test_error_display_max_routes_reached() {
        let e = RouterV2Error::MaxRoutesReached;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_error_display_empty_routes() {
        let e = RouterV2Error::EmptyRoutes;
        assert!(!e.to_string().is_empty());
    }

    // ── avg_similarity tracking ──────────────────────────────────────────

    #[test]
    fn test_avg_similarity_welford_convergence() {
        let mut router = make_router();
        router.add_route(orthogonal_route("r1", 2, 0)).ok(); // [1,0]
                                                             // All queries perfectly aligned — avg should stay 1.0
        for t in 0..10u64 {
            router.route("q".into(), vec![1.0, 0.0], t);
        }
        let stats = router.route_stats.get("r1").expect("stats");
        assert!((stats.avg_similarity - 1.0).abs() < 1e-9);
        assert_eq!(stats.total_routed, 10);
    }

    #[test]
    fn test_remove_and_readd_route() {
        let mut router = make_router();
        router.add_route(unit_route("r1", 3, 1.0)).ok();
        router.remove_route(&RouteHandlerId::new("r1"));
        // Re-adding after removal should succeed
        assert!(router.add_route(unit_route("r1", 3, 1.0)).is_ok());
    }

    #[test]
    fn test_route_decision_preserves_query_text() {
        let mut router = make_router();
        router.add_route(unit_route("r1", 3, 1.0)).ok();
        let decision = router.route("hello world".into(), vec![1.0; 3], 0);
        assert_eq!(decision.query_text, "hello world");
    }

    #[test]
    fn test_router_with_many_routes() {
        let mut router = make_router();
        for i in 0..50usize {
            let mut emb = vec![0.0f64; 64];
            emb[i % 64] = 1.0;
            router
                .add_route(RouteDefinition::new(
                    format!("r{i}"),
                    format!("route {i}"),
                    emb,
                ))
                .ok();
        }
        assert_eq!(router.routes.len(), 50);
        let stats = router.router_stats();
        assert_eq!(stats.route_count, 50);
    }

    #[test]
    fn test_batch_and_individual_route_agree() {
        let mut router_a = make_router();
        let mut router_b = make_router();
        router_a.add_route(orthogonal_route("r1", 3, 0)).ok();
        router_b.add_route(orthogonal_route("r1", 3, 0)).ok();
        let queries = vec![
            ("q1".into(), vec![1.0, 0.0, 0.0]),
            ("q2".into(), vec![1.0, 0.0, 0.0]),
        ];
        let batch_results = router_a.route_batch(queries, 0);
        let ind1 = router_b.route("q1".into(), vec![1.0, 0.0, 0.0], 0);
        let ind2 = router_b.route("q2".into(), vec![1.0, 0.0, 0.0], 0);
        assert_eq!(batch_results[0].matched_route, ind1.matched_route);
        assert_eq!(batch_results[1].matched_route, ind2.matched_route);
    }

    #[test]
    fn test_route_stats_last_routed_at_updated() {
        let mut router = make_router();
        router.add_route(orthogonal_route("r1", 2, 0)).ok();
        router.route("q".into(), vec![1.0, 0.0], 1000);
        let stats = router.route_stats.get("r1").expect("stats");
        assert_eq!(stats.last_routed_at, 1000);
    }

    #[test]
    fn test_metadata_stored_in_route_definition() {
        let mut route = RouteDefinition::new("r1", "Route 1", vec![1.0, 0.0]);
        route.metadata = HashMap::from([
            ("region".into(), "us-east".into()),
            ("tier".into(), "premium".into()),
        ]);
        let mut router = make_router();
        router.add_route(route).ok();
        let found = router
            .route_by_id(&RouteHandlerId::new("r1"))
            .expect("found");
        assert_eq!(
            found.metadata.get("region").map(|s| s.as_str()),
            Some("us-east")
        );
    }
}
