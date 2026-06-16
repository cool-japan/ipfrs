//! Caching support for query results and remote facts
//!
//! Provides:
//! - LRU cache for query results
//! - TTL-based cache for remote facts
//! - Thread-safe caching primitives
//! - Cache statistics

use crate::ir::{Predicate, Term};
use crate::reasoning::Substitution;
use ipfrs_core::Cid;
use lru::LruCache;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Query key for cache lookups
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QueryKey {
    /// Predicate name
    pub predicate_name: String,
    /// Ground arguments (for filtering)
    pub ground_args: Vec<GroundArg>,
}

/// Ground argument for query key
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GroundArg {
    /// String constant
    String(String),
    /// Integer constant
    Int(i64),
    /// Float constant (as bits for hashing)
    Float(u64),
    /// Variable (not ground)
    Variable,
}

impl QueryKey {
    /// Create a query key from a predicate
    pub fn from_predicate(pred: &Predicate) -> Self {
        let ground_args = pred
            .args
            .iter()
            .map(|arg| match arg {
                Term::Const(c) => match c {
                    crate::ir::Constant::String(s) => GroundArg::String(s.clone()),
                    crate::ir::Constant::Int(i) => GroundArg::Int(*i),
                    // Float is stored as String for deterministic hashing
                    crate::ir::Constant::Float(f) => {
                        let hash = f.parse::<f64>().map(|v| v.to_bits()).unwrap_or(0);
                        GroundArg::Float(hash)
                    }
                    crate::ir::Constant::Bool(b) => GroundArg::Int(if *b { 1 } else { 0 }),
                },
                Term::Var(_) | Term::Fun(_, _) | Term::Ref(_) => GroundArg::Variable,
            })
            .collect();

        Self {
            predicate_name: pred.name.clone(),
            ground_args,
        }
    }
}

/// Cached query result
#[derive(Debug, Clone)]
pub struct CachedResult {
    /// Query solutions (substitutions)
    pub solutions: Vec<Substitution>,
    /// When the result was cached
    pub cached_at: Instant,
    /// Time-to-live for this result
    pub ttl: Option<Duration>,
}

impl CachedResult {
    /// Create a new cached result
    pub fn new(solutions: Vec<Substitution>, ttl: Option<Duration>) -> Self {
        Self {
            solutions,
            cached_at: Instant::now(),
            ttl,
        }
    }

    /// Check if the cached result has expired
    #[inline]
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            self.cached_at.elapsed() > ttl
        } else {
            false
        }
    }

    /// Get remaining TTL
    #[inline]
    pub fn remaining_ttl(&self) -> Option<Duration> {
        self.ttl
            .map(|ttl| ttl.saturating_sub(self.cached_at.elapsed()))
    }
}

/// Cache statistics
#[derive(Debug, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: AtomicU64,
    /// Number of cache misses
    pub misses: AtomicU64,
    /// Number of evictions
    pub evictions: AtomicU64,
    /// Number of expirations
    pub expirations: AtomicU64,
}

impl CacheStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a hit
    #[inline]
    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a miss
    #[inline]
    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an eviction
    #[inline]
    pub fn record_eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an expiration
    #[inline]
    pub fn record_expiration(&self) {
        self.expirations.fetch_add(1, Ordering::Relaxed);
    }

    /// Get hit rate
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Get a snapshot of stats
    pub fn snapshot(&self) -> CacheStatsSnapshot {
        CacheStatsSnapshot {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            expirations: self.expirations.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatsSnapshot {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of evictions
    pub evictions: u64,
    /// Number of expirations
    pub expirations: u64,
}

impl CacheStatsSnapshot {
    /// Get hit rate
    #[inline]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// LRU cache for query results
pub struct QueryCache {
    /// The underlying LRU cache
    cache: RwLock<LruCache<QueryKey, CachedResult>>,
    /// Default TTL for cached results
    default_ttl: Option<Duration>,
    /// Cache statistics
    stats: Arc<CacheStats>,
}

impl QueryCache {
    /// Create a new query cache with the given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: RwLock::new(LruCache::new(
                NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(100).expect("100 > 0")),
            )),
            default_ttl: None,
            stats: Arc::new(CacheStats::new()),
        }
    }

    /// Create a new query cache with TTL
    pub fn with_ttl(capacity: usize, ttl: Duration) -> Self {
        Self {
            cache: RwLock::new(LruCache::new(
                NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(100).expect("100 > 0")),
            )),
            default_ttl: Some(ttl),
            stats: Arc::new(CacheStats::new()),
        }
    }

    /// Get a cached result
    #[inline]
    pub fn get(&self, key: &QueryKey) -> Option<Vec<Substitution>> {
        let mut cache = self.cache.write();

        if let Some(result) = cache.get(key) {
            if result.is_expired() {
                self.stats.record_expiration();
                cache.pop(key);
                self.stats.record_miss();
                return None;
            }
            self.stats.record_hit();
            Some(result.solutions.clone())
        } else {
            self.stats.record_miss();
            None
        }
    }

    /// Insert a result into the cache
    pub fn insert(&self, key: QueryKey, solutions: Vec<Substitution>) {
        let mut cache = self.cache.write();

        // Check if we need to evict
        if cache.len() >= cache.cap().get() {
            self.stats.record_eviction();
        }

        let result = CachedResult::new(solutions, self.default_ttl);
        cache.put(key, result);
    }

    /// Insert a result with custom TTL
    pub fn insert_with_ttl(&self, key: QueryKey, solutions: Vec<Substitution>, ttl: Duration) {
        let mut cache = self.cache.write();

        if cache.len() >= cache.cap().get() {
            self.stats.record_eviction();
        }

        let result = CachedResult::new(solutions, Some(ttl));
        cache.put(key, result);
    }

    /// Invalidate a cached result
    pub fn invalidate(&self, key: &QueryKey) -> bool {
        let mut cache = self.cache.write();
        cache.pop(key).is_some()
    }

    /// Invalidate all results for a predicate
    pub fn invalidate_predicate(&self, predicate_name: &str) {
        let mut cache = self.cache.write();
        let keys_to_remove: Vec<QueryKey> = cache
            .iter()
            .filter(|(k, _)| k.predicate_name == predicate_name)
            .map(|(k, _)| k.clone())
            .collect();

        for key in keys_to_remove {
            cache.pop(&key);
        }
    }

    /// Clear the entire cache
    pub fn clear(&self) {
        let mut cache = self.cache.write();
        cache.clear();
    }

    /// Get cache statistics
    #[inline]
    pub fn stats(&self) -> Arc<CacheStats> {
        self.stats.clone()
    }

    /// Get current cache size
    #[inline]
    pub fn len(&self) -> usize {
        self.cache.read().len()
    }

    /// Check if cache is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cache.read().is_empty()
    }

    /// Get cache capacity
    #[inline]
    pub fn capacity(&self) -> usize {
        self.cache.read().cap().get()
    }

    /// Remove expired entries
    pub fn evict_expired(&self) -> usize {
        let mut cache = self.cache.write();
        let mut expired_keys = Vec::new();

        for (key, result) in cache.iter() {
            if result.is_expired() {
                expired_keys.push(key.clone());
            }
        }

        let count = expired_keys.len();
        for key in expired_keys {
            cache.pop(&key);
            self.stats.record_expiration();
        }

        count
    }
}

impl Default for QueryCache {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Remote fact with metadata
#[derive(Debug, Clone)]
pub struct RemoteFact {
    /// The fact predicate
    pub fact: Predicate,
    /// Source peer CID
    pub source: Option<Cid>,
    /// When the fact was fetched
    pub fetched_at: Instant,
    /// Time-to-live
    pub ttl: Duration,
}

impl RemoteFact {
    /// Create a new remote fact
    pub fn new(fact: Predicate, source: Option<Cid>, ttl: Duration) -> Self {
        Self {
            fact,
            source,
            fetched_at: Instant::now(),
            ttl,
        }
    }

    /// Check if the fact has expired
    #[inline]
    pub fn is_expired(&self) -> bool {
        self.fetched_at.elapsed() > self.ttl
    }
}

/// Cache key for remote facts
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FactKey {
    /// Predicate name
    pub predicate_name: String,
    /// Serialized arguments
    pub args_hash: u64,
}

impl FactKey {
    /// Create a fact key from a predicate
    pub fn from_predicate(pred: &Predicate) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for arg in &pred.args {
            arg.hash(&mut hasher);
        }
        Self {
            predicate_name: pred.name.clone(),
            args_hash: hasher.finish(),
        }
    }
}

/// Cache for remote facts
pub struct RemoteFactCache {
    /// Facts by predicate name
    facts: RwLock<HashMap<String, Vec<RemoteFact>>>,
    /// Maximum facts per predicate
    max_per_predicate: usize,
    /// Default TTL
    default_ttl: Duration,
    /// Statistics
    stats: Arc<CacheStats>,
}

impl RemoteFactCache {
    /// Create a new remote fact cache
    pub fn new(max_per_predicate: usize, default_ttl: Duration) -> Self {
        Self {
            facts: RwLock::new(HashMap::new()),
            max_per_predicate,
            default_ttl,
            stats: Arc::new(CacheStats::new()),
        }
    }

    /// Get facts for a predicate
    pub fn get_facts(&self, predicate_name: &str) -> Vec<Predicate> {
        let facts = self.facts.read();

        if let Some(remote_facts) = facts.get(predicate_name) {
            let valid_facts: Vec<Predicate> = remote_facts
                .iter()
                .filter(|f| !f.is_expired())
                .map(|f| f.fact.clone())
                .collect();

            if valid_facts.is_empty() {
                self.stats.record_miss();
            } else {
                self.stats.record_hit();
            }

            valid_facts
        } else {
            self.stats.record_miss();
            Vec::new()
        }
    }

    /// Add a fact to the cache
    pub fn add_fact(&self, fact: Predicate, source: Option<Cid>) {
        self.add_fact_with_ttl(fact, source, self.default_ttl);
    }

    /// Add a fact with custom TTL
    pub fn add_fact_with_ttl(&self, fact: Predicate, source: Option<Cid>, ttl: Duration) {
        let mut facts = self.facts.write();
        let name = fact.name.clone();

        let remote_fact = RemoteFact::new(fact, source, ttl);

        let entry = facts.entry(name).or_default();

        // Remove expired facts
        entry.retain(|f| !f.is_expired());

        // Check capacity
        if entry.len() >= self.max_per_predicate {
            // Remove oldest
            entry.sort_by_key(|f| f.fetched_at);
            entry.remove(0);
            self.stats.record_eviction();
        }

        entry.push(remote_fact);
    }

    /// Add multiple facts
    pub fn add_facts(&self, facts: Vec<Predicate>, source: Option<Cid>) {
        for fact in facts {
            self.add_fact(fact, source);
        }
    }

    /// Invalidate facts for a predicate
    pub fn invalidate_predicate(&self, predicate_name: &str) {
        let mut facts = self.facts.write();
        facts.remove(predicate_name);
    }

    /// Clear all facts
    pub fn clear(&self) {
        let mut facts = self.facts.write();
        facts.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> Arc<CacheStats> {
        self.stats.clone()
    }

    /// Remove expired facts
    pub fn evict_expired(&self) -> usize {
        let mut facts = self.facts.write();
        let mut count = 0;

        for entry in facts.values_mut() {
            let before = entry.len();
            entry.retain(|f| !f.is_expired());
            count += before - entry.len();
        }

        for _ in 0..count {
            self.stats.record_expiration();
        }

        count
    }

    /// Get total number of cached facts
    pub fn len(&self) -> usize {
        let facts = self.facts.read();
        facts.values().map(|v| v.len()).sum()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for RemoteFactCache {
    fn default() -> Self {
        Self::new(1000, Duration::from_secs(300))
    }
}

/// Combined cache manager
pub struct CacheManager {
    /// Query result cache
    pub query_cache: QueryCache,
    /// Remote fact cache
    pub fact_cache: RemoteFactCache,
}

impl CacheManager {
    /// Create a new cache manager with default settings
    pub fn new() -> Self {
        Self {
            query_cache: QueryCache::new(10000),
            fact_cache: RemoteFactCache::new(1000, Duration::from_secs(300)),
        }
    }

    /// Create with custom settings
    pub fn with_config(
        query_capacity: usize,
        query_ttl: Option<Duration>,
        fact_capacity: usize,
        fact_ttl: Duration,
    ) -> Self {
        let query_cache = if let Some(ttl) = query_ttl {
            QueryCache::with_ttl(query_capacity, ttl)
        } else {
            QueryCache::new(query_capacity)
        };

        Self {
            query_cache,
            fact_cache: RemoteFactCache::new(fact_capacity, fact_ttl),
        }
    }

    /// Evict all expired entries
    pub fn evict_expired(&self) -> (usize, usize) {
        let queries = self.query_cache.evict_expired();
        let facts = self.fact_cache.evict_expired();
        (queries, facts)
    }

    /// Clear all caches
    pub fn clear_all(&self) {
        self.query_cache.clear();
        self.fact_cache.clear();
    }

    /// Get combined statistics
    pub fn stats(&self) -> CombinedCacheStats {
        CombinedCacheStats {
            query_stats: self.query_cache.stats().snapshot(),
            fact_stats: self.fact_cache.stats().snapshot(),
            query_cache_size: self.query_cache.len(),
            fact_cache_size: self.fact_cache.len(),
        }
    }
}

impl Default for CacheManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinedCacheStats {
    /// Query cache statistics
    pub query_stats: CacheStatsSnapshot,
    /// Fact cache statistics
    pub fact_stats: CacheStatsSnapshot,
    /// Current query cache size
    pub query_cache_size: usize,
    /// Current fact cache size
    pub fact_cache_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Constant;
    use std::thread::sleep;

    #[test]
    fn test_query_cache_basic() {
        let cache = QueryCache::new(100);

        let key = QueryKey {
            predicate_name: "test".to_string(),
            ground_args: vec![GroundArg::String("value".to_string())],
        };

        let solutions = vec![Substitution::new()];
        cache.insert(key.clone(), solutions.clone());

        let result = cache.get(&key);
        assert!(result.is_some());
        assert_eq!(result.expect("test: should succeed").len(), 1);
    }

    #[test]
    fn test_query_cache_ttl() {
        let cache = QueryCache::with_ttl(100, Duration::from_millis(50));

        let key = QueryKey {
            predicate_name: "test".to_string(),
            ground_args: vec![],
        };

        cache.insert(key.clone(), vec![Substitution::new()]);

        // Should be valid immediately
        assert!(cache.get(&key).is_some());

        // Wait for TTL to expire
        sleep(Duration::from_millis(100));

        // Should be expired now
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_query_cache_stats() {
        let cache = QueryCache::new(100);

        let key = QueryKey {
            predicate_name: "test".to_string(),
            ground_args: vec![],
        };

        // Miss
        cache.get(&key);

        // Insert and hit
        cache.insert(key.clone(), vec![]);
        cache.get(&key);

        let stats = cache.stats().snapshot();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_remote_fact_cache() {
        let cache = RemoteFactCache::new(100, Duration::from_secs(60));

        let fact = Predicate::new(
            "test".to_string(),
            vec![Term::Const(Constant::String("value".to_string()))],
        );

        cache.add_fact(fact.clone(), None);

        let facts = cache.get_facts("test");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].name, "test");
    }

    #[test]
    fn test_remote_fact_cache_ttl() {
        let cache = RemoteFactCache::new(100, Duration::from_millis(50));

        let fact = Predicate::new("test".to_string(), vec![]);
        cache.add_fact(fact, None);

        // Should be valid
        assert_eq!(cache.get_facts("test").len(), 1);

        // Wait for TTL
        sleep(Duration::from_millis(100));

        // Should be expired
        assert!(cache.get_facts("test").is_empty());
    }

    #[test]
    fn test_cache_manager() {
        let manager = CacheManager::new();

        // Test query cache
        let key = QueryKey {
            predicate_name: "test".to_string(),
            ground_args: vec![],
        };
        manager.query_cache.insert(key.clone(), vec![]);
        assert!(manager.query_cache.get(&key).is_some());

        // Test fact cache
        let fact = Predicate::new("fact".to_string(), vec![]);
        manager.fact_cache.add_fact(fact, None);
        assert_eq!(manager.fact_cache.get_facts("fact").len(), 1);

        // Test stats
        let stats = manager.stats();
        assert!(stats.query_cache_size > 0);
        assert!(stats.fact_cache_size > 0);
    }
}
