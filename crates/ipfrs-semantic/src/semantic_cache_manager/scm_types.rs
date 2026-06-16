// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash — deterministic, allocation-free.
#[inline]
pub(super) fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

/// Cosine similarity in [-1, 1]; returns 0.0 on empty or zero vectors.
#[inline]
pub(super) fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-10 || nb < 1e-10 {
        0.0
    } else {
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }
}

/// Xorshift64 PRNG — used only internally for tie-breaking during eviction.
#[inline]
pub(super) fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Uniform f64 in [0, 1) from xorshift64.
#[allow(dead_code)]
#[inline]
pub(super) fn xorshift_f64(state: &mut u64) -> f64 {
    (xorshift64(state) >> 11) as f64 / (1u64 << 53) as f64
}

// ──────────────────────────────────────────────────────────────────────────────
// CacheKey
// ──────────────────────────────────────────────────────────────────────────────

/// Cache key: combines the raw query text, its embedding, and a fast FNV-1a hash
/// of the query text used for exact-match lookups.
#[derive(Debug, Clone)]
pub struct ScmCacheKey {
    /// Original query text.
    pub query_text: String,
    /// Embedding vector for the query.
    pub embedding: Vec<f64>,
    /// FNV-1a 64-bit hash of `query_text`.
    pub query_hash: u64,
}

impl ScmCacheKey {
    /// Construct a [`ScmCacheKey`], computing `query_hash` automatically.
    pub fn new(query_text: String, embedding: Vec<f64>) -> Self {
        let query_hash = fnv1a_64(query_text.as_bytes());
        Self {
            query_text,
            embedding,
            query_hash,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CacheEntry
// ──────────────────────────────────────────────────────────────────────────────

/// A single cached result, keyed by a [`ScmCacheKey`].
#[derive(Debug, Clone)]
pub struct ScmCacheEntry {
    /// The cache key (query text + embedding + hash).
    pub key: ScmCacheKey,
    /// Serialised result payload.
    pub result: Vec<u8>,
    /// Microsecond timestamp when this entry was inserted (caller-supplied).
    pub inserted_at: u64,
    /// Microsecond timestamp of the last access (updated on every hit).
    pub last_accessed: u64,
    /// How many times this entry has been accessed (hits).
    pub access_count: u64,
    /// Optional TTL in microseconds from `inserted_at`; `None` = immortal.
    pub ttl_us: Option<u64>,
    /// Similarity of the original query to itself — always 1.0 at insertion.
    pub similarity_score: f64,
}

impl ScmCacheEntry {
    /// Total heap bytes consumed by this entry (approximate).
    pub fn byte_size(&self) -> usize {
        self.key.query_text.len()
            + self.key.embedding.len() * std::mem::size_of::<f64>()
            + self.result.len()
            + std::mem::size_of::<Self>()
    }

    /// Returns `true` when this entry has expired at `now` (microseconds).
    #[inline]
    pub fn is_expired(&self, now: u64) -> bool {
        match self.ttl_us {
            Some(ttl) => now.saturating_sub(self.inserted_at) >= ttl,
            None => false,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CacheHit
// ──────────────────────────────────────────────────────────────────────────────

/// Result returned by [`super::SemanticCacheManager::lookup`].
#[derive(Debug, Clone, PartialEq)]
pub enum ScmCacheHit {
    /// The query text was an exact match (same FNV-1a hash and text).
    Exact {
        /// Internal entry identifier.
        entry_id: u64,
    },
    /// No exact match, but a semantically similar entry was found.
    Semantic {
        /// Internal entry identifier.
        entry_id: u64,
        /// Cosine similarity of the incoming query to the cached query.
        similarity: f64,
        /// The original query text that produced this cached result.
        original_query: String,
    },
    /// No cached entry met the threshold.
    Miss,
}

// ──────────────────────────────────────────────────────────────────────────────
// EvictionStrategy
// ──────────────────────────────────────────────────────────────────────────────

/// Determines which entry to remove when the cache is full.
#[derive(Debug, Clone)]
pub enum ScmEvictionStrategy {
    /// Evict the entry least recently accessed.
    LRU,
    /// Evict the entry with the lowest access count.
    LFU,
    /// Evict from the largest semantic cluster (over-represented embeddings first).
    SemanticCluster,
    /// Evict the entry closest to expiry; fall back to LRU when no TTLs are set.
    TTLFirst,
    /// Weighted score: `weight * recency_score + (1 - weight) * frequency_score`.
    /// `weight = 1.0` behaves like LRU; `weight = 0.0` behaves like LFU.
    HybridScore(f64),
}

// ──────────────────────────────────────────────────────────────────────────────
// CacheConfig
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for [`super::SemanticCacheManager`].
#[derive(Debug, Clone)]
pub struct ScmCacheConfig {
    /// Maximum number of entries allowed in the cache.
    pub max_entries: usize,
    /// Maximum total payload bytes allowed across all entries.
    pub max_bytes: usize,
    /// Minimum cosine similarity required to declare a semantic hit (0–1).
    pub semantic_threshold: f64,
    /// Default TTL in microseconds for new entries; `None` = no expiry.
    pub ttl_us: Option<u64>,
    /// Eviction strategy used when the cache is over capacity.
    pub strategy: ScmEvictionStrategy,
    /// Radius (in cosine distance) used to group entries into clusters for
    /// [`ScmEvictionStrategy::SemanticCluster`] eviction and [`cluster_stats`][super::SemanticCacheManager::cluster_stats].
    pub cluster_radius: f64,
}

impl Default for ScmCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1024,
            max_bytes: 256 * 1024 * 1024,
            semantic_threshold: 0.90,
            ttl_us: None,
            strategy: ScmEvictionStrategy::LRU,
            cluster_radius: 0.20,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CacheStats
// ──────────────────────────────────────────────────────────────────────────────

/// Runtime statistics snapshot from [`super::SemanticCacheManager::stats`].
#[derive(Debug, Clone, Default)]
pub struct ScmCacheStats {
    /// Total number of successful insertions (not counting replacements).
    pub total_insertions: u64,
    /// Total exact-match hits.
    pub exact_hits: u64,
    /// Total semantic hits (similarity above threshold, but not exact).
    pub semantic_hits: u64,
    /// Total cache misses.
    pub misses: u64,
    /// Total evictions performed.
    pub evictions: u64,
    /// Current number of live entries.
    pub current_entries: usize,
    /// Current total byte consumption.
    pub current_bytes: usize,
    /// `semantic_hits / (exact_hits + semantic_hits + misses)`, or 0.0.
    pub semantic_hit_rate: f64,
    /// Running average of cosine similarity on all semantic hits.
    pub avg_similarity_on_semantic_hit: f64,
}

// ──────────────────────────────────────────────────────────────────────────────
// CacheError
// ──────────────────────────────────────────────────────────────────────────────

/// Errors returned by [`super::SemanticCacheManager`] operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ScmCacheError {
    /// The entry payload exceeds the cache's `max_bytes` budget.
    EntryTooLarge(usize),
    /// The cache is at capacity and eviction was unable to free space.
    CacheAtCapacity,
    /// No entry with the given `entry_id` was found.
    EntryNotFound(u64),
    /// The provided embedding dimension does not match existing entries.
    DimensionMismatch {
        /// The dimension inferred from existing entries.
        expected: usize,
        /// The dimension of the supplied embedding.
        got: usize,
    },
    /// The entry was found but has already expired.
    TtlExpired(u64),
}

impl std::fmt::Display for ScmCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EntryTooLarge(b) => write!(f, "entry is too large ({b} bytes)"),
            Self::CacheAtCapacity => write!(f, "cache is at capacity and could not evict"),
            Self::EntryNotFound(id) => write!(f, "entry {id} not found"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::TtlExpired(id) => write!(f, "entry {id} has expired"),
        }
    }
}

impl std::error::Error for ScmCacheError {}

// ──────────────────────────────────────────────────────────────────────────────
// Internal slot
// ──────────────────────────────────────────────────────────────────────────────

/// Internal storage cell: an entry paired with its auto-assigned ID.
#[derive(Debug, Clone)]
pub(super) struct Slot {
    pub(super) id: u64,
    pub(super) entry: ScmCacheEntry,
}
