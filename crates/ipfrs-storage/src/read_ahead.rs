//! Read-Ahead Scheduler for sequential block prefetching
//!
//! This module implements a lightweight, single-threaded read-ahead scheduler
//! that tracks raw block offset access sequences and issues prefetch hints
//! based on detected access patterns (sequential, strided, repeated, random).
//!
//! # Design Notes
//!
//! - `ReadAheadScheduler` is intentionally `!Send + !Sync` (uses `Instant` in a
//!   `HashMap` without any atomic synchronization).  All state is owned by a
//!   single thread.
//! - History is capped at 64 entries (ring-buffer semantics via `VecDeque`).
//! - Prefetch cache entries are deduplicated within their TTL (default 30 s).

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

// ──────────────────────────────────────────────────────────────
// AccessPattern
// ──────────────────────────────────────────────────────────────

/// Classification of block offset access patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadAheadPattern {
    /// Every access targets the same offset.
    Repeated,
    /// Each access advances by exactly 1 block offset.
    Sequential,
    /// Each access advances by the same constant stride (> 1).
    Strided {
        /// The constant stride between consecutive accesses.
        stride: u64,
    },
    /// No discernible pattern.
    Random,
}

impl ReadAheadPattern {
    /// Analyse a history slice and return the best-fitting pattern.
    ///
    /// Requires at least 2 entries; a single-entry (or empty) slice returns
    /// `Random` because there is insufficient evidence for any pattern.
    pub fn detect(history: &[u64]) -> Self {
        if history.len() < 2 {
            return Self::Random;
        }

        // Compute successive deltas.
        let deltas: Vec<u64> = history
            .windows(2)
            .map(|w| w[1].wrapping_sub(w[0]))
            .collect();

        let first = deltas[0];

        // All deltas must equal `first` for any of the structured patterns.
        if deltas.iter().all(|&d| d == first) {
            match first {
                0 => Self::Repeated,
                1 => Self::Sequential,
                s => Self::Strided { stride: s },
            }
        } else {
            Self::Random
        }
    }
}

// Keep the public-API name the spec requires (`AccessPattern`) as an alias so
// callers that use `read_ahead::AccessPattern` compile without ambiguity.
/// Public alias for [`ReadAheadPattern`].
pub type AccessPattern = ReadAheadPattern;

// ──────────────────────────────────────────────────────────────
// PrefetchHint
// ──────────────────────────────────────────────────────────────

/// A set of block offsets that the scheduler recommends prefetching.
#[derive(Debug, Clone)]
pub struct PrefetchHint {
    /// Ordered list of block offsets to prefetch.
    pub block_offsets: Vec<u64>,
    /// Pattern that generated this hint.
    pub pattern: ReadAheadPattern,
    /// Confidence score in `[0.0, 1.0]` that the pattern will continue.
    pub confidence: f32,
}

// ──────────────────────────────────────────────────────────────
// ReadAheadStats
// ──────────────────────────────────────────────────────────────

/// Cumulative statistics for a [`ReadAheadScheduler`].
///
/// Plain struct — no atomics; the scheduler is single-threaded by design.
#[derive(Debug, Clone, Default)]
pub struct ReadAheadStats {
    /// Total number of `record_access` calls.
    pub total_accesses: u64,
    /// Total number of non-`None` hints returned by `next_hints`.
    pub total_hints_issued: u64,
    /// Total number of offsets skipped due to recent prefetch cache hits.
    pub total_deduped: u64,
}

// ──────────────────────────────────────────────────────────────
// ReadAheadScheduler
// ──────────────────────────────────────────────────────────────

/// Maximum number of offset entries kept in history.
const HISTORY_CAP: usize = 64;

/// Single-threaded read-ahead scheduler.
///
/// # Example
/// ```rust
/// use ipfrs_storage::read_ahead::{ReadAheadScheduler, ReadAheadPattern};
///
/// let mut sched = ReadAheadScheduler::new();
/// for offset in 0u64..8 {
///     sched.record_access(offset);
/// }
/// if let Some(hint) = sched.next_hints() {
///     assert_eq!(hint.pattern, ReadAheadPattern::Sequential);
/// }
/// ```
pub struct ReadAheadScheduler {
    /// Ring-buffer of the last `HISTORY_CAP` accessed offsets.
    history: VecDeque<u64>,
    /// Number of blocks to look ahead when generating hints.
    lookahead: usize,
    /// Tracks recently prefetched offsets → time they were issued.
    prefetch_cache: HashMap<u64, Instant>,
    /// How long a prefetch cache entry remains valid before eviction.
    cache_ttl: Duration,
    /// Accumulated statistics.
    stats: ReadAheadStats,
}

impl Default for ReadAheadScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl ReadAheadScheduler {
    /// Create a scheduler with default settings (`lookahead = 8`, `ttl = 30 s`).
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(HISTORY_CAP),
            lookahead: 8,
            prefetch_cache: HashMap::new(),
            cache_ttl: Duration::from_secs(30),
            stats: ReadAheadStats::default(),
        }
    }

    /// Create a scheduler with custom lookahead and TTL.
    pub fn with_config(lookahead: usize, cache_ttl: Duration) -> Self {
        Self {
            history: VecDeque::with_capacity(HISTORY_CAP),
            lookahead,
            prefetch_cache: HashMap::new(),
            cache_ttl,
            stats: ReadAheadStats::default(),
        }
    }

    // ──────────────────────────────────────────────────────────
    // Public API
    // ──────────────────────────────────────────────────────────

    /// Record a block access at the given offset.
    ///
    /// Trims history to the last `HISTORY_CAP` entries.
    pub fn record_access(&mut self, offset: u64) {
        if self.history.len() == HISTORY_CAP {
            self.history.pop_front();
        }
        self.history.push_back(offset);
        self.stats.total_accesses += 1;
    }

    /// Compute and return the next prefetch hints, or `None` if the current
    /// pattern is `Random` or history is too short to classify.
    ///
    /// Offsets that are already in the prefetch cache (within TTL) are excluded
    /// from `block_offsets` and counted in `stats.total_deduped`.
    pub fn next_hints(&mut self) -> Option<PrefetchHint> {
        // Need at least 2 data points to detect a pattern.
        if self.history.len() < 2 {
            return None;
        }

        let history_slice: Vec<u64> = self.history.iter().copied().collect();
        let pattern = ReadAheadPattern::detect(&history_slice);

        let last = *self.history.back()?;

        let (raw_offsets, confidence) = match pattern {
            ReadAheadPattern::Sequential => {
                let offsets: Vec<u64> = (1..=(self.lookahead as u64))
                    .map(|i| last.wrapping_add(i))
                    .collect();
                (offsets, 0.95_f32)
            }
            ReadAheadPattern::Strided { stride } => {
                let offsets: Vec<u64> = (1..=(self.lookahead as u64))
                    .map(|i| last.wrapping_add(stride.wrapping_mul(i)))
                    .collect();
                (offsets, 0.85_f32)
            }
            ReadAheadPattern::Repeated => {
                // Hint the same offset — useful for keeping it warm.
                (vec![last], 0.5_f32)
            }
            ReadAheadPattern::Random => {
                return None;
            }
        };

        // Deduplicate against prefetch cache.
        let now = Instant::now();
        let mut deduped_count: u64 = 0;
        let mut block_offsets = Vec::with_capacity(raw_offsets.len());

        for offset in raw_offsets {
            let cached = self
                .prefetch_cache
                .get(&offset)
                .is_some_and(|&issued| now.duration_since(issued) < self.cache_ttl);
            if cached {
                deduped_count += 1;
            } else {
                block_offsets.push(offset);
                self.prefetch_cache.insert(offset, now);
            }
        }

        self.stats.total_deduped += deduped_count;

        // If every offset was deduped, still return `Some` but with an empty
        // list so callers know the hint existed but was fully satisfied.
        let hint = PrefetchHint {
            block_offsets,
            pattern,
            confidence,
        };
        self.stats.total_hints_issued += 1;
        Some(hint)
    }

    /// Remove prefetch cache entries whose age exceeds the TTL.
    pub fn evict_stale_cache(&mut self) {
        let ttl = self.cache_ttl;
        let now = Instant::now();
        self.prefetch_cache
            .retain(|_offset, &mut issued| now.duration_since(issued) < ttl);
    }

    /// Return the current access pattern based on recorded history.
    ///
    /// Returns `Random` if fewer than 2 accesses have been recorded.
    pub fn pattern(&self) -> ReadAheadPattern {
        if self.history.len() < 2 {
            return ReadAheadPattern::Random;
        }
        let slice: Vec<u64> = self.history.iter().copied().collect();
        ReadAheadPattern::detect(&slice)
    }

    /// Immutable reference to accumulated statistics.
    pub fn stats(&self) -> &ReadAheadStats {
        &self.stats
    }

    /// Mutable reference to accumulated statistics (for resetting etc.).
    pub fn stats_mut(&mut self) -> &mut ReadAheadStats {
        &mut self.stats
    }

    /// Current number of entries in the prefetch cache (including stale ones).
    pub fn prefetch_cache_len(&self) -> usize {
        self.prefetch_cache.len()
    }
}

// ──────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    // ── AccessPattern::detect ──────────────────────────────────

    #[test]
    fn test_detect_sequential() {
        let history = vec![10, 11, 12, 13, 14];
        assert_eq!(
            ReadAheadPattern::detect(&history),
            ReadAheadPattern::Sequential
        );
    }

    #[test]
    fn test_detect_strided() {
        let history = vec![0, 4, 8, 12, 16];
        assert_eq!(
            ReadAheadPattern::detect(&history),
            ReadAheadPattern::Strided { stride: 4 }
        );
    }

    #[test]
    fn test_detect_repeated() {
        let history = vec![7, 7, 7, 7, 7];
        assert_eq!(
            ReadAheadPattern::detect(&history),
            ReadAheadPattern::Repeated
        );
    }

    #[test]
    fn test_detect_random() {
        let history = vec![1, 5, 2, 9, 3];
        assert_eq!(ReadAheadPattern::detect(&history), ReadAheadPattern::Random);
    }

    #[test]
    fn test_detect_single_entry_is_random() {
        let history = vec![42];
        assert_eq!(ReadAheadPattern::detect(&history), ReadAheadPattern::Random);
    }

    #[test]
    fn test_detect_empty_is_random() {
        assert_eq!(ReadAheadPattern::detect(&[]), ReadAheadPattern::Random);
    }

    #[test]
    fn test_detect_two_entry_sequential() {
        let history = vec![100, 101];
        assert_eq!(
            ReadAheadPattern::detect(&history),
            ReadAheadPattern::Sequential
        );
    }

    // ── Sequential prefetch ────────────────────────────────────

    #[test]
    fn test_sequential_prefetch_offsets() {
        let mut sched = ReadAheadScheduler::new();
        for i in 0u64..4 {
            sched.record_access(i);
        }
        let hint = sched.next_hints().expect("expected a hint");
        assert_eq!(hint.pattern, ReadAheadPattern::Sequential);
        // Last recorded offset = 3; next 8 should be 4..=11
        let expected: Vec<u64> = (4..=11).collect();
        assert_eq!(hint.block_offsets, expected);
    }

    #[test]
    fn test_sequential_confidence_high() {
        let mut sched = ReadAheadScheduler::new();
        for i in 0u64..4 {
            sched.record_access(i);
        }
        let hint = sched.next_hints().expect("expected a hint");
        assert!(
            hint.confidence >= 0.9,
            "sequential confidence should be high"
        );
    }

    // ── Strided prefetch ───────────────────────────────────────

    #[test]
    fn test_strided_prefetch_offsets() {
        let mut sched = ReadAheadScheduler::new();
        for i in 0u64..4 {
            sched.record_access(i * 8);
        }
        let hint = sched.next_hints().expect("expected a hint");
        assert_eq!(hint.pattern, ReadAheadPattern::Strided { stride: 8 });
        // Last = 24; next 8 strided = 32, 40, 48, 56, 64, 72, 80, 88
        let expected: Vec<u64> = (1..=8).map(|i| 24 + i * 8).collect();
        assert_eq!(hint.block_offsets, expected);
    }

    // ── Random returns None ────────────────────────────────────

    #[test]
    fn test_random_returns_none() {
        let mut sched = ReadAheadScheduler::new();
        for &offset in &[1u64, 100, 5, 77, 42] {
            sched.record_access(offset);
        }
        assert!(
            sched.next_hints().is_none(),
            "random pattern should produce no hint"
        );
    }

    // ── Repeated hint ──────────────────────────────────────────

    #[test]
    fn test_repeated_returns_same_offset_hint() {
        let mut sched = ReadAheadScheduler::new();
        for _ in 0..5 {
            sched.record_access(99);
        }
        let hint = sched
            .next_hints()
            .expect("expected a hint for repeated pattern");
        assert_eq!(hint.pattern, ReadAheadPattern::Repeated);
        assert_eq!(hint.block_offsets, vec![99]);
        assert!(
            (hint.confidence - 0.5).abs() < f32::EPSILON,
            "repeated confidence should be 0.5"
        );
    }

    // ── Dedup / prefetch cache ─────────────────────────────────

    #[test]
    fn test_dedup_skips_recently_prefetched() {
        let mut sched = ReadAheadScheduler::new();
        // Seed sequential pattern starting at 0.
        for i in 0u64..4 {
            sched.record_access(i);
        }

        // First call — should return 8 fresh offsets.
        let hint1 = sched.next_hints().expect("expected hint");
        assert_eq!(hint1.block_offsets.len(), 8);
        assert_eq!(sched.stats().total_deduped, 0);

        // Record one more sequential offset so the window advances by 1.
        sched.record_access(4);

        // Second call — new "last" is 4, raw hints = 5..12 (8 offsets).
        // Offsets 5..11 were already cached from first call; only 12 is new.
        let hint2 = sched.next_hints().expect("expected hint");
        // 7 of the 8 offsets (5–11) should be deduped.
        assert_eq!(
            hint2.block_offsets.len(),
            1,
            "only offset 12 should be fresh"
        );
        assert_eq!(sched.stats().total_deduped, 7);
    }

    // ── evict_stale_cache ──────────────────────────────────────

    #[test]
    fn test_evict_stale_cache_removes_expired() {
        // Use a very short TTL so we can expire entries without sleeping long.
        let mut sched = ReadAheadScheduler::with_config(4, Duration::from_millis(50));

        // Drive a sequential pattern to populate the cache.
        for i in 0u64..4 {
            sched.record_access(i);
        }
        let _hint = sched.next_hints();
        assert!(sched.prefetch_cache_len() > 0, "cache should be populated");

        // Wait for entries to expire.
        thread::sleep(Duration::from_millis(80));

        sched.evict_stale_cache();
        assert_eq!(
            sched.prefetch_cache_len(),
            0,
            "all entries should be evicted"
        );
    }

    // ── History capped at 64 ───────────────────────────────────

    #[test]
    fn test_history_capped_at_64() {
        let mut sched = ReadAheadScheduler::new();
        for i in 0u64..200 {
            sched.record_access(i);
        }
        // The VecDeque should never exceed HISTORY_CAP.
        assert_eq!(sched.history.len(), 64);
        // Last entry should be 199.
        assert_eq!(*sched.history.back().expect("non-empty"), 199);
        // First entry should be 200 - 64 = 136.
        assert_eq!(*sched.history.front().expect("non-empty"), 136);
    }

    // ── Stats accumulate ──────────────────────────────────────

    #[test]
    fn test_stats_accumulate() {
        let mut sched = ReadAheadScheduler::new();

        // 10 accesses.
        for i in 0u64..10 {
            sched.record_access(i);
        }
        assert_eq!(sched.stats().total_accesses, 10);

        // Two separate hint calls on a sequential pattern.
        let h1 = sched.next_hints();
        let h2 = sched.next_hints();

        assert!(h1.is_some());
        assert!(h2.is_some());
        assert_eq!(sched.stats().total_hints_issued, 2);
    }

    // ── pattern() accessor ────────────────────────────────────

    #[test]
    fn test_pattern_accessor_sequential() {
        let mut sched = ReadAheadScheduler::new();
        for i in 0u64..6 {
            sched.record_access(i);
        }
        assert_eq!(sched.pattern(), ReadAheadPattern::Sequential);
    }

    #[test]
    fn test_pattern_accessor_insufficient_history() {
        let mut sched = ReadAheadScheduler::new();
        sched.record_access(5);
        assert_eq!(sched.pattern(), ReadAheadPattern::Random);
    }
}
