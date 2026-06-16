//! Pluggable cache eviction policy engine for IPFRS storage tier management.
//!
//! Supports LRU, LFU, FIFO, and SizePriority eviction strategies.

use std::collections::HashMap;

/// Eviction strategy selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EvictionStrategy {
    /// Least Recently Used — evict the entry accessed furthest in the past.
    Lru,
    /// Least Frequently Used — evict the entry with the fewest accesses;
    /// ties broken by earliest insertion tick.
    Lfu,
    /// First In, First Out — evict the entry inserted earliest.
    Fifo,
    /// Size Priority — evict the largest entry first;
    /// ties broken by smallest `block_id`.
    SizePriority,
}

/// A single entry tracked by the eviction policy.
#[derive(Clone, Debug)]
pub struct CacheEntry {
    /// Identifier of the cached block.
    pub block_id: u64,
    /// Size of the block in bytes.
    pub size_bytes: u64,
    /// Logical clock tick at which the entry was first inserted.
    pub inserted_at_tick: u64,
    /// Logical clock tick of the most recent access.
    pub last_accessed_tick: u64,
    /// Total number of times the entry has been accessed.
    pub access_count: u64,
}

/// A block selected for eviction.
#[derive(Clone, Debug)]
pub struct EvictionCandidate {
    /// Identifier of the evicted block.
    pub block_id: u64,
    /// Size of the evicted block in bytes.
    pub size_bytes: u64,
    /// The strategy that selected this entry for eviction.
    pub reason: EvictionStrategy,
}

/// Aggregate statistics for a `StorageEvictionPolicy` instance.
#[derive(Clone, Debug, Default)]
pub struct PolicyStats {
    /// Current number of entries in the cache.
    pub total_entries: usize,
    /// Current total size of all cached blocks in bytes.
    pub total_size_bytes: u64,
    /// Cumulative number of evictions since creation.
    pub total_evictions: u64,
    /// Cumulative cache hits (successful `access` calls).
    pub hits: u64,
    /// Cumulative cache misses (failed `access` calls).
    pub misses: u64,
}

/// Pluggable storage-tier eviction policy engine.
///
/// Maintains a map of [`CacheEntry`] items, enforces a byte-capacity limit
/// using the configured [`EvictionStrategy`], and tracks hit/miss/eviction
/// statistics via [`PolicyStats`].
pub struct StorageEvictionPolicy {
    /// All currently cached entries, keyed by `block_id`.
    pub entries: HashMap<u64, CacheEntry>,
    /// Active eviction strategy.
    pub strategy: EvictionStrategy,
    /// Maximum total cached bytes before eviction is triggered.
    pub capacity_bytes: u64,
    /// Running statistics.
    pub stats: PolicyStats,
}

impl StorageEvictionPolicy {
    /// Create a new policy with the given strategy and byte capacity.
    pub fn new(strategy: EvictionStrategy, capacity_bytes: u64) -> Self {
        Self {
            entries: HashMap::new(),
            strategy,
            capacity_bytes,
            stats: PolicyStats::default(),
        }
    }

    /// Insert a block into the cache.
    ///
    /// If `block_id` already exists, only `size_bytes` and
    /// `last_accessed_tick` are updated; `inserted_at_tick` is preserved.
    /// [`PolicyStats::total_size_bytes`] is adjusted accordingly.
    pub fn insert(&mut self, block_id: u64, size_bytes: u64, current_tick: u64) {
        if let Some(existing) = self.entries.get_mut(&block_id) {
            // Adjust size delta without changing inserted_at_tick.
            let old_size = existing.size_bytes;
            existing.size_bytes = size_bytes;
            existing.last_accessed_tick = current_tick;
            // Saturating arithmetic to avoid overflow.
            self.stats.total_size_bytes = self
                .stats
                .total_size_bytes
                .saturating_sub(old_size)
                .saturating_add(size_bytes);
        } else {
            let entry = CacheEntry {
                block_id,
                size_bytes,
                inserted_at_tick: current_tick,
                last_accessed_tick: current_tick,
                access_count: 0,
            };
            self.entries.insert(block_id, entry);
            self.stats.total_size_bytes = self.stats.total_size_bytes.saturating_add(size_bytes);
            self.stats.total_entries = self.entries.len();
        }
        self.stats.total_entries = self.entries.len();
    }

    /// Record an access to `block_id`.
    ///
    /// Returns `true` and increments `stats.hits` when the entry exists.
    /// Returns `false` and increments `stats.misses` when it does not.
    pub fn access(&mut self, block_id: u64, current_tick: u64) -> bool {
        if let Some(entry) = self.entries.get_mut(&block_id) {
            entry.last_accessed_tick = current_tick;
            entry.access_count = entry.access_count.saturating_add(1);
            self.stats.hits = self.stats.hits.saturating_add(1);
            true
        } else {
            self.stats.misses = self.stats.misses.saturating_add(1);
            false
        }
    }

    /// Evict entries until [`total_size_bytes`][PolicyStats::total_size_bytes]
    /// is within [`capacity_bytes`][StorageEvictionPolicy::capacity_bytes].
    ///
    /// Returns the list of [`EvictionCandidate`]s that were removed, in the
    /// order they were evicted.
    pub fn evict_to_fit(&mut self) -> Vec<EvictionCandidate> {
        let mut evicted = Vec::new();

        while self.stats.total_size_bytes > self.capacity_bytes {
            if self.entries.is_empty() {
                break;
            }

            let victim_id = self.select_victim();

            if let Some(entry) = self.entries.remove(&victim_id) {
                self.stats.total_size_bytes =
                    self.stats.total_size_bytes.saturating_sub(entry.size_bytes);
                self.stats.total_evictions = self.stats.total_evictions.saturating_add(1);
                self.stats.total_entries = self.entries.len();

                evicted.push(EvictionCandidate {
                    block_id: entry.block_id,
                    size_bytes: entry.size_bytes,
                    reason: self.strategy,
                });
            }
        }

        evicted
    }

    /// Select the `block_id` of the victim to evict based on the current strategy.
    fn select_victim(&self) -> u64 {
        match self.strategy {
            EvictionStrategy::Lru => {
                // Smallest last_accessed_tick.
                self.entries
                    .values()
                    .min_by_key(|e| e.last_accessed_tick)
                    .map(|e| e.block_id)
                    .expect("entries is non-empty")
            }
            EvictionStrategy::Lfu => {
                // Smallest access_count; tie: smallest inserted_at_tick.
                self.entries
                    .values()
                    .min_by_key(|e| (e.access_count, e.inserted_at_tick))
                    .map(|e| e.block_id)
                    .expect("entries is non-empty")
            }
            EvictionStrategy::Fifo => {
                // Smallest inserted_at_tick.
                self.entries
                    .values()
                    .min_by_key(|e| e.inserted_at_tick)
                    .map(|e| e.block_id)
                    .expect("entries is non-empty")
            }
            EvictionStrategy::SizePriority => {
                // Largest size_bytes; tie: smallest block_id.
                self.entries
                    .values()
                    .max_by_key(|e| (e.size_bytes, std::cmp::Reverse(e.block_id)))
                    .map(|e| e.block_id)
                    .expect("entries is non-empty")
            }
        }
    }

    /// Remove a specific entry from the cache.
    ///
    /// Returns `true` if the entry existed and was removed, `false` otherwise.
    pub fn remove(&mut self, block_id: u64) -> bool {
        if let Some(entry) = self.entries.remove(&block_id) {
            self.stats.total_size_bytes =
                self.stats.total_size_bytes.saturating_sub(entry.size_bytes);
            self.stats.total_entries = self.entries.len();
            true
        } else {
            false
        }
    }

    /// Returns `true` when [`total_size_bytes`][PolicyStats::total_size_bytes]
    /// exceeds [`capacity_bytes`][StorageEvictionPolicy::capacity_bytes].
    pub fn is_over_capacity(&self) -> bool {
        self.stats.total_size_bytes > self.capacity_bytes
    }

    /// Return a reference to the current [`PolicyStats`].
    pub fn stats(&self) -> &PolicyStats {
        &self.stats
    }

    /// Replace the active eviction strategy.
    pub fn set_strategy(&mut self, strategy: EvictionStrategy) {
        self.strategy = strategy;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ─────────────────────────────────────────────────────────────

    fn policy(strategy: EvictionStrategy, cap: u64) -> StorageEvictionPolicy {
        StorageEvictionPolicy::new(strategy, cap)
    }

    // ── insert ──────────────────────────────────────────────────────────────

    #[test]
    fn test_insert_adds_entry_and_updates_size() {
        let mut p = policy(EvictionStrategy::Lru, 1000);
        p.insert(1, 100, 1);
        assert_eq!(p.entries.len(), 1);
        assert_eq!(p.stats().total_size_bytes, 100);
        assert_eq!(p.stats().total_entries, 1);
    }

    #[test]
    fn test_insert_multiple_entries_accumulates_size() {
        let mut p = policy(EvictionStrategy::Lru, 1000);
        p.insert(1, 100, 1);
        p.insert(2, 200, 2);
        p.insert(3, 300, 3);
        assert_eq!(p.stats().total_size_bytes, 600);
        assert_eq!(p.stats().total_entries, 3);
    }

    #[test]
    fn test_insert_existing_block_updates_size_not_inserted_tick() {
        let mut p = policy(EvictionStrategy::Lru, 1000);
        p.insert(42, 100, 5);
        let original_inserted_at = p.entries[&42].inserted_at_tick;
        p.insert(42, 250, 10);
        assert_eq!(p.entries[&42].inserted_at_tick, original_inserted_at);
        assert_eq!(p.entries[&42].size_bytes, 250);
        assert_eq!(p.stats().total_size_bytes, 250);
        assert_eq!(p.stats().total_entries, 1);
    }

    #[test]
    fn test_insert_existing_block_size_decreases_correctly() {
        let mut p = policy(EvictionStrategy::Lru, 1000);
        p.insert(1, 500, 1);
        p.insert(2, 200, 2);
        p.insert(1, 100, 3); // shrink block 1
        assert_eq!(p.stats().total_size_bytes, 300);
    }

    // ── access ──────────────────────────────────────────────────────────────

    #[test]
    fn test_access_returns_true_for_existing_entry() {
        let mut p = policy(EvictionStrategy::Lru, 1000);
        p.insert(7, 50, 1);
        assert!(p.access(7, 2));
    }

    #[test]
    fn test_access_returns_false_for_missing_entry() {
        let mut p = policy(EvictionStrategy::Lru, 1000);
        assert!(!p.access(99, 1));
    }

    #[test]
    fn test_access_updates_last_accessed_tick_and_count() {
        let mut p = policy(EvictionStrategy::Lru, 1000);
        p.insert(1, 100, 1);
        p.access(1, 10);
        p.access(1, 20);
        let e = &p.entries[&1];
        assert_eq!(e.last_accessed_tick, 20);
        assert_eq!(e.access_count, 2);
    }

    #[test]
    fn test_access_increments_hits() {
        let mut p = policy(EvictionStrategy::Lru, 1000);
        p.insert(1, 100, 1);
        p.access(1, 2);
        p.access(1, 3);
        assert_eq!(p.stats().hits, 2);
        assert_eq!(p.stats().misses, 0);
    }

    #[test]
    fn test_access_increments_misses() {
        let mut p = policy(EvictionStrategy::Lru, 1000);
        p.access(999, 1);
        p.access(998, 1);
        assert_eq!(p.stats().misses, 2);
        assert_eq!(p.stats().hits, 0);
    }

    // ── evict_to_fit: LRU ───────────────────────────────────────────────────

    #[test]
    fn test_evict_lru_evicts_least_recently_used() {
        let mut p = policy(EvictionStrategy::Lru, 200);
        p.insert(1, 100, 1);
        p.insert(2, 100, 2);
        p.insert(3, 100, 3); // now over capacity (300 > 200)
                             // Access block 1 to make it recently used.
        p.access(1, 10);
        // Block 2 has last_accessed_tick = 2, block 3 = 3, block 1 = 10.
        let evicted = p.evict_to_fit();
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].block_id, 2);
        assert_eq!(evicted[0].reason, EvictionStrategy::Lru);
    }

    #[test]
    fn test_evict_lru_multiple_rounds() {
        let mut p = policy(EvictionStrategy::Lru, 100);
        p.insert(1, 100, 1);
        p.insert(2, 100, 2);
        p.insert(3, 100, 3); // 300 > 100 → evict 2 rounds
        let evicted = p.evict_to_fit();
        assert_eq!(evicted.len(), 2);
        let ids: Vec<u64> = evicted.iter().map(|c| c.block_id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }

    // ── evict_to_fit: LFU ───────────────────────────────────────────────────

    #[test]
    fn test_evict_lfu_evicts_least_frequently_used() {
        let mut p = policy(EvictionStrategy::Lfu, 200);
        p.insert(1, 100, 1);
        p.insert(2, 100, 2);
        p.insert(3, 100, 3); // over capacity
        p.access(1, 5);
        p.access(1, 6);
        p.access(2, 7);
        // block 3 access_count=0, block 2=1, block 1=2 → evict block 3
        let evicted = p.evict_to_fit();
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].block_id, 3);
        assert_eq!(evicted[0].reason, EvictionStrategy::Lfu);
    }

    #[test]
    fn test_evict_lfu_tie_broken_by_earliest_inserted() {
        let mut p = policy(EvictionStrategy::Lfu, 100);
        // Both blocks inserted with access_count=0; block 1 inserted earlier.
        p.insert(1, 100, 1);
        p.insert(2, 100, 2); // now 200 > 100
        let evicted = p.evict_to_fit();
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].block_id, 1); // inserted_at_tick=1 < 2
    }

    // ── evict_to_fit: FIFO ──────────────────────────────────────────────────

    #[test]
    fn test_evict_fifo_evicts_oldest_first() {
        let mut p = policy(EvictionStrategy::Fifo, 200);
        p.insert(1, 100, 1);
        p.insert(2, 100, 2);
        p.insert(3, 100, 3); // over capacity
                             // Access block 1 many times — FIFO ignores access recency.
        p.access(1, 10);
        p.access(1, 11);
        p.access(1, 12);
        let evicted = p.evict_to_fit();
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].block_id, 1); // inserted_at_tick=1
        assert_eq!(evicted[0].reason, EvictionStrategy::Fifo);
    }

    #[test]
    fn test_evict_fifo_multiple_entries() {
        let mut p = policy(EvictionStrategy::Fifo, 50);
        p.insert(10, 100, 5);
        p.insert(20, 100, 3);
        p.insert(30, 100, 7); // 300 > 50 → evict all but one
        let evicted = p.evict_to_fit();
        // Order: block 20 (tick=3), block 10 (tick=5)
        assert_eq!(evicted[0].block_id, 20);
        assert_eq!(evicted[1].block_id, 10);
    }

    // ── evict_to_fit: SizePriority ──────────────────────────────────────────

    #[test]
    fn test_evict_size_priority_evicts_largest_first() {
        let mut p = policy(EvictionStrategy::SizePriority, 200);
        p.insert(1, 50, 1);
        p.insert(2, 300, 2);
        p.insert(3, 100, 3); // total=450 > 200
        let evicted = p.evict_to_fit();
        assert_eq!(evicted[0].block_id, 2); // largest = 300
        assert_eq!(evicted[0].reason, EvictionStrategy::SizePriority);
    }

    #[test]
    fn test_evict_size_priority_tie_smallest_block_id() {
        let mut p = policy(EvictionStrategy::SizePriority, 100);
        // Both blocks have size=200; tie broken by smaller block_id.
        p.insert(5, 200, 1);
        p.insert(3, 200, 2); // 400 > 100
        let evicted = p.evict_to_fit();
        assert_eq!(evicted[0].block_id, 3); // block_id 3 < 5
    }

    // ── evict_to_fit: stops when under capacity ──────────────────────────────

    #[test]
    fn test_evict_to_fit_stops_when_under_capacity() {
        let mut p = policy(EvictionStrategy::Lru, 250);
        p.insert(1, 100, 1);
        p.insert(2, 100, 2);
        p.insert(3, 100, 3); // total=300 > 250 → evict 1
        let evicted = p.evict_to_fit();
        assert_eq!(evicted.len(), 1);
        assert!(p.stats().total_size_bytes <= 250);
    }

    #[test]
    fn test_evict_to_fit_noop_when_under_capacity() {
        let mut p = policy(EvictionStrategy::Lru, 1000);
        p.insert(1, 100, 1);
        let evicted = p.evict_to_fit();
        assert!(evicted.is_empty());
        assert_eq!(p.stats().total_evictions, 0);
    }

    // ── evict_to_fit: stats ─────────────────────────────────────────────────

    #[test]
    fn test_evict_increments_total_evictions() {
        let mut p = policy(EvictionStrategy::Lru, 100);
        p.insert(1, 100, 1);
        p.insert(2, 100, 2);
        p.insert(3, 100, 3); // evicts 2
        p.evict_to_fit();
        assert_eq!(p.stats().total_evictions, 2);
    }

    // ── remove ──────────────────────────────────────────────────────────────

    #[test]
    fn test_remove_existing_entry_returns_true_and_updates_size() {
        let mut p = policy(EvictionStrategy::Lru, 1000);
        p.insert(1, 400, 1);
        p.insert(2, 200, 2);
        assert!(p.remove(1));
        assert_eq!(p.stats().total_size_bytes, 200);
        assert_eq!(p.stats().total_entries, 1);
        assert!(!p.entries.contains_key(&1));
    }

    #[test]
    fn test_remove_nonexistent_entry_returns_false() {
        let mut p = policy(EvictionStrategy::Lru, 1000);
        assert!(!p.remove(999));
        assert_eq!(p.stats().total_size_bytes, 0);
    }

    // ── is_over_capacity ────────────────────────────────────────────────────

    #[test]
    fn test_is_over_capacity_true() {
        let mut p = policy(EvictionStrategy::Lru, 50);
        p.insert(1, 100, 1);
        assert!(p.is_over_capacity());
    }

    #[test]
    fn test_is_over_capacity_false() {
        let mut p = policy(EvictionStrategy::Lru, 500);
        p.insert(1, 100, 1);
        assert!(!p.is_over_capacity());
    }

    #[test]
    fn test_is_over_capacity_exactly_at_capacity() {
        let mut p = policy(EvictionStrategy::Lru, 100);
        p.insert(1, 100, 1);
        assert!(!p.is_over_capacity()); // equal is NOT over
    }

    // ── set_strategy ────────────────────────────────────────────────────────

    #[test]
    fn test_set_strategy_changes_policy() {
        let mut p = policy(EvictionStrategy::Lru, 200);
        assert_eq!(p.strategy, EvictionStrategy::Lru);
        p.set_strategy(EvictionStrategy::Fifo);
        assert_eq!(p.strategy, EvictionStrategy::Fifo);
    }

    #[test]
    fn test_set_strategy_affects_subsequent_eviction() {
        let mut p = policy(EvictionStrategy::Lru, 200);
        p.insert(1, 100, 1);
        p.insert(2, 100, 2);
        p.insert(3, 100, 3); // over capacity
                             // Switch to FIFO before evicting.
        p.set_strategy(EvictionStrategy::Fifo);
        let evicted = p.evict_to_fit();
        assert_eq!(evicted[0].reason, EvictionStrategy::Fifo);
        assert_eq!(evicted[0].block_id, 1); // oldest
    }

    // ── reason field ────────────────────────────────────────────────────────

    #[test]
    fn test_eviction_candidate_reason_matches_strategy_lru() {
        let mut p = policy(EvictionStrategy::Lru, 50);
        p.insert(1, 100, 1);
        let evicted = p.evict_to_fit();
        assert_eq!(evicted[0].reason, EvictionStrategy::Lru);
    }

    #[test]
    fn test_eviction_candidate_reason_matches_strategy_lfu() {
        let mut p = policy(EvictionStrategy::Lfu, 50);
        p.insert(1, 100, 1);
        let evicted = p.evict_to_fit();
        assert_eq!(evicted[0].reason, EvictionStrategy::Lfu);
    }

    #[test]
    fn test_eviction_candidate_reason_matches_strategy_fifo() {
        let mut p = policy(EvictionStrategy::Fifo, 50);
        p.insert(1, 100, 1);
        let evicted = p.evict_to_fit();
        assert_eq!(evicted[0].reason, EvictionStrategy::Fifo);
    }

    #[test]
    fn test_eviction_candidate_reason_matches_strategy_size_priority() {
        let mut p = policy(EvictionStrategy::SizePriority, 50);
        p.insert(1, 100, 1);
        let evicted = p.evict_to_fit();
        assert_eq!(evicted[0].reason, EvictionStrategy::SizePriority);
    }

    // ── stats accessor ───────────────────────────────────────────────────────

    #[test]
    fn test_stats_returns_current_state() {
        let mut p = policy(EvictionStrategy::Lru, 1000);
        p.insert(1, 300, 1);
        p.insert(2, 200, 2);
        p.access(1, 5);
        p.access(99, 6); // miss
        let s = p.stats();
        assert_eq!(s.total_entries, 2);
        assert_eq!(s.total_size_bytes, 500);
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
        assert_eq!(s.total_evictions, 0);
    }

    // ── eviction_candidate size field ────────────────────────────────────────

    #[test]
    fn test_eviction_candidate_carries_correct_size() {
        let mut p = policy(EvictionStrategy::SizePriority, 50);
        p.insert(7, 777, 1);
        let evicted = p.evict_to_fit();
        assert_eq!(evicted[0].size_bytes, 777);
        assert_eq!(evicted[0].block_id, 7);
    }
}
