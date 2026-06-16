//! Managed pool of pre-allocated embedding buffers for zero-copy semantic search operations.
//!
//! Reduces allocation overhead in high-throughput workloads by reusing fixed-dimension
//! embedding buffers across acquire/release cycles.

/// A single pre-allocated embedding buffer managed by the pool.
#[derive(Debug, Clone)]
pub struct EmbeddingBuffer {
    /// Unique identifier for this buffer within the pool.
    pub buffer_id: u64,
    /// Raw embedding data.
    pub data: Vec<f32>,
    /// Fixed dimension of this buffer.
    pub dim: usize,
    /// Whether this buffer is currently checked out by a caller.
    pub in_use: bool,
    /// Monotonically increasing counter, incremented each time the buffer returns to the pool.
    pub generation: u64,
}

impl EmbeddingBuffer {
    /// Returns `true` if this buffer's dimension matches `expected`.
    pub fn is_valid_dim(&self, expected: usize) -> bool {
        self.dim == expected
    }
}

/// Configuration for [`SemanticEmbeddingPool`].
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Fixed embedding dimension shared by all buffers in the pool.
    pub embedding_dim: usize,
    /// Number of buffers to pre-allocate when the pool is constructed.
    pub initial_capacity: usize,
    /// Hard upper limit on the total number of buffers the pool may hold.
    pub max_capacity: usize,
}

impl PoolConfig {
    /// Build a [`PoolConfig`] with sensible defaults for the given embedding dimension.
    pub fn default_for(dim: usize) -> Self {
        Self {
            embedding_dim: dim,
            initial_capacity: 16,
            max_capacity: 256,
        }
    }
}

/// Snapshot statistics for a [`SemanticEmbeddingPool`].
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total number of buffers currently managed by the pool (in-use + available).
    pub total_buffers: usize,
    /// Number of buffers currently checked out.
    pub in_use: usize,
    /// Number of buffers currently available for acquisition.
    pub available: usize,
    /// Cumulative number of times `acquire` was called.
    pub total_allocations: u64,
    /// Cumulative number of times `release` was called successfully.
    pub total_returns: u64,
    /// Acquisitions satisfied from an existing free buffer (cache hit).
    pub cache_hits: u64,
    /// Acquisitions that required allocating a new buffer (cache miss).
    pub cache_misses: u64,
}

/// Managed pool of pre-allocated, fixed-dimension embedding buffers.
///
/// Callers `acquire` a buffer id, `write` data into it, `read` from it, and
/// `release` it back to the pool. Releasing zeroes the buffer data and increments
/// the buffer's generation counter so stale ids are detectable.
pub struct SemanticEmbeddingPool {
    /// All buffers managed by this pool (both in-use and free).
    pub buffers: Vec<EmbeddingBuffer>,
    next_id: u64,
    /// Pool configuration.
    pub config: PoolConfig,
    /// Accumulated runtime statistics.
    pub stats: PoolStats,
}

impl SemanticEmbeddingPool {
    /// Create a new pool, pre-allocating `config.initial_capacity` zeroed buffers.
    pub fn new(config: PoolConfig) -> Self {
        let dim = config.embedding_dim;
        let initial = config.initial_capacity;
        let mut buffers = Vec::with_capacity(initial);
        for id in 0..initial as u64 {
            buffers.push(EmbeddingBuffer {
                buffer_id: id,
                data: vec![0.0_f32; dim],
                dim,
                in_use: false,
                generation: 0,
            });
        }
        let total = buffers.len();
        Self {
            buffers,
            next_id: initial as u64,
            config,
            stats: PoolStats {
                total_buffers: total,
                available: total,
                ..Default::default()
            },
        }
    }

    /// Acquire a free buffer, allocating a new one if necessary.
    ///
    /// Returns `Some(buffer_id)` on success, or `None` when the pool is at
    /// `max_capacity` and every buffer is already in use.
    pub fn acquire(&mut self) -> Option<u64> {
        self.stats.total_allocations += 1;

        // Try to find an existing free buffer first (cache hit).
        if let Some(buf) = self.buffers.iter_mut().find(|b| !b.in_use) {
            buf.in_use = true;
            self.stats.cache_hits += 1;
            self.stats.in_use += 1;
            self.stats.available = self.stats.available.saturating_sub(1);
            return Some(buf.buffer_id);
        }

        // No free buffer – allocate a new one if below max_capacity.
        if self.buffers.len() < self.config.max_capacity {
            let id = self.next_id;
            self.next_id += 1;
            let dim = self.config.embedding_dim;
            self.buffers.push(EmbeddingBuffer {
                buffer_id: id,
                data: vec![0.0_f32; dim],
                dim,
                in_use: true,
                generation: 0,
            });
            self.stats.cache_misses += 1;
            self.stats.total_buffers += 1;
            self.stats.in_use += 1;
            // available stays the same (new buffer goes straight to in_use)
            return Some(id);
        }

        // Pool exhausted.
        None
    }

    /// Write `data` into the buffer identified by `buffer_id`.
    ///
    /// Returns `false` when:
    /// - No buffer with `buffer_id` exists.
    /// - The buffer is not currently in use.
    /// - `data.len()` does not match the pool's embedding dimension.
    pub fn write(&mut self, buffer_id: u64, data: &[f32]) -> bool {
        let dim = self.config.embedding_dim;
        if data.len() != dim {
            return false;
        }
        match self.buffers.iter_mut().find(|b| b.buffer_id == buffer_id) {
            Some(buf) if buf.in_use => {
                buf.data.copy_from_slice(data);
                true
            }
            _ => false,
        }
    }

    /// Read the data slice from an in-use buffer.
    ///
    /// Returns `None` when the buffer is not found or is not currently in use.
    pub fn read(&self, buffer_id: u64) -> Option<&[f32]> {
        self.buffers
            .iter()
            .find(|b| b.buffer_id == buffer_id && b.in_use)
            .map(|b| b.data.as_slice())
    }

    /// Release a buffer back to the pool.
    ///
    /// Zeroes the buffer data and increments its generation counter.
    /// Returns `false` if no buffer with `buffer_id` is found.
    pub fn release(&mut self, buffer_id: u64) -> bool {
        match self.buffers.iter_mut().find(|b| b.buffer_id == buffer_id) {
            Some(buf) => {
                let was_in_use = buf.in_use;
                buf.in_use = false;
                buf.generation += 1;
                for v in buf.data.iter_mut() {
                    *v = 0.0;
                }
                self.stats.total_returns += 1;
                if was_in_use {
                    self.stats.in_use = self.stats.in_use.saturating_sub(1);
                    self.stats.available += 1;
                }
                true
            }
            None => false,
        }
    }

    /// Release every buffer in the pool back to the free state.
    pub fn clear_all(&mut self) {
        for buf in self.buffers.iter_mut() {
            if buf.in_use {
                buf.in_use = false;
                self.stats.in_use = self.stats.in_use.saturating_sub(1);
                self.stats.available += 1;
            }
            buf.generation += 1;
            for v in buf.data.iter_mut() {
                *v = 0.0;
            }
            self.stats.total_returns += 1;
        }
    }

    /// Return a snapshot of the current pool statistics.
    pub fn stats(&self) -> PoolStats {
        self.stats.clone()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pool(dim: usize, initial: usize, max: usize) -> SemanticEmbeddingPool {
        SemanticEmbeddingPool::new(PoolConfig {
            embedding_dim: dim,
            initial_capacity: initial,
            max_capacity: max,
        })
    }

    // 1. new() pre-allocates initial_capacity buffers
    #[test]
    fn test_new_preallocates_buffers() {
        let pool = make_pool(4, 8, 64);
        assert_eq!(pool.buffers.len(), 8);
        for buf in &pool.buffers {
            assert!(!buf.in_use);
            assert_eq!(buf.data.len(), 4);
            assert_eq!(buf.generation, 0);
        }
    }

    // 2. initial stats are consistent
    #[test]
    fn test_new_stats_consistent() {
        let pool = make_pool(4, 8, 64);
        let s = pool.stats();
        assert_eq!(s.total_buffers, 8);
        assert_eq!(s.in_use, 0);
        assert_eq!(s.available, 8);
        assert_eq!(s.total_allocations, 0);
        assert_eq!(s.total_returns, 0);
        assert_eq!(s.cache_hits, 0);
        assert_eq!(s.cache_misses, 0);
    }

    // 3. acquire returns buffer_id from pool (cache_hit)
    #[test]
    fn test_acquire_cache_hit() {
        let mut pool = make_pool(4, 4, 16);
        let id = pool.acquire().expect("should return id");
        let s = pool.stats();
        assert_eq!(s.cache_hits, 1);
        assert_eq!(s.cache_misses, 0);
        assert_eq!(s.total_allocations, 1);
        // The returned id should be marked in_use
        let buf = pool
            .buffers
            .iter()
            .find(|b| b.buffer_id == id)
            .expect("test: buffer with acquired id should exist");
        assert!(buf.in_use);
    }

    // 4. acquire allocates new when pool fully in-use but below max (cache_miss)
    #[test]
    fn test_acquire_cache_miss() {
        let mut pool = make_pool(4, 1, 16);
        // Exhaust the single pre-allocated buffer
        let _id1 = pool.acquire().expect("first acquire");
        assert_eq!(pool.stats().cache_hits, 1);
        // Second acquire must allocate
        let _id2 = pool.acquire().expect("second acquire");
        let s = pool.stats();
        assert_eq!(s.cache_hits, 1);
        assert_eq!(s.cache_misses, 1);
        assert_eq!(s.total_buffers, 2);
    }

    // 5. acquire returns None at max_capacity with no free buffers
    #[test]
    fn test_acquire_returns_none_at_max() {
        let mut pool = make_pool(4, 2, 2);
        let _id1 = pool.acquire().expect("first");
        let _id2 = pool.acquire().expect("second");
        let result = pool.acquire();
        assert!(result.is_none());
        assert_eq!(pool.stats().total_allocations, 3);
    }

    // 6. write copies data correctly
    #[test]
    fn test_write_copies_data() {
        let mut pool = make_pool(4, 2, 8);
        let id = pool.acquire().expect("acquire");
        let data = [1.0_f32, 2.0, 3.0, 4.0];
        assert!(pool.write(id, &data));
        let read_back = pool.read(id).expect("read");
        assert_eq!(read_back, &data);
    }

    // 7. write returns false for wrong dimension
    #[test]
    fn test_write_wrong_dim() {
        let mut pool = make_pool(4, 2, 8);
        let id = pool.acquire().expect("acquire");
        let bad = [1.0_f32, 2.0]; // only 2 elements, dim is 4
        assert!(!pool.write(id, &bad));
    }

    // 8. write returns false for a non-in-use buffer
    #[test]
    fn test_write_not_in_use() {
        let mut pool = make_pool(4, 2, 8);
        // Buffer 0 is free (not acquired)
        let free_id = pool.buffers[0].buffer_id;
        assert!(!pool.write(free_id, &[1.0, 2.0, 3.0, 4.0]));
    }

    // 9. write returns false for non-existent buffer_id
    #[test]
    fn test_write_nonexistent_id() {
        let mut pool = make_pool(4, 2, 8);
        assert!(!pool.write(9999, &[1.0, 2.0, 3.0, 4.0]));
    }

    // 10. read returns data for in-use buffer
    #[test]
    fn test_read_in_use() {
        let mut pool = make_pool(3, 2, 8);
        let id = pool.acquire().expect("acquire");
        pool.write(id, &[0.5, 0.6, 0.7]);
        let data = pool.read(id).expect("read");
        assert_eq!(data, &[0.5_f32, 0.6, 0.7]);
    }

    // 11. read returns None when buffer is not in use
    #[test]
    fn test_read_not_in_use() {
        let pool = make_pool(4, 2, 8);
        let free_id = pool.buffers[0].buffer_id;
        assert!(pool.read(free_id).is_none());
    }

    // 12. read returns None for non-existent id
    #[test]
    fn test_read_nonexistent_id() {
        let pool = make_pool(4, 2, 8);
        assert!(pool.read(9999).is_none());
    }

    // 13. release marks buffer free and zeroes data
    #[test]
    fn test_release_frees_and_zeroes() {
        let mut pool = make_pool(4, 2, 8);
        let id = pool.acquire().expect("acquire");
        pool.write(id, &[1.0, 2.0, 3.0, 4.0]);
        assert!(pool.release(id));
        let buf = pool
            .buffers
            .iter()
            .find(|b| b.buffer_id == id)
            .expect("test: released buffer should exist");
        assert!(!buf.in_use);
        assert!(buf.data.iter().all(|&v| v == 0.0));
    }

    // 14. release increments generation
    #[test]
    fn test_release_increments_generation() {
        let mut pool = make_pool(4, 2, 8);
        let id = pool.acquire().expect("acquire");
        let gen_before = pool
            .buffers
            .iter()
            .find(|b| b.buffer_id == id)
            .expect("test: buffer before release should exist")
            .generation;
        pool.release(id);
        let gen_after = pool
            .buffers
            .iter()
            .find(|b| b.buffer_id == id)
            .expect("test: buffer after release should exist")
            .generation;
        assert_eq!(gen_after, gen_before + 1);
    }

    // 15. release returns false for non-existent id
    #[test]
    fn test_release_nonexistent_id() {
        let mut pool = make_pool(4, 2, 8);
        assert!(!pool.release(9999));
    }

    // 16. release increments total_returns
    #[test]
    fn test_release_increments_total_returns() {
        let mut pool = make_pool(4, 2, 8);
        let id = pool.acquire().expect("acquire");
        pool.release(id);
        assert_eq!(pool.stats().total_returns, 1);
    }

    // 17. released buffer becomes available again for acquire
    #[test]
    fn test_released_buffer_reused() {
        let mut pool = make_pool(4, 1, 1);
        let id1 = pool.acquire().expect("first acquire");
        pool.release(id1);
        // At max=1 and now free, next acquire must succeed as a cache hit
        let id2 = pool.acquire().expect("second acquire");
        assert_eq!(id1, id2);
        assert_eq!(pool.stats().cache_hits, 2);
    }

    // 18. clear_all releases all buffers
    #[test]
    fn test_clear_all_releases_all() {
        let mut pool = make_pool(4, 4, 16);
        let _id0 = pool.acquire().expect("acquire 0");
        let _id1 = pool.acquire().expect("acquire 1");
        let _id2 = pool.acquire().expect("acquire 2");
        pool.clear_all();
        assert!(pool.buffers.iter().all(|b| !b.in_use));
        assert!(pool
            .buffers
            .iter()
            .all(|b| b.data.iter().all(|&v| v == 0.0)));
    }

    // 19. clear_all increments all generations
    #[test]
    fn test_clear_all_increments_generations() {
        let mut pool = make_pool(4, 3, 16);
        let _id = pool.acquire().expect("acquire");
        pool.clear_all();
        for buf in &pool.buffers {
            assert!(
                buf.generation >= 1,
                "generation should be at least 1 after clear_all"
            );
        }
    }

    // 20. stats: in_use + available == total_buffers
    #[test]
    fn test_stats_invariant() {
        let mut pool = make_pool(4, 6, 32);
        let _a = pool.acquire();
        let _b = pool.acquire();
        let _c = pool.acquire();
        let s = pool.stats();
        assert_eq!(s.in_use + s.available, s.total_buffers);
    }

    // 21. stats: cache_hits + cache_misses == total_allocations (when None not returned)
    #[test]
    fn test_stats_hits_plus_misses_eq_allocations() {
        let mut pool = make_pool(4, 2, 8);
        for _ in 0..5 {
            pool.acquire();
        }
        let s = pool.stats();
        assert_eq!(s.cache_hits + s.cache_misses, s.total_allocations);
    }

    // 22. is_valid_dim works correctly
    #[test]
    fn test_is_valid_dim() {
        let buf = EmbeddingBuffer {
            buffer_id: 0,
            data: vec![0.0; 128],
            dim: 128,
            in_use: false,
            generation: 0,
        };
        assert!(buf.is_valid_dim(128));
        assert!(!buf.is_valid_dim(64));
    }

    // 23. PoolConfig::default_for sets correct defaults
    #[test]
    fn test_pool_config_default_for() {
        let cfg = PoolConfig::default_for(768);
        assert_eq!(cfg.embedding_dim, 768);
        assert_eq!(cfg.initial_capacity, 16);
        assert_eq!(cfg.max_capacity, 256);
    }

    // 24. Pool handles zero initial_capacity gracefully
    #[test]
    fn test_zero_initial_capacity() {
        let mut pool = make_pool(4, 0, 4);
        assert_eq!(pool.buffers.len(), 0);
        let id = pool.acquire().expect("should allocate");
        assert_eq!(pool.stats().cache_misses, 1);
        assert!(pool.write(id, &[1.0, 2.0, 3.0, 4.0]));
        let data = pool.read(id).expect("read");
        assert_eq!(data, &[1.0_f32, 2.0, 3.0, 4.0]);
    }

    // 25. Multiple writes to same buffer accumulate correctly
    #[test]
    fn test_multiple_writes() {
        let mut pool = make_pool(3, 2, 8);
        let id = pool.acquire().expect("acquire");
        pool.write(id, &[1.0, 1.0, 1.0]);
        pool.write(id, &[9.0, 8.0, 7.0]);
        let data = pool.read(id).expect("read");
        assert_eq!(data, &[9.0_f32, 8.0, 7.0]);
    }

    // 26. Release then re-acquire clears previous data
    #[test]
    fn test_release_clears_data_before_reuse() {
        let mut pool = make_pool(4, 1, 4);
        let id = pool.acquire().expect("acquire");
        pool.write(id, &[5.0, 6.0, 7.0, 8.0]);
        pool.release(id);
        let id2 = pool.acquire().expect("reacquire");
        assert_eq!(id, id2);
        // After release the buffer was zeroed; reading without a new write should see zeros
        let data = pool.read(id2).expect("read");
        assert!(data.iter().all(|&v| v == 0.0));
    }

    // 27. stats available tracks correctly across acquire/release cycle
    #[test]
    fn test_stats_available_tracks() {
        let mut pool = make_pool(4, 4, 16);
        assert_eq!(pool.stats().available, 4);
        let id1 = pool.acquire().expect("a1");
        assert_eq!(pool.stats().available, 3);
        let id2 = pool.acquire().expect("a2");
        assert_eq!(pool.stats().available, 2);
        pool.release(id1);
        assert_eq!(pool.stats().available, 3);
        pool.release(id2);
        assert_eq!(pool.stats().available, 4);
    }
}
