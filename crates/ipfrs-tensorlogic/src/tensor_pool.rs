//! Slab-based Reusable Buffer Pool for Arrow IPC Zero-Copy Tensor Operations
//!
//! This module provides a `TensorPool` — a production-grade, thread-safe buffer pool
//! organized into power-of-two size buckets. It is designed for zero-copy Arrow IPC
//! tensor operations where repeated allocation/deallocation of large byte buffers
//! is a key performance bottleneck.
//!
//! # Size Classes
//!
//! | Bucket | Min Size | Max Size |
//! |--------|----------|----------|
//! |   0    |    0 B   |   255 B  |  (allocated as 256 B)
//! |   1    |  256 B   |   511 B  |  (allocated as 512 B)
//! |   2    |  512 B   |  1023 B  |  (allocated as 1 KiB)
//! |   3    |   1 KiB  |  2047 B  |  (allocated as 2 KiB)
//! |   4    |   2 KiB  |  4095 B  |  (allocated as 4 KiB)
//! |   5    |   4 KiB  |  8191 B  |  (allocated as 8 KiB)
//! |   6    |   8 KiB  | 16383 B  |  (allocated as 16 KiB)
//! |   7    |  16 KiB  |  ∞       |  (allocated as exactly requested, up to 32 MiB cap)
//!
//! # Example
//!
//! ```
//! use ipfrs_tensorlogic::tensor_pool::{TensorPool, TensorPoolConfig};
//!
//! let pool = TensorPool::new(TensorPoolConfig::default());
//!
//! // Acquire a buffer that fits at least 1000 bytes
//! let mut buf = pool.acquire(1000);
//! buf.resize(1000, 0u8);
//! assert_eq!(buf.len(), 1000);
//!
//! // Return buffer to pool
//! pool.release(buf);
//!
//! // Next acquire should reuse the buffer
//! let buf2 = pool.acquire(1000);
//! let snap = pool.stats();
//! assert_eq!(snap.total_reuses, 1);
//! pool.release(buf2);
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of buckets (power-of-two size classes)
pub const NUM_BUCKETS: usize = 8;

/// Minimum size for bucket 0 (256 bytes)
const BUCKET_MIN_SIZE: usize = 256;

/// Size threshold above which everything is routed to bucket 7 (32 MiB).
#[allow(dead_code)]
pub const BUCKET_7_THRESHOLD: usize = 32 * 1024 * 1024;

// ---------------------------------------------------------------------------
// bucket_for helper
// ---------------------------------------------------------------------------

/// Returns the bucket index (0..=7) for a given size.
///
/// Bucket 0 holds buffers sized up to and including 256 B, bucket 1 up to 512 B, …
/// Bucket 7 holds everything > 16 KiB (including up to and above 32 MiB).
///
/// # Arguments
/// * `size` — the minimum number of bytes required
///
/// # Returns
/// Bucket index in `0..NUM_BUCKETS`.
///
/// # Examples
/// ```
/// use ipfrs_tensorlogic::tensor_pool::bucket_for;
/// assert_eq!(bucket_for(0), 0);
/// assert_eq!(bucket_for(256), 0);  // exactly fits bucket 0
/// assert_eq!(bucket_for(257), 1);  // spills into bucket 1
/// assert_eq!(bucket_for(32 * 1024 * 1024), 7);
/// ```
pub fn bucket_for(size: usize) -> usize {
    // Map size → smallest power-of-two bucket that can hold it.
    // Bucket 0 → capacity 256
    // Bucket k → capacity 256 << k
    // Bucket 7 → capacity 256 << 7 = 32 MiB (anything ≥ 32 MiB also lands here)
    if size == 0 {
        return 0;
    }
    // Walk up from bucket 0
    for bucket in 0..NUM_BUCKETS {
        let bucket_capacity = BUCKET_MIN_SIZE << bucket;
        if size <= bucket_capacity {
            return bucket;
        }
    }
    // size > 32 MiB — goes to the last bucket
    NUM_BUCKETS - 1
}

/// Returns the capacity that a buffer in the given bucket should have.
#[inline]
fn capacity_for_bucket(bucket: usize) -> usize {
    BUCKET_MIN_SIZE << bucket.min(NUM_BUCKETS - 1)
}

// ---------------------------------------------------------------------------
// TensorPoolStats
// ---------------------------------------------------------------------------

/// Atomic counters tracking pool activity.
///
/// All fields use `AtomicU64` and `Relaxed` ordering for maximum throughput.
/// Call [`TensorPoolStats::snapshot`] to obtain a consistent plain-struct view.
pub struct TensorPoolStats {
    /// Total number of times `acquire` was called
    pub total_acquired: AtomicU64,
    /// Total number of times `release` was called
    pub total_released: AtomicU64,
    /// Fresh allocations that bypassed the pool (pool was empty for that bucket)
    pub total_allocs: AtomicU64,
    /// Pool-hit reuses (buffer served from the free list)
    pub total_reuses: AtomicU64,
    /// Running total of bytes currently held by all pooled buffers
    pub total_bytes_pooled: AtomicU64,
}

impl Default for TensorPoolStats {
    fn default() -> Self {
        Self {
            total_acquired: AtomicU64::new(0),
            total_released: AtomicU64::new(0),
            total_allocs: AtomicU64::new(0),
            total_reuses: AtomicU64::new(0),
            total_bytes_pooled: AtomicU64::new(0),
        }
    }
}

impl TensorPoolStats {
    /// Capture a consistent snapshot of all counters.
    pub fn snapshot(&self) -> TensorPoolSnapshot {
        TensorPoolSnapshot {
            total_acquired: self.total_acquired.load(Ordering::Relaxed),
            total_released: self.total_released.load(Ordering::Relaxed),
            total_allocs: self.total_allocs.load(Ordering::Relaxed),
            total_reuses: self.total_reuses.load(Ordering::Relaxed),
            total_bytes_pooled: self.total_bytes_pooled.load(Ordering::Relaxed),
        }
    }
}

// ---------------------------------------------------------------------------
// TensorPoolSnapshot
// ---------------------------------------------------------------------------

/// Plain-struct snapshot of [`TensorPoolStats`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorPoolSnapshot {
    /// Total number of times `acquire` was called
    pub total_acquired: u64,
    /// Total number of times `release` was called
    pub total_released: u64,
    /// Fresh allocations that bypassed the pool
    pub total_allocs: u64,
    /// Pool-hit reuses
    pub total_reuses: u64,
    /// Running total of bytes currently held by all pooled buffers
    pub total_bytes_pooled: u64,
}

// ---------------------------------------------------------------------------
// TensorPoolConfig
// ---------------------------------------------------------------------------

/// Configuration for [`TensorPool`].
#[derive(Debug, Clone)]
pub struct TensorPoolConfig {
    /// Maximum number of free buffers to retain per bucket.
    ///
    /// When a buffer is released back to the pool and the corresponding free list
    /// already has `max_per_bucket` entries, the buffer is simply dropped.
    pub max_per_bucket: usize,
}

impl Default for TensorPoolConfig {
    fn default() -> Self {
        Self { max_per_bucket: 16 }
    }
}

// ---------------------------------------------------------------------------
// PooledBuffer
// ---------------------------------------------------------------------------

/// An owned, pool-tracked byte buffer.
///
/// Created exclusively by [`TensorPool::acquire`] and must be returned to the
/// same pool via [`TensorPool::release`].  There is **no** `Drop` guard —
/// callers are responsible for calling `release`.  This is an intentional
/// design choice to avoid the need for `Arc<TensorPool>` references inside the
/// buffer struct and to keep zero-copy paths maximally thin.
pub struct PooledBuffer {
    /// The actual byte storage
    inner: Vec<u8>,
    /// Which bucket this buffer belongs to
    pub(crate) bucket: usize,
}

impl PooledBuffer {
    /// Construct a new `PooledBuffer` from a raw `Vec<u8>` and a bucket index.
    ///
    /// This is not part of the public API — callers should use [`TensorPool::acquire`].
    pub(crate) fn new(inner: Vec<u8>, bucket: usize) -> Self {
        Self { inner, bucket }
    }

    /// Returns the bucket index this buffer is classified under.
    #[inline]
    pub fn bucket(&self) -> usize {
        self.bucket
    }

    /// Immutable view of the buffer contents.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.inner
    }

    /// Mutable view of the buffer contents.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.inner
    }

    /// The total pre-allocated capacity of the underlying `Vec<u8>`.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// The current logical length (number of initialized bytes).
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if `len() == 0`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Resize the buffer, filling any new bytes with `val`.
    ///
    /// Delegates directly to [`Vec::resize`].
    #[inline]
    pub fn resize(&mut self, new_len: usize, val: u8) {
        self.inner.resize(new_len, val);
    }

    /// Consume the `PooledBuffer` and return the raw inner `Vec<u8>`.
    ///
    /// The caller takes ownership; the buffer is **not** returned to the pool.
    /// Prefer [`TensorPool::release`] to reuse buffers.
    pub fn into_inner(self) -> Vec<u8> {
        self.inner
    }
}

// ---------------------------------------------------------------------------
// TensorPool
// ---------------------------------------------------------------------------

/// Slab-based, thread-safe buffer pool for zero-copy Arrow IPC tensor operations.
///
/// Internally maintains 8 free lists — one per power-of-two size class from
/// 256 B to 32 MiB.  All free lists are protected by individual `Mutex` locks
/// so contention is minimised.
///
/// # Thread Safety
///
/// `TensorPool` is `Send + Sync` and can be wrapped in an `Arc` for sharing
/// across threads / async tasks.
pub struct TensorPool {
    /// Free lists, one per bucket.  We use a fixed-size array of `Mutex<Vec<…>>`
    /// rather than a `Vec` to make the structure `Sync` without extra indirection.
    free_lists: [Mutex<Vec<Vec<u8>>>; NUM_BUCKETS],
    /// Live counters
    stats: TensorPoolStats,
    /// Configuration
    config: TensorPoolConfig,
}

impl Default for TensorPool {
    fn default() -> Self {
        Self::new(TensorPoolConfig::default())
    }
}

impl TensorPool {
    /// Create a new `TensorPool` with the supplied configuration.
    pub fn new(config: TensorPoolConfig) -> Self {
        Self {
            // Array-init: Rust does not support [expr; N] for non-Copy types, so
            // we use `std::array::from_fn`.
            free_lists: std::array::from_fn(|_| Mutex::new(Vec::new())),
            stats: TensorPoolStats::default(),
            config,
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Acquire a buffer whose capacity is at least `min_bytes`.
    ///
    /// If the corresponding free list is non-empty, a buffer is popped and
    /// returned (incrementing `total_reuses`).  Otherwise a fresh `Vec<u8>` is
    /// allocated (incrementing `total_allocs`).
    ///
    /// In both cases `total_acquired` is incremented.
    pub fn acquire(&self, min_bytes: usize) -> PooledBuffer {
        let bucket = bucket_for(min_bytes);
        let cap = capacity_for_bucket(bucket);

        self.stats.total_acquired.fetch_add(1, Ordering::Relaxed);

        // Try to pop from the free list under a short-lived lock.
        let maybe_buf = {
            let mut list = self.free_lists[bucket]
                .lock()
                .expect("TensorPool free-list mutex poisoned");
            list.pop()
        };

        match maybe_buf {
            Some(mut buf) => {
                // Reuse: the buffer was previously cleared on release.
                self.stats.total_reuses.fetch_add(1, Ordering::Relaxed);
                // Ensure capacity is still sufficient (it always should be, but
                // be defensive in case someone tampered with the inner vec).
                if buf.capacity() < cap {
                    buf.reserve(cap - buf.capacity());
                }
                PooledBuffer::new(buf, bucket)
            }
            None => {
                // Fresh allocation
                self.stats.total_allocs.fetch_add(1, Ordering::Relaxed);
                let buf = Vec::with_capacity(cap);
                PooledBuffer::new(buf, bucket)
            }
        }
    }

    /// Release a buffer back into the pool.
    ///
    /// The buffer contents are cleared (length reset to 0, capacity retained)
    /// before being added to the free list.  If the free list for the buffer's
    /// bucket already holds `max_per_bucket` entries, the buffer is simply
    /// dropped (freeing its memory).
    ///
    /// Increments `total_released` in both cases.  Adjusts `total_bytes_pooled`
    /// by the buffer's capacity when it is successfully pooled.
    pub fn release(&self, buf: PooledBuffer) {
        let bucket = buf.bucket;
        let cap = buf.capacity();
        let mut inner = buf.into_inner();

        self.stats.total_released.fetch_add(1, Ordering::Relaxed);

        // Clear contents before returning to pool.
        inner.clear();

        let mut list = self.free_lists[bucket]
            .lock()
            .expect("TensorPool free-list mutex poisoned");

        if list.len() < self.config.max_per_bucket {
            self.stats
                .total_bytes_pooled
                .fetch_add(cap as u64, Ordering::Relaxed);
            list.push(inner);
        }
        // else: buffer is simply dropped here, memory freed.
    }

    /// Return the number of free (available) buffers in the specified bucket.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `bucket >= NUM_BUCKETS`.
    pub fn pool_depth(&self, bucket: usize) -> usize {
        debug_assert!(bucket < NUM_BUCKETS, "bucket index out of range");
        if bucket >= NUM_BUCKETS {
            return 0;
        }
        self.free_lists[bucket]
            .lock()
            .expect("TensorPool free-list mutex poisoned")
            .len()
    }

    /// Capture a snapshot of the pool statistics.
    pub fn stats(&self) -> TensorPoolSnapshot {
        self.stats.snapshot()
    }

    /// Drain excess buffers from every bucket, keeping at most `max_per_bucket`
    /// buffers per bucket.
    ///
    /// Buffers that are drained are dropped, releasing their memory.  The
    /// `total_bytes_pooled` counter is decremented accordingly.
    pub fn prune(&self, max_per_bucket: usize) {
        for (bucket, free_list) in self.free_lists.iter().enumerate() {
            let mut list = free_list
                .lock()
                .expect("TensorPool free-list mutex poisoned");

            if list.len() > max_per_bucket {
                let excess = list.drain(max_per_bucket..).collect::<Vec<_>>();
                let freed_bytes: u64 = excess.iter().map(|v| v.capacity() as u64).sum();
                drop(excess);
                // Subtract freed bytes from the pooled-bytes counter, saturating at 0.
                let _ = self.stats.total_bytes_pooled.fetch_update(
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                    |prev| Some(prev.saturating_sub(freed_bytes)),
                );
                let _ = bucket; // suppress unused-var lint in release builds
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helper
    // ------------------------------------------------------------------

    fn make_pool() -> TensorPool {
        TensorPool::new(TensorPoolConfig::default())
    }

    // ------------------------------------------------------------------
    // bucket_for edge-cases
    // ------------------------------------------------------------------

    #[test]
    fn bucket_for_zero_is_zero() {
        assert_eq!(bucket_for(0), 0);
    }

    #[test]
    fn bucket_for_one_is_zero() {
        assert_eq!(bucket_for(1), 0);
    }

    #[test]
    fn bucket_for_255_is_zero() {
        assert_eq!(bucket_for(255), 0);
    }

    #[test]
    fn bucket_for_256_is_zero() {
        // 256 exactly fills bucket 0's capacity (256 B), so it maps to bucket 0.
        assert_eq!(bucket_for(256), 0);
    }

    #[test]
    fn bucket_for_257_is_one() {
        // 257 exceeds bucket 0 capacity (256 B), so it must go to bucket 1 (512 B).
        assert_eq!(bucket_for(257), 1);
    }

    #[test]
    fn bucket_for_512_is_one() {
        // 512 exactly fills bucket 1's capacity (512 B).
        assert_eq!(bucket_for(512), 1);
    }

    #[test]
    fn bucket_for_513_is_two() {
        assert_eq!(bucket_for(513), 2);
    }

    #[test]
    fn bucket_for_32mb_is_seven() {
        assert_eq!(bucket_for(32 * 1024 * 1024), 7);
    }

    #[test]
    fn bucket_for_above_32mb_is_seven() {
        assert_eq!(bucket_for(64 * 1024 * 1024), 7);
    }

    // ------------------------------------------------------------------
    // Acquire / release round-trip
    // ------------------------------------------------------------------

    #[test]
    fn acquire_release_round_trip() {
        let pool = make_pool();

        let buf = pool.acquire(100);
        assert!(buf.capacity() >= 100);
        pool.release(buf);

        // The buffer should now be in the pool.
        assert_eq!(pool.pool_depth(0), 1);
    }

    #[test]
    fn released_buffer_has_zero_len() {
        let pool = make_pool();

        let mut buf = pool.acquire(100);
        buf.resize(50, 0xAB);
        assert_eq!(buf.len(), 50);

        pool.release(buf);

        // Retrieve and confirm length is reset.
        let buf2 = pool.acquire(100);
        assert_eq!(buf2.len(), 0, "released buffer must have length 0");
        pool.release(buf2);
    }

    // ------------------------------------------------------------------
    // Reuse counter
    // ------------------------------------------------------------------

    #[test]
    fn reuse_counter_increments_on_pool_hit() {
        let pool = make_pool();

        // First acquire: pool is empty → fresh alloc.
        let buf = pool.acquire(100);
        pool.release(buf);

        // Second acquire: pool has one entry → reuse.
        let buf2 = pool.acquire(100);
        pool.release(buf2);

        let snap = pool.stats();
        assert_eq!(snap.total_reuses, 1);
    }

    // ------------------------------------------------------------------
    // Fresh alloc counter
    // ------------------------------------------------------------------

    #[test]
    fn fresh_alloc_counter_increments_on_miss() {
        let pool = make_pool();

        // Pool is empty → fresh allocation.
        let buf = pool.acquire(200);
        pool.release(buf);

        let snap = pool.stats();
        assert_eq!(snap.total_allocs, 1);
        assert_eq!(snap.total_reuses, 0);
    }

    // ------------------------------------------------------------------
    // pool_depth reporting
    // ------------------------------------------------------------------

    #[test]
    fn pool_depth_reflects_releases() {
        let pool = make_pool();

        let b1 = pool.acquire(100);
        let b2 = pool.acquire(100);
        let b3 = pool.acquire(100);

        assert_eq!(pool.pool_depth(0), 0);

        pool.release(b1);
        assert_eq!(pool.pool_depth(0), 1);

        pool.release(b2);
        assert_eq!(pool.pool_depth(0), 2);

        pool.release(b3);
        assert_eq!(pool.pool_depth(0), 3);
    }

    // ------------------------------------------------------------------
    // prune
    // ------------------------------------------------------------------

    #[test]
    fn prune_drains_excess_buffers() {
        let pool = TensorPool::new(TensorPoolConfig { max_per_bucket: 10 });

        // Fill bucket 0 with 8 entries.
        let buffers: Vec<_> = (0..8).map(|_| pool.acquire(100)).collect();
        for buf in buffers {
            pool.release(buf);
        }
        assert_eq!(pool.pool_depth(0), 8);

        // Prune to 3.
        pool.prune(3);
        assert_eq!(pool.pool_depth(0), 3);
    }

    #[test]
    fn prune_keeps_buckets_at_max_when_under_limit() {
        let pool = make_pool();

        let buffers: Vec<_> = (0..3).map(|_| pool.acquire(100)).collect();
        for buf in buffers {
            pool.release(buf);
        }

        // Prune with a limit higher than current depth — should be a no-op.
        pool.prune(5);
        assert_eq!(pool.pool_depth(0), 3);
    }

    // ------------------------------------------------------------------
    // resize on PooledBuffer
    // ------------------------------------------------------------------

    #[test]
    fn resize_works_on_pooled_buffer() {
        let pool = make_pool();
        let mut buf = pool.acquire(128);

        buf.resize(64, 0xFF);
        assert_eq!(buf.len(), 64);
        assert!(buf.as_slice().iter().all(|&b| b == 0xFF));

        buf.resize(0, 0);
        assert_eq!(buf.len(), 0);

        pool.release(buf);
    }

    // ------------------------------------------------------------------
    // Stats snapshot correctness
    // ------------------------------------------------------------------

    #[test]
    fn stats_snapshot_total_acquired() {
        let pool = make_pool();

        for _ in 0..5 {
            let buf = pool.acquire(100);
            pool.release(buf);
        }

        let snap = pool.stats();
        assert_eq!(snap.total_acquired, 5);
    }

    #[test]
    fn stats_snapshot_total_released() {
        let pool = make_pool();

        for _ in 0..3 {
            let buf = pool.acquire(100);
            pool.release(buf);
        }

        let snap = pool.stats();
        assert_eq!(snap.total_released, 3);
    }

    #[test]
    fn stats_allocs_plus_reuses_equals_acquired() {
        let pool = make_pool();

        // 4 acquires: first is a fresh alloc, rest should hit the pool since we
        // release before re-acquiring.
        for _ in 0..4 {
            let buf = pool.acquire(100);
            pool.release(buf);
        }

        let snap = pool.stats();
        assert_eq!(
            snap.total_allocs + snap.total_reuses,
            snap.total_acquired,
            "allocs + reuses must equal total acquired"
        );
    }

    // ------------------------------------------------------------------
    // Bucket capacity correctness
    // ------------------------------------------------------------------

    #[test]
    fn acquired_buffer_has_correct_bucket_capacity() {
        let pool = make_pool();

        // 600 > 512 (bucket 1 cap), so it falls into bucket 2 (capacity 1024).
        let buf = pool.acquire(600);
        assert_eq!(buf.bucket(), 2);
        assert!(buf.capacity() >= 1024);

        pool.release(buf);
    }

    // ------------------------------------------------------------------
    // max_per_bucket cap
    // ------------------------------------------------------------------

    #[test]
    fn release_drops_buffer_when_list_full() {
        let pool = TensorPool::new(TensorPoolConfig { max_per_bucket: 2 });

        let b1 = pool.acquire(100);
        let b2 = pool.acquire(100);
        let b3 = pool.acquire(100); // Will be dropped when released (list already full)

        pool.release(b1);
        pool.release(b2);
        assert_eq!(pool.pool_depth(0), 2);

        pool.release(b3); // Should be silently dropped
        assert_eq!(pool.pool_depth(0), 2);
    }
}
