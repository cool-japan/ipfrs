//! Memory pooling for frequent allocations
//!
//! This module provides memory pools for common allocation patterns:
//! - Block buffer pool (reuse Bytes allocations)
//! - CID string pool (deduplicate strings)
//! - IPLD node pool
//!
//! Memory pooling reduces allocator pressure by reusing existing allocations.

use bytes::{Bytes, BytesMut};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A pool of reusable byte buffers
///
/// This pool maintains a collection of BytesMut buffers that can be reused
/// to reduce allocation overhead when creating blocks.
#[derive(Clone)]
pub struct BytesPool {
    /// Available buffers, organized by capacity bucket
    /// Each bucket contains buffers with capacity in a power-of-2 range
    pool: Arc<Mutex<HashMap<usize, Vec<BytesMut>>>>,
    /// Statistics for the pool
    stats: Arc<Mutex<PoolStats>>,
}

impl Default for BytesPool {
    fn default() -> Self {
        Self::new()
    }
}

impl BytesPool {
    /// Create a new bytes pool
    pub fn new() -> Self {
        Self {
            pool: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(PoolStats::default())),
        }
    }

    /// Get a buffer with at least the requested capacity
    ///
    /// If a suitable buffer is available in the pool, it will be reused.
    /// Otherwise, a new buffer will be allocated.
    pub fn get(&self, capacity: usize) -> BytesMut {
        let bucket = Self::capacity_bucket(capacity);

        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());

        if let Some(buffers) = pool.get_mut(&bucket) {
            if let Some(mut buf) = buffers.pop() {
                buf.clear();
                stats.hits += 1;
                return buf;
            }
        }

        stats.misses += 1;
        stats.allocations += 1;
        BytesMut::with_capacity(bucket)
    }

    /// Return a buffer to the pool for reuse
    ///
    /// The buffer will be cleared and made available for future use.
    pub fn put(&self, mut buf: BytesMut) {
        // Only pool buffers within reasonable size limits
        if buf.capacity() > 1024 * 1024 * 4 {
            // Too large, don't pool
            return;
        }

        buf.clear();
        let bucket = Self::capacity_bucket(buf.capacity());

        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        let buffers = pool.entry(bucket).or_default();

        // Limit pool size per bucket to prevent unbounded growth
        if buffers.len() < 100 {
            buffers.push(buf);
        }
    }

    /// Get the pool statistics
    pub fn stats(&self) -> PoolStats {
        *self.stats.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Clear all pooled buffers
    pub fn clear(&self) {
        self.pool.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// Round capacity up to the nearest power-of-2 bucket
    fn capacity_bucket(capacity: usize) -> usize {
        if capacity == 0 {
            return 1024; // Minimum 1KB
        }
        capacity.next_power_of_two().max(1024)
    }
}

/// A pool for CID strings to reduce duplication
///
/// This pool maintains a cache of CID strings that have been seen before,
/// allowing them to be deduplicated and reused.
#[derive(Clone)]
pub struct CidStringPool {
    /// Interned strings
    pool: Arc<Mutex<HashMap<String, Arc<str>>>>,
    /// Statistics for the pool
    stats: Arc<Mutex<PoolStats>>,
}

impl Default for CidStringPool {
    fn default() -> Self {
        Self::new()
    }
}

impl CidStringPool {
    /// Create a new CID string pool
    pub fn new() -> Self {
        Self {
            pool: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(PoolStats::default())),
        }
    }

    /// Intern a CID string
    ///
    /// If the string has been seen before, returns the existing Arc.
    /// Otherwise, creates a new Arc and stores it in the pool.
    pub fn intern(&self, s: &str) -> Arc<str> {
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());

        if let Some(existing) = pool.get(s) {
            stats.hits += 1;
            return Arc::clone(existing);
        }

        stats.misses += 1;
        let arc: Arc<str> = Arc::from(s);
        pool.insert(s.to_string(), Arc::clone(&arc));
        arc
    }

    /// Get the pool statistics
    pub fn stats(&self) -> PoolStats {
        *self.stats.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Clear the pool
    pub fn clear(&self) {
        self.pool.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// Get the number of unique strings in the pool
    pub fn len(&self) -> usize {
        self.pool.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Check if the pool is empty
    pub fn is_empty(&self) -> bool {
        self.pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }
}

/// Statistics for a memory pool
#[derive(Debug, Clone, Copy, Default)]
pub struct PoolStats {
    /// Number of successful retrievals from the pool
    pub hits: u64,
    /// Number of times a new allocation was needed
    pub misses: u64,
    /// Total number of allocations made
    pub allocations: u64,
}

impl PoolStats {
    /// Calculate the hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    /// Calculate the miss rate (0.0 to 1.0)
    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate()
    }
}

/// A global bytes pool instance
static GLOBAL_BYTES_POOL: once_cell::sync::Lazy<BytesPool> =
    once_cell::sync::Lazy::new(BytesPool::new);

/// A global CID string pool instance
static GLOBAL_CID_STRING_POOL: once_cell::sync::Lazy<CidStringPool> =
    once_cell::sync::Lazy::new(CidStringPool::new);

/// Get the global bytes pool
pub fn global_bytes_pool() -> &'static BytesPool {
    &GLOBAL_BYTES_POOL
}

/// Get the global CID string pool
pub fn global_cid_string_pool() -> &'static CidStringPool {
    &GLOBAL_CID_STRING_POOL
}

/// Helper to convert BytesMut to Bytes efficiently
pub fn freeze_bytes(buf: BytesMut) -> Bytes {
    buf.freeze()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_pool_basic() {
        let pool = BytesPool::new();

        // Get a buffer
        let buf1 = pool.get(1024);
        assert!(buf1.capacity() >= 1024);

        // Return it
        pool.put(buf1);

        // Get another buffer - should reuse the same one
        let buf2 = pool.get(1024);
        assert!(buf2.capacity() >= 1024);

        let stats = pool.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_bytes_pool_capacity_bucketing() {
        let pool = BytesPool::new();

        // Request different sizes
        let buf1 = pool.get(100);
        let buf2 = pool.get(1000);
        let buf3 = pool.get(2000);

        // All should be bucketed to power-of-2 sizes
        assert!(buf1.capacity() >= 100);
        assert!(buf2.capacity() >= 1000);
        assert!(buf3.capacity() >= 2000);

        pool.put(buf1);
        pool.put(buf2);
        pool.put(buf3);

        // Request similar sizes - should reuse
        let buf4 = pool.get(150); // Should reuse first bucket
        let buf5 = pool.get(1100); // Should reuse second bucket

        assert!(buf4.capacity() >= 150);
        assert!(buf5.capacity() >= 1100);
    }

    #[test]
    fn test_cid_string_pool_basic() {
        let pool = CidStringPool::new();

        // Intern a string
        let s1 = pool.intern("QmTest123");
        let s2 = pool.intern("QmTest123");

        // Should be the same Arc
        assert_eq!(s1.as_ref(), s2.as_ref());
        assert!(Arc::ptr_eq(&s1, &s2));

        let stats = pool.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cid_string_pool_different_strings() {
        let pool = CidStringPool::new();

        let s1 = pool.intern("QmTest1");
        let s2 = pool.intern("QmTest2");

        // Should be different
        assert_ne!(s1.as_ref(), s2.as_ref());
        assert!(!Arc::ptr_eq(&s1, &s2));

        let stats = pool.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 2);
    }

    #[test]
    fn test_pool_stats_hit_rate() {
        let stats = PoolStats {
            hits: 80,
            misses: 20,
            allocations: 20,
        };

        assert!((stats.hit_rate() - 0.8).abs() < 0.001);
        assert!((stats.miss_rate() - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_pool_stats_empty() {
        let stats = PoolStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
        assert_eq!(stats.miss_rate(), 1.0);
    }

    #[test]
    fn test_bytes_pool_clear() {
        let pool = BytesPool::new();
        let buf = pool.get(1024);
        pool.put(buf);

        pool.clear();

        // After clear, should allocate new buffer
        let _buf2 = pool.get(1024);
        let stats = pool.stats();
        assert_eq!(stats.misses, 2); // Both allocations were misses
    }

    #[test]
    fn test_cid_string_pool_len() {
        let pool = CidStringPool::new();
        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());

        pool.intern("QmTest1");
        assert_eq!(pool.len(), 1);
        assert!(!pool.is_empty());

        pool.intern("QmTest2");
        assert_eq!(pool.len(), 2);

        pool.intern("QmTest1"); // Duplicate
        assert_eq!(pool.len(), 2); // Should still be 2
    }

    #[test]
    fn test_bytes_pool_size_limit() {
        let pool = BytesPool::new();

        // Very large buffer shouldn't be pooled
        let large_buf = BytesMut::with_capacity(10 * 1024 * 1024);
        pool.put(large_buf);

        // Try to get a large buffer - should allocate new
        let _buf = pool.get(10 * 1024 * 1024);
        let stats = pool.stats();

        // Both should be misses (large buffer wasn't pooled)
        assert!(stats.misses >= 1);
    }

    #[test]
    fn test_global_pools() {
        let bytes_pool = global_bytes_pool();
        let cid_pool = global_cid_string_pool();

        // Just ensure they're accessible
        let _buf = bytes_pool.get(1024);
        let _s = cid_pool.intern("QmTest");
    }

    #[test]
    fn test_freeze_bytes() {
        let mut buf = BytesMut::with_capacity(1024);
        buf.extend_from_slice(b"Hello, world!");
        let bytes = freeze_bytes(buf);
        assert_eq!(&bytes[..], b"Hello, world!");
    }
}
