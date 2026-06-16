//! Arena allocator for inference pipeline tensor memory management.
//!
//! Inference pipelines allocate many small tensors. An arena allocator avoids
//! per-tensor heap allocation by bump-allocating from pre-allocated slabs.
//! This module provides [`TensorArena`], [`ArenaRegion`], [`ArenaSlice`],
//! [`ArenaStats`], and [`ArenaError`].
//!
//! # Design
//!
//! The arena is organized as a list of fixed-size [`ArenaRegion`] slabs.
//! Each region is a `Vec<u8>` with a bump pointer (`offset`). Allocations
//! are always 8-byte aligned. When a region is full a new one is created.
//!
//! Lifetimes are avoided by representing allocated memory as `(start, end)`
//! byte ranges within a region's slab, wrapped in [`ArenaSlice`].
//!
//! # Example
//!
//! ```
//! use ipfrs_tensorlogic::tensor_arena::{TensorArena, ArenaError};
//!
//! let mut arena = TensorArena::new(1024 * 1024); // 1 MB regions
//!
//! // Allocate space for 4 f32 values (16 bytes)
//! let slice = arena.allocate(4 * 4);
//!
//! // Write and read back
//! slice.write_f32(&mut arena, &[1.0, 2.0, 3.0, 4.0]).expect("example: should succeed in docs");
//! let values = slice.read_f32(&arena);
//! assert_eq!(values, &[1.0f32, 2.0, 3.0, 4.0]);
//!
//! // Reset for reuse (no deallocation)
//! arena.reset_all();
//! ```

use bytemuck::{cast_slice, cast_slice_mut};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Fixed alignment for all arena allocations (8 bytes).
const ARENA_ALIGN: usize = 8;

// ---------------------------------------------------------------------------
// ArenaError
// ---------------------------------------------------------------------------

/// Errors that can occur during arena operations.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ArenaError {
    /// The number of bytes provided does not match the slice's expected size.
    #[error("size mismatch: expected {expected} bytes, got {got} bytes")]
    SizeMismatch { expected: usize, got: usize },

    /// The region index stored in an [`ArenaSlice`] does not exist.
    #[error("region index {0} not found in arena")]
    RegionNotFound(usize),
}

// ---------------------------------------------------------------------------
// ArenaStats
// ---------------------------------------------------------------------------

/// Cumulative statistics for a [`TensorArena`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ArenaStats {
    /// Total number of successful `allocate` calls.
    pub total_allocations: u64,
    /// Total bytes handed out across all allocations.
    pub total_bytes_allocated: u64,
    /// Number of times [`TensorArena::reset_all`] has been called.
    pub total_resets: u64,
    /// Number of [`ArenaRegion`] slabs that have been created.
    pub regions_created: u64,
}

// ---------------------------------------------------------------------------
// ArenaRegion
// ---------------------------------------------------------------------------

/// A contiguous slab of memory with a bump pointer.
///
/// Memory is never freed individually. Call [`reset`](ArenaRegion::reset) to
/// reclaim the entire region at once.
#[derive(Debug)]
pub struct ArenaRegion {
    /// Backing byte storage.
    slab: Vec<u8>,
    /// Current bump pointer — index of the first unused byte.
    offset: usize,
    /// Total capacity in bytes.
    capacity: usize,
}

impl ArenaRegion {
    /// Create a new region with `capacity` bytes pre-allocated.
    pub fn new(capacity: usize) -> Self {
        Self {
            slab: vec![0u8; capacity],
            offset: 0,
            capacity,
        }
    }

    /// Bump-allocate `size` bytes aligned to `ARENA_ALIGN`.
    ///
    /// Returns `Some((start, end))` where the range `[start, end)` inside
    /// `slab` is exclusively owned by the caller until `reset` is called.
    /// Returns `None` if there is insufficient space.
    pub fn allocate(&mut self, size: usize) -> Option<(usize, usize)> {
        if size == 0 {
            return Some((self.offset, self.offset));
        }

        // Align the current offset up to ARENA_ALIGN.
        let aligned_start = align_up(self.offset, ARENA_ALIGN);
        let end = aligned_start.checked_add(size)?;

        if end > self.capacity {
            return None;
        }

        self.offset = end;
        Some((aligned_start, end))
    }

    /// Bytes still available for allocation (before alignment waste).
    pub fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.offset)
    }

    /// Fraction of capacity that has been allocated (`[0.0, 1.0]`).
    ///
    /// Returns `0.0` for a zero-capacity region to avoid division by zero.
    pub fn utilization(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.offset as f64 / self.capacity as f64
    }

    /// Reset the bump pointer to zero, logically freeing all allocations.
    ///
    /// This does **not** release the backing `Vec` memory.
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Immutable view of the backing slab.
    #[inline]
    pub fn slab(&self) -> &[u8] {
        &self.slab
    }

    /// Mutable view of the backing slab.
    #[inline]
    pub fn slab_mut(&mut self) -> &mut [u8] {
        &mut self.slab
    }

    /// Total capacity in bytes.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current bump-pointer offset.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }
}

// ---------------------------------------------------------------------------
// ArenaSlice
// ---------------------------------------------------------------------------

/// A handle to a contiguous byte range inside a [`TensorArena`] region.
///
/// Validity is not enforced by the type system — the caller must not use an
/// `ArenaSlice` after calling [`TensorArena::reset_all`] if new allocations
/// have been made that may overlap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaSlice {
    /// Index of the owning region in [`TensorArena::regions`].
    pub region_index: usize,
    /// Start byte offset (inclusive) within the region slab.
    pub start: usize,
    /// End byte offset (exclusive) within the region slab.
    pub end: usize,
}

impl ArenaSlice {
    /// Number of bytes in this slice.
    #[inline]
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Returns `true` if the slice covers zero bytes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Write `values` (as raw bytes) into this slice's range in the arena.
    ///
    /// # Errors
    ///
    /// - [`ArenaError::RegionNotFound`] — if `self.region_index` is out of
    ///   bounds.
    /// - [`ArenaError::SizeMismatch`] — if `values.len() * 4 != self.len()`.
    pub fn write_f32(&self, arena: &mut TensorArena, values: &[f32]) -> Result<(), ArenaError> {
        let region = arena
            .regions
            .get_mut(self.region_index)
            .ok_or(ArenaError::RegionNotFound(self.region_index))?;

        let expected = self.len();
        let got = std::mem::size_of_val(values);

        if expected != got {
            return Err(ArenaError::SizeMismatch { expected, got });
        }

        let dst: &mut [f32] = cast_slice_mut(&mut region.slab[self.start..self.end]);
        dst.copy_from_slice(values);
        Ok(())
    }

    /// Read a `&[f32]` from this slice's range in the arena.
    ///
    /// # Panics
    ///
    /// Panics if `self.region_index` is out of bounds or if the byte range is
    /// not properly aligned / sized for `f32`.  In production code the caller
    /// should ensure the slice was created by [`TensorArena::allocate`] with a
    /// size that is a multiple of 4.
    pub fn read_f32<'a>(&self, arena: &'a TensorArena) -> &'a [f32] {
        let region = &arena.regions[self.region_index];
        cast_slice(&region.slab[self.start..self.end])
    }
}

// ---------------------------------------------------------------------------
// TensorArena
// ---------------------------------------------------------------------------

/// Bump-allocating arena for inference-pipeline tensors.
///
/// Allocations are O(1) — a bump pointer is incremented and, when a region is
/// exhausted, a new fixed-size region is appended.  Freeing all tensors is
/// O(regions) via [`reset_all`](TensorArena::reset_all).
pub struct TensorArena {
    /// All memory regions, ordered by creation time.
    pub regions: Vec<ArenaRegion>,
    /// Default size (bytes) for each new region.
    pub region_size: usize,
    /// Cumulative statistics.
    pub stats: ArenaStats,
}

impl TensorArena {
    /// Construct a new arena whose regions will be `region_size` bytes each.
    pub fn new(region_size: usize) -> Self {
        let mut arena = Self {
            regions: Vec::new(),
            region_size,
            stats: ArenaStats::default(),
        };
        // Pre-allocate the first region eagerly so that the first allocation
        // does not trigger a regions_created bump that is surprising to callers
        // that inspect stats before any allocation.  The region is included in
        // regions_created.
        arena.push_region();
        arena
    }

    /// Allocate `size` bytes from the arena and return an [`ArenaSlice`].
    ///
    /// If the current (last) region cannot satisfy the request a new region is
    /// created.  `size` is always rounded up to the next multiple of
    /// `ARENA_ALIGN` internally.
    pub fn allocate(&mut self, size: usize) -> ArenaSlice {
        let size = align_up(size, ARENA_ALIGN);

        // Try the current last region first.
        let slice = self.try_allocate_in_last(size);

        let slice = if let Some(s) = slice {
            s
        } else {
            // Need a new region — make it large enough even for oversized
            // allocations.
            let region_size = self.region_size.max(size);
            // Temporarily store region_size; push_region_with_size uses it.
            let old_region_size = self.region_size;
            self.region_size = region_size;
            self.push_region();
            self.region_size = old_region_size;

            self.try_allocate_in_last(size)
                .expect("freshly created region must accommodate the allocation")
        };

        self.stats.total_allocations += 1;
        self.stats.total_bytes_allocated += size as u64;
        slice
    }

    /// Reset every region's bump pointer, allowing all memory to be reused.
    ///
    /// Outstanding [`ArenaSlice`] handles become stale after this call.
    pub fn reset_all(&mut self) {
        for region in &mut self.regions {
            region.reset();
        }
        self.stats.total_resets += 1;
    }

    /// Number of regions (slabs) currently held.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Total capacity in bytes across all regions.
    pub fn total_capacity(&self) -> u64 {
        self.regions.iter().map(|r| r.capacity() as u64).sum()
    }

    /// Total bytes that have been bump-allocated (not yet reset).
    pub fn total_used(&self) -> u64 {
        self.regions.iter().map(|r| r.offset() as u64).sum()
    }

    /// Overall utilization as `total_used / total_capacity`.
    ///
    /// Returns `0.0` if there are no regions.
    pub fn utilization(&self) -> f64 {
        let cap = self.total_capacity();
        if cap == 0 {
            return 0.0;
        }
        self.total_used() as f64 / cap as f64
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Try to allocate `size` bytes from the last region.
    fn try_allocate_in_last(&mut self, size: usize) -> Option<ArenaSlice> {
        let idx = self.regions.len().checked_sub(1)?;
        let region = &mut self.regions[idx];
        let (start, end) = region.allocate(size)?;
        Some(ArenaSlice {
            region_index: idx,
            start,
            end,
        })
    }

    /// Append a new region of `self.region_size` bytes.
    fn push_region(&mut self) {
        self.regions.push(ArenaRegion::new(self.region_size));
        self.stats.regions_created += 1;
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Round `n` up to the nearest multiple of `align` (which must be a power of 2).
#[inline]
fn align_up(n: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (n + align - 1) & !(align - 1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // ArenaRegion tests
    // -----------------------------------------------------------------------

    /// Test 1: allocate returns the correct (start, end) range.
    #[test]
    fn region_allocate_returns_correct_range() {
        let mut region = ArenaRegion::new(64);
        let result = region.allocate(16);
        assert_eq!(result, Some((0, 16)));
    }

    /// Test 2: multiple allocations are sequential and non-overlapping.
    #[test]
    fn region_multiple_allocations_are_sequential() {
        let mut region = ArenaRegion::new(128);
        let r1 = region.allocate(16).expect("test: should succeed");
        let r2 = region.allocate(16).expect("test: should succeed");
        // Second allocation must start where first ended.
        assert_eq!(r2.0, r1.1);
        assert_eq!(r1, (0, 16));
        assert_eq!(r2, (16, 32));
    }

    /// Test 3: remaining decreases after allocation.
    #[test]
    fn region_remaining_decreases_after_allocation() {
        let mut region = ArenaRegion::new(64);
        let before = region.remaining();
        region.allocate(16).expect("test: should succeed");
        let after = region.remaining();
        assert!(after < before, "remaining should decrease after allocation");
        assert_eq!(after, before - 16);
    }

    /// Test 4: allocate returns None when space is insufficient.
    #[test]
    fn region_allocate_returns_none_when_full() {
        let mut region = ArenaRegion::new(8);
        assert!(region.allocate(16).is_none());
    }

    /// Test 5: reset restores remaining to capacity.
    #[test]
    fn region_reset_restores_remaining() {
        let mut region = ArenaRegion::new(64);
        region.allocate(32).expect("test: should succeed");
        assert_eq!(region.remaining(), 32);
        region.reset();
        assert_eq!(region.remaining(), 64);
    }

    /// Test 6: utilization is 0 on fresh region.
    #[test]
    fn region_utilization_is_zero_initially() {
        let region = ArenaRegion::new(1024);
        assert_eq!(region.utilization(), 0.0);
    }

    /// Test 7: utilization reaches ~1.0 when almost full.
    #[test]
    fn region_utilization_approaches_one_when_full() {
        let mut region = ArenaRegion::new(64);
        region.allocate(64).expect("test: should succeed");
        let u = region.utilization();
        assert!((u - 1.0).abs() < 1e-9, "expected ~1.0, got {u}");
    }

    // -----------------------------------------------------------------------
    // TensorArena tests
    // -----------------------------------------------------------------------

    /// Test 8: arena starts with one region.
    #[test]
    fn arena_starts_with_one_region() {
        let arena = TensorArena::new(1024);
        assert_eq!(arena.region_count(), 1);
    }

    /// Test 9: allocate creates a new region when the first is full.
    #[test]
    fn arena_allocate_creates_new_region_when_needed() {
        let region_size = 32;
        let mut arena = TensorArena::new(region_size);
        // Fill the first region.
        arena.allocate(32);
        // Next allocation must spill into a second region.
        let slice = arena.allocate(8);
        assert_eq!(arena.region_count(), 2);
        assert_eq!(slice.region_index, 1);
    }

    /// Test 10: write_f32 / read_f32 round-trip.
    #[test]
    fn arena_write_read_f32_round_trip() {
        let mut arena = TensorArena::new(1024 * 1024);
        let values: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        let byte_len = values.len() * core::mem::size_of::<f32>();
        let slice = arena.allocate(byte_len);
        slice
            .write_f32(&mut arena, &values)
            .expect("test: should succeed");
        let read_back = slice.read_f32(&arena);
        assert_eq!(read_back, values.as_slice());
    }

    /// Test 11: reset_all allows memory to be reused.
    #[test]
    fn arena_reset_all_allows_reuse() {
        let mut arena = TensorArena::new(64);
        let s1 = arena.allocate(16);
        arena.reset_all();
        let s2 = arena.allocate(16);
        // After reset the bump pointer starts over, so offsets should match.
        assert_eq!(s1.start, s2.start);
        assert_eq!(s1.end, s2.end);
    }

    /// Test 12: utilization calculation is correct.
    #[test]
    fn arena_utilization_calculation() {
        let region_size = 1024;
        let mut arena = TensorArena::new(region_size);
        // Allocate exactly half the first region.
        arena.allocate(512);
        let u = arena.utilization();
        assert!((u - 0.5).abs() < 0.01, "expected ~0.5 utilization, got {u}");
    }

    /// Test 13: stats accumulate correctly across multiple allocations.
    #[test]
    fn arena_stats_accumulate() {
        let mut arena = TensorArena::new(1024 * 1024);
        arena.allocate(16);
        arena.allocate(32);
        arena.allocate(64);
        assert_eq!(arena.stats.total_allocations, 3);
        // Bytes are aligned to 8 so: 16 + 32 + 64 = 112
        assert_eq!(arena.stats.total_bytes_allocated, 112);
    }

    /// Test 14: large allocation triggers a new region sized for the request.
    #[test]
    fn arena_large_allocation_triggers_new_region() {
        let region_size = 64;
        let mut arena = TensorArena::new(region_size);
        // This is larger than the default region_size.
        let large = 4096;
        let slice = arena.allocate(large);
        // A new region must have been created to accommodate this.
        // (The first region has region_size = 64 bytes, but 4096 > 64.)
        let region = &arena.regions[slice.region_index];
        assert!(region.capacity() >= large);
    }

    /// Test 15: write_f32 returns SizeMismatch error on wrong length.
    #[test]
    fn arena_write_f32_size_mismatch_error() {
        let mut arena = TensorArena::new(1024);
        let slice = arena.allocate(16); // 4 f32
        let result = slice.write_f32(&mut arena, &[1.0, 2.0]); // only 2 f32
        assert!(matches!(result, Err(ArenaError::SizeMismatch { .. })));
    }

    /// Test 16: multiple allocations in the same region are non-overlapping.
    #[test]
    fn arena_multiple_allocs_in_same_region_non_overlapping() {
        let mut arena = TensorArena::new(1024 * 1024);
        let a = arena.allocate(64);
        let b = arena.allocate(64);
        let c = arena.allocate(128);
        // All in the first region.
        assert_eq!(a.region_index, b.region_index);
        assert_eq!(b.region_index, c.region_index);
        // Non-overlapping: a ends where b starts.
        assert_eq!(a.end, b.start);
        assert_eq!(b.end, c.start);
    }

    /// Test 17: reset_all increments total_resets stat.
    #[test]
    fn arena_reset_all_increments_stats() {
        let mut arena = TensorArena::new(1024);
        arena.reset_all();
        arena.reset_all();
        assert_eq!(arena.stats.total_resets, 2);
    }

    /// Test 18: region_index in ArenaSlice is valid after multi-region spill.
    #[test]
    fn arena_slice_region_index_valid_after_spill() {
        let region_size = 32;
        let mut arena = TensorArena::new(region_size);
        arena.allocate(32); // fill region 0
        let s = arena.allocate(8); // spills to region 1
                                   // The slice is 8 bytes (aligned), so write two f32 values.
        s.write_f32(&mut arena, &[42.0, 7.0])
            .expect("test: should succeed");
        let vals = s.read_f32(&arena);
        assert_eq!(vals, &[42.0f32, 7.0]);
    }
}
