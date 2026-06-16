//! Tensor Memory Pool
//!
//! This module provides two complementary memory pool implementations:
//!
//! 1. **[`TensorMemoryPool`]** — Slab-based pool with size-class bucketing for
//!    efficient allocation of variable-sized tensor buffers.
//!
//! 2. **[`TensorBlockPool`]** — Pre-allocated fixed-size block pool that reduces
//!    allocation overhead by maintaining a reservoir of identically-sized memory
//!    blocks with owner tracking, reservation support, and defragmentation.
//!
//! # Slab Pool (TensorMemoryPool)
//!
//! Uses four size classes (Small/Medium/Large/Huge) with power-of-two bucket
//! sizes.  Best suited when tensor sizes vary widely.
//!
//! ```
//! use ipfrs_tensorlogic::memory_pool::{TensorMemoryPool, SizeClass};
//!
//! let mut pool = TensorMemoryPool::new(64);
//! let slot_id = pool.allocate(2048, 1).expect("example: should succeed in docs");
//! assert!(pool.deallocate(slot_id, 2));
//! let slot_id2 = pool.allocate(1024, 3).expect("example: should succeed in docs");
//! assert_eq!(slot_id, slot_id2);
//! ```
//!
//! # Block Pool (TensorBlockPool)
//!
//! All blocks share a single configured size.  Supports reservation, owner
//! tracking, generation counting, defragmentation, and shrink-to-fit.
//!
//! ```
//! use ipfrs_tensorlogic::memory_pool::{TensorBlockPool, PoolConfig};
//!
//! let config = PoolConfig::default();
//! let mut pool = TensorBlockPool::new(config);
//! let id = pool.allocate("matmul").expect("example: should succeed in docs");
//! assert!(pool.deallocate(id).is_ok());
//! ```

use std::collections::HashMap;

// ===========================================================================
// Part 1: Slab-based TensorMemoryPool (original implementation)
// ===========================================================================

// ---------------------------------------------------------------------------
// SizeClass
// ---------------------------------------------------------------------------

/// Categorises a byte count into one of four allocation buckets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SizeClass {
    /// <= 4 096 bytes
    Small,
    /// <= 65 536 bytes
    Medium,
    /// <= 1 048 576 bytes
    Large,
    /// > 1 048 576 bytes
    Huge,
}

impl SizeClass {
    /// Classify a byte count into the appropriate size class.
    pub fn classify(bytes: u64) -> SizeClass {
        if bytes <= 4_096 {
            SizeClass::Small
        } else if bytes <= 65_536 {
            SizeClass::Medium
        } else if bytes <= 1_048_576 {
            SizeClass::Large
        } else {
            SizeClass::Huge
        }
    }

    /// Return the canonical bucket allocation size for this class.
    pub fn bucket_size(&self) -> u64 {
        match self {
            SizeClass::Small => 4_096,
            SizeClass::Medium => 65_536,
            SizeClass::Large => 1_048_576,
            SizeClass::Huge => 16_777_216,
        }
    }
}

// ---------------------------------------------------------------------------
// PoolSlot
// ---------------------------------------------------------------------------

/// A single slot managed by the [`TensorMemoryPool`].
#[derive(Clone, Debug)]
pub struct PoolSlot {
    /// Unique identifier for this slot.
    pub slot_id: u64,
    /// Size class this slot belongs to.
    pub size_class: SizeClass,
    /// Bytes requested when this slot was allocated (<= `size_class.bucket_size()`).
    pub allocated_bytes: u64,
    /// Whether this slot is currently checked out.
    pub in_use: bool,
    /// The monotonic tick at which this slot was last accessed.
    pub last_used_tick: u64,
}

// ---------------------------------------------------------------------------
// MemoryPoolStats (slab pool)
// ---------------------------------------------------------------------------

/// Aggregate statistics for a [`TensorMemoryPool`].
#[derive(Clone, Debug)]
pub struct MemoryPoolStats {
    /// Total number of slots currently tracked by the pool.
    pub total_slots: usize,
    /// Number of slots that are currently checked out.
    pub in_use_slots: usize,
    /// Number of slots available for reuse.
    pub free_slots: usize,
    /// Sum of `bucket_size` for every slot in the pool.
    pub total_allocated_bytes: u64,
    /// Sum of `(bucket_size - allocated_bytes)` for every in-use slot.
    pub wasted_bytes: u64,
}

impl MemoryPoolStats {
    /// Fraction of slots currently in use (`0.0` when the pool is empty).
    pub fn utilization(&self) -> f64 {
        if self.total_slots == 0 {
            0.0
        } else {
            self.in_use_slots as f64 / self.total_slots as f64
        }
    }
}

// ---------------------------------------------------------------------------
// TensorMemoryPool (slab-based)
// ---------------------------------------------------------------------------

/// A slab-based memory pool that pre-allocates tensor buffers organised into
/// size-class buckets to minimise allocation overhead on hot inference paths.
pub struct TensorMemoryPool {
    /// All slots, keyed by `slot_id`.
    pub slots: HashMap<u64, PoolSlot>,
    /// Counter used to generate unique slot IDs.
    pub next_slot_id: u64,
    /// Hard upper bound on the number of slots the pool may hold.
    pub max_slots: usize,
    /// Total successful allocations since creation.
    pub total_allocations: u64,
    /// Total successful deallocations since creation.
    pub total_deallocations: u64,
}

impl TensorMemoryPool {
    /// Create a new pool with the given slot capacity.
    pub fn new(max_slots: usize) -> Self {
        TensorMemoryPool {
            slots: HashMap::new(),
            next_slot_id: 0,
            max_slots,
            total_allocations: 0,
            total_deallocations: 0,
        }
    }

    /// Attempt to check out a slot suitable for `requested_bytes` at the given
    /// logical `tick`.
    pub fn allocate(&mut self, requested_bytes: u64, tick: u64) -> Option<u64> {
        let class = SizeClass::classify(requested_bytes);

        let reuse_id = self
            .slots
            .values()
            .filter(|s| !s.in_use && s.size_class == class)
            .map(|s| s.slot_id)
            .next();

        if let Some(id) = reuse_id {
            let slot = self.slots.get_mut(&id)?;
            slot.in_use = true;
            slot.allocated_bytes = requested_bytes;
            slot.last_used_tick = tick;
            self.total_allocations += 1;
            return Some(id);
        }

        if self.slots.len() >= self.max_slots {
            return None;
        }

        let id = self.next_slot_id;
        self.next_slot_id += 1;

        let slot = PoolSlot {
            slot_id: id,
            size_class: class,
            allocated_bytes: requested_bytes,
            in_use: true,
            last_used_tick: tick,
        };
        self.slots.insert(id, slot);
        self.total_allocations += 1;
        Some(id)
    }

    /// Return slot `slot_id` to the pool at the given `tick`.
    ///
    /// Returns `false` if the slot does not exist or was already free.
    pub fn deallocate(&mut self, slot_id: u64, tick: u64) -> bool {
        match self.slots.get_mut(&slot_id) {
            Some(slot) if slot.in_use => {
                slot.in_use = false;
                slot.last_used_tick = tick;
                self.total_deallocations += 1;
                true
            }
            _ => false,
        }
    }

    /// Evict every **free** slot whose last-used tick is at least `idle_ticks`
    /// ticks in the past.
    pub fn evict_idle(&mut self, tick: u64, idle_ticks: u64) {
        self.slots.retain(|_, slot| {
            if slot.in_use {
                return true;
            }
            let age = tick.saturating_sub(slot.last_used_tick);
            age < idle_ticks
        });
    }

    /// Return all slots belonging to `class`, sorted by `slot_id` ascending.
    pub fn slots_for_class(&self, class: SizeClass) -> Vec<&PoolSlot> {
        let mut result: Vec<&PoolSlot> = self
            .slots
            .values()
            .filter(|s| s.size_class == class)
            .collect();
        result.sort_by_key(|s| s.slot_id);
        result
    }

    /// Compute aggregate statistics for the current pool state.
    pub fn stats(&self) -> MemoryPoolStats {
        let total_slots = self.slots.len();
        let in_use_slots = self.slots.values().filter(|s| s.in_use).count();
        let free_slots = total_slots - in_use_slots;

        let total_allocated_bytes: u64 = self
            .slots
            .values()
            .map(|s| s.size_class.bucket_size())
            .sum();

        let wasted_bytes: u64 = self
            .slots
            .values()
            .filter(|s| s.in_use)
            .map(|s| s.size_class.bucket_size().saturating_sub(s.allocated_bytes))
            .sum();

        MemoryPoolStats {
            total_slots,
            in_use_slots,
            free_slots,
            total_allocated_bytes,
            wasted_bytes,
        }
    }
}

// ===========================================================================
// Part 2: TensorBlockPool — pre-allocated fixed-size block pool
// ===========================================================================

// ---------------------------------------------------------------------------
// BlockStatus
// ---------------------------------------------------------------------------

/// Status of a single block within a [`TensorBlockPool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockStatus {
    /// Available for allocation.
    Free,
    /// Currently owned by an operation.
    Allocated,
    /// Temporarily held for future use.
    Reserved,
}

// ---------------------------------------------------------------------------
// MemoryBlock
// ---------------------------------------------------------------------------

/// A single block managed by the [`TensorBlockPool`].
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    /// Unique identifier for this block.
    pub id: u64,
    /// Size of the block in bytes.
    pub size: usize,
    /// Current status of the block.
    pub status: BlockStatus,
    /// Name of the operation that currently owns this block.
    pub owner: Option<String>,
    /// Incremented on each reuse (deallocation cycle).
    pub generation: u64,
}

// ---------------------------------------------------------------------------
// PoolConfig
// ---------------------------------------------------------------------------

/// Configuration for a [`TensorBlockPool`].
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Number of blocks to pre-allocate on construction.
    pub initial_blocks: usize,
    /// Size of each block in bytes.
    pub block_size: usize,
    /// Maximum number of blocks the pool may hold.
    pub max_blocks: usize,
    /// Whether the pool may grow beyond `initial_blocks` (up to `max_blocks`).
    pub allow_growth: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig {
            initial_blocks: 64,
            block_size: 4096,
            max_blocks: 1024,
            allow_growth: true,
        }
    }
}

// ---------------------------------------------------------------------------
// BlockPoolStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for a [`TensorBlockPool`].
#[derive(Debug, Clone)]
pub struct BlockPoolStats {
    /// Total number of blocks in the pool.
    pub total_blocks: usize,
    /// Number of blocks with status [`BlockStatus::Free`].
    pub free_blocks: usize,
    /// Number of blocks with status [`BlockStatus::Allocated`].
    pub allocated_blocks: usize,
    /// Number of blocks with status [`BlockStatus::Reserved`].
    pub reserved_blocks: usize,
    /// Peak number of simultaneously allocated blocks.
    pub peak_allocated: usize,
    /// Total successful allocations since creation.
    pub total_allocations: u64,
    /// Total successful deallocations since creation.
    pub total_deallocations: u64,
    /// Fraction of blocks currently allocated (allocated / total).
    pub utilization: f64,
}

// ---------------------------------------------------------------------------
// TensorBlockPool
// ---------------------------------------------------------------------------

/// Pre-allocated memory pool for tensor operations to reduce allocation
/// overhead.
///
/// All blocks share a single configured size.  The pool supports allocation,
/// deallocation, reservation, defragmentation, and shrink-to-fit.
pub struct TensorBlockPool {
    config: PoolConfig,
    blocks: Vec<MemoryBlock>,
    next_id: u64,
    allocations: u64,
    deallocations: u64,
    peak_allocated: usize,
    current_allocated: usize,
}

impl TensorBlockPool {
    /// Create a new block pool, pre-allocating `config.initial_blocks` blocks.
    pub fn new(config: PoolConfig) -> Self {
        let mut blocks = Vec::with_capacity(config.initial_blocks);
        for i in 0..config.initial_blocks {
            blocks.push(MemoryBlock {
                id: i as u64,
                size: config.block_size,
                status: BlockStatus::Free,
                owner: None,
                generation: 0,
            });
        }
        TensorBlockPool {
            next_id: config.initial_blocks as u64,
            config,
            blocks,
            allocations: 0,
            deallocations: 0,
            peak_allocated: 0,
            current_allocated: 0,
        }
    }

    /// Allocate a free block, marking it as [`BlockStatus::Allocated`] and
    /// assigning it to `owner`.
    ///
    /// If no free block is available and `allow_growth` is true, a new block is
    /// created (up to `max_blocks`).  Returns an error if the pool is exhausted.
    pub fn allocate(&mut self, owner: &str) -> Result<u64, String> {
        // Search for a free block.
        if let Some(block) = self
            .blocks
            .iter_mut()
            .find(|b| b.status == BlockStatus::Free)
        {
            block.status = BlockStatus::Allocated;
            block.owner = Some(owner.to_string());
            self.allocations += 1;
            self.current_allocated += 1;
            if self.current_allocated > self.peak_allocated {
                self.peak_allocated = self.current_allocated;
            }
            return Ok(block.id);
        }

        // No free block — try to grow.
        if !self.config.allow_growth {
            return Err("pool exhausted and growth is disabled".to_string());
        }
        if self.blocks.len() >= self.config.max_blocks {
            return Err(format!(
                "pool exhausted: max_blocks ({}) reached",
                self.config.max_blocks
            ));
        }

        let id = self.next_id;
        self.next_id += 1;
        self.blocks.push(MemoryBlock {
            id,
            size: self.config.block_size,
            status: BlockStatus::Allocated,
            owner: Some(owner.to_string()),
            generation: 0,
        });
        self.allocations += 1;
        self.current_allocated += 1;
        if self.current_allocated > self.peak_allocated {
            self.peak_allocated = self.current_allocated;
        }
        Ok(id)
    }

    /// Deallocate the block with the given `block_id`, marking it
    /// [`BlockStatus::Free`], clearing its owner, and incrementing its
    /// generation.
    pub fn deallocate(&mut self, block_id: u64) -> Result<(), String> {
        let block = self
            .blocks
            .iter_mut()
            .find(|b| b.id == block_id)
            .ok_or_else(|| format!("block {} not found", block_id))?;

        if block.status != BlockStatus::Allocated {
            return Err(format!(
                "block {} is not allocated (status: {:?})",
                block_id, block.status
            ));
        }

        block.status = BlockStatus::Free;
        block.owner = None;
        block.generation += 1;
        self.deallocations += 1;
        self.current_allocated = self.current_allocated.saturating_sub(1);
        Ok(())
    }

    /// Reserve a free block for future use.
    ///
    /// Transitions a block from [`BlockStatus::Free`] to
    /// [`BlockStatus::Reserved`].
    pub fn reserve(&mut self, block_id: u64, owner: &str) -> Result<(), String> {
        let block = self
            .blocks
            .iter_mut()
            .find(|b| b.id == block_id)
            .ok_or_else(|| format!("block {} not found", block_id))?;

        if block.status != BlockStatus::Free {
            return Err(format!(
                "block {} is not free (status: {:?})",
                block_id, block.status
            ));
        }

        block.status = BlockStatus::Reserved;
        block.owner = Some(owner.to_string());
        Ok(())
    }

    /// Release a reservation, returning the block to [`BlockStatus::Free`].
    pub fn release_reservation(&mut self, block_id: u64) -> Result<(), String> {
        let block = self
            .blocks
            .iter_mut()
            .find(|b| b.id == block_id)
            .ok_or_else(|| format!("block {} not found", block_id))?;

        if block.status != BlockStatus::Reserved {
            return Err(format!(
                "block {} is not reserved (status: {:?})",
                block_id, block.status
            ));
        }

        block.status = BlockStatus::Free;
        block.owner = None;
        Ok(())
    }

    /// Look up a block by ID.
    pub fn get_block(&self, block_id: u64) -> Option<&MemoryBlock> {
        self.blocks.iter().find(|b| b.id == block_id)
    }

    /// Number of free blocks.
    pub fn free_count(&self) -> usize {
        self.blocks
            .iter()
            .filter(|b| b.status == BlockStatus::Free)
            .count()
    }

    /// Number of allocated blocks.
    pub fn allocated_count(&self) -> usize {
        self.blocks
            .iter()
            .filter(|b| b.status == BlockStatus::Allocated)
            .count()
    }

    /// Fraction of blocks currently allocated (allocated / total).
    /// Returns `0.0` if the pool is empty.
    pub fn utilization(&self) -> f64 {
        if self.blocks.is_empty() {
            0.0
        } else {
            self.allocated_count() as f64 / self.blocks.len() as f64
        }
    }

    /// Compact the block list: move all [`BlockStatus::Free`] blocks to the
    /// end.  Resets the generation of relocated free blocks to zero.
    ///
    /// Returns the number of free blocks that were relocated.
    pub fn defragment(&mut self) -> usize {
        // Partition: non-free first, free last.  Count how many free blocks
        // existed *before* the first non-free block (i.e. how many were moved).
        let before_free_positions: Vec<usize> = self
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.status == BlockStatus::Free)
            .map(|(i, _)| i)
            .collect();

        // Stable partition so allocated/reserved order is preserved.
        self.blocks.sort_by_key(|b| {
            if b.status == BlockStatus::Free {
                1u8
            } else {
                0u8
            }
        });

        // Reset generation on free blocks that were compacted.
        let mut relocated = 0usize;
        for block in &mut self.blocks {
            if block.status == BlockStatus::Free {
                block.generation = 0;
                relocated += 1;
            }
        }

        // Only count blocks that actually changed position.
        // A simpler metric: return the number of free blocks that existed
        // before (all of them were potentially moved during sort).
        let _ = before_free_positions; // consumed above for counting
        relocated
    }

    /// Remove trailing [`BlockStatus::Free`] blocks from the pool, shrinking
    /// it.  Returns the number of blocks removed.
    pub fn shrink_to_fit(&mut self) -> usize {
        let mut removed = 0usize;
        while self
            .blocks
            .last()
            .is_some_and(|b| b.status == BlockStatus::Free)
        {
            self.blocks.pop();
            removed += 1;
        }
        removed
    }

    /// Return a snapshot of pool statistics.
    pub fn block_stats(&self) -> BlockPoolStats {
        let total_blocks = self.blocks.len();
        let free_blocks = self.free_count();
        let allocated_blocks = self.allocated_count();
        let reserved_blocks = self
            .blocks
            .iter()
            .filter(|b| b.status == BlockStatus::Reserved)
            .count();

        BlockPoolStats {
            total_blocks,
            free_blocks,
            allocated_blocks,
            reserved_blocks,
            peak_allocated: self.peak_allocated,
            total_allocations: self.allocations,
            total_deallocations: self.deallocations,
            utilization: self.utilization(),
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =======================================================================
    // Part 1 — TensorMemoryPool (slab-based) tests
    // =======================================================================

    #[test]
    fn slab_new_starts_empty() {
        let pool = TensorMemoryPool::new(1024);
        assert_eq!(pool.slots.len(), 0);
        assert_eq!(pool.total_allocations, 0);
        assert_eq!(pool.total_deallocations, 0);
    }

    #[test]
    fn slab_allocate_creates_slot_when_empty() {
        let mut pool = TensorMemoryPool::new(1024);
        let id = pool.allocate(100, 1);
        assert!(id.is_some());
        assert_eq!(pool.slots.len(), 1);
    }

    #[test]
    fn slab_allocate_reuses_free_slot_of_same_class() {
        let mut pool = TensorMemoryPool::new(1024);
        let id1 = pool.allocate(100, 1);
        if let Some(id) = id1 {
            pool.deallocate(id, 2);
            let id2 = pool.allocate(200, 3);
            assert_eq!(id2, Some(id));
            assert_eq!(pool.slots.len(), 1);
        }
    }

    #[test]
    fn slab_allocate_creates_new_slot_when_no_free_of_class() {
        let mut pool = TensorMemoryPool::new(1024);
        let _id1 = pool.allocate(100, 1);
        let id2 = pool.allocate(10_000, 2);
        assert!(id2.is_some());
        assert_eq!(pool.slots.len(), 2);
    }

    #[test]
    fn slab_allocate_returns_none_at_max_slots() {
        let mut pool = TensorMemoryPool::new(2);
        let _a = pool.allocate(100, 1);
        let _b = pool.allocate(200, 2);
        let c = pool.allocate(300, 3);
        assert!(c.is_none());
    }

    #[test]
    fn slab_allocate_increments_total_allocations() {
        let mut pool = TensorMemoryPool::new(1024);
        pool.allocate(100, 1);
        pool.allocate(200, 2);
        assert_eq!(pool.total_allocations, 2);
    }

    #[test]
    fn classify_boundary_4096() {
        assert_eq!(SizeClass::classify(4096), SizeClass::Small);
    }

    #[test]
    fn classify_boundary_4097() {
        assert_eq!(SizeClass::classify(4097), SizeClass::Medium);
    }

    #[test]
    fn classify_boundary_65536() {
        assert_eq!(SizeClass::classify(65_536), SizeClass::Medium);
    }

    #[test]
    fn classify_boundary_65537() {
        assert_eq!(SizeClass::classify(65_537), SizeClass::Large);
    }

    #[test]
    fn classify_boundary_1048576() {
        assert_eq!(SizeClass::classify(1_048_576), SizeClass::Large);
    }

    #[test]
    fn classify_boundary_1048577() {
        assert_eq!(SizeClass::classify(1_048_577), SizeClass::Huge);
    }

    #[test]
    fn bucket_size_small() {
        assert_eq!(SizeClass::Small.bucket_size(), 4_096);
    }

    #[test]
    fn bucket_size_medium() {
        assert_eq!(SizeClass::Medium.bucket_size(), 65_536);
    }

    #[test]
    fn bucket_size_large() {
        assert_eq!(SizeClass::Large.bucket_size(), 1_048_576);
    }

    #[test]
    fn bucket_size_huge() {
        assert_eq!(SizeClass::Huge.bucket_size(), 16_777_216);
    }

    #[test]
    fn size_class_ordering() {
        assert!(SizeClass::Small < SizeClass::Medium);
        assert!(SizeClass::Medium < SizeClass::Large);
        assert!(SizeClass::Large < SizeClass::Huge);
    }

    #[test]
    fn slab_deallocate_marks_slot_free() {
        let mut pool = TensorMemoryPool::new(1024);
        if let Some(id) = pool.allocate(100, 1) {
            assert!(pool.slots[&id].in_use);
            pool.deallocate(id, 2);
            assert!(!pool.slots[&id].in_use);
        }
    }

    #[test]
    fn slab_deallocate_returns_false_for_unknown_id() {
        let mut pool = TensorMemoryPool::new(1024);
        assert!(!pool.deallocate(9999, 1));
    }

    #[test]
    fn slab_deallocate_returns_false_if_already_free() {
        let mut pool = TensorMemoryPool::new(1024);
        if let Some(id) = pool.allocate(100, 1) {
            pool.deallocate(id, 2);
            assert!(!pool.deallocate(id, 3));
        }
    }

    #[test]
    fn slab_deallocate_increments_total_deallocations() {
        let mut pool = TensorMemoryPool::new(1024);
        if let Some(id) = pool.allocate(100, 1) {
            pool.deallocate(id, 2);
            assert_eq!(pool.total_deallocations, 1);
        }
    }

    #[test]
    fn slab_evict_idle_removes_old_free_slots() {
        let mut pool = TensorMemoryPool::new(1024);
        if let Some(id) = pool.allocate(100, 0) {
            pool.deallocate(id, 1);
            pool.evict_idle(100, 10);
            assert!(pool.slots.is_empty());
        }
    }

    #[test]
    fn slab_evict_idle_keeps_recent_free_slots() {
        let mut pool = TensorMemoryPool::new(1024);
        if let Some(id) = pool.allocate(100, 0) {
            pool.deallocate(id, 95);
            pool.evict_idle(100, 10);
            assert_eq!(pool.slots.len(), 1);
        }
    }

    #[test]
    fn slab_evict_idle_does_not_evict_in_use_slots() {
        let mut pool = TensorMemoryPool::new(1024);
        let _id = pool.allocate(100, 0);
        pool.evict_idle(9999, 1);
        assert_eq!(pool.slots.len(), 1);
    }

    #[test]
    fn slab_slots_for_class_filters_and_sorts() {
        let mut pool = TensorMemoryPool::new(1024);
        let _s1 = pool.allocate(100, 1);
        let _m1 = pool.allocate(10_000, 2);
        let _s2 = pool.allocate(200, 3);
        let small_slots = pool.slots_for_class(SizeClass::Small);
        assert_eq!(small_slots.len(), 2);
        assert!(small_slots[0].slot_id < small_slots[1].slot_id);
        let medium_slots = pool.slots_for_class(SizeClass::Medium);
        assert_eq!(medium_slots.len(), 1);
    }

    #[test]
    fn slab_stats_total_slots_in_use_free() {
        let mut pool = TensorMemoryPool::new(1024);
        if let Some(id1) = pool.allocate(100, 1) {
            let _id2 = pool.allocate(200, 2);
            pool.deallocate(id1, 3);
            let s = pool.stats();
            assert_eq!(s.total_slots, 2);
            assert_eq!(s.in_use_slots, 1);
            assert_eq!(s.free_slots, 1);
        }
    }

    #[test]
    fn slab_stats_total_allocated_bytes() {
        let mut pool = TensorMemoryPool::new(1024);
        let _id = pool.allocate(100, 1);
        let s = pool.stats();
        assert_eq!(s.total_allocated_bytes, 4_096);
    }

    #[test]
    fn slab_stats_wasted_bytes_calculation() {
        let mut pool = TensorMemoryPool::new(1024);
        let _id = pool.allocate(100, 1);
        let s = pool.stats();
        assert_eq!(s.wasted_bytes, 4_096 - 100);
    }

    #[test]
    fn slab_stats_utilization_computed() {
        let mut pool = TensorMemoryPool::new(1024);
        if let Some(id1) = pool.allocate(100, 1) {
            let _id2 = pool.allocate(200, 2);
            pool.deallocate(id1, 3);
            let s = pool.stats();
            let util = s.utilization();
            assert!((util - 0.5).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn slab_stats_utilization_empty_pool() {
        let pool = TensorMemoryPool::new(1024);
        assert_eq!(pool.stats().utilization(), 0.0);
    }

    // =======================================================================
    // Part 2 — TensorBlockPool tests (25+)
    // =======================================================================

    // -- Construction / Pre-allocation --

    #[test]
    fn block_pool_initial_preallocation() {
        let config = PoolConfig {
            initial_blocks: 16,
            block_size: 1024,
            max_blocks: 64,
            allow_growth: true,
        };
        let pool = TensorBlockPool::new(config);
        assert_eq!(pool.blocks.len(), 16);
        assert_eq!(pool.free_count(), 16);
        assert_eq!(pool.allocated_count(), 0);
    }

    #[test]
    fn block_pool_default_config() {
        let config = PoolConfig::default();
        assert_eq!(config.initial_blocks, 64);
        assert_eq!(config.block_size, 4096);
        assert_eq!(config.max_blocks, 1024);
        assert!(config.allow_growth);
    }

    #[test]
    fn block_pool_zero_initial_blocks() {
        let config = PoolConfig {
            initial_blocks: 0,
            block_size: 256,
            max_blocks: 10,
            allow_growth: true,
        };
        let pool = TensorBlockPool::new(config);
        assert_eq!(pool.blocks.len(), 0);
        assert_eq!(pool.free_count(), 0);
    }

    // -- Allocate / Deallocate lifecycle --

    #[test]
    fn block_pool_allocate_returns_valid_id() {
        let mut pool = TensorBlockPool::new(PoolConfig::default());
        let id = pool.allocate("matmul");
        assert!(id.is_ok());
    }

    #[test]
    fn block_pool_allocate_marks_block_allocated() {
        let mut pool = TensorBlockPool::new(PoolConfig::default());
        let id = pool.allocate("conv2d").expect("should allocate");
        let block = pool.get_block(id).expect("block should exist");
        assert_eq!(block.status, BlockStatus::Allocated);
        assert_eq!(block.owner.as_deref(), Some("conv2d"));
    }

    #[test]
    fn block_pool_deallocate_frees_block() {
        let mut pool = TensorBlockPool::new(PoolConfig::default());
        let id = pool.allocate("relu").expect("should allocate");
        pool.deallocate(id).expect("should deallocate");
        let block = pool.get_block(id).expect("block should exist");
        assert_eq!(block.status, BlockStatus::Free);
        assert!(block.owner.is_none());
    }

    #[test]
    fn block_pool_deallocate_nonexistent_errors() {
        let mut pool = TensorBlockPool::new(PoolConfig::default());
        let result = pool.deallocate(99999);
        assert!(result.is_err());
    }

    #[test]
    fn block_pool_deallocate_free_block_errors() {
        let config = PoolConfig {
            initial_blocks: 4,
            block_size: 256,
            max_blocks: 8,
            allow_growth: false,
        };
        let mut pool = TensorBlockPool::new(config);
        // Block 0 is free (pre-allocated but not allocated).
        let result = pool.deallocate(0);
        assert!(result.is_err());
    }

    #[test]
    fn block_pool_allocate_deallocate_cycle() {
        let config = PoolConfig {
            initial_blocks: 2,
            block_size: 512,
            max_blocks: 2,
            allow_growth: false,
        };
        let mut pool = TensorBlockPool::new(config);

        let id1 = pool.allocate("op_a").expect("first alloc");
        let id2 = pool.allocate("op_b").expect("second alloc");
        assert!(pool.allocate("op_c").is_err()); // pool full

        pool.deallocate(id1).expect("dealloc id1");
        let id3 = pool.allocate("op_c").expect("reuse freed block");
        assert_eq!(id3, id1); // reuses the freed block
        pool.deallocate(id2).expect("dealloc id2");
        pool.deallocate(id3).expect("dealloc id3");

        assert_eq!(pool.free_count(), 2);
        assert_eq!(pool.allocated_count(), 0);
    }

    // -- Generation tracking --

    #[test]
    fn block_pool_generation_increments_on_reuse() {
        let mut pool = TensorBlockPool::new(PoolConfig::default());
        let id = pool.allocate("gen_test").expect("alloc");
        assert_eq!(pool.get_block(id).expect("exists").generation, 0);

        pool.deallocate(id).expect("dealloc");
        assert_eq!(pool.get_block(id).expect("exists").generation, 1);

        // Allocate again (same block reused), then deallocate.
        let id2 = pool.allocate("gen_test_2").expect("realloc");
        assert_eq!(id2, id);
        pool.deallocate(id2).expect("dealloc again");
        assert_eq!(pool.get_block(id).expect("exists").generation, 2);
    }

    // -- Reserve / Release --

    #[test]
    fn block_pool_reserve_free_block() {
        let config = PoolConfig {
            initial_blocks: 4,
            block_size: 256,
            max_blocks: 4,
            allow_growth: false,
        };
        let mut pool = TensorBlockPool::new(config);

        pool.reserve(0, "future_op").expect("should reserve");
        let block = pool.get_block(0).expect("exists");
        assert_eq!(block.status, BlockStatus::Reserved);
        assert_eq!(block.owner.as_deref(), Some("future_op"));
    }

    #[test]
    fn block_pool_reserve_non_free_errors() {
        let mut pool = TensorBlockPool::new(PoolConfig::default());
        let id = pool.allocate("busy").expect("alloc");
        let result = pool.reserve(id, "should_fail");
        assert!(result.is_err());
    }

    #[test]
    fn block_pool_release_reservation() {
        let config = PoolConfig {
            initial_blocks: 4,
            block_size: 256,
            max_blocks: 4,
            allow_growth: false,
        };
        let mut pool = TensorBlockPool::new(config);

        pool.reserve(1, "temp").expect("reserve");
        pool.release_reservation(1).expect("release");
        let block = pool.get_block(1).expect("exists");
        assert_eq!(block.status, BlockStatus::Free);
        assert!(block.owner.is_none());
    }

    #[test]
    fn block_pool_release_non_reserved_errors() {
        let mut pool = TensorBlockPool::new(PoolConfig::default());
        let id = pool.allocate("op").expect("alloc");
        let result = pool.release_reservation(id);
        assert!(result.is_err());
    }

    #[test]
    fn block_pool_reserved_blocks_not_allocatable() {
        let config = PoolConfig {
            initial_blocks: 1,
            block_size: 256,
            max_blocks: 1,
            allow_growth: false,
        };
        let mut pool = TensorBlockPool::new(config);
        pool.reserve(0, "held").expect("reserve");
        // The only block is reserved, so allocation should fail.
        let result = pool.allocate("want_block");
        assert!(result.is_err());
    }

    // -- Pool growth --

    #[test]
    fn block_pool_growth_when_allowed() {
        let config = PoolConfig {
            initial_blocks: 2,
            block_size: 128,
            max_blocks: 4,
            allow_growth: true,
        };
        let mut pool = TensorBlockPool::new(config);

        let _a = pool.allocate("a").expect("alloc a");
        let _b = pool.allocate("b").expect("alloc b");
        // Both initial blocks used; pool should grow.
        let c = pool.allocate("c");
        assert!(c.is_ok());
        assert_eq!(pool.blocks.len(), 3);
    }

    #[test]
    fn block_pool_growth_stops_at_max_blocks() {
        let config = PoolConfig {
            initial_blocks: 1,
            block_size: 128,
            max_blocks: 2,
            allow_growth: true,
        };
        let mut pool = TensorBlockPool::new(config);

        let _a = pool.allocate("a").expect("alloc a");
        let _b = pool.allocate("b").expect("alloc b (growth)");
        let c = pool.allocate("c");
        assert!(c.is_err());
    }

    #[test]
    fn block_pool_no_growth_when_disabled() {
        let config = PoolConfig {
            initial_blocks: 1,
            block_size: 128,
            max_blocks: 100,
            allow_growth: false,
        };
        let mut pool = TensorBlockPool::new(config);

        let _a = pool.allocate("a").expect("alloc a");
        let b = pool.allocate("b");
        assert!(b.is_err());
    }

    // -- Utilization --

    #[test]
    fn block_pool_utilization_calculation() {
        let config = PoolConfig {
            initial_blocks: 4,
            block_size: 256,
            max_blocks: 4,
            allow_growth: false,
        };
        let mut pool = TensorBlockPool::new(config);

        let _a = pool.allocate("a").expect("alloc");
        let _b = pool.allocate("b").expect("alloc");
        // 2 allocated out of 4.
        let u = pool.utilization();
        assert!((u - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn block_pool_utilization_empty() {
        let config = PoolConfig {
            initial_blocks: 0,
            block_size: 256,
            max_blocks: 10,
            allow_growth: true,
        };
        let pool = TensorBlockPool::new(config);
        assert!((pool.utilization()).abs() < f64::EPSILON);
    }

    // -- Peak tracking --

    #[test]
    fn block_pool_peak_allocated_tracking() {
        let mut pool = TensorBlockPool::new(PoolConfig::default());

        let a = pool.allocate("a").expect("alloc");
        let b = pool.allocate("b").expect("alloc");
        let _c = pool.allocate("c").expect("alloc");
        // peak = 3
        pool.deallocate(a).expect("dealloc");
        pool.deallocate(b).expect("dealloc");
        // current = 1, but peak should still be 3
        assert_eq!(pool.peak_allocated, 3);
    }

    // -- Free / Allocated counts --

    #[test]
    fn block_pool_free_allocated_counts() {
        let config = PoolConfig {
            initial_blocks: 5,
            block_size: 256,
            max_blocks: 5,
            allow_growth: false,
        };
        let mut pool = TensorBlockPool::new(config);

        let a = pool.allocate("a").expect("alloc");
        let _b = pool.allocate("b").expect("alloc");
        pool.reserve(2, "r").expect("reserve");

        assert_eq!(pool.free_count(), 2); // blocks 3, 4
        assert_eq!(pool.allocated_count(), 2); // a, b
        pool.deallocate(a).expect("dealloc");
        assert_eq!(pool.free_count(), 3);
        assert_eq!(pool.allocated_count(), 1);
    }

    // -- Stats --

    #[test]
    fn block_pool_stats_accuracy() {
        let config = PoolConfig {
            initial_blocks: 8,
            block_size: 512,
            max_blocks: 8,
            allow_growth: false,
        };
        let mut pool = TensorBlockPool::new(config);

        let a = pool.allocate("a").expect("alloc");
        let _b = pool.allocate("b").expect("alloc");
        let _c = pool.allocate("c").expect("alloc");
        pool.reserve(3, "reserved_op").expect("reserve");
        pool.deallocate(a).expect("dealloc");

        let stats = pool.block_stats();
        assert_eq!(stats.total_blocks, 8);
        assert_eq!(stats.free_blocks, 5); // 4 untouched + 1 deallocated
        assert_eq!(stats.allocated_blocks, 2);
        assert_eq!(stats.reserved_blocks, 1);
        assert_eq!(stats.peak_allocated, 3);
        assert_eq!(stats.total_allocations, 3);
        assert_eq!(stats.total_deallocations, 1);
        // utilization = 2/8 = 0.25
        assert!((stats.utilization - 0.25).abs() < f64::EPSILON);
    }

    // -- Defragment --

    #[test]
    fn block_pool_defragment_moves_free_to_end() {
        let config = PoolConfig {
            initial_blocks: 4,
            block_size: 256,
            max_blocks: 4,
            allow_growth: false,
        };
        let mut pool = TensorBlockPool::new(config);

        // Allocate all, then free blocks 0 and 2 to create gaps.
        let id0 = pool.allocate("a").expect("alloc");
        let _id1 = pool.allocate("b").expect("alloc");
        let id2 = pool.allocate("c").expect("alloc");
        let _id3 = pool.allocate("d").expect("alloc");
        pool.deallocate(id0).expect("dealloc");
        pool.deallocate(id2).expect("dealloc");

        let relocated = pool.defragment();
        assert!(relocated > 0);

        // After defragment, allocated blocks come first.
        let statuses: Vec<BlockStatus> = pool.blocks.iter().map(|b| b.status).collect();
        let first_free = statuses.iter().position(|s| *s == BlockStatus::Free);
        let last_alloc = statuses.iter().rposition(|s| *s == BlockStatus::Allocated);
        if let (Some(ff), Some(la)) = (first_free, last_alloc) {
            assert!(la < ff, "allocated blocks should precede free blocks");
        }
    }

    #[test]
    fn block_pool_defragment_resets_generation() {
        let config = PoolConfig {
            initial_blocks: 2,
            block_size: 256,
            max_blocks: 2,
            allow_growth: false,
        };
        let mut pool = TensorBlockPool::new(config);

        let id = pool.allocate("x").expect("alloc");
        pool.deallocate(id).expect("dealloc");
        // generation is now 1
        assert_eq!(pool.get_block(id).expect("exists").generation, 1);

        pool.defragment();
        // After defragment, free block generations are reset to 0.
        for block in &pool.blocks {
            if block.status == BlockStatus::Free {
                assert_eq!(block.generation, 0);
            }
        }
    }

    // -- Shrink to fit --

    #[test]
    fn block_pool_shrink_to_fit_removes_trailing_free() {
        let config = PoolConfig {
            initial_blocks: 6,
            block_size: 256,
            max_blocks: 6,
            allow_growth: false,
        };
        let mut pool = TensorBlockPool::new(config);

        // Allocate first 3, leave last 3 free.
        let _a = pool.allocate("a").expect("alloc");
        let _b = pool.allocate("b").expect("alloc");
        let _c = pool.allocate("c").expect("alloc");

        // Defragment so free blocks are at the end.
        pool.defragment();

        let removed = pool.shrink_to_fit();
        assert_eq!(removed, 3);
        assert_eq!(pool.blocks.len(), 3);
    }

    #[test]
    fn block_pool_shrink_to_fit_no_trailing_free() {
        let config = PoolConfig {
            initial_blocks: 2,
            block_size: 256,
            max_blocks: 2,
            allow_growth: false,
        };
        let mut pool = TensorBlockPool::new(config);

        let _a = pool.allocate("a").expect("alloc");
        let _b = pool.allocate("b").expect("alloc");
        let removed = pool.shrink_to_fit();
        assert_eq!(removed, 0);
    }

    // -- get_block --

    #[test]
    fn block_pool_get_block_existing() {
        let pool = TensorBlockPool::new(PoolConfig::default());
        let block = pool.get_block(0);
        assert!(block.is_some());
        assert_eq!(block.expect("exists").id, 0);
    }

    #[test]
    fn block_pool_get_block_nonexistent() {
        let pool = TensorBlockPool::new(PoolConfig::default());
        assert!(pool.get_block(99999).is_none());
    }

    // -- Block size in pre-allocated blocks --

    #[test]
    fn block_pool_blocks_have_correct_size() {
        let config = PoolConfig {
            initial_blocks: 3,
            block_size: 2048,
            max_blocks: 10,
            allow_growth: true,
        };
        let pool = TensorBlockPool::new(config);
        for block in &pool.blocks {
            assert_eq!(block.size, 2048);
        }
    }

    // -- Grown blocks also get correct size --

    #[test]
    fn block_pool_grown_block_has_correct_size() {
        let config = PoolConfig {
            initial_blocks: 0,
            block_size: 8192,
            max_blocks: 5,
            allow_growth: true,
        };
        let mut pool = TensorBlockPool::new(config);
        let id = pool.allocate("grown").expect("alloc via growth");
        let block = pool.get_block(id).expect("exists");
        assert_eq!(block.size, 8192);
    }
}
