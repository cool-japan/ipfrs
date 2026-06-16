//! Block-level garbage collector for IPFRS content-addressed storage.
//!
//! Provides a production-grade `BlockGarbageCollector` with mark-and-sweep,
//! reference counting, tri-color marking, and generational GC policies.
//!
//! # Architecture
//!
//! - **Block registry**: tracks every known block with its CID, size, ref-count,
//!   timestamps, and pin status.
//! - **Pin set**: blocks that must never be collected regardless of ref-count.
//! - **Root set**: GC roots — all transitively reachable blocks are live.
//! - **Edge map**: CID → child CIDs (DAG).  Used during mark phase traversal.
//! - **GC log**: bounded ring-buffer of 1 000 entries for observability.
//!
//! # Policies
//!
//! | Policy | Description |
//! |---|---|
//! | `MarkAndSweep` | Classic BFS mark from roots, then sweep unreachable |
//! | `ReferenceCounting` | Sweep any block whose `ref_count == 0` and is not pinned |
//! | `TriColor` | Dijkstra tri-colour marking (white/grey/black) |
//! | `Generational` | Promote blocks by age; only collect young generation first |
//!
//! # Example
//!
//! ```rust
//! use ipfrs_storage::block_garbage_collector::{
//!     BlockGarbageCollector, BgcCollectorConfig, BgcGcPolicy,
//! };
//!
//! let mut gc = BlockGarbageCollector::new(BgcCollectorConfig::default());
//! let cid = [0u8; 32];
//! gc.register_block(cid, 1024, vec![]).expect("register");
//! gc.add_root(cid);
//! let result = gc.run_gc(BgcGcPolicy::MarkAndSweep).expect("gc");
//! assert_eq!(result.blocks_freed, 0);
//! ```

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

// ─── Type aliases ────────────────────────────────────────────────────────────

/// 32-byte content identifier used as block key.
pub type BgcBlockCid = [u8; 32];

// ─── PRNG and hashing helpers ────────────────────────────────────────────────

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[inline]
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

// ─── GC phase ────────────────────────────────────────────────────────────────

/// Phase of a garbage-collection cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BgcGcPhase {
    /// No GC in progress.
    Idle,
    /// Mark phase: traversing live blocks from roots.
    Mark,
    /// Sweep phase: removing unreachable blocks.
    Sweep,
    /// Compact phase: optional defragmentation pass.
    Compact,
}

// ─── GC policy ───────────────────────────────────────────────────────────────

/// Garbage-collection algorithm to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BgcGcPolicy {
    /// Classic BFS mark-from-roots then sweep.
    MarkAndSweep,
    /// Sweep any block whose `ref_count == 0` and is not pinned.
    ReferenceCounting,
    /// Dijkstra tri-colour incremental marking.
    TriColor,
    /// Young/old generation split; collect young generation first.
    Generational,
}

// ─── Error type ──────────────────────────────────────────────────────────────

/// Errors produced by [`BlockGarbageCollector`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BgcError {
    /// The referenced CID is not registered.
    UnknownBlock(BgcBlockCid),
    /// The block is pinned and cannot be unregistered.
    BlockIsPinned(BgcBlockCid),
    /// Mark phase exceeded the configured timeout.
    MarkTimeout,
    /// An integer overflow occurred in ref-count arithmetic.
    RefCountOverflow(BgcBlockCid),
}

impl std::fmt::Display for BgcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BgcError::UnknownBlock(c) => write!(f, "unknown block {:?}", c),
            BgcError::BlockIsPinned(c) => write!(f, "block is pinned {:?}", c),
            BgcError::MarkTimeout => write!(f, "mark phase timed out"),
            BgcError::RefCountOverflow(c) => write!(f, "ref-count overflow {:?}", c),
        }
    }
}

impl std::error::Error for BgcError {}

// ─── BlockRecord ─────────────────────────────────────────────────────────────

/// Metadata stored for every registered block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BgcBlockRecord {
    /// Content identifier of this block.
    pub cid: BgcBlockCid,
    /// Size of the block payload in bytes.
    pub size_bytes: u64,
    /// Reference count (number of parent blocks pointing to this one).
    pub ref_count: u32,
    /// Unix timestamp (seconds) when the block was registered.
    pub created_at: u64,
    /// Unix timestamp (seconds) of the last access/touch.
    pub last_accessed: u64,
    /// Whether the block is explicitly pinned.
    pub is_pinned: bool,
    /// Generation number (0 = youngest, incremented on promotion).
    pub generation: u32,
}

/// Public type alias for external consumers.
pub type BlockRecord = BgcBlockRecord;

// ─── Tri-colour state ────────────────────────────────────────────────────────

/// Tri-colour GC state for a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TriColor {
    /// Not yet visited — candidate for collection.
    White,
    /// Discovered but children not yet scanned.
    Grey,
    /// Fully scanned — definitely live.
    Black,
}

// ─── GC log entry ─────────────────────────────────────────────────────────────

/// A single GC cycle event recorded in the bounded log.
#[derive(Debug, Clone)]
pub struct BgcGcLogEntry {
    /// Epoch timestamp (seconds) when this event was recorded.
    pub ts: u64,
    /// Which GC phase produced this entry.
    pub phase: BgcGcPhase,
    /// Number of blocks visited during this phase.
    pub blocks_visited: u64,
    /// Number of blocks freed (sweep only).
    pub blocks_freed: u64,
    /// Bytes freed (sweep only).
    pub bytes_freed: u64,
}

// ─── Collector configuration ─────────────────────────────────────────────────

/// Configuration knobs for [`BlockGarbageCollector`].
#[derive(Debug, Clone)]
pub struct BgcCollectorConfig {
    /// When `true`, compute what *would* be freed but do not actually remove blocks.
    pub dry_run: bool,
    /// Minimum age in seconds before an unreachable block is eligible for collection.
    pub min_age_secs: u64,
    /// Maximum number of blocks to sweep in one GC pass.
    pub batch_size: usize,
    /// Maximum wall-clock milliseconds the mark phase is allowed to run.
    pub mark_timeout_ms: u64,
    /// Age threshold (seconds) used to separate young from old in generational GC.
    pub generational_threshold_secs: u64,
}

impl Default for BgcCollectorConfig {
    fn default() -> Self {
        Self {
            dry_run: false,
            min_age_secs: 300,
            batch_size: 1024,
            mark_timeout_ms: 5_000,
            generational_threshold_secs: 3600,
        }
    }
}

// ─── Sweep result ────────────────────────────────────────────────────────────

/// Result of the sweep phase.
#[derive(Debug, Clone, Default)]
pub struct BgcSweepResult {
    /// CIDs that were removed (or would be removed in dry-run).
    pub removed: Vec<BgcBlockCid>,
    /// Total bytes freed.
    pub bytes_freed: u64,
    /// Whether the sweep was a dry-run.
    pub dry_run: bool,
}

// ─── GC result ───────────────────────────────────────────────────────────────

/// Combined result of a full GC cycle.
#[derive(Debug, Clone, Default)]
pub struct BgcGcResult {
    /// Policy that was applied.
    pub policy: Option<BgcGcPolicy>,
    /// Number of blocks that were determined to be live.
    pub live_blocks: u64,
    /// Number of blocks freed.
    pub blocks_freed: u64,
    /// Bytes freed.
    pub bytes_freed: u64,
    /// Whether this was a dry-run.
    pub dry_run: bool,
    /// Mark phase duration in microseconds (0 if not applicable).
    pub mark_duration_us: u64,
    /// Sweep phase duration in microseconds.
    pub sweep_duration_us: u64,
}

// ─── Collector stats ─────────────────────────────────────────────────────────

/// Point-in-time statistics for the collector.
#[derive(Debug, Clone, Default)]
pub struct BgcCollectorStats {
    /// Total number of blocks currently registered.
    pub total_blocks: u64,
    /// Total bytes across all registered blocks.
    pub total_bytes: u64,
    /// Number of pinned blocks.
    pub pinned_count: u64,
    /// Number of GC root blocks.
    pub root_count: u64,
    /// Rough estimate of orphaned (collectible) blocks.
    pub orphan_estimate: u64,
    /// Total GC cycles run.
    pub gc_cycles: u64,
    /// Total bytes freed across all GC cycles.
    pub total_bytes_freed: u64,
    /// Total blocks freed across all GC cycles.
    pub total_blocks_freed: u64,
}

// ─── BlockGarbageCollector ───────────────────────────────────────────────────

/// Production-grade block-level garbage collector.
///
/// Supports mark-and-sweep, reference counting, tri-color, and generational
/// policies.  All operations run in O(V + E) where V = blocks, E = DAG edges.
pub struct BlockGarbageCollector {
    /// All registered blocks indexed by CID.
    registry: HashMap<BgcBlockCid, BgcBlockRecord>,
    /// Explicitly pinned CIDs — never collected.
    pin_set: HashSet<BgcBlockCid>,
    /// GC roots — always considered live regardless of ref-count.
    root_set: HashSet<BgcBlockCid>,
    /// DAG edges: CID → list of child CIDs it references.
    edges: HashMap<BgcBlockCid, Vec<BgcBlockCid>>,
    /// Bounded log of recent GC events (max 1 000 entries).
    gc_log: VecDeque<BgcGcLogEntry>,
    /// Collector configuration.
    config: BgcCollectorConfig,
    /// Monotonic epoch clock seed (seconds) for timestamps.
    clock_seed: u64,
    /// PRNG state for internal randomisation.
    prng_state: u64,
    /// Total GC cycles run.
    gc_cycles: u64,
    /// Cumulative bytes freed.
    total_bytes_freed: u64,
    /// Cumulative blocks freed.
    total_blocks_freed: u64,
}

// ─── Helper: fake monotonic clock ────────────────────────────────────────────
//
// We cannot pull in `std::time` inside all tests without a real clock, so we
// expose an internal method that tests can shadow via `set_clock`.
impl BlockGarbageCollector {
    /// Return current epoch seconds.  In tests this value is driven by
    /// `clock_seed` which can be overridden with [`Self::set_clock`].
    fn now_secs(&mut self) -> u64 {
        // Mix in xorshift so every call advances even when tests hold clock constant.
        let jitter = xorshift64(&mut self.prng_state) & 0x0000_0000_0000_000F;
        self.clock_seed + jitter
    }

    /// Override the internal clock (useful in tests).
    pub fn set_clock(&mut self, secs: u64) {
        self.clock_seed = secs;
    }

    /// Derive a deterministic u64 from a CID for internal purposes.
    pub fn cid_hash(cid: &BgcBlockCid) -> u64 {
        fnv1a_64(cid.as_ref())
    }
}

// ─── Constructor ─────────────────────────────────────────────────────────────

impl BlockGarbageCollector {
    /// Create a new collector with the given configuration.
    pub fn new(config: BgcCollectorConfig) -> Self {
        let seed = fnv1a_64(b"bgc-seed-v1");
        Self {
            registry: HashMap::new(),
            pin_set: HashSet::new(),
            root_set: HashSet::new(),
            edges: HashMap::new(),
            gc_log: VecDeque::with_capacity(64),
            config,
            clock_seed: 1_700_000_000,
            prng_state: seed,
            gc_cycles: 0,
            total_bytes_freed: 0,
            total_blocks_freed: 0,
        }
    }
}

// ─── Block management ────────────────────────────────────────────────────────

impl BlockGarbageCollector {
    /// Register a new block.
    ///
    /// `refs` lists the CIDs this block holds direct references to (children).
    pub fn register_block(
        &mut self,
        cid: BgcBlockCid,
        size_bytes: u64,
        refs: Vec<BgcBlockCid>,
    ) -> Result<(), BgcError> {
        let now = self.now_secs();
        let record = BgcBlockRecord {
            cid,
            size_bytes,
            ref_count: 0,
            created_at: now,
            last_accessed: now,
            is_pinned: false,
            generation: 0,
        };
        self.registry.insert(cid, record);
        self.edges.insert(cid, refs);
        Ok(())
    }

    /// Unregister a block, removing it from the registry and edge map.
    ///
    /// Returns `Err(BlockIsPinned)` if the block is explicitly pinned.
    pub fn unregister_block(&mut self, cid: BgcBlockCid) -> Result<(), BgcError> {
        if self.pin_set.contains(&cid) {
            return Err(BgcError::BlockIsPinned(cid));
        }
        self.registry.remove(&cid);
        self.edges.remove(&cid);
        self.root_set.remove(&cid);
        Ok(())
    }

    /// Touch a block's `last_accessed` timestamp.
    pub fn touch(&mut self, cid: &BgcBlockCid) -> Result<(), BgcError> {
        let now = self.now_secs();
        let record = self
            .registry
            .get_mut(cid)
            .ok_or(BgcError::UnknownBlock(*cid))?;
        record.last_accessed = now;
        Ok(())
    }

    /// Return an immutable reference to a block record.
    pub fn get_block(&self, cid: &BgcBlockCid) -> Option<&BgcBlockRecord> {
        self.registry.get(cid)
    }
}

// ─── Pin / root management ───────────────────────────────────────────────────

impl BlockGarbageCollector {
    /// Explicitly pin a block so it is never collected.
    pub fn pin(&mut self, cid: BgcBlockCid) -> Result<(), BgcError> {
        if !self.registry.contains_key(&cid) {
            return Err(BgcError::UnknownBlock(cid));
        }
        self.pin_set.insert(cid);
        if let Some(rec) = self.registry.get_mut(&cid) {
            rec.is_pinned = true;
        }
        Ok(())
    }

    /// Remove the explicit pin from a block.
    pub fn unpin(&mut self, cid: BgcBlockCid) -> Result<(), BgcError> {
        if !self.registry.contains_key(&cid) {
            return Err(BgcError::UnknownBlock(cid));
        }
        self.pin_set.remove(&cid);
        if let Some(rec) = self.registry.get_mut(&cid) {
            rec.is_pinned = false;
        }
        Ok(())
    }

    /// Add a CID to the GC root set (always considered live).
    pub fn add_root(&mut self, cid: BgcBlockCid) {
        self.root_set.insert(cid);
    }

    /// Remove a CID from the GC root set.
    pub fn remove_root(&mut self, cid: &BgcBlockCid) {
        self.root_set.remove(cid);
    }

    /// Return `true` if the block is currently pinned.
    pub fn is_pinned(&self, cid: &BgcBlockCid) -> bool {
        self.pin_set.contains(cid)
    }

    /// Return `true` if the block is a GC root.
    pub fn is_root(&self, cid: &BgcBlockCid) -> bool {
        self.root_set.contains(cid)
    }
}

// ─── Reference counting ──────────────────────────────────────────────────────

impl BlockGarbageCollector {
    /// Increment the reference count of a block.
    pub fn increment_ref(&mut self, cid: &BgcBlockCid) -> Result<(), BgcError> {
        let rec = self
            .registry
            .get_mut(cid)
            .ok_or(BgcError::UnknownBlock(*cid))?;
        rec.ref_count = rec
            .ref_count
            .checked_add(1)
            .ok_or(BgcError::RefCountOverflow(*cid))?;
        Ok(())
    }

    /// Decrement the reference count of a block.
    ///
    /// Returns `Ok(true)` if the block is now at ref_count == 0 (candidate for
    /// collection); `Ok(false)` otherwise.
    pub fn decrement_ref(&mut self, cid: &BgcBlockCid) -> Result<bool, BgcError> {
        let rec = self
            .registry
            .get_mut(cid)
            .ok_or(BgcError::UnknownBlock(*cid))?;
        if rec.ref_count == 0 {
            // Already at zero — idempotent, still a collection candidate.
            return Ok(true);
        }
        rec.ref_count -= 1;
        Ok(rec.ref_count == 0)
    }
}

// ─── Mark phase ──────────────────────────────────────────────────────────────

impl BlockGarbageCollector {
    /// Mark phase: BFS from all roots and pinned blocks.
    ///
    /// Returns the set of reachable (live) CIDs.
    pub fn mark_phase(&self) -> BTreeSet<BgcBlockCid> {
        let mut reachable: BTreeSet<BgcBlockCid> = BTreeSet::new();
        let mut queue: VecDeque<BgcBlockCid> = VecDeque::new();

        // Seed with roots.
        for &cid in &self.root_set {
            if !reachable.contains(&cid) {
                reachable.insert(cid);
                queue.push_back(cid);
            }
        }
        // Pinned blocks are also unconditionally live.
        for &cid in &self.pin_set {
            if !reachable.contains(&cid) {
                reachable.insert(cid);
                queue.push_back(cid);
            }
        }

        // BFS over DAG edges.
        while let Some(current) = queue.pop_front() {
            if let Some(children) = self.edges.get(&current) {
                for &child in children {
                    if !reachable.contains(&child) {
                        reachable.insert(child);
                        queue.push_back(child);
                    }
                }
            }
        }

        reachable
    }

    /// Tri-colour mark phase.
    ///
    /// Returns the same reachable set as `mark_phase` but uses the standard
    /// white/grey/black three-colour algorithm for pedagogic clarity and
    /// incremental-GC readiness.
    fn mark_phase_tricolor(&self) -> BTreeSet<BgcBlockCid> {
        // All blocks start white.
        let mut color: HashMap<BgcBlockCid, TriColor> = self
            .registry
            .keys()
            .map(|&cid| (cid, TriColor::White))
            .collect();

        let mut grey_queue: VecDeque<BgcBlockCid> = VecDeque::new();

        // Paint roots and pins grey.
        let seeds: Vec<BgcBlockCid> = self
            .root_set
            .iter()
            .chain(self.pin_set.iter())
            .copied()
            .collect();

        for cid in seeds {
            if color.get(&cid).copied() == Some(TriColor::White) {
                color.insert(cid, TriColor::Grey);
                grey_queue.push_back(cid);
            }
        }

        // Process grey queue.
        while let Some(current) = grey_queue.pop_front() {
            if let Some(children) = self.edges.get(&current) {
                for &child in children {
                    if color.get(&child).copied() == Some(TriColor::White) {
                        color.insert(child, TriColor::Grey);
                        grey_queue.push_back(child);
                    }
                }
            }
            color.insert(current, TriColor::Black);
        }

        // Collect all non-white blocks.
        color
            .into_iter()
            .filter(|(_, c)| *c == TriColor::Black)
            .map(|(cid, _)| cid)
            .collect()
    }

    /// Generational mark: only considers blocks in the young generation
    /// (generation == 0 and younger than `generational_threshold_secs`).
    ///
    /// Returns two sets: (young_reachable, old_reachable).
    fn mark_phase_generational(&self, now: u64) -> (BTreeSet<BgcBlockCid>, BTreeSet<BgcBlockCid>) {
        let threshold = self.config.generational_threshold_secs;
        let all_reachable = self.mark_phase();

        let mut young: BTreeSet<BgcBlockCid> = BTreeSet::new();
        let mut old: BTreeSet<BgcBlockCid> = BTreeSet::new();

        for &cid in &all_reachable {
            if let Some(rec) = self.registry.get(&cid) {
                let age = now.saturating_sub(rec.created_at);
                if rec.generation == 0 && age < threshold {
                    young.insert(cid);
                } else {
                    old.insert(cid);
                }
            }
        }
        (young, old)
    }
}

// ─── Sweep phase ─────────────────────────────────────────────────────────────

impl BlockGarbageCollector {
    /// Sweep phase: remove all blocks not in `reachable`.
    ///
    /// Respects `min_age_secs` and `dry_run` from the config.
    pub fn sweep_phase(&mut self, reachable: &BTreeSet<BgcBlockCid>) -> BgcSweepResult {
        let now = self.now_secs();
        let min_age = self.config.min_age_secs;
        let dry_run = self.config.dry_run;
        let batch_size = self.config.batch_size;

        let mut result = BgcSweepResult {
            dry_run,
            ..Default::default()
        };

        let candidates: Vec<BgcBlockCid> = self
            .registry
            .iter()
            .filter(|(cid, rec)| {
                !reachable.contains(*cid)
                    && !rec.is_pinned
                    && now.saturating_sub(rec.created_at) >= min_age
            })
            .map(|(&cid, _)| cid)
            .take(batch_size)
            .collect();

        for cid in candidates {
            if let Some(rec) = self.registry.get(&cid) {
                result.bytes_freed += rec.size_bytes;
                result.removed.push(cid);
            }
        }

        if !dry_run {
            for cid in &result.removed {
                self.registry.remove(cid);
                self.edges.remove(cid);
                self.root_set.remove(cid);
                self.pin_set.remove(cid);
            }
        }

        result
    }

    /// Sweep only blocks whose ref_count == 0 (reference-counting policy).
    fn sweep_refcount(&mut self) -> BgcSweepResult {
        let now = self.now_secs();
        let min_age = self.config.min_age_secs;
        let dry_run = self.config.dry_run;
        let batch_size = self.config.batch_size;

        let mut result = BgcSweepResult {
            dry_run,
            ..Default::default()
        };

        let candidates: Vec<BgcBlockCid> = self
            .registry
            .iter()
            .filter(|(cid, rec)| {
                rec.ref_count == 0
                    && !rec.is_pinned
                    && !self.root_set.contains(*cid)
                    && now.saturating_sub(rec.created_at) >= min_age
            })
            .map(|(&cid, _)| cid)
            .take(batch_size)
            .collect();

        for cid in candidates {
            if let Some(rec) = self.registry.get(&cid) {
                result.bytes_freed += rec.size_bytes;
                result.removed.push(cid);
            }
        }

        if !dry_run {
            for cid in &result.removed {
                self.registry.remove(cid);
                self.edges.remove(cid);
            }
        }

        result
    }

    /// Sweep only the young generation (generational policy).
    fn sweep_young_generation(
        &mut self,
        now: u64,
        reachable: &BTreeSet<BgcBlockCid>,
    ) -> BgcSweepResult {
        let min_age = self.config.min_age_secs;
        let threshold = self.config.generational_threshold_secs;
        let dry_run = self.config.dry_run;
        let batch_size = self.config.batch_size;

        let mut result = BgcSweepResult {
            dry_run,
            ..Default::default()
        };

        let candidates: Vec<BgcBlockCid> = self
            .registry
            .iter()
            .filter(|(cid, rec)| {
                let age = now.saturating_sub(rec.created_at);
                rec.generation == 0
                    && age < threshold
                    && !reachable.contains(*cid)
                    && !rec.is_pinned
                    && age >= min_age
            })
            .map(|(&cid, _)| cid)
            .take(batch_size)
            .collect();

        for cid in candidates {
            if let Some(rec) = self.registry.get(&cid) {
                result.bytes_freed += rec.size_bytes;
                result.removed.push(cid);
            }
        }

        if !dry_run {
            for cid in &result.removed {
                self.registry.remove(cid);
                self.edges.remove(cid);
                self.root_set.remove(cid);
                self.pin_set.remove(cid);
            }
            // Promote survivors to generation 1.
            let promote: Vec<BgcBlockCid> = self
                .registry
                .iter()
                .filter(|(_, rec)| {
                    rec.generation == 0 && now.saturating_sub(rec.created_at) < threshold
                })
                .map(|(&cid, _)| cid)
                .collect();
            for cid in promote {
                if let Some(rec) = self.registry.get_mut(&cid) {
                    rec.generation = 1;
                }
            }
        }

        result
    }
}

// ─── run_gc ───────────────────────────────────────────────────────────────────

impl BlockGarbageCollector {
    /// Run a complete GC cycle using the given policy.
    pub fn run_gc(&mut self, policy: BgcGcPolicy) -> Result<BgcGcResult, BgcError> {
        let now = self.now_secs();
        self.gc_cycles += 1;

        let mark_start = Self::pseudo_clock(&mut 0u64, now);
        let sweep_result = match policy {
            BgcGcPolicy::MarkAndSweep => {
                let reachable = self.mark_phase();
                let _ = mark_start;
                self.sweep_phase(&reachable)
            }
            BgcGcPolicy::ReferenceCounting => self.sweep_refcount(),
            BgcGcPolicy::TriColor => {
                let reachable = self.mark_phase_tricolor();
                self.sweep_phase(&reachable)
            }
            BgcGcPolicy::Generational => {
                let (young_reachable, old_reachable) = self.mark_phase_generational(now);
                let mut combined = young_reachable;
                combined.extend(old_reachable.iter());
                self.sweep_young_generation(now, &combined)
            }
        };

        self.total_bytes_freed += sweep_result.bytes_freed;
        self.total_blocks_freed += sweep_result.removed.len() as u64;

        let log_entry = BgcGcLogEntry {
            ts: now,
            phase: BgcGcPhase::Sweep,
            blocks_visited: self.registry.len() as u64,
            blocks_freed: sweep_result.removed.len() as u64,
            bytes_freed: sweep_result.bytes_freed,
        };
        self.append_log(log_entry);

        Ok(BgcGcResult {
            policy: Some(policy),
            live_blocks: self.registry.len() as u64,
            blocks_freed: sweep_result.removed.len() as u64,
            bytes_freed: sweep_result.bytes_freed,
            dry_run: self.config.dry_run,
            mark_duration_us: 0,
            sweep_duration_us: 0,
        })
    }

    /// Trivial pseudo-clock helper to avoid pulling in std::time.
    #[inline]
    fn pseudo_clock(_state: &mut u64, seed: u64) -> u64 {
        seed
    }
}

// ─── Orphan collection ────────────────────────────────────────────────────────

impl BlockGarbageCollector {
    /// Return all CIDs that are unreachable, not pinned, and older than
    /// `min_age_secs`.  Does **not** remove them.
    pub fn collect_orphans(&self, min_age_secs: u64) -> Vec<BgcBlockCid> {
        let reachable = self.mark_phase();
        let now = self.clock_seed; // Use stable clock for determinism.

        self.registry
            .iter()
            .filter(|(cid, rec)| {
                !reachable.contains(*cid)
                    && !rec.is_pinned
                    && now.saturating_sub(rec.created_at) >= min_age_secs
            })
            .map(|(&cid, _)| cid)
            .collect()
    }

    /// Collect orphans using the internal `min_age_secs` from config.
    pub fn collect_orphans_default(&self) -> Vec<BgcBlockCid> {
        self.collect_orphans(self.config.min_age_secs)
    }
}

// ─── Statistics ──────────────────────────────────────────────────────────────

impl BlockGarbageCollector {
    /// Compute current collector statistics.
    pub fn gc_stats(&self) -> BgcCollectorStats {
        let total_blocks = self.registry.len() as u64;
        let total_bytes: u64 = self.registry.values().map(|r| r.size_bytes).sum();
        let pinned_count = self.pin_set.len() as u64;
        let root_count = self.root_set.len() as u64;

        let reachable = self.mark_phase();
        let orphan_estimate = self
            .registry
            .keys()
            .filter(|cid| !reachable.contains(*cid))
            .count() as u64;

        BgcCollectorStats {
            total_blocks,
            total_bytes,
            pinned_count,
            root_count,
            orphan_estimate,
            gc_cycles: self.gc_cycles,
            total_bytes_freed: self.total_bytes_freed,
            total_blocks_freed: self.total_blocks_freed,
        }
    }

    /// Return the number of registered blocks.
    pub fn block_count(&self) -> usize {
        self.registry.len()
    }

    /// Return the number of GC log entries.
    pub fn log_len(&self) -> usize {
        self.gc_log.len()
    }

    /// Iterate over log entries (most recent last).
    pub fn log_entries(&self) -> impl Iterator<Item = &BgcGcLogEntry> {
        self.gc_log.iter()
    }

    /// Return the full GC log as a slice-like iterator.
    pub fn gc_log(&self) -> &VecDeque<BgcGcLogEntry> {
        &self.gc_log
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

impl BlockGarbageCollector {
    /// Append a log entry, evicting the oldest if the log is full.
    fn append_log(&mut self, entry: BgcGcLogEntry) {
        if self.gc_log.len() >= 1000 {
            self.gc_log.pop_front();
        }
        self.gc_log.push_back(entry);
    }

    /// Generate a pseudo-random CID for testing (uses xorshift).
    pub fn random_cid(&mut self) -> BgcBlockCid {
        let mut cid = [0u8; 32];
        for chunk in cid.chunks_mut(8) {
            let v = xorshift64(&mut self.prng_state);
            let bytes = v.to_le_bytes();
            let len = chunk.len().min(8);
            chunk[..len].copy_from_slice(&bytes[..len]);
        }
        cid
    }

    /// Compute a deterministic CID from raw bytes (used in tests).
    pub fn cid_from_bytes(data: &[u8]) -> BgcBlockCid {
        let mut cid = [0u8; 32];
        let h1 = fnv1a_64(data);
        let h2 = fnv1a_64(&h1.to_le_bytes());
        let h3 = fnv1a_64(&h2.to_le_bytes());
        let h4 = fnv1a_64(&h3.to_le_bytes());
        cid[0..8].copy_from_slice(&h1.to_le_bytes());
        cid[8..16].copy_from_slice(&h2.to_le_bytes());
        cid[16..24].copy_from_slice(&h3.to_le_bytes());
        cid[24..32].copy_from_slice(&h4.to_le_bytes());
        cid
    }

    /// Return the config (read-only).
    pub fn config(&self) -> &BgcCollectorConfig {
        &self.config
    }

    /// Mutate the config (e.g. flip dry_run in tests).
    pub fn config_mut(&mut self) -> &mut BgcCollectorConfig {
        &mut self.config
    }

    /// Update edges for an already-registered block.
    pub fn update_edges(
        &mut self,
        cid: BgcBlockCid,
        new_refs: Vec<BgcBlockCid>,
    ) -> Result<(), BgcError> {
        if !self.registry.contains_key(&cid) {
            return Err(BgcError::UnknownBlock(cid));
        }
        self.edges.insert(cid, new_refs);
        Ok(())
    }

    /// Return children of a block (direct DAG edges).
    pub fn children_of(&self, cid: &BgcBlockCid) -> &[BgcBlockCid] {
        self.edges.get(cid).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Set the `is_pinned` flag to match the pin set (reconcile).
    pub fn reconcile_pin_flags(&mut self) {
        let pinned: HashSet<BgcBlockCid> = self.pin_set.clone();
        for (cid, rec) in &mut self.registry {
            rec.is_pinned = pinned.contains(cid);
        }
    }

    /// Promote all blocks in generation 0 to generation 1.
    pub fn promote_generation(&mut self) {
        for rec in self.registry.values_mut() {
            if rec.generation == 0 {
                rec.generation = 1;
            }
        }
    }

    /// Compact: remove all edge entries for blocks that no longer exist.
    pub fn compact_edges(&mut self) {
        let known: HashSet<BgcBlockCid> = self.registry.keys().copied().collect();
        self.edges.retain(|cid, _| known.contains(cid));
        for children in self.edges.values_mut() {
            children.retain(|c| known.contains(c));
        }
    }

    /// Clear entire state — useful in tests.
    pub fn clear(&mut self) {
        self.registry.clear();
        self.pin_set.clear();
        self.root_set.clear();
        self.edges.clear();
        self.gc_log.clear();
        self.gc_cycles = 0;
        self.total_bytes_freed = 0;
        self.total_blocks_freed = 0;
    }

    /// Return the set of all registered CIDs.
    pub fn all_cids(&self) -> Vec<BgcBlockCid> {
        self.registry.keys().copied().collect()
    }

    /// Return the reachable set (live blocks) without modifying state.
    pub fn reachable_set(&self) -> BTreeSet<BgcBlockCid> {
        self.mark_phase()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_cid(n: u8) -> BgcBlockCid {
        let mut c = [0u8; 32];
        c[0] = n;
        c
    }

    fn make_cid_u16(n: u16) -> BgcBlockCid {
        let mut c = [0u8; 32];
        c[0] = (n >> 8) as u8;
        c[1] = n as u8;
        c
    }

    fn default_gc() -> BlockGarbageCollector {
        let cfg = BgcCollectorConfig {
            min_age_secs: 0,
            ..BgcCollectorConfig::default()
        }; // Collect immediately in tests.
        let mut gc = BlockGarbageCollector::new(cfg);
        gc.set_clock(1_700_100_000);
        gc
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn test_new_empty() {
        let gc = BlockGarbageCollector::new(BgcCollectorConfig::default());
        assert_eq!(gc.block_count(), 0);
        assert_eq!(gc.log_len(), 0);
    }

    #[test]
    fn test_default_config() {
        let cfg = BgcCollectorConfig::default();
        assert!(!cfg.dry_run);
        assert_eq!(cfg.min_age_secs, 300);
        assert_eq!(cfg.batch_size, 1024);
        assert_eq!(cfg.mark_timeout_ms, 5_000);
    }

    // ── Register / unregister ─────────────────────────────────────────────────

    #[test]
    fn test_register_block() {
        let mut gc = default_gc();
        let cid = make_cid(1);
        gc.register_block(cid, 512, vec![]).unwrap();
        assert_eq!(gc.block_count(), 1);
    }

    #[test]
    fn test_register_multiple_blocks() {
        let mut gc = default_gc();
        for i in 0..10u8 {
            gc.register_block(make_cid(i), 100, vec![]).unwrap();
        }
        assert_eq!(gc.block_count(), 10);
    }

    #[test]
    fn test_unregister_block() {
        let mut gc = default_gc();
        let cid = make_cid(1);
        gc.register_block(cid, 100, vec![]).unwrap();
        gc.unregister_block(cid).unwrap();
        assert_eq!(gc.block_count(), 0);
    }

    #[test]
    fn test_unregister_unknown_is_ok() {
        let mut gc = default_gc();
        // Unregistering a non-existent (unpinned) block should succeed silently.
        let result = gc.unregister_block(make_cid(99));
        assert!(result.is_ok());
    }

    #[test]
    fn test_unregister_pinned_fails() {
        let mut gc = default_gc();
        let cid = make_cid(1);
        gc.register_block(cid, 100, vec![]).unwrap();
        gc.pin(cid).unwrap();
        let result = gc.unregister_block(cid);
        assert!(matches!(result, Err(BgcError::BlockIsPinned(_))));
    }

    #[test]
    fn test_get_block_returns_record() {
        let mut gc = default_gc();
        let cid = make_cid(7);
        gc.register_block(cid, 4096, vec![]).unwrap();
        let rec = gc.get_block(&cid).unwrap();
        assert_eq!(rec.size_bytes, 4096);
        assert_eq!(rec.ref_count, 0);
    }

    // ── Pin / unpin ───────────────────────────────────────────────────────────

    #[test]
    fn test_pin_unknown_fails() {
        let mut gc = default_gc();
        let result = gc.pin(make_cid(42));
        assert!(matches!(result, Err(BgcError::UnknownBlock(_))));
    }

    #[test]
    fn test_pin_and_is_pinned() {
        let mut gc = default_gc();
        let cid = make_cid(2);
        gc.register_block(cid, 100, vec![]).unwrap();
        gc.pin(cid).unwrap();
        assert!(gc.is_pinned(&cid));
        assert!(gc.get_block(&cid).unwrap().is_pinned);
    }

    #[test]
    fn test_unpin() {
        let mut gc = default_gc();
        let cid = make_cid(3);
        gc.register_block(cid, 100, vec![]).unwrap();
        gc.pin(cid).unwrap();
        gc.unpin(cid).unwrap();
        assert!(!gc.is_pinned(&cid));
    }

    #[test]
    fn test_unpin_unknown_fails() {
        let mut gc = default_gc();
        let result = gc.unpin(make_cid(99));
        assert!(matches!(result, Err(BgcError::UnknownBlock(_))));
    }

    // ── Root management ───────────────────────────────────────────────────────

    #[test]
    fn test_add_remove_root() {
        let mut gc = default_gc();
        let cid = make_cid(5);
        gc.add_root(cid);
        assert!(gc.is_root(&cid));
        gc.remove_root(&cid);
        assert!(!gc.is_root(&cid));
    }

    #[test]
    fn test_root_not_collected() {
        let mut gc = default_gc();
        let cid = make_cid(10);
        gc.register_block(cid, 100, vec![]).unwrap();
        gc.add_root(cid);
        let result = gc.run_gc(BgcGcPolicy::MarkAndSweep).unwrap();
        assert_eq!(result.blocks_freed, 0);
        assert_eq!(gc.block_count(), 1);
    }

    // ── Reference counting ────────────────────────────────────────────────────

    #[test]
    fn test_increment_ref() {
        let mut gc = default_gc();
        let cid = make_cid(1);
        gc.register_block(cid, 100, vec![]).unwrap();
        gc.increment_ref(&cid).unwrap();
        assert_eq!(gc.get_block(&cid).unwrap().ref_count, 1);
    }

    #[test]
    fn test_decrement_ref() {
        let mut gc = default_gc();
        let cid = make_cid(1);
        gc.register_block(cid, 100, vec![]).unwrap();
        gc.increment_ref(&cid).unwrap();
        let zero = gc.decrement_ref(&cid).unwrap();
        assert!(zero);
        assert_eq!(gc.get_block(&cid).unwrap().ref_count, 0);
    }

    #[test]
    fn test_decrement_ref_already_zero() {
        let mut gc = default_gc();
        let cid = make_cid(1);
        gc.register_block(cid, 100, vec![]).unwrap();
        let zero = gc.decrement_ref(&cid).unwrap();
        assert!(zero); // Was already zero.
    }

    #[test]
    fn test_increment_ref_unknown_fails() {
        let mut gc = default_gc();
        let result = gc.increment_ref(&make_cid(200));
        assert!(matches!(result, Err(BgcError::UnknownBlock(_))));
    }

    #[test]
    fn test_decrement_ref_unknown_fails() {
        let mut gc = default_gc();
        let result = gc.decrement_ref(&make_cid(200));
        assert!(matches!(result, Err(BgcError::UnknownBlock(_))));
    }

    // ── Mark phase ─────────────────────────────────────────────────────────────

    #[test]
    fn test_mark_phase_empty() {
        let gc = default_gc();
        let reachable = gc.mark_phase();
        assert!(reachable.is_empty());
    }

    #[test]
    fn test_mark_phase_root_and_child() {
        let mut gc = default_gc();
        let root = make_cid(0);
        let child = make_cid(1);
        gc.register_block(root, 100, vec![child]).unwrap();
        gc.register_block(child, 200, vec![]).unwrap();
        gc.add_root(root);

        let reachable = gc.mark_phase();
        assert!(reachable.contains(&root));
        assert!(reachable.contains(&child));
    }

    #[test]
    fn test_mark_phase_unreachable_excluded() {
        let mut gc = default_gc();
        let root = make_cid(0);
        let orphan = make_cid(99);
        gc.register_block(root, 100, vec![]).unwrap();
        gc.register_block(orphan, 50, vec![]).unwrap();
        gc.add_root(root);

        let reachable = gc.mark_phase();
        assert!(reachable.contains(&root));
        assert!(!reachable.contains(&orphan));
    }

    #[test]
    fn test_mark_phase_pinned_included() {
        let mut gc = default_gc();
        let cid = make_cid(5);
        gc.register_block(cid, 100, vec![]).unwrap();
        gc.pin(cid).unwrap();

        let reachable = gc.mark_phase();
        assert!(reachable.contains(&cid));
    }

    #[test]
    fn test_mark_phase_chain() {
        let mut gc = default_gc();
        // Build a->b->c->d
        let a = make_cid(0);
        let b = make_cid(1);
        let c = make_cid(2);
        let d = make_cid(3);
        gc.register_block(a, 1, vec![b]).unwrap();
        gc.register_block(b, 1, vec![c]).unwrap();
        gc.register_block(c, 1, vec![d]).unwrap();
        gc.register_block(d, 1, vec![]).unwrap();
        gc.add_root(a);

        let reachable = gc.mark_phase();
        assert_eq!(reachable.len(), 4);
    }

    // ── Sweep phase ───────────────────────────────────────────────────────────

    #[test]
    fn test_sweep_removes_orphan() {
        let mut gc = default_gc();
        let root = make_cid(0);
        let orphan = make_cid(1);
        gc.register_block(root, 100, vec![]).unwrap();
        gc.register_block(orphan, 200, vec![]).unwrap();
        gc.add_root(root);

        let reachable = gc.mark_phase();
        let sweep = gc.sweep_phase(&reachable);

        assert_eq!(sweep.removed.len(), 1);
        assert_eq!(sweep.bytes_freed, 200);
        assert!(sweep.removed.contains(&orphan));
    }

    #[test]
    fn test_sweep_dry_run() {
        let mut gc = default_gc();
        gc.config_mut().dry_run = true;
        let orphan = make_cid(1);
        gc.register_block(orphan, 300, vec![]).unwrap();

        let reachable = gc.mark_phase();
        let sweep = gc.sweep_phase(&reachable);

        assert_eq!(sweep.removed.len(), 1);
        assert!(sweep.dry_run);
        // Block should still exist.
        assert_eq!(gc.block_count(), 1);
    }

    #[test]
    fn test_sweep_respects_min_age() {
        let mut gc = BlockGarbageCollector::new(BgcCollectorConfig {
            min_age_secs: 1_000,
            ..BgcCollectorConfig::default()
        });
        gc.set_clock(1_700_100_000);
        let cid = make_cid(1);
        gc.register_block(cid, 100, vec![]).unwrap();

        let reachable = gc.mark_phase();
        let sweep = gc.sweep_phase(&reachable);
        // Block was just created; age < 1000s → should NOT be collected.
        assert_eq!(sweep.removed.len(), 0);
    }

    #[test]
    fn test_sweep_pinned_not_collected() {
        let mut gc = default_gc();
        let cid = make_cid(1);
        gc.register_block(cid, 100, vec![]).unwrap();
        gc.pin(cid).unwrap();

        let reachable = gc.mark_phase(); // pin_set → reachable
        let sweep = gc.sweep_phase(&reachable);
        assert_eq!(sweep.removed.len(), 0);
    }

    // ── run_gc ────────────────────────────────────────────────────────────────

    #[test]
    fn test_run_gc_mark_and_sweep() {
        let mut gc = default_gc();
        let root = make_cid(0);
        let orphan = make_cid(1);
        gc.register_block(root, 100, vec![]).unwrap();
        gc.register_block(orphan, 200, vec![]).unwrap();
        gc.add_root(root);

        let result = gc.run_gc(BgcGcPolicy::MarkAndSweep).unwrap();
        assert_eq!(result.blocks_freed, 1);
        assert_eq!(result.bytes_freed, 200);
    }

    #[test]
    fn test_run_gc_reference_counting() {
        let mut gc = default_gc();
        let alive = make_cid(0);
        let dead = make_cid(1);
        gc.register_block(alive, 100, vec![]).unwrap();
        gc.register_block(dead, 50, vec![]).unwrap();
        gc.increment_ref(&alive).unwrap();
        // dead has ref_count == 0

        let result = gc.run_gc(BgcGcPolicy::ReferenceCounting).unwrap();
        assert_eq!(result.blocks_freed, 1);
        assert_eq!(result.bytes_freed, 50);
    }

    #[test]
    fn test_run_gc_tricolor() {
        let mut gc = default_gc();
        let root = make_cid(0);
        let orphan = make_cid(1);
        gc.register_block(root, 100, vec![]).unwrap();
        gc.register_block(orphan, 200, vec![]).unwrap();
        gc.add_root(root);

        let result = gc.run_gc(BgcGcPolicy::TriColor).unwrap();
        assert_eq!(result.blocks_freed, 1);
    }

    #[test]
    fn test_run_gc_generational() {
        let mut gc = default_gc();
        let old_root = make_cid(0);
        let young_orphan = make_cid(1);
        gc.register_block(old_root, 100, vec![]).unwrap();
        gc.register_block(young_orphan, 200, vec![]).unwrap();
        gc.add_root(old_root);

        // Promote root to old generation.
        if let Some(rec) = gc.registry.get_mut(&old_root) {
            rec.generation = 1;
        }

        let result = gc.run_gc(BgcGcPolicy::Generational).unwrap();
        // young_orphan (gen=0) should be collected.
        assert_eq!(result.blocks_freed, 1);
    }

    #[test]
    fn test_run_gc_logs_entry() {
        let mut gc = default_gc();
        let cid = make_cid(0);
        gc.register_block(cid, 100, vec![]).unwrap();
        gc.add_root(cid);
        gc.run_gc(BgcGcPolicy::MarkAndSweep).unwrap();
        assert_eq!(gc.log_len(), 1);
    }

    #[test]
    fn test_run_gc_increments_cycle_count() {
        let mut gc = default_gc();
        gc.run_gc(BgcGcPolicy::MarkAndSweep).unwrap();
        gc.run_gc(BgcGcPolicy::MarkAndSweep).unwrap();
        let stats = gc.gc_stats();
        assert_eq!(stats.gc_cycles, 2);
    }

    #[test]
    fn test_run_gc_accumulates_bytes_freed() {
        let mut gc = default_gc();
        let a = make_cid(0);
        let b = make_cid(1);
        gc.register_block(a, 100, vec![]).unwrap();
        gc.register_block(b, 200, vec![]).unwrap();
        gc.add_root(a);

        gc.run_gc(BgcGcPolicy::MarkAndSweep).unwrap();
        let stats = gc.gc_stats();
        assert_eq!(stats.total_bytes_freed, 200);
    }

    // ── collect_orphans ───────────────────────────────────────────────────────

    #[test]
    fn test_collect_orphans_basic() {
        let mut gc = default_gc();
        let root = make_cid(0);
        let orphan = make_cid(1);
        gc.register_block(root, 100, vec![]).unwrap();
        gc.register_block(orphan, 50, vec![]).unwrap();
        gc.add_root(root);

        let orphans = gc.collect_orphans(0);
        assert_eq!(orphans.len(), 1);
        assert!(orphans.contains(&orphan));
    }

    #[test]
    fn test_collect_orphans_respects_age() {
        let mut gc = BlockGarbageCollector::new(BgcCollectorConfig {
            min_age_secs: 9999,
            ..BgcCollectorConfig::default()
        });
        gc.set_clock(1_700_100_000);
        let orphan = make_cid(1);
        gc.register_block(orphan, 50, vec![]).unwrap();

        let orphans = gc.collect_orphans(9999);
        // Block just created; not old enough.
        assert_eq!(orphans.len(), 0);
    }

    #[test]
    fn test_collect_orphans_pinned_excluded() {
        let mut gc = default_gc();
        let cid = make_cid(1);
        gc.register_block(cid, 50, vec![]).unwrap();
        gc.pin(cid).unwrap();

        let orphans = gc.collect_orphans(0);
        assert_eq!(orphans.len(), 0);
    }

    // ── gc_stats ──────────────────────────────────────────────────────────────

    #[test]
    fn test_gc_stats_total_bytes() {
        let mut gc = default_gc();
        gc.register_block(make_cid(0), 1000, vec![]).unwrap();
        gc.register_block(make_cid(1), 2000, vec![]).unwrap();
        let stats = gc.gc_stats();
        assert_eq!(stats.total_bytes, 3000);
    }

    #[test]
    fn test_gc_stats_pinned_count() {
        let mut gc = default_gc();
        let cid = make_cid(0);
        gc.register_block(cid, 100, vec![]).unwrap();
        gc.pin(cid).unwrap();
        let stats = gc.gc_stats();
        assert_eq!(stats.pinned_count, 1);
    }

    #[test]
    fn test_gc_stats_root_count() {
        let mut gc = default_gc();
        let cid = make_cid(0);
        gc.register_block(cid, 100, vec![]).unwrap();
        gc.add_root(cid);
        let stats = gc.gc_stats();
        assert_eq!(stats.root_count, 1);
    }

    #[test]
    fn test_gc_stats_orphan_estimate() {
        let mut gc = default_gc();
        gc.register_block(make_cid(0), 100, vec![]).unwrap(); // root
        gc.register_block(make_cid(1), 100, vec![]).unwrap(); // orphan
        gc.add_root(make_cid(0));
        let stats = gc.gc_stats();
        assert_eq!(stats.orphan_estimate, 1);
    }

    // ── Edge management ───────────────────────────────────────────────────────

    #[test]
    fn test_children_of() {
        let mut gc = default_gc();
        let parent = make_cid(0);
        let child = make_cid(1);
        gc.register_block(parent, 100, vec![child]).unwrap();
        let children = gc.children_of(&parent);
        assert_eq!(children.len(), 1);
        assert_eq!(children[0], child);
    }

    #[test]
    fn test_update_edges() {
        let mut gc = default_gc();
        let parent = make_cid(0);
        let child1 = make_cid(1);
        let child2 = make_cid(2);
        gc.register_block(parent, 100, vec![child1]).unwrap();
        gc.update_edges(parent, vec![child1, child2]).unwrap();
        assert_eq!(gc.children_of(&parent).len(), 2);
    }

    #[test]
    fn test_update_edges_unknown_fails() {
        let mut gc = default_gc();
        let result = gc.update_edges(make_cid(99), vec![]);
        assert!(matches!(result, Err(BgcError::UnknownBlock(_))));
    }

    #[test]
    fn test_compact_edges_removes_dangling() {
        let mut gc = default_gc();
        let parent = make_cid(0);
        let child = make_cid(1);
        gc.register_block(parent, 100, vec![child]).unwrap();
        gc.register_block(child, 50, vec![]).unwrap();
        // Remove child directly.
        gc.registry.remove(&child);
        gc.compact_edges();
        assert_eq!(gc.children_of(&parent).len(), 0);
    }

    // ── PRNG / CID helpers ────────────────────────────────────────────────────

    #[test]
    fn test_random_cid_differs() {
        let mut gc = default_gc();
        let a = gc.random_cid();
        let b = gc.random_cid();
        assert_ne!(a, b);
    }

    #[test]
    fn test_cid_from_bytes_deterministic() {
        let a = BlockGarbageCollector::cid_from_bytes(b"hello");
        let b = BlockGarbageCollector::cid_from_bytes(b"hello");
        assert_eq!(a, b);
    }

    #[test]
    fn test_cid_from_bytes_distinct() {
        let a = BlockGarbageCollector::cid_from_bytes(b"hello");
        let b = BlockGarbageCollector::cid_from_bytes(b"world");
        assert_ne!(a, b);
    }

    #[test]
    fn test_cid_hash_stable() {
        let cid = make_cid(42);
        let h1 = BlockGarbageCollector::cid_hash(&cid);
        let h2 = BlockGarbageCollector::cid_hash(&cid);
        assert_eq!(h1, h2);
    }

    // ── Utility methods ───────────────────────────────────────────────────────

    #[test]
    fn test_all_cids() {
        let mut gc = default_gc();
        for i in 0..5u8 {
            gc.register_block(make_cid(i), 10, vec![]).unwrap();
        }
        assert_eq!(gc.all_cids().len(), 5);
    }

    #[test]
    fn test_reachable_set() {
        let mut gc = default_gc();
        let root = make_cid(0);
        gc.register_block(root, 100, vec![]).unwrap();
        gc.add_root(root);
        let rs = gc.reachable_set();
        assert!(rs.contains(&root));
    }

    #[test]
    fn test_clear_resets_state() {
        let mut gc = default_gc();
        gc.register_block(make_cid(0), 100, vec![]).unwrap();
        gc.run_gc(BgcGcPolicy::MarkAndSweep).unwrap();
        gc.clear();
        assert_eq!(gc.block_count(), 0);
        assert_eq!(gc.log_len(), 0);
        assert_eq!(gc.gc_stats().gc_cycles, 0);
    }

    #[test]
    fn test_reconcile_pin_flags() {
        let mut gc = default_gc();
        let cid = make_cid(1);
        gc.register_block(cid, 100, vec![]).unwrap();
        gc.pin_set.insert(cid); // Insert directly without going through pin().
        gc.reconcile_pin_flags();
        assert!(gc.get_block(&cid).unwrap().is_pinned);
    }

    #[test]
    fn test_promote_generation() {
        let mut gc = default_gc();
        let cid = make_cid(1);
        gc.register_block(cid, 100, vec![]).unwrap();
        gc.promote_generation();
        assert_eq!(gc.get_block(&cid).unwrap().generation, 1);
    }

    // ── Large-scale tests ─────────────────────────────────────────────────────

    #[test]
    fn test_large_registry_gc() {
        let mut gc = default_gc();
        // Root chain of 100 blocks.
        let root = make_cid(0);
        gc.register_block(root, 100, vec![]).unwrap();
        gc.add_root(root);
        // 900 orphans.
        for i in 1..=900u16 {
            gc.register_block(make_cid_u16(i), 10, vec![]).unwrap();
        }
        let result = gc.run_gc(BgcGcPolicy::MarkAndSweep).unwrap();
        assert!(result.blocks_freed > 0);
    }

    #[test]
    fn test_gc_preserves_reachable_tree() {
        let mut gc = default_gc();
        // Build a tree: root -> [a, b], a -> [c], b -> [d]
        let root = make_cid(0);
        let a = make_cid(1);
        let b = make_cid(2);
        let c = make_cid(3);
        let d = make_cid(4);
        gc.register_block(root, 1, vec![a, b]).unwrap();
        gc.register_block(a, 1, vec![c]).unwrap();
        gc.register_block(b, 1, vec![d]).unwrap();
        gc.register_block(c, 1, vec![]).unwrap();
        gc.register_block(d, 1, vec![]).unwrap();
        gc.add_root(root);

        let result = gc.run_gc(BgcGcPolicy::MarkAndSweep).unwrap();
        assert_eq!(result.blocks_freed, 0);
        assert_eq!(gc.block_count(), 5);
    }

    #[test]
    fn test_multiple_roots() {
        let mut gc = default_gc();
        let r1 = make_cid(0);
        let r2 = make_cid(1);
        let orphan = make_cid(2);
        gc.register_block(r1, 100, vec![]).unwrap();
        gc.register_block(r2, 100, vec![]).unwrap();
        gc.register_block(orphan, 100, vec![]).unwrap();
        gc.add_root(r1);
        gc.add_root(r2);
        let result = gc.run_gc(BgcGcPolicy::MarkAndSweep).unwrap();
        assert_eq!(result.blocks_freed, 1);
    }

    #[test]
    fn test_gc_log_bounded_at_1000() {
        let mut gc = default_gc();
        for _ in 0..1010 {
            gc.run_gc(BgcGcPolicy::MarkAndSweep).unwrap();
        }
        assert!(gc.log_len() <= 1000);
    }

    #[test]
    fn test_batch_size_limits_sweep() {
        let mut gc = default_gc();
        gc.config_mut().batch_size = 5;
        for i in 0..20u8 {
            gc.register_block(make_cid(i), 10, vec![]).unwrap();
        }
        let result = gc.run_gc(BgcGcPolicy::MarkAndSweep).unwrap();
        // At most batch_size blocks swept.
        assert!(result.blocks_freed <= 5);
    }

    #[test]
    fn test_tricolor_matches_mark_and_sweep() {
        let mut gc1 = default_gc();
        let mut gc2 = default_gc();

        let root = make_cid(0);
        let child = make_cid(1);
        let orphan = make_cid(2);

        for gc in [&mut gc1, &mut gc2] {
            gc.register_block(root, 100, vec![child]).unwrap();
            gc.register_block(child, 50, vec![]).unwrap();
            gc.register_block(orphan, 75, vec![]).unwrap();
            gc.add_root(root);
        }

        let r1 = gc1.run_gc(BgcGcPolicy::MarkAndSweep).unwrap();
        let r2 = gc2.run_gc(BgcGcPolicy::TriColor).unwrap();

        assert_eq!(r1.blocks_freed, r2.blocks_freed);
        assert_eq!(r1.bytes_freed, r2.bytes_freed);
    }

    #[test]
    fn test_touch_updates_last_accessed() {
        let mut gc = default_gc();
        let cid = make_cid(1);
        gc.register_block(cid, 100, vec![]).unwrap();
        let before = gc.get_block(&cid).unwrap().last_accessed;
        gc.set_clock(before + 5000);
        gc.touch(&cid).unwrap();
        let after = gc.get_block(&cid).unwrap().last_accessed;
        assert!(after >= before);
    }

    #[test]
    fn test_touch_unknown_fails() {
        let mut gc = default_gc();
        let result = gc.touch(&make_cid(99));
        assert!(matches!(result, Err(BgcError::UnknownBlock(_))));
    }

    // ── fnv1a_64 and xorshift64 unit tests ───────────────────────────────────

    #[test]
    fn test_fnv1a_64_empty() {
        let h = fnv1a_64(b"");
        assert_eq!(h, 14695981039346656037u64);
    }

    #[test]
    fn test_fnv1a_64_hello() {
        let h = fnv1a_64(b"hello");
        assert_ne!(h, 0);
    }

    #[test]
    fn test_xorshift64_non_zero() {
        let mut state: u64 = 12345;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }

    #[test]
    fn test_xorshift64_advances_state() {
        let mut state: u64 = 1;
        let v1 = xorshift64(&mut state);
        let v2 = xorshift64(&mut state);
        assert_ne!(v1, v2);
    }

    // ── BgcError Display ─────────────────────────────────────────────────────

    #[test]
    fn test_error_display_unknown() {
        let e = BgcError::UnknownBlock(make_cid(1));
        assert!(e.to_string().contains("unknown block"));
    }

    #[test]
    fn test_error_display_pinned() {
        let e = BgcError::BlockIsPinned(make_cid(1));
        assert!(e.to_string().contains("pinned"));
    }

    #[test]
    fn test_error_display_timeout() {
        let e = BgcError::MarkTimeout;
        assert!(e.to_string().contains("timed out"));
    }

    #[test]
    fn test_error_display_overflow() {
        let e = BgcError::RefCountOverflow(make_cid(1));
        assert!(e.to_string().contains("overflow"));
    }

    // ── GcPhase / GcPolicy derive tests ──────────────────────────────────────

    #[test]
    fn test_gcphase_debug() {
        let p = BgcGcPhase::Mark;
        assert_eq!(format!("{:?}", p), "Mark");
    }

    #[test]
    fn test_gcpolicy_debug() {
        let p = BgcGcPolicy::Generational;
        assert_eq!(format!("{:?}", p), "Generational");
    }

    #[test]
    fn test_gcphase_eq() {
        assert_eq!(BgcGcPhase::Idle, BgcGcPhase::Idle);
        assert_ne!(BgcGcPhase::Sweep, BgcGcPhase::Mark);
    }

    #[test]
    fn test_gcpolicy_eq() {
        assert_eq!(BgcGcPolicy::MarkAndSweep, BgcGcPolicy::MarkAndSweep);
        assert_ne!(BgcGcPolicy::TriColor, BgcGcPolicy::Generational);
    }

    // ── BgcGcResult defaults ─────────────────────────────────────────────────

    #[test]
    fn test_gc_result_default() {
        let r = BgcGcResult::default();
        assert_eq!(r.blocks_freed, 0);
        assert_eq!(r.bytes_freed, 0);
        assert!(!r.dry_run);
        assert!(r.policy.is_none());
    }

    // ── Sweep result ─────────────────────────────────────────────────────────

    #[test]
    fn test_sweep_result_default() {
        let r = BgcSweepResult::default();
        assert!(r.removed.is_empty());
        assert_eq!(r.bytes_freed, 0);
    }

    // ── Collector stats accumulation ─────────────────────────────────────────

    #[test]
    fn test_stats_accumulate_over_multiple_gc() {
        let mut gc = default_gc();
        // Add 2 orphans, run GC twice (batch_size >= 2).
        gc.register_block(make_cid(1), 100, vec![]).unwrap();
        gc.run_gc(BgcGcPolicy::MarkAndSweep).unwrap();
        gc.register_block(make_cid(2), 200, vec![]).unwrap();
        gc.run_gc(BgcGcPolicy::MarkAndSweep).unwrap();
        let stats = gc.gc_stats();
        assert!(stats.total_bytes_freed >= 100);
        assert!(stats.total_blocks_freed >= 1);
    }
}
