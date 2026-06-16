//! Block Fragment Store
//!
//! Stores and reassembles fragmented blocks, supporting parallel fragment reception
//! and erasure recovery. Fragments are identified by their block CID and index.
//! Once all fragments for a block are received, they are assembled into the complete
//! block by concatenating fragments in index order after verifying checksums.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// FNV-1a helpers
// ---------------------------------------------------------------------------

/// FNV-1a 32-bit checksum (lower 32 bits of the 64-bit digest).
pub fn bfs_fnv1a_32(data: &[u8]) -> u32 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h as u32
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Uniquely identifies one fragment within a block.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FragmentId {
    /// The CID of the block this fragment belongs to.
    pub block_cid: String,
    /// Zero-based index of this fragment within the block.
    pub index: u32,
}

/// A single fragment carrying a slice of a block's data.
#[derive(Debug, Clone)]
pub struct BfsFragment {
    /// Identity of this fragment.
    pub id: FragmentId,
    /// Raw bytes for this fragment.
    pub data: Vec<u8>,
    /// Total number of fragments required to reassemble the block.
    pub total_fragments: u32,
    /// FNV-1a-32 checksum of `data`.
    pub checksum: u32,
}

impl BfsFragment {
    /// Construct a new fragment, computing its checksum automatically.
    pub fn new(
        block_cid: impl Into<String>,
        index: u32,
        data: Vec<u8>,
        total_fragments: u32,
    ) -> Self {
        let checksum = bfs_fnv1a_32(&data);
        BfsFragment {
            id: FragmentId {
                block_cid: block_cid.into(),
                index,
            },
            data,
            total_fragments,
            checksum,
        }
    }
}

/// Current state of a fragment set.
#[derive(Debug, Clone, PartialEq)]
pub enum FragmentSetState {
    /// Not all fragments have been received yet.
    Incomplete {
        /// Number of fragments received so far.
        received: u32,
        /// Total fragments expected.
        total: u32,
    },
    /// All fragments have been received and the block has been assembled.
    Complete,
    /// All fragments received but some checksums failed.
    Corrupted {
        /// Indices of fragments whose checksums do not match.
        bad_indices: Vec<u32>,
    },
}

/// A collection of fragments for one block.
#[derive(Debug, Clone)]
pub struct FragmentSet {
    /// CID of the block being assembled.
    pub block_cid: String,
    /// Expected total fragment count.
    pub total_fragments: u32,
    /// Received fragments keyed by index.
    pub fragments: HashMap<u32, BfsFragment>,
    /// Timestamp (ms) when the first fragment arrived.
    pub created_at: u64,
    /// Timestamp (ms) of the most recent fragment arrival.
    pub last_updated: u64,
}

impl FragmentSet {
    fn new(block_cid: String, total_fragments: u32, now: u64) -> Self {
        FragmentSet {
            block_cid,
            total_fragments,
            fragments: HashMap::new(),
            created_at: now,
            last_updated: now,
        }
    }

    /// Whether all fragments are present.
    fn is_complete(&self) -> bool {
        self.fragments.len() as u32 == self.total_fragments
    }

    /// Current state (does not perform assembly).
    fn state(&self) -> FragmentSetState {
        if self.is_complete() {
            // Verify checksums to decide Complete vs Corrupted.
            let bad: Vec<u32> = self
                .fragments
                .iter()
                .filter(|(_, f)| bfs_fnv1a_32(&f.data) != f.checksum)
                .map(|(idx, _)| *idx)
                .collect();
            if bad.is_empty() {
                FragmentSetState::Complete
            } else {
                let mut sorted = bad;
                sorted.sort_unstable();
                FragmentSetState::Corrupted {
                    bad_indices: sorted,
                }
            }
        } else {
            FragmentSetState::Incomplete {
                received: self.fragments.len() as u32,
                total: self.total_fragments,
            }
        }
    }
}

/// A fully assembled block produced from its constituent fragments.
#[derive(Debug, Clone)]
pub struct AssembledBlock {
    /// CID of this block.
    pub cid: String,
    /// Complete block data (fragments concatenated in index order).
    pub data: Vec<u8>,
    /// Number of fragments that were assembled.
    pub fragment_count: u32,
    /// Timestamp (ms) when this block was assembled.
    pub assembled_at: u64,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by [`BlockFragmentStore`] operations.
#[derive(Debug, Clone, PartialEq)]
pub enum FragmentError {
    /// A fragment with this (block_cid, index) already exists in the store.
    DuplicateFragment {
        /// Block CID.
        block_cid: String,
        /// Fragment index.
        index: u32,
    },
    /// The fragment index is out of range for the declared total.
    IndexOutOfRange {
        /// Fragment index received.
        index: u32,
        /// Total fragments declared.
        total: u32,
    },
    /// No fragment set found for the given block CID.
    BlockNotFound(String),
    /// Assembly requested but not all fragments have been received.
    AssemblyIncomplete {
        /// Fragments received so far.
        received: u32,
        /// Total fragments expected.
        total: u32,
    },
    /// A fragment failed its checksum during assembly.
    ChecksumMismatch {
        /// Index of the offending fragment.
        index: u32,
    },
    /// The pending-set or assembled-block cap has been reached.
    StoreFull,
}

impl std::fmt::Display for FragmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FragmentError::DuplicateFragment { block_cid, index } => {
                write!(f, "duplicate fragment {index} for block {block_cid}")
            }
            FragmentError::IndexOutOfRange { index, total } => {
                write!(f, "fragment index {index} out of range (total={total})")
            }
            FragmentError::BlockNotFound(cid) => {
                write!(f, "block not found: {cid}")
            }
            FragmentError::AssemblyIncomplete { received, total } => {
                write!(f, "assembly incomplete: {received}/{total} fragments")
            }
            FragmentError::ChecksumMismatch { index } => {
                write!(f, "checksum mismatch for fragment {index}")
            }
            FragmentError::StoreFull => write!(f, "store is full"),
        }
    }
}

impl std::error::Error for FragmentError {}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Aggregate statistics about the store.
#[derive(Debug, Clone)]
pub struct FragmentStats {
    /// Number of in-progress (pending) fragment sets.
    pub pending_sets: usize,
    /// Number of successfully assembled blocks in cache.
    pub assembled_blocks: usize,
    /// Total number of individual fragments currently stored (pending only).
    pub total_fragments_stored: usize,
    /// Average completion ratio across all pending sets (0.0–1.0).
    pub avg_completion_rate: f64,
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// Production-grade store for fragmented blocks.
///
/// Handles concurrent fragment arrival, checksum verification, automatic assembly
/// on completion, LRU-style eviction of the oldest pending sets when the cap is
/// exceeded, and a bounded assembled-block cache.
#[derive(Debug)]
pub struct BlockFragmentStore {
    /// Pending (incomplete) fragment sets keyed by block CID.
    sets: HashMap<String, FragmentSet>,
    /// Assembled (complete) blocks keyed by block CID.
    assembled: HashMap<String, AssembledBlock>,
    /// Maximum number of pending fragment sets before oldest is evicted.
    max_pending: usize,
    /// Maximum number of assembled blocks to cache.
    max_assembled: usize,
}

impl BlockFragmentStore {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new store with the given capacity limits.
    ///
    /// * `max_pending`  – maximum number of in-progress fragment sets.
    /// * `max_assembled` – maximum number of assembled blocks kept in cache.
    pub fn new(max_pending: usize, max_assembled: usize) -> Self {
        BlockFragmentStore {
            sets: HashMap::new(),
            assembled: HashMap::new(),
            max_pending,
            max_assembled,
        }
    }

    // -----------------------------------------------------------------------
    // Fragment storage
    // -----------------------------------------------------------------------

    /// Store a fragment.
    ///
    /// If the fragment completes its set, assembly is attempted automatically
    /// and the pending set is removed.  If `max_pending` would be exceeded when
    /// creating a new set, the oldest pending set (by `created_at`) is dropped.
    ///
    /// Returns the current [`FragmentSetState`] after the operation.
    pub fn store_fragment(
        &mut self,
        fragment: BfsFragment,
        now: u64,
    ) -> Result<FragmentSetState, FragmentError> {
        let block_cid = fragment.id.block_cid.clone();
        let index = fragment.id.index;
        let total = fragment.total_fragments;

        // Validate index range.
        if total == 0 || index >= total {
            return Err(FragmentError::IndexOutOfRange { index, total });
        }

        // If the block is already assembled, reject the fragment.
        if self.assembled.contains_key(&block_cid) {
            // Treat as duplicate: the block is already complete.
            return Ok(FragmentSetState::Complete);
        }

        // Get or create the fragment set.
        if !self.sets.contains_key(&block_cid) {
            // Evict oldest pending set if we are at capacity.
            if self.sets.len() >= self.max_pending {
                self.evict_oldest_pending();
                // If we still can't make room (e.g. max_pending == 0), error.
                if self.sets.len() >= self.max_pending {
                    return Err(FragmentError::StoreFull);
                }
            }
            self.sets.insert(
                block_cid.clone(),
                FragmentSet::new(block_cid.clone(), total, now),
            );
        }

        let set = self
            .sets
            .get_mut(&block_cid)
            .ok_or_else(|| FragmentError::BlockNotFound(block_cid.clone()))?;

        // Check for duplicate.
        if set.fragments.contains_key(&index) {
            return Err(FragmentError::DuplicateFragment { block_cid, index });
        }

        set.fragments.insert(index, fragment);
        set.last_updated = now;

        // Check if complete.
        if set.is_complete() {
            // Attempt assembly; this also validates checksums.
            match self.assemble_internal(&block_cid, now) {
                Ok(block) => {
                    // Evict oldest assembled if at capacity.
                    if self.assembled.len() >= self.max_assembled {
                        self.evict_oldest_assembled();
                    }
                    self.assembled.insert(block_cid.clone(), block);
                    self.sets.remove(&block_cid);
                    Ok(FragmentSetState::Complete)
                }
                Err(FragmentError::ChecksumMismatch { index: bad_idx }) => {
                    // Leave the set in place for potential repair.
                    Ok(FragmentSetState::Corrupted {
                        bad_indices: vec![bad_idx],
                    })
                }
                Err(e) => Err(e),
            }
        } else {
            let state = self
                .sets
                .get(&block_cid)
                .map(|s| s.state())
                .unwrap_or(FragmentSetState::Incomplete { received: 0, total });
            Ok(state)
        }
    }

    // -----------------------------------------------------------------------
    // State inspection
    // -----------------------------------------------------------------------

    /// Return the current state of a fragment set.
    ///
    /// Returns `None` if neither a pending set nor an assembled block exists for
    /// the given CID.
    pub fn get_state(&self, block_cid: &str) -> Option<FragmentSetState> {
        if self.assembled.contains_key(block_cid) {
            return Some(FragmentSetState::Complete);
        }
        self.sets.get(block_cid).map(|s| s.state())
    }

    /// Return the indices of fragments that have not yet been received, in
    /// ascending order.  Returns an empty vec if the block is assembled or
    /// unknown.
    pub fn missing_indices(&self, block_cid: &str) -> Vec<u32> {
        if self.assembled.contains_key(block_cid) {
            return Vec::new();
        }
        let set = match self.sets.get(block_cid) {
            Some(s) => s,
            None => return Vec::new(),
        };
        (0..set.total_fragments)
            .filter(|i| !set.fragments.contains_key(i))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Assembly
    // -----------------------------------------------------------------------

    /// Assemble a block from its fragments.
    ///
    /// All fragments must be present and pass checksum verification.  On success,
    /// the assembled block is stored in the cache and the pending set is removed.
    pub fn assemble(&mut self, block_cid: &str, now: u64) -> Result<AssembledBlock, FragmentError> {
        if let Some(block) = self.assembled.get(block_cid) {
            return Ok(block.clone());
        }

        let assembled = self.assemble_internal(block_cid, now)?;

        if self.assembled.len() >= self.max_assembled {
            self.evict_oldest_assembled();
        }
        let cid = block_cid.to_owned();
        self.assembled.insert(cid.clone(), assembled.clone());
        self.sets.remove(&cid);
        Ok(assembled)
    }

    /// Internal assembly logic (does not mutate `self.assembled` or `self.sets`).
    fn assemble_internal(
        &self,
        block_cid: &str,
        now: u64,
    ) -> Result<AssembledBlock, FragmentError> {
        let set = self
            .sets
            .get(block_cid)
            .ok_or_else(|| FragmentError::BlockNotFound(block_cid.to_owned()))?;

        if !set.is_complete() {
            return Err(FragmentError::AssemblyIncomplete {
                received: set.fragments.len() as u32,
                total: set.total_fragments,
            });
        }

        // Sort indices and verify.
        let mut indices: Vec<u32> = (0..set.total_fragments).collect();
        indices.sort_unstable();

        let mut data: Vec<u8> = Vec::new();
        for idx in &indices {
            let fragment = set
                .fragments
                .get(idx)
                .ok_or_else(|| FragmentError::BlockNotFound(block_cid.to_owned()))?;
            if bfs_fnv1a_32(&fragment.data) != fragment.checksum {
                return Err(FragmentError::ChecksumMismatch { index: *idx });
            }
            data.extend_from_slice(&fragment.data);
        }

        Ok(AssembledBlock {
            cid: block_cid.to_owned(),
            data,
            fragment_count: set.total_fragments,
            assembled_at: now,
        })
    }

    // -----------------------------------------------------------------------
    // Assembled block retrieval
    // -----------------------------------------------------------------------

    /// Return a reference to the data of an assembled block, or `None` if not
    /// present in the cache.
    pub fn get_assembled(&self, block_cid: &str) -> Option<&[u8]> {
        self.assembled.get(block_cid).map(|b| b.data.as_slice())
    }

    /// Return a reference to the full [`AssembledBlock`] struct, if cached.
    pub fn get_assembled_block(&self, block_cid: &str) -> Option<&AssembledBlock> {
        self.assembled.get(block_cid)
    }

    // -----------------------------------------------------------------------
    // Verification helpers
    // -----------------------------------------------------------------------

    /// Return `true` if the fragment's checksum matches the recomputed value.
    pub fn verify_fragment(fragment: &BfsFragment) -> bool {
        bfs_fnv1a_32(&fragment.data) == fragment.checksum
    }

    /// Return the indices of fragments in a set whose checksums do not match.
    pub fn verify_set(set: &FragmentSet) -> Vec<u32> {
        let mut bad: Vec<u32> = set
            .fragments
            .iter()
            .filter(|(_, f)| bfs_fnv1a_32(&f.data) != f.checksum)
            .map(|(idx, _)| *idx)
            .collect();
        bad.sort_unstable();
        bad
    }

    // -----------------------------------------------------------------------
    // Eviction
    // -----------------------------------------------------------------------

    /// Remove all pending (incomplete) sets whose `created_at` is older than
    /// `max_age_ms` milliseconds before `now`.  Returns the number of sets
    /// removed.
    pub fn evict_stale_pending(&mut self, max_age_ms: u64, now: u64) -> usize {
        let cutoff = now.saturating_sub(max_age_ms);
        let before = self.sets.len();
        self.sets.retain(|_, set| set.created_at >= cutoff);
        before - self.sets.len()
    }

    /// Remove an assembled block from the cache by CID.  Returns `true` if the
    /// block was present and removed.
    pub fn evict_assembled(&mut self, cid: &str) -> bool {
        self.assembled.remove(cid).is_some()
    }

    /// Evict the oldest pending set (by `created_at` timestamp).
    fn evict_oldest_pending(&mut self) {
        let oldest = self
            .sets
            .iter()
            .min_by_key(|(_, s)| s.created_at)
            .map(|(k, _)| k.clone());
        if let Some(key) = oldest {
            self.sets.remove(&key);
        }
    }

    /// Evict the oldest assembled block (by `assembled_at` timestamp).
    fn evict_oldest_assembled(&mut self) {
        let oldest = self
            .assembled
            .iter()
            .min_by_key(|(_, b)| b.assembled_at)
            .map(|(k, _)| k.clone());
        if let Some(key) = oldest {
            self.assembled.remove(&key);
        }
    }

    // -----------------------------------------------------------------------
    // Counts
    // -----------------------------------------------------------------------

    /// Number of pending (incomplete) fragment sets.
    pub fn pending_count(&self) -> usize {
        self.sets.len()
    }

    /// Number of assembled blocks in the cache.
    pub fn assembled_count(&self) -> usize {
        self.assembled.len()
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Aggregate statistics about the store.
    pub fn stats(&self) -> FragmentStats {
        let pending_sets = self.sets.len();
        let assembled_blocks = self.assembled.len();

        let total_fragments_stored: usize = self.sets.values().map(|s| s.fragments.len()).sum();

        let avg_completion_rate = if pending_sets == 0 {
            0.0
        } else {
            let sum: f64 = self
                .sets
                .values()
                .map(|s| {
                    if s.total_fragments == 0 {
                        0.0
                    } else {
                        s.fragments.len() as f64 / s.total_fragments as f64
                    }
                })
                .sum();
            sum / pending_sets as f64
        };

        FragmentStats {
            pending_sets,
            assembled_blocks,
            total_fragments_stored,
            avg_completion_rate,
        }
    }

    // -----------------------------------------------------------------------
    // Advanced / diagnostic helpers
    // -----------------------------------------------------------------------

    /// Return all pending fragment set CIDs.
    pub fn pending_cids(&self) -> Vec<&str> {
        self.sets.keys().map(String::as_str).collect()
    }

    /// Return all assembled block CIDs.
    pub fn assembled_cids(&self) -> Vec<&str> {
        self.assembled.keys().map(String::as_str).collect()
    }

    /// Clear all pending sets without assembling them.  Returns the number of
    /// sets dropped.
    pub fn clear_pending(&mut self) -> usize {
        let count = self.sets.len();
        self.sets.clear();
        count
    }

    /// Clear all assembled blocks from the cache.  Returns the number of blocks
    /// dropped.
    pub fn clear_assembled(&mut self) -> usize {
        let count = self.assembled.len();
        self.assembled.clear();
        count
    }

    /// Return how many more pending sets can be accepted before the cap is hit.
    pub fn pending_capacity_remaining(&self) -> usize {
        self.max_pending.saturating_sub(self.sets.len())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use crate::block_fragment_store::{
        bfs_fnv1a_32, BfsFragment, BlockFragmentStore, FragmentError, FragmentId, FragmentSet,
        FragmentSetState,
    };

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_fragment(cid: &str, index: u32, data: &[u8], total: u32) -> BfsFragment {
        BfsFragment::new(cid, index, data.to_vec(), total)
    }

    fn make_fragment_tampered(cid: &str, index: u32, data: &[u8], total: u32) -> BfsFragment {
        let mut f = make_fragment(cid, index, data, total);
        f.checksum = f.checksum.wrapping_add(1); // corrupt checksum
        f
    }

    fn store_with_defaults() -> BlockFragmentStore {
        BlockFragmentStore::new(16, 16)
    }

    // -----------------------------------------------------------------------
    // 1. FNV-1a checksum correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_fnv1a_empty() {
        // FNV-1a of empty slice should equal the FNV offset basis cast to u32.
        let h = bfs_fnv1a_32(&[]);
        let expected = 14_695_981_039_346_656_037_u64 as u32;
        assert_eq!(h, expected);
    }

    #[test]
    fn test_fnv1a_known_value() {
        // "abc" -> known FNV-1a-64 value 0xe71fa2190541574b, lower 32 bits = 0x0541574b
        let h = bfs_fnv1a_32(b"abc");
        assert_eq!(h, 0x0541_574b_u32);
    }

    #[test]
    fn test_fnv1a_different_inputs_differ() {
        assert_ne!(bfs_fnv1a_32(b"hello"), bfs_fnv1a_32(b"world"));
    }

    // -----------------------------------------------------------------------
    // 2. Fragment construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_fragment_new_computes_checksum() {
        let data = b"test data";
        let f = BfsFragment::new("cid1", 0, data.to_vec(), 3);
        assert_eq!(f.checksum, bfs_fnv1a_32(data));
    }

    #[test]
    fn test_fragment_id_equality() {
        let a = FragmentId {
            block_cid: "x".into(),
            index: 1,
        };
        let b = FragmentId {
            block_cid: "x".into(),
            index: 1,
        };
        assert_eq!(a, b);
    }

    // -----------------------------------------------------------------------
    // 3. Basic store_fragment / happy path
    // -----------------------------------------------------------------------

    #[test]
    fn test_store_single_fragment_incomplete() {
        let mut store = store_with_defaults();
        let f = make_fragment("cid1", 0, b"part0", 3);
        let state = store.store_fragment(f, 1000).expect("store ok");
        assert_eq!(
            state,
            FragmentSetState::Incomplete {
                received: 1,
                total: 3
            }
        );
    }

    #[test]
    fn test_store_all_fragments_completes() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("cid1", 0, b"aa", 2), 100)
            .expect("ok");
        let state = store
            .store_fragment(make_fragment("cid1", 1, b"bb", 2), 200)
            .expect("ok");
        assert_eq!(state, FragmentSetState::Complete);
    }

    #[test]
    fn test_assembled_block_data_correct() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("cid1", 0, b"hello", 2), 1)
            .expect("ok");
        store
            .store_fragment(make_fragment("cid1", 1, b"world", 2), 2)
            .expect("ok");
        let data = store.get_assembled("cid1").expect("assembled");
        assert_eq!(data, b"helloworld");
    }

    #[test]
    fn test_fragments_ordered_by_index() {
        let mut store = store_with_defaults();
        // Insert out of order
        store
            .store_fragment(make_fragment("cid2", 2, b"C", 3), 1)
            .expect("ok");
        store
            .store_fragment(make_fragment("cid2", 0, b"A", 3), 2)
            .expect("ok");
        store
            .store_fragment(make_fragment("cid2", 1, b"B", 3), 3)
            .expect("ok");
        let data = store.get_assembled("cid2").expect("assembled");
        assert_eq!(data, b"ABC");
    }

    // -----------------------------------------------------------------------
    // 4. Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_duplicate_fragment_error() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("cid1", 0, b"a", 2), 1)
            .expect("ok");
        let err = store
            .store_fragment(make_fragment("cid1", 0, b"a", 2), 2)
            .expect_err("dup");
        assert_eq!(
            err,
            FragmentError::DuplicateFragment {
                block_cid: "cid1".into(),
                index: 0
            }
        );
    }

    #[test]
    fn test_index_out_of_range_error() {
        let mut store = store_with_defaults();
        let f = make_fragment("cid1", 5, b"x", 3);
        let err = store.store_fragment(f, 1).expect_err("oob");
        assert_eq!(err, FragmentError::IndexOutOfRange { index: 5, total: 3 });
    }

    #[test]
    fn test_total_fragments_zero_error() {
        let mut store = store_with_defaults();
        let f = BfsFragment {
            id: FragmentId {
                block_cid: "cid1".into(),
                index: 0,
            },
            data: vec![1],
            total_fragments: 0,
            checksum: 0,
        };
        let err = store.store_fragment(f, 1).expect_err("zero total");
        assert_eq!(err, FragmentError::IndexOutOfRange { index: 0, total: 0 });
    }

    #[test]
    fn test_block_not_found_for_state() {
        let store = store_with_defaults();
        assert!(store.get_state("nonexistent").is_none());
    }

    #[test]
    fn test_assemble_not_found_error() {
        let mut store = store_with_defaults();
        let err = store.assemble("ghost", 0).expect_err("not found");
        assert_eq!(err, FragmentError::BlockNotFound("ghost".into()));
    }

    #[test]
    fn test_assemble_incomplete_error() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("cid1", 0, b"a", 3), 1)
            .expect("ok");
        let err = store.assemble("cid1", 2).expect_err("incomplete");
        assert_eq!(
            err,
            FragmentError::AssemblyIncomplete {
                received: 1,
                total: 3
            }
        );
    }

    // -----------------------------------------------------------------------
    // 5. Checksum / corruption
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_fragment_valid() {
        let f = make_fragment("cid", 0, b"data", 1);
        assert!(BlockFragmentStore::verify_fragment(&f));
    }

    #[test]
    fn test_verify_fragment_invalid() {
        let f = make_fragment_tampered("cid", 0, b"data", 1);
        assert!(!BlockFragmentStore::verify_fragment(&f));
    }

    #[test]
    fn test_verify_set_all_valid() {
        let mut set = FragmentSet::new("cid".into(), 2, 0);
        set.fragments.insert(0, make_fragment("cid", 0, b"a", 2));
        set.fragments.insert(1, make_fragment("cid", 1, b"b", 2));
        assert!(BlockFragmentStore::verify_set(&set).is_empty());
    }

    #[test]
    fn test_verify_set_one_bad() {
        let mut set = FragmentSet::new("cid".into(), 2, 0);
        set.fragments.insert(0, make_fragment("cid", 0, b"a", 2));
        set.fragments
            .insert(1, make_fragment_tampered("cid", 1, b"b", 2));
        let bad = BlockFragmentStore::verify_set(&set);
        assert_eq!(bad, vec![1u32]);
    }

    #[test]
    fn test_corrupted_state_on_bad_checksum() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("cid1", 0, b"good", 2), 1)
            .expect("ok");
        let state = store
            .store_fragment(make_fragment_tampered("cid1", 1, b"bad", 2), 2)
            .expect("stored despite tamper");
        // Depending on implementation the state should reflect corruption.
        match state {
            FragmentSetState::Corrupted { bad_indices } => {
                assert!(bad_indices.contains(&1));
            }
            // If the set remains pending it's also acceptable since checksum
            // will be caught at assembly time.
            FragmentSetState::Incomplete { .. } => {}
            FragmentSetState::Complete => panic!("should not be complete with bad checksum"),
        }
    }

    #[test]
    fn test_assemble_detects_checksum_mismatch() {
        let mut store = BlockFragmentStore::new(4, 4);
        // Manually insert a tampered fragment by first storing valid then
        // using a low-level approach: store both, manipulate the set directly.
        // Instead, build a set with a tampered fragment via store_fragment
        // for the valid one, then call assemble on an incomplete set (to get the
        // pending entry), then add a tampered one.
        store
            .store_fragment(make_fragment("cid1", 0, b"ok", 2), 1)
            .expect("ok");
        // Insert tampered fragment directly:
        let tampered = make_fragment_tampered("cid1", 1, b"bad", 2);
        // store_fragment will call assemble internally; we expect a Corrupted or
        // a ChecksumMismatch error propagated.
        let result = store.store_fragment(tampered, 2);
        match result {
            Ok(FragmentSetState::Corrupted { .. }) => {}  // expected
            Ok(FragmentSetState::Incomplete { .. }) => {} // tolerated
            Ok(FragmentSetState::Complete) => panic!("should not complete with bad checksum"),
            Err(FragmentError::ChecksumMismatch { .. }) => {} // also acceptable
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    // -----------------------------------------------------------------------
    // 6. Missing indices
    // -----------------------------------------------------------------------

    #[test]
    fn test_missing_indices_all_missing() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("cid1", 0, b"a", 4), 1)
            .expect("ok");
        let missing = store.missing_indices("cid1");
        assert_eq!(missing, vec![1u32, 2, 3]);
    }

    #[test]
    fn test_missing_indices_none_when_assembled() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("cid1", 0, b"a", 1), 1)
            .expect("ok");
        let missing = store.missing_indices("cid1");
        assert!(missing.is_empty());
    }

    #[test]
    fn test_missing_indices_unknown_cid() {
        let store = store_with_defaults();
        assert!(store.missing_indices("ghost").is_empty());
    }

    // -----------------------------------------------------------------------
    // 7. get_state
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_state_after_assembly() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("cid1", 0, b"x", 1), 1)
            .expect("ok");
        assert_eq!(store.get_state("cid1"), Some(FragmentSetState::Complete));
    }

    #[test]
    fn test_get_state_incomplete() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("cid1", 0, b"x", 3), 1)
            .expect("ok");
        let state = store.get_state("cid1").expect("exists");
        assert_eq!(
            state,
            FragmentSetState::Incomplete {
                received: 1,
                total: 3
            }
        );
    }

    // -----------------------------------------------------------------------
    // 8. Eviction – stale pending
    // -----------------------------------------------------------------------

    #[test]
    fn test_evict_stale_pending_removes_old() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("old_cid", 0, b"a", 2), 0)
            .expect("ok");
        store
            .store_fragment(make_fragment("new_cid", 0, b"b", 2), 5000)
            .expect("ok");
        // Evict sets older than 2000ms, simulating now=6000
        let removed = store.evict_stale_pending(2000, 6000);
        assert_eq!(removed, 1);
        assert!(store.get_state("old_cid").is_none());
        assert!(store.get_state("new_cid").is_some());
    }

    #[test]
    fn test_evict_stale_pending_no_removal_when_fresh() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("cid1", 0, b"a", 2), 5000)
            .expect("ok");
        let removed = store.evict_stale_pending(10000, 6000);
        assert_eq!(removed, 0);
    }

    // -----------------------------------------------------------------------
    // 9. Eviction – assembled cache
    // -----------------------------------------------------------------------

    #[test]
    fn test_evict_assembled_removes_block() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("cid1", 0, b"x", 1), 1)
            .expect("ok");
        assert!(store.get_assembled("cid1").is_some());
        let removed = store.evict_assembled("cid1");
        assert!(removed);
        assert!(store.get_assembled("cid1").is_none());
    }

    #[test]
    fn test_evict_assembled_missing_returns_false() {
        let mut store = store_with_defaults();
        assert!(!store.evict_assembled("ghost"));
    }

    // -----------------------------------------------------------------------
    // 10. Capacity / pending count
    // -----------------------------------------------------------------------

    #[test]
    fn test_pending_count_increments() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("a", 0, b"x", 2), 1)
            .expect("ok");
        store
            .store_fragment(make_fragment("b", 0, b"y", 2), 2)
            .expect("ok");
        assert_eq!(store.pending_count(), 2);
    }

    #[test]
    fn test_assembled_count_increments() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("a", 0, b"x", 1), 1)
            .expect("ok");
        store
            .store_fragment(make_fragment("b", 0, b"y", 1), 2)
            .expect("ok");
        assert_eq!(store.assembled_count(), 2);
    }

    #[test]
    fn test_max_pending_evicts_oldest() {
        let mut store = BlockFragmentStore::new(2, 16);
        store
            .store_fragment(make_fragment("a", 0, b"x", 2), 1)
            .expect("ok");
        store
            .store_fragment(make_fragment("b", 0, b"y", 2), 2)
            .expect("ok");
        // Adding a third should evict the oldest ("a", created_at=1).
        store
            .store_fragment(make_fragment("c", 0, b"z", 2), 3)
            .expect("ok");
        assert_eq!(store.pending_count(), 2);
        assert!(store.get_state("a").is_none(), "oldest should be evicted");
    }

    #[test]
    fn test_store_full_error_when_max_pending_zero() {
        let mut store = BlockFragmentStore::new(0, 16);
        let err = store
            .store_fragment(make_fragment("cid1", 0, b"x", 2), 1)
            .expect_err("store full");
        assert_eq!(err, FragmentError::StoreFull);
    }

    // -----------------------------------------------------------------------
    // 11. Stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_empty_store() {
        let store = store_with_defaults();
        let s = store.stats();
        assert_eq!(s.pending_sets, 0);
        assert_eq!(s.assembled_blocks, 0);
        assert_eq!(s.total_fragments_stored, 0);
        assert_eq!(s.avg_completion_rate, 0.0);
    }

    #[test]
    fn test_stats_with_pending() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("cid1", 0, b"a", 4), 1)
            .expect("ok");
        store
            .store_fragment(make_fragment("cid1", 1, b"b", 4), 2)
            .expect("ok");
        let s = store.stats();
        assert_eq!(s.pending_sets, 1);
        assert_eq!(s.total_fragments_stored, 2);
        let expected_rate = 2.0 / 4.0;
        assert!((s.avg_completion_rate - expected_rate).abs() < 1e-9);
    }

    #[test]
    fn test_stats_assembled_counted() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("cid1", 0, b"z", 1), 1)
            .expect("ok");
        let s = store.stats();
        assert_eq!(s.assembled_blocks, 1);
        assert_eq!(s.pending_sets, 0);
    }

    // -----------------------------------------------------------------------
    // 12. Multiple blocks
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_blocks_independent() {
        let mut store = store_with_defaults();
        for i in 0u32..5 {
            let cid = format!("block-{i}");
            for j in 0u32..3 {
                let data = format!("cid{i}-frag{j}");
                store
                    .store_fragment(
                        make_fragment(&cid, j, data.as_bytes(), 3),
                        (i * 10 + j) as u64,
                    )
                    .expect("ok");
            }
        }
        assert_eq!(store.assembled_count(), 5);
        assert_eq!(store.pending_count(), 0);
    }

    // -----------------------------------------------------------------------
    // 13. Explicit assemble()
    // -----------------------------------------------------------------------

    #[test]
    fn test_explicit_assemble_after_all_received() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("cid1", 0, b"X", 2), 1)
            .expect("ok");
        store
            .store_fragment(make_fragment("cid1", 1, b"Y", 2), 2)
            .expect("ok");
        // Block should already be assembled automatically; explicit assemble returns cached.
        let block = store.assemble("cid1", 99).expect("assembled");
        assert_eq!(block.data, b"XY");
    }

    #[test]
    fn test_explicit_assemble_stores_to_cache() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("cid1", 0, b"P", 2), 1)
            .expect("ok");
        store
            .store_fragment(make_fragment("cid1", 1, b"Q", 2), 2)
            .expect("ok");
        let block = store.assemble("cid1", 10).expect("ok");
        assert_eq!(block.fragment_count, 2);
        // get_assembled should now return data.
        assert_eq!(store.get_assembled("cid1"), Some(b"PQ".as_slice()));
    }

    // -----------------------------------------------------------------------
    // 14. Clear helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_clear_pending() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("a", 0, b"x", 2), 1)
            .expect("ok");
        store
            .store_fragment(make_fragment("b", 0, b"y", 2), 2)
            .expect("ok");
        let dropped = store.clear_pending();
        assert_eq!(dropped, 2);
        assert_eq!(store.pending_count(), 0);
    }

    #[test]
    fn test_clear_assembled() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("a", 0, b"x", 1), 1)
            .expect("ok");
        store
            .store_fragment(make_fragment("b", 0, b"y", 1), 2)
            .expect("ok");
        let dropped = store.clear_assembled();
        assert_eq!(dropped, 2);
        assert_eq!(store.assembled_count(), 0);
    }

    // -----------------------------------------------------------------------
    // 15. Capacity remaining
    // -----------------------------------------------------------------------

    #[test]
    fn test_pending_capacity_remaining() {
        let mut store = BlockFragmentStore::new(5, 16);
        assert_eq!(store.pending_capacity_remaining(), 5);
        store
            .store_fragment(make_fragment("a", 0, b"x", 2), 1)
            .expect("ok");
        assert_eq!(store.pending_capacity_remaining(), 4);
    }

    // -----------------------------------------------------------------------
    // 16. AssembledBlock fields
    // -----------------------------------------------------------------------

    #[test]
    fn test_assembled_block_fragment_count() {
        let mut store = store_with_defaults();
        for i in 0u32..4 {
            store
                .store_fragment(make_fragment("c", i, &[i as u8], 4), i as u64)
                .expect("ok");
        }
        let block = store.get_assembled_block("c").expect("assembled");
        assert_eq!(block.fragment_count, 4);
    }

    #[test]
    fn test_assembled_block_cid_matches() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("myCid", 0, b"z", 1), 1)
            .expect("ok");
        let block = store.get_assembled_block("myCid").expect("assembled");
        assert_eq!(block.cid, "myCid");
    }

    // -----------------------------------------------------------------------
    // 17. Pending CIDs / Assembled CIDs listing
    // -----------------------------------------------------------------------

    #[test]
    fn test_pending_cids_listed() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("p1", 0, b"a", 2), 1)
            .expect("ok");
        store
            .store_fragment(make_fragment("p2", 0, b"b", 2), 2)
            .expect("ok");
        let mut cids = store.pending_cids();
        cids.sort();
        assert_eq!(cids, vec!["p1", "p2"]);
    }

    #[test]
    fn test_assembled_cids_listed() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("done1", 0, b"x", 1), 1)
            .expect("ok");
        store
            .store_fragment(make_fragment("done2", 0, b"y", 1), 2)
            .expect("ok");
        let mut cids = store.assembled_cids();
        cids.sort();
        assert_eq!(cids, vec!["done1", "done2"]);
    }

    // -----------------------------------------------------------------------
    // 18. Single-fragment blocks
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_fragment_block() {
        let mut store = store_with_defaults();
        let data = b"singletons are valid";
        store
            .store_fragment(make_fragment("single", 0, data, 1), 1)
            .expect("ok");
        assert_eq!(store.get_assembled("single"), Some(data.as_slice()));
    }

    // -----------------------------------------------------------------------
    // 19. Large fragment count
    // -----------------------------------------------------------------------

    #[test]
    fn test_large_fragment_count() {
        let mut store = BlockFragmentStore::new(4, 4);
        let total = 256u32;
        for i in 0..total {
            let data = vec![i as u8; 64];
            store
                .store_fragment(make_fragment("big", i, &data, total), i as u64)
                .expect("ok");
        }
        let assembled = store.get_assembled("big").expect("assembled");
        assert_eq!(assembled.len(), 256 * 64);
        // Verify first bytes of each 64-byte chunk.
        for i in 0..total as usize {
            assert_eq!(assembled[i * 64], i as u8);
        }
    }

    // -----------------------------------------------------------------------
    // 20. Idempotent assemble call
    // -----------------------------------------------------------------------

    #[test]
    fn test_assemble_idempotent() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("idem", 0, b"A", 1), 1)
            .expect("ok");
        let b1 = store.assemble("idem", 2).expect("first");
        let b2 = store.assemble("idem", 3).expect("second");
        assert_eq!(b1.data, b2.data);
    }

    // -----------------------------------------------------------------------
    // 21. already-assembled block treated as complete in store_fragment
    // -----------------------------------------------------------------------

    #[test]
    fn test_store_fragment_for_assembled_block_returns_complete() {
        let mut store = store_with_defaults();
        store
            .store_fragment(make_fragment("done", 0, b"X", 1), 1)
            .expect("ok");
        // Now try to store another fragment for the same (already assembled) block.
        let state = store
            .store_fragment(make_fragment("done", 0, b"Y", 1), 2)
            .expect("ok");
        assert_eq!(state, FragmentSetState::Complete);
    }

    // -----------------------------------------------------------------------
    // 22. Stats avg_completion_rate multiple sets
    // -----------------------------------------------------------------------

    #[test]
    fn test_avg_completion_rate_two_sets() {
        let mut store = store_with_defaults();
        // Set 1: 1/4 complete
        store
            .store_fragment(make_fragment("s1", 0, b"a", 4), 1)
            .expect("ok");
        // Set 2: 3/4 complete
        store
            .store_fragment(make_fragment("s2", 0, b"a", 4), 2)
            .expect("ok");
        store
            .store_fragment(make_fragment("s2", 1, b"b", 4), 3)
            .expect("ok");
        store
            .store_fragment(make_fragment("s2", 2, b"c", 4), 4)
            .expect("ok");
        let s = store.stats();
        let expected = (0.25 + 0.75) / 2.0;
        assert!((s.avg_completion_rate - expected).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // 23. FragmentError Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_display_duplicate() {
        let e = FragmentError::DuplicateFragment {
            block_cid: "c".into(),
            index: 2,
        };
        let s = e.to_string();
        assert!(s.contains("duplicate"), "msg: {s}");
    }

    #[test]
    fn test_error_display_store_full() {
        let e = FragmentError::StoreFull;
        let s = e.to_string();
        assert!(s.contains("full"), "msg: {s}");
    }

    // -----------------------------------------------------------------------
    // 24. max_assembled eviction
    // -----------------------------------------------------------------------

    #[test]
    fn test_max_assembled_evicts_oldest() {
        let mut store = BlockFragmentStore::new(16, 2);
        // Assemble three 1-fragment blocks.
        store
            .store_fragment(make_fragment("a", 0, b"A", 1), 1)
            .expect("ok");
        store
            .store_fragment(make_fragment("b", 0, b"B", 1), 2)
            .expect("ok");
        // Both should be in cache.
        assert_eq!(store.assembled_count(), 2);
        // Adding a third should evict the oldest (assembled_at=1, which is "a").
        store
            .store_fragment(make_fragment("c", 0, b"C", 1), 3)
            .expect("ok");
        assert_eq!(store.assembled_count(), 2);
        assert!(
            store.get_assembled("a").is_none(),
            "oldest assembled evicted"
        );
        assert!(store.get_assembled("b").is_some());
        assert!(store.get_assembled("c").is_some());
    }

    // -----------------------------------------------------------------------
    // 25. Verify set with multiple bad indices sorted
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_set_multiple_bad_sorted() {
        let mut set = FragmentSet::new("cid".into(), 4, 0);
        set.fragments.insert(0, make_fragment("cid", 0, b"ok", 4));
        set.fragments
            .insert(1, make_fragment_tampered("cid", 1, b"bad1", 4));
        set.fragments.insert(2, make_fragment("cid", 2, b"ok2", 4));
        set.fragments
            .insert(3, make_fragment_tampered("cid", 3, b"bad3", 4));
        let bad = BlockFragmentStore::verify_set(&set);
        assert_eq!(bad, vec![1u32, 3]);
    }
}
