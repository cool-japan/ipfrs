//! Kademlia-style XOR-metric routing table manager for P2P overlay networks.
//!
//! Implements a full 256-bucket Kademlia routing table where each bucket stores
//! at most `k` entries (default 20) sorted by last-seen time.  A replacement
//! cache holds candidates that were not admitted because the bucket was full.
//!
//! ## Design highlights
//!
//! * `NodeId([u8; 32])` — 256-bit node identifier (SHA-256 hash of public key).
//! * `BucketEntry` — information about a known remote node.
//! * `KBucket` — a single k-bucket with a replacement cache (`VecDeque`).
//! * `RoutingTableManager` — 256 k-buckets indexed by the leading-zero-bit
//!   count of `XOR(local_id, target_id)`.
//!
//! ## Thread safety
//!
//! `RoutingTableManager` is *not* `Sync` by design; callers that need shared
//! access should wrap it in `parking_lot::RwLock` or `tokio::sync::RwLock`.
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_network::routing_table_manager::{NodeId, RoutingTableManager};
//!
//! let local = NodeId([0u8; 32]);
//! let mut rtm = RoutingTableManager::new(local, 20, 3);
//!
//! let mut target_bytes = [0u8; 32];
//! target_bytes[0] = 0x80; // distance bucket 0
//! let target = NodeId(target_bytes);
//!
//! rtm.add_node(target, "127.0.0.1:4001".to_string(), 5, 1000);
//! assert_eq!(rtm.total_nodes(), 1);
//! ```

use std::collections::VecDeque;

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────

/// Default k-bucket size (standard Kademlia value).
pub const DEFAULT_K: usize = 20;

/// Default alpha concurrency parameter.
pub const DEFAULT_ALPHA: usize = 3;

/// Maximum number of entries kept in a bucket's replacement cache.
pub const DEFAULT_REPLACEMENT_CACHE_SIZE: usize = 20;

/// Maximum number of failed queries before a node is evicted.
const MAX_FAILED_QUERIES: u32 = 3;

/// Number of k-buckets (one per bit of the 256-bit node ID).
const NUM_BUCKETS: usize = 256;

// ────────────────────────────────────────────────────────────────────────────
// NodeId
// ────────────────────────────────────────────────────────────────────────────

/// 256-bit node identifier, typically the SHA-256 hash of a public key.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct NodeId(pub [u8; 32]);

impl NodeId {
    /// Create a new `NodeId` from raw bytes.
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Borrow the underlying byte slice.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Return the XOR distance between `self` and `other`.
    pub fn xor_distance(&self, other: &NodeId) -> [u8; 32] {
        xor_distance(self, other)
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// BucketEntry
// ────────────────────────────────────────────────────────────────────────────

/// Metadata stored for every known remote node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BucketEntry {
    /// The remote node's 256-bit identifier.
    pub node_id: NodeId,
    /// Network address (e.g. `/ip4/1.2.3.4/tcp/4001`).
    pub address: String,
    /// Unix timestamp (seconds) when this entry was last contacted.
    pub last_seen: u64,
    /// Observed round-trip time in milliseconds.
    pub rtt_ms: u64,
    /// Number of consecutive failed queries to this node.
    pub failed_queries: u32,
}

impl BucketEntry {
    /// Construct a fresh bucket entry.
    pub fn new(node_id: NodeId, address: String, rtt_ms: u64, last_seen: u64) -> Self {
        Self {
            node_id,
            address,
            last_seen,
            rtt_ms,
            failed_queries: 0,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// KBucket
// ────────────────────────────────────────────────────────────────────────────

/// A single Kademlia k-bucket.
///
/// Entries are stored in LRU order: index 0 is the *least*-recently-seen node,
/// index `len-1` is the *most*-recently-seen node.  When the bucket is full a
/// newly observed node goes into `replacement_cache` rather than the bucket.
#[derive(Clone, Debug)]
pub struct KBucket {
    /// Active entries, ordered from least-recently-seen (front) to
    /// most-recently-seen (back).
    pub entries: Vec<BucketEntry>,
    /// Candidates waiting to replace evicted entries (FIFO).
    pub replacement_cache: VecDeque<BucketEntry>,
    /// Maximum number of active entries.
    pub max_size: usize,
}

impl KBucket {
    /// Create an empty k-bucket with the given maximum size.
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: Vec::with_capacity(max_size),
            replacement_cache: VecDeque::new(),
            max_size,
        }
    }

    /// Returns `true` if the bucket holds no active entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the number of active entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the active-entry list is at capacity.
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.max_size
    }

    /// Find the position of `node_id` in the active entries, if present.
    fn find_entry_index(&self, node_id: &NodeId) -> Option<usize> {
        self.entries.iter().position(|e| &e.node_id == node_id)
    }

    /// Find the position of `node_id` in the replacement cache, if present.
    fn find_cache_index(&self, node_id: &NodeId) -> Option<usize> {
        self.replacement_cache
            .iter()
            .position(|e| &e.node_id == node_id)
    }

    /// Insert or refresh a node in the bucket.
    ///
    /// * If the node already exists in active entries → move to tail (LRU update).
    /// * Else if the bucket is not full → append to tail.
    /// * Else if the node is in the replacement cache → update it there.
    /// * Else → push to the back of the replacement cache (bounded by
    ///   `DEFAULT_REPLACEMENT_CACHE_SIZE`).
    pub fn add(&mut self, entry: BucketEntry) {
        if let Some(idx) = self.find_entry_index(&entry.node_id) {
            // Refresh existing active entry and move it to the tail.
            let mut existing = self.entries.remove(idx);
            existing.last_seen = entry.last_seen;
            existing.rtt_ms = entry.rtt_ms;
            existing.failed_queries = 0; // successful contact resets failures
            self.entries.push(existing);
            return;
        }

        if !self.is_full() {
            self.entries.push(entry);
            return;
        }

        // Bucket is full — manage replacement cache.
        if let Some(idx) = self.find_cache_index(&entry.node_id) {
            // Update existing cache entry in place.
            if let Some(cached) = self.replacement_cache.get_mut(idx) {
                cached.last_seen = entry.last_seen;
                cached.rtt_ms = entry.rtt_ms;
                cached.failed_queries = 0;
            }
        } else {
            // Evict the oldest cache entry if at capacity.
            if self.replacement_cache.len() >= DEFAULT_REPLACEMENT_CACHE_SIZE {
                self.replacement_cache.pop_front();
            }
            self.replacement_cache.push_back(entry);
        }
    }

    /// Remove the node with the given ID from active entries.
    ///
    /// If a replacement candidate exists in the cache it is promoted.
    ///
    /// Returns `true` if the node was found and removed.
    pub fn remove(&mut self, node_id: &NodeId) -> bool {
        if let Some(idx) = self.find_entry_index(node_id) {
            self.entries.remove(idx);
            // Promote the freshest replacement (back of the cache).
            if let Some(replacement) = self.replacement_cache.pop_back() {
                self.entries.push(replacement);
            }
            true
        } else {
            false
        }
    }

    /// Increment the failure counter for a node.
    ///
    /// Returns `true` if the node was evicted due to too many failures.
    pub fn mark_failed(&mut self, node_id: &NodeId, now: u64) -> bool {
        if let Some(idx) = self.find_entry_index(node_id) {
            self.entries[idx].failed_queries += 1;
            if self.entries[idx].failed_queries >= MAX_FAILED_QUERIES {
                self.entries.remove(idx);
                // Promote replacement if available.
                if let Some(replacement) = self.replacement_cache.pop_back() {
                    self.entries.push(replacement);
                }
                return true;
            }
            // Update last_seen even on failure to avoid stale timestamps.
            self.entries[idx].last_seen = now;
        }
        false
    }

    /// Update the RTT and last-seen timestamp for a node.
    pub fn update_rtt(&mut self, node_id: &NodeId, rtt_ms: u64, now: u64) {
        if let Some(idx) = self.find_entry_index(node_id) {
            self.entries[idx].rtt_ms = rtt_ms;
            self.entries[idx].last_seen = now;
            self.entries[idx].failed_queries = 0;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Routing table statistics
// ────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics for the routing table.
#[derive(Clone, Debug)]
pub struct RoutingTableStats {
    /// Total number of active entries across all buckets.
    pub total_nodes: usize,
    /// Number of buckets that hold at least one entry.
    pub non_empty_buckets: usize,
    /// Average RTT (ms) across all active entries, or 0 if the table is empty.
    pub avg_rtt_ms: f64,
    /// Size of the largest bucket.
    pub max_bucket_size: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// Free-standing distance helpers
// ────────────────────────────────────────────────────────────────────────────

/// Compute the 256-bit XOR distance between two node IDs.
pub fn xor_distance(a: &NodeId, b: &NodeId) -> [u8; 32] {
    let mut dist = [0u8; 32];
    for (i, byte) in dist.iter_mut().enumerate() {
        *byte = a.0[i] ^ b.0[i];
    }
    dist
}

/// Return the index of the k-bucket for `target` relative to `local`.
///
/// The index equals the number of *leading zero bits* in `XOR(local, target)`,
/// capped at 255.  A result of 255 means the two IDs are identical (distance 0)
/// and the target is the local node itself.
pub fn bucket_index(local: &NodeId, target: &NodeId) -> usize {
    let dist = xor_distance(local, target);
    let mut leading = 0usize;
    for byte in &dist {
        if *byte == 0 {
            leading += 8;
        } else {
            leading += byte.leading_zeros() as usize;
            break;
        }
    }
    leading.min(255)
}

// ────────────────────────────────────────────────────────────────────────────
// RoutingTableManager
// ────────────────────────────────────────────────────────────────────────────

/// Kademlia-style XOR-metric routing table manager.
///
/// Manages 256 k-buckets (one per bit of the 256-bit node-ID space).  Bucket
/// `i` holds nodes whose XOR distance to the local node starts with exactly
/// `i` leading zero bits.
///
/// # Parameters
///
/// * `k` — maximum number of active entries per bucket (default: 20).
/// * `alpha` — concurrency factor for parallel lookups (default: 3; stored
///   but not used internally — callers read it when launching lookups).
pub struct RoutingTableManager {
    /// The 256 k-buckets.
    pub buckets: Vec<KBucket>,
    /// This node's own identifier.
    pub local_id: NodeId,
    /// Maximum entries per bucket.
    pub k: usize,
    /// Lookup concurrency parameter.
    pub alpha: usize,
}

impl RoutingTableManager {
    /// Create a new routing table manager.
    ///
    /// # Arguments
    ///
    /// * `local_id` — this node's 256-bit identifier.
    /// * `k` — k-bucket size (use `DEFAULT_K` for the standard Kademlia value).
    /// * `alpha` — concurrency factor (use `DEFAULT_ALPHA` for the standard value).
    pub fn new(local_id: NodeId, k: usize, alpha: usize) -> Self {
        let k = k.max(1); // guard against a degenerate k=0
        let buckets = (0..NUM_BUCKETS).map(|_| KBucket::new(k)).collect();
        Self {
            buckets,
            local_id,
            k,
            alpha,
        }
    }

    /// Create a routing table manager with default parameters (`k=20`, `alpha=3`).
    pub fn with_defaults(local_id: NodeId) -> Self {
        Self::new(local_id, DEFAULT_K, DEFAULT_ALPHA)
    }

    // ── Routing table mutations ───────────────────────────────────────────────

    /// Insert or refresh a node in the routing table.
    ///
    /// * If the node already appears in its bucket → LRU update (move to tail).
    /// * If the bucket is not full → append.
    /// * If the bucket is full → put into the replacement cache.
    ///
    /// Silently ignores attempts to add the local node itself.
    pub fn add_node(&mut self, node_id: NodeId, address: String, rtt_ms: u64, now: u64) {
        if node_id == self.local_id {
            return;
        }
        let idx = bucket_index(&self.local_id, &node_id);
        let entry = BucketEntry::new(node_id, address, rtt_ms, now);
        self.buckets[idx].add(entry);
    }

    /// Remove a node from the routing table.
    ///
    /// If the node was in a bucket's active list a replacement from the cache
    /// is promoted automatically.
    ///
    /// Returns `true` if the node was found in an active bucket.
    pub fn remove_node(&mut self, node_id: &NodeId) -> bool {
        let idx = bucket_index(&self.local_id, node_id);
        self.buckets[idx].remove(node_id)
    }

    /// Record a failed query for a node.
    ///
    /// After `MAX_FAILED_QUERIES` (3) consecutive failures the node is evicted
    /// and a replacement is promoted from the cache.
    pub fn mark_failed(&mut self, node_id: &NodeId, now: u64) {
        let idx = bucket_index(&self.local_id, node_id);
        self.buckets[idx].mark_failed(node_id, now);
    }

    /// Update the RTT observation and last-seen timestamp for a node.
    pub fn update_rtt(&mut self, node_id: &NodeId, rtt_ms: u64, now: u64) {
        let idx = bucket_index(&self.local_id, node_id);
        self.buckets[idx].update_rtt(node_id, rtt_ms, now);
    }

    // ── Lookup ────────────────────────────────────────────────────────────────

    /// Return up to `count` entries closest to `target` measured by XOR distance.
    ///
    /// The result is sorted ascending by XOR distance (closest first).
    pub fn find_closest(&self, target: &NodeId, count: usize) -> Vec<BucketEntry> {
        // Collect every active entry together with its XOR distance.
        let mut candidates: Vec<([u8; 32], BucketEntry)> = self
            .buckets
            .iter()
            .flat_map(|b| b.entries.iter())
            .map(|e| (xor_distance(target, &e.node_id), e.clone()))
            .collect();

        // Sort by distance (lexicographic on the 32-byte XOR result is correct).
        candidates.sort_by_key(|a| a.0);
        candidates.truncate(count);
        candidates.into_iter().map(|(_, e)| e).collect()
    }

    // ── Observation helpers ───────────────────────────────────────────────────

    /// Return the number of active entries in each of the 256 buckets.
    pub fn bucket_sizes(&self) -> Vec<usize> {
        self.buckets.iter().map(|b| b.len()).collect()
    }

    /// Return the total number of active entries across all buckets.
    pub fn total_nodes(&self) -> usize {
        self.buckets.iter().map(|b| b.len()).sum()
    }

    /// Compute aggregate statistics for the routing table.
    pub fn stats(&self) -> RoutingTableStats {
        let total_nodes = self.total_nodes();
        let non_empty_buckets = self.buckets.iter().filter(|b| !b.is_empty()).count();
        let max_bucket_size = self.buckets.iter().map(|b| b.len()).max().unwrap_or(0);

        let avg_rtt_ms = if total_nodes == 0 {
            0.0
        } else {
            let rtt_sum: u64 = self
                .buckets
                .iter()
                .flat_map(|b| b.entries.iter())
                .map(|e| e.rtt_ms)
                .sum();
            rtt_sum as f64 / total_nodes as f64
        };

        RoutingTableStats {
            total_nodes,
            non_empty_buckets,
            avg_rtt_ms,
            max_bucket_size,
        }
    }

    // ── Convenience accessors ─────────────────────────────────────────────────

    /// Return the k-bucket index for a given target node ID.
    pub fn bucket_index_for(&self, target: &NodeId) -> usize {
        bucket_index(&self.local_id, target)
    }

    /// Iterate over all active `BucketEntry` records.
    pub fn iter_entries(&self) -> impl Iterator<Item = &BucketEntry> {
        self.buckets.iter().flat_map(|b| b.entries.iter())
    }

    /// Return `true` if the routing table contains no active entries.
    pub fn is_empty(&self) -> bool {
        self.buckets.iter().all(|b| b.is_empty())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        bucket_index, xor_distance, BucketEntry, KBucket, NodeId, RoutingTableManager, DEFAULT_K,
        DEFAULT_REPLACEMENT_CACHE_SIZE, MAX_FAILED_QUERIES, NUM_BUCKETS,
    };

    // ── xorshift64 PRNG (no rand crate) ──────────────────────────────────────

    struct Xorshift64 {
        state: u64,
    }

    impl Xorshift64 {
        fn new(seed: u64) -> Self {
            // Avoid a zero state which would make the generator degenerate.
            Self {
                state: if seed == 0 { 0xdeadbeef_cafebabe } else { seed },
            }
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }

        fn next_node_id(&mut self) -> NodeId {
            let mut bytes = [0u8; 32];
            for chunk in bytes.chunks_exact_mut(8) {
                let val = self.next_u64().to_le_bytes();
                chunk.copy_from_slice(&val);
            }
            NodeId(bytes)
        }

        fn next_rtt(&mut self) -> u64 {
            (self.next_u64() % 500) + 1
        }

        fn next_ts(&mut self) -> u64 {
            (self.next_u64() % 1_000_000) + 1_000_000
        }
    }

    // ── Helper constructors ───────────────────────────────────────────────────

    fn zero_id() -> NodeId {
        NodeId([0u8; 32])
    }

    /// Build a NodeId where byte `byte_idx` has `value`, all others 0.
    fn id_with_byte(byte_idx: usize, value: u8) -> NodeId {
        let mut bytes = [0u8; 32];
        bytes[byte_idx] = value;
        NodeId(bytes)
    }

    fn make_rtm(k: usize) -> RoutingTableManager {
        RoutingTableManager::new(zero_id(), k, 3)
    }

    fn dummy_entry(id: NodeId, rtt: u64, ts: u64) -> BucketEntry {
        BucketEntry::new(id, format!("127.0.0.1:{}", rtt), rtt, ts)
    }

    // ── 1. xor_distance basics ────────────────────────────────────────────────

    #[test]
    fn test_xor_distance_identical() {
        let id = NodeId([0xABu8; 32]);
        let dist = xor_distance(&id, &id);
        assert_eq!(dist, [0u8; 32]);
    }

    #[test]
    fn test_xor_distance_complementary() {
        let a = NodeId([0xFF; 32]);
        let b = NodeId([0x00; 32]);
        let dist = xor_distance(&a, &b);
        assert_eq!(dist, [0xFF; 32]);
    }

    #[test]
    fn test_xor_distance_symmetry() {
        let mut rng = Xorshift64::new(42);
        let a = rng.next_node_id();
        let b = rng.next_node_id();
        assert_eq!(xor_distance(&a, &b), xor_distance(&b, &a));
    }

    #[test]
    fn test_xor_distance_triangle_inequality() {
        // XOR metric satisfies d(a,c) <= d(a,b) XOR d(b,c) byte-wise.
        // Simpler check: d(a,a) == 0.
        let mut rng = Xorshift64::new(7);
        let a = rng.next_node_id();
        assert_eq!(xor_distance(&a, &a), [0u8; 32]);
    }

    // ── 2. bucket_index ───────────────────────────────────────────────────────

    #[test]
    fn test_bucket_index_equal_ids() {
        let id = NodeId([0u8; 32]);
        // XOR is all zeros → 256 leading zeros, capped at 255
        assert_eq!(bucket_index(&id, &id), 255);
    }

    #[test]
    fn test_bucket_index_first_bit_differs() {
        let local = zero_id();
        let mut target_bytes = [0u8; 32];
        target_bytes[0] = 0x80; // MSB set → XOR has MSB set → 0 leading zeros
        let target = NodeId(target_bytes);
        assert_eq!(bucket_index(&local, &target), 0);
    }

    #[test]
    fn test_bucket_index_second_bit_differs() {
        let local = zero_id();
        let mut target_bytes = [0u8; 32];
        target_bytes[0] = 0x40; // second bit set → 1 leading zero
        let target = NodeId(target_bytes);
        assert_eq!(bucket_index(&local, &target), 1);
    }

    #[test]
    fn test_bucket_index_second_byte() {
        let local = zero_id();
        let mut target_bytes = [0u8; 32];
        target_bytes[1] = 0x80; // first byte zero, second byte MSB → 8 leading zeros
        let target = NodeId(target_bytes);
        assert_eq!(bucket_index(&local, &target), 8);
    }

    #[test]
    fn test_bucket_index_last_byte() {
        let local = zero_id();
        let mut target_bytes = [0u8; 32];
        target_bytes[31] = 0x01; // only last bit differs → 255 leading zeros, but capped
        let target = NodeId(target_bytes);
        assert_eq!(bucket_index(&local, &target), 255);
    }

    // ── 3. KBucket – add ─────────────────────────────────────────────────────

    #[test]
    fn test_kbucket_add_up_to_capacity() {
        let mut bucket = KBucket::new(3);
        for i in 0..3u8 {
            let id = id_with_byte(0, i + 1);
            bucket.add(dummy_entry(id, 10, 1000));
        }
        assert_eq!(bucket.len(), 3);
        assert!(bucket.replacement_cache.is_empty());
    }

    #[test]
    fn test_kbucket_overflow_goes_to_cache() {
        let mut bucket = KBucket::new(2);
        bucket.add(dummy_entry(id_with_byte(0, 1), 10, 1000));
        bucket.add(dummy_entry(id_with_byte(0, 2), 10, 1001));
        bucket.add(dummy_entry(id_with_byte(0, 3), 10, 1002)); // goes to cache
        assert_eq!(bucket.len(), 2);
        assert_eq!(bucket.replacement_cache.len(), 1);
    }

    #[test]
    fn test_kbucket_lru_move_to_tail() {
        let mut bucket = KBucket::new(4);
        let id_a = id_with_byte(0, 1);
        let id_b = id_with_byte(0, 2);
        bucket.add(dummy_entry(id_a, 10, 1000));
        bucket.add(dummy_entry(id_b, 10, 1001));
        // Re-add id_a with a newer timestamp
        bucket.add(dummy_entry(id_a, 20, 2000));
        // id_a should now be at the tail (index 1)
        assert_eq!(bucket.entries.last().map(|e| e.node_id), Some(id_a));
        assert_eq!(bucket.entries.len(), 2);
    }

    #[test]
    fn test_kbucket_lru_resets_failures() {
        let mut bucket = KBucket::new(4);
        let id = id_with_byte(0, 1);
        bucket.add(dummy_entry(id, 10, 1000));
        // Simulate a failure
        bucket.entries[0].failed_queries = 2;
        // Re-add the same node (successful contact)
        bucket.add(dummy_entry(id, 15, 2000));
        assert_eq!(bucket.entries.last().map(|e| e.failed_queries), Some(0));
    }

    // ── 4. KBucket – remove ──────────────────────────────────────────────────

    #[test]
    fn test_kbucket_remove_existing() {
        let mut bucket = KBucket::new(4);
        let id = id_with_byte(0, 1);
        bucket.add(dummy_entry(id, 10, 1000));
        assert!(bucket.remove(&id));
        assert!(bucket.is_empty());
    }

    #[test]
    fn test_kbucket_remove_absent() {
        let mut bucket = KBucket::new(4);
        assert!(!bucket.remove(&id_with_byte(0, 99)));
    }

    #[test]
    fn test_kbucket_remove_promotes_replacement() {
        let mut bucket = KBucket::new(2);
        let id_a = id_with_byte(0, 1);
        let id_b = id_with_byte(0, 2);
        let id_c = id_with_byte(0, 3); // will go to cache
        bucket.add(dummy_entry(id_a, 10, 1000));
        bucket.add(dummy_entry(id_b, 10, 1001));
        bucket.add(dummy_entry(id_c, 10, 1002));
        assert_eq!(bucket.replacement_cache.len(), 1);
        bucket.remove(&id_a);
        // id_c should have been promoted
        assert_eq!(bucket.len(), 2);
        assert!(bucket.entries.iter().any(|e| e.node_id == id_c));
        assert!(bucket.replacement_cache.is_empty());
    }

    // ── 5. KBucket – mark_failed ─────────────────────────────────────────────

    #[test]
    fn test_kbucket_mark_failed_eviction() {
        let mut bucket = KBucket::new(4);
        let id = id_with_byte(0, 1);
        bucket.add(dummy_entry(id, 10, 1000));
        // Two failures – still present.
        bucket.mark_failed(&id, 1001);
        bucket.mark_failed(&id, 1002);
        assert!(!bucket.is_empty());
        // Third failure – evicted.
        let evicted = bucket.mark_failed(&id, 1003);
        assert!(evicted);
        assert!(bucket.is_empty());
    }

    #[test]
    fn test_kbucket_mark_failed_promotes_replacement() {
        let mut bucket = KBucket::new(1);
        let id_a = id_with_byte(0, 1);
        let id_b = id_with_byte(0, 2);
        bucket.add(dummy_entry(id_a, 10, 1000));
        bucket.add(dummy_entry(id_b, 10, 1001)); // goes to cache
                                                 // Fail id_a three times
        for ts in 1..=3u64 {
            bucket.mark_failed(&id_a, 1000 + ts);
        }
        // id_b should have been promoted
        assert_eq!(bucket.len(), 1);
        assert_eq!(bucket.entries[0].node_id, id_b);
    }

    // ── 6. KBucket – update_rtt ──────────────────────────────────────────────

    #[test]
    fn test_kbucket_update_rtt() {
        let mut bucket = KBucket::new(4);
        let id = id_with_byte(0, 1);
        bucket.add(dummy_entry(id, 100, 1000));
        bucket.update_rtt(&id, 50, 2000);
        let entry = bucket
            .entries
            .iter()
            .find(|e| e.node_id == id)
            .expect("test: entry should be found in bucket after add");
        assert_eq!(entry.rtt_ms, 50);
        assert_eq!(entry.last_seen, 2000);
        assert_eq!(entry.failed_queries, 0);
    }

    // ── 7. RoutingTableManager – add_node ────────────────────────────────────

    #[test]
    fn test_rtm_add_node_basic() {
        let mut rtm = make_rtm(DEFAULT_K);
        let id = id_with_byte(0, 0x80); // bucket 0
        rtm.add_node(id, "addr".to_string(), 10, 1000);
        assert_eq!(rtm.total_nodes(), 1);
    }

    #[test]
    fn test_rtm_add_node_ignores_local() {
        let mut rtm = make_rtm(DEFAULT_K);
        rtm.add_node(zero_id(), "self".to_string(), 0, 0);
        assert_eq!(rtm.total_nodes(), 0);
    }

    #[test]
    fn test_rtm_add_many_nodes_distributed() {
        let mut rng = Xorshift64::new(1234);
        let local = rng.next_node_id();
        let mut rtm = RoutingTableManager::new(local, DEFAULT_K, 3);
        for _ in 0..100 {
            let id = rng.next_node_id();
            rtm.add_node(id, "a".to_string(), rng.next_rtt(), rng.next_ts());
        }
        assert!(rtm.total_nodes() > 0);
        // Nodes should be spread across multiple buckets.
        let non_empty = rtm.bucket_sizes().iter().filter(|&&s| s > 0).count();
        assert!(non_empty > 1, "expected multiple non-empty buckets");
    }

    // ── 8. RoutingTableManager – remove_node ─────────────────────────────────

    #[test]
    fn test_rtm_remove_node_present() {
        let mut rtm = make_rtm(DEFAULT_K);
        let id = id_with_byte(0, 0x80);
        rtm.add_node(id, "addr".to_string(), 10, 1000);
        assert!(rtm.remove_node(&id));
        assert_eq!(rtm.total_nodes(), 0);
    }

    #[test]
    fn test_rtm_remove_node_absent() {
        let mut rtm = make_rtm(DEFAULT_K);
        assert!(!rtm.remove_node(&id_with_byte(0, 1)));
    }

    // ── 9. RoutingTableManager – find_closest ────────────────────────────────

    #[test]
    fn test_rtm_find_closest_empty() {
        let rtm = make_rtm(DEFAULT_K);
        let target = id_with_byte(0, 0x80);
        let result = rtm.find_closest(&target, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_rtm_find_closest_count_limit() {
        let mut rng = Xorshift64::new(99);
        let local = rng.next_node_id();
        let mut rtm = RoutingTableManager::new(local, DEFAULT_K, 3);
        for _ in 0..50 {
            let id = rng.next_node_id();
            rtm.add_node(id, "a".to_string(), rng.next_rtt(), rng.next_ts());
        }
        let target = rng.next_node_id();
        let result = rtm.find_closest(&target, 10);
        assert!(result.len() <= 10);
    }

    #[test]
    fn test_rtm_find_closest_sorted_order() {
        let mut rng = Xorshift64::new(777);
        let local = rng.next_node_id();
        let mut rtm = RoutingTableManager::new(local, DEFAULT_K, 3);
        let target = rng.next_node_id();
        for _ in 0..30 {
            let id = rng.next_node_id();
            rtm.add_node(id, "a".to_string(), rng.next_rtt(), rng.next_ts());
        }
        let result = rtm.find_closest(&target, 10);
        // Verify ascending XOR distance order.
        let distances: Vec<[u8; 32]> = result
            .iter()
            .map(|e| xor_distance(&target, &e.node_id))
            .collect();
        for pair in distances.windows(2) {
            assert!(pair[0] <= pair[1], "result must be sorted by XOR distance");
        }
    }

    #[test]
    fn test_rtm_find_closest_returns_nearest() {
        let local = zero_id();
        let mut rtm = RoutingTableManager::new(local, DEFAULT_K, 3);
        // id_near is in bucket 0 (MSB differs), id_far is in bucket 1.
        let id_near = id_with_byte(0, 0x80); // distance starts with 10000000…
        let id_far = id_with_byte(0, 0x40); // distance starts with 01000000…
        rtm.add_node(id_near, "near".to_string(), 5, 1000);
        rtm.add_node(id_far, "far".to_string(), 5, 1000);
        // When the target is id_near itself, id_near should be returned first.
        let result = rtm.find_closest(&id_near, 2);
        assert_eq!(result[0].node_id, id_near);
    }

    // ── 10. RoutingTableManager – mark_failed ────────────────────────────────

    #[test]
    fn test_rtm_mark_failed_eviction() {
        let mut rtm = make_rtm(DEFAULT_K);
        let id = id_with_byte(0, 0x80);
        rtm.add_node(id, "a".to_string(), 10, 1000);
        for ts in 1..=3u64 {
            rtm.mark_failed(&id, 1000 + ts);
        }
        assert_eq!(rtm.total_nodes(), 0);
    }

    #[test]
    fn test_rtm_mark_failed_noop_on_absent() {
        let mut rtm = make_rtm(DEFAULT_K);
        // Should not panic or change anything.
        rtm.mark_failed(&id_with_byte(0, 99), 9999);
        assert_eq!(rtm.total_nodes(), 0);
    }

    // ── 11. RoutingTableManager – update_rtt ─────────────────────────────────

    #[test]
    fn test_rtm_update_rtt() {
        let mut rtm = make_rtm(DEFAULT_K);
        let id = id_with_byte(0, 0x80);
        rtm.add_node(id, "a".to_string(), 100, 1000);
        rtm.update_rtt(&id, 25, 2000);
        let entry = rtm
            .iter_entries()
            .find(|e| e.node_id == id)
            .expect("test: entry should be found in routing table");
        assert_eq!(entry.rtt_ms, 25);
        assert_eq!(entry.last_seen, 2000);
    }

    #[test]
    fn test_rtm_update_rtt_noop_on_absent() {
        let mut rtm = make_rtm(DEFAULT_K);
        // Should not panic.
        rtm.update_rtt(&id_with_byte(0, 7), 50, 9999);
    }

    // ── 12. RoutingTableManager – bucket_sizes / total_nodes ─────────────────

    #[test]
    fn test_rtm_bucket_sizes_length() {
        let rtm = make_rtm(DEFAULT_K);
        assert_eq!(rtm.bucket_sizes().len(), NUM_BUCKETS);
    }

    #[test]
    fn test_rtm_total_nodes_zero_on_empty() {
        let rtm = make_rtm(DEFAULT_K);
        assert_eq!(rtm.total_nodes(), 0);
    }

    #[test]
    fn test_rtm_total_nodes_after_add() {
        let mut rtm = make_rtm(DEFAULT_K);
        for i in 1u8..=5 {
            rtm.add_node(id_with_byte(0, i * 0x10), "a".to_string(), 10, 1000);
        }
        assert_eq!(rtm.total_nodes(), 5);
    }

    // ── 13. RoutingTableManager – stats ──────────────────────────────────────

    #[test]
    fn test_rtm_stats_empty() {
        let rtm = make_rtm(DEFAULT_K);
        let s = rtm.stats();
        assert_eq!(s.total_nodes, 0);
        assert_eq!(s.non_empty_buckets, 0);
        assert_eq!(s.avg_rtt_ms, 0.0);
        assert_eq!(s.max_bucket_size, 0);
    }

    #[test]
    fn test_rtm_stats_single_node() {
        let mut rtm = make_rtm(DEFAULT_K);
        rtm.add_node(id_with_byte(0, 0x80), "a".to_string(), 42, 1000);
        let s = rtm.stats();
        assert_eq!(s.total_nodes, 1);
        assert_eq!(s.non_empty_buckets, 1);
        assert_eq!(s.avg_rtt_ms, 42.0);
        assert_eq!(s.max_bucket_size, 1);
    }

    #[test]
    fn test_rtm_stats_avg_rtt() {
        let mut rtm = make_rtm(DEFAULT_K);
        // Two nodes in different buckets with RTTs 100 and 200
        rtm.add_node(id_with_byte(0, 0x80), "a".to_string(), 100, 1000);
        rtm.add_node(id_with_byte(0, 0x40), "b".to_string(), 200, 1001);
        let s = rtm.stats();
        assert_eq!(s.total_nodes, 2);
        assert!((s.avg_rtt_ms - 150.0).abs() < f64::EPSILON);
    }

    // ── 14. Replacement cache bounds ─────────────────────────────────────────

    #[test]
    fn test_replacement_cache_bounded() {
        let mut bucket = KBucket::new(1);
        // Fill the one active slot.
        bucket.add(dummy_entry(id_with_byte(0, 0), 10, 1000));
        // Add more than DEFAULT_REPLACEMENT_CACHE_SIZE to the cache.
        for i in 1u8..=(DEFAULT_REPLACEMENT_CACHE_SIZE as u8 + 10) {
            bucket.add(dummy_entry(id_with_byte(0, i), 10, 1000));
        }
        assert!(
            bucket.replacement_cache.len() <= DEFAULT_REPLACEMENT_CACHE_SIZE,
            "replacement cache must not exceed its maximum size"
        );
    }

    // ── 15. Multiple replacements ─────────────────────────────────────────────

    #[test]
    fn test_multiple_replacements_sequential() {
        let mut bucket = KBucket::new(2);
        let id_a = id_with_byte(0, 1);
        let id_b = id_with_byte(0, 2);
        let id_c = id_with_byte(0, 3);
        let id_d = id_with_byte(0, 4);
        bucket.add(dummy_entry(id_a, 10, 1000));
        bucket.add(dummy_entry(id_b, 10, 1001));
        bucket.add(dummy_entry(id_c, 10, 1002)); // cache
        bucket.add(dummy_entry(id_d, 10, 1003)); // cache
                                                 // Remove id_a → id_d (back of cache) promoted.
        bucket.remove(&id_a);
        assert_eq!(bucket.len(), 2);
        assert!(bucket.entries.iter().any(|e| e.node_id == id_d));
        // Remove id_b → id_c promoted.
        bucket.remove(&id_b);
        assert_eq!(bucket.len(), 2);
        assert!(bucket.entries.iter().any(|e| e.node_id == id_c));
    }

    // ── 16. RoutingTableManager – is_empty ───────────────────────────────────

    #[test]
    fn test_rtm_is_empty_initial() {
        let rtm = make_rtm(DEFAULT_K);
        assert!(rtm.is_empty());
    }

    #[test]
    fn test_rtm_is_not_empty_after_add() {
        let mut rtm = make_rtm(DEFAULT_K);
        rtm.add_node(id_with_byte(0, 1), "x".to_string(), 10, 0);
        assert!(!rtm.is_empty());
    }

    // ── 17. with_defaults constructor ─────────────────────────────────────────

    #[test]
    fn test_rtm_with_defaults() {
        let rtm = RoutingTableManager::with_defaults(zero_id());
        assert_eq!(rtm.k, DEFAULT_K);
        assert_eq!(rtm.alpha, super::DEFAULT_ALPHA);
        assert_eq!(rtm.buckets.len(), NUM_BUCKETS);
    }

    // ── 18. bucket_index_for ──────────────────────────────────────────────────

    #[test]
    fn test_rtm_bucket_index_for_consistency() {
        let local = id_with_byte(1, 0xAB);
        let rtm = RoutingTableManager::new(local, DEFAULT_K, 3);
        let target = id_with_byte(1, 0x00);
        let idx = rtm.bucket_index_for(&target);
        assert_eq!(idx, bucket_index(&local, &target));
    }

    // ── 19. Large-scale stress test ───────────────────────────────────────────

    #[test]
    fn test_rtm_stress_add_remove_find() {
        let mut rng = Xorshift64::new(0xC0FFEE);
        let local = rng.next_node_id();
        let mut rtm = RoutingTableManager::new(local, 5, 3);
        let mut ids: Vec<NodeId> = Vec::new();

        // Insert 200 nodes.
        for _ in 0..200 {
            let id = rng.next_node_id();
            rtm.add_node(id, "x".to_string(), rng.next_rtt(), rng.next_ts());
            ids.push(id);
        }

        // Remove every other node; they may have been replaced by cache entries
        // so some removals might return false — that is fine.
        for id in ids.iter().step_by(2) {
            let _ = rtm.remove_node(id);
        }

        // find_closest should always return at most k results.
        let target = rng.next_node_id();
        let result = rtm.find_closest(&target, 5);
        assert!(result.len() <= 5);
    }

    // ── 20. Duplicate add via cache ───────────────────────────────────────────

    #[test]
    fn test_kbucket_duplicate_in_cache_updated() {
        let mut bucket = KBucket::new(1);
        let id_a = id_with_byte(0, 1);
        let id_b = id_with_byte(0, 2);
        bucket.add(dummy_entry(id_a, 10, 1000));
        bucket.add(dummy_entry(id_b, 10, 1001)); // goes to cache
                                                 // Update id_b in cache via add.
        bucket.add(dummy_entry(id_b, 99, 9999));
        // Cache should still have exactly one entry for id_b with updated RTT.
        assert_eq!(bucket.replacement_cache.len(), 1);
        let cached = bucket
            .replacement_cache
            .front()
            .expect("test: replacement cache should have an entry after overflow add");
        assert_eq!(cached.node_id, id_b);
        assert_eq!(cached.rtt_ms, 99);
    }

    // ── 21. find_closest does not include local node ──────────────────────────

    #[test]
    fn test_rtm_find_closest_excludes_local() {
        let local = zero_id();
        let mut rtm = RoutingTableManager::new(local, DEFAULT_K, 3);
        // add_node silently drops the local id.
        rtm.add_node(local, "self".to_string(), 0, 0);
        let result = rtm.find_closest(&local, 20);
        assert!(result.iter().all(|e| e.node_id != local));
    }

    // ── 22. NodeId Display ────────────────────────────────────────────────────

    #[test]
    fn test_node_id_display() {
        let id = NodeId([0xABu8; 32]);
        let s = id.to_string();
        assert_eq!(s.len(), 64);
        assert!(s.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // ── 23. NodeId default ────────────────────────────────────────────────────

    #[test]
    fn test_node_id_default() {
        let id = NodeId::default();
        assert_eq!(id.0, [0u8; 32]);
    }

    // ── 24. NodeId as_bytes ───────────────────────────────────────────────────

    #[test]
    fn test_node_id_as_bytes_round_trip() {
        let bytes = [0x42u8; 32];
        let id = NodeId::new(bytes);
        assert_eq!(id.as_bytes(), &bytes);
    }

    // ── 25. Bucket index determinism ─────────────────────────────────────────

    #[test]
    fn test_bucket_index_deterministic() {
        let local = id_with_byte(5, 0xDE);
        let target = id_with_byte(5, 0xAD);
        let idx1 = bucket_index(&local, &target);
        let idx2 = bucket_index(&local, &target);
        assert_eq!(idx1, idx2);
    }

    // ── 26. RTM – update_rtt clears failures ─────────────────────────────────

    #[test]
    fn test_rtm_update_rtt_clears_failed_queries() {
        let mut rtm = make_rtm(DEFAULT_K);
        let id = id_with_byte(0, 0x80);
        rtm.add_node(id, "a".to_string(), 100, 1000);
        // Simulate two failures without eviction.
        rtm.mark_failed(&id, 1001);
        rtm.mark_failed(&id, 1002);
        // Successful contact resets failed_queries.
        rtm.update_rtt(&id, 50, 2000);
        let entry = rtm
            .iter_entries()
            .find(|e| e.node_id == id)
            .expect("test: node should still be in routing table after non-evicting failures");
        assert_eq!(entry.failed_queries, 0);
    }

    // ── 27. KBucket – is_full ────────────────────────────────────────────────

    #[test]
    fn test_kbucket_is_full() {
        let mut bucket = KBucket::new(2);
        assert!(!bucket.is_full());
        bucket.add(dummy_entry(id_with_byte(0, 1), 10, 1000));
        assert!(!bucket.is_full());
        bucket.add(dummy_entry(id_with_byte(0, 2), 10, 1001));
        assert!(bucket.is_full());
    }

    // ── 28. find_closest returns fewer than count when table is small ─────────

    #[test]
    fn test_find_closest_fewer_than_requested() {
        let mut rtm = make_rtm(DEFAULT_K);
        rtm.add_node(id_with_byte(0, 0x80), "a".to_string(), 10, 1000);
        rtm.add_node(id_with_byte(0, 0x40), "b".to_string(), 10, 1001);
        let result = rtm.find_closest(&id_with_byte(0, 0x80), 100);
        assert_eq!(result.len(), 2); // only 2 nodes in the table
    }

    // ── 29. MAX_FAILED_QUERIES constant value ────────────────────────────────

    #[test]
    fn test_max_failed_queries_constant() {
        // Standard Kademlia uses 3; verify our constant matches.
        assert_eq!(MAX_FAILED_QUERIES, 3);
    }

    // ── 30. Fuzz-style: random seed reproducibility ───────────────────────────

    #[test]
    fn test_xorshift64_reproducible() {
        let mut rng1 = Xorshift64::new(12345);
        let mut rng2 = Xorshift64::new(12345);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    // ── 31. RTM – k=1 edge case ───────────────────────────────────────────────

    #[test]
    fn test_rtm_k1_single_slot() {
        let local = zero_id();
        let mut rtm = RoutingTableManager::new(local, 1, 3);
        let id_a = id_with_byte(0, 0x80);
        let id_b = id_with_byte(0, 0xC0); // same bucket (bucket 0, MSB differs)
        rtm.add_node(id_a, "a".to_string(), 10, 1000);
        rtm.add_node(id_b, "b".to_string(), 10, 1001); // goes to cache
        assert_eq!(rtm.total_nodes(), 1); // only one active
    }

    // ── 32. iter_entries ──────────────────────────────────────────────────────

    #[test]
    fn test_rtm_iter_entries_count() {
        let mut rtm = make_rtm(DEFAULT_K);
        for i in 1u8..=8 {
            rtm.add_node(
                id_with_byte(0, i * 0x10),
                "x".to_string(),
                10,
                i as u64 * 100,
            );
        }
        let count = rtm.iter_entries().count();
        assert_eq!(count, rtm.total_nodes());
    }
}
