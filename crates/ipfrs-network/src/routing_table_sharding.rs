//! Sharded DHT routing table that distributes peers across multiple shards
//! based on XOR-distance partitioning, enabling parallel lookups and reduced
//! contention under high peer churn.
//!
//! ## Design overview
//!
//! * `NodeId([u8; 32])` — 256-bit node identifier.  Shard assignment uses
//!   `node_id.0[0] % num_shards` so that the bit-prefix distribution matches
//!   the Kademlia bucket model while keeping O(1) assignment.
//! * `RoutingTableSharding` stores one `HashMap<NodeId, RoutingEntry>` per
//!   shard and never needs a global lock for single-shard operations.
//! * Eviction is pluggable via `EvictionPolicy` — choose the least-recently-
//!   seen node, the highest-RTT node, or a deterministic pseudo-random node
//!   (xorshift64 seeded at construction time).
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_network::routing_table_sharding::{
//!     EvictionPolicy, NodeId as RtsNodeId, RoutingEntry as RtsRoutingEntry,
//!     RoutingTableSharding, ShardConfig,
//! };
//!
//! let cfg = ShardConfig::default();
//! let mut table = RoutingTableSharding::new(cfg);
//!
//! let id = RtsNodeId([1u8; 32]);
//! let entry = RtsRoutingEntry {
//!     node_id: id.clone(),
//!     addr: "127.0.0.1:4001".to_string(),
//!     last_seen: 1_000,
//!     rtt_ms: 10,
//!     shard: table.shard_for(&id),
//! };
//! let evicted = table.insert(entry, 1_000);
//! assert!(!evicted);
//! assert_eq!(table.total_entries(), 1);
//! ```

use std::cell::Cell;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// NodeId
// ─────────────────────────────────────────────────────────────────────────────

/// 256-bit node identifier used for XOR-distance routing.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub [u8; 32]);

impl std::fmt::Debug for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NodeId(")?;
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        write!(f, ")")
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

impl NodeId {
    /// Compute byte-wise XOR distance between `self` and `other`.
    #[inline]
    pub fn xor_distance(&self, other: &NodeId) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (o, (&a, &b)) in out.iter_mut().zip(self.0.iter().zip(other.0.iter())) {
            *o = a ^ b;
        }
        out
    }

    /// Count the number of leading zero *bits* in the raw bytes.
    ///
    /// This is used as the Kademlia bucket index.
    #[inline]
    pub fn leading_zeros(&self) -> u32 {
        let mut count = 0u32;
        for &byte in &self.0 {
            if byte == 0 {
                count += 8;
            } else {
                count += byte.leading_zeros();
                break;
            }
        }
        count
    }

    /// Parse a 64-character lower-case hex string into a `NodeId`.
    ///
    /// Returns `None` if the string is not exactly 64 hex characters.
    pub fn from_str_hex(s: &str) -> Option<NodeId> {
        let s = s.trim();
        if s.len() != 64 {
            return None;
        }
        let mut bytes = [0u8; 32];
        for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
            let hi = hex_nibble(chunk[0])?;
            let lo = hex_nibble(chunk[1])?;
            bytes[i] = (hi << 4) | lo;
        }
        Some(NodeId(bytes))
    }
}

/// Convert a single ASCII hex character to its nibble value.
#[inline]
fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ShardId
// ─────────────────────────────────────────────────────────────────────────────

/// Index of a shard in `[0, num_shards)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ShardId(pub u8);

// ─────────────────────────────────────────────────────────────────────────────
// RoutingEntry
// ─────────────────────────────────────────────────────────────────────────────

/// A single peer record stored inside a shard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutingEntry {
    /// The peer's 256-bit identifier.
    pub node_id: NodeId,
    /// Network address string (e.g. `"127.0.0.1:4001"`).
    pub addr: String,
    /// Unix-epoch millisecond timestamp of the last successful contact.
    pub last_seen: u64,
    /// Observed round-trip time in milliseconds.
    pub rtt_ms: u32,
    /// Pre-computed shard assignment for this entry.
    pub shard: ShardId,
}

// ─────────────────────────────────────────────────────────────────────────────
// EvictionPolicy
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy used to select the victim when a shard is full.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Evict the peer that has not been seen for the longest time.
    #[default]
    LeastRecentlySeen,
    /// Evict the peer with the highest observed RTT.
    HighestRtt,
    /// Deterministic pseudo-random eviction using an xorshift64 PRNG.
    ///
    /// The seed is advanced on every eviction decision.
    Random {
        /// Initial xorshift64 seed (must be non-zero; zero is replaced with 1).
        seed: u64,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// ShardConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for `RoutingTableSharding`.
#[derive(Clone, Debug)]
pub struct ShardConfig {
    /// Number of shards.  Must be in `[1, 255]`.
    pub num_shards: u8,
    /// Maximum entries stored in each shard before eviction is triggered.
    pub max_entries_per_shard: usize,
    /// Policy used to select the eviction victim.
    pub eviction_policy: EvictionPolicy,
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            num_shards: 16,
            max_entries_per_shard: 256,
            eviction_policy: EvictionPolicy::LeastRecentlySeen,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ShardStats
// ─────────────────────────────────────────────────────────────────────────────

/// Per-shard statistics snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct ShardStats {
    /// Shard identifier.
    pub shard_id: u8,
    /// Number of entries currently in the shard.
    pub entry_count: usize,
    /// Mean RTT across all entries in the shard; `0.0` when the shard is empty.
    pub avg_rtt_ms: f64,
    /// `last_seen` value of the oldest (least-recently-seen) entry; `0` when empty.
    pub oldest_entry_ms: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// RoutingTableSharding
// ─────────────────────────────────────────────────────────────────────────────

/// Sharded DHT routing table.
///
/// Peers are placed into shards by `node_id.0[0] % num_shards`, giving an
/// approximately uniform distribution for random node IDs while preserving the
/// XOR-metric locality needed for closest-peer queries.
pub struct RoutingTableSharding {
    /// Shard configuration (immutable after construction).
    pub config: ShardConfig,
    /// One `HashMap` per shard.
    pub shards: Vec<HashMap<NodeId, RoutingEntry>>,
    /// Monotonically increasing count of successful `insert` calls.
    pub total_insertions: u64,
    /// Monotonically increasing count of evictions caused by capacity enforcement.
    pub total_evictions: u64,
    /// Monotonically increasing count of `get` / `closest_nodes` calls.
    ///
    /// Uses `Cell` for interior-mutability so that read-only methods
    /// (`get`, `closest_nodes`) can increment the counter without requiring
    /// `&mut self`.
    pub total_lookups: Cell<u64>,
    /// Current xorshift64 state for `EvictionPolicy::Random`.
    rng_state: u64,
}

impl RoutingTableSharding {
    /// Construct a new routing table with `config.num_shards` empty shards.
    pub fn new(config: ShardConfig) -> Self {
        let n = config.num_shards as usize;
        // Initialise the RNG state from the seed embedded in the eviction policy
        // (if present); fall back to a fixed non-zero value.
        let rng_state = match &config.eviction_policy {
            EvictionPolicy::Random { seed } => {
                if *seed == 0 {
                    1
                } else {
                    *seed
                }
            }
            _ => 6_364_136_223_846_793_005_u64, // arbitrary non-zero constant
        };
        Self {
            shards: (0..n).map(|_| HashMap::new()).collect(),
            config,
            total_insertions: 0,
            total_evictions: 0,
            total_lookups: Cell::new(0),
            rng_state,
        }
    }

    // ── Shard routing ────────────────────────────────────────────────────────

    /// Compute which shard a `NodeId` belongs to.
    #[inline]
    pub fn shard_for(&self, node_id: &NodeId) -> ShardId {
        ShardId(node_id.0[0] % self.config.num_shards)
    }

    // ── Mutation ─────────────────────────────────────────────────────────────

    /// Insert `entry` into the appropriate shard.
    ///
    /// If the shard is already at capacity after the insertion, one entry is
    /// evicted according to `config.eviction_policy`.
    ///
    /// Returns `true` if an eviction occurred, `false` otherwise.
    pub fn insert(&mut self, entry: RoutingEntry, _now: u64) -> bool {
        let shard_idx = self.shard_for(&entry.node_id).0 as usize;
        let shard = match self.shards.get_mut(shard_idx) {
            Some(s) => s,
            None => return false,
        };
        shard.insert(entry.node_id, entry);
        self.total_insertions += 1;

        if shard.len() > self.config.max_entries_per_shard {
            self.evict_one(shard_idx);
            true
        } else {
            false
        }
    }

    /// Remove the entry for `node_id`.  Returns `true` if an entry existed.
    pub fn remove(&mut self, node_id: &NodeId) -> bool {
        let shard_idx = self.shard_for(node_id).0 as usize;
        match self.shards.get_mut(shard_idx) {
            Some(shard) => shard.remove(node_id).is_some(),
            None => false,
        }
    }

    /// Look up a single entry by `NodeId`.
    pub fn get(&self, node_id: &NodeId) -> Option<&RoutingEntry> {
        self.total_lookups_inc();
        let shard_idx = self.shard_for(node_id).0 as usize;
        self.shards.get(shard_idx)?.get(node_id)
    }

    /// Update the RTT and `last_seen` timestamp for an existing entry.
    ///
    /// Returns `true` if the entry was found and updated.
    pub fn update_rtt(&mut self, node_id: &NodeId, rtt_ms: u32, now: u64) -> bool {
        let shard_idx = self.shard_for(node_id).0 as usize;
        match self.shards.get_mut(shard_idx) {
            Some(shard) => match shard.get_mut(node_id) {
                Some(e) => {
                    e.rtt_ms = rtt_ms;
                    e.last_seen = now;
                    true
                }
                None => false,
            },
            None => false,
        }
    }

    // ── Queries ──────────────────────────────────────────────────────────────

    /// Return the `k` entries whose XOR distance to `target` is smallest,
    /// sorted ascending by distance (closest first).
    ///
    /// Scans all shards; O(N log k) where N is the total number of entries.
    pub fn closest_nodes(&self, target: &NodeId, k: usize) -> Vec<&RoutingEntry> {
        self.total_lookups_inc();
        let mut all: Vec<(&NodeId, &RoutingEntry)> =
            self.shards.iter().flat_map(|shard| shard.iter()).collect();

        all.sort_by(|(a_id, _), (b_id, _)| {
            let da = a_id.xor_distance(target);
            let db = b_id.xor_distance(target);
            da.cmp(&db)
        });

        all.into_iter().take(k).map(|(_, e)| e).collect()
    }

    /// Return all entries in the specified shard.
    pub fn nodes_in_shard(&self, shard: ShardId) -> Vec<&RoutingEntry> {
        match self.shards.get(shard.0 as usize) {
            Some(s) => s.values().collect(),
            None => vec![],
        }
    }

    /// Remove all entries where `now.saturating_sub(last_seen) > max_age_ms`.
    ///
    /// Returns the number of entries removed.
    pub fn evict_stale(&mut self, now: u64, max_age_ms: u64) -> usize {
        let mut removed = 0usize;
        for shard in &mut self.shards {
            let before = shard.len();
            shard.retain(|_, e| now.saturating_sub(e.last_seen) <= max_age_ms);
            removed += before - shard.len();
        }
        removed
    }

    // ── Statistics ───────────────────────────────────────────────────────────

    /// Statistics for a single shard at time `now`.
    pub fn shard_stats(&self, shard: ShardId, now: u64) -> ShardStats {
        let shard_idx = shard.0 as usize;
        let shard_map = match self.shards.get(shard_idx) {
            Some(s) => s,
            None => {
                return ShardStats {
                    shard_id: shard.0,
                    entry_count: 0,
                    avg_rtt_ms: 0.0,
                    oldest_entry_ms: 0,
                };
            }
        };

        let _ = now; // available for future use (e.g. age-based metrics)
        let entry_count = shard_map.len();
        if entry_count == 0 {
            return ShardStats {
                shard_id: shard.0,
                entry_count: 0,
                avg_rtt_ms: 0.0,
                oldest_entry_ms: 0,
            };
        }

        let sum_rtt: u64 = shard_map.values().map(|e| e.rtt_ms as u64).sum();
        let avg_rtt_ms = sum_rtt as f64 / entry_count as f64;
        let oldest_entry_ms = shard_map.values().map(|e| e.last_seen).min().unwrap_or(0);

        ShardStats {
            shard_id: shard.0,
            entry_count,
            avg_rtt_ms,
            oldest_entry_ms,
        }
    }

    /// Statistics for all shards, sorted ascending by `shard_id`.
    pub fn all_stats(&self, now: u64) -> Vec<ShardStats> {
        (0..self.config.num_shards)
            .map(|i| self.shard_stats(ShardId(i), now))
            .collect()
    }

    /// Total number of entries across all shards.
    pub fn total_entries(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    /// Global counters: `(total_insertions, total_evictions, total_lookups)`.
    pub fn global_stats(&self) -> (u64, u64, u64) {
        (
            self.total_insertions,
            self.total_evictions,
            self.total_lookups.get(),
        )
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Increment the `total_lookups` counter via `Cell<u64>` interior mutability.
    #[inline]
    fn total_lookups_inc(&self) {
        self.total_lookups
            .set(self.total_lookups.get().wrapping_add(1));
    }

    /// Pick and remove one victim from shard `shard_idx` according to the
    /// configured eviction policy.
    fn evict_one(&mut self, shard_idx: usize) {
        use EvictionPolicy::{HighestRtt, LeastRecentlySeen, Random};

        // Snapshot the node IDs so we can find the victim without holding a
        // mutable reference to the shard at the same time.
        let victim_id: Option<NodeId> = {
            let shard = match self.shards.get(shard_idx) {
                Some(s) => s,
                None => return,
            };

            match &self.config.eviction_policy {
                LeastRecentlySeen => shard
                    .values()
                    .min_by_key(|e| e.last_seen)
                    .map(|e| e.node_id),

                HighestRtt => shard.values().max_by_key(|e| e.rtt_ms).map(|e| e.node_id),

                Random { .. } => {
                    // We need a stable index-based traversal; collect keys.
                    let keys: Vec<NodeId> = shard.keys().copied().collect();
                    if keys.is_empty() {
                        None
                    } else {
                        // xorshift64 step
                        let mut state = self.rng_state;
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        // Store updated state before immutable borrow ends.
                        Some(keys[state as usize % keys.len()])
                        // state is updated below after victim_id is determined
                    }
                }
            }
        };

        // Advance PRNG state after the immutable borrow of `self.config` ends.
        if matches!(&self.config.eviction_policy, EvictionPolicy::Random { .. }) {
            let mut state = self.rng_state;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            self.rng_state = state;
        }

        if let Some(id) = victim_id {
            if let Some(shard) = self.shards.get_mut(shard_idx) {
                if shard.remove(&id).is_some() {
                    self.total_evictions += 1;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{EvictionPolicy, NodeId, RoutingEntry, RoutingTableSharding, ShardConfig, ShardId};

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_id(first_byte: u8) -> NodeId {
        let mut b = [0u8; 32];
        b[0] = first_byte;
        NodeId(b)
    }

    fn make_id_full(bytes: [u8; 32]) -> NodeId {
        NodeId(bytes)
    }

    fn make_entry(
        id: NodeId,
        addr: &str,
        last_seen: u64,
        rtt_ms: u32,
        shard: ShardId,
    ) -> RoutingEntry {
        RoutingEntry {
            node_id: id,
            addr: addr.to_string(),
            last_seen,
            rtt_ms,
            shard,
        }
    }

    fn default_table() -> RoutingTableSharding {
        RoutingTableSharding::new(ShardConfig::default())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // NodeId tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn node_id_xor_distance_identity() {
        let id = make_id(0xAB);
        let dist = id.xor_distance(&id);
        assert_eq!(dist, [0u8; 32]);
    }

    #[test]
    fn node_id_xor_distance_known_value() {
        let a = make_id(0b0000_0001);
        let b = make_id(0b0000_0011);
        let dist = a.xor_distance(&b);
        assert_eq!(dist[0], 0b0000_0010);
        assert_eq!(&dist[1..], &[0u8; 31]);
    }

    #[test]
    fn node_id_leading_zeros_all_zero() {
        let id = NodeId([0u8; 32]);
        assert_eq!(id.leading_zeros(), 256);
    }

    #[test]
    fn node_id_leading_zeros_first_bit_set() {
        let id = make_id(0x80); // 1000_0000
        assert_eq!(id.leading_zeros(), 0);
    }

    #[test]
    fn node_id_leading_zeros_second_bit_set() {
        let id = make_id(0x40); // 0100_0000
        assert_eq!(id.leading_zeros(), 1);
    }

    #[test]
    fn node_id_leading_zeros_cross_byte_boundary() {
        let mut b = [0u8; 32];
        b[1] = 0x80; // second byte first bit set → 8 leading zeros
        let id = NodeId(b);
        assert_eq!(id.leading_zeros(), 8);
    }

    #[test]
    fn node_id_from_str_hex_valid() {
        let hex = "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20";
        let id = NodeId::from_str_hex(hex).expect("valid hex");
        assert_eq!(id.0[0], 0x01);
        assert_eq!(id.0[31], 0x20);
    }

    #[test]
    fn node_id_from_str_hex_invalid_length() {
        assert!(NodeId::from_str_hex("0102").is_none());
    }

    #[test]
    fn node_id_from_str_hex_invalid_char() {
        let bad = "zz02030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20";
        assert!(NodeId::from_str_hex(bad).is_none());
    }

    #[test]
    fn node_id_from_str_hex_roundtrip() {
        let id = make_id(0xDE);
        let hex = format!("{id}");
        let id2 = NodeId::from_str_hex(&hex).expect("roundtrip");
        assert_eq!(id, id2);
    }

    #[test]
    fn node_id_ordering() {
        let a = make_id(1);
        let b = make_id(2);
        assert!(a < b);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ShardConfig / default
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn shard_config_default_values() {
        let cfg = ShardConfig::default();
        assert_eq!(cfg.num_shards, 16);
        assert_eq!(cfg.max_entries_per_shard, 256);
        assert_eq!(cfg.eviction_policy, EvictionPolicy::LeastRecentlySeen);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // shard_for
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn shard_for_is_first_byte_mod_num_shards() {
        let table = default_table();
        for first in 0u8..=255 {
            let id = make_id(first);
            assert_eq!(table.shard_for(&id).0, first % 16);
        }
    }

    #[test]
    fn shard_for_single_shard_always_zero() {
        let cfg = ShardConfig {
            num_shards: 1,
            ..Default::default()
        };
        let table = RoutingTableSharding::new(cfg);
        for first in 0u8..=255 {
            assert_eq!(table.shard_for(&make_id(first)).0, 0);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // insert / get / remove
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn insert_and_get_basic() {
        let mut table = default_table();
        let id = make_id(0x00);
        let shard = table.shard_for(&id);
        let entry = make_entry(id, "127.0.0.1:4001", 1000, 5, shard);
        let evicted = table.insert(entry, 1000);
        assert!(!evicted);
        let got = table.get(&id).expect("entry present");
        assert_eq!(got.addr, "127.0.0.1:4001");
        assert_eq!(got.rtt_ms, 5);
    }

    #[test]
    fn insert_increments_total_insertions() {
        let mut table = default_table();
        let id = make_id(0x10);
        let shard = table.shard_for(&id);
        table.insert(make_entry(id, "a", 0, 0, shard), 0);
        assert_eq!(table.total_insertions, 1);
    }

    #[test]
    fn remove_existing_entry() {
        let mut table = default_table();
        let id = make_id(0x20);
        let shard = table.shard_for(&id);
        table.insert(make_entry(id, "a", 0, 0, shard), 0);
        assert!(table.remove(&id));
        assert!(table.get(&id).is_none());
    }

    #[test]
    fn remove_absent_entry_returns_false() {
        let mut table = default_table();
        assert!(!table.remove(&make_id(0x30)));
    }

    #[test]
    fn total_entries_reflects_inserts_and_removes() {
        let mut table = default_table();
        let ids: Vec<NodeId> = (0..5).map(|i| make_id(i * 16)).collect();
        for &id in &ids {
            let shard = table.shard_for(&id);
            table.insert(make_entry(id, "x", 0, 0, shard), 0);
        }
        assert_eq!(table.total_entries(), 5);
        table.remove(&ids[0]);
        assert_eq!(table.total_entries(), 4);
    }

    #[test]
    fn duplicate_insert_overwrites_existing_entry() {
        let mut table = default_table();
        let id = make_id(0x01);
        let shard = table.shard_for(&id);
        table.insert(make_entry(id, "old", 100, 10, shard), 100);
        table.insert(make_entry(id, "new", 200, 20, shard), 200);
        let got = table.get(&id).expect("entry present");
        assert_eq!(got.addr, "new");
        assert_eq!(got.rtt_ms, 20);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // update_rtt
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn update_rtt_existing_entry() {
        let mut table = default_table();
        let id = make_id(0x02);
        let shard = table.shard_for(&id);
        table.insert(make_entry(id, "a", 1000, 50, shard), 1000);
        let ok = table.update_rtt(&id, 25, 2000);
        assert!(ok);
        let e = table.get(&id).expect("still present");
        assert_eq!(e.rtt_ms, 25);
        assert_eq!(e.last_seen, 2000);
    }

    #[test]
    fn update_rtt_missing_entry_returns_false() {
        let mut table = default_table();
        assert!(!table.update_rtt(&make_id(0xFF), 10, 0));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // closest_nodes
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn closest_nodes_returns_sorted_by_xor() {
        let mut table = RoutingTableSharding::new(ShardConfig {
            num_shards: 4,
            max_entries_per_shard: 256,
            eviction_policy: EvictionPolicy::LeastRecentlySeen,
        });
        // Create IDs with known distances from target 0x00...
        let target = NodeId([0u8; 32]);
        let ids: Vec<NodeId> = (1u8..=4).map(make_id).collect();
        for &id in &ids {
            let shard = table.shard_for(&id);
            table.insert(make_entry(id, "a", 0, 0, shard), 0);
        }
        let closest = table.closest_nodes(&target, 2);
        assert_eq!(closest.len(), 2);
        // node with first byte=1 is closer than first byte=2
        assert_eq!(closest[0].node_id, make_id(1));
        assert_eq!(closest[1].node_id, make_id(2));
    }

    #[test]
    fn closest_nodes_k_larger_than_total_returns_all() {
        let mut table = default_table();
        for i in 0u8..5 {
            let id = make_id(i);
            let shard = table.shard_for(&id);
            table.insert(make_entry(id, "a", 0, 0, shard), 0);
        }
        let result = table.closest_nodes(&NodeId([0u8; 32]), 100);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn closest_nodes_increments_lookup_counter() {
        let table = default_table();
        table.closest_nodes(&NodeId([0u8; 32]), 5);
        assert_eq!(table.total_lookups.get(), 1);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // nodes_in_shard
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn nodes_in_shard_returns_correct_entries() {
        let mut table = RoutingTableSharding::new(ShardConfig {
            num_shards: 4,
            max_entries_per_shard: 256,
            eviction_policy: EvictionPolicy::LeastRecentlySeen,
        });
        // first_byte=0 → shard 0; first_byte=1 → shard 1
        let id0 = make_id(0);
        let id1 = make_id(1);
        table.insert(make_entry(id0, "a", 0, 0, ShardId(0)), 0);
        table.insert(make_entry(id1, "b", 0, 0, ShardId(1)), 0);
        assert_eq!(table.nodes_in_shard(ShardId(0)).len(), 1);
        assert_eq!(table.nodes_in_shard(ShardId(1)).len(), 1);
        assert_eq!(table.nodes_in_shard(ShardId(2)).len(), 0);
    }

    #[test]
    fn nodes_in_shard_invalid_shard_returns_empty() {
        let table = default_table(); // 16 shards
        assert_eq!(table.nodes_in_shard(ShardId(200)).len(), 0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // evict_stale
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn evict_stale_removes_old_entries() {
        let mut table = default_table();
        let id_old = make_id(0x00);
        let id_new = make_id(0x10);
        let s_old = table.shard_for(&id_old);
        let s_new = table.shard_for(&id_new);
        table.insert(make_entry(id_old, "old", 100, 0, s_old), 100);
        table.insert(make_entry(id_new, "new", 2000, 0, s_new), 2000);

        // now=3000, max_age=500 → 3000-100=2900 > 500 → id_old stale
        //                        → 3000-2000=1000 > 500 → id_new also stale!
        // Use max_age=1500 to keep id_new
        let removed = table.evict_stale(3000, 1500);
        assert_eq!(removed, 1);
        assert!(table.get(&id_old).is_none());
        assert!(table.get(&id_new).is_some());
    }

    #[test]
    fn evict_stale_no_stale_entries_returns_zero() {
        let mut table = default_table();
        let id = make_id(0x05);
        let shard = table.shard_for(&id);
        table.insert(make_entry(id, "a", 1000, 0, shard), 1000);
        let removed = table.evict_stale(1500, 1000); // age=500 ≤ 1000
        assert_eq!(removed, 0);
    }

    #[test]
    fn evict_stale_removes_all_stale() {
        let mut table = default_table();
        for i in 0u8..8 {
            let id = make_id(i);
            let shard = table.shard_for(&id);
            table.insert(make_entry(id, "a", 0, 0, shard), 0);
        }
        let removed = table.evict_stale(10_000, 100); // all entries are older than 100 ms
        assert_eq!(removed, 8);
        assert_eq!(table.total_entries(), 0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // shard_stats / all_stats
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn shard_stats_empty_shard() {
        let table = default_table();
        let stats = table.shard_stats(ShardId(0), 1000);
        assert_eq!(stats.entry_count, 0);
        assert_eq!(stats.avg_rtt_ms, 0.0);
        assert_eq!(stats.oldest_entry_ms, 0);
    }

    #[test]
    fn shard_stats_single_entry() {
        let mut table = RoutingTableSharding::new(ShardConfig {
            num_shards: 4,
            max_entries_per_shard: 256,
            eviction_policy: EvictionPolicy::LeastRecentlySeen,
        });
        let id = make_id(0); // shard 0
        table.insert(make_entry(id, "a", 500, 20, ShardId(0)), 500);
        let stats = table.shard_stats(ShardId(0), 1000);
        assert_eq!(stats.entry_count, 1);
        assert!((stats.avg_rtt_ms - 20.0).abs() < f64::EPSILON);
        assert_eq!(stats.oldest_entry_ms, 500);
    }

    #[test]
    fn shard_stats_avg_rtt_multiple_entries() {
        let mut table = RoutingTableSharding::new(ShardConfig {
            num_shards: 1,
            max_entries_per_shard: 256,
            eviction_policy: EvictionPolicy::LeastRecentlySeen,
        });
        // All in shard 0 (num_shards=1)
        let ids: Vec<NodeId> = (0u8..4).map(make_id).collect();
        let rtts = [10u32, 20, 30, 40];
        for (id, &rtt) in ids.iter().zip(rtts.iter()) {
            table.insert(make_entry(*id, "a", 1000, rtt, ShardId(0)), 1000);
        }
        let stats = table.shard_stats(ShardId(0), 1000);
        assert_eq!(stats.entry_count, 4);
        assert!((stats.avg_rtt_ms - 25.0).abs() < 1e-9);
    }

    #[test]
    fn all_stats_returns_one_per_shard_sorted() {
        let table = default_table(); // 16 shards
        let stats = table.all_stats(1000);
        assert_eq!(stats.len(), 16);
        for (i, s) in stats.iter().enumerate() {
            assert_eq!(s.shard_id, i as u8);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Eviction policies
    // ─────────────────────────────────────────────────────────────────────────

    fn fill_shard_to_capacity(
        table: &mut RoutingTableSharding,
        shard_first_byte: u8,
        count: usize,
    ) {
        // Insert `count` entries all mapping to the same shard.
        // num_shards is assumed to divide evenly.  We vary bytes 1..31 to get
        // distinct NodeIds that still land in the same shard.
        let num_shards = table.config.num_shards;
        for i in 0u8..(count as u8) {
            let mut b = [0u8; 32];
            b[0] = shard_first_byte;
            b[1] = i;
            let id = NodeId(b);
            // Ensure the entry really maps to the intended shard
            debug_assert_eq!(id.0[0] % num_shards, shard_first_byte % num_shards);
            let shard = table.shard_for(&id);
            let entry = RoutingEntry {
                node_id: id,
                addr: format!("10.0.0.{i}:4001"),
                last_seen: i as u64 * 100, // ascending last_seen
                rtt_ms: 200 - i as u32,    // descending RTT (i=0 → rtt=200 highest)
                shard,
            };
            table.insert(entry, i as u64 * 100);
        }
    }

    #[test]
    fn eviction_lrs_removes_least_recently_seen() {
        let max = 4usize;
        let cfg = ShardConfig {
            num_shards: 16,
            max_entries_per_shard: max,
            eviction_policy: EvictionPolicy::LeastRecentlySeen,
        };
        let mut table = RoutingTableSharding::new(cfg);
        // Fill shard 0 to exactly `max` entries.
        // Entry with b[1]=0 has last_seen=0 (oldest).
        fill_shard_to_capacity(&mut table, 0, max);
        assert_eq!(table.total_entries(), max);

        // Insert one more → triggers eviction.
        let mut b = [0u8; 32];
        b[0] = 0;
        b[1] = 100; // unique
        let trigger_id = NodeId(b);
        let shard = table.shard_for(&trigger_id);
        table.insert(
            RoutingEntry {
                node_id: trigger_id,
                addr: "trigger".to_string(),
                last_seen: 9999,
                rtt_ms: 1,
                shard,
            },
            9999,
        );
        // Shard is back to `max` entries and total_evictions incremented.
        assert_eq!(table.nodes_in_shard(shard).len(), max);
        assert_eq!(table.total_evictions, 1);

        // Entry with last_seen=0 (b[1]=0) should be gone.
        let mut victim_b = [0u8; 32];
        victim_b[0] = 0;
        victim_b[1] = 0;
        assert!(
            table.get(&NodeId(victim_b)).is_none(),
            "LRS victim should be evicted"
        );
    }

    #[test]
    fn eviction_highest_rtt_removes_slowest_peer() {
        let max = 4usize;
        let cfg = ShardConfig {
            num_shards: 16,
            max_entries_per_shard: max,
            eviction_policy: EvictionPolicy::HighestRtt,
        };
        let mut table = RoutingTableSharding::new(cfg);
        fill_shard_to_capacity(&mut table, 0, max);

        let mut b = [0u8; 32];
        b[0] = 0;
        b[1] = 100;
        let trigger_id = NodeId(b);
        let shard = table.shard_for(&trigger_id);
        table.insert(
            RoutingEntry {
                node_id: trigger_id,
                addr: "trigger".to_string(),
                last_seen: 9999,
                rtt_ms: 1,
                shard,
            },
            9999,
        );
        assert_eq!(table.total_evictions, 1);

        // Entry with rtt_ms=200 (b[1]=0) should be gone.
        let mut victim_b = [0u8; 32];
        victim_b[0] = 0;
        victim_b[1] = 0;
        assert!(
            table.get(&NodeId(victim_b)).is_none(),
            "HighestRtt victim should be evicted"
        );
    }

    #[test]
    fn eviction_random_removes_one_entry() {
        let max = 4usize;
        let cfg = ShardConfig {
            num_shards: 16,
            max_entries_per_shard: max,
            eviction_policy: EvictionPolicy::Random {
                seed: 0xDEAD_BEEF_1234_5678,
            },
        };
        let mut table = RoutingTableSharding::new(cfg);
        fill_shard_to_capacity(&mut table, 0, max);

        let mut b = [0u8; 32];
        b[0] = 0;
        b[1] = 100;
        let trigger_id = NodeId(b);
        let shard = table.shard_for(&trigger_id);
        let evicted = table.insert(
            RoutingEntry {
                node_id: trigger_id,
                addr: "trigger".to_string(),
                last_seen: 9999,
                rtt_ms: 1,
                shard,
            },
            9999,
        );
        assert!(evicted, "Should have evicted one entry");
        assert_eq!(table.nodes_in_shard(ShardId(0)).len(), max);
        assert_eq!(table.total_evictions, 1);
    }

    #[test]
    fn eviction_random_seed_zero_replaced_with_one() {
        // seed=0 is replaced with 1; construction must not panic/infinite-loop.
        let cfg = ShardConfig {
            num_shards: 1,
            max_entries_per_shard: 2,
            eviction_policy: EvictionPolicy::Random { seed: 0 },
        };
        let mut table = RoutingTableSharding::new(cfg);
        for i in 0u8..3 {
            let id = make_id(i);
            let shard = table.shard_for(&id);
            table.insert(make_entry(id, "a", i as u64 * 10, 10, shard), i as u64 * 10);
        }
        assert_eq!(table.total_entries(), 2);
        assert_eq!(table.total_evictions, 1);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // global_stats
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn global_stats_tracks_all_counters() {
        let mut table = default_table();
        let id = make_id(0x04);
        let shard = table.shard_for(&id);
        table.insert(make_entry(id, "a", 0, 0, shard), 0);
        let _ = table.get(&id);
        table.closest_nodes(&NodeId([0u8; 32]), 10);
        let (ins, evic, look) = table.global_stats();
        assert_eq!(ins, 1);
        assert_eq!(evic, 0);
        assert_eq!(look, 2); // one get + one closest_nodes
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Misc / edge cases
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn empty_table_closest_nodes_returns_empty() {
        let table = default_table();
        assert!(table.closest_nodes(&NodeId([0u8; 32]), 10).is_empty());
    }

    #[test]
    fn empty_table_evict_stale_returns_zero() {
        let mut table = default_table();
        assert_eq!(table.evict_stale(9999, 0), 0);
    }

    #[test]
    fn large_insertion_stress() {
        let cfg = ShardConfig {
            num_shards: 8,
            max_entries_per_shard: 50,
            eviction_policy: EvictionPolicy::LeastRecentlySeen,
        };
        let mut table = RoutingTableSharding::new(cfg);
        for i in 0u32..1000 {
            let mut b = [0u8; 32];
            b[0] = (i % 256) as u8;
            b[1] = (i / 256) as u8;
            let id = NodeId(b);
            let shard = table.shard_for(&id);
            table.insert(
                RoutingEntry {
                    node_id: id,
                    addr: "x".to_string(),
                    last_seen: i as u64,
                    rtt_ms: 1,
                    shard,
                },
                i as u64,
            );
        }
        // Each shard holds at most max_entries_per_shard
        for shard_map in &table.shards {
            assert!(
                shard_map.len() <= 50,
                "Shard overflow: {} > 50",
                shard_map.len()
            );
        }
        assert!(
            table.total_evictions > 0,
            "Expected some evictions under stress"
        );
    }

    #[test]
    fn node_id_xor_distance_commutativity() {
        let a = make_id_full({
            let mut b = [0u8; 32];
            b[0] = 0xAB;
            b[15] = 0xCD;
            b
        });
        let b = make_id_full({
            let mut b = [0u8; 32];
            b[0] = 0x12;
            b[15] = 0x34;
            b
        });
        assert_eq!(a.xor_distance(&b), b.xor_distance(&a));
    }

    #[test]
    fn multiple_evictions_counted_correctly() {
        let cfg = ShardConfig {
            num_shards: 1,
            max_entries_per_shard: 3,
            eviction_policy: EvictionPolicy::HighestRtt,
        };
        let mut table = RoutingTableSharding::new(cfg);
        for i in 0u8..10 {
            let id = make_id(i);
            let shard = table.shard_for(&id);
            table.insert(make_entry(id, "a", i as u64, i as u32, shard), i as u64);
        }
        assert_eq!(table.total_evictions, 7); // 10 inserts - 3 remaining = 7 evictions
        assert_eq!(table.total_entries(), 3);
    }

    #[test]
    fn update_rtt_does_not_change_shard_assignment() {
        let mut table = default_table();
        let id = make_id(0x07);
        let shard_before = table.shard_for(&id);
        table.insert(make_entry(id, "a", 0, 10, shard_before), 0);
        table.update_rtt(&id, 99, 500);
        let entry = table.get(&id).expect("still present");
        assert_eq!(entry.shard, shard_before);
    }

    #[test]
    fn closest_nodes_across_multiple_shards() {
        let mut table = RoutingTableSharding::new(ShardConfig {
            num_shards: 16,
            max_entries_per_shard: 256,
            eviction_policy: EvictionPolicy::LeastRecentlySeen,
        });
        // Spread entries across many shards.
        for i in 0u8..32 {
            let id = make_id(i);
            let shard = table.shard_for(&id);
            table.insert(make_entry(id, "a", 0, 0, shard), 0);
        }
        let target = NodeId([0u8; 32]);
        let result = table.closest_nodes(&target, 5);
        assert_eq!(result.len(), 5);
        // Verify sorted order: XOR distance should be non-decreasing
        for w in result.windows(2) {
            let d0 = w[0].node_id.xor_distance(&target);
            let d1 = w[1].node_id.xor_distance(&target);
            assert!(d0 <= d1);
        }
    }
}
