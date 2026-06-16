//! Gossip Anti-Entropy — Merkle-digest-based state reconciliation between peers.
//!
//! Implements efficient detection and repair of state divergence using FNV-1a hashes,
//! sorted digest entries, and a lightweight reconciliation protocol that classifies
//! differences into sent, requested, and conflict categories.
//!
//! ## Overview
//!
//! Anti-entropy works in three phases:
//! 1. **Build** — each node builds a [`MerkleDigest`] from its local [`DigestEntry`] table.
//! 2. **Compare** — `diff_keys` finds keys that differ between two digests.
//! 3. **Reconcile** — [`GossipAntiEntropy::reconcile`] classifies each diff key and returns
//!    a [`ReconcileResult`] describing what to send, what to request, and what conflicts exist.
//!
//! ## Example
//!
//! ```rust
//! use ipfrs_network::anti_entropy::{AntiEntropyConfig, GossipAntiEntropy};
//!
//! let config = AntiEntropyConfig::default();
//! let mut ae = GossipAntiEntropy::new(config);
//!
//! ae.upsert("block/abc".to_string(), 0xdeadbeef, 1, 1_700_000_000);
//! ae.upsert("block/xyz".to_string(), 0xc0ffee00, 2, 1_700_000_001);
//!
//! let digest = ae.build_digest();
//! assert_eq!(digest.entries.len(), 2);
//! let (count, _hash) = ae.stats();
//! assert_eq!(count, 2);
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// FNV-1a constants and helpers
// ---------------------------------------------------------------------------

const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;

/// Compute a FNV-1a 64-bit hash of the given byte slice.
fn fnv1a(data: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Hash a UTF-8 string with FNV-1a.
fn fnv1a_str(s: &str) -> u64 {
    fnv1a(s.as_bytes())
}

// ---------------------------------------------------------------------------
// DigestEntry
// ---------------------------------------------------------------------------

/// A single entry in a Merkle digest representing a keyed piece of state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DigestEntry {
    /// Logical key identifying this piece of state.
    pub key: String,
    /// FNV-1a hash of the serialized value at this key.
    pub value_hash: u64,
    /// Monotonic version counter; higher means newer.
    pub version: u64,
    /// Wall-clock timestamp (seconds since Unix epoch) when last updated.
    pub updated_at_secs: u64,
}

impl DigestEntry {
    /// Create a new digest entry.
    pub fn new(key: String, value_hash: u64, version: u64, updated_at_secs: u64) -> Self {
        Self {
            key,
            value_hash,
            version,
            updated_at_secs,
        }
    }

    /// Compute the per-entry contribution to the root hash.
    ///
    /// Combines `key_hash XOR value_hash XOR version` so that any change to
    /// any field shifts the root.
    fn root_contribution(&self) -> u64 {
        fnv1a_str(&self.key) ^ self.value_hash ^ self.version
    }
}

// ---------------------------------------------------------------------------
// MerkleDigest
// ---------------------------------------------------------------------------

/// A compact, sortable digest of local state suitable for peer comparison.
///
/// Entries are always kept sorted by key so that `diff_keys` is deterministic
/// and independent of insertion order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MerkleDigest {
    /// Sorted (by key) snapshot of local state entries.
    pub entries: Vec<DigestEntry>,
    /// XOR-fold of all per-entry `(key_hash XOR value_hash XOR version)` tuples.
    pub root_hash: u64,
}

impl MerkleDigest {
    /// Compute a [`MerkleDigest`] from an already-sorted list of entries.
    ///
    /// Assumes `entries` is sorted by key; callers must ensure this invariant.
    pub fn from_sorted(entries: Vec<DigestEntry>) -> Self {
        let root_hash = entries
            .iter()
            .fold(0u64, |acc, e| acc ^ e.root_contribution());
        Self { entries, root_hash }
    }

    /// Returns `true` when both digests share the same root hash, meaning all
    /// keys, values, and versions are identical.
    pub fn matches(&self, other: &MerkleDigest) -> bool {
        self.root_hash == other.root_hash
    }

    /// Return the set of keys that differ between `self` and `other`.
    ///
    /// A key is included when:
    /// - It exists in one digest but not the other.
    /// - It exists in both but `value_hash` or `version` differ.
    pub fn diff_keys(&self, other: &MerkleDigest) -> Vec<String> {
        // Build a lookup map from the `other` digest.
        let other_map: HashMap<&str, &DigestEntry> =
            other.entries.iter().map(|e| (e.key.as_str(), e)).collect();
        let self_map: HashMap<&str, &DigestEntry> =
            self.entries.iter().map(|e| (e.key.as_str(), e)).collect();

        let mut diff: Vec<String> = Vec::new();

        // Keys in self: check for missing or changed entries in other.
        for e in &self.entries {
            match other_map.get(e.key.as_str()) {
                None => diff.push(e.key.clone()),
                Some(o) => {
                    if o.value_hash != e.value_hash || o.version != e.version {
                        diff.push(e.key.clone());
                    }
                }
            }
        }

        // Keys in other but missing from self.
        for e in &other.entries {
            if !self_map.contains_key(e.key.as_str()) {
                diff.push(e.key.clone());
            }
        }

        diff.sort();
        diff.dedup();
        diff
    }
}

// ---------------------------------------------------------------------------
// ReconcileResult
// ---------------------------------------------------------------------------

/// Outcome of a reconciliation round between two peers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReconcileResult {
    /// Keys that this peer should *send* to the remote (we are ahead or remote is missing them).
    pub sent_keys: Vec<String>,
    /// Keys that this peer needs to *request* from the remote (remote is ahead or we are missing them).
    pub requested_keys: Vec<String>,
    /// Keys where both sides disagree on `value_hash` at the *same* version (true conflicts).
    pub conflict_keys: Vec<String>,
}

// ---------------------------------------------------------------------------
// AntiEntropyConfig
// ---------------------------------------------------------------------------

/// Configuration knobs for [`GossipAntiEntropy`].
#[derive(Debug, Clone)]
pub struct AntiEntropyConfig {
    /// How often (in seconds) to run anti-entropy synchronisation with each peer.
    pub sync_interval_secs: u64,
    /// Maximum number of diff keys to process in a single reconciliation round.
    /// Prevents unbounded message sizes during initial bootstrap.
    pub max_diff_keys: usize,
}

impl Default for AntiEntropyConfig {
    fn default() -> Self {
        Self {
            sync_interval_secs: 30,
            max_diff_keys: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// GossipAntiEntropy
// ---------------------------------------------------------------------------

/// Anti-entropy engine that maintains local state and reconciles with remote peers.
///
/// Use [`GossipAntiEntropy::upsert`] to register or update state entries,
/// [`GossipAntiEntropy::build_digest`] to create a digest for exchange, and
/// [`GossipAntiEntropy::reconcile`] to classify differences once you have
/// the remote peer's digest.
pub struct GossipAntiEntropy {
    /// Local keyed state.  Key is the entry's logical key string.
    local_state: HashMap<String, DigestEntry>,
    /// Configuration controlling sync frequency and batch limits.
    config: AntiEntropyConfig,
}

impl GossipAntiEntropy {
    /// Create a new anti-entropy engine with the given configuration.
    pub fn new(config: AntiEntropyConfig) -> Self {
        Self {
            local_state: HashMap::new(),
            config,
        }
    }

    /// Insert or update an entry.
    ///
    /// The update is applied only when `version` is strictly greater than the
    /// version already stored for `key`.  Stale updates are silently ignored.
    pub fn upsert(&mut self, key: String, value_hash: u64, version: u64, now_secs: u64) {
        let should_insert = match self.local_state.get(&key) {
            None => true,
            Some(existing) => version > existing.version,
        };

        if should_insert {
            self.local_state.insert(
                key.clone(),
                DigestEntry::new(key, value_hash, version, now_secs),
            );
        }
    }

    /// Build a [`MerkleDigest`] snapshot of the current local state.
    ///
    /// Entries are sorted by key before computing the root hash so that
    /// digests are comparable regardless of `HashMap` ordering.
    pub fn build_digest(&self) -> MerkleDigest {
        let mut entries: Vec<DigestEntry> = self.local_state.values().cloned().collect();
        entries.sort_by(|a, b| a.key.cmp(&b.key));
        MerkleDigest::from_sorted(entries)
    }

    /// Reconcile a local digest against a remote digest.
    ///
    /// Classifies each differing key into one of three buckets:
    /// - `sent_keys`: we are ahead (higher version) or remote is missing the key entirely.
    /// - `requested_keys`: remote is ahead or we are missing the key entirely.
    /// - `conflict_keys`: same version but different `value_hash` — true conflict.
    ///
    /// The total number of keys across all three buckets is capped at
    /// `config.max_diff_keys` to bound message sizes.
    pub fn reconcile(&self, local: &MerkleDigest, remote: &MerkleDigest) -> ReconcileResult {
        if local.matches(remote) {
            return ReconcileResult::default();
        }

        let diff = local.diff_keys(remote);

        // Build lookup maps for O(1) access.
        let local_map: HashMap<&str, &DigestEntry> =
            local.entries.iter().map(|e| (e.key.as_str(), e)).collect();
        let remote_map: HashMap<&str, &DigestEntry> =
            remote.entries.iter().map(|e| (e.key.as_str(), e)).collect();

        let mut result = ReconcileResult::default();
        let mut total = 0usize;

        for key in &diff {
            if total >= self.config.max_diff_keys {
                break;
            }

            let local_entry = local_map.get(key.as_str()).copied();
            let remote_entry = remote_map.get(key.as_str()).copied();

            match (local_entry, remote_entry) {
                // We have it, peer does not → send it.
                (Some(_), None) => {
                    result.sent_keys.push(key.clone());
                    total += 1;
                }
                // Peer has it, we do not → request it.
                (None, Some(_)) => {
                    result.requested_keys.push(key.clone());
                    total += 1;
                }
                // Both have it: compare versions.
                (Some(l), Some(r)) => {
                    if l.version > r.version {
                        // We are ahead.
                        result.sent_keys.push(key.clone());
                        total += 1;
                    } else if r.version > l.version {
                        // Peer is ahead.
                        result.requested_keys.push(key.clone());
                        total += 1;
                    } else {
                        // Same version but different value_hash → conflict.
                        result.conflict_keys.push(key.clone());
                        total += 1;
                    }
                }
                // Both missing — should not occur in practice (diff_keys
                // only returns keys that are in at least one digest), but
                // handle gracefully.
                (None, None) => {}
            }
        }

        result
    }

    /// Apply a remote entry, accepting it if its version is strictly newer.
    pub fn apply_remote_entry(&mut self, entry: DigestEntry) {
        self.upsert(
            entry.key.clone(),
            entry.value_hash,
            entry.version,
            entry.updated_at_secs,
        );
    }

    /// Return basic statistics about the current local state.
    ///
    /// Returns `(entry_count, root_hash)`.
    pub fn stats(&self) -> (usize, u64) {
        let digest = self.build_digest();
        (self.local_state.len(), digest.root_hash)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ae() -> GossipAntiEntropy {
        GossipAntiEntropy::new(AntiEntropyConfig::default())
    }

    // ------------------------------------------------------------------
    // upsert tests
    // ------------------------------------------------------------------

    /// T01 — upsert creates a new entry when the key is absent.
    #[test]
    fn test_upsert_creates_entry() {
        let mut ae = make_ae();
        ae.upsert("key/a".to_string(), 0xABCD, 1, 1000);
        let (count, _) = ae.stats();
        assert_eq!(count, 1);
    }

    /// T02 — upsert with a higher version updates the stored entry.
    #[test]
    fn test_upsert_newer_version_updates() {
        let mut ae = make_ae();
        ae.upsert("key/a".to_string(), 0x0001, 1, 1000);
        ae.upsert("key/a".to_string(), 0x0002, 2, 1001);
        let digest = ae.build_digest();
        assert_eq!(digest.entries.len(), 1);
        assert_eq!(digest.entries[0].value_hash, 0x0002);
        assert_eq!(digest.entries[0].version, 2);
    }

    /// T03 — upsert with an older or equal version must NOT overwrite.
    #[test]
    fn test_upsert_ignores_older_version() {
        let mut ae = make_ae();
        ae.upsert("key/a".to_string(), 0xAAAA, 5, 2000);
        ae.upsert("key/a".to_string(), 0xBBBB, 3, 1000); // stale
        ae.upsert("key/a".to_string(), 0xCCCC, 5, 2001); // same version, different hash
        let digest = ae.build_digest();
        assert_eq!(digest.entries[0].value_hash, 0xAAAA);
        assert_eq!(digest.entries[0].version, 5);
    }

    /// T04 — upsert of multiple distinct keys accumulates all entries.
    #[test]
    fn test_upsert_multiple_keys() {
        let mut ae = make_ae();
        for i in 0u64..5 {
            ae.upsert(format!("key/{}", i), i * 100, i + 1, 1000 + i);
        }
        let (count, _) = ae.stats();
        assert_eq!(count, 5);
    }

    // ------------------------------------------------------------------
    // build_digest / sorting tests
    // ------------------------------------------------------------------

    /// T05 — build_digest returns entries sorted lexicographically by key.
    #[test]
    fn test_build_digest_sorted() {
        let mut ae = make_ae();
        ae.upsert("zebra".to_string(), 1, 1, 0);
        ae.upsert("apple".to_string(), 2, 1, 0);
        ae.upsert("mango".to_string(), 3, 1, 0);
        let digest = ae.build_digest();
        let keys: Vec<&str> = digest.entries.iter().map(|e| e.key.as_str()).collect();
        assert_eq!(keys, vec!["apple", "mango", "zebra"]);
    }

    /// T06 — build_digest on empty state produces zero root_hash and empty entries.
    #[test]
    fn test_build_digest_empty() {
        let ae = make_ae();
        let digest = ae.build_digest();
        assert!(digest.entries.is_empty());
        assert_eq!(digest.root_hash, 0);
    }

    // ------------------------------------------------------------------
    // MerkleDigest::matches
    // ------------------------------------------------------------------

    /// T07 — two digests built from identical state must match.
    #[test]
    fn test_digest_matches_same_state() {
        let mut ae1 = make_ae();
        let mut ae2 = make_ae();
        ae1.upsert("k1".to_string(), 111, 1, 0);
        ae1.upsert("k2".to_string(), 222, 2, 0);
        ae2.upsert("k1".to_string(), 111, 1, 0);
        ae2.upsert("k2".to_string(), 222, 2, 0);
        assert!(ae1.build_digest().matches(&ae2.build_digest()));
    }

    /// T08 — digests with different value_hashes must not match.
    #[test]
    fn test_digest_not_matches_different_hash() {
        let mut ae1 = make_ae();
        let mut ae2 = make_ae();
        ae1.upsert("k1".to_string(), 111, 1, 0);
        ae2.upsert("k1".to_string(), 999, 1, 0);
        assert!(!ae1.build_digest().matches(&ae2.build_digest()));
    }

    // ------------------------------------------------------------------
    // MerkleDigest::diff_keys
    // ------------------------------------------------------------------

    /// T09 — diff_keys returns keys missing from the other digest.
    #[test]
    fn test_diff_keys_finds_missing() {
        let mut ae1 = make_ae();
        let mut ae2 = make_ae();
        ae1.upsert("a".to_string(), 1, 1, 0);
        ae1.upsert("b".to_string(), 2, 1, 0);
        ae2.upsert("a".to_string(), 1, 1, 0);
        // "b" is missing from ae2

        let d1 = ae1.build_digest();
        let d2 = ae2.build_digest();
        let diff = d1.diff_keys(&d2);
        assert_eq!(diff, vec!["b"]);
    }

    /// T10 — diff_keys detects a version difference on a shared key.
    #[test]
    fn test_diff_keys_finds_version_diff() {
        let mut ae1 = make_ae();
        let mut ae2 = make_ae();
        ae1.upsert("x".to_string(), 0xAA, 3, 0);
        ae2.upsert("x".to_string(), 0xAA, 1, 0); // same hash, older version

        let diff = ae1.build_digest().diff_keys(&ae2.build_digest());
        assert_eq!(diff, vec!["x"]);
    }

    /// T11 — diff_keys detects a value_hash difference at the same version.
    #[test]
    fn test_diff_keys_finds_hash_diff_same_version() {
        let mut ae1 = make_ae();
        let mut ae2 = make_ae();
        ae1.upsert("x".to_string(), 0xAA, 5, 0);
        ae2.upsert("x".to_string(), 0xBB, 5, 0); // different hash, same version

        let diff = ae1.build_digest().diff_keys(&ae2.build_digest());
        assert_eq!(diff, vec!["x"]);
    }

    /// T12 — diff_keys returns empty when digests are identical.
    #[test]
    fn test_diff_keys_empty_when_identical() {
        let mut ae = make_ae();
        ae.upsert("p".to_string(), 7, 7, 7);
        let d = ae.build_digest();
        assert!(d.diff_keys(&d.clone()).is_empty());
    }

    // ------------------------------------------------------------------
    // reconcile tests
    // ------------------------------------------------------------------

    /// T13 — reconcile correctly classifies a key that only we have as sent.
    #[test]
    fn test_reconcile_sent_key_local_only() {
        let mut ae = make_ae();
        ae.upsert("local_only".to_string(), 1, 1, 0);
        let local = ae.build_digest();
        let remote = MerkleDigest::from_sorted(vec![]);
        let result = ae.reconcile(&local, &remote);
        assert_eq!(result.sent_keys, vec!["local_only"]);
        assert!(result.requested_keys.is_empty());
        assert!(result.conflict_keys.is_empty());
    }

    /// T14 — reconcile classifies a key that only the remote has as requested.
    #[test]
    fn test_reconcile_requested_key_remote_only() {
        let ae = make_ae();
        let local = MerkleDigest::from_sorted(vec![]);
        let remote =
            MerkleDigest::from_sorted(vec![DigestEntry::new("remote_only".to_string(), 42, 1, 0)]);
        let result = ae.reconcile(&local, &remote);
        assert!(result.sent_keys.is_empty());
        assert_eq!(result.requested_keys, vec!["remote_only"]);
        assert!(result.conflict_keys.is_empty());
    }

    /// T15 — reconcile classifies same-version/different-hash keys as conflicts.
    #[test]
    fn test_reconcile_conflict_same_version_diff_hash() {
        let mut ae = make_ae();
        ae.upsert("conflict_key".to_string(), 0xAA, 5, 0);
        let local = ae.build_digest();
        let remote = MerkleDigest::from_sorted(vec![DigestEntry::new(
            "conflict_key".to_string(),
            0xBB,
            5,
            0,
        )]);
        let result = ae.reconcile(&local, &remote);
        assert!(result.sent_keys.is_empty());
        assert!(result.requested_keys.is_empty());
        assert_eq!(result.conflict_keys, vec!["conflict_key"]);
    }

    /// T16 — reconcile sent when local version is higher.
    #[test]
    fn test_reconcile_sent_local_ahead() {
        let mut ae = make_ae();
        ae.upsert("k".to_string(), 0xAA, 10, 0);
        let local = ae.build_digest();
        let remote = MerkleDigest::from_sorted(vec![DigestEntry::new("k".to_string(), 0xAA, 2, 0)]);
        let result = ae.reconcile(&local, &remote);
        assert_eq!(result.sent_keys, vec!["k"]);
        assert!(result.requested_keys.is_empty());
    }

    /// T17 — reconcile requested when remote version is higher.
    #[test]
    fn test_reconcile_requested_remote_ahead() {
        let mut ae = make_ae();
        ae.upsert("k".to_string(), 0xAA, 2, 0);
        let local = ae.build_digest();
        let remote =
            MerkleDigest::from_sorted(vec![DigestEntry::new("k".to_string(), 0xBB, 10, 0)]);
        let result = ae.reconcile(&local, &remote);
        assert!(result.sent_keys.is_empty());
        assert_eq!(result.requested_keys, vec!["k"]);
    }

    /// T18 — reconcile returns empty result when digests match.
    #[test]
    fn test_reconcile_empty_when_identical() {
        let mut ae = make_ae();
        ae.upsert("same".to_string(), 0x1234, 3, 0);
        let digest = ae.build_digest();
        let result = ae.reconcile(&digest, &digest.clone());
        assert_eq!(result, ReconcileResult::default());
    }

    /// T19 — max_diff_keys cap limits total reconciled keys.
    #[test]
    fn test_reconcile_max_diff_keys_cap() {
        let config = AntiEntropyConfig {
            sync_interval_secs: 30,
            max_diff_keys: 3,
        };
        let mut ae = GossipAntiEntropy::new(config);
        // Add 10 keys that the remote does not have.
        for i in 0u64..10 {
            ae.upsert(format!("key/{:02}", i), i, i + 1, 0);
        }
        let local = ae.build_digest();
        let remote = MerkleDigest::from_sorted(vec![]);
        let result = ae.reconcile(&local, &remote);
        // Total across all buckets must not exceed max_diff_keys.
        let total =
            result.sent_keys.len() + result.requested_keys.len() + result.conflict_keys.len();
        assert!(total <= 3, "expected ≤3, got {}", total);
    }

    // ------------------------------------------------------------------
    // apply_remote_entry
    // ------------------------------------------------------------------

    /// T20 — apply_remote_entry accepts a newer remote entry.
    #[test]
    fn test_apply_remote_entry_updates() {
        let mut ae = make_ae();
        ae.upsert("k".to_string(), 0x01, 1, 0);
        ae.apply_remote_entry(DigestEntry::new("k".to_string(), 0x99, 5, 100));
        let digest = ae.build_digest();
        assert_eq!(digest.entries[0].value_hash, 0x99);
        assert_eq!(digest.entries[0].version, 5);
    }

    /// T21 — apply_remote_entry rejects an older remote entry.
    #[test]
    fn test_apply_remote_entry_rejects_older() {
        let mut ae = make_ae();
        ae.upsert("k".to_string(), 0xFF, 10, 0);
        ae.apply_remote_entry(DigestEntry::new("k".to_string(), 0x00, 3, 50));
        let digest = ae.build_digest();
        assert_eq!(digest.entries[0].value_hash, 0xFF);
        assert_eq!(digest.entries[0].version, 10);
    }

    /// T22 — apply_remote_entry inserts a brand-new key.
    #[test]
    fn test_apply_remote_entry_new_key() {
        let mut ae = make_ae();
        ae.apply_remote_entry(DigestEntry::new("new_key".to_string(), 42, 1, 999));
        let (count, _) = ae.stats();
        assert_eq!(count, 1);
        let digest = ae.build_digest();
        assert_eq!(digest.entries[0].key, "new_key");
    }

    // ------------------------------------------------------------------
    // stats
    // ------------------------------------------------------------------

    /// T23 — stats returns correct entry count and a deterministic root_hash.
    #[test]
    fn test_stats_deterministic() {
        let mut ae1 = make_ae();
        let mut ae2 = make_ae();
        // Insert in different orders.
        ae1.upsert("alpha".to_string(), 1, 1, 0);
        ae1.upsert("beta".to_string(), 2, 1, 0);
        ae2.upsert("beta".to_string(), 2, 1, 0);
        ae2.upsert("alpha".to_string(), 1, 1, 0);
        let (c1, h1) = ae1.stats();
        let (c2, h2) = ae2.stats();
        assert_eq!(c1, c2);
        assert_eq!(h1, h2);
    }

    // ------------------------------------------------------------------
    // Empty-digest reconciliation
    // ------------------------------------------------------------------

    /// T24 — reconciling two empty digests produces an empty result.
    #[test]
    fn test_reconcile_both_empty() {
        let ae = make_ae();
        let empty = MerkleDigest::from_sorted(vec![]);
        let result = ae.reconcile(&empty, &empty.clone());
        assert_eq!(result, ReconcileResult::default());
    }

    // ------------------------------------------------------------------
    // Root-hash stability
    // ------------------------------------------------------------------

    /// T25 — root_hash changes when a value_hash is updated.
    #[test]
    fn test_root_hash_changes_on_update() {
        let mut ae = make_ae();
        ae.upsert("key".to_string(), 0x1111, 1, 0);
        let h1 = ae.build_digest().root_hash;
        ae.upsert("key".to_string(), 0x2222, 2, 0);
        let h2 = ae.build_digest().root_hash;
        assert_ne!(h1, h2);
    }

    /// T26 — reconcile mixed bag: some sent, some requested, some conflicts, capped.
    #[test]
    fn test_reconcile_mixed_classification() {
        let config = AntiEntropyConfig {
            sync_interval_secs: 30,
            max_diff_keys: 100,
        };
        let mut ae = GossipAntiEntropy::new(config);
        // local_only: we have it, remote does not.
        ae.upsert("local_only".to_string(), 1, 1, 0);
        // shared_ahead: we have higher version.
        ae.upsert("shared_ahead".to_string(), 10, 10, 0);
        // shared_conflict: same version, different hash.
        ae.upsert("shared_conflict".to_string(), 0xAA, 5, 0);

        let local = ae.build_digest();
        let remote = MerkleDigest::from_sorted(vec![
            DigestEntry::new("remote_only".to_string(), 99, 1, 0),
            DigestEntry::new("shared_ahead".to_string(), 10, 2, 0), // older version
            DigestEntry::new("shared_conflict".to_string(), 0xBB, 5, 0), // same version, different hash
        ]);

        let result = ae.reconcile(&local, &remote);
        assert!(result.sent_keys.contains(&"local_only".to_string()));
        assert!(result.sent_keys.contains(&"shared_ahead".to_string()));
        assert!(result.requested_keys.contains(&"remote_only".to_string()));
        assert!(result
            .conflict_keys
            .contains(&"shared_conflict".to_string()));
    }
}
