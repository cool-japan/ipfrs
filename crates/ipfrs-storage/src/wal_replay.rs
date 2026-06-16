//! WAL Replay Engine — crash recovery and state reconstruction from WAL entries.
//!
//! [`StorageWALReplay`] replays a sequence of [`WalEntry`] records to reconstruct
//! in-memory storage state, supporting full replay, checkpoint-scoped replay,
//! sequence-bounded replay, and transactional semantics (Begin/Commit/Rollback).

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// FNV-1a 32-bit checksum (standalone, no external dep)
// ---------------------------------------------------------------------------

/// FNV-1a 32-bit hash used for WAL entry checksums.
#[inline]
fn fnv1a_32_bytes(data: &[u8]) -> u32 {
    const OFFSET_BASIS: u32 = 2_166_136_261;
    const PRIME: u32 = 16_777_619;
    let mut h = OFFSET_BASIS;
    for &b in data {
        h ^= u32::from(b);
        h = h.wrapping_mul(PRIME);
    }
    h
}

// ---------------------------------------------------------------------------
// WalEntryType
// ---------------------------------------------------------------------------

/// Discriminates the semantic role of a [`WalEntry`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WalEntryType {
    /// Write a key-value pair.
    Put,
    /// Remove a key.
    Delete,
    /// Mark a durable checkpoint that replay may start from.
    Checkpoint,
    /// Open a new transaction.
    Begin,
    /// Commit an open transaction, flushing its buffered ops.
    Commit,
    /// Rollback an open transaction, discarding its buffered ops.
    Rollback,
}

// ---------------------------------------------------------------------------
// WalEntry
// ---------------------------------------------------------------------------

/// A single record in the write-ahead log.
#[derive(Debug, Clone)]
pub struct WalEntry {
    /// Monotonically increasing, globally unique sequence number.
    pub sequence: u64,
    /// Semantic type of this entry.
    pub entry_type: WalEntryType,
    /// Key this entry addresses (empty for Begin/Commit/Rollback).
    pub key: String,
    /// Optional payload for Put operations; `None` for Delete / control entries.
    pub value: Option<Vec<u8>>,
    /// Transaction this entry belongs to, if any.
    pub transaction_id: Option<u64>,
    /// FNV-1a 32-bit checksum over `(sequence, key, value)`.
    pub checksum: u32,
}

impl WalEntry {
    /// Computes FNV-1a 32-bit checksum over `sequence || key || value`.
    pub fn compute_checksum(seq: u64, key: &str, value: &Option<Vec<u8>>) -> u32 {
        let mut buf: Vec<u8> =
            Vec::with_capacity(8 + key.len() + value.as_deref().map_or(0, |v| v.len()));
        buf.extend_from_slice(&seq.to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        if let Some(v) = value {
            buf.extend_from_slice(v);
        }
        fnv1a_32_bytes(&buf)
    }

    /// Returns `true` iff the stored checksum matches the recomputed one.
    pub fn is_valid(&self) -> bool {
        self.checksum == Self::compute_checksum(self.sequence, &self.key, &self.value)
    }
}

// ---------------------------------------------------------------------------
// ReplayPolicy
// ---------------------------------------------------------------------------

/// Controls which entries are included in a replay run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayPolicy {
    /// Process every entry from the oldest.
    Full,
    /// Start from the last Checkpoint entry (inclusive).
    SinceCheckpoint,
    /// Skip entries whose sequence number is strictly less than `s`.
    SinceSequence(u64),
    /// Replay only the last `n` entries.
    LastN(usize),
}

// ---------------------------------------------------------------------------
// ReplayStats
// ---------------------------------------------------------------------------

/// Counters collected during a single [`StorageWALReplay::replay`] invocation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReplayStats {
    /// Total WAL entries examined.
    pub entries_read: usize,
    /// Entries whose effects were applied to the reconstructed state.
    pub entries_applied: usize,
    /// Entries skipped due to policy (before the replay window).
    pub entries_skipped: usize,
    /// Checkpoint entries encountered in the replay window.
    pub checkpoints_found: usize,
    /// Entries with a bad checksum that were discarded.
    pub invalid_entries: usize,
    /// Sequence number of the last applied entry (0 if none applied).
    pub final_sequence: u64,
}

// ---------------------------------------------------------------------------
// ReplayState
// ---------------------------------------------------------------------------

/// The reconstructed storage state produced by [`StorageWALReplay::replay`].
#[derive(Debug, Clone, Default)]
pub struct ReplayState {
    /// Final key-value store after applying all replayed entries.
    pub store: HashMap<String, Vec<u8>>,
    /// Sequence number of the last entry that was applied.
    pub sequence: u64,
    /// Transactions still open (Begin seen, no Commit/Rollback yet).
    pub active_transactions: HashMap<u64, Vec<WalEntry>>,
    /// Number of transactions successfully committed.
    pub committed_count: u64,
    /// Number of transactions rolled back.
    pub rolled_back_count: u64,
}

// ---------------------------------------------------------------------------
// WalStats
// ---------------------------------------------------------------------------

/// Aggregate statistics describing the current state of a [`StorageWALReplay`] log.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WalStats {
    /// Total number of entries currently in the log.
    pub total_entries: usize,
    /// Number of Checkpoint entries in the log.
    pub checkpoint_count: usize,
    /// Sequence number of the oldest entry (0 if empty).
    pub oldest_sequence: u64,
    /// Sequence number of the newest entry (0 if empty).
    pub newest_sequence: u64,
    /// Rough byte estimate of all entries (keys + values + fixed overhead).
    pub estimated_replay_size: usize,
}

// ---------------------------------------------------------------------------
// StorageWALReplay
// ---------------------------------------------------------------------------

/// Write-ahead log replay engine.
///
/// Maintains an in-memory ring buffer of [`WalEntry`] records.  When the ring
/// exceeds `max_entries` the oldest entry is evicted automatically.  Use
/// [`replay`](Self::replay) to reconstruct storage state from the buffered
/// entries according to a chosen [`ReplayPolicy`].
pub struct StorageWALReplay {
    /// Ring-buffer of WAL entries, oldest first.
    entries: VecDeque<WalEntry>,
    /// Maximum number of entries to retain.
    max_entries: usize,
    /// Next sequence number to assign.
    next_sequence: u64,
    /// Sequence numbers of all Checkpoint entries currently in the buffer.
    checkpoint_sequences: Vec<u64>,
}

impl StorageWALReplay {
    /// Creates a new [`StorageWALReplay`] that retains at most `max_entries`.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries: max_entries.max(1),
            next_sequence: 1,
            checkpoint_sequences: Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // append
    // -----------------------------------------------------------------------

    /// Appends a new entry to the log and returns it.
    ///
    /// - Assigns the next monotonic sequence number.
    /// - Computes and stores the FNV-1a checksum.
    /// - If `entry_type` is [`WalEntryType::Checkpoint`], records the sequence
    ///   in `checkpoint_sequences`.
    /// - Evicts the oldest entry when the buffer is at capacity.
    pub fn append(
        &mut self,
        entry_type: WalEntryType,
        key: String,
        value: Option<Vec<u8>>,
        tx_id: Option<u64>,
    ) -> WalEntry {
        let seq = self.next_sequence;
        self.next_sequence += 1;

        let checksum = WalEntry::compute_checksum(seq, &key, &value);
        let is_checkpoint = entry_type == WalEntryType::Checkpoint;

        let entry = WalEntry {
            sequence: seq,
            entry_type,
            key,
            value,
            transaction_id: tx_id,
            checksum,
        };

        // Evict oldest if at capacity.
        if self.entries.len() >= self.max_entries {
            if let Some(evicted) = self.entries.pop_front() {
                // Remove evicted checkpoint sequence if present.
                self.checkpoint_sequences.retain(|&s| s != evicted.sequence);
            }
        }

        if is_checkpoint {
            self.checkpoint_sequences.push(seq);
        }

        self.entries.push_back(entry.clone());
        entry
    }

    // -----------------------------------------------------------------------
    // replay
    // -----------------------------------------------------------------------

    /// Reconstructs storage state by replaying entries according to `policy`.
    ///
    /// Returns `(state, stats)`.
    pub fn replay(&self, policy: ReplayPolicy) -> (ReplayState, ReplayStats) {
        let mut stats = ReplayStats::default();
        let mut state = ReplayState::default();

        // Collect the replay window as a Vec<&WalEntry> to avoid lifetime issues
        // with boxed heterogeneous iterators.
        let total = self.entries.len();
        let window_entries: Vec<&WalEntry> = match policy {
            ReplayPolicy::Full => {
                stats.entries_read = total;
                self.entries.iter().collect()
            }
            ReplayPolicy::SinceCheckpoint => {
                // Find the last checkpoint sequence that still exists in the buffer.
                let start_seq = self
                    .checkpoint_sequences
                    .iter()
                    .copied()
                    .rev()
                    .find(|&s| self.entries.iter().any(|e| e.sequence == s))
                    .unwrap_or(0);
                let skip_count = self
                    .entries
                    .iter()
                    .filter(|e| e.sequence < start_seq)
                    .count();
                stats.entries_skipped = skip_count;
                stats.entries_read = total.saturating_sub(skip_count);
                self.entries
                    .iter()
                    .filter(|e| e.sequence >= start_seq)
                    .collect()
            }
            ReplayPolicy::SinceSequence(min_seq) => {
                let skip_count = self.entries.iter().filter(|e| e.sequence < min_seq).count();
                stats.entries_skipped = skip_count;
                stats.entries_read = total.saturating_sub(skip_count);
                self.entries
                    .iter()
                    .filter(|e| e.sequence >= min_seq)
                    .collect()
            }
            ReplayPolicy::LastN(n) => {
                let skip_count = total.saturating_sub(n);
                let take = total.min(n);
                stats.entries_skipped = skip_count;
                stats.entries_read = take;
                self.entries.iter().skip(skip_count).take(take).collect()
            }
        };

        for entry in window_entries {
            // Validate checksum; discard and count corrupt entries.
            if !entry.is_valid() {
                stats.invalid_entries += 1;
                continue;
            }

            match &entry.entry_type {
                WalEntryType::Checkpoint => {
                    stats.checkpoints_found += 1;
                    stats.entries_applied += 1;
                    state.sequence = entry.sequence;
                    stats.final_sequence = entry.sequence;
                }
                WalEntryType::Begin => {
                    if let Some(tx_id) = entry.transaction_id {
                        state.active_transactions.entry(tx_id).or_default();
                    }
                    stats.entries_applied += 1;
                    state.sequence = entry.sequence;
                    stats.final_sequence = entry.sequence;
                }
                WalEntryType::Commit => {
                    if let Some(tx_id) = entry.transaction_id {
                        if let Some(buffered) = state.active_transactions.remove(&tx_id) {
                            for buffered_entry in buffered {
                                match buffered_entry.entry_type {
                                    WalEntryType::Put => {
                                        if let Some(v) = buffered_entry.value {
                                            state.store.insert(buffered_entry.key, v);
                                        }
                                    }
                                    WalEntryType::Delete => {
                                        state.store.remove(&buffered_entry.key);
                                    }
                                    _ => {}
                                }
                            }
                            state.committed_count += 1;
                        }
                    }
                    stats.entries_applied += 1;
                    state.sequence = entry.sequence;
                    stats.final_sequence = entry.sequence;
                }
                WalEntryType::Rollback => {
                    if let Some(tx_id) = entry.transaction_id {
                        state.active_transactions.remove(&tx_id);
                        state.rolled_back_count += 1;
                    }
                    stats.entries_applied += 1;
                    state.sequence = entry.sequence;
                    stats.final_sequence = entry.sequence;
                }
                WalEntryType::Put => {
                    if let Some(tx_id) = entry.transaction_id {
                        state
                            .active_transactions
                            .entry(tx_id)
                            .or_default()
                            .push(entry.clone());
                    } else if let Some(v) = &entry.value {
                        state.store.insert(entry.key.clone(), v.clone());
                    }
                    stats.entries_applied += 1;
                    state.sequence = entry.sequence;
                    stats.final_sequence = entry.sequence;
                }
                WalEntryType::Delete => {
                    if let Some(tx_id) = entry.transaction_id {
                        state
                            .active_transactions
                            .entry(tx_id)
                            .or_default()
                            .push(entry.clone());
                    } else {
                        state.store.remove(&entry.key);
                    }
                    stats.entries_applied += 1;
                    state.sequence = entry.sequence;
                    stats.final_sequence = entry.sequence;
                }
            }
        }

        (state, stats)
    }

    // -----------------------------------------------------------------------
    // verify_entries
    // -----------------------------------------------------------------------

    /// Verifies all entries in the log.
    ///
    /// Returns `(valid_count, invalid_count)`.
    pub fn verify_entries(&self) -> (usize, usize) {
        let mut valid = 0usize;
        let mut invalid = 0usize;
        for entry in &self.entries {
            if entry.is_valid() {
                valid += 1;
            } else {
                invalid += 1;
            }
        }
        (valid, invalid)
    }

    // -----------------------------------------------------------------------
    // last_checkpoint_sequence
    // -----------------------------------------------------------------------

    /// Returns the sequence number of the most recent Checkpoint entry, or `None`.
    pub fn last_checkpoint_sequence(&self) -> Option<u64> {
        self.checkpoint_sequences.last().copied()
    }

    // -----------------------------------------------------------------------
    // truncate_before
    // -----------------------------------------------------------------------

    /// Removes all entries with sequence number strictly less than `sequence`.
    ///
    /// Returns the number of entries removed.
    pub fn truncate_before(&mut self, sequence: u64) -> usize {
        let before = self.entries.len();
        self.entries.retain(|e| e.sequence >= sequence);
        self.checkpoint_sequences.retain(|&s| s >= sequence);
        before - self.entries.len()
    }

    // -----------------------------------------------------------------------
    // wal_stats
    // -----------------------------------------------------------------------

    /// Returns aggregate statistics describing the current WAL buffer.
    pub fn wal_stats(&self) -> WalStats {
        let total_entries = self.entries.len();
        let checkpoint_count = self
            .entries
            .iter()
            .filter(|e| e.entry_type == WalEntryType::Checkpoint)
            .count();

        let oldest_sequence = self.entries.front().map_or(0, |e| e.sequence);
        let newest_sequence = self.entries.back().map_or(0, |e| e.sequence);

        let estimated_replay_size = self
            .entries
            .iter()
            .map(|e| {
                // Fixed overhead + key + value
                32usize + e.key.len() + e.value.as_deref().map_or(0, |v| v.len())
            })
            .sum();

        WalStats {
            total_entries,
            checkpoint_count,
            oldest_sequence,
            newest_sequence,
            estimated_replay_size,
        }
    }

    // -----------------------------------------------------------------------
    // Accessor helpers
    // -----------------------------------------------------------------------

    /// Returns the number of entries currently buffered.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the log contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the next sequence number that will be assigned.
    pub fn next_sequence(&self) -> u64 {
        self.next_sequence
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{ReplayPolicy, StorageWALReplay, WalEntry, WalEntryType};

    // -----------------------------------------------------------------------
    // Helper builders
    // -----------------------------------------------------------------------

    fn make_replay(max: usize) -> StorageWALReplay {
        StorageWALReplay::new(max)
    }

    fn put(log: &mut StorageWALReplay, key: &str, val: &[u8]) -> WalEntry {
        log.append(WalEntryType::Put, key.to_string(), Some(val.to_vec()), None)
    }

    fn delete(log: &mut StorageWALReplay, key: &str) -> WalEntry {
        log.append(WalEntryType::Delete, key.to_string(), None, None)
    }

    fn checkpoint(log: &mut StorageWALReplay) -> WalEntry {
        log.append(WalEntryType::Checkpoint, String::new(), None, None)
    }

    fn begin(log: &mut StorageWALReplay, tx: u64) -> WalEntry {
        log.append(WalEntryType::Begin, String::new(), None, Some(tx))
    }

    fn commit(log: &mut StorageWALReplay, tx: u64) -> WalEntry {
        log.append(WalEntryType::Commit, String::new(), None, Some(tx))
    }

    fn rollback(log: &mut StorageWALReplay, tx: u64) -> WalEntry {
        log.append(WalEntryType::Rollback, String::new(), None, Some(tx))
    }

    fn tx_put(log: &mut StorageWALReplay, key: &str, val: &[u8], tx: u64) -> WalEntry {
        log.append(
            WalEntryType::Put,
            key.to_string(),
            Some(val.to_vec()),
            Some(tx),
        )
    }

    fn tx_delete(log: &mut StorageWALReplay, key: &str, tx: u64) -> WalEntry {
        log.append(WalEntryType::Delete, key.to_string(), None, Some(tx))
    }

    // -----------------------------------------------------------------------
    // 1. Basic construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_is_empty() {
        let log = make_replay(100);
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
        assert_eq!(log.next_sequence(), 1);
    }

    // -----------------------------------------------------------------------
    // 2. Sequence assignment
    // -----------------------------------------------------------------------

    #[test]
    fn test_sequence_increments() {
        let mut log = make_replay(100);
        let e1 = put(&mut log, "a", b"1");
        let e2 = put(&mut log, "b", b"2");
        let e3 = put(&mut log, "c", b"3");
        assert_eq!(e1.sequence, 1);
        assert_eq!(e2.sequence, 2);
        assert_eq!(e3.sequence, 3);
        assert_eq!(log.next_sequence(), 4);
    }

    // -----------------------------------------------------------------------
    // 3. Checksum validity
    // -----------------------------------------------------------------------

    #[test]
    fn test_checksum_valid_on_append() {
        let mut log = make_replay(100);
        let entry = put(&mut log, "key", b"value");
        assert!(entry.is_valid());
    }

    #[test]
    fn test_checksum_detects_tampering() {
        let mut entry = WalEntry {
            sequence: 1,
            entry_type: WalEntryType::Put,
            key: "k".to_string(),
            value: Some(b"v".to_vec()),
            transaction_id: None,
            checksum: WalEntry::compute_checksum(1, "k", &Some(b"v".to_vec())),
        };
        assert!(entry.is_valid());
        // Tamper with the key.
        entry.key = "tampered".to_string();
        assert!(!entry.is_valid());
    }

    #[test]
    fn test_checksum_value_none_vs_some() {
        let cs_none = WalEntry::compute_checksum(1, "k", &None);
        let cs_some = WalEntry::compute_checksum(1, "k", &Some(b"data".to_vec()));
        assert_ne!(cs_none, cs_some);
    }

    // -----------------------------------------------------------------------
    // 4. verify_entries
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_all_valid() {
        let mut log = make_replay(100);
        put(&mut log, "a", b"1");
        put(&mut log, "b", b"2");
        let (valid, invalid) = log.verify_entries();
        assert_eq!(valid, 2);
        assert_eq!(invalid, 0);
    }

    // -----------------------------------------------------------------------
    // 5. Full replay — basic Put
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_replay_puts() {
        let mut log = make_replay(100);
        put(&mut log, "x", b"hello");
        put(&mut log, "y", b"world");
        let (state, stats) = log.replay(ReplayPolicy::Full);
        assert_eq!(
            state.store.get("x").map(|v| v.as_slice()),
            Some(b"hello".as_ref())
        );
        assert_eq!(
            state.store.get("y").map(|v| v.as_slice()),
            Some(b"world".as_ref())
        );
        assert_eq!(stats.entries_applied, 2);
        assert_eq!(stats.entries_skipped, 0);
        assert_eq!(stats.invalid_entries, 0);
        assert_eq!(stats.final_sequence, 2);
    }

    // -----------------------------------------------------------------------
    // 6. Full replay — Put then Delete
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_replay_delete_removes_key() {
        let mut log = make_replay(100);
        put(&mut log, "k", b"val");
        delete(&mut log, "k");
        let (state, stats) = log.replay(ReplayPolicy::Full);
        assert!(!state.store.contains_key("k"));
        assert_eq!(stats.entries_applied, 2);
    }

    // -----------------------------------------------------------------------
    // 7. Full replay — overwrite key
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_replay_overwrite() {
        let mut log = make_replay(100);
        put(&mut log, "k", b"v1");
        put(&mut log, "k", b"v2");
        let (state, _) = log.replay(ReplayPolicy::Full);
        assert_eq!(state.store["k"], b"v2");
    }

    // -----------------------------------------------------------------------
    // 8. Checkpoint replay — SinceCheckpoint
    // -----------------------------------------------------------------------

    #[test]
    fn test_since_checkpoint_skips_pre_checkpoint() {
        let mut log = make_replay(100);
        put(&mut log, "before", b"old");
        checkpoint(&mut log);
        put(&mut log, "after", b"new");
        let (state, stats) = log.replay(ReplayPolicy::SinceCheckpoint);
        // "before" was put before the checkpoint so it may be included since
        // SinceCheckpoint starts AT the checkpoint entry (seq 2), not after it.
        // The checkpoint itself is at seq=2, "before" is at seq=1 → skipped.
        assert!(!state.store.contains_key("before"));
        assert!(state.store.contains_key("after"));
        assert_eq!(stats.entries_skipped, 1);
        assert_eq!(stats.checkpoints_found, 1);
    }

    // -----------------------------------------------------------------------
    // 9. SinceCheckpoint — no checkpoint falls back to empty replay
    // -----------------------------------------------------------------------

    #[test]
    fn test_since_checkpoint_no_checkpoint_replays_all() {
        let mut log = make_replay(100);
        put(&mut log, "a", b"1");
        put(&mut log, "b", b"2");
        // No checkpoint → start_seq = 0, everything replayed.
        let (state, stats) = log.replay(ReplayPolicy::SinceCheckpoint);
        assert_eq!(state.store.len(), 2);
        assert_eq!(stats.entries_skipped, 0);
    }

    // -----------------------------------------------------------------------
    // 10. SinceSequence replay
    // -----------------------------------------------------------------------

    #[test]
    fn test_since_sequence_skips_older() {
        let mut log = make_replay(100);
        put(&mut log, "a", b"1"); // seq 1
        put(&mut log, "b", b"2"); // seq 2
        put(&mut log, "c", b"3"); // seq 3
        let (state, stats) = log.replay(ReplayPolicy::SinceSequence(2));
        assert!(!state.store.contains_key("a"));
        assert!(state.store.contains_key("b"));
        assert!(state.store.contains_key("c"));
        assert_eq!(stats.entries_skipped, 1);
        assert_eq!(stats.entries_applied, 2);
    }

    // -----------------------------------------------------------------------
    // 11. LastN replay
    // -----------------------------------------------------------------------

    #[test]
    fn test_last_n_replay() {
        let mut log = make_replay(100);
        put(&mut log, "a", b"1");
        put(&mut log, "b", b"2");
        put(&mut log, "c", b"3");
        put(&mut log, "d", b"4");
        let (state, stats) = log.replay(ReplayPolicy::LastN(2));
        // Only last 2: "c" and "d".
        assert!(!state.store.contains_key("a"));
        assert!(!state.store.contains_key("b"));
        assert!(state.store.contains_key("c"));
        assert!(state.store.contains_key("d"));
        assert_eq!(stats.entries_skipped, 2);
        assert_eq!(stats.entries_applied, 2);
    }

    #[test]
    fn test_last_n_larger_than_log() {
        let mut log = make_replay(100);
        put(&mut log, "a", b"1");
        put(&mut log, "b", b"2");
        let (state, stats) = log.replay(ReplayPolicy::LastN(50));
        assert_eq!(state.store.len(), 2);
        assert_eq!(stats.entries_skipped, 0);
        assert_eq!(stats.entries_applied, 2);
    }

    // -----------------------------------------------------------------------
    // 12. Eviction when at capacity
    // -----------------------------------------------------------------------

    #[test]
    fn test_eviction_at_capacity() {
        let mut log = make_replay(3);
        put(&mut log, "a", b"1"); // seq 1
        put(&mut log, "b", b"2"); // seq 2
        put(&mut log, "c", b"3"); // seq 3
                                  // Now at capacity — next append evicts seq 1.
        put(&mut log, "d", b"4"); // seq 4
        assert_eq!(log.len(), 3);
        let (state, _) = log.replay(ReplayPolicy::Full);
        assert!(!state.store.contains_key("a"));
        assert!(state.store.contains_key("b"));
        assert!(state.store.contains_key("c"));
        assert!(state.store.contains_key("d"));
    }

    // -----------------------------------------------------------------------
    // 13. Checkpoint eviction
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_evicted_from_sequences() {
        let mut log = make_replay(3);
        checkpoint(&mut log); // seq 1 — evicted when seq 4 arrives
        put(&mut log, "a", b"1");
        put(&mut log, "b", b"2");
        // seq 4 evicts seq 1 (the checkpoint).
        put(&mut log, "c", b"3");
        // The checkpoint at seq=1 should no longer be tracked.
        assert!(log.last_checkpoint_sequence().is_none());
    }

    // -----------------------------------------------------------------------
    // 14. last_checkpoint_sequence
    // -----------------------------------------------------------------------

    #[test]
    fn test_last_checkpoint_sequence() {
        let mut log = make_replay(100);
        assert_eq!(log.last_checkpoint_sequence(), None);
        checkpoint(&mut log); // seq 1
        assert_eq!(log.last_checkpoint_sequence(), Some(1));
        put(&mut log, "k", b"v");
        checkpoint(&mut log); // seq 3
        assert_eq!(log.last_checkpoint_sequence(), Some(3));
    }

    // -----------------------------------------------------------------------
    // 15. truncate_before
    // -----------------------------------------------------------------------

    #[test]
    fn test_truncate_before_removes_entries() {
        let mut log = make_replay(100);
        put(&mut log, "a", b"1"); // seq 1
        put(&mut log, "b", b"2"); // seq 2
        put(&mut log, "c", b"3"); // seq 3
        let removed = log.truncate_before(2);
        assert_eq!(removed, 1);
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_truncate_before_removes_checkpoint_sequences() {
        let mut log = make_replay(100);
        checkpoint(&mut log); // seq 1
        put(&mut log, "a", b"1");
        checkpoint(&mut log); // seq 3
        let removed = log.truncate_before(3);
        assert_eq!(removed, 2); // seq 1 checkpoint + seq 2 put
                                // Only checkpoint at seq 3 should remain.
        assert_eq!(log.last_checkpoint_sequence(), Some(3));
    }

    #[test]
    fn test_truncate_before_zero_removes_nothing() {
        let mut log = make_replay(100);
        put(&mut log, "a", b"1");
        put(&mut log, "b", b"2");
        let removed = log.truncate_before(0);
        assert_eq!(removed, 0);
        assert_eq!(log.len(), 2);
    }

    // -----------------------------------------------------------------------
    // 16. wal_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_wal_stats_empty() {
        let log = make_replay(100);
        let s = log.wal_stats();
        assert_eq!(s.total_entries, 0);
        assert_eq!(s.checkpoint_count, 0);
        assert_eq!(s.oldest_sequence, 0);
        assert_eq!(s.newest_sequence, 0);
        assert_eq!(s.estimated_replay_size, 0);
    }

    #[test]
    fn test_wal_stats_with_entries() {
        let mut log = make_replay(100);
        put(&mut log, "a", b"hello");
        checkpoint(&mut log);
        put(&mut log, "b", b"world");
        let s = log.wal_stats();
        assert_eq!(s.total_entries, 3);
        assert_eq!(s.checkpoint_count, 1);
        assert_eq!(s.oldest_sequence, 1);
        assert_eq!(s.newest_sequence, 3);
        assert!(s.estimated_replay_size > 0);
    }

    // -----------------------------------------------------------------------
    // 17. Transaction: Begin + Put + Commit
    // -----------------------------------------------------------------------

    #[test]
    fn test_transaction_commit_applies_ops() {
        let mut log = make_replay(100);
        begin(&mut log, 42);
        tx_put(&mut log, "k", b"value", 42);
        commit(&mut log, 42);
        let (state, stats) = log.replay(ReplayPolicy::Full);
        assert_eq!(
            state.store.get("k").map(|v| v.as_slice()),
            Some(b"value".as_ref())
        );
        assert_eq!(state.committed_count, 1);
        assert!(state.active_transactions.is_empty());
        assert_eq!(stats.entries_applied, 3);
    }

    // -----------------------------------------------------------------------
    // 18. Transaction: Begin + Put + Rollback
    // -----------------------------------------------------------------------

    #[test]
    fn test_transaction_rollback_discards_ops() {
        let mut log = make_replay(100);
        begin(&mut log, 7);
        tx_put(&mut log, "k", b"should_not_appear", 7);
        rollback(&mut log, 7);
        let (state, stats) = log.replay(ReplayPolicy::Full);
        assert!(!state.store.contains_key("k"));
        assert_eq!(state.rolled_back_count, 1);
        assert_eq!(stats.entries_applied, 3);
    }

    // -----------------------------------------------------------------------
    // 19. Transaction: Delete buffered and committed
    // -----------------------------------------------------------------------

    #[test]
    fn test_transaction_commit_delete() {
        let mut log = make_replay(100);
        // Pre-populate outside any tx.
        put(&mut log, "k", b"v");
        begin(&mut log, 1);
        tx_delete(&mut log, "k", 1);
        commit(&mut log, 1);
        let (state, _) = log.replay(ReplayPolicy::Full);
        assert!(!state.store.contains_key("k"));
    }

    // -----------------------------------------------------------------------
    // 20. Multiple concurrent transactions
    // -----------------------------------------------------------------------

    #[test]
    fn test_two_concurrent_transactions() {
        let mut log = make_replay(100);
        begin(&mut log, 1);
        begin(&mut log, 2);
        tx_put(&mut log, "a", b"from-tx1", 1);
        tx_put(&mut log, "b", b"from-tx2", 2);
        commit(&mut log, 1);
        rollback(&mut log, 2);
        let (state, _) = log.replay(ReplayPolicy::Full);
        assert_eq!(
            state.store.get("a").map(|v| v.as_slice()),
            Some(b"from-tx1".as_ref())
        );
        assert!(!state.store.contains_key("b"));
        assert_eq!(state.committed_count, 1);
        assert_eq!(state.rolled_back_count, 1);
    }

    // -----------------------------------------------------------------------
    // 21. Transaction left open (no Commit/Rollback)
    // -----------------------------------------------------------------------

    #[test]
    fn test_open_transaction_remains_in_active() {
        let mut log = make_replay(100);
        begin(&mut log, 99);
        tx_put(&mut log, "k", b"v", 99);
        // No commit or rollback.
        let (state, _) = log.replay(ReplayPolicy::Full);
        assert!(!state.store.contains_key("k"));
        assert!(state.active_transactions.contains_key(&99));
    }

    // -----------------------------------------------------------------------
    // 22. Invalid entry skipped in stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_entry_counted_not_applied() {
        let mut log = make_replay(100);
        // Append a valid entry.
        put(&mut log, "valid_key", b"v");
        // Manually push a corrupt entry.
        log.entries.push_back(WalEntry {
            sequence: 999,
            entry_type: WalEntryType::Put,
            key: "bad_key".to_string(),
            value: Some(b"data".to_vec()),
            transaction_id: None,
            checksum: 0xDEAD_BEEF, // deliberately wrong
        });
        let (state, stats) = log.replay(ReplayPolicy::Full);
        assert_eq!(stats.invalid_entries, 1);
        assert!(!state.store.contains_key("bad_key"));
        assert!(state.store.contains_key("valid_key"));
    }

    // -----------------------------------------------------------------------
    // 23. Empty log replay
    // -----------------------------------------------------------------------

    #[test]
    fn test_replay_empty_log() {
        let log = make_replay(100);
        let (state, stats) = log.replay(ReplayPolicy::Full);
        assert!(state.store.is_empty());
        assert_eq!(stats.entries_read, 0);
        assert_eq!(stats.entries_applied, 0);
        assert_eq!(stats.final_sequence, 0);
    }

    // -----------------------------------------------------------------------
    // 24. SinceSequence with exact boundary
    // -----------------------------------------------------------------------

    #[test]
    fn test_since_sequence_exact_boundary() {
        let mut log = make_replay(100);
        put(&mut log, "a", b"1"); // seq 1
        put(&mut log, "b", b"2"); // seq 2
                                  // SinceSequence(1) should include everything.
        let (state, stats) = log.replay(ReplayPolicy::SinceSequence(1));
        assert_eq!(state.store.len(), 2);
        assert_eq!(stats.entries_skipped, 0);
    }

    // -----------------------------------------------------------------------
    // 25. Checkpoint count in stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_replay_stats_checkpoint_count() {
        let mut log = make_replay(100);
        checkpoint(&mut log);
        put(&mut log, "a", b"1");
        checkpoint(&mut log);
        let (_, stats) = log.replay(ReplayPolicy::Full);
        assert_eq!(stats.checkpoints_found, 2);
    }

    // -----------------------------------------------------------------------
    // 26. wal_stats estimated_replay_size
    // -----------------------------------------------------------------------

    #[test]
    fn test_wal_stats_estimated_size_increases_with_entries() {
        let mut log = make_replay(100);
        let s0 = log.wal_stats().estimated_replay_size;
        put(&mut log, "key", b"some_value");
        let s1 = log.wal_stats().estimated_replay_size;
        assert!(s1 > s0);
    }

    // -----------------------------------------------------------------------
    // 27. Replay SinceCheckpoint — multiple checkpoints uses last one
    // -----------------------------------------------------------------------

    #[test]
    fn test_since_checkpoint_uses_latest() {
        let mut log = make_replay(100);
        put(&mut log, "early", b"e"); // seq 1
        checkpoint(&mut log); // seq 2
        put(&mut log, "mid", b"m"); // seq 3
        checkpoint(&mut log); // seq 4
        put(&mut log, "late", b"l"); // seq 5
        let (state, stats) = log.replay(ReplayPolicy::SinceCheckpoint);
        // Should start from seq 4 (last checkpoint).
        assert!(!state.store.contains_key("early"));
        assert!(!state.store.contains_key("mid"));
        assert!(state.store.contains_key("late"));
        assert_eq!(stats.checkpoints_found, 1); // only the last checkpoint in window
    }

    // -----------------------------------------------------------------------
    // 28. LastN(0) replays nothing
    // -----------------------------------------------------------------------

    #[test]
    fn test_last_n_zero() {
        let mut log = make_replay(100);
        put(&mut log, "a", b"1");
        put(&mut log, "b", b"2");
        let (state, stats) = log.replay(ReplayPolicy::LastN(0));
        assert!(state.store.is_empty());
        assert_eq!(stats.entries_applied, 0);
        assert_eq!(stats.entries_skipped, 2);
    }

    // -----------------------------------------------------------------------
    // 29. Truncate then replay
    // -----------------------------------------------------------------------

    #[test]
    fn test_truncate_then_replay() {
        let mut log = make_replay(100);
        put(&mut log, "a", b"1"); // seq 1
        put(&mut log, "b", b"2"); // seq 2
        put(&mut log, "c", b"3"); // seq 3
        log.truncate_before(3);
        let (state, stats) = log.replay(ReplayPolicy::Full);
        assert!(!state.store.contains_key("a"));
        assert!(!state.store.contains_key("b"));
        assert!(state.store.contains_key("c"));
        assert_eq!(stats.entries_applied, 1);
    }

    // -----------------------------------------------------------------------
    // 30. ReplayState sequence tracks last applied entry
    // -----------------------------------------------------------------------

    #[test]
    fn test_replay_state_sequence_final() {
        let mut log = make_replay(100);
        put(&mut log, "a", b"1");
        put(&mut log, "b", b"2");
        put(&mut log, "c", b"3");
        let (state, stats) = log.replay(ReplayPolicy::Full);
        assert_eq!(state.sequence, 3);
        assert_eq!(stats.final_sequence, 3);
    }

    // -----------------------------------------------------------------------
    // 31. Checkpoint entry type round-trips through append
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_entry_is_valid() {
        let mut log = make_replay(100);
        let e = checkpoint(&mut log);
        assert_eq!(e.entry_type, WalEntryType::Checkpoint);
        assert!(e.is_valid());
    }

    // -----------------------------------------------------------------------
    // 32. Multiple Puts to same key, only latest survives
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_puts_last_wins() {
        let mut log = make_replay(100);
        for i in 0u8..10 {
            let val = vec![i];
            put(&mut log, "k", &val);
        }
        let (state, _) = log.replay(ReplayPolicy::Full);
        assert_eq!(state.store["k"], vec![9u8]);
    }

    // -----------------------------------------------------------------------
    // 33. Transaction with multiple Puts and Deletes
    // -----------------------------------------------------------------------

    #[test]
    fn test_transaction_multiple_ops_committed() {
        let mut log = make_replay(100);
        // Pre-populate some keys outside tx.
        put(&mut log, "keep", b"k");
        put(&mut log, "remove", b"r");
        begin(&mut log, 1);
        tx_put(&mut log, "new_key", b"n", 1);
        tx_delete(&mut log, "remove", 1);
        commit(&mut log, 1);
        let (state, _) = log.replay(ReplayPolicy::Full);
        assert!(state.store.contains_key("keep"));
        assert!(!state.store.contains_key("remove"));
        assert!(state.store.contains_key("new_key"));
    }

    // -----------------------------------------------------------------------
    // 34. WalStats with max_entries respected
    // -----------------------------------------------------------------------

    #[test]
    fn test_wal_stats_after_eviction() {
        let mut log = make_replay(5);
        for i in 0u8..10 {
            put(&mut log, "k", &[i]);
        }
        let s = log.wal_stats();
        assert_eq!(s.total_entries, 5);
        // After eviction, oldest seq should be 6.
        assert_eq!(s.oldest_sequence, 6);
        assert_eq!(s.newest_sequence, 10);
    }

    // -----------------------------------------------------------------------
    // 35. compute_checksum is deterministic
    // -----------------------------------------------------------------------

    #[test]
    fn test_checksum_deterministic() {
        let v = Some(b"data".to_vec());
        let c1 = WalEntry::compute_checksum(42, "mykey", &v);
        let c2 = WalEntry::compute_checksum(42, "mykey", &v);
        assert_eq!(c1, c2);
    }

    // -----------------------------------------------------------------------
    // 36. Delete non-existent key is a no-op
    // -----------------------------------------------------------------------

    #[test]
    fn test_delete_nonexistent_key_noop() {
        let mut log = make_replay(100);
        delete(&mut log, "ghost");
        let (state, stats) = log.replay(ReplayPolicy::Full);
        assert!(state.store.is_empty());
        assert_eq!(stats.entries_applied, 1);
    }

    // -----------------------------------------------------------------------
    // 37. SinceSequence beyond newest replays nothing
    // -----------------------------------------------------------------------

    #[test]
    fn test_since_sequence_beyond_newest() {
        let mut log = make_replay(100);
        put(&mut log, "a", b"1");
        put(&mut log, "b", b"2");
        let (state, stats) = log.replay(ReplayPolicy::SinceSequence(999));
        assert!(state.store.is_empty());
        assert_eq!(stats.entries_applied, 0);
        assert_eq!(stats.entries_skipped, 2);
    }
}
