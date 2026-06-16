//! Write-Ahead Log (WAL) for crash-recovery of block store writes.
//!
//! Operations are written to a sequential log before being applied,
//! so interrupted writes can be replayed on restart.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// FNV-1a 32-bit hash
// ---------------------------------------------------------------------------

/// Computes FNV-1a 32-bit hash of `data`.
pub fn fnv1a_32(data: &[u8]) -> u32 {
    const OFFSET_BASIS: u32 = 2_166_136_261;
    const PRIME: u32 = 16_777_619;

    let mut hash = OFFSET_BASIS;
    for &byte in data {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during WAL operations.
#[derive(Debug, Error)]
pub enum WalError {
    /// The WAL has reached its maximum entry capacity.
    #[error("WAL capacity exceeded: current={current}, max={max}")]
    CapacityExceeded { current: usize, max: usize },

    /// A checkpoint operation failed.
    #[error("Checkpoint failed: {0}")]
    CheckpointFailed(String),

    /// Serialization or deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),
}

// ---------------------------------------------------------------------------
// WalOp
// ---------------------------------------------------------------------------

/// A WAL operation that can be recorded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WalOp {
    /// Put a single block by CID.
    Put {
        /// Content identifier of the block.
        cid: String,
        /// Length of the data in bytes.
        data_len: u64,
    },
    /// Delete a single block by CID.
    Delete {
        /// Content identifier of the block to delete.
        cid: String,
    },
    /// Put a batch of blocks.
    BatchPut {
        /// Content identifiers of the blocks.
        cids: Vec<String>,
        /// Total bytes in the batch.
        total_bytes: u64,
    },
    /// A checkpoint marking a stable recovery point.
    Checkpoint {
        /// Sequence number up to which all entries are checkpointed.
        sequence: u64,
    },
}

impl WalOp {
    /// Serializes the op to bytes for CRC computation.
    fn to_bytes_for_crc(&self) -> Result<Vec<u8>, WalError> {
        serde_json::to_vec(self).map_err(|e| WalError::Serialization(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// WalEntry
// ---------------------------------------------------------------------------

/// A single entry in the write-ahead log.
#[derive(Debug, Clone)]
pub struct WalEntry {
    /// Monotonically increasing sequence number.
    pub sequence: u64,
    /// The operation recorded.
    pub op: WalOp,
    /// CRC-32 (FNV-1a) of the serialized op bytes.
    pub crc32: u32,
    /// Unix timestamp in milliseconds when the entry was created.
    pub timestamp_ms: u64,
}

// ---------------------------------------------------------------------------
// WalStats
// ---------------------------------------------------------------------------

/// Atomic statistics for a [`WriteAheadLog`].
#[derive(Debug, Default)]
pub struct WalStats {
    /// Total number of entries ever appended.
    pub total_appended: AtomicU64,
    /// Total number of checkpoints ever taken.
    pub total_checkpoints: AtomicU64,
    /// Total number of entries ever truncated.
    pub total_truncated: AtomicU64,
    /// Total number of entries ever returned by replay.
    pub total_replayed: AtomicU64,
}

/// A point-in-time snapshot of [`WalStats`].
#[derive(Debug, Clone, PartialEq)]
pub struct WalStatsSnapshot {
    pub total_appended: u64,
    pub total_checkpoints: u64,
    pub total_truncated: u64,
    pub total_replayed: u64,
}

impl WalStats {
    /// Returns a consistent point-in-time snapshot of the stats.
    pub fn snapshot(&self) -> WalStatsSnapshot {
        WalStatsSnapshot {
            total_appended: self.total_appended.load(Ordering::Relaxed),
            total_checkpoints: self.total_checkpoints.load(Ordering::Relaxed),
            total_truncated: self.total_truncated.load(Ordering::Relaxed),
            total_replayed: self.total_replayed.load(Ordering::Relaxed),
        }
    }
}

// ---------------------------------------------------------------------------
// WriteAheadLog
// ---------------------------------------------------------------------------

/// In-memory write-ahead log providing crash-recovery for block store writes.
///
/// # Example
/// ```
/// use ipfrs_storage::wal::{WriteAheadLog, WalOp};
///
/// let wal = WriteAheadLog::new(1000);
/// let seq = wal.append(WalOp::Put { cid: "bafytest".into(), data_len: 128 }).unwrap();
/// assert_eq!(seq, 1);
/// let ops = wal.replay_ops();
/// assert_eq!(ops.len(), 1);
/// ```
pub struct WriteAheadLog {
    entries: Mutex<Vec<WalEntry>>,
    next_sequence: AtomicU64,
    checkpoint_sequence: AtomicU64,
    /// Maximum number of entries allowed in the log before truncation is required.
    pub max_entries: usize,
    /// Live statistics.
    pub stats: WalStats,
}

impl WriteAheadLog {
    /// Creates a new [`WriteAheadLog`] with the given maximum entry count.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
            next_sequence: AtomicU64::new(1),
            checkpoint_sequence: AtomicU64::new(0),
            max_entries,
            stats: WalStats::default(),
        }
    }

    /// Creates a [`WriteAheadLog`] with the default maximum of 10 000 entries.
    pub fn with_defaults() -> Self {
        Self::new(10_000)
    }

    /// Returns the current timestamp in milliseconds since the Unix epoch.
    fn now_ms() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Appends an operation to the WAL.
    ///
    /// Assigns the next sequence number, computes the CRC-32 (FNV-1a) of the
    /// serialized op, and stores the entry.  Returns the assigned sequence number.
    ///
    /// # Errors
    /// Returns [`WalError::CapacityExceeded`] when the number of stored entries
    /// already equals or exceeds `max_entries`.
    pub fn append(&self, op: WalOp) -> Result<u64, WalError> {
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| WalError::CheckpointFailed("lock poisoned during append".into()))?;

        let current = entries.len();
        if current >= self.max_entries {
            return Err(WalError::CapacityExceeded {
                current,
                max: self.max_entries,
            });
        }

        let crc32 = {
            let bytes = op.to_bytes_for_crc()?;
            fnv1a_32(&bytes)
        };

        let sequence = self.next_sequence.fetch_add(1, Ordering::SeqCst);

        entries.push(WalEntry {
            sequence,
            op,
            crc32,
            timestamp_ms: Self::now_ms(),
        });

        self.stats.total_appended.fetch_add(1, Ordering::Relaxed);
        Ok(sequence)
    }

    /// Appends a `WalOp::Checkpoint` entry and updates `checkpoint_sequence`.
    ///
    /// Returns the sequence number of the checkpoint entry.
    pub fn checkpoint(&self) -> Result<u64, WalError> {
        // Determine the sequence the checkpoint covers — it will be the sequence
        // of the checkpoint entry itself (the log is up-to-date at that point).
        let checkpoint_seq = self.next_sequence.load(Ordering::SeqCst);

        let seq = self.append(WalOp::Checkpoint {
            sequence: checkpoint_seq,
        })?;

        self.checkpoint_sequence.store(seq, Ordering::SeqCst);
        self.stats.total_checkpoints.fetch_add(1, Ordering::Relaxed);
        Ok(seq)
    }

    /// Returns all entries whose sequence number is strictly greater than the
    /// last checkpointed sequence.
    pub fn entries_since_checkpoint(&self) -> Vec<WalEntry> {
        let checkpoint = self.checkpoint_sequence.load(Ordering::SeqCst);
        let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        entries
            .iter()
            .filter(|e| e.sequence > checkpoint)
            .cloned()
            .collect()
    }

    /// Removes all entries whose sequence number is strictly less than
    /// `sequence`.  Returns the number of entries removed.
    pub fn truncate_before(&self, sequence: u64) -> usize {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let before = entries.len();
        entries.retain(|e| e.sequence >= sequence);
        let removed = before - entries.len();
        self.stats
            .total_truncated
            .fetch_add(removed as u64, Ordering::Relaxed);
        removed
    }

    /// Returns the number of entries currently in the log.
    pub fn entry_count(&self) -> usize {
        let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        entries.len()
    }

    /// Returns the ops from entries since the last checkpoint, excluding
    /// `WalOp::Checkpoint` ops themselves (those are metadata, not data ops).
    pub fn replay_ops(&self) -> Vec<WalOp> {
        let ops: Vec<WalOp> = self
            .entries_since_checkpoint()
            .into_iter()
            .filter_map(|e| match e.op {
                WalOp::Checkpoint { .. } => None,
                other => Some(other),
            })
            .collect();

        self.stats
            .total_replayed
            .fetch_add(ops.len() as u64, Ordering::Relaxed);
        ops
    }

    /// Returns the current checkpoint sequence number.
    pub fn checkpoint_sequence(&self) -> u64 {
        self.checkpoint_sequence.load(Ordering::SeqCst)
    }

    /// Returns the next sequence number that will be assigned (without
    /// consuming it).
    pub fn next_sequence(&self) -> u64 {
        self.next_sequence.load(Ordering::SeqCst)
    }
}

impl std::fmt::Debug for WriteAheadLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WriteAheadLog")
            .field("max_entries", &self.max_entries)
            .field("next_sequence", &self.next_sequence.load(Ordering::Relaxed))
            .field(
                "checkpoint_sequence",
                &self.checkpoint_sequence.load(Ordering::Relaxed),
            )
            .field("entry_count", &self.entry_count())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// StorageWriteAheadLog — in-memory WAL for crash-safe block operations
// ---------------------------------------------------------------------------

/// The kind of operation recorded in a [`StorageWalEntry`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalEntryKind {
    /// A block put operation.
    Put,
    /// A block delete operation.
    Delete,
    /// A block update (overwrite) operation.
    Update,
}

/// A single entry in the [`StorageWriteAheadLog`].
#[derive(Debug, Clone)]
pub struct StorageWalEntry {
    /// Monotonically increasing sequence number.
    pub sequence: u64,
    /// The kind of operation.
    pub kind: WalEntryKind,
    /// The key (e.g. CID) this operation targets.
    pub key: String,
    /// The data payload (empty for deletes).
    pub data: Vec<u8>,
    /// Logical tick at the time the entry was appended.
    pub timestamp_tick: u64,
}

/// Configuration for [`StorageWriteAheadLog`].
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Maximum number of entries before the oldest must be truncated.
    pub max_entries: usize,
    /// Maximum total data bytes across all entries.
    pub max_data_bytes: u64,
    /// Number of ticks between automatic checkpoint hints.
    pub checkpoint_interval: u64,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            max_data_bytes: 100_000_000, // 100 MB
            checkpoint_interval: 50,
        }
    }
}

/// Point-in-time statistics for a [`StorageWriteAheadLog`].
#[derive(Debug, Clone)]
pub struct StorageWalStats {
    /// Number of entries currently in the log.
    pub entry_count: usize,
    /// Total data bytes across all entries.
    pub data_bytes: u64,
    /// Number of checkpoints recorded.
    pub checkpoint_count: usize,
    /// The next sequence number that will be assigned.
    pub next_sequence: u64,
    /// Sequence number of the oldest entry, if any.
    pub oldest_sequence: Option<u64>,
}

/// In-memory write-ahead log for crash-safe block operations.
///
/// Unlike [`WriteAheadLog`] which is CID-centric and thread-safe via
/// `Mutex`/`AtomicU64`, `StorageWriteAheadLog` is a single-threaded,
/// tick-driven WAL that tracks total data bytes and supports indexed
/// checkpoint replay.
pub struct StorageWriteAheadLog {
    config: WalConfig,
    entries: VecDeque<StorageWalEntry>,
    next_sequence: u64,
    current_tick: u64,
    total_data_bytes: u64,
    checkpoints: Vec<u64>,
    last_checkpoint_tick: u64,
}

impl StorageWriteAheadLog {
    /// Creates a new `StorageWriteAheadLog` with the given configuration.
    pub fn new(config: WalConfig) -> Self {
        Self {
            config,
            entries: VecDeque::new(),
            next_sequence: 1,
            current_tick: 0,
            total_data_bytes: 0,
            checkpoints: Vec::new(),
            last_checkpoint_tick: 0,
        }
    }

    /// Appends a new entry to the WAL.
    ///
    /// If appending would exceed `max_entries`, the oldest entry is evicted
    /// first. Returns the assigned sequence number.
    ///
    /// # Errors
    /// Returns an error if adding `data` would exceed `max_data_bytes`.
    pub fn append(&mut self, kind: WalEntryKind, key: &str, data: Vec<u8>) -> Result<u64, String> {
        let new_data_len = data.len() as u64;

        // Check data-byte budget (after potential eviction the freed bytes
        // might make room, so we calculate the *would-be* total).
        let would_be = self.total_data_bytes + new_data_len;
        if would_be > self.config.max_data_bytes {
            return Err(format!(
                "WAL data limit exceeded: would be {} bytes, max {} bytes",
                would_be, self.config.max_data_bytes
            ));
        }

        // Evict oldest if at capacity
        if self.entries.len() >= self.config.max_entries {
            if let Some(evicted) = self.entries.pop_front() {
                self.total_data_bytes = self
                    .total_data_bytes
                    .saturating_sub(evicted.data.len() as u64);
            }
        }

        let seq = self.next_sequence;
        self.next_sequence += 1;

        self.entries.push_back(StorageWalEntry {
            sequence: seq,
            kind,
            key: key.to_string(),
            data,
            timestamp_tick: self.current_tick,
        });
        self.total_data_bytes += new_data_len;

        Ok(seq)
    }

    /// Returns a reference to the entry with the given sequence number, if present.
    pub fn get_entry(&self, sequence: u64) -> Option<&StorageWalEntry> {
        // The deque is ordered by sequence. We can binary-search if the front
        // sequence is known.
        let front_seq = self.entries.front().map(|e| e.sequence)?;
        if sequence < front_seq {
            return None;
        }
        let idx = (sequence - front_seq) as usize;
        self.entries.get(idx).filter(|e| e.sequence == sequence)
    }

    /// Returns references to all entries with sequence numbers strictly
    /// greater than `sequence`.
    pub fn entries_since(&self, sequence: u64) -> Vec<&StorageWalEntry> {
        self.entries
            .iter()
            .filter(|e| e.sequence > sequence)
            .collect()
    }

    /// Records the current head sequence as a checkpoint and returns it.
    pub fn create_checkpoint(&mut self) -> u64 {
        let cp_seq = if let Some(last) = self.entries.back() {
            last.sequence
        } else {
            self.next_sequence.saturating_sub(1)
        };
        self.checkpoints.push(cp_seq);
        self.last_checkpoint_tick = self.current_tick;
        cp_seq
    }

    /// Removes all entries with sequence numbers strictly less than `sequence`,
    /// updating `total_data_bytes` accordingly.
    pub fn truncate_before(&mut self, sequence: u64) {
        while let Some(front) = self.entries.front() {
            if front.sequence < sequence {
                let evicted = self
                    .entries
                    .pop_front()
                    .expect("front existed in condition");
                self.total_data_bytes = self
                    .total_data_bytes
                    .saturating_sub(evicted.data.len() as u64);
            } else {
                break;
            }
        }
    }

    /// Returns all entries from the given checkpoint index to the end.
    ///
    /// # Errors
    /// Returns an error if `checkpoint_idx` is out of range.
    pub fn replay_from_checkpoint(
        &self,
        checkpoint_idx: usize,
    ) -> Result<Vec<&StorageWalEntry>, String> {
        let cp_seq = self.checkpoints.get(checkpoint_idx).ok_or_else(|| {
            format!(
                "checkpoint index {} out of range (have {})",
                checkpoint_idx,
                self.checkpoints.len()
            )
        })?;
        Ok(self.entries_since(*cp_seq))
    }

    /// Returns `true` if the number of ticks since the last checkpoint is
    /// at least `checkpoint_interval`.
    pub fn should_checkpoint(&self) -> bool {
        self.current_tick.saturating_sub(self.last_checkpoint_tick)
            >= self.config.checkpoint_interval
    }

    /// Advances the logical clock by one tick.
    pub fn tick(&mut self) {
        self.current_tick += 1;
    }

    /// Returns the number of entries currently in the log.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Returns the total data bytes across all entries.
    pub fn data_bytes(&self) -> u64 {
        self.total_data_bytes
    }

    /// Returns a point-in-time statistics snapshot.
    pub fn stats(&self) -> StorageWalStats {
        StorageWalStats {
            entry_count: self.entries.len(),
            data_bytes: self.total_data_bytes,
            checkpoint_count: self.checkpoints.len(),
            next_sequence: self.next_sequence,
            oldest_sequence: self.entries.front().map(|e| e.sequence),
        }
    }
}

impl std::fmt::Debug for StorageWriteAheadLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageWriteAheadLog")
            .field("next_sequence", &self.next_sequence)
            .field("current_tick", &self.current_tick)
            .field("entry_count", &self.entries.len())
            .field("total_data_bytes", &self.total_data_bytes)
            .field("checkpoint_count", &self.checkpoints.len())
            .finish()
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

    fn make_put(cid: &str, data_len: u64) -> WalOp {
        WalOp::Put {
            cid: cid.to_string(),
            data_len,
        }
    }

    fn make_delete(cid: &str) -> WalOp {
        WalOp::Delete {
            cid: cid.to_string(),
        }
    }

    fn make_batch_put(cids: &[&str], total_bytes: u64) -> WalOp {
        WalOp::BatchPut {
            cids: cids.iter().map(|s| s.to_string()).collect(),
            total_bytes,
        }
    }

    // ------------------------------------------------------------------
    // 1. Append single entry, sequence increments
    // ------------------------------------------------------------------

    #[test]
    fn test_append_single_entry_sequence_starts_at_one() {
        let wal = WriteAheadLog::new(100);
        let seq = wal.append(make_put("bafya", 64)).expect("append failed");
        assert_eq!(seq, 1, "first sequence should be 1");
        assert_eq!(wal.entry_count(), 1);
    }

    #[test]
    fn test_append_multiple_entries_sequence_increments() {
        let wal = WriteAheadLog::new(100);
        let s1 = wal.append(make_put("cid1", 10)).expect("append 1");
        let s2 = wal.append(make_delete("cid1")).expect("append 2");
        let s3 = wal.append(make_put("cid2", 20)).expect("append 3");

        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
        assert_eq!(s3, 3);
        assert_eq!(wal.entry_count(), 3);
    }

    // ------------------------------------------------------------------
    // 2. Append exceeds max_entries → error
    // ------------------------------------------------------------------

    #[test]
    fn test_append_exceeds_max_entries_returns_error() {
        let wal = WriteAheadLog::new(2);
        wal.append(make_put("c1", 1)).expect("first append");
        wal.append(make_put("c2", 2)).expect("second append");

        let result = wal.append(make_put("c3", 3));
        match result {
            Err(WalError::CapacityExceeded { current, max }) => {
                assert_eq!(current, 2);
                assert_eq!(max, 2);
            }
            other => panic!("expected CapacityExceeded, got {:?}", other),
        }
    }

    // ------------------------------------------------------------------
    // 3. Checkpoint records sequence
    // ------------------------------------------------------------------

    #[test]
    fn test_checkpoint_records_sequence() {
        let wal = WriteAheadLog::new(100);
        wal.append(make_put("x", 1)).expect("append");
        wal.append(make_put("y", 2)).expect("append");

        let cp_seq = wal.checkpoint().expect("checkpoint");
        assert!(cp_seq >= 3, "checkpoint entry should get next sequence (3)");
        assert_eq!(wal.checkpoint_sequence(), cp_seq);
    }

    // ------------------------------------------------------------------
    // 4. entries_since_checkpoint returns only new entries
    // ------------------------------------------------------------------

    #[test]
    fn test_entries_since_checkpoint_returns_only_new() {
        let wal = WriteAheadLog::new(100);
        wal.append(make_put("before", 1)).expect("append");
        wal.checkpoint().expect("checkpoint");

        wal.append(make_put("after1", 2)).expect("append after");
        wal.append(make_put("after2", 3)).expect("append after");

        let since = wal.entries_since_checkpoint();
        assert_eq!(since.len(), 2);
        for entry in &since {
            match &entry.op {
                WalOp::Put { cid, .. } => {
                    assert!(cid.starts_with("after"), "unexpected CID: {}", cid);
                }
                other => panic!("unexpected op: {:?}", other),
            }
        }
    }

    // ------------------------------------------------------------------
    // 5. After checkpoint, entries_since_checkpoint is empty (excl. Checkpoint)
    // ------------------------------------------------------------------

    #[test]
    fn test_entries_since_checkpoint_empty_after_fresh_checkpoint() {
        let wal = WriteAheadLog::new(100);
        wal.append(make_put("a", 1)).expect("a");
        wal.append(make_put("b", 2)).expect("b");
        wal.checkpoint().expect("checkpoint");

        // entries_since_checkpoint includes the Checkpoint entry itself only
        // if its sequence > checkpoint_sequence — but checkpoint_sequence IS set
        // to the checkpoint entry's sequence, so nothing should appear.
        let since = wal.entries_since_checkpoint();
        assert!(
            since.is_empty(),
            "expected empty, got {} entries",
            since.len()
        );
    }

    // ------------------------------------------------------------------
    // 6. truncate_before removes correct entries
    // ------------------------------------------------------------------

    #[test]
    fn test_truncate_before_removes_correct_entries() {
        let wal = WriteAheadLog::new(100);
        for i in 0..5u64 {
            wal.append(make_put(&format!("c{i}"), i)).expect("append");
        }
        // Sequences 1..=5 in log. Remove < 3 (i.e. 1 and 2).
        let removed = wal.truncate_before(3);
        assert_eq!(removed, 2, "should remove exactly 2 entries");
        assert_eq!(wal.entry_count(), 3, "3 entries should remain");
    }

    #[test]
    fn test_truncate_before_nothing_to_remove() {
        let wal = WriteAheadLog::new(100);
        wal.append(make_put("a", 1)).expect("append");
        let removed = wal.truncate_before(1); // seq >= 1 kept, nothing < 1
        assert_eq!(removed, 0);
        assert_eq!(wal.entry_count(), 1);
    }

    // ------------------------------------------------------------------
    // 7. replay_ops excludes Checkpoint ops
    // ------------------------------------------------------------------

    #[test]
    fn test_replay_ops_excludes_checkpoint_ops() {
        let wal = WriteAheadLog::new(100);
        // Before checkpoint
        wal.append(make_put("old", 1)).expect("append old");
        wal.checkpoint().expect("cp1");

        // After checkpoint: 2 data ops + another checkpoint
        wal.append(make_put("new1", 2)).expect("new1");
        wal.append(make_delete("new1")).expect("del");
        wal.checkpoint().expect("cp2");

        // replay_ops returns only data ops since last checkpoint
        let ops = wal.replay_ops();
        assert!(
            ops.is_empty(),
            "after second checkpoint there are no data ops to replay, got {:?}",
            ops
        );
    }

    #[test]
    fn test_replay_ops_returns_data_ops_since_checkpoint() {
        let wal = WriteAheadLog::new(100);
        wal.append(make_put("a", 1)).expect("a");
        wal.checkpoint().expect("cp");

        wal.append(make_put("b", 2)).expect("b");
        wal.append(make_delete("a")).expect("del a");
        wal.append(make_batch_put(&["c", "d"], 500)).expect("batch");

        let ops = wal.replay_ops();
        assert_eq!(ops.len(), 3);
        assert!(matches!(ops[0], WalOp::Put { ref cid, .. } if cid == "b"));
        assert!(matches!(ops[1], WalOp::Delete { ref cid } if cid == "a"));
        assert!(matches!(ops[2], WalOp::BatchPut { .. }));
    }

    // ------------------------------------------------------------------
    // 8. CRC-32 computed (non-zero for non-empty data)
    // ------------------------------------------------------------------

    #[test]
    fn test_crc32_non_zero_for_non_empty_data() {
        let data = b"hello ipfrs wal";
        let crc = fnv1a_32(data);
        assert_ne!(
            crc, 0,
            "FNV-1a should produce non-zero hash for non-empty input"
        );
    }

    #[test]
    fn test_crc32_entry_stored_correctly() {
        let wal = WriteAheadLog::new(100);
        let op = make_put("bafytest123", 256);
        let seq = wal.append(op.clone()).expect("append");

        let entries = wal
            .entries_since_checkpoint()
            .into_iter()
            .find(|e| e.sequence == seq)
            .expect("entry not found");

        let expected_crc = fnv1a_32(&serde_json::to_vec(&op).unwrap());
        assert_eq!(entries.crc32, expected_crc);
        assert_ne!(entries.crc32, 0);
    }

    // ------------------------------------------------------------------
    // 9. Stats accumulation
    // ------------------------------------------------------------------

    #[test]
    fn test_stats_total_appended() {
        let wal = WriteAheadLog::new(100);
        wal.append(make_put("x", 1)).expect("x");
        wal.append(make_delete("x")).expect("del x");
        let snap = wal.stats.snapshot();
        // The checkpoint() itself calls append(), so here just 2
        assert_eq!(snap.total_appended, 2);
    }

    #[test]
    fn test_stats_total_checkpoints() {
        let wal = WriteAheadLog::new(100);
        wal.append(make_put("a", 1)).expect("a");
        wal.checkpoint().expect("cp1");
        wal.checkpoint().expect("cp2");
        let snap = wal.stats.snapshot();
        assert_eq!(snap.total_checkpoints, 2);
        // checkpoint also calls append internally → 3 appends total
        assert_eq!(snap.total_appended, 3);
    }

    #[test]
    fn test_stats_total_truncated() {
        let wal = WriteAheadLog::new(100);
        for i in 0..6u64 {
            wal.append(make_put(&format!("c{i}"), i)).expect("append");
        }
        wal.truncate_before(4); // removes seq 1,2,3
        let snap = wal.stats.snapshot();
        assert_eq!(snap.total_truncated, 3);
    }

    #[test]
    fn test_stats_total_replayed() {
        let wal = WriteAheadLog::new(100);
        wal.append(make_put("a", 1)).expect("a");
        wal.checkpoint().expect("cp");
        wal.append(make_put("b", 2)).expect("b");
        wal.append(make_delete("a")).expect("del");

        let ops = wal.replay_ops();
        assert_eq!(ops.len(), 2);

        let snap = wal.stats.snapshot();
        assert_eq!(snap.total_replayed, 2);
    }

    // ------------------------------------------------------------------
    // 10. Multiple Put/Delete/BatchPut ops
    // ------------------------------------------------------------------

    #[test]
    fn test_multiple_op_types_roundtrip() {
        let wal = WriteAheadLog::new(100);

        let s1 = wal.append(make_put("cid-put-1", 100)).expect("put1");
        let s2 = wal.append(make_put("cid-put-2", 200)).expect("put2");
        let s3 = wal
            .append(make_batch_put(&["cid-a", "cid-b", "cid-c"], 750))
            .expect("batch");
        let s4 = wal.append(make_delete("cid-put-1")).expect("del");

        assert!(s1 < s2 && s2 < s3 && s3 < s4, "sequences must be monotonic");
        assert_eq!(wal.entry_count(), 4);

        let ops = wal.replay_ops();
        assert_eq!(ops.len(), 4);

        assert!(matches!(&ops[0], WalOp::Put { cid, data_len: 100 } if cid == "cid-put-1"));
        assert!(matches!(&ops[1], WalOp::Put { cid, data_len: 200 } if cid == "cid-put-2"));
        assert!(matches!(&ops[2], WalOp::BatchPut { cids, total_bytes: 750 } if cids.len() == 3));
        assert!(matches!(&ops[3], WalOp::Delete { cid } if cid == "cid-put-1"));
    }

    // ------------------------------------------------------------------
    // 11. FNV-1a determinism and known value
    // ------------------------------------------------------------------

    #[test]
    fn test_fnv1a_deterministic() {
        let data = b"ipfrs-wal";
        let h1 = fnv1a_32(data);
        let h2 = fnv1a_32(data);
        assert_eq!(h1, h2, "FNV-1a must be deterministic");
    }

    #[test]
    fn test_fnv1a_empty_input_is_offset_basis() {
        let h = fnv1a_32(b"");
        assert_eq!(h, 2_166_136_261u32, "empty input should equal offset basis");
    }

    // ------------------------------------------------------------------
    // 12. Concurrent appends
    // ------------------------------------------------------------------

    #[test]
    fn test_concurrent_appends_unique_sequences() {
        use std::sync::Arc;
        use std::thread;

        let wal = Arc::new(WriteAheadLog::new(10_000));
        let threads = 8usize;
        let per_thread = 50usize;

        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let w = Arc::clone(&wal);
                thread::spawn(move || {
                    (0..per_thread)
                        .map(|i| {
                            w.append(make_put(&format!("t{t}-c{i}"), i as u64))
                                .expect("concurrent append")
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();

        let mut all_seqs: Vec<u64> = handles
            .into_iter()
            .flat_map(|h| h.join().expect("thread panicked"))
            .collect();

        all_seqs.sort_unstable();
        all_seqs.dedup();
        assert_eq!(
            all_seqs.len(),
            threads * per_thread,
            "all sequences must be unique"
        );
        assert_eq!(wal.entry_count(), threads * per_thread);
    }

    // ------------------------------------------------------------------
    // 13. Debug impl smoke test
    // ------------------------------------------------------------------

    #[test]
    fn test_debug_impl_does_not_panic() {
        let wal = WriteAheadLog::with_defaults();
        wal.append(make_put("dbg", 1)).expect("append");
        let _ = format!("{:?}", wal);
    }

    // ------------------------------------------------------------------
    // 14. WalError display
    // ------------------------------------------------------------------

    #[test]
    fn test_wal_error_display() {
        let e = WalError::CapacityExceeded { current: 5, max: 5 };
        let msg = e.to_string();
        assert!(
            msg.contains("current=5") && msg.contains("max=5"),
            "{}",
            msg
        );

        let e2 = WalError::CheckpointFailed("disk full".into());
        assert!(e2.to_string().contains("disk full"));
    }

    // ==================================================================
    // StorageWriteAheadLog tests
    // ==================================================================

    fn swal_config() -> WalConfig {
        WalConfig {
            max_entries: 100,
            max_data_bytes: 10_000,
            checkpoint_interval: 5,
        }
    }

    fn swal_default() -> StorageWriteAheadLog {
        StorageWriteAheadLog::new(swal_config())
    }

    // ------------------------------------------------------------------
    // 15. Append entries and verify sequence numbers
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_append_sequence_starts_at_one() {
        let mut wal = swal_default();
        let seq = wal
            .append(WalEntryKind::Put, "key1", vec![1, 2, 3])
            .expect("append");
        assert_eq!(seq, 1);
        assert_eq!(wal.entry_count(), 1);
    }

    #[test]
    fn test_swal_append_multiple_sequences_increment() {
        let mut wal = swal_default();
        let s1 = wal.append(WalEntryKind::Put, "k1", vec![1]).expect("s1");
        let s2 = wal.append(WalEntryKind::Delete, "k1", vec![]).expect("s2");
        let s3 = wal.append(WalEntryKind::Update, "k2", vec![9]).expect("s3");
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
        assert_eq!(s3, 3);
        assert_eq!(wal.entry_count(), 3);
    }

    // ------------------------------------------------------------------
    // 16. get_entry by sequence
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_get_entry_existing() {
        let mut wal = swal_default();
        let seq = wal.append(WalEntryKind::Put, "abc", vec![10]).expect("a");
        let entry = wal.get_entry(seq).expect("entry should exist");
        assert_eq!(entry.key, "abc");
        assert_eq!(entry.data, vec![10]);
        assert_eq!(entry.kind, WalEntryKind::Put);
    }

    #[test]
    fn test_swal_get_entry_missing() {
        let wal = swal_default();
        assert!(wal.get_entry(999).is_none());
    }

    #[test]
    fn test_swal_get_entry_after_truncation() {
        let mut wal = swal_default();
        wal.append(WalEntryKind::Put, "a", vec![1]).expect("a");
        wal.append(WalEntryKind::Put, "b", vec![2]).expect("b");
        wal.truncate_before(2);
        assert!(wal.get_entry(1).is_none());
        assert!(wal.get_entry(2).is_some());
    }

    // ------------------------------------------------------------------
    // 17. entries_since returns correct slice
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_entries_since() {
        let mut wal = swal_default();
        for i in 0..5 {
            wal.append(WalEntryKind::Put, &format!("k{i}"), vec![i as u8])
                .expect("append");
        }
        let since = wal.entries_since(3);
        assert_eq!(since.len(), 2);
        assert_eq!(since[0].sequence, 4);
        assert_eq!(since[1].sequence, 5);
    }

    #[test]
    fn test_swal_entries_since_zero_returns_all() {
        let mut wal = swal_default();
        wal.append(WalEntryKind::Put, "x", vec![]).expect("x");
        wal.append(WalEntryKind::Delete, "y", vec![]).expect("y");
        let all = wal.entries_since(0);
        assert_eq!(all.len(), 2);
    }

    // ------------------------------------------------------------------
    // 18. Checkpoint creation
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_create_checkpoint() {
        let mut wal = swal_default();
        wal.append(WalEntryKind::Put, "a", vec![1]).expect("a");
        wal.append(WalEntryKind::Put, "b", vec![2]).expect("b");
        let cp = wal.create_checkpoint();
        assert_eq!(cp, 2); // last entry sequence
        assert_eq!(wal.stats().checkpoint_count, 1);
    }

    #[test]
    fn test_swal_create_multiple_checkpoints() {
        let mut wal = swal_default();
        wal.append(WalEntryKind::Put, "a", vec![]).expect("a");
        let cp1 = wal.create_checkpoint();
        wal.append(WalEntryKind::Put, "b", vec![]).expect("b");
        wal.append(WalEntryKind::Put, "c", vec![]).expect("c");
        let cp2 = wal.create_checkpoint();
        assert!(cp2 > cp1);
        assert_eq!(wal.stats().checkpoint_count, 2);
    }

    // ------------------------------------------------------------------
    // 19. truncate_before removes old entries and updates data bytes
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_truncate_before() {
        let mut wal = swal_default();
        wal.append(WalEntryKind::Put, "a", vec![1, 2, 3])
            .expect("a");
        wal.append(WalEntryKind::Put, "b", vec![4, 5]).expect("b");
        wal.append(WalEntryKind::Put, "c", vec![6]).expect("c");

        let bytes_before = wal.data_bytes();
        assert_eq!(bytes_before, 6);

        wal.truncate_before(3); // remove seq 1, 2
        assert_eq!(wal.entry_count(), 1);
        assert_eq!(wal.data_bytes(), 1); // only "c" data remains
    }

    #[test]
    fn test_swal_truncate_before_nothing() {
        let mut wal = swal_default();
        wal.append(WalEntryKind::Put, "x", vec![1]).expect("x");
        wal.truncate_before(1); // seq 1 is NOT less than 1
        assert_eq!(wal.entry_count(), 1);
    }

    #[test]
    fn test_swal_truncate_before_all() {
        let mut wal = swal_default();
        wal.append(WalEntryKind::Put, "a", vec![1]).expect("a");
        wal.append(WalEntryKind::Put, "b", vec![2]).expect("b");
        wal.truncate_before(100);
        assert_eq!(wal.entry_count(), 0);
        assert_eq!(wal.data_bytes(), 0);
    }

    // ------------------------------------------------------------------
    // 20. replay_from_checkpoint
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_replay_from_checkpoint() {
        let mut wal = swal_default();
        wal.append(WalEntryKind::Put, "old", vec![1]).expect("old");
        wal.create_checkpoint(); // checkpoint 0 at seq 1
        wal.append(WalEntryKind::Put, "new1", vec![2]).expect("n1");
        wal.append(WalEntryKind::Delete, "old", vec![]).expect("n2");

        let replayed = wal.replay_from_checkpoint(0).expect("replay");
        assert_eq!(replayed.len(), 2);
        assert_eq!(replayed[0].key, "new1");
        assert_eq!(replayed[1].key, "old");
    }

    #[test]
    fn test_swal_replay_from_checkpoint_invalid_index() {
        let wal = swal_default();
        let result = wal.replay_from_checkpoint(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_swal_replay_from_second_checkpoint() {
        let mut wal = swal_default();
        wal.append(WalEntryKind::Put, "a", vec![]).expect("a");
        wal.create_checkpoint(); // cp 0
        wal.append(WalEntryKind::Put, "b", vec![]).expect("b");
        wal.create_checkpoint(); // cp 1
        wal.append(WalEntryKind::Put, "c", vec![]).expect("c");

        let replayed = wal.replay_from_checkpoint(1).expect("replay from cp1");
        assert_eq!(replayed.len(), 1);
        assert_eq!(replayed[0].key, "c");
    }

    // ------------------------------------------------------------------
    // 21. max_entries overflow → oldest evicted
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_max_entries_overflow_evicts_oldest() {
        let cfg = WalConfig {
            max_entries: 3,
            max_data_bytes: 100_000,
            checkpoint_interval: 50,
        };
        let mut wal = StorageWriteAheadLog::new(cfg);
        wal.append(WalEntryKind::Put, "a", vec![1]).expect("a");
        wal.append(WalEntryKind::Put, "b", vec![2]).expect("b");
        wal.append(WalEntryKind::Put, "c", vec![3]).expect("c");
        // Now at capacity, next append evicts "a"
        wal.append(WalEntryKind::Put, "d", vec![4]).expect("d");

        assert_eq!(wal.entry_count(), 3);
        assert!(wal.get_entry(1).is_none(), "seq 1 should be evicted");
        assert!(wal.get_entry(2).is_some());
        assert!(wal.get_entry(4).is_some());
    }

    // ------------------------------------------------------------------
    // 22. max_data_bytes enforcement
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_max_data_bytes_enforced() {
        let cfg = WalConfig {
            max_entries: 100,
            max_data_bytes: 10,
            checkpoint_interval: 50,
        };
        let mut wal = StorageWriteAheadLog::new(cfg);
        wal.append(WalEntryKind::Put, "a", vec![0; 8]).expect("a");
        // 8 bytes used; adding 5 more → 13 > 10 → error
        let result = wal.append(WalEntryKind::Put, "b", vec![0; 5]);
        assert!(result.is_err());
        assert!(result
            .as_ref()
            .err()
            .is_some_and(|e| e.contains("data limit exceeded")));
    }

    #[test]
    fn test_swal_max_data_bytes_exact_limit() {
        let cfg = WalConfig {
            max_entries: 100,
            max_data_bytes: 10,
            checkpoint_interval: 50,
        };
        let mut wal = StorageWriteAheadLog::new(cfg);
        // Exactly 10 bytes should succeed
        wal.append(WalEntryKind::Put, "a", vec![0; 10])
            .expect("exact limit");
        assert_eq!(wal.data_bytes(), 10);
        // One more byte should fail
        let result = wal.append(WalEntryKind::Put, "b", vec![1]);
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // 23. should_checkpoint timing
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_should_checkpoint_timing() {
        let cfg = WalConfig {
            max_entries: 100,
            max_data_bytes: 100_000,
            checkpoint_interval: 3,
        };
        let mut wal = StorageWriteAheadLog::new(cfg);
        // At tick 0, last_checkpoint_tick is 0, interval is 3
        // 0 - 0 = 0 < 3
        assert!(!wal.should_checkpoint());
        wal.tick(); // tick 1
        assert!(!wal.should_checkpoint());
        wal.tick(); // tick 2
        assert!(!wal.should_checkpoint());
        wal.tick(); // tick 3
        assert!(wal.should_checkpoint());

        wal.create_checkpoint(); // resets last_checkpoint_tick to 3
        assert!(!wal.should_checkpoint());
        wal.tick(); // 4
        wal.tick(); // 5
        wal.tick(); // 6
        assert!(wal.should_checkpoint());
    }

    // ------------------------------------------------------------------
    // 24. stats accuracy
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_stats_accuracy() {
        let mut wal = swal_default();
        wal.append(WalEntryKind::Put, "a", vec![1, 2]).expect("a");
        wal.append(WalEntryKind::Delete, "b", vec![]).expect("b");
        wal.create_checkpoint();

        let s = wal.stats();
        assert_eq!(s.entry_count, 2);
        assert_eq!(s.data_bytes, 2);
        assert_eq!(s.checkpoint_count, 1);
        assert_eq!(s.next_sequence, 3);
        assert_eq!(s.oldest_sequence, Some(1));
    }

    #[test]
    fn test_swal_stats_after_truncate() {
        let mut wal = swal_default();
        wal.append(WalEntryKind::Put, "a", vec![1]).expect("a");
        wal.append(WalEntryKind::Put, "b", vec![2, 3]).expect("b");
        wal.truncate_before(2);
        let s = wal.stats();
        assert_eq!(s.entry_count, 1);
        assert_eq!(s.data_bytes, 2);
        assert_eq!(s.oldest_sequence, Some(2));
    }

    // ------------------------------------------------------------------
    // 25. Empty WAL behavior
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_empty_wal() {
        let wal = swal_default();
        assert_eq!(wal.entry_count(), 0);
        assert_eq!(wal.data_bytes(), 0);
        assert!(wal.get_entry(1).is_none());
        assert!(wal.entries_since(0).is_empty());
        let s = wal.stats();
        assert_eq!(s.entry_count, 0);
        assert_eq!(s.oldest_sequence, None);
        assert_eq!(s.checkpoint_count, 0);
    }

    // ------------------------------------------------------------------
    // 26. Mixed Put/Delete/Update entries
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_mixed_entry_kinds() {
        let mut wal = swal_default();
        wal.append(WalEntryKind::Put, "file1", vec![10, 20])
            .expect("put");
        wal.append(WalEntryKind::Update, "file1", vec![30, 40])
            .expect("update");
        wal.append(WalEntryKind::Delete, "file1", vec![])
            .expect("delete");

        assert_eq!(wal.entry_count(), 3);
        let e1 = wal.get_entry(1).expect("e1");
        assert_eq!(e1.kind, WalEntryKind::Put);
        let e2 = wal.get_entry(2).expect("e2");
        assert_eq!(e2.kind, WalEntryKind::Update);
        let e3 = wal.get_entry(3).expect("e3");
        assert_eq!(e3.kind, WalEntryKind::Delete);
    }

    // ------------------------------------------------------------------
    // 27. Tick advances clock
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_tick_advances_clock() {
        let mut wal = swal_default();
        wal.tick();
        wal.tick();
        wal.append(WalEntryKind::Put, "t", vec![]).expect("t");
        let entry = wal.get_entry(1).expect("entry");
        assert_eq!(entry.timestamp_tick, 2);
    }

    // ------------------------------------------------------------------
    // 28. data_bytes tracks correctly through append + truncate
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_data_bytes_tracking() {
        let mut wal = swal_default();
        wal.append(WalEntryKind::Put, "a", vec![0; 100]).expect("a");
        assert_eq!(wal.data_bytes(), 100);
        wal.append(WalEntryKind::Put, "b", vec![0; 50]).expect("b");
        assert_eq!(wal.data_bytes(), 150);
        wal.truncate_before(2);
        assert_eq!(wal.data_bytes(), 50);
    }

    // ------------------------------------------------------------------
    // 29. Overflow eviction updates data_bytes
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_overflow_eviction_updates_data_bytes() {
        let cfg = WalConfig {
            max_entries: 2,
            max_data_bytes: 100_000,
            checkpoint_interval: 50,
        };
        let mut wal = StorageWriteAheadLog::new(cfg);
        wal.append(WalEntryKind::Put, "a", vec![0; 30]).expect("a");
        wal.append(WalEntryKind::Put, "b", vec![0; 20]).expect("b");
        assert_eq!(wal.data_bytes(), 50);

        // Evicts "a" (30 bytes), adds "c" (10 bytes)
        wal.append(WalEntryKind::Put, "c", vec![0; 10]).expect("c");
        assert_eq!(wal.data_bytes(), 30); // 20 + 10
        assert_eq!(wal.entry_count(), 2);
    }

    // ------------------------------------------------------------------
    // 30. Checkpoint on empty WAL
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_checkpoint_on_empty() {
        let mut wal = swal_default();
        let cp = wal.create_checkpoint();
        assert_eq!(cp, 0); // next_sequence(1) - 1 = 0
        assert_eq!(wal.stats().checkpoint_count, 1);
    }

    // ------------------------------------------------------------------
    // 31. entries_since with no matches
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_entries_since_none() {
        let mut wal = swal_default();
        wal.append(WalEntryKind::Put, "a", vec![]).expect("a");
        let since = wal.entries_since(100);
        assert!(since.is_empty());
    }

    // ------------------------------------------------------------------
    // 32. Default config values
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_default_config() {
        let cfg = WalConfig::default();
        assert_eq!(cfg.max_entries, 10_000);
        assert_eq!(cfg.max_data_bytes, 100_000_000);
        assert_eq!(cfg.checkpoint_interval, 50);
    }

    // ------------------------------------------------------------------
    // 33. Debug impl for StorageWriteAheadLog
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_debug_does_not_panic() {
        let mut wal = swal_default();
        wal.append(WalEntryKind::Put, "dbg", vec![1]).expect("a");
        let dbg = format!("{:?}", wal);
        assert!(dbg.contains("StorageWriteAheadLog"));
    }

    // ------------------------------------------------------------------
    // 34. WalEntryKind equality
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_entry_kind_equality() {
        assert_eq!(WalEntryKind::Put, WalEntryKind::Put);
        assert_ne!(WalEntryKind::Put, WalEntryKind::Delete);
        assert_ne!(WalEntryKind::Delete, WalEntryKind::Update);
    }

    // ------------------------------------------------------------------
    // 35. StorageWalEntry clone
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_entry_clone() {
        let entry = StorageWalEntry {
            sequence: 42,
            kind: WalEntryKind::Update,
            key: "cloned".to_string(),
            data: vec![7, 8, 9],
            timestamp_tick: 10,
        };
        let cloned = entry.clone();
        assert_eq!(cloned.sequence, 42);
        assert_eq!(cloned.kind, WalEntryKind::Update);
        assert_eq!(cloned.key, "cloned");
        assert_eq!(cloned.data, vec![7, 8, 9]);
    }

    // ------------------------------------------------------------------
    // 36. StorageWalStats clone and debug
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_stats_clone_and_debug() {
        let mut wal = swal_default();
        wal.append(WalEntryKind::Put, "s", vec![1, 2, 3])
            .expect("s");
        let stats = wal.stats();
        let cloned = stats.clone();
        assert_eq!(cloned.entry_count, 1);
        let dbg = format!("{:?}", cloned);
        assert!(dbg.contains("entry_count"));
    }

    // ------------------------------------------------------------------
    // 37. Large sequential append + replay
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_large_sequential_append_and_replay() {
        let cfg = WalConfig {
            max_entries: 500,
            max_data_bytes: 100_000,
            checkpoint_interval: 100,
        };
        let mut wal = StorageWriteAheadLog::new(cfg);

        for i in 0..200 {
            wal.append(WalEntryKind::Put, &format!("key{i}"), vec![i as u8])
                .expect("append");
        }
        assert_eq!(wal.entry_count(), 200);

        wal.create_checkpoint(); // at seq 200
        for i in 200..250 {
            wal.append(WalEntryKind::Put, &format!("key{i}"), vec![i as u8])
                .expect("append");
        }

        let replayed = wal.replay_from_checkpoint(0).expect("replay");
        assert_eq!(replayed.len(), 50);
    }

    // ------------------------------------------------------------------
    // 38. should_checkpoint at tick 0 with interval 0
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_should_checkpoint_interval_zero() {
        let cfg = WalConfig {
            max_entries: 100,
            max_data_bytes: 100_000,
            checkpoint_interval: 0,
        };
        let wal = StorageWriteAheadLog::new(cfg);
        // 0 - 0 >= 0 → true
        assert!(wal.should_checkpoint());
    }

    // ------------------------------------------------------------------
    // 39. Delete entries have zero data bytes
    // ------------------------------------------------------------------

    #[test]
    fn test_swal_delete_entries_zero_data() {
        let mut wal = swal_default();
        wal.append(WalEntryKind::Delete, "gone", vec![])
            .expect("del");
        assert_eq!(wal.data_bytes(), 0);
        assert_eq!(wal.entry_count(), 1);
    }
}
