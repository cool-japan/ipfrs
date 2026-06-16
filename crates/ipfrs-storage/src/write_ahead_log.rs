//! Production-grade Write-Ahead Log (WAL) for crash recovery and durability.
//!
//! Entries are encoded in a binary format and stored in an in-memory `Vec<u8>`
//! segment buffer. The encoding is self-describing and checksum-protected so
//! that any corruption is detected deterministically during recovery.
//!
//! # Entry layout
//!
//! ```text
//! [magic(4)] [seq_num(8)] [tx_id(8)] [op_type(1)] [key_len(4)] [key(n)]
//! [value_len(4)] [value(m)] [timestamp(8)] [checksum(8)]
//! ```
//!
//! The checksum is the FNV-1a 64-bit hash of all preceding bytes in the entry.

use std::collections::HashMap;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// FNV-1a 64-bit
// ─────────────────────────────────────────────────────────────────────────────

/// Computes the FNV-1a 64-bit hash of `data`.
#[inline]
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

/// WAL entry magic bytes — detect non-WAL or truncated data immediately.
pub const WAL_MAGIC: [u8; 4] = *b"WALX";

/// Minimum byte size of an encoded WAL entry (no key, no value).
/// magic(4) + seq(8) + tx_id(8) + op_type(1) + key_len(4) + value_len(4)
/// + timestamp(8) + checksum(8) = 45
const MIN_ENTRY_BYTES: usize = 45;

// ─────────────────────────────────────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────────────────────────────────────

/// Errors returned by [`WalWriteAheadLog`] operations.
#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum WalError {
    /// A decoded entry's checksum does not match the computed value.
    #[error("checksum mismatch for entry {entry_seq}: expected {expected:#018x}, got {got:#018x}")]
    ChecksumMismatch {
        entry_seq: u64,
        expected: u64,
        got: u64,
    },

    /// The entry's byte representation is malformed.
    #[error("corrupted entry at seq {0}")]
    CorruptedEntry(u64),

    /// No transaction with the given id is tracked.
    #[error("transaction {0} not found")]
    TransactionNotFound(u64),

    /// The encoded entry exceeds the configured maximum.
    #[error("entry too large: {0} bytes")]
    EntryTooLarge(usize),

    /// The segment buffer is full.
    #[error("segment full")]
    SegmentFull,

    /// The four-byte magic header is invalid.
    #[error("invalid WAL magic bytes")]
    InvalidMagic,
}

// ─────────────────────────────────────────────────────────────────────────────
// WalOpType
// ─────────────────────────────────────────────────────────────────────────────

/// Discriminant for WAL entry operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum WalOpType {
    /// Regular key-value put.
    Put = 1,
    /// Key deletion.
    Delete = 2,
    /// Transaction begin marker.
    Begin = 3,
    /// Transaction commit marker.
    Commit = 4,
    /// Transaction rollback marker.
    Rollback = 5,
    /// Checkpoint — all prior committed entries may be discarded.
    Checkpoint = 6,
}

impl WalOpType {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Put),
            2 => Some(Self::Delete),
            3 => Some(Self::Begin),
            4 => Some(Self::Commit),
            5 => Some(Self::Rollback),
            6 => Some(Self::Checkpoint),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WalEntry
// ─────────────────────────────────────────────────────────────────────────────

/// A single decoded entry in the write-ahead log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalEntry {
    /// Monotonically increasing sequence number assigned at write time.
    pub seq_num: u64,
    /// Transaction identifier (0 = no transaction / standalone write).
    pub tx_id: u64,
    /// Operation type.
    pub op_type: WalOpType,
    /// Key bytes (may be empty for Begin/Commit/Rollback/Checkpoint).
    pub key: Vec<u8>,
    /// Value bytes (may be empty for Delete/Begin/Commit/Rollback/Checkpoint).
    pub value: Vec<u8>,
    /// Unix timestamp in nanoseconds captured at write time.
    pub timestamp: u64,
    /// FNV-1a 64-bit checksum of all preceding bytes in the encoded entry.
    pub checksum: u64,
}

impl WalEntry {
    /// Encodes this entry to its wire format.
    ///
    /// Layout:
    /// `[magic(4)][seq_num(8)][tx_id(8)][op_type(1)][key_len(4)][key(n)]`
    /// `[value_len(4)][value(m)][timestamp(8)][checksum(8)]`
    pub fn encode(&self) -> Vec<u8> {
        let total = 4 + 8 + 8 + 1 + 4 + self.key.len() + 4 + self.value.len() + 8 + 8;
        let mut buf = Vec::with_capacity(total);

        buf.extend_from_slice(&WAL_MAGIC);
        buf.extend_from_slice(&self.seq_num.to_le_bytes());
        buf.extend_from_slice(&self.tx_id.to_le_bytes());
        buf.push(self.op_type as u8);
        buf.extend_from_slice(&(self.key.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.key);
        buf.extend_from_slice(&(self.value.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.value);
        buf.extend_from_slice(&self.timestamp.to_le_bytes());

        // Checksum covers everything written so far.
        let checksum = fnv1a_64(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());

        buf
    }

    /// Attempts to decode one entry starting at `data[offset..]`.
    ///
    /// Returns `(entry, bytes_consumed)` on success.
    ///
    /// # Errors
    /// - [`WalError::InvalidMagic`] if the four-byte header does not match.
    /// - [`WalError::CorruptedEntry`] if the buffer is too short or an unknown
    ///   op-type is encountered.
    /// - [`WalError::ChecksumMismatch`] if the stored checksum disagrees.
    pub fn decode(data: &[u8], offset: usize) -> Result<(WalEntry, usize), WalError> {
        let buf = &data[offset..];

        if buf.len() < MIN_ENTRY_BYTES {
            return Err(WalError::CorruptedEntry(0));
        }

        // Magic
        if buf[..4] != WAL_MAGIC {
            return Err(WalError::InvalidMagic);
        }

        let seq_num = u64::from_le_bytes(
            buf[4..12]
                .try_into()
                .map_err(|_| WalError::CorruptedEntry(0))?,
        );
        let tx_id = u64::from_le_bytes(
            buf[12..20]
                .try_into()
                .map_err(|_| WalError::CorruptedEntry(seq_num))?,
        );
        let op_u8 = buf[20];
        let op_type = WalOpType::from_u8(op_u8).ok_or(WalError::CorruptedEntry(seq_num))?;

        let key_len = u32::from_le_bytes(
            buf[21..25]
                .try_into()
                .map_err(|_| WalError::CorruptedEntry(seq_num))?,
        ) as usize;

        let key_end = 25 + key_len;
        if buf.len() < key_end + 4 {
            return Err(WalError::CorruptedEntry(seq_num));
        }
        let key = buf[25..key_end].to_vec();

        let value_len = u32::from_le_bytes(
            buf[key_end..key_end + 4]
                .try_into()
                .map_err(|_| WalError::CorruptedEntry(seq_num))?,
        ) as usize;

        let value_start = key_end + 4;
        let value_end = value_start + value_len;
        let ts_end = value_end + 8;
        let cs_end = ts_end + 8;

        if buf.len() < cs_end {
            return Err(WalError::CorruptedEntry(seq_num));
        }

        let value = buf[value_start..value_end].to_vec();
        let timestamp = u64::from_le_bytes(
            buf[value_end..ts_end]
                .try_into()
                .map_err(|_| WalError::CorruptedEntry(seq_num))?,
        );
        let stored_checksum = u64::from_le_bytes(
            buf[ts_end..cs_end]
                .try_into()
                .map_err(|_| WalError::CorruptedEntry(seq_num))?,
        );

        // Verify checksum — covers the bytes *before* the 8-byte checksum field.
        let computed = fnv1a_64(&buf[..ts_end]);
        if computed != stored_checksum {
            return Err(WalError::ChecksumMismatch {
                entry_seq: seq_num,
                expected: stored_checksum,
                got: computed,
            });
        }

        let entry = WalEntry {
            seq_num,
            tx_id,
            op_type,
            key,
            value,
            timestamp,
            checksum: stored_checksum,
        };

        Ok((entry, cs_end))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Transaction state
// ─────────────────────────────────────────────────────────────────────────────

/// Lifecycle state of an in-flight or completed transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxState {
    /// Transaction is open and accepting writes.
    Active,
    /// Transaction has been durably committed.
    Committed,
    /// Transaction was explicitly rolled back.
    RolledBack,
}

/// An in-memory transaction record tracking accumulated entries.
#[derive(Debug, Clone)]
pub struct Transaction {
    /// Unique transaction identifier.
    pub tx_id: u64,
    /// Entries buffered within this transaction.
    pub entries: Vec<WalEntry>,
    /// Current lifecycle state.
    pub state: TxState,
}

impl Transaction {
    fn new(tx_id: u64) -> Self {
        Self {
            tx_id,
            entries: Vec::new(),
            state: TxState::Active,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration knobs for [`WalWriteAheadLog`].
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Maximum byte size of a single segment buffer before it is considered
    /// full.  `0` means unlimited.
    pub max_segment_size: usize,

    /// Whether each write should trigger an immediate sync (no-op for
    /// in-memory implementation, included for API completeness).
    pub sync_on_write: bool,

    /// Maximum number of entries per segment before a new one is started.
    /// `0` means unlimited.
    pub max_entries_per_segment: usize,

    /// How many entries to accumulate between automatic checkpoints.
    /// `0` disables auto-checkpointing.
    pub checkpoint_interval: usize,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            max_segment_size: 64 * 1024 * 1024, // 64 MiB
            sync_on_write: false,
            max_entries_per_segment: 100_000,
            checkpoint_interval: 10_000,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RecoveryResult
// ─────────────────────────────────────────────────────────────────────────────

/// Summary of a recovery scan over a WAL segment buffer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecoveryResult {
    /// Total entries successfully decoded from the byte buffer.
    pub entries_recovered: usize,
    /// Number of distinct transaction ids encountered.
    pub transactions_recovered: usize,
    /// Number of entries that were replayed (committed Put/Delete only).
    pub entries_replayed: usize,
    /// Number of entries discarded because they belonged to uncommitted txns.
    pub partial_tx_discarded: usize,
    /// Highest sequence number seen during recovery.
    pub last_seq_num: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// WalStats
// ─────────────────────────────────────────────────────────────────────────────

/// Live statistics snapshot of a [`WalWriteAheadLog`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WalStats {
    /// Total entries ever written (including control entries).
    pub total_entries: u64,
    /// Total encoded bytes ever appended to segment buffers.
    pub total_bytes: u64,
    /// Number of currently active (uncommitted) transactions.
    pub active_transactions: usize,
    /// Number of segment buffers managed (always 1 in this implementation).
    pub segments_count: usize,
    /// Sequence number of the most recent checkpoint entry.
    pub last_checkpoint_seq: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// WriteAheadLog
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory, crash-recovery Write-Ahead Log.
///
/// All entries are serialised into a `Vec<u8>` segment buffer using a
/// length-framed binary format with FNV-1a-64 checksums.  The buffer can be
/// exported, persisted externally, and then fed back into [`Self::recover`] on
/// restart.
///
/// # Example
///
/// ```rust
/// use ipfrs_storage::write_ahead_log::{WalWriteAheadLog, WalConfig, WalOpType};
///
/// let mut wal = WalWriteAheadLog::new(WalConfig::default());
/// let seq = wal.write(b"hello".to_vec(), b"world".to_vec(), WalOpType::Put).unwrap();
/// assert!(seq > 0);
/// let data = wal.export_segment();
/// assert!(!data.is_empty());
/// ```
pub struct WalWriteAheadLog {
    /// Encoded entry bytes for the current segment.
    segment: Vec<u8>,
    /// All decoded entries in insertion order (mirrors the segment buffer).
    entries: Vec<WalEntry>,
    /// Active and recently-ended transaction records.
    transactions: HashMap<u64, Transaction>,
    /// Monotonically increasing sequence counter.
    next_seq: u64,
    /// Monotonically increasing transaction id counter.
    next_tx_id: u64,
    /// Configuration.
    config: WalConfig,
    /// Cumulative statistics.
    stats: WalStats,
    /// Sequence number of the most recent committed checkpoint entry.
    last_checkpoint_seq: u64,
    /// Write count since last auto-checkpoint.
    writes_since_checkpoint: usize,
}

impl WalWriteAheadLog {
    /// Creates a new WAL with the given configuration.
    pub fn new(config: WalConfig) -> Self {
        Self {
            segment: Vec::new(),
            entries: Vec::new(),
            transactions: HashMap::new(),
            next_seq: 1,
            next_tx_id: 1,
            config,
            stats: WalStats {
                segments_count: 1,
                ..WalStats::default()
            },
            last_checkpoint_seq: 0,
            writes_since_checkpoint: 0,
        }
    }

    /// Returns the current wall-clock time in nanoseconds.
    fn now_ns() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }

    /// Allocates the next sequence number.
    fn alloc_seq(&mut self) -> u64 {
        let s = self.next_seq;
        self.next_seq += 1;
        s
    }

    /// Allocates the next transaction id.
    fn alloc_tx_id(&mut self) -> u64 {
        let id = self.next_tx_id;
        self.next_tx_id += 1;
        id
    }

    /// Appends a pre-constructed [`WalEntry`] to the segment and the entry list.
    fn append_entry(&mut self, entry: WalEntry) -> Result<u64, WalError> {
        let encoded = entry.encode();
        let encoded_len = encoded.len();

        // Guard: segment size limit
        if self.config.max_segment_size > 0
            && self.segment.len() + encoded_len > self.config.max_segment_size
        {
            return Err(WalError::SegmentFull);
        }

        // Guard: max entries per segment
        if self.config.max_entries_per_segment > 0
            && self.entries.len() >= self.config.max_entries_per_segment
        {
            return Err(WalError::SegmentFull);
        }

        let seq = entry.seq_num;
        self.segment.extend_from_slice(&encoded);
        self.stats.total_entries += 1;
        self.stats.total_bytes += encoded_len as u64;
        self.entries.push(entry);

        Ok(seq)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /// Writes a standalone (non-transactional) entry and returns its sequence
    /// number.  Auto-checkpoints when the configured interval is reached.
    pub fn write(&mut self, key: Vec<u8>, value: Vec<u8>, op: WalOpType) -> Result<u64, WalError> {
        let seq = self.alloc_seq();
        let entry = WalEntry {
            seq_num: seq,
            tx_id: 0,
            op_type: op,
            key,
            value,
            timestamp: Self::now_ns(),
            checksum: 0, // filled by encode()
        };
        let seq = self.append_entry(entry)?;

        self.writes_since_checkpoint += 1;
        if self.config.checkpoint_interval > 0
            && self.writes_since_checkpoint >= self.config.checkpoint_interval
        {
            self.checkpoint()?;
        }

        Ok(seq)
    }

    /// Begins a new transaction.  Returns the transaction id.
    pub fn begin_transaction(&mut self) -> u64 {
        let tx_id = self.alloc_tx_id();
        let seq = self.alloc_seq();
        let entry = WalEntry {
            seq_num: seq,
            tx_id,
            op_type: WalOpType::Begin,
            key: Vec::new(),
            value: Vec::new(),
            timestamp: Self::now_ns(),
            checksum: 0,
        };
        // Best-effort: silently ignore segment-full when starting a transaction
        // so the caller can still inspect the error on the first real write.
        let _ = self.append_entry(entry);

        let tx = Transaction::new(tx_id);
        self.transactions.insert(tx_id, tx);
        self.stats.active_transactions = self
            .transactions
            .values()
            .filter(|t| t.state == TxState::Active)
            .count();

        tx_id
    }

    /// Writes an entry within a transaction.  Returns the entry sequence number.
    pub fn write_tx(
        &mut self,
        tx_id: u64,
        key: Vec<u8>,
        value: Vec<u8>,
        op: WalOpType,
    ) -> Result<u64, WalError> {
        // Verify transaction is active.
        {
            let tx = self
                .transactions
                .get(&tx_id)
                .ok_or(WalError::TransactionNotFound(tx_id))?;
            if tx.state != TxState::Active {
                return Err(WalError::TransactionNotFound(tx_id));
            }
        }

        let seq = self.alloc_seq();
        let entry = WalEntry {
            seq_num: seq,
            tx_id,
            op_type: op,
            key,
            value,
            timestamp: Self::now_ns(),
            checksum: 0,
        };

        let seq = self.append_entry(entry.clone())?;

        // Buffer entry in the transaction record.
        if let Some(tx) = self.transactions.get_mut(&tx_id) {
            tx.entries.push(entry);
        }

        Ok(seq)
    }

    /// Commits a transaction: writes the Commit marker and marks it committed.
    pub fn commit_transaction(&mut self, tx_id: u64) -> Result<(), WalError> {
        {
            let tx = self
                .transactions
                .get(&tx_id)
                .ok_or(WalError::TransactionNotFound(tx_id))?;
            if tx.state != TxState::Active {
                return Err(WalError::TransactionNotFound(tx_id));
            }
        }

        let seq = self.alloc_seq();
        let entry = WalEntry {
            seq_num: seq,
            tx_id,
            op_type: WalOpType::Commit,
            key: Vec::new(),
            value: Vec::new(),
            timestamp: Self::now_ns(),
            checksum: 0,
        };
        self.append_entry(entry)?;

        if let Some(tx) = self.transactions.get_mut(&tx_id) {
            tx.state = TxState::Committed;
        }

        self.stats.active_transactions = self
            .transactions
            .values()
            .filter(|t| t.state == TxState::Active)
            .count();

        Ok(())
    }

    /// Rolls back a transaction: writes the Rollback marker and marks it rolled back.
    pub fn rollback_transaction(&mut self, tx_id: u64) -> Result<(), WalError> {
        {
            let tx = self
                .transactions
                .get(&tx_id)
                .ok_or(WalError::TransactionNotFound(tx_id))?;
            if tx.state != TxState::Active {
                return Err(WalError::TransactionNotFound(tx_id));
            }
        }

        let seq = self.alloc_seq();
        let entry = WalEntry {
            seq_num: seq,
            tx_id,
            op_type: WalOpType::Rollback,
            key: Vec::new(),
            value: Vec::new(),
            timestamp: Self::now_ns(),
            checksum: 0,
        };
        self.append_entry(entry)?;

        if let Some(tx) = self.transactions.get_mut(&tx_id) {
            tx.state = TxState::RolledBack;
        }

        self.stats.active_transactions = self
            .transactions
            .values()
            .filter(|t| t.state == TxState::Active)
            .count();

        Ok(())
    }

    /// Creates a checkpoint entry and returns its sequence number.
    ///
    /// All entries whose `seq_num` is less than or equal to the previous
    /// checkpoint sequence, **and** which do not belong to an active
    /// transaction, are removed from the in-memory entry list and the segment
    /// is rebuilt.
    pub fn checkpoint(&mut self) -> Result<u64, WalError> {
        let seq = self.alloc_seq();
        let entry = WalEntry {
            seq_num: seq,
            tx_id: 0,
            op_type: WalOpType::Checkpoint,
            key: Vec::new(),
            value: Vec::new(),
            timestamp: Self::now_ns(),
            checksum: 0,
        };
        self.append_entry(entry)?;

        // Collect tx_ids of currently active transactions so we don't discard
        // entries they might still need.
        let active_tx_ids: std::collections::HashSet<u64> = self
            .transactions
            .values()
            .filter(|t| t.state == TxState::Active)
            .map(|t| t.tx_id)
            .collect();

        let truncate_before = self.last_checkpoint_seq;

        // Retain entries that are either:
        // - From an active transaction (cannot discard yet), OR
        // - At or after the previous checkpoint sequence.
        self.entries
            .retain(|e| e.seq_num > truncate_before || active_tx_ids.contains(&e.tx_id));

        // Rebuild the segment buffer from the retained entries.
        let mut rebuilt = Vec::with_capacity(self.segment.len());
        for e in &self.entries {
            rebuilt.extend_from_slice(&e.encode());
        }
        self.segment = rebuilt;

        self.last_checkpoint_seq = seq;
        self.stats.last_checkpoint_seq = seq;
        self.writes_since_checkpoint = 0;

        Ok(seq)
    }

    /// Scans a raw byte buffer, decodes all valid entries, validates checksums,
    /// and returns a [`RecoveryResult`].
    ///
    /// Entries belonging to transactions that were never committed (i.e. no
    /// matching Commit entry appears) are discarded from the result.
    pub fn recover(data: &[u8]) -> Result<RecoveryResult, WalError> {
        let mut all_entries: Vec<WalEntry> = Vec::new();
        let mut pos = 0usize;

        while pos < data.len() {
            // Skip any padding / garbage between entries gracefully if the
            // magic bytes don't match — attempt to find the next magic marker.
            if pos + 4 <= data.len() && data[pos..pos + 4] != WAL_MAGIC {
                pos += 1;
                continue;
            }

            match WalEntry::decode(data, pos) {
                Ok((entry, consumed)) => {
                    pos += consumed;
                    all_entries.push(entry);
                }
                Err(WalError::ChecksumMismatch { .. }) => {
                    // Stop on first checksum failure to avoid replaying garbage.
                    break;
                }
                Err(_) => {
                    // Truncated entry — stop scanning.
                    break;
                }
            }
        }

        // Determine which transactions are fully committed.
        let mut committed_tx: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut rolled_back_tx: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut seen_tx: std::collections::HashSet<u64> = std::collections::HashSet::new();

        for e in &all_entries {
            if e.tx_id != 0 {
                seen_tx.insert(e.tx_id);
                match e.op_type {
                    WalOpType::Commit => {
                        committed_tx.insert(e.tx_id);
                    }
                    WalOpType::Rollback => {
                        rolled_back_tx.insert(e.tx_id);
                    }
                    _ => {}
                }
            }
        }

        let last_seq_num = all_entries.iter().map(|e| e.seq_num).max().unwrap_or(0);

        let entries_recovered = all_entries.len();
        let transactions_recovered = seen_tx.len();

        // Count discarded entries: entries from transactions that were started
        // but never committed (and not rolled back) are considered partial.
        let partial_tx_ids: std::collections::HashSet<u64> = seen_tx
            .iter()
            .filter(|id| !committed_tx.contains(id) && !rolled_back_tx.contains(id))
            .copied()
            .collect();

        let partial_tx_discarded: usize = all_entries
            .iter()
            .filter(|e| e.tx_id != 0 && partial_tx_ids.contains(&e.tx_id))
            .count();

        // Count replayable entries: standalone committed Put/Delete, or
        // transactional Put/Delete from committed transactions.
        let entries_replayed: usize = all_entries
            .iter()
            .filter(|e| {
                let is_data_op = matches!(e.op_type, WalOpType::Put | WalOpType::Delete);
                if !is_data_op {
                    return false;
                }
                if e.tx_id == 0 {
                    // Standalone entry — always replay.
                    return true;
                }
                // Only replay if the transaction committed.
                committed_tx.contains(&e.tx_id)
            })
            .count();

        Ok(RecoveryResult {
            entries_recovered,
            transactions_recovered,
            entries_replayed,
            partial_tx_discarded,
            last_seq_num,
        })
    }

    /// Iterates over the provided entries and calls `handler` for each committed
    /// Put or Delete entry.  Returns the number of entries passed to the handler.
    ///
    /// Standalone (tx_id == 0) Put/Delete entries are always replayed.
    /// Transactional entries are only replayed when their transaction's Commit
    /// marker is present in `entries`.
    pub fn replay(entries: &[WalEntry], handler: &mut dyn FnMut(&WalEntry)) -> usize {
        // First pass: determine which transactions committed.
        let committed_tx: std::collections::HashSet<u64> = entries
            .iter()
            .filter(|e| e.op_type == WalOpType::Commit && e.tx_id != 0)
            .map(|e| e.tx_id)
            .collect();

        // Second pass: invoke handler for each replayable entry in order.
        let mut count = 0usize;
        for entry in entries {
            let is_data = matches!(entry.op_type, WalOpType::Put | WalOpType::Delete);
            if !is_data {
                continue;
            }
            let replayable = entry.tx_id == 0 || committed_tx.contains(&entry.tx_id);
            if replayable {
                handler(entry);
                count += 1;
            }
        }
        count
    }

    /// Returns a clone of the current segment buffer.
    pub fn export_segment(&self) -> Vec<u8> {
        self.segment.clone()
    }

    /// Returns a snapshot of current statistics.
    pub fn stats(&self) -> WalStats {
        WalStats {
            total_entries: self.stats.total_entries,
            total_bytes: self.stats.total_bytes,
            active_transactions: self.stats.active_transactions,
            segments_count: 1,
            last_checkpoint_seq: self.last_checkpoint_seq,
        }
    }

    /// Removes all entries with `seq_num < seq_num` from the in-memory list and
    /// rebuilds the segment buffer.  Returns the number of entries removed.
    pub fn truncate_before(&mut self, seq_num: u64) -> usize {
        let before = self.entries.len();
        self.entries.retain(|e| e.seq_num >= seq_num);
        let removed = before - self.entries.len();

        if removed > 0 {
            let mut rebuilt = Vec::with_capacity(self.segment.len());
            for e in &self.entries {
                rebuilt.extend_from_slice(&e.encode());
            }
            self.segment = rebuilt;
        }

        removed
    }

    /// Returns the total number of in-memory entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Inline xorshift64 PRNG (no rand crate) ────────────────────────────────

    struct Xorshift64 {
        state: u64,
    }

    impl Xorshift64 {
        fn new(seed: u64) -> Self {
            Self {
                state: if seed == 0 {
                    0xDEAD_BEEF_CAFE_1234
                } else {
                    seed
                },
            }
        }

        fn next(&mut self) -> u64 {
            self.state ^= self.state << 13;
            self.state ^= self.state >> 7;
            self.state ^= self.state << 17;
            self.state
        }

        fn next_bytes(&mut self, len: usize) -> Vec<u8> {
            let mut out = Vec::with_capacity(len);
            while out.len() < len {
                let v = self.next();
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.truncate(len);
            out
        }
    }

    fn default_wal() -> WalWriteAheadLog {
        WalWriteAheadLog::new(WalConfig::default())
    }

    // ── 1. FNV-1a hash ────────────────────────────────────────────────────────

    #[test]
    fn test_fnv1a_64_empty() {
        assert_eq!(fnv1a_64(&[]), 14_695_981_039_346_656_037u64);
    }

    #[test]
    fn test_fnv1a_64_known_value() {
        // "foobar" → well-known FNV-1a-64 value
        let h = fnv1a_64(b"foobar");
        assert_ne!(h, 0);
        // Deterministic: same input same output
        assert_eq!(h, fnv1a_64(b"foobar"));
    }

    #[test]
    fn test_fnv1a_64_differs_by_one_byte() {
        let a = fnv1a_64(b"hello");
        let b = fnv1a_64(b"hEllo");
        assert_ne!(a, b);
    }

    // ── 2. WalOpType round-trip ───────────────────────────────────────────────

    #[test]
    fn test_wal_op_type_from_u8_all_valid() {
        for (byte, expected) in &[
            (1u8, WalOpType::Put),
            (2, WalOpType::Delete),
            (3, WalOpType::Begin),
            (4, WalOpType::Commit),
            (5, WalOpType::Rollback),
            (6, WalOpType::Checkpoint),
        ] {
            assert_eq!(WalOpType::from_u8(*byte), Some(*expected));
        }
    }

    #[test]
    fn test_wal_op_type_from_u8_invalid() {
        assert_eq!(WalOpType::from_u8(0), None);
        assert_eq!(WalOpType::from_u8(7), None);
        assert_eq!(WalOpType::from_u8(255), None);
    }

    // ── 3. WalEntry encode / decode round-trips ───────────────────────────────

    #[test]
    fn test_encode_decode_put_entry() {
        let key = b"my-key".to_vec();
        let value = b"my-value-data".to_vec();
        let entry = WalEntry {
            seq_num: 42,
            tx_id: 0,
            op_type: WalOpType::Put,
            key: key.clone(),
            value: value.clone(),
            timestamp: 123_456_789,
            checksum: 0,
        };
        let encoded = entry.encode();
        // Checksum is computed by encode, so retrieve it
        let cs = u64::from_le_bytes(encoded[encoded.len() - 8..].try_into().unwrap());

        let (decoded, consumed) = WalEntry::decode(&encoded, 0).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.seq_num, 42);
        assert_eq!(decoded.tx_id, 0);
        assert_eq!(decoded.op_type, WalOpType::Put);
        assert_eq!(decoded.key, key);
        assert_eq!(decoded.value, value);
        assert_eq!(decoded.timestamp, 123_456_789);
        assert_eq!(decoded.checksum, cs);
    }

    #[test]
    fn test_encode_decode_delete_entry() {
        let entry = WalEntry {
            seq_num: 1,
            tx_id: 5,
            op_type: WalOpType::Delete,
            key: b"del-key".to_vec(),
            value: Vec::new(),
            timestamp: 0,
            checksum: 0,
        };
        let enc = entry.encode();
        let (dec, _) = WalEntry::decode(&enc, 0).unwrap();
        assert_eq!(dec.op_type, WalOpType::Delete);
        assert_eq!(dec.key, b"del-key");
        assert!(dec.value.is_empty());
    }

    #[test]
    fn test_encode_decode_empty_key_value() {
        let entry = WalEntry {
            seq_num: 99,
            tx_id: 0,
            op_type: WalOpType::Checkpoint,
            key: Vec::new(),
            value: Vec::new(),
            timestamp: 0,
            checksum: 0,
        };
        let enc = entry.encode();
        let (dec, consumed) = WalEntry::decode(&enc, 0).unwrap();
        assert_eq!(consumed, enc.len());
        assert_eq!(dec.op_type, WalOpType::Checkpoint);
        assert!(dec.key.is_empty());
        assert!(dec.value.is_empty());
    }

    #[test]
    fn test_encode_decode_with_offset() {
        let padding = vec![0u8; 16];
        let entry = WalEntry {
            seq_num: 7,
            tx_id: 3,
            op_type: WalOpType::Put,
            key: b"k".to_vec(),
            value: b"v".to_vec(),
            timestamp: 1,
            checksum: 0,
        };
        let mut buf = padding.clone();
        buf.extend_from_slice(&entry.encode());

        let (dec, _) = WalEntry::decode(&buf, 16).unwrap();
        assert_eq!(dec.seq_num, 7);
    }

    #[test]
    fn test_encode_decode_large_payload() {
        let mut rng = Xorshift64::new(0xABCDEF01);
        let key = rng.next_bytes(512);
        let value = rng.next_bytes(4096);
        let entry = WalEntry {
            seq_num: 1000,
            tx_id: 0,
            op_type: WalOpType::Put,
            key: key.clone(),
            value: value.clone(),
            timestamp: 999,
            checksum: 0,
        };
        let enc = entry.encode();
        let (dec, _) = WalEntry::decode(&enc, 0).unwrap();
        assert_eq!(dec.key, key);
        assert_eq!(dec.value, value);
    }

    // ── 4. Checksum validation ────────────────────────────────────────────────

    #[test]
    fn test_checksum_mismatch_detected() {
        let entry = WalEntry {
            seq_num: 1,
            tx_id: 0,
            op_type: WalOpType::Put,
            key: b"k".to_vec(),
            value: b"v".to_vec(),
            timestamp: 0,
            checksum: 0,
        };
        let mut enc = entry.encode();
        // Flip a byte in the value region (byte 30 is within value for short entries)
        let flip_pos = enc.len() - 10; // well before checksum
        enc[flip_pos] ^= 0xFF;

        let result = WalEntry::decode(&enc, 0);
        assert!(matches!(result, Err(WalError::ChecksumMismatch { .. })));
    }

    #[test]
    fn test_invalid_magic_detected() {
        let entry = WalEntry {
            seq_num: 2,
            tx_id: 0,
            op_type: WalOpType::Put,
            key: b"k".to_vec(),
            value: b"v".to_vec(),
            timestamp: 0,
            checksum: 0,
        };
        let mut enc = entry.encode();
        enc[0] = b'Z'; // corrupt magic

        let result = WalEntry::decode(&enc, 0);
        assert!(matches!(result, Err(WalError::InvalidMagic)));
    }

    #[test]
    fn test_corrupted_entry_too_short() {
        let data = vec![b'W', b'A', b'L', b'X', 0x01]; // too short
        let result = WalEntry::decode(&data, 0);
        assert!(matches!(result, Err(WalError::CorruptedEntry(_))));
    }

    #[test]
    fn test_corrupted_entry_truncated_key() {
        let entry = WalEntry {
            seq_num: 5,
            tx_id: 0,
            op_type: WalOpType::Put,
            key: b"longkey12345678".to_vec(),
            value: b"val".to_vec(),
            timestamp: 0,
            checksum: 0,
        };
        let mut enc = entry.encode();
        enc.truncate(enc.len() / 2); // chop in half
        let result = WalEntry::decode(&enc, 0);
        assert!(result.is_err());
    }

    // ── 5. WriteAheadLog::write ───────────────────────────────────────────────

    #[test]
    fn test_write_returns_sequential_seq_nums() {
        let mut wal = default_wal();
        let s1 = wal
            .write(b"k1".to_vec(), b"v1".to_vec(), WalOpType::Put)
            .unwrap();
        let s2 = wal
            .write(b"k2".to_vec(), b"v2".to_vec(), WalOpType::Put)
            .unwrap();
        let s3 = wal
            .write(b"k3".to_vec(), b"v3".to_vec(), WalOpType::Delete)
            .unwrap();
        assert!(s1 < s2);
        assert!(s2 < s3);
    }

    #[test]
    fn test_write_increases_entry_count() {
        let mut wal = default_wal();
        assert_eq!(wal.entry_count(), 0);
        wal.write(b"a".to_vec(), b"b".to_vec(), WalOpType::Put)
            .unwrap();
        assert_eq!(wal.entry_count(), 1);
        wal.write(b"c".to_vec(), b"d".to_vec(), WalOpType::Put)
            .unwrap();
        assert_eq!(wal.entry_count(), 2);
    }

    #[test]
    fn test_write_segment_not_empty() {
        let mut wal = default_wal();
        wal.write(b"key".to_vec(), b"val".to_vec(), WalOpType::Put)
            .unwrap();
        assert!(!wal.export_segment().is_empty());
    }

    #[test]
    fn test_stats_tracks_entries_and_bytes() {
        let mut wal = default_wal();
        wal.write(b"x".to_vec(), b"y".to_vec(), WalOpType::Put)
            .unwrap();
        wal.write(b"x".to_vec(), Vec::new(), WalOpType::Delete)
            .unwrap();
        let s = wal.stats();
        assert_eq!(s.total_entries, 2);
        assert!(s.total_bytes > 0);
    }

    // ── 6. begin / commit / rollback ─────────────────────────────────────────

    #[test]
    fn test_begin_transaction_returns_unique_ids() {
        let mut wal = default_wal();
        let t1 = wal.begin_transaction();
        let t2 = wal.begin_transaction();
        let t3 = wal.begin_transaction();
        assert_ne!(t1, t2);
        assert_ne!(t2, t3);
    }

    #[test]
    fn test_write_tx_valid_transaction() {
        let mut wal = default_wal();
        let tx = wal.begin_transaction();
        let seq = wal
            .write_tx(tx, b"key".to_vec(), b"value".to_vec(), WalOpType::Put)
            .unwrap();
        assert!(seq > 0);
    }

    #[test]
    fn test_write_tx_unknown_transaction_errors() {
        let mut wal = default_wal();
        let result = wal.write_tx(9999, b"k".to_vec(), b"v".to_vec(), WalOpType::Put);
        assert!(matches!(result, Err(WalError::TransactionNotFound(9999))));
    }

    #[test]
    fn test_commit_transaction_succeeds() {
        let mut wal = default_wal();
        let tx = wal.begin_transaction();
        wal.write_tx(tx, b"k".to_vec(), b"v".to_vec(), WalOpType::Put)
            .unwrap();
        wal.commit_transaction(tx).unwrap();
    }

    #[test]
    fn test_commit_unknown_transaction_errors() {
        let mut wal = default_wal();
        let result = wal.commit_transaction(42);
        assert!(matches!(result, Err(WalError::TransactionNotFound(42))));
    }

    #[test]
    fn test_rollback_transaction_succeeds() {
        let mut wal = default_wal();
        let tx = wal.begin_transaction();
        wal.write_tx(tx, b"k".to_vec(), b"v".to_vec(), WalOpType::Put)
            .unwrap();
        wal.rollback_transaction(tx).unwrap();
    }

    #[test]
    fn test_rollback_unknown_transaction_errors() {
        let mut wal = default_wal();
        let result = wal.rollback_transaction(77);
        assert!(matches!(result, Err(WalError::TransactionNotFound(77))));
    }

    #[test]
    fn test_double_commit_errors() {
        let mut wal = default_wal();
        let tx = wal.begin_transaction();
        wal.commit_transaction(tx).unwrap();
        // Second commit should fail — tx is no longer active.
        let result = wal.commit_transaction(tx);
        assert!(result.is_err());
    }

    #[test]
    fn test_double_rollback_errors() {
        let mut wal = default_wal();
        let tx = wal.begin_transaction();
        wal.rollback_transaction(tx).unwrap();
        let result = wal.rollback_transaction(tx);
        assert!(result.is_err());
    }

    #[test]
    fn test_active_transactions_count_decreases_after_commit() {
        let mut wal = default_wal();
        let t1 = wal.begin_transaction();
        let t2 = wal.begin_transaction();
        assert_eq!(wal.stats().active_transactions, 2);
        wal.commit_transaction(t1).unwrap();
        assert_eq!(wal.stats().active_transactions, 1);
        wal.rollback_transaction(t2).unwrap();
        assert_eq!(wal.stats().active_transactions, 0);
    }

    // ── 7. Checkpoint ─────────────────────────────────────────────────────────

    #[test]
    fn test_checkpoint_returns_seq_num() {
        let mut wal = default_wal();
        wal.write(b"a".to_vec(), b"b".to_vec(), WalOpType::Put)
            .unwrap();
        let cp_seq = wal.checkpoint().unwrap();
        assert!(cp_seq > 0);
        assert_eq!(wal.stats().last_checkpoint_seq, cp_seq);
    }

    #[test]
    fn test_checkpoint_truncates_old_entries() {
        let mut wal = default_wal();
        for i in 0u8..50 {
            wal.write(vec![i], vec![i], WalOpType::Put).unwrap();
        }
        let before = wal.entry_count();
        wal.checkpoint().unwrap();
        // A second checkpoint should truncate the entries before the first.
        for i in 50u8..100 {
            wal.write(vec![i], vec![i], WalOpType::Put).unwrap();
        }
        wal.checkpoint().unwrap();
        let after = wal.entry_count();
        // After two checkpoints the oldest batch should be gone.
        assert!(after < before + 51); // strictly fewer than before + one checkpoint entry
    }

    #[test]
    fn test_checkpoint_preserves_active_tx_entries() {
        let mut wal = default_wal();
        let tx = wal.begin_transaction();
        wal.write_tx(tx, b"k".to_vec(), b"v".to_vec(), WalOpType::Put)
            .unwrap();
        wal.checkpoint().unwrap();
        // Active tx entry must survive the checkpoint.
        assert!(wal.entries.iter().any(|e| e.tx_id == tx));
    }

    #[test]
    fn test_auto_checkpoint_triggers() {
        let config = WalConfig {
            checkpoint_interval: 3,
            ..WalConfig::default()
        };
        let mut wal = WalWriteAheadLog::new(config);
        for i in 0u8..3 {
            wal.write(vec![i], vec![i], WalOpType::Put).unwrap();
        }
        // After 3 writes the auto-checkpoint should have fired.
        assert_eq!(wal.writes_since_checkpoint, 0);
        assert!(wal.last_checkpoint_seq > 0);
    }

    // ── 8. Segment export / rebuild ───────────────────────────────────────────

    #[test]
    fn test_export_segment_contains_all_entries() {
        let mut wal = default_wal();
        wal.write(b"k1".to_vec(), b"v1".to_vec(), WalOpType::Put)
            .unwrap();
        wal.write(b"k2".to_vec(), b"v2".to_vec(), WalOpType::Delete)
            .unwrap();
        let seg = wal.export_segment();
        // Parse back: should find exactly 2 entries.
        let mut pos = 0usize;
        let mut count = 0usize;
        while pos < seg.len() {
            let (_, consumed) = WalEntry::decode(&seg, pos).unwrap();
            pos += consumed;
            count += 1;
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn test_export_segment_is_deterministic() {
        let mut wal = default_wal();
        wal.write(b"key".to_vec(), b"val".to_vec(), WalOpType::Put)
            .unwrap();
        let a = wal.export_segment();
        let b = wal.export_segment();
        assert_eq!(a, b);
    }

    // ── 9. Recovery ───────────────────────────────────────────────────────────

    #[test]
    fn test_recover_clean_log() {
        let mut wal = default_wal();
        wal.write(b"a".to_vec(), b"1".to_vec(), WalOpType::Put)
            .unwrap();
        wal.write(b"b".to_vec(), b"2".to_vec(), WalOpType::Put)
            .unwrap();
        wal.write(b"c".to_vec(), Vec::new(), WalOpType::Delete)
            .unwrap();
        let seg = wal.export_segment();

        let rr = WalWriteAheadLog::recover(&seg).unwrap();
        assert_eq!(rr.entries_recovered, 3);
        assert_eq!(rr.entries_replayed, 3); // all standalone
        assert_eq!(rr.partial_tx_discarded, 0);
        assert_eq!(rr.transactions_recovered, 0);
        assert!(rr.last_seq_num >= 3);
    }

    #[test]
    fn test_recover_committed_transaction() {
        let mut wal = default_wal();
        let tx = wal.begin_transaction();
        wal.write_tx(tx, b"k".to_vec(), b"v".to_vec(), WalOpType::Put)
            .unwrap();
        wal.commit_transaction(tx).unwrap();
        let seg = wal.export_segment();

        let rr = WalWriteAheadLog::recover(&seg).unwrap();
        assert_eq!(rr.entries_replayed, 1); // the Put inside the committed tx
        assert_eq!(rr.partial_tx_discarded, 0);
        assert_eq!(rr.transactions_recovered, 1);
    }

    #[test]
    fn test_recover_partial_tx_discarded() {
        let mut wal = default_wal();
        let tx = wal.begin_transaction();
        wal.write_tx(tx, b"k".to_vec(), b"v".to_vec(), WalOpType::Put)
            .unwrap();
        // Deliberately do NOT commit — simulate crash.
        let seg = wal.export_segment();

        let rr = WalWriteAheadLog::recover(&seg).unwrap();
        assert_eq!(rr.entries_replayed, 0); // tx never committed
        assert!(rr.partial_tx_discarded > 0);
    }

    #[test]
    fn test_recover_rolled_back_tx_not_replayed() {
        let mut wal = default_wal();
        let tx = wal.begin_transaction();
        wal.write_tx(tx, b"k".to_vec(), b"v".to_vec(), WalOpType::Put)
            .unwrap();
        wal.rollback_transaction(tx).unwrap();
        let seg = wal.export_segment();

        let rr = WalWriteAheadLog::recover(&seg).unwrap();
        assert_eq!(rr.entries_replayed, 0);
        // Rolled back is not "partial" — it has an explicit end marker.
        assert_eq!(rr.partial_tx_discarded, 0);
    }

    #[test]
    fn test_recover_mixed_committed_and_partial() {
        let mut wal = default_wal();
        // Committed tx
        let t1 = wal.begin_transaction();
        wal.write_tx(t1, b"k1".to_vec(), b"v1".to_vec(), WalOpType::Put)
            .unwrap();
        wal.commit_transaction(t1).unwrap();
        // Partial tx (no commit)
        let t2 = wal.begin_transaction();
        wal.write_tx(t2, b"k2".to_vec(), b"v2".to_vec(), WalOpType::Put)
            .unwrap();
        wal.write_tx(t2, b"k3".to_vec(), b"v3".to_vec(), WalOpType::Put)
            .unwrap();
        let seg = wal.export_segment();

        let rr = WalWriteAheadLog::recover(&seg).unwrap();
        assert_eq!(rr.entries_replayed, 1); // only t1's Put
        assert!(rr.partial_tx_discarded >= 2); // t2's 2 Puts + Begin
        assert_eq!(rr.transactions_recovered, 2);
    }

    #[test]
    fn test_recover_empty_buffer() {
        let rr = WalWriteAheadLog::recover(&[]).unwrap();
        assert_eq!(rr.entries_recovered, 0);
        assert_eq!(rr.last_seq_num, 0);
    }

    #[test]
    fn test_recover_garbage_prefix_skipped() {
        let mut wal = default_wal();
        wal.write(b"k".to_vec(), b"v".to_vec(), WalOpType::Put)
            .unwrap();
        let mut seg = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00];
        seg.extend_from_slice(&wal.export_segment());

        let rr = WalWriteAheadLog::recover(&seg).unwrap();
        assert_eq!(rr.entries_recovered, 1);
    }

    #[test]
    fn test_recover_last_seq_num_correct() {
        let mut wal = default_wal();
        for i in 0u8..10 {
            wal.write(vec![i], vec![i], WalOpType::Put).unwrap();
        }
        let seg = wal.export_segment();
        let rr = WalWriteAheadLog::recover(&seg).unwrap();
        assert_eq!(rr.last_seq_num, 10);
    }

    // ── 10. Replay ────────────────────────────────────────────────────────────

    #[test]
    fn test_replay_standalone_entries() {
        let mut wal = default_wal();
        wal.write(b"k1".to_vec(), b"v1".to_vec(), WalOpType::Put)
            .unwrap();
        wal.write(b"k2".to_vec(), b"v2".to_vec(), WalOpType::Put)
            .unwrap();
        wal.write(b"k3".to_vec(), Vec::new(), WalOpType::Delete)
            .unwrap();

        let entries: Vec<WalEntry> = wal.entries.clone();
        let mut replayed = Vec::new();
        let count = WalWriteAheadLog::replay(&entries, &mut |e| {
            replayed.push(e.clone());
        });
        assert_eq!(count, 3);
        assert_eq!(replayed.len(), 3);
    }

    #[test]
    fn test_replay_skips_control_entries() {
        let mut wal = default_wal();
        wal.write(b"k".to_vec(), b"v".to_vec(), WalOpType::Put)
            .unwrap();
        wal.checkpoint().unwrap(); // adds a Checkpoint entry

        let entries: Vec<WalEntry> = wal.entries.clone();
        let mut count_called = 0usize;
        WalWriteAheadLog::replay(&entries, &mut |_| {
            count_called += 1;
        });
        // Only the Put entry should be replayed, not the Checkpoint.
        assert_eq!(count_called, 1);
    }

    #[test]
    fn test_replay_skips_uncommitted_tx() {
        let mut wal = default_wal();
        let tx = wal.begin_transaction();
        wal.write_tx(tx, b"k".to_vec(), b"v".to_vec(), WalOpType::Put)
            .unwrap();
        // No commit

        let entries: Vec<WalEntry> = wal.entries.clone();
        let mut count_called = 0usize;
        WalWriteAheadLog::replay(&entries, &mut |_| {
            count_called += 1;
        });
        assert_eq!(count_called, 0);
    }

    #[test]
    fn test_replay_includes_committed_tx() {
        let mut wal = default_wal();
        let tx = wal.begin_transaction();
        wal.write_tx(tx, b"k".to_vec(), b"v".to_vec(), WalOpType::Put)
            .unwrap();
        wal.write_tx(tx, b"k2".to_vec(), Vec::new(), WalOpType::Delete)
            .unwrap();
        wal.commit_transaction(tx).unwrap();

        let entries: Vec<WalEntry> = wal.entries.clone();
        let mut seen_keys: Vec<Vec<u8>> = Vec::new();
        WalWriteAheadLog::replay(&entries, &mut |e| {
            seen_keys.push(e.key.clone());
        });
        assert_eq!(seen_keys.len(), 2);
        assert!(seen_keys.contains(&b"k".to_vec()));
        assert!(seen_keys.contains(&b"k2".to_vec()));
    }

    #[test]
    fn test_replay_order_preserved() {
        let mut wal = default_wal();
        let keys: &[&[u8]] = &[b"alpha", b"beta_", b"gamma", b"delta"];
        for k in keys {
            wal.write(k.to_vec(), b"v".to_vec(), WalOpType::Put)
                .unwrap();
        }
        let entries: Vec<WalEntry> = wal.entries.clone();
        let mut order: Vec<Vec<u8>> = Vec::new();
        WalWriteAheadLog::replay(&entries, &mut |e| {
            order.push(e.key.clone());
        });
        for (i, k) in keys.iter().enumerate() {
            assert_eq!(order[i], k.to_vec());
        }
    }

    #[test]
    fn test_replay_empty_entries() {
        let count = WalWriteAheadLog::replay(&[], &mut |_| {});
        assert_eq!(count, 0);
    }

    // ── 11. Truncate ──────────────────────────────────────────────────────────

    #[test]
    fn test_truncate_before_removes_entries() {
        let mut wal = default_wal();
        for i in 1u8..=10 {
            wal.write(vec![i], vec![i], WalOpType::Put).unwrap();
        }
        assert_eq!(wal.entry_count(), 10);
        let removed = wal.truncate_before(6);
        assert_eq!(removed, 5); // seq 1-5 removed
        assert_eq!(wal.entry_count(), 5);
    }

    #[test]
    fn test_truncate_before_zero_removes_nothing() {
        let mut wal = default_wal();
        wal.write(b"k".to_vec(), b"v".to_vec(), WalOpType::Put)
            .unwrap();
        let removed = wal.truncate_before(0);
        assert_eq!(removed, 0);
        assert_eq!(wal.entry_count(), 1);
    }

    #[test]
    fn test_truncate_before_large_seq_removes_all() {
        let mut wal = default_wal();
        for i in 0u8..5 {
            wal.write(vec![i], vec![i], WalOpType::Put).unwrap();
        }
        let removed = wal.truncate_before(u64::MAX);
        assert_eq!(removed, 5);
        assert_eq!(wal.entry_count(), 0);
        assert!(wal.export_segment().is_empty());
    }

    #[test]
    fn test_truncate_rebuilds_segment() {
        let mut wal = default_wal();
        for i in 1u8..=6 {
            wal.write(vec![i], vec![i], WalOpType::Put).unwrap();
        }
        let seg_before = wal.export_segment().len();
        wal.truncate_before(4);
        let seg_after = wal.export_segment().len();
        assert!(seg_after < seg_before);
    }

    // ── 12. Stats ─────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_segments_count() {
        let wal = default_wal();
        assert_eq!(wal.stats().segments_count, 1);
    }

    #[test]
    fn test_stats_last_checkpoint_seq_initial_zero() {
        let wal = default_wal();
        assert_eq!(wal.stats().last_checkpoint_seq, 0);
    }

    #[test]
    fn test_stats_after_checkpoint() {
        let mut wal = default_wal();
        wal.write(b"k".to_vec(), b"v".to_vec(), WalOpType::Put)
            .unwrap();
        let cp = wal.checkpoint().unwrap();
        assert_eq!(wal.stats().last_checkpoint_seq, cp);
    }

    #[test]
    fn test_stats_total_bytes_grows() {
        let mut wal = default_wal();
        let b0 = wal.stats().total_bytes;
        wal.write(b"key-large".to_vec(), vec![0u8; 256], WalOpType::Put)
            .unwrap();
        let b1 = wal.stats().total_bytes;
        assert!(b1 > b0);
    }

    // ── 13. Segment full / size limits ────────────────────────────────────────

    #[test]
    fn test_segment_full_on_size_limit() {
        let config = WalConfig {
            max_segment_size: 100, // very small
            ..WalConfig::default()
        };
        let mut wal = WalWriteAheadLog::new(config);
        // First write might fit; eventually we should hit the limit.
        let mut hit_full = false;
        for i in 0u8..100 {
            let result = wal.write(vec![i], vec![i; 10], WalOpType::Put);
            if matches!(result, Err(WalError::SegmentFull)) {
                hit_full = true;
                break;
            }
        }
        assert!(hit_full, "Expected SegmentFull error to be triggered");
    }

    #[test]
    fn test_segment_full_on_entry_count_limit() {
        let config = WalConfig {
            max_entries_per_segment: 3,
            max_segment_size: 0, // unlimited size
            checkpoint_interval: 0,
            ..WalConfig::default()
        };
        let mut wal = WalWriteAheadLog::new(config);
        wal.write(b"a".to_vec(), b"1".to_vec(), WalOpType::Put)
            .unwrap();
        wal.write(b"b".to_vec(), b"2".to_vec(), WalOpType::Put)
            .unwrap();
        wal.write(b"c".to_vec(), b"3".to_vec(), WalOpType::Put)
            .unwrap();
        let result = wal.write(b"d".to_vec(), b"4".to_vec(), WalOpType::Put);
        assert!(matches!(result, Err(WalError::SegmentFull)));
    }

    // ── 14. Multiple transactions interleaved ─────────────────────────────────

    #[test]
    fn test_multiple_concurrent_transactions() {
        let mut wal = default_wal();
        let t1 = wal.begin_transaction();
        let t2 = wal.begin_transaction();
        wal.write_tx(t1, b"t1k1".to_vec(), b"v1".to_vec(), WalOpType::Put)
            .unwrap();
        wal.write_tx(t2, b"t2k1".to_vec(), b"v2".to_vec(), WalOpType::Put)
            .unwrap();
        wal.write_tx(t1, b"t1k2".to_vec(), b"v3".to_vec(), WalOpType::Put)
            .unwrap();
        wal.commit_transaction(t1).unwrap();
        wal.rollback_transaction(t2).unwrap();

        let entries: Vec<WalEntry> = wal.entries.clone();
        let mut replayed_keys: Vec<Vec<u8>> = Vec::new();
        WalWriteAheadLog::replay(&entries, &mut |e| {
            replayed_keys.push(e.key.clone());
        });
        // Only t1 keys should appear
        assert!(replayed_keys.contains(&b"t1k1".to_vec()));
        assert!(replayed_keys.contains(&b"t1k2".to_vec()));
        assert!(!replayed_keys.contains(&b"t2k1".to_vec()));
    }

    // ── 15. Encode / decode multiple entries sequentially ────────────────────

    #[test]
    fn test_sequential_encode_decode_in_buffer() {
        let mut buf = Vec::new();
        let mut entries = Vec::new();
        for i in 0u64..20 {
            let e = WalEntry {
                seq_num: i + 1,
                tx_id: 0,
                op_type: WalOpType::Put,
                key: format!("key-{i}").into_bytes(),
                value: format!("val-{i}").into_bytes(),
                timestamp: i * 1000,
                checksum: 0,
            };
            buf.extend_from_slice(&e.encode());
            entries.push(e);
        }

        let mut pos = 0usize;
        let mut decoded = Vec::new();
        while pos < buf.len() {
            let (e, consumed) = WalEntry::decode(&buf, pos).unwrap();
            decoded.push(e);
            pos += consumed;
        }
        assert_eq!(decoded.len(), 20);
        for (orig, dec) in entries.iter().zip(decoded.iter()) {
            assert_eq!(orig.seq_num, dec.seq_num);
            assert_eq!(orig.key, dec.key);
            assert_eq!(orig.value, dec.value);
        }
    }

    // ── 16. WalConfig defaults ────────────────────────────────────────────────

    #[test]
    fn test_wal_config_defaults_reasonable() {
        let cfg = WalConfig::default();
        assert!(cfg.max_segment_size > 0);
        assert!(cfg.max_entries_per_segment > 0);
        assert!(cfg.checkpoint_interval > 0);
    }

    // ── 17. Stress / randomised round-trip ───────────────────────────────────

    #[test]
    fn test_stress_random_writes_and_recovery() {
        let mut rng = Xorshift64::new(42);
        let mut wal = default_wal();

        for _ in 0..200 {
            let key_len = (rng.next() % 64 + 1) as usize;
            let val_len = (rng.next() % 256) as usize;
            let key = rng.next_bytes(key_len);
            let val = rng.next_bytes(val_len);
            wal.write(key, val, WalOpType::Put).unwrap();
        }

        let seg = wal.export_segment();
        let rr = WalWriteAheadLog::recover(&seg).unwrap();
        assert_eq!(rr.entries_recovered, 200);
        assert_eq!(rr.entries_replayed, 200);
    }

    #[test]
    fn test_stress_mixed_transactions_and_standalone() {
        let mut rng = Xorshift64::new(7777);
        let mut wal = default_wal();
        let mut committed_put_count = 0usize;
        let mut standalone_count = 0usize;

        for _ in 0..10 {
            // 5 standalone writes
            for j in 0u8..5 {
                wal.write(vec![j], rng.next_bytes(8), WalOpType::Put)
                    .unwrap();
                standalone_count += 1;
            }
            // A committed transaction with 3 puts
            let tx = wal.begin_transaction();
            for j in 0u8..3 {
                wal.write_tx(tx, vec![j], rng.next_bytes(8), WalOpType::Put)
                    .unwrap();
                committed_put_count += 1;
            }
            wal.commit_transaction(tx).unwrap();
            // An uncommitted transaction (simulated crash)
            let tx2 = wal.begin_transaction();
            wal.write_tx(tx2, b"partial".to_vec(), b"v".to_vec(), WalOpType::Put)
                .unwrap();
            // No commit
        }

        let seg = wal.export_segment();
        let rr = WalWriteAheadLog::recover(&seg).unwrap();
        assert_eq!(rr.entries_replayed, standalone_count + committed_put_count);
        assert!(rr.partial_tx_discarded > 0);
    }
}
