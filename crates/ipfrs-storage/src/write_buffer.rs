//! Storage Write-Ahead Buffer — buffers incoming writes in memory before
//! flushing to the underlying storage backend.
//!
//! Provides durability semantics via monotonic sequence numbers so that any
//! unflushed entries can be replayed after a crash.

// ---------------------------------------------------------------------------
// WriteOp
// ---------------------------------------------------------------------------

/// An individual write operation staged in the buffer.
#[derive(Clone, Debug, PartialEq)]
pub enum WriteOp {
    /// Store a new block identified by `cid`.
    Put {
        /// Content identifier.
        cid: String,
        /// FNV / SHA hash of the block data.
        data_hash: u64,
        /// Size of the block payload in bytes.
        size_bytes: u64,
    },
    /// Remove the block identified by `cid`.
    Delete {
        /// Content identifier of the block to remove.
        cid: String,
    },
    /// Update the stored hash for an existing block.
    Update {
        /// Content identifier of the block to update.
        cid: String,
        /// Replacement hash value.
        new_hash: u64,
    },
}

// ---------------------------------------------------------------------------
// BufferedEntry
// ---------------------------------------------------------------------------

/// A single entry held in the [`StorageWriteAheadBuffer`].
#[derive(Clone, Debug)]
pub struct BufferedEntry {
    /// Monotonically increasing sequence number assigned at write time.
    pub seq: u64,
    /// The operation being buffered.
    pub op: WriteOp,
    /// Wall-clock seconds at which the entry was buffered (caller-supplied).
    pub buffered_at_secs: u64,
    /// `true` once the entry has been included in a [`flush`](StorageWriteAheadBuffer::flush) call.
    pub flushed: bool,
}

// ---------------------------------------------------------------------------
// FlushResult
// ---------------------------------------------------------------------------

/// Summary returned by [`StorageWriteAheadBuffer::flush`].
#[derive(Clone, Debug, PartialEq)]
pub struct FlushResult {
    /// Number of entries that were marked flushed in this batch.
    pub flushed_count: usize,
    /// Lowest sequence number in the flush batch (`u64::MAX` when batch is empty).
    pub from_seq: u64,
    /// Highest sequence number in the flush batch (`0` when batch is empty).
    pub to_seq: u64,
    /// Sum of `size_bytes` for every `Put` operation in the flush batch.
    pub total_bytes: u64,
}

// ---------------------------------------------------------------------------
// BufferConfig
// ---------------------------------------------------------------------------

/// Configuration knobs for [`StorageWriteAheadBuffer`].
#[derive(Clone, Debug)]
pub struct BufferConfig {
    /// Auto-flush when the number of buffered (unflushed) entries reaches this limit.
    pub max_buffered_entries: usize,
    /// Auto-flush when the total payload bytes of buffered `Put` ops reaches this limit.
    pub max_buffered_bytes: u64,
    /// Auto-flush when the oldest unflushed entry is at least this many seconds old.
    pub flush_interval_secs: u64,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            max_buffered_entries: 1_000,
            max_buffered_bytes: 64 * 1024 * 1024, // 64 MiB
            flush_interval_secs: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// StorageWriteAheadBuffer
// ---------------------------------------------------------------------------

/// In-memory write-ahead buffer with sequence-numbered entries.
///
/// Incoming [`WriteOp`]s are appended with monotonically increasing sequence
/// numbers.  The buffer tracks total payload bytes and the age of the oldest
/// unflushed entry so that callers can decide when to flush.
///
/// After a flush the entries remain in `entries` (marked `flushed = true`)
/// until [`trim_flushed`](Self::trim_flushed) is called, enabling crash-recovery
/// replay via [`replay_from`](Self::replay_from).
pub struct StorageWriteAheadBuffer {
    /// All entries in arrival order.
    pub entries: Vec<BufferedEntry>,
    /// Configuration controlling flush thresholds.
    pub config: BufferConfig,
    /// Next sequence number to assign.
    pub next_seq: u64,
    /// Running total of `size_bytes` for unflushed `Put` ops.
    pub total_buffered_bytes: u64,
}

impl StorageWriteAheadBuffer {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new, empty buffer with the supplied configuration.
    pub fn new(config: BufferConfig) -> Self {
        Self {
            entries: Vec::new(),
            config,
            next_seq: 0,
            total_buffered_bytes: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Write
    // -----------------------------------------------------------------------

    /// Stage `op` in the buffer and return the assigned sequence number.
    ///
    /// * `Put` ops increment [`total_buffered_bytes`](Self::total_buffered_bytes)
    ///   by their `size_bytes`.
    /// * `Delete` and `Update` ops do not affect the byte counter.
    pub fn write(&mut self, op: WriteOp, now_secs: u64) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;

        if let WriteOp::Put { size_bytes, .. } = &op {
            self.total_buffered_bytes = self.total_buffered_bytes.saturating_add(*size_bytes);
        }

        self.entries.push(BufferedEntry {
            seq,
            op,
            buffered_at_secs: now_secs,
            flushed: false,
        });

        seq
    }

    // -----------------------------------------------------------------------
    // Flush decision
    // -----------------------------------------------------------------------

    /// Returns `true` if any flush threshold has been reached.
    ///
    /// Triggers when:
    /// * the number of unflushed entries ≥ [`max_buffered_entries`](BufferConfig::max_buffered_entries), **or**
    /// * [`total_buffered_bytes`](Self::total_buffered_bytes) ≥ [`max_buffered_bytes`](BufferConfig::max_buffered_bytes), **or**
    /// * the oldest unflushed entry's age ≥ [`flush_interval_secs`](BufferConfig::flush_interval_secs).
    pub fn should_flush(&self, now_secs: u64) -> bool {
        let pending: Vec<&BufferedEntry> = self.pending_entries();

        if pending.len() >= self.config.max_buffered_entries {
            return true;
        }

        if self.total_buffered_bytes >= self.config.max_buffered_bytes {
            return true;
        }

        if let Some(oldest) = pending.first() {
            let age = now_secs.saturating_sub(oldest.buffered_at_secs);
            if age >= self.config.flush_interval_secs {
                return true;
            }
        }

        false
    }

    // -----------------------------------------------------------------------
    // Flush
    // -----------------------------------------------------------------------

    /// Mark every unflushed entry as flushed and return a [`FlushResult`].
    ///
    /// After this call [`total_buffered_bytes`](Self::total_buffered_bytes) is
    /// reset to `0`.  Flushed entries are retained in [`entries`](Self::entries)
    /// until [`trim_flushed`](Self::trim_flushed) is called.
    pub fn flush(&mut self, _now_secs: u64) -> FlushResult {
        let mut flushed_count: usize = 0;
        let mut from_seq: u64 = u64::MAX;
        let mut to_seq: u64 = 0;
        let mut total_bytes: u64 = 0;

        for entry in self.entries.iter_mut().filter(|e| !e.flushed) {
            entry.flushed = true;
            flushed_count += 1;

            if entry.seq < from_seq {
                from_seq = entry.seq;
            }
            if entry.seq > to_seq {
                to_seq = entry.seq;
            }

            if let WriteOp::Put { size_bytes, .. } = &entry.op {
                total_bytes = total_bytes.saturating_add(*size_bytes);
            }
        }

        self.total_buffered_bytes = 0;

        FlushResult {
            flushed_count,
            from_seq,
            to_seq,
            total_bytes,
        }
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Returns references to all entries that have **not** yet been flushed,
    /// in arrival order.
    pub fn pending_entries(&self) -> Vec<&BufferedEntry> {
        self.entries.iter().filter(|e| !e.flushed).collect()
    }

    /// Returns references to all entries whose sequence number is ≥ `seq`,
    /// regardless of flushed state.  Useful for crash-recovery replay.
    pub fn replay_from(&self, seq: u64) -> Vec<&BufferedEntry> {
        self.entries.iter().filter(|e| e.seq >= seq).collect()
    }

    // -----------------------------------------------------------------------
    // Maintenance
    // -----------------------------------------------------------------------

    /// Remove all entries that have been flushed, freeing memory.
    ///
    /// After this call, [`replay_from`](Self::replay_from) will only see
    /// entries that have not yet been flushed.
    pub fn trim_flushed(&mut self) {
        self.entries.retain(|e| !e.flushed);
    }

    /// Returns `(total_entry_count, unflushed_entry_count)`.
    pub fn stats(&self) -> (usize, u64) {
        let total = self.entries.len();
        let unflushed = self.entries.iter().filter(|e| !e.flushed).count() as u64;
        (total, unflushed)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_buf() -> StorageWriteAheadBuffer {
        StorageWriteAheadBuffer::new(BufferConfig::default())
    }

    fn put_op(cid: &str, size: u64) -> WriteOp {
        WriteOp::Put {
            cid: cid.to_string(),
            data_hash: 0xDEAD_BEEF,
            size_bytes: size,
        }
    }

    fn delete_op(cid: &str) -> WriteOp {
        WriteOp::Delete {
            cid: cid.to_string(),
        }
    }

    fn update_op(cid: &str, hash: u64) -> WriteOp {
        WriteOp::Update {
            cid: cid.to_string(),
            new_hash: hash,
        }
    }

    // 1. Sequential sequence numbers
    #[test]
    fn test_write_assigns_sequential_seqs() {
        let mut buf = default_buf();
        let s0 = buf.write(put_op("cid0", 100), 0);
        let s1 = buf.write(put_op("cid1", 200), 0);
        let s2 = buf.write(delete_op("cid0"), 0);
        assert_eq!(s0, 0);
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
    }

    // 2. next_seq advances correctly
    #[test]
    fn test_next_seq_increments() {
        let mut buf = default_buf();
        assert_eq!(buf.next_seq, 0);
        buf.write(put_op("a", 10), 0);
        assert_eq!(buf.next_seq, 1);
        buf.write(put_op("b", 20), 0);
        assert_eq!(buf.next_seq, 2);
    }

    // 3. Put increments total_buffered_bytes
    #[test]
    fn test_put_increments_buffered_bytes() {
        let mut buf = default_buf();
        buf.write(put_op("a", 512), 0);
        assert_eq!(buf.total_buffered_bytes, 512);
        buf.write(put_op("b", 1024), 0);
        assert_eq!(buf.total_buffered_bytes, 1536);
    }

    // 4. Delete does NOT increment total_buffered_bytes
    #[test]
    fn test_delete_does_not_increment_bytes() {
        let mut buf = default_buf();
        buf.write(delete_op("a"), 0);
        assert_eq!(buf.total_buffered_bytes, 0);
    }

    // 5. Update does NOT increment total_buffered_bytes
    #[test]
    fn test_update_does_not_increment_bytes() {
        let mut buf = default_buf();
        buf.write(update_op("a", 42), 0);
        assert_eq!(buf.total_buffered_bytes, 0);
    }

    // 6. should_flush triggers on count threshold
    #[test]
    fn test_should_flush_on_count() {
        let cfg = BufferConfig {
            max_buffered_entries: 3,
            max_buffered_bytes: u64::MAX,
            flush_interval_secs: u64::MAX,
        };
        let mut buf = StorageWriteAheadBuffer::new(cfg);
        buf.write(put_op("a", 1), 0);
        buf.write(put_op("b", 1), 0);
        assert!(!buf.should_flush(0));
        buf.write(put_op("c", 1), 0);
        assert!(buf.should_flush(0));
    }

    // 7. should_flush triggers on byte threshold
    #[test]
    fn test_should_flush_on_bytes() {
        let cfg = BufferConfig {
            max_buffered_entries: usize::MAX,
            max_buffered_bytes: 100,
            flush_interval_secs: u64::MAX,
        };
        let mut buf = StorageWriteAheadBuffer::new(cfg);
        buf.write(put_op("a", 60), 0);
        assert!(!buf.should_flush(0));
        buf.write(put_op("b", 40), 0);
        assert!(buf.should_flush(0));
    }

    // 8. should_flush triggers on age threshold
    #[test]
    fn test_should_flush_on_age() {
        let cfg = BufferConfig {
            max_buffered_entries: usize::MAX,
            max_buffered_bytes: u64::MAX,
            flush_interval_secs: 5,
        };
        let mut buf = StorageWriteAheadBuffer::new(cfg);
        buf.write(put_op("a", 1), 100); // buffered at t=100
        assert!(!buf.should_flush(104)); // age = 4 < 5
        assert!(buf.should_flush(105)); // age = 5 >= 5
    }

    // 9. Empty buffer never triggers should_flush (age path skipped)
    #[test]
    fn test_should_flush_empty() {
        let buf = default_buf();
        assert!(!buf.should_flush(9999));
    }

    // 10. flush marks all unflushed entries as flushed
    #[test]
    fn test_flush_marks_entries_flushed() {
        let mut buf = default_buf();
        buf.write(put_op("a", 10), 0);
        buf.write(delete_op("b"), 0);
        buf.flush(1);
        assert!(buf.entries.iter().all(|e| e.flushed));
    }

    // 11. FlushResult flushed_count
    #[test]
    fn test_flush_result_count() {
        let mut buf = default_buf();
        buf.write(put_op("a", 10), 0);
        buf.write(put_op("b", 20), 0);
        buf.write(delete_op("c"), 0);
        let result = buf.flush(1);
        assert_eq!(result.flushed_count, 3);
    }

    // 12. FlushResult from_seq / to_seq
    #[test]
    fn test_flush_result_seq_range() {
        let mut buf = default_buf();
        buf.write(put_op("a", 10), 0); // seq 0
        buf.write(put_op("b", 20), 0); // seq 1
        buf.write(delete_op("c"), 0); // seq 2
        let result = buf.flush(1);
        assert_eq!(result.from_seq, 0);
        assert_eq!(result.to_seq, 2);
    }

    // 13. FlushResult total_bytes sums only Put ops
    #[test]
    fn test_flush_result_total_bytes() {
        let mut buf = default_buf();
        buf.write(put_op("a", 100), 0);
        buf.write(delete_op("b"), 0);
        buf.write(put_op("c", 200), 0);
        buf.write(update_op("d", 7), 0);
        let result = buf.flush(1);
        assert_eq!(result.total_bytes, 300);
    }

    // 14. flush resets total_buffered_bytes to 0
    #[test]
    fn test_flush_resets_buffered_bytes() {
        let mut buf = default_buf();
        buf.write(put_op("a", 512), 0);
        assert_eq!(buf.total_buffered_bytes, 512);
        buf.flush(1);
        assert_eq!(buf.total_buffered_bytes, 0);
    }

    // 15. pending_entries returns 0 after flush
    #[test]
    fn test_pending_entries_after_flush_is_zero() {
        let mut buf = default_buf();
        buf.write(put_op("a", 10), 0);
        buf.write(delete_op("b"), 0);
        buf.flush(1);
        assert_eq!(buf.pending_entries().len(), 0);
    }

    // 16. pending_entries only returns unflushed
    #[test]
    fn test_pending_entries_partial_flush() {
        let mut buf = default_buf();
        buf.write(put_op("a", 10), 0); // seq 0
        buf.write(put_op("b", 20), 0); // seq 1
        buf.flush(1); // both flushed
        buf.write(put_op("c", 30), 0); // seq 2 — new, unflushed
        let pending = buf.pending_entries();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].seq, 2);
    }

    // 17. replay_from returns entries with seq >= given value
    #[test]
    fn test_replay_from() {
        let mut buf = default_buf();
        buf.write(put_op("a", 10), 0); // seq 0
        buf.write(put_op("b", 20), 0); // seq 1
        buf.write(delete_op("c"), 0); // seq 2
        buf.flush(1);

        let replayed = buf.replay_from(1);
        assert_eq!(replayed.len(), 2);
        assert_eq!(replayed[0].seq, 1);
        assert_eq!(replayed[1].seq, 2);
    }

    // 18. replay_from(0) returns all entries
    #[test]
    fn test_replay_from_zero_returns_all() {
        let mut buf = default_buf();
        buf.write(put_op("a", 10), 0);
        buf.write(put_op("b", 20), 0);
        buf.flush(1);
        assert_eq!(buf.replay_from(0).len(), 2);
    }

    // 19. trim_flushed removes flushed entries
    #[test]
    fn test_trim_flushed_removes_flushed() {
        let mut buf = default_buf();
        buf.write(put_op("a", 10), 0); // seq 0
        buf.write(put_op("b", 20), 0); // seq 1
        buf.flush(1);
        buf.write(put_op("c", 30), 0); // seq 2 — unflushed
        buf.trim_flushed();
        assert_eq!(buf.entries.len(), 1);
        assert_eq!(buf.entries[0].seq, 2);
    }

    // 20. trim_flushed on all-unflushed buffer is a no-op
    #[test]
    fn test_trim_flushed_noop_when_nothing_flushed() {
        let mut buf = default_buf();
        buf.write(put_op("a", 10), 0);
        buf.write(put_op("b", 20), 0);
        buf.trim_flushed();
        assert_eq!(buf.entries.len(), 2);
    }

    // 21. stats returns correct total and unflushed counts
    #[test]
    fn test_stats_unflushed_count() {
        let mut buf = default_buf();
        buf.write(put_op("a", 10), 0); // seq 0
        buf.write(put_op("b", 20), 0); // seq 1
        buf.write(delete_op("c"), 0); // seq 2
        let (total, unflushed) = buf.stats();
        assert_eq!(total, 3);
        assert_eq!(unflushed, 3);

        buf.flush(1);
        let (total2, unflushed2) = buf.stats();
        assert_eq!(total2, 3);
        assert_eq!(unflushed2, 0);
    }

    // 22. stats after trim
    #[test]
    fn test_stats_after_trim() {
        let mut buf = default_buf();
        buf.write(put_op("a", 10), 0);
        buf.write(put_op("b", 20), 0);
        buf.flush(1);
        buf.write(put_op("c", 30), 0);
        buf.trim_flushed();
        let (total, unflushed) = buf.stats();
        assert_eq!(total, 1);
        assert_eq!(unflushed, 1);
    }

    // 23. Empty flush result sentinel values
    #[test]
    fn test_flush_empty_buffer_sentinels() {
        let mut buf = default_buf();
        let result = buf.flush(0);
        assert_eq!(result.flushed_count, 0);
        assert_eq!(result.from_seq, u64::MAX);
        assert_eq!(result.to_seq, 0);
        assert_eq!(result.total_bytes, 0);
    }

    // 24. Second flush (all already flushed) returns zero count
    #[test]
    fn test_second_flush_is_empty() {
        let mut buf = default_buf();
        buf.write(put_op("a", 10), 0);
        buf.flush(1);
        let result2 = buf.flush(2);
        assert_eq!(result2.flushed_count, 0);
    }

    // 25. WriteOp derives Clone/Debug/PartialEq
    #[test]
    fn test_write_op_derives() {
        let op1 = WriteOp::Put {
            cid: "c1".to_string(),
            data_hash: 1,
            size_bytes: 64,
        };
        let op2 = op1.clone();
        assert_eq!(op1, op2);
        let _ = format!("{op1:?}");
    }
}
