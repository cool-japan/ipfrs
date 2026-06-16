//! Storage Write Journal — append-only record of all storage mutations for
//! crash recovery, audit, and replication purposes.

// ---------------------------------------------------------------------------
// FNV-1a 64-bit hash
// ---------------------------------------------------------------------------

/// Computes the FNV-1a 64-bit hash of `bytes`.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;

    let mut hash = OFFSET_BASIS;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// JournalEntryKind
// ---------------------------------------------------------------------------

/// The kind of mutation recorded in a journal entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JournalEntryKind {
    /// A block was written to storage.
    Put,
    /// A block was removed from storage.
    Delete,
    /// A block was pinned (protected from GC).
    Pin,
    /// A block was unpinned.
    Unpin,
    /// A compaction event occurred (no CID; use empty string).
    Compact,
}

// ---------------------------------------------------------------------------
// JournalEntry
// ---------------------------------------------------------------------------

/// A single record in the write journal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JournalEntry {
    /// Monotonically increasing sequence number, starting at 1.
    pub sequence: u64,
    /// The kind of storage mutation.
    pub kind: JournalEntryKind,
    /// Content identifier of the affected block (empty for `Compact`).
    pub cid: String,
    /// Size in bytes of the block; 0 for non-`Put` entries.
    pub size_bytes: u64,
    /// Wall-clock timestamp in seconds (Unix epoch).
    pub timestamp_secs: u64,
    /// FNV-1a checksum of `cid.as_bytes()` XOR `sequence`.
    pub checksum: u64,
}

// ---------------------------------------------------------------------------
// JournalCursor
// ---------------------------------------------------------------------------

/// Opaque cursor into the write journal.
///
/// Used to track the position of a consumer so that only new entries need
/// to be delivered on subsequent calls to [`StorageWriteJournal::entries_since`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JournalCursor {
    /// The sequence number of the last entry seen by this cursor.
    /// `0` indicates the cursor is positioned before the very first entry.
    pub last_sequence: u64,
}

impl JournalCursor {
    /// Returns `true` when the cursor is positioned before any entry (i.e.,
    /// `last_sequence == 0`).
    pub fn is_at_start(&self) -> bool {
        self.last_sequence == 0
    }
}

// ---------------------------------------------------------------------------
// JournalStats
// ---------------------------------------------------------------------------

/// Aggregate statistics derived from the current state of the journal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JournalStats {
    /// Total number of entries currently held in the journal.
    pub total_entries: usize,
    /// Number of `Put` entries.
    pub put_count: usize,
    /// Number of `Delete` entries.
    pub delete_count: usize,
    /// Sum of `size_bytes` across all `Put` entries.
    pub total_bytes_journaled: u64,
    /// Sequence number of the oldest retained entry, or `None` if empty.
    pub oldest_sequence: Option<u64>,
    /// Sequence number of the newest retained entry, or `None` if empty.
    pub newest_sequence: Option<u64>,
}

// ---------------------------------------------------------------------------
// StorageWriteJournal
// ---------------------------------------------------------------------------

/// An append-only write journal that records all storage mutations.
///
/// The journal enforces a bounded capacity: when the number of stored entries
/// exceeds `max_entries`, the oldest entry is evicted.  Sequence numbers are
/// never reused — they continue to increment even after eviction, so consumers
/// holding a [`JournalCursor`] can detect that they have fallen behind.
pub struct StorageWriteJournal {
    /// Retained entries, in ascending sequence order.  Never reordered.
    pub entries: Vec<JournalEntry>,
    /// The sequence number that will be assigned to the *next* appended entry.
    pub next_sequence: u64,
    /// Maximum number of entries to retain.  When exceeded, the oldest entry
    /// is dropped.
    pub max_entries: usize,
}

impl StorageWriteJournal {
    /// Creates a new, empty journal with the given capacity limit.
    ///
    /// `next_sequence` is initialised to `1`.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            next_sequence: 1,
            max_entries,
        }
    }

    /// Appends a new entry to the journal.
    ///
    /// The checksum is computed as `fnv1a(cid.as_bytes()) ^ sequence`.
    ///
    /// If `entries.len() > max_entries` after appending, the oldest entry
    /// (index 0) is removed.
    ///
    /// Returns the sequence number assigned to the new entry.
    pub fn append(
        &mut self,
        kind: JournalEntryKind,
        cid: &str,
        size_bytes: u64,
        timestamp_secs: u64,
    ) -> u64 {
        let sequence = self.next_sequence;
        let checksum = fnv1a(cid.as_bytes()) ^ sequence;

        let entry = JournalEntry {
            sequence,
            kind,
            cid: cid.to_owned(),
            size_bytes,
            timestamp_secs,
            checksum,
        };

        self.entries.push(entry);

        if self.entries.len() > self.max_entries {
            self.entries.remove(0);
        }

        self.next_sequence += 1;
        sequence
    }

    /// Returns all entries with `sequence > cursor.last_sequence`, in order.
    pub fn entries_since<'a>(&'a self, cursor: &JournalCursor) -> Vec<&'a JournalEntry> {
        self.entries
            .iter()
            .filter(|e| e.sequence > cursor.last_sequence)
            .collect()
    }

    /// Returns a cursor pointing at the newest retained entry.
    ///
    /// If the journal is empty the cursor's `last_sequence` is `0`.
    pub fn cursor(&self) -> JournalCursor {
        let last_sequence = self.entries.last().map(|e| e.sequence).unwrap_or(0);
        JournalCursor { last_sequence }
    }

    /// Verifies the checksum of `entry`.
    ///
    /// Returns `true` when `fnv1a(entry.cid.as_bytes()) ^ entry.sequence`
    /// equals `entry.checksum`.
    pub fn verify_checksum(&self, entry: &JournalEntry) -> bool {
        let expected = fnv1a(entry.cid.as_bytes()) ^ entry.sequence;
        expected == entry.checksum
    }

    /// Returns aggregate statistics for the current journal contents.
    pub fn stats(&self) -> JournalStats {
        let mut put_count = 0usize;
        let mut delete_count = 0usize;
        let mut total_bytes_journaled = 0u64;

        for entry in &self.entries {
            match entry.kind {
                JournalEntryKind::Put => {
                    put_count += 1;
                    total_bytes_journaled = total_bytes_journaled.saturating_add(entry.size_bytes);
                }
                JournalEntryKind::Delete => {
                    delete_count += 1;
                }
                _ => {}
            }
        }

        let oldest_sequence = self.entries.first().map(|e| e.sequence);
        let newest_sequence = self.entries.last().map(|e| e.sequence);

        JournalStats {
            total_entries: self.entries.len(),
            put_count,
            delete_count,
            total_bytes_journaled,
            oldest_sequence,
            newest_sequence,
        }
    }

    /// Removes all entries whose sequence number is strictly less than
    /// `sequence`.
    pub fn truncate_before(&mut self, sequence: u64) {
        self.entries.retain(|e| e.sequence >= sequence);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- fnv1a ---

    #[test]
    fn test_fnv1a_empty_is_offset_basis() {
        assert_eq!(fnv1a(b""), 14_695_981_039_346_656_037);
    }

    #[test]
    fn test_fnv1a_deterministic() {
        assert_eq!(fnv1a(b"hello"), fnv1a(b"hello"));
    }

    #[test]
    fn test_fnv1a_different_inputs_differ() {
        assert_ne!(fnv1a(b"foo"), fnv1a(b"bar"));
    }

    // --- new() ---

    #[test]
    fn test_new_starts_empty() {
        let journal = StorageWriteJournal::new(100);
        assert!(journal.entries.is_empty());
    }

    #[test]
    fn test_new_next_sequence_is_one() {
        let journal = StorageWriteJournal::new(100);
        assert_eq!(journal.next_sequence, 1);
    }

    // --- append() ---

    #[test]
    fn test_append_returns_correct_sequence() {
        let mut journal = StorageWriteJournal::new(100);
        let seq = journal.append(JournalEntryKind::Put, "bafyabc", 512, 1_000);
        assert_eq!(seq, 1);
    }

    #[test]
    fn test_append_sequence_monotonically_increasing() {
        let mut journal = StorageWriteJournal::new(100);
        let s1 = journal.append(JournalEntryKind::Put, "cid1", 100, 1);
        let s2 = journal.append(JournalEntryKind::Delete, "cid2", 0, 2);
        let s3 = journal.append(JournalEntryKind::Pin, "cid3", 0, 3);
        assert!(s1 < s2 && s2 < s3);
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
        assert_eq!(s3, 3);
    }

    #[test]
    fn test_append_stores_entry_correctly() {
        let mut journal = StorageWriteJournal::new(100);
        journal.append(JournalEntryKind::Put, "bafytest", 1024, 999);
        let entry = &journal.entries[0];
        assert_eq!(entry.sequence, 1);
        assert_eq!(entry.kind, JournalEntryKind::Put);
        assert_eq!(entry.cid, "bafytest");
        assert_eq!(entry.size_bytes, 1024);
        assert_eq!(entry.timestamp_secs, 999);
    }

    #[test]
    fn test_append_checksum_is_fnv1a_xor_sequence() {
        let mut journal = StorageWriteJournal::new(100);
        let seq = journal.append(JournalEntryKind::Put, "bafychecksum", 0, 0);
        let entry = &journal.entries[0];
        let expected = fnv1a(b"bafychecksum") ^ seq;
        assert_eq!(entry.checksum, expected);
    }

    // --- verify_checksum() ---

    #[test]
    fn test_verify_checksum_true_for_valid_entry() {
        let mut journal = StorageWriteJournal::new(100);
        journal.append(JournalEntryKind::Put, "bafyvalid", 256, 42);
        let entry = &journal.entries[0];
        assert!(journal.verify_checksum(entry));
    }

    #[test]
    fn test_verify_checksum_false_for_tampered_cid() {
        let mut journal = StorageWriteJournal::new(100);
        journal.append(JournalEntryKind::Put, "bafyoriginal", 256, 42);
        let mut tampered = journal.entries[0].clone();
        tampered.cid = "bafytampered".to_owned();
        assert!(!journal.verify_checksum(&tampered));
    }

    #[test]
    fn test_verify_checksum_false_for_tampered_checksum() {
        let mut journal = StorageWriteJournal::new(100);
        journal.append(JournalEntryKind::Put, "bafyoriginal2", 256, 42);
        let mut tampered = journal.entries[0].clone();
        tampered.checksum ^= 0xDEAD_BEEF;
        assert!(!journal.verify_checksum(&tampered));
    }

    // --- entries_since() ---

    #[test]
    fn test_entries_since_zero_cursor_returns_all() {
        let mut journal = StorageWriteJournal::new(100);
        journal.append(JournalEntryKind::Put, "a", 1, 0);
        journal.append(JournalEntryKind::Put, "b", 2, 0);
        journal.append(JournalEntryKind::Delete, "c", 0, 0);
        let cursor = JournalCursor { last_sequence: 0 };
        let result = journal.entries_since(&cursor);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_entries_since_cursor_returns_only_newer() {
        let mut journal = StorageWriteJournal::new(100);
        journal.append(JournalEntryKind::Put, "a", 1, 0);
        journal.append(JournalEntryKind::Put, "b", 2, 0);
        journal.append(JournalEntryKind::Delete, "c", 0, 0);
        let cursor = JournalCursor { last_sequence: 1 };
        let result = journal.entries_since(&cursor);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].sequence, 2);
        assert_eq!(result[1].sequence, 3);
    }

    #[test]
    fn test_entries_since_preserves_order() {
        let mut journal = StorageWriteJournal::new(100);
        for i in 0..5u64 {
            journal.append(JournalEntryKind::Put, &format!("cid{i}"), i * 10, i);
        }
        let cursor = JournalCursor { last_sequence: 0 };
        let result = journal.entries_since(&cursor);
        let seqs: Vec<u64> = result.iter().map(|e| e.sequence).collect();
        let mut sorted = seqs.clone();
        sorted.sort_unstable();
        assert_eq!(seqs, sorted);
    }

    // --- cursor() ---

    #[test]
    fn test_cursor_returns_zero_when_empty() {
        let journal = StorageWriteJournal::new(100);
        assert_eq!(journal.cursor().last_sequence, 0);
    }

    #[test]
    fn test_cursor_returns_newest_sequence() {
        let mut journal = StorageWriteJournal::new(100);
        journal.append(JournalEntryKind::Put, "x", 0, 0);
        journal.append(JournalEntryKind::Put, "y", 0, 0);
        journal.append(JournalEntryKind::Put, "z", 0, 0);
        assert_eq!(journal.cursor().last_sequence, 3);
    }

    // --- is_at_start() ---

    #[test]
    fn test_is_at_start_true_when_last_sequence_zero() {
        let cursor = JournalCursor { last_sequence: 0 };
        assert!(cursor.is_at_start());
    }

    #[test]
    fn test_is_at_start_false_when_last_sequence_nonzero() {
        let cursor = JournalCursor { last_sequence: 5 };
        assert!(!cursor.is_at_start());
    }

    // --- max_entries enforcement ---

    #[test]
    fn test_max_entries_evicts_oldest() {
        let mut journal = StorageWriteJournal::new(3);
        journal.append(JournalEntryKind::Put, "a", 0, 0); // seq 1
        journal.append(JournalEntryKind::Put, "b", 0, 0); // seq 2
        journal.append(JournalEntryKind::Put, "c", 0, 0); // seq 3
        journal.append(JournalEntryKind::Put, "d", 0, 0); // seq 4 → evicts seq 1
        assert_eq!(journal.entries.len(), 3);
        assert_eq!(journal.entries[0].sequence, 2);
        assert_eq!(journal.entries[2].sequence, 4);
    }

    #[test]
    fn test_append_after_max_entries_has_correct_sequences() {
        let mut journal = StorageWriteJournal::new(2);
        for i in 0..5u64 {
            journal.append(JournalEntryKind::Put, &format!("cid{i}"), 0, 0);
        }
        // Only the 2 newest entries remain, with sequences 4 and 5.
        assert_eq!(journal.entries.len(), 2);
        assert_eq!(journal.entries[0].sequence, 4);
        assert_eq!(journal.entries[1].sequence, 5);
    }

    // --- truncate_before() ---

    #[test]
    fn test_truncate_before_removes_correct_entries() {
        let mut journal = StorageWriteJournal::new(100);
        for i in 0..5u64 {
            journal.append(JournalEntryKind::Put, &format!("cid{i}"), 0, 0);
        }
        journal.truncate_before(3);
        assert_eq!(journal.entries.len(), 3);
        assert_eq!(journal.entries[0].sequence, 3);
    }

    // --- stats() ---

    #[test]
    fn test_stats_total_entries_correct() {
        let mut journal = StorageWriteJournal::new(100);
        journal.append(JournalEntryKind::Put, "a", 10, 0);
        journal.append(JournalEntryKind::Delete, "b", 0, 0);
        journal.append(JournalEntryKind::Pin, "c", 0, 0);
        assert_eq!(journal.stats().total_entries, 3);
    }

    #[test]
    fn test_stats_put_count_correct() {
        let mut journal = StorageWriteJournal::new(100);
        journal.append(JournalEntryKind::Put, "a", 10, 0);
        journal.append(JournalEntryKind::Put, "b", 20, 0);
        journal.append(JournalEntryKind::Delete, "c", 0, 0);
        assert_eq!(journal.stats().put_count, 2);
    }

    #[test]
    fn test_stats_delete_count_correct() {
        let mut journal = StorageWriteJournal::new(100);
        journal.append(JournalEntryKind::Delete, "a", 0, 0);
        journal.append(JournalEntryKind::Put, "b", 50, 0);
        journal.append(JournalEntryKind::Delete, "c", 0, 0);
        assert_eq!(journal.stats().delete_count, 2);
    }

    #[test]
    fn test_stats_total_bytes_journaled_sums_put_entries_only() {
        let mut journal = StorageWriteJournal::new(100);
        journal.append(JournalEntryKind::Put, "a", 100, 0);
        journal.append(JournalEntryKind::Delete, "b", 999, 0); // size_bytes should be ignored
        journal.append(JournalEntryKind::Put, "c", 200, 0);
        // Only Put entries contribute → 100 + 200 = 300.
        assert_eq!(journal.stats().total_bytes_journaled, 300);
    }

    #[test]
    fn test_stats_oldest_newest_sequence() {
        let mut journal = StorageWriteJournal::new(100);
        journal.append(JournalEntryKind::Put, "a", 0, 0);
        journal.append(JournalEntryKind::Put, "b", 0, 0);
        journal.append(JournalEntryKind::Put, "c", 0, 0);
        let stats = journal.stats();
        assert_eq!(stats.oldest_sequence, Some(1));
        assert_eq!(stats.newest_sequence, Some(3));
    }

    #[test]
    fn test_stats_none_when_empty() {
        let journal = StorageWriteJournal::new(100);
        let stats = journal.stats();
        assert_eq!(stats.oldest_sequence, None);
        assert_eq!(stats.newest_sequence, None);
    }

    // --- JournalEntryKind::Compact ---

    #[test]
    fn test_compact_entry_appended_correctly() {
        let mut journal = StorageWriteJournal::new(100);
        let seq = journal.append(JournalEntryKind::Compact, "", 0, 77);
        assert_eq!(seq, 1);
        let entry = &journal.entries[0];
        assert_eq!(entry.kind, JournalEntryKind::Compact);
        assert_eq!(entry.cid, "");
        assert_eq!(entry.size_bytes, 0);
        assert_eq!(entry.timestamp_secs, 77);
        // Checksum must verify correctly even for empty CID.
        assert!(journal.verify_checksum(entry));
    }
}
