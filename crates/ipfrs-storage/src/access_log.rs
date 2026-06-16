//! Append-only storage access log with configurable retention, pattern analysis, and export.
//!
//! Provides [`StorageAccessLog`] for recording every storage operation with
//! a sliding-window of entries, cumulative statistics, and pattern detection.

use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────────────────────────────────────
// LogOperation
// ─────────────────────────────────────────────────────────────────────────────

/// The kind of storage operation that was performed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LogOperation {
    /// A block was read.
    Read,
    /// A block was written.
    Write,
    /// A block was deleted.
    Delete,
    /// Block metadata / existence was queried.
    Stat,
    /// A directory or namespace was listed.
    List,
}

// ─────────────────────────────────────────────────────────────────────────────
// AccessLogEntry
// ─────────────────────────────────────────────────────────────────────────────

/// One record in the access log.
#[derive(Clone, Debug)]
pub struct AccessLogEntry {
    /// Monotonically increasing ID assigned at log time.
    pub entry_id: u64,
    /// The numeric block identifier.
    pub block_id: u64,
    /// Content identifier (CID) string.
    pub cid: String,
    /// Which operation was performed.
    pub operation: LogOperation,
    /// Number of bytes involved in the operation.
    pub size_bytes: u64,
    /// Logical tick at which the operation occurred.
    pub tick: u64,
    /// Peer that triggered this operation, if it was a remote request.
    pub peer_id: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// AccessPattern
// ─────────────────────────────────────────────────────────────────────────────

/// Detected access pattern for the recent window of log entries.
#[derive(Clone, Debug, PartialEq)]
pub enum AccessPattern {
    /// Recent reads have strictly increasing block IDs.
    Sequential,
    /// No discernible order in recent accesses.
    Random,
    /// The same block has been accessed multiple times recently.
    Repeated {
        /// The block ID that was repeatedly accessed.
        block_id: u64,
    },
    /// Many [`LogOperation::Write`] operations in a short tick window.
    BurstWrite,
}

// ─────────────────────────────────────────────────────────────────────────────
// LogConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`StorageAccessLog`].
#[derive(Clone, Debug)]
pub struct LogConfig {
    /// Maximum number of entries kept in the sliding window (FIFO).
    pub max_entries: usize,
    /// Number of writes within `burst_window_ticks` that triggers [`AccessPattern::BurstWrite`].
    pub burst_write_threshold: usize,
    /// Tick window width used for burst-write detection.
    pub burst_window_ticks: u64,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            burst_write_threshold: 10,
            burst_window_ticks: 10,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AccessLogStats
// ─────────────────────────────────────────────────────────────────────────────

/// Snapshot of cumulative statistics for the access log.
#[derive(Clone, Debug)]
pub struct AccessLogStats {
    /// Total number of entries recorded (including evicted ones).
    pub total_entries: usize,
    /// Cumulative per-operation counts.
    pub by_operation: HashMap<LogOperation, u64>,
    /// Cumulative bytes read across all [`LogOperation::Read`] entries.
    pub total_bytes_read: u64,
    /// Cumulative bytes written across all [`LogOperation::Write`] entries.
    pub total_bytes_written: u64,
    /// Number of distinct block IDs ever accessed.
    pub unique_blocks_accessed: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// StorageAccessLog
// ─────────────────────────────────────────────────────────────────────────────

/// Append-only access log for storage operations.
///
/// Keeps a sliding window of at most `config.max_entries` entries; older
/// entries are evicted FIFO. Cumulative statistics survive eviction.
pub struct StorageAccessLog {
    /// Sliding window of log entries.
    pub entries: Vec<AccessLogEntry>,
    /// Monotonically increasing counter used to assign [`AccessLogEntry::entry_id`].
    pub next_entry_id: u64,
    /// Active configuration.
    pub config: LogConfig,
    /// Cumulative per-operation counts (never decremented on eviction).
    pub cumulative_ops: HashMap<LogOperation, u64>,
    /// Total bytes read, cumulative.
    pub cumulative_read_bytes: u64,
    /// Total bytes written, cumulative.
    pub cumulative_write_bytes: u64,
    /// Set of every block ID ever logged.
    pub seen_blocks: HashSet<u64>,
}

impl StorageAccessLog {
    /// Create a new [`StorageAccessLog`] with the provided configuration.
    pub fn new(config: LogConfig) -> Self {
        Self {
            entries: Vec::new(),
            next_entry_id: 0,
            config,
            cumulative_ops: HashMap::new(),
            cumulative_read_bytes: 0,
            cumulative_write_bytes: 0,
            seen_blocks: HashSet::new(),
        }
    }

    /// Record a storage operation.
    ///
    /// If the sliding window is full (`entries.len() >= max_entries`) the
    /// oldest entry is removed before the new one is appended.
    pub fn log(
        &mut self,
        block_id: u64,
        cid: String,
        operation: LogOperation,
        size_bytes: u64,
        tick: u64,
        peer_id: Option<String>,
    ) {
        // Enforce FIFO cap *before* inserting so we never exceed max_entries.
        if self.entries.len() >= self.config.max_entries {
            self.entries.remove(0);
        }

        let entry = AccessLogEntry {
            entry_id: self.next_entry_id,
            block_id,
            cid,
            operation,
            size_bytes,
            tick,
            peer_id,
        };

        self.next_entry_id += 1;

        // Update cumulative stats.
        *self.cumulative_ops.entry(operation).or_insert(0) += 1;
        match operation {
            LogOperation::Read => self.cumulative_read_bytes += size_bytes,
            LogOperation::Write => self.cumulative_write_bytes += size_bytes,
            _ => {}
        }
        self.seen_blocks.insert(block_id);

        self.entries.push(entry);
    }

    /// Detect the access pattern in the recent window.
    ///
    /// Priority order:
    /// 1. **BurstWrite** — many writes in the last `burst_window_ticks` ticks.
    /// 2. **Repeated** — any block ID appears ≥ 3 times in the last 10 reads.
    /// 3. **Sequential** — the last 5 reads have strictly increasing block IDs.
    /// 4. **Random** — fallthrough.
    pub fn detect_pattern(&self, current_tick: u64) -> AccessPattern {
        // ── BurstWrite check ──────────────────────────────────────────────
        let window_start = current_tick.saturating_sub(self.config.burst_window_ticks);
        let burst_writes = self
            .entries
            .iter()
            .filter(|e| e.operation == LogOperation::Write && e.tick >= window_start)
            .count();
        if burst_writes >= self.config.burst_write_threshold {
            return AccessPattern::BurstWrite;
        }

        // Take last 10 entries for the remaining checks.
        let recent: Vec<&AccessLogEntry> = {
            let start = self.entries.len().saturating_sub(10);
            self.entries[start..].iter().collect()
        };

        // ── Repeated check ────────────────────────────────────────────────
        // Count per-block occurrences in the last 10 entries (all ops).
        let mut freq: HashMap<u64, usize> = HashMap::new();
        for e in &recent {
            *freq.entry(e.block_id).or_insert(0) += 1;
        }
        // Return the first (lowest entry_id order) block that appears ≥ 3 times.
        for e in &recent {
            if freq.get(&e.block_id).copied().unwrap_or(0) >= 3 {
                return AccessPattern::Repeated {
                    block_id: e.block_id,
                };
            }
        }

        // ── Sequential check ──────────────────────────────────────────────
        // Collect block IDs of the last 5 Read entries (from the full window).
        let last_five_reads: Vec<u64> = self
            .entries
            .iter()
            .filter(|e| e.operation == LogOperation::Read)
            .rev()
            .take(5)
            .map(|e| e.block_id)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        if last_five_reads.len() == 5 && last_five_reads.windows(2).all(|w| w[0] < w[1]) {
            return AccessPattern::Sequential;
        }

        AccessPattern::Random
    }

    /// Return all entries for a specific block ID, in log order.
    pub fn entries_for_block(&self, block_id: u64) -> Vec<&AccessLogEntry> {
        self.entries
            .iter()
            .filter(|e| e.block_id == block_id)
            .collect()
    }

    /// Return all entries whose tick falls in `[from_tick, to_tick]` (inclusive).
    pub fn entries_in_range(&self, from_tick: u64, to_tick: u64) -> Vec<&AccessLogEntry> {
        self.entries
            .iter()
            .filter(|e| e.tick >= from_tick && e.tick <= to_tick)
            .collect()
    }

    /// Snapshot of cumulative statistics.
    pub fn stats(&self) -> AccessLogStats {
        AccessLogStats {
            total_entries: self.next_entry_id as usize,
            by_operation: self.cumulative_ops.clone(),
            total_bytes_read: self.cumulative_read_bytes,
            total_bytes_written: self.cumulative_write_bytes,
            unique_blocks_accessed: self.seen_blocks.len(),
        }
    }

    /// Clear the sliding window of entries.
    ///
    /// Cumulative statistics (ops counts, bytes, seen_blocks) are *not* reset.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_log(max: usize) -> StorageAccessLog {
        let cfg = LogConfig {
            max_entries: max,
            burst_write_threshold: 10,
            burst_window_ticks: 10,
        };
        StorageAccessLog::new(cfg)
    }

    fn log_op(log: &mut StorageAccessLog, block_id: u64, op: LogOperation, size: u64, tick: u64) {
        log.log(block_id, format!("cid-{block_id}"), op, size, tick, None);
    }

    // ── 1. log appends entry ─────────────────────────────────────────────────
    #[test]
    fn test_log_appends_entry() {
        let mut log = make_log(100);
        log_op(&mut log, 1, LogOperation::Read, 512, 1);
        assert_eq!(log.entries.len(), 1);
        assert_eq!(log.entries[0].block_id, 1);
        assert_eq!(log.entries[0].operation, LogOperation::Read);
        assert_eq!(log.entries[0].size_bytes, 512);
        assert_eq!(log.entries[0].tick, 1);
    }

    // ── 2. entry_id is monotonically increasing ──────────────────────────────
    #[test]
    fn test_entry_id_monotonic() {
        let mut log = make_log(100);
        for i in 0..5_u64 {
            log_op(&mut log, i, LogOperation::Read, 1, i);
        }
        for (i, e) in log.entries.iter().enumerate() {
            assert_eq!(e.entry_id, i as u64);
        }
    }

    // ── 3. FIFO eviction when over max_entries ───────────────────────────────
    #[test]
    fn test_fifo_eviction() {
        let mut log = make_log(3);
        for i in 0..5_u64 {
            log_op(&mut log, i, LogOperation::Read, 1, i);
        }
        // Only last 3 entries should remain.
        assert_eq!(log.entries.len(), 3);
        assert_eq!(log.entries[0].block_id, 2);
        assert_eq!(log.entries[2].block_id, 4);
    }

    // ── 4. FIFO eviction does not decrement next_entry_id ───────────────────
    #[test]
    fn test_next_entry_id_not_decremented_on_eviction() {
        let mut log = make_log(2);
        for i in 0..5_u64 {
            log_op(&mut log, i, LogOperation::Read, 1, i);
        }
        assert_eq!(log.next_entry_id, 5);
    }

    // ── 5. cumulative ops stats update ────────────────────────────────────────
    #[test]
    fn test_cumulative_ops() {
        let mut log = make_log(100);
        log_op(&mut log, 1, LogOperation::Read, 100, 1);
        log_op(&mut log, 2, LogOperation::Write, 200, 2);
        log_op(&mut log, 3, LogOperation::Read, 300, 3);
        let stats = log.stats();
        assert_eq!(
            *stats.by_operation.get(&LogOperation::Read).unwrap_or(&0),
            2
        );
        assert_eq!(
            *stats.by_operation.get(&LogOperation::Write).unwrap_or(&0),
            1
        );
        assert_eq!(stats.total_bytes_read, 400);
        assert_eq!(stats.total_bytes_written, 200);
    }

    // ── 6. cumulative stats survive eviction ─────────────────────────────────
    #[test]
    fn test_cumulative_stats_survive_eviction() {
        let mut log = make_log(2);
        log_op(&mut log, 1, LogOperation::Read, 100, 1);
        log_op(&mut log, 2, LogOperation::Write, 200, 2);
        log_op(&mut log, 3, LogOperation::Read, 300, 3);
        // entry 0 (block 1, Read) should have been evicted.
        assert_eq!(log.entries.len(), 2);
        let stats = log.stats();
        assert_eq!(stats.total_bytes_read, 400); // 100 + 300
        assert_eq!(
            *stats.by_operation.get(&LogOperation::Read).unwrap_or(&0),
            2
        );
    }

    // ── 7. total_entries equals next_entry_id ────────────────────────────────
    #[test]
    fn test_total_entries_equals_logged_count() {
        let mut log = make_log(3);
        for i in 0..7_u64 {
            log_op(&mut log, i, LogOperation::Stat, 0, i);
        }
        assert_eq!(log.stats().total_entries, 7);
    }

    // ── 8. seen_blocks unique count ───────────────────────────────────────────
    #[test]
    fn test_seen_blocks_unique_count() {
        let mut log = make_log(100);
        for _ in 0..4 {
            log_op(&mut log, 42, LogOperation::Read, 1, 1);
        }
        log_op(&mut log, 99, LogOperation::Write, 1, 2);
        assert_eq!(log.stats().unique_blocks_accessed, 2);
    }

    // ── 9. seen_blocks survives eviction ─────────────────────────────────────
    #[test]
    fn test_seen_blocks_survive_eviction() {
        let mut log = make_log(2);
        log_op(&mut log, 10, LogOperation::Read, 1, 1);
        log_op(&mut log, 20, LogOperation::Read, 1, 2);
        log_op(&mut log, 30, LogOperation::Read, 1, 3);
        // block 10 was evicted but still counted.
        assert_eq!(log.stats().unique_blocks_accessed, 3);
    }

    // ── 10. detect_pattern BurstWrite ────────────────────────────────────────
    #[test]
    fn test_detect_pattern_burst_write() {
        let _log = make_log(100);
        let cfg = LogConfig {
            max_entries: 100,
            burst_write_threshold: 5,
            burst_window_ticks: 10,
        };
        let mut log = StorageAccessLog::new(cfg);
        for i in 0..5_u64 {
            log_op(&mut log, i, LogOperation::Write, 1, 5 + i);
        }
        assert_eq!(log.detect_pattern(10), AccessPattern::BurstWrite);
    }

    // ── 11. BurstWrite not triggered below threshold ──────────────────────────
    #[test]
    fn test_detect_pattern_no_burst_below_threshold() {
        let cfg = LogConfig {
            max_entries: 100,
            burst_write_threshold: 5,
            burst_window_ticks: 10,
        };
        let mut log = StorageAccessLog::new(cfg);
        for i in 0..4_u64 {
            log_op(&mut log, i, LogOperation::Write, 1, 5 + i);
        }
        // 4 writes < threshold 5 → should not be BurstWrite.
        assert_ne!(log.detect_pattern(10), AccessPattern::BurstWrite);
    }

    // ── 12. detect_pattern Repeated (block appears 3+ times) ─────────────────
    #[test]
    fn test_detect_pattern_repeated() {
        let mut log = make_log(100);
        for _ in 0..3 {
            log_op(&mut log, 77, LogOperation::Read, 1, 1);
        }
        log_op(&mut log, 99, LogOperation::Read, 1, 2);
        assert_eq!(
            log.detect_pattern(2),
            AccessPattern::Repeated { block_id: 77 }
        );
    }

    // ── 13. Repeated not triggered at 2 occurrences ──────────────────────────
    #[test]
    fn test_detect_pattern_repeated_needs_three() {
        let mut log = make_log(100);
        log_op(&mut log, 77, LogOperation::Read, 1, 1);
        log_op(&mut log, 77, LogOperation::Read, 1, 2);
        log_op(&mut log, 10, LogOperation::Read, 1, 3);
        log_op(&mut log, 20, LogOperation::Read, 1, 4);
        log_op(&mut log, 30, LogOperation::Read, 1, 5);
        // Only 2 occurrences of block 77 — should not be Repeated.
        let pat = log.detect_pattern(5);
        assert!(
            pat != AccessPattern::Repeated { block_id: 77 },
            "Should not be Repeated with only 2 hits"
        );
    }

    // ── 14. detect_pattern Sequential (strictly increasing block IDs) ─────────
    #[test]
    fn test_detect_pattern_sequential() {
        let mut log = make_log(100);
        for i in 1..=5_u64 {
            log_op(&mut log, i * 10, LogOperation::Read, 1, i);
        }
        assert_eq!(log.detect_pattern(5), AccessPattern::Sequential);
    }

    // ── 15. Sequential not triggered when not strictly increasing ─────────────
    #[test]
    fn test_detect_pattern_sequential_requires_strict_increase() {
        let mut log = make_log(100);
        // block IDs: 10, 20, 20, 30, 40 — not strictly increasing (dupe at 20).
        for bid in [10_u64, 20, 20, 30, 40] {
            log_op(&mut log, bid, LogOperation::Read, 1, 1);
        }
        assert_ne!(log.detect_pattern(1), AccessPattern::Sequential);
    }

    // ── 16. detect_pattern Random ────────────────────────────────────────────
    #[test]
    fn test_detect_pattern_random() {
        let mut log = make_log(100);
        for bid in [5_u64, 2, 9, 1] {
            log_op(&mut log, bid, LogOperation::Read, 1, 1);
        }
        assert_eq!(log.detect_pattern(1), AccessPattern::Random);
    }

    // ── 17. detect_pattern Random on empty log ───────────────────────────────
    #[test]
    fn test_detect_pattern_empty_is_random() {
        let log = make_log(100);
        assert_eq!(log.detect_pattern(0), AccessPattern::Random);
    }

    // ── 18. entries_for_block filters correctly ───────────────────────────────
    #[test]
    fn test_entries_for_block() {
        let mut log = make_log(100);
        log_op(&mut log, 1, LogOperation::Read, 1, 1);
        log_op(&mut log, 2, LogOperation::Write, 2, 2);
        log_op(&mut log, 1, LogOperation::Stat, 3, 3);
        log_op(&mut log, 3, LogOperation::Read, 4, 4);

        let result = log.entries_for_block(1);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].tick, 1);
        assert_eq!(result[1].tick, 3);
    }

    // ── 19. entries_for_block returns empty when block not present ────────────
    #[test]
    fn test_entries_for_block_not_found() {
        let mut log = make_log(100);
        log_op(&mut log, 42, LogOperation::Read, 1, 1);
        assert!(log.entries_for_block(99).is_empty());
    }

    // ── 20. entries_in_range inclusive bounds ─────────────────────────────────
    #[test]
    fn test_entries_in_range_inclusive() {
        let mut log = make_log(100);
        for tick in 1..=5_u64 {
            log_op(&mut log, tick, LogOperation::Read, 1, tick);
        }
        let result = log.entries_in_range(2, 4);
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|e| e.tick >= 2 && e.tick <= 4));
    }

    // ── 21. entries_in_range empty when range is outside ─────────────────────
    #[test]
    fn test_entries_in_range_outside() {
        let mut log = make_log(100);
        log_op(&mut log, 1, LogOperation::Read, 1, 5);
        assert!(log.entries_in_range(10, 20).is_empty());
    }

    // ── 22. entries_in_range single-tick range ────────────────────────────────
    #[test]
    fn test_entries_in_range_single_tick() {
        let mut log = make_log(100);
        log_op(&mut log, 1, LogOperation::Read, 1, 7);
        log_op(&mut log, 2, LogOperation::Read, 1, 7);
        log_op(&mut log, 3, LogOperation::Write, 1, 8);
        let result = log.entries_in_range(7, 7);
        assert_eq!(result.len(), 2);
    }

    // ── 23. clear keeps cumulative stats ─────────────────────────────────────
    #[test]
    fn test_clear_keeps_cumulative_stats() {
        let mut log = make_log(100);
        log_op(&mut log, 1, LogOperation::Read, 100, 1);
        log_op(&mut log, 2, LogOperation::Write, 200, 2);
        log.clear();

        assert!(log.entries.is_empty(), "entries should be cleared");
        let stats = log.stats();
        assert_eq!(stats.total_entries, 2); // next_entry_id not reset
        assert_eq!(stats.total_bytes_read, 100);
        assert_eq!(stats.total_bytes_written, 200);
        assert_eq!(stats.unique_blocks_accessed, 2);
    }

    // ── 24. clear does not reset next_entry_id ───────────────────────────────
    #[test]
    fn test_clear_does_not_reset_next_entry_id() {
        let mut log = make_log(100);
        log_op(&mut log, 1, LogOperation::Read, 1, 1);
        log_op(&mut log, 2, LogOperation::Read, 1, 2);
        log.clear();
        // Next log after clear should continue IDs.
        log_op(&mut log, 3, LogOperation::Read, 1, 3);
        assert_eq!(log.entries[0].entry_id, 2);
    }

    // ── 25. peer_id optional — None case ─────────────────────────────────────
    #[test]
    fn test_peer_id_none() {
        let mut log = make_log(100);
        log.log(1, "cid-1".into(), LogOperation::Read, 1, 1, None);
        assert!(log.entries[0].peer_id.is_none());
    }

    // ── 26. peer_id optional — Some case ─────────────────────────────────────
    #[test]
    fn test_peer_id_some() {
        let mut log = make_log(100);
        log.log(
            1,
            "cid-1".into(),
            LogOperation::Read,
            1,
            1,
            Some("peer-abc".into()),
        );
        assert_eq!(log.entries[0].peer_id.as_deref(), Some("peer-abc"));
    }

    // ── 27. Delete and Stat ops accumulate in by_operation ────────────────────
    #[test]
    fn test_delete_stat_list_in_by_operation() {
        let mut log = make_log(100);
        log_op(&mut log, 1, LogOperation::Delete, 0, 1);
        log_op(&mut log, 2, LogOperation::Stat, 0, 2);
        log_op(&mut log, 3, LogOperation::List, 0, 3);
        let stats = log.stats();
        assert_eq!(
            *stats.by_operation.get(&LogOperation::Delete).unwrap_or(&0),
            1
        );
        assert_eq!(
            *stats.by_operation.get(&LogOperation::Stat).unwrap_or(&0),
            1
        );
        assert_eq!(
            *stats.by_operation.get(&LogOperation::List).unwrap_or(&0),
            1
        );
    }

    // ── 28. BurstWrite ignores writes outside the window ──────────────────────
    #[test]
    fn test_burst_write_ignores_old_writes() {
        let cfg = LogConfig {
            max_entries: 100,
            burst_write_threshold: 3,
            burst_window_ticks: 5,
        };
        let mut log = StorageAccessLog::new(cfg);
        // 3 writes at tick 1 (far outside window when current_tick = 100)
        for i in 0..3_u64 {
            log_op(&mut log, i, LogOperation::Write, 1, 1);
        }
        // Only 2 writes within the window.
        log_op(&mut log, 10, LogOperation::Write, 1, 96);
        log_op(&mut log, 11, LogOperation::Write, 1, 97);
        // window_start = 100 - 5 = 95; only ticks 96 and 97 qualify → 2 < 3.
        assert_ne!(log.detect_pattern(100), AccessPattern::BurstWrite);
    }

    // ── 29. cid stored correctly ──────────────────────────────────────────────
    #[test]
    fn test_cid_stored_correctly() {
        let mut log = make_log(100);
        log.log(7, "bafyxyz".into(), LogOperation::Read, 1, 1, None);
        assert_eq!(log.entries[0].cid, "bafyxyz");
    }

    // ── 30. Repeated first-found ordering ────────────────────────────────────
    #[test]
    fn test_repeated_returns_first_found_block() {
        let mut log = make_log(100);
        // block 99 appears 3 times, block 77 appears 3 times; 99 comes first.
        for tick in 1..=3_u64 {
            log_op(&mut log, 99, LogOperation::Read, 1, tick);
        }
        for tick in 4..=6_u64 {
            log_op(&mut log, 77, LogOperation::Read, 1, tick);
        }
        // Last 10 entries has both. First in iteration order should win.
        let pat = log.detect_pattern(6);
        assert_eq!(pat, AccessPattern::Repeated { block_id: 99 });
    }
}
