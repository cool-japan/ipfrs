//! Storage access logger for audit trails and access pattern analysis.
//!
//! Maintains a structured, bounded FIFO log of storage operations,
//! enabling audit trails, compliance reporting, and access pattern detection.

use std::collections::VecDeque;

/// The type of storage operation recorded in an access entry.
#[derive(Clone, Debug, PartialEq)]
pub enum AccessOp {
    /// A single block retrieval.
    Get,
    /// A single block write.
    Put,
    /// A single block deletion.
    Delete,
    /// A block existence check.
    Exists,
    /// A batch retrieval of multiple blocks.
    BatchGet {
        /// Number of blocks in the batch.
        count: usize,
    },
    /// A batch write of multiple blocks.
    BatchPut {
        /// Number of blocks in the batch.
        count: usize,
    },
}

/// A single logged storage operation.
#[derive(Clone, Debug)]
pub struct AccessEntry {
    /// Monotonically increasing entry identifier.
    pub entry_id: u64,
    /// Content identifier of the block involved.
    pub cid: String,
    /// The type of operation performed.
    pub op: AccessOp,
    /// Bytes written for Put/BatchPut operations; `None` for Get, Exists, Delete.
    pub size_bytes: Option<u64>,
    /// Operation latency in microseconds.
    pub latency_us: u64,
    /// Whether the operation completed successfully.
    pub success: bool,
    /// Unix timestamp (seconds) at which the operation occurred.
    pub timestamp_secs: u64,
    /// Identifies the calling component (e.g. `"bitswap"`, `"gc"`).
    pub caller_tag: String,
}

/// Detected access pattern over the recent log window.
#[derive(Clone, Debug, PartialEq)]
pub enum AccessPattern {
    /// Consecutive entries share the same CID prefix (>50% of last 10 entries).
    Sequential,
    /// No identifiable structure — the default.
    Random,
    /// The same CID has been accessed 3 or more times in the last 10 entries.
    Repeated {
        /// The CID that was accessed repeatedly.
        cid: String,
    },
}

/// Aggregated statistics over all logged operations.
#[derive(Clone, Debug, Default)]
pub struct AccessStats {
    /// Total number of operations logged (including failures).
    pub total_ops: u64,
    /// Number of `Get` operations.
    pub gets: u64,
    /// Number of `Put` operations.
    pub puts: u64,
    /// Number of `Delete` operations.
    pub deletes: u64,
    /// Total bytes written across all `Put` and `BatchPut` operations.
    pub total_bytes_written: u64,
    /// Sum of latencies (microseconds) across all operations.
    pub total_latency_us: u64,
    /// Number of operations that did not succeed.
    pub failures: u64,
}

impl AccessStats {
    /// Average latency in microseconds. Returns `0.0` when no operations have been recorded.
    pub fn avg_latency_us(&self) -> f64 {
        self.total_latency_us as f64 / self.total_ops.max(1) as f64
    }

    /// Fraction of operations that failed. Returns `0.0` when no operations have been recorded.
    pub fn error_rate(&self) -> f64 {
        self.failures as f64 / self.total_ops.max(1) as f64
    }
}

/// Bounded, structured audit log for storage operations.
///
/// Maintains a FIFO ring of [`AccessEntry`] records up to `max_entries` in
/// length. When the ring is full, the oldest entry is evicted before the new
/// one is appended.  All mutations also update the running [`AccessStats`].
pub struct StorageAccessLogger {
    /// The ring of logged entries.
    pub entries: VecDeque<AccessEntry>,
    /// Maximum number of entries retained before oldest entries are dropped.
    pub max_entries: usize,
    /// Cumulative statistics over the lifetime of this logger (cleared with [`Self::clear`]).
    pub stats: AccessStats,
    /// Monotonic counter for the next [`AccessEntry::entry_id`].
    pub next_id: u64,
}

impl StorageAccessLogger {
    /// Create a new logger with the given capacity.
    ///
    /// `max_entries` sets the maximum number of entries held in memory.
    /// When this limit is reached the oldest entry is dropped on each new
    /// [`Self::log`] call. The default value used by higher-level helpers is
    /// `5000`.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(max_entries.min(4096)),
            max_entries,
            stats: AccessStats::default(),
            next_id: 0,
        }
    }

    /// Record a storage operation.
    ///
    /// # Parameters
    /// * `cid`           – Content identifier for the block involved.
    /// * `op`            – The operation type.
    /// * `size_bytes`    – Bytes written (meaningful for Put/BatchPut; pass `None` otherwise).
    /// * `latency_us`    – Operation latency in microseconds.
    /// * `success`       – Whether the operation completed without error.
    /// * `timestamp_secs`– Unix timestamp (seconds).
    /// * `caller_tag`    – String label for the calling component.
    #[allow(clippy::too_many_arguments)]
    pub fn log(
        &mut self,
        cid: String,
        op: AccessOp,
        size_bytes: Option<u64>,
        latency_us: u64,
        success: bool,
        timestamp_secs: u64,
        caller_tag: String,
    ) {
        let entry_id = self.next_id;
        self.next_id += 1;

        // Update statistics.
        self.stats.total_ops += 1;
        self.stats.total_latency_us += latency_us;
        if !success {
            self.stats.failures += 1;
        }
        match &op {
            AccessOp::Get => self.stats.gets += 1,
            AccessOp::Put => {
                self.stats.puts += 1;
                if let Some(bytes) = size_bytes {
                    self.stats.total_bytes_written += bytes;
                }
            }
            AccessOp::Delete => self.stats.deletes += 1,
            AccessOp::Exists => {}
            AccessOp::BatchGet { .. } => self.stats.gets += 1,
            AccessOp::BatchPut { .. } => {
                self.stats.puts += 1;
                if let Some(bytes) = size_bytes {
                    self.stats.total_bytes_written += bytes;
                }
            }
        }

        let entry = AccessEntry {
            entry_id,
            cid,
            op,
            size_bytes,
            latency_us,
            success,
            timestamp_secs,
            caller_tag,
        };

        // Enforce capacity bound.
        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    /// Return references to the last `n` entries in chronological order.
    ///
    /// If `n` exceeds the number of stored entries, all entries are returned.
    pub fn recent(&self, n: usize) -> Vec<&AccessEntry> {
        let len = self.entries.len();
        let skip = len.saturating_sub(n);
        self.entries.iter().skip(skip).collect()
    }

    /// Return all entries whose [`AccessEntry::cid`] matches `cid`.
    pub fn entries_for_cid(&self, cid: &str) -> Vec<&AccessEntry> {
        self.entries.iter().filter(|e| e.cid == cid).collect()
    }

    /// Return all entries whose [`AccessEntry::caller_tag`] matches `caller`.
    pub fn entries_for_caller(&self, caller: &str) -> Vec<&AccessEntry> {
        self.entries
            .iter()
            .filter(|e| e.caller_tag == caller)
            .collect()
    }

    /// Analyse the last 10 entries to detect an [`AccessPattern`].
    ///
    /// Detection priority:
    /// 1. **Repeated** — if any single CID appears 3 or more times.
    /// 2. **Random** — fallback (Sequential detection is deferred).
    pub fn detect_pattern(&self) -> AccessPattern {
        let window: Vec<&AccessEntry> = self.recent(10);

        // Check for repeated CID (≥3 occurrences).
        let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for entry in &window {
            *counts.entry(entry.cid.as_str()).or_insert(0) += 1;
        }
        if let Some((&cid, _)) = counts.iter().find(|(_, &c)| c >= 3) {
            return AccessPattern::Repeated {
                cid: cid.to_owned(),
            };
        }

        AccessPattern::Random
    }

    /// Return a reference to the current [`AccessStats`].
    pub fn stats(&self) -> &AccessStats {
        &self.stats
    }

    /// Reset the logger to its initial empty state.
    ///
    /// Clears all stored entries, resets statistics to zero, and restarts
    /// the entry-id counter from zero.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.stats = AccessStats::default();
        self.next_id = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience helper — logs a single Get with default metadata.
    fn log_get(logger: &mut StorageAccessLogger, cid: &str) {
        logger.log(
            cid.to_owned(),
            AccessOp::Get,
            None,
            100,
            true,
            1_000_000,
            "test".to_owned(),
        );
    }

    // -----------------------------------------------------------------------
    // 1. new() produces an empty logger
    // -----------------------------------------------------------------------
    #[test]
    fn test_new_empty() {
        let logger = StorageAccessLogger::new(5000);
        assert_eq!(logger.entries.len(), 0);
        assert_eq!(logger.stats.total_ops, 0);
        assert_eq!(logger.next_id, 0);
        assert_eq!(logger.max_entries, 5000);
    }

    // -----------------------------------------------------------------------
    // 2. log() Get updates stats.gets
    // -----------------------------------------------------------------------
    #[test]
    fn test_log_get_updates_gets() {
        let mut logger = StorageAccessLogger::new(100);
        log_get(&mut logger, "cid1");
        assert_eq!(logger.stats.gets, 1);
        assert_eq!(logger.stats.total_ops, 1);
        assert_eq!(logger.stats.puts, 0);
        assert_eq!(logger.stats.deletes, 0);
    }

    // -----------------------------------------------------------------------
    // 3. log() Put updates stats.puts and total_bytes_written
    // -----------------------------------------------------------------------
    #[test]
    fn test_log_put_updates_puts_and_bytes() {
        let mut logger = StorageAccessLogger::new(100);
        logger.log(
            "cid1".to_owned(),
            AccessOp::Put,
            Some(512),
            200,
            true,
            1_000_000,
            "writer".to_owned(),
        );
        assert_eq!(logger.stats.puts, 1);
        assert_eq!(logger.stats.total_bytes_written, 512);
        assert_eq!(logger.stats.gets, 0);
    }

    // -----------------------------------------------------------------------
    // 4. log() Delete updates stats.deletes
    // -----------------------------------------------------------------------
    #[test]
    fn test_log_delete_updates_deletes() {
        let mut logger = StorageAccessLogger::new(100);
        logger.log(
            "cid1".to_owned(),
            AccessOp::Delete,
            None,
            50,
            true,
            1_000_000,
            "gc".to_owned(),
        );
        assert_eq!(logger.stats.deletes, 1);
        assert_eq!(logger.stats.total_ops, 1);
    }

    // -----------------------------------------------------------------------
    // 5. log() failure increments failures
    // -----------------------------------------------------------------------
    #[test]
    fn test_log_failure_increments_failures() {
        let mut logger = StorageAccessLogger::new(100);
        logger.log(
            "cid1".to_owned(),
            AccessOp::Get,
            None,
            300,
            false,
            1_000_000,
            "bitswap".to_owned(),
        );
        assert_eq!(logger.stats.failures, 1);
        assert_eq!(logger.stats.total_ops, 1);
    }

    // -----------------------------------------------------------------------
    // 6. log() latency accumulated in total_latency_us
    // -----------------------------------------------------------------------
    #[test]
    fn test_log_latency_accumulated() {
        let mut logger = StorageAccessLogger::new(100);
        logger.log(
            "cid1".to_owned(),
            AccessOp::Get,
            None,
            100,
            true,
            1_000_000,
            "t".to_owned(),
        );
        logger.log(
            "cid2".to_owned(),
            AccessOp::Get,
            None,
            250,
            true,
            1_000_001,
            "t".to_owned(),
        );
        assert_eq!(logger.stats.total_latency_us, 350);
    }

    // -----------------------------------------------------------------------
    // 7. max_entries cap drops oldest entry
    // -----------------------------------------------------------------------
    #[test]
    fn test_max_entries_cap_drops_oldest() {
        let mut logger = StorageAccessLogger::new(3);
        for i in 0..5_u64 {
            logger.log(
                format!("cid{}", i),
                AccessOp::Get,
                None,
                10,
                true,
                i,
                "t".to_owned(),
            );
        }
        assert_eq!(logger.entries.len(), 3);
        // The three remaining entries should be the last three logged.
        let cids: Vec<&str> = logger.entries.iter().map(|e| e.cid.as_str()).collect();
        assert_eq!(cids, vec!["cid2", "cid3", "cid4"]);
    }

    // -----------------------------------------------------------------------
    // 8. recent() returns the last n entries
    // -----------------------------------------------------------------------
    #[test]
    fn test_recent_returns_last_n() {
        let mut logger = StorageAccessLogger::new(100);
        for i in 0..10_u64 {
            logger.log(
                format!("cid{}", i),
                AccessOp::Get,
                None,
                10,
                true,
                i,
                "t".to_owned(),
            );
        }
        let recent = logger.recent(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].cid, "cid7");
        assert_eq!(recent[1].cid, "cid8");
        assert_eq!(recent[2].cid, "cid9");
    }

    // -----------------------------------------------------------------------
    // 9. recent() n > len returns all entries
    // -----------------------------------------------------------------------
    #[test]
    fn test_recent_n_larger_than_len_returns_all() {
        let mut logger = StorageAccessLogger::new(100);
        for i in 0..5_u64 {
            log_get(&mut logger, &format!("cid{}", i));
        }
        let recent = logger.recent(50);
        assert_eq!(recent.len(), 5);
    }

    // -----------------------------------------------------------------------
    // 10. entries_for_cid filters correctly
    // -----------------------------------------------------------------------
    #[test]
    fn test_entries_for_cid_filters_correctly() {
        let mut logger = StorageAccessLogger::new(100);
        log_get(&mut logger, "cid_a");
        log_get(&mut logger, "cid_b");
        log_get(&mut logger, "cid_a");

        let hits = logger.entries_for_cid("cid_a");
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|e| e.cid == "cid_a"));

        let misses = logger.entries_for_cid("cid_x");
        assert!(misses.is_empty());
    }

    // -----------------------------------------------------------------------
    // 11. entries_for_caller filters correctly
    // -----------------------------------------------------------------------
    #[test]
    fn test_entries_for_caller_filters_correctly() {
        let mut logger = StorageAccessLogger::new(100);
        logger.log(
            "cid1".to_owned(),
            AccessOp::Get,
            None,
            10,
            true,
            1,
            "bitswap".to_owned(),
        );
        logger.log(
            "cid2".to_owned(),
            AccessOp::Put,
            Some(64),
            20,
            true,
            2,
            "gc".to_owned(),
        );
        logger.log(
            "cid3".to_owned(),
            AccessOp::Get,
            None,
            15,
            true,
            3,
            "bitswap".to_owned(),
        );

        let bitswap_entries = logger.entries_for_caller("bitswap");
        assert_eq!(bitswap_entries.len(), 2);
        assert!(bitswap_entries.iter().all(|e| e.caller_tag == "bitswap"));

        let gc_entries = logger.entries_for_caller("gc");
        assert_eq!(gc_entries.len(), 1);
    }

    // -----------------------------------------------------------------------
    // 12. detect_pattern: Repeated when same CID accessed >= 3 times in last 10
    // -----------------------------------------------------------------------
    #[test]
    fn test_detect_pattern_repeated() {
        let mut logger = StorageAccessLogger::new(100);
        // Fill with unrelated entries first.
        log_get(&mut logger, "other1");
        log_get(&mut logger, "other2");
        // Log "hot_cid" three times within the last 10.
        log_get(&mut logger, "hot_cid");
        log_get(&mut logger, "hot_cid");
        log_get(&mut logger, "hot_cid");

        let pattern = logger.detect_pattern();
        assert_eq!(
            pattern,
            AccessPattern::Repeated {
                cid: "hot_cid".to_owned()
            }
        );
    }

    // -----------------------------------------------------------------------
    // 13. detect_pattern: Random when no repetition
    // -----------------------------------------------------------------------
    #[test]
    fn test_detect_pattern_random() {
        let mut logger = StorageAccessLogger::new(100);
        for i in 0..10_u64 {
            log_get(&mut logger, &format!("unique_cid_{}", i));
        }
        assert_eq!(logger.detect_pattern(), AccessPattern::Random);
    }

    // -----------------------------------------------------------------------
    // 14. avg_latency_us calculation
    // -----------------------------------------------------------------------
    #[test]
    fn test_avg_latency_us_calculation() {
        let mut logger = StorageAccessLogger::new(100);
        logger.log(
            "c1".to_owned(),
            AccessOp::Get,
            None,
            100,
            true,
            1,
            "t".to_owned(),
        );
        logger.log(
            "c2".to_owned(),
            AccessOp::Get,
            None,
            300,
            true,
            2,
            "t".to_owned(),
        );
        // avg = (100 + 300) / 2 = 200
        let avg = logger.stats().avg_latency_us();
        assert!((avg - 200.0_f64).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // 15. error_rate calculation
    // -----------------------------------------------------------------------
    #[test]
    fn test_error_rate_calculation() {
        let mut logger = StorageAccessLogger::new(100);
        // 1 success
        logger.log(
            "c1".to_owned(),
            AccessOp::Get,
            None,
            10,
            true,
            1,
            "t".to_owned(),
        );
        // 1 failure
        logger.log(
            "c2".to_owned(),
            AccessOp::Get,
            None,
            10,
            false,
            2,
            "t".to_owned(),
        );
        // error_rate = 1/2 = 0.5
        let rate = logger.stats().error_rate();
        assert!((rate - 0.5_f64).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // 16. stats updated correctly after multiple mixed ops
    // -----------------------------------------------------------------------
    #[test]
    fn test_stats_updated_after_multiple_ops() {
        let mut logger = StorageAccessLogger::new(100);
        log_get(&mut logger, "cid1");
        logger.log(
            "cid2".to_owned(),
            AccessOp::Put,
            Some(1024),
            50,
            true,
            2,
            "w".to_owned(),
        );
        logger.log(
            "cid3".to_owned(),
            AccessOp::Delete,
            None,
            20,
            true,
            3,
            "gc".to_owned(),
        );
        logger.log(
            "cid4".to_owned(),
            AccessOp::Exists,
            None,
            5,
            false,
            4,
            "t".to_owned(),
        );

        let s = logger.stats();
        assert_eq!(s.total_ops, 4);
        assert_eq!(s.gets, 1);
        assert_eq!(s.puts, 1);
        assert_eq!(s.deletes, 1);
        assert_eq!(s.total_bytes_written, 1024);
        assert_eq!(s.failures, 1);
        assert_eq!(s.total_latency_us, 100 + 50 + 20 + 5);
    }

    // -----------------------------------------------------------------------
    // 17. clear() resets everything
    // -----------------------------------------------------------------------
    #[test]
    fn test_clear_resets_everything() {
        let mut logger = StorageAccessLogger::new(100);
        log_get(&mut logger, "cid1");
        log_get(&mut logger, "cid2");
        logger.clear();

        assert_eq!(logger.entries.len(), 0);
        assert_eq!(logger.stats.total_ops, 0);
        assert_eq!(logger.stats.gets, 0);
        assert_eq!(logger.stats.puts, 0);
        assert_eq!(logger.stats.deletes, 0);
        assert_eq!(logger.stats.total_bytes_written, 0);
        assert_eq!(logger.stats.total_latency_us, 0);
        assert_eq!(logger.stats.failures, 0);
        assert_eq!(logger.next_id, 0);
    }
}
