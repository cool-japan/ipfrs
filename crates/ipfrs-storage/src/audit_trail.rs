//! Immutable audit trail for storage operations.
//!
//! Provides tamper-evident logging of all storage mutations and access events
//! with FNV-1a integrity checksums, configurable retention, and flexible querying.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Audit event types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AuditEventType {
    BlockPut,
    BlockGet,
    BlockDelete,
    BlockPin,
    BlockUnpin,
    GarbageCollect,
    Compaction,
    Migration,
    Replication,
    AccessDenied,
}

impl AuditEventType {
    fn as_str(&self) -> &'static str {
        match self {
            Self::BlockPut => "BlockPut",
            Self::BlockGet => "BlockGet",
            Self::BlockDelete => "BlockDelete",
            Self::BlockPin => "BlockPin",
            Self::BlockUnpin => "BlockUnpin",
            Self::GarbageCollect => "GarbageCollect",
            Self::Compaction => "Compaction",
            Self::Migration => "Migration",
            Self::Replication => "Replication",
            Self::AccessDenied => "AccessDenied",
        }
    }
}

/// Single audit entry
#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub sequence: u64,
    pub timestamp: u64,
    pub event_type: AuditEventType,
    pub cid: String,
    pub actor: String,
    pub details: String,
    pub bytes_affected: u64,
    pub success: bool,
    pub checksum: u64,
}

/// Audit trail configuration
#[derive(Debug, Clone)]
pub struct AuditTrailConfig {
    pub max_entries: usize,
    pub retention_seconds: u64,
    pub enable_checksums: bool,
    pub log_reads: bool,
}

impl Default for AuditTrailConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            retention_seconds: 86_400, // 24 hours
            enable_checksums: true,
            log_reads: false,
        }
    }
}

/// Query filter for audit trail
#[derive(Debug, Clone, Default)]
pub struct AuditFilter {
    pub event_type: Option<AuditEventType>,
    pub cid: Option<String>,
    pub actor: Option<String>,
    pub since: Option<u64>,
    pub until: Option<u64>,
    pub success_only: bool,
}

/// Aggregate statistics for the audit trail
#[derive(Debug, Clone, Default)]
pub struct AuditTrailStats {
    pub total_entries: u64,
    pub entries_pruned: u64,
    pub integrity_failures: u64,
    pub events_by_type: HashMap<String, u64>,
}

/// Audit trail manager
pub struct AuditTrail {
    config: AuditTrailConfig,
    entries: Vec<AuditEntry>,
    next_sequence: u64,
    stats: AuditTrailStats,
}

// ---------------------------------------------------------------------------
// FNV-1a helper
// ---------------------------------------------------------------------------

/// Compute an FNV-1a 64-bit hash over the semantic fields of an [`AuditEntry`].
///
/// The checksum covers `sequence`, `timestamp`, `event_type`, `cid`, `actor`,
/// `details`, `bytes_affected`, and `success` — but **not** the `checksum`
/// field itself.
pub fn compute_checksum(entry: &AuditEntry) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;

    let mut hash = FNV_OFFSET;

    // Helper closure — feed bytes into the running hash.
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };

    feed(&entry.sequence.to_le_bytes());
    feed(&entry.timestamp.to_le_bytes());
    feed(entry.event_type.as_str().as_bytes());
    feed(entry.cid.as_bytes());
    feed(entry.actor.as_bytes());
    feed(entry.details.as_bytes());
    feed(&entry.bytes_affected.to_le_bytes());
    feed(&[u8::from(entry.success)]);

    hash
}

// ---------------------------------------------------------------------------
// Current-time helper (seconds since UNIX epoch)
// ---------------------------------------------------------------------------

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ---------------------------------------------------------------------------
// AuditTrail implementation
// ---------------------------------------------------------------------------

impl AuditTrail {
    /// Create a new audit trail with the given configuration.
    pub fn new(config: AuditTrailConfig) -> Self {
        Self {
            config,
            entries: Vec::new(),
            next_sequence: 1,
            stats: AuditTrailStats::default(),
        }
    }

    /// Record an event and return its sequence number.
    ///
    /// If `config.log_reads` is `false`, `BlockGet` events are silently
    /// dropped and the returned sequence number is `0`.
    pub fn log_event(
        &mut self,
        event_type: AuditEventType,
        cid: &str,
        actor: &str,
        details: &str,
        bytes: u64,
        success: bool,
    ) -> u64 {
        // Honour the log_reads flag.
        if event_type == AuditEventType::BlockGet && !self.config.log_reads {
            return 0;
        }

        let seq = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1);

        let mut entry = AuditEntry {
            sequence: seq,
            timestamp: now_secs(),
            event_type: event_type.clone(),
            cid: cid.to_string(),
            actor: actor.to_string(),
            details: details.to_string(),
            bytes_affected: bytes,
            success,
            checksum: 0,
        };

        if self.config.enable_checksums {
            entry.checksum = compute_checksum(&entry);
        }

        // Update stats.
        self.stats.total_entries += 1;
        *self
            .stats
            .events_by_type
            .entry(event_type.as_str().to_string())
            .or_insert(0) += 1;

        self.entries.push(entry);

        seq
    }

    /// Return entries matching every predicate in `filter`.
    pub fn query(&self, filter: &AuditFilter) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| {
                if let Some(ref et) = filter.event_type {
                    if e.event_type != *et {
                        return false;
                    }
                }
                if let Some(ref c) = filter.cid {
                    if e.cid != *c {
                        return false;
                    }
                }
                if let Some(ref a) = filter.actor {
                    if e.actor != *a {
                        return false;
                    }
                }
                if let Some(since) = filter.since {
                    if e.timestamp < since {
                        return false;
                    }
                }
                if let Some(until) = filter.until {
                    if e.timestamp > until {
                        return false;
                    }
                }
                if filter.success_only && !e.success {
                    return false;
                }
                true
            })
            .collect()
    }

    /// Look up an entry by its sequence number.
    pub fn get_entry(&self, sequence: u64) -> Option<&AuditEntry> {
        self.entries.iter().find(|e| e.sequence == sequence)
    }

    /// Verify the integrity checksum of every entry.
    ///
    /// Returns `Ok(())` when all entries pass, or `Err(bad_seqs)` with
    /// the sequence numbers that failed.
    pub fn verify_integrity(&self) -> Result<(), Vec<u64>> {
        let bad: Vec<u64> = self
            .entries
            .iter()
            .filter(|e| compute_checksum(e) != e.checksum)
            .map(|e| e.sequence)
            .collect();

        if bad.is_empty() {
            Ok(())
        } else {
            Err(bad)
        }
    }

    /// Remove all entries whose timestamp is strictly before `timestamp`.
    ///
    /// Returns the number of entries removed.
    pub fn prune_before(&mut self, timestamp: u64) -> usize {
        let before = self.entries.len();
        self.entries.retain(|e| e.timestamp >= timestamp);
        let pruned = before - self.entries.len();
        self.stats.entries_pruned += pruned as u64;
        pruned
    }

    /// Trim the trail to `config.max_entries`, removing the oldest entries
    /// first. Returns the number of entries removed.
    pub fn prune_to_capacity(&mut self) -> usize {
        let max = self.config.max_entries;
        if self.entries.len() <= max {
            return 0;
        }
        let excess = self.entries.len() - max;
        self.entries.drain(..excess);
        self.stats.entries_pruned += excess as u64;
        excess
    }

    /// Number of entries currently held.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// The most recently appended entry, if any.
    pub fn last_entry(&self) -> Option<&AuditEntry> {
        self.entries.last()
    }

    /// Current aggregate statistics.
    pub fn stats(&self) -> &AuditTrailStats {
        &self.stats
    }

    /// All events that reference a particular CID.
    pub fn events_for_cid(&self, cid: &str) -> Vec<&AuditEntry> {
        self.entries.iter().filter(|e| e.cid == cid).collect()
    }

    /// The most recent `count` entries where `success == false`, newest first.
    pub fn recent_failures(&self, count: usize) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .rev()
            .filter(|e| !e.success)
            .take(count)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_trail() -> AuditTrail {
        AuditTrail::new(AuditTrailConfig {
            max_entries: 100,
            retention_seconds: 3600,
            enable_checksums: true,
            log_reads: true,
        })
    }

    // -- basic logging -------------------------------------------------------

    #[test]
    fn test_log_block_put() {
        let mut trail = default_trail();
        let seq = trail.log_event(
            AuditEventType::BlockPut,
            "cid1",
            "local",
            "put block",
            1024,
            true,
        );
        assert_eq!(seq, 1);
        assert_eq!(trail.entry_count(), 1);
    }

    #[test]
    fn test_log_block_get() {
        let mut trail = default_trail();
        let seq = trail.log_event(
            AuditEventType::BlockGet,
            "cid2",
            "peer1",
            "get block",
            512,
            true,
        );
        assert!(seq > 0);
        assert_eq!(trail.entry_count(), 1);
    }

    #[test]
    fn test_log_block_delete() {
        let mut trail = default_trail();
        let seq = trail.log_event(
            AuditEventType::BlockDelete,
            "cid3",
            "local",
            "delete",
            256,
            true,
        );
        assert_eq!(seq, 1);
        let entry = trail.get_entry(seq).expect("entry should exist");
        assert_eq!(entry.event_type, AuditEventType::BlockDelete);
    }

    #[test]
    fn test_log_block_pin() {
        let mut trail = default_trail();
        let seq = trail.log_event(AuditEventType::BlockPin, "cid4", "local", "pin", 0, true);
        assert_eq!(
            trail.get_entry(seq).expect("entry").event_type,
            AuditEventType::BlockPin
        );
    }

    #[test]
    fn test_log_block_unpin() {
        let mut trail = default_trail();
        let seq = trail.log_event(
            AuditEventType::BlockUnpin,
            "cid5",
            "local",
            "unpin",
            0,
            true,
        );
        assert_eq!(
            trail.get_entry(seq).expect("entry").event_type,
            AuditEventType::BlockUnpin
        );
    }

    #[test]
    fn test_log_garbage_collect() {
        let mut trail = default_trail();
        let seq = trail.log_event(
            AuditEventType::GarbageCollect,
            "",
            "local",
            "gc round",
            4096,
            true,
        );
        assert_eq!(
            trail.get_entry(seq).expect("entry").event_type,
            AuditEventType::GarbageCollect
        );
    }

    #[test]
    fn test_log_compaction() {
        let mut trail = default_trail();
        let seq = trail.log_event(
            AuditEventType::Compaction,
            "",
            "local",
            "compact",
            8192,
            true,
        );
        assert_eq!(
            trail.get_entry(seq).expect("entry").event_type,
            AuditEventType::Compaction
        );
    }

    #[test]
    fn test_log_migration() {
        let mut trail = default_trail();
        let seq = trail.log_event(
            AuditEventType::Migration,
            "cid6",
            "local",
            "migrate",
            2048,
            true,
        );
        assert_eq!(
            trail.get_entry(seq).expect("entry").event_type,
            AuditEventType::Migration
        );
    }

    #[test]
    fn test_log_replication() {
        let mut trail = default_trail();
        let seq = trail.log_event(
            AuditEventType::Replication,
            "cid7",
            "peer2",
            "replicate",
            1024,
            true,
        );
        assert_eq!(
            trail.get_entry(seq).expect("entry").event_type,
            AuditEventType::Replication
        );
    }

    #[test]
    fn test_log_access_denied() {
        let mut trail = default_trail();
        let seq = trail.log_event(
            AuditEventType::AccessDenied,
            "cid8",
            "peer3",
            "denied",
            0,
            false,
        );
        let entry = trail.get_entry(seq).expect("entry");
        assert_eq!(entry.event_type, AuditEventType::AccessDenied);
        assert!(!entry.success);
    }

    // -- log_reads config ----------------------------------------------------

    #[test]
    fn test_log_reads_disabled() {
        let mut trail = AuditTrail::new(AuditTrailConfig {
            log_reads: false,
            ..AuditTrailConfig::default()
        });
        let seq = trail.log_event(AuditEventType::BlockGet, "cid", "local", "", 0, true);
        assert_eq!(seq, 0);
        assert_eq!(trail.entry_count(), 0);
    }

    // -- sequence numbering --------------------------------------------------

    #[test]
    fn test_sequence_numbering() {
        let mut trail = default_trail();
        let s1 = trail.log_event(AuditEventType::BlockPut, "a", "local", "", 0, true);
        let s2 = trail.log_event(AuditEventType::BlockPut, "b", "local", "", 0, true);
        let s3 = trail.log_event(AuditEventType::BlockDelete, "c", "local", "", 0, true);
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
        assert_eq!(s3, 3);
    }

    // -- query filters -------------------------------------------------------

    #[test]
    fn test_query_by_event_type() {
        let mut trail = default_trail();
        trail.log_event(AuditEventType::BlockPut, "a", "local", "", 0, true);
        trail.log_event(AuditEventType::BlockGet, "b", "local", "", 0, true);
        trail.log_event(AuditEventType::BlockPut, "c", "local", "", 0, true);

        let filter = AuditFilter {
            event_type: Some(AuditEventType::BlockPut),
            ..Default::default()
        };
        let results = trail.query(&filter);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_by_cid() {
        let mut trail = default_trail();
        trail.log_event(AuditEventType::BlockPut, "cid_x", "local", "", 0, true);
        trail.log_event(AuditEventType::BlockPut, "cid_y", "local", "", 0, true);

        let filter = AuditFilter {
            cid: Some("cid_x".to_string()),
            ..Default::default()
        };
        assert_eq!(trail.query(&filter).len(), 1);
    }

    #[test]
    fn test_query_by_actor() {
        let mut trail = default_trail();
        trail.log_event(AuditEventType::BlockPut, "a", "local", "", 0, true);
        trail.log_event(AuditEventType::BlockPut, "b", "peer1", "", 0, true);

        let filter = AuditFilter {
            actor: Some("peer1".to_string()),
            ..Default::default()
        };
        assert_eq!(trail.query(&filter).len(), 1);
    }

    #[test]
    fn test_query_by_since() {
        let mut trail = default_trail();
        trail.log_event(AuditEventType::BlockPut, "a", "local", "", 0, true);

        let filter = AuditFilter {
            since: Some(0),
            ..Default::default()
        };
        assert_eq!(trail.query(&filter).len(), 1);

        // Far-future timestamp should yield nothing.
        let filter2 = AuditFilter {
            since: Some(u64::MAX),
            ..Default::default()
        };
        assert_eq!(trail.query(&filter2).len(), 0);
    }

    #[test]
    fn test_query_by_until() {
        let mut trail = default_trail();
        trail.log_event(AuditEventType::BlockPut, "a", "local", "", 0, true);

        let filter = AuditFilter {
            until: Some(u64::MAX),
            ..Default::default()
        };
        assert_eq!(trail.query(&filter).len(), 1);

        let filter2 = AuditFilter {
            until: Some(0),
            ..Default::default()
        };
        assert_eq!(trail.query(&filter2).len(), 0);
    }

    #[test]
    fn test_query_success_only() {
        let mut trail = default_trail();
        trail.log_event(AuditEventType::BlockPut, "a", "local", "", 0, true);
        trail.log_event(AuditEventType::BlockPut, "b", "local", "", 0, false);

        let filter = AuditFilter {
            success_only: true,
            ..Default::default()
        };
        assert_eq!(trail.query(&filter).len(), 1);
    }

    #[test]
    fn test_query_combined_filters() {
        let mut trail = default_trail();
        trail.log_event(AuditEventType::BlockPut, "cid_a", "local", "", 10, true);
        trail.log_event(AuditEventType::BlockPut, "cid_a", "peer1", "", 20, true);
        trail.log_event(AuditEventType::BlockGet, "cid_a", "local", "", 10, true);
        trail.log_event(AuditEventType::BlockPut, "cid_b", "local", "", 10, false);

        let filter = AuditFilter {
            event_type: Some(AuditEventType::BlockPut),
            cid: Some("cid_a".to_string()),
            actor: Some("local".to_string()),
            success_only: true,
            ..Default::default()
        };
        let results = trail.query(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].cid, "cid_a");
        assert_eq!(results[0].actor, "local");
    }

    // -- integrity -----------------------------------------------------------

    #[test]
    fn test_integrity_valid() {
        let mut trail = default_trail();
        trail.log_event(AuditEventType::BlockPut, "a", "local", "ok", 100, true);
        trail.log_event(AuditEventType::BlockDelete, "b", "peer1", "del", 200, true);
        assert!(trail.verify_integrity().is_ok());
    }

    #[test]
    fn test_integrity_tampered() {
        let mut trail = default_trail();
        trail.log_event(AuditEventType::BlockPut, "a", "local", "ok", 100, true);
        trail.log_event(AuditEventType::BlockPut, "b", "local", "ok", 200, true);

        // Tamper with the first entry.
        trail.entries[0].bytes_affected = 999_999;

        let result = trail.verify_integrity();
        assert!(result.is_err());
        let bad = result.expect_err("should have bad entries");
        assert_eq!(bad.len(), 1);
        assert_eq!(bad[0], 1);
    }

    #[test]
    fn test_integrity_multiple_tampered() {
        let mut trail = default_trail();
        for i in 0..5 {
            trail.log_event(
                AuditEventType::BlockPut,
                &format!("c{i}"),
                "local",
                "",
                0,
                true,
            );
        }
        trail.entries[1].cid = "TAMPERED".to_string();
        trail.entries[3].actor = "EVIL".to_string();

        let bad = trail.verify_integrity().expect_err("should fail");
        assert_eq!(bad.len(), 2);
        assert!(bad.contains(&2));
        assert!(bad.contains(&4));
    }

    // -- checksum consistency ------------------------------------------------

    #[test]
    fn test_checksum_deterministic() {
        let entry = AuditEntry {
            sequence: 42,
            timestamp: 1_700_000_000,
            event_type: AuditEventType::BlockPut,
            cid: "bafy123".to_string(),
            actor: "local".to_string(),
            details: "test".to_string(),
            bytes_affected: 1024,
            success: true,
            checksum: 0,
        };
        let c1 = compute_checksum(&entry);
        let c2 = compute_checksum(&entry);
        assert_eq!(c1, c2);
        assert_ne!(c1, 0);
    }

    #[test]
    fn test_checksum_changes_with_fields() {
        let mut entry = AuditEntry {
            sequence: 1,
            timestamp: 100,
            event_type: AuditEventType::BlockPut,
            cid: "abc".to_string(),
            actor: "local".to_string(),
            details: "".to_string(),
            bytes_affected: 0,
            success: true,
            checksum: 0,
        };
        let c1 = compute_checksum(&entry);
        entry.cid = "xyz".to_string();
        let c2 = compute_checksum(&entry);
        assert_ne!(c1, c2);
    }

    // -- pruning -------------------------------------------------------------

    #[test]
    fn test_prune_before() {
        let mut trail = default_trail();
        // Insert entries and manually set timestamps.
        trail.log_event(AuditEventType::BlockPut, "a", "local", "", 0, true);
        trail.log_event(AuditEventType::BlockPut, "b", "local", "", 0, true);
        trail.log_event(AuditEventType::BlockPut, "c", "local", "", 0, true);
        trail.entries[0].timestamp = 100;
        trail.entries[1].timestamp = 200;
        trail.entries[2].timestamp = 300;
        // Recompute checksums after mutation.
        for e in &mut trail.entries {
            e.checksum = compute_checksum(e);
        }

        let removed = trail.prune_before(200);
        assert_eq!(removed, 1);
        assert_eq!(trail.entry_count(), 2);
        assert!(trail.verify_integrity().is_ok());
    }

    #[test]
    fn test_prune_before_all() {
        let mut trail = default_trail();
        trail.log_event(AuditEventType::BlockPut, "a", "local", "", 0, true);
        trail.log_event(AuditEventType::BlockPut, "b", "local", "", 0, true);

        let removed = trail.prune_before(u64::MAX);
        assert_eq!(removed, 2);
        assert_eq!(trail.entry_count(), 0);
    }

    #[test]
    fn test_prune_to_capacity() {
        let mut trail = AuditTrail::new(AuditTrailConfig {
            max_entries: 3,
            enable_checksums: true,
            log_reads: true,
            ..AuditTrailConfig::default()
        });

        for i in 0..5 {
            trail.log_event(
                AuditEventType::BlockPut,
                &format!("c{i}"),
                "local",
                "",
                0,
                true,
            );
        }
        assert_eq!(trail.entry_count(), 5);

        let pruned = trail.prune_to_capacity();
        assert_eq!(pruned, 2);
        assert_eq!(trail.entry_count(), 3);
        // Oldest entries (seq 1, 2) should be gone.
        assert!(trail.get_entry(1).is_none());
        assert!(trail.get_entry(2).is_none());
        assert!(trail.get_entry(3).is_some());
    }

    #[test]
    fn test_prune_to_capacity_no_op() {
        let mut trail = default_trail();
        trail.log_event(AuditEventType::BlockPut, "a", "local", "", 0, true);
        assert_eq!(trail.prune_to_capacity(), 0);
    }

    // -- stats ---------------------------------------------------------------

    #[test]
    fn test_stats_by_event_type() {
        let mut trail = default_trail();
        trail.log_event(AuditEventType::BlockPut, "a", "local", "", 0, true);
        trail.log_event(AuditEventType::BlockPut, "b", "local", "", 0, true);
        trail.log_event(AuditEventType::BlockDelete, "c", "local", "", 0, true);

        let s = trail.stats();
        assert_eq!(s.total_entries, 3);
        assert_eq!(*s.events_by_type.get("BlockPut").unwrap_or(&0), 2);
        assert_eq!(*s.events_by_type.get("BlockDelete").unwrap_or(&0), 1);
    }

    #[test]
    fn test_stats_pruned_count() {
        let mut trail = AuditTrail::new(AuditTrailConfig {
            max_entries: 2,
            enable_checksums: true,
            log_reads: true,
            ..AuditTrailConfig::default()
        });
        for i in 0..5 {
            trail.log_event(
                AuditEventType::BlockPut,
                &format!("c{i}"),
                "local",
                "",
                0,
                true,
            );
        }
        trail.prune_to_capacity();
        assert_eq!(trail.stats().entries_pruned, 3);
    }

    // -- events_for_cid ------------------------------------------------------

    #[test]
    fn test_events_for_cid() {
        let mut trail = default_trail();
        trail.log_event(AuditEventType::BlockPut, "target", "local", "", 100, true);
        trail.log_event(AuditEventType::BlockGet, "target", "peer1", "", 100, true);
        trail.log_event(AuditEventType::BlockPut, "other", "local", "", 50, true);

        let events = trail.events_for_cid("target");
        assert_eq!(events.len(), 2);
        for e in &events {
            assert_eq!(e.cid, "target");
        }
    }

    #[test]
    fn test_events_for_cid_empty() {
        let trail = default_trail();
        assert!(trail.events_for_cid("nonexistent").is_empty());
    }

    // -- recent_failures -----------------------------------------------------

    #[test]
    fn test_recent_failures() {
        let mut trail = default_trail();
        trail.log_event(AuditEventType::BlockPut, "a", "local", "", 0, false);
        trail.log_event(AuditEventType::BlockPut, "b", "local", "", 0, true);
        trail.log_event(AuditEventType::BlockPut, "c", "local", "", 0, false);
        trail.log_event(AuditEventType::BlockDelete, "d", "local", "", 0, false);

        let failures = trail.recent_failures(2);
        assert_eq!(failures.len(), 2);
        // Newest first.
        assert_eq!(failures[0].cid, "d");
        assert_eq!(failures[1].cid, "c");
    }

    #[test]
    fn test_recent_failures_none() {
        let mut trail = default_trail();
        trail.log_event(AuditEventType::BlockPut, "a", "local", "", 0, true);
        assert!(trail.recent_failures(10).is_empty());
    }

    // -- edge cases ----------------------------------------------------------

    #[test]
    fn test_empty_trail_query() {
        let trail = default_trail();
        let filter = AuditFilter::default();
        assert!(trail.query(&filter).is_empty());
    }

    #[test]
    fn test_empty_trail_last_entry() {
        let trail = default_trail();
        assert!(trail.last_entry().is_none());
    }

    #[test]
    fn test_empty_trail_verify_integrity() {
        let trail = default_trail();
        assert!(trail.verify_integrity().is_ok());
    }

    #[test]
    fn test_empty_trail_prune() {
        let mut trail = default_trail();
        assert_eq!(trail.prune_before(100), 0);
        assert_eq!(trail.prune_to_capacity(), 0);
    }

    #[test]
    fn test_get_entry_missing() {
        let trail = default_trail();
        assert!(trail.get_entry(999).is_none());
    }

    #[test]
    fn test_last_entry() {
        let mut trail = default_trail();
        trail.log_event(AuditEventType::BlockPut, "first", "local", "", 0, true);
        trail.log_event(AuditEventType::BlockDelete, "last", "local", "", 0, true);
        let last = trail.last_entry().expect("should have last");
        assert_eq!(last.cid, "last");
    }

    // -- large trail ---------------------------------------------------------

    #[test]
    fn test_large_audit_trail() {
        let mut trail = AuditTrail::new(AuditTrailConfig {
            max_entries: 5000,
            enable_checksums: true,
            log_reads: true,
            ..AuditTrailConfig::default()
        });

        for i in 0..1200u64 {
            let event = if i % 3 == 0 {
                AuditEventType::BlockPut
            } else if i % 3 == 1 {
                AuditEventType::BlockGet
            } else {
                AuditEventType::BlockDelete
            };
            trail.log_event(event, &format!("cid_{i}"), "local", "", i * 10, true);
        }

        assert_eq!(trail.entry_count(), 1200);
        assert!(trail.verify_integrity().is_ok());
        assert_eq!(trail.stats().total_entries, 1200);

        // Query subset.
        let filter = AuditFilter {
            event_type: Some(AuditEventType::BlockDelete),
            ..Default::default()
        };
        assert_eq!(trail.query(&filter).len(), 400);
    }

    // -- retention period enforcement ----------------------------------------

    #[test]
    fn test_retention_period_enforcement() {
        let mut trail = default_trail();
        trail.log_event(AuditEventType::BlockPut, "old", "local", "", 0, true);
        trail.log_event(AuditEventType::BlockPut, "new", "local", "", 0, true);

        // Manually age the first entry.
        trail.entries[0].timestamp = 1_000;
        trail.entries[0].checksum = compute_checksum(&trail.entries[0]);
        trail.entries[1].timestamp = 2_000;
        trail.entries[1].checksum = compute_checksum(&trail.entries[1]);

        let cutoff = trail.entries[1]
            .timestamp
            .saturating_sub(trail.config.retention_seconds);
        // With retention_seconds = 3600, cutoff = 0 so nothing gets pruned.
        // Use a tighter retention to actually prune.
        trail.config.retention_seconds = 500;
        let cutoff2 = trail.entries[1]
            .timestamp
            .saturating_sub(trail.config.retention_seconds);
        let pruned = trail.prune_before(cutoff2);
        assert_eq!(pruned, 1);
        assert_eq!(trail.entry_count(), 1);
        assert_eq!(trail.entries[0].cid, "new");

        // Sanity — cutoff with the original wide window keeps everything.
        let _ = cutoff; // used above for reasoning
    }

    // -- checksums disabled --------------------------------------------------

    #[test]
    fn test_checksums_disabled() {
        let mut trail = AuditTrail::new(AuditTrailConfig {
            enable_checksums: false,
            log_reads: true,
            ..AuditTrailConfig::default()
        });
        trail.log_event(AuditEventType::BlockPut, "a", "local", "", 0, true);
        let entry = trail.last_entry().expect("entry");
        assert_eq!(entry.checksum, 0);
    }

    // -- entry fields --------------------------------------------------------

    #[test]
    fn test_entry_fields_populated() {
        let mut trail = default_trail();
        trail.log_event(
            AuditEventType::Replication,
            "bafy_cid",
            "peer42",
            "replicated to zone-b",
            4096,
            true,
        );
        let e = trail.last_entry().expect("entry");
        assert_eq!(e.event_type, AuditEventType::Replication);
        assert_eq!(e.cid, "bafy_cid");
        assert_eq!(e.actor, "peer42");
        assert_eq!(e.details, "replicated to zone-b");
        assert_eq!(e.bytes_affected, 4096);
        assert!(e.success);
        assert_ne!(e.checksum, 0);
    }
}
